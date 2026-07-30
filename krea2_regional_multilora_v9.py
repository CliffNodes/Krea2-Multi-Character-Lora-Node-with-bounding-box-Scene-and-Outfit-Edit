"""
Krea2RegionalMultiLoRAV9 (By Fedor)

Architecture (stacked, not mutually exclusive):

  LIKENESS — always the original V2 engine
    * regional character LoRA (activation-delta, masked to each box)
    * photo latent-mold (post-CFG pull toward each region's ref photo)
    Source of truth for "does this look like the person?"

  EDIT — opt-in krea2edit layer (edit anything)
    * Identity Edit LoRA + grounded encode + official two-ref / sequential
    * Triggered by: use_krea2edit=True  OR  any IMAGE on extra_ref_1..4
    * Character LoRAs + molds stay armed on the final sample

Reference photos — BOTH paths kept:

  1) In-node: per-region "Load Ref" upload (regions_json.ref_image)
     → character identity for V2 mold (+ edit subject frames when editing)
     → optional auto-portrait is a visual identity reference / diagnostic;
       portrait-off generation uses bbox-routed region text + direct LoRA deltas

  2) Standalone LoadImage / Load Photo nodes → extra_ref_1..4
     → edit plates, typed by refs_json: scene (becomes the canvas) or
       person / object / style (each becomes its own krea2edit reference frame).
       The role also decides the referring text the node writes, so clauses name
       the right frame ordinal — something a hand-typed prompt cannot do, since
       the frame order depends on how many plates are wired.

No Pose Studio / SAM required.
"""

from __future__ import annotations

import gc
import json
import logging
import re

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

import comfy.model_management as mm
import comfy.sd
import comfy.utils

from .krea2_regional_multilora import (
    _pext,
    _WRAPPER_ENUM,
    _auto_split_norm,
    _coerce_bbox_norm,
    _load_lora_matrices,
    _normalize_bboxes,
    _resolve_lora_path,
    _RegionalSession,
)
from .krea2_regional_multilora_v3 import _load_ref_image_tensor, _parse_regions_v3
from .krea2_regional_edit_patch import _parse_box
from .krea2_reference_lock import (
    _blend_ref_molds,
    _build_mold,
    _denoised_as_4d,
    _encode_reference,
    _in_window,
    _sigma_window,
)
from .krea2_regional_multilora_v7 import (
    DEFAULT_PORTRAIT_PROMPT,
    Krea2RegionalMultiLoRAV7,
    _build_collage,
    _cover_resize,
    _edit_lora_choices,
    _grounded_cond,
    _paste_tile,
    _portrait_canvas_size,
    _stretch_resize,
)

WRAPPER_KEY_V9 = "krea2_regional_multilora_v9"

DEFAULT_REGIONS_JSON_V9 = (
    "[\n"
    '  {"lora": "None", "strength": 1.15, "enable": true, "ref_image": "", "prompt": "a man"},\n'
    '  {"lora": "None", "strength": 1.15, "enable": true, "ref_image": "", "prompt": "a woman"}\n'
    "]"
)


def _encode_text(clip, text: str):
    tokens = clip.tokenize(text or "")
    return clip.encode_from_tokens_scheduled(tokens)


_ORDINALS = ("first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth")


def _ordinal(i):
    """0-based frame index -> the word krea2edit was trained on."""
    return _ORDINALS[i] if 0 <= i < len(_ORDINALS) else f"{i + 1}th"


def _identity_anchor(user_prompt, who_tag, side, subj_frame, prior_subjects=False):
    """Identity binding that names the subject's ACTUAL reference frame.

    V7's _merge_identity_prompt hardcodes "the second reference" because back then
    the frame list was always exactly [scene, subject]. V9 sends
    [scene | aux plates... | subject], so every extra_ref plate pushes the subject
    further back — saying "second" then points the model at an outfit/prop plate
    instead of the face.

    The scene clause is also deliberately weaker than V7's "keep the scene
    unchanged": that phrasing suppresses any box prompt asking for a new pose or
    action, which is most of what this node is used for.

    prior_subjects is how many characters are ALREADY composed into the scene
    frame, so the clause can agree in number. It was a bool that always said "the
    people already in the first reference" / "both faces": with the usual two
    characters exactly one is in there, and asking for people plural where there
    is one is an invitation to draw an extra person — which is what the "a third
    woman appears sometimes" reports were. bool still works (True -> 1).
    """
    # Region portrait prompts are usually written "a man" / "the woman", which the
    # "the {who}" template below turns into "the a man". Drop the leading article.
    who = re.sub(r"^(?:a|an|the)\s+", "", (who_tag or "person").strip(), flags=re.I)
    who = who or "person"
    nth = _ordinal(subj_frame)
    n_prior = int(prior_subjects)
    if n_prior:
        those = "person" if n_prior == 1 else "people"
        faces = "both faces" if n_prior == 1 else "all faces"
        anchor = (f"the {those} already in the first reference and the {who} from the "
                  f"{nth} reference standing {side}, preserve the setting and lighting "
                  f"of the first reference, {faces} clearly visible, candid photo")
    else:
        anchor = (f"the {who} from the {nth} reference, standing {side}, "
                  f"preserve the setting and lighting of the first reference, "
                  f"candid photo, face clearly visible")
    u = (user_prompt or "").strip()
    return f"{u}, {anchor}" if u else anchor


# ---------------------------------------------------------------------------
# Reference roles
# ---------------------------------------------------------------------------
# refs_json describes the extra_ref_N sockets positionally (entry 0 = extra_ref_1).
# A role does two things at once: it decides whether the plate becomes the canvas
# or its own krea2edit reference frame, and it lets the node write the referring
# clause itself. That second half is the point — the frame index depends on how
# many plates happen to be wired, so a hand-typed "reference photo 1" can never
# name the right one.
REF_ROLES = ("auto", "scene", "person", "object", "style")

DEFAULT_REFS_JSON_V9 = "[]"


def _parse_refs_json(text, count):
    """-> [{"role", "note"}] * count. Missing/!JSON entries fall back to auto."""
    meta = []
    raw = (text or "").strip()
    if raw:
        try:
            data = json.loads(raw)
            if isinstance(data, dict):
                data = [data]
            for item in (data or []):
                if isinstance(item, str):
                    item = {"role": item}
                if not isinstance(item, dict):
                    item = {}
                role = str(item.get("role", "auto") or "auto").strip().lower()
                if role not in REF_ROLES:
                    logging.warning(
                        "[V9] reference role %r is not one of %s; using auto.",
                        role, ", ".join(REF_ROLES))
                    role = "auto"
                meta.append({"role": role,
                             "note": str(item.get("note", "") or "").strip()})
        except Exception as e:
            meta = []
            logging.warning("[V9] refs_json is not valid JSON (%s); all roles = auto.", e)
    while len(meta) < count:
        meta.append({"role": "auto", "note": ""})
    return meta[:count]


def _notice_image(lines, w=640, h=360):
    """Render a short status message as an IMAGE tensor [1,H,W,3].

    reference_preview is always wired to a PreviewImage node, so it has to emit
    something. A readable notice beats an empty tensor or a leftover frame: the
    point is to tell at a glance whether THIS job rendered any portraits, rather
    than squinting at a preview and wondering if it is stale.
    """
    img = Image.new("RGB", (w, h), (22, 22, 26))
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.load_default(size=22)
    except TypeError:  # Pillow < 10.1 takes no size argument
        font = ImageFont.load_default()

    boxes = [draw.textbbox((0, 0), ln, font=font) for ln in lines]
    gap = 14
    total = sum(b[3] - b[1] for b in boxes) + gap * max(0, len(lines) - 1)
    y = max(8, (h - total) // 2)
    for ln, b in zip(lines, boxes):
        draw.text(((w - (b[2] - b[0])) // 2, y), ln, font=font,
                  fill=(212, 212, 218))
        y += (b[3] - b[1]) + gap

    arr = np.asarray(img).astype(np.float32) / 255.0
    return torch.from_numpy(arr)[None]


def _release_transient_clones():
    """Nudge Python + ComfyUI to drop the model clones we produced.

    Each V9 generation clones the base model 2-3 times (edit LoRA, regional
    LoRAs, edit wrapper). Those clones share a ModelPatcher with the base and
    keep it pinned in ComfyUI's model_management cache. When the cache tries to
    evict later it finds the refcount still > 1 and emits the 'memory leak with
    model Krea2' stampede seen after long idles. Explicit GC + soft cache flush
    drops the transient clones without touching the base model still needed for
    the next generation.
    """
    gc.collect()
    try:
        mm.soft_empty_cache()
    except Exception:
        pass


def _apply_portrait_flags(regions, regions_json, default):
    """Restore V9-only row metadata lost by the shared V3 parser.

    A portrait is an extra 8-step render plus a model reload, and it forces the
    sequential compose path, so it roughly triples generation time. It is worth
    that only when you want to eyeball how faithful a LoRA is on its own — once a
    LoRA is known good, leave its toggle off and the final image is unaffected.

    _parse_regions_v3 carries only ref_image/ref_enable across. V9 also needs the
    row prompt for portrait rendering AND portrait-off spatial conditioning.
    Previously every ``r.get("prompt")`` below silently returned empty even when
    regions_json contained "a man" / "a woman". Keep the fix local to V9 rather
    than changing the schema inherited by V3/V8.
    """
    try:
        raw = json.loads(regions_json)
    except (ValueError, TypeError):
        raw = []
    if isinstance(raw, dict):
        raw = [raw]
    rows = [x for x in raw if isinstance(x, dict)] if isinstance(raw, list) else []
    for i, r in enumerate(regions):
        flag = rows[i].get("portrait") if i < len(rows) else None
        r["portrait"] = bool(default) if flag is None else bool(flag)
        r["prompt"] = (
            str(rows[i].get("prompt", "") or "").strip()
            if i < len(rows) else ""
        )
        r["_row_index"] = i
    return regions


def _builder_region_descriptions(prompt):
    """Region descriptions from an Ideogram4PromptBuilderKJ JSON caption.

    The bbox builder emits ``compositional_deconstruction.elements`` in the
    same order as its BOUNDING_BOX output. These descriptions are therefore the
    authoritative fallback when a V9 region row has no explicit prompt.
    """
    try:
        data = json.loads(prompt or "")
        elements = data.get("compositional_deconstruction", {}).get("elements", [])
    except (TypeError, ValueError, AttributeError):
        return []
    if not isinstance(elements, list):
        return []
    return [
        str(e.get("desc", "") or "").strip() if isinstance(e, dict) else ""
        for e in elements
    ]


def _spatial_region_prompts(prompt, regions):
    """One short, non-cross-named subject phrase per original region row."""
    builder_descs = _builder_region_descriptions(prompt)
    out = []
    for i, r in enumerate(regions):
        row = int(r.get("_row_index", i))
        text = str(r.get("prompt", "") or "").strip()
        if not text and row < len(builder_descs):
            text = builder_descs[row]
        if not text:
            text = str(r.get("name", "") or f"person {row + 1}").strip()
        out.append(text)
    return out


def _append_spatial_conditioning(clip, conditioning, region_prompts):
    """Append independently encoded region text and return exact token ranges.

    Krea 2 is single-stream: text and image tokens share every transformer
    block. Encoding each short region phrase separately and concatenating its
    hidden states gives us exact, tokenizer-independent token ranges. This is
    much safer than searching the grounded Qwen token stream, whose image-token
    expansion and stripped chat prefix make guessed offsets wrong.

    The Qwen text encoder runs once per region, but no diffusion or VAE pass is
    added. On a 5090 this is a small fixed cost compared with the two portrait
    renders and sequential compose pass this feature replaces.
    """
    if not conditioning or not region_prompts:
        return conditioning, []

    encoded = []
    for text in region_prompts:
        c = _encode_text(clip, text)
        if not c or not torch.is_tensor(c[0][0]):
            logging.warning("[V9/spatial] could not encode region prompt %r; routing disabled.", text)
            return conditioning, []
        encoded.append(c[0][0])

    out = []
    ranges = None
    for base, meta in conditioning:
        if not torch.is_tensor(base) or base.dim() != 3:
            return conditioning, []
        pieces = [base]
        local_ranges = []
        cursor = int(base.shape[1])
        for emb in encoded:
            if emb.shape[0] != base.shape[0]:
                if emb.shape[0] == 1:
                    emb = emb.expand(base.shape[0], -1, -1)
                else:
                    logging.warning("[V9/spatial] region conditioning batch mismatch; routing disabled.")
                    return conditioning, []
            if emb.shape[2] != base.shape[2]:
                logging.warning("[V9/spatial] region conditioning width mismatch; routing disabled.")
                return conditioning, []
            pieces.append(emb.to(device=base.device, dtype=base.dtype))
            end = cursor + int(emb.shape[1])
            local_ranges.append((cursor, end))
            cursor = end

        new_meta = dict(meta)
        # Krea2's DiT currently ignores this key, but keep metadata internally
        # consistent for future core versions that honor it.
        am = new_meta.get("attention_mask")
        if torch.is_tensor(am):
            extra = torch.ones(
                am.shape[0], cursor - int(base.shape[1]),
                device=am.device, dtype=am.dtype,
            )
            new_meta["attention_mask"] = torch.cat((am, extra), dim=-1)
        out.append([torch.cat(pieces, dim=1), new_meta])
        ranges = local_ranges

    return out, (ranges or [])


def _split_note(note, fallback):
    """"gold pistol, in his right hand" -> ("gold pistol", "in his right hand")."""
    n = re.sub(r"^(?:a|an|the)\s+", "", (note or "").strip(), flags=re.I)
    if not n:
        return fallback, ""
    head, _, tail = n.partition(",")
    return (head.strip() or fallback), tail.strip()


def _reference_clause(role, note, frame, subject=False):
    """Referring prose for one aux reference frame; "" when nothing to say.

    `frame` is the plate's real index in the src_latents list, and `subject` says
    whether a character is also in the shot (an object then gets held rather than
    just being present).
    """
    nth = _ordinal(frame)
    if role == "style":
        noun, tail = _split_note(note, "")
        extra = ", ".join(x for x in (noun, tail) if x)
        return f"in the style of the {nth} reference" + (f", {extra}" if extra else "")
    if role == "person":
        noun, tail = _split_note(note, "person")
        return (f"the {noun} from the {nth} reference"
                + (f", {tail}" if tail else ", face unchanged"))
    if role == "object":
        noun, tail = _split_note(note, "object")
        clause = f"the {noun} from the {nth} reference"
        if tail:
            return f"{clause}, {tail}"
        return f"holding {clause}" if subject else clause
    # scene: geometry only, the anchor already speaks for the first reference.
    # auto: no declared meaning, so only an explicit note is worth saying.
    if role == "auto":
        noun, tail = _split_note(note, "")
        extra = ", ".join(x for x in (noun, tail) if x)
        return f"{extra} from the {nth} reference" if extra else ""
    return ""


def _cap_long_side(px, max_side):
    """Downscale [1,H,W,3] so its long side <= max_side. 0 or None disables.

    krea2edit concatenates every reference frame into the diffusion attention
    sequence at whatever resolution it arrives in, and attention is quadratic in
    sequence length. An uncropped phone photo therefore costs more than the image
    being generated — a 1484x1624 reference is 9292 tokens against a 1536x1024
    target's 6144, and a 5712x4284 one is 95k.

    Aspect ratio is preserved (rounded to /16 for a clean token grid) because
    krea2edit loses identity when a reference frame's AR differs from the target.
    """
    if not max_side or max_side <= 0:
        return px
    h, w = int(px.shape[1]), int(px.shape[2])
    if max(h, w) <= max_side:
        return px
    s = float(max_side) / float(max(h, w))
    nw = max(16, int(round(w * s / 16)) * 16)
    nh = max(16, int(round(h * s / 16)) * 16)
    x = px.movedim(-1, 1).float()
    x = torch.nn.functional.interpolate(x, size=(nh, nw), mode="bilinear",
                                        align_corners=False)
    logging.info("[V9/edit] reference frame %dx%d -> %dx%d (%d -> %d tokens)",
                 w, h, nw, nh, (w // 16) * (h // 16), (nw // 16) * (nh // 16))
    return x.movedim(1, -1).clamp(0, 1)


def _clip_mask_to_box(mask, box, H, W):
    """Confine a mold mask to its box, so it can only feather INWARD.

    _build_mold writes the reference latent inside the box and leaves zeros
    outside it, but the mask is a sigmoid-feathered rect that extends past the
    box edges. _blend_ref_molds computes denoised + w*mask*(mold - denoised), so
    wherever the mask leaks outside the box it drags the latent toward ZERO —
    washed-out, blotchy patches that scale with ref_feather.
    """
    x0, y0, x1, y1 = box
    bx0, bx1 = int(round(x0 * W)), int(round(x1 * W))
    by0, by1 = int(round(y0 * H)), int(round(y1 * H))
    bx0 = max(0, min(bx0, W - 1))
    by0 = max(0, min(by0, H - 1))
    bx1 = min(max(bx1, bx0 + 1), W)
    by1 = min(max(by1, by0 + 1), H)
    support = torch.zeros_like(mask)
    support[:, :, by0:by1, bx0:bx1] = 1.0
    return mask * support


class _EditRegionalSession(_RegionalSession):
    """Regional session for the krea2edit forward.

    krea2_edit_forward builds the sequence as [text | refs... | target], so the
    canvas tokens are the TRAILING block. The base session anchors masks at
    txtlen, which in edit mode lands them on the reference frames instead of the
    canvas. Reporting no txtlen makes _full_mask use the trailing block.
    """

    def _extract_txtlen(self, args, kwargs):
        return None


_SPATIAL_NEG = 6.0
_SPATIAL_POS = 2.0
_SPATIAL_IMG_NEG = 3.0


def _make_spatial_attn_override(session, previous=None):
    """Route each region's text tokens to its bbox in Krea2 joint attention."""
    def override(func, q, k, v, heads, **kwargs):
        try:
            bias = session._spatial_bias(int(q.shape[-2]), q.device, q.dtype)
        except Exception as e:
            if not session._spatial_failed:
                logging.warning("[V9/spatial] attention routing failed (%s); running unmasked.", e)
                session._spatial_failed = True
            bias = None
        if bias is not None:
            prior_mask = kwargs.get("mask")
            kwargs["mask"] = bias if prior_mask is None else prior_mask + bias
        if previous is not None:
            return previous(func, q, k, v, heads, **kwargs)
        return func(q, k, v, heads, **kwargs)
    return override


class _EditSpatialRegionalSession(_EditRegionalSession):
    """V9 edit session whose boxes control semantics as well as LoRA deltas.

    The edit forward's sequence is ``[text | reference frames | target]``.
    Character LoRAs were already masked onto the trailing target block, but the
    text stream remained globally connected. Consequently a seed could draw the
    woman in the man's box and the mask would faithfully apply the MAN LoRA to
    that face. This session closes that missing half of regional generation:

      * target tokens inside box i are attracted to region-text i;
      * region-text i is strongly suppressed outside box i;
      * target tokens in different identity boxes receive a softer mutual block
        so identity-flavoured activations cannot hop boxes at the next layer;
      * global prompt and reference-frame tokens remain fully connected, keeping
        scene geometry, lighting, outfit transfer and interaction coherent.

    It uses ComfyUI's official ``optimized_attention_override`` hook. No core
    model monkey-patch and no extra diffusion pass are required.
    """

    def __init__(self, *args, spatial_token_ranges=None, spatial_prompts=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.spatial_token_ranges = list(spatial_token_ranges or [])
        self.spatial_prompts = list(spatial_prompts or [])
        self._spatial_txtlen = None
        self._spatial_bias_cache = {}
        self._spatial_logged = False
        self._spatial_failed = False

    @staticmethod
    def _get_topts(args, kwargs):
        if len(args) >= 5 and isinstance(args[4], dict):
            return args[4]
        out = kwargs.get("transformer_options")
        return out if isinstance(out, dict) else None

    @staticmethod
    def _context_len(args, kwargs):
        ctx = None
        if len(args) >= 3 and torch.is_tensor(args[2]) and args[2].dim() == 3:
            ctx = args[2]
        elif torch.is_tensor(kwargs.get("context")) and kwargs["context"].dim() == 3:
            ctx = kwargs["context"]
        return int(ctx.shape[1]) if ctx is not None else None

    def _spatial_bias(self, seq, device, dtype):
        # txtfusion attention has a different sequence length. Only touch the
        # 28 main single-stream blocks where refs+target follow the text.
        txt = self._spatial_txtlen
        n_img = self.n_img
        if not txt or n_img <= 0 or seq < txt + n_img:
            return None
        if len(self.spatial_token_ranges) != len(self._masks_d):
            return None
        if any(a < 0 or b <= a or b > txt for a, b in self.spatial_token_ranges):
            return None

        target_start = seq - n_img
        if target_start < txt:  # malformed [text|refs|target] layout
            return None
        key = (seq, txt, n_img, str(device), str(dtype))
        cached = self._spatial_bias_cache.get(key)
        if cached is not None:
            return cached

        bias = torch.zeros((1, 1, seq, seq), device=device, dtype=dtype)
        masks = [m.to(device=device, dtype=dtype).clamp(0, 1) for m in self._masks_d]

        # Exact text<->target routing. Outside its bbox a character phrase gets
        # -6; inside it gets up to +2. The inward masks are zero outside, so
        # there is no semantic tail leaking into the neighbouring identity.
        for (a, b), mask in zip(self.spatial_token_ranges, masks):
            route = (_SPATIAL_POS + _SPATIAL_NEG) * mask - _SPATIAL_NEG
            bias[:, :, target_start:, a:b] += route.view(1, 1, n_img, 1)
            bias[:, :, a:b, target_start:] += route.view(1, 1, 1, n_img)

        # LoRA deltas are spatially confined, but ordinary image self-attention
        # can otherwise carry identity features from one box into another on
        # the next layer. Suppress only box-to-box pairs; references and the
        # unboxed scene stay open, preserving global composition.
        for i in range(len(masks)):
            mi = masks[i]
            for j in range(i + 1, len(masks)):
                mj = masks[j]
                cross_ij = mi.view(n_img, 1) * mj.view(1, n_img)
                cross_ji = mj.view(n_img, 1) * mi.view(1, n_img)
                bias[:, :, target_start:, target_start:] -= (
                    _SPATIAL_IMG_NEG * (cross_ij + cross_ji)
                ).view(1, 1, n_img, n_img)

        self._spatial_bias_cache[key] = bias
        if not self._spatial_logged:
            refs = target_start - txt
            labels = ", ".join(
                f"{p!r}@{a}:{b}" for p, (a, b)
                in zip(self.spatial_prompts, self.spatial_token_ranges)
            )
            mb = bias.numel() * bias.element_size() / (1024 * 1024)
            logging.info(
                "[V9/spatial] bbox text routing ACTIVE: seq=%d text=%d refs=%d "
                "target=%d bias=%.1fMiB neg=%.1f pos=%.1f img_cross=%.1f | %s",
                seq, txt, refs, n_img, mb, _SPATIAL_NEG, _SPATIAL_POS,
                _SPATIAL_IMG_NEG, labels,
            )
            self._spatial_logged = True
        return bias

    def run(self, executor, *args, **kwargs):
        self._spatial_txtlen = self._context_len(args, kwargs)
        topts = self._get_topts(args, kwargs)
        if not self.spatial_token_ranges or topts is None:
            return super().run(executor, *args, **kwargs)
        previous = topts.get("optimized_attention_override")
        topts["optimized_attention_override"] = _make_spatial_attn_override(self, previous)
        try:
            return super().run(executor, *args, **kwargs)
        finally:
            if previous is None:
                topts.pop("optimized_attention_override", None)
            else:
                topts["optimized_attention_override"] = previous


def _stack_ref_preview(ref_images):
    if not ref_images:
        return torch.zeros(1, 64, 64, 3)
    out = []
    for img in ref_images:
        x = img[:1, :, :, :3].movedim(-1, 1).float()
        x = torch.nn.functional.interpolate(x, size=(512, 512), mode="bilinear")
        out.append(x.movedim(1, -1).clamp(0, 1))
    return torch.cat(out, dim=0)


class Krea2RegionalMultiLoRAV9(Krea2RegionalMultiLoRAV7):
    """V2 LoRA+mold likeness always; krea2edit for any edit; both ref upload paths."""

    @classmethod
    def INPUT_TYPES(cls):
        # Keep edit_lora REQUIRED and early (same as V8) so workflow widgets_values
        # map onto a real combo list — optional combos often load as empty/"not available".
        return {
            "required": {
                "model": ("MODEL", {
                    "tooltip": "Krea 2 diffusion model (wire from your LoRA loader / checkpoint loader).",
                }),
                "clip": ("CLIP", {
                    "tooltip": "Krea 2 text encoder. Used for the unified caption and grounded conditioning.",
                }),
                "vae": ("VAE", {
                    "tooltip": "Required for V2 photo molds and (when editing) krea2edit encodes.",
                }),
                "edit_lora": (_edit_lora_choices(), {
                    "tooltip": "Identity Edit LoRA — used when krea2edit is active.",
                }),
                "canvas_width": ("INT", {
                    "default": 1216, "min": 64, "max": 16384, "step": 16,
                    "tooltip": "Output width in pixels. Wire from your resolution/box-builder node so boxes and canvas always agree.",
                }),
                "canvas_height": ("INT", {
                    "default": 832, "min": 64, "max": 16384, "step": 16,
                    "tooltip": "Output height in pixels. Wire from your resolution/box-builder node so boxes and canvas always agree.",
                }),
                "regions_json": ("STRING", {
                    "multiline": True,
                    "default": DEFAULT_REGIONS_JSON_V9,
                    "tooltip": (
                        "One row per box. Set character LoRA + in-node Load Ref photo "
                        "for V2 likeness. Standalone LoadImage refs go on extra_ref_*."
                    ),
                }),
                "seam_feather": ("FLOAT", {
                    "default": 0.08, "min": 0.0, "max": 0.5, "step": 0.01,
                    "tooltip": "V2 regional LoRA edge softness.",
                }),
                "ref_strength": ("FLOAT", {
                    "default": 0.30, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "V2 latent-mold pull toward each in-node ref photo (0.2–0.4).",
                }),
                "ref_start_percent": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Sampling progress (0..1) at which the photo mold starts pulling. 0 = from the first step.",
                }),
                "ref_end_percent": ("FLOAT", {
                    "default": 0.60, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "End mold window early (~0.5–0.7) so lighting can integrate.",
                }),
                "ref_feather": ("FLOAT", {
                    "default": 0.06, "min": 0.0, "max": 0.5, "step": 0.01,
                    "tooltip": "Edge softness of the photo-mold region so the pulled area blends into its surroundings.",
                }),
            },
            "optional": {
                # forceInput: the new frontend otherwise attaches an x/y/width/
                # height widget group to BOUNDING_BOX inputs; those sub-widgets
                # serialize into widgets_values and phase-shift every widget
                # after them in saved graphs.
                "bboxes": ("BOUNDING_BOX", {
                    "forceInput": True,
                    "tooltip": (
                        "Bounding boxes from a box-builder node (e.g. "
                        "Ideogram4PromptBuilderKJ). Box i drives region i: LoRA "
                        "masking, text routing, and subject placement."
                    ),
                }),
                "extra_ref_1": ("IMAGE", {
                    "tooltip": (
                        "Standalone LoadImage / Load Photo — any edit plate "
                        "(scene, object, background, etc.). Enables krea2edit."
                    ),
                }),
                "extra_box_1": ("STRING", {
                    "default": "0,0,1,1",
                    "tooltip": "x,y,w,h in 0..1 for extra_ref_1. Full canvas = 0,0,1,1.",
                }),
                "extra_ref_2": ("IMAGE", {
                    "tooltip": (
                        "Second standalone plate. With a full-canvas box it becomes an "
                        "extra krea2edit REFERENCE FRAME (outfit, object, style) and "
                        "leaves extra_ref_1's scene intact. Give it a sub-box instead "
                        "to hard-paste it into the scene."
                    ),
                }),
                "extra_box_2": ("STRING", {
                    "default": "0,0,1,1",
                    "tooltip": (
                        "x,y,w,h in 0..1 for extra_ref_2. Full canvas (0,0,1,1) = use "
                        "as a reference frame; a sub-box pastes it into the scene."
                    ),
                }),
                "edit_lora_strength": ("FLOAT", {
                    "default": 0.7, "min": 0.0, "max": 2.0, "step": 0.05,
                    "tooltip": (
                        "Identity Edit LoRA weight. Lower than V7/V8 on purpose: there "
                        "the character LoRA was inactive during the compose pass, while "
                        "V9 runs both in the SAME forward, so the two deltas add. Too "
                        "high together gives mottled, crumpled-looking texture. If you "
                        "raise region strengths, lower this."
                    ),
                }),
                "compose_steps": ("INT", {
                    "default": 10, "min": 4, "max": 40, "step": 1,
                    "tooltip": "Internal krea2edit compose steps (edit path only).",
                }),
                "compose_seed": ("INT", {
                    "default": 0, "min": 0, "max": 0xffffffffffffffff,
                    "tooltip": "Seed for the internal krea2edit compose pass. 0 reuses the incoming noise seed.",
                }),
                "prompt": ("STRING", {
                    "multiline": True, "default": "",
                    "tooltip": (
                        "Scene prompt. Wire your box-builder's prompt output here; "
                        "V12 recompiles it into one unified caption with exact "
                        "per-region token spans."
                    ),
                }),
                "negative_prompt": ("STRING", {
                    "multiline": True, "default": "",
                    "tooltip": "Negative prompt for the whole canvas.",
                }),
                "grounding_px": ("INT", {
                    "default": 1024, "min": 0, "max": 4096, "step": 64,
                    "tooltip": "Qwen3-VL grounding cap for krea2edit. Ignored when edit is off.",
                }),
                "base_strength": ("FLOAT", {
                    "default": 1.0, "min": -10.0, "max": 10.0, "step": 0.05,
                    "tooltip": (
                        "Global multiplier applied on top of every region's own "
                        "LoRA strength. 1.0 = use region strengths as-is; nudge "
                        "to 1.1–1.2 if all identities look slightly soft."
                    ),
                }),
                "blend_override": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": (
                        "LEAVE AT 0. Any value above 0 blends every LoRA across "
                        "the WHOLE canvas, which causes identity bleeding between "
                        "regions. Only useful for single-LoRA style effects."
                    ),
                }),
                "use_krea2edit": ("BOOLEAN", {
                    "default": False,
                    "tooltip": (
                        "Force krea2edit even with no extra_ref LoadImage "
                        "(spatial collage of in-node region photos). "
                        "Likeness still from V2 LoRA+mold."
                    ),
                }),
                # --- appended below on purpose ---------------------------------
                # Widget order IS the workflow save format (widgets_values is a
                # positional array). Anything inserted above this point silently
                # re-reads every later value in existing saves, which is exactly
                # what put 1536 into edit_lora. New widgets go LAST, always.
                "extra_ref_3": ("IMAGE", {
                    "tooltip": "Third plate. Role comes from refs_json entry 3.",
                }),
                "extra_ref_4": ("IMAGE", {
                    "tooltip": "Fourth plate. Role comes from refs_json entry 4.",
                }),
                "refs_json": ("STRING", {
                    "multiline": True,
                    "default": DEFAULT_REFS_JSON_V9,
                    "tooltip": (
                        "Role per extra_ref, in order (entry 1 = extra_ref_1):\n"
                        '[{"role":"scene"},\n'
                        ' {"role":"object","note":"gold pistol, in his right hand"},\n'
                        ' {"role":"person","note":"woman in the red coat"}]\n'
                        "roles: auto | scene | person | object | style.\n"
                        "scene becomes the canvas; person/object/style each become "
                        "their own krea2edit reference frame, and the node writes "
                        "the referring text (\"the gold pistol from the third "
                        "reference\") with the frame number you cannot know in "
                        "advance. note = 'noun, where it goes'. Empty = old "
                        "behaviour (first full-canvas plate is the scene)."
                    ),
                }),
                "auto_portrait": ("BOOLEAN", {
                    "default": False,
                    "tooltip": (
                        "DEFAULT for regions whose own Portrait toggle is unset. "
                        "Per-region toggles always win, so use those for normal "
                        "work and leave this off.\n\n"
                        "A region with a character LoRA but no photo renders that "
                        "LoRA alone into a portrait, which then goes in as its "
                        "subject reference frame — how a bare LoRA gets injected "
                        "into a scene.\n\n"
                        "Each portrait costs an extra render plus a model reload, "
                        "and it changes which engine path runs. One bare LoRA: "
                        "plate_only (1 sampling pass) becomes plate_tworef (2). "
                        "TWO bare LoRAs: plate_sequential — 4 passes, and only the "
                        "LAST character gets a live reference frame with its LoRA "
                        "armed, so the other comes out weaker."
                    ),
                }),
                "portrait_steps": ("INT", {
                    "default": 8, "min": 4, "max": 40, "step": 1,
                    "tooltip": "Steps for each auto portrait.",
                }),
                "portrait_seed": ("INT", {
                    "default": 0, "min": 0, "max": 0xffffffffffffffff,
                    "tooltip": "Seed for auto-portrait renders (portrait-enabled regions only).",
                }),
                "extra_box_3": ("STRING", {
                    "default": "0,0,1,1",
                    "tooltip": "x,y,w,h for extra_ref_3 (auto role only).",
                }),
                "extra_box_4": ("STRING", {
                    "default": "0,0,1,1",
                    "tooltip": "x,y,w,h for extra_ref_4 (auto role only).",
                }),
                "ref_max_side": ("INT", {
                    # RESTORE 2026-07-24: no cap existed then, so references were
                    # encoded at native resolution and got the full token budget
                    # for identity. Defaulting to 1024 costs likeness for speed.
                    "default": 0, "min": 0, "max": 4096, "step": 64,
                    "tooltip": (
                        "Downscale every reference frame (subjects + extra_ref "
                        "plates) to this long side before encoding. 0 = native.\n\n"
                        "This is the main speed knob. Reference frames are "
                        "concatenated into the attention sequence, which is "
                        "quadratic — a 1484x1624 photo is 9292 tokens against a "
                        "1536x1024 target's 6144, so one uncropped reference can "
                        "cost more than the image itself. Aspect ratio is kept. "
                        "Raise it if identity looks soft; lower it for speed."
                    ),
                }),
                "portrait_preview": ("BOOLEAN", {
                    "default": True,
                    "tooltip": (
                        "Send rendered portraits to the reference_preview output so "
                        "you can compare a LoRA's solo likeness against the final "
                        "image.\n\n"
                        "When no region has its portrait toggle on, the output is a "
                        "notice saying so — that is how you tell whether this "
                        "particular job produced any previews at all. Turning this "
                        "off saves almost no time: the portrait renders are the "
                        "cost, and those are controlled per region."
                    ),
                }),
            },
        }

    RETURN_TYPES = ("MODEL", "CONDITIONING", "CONDITIONING", "IMAGE", "KREA2_DATA")
    RETURN_NAMES = ("model", "positive", "negative", "reference_preview", "data")
    FUNCTION = "apply"
    CATEGORY = "Krea2/By Fedor"
    DESCRIPTION = (
        "Krea2 Regional Multi-LoRA v9 (By Fedor). Likeness ALWAYS from the V2 engine "
        "(regional character LoRA + in-node photo mold). Enable krea2edit via "
        "use_krea2edit or by wiring standalone LoadImage plates to extra_ref_* "
        "(edit anything). Keep both ref paths: in-node Load Ref per row, and "
        "LoadImage → extra_ref."
    )

    def apply(
        self,
        model,
        clip,
        vae,
        edit_lora,
        canvas_width,
        canvas_height,
        regions_json,
        seam_feather,
        ref_strength,
        ref_start_percent,
        ref_end_percent,
        ref_feather,
        bboxes=None,
        extra_ref_1=None,
        extra_box_1="0,0,1,1",
        extra_ref_2=None,
        extra_box_2="0,0,1,1",
        edit_lora_strength=1.0,
        compose_steps=10,
        compose_seed=0,
        prompt="",
        negative_prompt="",
        grounding_px=1024,
        base_strength=1.0,
        blend_override=0.0,
        use_krea2edit=False,
        extra_ref_3=None,
        extra_ref_4=None,
        refs_json=DEFAULT_REFS_JSON_V9,
        auto_portrait=False,
        portrait_steps=8,
        portrait_seed=0,
        extra_box_3="0,0,1,1",
        extra_box_4="0,0,1,1",
        ref_max_side=1024,
        portrait_preview=True,
        force_edit_mode=False,  # legacy alias from early V9 builds
    ):
        # Stale graphs (pre-reorder V9 saves) can hand us a number here.
        if not isinstance(edit_lora, str) or not edit_lora.endswith(".safetensors"):
            choices = _edit_lora_choices()
            logging.warning(
                "[V9] edit_lora=%r is not a LoRA filename (stale workflow?); using %r.",
                edit_lora, choices[0] if choices else None,
            )
            edit_lora = choices[0] if choices else None

        seam_feather = min(max(float(seam_feather), 0.0), 0.5)
        ref_feather = min(max(float(ref_feather), 0.0), 0.5)

        sockets = (
            (extra_ref_1, extra_box_1),
            (extra_ref_2, extra_box_2),
            (extra_ref_3, extra_box_3),
            (extra_ref_4, extra_box_4),
        )
        # Roles are declared per SOCKET, so read all four and drop the unwired
        # ones afterwards. Matching roles against the compacted list instead
        # would slide extra_ref_3's role onto extra_ref_2 the moment someone
        # mutes a plate mid-chain.
        refs_meta = _parse_refs_json(refs_json, len(sockets))
        extras = []
        for (img, boxstr), meta in zip(sockets, refs_meta):
            if img is not None:
                extras.append((img, boxstr or "0,0,1,1", meta["role"], meta["note"]))

        # Edit layer: any standalone LoadImage plate OR explicit toggle.
        use_edit = bool(extras) or bool(use_krea2edit) or bool(force_edit_mode)
        if use_edit:
            return self._apply_edit_mode(
                model, clip, vae, canvas_width, canvas_height, regions_json,
                bboxes, extras, edit_lora, edit_lora_strength, compose_steps,
                compose_seed, prompt, negative_prompt, grounding_px,
                base_strength, ref_strength, ref_start_percent, ref_end_percent,
                ref_feather, seam_feather, blend_override,
                bool(auto_portrait), int(portrait_steps), int(portrait_seed),
                int(ref_max_side), bool(portrait_preview),
            )
        # Pure V2 likeness (in-node LoRA + ref photos, one KSampler).
        return self._apply_v2_likeness(
            model, clip, vae, canvas_width, canvas_height, regions_json,
            bboxes, seam_feather, blend_override, ref_strength,
            ref_start_percent, ref_end_percent, ref_feather,
            base_strength, prompt, negative_prompt,
        )

    # ------------------------------------------------------------------
    # Likeness-only — original V2 engine (regional LoRA + photo mold)
    # ------------------------------------------------------------------
    def _apply_v2_likeness(
        self, model, clip, vae, canvas_width, canvas_height, regions_json,
        bboxes, seam_feather, blend_override, ref_strength,
        ref_start_percent, ref_end_percent, ref_feather,
        base_strength, prompt, negative_prompt,
    ):
        regions = _parse_regions_v3(regions_json)

        def has_lora(r):
            return r["lora"] not in ("None", "") and (r["strength"] * base_strength) != 0.0

        def has_ref(r):
            return bool(r.get("ref_image")) and r.get("ref_enable", True)

        cw, ch = int(canvas_width), int(canvas_height)
        frame = _normalize_bboxes(bboxes)
        pairs = self._pair_boxes(
            regions, frame, cw, ch,
            lambda r: r["enable"] and (has_lora(r) or has_ref(r)))
        active = [r for r, _ in pairs]
        norm_boxes = [b for _, b in pairs]

        empty = torch.zeros(1, 64, 64, 3)
        if not active:
            logging.warning("[V9/V2] no active regions; passthrough.")
            return (
                model,
                _encode_text(clip, prompt),
                _encode_text(clip, negative_prompt),
                empty,
                {"adapters": [], "engine": "v9_v2_empty"},
            )

        if not frame:
            logging.warning("[V9/V2] no bboxes; auto_vertical split.")
            norm_boxes = _auto_split_norm(len(active), "auto_vertical")

        patched, strength_eff, n_lora = self._arm_regional_loras(
            model, active, norm_boxes, cw, ch, seam_feather, blend_override, base_strength)

        preview_imgs = []
        n_mold = self._arm_photo_molds(
            patched, vae, active, norm_boxes, ref_strength,
            ref_start_percent, ref_end_percent, ref_feather, preview_imgs)

        positive = _encode_text(clip, prompt)
        negative = _encode_text(clip, negative_prompt)
        preview = _stack_ref_preview(preview_imgs)
        data = {
            "adapters": [
                {
                    "name": r["name"],
                    "lora": r["lora"],
                    "strength": s,
                    "ref_image": r.get("ref_image", ""),
                    "ref_enable": r.get("ref_enable", True),
                }
                for r, s in zip(active, strength_eff)
            ],
            "model_type": "krea2",
            "engine": "v9_v2_lora+mold",
            "mode": "v2_likeness",
            "loras": n_lora,
            "molds": n_mold,
            "ref_paths": {
                "in_node_region_refs": n_mold,
                "standalone_extra_refs": 0,
            },
        }
        logging.info(
            "[V9] V2 likeness: %d regions (%d LoRA, %d in-node mold), "
            "ref_strength=%.2f window %.2f-%.2f",
            len(active), n_lora, n_mold, float(ref_strength),
            float(ref_start_percent), float(ref_end_percent),
        )
        _release_transient_clones()
        return (patched, positive, negative, preview, data)

    def _render_portrait(self, model, clip, vae, region, idx, cw, ch,
                         base_strength, steps, seed, who):
        """Render one region's LoRA ALONE into a subject frame. None on failure.

        Alone is the whole trick: rendering characters together is where the
        cross-LoRA bleed came from. Uses the plain model, not edit_model, so the
        Identity Edit LoRA does not colour the portrait. The aspect ratio has to
        match the canvas — krea2edit drops identity badly when a reference frame's
        AR differs from the target's.
        """
        pw, ph = _portrait_canvas_size(cw, ch, max_side=1024)
        p = who or DEFAULT_PORTRAIT_PROMPT
        if "background" not in p.lower():
            p = (f"{p}, plain neutral gray background, upper body, looking at camera, "
                 "sharp focus, soft natural lighting")
        s = float(region["strength"]) * float(base_strength)
        logging.info("[V9/edit] region %d: auto portrait from %s @%dx%d (%d steps, str %.2f)",
                     idx, region["lora"], pw, ph, int(steps), s)
        try:
            lat = self._gen_portrait_latent(model, clip, region["lora"], s, p, pw, ph,
                                            int(steps), int(seed) + idx * 1000)
            return self._decode_px(vae, lat)
        except Exception as e:
            logging.warning("[V9/edit] auto portrait failed for region %d (%s): %s",
                            idx, region.get("lora"), e)
            return None

    # ------------------------------------------------------------------
    # Edit layer — krea2edit (anything) + V2 LoRA+mold still armed
    # ------------------------------------------------------------------
    def _portrait_preview(self, frames, enabled):
        """The portrait sheet for this job, or a notice explaining its absence.

        Previously this output carried the scene plate or the collage, which made
        it impossible to tell whether a portrait had actually been rendered — the
        node is titled "Reference portraits", so a scene plate sitting in it reads
        as a stale preview. Now it shows portraits or says there were none.
        """
        if not enabled:
            return _notice_image([
                "Portrait previews are OFF",
                "portrait_preview toggle on the V9 node",
            ])
        if not frames:
            return _notice_image([
                "No portrait previews for this job",
                "No region has its portrait toggle on.",
                "Turn one on to inspect that LoRA's likeness.",
            ])
        return self._stack_previews([f.movedim(-1, 1).float() for f in frames])

    def _apply_edit_mode(
        self, model, clip, vae, canvas_width, canvas_height, regions_json,
        bboxes, extras, edit_lora, edit_lora_strength, compose_steps,
        compose_seed, prompt, negative_prompt, grounding_px,
        base_strength, ref_strength, ref_start_percent, ref_end_percent,
        ref_feather, seam_feather, blend_override,
        auto_portrait=False, portrait_steps=8, portrait_seed=0,
        ref_max_side=1024, portrait_preview=True,
    ):
        if not edit_lora or edit_lora in ("None", ""):
            choices = _edit_lora_choices()
            edit_lora = choices[0] if choices else None
        if not edit_lora or edit_lora in ("None", ""):
            raise ValueError(
                "[V9] krea2edit needs an Identity Edit LoRA (edit_lora). "
                "Load krea2_identity_edit_*.safetensors."
            )

        cw, ch = int(canvas_width), int(canvas_height)
        gpx = int(grounding_px)
        builder_prompt = prompt  # preserve the original JSON before reference clauses are appended
        regions = _parse_regions_v3(regions_json)

        def has_ref(r):
            return bool(r.get("ref_image")) and r.get("ref_enable", True)

        def has_lora(r):
            return r["lora"] not in ("None", "") and (r["strength"] * base_strength) != 0.0

        # Edit subjects: an in-node Load Ref photo, or — where the region opted in
        # via its portrait toggle — the region's LoRA rendered alone. Without a
        # frame a bare LoRA has nothing in src_latents for krea2edit to insert.
        # Portrait-off regions are still spatially anchored below by their own
        # text tokens plus the LoRA delta on the same bbox.
        _apply_portrait_flags(regions, regions_json, auto_portrait)

        def wants_portrait(r):
            return bool(r.get("portrait"))

        frame = _normalize_bboxes(bboxes)
        subj_pairs = self._pair_boxes(
            regions, frame, cw, ch,
            lambda r: r["enable"] and (has_ref(r) or (wants_portrait(r) and has_lora(r))))
        active = [r for r, _ in subj_pairs]
        norm_boxes = [b for _, b in subj_pairs]
        for r in regions:
            if (r["enable"] and not has_ref(r) and has_lora(r)
                    and not wants_portrait(r)):
                logging.info(
                    "[V9/edit] region '%s': portrait off — direct LoRA + bbox text "
                    "routing will be used (no portrait diffusion pass).",
                    r.get("name") or r.get("lora") or "?",
                )

        empty = torch.zeros(1, 64, 64, 3)
        if not active and not extras:
            logging.warning(
                "[V9/edit] nothing to compose (no in-node refs, no LoadImage extras); "
                "falling back to V2 likeness."
            )
            return self._apply_v2_likeness(
                model, clip, vae, canvas_width, canvas_height, regions_json,
                bboxes, seam_feather, blend_override, ref_strength,
                ref_start_percent, ref_end_percent, ref_feather,
                base_strength, prompt, negative_prompt,
            )

        edit_path = _resolve_lora_path(edit_lora)
        edit_sd = comfy.utils.load_torch_file(edit_path, safe_load=True)
        edit_model = comfy.sd.load_lora_for_models(
            model, None, edit_sd, float(edit_lora_strength), 0)[0]

        placed = []
        portrait_frames = []  # subset of `placed` that we rendered, for the preview
        for i, (r, nb) in enumerate(zip(active, norm_boxes)):
            who = str(r.get("prompt", "") or "").strip()
            who_tag = who.split(",")[0].strip() if who else f"person {i + 1}"
            subj_px = None
            if has_ref(r):
                try:
                    img = _load_ref_image_tensor(r["ref_image"])
                    subj_px = img[:1, :, :, :3]
                except Exception as e:
                    logging.warning(
                        "[V9/edit] in-node ref '%s' failed: %s; trying LoRA portrait.",
                        r.get("ref_image"), e)
            rendered = False
            if subj_px is None and wants_portrait(r) and has_lora(r):
                subj_px = self._render_portrait(
                    model, clip, vae, r, i, cw, ch, base_strength,
                    portrait_steps, portrait_seed, who)
                rendered = subj_px is not None
            if subj_px is None:
                logging.warning("[V9/edit] region %d has no usable subject image; skipped.", i)
                continue
            # Cap here, once: everything downstream (_compose_once, _enc_ms,
            # the grounded encode, the preview sheet) reads from `placed`.
            capped = _cap_long_side(subj_px, ref_max_side)
            placed.append((capped, nb, who_tag))
            if rendered:
                portrait_frames.append(capped)

        def _enc_ms(px):
            return edit_model.model.process_latent_in(vae.encode(px[:, :, :, :3]))

        steps = max(4, int(compose_steps))
        if extras:
            # Standalone LoadImage plate(s), sorted by role.
            #
            # Only a scene plate becomes the canvas. krea2edit takes a LIST of
            # reference frames, so a person / object / style plate goes in as its
            # own frame; compositing it into the canvas would just overwrite the
            # scene, since full-canvas plates all carry the same box.
            scene_canvas = None
            aux = []          # (pixels, role, note), in reference-frame order
            for img, boxstr, role, note in extras:
                px = img[:1, :, :, :3]
                bx, by, bw, bh = _parse_box(boxstr)
                is_full = (bw >= 0.999 and bh >= 0.999 and bx <= 1e-3 and by <= 1e-3)
                as_scene = (role == "scene") or (role == "auto" and is_full)
                if as_scene and scene_canvas is None:
                    scene_canvas = _cover_resize(px, cw, ch)
                elif role == "auto" and not is_full:
                    if scene_canvas is None:
                        scene_canvas = torch.full(
                            (1, ch, cw, 3), 0.5, dtype=torch.float32)
                    _paste_tile(
                        scene_canvas, px,
                        round(bx * cw), round(by * ch),
                        round(bw * cw), round(bh * ch),
                    )
                else:
                    if as_scene:
                        logging.info(
                            "[V9/edit] scene is already taken; this plate becomes a "
                            "reference frame instead of replacing it.")
                    aux.append((_cap_long_side(px, ref_max_side)
                                .float().clamp(0, 1), role, note))
            if scene_canvas is None:
                scene_canvas = torch.full((1, ch, cw, 3), 0.5, dtype=torch.float32)
            current = scene_canvas.clamp(0, 1)
            aux_plates = [p for p, _r, _n in aux]

            for i, (subj_px, nb, who_tag) in enumerate(placed[:-1]):
                side = self._side_label(nb, i)
                # _compose_once sends exactly [scene, subject], so subject = frame 1.
                pass_prompt = _identity_anchor(prompt, who_tag, side, 1)
                current = self._compose_once(
                    edit_model, clip, vae, current, subj_px, pass_prompt, negative_prompt,
                    gpx, cw, ch, steps, int(compose_seed) + 10000 + i)

            # Scene stays frame 1, subject stays last; aux plates sit between.
            aux_latents = [_enc_ms(p) for p in aux_plates]

            # src_latents is [scene(0), *aux(1..n), subject(last)], so an aux
            # plate's ordinal is simply its index in `aux`. Writing these clauses
            # here is the only way they can be right: the number depends on how
            # many plates are wired, which the person typing the prompt has no
            # way to know.
            clauses = []
            for j, (_px, role, note) in enumerate(aux):
                c = _reference_clause(role, note, j + 1, subject=bool(placed))
                if c:
                    clauses.append(c)
            role_tail = ", ".join(clauses)
            if role_tail:
                logging.info("[V9/edit] reference roles -> %s", role_tail)

            if placed:
                subj_px, nb, who_tag = placed[-1]
                side = self._side_label(nb, len(placed) - 1)
                # The subject sits at 1 + len(aux_latents), not always at index 1.
                final_prompt = _identity_anchor(
                    prompt, who_tag, side, 1 + len(aux_latents),
                    prior_subjects=len(placed) - 1)
                if role_tail:
                    final_prompt = f"{final_prompt}, {role_tail}"
                scene = _stretch_resize(current, cw, ch)
                subj = subj_px[:1, :, :, :3].float().clamp(0, 1)
                src_latents = [_enc_ms(scene)] + aux_latents + [_enc_ms(subj)]
                ground_imgs = [scene] + aux_plates + [subj]
                preview = self._portrait_preview(portrait_frames, portrait_preview)
                engine = ("v9_edit_plate_tworef" if len(placed) == 1
                          else "v9_edit_plate_sequential")
            else:
                scene = _stretch_resize(current, cw, ch)
                src_latents = [_enc_ms(scene)] + aux_latents
                ground_imgs = [scene] + aux_plates
                preview = self._portrait_preview(portrait_frames, portrait_preview)
                final_prompt = ", ".join(
                    x for x in ((prompt or "").strip(), role_tail) if x)
                engine = "v9_edit_plate_only"
            if aux_latents:
                engine += f"+{len(aux_latents)}ref"
            prompt = final_prompt
        else:
            # use_krea2edit with in-node photos only — spatial collage edit.
            collage = _build_collage([(p, nb) for p, nb, _ in placed], extras, cw, ch)
            src_latents = [_enc_ms(collage)]
            ground_imgs = [collage]
            preview = self._portrait_preview(portrait_frames, portrait_preview)
            engine = "v9_edit_spatial_collage"

        # ORDER IS LOAD-BEARING. Wrappers run outermost-first in registration
        # order, and krea2_edit_forward does NOT call through to the next
        # executor — it invokes the forward itself. So the regional LoRA session
        # must be registered FIRST (outermost); registered after the edit
        # wrapper it is never reached and the character LoRAs do nothing.
        lora_pairs = self._pair_boxes(
            regions, frame, cw, ch,
            lambda r: r["enable"] and (has_lora(r) or has_ref(r)))
        lora_active = [r for r, _ in lora_pairs]
        lora_boxes = [b for _, b in lora_pairs]

        # No bboxes wired = _pair_boxes hands every region the FULL canvas, so
        # with 2+ LoRAs every face receives every identity (guaranteed bleed).
        # The V2 path already falls back to equal strips; mirror it here.
        if not frame and sum(1 for r in lora_active if has_lora(r)) >= 2:
            logging.warning(
                "[V9/edit] no bboxes wired but %d LoRA regions — falling back "
                "to equal vertical strips so identities don't overlap the full "
                "canvas. Wire the builder's bboxes output for real placement.",
                len(lora_active))
            lora_boxes = _auto_split_norm(len(lora_active), "auto_vertical")

        # Spatial conditioning is the missing half of the regional LoRA engine.
        # The global grounded prompt still controls the whole scene; compact
        # per-character embeddings are appended and attention-routed to the same
        # boxes as their LoRA deltas. This is a single final diffusion pass —
        # portraits and sequential composition remain optional.
        positive = _grounded_cond(clip, prompt, ground_imgs, gpx)
        spatial_prompts = _spatial_region_prompts(builder_prompt, lora_active)
        positive, spatial_ranges = _append_spatial_conditioning(
            clip, positive, spatial_prompts)
        spatial_on = bool(spatial_ranges) and len(spatial_ranges) == len(lora_boxes)
        if not spatial_on and len(lora_boxes) >= 2:
            logging.warning(
                "[V9/spatial] regional text routing could not be armed; "
                "LoRA deltas remain boxed but placement may be seed-dependent."
            )

        patched, _, n_lora = self._arm_regional_loras(
            edit_model, lora_active, lora_boxes, cw, ch,
            seam_feather, blend_override, base_strength,
            wrapper_key=WRAPPER_KEY_V9,
            session_cls=(_EditSpatialRegionalSession if spatial_on
                         else _EditRegionalSession),
            into_transformer_options=True,
            session_kwargs=({
                "spatial_token_ranges": spatial_ranges,
                "spatial_prompts": spatial_prompts,
            } if spatial_on else None))

        patched = self._arm_edit_official(patched, src_latents)

        # post-CFG hook: must be set on the final patcher, after both clones.
        mold_previews = []
        n_mold = self._arm_photo_molds(
            patched, vae, lora_active, lora_boxes, ref_strength,
            ref_start_percent, ref_end_percent, ref_feather, mold_previews)

        negative = _grounded_cond(clip, negative_prompt or "", ground_imgs, gpx)
        data = {
            "adapters": [
                {
                    "name": r["name"],
                    "lora": r["lora"],
                    "strength": float(r.get("strength", 1.0)) * float(base_strength),
                    "ref_image": r.get("ref_image", ""),
                    "source": "in_node_ref",
                }
                for r in active[: len(placed)]
            ],
            "model_type": "krea2",
            "engine": engine,
            "mode": "krea2edit+v2_likeness",
            "subjects": len(placed),
            "externals": len(extras),
            "character_loras_armed": n_lora,
            "molds": n_mold,
            "reference_frames": len(src_latents),
            "spatial_conditioning": spatial_on,
            "spatial_prompts": spatial_prompts if spatial_on else [],
            "ref_paths": {
                "in_node_region_refs": len(placed),
                "standalone_extra_refs": len(extras),
            },
        }
        # edit_lora_strength is logged next to the character strengths because the
        # two deltas stack in one forward; their SUM is what drives artifacts.
        char_strengths = [
            round(float(r.get("strength", 1.0)) * float(base_strength), 2)
            for r in lora_active if has_lora(r)
        ]
        # Sequence length is the dominant cost in this node and it is invisible
        # otherwise, so spell it out: every reference frame is concatenated into
        # the attention sequence and attention is quadratic in that total.
        frame_tok = [(int(g.shape[2]) // 16) * (int(g.shape[1]) // 16)
                     for g in ground_imgs]
        tgt_tok = (cw // 16) * (ch // 16)
        logging.info(
            "[V9/edit] attention budget: frames %s + target %d = %d tokens "
            "(a plain generation would be %d) | ref_max_side=%s",
            frame_tok, tgt_tok, sum(frame_tok) + tgt_tok, tgt_tok,
            ref_max_side or "native",
        )
        logging.info(
            "[V9] krea2edit+V2: %s | %d in-node photo(s), %d LoadImage plate(s), "
            "%d krea2edit frame(s), %d char LoRA(s) %s, %d mold(s) strength=%.2f, "
            "edit_lora=%.2f",
            engine, len(placed), len(extras), len(src_latents), n_lora,
            char_strengths, n_mold, float(ref_strength), float(edit_lora_strength),
        )
        _release_transient_clones()
        return (patched, positive, negative, preview, data)

    # ------------------------------------------------------------------
    # shared arming helpers (V2 engine)
    # ------------------------------------------------------------------
    @staticmethod
    def _pair_boxes(regions, frame, cw, ch, keep):
        """Pair kept regions with their box by ORIGINAL row index.

        Row i of regions_json belongs to box i (the UI keeps row count synced to
        the box count). Indexing by position in the filtered list instead would
        hand row 2's LoRA the box belonging to row 1 whenever an earlier row is
        empty or disabled.
        """
        pairs = []
        for idx, r in enumerate(regions):
            if not keep(r):
                continue
            if frame and idx < len(frame):
                box = _coerce_bbox_norm(frame[idx], cw, ch)
            else:
                box = (0.0, 0.0, 1.0, 1.0)
            pairs.append((r, box))
        return pairs

    def _arm_regional_loras(
        self, model, active, norm_boxes, cw, ch, seam_feather, blend_override,
        base_strength, wrapper_key=WRAPPER_KEY_V9, session_cls=_RegionalSession,
        into_transformer_options=False, session_kwargs=None,
    ):
        file_cache = {}
        region_loras = []
        strength_eff = []
        for r in active:
            if r["lora"] in ("None", "") or (r["strength"] * base_strength) == 0.0:
                region_loras.append({})
                strength_eff.append(0.0)
                continue
            path = _resolve_lora_path(r["lora"])
            if path not in file_cache:
                file_cache[path] = _load_lora_matrices(path)
            s = float(r["strength"]) * float(base_strength)
            strength_eff.append(s)
            base_mats = file_cache[path]
            mats = {
                sig: {**{k: v for k, v in d.items() if k != "scale"},
                      "scale": d["scale"] * s}
                for sig, d in base_mats.items()
            }
            region_loras.append(mats)

        patched = model.clone()
        n_lora = sum(1 for m in region_loras if m)
        if n_lora:
            session = session_cls(
                patched, region_loras, norm_boxes,
                float(seam_feather), float(blend_override), cw, ch,
                **(session_kwargs or {}),
            )

            def wrapper(executor, *args, **kwargs):
                return session.run(executor, *args, **kwargs)

            if into_transformer_options and _pext is None:
                logging.warning(
                    "[V9] comfy.patcher_extension unavailable; regional LoRAs may be "
                    "shadowed by the krea2edit forward."
                )
                into_transformer_options = False

            if into_transformer_options:
                # Wrappers on ModelPatcher.wrappers are merged into
                # transformer_options AFTER anything already registered there
                # (comfy/sampler_helpers.py), which would put the krea2edit
                # wrapper outermost. Since krea2_edit_forward never calls the
                # next executor, this session would then never run. Registering
                # here — before the edit wrapper — keeps it outermost.
                to = patched.model_options.setdefault("transformer_options", {})
                _pext.add_wrapper_with_key(_WRAPPER_ENUM, wrapper_key, wrapper, to)
            elif hasattr(patched, "add_wrapper_with_key"):
                patched.add_wrapper_with_key(_WRAPPER_ENUM, wrapper_key, wrapper)
            elif hasattr(patched, "add_wrapper"):
                patched.add_wrapper(_WRAPPER_ENUM, wrapper)
            else:
                raise RuntimeError("ComfyUI build lacks model wrapper support.")
        return patched, strength_eff, n_lora

    def _arm_photo_molds(
        self, patched, vae, active, norm_boxes, ref_strength,
        ref_start_percent, ref_end_percent, ref_feather, preview_imgs_out,
    ):
        if vae is None or float(ref_strength) <= 0.0:
            return 0
        if float(ref_end_percent) <= float(ref_start_percent):
            logging.warning(
                "[V9] mold window is empty (ref_start=%.2f >= ref_end=%.2f) — no molds. "
                "Expected start ~0.0 and end ~0.60; are these two swapped?",
                float(ref_start_percent), float(ref_end_percent),
            )
            return 0

        entries = []
        for i, r in enumerate(active):
            if not (r.get("ref_image") and r.get("ref_enable", True)):
                continue
            try:
                img = _load_ref_image_tensor(r["ref_image"])
            except Exception as e:
                logging.warning("[V9] could not load in-node ref '%s': %s", r.get("ref_image"), e)
                continue
            preview_imgs_out.append(img[:1, :, :, :3])
            box = norm_boxes[i]
            if box[2] - box[0] < 1e-3 or box[3] - box[1] < 1e-3:
                continue
            entries.append((box, _encode_reference(patched, vae, img)))

        if not entries:
            return 0

        sigma_start, sigma_end = _sigma_window(patched, ref_start_percent, ref_end_percent)
        w, fth = float(ref_strength), float(ref_feather)
        state = {"key": None, "built": [], "logged": False}

        def post_cfg(args):
            denoised = args["denoised"]
            # RESTORE 2026-07-24: Krea2 samples in 5D (B,C,1,H,W), so this guard
            # returns on every single step and the mold never runs. ref_strength
            # is therefore inert and identity comes purely from the regional LoRA
            # engine — the configuration that produced the best likeness.
            # To re-enable the mold, put back:
            #     d4, _ = _denoised_as_4d(denoised)
            #     if d4 is None or not _in_window(...): return denoised
            if denoised.dim() != 4 or not _in_window(args["sigma"], sigma_start, sigma_end):
                return denoised
            d4 = denoised
            C, H, W = d4.shape[1], d4.shape[2], d4.shape[3]
            if state["key"] != (C, H, W):
                built = []
                for box, ref_ms in entries:
                    mm = _build_mold(ref_ms, box, C, H, W, fth, d4.device)
                    if mm is not None:
                        built.append((mm[0], _clip_mask_to_box(mm[1], box, H, W)))
                state["built"], state["key"] = built, (C, H, W)
                if not state["logged"]:
                    logging.info(
                        "[V9] V2 mold: %d armed, latent %dx%d sigma %.4f->%.4f strength %.2f",
                        len(built), W, H, sigma_start, sigma_end, w,
                    )
                    state["logged"] = True
            return _blend_ref_molds(denoised, state["built"], w)

        patched.set_model_sampler_post_cfg_function(post_cfg)
        return len(entries)


NODE_CLASS_MAPPINGS = {"Krea2RegionalMultiLoRAV9": Krea2RegionalMultiLoRAV9}
NODE_DISPLAY_NAME_MAPPINGS = {
    "Krea2RegionalMultiLoRAV9": (
        "Krea2 Regional Multi-LoRA v9 (V2 likeness + edit anything, By Fedor)"
    ),
}

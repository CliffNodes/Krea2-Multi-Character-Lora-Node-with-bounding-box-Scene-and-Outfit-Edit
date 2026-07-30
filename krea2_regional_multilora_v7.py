"""
Krea2RegionalMultiLoRAV7 (By Fedor) - V5 shape, krea2edit power.

Same one-node UX as v5 (a box builder feeds bboxes; regions_json lists a LoRA
and/or a reference image per region, in box order), but the reference engine is
the krea2edit IN-CONTEXT injection we validated in Krea2RegionalEditPatch instead
of the soft "latent mold". Per region we obtain a clean reference latent and inject
it as its own RoPE frame placed at the region's target bounding box:

    sequence per step = [text | ref_1 | ... | ref_N | target]

Each reference latent comes from ONE of two sources (v5 mental model preserved):
  - ref_image (a loaded photo)                     -> VAE-encoded, or
  - lora only (no photo)                           -> the node auto-generates a clean
                                                      single-subject portrait FROM that
                                                      LoRA (each LoRA rendered ALONE =
                                                      zero cross-LoRA bleed), then injects
                                                      that portrait as the region frame.

Because identities are baked before the compose, there is no inline LoRA masking during
the main pass (that engine lives in v3/v4/v5 and is untouched). The Krea2 Identity Edit
LoRA is applied to the model internally, so the workflow just wires the base model in.

Options:
  - attention_isolation: per-region (seq,seq) mask so references never cross-attend and a
    target token only reads the reference(s) of the box it sits in (kills bleed for
    similar/overlapping subjects). Needs a mask-capable attention backend.
  - latent_mold (optional, default off): ALSO run the v3 latent-mold pull toward the same
    reference latents, for an extra identity lock.

regions_json schema (superset of v3/v5):
    [
      {"lora": "charA.safetensors", "strength": 1.1, "enable": true,
       "ref_image": "", "prompt": "a man"},          # -> auto portrait from LoRA
      {"lora": "None", "ref_image": "friend.png", "enable": true}  # -> real photo
    ]
"""
import importlib.util
import logging
import math
from pathlib import Path

import numpy as np
import torch

import comfy.patcher_extension
import comfy.sample
import comfy.sd
import comfy.utils
import folder_paths

from .krea2_regional_multilora import (
    _coerce_bbox_norm,
    _normalize_bboxes,
    _resolve_lora_path,
)
from .krea2_regional_multilora_v3 import (
    _load_ref_image_tensor,
    _parse_regions_v3,
)
from .krea2_regional_edit_patch import regional_edit_forward, _parse_box
from .krea2_reference_lock import _build_mold, _in_window, _sigma_window


_KREA2_EDIT_FORWARD = None  # lazy-resolved on first compose


def _get_krea2_edit_forward():
    """Official full-frame edit forward from comfyui-krea2edit (training-matched).

    Resolved lazily so we pick up the already-loaded custom node inside ComfyUI
    (sys.modules) instead of re-executing its __init__ at import time.
    """
    global _KREA2_EDIT_FORWARD
    if _KREA2_EDIT_FORWARD is not None:
        return _KREA2_EDIT_FORWARD if _KREA2_EDIT_FORWARD is not False else None

    import sys
    for name, mod in list(sys.modules.items()):
        if not mod or "krea2edit" not in name.replace("-", "").lower():
            continue
        fn = getattr(mod, "krea2_edit_forward", None)
        if callable(fn):
            _KREA2_EDIT_FORWARD = fn
            return fn

    try:
        p = Path(__file__).resolve().parent.parent / "comfyui-krea2edit" / "__init__.py"
        spec = importlib.util.spec_from_file_location("_krea2edit_fwd", p)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _KREA2_EDIT_FORWARD = mod.krea2_edit_forward
        return _KREA2_EDIT_FORWARD
    except Exception as e:
        logging.warning("[V7] krea2_edit_forward unavailable (%s); using regional fallback", e)
        _KREA2_EDIT_FORWARD = False
        return None

WRAPPER_KEY_V7 = "krea2_regional_multilora_v7"

DEFAULT_REGIONS_JSON_V7 = (
    "[\n"
    '  {"lora": "None", "strength": 1.0, "enable": true, "ref_image": "", "prompt": ""},\n'
    '  {"lora": "None", "strength": 1.0, "enable": true, "ref_image": "", "prompt": ""}\n'
    "]"
)

DEFAULT_PORTRAIT_PROMPT = (
    "professional photorealistic studio portrait photo of a person, plain neutral gray "
    "background, upper body, looking at camera, sharp focus, soft natural lighting"
)


def _edit_lora_choices():
    files = folder_paths.get_filename_list("loras")
    default = "krea2_identity_edit_v1_2.safetensors"
    if default in files:
        files = [default] + [f for f in files if f != default]
    return files or [default]


def _portrait_size(box_norm, cw, ch):
    """Region pixel size (matching the box AR), clamped + rounded to /16."""
    x0, y0, x1, y1 = box_norm
    w = max(1.0, (x1 - x0) * cw)
    h = max(1.0, (y1 - y0) * ch)
    ar = w / h
    area = 832.0 * 832.0
    ph = math.sqrt(area / ar)
    pw = ph * ar
    scale = 1.0
    if min(pw, ph) < 640:
        scale = 640.0 / min(pw, ph)
    pw, ph = pw * scale, ph * scale
    if max(pw, ph) > 1152:
        scale = 1152.0 / max(pw, ph)
        pw, ph = pw * scale, ph * scale
    r16 = lambda v: max(256, int(round(v / 16.0)) * 16)
    return r16(pw), r16(ph)


def _region_prompt(row):
    p = str(row.get("prompt", "") or "").strip()
    return p if p else DEFAULT_PORTRAIT_PROMPT


# --- Grounded encode (krea2edit semantic path) -----------------------------
# Mirrors comfyui-krea2edit's Krea2EditGroundedEncode, generalized to N refs:
# the instruction is encoded THROUGH Qwen3-VL together with the reference images
# (scene first, then subjects) - the semantic half of the krea2edit recipe.
_KREA2_GROUND_SYS = (
    "<|im_start|>system\nDescribe the image by detailing the color, shape, size, "
    "texture, quantity, text, spatial relationships of the objects and background:"
    "<|im_end|>\n<|im_start|>user\n"
)
_KREA2_GROUND_TAIL = "{}<|im_end|>\n<|im_start|>assistant\n"
_VISION_BLOCK = "<|vision_start|><|image_pad|><|vision_end|>"


def _grounding_template(n):
    return _KREA2_GROUND_SYS + (_VISION_BLOCK * max(1, n)) + _KREA2_GROUND_TAIL


def _prep_ground_img(image, gpx):
    """[B,H,W,3] -> capped longest side, RGB. Matches Krea2EditGroundedEncode._prep."""
    samples = image.movedim(-1, 1)  # B,H,W,C -> B,C,H,W
    h, w = samples.shape[2], samples.shape[3]
    if gpx and max(h, w) > gpx:
        s = gpx / max(h, w)
        samples = comfy.utils.common_upscale(samples, round(w * s), round(h * s), "area", "disabled")
    return samples.movedim(1, -1)[:, :, :, :3]


def _grounded_cond(clip, prompt, imgs, gpx):
    """Grounded conditioning if imgs given, else plain text encode (fallback)."""
    prompt = prompt or ""
    if imgs:
        prepped = [_prep_ground_img(im, gpx) for im in imgs]
        tokens = clip.tokenize(prompt, images=prepped, llama_template=_grounding_template(len(prepped)))
    else:
        tokens = clip.tokenize(prompt)
    return clip.encode_from_tokens_scheduled(tokens)


# --- Spatial reference collage (the proven compositing engine) --------------
# Composite each subject into its target bbox on a scene canvas, then feed the
# WHOLE collage as ONE full-frame krea2edit source. This is in-distribution for
# the edit LoRA (full-frame ref -> full-frame target) => strong identity + cohesive
# harmonization, unlike shrunken per-region frames.
def _cover_resize(img, tw, th):
    """[1,H,W,3] 0..1 -> [1,th,tw,3], scaled to cover then center-cropped.

    OK for filling a scene canvas / pasting into a bbox. Do NOT use on identity
    subject sheets before compose — cropping throws away face pixels.
    """
    x = img.movedim(-1, 1).float()   # [1,3,H,W]
    _, _, h, w = x.shape
    s = max(tw / max(1, w), th / max(1, h))
    nw, nh = max(1, round(w * s)), max(1, round(h * s))
    x = torch.nn.functional.interpolate(x, size=(nh, nw), mode="bilinear", align_corners=False)
    y0, x0 = max(0, (nh - th) // 2), max(0, (nw - tw) // 2)
    x = x[:, :, y0:y0 + th, x0:x0 + tw]
    return x.movedim(1, -1)          # [1,th,tw,3]


def _stretch_resize(img, tw, th):
    """[1,H,W,3] -> [1,th,tw,3] bilinear stretch. Keeps every pixel (may distort AR).

    Matches official krea2edit latent-side interpolate: identity sheets must not be
    cover-cropped before VAE encode.
    """
    x = img.movedim(-1, 1).float()
    x = torch.nn.functional.interpolate(x, size=(th, tw), mode="bilinear", align_corners=False)
    return x.movedim(1, -1)


def _portrait_canvas_size(cw, ch, max_side=1024):
    """Portrait at the SAME aspect ratio as the final canvas (krea2edit requirement).

    Caps longest side so auto-portraits stay fast, but never forces square — square
    portraits cover-cropped onto a landscape scene were destroying face identity.
    """
    cw, ch = max(64, int(cw)), max(64, int(ch))
    m = max(cw, ch)
    if m > max_side:
        s = max_side / m
        cw, ch = int(cw * s), int(ch * s)
    cw = max(64, int(round(cw / 16.0) * 16))
    ch = max(64, int(round(ch / 16.0) * 16))
    return cw, ch


def _merge_identity_prompt(user_prompt, who_tag, side, multi=False):
    """Always keep the training-matched 'from the Nth reference' anchor.

    A bare scene description without that phrase is why likeness looked perfect in
    the portrait preview and then vanished in the composite.
    """
    who = (who_tag or "person").strip()
    if multi:
        anchor = (f"the people already in the first reference and the {who} from the "
                  f"second reference standing {side}, keep the scene unchanged, "
                  f"both faces clearly visible, candid photo")
    else:
        anchor = (f"the {who} from the second reference, standing {side}, "
                  f"keep the scene from the first reference unchanged, "
                  f"candid photo, face clearly visible")
    u = (user_prompt or "").strip()
    if not u:
        return anchor
    # User text first (scene/action), then the hard identity binding.
    return f"{u}, {anchor}"


def _paste_tile(canvas, src, px, py, pw, ph):
    H, W = canvas.shape[1], canvas.shape[2]
    pw, ph = max(1, pw), max(1, ph)
    px, py = max(0, min(px, W - 1)), max(0, min(py, H - 1))
    pw, ph = min(pw, W - px), min(ph, H - py)
    tile = _cover_resize(src, pw, ph)
    canvas[:, py:py + ph, px:px + pw, :] = tile[:, :ph, :pw, :].to(canvas.dtype)


def _build_collage(placed, extras, cw, ch):
    """placed: [(subject_pixels[1,H,W,3], (x0,y0,x1,y1))]. extras: [(img, 'x,y,w,h')].
    Extras (scene/object) are laid down first (full-canvas box = background); subjects
    on top at their boxes. Returns [1,ch,cw,3] float in 0..1."""
    canvas = torch.full((1, ch, cw, 3), 0.5, dtype=torch.float32)
    for img, boxstr in extras:
        x, y, w, h = _parse_box(boxstr)
        _paste_tile(canvas, img[:1, :, :, :3], round(x * cw), round(y * ch),
                    round(w * cw), round(h * ch))
    for subj, nb in placed:
        x0, y0, x1, y1 = nb
        _paste_tile(canvas, subj, round(x0 * cw), round(y0 * ch),
                    round((x1 - x0) * cw), round((y1 - y0) * ch))
    return canvas.clamp(0, 1)


class Krea2RegionalMultiLoRAV7:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "Base Krea2 UNet (no edit LoRA; it is applied here)."}),
                "clip": ("CLIP",),
                "vae": ("VAE", {"tooltip": "Encodes reference photos and previews portraits."}),
                "edit_lora": (_edit_lora_choices(), {
                    "tooltip": "Krea2 Identity Edit LoRA, applied internally for the compose pass.",
                }),
                "canvas_width": ("INT", {"default": 1216, "min": 64, "max": 16384, "step": 16}),
                "canvas_height": ("INT", {"default": 832, "min": 64, "max": 16384, "step": 16}),
                "regions_json": ("STRING", {
                    "multiline": True,
                    "default": DEFAULT_REGIONS_JSON_V7,
                    "tooltip": (
                        "One row per region, in box order. Each: {\"lora\": \"file\", "
                        "\"strength\": 1.0, \"enable\": true, \"ref_image\": \"photo.png\", "
                        "\"prompt\": \"a man\"}. ref_image (a loaded photo) is used as the "
                        "region's reference if present; otherwise the LoRA is rendered alone "
                        "into a portrait and that is used. 'prompt' guides the auto portrait."
                    ),
                }),
                "attention_isolation": ("BOOLEAN", {
                    "default": False,
                    "tooltip": (
                        "Isolate each region's reference (kills cross-identity bleed). Needs a "
                        "mask-capable attention backend; disable if using --use-sage-attention "
                        "and it breaks (placement is already strong without it)."
                    ),
                }),
                "portrait_steps": ("INT", {"default": 10, "min": 4, "max": 40, "step": 1,
                    "tooltip": "Turbo steps for each auto-generated LoRA portrait."}),
                "portrait_seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            },
            "optional": {
                "bboxes": ("BOUNDING_BOX", {
                    "forceInput": True,  # keep socket-only; frontend widget group corrupts saves
                    "tooltip": "Boxes from a box builder (e.g. Ideogram4PromptBuilderKJ)."}),
                "extra_ref_1": ("IMAGE", {
                    "tooltip": "External reference (a scene/setting, object, or anything). Load "
                               "it with a LoadImage node. Injected as an in-context frame at "
                               "extra_box_1 - outside the per-region LoRA refs."}),
                "extra_box_1": ("STRING", {"default": "0,0,1,1",
                    "tooltip": "x,y,w,h in 0..1 for extra_ref_1. Full canvas 0,0,1,1 = a "
                               "scene/background reference; a smaller box = place an object."}),
                "extra_ref_2": ("IMAGE", {
                    "tooltip": "Second external reference image (scene/object/thing)."}),
                "extra_box_2": ("STRING", {"default": "0,0,1,1",
                    "tooltip": "x,y,w,h in 0..1 for extra_ref_2."}),
                "edit_lora_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05}),
                "base_strength": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.05,
                    "tooltip": "Global multiplier on every region's LoRA strength (portrait gen)."}),
                "portrait_prompt": ("STRING", {"multiline": True, "default": DEFAULT_PORTRAIT_PROMPT,
                    "tooltip": "Fallback portrait prompt for LoRA rows without their own 'prompt'."}),
                "prompt": ("STRING", {"multiline": True, "default": "",
                    "tooltip": "Scene / instruction for the composite. Encoded THROUGH Qwen3-VL "
                               "with the reference images (grounded encode = the krea2edit "
                               "semantic path). Describe the whole image + where each subject is, "
                               "e.g. 'a man on the left and a woman on the right on a rooftop at "
                               "golden hour'. Feeds the positive output."}),
                "negative_prompt": ("STRING", {"multiline": True, "default": "",
                    "tooltip": "Negative prompt (grounded on the same refs). Usually empty for turbo/CFG 1."}),
                "grounding_px": ("INT", {"default": 1024, "min": 0, "max": 4096, "step": 64,
                    "tooltip": "Cap longest side fed to Qwen3-VL. Higher = stronger identity "
                               "(1024+ for people); lower = stronger scene edits. 0 = native."}),
                "latent_mold": ("BOOLEAN", {"default": False,
                    "tooltip": "Also pull the denoised latent toward each region reference (extra lock)."}),
                "mold_strength": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 1.0, "step": 0.01}),
                "mold_end_percent": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
        }

    RETURN_TYPES = ("MODEL", "CONDITIONING", "CONDITIONING", "IMAGE", "KREA2_DATA")
    RETURN_NAMES = ("model", "positive", "negative", "reference_preview", "data")
    FUNCTION = "apply"
    CATEGORY = "Krea2/By Fedor"
    DESCRIPTION = (
        "Krea2 Regional Multi-LoRA v7 (By Fedor). V5 shape (box builder + one regions_json "
        "node), krea2edit power. Each region's LoRA is rendered ALONE into a clean portrait "
        "(or a loaded photo is used) so identities never cross-bleed. Two compose engines, "
        "auto-selected: (1) if a scene/external reference is loaded, the scene is kept as a "
        "CLEAN full-frame krea2edit reference (never regenerated) and each subject is injected "
        "as its own frame at its bbox - the proven two-ref recipe, so the ACTUAL reference "
        "scene is preserved; (2) with no scene, subjects are composited onto a neutral canvas "
        "and the whole collage is one full-frame source while the prompt paints the setting. "
        "Applies the Krea2 Identity Edit LoRA internally and grounds the prompt on the refs "
        "through Qwen3-VL, outputting positive/negative CONDITIONING. Optional latent-mold lock."
    )

    def _gen_portrait_latent(self, base_model, clip, lora_name, strength, prompt, pw, ph, steps, seed):
        model_l, clip_l = base_model, clip
        if lora_name not in ("None", "") and strength != 0.0:
            path = _resolve_lora_path(lora_name)
            lora_sd = comfy.utils.load_torch_file(path, safe_load=True)
            model_l, clip_l = comfy.sd.load_lora_for_models(base_model, clip, lora_sd, strength, strength)
        pos = clip_l.encode_from_tokens_scheduled(clip_l.tokenize(prompt))
        neg = clip_l.encode_from_tokens_scheduled(clip_l.tokenize(""))
        latent = torch.zeros([1, 16, ph // 8, pw // 8])
        latent = comfy.sample.fix_empty_latent_channels(model_l, latent)
        noise = comfy.sample.prepare_noise(latent, seed)
        samples = comfy.sample.sample(
            model_l, noise, steps, 1.0, "euler", "simple", pos, neg, latent,
            denoise=1.0, disable_pbar=True, seed=seed)
        return samples

    def _arm_edit(self, edit_model, region_payload, isolation=False):
        """Clone edit_model and wrap with regional_edit_forward for the given frames."""
        patched = edit_model.clone()
        iso = bool(isolation)

        def wrapper(executor, x, timesteps, context, attention_mask=None,
                    transformer_options={}, **kwargs):
            return regional_edit_forward(executor.class_obj, x, timesteps, context,
                                         region_payload, iso, transformer_options)

        to = patched.model_options.setdefault("transformer_options", {})
        comfy.patcher_extension.add_wrapper_with_key(
            comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL, WRAPPER_KEY_V7, wrapper, to)
        return patched

    def _arm_edit_official(self, edit_model, src_latents):
        """Training-matched full-frame path: official krea2_edit_forward (list = multi-ref).

        Falls back to regional_edit_forward with full-canvas boxes if the official
        module is unavailable.
        """
        fwd = _get_krea2_edit_forward()
        if fwd is None:
            payload = [{"latent": sl, "box": (0.0, 0.0, 1.0, 1.0)} for sl in src_latents]
            return self._arm_edit(edit_model, payload, isolation=False)

        patched = edit_model.clone()
        srcs = list(src_latents)

        def wrapper(executor, x, timesteps, context, attention_mask=None,
                    transformer_options={}, **kwargs):
            return fwd(executor.class_obj, x, timesteps, context, srcs, transformer_options)

        to = patched.model_options.setdefault("transformer_options", {})
        comfy.patcher_extension.add_wrapper_with_key(
            comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL, WRAPPER_KEY_V7, wrapper, to)
        return patched

    def _decode_px(self, vae, samples):
        dec = vae.decode(samples)
        if dec.dim() > 4:
            dec = dec.reshape(-1, *dec.shape[-3:])
        return dec[:1].clamp(0, 1)

    def _compose_once(self, edit_model, clip, vae, scene_px, subj_px, prompt, neg_prompt,
                      gpx, cw, ch, steps, seed):
        """One official two-ref insert: scene (frame1) + subject (frame2).

        Scene is stretched to the target canvas (AR match). Subject is encoded at its
        NATIVE resolution — krea2_edit_forward bilinear-resizes in latent space, which
        is what the edit LoRA was trained on. Never cover-crop the subject.
        """
        def _enc_ms(px):
            return edit_model.model.process_latent_in(vae.encode(px[:, :, :, :3]))

        scene = _stretch_resize(scene_px[:1, :, :, :3], cw, ch)
        subj = subj_px[:1, :, :, :3].float().clamp(0, 1)  # native; no crop
        srcs = [_enc_ms(scene), _enc_ms(subj)]
        patched = self._arm_edit_official(edit_model, srcs)
        # Ground on the same images the VAE path sees (native subject, canvas scene).
        ground = [scene, subj]
        pos = _grounded_cond(clip, prompt, ground, gpx)
        neg = _grounded_cond(clip, neg_prompt or "", ground, gpx)
        latent = torch.zeros([1, 16, ch // 8, cw // 8])
        latent = comfy.sample.fix_empty_latent_channels(patched, latent)
        noise = comfy.sample.prepare_noise(latent, seed)
        samples = comfy.sample.sample(
            patched, noise, int(steps), 1.0, "euler", "simple", pos, neg, latent,
            denoise=1.0, disable_pbar=True, seed=seed)
        return self._decode_px(vae, samples)

    @staticmethod
    def _side_label(nb, i):
        """Rough left/right/center from a normalized box for per-pass prompts."""
        x0, y0, x1, y1 = nb
        cx = 0.5 * (x0 + x1)
        if cx < 0.4:
            return "on the left"
        if cx > 0.6:
            return "on the right"
        return f"in region {i + 1}"

    def apply(self, model, clip, vae, edit_lora, canvas_width, canvas_height, regions_json,
              attention_isolation, portrait_steps, portrait_seed, bboxes=None,
              extra_ref_1=None, extra_box_1="0,0,1,1", extra_ref_2=None, extra_box_2="0,0,1,1",
              edit_lora_strength=1.0, base_strength=1.0, portrait_prompt=DEFAULT_PORTRAIT_PROMPT,
              prompt="", negative_prompt="", grounding_px=768,
              latent_mold=False, mold_strength=0.25, mold_end_percent=0.5):
        cw, ch = int(canvas_width), int(canvas_height)
        gpx = int(grounding_px)
        regions = _parse_regions_v3(regions_json)

        def has_lora(r):
            return r["lora"] not in ("None", "") and (r["strength"] * base_strength) != 0.0

        def has_ref(r):
            return bool(r.get("ref_image")) and r.get("ref_enable", True)

        active = [r for r in regions if r["enable"] and (has_lora(r) or has_ref(r))]
        extras = [(extra_ref_1, extra_box_1), (extra_ref_2, extra_box_2)]
        extras = [(img, box) for img, box in extras if img is not None]
        empty_img = torch.zeros(1, 64, 64, 3)
        if not active and not extras:
            logging.warning("[V7] no active regions or external refs; passing model through.")
            pos = _grounded_cond(clip, prompt, [], gpx)
            neg = _grounded_cond(clip, negative_prompt, [], gpx)
            return (model, pos, neg, empty_img, {"adapters": []})

        frame = _normalize_bboxes(bboxes)
        norm_boxes = []
        for i in range(len(active)):
            if frame and i < len(frame):
                norm_boxes.append(_coerce_bbox_norm(frame[i], cw, ch))
            else:
                logging.warning("[V7] region %d has no bbox; using full canvas.", i)
                norm_boxes.append((0.0, 0.0, 1.0, 1.0))

        # Apply the edit LoRA once for the compose pass.
        edit_path = _resolve_lora_path(edit_lora)
        edit_sd = comfy.utils.load_torch_file(edit_path, safe_load=True)
        edit_model = comfy.sd.load_lora_for_models(model, None, edit_sd, float(edit_lora_strength), 0)[0]

        # Build each subject's reference image (a loaded photo, or an auto portrait
        # rendered from that region's LoRA ALONE = zero cross-LoRA bleed).
        placed = []           # (subject_pixels[1,H,W,3], (x0,y0,x1,y1), who_label)
        mold_latents = []     # (model_space_latent, norm_box) for optional latent-mold
        default_prompt = portrait_prompt or DEFAULT_PORTRAIT_PROMPT
        for i, (r, nb) in enumerate(zip(active, norm_boxes)):
            subj_px = None
            subj_raw = None
            # Keep a short identity tag for sequential prompts ("a man" / "a woman").
            who = str(r.get("prompt", "") or "").strip()
            who_tag = who.split(",")[0].strip() if who else f"person {i + 1}"
            if has_ref(r):
                try:
                    img = _load_ref_image_tensor(r["ref_image"])
                    subj_px = img[:1, :, :, :3]
                except Exception as e:
                    logging.warning("[V7] ref '%s' (region %d) failed: %s; falling back to LoRA.",
                                    r.get("ref_image"), i, e)
                    subj_px = None
            if subj_px is None and has_lora(r):
                # Same aspect ratio as the final canvas (krea2edit README: mismatched AR
                # visibly kills identity). Cap longest side for speed; encode native later.
                pw, ph = _portrait_canvas_size(cw, ch, max_side=1024)
                prompt_i = str(r.get("prompt", "") or "").strip() or default_prompt
                if "background" not in prompt_i.lower():
                    prompt_i = (f"{prompt_i}, plain neutral gray background, upper body, "
                                "looking at camera, sharp focus, soft natural lighting")
                s = float(r["strength"]) * float(base_strength)
                seed = int(portrait_seed) + i * 1000
                logging.info("[V7] region %d: rendering portrait from %s @%dx%d (%d steps)",
                             i, r["lora"], pw, ph, int(portrait_steps))
                subj_raw = self._gen_portrait_latent(model, clip, r["lora"], s, prompt_i, pw, ph,
                                                     int(portrait_steps), seed)
                try:
                    subj_px = self._decode_px(vae, subj_raw)
                except Exception as e:
                    logging.warning("[V7] portrait decode failed region %d: %s", i, e)
                    subj_px = None
            if subj_px is None:
                logging.warning("[V7] region %d has neither ref nor LoRA image; skipped.", i)
                continue
            placed.append((subj_px, nb, who_tag))
            if latent_mold:
                try:
                    enc = subj_raw if subj_raw is not None else vae.encode(subj_px[:, :, :, :3])
                    mold_latents.append((edit_model.model.process_latent_in(enc), nb))
                except Exception:
                    pass

        if not placed and not extras:
            logging.warning("[V7] no usable references; passing edit model through.")
            pos = _grounded_cond(clip, prompt, [], gpx)
            neg = _grounded_cond(clip, negative_prompt, [], gpx)
            return (edit_model, pos, neg, empty_img, {"adapters": []})

        def _enc_ms(px):
            return edit_model.model.process_latent_in(vae.encode(px[:, :, :, :3]))

        # Engines:
        #  * SCENE + 1 subject  -> official two-ref (scene + full-frame subject).
        #  * SCENE + 2+ subjects -> CHAINED single-subject inserts (krea2edit README:
        #    more face-faithful than one multi-person pass). N-1 inserts run inside
        #    this node; the external KSampler does the final insert.
        #  * NO scene           -> spatial collage (prompt paints the setting).
        compose_steps = max(8, min(12, int(portrait_steps)))
        if extras:
            # Build the base scene canvas (full-frame extras cover-resized; smaller boxes
            # are pasted onto a neutral canvas first so objects still land correctly).
            scene_canvas = torch.full((1, ch, cw, 3), 0.5, dtype=torch.float32)
            has_full_scene = False
            for img, boxstr in extras:
                bx, by, bw, bh = _parse_box(boxstr)
                if bw >= 0.999 and bh >= 0.999 and bx <= 1e-3 and by <= 1e-3:
                    scene_canvas = _cover_resize(img[:1, :, :, :3], cw, ch)
                    has_full_scene = True
                else:
                    _paste_tile(scene_canvas, img[:1, :, :, :3],
                                round(bx * cw), round(by * ch),
                                round(bw * cw), round(bh * ch))
            if not has_full_scene and extras:
                # Only objects — still treat the canvas (with objects pasted) as the scene.
                pass

            current = scene_canvas.clamp(0, 1)
            # Chain all but the last subject inside the node.
            for i, (subj_px, nb, who_tag) in enumerate(placed[:-1]):
                side = self._side_label(nb, i)
                pass_prompt = _merge_identity_prompt(prompt, who_tag, side, multi=False)
                logging.info("[V7] sequential insert %d/%d: %s %s", i + 1, len(placed), who_tag, side)
                current = self._compose_once(
                    edit_model, clip, vae, current, subj_px, pass_prompt, negative_prompt,
                    gpx, cw, ch, compose_steps, int(portrait_seed) + 10000 + i)

            # Arm the external KSampler for the final subject (or scene-only if no subjects).
            use_official = True
            if placed:
                subj_px, nb, who_tag = placed[-1]
                side = self._side_label(nb, len(placed) - 1)
                final_prompt = _merge_identity_prompt(
                    prompt, who_tag, side, multi=(len(placed) > 1))
                scene = _stretch_resize(current, cw, ch)
                subj = subj_px[:1, :, :, :3].float().clamp(0, 1)  # native — no cover crop
                src_latents = [_enc_ms(scene), _enc_ms(subj)]
                region_payload = None  # official arm uses src_latents
                ground_imgs = [scene, subj]
                prevs = [scene_canvas] + [sp for sp, _, _ in placed]
                preview_batch = self._stack_previews(
                    [p.movedim(-1, 1).float() for p in prevs])
                engine_name = ("krea2edit_scene_tworef" if len(placed) == 1
                               else "krea2edit_scene_sequential")
            else:
                scene = _stretch_resize(current, cw, ch)
                src_latents = [_enc_ms(scene)]
                region_payload = None
                ground_imgs = [scene]
                preview_batch = self._collage_preview(scene)
                final_prompt = prompt
                engine_name = "krea2edit_scene_only"
            iso_flag = False
            prompt = final_prompt
        else:
            collage = _build_collage([(p, nb) for p, nb, _ in placed], extras, cw, ch)
            src_latents = [_enc_ms(collage)]
            region_payload = None
            use_official = True
            ground_imgs = [collage]
            preview_batch = self._collage_preview(collage)
            iso_flag = False
            engine_name = "krea2edit_spatial_collage"

        if use_official:
            patched = self._arm_edit_official(edit_model, src_latents)
        else:
            patched = self._arm_edit(edit_model, region_payload, isolation=iso_flag)

        if latent_mold and float(mold_strength) > 0.0 and mold_latents:
            self._arm_latent_mold(patched, mold_latents, float(mold_strength), float(mold_end_percent))

        positive = _grounded_cond(clip, prompt, ground_imgs, gpx)
        negative = _grounded_cond(clip, negative_prompt, ground_imgs, gpx)
        node_data = {
            "adapters": [{"name": r["name"], "lora": r["lora"],
                          "strength": float(r["strength"]) * float(base_strength),
                          "ref_image": r.get("ref_image", ""),
                          "source": "photo" if has_ref(r) else "lora_portrait"}
                         for r in active[:len(placed)]],
            "model_type": "krea2",
            "engine": engine_name,
            "subjects": len(placed),
            "externals": len(extras),
        }
        logging.info("[V7] armed %s: %d subject(s), %d external ref(s), isolation=%s, latent_mold=%s",
                     engine_name, len(placed), len(extras), iso_flag, bool(latent_mold))
        return (patched, positive, negative, preview_batch, node_data)

    def _arm_latent_mold(self, patched, mold_latents, strength, end_percent):
        sigma_start, sigma_end = _sigma_window(patched, 0.0, end_percent)
        state = {"key": None, "built": []}

        def post_cfg(args):
            denoised = args["denoised"]
            if denoised.dim() != 4 or not _in_window(args["sigma"], sigma_start, sigma_end):
                return denoised
            C, H, W = denoised.shape[1], denoised.shape[2], denoised.shape[3]
            if state["key"] != (C, H, W):
                built = []
                for ms, nb in mold_latents:
                    mm = _build_mold(ms, nb, C, H, W, 0.06, denoised.device)
                    if mm is not None:
                        built.append(mm)
                state["built"], state["key"] = built, (C, H, W)
            d32 = denoised.float()
            for mold, mask in state["built"]:
                d32 = d32 + (strength * mask) * (mold - d32)
            return d32.to(denoised.dtype)

        patched.set_model_sampler_post_cfg_function(post_cfg)

    @staticmethod
    def _collage_preview(collage):
        """Downscale the collage to a preview (longest side <= 768)."""
        x = collage.movedim(-1, 1).float()   # [1,3,H,W]
        _, _, h, w = x.shape
        m = max(h, w)
        if m > 768:
            s = 768.0 / m
            x = torch.nn.functional.interpolate(x, size=(max(1, round(h * s)), max(1, round(w * s))),
                                                mode="bilinear", align_corners=False)
        return x.movedim(1, -1).clamp(0, 1)

    @staticmethod
    def _stack_previews(previews):
        imgs = [p for p in previews if p is not None]
        if not imgs:
            return torch.zeros(1, 64, 64, 3)
        out = []
        for p in imgs:
            x = torch.nn.functional.interpolate(p.float(), size=(512, 512), mode="bilinear")
            out.append(x.movedim(1, -1).clamp(0, 1))
        return torch.cat(out, dim=0)


NODE_CLASS_MAPPINGS = {"Krea2RegionalMultiLoRAV7": Krea2RegionalMultiLoRAV7}
NODE_DISPLAY_NAME_MAPPINGS = {
    "Krea2RegionalMultiLoRAV7": "Krea2 Regional Multi-LoRA v7 (edit-frame refs, By Fedor)"}

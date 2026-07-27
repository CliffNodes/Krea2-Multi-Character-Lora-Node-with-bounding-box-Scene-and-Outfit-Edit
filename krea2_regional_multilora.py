"""
Krea2RegionalMultiLoRA (By Fedor) - regional multi-LoRA for Krea 2 via masked
activation-delta injection (unlimited regions).

Each region's LoRA activation delta (x @ down.T @ up.T * scale) is added AT
FORWARD TIME, multiplied by a per-region token mask. Outside its box the mask
is 0, so the LoRA physically cannot act there - a hard spatial guarantee, not
an attention-bias nudge.

Features:
  * UNLIMITED regions via the regions_json widget + dynamic "Add Region" rows
    in web/krea2_regional_multilora.js. Rows auto-sync to the number of boxes
    drawn in a connected bounding-box builder.
  * BOUNDING_BOX wire (region i -> box i).
  * split_mode auto_vertical / auto_horizontal fallbacks when no boxes wired.
  * LoRA A/B matrices are loaded raw and matched to live model Linears by
    normalised name (works on fp8 models - never touches quantized weights).
    Both ComfyUI-native (blocks.0.attn.wq) and diffusers-style
    (transformer_blocks.0.attn.to_q) key naming are accepted.
  * Token masks are built at RUNTIME from the real latent grid, so canvas_width
    / canvas_height only matter for interpreting pixel-space bboxes.
  * clip passes through untouched (activation injection is UNet-side).

regions_json schema:
    [
      {"lora": "character_A.safetensors", "strength": 1.1, "enable": true},
      {"lora": "character_B.safetensors", "strength": 1.1, "enable": true}
    ]
"""

import json
import logging
import math
import re

import torch
import safetensors.torch

import folder_paths

try:
    import comfy.patcher_extension as _pext
    _WRAPPER_ENUM = _pext.WrappersMP.DIFFUSION_MODEL
except Exception:  # pragma: no cover
    _pext = None
    _WRAPPER_ENUM = "diffusion_model"

WRAPPER_KEY = "krea2_regional_multilora"
_COMPUTE_DTYPE = torch.bfloat16

DEFAULT_REGIONS_JSON = (
    "[\n"
    '  {"lora": "None", "strength": 1.1, "enable": true},\n'
    '  {"lora": "None", "strength": 1.1, "enable": true}\n'
    "]"
)


# ---------------------------------------------------------------------------
# region / bbox parsing
# ---------------------------------------------------------------------------
def _parse_regions(regions_json: str) -> list:
    if not regions_json or not regions_json.strip():
        return []
    try:
        raw = json.loads(regions_json)
    except (ValueError, TypeError) as e:
        logging.warning("[Krea2RegionalMultiLoRA] regions_json is not valid JSON (%s); no regions.", e)
        return []
    if isinstance(raw, dict):
        raw = [raw]
    if not isinstance(raw, list):
        return []

    out = []
    for i, item in enumerate(raw):
        if not isinstance(item, dict):
            continue
        lora = str(item.get("lora", item.get("lora_name", "None")) or "None")
        try:
            strength = float(item.get("strength", item.get("strength_model", 1.0)))
        except (TypeError, ValueError):
            strength = 1.0
        enable = bool(item.get("enable", True))
        name = str(item.get("name", "") or f"region{i}").strip() or f"region{i}"
        out.append({"name": name, "lora": lora, "strength": strength, "enable": enable})
    return out


def _normalize_bboxes(bboxes) -> list:
    if bboxes is None:
        return []
    if isinstance(bboxes, dict):
        return [bboxes]
    if bboxes and isinstance(bboxes[0], (list, tuple)):
        return list(bboxes[0])
    if bboxes:
        return list(bboxes)
    return []


def _coerce_bbox_norm(box, canvas_w, canvas_h):
    """Return (x0, y0, x1, y1) normalised 0..1 from a bbox dict/sequence.
    Accepts {x,y,width,height} / {x,y,w,h} / {x0,y0,x1,y1} / [x0,y0,x1,y1];
    pixel coords are divided by the canvas dims."""
    if isinstance(box, dict):
        if "x1" in box and "y1" in box:
            vals = [box.get("x", box.get("x0", 0)), box.get("y", box.get("y0", 0)),
                    box["x1"], box["y1"]]
        else:
            x = box.get("x", 0)
            y = box.get("y", 0)
            w = box.get("width", box.get("w", 0))
            h = box.get("height", box.get("h", 0))
            vals = [x, y, x + w, y + h]
    else:
        vals = list(box)[:4]
    x0, y0, x1, y1 = [float(v) for v in vals[:4]]
    if max(abs(x0), abs(y0), abs(x1), abs(y1)) > 1.0:
        x0, x1 = x0 / max(1, canvas_w), x1 / max(1, canvas_w)
        y0, y1 = y0 / max(1, canvas_h), y1 / max(1, canvas_h)
    if x1 < x0:
        x0, x1 = x1, x0
    if y1 < y0:
        y0, y1 = y1, y0
    return (max(0.0, x0), max(0.0, y0), min(1.0, x1), min(1.0, y1))


def _auto_split_norm(n: int, mode: str) -> list:
    """n equal strips as normalised (x0,y0,x1,y1)."""
    boxes = []
    if n <= 0:
        return boxes
    for i in range(n):
        if mode == "auto_horizontal":
            boxes.append((0.0, i / n, 1.0, (i + 1) / n))
        else:
            boxes.append((i / n, 0.0, (i + 1) / n, 1.0))
    return boxes


# ---------------------------------------------------------------------------
# LoRA loading + layer matching
# ---------------------------------------------------------------------------
def _norm_key(s):
    s = s.lower()
    for pre in ("lora_unet_", "lora_te_", "lora_", "diffusion_model.",
                "diffusion_model_", "transformer.", "model.diffusion_model.",
                "model.", "base_model."):
        if s.startswith(pre):
            s = s[len(pre):]
    return s.replace(".", "").replace("_", "")


# Krea 2 LoRAs ship in two naming conventions. ComfyUI-native ones use the
# real module paths (diffusion_model.blocks.0.attn.wq), which _norm_key already
# lines up with the live modules. Diffusers-style ones (ai-toolkit, DiffSynth,
# most civitai Krea2 LoRAs: transformer.transformer_blocks.0.attn.to_q) do not,
# so they matched 0 layers even though ComfyUI's own LoraLoader handles them -
# it translates through comfy.utils.krea2_to_diffusers. The tables below mirror
# that map so both conventions resolve to the same normalised signature.
_KREA2_DIFFUSERS_LEAF = {
    "attn.to_q": "attn.wq",
    "attn.to_k": "attn.wk",
    "attn.to_v": "attn.wv",
    "attn.to_gate": "attn.gate",
    "attn.to_out.0": "attn.wo",
    "attn.to_out": "attn.wo",   # some tools drop the ".0"
    "ff.gate": "mlp.gate",
    "ff.up": "mlp.up",
    "ff.down": "mlp.down",
}

_KREA2_DIFFUSERS_STEMS = {
    "transformer_blocks": "blocks",
    "text_fusion.layerwise_blocks": "txtfusion.layerwise_blocks",
    "text_fusion.refiner_blocks": "txtfusion.refiner_blocks",
}

_KREA2_DIFFUSERS_BASIC = {
    "img_in": "first",
    "time_embed.linear_1": "tmlp.0",
    "time_embed.linear_2": "tmlp.2",
    "time_mod_proj": "tproj.1",
    "txt_in.linear_1": "txtmlp.1",
    "txt_in.linear_2": "txtmlp.3",
    "text_fusion.projector": "txtfusion.projector",
    "final_layer.linear": "last.linear",
}


def _build_krea2_alias_map(max_blocks=128, max_txt_blocks=8):
    """{normalised diffusers sig: normalised ComfyUI sig}. Keyed on _norm_key,
    so dotted (diffusers) and underscored (kohya/lycoris) spellings of the same
    key both land on the same entry."""
    amap = {}
    for d, c in _KREA2_DIFFUSERS_BASIC.items():
        amap[_norm_key(d)] = _norm_key(c)
    for stem_d, stem_c in _KREA2_DIFFUSERS_STEMS.items():
        n = max_blocks if stem_d == "transformer_blocks" else max_txt_blocks
        for i in range(n):
            for d, c in _KREA2_DIFFUSERS_LEAF.items():
                amap[_norm_key("{}.{}.{}".format(stem_d, i, d))] = \
                    _norm_key("{}.{}.{}".format(stem_c, i, c))
    return amap


_KREA2_ALIASES = _build_krea2_alias_map()


def _module_sig(base):
    """Normalised signature for a LoRA module path, with diffusers-style Krea 2
    names translated onto their ComfyUI module names."""
    sig = _norm_key(base)
    return _KREA2_ALIASES.get(sig, sig)


def _load_lora_matrices(path):
    """{ module_sig: entry } in fp32 on CPU.
    LoRA entry: {'kind':'lora', 'down':T, 'up':T, 'scale':float}
      - kohya (lora_down/up + alpha) and diffusers (lora_A/B).
    LoKr entry: {'kind':'lokr', 'w1':T, 'w2':T, 'scale':float}
      - Kronecker factors (ai-toolkit / LyCORIS), direct or a@b decomposed.
        Full weight diff = kron(w1, w2); applied efficiently in the hook."""
    sd = safetensors.torch.load_file(path)
    groups = {}
    lokr_groups = {}
    alphas = {}
    for k, v in sd.items():
        if k.endswith(".alpha") or k.endswith("alpha"):
            base = re.sub(r"\.?alpha$", "", k)
            alphas[base] = float(v.flatten()[0].item())
            continue
        m = re.search(r"(.*?)\.(lora_down|lora_A)\.weight$", k)
        if m:
            groups.setdefault(m.group(1), {})["down"] = v.float()
            continue
        m = re.search(r"(.*?)\.(lora_up|lora_B)\.weight$", k)
        if m:
            groups.setdefault(m.group(1), {})["up"] = v.float()
            continue
        m = re.search(r"(.*?)\.(lokr_w1|lokr_w1_a|lokr_w1_b|lokr_w2|lokr_w2_a|lokr_w2_b|lokr_t2)$", k)
        if m:
            lokr_groups.setdefault(m.group(1), {})[m.group(2)] = v.float()
            continue

    out = {}
    n_aliased = 0
    for base, mats in groups.items():
        if "down" not in mats or "up" not in mats:
            continue
        down, up = mats["down"], mats["up"]
        rank = down.shape[0]
        alpha = alphas.get(base, alphas.get(base + ".alpha", float(rank)))
        sig = _module_sig(base)
        n_aliased += (sig != _norm_key(base))
        out[sig] = {
            "kind": "lora",
            "down": down,
            "up": up,
            "scale": float(alpha) / float(rank),
        }

    for base, mats in lokr_groups.items():
        if "lokr_t2" in mats:
            logging.warning("[Krea2RegionalMultiLoRA] '%s' uses tucker LoKr (conv); "
                            "skipping module %s.", path, base)
            continue
        # Rebuild each factor (direct tensor or a @ b decomposition). Alpha
        # scaling follows ComfyUI's LoKrAdapter: alpha/rank only when a
        # decomposed side exists, else 1.0.
        dim = None
        if "lokr_w1" in mats:
            w1 = mats["lokr_w1"]
        elif "lokr_w1_a" in mats and "lokr_w1_b" in mats:
            w1 = mats["lokr_w1_a"] @ mats["lokr_w1_b"]
            dim = mats["lokr_w1_b"].shape[0]
        else:
            continue
        if "lokr_w2" in mats:
            w2 = mats["lokr_w2"]
        elif "lokr_w2_a" in mats and "lokr_w2_b" in mats:
            w2 = mats["lokr_w2_a"] @ mats["lokr_w2_b"]
            dim = mats["lokr_w2_b"].shape[0]
        else:
            continue
        if w1.dim() != 2 or w2.dim() != 2:
            continue
        alpha = alphas.get(base, None)
        scale = (alpha / dim) if (alpha is not None and dim is not None) else 1.0
        sig = _module_sig(base)
        n_aliased += (sig != _norm_key(base))
        out[sig] = {
            "kind": "lokr",
            "w1": w1,
            "w2": w2,
            "scale": float(scale),
        }
    if n_aliased:
        logging.info("[Krea2RegionalMultiLoRA] '%s': translated %d/%d diffusers-style "
                     "keys to ComfyUI module names.", path, n_aliased, len(out))
    return out


def _iter_named_linears(module):
    for name, sub in module.named_modules():
        if isinstance(sub, torch.nn.Linear) or hasattr(sub, "weight"):
            yield name, sub


def _resolve_lora_path(name):
    p = folder_paths.get_full_path("loras", name)
    return p or name


# ---------------------------------------------------------------------------
# token-grid masks
# ---------------------------------------------------------------------------
def _rect_token_mask(rows, cols, nx0, ny0, nx1, ny1, feather):
    """Soft-edged rectangle (normalised coords) on the rows x cols token grid."""
    c0, c1 = nx0 * cols, nx1 * cols
    r0, r1 = ny0 * rows, ny1 * rows
    fc = max(1e-3, feather * cols)
    fr = max(1e-3, feather * rows)
    cc = torch.arange(cols, dtype=torch.float32).unsqueeze(0)
    rr = torch.arange(rows, dtype=torch.float32).unsqueeze(1)
    in_x = torch.sigmoid((cc - c0) / fc) * torch.sigmoid((c1 - cc) / fc)
    in_y = torch.sigmoid((rr - r0) / fr) * torch.sigmoid((r1 - rr) / fr)
    return (in_y * in_x).reshape(-1).clamp(0.0, 1.0)


# ---------------------------------------------------------------------------
# forward-time injection session
# ---------------------------------------------------------------------------
def _lokr_delta(xf, w1_d, w2_d):
    """Efficient kron(w1, w2) @ x without materializing the full weight.
    Mirrors ComfyUI's LoKrAdapter.h(): group input by w1's inner dim, apply
    w2 per group, then mix groups with w1."""
    uq = w1_d.shape[1]
    hg = xf.reshape(*xf.shape[:-1], uq, -1)          # [..., uq, in_n]
    hb = torch.nn.functional.linear(hg, w2_d)        # [..., uq, out_k]
    hc = torch.nn.functional.linear(hb.transpose(-1, -2), w1_d)  # [..., out_k, out_l]
    return hc.transpose(-1, -2).reshape(*xf.shape[:-1], -1)      # [..., out_l*out_k]


def _make_hook(session, entries):
    """entries = [(region_idx, prepared_entry), ...] for ONE Linear.
    LoRA:  out += mask_i * (x @ down_i.T @ up_i.T)   (scale folded into up_d)
    LoKr:  out += mask_i * (kron(w1_i, w2_i) @ x)    (scale folded into w1_d)"""
    def hook(module, inp, out):
        if not torch.is_tensor(out) or out.dim() < 2:
            return out
        x = inp[0]
        if not torch.is_tensor(x) or x.dim() < 2:
            return out
        seq = x.shape[-2]
        xf = x.to(_COMPUTE_DTYPE)
        res = None
        for ridx, d in entries:
            if d["kind"] == "lokr":
                delta = _lokr_delta(xf, d["w1_d"], d["w2_d"])
            else:
                delta = (xf @ d["down_d"].t()) @ d["up_d"].t()
            masked = session._full_mask(ridx, seq, out.dim()) * delta
            res = masked if res is None else res + masked
        if res is None:
            return out
        return out + res.to(out.dtype)
    return hook


class _RegionalSession:
    """Builds token masks at runtime from the real latent grid and installs
    forward hooks on every Linear any region's LoRA targets."""

    def __init__(self, patcher, region_loras, norm_boxes, seam_feather,
                 blend_override, canvas_w, canvas_h):
        self.patcher = patcher
        self.region_loras = region_loras      # [{sig: {down,up,scale}}] per region
        self.norm_boxes = norm_boxes          # [(x0,y0,x1,y1)] per region
        self.seam_feather = seam_feather
        self.blend_override = blend_override
        self.canvas_w, self.canvas_h = canvas_w, canvas_h
        self.n_img = 0
        self._txtlen = None
        self._layer_map = None
        self._prepared = False
        self._full_mask_cache = {}
        self._masks_d = []

    def _diffusion_model(self):
        m = self.patcher.model
        return getattr(m, "diffusion_model", m)

    def _build_layer_map(self, dm):
        layer_map = {}
        matched_per_region = [0] * len(self.region_loras)
        for name, mod in _iter_named_linears(dm):
            sig = _norm_key(name)
            entries = []
            for ridx, lora in enumerate(self.region_loras):
                d = lora.get(sig)
                if d is not None:
                    entries.append((ridx, d))
                    matched_per_region[ridx] += 1
            if entries:
                layer_map[name] = (mod, entries)
        for ridx, count in enumerate(matched_per_region):
            targets = len(self.region_loras[ridx])
            logging.info("[Krea2RegionalMultiLoRA] region %d: matched %d/%d LoRA layers.",
                         ridx, count, targets)
            if count == 0 and targets > 0:
                logging.warning("[Krea2RegionalMultiLoRA] region %d matched 0 layers - "
                                "LoRA key format may not map onto this model. "
                                "LoRA sigs e.g. %s | model sigs e.g. %s", ridx,
                                sorted(self.region_loras[ridx])[:3],
                                [_norm_key(n) for n, _ in _iter_named_linears(dm)][:3])
        return layer_map

    def _infer_device(self, dm, args):
        x0 = args[0] if args else None
        if torch.is_tensor(x0):
            return x0.device
        try:
            return next(dm.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    def _resolve_grid(self, x):
        """Token grid from the runtime latent [B,C,H,W]; Krea2 patch size = 2."""
        if torch.is_tensor(x) and x.dim() >= 4:
            H, W = int(x.shape[-2]), int(x.shape[-1])
            rows, cols = H // 2, W // 2
            if rows > 0 and cols > 0:
                return rows, cols, "latent"
        rows = max(1, self.canvas_h // 16)
        cols = max(1, self.canvas_w // 16)
        return rows, cols, "canvas-fallback"

    def _build_masks_now(self, rows, cols):
        n_regions = len(self.norm_boxes)
        masks = []
        for (x0, y0, x1, y1) in self.norm_boxes:
            masks.append(_rect_token_mask(rows, cols, x0, y0, x1, y1, self.seam_feather))
        blend = float(max(0.0, min(1.0, self.blend_override)))
        if blend > 0.0 and n_regions > 0:
            uniform = 1.0 / n_regions
            masks = [(1.0 - blend) * m + blend * uniform for m in masks]
        return masks

    def _prepare(self, dev, x):
        cdt = _COMPUTE_DTYPE
        self._dev = dev
        for name, (mod, entries) in self._layer_map.items():
            for ridx, d in entries:
                if "down_d" in d or "w1_d" in d:
                    continue
                if d["kind"] == "lokr":
                    # delta is linear in w1, so fold scale*strength into it
                    d["w1_d"] = d["w1"].to(dev, cdt) * d["scale"]
                    d["w2_d"] = d["w2"].to(dev, cdt)
                else:
                    d["down_d"] = d["down"].to(dev, cdt)
                    d["up_d"] = d["up"].to(dev, cdt) * d["scale"]
        rows, cols, src = self._resolve_grid(x)
        self.n_img = rows * cols
        masks = self._build_masks_now(rows, cols)
        self._masks_d = [m.to(dev, cdt) for m in masks]
        self._full_mask_cache = {}
        self._grid_info = (rows, cols, src)
        self._prepared = True

    def _full_mask(self, ridx, seq, ndim):
        """Full-sequence mask: zeros over the text prefix, region mask over the
        image-token block. Krea2's combined sequence is [text | image (| pad)];
        we place the mask at [txtlen : txtlen + n_img] when the text length is
        known, and fall back to the trailing block otherwise (which is exact on
        the current ComfyUI port, where the sequence is not padded)."""
        key = (ridx, seq, ndim, self._txtlen)
        fm = self._full_mask_cache.get(key)
        if fm is None:
            mv = self._masks_d[ridx]
            base = torch.zeros(seq, device=self._dev, dtype=_COMPUTE_DTYPE)
            n_img = self.n_img
            if n_img <= 0 or n_img > seq:
                base[:] = mv.mean()
            else:
                start = seq - n_img  # trailing block (correct when unpadded)
                if self._txtlen is not None and 0 <= self._txtlen <= seq - n_img:
                    start = self._txtlen  # exact image span, padding-safe
                base[start:start + n_img] = mv
            fm = base.view(*([1] * (ndim - 2)), seq, 1)
            self._full_mask_cache[key] = fm
        return fm

    def _extract_txtlen(self, args, kwargs):
        """Text-token count from the diffusion model's `context` arg.
        Krea2's forward is (x, timesteps, context, ...); context is
        (B, txt_seq, features). Returns None if it can't be identified."""
        ctx = None
        if len(args) >= 3 and torch.is_tensor(args[2]) and args[2].dim() == 3:
            ctx = args[2]
        elif torch.is_tensor(kwargs.get("context")) and kwargs["context"].dim() == 3:
            ctx = kwargs["context"]
        if ctx is not None:
            return int(ctx.shape[1])
        return None

    def run(self, executor, *args, **kwargs):
        dm = self._diffusion_model()
        if self._layer_map is None:
            self._layer_map = self._build_layer_map(dm)
        self._txtlen = self._extract_txtlen(args, kwargs)
        if not self._prepared:
            dev = self._infer_device(dm, args)
            x0 = args[0] if args else None
            self._prepare(dev, x0)
            rows, cols, src = self._grid_info
            shp = tuple(x0.shape) if torch.is_tensor(x0) else None
            logging.info("[Krea2RegionalMultiLoRA] prepared on %s | latent=%s "
                         "grid=%dx%d (%s) n_img=%d regions=%d",
                         dev, shp, rows, cols, src, self.n_img, len(self._masks_d))
        handles = []
        try:
            for name, (mod, entries) in self._layer_map.items():
                handles.append(mod.register_forward_hook(_make_hook(self, entries)))
            return executor(*args, **kwargs)
        finally:
            for h in handles:
                h.remove()


# ---------------------------------------------------------------------------
# the node
# ---------------------------------------------------------------------------
class Krea2RegionalMultiLoRA:
    """Regional multi-LoRA for Krea 2 via masked activation-delta injection."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "canvas_width": ("INT", {
                    "default": 1024, "min": 64, "max": 16384, "step": 16,
                    "tooltip": "Pixel-space canvas width (used to interpret pixel bboxes).",
                }),
                "canvas_height": ("INT", {
                    "default": 1024, "min": 64, "max": 16384, "step": 16,
                    "tooltip": "Pixel-space canvas height.",
                }),
                "regions_json": ("STRING", {
                    "multiline": True,
                    "default": DEFAULT_REGIONS_JSON,
                    "tooltip": (
                        "JSON array of regions (one per character), in canvas order. "
                        "The 'Add Region' / 'Remove' buttons edit this for you. "
                        'Each: {"lora": "file.safetensors", "strength": 1.1, "enable": true}. '
                        "Region i maps to bounding-box i when split_mode=bbox."
                    ),
                }),
                "split_mode": (["bbox", "auto_vertical", "auto_horizontal"], {
                    "default": "bbox",
                    "tooltip": (
                        "bbox = use the wired bounding boxes (region i -> box i). "
                        "auto_vertical / auto_horizontal = split the canvas into N "
                        "equal strips (N = number of enabled regions)."
                    ),
                }),
                "seam_feather": ("FLOAT", {
                    "default": 0.08, "min": 0.0, "max": 0.5, "step": 0.01,
                    "tooltip": (
                        "Region-edge softness as a fraction of the token grid. "
                        "0 = hard cut. Higher = smoother seam, more identity bleed."
                    ),
                }),
                "blend_override": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": (
                        "0 = clean regional split (recommended). Raising it mixes every "
                        "LoRA uniformly across the whole image; identities collapse past ~0.8."
                    ),
                }),
            },
            "optional": {
                "bboxes": ("BOUNDING_BOX", {
                    "tooltip": "Bounding boxes from a box builder (e.g. Ideogram4PromptBuilderKJ). Used when split_mode=bbox.",
                }),
                "base_strength": ("FLOAT", {
                    "default": 1.0, "min": -10.0, "max": 10.0, "step": 0.05,
                    "tooltip": "Global multiplier applied to every region's strength.",
                }),
                "include_background": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Emit a '__background__' mask in the masks output (debug/preview only).",
                }),
            },
        }

    RETURN_TYPES = ("MODEL", "CLIP", "KREA2_MASKS", "KREA2_DATA")
    RETURN_NAMES = ("model", "clip", "masks", "data")
    FUNCTION = "apply"
    CATEGORY = "Krea2/By Fedor"

    DESCRIPTION = (
        "Krea2 Regional Multi-LoRA (By Fedor). Each region's LoRA activation delta "
        "is injected at forward time ONLY into the image tokens inside its bounding "
        "box, so LoRAs physically cannot act outside their region. Add as many "
        "regions as you want. Wire model in -> model out to KSampler."
    )

    def apply(
        self,
        model,
        clip,
        canvas_width,
        canvas_height,
        regions_json,
        split_mode,
        seam_feather,
        blend_override,
        bboxes=None,
        base_strength=1.0,
        include_background=True,
    ):
        regions = _parse_regions(regions_json)
        enabled = [
            r for r in regions
            if r["enable"] and r["lora"] not in ("None", "") and (r["strength"] * base_strength) != 0.0
        ]

        empty_masks = {"masks": {}, "similarity_maps": {}, "text_mask_value": float(blend_override)}
        if not enabled:
            logging.warning("[Krea2RegionalMultiLoRA] No enabled regions with a LoRA; passing model through.")
            return (model, clip, empty_masks, {"adapters": []})

        cw, ch = int(canvas_width), int(canvas_height)

        # Resolve one normalised box per enabled region.
        if split_mode == "bbox":
            frame = _normalize_bboxes(bboxes)
            if frame:
                norm_boxes = []
                for i in range(len(enabled)):
                    if i < len(frame):
                        norm_boxes.append(_coerce_bbox_norm(frame[i], cw, ch))
                    else:
                        logging.warning("[Krea2RegionalMultiLoRA] region %d has no bbox; "
                                        "using full canvas.", i)
                        norm_boxes.append((0.0, 0.0, 1.0, 1.0))
            else:
                logging.warning("[Krea2RegionalMultiLoRA] split_mode=bbox but no bboxes wired; "
                                "falling back to auto_vertical.")
                norm_boxes = _auto_split_norm(len(enabled), "auto_vertical")
        else:
            norm_boxes = _auto_split_norm(len(enabled), split_mode)

        # Load each region's LoRA matrices (cached per file).
        file_cache = {}
        region_loras = []
        strength_eff = []
        for r in enabled:
            path = _resolve_lora_path(r["lora"])
            if path not in file_cache:
                file_cache[path] = _load_lora_matrices(path)
            base_mats = file_cache[path]
            s = r["strength"] * float(base_strength)
            strength_eff.append(s)
            # per-region shallow copy with per-region scale (so the same file can
            # be used at different strengths in different regions)
            mats = {
                sig: {**{k: v for k, v in d.items() if k != "scale"},
                      "scale": d["scale"] * s}
                for sig, d in base_mats.items()
            }
            if not mats:
                logging.warning("[Krea2RegionalMultiLoRA] '%s' contains no LoRA (A/B) or "
                                "LoKr (kron factor) pairs - raw-diff files belong in a "
                                "normal LoraLoader, not here.", r["lora"])
            region_loras.append(mats)

        patched = model.clone()
        session = _RegionalSession(
            patched, region_loras, norm_boxes,
            float(seam_feather), float(blend_override), cw, ch,
        )

        def wrapper(executor, *args, **kwargs):
            return session.run(executor, *args, **kwargs)

        if hasattr(patched, "add_wrapper_with_key"):
            patched.add_wrapper_with_key(_WRAPPER_ENUM, WRAPPER_KEY, wrapper)
        elif hasattr(patched, "add_wrapper"):
            patched.add_wrapper(_WRAPPER_ENUM, wrapper)
        else:
            raise RuntimeError("This ComfyUI build lacks model wrapper support. Update ComfyUI.")

        # Debug/preview masks output (latent-res 2D).
        latent_w = max(4, int(math.ceil(cw / 16)))
        latent_h = max(4, int(math.ceil(ch / 16)))
        masks_2d = {}
        for i, (r, (x0, y0, x1, y1)) in enumerate(zip(enabled, norm_boxes)):
            m = _rect_token_mask(latent_h, latent_w, x0, y0, x1, y1, float(seam_feather))
            masks_2d[r["name"]] = m.reshape(latent_h, latent_w)
        if include_background and masks_2d:
            union = torch.zeros(latent_h, latent_w)
            for m in masks_2d.values():
                union = torch.maximum(union, m)
            masks_2d["__background__"] = (1.0 - union).clamp(0.0, 1.0)

        masks_payload = {
            "masks": masks_2d,
            "similarity_maps": {},
            "text_mask_value": float(max(0.0, min(1.0, blend_override))),
        }
        node_data = {
            "adapters": [
                {"name": r["name"], "lora": r["lora"], "strength": s}
                for r, s in zip(enabled, strength_eff)
            ],
            "model_type": "krea2",
            "engine": "activation_delta",
        }

        logging.info(
            "[Krea2RegionalMultiLoRA] armed: %d regions, split=%s, feather=%.2f, "
            "blend=%.2f (masks are built at runtime from the real latent).",
            len(enabled), split_mode, seam_feather, blend_override,
        )
        return (patched, clip, masks_payload, node_data)


NODE_CLASS_MAPPINGS = {
    "Krea2RegionalMultiLoRA": Krea2RegionalMultiLoRA,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Krea2RegionalMultiLoRA": "Krea2 Regional Multi-LoRA (By Fedor)",
}

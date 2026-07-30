"""
Krea2RegionalMultiLoRAV3 (By Fedor) - ONE node: regional multi-LoRA masking
PLUS per-region reference-image "latent mold" guidance.

v3 = v1's masked activation-delta LoRA injection + v2's Reference Lock,
unified. Each region row can carry:
  * a LoRA (masked to its box at forward time - the v1 hard guarantee), and/or
  * a reference image (the "mold"): its VAE latent steers the box's denoised
    latent every step over a scheduled window:
        denoised[box] += ref_strength * mask * (mold - denoised[box])

The reference image is picked per row via a "load ref" button in the node UI
(web/krea2_regional_multilora_v3.js) which uploads to ComfyUI's input folder
and shows an inline thumbnail. The filename is stored in regions_json as
"ref_image", so workflows round-trip through save/load and the API.

regions_json schema (v3):
    [
      {"lora": "character_A.safetensors", "strength": 1.1, "enable": true,
       "ref_image": "charA_ref.png"},
      {"lora": "None", "strength": 1.1, "enable": true,
       "ref_image": "prop.png"}          # ref-only region: no LoRA, still molded
    ]

v1 (Krea2RegionalMultiLoRA) and v2 (Krea2ReferenceLock*) are untouched; this
is a separate node class.
"""

import json
import logging
import math

import numpy as np
import torch
from PIL import Image, ImageOps

import folder_paths

from .krea2_regional_multilora import (
    _auto_split_norm,
    _coerce_bbox_norm,
    _load_lora_matrices,
    _normalize_bboxes,
    _parse_regions,
    _rect_token_mask,
    _resolve_lora_path,
    _RegionalSession,
    _WRAPPER_ENUM,
)
from .krea2_reference_lock import (
    _build_mold,
    _encode_reference,
    _in_window,
    _sigma_window,
)

WRAPPER_KEY_V3 = "krea2_regional_multilora_v3"

DEFAULT_REGIONS_JSON_V3 = (
    "[\n"
    '  {"lora": "None", "strength": 1.1, "enable": true, "ref_image": ""},\n'
    '  {"lora": "None", "strength": 1.1, "enable": true, "ref_image": ""}\n'
    "]"
)


def _parse_regions_v3(regions_json: str) -> list:
    """v1 parsing + the per-row ref_image field."""
    base = _parse_regions(regions_json)
    try:
        raw = json.loads(regions_json)
        if isinstance(raw, dict):
            raw = [raw]
    except (ValueError, TypeError):
        raw = []
    raw_items = [x for x in raw if isinstance(x, dict)] if isinstance(raw, list) else []
    for i, r in enumerate(base):
        ref = ""
        ref_enable = True
        if i < len(raw_items):
            ref = str(raw_items[i].get("ref_image", "") or "").strip()
            ref_enable = bool(raw_items[i].get("ref_enable", True))
        r["ref_image"] = ref
        r["ref_enable"] = ref_enable
    return base


def _load_ref_image_tensor(name):
    """Input-folder image -> ComfyUI IMAGE tensor [1,H,W,3] float 0..1."""
    path = folder_paths.get_annotated_filepath(name)
    img = Image.open(path)
    img = ImageOps.exif_transpose(img).convert("RGB")
    arr = np.asarray(img).astype(np.float32) / 255.0
    return torch.from_numpy(arr)[None]


class Krea2RegionalMultiLoRAV3:
    """Regional multi-LoRA + per-region reference-image mold guidance, one node."""

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
                    "default": DEFAULT_REGIONS_JSON_V3,
                    "tooltip": (
                        "JSON array of regions, in box order. The row buttons edit this "
                        'for you. Each: {"lora": "file.safetensors", "strength": 1.1, '
                        '"enable": true, "ref_image": "uploaded.png"}. '
                        "ref_image is set by the per-row 'load ref' button."
                    ),
                }),
                "split_mode": (["bbox", "auto_vertical", "auto_horizontal"], {
                    "default": "bbox",
                    "tooltip": "bbox = wired boxes (region i -> box i); autos = equal strips.",
                }),
                "seam_feather": ("FLOAT", {
                    "default": 0.08, "min": 0.0, "max": 0.5, "step": 0.01,
                    "tooltip": "LoRA region edge softness (fraction of token grid).",
                }),
                "blend_override": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "0 = clean regional split (recommended).",
                }),
                "ref_strength": ("FLOAT", {
                    "default": 0.30, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": (
                        "Per-step pull toward each region's reference image. "
                        "0.2-0.4 anchors while integrating; 0.7+ approaches a paste. "
                        "0 disables reference guidance entirely."
                    ),
                }),
                "ref_start_percent": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Start of the reference guidance window.",
                }),
                "ref_end_percent": ("FLOAT", {
                    "default": 0.60, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": (
                        "End of the reference guidance window. ~0.5-0.7 locks identity "
                        "early, then lets the model blend seams/lighting."
                    ),
                }),
                "ref_feather": ("FLOAT", {
                    "default": 0.06, "min": 0.0, "max": 0.5, "step": 0.01,
                    "tooltip": "Soft edge of the reference guidance mask.",
                }),
            },
            "optional": {
                "bboxes": ("BOUNDING_BOX", {
                    "forceInput": True,  # keep socket-only; frontend widget group corrupts saves
                    "tooltip": "Boxes from a box builder (e.g. Ideogram4PromptBuilderKJ).",
                }),
                "vae": ("VAE", {
                    "tooltip": "Required for reference images (encodes each ref to its latent mold).",
                }),
                "base_strength": ("FLOAT", {
                    "default": 1.0, "min": -10.0, "max": 10.0, "step": 0.05,
                    "tooltip": "Global multiplier applied to every region's LoRA strength.",
                }),
                "include_background": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Emit a '__background__' mask in the masks output (debug only).",
                }),
            },
        }

    RETURN_TYPES = ("MODEL", "CLIP", "KREA2_MASKS", "KREA2_DATA")
    RETURN_NAMES = ("model", "clip", "masks", "data")
    FUNCTION = "apply"
    CATEGORY = "Krea2/By Fedor"

    DESCRIPTION = (
        "Krea2 Regional Multi-LoRA v3 (By Fedor). One node: per-box LoRA masking "
        "(activation-delta injection, hard spatial guarantee) PLUS per-box reference-"
        "image guidance (latent mold steering over a scheduled window). Use the "
        "'load ref' button on each region row to attach a reference image; a "
        "thumbnail shows which image each LoRA references. Wire a VAE to enable refs."
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
        ref_strength,
        ref_start_percent,
        ref_end_percent,
        ref_feather,
        bboxes=None,
        vae=None,
        base_strength=1.0,
        include_background=True,
    ):
        regions = _parse_regions_v3(regions_json)

        def has_lora(r):
            return r["lora"] not in ("None", "") and (r["strength"] * base_strength) != 0.0

        def has_ref(r):
            # A reference only counts if it's loaded AND its per-row toggle is on.
            return bool(r.get("ref_image")) and r.get("ref_enable", True)

        # A row is active if it contributes a LoRA and/or a reference image.
        # Active rows claim boxes in order (row i -> box i among active rows).
        active = [r for r in regions if r["enable"] and (has_lora(r) or has_ref(r))]

        empty_masks = {"masks": {}, "similarity_maps": {}, "text_mask_value": float(blend_override)}
        if not active:
            logging.warning("[Krea2RegionalMultiLoRAV3] No active regions; passing model through.")
            return (model, clip, empty_masks, {"adapters": []})

        cw, ch = int(canvas_width), int(canvas_height)

        # One normalised box per active region (identical to v1 resolution).
        if split_mode == "bbox":
            frame = _normalize_bboxes(bboxes)
            if frame:
                norm_boxes = []
                for i in range(len(active)):
                    if i < len(frame):
                        norm_boxes.append(_coerce_bbox_norm(frame[i], cw, ch))
                    else:
                        logging.warning("[Krea2RegionalMultiLoRAV3] region %d has no bbox; "
                                        "using full canvas.", i)
                        norm_boxes.append((0.0, 0.0, 1.0, 1.0))
            else:
                logging.warning("[Krea2RegionalMultiLoRAV3] split_mode=bbox but no bboxes wired; "
                                "falling back to auto_vertical.")
                norm_boxes = _auto_split_norm(len(active), "auto_vertical")
        else:
            norm_boxes = _auto_split_norm(len(active), split_mode)

        # ------------------------------------------------------------------
        # LoRA path (v1 engine, unchanged mechanism)
        # ------------------------------------------------------------------
        file_cache = {}
        region_loras = []
        strength_eff = []
        for r in active:
            if not has_lora(r):
                region_loras.append({})   # ref-only row: holds its box, no LoRA
                strength_eff.append(0.0)
                continue
            path = _resolve_lora_path(r["lora"])
            if path not in file_cache:
                file_cache[path] = _load_lora_matrices(path)
            base_mats = file_cache[path]
            s = r["strength"] * float(base_strength)
            strength_eff.append(s)
            mats = {
                sig: {**{k: v for k, v in d.items() if k != "scale"},
                      "scale": d["scale"] * s}
                for sig, d in base_mats.items()
            }
            if not mats:
                logging.warning("[Krea2RegionalMultiLoRAV3] '%s' contains no LoRA (A/B) "
                                "or LoKr (kron factor) pairs.", r["lora"])
            region_loras.append(mats)

        patched = model.clone()
        if any(region_loras):
            session = _RegionalSession(
                patched, region_loras, norm_boxes,
                float(seam_feather), float(blend_override), cw, ch,
            )

            def wrapper(executor, *args, **kwargs):
                return session.run(executor, *args, **kwargs)

            if hasattr(patched, "add_wrapper_with_key"):
                patched.add_wrapper_with_key(_WRAPPER_ENUM, WRAPPER_KEY_V3, wrapper)
            elif hasattr(patched, "add_wrapper"):
                patched.add_wrapper(_WRAPPER_ENUM, wrapper)
            else:
                raise RuntimeError("This ComfyUI build lacks model wrapper support. Update ComfyUI.")

        # ------------------------------------------------------------------
        # Reference path (v2 latent-mold engine, per active row with ref_image)
        # ------------------------------------------------------------------
        ref_entries = []  # (box_norm, ref_model_space, row_name)
        n_refs_wanted = sum(1 for r in active if has_ref(r))
        if n_refs_wanted and vae is None:
            logging.warning("[Krea2RegionalMultiLoRAV3] %d region(s) have reference images "
                            "but no VAE is wired; reference guidance skipped.", n_refs_wanted)
        elif n_refs_wanted and float(ref_strength) > 0.0 \
                and float(ref_end_percent) > float(ref_start_percent):
            for i, r in enumerate(active):
                if not has_ref(r):
                    continue
                try:
                    img = _load_ref_image_tensor(r["ref_image"])
                except Exception as e:
                    logging.warning("[Krea2RegionalMultiLoRAV3] could not load ref '%s' "
                                    "for region %d: %s", r["ref_image"], i, e)
                    continue
                box = norm_boxes[i]
                if box[2] - box[0] < 1e-3 or box[3] - box[1] < 1e-3:
                    logging.warning("[Krea2RegionalMultiLoRAV3] region %d degenerate box; "
                                    "ref skipped.", i)
                    continue
                ref_entries.append((box, _encode_reference(patched, vae, img), r["name"]))

        if ref_entries:
            sigma_start, sigma_end = _sigma_window(patched, ref_start_percent, ref_end_percent)
            w, fth = float(ref_strength), float(ref_feather)
            state = {"key": None, "built": [], "logged": False}

            def post_cfg(args):
                denoised = args["denoised"]
                if denoised.dim() != 4 or not _in_window(args["sigma"], sigma_start, sigma_end):
                    return denoised
                C, H, W = denoised.shape[1], denoised.shape[2], denoised.shape[3]
                if state["key"] != (C, H, W):
                    built = []
                    for box, ref_ms, _name in ref_entries:
                        mm = _build_mold(ref_ms, box, C, H, W, fth, denoised.device)
                        if mm is not None:
                            built.append(mm)
                    state["built"] = built
                    state["key"] = (C, H, W)
                    if not state["logged"]:
                        logging.info("[Krea2RegionalMultiLoRAV3] %d mold(s) armed: latent %dx%d "
                                     "window sigma %.4f->%.4f ref_strength %.2f",
                                     len(built), W, H, sigma_start, sigma_end, w)
                        state["logged"] = True
                if not state["built"]:
                    return denoised
                d32 = denoised.float()
                for mold, mask in state["built"]:
                    d32 = d32 + (w * mask) * (mold - d32)
                return d32.to(denoised.dtype)

            patched.set_model_sampler_post_cfg_function(post_cfg)

        # ------------------------------------------------------------------
        # Debug/preview masks + data outputs
        # ------------------------------------------------------------------
        latent_w = max(4, int(math.ceil(cw / 16)))
        latent_h = max(4, int(math.ceil(ch / 16)))
        masks_2d = {}
        for r, (x0, y0, x1, y1) in zip(active, norm_boxes):
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
                {"name": r["name"], "lora": r["lora"], "strength": s,
                 "ref_image": r.get("ref_image", ""),
                 "ref_enable": r.get("ref_enable", True)}
                for r, s in zip(active, strength_eff)
            ],
            "model_type": "krea2",
            "engine": "activation_delta+latent_mold",
        }

        logging.info(
            "[Krea2RegionalMultiLoRAV3] armed: %d regions (%d with LoRA, %d with ref), "
            "split=%s, feather=%.2f, ref_strength=%.2f window %.2f-%.2f",
            len(active), sum(1 for r in active if has_lora(r)), len(ref_entries),
            split_mode, seam_feather, ref_strength, ref_start_percent, ref_end_percent,
        )
        return (patched, clip, masks_payload, node_data)


NODE_CLASS_MAPPINGS = {
    "Krea2RegionalMultiLoRAV3": Krea2RegionalMultiLoRAV3,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Krea2RegionalMultiLoRAV3": "Krea2 Regional Multi-LoRA v3 + Ref Lock (By Fedor)",
}

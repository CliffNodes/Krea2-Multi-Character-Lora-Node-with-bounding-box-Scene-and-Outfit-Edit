"""Krea2RegionalDetailer -- per-region face/body refinement pass.

Distant subjects only get a handful of latent pixels, so even a perfectly
routed LoRA cannot express full identity detail. This node runs AFTER the
main generation: for each region in the V12 detail plan it crops the
subject's box (with padding), upscales the crop, re-renders it img2img with
that region's character LoRA patched into the whole model (a crop contains
only its own subject, so no regional masking is needed), then pastes the
result back with a feathered seam. An optional second pass does the same on
the detected face inside the refined crop for maximum identity fidelity.

Cost scales with subject count and crop size, not canvas size; regions whose
box is already large can be skipped via skip_above_px.
"""
from __future__ import annotations

import logging
import math

import torch

import comfy.samplers
import comfy.sd
import comfy.utils
from nodes import common_ksampler

from .krea2_regional_multilora import _resolve_lora_path


def _align16(value, minimum=64):
    return max(minimum, int(round(value / 16.0)) * 16)


def _encode_text(clip, text):
    tokens = clip.tokenize(text or "")
    return clip.encode_from_tokens_scheduled(tokens)


def _clean_label(label):
    import re

    text = re.sub(
        r"\bone single (man|woman|person) only\b", r"a \1", label or "",
        flags=re.I,
    ).strip().rstrip(".")
    return text or "a person"


_FACE_MODEL = None


def _detect_faces(image, conf_min=0.45):
    """All faces [(x0, y0, x1, y1), ...] in a [H,W,C] 0..1 tensor.

    Uses face_yolov8m (the ADetailer face model). Detection runs on the FULL
    final image, not per-box crops: subjects regularly drift out of their
    planned boxes (couple poses squeeze the pair together), and the whole
    point of the face pass is to restore identity wherever the face actually
    rendered.
    """
    global _FACE_MODEL
    try:
        import os

        import folder_paths
        from ultralytics import YOLO

        if _FACE_MODEL is None:
            path = os.path.join(
                folder_paths.models_dir, "ultralytics", "bbox",
                "face_yolov8m.pt",
            )
            if not os.path.exists(path):
                logging.info("[Krea2Detailer] face_yolov8m.pt not found")
                return []
            _FACE_MODEL = YOLO(path)
        # ::-1 alone yields a negative-stride view that cv2.resize rejects
        bgr = (image * 255.0).clamp(0, 255).byte().cpu().numpy()[:, :, ::-1].copy()
        boxes = _FACE_MODEL(bgr, verbose=False)[0].boxes
        if boxes is None or len(boxes) == 0:
            return []
        faces = []
        for xyxy, conf in zip(boxes.xyxy, boxes.conf):
            if float(conf) >= conf_min:
                faces.append(tuple(int(v) for v in xyxy.tolist()))
        return faces
    except Exception:  # detector must never kill the render
        logging.exception(
            "[Krea2Detailer] face detect failed (image shape %s)",
            tuple(image.shape),
        )
        return []


def _assign_faces(faces, plan, width, height):
    """One-to-one greedy assignment of detected faces to plan regions.

    Nearest-box-per-face is NOT enough: when subjects drift sideways, two
    faces can both sit closest to the same box (observed in run 00049, where
    the pair rendered right of both boxes and the man's face was nearest the
    woman's box). Globally sorting all pairs and taking each face and each
    region at most once keeps the relative arrangement correct.
    """
    pairs = []
    for f_idx, (fx0, fy0, fx1, fy1) in enumerate(faces):
        fcx, fcy = (fx0 + fx1) / 2.0, (fy0 + fy1) / 2.0
        for r_idx, entry in enumerate(plan):
            x, y, w, h = entry["box"]
            bcx, bcy = (x + 0.5 * w) * width, (y + 0.5 * h) * height
            dist = (fcx - bcx) ** 2 + (fcy - bcy) ** 2
            pairs.append((dist, f_idx, r_idx))
    pairs.sort()
    used_faces, used_regions, assigned = set(), set(), []
    for _dist, f_idx, r_idx in pairs:
        if f_idx in used_faces or r_idx in used_regions:
            continue
        used_faces.add(f_idx)
        used_regions.add(r_idx)
        assigned.append((r_idx, faces[f_idx]))
    return assigned


def _paste_mask(h, w, feather, edges, device, dtype):
    """[h,w] smoothstep mask; ramps only on edges not at the image border."""
    top, left, bottom, right = edges
    ys = torch.arange(h, device=device, dtype=torch.float32)
    xs = torch.arange(w, device=device, dtype=torch.float32)
    fy = max(1.0, float(feather))
    fx = max(1.0, float(feather))
    my = torch.ones(h, device=device)
    mx = torch.ones(w, device=device)
    if top:
        my = torch.minimum(my, (ys / fy).clamp(0, 1))
    if bottom:
        my = torch.minimum(my, ((h - 1 - ys) / fy).clamp(0, 1))
    if left:
        mx = torch.minimum(mx, (xs / fx).clamp(0, 1))
    if right:
        mx = torch.minimum(mx, ((w - 1 - xs) / fx).clamp(0, 1))
    mask = my.unsqueeze(1) * mx.unsqueeze(0)
    mask = mask * mask * (3.0 - 2.0 * mask)  # smoothstep
    return mask.unsqueeze(-1).to(dtype)


class Krea2RegionalDetailer:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {
                    "tooltip": "The decoded image from your main generation "
                               "(VAEDecode output).",
                }),
                "model": ("MODEL", {
                    "tooltip": "BASE model (loader / Power Lora Loader output), "
                               "NOT the V12 node's wrapped model output.",
                }),
                "clip": ("CLIP", {
                    "tooltip": "Text encoder for the refinement prompts.",
                }),
                "vae": ("VAE", {
                    "tooltip": "VAE for the img2img encode/decode of each crop.",
                }),
                "data": ("KREA2_DATA", {
                    "tooltip": "data output of the V12 node (carries the "
                               "detail plan: box + LoRA per region).",
                }),
                "enable": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Off = image passes through untouched (quick "
                               "way to A/B the detailer's effect).",
                }),
                "body_denoise": ("FLOAT", {
                    "default": 0.30, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "img2img denoise for the body-crop pass. "
                               "0 disables the body pass.",
                }),
                "face_denoise": ("FLOAT", {
                    "default": 0.40, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "img2img denoise for the face pass. "
                               "0 disables the face pass.",
                }),
                "steps": ("INT", {
                    "default": 8, "min": 1, "max": 40,
                    "tooltip": "Sampling steps per refinement pass. 8 is "
                               "plenty for turbo-style models.",
                }),
                "cfg": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 20.0, "step": 0.1,
                    "tooltip": "CFG for refinement passes. Keep at 1.0 for "
                               "turbo/distilled models.",
                }),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS, {
                    "default": "euler",
                    "tooltip": "Sampler for refinement passes; match your "
                               "main KSampler.",
                }),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS, {
                    "default": "simple",
                    "tooltip": "Scheduler for refinement passes; match your "
                               "main KSampler.",
                }),
                "seed": ("INT", {
                    "default": 0, "min": 0, "max": 0xffffffffffffffff,
                    "tooltip": "Base seed for refinement passes (each region "
                               "offsets it deterministically).",
                }),
                "target_px": ("INT", {
                    "default": 1024, "min": 256, "max": 2048, "step": 64,
                    "tooltip": "Crop is upscaled so its short side is about "
                               "this before re-rendering.",
                }),
                "crop_pad": ("FLOAT", {
                    "default": 0.25, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Extra context around the box (fraction of box "
                               "size) included in the crop.",
                }),
                "lora_scale": ("FLOAT", {
                    "default": 0.70, "min": 0.0, "max": 2.0, "step": 0.05,
                    "tooltip": "Multiplies each region's strength for this "
                               "pass. Regional strengths (1.3-1.6) are tuned "
                               "for delta injection; a standard full-model "
                               "patch runs hotter, so scale down (~0.7).",
                }),
                "skip_above_px": ("INT", {
                    "default": 0, "min": 0, "max": 8192, "step": 32,
                    "tooltip": "Skip subjects whose box is already taller "
                               "than this many pixels (0 = refine all).",
                }),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("image", "face_preview")
    FUNCTION = "refine"
    CATEGORY = "Krea2/By Fedor"
    DESCRIPTION = (
        "Face + body refinement for Krea2 regional generations. Re-renders "
        "each subject's crop with its own character LoRA at high resolution "
        "and pastes it back seamlessly. Wire data from the V12 node."
    )

    def refine(self, image, model, clip, vae, data, enable, body_denoise,
               face_denoise, steps, cfg, sampler_name, scheduler, seed,
               target_px, crop_pad, lora_scale, skip_above_px):
        blank = torch.zeros(1, 64, 64, 3)
        plan = (data or {}).get("detail_plan") if isinstance(data, dict) else None
        plan = [
            p for p in (plan or [])
            if p.get("lora") not in ("", "None", None)
            and float(p.get("strength", 0.0)) != 0.0
        ]
        if not enable or not plan:
            if enable:
                logging.info("[Krea2Detailer] no detail plan; passthrough")
            return (image, blank)

        out = image.clone()
        batch, height, width, _ = out.shape
        lora_cache = {}
        face_previews = []

        for b in range(batch):
            # ---- pass 1: body refinement inside each PLANNED box ----------
            if body_denoise > 0.0:
                for ridx, entry in enumerate(plan):
                    x, y, w, h = entry["box"]
                    box_h_px = h * height
                    if skip_above_px and box_h_px >= skip_above_px:
                        logging.info(
                            "[Krea2Detailer] '%s' box %dpx tall >= %d; skipped",
                            entry.get("label", ridx), int(box_h_px),
                            skip_above_px,
                        )
                        continue
                    pad_x, pad_y = crop_pad * w, crop_pad * h
                    px0 = max(0, int((x - pad_x) * width))
                    py0 = max(0, int((y - pad_y) * height))
                    px1 = min(width, int(math.ceil((x + w + pad_x) * width)))
                    py1 = min(height, int(math.ceil((y + h + pad_y) * height)))
                    if px1 - px0 < 32 or py1 - py0 < 32:
                        continue
                    model_l, clip_l = self._lora_models(
                        model, clip, entry, lora_scale, lora_cache)
                    label = _clean_label(entry.get("label", ""))
                    region_seed = (seed + b * 1000 + ridx) & 0xffffffffffffffff
                    crop = out[b, py0:py1, px0:px1, :]
                    refined = self._refine_crop(
                        crop, model_l, clip_l, vae,
                        f"high quality photo of {label}, sharp focus, "
                        "detailed face, natural skin texture",
                        body_denoise, steps, cfg, sampler_name, scheduler,
                        region_seed, target_px,
                    )
                    self._paste(out, b, refined, px0, py0, px1, py1,
                                width, height)

            # ---- pass 2: face refinement anchored to DETECTED faces -------
            # Subjects drift out of their boxes (verified: runs 00049/00051
            # had faces 25-90% outside their LoRA masks, which is exactly
            # what dilutes likeness). The face pass therefore locates real
            # faces in the rendered image and restores each with its own
            # region's LoRA, wherever it landed.
            if face_denoise > 0.0:
                faces = _detect_faces(out[b])
                assigned = _assign_faces(faces, plan, width, height)
                if not assigned:
                    logging.info("[Krea2Detailer] no faces detected")
                for ridx, (fx0, fy0, fx1, fy1) in assigned:
                    entry = plan[ridx]
                    model_l, clip_l = self._lora_models(
                        model, clip, entry, lora_scale, lora_cache)
                    label = _clean_label(entry.get("label", ""))
                    fw, fh = fx1 - fx0, fy1 - fy0
                    gx0 = max(0, int(fx0 - 0.45 * fw))
                    gy0 = max(0, int(fy0 - 0.60 * fh))
                    gx1 = min(width, int(fx1 + 0.45 * fw))
                    gy1 = min(height, int(fy1 + 0.45 * fh))
                    if gx1 - gx0 < 32 or gy1 - gy0 < 32:
                        continue
                    logging.info(
                        "[Krea2Detailer] face pass '%s' (%s) at "
                        "x=%.3f y=%.3f", label, entry.get("lora"),
                        (fx0 + fx1) / 2.0 / width,
                        (fy0 + fy1) / 2.0 / height,
                    )
                    face_seed = (seed + b * 1000 + 500 + ridx) & 0xffffffffffffffff
                    crop = out[b, gy0:gy1, gx0:gx1, :]
                    refined = self._refine_crop(
                        crop, model_l, clip_l, vae,
                        f"close-up portrait photo of {label}, sharp detailed "
                        "face, natural skin texture, detailed eyes",
                        face_denoise, steps, cfg, sampler_name, scheduler,
                        face_seed, target_px,
                    )
                    self._paste(out, b, refined, gx0, gy0, gx1, gy1,
                                width, height)
                    face_previews.append(comfy.utils.common_upscale(
                        refined.movedim(-1, 0).unsqueeze(0),
                        256, 256, "lanczos", "center",
                    ).squeeze(0).movedim(0, -1).clamp(0, 1))

        preview = (
            torch.stack(face_previews, dim=0).cpu()
            if face_previews else blank
        )
        return (out, preview)

    @staticmethod
    def _paste(out, b, refined, px0, py0, px1, py1, width, height):
        crop = out[b, py0:py1, px0:px1, :]
        edges = (py0 > 0, px0 > 0, py1 < height, px1 < width)
        feather = 0.06 * min(py1 - py0, px1 - px0)
        mask = _paste_mask(
            py1 - py0, px1 - px0, feather, edges,
            refined.device, refined.dtype,
        )
        out[b, py0:py1, px0:px1, :] = (
            mask * refined + (1.0 - mask) * crop.to(refined.device)
        ).to(out.dtype).cpu()

    @staticmethod
    def _lora_models(model, clip, entry, lora_scale, cache):
        strength = float(entry.get("strength", 1.0)) * float(lora_scale)
        key = (entry["lora"], round(strength, 4))
        if key not in cache:
            path = _resolve_lora_path(entry["lora"])
            lora_sd = comfy.utils.load_torch_file(path, safe_load=True)
            cache[key] = comfy.sd.load_lora_for_models(
                model, clip, lora_sd, strength, strength)
            logging.info(
                "[Krea2Detailer] LoRA %s @ %.2f (regional %.2f x scale)",
                entry["lora"], strength, float(entry.get("strength", 1.0)),
            )
        return cache[key]

    def _refine_crop(self, crop, model_l, clip_l, vae, prompt_text, denoise,
                     steps, cfg, sampler_name, scheduler, seed, target_px):
        """Upscale a crop, re-render it img2img with the armed LoRA, and
        return it resized back to the crop's original dimensions."""
        ch, cw = int(crop.shape[0]), int(crop.shape[1])
        scale = max(1.0, float(target_px) / min(ch, cw))
        # Bound the working area so a huge padded box cannot exceed the main
        # generation's cost.
        max_pixels = 1_600_000
        if (ch * scale) * (cw * scale) > max_pixels:
            scale = math.sqrt(max_pixels / float(ch * cw))
        nh, nw = _align16(ch * scale), _align16(cw * scale)

        pixels = crop.unsqueeze(0).movedim(-1, 1)  # [1,C,H,W]
        up = comfy.utils.common_upscale(pixels, nw, nh, "lanczos", "disabled")
        up = up.movedim(1, -1)  # [1,H,W,C]

        positive = _encode_text(clip_l, prompt_text)
        negative = _encode_text(clip_l, "")
        refined = self._sample(
            model_l, vae, up, positive, negative, steps, cfg,
            sampler_name, scheduler, seed, denoise,
        )
        back = comfy.utils.common_upscale(
            refined.movedim(-1, 1), cw, ch, "lanczos", "disabled",
        ).movedim(1, -1)
        return back[0].clamp(0, 1)

    @staticmethod
    def _sample(model_l, vae, pixels, positive, negative, steps, cfg,
                sampler_name, scheduler, seed, denoise):
        latent = {"samples": vae.encode(pixels[:, :, :, :3])}
        result = common_ksampler(
            model_l, seed, steps, cfg, sampler_name, scheduler,
            positive, negative, latent, denoise=denoise,
        )[0]
        images = vae.decode(result["samples"]).clamp(0, 1)
        if images.dim() == 5:
            # video-style VAEs (WanVAE) return [B,T,H,W,C]; flatten like the
            # core VAEDecode node does
            images = images.reshape(
                -1, images.shape[-3], images.shape[-2], images.shape[-1])
        return images


NODE_CLASS_MAPPINGS = {"Krea2RegionalDetailer": Krea2RegionalDetailer}
NODE_DISPLAY_NAME_MAPPINGS = {
    "Krea2RegionalDetailer": "Krea2 Regional Detailer (Face+Body, By Fedor)",
}

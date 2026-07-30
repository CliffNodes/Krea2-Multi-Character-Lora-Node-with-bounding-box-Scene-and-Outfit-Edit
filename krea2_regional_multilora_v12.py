"""Krea2RegionalMultiLoRAV12 -- unified regional prompting for krea2edit.

V12 is intentionally isolated from V9/V11. It subclasses the stable V9 node
without changing that module, compiles one scene-wide prompt with exact token
spans, removes V9's temporary appended region encodings before sampling, and
uses fused block-sparse attention to give each regional text span exclusive
cross-modal ownership of a centered subject field inside its matching box.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import logging
import math
import re

import torch
from torch.nn.attention.flex_attention import create_block_mask, flex_attention

from .krea2_regional_multilora import _rect_token_mask_inward
from .krea2_regional_multilora_v3 import _parse_regions_v3
from .krea2_regional_multilora_v9 import (
    Krea2RegionalMultiLoRAV9,
    _EditRegionalSession,
)


WRAPPER_KEY_V12 = "krea2_regional_multilora_v12"

# LoRA delta masks extend this far past the box edge (fraction of the box's
# own size, with a canvas-fraction floor for tiny boxes) so subjects that
# slightly overflow keep full identity instead of a half-LoRA'd face.
_V12_MASK_PAD_FRAC = 0.35
_V12_MASK_PAD_MIN = 0.06

# Feather cap as a fraction of the box's smaller dimension. seam_feather is a
# CANVAS fraction, so on a tiny box the inward ramp consumed the whole
# interior and the delta never plateaued at 1.0 -- a tiny box silently ran
# its LoRA at roughly half weight. Capping the ramp per box guarantees a
# genuine full-strength core at every box size.
_V12_FEATHER_BOX_FRAC = 0.30

# Pre-softmax logit boost binding each regional text span to its full box.
# V9's soft spatial routing used +2.0 positive / -6.0 negative; V12's hard
# block replaces the negative side. Tested 2026-07-30: 2.5 pulls subjects
# toward their boxes with likeness fully intact; 4.0 visibly degrades face
# identity (span text outcompetes the LoRA-carrying image attention).
_V12_ATTRACT = 2.5


def _flex_attention_v12(q, k, v, block_mask, scale):
    # Attraction channels push head_dim past 128, which the kernel pads to
    # 256; default tile sizes then exceed shared memory (172KB vs 101KB
    # hardware limit). Smaller tiles keep the fused kernel launchable.
    return flex_attention(
        q, k, v, block_mask=block_mask, scale=scale,
        kernel_options={"BLOCK_M": 32, "BLOCK_N": 32, "num_stages": 1},
    )


# Dynamic shapes are mandatory here. A static compilation accumulated one
# specialization per resolution/reference count; after eight shapes Dynamo
# silently ran FlexAttention eagerly, materializing a dense [H,L,L] score
# tensor (17.88 GiB at L=10,000).
_FLEX_ATTENTION_V12 = torch.compile(
    _flex_attention_v12, dynamic=True, fullgraph=True
)


@dataclass(frozen=True)
class _UnifiedPlan:
    prompt: str
    spans: tuple[tuple[int, int], ...]
    labels: tuple[str, ...]
    # Plain-wrapper layout of the full caption; used to locate the caption
    # inside the grounded encoding, whose vision tokens precede the text.
    user_len: int = 0
    tail_len: int = 0
    # Normalized (x, y, w, h) box and original regions_json row index per
    # active region, in span order. Consumed by the detailer stage.
    boxes: tuple[tuple[float, float, float, float], ...] = ()
    indices: tuple[int, ...] = ()


def _prompt_token_layout(clip, text):
    """(user_token_count, tail_token_count) inside Krea's Qwen chat wrapper.

    tail counts everything from the user <|im_end|> to the end of the
    sequence. The grounding template shares the same tail, so the caption's
    position inside a grounded encoding of length L is
    [L - tail - user_count, L - tail).
    """
    tokenized = clip.tokenize(text or "")
    if not tokenized:
        raise ValueError("Krea tokenization returned no groups")
    batches = next(iter(tokenized.values()))
    if len(batches) != 1:
        raise ValueError("V12 unified prompting requires one token batch")
    pairs = batches[0]
    second_start = None
    seen = 0
    for index, pair in enumerate(pairs):
        if isinstance(pair[0], int) and pair[0] == 151644:  # <|im_start|>
            seen += 1
            if seen == 2:
                second_start = index
                break
    if second_start is None:
        raise ValueError("Krea Qwen wrapper has no second <|im_start|>")
    start = second_start + 1
    if len(pairs) > start + 1 and pairs[start][0] == 872 and pairs[start + 1][0] == 198:
        start += 2  # optional "user\n"
    for index in range(start, len(pairs)):
        if pairs[index][0] == 151645:  # <|im_end|>
            return index - start, len(pairs) - index
    raise ValueError("Krea Qwen wrapper has no user <|im_end|>")


def _prompt_token_count(clip, text):
    """Count only user-prompt lanes inside Krea's fixed Qwen chat wrapper."""
    return _prompt_token_layout(clip, text)[0]


def _horizontal(cx):
    if cx < 0.20:
        return "far-left side"
    if cx < 0.40:
        return "left side"
    if cx < 0.60:
        return "center"
    if cx < 0.80:
        return "right side"
    return "far-right side"


def _vertical(cy):
    if cy < 0.25:
        return "top"
    if cy < 0.45:
        return "upper portion"
    if cy < 0.65:
        return "middle"
    if cy < 0.82:
        return "lower portion"
    return "bottom"


def _framing(height):
    if height >= 0.70:
        return "a large prominent near-frame-height foreground subject"
    if height >= 0.45:
        return "a prominent medium-to-large subject"
    if height >= 0.25:
        return (
            "a medium-distance subject standing several steps from the "
            "camera, full body visible"
        )
    return (
        "a small distant background figure far from the camera, whole body "
        "occupying only a small part of the frame"
    )


def _builder_data(prompt):
    try:
        data = json.loads(prompt or "")
    except (TypeError, ValueError):
        return {}, []
    if not isinstance(data, dict):
        return {}, []
    comp = data.get("compositional_deconstruction") or {}
    elements = comp.get("elements") if isinstance(comp, dict) else []
    return data, elements if isinstance(elements, list) else []


def _element_box(element, fallback_index, count):
    if isinstance(element, dict):
        box = element.get("bbox")
        if isinstance(box, (list, tuple)) and len(box) == 4:
            try:
                ymin, xmin, ymax, xmax = (float(value) / 1000.0 for value in box)
                return (
                    max(0.0, min(1.0, xmin)),
                    max(0.0, min(1.0, ymin)),
                    max(1e-4, min(1.0, xmax - xmin)),
                    max(1e-4, min(1.0, ymax - ymin)),
                )
            except (TypeError, ValueError):
                pass
        if isinstance(box, dict):
            try:
                x = float(box.get("x", 0.0))
                y = float(box.get("y", 0.0))
                w = float(box.get("w", 1.0))
                h = float(box.get("h", 1.0))
                return (
                    max(0.0, min(1.0, x)),
                    max(0.0, min(1.0, y)),
                    max(1e-4, min(1.0, w)),
                    max(1e-4, min(1.0, h)),
                )
            except (TypeError, ValueError):
                pass
    width = 1.0 / max(1, count)
    return fallback_index * width, 0.0, width, 1.0


def _compile_unified_plan(clip, builder_prompt, regions_json):
    data, elements = _builder_data(builder_prompt)
    raw_rows = json.loads(regions_json or "[]")
    rows = raw_rows if isinstance(raw_rows, list) else []
    parsed = _parse_regions_v3(regions_json)

    comp = data.get("compositional_deconstruction") or {}
    global_parts = []
    high = str(data.get("high_level_description", "") or "").strip()
    background = str(comp.get("background", "") or "").strip() if isinstance(comp, dict) else ""
    active = []
    for index, region in enumerate(parsed):
        if not region.get("enable", True):
            continue
        has_lora = region.get("lora") not in ("", "None", None) and float(
            region.get("strength", 0.0)
        ) != 0.0
        has_ref = bool(region.get("ref_image")) and region.get("ref_enable", True)
        if not (has_lora or has_ref):
            continue
        row = rows[index] if index < len(rows) and isinstance(rows[index], dict) else {}
        element = elements[index] if index < len(elements) else {}
        description = str(row.get("prompt", "") or "").strip()
        if not description and isinstance(element, dict):
            description = str(element.get("desc", "") or "").strip()
        if not description:
            description = f"person {index + 1}"
        description = re.sub(
            r"^(?:a|an|one)\s+(man|woman|person)\b",
            r"one single \1 only",
            description,
            flags=re.I,
        )
        active.append((index, description, _element_box(element, index, len(parsed))))

    if high and active:
        # Boxes are authoritative over framing. The camera sentence is derived
        # from the largest active box, and close-up wording in the user's text
        # ("selfie" implies an arm's-length camera) is rewritten whenever it
        # would contradict small boxes -- otherwise the model splits the
        # difference and renders subjects overflowing or duplicated (observed:
        # tiny fridge boxes + "taking a selfie" produced a foreground selfie
        # pair PLUS a duplicate man standing inside the box).
        max_h = max(box[3] for _, _, box in active)
        if max_h >= 0.70:
            camera = "Wide-angle photograph with natural perspective"
            if re.search(r"\bselfie\b", high, re.I):
                camera = (
                    "Wide-angle selfie camera perspective with natural lens "
                    "geometry"
                )
        elif max_h >= 0.45:
            camera = (
                "Medium shot photographed from a normal standing distance; "
                "every person's full body fits inside their assigned image area"
            )
        else:
            camera = (
                "Wide shot photographed from across the room; the people are "
                "far from the camera and each whole body stays inside its "
                "small assigned image area"
            )
        global_parts.append(camera + ".")
        # Keep the user's scene/interaction description as un-owned global
        # text so subjects can pose together (e.g. an arm around the other).
        # Cardinality is enforced by the hard token routing, not by stripping
        # this sentence; only soften explicit count phrases slightly.
        interaction = re.sub(
            r"\b(?:exactly|just|only)\s+(?:the\s+)?(one|two|three|\d+)\b",
            r"\1",
            high,
            flags=re.I,
        ).strip()
        if max_h < 0.70:
            interaction = re.sub(
                r"\btak\w*\s+(?:a\s+)?selfies?\b", "posing for a photo",
                interaction, flags=re.I,
            )
            interaction = re.sub(
                r"\bselfies?\b", "photo", interaction, flags=re.I,
            )
            interaction = re.sub(
                r"\bclose[- ]?ups?\b", "photo", interaction, flags=re.I,
            )
        if interaction:
            global_parts.append(interaction.rstrip(".") + ".")
    elif high:
        global_parts.append(high.rstrip(".") + ".")
    if background:
        global_parts.append("Scene background: " + background.rstrip(".") + ".")
    if not global_parts:
        global_parts.append("A coherent scene using the assigned regional subjects.")

    prompt = " ".join(global_parts)
    char_spans = []
    labels = []
    centers = []
    shared_action = ""
    if re.search(r"\bsitting\s+(?:together\s+)?on\s+the\s+couch\b", high, re.I):
        shared_action = " seated naturally on the couch"
    boxes = []
    indices = []
    for _index, description, (x, y, w, h) in active:
        boxes.append((x, y, w, h))
        indices.append(_index)
        cx, cy = x + 0.5 * w, y + 0.5 * h
        clause = (
            f"In the {_vertical(cy)} {_horizontal(cx)}, render "
            f"{description.rstrip('.')}{shared_action} as {_framing(h)}. "
            "Exactly one visible subject "
            "must fill most of its assigned image area with minimal empty margin "
            "and remain completely inside that area."
        )
        if prompt:
            prompt += "\n"
        start = len(prompt)
        prompt += clause
        char_spans.append((start, len(prompt)))
        labels.append(description)
        centers.append(cx)

    if len(centers) > 1:
        prompt += (
            "\nMaintain exactly one distinct subject per assigned image area. "
            "Do not duplicate any subject."
        )

    spans = tuple(
        (
            _prompt_token_count(clip, prompt[:start]),
            _prompt_token_count(clip, prompt[:end]),
        )
        for start, end in char_spans
    )
    if any(end <= start for start, end in spans):
        raise ValueError("V12 compiled an empty regional token span")
    user_len, tail_len = _prompt_token_layout(clip, prompt)
    return _UnifiedPlan(
        prompt=prompt, spans=spans, labels=tuple(labels),
        user_len=user_len, tail_len=tail_len,
        boxes=tuple(boxes), indices=tuple(indices),
    )


def _trim_conditioning(conditioning, text_len):
    out = []
    for tensor, meta in conditioning:
        if not torch.is_tensor(tensor) or tensor.dim() != 3 or tensor.shape[1] < text_len:
            raise ValueError(
                f"V12 cannot trim conditioning shape {getattr(tensor, 'shape', None)} "
                f"to {text_len} text tokens"
            )
        new_meta = dict(meta)
        attention_mask = new_meta.get("attention_mask")
        if torch.is_tensor(attention_mask) and attention_mask.shape[-1] >= text_len:
            new_meta["attention_mask"] = attention_mask[..., :text_len]
        out.append([tensor[:, :text_len, :], new_meta])
    return out


def _exclusive_lora_masks(masks):
    if len(masks) < 2:
        return list(masks)
    stacked = torch.stack(masks, dim=0)
    winner = stacked.argmax(dim=0)
    return [
        mask * (winner == index).to(mask.dtype)
        for index, mask in enumerate(masks)
    ]


class _EditUnifiedRegionalSession(_EditRegionalSession):
    """Unified hard-partition attention with continuous image/reference flow."""

    def __init__(
        self,
        *args,
        unified_spans=None,
        unified_labels=None,
        unified_text_len=0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.unified_spans = tuple(tuple(x) for x in (unified_spans or ()))
        self.unified_labels = tuple(unified_labels or ())
        self.unified_text_len = int(unified_text_len)
        self._unified_cache = {}
        self._unified_logged = False

    def _build_masks_now(self, rows, cols):
        # LoRA delta skirt. The attraction field steers the subject toward
        # its box but composition text (e.g. "selfie") can make it overflow;
        # a mask clipped hard at the box edge then applies the LoRA to half
        # a face, destroying likeness (observed with small boxes). Pad the
        # DELTA masks outward so a moderately overflowing subject keeps its
        # full identity; attention ownership still uses the true boxes, and
        # _exclusive_lora_masks resolves any skirt overlap winner-take-all.
        if not self.unified_spans:
            return super()._build_masks_now(rows, cols)
        original = self.norm_boxes
        masks = []
        for index, (x0, y0, x1, y1) in enumerate(original):
            pad_x = max(_V12_MASK_PAD_FRAC * (x1 - x0), _V12_MASK_PAD_MIN)
            pad_y = max(_V12_MASK_PAD_FRAC * (y1 - y0), _V12_MASK_PAD_MIN)
            # Voronoi limit: a skirt must never reach past the midpoint of
            # the gap toward a neighboring box, or a subject rendered near
            # the shared border receives the NEIGHBOR's identity (observed:
            # side-by-side boxes swapped the woman's face to the man's LoRA).
            left, top, right, bottom = pad_x, pad_y, pad_x, pad_y
            for j, (ox0, oy0, ox1, oy1) in enumerate(original):
                if j == index:
                    continue
                rows_overlap = not (oy1 <= y0 or oy0 >= y1)
                cols_overlap = not (ox1 <= x0 or ox0 >= x1)
                if rows_overlap:
                    if ox1 <= x0:
                        left = min(left, (x0 - ox1) / 2.0)
                    if ox0 >= x1:
                        right = min(right, (ox0 - x1) / 2.0)
                if cols_overlap:
                    if oy1 <= y0:
                        top = min(top, (y0 - oy1) / 2.0)
                    if oy0 >= y1:
                        bottom = min(bottom, (oy0 - y1) / 2.0)
            px0 = max(0.0, x0 - max(0.0, left))
            py0 = max(0.0, y0 - max(0.0, top))
            px1 = min(1.0, x1 + max(0.0, right))
            py1 = min(1.0, y1 + max(0.0, bottom))
            # Per-box feather cap: the ramp may use at most a fixed fraction
            # of the padded box's smaller dimension, so even a tiny box keeps
            # a broad plateau at exactly 1.0 (full LoRA weight) instead of
            # spending its whole interior climbing the canvas-scaled ramp.
            feather = min(
                float(self.seam_feather),
                _V12_FEATHER_BOX_FRAC * min(px1 - px0, py1 - py0),
            )
            masks.append(
                _rect_token_mask_inward(rows, cols, px0, py0, px1, py1, feather)
            )
        blend = float(max(0.0, min(1.0, self.blend_override)))
        if blend > 0.0 and masks:
            uniform = 1.0 / len(masks)
            masks = [(1.0 - blend) * m + blend * uniform for m in masks]
        return masks

    def _prepare(self, dev, x):
        super()._prepare(dev, x)
        if len(self._masks_d) < 2:
            return
        # The V2 hook adds every regional delta. Overlapping boxes therefore
        # summed identities even though V12 attention had one semantic owner.
        # Make LoRA ownership deterministic too: strongest inward mask wins;
        # equal ties keep the earlier region, matching the attention owner.
        self._masks_d = _exclusive_lora_masks(self._masks_d)
        self._full_mask_cache = {}
        self._unified_cache = {}

    @staticmethod
    def _get_topts(args, kwargs):
        if len(args) >= 5 and isinstance(args[4], dict):
            return args[4]
        value = kwargs.get("transformer_options")
        return value if isinstance(value, dict) else None

    def _owners(self, device):
        key = (str(device), self.n_img, self.unified_text_len)
        cached = self._unified_cache.get(key)
        if cached is not None:
            return cached
        text = torch.zeros(self.unified_text_len, dtype=torch.int16, device=device)
        for owner, (start, end) in enumerate(self.unified_spans, 1):
            text[start:end] = owner
        image = torch.zeros(self.n_img, dtype=torch.int16, device=device)
        fields = []
        for owner, mask in enumerate(self._masks_d, 1):
            strict = mask.to(device=device) > 0
            image[(image == 0) & strict] = owner
            fields.append(torch.where(
                strict,
                0.85 + 0.15 * mask.to(device=device, dtype=torch.float32),
                torch.zeros_like(mask, device=device, dtype=torch.float32),
            ))
        cached = text, image, tuple(fields)
        self._unified_cache[key] = cached
        return cached

    @staticmethod
    def _apply_prior_mask(scores, mask, start, end):
        if mask is None:
            return
        value = mask
        if value.ndim == 2:
            value = value.unsqueeze(0).unsqueeze(0)
        elif value.ndim == 3:
            value = value.unsqueeze(1)
        scores.add_(value[..., start:end, :].to(device=scores.device, dtype=scores.dtype))

    def _partition_text(self, scores, start, end, text_owners):
        query = text_owners[start:end]
        blocked = (text_owners.view(1, -1) > 0) & (
            query.view(-1, 1) != text_owners.view(1, -1)
        )
        scores.masked_fill_(blocked.view(1, 1, end - start, -1), float("-inf"))

    def _partition_main(self, scores, start, end, text, image, target_start):
        txt = self.unified_text_len
        text_end = min(end, txt)
        if start < text_end:
            query = text[start:text_end]
            blocked_text = (text.view(1, -1) > 0) & (
                query.view(-1, 1) != text.view(1, -1)
            )
            scores[:, :, : text_end - start, :txt].masked_fill_(
                blocked_text.view(1, 1, text_end - start, -1), float("-inf")
            )
            blocked_target = (image.view(1, -1) > 0) & (
                query.view(-1, 1) != image.view(1, -1)
            )
            scores[:, :, : text_end - start, target_start:].masked_fill_(
                blocked_target.view(1, 1, text_end - start, -1), float("-inf")
            )

        image_start = max(start, target_start)
        if image_start < end:
            query = image[image_start - target_start : end - target_start]
            blocked_text = (text.view(1, -1) > 0) & (
                query.view(-1, 1) != text.view(1, -1)
            )
            scores[:, :, image_start - start :, :txt].masked_fill_(
                blocked_text.view(1, 1, end - image_start, -1), float("-inf")
            )

    def _add_bias(self, scores, start, end, fields, target_start):
        for (span_start, span_end), field in zip(self.unified_spans, fields):
            text_start = max(start, span_start)
            text_end = min(end, span_end)
            if text_start < text_end:
                scores[:, :, text_start - start : text_end - start, target_start:].add_(
                    field.view(1, 1, 1, -1), alpha=self.inside_boost
                )
            image_start = max(start, target_start)
            if image_start < end:
                part = field[image_start - target_start : end - target_start]
                scores[
                    :, :, image_start - start :, span_start:span_end
                ].add_(part.view(1, 1, -1, 1), alpha=self.inside_boost)

    def _attention(self, original, q, k, v, heads, **kwargs):
        seq_q, seq_k = int(q.shape[-2]), int(k.shape[-2])
        txt, n_img = self.unified_text_len, self.n_img
        main = n_img > 0 and seq_q == seq_k and seq_q >= txt + n_img
        if not main:
            return original(q, k, v, heads, **kwargs)
        if q.ndim != 4 or not kwargs.get("skip_reshape", False):
            return original(q, k, v, heads, **kwargs)

        target_start = seq_q - n_img
        if not self._unified_logged:
            logging.info(
                "[V12/unified] attention backend=%s q=%s heads=%d skip_reshape=%s "
                "mask=%s contiguous=%s",
                getattr(original, "__name__", type(original).__name__),
                tuple(q.shape), int(heads), kwargs.get("skip_reshape", False),
                kwargs.get("mask") is not None, q.is_contiguous(),
            )
        if kwargs.get("mask") is not None:
            return original(q, k, v, heads, **kwargs)

        cache_key = ("block_mask", seq_q, str(q.device))
        cached_partition = self._unified_cache.get(cache_key)
        if cached_partition is None:
            rows, cols, _source = self._grid_info
            pixel_boxes = tuple(
                (
                    max(0, min(cols, int((x0 + 0.25 * (x1 - x0)) * cols))),
                    max(0, min(rows, int((y0 + 0.08 * (y1 - y0)) * rows))),
                    max(0, min(
                        cols,
                        int(((x1 - 0.25 * (x1 - x0)) * cols) + 0.999999),
                    )),
                    max(0, min(
                        rows,
                        int(((y1 - 0.08 * (y1 - y0)) * rows) + 0.999999),
                    )),
                )
                for x0, y0, x1, y1 in self.norm_boxes
            )
            full_pixel_boxes = tuple(
                (
                    max(0, min(cols, int(x0 * cols))),
                    max(0, min(rows, int(y0 * rows))),
                    max(0, min(cols, int(x1 * cols + 0.999999))),
                    max(0, min(rows, int(y1 * rows + 0.999999))),
                )
                for x0, y0, x1, y1 in self.norm_boxes
            )
            spans = self.unified_spans

            def owner(index):
                result = torch.zeros_like(index)
                for region, (start, end) in enumerate(spans, 1):
                    result = torch.where(
                        (result == 0) & (index >= start) & (index < end),
                        torch.full_like(result, region),
                        result,
                    )
                pixel = index - target_start
                x, y = pixel % cols, pixel // cols
                is_target = index >= target_start
                for region, (x0, y0, x1, y1) in enumerate(pixel_boxes, 1):
                    result = torch.where(
                        (result == 0)
                        & is_target
                        & (x >= x0) & (x < x1)
                        & (y >= y0) & (y < y1),
                        torch.full_like(result, region),
                        result,
                    )
                return result

            def mask_mod(batch, head, query, key):
                query_owner, key_owner = owner(query), owner(key)
                query_text, key_text = query < txt, key < txt
                query_target, key_target = (
                    query >= target_start,
                    key >= target_start,
                )
                blocked_text = (
                    key_text
                    & (key_owner > 0)
                    & (query_text | query_target)
                    & (query_owner != key_owner)
                )
                blocked_target = (
                    key_target
                    & (key_owner > 0)
                    & query_text
                    & (query_owner != key_owner)
                )
                return ~(blocked_text | blocked_target)

            block_mask = create_block_mask(
                mask_mod, None, None, seq_q, seq_q, device=q.device
            )

            # Attraction field: hard blocking alone only *prevents* identity
            # leakage; nothing pulls a subject into its box, so un-owned
            # global text (scene/pose sentence) freely seeds subjects at the
            # model's preferred composition. score_mod fails Inductor's
            # dynamic-shape lowering here, so encode the boost as extra
            # indicator channels on q/k instead: for region r, channel r is
            # set on in-box target queries and in-span text keys (and the
            # mirrored pair on channels R+r), so q_aug @ k_aug adds exactly
            # alpha^2 to the logit for cross-modal same-region pairs. The
            # FULL box is used (not the centered blocking core) so subjects
            # materialize across the whole box.
            index = torch.arange(seq_q, device=q.device)
            pixel = index - target_start
            px, py = pixel % cols, pixel // cols
            is_target = index >= target_start
            n_regions = len(spans)
            flags_q = torch.zeros(seq_q, 2 * n_regions, device=q.device)
            flags_k = torch.zeros(seq_q, 2 * n_regions, device=q.device)
            for r, ((start, end), (x0, y0, x1, y1)) in enumerate(
                zip(spans, full_pixel_boxes)
            ):
                in_span = ((index >= start) & (index < end)).float()
                in_box = (
                    is_target
                    & (px >= x0) & (px < x1)
                    & (py >= y0) & (py < y1)
                ).float()
                flags_q[:, r] = in_box
                flags_q[:, n_regions + r] = in_span
                flags_k[:, r] = in_span
                flags_k[:, n_regions + r] = in_box
            cached_partition = (
                block_mask,
                flags_q.view(1, 1, seq_q, -1),
                flags_k.view(1, 1, seq_q, -1),
            )
            self._unified_cache[cache_key] = cached_partition
        block_mask, flags_q, flags_k = cached_partition

        if main and not self._unified_logged:
            logging.info(
                "[V12/unified] ACTIVE fused-partition text=%d refs=%d "
                "target=%d attract=%.2f spans=%s",
                txt, target_start - txt, n_img, _V12_ATTRACT,
                list(zip(self.unified_labels, self.unified_spans)),
            )
            self._unified_logged = True

        scale = kwargs.get("scale")
        if scale is None:
            scale = 1.0 / math.sqrt(q.shape[-1])
        alpha = math.sqrt(_V12_ATTRACT / float(scale))
        batch, n_heads = q.shape[0], q.shape[1]
        pad_q = (alpha * flags_q).to(q.dtype).expand(batch, n_heads, -1, -1)
        pad_k = (alpha * flags_k).to(k.dtype).expand(batch, n_heads, -1, -1)
        pad_v = torch.zeros(
            batch, n_heads, v.shape[2], pad_q.shape[-1],
            device=v.device, dtype=v.dtype,
        )
        output = _FLEX_ATTENTION_V12(
            torch.cat([q, pad_q], dim=-1).contiguous(),
            torch.cat([k, pad_k], dim=-1).contiguous(),
            torch.cat([v, pad_v], dim=-1).contiguous(),
            block_mask,
            float(scale),
        )[..., : v.shape[-1]]
        if kwargs.get("skip_output_reshape", False):
            return output
        return output.transpose(1, 2).reshape(output.shape[0], output.shape[2], -1)

    def run(self, executor, *args, **kwargs):
        topts = self._get_topts(args, kwargs)
        if topts is None or not self.unified_spans:
            return super().run(executor, *args, **kwargs)
        previous = topts.get("optimized_attention_override")

        def override(func, q, k, v, heads, **attn_kwargs):
            base = func
            if previous is not None:
                def base(*inner_args, **inner_kwargs):
                    return previous(func, *inner_args, **inner_kwargs)
            return self._attention(base, q, k, v, heads, **attn_kwargs)

        topts["optimized_attention_override"] = override
        try:
            return super().run(executor, *args, **kwargs)
        finally:
            if previous is None:
                topts.pop("optimized_attention_override", None)
            else:
                topts["optimized_attention_override"] = previous


class Krea2RegionalMultiLoRAV12(Krea2RegionalMultiLoRAV9):
    """V9 likeness engine plus isolated unified regional placement."""

    DESCRIPTION = (
        "Krea2 Regional Multi-LoRA V12 (By Fedor). Place multiple character "
        "LoRAs in one image, each confined to its own bounding box, with the "
        "box also controlling WHERE and HOW LARGE the subject renders.\n\n"
        "How it works: compiles your box-builder prompt into one unified "
        "caption with exact per-region token spans, then runs fused "
        "block-sparse attention that gives each region's text exclusive "
        "ownership of its box (no identity bleeding between neighbors) plus "
        "an attraction field that pulls each subject into its box. LoRA "
        "deltas are masked per box with overflow skirts and per-box feather "
        "so tiny boxes keep full likeness.\n\n"
        "Scene transfer: wire a photo to extra_ref_1 and the whole image is "
        "re-generated inside that scene — only the edit LoRA is needed. "
        "Outfit/object transfer: wire a second photo to extra_ref_2 with a "
        "full-canvas box and describe its role in refs_json."
    )

    def _arm_regional_loras(
        self, model, active, norm_boxes, cw, ch, seam_feather, blend_override,
        base_strength, wrapper_key=None, session_cls=None,
        into_transformer_options=False, session_kwargs=None,
    ):
        plan = getattr(self, "_v12_plan", None)
        if (
            plan is not None
            and session_kwargs
            and len(plan.spans) == len(norm_boxes)
            and session_kwargs.get("spatial_token_ranges")
        ):
            appended = session_kwargs["spatial_token_ranges"]
            self._v12_base_text_len = int(appended[0][0])
            # The grounded encoding embeds ~1k+ vision tokens BEFORE the
            # caption; plan.spans are caption-relative, so shift them to the
            # caption's true position: [base - tail - user_len, base - tail).
            caption_start = (
                self._v12_base_text_len - plan.tail_len - plan.user_len
            )
            if caption_start < 0:
                raise RuntimeError(
                    f"V12 caption offset invalid: base={self._v12_base_text_len} "
                    f"tail={plan.tail_len} user={plan.user_len}"
                )
            shifted_spans = tuple(
                (caption_start + start, caption_start + end)
                for start, end in plan.spans
            )
            if any(end > self._v12_base_text_len - plan.tail_len
                   for _start, end in shifted_spans):
                raise RuntimeError(
                    f"V12 shifted spans overflow the caption: {shifted_spans} "
                    f"caption_end={self._v12_base_text_len - plan.tail_len}"
                )
            logging.info(
                "[V12/unified] caption at %d:%d of %d grounded tokens; "
                "spans shifted to %s",
                caption_start, self._v12_base_text_len - plan.tail_len,
                self._v12_base_text_len,
                list(zip(plan.labels, shifted_spans)),
            )
            session_cls = _EditUnifiedRegionalSession
            session_kwargs = {
                "unified_spans": shifted_spans,
                "unified_labels": plan.labels,
                "unified_text_len": self._v12_base_text_len,
            }
            wrapper_key = WRAPPER_KEY_V12
        return super()._arm_regional_loras(
            model, active, norm_boxes, cw, ch, seam_feather, blend_override,
            base_strength,
            wrapper_key=wrapper_key or WRAPPER_KEY_V12,
            session_cls=session_cls or _EditRegionalSession,
            into_transformer_options=into_transformer_options,
            session_kwargs=session_kwargs,
        )

    def _apply_edit_mode(
        self, model, clip, vae, canvas_width, canvas_height, regions_json,
        bboxes, extras, edit_lora, edit_lora_strength, compose_steps,
        compose_seed, prompt, negative_prompt, grounding_px,
        base_strength, ref_strength, ref_start_percent, ref_end_percent,
        ref_feather, seam_feather, blend_override,
        auto_portrait, portrait_steps, portrait_seed, ref_max_side,
        portrait_preview,
    ):
        plan = _compile_unified_plan(clip, prompt, regions_json)
        self._v12_plan = plan
        self._v12_base_text_len = 0
        logging.info("[V12/unified] compiled prompt: %.500s", plan.prompt)
        try:
            model_out, positive, negative, preview, data = super()._apply_edit_mode(
                model, clip, vae, canvas_width, canvas_height, regions_json,
                bboxes, extras, edit_lora, edit_lora_strength, compose_steps,
                compose_seed, plan.prompt, negative_prompt, grounding_px,
                base_strength, ref_strength, ref_start_percent, ref_end_percent,
                ref_feather, seam_feather, blend_override,
                auto_portrait, portrait_steps, portrait_seed, ref_max_side,
                portrait_preview,
            )
            if self._v12_base_text_len <= 0:
                raise RuntimeError("V12 unified attention was not armed")
            positive = _trim_conditioning(positive, self._v12_base_text_len)
            payload = dict(data) if isinstance(data, dict) else {}
            parsed = _parse_regions_v3(regions_json)
            detail_plan = []
            for label, box, index in zip(plan.labels, plan.boxes, plan.indices):
                region = parsed[index] if index < len(parsed) else {}
                detail_plan.append({
                    "label": label,
                    "box": [float(v) for v in box],
                    "lora": str(region.get("lora", "") or ""),
                    "strength": float(region.get("strength", 1.0))
                    * float(base_strength),
                })
            payload.update({
                "engine": "v12_unified_spatial",
                "unified_prompt": plan.prompt,
                "unified_token_spans": [
                    {"label": label, "start": start, "end": end}
                    for label, (start, end) in zip(plan.labels, plan.spans)
                ],
                "detail_plan": detail_plan,
            })
            return model_out, positive, negative, preview, payload
        finally:
            self._v12_plan = None


NODE_CLASS_MAPPINGS = {"Krea2RegionalMultiLoRAV12": Krea2RegionalMultiLoRAV12}
NODE_DISPLAY_NAME_MAPPINGS = {
    "Krea2RegionalMultiLoRAV12": (
        "Krea2 Regional Multi-LoRA V12 (Unified Spatial, By Fedor)"
    ),
}

"""Krea2 Regional Edit Patch - region-placed in-context references + attention isolation.

Generalizes the krea2edit in-context edit mechanism (ComfyUI-Krea2Edit) from one/two
full-frame reference frames to N *region-placed* reference frames:

  - Each region's reference latent is patchified and injected as its own RoPE frame
    (frame = i+1), with its h/w position ids set to the TARGET bounding box coords.
    So reference i's clean appearance tokens live exactly where region i should be -
    precise per-bbox placement + strong identity, in a single pass, without the
    background artifacts of a pre-composited collage.
  - Optional attention isolation: a (seq, seq) mask so a target token only attends to
    the reference of the region(s) that contain it, and references never cross-attend.
    Kills the cross-identity bleed that one-pass multi-ref otherwise suffers.

Sequence built per step: [text | ref_1 | ref_2 | ... | ref_N | target].
Requires the Krea2 Identity Edit LoRA applied to the model (LoraLoaderModelOnly).

NOTE: attention isolation passes a mask through the DiT blocks. This needs a
mask-capable attention backend; if you launched ComfyUI with --use-sage-attention
and isolation misbehaves, turn isolation off (placement alone is already strong) or
relaunch without sage.
"""
import logging

import torch
import torch.nn.functional as F
from einops import rearrange

import comfy.ldm.common_dit
import comfy.patcher_extension
from comfy.ldm.flux.layers import timestep_embedding

MAX_REGIONS = 4


def _to_4d(v):
    if v.ndim == 5:
        b, c, t, h, w = v.shape
        return v.reshape(b * t, c, h, w)
    return v


def _parse_box(s):
    default = (0.0, 0.0, 1.0, 1.0)
    if not s or not str(s).strip():
        return default
    try:
        parts = [float(v) for v in str(s).replace(";", ",").split(",")]
        if len(parts) == 4:
            x, y, w, h = parts
            return (max(0.0, x), max(0.0, y), max(1e-3, w), max(1e-3, h))
    except ValueError:
        pass
    return default


def _region_grid(box, h_, w_):
    """Normalized (x,y,w,h) -> integer token-grid span (gx0, gy0, gw, gh)."""
    bx, by, bw, bh = box
    gx0 = min(max(0, int(round(bx * w_))), w_ - 1)
    gy0 = min(max(0, int(round(by * h_))), h_ - 1)
    gw = max(1, min(int(round(bw * w_)), w_ - gx0))
    gh = max(1, min(int(round(bh * h_)), h_ - gy0))
    return gx0, gy0, gw, gh


def _region_ids(bs, frame, gx0, gy0, gw, gh, device):
    ids = torch.zeros(gh, gw, 3, device=device, dtype=torch.float32)
    ids[..., 0] = frame
    ids[..., 1] = torch.arange(gy0, gy0 + gh, device=device, dtype=torch.float32)[:, None]
    ids[..., 2] = torch.arange(gx0, gx0 + gw, device=device, dtype=torch.float32)[None, :]
    return ids.reshape(1, gh * gw, 3).repeat(bs, 1, 1)


def _build_isolation_mask(txtlen, ref_lens, tgtlen, spans, h_, w_, device):
    """Bool (seq, seq) attention mask (True = attend).

    - text attends everything;
    - ref_i attends text + itself only (no ref<->ref, no ref->target);
    - target token attends text + all target + refs whose bbox contains it.
    """
    seq = txtlen + sum(ref_lens) + tgtlen
    m = torch.ones(seq, seq, dtype=torch.bool, device=device)

    ref_ranges = []
    off = txtlen
    for rl in ref_lens:
        ref_ranges.append((off, off + rl))
        off += rl
    tgt0 = txtlen + sum(ref_lens)

    # ref rows: block everything except text + self
    for i, (a, b) in enumerate(ref_ranges):
        m[a:b, :] = False
        m[a:b, :txtlen] = True
        m[a:b, a:b] = True

    # target rows: block all ref cols by default, then re-open per-region membership
    if ref_ranges:
        m[tgt0:tgt0 + tgtlen, txtlen:tgt0] = False
        # target token linear index = gy * w_ + gx (rearrange "b (h w) c")
        gy = torch.arange(h_, device=device).view(h_, 1).expand(h_, w_).reshape(-1)
        gx = torch.arange(w_, device=device).view(1, w_).expand(h_, w_).reshape(-1)
        for (a, b), (gx0, gy0, gw, gh) in zip(ref_ranges, spans):
            inside = (gx >= gx0) & (gx < gx0 + gw) & (gy >= gy0) & (gy < gy0 + gh)
            rows = tgt0 + torch.nonzero(inside, as_tuple=False).squeeze(1)
            if rows.numel():
                m[rows.unsqueeze(1), torch.arange(a, b, device=device).unsqueeze(0)] = True
    return m


def regional_edit_forward(m, x, timesteps, context, regions, isolation, transformer_options):
    patch = m.patch
    temporal = x.ndim == 5
    if temporal:
        b5, c5, t5, h5, w5 = x.shape
    x = _to_4d(x)
    bs, c, H_orig, W_orig = x.shape
    x = comfy.ldm.common_dit.pad_to_patch_size(x, (patch, patch))
    H, W = x.shape[-2], x.shape[-1]
    h_, w_ = H // patch, W // patch
    device = x.device

    context = m._unpack_context(context)
    tgt_img = m.first(rearrange(x, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=patch, pw=patch))

    t = m.tmlp(timestep_embedding(timesteps, m.tdim).unsqueeze(1).to(tgt_img.dtype))
    tvec = m.tproj(t)
    context = m.txtfusion(context, mask=None, transformer_options=transformer_options)
    context = m.txtmlp(context)
    txtlen = context.shape[1]
    tgtlen = tgt_img.shape[1]

    ref_imgs, ref_pos, ref_lens, spans = [], [], [], []
    for i, r in enumerate(regions):
        gx0, gy0, gw, gh = _region_grid(r["box"], h_, w_)
        src = _to_4d(r["latent"]).to(device, x.dtype)
        if src.shape[0] != bs:
            src = src[:1].expand(bs, *src.shape[1:])
        src = F.interpolate(src.float(), size=(gh * patch, gw * patch), mode="bilinear").to(x.dtype)
        tok = m.first(rearrange(src, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=patch, pw=patch))
        ref_imgs.append(tok)
        ref_pos.append(_region_ids(bs, i + 1, gx0, gy0, gw, gh, device))
        ref_lens.append(tok.shape[1])
        spans.append((gx0, gy0, gw, gh))

    combined = torch.cat([context] + ref_imgs + [tgt_img], dim=1)

    txtpos = torch.zeros(bs, txtlen, 3, device=device, dtype=torch.float32)
    tgtids = torch.zeros(h_, w_, 3, device=device, dtype=torch.float32)
    tgtids[..., 1] = torch.arange(h_, device=device, dtype=torch.float32)[:, None]
    tgtids[..., 2] = torch.arange(w_, device=device, dtype=torch.float32)[None, :]
    tgtpos = tgtids.reshape(1, h_ * w_, 3).repeat(bs, 1, 1)
    pos = torch.cat([txtpos] + ref_pos + [tgtpos], dim=1)
    freqs = m.pe_embedder(pos)

    mask = None
    if isolation and ref_imgs:
        mask = _build_isolation_mask(txtlen, ref_lens, tgtlen, spans, h_, w_, device)

    for block in m.blocks:
        combined = block(combined, tvec, freqs, mask, transformer_options=transformer_options)

    final = m.last(combined, t)
    srclen = sum(ref_lens)
    out = final[:, txtlen + srclen: txtlen + srclen + tgtlen, :]
    out = rearrange(out, "b (h w) (c ph pw) -> b c (h ph) (w pw)",
                    h=h_, w=w_, ph=patch, pw=patch, c=m.channels)
    out = out[:, :, :H_orig, :W_orig]
    if temporal:
        out = out.reshape(b5, t5, m.channels, H_orig, W_orig).movedim(1, 2)
    return out


class Krea2RegionalEditPatch:
    @classmethod
    def INPUT_TYPES(cls):
        box_defaults = ["0.02,0.04,0.47,0.92", "0.51,0.04,0.47,0.92",
                        "0.02,0.04,0.31,0.92", "0.67,0.04,0.31,0.92"]
        opt = {"attention_isolation": ("BOOLEAN", {"default": False,
               "tooltip": "Isolate each region's reference from the others (kills "
                          "cross-identity bleed). Needs a mask-capable attention "
                          "backend; disable if using --use-sage-attention and it breaks."})}
        for i in range(1, MAX_REGIONS + 1):
            opt[f"region_latent_{i}"] = ("LATENT", {"tooltip": "VAEEncode of this region's reference"})
            opt[f"box_{i}"] = ("STRING", {"default": box_defaults[i - 1],
                                          "tooltip": "x,y,w,h in 0..1 target canvas fractions"})
        return {"required": {"model": ("MODEL",)}, "optional": opt}

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = "krea2/regional"
    DESCRIPTION = ("Inject N region-placed reference latents as krea2edit in-context "
                   "frames at their target bbox positions (+ optional attention "
                   "isolation). Apply the Krea2 Identity Edit LoRA to the model first.")

    def patch(self, model, attention_isolation=False, **kw):
        regions = []
        for i in range(1, MAX_REGIONS + 1):
            lat = kw.get(f"region_latent_{i}")
            if lat is None:
                continue
            samples = model.model.process_latent_in(lat["samples"])
            regions.append({"latent": samples, "box": _parse_box(kw.get(f"box_{i}"))})

        if not regions:
            logging.warning("[Krea2RegionalEditPatch] no region_latent inputs; passthrough.")
            return (model,)

        m = model.clone()
        iso = bool(attention_isolation)
        n = len(regions)

        def wrapper(executor, x, timesteps, context, attention_mask=None,
                    transformer_options={}, **kwargs):
            dm = executor.class_obj
            return regional_edit_forward(dm, x, timesteps, context, regions, iso, transformer_options)

        to = m.model_options.setdefault("transformer_options", {})
        comfy.patcher_extension.add_wrapper_with_key(
            comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL,
            "krea2_regional_edit", wrapper, to)
        logging.info("[Krea2RegionalEditPatch] %d region ref(s) armed, isolation=%s", n, iso)
        return (m,)


NODE_CLASS_MAPPINGS = {"Krea2RegionalEditPatch": Krea2RegionalEditPatch}
NODE_DISPLAY_NAME_MAPPINGS = {"Krea2RegionalEditPatch": "Krea2 Regional Edit Patch (region refs + isolation)"}

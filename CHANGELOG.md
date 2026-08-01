# Changelog

## 2.0.0 — V12 Unified Spatial + krea2edit + Regional Detailer

The headline: **bounding boxes now control WHERE and HOW LARGE each subject
renders**, not just where its LoRA is allowed to act — and you can drop your
LoRA characters into any real photo (scene transfer) or dress them from a
second photo (outfit transfer) **with nothing but the standard identity edit
LoRA**. No per-character reference photos, no portrait pre-renders required.

### New node: `Krea2 Regional Multi-LoRA V12 (Unified Spatial)`

- **Unified caption with exact token spans.** Instead of appending separate
  per-region text encodings, V12 compiles your box-builder prompt into ONE
  scene-wide caption and resolves the exact Qwen token span of each region's
  subject clause — including the offset math to locate the caption inside the
  grounded (vision-token-prefixed) encoding. Every routing decision operates
  on the real tokens, not approximations.
- **Fused block-sparse attention ownership.** A FlexAttention block mask
  gives each region's text span exclusive cross-modal ownership of a field
  inside its box: subject A's tokens cannot influence subject B's pixels and
  vice versa. This is a hard block, not a bias — the main cure for identity
  bleeding and "one box accurate, the neighbor generic".
- **Attraction field.** Hard blocking alone only *prevents* leakage; nothing
  pulls a subject into its box. V12 adds a pre-softmax logit boost binding
  each regional span to its full box (encoded as indicator channels on Q/K,
  which survives torch.compile where score_mod does not). Subjects
  materialize inside their boxes instead of wherever the model prefers.
- **Box-authoritative framing.** The compiled caption's camera sentence is
  derived from the largest active box height, and conflicting close-up
  wording (e.g. "selfie" with knee-high boxes) is rewritten automatically.
  Box size is the framing contract: tall box = large subject, small box =
  distant subject.
- **LoRA mask skirts with a Voronoi limit.** Delta masks extend past the box
  edge (35% of box size, 6%-of-canvas floor) so a subject that slightly
  overflows keeps full identity — but each skirt stops halfway across the
  gap to any neighboring box, so skirts can never cause cross-identity
  bleed.
- **Per-box feather cap.** Feathering is now capped per box (30% of the
  box's smaller side). Previously the canvas-fraction feather could consume
  a tiny box's whole interior, silently running its LoRA at ~half weight —
  tiny boxes now always reach a full-strength core.
- **Overlap competition.** Where two LoRA masks overlap, the region with the
  strongest inward mask wins outright (winner-take-all) instead of both
  identities summing into a blend.
- **Coexists with V9/V3/v1.** V12 is a separate node and module; existing
  workflows using the older nodes are untouched.

### Scene + outfit transfer (krea2edit path) — only the edit LoRA required

- Wire any photo into `extra_ref_1` and the whole generation is re-composed
  inside that scene: global lighting, perspective and reflections integrate
  naturally because the image is generated from noise with the scene as a
  reference frame — not latent-pasted.
- Wire a second photo into `extra_ref_2` (full-canvas box) and give it a
  role in `refs_json` (e.g. `{"role":"object","note":"outfit, worn by the
  woman"}`) for outfit/object/style transfer. The node writes the referring
  text with the correct frame number automatically.
- Requires only the standard Krea2 identity edit LoRA in the `edit_lora`
  slot. Character likeness comes entirely from your per-region character
  LoRAs — portraits and per-region reference photos are optional extras,
  not requirements.

### New node: `Krea2 Regional Detailer`

- Optional post-pass that re-renders each subject at high resolution with
  its OWN character LoRA: a body pass over each planned box, then a face
  pass anchored to **detected** faces (YOLOv8 `face_yolov8m.pt` if
  installed, OpenCV Haar fallback).
- Faces are detected on the full final image and assigned one-to-one to
  regions by proximity — so even a subject that drifted out of its box gets
  its correct LoRA applied exactly where its face actually rendered. This
  recovers likeness in the residual failure case where a subject renders
  across the box seam.
- Feathered seamless paste-back; total pixel budget capped so a huge box
  cannot exceed the main generation's cost; `skip_above_px` gate to only
  refine small/distant subjects.

### Fixes

- **Major trainer key compatibility**: the shared regional loader now
  canonicalizes Krea 2 LoRA keys from AI Toolkit/Diffusers/PEFT,
  Musubi/Kohya, and OneTrainer. Repeated wrappers such as
  `base_model.model.transformer.`, flat `lora_unet_` /
  `lora_transformer_` names, OneTrainer `__` separators, and named PEFT
  adapters are supported. Native ComfyUI and LoKr paths remain unchanged.
  Real 264-layer Diffusers fixtures now translate 264/264 modules instead of
  matching zero; native 256-layer LoRA and LoKr fixtures remain 256/256.
- **Public package import error**: `__init__.py` referenced modules that
  were not in the repository, breaking fresh installs. The registration
  list now matches the shipped files exactly.
- **Memory-leak warnings / VRAM pressure**: regional sessions now hold weak
  references to the model patcher and UNet layers, so ComfyUI can actually
  garbage-collect swapped models ("Potential memory leak" warnings gone).
- **Widget corruption via BOUNDING_BOX widgets**: all `BOUNDING_BOX` inputs
  are `forceInput` sockets, preventing the Node 2.0 frontend from injecting
  phantom x/y/width/height widgets that phase-shifted every saved value
  after them.
- **OOM at high resolution**: FlexAttention is compiled with dynamic shapes
  (no recompile-limit fallback to a dense 17 GB score tensor) and uses
  reduced kernel tiles so the fused kernel fits shared memory even with the
  attraction channels.
- **OOM with native-resolution reference frames**: block-mask construction is
  now compiled, so it reduces directly to sparse blocks instead of first
  materializing a dense Q-by-K mask. The reported 63,662-token case dropped
  from a 30.27 GiB allocation request to a measured 99 MiB peak.
- **Duplicate-subject / cardinality bugs**: centered semantic ownership
  cores and per-region cardinality phrasing stop one wide box from seeding
  two people.
- Tooltips added or expanded on nearly every input of the V9/V12 and
  Detailer nodes — hover any field for what it does and how to set it.

## 1.x

- **v3** — Reference Lock: per-region reference images via latent-mold
  guidance at the sampler (identity anchoring across generations), per-row
  reference upload with inline thumbnails, ref-only regions, scheduled
  guidance window.
- **LoKr support** — Kronecker-factored LoRAs (`lokr_w1`/`lokr_w2`) from
  newer ai-toolkit builds now load and apply correctly.
- **Box-row sync fixes** — region rows reliably track box creation and
  deletion in the builder.
- **v1** — original release: hard per-box LoRA masking via masked
  activation-delta injection.

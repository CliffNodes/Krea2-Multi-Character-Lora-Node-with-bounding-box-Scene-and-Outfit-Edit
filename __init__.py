"""
ComfyUI-Krea2-Regional-MultiLoRA (By Fedor)

Regional multi-LoRA for Krea 2 via masked activation-delta injection. Each
region's LoRA is constrained to its bounding box at forward time - a hard
spatial mask, not an attention-bias nudge.

V12 adds unified spatial routing on top of the krea2edit engine: one compiled
scene prompt with exact per-region token spans, fused block-sparse attention
that gives each subject's text exclusive ownership of its box, scene/outfit
reference transfer, and an optional face/body detailer pass.
"""

from .krea2_regional_multilora import (
    NODE_CLASS_MAPPINGS as _MULTILORA_CLASSES,
    NODE_DISPLAY_NAME_MAPPINGS as _MULTILORA_NAMES,
)
from .krea2_reference_lock import (
    NODE_CLASS_MAPPINGS as _REFLOCK_CLASSES,
    NODE_DISPLAY_NAME_MAPPINGS as _REFLOCK_NAMES,
)
from .krea2_regional_multilora_v3 import (
    NODE_CLASS_MAPPINGS as _V3_CLASSES,
    NODE_DISPLAY_NAME_MAPPINGS as _V3_NAMES,
)
from .krea2_regional_multilora_v9 import (
    NODE_CLASS_MAPPINGS as _V9_CLASSES,
    NODE_DISPLAY_NAME_MAPPINGS as _V9_NAMES,
)
from .krea2_regional_multilora_v12 import (
    NODE_CLASS_MAPPINGS as _V12_CLASSES,
    NODE_DISPLAY_NAME_MAPPINGS as _V12_NAMES,
)
from .krea2_regional_detailer import (
    NODE_CLASS_MAPPINGS as _DETAILER_CLASSES,
    NODE_DISPLAY_NAME_MAPPINGS as _DETAILER_NAMES,
)

NODE_CLASS_MAPPINGS = {
    **_MULTILORA_CLASSES, **_REFLOCK_CLASSES, **_V3_CLASSES,
    **_V9_CLASSES, **_V12_CLASSES, **_DETAILER_CLASSES,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    **_MULTILORA_NAMES, **_REFLOCK_NAMES, **_V3_NAMES,
    **_V9_NAMES, **_V12_NAMES, **_DETAILER_NAMES,
}

WEB_DIRECTORY = "./web"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]

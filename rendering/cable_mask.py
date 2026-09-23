"""B4: cable binary mask generation (Mujoco Plan 0, Part B4).

Uses Plan 0's Option A (color threshold): the cable materials are already
distinctively colored (mat_cable_red rgba="0.95 0.03 0.02 1", mat_cable_blue
rgba="0.02 0.18 0.95 1" -- see mujoco_scenes/generate_triple_scene.py), so a
per-color HSV threshold segments both cables directly with no geom-group
rendering pass needed (Plan 0's Option B is unnecessary here).
"""

from __future__ import annotations

import cv2
import numpy as np

# HSV ranges derived from the known material RGBAs, with generous tolerance
# for lighting/shading variation across the capsule surface.
_HUE_RANGES = {
    "red": [(0, 12), (245, 255)],   # red wraps around hue 0/255 in OpenCV's 0-255 hue scale
    "blue": [(150, 180)],
}
_SAT_MIN = 80
# HSV saturation is numerically unstable for near-black pixels (a faint
# color cast in a dark shadow can read as high "saturation" despite being
# visually black), so V_MIN must be high enough to reject those false
# positives while still passing the cable material's actual brightness
# (mat_cable_red/blue render at V well above 150 in practice).
_VAL_MIN = 100


def get_cable_mask(rgb: np.ndarray, cable_color: str) -> np.ndarray:
    if cable_color not in _HUE_RANGES:
        raise ValueError(f"cable_color must be one of {list(_HUE_RANGES)}, got '{cable_color}'")

    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    hue, sat, val = hsv[..., 0], hsv[..., 1], hsv[..., 2]

    hue_mask = np.zeros(hue.shape, dtype=bool)
    for lo, hi in _HUE_RANGES[cable_color]:
        hue_mask |= (hue >= lo) & (hue <= hi)

    return hue_mask & (sat >= _SAT_MIN) & (val >= _VAL_MIN)

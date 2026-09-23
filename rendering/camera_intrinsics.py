"""B1: camera intrinsics matching D435 color-stream conventions (Mujoco Plan 0, Part B1).

CC5: this repo's per-arm cameras ({prefix}d435i_rgb at fovy=42.5 deg,
{prefix}d435i_depth at fovy=62 deg) are separate, non-co-located cameras,
unlike a real D435's aligned-depth-to-color output. Per the approved
critical change, Plan 0's single-camera-name function signatures are used
literally throughout this pipeline -- pick ONE camera per arm (this module
computes intrinsics for whichever camera_name you pass) rather than
reprojecting between the two.

No MJCF <camera> in this scene has a `resolution` attribute (MuJoCo does
not size offscreen buffers that way); resolution is fixed at the Python
`mujoco.Renderer(height=, width=)` call, so width/height must be passed in
explicitly here to match whatever renderer is in use.
"""

from __future__ import annotations

import math

import mujoco
import numpy as np


def get_intrinsics(model: mujoco.MjModel, camera_name: str, width: int, height: int) -> np.ndarray:
    """Return the 3x3 intrinsic matrix K for `camera_name` at the given render resolution.

    fy is derived from the camera's vertical FOV (model.cam_fovy) and pixel
    height, matching the D435 color-stream convention Plan 0 targets
    (fx=fy=615, cx=320, cy=240 at 640x480). fx is set equal to fy (square
    pixel assumption), matching that same real-D435 approximation.
    """
    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
    if cam_id < 0:
        raise ValueError(f"Camera '{camera_name}' not found.")
    fovy_deg = float(model.cam_fovy[cam_id])
    fy = height / (2.0 * math.tan(math.radians(fovy_deg) / 2.0))
    fx = fy
    cx = width / 2.0
    cy = height / 2.0
    return np.array(
        [
            [fx, 0.0, cx],
            [0.0, fy, cy],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

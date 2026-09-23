"""C2: world-to-camera extrinsic + mandatory projection self-test (Mujoco Plan 0, Part C2).

CC5: this module operates on a single camera per arm ("{prefix}d435i_depth"
by default), matching Plan 0's literal one-camera-name function signatures.
"""

from __future__ import annotations

import mujoco
import numpy as np


def get_camera_extrinsic(model: mujoco.MjModel, data: mujoco.MjData, camera_name: str) -> np.ndarray:
    """Return the 4x4 world-to-camera transform, converting MuJoCo's
    (+X right, +Y up, -Z forward) camera convention to OpenCV's
    (+X right, +Y down, +Z forward) convention.

    NOTE (found by C2's own mandatory projection self-test, see
    verify_scene.py): Plan 0's original pseudocode used `cam_xmat` directly
    as if it were already the world-to-camera rotation and only applied the
    axis flip on top of it. `data.cam_xmat` is actually the camera's
    local-to-world rotation (same convention as body `xmat`), so it must be
    transposed to world-to-camera BEFORE the OpenCV axis flip -- skipping the
    transpose produced wildly wrong (thousands-of-pixels) projections in
    testing. This is exactly the "silent error" class Plan 0's own D3
    checklist warns about.
    """
    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
    if cam_id < 0:
        raise ValueError(f"Camera '{camera_name}' not found.")
    pos = data.cam_xpos[cam_id].copy()
    rot_local_to_world = data.cam_xmat[cam_id].reshape(3, 3).copy()
    rot = rot_local_to_world.T  # world-to-camera
    # MuJoCo camera local frame: +X right, +Y up, -Z forward
    # OpenCV frame:               +X right, +Y down, +Z forward
    # Conversion: negate Y (up→down) and Z (-Z→+Z); X is unchanged
    rot[1, :] = -rot[1, :]
    rot[2, :] = -rot[2, :]
    t_world_to_cam = np.eye(4, dtype=np.float64)
    t_world_to_cam[:3, :3] = rot
    t_world_to_cam[:3, 3] = -rot @ pos
    return t_world_to_cam


def project_points(gt_positions: np.ndarray, k_matrix: np.ndarray, t_world_to_cam: np.ndarray) -> np.ndarray:
    """Project Nx3 world-frame points into Nx2 pixel coordinates.

    This is the mandatory self-test from Plan 0's C2: overlay the returned
    pixel coordinates on the RGB frame and confirm they land on the cable
    before trusting anything downstream (B5 point clouds, C3 occlusion
    labels). A convention bug here silently corrupts every later step.
    """
    n = gt_positions.shape[0]
    homogeneous = np.concatenate([gt_positions, np.ones((n, 1))], axis=1)
    cam_frame = (t_world_to_cam @ homogeneous.T).T[:, :3]
    pixels_h = (k_matrix @ cam_frame.T).T
    pixels = pixels_h[:, :2] / pixels_h[:, 2:3]
    return pixels

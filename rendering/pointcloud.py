"""B5: depth + mask -> segmented cable point cloud (Mujoco Plan 0, Part B5).

Pure numpy, no MuJoCo dependency -- reused by ground_truth's C2 self-test
for the B5 wrap-distance verification check.
"""

from __future__ import annotations

import numpy as np


def mask_to_pointcloud(depth_meters: np.ndarray, mask: np.ndarray, k_matrix: np.ndarray) -> np.ndarray:
    fx, fy = k_matrix[0, 0], k_matrix[1, 1]
    cx, cy = k_matrix[0, 2], k_matrix[1, 2]
    vs, us = np.where(mask & (depth_meters > 0))
    zs = depth_meters[vs, us]
    xs = (us - cx) * zs / fx
    ys = (vs - cy) * zs / fy
    return np.stack([xs, ys, zs], axis=-1).astype(np.float32)

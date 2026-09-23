"""C1: ordered ground-truth cable node positions (Mujoco Plan 0, Part C1)."""

from __future__ import annotations

import sys
from pathlib import Path

import mujoco
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "mujoco_scenes"))
from generate_triple_scene import CABLE_SEGMENTS  # noqa: E402


def get_cable_ground_truth(model: mujoco.MjModel, data: mujoco.MjData, cable_prefix: str = "red_cable") -> np.ndarray:
    positions = []
    for i in range(CABLE_SEGMENTS):
        body_name = f"{cable_prefix}_{i:02d}"
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if body_id < 0:
            raise ValueError(f"Body '{body_name}' not found -- check cable_prefix and CABLE_SEGMENTS.")
        positions.append(data.xpos[body_id].copy())
    return np.array(positions, dtype=np.float32)

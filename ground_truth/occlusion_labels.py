"""C3: per-node occlusion labeling via mj_ray (Mujoco Plan 0, Part C3)."""

from __future__ import annotations

import mujoco
import numpy as np

VISIBLE = 0
OCCLUDED = 1

# The D435i visual mesh (group 2, contype=0) sits right at the camera mount
# and would self-occlude every ray at near-zero distance if included -- the
# same issue HANDOFF.md documents for rendering (group 2 is hidden there via
# MjvOption). Exclude group 2 from the ray-cast geom filter for the same
# reason.
_GEOMGROUP_EXCLUDE_VISUAL = np.array([1, 1, 0, 1, 1, 1], dtype=np.uint8)


def label_node_occlusion(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    camera_name: str,
    gt_positions: np.ndarray,
) -> tuple[np.ndarray, list[str | None]]:
    """Ray-cast from the camera through each ground-truth node.

    Returns (labels, occluder_names): labels[i] is VISIBLE or OCCLUDED;
    occluder_names[i] is the geom name that caused occlusion (None if
    visible), classified as "cable" (self-occlusion), a "clip_"-prefixed
    fixture geom (A5), or any other occluder (table/gripper/etc).
    """
    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
    if cam_id < 0:
        raise ValueError(f"Camera '{camera_name}' not found.")
    cam_pos = data.cam_xpos[cam_id].copy()

    labels = []
    occluder_names: list[str | None] = []
    geom_id_arr = np.zeros(1, dtype=np.int32)
    for p_i in gt_positions:
        direction = p_i - cam_pos
        norm = np.linalg.norm(direction)
        if norm < 1e-9:
            labels.append(OCCLUDED)
            occluder_names.append(None)
            continue
        direction = direction / norm

        dist = mujoco.mj_ray(
            model, data, cam_pos, direction,
            _GEOMGROUP_EXCLUDE_VISUAL,  # geomgroup filter: all but the D435i visual mesh (group 2)
            1,     # flg_static
            -1,    # bodyexclude: none
            geom_id_arr,
        )
        hit_geom_id = int(geom_id_arr[0])
        if hit_geom_id < 0 or dist < 0:
            labels.append(OCCLUDED)
            occluder_names.append(None)
            continue

        hit_geom_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, hit_geom_id)
        if hit_geom_name and "cable" in hit_geom_name:
            hit_point = cam_pos + direction * dist
            if np.linalg.norm(hit_point - p_i) < 0.02:
                labels.append(VISIBLE)
                occluder_names.append(None)
            else:
                labels.append(OCCLUDED)
                occluder_names.append(hit_geom_name)
        else:
            labels.append(OCCLUDED)
            occluder_names.append(hit_geom_name)

    return np.array(labels, dtype=np.int32), occluder_names

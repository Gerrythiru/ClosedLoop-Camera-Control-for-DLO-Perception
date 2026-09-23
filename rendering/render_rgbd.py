"""B2: RGB + depth rendering utility (Mujoco Plan 0, Part B2).

CC4: Plan 0's original B2 assumes MuJoCo returns a raw OpenGL z-buffer
needing a nonlinear remap (`znear*zfar/(zfar - buf*(zfar-znear))`) to get
metric depth. This repo's `mujoco.Renderer` already returns linear metric
depth directly when depth rendering is enabled (confirmed in the pre-existing
extract_camera_data.py and documented in HANDOFF.md's camera notes). Do NOT
apply that remap here -- doing so on top of already-linear depth would
double-convert and corrupt every downstream point cloud. This function
returns the renderer's depth output as-is.
"""

from __future__ import annotations

import mujoco
import numpy as np


def render_rgbd(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    renderer: mujoco.Renderer,
    camera_name: str,
    scene_option: mujoco.MjvOption | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Render one (rgb, depth_meters) pair from `camera_name`.

    depth_meters is already linear metric depth -- see the CC4 note above.
    """
    renderer.update_scene(data, camera=camera_name, scene_option=scene_option)

    renderer.disable_depth_rendering()
    rgb = renderer.render().copy()

    renderer.enable_depth_rendering()
    depth_meters = renderer.render().copy()
    renderer.disable_depth_rendering()

    return rgb, depth_meters


def hide_visual_only_geoms() -> mujoco.MjvOption:
    """Scene option hiding group-2 visual-only geoms (arm meshes, D435i
    housing) so a camera does not see its own mount/housing.
    """
    opt = mujoco.MjvOption()
    opt.geomgroup[2] = 0
    return opt

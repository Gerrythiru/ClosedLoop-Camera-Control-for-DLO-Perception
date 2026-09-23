"""DLO cable dangle task entry point (v5 -- forked from run_dlo_v3_freeze.py,
points at the v5 scene and dual-arm dlo_route_v5_freeze.DanglePlanner: r2
grasps the cable's leftmost node and carries it through 4 waypoints while r3
*simultaneously* grasps the rightmost node and carries it to a single fixed
hold position, both overlapping r1's own 140s wrist-camera choreography
instead of running only after it (V3's freeze sequence only ran a 140s
settle with r2/r3 idle; V5 restores real dual-arm manipulation. V3's own
files are untouched).

Usage:

    python run_dlo_v5_freeze.py            # headless: r1's eye-in-hand capture only
    python run_dlo_v5_freeze.py --view     # also opens a live MuJoCo viewer window,
                                     # paced to roughly real time, while the
                                     # sequence runs (close the window to stop
                                     # early)

Uses generate_triple_scene_v5.py / dlo_route_v5_freeze.py -- fully
independent of the v2/v3 pipelines, so scene edits here don't affect them.

During each of r1's 3 camera-choreography pose holds (A/B/C, durations
vary -- see r1_camera_ik.py's SCHEDULE), captures RGB+depth image pairs at
30Hz from r1's own wrist camera (r1_d435i_rgb / r1_d435i_depth) plus
per-capture camera pose + r1 qpos + camera intrinsics, to
results/r1_eye_in_hand_calib_v5/pose_{A,B,C}/ -- this is the sole capture
pipeline now; the world-fixed cable_side/cable_init_top cameras still
exist in the scene (viewable live with --view, Tab to cycle) but are no
longer recorded to disk -- the tracking pipeline
(trackdlo_eval/run_r1_tracking.py) is built entirely on r1's own moving
camera. See r1_camera_capture.py.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import mujoco
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "mujoco_scenes"))
sys.path.insert(0, str(PROJECT_ROOT / "trajectories"))
sys.path.insert(0, str(PROJECT_ROOT / "rendering"))
from generate_triple_scene_v5 import generate  # noqa: E402
from dlo_route_v5_freeze import DanglePlanner, _GRASP_TOOL_RAW_CENTER_M  # noqa: E402
from generate_triple_scene_v5 import _GRASP_TOOL_RAW_EXTENT_M  # noqa: E402
from r1_camera_ik_v5 import CameraArm, set_solved_waypoints, target_at, rail_at, hold_window_at, hold_windows, NBVCameraScheduler  # noqa: E402
from r1_camera_capture_v5 import R1CalibCapture, RGB_CAMERA, CAPTURE_WIDTH, CAPTURE_HEIGHT  # noqa: E402
from camera_intrinsics import get_intrinsics  # noqa: E402

# r1's wrist-camera choreography joint waypoints -- solved offline (see
# r1_camera_ik_v5.py's module docstring) by
# trajectories/verify_r1_camera_reachability_v5.py.
# TEMP (2026-09-08): A -> B only (Pose C disabled). r1 held STATIONARY at
# the fixed rail RAIL_X = -0.025 (base_x = 0.175). Schedule: A holds to
# t=36s; A->B xfer [36,38] (arrives B by t=38); B holds [38,140]. QPOS_C is
# a placeholder (unused).
#
# 2026-09-09 raised/pushed-back Pose A camera (r1_camera_ik_v5.py) -- this
# STILL diverged (node0 runaway to ~340mm from t~25.5s, see Today_SUMM).
# Was reinstated 2026-09-10 as the active pose (see below for why); kept
# here, commented, as the reference/fallback:
# set_solved_waypoints(
#     np.array([2.329373, -0.311831, 1.419447, 2.430444, -1.451699, -0.377552]),  # Pose A
#     np.array([0.281596, 0.223741, 1.582207, 0.901388, 1.437914, -0.173255]),  # Pose B
#     np.zeros(6),  # Pose C (disabled)
# )
#
# 2026-09-12: Pose A replaced again for the invoke_nbv mitigation-validation
# test (step 4, see Today_SUMM) -- user-supplied, deliberately chosen for
# genuine geometric occlusion (not the color blind spot the candidate below
# ran into). verify_r1_camera_reachability_v5.py: REACHABLE + collision-free
# (A 1.61mm/1.65deg, B 4.52mm/1.10deg), clean home->A/A->B transitions
# (0/300 each). Moving-collision check (e4b-style) not yet re-run for this
# pose. Labeled "Occlusion_Test_pose_1" -- superseded 2026-09-14 (see
# below). Kept here, commented, as a fallback (unchanged RAIL_X = -0.025):
# set_solved_waypoints(
#     np.array([2.796292, 0.029889, 0.781671, -1.065743, 1.356854, -2.349739]),  # Pose A
#     np.array([0.276843, 0.236248, 1.636403, -2.267674, -1.496076, -3.292497]),  # Pose B
#     np.zeros(6),  # Pose C (disabled)
# )
#
# 2026-09-14: POSE_A replaced with "Occlusion_Test_pose_2", at the SAME
# fixed RAIL_X = -0.025 as Pose B -- verify_r1_camera_reachability_v5.py:
# REACHABLE + collision-free (A 2.79mm/1.15deg, B 4.52mm/1.10deg -- Pose B
# unchanged, same rail), clean home->A/A->B transitions (0/300 each), and
# e4b_moving_collision_check_v5.py confirmed no collision with the live
# r2/r3 carry.
# set_solved_waypoints(
#     np.array([3.033264, 0.325509, 0.830141, 2.099574, -1.515221, 1.110904]),  # Pose A
#     np.array([0.276843, 0.236248, 1.636403, 4.015511, -1.496076, 2.990689]),  # Pose B
#     np.zeros(6),  # Pose C (disabled)
# )
#
# 2026-09-21: POSE_A replaced again, user-supplied -- same fixed
# RAIL_X = -0.025. verify_r1_camera_reachability_v5.py: REACHABLE +
# collision-free (A 1.69mm/1.01deg, B 4.47mm/1.08deg), clean home->A/A->B
# transitions (0/300 each). Moving-collision check (e4b-style) not yet
# re-run for this pose.
set_solved_waypoints(
    np.array([2.858189, 0.308078, 0.864415, -1.165459, 1.376274, -2.130289]),  # Pose A
    np.array([0.281769, 0.223724, 1.582085, -2.240125, -1.437627, -3.314638]),  # Pose B
    np.zeros(6),  # Pose C (disabled)
)
#
# 2026-09-10 candidate tried, e4_reachable_pose_sweep_v5.py's (Q2) winning
# pick -- 99% both-grasped-ends-visible by the sweep's geometric-occlusion
# test, REACHABLE + collision-free at production tolerance (A 4.37mm/1.10deg,
# B 4.52mm/1.10deg), clean home->A/A->B transitions (0/300 each), and
# e4b_moving_collision_check_v5.py confirmed no collision with the live
# r2/r3 carry (0/340 samples) -- but didn't fix the real failure (see
# above). Kept here, commented, in case a geometric-occlusion scenario
# needs it again:
# set_solved_waypoints(
#     np.array([0.441072, -0.032921, 1.30003, 0.91137, 1.318023, -0.102863]),  # Pose A
#     np.array([0.276843, 0.236248, 1.636403, 4.015511, -1.496076, 2.990689]),  # Pose B
#     np.zeros(6),  # Pose C (disabled)
# )

FRAME_DT = 1.0 / 30.0
BORE_QUAT_PRINT_PERIOD_S = 1.0  # throttle for the "B" live bore-quaternion toggle

# 2026-09-18: raw-mesh-local center of each of the tool's 6 bounding-box
# faces (same raw frame _GRASP_TOOL_RAW_CENTER_M already uses -- add
# tool_xpos + tool_xmat @ offset to get a world point, same pattern as
# _bore_center_world). Named by which raw-mesh axis they're on, not by
# any assumed real-world direction -- the tool's mounted orientation
# (GRASP_TOOL_LOCAL_QUAT) determines which of these ends up "up" once
# mounted, which is exactly what the user is trying to pin down by
# comparing the markers against the live viewer, so this deliberately
# doesn't presuppose an answer. z_min/z_max are the two flat
# circular-cut faces (bore axis = raw mesh Z, established earlier this
# session); x_min/x_max/y_min/y_max are the ~22mm square's side edges.
_ex, _ey, _ez = _GRASP_TOOL_RAW_EXTENT_M
FACE_OFFSETS_LOCAL = {
    "x_min": np.array([0.0, _ey / 2, _ez / 2]),
    "x_max": np.array([_ex, _ey / 2, _ez / 2]),
    "y_min": np.array([_ex / 2, 0.0, _ez / 2]),
    "y_max": np.array([_ex / 2, _ey, _ez / 2]),
    "z_min": np.array([_ex / 2, _ey / 2, 0.0]),
    "z_max": np.array([_ex / 2, _ey / 2, _ez]),
}
FACE_COLORS_RGBA = {
    "x_min": (1.0, 0.0, 0.0, 1.0),   # red
    "x_max": (0.0, 1.0, 1.0, 1.0),   # cyan
    "y_min": (0.0, 1.0, 0.0, 1.0),   # green
    "y_max": (1.0, 0.0, 1.0, 1.0),   # magenta
    "z_min": (0.1, 0.3, 1.0, 1.0),   # blue
    "z_max": (1.0, 0.55, 0.0, 1.0),  # orange
}
FACE_MARKER_RADIUS_M = 0.004  # a bit smaller than BORE_MARKER_RADIUS_M so
                               # the 6 face markers + the bore-center marker
                               # stay visually distinguishable when both are on


def _bore_center_world(model: mujoco.MjModel, data: mujoco.MjData, prefix: str) -> np.ndarray:
    """World-frame position of `prefix`'s grasp tool's bore-center
    midpoint (see _GRASP_TOOL_RAW_CENTER_M) -- recomputed live every
    call. Shared by _bore_relative_pose and the "B"-toggle marker in
    main()."""
    tool_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{prefix}grasp_tool")
    tool_pos = data.xpos[tool_bid]
    tool_mat = data.xmat[tool_bid].reshape(3, 3)
    return tool_pos + tool_mat @ _GRASP_TOOL_RAW_CENTER_M


def _face_centers_world(model: mujoco.MjModel, data: mujoco.MjData, prefix: str) -> dict[str, np.ndarray]:
    """{face_name: world_pos} for `prefix`'s grasp tool's 6 bounding-box
    face centers (see FACE_OFFSETS_LOCAL) -- recomputed live every call."""
    tool_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{prefix}grasp_tool")
    tool_pos = data.xpos[tool_bid]
    tool_mat = data.xmat[tool_bid].reshape(3, 3)
    return {face: tool_pos + tool_mat @ offset for face, offset in FACE_OFFSETS_LOCAL.items()}


def _bore_relative_pose(model: mujoco.MjModel, data: mujoco.MjData, prefix: str) -> tuple[np.ndarray, np.ndarray]:
    """(rel_pos, rel_quat(wxyz)) of the grasp tool's bore-center midpoint
    relative to `prefix`'s base joint (joint1, which lives in body
    `{prefix}link1` -- see vendor/mujoco_menagerie/ufactory_lite6/lite6.xml:
    joint1 has no pos offset, so link1's own frame IS the joint's frame).
    Recomputed live from the current data.xpos/xquat/xmat every call --
    used by run_dlo_v5_freeze.py's --view "B" key toggle so the printed
    value tracks the arm's actual current configuration, not a one-time
    snapshot, letting the user visually cross-check it against the live
    tool pose in the viewer."""
    base_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{prefix}link1")
    base_pos = data.xpos[base_bid]
    base_mat = data.xmat[base_bid].reshape(3, 3)
    base_quat = data.xquat[base_bid]
    tool_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{prefix}grasp_tool")
    tool_quat = data.xquat[tool_bid]
    bore_center_world = _bore_center_world(model, data, prefix)
    rel_pos = base_mat.T @ (bore_center_world - base_pos)
    base_quat_inv = np.zeros(4)
    mujoco.mju_negQuat(base_quat_inv, base_quat)
    rel_quat = np.zeros(4)
    mujoco.mju_mulQuat(rel_quat, base_quat_inv, tool_quat)
    return rel_pos, rel_quat


# r1's static camera-viewing home pose, held before the settle sequence
# starts and for the first 2s of it, before r1 starts moving toward Pose A
# (see trajectories/r1_camera_ik.py's SCHEDULE).
R1_HOME = {1: 0.5, 2: -0.9, 3: 1.5, 4: 0.0, 5: -1.25, 6: 0.0}


BORE_MARKER_RADIUS_M = 0.006   # visual sphere radius -- a bit larger than
                                # CABLE_RADIUS so it's easy to spot but still
                                # reads as a point at this scene's scale
BORE_MARKER_RGBA = (1.0, 0.9, 0.0, 1.0)  # bright yellow -- "lit up"
PURPLE_MARKER_RADIUS_M = 0.005
PURPLE_MARKER_RGBA = (0.6, 0.1, 0.9, 1.0)  # purple -- live TrackDLO-tracked nodes (see --ros-live)


def _add_sphere_marker(scn, pos: np.ndarray, radius: float, rgba) -> None:
    mujoco.mjv_initGeom(
        scn.geoms[scn.ngeom],
        type=mujoco.mjtGeom.mjGEOM_SPHERE,
        size=np.array([radius, 0, 0]),
        pos=pos,
        mat=np.eye(3).flatten(),
        rgba=np.array(rgba, dtype=np.float32),
    )
    scn.ngeom += 1


# 2026-09-22: --nbv's gripper-visibility trigger. 8 bounding-box corners of
# the grasp-tool mesh, tool-local frame (same corner-origin convention as
# _GRASP_TOOL_RAW_CENTER_M/FACE_OFFSETS_LOCAL above -- reuses the already-
# unpacked _ex/_ey/_ez rather than re-deriving them).
_TOOL_CORNERS_LOCAL = np.array([
    [x, y, z] for x in (0.0, _ex) for y in (0.0, _ey) for z in (0.0, _ez)
])
GRIPPER_VISIBILITY_FRACTION_THRESHOLD = 0.5  # fraction of the 8 corners that must be visible
GRIPPER_HIDDEN_TRIGGER_S = 2.0               # advance once a gripper's been below that for this long
NBV_RECOVERY_CHECK_DELAY_S = 2.0             # one-shot visibility-recovery check, this long into a new hold


def _point_visible(model: mujoco.MjModel, data: mujoco.MjData, cam_id: int,
                    point_world: np.ndarray, tol: float = 0.01) -> bool:
    """True iff point_world is inside cam_id's frustum (in front of the
    camera, within pixel bounds) AND unoccluded by ANY scene geometry --
    deliberately including r1's own arm/camera mount (no self-occlusion
    skip, unlike dump_v5_ground_truth.py's label_node_occlusion_wrist,
    since a real camera would genuinely be blocked by its own mount too).
    Frustum-projection math mirrors label_node_occlusion_wrist's; the
    ray-cast itself is a single mujoco.mj_ray call with no skip-loop."""
    cam_pos = data.cam_xpos[cam_id].copy()
    rot_l2w = data.cam_xmat[cam_id].reshape(3, 3)
    R = rot_l2w.T.copy()
    R[1] = -R[1]
    R[2] = -R[2]
    t_vec = -R @ cam_pos
    p_cam = R @ point_world + t_vec
    if p_cam[2] <= 0:
        return False
    fovy_rad = math.radians(model.cam_fovy[cam_id])
    fy = CAPTURE_HEIGHT / (2.0 * math.tan(fovy_rad / 2.0))
    cx, cy = CAPTURE_WIDTH / 2.0, CAPTURE_HEIGHT / 2.0
    u = fy * p_cam[0] / p_cam[2] + cx
    v = fy * p_cam[1] / p_cam[2] + cy
    if u < 0 or u >= CAPTURE_WIDTH or v < 0 or v >= CAPTURE_HEIGHT:
        return False

    direction = point_world - cam_pos
    dist_to_point = float(np.linalg.norm(direction))
    if dist_to_point < 1e-9:
        return True
    direction = direction / dist_to_point
    geomgroup = np.ones(6, dtype=np.uint8)
    gid_arr = np.zeros(1, dtype=np.int32)
    dist = mujoco.mj_ray(model, data, cam_pos, direction, geomgroup, 1, -1, gid_arr)
    if dist < 0:
        return False
    return abs(dist - dist_to_point) < tol


def _gripper_visibility_fraction(model: mujoco.MjModel, data: mujoco.MjData, cam_id: int, prefix: str) -> float:
    """Fraction of `prefix`'s grasp-tool bounding-box corners (see
    _TOOL_CORNERS_LOCAL) currently visible from cam_id."""
    tool_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{prefix}grasp_tool")
    tool_pos = data.xpos[tool_bid]
    tool_mat = data.xmat[tool_bid].reshape(3, 3)
    visible = sum(
        1 for corner in _TOOL_CORNERS_LOCAL
        if _point_visible(model, data, cam_id, tool_pos + tool_mat @ corner)
    )
    return visible / len(_TOOL_CORNERS_LOCAL)


def _make_on_frame(viewer_handle, show_bore_quats: list[bool], show_face_markers: list[bool],
                    pace: bool = False, live_tracking_manager=None,
                    show_live_markers: list[bool] | None = None) -> callable:
    """Live-viewer sync + real-time pacing, if a viewer handle is given;
    a no-op callback otherwise.

    `show_bore_quats` (toggled by "B") -- while True: (a) prints r2's and
    r3's live bore-center-relative-to-base-joint quaternion (see
    _bore_relative_pose) to the terminal every BORE_QUAT_PRINT_PERIOD_S of
    simulated time, and (b) draws a yellow sphere at each tool's live
    bore-center world position.

    `show_face_markers` (toggled by "F", independent of "B" -- both can be
    on at once) -- while True, draws one distinctly-colored sphere per
    tool at each of its 6 raw-mesh bounding-box face centers (see
    FACE_OFFSETS_LOCAL/FACE_COLORS_RGBA), so the user can identify which
    face is which in the live viewer and describe corrections in those
    terms (e.g. "move it toward the blue face") instead of guessing axis
    names blind.

    `live_tracking_manager`/`show_live_markers` (toggled by "L", 2026-09-21,
    see --ros-live) -- while show_live_markers[0] is True and a manager is
    given, draws one purple sphere per TrackDLO-tracked node at its live
    world position (trajectories/r1_live_tracking_bridge.py). This reads
    from mujoco.Renderer's OFFSCREEN camera capture path -- drawing into
    viewer_handle.user_scn here is a completely separate, passive-viewer-
    only scene, so these markers cannot leak into what the camera actually
    captures/publishes to ROS.

    All markers are redrawn every frame (not throttled) so they visibly
    track the tool as the arms move, and are cleared the instant their
    toggle goes False.

    `pace` (2026-09-21): wall-clock throttling to ~30Hz was previously only
    ever active when a viewer was open; --ros-live needs it too (so ROS
    message timestamps correspond to real time and trackdlo can keep up),
    independent of --view. Marker drawing stays gated on the viewer
    existing (scn is not None); pacing is now a separate, independent
    gate -- no behavior change when --view is passed without --ros-live
    (pace is still True only because args.view is True, same as before)."""
    last_wall = [time.time()]
    last_print_t = [-1e9]

    def on_frame(model: mujoco.MjModel, data: mujoco.MjData, t: float) -> None:
        if show_bore_quats[0] and t - last_print_t[0] >= BORE_QUAT_PRINT_PERIOD_S:
            last_print_t[0] = t
            for prefix in ("r2_", "r3_"):
                rel_pos, rel_quat = _bore_relative_pose(model, data, prefix)
                print(f"[dlo] t={t:.2f} {prefix}bore-center rel. to {prefix}link1: "
                      f"pos={np.round(rel_pos, 5).tolist()} quat(wxyz)={np.round(rel_quat, 5).tolist()}",
                      flush=True)
        if viewer_handle is not None:
            if not viewer_handle.is_running():
                raise RuntimeError("viewer window closed")
            scn = viewer_handle.user_scn
            if scn is not None:
                scn.ngeom = 0
                if show_bore_quats[0]:
                    for prefix in ("r2_", "r3_"):
                        _add_sphere_marker(scn, _bore_center_world(model, data, prefix),
                                            BORE_MARKER_RADIUS_M, BORE_MARKER_RGBA)
                if show_face_markers[0]:
                    for prefix in ("r2_", "r3_"):
                        for face, pos in _face_centers_world(model, data, prefix).items():
                            _add_sphere_marker(scn, pos, FACE_MARKER_RADIUS_M, FACE_COLORS_RGBA[face])
                if (live_tracking_manager is not None and show_live_markers is not None
                        and show_live_markers[0]):
                    nodes_world = live_tracking_manager.latest_markers_world()
                    if nodes_world is not None:
                        for pos in nodes_world:
                            _add_sphere_marker(scn, pos, PURPLE_MARKER_RADIUS_M, PURPLE_MARKER_RGBA)
            viewer_handle.sync()
        if pace:
            elapsed = time.time() - last_wall[0]
            remaining = FRAME_DT - elapsed
            if remaining > 0:
                time.sleep(remaining)
            last_wall[0] = time.time()

    return on_frame


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--view", action="store_true",
        help="Open a live MuJoCo viewer window while the sequence runs, paced to ~30Hz real time.",
    )
    parser.add_argument(
        "--ros-live", action="store_true",
        help="Live-publish r1's captured RGB+depth frames to ROS/trackdlo during each hold, "
             "in addition to the existing disk save (see trajectories/r1_live_tracking_bridge.py "
             "and OFlinevsONline.md). Requires rospy/roslaunch -- run this under the ros_noetic "
             "conda env with ~/tracking_ws sourced, NOT this project's venv. Implies wall-clock "
             "pacing even without --view.",
    )
    parser.add_argument(
        "--nbv", action="store_true",
        help="Variable-duration camera holds: Pose A/B stay the existing scripted waypoints/order "
             "(trajectories/r1_camera_ik_v5.py's NBVCameraScheduler), but each hold's END is driven "
             "by a gripper-visibility signal -- r2's or r3's grasp-tool geometry hidden from the "
             "camera for more than 2s -- instead of a fixed clock. Proves open-ended-hold plumbing "
             "ahead of a future real NBV algorithm; no safety cap, so a pose that never loses "
             "gripper visibility holds indefinitely (accepted trade-off). Implies --ros-live -- "
             "an --nbv run always publishes live to ROS/trackdlo.",
    )
    args = parser.parse_args()
    if args.nbv:
        args.ros_live = True
    sys.stdout.reconfigure(line_buffering=True)  # stream progress prints live instead of buffering

    generate(layout="dlo")
    model = mujoco.MjModel.from_xml_path(str(PROJECT_ROOT / "triple_lite6_cable_routing_dlo_v5.xml"))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    # Set r1 to a safe camera-viewing pose before settling.
    # Arm joint actuators are unnamed in the Lite6 model -- look them up by joint ID.
    _r1_actuator_ids: dict[int, int] = {}
    for j_num in R1_HOME:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"r1_joint{j_num}")
        for aid in range(model.nu):
            if int(model.actuator_trntype[aid]) == 0 and int(model.actuator_trnid[aid, 0]) == jid:
                _r1_actuator_ids[j_num] = aid
                break
    for j_num, angle in R1_HOME.items():
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"r1_joint{j_num}")
        data.qpos[model.jnt_qposadr[jid]] = angle
        data.ctrl[_r1_actuator_ids[j_num]] = angle
    mujoco.mj_forward(model, data)

    live_tracking_manager = None
    if args.ros_live:
        import rospy
        from r1_live_tracking_bridge import LiveTrackingManager
        rospy.init_node("run_dlo_v5_live_bridge")
        k_matrix = get_intrinsics(model, RGB_CAMERA, CAPTURE_WIDTH, CAPTURE_HEIGHT)
        live_tracking_manager = LiveTrackingManager(k_matrix, CAPTURE_HEIGHT, CAPTURE_WIDTH)

    viewer_handle = None
    show_bore_quats = [False]    # toggled by the "B" key below; read by _make_on_frame
    show_face_markers = [False]  # toggled by the "F" key below; read by _make_on_frame
    show_live_markers = [True]   # toggled by the "L" key below; read by _make_on_frame -- on by
                                  # default when --ros-live is active (see --ros-live's help)
    if args.view:
        import mujoco.viewer as mj_viewer

        _cam_names = ["cable_lateral", "cable_side", "cable_init_top", "r1_d435i_rgb", "box_focus", "fixture_iso", "overview"]
        _cam_idx = [0]

        def _key_callback(key: int) -> None:
            # Tab (GLFW 258) cycles through all scene cameras
            if key == 258:
                _cam_idx[0] = (_cam_idx[0] + 1) % len(_cam_names)
                name = _cam_names[_cam_idx[0]]
                cid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, name)
                viewer_handle.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
                viewer_handle.cam.fixedcamid = cid
                print(f"[dlo] camera → {name}", flush=True)
            # P (GLFW 80) prints the current free-camera pose (Esc drops
            # into free camera first). Prints the raw orbit parameters
            # (lookat/distance/azimuth/elevation) -- these are the exact,
            # unconverted mjvCamera fields, not a derived pos/xyaxes MJCF
            # string (that conversion needs mujoco's own math -- use the
            # viewer's built-in "Copy camera" sidebar button, under
            # Visualization, for that instead; this is a quick terminal
            # readout for iterating on a pose live).
            elif key == 80:
                cam = viewer_handle.cam
                if cam.type != mujoco.mjtCamera.mjCAMERA_FREE:
                    print("[dlo] not in free camera mode -- press Esc first, then P", flush=True)
                else:
                    print(f"[dlo] free camera pose: lookat={cam.lookat.tolist()} "
                          f"distance={cam.distance:.4f} azimuth={cam.azimuth:.2f} "
                          f"elevation={cam.elevation:.2f}", flush=True)
            # B (GLFW 66) toggles a live terminal readout of r2's/r3's grasp
            # tool bore-center midpoint, relative to each arm's own base
            # joint (link1), printed as pos + quaternion (see
            # _bore_relative_pose) -- lets the quaternion given earlier be
            # cross-checked against the tool's actual live orientation
            # while watching the viewer, rather than trusting a one-off
            # offline computation. Also lights up a small yellow sphere
            # at each tool's live bore-center world position (see
            # _make_on_frame) so the midpoint itself is visible, not just
            # printed as numbers.
            elif key == 66:
                show_bore_quats[0] = not show_bore_quats[0]
                state = "ON" if show_bore_quats[0] else "OFF"
                print(f"[dlo] bore-center quaternion readout + marker: {state}", flush=True)
                if show_bore_quats[0]:
                    for prefix in ("r2_", "r3_"):
                        rel_pos, rel_quat = _bore_relative_pose(model, data, prefix)
                        print(f"[dlo] t=(live) {prefix}bore-center rel. to {prefix}link1: "
                              f"pos={np.round(rel_pos, 5).tolist()} quat(wxyz)={np.round(rel_quat, 5).tolist()}",
                              flush=True)
            # F (GLFW 70) toggles 6 distinctly-colored per-tool markers, one
            # at the center of each of the tool's raw-mesh bounding-box
            # faces (see FACE_OFFSETS_LOCAL/FACE_COLORS_RGBA) -- independent
            # of the "B" toggle, so both can be on together. Use these to
            # describe corrections to the bore-center point in terms of
            # which colored face it should move toward/away from, instead
            # of guessing axis directions blind.
            elif key == 70:
                show_face_markers[0] = not show_face_markers[0]
                state = "ON" if show_face_markers[0] else "OFF"
                print(f"[dlo] per-face markers: {state} "
                      f"(x_min=red x_max=cyan y_min=green y_max=magenta z_min=blue z_max=orange)", flush=True)
            # L (GLFW 76) toggles the purple live-TrackDLO-tracked-node
            # markers (see --ros-live / trajectories/r1_live_tracking_bridge.py).
            # No-op (but harmless) if --ros-live wasn't passed.
            elif key == 76:
                show_live_markers[0] = not show_live_markers[0]
                print(f"[dlo] live-tracked purple markers: {'ON' if show_live_markers[0] else 'OFF'}", flush=True)

        viewer_handle = mj_viewer.launch_passive(model, data, key_callback=_key_callback)

        # Group 4 = r1's own visual meshes/D435i housing (see
        # generate_triple_scene_v5.py's add_robot_instances/
        # add_cameras_to_arms) -- off by default in MuJoCo's default
        # MjvOption (only groups 0-2 are on by default), so without this
        # r1 is invisible in the live viewer even though it's driven and
        # moving. R1CalibCapture's own offscreen renders already handle
        # this correctly via hide_own_arm_geoms(); the interactive viewer
        # needs the same group turned on explicitly since it starts from
        # plain defaults.
        viewer_handle.opt.geomgroup[4] = 1

        # Start on cable_lateral
        cable_lateral_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "cable_lateral")
        viewer_handle.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
        viewer_handle.cam.fixedcamid = cable_lateral_id
        print("[dlo] live viewer open -- Tab to cycle cameras (cable_lateral / cable_side / cable_init_top / r1_d435i_rgb / box_focus / fixture_iso / overview), "
              "B to toggle r2/r3 bore-center quaternion readout + marker, "
              "F to toggle per-face colored markers, "
              "L to toggle live-tracked purple markers. Close window to stop early.")

    for step in range(500):
        for j_num, angle in R1_HOME.items():
            data.ctrl[_r1_actuator_ids[j_num]] = angle
        mujoco.mj_step(model, data)
        if viewer_handle is not None and step % 20 == 0:
            if not viewer_handle.is_running():
                viewer_handle.close()
                print("[dlo] viewer closed during settling -- exiting.")
                return 0
            viewer_handle.sync()

    planner = DanglePlanner(model, data)
    r1_cam_arm = CameraArm(model, data)
    r1_capture = R1CalibCapture(
        model, r1_cam_arm, PROJECT_ROOT / "results" / "r1_eye_in_hand_calib_v5",
        pose_windows=(None if args.nbv else hold_windows()),
        live_tracking_manager=live_tracking_manager,
        open_ended=args.nbv,
    )
    r1_rail_aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "r1_rail_x")

    nbv_scheduler = NBVCameraScheduler() if args.nbv else None
    nbv_state = {"hold_start": None, "last_visible_t": {}, "recovery_checked": True}
    nbv_transitions: list[dict] = []

    def _combined_on_frame(model, data, t):
        if not args.nbv:
            target_qpos = target_at(t)
            if target_qpos is not None:
                r1_cam_arm.set_ctrl_to_qpos(target_qpos)
            data.ctrl[r1_rail_aid] = rail_at(t)
            window = hold_window_at(t)
            r1_capture.maybe_capture(model, data, t, window[0] if window is not None else None)
            base_on_frame(model, data, t)
            return

        qpos, pose_label = nbv_scheduler.step(t)
        if qpos is not None:
            r1_cam_arm.set_ctrl_to_qpos(qpos)
        data.ctrl[r1_rail_aid] = rail_at(t)
        r1_capture.maybe_capture_open_ended(model, data, t, pose_label)

        hold_start = nbv_scheduler.current_hold_start()
        if hold_start is not None:
            if nbv_state["hold_start"] != hold_start:
                # Just entered a new hold -- reset visibility timers + arm the
                # one-shot recovery check, so stale state from the PREVIOUS
                # hold (whose last gripper sighting could be seconds old --
                # that's exactly what triggered this transition) can't
                # immediately re-fire the advance signal again.
                nbv_state["hold_start"] = hold_start
                nbv_state["last_visible_t"] = {"r2_": hold_start, "r3_": hold_start}
                nbv_state["recovery_checked"] = False

            rgb_cam_id = r1_capture.rgb_cam_id
            fracs = {p: _gripper_visibility_fraction(model, data, rgb_cam_id, p) for p in ("r2_", "r3_")}
            for prefix, frac in fracs.items():
                if frac >= GRIPPER_VISIBILITY_FRACTION_THRESHOLD:
                    nbv_state["last_visible_t"][prefix] = t
            for prefix in ("r2_", "r3_"):
                if t - nbv_state["last_visible_t"][prefix] > GRIPPER_HIDDEN_TRIGGER_S:
                    print(f"[dlo-nbv] {prefix}grasp_tool hidden >{GRIPPER_HIDDEN_TRIGGER_S}s "
                          f"at t={t:.2f} -- advancing past pose {pose_label}", flush=True)
                    nbv_scheduler.notify_advance(t)
                    break

            if not nbv_state["recovery_checked"] and t - hold_start >= NBV_RECOVERY_CHECK_DELAY_S:
                nbv_state["recovery_checked"] = True
                recovered = all(f >= GRIPPER_VISIBILITY_FRACTION_THRESHOLD for f in fracs.values())
                nbv_transitions.append({
                    "pose": pose_label, "t": float(t), "recovered": recovered,
                    "frac_r2": float(fracs["r2_"]), "frac_r3": float(fracs["r3_"]),
                })
                print(f"[dlo-nbv] visibility recovery check at pose {pose_label}: recovered={recovered} "
                      f"(r2={fracs['r2_']:.2f} r3={fracs['r3_']:.2f})", flush=True)

        base_on_frame(model, data, t)

    base_on_frame = _make_on_frame(
        viewer_handle, show_bore_quats, show_face_markers,
        pace=(args.view or args.ros_live),
        live_tracking_manager=live_tracking_manager, show_live_markers=show_live_markers,
    )

    print("[dlo] running v5 sequence: r2+r3 grasp/carry the cable's two ends "
          "concurrently with r1's 140s camera choreography A->B->C...")
    try:
        planner.run_sequence(on_frame=_combined_on_frame)
    except RuntimeError as exc:
        print(f"[dlo] {exc} -- stopping early.")
    else:
        print("[dlo] sequence complete.")

    if viewer_handle is not None and viewer_handle.is_running():
        viewer_handle.close()

    r1_capture.save_all_metadata()
    r1_capture.close()
    print(f"[dlo] r1 eye-in-hand calibration captures saved to {PROJECT_ROOT / 'results' / 'r1_eye_in_hand_calib_v5'}")

    if live_tracking_manager is not None:
        live_tracking_manager.save_output()
        live_tracking_manager.shutdown()

    if args.nbv:
        out_path = PROJECT_ROOT / "results" / "r1_dlo_tracking_v5_live" / "nbv_transitions.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(nbv_transitions, f, indent=2)
        print(f"[dlo-nbv] saved {len(nbv_transitions)} transition record(s) to {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""DLO cable dangle task entry point (v4 -- forked from run_dlo_v3_freeze.py,
points at the v4 scene and dual-arm dlo_route_v4_freeze.DanglePlanner: r2
grasps the cable's leftmost node and carries it through 4 waypoints while r3
*simultaneously* grasps the rightmost node and carries it to a single fixed
hold position, both overlapping r1's own 140s wrist-camera choreography
instead of running only after it (V3's freeze sequence only ran a 140s
settle with r2/r3 idle; V4 restores real dual-arm manipulation. V3's own
files are untouched).

Usage:

    python run_dlo_v4_freeze.py            # headless: r1's eye-in-hand capture only
    python run_dlo_v4_freeze.py --view     # also opens a live MuJoCo viewer window,
                                     # paced to roughly real time, while the
                                     # sequence runs (close the window to stop
                                     # early)

Uses generate_triple_scene_v4.py / dlo_route_v4_freeze.py -- fully
independent of the v2/v3 pipelines, so scene edits here don't affect them.

During each of r1's 3 camera-choreography pose holds (A/B/C, durations
vary -- see r1_camera_ik.py's SCHEDULE), captures RGB+depth image pairs at
30Hz from r1's own wrist camera (r1_d435i_rgb / r1_d435i_depth) plus
per-capture camera pose + r1 qpos + camera intrinsics, to
results/r1_eye_in_hand_calib_v4/pose_{A,B,C}/ -- this is the sole capture
pipeline now; the world-fixed cable_side/cable_init_top cameras still
exist in the scene (viewable live with --view, Tab to cycle) but are no
longer recorded to disk -- the tracking pipeline
(trackdlo_eval/run_r1_tracking.py) is built entirely on r1's own moving
camera. See r1_camera_capture.py.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import mujoco
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "mujoco_scenes"))
sys.path.insert(0, str(PROJECT_ROOT / "trajectories"))
sys.path.insert(0, str(PROJECT_ROOT / "rendering"))
from generate_triple_scene_v4 import generate  # noqa: E402
from dlo_route_v4_freeze import DanglePlanner  # noqa: E402
from r1_camera_ik import CameraArm, set_solved_waypoints, target_at, rail_at, hold_window_at, hold_windows  # noqa: E402
from r1_camera_capture import R1CalibCapture  # noqa: E402

# r1's wrist-camera choreography waypoints -- see r1_camera_ik.py's module
# docstring for why these are solved offline rather than IK'd live during
# the physics-stepped run. Re-solved via
# trajectories/verify_r1_camera_reachability_v4.py, rail-aware (rail_x=
# -0.45 for Pose A, +0.10 for Pose B, -0.65 for Pose C -- see
# r1_camera_ik.py's RAIL_A/RAIL_B/RAIL_C/rail_at()). Pose D removed --
# the schedule now ends holding Pose C through t=140s. Pose A holds until
# t=26.0s; Pose B arrives by t=28.0s and holds until t=35.0s; Pose C
# arrives by t=36.0s (all user-specified). Pose C's own qpos below is NOT
# the verifier's chained-seed result (that landed in a worse local
# minimum, 16mm/19deg) -- it's the qpos found by a dedicated orientation
# search (see r1_camera_ik.py's POSE_C comment), independently confirmed
# to match Pose C almost exactly (0.0002mm/0.02deg) and collision-free.
# Pose B's qpos is UNCHANGED from the original rail_x=+0.20 solve -- per
# user direction, its rail position was shifted -0.20m then +0.10m (net
# -0.10m from the original) while keeping the same arm joint angles (pure
# rail translations, not new IK solves; see r1_camera_ik.py's POSE_B
# comment). home->A, A->B, B->C transition paths (arm qpos AND rail
# position moving together) re-verified collision-free (300-sample sweep
# each).
set_solved_waypoints(
    np.array([2.220378, -0.22875, 1.233465, 2.193019, -1.607109, 0.128412]),  # Pose A (reachable, 1.37mm/1.82deg, rail_x=-0.45, user-supplied target)
    np.array([-3.363153, -0.728326, 1.141771, 2.389318, -1.607312, 5.889064]),  # Pose B (reachable, 3.61mm/1.79deg, rail_x=+0.10, base net -0.10m from original at fixed joint angles)
    np.array([-4.809949, 1.272049, 2.506624, 1.940535, -1.368249, -6.28319]),  # Pose C (reachable, 4.98mm/1.32deg, rail_x=-0.65, ~20deg orientation adjustment from user's exact target)
)

FRAME_DT = 1.0 / 30.0

# r1's static camera-viewing home pose, held before the settle sequence
# starts and for the first 2s of it, before r1 starts moving toward Pose A
# (see trajectories/r1_camera_ik.py's SCHEDULE).
R1_HOME = {1: 0.5, 2: -0.9, 3: 1.5, 4: 0.0, 5: -1.25, 6: 0.0}


def _make_on_frame(viewer_handle) -> callable:
    """Live-viewer sync + real-time pacing, if a viewer handle is given;
    a no-op callback otherwise."""
    last_wall = [time.time()]

    def on_frame(model: mujoco.MjModel, data: mujoco.MjData, t: float) -> None:
        if viewer_handle is not None:
            if not viewer_handle.is_running():
                raise RuntimeError("viewer window closed")
            viewer_handle.sync()
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
    args = parser.parse_args()
    sys.stdout.reconfigure(line_buffering=True)  # stream progress prints live instead of buffering

    generate(layout="dlo")
    model = mujoco.MjModel.from_xml_path(str(PROJECT_ROOT / "triple_lite6_cable_routing_dlo_v4.xml"))
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

    viewer_handle = None
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

        viewer_handle = mj_viewer.launch_passive(model, data, key_callback=_key_callback)

        # Group 4 = r1's own visual meshes/D435i housing (see
        # generate_triple_scene_v4.py's add_robot_instances/
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
        print("[dlo] live viewer open -- Tab to cycle cameras (cable_lateral / cable_side / cable_init_top / r1_d435i_rgb / box_focus / fixture_iso / overview). Close window to stop early.")

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
        model, r1_cam_arm, PROJECT_ROOT / "results" / "r1_eye_in_hand_calib_v4", hold_windows(),
    )
    r1_rail_aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "r1_rail_x")

    def _combined_on_frame(model, data, t):
        target_qpos = target_at(t)
        if target_qpos is not None:
            r1_cam_arm.set_ctrl_to_qpos(target_qpos)
        data.ctrl[r1_rail_aid] = rail_at(t)
        window = hold_window_at(t)
        r1_capture.maybe_capture(model, data, t, window[0] if window is not None else None)
        base_on_frame(model, data, t)

    base_on_frame = _make_on_frame(viewer_handle)

    print("[dlo] running v4 sequence: r2+r3 grasp/carry the cable's two ends "
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
    print(f"[dlo] r1 eye-in-hand calibration captures saved to {PROJECT_ROOT / 'results' / 'r1_eye_in_hand_calib_v4'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

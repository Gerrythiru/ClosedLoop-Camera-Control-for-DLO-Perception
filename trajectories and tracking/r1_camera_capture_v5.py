"""V5 fork of r1_camera_capture.py -- CAPTURE_STOP_OVERRIDE emptied so each
pose captures its full hold window (see below). Otherwise identical.

r1 wrist-camera capture for future eye-in-hand calibration, and for
trackdlo_eval/run_r1_tracking.py's tracking pipeline.

r1_d435i_rgb and r1_d435i_depth are separate, non-co-located cameras (see
mujoco_scenes/generate_triple_scene_v3.py's add_cameras_to_arms()) --
deliberate hardware realism for the eye-in-hand calibration use case, but
wrong for tracking: publishing r1_d435i_rgb's color alongside
r1_d435i_depth's depth as if pixel (u,v) meant the same physical ray in
both is false (different fovy AND a real baseline separation -- confirmed
this produces real, spatially-coherent misregistration, e.g. a genuinely
visible marker color landing on the OTHER camera's background-depth
reading at that same pixel coordinate). So every capture ALSO renders and
saves r1_d435i_rgb's own depth channel (`depth_rgb_*.npy`, co-located with
the RGB by construction -- same ray, same intrinsics, zero registration
error) alongside the original `depth_*.npy` (r1_d435i_depth's own, kept
untouched for the calibration use case). Tracking should consume
depth_rgb_*; calibration keeps using depth_*.

Captures RGB+depth pairs at 30Hz per pose hold (A/B/C) during
run_dlo_v4_freeze.py's r1 camera choreography (Pose C stops early at
absolute t=51.5s rather than running to its hold's actual end -- see
CAPTURE_STOP_OVERRIDE), saving per-capture camera pose (both cameras) +
r1's qpos + per-pose intrinsics, for a future (not-yet-built) calibration
step to consume.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import mujoco
import numpy as np

from render_rgbd import render_rgbd, hide_own_arm_geoms
from camera_intrinsics import get_intrinsics

RGB_CAMERA = "r1_d435i_rgb"
DEPTH_CAMERA = "r1_d435i_depth"
CAPTURE_WIDTH = 1280
CAPTURE_HEIGHT = 720
CAPTURE_HZ = 15.0   # (2026-09-08: was 30.0)
CAPTURE_PERIOD_S = 1.0 / CAPTURE_HZ
START_MARGIN_S = 1.0   # skip this much at the start of each hold before first capture
END_MARGIN_S = 1.0     # leave this much before hold end for the last capture

# V5 (2026-09-08): Pose B's hold is (38.0, 140.0) but capture stops at
# absolute t=90.0s -- covers grasp + the full zigzag carry (r2 reaches
# WP11 by ~t=82) + a little settle; the rest of B's hold is a static
# scene not worth ~750 more frames of disk. No override for A. (The shared
# r1_camera_capture.py keeps V4's {"C": 51.5}.)
CAPTURE_STOP_OVERRIDE: dict[str, float] = {"B": 90.0}


def capture_instants(start: float, end: float, stop_at: float | None = None) -> np.ndarray:
    """Capture times at CAPTURE_HZ (30 per second), starting at
    start+START_MARGIN_S, spaced CAPTURE_PERIOD_S apart, not exceeding
    min(end-END_MARGIN_S, stop_at) if stop_at is given (absolute sequence
    time -- see CAPTURE_STOP_OVERRIDE). Count varies with hold-window
    duration instead of a fixed count per pose. Raises if the resulting
    window is too short for the configured margins (fails loudly rather
    than silently producing a degenerate/negative-length window)."""
    lo, hi = start + START_MARGIN_S, end - END_MARGIN_S
    if stop_at is not None:
        hi = min(hi, stop_at)
    if hi <= lo:
        raise ValueError(f"hold window [{start},{end}] (stop_at={stop_at}) too short for margins")
    n = int(np.floor((hi - lo) / CAPTURE_PERIOD_S)) + 1
    return lo + np.arange(n) * CAPTURE_PERIOD_S


def _cam_pose(data: mujoco.MjData, cam_id: int) -> tuple[np.ndarray, np.ndarray]:
    pos = data.cam_xpos[cam_id].copy()
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, data.cam_xmat[cam_id])
    return pos, quat


class R1CalibCapture:
    """Owns the 1280x720 renderer, output dirs, per-pose metadata
    accumulation, and per-pose capture-index ratchet state. One instance
    per run_dlo_v3_freeze.py invocation."""

    def __init__(
        self,
        model: mujoco.MjModel,
        r1_cam_arm,
        output_root: Path,
        pose_windows: dict[str, tuple[float, float]] | None = None,
        live_tracking_manager=None,
        open_ended: bool = False,
    ) -> None:
        # 2026-09-22: open_ended=True (run_dlo_v5_freeze.py's --nbv) skips the
        # precomputed-instants-array path below, which requires every label's
        # end time known up front -- incompatible with a signal-driven hold
        # whose end isn't known until it happens. See maybe_capture_open_ended().
        self.open_ended = open_ended
        self._next_capture_t: dict[str, float] = {}
        self.model = model
        self.r1_cam_arm = r1_cam_arm
        # 2026-09-21: optional live-publish hook (trajectories/
        # r1_live_tracking_bridge.py's LiveTrackingManager) -- None by
        # default, so this file keeps zero ROS dependency and zero
        # behavior change unless run_dlo_v5_freeze.py's --ros-live flag
        # explicitly constructs and passes one in.
        self.live_tracking_manager = live_tracking_manager
        self.renderer = mujoco.Renderer(model, height=CAPTURE_HEIGHT, width=CAPTURE_WIDTH)
        self.scene_option = hide_own_arm_geoms()
        self.rgb_cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, RGB_CAMERA)
        self.depth_cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, DEPTH_CAMERA)
        if self.rgb_cam_id < 0 or self.depth_cam_id < 0:
            raise ValueError("r1_d435i_rgb / r1_d435i_depth camera missing from model")

        self.rgb_intrinsics = get_intrinsics(model, RGB_CAMERA, CAPTURE_WIDTH, CAPTURE_HEIGHT)
        self.depth_intrinsics = get_intrinsics(model, DEPTH_CAMERA, CAPTURE_WIDTH, CAPTURE_HEIGHT)

        # Cable first / last node -- r2's and r3's grasped ends (see
        # dlo_route_*_freeze.py's weld mechanism). Recorded per capture so
        # a downstream tracker can recover each grasped end's world
        # position at any captured instant directly from the robots' own
        # known state (rigidly welded to the gripper once grasped), without
        # needing to re-run the simulation. The last-node index is
        # discovered from the model (V4 cable has 20 segments -> node 19,
        # V5 has 15 -> node 14), so the metadata key "grasp_node19_pos"
        # below is a fixed historical label, not the actual index.
        self.grasp0_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "red_cable_00")
        self.grasp1_body_id = -1
        _i = 0
        while True:
            _bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"red_cable_{_i:02d}")
            if _bid < 0:
                break
            self.grasp1_body_id = _bid
            _i += 1

        self.output_root = Path(output_root)
        if open_ended:
            # Labels/output dirs are created lazily on first sight in
            # maybe_capture_open_ended() -- no pose_windows required.
            self._instants: dict[str, np.ndarray] = {}
            self._next_idx: dict[str, int] = {}
            self._captures: dict[str, list[dict]] = {}
        else:
            pose_windows = pose_windows or {}
            self._instants = {
                label: capture_instants(s, e, stop_at=CAPTURE_STOP_OVERRIDE.get(label))
                for label, (s, e) in pose_windows.items()
            }
            self._next_idx = {label: 0 for label in pose_windows}
            self._captures = {label: [] for label in pose_windows}
            for label in pose_windows:
                (self.output_root / f"pose_{label}").mkdir(parents=True, exist_ok=True)

    def maybe_capture(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        t: float,
        pose_label: Optional[str],
    ) -> None:
        """Call once per on_frame tick. No-ops outside a hold window or
        once all captures for the current pose (see capture_instants) are
        done. Forward-ratchet: only ever compares t against the single
        next unfired instant for pose_label, so it cannot double-fire or
        skip regardless of on_frame's exact cadence, as long as t is
        monotonically non-decreasing (true for dlo_route_v2_Freeze.py's
        _run)."""
        if pose_label is None or pose_label not in self._next_idx:
            return
        idx = self._next_idx[pose_label]
        if idx >= len(self._instants[pose_label]):
            return
        if t < self._instants[pose_label][idx]:
            return
        self._do_capture(model, data, t, pose_label, idx)
        self._next_idx[pose_label] += 1

    def maybe_capture_open_ended(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        t: float,
        pose_label: Optional[str],
    ) -> None:
        """open_ended=True counterpart to maybe_capture() -- for a hold whose
        end isn't known in advance (run_dlo_v5_freeze.py's --nbv). Instead of
        comparing against a precomputed instants array, ticks a simple
        per-label "next capture due" time forward by CAPTURE_PERIOD_S after
        each capture. First sight of a label lazily creates its output dir
        and schedules its first capture START_MARGIN_S after that moment
        (mirroring capture_instants()'s start margin). Capture simply stops
        the instant pose_label stops matching -- no END_MARGIN_S/stop_at,
        since there's no known end to leave a margin before."""
        if not self.open_ended:
            raise RuntimeError("maybe_capture_open_ended() requires open_ended=True")
        if pose_label is None:
            return
        if pose_label not in self._next_idx:
            self._next_idx[pose_label] = 0
            self._captures[pose_label] = []
            self._next_capture_t[pose_label] = t + START_MARGIN_S
            (self.output_root / f"pose_{pose_label}").mkdir(parents=True, exist_ok=True)
        if t < self._next_capture_t[pose_label]:
            return
        idx = self._next_idx[pose_label]
        self._do_capture(model, data, t, pose_label, idx)
        self._next_idx[pose_label] += 1
        self._next_capture_t[pose_label] += CAPTURE_PERIOD_S

    def _do_capture(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        t: float,
        pose_label: str,
        idx: int,
    ) -> None:
        rgb, depth_rgb_cam = render_rgbd(model, data, self.renderer, RGB_CAMERA, self.scene_option)
        _, depth = render_rgbd(model, data, self.renderer, DEPTH_CAMERA, self.scene_option)

        pose_dir = self.output_root / f"pose_{pose_label}"
        rgb_file = f"rgb_{idx:02d}.png"
        depth_file = f"depth_{idx:02d}.npy"
        depth_rgb_file = f"depth_rgb_{idx:02d}.npy"

        import imageio
        imageio.imwrite(str(pose_dir / rgb_file), rgb)
        np.save(str(pose_dir / depth_file), depth.astype(np.float32))
        np.save(str(pose_dir / depth_rgb_file), depth_rgb_cam.astype(np.float32))

        rgb_pos, rgb_quat = _cam_pose(data, self.rgb_cam_id)
        depth_pos, depth_quat = _cam_pose(data, self.depth_cam_id)
        qpos = self.r1_cam_arm.get_qpos()
        grasp0_pos = data.xpos[self.grasp0_body_id].copy() if self.grasp0_body_id >= 0 else None
        grasp1_pos = data.xpos[self.grasp1_body_id].copy() if self.grasp1_body_id >= 0 else None

        # 2026-09-21: live-publish, alongside the disk save above -- reuses
        # the SAME rendered rgb/depth_rgb_cam arrays (no duplicate render).
        # depth_rgb_cam (not depth) matches the offline pipeline's own
        # established choice (see run_r1_tracking_v5.py's load_pose_captures
        # docstring): it's the RGB camera's own co-located depth channel,
        # pixel-aligned with rgb -- the separate DEPTH_CAMERA sensor has a
        # real baseline offset + different fovy and would misregister.
        if self.live_tracking_manager is not None:
            self.live_tracking_manager.on_capture(rgb, depth_rgb_cam, rgb_pos, rgb_quat, t, pose_label)

        self._captures[pose_label].append({
            "index": idx,
            "t": float(t),
            "qpos": qpos.tolist(),
            "rgb_cam_pos": rgb_pos.tolist(),
            "rgb_cam_quat": rgb_quat.tolist(),
            "depth_cam_pos": depth_pos.tolist(),
            "depth_cam_quat": depth_quat.tolist(),
            "rgb_file": rgb_file,
            "depth_file": depth_file,
            "depth_rgb_file": depth_rgb_file,
            "grasp_node0_pos": grasp0_pos.tolist() if grasp0_pos is not None else None,
            "grasp_node19_pos": grasp1_pos.tolist() if grasp1_pos is not None else None,
        })
        if self.open_ended:
            print(f"[r1_calib_capture] pose {pose_label} capture {idx} at t={t:.2f}s", flush=True)
        else:
            print(f"[r1_calib_capture] pose {pose_label} capture {idx}/{len(self._instants[pose_label])-1} at t={t:.2f}s", flush=True)

    def save_all_metadata(self) -> None:
        for label, captures in self._captures.items():
            meta = {
                "rgb_camera": RGB_CAMERA,
                "depth_camera": DEPTH_CAMERA,
                "resolution": {"width": CAPTURE_WIDTH, "height": CAPTURE_HEIGHT},
                "rgb_intrinsics": self.rgb_intrinsics.tolist(),
                "depth_intrinsics": self.depth_intrinsics.tolist(),
                "captures": captures,
            }
            with open(self.output_root / f"pose_{label}" / "metadata.json", "w") as f:
                json.dump(meta, f, indent=2)

    def close(self) -> None:
        self.renderer.close()

"""Live (in-process) TrackDLO bridge -- publishes r1's captured frames to
ROS AS the MuJoCo sim steps, instead of trackdlo_eval/run_r1_tracking_v5.py's
disk-replay-after-the-fact approach. See OFlinevsONline.md (repo root) for
the full offline-vs-live comparison this module implements.

Scope (see /home/nasta/.claude/plans/proud-tinkering-anchor.md):
- r1's camera still follows its existing fixed Pose A -> Pose B schedule;
  this module only reacts to hold-window transitions, it doesn't drive them.
- Node-identity resolution is "Option B": every hold gets a fresh
  trackdlo+init_tracker relaunch and a real HSV+skeleton re-init (no
  bridge_seed/synthetic-init-nodes position-carrying -- that approach was
  tried earlier in this codebase and abandoned, see run_r1_tracking_v5.py's
  bridge_seed docstring). Only node ORDER IDENTITY is chained across holds
  (via _endpoint_flip against the previous hold's last live result), never
  position.
- No node_support/tracking_diag (Bucket-3) logging here -- nothing in this
  pass's confirmed scope consumes it; the offline script remains the place
  for that. Keeps this module to exactly what's needed to prove live
  tracking works.

IMPORTANT: requires rospy/roslaunch (the ros_noetic conda env with
~/tracking_ws sourced), NOT this project's venv -- see module docstring of
run_dlo_v5_freeze.py's --ros-live flag. This module is only ever imported
when that flag is passed, so it has no import-time effect otherwise.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "trackdlo_eval"))
sys.path.insert(0, str(REPO_ROOT / "rendering"))

from run_r1_tracking_v5 import (  # noqa: E402
    NUM_NODES, RESULT_FRAME_ID, CAMERA_INFO_TOPIC, RGB_TOPIC, DEPTH_TOPIC, RESULTS_TOPIC,
    VISIBILITY_THRESHOLD,
    world_to_cam_transform, cam_to_world, _marker_flip, _endpoint_flip,
    publish_frame_ros, _launch_trackdlo_node, PoseSession,
)

DEFAULT_OUTPUT_DIR = REPO_ROOT / "results" / "r1_dlo_tracking_v5_live"


class LiveTrackSession:
    """One hold's worth of live publish + tracked-result bookkeeping. A
    fresh instance is created by LiveTrackingManager every time a new hold
    starts (see that class -- a fresh trackdlo/init_tracker process pair is
    relaunched alongside it, since trackdlo_node.cpp cannot be live-reset)."""

    def __init__(self, k_matrix: np.ndarray, height: int, width: int,
                 paint_markers: bool = True, prev_hold_last_world: np.ndarray | None = None) -> None:
        import rospy
        from sensor_msgs.msg import CameraInfo, Image, PointCloud2

        self.k_matrix = k_matrix
        self.height, self.width = height, width
        self.paint_markers = paint_markers
        self._prev_hold_last_world = prev_hold_last_world

        self.camera_info_pub = rospy.Publisher(CAMERA_INFO_TOPIC, CameraInfo, queue_size=1, latch=True)
        self.rgb_pub = rospy.Publisher(RGB_TOPIC, Image, queue_size=10)
        self.depth_pub = rospy.Publisher(DEPTH_TOPIC, Image, queue_size=10)

        self._stamp_to_pose: dict = {}          # stamp -> (cam_pos, cam_quat)
        self._first_rgb: np.ndarray | None = None  # raw (unpainted) rgb of this hold's first published frame

        self.latest_result_cam: np.ndarray | None = None
        self.latest_result_world: np.ndarray | None = None
        self._pending_flip_resolution = True
        self._flip: bool | None = None

        rospy.Subscriber(RESULTS_TOPIC, PointCloud2, self._on_results)

    def publish_frame(self, rgb: np.ndarray, depth: np.ndarray, cam_pos: np.ndarray, cam_quat: np.ndarray,
                       stamp) -> None:
        if self._first_rgb is None:
            self._first_rgb = rgb.copy()
        self._stamp_to_pose[stamp] = (np.asarray(cam_pos, dtype=np.float64), np.asarray(cam_quat, dtype=np.float64))
        publish_frame_ros(rgb, depth, self.k_matrix, self.height, self.width, stamp,
                           self.camera_info_pub, self.rgb_pub, self.depth_pub,
                           paint_markers=self.paint_markers)

    def _on_results(self, msg) -> None:
        pose = self._stamp_to_pose.get(msg.header.stamp)
        if pose is None:
            return
        cam_pos, cam_quat = pose
        nodes_cam = PoseSession._pc2_to_array(msg)
        if nodes_cam.shape[0] != NUM_NODES:
            return
        self.latest_result_cam = nodes_cam
        w2c = world_to_cam_transform(cam_pos, cam_quat)
        nodes_world = cam_to_world(nodes_cam, w2c)

        if self._pending_flip_resolution:
            if self._prev_hold_last_world is not None:
                self._flip = _endpoint_flip(nodes_world, self._prev_hold_last_world)
                why = "chained to previous hold's last live result"
            elif self._first_rgb is not None:
                self._flip = _marker_flip(self._first_rgb, nodes_cam, self.k_matrix)
                why = "marker-anchored (node 0 = r2 grasp end)"
            else:
                self._flip = False
                why = "no reference available -- keeping native order"
            self._pending_flip_resolution = False
            print(f"[live-track] node order {'REVERSED' if self._flip else 'forward'} ({why})", flush=True)

        if self._flip:
            nodes_world = nodes_world[::-1].copy()
        self.latest_result_world = nodes_world


class LiveTrackingManager:
    """Owned by run_dlo_v5_freeze.py's main() -- watches which hold
    (pose_label) is active via on_capture() and relaunches trackdlo/
    init_tracker at every hold transition. latest_markers_world() is read
    by the viewer's on_frame for the purple marker overlay."""

    def __init__(self, k_matrix: np.ndarray, height: int, width: int,
                 output_dir: Path | None = None, paint_markers: bool = True,
                 tau_vis: float = VISIBILITY_THRESHOLD) -> None:
        self.k_matrix = k_matrix
        self.height, self.width = height, width
        self.output_dir = Path(output_dir) if output_dir is not None else DEFAULT_OUTPUT_DIR
        self.paint_markers = paint_markers
        self.tau_vis = tau_vis

        self._current_label: str | None = None
        self._launch = None
        self._session: LiveTrackSession | None = None
        self._prev_hold_last_world: np.ndarray | None = None
        self._saved_t: list[float] = []
        self._saved_nodes: list[np.ndarray] = []

    def on_capture(self, rgb: np.ndarray, depth: np.ndarray, cam_pos: np.ndarray, cam_quat: np.ndarray,
                   t_sim: float, pose_label: str) -> None:
        """Call once per R1CalibCapture._do_capture -- i.e. only at
        CAPTURE_HZ, only inside a hold window (pose_label is never None
        here; R1CalibCapture already gates that)."""
        import rospy

        if pose_label != self._current_label:
            self._enter_hold(pose_label)
        self._session.publish_frame(rgb, depth, cam_pos, cam_quat, rospy.Time.now())
        if self._session.latest_result_world is not None:
            self._saved_t.append(t_sim)
            self._saved_nodes.append(self._session.latest_result_world.copy())

    def _enter_hold(self, pose_label: str) -> None:
        import rospy

        if self._launch is not None:
            if self._session is not None and self._session.latest_result_world is not None:
                self._prev_hold_last_world = self._session.latest_result_world.copy()
            print(f"[live-track] leaving hold {self._current_label}: stopping trackdlo/init_tracker", flush=True)
            self._launch.stop()
            rospy.sleep(2.0)  # let the old node fully release its topics -- mirrors the offline
                              # script's inter-pose drain (init_tracker/trackdlo's subscribers are
                              # one-shot, so the new process needs a clean slate on the same topics)

        print(f"[live-track] entering hold {pose_label}: relaunching trackdlo+init_tracker", flush=True)
        self._launch = _launch_trackdlo_node(with_init_tracker=True, tau_vis=self.tau_vis)
        rospy.sleep(2.0)  # give the freshly-launched nodes time to come up before the first publish
        self._session = LiveTrackSession(self.k_matrix, self.height, self.width,
                                          paint_markers=self.paint_markers,
                                          prev_hold_last_world=self._prev_hold_last_world)
        self._current_label = pose_label

    def latest_markers_world(self) -> np.ndarray | None:
        return self._session.latest_result_world if self._session is not None else None

    def save_output(self, path: Path | None = None) -> None:
        path = Path(path) if path is not None else (self.output_dir / "combined_tracked_trajectory_live.npz")
        path.parent.mkdir(parents=True, exist_ok=True)
        t_arr = np.array(self._saved_t, dtype=np.float64)
        nodes_arr = (np.array(self._saved_nodes, dtype=np.float64) if self._saved_nodes
                     else np.zeros((0, NUM_NODES, 3)))
        np.savez(path, t=t_arr, nodes=nodes_arr)
        print(f"[live-track] saved {len(t_arr)} live-tracked frames to {path}", flush=True)

    def shutdown(self) -> None:
        if self._launch is not None:
            print(f"[live-track] shutting down (last hold: {self._current_label})", flush=True)
            self._launch.stop()
            self._launch = None

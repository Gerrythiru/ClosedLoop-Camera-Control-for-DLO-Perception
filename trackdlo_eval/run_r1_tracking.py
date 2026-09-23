"""Continuous DLO tracking across r1's 3 moving camera poses (V4).

Runs ONE continuous TrackDLO-based tracking process across Pose A -> B ->
C instead of 3 independent static sessions stitched after the fact. See
/home/nasta/.claude/plans/i-want-to-track-peaceful-aho.md (also copied to
this repo's own i-want-to-track-peaceful-aho.md) for the full design
rationale -- short version:

- Pose A gets a real HSV+skeleton init (trackdlo's own `init_tracker`
  node). Its raw node order is aligned ONCE here so index 0 means "r2's
  grasp end" (matching get_cable_ground_truth()'s node-0 convention),
  using the scene's marker-colored ends as the alignment signal.
- Pose B and C's `init_tracker` is never run. Instead, the previous pose's
  last tracked frame is transformed to world frame, its two known-moving
  endpoints (node 0 = r2's grasp, node 19 = r3's grasp -- rigidly welded
  to their grippers, see dlo_route_v4_freeze.py) are advanced to their
  actual position at the new pose's first capture (read directly from
  r1_camera_capture.py's recorded grasp_node0_pos/grasp_node19_pos, no
  re-simulation needed), transformed into the new pose's camera frame, and
  published directly as a synthetic /trackdlo/init_nodes message -- only
  `trackdlo`'s C++ tracking node is (re)launched per pose, not
  `init_tracker`.
- Node identity is therefore never independently re-derived per pose:
  it's the same array carried through a coordinate transform each time.
  Nothing needs to be reconciled/stitched after the fact.
- The camera-in-transit windows (xfer, ~1-2s each) produce no observations
  and are simply absent from the output timeline -- not interpolated.

Output: results/r1_dlo_tracking_v4/combined_tracked_trajectory.npz, with
`t` (T,) and `nodes` (T, 20, 3) in world frame.

IMPORTANT: this module's ROS-dependent pieces (PoseSession, main) require
a sourced ROS1/tracking_ws environment (rospy, roslaunch, the trackdlo
package) that is NOT available in the sandboxed environment this file was
authored in -- they are written to mirror bridge_publish_trajectory.py's
established patterns exactly, but have not been executed/tested here.
Run and debug them in the environment where ~/tracking_ws is built and
sourced. The pure-numpy helpers below (load_pose_captures through
bridge_seed) have no ROS dependency and ARE tested (see this file's
docstring-adjacent test invocations / the accompanying smoke test run).
"""

from __future__ import annotations

import json
import struct
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "rendering"))

CALIB_ROOT = REPO_ROOT / "results" / "r1_eye_in_hand_calib_v4"
OUTPUT_DIR = REPO_ROOT / "results" / "r1_dlo_tracking_v4"
OUTPUT_NPZ = OUTPUT_DIR / "combined_tracked_trajectory.npz"

NUM_NODES = 20            # matches get_cable_ground_truth()'s node-0..19 convention
POSES = ("A", "B", "C")

CAMERA_INFO_TOPIC = "/sim/camera_info"
RGB_TOPIC = "/sim/rgb/image_raw"
DEPTH_TOPIC = "/sim/depth/image_raw"
INIT_NODES_TOPIC = "/trackdlo/init_nodes"
RESULTS_TOPIC = "/trackdlo/results_pc"
RESULT_FRAME_ID = "sim_camera_optical_frame"
FRAME_RATE_HZ = 30.0        # rate the frames were CAPTURED at (fixed)
# Rate the bridge PUBLISHES frames to ROS at. The trackdlo C++ node's
# tracking step takes ~80 ms/frame (data-dependent), so publishing at the
# 30 Hz capture rate overruns it and its message_filters queue (size 10)
# drops ~60% of frames before they are ever tracked. Publishing at ~10 Hz
# lets the tracker keep up and process nearly every frame. Timestamps are
# still start_wall + cap["t"], so rgb/depth sync and result<->frame
# matching are unaffected -- this only spaces the publishes out in wall
# time. Set to 30.0 for the original (frame-dropping) behaviour.
DEFAULT_PUBLISH_RATE_HZ = 10.0

# HSV threshold for the RED cable (mat_cable_red rgba="0.95 0.03 0.02 1").
# The vendored trackdlo.launch default (H 90-130) is for their BLUE rope
# and segments almost none of our cable -- it was the cause of Pose A
# tracking only 34/661 frames and Pose B tracking 0. Measured directly
# against the r1_eye_in_hand_calib_v4 captures: visible cable pixels sit
# at hue~0, S~245, V 160-255 across all three poses (see
# rendering/cable_mask.py, which uses hue 0-12 for "red"). Single lower
# hue band is enough -- the pure-red material does not wrap past ~14.
# Consumed identically by trackdlo_node.cpp (BGR2HSV) and initialize.py
# (RGB2HSV); both land red at hue~0, so one range works for both.
HSV_UPPER = "14 255 255"
HSV_LOWER = "0 120 80"   # was "0 80 60". Tighter S/V floor -- the looser version
                         # occasionally passed enough background in Pose B/C that
                         # the PCL VoxelGrid integer-indexed past its limit
                         # ("Leaf size is too small ... would overflow") and skipped
                         # downsampling, giving 8k-point clouds and ~2s tracking
                         # steps. Measured visible cable is S~245, V 160-255, so
                         # S>=120 V>=80 keeps all of it.
DLO_PIXEL_WIDTH = 6  # measured median rendered cable width across Pose A/B/C (was 40, tuned for the old cable_side camera)
MULTI_COLOR_DLO = False
VISUALIZE_INIT = False

# trackdlo (C++) node's own algorithm-tuning params -- not camera/scene
# specific, these are the vendored launch file's recommended defaults
# verbatim (trackdlo.launch lines 28-61).
BETA = 0.5                 # MCT weight -- larger = more rigid
LAMBDA = 50000              # MCT weight -- larger = more rigid
ALPHA = 3                   # alignment strength
MU = 0.1                    # 0-1, larger = noisier point cloud assumed
MAX_ITER = 50                # EM loop iteration cap
TOL = 0.0002                # EM convergence tolerance
K_VIS = 500                  # visibility info's effect on membership probability
D_VIS = 0.04                 # max geodesic distance between visible nodes to still count as visible
VISIBILITY_THRESHOLD = 0.012  # tau_vis. Authors' value is 0.005; bumped for the chained
                              # A->B->C run. trackdlo_node.cpp skips the whole tracking step
                              # ("No visible nodes this frame") when no node is within tau_vis
                              # of the cloud -- so a freshly-(re)initialised pose whose nodes
                              # start a bit off the cable never gets a CPD step to pull them
                              # in, and skips forever (observed on Pose B/C). A looser tau_vis
                              # gives that recovery slack. Proper fix is a per-frame relaxation
                              # in the C++ node; this is the no-rebuild stand-in.
BETA_PRE_PROC = 3.0           # GLTP pre-processing registration
LAMBDA_PRE_PROC = 1.0
LLE_WEIGHT = 10.0
DOWNSAMPLE_LEAF_SIZE = 0.008


# ---------------------------------------------------------------------
# Pure-numpy helpers -- no ROS dependency, testable standalone.
# ---------------------------------------------------------------------

def load_pose_captures(pose_label: str) -> tuple[dict, list[dict]]:
    """Load results/r1_eye_in_hand_calib_v4/pose_{label}/metadata.json,
    return (meta, captures) with captures sorted by t and rgb/depth paths
    resolved to Path objects. Uses depth_rgb_file (r1_d435i_rgb's own
    depth channel, co-located with the RGB by construction) rather than
    depth_file (r1_d435i_depth's separate, non-co-located sensor) --
    confirmed the latter produces real pixel misregistration against the
    RGB image (different fovy + a real baseline separation), which was
    corrupting the point cloud handed to trackdlo. depth_file is still
    saved for the eye-in-hand calibration use case, just not used here."""
    pose_dir = CALIB_ROOT / f"pose_{pose_label}"
    with open(pose_dir / "metadata.json") as f:
        meta = json.load(f)
    captures = sorted(meta["captures"], key=lambda c: c["t"])
    for c in captures:
        c["rgb_path"] = pose_dir / c["rgb_file"]
        c["depth_path"] = pose_dir / c["depth_rgb_file"]
    return meta, captures


def quat_to_mat(quat_wxyz: np.ndarray) -> np.ndarray:
    """wxyz quaternion -> 3x3 rotation matrix, matching MuJoCo's own
    mju_quat2Mat convention. Pure numpy (no mujoco import needed) so this
    works against saved metadata without a live model/data."""
    w, x, y, z = np.asarray(quat_wxyz, dtype=np.float64)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def world_to_cam_transform(cam_pos: np.ndarray, cam_quat_wxyz: np.ndarray) -> np.ndarray:
    """4x4 world-to-camera transform (OpenCV convention: +X right, +Y
    down, +Z forward), from a saved raw MuJoCo (cam_xpos, cam_xmat-derived
    quat) pair -- same convention/derivation as
    ground_truth/camera_extrinsic.py's get_camera_extrinsic, just built
    from a stored pos/quat instead of live model/data (R1CalibCapture
    stores raw MuJoCo local-to-world cam pose, not the OpenCV-flipped
    extrinsic, so the transpose + Y/Z flip below is required here too)."""
    rot_local_to_world = quat_to_mat(cam_quat_wxyz)
    rot = rot_local_to_world.T  # world-to-camera
    rot = rot.copy()
    rot[1, :] = -rot[1, :]
    rot[2, :] = -rot[2, :]
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = rot
    T[:3, 3] = -rot @ np.asarray(cam_pos, dtype=np.float64)
    return T


def world_to_cam(points_world: np.ndarray, world_to_cam_4x4: np.ndarray) -> np.ndarray:
    n = points_world.shape[0]
    h = np.concatenate([points_world, np.ones((n, 1))], axis=1)
    return (world_to_cam_4x4 @ h.T).T[:, :3]


def cam_to_world(points_cam: np.ndarray, world_to_cam_4x4: np.ndarray) -> np.ndarray:
    """Inverse of world_to_cam: world_to_cam applies p_cam = R@p_world + t,
    so p_world = R.T @ (p_cam - t)."""
    R = world_to_cam_4x4[:3, :3]
    t = world_to_cam_4x4[:3, 3]
    return (R.T @ (np.asarray(points_cam) - t).T).T


def pinhole_project(points_cam: np.ndarray, k_matrix: np.ndarray) -> np.ndarray:
    """Nx3 camera-frame points -> Nx2 pixel coordinates (standard pinhole,
    points already in the camera's own frame -- no extrinsic needed)."""
    ph = (k_matrix @ points_cam.T).T
    return ph[:, :2] / ph[:, 2:3]


def _marker_flip(rgb: np.ndarray, points_cam: np.ndarray, k_matrix: np.ndarray) -> bool:
    """First pose only: TrackDLO's skeleton traversal picks an arbitrary
    start/end. Returns True if the tracked node array should be reversed so
    index 0 is the "marker_start" (green) end -- matching
    get_cable_ground_truth()'s node-0 = r2's-grasp convention -- using the
    scene's marker-coloured ends (mat_cable_marker_start/end). Returns False
    (keep native order) if neither marker is visible."""
    from cable_mask import get_cable_mask

    def centroid(mask: np.ndarray) -> np.ndarray | None:
        ys, xs = np.where(mask)
        return None if len(xs) == 0 else np.array([xs.mean(), ys.mean()])

    start_c = centroid(get_cable_mask(rgb, "marker_start"))
    end_c = centroid(get_cable_mask(rgb, "marker_end"))
    if start_c is None and end_c is None:
        print("[run_r1_tracking] WARNING: neither marker visible in the first pose's init "
              "frame -- keeping TrackDLO's native node order as the reference.", flush=True)
        return False

    pix = pinhole_project(points_cam, k_matrix)
    if start_c is not None:
        return bool(np.linalg.norm(pix[-1] - start_c) < np.linalg.norm(pix[0] - start_c))
    return bool(np.linalg.norm(pix[0] - end_c) < np.linalg.norm(pix[-1] - end_c))


def _paint_end_markers(rgb: np.ndarray) -> np.ndarray:
    """Recolor the cable's end-marker segments -- marker_start (green,
    node 0 / r2's grasp) and marker_end (cyan, node 19 / r3's grasp) -- to
    the cable's own red, so trackdlo's internal HSV thresholding treats
    them as cable instead of ignoring them.

    Those two segments are a distinctly-coloured material (for
    _marker_flip's node-order anchoring); the red-only HSV mask never
    matched them, so node 0 and node 19 got ZERO point-cloud support from
    ANY camera angle -- a ~34-80mm standing bias in V4 (see Today_SUMM).
    Applied to every published frame by default (disable with
    --no-paint-markers). _marker_flip still works: it reads the raw,
    unpainted first capture. Ported from run_r1_tracking_v5.py."""
    from cable_mask import get_cable_mask

    red_mask = get_cable_mask(rgb, "red")
    if not red_mask.any():
        return rgb
    red_color = np.median(rgb[red_mask], axis=0).astype(rgb.dtype)

    out = rgb.copy()
    out[get_cable_mask(rgb, "marker_start")] = red_color
    out[get_cable_mask(rgb, "marker_end")] = red_color
    return out


def _endpoint_flip(frame: np.ndarray, ref: np.ndarray) -> bool:
    """True if `frame` (20,3) should be reversed so its node identity lines
    up with `ref` (20,3) -- compares both endpoints forward vs reversed.
    Used to chain each pose's node order to the previous pose's last frame."""
    d_fwd = np.linalg.norm(frame[0] - ref[0]) + np.linalg.norm(frame[-1] - ref[-1])
    d_rev = np.linalg.norm(frame[0] - ref[-1]) + np.linalg.norm(frame[-1] - ref[0])
    return bool(d_rev < d_fwd)


def bridge_seed(prev_world_nodes: np.ndarray, next_first_capture: dict) -> np.ndarray:
    """Advance the two known-moving grasped endpoints (node 0 = r2's
    grasp, node 19 = r3's grasp) to their actual position at the start of
    the next pose's captures -- read directly from that capture's
    recorded grasp_node0_pos/grasp_node19_pos (rigidly known via the weld,
    see dlo_route_v4_freeze.py/r1_camera_capture.py; not a GT-cheat, this
    mirrors what a real system would know from its own gripper's forward
    kinematics + a known rigid grasp). The other 18 nodes are carried over
    stale from the previous pose's last tracked frame -- fine for a seed,
    real tracking resumes once frames start flowing again."""
    seed = prev_world_nodes.copy()
    g0 = next_first_capture.get("grasp_node0_pos")
    g19 = next_first_capture.get("grasp_node19_pos")
    if g0 is not None:
        seed[0] = np.asarray(g0, dtype=np.float64)
    if g19 is not None:
        seed[-1] = np.asarray(g19, dtype=np.float64)
    return seed


def publish_frame_ros(rgb: np.ndarray, depth: np.ndarray, k_matrix: np.ndarray, height: int, width: int,
                       stamp, camera_info_pub, rgb_pub, depth_pub, paint_markers: bool = True) -> None:
    """Construct and publish one frame's CameraInfo + RGB Image + Depth
    Image messages to ROS -- the exact per-frame message-construction logic
    PoseSession.run() used inline (rgb8 encoding, depth->16UC1 millimeters,
    matching header stamps across all three messages), factored out here so
    trajectories/r1_live_tracking_bridge_v4.py's live path can reuse it
    verbatim instead of re-deriving the message formats. `depth` is
    expected in METERS (as rendered/saved); converted to millimeters
    internally to match trackdlo_node.cpp's expected encoding.

    2026-09-22: extracted, additive only, mirroring the identical
    extraction already done in run_r1_tracking_v5.py -- PoseSession.run()
    now calls this too instead of duplicating the construction; no
    behavior change for the existing disk-replay path."""
    from sensor_msgs.msg import CameraInfo, Image

    if paint_markers:
        rgb = _paint_end_markers(rgb)

    p_matrix = np.zeros((3, 4))
    p_matrix[:3, :3] = k_matrix
    info = CameraInfo()
    info.header.frame_id = RESULT_FRAME_ID
    info.header.stamp = stamp
    info.width, info.height = width, height
    info.K = k_matrix.flatten().tolist()
    info.P = p_matrix.flatten().tolist()
    info.distortion_model = "plumb_bob"
    info.D = [0.0] * 5

    depth_mm = np.round(depth * 1000.0).astype(np.uint16)

    rgb_msg = Image()
    rgb_msg.height, rgb_msg.width = rgb.shape[:2]
    rgb_msg.encoding = "rgb8"
    rgb_msg.step = rgb.shape[1] * 3
    rgb_msg.data = rgb.tobytes()
    rgb_msg.header.stamp = stamp
    rgb_msg.header.frame_id = RESULT_FRAME_ID

    depth_msg = Image()
    depth_msg.height, depth_msg.width = depth_mm.shape
    depth_msg.encoding = "16UC1"
    depth_msg.step = depth_mm.shape[1] * 2
    depth_msg.data = depth_mm.tobytes()
    depth_msg.header.stamp = stamp
    depth_msg.header.frame_id = RESULT_FRAME_ID

    camera_info_pub.publish(info)
    rgb_pub.publish(rgb_msg)
    depth_pub.publish(depth_msg)


# ---------------------------------------------------------------------
# ROS-dependent orchestration -- untested in this sandboxed environment,
# see module docstring. Mirrors bridge_publish_trajectory.py's
# ContinuousBridge pub/sub pattern.
# ---------------------------------------------------------------------

class PoseSession:
    """Publishes one pose's captured frames to ROS and collects TrackDLO's
    per-frame tracked output. One instance per pose (A, B, C) -- a fresh
    `trackdlo` C++ node is (re)launched per instance (see main()); this
    class only handles topic pub/sub for that session's frames."""

    def __init__(self, pose_label: str, captures: list[dict], k_matrix: np.ndarray, height: int, width: int,
                 publish_rate_hz: float = DEFAULT_PUBLISH_RATE_HZ, paint_markers: bool = True) -> None:
        import rospy
        from sensor_msgs.msg import CameraInfo, Image, PointCloud2
        from std_msgs.msg import Float64MultiArray

        self.pose_label = pose_label
        self.captures = captures
        self.k_matrix = k_matrix
        self.height, self.width = height, width
        self.publish_rate_hz = publish_rate_hz
        self.paint_markers = paint_markers  # recolor the end-marker segments to cable-red (see _paint_end_markers)

        self.camera_info_pub = rospy.Publisher(CAMERA_INFO_TOPIC, CameraInfo, queue_size=1, latch=True)
        self.rgb_pub = rospy.Publisher(RGB_TOPIC, Image, queue_size=10)
        self.depth_pub = rospy.Publisher(DEPTH_TOPIC, Image, queue_size=10)
        self.init_nodes_pub = rospy.Publisher(INIT_NODES_TOPIC, PointCloud2, queue_size=1, latch=True)

        self.stamp_to_idx: dict = {}
        self.results_by_idx: dict[int, np.ndarray] = {}
        self.init_nodes_result: np.ndarray | None = None  # only populated for Pose A (real init_tracker)

        # Bucket-3 (failure-predictor logging): per-frame node_support (P1)
        # and tracking_diag (iterations/converged/step-time/cloud-size).
        # Neither Float64MultiArray carries a header, so the frame's stamp
        # is appended as the array's last element (see trackdlo_node.cpp) --
        # logged here as (capture_t, ...) pairs, capture_t recovered as
        # stamp_sec - run_start_wall.
        self.node_support_log: list[tuple[float, np.ndarray]] = []
        self.tracking_diag_log: list[tuple[float, float, float, float, float]] = []

        rospy.Subscriber("/trackdlo/node_support", Float64MultiArray, self._on_node_support)
        rospy.Subscriber("/trackdlo/tracking_diag", Float64MultiArray, self._on_tracking_diag)

        # Diagnostic-only (stale-seed hypothesis check, not yet a fix):
        # timing of when the seed actually lands vs. when the publish loop
        # starts, and when the first real (non-"no visible nodes") tracked
        # frame comes back, so we can measure the gap in frames/seconds
        # between "what frame the seed reflects" and "what frame trackdlo
        # was actually looking at" -- see run_r1_tracking's sync/stale-seed
        # discussion.
        self.run_start_wall = None
        self.init_nodes_wall = None
        self.first_result_wall = None
        self.first_result_idx: int | None = None

        rospy.Subscriber(INIT_NODES_TOPIC, PointCloud2, self._on_init_nodes)
        rospy.Subscriber(RESULTS_TOPIC, PointCloud2, self._on_results)

    def _on_init_nodes(self, msg) -> None:
        import rospy

        if self.init_nodes_result is None:
            self.init_nodes_result = self._pc2_to_array(msg)
            self.init_nodes_wall = rospy.Time.now()
            if self.run_start_wall is not None:
                delay_sec = (self.init_nodes_wall - self.run_start_wall).to_sec()
                delay_frames = delay_sec * self.publish_rate_hz
                print(f"[run_r1_tracking] pose {self.pose_label}: /trackdlo/init_nodes received "
                      f"{delay_sec:.3f}s after publish-loop start (~frame idx {delay_frames:.1f} "
                      f"was current at that moment)", flush=True)
            else:
                print(f"[run_r1_tracking] pose {self.pose_label}: /trackdlo/init_nodes received "
                      f"before publish loop started (run_start_wall not set yet)", flush=True)

    def _on_node_support(self, msg) -> None:
        if self.run_start_wall is None:
            return
        stamp_sec, values = msg.data[-1], msg.data[:-1]
        cap_t = stamp_sec - self.run_start_wall.to_sec()
        self.node_support_log.append((cap_t, np.asarray(values, dtype=np.float64)))

    def _on_tracking_diag(self, msg) -> None:
        if self.run_start_wall is None:
            return
        iterations, converged, step_ms, cloud_size, stamp_sec = msg.data
        cap_t = stamp_sec - self.run_start_wall.to_sec()
        self.tracking_diag_log.append((cap_t, iterations, converged, step_ms, cloud_size))

    def _on_results(self, msg) -> None:
        import rospy

        idx = self.stamp_to_idx.get(msg.header.stamp)
        if idx is not None:
            is_first = len(self.results_by_idx) == 0
            self.results_by_idx[idx] = self._pc2_to_array(msg)
            if is_first:
                self.first_result_wall = rospy.Time.now()
                self.first_result_idx = idx
                msg_parts = [f"[run_r1_tracking] pose {self.pose_label}: FIRST real tracked "
                             f"result at frame idx {idx}"]
                if self.run_start_wall is not None:
                    delay_sec = (self.first_result_wall - self.run_start_wall).to_sec()
                    msg_parts.append(f" ({delay_sec:.3f}s after publish-loop start)")
                if self.init_nodes_wall is not None:
                    seed_to_first_sec = (self.first_result_wall - self.init_nodes_wall).to_sec()
                    seed_to_first_frames = seed_to_first_sec * self.publish_rate_hz
                    msg_parts.append(f" -- {seed_to_first_sec:.3f}s / ~{seed_to_first_frames:.1f} "
                                      f"frames after the seed (/trackdlo/init_nodes) landed")
                print("".join(msg_parts), flush=True)

    @staticmethod
    def _pc2_to_array(msg) -> np.ndarray:
        from sensor_msgs import point_cloud2
        pts = list(point_cloud2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True))
        return np.array(pts, dtype=np.float64)

    def publish_synthetic_init_nodes(self, seed_cam_frame: np.ndarray) -> None:
        """Publish a seed directly to /trackdlo/init_nodes, bypassing
        init_tracker's own HSV+skeleton detection -- used for Pose B/C.
        Message format mirrors trackdlo's own initialize.py exactly
        (x,y,z FLOAT32 + rgba UINT32 fields)."""
        import rospy
        import std_msgs.msg
        from sensor_msgs.msg import PointCloud2, PointField
        from sensor_msgs import point_cloud2 as pcl2

        header = std_msgs.msg.Header()
        header.frame_id = RESULT_FRAME_ID
        header.stamp = rospy.Time.now()
        fields = [
            PointField("x", 0, PointField.FLOAT32, 1),
            PointField("y", 4, PointField.FLOAT32, 1),
            PointField("z", 8, PointField.FLOAT32, 1),
            PointField("rgba", 12, PointField.UINT32, 1),
        ]
        rgba = struct.unpack("I", struct.pack("BBBB", 255, 40, 40, 255))[0]
        pc = np.hstack([seed_cam_frame, np.full((len(seed_cam_frame), 1), rgba)]).astype(object)
        pc[:, 3] = pc[:, 3].astype(int)
        msg = pcl2.create_cloud(header, fields, pc)
        self.init_nodes_pub.publish(msg)
        # latched publisher -- trackdlo_node.cpp's own init_nodes subscriber
        # (inside trackdlo_node.cpp, not initialize.py) picks this up same
        # as if init_tracker had published it.
        self.init_nodes_result = seed_cam_frame

    def run(self) -> None:
        import rospy

        rate = rospy.Rate(self.publish_rate_hz)
        start_wall = rospy.Time.now()
        self.run_start_wall = start_wall
        for idx, cap in enumerate(self.captures):
            if rospy.is_shutdown():
                break
            stamp = start_wall + rospy.Duration.from_sec(float(cap["t"]))

            import imageio
            rgb = imageio.imread(cap["rgb_path"])
            depth = np.load(cap["depth_path"])
            self.stamp_to_idx[stamp] = idx

            publish_frame_ros(rgb, depth, self.k_matrix, self.height, self.width, stamp,
                               self.camera_info_pub, self.rgb_pub, self.depth_pub,
                               paint_markers=self.paint_markers)

            if idx % 30 == 0:
                print(f"[run_r1_tracking] pose {self.pose_label}: published {idx}/{len(self.captures)} "
                      f"({len(self.results_by_idx)} results so far)", flush=True)
            rate.sleep()

        print(f"[run_r1_tracking] pose {self.pose_label}: all frames published, "
              f"draining for 10s...", flush=True)
        rospy.sleep(10.0)

        # Diagnostic summary (stale-seed hypothesis check).
        if self.init_nodes_wall is None:
            print(f"[run_r1_tracking] pose {self.pose_label}: DIAGNOSTIC -- no "
                  f"/trackdlo/init_nodes ever received; can't check seed-vs-tracking gap.",
                  flush=True)
        elif self.first_result_idx is None:
            print(f"[run_r1_tracking] pose {self.pose_label}: DIAGNOSTIC -- seed landed but "
                  f"NO real tracked frame ever came back this session.", flush=True)
        else:
            seed_delay_sec = (self.init_nodes_wall - self.run_start_wall).to_sec()
            seed_delay_frames = seed_delay_sec * self.publish_rate_hz
            gap_frames = self.first_result_idx - seed_delay_frames
            print(f"[run_r1_tracking] pose {self.pose_label}: DIAGNOSTIC SUMMARY -- seed "
                  f"landed at ~frame {seed_delay_frames:.1f} ({seed_delay_sec:.3f}s after "
                  f"publish start); first real tracked frame was idx {self.first_result_idx} "
                  f"({len(self.captures)} total captures) -- gap of ~{gap_frames:.1f} frames "
                  f"(~{gap_frames / self.publish_rate_hz:.3f}s) between seed and lock-on.", flush=True)


def _launch_trackdlo_node(with_init_tracker: bool, tau_vis: float = VISIBILITY_THRESHOLD):
    """Programmatically (re)launch the `trackdlo` C++ node, and
    `init_tracker` only if with_init_tracker=True (Pose A only). Returns
    a roslaunch.parent.ROSLaunchParent-like handle with .shutdown()."""
    import roslaunch
    import rospy

    node_trackdlo = roslaunch.core.Node(
        package="trackdlo", node_type="trackdlo", name="trackdlo", output="screen",
    )
    # Full param set from trackdlo.launch's <node name="trackdlo"> block
    # (lines 20-62) -- algorithm-tuning params (beta..downsample_leaf_size)
    # are the launch file's recommended defaults verbatim, not camera/scene
    # specific.
    for key, val in {
        "camera_info_topic": CAMERA_INFO_TOPIC, "rgb_topic": RGB_TOPIC, "depth_topic": DEPTH_TOPIC,
        "result_frame_id": RESULT_FRAME_ID, "hsv_threshold_upper_limit": HSV_UPPER,
        "hsv_threshold_lower_limit": HSV_LOWER, "beta": BETA, "lambda": LAMBDA, "alpha": ALPHA,
        "mu": MU, "max_iter": MAX_ITER, "tol": TOL, "k_vis": K_VIS, "d_vis": D_VIS,
        "visibility_threshold": tau_vis, "dlo_pixel_width": DLO_PIXEL_WIDTH,
        "beta_pre_proc": BETA_PRE_PROC, "lambda_pre_proc": LAMBDA_PRE_PROC, "lle_weight": LLE_WEIGHT,
        "downsample_leaf_size": DOWNSAMPLE_LEAF_SIZE, "multi_color_dlo": MULTI_COLOR_DLO,
    }.items():
        rospy.set_param(f"/trackdlo/{key}", val)

    nodes = [node_trackdlo]
    if with_init_tracker:
        node_init = roslaunch.core.Node(
            package="trackdlo", node_type="initialize.py", name="init_tracker", output="screen",
        )
        # Full param set from trackdlo.launch's <node name="init_tracker"> block (lines 66-78).
        for key, val in {
            "camera_info_topic": CAMERA_INFO_TOPIC, "rgb_topic": RGB_TOPIC, "depth_topic": DEPTH_TOPIC,
            "result_frame_id": RESULT_FRAME_ID, "num_of_nodes": NUM_NODES,
            "multi_color_dlo": MULTI_COLOR_DLO, "visualize_initialization_process": VISUALIZE_INIT,
            "hsv_threshold_upper_limit": HSV_UPPER, "hsv_threshold_lower_limit": HSV_LOWER,
        }.items():
            rospy.set_param(f"/init_tracker/{key}", val)
        nodes.append(node_init)

    uuid = roslaunch.rlutil.get_or_generate_uuid(None, False)
    roslaunch.configure_logging(uuid)
    launch = roslaunch.scriptapi.ROSLaunch()
    launch.parent = roslaunch.parent.ROSLaunchParent(uuid, [])
    launch.start()
    for node in nodes:
        launch.launch(node)
    return launch


def main() -> int:
    import argparse
    import rospy

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pose", choices=POSES, default=None,
        help="Run only this single pose (real init_tracker, no bridging/handoff from a "
             "previous pose) instead of the full A->B->C loop -- for dry-running one pose "
             "at a time before trusting the full automated sequence. Saved output filename "
             "gets a _{pose} suffix so it doesn't collide with a full-run output.",
    )
    parser.add_argument(
        "--rate", type=float, default=DEFAULT_PUBLISH_RATE_HZ,
        help=f"Frame publish rate in Hz (default {DEFAULT_PUBLISH_RATE_HZ}). The trackdlo node "
             f"tracks at ~12 fps; publishing faster makes its message_filters queue drop frames. "
             f"Use {FRAME_RATE_HZ} for the original 30 Hz (frame-dropping) behaviour. Lower = "
             f"more frames tracked, longer wall-clock run.",
    )
    parser.add_argument(
        "--tau-vis", type=float, default=VISIBILITY_THRESHOLD,
        help=f"visibility_threshold (tau_vis) passed to the trackdlo node (default "
             f"{VISIBILITY_THRESHOLD}). The TrackDLO authors' value is 0.005; the default here "
             f"is bumped to give a freshly-(re)initialised pose recovery slack before the "
             f"node's 'no visible nodes -> skip' guard bails. Set 0.005 to run the stock config.",
    )
    parser.add_argument("--no-paint-markers", dest="paint_markers", action="store_false",
                        help="Disable the default recoloring of the green/cyan end-marker "
                             "segments to cable-red (see _paint_end_markers). Painting is ON "
                             "by default -- it removes the color-mask blind spot at node 0 / "
                             "node 19; use this only to reproduce the pre-fix behavior.")
    args = parser.parse_args()
    poses_to_run = (args.pose,) if args.pose else POSES
    output_path = OUTPUT_DIR / f"combined_tracked_trajectory_{args.pose}.npz" if args.pose else OUTPUT_NPZ

    rospy.init_node("run_r1_tracking")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    _suffix = output_path.stem.replace("combined_tracked_trajectory", "").lstrip("_") or "default"
    run_metadata = dict(
        num_nodes=NUM_NODES, publish_rate_hz=args.rate, tau_vis=args.tau_vis,
        beta=BETA, lambda_=LAMBDA, alpha=ALPHA, mu=MU, max_iter=MAX_ITER, tol=TOL,
        lle_weight=LLE_WEIGHT, dlo_pixel_width=DLO_PIXEL_WIDTH, hsv_lower=HSV_LOWER, hsv_upper=HSV_UPPER,
        paint_markers=args.paint_markers, poses_run=list(poses_to_run),
    )
    meta_path = OUTPUT_DIR / f"run_metadata_{_suffix}.json"
    with open(meta_path, "w") as f:
        json.dump(run_metadata, f, indent=2)
    print(f"[run_r1_tracking] wrote run metadata to {meta_path}", flush=True)

    combined_t: list[float] = []
    combined_nodes: list[np.ndarray] = []
    combined_node_support: list[tuple[float, np.ndarray]] = []  # Bucket-3
    combined_tracking_diag: list[tuple] = []  # Bucket-3
    prev_world_last_frame: np.ndarray | None = None

    for i, pose_label in enumerate(poses_to_run):
        meta, captures = load_pose_captures(pose_label)
        k_matrix = np.array(meta["rgb_intrinsics"])
        height, width = meta["resolution"]["height"], meta["resolution"]["width"]
        is_first = i == 0

        # Every pose gets its OWN skeleton detection (init_tracker). The
        # earlier design seeded B/C from the previous pose's last (by then
        # heavily occluded) frame carried across the transit blackout -- that
        # seed landed too far from the cable and trackdlo's "no visible
        # nodes -> skip" guard meant CPD never ran to correct it. A fresh
        # detection is guaranteed on-cable. Node *identity* is still chained:
        # each pose's whole result block is flipped (if needed) to line its
        # endpoints up with the previous pose's last frame -- so node i means
        # the same material point across the full A->B->C timeline.
        launch = _launch_trackdlo_node(with_init_tracker=True, tau_vis=args.tau_vis)
        rospy.sleep(2.0)

        session = PoseSession(pose_label, captures, k_matrix, height, width, publish_rate_hz=args.rate,
                              paint_markers=args.paint_markers)
        print(f"[run_r1_tracking] pose {pose_label}: publishing {len(captures)} frames at "
              f"{args.rate:.1f} Hz (~{len(captures)/args.rate:.0f}s)", flush=True)
        session.run()

        # this pose's frames, in world coords (each with its own extrinsic)
        pose_t: list[float] = []
        pose_frames: list[np.ndarray] = []
        for idx in sorted(session.results_by_idx.keys()):
            cap = captures[idx]
            nodes_cam = session.results_by_idx[idx]
            if nodes_cam.shape[0] != NUM_NODES:
                print(f"[run_r1_tracking] pose {pose_label} frame {idx}: got {nodes_cam.shape[0]} "
                      f"nodes, expected {NUM_NODES} -- skipping.", flush=True)
                continue
            w2c = world_to_cam_transform(np.asarray(cap["rgb_cam_pos"]), np.asarray(cap["rgb_cam_quat"]))
            pose_t.append(float(cap["t"]))
            pose_frames.append(cam_to_world(nodes_cam, w2c))

        if not pose_frames:
            print(f"[run_r1_tracking] pose {pose_label}: NO tracked frames -- "
                  "check init_tracker/HSV. Skipping this pose.", flush=True)
            launch.stop()
            rospy.sleep(2.0)
            continue

        # resolve node order
        if is_first or prev_world_last_frame is None:
            import imageio
            first_rgb = imageio.imread(captures[0]["rgb_path"])
            flip = (session.init_nodes_result is not None
                    and _marker_flip(first_rgb, session.init_nodes_result, k_matrix))
            why = "marker-anchored (node 0 = r2 grasp end)"
        else:
            flip = _endpoint_flip(pose_frames[0], prev_world_last_frame)
            why = f"chained to pose {poses_to_run[i - 1]}'s last frame"
        if flip:
            pose_frames = [f[::-1].copy() for f in pose_frames]
        print(f"[run_r1_tracking] pose {pose_label}: {len(pose_frames)}/{len(captures)} frames, "
              f"node order {'REVERSED' if flip else 'forward'} ({why})", flush=True)

        combined_t.extend(pose_t)
        combined_nodes.extend(pose_frames)
        combined_node_support.extend(session.node_support_log)
        combined_tracking_diag.extend(session.tracking_diag_log)
        prev_world_last_frame = pose_frames[-1]

        launch.stop()
        rospy.sleep(2.0)  # let the old trackdlo node fully release its topics before the next pose

    t_arr = np.array(combined_t, dtype=np.float64)
    nodes_arr = np.array(combined_nodes, dtype=np.float64) if combined_nodes else np.zeros((0, NUM_NODES, 3))
    np.savez(output_path, t=t_arr, nodes=nodes_arr)
    print(f"[run_r1_tracking] saved {len(t_arr)} tracked frames to {output_path}", flush=True)

    # Bucket-3: companion diagnostics file. Each stream keeps its OWN `t`
    # (not resampled onto t_arr) -- nearest-matching is left to the
    # downstream join step.
    diag_path = OUTPUT_DIR / f"tracker_diag_{_suffix}.npz"
    ns_t = np.array([x[0] for x in combined_node_support], dtype=np.float64)
    ns_vals = np.array([x[1] for x in combined_node_support], dtype=np.float64) if combined_node_support else np.zeros((0, NUM_NODES))
    diag_t = np.array([x[0] for x in combined_tracking_diag], dtype=np.float64)
    diag_arr = np.array([x[1:] for x in combined_tracking_diag], dtype=np.float64) if combined_tracking_diag else np.zeros((0, 4))
    np.savez(
        diag_path,
        node_support_t=ns_t, node_support=ns_vals,
        diag_t=diag_t,
        iterations=diag_arr[:, 0] if len(diag_arr) else np.zeros(0),
        converged=diag_arr[:, 1].astype(bool) if len(diag_arr) else np.zeros(0, dtype=bool),
        tracking_step_ms=diag_arr[:, 2] if len(diag_arr) else np.zeros(0),
        cloud_size=diag_arr[:, 3] if len(diag_arr) else np.zeros(0),
    )
    print(f"[run_r1_tracking] saved {len(ns_t)} node_support + {len(diag_t)} tracking_diag "
          f"records to {diag_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Regime 2 / Step 1: per-frame cable ground truth for the V4 freeze sequence.

The V4 capture (results/r1_eye_in_hand_calib_v4/pose_{A,B,C}/) stores RGB +
depth + camera pose + r1 qpos + the two welded endpoint positions per
frame, but NOT the full 20-node cable shape -- that is a physics state,
not recoverable from what is on disk. This script re-runs ONLY the MuJoCo
physics of the (fully deterministic) V4 sequence, headless and with no
rendering, and reads data.xpos for all 20 red_cable nodes at exactly the
same capture instants the recorder used, plus a per-node VISIBLE /
OCCLUDED / OUT_OF_FRAME label for r1's wrist camera.

It does NOT touch any existing file in results/r1_eye_in_hand_calib_v4/ --
it only writes a small `ground_truth.npz` into each pose_{label}/ dir:

    t          (N,)        capture time, matches metadata.json captures[].t
    gt_nodes   (N, 20, 3)  world-frame position of red_cable_00..19
    occlusion  (N, 20)     0=VISIBLE 1=OCCLUDED 2=OUT_OF_FRAME (r1_d435i_rgb)
    occluders  (N, 20)     object: geom name that occluded each node, or None

With --full it ALSO records a continuous ~30 Hz stream over the whole
replay (independent of the per-pose hold windows) into CALIB_ROOT:

    bridge_kinematics.npz  -- the r2/r3 GRIPPER BODY world poses (the
        proprioception signal a real robot has during a transit blackout:
        joint-encoder FK of the gripper), plus a one-time grasp offset.
        Regime-2's kinematic transit-gap bridge reconstructs each grasped
        endpoint during a gap as gripper_pose (x) grasp_offset from THIS
        file -- never from cable ground truth.
    ground_truth_full.npz  -- the full 20-node cable GT + wrist-camera
        occlusion on the same grid, for OFFLINE SCORING of the bridge
        ONLY. Never fed back into the trajectory.

Sim setup (XML load, solved waypoints, R1_HOME settle, planner) mirrors
run_dlo_v4_freeze.py exactly so the cable states are bit-identical; the
only change is swapping R1CalibCapture's render+save for GT/occlusion
recording on the same forward-ratchet capture schedule. A post-run check
asserts the recorded `t` and endpoint positions match the existing
metadata.json to within one physics timestep.

Usage:  python dump_v4_ground_truth.py
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import mujoco
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "mujoco_scenes"))
sys.path.insert(0, str(PROJECT_ROOT / "trajectories"))
sys.path.insert(0, str(PROJECT_ROOT / "rendering"))

from dlo_route_v4_freeze import DanglePlanner  # noqa: E402
from r1_camera_ik import CameraArm, set_solved_waypoints, target_at, rail_at, hold_window_at, hold_windows  # noqa: E402
from r1_camera_capture import capture_instants, CAPTURE_STOP_OVERRIDE, CAPTURE_WIDTH, CAPTURE_HEIGHT  # noqa: E402
from ground_truth.cable_gt import get_cable_ground_truth  # noqa: E402
from ground_truth.occlusion_labels import VISIBLE, OCCLUDED, OUT_OF_FRAME  # noqa: E402

SCENE_XML = PROJECT_ROOT / "triple_lite6_cable_routing_dlo_v4.xml"
CALIB_ROOT = PROJECT_ROOT / "results" / "r1_eye_in_hand_calib_v4"
RGB_CAMERA = "r1_d435i_rgb"

# ground_truth.occlusion_labels.label_node_occlusion is written for a
# world-fixed camera: its geom filter only drops group 2 (r2/r3's visual
# meshes in this scene) so r1's own D435i housing + arm meshes (group 4,
# on bodies r1_*) block every ray at ~2mm and label all nodes OCCLUDED.
# This wrist-camera variant iteratively skips any hit on an r1_* body so
# r1 never self-occludes, while r2/r3 (arms that CAN block r1's view of
# the cable) stay as real occluders.
_HIT_TOL_M = 0.02        # hit within this of the node -> that node is what the ray reached
_SKIP_STEP_M = 5e-4      # advance past an r1 self-hit before re-casting
_MAX_SKIPS = 12

# Exact solved waypoints from run_dlo_v4_freeze.py (lines 72-76) -- these
# drive r1's camera choreography and MUST match for the replay to
# reproduce the recorded camera poses.
QPOS_A = np.array([2.220378, -0.22875, 1.233465, 2.193019, -1.607109, 0.128412])
QPOS_B = np.array([-3.363153, -0.728326, 1.141771, 2.389318, -1.607312, 5.889064])
QPOS_C = np.array([-4.809949, 1.272049, 2.506624, 1.940535, -1.368249, -6.28319])

# r1's static camera-viewing home pose, held for the pre-sequence settle
# (run_dlo_v4_freeze.py:83).
R1_HOME = {1: 0.5, 2: -0.9, 3: 1.5, 4: 0.0, 5: -1.25, 6: 0.0}


def _r1_geom_mask(model: mujoco.MjModel) -> np.ndarray:
    """Boolean (ngeom,): True for geoms on an r1_* body (its own arm links,
    gripper, and D435i housing) -- the things r1's wrist camera must not be
    self-occluded by."""
    mask = np.zeros(model.ngeom, dtype=bool)
    for gid in range(model.ngeom):
        bname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[gid]))
        if bname and bname.startswith("r1_"):
            mask[gid] = True
    return mask


def label_node_occlusion_wrist(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    cam_id: int,
    gt_positions: np.ndarray,
    r1_geom_mask: np.ndarray,
    width: int,
    height: int,
) -> tuple[np.ndarray, list[str | None]]:
    """Per-node VISIBLE / OCCLUDED / OUT_OF_FRAME for a wrist-mounted
    camera. Same frustum test and hit tolerance as
    ground_truth.occlusion_labels.label_node_occlusion, but re-casts past
    any hit on an r1_* geom so r1 never self-occludes."""
    cam_pos = data.cam_xpos[cam_id].copy()
    rot_l2w = data.cam_xmat[cam_id].reshape(3, 3)
    R = rot_l2w.T.copy()
    R[1] = -R[1]
    R[2] = -R[2]
    t_vec = -R @ cam_pos
    fovy_rad = math.radians(model.cam_fovy[cam_id])
    fy = height / (2.0 * math.tan(fovy_rad / 2.0))
    cx, cy = width / 2.0, height / 2.0
    # all groups; r1 self-hits are handled by the skip loop, not the filter
    geomgroup = np.ones(6, dtype=np.uint8)
    gid_arr = np.zeros(1, dtype=np.int32)

    labels: list[int] = []
    occluders: list[str | None] = []
    for p_i in gt_positions:
        p_cam = R @ p_i + t_vec
        if p_cam[2] <= 0:
            labels.append(OUT_OF_FRAME); occluders.append(None); continue
        u = fy * p_cam[0] / p_cam[2] + cx
        v = fy * p_cam[1] / p_cam[2] + cy
        if u < 0 or u >= width or v < 0 or v >= height:
            labels.append(OUT_OF_FRAME); occluders.append(None); continue

        direction = p_i - cam_pos
        norm = float(np.linalg.norm(direction))
        if norm < 1e-9:
            labels.append(OCCLUDED); occluders.append(None); continue
        direction = direction / norm

        start = cam_pos.copy()
        label, occ = OCCLUDED, None
        for _ in range(_MAX_SKIPS):
            dist = mujoco.mj_ray(model, data, start, direction, geomgroup, 1, -1, gid_arr)
            hid = int(gid_arr[0])
            if hid < 0 or dist < 0:
                break  # nothing more along the ray -> stays OCCLUDED/None
            if r1_geom_mask[hid]:
                start = start + direction * (dist + _SKIP_STEP_M)
                continue
            hit_point = start + direction * dist
            hname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, hid)
            if np.linalg.norm(hit_point - p_i) < _HIT_TOL_M:
                label, occ = VISIBLE, None
            else:
                label, occ = OCCLUDED, hname
            break
        labels.append(label)
        occluders.append(occ)

    return np.array(labels, dtype=np.int32), occluders


def _build_settled_model() -> tuple[mujoco.MjModel, mujoco.MjData]:
    """Loads SCENE_XML and runs the r1-home settle -- identical setup used
    for both the real GTCapture replay and _precompute_arm_trajectory's dry
    pass, factored out so the two can't silently drift apart."""
    model = mujoco.MjModel.from_xml_path(str(SCENE_XML))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    r1_act: dict[int, int] = {}
    for j_num in R1_HOME:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"r1_joint{j_num}")
        for aid in range(model.nu):
            if int(model.actuator_trntype[aid]) == 0 and int(model.actuator_trnid[aid, 0]) == jid:
                r1_act[j_num] = aid
                break
    for j_num, angle in R1_HOME.items():
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"r1_joint{j_num}")
        data.qpos[model.jnt_qposadr[jid]] = angle
        data.ctrl[r1_act[j_num]] = angle
    mujoco.mj_forward(model, data)

    for _ in range(500):
        for j_num, angle in R1_HOME.items():
            data.ctrl[r1_act[j_num]] = angle
        mujoco.mj_step(model, data)

    return model, data


LOOKAHEAD_S = (1.0, 3.0)  # horizons for the per-node analytic self-occlusion-ahead prediction
_ARM_SAMPLE_DT = 0.05     # r2/r3 trajectory pre-pass sampling interval


def _precompute_arm_trajectory() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Dry physics pass (identical setup to the real replay, fully
    deterministic -- see module docstring) that ONLY records r2/r3's actual
    qpos on a fine time grid. Used to answer "what will r2/r3's arm
    configuration be at t+1s / t+3s" during the real replay, without a
    closed-form motion profile (DanglePlanner drives actuators toward
    targets, so realized qpos isn't a simple function of t) and without
    running physics twice interleaved (a clean second pass is simpler and
    safer than trying to peek ahead inside the first one)."""
    model, data = _build_settled_model()
    planner = DanglePlanner(model, data)
    r1_cam_arm = CameraArm(model, data)
    r1_rail_aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "r1_rail_x")
    r2_qids = planner._arms["r2"]["qpos_ids"]
    r3_qids = planner._arms["r3"]["qpos_ids"]

    stop_t = max(min(end, CAPTURE_STOP_OVERRIDE.get(label, end)) for label, (_, end) in hold_windows().items())
    stop_t += max(LOOKAHEAD_S)

    ts: list[float] = []
    r2s: list[np.ndarray] = []
    r3s: list[np.ndarray] = []
    next_sample = [0.0]

    class _DoneSampling(Exception):
        pass

    def _on_frame(model, data, t):
        target_qpos = target_at(t)
        if target_qpos is not None:
            r1_cam_arm.set_ctrl_to_qpos(target_qpos)
        data.ctrl[r1_rail_aid] = rail_at(t)
        if t >= next_sample[0]:
            ts.append(float(t))
            r2s.append(data.qpos[list(r2_qids)].copy())
            r3s.append(data.qpos[list(r3_qids)].copy())
            next_sample[0] += _ARM_SAMPLE_DT
        if t > stop_t:
            raise _DoneSampling

    print(f"[gt] pre-pass: recording r2/r3 trajectory for lookahead self-occlusion "
          f"(up to t={stop_t:.1f}s)...", flush=True)
    try:
        planner.run_sequence(on_frame=_on_frame)
    except _DoneSampling:
        pass
    print(f"[gt] pre-pass done: {len(ts)} samples over t={ts[0]:.1f}-{ts[-1]:.1f}s", flush=True)
    return np.array(ts), np.stack(r2s), np.stack(r3s)


class _AllCaptured(RuntimeError):
    """Raised from on_frame once every pose's ratchet is exhausted, to bail
    out of planner.run_sequence early -- the remaining sim time (t~=51.5s
    to 140s, r2's waypoint carry) has no captures and is pure wasted
    stepping. run_sequence calls on_frame directly (no try/except), so this
    propagates straight out to main()."""


class GTCapture:
    """Mirrors R1CalibCapture's per-pose forward-ratchet capture schedule,
    but records cable GT + occlusion instead of rendering/saving images.

    Bucket-2 addition (failure-predictor logging, 2026-09-11): also records,
    at the same per-capture-instant cadence, r2/r3's current qpos + gripper
    world pose (occluder geometry a real robot knows from its own FK), a
    per-node material tag (marker vs cable -- static), and a per-node
    analytic self-occlusion-ahead prediction at LOOKAHEAD_S horizons. The
    lookahead re-uses the CURRENT cable node positions against r2/r3's
    FUTURE (pre-pass-recorded) configuration -- predicting future cable
    shape is out of scope; this answers "will an arm sweep into the
    sightline of roughly where this node is now," which is the actual
    causal mechanism behind the occlusion failures investigated today."""

    def __init__(self, pose_windows: dict[str, tuple[float, float]], r1_geom_mask: np.ndarray,
                 arm_qpos_ids: dict[str, tuple[int, ...]] | None = None,
                 arm_gripper_body_ids: dict[str, int] | None = None,
                 arm_trajectory: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None) -> None:
        self._instants: dict[str, np.ndarray] = {
            label: capture_instants(s, e, stop_at=CAPTURE_STOP_OVERRIDE.get(label))
            for label, (s, e) in pose_windows.items()
        }
        self._next_idx: dict[str, int] = {label: 0 for label in pose_windows}
        self._records: dict[str, list[dict]] = {label: [] for label in pose_windows}
        self._r1_geom_mask = r1_geom_mask
        self._arm_qpos_ids = arm_qpos_ids
        self._arm_gripper_body_ids = arm_gripper_body_ids
        self._arm_t, self._arm_r2, self._arm_r3 = arm_trajectory if arm_trajectory is not None else (None, None, None)

    def all_done(self) -> bool:
        return all(self._next_idx[l] >= len(self._instants[l]) for l in self._next_idx)

    def _lookahead_occlusion(self, model, data, cam_id, gt, horizon_s: float) -> np.ndarray | None:
        """Re-labels `gt` (current cable positions) against r2/r3's
        recorded-future configuration at t+horizon_s. Restores r2/r3's
        actual current qpos before returning. None if the trajectory
        pre-pass doesn't cover that far ahead (near the end of the run)."""
        if self._arm_t is None:
            return None
        t_now = float(data.time)
        target_t = t_now + horizon_s
        if target_t > self._arm_t[-1]:
            return None
        future_idx = int(np.argmin(np.abs(self._arm_t - target_t)))

        r2_ids, r3_ids = list(self._arm_qpos_ids["r2"]), list(self._arm_qpos_ids["r3"])
        r2_now, r3_now = data.qpos[r2_ids].copy(), data.qpos[r3_ids].copy()
        data.qpos[r2_ids] = self._arm_r2[future_idx]
        data.qpos[r3_ids] = self._arm_r3[future_idx]
        mujoco.mj_forward(model, data)
        future_labels, _ = label_node_occlusion_wrist(
            model, data, cam_id, gt, self._r1_geom_mask, width=CAPTURE_WIDTH, height=CAPTURE_HEIGHT,
        )
        data.qpos[r2_ids], data.qpos[r3_ids] = r2_now, r3_now
        mujoco.mj_forward(model, data)
        return np.asarray(future_labels, dtype=np.int8)

    def maybe_capture(self, model, data, t: float, pose_label: str | None) -> None:
        if pose_label is None or pose_label not in self._next_idx:
            return
        idx = self._next_idx[pose_label]
        if idx >= len(self._instants[pose_label]):
            return
        if t < self._instants[pose_label][idx]:
            return
        gt = get_cable_ground_truth(model, data)  # (20, 3) world
        M = gt.shape[0]
        cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, RGB_CAMERA)
        labels, occluders = label_node_occlusion_wrist(
            model, data, cam_id, gt, self._r1_geom_mask,
            width=CAPTURE_WIDTH, height=CAPTURE_HEIGHT,
        )

        record = {
            "index": idx,
            "t": float(t),
            "gt_nodes": gt.astype(np.float64),
            "occlusion": np.asarray(labels, dtype=np.int8),
            "occluders": np.asarray(occluders, dtype=object),
            "material_tag": np.zeros(M, dtype=np.int8),  # 0=cable, 1=marker_start, 2=marker_end
        }
        record["material_tag"][0] = 1
        record["material_tag"][-1] = 2

        if self._arm_qpos_ids is not None:
            record["r2_qpos"] = data.qpos[list(self._arm_qpos_ids["r2"])].copy()
            record["r3_qpos"] = data.qpos[list(self._arm_qpos_ids["r3"])].copy()
            record["r2_gripper_pos"] = data.xpos[self._arm_gripper_body_ids["r2"]].copy()
            record["r2_gripper_quat"] = data.xquat[self._arm_gripper_body_ids["r2"]].copy()
            record["r3_gripper_pos"] = data.xpos[self._arm_gripper_body_ids["r3"]].copy()
            record["r3_gripper_quat"] = data.xquat[self._arm_gripper_body_ids["r3"]].copy()
            for h in LOOKAHEAD_S:
                ahead = self._lookahead_occlusion(model, data, cam_id, gt, h)
                record[f"self_occ_ahead_{h:g}s"] = ahead if ahead is not None else np.full(M, -1, dtype=np.int8)

        self._records[pose_label].append(record)
        self._next_idx[pose_label] += 1
        n = len(self._instants[pose_label])
        if idx % 60 == 0 or idx == n - 1:
            print(f"[gt] pose {pose_label}: {idx + 1}/{n} at t={t:.2f}s", flush=True)

    def save(self, output_root: Path) -> None:
        for label, recs in self._records.items():
            if not recs:
                print(f"[gt] pose {label}: no records -- skipping save", flush=True)
                continue
            out = output_root / f"pose_{label}" / "ground_truth.npz"
            arrays = dict(
                t=np.array([r["t"] for r in recs], dtype=np.float64),
                gt_nodes=np.stack([r["gt_nodes"] for r in recs]),
                occlusion=np.stack([r["occlusion"] for r in recs]),
                occluders=np.stack([r["occluders"] for r in recs]),
                material_tag=recs[0]["material_tag"],  # static across the run
            )
            if "r2_qpos" in recs[0]:
                for key in ("r2_qpos", "r3_qpos", "r2_gripper_pos", "r2_gripper_quat",
                            "r3_gripper_pos", "r3_gripper_quat"):
                    arrays[key] = np.stack([r[key] for r in recs])
                for h in LOOKAHEAD_S:
                    key = f"self_occ_ahead_{h:g}s"
                    arrays[key] = np.stack([r[key] for r in recs])
            np.savez(out, **arrays)
            print(f"[gt] wrote {out} ({len(recs)} frames"
                  f"{', +Bucket-2 fields' if 'r2_qpos' in recs[0] else ''})", flush=True)


def _quat2mat(quat: np.ndarray) -> np.ndarray:
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, np.ascontiguousarray(quat, dtype=np.float64))
    return R.reshape(3, 3)


def _to_local(body_pos: np.ndarray, body_quat: np.ndarray, world_pt: np.ndarray) -> np.ndarray:
    """world_pt expressed in the body frame given the body's world pose."""
    return _quat2mat(body_quat).T @ (np.asarray(world_pt) - np.asarray(body_pos))


class FullGridCapture:
    """Continuous ~30 Hz recorder over the whole replay (independent of the
    per-pose hold windows). Records the r2/r3 GRIPPER BODY world poses --
    the proprioception signal a real robot has during a transit blackout
    (joint-encoder FK of the gripper) -- plus, once the grasp welds engage,
    a one-time grasp offset (each grasped cable node expressed in its
    gripper's frame). Downstream the grasped endpoints during a gap are
    reconstructed as gripper_pose (x) grasp_offset; the interior cable
    shape is NEVER fed back into the trajectory.

    Also records the full 20-node cable GT + wrist occlusion on the same
    grid, for OFFLINE SCORING of the bridge ONLY (ground_truth_full.npz)."""

    # The grasp offset is NOT sampled at the instant both welds latch: the
    # weld constraint is compliant and takes ~4 s under the pick-up load to
    # relax into its steady state (measured: r2's node-0 offset drifts
    # ~3.5 mm over t=16..20 then holds). Sample it once the weld has
    # settled -- still well before the first transit gap (t~=25 s).
    _OFFSET_SETTLE_S = 4.0

    def __init__(self, model: mujoco.MjModel, r1_geom_mask: np.ndarray) -> None:
        self._r1_geom_mask = r1_geom_mask
        self._cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, RGB_CAMERA)
        self._r2_grip_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "r2_gripper_body")
        self._r3_grip_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "r3_gripper_body")
        self._r2_weld = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, "r2_cable_grasp_weld")
        self._r3_weld = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, "r3_cable_grasp_weld")
        for nm, i in (("r2_gripper_body", self._r2_grip_bid), ("r3_gripper_body", self._r3_grip_bid),
                      ("r2_cable_grasp_weld", self._r2_weld), ("r3_cable_grasp_weld", self._r3_weld)):
            if i < 0:
                raise ValueError(f"missing {nm} in model")
        self.grasp_offset_0: np.ndarray | None = None
        self.grasp_offset_19: np.ndarray | None = None
        self._t_welds_active: float | None = None   # instant both welds latched
        self._t_offset: float | None = None         # instant the offset was measured
        self._rows: list[dict] = []

    def record(self, model, data, t: float) -> None:
        gt = get_cable_ground_truth(model, data)  # (20, 3) world -- scoring only
        labels, _ = label_node_occlusion_wrist(
            model, data, self._cam_id, gt, self._r1_geom_mask,
            width=CAPTURE_WIDTH, height=CAPTURE_HEIGHT,
        )
        g2p = data.xpos[self._r2_grip_bid].copy()
        g2q = data.xquat[self._r2_grip_bid].copy()
        g3p = data.xpos[self._r3_grip_bid].copy()
        g3q = data.xquat[self._r3_grip_bid].copy()
        if self._t_welds_active is None and bool(data.eq_active[self._r2_weld]) and bool(data.eq_active[self._r3_weld]):
            self._t_welds_active = float(t)
            print(f"[full] both grasp welds latched at t={t:.2f}s -- "
                  f"offset sampled after {self._OFFSET_SETTLE_S:.0f}s settle", flush=True)
        if (self.grasp_offset_0 is None and self._t_welds_active is not None
                and t >= self._t_welds_active + self._OFFSET_SETTLE_S):
            self.grasp_offset_0 = _to_local(g2p, g2q, gt[0])
            self.grasp_offset_19 = _to_local(g3p, g3q, gt[19])
            self._t_offset = float(t)
            print(f"[full] grasp offsets calibrated at t={t:.2f}s  "
                  f"|o0|={np.linalg.norm(self.grasp_offset_0)*1000:.1f}mm  "
                  f"|o19|={np.linalg.norm(self.grasp_offset_19)*1000:.1f}mm", flush=True)
        self._rows.append({
            "t": float(t),
            "r2_grip_pos": g2p, "r2_grip_quat": g2q,
            "r3_grip_pos": g3p, "r3_grip_quat": g3q,
            "gt_nodes": gt.astype(np.float64),
            "occlusion": np.asarray(labels, dtype=np.int8),
        })
        if len(self._rows) % 300 == 0:
            print(f"[full] {len(self._rows)} rows (t={t:.2f}s)", flush=True)

    def save(self, output_root: Path) -> None:
        if not self._rows:
            print("[full] no rows recorded -- skipping save", flush=True)
            return
        if self.grasp_offset_0 is None:
            print("[full] WARNING: welds never both-active -- no grasp offset; "
                  "bridge_kinematics.npz will have NaN offsets", flush=True)
        r = self._rows
        o0 = self.grasp_offset_0 if self.grasp_offset_0 is not None else np.full(3, np.nan)
        o19 = self.grasp_offset_19 if self.grasp_offset_19 is not None else np.full(3, np.nan)
        np.savez(
            output_root / "bridge_kinematics.npz",
            t=np.array([x["t"] for x in r], dtype=np.float64),
            r2_grip_pos=np.stack([x["r2_grip_pos"] for x in r]),
            r2_grip_quat=np.stack([x["r2_grip_quat"] for x in r]),
            r3_grip_pos=np.stack([x["r3_grip_pos"] for x in r]),
            r3_grip_quat=np.stack([x["r3_grip_quat"] for x in r]),
            grasp_offset_0=o0,
            grasp_offset_19=o19,
            t_welds_active=np.array(self._t_welds_active if self._t_welds_active is not None else np.nan),
            t_offset=np.array(self._t_offset if self._t_offset is not None else np.nan),
        )
        np.savez(
            output_root / "ground_truth_full.npz",
            t=np.array([x["t"] for x in r], dtype=np.float64),
            gt_nodes=np.stack([x["gt_nodes"] for x in r]),
            occlusion=np.stack([x["occlusion"] for x in r]),
        )
        print(f"[full] wrote bridge_kinematics.npz + ground_truth_full.npz "
              f"({len(r)} rows, t={r[0]['t']:.2f}..{r[-1]['t']:.2f}s)", flush=True)


def _crosscheck_full(output_root: Path) -> None:
    """Reconstruct each grasped endpoint from the recorded GRIPPER pose +
    the one-time grasp offset and compare to the full-grid cable GT node
    0/19 -- a rigid weld should give sub-mm agreement post-grasp. Also
    report grid spacing."""
    bk = np.load(output_root / "bridge_kinematics.npz")
    gf = np.load(output_root / "ground_truth_full.npz")
    o0, o19 = bk["grasp_offset_0"], bk["grasp_offset_19"]
    if not np.isfinite(o0).all():
        print("[check-full] no grasp offset -- cannot reconstruct endpoints", flush=True)
        return

    def _recon(pos, quat, off):
        out = np.empty_like(pos)
        for i in range(len(pos)):
            out[i] = pos[i] + _quat2mat(quat[i]) @ off
        return out

    r0 = _recon(bk["r2_grip_pos"], bk["r2_grip_quat"], o0)
    r19 = _recon(bk["r3_grip_pos"], bk["r3_grip_quat"], o19)
    e0 = np.linalg.norm(r0 - gf["gt_nodes"][:, 0], axis=1)
    e19 = np.linalg.norm(r19 - gf["gt_nodes"][:, 19], axis=1)
    t_off = float(bk["t_offset"])
    m = bk["t"] >= t_off  # reconstruction is only meaningful from the offset instant on
    dt = np.diff(bk["t"])
    print(f"[check-full] grid n={len(bk['t'])}  dt mean={dt.mean()*1000:.1f}ms "
          f"max={dt.max()*1000:.1f}ms  span={bk['t'][0]:.2f}..{bk['t'][-1]:.2f}s", flush=True)
    print(f"[check-full] welds latched t={float(bk['t_welds_active']):.2f}s, "
          f"offset sampled t={t_off:.2f}s", flush=True)
    print(f"[check-full] endpoint reconstruction (gripper pose (x) grasp offset), "
          f"t>={t_off:.1f}s n={int(m.sum())}:", flush=True)
    print(f"[check-full]   node 0  err mean={e0[m].mean()*1000:.3f}mm  max={e0[m].max()*1000:.3f}mm", flush=True)
    print(f"[check-full]   node 19 err mean={e19[m].mean()*1000:.3f}mm  max={e19[m].max()*1000:.3f}mm", flush=True)
    ok = e0[m].max() < 2e-3 and e19[m].max() < 2e-3
    print(f"[check-full] {'OK' if ok else 'HIGH RECON ERROR -- check gripper body / offset'}", flush=True)


def _crosscheck(output_root: Path) -> None:
    """For every metadata.json capture instant, assert the replay recorded
    a GT row at the same t (within one physics timestep, 0.5 ms) and that
    its node 0 / node 19 match metadata's recorded grasp endpoints. The
    replay grid is denser than metadata (30 Hz vs the recorder's saved-RGB
    rate), so this is a subset match, not a count match."""
    ok = True
    for label in ("A", "B", "C"):
        gt_path = output_root / f"pose_{label}" / "ground_truth.npz"
        meta_path = output_root / f"pose_{label}" / "metadata.json"
        if not gt_path.exists() or not meta_path.exists():
            print(f"[check] pose {label}: missing file -- skipped", flush=True)
            continue
        gt = np.load(gt_path, allow_pickle=True)
        meta = json.loads(meta_path.read_text())
        caps = sorted(meta["captures"], key=lambda c: c["t"])
        key = {round(float(v), 6): i for i, v in enumerate(gt["t"])}
        gi = np.array([key.get(round(float(c["t"]), 6), -1) for c in caps])
        if (gi < 0).any():
            n_miss = int((gi < 0).sum())
            print(f"[check] pose {label}: {n_miss}/{len(caps)} metadata instants "
                  f"have NO matching replay GT row", flush=True)
            ok = False
            continue
        dt = np.abs(gt["t"][gi] - np.array([c["t"] for c in caps]))
        d0 = np.array([
            np.linalg.norm(gt["gt_nodes"][gi[i], 0] - np.array(caps[i]["grasp_node0_pos"]))
            for i in range(len(caps)) if caps[i].get("grasp_node0_pos") is not None
        ])
        d19 = np.array([
            np.linalg.norm(gt["gt_nodes"][gi[i], 19] - np.array(caps[i]["grasp_node19_pos"]))
            for i in range(len(caps)) if caps[i].get("grasp_node19_pos") is not None
        ])
        vis_frac = float(np.mean(gt["occlusion"] == 0))
        msg = (f"[check] pose {label}: metadata n={len(caps)} matched into replay "
               f"n={len(gt['t'])}  max|dt|={dt.max()*1000:.3f}ms  "
               f"endpoint0 max={d0.max()*1000 if d0.size else float('nan'):.2f}mm  "
               f"endpoint19 max={d19.max()*1000 if d19.size else float('nan'):.2f}mm  "
               f"VISIBLE frac={vis_frac:.2f}")
        print(msg, flush=True)
        if dt.max() > 5e-4 or (d0.size and d0.max() > 1e-3) or (d19.size and d19.max() > 1e-3):
            print(f"[check] pose {label}: OUT OF TOLERANCE", flush=True)
            ok = False
    print(f"[check] {'ALL OK' if ok else 'FAILURES -- do not trust this GT'}", flush=True)


def main() -> int:
    sys.stdout.reconfigure(line_buffering=True)
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--full", action="store_true",
                    help="also record the continuous ~30 Hz gripper-pose stream "
                         "(bridge_kinematics.npz) + full-grid cable GT (ground_truth_full.npz)")
    args = ap.parse_args()

    if not SCENE_XML.exists():
        print(f"missing {SCENE_XML} -- run generate_triple_scene_v4.py first", file=sys.stderr)
        return 1

    set_solved_waypoints(QPOS_A, QPOS_B, QPOS_C)

    arm_trajectory = _precompute_arm_trajectory()  # Bucket-2 lookahead pre-pass

    model, data = _build_settled_model()

    planner = DanglePlanner(model, data)
    r1_cam_arm = CameraArm(model, data)
    r1_mask = _r1_geom_mask(model)
    gt_capture = GTCapture(
        hold_windows(), r1_mask,
        arm_qpos_ids={"r2": planner._arms["r2"]["qpos_ids"], "r3": planner._arms["r3"]["qpos_ids"]},
        arm_gripper_body_ids={"r2": planner._arms["r2"]["gripper_body_id"], "r3": planner._arms["r3"]["gripper_body_id"]},
        arm_trajectory=arm_trajectory,
    )
    full_capture = FullGridCapture(model, r1_mask) if args.full else None
    r1_rail_aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "r1_rail_x")

    def _on_frame(model, data, t):
        target_qpos = target_at(t)
        if target_qpos is not None:
            r1_cam_arm.set_ctrl_to_qpos(target_qpos)
        data.ctrl[r1_rail_aid] = rail_at(t)
        window = hold_window_at(t)
        gt_capture.maybe_capture(model, data, t, window[0] if window is not None else None)
        if full_capture is not None:
            full_capture.record(model, data, t)
        if gt_capture.all_done():
            raise _AllCaptured

    print("[gt] replaying V4 physics (headless, no rendering)...", flush=True)
    try:
        planner.run_sequence(on_frame=_on_frame)
    except _AllCaptured:
        print("[gt] all capture instants recorded -- stopping replay early "
              "(remaining sim time has no captures).", flush=True)

    gt_capture.save(CALIB_ROOT)
    _crosscheck(CALIB_ROOT)
    if full_capture is not None:
        full_capture.save(CALIB_ROOT)
        _crosscheck_full(CALIB_ROOT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

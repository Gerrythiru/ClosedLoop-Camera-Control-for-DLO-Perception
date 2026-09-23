"""V5 fork of r1_camera_ik.py. Forked (2026-09-08) because V5's wrist-camera
choreography diverges from V4's: r1 stays STATIONARY on its rail at the
ROI-centred position (RAIL_A/B/C = -0.015, rail held constant for the whole
run) instead of translating between three rail positions per pose. POSE_A/B/C
below are still V4's values -- placeholders until re-tuned for the shorter
V5 cable/box. Everything else (CameraArm IK, the A/B/C arm-pose timing
schedule) is unchanged from the shared module. The shared r1_camera_ik.py
keeps V4's moving-rail behaviour for the V4 pipeline.

World-frame pose choreography for r1's wrist camera (r1_d435i_rgb),
driven independently of DanglePlanner during the settle-only freeze
sequence (dlo_route_v2_Freeze.py, left untouched).

Design note -- why kinematic IK, not live dynamics-coupled IK: an earlier
version of this module ran Jacobian-based IK correction every physics step
while stepping real dynamics (accumulating a persistent ctrl setpoint from
a Jacobian-derived joint-velocity estimate). That was empirically unstable
for large target jumps (from _r1_home to Pose A spans tens of degrees on
several joints) -- verified directly: position/orientation error oscillated
between roughly 400-900mm / 55-165deg over thousands of steps instead of
converging, because the correction was computed from the CURRENT (position-
actuator-lagged) qpos every step and kept accumulating on top of an
already-in-flight setpoint, overshooting and correcting past the target
repeatedly.

Fix: decouple IK solving from real-time dynamics entirely.
  1. Solve each target pose's joint configuration KINEMATICALLY --
     mj_kinematics() only updates xpos/xmat/cam_xpos/cam_xmat from qpos, no
     dynamics/actuator/contact stepping -- so large, stable Newton-style
     correction steps can be taken every iteration with no risk of dynamic
     instability (verified: converges cleanly in a few hundred iterations,
     see solve_ik_kinematic).
  2. At runtime, just command the SOLVED joint angles as ctrl setpoints
     directly (interpolating in JOINT SPACE between consecutive solved
     waypoints during transition windows) and let the position actuators'
     own dynamics (kp=2000, verified to converge a large fixed-ctrl jump
     cleanly within ~600 physics steps / 0.3s) do the real-time tracking.
     No live Jacobian correction runs during the actual physics-stepped
     sequence at all.

Orientation-error sign convention and the xyaxes->quat construction were
both verified empirically against MuJoCo's own ground truth: mju_subQuat's
direction (0.1 rad about +Z -> err=[0,0,0.1] exactly), and xyaxes_to_quat
checked against the compiled cam_mat0 of the existing `cable_side` camera
(exact match, no transpose needed).
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import mujoco

ARM_PREFIX = "r1"
CAMERA_NAME = "r1_d435i_rgb"


def xyaxes_to_quat(xyaxes: tuple[float, ...]) -> np.ndarray:
    """MJCF xyaxes (x0,x1,x2, y0,y1,y2) -> unit quaternion (wxyz).
    Re-orthonormalizes: x normalized as given; z = normalize(cross(x,y));
    y re-derived as cross(z,x) so hand-tuned viewer xyaxes pairs that
    aren't perfectly orthonormal still produce a valid rotation matrix."""
    xa = np.asarray(xyaxes[0:3], dtype=np.float64)
    ya = np.asarray(xyaxes[3:6], dtype=np.float64)
    x = xa / np.linalg.norm(xa)
    z = np.cross(x, ya)
    z /= np.linalg.norm(z)
    y = np.cross(z, x)
    mat = np.column_stack([x, y, z]).reshape(-1)
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, mat)
    return quat


def slerp(q0: np.ndarray, q1: np.ndarray, frac: float) -> np.ndarray:
    """Shortest-path SLERP between two unit quaternions (wxyz). Not used
    for runtime joint-space interpolation (see module docstring) but kept
    for completeness / diagnostics."""
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    dot = min(1.0, dot)
    if dot > 0.9995:
        out = q0 + frac * (q1 - q0)
        return out / np.linalg.norm(out)
    theta0 = np.arccos(dot)
    theta = theta0 * frac
    q_perp = q1 - q0 * dot
    q_perp /= np.linalg.norm(q_perp)
    return q0 * np.cos(theta) + q_perp * np.sin(theta)


class CameraArm:
    """r1's 6-joint chain + r1_d435i_rgb camera state."""

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        prefix: str = ARM_PREFIX,
        camera_name: str = CAMERA_NAME,
    ) -> None:
        self.model = model
        self.data = data
        joint_ids = tuple(self._joint_id(f"{prefix}_joint{i}") for i in range(1, 7))
        self.qpos_ids = tuple(int(model.jnt_qposadr[j]) for j in joint_ids)
        self.dof_ids = tuple(int(model.jnt_dofadr[j]) for j in joint_ids)
        self.act_ids = tuple(self._actuator_for_joint(j) for j in joint_ids)
        self.jnt_range = np.array([model.jnt_range[j] for j in joint_ids])
        self.cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
        if self.cam_id < 0:
            raise ValueError(f"Missing camera {camera_name}")
        self.cam_body_id = int(model.cam_bodyid[self.cam_id])
        self._ctrl_min = model.actuator_ctrlrange[:, 0].copy()
        self._ctrl_max = model.actuator_ctrlrange[:, 1].copy()
        self._ctrl_limited = model.actuator_ctrllimited.copy()

    def _joint_id(self, name: str) -> int:
        jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            raise ValueError(f"Missing joint {name}")
        return jid

    def _actuator_for_joint(self, jid: int) -> int:
        for aid in range(self.model.nu):
            if int(self.model.actuator_trnid[aid, 0]) == jid:
                return aid
        raise ValueError(f"No actuator for joint id {jid}")

    @staticmethod
    def _update_kinematics(model: mujoco.MjModel, data: mujoco.MjData) -> None:
        """Minimal chain needed for both cam_xpos/cam_xmat AND mj_jac to be
        correct after only changing qpos (no dynamics stepping needed).
        VERIFIED empirically: mj_kinematics() ALONE leaves cam_xpos/cam_xmat
        stale (NOT part of mj_kinematics -- camera/light poses are computed
        by the separate mj_camlight()) and mj_jac's output wrong (it reads
        `cdof`, which mj_comPos computes, not mj_kinematics). All three
        calls are required, in this order, matching mj_forward's own
        internal sequence but skipping the expensive dynamics/contact/
        constraint stages this IK solver doesn't need."""
        mujoco.mj_kinematics(model, data)
        mujoco.mj_comPos(model, data)
        mujoco.mj_camlight(model, data)

    def get_qpos(self) -> np.ndarray:
        return np.array([self.data.qpos[q] for q in self.qpos_ids], dtype=np.float64)

    def set_qpos(self, q: np.ndarray) -> None:
        for li, qid in enumerate(self.qpos_ids):
            self.data.qpos[qid] = q[li]

    def cam_pose(self) -> tuple[np.ndarray, np.ndarray]:
        """Current camera world pos (3,) and quat wxyz (4,). Valid after
        mj_kinematics/mj_forward/mj_step."""
        pos = self.data.cam_xpos[self.cam_id].copy()
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, self.data.cam_xmat[self.cam_id])
        return pos, quat

    def set_ctrl_to_qpos(self, q: np.ndarray) -> None:
        """Command the position actuators directly to joint targets q,
        clamped to each actuator's ctrlrange (confirmed == jnt_range for
        all 6 r1 joints from the vendored Lite6 XML)."""
        for li, aid in enumerate(self.act_ids):
            value = float(q[li])
            if self._ctrl_limited[aid]:
                value = float(np.clip(value, self._ctrl_min[aid], self._ctrl_max[aid]))
            self.data.ctrl[aid] = value

    def solve_ik_kinematic(
        self,
        target_pos: np.ndarray,
        target_quat: np.ndarray,
        q_init: np.ndarray,
        *,
        pos_damping: float = 0.0025,
        rot_damping: float = 0.02,
        max_dq: float = 0.2,
        max_iters: int = 2000,
        pos_tol: float = 1e-4,
        rot_tol: float = 1e-4,
    ) -> tuple[np.ndarray, float, float]:
        """Pure-kinematic damped-least-squares IK: iterates mj_kinematics()
        (forward kinematics only, no dynamics/actuator/contact stepping) at
        successive qpos guesses, starting from q_init, until convergence or
        max_iters. Mutates self.data.qpos as a scratch space during the
        solve (restore it yourself afterward if the caller cares about
        data's qpos elsewhere -- callers in this codebase always
        immediately overwrite qpos with a real seed for the next solve or
        proceed to command ctrl, so no restore is done here).

        Returns (solved_qpos (6,), final_pos_err_m, final_rot_err_rad)."""
        m, d = self.model, self.data
        q = q_init.copy()
        self.set_qpos(q)
        self._update_kinematics(m, d)
        pos_err = rot_err = np.inf
        for _ in range(max_iters):
            cur_pos, cur_quat = self.cam_pose()
            pos_err_vec = target_pos - cur_pos
            rot_err_vec = np.zeros(3)
            mujoco.mju_subQuat(rot_err_vec, target_quat, cur_quat)
            pos_err = float(np.linalg.norm(pos_err_vec))
            rot_err = float(np.linalg.norm(rot_err_vec))
            if pos_err < pos_tol and rot_err < rot_tol:
                break

            jacp = np.zeros((3, m.nv))
            jacr = np.zeros((3, m.nv))
            mujoco.mj_jac(m, d, jacp, jacr, cur_pos, self.cam_body_id)
            jp = jacp[:, list(self.dof_ids)]
            jr = jacr[:, list(self.dof_ids)]

            dq_p = jp.T @ np.linalg.solve(jp @ jp.T + pos_damping * np.eye(3), pos_err_vec)
            dq_r = jr.T @ np.linalg.solve(jr @ jr.T + rot_damping * np.eye(3), rot_err_vec)
            dq = np.clip(dq_p + dq_r, -max_dq, max_dq)
            q = q + dq
            q = np.clip(q, self.jnt_range[:, 0], self.jnt_range[:, 1])
            self.set_qpos(q)
            self._update_kinematics(m, d)

        return q, pos_err, rot_err


# V5 target world poses (pos, quat), user-supplied 2026-09-08 (revised
# 2026-09-08 for the 75% cable/box) with r1 held at the fixed rail
# (RAIL_X = +0.10). Given as MJCF <camera pos=... xyaxes=.../>; xyaxes
# converted here. verify_r1_camera_reachability_v5.py solves the
# JOINT-SPACE waypoints (QPOS_A..QPOS_C) at RAIL_X and its printed output
# is pasted below + into run_dlo_v5_freeze.py / dump_v5_ground_truth.py.
# POSE_A revised 2026-09-09: raised + pushed back (pos z 1.185 -> 1.238,
# y -0.408 -> -0.436) and tilted more downward to clear r2's gripper/wrist
# from the cable's r2 end during the carry -- the previous Pose A lost node
# 0 to a CPD-LLE divergence at t~=25.5s when the r2 terminus went
# gripper-occluded from that viewpoint. This ALSO diverged (node0 up to
# ~340mm, see Today_SUMM) -- reinstated 2026-09-10 as the active pose (see
# below): the E4 sweep's replacement candidate was reachable/collision-free
# but its predicted node-0 visibility didn't hold up in closed-loop testing
# -- node 0's mask support was 0/81 throughout, because node 0 sits on the
# cable's GREEN marker_start segment, not the red-thresholded material, a
# color-mask blind spot no camera angle fixes (see Today_SUMM). E5-mask
# (repainting the marker to cable-red) is the real fix for that node.
# POSE_A = (np.array([0.085, -0.436, 1.238]),
#           xyaxes_to_quat((0.968, -0.250, 0.000, 0.178, 0.688, 0.704)))
#
# POSE_A replaced 2026-09-12 for the invoke_nbv mitigation-validation test
# (Q4/step 4, see Today_SUMM): user-supplied, deliberately chosen to
# produce genuine geometric occlusion (not the color-mask blind spot from
# the candidate below) for classify_risk.py's geometric_occlusion class to
# be tested against end-to-end -- capture/track/predict, then an NBV
# reposition, then re-check. Labeled "Occlusion_Test_pose_1" -- superseded
# 2026-09-14 by a second occlusion pose to grow the geometric_occlusion
# sample size (pose_1 gave only one clean lead-time measurement: node14,
# +1.00s). Kept here, commented, as a fallback / for re-use:
# POSE_A = (np.array([-0.005, -0.473, 1.025]),
#           xyaxes_to_quat((0.938, -0.346, -0.000, 0.168, 0.455, 0.875)))
#
# POSE_A replaced 2026-09-14 ("Occlusion_Test_pose_2"), user-supplied, at
# the SAME fixed RAIL_X=-0.025 as Pose B: REACHABLE + collision-free
# (A 2.79mm/1.15deg, B 4.52mm/1.10deg), clean home->A/A->B transitions
# (0/300 each), and e4b_moving_collision_check_v5.py confirmed no collision
# with the live r2/r3 carry.
# POSE_A = (np.array([-0.030, -0.519, 0.972]),
#           xyaxes_to_quat((0.943, -0.334, 0.000, 0.093, 0.264, 0.960)))
#
# POSE_A replaced 2026-09-21, user-supplied.
POSE_A = (np.array([-0.036, -0.474, 0.980]),
          xyaxes_to_quat((0.944, -0.330, -0.000, 0.125, 0.357, 0.926)))
#
# "Occlusion_Test_pose_3" candidate tried 2026-09-14 but UNREACHABLE at the
# fixed RAIL_X=-0.025 (10.32mm/3.80deg error, self-collisions AND a
# workstation-table collision at the best-of-25-multistart solve; both
# home->A and A->B transitions collided badly, 175/300 and 236/300). Not
# saved as the active pose pending a decision on how to proceed.
# POSE_A = (np.array([-0.068, -0.647, 0.852]),
#           xyaxes_to_quat((0.941, -0.338, 0.000, -0.018, -0.049, 0.999)))
#
# POSE_A candidate tried 2026-09-10, e4_reachable_pose_sweep_v5.py's (Q2)
# winning pick: az=300 el=40 radius=0.55m around the carry's ROI centroid --
# 99% both-grasped-ends-visible by the sweep's ray-cast (geometric
# occlusion) test over t=2-36s, REACHABLE + collision-free at production
# tolerance (verify_r1_camera_reachability_v5.py) including against the
# live r2/r3 carry (e4b_moving_collision_check_v5.py, 0/340 contacts) --
# but didn't fix the real failure (see above). Kept here, commented, in
# case a geometric-occlusion scenario needs it again:
# POSE_A = (np.array([0.385789, -0.352462, 1.162983]),
#           xyaxes_to_quat((0.866025, 0.5, -0.0, -0.321394, 0.55667, 0.766044)))
POSE_B = (np.array([0.450, -0.360, 1.165]),
          xyaxes_to_quat((0.877, 0.481, -0.000, -0.306, 0.558, 0.772)))
# TEMP (2026-09-08): Pose C disabled -- the choreography is A -> B (arrive
# B by t=38s) then hold B for the rest of the run (see _schedule()).
# Uncomment POSE_C + the C rows in _schedule() to restore A->B->C.
# verify_r1_camera_reachability_v5.py tolerates a missing POSE_C.
# POSE_C = (np.array([0.482, -0.122, 1.059]),
#           xyaxes_to_quat((0.612, 0.791, 0.000, -0.521, 0.404, 0.752)))

R1_HOME_QPOS = np.array([0.5, -0.9, 1.5, 0.0, -1.25, 0.0])

# Populated by verify_r1_camera_reachability.py's solved joint-space
# waypoints (REACHABLE or nearest-reachable substitute per pose). Runtime
# scheduling (SCHEDULE/target_at) interpolates directly between these
# qpos vectors -- no live IK during the physics-stepped sequence.
QPOS_A: Optional[np.ndarray] = None
QPOS_B: Optional[np.ndarray] = None
QPOS_C: Optional[np.ndarray] = None


def set_solved_waypoints(qpos_a: np.ndarray, qpos_b: np.ndarray, qpos_c: np.ndarray) -> None:
    """Called once (by run_dlo_v3_freeze.py, with values hardcoded in from
    verify_r1_camera_reachability.py's printed output) to populate the
    module-level joint-space waypoints target_at() interpolates between."""
    global QPOS_A, QPOS_B, QPOS_C
    QPOS_A, QPOS_B, QPOS_C = qpos_a, qpos_b, qpos_c


# V5: r1's mount is held STATIONARY on its rail for the entire run. RAIL_X
# is the single fixed rail displacement (base_x = 0.20 + RAIL_X).
# TEMP (2026-09-08): with the A->B choreography (Pose C disabled), A and B
# are simultaneously reachable + collision-free only for RAIL_X in
# [-0.10, +0.05] -- Pose B's camera at x=0.450 wants the base +X, Pose A's
# at x=-0.022 caps how far. -0.025 (base_x 0.175) is the middle of that
# band (A ~3mm/1.6deg, B ~4.5mm/1.1deg). (The full A->B->C choreography
# needed RAIL_X = +0.10 -- see git-less history / the commented C rows.)
# (Was a per-pose moving rail in V4 -- see the shared r1_camera_ik.py.)
#
# 2026-09-14: was briefly moved to 0.100 to make the (later reverted)
# "Occlusion_Test_pose_2" POSE_A reachable (unreachable at -0.025 --
# self-collision + colliding transitions). Reverted back to -0.025 along
# with POSE_A. If pose_2 is revisited, RAIL_X needs to move to 0.100 again
# (see the commented POSE_A above) -- swept band [0.000, 0.200] was clean
# for that pose; 0.100 gave margin, Pose A 2.27mm/1.40deg re-solved there,
# Pose B 4.58mm/1.21deg re-solved there, both transitions 0/300 clean.
RAIL_X = -0.025
RAIL_A = RAIL_B = RAIL_C = RAIL_X


def _rail_schedule():
    """V5: the rail is held constant at RAIL_X for the whole timeline --
    one hold window, no xfer segments (kept as a function for parity with
    _schedule() / in case anything introspects it)."""
    return [(0.0, 140.0, "hold", RAIL_X)]


def rail_at(t: float) -> float:
    """V5: constant rail ctrl target (world_x displacement from rest) --
    r1 does not move on the rail during the run."""
    return RAIL_X


# (start_s, end_s, kind, *waypoints) -- kind "hold_home" leaves r1 undriven
# by this module (stays on its pre-existing static R1_HOME_QPOS ctrl);
# "hold" freezes at the single given qpos (plus a pose label "A"/"B"/"C",
# used by hold_window_at/hold_windows below -- target_at() only ever reads
# waypoints[0] for holds, so the extra label is backward compatible);
# "xfer" linearly interpolates joint-space between two waypoints over the
# window.
#
# TEMP (2026-09-08): A -> B choreography, no Pose C. r1 holds Pose A to
# t=36s, transitions A->B over [36,38] (arrives B by t=38, user-specified),
# then holds Pose B for the rest of the run. To restore A->B->C, uncomment
# the C rows below + POSE_C above, and shorten B's window.
def _schedule():
    return [
        (0.0, 2.0, "hold_home"),
        (2.0, 36.0, "hold", QPOS_A, "A"),
        (36.0, 38.0, "xfer", QPOS_A, QPOS_B),
        (38.0, 140.0, "hold", QPOS_B, "B"),
        # (54.0, 56.0, "xfer", QPOS_B, QPOS_C),
        # (56.0, 140.0, "hold", QPOS_C, "C"),
    ]


def target_at(t: float) -> Optional[np.ndarray]:
    """Returns the joint-space qpos target (6,) for time t, or None during
    0-2s (r1 stays parked at R1_HOME_QPOS, driven by the existing
    pre-loop/static ctrl, not by this module). Raises if
    set_solved_waypoints() hasn't been called yet."""
    if QPOS_A is None:
        raise RuntimeError("call set_solved_waypoints() before target_at()")
    for start, end, kind, *waypoints in _schedule():
        if start <= t < end:
            if kind == "hold_home":
                return None
            if kind == "hold":
                return waypoints[0]
            q0, q1 = waypoints
            frac = np.clip((t - start) / (end - start), 0.0, 1.0)
            return q0 + (q1 - q0) * frac
    return QPOS_B  # t >= 140 safety fallback (TEMP: was QPOS_C; r1 holds B at end)


def hold_window_at(t: float) -> Optional[tuple[str, float, float]]:
    """(pose_label, start, end) if t falls within a labeled "hold" window
    ("A"/"B"/"C"), else None (during hold_home or an xfer transition).
    Does not require set_solved_waypoints() to have been called -- pose
    labels are static strings present in _schedule() even while QPOS_* are
    still None, unlike target_at()."""
    for start, end, kind, *rest in _schedule():
        if start <= t < end and kind == "hold" and len(rest) >= 2:
            return rest[1], start, end
    return None


def hold_windows() -> dict[str, tuple[float, float]]:
    """All labeled hold windows as {label: (start, end)} -- for callers
    that need the full set up front (e.g. R1CalibCapture's constructor)
    without duplicating the schedule's magic numbers. Safe to call before
    set_solved_waypoints()."""
    return {
        rest[1]: (start, end)
        for start, end, kind, *rest in _schedule()
        if kind == "hold" and len(rest) >= 2
    }


# 2026-09-22: fixed transit duration reused by NBVCameraScheduler below --
# matches _schedule()'s (36.0, 38.0, "xfer", ...) row's width.
XFER_SECONDS = 2.0


class NBVCameraScheduler:
    """Same pose list/order/transit as _schedule() (hold_home -> hold A ->
    xfer A->B -> hold B), but each "hold" phase's END is driven by an
    external advance signal (notify_advance) instead of a fixed t.
    "hold_home"/"xfer" phases keep their existing fixed durations -- only
    "hold" phases are open-ended. Used only under run_dlo_v5_freeze.py's
    --nbv flag; _schedule()/target_at()/hold_window_at()/hold_windows()
    above are untouched and still drive the default/--ros-live paths."""

    def __init__(self) -> None:
        if QPOS_A is None:
            raise RuntimeError("call set_solved_waypoints() before constructing NBVCameraScheduler")
        # (kind, qpos_target_or_None, pose_label_or_None, duration_or_None) --
        # duration is None for "hold" (open-ended; advanced via notify_advance).
        self._phases = [
            ("hold_home", None, None, 2.0),
            ("hold", QPOS_A, "A", None),
            ("xfer", QPOS_B, None, XFER_SECONDS),  # interpolates from the PREVIOUS hold's qpos to this
            ("hold", QPOS_B, "B", None),
        ]
        self._idx = 0
        self._phase_start_t = 0.0

    def _current(self):
        return self._phases[self._idx]

    def step(self, t: float) -> tuple[Optional[np.ndarray], Optional[str]]:
        """Call once per frame with the current sim time. Returns
        (qpos_target_or_None, pose_label_or_None). Auto-advances past
        "hold_home"/"xfer" once their fixed duration elapses; "hold" phases
        stay until notify_advance() is called."""
        kind, qpos, label, duration = self._current()
        if duration is not None and (t - self._phase_start_t) >= duration:
            self._advance(t)
            kind, qpos, label, duration = self._current()

        if kind == "hold_home":
            return None, None
        if kind == "hold":
            return qpos, label
        # "xfer": interpolate from the previous phase's qpos to this phase's qpos
        prev_qpos = self._phases[self._idx - 1][1] if self._idx > 0 else qpos
        frac = np.clip((t - self._phase_start_t) / duration, 0.0, 1.0)
        return prev_qpos + (qpos - prev_qpos) * frac, None

    def notify_advance(self, t: float) -> None:
        """Call when the visibility trigger fires. No-op unless currently in
        a "hold" phase, and no-op if already at the last phase (matches
        target_at()'s t>=140 safety fallback -- holding the last pose
        forever once there is correct, not a bug)."""
        kind, *_ = self._current()
        if kind != "hold" or self._idx >= len(self._phases) - 1:
            return
        self._advance(t)

    def _advance(self, t: float) -> None:
        if self._idx < len(self._phases) - 1:
            self._idx += 1
            self._phase_start_t = t

    def current_hold_start(self) -> Optional[float]:
        kind, *_ = self._current()
        return self._phase_start_t if kind == "hold" else None

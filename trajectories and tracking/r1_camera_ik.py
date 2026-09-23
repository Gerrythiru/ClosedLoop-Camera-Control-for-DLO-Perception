"""World-frame pose choreography for r1's wrist camera (r1_d435i_rgb),
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


# Target world poses (pos, quat) -- see run_dlo_v3_freeze.py for the
# calling code. Verified via verify_r1_camera_reachability.py, which also
# determines the actual JOINT-SPACE waypoints used at runtime (QPOS_A..
# QPOS_C, populated below by that script's output -- see its usage
# instructions).
POSE_A = (np.array([-0.356, -0.406, 1.188]), xyaxes_to_quat((0.692, -0.722, -0.000, 0.412, 0.394, 0.822)))
# Rail-only shifts from the original target (same y/z, same orientation,
# same QPOS_B arm joint angles throughout) -- first -0.20m (to reduce
# gripper occlusion), then +0.10m back (net -0.10m from the original).
# Since only the rail translates (no rotation), the camera's orientation
# is unaffected and self-collision is unchanged (a property of the joint
# angles alone); only the resulting camera position moves, and only NEW
# env-collisions (table/box/r2/r3) are possible -- verified clean at each
# shifted position.
POSE_B = (np.array([0.271326, -0.503475, 1.226723]), xyaxes_to_quat((0.971, 0.238, -0.000, -0.163, 0.666, 0.728)))
# User's exact target (pos=(-0.445,-0.125,1.009), xyaxes=(0.229,-0.973,
# -0.000,0.202,0.048,0.978)) needed an orientation ~20deg from anything
# reachable at that position (position itself converges to ~0.02mm
# everywhere on the rail -- purely an orientation limit). Position kept
# exact; orientation adjusted ~20deg per user direction, reachable at
# rail_x=-0.65 (4.98mm/1.32deg).
POSE_C = (np.array([-0.44946707, -0.12353978, 1.0106514]),
          xyaxes_to_quat((0.284, -0.925, 0.255, 0.328, 0.343, 0.880)))
# Pose D removed -- the schedule now ends holding Pose C.

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


# v4 only: r1's mount rides an actuated horizontal rail (r1_rail_x, see
# generate_triple_scene_v4.py), so some poses may need the mount displaced
# from its rest position (rail_x=0) to be reachable/collision-free by the
# 6-DOF arm alone -- found via trajectories/verify_r1_camera_reachability_v4.py's
# rail-aware search, which sweeps rail_x and keeps whichever position the
# 6-DOF IK actually solves cleanly at. See that script's own printed output
# for each pose's current rail_x (re-run after changing any POSE_* target).
RAIL_A = -0.45
RAIL_B = 0.10
RAIL_C = -0.65


def _rail_schedule():
    """Mirrors _schedule()'s exact time windows, but for the rail's scalar
    ctrl target instead of the arm's 6-DOF qpos. The 0-2s window holds at
    rail_x=0 explicitly (unlike _schedule()'s "hold_home", which leaves the
    ARM undriven by this module -- the rail has no equivalent pre-existing
    static ctrl to fall back on, so it must be actively held)."""
    return [
        (0.0, 2.0, "hold", 0.0),
        (2.0, 26.0, "hold", RAIL_A),
        (26.0, 28.0, "xfer", RAIL_A, RAIL_B),
        (28.0, 35.0, "hold", RAIL_B),
        (35.0, 36.0, "xfer", RAIL_B, RAIL_C),
        (36.0, 140.0, "hold", RAIL_C),
    ]


def rail_at(t: float) -> float:
    """Returns the rail's ctrl target (world_x displacement from rest) for
    time t, following the same hold/xfer windows as target_at(). Valid for
    the whole timeline from t=0 (holds rest until the 2s mark, then follows
    each pose's own rail position)."""
    for start, end, kind, *waypoints in _rail_schedule():
        if start <= t < end:
            if kind == "hold":
                return waypoints[0]
            r0, r1 = waypoints
            frac = np.clip((t - start) / (end - start), 0.0, 1.0)
            return r0 + (r1 - r0) * frac
    return RAIL_C  # t >= 140 safety fallback


# (start_s, end_s, kind, *waypoints) -- kind "hold_home" leaves r1 undriven
# by this module (stays on its pre-existing static R1_HOME_QPOS ctrl);
# "hold" freezes at the single given qpos (plus a pose label "A"/"B"/"C",
# used by hold_window_at/hold_windows below -- target_at() only ever reads
# waypoints[0] for holds, so the extra label is backward compatible);
# "xfer" linearly interpolates joint-space between two waypoints over the
# window.
#
# v4: Pose D removed -- Pose C now holds through the end of the run.
# Pose A holds until t=26.0s; Pose B arrives by t=28.0s (2s xfer) and
# holds until t=35.0s; Pose C arrives by t=36.0s (1s xfer) and holds until
# t=140.0s. All user-specified.
def _schedule():
    return [
        (0.0, 2.0, "hold_home"),
        (2.0, 26.0, "hold", QPOS_A, "A"),
        (26.0, 28.0, "xfer", QPOS_A, QPOS_B),
        (28.0, 35.0, "hold", QPOS_B, "B"),
        (35.0, 36.0, "xfer", QPOS_B, QPOS_C),
        (36.0, 140.0, "hold", QPOS_C, "C"),
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
    return QPOS_C  # t >= 140 safety fallback


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


# 2026-09-22: fixed transit durations reused by NBVCameraScheduler below --
# match _schedule()'s (26,28)/(35,36) "xfer" row widths. Mirrors
# r1_camera_ik_v5.py's NBVCameraScheduler, extended to V4's 3-pose A->B->C
# schedule and moving rail (V5's rail is held constant, so its version
# doesn't need to step a rail target at all).
XFER_AB_SECONDS = 2.0
XFER_BC_SECONDS = 1.0


class NBVCameraScheduler:
    """Same pose list/order/transit as _schedule()+_rail_schedule() (hold_home
    -> hold A -> xfer A->B -> hold B -> xfer B->C -> hold C), but each "hold"
    phase's END is driven by an external advance signal (notify_advance)
    instead of a fixed t. "hold_home"/"xfer" phases keep their existing
    fixed durations -- only "hold" phases are open-ended. Also steps the
    rail's ctrl target in lockstep with the camera qpos (unlike V5, V4's
    rail moves per-pose, so it must stay synchronized with whichever phase
    is active rather than being read independently off a fixed clock). Used
    only under run_dlo_v4_freeze.py's --nbv flag; _schedule()/target_at()/
    _rail_schedule()/rail_at()/hold_window_at()/hold_windows() above are
    untouched and still drive the default/--ros-live paths."""

    def __init__(self) -> None:
        if QPOS_A is None:
            raise RuntimeError("call set_solved_waypoints() before constructing NBVCameraScheduler")
        # (kind, qpos_target_or_None, rail_target, pose_label_or_None, duration_or_None) --
        # duration is None for "hold" (open-ended; advanced via notify_advance).
        self._phases = [
            ("hold_home", None, 0.0, None, 2.0),
            ("hold", QPOS_A, RAIL_A, "A", None),
            ("xfer", QPOS_B, RAIL_B, None, XFER_AB_SECONDS),  # interpolates from the PREVIOUS hold
            ("hold", QPOS_B, RAIL_B, "B", None),
            ("xfer", QPOS_C, RAIL_C, None, XFER_BC_SECONDS),
            ("hold", QPOS_C, RAIL_C, "C", None),
        ]
        self._idx = 0
        self._phase_start_t = 0.0

    def _current(self):
        return self._phases[self._idx]

    def step(self, t: float) -> tuple[Optional[np.ndarray], float, Optional[str]]:
        """Call once per frame with the current sim time. Returns
        (qpos_target_or_None, rail_target, pose_label_or_None). Auto-advances
        past "hold_home"/"xfer" once their fixed duration elapses; "hold"
        phases stay until notify_advance() is called."""
        kind, qpos, rail, label, duration = self._current()
        if duration is not None and (t - self._phase_start_t) >= duration:
            self._advance(t)
            kind, qpos, rail, label, duration = self._current()

        if kind == "hold_home":
            return None, rail, None
        if kind == "hold":
            return qpos, rail, label
        # "xfer": interpolate from the previous phase's qpos/rail to this phase's
        prev_qpos = self._phases[self._idx - 1][1] if self._idx > 0 else qpos
        prev_rail = self._phases[self._idx - 1][2] if self._idx > 0 else rail
        frac = np.clip((t - self._phase_start_t) / duration, 0.0, 1.0)
        interp_qpos = prev_qpos + (qpos - prev_qpos) * frac
        interp_rail = prev_rail + (rail - prev_rail) * frac
        return interp_qpos, interp_rail, None

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

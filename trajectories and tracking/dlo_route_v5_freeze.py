"""Dual-arm cable dangle planner for the V5 DLO scene.

r2 and r3 grasp the two ends of the red cable simultaneously and carry it
through the routing box, while r1 (driven separately by
run_dlo_v5_freeze.py's own per-frame callback) runs its own 4-pose camera
choreography concurrently -- unlike V3, this sequence is short enough
(nominal ~19s to both grasps, full run 140s) to genuinely overlap r1's
schedule instead of only starting after it.

r2: settle -> rotate base +120 deg -> rise to clear the routing box's guide
rails -> rotate base back to home heading (avoids a self-collision/box-wall
stall reaching down from the rotated heading) -> approach red_cable_00's (the
leftmost end) actual settled position with zero offset -> grasp it with a
real weld constraint engaged the instant the gripper fingers contact the
cable -> pick up to z=0.825 (inside the box guide rails' own height band) ->
carry through waypoints inside the box (translate + hold each); V5 inserts
4 extra waypoints that alternate +-0.045 m in Y so the carry into WP8 and
WP9 zigzags laterally across the channel.

r3 mirrors r2's phases 1-7 (settle/rotate/clear/un-rotate/approach/grasp/
pickup), targeting the cable's *last* node instead, then just translates to
one fixed hold position and stays there for the remainder of the run.

Both arms' final hold durations stretch to fill out TOTAL_SECONDS (matching
r1's own schedule total in r1_camera_ik.py), so all three arms' sequences
end together regardless of small timing variance in the contact-triggered
grasp phase.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Callable, Optional

import mujoco
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "mujoco_scenes"))
from generate_triple_scene_v5 import (  # noqa: E402
    CABLE_SEGMENTS, ROUTING_BOX_GUIDE_TOP_Z, _GRASP_TOOL_RAW_EXTENT_M,
)

# 2026-09-18: corrected via live visual verification (--view's "B"/"F"
# marker toggles in run_dlo_v5_freeze.py) -- this is NOT the enclosed
# circular bore's center (that assumption was wrong; the enclosed bore
# was never the actual capture feature). The real grasp point is the
# NOTCH -- a slot cut into the tool, open on both of its Y edges (raw
# mesh Y = 0 and Y = extent[1]), that the cable is captured in by
# landing on it from vertically above, not by axial threading through a
# hole. The notch's midpoint is at raw-mesh Y=0 (confirmed directly
# against the "F"-toggle's per-face markers: this is the y_min face
# center), centered in X and Z like the old (wrong) bore-center
# assumption. Local frame, tool BODY's own frame (the mesh has no
# additional pos/quat offset inside that body, so this is directly
# usable as an xmat-rotatable offset from the tool body's xpos).
_ex, _ey, _ez = _GRASP_TOOL_RAW_EXTENT_M
_GRASP_TOOL_RAW_CENTER_M = np.array([_ex / 2.0, 0.0, _ez / 2.0])

CABLE_PREFIX = "red"
FRAME_DT = 1.0 / 30.0

# Matches r1's total camera-choreography schedule length (r1_camera_ik.py's
# SCHEDULE ends at t=140.0) -- both arms' final holds stretch to end exactly
# here so all three robots' sequences finish together.
TOTAL_SECONDS = 140.0

SETTLE_SECONDS = 3.0         # free-settle before r2/r3 grasp (v2's original default)
ROTATE_SECONDS = 2.0         # base joint1 rotates off its settled home angle
BASE_ROTATE_RAD = math.radians(120.0)
CLEAR_SECONDS = 2.0          # vertical rise to clear the box guide rails
CLEARANCE_MARGIN = 0.10      # m -- clear the tallest box geometry by this much
CABLE_SETTLE_VEL_MPS = 0.05         # pre-approach settle gate (see run_sequence
                                     # phase 5) -- the free-hanging cable end is still
                                     # measurably moving (~0.22 m/s observed) at the
                                     # original fixed grasp timing
CABLE_SETTLE_MAX_WAIT_SECONDS = 2.0  # bounded extra wait -- never stalls the schedule
                                      # indefinitely if the cable never fully quiesces

# 2026-09-18: grasp redesigned around the NOTCH (see _GRASP_TOOL_RAW_CENTER_M's
# comment) -- the cable is captured by the tool's notch landing on it from
# vertically above, weld engaging the instant contact happens (confirmed by
# the user; no convergence gate needed the way the old, incorrect enclosed-
# bore assumption required). Kept deliberately SIMPLE, mirroring this
# file's original pre-tool approach style (_run_multi/_ik_step, blind,
# position-only, no orientation objective at all -- an intermediate
# attempt added 2-DOF tangent-axis alignment during the approach and the
# user rejected it as unnecessary/awkward-looking movement, see git
# history) -- just retargeted at the tool's grasp point instead of the
# raw cable position, and split into two stages so the tool's solid body
# doesn't sweep sideways through the cable the way a single direct blind
# approach did (rammed it at full speed, also see git history): (1) a
# single blind pass to a point HOVER_CLEARANCE_M above the live grasp
# target; (2) a straight-down-only, PACED descent (z_offset interpolated
# HOVER_CLEARANCE_M -> 0 over DESCENT_SECONDS, same target-pacing idea as
# _run_multi_interp) onto the cable, contact-triggering the weld.
HOVER_CLEARANCE_M = 0.05        # height above the live cable position to
                                 # approach to before descending
HOVER_APPROACH_SECONDS = 10.0   # bound on stage 1 (hover-approach)
DESCENT_SECONDS = 4.0           # paced descent duration -- HOVER_CLEARANCE_M
                                 # / DESCENT_SECONDS = ~12.5mm/s, a gentle
                                 # landing speed
DESCENT_SETTLE_SECONDS = 2.0    # bounded extra window if the paced descent
                                 # finishes without contact (shouldn't
                                 # normally happen -- the descent target
                                 # lands exactly on the live cable position)
PICKUP_SECONDS = 5.0         # vertical pick-up after grasp (phase 7)
PICKUP_Z = 0.825             # pick-up target height (inside the box guides' own height band)

# r2's post-pickup waypoint carry. V5: X rescaled about the routing-box
# center (x=0.20) by new_x = 0.20 + 0.75*(old_x - 0.20) to track the 75%-
# shortened box's guide gaps; Y/Z unchanged. Originals (dlo_route_v2_Freeze.py):
# WP8 (-0.01,...), WP9 (0.15,...), WP10 (0.27,...), WP11 (0.10,...), R3_HOLD (0.45,...).
WAYPOINT_8 = (0.0425, 0.035, 0.825)
WAYPOINT_9 = (0.1625, 0.0, 0.94)
WAYPOINT_10 = (0.2525, -0.035, 0.825)
WAYPOINT_11 = (0.125, 0.0, 0.94)
WAYPOINT_TRANSLATE_SECONDS = 5.0   # each translate leg
WAYPOINT_HOLD_SECONDS = 3.0        # hold after waypoints 8, 9, 10

# V5: two extra waypoints on the leg into WP8 and two on the leg into WP9,
# each a full translate + hold, alternating +-0.045 m in Y to make r2's
# carry zigzag laterally across the box channel (inner half-width 0.095 m,
# so ~0.05 m clearance -- channel width is unscaled). X re-placed for the
# 75% box (roughly 1/3 and 2/3 along the lifted->WP8 and WP8->WP9 legs).
# Z: flat 0.825 through the WP8 approach, monotone rise 0.825 -> 0.94
# through the WP9 approach (no Z zigzag). Resulting Y sequence
# lifted->..->WP9: 0, +.045, -.045, +.035, -.045, +.045, 0.
WAYPOINT_8A = (-0.053, 0.045, 0.825)
WAYPOINT_8B = (-0.007, -0.045, 0.825)
WAYPOINT_9A = (0.083, -0.045, 0.865)
WAYPOINT_9B = (0.123, 0.045, 0.905)

# r3's single post-pickup hold position. V5: X rescaled as above.
R3_HOLD_POS = (0.3875, 0.035, 0.825)
R3_TRANSLATE_SECONDS = 5.0         # matches r2's per-leg translate pace

ARM_NAMES = ("r2", "r3")

OnFrame = Callable[[mujoco.MjModel, mujoco.MjData, float], None]


class DanglePlanner:
    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        self.model = model
        self.data = data
        mujoco.mj_forward(model, data)

        self._arms = {name: self._build_arm(name) for name in ARM_NAMES}
        self.cable_body_ids = tuple(
            self._body_id(f"{CABLE_PREFIX}_cable_{i:02d}") for i in range(CABLE_SEGMENTS)
        )
        self.cable_geom_ids = tuple(
            self._geom_id(f"{CABLE_PREFIX}_cable_geom_{i:02d}") for i in range(CABLE_SEGMENTS)
        )
        self._ctrl_min = model.actuator_ctrlrange[:, 0].copy()
        self._ctrl_max = model.actuator_ctrlrange[:, 1].copy()
        self._ctrl_limited = model.actuator_ctrllimited.copy()

        self._weld_ids = {
            "r2": self._equality_id("r2_cable_grasp_weld"),
            "r3": self._equality_id("r3_cable_grasp_weld"),
        }
        self._gripper_finger_geom_ids = {
            name: (self._geom_id(f"{name}_gripper_left_finger"), self._geom_id(f"{name}_gripper_right_finger"))
            for name in ARM_NAMES
        }
        for name in ARM_NAMES:
            # 2026-09-17: closed from t=0 (was False/open) so the grasp
            # tool (mounted rigidly in the closed-jaw position, see
            # add_grasp_tools in generate_triple_scene_v5.py) is clamped
            # from scene load onward -- every OTHER _set_gripper call in
            # this file already passes True; this was the only place that
            # ever opened it.
            self._set_gripper(name, True)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _joint_id(self, name: str) -> int:
        jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            raise ValueError(f"Missing joint {name}")
        return jid

    def _body_id(self, name: str) -> int:
        bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
        if bid < 0:
            raise ValueError(f"Missing body {name}")
        return bid

    def _site_id(self, name: str) -> int:
        sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, name)
        if sid < 0:
            raise ValueError(f"Missing site {name}")
        return sid

    def _geom_id(self, name: str) -> int:
        gid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name)
        if gid < 0:
            raise ValueError(f"Missing geom {name}")
        return gid

    def _equality_id(self, name: str) -> int:
        eid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_EQUALITY, name)
        if eid < 0:
            raise ValueError(f"Missing equality constraint {name}")
        return eid

    def _actuator_for_joint(self, jid: int) -> int:
        for aid in range(self.model.nu):
            if int(self.model.actuator_trnid[aid, 0]) == jid:
                return aid
        raise ValueError(f"No actuator for joint id {jid}")

    def _build_arm(self, prefix: str) -> dict:
        joint_ids = tuple(self._joint_id(f"{prefix}_joint{i}") for i in range(1, 7))
        return {
            "qpos_ids": tuple(int(self.model.jnt_qposadr[j]) for j in joint_ids),
            "act_ids": tuple(self._actuator_for_joint(j) for j in joint_ids),
            "dof_ids": tuple(int(self.model.jnt_dofadr[j]) for j in joint_ids),
            "site_id": self._site_id(f"{prefix}_end_effector"),
            "gripper_id": mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{prefix}_gripper"
            ),
            "gripper_body_id": self._body_id(f"{prefix}_gripper_body"),
            "tool_geom_id": self._geom_id(f"{prefix}_grasp_tool_geom"),
        }

    def _set_ctrl(self, act_id: int, value: float) -> None:
        if self._ctrl_limited[act_id]:
            value = float(np.clip(value, self._ctrl_min[act_id], self._ctrl_max[act_id]))
        self.data.ctrl[act_id] = value

    def _set_gripper(self, name: str, closed: bool) -> None:
        self.data.ctrl[self._arms[name]["gripper_id"]] = 8.0 if closed else -4.0

    def _ik_step(self, name: str, target: np.ndarray) -> None:
        arm = self._arms[name]
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        mujoco.mj_jacSite(self.model, self.data, jacp, jacr, arm["site_id"])
        jac = jacp[:, list(arm["dof_ids"])]
        error = target - self.data.site_xpos[arm["site_id"]]
        dq = np.clip(jac.T @ np.linalg.solve(jac @ jac.T + 0.0025 * np.eye(3), error), -0.12, 0.12)
        for li, aid in enumerate(arm["act_ids"]):
            self._set_ctrl(aid, self.data.qpos[arm["qpos_ids"][li]] + dq[li])

    def _gripper_touching(self, name: str, geom_id: int) -> bool:
        finger_ids = self._gripper_finger_geom_ids[name]
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            if c.geom1 == geom_id and c.geom2 in finger_ids:
                return True
            if c.geom2 == geom_id and c.geom1 in finger_ids:
                return True
        return False

    def _tool_touching(self, name: str, geom_id: int) -> bool:
        """True if geom_id (the cable) is in contact with arm `name`'s
        grasp tool geom itself -- NOT the fingers (see _gripper_touching).
        2026-09-18: the grasp point was corrected from the tool's enclosed
        circular bore to its NOTCH (open on both edges, see
        _GRASP_TOOL_RAW_CENTER_M's comment) -- the cable is captured by
        the notch's surface contacting it directly as the tool descends
        onto it from above, not by the fingers closing on it, so contact
        detection has to watch the tool geom, not the fingers."""
        tool_gid = self._arms[name]["tool_geom_id"]
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            if c.geom1 == geom_id and c.geom2 == tool_gid:
                return True
            if c.geom2 == geom_id and c.geom1 == tool_gid:
                return True
        return False

    def _tool_pose(self, name: str) -> tuple[np.ndarray, np.ndarray]:
        """(tool_xpos, tool_xmat(3,3)) for arm `name`'s grasp tool body --
        shared fetch, used by _grasp_aligned_target/_engage_weld."""
        tool_bid = self._body_id(f"{name}_grasp_tool")
        return self.data.xpos[tool_bid], self.data.xmat[tool_bid].reshape(3, 3)

    def _grasp_aligned_target(self, name: str, cable_body_id: int, z_offset: float = 0.0) -> np.ndarray:
        """IK target for arm `name`'s end_effector SITE such that the
        grasp point (the notch midpoint, _GRASP_TOOL_RAW_CENTER_M -- NOT
        the site itself) lands on cable_body_id's CURRENT actual position
        (offset +z_offset in world Z, e.g. for hovering above it before
        descending) -- recomputed fresh on every call, not a one-time
        snapshot (a stale snapshot was a previously-diagnosed and fixed
        bug this session: the free-hanging cable drifts under gravity
        during the multi-second approach window, ~21.6mm measured once).

        2026-09-18: renamed from _bore_aligned_target/generalized with
        z_offset when the grasp point was corrected from the tool's
        enclosed bore to its notch (see _GRASP_TOOL_RAW_CENTER_M) -- the
        underlying site-offset math (site + (target - grasp_point)) is
        unchanged."""
        site = self.data.site_xpos[self._arms[name]["site_id"]]
        tool_xpos, tool_xmat = self._tool_pose(name)
        grasp_point = tool_xpos + tool_xmat @ _GRASP_TOOL_RAW_CENTER_M
        target = self.data.xpos[cable_body_id].copy()
        target[2] += z_offset
        return target - (grasp_point - site)

    def _engage_weld(self, name: str, cable_body_id: int) -> None:
        """Lock cable_body_id to arm `name`'s gripper via its pre-defined
        weld, using the bodies' *current* relative pose for ORIENTATION
        (so the constraint starts at zero orientation error -- no snap on
        activation) but the grasp TOOL's known, deterministic bore center
        for POSITION, instead of wherever the cable's own body happened to
        be at the instant contact was detected.

        2026-09-17: previously used the cable body's own actual (incidental
        contact-time) position for both position and orientation -- this is
        the root cause of the "levitating" cable-offset artifact (node0/
        node14 sitting near, but not exactly at, the gripper's grasp point,
        not reconstructible from any current robot state, see Today_SUMM.md
        Section 12 item B9). The tool is now rigidly mounted with a KNOWN
        bore center (_GRASP_TOOL_RAW_CENTER_M, in the tool body's own local
        frame); using that instead of the cable's actual position makes the
        grasp point exactly the tool's circular cut, every time, regardless
        of exactly where/how contact occurred. Orientation is intentionally
        left as-is (captured from the actual contact-time relative pose,
        unchanged) -- only the POSITION target changed.

        The weld's body1=cable_body_id, body2=gripper. MuJoCo's weld
        `relpose` encodes body2's pose expressed in body1's frame (verified
        empirically -- NOT body1 in body2's frame, which is the more
        "intuitive" reading and produces a double-magnitude position error
        that snaps violently on activation)."""
        weld_id = self._weld_ids[name]
        gripper_bid = self._arms[name]["gripper_body_id"]
        tool_xpos, tool_xmat = self._tool_pose(name)
        bore_center_world = tool_xpos + tool_xmat @ _GRASP_TOOL_RAW_CENTER_M

        xmat_c = self.data.xmat[cable_body_id].reshape(3, 3)
        rel_pos = xmat_c.T @ (self.data.xpos[gripper_bid] - bore_center_world)

        inv_quat_c = np.zeros(4)
        mujoco.mju_negQuat(inv_quat_c, self.data.xquat[cable_body_id])
        rel_quat = np.zeros(4)
        mujoco.mju_mulQuat(rel_quat, inv_quat_c, self.data.xquat[gripper_bid])

        self.model.eq_data[weld_id, 0:3] = 0.0
        self.model.eq_data[weld_id, 3:6] = rel_pos
        self.model.eq_data[weld_id, 6:10] = rel_quat
        self.model.eq_data[weld_id, 10] = 1.0
        self.data.eq_active[weld_id] = True

    def _run_multi(
        self,
        seconds: float,
        t: float,
        on_frame: Optional[OnFrame],
        *,
        ee_targets: Optional[dict[str, np.ndarray]] = None,
        joint1_targets: Optional[dict[str, float]] = None,
        gripper_closed: Optional[dict[str, bool]] = None,
    ) -> float:
        ee_targets = ee_targets or {}
        joint1_targets = joint1_targets or {}
        gripper_closed = gripper_closed or {}
        steps_per_frame = max(1, round(FRAME_DT / self.model.opt.timestep))
        n_steps = max(1, round(seconds / self.model.opt.timestep))
        for step in range(n_steps):
            self.data.xfrc_applied[:, :] = 0.0
            for name in ARM_NAMES:
                if name in ee_targets:
                    self._ik_step(name, ee_targets[name])
                if name in joint1_targets:
                    self._set_ctrl(self._arms[name]["act_ids"][0], joint1_targets[name])
                if name in gripper_closed:
                    self._set_gripper(name, gripper_closed[name])
            mujoco.mj_step(self.model, self.data)
            t += self.model.opt.timestep
            if on_frame is not None and step % steps_per_frame == 0:
                on_frame(self.model, self.data, t)
        return t

    def _run_multi_interp(
        self,
        seconds: float,
        t: float,
        on_frame: Optional[OnFrame],
        *,
        starts: dict[str, np.ndarray],
        ends: dict[str, np.ndarray],
        joint1_targets: Optional[dict[str, float]] = None,
        gripper_closed: Optional[dict[str, bool]] = None,
    ) -> float:
        """Like _run_multi, but each arm's IK target is linearly interpolated
        from starts[name] to ends[name] over the full duration (paces the
        setpoint itself so the motion visibly takes the requested duration,
        instead of converging early and idling). An arm with a joint1_target
        entry has its base joint driven directly instead of via the IK
        Jacobian (matching _run_multi's convention for locking joint1 during
        a base-rotate phase)."""
        joint1_targets = joint1_targets or {}
        gripper_closed = gripper_closed or {}
        steps_per_frame = max(1, round(FRAME_DT / self.model.opt.timestep))
        n_steps = max(1, round(seconds / self.model.opt.timestep))
        for step in range(n_steps):
            self.data.xfrc_applied[:, :] = 0.0
            frac = (step + 1) / n_steps
            for name in starts:
                self._ik_step(name, starts[name] + (ends[name] - starts[name]) * frac)
            for name, target in joint1_targets.items():
                self._set_ctrl(self._arms[name]["act_ids"][0], target)
            for name in gripper_closed:
                self._set_gripper(name, gripper_closed[name])
            mujoco.mj_step(self.model, self.data)
            t += self.model.opt.timestep
            if on_frame is not None and step % steps_per_frame == 0:
                on_frame(self.model, self.data, t)
        return t

    # ------------------------------------------------------------------
    # Main sequence
    # ------------------------------------------------------------------

    def run_sequence(self, on_frame: Optional[OnFrame] = None) -> None:
        t = 0.0
        r2_cable_bid = self.cable_body_ids[0]           # red_cable_00 = leftmost end
        r3_cable_bid = self.cable_body_ids[-1]           # red_cable_{N-1} = rightmost end

        print(f"[dlo-v5] phase 1/8: settling cable ({SETTLE_SECONDS:.0f} s)...", flush=True)
        t = self._run_multi(SETTLE_SECONDS, t, on_frame)

        # r2 clears the box via its verified rotate(+120deg)/rise/un-rotate
        # dance (dodges a measured self-collision + box-wall stall on a
        # direct approach). r3 sits on the opposite side of the box and
        # that same maneuver stalls it against the box guide rails and its
        # own base (verified empirically) -- r3 instead rises straight up
        # (no base rotation), translates laterally at clearance height to
        # sit directly over its grasp target, then holds -- an equally
        # obstacle-avoiding approach for r3's own geometry, still split
        # across the same 3 phases/durations so both arms stay in lockstep.
        j1_home = {name: self.data.qpos[self._arms[name]["qpos_ids"][0]] for name in ARM_NAMES}
        j1_rotated_r2 = j1_home["r2"] + BASE_ROTATE_RAD
        clearance_z = ROUTING_BOX_GUIDE_TOP_Z + CLEARANCE_MARGIN
        r3_up = self.data.site_xpos[self._arms["r3"]["site_id"]].copy()
        r3_up[2] = clearance_z

        print(f"[dlo-v5] phase 2/8: r2 rotating base {math.degrees(BASE_ROTATE_RAD):.0f} deg, "
              f"r3 rising to clear the box (z={clearance_z:.3f} m)...", flush=True)
        t = self._run_multi(
            ROTATE_SECONDS, t, on_frame,
            ee_targets={"r3": r3_up}, joint1_targets={"r2": j1_rotated_r2},
        )

        r2_clear = self.data.site_xpos[self._arms["r2"]["site_id"]].copy()
        r2_clear[2] = clearance_z
        r3_over = self.data.xpos[r3_cable_bid].copy()
        r3_over[2] = clearance_z
        r3_ee_now = self.data.site_xpos[self._arms["r3"]["site_id"]].copy()
        print("[dlo-v5] phase 3/8: r2 holding clear of the box, "
              "r3 translating laterally to over its grasp target...", flush=True)
        t = self._run_multi_interp(
            CLEAR_SECONDS, t, on_frame,
            starts={"r2": r2_clear, "r3": r3_ee_now}, ends={"r2": r2_clear, "r3": r3_over},
            joint1_targets={"r2": j1_rotated_r2},
        )

        print("[dlo-v5] phase 4/8: r2 rotating base back to home heading, r3 holding...", flush=True)
        t = self._run_multi(
            ROTATE_SECONDS, t, on_frame,
            ee_targets={"r3": r3_over}, joint1_targets={"r2": j1_home["r2"]},
        )

        target_bids = {"r2": r2_cable_bid, "r3": r3_cable_bid}
        steps_per_frame = max(1, round(FRAME_DT / self.model.opt.timestep))

        # Bounded pre-approach settle gate: the free-hanging cable end is
        # still measurably moving (~0.22 m/s observed) at the original
        # fixed grasp timing -- not a one-time transient a longer FIXED
        # wait reliably clears (looks like lightly-damped oscillation), so
        # this checks actual current velocity instead of guessing a wait
        # time, bounded so it can't stall the schedule if the cable never
        # fully quiesces.
        settle_wait_steps = max(1, round(CABLE_SETTLE_MAX_WAIT_SECONDS / self.model.opt.timestep))
        for step in range(settle_wait_steps):
            speeds = [float(np.linalg.norm(self.data.cvel[target_bids[name]][3:6])) for name in ARM_NAMES]
            if all(s < CABLE_SETTLE_VEL_MPS for s in speeds):
                break
            self.data.xfrc_applied[:, :] = 0.0
            mujoco.mj_step(self.model, self.data)
            t += self.model.opt.timestep
            if on_frame is not None and step % steps_per_frame == 0:
                on_frame(self.model, self.data, t)
        else:
            print(f"[dlo-v5] cable still above {CABLE_SETTLE_VEL_MPS} m/s after "
                  f"{CABLE_SETTLE_MAX_WAIT_SECONDS:.1f}s settle wait -- proceeding anyway", flush=True)

        # Phase 5: simple, blind, position-only approach to a point
        # HOVER_CLEARANCE_M above the (live) grasp target -- same style as
        # this file's original pre-tool approach (_run_multi/_ik_step,
        # no orientation objective at all), just aimed at a point above
        # the target instead of directly at it, so the tool's solid body
        # never sweeps sideways through the cable on the way in (that
        # direct-approach version rammed the tool into the cable at full
        # speed, see git history). A single blind pass is fine here --
        # there's nothing to collide with 50mm above the cable.
        print("[dlo-v5] phase 5/8: r2+r3 approaching a hover point above the grasp target...", flush=True)
        hover_targets = {name: self._grasp_aligned_target(name, target_bids[name], z_offset=HOVER_CLEARANCE_M)
                          for name in ARM_NAMES}
        t = self._run_multi(HOVER_APPROACH_SECONDS, t, on_frame, ee_targets=hover_targets)

        # Phase 6: simple, straight-down (vertical only) descent onto the
        # live grasp target -- z_offset paced from HOVER_CLEARANCE_M to 0
        # over DESCENT_SECONDS (same target-pacing idea as
        # _run_multi_interp), X/Y held at the target throughout. Weld
        # engages the INSTANT the tool's notch contacts the cable
        # (_tool_touching) -- contact alone is the trigger, per the
        # user's explicit design, no convergence gate.
        print("[dlo-v5] phase 6/8: r2+r3 descending onto the grasp target (weld engages on contact)...", flush=True)
        target_geoms = {"r2": self.cable_geom_ids[0], "r3": self.cable_geom_ids[-1]}
        n_steps = max(1, round(DESCENT_SECONDS / self.model.opt.timestep))
        engaged = {"r2": False, "r3": False}
        for step in range(n_steps):
            self.data.xfrc_applied[:, :] = 0.0
            frac = (step + 1) / n_steps
            for name in ARM_NAMES:
                if engaged[name]:
                    continue
                z_offset = HOVER_CLEARANCE_M * (1.0 - frac)
                target_pos = self._grasp_aligned_target(name, target_bids[name], z_offset=z_offset)
                self._ik_step(name, target_pos)
                self._set_gripper(name, True)
            mujoco.mj_step(self.model, self.data)
            t += self.model.opt.timestep
            if on_frame is not None and step % steps_per_frame == 0:
                on_frame(self.model, self.data, t)
            for name in ARM_NAMES:
                if not engaged[name] and self._tool_touching(name, target_geoms[name]):
                    print(f"[dlo-v5] {name}: contact at t={t:.2f}s -- engaging weld", flush=True)
                    self._engage_weld(name, target_bids[name])
                    engaged[name] = True
            if all(engaged.values()):
                break

        # Bounded extra window if the descent finished without contact
        # (shouldn't normally happen -- the descent target lands exactly
        # on the live cable position -- bounded so a miss can't stall
        # the schedule).
        if not all(engaged.values()):
            print(f"[dlo-v5] extra {DESCENT_SETTLE_SECONDS:.1f}s window for any remaining contact...", flush=True)
            n_steps = max(1, round(DESCENT_SETTLE_SECONDS / self.model.opt.timestep))
            for step in range(n_steps):
                self.data.xfrc_applied[:, :] = 0.0
                for name in ARM_NAMES:
                    if engaged[name]:
                        continue
                    target_pos = self._grasp_aligned_target(name, target_bids[name], z_offset=0.0)
                    self._ik_step(name, target_pos)
                mujoco.mj_step(self.model, self.data)
                t += self.model.opt.timestep
                if on_frame is not None and step % steps_per_frame == 0:
                    on_frame(self.model, self.data, t)
                for name in ARM_NAMES:
                    if not engaged[name] and self._tool_touching(name, target_geoms[name]):
                        print(f"[dlo-v5] {name}: contact at t={t:.2f}s -- engaging weld", flush=True)
                        self._engage_weld(name, target_bids[name])
                        engaged[name] = True
                if all(engaged.values()):
                    break
        for name in ARM_NAMES:
            if not engaged[name]:
                print(f"[dlo-v5] {name}: no contact detected -- engaging weld anyway", flush=True)
                self._engage_weld(name, target_bids[name])
        fine_targets = {name: self._grasp_aligned_target(name, target_bids[name]) for name in ARM_NAMES}

        lifted_pos = {name: np.array([fine_targets[name][0], fine_targets[name][1], PICKUP_Z]) for name in ARM_NAMES}

        print(f"[dlo-v5] phase 7/8: picking up to z={PICKUP_Z:.3f} m ({PICKUP_SECONDS:.0f} s)...", flush=True)
        t = self._run_multi_interp(
            PICKUP_SECONDS, t, on_frame,
            starts=fine_targets, ends=lifted_pos,
            gripper_closed={"r2": True, "r3": True},
        )

        # r3's phase 8: translate to its single fixed hold position, then
        # hold for whatever remains of TOTAL_SECONDS.
        r3_hold = np.array(R3_HOLD_POS)
        r3_remaining_after_translate = max(0.0, TOTAL_SECONDS - (t + R3_TRANSLATE_SECONDS))
        print(f"[dlo-v5] phase 8: r3 translating to {tuple(r3_hold.round(3))} "
              f"({R3_TRANSLATE_SECONDS:.0f}s) then holding for the rest of the cycle "
              f"(~{r3_remaining_after_translate:.0f}s)...", flush=True)
        # V5: r2 zigzags into WP8 (via WP8A/WP8B) then into WP9 (via
        # WP9A/WP9B), each a full translate+hold, alternating +-0.045 m in
        # Y. r3 still does its single translate-to-hold on the first leg;
        # every leg after that drives r2 only and r3 holds at r3_hold.
        wp8a, wp8b, wp8 = np.array(WAYPOINT_8A), np.array(WAYPOINT_8B), np.array(WAYPOINT_8)
        wp9a, wp9b, wp9 = np.array(WAYPOINT_9A), np.array(WAYPOINT_9B), np.array(WAYPOINT_9)

        print(f"[dlo-v5] r2 phase 8/15: zigzag -> WP8A ({WAYPOINT_TRANSLATE_SECONDS:.0f}s)...", flush=True)
        t = self._run_multi_interp(
            WAYPOINT_TRANSLATE_SECONDS, t, on_frame,
            starts={"r2": lifted_pos["r2"], "r3": lifted_pos["r3"]},
            ends={"r2": wp8a, "r3": r3_hold},
            gripper_closed={"r2": True, "r3": True},
        )
        t = self._run_multi(
            WAYPOINT_HOLD_SECONDS, t, on_frame,
            ee_targets={"r2": wp8a, "r3": r3_hold}, gripper_closed={"r2": True, "r3": True},
        )

        for lbl, name, src, dst in (
            ("9/15", "WP8B", wp8a, wp8b), ("10/15", "WP8", wp8b, wp8),
            ("11/15", "WP9A", wp8, wp9a), ("12/15", "WP9B", wp9a, wp9b), ("13/15", "WP9", wp9b, wp9),
        ):
            print(f"[dlo-v5] r2 phase {lbl}: zigzag -> {name} ({WAYPOINT_TRANSLATE_SECONDS:.0f}s)...", flush=True)
            t = self._run_multi_interp(
                WAYPOINT_TRANSLATE_SECONDS, t, on_frame,
                starts={"r2": src}, ends={"r2": dst},
                gripper_closed={"r2": True, "r3": True},
            )
            t = self._run_multi(
                WAYPOINT_HOLD_SECONDS, t, on_frame,
                ee_targets={"r2": dst, "r3": r3_hold}, gripper_closed={"r2": True, "r3": True},
            )

        wp10 = np.array(WAYPOINT_10)
        print(f"[dlo-v5] r2 phase 14/15: translating to waypoint 10 ({WAYPOINT_TRANSLATE_SECONDS:.0f}s)...", flush=True)
        t = self._run_multi_interp(
            WAYPOINT_TRANSLATE_SECONDS, t, on_frame,
            starts={"r2": wp9}, ends={"r2": wp10},
            gripper_closed={"r2": True, "r3": True},
        )
        t = self._run_multi(
            WAYPOINT_HOLD_SECONDS, t, on_frame,
            ee_targets={"r2": wp10, "r3": r3_hold},
            gripper_closed={"r2": True, "r3": True},
        )

        wp11 = np.array(WAYPOINT_11)
        print(f"[dlo-v5] r2 phase 15/15: translating to waypoint 11 ({WAYPOINT_TRANSLATE_SECONDS:.0f}s), "
              f"then both arms hold to t={TOTAL_SECONDS:.0f}s...", flush=True)
        t = self._run_multi_interp(
            WAYPOINT_TRANSLATE_SECONDS, t, on_frame,
            starts={"r2": wp10}, ends={"r2": wp11},
            gripper_closed={"r2": True, "r3": True},
        )

        final_hold_seconds = max(0.0, TOTAL_SECONDS - t)
        t = self._run_multi(
            final_hold_seconds, t, on_frame,
            ee_targets={"r2": wp11, "r3": r3_hold},
            gripper_closed={"r2": True, "r3": True},
        )

        print(f"[dlo-v5] sequence complete -- t={t:.1f} s", flush=True)

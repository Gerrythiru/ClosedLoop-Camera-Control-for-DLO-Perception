"""Dual-arm cable dangle planner for the V4 DLO scene.

r2 and r3 grasp the two ends of the red cable simultaneously and carry it
through the routing box, while r1 (driven separately by
run_dlo_v4_freeze.py's own per-frame callback) runs its own 4-pose camera
choreography concurrently -- unlike V3, this sequence is short enough
(nominal ~19s to both grasps, full run 140s) to genuinely overlap r1's
schedule instead of only starting after it.

r2: settle -> rotate base +120 deg -> rise to clear the routing box's guide
rails -> rotate base back to home heading (avoids a self-collision/box-wall
stall reaching down from the rotated heading) -> approach red_cable_00's (the
leftmost end) actual settled position with zero offset -> grasp it with a
real weld constraint engaged the instant the gripper fingers contact the
cable -> pick up to z=0.825 (inside the box guide rails' own height band) ->
carry through 4 more waypoints inside the box (translate + hold each).

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
from generate_triple_scene_v4 import (  # noqa: E402
    CABLE_SEGMENTS, ROUTING_BOX_GUIDE_TOP_Z,
)

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
APPROACH_SECONDS = 5.0       # IK convergence onto the exact grasp position
GRASP_TIMEOUT_SECONDS = 2.0  # gripper closes; weld engages on contact, or at this timeout as a fallback
PICKUP_SECONDS = 5.0         # vertical pick-up after grasp (phase 7)
PICKUP_Z = 0.825             # pick-up target height (inside the box guides' own height band)

# r2's post-pickup waypoint carry, unchanged from dlo_route_v2_Freeze.py.
WAYPOINT_8 = (-0.01, 0.035, 0.825)
WAYPOINT_9 = (0.15, 0.0, 0.94)
WAYPOINT_10 = (0.27, -0.035, 0.825)
WAYPOINT_11 = (0.10, 0.0, 0.94)
WAYPOINT_TRANSLATE_SECONDS = 5.0   # each translate leg
WAYPOINT_HOLD_SECONDS = 3.0        # hold after waypoints 8, 9, 10

# r3's single post-pickup hold position.
R3_HOLD_POS = (0.45, 0.035, 0.825)
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
            self._set_gripper(name, False)

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

    def _engage_weld(self, name: str, cable_body_id: int) -> None:
        """Lock cable_body_id to arm `name`'s gripper via its pre-defined
        weld, using the bodies' *current* relative pose so the constraint
        starts at zero error (no snap on activation).

        The weld's body1=cable_body_id, body2=gripper. MuJoCo's weld
        `relpose` encodes body2's pose expressed in body1's frame (verified
        empirically -- NOT body1 in body2's frame, which is the more
        "intuitive" reading and produces a double-magnitude position error
        that snaps violently on activation)."""
        weld_id = self._weld_ids[name]
        gripper_bid = self._arms[name]["gripper_body_id"]
        xmat_c = self.data.xmat[cable_body_id].reshape(3, 3)
        rel_pos = xmat_c.T @ (self.data.xpos[gripper_bid] - self.data.xpos[cable_body_id])

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

        print(f"[dlo-v4] phase 1/8: settling cable ({SETTLE_SECONDS:.0f} s)...", flush=True)
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

        print(f"[dlo-v4] phase 2/8: r2 rotating base {math.degrees(BASE_ROTATE_RAD):.0f} deg, "
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
        print("[dlo-v4] phase 3/8: r2 holding clear of the box, "
              "r3 translating laterally to over its grasp target...", flush=True)
        t = self._run_multi_interp(
            CLEAR_SECONDS, t, on_frame,
            starts={"r2": r2_clear, "r3": r3_ee_now}, ends={"r2": r2_clear, "r3": r3_over},
            joint1_targets={"r2": j1_rotated_r2},
        )

        print("[dlo-v4] phase 4/8: r2 rotating base back to home heading, r3 holding...", flush=True)
        t = self._run_multi(
            ROTATE_SECONDS, t, on_frame,
            ee_targets={"r3": r3_over}, joint1_targets={"r2": j1_home["r2"]},
        )

        grasp_pos = {"r2": self.data.xpos[r2_cable_bid].copy(), "r3": self.data.xpos[r3_cable_bid].copy()}
        print(f"[dlo-v4] r2 grasp target {grasp_pos['r2'].round(3)}, r3 grasp target {grasp_pos['r3'].round(3)}", flush=True)

        # Phase 5 approach, split in two: a first pass toward the raw cable
        # position, then a correction pass that compensates for each arm's
        # own site-to-gripper-fingers offset (the EE site the IK targets is
        # ~10mm from the actual midpoint between the two finger geoms, and
        # that offset's direction depends on the arm's converged wrist
        # orientation -- measuring it after the first pass and re-aiming
        # gets the fingers themselves, not just the wrist site, onto the
        # cable). Same APPROACH_SECONDS total as a single static approach,
        # so timing is unaffected.
        print("[dlo-v4] phase 5/8: r2+r3 approaching grasp positions...", flush=True)
        t = self._run_multi(APPROACH_SECONDS * 0.6, t, on_frame, ee_targets=grasp_pos)
        fine_targets = {}
        for name in ARM_NAMES:
            site = self.data.site_xpos[self._arms[name]["site_id"]].copy()
            lf, rf = self._gripper_finger_geom_ids[name]
            finger_mid = (self.data.geom_xpos[lf] + self.data.geom_xpos[rf]) / 2.0
            fine_targets[name] = grasp_pos[name] - (finger_mid - site)
        t = self._run_multi(APPROACH_SECONDS * 0.4, t, on_frame, ee_targets=fine_targets)

        print("[dlo-v4] phase 6/8: grasping (welds engage on finger-cable contact)...", flush=True)
        target_geoms = {"r2": self.cable_geom_ids[0], "r3": self.cable_geom_ids[-1]}
        target_bids = {"r2": r2_cable_bid, "r3": r3_cable_bid}
        steps_per_frame = max(1, round(FRAME_DT / self.model.opt.timestep))
        n_steps = max(1, round(GRASP_TIMEOUT_SECONDS / self.model.opt.timestep))
        engaged = {"r2": False, "r3": False}
        for step in range(n_steps):
            self.data.xfrc_applied[:, :] = 0.0
            for name in ARM_NAMES:
                self._ik_step(name, fine_targets[name])
                self._set_gripper(name, True)
            mujoco.mj_step(self.model, self.data)
            t += self.model.opt.timestep
            if on_frame is not None and step % steps_per_frame == 0:
                on_frame(self.model, self.data, t)
            for name in ARM_NAMES:
                if not engaged[name] and self._gripper_touching(name, target_geoms[name]):
                    print(f"[dlo-v4] {name}: finger-cable contact detected at t={t:.2f}s -- engaging weld", flush=True)
                    self._engage_weld(name, target_bids[name])
                    engaged[name] = True
            if all(engaged.values()):
                break
        for name in ARM_NAMES:
            if not engaged[name]:
                print(f"[dlo-v4] {name}: no contact detected within {GRASP_TIMEOUT_SECONDS:.1f}s -- engaging weld anyway", flush=True)
                self._engage_weld(name, target_bids[name])

        lifted_pos = {name: np.array([fine_targets[name][0], fine_targets[name][1], PICKUP_Z]) for name in ARM_NAMES}

        print(f"[dlo-v4] phase 7/8: picking up to z={PICKUP_Z:.3f} m ({PICKUP_SECONDS:.0f} s)...", flush=True)
        t = self._run_multi_interp(
            PICKUP_SECONDS, t, on_frame,
            starts=fine_targets, ends=lifted_pos,
            gripper_closed={"r2": True, "r3": True},
        )

        # r3's phase 8: translate to its single fixed hold position, then
        # hold for whatever remains of TOTAL_SECONDS.
        r3_hold = np.array(R3_HOLD_POS)
        r3_remaining_after_translate = max(0.0, TOTAL_SECONDS - (t + R3_TRANSLATE_SECONDS))
        print(f"[dlo-v4] phase 8: r3 translating to {tuple(r3_hold.round(3))} "
              f"({R3_TRANSLATE_SECONDS:.0f}s) then holding for the rest of the cycle "
              f"(~{r3_remaining_after_translate:.0f}s)...", flush=True)
        print(f"[dlo-v4] r2 phase 8/11: translating to waypoint 8 ({WAYPOINT_TRANSLATE_SECONDS:.0f}s)...", flush=True)
        wp8 = np.array(WAYPOINT_8)
        t = self._run_multi_interp(
            WAYPOINT_TRANSLATE_SECONDS, t, on_frame,
            starts={"r2": lifted_pos["r2"], "r3": lifted_pos["r3"]},
            ends={"r2": wp8, "r3": r3_hold},
            gripper_closed={"r2": True, "r3": True},
        )
        t = self._run_multi(
            WAYPOINT_HOLD_SECONDS, t, on_frame,
            ee_targets={"r2": wp8, "r3": r3_hold},
            gripper_closed={"r2": True, "r3": True},
        )

        wp9 = np.array(WAYPOINT_9)
        print(f"[dlo-v4] r2 phase 9/11: translating to waypoint 9 ({WAYPOINT_TRANSLATE_SECONDS:.0f}s)...", flush=True)
        t = self._run_multi_interp(
            WAYPOINT_TRANSLATE_SECONDS, t, on_frame,
            starts={"r2": wp8}, ends={"r2": wp9},
            gripper_closed={"r2": True, "r3": True},
        )
        t = self._run_multi(
            WAYPOINT_HOLD_SECONDS, t, on_frame,
            ee_targets={"r2": wp9, "r3": r3_hold},
            gripper_closed={"r2": True, "r3": True},
        )

        wp10 = np.array(WAYPOINT_10)
        print(f"[dlo-v4] r2 phase 10/11: translating to waypoint 10 ({WAYPOINT_TRANSLATE_SECONDS:.0f}s)...", flush=True)
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
        print(f"[dlo-v4] r2 phase 11/11: translating to waypoint 11 ({WAYPOINT_TRANSLATE_SECONDS:.0f}s), "
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

        print(f"[dlo-v4] sequence complete -- t={t:.1f} s", flush=True)

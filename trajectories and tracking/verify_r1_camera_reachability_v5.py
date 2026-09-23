"""V5 variant of verify_r1_camera_reachability.py -- targets
generate_triple_scene_v5's scene instead of v3's, since v5 moved r1's base
(and its rail) 0.25m further in -Y (see DLO_PLACEMENTS in
generate_triple_scene_v5.py), which invalidates any pose solved against
v3's r1 base position. Otherwise identical to the v3 script.

Offline reachability check for r1's 3 wrist-camera target poses
(Pose A-C, trajectories/r1_camera_ik.py). Solves each pose KINEMATICALLY
(mj_kinematics only, no dynamics stepping -- see r1_camera_ik.py's module
docstring for why), seeding each pose's solve from the PREVIOUS pose's
solved qpos (Pose A seeds from R1_HOME_QPOS) -- mirrors the runtime
scheduler's continuous A->B->C chain, not independent re-seeds.

For each pose: reports position/orientation error and collision status at
the solved (or best-effort, if unconverged) qpos, and prints a
REACHABLE/UNREACHABLE verdict.

"Nearest reachable pose" is defined precisely as whatever qpos the
constrained (joint-limited) kinematic IK solve actually converges to --
there is no separate "project onto feasible region" step; the solver's own
best-effort result under joint limits IS the nearest-reachable substitute
by construction. If that solved qpos also collides, it's still reported
(flagged UNREACHABLE with the collision noted) -- real collision avoidance
in the IK itself is out of scope unless a pose actually needs it in
practice.

Usage: venv/bin/python3 trajectories/verify_r1_camera_reachability_v5.py

Prints a ready-to-paste `set_solved_waypoints(...)` call at the end --
paste its output into run_dlo_v5_freeze.py as instructed.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import mujoco

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "mujoco_scenes"))
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "trajectories"))

from generate_triple_scene_v5 import generate  # noqa: E402
import r1_camera_ik_v5 as ik  # noqa: E402
from r1_camera_ik_v5 import CameraArm, R1_HOME_QPOS, RAIL_X  # noqa: E402

# Only the poses actually defined in r1_camera_ik_v5 -- supports the
# temporary A-only or A/B choreographies where POSE_C (etc.) is commented
# out. r1's rail is held constant at RAIL_X in V5.
_POSES = [(lbl, getattr(ik, f"POSE_{lbl}")) for lbl in ("A", "B", "C") if hasattr(ik, f"POSE_{lbl}")]

POS_TOL_M = 0.005              # 5 mm, adjustable
ROT_TOL_RAD = np.radians(2.0)  # 2 deg, adjustable
MAX_ITERS = 2000

BOX_GEOM_PREFIXES = ("box_", "workstation_")


def _wrap_near(q: np.ndarray, ref: np.ndarray, jnt_range: np.ndarray) -> np.ndarray:
    """Shift each joint of q by whole revolutions to the branch nearest
    ref, so a joint-space linear xfer between consecutive waypoints takes
    the SHORT path (the multi-start IK picks by pose error alone and can
    land on a 2*pi-shifted branch -- kinematically identical but it makes
    the wrist/base spin a full turn during the transition). Keeps the
    unwrapped value for any joint whose wrapped value would exit its
    limits."""
    out = q.copy()
    for i in range(len(q)):
        cand = ref[i] + ((q[i] - ref[i] + np.pi) % (2 * np.pi) - np.pi)
        if jnt_range[i, 0] <= cand <= jnt_range[i, 1]:
            out[i] = cand
    return out

# gripper_left_finger_base <-> gripper_right_finger_base is a benign,
# always-present near-zero-depth contact (dist ~= -7e-11, floating-point
# noise) baked into the coupled gripper joint's own resting geometry --
# confirmed present identically on r1/r2/r3 at their untouched default
# poses, unrelated to any IK solution. Excluded here, not a real collision.
_BENIGN_CONTACT_PAIRS = {frozenset(("gripper_left_finger_base", "gripper_right_finger_base"))}


def _is_benign(n1: str, n2: str) -> bool:
    suffix1 = n1.split("_", 1)[1] if "_" in n1 else n1
    suffix2 = n2.split("_", 1)[1] if "_" in n2 else n2
    return frozenset((suffix1, suffix2)) in _BENIGN_CONTACT_PAIRS


def _collisions(model: mujoco.MjModel, data: mujoco.MjData) -> list[str]:
    mujoco.mj_forward(model, data)
    hits = []
    for i in range(data.ncon):
        c = data.contact[i]
        n1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, c.geom1) or ""
        n2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, c.geom2) or ""
        if _is_benign(n1, n2):
            continue
        is_r1 = lambda n: n.startswith("r1_")
        is_box = lambda n: n.startswith(BOX_GEOM_PREFIXES)
        if (is_r1(n1) and is_box(n2)) or (is_r1(n2) and is_box(n1)):
            hits.append(f"{n1} <-> {n2}")
        elif is_r1(n1) and is_r1(n2) and n1 != n2:
            hits.append(f"self-collision {n1} <-> {n2}")
    return hits


N_RANDOM_SEEDS = 25  # multi-start count per pose, see main()'s docstring note


def _best_of_multistart(arm: CameraArm, model, data, pos, quat, chained_seed: np.ndarray,
                         rng: np.random.Generator) -> tuple[np.ndarray, float, float, list[str]]:
    """Solves from the chained (previous-pose) seed AND several random
    seeds spanning the joint-limit box, keeping whichever result has the
    lowest combined error. Multi-start is necessary here: a single-seed
    solve was found to consistently land in poor local minima for at least
    one pose (confirmed empirically -- 15 random restarts for that pose all
    converged to 214-615mm, never near zero, while a DIFFERENT pose from a
    fresh seed converged to 0.02mm/0.01deg in <200 iterations -- so a
    single seed's result cannot be trusted as the true nearest-reachable
    pose without checking whether a better local minimum exists)."""
    candidates = [chained_seed] + [
        np.array([rng.uniform(lo, hi) for lo, hi in arm.jnt_range]) for _ in range(N_RANDOM_SEEDS)
    ]
    best = None
    for seed in candidates:
        q, pos_err, rot_err = arm.solve_ik_kinematic(
            pos, quat, seed, max_iters=MAX_ITERS, pos_tol=POS_TOL_M, rot_tol=ROT_TOL_RAD,
        )
        arm.set_qpos(q)
        collisions = _collisions(model, data)
        score = (pos_err >= POS_TOL_M or rot_err >= ROT_TOL_RAD or bool(collisions), pos_err, rot_err)
        if best is None or score < best[0]:
            best = (score, q, pos_err, rot_err, collisions)
    _, q, pos_err, rot_err, collisions = best
    return q, pos_err, rot_err, collisions


def _sweep_transition(arm, model, data, rail_qadr, q0, rail0, q1, rail1, label, n=300):
    """Collision sweep across a transition where BOTH the 6-DOF arm qpos
    AND the rail position move together (linearly, matching how the
    runtime schedule interpolates each independently but simultaneously
    over the same xfer window)."""
    worst = []
    for frac in np.linspace(0, 1, n):
        q = q0 + (q1 - q0) * frac
        rail_x = rail0 + (rail1 - rail0) * frac
        data.qpos[rail_qadr] = rail_x
        arm.set_qpos(q)
        hits = _collisions(model, data)
        if hits:
            worst.append((frac, hits))
    if worst:
        print(f"[verify] transition {label}: {len(worst)}/{n} samples with contact. "
              f"First: frac={worst[0][0]:.3f} {worst[0][1]}", flush=True)
        return False
    print(f"[verify] transition {label}: clean, 0/{n} samples with contact", flush=True)
    return True


def main() -> int:
    generate(layout="dlo")
    model = mujoco.MjModel.from_xml_path(str(PROJECT_ROOT / "triple_lite6_cable_routing_dlo_v5.xml"))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    arm = CameraArm(model, data)
    rng = np.random.default_rng(0)

    rail_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "r1_rail_x")
    rail_qadr = model.jnt_qposadr[rail_jid]

    q_seed = R1_HOME_QPOS.copy()
    solved = {}
    labels = [lbl for lbl, _ in _POSES]
    all_reachable = True
    for label, (pos, quat) in _POSES:
        data.qpos[rail_qadr] = RAIL_X
        q_sol, pos_err, rot_err, collisions = _best_of_multistart(arm, model, data, pos, quat, q_seed, rng)
        # unwrap to the revolution nearest the previous waypoint -- same
        # pose (2*pi-periodic in each revolute joint), shortest xfer path
        q_sol = _wrap_near(q_sol, q_seed, arm.jnt_range)
        reachable = pos_err < POS_TOL_M and rot_err < ROT_TOL_RAD and not collisions
        all_reachable &= reachable
        solved[label] = q_sol

        verdict = "REACHABLE" if reachable else "UNREACHABLE -- nearest-reachable qpos used below"
        print(f"[verify] Pose {label}: {verdict}  rail_x={RAIL_X:+.3f}  "
              f"pos_err={pos_err*1000:.2f}mm rot_err={np.degrees(rot_err):.2f}deg "
              f"collisions={collisions}", flush=True)
        print(f"           solved qpos (rad) = {np.round(q_sol, 5).tolist()}", flush=True)

        q_seed = q_sol  # next pose's solve chains from this one, matching runtime

    print()
    print(f"[verify] {'ALL %d REACHABLE' % len(labels) if all_reachable else 'SOME POSES UNREACHABLE -- see nearest-reachable qpos above'}")
    print()

    # transitions: home -> first pose, then between consecutive poses (rail constant)
    all_clean = True
    prev_q, prev_lbl = R1_HOME_QPOS, "home"
    for label in labels:
        all_clean &= _sweep_transition(arm, model, data, rail_qadr, prev_q, RAIL_X, solved[label], RAIL_X, f"{prev_lbl}->{label}")
        prev_q, prev_lbl = solved[label], label
    print()
    print(f"[verify] {'ALL TRANSITIONS CLEAN' if all_clean else 'SOME TRANSITIONS COLLIDE -- see above'}")
    print()

    print("[verify] Paste this into run_dlo_v5_freeze.py (call once, before planner.run_sequence()):")
    print("from r1_camera_ik_v5 import set_solved_waypoints")
    print("set_solved_waypoints(")
    for label in ("A", "B", "C"):
        if label in solved:
            print(f"    np.array({np.round(solved[label], 6).tolist()}),  # Pose {label}")
        else:
            print(f"    np.zeros(6),  # Pose {label} (disabled / not defined)")
    print(")")

    return 0 if (all_reachable and all_clean) else 1


if __name__ == "__main__":
    raise SystemExit(main())

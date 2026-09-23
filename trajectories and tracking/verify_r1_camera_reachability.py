"""Offline reachability check for r1's 4 wrist-camera target poses
(Pose A-D, trajectories/r1_camera_ik.py). Solves each pose KINEMATICALLY
(mj_kinematics only, no dynamics stepping -- see r1_camera_ik.py's module
docstring for why), seeding each pose's solve from the PREVIOUS pose's
solved qpos (Pose A seeds from R1_HOME_QPOS) -- mirrors the runtime
scheduler's continuous A->B->C->D chain, not independent re-seeds.

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

Usage: venv/bin/python3 trajectories/verify_r1_camera_reachability.py

Prints a ready-to-paste `set_solved_waypoints(...)` call at the end --
paste its output into run_dlo_v3_freeze.py as instructed.
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

from generate_triple_scene_v3 import generate  # noqa: E402
from r1_camera_ik import CameraArm, POSE_A, POSE_B, POSE_C, POSE_D, R1_HOME_QPOS  # noqa: E402

POS_TOL_M = 0.005              # 5 mm, adjustable
ROT_TOL_RAD = np.radians(2.0)  # 2 deg, adjustable
MAX_ITERS = 2000

BOX_GEOM_PREFIXES = ("box_", "workstation_")

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


def main() -> int:
    generate(layout="dlo")
    model = mujoco.MjModel.from_xml_path(str(PROJECT_ROOT / "triple_lite6_cable_routing_dlo_v3.xml"))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    arm = CameraArm(model, data)
    rng = np.random.default_rng(0)

    q_seed = R1_HOME_QPOS.copy()
    solved = {}
    all_reachable = True
    for label, (pos, quat) in (("A", POSE_A), ("B", POSE_B), ("C", POSE_C), ("D", POSE_D)):
        q_sol, pos_err, rot_err, collisions = _best_of_multistart(arm, model, data, pos, quat, q_seed, rng)
        reachable = pos_err < POS_TOL_M and rot_err < ROT_TOL_RAD and not collisions
        all_reachable &= reachable
        solved[label] = q_sol

        verdict = "REACHABLE" if reachable else "UNREACHABLE -- nearest-reachable qpos used below"
        print(f"[verify] Pose {label}: {verdict}  "
              f"pos_err={pos_err*1000:.2f}mm rot_err={np.degrees(rot_err):.2f}deg "
              f"collisions={collisions}", flush=True)
        print(f"           solved qpos (rad) = {np.round(q_sol, 5).tolist()}", flush=True)

        q_seed = q_sol  # next pose's solve chains from this one, matching runtime

    print()
    print(f"[verify] {'ALL 4 REACHABLE' if all_reachable else 'SOME POSES UNREACHABLE -- see nearest-reachable qpos above'}")
    print()
    print("[verify] Paste this into run_dlo_v3_freeze.py (call once, before planner.run_sequence()):")
    print("from r1_camera_ik import set_solved_waypoints")
    print("set_solved_waypoints(")
    for label in ("A", "B", "C", "D"):
        print(f"    np.array({np.round(solved[label], 6).tolist()}),  # Pose {label}")
    print(")")

    return 0 if all_reachable else 1


if __name__ == "__main__":
    raise SystemExit(main())

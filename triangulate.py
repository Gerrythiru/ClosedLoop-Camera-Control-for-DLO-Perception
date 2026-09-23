"""Step 3: multi-view triangulation, implementing the paper's Section V-B
math exactly (Eq. 1's per-camera projection matrix V_i, and the multiview
pseudoinverse solve), fed by step 2's GT-associated 2D points
(trackdlo_eval/gt_correspondence.py) -- i.e. triangulating from the
EXTRACTED/associated pixel positions, not the raw GT projections, matching
what a real system would actually have available.

For each GT node index, gathers a ray from every pose where that node was
associated (>=1 pose may see it; triangulation itself needs >=2 non-
parallel rays to be well-posed -- nodes with fewer are flagged, not
silently triangulated from an underdetermined system).

Usage: venv/bin/python3 trackdlo_eval/triangulate.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import mujoco
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "mujoco_scenes"))
sys.path.insert(0, str(PROJECT_ROOT / "trajectories"))
sys.path.insert(0, str(PROJECT_ROOT / "rendering"))
sys.path.insert(0, str(PROJECT_ROOT))

from generate_triple_scene_v3 import generate  # noqa: E402
from r1_camera_ik import CameraArm, set_solved_waypoints  # noqa: E402
import r1_camera_ik as rik  # noqa: E402
from ground_truth.cable_gt import get_cable_ground_truth  # noqa: E402
from camera_intrinsics import get_intrinsics  # noqa: E402

CORR_DIR = PROJECT_ROOT / "results" / "r1_gt_correspondence"
OUT_DIR = PROJECT_ROOT / "results" / "r1_triangulation"
W, H = 1280, 720
RGB_CAMERA = "r1_d435i_rgb"
MIN_VIEWS = 2


def pixel_to_world_ray(u: float, v: float, k_matrix: np.ndarray, cam_xmat: np.ndarray) -> np.ndarray:
    """Paper's Eq. exactly: nu' = [px-cx, py-cy, f] in camera frame
    (f = fx = fy, square-pixel assumption matching get_intrinsics), then
    normalized and rotated into world frame via the camera's local-to-world
    rotation (data.cam_xmat).

    CRITICAL convention fix (same class of bug camera_extrinsic.py's own
    docstring warns about): nu' = [u-cx, v-cy, f] is written in OpenCV
    camera convention (+X right, +Y down, +Z forward -- v increasing
    downward, f along +Z). data.cam_xmat is MuJoCo's OWN camera-local-to-
    world rotation, in MuJoCo's convention (+X right, +Y UP, -Z forward).
    Rotating an OpenCV-convention ray directly by cam_xmat silently points
    it almost backward -- verified directly: dot product against the true
    ray direction was -0.94 (nearly opposite) before this fix, 1.0 (exact)
    after. Fix: convert the ray from OpenCV to MuJoCo convention (negate Y
    and Z) BEFORE applying cam_xmat."""
    cx, cy = k_matrix[0, 2], k_matrix[1, 2]
    f = k_matrix[0, 0]
    nu_prime = np.array([u - cx, v - cy, f])
    nu_cam_opencv = nu_prime / np.linalg.norm(nu_prime)
    nu_cam_mujoco = np.array([nu_cam_opencv[0], -nu_cam_opencv[1], -nu_cam_opencv[2]])
    nu_world = cam_xmat.reshape(3, 3) @ nu_cam_mujoco
    return nu_world


def triangulate_point(rays: list[np.ndarray], cam_positions: list[np.ndarray]) -> np.ndarray:
    """Paper's Eq. 1 + multiview pseudoinverse solve, for one 3D point from
    m>=2 rays: V_i = I - nu_i nu_i^T (per view), p = (sum V_i)^-1 (sum V_i @ t_ci)."""
    V_sum = np.zeros((3, 3))
    Vt_sum = np.zeros(3)
    for nu, t_ci in zip(rays, cam_positions):
        V = np.eye(3) - np.outer(nu, nu)
        V_sum += V
        Vt_sum += V @ t_ci
    return np.linalg.solve(V_sum, Vt_sum)


def main() -> int:
    set_solved_waypoints(
        np.array([-3.864376, 0.31669, 1.842791, -0.121133, 1.344281, 2.163996]),
        np.array([-6.010524, -0.289532, 0.326319, 0.870285, 1.338712, -1.011715]),
        np.array([2.440836, 0.252604, 0.914965, -3.993161, -1.404551, -5.308524]),
        np.array([0.241035, 0.127787, 1.00072, 0.613259, 1.471743, -0.91295]),
    )
    generate(layout="dlo")
    model = mujoco.MjModel.from_xml_path(str(PROJECT_ROOT / "triple_lite6_cable_routing_dlo_v3.xml"))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    arm = CameraArm(model, data)

    for _ in range(60000):
        mujoco.mj_step(model, data)
    mujoco.mj_forward(model, data)
    gt = get_cable_ground_truth(model, data, "red_cable")
    k_matrix = get_intrinsics(model, RGB_CAMERA, W, H)
    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, RGB_CAMERA)

    poses = {"A": rik.QPOS_A, "B": rik.QPOS_B, "C": rik.QPOS_C, "D": rik.QPOS_D}

    # Recompute each pose's camera position + rotation (needed for ray math).
    cam_pose_by_label = {}
    for label, qpos in poses.items():
        arm.set_qpos(qpos)
        mujoco.mj_forward(model, data)
        cam_pose_by_label[label] = (data.cam_xpos[cam_id].copy(), data.cam_xmat[cam_id].copy())

    # Load step 2's per-pose associations: node_idx -> assoc_pixel.
    node_observations: dict[int, list[tuple[str, np.ndarray]]] = {i: [] for i in range(20)}
    for label in poses:
        npz_path = CORR_DIR / f"pose_{label}_correspondence.npz"
        if not npz_path.exists():
            print(f"[triangulate] {npz_path} missing -- run gt_correspondence.py first")
            continue
        d = np.load(npz_path)
        for idx, px in zip(d["node_idx"], d["assoc_pixel"]):
            node_observations[int(idx)].append((label, px))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results = []
    print(f"{'node':>4} {'#views':>7} {'views':>12} {'err_mm':>8}  status")
    for i in range(20):
        obs = node_observations[i]
        if len(obs) < MIN_VIEWS:
            print(f"{i:>4} {len(obs):>7} {''.join(l for l,_ in obs):>12} {'--':>8}  SKIPPED (need >={MIN_VIEWS} views)")
            results.append({"node": i, "n_views": len(obs), "triangulated": None, "err_mm": None})
            continue

        rays, cam_positions = [], []
        for label, (u, v) in obs:
            cam_pos, cam_xmat = cam_pose_by_label[label]
            rays.append(pixel_to_world_ray(u, v, k_matrix, cam_xmat))
            cam_positions.append(cam_pos)

        p_hat = triangulate_point(rays, cam_positions)
        err_mm = float(np.linalg.norm(p_hat - gt[i])) * 1000
        views_str = "".join(l for l, _ in obs)
        print(f"{i:>4} {len(obs):>7} {views_str:>12} {err_mm:>8.2f}  OK")
        results.append({"node": i, "n_views": len(obs), "triangulated": p_hat.tolist(),
                         "gt": gt[i].tolist(), "err_mm": err_mm})

    triangulated_errs = [r["err_mm"] for r in results if r["err_mm"] is not None]
    n_skipped = sum(1 for r in results if r["err_mm"] is None)
    print()
    print(f"[triangulate] {len(triangulated_errs)}/20 nodes triangulated, {n_skipped}/20 skipped (insufficient views)")
    if triangulated_errs:
        arr = np.array(triangulated_errs)
        print(f"[triangulate] error (mm): mean={arr.mean():.2f} median={np.median(arr):.2f} "
              f"max={arr.max():.2f} min={arr.min():.2f}")

    import json
    with open(OUT_DIR / "triangulation_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"[triangulate] saved -> {OUT_DIR / 'triangulation_results.json'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

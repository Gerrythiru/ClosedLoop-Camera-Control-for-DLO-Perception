"""Step 2: correspondence between MuJoCo GT nodes and each pose's extracted
2D centerline (trackdlo_eval/extract_2d_shape.py), via GT-projection --
standing in for the paper's NN-predicted-state projection (Section V-A3),
since we don't have a working predictive model for our cable (see the
NN-Cosserat calibration work). Legitimate substitute here specifically
because the goal is validating triangulation against GT, not real
deployment.

Per visible GT node: project into the pose's image (ground_truth/
camera_extrinsic.py), then associate to the nearest point on the extracted
2D path via edge projection -- reusing point_to_curve_distances
(trackdlo_eval/point_to_curve.py), which is dimension-agnostic (works for
2D pixel coords the same as 3D world points) and already implements the
paper's exact "project onto nearest segment, clamped" formula.

Occluded/out-of-frame nodes (per the earlier coverage check) are skipped --
no meaningful 2D observation to associate with.

Usage: venv/bin/python3 trackdlo_eval/gt_correspondence.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
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
from ground_truth.occlusion_labels import label_node_occlusion, VISIBLE  # noqa: E402
from ground_truth.camera_extrinsic import get_camera_extrinsic, project_points  # noqa: E402
from camera_intrinsics import get_intrinsics  # noqa: E402
from point_to_curve import point_to_curve_distances  # noqa: E402

CALIB_DIR = PROJECT_ROOT / "results" / "r1_eye_in_hand_calib"
SHAPE_DIR = PROJECT_ROOT / "results" / "r1_2d_shape_extraction"
OUT_DIR = PROJECT_ROOT / "results" / "r1_gt_correspondence"
W, H = 1280, 720
RGB_CAMERA = "r1_d435i_rgb"


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

    # Settle the cable fully (matches the earlier coverage check).
    for _ in range(60000):
        mujoco.mj_step(model, data)
    mujoco.mj_forward(model, data)
    gt = get_cable_ground_truth(model, data, "red_cable")
    k_matrix = get_intrinsics(model, RGB_CAMERA, W, H)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    poses = {"A": rik.QPOS_A, "B": rik.QPOS_B, "C": rik.QPOS_C, "D": rik.QPOS_D}

    results = {}
    for label, qpos in poses.items():
        arm.set_qpos(qpos)
        mujoco.mj_forward(model, data)

        labels, _ = label_node_occlusion(model, data, RGB_CAMERA, gt, width=W, height=H)
        t_world_to_cam = get_camera_extrinsic(model, data, RGB_CAMERA)
        gt_pixels = project_points(gt, k_matrix, t_world_to_cam)  # (20, 2)

        path_file = SHAPE_DIR / f"pose_{label}_path_uv.npy"
        if not path_file.exists():
            print(f"[gt_correspondence] {path_file} missing -- run extract_2d_shape.py first")
            continue
        path_uv = np.load(path_file)

        visible_idx = [i for i in range(20) if labels[i] == VISIBLE]
        if not visible_idx:
            print(f"[gt_correspondence] pose {label}: no visible GT nodes, skipping")
            continue
        query = gt_pixels[visible_idx]
        dists, assoc_points = point_to_curve_distances(query, path_uv)

        entry = {
            "node_idx": visible_idx,
            "gt_pixel": gt_pixels[visible_idx],
            "assoc_pixel": assoc_points,
            "assoc_dist_px": dists,
        }
        results[label] = entry

        print(f"--- Pose {label} ---")
        print(f"  {len(visible_idx)} visible nodes associated. "
              f"assoc_dist_px: mean={dists.mean():.2f} max={dists.max():.2f} min={dists.min():.2f}")
        for i, gpx, apx, d in zip(visible_idx, gt_pixels[visible_idx], assoc_points, dists):
            print(f"    node {i:2d}: gt_proj=({gpx[0]:7.1f},{gpx[1]:7.1f})  "
                  f"assoc=({apx[0]:7.1f},{apx[1]:7.1f})  dist={d:6.2f}px")

        # Overlay: GT-projected (blue), associated path point (green), connecting line (yellow).
        rgb_path = CALIB_DIR / f"pose_{label}" / "rgb_00.png"
        rgb = cv2.imread(str(rgb_path))
        for gpx, apx in zip(gt_pixels[visible_idx], assoc_points):
            gpt = tuple(gpx.astype(int))
            apt = tuple(apx.astype(int))
            cv2.line(rgb, gpt, apt, (0, 255, 255), 1)
            cv2.circle(rgb, gpt, 4, (255, 0, 0), -1)   # GT projection = blue
            cv2.circle(rgb, apt, 3, (0, 255, 0), -1)   # associated path point = green
        out_path = OUT_DIR / f"pose_{label}_correspondence.png"
        cv2.imwrite(str(out_path), rgb)
        print(f"  saved overlay -> {out_path}")

        np.savez(
            OUT_DIR / f"pose_{label}_correspondence.npz",
            node_idx=np.array(visible_idx), gt_pixel=gt_pixels[visible_idx],
            assoc_pixel=assoc_points, assoc_dist_px=dists,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

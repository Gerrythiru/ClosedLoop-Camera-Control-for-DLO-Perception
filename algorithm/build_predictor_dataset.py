"""Step 1 of the failure-predictor build: join Bucket 1/2/3 logging into one
flat, per-frame feature table for a single run.

Sources joined (all onto the tracked-trajectory's own `t` grid, nearest-match
-- the same join pattern used by hand throughout today's investigation):
  - combined_tracked_trajectory*.npz   tracked node positions (world)
  - ground_truth.npz (per pose)        Bucket 2: GT nodes, ray-cast occlusion,
                                        material tag, r2/r3 qpos, self-occ-ahead
  - tracker_diag_*.npz                 Bucket 3: node_support (P1),
                                        iterations, converged, step time, cloud size
  - run_metadata_*.json                which poses ran, paint_markers flag, etc.
  - RGB captures (metadata.json)       Bucket 1: red-mask pixel support at both
                                        the tracked and true (GT) node positions

Also computes, purely from the tracked positions (no new capture needed):
total polyline length, max segment length, per-node turning angle (curvature),
and per-node inter-frame displacement (NaN across a >1s gap). Also computes
tangent_view_angle_deg (T,N) -- camera-ray vs. local-tangent angle at each
GT node, 0=grazing/foreshortened, 90=perpendicular/best-viewed -- added
2026-09-15 to let classify_risk.py separate genuine grazing-angle
foreshortening from other causes of an unexplained support collapse.

Output: results/r1_dlo_tracking_<pipeline>/predictor_features_<suffix>.npz,
one row per tracked frame. `per_node_error_mm` (tracked vs GT) is included as
the offline validation LABEL only -- it is not a deployment-available input
and must never be fed to the predictor itself; see today's design discussion.

Usage: venv/bin/python3 build_predictor_dataset.py --pipeline v5 [--suffix fkprior]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT / "trackdlo_eval"))
sys.path.insert(0, str(REPO_ROOT / "rendering"))

from run_r1_tracking_v5 import world_to_cam_transform, world_to_cam, pinhole_project, _paint_end_markers  # noqa: E402
from cable_mask import get_cable_mask  # noqa: E402

MASK_SAMPLE_RADIUS = 4          # px, half-width of the window sampled around each projected node
MAX_CAPTURE_GAP_S = 0.15        # skip mask-support if no real capture is this close to a tracked frame


def _run_paths(pipeline: str, suffix: str | None) -> dict:
    calib_root = REPO_ROOT / f"results/r1_eye_in_hand_calib_{pipeline}"
    track_root = REPO_ROOT / f"results/r1_dlo_tracking_{pipeline}"
    stem = "combined_tracked_trajectory" + (f"_{suffix}" if suffix else "")
    return dict(
        calib_root=calib_root,
        track_root=track_root,
        tracked_path=track_root / f"{stem}.npz",
        diag_path=track_root / f"tracker_diag_{suffix or 'default'}.npz",
        meta_path=track_root / f"run_metadata_{suffix or 'default'}.json",
        out_path=track_root / f"predictor_features_{suffix or 'default'}.npz",
    )


def _load_combined_gt(calib_root: Path, poses: list[str]) -> dict:
    """Concatenate ground_truth.npz across poses, sorted by t."""
    fields = ["t", "gt_nodes", "occlusion", "occluders", "r2_qpos", "r3_qpos",
              "r2_gripper_pos", "r2_gripper_quat", "r3_gripper_pos", "r3_gripper_quat",
              "self_occ_ahead_1s", "self_occ_ahead_3s"]
    acc = {f: [] for f in fields}
    material_tag = None
    for p in poses:
        z = np.load(calib_root / f"pose_{p}" / "ground_truth.npz", allow_pickle=True)
        for f in fields:
            acc[f].append(z[f])
        if material_tag is None:
            material_tag = z["material_tag"]
    order = np.argsort(np.concatenate(acc["t"]))
    out = {f: np.concatenate(acc[f])[order] for f in fields}
    out["material_tag"] = material_tag
    return out


def _load_combined_captures(calib_root: Path, poses: list[str]) -> list[dict]:
    """Per-capture metadata (rgb path + camera pose + t), across poses, sorted by t."""
    caps = []
    for p in poses:
        meta = json.load(open(calib_root / f"pose_{p}" / "metadata.json"))
        K = np.array(meta["rgb_intrinsics"], float)
        for c in meta["captures"]:
            caps.append(dict(t=float(c["t"]), rgb_path=calib_root / f"pose_{p}" / c["rgb_file"],
                              cam_pos=np.asarray(c["rgb_cam_pos"]), cam_quat=np.asarray(c["rgb_cam_quat"]), K=K))
    caps.sort(key=lambda c: c["t"])
    return caps


def _nearest_idx(query_t: np.ndarray, ref_t: np.ndarray) -> np.ndarray:
    ref_t = np.asarray(ref_t)
    return np.array([int(np.argmin(np.abs(ref_t - q))) for q in query_t])


def _project_px(world_pts: np.ndarray, cam_pos: np.ndarray, cam_quat: np.ndarray, K: np.ndarray) -> np.ndarray:
    w2c = world_to_cam_transform(cam_pos, cam_quat)
    return pinhole_project(world_to_cam(world_pts, w2c), K)


def _sample_mask(mask: np.ndarray, px: np.ndarray, radius: int = MASK_SAMPLE_RADIUS) -> np.ndarray:
    h, w = mask.shape
    out = np.full(len(px), -1, dtype=np.int32)
    for i, (u, v) in enumerate(px):
        u, v = int(round(u)), int(round(v))
        if 0 <= v < h and 0 <= u < w:
            out[i] = mask[max(0, v - radius):v + radius + 1, max(0, u - radius):u + radius + 1].sum()
    return out


def _tangent_view_angle(gt_nodes: np.ndarray, cam_pos: np.ndarray) -> np.ndarray:
    """gt_nodes (T,N,3), cam_pos (T,3) -> tangent_view_angle_deg (T,N).
    Angle between the camera->node viewing ray and the LOCAL cable tangent
    (central difference along the GT node chain; one-sided at the two
    endpoints), folded into [0, 90] since tangent direction sign is
    arbitrary. 0 = ray parallel to the tangent (grazing/foreshortened --
    the cable's cross-section projects to near-zero pixels even though
    nothing occludes it); 90 = ray perpendicular (best-viewed).

    Uses GT node positions, not the tracked estimate -- diagnostic-only,
    same rationale as mask_support_gt above: isolates the TRUE geometric
    condition independent of whether the tracker's own estimate is already
    corrupted. A live deployment without sim GT would need this computed
    from the tracker's current best estimate instead; not addressed here."""
    T, N, _ = gt_nodes.shape
    tangent = np.zeros_like(gt_nodes)
    tangent[:, 1:-1] = gt_nodes[:, 2:] - gt_nodes[:, :-2]
    tangent[:, 0] = gt_nodes[:, 1] - gt_nodes[:, 0]
    tangent[:, -1] = gt_nodes[:, -1] - gt_nodes[:, -2]
    tangent /= np.linalg.norm(tangent, axis=2, keepdims=True).clip(min=1e-9)

    view_ray = gt_nodes - cam_pos[:, None, :]
    view_ray /= np.linalg.norm(view_ray, axis=2, keepdims=True).clip(min=1e-9)

    dot = np.abs(np.sum(tangent * view_ray, axis=2)).clip(0, 1)
    return np.degrees(np.arccos(dot))


def _segment_stats(nodes: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """nodes: (T,N,3) -> total polyline length (T,), max segment length (T,),
    per-node turning angle in degrees (T,N) -- NaN at the two endpoints, where
    curvature isn't defined."""
    diffs = np.diff(nodes, axis=1)  # (T, N-1, 3)
    seg_len = np.linalg.norm(diffs, axis=2)
    total_len = seg_len.sum(axis=1)
    max_seg = seg_len.max(axis=1)

    T, N, _ = nodes.shape
    turning = np.full((T, N), np.nan)
    v1, v2 = diffs[:, :-1, :], diffs[:, 1:, :]
    n1, n2 = np.linalg.norm(v1, axis=2), np.linalg.norm(v2, axis=2)
    cos = np.sum(v1 * v2, axis=2) / np.clip(n1 * n2, 1e-9, None)
    turning[:, 1:-1] = np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))
    return total_len, max_seg, turning


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pipeline", choices=("v4", "v5"), required=True)
    ap.add_argument("--suffix", default=None,
                    help="tracked-trajectory suffix (e.g. 'fkprior'). Omit for the base/default run.")
    args = ap.parse_args()

    paths = _run_paths(args.pipeline, args.suffix)
    run_metadata = json.load(open(paths["meta_path"]))
    poses = run_metadata["poses_run"]

    trk = np.load(paths["tracked_path"])
    t, nodes = trk["t"], trk["nodes"]
    T, N, _ = nodes.shape
    print(f"[build] {args.pipeline} suffix={args.suffix!r}: {T} tracked frames, {N} nodes, poses={poses}", flush=True)

    gt = _load_combined_gt(paths["calib_root"], poses)
    caps = _load_combined_captures(paths["calib_root"], poses)
    cap_t = np.array([c["t"] for c in caps])

    diag_path = paths["diag_path"]
    diag = np.load(diag_path) if diag_path.exists() else None

    # --- join everything onto the tracked trajectory's own t grid ---
    gi = _nearest_idx(t, gt["t"])
    gt_nodes = gt["gt_nodes"][gi]
    occlusion = gt["occlusion"][gi]
    occluders = gt["occluders"][gi]
    self_occ_ahead_1s = gt["self_occ_ahead_1s"][gi]
    self_occ_ahead_3s = gt["self_occ_ahead_3s"][gi]
    r2_qpos, r3_qpos = gt["r2_qpos"][gi], gt["r3_qpos"][gi]

    if diag is not None and len(diag["node_support_t"]):
        nsi = _nearest_idx(t, diag["node_support_t"])
        node_support = diag["node_support"][nsi]
    else:
        node_support = np.full((T, N), np.nan)
    if diag is not None and len(diag["diag_t"]):
        dgi = _nearest_idx(t, diag["diag_t"])
        iterations = diag["iterations"][dgi]
        converged = diag["converged"][dgi]
        tracking_step_ms = diag["tracking_step_ms"][dgi]
        cloud_size = diag["cloud_size"][dgi]
    else:
        iterations = converged = tracking_step_ms = cloud_size = np.full(T, np.nan)

    ci = _nearest_idx(t, cap_t)
    cam_pos_per_frame = np.array([caps[i]["cam_pos"] for i in ci])

    # --- derived tracked-shape features (no new capture needed) ---
    total_len, max_seg, turning = _segment_stats(nodes)
    tangent_view_angle_deg = _tangent_view_angle(gt_nodes, cam_pos_per_frame)
    inter_frame_jump = np.full((T, N), np.nan)
    dt = np.diff(t)
    for k in range(1, T):
        if dt[k - 1] <= 1.0:  # skip across real gaps (pose transitions / dropped stretches)
            inter_frame_jump[k] = np.linalg.norm(nodes[k] - nodes[k - 1], axis=1)

    # --- mask support at both the tracked position and the true (GT) position ---
    # tracked-position support is the deployment-real input; GT-position support
    # is a diagnostic-only aid (needs sim GT) for telling "the tracker drifted
    # off a visible cable" apart from "the true position genuinely wasn't visible".
    paint_markers = bool(run_metadata.get("paint_markers", False))
    mask_support_tracked = np.full((T, N), -1, dtype=np.int32)
    mask_support_gt = np.full((T, N), -1, dtype=np.int32)
    print(f"[build] computing mask support per frame (loads each frame's RGB once; "
          f"paint_markers={paint_markers} -- replaying the same in-memory repaint "
          f"run_r1_tracking applied before publishing, since it's never saved to disk)...", flush=True)
    for k in range(T):
        c = caps[ci[k]]
        if abs(c["t"] - t[k]) > MAX_CAPTURE_GAP_S:
            continue
        rgb = np.array(Image.open(c["rgb_path"]).convert("RGB"))
        if paint_markers:
            rgb = _paint_end_markers(rgb)
        mask = get_cable_mask(rgb, "red")
        mask_support_tracked[k] = _sample_mask(mask, _project_px(nodes[k], c["cam_pos"], c["cam_quat"], c["K"]))
        mask_support_gt[k] = _sample_mask(mask, _project_px(gt_nodes[k], c["cam_pos"], c["cam_quat"], c["K"]))
        if k % 200 == 0:
            print(f"[build]   {k}/{T}", flush=True)

    # --- label (sim-only, offline validation -- NEVER a predictor input) ---
    per_node_error_mm = np.linalg.norm(nodes - gt_nodes, axis=2) * 1000.0

    np.savez(
        paths["out_path"],
        pipeline=args.pipeline, suffix=args.suffix or "default",
        t=t, material_tag=gt["material_tag"], occluders=occluders,
        gt_nodes=gt_nodes, occlusion=occlusion,
        self_occ_ahead_1s=self_occ_ahead_1s, self_occ_ahead_3s=self_occ_ahead_3s,
        r2_qpos=r2_qpos, r3_qpos=r3_qpos,
        node_support=node_support, iterations=iterations, converged=converged,
        tracking_step_ms=tracking_step_ms, cloud_size=cloud_size,
        total_length_m=total_len, max_segment_m=max_seg, turning_angle_deg=turning,
        tangent_view_angle_deg=tangent_view_angle_deg,
        inter_frame_jump_m=inter_frame_jump,
        mask_support_tracked=mask_support_tracked, mask_support_gt=mask_support_gt,
        per_node_error_mm=per_node_error_mm,
        paint_markers=paint_markers,
    )
    print(f"[build] wrote {paths['out_path']}  ({T} frames x {N} nodes)", flush=True)

    print(f"\n[build] quick summary -- node0 mean support(tracked/gt)="
          f"{mask_support_tracked[:, 0][mask_support_tracked[:, 0] >= 0].mean():.1f}/"
          f"{mask_support_gt[:, 0][mask_support_gt[:, 0] >= 0].mean():.1f}  "
          f"node0 mean error={per_node_error_mm[:, 0].mean():.1f}mm  "
          f"did_not_converge={int((~converged.astype(bool)).sum())}/{T}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

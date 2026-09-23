"""GT vs. TrackDLO overlay GIF for a V5 single-pose run -- the r1-wrist-camera
equivalent of the old render_gt_vs_tracked_overlay.py.

For sampled tracked frames it loads that frame's saved RGB
(results/r1_eye_in_hand_calib_v5/pose_{POSE}/rgb_*.png), projects both the
GT cable nodes (green) and TrackDLO's tracked nodes (red) into the wrist
camera using that frame's recorded extrinsic + intrinsics, and writes an
animated GIF. Node order is auto-flipped to line up with GT (same check as
evaluate_v5.py). GT nodes that were OCCLUDED that frame are drawn hollow.

Usage:
    venv/bin/python3 trackdlo_eval/render_v5_gt_vs_tracked.py            # pose A, 150 frames
    venv/bin/python3 trackdlo_eval/render_v5_gt_vs_tracked.py --pose A --n 220 --downscale 2
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "trackdlo_eval"))

from run_r1_tracking_v5 import world_to_cam_transform, world_to_cam, pinhole_project  # noqa: E402
from point_to_curve import point_to_curve_distances  # noqa: E402

CALIB_ROOT = PROJECT_ROOT / "results" / "r1_eye_in_hand_calib_v5"
TRACK_ROOT = PROJECT_ROOT / "results" / "r1_dlo_tracking_v5"
GT_COLOR = (40, 230, 70)      # green
TRK_COLOR = (0, 200, 255)     # cyan -- contrasts with the red cable (red-on-red is unreadable)


def _match_by_t(track_t, gt_t):
    key = {round(float(v), 6): i for i, v in enumerate(gt_t)}
    return np.array([key.get(round(float(v), 6), -1) for v in track_t])


def _draw_poly(draw, px, in_img, color, occluded=None):
    for i in range(len(px) - 1):
        if in_img[i] and in_img[i + 1]:
            draw.line([tuple(px[i]), tuple(px[i + 1])], fill=color, width=2)
    for i, (x, y) in enumerate(px):
        if not in_img[i]:
            continue
        r = 4
        if occluded is not None and occluded[i]:
            draw.ellipse([x - r, y - r, x + r, y + r], outline=color, width=2)
        else:
            draw.ellipse([x - r, y - r, x + r, y + r], fill=color, outline=(0, 0, 0))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pose", default="A", choices=("A", "B", "C"))
    ap.add_argument("--tracked", type=Path, default=None)
    ap.add_argument("--n", type=int, default=150, help="number of tracked frames to sample")
    ap.add_argument("--downscale", type=int, default=2)
    ap.add_argument("--duration", type=int, default=100, help="ms per GIF frame")
    args = ap.parse_args()
    pose = args.pose
    tracked_path = args.tracked or TRACK_ROOT / f"combined_tracked_trajectory_{pose}.npz"

    trk = np.load(tracked_path)
    track_t, track_nodes = trk["t"], trk["nodes"]
    gt = np.load(CALIB_ROOT / f"pose_{pose}" / "ground_truth.npz", allow_pickle=True)
    gt_t, gt_nodes, occ = gt["t"], gt["gt_nodes"], gt["occlusion"]
    meta = json.loads((CALIB_ROOT / f"pose_{pose}" / "metadata.json").read_text())
    caps = sorted(meta["captures"], key=lambda c: c["t"])
    K = np.array(meta["rgb_intrinsics"], dtype=np.float64)
    pose_dir = CALIB_ROOT / f"pose_{pose}"

    gi = _match_by_t(track_t, gt_t)
    if (gi < 0).any():
        raise SystemExit("some tracked frames have no GT/metadata match by t")
    gt_m, occ_m = gt_nodes[gi], occ[gi]

    d_fwd = np.linalg.norm(track_nodes[:, 0] - gt_m[:, 0], axis=1) + np.linalg.norm(track_nodes[:, -1] - gt_m[:, -1], axis=1)
    d_rev = np.linalg.norm(track_nodes[:, 0] - gt_m[:, -1], axis=1) + np.linalg.norm(track_nodes[:, -1] - gt_m[:, 0], axis=1)
    flipped = d_rev.mean() < d_fwd.mean()
    if flipped:
        track_nodes = track_nodes[:, ::-1, :].copy()
    print(f"[overlay] pose {pose}: {len(track_t)} tracked frames, node order "
          f"{'REVERSED->flipped' if flipped else 'forward'}")

    sel = np.unique(np.linspace(0, len(track_t) - 1, min(args.n, len(track_t))).round().astype(int))
    ds = args.downscale
    frames = []
    for k in sel:
        cap = caps[int(gi[k])]
        rgb = Image.open(pose_dir / cap["rgb_file"]).convert("RGB")
        W, H = rgb.size
        w2c = world_to_cam_transform(np.asarray(cap["rgb_cam_pos"]), np.asarray(cap["rgb_cam_quat"]))

        gt_cam = world_to_cam(gt_m[k], w2c)
        trk_cam = world_to_cam(track_nodes[k], w2c)
        gt_px = pinhole_project(gt_cam, K)
        trk_px = pinhole_project(trk_cam, K)
        gt_in = [(gt_cam[i, 2] > 0 and 0 <= gt_px[i, 0] < W and 0 <= gt_px[i, 1] < H) for i in range(len(gt_cam))]
        trk_in = [(trk_cam[i, 2] > 0 and 0 <= trk_px[i, 0] < W and 0 <= trk_px[i, 1] < H) for i in range(len(trk_cam))]

        d, _ = point_to_curve_distances(gt_m[k], track_nodes[k])
        rmse = float(np.sqrt(np.mean((d * 1000) ** 2)))

        draw = ImageDraw.Draw(rgb)
        _draw_poly(draw, gt_px, gt_in, GT_COLOR, occluded=(occ_m[k] == 1))
        _draw_poly(draw, trk_px, trk_in, TRK_COLOR)
        draw.text((8, 8), f"pose {pose}  t={cap['t']:.2f}s  frame {cap['index']}  "
                          f"point-to-curve RMSE {rmse:.1f} mm", fill=(255, 255, 0))
        draw.text((8, 22), "green=GT (hollow=occluded)   cyan=TrackDLO", fill=(255, 255, 0))
        frames.append(rgb.resize((W // ds, H // ds), Image.LANCZOS))

    out = TRACK_ROOT / f"gt_vs_tracked_{pose}.gif"
    out.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(out, save_all=True, append_images=frames[1:], duration=args.duration, loop=0, optimize=True)
    print(f"[overlay] wrote {out} ({len(frames)} frames, {out.stat().st_size/1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

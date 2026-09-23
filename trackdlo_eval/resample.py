"""D4: arc-length resampling for fair TrackDLO-vs-ground-truth RMSE comparison
(Mujoco Plan 0, Part D4).

TrackDLO returns M nodes; ground truth has N=CABLE_SEGMENTS nodes. Comparing
node i of one against node i of the other is only valid if M == N and both
are arc-length-uniform, which won't generally hold. This module resamples
both polylines to a common set of K uniformly-spaced arc-length positions so
per-node error is comparable.

Run standalone for the D4 self-check:

    python trackdlo_eval/resample.py

which resamples the current scene's ground-truth centerline at K=CABLE_SEGMENTS
and confirms the result matches the original segment centroids to < 1 mm --
expected since the raw ground truth is already uniformly spaced at ~40 mm
intervals (Plan 0's own D4 verification artifact).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "mujoco_scenes"))
sys.path.insert(0, str(PROJECT_ROOT))


def resample_polyline(nodes: np.ndarray, k: int) -> np.ndarray:
    """Resample a polyline to k uniformly-spaced arc-length positions."""
    nodes = np.asarray(nodes, dtype=np.float64)
    diffs = np.diff(nodes, axis=0)
    seg_lengths = np.linalg.norm(diffs, axis=1)
    cum_lengths = np.concatenate([[0.0], np.cumsum(seg_lengths)])
    total_length = cum_lengths[-1]
    target_lengths = np.linspace(0.0, total_length, k)
    resampled = np.zeros((k, 3), dtype=np.float64)
    for i, s in enumerate(target_lengths):
        idx = np.searchsorted(cum_lengths, s, side="right") - 1
        idx = int(np.clip(idx, 0, len(nodes) - 2))
        t = (s - cum_lengths[idx]) / max(seg_lengths[idx], 1e-8)
        t = float(np.clip(t, 0.0, 1.0))
        resampled[i] = nodes[idx] * (1 - t) + nodes[idx + 1] * t
    return resampled.astype(np.float32)


def main() -> int:
    import mujoco

    from generate_triple_scene import CABLE_SEGMENTS, generate  # noqa: E402
    from ground_truth.cable_gt import get_cable_ground_truth  # noqa: E402

    generate()
    model = mujoco.MjModel.from_xml_path(str(PROJECT_ROOT / "triple_lite6_cable_routing.xml"))
    data = mujoco.MjData(model)
    # Deliberately checked at the as-generated pose (mj_forward only), not
    # after settling: gravity + joint bending curls the chain, and a bent
    # joint's centroid-to-centroid distance is a chord shorter than the 40 mm
    # segment length -- that's correct physics, not resampling error, but it
    # means a settled cable isn't the uniform-spacing polyline Plan 0's D4
    # check assumes. This check validates resample_polyline's arithmetic
    # against the one configuration where uniform spacing actually holds.
    mujoco.mj_forward(model, data)

    gt = get_cable_ground_truth(model, data, "red_cable")
    resampled = resample_polyline(gt, CABLE_SEGMENTS)

    diffs = np.linalg.norm(resampled - gt, axis=1) * 1000.0
    print(f"[D4] resampled {CABLE_SEGMENTS} ground-truth nodes at K={CABLE_SEGMENTS}: "
          f"mean diff={diffs.mean():.3f} mm, max diff={diffs.max():.3f} mm "
          f"(expected < 1 mm -- raw GT is already ~40 mm uniform spacing)")

    if diffs.max() >= 1.0:
        print("[D4] WARNING: resampled centerline deviates >= 1 mm from raw ground truth.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

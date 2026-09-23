"""Step 2 of the failure-predictor build: turn per_node_error_mm (the
tracked-vs-GT label from build_predictor_dataset.py) into a precise,
reproducible per-node FAILURE boolean, so predictor precision/recall/lead-
time can be scored against something more principled than "eyeballed it".

Definition -- a hysteresis state machine per node, not a bare threshold:
  - a failure EPISODE starts once error exceeds FAIL_THRESHOLD_MM for
    MIN_CONSECUTIVE consecutive frames (onset = the FIRST of those frames,
    not the confirming one -- this is what "lead time" gets measured against)
  - it ends once error drops back to RECOVER_THRESHOLD_MM or below for
    MIN_CONSECUTIVE consecutive frames (hysteresis -- a bare single-threshold
    rule would flap right at the boundary)

Thresholds:
  FAIL_THRESHOLD_MM = 60     lowered 2026-09-14 from the original 100 (user
                             instruction) to catch smaller occlusion-driven
                             errors (Occlusion_Test_pose_2's real
                             geometric_occlusion nodes peaked at 36-71mm --
                             all real events, none flagged as failures at
                             the 100mm bar). NOTE: this invalidates the
                             original rationale below -- V4's worst standing
                             bias (~90-95mm) is now ABOVE this threshold, so
                             V4 is no longer a clean negative control at
                             this setting; re-check V4 runs if used again.
                             Original 100mm rationale (V5's node-0 runaway
                             crosses this repeatedly and stays there, 38
                             frames >100mm in the primary run; V4 never
                             crossed it in either pose, even during Pose C's
                             node-0 occlusion bias, tops out ~90mm) kept for
                             history, see Today_SUMM.
  RECOVER_THRESHOLD_MM = 50  UNCHANGED. At FAIL_THRESHOLD_MM=60 this leaves
                             only a 10mm hysteresis band (was 50mm) --
                             narrower margin against frame-to-frame noise
                             flapping the failing/recovered state than the
                             original design assumed. Not re-tuned; flag if
                             flapping shows up in a run's is_failure trace.
  MIN_CONSECUTIVE = 3        ~0.75s at the ~4Hz rate these runs use -- long
                             enough to reject a single noisy frame, short
                             enough not to eat into the lead-time budget.

Usage: venv/bin/python3 label_failures.py --pipeline v5 [--suffix fkprior]
Output: results/r1_dlo_tracking_<pipeline>/failure_labels_<suffix>.npz
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent

FAIL_THRESHOLD_MM = 60.0
RECOVER_THRESHOLD_MM = 50.0
MIN_CONSECUTIVE = 3


def _label_node(err: np.ndarray, fail_thr: float, recover_thr: float, min_consec: int) -> np.ndarray:
    """err (T,) -> is_failure (T,) bool, via the hysteresis rule above."""
    T = len(err)
    is_fail = np.zeros(T, dtype=bool)
    failing = False
    above_run = 0
    below_run = 0
    pending_start = None
    for k in range(T):
        e = err[k]
        if not failing:
            if e > fail_thr:
                above_run += 1
                if pending_start is None:
                    pending_start = k
            else:
                above_run = 0
                pending_start = None
            if above_run >= min_consec:
                failing = True
                is_fail[pending_start:k + 1] = True
                below_run = 0
        else:
            is_fail[k] = True
            below_run = below_run + 1 if e <= recover_thr else 0
            if below_run >= min_consec:
                failing = False
                is_fail[k - min_consec + 1:k + 1] = False
                above_run = 0
                pending_start = None
    return is_fail


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pipeline", choices=("v4", "v5"), required=True)
    ap.add_argument("--suffix", default=None)
    args = ap.parse_args()

    track_root = REPO_ROOT / f"results/r1_dlo_tracking_{args.pipeline}"
    suffix = args.suffix or "default"
    feat_path = track_root / f"predictor_features_{suffix}.npz"
    out_path = track_root / f"failure_labels_{suffix}.npz"

    z = np.load(feat_path)
    t, err = z["t"], z["per_node_error_mm"]
    T, N = err.shape

    is_failure = np.zeros((T, N), dtype=bool)
    onset_t = np.full(N, np.nan)
    peak_error_mm = err.max(axis=0)
    n_failing_frames = np.zeros(N, dtype=int)
    for i in range(N):
        is_failure[:, i] = _label_node(err[:, i], FAIL_THRESHOLD_MM, RECOVER_THRESHOLD_MM, MIN_CONSECUTIVE)
        onset_idx = np.argmax(is_failure[:, i]) if is_failure[:, i].any() else None
        if onset_idx is not None:
            onset_t[i] = t[onset_idx]
        n_failing_frames[i] = is_failure[:, i].sum()

    any_node_failing = is_failure.any(axis=1)

    np.savez(out_path, t=t, is_failure=is_failure, onset_t=onset_t,
             peak_error_mm=peak_error_mm, n_failing_frames=n_failing_frames,
             any_node_failing=any_node_failing,
             fail_threshold_mm=FAIL_THRESHOLD_MM, recover_threshold_mm=RECOVER_THRESHOLD_MM,
             min_consecutive=MIN_CONSECUTIVE)
    print(f"[label] wrote {out_path}")

    print(f"\n[label] {args.pipeline} suffix={suffix!r}: {T} frames, {N} nodes")
    print(f"  any-node-failing: {int(any_node_failing.sum())}/{T} frames")
    print(f"  per-node: onset_t / peak_error_mm / failing_frames")
    for i in range(N):
        if n_failing_frames[i] > 0 or peak_error_mm[i] > RECOVER_THRESHOLD_MM:
            onset_str = f"{onset_t[i]:.2f}s" if not np.isnan(onset_t[i]) else "never"
            print(f"    node{i:2d}: onset={onset_str:>8}  peak={peak_error_mm[i]:6.1f}mm  frames={n_failing_frames[i]:4d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

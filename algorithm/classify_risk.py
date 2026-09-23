"""Step 3 of the failure-predictor build: per-frame risk score + failure
CLASS, from signals available live (node_support, iterations/converged,
ray-cast occlusion, material tag, paint flag) -- never per_node_error_mm,
which is the sim-only label from step 2, used only to validate this script,
not as an input to it.

Two risk components, both trend-based (a single absolute threshold doesn't
generalize across nodes/pipelines with different baseline support -- node0's
healthy support is ~6-17, interior nodes' is ~15-20, so "collapsed" has to be
relative to each node's OWN recent baseline, not a fixed number):

  support_collapsed[k,i]  node_support[k,i] < SUPPORT_DROP_FRAC times the
                          trailing median of node_support[:,i] over the last
                          BASELINE_WINDOW frames (only flagged if that
                          baseline itself was reasonably healthy -- an
                          already-low node can't "collapse" further)

  solve_stress[k]         FRAME-level, not per-node: the trailing
                          did-not-converge rate over the last BASELINE_WINDOW
                          frames exceeds STRESS_RATE_THR. Frame-level because
                          today's investigation (E3b) found the CPD-LLE
                          joint-solve corruption hits the whole node vector
                          in one linear system each EM step, not one node in
                          isolation.

support_collapsed also excludes frames within TRANSITION_GUARD_WINDOWS of a
camera-pose-schedule hold window starting (transition_guard, see
_transition_guard) -- 2026-09-15 addition, after finding a camera transition
alone (no real tracking problem) produces a spurious collapse for the first
few frames of the new pose, because the trailing-median baseline is still
partly built from the PREVIOUS pose's (different) steady-state support level.

Classification (only assigned where risk is flagged) reconciles WHY, using
only analytic/kinematic signals a real deployment would actually have --
material_tag (known scene/grasp design), the paint_markers config flag,
ray-cast occlusion (in deployment: computed from known camera pose + robot
FK + a mesh model, not from vision), and tangent_view_angle_deg (camera-ray
vs. local-cable-tangent angle, see build_predictor_dataset.py -- diagnostic,
computed from GT nodes, see that function's docstring for the deployment
caveat):

  material==marker, paint off               -> blind_spot            (expected)
  material==marker, paint on, collapsed     -> blind_spot_anomaly    (pipeline bug flag)
  ray_cast==OCCLUDED                        -> geometric_occlusion   (NBV-fixable)
  ray_cast==VISIBLE, solve_stress           -> joint_solve_degeneracy
  ray_cast==VISIBLE, collapsed, angle<GRAZING_ANGLE_THR_DEG -> foreshortening (NBV-fixable)
  ray_cast==VISIBLE, collapsed, angle NOT confirmed grazing -> unexplained_support_collapse
                          (2026-09-15 addition -- was previously misclassified as
                          foreshortening by elimination; real cause is most often a
                          sub-threshold joint-solve disturbance too brief/mild to trip
                          STRESS_RATE_THR, see today's node0/1/2/13 trace. diagnostic_only,
                          NOT invoke_nbv -- no evidence a reposition would help this class.)
  neither support nor ray_cast flagged, but shape_stress             -> slip

Usage: venv/bin/python3 classify_risk.py --pipeline v5 [--suffix fkprior]
Output: results/r1_dlo_tracking_<pipeline>/risk_classes_<suffix>.npz
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT / "trajectories"))

BASELINE_WINDOW = 10        # trailing frames used for both rolling baselines
SUPPORT_DROP_FRAC = 0.3     # "collapsed" = below 30% of the node's own recent healthy median
MIN_BASELINE_SUPPORT = 2.0  # a node whose own baseline is already this low can't "collapse" further
STRESS_RATE_THR = 0.15      # >=15% did-not-converge in the trailing window = frame-level solve stress
JUMP_THR_M = 0.03           # per-node inter-frame jump above this (3cm at ~4Hz) flags shape stress
SUPPORT_COLLAPSE_MIN_CONSECUTIVE = 2  # reject single-frame node_support noise (see Today_SUMM); shorter
                                       # than label_failures.py's 3 -- this is meant to fire earlier
GRAZING_ANGLE_THR_DEG = 20.0  # tangent_view_angle_deg below this = genuinely near-parallel view --
                               # first-pass heuristic (see build_predictor_dataset.py's
                               # tangent_view_angle_deg), not yet validated against real grazing cases
TRANSITION_GUARD_WINDOWS = 1  # guard = this many BASELINE_WINDOW-lengths (in seconds, scaled by the
                               # run's own median frame dt) after each camera hold window starts --
                               # rejects the baseline-catch-up artifact a camera transition causes
                               # (see today's node7/node11 finding): the trailing-median baseline is
                               # still partly built from the PREVIOUS pose's support level right after
                               # a transition, so a real regime change reads as a false "collapse"

VISIBLE = 0  # matches ground_truth.occlusion_labels' convention (see dump_v*_ground_truth.py)

CLASSES = ("healthy", "blind_spot", "blind_spot_anomaly", "geometric_occlusion",
           "joint_solve_degeneracy", "foreshortening", "unexplained_support_collapse", "slip")


def _trailing_median(x: np.ndarray, window: int) -> np.ndarray:
    """x: (T,) -> (T,) trailing median over the last `window` samples
    (including the current one), expanding at the start."""
    T = len(x)
    out = np.empty(T)
    for k in range(T):
        lo = max(0, k - window + 1)
        out[k] = np.median(x[lo:k + 1])
    return out


def _trailing_rate(flags: np.ndarray, window: int) -> np.ndarray:
    T = len(flags)
    out = np.empty(T)
    for k in range(T):
        lo = max(0, k - window + 1)
        out[k] = flags[lo:k + 1].mean()
    return out


def _transition_guard(t: np.ndarray, pipeline: str) -> np.ndarray:
    """t: (T,) -> (T,) bool, True for the first TRANSITION_GUARD_WINDOWS*
    BASELINE_WINDOW tracked ROWS of a camera hold window (or during an
    actual transition / the pre-sequence hold_home, where hold_window_at
    returns None). Both v4/v5 camera-choreography modules expose an
    identical hold_window_at(t) -> (label, start, end) | None.

    Deliberately counts ROWS since the hold began, not elapsed seconds --
    _trailing_median (the thing this guards) windows by array row too, and
    a camera transition typically drops tracked frames entirely (no
    captures during the ~2s xfer), so the first post-transition row can
    already be >1s past the hold's nominal start time while still being
    row 1 of that hold's own data. A seconds-based guard was tried first
    and missed exactly this case (node7/11, 2026-09-15)."""
    ik = __import__("r1_camera_ik_v5" if pipeline == "v5" else "r1_camera_ik")
    window_len = TRANSITION_GUARD_WINDOWS * BASELINE_WINDOW
    guard = np.zeros(len(t), dtype=bool)
    prev_label, run_len = None, 0
    for k, tk in enumerate(t):
        w = ik.hold_window_at(float(tk))
        label = w[0] if w is not None else None
        if label is None or label != prev_label:
            run_len = 0
        run_len += 1
        prev_label = label
        guard[k] = (label is None) or (run_len <= window_len)
    return guard


def _require_persistence(flags: np.ndarray, min_consecutive: int) -> np.ndarray:
    """Zero out any run of True shorter than min_consecutive; keep runs
    >= min_consecutive as-is. Rejects a single noisy sample (e.g. node12/13's
    one-frame node_support dips) without eating into a real multi-frame
    episode's (e.g. node14's, node0's) onset time."""
    out = np.zeros_like(flags)
    T = len(flags)
    k = 0
    while k < T:
        if flags[k]:
            j = k
            while j < T and flags[j]:
                j += 1
            if j - k >= min_consecutive:
                out[k:j] = True
            k = j
        else:
            k += 1
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pipeline", choices=("v4", "v5"), required=True)
    ap.add_argument("--suffix", default=None)
    args = ap.parse_args()

    track_root = REPO_ROOT / f"results/r1_dlo_tracking_{args.pipeline}"
    suffix = args.suffix or "default"
    feat = np.load(track_root / f"predictor_features_{suffix}.npz", allow_pickle=True)
    out_path = track_root / f"risk_classes_{suffix}.npz"

    t = feat["t"]
    node_support = feat["node_support"]
    T, N = node_support.shape
    converged = feat["converged"].astype(bool)
    occlusion = feat["occlusion"]
    material_tag = feat["material_tag"]
    inter_frame_jump = feat["inter_frame_jump_m"]
    paint_markers = bool(feat["paint_markers"])
    if "tangent_view_angle_deg" in feat.files:
        tangent_angle = feat["tangent_view_angle_deg"]
    else:
        # older predictor_features file, predates this feature -- treat as
        # unknown so nothing is ever misclassified as genuine foreshortening
        tangent_angle = np.full((T, N), np.nan)

    # --- frame-level solve stress ---
    dnc_rate = _trailing_rate(~converged, BASELINE_WINDOW)
    solve_stress = dnc_rate > STRESS_RATE_THR

    # --- frames near a camera-pose transition, where the trailing baseline
    # is still partly built from the PREVIOUS pose's support level (see
    # TRANSITION_GUARD_WINDOWS above) ---
    transition_guard = _transition_guard(t, args.pipeline)

    # --- per-node support collapse, relative to each node's own trailing baseline ---
    support_baseline = np.zeros((T, N))
    support_collapsed = np.zeros((T, N), dtype=bool)
    for i in range(N):
        support_baseline[:, i] = _trailing_median(node_support[:, i], BASELINE_WINDOW)
        raw_collapsed = (
            (support_baseline[:, i] > MIN_BASELINE_SUPPORT)
            & (node_support[:, i] < SUPPORT_DROP_FRAC * support_baseline[:, i])
            & ~transition_guard
        )
        support_collapsed[:, i] = _require_persistence(raw_collapsed, SUPPORT_COLLAPSE_MIN_CONSECUTIVE)

    # --- per-node shape stress (slip) ---
    shape_stress = np.nan_to_num(inter_frame_jump, nan=0.0) > JUMP_THR_M

    # --- risk score + classification ---
    risk_score = 0.5 * support_collapsed.astype(float) + 0.5 * solve_stress[:, None]
    risk_score = np.maximum(risk_score, 0.3 * shape_stress.astype(float))

    risk_class = np.full((T, N), "healthy", dtype=object)
    for k in range(T):
        for i in range(N):
            is_marker = material_tag[i] != 0
            # Priority order matters: geometric_occlusion and
            # joint_solve_degeneracy are concrete, well-evidenced
            # explanations and must win over blind_spot_anomaly whenever
            # they apply -- a marker node can collapse for the SAME
            # joint-solve reason an interior node does (see today's E3b),
            # so "it's a marker node" alone is not evidence of a paint
            # failure. blind_spot_anomaly is a last-resort / residual
            # label: only when paint is on, nothing else explains the
            # collapse, and it's isolated to the marker itself.
            if is_marker and not paint_markers:
                risk_class[k, i] = "blind_spot"
            elif occlusion[k, i] != VISIBLE and support_collapsed[k, i]:
                risk_class[k, i] = "geometric_occlusion"
            elif solve_stress[k] and support_collapsed[k, i]:
                risk_class[k, i] = "joint_solve_degeneracy"
            elif support_collapsed[k, i] and is_marker and paint_markers:
                risk_class[k, i] = "blind_spot_anomaly"
            elif support_collapsed[k, i] and tangent_angle[k, i] < GRAZING_ANGLE_THR_DEG:
                # genuine near-parallel camera-ray-vs-tangent angle --
                # confirmed grazing view, not just "nothing else explains
                # it" (see 2026-09-14 investigation: node7/11's old
                # "foreshortening" turned out to be a transition-baseline
                # artifact, now excluded by transition_guard above, and
                # several others looked like sub-threshold joint-solve
                # wobble that never crossed STRESS_RATE_THR -- neither is
                # real foreshortening, so both now fall to the class below)
                risk_class[k, i] = "foreshortening"
            elif support_collapsed[k, i]:
                # support collapsed, visible, not a marker/paint issue, no
                # frame-level solve stress, and NOT a confirmed grazing
                # angle (or angle unknown) -- most likely sub-threshold
                # joint-solve disturbance too brief/mild to trip
                # STRESS_RATE_THR (see today's node0/1/2/13 trace: a
                # multi-node support hand-off within 1-2 frames, iterations
                # only mildly elevated, converged stays True throughout).
                # diagnostic_only, not invoke_nbv -- no evidence repositioning
                # the camera would help an algorithmic solve issue.
                risk_class[k, i] = "unexplained_support_collapse"
            elif shape_stress[k, i]:
                risk_class[k, i] = "slip"

    mitigation = np.where(
        np.isin(risk_class, ["geometric_occlusion", "foreshortening"]), "invoke_nbv",
        np.where(risk_class == "blind_spot_anomaly", "flag_pipeline_anomaly",
        np.where(risk_class == "blind_spot", "flag_pipeline_anomaly",
        np.where(risk_class == "joint_solve_degeneracy", "diagnostic_only",
        np.where(risk_class == "unexplained_support_collapse", "diagnostic_only",
        np.where(risk_class == "slip", "diagnostic_only", "none"))))))

    np.savez(out_path, t=t, risk_score=risk_score, risk_class=risk_class, mitigation=mitigation,
             support_collapsed=support_collapsed, solve_stress=solve_stress, shape_stress=shape_stress,
             transition_guard=transition_guard, tangent_view_angle_deg=tangent_angle,
             dnc_rate=dnc_rate, support_baseline=support_baseline)
    print(f"[classify] wrote {out_path}")

    # --- quick validation summary ---
    labels = np.load(track_root / f"failure_labels_{suffix}.npz")
    label_onset = labels["onset_t"]
    print(f"\n[classify] {args.pipeline} suffix={suffix!r}: {T} frames, {N} nodes  (paint_markers={paint_markers})")
    any_risk = (risk_class != "healthy").any(axis=1)
    print(f"  any-node-at-risk: {int(any_risk.sum())}/{T} frames")
    for i in range(N):
        classes_seen = sorted(set(risk_class[:, i]) - {"healthy"})
        if not classes_seen:
            continue
        first_risk_idx = np.argmax(risk_class[:, i] != "healthy")
        first_risk_t = t[first_risk_idx]
        lead = f"  (lead={label_onset[i]-first_risk_t:+.2f}s vs label onset)" if not np.isnan(label_onset[i]) else ""
        print(f"    node{i:2d}: first risk at t={first_risk_t:6.2f}s  classes={classes_seen}{lead}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

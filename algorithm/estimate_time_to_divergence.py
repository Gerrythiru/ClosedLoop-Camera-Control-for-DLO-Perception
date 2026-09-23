"""Step 4 of the failure-predictor build: for a node CURRENTLY inside a
support_collapsed risk episode (classify_risk.py), estimate seconds
remaining until it would cross label_failures.py's FAIL_THRESHOLD_MM --
using ONLY live-available signals (episode duration so far, risk class).
NEVER per_node_error_mm, the sim-only validation label.

Calibration: pooled from 3 known real runs (KNOWN_CALIBRATION_RUNS below),
each already generated earlier in this investigation. For every node with
a real onset_t (label_failures.py), the nearest support_collapsed episode
(classify_risk.py) is matched to that onset via _match_episode_to_onset
(within EPISODE_ONSET_MAX_GAP_S) -- this is deliberately NOT "first-ever
collapse for this node": node5 in the 'default' run collapses briefly at
t=9.87-10.14s but its real onset is 64s later at t=74.21s with healthy
support the whole time between -- matching that would fabricate a false
+64s lead. Matched samples are tagged:
  "lead"          episode precedes (or contains) onset within the gap -- a
                  real precursor signal, this is what estimation is built from
  "negative"      the episode starts only AFTER onset -- the risk flag would
                  have fired too late to be useful, kept for the backtest
                  report, never used to estimate a positive lead
  "no_precursor"  no episode within EPISODE_ONSET_MAX_GAP_S of onset at all --
                  this class of failure has NO detectable precursor in this
                  signal family (see node5, node7's real onset with zero risk
                  flags ever)

Scope note: episodes are built from support_collapsed only, not the full
risk_class taxonomy -- blind_spot fires from material_tag/paint_markers
alone, independent of support_collapsed, and doesn't need a duration-based
ETA (it's already known instantly, not something to count down to).

Given only 2 real "lead" samples total for the best-covered class
(geometric_occlusion: node14 +1.00s, node6 +11.00s -- a 10x spread), this
deliberately does NOT fit a regression: a fitted curve over 1-2 points per
class would imply a functional relationship that hasn't been observed to
even exist. It reports the per-class MEDIAN of "lead" samples, explicitly
tagged by how many samples back it (see STATUS_*), and refuses to produce
a number for a class with zero real lead samples. This mirrors
label_failures.py/classify_risk.py's existing practice of stating a
constant's exact provenance rather than hiding how little it's based on.

Backtest (see main()'s printed report): leave-one-out, raw comparisons
only -- no aggregate error metric. With n=2 for the only class that has
enough samples to attempt this, a single MAE number would overstate how
much this has actually been validated.

Usage: venv/bin/python3 estimate_time_to_divergence.py --pipeline v5 [--suffix fkprior]
Output: results/r1_dlo_tracking_<pipeline>/time_to_divergence_<suffix>.npz
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent

KNOWN_CALIBRATION_RUNS = ("divergent_pose_baseline", "occlusion_pose1_baseline", "default")
EPISODE_ONSET_MAX_GAP_S = 15.0  # max gap between an episode ENDING and the onset for them to be
                                 # considered causally linked. Not tight -- discovered while building
                                 # this script that support_collapsed's boolean flag does NOT stay
                                 # True for the full duration of a sustained collapse: once support
                                 # has been near-0 for more than ~BASELINE_WINDOW frames, the trailing
                                 # median baseline itself decays toward 0 too, so "collapsed relative
                                 # to baseline" stops being true even though the node is still in real
                                 # distress (see classify_risk.py's MIN_BASELINE_SUPPORT comment).
                                 # Confirmed real cases: node0/divergent_pose_baseline's episode ends
                                 # 5.99s before its onset, node6/default's ends 10.67s before its onset
                                 # -- both genuine, previously hand-verified leads (+6.26s, +11.01s) that
                                 # a tighter gap (5.0s, tried first) incorrectly excluded. node5/default's
                                 # spurious 64s gap stays excluded either way -- see the module docstring.
MIN_SAMPLES_FOR_ESTIMATE = 2    # below this: tag "single_sample_estimate", never presented as a median

STATUS_NOT_COLLAPSED = "not_collapsed"
STATUS_ESTIMATED = "estimated"
STATUS_SINGLE = "single_sample_estimate"
STATUS_INSUFFICIENT = "insufficient_class_data"

# (run_suffix, node_idx, class, tag, lead_s) -- tag one of "lead"/"negative"/"no_precursor";
# lead_s is NaN for "no_precursor" rows.
CalibRow = tuple[str, int, str, str, float]


def _episodes(flags: np.ndarray, t: np.ndarray, class_col: np.ndarray) -> list[tuple[float, float, str]]:
    """flags/class_col: (T,) for one node -> list of (start_t, end_t, class_at_start)
    for each contiguous True run in flags. Mirrors classify_risk.py's
    _require_persistence run-finding logic."""
    out = []
    T = len(flags)
    k = 0
    while k < T:
        if flags[k]:
            j = k
            while j < T and flags[j]:
                j += 1
            out.append((float(t[k]), float(t[j - 1]), str(class_col[k])))
            k = j
        else:
            k += 1
    return out


def _match_episode_to_onset(
    episodes: list[tuple[float, float, str]], onset_t: float, max_gap_s: float
) -> tuple[str, float, str] | None:
    """-> ("lead", lead_s, cls) | ("negative", lead_s, cls) | None (no
    detectable precursor). "lead": an episode that contains onset_t, or
    ends within max_gap_s before it. "negative": the closest episode
    instead STARTS after onset_t (risk flagged only after the node had
    already failed). Ties broken by proximity to onset_t."""
    if not episodes or np.isnan(onset_t):
        return None
    candidates: list[tuple[str, float, str, float]] = []
    for start, end, cls in episodes:
        if start <= onset_t and (onset_t - end) <= max_gap_s:
            candidates.append(("lead", onset_t - start, cls, abs(onset_t - start)))
        elif start > onset_t and (start - onset_t) <= max_gap_s:
            candidates.append(("negative", onset_t - start, cls, start - onset_t))
    if not candidates:
        return None
    candidates.sort(key=lambda c: c[3])
    tag, lead, cls, _ = candidates[0]
    return tag, lead, cls


def _build_calibration_detail(pipeline: str) -> list[CalibRow]:
    """Recomputed at runtime from KNOWN_CALIBRATION_RUNS' actual
    risk_classes_*.npz + failure_labels_*.npz files -- never hardcoded
    literals, so this can't silently drift out of sync if thresholds
    change again (as FAIL_THRESHOLD_MM already did once)."""
    track_root = REPO_ROOT / f"results/r1_dlo_tracking_{pipeline}"
    rows: list[CalibRow] = []
    for run in KNOWN_CALIBRATION_RUNS:
        risk_path = track_root / f"risk_classes_{run}.npz"
        lab_path = track_root / f"failure_labels_{run}.npz"
        if not risk_path.exists() or not lab_path.exists():
            print(f"[eta] WARNING: calibration run {run!r} missing ({risk_path.name}/{lab_path.name}) -- skipped")
            continue
        risk = np.load(risk_path, allow_pickle=True)
        lab = np.load(lab_path)
        if "transition_guard" not in risk.files:
            # predates the 2026-09-15 transition-guard + angle-validated-foreshortening fix --
            # any "foreshortening" row from this run reflects the OLD, elimination-based
            # definition (shown that session to have ZERO confirmed real grazing-angle cases),
            # not the current angle-confirmed one, and "unexplained_support_collapse" didn't
            # exist as a class yet. Not excluded here (still real support_collapsed/onset data),
            # but flagged so it isn't silently trusted as equivalent to current-run classes.
            print(f"[eta] WARNING: calibration run {run!r} predates the transition-guard/angle-split "
                  f"fix -- its 'foreshortening' rows use the OLD unvalidated definition", flush=True)
        t = risk["t"]
        support_collapsed = risk["support_collapsed"]
        risk_class = risk["risk_class"]
        onset_t = lab["onset_t"]
        T, N = support_collapsed.shape
        for i in range(N):
            if np.isnan(onset_t[i]):
                continue  # node never failed in this run -- nothing to calibrate against
            episodes = _episodes(support_collapsed[:, i], t, risk_class[:, i])
            m = _match_episode_to_onset(episodes, float(onset_t[i]), EPISODE_ONSET_MAX_GAP_S)
            if m is None:
                rows.append((run, i, "none", "no_precursor", float("nan")))
            else:
                tag, lead, cls = m
                rows.append((run, i, cls, tag, lead))
    return rows


def _per_class_estimate(calib: list[CalibRow]) -> dict[str, tuple[float, str, int]]:
    """class -> (eta_estimate_s, status, n_lead_samples), using only
    "lead"-tagged rows. Median if n >= MIN_SAMPLES_FOR_ESTIMATE, else the
    single observed value tagged as low-confidence."""
    by_class: dict[str, list[float]] = {}
    for _run, _node, cls, tag, lead in calib:
        if tag == "lead":
            by_class.setdefault(cls, []).append(lead)
    out: dict[str, tuple[float, str, int]] = {}
    for cls, leads in by_class.items():
        n = len(leads)
        if n >= MIN_SAMPLES_FOR_ESTIMATE:
            out[cls] = (float(np.median(leads)), STATUS_ESTIMATED, n)
        else:
            out[cls] = (float(leads[0]), STATUS_SINGLE, n)
    return out


def _episode_duration_so_far(flags_col: np.ndarray, t: np.ndarray, k: int) -> float:
    """Seconds since the start of the contiguous True-run in flags_col
    ending at (and including) index k. NaN if flags_col[k] is False."""
    if not flags_col[k]:
        return float("nan")
    start_idx = k
    while start_idx > 0 and flags_col[start_idx - 1]:
        start_idx -= 1
    return float(t[k] - t[start_idx])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pipeline", choices=("v4", "v5"), required=True)
    ap.add_argument("--suffix", default=None)
    args = ap.parse_args()

    track_root = REPO_ROOT / f"results/r1_dlo_tracking_{args.pipeline}"
    suffix = args.suffix or "default"
    risk = np.load(track_root / f"risk_classes_{suffix}.npz", allow_pickle=True)
    out_path = track_root / f"time_to_divergence_{suffix}.npz"

    t = risk["t"]
    support_collapsed = risk["support_collapsed"]
    risk_class = risk["risk_class"]
    T, N = support_collapsed.shape

    calib = _build_calibration_detail(args.pipeline)
    per_class = _per_class_estimate(calib)

    eta_s = np.full((T, N), np.nan)
    status = np.full((T, N), STATUS_NOT_COLLAPSED, dtype=object)
    episode_duration_s = np.full((T, N), np.nan)
    matched_class = np.full((T, N), "", dtype=object)
    calibration_n_lead = np.zeros((T, N), dtype=int)

    for i in range(N):
        for k in range(T):
            if not support_collapsed[k, i]:
                continue
            d = _episode_duration_so_far(support_collapsed[:, i], t, k)
            episode_duration_s[k, i] = d
            cls = str(risk_class[k, i])
            matched_class[k, i] = cls
            if cls in per_class:
                med, st, n = per_class[cls]
                eta_s[k, i] = med - d
                status[k, i] = st
                calibration_n_lead[k, i] = n
            else:
                status[k, i] = STATUS_INSUFFICIENT

    np.savez(
        out_path, t=t, eta_s=eta_s, status=status, episode_duration_s=episode_duration_s,
        matched_class=matched_class, calibration_n_lead=calibration_n_lead,
        calib_run=np.array([r[0] for r in calib], dtype=object),
        calib_node=np.array([r[1] for r in calib], dtype=int),
        calib_class=np.array([r[2] for r in calib], dtype=object),
        calib_tag=np.array([r[3] for r in calib], dtype=object),
        calib_lead_s=np.array([r[4] for r in calib], dtype=float),
    )
    print(f"[eta] wrote {out_path}")

    # --- calibration summary ---
    print(f"\n[eta] calibration pooled from {KNOWN_CALIBRATION_RUNS}:")
    for cls, (val, st, n) in sorted(per_class.items()):
        print(f"    {cls:28s} eta_estimate={val:+6.2f}s  status={st}  n_lead={n}")

    # --- backtest: leave-one-out, raw comparisons only (see module docstring) ---
    print(f"\n[eta] backtest (leave-one-out, raw comparisons only -- n too small for an aggregate metric):")
    lead_rows = [r for r in calib if r[3] == "lead"]
    by_class: dict[str, list[CalibRow]] = {}
    for r in lead_rows:
        by_class.setdefault(r[2], []).append(r)
    for cls, rows in sorted(by_class.items()):
        if len(rows) < 2:
            run, node, _, _, lead = rows[0]
            print(f"    {cls}: n=1 lead sample ({run} node{node} lead={lead:+.2f}s) -- insufficient for a leave-one-out check")
            continue
        for i, (run, node, _, _, lead) in enumerate(rows):
            others = [rows[j][4] for j in range(len(rows)) if j != i]
            pred = float(np.median(others))
            print(f"    {cls}: predicted {run} node{node} using the other {len(others)} sample(s) "
                  f"-> {pred:+.2f}s, actual {lead:+.2f}s")

    no_prec = [r for r in calib if r[3] == "no_precursor"]
    neg = [r for r in calib if r[3] == "negative"]
    for run, node, cls, _tag, _lead in no_prec:
        print(f"    {run} node{node}: no_precursor -- failed with zero detected risk episode, "
              f"undetectable by this signal family")
    for run, node, cls, _tag, lead in neg:
        print(f"    {run} node{node} ({cls}): negative lead={lead:+.2f}s -- risk flag fired AFTER "
              f"onset, would have been false confidence")

    n_active = int((status != STATUS_NOT_COLLAPSED).sum())
    print(f"\n[eta] {args.pipeline} suffix={suffix!r}: {n_active} (frame,node) cells currently inside a "
          f"risk episode; see status/eta_s for per-cell detail")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

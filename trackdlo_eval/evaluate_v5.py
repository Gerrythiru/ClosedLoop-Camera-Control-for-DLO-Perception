"""Numeric tracking-accuracy evaluation for a V5 single-pose TrackDLO run.

Compares results/r1_dlo_tracking_v5/combined_tracked_trajectory_{POSE}.npz
(world-frame, (T,N,3) with N = cable node count = 15 in V5, from
run_r1_tracking_v5.py --pose POSE) against the per-frame ground truth from
dump_v5_ground_truth.py
(results/r1_eye_in_hand_calib_v5/pose_{POSE}/ground_truth.npz).

Both are world-frame and N-node; frames match exactly by `t` (same
capture instants). Unlike the old v2 pipeline, node identity is carried
through TrackDLO's registration, so per-index error is meaningful here --
we report BOTH:

  * point-to-curve : each GT node's distance to the nearest point on the
                     tracked polyline (direction/param-independent, the
                     evaluate.py metric).
  * per-index      : ||tracked[i] - gt[i]|| for matched node index i
                     (only trustworthy where correspondence holds; we
                     auto-detect and correct a whole-cable order flip).

Errors are bucketed by the GT occlusion label (VISIBLE / OCCLUDED;
OUT_OF_FRAME dropped).

Usage:
    venv/bin/python3 trackdlo_eval/evaluate_v5.py            # pose A
    venv/bin/python3 trackdlo_eval/evaluate_v5.py --pose A --tracked <path.npz>
    venv/bin/python3 trackdlo_eval/evaluate_v5.py --chained  # A->B->C combined run
    venv/bin/python3 trackdlo_eval/evaluate_v5.py --bridged  # + transit-gap bridge, split by source
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "trackdlo_eval"))

from point_to_curve import point_to_curve_distances  # noqa: E402

CALIB_ROOT = PROJECT_ROOT / "results" / "r1_eye_in_hand_calib_v5"
TRACK_ROOT = PROJECT_ROOT / "results" / "r1_dlo_tracking_v5"
VISIBLE, OCCLUDED, OUT_OF_FRAME = 0, 1, 2


def _load_gt(label: str):
    """label 'A'/'B'/'C' -> that pose's ground_truth.npz. label 'chained'
    -> all three concatenated and sorted by t (for the A->B->C seed-handoff
    run's single combined_tracked_trajectory.npz). label 'full' ->
    ground_truth_full.npz, the continuous ~30 Hz replay grid (needed for the
    bridged run: bridge frames fall at instants that only exist there)."""
    if label == "full":
        g = np.load(CALIB_ROOT / "ground_truth_full.npz", allow_pickle=True)
        return g["t"], g["gt_nodes"], g["occlusion"]
    if label != "chained":
        g = np.load(CALIB_ROOT / f"pose_{label}" / "ground_truth.npz", allow_pickle=True)
        return g["t"], g["gt_nodes"], g["occlusion"]
    ts, ns, os_ = [], [], []
    for p in ("A", "B"):   # TEMP: Pose C disabled
        g = np.load(CALIB_ROOT / f"pose_{p}" / "ground_truth.npz", allow_pickle=True)
        ts.append(g["t"]); ns.append(g["gt_nodes"]); os_.append(g["occlusion"])
    t = np.concatenate(ts); order = np.argsort(t)
    return t[order], np.concatenate(ns)[order], np.concatenate(os_)[order]


def _match_by_t(track_t: np.ndarray, gt_t: np.ndarray) -> np.ndarray:
    """For each tracked frame, the index into gt_t with the same capture
    time (they come from the identical instant list). Raises on any miss."""
    gt_key = {round(float(v), 6): i for i, v in enumerate(gt_t)}
    idx = np.array([gt_key.get(round(float(v), 6), -1) for v in track_t])
    if (idx < 0).any():
        raise SystemExit(f"{int((idx < 0).sum())} tracked frames have no GT match by t")
    return idx


def _match_nearest(track_t: np.ndarray, gt_t: np.ndarray, tol_s: float) -> np.ndarray:
    """Nearest-time match (for the bridged run: bridge-frame timestamps sit
    on a 1/30 s grid that is NOT aligned to the GT replay ticks). Raises if
    any frame's nearest GT is further than tol_s."""
    order = np.argsort(gt_t)
    gt_sorted = gt_t[order]
    pos = np.searchsorted(gt_sorted, track_t)
    pos = np.clip(pos, 1, len(gt_sorted) - 1)
    left, right = gt_sorted[pos - 1], gt_sorted[pos]
    pick_left = (track_t - left) <= (right - track_t)
    idx = order[np.where(pick_left, pos - 1, pos)]
    dt = np.abs(gt_t[idx] - track_t)
    if dt.max() > tol_s:
        raise SystemExit(f"{int((dt > tol_s).sum())} frames have no GT within {tol_s*1000:.0f} ms "
                         f"(max {dt.max()*1000:.1f} ms)")
    print(f"[eval] nearest-t match: max |dt| = {dt.max()*1000:.1f} ms, mean {dt.mean()*1000:.2f} ms")
    return idx


def _tangential_split(trk_m: np.ndarray, gt_m: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split each per-index error vector trk[i]-gt[i] into the component
    ALONG the GT curve tangent at node i (arc-length parameterization slip
    -- a definitional mismatch, not a tracking failure) and PERPENDICULAR
    to it (actual shape error). Both (T,N), metres. Lets the report
    separate 'the curve is X mm off' from 'node i sits Y mm further along
    the same curve than GT's node i' (see Today_SUMM 2026-09-01 sec 4)."""
    tang = np.gradient(gt_m, axis=1)
    tang /= np.linalg.norm(tang, axis=2, keepdims=True) + 1e-12
    e = trk_m - gt_m
    along = np.einsum("fij,fij->fi", e, tang)
    perp = np.linalg.norm(e - along[..., None] * tang, axis=2)
    return along, perp


def _resolve_orientation(trk: np.ndarray, gt: np.ndarray) -> tuple[np.ndarray, bool]:
    """trk,gt: (F,20,3) already frame-matched. Returns (trk_maybe_flipped,
    was_flipped) so trk node index lines up with gt node index. Decision is
    global (mean over all frames) -- TrackDLO carries a fixed traversal
    direction within a session."""
    d_fwd = np.linalg.norm(trk[:, 0] - gt[:, 0], axis=1) + np.linalg.norm(trk[:, -1] - gt[:, -1], axis=1)
    d_rev = np.linalg.norm(trk[:, 0] - gt[:, -1], axis=1) + np.linalg.norm(trk[:, -1] - gt[:, 0], axis=1)
    if d_rev.mean() < d_fwd.mean():
        return trk[:, ::-1, :].copy(), True
    return trk, False


def _stats(x: np.ndarray) -> dict:
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {k: float("nan") for k in ("rmse", "mean", "median", "p95", "max", "n")}
    return {
        "rmse": float(np.sqrt(np.mean(x ** 2))),
        "mean": float(np.mean(x)),
        "median": float(np.median(x)),
        "p95": float(np.percentile(x, 95)),
        "max": float(np.max(x)),
        "n": int(x.size),
    }


def _fmt(s: dict) -> str:
    return (f"rmse={s['rmse']:6.1f}  mean={s['mean']:6.1f}  median={s['median']:6.1f}  "
            f"p95={s['p95']:6.1f}  max={s['max']:7.1f}  n={s['n']}")


def _shade_bridged(ax, t: np.ndarray, bridged_mask: np.ndarray | None) -> None:
    if bridged_mask is None or not bridged_mask.any():
        return
    b = bridged_mask.astype(int)
    edges = np.diff(np.concatenate([[0], b, [0]]))
    for s_i, e_i in zip(np.where(edges == 1)[0], np.where(edges == -1)[0] - 1):
        ax.axvspan(t[s_i], t[e_i], color="orange", alpha=0.18, lw=0,
                   label="_bridged" if s_i else "bridged span")


def _plots(out_dir: Path, pose: str, t: np.ndarray, rmse_p2c_t: np.ndarray, rmse_idx_t: np.ndarray,
           occ_frac_t: np.ndarray, node_p2c: np.ndarray, node_idx: np.ndarray,
           node_occ_frac: np.ndarray, p2c_vis: np.ndarray, p2c_occ: np.ndarray,
           bridged_mask: np.ndarray | None = None) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(11, 4))
    _shade_bridged(ax, t, bridged_mask)
    ax.plot(t, rmse_p2c_t, ".-", ms=4, label="point-to-curve RMSE (mm)")
    ax.plot(t, rmse_idx_t, ".-", ms=4, alpha=0.7, label="per-index RMSE (mm)")
    ax.set_xlabel("sim time t (s)"); ax.set_ylabel("per-frame RMSE (mm)")
    ax2 = ax.twinx()
    ax2.fill_between(t, occ_frac_t, alpha=0.15, color="red", step="mid")
    ax2.set_ylabel("fraction of nodes occluded", color="red"); ax2.set_ylim(0, 1)
    ax.set_title(f"{pose}: per-frame tracking error over time  ({len(t)} tracked frames)")
    ax.legend(loc="upper left"); fig.tight_layout()
    fig.savefig(out_dir / "rmse_over_time.png", dpi=110); plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 4))
    x = np.arange(len(node_p2c))
    ax.bar(x - 0.2, node_p2c, 0.4, label="point-to-curve RMSE (mm)")
    ax.bar(x + 0.2, node_idx, 0.4, label="per-index RMSE (mm)", alpha=0.8)
    for i in x:
        ax.text(i, max(node_p2c[i], node_idx[i]), f"{node_occ_frac[i]*100:.0f}%",
                ha="center", va="bottom", fontsize=7, color="red")
    ax.set_xlabel(f"node index (0 = r2 grasp end, {len(node_p2c)-1} = r3 grasp end)")
    ax.set_ylabel("RMSE over tracked frames (mm)")
    ax.set_title(f"{pose}: per-node error  (red % = fraction of frames that node was occluded)")
    ax.set_xticks(x); ax.legend(); fig.tight_layout()
    fig.savefig(out_dir / "rmse_by_node.png", dpi=110); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    data = [p2c_vis[np.isfinite(p2c_vis)], p2c_occ[np.isfinite(p2c_occ)]]
    ax.boxplot(data, tick_labels=[f"VISIBLE\n(n={data[0].size})", f"OCCLUDED\n(n={data[1].size})"], showfliers=False)
    ax.set_ylabel("point-to-curve error (mm)")
    ax.set_title(f"{pose}: error by GT occlusion label")
    fig.tight_layout(); fig.savefig(out_dir / "error_by_occlusion.png", dpi=110); plt.close(fig)
    print(f"[eval] plots -> {out_dir}/")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pose", default="A", choices=("A", "B", "C"))
    ap.add_argument("--chained", action="store_true",
                    help="evaluate the A->B->C seed-handoff run "
                         "(combined_tracked_trajectory.npz vs all-3-poses GT) instead of one pose")
    ap.add_argument("--bridged", action="store_true",
                    help="evaluate the transit-gap-bridged chained run "
                         "(combined_tracked_trajectory_bridged.npz vs ground_truth_full.npz); "
                         "metrics are also split by frame source (tracked vs bridged)")
    ap.add_argument("--tracked", type=Path, default=None,
                    help="tracked npz (default: combined_tracked_trajectory[_{POSE}|_bridged].npz)")
    args = ap.parse_args()
    if args.bridged:
        label, gt_label = "bridged", "full"
        tracked_path = args.tracked or TRACK_ROOT / "combined_tracked_trajectory_bridged.npz"
    elif args.chained:
        label, gt_label = "chained", "chained"
        tracked_path = args.tracked or TRACK_ROOT / "combined_tracked_trajectory.npz"
    else:
        label, gt_label = args.pose, args.pose
        tracked_path = args.tracked or TRACK_ROOT / f"combined_tracked_trajectory_{label}.npz"

    if not tracked_path.exists():
        raise SystemExit(f"{tracked_path} not found -- "
                         + ("run `trackdlo_eval/bridge_transit_gaps.py` first" if args.bridged
                            else "run `python3 trackdlo_eval/run_r1_tracking.py` (no --pose) first" if args.chained
                            else f"run `run_r1_tracking.py --pose {label}` first"))
    trk_npz = np.load(tracked_path)
    track_t, track_nodes = trk_npz["t"], trk_npz["nodes"]  # (T,), (T,N,3) world
    source = trk_npz["source"] if "source" in trk_npz.files else None
    gt_t, gt_nodes, occ = _load_gt(gt_label)

    gi = _match_nearest(track_t, gt_t, tol_s=0.02) if args.bridged else _match_by_t(track_t, gt_t)
    gt_m = gt_nodes[gi]           # (T,N,3)
    occ_m = occ[gi]              # (T,N)
    trk_m, flipped = _resolve_orientation(track_nodes, gt_m)

    T = len(track_t)
    N = trk_m.shape[1]                                            # cable node count (V4: 20, V5: 11)
    per_idx = np.linalg.norm(trk_m - gt_m, axis=2) * 1000.0        # (T,N) mm
    _along, _perp = _tangential_split(trk_m, gt_m)
    slip_mm = np.abs(_along) * 1000.0                              # (T,N) arc-length slip
    shape_mm = _perp * 1000.0                                     # (T,N) perpendicular / shape error
    p2c = np.zeros((T, N))
    for f in range(T):
        d, _ = point_to_curve_distances(gt_m[f], trk_m[f])
        p2c[f] = d * 1000.0

    vis_mask = occ_m == VISIBLE
    occ_mask = occ_m == OCCLUDED
    in_frame = occ_m != OUT_OF_FRAME

    # per-frame
    rmse_p2c_t = np.sqrt(np.nanmean(np.where(in_frame, p2c, np.nan) ** 2, axis=1))
    rmse_idx_t = np.sqrt(np.nanmean(np.where(in_frame, per_idx, np.nan) ** 2, axis=1))
    occ_frac_t = occ_mask.sum(1) / np.clip(in_frame.sum(1), 1, None)
    # per-node
    node_p2c = np.sqrt(np.nanmean(np.where(in_frame, p2c, np.nan) ** 2, axis=0))
    node_idx = np.sqrt(np.nanmean(np.where(in_frame, per_idx, np.nan) ** 2, axis=0))
    node_occ_frac = occ_mask.sum(0) / np.clip(in_frame.sum(0), 1, None)

    dur_gt = float(gt_t.max() - gt_t.min())
    print("=" * 78)
    print(f"{label:7} tracked {tracked_path.name}")
    print(f"  frames tracked         : {T} / {len(gt_t)}  ({100*T/len(gt_t):.0f}%)")
    print(f"  tracked time span      : {track_t.min():.2f}-{track_t.max():.2f}s "
          f"of {gt_t.min():.2f}-{gt_t.max():.2f}s")
    print(f"  node-order vs GT        : {'REVERSED (auto-corrected for this eval)' if flipped else 'forward'}")
    print(f"  GT node visibility      : VISIBLE {vis_mask.sum()}  OCCLUDED {occ_mask.sum()}  "
          f"OUT_OF_FRAME {(occ_m == OUT_OF_FRAME).sum()}  (of {T*N})")
    print("-" * 78)
    print("point-to-curve (GT node -> nearest point on tracked polyline), mm:")
    print(f"  all in-frame : {_fmt(_stats(p2c[in_frame]))}")
    print(f"  VISIBLE only : {_fmt(_stats(p2c[vis_mask]))}")
    print(f"  OCCLUDED only: {_fmt(_stats(p2c[occ_mask]))}")
    print("per-index (||tracked[i] - gt[i]||), mm:")
    print(f"  all in-frame : {_fmt(_stats(per_idx[in_frame]))}")
    print(f"  VISIBLE only : {_fmt(_stats(per_idx[vis_mask]))}")
    print(f"  OCCLUDED only: {_fmt(_stats(per_idx[occ_mask]))}")
    print("  decomposition of the SAME error vector (see Today_SUMM 2026-09-01 sec 4):")
    print(f"    perpendicular (shape)     : {_fmt(_stats(shape_mm[in_frame]))}")
    print(f"    tangential (arc-len slip) : {_fmt(_stats(slip_mm[in_frame]))}")
    print("-" * 78)
    print(f"per-node point-to-curve RMSE (mm), 0=r2 end .. {N-1}=r3 end:")
    print("  " + "  ".join(f"{i:2d}:{node_p2c[i]:.0f}" for i in range(N)))
    node_shape = np.sqrt(np.nanmean(np.where(in_frame, shape_mm, np.nan) ** 2, axis=0))
    print("per-node perpendicular (shape) RMSE (mm) -- parameterization slip removed:")
    print("  " + "  ".join(f"{i:2d}:{node_shape[i]:.0f}" for i in range(N)))
    print("per-node occluded-fraction:")
    print("  " + "  ".join(f"{i:2d}:{node_occ_frac[i]*100:.0f}%" for i in range(N)))

    if source is not None:
        trk_b = (source == 0)[:, None] & in_frame
        brg_b = (source == 1)[:, None] & in_frame
        print("-" * 78)
        print(f"by frame source  (tracked {int((source==0).sum())} frames, "
              f"bridged {int((source==1).sum())} frames):")
        print(f"  point-to-curve  tracked : {_fmt(_stats(p2c[trk_b]))}")
        print(f"  point-to-curve  bridged : {_fmt(_stats(p2c[brg_b]))}")
        print(f"  per-index       tracked : {_fmt(_stats(per_idx[trk_b]))}")
        print(f"  per-index       bridged : {_fmt(_stats(per_idx[brg_b]))}")
        # per contiguous bridged run (= one transit gap)
        b = (source == 1).astype(int)
        edges = np.diff(np.concatenate([[0], b, [0]]))
        runs = list(zip(np.where(edges == 1)[0], np.where(edges == -1)[0] - 1))
        for gi_, (s_i, e_i) in enumerate(runs, 1):
            sl = slice(s_i, e_i + 1)
            fr = in_frame[sl]
            n0 = _stats(per_idx[sl][:, 0][fr[:, 0]])
            n19 = _stats(per_idx[sl][:, -1][fr[:, -1]])
            print(f"  gap {gi_} (t {track_t[s_i]:.2f}-{track_t[e_i]:.2f}s, {e_i-s_i+1} frames): "
                  f"p2c rmse={_stats(p2c[sl][fr])['rmse']:.1f}  "
                  f"per-index rmse={_stats(per_idx[sl][fr])['rmse']:.1f}  "
                  f"node0 rmse={n0['rmse']:.1f} (max {n0['max']:.0f})  "
                  f"node19 rmse={n19['rmse']:.1f} (max {n19['max']:.0f})")
    print("=" * 78)

    out_dir = TRACK_ROOT / f"eval_{label}"
    _plots(out_dir, label, track_t, rmse_p2c_t, rmse_idx_t, occ_frac_t,
           node_p2c, node_idx, node_occ_frac, p2c[vis_mask], p2c[occ_mask],
           bridged_mask=(source == 1) if source is not None else None)
    np.savez(out_dir / "summary.npz",
             t=track_t, per_index_mm=per_idx, point_to_curve_mm=p2c, occlusion=occ_m,
             shape_mm=shape_mm, slip_mm=slip_mm,
             rmse_p2c_over_time=rmse_p2c_t, rmse_idx_over_time=rmse_idx_t,
             node_p2c_rmse=node_p2c, node_idx_rmse=node_idx, node_shape_rmse=node_shape,
             flipped=flipped,
             source=source if source is not None else np.zeros(T, dtype=np.int8))
    print(f"[eval] summary -> {out_dir}/summary.npz")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

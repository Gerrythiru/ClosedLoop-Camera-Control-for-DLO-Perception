"""Stitch the A/B/C single-pose TrackDLO runs into one world-frame timeline
and animate GT vs. tracked cable shape in 3D over time.

Inputs (per pose P in A,B,C):
  results/r1_dlo_tracking_v4/combined_tracked_trajectory_{P}.npz  (t, nodes)
  results/r1_eye_in_hand_calib_v4/pose_{P}/ground_truth.npz       (t, gt_nodes, occlusion)

Both are world/base frame, 20 nodes, matched exactly by capture time `t`.
Each single-pose run's node order is auto-checked and flipped so tracked
node i lines up with GT node i. The two camera-transit gaps (~25-30s,
~34-37s) are simply absent from the timeline.

Outputs:
  results/r1_dlo_tracking_v4/stitch_3d_gt_vs_tracked.html   (plotly: rotate/zoom + time slider)
  results/r1_dlo_tracking_v4/stitch_3d_gt_vs_tracked.gif    (matplotlib: fixed iso view)

Encoding (both):
  * GT polyline green; GT nodes that were OCCLUDED from r1's camera that
    frame drawn hollow.
  * tracked polyline: single colour (cyan).
  * thin grey connectors tracked[i]<->gt[i] (per-index drift).
  * per-frame point-to-curve RMSE (mm) in the title.

Usage:
    venv/bin/python3 trackdlo_eval/stitch_3d_gt_vs_tracked.py
    venv/bin/python3 trackdlo_eval/stitch_3d_gt_vs_tracked.py --gif-frames 250 --no-html
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "trackdlo_eval"))
from point_to_curve import point_to_curve_distances  # noqa: E402

CALIB_ROOT = PROJECT_ROOT / "results" / "r1_eye_in_hand_calib_v4"
TRACK_ROOT = PROJECT_ROOT / "results" / "r1_dlo_tracking_v4"
POSES = ("A", "B", "C")
TRK_COLOR = "#00b7eb"   # tracked polyline (single colour, both GIF and HTML)
BRIDGE_COLOR = "#e83e8c"  # tracked polyline on kinematically-bridged (synthetic) frames
GT_COLOR = "#2ca02c"
CONN_COLOR = "rgba(120,120,120,0.55)"


def _load_pose(pose: str):
    trk = np.load(TRACK_ROOT / f"combined_tracked_trajectory_{pose}.npz")
    gt = np.load(CALIB_ROOT / f"pose_{pose}" / "ground_truth.npz", allow_pickle=True)
    t_trk, nodes = trk["t"], trk["nodes"]                      # (T,), (T,20,3)
    gt_key = {round(float(v), 6): i for i, v in enumerate(gt["t"])}
    gi = np.array([gt_key.get(round(float(v), 6), -1) for v in t_trk])
    if (gi < 0).any():
        raise SystemExit(f"pose {pose}: {(gi < 0).sum()} tracked frames unmatched to GT by t")
    gt_m = gt["gt_nodes"][gi]
    occ_m = gt["occlusion"][gi]
    # global node-order check (TrackDLO keeps a fixed traversal per session)
    d_fwd = np.linalg.norm(nodes[:, 0] - gt_m[:, 0], axis=1) + np.linalg.norm(nodes[:, -1] - gt_m[:, -1], axis=1)
    d_rev = np.linalg.norm(nodes[:, 0] - gt_m[:, -1], axis=1) + np.linalg.norm(nodes[:, -1] - gt_m[:, 0], axis=1)
    flipped = d_rev.mean() < d_fwd.mean()
    if flipped:
        nodes = nodes[:, ::-1, :].copy()
    print(f"[stitch] pose {pose}: {len(t_trk)} frames  t={t_trk.min():.2f}-{t_trk.max():.2f}s  "
          f"node order {'REVERSED->flipped' if flipped else 'forward'}")
    return t_trk, nodes, gt_m, occ_m


def _pose_at(t: float) -> str:
    return "A" if t < 27.0 else "B" if t < 36.0 else "C"


def _rows_from(t, trk, gt, occ, pose_fn, bridged=None):
    rows = []
    for k in range(len(t)):
        d, _ = point_to_curve_distances(gt[k], trk[k])
        rows.append({"t": float(t[k]), "pose": pose_fn(t[k]), "trk": trk[k], "gt": gt[k],
                     "occ": occ[k], "rmse": float(np.sqrt(np.mean((d * 1000) ** 2))),
                     "bridged": bool(bridged[k]) if bridged is not None else False})
    return rows


def _assemble():
    """3 independent single-pose runs (combined_tracked_trajectory_{A,B,C}.npz)."""
    rows = []
    for p in POSES:
        t, trk, gt, occ = _load_pose(p)
        rows += _rows_from(t, trk, gt, occ, lambda _t, p=p: p)
    rows.sort(key=lambda r: r["t"])
    return rows


def _assemble_chained():
    """The A->B->C seed-handoff run: one combined_tracked_trajectory.npz vs
    all-3-poses GT concatenated. One global node-order check (the chained
    run carries a single node array through the whole timeline)."""
    trk_npz = np.load(TRACK_ROOT / "combined_tracked_trajectory.npz")
    t_trk, nodes = trk_npz["t"], trk_npz["nodes"]
    gts = [np.load(CALIB_ROOT / f"pose_{p}" / "ground_truth.npz", allow_pickle=True) for p in POSES]
    gt_t = np.concatenate([g["t"] for g in gts])
    gt_nodes = np.concatenate([g["gt_nodes"] for g in gts])
    gt_occ = np.concatenate([g["occlusion"] for g in gts])
    key = {round(float(v), 6): i for i, v in enumerate(gt_t)}
    gi = np.array([key.get(round(float(v), 6), -1) for v in t_trk])
    if (gi < 0).any():
        raise SystemExit(f"chained: {(gi < 0).sum()} tracked frames unmatched to GT by t")
    gt_m, occ_m = gt_nodes[gi], gt_occ[gi]
    d_fwd = np.linalg.norm(nodes[:, 0] - gt_m[:, 0], axis=1) + np.linalg.norm(nodes[:, -1] - gt_m[:, -1], axis=1)
    d_rev = np.linalg.norm(nodes[:, 0] - gt_m[:, -1], axis=1) + np.linalg.norm(nodes[:, -1] - gt_m[:, 0], axis=1)
    flipped = d_rev.mean() < d_fwd.mean()
    if flipped:
        nodes = nodes[:, ::-1, :].copy()
    print(f"[stitch] chained: {len(t_trk)} frames  t={t_trk.min():.2f}-{t_trk.max():.2f}s  "
          f"node order {'REVERSED->flipped' if flipped else 'forward'}")
    order = np.argsort(t_trk)
    return _rows_from(t_trk[order], nodes[order], gt_m[order], occ_m[order], _pose_at)


def _assemble_bridged():
    """The transit-gap-bridged chained run: combined_tracked_trajectory_bridged.npz
    (t, nodes, source) vs ground_truth_full.npz (the continuous ~30 Hz replay
    grid -- bridge frames sit between the per-pose GT ticks, so match nearest-t).
    Bridged frames are flagged so the viz styles them distinctly."""
    trk_npz = np.load(TRACK_ROOT / "combined_tracked_trajectory_bridged.npz")
    t_trk, nodes, source = trk_npz["t"], trk_npz["nodes"], trk_npz["source"]
    gf = np.load(CALIB_ROOT / "ground_truth_full.npz", allow_pickle=True)
    gt_t, gt_nodes, gt_occ = gf["t"], gf["gt_nodes"], gf["occlusion"]
    o = np.argsort(gt_t); gt_t, gt_nodes, gt_occ = gt_t[o], gt_nodes[o], gt_occ[o]
    pos = np.clip(np.searchsorted(gt_t, t_trk), 1, len(gt_t) - 1)
    gi = np.where((t_trk - gt_t[pos - 1]) <= (gt_t[pos] - t_trk), pos - 1, pos)
    dt = np.abs(gt_t[gi] - t_trk)
    if dt.max() > 0.02:
        raise SystemExit(f"bridged: {int((dt > 0.02).sum())} frames >20 ms from GT (max {dt.max()*1000:.1f} ms)")
    gt_m, occ_m = gt_nodes[gi], gt_occ[gi]
    d_fwd = np.linalg.norm(nodes[:, 0] - gt_m[:, 0], axis=1) + np.linalg.norm(nodes[:, -1] - gt_m[:, -1], axis=1)
    d_rev = np.linalg.norm(nodes[:, 0] - gt_m[:, -1], axis=1) + np.linalg.norm(nodes[:, -1] - gt_m[:, 0], axis=1)
    flipped = d_rev.mean() < d_fwd.mean()
    if flipped:
        nodes = nodes[:, ::-1, :].copy()
    order = np.argsort(t_trk)
    print(f"[stitch] bridged: {len(t_trk)} frames ({int((source == 1).sum())} bridged)  "
          f"t={t_trk.min():.2f}-{t_trk.max():.2f}s  node order "
          f"{'REVERSED->flipped' if flipped else 'forward'}  max|dt|={dt.max()*1000:.1f} ms")
    return _rows_from(t_trk[order], nodes[order], gt_m[order], occ_m[order], _pose_at,
                      bridged=source[order])


def _bounds(rows):
    """Per-axis tight bounds + 5% margin. True proportions are preserved by
    setting the 3D box aspect to (hi-lo), not by forcing a cube -- avoids
    the large empty vertical band from the cable's small z extent."""
    allp = np.concatenate([np.vstack([r["trk"], r["gt"]]) for r in rows])
    lo, hi = allp.min(0), allp.max(0)
    m = (hi - lo) * 0.05 + 0.01
    return lo - m, hi + m


def _connector_xyz(trk, gt):
    x, y, z = [], [], []
    for i in range(len(trk)):
        x += [trk[i, 0], gt[i, 0], None]
        y += [trk[i, 1], gt[i, 1], None]
        z += [trk[i, 2], gt[i, 2], None]
    return x, y, z


def _resample_realtime(rows, fps: float, playback_seconds: float, gap_thresh: float = 0.75):
    """Map the (unevenly time-spaced) tracked frames onto a uniform grid so
    playback speed is constant *relative to sim time* -- sparse stretches
    (e.g. pose C, ~11 Hz effective) linger instead of whipping past, and
    the whole covered span plays in `playback_seconds`. Returns
    [(row, is_gap, t_grid), ...] and the per-frame duration in ms.
    A grid point whose nearest real tracked frame is >gap_thresh s away
    (the two camera-transit windows) is flagged is_gap -> shown as the
    last real cable pose, held, labelled 'camera repositioning'."""
    t = np.array([r["t"] for r in rows])
    n = max(2, int(round((t.max() - t.min()) * fps)))
    grid = np.linspace(t.min(), t.max(), n)
    out = []
    for tg in grid:
        k = int(np.argmin(np.abs(t - tg)))
        out.append((rows[k], abs(t[k] - tg) > gap_thresh, float(tg)))
    return out, round(1000.0 * playback_seconds / n)


def build_html(rows, out_path: Path, lo, hi, fps: float, playback_seconds: float,
               play_frames: int) -> None:
    import plotly.graph_objects as go

    span = np.asarray(hi) - np.asarray(lo)
    ar = span / span.max()
    disp, _ = _resample_realtime(rows, fps, playback_seconds)
    # The slider scrubs all `disp` frames, but a full 3D redraw costs
    # ~100-500ms/frame in-browser, so the PLAY button can't hit a real-time
    # cadence over ~900 frames. Have play step through only an evenly-
    # spaced subset of `play_frames` frames; its per-frame duration is then
    # playback_seconds/play_frames, which is large enough to dominate the
    # redraw cost -> one playthrough takes ~playback_seconds regardless of
    # machine speed. Raise --play-frames on a fast machine for smoother
    # play; the scrubber resolution is unaffected either way.
    play_idx = np.unique(np.linspace(0, len(disp) - 1, min(play_frames, len(disp))).round().astype(int))
    play_names = [str(int(i)) for i in play_idx]
    play_ms = max(1, round(1000.0 * playback_seconds / len(play_names)))
    print(f"[stitch] html: {len(disp)} scrub frames; play uses {len(play_names)} @ {play_ms} ms "
          f"(~{len(play_names)*play_ms/1000:.0f}s/playthrough); {sum(g for _, g, _ in disp)} gap frames")

    def frame_traces(r):
        occ_open = r["occ"] == 1
        sym = np.where(occ_open, "circle-open", "circle")
        gt, trk = r["gt"], r["trk"]
        cx, cy, cz = _connector_xyz(trk, gt)
        bridged = r.get("bridged", False)
        tcol = BRIDGE_COLOR if bridged else TRK_COLOR
        return [
            go.Scatter3d(x=cx, y=cy, z=cz, mode="lines",
                         line=dict(color=CONN_COLOR, width=2), name="tracked-GT link", showlegend=False),
            go.Scatter3d(x=gt[:, 0], y=gt[:, 1], z=gt[:, 2], mode="lines+markers",
                         line=dict(color=GT_COLOR, width=6),
                         marker=dict(size=4, color=GT_COLOR, symbol=list(sym),
                                     line=dict(color=GT_COLOR, width=2)),
                         name="ground truth"),
            go.Scatter3d(x=trk[:, 0], y=trk[:, 1], z=trk[:, 2], mode="lines+markers",
                         line=dict(color=tcol, width=6),
                         marker=dict(size=4, color=tcol, symbol="diamond" if bridged else "circle"),
                         name="tracked (bridged)" if bridged else "tracked"),
        ]

    def title_for(r, is_gap, tg):
        if is_gap:
            return f"t = {tg:.2f} s   — camera repositioning (no tracking) —"
        tag = "   — kinematic bridge (synthetic) —" if r.get("bridged", False) else ""
        return (f"t = {tg:.2f} s   pose {r['pose']}   "
                f"point-to-curve RMSE = {r['rmse']:.1f} mm{tag}")

    fig = go.Figure(
        data=frame_traces(disp[0][0]),
        frames=[go.Frame(data=frame_traces(r), name=f"{i}",
                         layout=go.Layout(title_text=title_for(r, is_gap, tg)))
                for i, (r, is_gap, tg) in enumerate(disp)],
    )
    steps = [dict(method="animate", label=f"{tg:.1f}",
                  args=[[f"{i}"], dict(mode="immediate", frame=dict(duration=0, redraw=True),
                                       transition=dict(duration=0))])
             for i, (r, is_gap, tg) in enumerate(disp)]
    fig.update_layout(
        title_text=title_for(*disp[0]),
        scene=dict(
            # autorange=False + a fixed manual aspectratio locks BOTH the
            # data range and the box shape for the whole animation --
            # without this, plotly re-autoranges the 3D scene on every
            # frame and the volume visibly jumps around.
            xaxis=dict(range=[lo[0], hi[0]], autorange=False, title="x (m)"),
            yaxis=dict(range=[lo[1], hi[1]], autorange=False, title="y (m)"),
            zaxis=dict(range=[lo[2], hi[2]], autorange=False, title="z (m)"),
            aspectmode="manual",
            aspectratio=dict(x=float(ar[0]), y=float(ar[1]), z=float(ar[2])),
        ),
        uirevision="stitch3d",  # keep the user's rotate/zoom across frame changes
        updatemenus=[dict(type="buttons", showactive=False, x=0.05, y=0.05, xanchor="left",
                          buttons=[
                              dict(label="play", method="animate",
                                   args=[play_names, dict(mode="immediate",
                                                         frame=dict(duration=play_ms, redraw=True),
                                                         transition=dict(duration=0))]),
                              dict(label="pause", method="animate",
                                   args=[[None], dict(mode="immediate", frame=dict(duration=0, redraw=False))])])],
        sliders=[dict(active=0, x=0.12, len=0.85, currentvalue=dict(prefix="t = ", suffix=" s"),
                      pad=dict(t=40), steps=steps)],
        legend=dict(x=0.02, y=0.98),
        margin=dict(l=0, r=0, t=40, b=0),
    )
    fig.write_html(out_path, include_plotlyjs="cdn", auto_play=False)
    print(f"[stitch] wrote {out_path}  ({out_path.stat().st_size/1e6:.1f} MB, {len(disp)} frames)")


def build_gif(rows, out_path: Path, lo, hi, n_frames: int) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import PillowWriter

    sel = np.unique(np.linspace(0, len(rows) - 1, min(n_frames, len(rows))).round().astype(int))
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    writer = PillowWriter(fps=15)
    with writer.saving(fig, str(out_path), dpi=90):
        for k in sel:
            r = rows[int(k)]
            ax.clear()
            gt, trk = r["gt"], r["trk"]
            for i in range(20):
                ax.plot([trk[i, 0], gt[i, 0]], [trk[i, 1], gt[i, 1]], [trk[i, 2], gt[i, 2]],
                        color="0.6", lw=0.7, alpha=0.6)
            ax.plot(gt[:, 0], gt[:, 1], gt[:, 2], color=GT_COLOR, lw=2, label="ground truth")
            vis = r["occ"] != 1
            ax.scatter(gt[vis, 0], gt[vis, 1], gt[vis, 2], c=GT_COLOR, s=16)
            ax.scatter(gt[~vis, 0], gt[~vis, 1], gt[~vis, 2], facecolors="none", edgecolors=GT_COLOR, s=22)
            bridged = r.get("bridged", False)
            tcol = BRIDGE_COLOR if bridged else TRK_COLOR
            ax.plot(trk[:, 0], trk[:, 1], trk[:, 2], color=tcol, lw=2,
                    label="tracked (bridged)" if bridged else "tracked")
            ax.scatter(trk[:, 0], trk[:, 1], trk[:, 2], c=tcol, s=14)
            ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1]); ax.set_zlim(lo[2], hi[2])
            ax.set_box_aspect(tuple(hi - lo))
            ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)"); ax.set_zlabel("z (m)")
            ax.view_init(elev=22, azim=-60)
            tag = "   [kinematic bridge]" if bridged else ""
            ax.set_title(f"t = {r['t']:.2f} s   pose {r['pose']}   "
                         f"point-to-curve RMSE = {r['rmse']:.1f} mm{tag}")
            ax.legend(loc="upper left", fontsize=8)
            writer.grab_frame()
    plt.close(fig)
    print(f"[stitch] wrote {out_path}  ({out_path.stat().st_size/1e6:.1f} MB, {len(sel)} frames)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gif-frames", type=int, default=280)
    ap.add_argument("--playback-seconds", type=float, default=50.0,
                    help="wall-clock length of one full HTML playthrough (default 50)")
    ap.add_argument("--fps", type=float, default=20.0,
                    help="HTML scrubber frame density (uniform vs sim time)")
    ap.add_argument("--play-frames", type=int, default=75,
                    help="frames the HTML PLAY button steps through (default 90). "
                         "Lower if playback is still slower than --playback-seconds "
                         "(your browser's 3D redraw is the limit); raise for smoother play.")
    ap.add_argument("--chained", action="store_true",
                    help="use the A->B->C seed-handoff run (combined_tracked_trajectory.npz) "
                         "instead of stitching the 3 independent single-pose runs")
    ap.add_argument("--bridged", action="store_true",
                    help="use the transit-gap-bridged chained run "
                         "(combined_tracked_trajectory_bridged.npz vs ground_truth_full.npz); "
                         "bridge frames drawn in magenta / labelled 'kinematic bridge'")
    ap.add_argument("--no-html", action="store_true")
    ap.add_argument("--no-gif", action="store_true")
    args = ap.parse_args()

    if args.bridged:
        rows, suffix, kind = _assemble_bridged(), "_bridged", "bridged"
    elif args.chained:
        rows, suffix, kind = _assemble_chained(), "_chained", "chained"
    else:
        rows, suffix, kind = _assemble(), "", "stitched"
    lo, hi = _bounds(rows)
    print(f"[stitch] {len(rows)} {kind} frames, t={rows[0]['t']:.2f}-{rows[-1]['t']:.2f}s")
    if not args.no_html:
        build_html(rows, TRACK_ROOT / f"stitch_3d_gt_vs_tracked{suffix}.html", lo, hi,
                   args.fps, args.playback_seconds, args.play_frames)
    if not args.no_gif:
        build_gif(rows, TRACK_ROOT / f"stitch_3d_gt_vs_tracked{suffix}.gif", lo, hi, args.gif_frames)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

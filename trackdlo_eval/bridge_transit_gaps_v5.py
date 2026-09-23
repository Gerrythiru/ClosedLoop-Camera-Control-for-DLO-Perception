"""Regime 2 / B2: kinematic transit-gap bridge.

The chained V5 trajectory (results/r1_dlo_tracking_v5/combined_tracked_trajectory.npz)
has no frames during the two camera transits (A->B ~t 25-29 s, B->C ~t 34-37 s):
the wrist camera is moving, nothing is captured, so the tracker emits nothing and
the cable shape teleports at each pose boundary.

This fills those holes. During a blackout the only physically-knowable quantity is
the two grasped ends -- node 0 is welded to r2's gripper, node 19 to r3's -- and a
robot always knows its own gripper pose (joint-encoder FK). B1 dumped that signal
to bridge_kinematics.npz (gripper body poses + a one-time grasp offset). For each
missing timestamp this script:

  * nodes 0 / 19   <- the tracker's LAST real endpoint, plus the gripper's FK
                      displacement through the gap, plus the small end-to-end
                      residual (post-gap tracked endpoint minus FK-propagated
                      endpoint) eased in with a smoothstep. This follows the real
                      gripper path but pins both seams to the tracker, so there is
                      no jump entering or leaving the bridge. The ~30 mm static
                      offset between the tracker's end node and the physical grasp
                      point is absorbed into the pre-gap anchor, not propagated.
  * nodes 1..18    <- smoothstep morph from the last pre-gap tracked shape to the
                      first post-gap tracked shape              (a guess; no observation)
  * a short taper near each end blends the morph toward the bridged endpoint so the
    interior doesn't kink where it meets node 0 / 19.

Output combined_tracked_trajectory_bridged.npz adds a `source` array
(0 = tracked, 1 = bridged) so evaluation / rendering can exclude the synthetic
frames. NO cable ground truth is read here.

Usage:
    venv/bin/python3 trackdlo_eval/bridge_transit_gaps.py
    venv/bin/python3 trackdlo_eval/bridge_transit_gaps.py --tracked <path> --gap-thresh 1.0 --fps 30
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRACK_ROOT = PROJECT_ROOT / "results" / "r1_dlo_tracking_v5"
CALIB_ROOT = PROJECT_ROOT / "results" / "r1_eye_in_hand_calib_v5"

# Interior nodes within this many indices of an end are blended toward that
# end's FK'd position (linear taper, weight 1 at the end -> 0 at TAPER).
# V5: 2 (was 3) -- an 11-node cable is ~half as long, so a 3-node taper
# covered too large a fraction of it.
TAPER = 2


def _quat2mat(q: np.ndarray) -> np.ndarray:
    """MuJoCo wxyz unit quaternion -> 3x3 rotation."""
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w),     2 * (x * z + y * w)],
        [2 * (x * y + z * w),     1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w),     2 * (y * z + x * w),     1 - 2 * (x * x + y * y)],
    ])


def _endpoints_at(bk, tb: float) -> tuple[np.ndarray, np.ndarray]:
    """(node 0, node 19) world position at time tb, reconstructed from the
    nearest recorded gripper pose (x) the one-time grasp offset."""
    k = int(np.argmin(np.abs(bk["t"] - tb)))
    ep0 = bk["r2_grip_pos"][k] + _quat2mat(bk["r2_grip_quat"][k]) @ bk["grasp_offset_0"]
    ep19 = bk["r3_grip_pos"][k] + _quat2mat(bk["r3_grip_quat"][k]) @ bk["grasp_offset_19"]
    return ep0, ep19


def _bridge_gap(pre, post, t0, t1, bk, dt):
    """Frames strictly inside (t0, t1) at spacing dt. pre/post are the last
    pre-gap and first post-gap tracked frames (N, 3), N = cable node count.

    Endpoints: b(x) = pre_end + (fk(x) - fk(t0)) + s(x) * resid, where
    resid = post_end - (pre_end + fk(t1) - fk(t0)). s is a smoothstep 0->1.
    s=0 at t0 -> b = pre_end (no jump in); s=1 at t1 -> b = post_end (no jump
    out); between, b follows the gripper's FK displacement with the small
    residual eased in. Returns (times, frames, resid0, resid19) -- the
    residual norms are reported as a diagnostic."""
    fk0_a, fk19_a = _endpoints_at(bk, t0)
    fk0_b, fk19_b = _endpoints_at(bk, t1)
    resid0 = post[0] - (pre[0] + fk0_b - fk0_a)
    resid19 = post[-1] - (pre[-1] + fk19_b - fk19_a)

    times = np.arange(t0 + dt, t1 - 1e-9, dt)
    out = []
    for x in times:
        s = (x - t0) / (t1 - t0)
        s = s * s * (3.0 - 2.0 * s)                 # smoothstep
        f = ((1.0 - s) * pre + s * post).copy()     # interior morph
        fk0_x, fk19_x = _endpoints_at(bk, x)
        b0 = pre[0] + (fk0_x - fk0_a) + s * resid0
        b19 = pre[-1] + (fk19_x - fk19_a) + s * resid19
        f[0], f[-1] = b0, b19
        for j in range(1, TAPER + 1):               # soften the kink at each end
            w = (TAPER + 1 - j) / (TAPER + 1)
            f[j] = (1.0 - w) * f[j] + w * b0
            f[-1 - j] = (1.0 - w) * f[-1 - j] + w * b19
        out.append(f)
    return times, np.array(out).reshape(-1, pre.shape[0], 3), float(np.linalg.norm(resid0)), float(np.linalg.norm(resid19))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tracked", type=Path, default=TRACK_ROOT / "combined_tracked_trajectory.npz")
    ap.add_argument("--bridge-kinematics", type=Path, default=CALIB_ROOT / "bridge_kinematics.npz")
    ap.add_argument("--out", type=Path, default=TRACK_ROOT / "combined_tracked_trajectory_bridged.npz")
    ap.add_argument("--gap-thresh", type=float, default=1.0, help="s; tracked dt above this is a transit gap")
    ap.add_argument("--fps", type=float, default=30.0, help="bridge frame rate")
    args = ap.parse_args()

    for p in (args.tracked, args.bridge_kinematics):
        if not p.exists():
            raise SystemExit(f"missing {p}")

    trk = np.load(args.tracked)
    t, nodes = trk["t"].astype(np.float64).copy(), trk["nodes"].astype(np.float64).copy()
    if nodes.ndim != 3 or nodes.shape[2] != 3 or nodes.shape[1] < 3:
        raise SystemExit(f"expected (T, N>=3, 3) nodes, got {nodes.shape}")
    order = np.argsort(t, kind="stable")
    t, nodes = t[order], nodes[order]

    bk = np.load(args.bridge_kinematics)
    if not np.isfinite(bk["grasp_offset_0"]).all() or not np.isfinite(bk["grasp_offset_19"]).all():
        raise SystemExit("bridge_kinematics.npz has no settled grasp offset -- re-run dump_v5_ground_truth.py --full")
    t_offset = float(bk["t_offset"])

    gaps = np.where(np.diff(t) > args.gap_thresh)[0]
    print(f"[bridge] tracked: {len(t)} frames, t {t[0]:.2f}..{t[-1]:.2f}s")
    print(f"[bridge] gaps (dt > {args.gap_thresh}s): {len(gaps)}  ->  "
          + ", ".join(f"{t[i]:.2f}->{t[i+1]:.2f}s ({t[i+1]-t[i]:.2f}s)" for i in gaps))
    if len(gaps) == 0:
        raise SystemExit("no transit gaps found -- nothing to bridge")

    dt = 1.0 / args.fps
    bt_all, bn_all = [], []
    for i in gaps:
        t0, t1 = float(t[i]), float(t[i + 1])
        pre, post = nodes[i], nodes[i + 1]
        if t0 < t_offset:
            print(f"[bridge] WARNING: gap starts t={t0:.2f}s < grasp-offset instant "
                  f"{t_offset:.2f}s -- endpoint reconstruction may be unreliable")
        # node-order sanity: tracked node 0 must track the r2 endpoint, node 19 the r3 endpoint
        ep0_0, ep19_0 = _endpoints_at(bk, t0)
        d_fwd = np.linalg.norm(pre[0] - ep0_0) + np.linalg.norm(pre[-1] - ep19_0)
        d_rev = np.linalg.norm(pre[0] - ep19_0) + np.linalg.norm(pre[-1] - ep0_0)
        if d_fwd >= d_rev:
            raise SystemExit(f"gap at t={t0:.2f}s: tracked node order disagrees with the grippers "
                             f"(fwd {d_fwd*1000:.0f}mm vs rev {d_rev*1000:.0f}mm) -- fix the trajectory first")
        bt, bn, r0, r19 = _bridge_gap(pre, post, t0, t1, bk, dt)
        # seam continuity: mean per-node gap between the last real frame and the
        # first bridged frame (step-in), and the last bridged and first real
        # frame after the gap (step-out). Should now be ~one interpolation step.
        j_start = np.linalg.norm(bn[0] - pre, axis=1).mean()
        j_end = np.linalg.norm(bn[-1] - post, axis=1).mean()
        print(f"[bridge] gap t={t0:.2f}->{t1:.2f}s: {len(bt)} frames  "
              f"fk-vs-tracker offset {d_fwd*1000:.1f}mm  end-residual {r0*1000:.1f}/{r19*1000:.1f}mm  "
              f"step-in {j_start*1000:.1f}mm  step-out {j_end*1000:.1f}mm")
        bt_all.append(bt)
        bn_all.append(bn)

    bt_all = np.concatenate(bt_all)
    bn_all = np.concatenate(bn_all, axis=0)
    all_t = np.concatenate([t, bt_all])
    all_n = np.concatenate([nodes, bn_all], axis=0)
    src = np.concatenate([np.zeros(len(t), dtype=np.int8), np.ones(len(bt_all), dtype=np.int8)])
    order = np.argsort(all_t, kind="stable")
    all_t, all_n, src = all_t[order], all_n[order], src[order]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, t=all_t, nodes=all_n, source=src)
    print(f"[bridge] wrote {args.out}")
    print(f"[bridge]   {len(all_t)} frames ({int(src.sum())} bridged), "
          f"max dt {np.diff(all_t).max():.3f}s (was {np.diff(t).max():.3f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

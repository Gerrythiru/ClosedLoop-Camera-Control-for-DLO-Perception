"""Non-destructive comparison: how much would classify_risk.py's live
output change if its `occlusion` input came from FK-only reconstruction
(validate_fk_occlusion.py, tracked-input variant -- the only variant a
real deployment could ever have, since it never has the true position)
instead of sim ground truth?

Reuses both scripts' machinery directly rather than duplicating it -- only
the short final classification elif-chain from classify_risk.py's main()
is duplicated here (see _classify below), since that logic isn't factored
into an importable function there and this is a one-off diagnostic, not a
permanent pipeline stage. If classify_risk.py's classification logic
changes again, check this copy against it.

Neither build_predictor_dataset.py nor classify_risk.py is modified --
this only reads their existing outputs plus a new fk_occlusion
computation (reusing validate_fk_occlusion.py's shadow-model machinery).

Scope: V5 'default' run only, matching validate_fk_occlusion.py.

Usage: venv/bin/python3 compare_fk_occlusion_classification.py
Output: results/r1_dlo_tracking_v5/fk_occlusion_classification_compare.npz
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import mujoco
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "trajectories"))

from build_predictor_dataset import _run_paths, _load_combined_gt, _load_combined_captures, _nearest_idx  # noqa: E402
from dump_v5_ground_truth import (  # noqa: E402
    QPOS_A, QPOS_B, QPOS_C, set_solved_waypoints, label_node_occlusion_wrist, _finger_qpos_ids,
)
from r1_camera_capture_v5 import CAPTURE_WIDTH, CAPTURE_HEIGHT  # noqa: E402
from r1_camera_ik_v5 import CameraArm  # noqa: E402
from validate_fk_occlusion import (  # noqa: E402
    _build_shadow_model, _set_frame, _position_cable_geoms, _cable_geom_ids,
    _arm_qpos_ids, _load_combined_finger_qpos, MAX_CAPTURE_GAP_S,
)
from classify_risk import (  # noqa: E402
    _trailing_median, _trailing_rate, _require_persistence, _transition_guard,
    BASELINE_WINDOW, SUPPORT_DROP_FRAC, MIN_BASELINE_SUPPORT, STRESS_RATE_THR,
    SUPPORT_COLLAPSE_MIN_CONSECUTIVE, GRAZING_ANGLE_THR_DEG, JUMP_THR_M, VISIBLE,
)


def _classify(t, node_support, converged, occlusion, material_tag, paint_markers,
              tangent_angle, inter_frame_jump, pipeline):
    """Duplicated from classify_risk.py's main() classification loop (see
    module docstring for why) -- everything ELSE (trailing stats,
    persistence, transition guard, constants) is imported, not duplicated."""
    T, N = node_support.shape
    dnc_rate = _trailing_rate(~converged, BASELINE_WINDOW)
    solve_stress = dnc_rate > STRESS_RATE_THR
    transition_guard = _transition_guard(t, pipeline)

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

    shape_stress = np.nan_to_num(inter_frame_jump, nan=0.0) > JUMP_THR_M

    risk_class = np.full((T, N), "healthy", dtype=object)
    for k in range(T):
        for i in range(N):
            is_marker = material_tag[i] != 0
            if is_marker and not paint_markers:
                risk_class[k, i] = "blind_spot"
            elif occlusion[k, i] != VISIBLE and support_collapsed[k, i]:
                risk_class[k, i] = "geometric_occlusion"
            elif solve_stress[k] and support_collapsed[k, i]:
                risk_class[k, i] = "joint_solve_degeneracy"
            elif support_collapsed[k, i] and is_marker and paint_markers:
                risk_class[k, i] = "blind_spot_anomaly"
            elif support_collapsed[k, i] and tangent_angle[k, i] < GRAZING_ANGLE_THR_DEG:
                risk_class[k, i] = "foreshortening"
            elif support_collapsed[k, i]:
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

    return risk_class, mitigation, support_collapsed


def main() -> int:
    paths = _run_paths("v5", None)
    run_metadata = json.load(open(paths["meta_path"]))
    poses = run_metadata["poses_run"]

    feat = np.load(paths["out_path"], allow_pickle=True)
    risk_saved = np.load(paths["track_root"] / "risk_classes_default.npz", allow_pickle=True)

    t = feat["t"]
    node_support = feat["node_support"]
    converged = feat["converged"].astype(bool)
    sim_occlusion = feat["occlusion"]
    material_tag = feat["material_tag"]
    paint_markers = bool(feat["paint_markers"])
    tangent_angle = feat["tangent_view_angle_deg"]
    inter_frame_jump = feat["inter_frame_jump_m"]
    T, N = node_support.shape
    print(f"[compare] {T} frames, {N} nodes", flush=True)

    # --- sanity check: re-deriving from predictor_features with sim occlusion
    # must exactly reproduce the already-saved risk_classes_default.npz ---
    risk_class_sim, mitigation_sim, support_collapsed = _classify(
        t, node_support, converged, sim_occlusion, material_tag, paint_markers,
        tangent_angle, inter_frame_jump, "v5")
    n_mismatch = int((risk_class_sim != risk_saved["risk_class"]).sum())
    print(f"[compare] sanity check: {n_mismatch} mismatches vs risk_classes_default.npz "
          f"(must be 0 to trust the rest of this script)", flush=True)
    if n_mismatch:
        print("[compare] ABORTING -- classification re-derivation doesn't match the live "
              "pipeline's own output; do not trust the fk-occlusion comparison below.")
        return 1

    # --- compute fk_occlusion (tracked-input variant) ---
    trk = np.load(paths["tracked_path"])
    assert np.array_equal(trk["t"], t), "tracked trajectory t grid must match predictor_features t"
    tracked_nodes = trk["nodes"]

    gt = _load_combined_gt(paths["calib_root"], poses)
    caps = _load_combined_captures(paths["calib_root"], poses)
    cap_t = np.array([c["t"] for c in caps])
    finger_qpos = _load_combined_finger_qpos(paths["calib_root"], poses)
    if finger_qpos is None:
        print("[compare] WARNING: ground_truth.npz predates finger qpos -- using compiled default", flush=True)

    gi = _nearest_idx(t, gt["t"])
    r2_qpos = gt["r2_qpos"][gi]
    r3_qpos = gt["r3_qpos"][gi]
    if finger_qpos is not None:
        r2_finger_qpos = finger_qpos[0][gi]
        r3_finger_qpos = finger_qpos[1][gi]
    ci = _nearest_idx(t, cap_t)
    cam_pos_real = np.array([caps[i]["cam_pos"] for i in ci])
    cam_quat_real = np.array([caps[i]["cam_quat"] for i in ci])
    has_real_cam = np.abs(cap_t[ci] - t) <= MAX_CAPTURE_GAP_S

    model, data, skip_mask, cam_id = _build_shadow_model()
    set_solved_waypoints(QPOS_A, QPOS_B, QPOS_C)
    rail_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "r1_rail_x")
    rail_qadr = model.jnt_qposadr[rail_jid]
    arm = CameraArm(model, data)
    r2_ids = _arm_qpos_ids(model, "r2")
    r3_ids = _arm_qpos_ids(model, "r3")
    finger_ids = {"r2": _finger_qpos_ids(model, "r2"), "r3": _finger_qpos_ids(model, "r3")} \
        if finger_qpos is not None else None
    cable_geom_ids = _cable_geom_ids(model, N)

    fk_occlusion = np.full((T, N), -1, dtype=np.int32)
    print(f"[compare] computing fk_occlusion (tracked-input) over {T} frames...", flush=True)
    for k in range(T):
        _set_frame(model, data, arm, rail_qadr, r2_ids, r3_ids, float(t[k]), r2_qpos[k], r3_qpos[k],
                   cam_id, cam_pos_real[k] if has_real_cam[k] else None,
                   cam_quat_real[k] if has_real_cam[k] else None,
                   finger_ids, r2_finger_qpos[k] if finger_qpos is not None else None,
                   r3_finger_qpos[k] if finger_qpos is not None else None)
        _position_cable_geoms(data, cable_geom_ids, tracked_nodes[k])
        labels, _ = label_node_occlusion_wrist(model, data, cam_id, tracked_nodes[k], skip_mask,
                                                CAPTURE_WIDTH, CAPTURE_HEIGHT)
        fk_occlusion[k] = labels
        if k % 200 == 0:
            print(f"[compare]   {k}/{T}", flush=True)

    risk_class_fk, mitigation_fk, _ = _classify(
        t, node_support, converged, fk_occlusion, material_tag, paint_markers,
        tangent_angle, inter_frame_jump, "v5")

    # --- diff + report ---
    match = risk_class_sim == risk_class_fk
    print(f"\n[compare] overall risk_class agreement (sim-occlusion vs fk-occlusion classification): "
          f"{match.mean():.1%}", flush=True)
    print(f"[compare] per-node agreement:", flush=True)
    for i in range(N):
        print(f"    node{i:2d}: {match[:, i].mean():.1%}", flush=True)

    trans = Counter()
    for k in range(T):
        for i in range(N):
            if risk_class_sim[k, i] != risk_class_fk[k, i]:
                trans[(risk_class_sim[k, i], risk_class_fk[k, i])] += 1
    print(f"\n[compare] disagreement transitions (sim -> fk), most common first:", flush=True)
    for (a, b), n in trans.most_common(20):
        print(f"    {a:28s} -> {b:28s}: {n}", flush=True)

    nbv_sim = mitigation_sim == "invoke_nbv"
    nbv_fk = mitigation_fk == "invoke_nbv"
    gained = int((~nbv_sim & nbv_fk).sum())
    lost = int((nbv_sim & ~nbv_fk).sum())
    print(f"\n[compare] invoke_nbv mitigation: {gained} cells GAINED it, {lost} cells LOST it "
          f"(out of {int(nbv_sim.sum())} sim cells that had it)", flush=True)

    occ_disagree = sim_occlusion != fk_occlusion
    consulted = support_collapsed
    print(f"\n[compare] raw per-frame occlusion label disagreement: {occ_disagree.mean():.1%} of all cells", flush=True)
    if consulted.any():
        print(f"[compare]   on support_collapsed=True cells (occlusion actually consulted): "
              f"{occ_disagree[consulted].mean():.1%} disagree", flush=True)
    print(f"[compare]   on support_collapsed=False cells (occlusion never read): "
          f"{occ_disagree[~consulted].mean():.1%} disagree", flush=True)

    out_path = paths["track_root"] / "fk_occlusion_classification_compare.npz"
    np.savez(out_path, t=t, risk_class_sim=risk_class_sim, risk_class_fk=risk_class_fk,
             mitigation_sim=mitigation_sim, mitigation_fk=mitigation_fk,
             sim_occlusion=sim_occlusion, fk_occlusion=fk_occlusion,
             support_collapsed=support_collapsed)
    print(f"\n[compare] wrote {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

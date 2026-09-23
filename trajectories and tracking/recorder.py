"""Recording infrastructure for E1/E2 (Mujoco Plan 0, Parts E1 storage / E2
input). Captures per-frame RGB, depth, cable mask, ground truth, and
occlusion labels at 30 Hz during a scripted trajectory, and saves the whole
run to trajectories/recorded_data/ for the TrackDLO bridge (E2) and
this script's own verification artifact (E1) to consume.

Reuses this repo's existing B2/B4/C1/C3 implementations (rendering/,
ground_truth/) rather than re-deriving them -- this module is purely the
per-frame capture/storage glue, matching Plan 0's own E2 description of
"store per-frame data in a structured format for offline analysis."

Multi-camera: originally single-camera (box_focus only, Plan 0's E1 "B0
static camera baseline"). Generalized for the dlo layout/run_dlo.py, which
records box_focus AND r1's active eye-in-hand D435i feed side by side (see
HANDOFF.md's dlo layout section) -- ground truth (camera-independent) is
computed once per frame; mask and occlusion labels are computed per camera,
since occlusion genuinely differs by viewpoint.
"""

from __future__ import annotations

import sys
from pathlib import Path

import mujoco
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from ground_truth.cable_gt import get_cable_ground_truth  # noqa: E402
from ground_truth.occlusion_labels import label_node_occlusion  # noqa: E402
from rendering.cable_mask import get_cable_mask  # noqa: E402
from rendering.render_rgbd import render_rgbd  # noqa: E402

DEFAULT_CAMERA = "box_focus"
OUTPUT_DIR = PROJECT_ROOT / "trajectories" / "recorded_data"


class TrajectoryRecorder:
    def __init__(
        self,
        renderer: mujoco.Renderer,
        cameras: dict[str, str] | None = None,
        cable_color: str = "red",
        cable_prefix: str = "red_cable",
    ) -> None:
        """`cameras` maps a storage-key prefix -> MuJoCo camera name, e.g.
        {"box_focus": "box_focus", "r1_gaze": "r1_d435i_rgb"}. Defaults to
        the original single fixed-camera behavior ({"box_focus": "box_focus"})
        for backward compatibility with anything still constructing this
        with no `cameras` argument.
        """
        self.renderer = renderer
        self.cameras = cameras or {DEFAULT_CAMERA: DEFAULT_CAMERA}
        self.cable_color = cable_color
        self.cable_prefix = cable_prefix
        self.frames: list[dict[str, np.ndarray | float]] = []

    def record(self, model: mujoco.MjModel, data: mujoco.MjData, t: float) -> None:
        """Matches ClipRoutePlanner/TwoArmClipRoutePlanner's run_sequence
        on_frame(model, data, t) hook.

        No group-2 (visual-only) geom filtering here, unlike verify_scene.py's
        checks: that filter exists so a D435i camera doesn't see its own
        mount/housing (HANDOFF.md), but this recorder's fixed external camera
        (box_focus) needs the arms visible (confirmed empirically: rendering
        `overview` with the filter shows only bare collision-cylinder stubs
        where the arms should be) -- and the r1 eye-in-hand feed is looking
        AT the clips/cable, not at r1's own housing, so it doesn't need the
        filter either.
        """
        gt = get_cable_ground_truth(model, data, self.cable_prefix)
        frame: dict[str, np.ndarray | float] = {"t": t, "ground_truth": gt}

        for cam_key, cam_name in self.cameras.items():
            rgb, depth = render_rgbd(model, data, self.renderer, cam_name)
            mask = get_cable_mask(rgb, self.cable_color)
            occlusion_labels, occluders = label_node_occlusion(
                model, data, cam_name, gt,
                width=self.renderer.width, height=self.renderer.height,
            )
            frame[f"{cam_key}_rgb"] = rgb
            frame[f"{cam_key}_depth"] = depth
            frame[f"{cam_key}_mask"] = mask
            frame[f"{cam_key}_occlusion_labels"] = occlusion_labels
            # occluder names are variable-length strings/None per node --
            # kept as an object array so np.savez can store them alongside
            # the fixed-shape numeric arrays above.
            frame[f"{cam_key}_occluders"] = np.array(occluders, dtype=object)

        self.frames.append(frame)

    def save(self, path: Path = OUTPUT_DIR / "trajectory.npz") -> Path:
        """Save all recorded frames as one .npz archive (Plan 0's E2 storage
        suggestion), stacking each field across frames so a downstream
        consumer can index by frame number.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        if not self.frames:
            raise ValueError("No frames recorded -- call record() during the trajectory first.")

        stacked: dict[str, np.ndarray] = {
            "t": np.array([f["t"] for f in self.frames], dtype=np.float64),
            "ground_truth": np.stack([f["ground_truth"] for f in self.frames]),
        }
        for cam_key in self.cameras:
            for field in ("rgb", "depth", "mask", "occlusion_labels", "occluders"):
                key = f"{cam_key}_{field}"
                stacked[key] = np.stack([f[key] for f in self.frames])

        np.savez_compressed(path, **stacked)
        print(f"[recorder] saved {len(self.frames)} frames ({list(self.cameras)}) to {path}")
        return path

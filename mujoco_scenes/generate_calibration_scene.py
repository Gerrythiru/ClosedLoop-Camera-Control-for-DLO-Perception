"""Minimal single-cable scene for Cosserat `youngs_modulus`/`nu` calibration
(see MULTIVIEW_TRACKING_SCOPING.md and the calibration/ package).

No robots, table, routing box, or fixture panel -- just gravity and one cable
whose root is rigidly clamped (no <freejoint>, so MuJoCo implicitly welds the
root body to worldbody at 0 DOF) instead of free, so it settles into a clean
cantilever droop uncontaminated by contact with anything else. Reuses
add_physical_cable() from generate_triple_scene_v2.py unchanged -- the joint
damping/armature values must match the deployed scene exactly, since that's
the actual behavior being calibrated against.

Same generated scene is used for both calibration tests (static settle and
transient release): both start from the same straight, clamped-root pose:
the runner script (calibration/mujoco_runner.py) decides whether to run to
convergence (static) or record every step from t=0 (transient).
"""

from __future__ import annotations

import sys
import xml.etree.ElementTree as ET
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from generate_triple_scene_v2 import (  # noqa: E402
    CABLE_RADIUS, CABLE_SEGMENTS, add_physical_cable, indent, project_root,
)

OUTPUT_NAME = "calibration_cable.xml"
CABLE_PREFIX = "cal"


def generate(
    output_name: str = OUTPUT_NAME,
    segments: int = CABLE_SEGMENTS,
    radius: float = CABLE_RADIUS,
) -> Path:
    root_dir = project_root()
    output_path = root_dir / output_name

    mjcf = ET.Element("mujoco", model="calibration_cable")
    ET.SubElement(mjcf, "compiler", angle="radian", autolimits="true")
    ET.SubElement(
        mjcf, "option",
        timestep="0.0005", integrator="implicitfast", gravity="0 0 -9.81",
        cone="elliptic", iterations="80", ls_iterations="20",
    )

    asset = ET.SubElement(mjcf, "asset")
    ET.SubElement(asset, "material", name="mat_cable_red", rgba="0.95 0.03 0.02 1")

    worldbody = ET.SubElement(mjcf, "worldbody")
    ET.SubElement(worldbody, "light", name="key_light", pos="-0.4 -0.8 2.5", dir="0.3 0.6 -1", diffuse="0.9 0.9 0.85")

    # Root placed high enough that the fully-drooped 20-segment (0.8 m) chain
    # never approaches z=0 -- there's no floor in this scene at all.
    add_physical_cable(
        worldbody, CABLE_PREFIX, "mat_cable_red", 0.0,
        segments=segments, radius=radius, start_pos=(0.0, 0.0, 0.5),
    )

    # Clamp the root: remove the <freejoint> add_physical_cable() added, so
    # the root body has zero joints -- MJCF implicitly welds a joint-less
    # body to its parent (worldbody), giving a true 0-DOF clamp. Simpler and
    # stiffer than keeping the freejoint plus a <weld> equality (which has
    # finite solref/solimp compliance and isn't needed here).
    root_body = worldbody.find(f"body[@name='{CABLE_PREFIX}_cable_00']")
    if root_body is None:
        raise ValueError(f"Expected root cable body '{CABLE_PREFIX}_cable_00' not found.")
    freejoint = root_body.find("freejoint")
    if freejoint is None:
        raise ValueError("Expected <freejoint> on the cable root body, found none to remove.")
    root_body.remove(freejoint)

    indent(mjcf)
    ET.ElementTree(mjcf).write(output_path, encoding="utf-8", xml_declaration=True)
    return output_path


if __name__ == "__main__":
    path = generate()
    print(f"[calibration] wrote {path}")

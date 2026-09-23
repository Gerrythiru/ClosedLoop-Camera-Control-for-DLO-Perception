from __future__ import annotations

import copy
import math
import sys
import xml.etree.ElementTree as ET
from pathlib import Path


MODEL_RELATIVE = Path("vendor") / "mujoco_menagerie" / "ufactory_lite6" / "lite6_gripper_wide.xml"
OUTPUT_NAME = "triple_lite6_cable_routing.xml"
DLO_OUTPUT_NAME = "triple_lite6_cable_routing_dlo_v3.xml"
CABLE_DIAMETER = 0.010
CABLE_RADIUS = CABLE_DIAMETER / 2.0
CABLE_SEGMENT_LENGTH = 0.04
CABLE_SEGMENTS = 20

# Routing box floor top surface, world z. box body at z=0.82 (add_routing_box),
# box_floor geom local pos z=-0.038, half-height 0.012 -> 0.82 - 0.038 + 0.012.
ROUTING_BOX_FLOOR_TOP_Z = 0.82 - 0.038 + 0.012

# Routing box guide rail top surface (tallest box geometry), world z. box body
# at z=0.82, guide geoms local pos z=0.005, half-height 0.045 -> 0.82+0.005+0.045.
ROUTING_BOX_GUIDE_TOP_Z = 0.82 + 0.005 + 0.045

# Routing box horizontal center, world x (box body pos, add_routing_box).
ROUTING_BOX_CENTER_X = 0.20

REFERENCE_ATTRS = {
    "body",
    "body1",
    "body2",
    "joint",
    "joint1",
    "joint2",
    "geom",
    "geom1",
    "geom2",
    "site",
    "site1",
    "site2",
    "tendon",
    "tendon1",
    "tendon2",
}


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def indent(elem: ET.Element) -> None:
    try:
        ET.indent(elem, space="  ")
    except AttributeError:
        pass


def prefix_token(value: str, prefix: str) -> str:
    if not value or value.startswith(prefix):
        return value
    return f"{prefix}{value}"


def prefix_robot_tree(elem: ET.Element, prefix: str) -> None:
    for node in elem.iter():
        if "name" in node.attrib:
            node.set("name", prefix_token(node.get("name", ""), prefix))
        for attr in REFERENCE_ATTRS:
            if attr in node.attrib:
                node.set(attr, prefix_token(node.get(attr, ""), prefix))


def clone_prefixed_children(source: ET.Element | None, prefix: str) -> list[ET.Element]:
    if source is None:
        return []
    cloned = [copy.deepcopy(child) for child in list(source)]
    for child in cloned:
        prefix_robot_tree(child, prefix)
    return cloned


def text_vec(values: tuple[float, ...]) -> str:
    return " ".join(f"{value:.6g}" for value in values)


def add_geom(parent: ET.Element, **attrs: str) -> ET.Element:
    return ET.SubElement(parent, "geom", {key: str(value) for key, value in attrs.items()})


def add_site(parent: ET.Element, **attrs: str) -> ET.Element:
    return ET.SubElement(parent, "site", {key: str(value) for key, value in attrs.items()})


def first_world_body(source_worldbody: ET.Element) -> ET.Element:
    bodies = [child for child in list(source_worldbody) if child.tag == "body"]
    if not bodies:
        raise ValueError("The official Lite 6 XML does not contain a body in <worldbody>.")
    return bodies[0]


def add_workstation(worldbody: ET.Element) -> None:
    add_geom(
        worldbody,
        name="workstation_60x30_top",
        type="box",
        pos="0 0 0.72",
        size="0.762 0.381 0.035",
        material="mat_worktop",
        contype="1",
        conaffinity="1",
    )
    add_geom(
        worldbody,
        name="workstation_left_leg",
        type="box",
        pos="-0.68 -0.30 0.35",
        size="0.035 0.035 0.35",
        material="mat_frame",
    )
    add_geom(
        worldbody,
        name="workstation_right_leg",
        type="box",
        pos="0.68 -0.30 0.35",
        size="0.035 0.035 0.35",
        material="mat_frame",
    )
    add_geom(
        worldbody,
        name="workstation_back_left_leg",
        type="box",
        pos="-0.68 0.30 0.35",
        size="0.035 0.035 0.35",
        material="mat_frame",
    )
    add_geom(
        worldbody,
        name="workstation_back_right_leg",
        type="box",
        pos="0.68 0.30 0.35",
        size="0.035 0.035 0.35",
        material="mat_frame",
    )


def add_routing_box(worldbody: ET.Element) -> None:
    box = ET.SubElement(worldbody, "body", name="grey_routing_box", pos="0.20 0 0.82")
    add_geom(box, name="box_floor", type="box", pos="0 0 -0.038", size="0.31 0.15 0.012", material="mat_box")
    # Front and back guides are split into 3 segments each to create two
    # full-height/depth extrude cuts: 110mm starting at 200mm and 450mm from
    # the left edge (local X = -0.31). Segment centers/half-lengths:
    #   A: center=-0.21, half=0.10   (0–200mm)
    #   gap 1: -0.11 to 0.00         (200–310mm)
    #   B: center=0.07,  half=0.07   (310–450mm)
    #   gap 2: 0.14 to 0.25          (450–560mm)
    #   C: center=0.265, half=0.015  (550–580mm) -- shrunk to half length,
    #     kept the -X edge fixed at 0.25 (world x=0.45), removed the +X half
    #     (world x=[0.48,0.51] is now the gap) -- remaining material spans
    #     world x=[0.45, 0.48], per explicit spec.
    for y, tag in ((-0.105, "front"), (0.105, "back")):
        add_geom(box, name=f"box_{tag}_guide_a", type="box", pos=f"-0.21 {y} 0.005", size="0.10 0.010 0.045", material="mat_box")
        add_geom(box, name=f"box_{tag}_guide_b", type="box", pos=f"0.07 {y} 0.005",  size="0.07 0.010 0.045", material="mat_box")
        add_geom(box, name=f"box_{tag}_guide_c", type="box", pos=f"0.265 {y} 0.005", size="0.015 0.010 0.045", material="mat_box")

    add_site(box, name="box_inlet_red", pos="-0.30 -0.045 0.025", size="0.018", rgba="0.05 0.25 1 1")
    add_site(box, name="box_outlet_red", pos="0.30 -0.045 0.025", size="0.018", rgba="0.05 0.25 1 1")
    add_site(box, name="box_inlet_blue", pos="-0.30 0.045 0.025", size="0.018", rgba="0.05 0.25 1 1")
    add_site(box, name="box_outlet_blue", pos="0.30 0.045 0.025", size="0.018", rgba="0.05 0.25 1 1")


# A5: fixture-panel/clip geometry, sized for the A2 cable diameter.
PANEL_TILT_DEG = 20.0
CLIP_COUNT = 4
CLIP_SPACING = 0.12
CLIP_WALL_HEIGHT = 0.015
CLIP_WALL_THICKNESS = 0.003
CLIP_WALL_DEPTH = 0.020


DEFAULT_PANEL_POS = (0.05, 0.06, 0.90)

# DLO layout: found via a numerical sweep against verify_dlo_scene.py's
# per-clip reachability check (not the "200-300mm forward of midpoint"
# center-point estimate, which left only ~17mm margin and actually failed
# r1's reach for clips 3-4 once real per-clip distances were checked). This
# position balances worst-case per-clip distance for r1 and r2 to within
# 1.4mm of each other (~383mm each), leaving ~51mm margin under the 440mm
# reach limit for both arms -- lower panel height (0.82 vs the default
# layout's 0.90) trades a little "on-the-panel" realism for meaningfully
# more reach margin; still 65mm above the table surface (z=0.755).
DLO_PANEL_POS = (-0.17, 0.01, 0.82)


def add_fixture_panel(
    worldbody: ET.Element,
    cable_diameter: float = CABLE_DIAMETER,
    pos: tuple = DEFAULT_PANEL_POS,
) -> ET.Element:
    """A5: rigid angled panel with 3-5 U-channel clips.

    `pos` defaults to the original placement (reachable by r2 only). The
    `dlo` layout (see generate()'s `layout` parameter) passes DLO_PANEL_POS
    instead, positioned per Plan 0's literal A5 spec -- reachable by both r1
    and r2 once DLO_PLACEMENTS also moves r1 closer to r2 (see
    add_robot_instances). Clip floor width tracks the actual cable diameter
    (A2) so the 12 mm clearance figure in Plan 0 stays correct if --diameter
    is overridden at generation time.
    """
    floor_width = cable_diameter + 0.002
    panel = ET.SubElement(
        worldbody,
        "body",
        name="fixture_panel",
        pos=text_vec(pos),
        euler=f"{PANEL_TILT_DEG} 0 0",
    )
    add_geom(
        panel,
        name="panel_base_plate",
        type="box",
        pos="0 0 -0.006",
        size=f"{(CLIP_COUNT - 1) * CLIP_SPACING / 2.0 + 0.05:.6g} 0.05 0.006",
        material="mat_frame",
    )
    add_site(panel, name="panel_anchor", pos="{:.6g} 0 0.006".format(-(CLIP_COUNT - 1) * CLIP_SPACING / 2.0 - 0.06), size="0.006", rgba="1 1 0 1")

    half_floor = floor_width / 2.0
    wall_y = half_floor + CLIP_WALL_THICKNESS / 2.0
    for i in range(CLIP_COUNT):
        clip_x = (i - (CLIP_COUNT - 1) / 2.0) * CLIP_SPACING
        clip = ET.SubElement(panel, "body", name=f"clip_{i + 1}", pos=text_vec((clip_x, 0.0, 0.0)))
        add_geom(
            clip,
            name=f"clip_{i + 1}_floor",
            type="box",
            pos="0 0 -0.003",
            size=f"0.010 {half_floor:.6g} 0.003",
            material="mat_frame",
        )
        add_geom(
            clip,
            name=f"clip_{i + 1}_wall_l",
            type="box",
            pos=f"0 {-wall_y:.6g} {CLIP_WALL_HEIGHT / 2.0:.6g}",
            size=f"{CLIP_WALL_DEPTH / 2.0:.6g} {CLIP_WALL_THICKNESS / 2.0:.6g} {CLIP_WALL_HEIGHT / 2.0:.6g}",
            material="mat_frame",
        )
        add_geom(
            clip,
            name=f"clip_{i + 1}_wall_r",
            type="box",
            pos=f"0 {wall_y:.6g} {CLIP_WALL_HEIGHT / 2.0:.6g}",
            size=f"{CLIP_WALL_DEPTH / 2.0:.6g} {CLIP_WALL_THICKNESS / 2.0:.6g} {CLIP_WALL_HEIGHT / 2.0:.6g}",
            material="mat_frame",
        )
        add_site(clip, name=f"clip_{i + 1}_center", pos="0 0 0.006", size="0.006", rgba="0 1 0 1")
    return panel


def add_physical_cable(
    worldbody: ET.Element,
    cable_name: str,
    material: str,
    y_offset: float,
    segments: int = CABLE_SEGMENTS,
    radius: float = CABLE_RADIUS,
    start_pos: tuple | None = None,
    marker_materials: dict[int, str] | None = None,
) -> None:
    if start_pos is None:
        start_pos = (-0.70, y_offset, 0.845)
    body = ET.SubElement(
        worldbody,
        "body",
        name=f"{cable_name}_cable_00",
        pos=text_vec(start_pos),
    )
    ET.SubElement(body, "freejoint", name=f"{cable_name}_cable_root")
    for index in range(segments):
        if index > 0:
            body = ET.SubElement(
                body,
                "body",
                name=f"{cable_name}_cable_{index:02d}",
                pos=text_vec((CABLE_SEGMENT_LENGTH, 0.0, 0.0)),
            )
            ET.SubElement(
                body,
                "joint",
                name=f"{cable_name}_cable_ball_{index:02d}",
                type="ball",
                pos=text_vec((-CABLE_SEGMENT_LENGTH / 2.0, 0.0, 0.0)),
                # A4: retuned from damping=0.03/armature=0.00005 so inter-frame
                # node motion stays within TrackDLO's motion-coherence budget
                # (< 20 mm at 30 Hz) once the heavier A2 (10 mm) capsules land.
                damping="0.15",
                armature="0.0005",
            )
        add_geom(
            body,
            name=f"{cable_name}_cable_geom_{index:02d}",
            type="capsule",
            fromto=text_vec((-CABLE_SEGMENT_LENGTH / 2.0, 0.0, 0.0, CABLE_SEGMENT_LENGTH / 2.0, 0.0, 0.0)),
            size=f"{radius:.6g}",
            material=(marker_materials or {}).get(index, material),
            density="1250",
            condim="4",
            friction="1.2 0.02 0.001",
            solref="0.008 1",
            solimp="0.95 0.99 0.001",
        )
        add_site(body, name=f"{cable_name}_cable_site_{index:02d}", size="0.004", rgba="1 1 1 0.25")


def add_cable_anchor(equality: ET.Element, cable_body: str = "red_cable_00") -> None:
    """A3: weld the cable root to the fixture panel's anchor site (A5).

    Only the red cable is anchored -- the blue cable is left free (per
    Plan 0's single-fixture-panel framing) so a future session can compare
    anchored vs. free-cable tracking behavior.
    """
    ET.SubElement(
        equality,
        "weld",
        name="red_cable_panel_anchor",
        body1=cable_body,
        body2="fixture_panel",
        relpose="0 0 0.01 1 0 0 0",
        solref="0.002 1",
        solimp="0.95 0.99 0.001",
    )


def add_cable_grasp_weld(equality: ET.Element, cable_body: str = "red_cable_00") -> None:
    """dlo layout: inactive weld between the cable root and r2's gripper.

    Starts inactive (active="false") -- DanglePlanner engages it at runtime
    once the gripper closes, after overwriting eq_data with the cable's
    actual relative pose at that instant so the weld starts at zero error.
    The relpose/solref values here are placeholders, fully overwritten
    before activation.
    """
    ET.SubElement(
        equality,
        "weld",
        name="r2_cable_grasp_weld",
        body1=cable_body,
        body2="r2_gripper_body",
        relpose="0 0 0 1 0 0 0",
        active="false",
        solref="0.002 1",
        solimp="0.95 0.99 0.001",
    )


DEFAULT_PLACEMENTS = [
    ("r1_", (-0.54, -0.22, 0.755), 0.20),
    ("r2_", (-0.05, 0.24, 0.755), -1.15),
    ("r3_", (0.52, -0.22, 0.755), 2.80),
]

# DLO layout: r1 moved from -0.54 to -0.30 in x (toward r2), giving ~523mm
# r1/r2 base separation (measured this session) -- close to Plan 0's A1
# assumption of "~500mm apart". r2 unchanged.
# r3 moved from (0.52, -0.22) to (0.52, 0.22) -- mirrored in Y about Y=0
# (r1/r3's own side vs. r2's side), i.e. the opposite side of the routing
# box from its original position. Position only -- yaw (2.80) left
# unchanged. r3 remains "idle" behaviorally (unused by run_dlo_v2.py).
#
# v3: r1 moved again, from (-0.30, -0.22) to (0.20, -0.22) -- now centered
# on the routing box's own X midpoint (x=0.20 == ROUTING_BOX_CENTER_X,
# box world-x span is [-0.11, 0.51], midpoint 0.20) instead of off to one
# side, sitting directly in front of the box (front guide rails/box_floor's
# -Y edge). y unchanged at -0.22 (already an established box-front standoff
# in this scene -- matches r1's own original DEFAULT_PLACEMENTS value and
# r3's DLO_PLACEMENTS value). Verified collision-free between r1's base
# link (r1_link_base_c) and every box_* geom by direct MuJoCo collision
# check (mj_forward + data.contact) at this position -- in fact the base
# stayed clear across the entire tested range down to y=-0.02, well inside
# the box's own footprint, so -0.22 has comfortable margin, not a
# minimum-clearance value. Yaw (0.20) left unchanged -- only position was
# requested to move.
#
# v3, second pass: r1's y moved again, from -0.22 to -0.27, closer to the
# table's own -Y edge. workstation_60x30_top spans world y=[-0.381, 0.381]
# (pos y=0, half-extent 0.381) -- the mount plate (radius 0.105m, the
# actual physical footprint the robot rests on) stays fully supported on
# the table as long as its center y >= -0.381+0.105 = -0.276. -0.27 is a
# real 5cm step closer to the edge than -0.22, with a small (~6mm) margin
# below the hard "starts overhanging" limit. Re-verified collision-free
# against every box_* geom at this position (still comfortably clear, per
# the note above -- the base doesn't collide with the box even at y=-0.02).
DLO_PLACEMENTS = [
    ("r1_", (0.20, -0.27, 0.755), 0.20),
    ("r2_", (-0.05, 0.24, 0.755), -1.15),
    ("r3_", (0.52, 0.22, 0.755), 2.80),
]


def add_robot_instances(
    worldbody: ET.Element,
    source_worldbody: ET.Element,
    placements: list = DEFAULT_PLACEMENTS,
) -> None:
    source_robot = first_world_body(source_worldbody)
    for prefix, pos, yaw in placements:
        mount = ET.SubElement(
            worldbody,
            "body",
            name=f"{prefix}mount",
            pos=text_vec(pos),
            euler=text_vec((0.0, 0.0, yaw)),
        )
        robot = copy.deepcopy(source_robot)
        prefix_robot_tree(robot, prefix)
        mount.append(robot)
        add_geom(
            mount,
            name=f"{prefix}mount_plate",
            type="cylinder",
            pos="0 0 0.012",
            size="0.105 0.012",
            material="mat_mount",
            contype="1",
            conaffinity="1",
        )


D435I_MESH_RELDIR = "../../realsense_d435i/assets"

D435I_MESH_MATERIALS = [
    ("d435i_0", "d435i_IR_Lens"),
    ("d435i_1", "d435i_IR_Emitter_Lens"),
    ("d435i_2", "d435i_IR_Rim"),
    ("d435i_3", "d435i_IR_Lens"),
    ("d435i_4", "d435i_Cameras_Gray"),
    ("d435i_5", "d435i_Black_Acrylic"),
    ("d435i_6", "d435i_Black_Acrylic"),
    ("d435i_7", "d435i_RGB_Pupil"),
    ("d435i_8", "d435i_Metal_Casing"),
]


def add_assets(root: ET.Element, source_asset: ET.Element | None) -> ET.Element:
    asset = ET.SubElement(root, "asset")
    if source_asset is not None:
        for child in list(source_asset):
            asset.append(copy.deepcopy(child))
    ET.SubElement(asset, "material", name="mat_worktop", rgba="0.72 0.70 0.66 1")
    ET.SubElement(asset, "material", name="mat_frame", rgba="0.20 0.22 0.24 1")
    ET.SubElement(asset, "material", name="mat_mount", rgba="0.08 0.09 0.10 1")
    ET.SubElement(asset, "material", name="mat_box", rgba="0.46 0.48 0.50 1")
    ET.SubElement(asset, "material", name="mat_cable_red", rgba="0.95 0.03 0.02 1")
    ET.SubElement(asset, "material", name="mat_cable_blue", rgba="0.02 0.18 0.95 1")
    # End-identification markers for real (non-GT) 2D-to-node correspondence:
    # segment 0 (node 0, the anchored/grasped end) and the last segment
    # (node CABLE_SEGMENTS-1, the free end) of the red cable get distinct
    # colors so a single view can self-identify which physical end it sees.
    # Hues chosen well clear of mat_cable_red/blue and the existing
    # yellow/green site markers -- see rendering/cable_mask.py's _HUE_RANGES.
    # Chosen and empirically verified (via a real rendered test frame) to
    # land at hue~62 (green) and hue~92 (cyan) respectively -- well clear of
    # red's true hue~0, blue's true hue~115 AND cable_mask.py's (inaccurate
    # but pre-existing) declared blue range (150,180). An initial magenta/
    # violet-pink choice drifted under shading straight into those bands.
    ET.SubElement(asset, "material", name="mat_cable_marker_start", rgba="0.05 0.85 0.10 1")  # green
    ET.SubElement(asset, "material", name="mat_cable_marker_end", rgba="0.05 0.80 0.85 1")  # cyan
    for i in range(9):
        ET.SubElement(asset, "mesh", name=f"d435i_{i}", file=f"{D435I_MESH_RELDIR}/d435i_{i}.obj")
    ET.SubElement(asset, "material", name="d435i_Black_Acrylic", rgba="0.070 0.070 0.070 1")
    ET.SubElement(asset, "material", name="d435i_Cameras_Gray", rgba="0.296 0.296 0.296 1")
    ET.SubElement(asset, "material", name="d435i_IR_Emitter_Lens", rgba="0.287 0.665 0.328 1")
    ET.SubElement(asset, "material", name="d435i_IR_Lens", rgba="0.036 0.036 0.036 1")
    ET.SubElement(asset, "material", name="d435i_IR_Rim", rgba="0.799 0.807 0.799 1")
    ET.SubElement(asset, "material", name="d435i_Metal_Casing", rgba="1 1 1 1")
    ET.SubElement(asset, "material", name="d435i_RGB_Pupil", rgba="0.087 0.003 0.009 1")
    return asset


def find_body(root: ET.Element, name: str) -> ET.Element | None:
    for body in root.iter("body"):
        if body.get("name") == name:
            return body
    return None


D435I_MOUNT_QUATS = {
    # Identity: camera's default -Z look-direction = link6's -Z = toward
    # workspace (see comment below). r1_ overridden to face the opposite
    # way (180 deg about local X -- "0 1 0 0" in MuJoCo's wxyz convention)
    # -- flips r1's camera POV to the backside of its default view. Only
    # r1_ is overridden: r2_'s camera feeds the TrackDLO tracking pipeline
    # elsewhere in this project (see trackdlo_eval/), so its POV is left
    # untouched; r3_ is unused/idle and also left at the default.
    "r1_": "0 1 0 0",
}


def add_cameras_to_arms(worldbody: ET.Element) -> None:
    for prefix in ("r1_", "r2_", "r3_"):
        link6 = find_body(worldbody, f"{prefix}link6")
        if link6 is None:
            raise ValueError(f"Could not find {prefix}link6 in worldbody tree.")
        # Camera mount on link6 wrist: 45 mm lateral (Y), 10 mm behind joint (−Z).
        # Identity quat: camera's default −Z look-direction = link6's −Z = toward workspace.
        # (link6 +Z points away from the workspace when the arm reaches forward, so the
        # earlier 180°-around-X quat "0 1 0 0" that aligned camera look with link6 +Z
        # was causing the camera to face backward. See D435I_MOUNT_QUATS above for the
        # per-prefix override -- r1_'s camera is now deliberately backward-facing.)
        mount = ET.SubElement(
            link6,
            "body",
            name=f"{prefix}d435i",
            pos="0 0.045 -0.01",
            quat=D435I_MOUNT_QUATS.get(prefix, "1 0 0 0"),
        )
        for mesh, mat in D435I_MESH_MATERIALS[:-1]:
            ET.SubElement(
                mount, "geom",
                type="mesh", mesh=mesh, material=mat,
                contype="0", conaffinity="0", group="2", mass="0",
            )
        # Metal casing: carries physical mass of 72 g
        ET.SubElement(
            mount, "geom",
            type="mesh", mesh="d435i_8", material="d435i_Metal_Casing",
            contype="0", conaffinity="0", group="2", mass="0.072",
        )
        # RGB camera: 42.5° FoV. Offset from mount origin = RGB pupil center in the
        # D435i mesh (vertex mean of d435i_7.obj): +32.49 mm along housing long axis (X),
        # centered in Y, 3.62 mm recessed from the front face (−Z).
        rgb_body = ET.SubElement(mount, "body", name=f"{prefix}d435i_rgb_body", pos="0.0325 0 -0.0036")
        ET.SubElement(rgb_body, "camera", name=f"{prefix}d435i_rgb", fovy="42.5")
        # Depth/stereo camera: 62° FoV. Midpoint between the two IR lenses (~−7.5 mm X,
        # centered Y, 1.3 mm from front face).
        depth_body = ET.SubElement(mount, "body", name=f"{prefix}d435i_depth_body", pos="-0.0075 0 -0.0013")
        ET.SubElement(depth_body, "camera", name=f"{prefix}d435i_depth", fovy="62")


def add_prefixed_top_level(
    root: ET.Element,
    source_root: ET.Element,
    tag: str,
    target: ET.Element | None = None,
) -> None:
    source = source_root.find(tag)
    if source is None:
        return
    if target is None:
        target = ET.SubElement(root, tag)
    for prefix in ("r1_", "r2_", "r3_"):
        for child in clone_prefixed_children(source, prefix):
            target.append(child)


def generate(
    segments: int = CABLE_SEGMENTS,
    diameter: float = CABLE_DIAMETER,
    output_name: str | None = None,
    layout: str = "default",
    marker_materials: dict[int, str] | None = None,
) -> Path:
    """`layout="default"` (r1/r2/r3 per DEFAULT_PLACEMENTS, panel at
    DEFAULT_PANEL_POS) is what run_demo.py/verify_scene.py/the old
    box-routing task use -- unaffected by this parameter's existence, since
    they all call generate() with no arguments. `layout="dlo"` swaps in
    DLO_PLACEMENTS (r1 moved closer to r2) and DLO_PANEL_POS (panel
    repositioned per Plan 0's literal A5 spec, reachable by both r1 and r2),
    for run_dlo.py's separate two-arm task. `output_name` defaults to
    OUTPUT_NAME or DLO_OUTPUT_NAME based on `layout` unless explicitly
    overridden.
    """
    if layout not in ("default", "dlo"):
        raise ValueError(f"layout must be 'default' or 'dlo', got {layout!r}")
    if output_name is None:
        output_name = OUTPUT_NAME if layout == "default" else DLO_OUTPUT_NAME
    placements = DEFAULT_PLACEMENTS if layout == "default" else DLO_PLACEMENTS
    panel_pos = DEFAULT_PANEL_POS if layout == "default" else DLO_PANEL_POS

    radius = diameter / 2.0
    root_dir = project_root()
    source_path = root_dir / MODEL_RELATIVE
    output_path = root_dir / output_name
    if not source_path.exists():
        raise FileNotFoundError(
            f"Missing official Lite 6 XML: {source_path}. Run tools/download_menagerie_lite6.py first."
        )

    source_tree = ET.parse(source_path)
    source_root = source_tree.getroot()
    source_worldbody = source_root.find("worldbody")
    if source_worldbody is None:
        raise ValueError("The official Lite 6 XML does not contain <worldbody>.")

    model_name = "triple_lite6_cable_routing_dlo_v3" if layout == "dlo" else "triple_lite6_cable_routing"
    mjcf = ET.Element("mujoco", model=model_name)
    ET.SubElement(
        mjcf,
        "compiler",
        angle="radian",
        autolimits="true",
        meshdir="vendor/mujoco_menagerie/ufactory_lite6/assets",
        texturedir="vendor/mujoco_menagerie/ufactory_lite6/assets",
    )
    ET.SubElement(
        mjcf,
        "option",
        timestep="0.0005",
        integrator="implicitfast",
        gravity="0 0 -9.81",
        cone="elliptic",
        iterations="80",
        ls_iterations="20",
    )
    ET.SubElement(mjcf, "size", njmax="9000", nconmax="2500")
    # Offscreen framebuffer large enough for cable_side's higher-resolution
    # tracking render (see run_dlo_v2.py / bridge_publish_trajectory.py --
    # bumped to get the rendered cable comfortably above the ModeFilter
    # survival width without moving the camera or retuning the tracker).
    visual = ET.SubElement(mjcf, "visual")
    ET.SubElement(visual, "global", offwidth="2560", offheight="1920")

    for tag in ("default", "extension", "custom"):
        source_node = source_root.find(tag)
        if source_node is not None:
            mjcf.append(copy.deepcopy(source_node))

    add_assets(mjcf, source_root.find("asset"))

    worldbody = ET.SubElement(mjcf, "worldbody")
    ET.SubElement(worldbody, "light", name="key_light", pos="-0.4 -0.8 2.5", dir="0.3 0.6 -1", diffuse="0.9 0.9 0.85")
    ET.SubElement(worldbody, "light", name="fill_light", pos="0.8 0.6 2.0", dir="-0.2 -0.2 -1", diffuse="0.35 0.38 0.42")
    ET.SubElement(worldbody, "camera", name="overview", pos="1.55 -1.35 1.55", xyaxes="0.68 0.73 0 -0.36 0.34 0.87")
    # Top-down view: centred over the full scene (clips x=-0.35 to grey box x=+0.51),
    # 0.9 m above the fixture plane. xyaxes "1 0 0 0 1 0" → camera looks straight down.
    ET.SubElement(worldbody, "camera", name="box_focus", pos="0.08 0 1.72", xyaxes="1 0 0 0 1 0", fovy="45")
    # Isometric view: front-right-above at ~30° elevation, looking toward scene centre.
    # xyaxes derived so camera +X ≈ world (+X,+Y) and image up has positive world Z.
    ET.SubElement(worldbody, "camera", name="fixture_iso", pos="0.74 -0.69 1.54", xyaxes="0.722 0.691 0 -0.416 0.435 0.799", fovy="45")
    add_geom(worldbody, name="floor", type="plane", size="2.5 2.0 0.02", rgba="0.18 0.19 0.20 1")

    add_workstation(worldbody)
    add_routing_box(worldbody)
    if layout == "default":
        add_fixture_panel(worldbody, cable_diameter=diameter, pos=panel_pos)
        add_physical_cable(worldbody, "red", "mat_cable_red", -0.045, segments=segments, radius=radius)
        add_physical_cable(worldbody, "blue", "mat_cable_blue", 0.045, segments=segments, radius=radius)
    else:
        # dlo layout: single red cable starting freely at the routing box centre.
        # No fixture panel, no anchor weld, no blue cable.
        default_markers = {0: "mat_cable_marker_start", segments - 1: "mat_cable_marker_end"}
        add_physical_cable(worldbody, "red", "mat_cable_red", 0,
                           segments=segments, radius=radius, start_pos=(-0.20, 0, 0.845),
                           marker_materials=default_markers if marker_materials is None else marker_materials)
        # Close top-down camera for G1 init (prepare_full_cable_frame.py only).
        # Standoff 1.014 m (z=1.81, cable rests at z~=0.799) -> ~20px cable width
        # at 2240x1680, chosen so both cable ends fit in frame (required
        # horizontal field ~1.12m vs. ~0.85m physical cable extent). x recentered
        # to the cable's midpoint (0.18) instead of the routing-box center (0.20).
        ET.SubElement(worldbody, "camera",
                      name="cable_init_top",
                      pos="0.18 0 1.81",
                      xyaxes="1 0 0 0 1 0",
                      fovy="45")
        # Lateral camera from +Y side (same side as r2) for G2/G3 tracking (bridge).
        # xyaxes: fwd=(0.394,-0.848,-0.355), x=cross(fwd,Z), y=cross(-fwd,x). Standoff ~0.51 m → ~11 px geometric.
        ET.SubElement(worldbody, "camera",
                      name="cable_lateral",
                      pos="0.0 0.43 1.0",
                      xyaxes="-0.907 -0.422 0.000 0.150 -0.322 0.935",
                      fovy="45")
        # Head-on side elevation of the whole box from the -Y side (opposite
        # cable_lateral / r2's side), sitting between r1 (x=-0.30) and r3
        # (x=0.52) -- x centered on the midpoint of the required span, not on
        # r1/r3's own position. Level, no elevation/tilt beyond a slight
        # downward aim. Standoff (0.68m) is the closest that still keeps both
        # red_cable_00's grasp point (x=-0.194, outside the box's own left
        # edge) and the box's far right edge (x=0.51) inside frame at fovy=45
        # -- verified via projection math, not just eyeballed. v2 only.
        # fwd=(0.0,0.9973,-0.0733), aimed at (0.16, 0, 0.82).
        ET.SubElement(worldbody, "camera",
                      name="cable_side",
                      pos="0.16 -0.68 0.87",
                      xyaxes="1.0 -0.0 0.0 0.0 0.0733 0.9973",
                      fovy="45")
    add_robot_instances(worldbody, source_worldbody, placements=placements)
    add_cameras_to_arms(worldbody)

    equality = ET.SubElement(mjcf, "equality")
    if layout == "default":
        add_cable_anchor(equality)
    else:
        add_cable_grasp_weld(equality)
    for tag in ("contact", "tendon", "actuator", "sensor"):
        add_prefixed_top_level(mjcf, source_root, tag)
    add_prefixed_top_level(mjcf, source_root, "equality", target=equality)

    indent(mjcf)
    ET.ElementTree(mjcf).write(output_path, encoding="utf-8", xml_declaration=True)
    print(f"Wrote {output_path}")
    return output_path


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--segments", type=int, default=CABLE_SEGMENTS,
        help=f"Cable segment count (default {CABLE_SEGMENTS}; original variant was 36).",
    )
    parser.add_argument(
        "--diameter", type=float, default=CABLE_DIAMETER,
        help=f"Cable diameter in meters (default {CABLE_DIAMETER}; thin variant is 0.005).",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help=f"Output MJCF filename (default {OUTPUT_NAME}, or {DLO_OUTPUT_NAME} for --layout dlo).",
    )
    parser.add_argument(
        "--layout", type=str, choices=("default", "dlo"), default="default",
        help="'default' is the original r1/r2/r3 layout (box-routing task, unaffected). "
             "'dlo' repositions r1 closer to r2 (~500mm apart) and moves the fixture panel "
             "to be reachable by both, per Plan 0's literal A5 spec -- for run_dlo.py.",
    )
    args = parser.parse_args()

    try:
        generate(segments=args.segments, diameter=args.diameter, output_name=args.output, layout=args.layout)
    except Exception as exc:
        print(f"Scene generation failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""E2 (P3-TWIN-ROBOT): build sim/twin/robot/so101_twin_v2.xml from so101_twin.xml (v1, unchanged) and the
P3-TWIN-DYN parameters in sim/twin/dynamics/twin_dynamics_v1.json. Run with: python -I -B make_v2.py"""
import copy
import json
import xml.etree.ElementTree as ET

R = "/home/intel/intel-junction-2026"
V1 = f"{R}/sim/twin/robot/so101_twin.xml"
OUT = f"{R}/sim/twin/robot/so101_twin_v2.xml"
DYN = f"{R}/sim/twin/dynamics/twin_dynamics_v1.json"
ARM_LINKS = ["base_link", "shoulder_link", "upper_arm_link", "lower_arm_link", "wrist_link"]
GRIP_BODY = "gripper_link"
JAW = "moving_jaw_so101_v1_link"
JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
# parent-child pairs and base-vs-mat are excluded; arm links never collide with each other (contype 2 / conaffinity 1)
EXCLUDES = [("world", "base_link"), ("wrist_link", GRIP_BODY), (GRIP_BODY, JAW)]
HEADER = (
    "<!-- so101_twin_v2.xml (E2, P3-TWIN-ROBOT). Built by work/P3-TWIN-ROBOT/E2/tools/make_v2.py from so101_twin.xml (v1,\n"
    "     unchanged) and sim/twin/dynamics/twin_dynamics_v1.json (P3-TWIN-DYN E1).\n"
    "     Collision: convex hulls of the v1 meshes on group 3 (arm links contype 2 conaffinity 1; gripper body and\n"
    "     moving jaw contype 1 conaffinity 1). Visual geoms stay group 2, contype 0, conaffinity 0.\n"
    "     Excludes: world-base_link, wrist_link-gripper_link, gripper_link-moving jaw.\n"
    "     Actuators: <position>, kp/kv/forcerange from the DYN fit. ctrl is in radians.\n"
    "     Arm ctrl = q (equals k_j * goal, the urdf_zero law used by the DYN fit).\n"
    "     Gripper ctrl = -0.174533 + k_g*(goal - 0.52) (units.py closed_at_empty_reading); DYN gripper gains are\n"
    "     applied under that law, see README. -->\n"
)


def main():
    dyn = json.load(open(DYN))
    par = dyn["parameters"]
    root = ET.parse(V1).getroot()

    for body in root.iter("body"):
        name = body.get("name")
        if name in ARM_LINKS:
            ct, ca = "2", "1"
        elif name in (GRIP_BODY, JAW):
            ct, ca = "1", "1"
        else:
            continue
        for g in list(body.findall("geom")):
            if g.get("group") != "2":
                continue
            c = copy.deepcopy(g)
            c.set("group", "3")
            c.set("contype", ct)
            c.set("conaffinity", ca)
            c.set("rgba", "0.2 0.6 1.0 0.3")
            body.insert(list(body).index(g) + 1, c)

    for body in root.iter("body"):
        for j in body.findall("joint"):
            jn = j.get("name")
            if jn in par:
                j.set("armature", repr(par[jn]["armature"]))
                j.set("frictionloss", repr(par[jn]["frictionloss"]))
                j.set("damping", "0")

    root.insert(1, ET.Element("option", timestep="0.004", integrator="implicitfast", gravity="0 0 -9.81"))
    contact = ET.SubElement(root, "contact")
    for a, b in EXCLUDES:
        ET.SubElement(contact, "exclude", body1=a, body2=b)

    act = ET.SubElement(root, "actuator")
    for jn in JOINTS:
        p = par[jn]
        fr = f"{-p['forcerange']} {p['forcerange']}"
        ET.SubElement(act, "position", name=f"act_{jn}", joint=jn, kp=repr(p["kp"]), kv=repr(p["kv"]),
                      forcelimited="true", forcerange=fr)

    ET.indent(root, space="  ")
    text = ET.tostring(root, encoding="unicode")
    with open(OUT, "w") as fh:
        fh.write(HEADER + text + "\n")
    print("wrote", OUT)


if __name__ == "__main__":
    main()

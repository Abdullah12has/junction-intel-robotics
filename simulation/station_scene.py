"""Approximate reconstruction from the two RealSense frames, 2026-11-14.

Geometry is visually fitted, not camera/robot calibration. No hardware access.
"""
import math
import xml.etree.ElementTree as ET

import numpy as np

BOWL_CENTERS = {"blue": (-0.105, -0.13), "green": (0.235, -0.16)}
BOWL_FLOOR = 0.004
BRICK_HALF_HEIGHT = 0.0096
REFERENCE_COLORS = ["red", "yellow", "orange", "purple", "white", "cyan"]
REFERENCE_POSITIONS = [(0.035, -0.235), (0.12, -0.065), (0.105, -0.275),
                       (0.08, -0.33), (0.14, -0.235), (0.10, -0.16)]
HOME = [0, -0.5, 0.8, 0.6, 0, 0.5]


def vector(values):
    return " ".join(str(float(v)) for v in values)


def ring_mesh(asset, name, profile, segments=48):
    """Closed ring surface; render-only, collision uses separate convex pieces."""
    vertices = []
    for radius, z in profile:
        for i in range(segments):
            theta = 2 * math.pi * i / segments
            vertices.append((radius * math.cos(theta), radius * math.sin(theta), z))
    faces = []
    for row in range(len(profile)):
        next_row = (row + 1) % len(profile)
        for i in range(segments):
            j = (i + 1) % segments
            a, b = row * segments + i, row * segments + j
            c, d = next_row * segments + j, next_row * segments + i
            faces.extend([(a, b, c), (a, c, d)])
    ET.SubElement(asset, "mesh", name=name, vertex=" ".join(vector(v) for v in vertices),
                  face=" ".join(" ".join(map(str, f)) for f in faces))


def camera_look_at(camera, position, target):
    z = np.array(position, float) - np.array(target, float)
    z /= np.linalg.norm(z)
    x = np.cross([0, 0, 1], z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    camera.set("pos", vector(position))
    camera.set("xyaxes", vector([*x, *y]))


def recreate_station(root):
    world, asset = root.find("worldbody"), root.find("asset")
    # Station forward is -Y. Orient the model base accordingly, preserving the
    # previous shoulder-axis location and the robot's original joint limits.
    base = world.find("body[@name='base']")
    base.set("quat", "0.7071067812 0 0 -0.7071067812")
    base.set("pos", "0.0388353 0.0388353 0")
    table = world.find("geom[@name='table']")
    table.set("pos", "0 0 -0.004")
    table.set("rgba", "0.62 0.37 0.16 1")
    ET.SubElement(world, "geom", name="black_mat", type="box", pos="0.035 -0.22 -0.002",
                  size="0.265 0.27 0.002", rgba="0.025 0.028 0.03 1", friction="1 0.01 0.001")
    # Match the observed black links and orange printed brackets.
    for material in asset.findall("material"):
        name = material.get("name", "")
        if "sts3215" in name:
            material.set("rgba", "0.035 0.035 0.04 1")
        elif "moving_jaw" in name or "wrist_roll_follower" in name:
            material.set("rgba", "0.95 0.22 0.025 1")
        else:
            material.set("rgba", "0.065 0.065 0.07 1")
    overhead = world.find("camera[@name='overhead']")
    camera_look_at(overhead, [-0.025, -0.46, 0.61], [-0.025, -0.22, 0])
    overhead.set("fovy", "40")
    # Match camera2's observed close view. Its actual mounting is unverified;
    # represent it as a fixed camera until motion-based calibration establishes it.
    wrist = world.find(".//camera[@name='wrist']")
    world.find(".//body[@name='gripper']").remove(wrist)
    world.append(wrist)
    camera_look_at(wrist, [0.025, -0.45, 0.22], [0.065, -0.22, 0.015])
    wrist.set("fovy", "40")
    light = world.find("light")
    light.set("pos", "-0.4 -0.2 1.5")
    light.set("diffuse", "0.65 0.65 0.65")
    ET.SubElement(root.find("visual"), "headlight", ambient="0.4 0.4 0.4", diffuse="0.4 0.4 0.4", specular="0.1 0.1 0.1")
    # Context objects are visual-only approximations, not robot collision geometry.
    def prop(name, pos, size, rgba):
        ET.SubElement(world, "geom", name=name, type="box", pos=vector(pos), size=vector(size),
                      rgba=rgba, contype="0", conaffinity="0")
    prop("laptop_base", [-0.315, -0.085, 0.006], [.085, .125, .01], ".09 .10 .12 1")
    prop("laptop_trackpad", [-.315, -.04, .017], [.04, .035, .001], ".14 .15 .17 1")
    for row in range(5):
        for col in range(10):
            prop(f"key_{row}_{col}", [-.388 + col * .0155, -.11 - row * .016, .018],
                 [.0065, .0065, .001], ".045 .05 .06 1")
    prop("power_adapter", [-.235, -.3, .014], [.033, .05, .018], ".07 .075 .08 1")
    prop("foreground_camera", [-.01, -.405, .025], [.09, .013, .018], ".6 .62 .63 1")
    prop("foreground_mount", [-.01, -.415, .006], [.13, .03, .006], ".3 .31 .32 1")
    for offset in (0, .013, .028):
        pts = [(-.21-offset,.07,.003),(-.245-offset,-.09,.006),(-.23-offset,-.22,.008),
               (-.28-offset,-.35,.005),(-.19-offset,-.43,.006)]
        for j, (a,b) in enumerate(zip(pts,pts[1:])):
            ET.SubElement(world, "geom", name=f"cable_{offset}_{j}", type="capsule", fromto=vector([*a,*b]),
                          size="0.002", rgba=".015 .015 .018 1", contype="0", conaffinity="0")


def add_bowl(root, name, center):
    world, asset = root.find("worldbody"), root.find("asset")
    rgba = "0.015 0.72 0.95 1" if name == "blue" else "0.045 0.65 0.29 1"
    ET.SubElement(asset, "material", name=f"bowl_{name}_material", rgba=rgba, specular="0.45", shininess="0.65")
    material = f"bowl_{name}_material"
    x,y = center
    ET.SubElement(world, "geom", name=f"plate_{name}", type="cylinder", pos=f"{x} {y} .002",
                  size=".064 .002", material=material, friction=".8 .01 .001")
    profile = [(.062,.004),(.083,.022),(.087,.023),(.067,0)]
    ring_mesh(asset, f"bowl_{name}_mesh", profile)
    ET.SubElement(world, "geom", name=f"bowl_{name}_visual", type="mesh", mesh=f"bowl_{name}_mesh",
                  pos=f"{x} {y} 0", material=material, contype="0", conaffinity="0")
    # Convex sectors preserve the cavity instead of using a filled convex hull.
    for i in range(40):
        vertices = []
        for theta in (2*math.pi*i/40, 2*math.pi*(i+1)/40):
            vertices.extend((r*math.cos(theta),r*math.sin(theta),z) for r,z in profile)
        mesh = f"bowl_{name}_sector_{i}"
        ET.SubElement(asset,"mesh",name=mesh,vertex=" ".join(vector(v) for v in vertices))
        ET.SubElement(world,"geom",name=mesh,type="mesh",mesh=mesh,pos=f"{x} {y} 0",
                      rgba="0 0 0 0",friction=".8 .01 .001")


def enlarge_brick(body, asset, prefix, brick_type):
    """DUPLO-scale body and hollow visual studs; collision studs are cylinders."""
    body_geom = body.find("geom[@type='box']")
    half = [0.0159 if brick_type == "2x2" else 0.0319, 0.0159, BRICK_HALF_HEIGHT]
    body_geom.set("size", vector(half))
    body_geom.set("mass", "0.006" if brick_type == "2x2" else "0.012")
    for geom in list(body.findall("geom[@type='cylinder']")):
        pos = np.fromstring(geom.get("pos"),sep=" ")
        pos[:2] *= 2
        pos[2] = BRICK_HALF_HEIGHT + .0018
        geom.set("pos",vector(pos))
        geom.set("size",".0048 .0018")
        geom.set("rgba","0 0 0 0")
        mesh_name=prefix+geom.get("name")+"_ring"
        ring_mesh(asset,mesh_name,[(.0033,-.0018),(.0033,.0018),(.0048,.0018),(.0048,-.0018)],24)
        ET.SubElement(body,"geom",name=mesh_name,type="mesh",mesh=mesh_name,pos=vector(pos),
                      mass="0",contype="0",conaffinity="0",rgba=".8 .1 .1 1")

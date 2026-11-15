"""E1 (P3-TWIN-ROBOT) URDF helpers. Reads a URDF as text: never executes it.

parse_urdf(path) -> {"name", "links", "joints", "visuals", "materials"}
compare_urdf(a, b) -> per-joint and per-link differences (mm, deg, kg, kg m^2)
build_mjcf(parsed, tip_link, model_name, source_note, with_visuals) -> MJCF text with the same kinematic tree:
  one body per URDF link, body pos/quat = URDF joint origin, revolute joint = MuJoCo hinge about the URDF axis,
  inertials from the URDF, the massless tip link as a site on its parent body, and (optional) mesh visual geoms
  placed with the URDF visual origins. Visual geoms have contype=0 conaffinity=0 (no collision).
"""
import math
import xml.etree.ElementTree as ET


def _vec(text, default):
    if text is None:
        return [float(v) for v in default]
    return [float(v) for v in text.split()]


def matmul(a, b):
    return [[sum(a[i][k] * b[k][j] for k in range(3)) for j in range(3)] for i in range(3)]


def transpose(a):
    return [[a[j][i] for j in range(3)] for i in range(3)]


def rpy_to_matrix(rpy):
    """URDF roll-pitch-yaw: R = Rz(yaw) @ Ry(pitch) @ Rx(roll)."""
    r, p, y = rpy
    cr, sr = math.cos(r), math.sin(r)
    cp, sp = math.cos(p), math.sin(p)
    cy, sy = math.cos(y), math.sin(y)
    rz = [[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]]
    ry = [[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]]
    rx = [[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]]
    return matmul(rz, matmul(ry, rx))


def matrix_to_quat(r):
    """Rotation matrix to unit quaternion (w, x, y, z), w >= 0 (MuJoCo order)."""
    t = r[0][0] + r[1][1] + r[2][2]
    if t > 0.0:
        s = math.sqrt(t + 1.0) * 2.0
        q = (0.25 * s, (r[2][1] - r[1][2]) / s, (r[0][2] - r[2][0]) / s, (r[1][0] - r[0][1]) / s)
    elif r[0][0] > r[1][1] and r[0][0] > r[2][2]:
        s = math.sqrt(1.0 + r[0][0] - r[1][1] - r[2][2]) * 2.0
        q = ((r[2][1] - r[1][2]) / s, 0.25 * s, (r[0][1] + r[1][0]) / s, (r[0][2] + r[2][0]) / s)
    elif r[1][1] > r[2][2]:
        s = math.sqrt(1.0 + r[1][1] - r[0][0] - r[2][2]) * 2.0
        q = ((r[0][2] - r[2][0]) / s, (r[0][1] + r[1][0]) / s, 0.25 * s, (r[1][2] + r[2][1]) / s)
    else:
        s = math.sqrt(1.0 + r[2][2] - r[0][0] - r[1][1]) * 2.0
        q = ((r[1][0] - r[0][1]) / s, (r[0][2] + r[2][0]) / s, (r[1][2] + r[2][1]) / s, 0.25 * s)
    n = math.sqrt(sum(v * v for v in q))
    q = [v / n for v in q]
    if q[0] < 0.0:
        q = [-v for v in q]
    return q


def parse_urdf(path):
    root = ET.parse(path).getroot()
    materials = {}
    for el in root.findall("material"):
        color = el.find("color")
        if color is not None:
            materials[el.get("name")] = color.get("rgba")
    links, visuals = {}, {}
    for el in root.findall("link"):
        name = el.get("name")
        inertial = el.find("inertial")
        if inertial is None:
            links[name] = None
        else:
            origin = inertial.find("origin")
            mass = inertial.find("mass")
            ie = inertial.find("inertia")
            links[name] = {
                "xyz": _vec(origin.get("xyz") if origin is not None else None, (0, 0, 0)),
                "rpy": _vec(origin.get("rpy") if origin is not None else None, (0, 0, 0)),
                "mass": float(mass.get("value")),
                "inertia": {k: float(ie.get(k, "0")) for k in ("ixx", "ixy", "ixz", "iyy", "iyz", "izz")},
            }
        vis = []
        for v in el.findall("visual"):
            origin = v.find("origin")
            mesh = v.find("geometry/mesh")
            mat = v.find("material")
            if mesh is None:
                raise ValueError(f"visual of link {name} is not a mesh")
            mat_name = mat.get("name") if mat is not None else None
            vis.append({
                "xyz": _vec(origin.get("xyz") if origin is not None else None, (0, 0, 0)),
                "rpy": _vec(origin.get("rpy") if origin is not None else None, (0, 0, 0)),
                "mesh_file": mesh.get("filename").split("/")[-1],
                "material": mat_name,
                "rgba": materials.get(mat_name),
            })
        visuals[name] = vis
    joints = []
    for el in root.findall("joint"):
        origin = el.find("origin")
        axis = el.find("axis")
        limit = el.find("limit")
        lower = limit.get("lower") if limit is not None else None
        upper = limit.get("upper") if limit is not None else None
        joints.append({
            "name": el.get("name"),
            "type": el.get("type"),
            "parent": el.find("parent").get("link"),
            "child": el.find("child").get("link"),
            "xyz": _vec(origin.get("xyz") if origin is not None else None, (0, 0, 0)),
            "rpy": _vec(origin.get("rpy") if origin is not None else None, (0, 0, 0)),
            "axis": _vec(axis.get("xyz") if axis is not None else None, (1, 0, 0)),
            "lower": float(lower) if lower is not None else None,
            "upper": float(upper) if upper is not None else None,
        })
    return {"name": root.get("name"), "links": links, "joints": joints,
            "visuals": visuals, "materials": materials}


def _rot_angle_deg(a, b):
    m = matmul(transpose(a), b)
    c = max(-1.0, min(1.0, (m[0][0] + m[1][1] + m[2][2] - 1.0) / 2.0))
    return math.degrees(math.acos(c))


def _axis_angle_deg(a, b):
    """Angle between two joint axes; None when either axis is zero (fixed joints)."""
    na = math.sqrt(sum(v * v for v in a))
    nb = math.sqrt(sum(v * v for v in b))
    if na == 0.0 or nb == 0.0:
        return None
    c = max(-1.0, min(1.0, sum(x * y for x, y in zip(a, b)) / (na * nb)))
    return math.degrees(math.acos(c))


def compare_urdf(a, b):
    ja = {j["name"]: j for j in a["joints"]}
    jb = {j["name"]: j for j in b["joints"]}
    out = {
        "joints_only_in_a": sorted(set(ja) - set(jb)),
        "joints_only_in_b": sorted(set(jb) - set(ja)),
        "joints": {},
        "links": {},
    }
    for name in sorted(set(ja) & set(jb)):
        x, y = ja[name], jb[name]
        out["joints"][name] = {
            "type_a": x["type"],
            "type_b": y["type"],
            "same_parent_child": (x["parent"], x["child"]) == (y["parent"], y["child"]),
            "origin_xyz_diff_mm": 1000.0 * math.dist(x["xyz"], y["xyz"]),
            "origin_rot_diff_deg": _rot_angle_deg(rpy_to_matrix(x["rpy"]), rpy_to_matrix(y["rpy"])),
            "axis_a": x["axis"],
            "axis_b": y["axis"],
            "axis_angle_diff_deg": _axis_angle_deg(x["axis"], y["axis"]),
            "limit_a": [x["lower"], x["upper"]],
            "limit_b": [y["lower"], y["upper"]],
        }
    for name in sorted(set(a["links"]) & set(b["links"])):
        la, lb = a["links"][name], b["links"][name]
        if la is None or lb is None:
            out["links"][name] = {"inertial_a": la is not None, "inertial_b": lb is not None}
            continue
        ia = [la["inertia"][k] for k in ("ixx", "iyy", "izz", "ixy", "ixz", "iyz")]
        ib = [lb["inertia"][k] for k in ("ixx", "iyy", "izz", "ixy", "ixz", "iyz")]
        out["links"][name] = {
            "mass_a": la["mass"],
            "mass_b": lb["mass"],
            "com_diff_mm": 1000.0 * math.dist(la["xyz"], lb["xyz"]),
            "inertia_max_abs_diff": max(abs(p - q) for p, q in zip(ia, ib)),
        }
    return out


def _fmt(values):
    return " ".join(f"{float(v):.10g}" for v in values)


def build_mjcf(parsed, tip_link, model_name, source_note, with_visuals=False):
    links, joints = parsed["links"], parsed["joints"]
    children = {}
    for j in joints:
        children.setdefault(j["parent"], []).append(j)
    child_names = {j["child"] for j in joints}
    roots = [n for n in links if n not in child_names]
    if len(roots) != 1:
        raise ValueError(f"expected one root link, found {roots}")

    def inertial(link, ind):
        info = links.get(link)
        if info is None:
            raise ValueError(f"link {link} has no inertial")
        if any(abs(v) > 1e-12 for v in info["rpy"]):
            raise ValueError(f"inertial frame of {link} is rotated; not handled by this converter")
        ie = info["inertia"]
        full = _fmt([ie["ixx"], ie["iyy"], ie["izz"], ie["ixy"], ie["ixz"], ie["iyz"]])
        return f'{ind}<inertial pos="{_fmt(info["xyz"])}" mass="{info["mass"]:.10g}" fullinertia="{full}"/>'

    def visual_geoms(link, ind):
        if not with_visuals:
            return []
        out = []
        for v in parsed["visuals"].get(link, []):
            stem = v["mesh_file"][:-4]
            quat = _fmt(matrix_to_quat(rpy_to_matrix(v["rpy"])))
            rgba = v["rgba"] or "0.5 0.5 0.5 1"
            out.append(f'{ind}<geom type="mesh" mesh="{stem}" pos="{_fmt(v["xyz"])}" quat="{quat}" '
                       f'rgba="{rgba}" contype="0" conaffinity="0" group="2"/>')
        return out

    def emit(link, depth, out):
        ind = "  " * depth
        for j in children.get(link, []):
            child = j["child"]
            pos = _fmt(j["xyz"])
            quat = _fmt(matrix_to_quat(rpy_to_matrix(j["rpy"])))
            if child == tip_link:
                out.append(f'{ind}<site name="{child}" pos="{pos}" quat="{quat}"/>')
                continue
            out.append(f'{ind}<body name="{child}" pos="{pos}" quat="{quat}">')
            if j["type"] == "revolute":
                out.append(
                    f'{ind}  <joint name="{j["name"]}" type="hinge" axis="{_fmt(j["axis"])}" '
                    f'range="{_fmt([j["lower"], j["upper"]])}" limited="true"/>'
                )
            elif j["type"] != "fixed":
                raise ValueError(f"joint {j['name']} type {j['type']} not handled")
            out.append(inertial(child, ind + "  "))
            out.extend(visual_geoms(child, ind + "  "))
            emit(child, depth + 1, out)
            out.append(f"{ind}</body>")

    root = roots[0]
    body = [f'    <body name="{root}">', inertial(root, "      ")]
    body.extend(visual_geoms(root, "      "))
    emit(root, 3, body)
    body.append("    </body>")
    head = [
        '<mujoco model="%s">' % model_name,
        "  <!-- %s -->" % source_note,
        '  <compiler angle="radian" autolimits="true"%s/>' % (' meshdir="meshes"' if with_visuals else ""),
    ]
    if with_visuals:
        stems = []
        for vis in parsed["visuals"].values():
            for v in vis:
                if v["mesh_file"][:-4] not in stems:
                    stems.append(v["mesh_file"][:-4])
        head.append("  <asset>")
        head.extend(f'    <mesh name="{s}" file="{s}.stl"/>' for s in stems)
        head.append("  </asset>")
    head.append("  <worldbody>")
    tail = ["  </worldbody>", "</mujoco>", ""]
    return "\n".join(head + body + tail)

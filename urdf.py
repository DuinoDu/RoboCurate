"""Minimal URDF reader: enough of the format to draw a robot, nothing more.

The 3D panel needs a kinematic tree and a visual shape per link. That is a
small corner of URDF, so this parses it with the standard library instead of
pulling in a robotics stack the rest of the tool does not need.

Deliberately ignored: inertial, collision, transmission, gazebo, mimic joints
and `<xacro:*>`. A file needing any of those is not a plain URDF and should be
expanded before it reaches here.

The output is JSON-ready and consumed by `static/robot3d.js`, which does the
forward kinematics in the browser:

    {"root": "pelvis",
     "links": [{"name": ..., "visuals": [{"kind": "mesh", "file": ...,
                                          "xyz": [3], "rpy": [3],
                                          "scale": [3], "color": [4]}]}],
     "joints": [{"name": ..., "type": "revolute", "parent": ..., "child": ...,
                 "xyz": [3], "rpy": [3], "axis": [3], "limit": [lo, hi]}]}

`rpy` keeps URDF's fixed-axis convention (R = Rz·Ry·Rx); the viewer applies it
as a `THREE.Euler(r, p, y, "ZYX")`, which was checked against that product
rather than assumed.
"""

from __future__ import annotations

import os
import xml.etree.ElementTree as ET
from pathlib import Path

# Only these reach the browser. Anything else in a <geometry> is skipped with a
# note rather than silently drawn as the wrong shape.
MESH_SUFFIXES = {".stl", ".dae", ".obj"}

DEFAULT_COLOR = (0.75, 0.75, 0.78, 1.0)


def _floats(text: str | None, default: tuple[float, ...]) -> list[float]:
    if not text:
        return list(default)
    parts = text.replace(",", " ").split()
    try:
        vals = [float(p) for p in parts]
    except ValueError:
        return list(default)
    return vals if len(vals) == len(default) else list(default)


def _origin(node: ET.Element | None) -> tuple[list[float], list[float]]:
    """`<origin xyz rpy>` with URDF's default of identity when absent."""
    el = None if node is None else node.find("origin")
    if el is None:
        return [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]
    return (
        _floats(el.get("xyz"), (0.0, 0.0, 0.0)),
        _floats(el.get("rpy"), (0.0, 0.0, 0.0)),
    )


def _materials(root: ET.Element) -> dict[str, list[float]]:
    """Top-level `<material>` definitions, which links reference by name."""
    out: dict[str, list[float]] = {}
    for mat in root.findall("material"):
        name = mat.get("name")
        color = mat.find("color")
        if name and color is not None:
            out[name] = _floats(color.get("rgba"), DEFAULT_COLOR)
    return out


def _resolve_mesh(filename: str, urdf_dir: Path) -> Path | None:
    """Turn a URDF mesh reference into a real path.

    Handles the two spellings that appear in practice: a path relative to the
    URDF, and `package://pkg/rest`. For the latter the package name is dropped
    and `rest` is tried against the URDF directory and its parent, which is
    where a `*_description` package keeps its meshes.
    """
    ref = filename
    if ref.startswith("package://"):
        rest = ref[len("package://") :]
        ref = rest.split("/", 1)[1] if "/" in rest else rest
    elif ref.startswith("file://"):
        ref = ref[len("file://") :]

    candidates = [urdf_dir / ref, urdf_dir.parent / ref]
    if Path(ref).is_absolute():
        candidates.insert(0, Path(ref))
    for cand in candidates:
        if cand.is_file():
            return cand.resolve()
    return None


def _visuals(
    link: ET.Element, urdf_dir: Path, materials: dict[str, list[float]]
) -> tuple[list[dict], list[str]]:
    out: list[dict] = []
    warnings: list[str] = []
    for vis in link.findall("visual"):
        geom = vis.find("geometry")
        if geom is None:
            continue
        xyz, rpy = _origin(vis)

        color = list(DEFAULT_COLOR)
        mat = vis.find("material")
        if mat is not None:
            own = mat.find("color")
            if own is not None:
                color = _floats(own.get("rgba"), DEFAULT_COLOR)
            elif mat.get("name") in materials:
                color = materials[mat.get("name")]

        shape: dict | None = None
        mesh = geom.find("mesh")
        box = geom.find("box")
        cyl = geom.find("cylinder")
        sph = geom.find("sphere")
        if mesh is not None:
            path = _resolve_mesh(mesh.get("filename", ""), urdf_dir)
            if path is None:
                warnings.append(f"missing mesh {mesh.get('filename')!r}")
                continue
            if path.suffix.lower() not in MESH_SUFFIXES:
                warnings.append(f"unsupported mesh type {path.name!r}")
                continue
            shape = {
                "kind": "mesh",
                "file": path.name,
                "scale": _floats(mesh.get("scale"), (1.0, 1.0, 1.0)),
            }
        elif box is not None:
            shape = {
                "kind": "box",
                "size": _floats(box.get("size"), (0.1,) * 3),
            }
        elif cyl is not None:
            shape = {
                "kind": "cylinder",
                "radius": float(cyl.get("radius", 0.05)),
                "length": float(cyl.get("length", 0.1)),
            }
        elif sph is not None:
            shape = {
                "kind": "sphere",
                "radius": float(sph.get("radius", 0.05)),
            }
        else:
            warnings.append(f"link {link.get('name')!r} has no known geometry")
            continue

        shape.update({"xyz": xyz, "rpy": rpy, "color": color})
        out.append(shape)
    return out, warnings


def parse_urdf(path: Path) -> dict:
    """Read `path` into the JSON structure documented at module level.

    Raises ValueError when the file is not a URDF or has no single root link,
    because a forest cannot be posed as one robot.
    """
    root = ET.parse(path).getroot()
    if root.tag != "robot":
        raise ValueError(
            f"{path.name}: root element is <{root.tag}>, not <robot>"
        )

    urdf_dir = path.parent.resolve()
    materials = _materials(root)
    warnings: list[str] = []

    links = []
    mesh_files: set[str] = set()
    for link in root.findall("link"):
        name = link.get("name")
        if not name:
            continue
        visuals, warns = _visuals(link, urdf_dir, materials)
        warnings.extend(warns)
        for v in visuals:
            if v["kind"] == "mesh":
                mesh_files.add(v["file"])
        links.append({"name": name, "visuals": visuals})

    joints = []
    children: set[str] = set()
    for joint in root.findall("joint"):
        parent_el = joint.find("parent")
        child_el = joint.find("child")
        if parent_el is None or child_el is None:
            continue
        xyz, rpy = _origin(joint)
        axis_el = joint.find("axis")
        limit_el = joint.find("limit")
        child = child_el.get("link")
        children.add(child)
        joints.append(
            {
                "name": joint.get("name"),
                "type": joint.get("type", "fixed"),
                "parent": parent_el.get("link"),
                "child": child,
                "xyz": xyz,
                "rpy": rpy,
                "axis": _floats(
                    None if axis_el is None else axis_el.get("xyz"),
                    (1.0, 0.0, 0.0),
                ),
                "limit": (
                    None
                    if limit_el is None
                    else [
                        float(limit_el.get("lower", 0.0)),
                        float(limit_el.get("upper", 0.0)),
                    ]
                ),
            }
        )

    link_names = [link["name"] for link in links]
    roots = [n for n in link_names if n not in children]
    if len(roots) != 1:
        raise ValueError(
            f"{path.name}: expected exactly one root link, "
            f"found {len(roots)}: {roots[:5]}"
        )

    return {
        "name": root.get("name", path.stem),
        "source": str(path),
        "root": roots[0],
        "links": links,
        "joints": joints,
        "mesh_dir": str(_mesh_root(mesh_files, urdf_dir)),
        "mesh_files": sorted(mesh_files),
        "warnings": warnings,
    }


def _mesh_root(mesh_files: set[str], urdf_dir: Path) -> Path:
    """Directory the server is allowed to serve meshes from.

    Every mesh resolves under one directory in the layouts seen here; pinning
    it lets the mesh route accept a bare filename and reject anything else,
    rather than taking a caller-supplied path.
    """
    for candidate in (urdf_dir / "meshes", urdf_dir):
        if all((candidate / f).is_file() for f in mesh_files):
            return candidate.resolve()
    return urdf_dir


def find_default_urdf(start: Path) -> Path | None:
    """Locate a G1 description, in decreasing order of explicitness.

    1. ``$MCAP_VIZ_URDF`` — an operator pointing at a specific description
       always wins over anything discovered.
    2. ``robot/`` beside this tool — the vendored copy, which is what makes
       the viewer self-contained now that it no longer lives inside the
       collection stack's checkout.
    3. A ``unitree_ros`` submodule above ``start`` — retained so the tool
       still works when dropped back inside the collection stack's checkout,
       where no vendored copy exists.

    Absent description => no 3D view, not an error.
    """
    env = os.environ.get("MCAP_VIZ_URDF")
    if env:
        cand = Path(env).expanduser()
        # An explicit request that cannot be honoured is worth failing on:
        # silently falling through would draw the wrong robot.
        if not cand.is_file():
            raise ValueError(f"MCAP_VIZ_URDF is not a file: {cand}")
        return cand.resolve()

    vendored = Path(__file__).resolve().parent / "robot"
    if vendored.is_dir():
        for cand in sorted(vendored.glob("*.urdf")):
            return cand.resolve()

    rel = Path("thirdparties/unitree_ros/robots/g1_description")
    for base in [start, *start.parents]:
        for stem in ("g1_29dof_rev_1_0", "g1_29dof"):
            cand = base / rel / f"{stem}.urdf"
            if cand.is_file():
                return cand.resolve()
    return None

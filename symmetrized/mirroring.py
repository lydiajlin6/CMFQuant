"""Which side is right, and the symmetrized landmark set (the kept side copied plus its mirror image)."""

import numpy as np

import slicer

from .settings import MIRROR_ATTRIBUTE, MIRROR_PREFIX, SIDES
from .scene import _case_positions, _log, _other_side, _remove_own, _unique_node_name, _unit, landmark_position
from .planes import plane_geometry


def reflection_matrix(origin, normal):
    """4 x 4 matrix of the reflection across the plane through origin with this normal: p -> p - 2((p - o).n)n."""
    n = _unit(normal)
    o = np.asarray(origin, dtype=float)
    m = np.eye(4)
    m[:3, :3] -= 2.0 * np.outer(n, n)
    m[:3, 3] = 2.0 * float(np.dot(o, n)) * n
    return m


def paired_names(names):
    """[(R name, L name)] for the landmarks that come in pairs: "R" + rest and "L" + rest both present."""
    names = set(names)
    return [(n, "L" + n[1:]) for n in sorted(names) if n.startswith("R") and len(n) > 1 and "L" + n[1:] in names]


def right_side_normal(planeNode, case):
    """(unit normal of the plane pointing to the patient's right, how that was decided).

    From the paired landmarks of case when there are any: the side where the R landmarks lie on average. Otherwise
    world +x (right in RAS).
    """
    origin, normal = plane_geometry(planeNode)
    pairs = paired_names(case["names"])
    if pairs:
        signed = lambda name: float(np.dot(_case_positions(case, [name])[0] - origin, normal))
        lean = float(np.mean([signed(r) - signed(l) for r, l in pairs]))
        if abs(lean) > 1e-6:
            return (normal if lean > 0 else -normal), f"from {len(pairs)} R/L landmark pair(s)"
    if abs(normal[0]) < 1e-6:
        raise ValueError("Cannot tell which side of the sagittal plane is right: there are no R/L landmark pairs and the "
                         "plane is parallel to the world x axis.")
    return (normal if normal[0] > 0 else -normal), "from the world +x direction (no R/L landmark pairs)"


def mirror_landmarks(case, planeNode, kept_side, exclude=(), prefix=MIRROR_PREFIX):
    """A new landmark set "<prefix><set name>": each landmark of case on kept_side (name starting with "R" or "L"),
    except exclude, copied as it is and followed by its mirror image across the plane, labeled for the side it now lies
    on (keep R: RZf, then its mirror LZf). The other side and landmarks without a side letter are not used. It replaces
    one made earlier; its positions are world positions (no parent transform).
    """
    other = _other_side(kept_side)
    origin, normal = plane_geometry(planeNode)
    m = reflection_matrix(origin, normal)
    name = f"{prefix}{case['node'].GetName()}"
    if not _remove_own(name, MIRROR_ATTRIBUTE):
        name = _unique_node_name(name)
    node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsFiducialNode", name)
    node.SetAttribute(MIRROR_ATTRIBUTE, case["node"].GetName())
    node.SetAttribute("MirroredAcross", planeNode.GetName())
    node.SetAttribute("KeptSide", SIDES[kept_side])
    skipped, mirrored_from = [], []
    for label, index in case["names"].items():
        if label in exclude or not (label.startswith(kept_side) and len(label) > 1):
            skipped.append(label)
            continue
        p = landmark_position(case["node"], index)
        node.AddControlPoint(list(p), label)                                   # the kept landmark, copied
        node.AddControlPoint(list((m @ np.append(p, 1.0))[:3]), other + label[1:])   # its mirror image
        mirrored_from.append(label)
    node.SetLocked(True)
    display = node.GetDisplayNode()
    if display is not None:
        display.SetSelectedColor(0.2, 0.6, 1.0)
    _log(f"{name}: {len(mirrored_from)} {SIDES[kept_side].lower()} landmark(s) copied and mirrored across {planeNode.GetName()} "
         f"to the {SIDES[other].lower()} ({', '.join(mirrored_from)}); not used: {', '.join(skipped) or 'none'}")
    return node


def side_check(case, planeNode, right_normal):
    """Notes for landmarks named R... / L... (paired) that lie on the other side of the sagittal plane."""
    origin = plane_geometry(planeNode)[0]
    notes = []
    for r, l in paired_names(case["names"]):
        for name, sign in ((r, 1.0), (l, -1.0)):
            d = float(np.dot(_case_positions(case, [name])[0] - origin, right_normal))
            if d * sign < 0:
                notes.append(f"{case['case']}: {name} is on the {'left' if sign > 0 else 'right'} of the sagittal plane ({d:+.2f} mm).")
    return notes

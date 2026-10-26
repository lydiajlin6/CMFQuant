"""Landmark sets in the scene: which case a set belongs to, landmark names and positions, and small helpers."""

import os

import numpy as np

import slicer

from .settings import _SET_SUFFIX, ALIGN_ATTRIBUTE, MIRROR_ATTRIBUTE, SEPARATORS


class _Missing(ValueError):
    """A case lacks a landmark or plane that a measurement needs."""


log_listeners = []   # functions called with every log message (for tests)


def _log(message):
    print(f"[symmetrized] {message}")
    for listener in log_listeners:
        listener(message)


def _other_side(side):
    return "L" if side == "R" else "R"


def _unit(v):
    v = np.asarray(v, dtype=float)
    return v / np.linalg.norm(v)


def _is_defined(node, index):
    try:
        return node.GetNthControlPointPositionStatus(index) == node.PositionDefined
    except AttributeError:
        return True


def _has_prefix(text, prefix):
    """text starts with prefix as a whole word: the prefix is followed by a separator, or is all of text."""
    return bool(prefix) and text.startswith(prefix) and (len(text) == len(prefix) or text[len(prefix)] in SEPARATORS)


def short_name(label, case=None):
    """A landmark label without its case prefix: "ct01_N" -> "N". Labels without the prefix are left as they are."""
    label = label.strip()
    if case and _has_prefix(label, case):
        return label[len(case):].lstrip(SEPARATORS) or label
    return label


def landmark_names(node, case=None):
    """{control point index: display name} for the points with a position; unlabeled points are "#<n>", and a label
    shared by several points gets "(#<n>)" added. With case given, that case's prefix is removed from the labels first.
    """
    indices = [i for i in range(node.GetNumberOfControlPoints()) if _is_defined(node, i)]
    labels = {i: short_name(node.GetNthControlPointLabel(i), case) for i in indices}
    counts = {}
    for label in labels.values():
        counts[label] = counts.get(label, 0) + 1
    names = {}
    for i in indices:
        label = labels[i]
        if not label:
            names[i] = f"#{i + 1}"
        elif counts[label] > 1:
            names[i] = f"{label} (#{i + 1})"
        else:
            names[i] = label
    return names


def volume_names():
    """Names of the (non-label-map) volumes in the scene: the case names."""
    return [n.GetName() for n in slicer.util.getNodesByClass("vtkMRMLScalarVolumeNode") if not n.IsA("vtkMRMLLabelMapVolumeNode")]


def case_id(node, volumes=None):
    """The case a landmark set belongs to: the volume whose name prefixes the set's name or all its labels; failing
    that the set's name without a trailing "Landmarks"; failing that the labels' shared prefix (3 or more labels);
    failing that the set's own name.
    """
    name = node.GetName()
    labels = [node.GetNthControlPointLabel(i).strip() for i in range(node.GetNumberOfControlPoints()) if _is_defined(node, i)]
    for volume in sorted(volumes if volumes is not None else volume_names(), key=len, reverse=True):
        if _has_prefix(name, volume) or (labels and all(_has_prefix(label, volume) for label in labels)):
            return volume
    stripped = _SET_SUFFIX.sub("", name).rstrip(SEPARATORS)
    if stripped and stripped != name:
        return stripped
    if len(labels) >= 3:
        common = os.path.commonprefix(labels)
        cut = max((common.rfind(c) for c in SEPARATORS), default=-1)
        if cut > 0:
            return common[:cut]
    return name


def _case_from_node(node, case):
    names = landmark_names(node, case)
    return {"case": case, "node": node, "names": {name: index for index, name in names.items()},
            "points": {name: (node, index) for index, name in names.items()}}


def find_cases():
    """[{"case", "node", "names", "points"}] for every landmark set with positioned points, in scene order.

    The mirrored set and the _align lists made by this script are not cases. "names" maps each landmark's name without the case prefix to
    its control point index; "points" maps it to (node, index). Two sets that resolve to the same case keep distinct ids
    by falling back to the second set's own name.
    """
    volumes = volume_names()
    cases, used = [], set()
    for node in slicer.util.getNodesByClass("vtkMRMLMarkupsFiducialNode"):
        if node.GetAttribute(MIRROR_ATTRIBUTE) or node.GetAttribute(ALIGN_ATTRIBUTE):
            continue
        case = case_id(node, volumes)
        if case in used:
            case = node.GetName()
        entry = _case_from_node(node, case)
        if not entry["names"]:
            continue
        used.add(case)
        cases.append(entry)
    return cases


def landmark_position(node, index):
    position = [0.0, 0.0, 0.0]
    node.GetNthControlPointPositionWorld(index, position)
    return np.array(position, dtype=float)


def _case_positions(case, names):
    """World positions of a case's landmarks, by name without the case prefix."""
    positions = []
    for name in names:
        if name not in case["points"]:
            raise _Missing(f"{case['case']} has no landmark named {name!r}")
        positions.append(landmark_position(*case["points"][name]))
    return positions


def _unique_node_name(base):
    if slicer.mrmlScene.GetFirstNodeByName(base) is None:
        return base
    n = 2
    while slicer.mrmlScene.GetFirstNodeByName(f"{base}_{n}") is not None:
        n += 1
    return f"{base}_{n}"


def _remove_own(name, attribute):
    """Remove the node of this name if this script made it (it has attribute); True if the name is now free."""
    existing = slicer.mrmlScene.GetFirstNodeByName(name)
    if existing is None:
        return True
    if existing.GetAttribute(attribute):
        slicer.mrmlScene.RemoveNode(existing)
        return True
    return False

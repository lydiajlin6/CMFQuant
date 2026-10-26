"""Plane fitting and markups plane nodes, and the reference planes made from the PRE landmarks."""

import numpy as np

import slicer

from .settings import ORIENTATIONS, PLANE_SIZE_MM, REFERENCE_PLANES, SAGITTAL_PLANE, SUPERIOR_ATTRIBUTE, SUPERIOR_LANDMARK
from .scene import _case_positions, _log, _unique_node_name, _unit


def fit_plane(points):
    """Best-fit plane through 3 or more points: {origin, normal, rms} (mm). For 3 points it is the plane through them,
    normal by the right-hand rule from point 1 to 2 to 3; with more, the least-squares normal, its sign agreeing with
    that rule on the first three points.
    """
    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 3 or len(pts) < 3:
        raise ValueError("A plane needs at least 3 landmarks.")
    centroid = pts.mean(axis=0)
    centered = pts - centroid
    _, singular, vt = np.linalg.svd(centered, full_matrices=False)
    if singular[0] < 1e-6 or singular[1] < 1e-3 * singular[0]:
        raise ValueError("These landmarks are (nearly) in a line, or the same point, so they do not define a plane.")
    normal = vt[2]
    reference = np.cross(pts[1] - pts[0], pts[2] - pts[0])
    if np.linalg.norm(reference) > 1e-9 and np.dot(normal, reference) < 0:
        normal = -normal
    distances = centered @ normal
    return {"origin": centroid, "normal": normal, "rms": float(np.sqrt(np.mean(distances ** 2)))}


def fit_plane_perpendicular_to(points, other_normal):
    """Best-fit plane through 3 or more points among the planes perpendicular to a plane with normal other_normal:
    {origin, normal, rms}. The plane contains the direction other_normal, so its normal is perpendicular to it; of those,
    the one with the smallest sum of squared distances to the points (through their centroid). The normal's sign
    follows the right-hand rule on the first three points, as in fit_plane.
    """
    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 3 or len(pts) < 3:
        raise ValueError("A plane needs at least 3 landmarks.")
    o = _unit(other_normal)
    hint = np.array([1.0, 0.0, 0.0]) if abs(o[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = _unit(hint - np.dot(hint, o) * o)
    v = np.cross(o, u)
    centroid = pts.mean(axis=0)
    centered = pts - centroid
    projected = np.column_stack([centered @ u, centered @ v])   # the components the normal can have
    values, vectors = np.linalg.eigh(projected.T @ projected)
    if values[1] < 1e-6:
        raise ValueError("These landmarks lie on a line along the other plane's normal, so they do not define a plane.")
    a, b = vectors[:, 0]                                         # smallest eigenvalue
    normal = _unit(a * u + b * v)
    reference = np.cross(pts[1] - pts[0], pts[2] - pts[0])
    if np.linalg.norm(reference) > 1e-9 and np.dot(normal, reference) < 0:
        normal = -normal
    distances = centered @ normal
    return {"origin": centroid, "normal": normal, "rms": float(np.sqrt(np.mean(distances ** 2)))}


def existing_planes():
    return list(slicer.util.getNodesByClass("vtkMRMLMarkupsPlaneNode"))


def plane_names():
    return [node.GetName() for node in existing_planes()]


def get_plane(name):
    node = slicer.mrmlScene.GetFirstNodeByName(name)
    return node if node is not None and node.IsA("vtkMRMLMarkupsPlaneNode") else None


def plane_geometry(planeNode):
    """(origin, unit normal) of a markups plane in world coordinates."""
    origin, normal = [0.0] * 3, [0.0] * 3
    planeNode.GetOriginWorld(origin)
    planeNode.GetNormalWorld(normal)
    return np.array(origin, dtype=float), _unit(normal)


def plane_axes(planeNode):
    """(origin, x, y, z) of a markups plane in world coordinates: unit in-plane axes x and y, and the normal z."""
    x, y, z = [0.0] * 3, [0.0] * 3, [0.0] * 3
    planeNode.GetAxesWorld(x, y, z)
    origin = [0.0] * 3
    planeNode.GetOriginWorld(origin)
    return np.array(origin, dtype=float), _unit(x), _unit(y), _unit(z)


def perpendicular_to_planes_geometry(point, normal1, normal2):
    """(origin, normal) of the plane through point perpendicular to two planes: normal along their line of
    intersection, signed so its largest component is positive (anterior for a plane across a head in RAS)."""
    cross = np.cross(np.asarray(normal1, dtype=float), np.asarray(normal2, dtype=float))
    length = float(np.linalg.norm(cross))
    if length < 1e-3:
        raise ValueError("The two planes are parallel, so there is no single plane perpendicular to both.")
    normal = cross / length
    if normal[int(np.argmax(np.abs(normal)))] < 0:
        normal = -normal
    return np.asarray(point, dtype=float), normal


def _build_plane_node(name, origin, normal, axis_hint=None):
    """Add a 3-point markups plane through origin with this normal (PLANE_SIZE_MM square, no axes/handles/arrow).

    A new plane node defaults to the point-normal type, which ignores 3 control points, so the type is set first.
    """
    normal = _unit(normal)
    origin = np.asarray(origin, dtype=float)
    if axis_hint is None:
        axis_hint = np.array([1.0, 0.0, 0.0]) if abs(normal[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    axis1 = np.asarray(axis_hint, dtype=float)
    axis1 = _unit(axis1 - np.dot(axis1, normal) * normal)
    axis2 = np.cross(normal, axis1)

    planeNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsPlaneNode")
    planeNode.SetName(name)
    planeNode.SetPlaneType(slicer.vtkMRMLMarkupsPlaneNode.PlaneType3Points)
    planeNode.RemoveAllControlPoints()
    half = 50.0  # control point spacing only; the drawn size is set below
    planeNode.AddControlPoint(list(origin))
    planeNode.AddControlPoint(list(origin + axis1 * half))
    planeNode.AddControlPoint(list(origin + axis2 * half))
    planeNode.CreateDefaultDisplayNodes()
    planeNode.SetSizeMode(slicer.vtkMRMLMarkupsPlaneNode.SizeModeAbsolute)
    planeNode.SetSize(PLANE_SIZE_MM, PLANE_SIZE_MM)
    displayNode = planeNode.GetDisplayNode()
    if displayNode is not None:
        displayNode.SetTranslationHandleVisibility(False)
        displayNode.SetRotationHandleVisibility(False)
        displayNode.SetScaleHandleVisibility(False)
        displayNode.SetNormalVisibility(False)
    return planeNode


def _node_name_for(base, replace):
    """A name for a new plane: a taken name gets "_2", ... unless replace is set; then a plane made by this script (or
    the shared/per-case plane scripts: it has a "SourceLandmarkSet") of that name is replaced, anything else is an error.
    """
    existing = slicer.mrmlScene.GetFirstNodeByName(base)
    if existing is None:
        return base
    if not replace:
        return _unique_node_name(base)
    if existing.IsA("vtkMRMLMarkupsPlaneNode") and existing.GetAttribute("SourceLandmarkSet"):
        slicer.mrmlScene.RemoveNode(existing)
        return base
    raise ValueError(f"A node named {base!r} already exists and was not made by this script.")


def _next_default_plane_name():
    n = 1
    while slicer.mrmlScene.GetFirstNodeByName(f"Plane_{n}") is not None:
        n += 1
    return f"Plane_{n}"


def create_plane_from_points(name, points, labels, source, replace=False):
    """A plane through (3) or best-fit to (more) these world positions; labels and source are recorded on the node."""
    if len(set(labels)) != len(labels):
        raise ValueError("The same landmark was chosen more than once.")
    fit = fit_plane(points)
    planeNode = _build_plane_node(_node_name_for(name or _next_default_plane_name(), replace), fit["origin"], fit["normal"])
    planeNode.SetAttribute("SourceLandmarkSet", source)
    planeNode.SetAttribute("SourceLandmarks", ", ".join(labels))
    planeNode.SetAttribute("FitRMS_mm", f"{fit['rms']:.4f}")
    _log(f"{planeNode.GetName()}: {len(labels)} landmark(s) ({', '.join(labels)}), RMS distance to plane {fit['rms']:.3f} mm")
    return planeNode


def create_plane(definition, template, replace=False):
    """Create one plane from the template case's landmarks (its own and, if added, the mirrored ones).

    definition is {"name", "type": "landmarks" | "landmarks_orthogonal" | "orthogonal" | "orthogonal2",
    "landmarks": [names], "orthogonal_to": plane name for landmarks_orthogonal (best fit among the planes perpendicular
    to it), "reference" + "orientation" ("x"/"y") for orthogonal, "references": [2 plane names] for orthogonal2}.
    """
    labels = list(definition["landmarks"])
    points = _case_positions(template, labels)
    source = template["node"].GetName()
    name = definition["name"]
    if definition["type"] == "landmarks":
        return create_plane_from_points(name, points, labels, source, replace)
    if definition["type"] == "landmarks_orthogonal":
        other = get_plane(definition["orthogonal_to"])
        if other is None:
            raise ValueError(f"There is no plane named {definition['orthogonal_to']!r}")
        if len(set(labels)) != len(labels):
            raise ValueError("The same landmark was chosen more than once.")
        fit = fit_plane_perpendicular_to(points, plane_geometry(other)[1])
        planeNode = _build_plane_node(_node_name_for(name or _next_default_plane_name(), replace), fit["origin"], fit["normal"])
        planeNode.SetAttribute("SourceLandmarkSet", source)
        planeNode.SetAttribute("SourceLandmarks", ", ".join(labels))
        planeNode.SetAttribute("OrthogonalTo", other.GetName())
        planeNode.SetAttribute("FitRMS_mm", f"{fit['rms']:.4f}")
        _log(f"{planeNode.GetName()}: best fit to {', '.join(labels)} perpendicular to {other.GetName()}, "
             f"RMS distance to plane {fit['rms']:.3f} mm")
        return planeNode
    if definition["type"] == "orthogonal2":
        planes = [get_plane(n) for n in definition["references"]]
        for n, node in zip(definition["references"], planes):
            if node is None:
                raise ValueError(f"There is no plane named {n!r}")
        normal_a, normal_b = plane_geometry(planes[0])[1], plane_geometry(planes[1])[1]
        origin, normal = perpendicular_to_planes_geometry(points[0], normal_a, normal_b)
        planeNode = _build_plane_node(_node_name_for(name or _next_default_plane_name(), replace), origin, normal, axis_hint=normal_a)
        planeNode.SetAttribute("OrthogonalTo", f"{planes[0].GetName()}, {planes[1].GetName()}")
        description = f"perpendicular to {planes[0].GetName()} and {planes[1].GetName()}"
    else:
        reference = get_plane(definition["reference"])
        if reference is None:
            raise ValueError(f"There is no plane named {definition['reference']!r}")
        orientation = definition["orientation"]
        if orientation not in ORIENTATIONS:
            raise ValueError(f"Orientation must be one of {list(ORIENTATIONS)}, got {orientation!r}.")
        _, x, y, z = plane_axes(reference)
        planeNode = _build_plane_node(_node_name_for(name or _next_default_plane_name(), replace), points[0],
                                      y if orientation == "x" else x, axis_hint=z)
        planeNode.SetAttribute("OrthogonalTo", reference.GetName())
        planeNode.SetAttribute("Orientation", f"parallel to {orientation} of {reference.GetName()}")
        description = f"perpendicular to {reference.GetName()}, parallel to its {orientation} direction"
    planeNode.SetAttribute("SourceLandmarkSet", source)
    planeNode.SetAttribute("SourceLandmarks", labels[0])
    _log(f"{planeNode.GetName()}: {description}, through {labels[0]}")
    return planeNode


def _rename_sagittal(name, sagittal_name):
    return sagittal_name if name == SAGITTAL_PLANE else name


def mark_superior_side(planeNode, point, min_distance_mm=1.0):
    """Record on the plane which of its sides is superior: the side point (a superior landmark, such as CG) is on, as
    "+1" (the normal points superior) or "-1". Not recorded, and None returned, when the point lies within
    min_distance_mm of the plane (it cannot tell the sides apart, as for the sagittal plane and CG)."""
    origin, normal = plane_geometry(planeNode)
    distance = float(np.dot(np.asarray(point, dtype=float) - origin, normal))
    if abs(distance) < min_distance_mm:
        return None
    sign = 1 if distance > 0 else -1
    planeNode.SetAttribute(SUPERIOR_ATTRIBUTE, str(sign))
    return sign


def superior_sign(planeNode):
    """+1 if the plane's normal points to its superior side, -1 if not: as recorded by mark_superior_side, else from world
    superior (+z in RAS)."""
    recorded = planeNode.GetAttribute(SUPERIOR_ATTRIBUTE)
    if recorded in ("1", "+1", "-1"):
        return int(recorded)
    return 1 if plane_geometry(planeNode)[1][2] >= 0 else -1


def make_reference_planes(pre, sagittal=None):
    """Make REFERENCE_PLANES from the PRE case's landmarks (replacing ones made earlier).

    sagittal (a plane name) uses that existing plane instead of making SAGITTAL_PLANE. Returns ({name: plane node},
    sagittal plane node, [landmarks the sagittal plane was made from], [problems]). The sagittal plane is needed for
    everything else, so failing to make it is an error; if the plane it should be perpendicular to cannot be made, it is
    an unconstrained best fit instead (reported). Other planes that cannot be made are reported and skipped.
    """
    made, problems, failed = {}, [], set()
    sagittal_node, sagittal_landmarks = None, []

    def drop_stale(name):   # a plane of this name made by an earlier run must not be measured against
        old = get_plane(name)
        if old is not None and old.GetAttribute("SourceLandmarkSet"):
            slicer.mrmlScene.RemoveNode(old)

    sagittal_name = SAGITTAL_PLANE
    if sagittal is not None:
        sagittal_node = get_plane(sagittal)
        if sagittal_node is None:
            raise ValueError(f"There is no plane named {sagittal!r}. Planes: {plane_names()}")
        sagittal_name = sagittal_node.GetName()
        sagittal_landmarks = [x.strip() for x in (sagittal_node.GetAttribute("SourceLandmarks") or "").split(",") if x.strip()]
        made[sagittal_name] = sagittal_node
    for preset in REFERENCE_PLANES:
        name = preset["name"]
        is_sagittal = name == SAGITTAL_PLANE
        if is_sagittal and sagittal_node is not None:
            continue
        needs = [_rename_sagittal(n, sagittal_name) for n in
                 (preset.get("references") or ([preset["reference"]] if preset.get("reference") else []))]
        unmade = [n for n in needs if n in failed]
        if unmade:
            failed.add(name)
            drop_stale(name)
            problems.append(f"Plane {name} not made: it needs the plane {' and '.join(unmade)}, which could not be made.")
            continue
        landmarks = list(preset["landmarks"])
        containing = preset.get("also_containing")
        if containing:
            landmarks += [n for n in pre["names"] if containing in n and n not in landmarks]
        kind = preset["type"]
        if kind == "landmarks_orthogonal" and preset["orthogonal_to"] in failed:
            kind = "landmarks"
            problems.append(f"Plane {name} is an unconstrained best fit to {', '.join(landmarks)}: the plane "
                            f"{preset['orthogonal_to']} it should be perpendicular to could not be made.")
        definition = {"name": name, "type": kind, "landmarks": landmarks, "orthogonal_to": preset.get("orthogonal_to"),
                      "reference": _rename_sagittal(preset.get("reference"), sagittal_name),
                      "orientation": preset.get("orientation"), "references": needs if preset.get("references") else None}
        try:
            made[name] = create_plane(definition, pre, replace=True)
            if SUPERIOR_LANDMARK in pre["points"]:
                mark_superior_side(made[name], _case_positions(pre, [SUPERIOR_LANDMARK])[0])
        except ValueError as exc:
            if is_sagittal:
                raise ValueError(f"The sagittal plane could not be made: {exc}") from exc
            failed.add(name)
            drop_stale(name)
            problems.append(f"Plane {name} not made: {exc}")
            continue
        if is_sagittal:
            sagittal_node, sagittal_landmarks = made[name], landmarks
    return made, sagittal_node, sagittal_landmarks, problems

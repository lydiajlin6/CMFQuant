"""Measurements (distances, angles, to planes), the presets, and the terminal and CSV output."""

import csv
import math
import os

import numpy as np

import slicer

from .settings import preset_measurement_specs, SAGITTAL_PLANE
from .scene import _case_positions, _log, _Missing
from .planes import _rename_sagittal, get_plane, plane_geometry, superior_sign
from .dialogs import _ask_csv_path


def preset_measurements(cases, sagittal_name=SAGITTAL_PLANE):
    """The preset measurements that can be calculated: [specs], [problems]. One is kept when at least one landmark set
    has all its landmarks and its plane exists; sets without them get blank cells. SAGITTAL_PLANE is read as
    sagittal_name."""
    specs, problems = [], []
    for spec in preset_measurement_specs():
        spec["plane"] = _rename_sagittal(spec["plane"], sagittal_name)
        if not any(all(n in c["points"] for n in spec["landmarks"]) for c in cases):
            problems.append(f"Measurement {spec['name']} skipped: no landmark set has {', '.join(spec['landmarks'])}.")
        elif spec["plane"] and get_plane(spec["plane"]) is None:
            problems.append(f"Measurement {spec['name']} skipped: the plane {spec['plane']} is not available.")
        else:
            specs.append(spec)
    return specs, problems


def _angle_deg(a, vertex, b):
    v1, v2 = np.asarray(a, dtype=float) - vertex, np.asarray(b, dtype=float) - vertex
    norm = np.linalg.norm(v1) * np.linalg.norm(v2)
    if norm == 0:
        return None
    return math.degrees(math.acos(max(-1.0, min(1.0, float(np.dot(v1, v2) / norm)))))


def measure(kind, points, labels=None):
    """Distance (2 points; with dx, dy, dz = point 2 minus point 1, world RAS) or angle at the middle of 3 points."""
    pts = [np.asarray(p, dtype=float) for p in points]
    labels = list(labels) if labels else [f"point {i + 1}" for i in range(len(pts))]
    if kind == "distance":
        if len(pts) != 2:
            raise ValueError("A distance needs exactly 2 landmarks.")
        dx, dy, dz = (float(v) for v in pts[1] - pts[0])
        return {"type": "distance", "landmarks": labels, "value": math.dist(*pts), "unit": "mm", "dx": dx, "dy": dy, "dz": dz}
    if kind == "angle":
        if len(pts) != 3:
            raise ValueError("An angle needs exactly 3 landmarks.")
        value = _angle_deg(*pts)
        if value is None:
            raise ValueError("Two of the landmarks coincide with the vertex, so there is no angle.")
        return {"type": "angle", "landmarks": labels, "value": value, "unit": "deg"}
    raise ValueError(f"Unknown measurement type {kind!r}; use 'distance' or 'angle'.")


def measure_angle_to_plane(points, planeNode, labels=None):
    """Angle at the vertex (points[1]) between the end point (points[0]) and the perpendicular from the vertex to the
    plane (0 to 180 degrees); undefined when the vertex is on the plane or the two coincide."""
    end, vertex = [np.asarray(p, dtype=float) for p in points]
    origin, normal = plane_geometry(planeNode)
    signed = float(np.dot(vertex - origin, normal))
    if abs(signed) < 1e-9:
        raise ValueError("The vertex lies on the plane, so there is no perpendicular line from it to the plane.")
    to_end = end - vertex
    if np.linalg.norm(to_end) < 1e-9:
        raise ValueError("The two landmarks coincide, so there is no line between them.")
    to_plane = -signed * normal
    cosine = float(np.dot(to_end, to_plane) / (np.linalg.norm(to_end) * np.linalg.norm(to_plane)))
    return {"type": "plane_angle", "landmarks": list(labels) if labels else ["end point", "vertex"], "plane": planeNode.GetName(),
            "value": math.degrees(math.acos(max(-1.0, min(1.0, cosine)))), "vertex_to_plane_mm": abs(signed), "unit": "deg"}


def measure_line_to_plane_angle(points, planeNode, labels=None):
    """Angle between the line through two landmarks and a plane (0 parallel .. 90 along the normal), plus signed_deg
    (positive when the line from landmark 1 to 2 rises toward the normal's side)."""
    pts = [np.asarray(p, dtype=float) for p in points]
    direction = pts[1] - pts[0]
    length = float(np.linalg.norm(direction))
    if length < 1e-9:
        raise ValueError("The two landmarks coincide, so they do not form a line.")
    _, normal = plane_geometry(planeNode)
    signed = math.degrees(math.asin(max(-1.0, min(1.0, float(np.dot(direction, normal)) / length))))
    return {"type": "line_plane_angle", "landmarks": list(labels) if labels else ["point 1", "point 2"], "plane": planeNode.GetName(),
            "value": abs(signed), "signed_deg": signed, "unit": "deg"}


def measure_to_plane(point, planeNode, label=None):
    """Perpendicular distance from a landmark to a plane; signed_mm is positive on the normal's side."""
    origin, normal = plane_geometry(planeNode)
    signed = float(np.dot(np.asarray(point, dtype=float) - origin, normal))
    return {"type": "plane_distance", "landmarks": [label or "landmark"], "plane": planeNode.GetName(),
            "value": abs(signed), "signed_mm": signed, "unit": "mm"}


def describe(result):
    names = result["landmarks"]
    label = f"{result['name']}: " if result.get("name") else ""
    if result["type"] == "difference":
        return (f"{label}{result['minuend'][0]} {result['minuend'][1]:.2f} mm - {result['subtrahend'][0]} "
                f"{result['subtrahend'][1]:.2f} mm = {result['value']:+.2f} mm")
    if result["type"] == "distance":
        return (f"{label}Distance {names[0]} to {names[1]}: {result['value']:.2f} mm "
                f"(dx {result['dx']:+.2f}, dy {result['dy']:+.2f}, dz {result['dz']:+.2f} mm)")
    if result["type"] == "plane_distance":
        side = "on the normal's side" if result["signed_mm"] > 0 else ("on the opposite side" if result["signed_mm"] < 0 else "on the plane")
        return f"{label}Distance from {names[0]} to plane {result['plane']} (perpendicular to the plane): {result['value']:.2f} mm, {side}"
    if result["type"] == "line_plane_angle":
        return (f"{label}Angle between the line {names[0]} - {names[1]} and plane {result['plane']}: "
                f"{result['value']:.2f} deg ({result['signed_deg']:+.2f} deg signed)")
    if result["type"] == "plane_angle":
        return f"{label}Angle at {names[1]} between {names[0]} and the perpendicular to plane {result['plane']}: {result['value']:.2f} deg"
    return f"{label}Angle {names[0]} - {names[1]} - {names[2]} (at {names[1]}): {result['value']:.2f} deg"


_VALUE_KEYS = {
    "distance": ["value", "dx", "dy", "dz"],
    "angle": ["value"],
    "plane_distance": ["value", "signed_mm"],
    "plane_angle": ["value"],
    "line_plane_angle": ["value", "signed_deg"],
}


def column_headers(spec):
    lm, name, plane = spec["landmarks"], spec.get("name"), spec.get("plane")
    kind = spec["kind"]
    if kind == "distance":
        base = name or f"{lm[0]}-{lm[1]}"
        return [f"{base} distance (mm)", f"{base} dx (mm)", f"{base} dy (mm)", f"{base} dz (mm)"]
    if kind == "angle":
        return [f"{name} angle (deg)" if name else f"{lm[0]}-{lm[1]}-{lm[2]} angle at {lm[1]} (deg)"]
    if kind == "plane_distance":
        base = name or f"{lm[0]} to plane {plane}"
        return [f"{base} distance (mm)", f"{base} signed distance (mm)"]
    if kind == "line_plane_angle":
        base = name or f"{lm[0]}-{lm[1]} line to plane {plane}"
        return [f"{base} angle (deg)", f"{base} signed angle (deg)"]
    return [f"{name} angle (deg)" if name else f"{lm[0]}-{lm[1]} angle to the perpendicular to plane {plane} (deg)"]


def spec_title(spec):
    lm, plane, kind = spec["landmarks"], spec.get("plane"), spec["kind"]
    if kind == "distance":
        text = f"Distance {lm[0]} to {lm[1]}"
    elif kind == "angle":
        text = f"Angle {lm[0]} - {lm[1]} - {lm[2]}"
    elif kind == "plane_distance":
        text = f"Distance from {lm[0]} to plane {plane}"
    elif kind == "line_plane_angle":
        text = f"Angle between the line {lm[0]} - {lm[1]} and plane {plane}"
    else:
        text = f"Angle at {lm[1]} between {lm[0]} and the perpendicular to plane {plane}"
    return f"{spec['name']} ({text})" if spec.get("name") else text


def usable_differences(differences, specs):
    names = {spec.get("name") for spec in specs}
    return [d for d in differences if d["minuend"] in names and d["subtrahend"] in names]


def difference_results(differences, specs, results, case_name=None):
    """One result per (usable) difference: minuend value minus subtrahend value (main values), or None."""
    by_name = {}
    for spec, result in zip(specs, results):
        by_name.setdefault(spec.get("name"), result)
    out = []
    for d in differences:
        a, b = by_name[d["minuend"]], by_name[d["subtrahend"]]
        if a is None or b is None:
            out.append(None)
            continue
        out.append({"type": "difference", "name": d["name"], "landmarks": [], "value": a["value"] - b["value"], "unit": "mm",
                    "minuend": (d["minuend"], a["value"]), "subtrahend": (d["subtrahend"], b["value"]), "case": case_name})
    return out


def evaluate(spec, case):
    """Compute one measurement spec ({"kind", "landmarks", "plane", "name"}) in one case. For a line-to-plane angle,
    "signed": "first_superior" makes the signed value positive when the first landmark is superior to the second
    (planes.superior_sign) instead of following the plane's normal.
    Raises _Missing (a ValueError) for a missing landmark or plane, ValueError for degenerate geometry."""
    points = _case_positions(case, spec["landmarks"])
    labels = list(spec["landmarks"])
    kind, base = spec["kind"], spec.get("plane")
    planeNode = None
    if base:
        planeNode = get_plane(base)
        if planeNode is None:
            raise _Missing(f"there is no plane named {base!r}")
    if kind == "plane_distance":
        result = measure_to_plane(points[0], planeNode, labels[0])
    elif kind == "plane_angle":
        result = measure_angle_to_plane(points, planeNode, labels)
    elif kind == "line_plane_angle":
        result = measure_line_to_plane_angle(points, planeNode, labels)
        if spec.get("signed") == "first_superior":
            # positive when the first landmark is superior to the second, whichever way the plane's normal points
            result["signed_deg"] = -superior_sign(planeNode) * result["signed_deg"]
    else:
        result = measure(kind, points, labels)
    result["name"] = spec.get("name") or None
    result["case"] = case["case"]
    return result


def export_measurements_csv(specs, rows, csv_path, differences=()):
    """case_id, then each measurement's columns, then one column per difference; one row per case. rows is
    [(case id, [result dict or None per spec, then per difference])]. Values rounded to 3 decimals; overwrites."""
    if not specs:
        raise ValueError("No measurements to export.")
    headers, seen = ["case_id"], {}
    for header in [h for spec in specs for h in column_headers(spec)] + [d["name"] for d in differences]:
        seen[header] = seen.get(header, 0) + 1
        headers.append(header if seen[header] == 1 else f"{header} ({seen[header]})")
    csv_path = os.path.abspath(csv_path)
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for case, results in rows:
            row = [case]
            for spec, result in zip(specs, results):
                row.extend("" if result is None else round(float(result[key]), 3) for key in _VALUE_KEYS[spec["kind"]])
            for result in results[len(specs):len(specs) + len(differences)]:
                row.append("" if result is None else round(float(result["value"]), 3))
            writer.writerow(row)
    return csv_path


def finalize_measurements(specs, cases, csv_path=None, differences=()):
    """Calculate the measurements in every case (landmark set), print them and save a CSV (one row per case).

    differences add a column each when both their measurements were selected. Returns
    ({case: [result or None per spec, then per difference]}, CSV path or None).
    """
    if not specs:
        return {}, None
    used = usable_differences(differences, specs)
    rows, notes = [], []
    for d in differences:
        if d not in used:
            notes.append(f"{d['name']} not calculated: {d['minuend']} and {d['subtrahend']} were not both selected.")
    for case in cases:
        results = []
        for spec in specs:
            try:
                results.append(evaluate(spec, case))
            except ValueError as exc:
                results.append(None)
                notes.append(f"{case['case']}: left blank for {spec_title(spec)}: {exc}")
        results += difference_results(used, specs, results, case["case"])
        rows.append((case["case"], results))
    _log(f"{len(specs)} measurement(s)" + (f" and {len(used)} calculation(s)" if used else "") +
         f" in {len(cases)} case(s) (world RAS axes: x right+, y anterior+, z superior+; mm / degrees):")
    for case_name, results in rows:
        _log(f"{case_name}:")
        for number, result in enumerate(results, start=1):
            if result is not None:
                _log(f"  {number}. {describe(result)}")
    for note in notes:
        _log(note)
    if csv_path is None:
        url = slicer.mrmlScene.GetURL()
        default_dir = os.path.dirname(url) if url else os.path.expanduser("~")
        csv_path = _ask_csv_path(os.path.join(default_dir, "symmetrized_measurements.csv"))
    if csv_path is None:
        _log("No CSV saved (the save dialog was cancelled).")
        return dict(rows), None
    saved = export_measurements_csv(specs, rows, csv_path, used)
    _log(f"Wrote {len(specs)} measurement(s)" + (f" and {len(used)} calculation(s)" if used else "") + f" for {len(rows)} case(s) to {saved}")
    return dict(rows), saved

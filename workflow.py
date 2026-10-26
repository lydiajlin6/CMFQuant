"""The whole interactive run: landmarks, reference planes, side, model, mirroring, measurements."""

import slicer

from .settings import (
    ALIGN_SUFFIX, FINAL_TRANSFORM_NAME, INITIAL_TRANSFORM_NAME, MEASUREMENT_KINDS, MIRROR_PREFIX, NO_LANDMARKS_MESSAGE,
    PLANE_KINDS, PLANE_TYPES, POST_CASE, POST_SET_NAME, PRE_CASE, PRE_SET_NAME, PRESET_DIFFERENCES, SIDES,
)
from .scene import _case_from_node, _log, find_cases
from .planes import _next_default_plane_name, create_plane, existing_planes, make_reference_planes, plane_names
from .mirroring import mirror_landmarks, right_side_normal, side_check
from .model import create_symmetrized_model, model_sources
from .dialogs import _ask_buttons, _ask_text, _ask_yes_no, _choose_landmarks, _choose_option, ask_side
from .measurements import evaluate, finalize_measurements, preset_measurements, spec_title
from .placement import create_alignment_lists, define_landmarks
from .segmentation import segment_volumes
from .registration import (
    apply_final_align, find_volume, fiducial_registration, intensity_registration, post_under_final_align,
)


class _Abandoned(Exception):
    """The user cancelled part-way through a measurement."""


def generate_planes(template):
    """Create planes from landmark picks on the template (the PRE landmarks) until the user stops."""
    planes = {}
    template_names = list(template["points"])
    retry_orthogonal = None
    while True:
        orthogonal = False
        names_in_scene = plane_names()
        if retry_orthogonal is not None:
            orthogonal, retry_orthogonal = retry_orthogonal, None
        elif names_in_scene:
            plane_type = _choose_option("Plane type", "What kind of plane would you like to make?", PLANE_TYPES)
            if plane_type is None:
                break
            orthogonal = plane_type == 1
        default_name = _next_default_plane_name()
        if orthogonal:
            choice = _choose_landmarks(
                "Create a plane perpendicular to another",
                "Choose the existing plane, which direction of it the new plane runs parallel to, and one landmark "
                "for the new plane to pass through.",
                template_names, min_count=1, single=True, default_name=default_name, plane_bases=names_in_scene, orientation=True)
        else:
            choice = _choose_landmarks(
                "Create a plane",
                "Choose 3 or more landmarks of the preoperative set. Three define the plane through them; "
                "more are fitted with a best-fit plane. The plane is made once and used for every case.",
                template_names, min_count=3, default_name=default_name)
        if choice is None:
            break
        definition = {"name": choice["name"] or default_name, "type": "orthogonal" if orthogonal else "landmarks",
                      "landmarks": choice["names"], "reference": choice["plane"], "orientation": choice["orientation"]}
        try:
            node = create_plane(definition, template)
        except ValueError as exc:
            slicer.util.warningDisplay(str(exc), windowTitle="Could not make that plane")
            retry_orthogonal = orthogonal
            continue
        planes[node.GetName()] = node
        slicer.util.infoDisplay(f"Added plane '{node.GetName()}'.", windowTitle="Plane created")
        if not _ask_yes_no("Generate another plane?", "Planes"):
            break
    return planes


def select_measurements(template):
    """Pick measurements one after another on the template (the PRE landmarks). Returns specs."""
    specs = []
    template_names = list(template["points"])
    while True:
        available = dict(MEASUREMENT_KINDS)
        if existing_planes():
            available.update(PLANE_KINDS)
        kinds = list(available)
        kind_index = _choose_option("Measurement", "What would you like to measure?", kinds)
        if kind_index is None:
            break
        kind, roles = available[kinds[kind_index]]
        spec = {"kind": kind, "landmarks": [], "plane": None, "name": None}
        label = kind.replace("_", " ").capitalize()
        for number, role in enumerate(roles, start=1):
            choice = _choose_landmarks(f"{label}: landmark {number} of {len(roles)}",
                                       f"Choose the landmark for: {role}.", template_names, min_count=1, single=True)
            if choice is None:
                break
            spec["landmarks"].append(choice["names"][0])
        else:
            try:
                if kind in ("plane_distance", "plane_angle", "line_plane_angle"):
                    names_in_scene = plane_names()
                    index = _choose_option(f"{label}: the plane", "Measure against which plane?", names_in_scene)
                    if index is None:
                        raise _Abandoned()
                    spec["plane"] = names_in_scene[index]
                evaluate(spec, template)
                name = _ask_text("Name the measurement", "Name for this measurement (optional; leave blank to name it after its landmarks).")
                if name is None:
                    raise _Abandoned()
                spec["name"] = name or None
            except _Abandoned:
                pass
            except ValueError as exc:
                slicer.util.warningDisplay(str(exc), windowTitle="Could not measure")
            else:
                specs.append(spec)
        queued = f"{len(specs)} measurement(s) selected so far, the latest: {spec_title(specs[-1])}. " if specs else ""
        if not _ask_yes_no(f"{queued}Add another measurement?", "Measurements"):
            break
    return specs


def _choose_pre(cases, pre):
    if pre is not None:
        matches = [c for c in cases if c["case"] == pre]
        if not matches:
            raise ValueError(f"No case named {pre!r}. Cases: {[c['case'] for c in cases]}")
        return matches[0]
    matches = [c for c in cases if c["case"] == PRE_CASE]
    if matches:
        return matches[0]
    if len(cases) == 1:
        return cases[0]
    index = _choose_option("Preoperative landmark set",
                           "Choose the preoperative landmark set. The sagittal plane is made from it, and its landmarks are mirrored.",
                           [f"{c['case']}   (landmark set {c['node'].GetName()}, {len(c['names'])} landmarks)" for c in cases])
    return None if index is None else cases[index]


def _choose_source(source):
    """The model/segmentation node to symmetrize: by name, the only one, or chosen (None if none or cancelled)."""
    nodes = model_sources()
    if source is not None:
        matches = [n for n in nodes if n.GetName() == source]
        if not matches:
            raise ValueError(f"There is no model or segmentation named {source!r}. Available: {[n.GetName() for n in nodes]}")
        return matches[0]
    if len(nodes) <= 1:
        return nodes[0] if nodes else None
    kind = lambda n: "segmentation" if n.IsA("vtkMRMLSegmentationNode") else "model"
    index = _choose_option("Preoperative model", "Choose the preoperative model or segmentation to symmetrize.",
                           [f"{n.GetName()}   ({kind(n)})" for n in nodes])
    return None if index is None else nodes[index]


def run(csv_path=None, pre=None, side=None, source=None, sagittal=None, presets=True, landmarks="ask", initial_align=True,
        final_align=True, apply_final=True, segment=True):
    """Reference planes from PRE, Right/Left question, symmetrized model and landmark set, then measurements for every
    landmark set (PRE, mirPRE, POST, ...). See the module docstring.

    pre: the preoperative case name (default "PRE", else asked). side: "R" or "L" (else asked). source: the model or
    segmentation name (else the only one, or asked). sagittal: use this existing plane instead of making Sagittal.
    presets: select the preset measurements. csv_path skips the save dialog. landmarks: "ask" (first ask whether to
    define new landmarks or use those already set), True (define new ones), False (use those in the scene, no question).
    initial_align: once PRE_align and POST_align exist, make the rigid transform InitialAlign (POST_align onto PRE_align)
    with the Fiducial Registration Wizard (see registration.fiducial_registration); it is not applied to any node.
    final_align: then register the POST volume to the PRE volume with General Registration (BRAINS), starting from
    InitialAlign, making the rigid transform FinalAlign (see registration.intensity_registration); not applied either.
    Skipped, with a note in the log only, when the scene has neither a PRE nor a POST volume. Both registration steps
    (and remaking the _align lists) are skipped, with a note in the log, when the POST volume or POST landmark set is
    already under FinalAlign; "registration" then holds {"skipped": [those node names]}.
    apply_final: put the POST data (POST volume, POST landmark set, every node named POST...) under FinalAlign before
    anything is calculated (see registration.apply_final_align). On by default.
    segment: first of all (before the landmarks, which can then be placed on it) segment the head (skin surface), skull,
    mandible and teeth of the PRE and POST volumes with TotalSegmentator, one segmentation per volume without the sinus
    segments (see segmentation.segment_volumes). On by default; a segmentation made by an earlier run with the same kept
    segments is reused; with no PRE or POST volume it is skipped with a note in the log.
    Returns {"planes", "measurements", "model", "mirrored", "side", "initial_align", "final_align", "registration"}
    (empty dict if cancelled).
    """
    align_problems = []
    if segment:   # first, so the landmarks can be placed on it (and POST_HeadSkull later goes under FinalAlign)
        align_problems += segment_volumes()[1]
    if landmarks == "ask":
        answer = _ask_buttons("Landmarks", "Would you like to begin defining landmarks, or are the landmarks already set?",
                              ["Define new landmarks", "Use landmarks already set"],
                              f"Defining new landmarks walks you through every point of {PRE_SET_NAME} and then {POST_SET_NAME}.")
        if answer is None:
            return {}
        landmarks = answer.startswith("Define")
    if landmarks:
        if not define_landmarks():
            return {}
    # POST already put under FinalAlign (a re-run after applying it): registering again would start from POST positions
    # that are already aligned, so the alignment lists, InitialAlign and FinalAlign are left as they are.
    already_aligned = post_under_final_align() if (initial_align or final_align) else []
    if already_aligned:
        _log(f"Skipping InitialAlign and FinalAlign: {', '.join(already_aligned)} already under {FINAL_TRANSFORM_NAME}, "
             f"so the existing transforms and {PRE_CASE}{ALIGN_SUFFIX} / {POST_CASE}{ALIGN_SUFFIX} are kept. To register "
             f"again, take the {POST_CASE} data off {FINAL_TRANSFORM_NAME} first.")
    elif not landmarks and any(c["case"] in (PRE_CASE, POST_CASE) for c in find_cases()):
        align_problems = create_alignment_lists()[1]
    initial = None
    if already_aligned:
        initial = slicer.mrmlScene.GetFirstNodeByName(INITIAL_TRANSFORM_NAME)
    elif initial_align:
        if all(slicer.mrmlScene.GetFirstNodeByName(f"{c}{ALIGN_SUFFIX}") for c in (PRE_CASE, POST_CASE)):
            initial, more = fiducial_registration()
            align_problems += more
        else:
            align_problems.append(f"No InitialAlign: it needs both {PRE_CASE}{ALIGN_SUFFIX} and {POST_CASE}{ALIGN_SUFFIX}.")
    final, registration = None, {}
    if already_aligned:
        final = slicer.mrmlScene.GetFirstNodeByName(FINAL_TRANSFORM_NAME)
        registration = {"skipped": already_aligned}
    elif final_align:
        if find_volume(PRE_CASE) is None and find_volume(POST_CASE) is None:
            _log(f"No FinalAlign: the scene has no {PRE_CASE} or {POST_CASE} volume (landmarks only).")
        else:
            final, more, registration = intensity_registration()
            align_problems += more
    if apply_final and final is not None:   # so that every calculation below uses POST aligned to PRE
        align_problems += apply_final_align()[1]
    cases = find_cases()
    if not cases:
        raise RuntimeError(NO_LANDMARKS_MESSAGE)
    _log(f"{len(cases)} case(s) found: " + ", ".join(f"{c['case']} ({c['node'].GetName()})" for c in cases))
    pre_case = _choose_pre(cases, pre)
    if pre_case is None:
        return {}
    _log(f"Preoperative set: {pre_case['case']}")

    planes, sagittal_node, sagittal_landmarks, problems = make_reference_planes(pre_case, sagittal)
    problems = align_problems + problems
    _log(f"Reference planes from {pre_case['case']}: {', '.join(planes)}")
    if side is None:
        side = ask_side()
        if side is None:
            return {}
    side = side.upper()[:1]
    if side not in SIDES:
        raise ValueError(f"side must be 'R' or 'L', got {side!r}")
    right_normal, how = right_side_normal(sagittal_node, pre_case)
    keep_normal = right_normal if side == "R" else -right_normal
    _log(f"Keeping the {SIDES[side].lower()} side; which side is right was decided {how}.")
    problems += side_check(pre_case, sagittal_node, right_normal)

    model = None
    source_node = _choose_source(source)
    if source_node is None:
        problems.append("No symmetrized model: there is no model or segmentation in the scene (or the choice was cancelled).")
    else:
        try:
            model = create_symmetrized_model(source_node, sagittal_node, keep_normal, side)
        except ValueError as exc:
            problems.append(f"No symmetrized model: {exc}")

    mirrored = mirror_landmarks(pre_case, sagittal_node, side, exclude=sagittal_landmarks)
    measured = []
    for c in cases:   # the symmetrized set is measured like a case, right after PRE
        measured.append(c)
        if c is pre_case:
            measured.append(_case_from_node(mirrored, f"{MIRROR_PREFIX}{pre_case['case']}"))

    specs = []
    if presets:
        specs, more = preset_measurements(measured, sagittal_node.GetName())
        problems += more
        _log(f"{len(specs)} preset measurement(s) selected.")
    for problem in problems:
        _log(problem)
    if problems:
        slicer.util.warningDisplay("\n".join(problems), windowTitle="Symmetrized planes and measurements")
    if _ask_yes_no(f"Made the planes: {', '.join(planes) or 'none'}.\n\nWould you like to add any additional planes?", "Planes"):
        planes.update(generate_planes(pre_case))
    if _ask_yes_no(f"{len(specs)} preset measurement(s) will be calculated.\n\nWould you like to add any additional measurements?", "Measurements"):
        specs += select_measurements(pre_case)
    measurements = finalize_measurements(specs, measured, csv_path, PRESET_DIFFERENCES if presets else ())[0]
    _log(f"Done: {len(planes)} plane(s), {len(specs)} measurement(s) in {len(measured)} landmark set(s).")
    return {"planes": planes, "measurements": measurements, "model": model, "mirrored": mirrored, "side": side,
            "initial_align": initial, "final_align": final, "registration": registration}

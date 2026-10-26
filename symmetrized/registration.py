"""Registering POST to PRE: InitialAlign from the _align lists with SlicerIGT's Fiducial Registration Wizard, then
FinalAlign from the PRE and POST volumes with BRAINSFit (General Registration (BRAINS)), starting from InitialAlign."""

import math

import numpy as np

import slicer

from .settings import (
    ALIGN_ATTRIBUTE, ALIGN_SUFFIX, BRAINS_SAMPLING_PERCENTAGE, FINAL_TRANSFORM_NAME, INITIAL_TRANSFORM_NAME, POST_CASE,
    PRE_CASE, REGISTRATION_ATTRIBUTE, REGISTRATION_NODE_NAME,
)
from .scene import _has_prefix, _log, _unique_node_name, find_cases


def _own_node(name, class_name):
    """The node of this name and class made by this script earlier (reused, so whatever is under a transform stays
    under it), else a new one (with "_2", ... added if the name is taken by a node this script did not make)."""
    existing = slicer.mrmlScene.GetFirstNodeByName(name)
    if existing is not None and existing.IsA(class_name) and existing.GetAttribute(REGISTRATION_ATTRIBUTE):
        return existing
    node = slicer.mrmlScene.AddNewNodeByClass(class_name, _unique_node_name(name) if existing is not None else name)
    node.SetAttribute(REGISTRATION_ATTRIBUTE, "1")
    return node


def fiducial_registration(from_name=f"{POST_CASE}{ALIGN_SUFFIX}", to_name=f"{PRE_CASE}{ALIGN_SUFFIX}",
                          transform_name=INITIAL_TRANSFORM_NAME):
    """Run SlicerIGT's Fiducial Registration Wizard: From fiducials from_name (POST_align), To fiducials to_name
    (PRE_align), result transform type Rigid, point matching Manual (the points correspond in list order), update mode
    Manual, output a linear transform named transform_name ("InitialAlign"), then update it once (the wizard's Update
    button). The transform maps the From points onto the To points (POST onto PRE). It is not applied to any node.

    The wizard node ("REGISTRATION_NODE_NAME") and the transform made by an earlier run are reused. Returns (transform
    node or None, [problems]); the wizard's RMS error is in the log and on the transform as "FiducialRMS_mm".
    """
    if not hasattr(slicer.modules, "fiducialregistrationwizard"):
        return None, ["No InitialAlign: the Fiducial Registration Wizard (SlicerIGT extension) is not installed. Install "
                      "SlicerIGT from the Extensions Manager and restart Slicer."]
    lists = [slicer.mrmlScene.GetFirstNodeByName(n) for n in (from_name, to_name)]
    for name, node in zip((from_name, to_name), lists):
        if node is None or not node.IsA("vtkMRMLMarkupsFiducialNode"):
            return None, [f"No {transform_name}: there is no point list named {name}."]
    counts = [node.GetNumberOfControlPoints() for node in lists]
    if counts[0] != counts[1] or counts[0] < 3:
        return None, [f"No {transform_name}: {from_name} has {counts[0]} and {to_name} has {counts[1]} point(s); a rigid "
                      "registration with manual point matching needs the same number in both, at least 3."]
    labels = [[node.GetNthControlPointLabel(i) for i in range(node.GetNumberOfControlPoints())] for node in lists]
    problems = []
    if labels[0] != labels[1]:
        problems.append(f"{transform_name}: the points of {from_name} and {to_name} are paired by order, but their labels "
                        f"differ ({', '.join(labels[0])} vs {', '.join(labels[1])}).")

    transform = _own_node(transform_name, "vtkMRMLLinearTransformNode")
    wizard = _own_node(REGISTRATION_NODE_NAME, "vtkMRMLFiducialRegistrationWizardNode")
    wizard.SetAndObserveFromFiducialListNodeId(lists[0].GetID())
    wizard.SetAndObserveToFiducialListNodeId(lists[1].GetID())
    wizard.SetOutputTransformNodeId(transform.GetID())
    wizard.SetRegistrationModeToRigid()
    wizard.SetPointMatching(wizard.POINT_MATCHING_MANUAL)
    wizard.SetUpdateModeToManual()
    slicer.modules.fiducialregistrationwizard.logic().UpdateCalibration(wizard)

    error = wizard.GetCalibrationError()
    message = (wizard.GetCalibrationStatusMessage() or "").strip()
    transform.SetAttribute("FiducialRMS_mm", f"{error:.4f}")
    _log(f"{transform.GetName()}: rigid fiducial registration of {from_name} (From) to {to_name} (To), "
         f"{counts[0]} point pairs, RMS error {error:.3f} mm. {message}")
    return transform, problems


def find_volume(name):
    """The scalar volume with exactly this name (label maps do not count), or None."""
    for node in slicer.util.getNodesByClass("vtkMRMLScalarVolumeNode"):
        if node.GetName() == name and not node.IsA("vtkMRMLLabelMapVolumeNode"):
            return node
    return None


def _matrix(transform_node):
    """The 4 x 4 matrix of a linear transform node (to its parent) as a NumPy array."""
    import vtk

    m = vtk.vtkMatrix4x4()
    transform_node.GetMatrixTransformToParent(m)
    return np.array([[m.GetElement(i, j) for j in range(4)] for i in range(4)])


def _vtk_matrix(matrix):
    import vtk

    m = vtk.vtkMatrix4x4()
    for i in range(4):
        for j in range(4):
            m.SetElement(i, j, float(matrix[i][j]))
    return m


def _rotation_deg(matrix):
    return math.degrees(math.acos(max(-1.0, min(1.0, (np.trace(matrix[:3, :3]) - 1.0) / 2.0))))


def _align_rmse(matrix, from_name=f"{POST_CASE}{ALIGN_SUFFIX}", to_name=f"{PRE_CASE}{ALIGN_SUFFIX}"):
    """RMS distance (mm) between the To points and the From points moved by matrix, pairing them in list order; None if
    the lists are missing or differ in length. Uses each list's own coordinates (they have no parent transform)."""
    lists = [slicer.mrmlScene.GetFirstNodeByName(n) for n in (from_name, to_name)]
    if any(n is None for n in lists) or lists[0].GetNumberOfControlPoints() != lists[1].GetNumberOfControlPoints():
        return None
    total = 0.0
    count = lists[0].GetNumberOfControlPoints()
    for i in range(count):
        a, b = [0.0] * 3, [0.0] * 3
        lists[0].GetNthControlPointPosition(i, a)
        lists[1].GetNthControlPointPosition(i, b)
        moved = (matrix @ np.append(np.array(a, dtype=float), 1.0))[:3]
        total += float(np.sum((moved - np.array(b, dtype=float)) ** 2))
    return math.sqrt(total / count) if count else None


def intensity_registration(fixed_name=PRE_CASE, moving_name=POST_CASE, initial_name=INITIAL_TRANSFORM_NAME,
                           output_name=FINAL_TRANSFORM_NAME, sampling_percentage=BRAINS_SAMPLING_PERCENTAGE,
                           initialize_mode="Off", mask_mode="NOMASK"):
    """General Registration (BRAINS), Slicer's BRAINSFit CLI module: fixed image volume fixed_name (PRE), moving image
    volume moving_name (POST), percentage of samples sampling_percentage (0.002: a fraction, 0.2% of the fixed volume's
    voxels), initialization transform initial_name (InitialAlign), only the Rigid (6 DOF) stage, and a Slicer linear
    transform output named output_name (FinalAlign; reused and reset on a re-run). No other output.

    BRAINSFit's output already includes the initialization transform, so FinalAlign is the whole POST-to-PRE transform
    (use it instead of InitialAlign, not on top of it). It is not applied to any node. The moving volume is taken off
    its parent transform while BRAINSFit runs (and put back), so applying InitialAlign to POST first does not count
    twice; the fixed volume must not be under a transform. initialize_mode is BRAINSFit's Initialize Transform Mode,
    which its documentation says must be "Off" when an initialization transform is given; mask_mode ("NOMASK",
    "ROIAUTO") is its Masking Option.

    Returns (transform node or None, [problems], summary): summary holds "rotation_deg" and "center_shift_mm" (how far
    FinalAlign moved the POST volume's centre from where InitialAlign put it) and "rmse_initial" / "rmse_final" (RMS
    distance of POST_align onto PRE_align under each, mm).
    """
    fixed, moving = find_volume(fixed_name), find_volume(moving_name)
    initial = slicer.mrmlScene.GetFirstNodeByName(initial_name)
    for name, node in ((fixed_name, fixed), (moving_name, moving)):
        if node is None:
            return None, [f"No {output_name}: there is no volume named {name}."], {}
    if initial is None or not initial.IsA("vtkMRMLLinearTransformNode"):
        return None, [f"No {output_name}: there is no linear transform named {initial_name} to start from."], {}
    if fixed.GetParentTransformNode() is not None:
        return None, [f"No {output_name}: the fixed volume {fixed_name} is under the transform "
                      f"{fixed.GetParentTransformNode().GetName()}; it has to be free of transforms."], {}
    if not hasattr(slicer.modules, "brainsfit"):
        return None, [f"No {output_name}: the General Registration (BRAINS) module is not available."], {}
    initial_matrix = _matrix(initial)

    out = _own_node(output_name, "vtkMRMLLinearTransformNode")
    out.SetMatrixTransformToParent(_vtk_matrix(np.eye(4)))
    parameters = {
        "fixedVolume": fixed.GetID(),
        "movingVolume": moving.GetID(),
        "samplingPercentage": float(sampling_percentage),
        "initialTransform": initial.GetID(),
        "initializeTransformMode": initialize_mode,
        "useRigid": True,
        "linearTransform": out.GetID(),
        "maskProcessingMode": mask_mode,
    }
    _log(f"General Registration (BRAINS): fixed {fixed_name}, moving {moving_name}, sampling {sampling_percentage}, "
         f"initialization {initial.GetName()}, Rigid only, output {out.GetName()}")
    previous_parent = moving.GetParentTransformNode()
    moving.SetAndObserveTransformNodeID(None)
    try:
        cli = slicer.cli.runSync(slicer.modules.brainsfit, None, parameters)
        status, error = cli.GetStatusString(), cli.GetErrorText()
        slicer.mrmlScene.RemoveNode(cli)
    finally:
        moving.SetAndObserveTransformNodeID(previous_parent.GetID() if previous_parent is not None else None)
    if status != "Completed":
        return None, [f"No {output_name}: the registration did not complete ({status}). {error}"], {}

    final_matrix = _matrix(out)
    refinement = final_matrix @ np.linalg.inv(initial_matrix)   # from where InitialAlign puts POST to where FinalAlign does
    import vtk

    ijk_to_ras = vtk.vtkMatrix4x4()
    moving.GetIJKToRASMatrix(ijk_to_ras)
    dims = np.array(moving.GetImageData().GetDimensions(), dtype=float)
    center = np.array([[ijk_to_ras.GetElement(i, j) for j in range(4)] for i in range(4)]) @ np.append((dims - 1) / 2.0, 1.0)
    placed = initial_matrix @ center
    summary = {
        "rotation_deg": _rotation_deg(refinement),
        "center_shift_mm": float(np.linalg.norm((refinement @ placed)[:3] - placed[:3])),
        "rmse_initial": _align_rmse(initial_matrix),
        "rmse_final": _align_rmse(final_matrix),
    }
    out.SetAttribute("RefinementRotation_deg", f"{summary['rotation_deg']:.4f}")
    out.SetAttribute("RefinementShift_mm", f"{summary['center_shift_mm']:.4f}")
    _log(f"{out.GetName()}: {summary['rotation_deg']:.3f} deg rotation and {summary['center_shift_mm']:.3f} mm shift of the "
         f"{moving_name} volume's centre relative to {initial.GetName()}.")
    if summary["rmse_initial"] is not None:
        _log(f"{PRE_CASE}{ALIGN_SUFFIX} vs {POST_CASE}{ALIGN_SUFFIX} RMS: {summary['rmse_initial']:.3f} mm with "
             f"{initial.GetName()}, {summary['rmse_final']:.3f} mm with {out.GetName()} (landmarks on bone the surgery "
             "moved legitimately disagree).")
    return out, [], summary


def _is_under(node, transform):
    """True if transform is node's parent transform, or a parent of that, and so on."""
    parent = node.GetParentTransformNode()
    while parent is not None:
        if parent.GetID() == transform.GetID():
            return True
        parent = parent.GetParentTransformNode()
    return False


def post_under_final_align(transform_name=FINAL_TRANSFORM_NAME, moving_name=POST_CASE):
    """Names of the POST volume and POST landmark set that are already under the transform transform_name (FinalAlign),
    directly or through other transforms; empty if there is no such transform or neither is under it."""
    transform = slicer.mrmlScene.GetFirstNodeByName(transform_name)
    if transform is None or not transform.IsA("vtkMRMLTransformNode"):
        return []
    nodes = [find_volume(moving_name)] + [c["node"] for c in find_cases() if c["case"] == moving_name]
    return [n.GetName() for n in nodes if n is not None and _is_under(n, transform)]


def apply_final_align(transform_name=FINAL_TRANSFORM_NAME, moving_name=POST_CASE, initial_name=INITIAL_TRANSFORM_NAME):
    """Put the POST data under the transform transform_name (FinalAlign), as a soft transform (nothing is hardened).

    The POST data are the nodes whose name starts with moving_name as a whole word (the POST volume, POST_Landmarks,
    POST_Segmentation, "POST skull", ... but not "POSTER"), and the POST landmark set whatever its name. A node that is
    not under a transform, or is under initial_name (InitialAlign, which FinalAlign already includes), is put under
    FinalAlign; one already under FinalAlign (directly or through other transforms) is left as it is; one under some
    other transform is left alone and reported. POST_align is never moved (it holds copied world positions).

    Returns ([names of the nodes put under FinalAlign], [problems]); nothing happens if there is no such transform.
    """
    transform = slicer.mrmlScene.GetFirstNodeByName(transform_name)
    if transform is None or not transform.IsA("vtkMRMLTransformNode"):
        return [], []
    initial = slicer.mrmlScene.GetFirstNodeByName(initial_name)
    post_sets = {c["node"].GetID() for c in find_cases() if c["case"] == moving_name}
    moved, problems = [], []
    for node in slicer.util.getNodesByClass("vtkMRMLTransformableNode"):
        if node.IsA("vtkMRMLTransformNode") or node.GetHideFromEditors() or node.GetAttribute(ALIGN_ATTRIBUTE):
            continue
        if not (_has_prefix(node.GetName(), moving_name) or node.GetID() in post_sets):
            continue
        parent = node.GetParentTransformNode()
        if parent is not None and (parent.GetID() == transform.GetID() or _is_under(node, transform)):
            continue
        if parent is None or (initial is not None and parent.GetID() == initial.GetID()):
            node.SetAndObserveTransformNodeID(transform.GetID())
            moved.append(node.GetName())
        else:
            problems.append(f"{node.GetName()} was not put under {transform_name}: it is under the transform {parent.GetName()}.")
    if moved:
        _log(f"Applied {transform_name} to {', '.join(moved)} (a soft transform; nothing is hardened).")
    return moved, problems


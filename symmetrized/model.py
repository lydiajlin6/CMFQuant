"""The symmetrized model: the PRE model or segmentation cut at the sagittal plane, one half mirrored and joined."""

import slicer

from .settings import MODEL_ATTRIBUTE, SIDES
from .scene import _log, _remove_own, _unique_node_name, _unit
from .planes import plane_geometry
from .mirroring import reflection_matrix


def model_sources():
    """Models and segmentations that can be symmetrized (not Slicer's hidden ones, not ones this script made)."""
    nodes = [n for n in slicer.util.getNodesByClass("vtkMRMLModelNode") if not n.GetHideFromEditors() and not n.GetAttribute(MODEL_ATTRIBUTE)]
    nodes += list(slicer.util.getNodesByClass("vtkMRMLSegmentationNode"))
    return nodes


def source_polydata(node):
    """The surface of a model, or of all the segments of a segmentation (closed surface), in world coordinates."""
    import vtk

    if node.IsA("vtkMRMLSegmentationNode"):
        node.CreateClosedSurfaceRepresentation()
        ids = vtk.vtkStringArray()
        node.GetSegmentation().GetSegmentIDs(ids)
        append = vtk.vtkAppendPolyData()
        for i in range(ids.GetNumberOfValues()):
            surface = node.GetClosedSurfaceInternalRepresentation(ids.GetValue(i))
            if surface is not None and surface.GetNumberOfPoints():
                append.AddInputData(surface)
        if append.GetNumberOfInputConnections(0) == 0:
            raise ValueError(f"The segmentation {node.GetName()!r} has no segments with a surface.")
        append.Update()
        polydata = append.GetOutput()
    else:
        polydata = node.GetPolyData()
        if polydata is None or not polydata.GetNumberOfPoints():
            raise ValueError(f"The model {node.GetName()!r} is empty.")
    parent = node.GetParentTransformNode()
    if parent is None:
        out = vtk.vtkPolyData()
        out.DeepCopy(polydata)
        return out
    to_world = vtk.vtkGeneralTransform()
    slicer.vtkMRMLTransformNode.GetTransformBetweenNodes(parent, None, to_world)
    transform = vtk.vtkTransformPolyDataFilter()
    transform.SetTransform(to_world)
    transform.SetInputData(polydata)
    transform.Update()
    return transform.GetOutput()


def symmetrize_polydata(polydata, origin, keep_normal, tolerance_mm=1e-4):
    """The part of the surface on the keep_normal side of the plane, joined with its mirror image across the plane.

    The cut edges of the two halves coincide on the plane and are merged (points closer than tolerance_mm), so a closed
    surface gives a closed surface. The mirrored half's triangles are reversed so its normals still point outward.
    """
    import vtk

    copy = vtk.vtkPolyData()
    copy.DeepCopy(polydata)
    copy.GetPointData().SetNormals(None)
    copy.GetCellData().SetNormals(None)
    triangles = vtk.vtkTriangleFilter()
    triangles.SetInputData(copy)
    plane = vtk.vtkPlane()
    plane.SetOrigin(*[float(v) for v in origin])
    plane.SetNormal(*[float(v) for v in _unit(keep_normal)])
    clip = vtk.vtkClipPolyData()   # keeps where the plane function is above 0: the keep_normal side
    clip.SetInputConnection(triangles.GetOutputPort())
    clip.SetClipFunction(plane)
    clip.SetValue(0.0)
    clip.Update()
    kept = vtk.vtkPolyData()
    kept.DeepCopy(clip.GetOutput())
    if not kept.GetNumberOfCells():
        raise ValueError("Nothing of the surface lies on the side to keep.")

    matrix = vtk.vtkMatrix4x4()
    for i, row in enumerate(reflection_matrix(origin, keep_normal)):
        for j, value in enumerate(row):
            matrix.SetElement(i, j, float(value))
    reflection = vtk.vtkTransform()
    reflection.SetMatrix(matrix)
    mirror = vtk.vtkTransformPolyDataFilter()
    mirror.SetTransform(reflection)
    mirror.SetInputData(kept)
    reverse = vtk.vtkReverseSense()   # a reflection turns the triangles inside out
    reverse.SetInputConnection(mirror.GetOutputPort())
    reverse.ReverseCellsOn()
    reverse.ReverseNormalsOff()

    append = vtk.vtkAppendPolyData()
    append.AddInputData(kept)
    append.AddInputConnection(reverse.GetOutputPort())
    clean = vtk.vtkCleanPolyData()
    clean.SetInputConnection(append.GetOutputPort())
    clean.ToleranceIsAbsoluteOn()
    clean.SetAbsoluteTolerance(tolerance_mm)
    normals = vtk.vtkPolyDataNormals()
    normals.SetInputConnection(clean.GetOutputPort())
    normals.SplittingOff()
    normals.ConsistencyOn()
    normals.AutoOrientNormalsOff()
    normals.Update()
    out = vtk.vtkPolyData()
    out.DeepCopy(normals.GetOutput())
    return out


def create_symmetrized_model(source, planeNode, keep_normal, side):
    """A new model "<source>_symmetrized_<side>": source (a model or segmentation node, or its name) cut at the plane,
    the keep_normal half mirrored and joined to it. Replaces one made earlier from the same name."""
    if isinstance(source, str):
        matches = [n for n in model_sources() if n.GetName() == source]
        if not matches:
            raise ValueError(f"There is no model or segmentation named {source!r}. Available: {[n.GetName() for n in model_sources()]}")
        source = matches[0]
    origin = plane_geometry(planeNode)[0]
    polydata = symmetrize_polydata(source_polydata(source), origin, keep_normal)
    name = f"{source.GetName()}_symmetrized_{side}"
    if not _remove_own(name, MODEL_ATTRIBUTE):
        name = _unique_node_name(name)
    model = slicer.modules.models.logic().AddModel(polydata)
    model.SetName(name)
    model.SetAttribute(MODEL_ATTRIBUTE, source.GetName())
    model.SetAttribute("MirroredAcross", planeNode.GetName())
    model.SetAttribute("KeptSide", SIDES[side])
    display = model.GetDisplayNode()
    if display is not None:
        display.SetColor(0.95, 0.85, 0.55)
    _log(f"{name}: {SIDES[side].lower()} half of {source.GetName()} mirrored across {planeNode.GetName()} "
         f"({polydata.GetNumberOfPoints()} points, {polydata.GetNumberOfCells()} triangles)")
    return model

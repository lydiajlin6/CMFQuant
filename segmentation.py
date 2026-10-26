"""Optional: a head (skin surface), skull, mandible and teeth segmentation of each volume, from the TotalSegmentator
extension's "head: craniofacial structures" task (its sinus segments are removed)."""

import slicer

from .settings import (
    KEPT_SEGMENTS_ATTRIBUTE, POST_CASE, PRE_CASE, SEGMENTATION_ATTRIBUTE, SEGMENTATION_SUFFIX, TOTALSEG_HIGHER_ORDER_RESAMPLING,
    TOTALSEG_KEEP_SEGMENTS, TOTALSEG_TASK,
)
from .scene import _log, _unique_node_name
from .registration import find_volume


def totalsegmentator_logic():
    """The TotalSegmentator extension's logic, or None if the extension is not installed."""
    if not hasattr(slicer.modules, "totalsegmentator"):
        return None
    import TotalSegmentator

    return TotalSegmentator.TotalSegmentatorLogic()


def _keep_only(segmentation_node, keep):
    """Remove every segment except those whose ID or name is in keep (case-insensitive). Returns the kept names in the
    order of keep, and the missing ones."""
    segmentation = segmentation_node.GetSegmentation()
    wanted = [k.lower() for k in keep]
    found = {}
    for segment_id in list(segmentation.GetSegmentIDs()):
        segment = segmentation.GetSegment(segment_id)
        key = segment_id.lower() if segment_id.lower() in wanted else segment.GetName().lower()
        if key in wanted and key not in found:
            found[key] = segment_id
        else:
            segmentation.RemoveSegment(segment_id)
    return [found[k] for k in wanted if k in found], [k for k in wanted if k not in found]


def segment_head_and_skull(volume, logic=None, cpu=False):
    """Segment one volume with TotalSegmentator's "craniofacial_structures" task (TOTALSEG_TASK) with high order resampling
    on (TOTALSEG_HIGHER_ORDER_RESAMPLING), and keep only the segments in TOTALSEG_KEEP_SEGMENTS (head, skull, mandible,
    lower and upper teeth) in one segmentation, "<volume name>_HeadSkull" (SEGMENTATION_SUFFIX). It is placed in the
    volume's study (patient / study folder) in the Data module and under the volume's parent transform, if any.

    A segmentation made by this function earlier for the same volume, with the same TOTALSEG_KEEP_SEGMENTS, is reused, not
    recomputed (delete it to redo it); one made with a different list is replaced. Returns (segmentation node or None,
    [problems]).
    """
    name = f"{volume.GetName()}{SEGMENTATION_SUFFIX}"
    existing = slicer.mrmlScene.GetFirstNodeByName(name)
    if _is_current(existing):
        _log(f"{name}: already made earlier, reused (delete it to segment {volume.GetName()} again).")
        return existing, []
    if _is_own(existing):
        _log(f"{name}: made earlier with the segments {existing.GetAttribute(KEPT_SEGMENTS_ATTRIBUTE) or 'head, skull'}; "
             f"segmenting {volume.GetName()} again to keep {', '.join(TOTALSEG_KEEP_SEGMENTS)}.")
        slicer.mrmlScene.RemoveNode(existing)
        existing = None
    logic = logic or totalsegmentator_logic()
    if logic is None:
        return None, [f"No {name}: the TotalSegmentator extension is not installed (Extensions Manager)."]

    node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode", _unique_node_name(name) if existing is not None else name)
    node.CreateDefaultDisplayNodes()
    logic.higherOrderResampling = TOTALSEG_HIGHER_ORDER_RESAMPLING
    _log(f"TotalSegmentator, task {TOTALSEG_TASK}, high order resampling {TOTALSEG_HIGHER_ORDER_RESAMPLING}: segmenting "
         f"{volume.GetName()} (this can take several minutes)...")
    try:
        logic.process(volume, node, quality="normal", cpu=cpu, task=TOTALSEG_TASK, interactive=False)
    except Exception as exc:
        slicer.mrmlScene.RemoveNode(node)
        return None, [f"No {name}: TotalSegmentator failed on {volume.GetName()}: {exc}"]
    kept, missing = _keep_only(node, TOTALSEG_KEEP_SEGMENTS)
    node.SetAttribute(SEGMENTATION_ATTRIBUTE, volume.GetName())
    node.SetAttribute(KEPT_SEGMENTS_ATTRIBUTE, ",".join(TOTALSEG_KEEP_SEGMENTS))

    # the same place as the volume: its study in the Data module, and its parent transform
    sh = slicer.vtkMRMLSubjectHierarchyNode.GetSubjectHierarchyNode(slicer.mrmlScene)
    sh.SetItemParent(sh.GetItemByDataNode(node), sh.GetItemParent(sh.GetItemByDataNode(volume)))
    parent = volume.GetParentTransformNode()
    node.SetAndObserveTransformNodeID(parent.GetID() if parent is not None else None)
    node.SetReferenceImageGeometryParameterFromVolumeNode(volume)

    problems = [f"{node.GetName()}: TotalSegmentator found no {', '.join(missing)} in {volume.GetName()}."] if missing else []
    names = [node.GetSegmentation().GetSegment(i).GetName() for i in kept]
    _log(f"{node.GetName()}: segments {', '.join(names) or 'none'} (the other {TOTALSEG_TASK} segments were removed).")
    return node, problems


def segment_volumes(volume_names=(PRE_CASE, POST_CASE), cpu=False):
    """segment_head_and_skull() for each volume named in volume_names that is in the scene. Installs TotalSegmentator's
    Python packages first if they are missing (as its module does). Returns ({volume name: segmentation node}, [problems])."""
    volumes = [v for v in (find_volume(n) for n in volume_names) if v is not None]
    if not volumes:
        _log(f"No TotalSegmentator segmentation: there is no volume named {' or '.join(volume_names)} (landmarks only).")
        return {}, []
    to_do = [v for v in volumes if not _is_current(slicer.mrmlScene.GetFirstNodeByName(f"{v.GetName()}{SEGMENTATION_SUFFIX}"))]
    logic = None
    if to_do:
        logic = totalsegmentator_logic()
        if logic is None:
            return {}, ["No TotalSegmentator segmentation: the TotalSegmentator extension is not installed (Extensions Manager)."]
        try:
            logic.setupPythonRequirements()
        except Exception as exc:
            return {}, [f"No TotalSegmentator segmentation: TotalSegmentator's Python packages could not be set up ({exc}). "
                        "Open the TotalSegmentator module once, or restart Slicer if it asks to."]
    made, problems = {}, []
    for volume in volumes:
        node, more = segment_head_and_skull(volume, logic, cpu)
        problems += more
        if node is not None:
            made[volume.GetName()] = node
    return made, problems


def _is_own(node):
    """A segmentation made by segment_head_and_skull."""
    return node is not None and node.IsA("vtkMRMLSegmentationNode") and bool(node.GetAttribute(SEGMENTATION_ATTRIBUTE))


def _is_current(node):
    """A segmentation made by segment_head_and_skull with the current TOTALSEG_KEEP_SEGMENTS."""
    return _is_own(node) and node.GetAttribute(KEPT_SEGMENTS_ATTRIBUTE) == ",".join(TOTALSEG_KEEP_SEGMENTS)

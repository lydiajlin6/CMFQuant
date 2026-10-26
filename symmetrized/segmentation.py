"""A head (skin surface), skull, mandible and teeth segmentation of each volume, from the TotalSegmentator extension's
"head: craniofacial structures" task (its sinus segments are removed); or a head and skull segmentation you make yourself
by thresholding in the Segment Editor (segment_volumes_manually)."""

import slicer

from .settings import (
    KEPT_SEGMENTS_ATTRIBUTE, MANUAL_SEGMENTS, MANUAL_THRESHOLDS, POST_CASE, PRE_CASE, SEGMENTATION_ATTRIBUTE,
    SEGMENTATION_METHOD_ATTRIBUTE, SEGMENTATION_SUFFIX, TOTALSEG_HIGHER_ORDER_RESAMPLING, TOTALSEG_KEEP_SEGMENTS, TOTALSEG_TASK,
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
        how = "by hand" if _is_manual(existing) else "earlier"
        _log(f"{name}: already made {how}, reused (delete it to segment {volume.GetName()} again).")
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


def _is_manual(node):
    """A segmentation made by hand with segment_manually (kept; never replaced by an automatic one)."""
    return _is_own(node) and node.GetAttribute(SEGMENTATION_METHOD_ATTRIBUTE) == "manual"


def _is_current(node):
    """A segmentation to reuse: made by hand, or by segment_head_and_skull with the current TOTALSEG_KEEP_SEGMENTS."""
    return _is_manual(node) or (_is_own(node) and node.GetAttribute(KEPT_SEGMENTS_ATTRIBUTE) == ",".join(TOTALSEG_KEEP_SEGMENTS))


class _ThresholdPanel:
    """The small non-modal window shown while you set a threshold in the Segment Editor (the views stay usable)."""

    def __init__(self, title):
        import qt

        self.apply = self.cancel = False
        self.dialog = qt.QDialog(slicer.util.mainWindow())
        self.dialog.setObjectName("manualSegmentationPanel")
        self.dialog.setWindowTitle(title)
        self.dialog.setWindowModality(qt.Qt.NonModal)
        self.dialog.setMinimumWidth(460)
        layout = qt.QVBoxLayout(self.dialog)
        self.prompt = qt.QLabel("")
        self.prompt.setObjectName("promptLabel")
        self.prompt.setWordWrap(True)
        font = self.prompt.font
        font.setBold(True)
        self.prompt.setFont(font)
        layout.addWidget(self.prompt)
        hint = qt.QLabel("In the Segment Editor (left), drag the Threshold range until the highlighted area in the slice "
                         "views covers what you want, then click the button below. The largest connected piece is kept "
                         "(Islands: keep largest island).")
        hint.setWordWrap(True)
        layout.addWidget(hint)
        row = qt.QHBoxLayout()
        for text, name, flag in (("Apply threshold and keep largest island", "applyButton", "apply"), ("Cancel", "cancelButton", "cancel")):
            button = qt.QPushButton(text)
            button.setObjectName(name)
            button.clicked.connect(lambda *_, f=flag: setattr(self, f, True))
            row.addWidget(button)
        layout.addLayout(row)
        self.dialog.show()

    def close(self):
        self.dialog.close()
        self.dialog.deleteLater()


def _segment_editor():
    """The Segment Editor module's editor widget, with the module shown."""
    slicer.util.selectModule("SegmentEditor")
    return slicer.modules.segmenteditor.widgetRepresentation().self().editor


def _start_threshold(editor, volume, segment_id):
    editor.setCurrentSegmentID(segment_id)
    editor.setActiveEffectByName("Threshold")
    effect = editor.activeEffect()
    lo, hi = volume.GetImageData().GetScalarRange()
    start_min, start_max = MANUAL_THRESHOLDS.get(segment_id, (None, None))
    start_min = lo + (hi - lo) * 0.25 if start_min is None else min(max(float(start_min), lo), hi)
    start_max = hi if start_max is None else min(max(float(start_max), lo), hi)
    effect.setParameter("MinimumThreshold", str(start_min))
    effect.setParameter("MaximumThreshold", str(start_max))


def segment_manually(volume, poll_seconds=0.05):
    """Make the head and skull segmentation of one volume yourself: a segmentation "<volume name>_HeadSkull" with the
    segments in MANUAL_SEGMENTS ("head", then "skull") is made and the Segment Editor opened with the Threshold effect on
    it; for each segment you adjust the threshold and click "Apply threshold and keep largest island" in the small
    window, which applies the threshold and then Islands "keep largest island". The segmentation is tagged like the
    automatic one, placed in the volume's study and under its transform, so the rest of the tool uses it the same way.

    While you threshold, the volume is taken off its parent transform (and put back afterwards), so the threshold works on
    the volume's own voxels without resampling; and the Segment Editor is set not to overwrite other segments, to edit
    everywhere and to use no intensity mask (your previous settings are put back), so each segment is exactly its own
    threshold's largest island and the head stays solid (it includes the bone; the skull segment overlaps it). A
    segmentation made by hand earlier for the volume is reused; one made automatically is replaced. Returns
    (segmentation node or None, [problems], cancelled).
    """
    import time

    import SegmentEditorEffects

    name = f"{volume.GetName()}{SEGMENTATION_SUFFIX}"
    existing = slicer.mrmlScene.GetFirstNodeByName(name)
    if _is_manual(existing):
        _log(f"{name}: made by hand earlier, reused (delete it to make it again).")
        return existing, [], False
    if _is_own(existing):
        _log(f"{name}: made automatically earlier; replaced by the one you make now.")
        slicer.mrmlScene.RemoveNode(existing)
        existing = None
    parent = volume.GetParentTransformNode()
    volume.SetAndObserveTransformNodeID(None)   # threshold the volume's own voxels (no resampling through a transform)
    node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode", _unique_node_name(name) if existing is not None else name)
    node.CreateDefaultDisplayNodes()
    node.SetReferenceImageGeometryParameterFromVolumeNode(volume)   # both without a transform here: the volume's own grid
    for segment_id, _label in MANUAL_SEGMENTS:
        node.GetSegmentation().AddEmptySegment(segment_id, segment_id)
    node.SetAttribute(SEGMENTATION_ATTRIBUTE, volume.GetName())
    node.SetAttribute(SEGMENTATION_METHOD_ATTRIBUTE, "manual")
    sh = slicer.vtkMRMLSubjectHierarchyNode.GetSubjectHierarchyNode(slicer.mrmlScene)
    sh.SetItemParent(sh.GetItemByDataNode(node), sh.GetItemParent(sh.GetItemByDataNode(volume)))
    slicer.util.setSliceViewerLayers(background=volume, fit=True)

    editor = _segment_editor()
    editor.setSegmentationNode(node)
    editor.setSourceVolumeNode(volume)
    editor_node = editor.mrmlSegmentEditorNode()
    previous = (editor_node.GetOverwriteMode(), editor_node.GetMaskMode(), editor_node.GetSourceVolumeIntensityMask())
    editor_node.SetOverwriteMode(slicer.vtkMRMLSegmentEditorNode.OverwriteNone)
    editor_node.SetMaskMode(slicer.vtkMRMLSegmentationNode.EditAllowedEverywhere)
    editor_node.SetSourceVolumeIntensityMask(False)
    problems = []
    try:
        for segment_id, label in MANUAL_SEGMENTS:
            _start_threshold(editor, volume, segment_id)
            panel = _ThresholdPanel(f"Make your own segmentation: {name}")
            panel.prompt.setText(f"{volume.GetName()}: set the threshold for the {label}.")
            try:
                while True:
                    slicer.app.processEvents()
                    if panel.cancel or not panel.dialog.isVisible():
                        slicer.mrmlScene.RemoveNode(node)
                        _log(f"{name}: making it by hand was cancelled.")
                        return None, [], True
                    if panel.apply:
                        panel.apply = False
                        effect = editor.activeEffect()
                        if effect is not None and effect.name == "Threshold":
                            effect.self().onApply()   # (if you already clicked the effect's own Apply, this is the same)
                        if not slicer.util.arrayFromSegmentBinaryLabelmap(node, segment_id, volume).any():
                            slicer.util.warningDisplay(f"The {label} segment is empty: widen the threshold range.",
                                                       windowTitle="Make your own segmentation")
                            _start_threshold(editor, volume, segment_id)
                            continue
                        editor.setCurrentSegmentID(segment_id)
                        editor.setActiveEffectByName("Islands")
                        islands = editor.activeEffect()
                        islands.setParameter("Operation", SegmentEditorEffects.KEEP_LARGEST_ISLAND)
                        islands.setParameter("MinimumSize", "0")
                        islands.self().onApply()
                        _log(f"{name}: {segment_id} thresholded and its largest island kept.")
                        break
                    time.sleep(poll_seconds)
            finally:
                panel.close()
    finally:
        editor.setActiveEffect(None)
        editor_node.SetOverwriteMode(previous[0])
        editor_node.SetMaskMode(previous[1])
        editor_node.SetSourceVolumeIntensityMask(previous[2])
        volume.SetAndObserveTransformNodeID(parent.GetID() if parent is not None else None)
        if slicer.mrmlScene.IsNodePresent(node):
            node.SetAndObserveTransformNodeID(parent.GetID() if parent is not None else None)
        if hasattr(slicer.modules, "symmetrizedplanning"):
            slicer.util.selectModule("SymmetrizedPlanning")
    return node, problems, False


def segment_volumes_manually(volume_names=(PRE_CASE, POST_CASE)):
    """segment_manually() for each volume named in volume_names that is in the scene (PRE, then POST). Returns
    ({volume name: segmentation node}, [problems], cancelled)."""
    volumes = [v for v in (find_volume(n) for n in volume_names) if v is not None]
    if not volumes:
        _log(f"No segmentation to make: there is no volume named {' or '.join(volume_names)} (landmarks only).")
        return {}, [], False
    made, problems = {}, []
    for volume in volumes:
        node, more, cancelled = segment_manually(volume)
        problems += more
        if cancelled:
            return made, problems, True
        if node is not None:
            made[volume.GetName()] = node
    return made, problems, False

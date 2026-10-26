"""Guided landmark definition (the placement panel) and the PRE_align / POST_align lists."""

import slicer

from .settings import (
    ALIGN_ATTRIBUTE, ALIGN_LANDMARKS, ALIGN_SUFFIX, HIDDEN_WHILE_PLACING, LANDMARKS, OCCLUSAL_POINT, POST_CASE,
    POST_SET_NAME, PRE_CASE, PRE_ONLY_AFTER, PRE_ONLY_LANDMARKS, PRE_SET_NAME, SEGMENTATION_ATTRIBUTE,
    SOFT_TISSUE_LANDMARKS,
)
from .scene import _is_defined, _log, _remove_own, _unique_node_name, find_cases
from .dialogs import _ask_buttons, _ask_yes_no


class _PlacementPanel:
    """The small non-modal window shown while the points of one landmark set are placed (the views stay usable)."""

    def __init__(self, set_name):
        import qt

        self.undo = self.cancel = self.rearm = False
        self.dialog = qt.QDialog(slicer.util.mainWindow())
        self.dialog.setObjectName("landmarkPlacementPanel")
        self.dialog.setWindowTitle(f"Place landmarks: {set_name}")
        self.dialog.setWindowModality(qt.Qt.NonModal)
        self.dialog.setMinimumWidth(460)
        layout = qt.QVBoxLayout(self.dialog)
        self.prompt = qt.QLabel("")
        self.prompt.setObjectName("promptLabel")
        self.prompt.setWordWrap(True)
        font = self.prompt.font
        font.setPointSize(font.pointSize() + 2)
        font.setBold(True)
        self.prompt.setFont(font)
        layout.addWidget(self.prompt)
        self.progress = qt.QLabel("")
        self.progress.setObjectName("progressLabel")
        layout.addWidget(self.progress)
        hint = qt.QLabel("Click on the scan in a slice view, or on the model in the 3D view, to place the point. Each point "
                         "is labeled for you. You can drag a placed point to adjust it. If clicking no longer places a point "
                         "(for example after pressing Esc), click \"Place point\".")
        hint.setWordWrap(True)
        layout.addWidget(hint)
        row = qt.QHBoxLayout()
        for text, name, flag in (("Place point", "placeButton", "rearm"), ("Undo last point", "undoButton", "undo"),
                                 ("Cancel", "cancelButton", "cancel")):
            button = qt.QPushButton(text)
            button.setObjectName(name)
            button.clicked.connect(lambda *_, f=flag: setattr(self, f, True))
            row.addWidget(button)
        layout.addLayout(row)
        self.dialog.show()

    def show_step(self, number, total, abbreviation, full_name):
        self.prompt.setText(f"Please set the landmark for the: {full_name} ({abbreviation})")
        self.progress.setText(f"Point {number} of {total}")

    def close(self):
        self.dialog.close()
        self.dialog.deleteLater()


def _start_place_mode(node):
    interaction = slicer.mrmlScene.GetNodeByID("vtkMRMLInteractionNodeSingleton")
    try:
        logic = slicer.modules.markups.logic()
        logic.SetActiveList(node)
        logic.StartPlaceMode(True)   # persistent: every click places a point until the set is complete
    except Exception:  # older Slicer: set the active list through the selection node
        selection = slicer.mrmlScene.GetNodeByID("vtkMRMLSelectionNodeSingleton")
        selection.SetReferenceActivePlaceNodeClassName("vtkMRMLMarkupsFiducialNode")
        selection.SetActivePlaceNodeID(node.GetID())
        interaction.SetPlaceModePersistence(1)
        interaction.SetCurrentInteractionMode(interaction.Place)


def _stop_place_mode():
    interaction = slicer.mrmlScene.GetNodeByID("vtkMRMLInteractionNodeSingleton")
    interaction.SetCurrentInteractionMode(interaction.ViewTransform)


def _show_volume(case_name):
    """Show the volume named case_name (PRE, POST) in the slice views, if there is one."""
    for volume in slicer.util.getNodesByClass("vtkMRMLScalarVolumeNode"):
        if volume.GetName() == case_name and not volume.IsA("vtkMRMLLabelMapVolumeNode"):
            slicer.util.setSliceViewerLayers(background=volume, fit=True)
            return


def hide_other_landmark_sets(node):
    """Hide every point list (fiducial markups node) other than node, for example PRE_Landmarks while POST_Landmarks is
    placed, so that only the set being placed is shown. Returns a function that shows them again as they were."""
    undo = []
    for other in slicer.util.getNodesByClass("vtkMRMLMarkupsFiducialNode"):
        display = other.GetDisplayNode()
        if other.GetID() == node.GetID() or display is None or not display.GetVisibility():
            continue
        display.SetVisibility(False)
        undo.append(display)
    if undo:
        _log(f"Placing {node.GetName()}: hid the other landmark sets while placing "
             f"({', '.join(d.GetDisplayableNode().GetName() for d in undo)}).")

    def restore():
        for display in undo:
            display.SetVisibility(True)

    return restore


def show_segmentation_for_placing(case_name):
    """Show the TotalSegmentator segmentation of case_name's volume in 3D (as a surface) and hide the other ones, with the
    segments in HIDDEN_WHILE_PLACING (the head / skin) hidden in 3D so the bone underneath can be clicked. Returns a
    function that puts back what was changed. Does nothing if there is no such segmentation."""
    undo = []
    for seg in slicer.util.getNodesByClass("vtkMRMLSegmentationNode"):
        owner = seg.GetAttribute(SEGMENTATION_ATTRIBUTE)
        if not owner:
            continue
        if seg.GetDisplayNode() is None:
            seg.CreateDefaultDisplayNodes()
        display = seg.GetDisplayNode()
        was_visible = display.GetVisibility()
        undo.append(lambda d=display, v=was_visible: d.SetVisibility(v))
        if owner != case_name:
            display.SetVisibility(False)
            continue
        seg.CreateClosedSurfaceRepresentation()
        display.SetVisibility(True)
        display.SetVisibility3D(True)
        hidden = [h.lower() for h in HIDDEN_WHILE_PLACING]
        segmentation = seg.GetSegmentation()
        for segment_id in segmentation.GetSegmentIDs():
            if segment_id.lower() in hidden or segmentation.GetSegment(segment_id).GetName().lower() in hidden:
                was = display.GetSegmentVisibility3D(segment_id)
                display.SetSegmentVisibility3D(segment_id, False)
                undo.append(lambda d=display, i=segment_id, w=was: d.SetSegmentVisibility3D(i, w))
        _log(f"Placing {case_name} landmarks on {seg.GetName()} (shown in 3D; {', '.join(HIDDEN_WHILE_PLACING)} hidden "
             "while placing).")

    def restore():
        for step in reversed(undo):
            step()

    return restore


def set_soft_tissue_visible(case_name, visible):
    """Show (or hide) the HIDDEN_WHILE_PLACING segments (the head / skin) of case_name's TotalSegmentator segmentation in 3D,
    for placing soft tissue landmarks on the skin."""
    hidden = [h.lower() for h in HIDDEN_WHILE_PLACING]
    for seg in slicer.util.getNodesByClass("vtkMRMLSegmentationNode"):
        if seg.GetAttribute(SEGMENTATION_ATTRIBUTE) != case_name or seg.GetDisplayNode() is None:
            continue
        segmentation = seg.GetSegmentation()
        for segment_id in segmentation.GetSegmentIDs():
            if segment_id.lower() in hidden or segmentation.GetSegment(segment_id).GetName().lower() in hidden:
                seg.GetDisplayNode().SetSegmentVisibility3D(segment_id, visible)
        _log(f"{seg.GetName()}: {', '.join(HIDDEN_WHILE_PLACING)} {'shown for the soft tissue' if visible else 'hidden for the bony'} "
             "landmarks.")


MORE_OCCLUSAL = None   # in landmark_plan(): where the "Add another occluding surface point?" question comes


def landmark_plan(pre):
    """The order the landmarks of a set are asked in: [(abbreviation, full name), ...], for PRE with PRE_ONLY_LANDMARKS
    right after PRE_ONLY_AFTER followed by MORE_OCCLUSAL (the question about more molar points)."""
    common = list(LANDMARKS)
    if not pre:
        return common
    abbreviations = [a for a, _ in common]
    cut = abbreviations.index(PRE_ONLY_AFTER) + 1 if PRE_ONLY_AFTER in abbreviations else len(common)
    return common[:cut] + list(PRE_ONLY_LANDMARKS) + [MORE_OCCLUSAL] + common[cut:]


def place_landmarks(node, landmarks, poll_seconds=0.05, first_number=1, total=None):
    """Guide the user through placing landmarks [(abbreviation, full name)] in the fiducial list node, one by one.

    Points placed after the ones the list already has are labeled with the abbreviation (and the full name as the
    description) in the order given, whatever label they got; that is re-applied until the end, so a renamed point is
    put right. Points beyond the last landmark are removed, as are stray lists made by the Markups toolbar meanwhile
    (their points are moved into node first). Returns True when every landmark was placed, False if cancelled (the
    points placed so far stay). first_number and total only change the "Point n of N" shown (for a set placed in parts).
    """
    import time

    start = node.GetNumberOfControlPoints()
    before_ids = {n.GetID() for n in slicer.util.getNodesByClass("vtkMRMLMarkupsFiducialNode")}
    panel = _PlacementPanel(node.GetName())
    _start_place_mode(node)
    shown = None
    try:
        while True:
            slicer.app.processEvents()
            for stray in [n for n in slicer.util.getNodesByClass("vtkMRMLMarkupsFiducialNode") if n.GetID() not in before_ids]:
                for i in range(stray.GetNumberOfControlPoints()):
                    if _is_defined(stray, i):
                        position = [0.0, 0.0, 0.0]
                        stray.GetNthControlPointPositionWorld(i, position)
                        node.AddControlPoint(position)
                slicer.mrmlScene.RemoveNode(stray)
                _start_place_mode(node)
            if panel.undo:
                panel.undo = False
                placed_now = [i for i in range(start, node.GetNumberOfControlPoints()) if _is_defined(node, i)]
                if placed_now:
                    node.RemoveNthControlPoint(placed_now[-1])
                _start_place_mode(node)
            if panel.rearm:
                panel.rearm = False
                _start_place_mode(node)
            placed = [i for i in range(start, node.GetNumberOfControlPoints()) if _is_defined(node, i)]
            for index in reversed(placed[len(landmarks):]):   # clicked once too often
                node.RemoveNthControlPoint(index)
            placed = placed[:len(landmarks)]
            for index, (abbreviation, full_name) in zip(placed, landmarks):
                if node.GetNthControlPointLabel(index) != abbreviation:
                    node.SetNthControlPointLabel(index, abbreviation)
                if node.GetNthControlPointDescription(index) != full_name:
                    node.SetNthControlPointDescription(index, full_name)
            if len(placed) == len(landmarks):
                _log(f"{node.GetName()}: placed {', '.join(a for a, _ in landmarks)}")
                return True
            if panel.cancel or not panel.dialog.isVisible():
                _log(f"{node.GetName()}: placement cancelled after {len(placed)} of {len(landmarks)} point(s).")
                return False
            step = (len(placed),)
            if step != shown:
                shown = step
                abbreviation, full_name = landmarks[len(placed)]
                panel.show_step(first_number + len(placed), total or len(landmarks), abbreviation, full_name)
            time.sleep(poll_seconds)
    finally:
        # remove a preview point left by place mode, then leave place mode
        for i in reversed(range(start, node.GetNumberOfControlPoints())):
            if not _is_defined(node, i):
                node.RemoveNthControlPoint(i)
        _stop_place_mode()
        panel.close()


def _new_landmark_set(name):
    """A new empty fiducial list called name, or None to keep the existing one of that name (asked)."""
    existing = slicer.mrmlScene.GetFirstNodeByName(name)
    if existing is not None:
        if not _ask_yes_no(f"A landmark set named {name} is already in the scene ({existing.GetNumberOfControlPoints()} "
                           f"point(s)).\n\nReplace it with new landmarks? (No keeps it as it is.)", "Landmarks"):
            return None
        slicer.mrmlScene.RemoveNode(existing)
    node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsFiducialNode", name)
    node.CreateDefaultDisplayNodes()
    return node


def define_landmarks(pre_name=PRE_SET_NAME, post_name=POST_SET_NAME):
    """Guided definition of the PRE and then the POST landmark set (see the module docstring). Returns True when both
    are done (or kept), False if cancelled. Makes the _align lists at the end."""
    import itertools

    for name, case_name in ((pre_name, PRE_CASE), (post_name, POST_CASE)):
        if name == post_name:
            slicer.util.infoDisplay(f"The preoperative landmarks are done.\n\nNext: the postoperative landmarks ({post_name}). "
                                    "Show the postoperative scan in the views if it is not shown already.",
                                    windowTitle="Postoperative landmarks")
        node = _new_landmark_set(name)
        if node is None:
            _log(f"Kept the existing {name}.")
            continue
        _show_volume(case_name)
        restore_segmentation = show_segmentation_for_placing(case_name)
        restore_landmarks = hide_other_landmark_sets(node)
        plan = landmark_plan(name == pre_name)
        total = sum(1 for item in plan if item is not MORE_OCCLUSAL)
        done = 0
        soft = None
        kind = lambda item: "more" if item is MORE_OCCLUSAL else ("soft" if item[0] in SOFT_TISSUE_LANDMARKS else "bone")
        try:
            for part, items in itertools.groupby(plan, key=kind):
                items = list(items)
                if part == "more":
                    counts = {"L": 1, "R": 1}
                    while True:
                        answer = _ask_buttons("Occlusal points", "Add another occluding surface point of the molars?",
                                              ["Right", "Left", "Done"], "At least one per side has been placed. More "
                                              "points give a better occlusal plane.")
                        if answer in (None, "Done"):
                            break
                        side = answer[0]
                        counts[side] += 1
                        total += 1
                        extra = (f"{side}Oc{counts[side]}", f"{OCCLUSAL_POINT[side]}, point {counts[side]}")
                        if not place_landmarks(node, [extra], first_number=done + 1, total=total):
                            return False
                        done += 1
                    continue
                if (part == "soft") != bool(soft):
                    soft = part == "soft"
                    if soft or done:   # the head is already hidden at the start
                        set_soft_tissue_visible(case_name, soft)
                if not place_landmarks(node, items, first_number=done + 1, total=total):
                    return False
                done += len(items)
        finally:
            restore_landmarks()
            restore_segmentation()
    for problem in create_alignment_lists()[1]:
        _log(problem)
    return True


def create_alignment_lists(case_names=(PRE_CASE, POST_CASE)):
    """For each case in case_names: a new list "<case>_align" with ALIGN_LANDMARKS copied from its landmark set, in that
    order. The positions are where the points are shown (world coordinates, so a parent transform of the set is
    included) and the list has no parent transform: the Fiducial Registration Wizard reads each list's own coordinates
    and ignores parent transforms, so this makes InitialAlign map POST as shown onto PRE as shown. Replaces one made
    earlier. Returns ({case: list node}, [problems])."""
    made, problems = {}, []
    cases = {c["case"]: c for c in find_cases()}
    for case_name in case_names:
        case = cases.get(case_name)
        if case is None:
            problems.append(f"No {case_name}{ALIGN_SUFFIX} list: there is no {case_name} landmark set.")
            continue
        missing = [n for n in ALIGN_LANDMARKS if n not in case["names"]]
        if missing:
            problems.append(f"{case_name}{ALIGN_SUFFIX} lacks {', '.join(missing)}: {case_name} has no landmark of that name.")
        name = f"{case_name}{ALIGN_SUFFIX}"
        if not _remove_own(name, ALIGN_ATTRIBUTE):
            name = _unique_node_name(name)
        node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsFiducialNode", name)
        node.SetAttribute(ALIGN_ATTRIBUTE, case["node"].GetName())
        for label in ALIGN_LANDMARKS:
            if label in case["names"]:
                position = [0.0, 0.0, 0.0]
                case["node"].GetNthControlPointPositionWorld(case["names"][label], position)
                node.AddControlPoint(position, label)
        made[case_name] = node
        _log(f"{name}: {', '.join(node.GetNthControlPointLabel(i) for i in range(node.GetNumberOfControlPoints()))} "
             f"from {case['node'].GetName()}")
    return made, problems

"""Small Qt dialogs: questions, pickers, the Right/Left box and the CSV save dialog."""

import os

import slicer

from .settings import ORIENTATIONS


def _ask_yes_no(question, title):
    return bool(slicer.util.confirmYesNoDisplay(question, windowTitle=title))


def _button_row(dialog, ok_text="OK"):
    import qt

    ok = qt.QPushButton(ok_text)
    ok.setObjectName("okButton")
    ok.setDefault(True)
    cancel = qt.QPushButton("Cancel")
    cancel.setObjectName("cancelButton")
    ok.clicked.connect(lambda *_: dialog.accept())   # clicked passes a "checked" flag
    cancel.clicked.connect(lambda *_: dialog.reject())
    row = qt.QHBoxLayout()
    row.addStretch(1)
    row.addWidget(ok)
    row.addWidget(cancel)
    return row, ok


def _choose_option(title, prompt, options):
    """Index of the option the user picks, or None if cancelled."""
    import qt

    dialog = qt.QDialog(slicer.util.mainWindow())
    dialog.setWindowTitle(title)
    dialog.setMinimumWidth(380)
    layout = qt.QVBoxLayout(dialog)
    label = qt.QLabel(prompt)
    label.setWordWrap(True)
    layout.addWidget(label)
    combo = qt.QComboBox()
    combo.setObjectName("optionCombo")
    combo.addItems(list(options))
    layout.addWidget(combo)
    row, _ = _button_row(dialog)
    layout.addLayout(row)
    if not dialog.exec_():
        return None
    return int(combo.currentIndex)


def _ask_text(title, prompt, default=""):
    """Text the user types (possibly empty), or None if cancelled."""
    import qt

    dialog = qt.QDialog(slicer.util.mainWindow())
    dialog.setWindowTitle(title)
    dialog.setMinimumWidth(380)
    layout = qt.QVBoxLayout(dialog)
    label = qt.QLabel(prompt)
    label.setWordWrap(True)
    layout.addWidget(label)
    edit = qt.QLineEdit(default)
    edit.setObjectName("nameEdit")
    layout.addWidget(edit)
    row, _ = _button_row(dialog)
    layout.addLayout(row)
    if not dialog.exec_():
        return None
    return edit.text.strip()


def _ask_buttons(title, text, buttons, informative=""):
    """The text of the button the user clicks in a question box (None for Cancel or closing it)."""
    import qt

    box = qt.QMessageBox(slicer.util.mainWindow())
    box.setWindowTitle(title)
    box.setIcon(qt.QMessageBox.Question)
    box.setText(text)
    if informative:
        box.setInformativeText(informative)
    made = {box.addButton(b, qt.QMessageBox.AcceptRole): b for b in buttons}
    box.addButton(qt.QMessageBox.Cancel)
    box.exec_()
    clicked = box.clickedButton()
    for button, text_ in made.items():
        if clicked == button:
            return text_
    return None


def ask_side():
    """"R" or "L" from a Right / Left question box, or None if cancelled."""
    import qt

    box = qt.QMessageBox(slicer.util.mainWindow())
    box.setWindowTitle("Side to keep")
    box.setIcon(qt.QMessageBox.Question)
    box.setText("Which side should be kept and mirrored across the sagittal plane?")
    box.setInformativeText("The other side of the preoperative model and landmarks is replaced by the mirror image of this one.")
    right = box.addButton("Right", qt.QMessageBox.AcceptRole)
    left = box.addButton("Left", qt.QMessageBox.AcceptRole)
    box.addButton(qt.QMessageBox.Cancel)
    box.exec_()
    clicked = box.clickedButton()
    if clicked == right:
        return "R"
    if clicked == left:
        return "L"
    return None


def _choose_landmarks(title, prompt, names, min_count=1, single=False, default_name=None, plane_bases=None, orientation=False):
    """Dialog: pick landmarks by name, and optionally a plane name, an existing plane and an orientation.
    Returns {"names", "name", "plane", "orientation"} or None if cancelled."""
    import qt

    names = list(names)
    dialog = qt.QDialog(slicer.util.mainWindow())
    dialog.setWindowTitle(title)
    dialog.setMinimumWidth(420)
    layout = qt.QVBoxLayout(dialog)
    prompt_label = qt.QLabel(prompt)
    prompt_label.setWordWrap(True)
    layout.addWidget(prompt_label)

    plane_combo = orientation_combo = None
    if plane_bases is not None:
        layout.addWidget(qt.QLabel("Existing plane to be perpendicular to:"))
        plane_combo = qt.QComboBox()
        plane_combo.setObjectName("referencePlaneCombo")
        for base in plane_bases:
            plane_combo.addItem(base)
        layout.addWidget(plane_combo)
    if orientation:
        layout.addWidget(qt.QLabel("Orientation of the new plane:"))
        orientation_combo = qt.QComboBox()
        orientation_combo.setObjectName("orientationCombo")
        for text in ORIENTATIONS.values():
            orientation_combo.addItem(text)
        layout.addWidget(orientation_combo)

    layout.addWidget(qt.QLabel("Landmark:" if single else "Landmarks (click to select several):"))
    list_widget = qt.QListWidget()
    list_widget.setObjectName("landmarkList")
    list_widget.setSelectionMode(qt.QAbstractItemView.SingleSelection if single else qt.QAbstractItemView.MultiSelection)
    for name in names:
        list_widget.addItem(name)
    layout.addWidget(list_widget)

    name_edit = None
    if default_name is not None:
        layout.addWidget(qt.QLabel("Plane name:"))
        name_edit = qt.QLineEdit(default_name)
        name_edit.setObjectName("nameEdit")
        layout.addWidget(name_edit)

    status = qt.QLabel("")
    layout.addWidget(status)
    row, ok_button = _button_row(dialog)
    layout.addLayout(row)

    def selected_rows():
        return [r for r in range(list_widget.count) if list_widget.item(r).isSelected()]

    def update_status(*_):
        count = len(selected_rows())
        status.setText(f"Selected: {count} (need {'1 landmark' if single else f'at least {min_count}'})")
        ok_button.setEnabled(count >= min_count and (not single or count == 1))

    list_widget.itemSelectionChanged.connect(update_status)
    update_status()
    if not dialog.exec_():
        return None
    return {
        "names": [names[r] for r in selected_rows()],
        "name": name_edit.text.strip() if name_edit is not None else None,
        "plane": plane_bases[plane_combo.currentIndex] if plane_combo is not None else None,
        "orientation": list(ORIENTATIONS)[orientation_combo.currentIndex] if orientation_combo is not None else None,
    }


def _ask_csv_path(default_path):
    """Where to save the CSV (a save dialog), or None if cancelled."""
    import qt

    dialog = qt.QFileDialog(slicer.util.mainWindow(), "Save measurements CSV", os.path.dirname(default_path), "CSV files (*.csv)")
    dialog.setObjectName("csvDialog")
    dialog.setAcceptMode(qt.QFileDialog.AcceptSave)
    dialog.setOption(qt.QFileDialog.DontUseNativeDialog, True)
    dialog.setDefaultSuffix("csv")
    dialog.selectFile(os.path.basename(default_path))
    if not dialog.exec_():
        return None
    files = dialog.selectedFiles()
    return str(files[0]) if files else None

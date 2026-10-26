"""3D Slicer module that starts the symmetrized tool: Modules menu > Craniofacial > Symmetrized Planning > Start.

Install it once: drag this file (or the SlicerModule folder that holds it) onto the Slicer window, and in the dialog
that appears keep "Add selected module to 'Additional module paths'" ticked and click Yes. Slicer then loads the module
every time it starts. See ../README.md.

This file only opens the tool; the code is the symmetrized package two folders up, which must stay next to it (keep
the folder structure symmetrized/SlicerModule/SymmetrizedPlanning.py).
"""

import os
import sys
import traceback

import qt
import slicer
from slicer.ScriptedLoadableModule import *  # noqa: F403 (ScriptedLoadableModule, ScriptedLoadableModuleWidget)

# symmetrized/SlicerModule/this file -> the folder that contains symmetrized/
PACKAGE_FOLDER = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PACKAGE_PARENT = os.path.dirname(PACKAGE_FOLDER)
SEGMENTATION_CHOICES = [("Automatic (TotalSegmentator)", "auto"),
                        ("Make my own (threshold in the Segment Editor)", "manual"),
                        ("None", "none")]


class SymmetrizedPlanning(ScriptedLoadableModule):  # noqa: F405
    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)  # noqa: F405
        parent.title = "Symmetrized Planning"
        parent.categories = ["Craniofacial"]
        parent.dependencies = []
        parent.contributors = ["CHLA craniofacial research"]
        parent.helpText = (
            "Guided PRE/POST landmarks, POST-to-PRE registration (InitialAlign, FinalAlign), reference planes from the "
            "PRE landmarks, a symmetrized model and landmark set, and measurements of every landmark set in one CSV. "
            "Click Start and answer the questions. Full instructions: README.md in the symmetrized folder.")
        parent.acknowledgementText = "Research tool, not a validated clinical device."


class SymmetrizedPlanningWidget(ScriptedLoadableModuleWidget):  # noqa: F405
    def setup(self):
        ScriptedLoadableModuleWidget.setup(self)  # noqa: F405
        intro = qt.QLabel(
            "<b>Before you start</b>, load the preoperative scan and name the volume <b>PRE</b>, the postoperative scan "
            "named <b>POST</b>, and the preoperative model or segmentation of the skull.<br><br>"
            "Then click <b>Start</b> and answer the questions. A small window guides you through placing every "
            "landmark; at the end you choose where to save the CSV.")
        intro.wordWrap = True
        self.layout.addWidget(intro)

        start = qt.QPushButton("Start")
        start.objectName = "startButton"
        start.toolTip = "Run the whole tool (it first asks whether to define new landmarks or use those already set)."
        start.minimumHeight = 40
        start.connect("clicked(bool)", self.onStart)

        self.layout.addWidget(qt.QLabel("Segmentation of PRE and POST, made first and used for placing the landmarks:"))
        self.segmentationCombo = qt.QComboBox()
        self.segmentationCombo.objectName = "segmentationCombo"
        for text, value in SEGMENTATION_CHOICES:
            self.segmentationCombo.addItem(text, value)
        self.segmentationCombo.toolTip = (
            "Automatic: TotalSegmentator (needs the extension; minutes per scan the first time, reused afterwards) makes "
            "PRE_HeadSkull and POST_HeadSkull with head, skull, mandible and teeth. Make my own: the Segment Editor opens "
            "and you set a threshold for the head, then the skull (largest island kept). None: no segmentation.")
        self.layout.addWidget(self.segmentationCombo)
        self.layout.addWidget(start)

        readme = qt.QPushButton("Open the instructions (README)")
        readme.objectName = "readmeButton"
        readme.connect("clicked(bool)", self.onReadme)
        self.layout.addWidget(readme)

        where = qt.QLabel(f"Code folder: {PACKAGE_FOLDER}")
        where.wordWrap = True
        where.setStyleSheet("color: gray")
        self.layout.addWidget(where)
        self.layout.addStretch(1)

    def onStart(self, *_):
        if PACKAGE_PARENT not in sys.path:
            sys.path.insert(0, PACKAGE_PARENT)
        try:
            import symmetrized
            symmetrized.start(segment=self.segmentationCombo.currentData)
        except Exception as exc:  # show it instead of failing silently in the console
            slicer.util.errorDisplay(f"The symmetrized tool stopped with an error:\n\n{exc}", windowTitle="Symmetrized Planning",
                                     detailedText=traceback.format_exc())

    def onReadme(self, *_):
        qt.QDesktopServices.openUrl(qt.QUrl.fromLocalFile(os.path.join(PACKAGE_FOLDER, "README.md")))

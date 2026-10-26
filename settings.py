"""Everything you might want to change: the landmark lists, the reference planes, the preset
measurements and the names used.

Edit this file, then start the tool again with symmetrized.start() (which reloads it)."""

import re


# Guided landmark definition (define_landmarks): (abbreviation = control point label, full name), in the order asked.
PRE_SET_NAME = "PRE_Landmarks"

POST_SET_NAME = "POST_Landmarks"

LANDMARKS = [   # both PRE and POST
    ("CG", "Crista Galli"), ("Se", "Sella"), ("Ba", "Basion"), ("ANS", "Anterior Nasal Spine"),
    ("LM", "Left Mastoid"), ("RM", "Right Mastoid"),
    ("LD", "Left Dacryon"), ("LZf", "Left Zygomaticofrontal Suture"), ("LsOr", "Left Superior Orbitale"),
    ("LlOr", "Left Lateral Orbitale"), ("LiOr", "Left Inferior Orbitale"),
    ("RD", "Right Dacryon"), ("RZf", "Right Zygomaticofrontal Suture"), ("RsOr", "Right Superior Orbitale"),
    ("RlOr", "Right Lateral Orbitale"), ("RiOr", "Right Inferior Orbitale"),
    ("LMC", "Left Medial Canthus"), ("LLC", "Left Lateral Canthus"),
    ("RMC", "Right Medial Canthus"), ("RLC", "Right Lateral Canthus"),
]

PRE_ONLY_LANDMARKS = [   # PRE only; more occluding molar surface points (LOc2, ROc2, ...) can follow them
    ("LIn", "Left Central Incisal Edge"), ("RIn", "Right Central Incisal Edge"),
    ("LOc1", "Left Occluding surface of the molars, point 1"), ("ROc1", "Right Occluding surface of the molars, point 1"),
]
PRE_ONLY_AFTER = "ANS"   # in PRE, PRE_ONLY_LANDMARKS (and the extra molar points) are asked right after this landmark

# Soft tissue landmarks: while these are placed, the head (skin) segment is shown in 3D (it is hidden for the others).
SOFT_TISSUE_LANDMARKS = ["LMC", "LLC", "RMC", "RLC"]

# Additional landmarks: after each set's list you are asked whether to add more (an abbreviation and a full name of your
# choice, then you place it); the ones added to PRE are then asked for in POST too. False skips the question. Landmarks
# you always want can instead be added to LANDMARKS above.
ASK_FOR_ADDITIONAL_LANDMARKS = True

OCCLUSAL_POINT = {"L": "Left Occluding surface of the molars", "R": "Right Occluding surface of the molars"}

# Lists for a manual fiducial registration: these landmarks, in this order, copied into "<case>_align".
ALIGN_LANDMARKS = ["CG", "Se", "Ba", "ANS", "RM", "LM"]
# The rigid transform the Fiducial Registration Wizard makes from them (POST_align onto PRE_align).
INITIAL_TRANSFORM_NAME = "InitialAlign"
# General Registration (BRAINS) of the POST volume to the PRE volume, starting from InitialAlign: rigid (6 DOF) only.
FINAL_TRANSFORM_NAME = "FinalAlign"
BRAINS_SAMPLING_PERCENTAGE = 0.002   # BRAINSFit's "Percentage of samples": a fraction of the fixed volume's voxels
# Segmentation of the PRE and POST volumes (on by default; run(segment=False) or the module's checkbox turns it off),
# done first so it can be used for placing the landmarks, with the TotalSegmentator extension's
# "head: craniofacial structures" task, which gives mandible, teeth_lower, skull, head, sinus_maxillary, sinus_frontal and
# teeth_upper. Only the segments listed here are kept (the sinuses are removed), in one segmentation named
# "<volume name>_HeadSkull". A segmentation made earlier with a different list is made again.
TOTALSEG_TASK = "craniofacial_structures"
TOTALSEG_KEEP_SEGMENTS = ["head", "skull", "mandible", "teeth_lower", "teeth_upper"]
TOTALSEG_HIGHER_ORDER_RESAMPLING = True   # the advanced option "high order resampling"
SEGMENTATION_SUFFIX = "_HeadSkull"
# While a landmark set is placed, its segmentation is shown in 3D without these segments, so that the bone underneath
# can be clicked; they are shown for the SOFT_TISSUE_LANDMARKS and again afterwards.
HIDDEN_WHILE_PLACING = ["head"]
# Making the segmentations yourself instead (run(segment="manual"), or the module's choice "Make my own"): the Segment
# Editor's Threshold effect, first for the head (skin), then for the skull, each followed by Islands "keep largest
# island". The threshold range the Threshold effect starts at for each (in the volume's units, Hounsfield units for CT;
# None = the volume's maximum); you adjust it for each scan.
MANUAL_SEGMENTS = [("head", "head (skin surface)"), ("skull", "skull (bone)")]
MANUAL_THRESHOLDS = {"head": (-300, None), "skull": (300, None)}

# The sagittal plane (the mirror plane). Its landmarks are not mirrored.
SAGITTAL_PLANE = "Sagittal"

SAGITTAL_LANDMARKS = ["CG", "Se", "Ba", "ANS"]

# Which side of a reference plane is superior: the side this PRE landmark is on (recorded when the planes are made; a
# plane made without it uses world superior, +z in RAS). Used for signed values that must not depend on which way a
# plane's normal happens to point, such as the canthal tilt.
SUPERIOR_LANDMARK = "CG"

# Reference planes, all made from the PRE landmarks, in this order (a later one may use an earlier one).
# "also_containing": every PRE landmark whose name contains this text is added (case-sensitive). "landmarks_orthogonal":
# best fit to the landmarks among the planes perpendicular to the plane "orthogonal_to".
REFERENCE_PLANES = [
    {"name": "Occlusal", "type": "landmarks", "landmarks": ["RIn", "LIn"], "also_containing": "Oc"},
    {"name": SAGITTAL_PLANE, "type": "landmarks_orthogonal", "landmarks": SAGITTAL_LANDMARKS, "orthogonal_to": "Occlusal"},
    {"name": "Paracoronal", "type": "orthogonal2", "landmarks": ["Se"], "references": [SAGITTAL_PLANE, "Occlusal"]},
]

PRESET_SIDES = ("R", "L")  # the landmark names start with the side: RZf and LZf, RsOr and LsOr, ...


def preset_measurement_specs():
    """The measurements that are calculated automatically, for each side."""
    specs = []
    for side in PRESET_SIDES:
        specs += [
            {"kind": "plane_angle", "landmarks": [f"{side}iOr", f"{side}sOr"], "plane": "Paracoronal", "name": f"{side} pitch (x rotation)"},
            {"kind": "plane_angle", "landmarks": [f"{side}Zf", f"{side}D"], "plane": "Paracoronal", "name": f"{side} yaw (y rotation)"},
            {"kind": "plane_angle", "landmarks": [f"{side}Zf", f"{side}D"], "plane": "Occlusal", "name": f"{side} roll (z rotation)"},
            {"kind": "plane_distance", "landmarks": [f"{side}sOr"], "plane": "Paracoronal", "name": f"{side} superior orbital AP"},
            {"kind": "plane_distance", "landmarks": [f"{side}lOr"], "plane": "Paracoronal", "name": f"{side} lateral orbital AP"},
            {"kind": "plane_distance", "landmarks": [f"{side}iOr"], "plane": "Paracoronal", "name": f"{side} inferior orbital AP"},
            {"kind": "plane_distance", "landmarks": [f"{side}sOr"], "plane": SAGITTAL_PLANE, "name": f"{side} superior orbital ML"},
            {"kind": "plane_distance", "landmarks": [f"{side}lOr"], "plane": SAGITTAL_PLANE, "name": f"{side} lateral orbital ML"},
            {"kind": "plane_distance", "landmarks": [f"{side}iOr"], "plane": SAGITTAL_PLANE, "name": f"{side} inferior orbital ML"},
            {"kind": "plane_distance", "landmarks": [f"{side}sOr"], "plane": "Occlusal", "name": f"{side} superior orbital vertical"},
            {"kind": "plane_distance", "landmarks": [f"{side}lOr"], "plane": "Occlusal", "name": f"{side} lateral orbital vertical"},
            {"kind": "plane_distance", "landmarks": [f"{side}iOr"], "plane": "Occlusal", "name": f"{side} inferior orbital vertical"},
            {"kind": "plane_distance", "landmarks": [f"{side}Zf"], "plane": "Occlusal", "name": f"{side} Zf vertical"},
            {"kind": "plane_distance", "landmarks": [f"{side}D"], "plane": "Occlusal", "name": f"{side} dacryon vertical"},
            # canthal tilt: the line from the lateral to the medial canthus against the Occlusal plane (0 to 90 degrees), plus
            # a signed value that is positive when the lateral canthus is superior to the medial one ("first_superior")
            {"kind": "line_plane_angle", "landmarks": [f"{side}LC", f"{side}MC"], "plane": "Occlusal", "name": f"{side} canthal tilt",
             "signed": "first_superior"},
            {"kind": "plane_distance", "landmarks": [f"{side}MC"], "plane": "Paracoronal", "name": f"{side} medial canthus AP"},
            {"kind": "plane_distance", "landmarks": [f"{side}MC"], "plane": SAGITTAL_PLANE, "name": f"{side} medial canthus mediolateral"},
            {"kind": "plane_distance", "landmarks": [f"{side}MC"], "plane": "Occlusal", "name": f"{side} medial canthus vertical"},
            {"kind": "plane_distance", "landmarks": [f"{side}LC"], "plane": "Paracoronal", "name": f"{side} lateral canthus AP"},
            {"kind": "plane_distance", "landmarks": [f"{side}LC"], "plane": SAGITTAL_PLANE, "name": f"{side} lateral canthus mediolateral"},
            {"kind": "plane_distance", "landmarks": [f"{side}LC"], "plane": "Occlusal", "name": f"{side} lateral canthus vertical"},
        ]
    return specs


# Extra columns: the difference between two named measurements (the minuend minus the subtrahend, both in mm).
PRESET_DIFFERENCES = [
    {"name": "orbital symmetry (D)", "minuend": "R dacryon vertical", "subtrahend": "L dacryon vertical"},
    {"name": "orbital symmetry (Zf)", "minuend": "R Zf vertical", "subtrahend": "L Zf vertical"},
]

# ---------------------------------------------------------------------------
# Internal names and texts: you normally do not need to change anything below.
# ---------------------------------------------------------------------------

PRE_CASE = "PRE"          # the case used as the preoperative set when it exists
MIRROR_PREFIX = "mir"     # the symmetrized set is "mir<PRE set name>", measured as the case "mir<PRE case>"
ALIGN_SUFFIX = "_align"
SIDES = {"R": "Right", "L": "Left"}
MIRROR_ATTRIBUTE = "SymmetrizedMirrorOf"      # set on the mirrored landmark set (holds the PRE set's name)
MODEL_ATTRIBUTE = "SymmetrizedFrom"           # set on the symmetrized model (holds the source's name)
ALIGN_ATTRIBUTE = "SymmetrizedAlignmentOf"   # set on the _align lists (holds the source set's name)
REGISTRATION_ATTRIBUTE = "SymmetrizedRegistration"   # set on the InitialAlign transform and the wizard node
REGISTRATION_NODE_NAME = "InitialAlign registration"   # the Fiducial Registration Wizard node
POST_CASE = "POST"        # the case registered to PRE_CASE
SEGMENTATION_ATTRIBUTE = "SymmetrizedHeadSkullOf"   # set on the _HeadSkull segmentations (holds the volume's name)
KEPT_SEGMENTS_ATTRIBUTE = "SymmetrizedKeptSegments"   # the TOTALSEG_KEEP_SEGMENTS it was made with
SEGMENTATION_METHOD_ATTRIBUTE = "SymmetrizedSegmentationMethod"   # "manual" for a segmentation made by threshold
SUPERIOR_ATTRIBUTE = "SuperiorSide"   # set on reference planes: "+1" if the normal points superior, "-1" if inferior

NO_LANDMARKS_MESSAGE = (
    "No landmark sets (fiducial markups) with positioned points were found in the scene. Load the cases' scenes "
    "(.mrb), or landmarks with slicer.util.loadMarkups(path), then run this again."
)

MEASUREMENT_KINDS = {
    "Distance (2 landmarks)": ("distance", ["Point 1", "Point 2"]),
    "Angle (3 landmarks, measured at the middle one)": ("angle", ["Point 1", "Vertex (the angle is measured here)", "Point 3"]),
}

PLANE_KINDS = {
    "Distance from a landmark to a plane (perpendicular)": ("plane_distance", ["Landmark"]),
    "Angle between an end point and the perpendicular from a vertex to a plane": ("plane_angle", ["End point", "Vertex (the perpendicular to the plane starts here)"]),
    "Angle between a line (2 landmarks) and a plane": ("line_plane_angle", ["Line start", "Line end"]),
}

PLANE_SIZE_MM = 1000.0

PLANE_TYPES = [
    "Through 3 or more landmarks",
    "Perpendicular to an existing plane, through 1 landmark",
]

ORIENTATIONS = {
    "x": "Parallel to the x direction of the plane",
    "y": "Parallel to the y direction of the plane",
}

SEPARATORS = "_-. :"
_SET_SUFFIX = re.compile(r"[_\-. ]*(landmarks?|fiducials?|points?|markups?)$", re.IGNORECASE)

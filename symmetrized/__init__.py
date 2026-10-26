r"""Symmetrized model and landmarks: mirror one side of the preoperative skull and landmarks across the sagittal plane,
and measure every landmark set (PRE, the symmetrized set, POST, ...) against reference planes made from the PRE landmarks.

Standalone package: it needs only 3D Slicer (with its VTK and NumPy) and, for InitialAlign, the SlicerIGT extension.
See README.md in this folder. Its modules are
settings (everything you might change), scene, planes, mirroring, model, measurements, dialogs, placement,
registration and workflow.

The scene holds one landmark set (fiducial markups node) per case, named after its volume
("PRE_Landmarks", "POST_Landmarks", labels "N", "Se", ... or "PRE_N", ...), with corresponding landmark names. The
measurements use world positions, so the POST landmarks count as aligned with PRE once you put them under FinalAlign
(the tool makes InitialAlign and FinalAlign but does not apply them; see README.md). From the Python console:

    import sys; sys.path.append(r"C:\path\to\the\folder\that\holds")   # the folder that contains symmetrized/
    import symmetrized
    symmetrized.start()     # reloads every module (so edits to settings.py count), then runs

What run() does, in order:
S. Segmentation (segment="auto" by default; "manual" or False): the PRE and POST volumes are segmented with the
   TotalSegmentator extension (head: craniofacial structures, high order resampling), keeping head, skull, mandible and
   teeth in PRE_HeadSkull / POST_HeadSkull; or you make head and skull yourself by thresholding in the Segment Editor
   (largest island kept) (segmentation.py). This comes first so the landmarks can be placed on it: while a set is
   placed, its segmentation is shown in 3D with the head (skin) hidden. After each landmark set you can add
   additional landmarks of your own (asked for in POST too).
0. Asks whether to define new landmarks or use landmarks already in the scene (landmarks=False skips the question,
   landmarks=True goes straight to defining). Defining new ones walks you through every point with a small panel
   ("Please set the landmark for the: Crista Galli (CG)"): first the preoperative set "PRE_Landmarks" (settings.LANDMARKS plus
   settings.PRE_ONLY_LANDMARKS: the incisal edges and at least one occluding molar surface per side, more if you like), then the
   postoperative set "POST_Landmarks" (settings.LANDMARKS). Each point is labeled with its abbreviation automatically. Then the
   lists "PRE_align" and "POST_align" are made, with CG, Se, Ba, ANS, RM, LM in that order (settings.ALIGN_LANDMARKS), for a
   manual fiducial registration; they are also made from landmarks already in the scene. Then SlicerIGT's Fiducial
   Registration Wizard makes the rigid transform "InitialAlign" from them: From POST_align, To PRE_align, Rigid,
   manual point matching, updated once (registration.py; initial_align=False skips it). It is not applied to any node.
   Then General Registration (BRAINS) registers the POST volume to the PRE volume starting from InitialAlign: sampling
   0.002, Rigid (6 DOF) only, output the transform "FinalAlign" (final_align=False skips it). FinalAlign already
   includes InitialAlign. FinalAlign is then applied (a soft transform) to the POST volume, the POST landmark set and
   every other node named POST... (POST_Segmentation, ...), before anything is calculated (apply_final=False leaves it
   unapplied). InitialAlign itself is not applied.
1. Picks the preoperative landmark set: the case named "PRE", else you choose (pre="..." skips the question).
2. Makes the reference planes from the PRE landmarks (settings.REFERENCE_PLANES), before anything is mirrored:
     "Occlusal"     best fit to RIn, LIn and every PRE landmark whose name contains "Oc" (case-sensitive);
     "Sagittal"     best fit to CG, Se, Ba, ANS, constrained to be perpendicular to Occlusal (the plane through those
                    landmarks, among the planes that contain Occlusal's normal, with the smallest RMS distance);
     "Paracoronal"  perpendicular to Sagittal and Occlusal, through Se.
   If Occlusal cannot be made, Sagittal is an unconstrained best fit (reported) and Paracoronal is skipped.
   sagittal="plane name" uses an existing plane instead of making Sagittal.
3. Asks which side to keep and mirror: Right or Left (side="R" / "L" skips the question). Which side of the plane is
   the patient's right is found from the paired landmarks (RZf/LZf, ...): the side where the R ones lie. Without
   pairs it falls back to the world +x direction (right in RAS).
4. Symmetrized model: the preoperative model, or segmentation (all its segments, as a closed surface), is cut at the
   sagittal plane, the kept half is mirrored across it and the two halves are joined into one new model,
   "<source>_symmetrized_<side>". A choice dialog appears when the scene has several models/segmentations
   (source="..." skips it). The source is not changed. With no model or segmentation in the scene this step is
   skipped and reported.
5. Symmetrized landmark set "mir<PRE set name>" (mirPRE_Landmarks): a full R and L set made only from the kept side.
   The PRE landmarks of the kept side (names starting with its letter, R or L; not those used to make the sagittal
   plane) are copied into it unchanged, and each is also mirrored across the sagittal plane and labeled for the side it
   now lies on. Keep R: RZf is copied as RZf, and its mirror image is LZf. The PRE landmarks of the other side and
   those without a side letter are not used. It is measured as the case "mirPRE".
6. Preset measurements (settings.preset_measurement_specs(): the per-side pitch/yaw/roll and orbital AP/ML/vertical set of the
   shared script) and the settings.PRESET_DIFFERENCES columns. You are then asked whether to add more planes and measurements,
   with the same dialogs as the shared script (on the PRE landmarks).
7. Every measurement is calculated for every landmark set: PRE, mirPRE, then POST and any others, printed in the
   terminal and saved to a CSV (one row per set). A set that lacks a landmark a measurement needs (mirPRE has no N,
   Se, ...) gets a blank cell.

Every plane is 1000 x 1000 mm, drawn without axes, handles or a normal arrow. Re-running replaces the planes, the
mirrored set and the symmetrized model this script made earlier (nodes of those names it did not make are an error
for planes, and are left alone, with a new name, for the model). The mirrored landmarks and the model are a snapshot:
moving the PRE landmarks or the planes afterwards does not update them; run again.

run() returns {"planes", "measurements", "model", "mirrored", "side"}. The parts work without the dialogs too:
define_landmarks(), place_landmarks(), create_alignment_lists(), find_cases(), make_reference_planes(),
fit_plane_perpendicular_to(), right_side_normal(), mirror_landmarks(),
symmetrize_polydata(), create_symmetrized_model(), evaluate(), finalize_measurements().
"""

import importlib
import sys

from .scene import short_name, landmark_names, volume_names, case_id, find_cases, landmark_position
from .planes import (
    fit_plane, fit_plane_perpendicular_to, existing_planes, plane_names, get_plane, plane_geometry, plane_axes,
    perpendicular_to_planes_geometry, create_plane_from_points, create_plane, make_reference_planes,
)
from .mirroring import reflection_matrix, paired_names, right_side_normal, mirror_landmarks, side_check
from .model import model_sources, source_polydata, symmetrize_polydata, create_symmetrized_model
from .dialogs import ask_side
from .measurements import (
    preset_measurements, measure, measure_angle_to_plane, measure_line_to_plane_angle, measure_to_plane, describe,
    column_headers, spec_title, usable_differences, difference_results, evaluate, export_measurements_csv,
    finalize_measurements,
)
from .placement import place_landmarks, define_landmarks, create_alignment_lists
from .segmentation import segment_volumes, segment_head_and_skull, segment_volumes_manually, segment_manually
from .registration import (
    find_volume, fiducial_registration, intensity_registration, post_under_final_align, apply_final_align,
)
from .workflow import generate_planes, select_measurements, run
from . import dialogs, mirroring, measurements, model, placement, planes, registration, scene, segmentation, settings, workflow

_MODULES = ["settings", "scene", "planes", "mirroring", "model", "dialogs", "measurements", "placement", "registration",
            "segmentation", "workflow"]


def start(**options):
    """Reload every module of the package (so changes to settings.py or any other file count without restarting
    Slicer), then run the tool. options are passed to run(), for example start(landmarks=False, side="R")."""
    for name in _MODULES:
        importlib.reload(sys.modules[f"{__name__}.{name}"])
    package = importlib.reload(sys.modules[__name__])
    return package.workflow.run(**options)

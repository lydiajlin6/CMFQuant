# symmetrized: symmetrized model, landmarks and measurements for 3D Slicer

A self-contained 3D Slicer tool for craniofacial planning in asymmetric cases. From manually placed preoperative (PRE)
and postoperative (POST) landmarks it:

1. walks you through placing every landmark, with each point labeled for you;
2. registers POST to PRE, first from six landmarks (Fiducial Registration Wizard → `InitialAlign`) and then from the
   images (General Registration (BRAINS) → `FinalAlign`), and puts the POST data under `FinalAlign`;
3. makes the reference planes (Occlusal, Sagittal, Paracoronal) from the PRE landmarks;
4. asks which side to keep (Right or Left), and builds a **symmetrized model** (the PRE model cut at the sagittal plane,
   the kept half mirrored and joined) and a **symmetrized landmark set** (the kept side plus its mirror image);
5. measures every landmark set (PRE, the symmetrized set, POST) against the reference planes, prints the results and
   saves them to a CSV.

**This is a research tool, not a validated clinical device.** Check every plane, registration and measurement
on your own data before relying on it.

## Requirements

- **3D Slicer** 5.x (developed and tested on 5.12.3, Windows). Everything it needs (Python, NumPy, VTK, Qt, the
  General Registration (BRAINS) module) comes with Slicer.
- **SlicerIGT** extension (Extensions Manager, then restart Slicer), for the Fiducial Registration Wizard that makes
  `InitialAlign`. Without it that step is skipped and reported; everything else still works.
- **TotalSegmentator** extension (with its PyTorch dependency), for the segmentation the landmarks are placed on (on
  by default; it can be switched off). A GPU makes it much faster. Without it that step is skipped and reported.
- Nothing else: the folder does not use any other file or project.

## Installation (once)

1. Copy the whole `symmetrized` folder anywhere (keep the folder name, and keep `SlicerModule` inside it).
2. Start Slicer and **drag the `SlicerModule` folder** (inside `symmetrized`) from File Explorer onto the Slicer
   window. (Dropping the file `SlicerModule\SymmetrizedPlanning.py` works too; dropping the `symmetrized` folder
   itself does not, because Slicer skips folders that contain Python package files.)
3. Slicer asks "The following module can be loaded. Would you like to load it now?". Keep **Add selected module to
   'Additional module paths'** ticked (this makes it permanent: Slicer loads it every time it starts). "Enable developer
   mode" is not needed; you can untick it. Click **Yes**.

The tool is now in the Modules menu under **Craniofacial > Symmetrized Planning**. To remove it later: Edit >
Application Settings > Modules, select its path under "Additional module paths" and remove it.

## Quick start

1. In Slicer, load the preoperative scan and name the volume **`PRE`**, the postoperative scan named **`POST`**, and
   the preoperative model or segmentation of the skull. Name other postoperative data so it starts with `POST`
   (for example `POST_Segmentation`), so it is aligned with the rest.
2. Open **Modules > Craniofacial > Symmetrized Planning** and click **Start**. (The segmentation checkbox is ticked by
   default; untick it if you do not want the scans segmented first.)
3. Answer the questions. At the end a save dialog asks where to put the CSV.

The module's **Open the instructions (README)** button opens this file.

Without installing the module, the same can be started from the Python console (View > Python Console), with your own
path to the folder that **contains** `symmetrized`:

```python
import sys; sys.path.append(r"C:\path\to\the\folder\that\contains")
import symmetrized
result = symmetrized.start()
```

`start()` (which is also what the Start button runs) reloads every file of the package first, so changes you make (for
example to `settings.py`) take effect without restarting Slicer.

## What happens, step by step

### 0. Segmentation (first, on by default)

Before anything else, each of the `PRE` and `POST` volumes is segmented, so that the landmarks can be placed on it. In
the module the checkbox **Segment head (skin), skull, mandible and teeth of PRE and POST with TotalSegmentator first** is
ticked by default; untick it (or call `start(segment=False)`) to skip this step. The TotalSegmentator extension's
**head: craniofacial structures** task (`craniofacial_structures`) is run with the advanced option **high order
resampling** on. It produces mandible, lower teeth, skull, head, maxillary sinus, frontal sinus and upper teeth; the two
**sinus segments are removed** and **head, skull, mandible, lower teeth and upper teeth** are kept
(`TOTALSEG_KEEP_SEGMENTS`), in one segmentation per volume: `PRE_HeadSkull` and `POST_HeadSkull`. Each is placed in
its volume's patient/study folder in the Data module and under the same transform as its volume; `POST_HeadSkull` is
later put under `FinalAlign` with the other POST data.

- On a real CT here it took 1 to 2 minutes per volume on the GPU (much longer on a CPU).
- A segmentation made earlier is reused on later runs (delete it to segment that volume again). One made with a
  different list of kept segments (for example the earlier head + skull only version) is made again automatically.
- TotalSegmentator writes one label per voxel, so the segments do not overlap: the **head** segment does not contain
  the voxels labeled as skull, mandible, teeth or sinuses (its outer surface is the skin, with hollows where those
  structures are), and the removed sinuses leave empty spaces.
- TotalSegmentator's Python package and model weights are downloaded the first time if they are missing (needs
  internet; Slicer may ask to restart once).
- If the extension is missing, it fails or a segment is missing, this is reported and the rest of the tool carries on.
  With no `PRE` or `POST` volume in the scene (landmarks only) the step is skipped with a note in the terminal.

### 1. Landmarks

The first question: **Define new landmarks** or **Use landmarks already set**.

Defining walks you through every point with a small window that stays open beside the views:
"Please set the landmark for the: Crista Galli (CG)", "Point 1 of 24". Click the point on the scan (slice view) or the
model (3D view); it is labeled with its abbreviation (and the full name as its description) and the window moves on.
A wrongly labeled or renamed point is put right, and an extra click after the last point is removed.

- **Place point** turns placing back on (for example after pressing Esc), **Undo last point** removes the last one,
  **Cancel** stops.
- The volume named `PRE` (then `POST`) is shown in the slice views when there is one, and its segmentation from step 0
  (`PRE_HeadSkull`, then `POST_HeadSkull`) in the 3D view, so points can be clicked on the bone. While you place them
  the **head (skin) segment is hidden** in 3D (it would cover the skull) and the other volume's segmentation is hidden;
  both are shown again afterwards (`HIDDEN_WHILE_PLACING` in `settings.py`). For the **soft tissue landmarks** (the
  canthi, `SOFT_TISSUE_LANDMARKS`) the head (skin) segment is shown, so they can be clicked on the skin.
- Only the landmark set being placed is shown: the other point lists (for example `PRE_Landmarks` while
  `POST_Landmarks` is placed) are hidden while you place, and shown again afterwards (a list you had hidden
  yourself stays hidden).
- A landmark set of that name already in the scene is replaced only if you say so.

| Set | Landmarks, in order |
| --- | --- |
| `PRE_Landmarks` (24 points, plus any extra molar points) | CG Crista Galli, Se Sella, Ba Basion, ANS Anterior Nasal Spine, then LIn Left Central Incisal Edge, RIn Right Central Incisal Edge, LOc1 and ROc1 (an occluding surface of the molars on each side; then the question about more molar points), then LM Left Mastoid, RM Right Mastoid, LD Left Dacryon, LZf Left Zygomaticofrontal Suture, LsOr Left Superior Orbitale, LlOr Left Lateral Orbitale, LiOr Left Inferior Orbitale, RD, RZf, RsOr, RlOr, RiOr (the same on the right), then the soft tissue landmarks LMC Left Medial Canthus, LLC Left Lateral Canthus, RMC Right Medial Canthus, RLC Right Lateral Canthus |
| `POST_Landmarks` (20 points) | the same without the incisal and occlusal points: CG, Se, Ba, ANS, LM, RM, LD, LZf, LsOr, LlOr, LiOr, RD, RZf, RsOr, RlOr, RiOr, LMC, LLC, RMC, RLC |

Right after LOc1 and ROc1 you are asked "Add another occluding surface point of the molars?" (Right / Left / Done);
extra points are ROc2, LOc2, ..., then the list carries on with LM. All of them feed the Occlusal plane. The Oc points carry the side letter so they can be
mirrored and paired.

With **Use landmarks already set**, the landmark sets already in the scene are used. One set per case, named after its
volume (`PRE_Landmarks`, `POST_Landmarks`; labels `CG`, ... or `PRE_CG`, ...), with the same landmark names in each.

### 2. Alignment lists

`PRE_align` and `POST_align` are made: CG, Se, Ba, ANS, RM, LM in that order, copied from each set where the points are
shown (world positions, with no parent transform). They are used only for the registration and are not measured.

### 3. InitialAlign (Fiducial Registration Wizard)

SlicerIGT's Fiducial Registration Wizard: From fiducials `POST_align`, To fiducials `PRE_align`, result transform type
**Rigid**, point matching **Manual** (the points pair up in list order), update mode Manual, output a new linear
transform **`InitialAlign`**, updated once. It maps POST onto PRE; its RMS error is printed and stored on it as the
attribute `FiducialRMS_mm`.

The wizard reads each list's own coordinates and ignores parent transforms, which is why the alignment lists hold world
positions.

### 4. FinalAlign (General Registration (BRAINS))

Slicer's BRAINSFit CLI module with these settings:

| BRAINS setting | Value |
| --- | --- |
| Fixed image volume | `PRE` |
| Moving image volume | `POST` |
| Percentage of samples | 0.002 (a fraction: 0.2% of PRE's voxels) |
| Initialization transform | `InitialAlign` (Initialize Transform Mode Off, as BRAINSFit requires with one) |
| Registration phases | Rigid (6 DOF) only |
| Slicer Linear Transform output | a new transform **`FinalAlign`** |

BRAINSFit's output already includes the initialization transform, so **FinalAlign is the whole POST-to-PRE transform:
use it instead of InitialAlign, not on top of it**. The terminal shows how far FinalAlign moved POST from where
InitialAlign put it (degrees, and mm at the POST volume's centre), and the PRE_align/POST_align RMS under each
transform. Landmarks on bone the surgery moved legitimately disagree.

`PRE` must not be under a transform. `POST` may be: it is ignored while registering, so InitialAlign is not counted
twice. With no `PRE` or `POST` volume in the scene (landmarks only) this step is skipped with a note in the terminal.
On real scans it can take a while.

### 4b. FinalAlign is applied to the POST data

Before anything is calculated, the POST data are put under `FinalAlign` (a soft transform; nothing is hardened, so it
can be undone in the Data or Transforms module): the `POST` volume, the POST landmark set, and every other node whose
name starts with `POST` as a whole word (`POST_Segmentation`, `POST skull`, ... but not `POSTER`). A POST node under
`InitialAlign` is moved to `FinalAlign` (which already includes InitialAlign). A POST node under some other transform
is left alone and reported. `POST_align` is never moved: it holds copied world positions. The terminal lists what was
moved ("Applied FinalAlign to ..."). `InitialAlign` itself is not applied to anything.

### 5. Reference planes (from the PRE landmarks)

| Plane | Definition |
| --- | --- |
| `Occlusal` | best fit to RIn, LIn and every PRE landmark whose name contains "Oc" |
| `Sagittal` | best fit to CG, Se, Ba, ANS, constrained to be perpendicular to Occlusal (of the planes that contain the occlusal normal, the one with the smallest RMS distance to the four landmarks; the RMS is printed and stored as `FitRMS_mm`) |
| `Paracoronal` | perpendicular to Sagittal and Occlusal, through Se |

If Occlusal cannot be made (for example no RIn), Sagittal is an unconstrained best fit and Paracoronal is skipped;
both are reported. Every plane is 1000 × 1000 mm, drawn without handles.

### 6. Right or Left

A question box asks which side to keep and mirror. Which side of the sagittal plane is the patient's right is found
from the R/L landmark pairs (the side the R ones lie on); without pairs, world +x (right in RAS) is used. A landmark
named R... that lies on the left of the plane (or the reverse) is reported.

### 7. Symmetrized model and landmark set

- **Model** `<source>_symmetrized_R` (or `_L`): the PRE model, or segmentation (all its segments, as closed surfaces),
  is cut at the sagittal plane; the kept half is mirrored and the halves are joined. A closed surface stays closed;
  the source is not changed. You choose the source when the scene has several models or segmentations; with none this
  step is skipped and reported.
- **Landmark set** `mirPRE_Landmarks`: a full R and L set made only from the kept side. Keep Right: the PRE right
  landmarks are copied (RZf, RD, ...) and each is followed by its mirror image, labeled for the side it now lies on
  (LZf, LD, ...). PRE landmarks of the other side and those without a side letter (CG, Se, Ba, ANS, ...) are not used.

### 8. Measurements and CSV

You are asked whether to add planes and measurements (distance with dx/dy/dz, angle at a vertex, landmark-to-plane
distance, angle to the perpendicular from a plane, line-to-plane angle) on the PRE landmarks. Then everything is
calculated for every landmark set, printed in the terminal and saved to a CSV (default
`symmetrized_measurements.csv`).

Preset measurements, for each side (R, then L):

| Name | Definition |
| --- | --- |
| pitch (x rotation) | angle at sOr between iOr and the perpendicular to Paracoronal |
| yaw (y rotation) | angle at D between Zf and the perpendicular to Paracoronal |
| roll (z rotation) | angle at D between Zf and the perpendicular to Occlusal |
| superior / lateral / inferior orbital AP | distance from sOr / lOr / iOr to Paracoronal |
| superior / lateral / inferior orbital ML | distance from sOr / lOr / iOr to Sagittal |
| superior / lateral / inferior orbital vertical | distance from sOr / lOr / iOr to Occlusal |
| Zf vertical, dacryon vertical | distance from Zf / D to Occlusal |
| canthal tilt | angle between the line from the lateral to the medial canthus (LC → MC) and Occlusal, 0 to 90°, plus a signed column |
| medial canthus AP / mediolateral / vertical | distance from MC to Paracoronal / Sagittal / Occlusal |
| lateral canthus AP / mediolateral / vertical | distance from LC to Paracoronal / Sagittal / Occlusal |

Distances give an unsigned and a signed column. The canthal tilt's signed column is **positive when the lateral canthus
is superior to (higher than) the medial canthus**, on both sides and whichever way the Occlusal plane's normal points.
"Superior" is the side of the Occlusal plane that the PRE Crista Galli (CG, `SUPERIOR_LANDMARK`) is on, recorded on the
plane when it is made; a plane made without it uses world superior (+z). Two more columns: **orbital symmetry (D)** = R dacryon vertical − L
dacryon vertical, and **orbital symmetry (Zf)** = R Zf vertical − L Zf vertical.

The CSV has one row per landmark set: `PRE`, `mirPRE` (the symmetrized set; its R and L values match, so the symmetry
columns are 0), then `POST` and any others. A set that lacks a landmark gets a blank cell (`mirPRE` has no CG, Se, ...).
Values are in mm and degrees, in world RAS (x right+, y anterior+, z superior+).

## Running again after the registration

Because the POST data are now under `FinalAlign`, a later run (choose **Use landmarks already set**) **skips the
registration automatically**: the terminal says "Skipping InitialAlign and FinalAlign: ... already under FinalAlign",
and `InitialAlign`, `FinalAlign`, `PRE_align` and `POST_align` are kept as they are. POST data that is not yet under
`FinalAlign` (for example a segmentation loaded since) is still put under it.

This also counts when POST data are under another transform that is itself under `FinalAlign`. Registering again would
start from POST positions that are already aligned (InitialAlign would be close to identity, and BRAINS, which ignores
POST's own transform, would start from the wrong place). To register again, take the POST data off `FinalAlign` first
(Data module: drag them out of the transform, or set their transform to None).

## Options

`symmetrized.start(...)` and `symmetrized.run(...)` take the same options:

| Option | Default | Meaning |
| --- | --- | --- |
| `landmarks` | `"ask"` | `True`: define new landmarks; `False`: use those in the scene (no question) |
| `initial_align` | `True` | make InitialAlign with the Fiducial Registration Wizard |
| `final_align` | `True` | make FinalAlign with General Registration (BRAINS) |
| | | (both registration steps are skipped automatically when POST is already under FinalAlign, see above) |
| `apply_final` | `True` | put the POST data under FinalAlign before anything is calculated |
| `segment` | `True` | first of all, segment head/skull/mandible/teeth of PRE and POST with TotalSegmentator (the module's checkbox) |
| `pre` | `"PRE"` (else asked) | the preoperative landmark set's case name |
| `side` | asked | `"R"` or `"L"`: the side to keep |
| `source` | the only one, else asked | the model or segmentation to symmetrize |
| `sagittal` | made | the name of an existing plane to use as the sagittal plane |
| `presets` | `True` | include the preset measurements |
| `csv_path` | save dialog | where to write the CSV |

`run()` returns a dict with `planes`, `measurements`, `model`, `mirrored`, `side`, `initial_align`, `final_align` and
`registration` (the BRAINS summary).

## Changing the landmarks, planes and measurements

Edit **`settings.py`**, then run `symmetrized.start()` again. The top part is meant to be edited:

- `LANDMARKS`, `PRE_ONLY_LANDMARKS`: (abbreviation, full name) pairs, in the order they are asked; `PRE_ONLY_AFTER`
  (`"ANS"`): the landmark after which the PRE-only points are asked; `SOFT_TISSUE_LANDMARKS`: placed with the head (skin)
  segment shown.
- `ALIGN_LANDMARKS`: the landmarks copied into the alignment lists, in order.
- `INITIAL_TRANSFORM_NAME`, `FINAL_TRANSFORM_NAME`, `BRAINS_SAMPLING_PERCENTAGE`.
- `HIDDEN_WHILE_PLACING` (`["head"]`): segments hidden in 3D while landmarks are placed.
- `TOTALSEG_TASK`, `TOTALSEG_KEEP_SEGMENTS` (`["head", "skull", "mandible", "teeth_lower", "teeth_upper"]`), `TOTALSEG_HIGHER_ORDER_RESAMPLING`, `SEGMENTATION_SUFFIX`.
- `SAGITTAL_LANDMARKS`, `REFERENCE_PLANES`: how the planes are made.
- `SUPERIOR_LANDMARK` (`"CG"`): the PRE landmark that tells which side of a reference plane is superior.
- `preset_measurement_specs()`, `PRESET_DIFFERENCES`: the preset measurements and difference columns.

Below the divider are internal names you normally do not need to change.

## Re-running

Running again replaces the planes, `mirPRE_Landmarks`, the alignment lists and the symmetrized model this tool made,
and updates `InitialAlign` and `FinalAlign` in place (whatever you put under them stays under them). A reference plane
that cannot be made this time is removed rather than left from the earlier run. The mirrored landmarks and the model
are a snapshot: after moving a PRE landmark or a plane, run again.

## Files

| File | What is in it |
| --- | --- |
| `settings.py` | everything you might change |
| `workflow.py` | `run()`: the order of the steps and the plane/measurement pickers |
| `placement.py` | the guided landmark placement window, `define_landmarks()`, the alignment lists |
| `registration.py` | `fiducial_registration()` (InitialAlign), `intensity_registration()` (FinalAlign), `apply_final_align()` |
| `segmentation.py` | the optional TotalSegmentator segmentation (`segment_volumes()`) |
| `planes.py` | plane fitting, including the constrained sagittal fit, and the reference planes |
| `mirroring.py` | which side is right, `mirPRE_Landmarks` |
| `model.py` | the symmetrized model |
| `measurements.py` | the measurements, presets, terminal output and CSV |
| `scene.py` | landmark sets and case names in the scene, small helpers |
| `dialogs.py` | the question boxes and pickers |
| `SlicerModule/SymmetrizedPlanning.py` | the Slicer module (Modules > Craniofacial > Symmetrized Planning, Start button) |
| `tests/` | the automated test (below) |

Each step also works on its own from Python, for example `symmetrized.define_landmarks()`,
`symmetrized.fiducial_registration()`, `symmetrized.intensity_registration(sampling_percentage=0.01)`,
`symmetrized.make_reference_planes(case)`, `symmetrized.create_symmetrized_model(...)`. See each function's docstring.

## Testing

`tests/test_symmetrized_slicer.py` runs the real code in Slicer, with the dialogs clicked by a timer and expected
values computed independently (NumPy and SciPy, not through the code under test):

```
"C:\path\to\Slicer.exe" --no-splash --python-script "C:\path\to\symmetrized\tests\test_symmetrized_slicer.py"
```

Slicer opens, runs the checks and closes; the results are in `tests/symmetrized_slicer_results.txt` (ending in
ALL PASSED or FAILURES). It needs SlicerIGT installed. It covers the guided placement (with simulated clicks), the
alignment lists, InitialAlign against an independent least-squares fit and a known transform, FinalAlign on a phantom
with a known misalignment, the constrained sagittal fit against a separate minimisation, the symmetrized model (exact
volumes, closed surface), the mirrored landmarks, the measurements and the CSV, FinalAlign being applied, and the
Slicer module loaded through Slicer's own drag-and-drop routine (with the dialog's checkboxes unticked, so your
settings are not changed; if the module is already installed in that Slicer, this part is noted and skipped) and its
Start button, and the TotalSegmentator segmentation with TotalSegmentator replaced by a stand-in (the settings it is
given, removing the sinuses, placement, reuse, re-making an outdated one, failures, that it runs first and by
default, and what is shown and hidden in 3D while landmarks are placed, including the other landmark sets).

## Not verified

- Nothing has been run on real PRE/POST patient scans yet: in particular how well the registration at 0.002 sampling
  works on real CBCT, and how long it takes. On the test phantom it recovered a known misalignment to about 0.02° and
  0.03 mm.
- Real mouse placement was smoke-tested once (Slicer's own placement in a slice view); the automated test simulates
  clicks. The windows' look and feel has not been checked by a user.
- The module installation was tested by calling the routine Slicer runs on a drop and by starting Slicer with the module
  folder as a module path, not by physically dragging the folder onto the window.
- The TotalSegmentator segmentation was run for real on one CT only (MG_test_scan1, 1 to 2 minutes on the GPU; head,
  skull, mandible, lower and upper teeth found, sinuses removed); the automated test replaces TotalSegmentator with a
  stand-in. Its accuracy has not been assessed.
- The registration uses the whole volume (no mask). If surgically moved bone pulls the result,
  `intensity_registration(mask_mode="ROIAUTO")` restricts it to an automatic head mask.

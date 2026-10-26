"""Headless test for the symmetrized package (the folder above this one). Run with the Slicer main window:
    Slicer.exe --no-splash --python-script test_symmetrized_slicer.py
Then read symmetrized_slicer_results.txt next to this file (ends with ALL PASSED or FAILURES).

The scene is built in a "canonical" head frame where the sagittal plane is x = 0 and the patient's right is +x, then
moved by a rigid transform T so nothing is axis-aligned. Expected values are computed in the canonical frame (a mirror
is x -> -x there; planes are fitted with numpy here) and mapped by T, not through the code under test. In that scene
the PRE occlusal landmarks are symmetric, so the constrained sagittal fit is exactly x = 0; a second scene ("ASYM",
at the end) has asymmetric occlusal landmarks and off-plane sagittal landmarks and checks the constrained fit against
a separate numeric minimisation (scipy).
Written 2026-10-16 against the script as of that date; update the expectations if the presets change."""
import csv, math, os, shutil, sys, tempfile, time, traceback
import numpy as np
import qt
import vtk
import slicer

SP = os.path.dirname(os.path.abspath(__file__))   # results are written here, next to this file
PACKAGE = os.path.dirname(SP)                      # the symmetrized package folder
RESULTS = os.path.join(SP, "symmetrized_slicer_results.txt")
lines = []
ok = [True]


def check(name, cond, extra=""):
    ok[0] &= bool(cond)
    lines.append(("PASS " if cond else "FAIL ") + name + (f"   [{extra}]" if extra else ""))


def unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


def rot(axis, deg):
    a = unit(axis)
    t = math.radians(deg)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + math.sin(t) * K + (1 - math.cos(t)) * K @ K


R_ = rot([0.3, 0.2, 1.0], 25.0)
t_ = np.array([10.0, -5.0, 7.0])
T = np.eye(4); T[:3, :3] = R_; T[:3, 3] = t_
to_world = lambda p: R_ @ np.asarray(p, float) + t_
MIRROR_C = lambda p: np.array([-p[0], p[1], p[2]], float)   # mirror in the canonical frame


def svd_plane(points):
    pts = np.asarray(points, float)
    c = pts.mean(axis=0)
    return c, np.linalg.svd(pts - c)[2][2]


CANON = {
    "CG": (0, 60, 40), "Se": (0, 20, 20), "Ba": (0, -10, -10), "ANS": (0, 70, -30),
    "RZf": (45, 50, 25), "LZf": (-40, 52, 22),
    "RD": (12, 62, 15), "LD": (-14, 60, 13),
    "RsOr": (30, 60, 30), "LsOr": (-28, 58, 31),
    "RlOr": (42, 55, 15), "LlOr": (-40, 54, 12),
    "RiOr": (32, 62, 2), "LiOr": (-30, 60, 0),
    "RIn": (3, 75, -45), "LIn": (-3, 75, -45),          # occlusal landmarks symmetric in this scene
    "ROc6": (25, 40, -40), "LOc6": (-25, 40, -40),
    "ROc3": (15, 60, -42), "LOc3": (-15, 60, -42),
    "N": (0.3, 80, 25), "MidOc": (0, 70, -44),
    "RM": (50, -5, -20), "LM": (-48, -6, -21),
    "RMC": (14, 70, 18), "RLC": (38, 64, 20), "LMC": (-15, 69, 17), "LLC": (-37, 63, 22),
}
OCC = ["RIn", "LIn", "ROc6", "LOc6", "ROc3", "LOc3", "MidOc"]   # the PRE Occlusal landmarks, in the script's order
CANON = {k: np.array(v, float) for k, v in CANON.items()}
SAG = ["CG", "Se", "Ba", "ANS"]
# POST: the left side moved to the mirror of the PRE right side plus 1 mm in canonical x; LlOr not placed.
POST_C = dict(CANON)
for k in list(CANON):
    if k[0] == "L" and "R" + k[1:] in CANON:
        POST_C[k] = MIRROR_C(CANON["R" + k[1:]]) + np.array([1.0, 0, 0])
del POST_C["LlOr"]
FU_C = {k: v + np.array([0, 0, 0.5]) for k, v in CANON.items()}   # follow-up: everything 0.5 mm up (canonical z)


def add_set(name, points, label_prefix=""):
    node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsFiducialNode", name)
    for label, p in points.items():
        node.AddControlPoint([float(v) for v in p], label_prefix + label)
    return node


def vtk_matrix(m):
    out = vtk.vtkMatrix4x4()
    for i in range(4):
        for j in range(4):
            out.SetElement(i, j, float(m[i][j]))
    return out


def tri_arrays(pd):
    pts = np.array([pd.GetPoint(i) for i in range(pd.GetNumberOfPoints())])
    tris = []
    for c in range(pd.GetNumberOfCells()):
        ids = pd.GetCell(c).GetPointIds()
        n = ids.GetNumberOfIds()
        for k in range(1, n - 1):   # fan, in case of polygons
            tris.append((ids.GetId(0), ids.GetId(k), ids.GetId(k + 1)))
    return pts, np.array(tris)


def signed_volume(pts, tris):
    a, b, c = pts[tris[:, 0]], pts[tris[:, 1]], pts[tris[:, 2]]
    return float(np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6.0)


def open_edges(tris):
    count = {}
    for t in tris:
        for e in ((t[0], t[1]), (t[1], t[2]), (t[2], t[0])):
            k = tuple(sorted(e))
            count[k] = count.get(k, 0) + 1
    return sum(1 for v in count.values() if v != 2)


def kabsch(src, dst):
    """Least-squares rigid transform (4x4) taking the points src onto dst (SVD, reflection excluded)."""
    src, dst = np.asarray(src, float), np.asarray(dst, float)
    cs, cd = src.mean(axis=0), dst.mean(axis=0)
    u, _, vt = np.linalg.svd((src - cs).T @ (dst - cd))
    d = np.sign(np.linalg.det(vt.T @ u.T))
    rot = vt.T @ np.diag([1, 1, d]) @ u.T
    m = np.eye(4)
    m[:3, :3] = rot
    m[:3, 3] = cd - rot @ cs
    return m


def world_points(node):
    out = []
    for i in range(node.GetNumberOfControlPoints()):
        q = [0.0] * 3
        node.GetNthControlPointPositionWorld(i, q)
        out.append(q)
    return np.array(out)


def matrix_of(transform_node):
    m = vtk.vtkMatrix4x4()
    transform_node.GetMatrixTransformToParent(m)
    return np.array([[m.GetElement(i, j) for j in range(4)] for i in range(4)])


def raises(fn, exc=ValueError):
    try:
        fn()
    except exc:
        return True
    return False


try:
    _lone = tempfile.mkdtemp()
    shutil.copytree(PACKAGE, os.path.join(_lone, "symmetrized"), ignore=shutil.ignore_patterns("tests", "__pycache__"))
    sys.path.insert(0, _lone)
    import symmetrized as sym
    check("package is standalone (imports with no other project module on the path)", sym.__file__.startswith(_lone))
    check("empty scene -> clear error", raises(lambda: sym.run(landmarks=False), RuntimeError))

    # ---------------- pure helpers ----------------
    check("no mirror-target presets any more", not any("mirror" in sp["name"] for sp in sym.settings.preset_measurement_specs()))
    p_, n_ = np.array([1.0, 2, 3]), unit([1, 1, 0])
    M = sym.reflection_matrix(p_, n_)
    q = np.array([5.0, -1, 4])
    check("reflection_matrix = p - 2((p-o).n)n and is an involution",
          np.allclose((M @ np.append(q, 1))[:3], q - 2 * np.dot(q - p_, n_) * n_) and np.allclose(M @ M, np.eye(4)))
    check("paired_names", sym.paired_names(["RZf", "LZf", "RIn", "N", "R"]) == [("RZf", "LZf")])

    # ---------------- the scene ----------------
    tnode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLinearTransformNode", "HeadPose")
    tnode.SetMatrixTransformToParent(vtk_matrix(T))
    pre_node = add_set("PRE_Landmarks", CANON)                         # canonical coordinates under HeadPose
    pre_node.SetAndObserveTransformNodeID(tnode.GetID())
    post_node = add_set("POST_Landmarks", {k: to_world(v) for k, v in POST_C.items()})   # world coordinates
    fu_node = add_set("FU_Landmarks", {k: to_world(v) for k, v in FU_C.items()}, label_prefix="FU_")
    # a placeholder point in POST, must be ignored
    post_node.AddControlPoint([0, 0, 0], "Ghost")
    post_node.UnsetNthControlPointPosition(post_node.GetNumberOfControlPoints() - 1)

    cube = vtk.vtkCubeSource()
    cube.SetBounds(-20, 40, 0, 60, -40, 20)   # canonical; x from -20 to 40
    cube.Update()
    cube_model = slicer.modules.models.logic().AddModel(cube.GetOutput())
    cube_model.SetName("Cube")
    cube_model.SetAndObserveTransformNodeID(tnode.GetID())

    # a segmentation from a labelmap box, canonical x from -10 to 30 (1 mm voxels), placed by HeadPose too
    arr = np.zeros((40, 50, 60), dtype=np.uint8)   # numpy order k, j, i
    # voxel i -> x = -30 + i ; box i in [20, 60) -> x in [-10, 30) (voxel centres), j -> y = 0..49, k -> z = -20..19
    arr[5:35, 10:40, 20:60] = 1
    lab = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLabelMapVolumeNode", "BoxLabel")
    slicer.util.updateVolumeFromArray(lab, arr)
    lab.SetOrigin(-30, 0, -20)   # RAS origin; IJK->RAS default directions are identity in RAS here
    lab.SetIJKToRASDirections(1, 0, 0, 0, 1, 0, 0, 0, 1)
    seg = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode", "BoxSeg")
    slicer.modules.segmentations.logic().ImportLabelmapToSegmentationNode(lab, seg)
    seg.SetAndObserveTransformNodeID(tnode.GetID())
    slicer.mrmlScene.RemoveNode(lab)

    cases = sym.find_cases()
    check("three cases found, ids from set names / label prefixes", [c["case"] for c in cases] == ["PRE", "POST", "FU"], str([c["case"] for c in cases]))
    check("label prefix stripped, placeholder ignored", sorted(cases[2]["names"]) == sorted(CANON) and "Ghost" not in cases[1]["names"])

    # ---------------- independent expectations (canonical frame) ----------------
    def expected_planes(keep):
        other = "L" if keep == "R" else "R"
        pts = dict(CANON)
        sel = list(OCC)   # Occlusal from the PRE landmarks, whichever side is kept
        occ_o, occ_n = svd_plane([pts[n] for n in sel])
        sag_n = np.array([1.0, 0, 0])
        para_n = unit(np.cross(sag_n, occ_n))
        return pts, sorted(set(sel)), (occ_o, occ_n), para_n

    def plane_world(name):
        node = sym.get_plane(name)
        o, n = [0.0] * 3, [0.0] * 3
        node.GetOriginWorld(o); node.GetNormalWorld(n)
        return np.array(o), unit(n)

    def dist_to_plane_c(p_c, o_c, n_c):
        return abs(float(np.dot(p_c - o_c, unit(n_c))))

    # ---------------- the real dialogs, driven end to end ----------------
    steps, errors, started = [], [], time.time()
    seen = {}

    def widget(w, name):
        return slicer.util.findChild(w, name)

    def msg(button_name):
        def action(w):
            button = w.button(getattr(qt.QMessageBox, button_name))
            if button is None:
                w.accept()
            else:
                button.click()
        return action

    def side_button(text):
        def action(w):
            seen["side_buttons"] = [b.text for b in w.buttons()]
            seen.setdefault("buttons_by_title", {})[w.windowTitle] = [b.text for b in w.buttons()]
            for b in w.buttons():
                if b.text == text:
                    b.click()
                    return
            raise RuntimeError(f"no button {text}")
        return action

    def tick():
        w = slicer.app.activeModalWidget()
        if w is None:
            return
        if time.time() - started > 400:
            errors.append("timeout")
            w.reject()
            return
        if not steps:
            errors.append(f"unexpected dialog {w.windowTitle!r}: {getattr(w, 'text', '')}")
            w.reject()
            return
        expected, fn = steps.pop(0)
        title = (w.windowTitle or "").strip()
        if expected not in title:
            errors.append(f"expected a dialog titled like {expected!r}, got {title!r}: {getattr(w, 'text', '')}")
            steps.clear()
            w.reject()
            return
        try:
            fn(w)
        except Exception:
            errors.append(traceback.format_exc())
            w.reject()

    timer = qt.QTimer()
    timer.setInterval(40)
    timer.timeout.connect(tick)
    timer.start()

    def select_names(w, names):
        listw = widget(w, "landmarkList")
        for r in range(listw.count):
            if listw.item(r).text() in names:
                listw.item(r).setSelected(True)

    def option(index=None, text=None, record=None):
        def action(w):
            combo = widget(w, "optionCombo")
            if record:
                seen.setdefault(record, [combo.itemText(i) for i in range(combo.count)])
            combo.setCurrentIndex(index if index is not None else combo.findText(text))
            widget(w, "okButton").click()
        return action

    def landmark(name):
        def action(w):
            select_names(w, [name])
            widget(w, "okButton").click()
        return action

    def name_it(text):
        def action(w):
            widget(w, "nameEdit").setText(text)
            widget(w, "okButton").click()
        return action

    def plane_picker(w):
        listw = widget(w, "landmarkList")
        seen["picker_rows"] = [listw.item(r).text() for r in range(listw.count)]
        select_names(w, ["LZf", "LD", "LsOr"])
        widget(w, "nameEdit").setText("Tri")
        widget(w, "okButton").click()

    csv1 = os.path.join(SP, "symmetrized_dialog_measurements.csv")
    if os.path.exists(csv1):
        os.remove(csv1)

    def csv_step(w):
        seen["csv_default_name"] = widget(w, "fileNameEdit").text
        widget(w, "fileNameEdit").setText(csv1)
        w.accept()

    steps[:] = [
        ("Landmarks", side_button("Use landmarks already set")),
        ("Side to keep", side_button("Right")),
        ("Preoperative model", option(text="Cube   (model)", record="sources")),
        ("Planes", msg("Yes")),
        ("Plane type", option(index=0)),
        ("Create a plane", plane_picker),
        ("Plane created", msg("Ok")),
        ("Planes", msg("No")),
        ("Measurements", msg("Yes")),
        ("Measurement", option(index=0)),
        ("Distance: landmark 1 of 2", landmark("LZf")),
        ("Distance: landmark 2 of 2", landmark("RZf")),
        ("Name the measurement", name_it("")),
        ("Measurements", msg("No")),
        ("Save measurements CSV", csv_step),
    ]
    logged = []
    sym.scene.log_listeners.append(logged.append)
    out = sym.run()
    sym.scene.log_listeners.remove(logged.append)
    check("scripted dialog run (keep Right) completed with no harness errors", not errors and not steps,
          "; ".join(e.splitlines()[-1] for e in errors) + f" | steps left {len(steps)}")
    check("side question offers Right, Left, Cancel", seen.get("side_buttons", [None])[:2] == ["Right", "Left"] and len(seen.get("side_buttons", [])) == 3, str(seen.get("side_buttons")))
    check("source question lists the model and the segmentation", seen.get("sources") == ["Cube   (model)", "BoxSeg   (segmentation)"], str(seen.get("sources")))
    check("plane picker offers the PRE landmarks only (no mirrored ones)", seen.get("picker_rows") == list(CANON), str(seen.get("picker_rows")))
    check("first question: define new landmarks or use those already set",
          seen.get("buttons_by_title", {}).get("Landmarks") == ["Define new landmarks", "Use landmarks already set", "Cancel"],
          str(seen.get("buttons_by_title", {}).get("Landmarks")))
    for case_, src in (("PRE", CANON), ("POST", POST_C)):
        al = slicer.mrmlScene.GetFirstNodeByName(f"{case_}_align")
        ok_al = al is not None and [al.GetNthControlPointLabel(i) for i in range(al.GetNumberOfControlPoints())] == ["CG", "Se", "Ba", "ANS", "RM", "LM"]
        if ok_al:
            for i, lab_ in enumerate(["CG", "Se", "Ba", "ANS", "RM", "LM"]):
                q_ = [0.0] * 3
                al.GetNthControlPointPositionWorld(i, q_)
                ok_al &= np.linalg.norm(np.array(q_) - to_world(src[lab_])) < 1e-9
        check(f"{case_}_align made from the existing {case_} set: CG, Se, Ba, ANS, RM, LM in that order, same world positions", ok_al)
    ia = slicer.mrmlScene.GetFirstNodeByName("InitialAlign")
    wiz = slicer.mrmlScene.GetFirstNodeByName("InitialAlign registration")
    check("InitialAlign made (a linear transform) and returned by run()", ia is not None and ia.IsA("vtkMRMLLinearTransformNode") and out.get("initial_align") is not None and out["initial_align"].GetID() == ia.GetID())
    check("wizard settings: From POST_align, To PRE_align, output InitialAlign, Rigid, Manual point matching, Manual update",
          wiz is not None and wiz.GetFromFiducialListNode().GetName() == "POST_align" and wiz.GetToFiducialListNode().GetName() == "PRE_align"
          and wiz.GetOutputTransformNode().GetName() == "InitialAlign" and wiz.GetRegistrationMode() == wiz.REGISTRATION_MODE_RIGID
          and wiz.GetPointMatching() == wiz.POINT_MATCHING_MANUAL and wiz.GetUpdateMode() == wiz.UPDATE_MODE_MANUAL,
          f"{wiz.RegistrationModeAsString(wiz.GetRegistrationMode())}, {wiz.PointMatchingAsDisplayableString(wiz.GetPointMatching())}, "
          f"{wiz.UpdateModeAsString(wiz.GetUpdateMode())}" if wiz else "no wizard node")
    pre_al_w = world_points(slicer.mrmlScene.GetFirstNodeByName("PRE_align"))
    post_al_w = world_points(slicer.mrmlScene.GetFirstNodeByName("POST_align"))
    expect_ia = kabsch(post_al_w, pre_al_w)
    got_ia = matrix_of(ia)
    check("InitialAlign = independent least-squares rigid fit of POST_align (world) onto PRE_align (world)",
          np.allclose(got_ia, expect_ia, atol=1e-6), f"max diff {np.abs(got_ia - expect_ia).max():.2e}")
    resid = (got_ia @ np.c_[post_al_w, np.ones(len(post_al_w))].T).T[:, :3] - pre_al_w
    rms_ia = math.sqrt(float(np.mean(np.sum(resid ** 2, axis=1))))
    check("its recorded RMS error = the RMS of the residuals (POST LM sits away from PRE LM in this scene)",
          abs(float(ia.GetAttribute("FiducialRMS_mm")) - rms_ia) < 1e-3 and rms_ia > 0.1, f"{ia.GetAttribute('FiducialRMS_mm')} vs {rms_ia:.4f}")
    check("PRE_align and POST_align have no parent transform (PRE_Landmarks does)",
          all(slicer.mrmlScene.GetFirstNodeByName(n).GetParentTransformNode() is None for n in ("PRE_align", "POST_align"))
          and slicer.mrmlScene.GetFirstNodeByName("PRE_Landmarks").GetParentTransformNode() is not None)
    check("InitialAlign is not applied: no node is under it", not [n for n in slicer.util.getNodesByClass("vtkMRMLTransformableNode")
          if n.GetParentTransformNode() is not None and n.GetParentTransformNode().GetID() == ia.GetID()])
    check("no FU_align (only PRE and POST); the _align lists are not cases",
          slicer.mrmlScene.GetFirstNodeByName("FU_align") is None and [c["case"] for c in sym.find_cases()] == ["PRE", "POST", "FU"])
    check("CSV dialog default name", seen.get("csv_default_name") == "symmetrized_measurements.csv", str(seen.get("csv_default_name")))
    check("run returned side R and the four planes", out.get("side") == "R" and sorted(out["planes"]) == ["Occlusal", "Paracoronal", "Sagittal", "Tri"], str(sorted(out.get("planes", {}))))

    # sagittal plane: canonical x = 0 mapped to the world
    so, sn = plane_world("Sagittal")
    check("Sagittal plane = canonical x=0 in the world (normal parallel to R e_x, origin on it)",
          abs(abs(np.dot(sn, R_ @ [1, 0, 0])) - 1) < 1e-9 and abs(np.dot(so - t_, R_ @ [1, 0, 0])) < 1e-9)
    check("Sagittal is recorded as made perpendicular to Occlusal from CG, Se, Ba, ANS",
          sym.get_plane("Sagittal").GetAttribute("OrthogonalTo") == "Occlusal" and sym.get_plane("Sagittal").GetAttribute("SourceLandmarks") == "CG, Se, Ba, ANS")

    # mirrored landmarks
    mir = out["mirrored"]
    labels = [mir.GetNthControlPointLabel(i) for i in range(mir.GetNumberOfControlPoints())]
    rnames = [k for k in CANON if k.startswith("R")]
    check("mirPRE_Landmarks = the PRE right landmarks copied + their mirrors labeled L (full R and L set); no PRE L, N, MidOc, CG/Se/Ba/ANS",
          mir.GetName() == "mirPRE_Landmarks" and labels == [x for k in rnames for x in (k, "L" + k[1:])], str(labels))
    worst, left_ok = 0.0, True
    for i, lab_ in enumerate(labels):
        p = [0.0] * 3
        mir.GetNthControlPointPositionWorld(i, p)
        expect = CANON[lab_] if lab_.startswith("R") else MIRROR_C(CANON["R" + lab_[1:]])
        worst = max(worst, float(np.linalg.norm(np.array(p) - to_world(expect))))
        if lab_.startswith("L"):
            left_ok &= float(np.linalg.inv(T)[:3] @ np.append(p, 1.0) @ [1, 0, 0]) < 0   # canonical x < 0: patient left
    check("R points = the PRE R landmarks, L points = their mirror in the canonical frame (PRE set was under a transform)", worst < 1e-9, f"max err {worst:.2e}")
    check("the L-labeled (mirrored) points lie on the patient's left", left_ok)
    pre_lzf = [0.0] * 3
    pre_node.GetNthControlPointPositionWorld(list(CANON).index("LZf"), pre_lzf)
    check("the PRE landmarks themselves are unchanged (PRE LZf still at its own position)", np.linalg.norm(np.array(pre_lzf) - to_world(CANON["LZf"])) < 1e-9)
    check("the mirrored set is not a case", [c["case"] for c in sym.find_cases()] == ["PRE", "POST", "FU"])

    # occlusal + paracoronal
    pts_c, sel_R, (occ_o, occ_n), para_n = expected_planes("R")
    oo, on = plane_world("Occlusal")
    check("Occlusal made from the PRE landmarks RIn, LIn and the Oc ones (not the mirrored set)",
          sorted(s.strip() for s in sym.get_plane("Occlusal").GetAttribute("SourceLandmarks").split(",")) == sel_R,
          sym.get_plane("Occlusal").GetAttribute("SourceLandmarks"))
    check("Occlusal = independent numpy fit, mapped to the world", abs(abs(np.dot(on, R_ @ occ_n)) - 1) < 1e-9 and abs(np.dot(oo - to_world(occ_o), on)) < 1e-9)
    check("Occlusal is perpendicular to Sagittal", abs(np.dot(on, sn)) < 1e-9, f"{np.dot(on, sn):.2e}")
    po, pn = plane_world("Paracoronal")
    check("Paracoronal normal along Sagittal x Occlusal, through Se", abs(abs(np.dot(pn, R_ @ para_n)) - 1) < 1e-9 and abs(np.dot(po - to_world(CANON["Se"]), pn)) < 1e-9)
    check("Tri plane through the PRE LZf, LD, LsOr", all(abs(np.dot(to_world(CANON[k]) - plane_world("Tri")[0], plane_world("Tri")[1])) < 1e-9 for k in ["LZf", "LD", "LsOr"]))

    # symmetrized model
    model = out["model"]
    pd = model.GetPolyData()
    pts, tris = tri_arrays(pd)
    vol = abs(signed_volume(pts, tris))
    check("symmetrized model named Cube_symmetrized_R, not under a transform", model.GetName() == "Cube_symmetrized_R" and model.GetParentTransformNode() is None)
    # VTK stores model points as float32 (about 1e-6 mm at these sizes), so model checks use float32-level tolerances
    check("keep-R cube volume = 80 x 60 x 60 = 288000 mm^3 (own divergence-theorem sum)", abs(vol - 288000) < 0.05, f"{vol:.4f}")
    check("triangles consistently oriented outward (signed volume positive)", signed_volume(pts, tris) > 0, f"{signed_volume(pts, tris):.2f}")
    check("closed surface: every edge shared by exactly 2 triangles", open_edges(tris) == 0, f"open/nonmanifold edges {open_edges(tris)}")
    canon_pts = (np.linalg.inv(T) @ np.c_[pts, np.ones(len(pts))].T).T[:, :3]
    mirrored_pts = canon_pts * np.array([-1, 1, 1])
    dmin = max(np.min(np.linalg.norm(canon_pts - m, axis=1)) for m in mirrored_pts)
    check("every model point's mirror image is also a model point", dmin < 1e-4, f"{dmin:.2e}")
    check("canonical bounds x -40..40, y 0..60, z -40..20",
          np.allclose(canon_pts.min(axis=0), [-40, 0, -40], atol=1e-4) and np.allclose(canon_pts.max(axis=0), [40, 60, 20], atol=1e-4),
          f"{canon_pts.min(axis=0)} {canon_pts.max(axis=0)}")
    check("source model unchanged", cube_model.GetPolyData().GetNumberOfPoints() == cube.GetOutput().GetNumberOfPoints() and cube_model.GetParentTransformNode() is not None)

    # measurements
    meas = out["measurements"]
    check("rows: PRE, mirPRE, POST, FU (no change rows)", list(meas) == ["PRE", "mirPRE", "POST", "FU"], str(list(meas)))
    with open(csv1, newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    header = rows[0]
    byrow = {r[0]: dict(zip(header, r)) for r in rows[1:]}
    check("CSV has the same rows", list(byrow) == ["PRE", "mirPRE", "POST", "FU"], str(list(byrow)))
    check("no mirror-target columns in the CSV", not any("mirror" in h for h in header), str([h for h in header if "mirror" in h]))
    num = lambda row, col: float(byrow[row][col])

    col = "LZf-RZf distance (mm)"
    check("added LZf-RZf distance in every set: PRE, mirPRE (= 2 x 45 mm, mirror pair), POST",
          abs(num("PRE", col) - np.linalg.norm(CANON["LZf"] - CANON["RZf"])) < 1e-3 and abs(num("mirPRE", col) - 90.0) < 1e-3
          and abs(num("POST", col) - np.linalg.norm(POST_C["LZf"] - CANON["RZf"])) < 1e-3,
          f"{byrow['PRE'][col]} {byrow['mirPRE'][col]} {byrow['POST'][col]}")
    dvec = R_ @ (CANON["RZf"] - CANON["LZf"])
    check("its dx, dy, dz are the world vector LZf -> RZf (PRE)",
          all(abs(num("PRE", f"LZf-RZf {a} (mm)") - dvec[i]) < 1e-3 for i, a in enumerate("dx dy dz".split())))
    check("R superior orbital ML = |x| of RsOr = 30 mm in PRE", abs(num("PRE", "R superior orbital ML distance (mm)") - 30) < 1e-3)
    rzf_v = dist_to_plane_c(CANON["RZf"], occ_o, occ_n)
    lzf_v = dist_to_plane_c(CANON["LZf"], occ_o, occ_n)
    check("R Zf vertical / L Zf vertical (to the PRE Occlusal) in PRE",
          abs(num("PRE", "R Zf vertical distance (mm)") - rzf_v) < 1e-3 and abs(num("PRE", "L Zf vertical distance (mm)") - lzf_v) < 1e-3, f"{rzf_v:.3f} {lzf_v:.3f}")
    check("orbital symmetry (Zf) = R - L", abs(num("PRE", "orbital symmetry (Zf)") - (rzf_v - lzf_v)) < 2e-3)
    post_lzf_v = dist_to_plane_c(POST_C["LZf"], occ_o, occ_n)
    check("POST L Zf vertical uses the same (PRE) plane", abs(num("POST", "L Zf vertical distance (mm)") - post_lzf_v) < 1e-3)
    check("FU R Zf vertical uses the same plane", abs(num("FU", "R Zf vertical distance (mm)") - dist_to_plane_c(FU_C["RZf"], occ_o, occ_n)) < 1e-3)
    # pitch: angle at RsOr between RiOr and the perpendicular to Paracoronal
    v = CANON["RsOr"]
    s_ = np.dot(v - CANON["Se"], para_n)
    to_plane = -s_ * para_n
    to_end = CANON["RiOr"] - v
    pitch = math.degrees(math.acos(np.dot(to_end, to_plane) / np.linalg.norm(to_end) / np.linalg.norm(to_plane)))
    # canthal presets, computed in the canonical frame; signed tilt positive when the lateral canthus is superior, with
    # "superior" the side of the Occlusal plane that CG is on
    n_up = occ_n if np.dot(occ_n, CANON["CG"] - occ_o) > 0 else -occ_n
    for side_, mc_, lc_ in (("R", "RMC", "RLC"), ("L", "LMC", "LLC")):
        d_ = CANON[mc_] - CANON[lc_]
        tilt_signed = math.degrees(math.asin(np.dot(n_up, CANON[lc_] - CANON[mc_]) / np.linalg.norm(d_)))
        expect_ = {
            f"{side_} canthal tilt angle (deg)": abs(tilt_signed),
            f"{side_} canthal tilt signed angle (deg)": tilt_signed,
            f"{side_} medial canthus AP distance (mm)": abs(np.dot(CANON[mc_] - CANON["Se"], para_n)),
            f"{side_} medial canthus mediolateral distance (mm)": abs(CANON[mc_][0]),
            f"{side_} medial canthus vertical distance (mm)": dist_to_plane_c(CANON[mc_], occ_o, occ_n),
            f"{side_} lateral canthus AP distance (mm)": abs(np.dot(CANON[lc_] - CANON["Se"], para_n)),
            f"{side_} lateral canthus mediolateral distance (mm)": abs(CANON[lc_][0]),
            f"{side_} lateral canthus vertical distance (mm)": dist_to_plane_c(CANON[lc_], occ_o, occ_n),
        }
        bad_ = {k: (byrow["PRE"].get(k), round(float(v), 3)) for k, v in expect_.items()
                if k not in byrow["PRE"] or abs(num("PRE", k) - v) > 2e-3}
        check(f"PRE {side_} canthal tilt (line {lc_} -> {mc_} to Occlusal, unsigned and signed) and {side_} medial/lateral canthus "
              "AP (Paracoronal), mediolateral (Sagittal) and vertical (Occlusal) = independent values", not bad_, str(bad_))
    check("signed canthal tilt is positive on both sides here, where each lateral canthus is above its medial one",
          num("PRE", "R canthal tilt signed angle (deg)") > 0 and num("PRE", "L canthal tilt signed angle (deg)") > 0)
    check("the Occlusal plane records its superior side (the side CG is on)",
          sym.get_plane("Occlusal").GetAttribute("SuperiorSide") == ("1" if np.dot(on, R_ @ n_up) > 0 else "-1"),
          str(sym.get_plane("Occlusal").GetAttribute("SuperiorSide")))
    occ_o_w, occ_n_w = plane_world("Occlusal")
    flipped = sym.planes._build_plane_node("OccFlipped", occ_o_w, -occ_n_w)
    sym.planes.mark_superior_side(flipped, to_world(CANON["CG"]))
    pre_case_ = [c for c in sym.find_cases() if c["case"] == "PRE"][0]
    spec_ = dict(sym.settings.preset_measurement_specs()[[sp["name"] for sp in sym.settings.preset_measurement_specs()].index("R canthal tilt")])
    r_orig = sym.evaluate(spec_, pre_case_)
    spec_["plane"] = "OccFlipped"
    r_flip = sym.evaluate(spec_, pre_case_)
    plain_ = {"kind": "line_plane_angle", "landmarks": ["RLC", "RMC"], "plane": "OccFlipped", "name": None}
    check("with the Occlusal normal flipped the signed canthal tilt is unchanged (a plain line-to-plane angle changes sign)",
          abs(r_flip["signed_deg"] - r_orig["signed_deg"]) < 1e-9 and abs(sym.evaluate(plain_, pre_case_)["signed_deg"] + sym.evaluate(dict(plain_, plane="Occlusal"), pre_case_)["signed_deg"]) < 1e-9,
          f"{r_orig['signed_deg']:.4f} {r_flip['signed_deg']:.4f}")
    slicer.mrmlScene.RemoveNode(flipped)
    check("mirPRE: the left canthal values equal the right ones (symmetric set)",
          all(abs(num("mirPRE", f"L {k}") - num("mirPRE", f"R {k}")) < 2e-3 for k in
              ("canthal tilt angle (deg)", "medial canthus AP distance (mm)", "medial canthus mediolateral distance (mm)",
               "lateral canthus vertical distance (mm)")))
    check("R pitch (x rotation) = independent angle", abs(num("PRE", "R pitch (x rotation) angle (deg)") - pitch) < 1e-3, f"{num('PRE', 'R pitch (x rotation) angle (deg)')} vs {pitch:.3f}")
    check("POST lacks LlOr: L lateral orbital AP blank in POST, filled in PRE and mirPRE",
          byrow["POST"]["L lateral orbital AP distance (mm)"] == "" and byrow["PRE"]["L lateral orbital AP distance (mm)"] != ""
          and byrow["mirPRE"]["L lateral orbital AP distance (mm)"] != "")
    # the mirPRE row: right values = PRE right values (copies), left values = the same numbers (mirrors)
    pairs_ = [("R Zf vertical distance (mm)", "L Zf vertical distance (mm)"), ("R superior orbital ML distance (mm)", "L superior orbital ML distance (mm)"),
              ("R pitch (x rotation) angle (deg)", "L pitch (x rotation) angle (deg)"), ("R yaw (y rotation) angle (deg)", "L yaw (y rotation) angle (deg)"),
              ("R inferior orbital AP distance (mm)", "L inferior orbital AP distance (mm)")]
    check("mirPRE row: every R value equals PRE's R value", all(byrow["mirPRE"][r_] == byrow["PRE"][r_] for r_, _ in pairs_),
          str([(byrow["mirPRE"][r_], byrow["PRE"][r_]) for r_, _ in pairs_]))
    check("mirPRE row: L values equal its R values, orbital symmetry (Zf) = 0",
          all(abs(num("mirPRE", r_) - num("mirPRE", l_)) < 2e-3 for r_, l_ in pairs_) and abs(num("mirPRE", "orbital symmetry (Zf)")) < 2e-3,
          str([(byrow["mirPRE"][r_], byrow["mirPRE"][l_]) for r_, l_ in pairs_]))
    check("nothing about change rows was printed", not any("(change)" in m for m in logged))
    mlog = [m for m in logged if m.startswith("mirPRE_Landmarks:")]
    check("the mirroring step was logged: 11 right landmarks copied and mirrored, the rest listed as not used",
          len(mlog) == 1 and "11 right landmark(s) copied and mirrored" in mlog[0] and "not used: CG, Se, Ba, ANS, LZf" in mlog[0] and "MidOc" in mlog[0].split("not used:")[1],
          str(mlog))

    # ---------------- second run: keep Left, no dialogs except the two questions ----------------
    n_planes_before = len(sym.existing_planes())
    steps[:] = [("Planes", msg("No")), ("Measurements", msg("No"))]
    csv2 = os.path.join(SP, "symmetrized_left.csv")
    out2 = sym.run(csv_path=csv2, side="L", source="Cube", landmarks=False)
    check("second run (side='L', source='Cube') with no harness errors", not errors and not steps, str(errors))
    check("planes replaced, not duplicated", len(sym.existing_planes()) == n_planes_before and sym.get_plane("Sagittal_2") is None, str(sym.plane_names()))
    check("re-run reuses InitialAlign and its wizard node (no InitialAlign_2)",
          slicer.mrmlScene.GetFirstNodeByName("InitialAlign_2") is None and out2.get("initial_align") is not None and out2["initial_align"].GetID() == ia.GetID()
          and len([n for n in slicer.util.getNodesByClass("vtkMRMLFiducialRegistrationWizardNode")]) == 1)
    check("mirrored set replaced (one mirPRE_Landmarks)", len([n for n in slicer.util.getNodesByClass("vtkMRMLMarkupsFiducialNode") if n.GetName().startswith("mirPRE")]) == 1)
    m2 = out2["mirrored"]
    l2 = [m2.GetNthControlPointLabel(i) for i in range(m2.GetNumberOfControlPoints())]
    check("keep L: symmetrized set holds the PRE left landmarks and their mirrors labeled R", l2 == [x for k in CANON if k.startswith("L") for x in (k, "R" + k[1:])], str(l2))
    _, sel_L, (occL_o, occL_n), _ = expected_planes("L")
    check("keep L: Occlusal still from the PRE landmarks",
          sorted(s.strip() for s in sym.get_plane("Occlusal").GetAttribute("SourceLandmarks").split(",")) == sel_L,
          sym.get_plane("Occlusal").GetAttribute("SourceLandmarks"))
    check("keep L: Occlusal = independent fit", abs(abs(np.dot(plane_world("Occlusal")[1], R_ @ occL_n)) - 1) < 1e-9)
    pts2, tris2 = tri_arrays(out2["model"].GetPolyData())
    check("keep-L cube: Cube_symmetrized_L, volume 40 x 60 x 60 = 144000, closed",
          out2["model"].GetName() == "Cube_symmetrized_L" and abs(abs(signed_volume(pts2, tris2)) - 144000) < 0.05 and open_edges(tris2) == 0,
          f"{signed_volume(pts2, tris2):.3f}")
    with open(csv2, newline="", encoding="utf-8") as f:
        r2 = list(csv.DictReader(f))
    by2 = {r["case_id"]: r for r in r2}
    check("keep L: rows PRE, mirPRE, POST, FU; mirPRE L values = PRE L values and R = L",
          list(by2) == ["PRE", "mirPRE", "POST", "FU"]
          and by2["mirPRE"]["L Zf vertical distance (mm)"] == by2["PRE"]["L Zf vertical distance (mm)"]
          and by2["mirPRE"]["R Zf vertical distance (mm)"] == by2["mirPRE"]["L Zf vertical distance (mm)"], str(list(by2)))

    # ---------------- segmentation source ----------------
    origin_w, normal_w = plane_world("Sagittal")
    right_w = R_ @ np.array([1.0, 0, 0])
    segmodel = sym.create_symmetrized_model("BoxSeg", sym.get_plane("Sagittal"), right_w, "R")
    p3, t3 = tri_arrays(segmodel.GetPolyData())
    # labelmap box: x from -10.5 to 29.5 (voxel edges), keep right (x>0): 29.5 wide -> mirrored 59 x 30 x 30
    vol3 = abs(signed_volume(p3, t3))
    check("segmentation source (labelmap -> closed surface): keep-R volume close to 59 x 30 x 30 = 53100 (surface smoothing allowed 6%)",
          abs(vol3 - 53100) / 53100 < 0.06, f"{vol3:.0f}")
    c3 = (np.linalg.inv(T) @ np.c_[p3, np.ones(len(p3))].T).T[:, :3]
    check("segmentation result symmetric about canonical x=0 (bounds)", abs(c3[:, 0].min() + c3[:, 0].max()) < 1e-4, f"{c3[:, 0].min():.3f} {c3[:, 0].max():.3f}")
    check("segmentation result closed", open_edges(t3) == 0, str(open_edges(t3)))

    # ---------------- side fallback, cancel, errors ----------------
    nopairs = add_set("Mid_Landmarks", {k: to_world(CANON[k]) for k in SAG + ["N"]})
    cnp = [c for c in sym.find_cases() if c["case"] == "Mid"][0]
    rn, how = sym.right_side_normal(sym.get_plane("Sagittal"), cnp)
    check("no R/L pairs: right = the plane normal with positive world x", "world +x" in how and rn[0] > 0 and abs(abs(np.dot(rn, sn)) - 1) < 1e-12, how)
    rn2, how2 = sym.right_side_normal(sym.get_plane("Sagittal"), sym.find_cases()[0])
    check("with pairs: right normal points to canonical +x", np.dot(rn2, right_w) > 0.999 and "pair" in how2, how2)
    slicer.mrmlScene.RemoveNode(nopairs)
    steps[:] = [("Side to keep", side_button("Cancel"))]
    check("cancelling the side question ends the run", sym.run(csv_path=csv2, landmarks=False) == {} and not errors and not steps, str(errors))

    # start() reloads every module, so an edit to settings.py counts without restarting Slicer
    settings_file = os.path.join(_lone, "symmetrized", "settings.py")
    original_settings = open(settings_file, encoding="utf-8").read()
    assert original_settings.count('ALIGN_LANDMARKS = ["CG", "Se", "Ba", "ANS", "RM", "LM"]') == 1
    open(settings_file, "w", encoding="utf-8").write(original_settings.replace(
        'ALIGN_LANDMARKS = ["CG", "Se", "Ba", "ANS", "RM", "LM"]', 'ALIGN_LANDMARKS = ["Se", "CG"]'))
    steps[:] = [("Side to keep", side_button("Cancel"))]
    started_out = sym.start(csv_path=csv2, landmarks=False)
    al = slicer.mrmlScene.GetFirstNodeByName("PRE_align")
    check("start(): reloads the edited settings.py (PRE_align now Se, CG) and runs",
          started_out == {} and not errors and not steps and sym.settings.ALIGN_LANDMARKS == ["Se", "CG"]
          and [al.GetNthControlPointLabel(i) for i in range(al.GetNumberOfControlPoints())] == ["Se", "CG"], str(errors))
    open(settings_file, "w", encoding="utf-8").write(original_settings)
    steps[:] = [("Side to keep", side_button("Cancel"))]
    sym.start(csv_path=csv2, landmarks=False)
    check("start() again after restoring settings.py: back to the six alignment landmarks",
          sym.settings.ALIGN_LANDMARKS == ["CG", "Se", "Ba", "ANS", "RM", "LM"] and sym.placement.ALIGN_LANDMARKS == ["CG", "Se", "Ba", "ANS", "RM", "LM"]
          and not errors and not steps, str(errors))
    check("bad side and bad source names are errors", raises(lambda: sym.run(csv_path=csv2, side="X", landmarks=False)) and raises(lambda: sym.run(csv_path=csv2, side="R", source="Nope", landmarks=False)))
    check("unknown pre case is an error", raises(lambda: sym.run(csv_path=csv2, pre="Nope", landmarks=False)))

    # fiducial registration: a known rigid transform is recovered; bad inputs are reported
    known = np.eye(4)
    known[:3, :3] = rot([-0.4, 1.0, 0.3], 17.0)
    known[:3, 3] = [12.0, -7.5, 4.25]
    to_pts = {k: to_world(CANON[k]) for k in ["CG", "Se", "Ba", "ANS", "RM", "LM"]}
    from_pts = {k: (np.linalg.inv(known) @ np.append(v, 1.0))[:3] for k, v in to_pts.items()}   # From = known^-1 (To)
    add_set("KnownFrom", from_pts)
    add_set("KnownTo", to_pts)
    kt, kprob = sym.fiducial_registration("KnownFrom", "KnownTo", "KnownAlign")
    check("a known rigid transform is recovered exactly (From onto To), RMS ~0",
          kt is not None and not kprob and np.allclose(matrix_of(kt), known, atol=1e-6) and float(kt.GetAttribute("FiducialRMS_mm")) < 1e-4,
          f"{np.abs(matrix_of(kt) - known).max():.2e} rms {kt.GetAttribute('FiducialRMS_mm') if kt else None}")
    shift = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLinearTransformNode", "Shift")
    sm = np.eye(4)
    sm[:3, 3] = [30.0, 0, 0]
    shift.SetMatrixTransformToParent(vtk_matrix(sm))
    slicer.mrmlScene.GetFirstNodeByName("KnownTo").SetAndObserveTransformNodeID(shift.GetID())
    kt2, _ = sym.fiducial_registration("KnownFrom", "KnownTo", "KnownAlign")
    check("(why the _align lists hold world positions) the wizard ignores a parent transform of the To list",
          np.allclose(matrix_of(kt2), known, atol=1e-6), f"{np.abs(matrix_of(kt2) - known).max():.2e}")
    add_set("ShortFrom", {k: from_pts[k] for k in ["CG", "Se"]})
    none_t, short_prob = sym.fiducial_registration("ShortFrom", "KnownTo", "ShortAlign")
    check("different point counts: no transform, reported", none_t is None and short_prob and "same number" in short_prob[0], str(short_prob))
    none_t, missing_prob = sym.fiducial_registration("NoSuchList", "KnownTo", "NoAlign")
    check("a missing list: no transform, reported", none_t is None and "no point list named NoSuchList" in missing_prob[0], str(missing_prob))
    slicer.mrmlScene.RemoveNode(slicer.mrmlScene.GetFirstNodeByName("InitialAlign"))
    steps[:] = [("Side to keep", side_button("Cancel"))]
    sym.run(csv_path=csv2, landmarks=False, initial_align=False)
    check("initial_align=False: no InitialAlign is made", slicer.mrmlScene.GetFirstNodeByName("InitialAlign") is None and not errors and not steps, str(errors))

    # ---------------- ASYM: constrained sagittal fit against an independent minimisation ----------------
    from scipy.optimize import minimize_scalar
    asym_c = dict(CANON)
    asym_c.update({"CG": np.array([1.5, 60, 40]), "Se": np.array([-0.8, 20, 20]), "Ba": np.array([0.6, -10, -10]), "ANS": np.array([-1.2, 70, -30]),
                   "LIn": np.array([-4.0, 74, -47]), "LOc6": np.array([-22.0, 41, -43]), "LOc3": np.array([-14.0, 61, -44]), "MidOc": np.array([0.5, 70, -44])})
    asym = add_set("ASYM_Landmarks", {k: to_world(v) for k, v in asym_c.items()})
    acase = [c for c in sym.find_cases() if c["case"] == "ASYM"][0]
    planes_a, sag_a, sag_lm, probs_a = sym.make_reference_planes(acase)
    check("ASYM: reference planes made with no problems", sorted(planes_a) == ["Occlusal", "Paracoronal", "Sagittal"] and not probs_a and sag_lm == SAG, str(probs_a))
    occ_oa, occ_na = svd_plane([asym_c[k] for k in OCC])
    oa_o, oa_n = plane_world("Occlusal")
    check("ASYM: Occlusal = numpy fit of the PRE occlusal landmarks", abs(abs(np.dot(oa_n, R_ @ occ_na)) - 1) < 1e-9)
    sa_o, sa_n = plane_world("Sagittal")
    check("ASYM: Sagittal perpendicular to Occlusal", abs(np.dot(sa_n, oa_n)) < 1e-9, f"{np.dot(sa_n, oa_n):.2e}")
    sp = np.array([asym_c[k] for k in SAG])
    cen = sp.mean(axis=0)
    hint = np.array([1.0, 0, 0]) if abs(occ_na[0]) < 0.9 else np.array([0, 1.0, 0])
    u_ = unit(hint - np.dot(hint, occ_na) * occ_na)
    v_ = np.cross(occ_na, u_)
    sse = lambda th: float(np.sum(((sp - cen) @ (math.cos(th) * u_ + math.sin(th) * v_)) ** 2))
    grid = np.linspace(0, math.pi, 7201)
    th0 = grid[int(np.argmin([sse(t) for t in grid]))]
    th = minimize_scalar(sse, bounds=(th0 - 1e-3, th0 + 1e-3), method="bounded", options={"xatol": 1e-12}).x
    n_best = math.cos(th) * u_ + math.sin(th) * v_
    check("ASYM: Sagittal normal = the independent minimisation over planes perpendicular to Occlusal",
          abs(abs(np.dot(sa_n, R_ @ n_best)) - 1) < 1e-9, f"1-|cos| = {1 - abs(np.dot(sa_n, R_ @ n_best)):.2e}")
    check("ASYM: Sagittal passes through the centroid of CG, Se, Ba, ANS", abs(np.dot(to_world(cen) - sa_o, sa_n)) < 1e-9)
    rms_best = math.sqrt(sse(th) / 4)
    check("ASYM: recorded RMS = the minimised RMS", abs(float(sag_a.GetAttribute("FitRMS_mm")) - rms_best) < 1e-4, f"{sag_a.GetAttribute('FitRMS_mm')} vs {rms_best:.4f}")
    free_o, free_n = svd_plane(sp)
    check("ASYM: the constraint matters here (the unconstrained fit differs by more than 0.5 deg)",
          math.degrees(math.acos(min(1.0, abs(np.dot(free_n, n_best))))) > 0.5, f"{math.degrees(math.acos(min(1.0, abs(np.dot(free_n, n_best))))):.3f} deg")
    pa_o, pa_n = plane_world("Paracoronal")
    check("ASYM: Paracoronal perpendicular to Sagittal and Occlusal, through Se",
          abs(np.dot(pa_n, sa_n)) < 1e-9 and abs(np.dot(pa_n, oa_n)) < 1e-9 and abs(np.dot(to_world(asym_c["Se"]) - pa_o, pa_n)) < 1e-9)
    noocc = add_set("NOOCC_Landmarks", {k: to_world(v) for k, v in asym_c.items() if k not in ("RIn",)})
    ncase = [c for c in sym.find_cases() if c["case"] == "NOOCC"][0]
    planes_n, sag_n_, _, probs_n = sym.make_reference_planes(ncase)
    fo, fn = plane_world("Sagittal")
    check("no RIn: Occlusal and Paracoronal not made, Sagittal is the unconstrained fit and this is reported",
          sorted(planes_n) == ["Sagittal"] and sym.get_plane("Occlusal") is None and sym.get_plane("Paracoronal") is None and abs(abs(np.dot(fn, R_ @ free_n)) - 1) < 1e-9
          and any("unconstrained" in m for m in probs_n) and any("Paracoronal not made" in m for m in probs_n), str(probs_n))
    nocg = add_set("NOCG_Landmarks", {k: to_world(v) for k, v in asym_c.items() if k != "CG"})
    check("no CG: the sagittal plane cannot be made -> error", raises(lambda: sym.make_reference_planes([c for c in sym.find_cases() if c["case"] == "NOCG"][0])))

    # ---------------- guided landmark definition, from an empty scene ----------------
    import re as _re
    slicer.mrmlScene.Clear(0)
    POS = {}
    rng = np.random.default_rng(7)
    sim = {"acted": None, "undo": "pending", "cancel": False, "modes": [], "prompts": [], "others": [], "head": []}

    def panel_widget():
        for w in slicer.app.topLevelWidgets():
            if w.objectName == "landmarkPlacementPanel" and w.visible:
                return w
        return None

    def sim_tick():
        """Plays the user: one 'click' (a point added to the list armed for placing) per new prompt."""
        try:
            w = panel_widget()
            if w is None:
                return
            prompt = widget(w, "promptLabel").text
            m_ = _re.search(r"\(([A-Za-z0-9]+)\)$", prompt)
            if not m_:
                return
            abbr = m_.group(1)
            sel = slicer.mrmlScene.GetNodeByID("vtkMRMLSelectionNodeSingleton")
            inter = slicer.mrmlScene.GetNodeByID("vtkMRMLInteractionNodeSingleton")
            node = slicer.mrmlScene.GetNodeByID(sel.GetActivePlaceNodeID())
            token = (w.windowTitle, abbr, node.GetNumberOfControlPoints(), sim["undo"])
            if token == sim["acted"]:
                return
            sim["acted"] = token
            sim["prompts"].append((w.windowTitle, prompt, widget(w, "progressLabel").text))
            sim["modes"].append((node.GetName(), inter.GetCurrentInteractionMode() == inter.Place))
            seg_ = slicer.mrmlScene.GetFirstNodeByName(node.GetName().split("_")[0] + "_HeadSkull")
            if seg_ is not None:
                sim["head"].append((node.GetName(), abbr, bool(seg_.GetDisplayNode().GetSegmentVisibility3D("head")),
                                    bool(seg_.GetDisplayNode().GetSegmentVisibility3D("skull"))))
            sim["others"].append((node.GetName(), {n.GetName(): bool(n.GetDisplayNode().GetVisibility())
                                                   for n in slicer.util.getNodesByClass("vtkMRMLMarkupsFiducialNode")
                                                   if n.GetID() != node.GetID() and n.GetDisplayNode() is not None}))
            if sim["cancel"]:
                widget(w, "cancelButton").click()
                return
            name_ = node.GetName()
            if name_ == "PRE_Landmarks" and abbr == "Ba" and sim["undo"] == "pending":
                node.AddControlPoint([999.0, 999.0, 999.0], "oops")      # a wrong click ...
                sim["undo"] = "placed"
                return
            if name_ == "PRE_Landmarks" and abbr == "ANS" and sim["undo"] == "placed":
                widget(w, "undoButton").click()                           # ... undone
                sim["undo"] = "done"
                return
            pos = rng.uniform(-50, 50, 3)
            POS[(name_, abbr)] = pos
            node.AddControlPoint([float(v) for v in pos], "wrong")      # labeled wrongly on purpose
            if name_ == "POST_Landmarks" and abbr == "RLC":
                node.AddControlPoint([5.0, 5.0, 5.0], "extra")            # one click too many at the end
        except Exception:
            errors.append(traceback.format_exc())

    sim_timer = qt.QTimer()
    sim_timer.setInterval(40)
    sim_timer.timeout.connect(sim_tick)
    sim_timer.start()
    errors.clear()
    steps[:] = [
        ("Landmarks", side_button("Define new landmarks")),
        ("Occlusal points", side_button("Right")),
        ("Occlusal points", side_button("Left")),
        ("Occlusal points", side_button("Done")),
        ("Postoperative landmarks", msg("Ok")),
        ("Side to keep", side_button("Cancel")),
    ]
    started = time.time()
    for case_ in ("PRE", "POST"):   # segmentations as segment_volumes leaves them (head and skull are enough here)
        sgn = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode", f"{case_}_HeadSkull")
        sgn.CreateDefaultDisplayNodes()
        sgn.SetAttribute("SymmetrizedHeadSkullOf", case_)
        sgn.SetAttribute("SymmetrizedKeptSegments", ",".join(sym.settings.TOTALSEG_KEEP_SEGMENTS))
        for sid in ("head", "skull"):
            sgn.GetSegmentation().AddEmptySegment(sid, sid)
    old_points = add_set("Old_points", {"A": np.array([1.0, 2.0, 3.0])})          # an unrelated visible point list
    old_points.CreateDefaultDisplayNodes()
    hidden_points = add_set("Hidden_points", {"B": np.array([4.0, 5.0, 6.0])})     # one the user had hidden
    hidden_points.CreateDefaultDisplayNodes()
    hidden_points.GetDisplayNode().SetVisibility(False)
    out3 = sym.run(csv_path=csv2)
    check("guided definition from an empty scene ran to the side question with no harness errors",
          out3 == {} and not errors and not steps, "; ".join(e.splitlines()[-1] for e in errors) + f" | steps left {len(steps)}")
    BOTH = [a for a, _ in sym.settings.LANDMARKS]
    check("both lists end with the canthi LMC, LLC, RMC, RLC", BOTH[-4:] == ["LMC", "LLC", "RMC", "RLC"] and len(BOTH) == 20, str(BOTH))
    pre_expected = BOTH[:4] + ["LIn", "RIn", "LOc1", "ROc1", "ROc2", "LOc2"] + BOTH[4:]
    pre3 = slicer.mrmlScene.GetFirstNodeByName("PRE_Landmarks")
    post3 = slicer.mrmlScene.GetFirstNodeByName("POST_Landmarks")
    lab3 = [pre3.GetNthControlPointLabel(i) for i in range(pre3.GetNumberOfControlPoints())] if pre3 else None
    check("PRE_Landmarks: CG, Se, Ba, ANS, then LIn, RIn, LOc1, ROc1 and the extra ROc2, LOc2, then LM, RM, the dacryons ... and "
          "the canthi, each labeled by its abbreviation",
          lab3 == pre_expected, str(lab3))
    lab4 = [post3.GetNthControlPointLabel(i) for i in range(post3.GetNumberOfControlPoints())] if post3 else None
    check("POST_Landmarks: the 20 common landmarks only, ending with the canthi (the extra click at the end removed)", lab4 == BOTH, str(lab4))
    names_full = dict(sym.settings.LANDMARKS + sym.settings.PRE_ONLY_LANDMARKS)
    names_full["ROc2"] = "Right Occluding surface of the molars, point 2"
    names_full["LOc2"] = "Left Occluding surface of the molars, point 2"
    check("descriptions hold the full names", pre3 is not None and all(pre3.GetNthControlPointDescription(i) == names_full[a]
          for i, a in enumerate(pre_expected)), str([pre3.GetNthControlPointDescription(i) for i in (8, 20, 23)]) if pre3 else "")
    worst3 = 0.0
    for node_, labs in ((pre3, pre_expected), (post3, BOTH)):
        for i, a in enumerate(labs):
            q_ = [0.0] * 3
            node_.GetNthControlPointPositionWorld(i, q_)
            worst3 = max(worst3, float(np.linalg.norm(np.array(q_) - POS[(node_.GetName(), a)])))
    check("every point is where it was clicked (the undone wrong click is gone)", worst3 < 1e-9 and sim["undo"] == "done", f"{worst3:.2e} {sim['undo']}")
    check("the first prompt and progress (24 points for PRE)", sim["prompts"] and sim["prompts"][0] == ("Place landmarks: PRE_Landmarks", "Please set the landmark for the: Crista Galli (CG)", "Point 1 of 24"),
          str(sim["prompts"][:1]))
    check("after ANS comes LIn ('Point 5 of 24'); the extra occlusal point is asked for by name and counted ('Point 9 of 25')",
          ("Place landmarks: PRE_Landmarks", "Please set the landmark for the: Left Central Incisal Edge (LIn)", "Point 5 of 24") in sim["prompts"]
          and ("Place landmarks: PRE_Landmarks", "Please set the landmark for the: Right Occluding surface of the molars, point 2 (ROc2)", "Point 9 of 25") in sim["prompts"])
    check("the last PRE point is the right lateral canthus, 'Point 26 of 26'",
          ("Place landmarks: PRE_Landmarks", "Please set the landmark for the: Right Lateral Canthus (RLC)", "Point 26 of 26") in sim["prompts"])
    check("POST prompts start again at CG, 'Point 1 of 20'", ("Place landmarks: POST_Landmarks", "Please set the landmark for the: Crista Galli (CG)", "Point 1 of 20") in sim["prompts"])
    soft_ = set(sym.settings.SOFT_TISSUE_LANDMARKS)
    wrong_ = [(n_, a_, h_) for n_, a_, h_, k_ in sim["head"] if h_ != (a_ in soft_) or not k_]
    check("the head (skin) segment is hidden in 3D for the bony landmarks and shown for the canthi; the skull always shown",
          len(sim["head"]) >= 40 and not wrong_ and any(a_ in soft_ for _, a_, _, _ in sim["head"]), str(wrong_[:4]))
    check("afterwards the head segment is shown again in both segmentations",
          all(slicer.mrmlScene.GetFirstNodeByName(f"{c_}_HeadSkull").GetDisplayNode().GetSegmentVisibility3D("head") for c_ in ("PRE", "POST")))
    check("place mode was armed on the right list at every prompt", sim["modes"] and all(m_[1] for m_ in sim["modes"]), str([m_ for m_ in sim["modes"] if not m_[1]][:3]))
    inter = slicer.mrmlScene.GetNodeByID("vtkMRMLInteractionNodeSingleton")
    check("place mode is off afterwards", inter.GetCurrentInteractionMode() == inter.ViewTransform)
    pre_seen = [o for name_, o in sim["others"] if name_ == "PRE_Landmarks"]
    post_seen = [o for name_, o in sim["others"] if name_ == "POST_Landmarks"]
    check("while PRE_Landmarks is placed, the other point lists are hidden; while POST_Landmarks is placed, PRE_Landmarks is hidden too",
          pre_seen and post_seen and all(not any(o.values()) for o in pre_seen + post_seen)
          and all("PRE_Landmarks" in o and "Old_points" in o for o in post_seen),
          f"{pre_seen[:1]} {post_seen[:1]}")
    vis = lambda name_: bool(slicer.mrmlScene.GetFirstNodeByName(name_).GetDisplayNode().GetVisibility())
    check("afterwards they are shown again (PRE_Landmarks, POST_Landmarks, Old_points); one the user had hidden stays hidden",
          vis("PRE_Landmarks") and vis("POST_Landmarks") and vis("Old_points") and not vis("Hidden_points"))
    slicer.mrmlScene.RemoveNode(old_points)
    slicer.mrmlScene.RemoveNode(hidden_points)
    ok_al = True
    for set_, case_ in (("PRE_Landmarks", "PRE"), ("POST_Landmarks", "POST")):
        al = slicer.mrmlScene.GetFirstNodeByName(f"{case_}_align")
        labs = [al.GetNthControlPointLabel(i) for i in range(al.GetNumberOfControlPoints())] if al else None
        ok_al &= labs == ["CG", "Se", "Ba", "ANS", "RM", "LM"]
        if al:
            for i, a in enumerate(labs):
                q_ = [0.0] * 3
                al.GetNthControlPointPositionWorld(i, q_)
                ok_al &= np.linalg.norm(np.array(q_) - POS[(set_, a)]) < 1e-9
    check("PRE_align and POST_align: CG, Se, Ba, ANS, RM, LM in that order, copied from the new sets", ok_al)
    ia3 = slicer.mrmlScene.GetFirstNodeByName("InitialAlign")
    exp3 = kabsch([POS[("POST_Landmarks", a)] for a in ["CG", "Se", "Ba", "ANS", "RM", "LM"]],
                  [POS[("PRE_Landmarks", a)] for a in ["CG", "Se", "Ba", "ANS", "RM", "LM"]])
    check("after defining new landmarks, InitialAlign = independent rigid fit of the new POST_align onto PRE_align",
          ia3 is not None and np.allclose(matrix_of(ia3), exp3, atol=1e-6), f"{np.abs(matrix_of(ia3) - exp3).max():.2e}" if ia3 else "none")

    # keep the existing sets (No to "Replace?"), then cancel a placement
    steps[:] = [("Landmarks", msg("No")), ("Postoperative landmarks", msg("Ok")), ("Landmarks", msg("No"))]
    kept = sym.define_landmarks()
    check("No to 'Replace it?' keeps both sets unchanged; the _align lists are remade (one each)",
          kept is True and not errors and not steps and [pre3.GetNthControlPointLabel(i) for i in range(pre3.GetNumberOfControlPoints())] == pre_expected
          and len([n for n in slicer.util.getNodesByClass("vtkMRMLMarkupsFiducialNode") if n.GetName().endswith("_align")]) == 2, str(errors))
    sim["cancel"] = True
    sim["acted"] = None
    steps[:] = [("Landmarks", msg("Yes"))]
    cancelled = sym.define_landmarks()
    check("Cancel in the placement panel stops the definition (returns False), place mode off, panel closed",
          cancelled is False and not errors and not steps and inter.GetCurrentInteractionMode() == inter.ViewTransform and panel_widget() is None, str(errors))
    sim_timer.stop()

    # ---------------- General Registration (BRAINS): FinalAlign on a phantom with a known misalignment ----------------
    # Segmentation is on by default and this scene has PRE and POST volumes: the real TotalSegmentator would run on the
    # phantom, so its logic class is replaced (at the class, which start()'s reloads do not touch) by a quick stand-in.
    import TotalSegmentator

    class QuickTotalSeg:
        higherOrderResampling = False

        def setupPythonRequirements(self, upgrade=False):
            pass

        def process(self, inputVolume, outputSegmentation, quality=None, cpu=False, task=None, subset=None,
                    interactive=False, sequenceBrowserNode=None):
            for sid in ("mandible", "teeth_lower", "skull", "head", "sinus_maxillary", "sinus_frontal", "teeth_upper"):
                outputSegmentation.GetSegmentation().AddEmptySegment(sid, sid)

    _RealTotalSegLogic = TotalSegmentator.TotalSegmentatorLogic
    TotalSegmentator.TotalSegmentatorLogic = QuickTotalSeg
    from scipy import ndimage
    slicer.mrmlScene.Clear(0)
    rngp = np.random.default_rng(3)
    arr = np.zeros((80, 90, 100), dtype=np.float32)                 # k, j, i
    for _ in range(14):                                              # asymmetric blobs of different brightness
        c = rngp.uniform([15, 15, 15], [65, 75, 85])
        r = rngp.uniform(4, 10, 3)
        kk, jj, ii = np.ogrid[:80, :90, :100]
        arr[((kk - c[0]) / r[0]) ** 2 + ((jj - c[1]) / r[1]) ** 2 + ((ii - c[2]) / r[2]) ** 2 < 1] += rngp.uniform(200, 1000)
    arr[10:20, 10:80, 10:30] += 600                                   # a slab to break any symmetry
    arr = ndimage.gaussian_filter(arr, 1.5).astype(np.int16)
    base = np.eye(4)
    base[:3, 3] = [-50.0, -45.0, -40.0]                               # IJK -> RAS of PRE (1 mm voxels)
    tk = np.eye(4)
    tk[:3, :3] = rot([0.2, 1.0, 0.3], 6.0)
    tk[:3, 3] = [4.0, -3.0, 5.0]
    truth = np.linalg.inv(tk)                                         # POST voxel at tk*base*ijk, PRE at base*ijk

    def volume(name, ijk_to_ras):
        v = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode", name)
        slicer.util.updateVolumeFromArray(v, arr)
        v.SetIJKToRASMatrix(vtk_matrix(ijk_to_ras))
        return v

    pre_v, post_v = volume("PRE", base), volume("POST", tk @ base)
    centre_post = (tk @ base @ np.append((np.array([100, 90, 80]) - 1) / 2.0, 1.0))

    def err(m):
        """(rotation deg, mm at the POST centre) between m and the true POST->PRE transform."""
        d = m @ np.linalg.inv(truth)
        rot_deg = math.degrees(math.acos(max(-1.0, min(1.0, (np.trace(d[:3, :3]) - 1) / 2))))
        return rot_deg, float(np.linalg.norm((m @ centre_post)[:3] - (truth @ centre_post)[:3]))

    perturb = np.eye(4)
    perturb[:3, :3] = rot([1.0, -0.5, 0.2], 2.0)
    perturb[:3, 3] = [1.5, -1.0, 1.0]
    start_m = perturb @ truth
    ia_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLinearTransformNode", "InitialAlign")
    ia_node.SetMatrixTransformToParent(vtk_matrix(start_m))
    captured = []
    _run_sync = slicer.cli.runSync

    def spy(module, node=None, parameters=None, *a, **k):
        captured.append((module.name, dict(parameters or {})))
        return _run_sync(module, node, parameters, *a, **k)

    slicer.cli.runSync = spy
    try:
        fa, fprob, fsum = sym.intensity_registration(sampling_percentage=0.05)
        e0, e1 = err(start_m), err(matrix_of(fa)) if fa else (None, None)
        check("BRAINS (5% sampling, to verify the setup): FinalAlign recovers the known transform from a 2 deg / ~2 mm start",
              fa is not None and not fprob and e1[0] < 0.3 and e1[1] < 0.3,
              f"start {e0[0]:.2f} deg {e0[1]:.2f} mm -> {e1[0]:.3f} deg {e1[1]:.3f} mm; {fprob}")
        check("its refinement summary matches the change from InitialAlign",
              abs(fsum["rotation_deg"] - math.degrees(math.acos(max(-1.0, min(1.0, (np.trace((matrix_of(fa) @ np.linalg.inv(start_m))[:3, :3]) - 1) / 2))))) < 1e-6)
        first_m = matrix_of(fa)
        captured.clear()
        post_v.SetAndObserveTransformNodeID(ia_node.GetID())          # as if the user applied InitialAlign to POST
        fa2, fprob2, _ = sym.intensity_registration()                  # the defaults: sampling 0.002
        prm = captured[0][1] if captured else {}
        check("defaults passed to BRAINSFit: fixed PRE, moving POST, sampling 0.002, initialization InitialAlign, Rigid only, "
              "linear output FinalAlign, initialize mode Off",
              captured and captured[0][0] == "BRAINSFit" and prm.get("fixedVolume") == pre_v.GetID() and prm.get("movingVolume") == post_v.GetID()
              and prm.get("samplingPercentage") == 0.002 and prm.get("initialTransform") == ia_node.GetID() and prm.get("useRigid") is True
              and prm.get("linearTransform") == fa2.GetID() and fa2.GetName() == "FinalAlign" and prm.get("initializeTransformMode") == "Off"
              and not any(prm.get(k) for k in ("useAffine", "useBSpline", "useScaleVersor3D", "useScaleSkewVersor3D", "bsplineTransform", "outputVolume")),
              str(sorted(prm)))
        e2 = err(matrix_of(fa2))
        lines.append(f"INFO BRAINS at the default 0.002 sampling on this 100x90x80 phantom: start {e0[0]:.2f} deg {e0[1]:.2f} mm -> "
                     f"{e2[0]:.3f} deg {e2[1]:.3f} mm (few samples on a small volume; real CBCT has far more voxels)")
        check("FinalAlign reused on a re-run (no FinalAlign_2); POST put back under InitialAlign afterwards; FinalAlign not applied",
              fa2.GetID() == fa.GetID() and slicer.mrmlScene.GetFirstNodeByName("FinalAlign_2") is None
              and post_v.GetParentTransformNode().GetID() == ia_node.GetID()
              and not [n for n in slicer.util.getNodesByClass("vtkMRMLTransformableNode") if n.GetParentTransformNode() is not None
                       and n.GetParentTransformNode().GetID() == fa.GetID()])
        captured.clear()
        fa3, _, _ = sym.intensity_registration(sampling_percentage=0.05)
        check("POST under InitialAlign while registering gives the same result as without (not counted twice)",
              np.allclose(matrix_of(fa3), first_m, atol=1e-4), f"{np.abs(matrix_of(fa3) - first_m).max():.2e}")
        post_v.SetAndObserveTransformNodeID(None)
        pre_v.SetAndObserveTransformNodeID(ia_node.GetID())
        none_f, fixed_prob, _ = sym.intensity_registration()
        check("fixed volume under a transform: no FinalAlign, reported", none_f is None and "free of transforms" in fixed_prob[0], str(fixed_prob))
        pre_v.SetAndObserveTransformNodeID(None)

        # in the pipeline: landmarks -> PRE_align/POST_align -> InitialAlign (exact here) -> FinalAlign
        slicer.mrmlScene.RemoveNode(ia_node)
        slicer.mrmlScene.RemoveNode(fa)
        lm_pre = {"CG": (0, 25, 30), "Se": (0, -5, 10), "Ba": (0, -30, -15), "ANS": (0, 35, -25), "RM": (35, -25, -20), "LM": (-35, -25, -20)}
        add_set("PRE_Landmarks", {k: np.array(v, float) for k, v in lm_pre.items()})
        add_set("POST_Landmarks", {k: (tk @ np.append(np.array(v, float), 1.0))[:3] for k, v in lm_pre.items()})
        post_seg = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode", "POST_Segmentation")
        sphere = vtk.vtkSphereSource()
        sphere.Update()
        post_model = slicer.modules.models.logic().AddModel(sphere.GetOutput())
        post_model.SetName("POST skull")
        poster = slicer.modules.models.logic().AddModel(sphere.GetOutput())
        poster.SetName("POSTER")
        unrelated = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLinearTransformNode", "Unrelated")
        post_other = slicer.modules.models.logic().AddModel(sphere.GetOutput())
        post_other.SetName("POST_other")
        post_other.SetAndObserveTransformNodeID(unrelated.GetID())
        captured.clear()
        logs6 = []
        sym.scene.log_listeners.append(logs6.append)
        steps[:] = [("Side to keep", side_button("Cancel"))]
        sym.run(csv_path=csv2, landmarks=False)
        sym.scene.log_listeners.remove(logs6.append)
        names = [c[0] for c in captured]
        ia4, fa4 = slicer.mrmlScene.GetFirstNodeByName("InitialAlign"), slicer.mrmlScene.GetFirstNodeByName("FinalAlign")
        check("run(): InitialAlign from the landmarks (= the true transform here), then BRAINSFit started from it, making FinalAlign",
              ia4 is not None and fa4 is not None and np.allclose(matrix_of(ia4), truth, atol=1e-6) and names == ["BRAINSFit"]
              and captured[0][1].get("initialTransform") == ia4.GetID() and not errors and not steps, f"{names} {errors}")
        e4 = err(matrix_of(fa4))
        lines.append(f"INFO pipeline FinalAlign at 0.002 starting from the exact InitialAlign: {e4[0]:.3f} deg {e4[1]:.3f} mm from the truth")
        parent_name = lambda n: n.GetParentTransformNode().GetName() if n.GetParentTransformNode() else None
        post_vol = slicer.mrmlScene.GetFirstNodeByName("POST")
        post_lm = slicer.mrmlScene.GetFirstNodeByName("POST_Landmarks")
        check("FinalAlign applied before the calculations: POST volume, POST_Landmarks, POST_Segmentation and 'POST skull' under it",
              all(parent_name(n) == "FinalAlign" for n in (post_vol, post_lm, post_seg, post_model)),
              str([(n.GetName(), parent_name(n)) for n in (post_vol, post_lm, post_seg, post_model)]))
        check("not moved: PRE, PRE_Landmarks, POST_align (copied world positions), 'POSTER' (not a POST name); "
              "POST_other under another transform left there",
              all(parent_name(slicer.mrmlScene.GetFirstNodeByName(n)) is None for n in ("PRE", "PRE_Landmarks", "POST_align"))
              and parent_name(poster) is None and parent_name(post_other) == "Unrelated")
        again_moved, again_problems = sym.apply_final_align()
        check("the application is said in the terminal; POST_other under another transform is reported (and nothing is moved twice)",
              any(m.startswith("Applied FinalAlign to") and "POST_Segmentation" in m for m in logs6)
              and again_moved == [] and len(again_problems) == 1 and "POST_other was not put under FinalAlign" in again_problems[0],
              str(again_problems))
        pre_lm_w = world_points(slicer.mrmlScene.GetFirstNodeByName("PRE_Landmarks"))
        post_lm_w = world_points(post_lm)
        gap = float(np.max(np.linalg.norm(post_lm_w - pre_lm_w, axis=1)))
        check("so the POST landmarks the measurements read now sit on the PRE ones (within 0.2 mm; ~40 mm apart before)",
              gap < 0.2, f"max gap {gap:.3f} mm")
        for n in (post_vol, post_lm, post_seg, post_model):
            n.SetAndObserveTransformNodeID(None)
        captured.clear()
        slicer.mrmlScene.RemoveNode(fa4)
        steps[:] = [("Side to keep", side_button("Cancel"))]
        sym.run(csv_path=csv2, landmarks=False, final_align=False)
        check("final_align=False: BRAINSFit is not run", not captured and slicer.mrmlScene.GetFirstNodeByName("FinalAlign") is None and not errors and not steps)

        # a re-run once POST is under FinalAlign: no registration; the transforms and the _align lists are kept
        steps[:] = [("Side to keep", side_button("Cancel"))]
        sym.run(csv_path=csv2, landmarks=False)                       # makes InitialAlign and FinalAlign again
        ia5, fa5 = slicer.mrmlScene.GetFirstNodeByName("InitialAlign"), slicer.mrmlScene.GetFirstNodeByName("FinalAlign")
        post_all = (post_vol, post_lm, post_seg, post_model)
        check("the run that made FinalAlign applied it", all(parent_name(n) == "FinalAlign" for n in post_all))
        nested = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLinearTransformNode", "Tweak")
        nested.SetAndObserveTransformNodeID(fa5.GetID())
        logs5 = []
        sym.scene.log_listeners.append(logs5.append)
        scenarios = [("the POST volume under FinalAlign", [(post_vol, fa5)], "POST"),
                     ("POST_Landmarks under FinalAlign", [(post_lm, fa5)], "POST_Landmarks"),
                     ("the POST volume under a transform that is under FinalAlign", [(post_vol, nested)], "POST")]
        for title, moves, named in scenarios:
            for n in post_all:
                n.SetAndObserveTransformNodeID(None)
            before = {"ia": (matrix_of(ia5), ia5.GetMTime()), "fa": (matrix_of(fa5), fa5.GetMTime()),
                      "pre_al": slicer.mrmlScene.GetFirstNodeByName("PRE_align").GetID(),
                      "post_al": slicer.mrmlScene.GetFirstNodeByName("POST_align").GetID()}
            for node_, parent_ in moves:
                node_.SetAndObserveTransformNodeID(parent_.GetID())
            captured.clear()
            logs5.clear()
            steps[:] = [("Side to keep", side_button("Cancel"))]
            sym.run(csv_path=csv2, landmarks=False)
            skip_logs = [m for m in logs5 if m.startswith("Skipping InitialAlign and FinalAlign")]
            check(f"re-run with {title}: no BRAINSFit, no wizard update, transforms and _align lists kept, said in the terminal",
                  not captured and not errors and not steps
                  and np.allclose(matrix_of(ia5), before["ia"][0]) and ia5.GetMTime() == before["ia"][1]
                  and np.allclose(matrix_of(fa5), before["fa"][0]) and fa5.GetMTime() == before["fa"][1]
                  and slicer.mrmlScene.GetFirstNodeByName("PRE_align").GetID() == before["pre_al"]
                  and slicer.mrmlScene.GetFirstNodeByName("POST_align").GetID() == before["post_al"]
                  and len(skip_logs) == 1 and named in skip_logs[0],
                  f"captured {[c[0] for c in captured]} logs {skip_logs} {errors}")
            check(f"... and with {title}, FinalAlign is applied to the other POST data",
                  all(parent_name(n) in ("FinalAlign", "Tweak") for n in post_all), str([(n.GetName(), parent_name(n)) for n in post_all]))
        for n in post_all:
            n.SetAndObserveTransformNodeID(None)
        captured.clear()
        logs5.clear()
        steps[:] = [("Side to keep", side_button("Cancel"))]
        sym.run(csv_path=csv2, landmarks=False)
        check("POST taken off FinalAlign again: the next run registers again (BRAINSFit called, no skip message)",
              [c[0] for c in captured] == ["BRAINSFit"] and not any(m.startswith("Skipping InitialAlign") for m in logs5) and not errors and not steps)
        sym.scene.log_listeners.remove(logs5.append)
    finally:
        slicer.cli.runSync = _run_sync

    # ---------------- the Slicer module, loaded through Slicer's own drag-and-drop routine ----------------
    import ExtensionWizard
    from ExtensionWizardLib.ModuleInfo import ModuleInfo
    module_dir = os.path.join(PACKAGE, "SlicerModule")
    check("Slicer's drop detection finds the module in SlicerModule/ (and its file), not in the package folder itself",
          [m.key for m in ModuleInfo.findModules(module_dir, 1)] == ["SymmetrizedPlanning"]
          and [m.key for m in ModuleInfo.findModules(os.path.join(module_dir, "SymmetrizedPlanning.py"), 1)] == ["SymmetrizedPlanning"]
          and ModuleInfo.findModules(PACKAGE, 1) == [])
    settings_before = slicer.app.revisionUserSettings().value("Modules/AdditionalPaths")
    devmode_before = qt.QSettings().value("Developer/DeveloperMode")

    def accept_load(w):
        boxes = w.findChildren(qt.QCheckBox)
        seen["load_boxes"] = [b.text for b in boxes]
        for b in boxes:
            b.checked = False                  # do not change the user's Slicer settings in a test
        w.findChild(qt.QDialogButtonBox).button(qt.QDialogButtonBox.Yes).click()

    already_installed = hasattr(slicer.modules, "symmetrizedplanning")   # e.g. installed in this Slicer's settings
    steps[:] = [] if already_installed else [("", accept_load)]
    ExtensionWizard.ExtensionWizardWidget.loadModules(path=module_dir)
    # Loading a scripted module unbinds, in this script's namespace, the names the module file imports at its top
    # (os, sys, traceback), so they are imported again here.
    import os, sys, traceback
    if already_installed:
        lines.append("INFO the SymmetrizedPlanning module is already installed in this Slicer (loaded at start-up), so the "
                     "drop dialog cannot be shown here; that part was checked on a Slicer without it")
        check("module already installed: dropping it again offers nothing to load, settings unchanged",
              not errors and not steps and slicer.app.revisionUserSettings().value("Modules/AdditionalPaths") == settings_before)
    else:
        check("loaded through the drop dialog (offering 'Add ... to Additional module paths'); your settings unchanged",
              hasattr(slicer.modules, "symmetrizedplanning") and not errors and not steps
              and any("Additional module paths" in t for t in seen.get("load_boxes", []))
              and slicer.app.revisionUserSettings().value("Modules/AdditionalPaths") == settings_before
              and qt.QSettings().value("Developer/DeveloperMode") == devmode_before, str(seen.get("load_boxes")) + str(errors))
    slicer.util.selectModule("SymmetrizedPlanning")
    widget_m = slicer.modules.symmetrizedplanning.widgetRepresentation()
    start_button = slicer.util.findChild(widget_m, "startButton")
    norm = lambda q: os.path.normcase(os.path.normpath(q))
    cats = list(slicer.modules.symmetrizedplanning.categories)
    folder_m = sys.modules["SymmetrizedPlanning"].PACKAGE_FOLDER if "SymmetrizedPlanning" in sys.modules else None
    check("the module is under Craniofacial and finds the package next to it",
          cats == ["Craniofacial"] and folder_m is not None and norm(folder_m) == norm(PACKAGE), f"{cats} {folder_m!r} vs {PACKAGE!r}")
    calls = []
    real_start = sym.start
    sym.start = lambda **k: calls.append(k)
    start_button.click()
    sym.start = real_start
    check("its Start button calls symmetrized.start() (segmentation on: the checkbox is ticked by default)", calls == [{"segment": True}], str(calls))

    def boom(**k):
        raise RuntimeError("test failure")

    sym.start = boom
    steps[:] = [("Symmetrized Planning", msg("Ok"))]
    start_button.click()
    sym.start = real_start
    check("an error in the tool is shown in a message box, not lost in the console", not errors and not steps, str(errors))
    seen["buttons_by_title"] = {}
    steps[:] = [("Landmarks", side_button("Cancel"))]
    start_button.click()
    check("Start with the real tool opens its first question (cancelled here)",
          not errors and not steps and seen.get("buttons_by_title", {}).get("Landmarks", [None])[0] == "Define new landmarks", str(errors))

    # ---------------- optional head/skull segmentation (TotalSegmentator replaced by a stand-in) ----------------
    CRANIOFACIAL = [("mandible", "Mandible"), ("teeth_lower", "Lower teeth"), ("skull", "Skull"), ("head", "Head"),
                    ("sinus_maxillary", "Maxillary sinus"), ("sinus_frontal", "Frontal sinus"), ("teeth_upper", "Upper teeth")]

    class FakeTotalSeg:
        """Stands in for TotalSegmentatorLogic: what its process() leaves in the segmentation for this task (segment IDs
        are the class names, as in its readSegmentation), and what it was asked."""

        def __init__(self, segments=CRANIOFACIAL, fail=False):
            self.higherOrderResampling, self.calls, self.setups, self.segments, self.fail = False, [], 0, segments, fail

        def setupPythonRequirements(self, upgrade=False):
            self.setups += 1

        def process(self, inputVolume, outputSegmentation, quality=None, cpu=False, task=None, subset=None,
                    interactive=False, sequenceBrowserNode=None):
            self.calls.append({"volume": inputVolume.GetName(), "task": task, "quality": quality, "cpu": cpu,
                               "interactive": interactive, "subset": subset, "higherOrderResampling": self.higherOrderResampling})
            if self.fail:
                raise RuntimeError("simulated failure")
            for sid, name in self.segments:
                outputSegmentation.GetSegmentation().AddEmptySegment(sid, name)

    for n in [n for n in slicer.util.getNodesByClass("vtkMRMLSegmentationNode") if n.GetName().endswith("_HeadSkull")]:
        slicer.mrmlScene.RemoveNode(n)                       # made by the quick stand-in above
    fake = FakeTotalSeg()
    real_logic_fn = sym.segmentation.totalsegmentator_logic
    sym.segmentation.totalsegmentator_logic = lambda: fake
    try:
        shn = slicer.vtkMRMLSubjectHierarchyNode.GetSubjectHierarchyNode(slicer.mrmlScene)
        patient = shn.CreateSubjectItem(shn.GetSceneItemID(), "Patient")
        studies = {}
        for vname in ("PRE", "POST"):
            studies[vname] = shn.CreateStudyItem(patient, f"Study {vname}")
            shn.SetItemParent(shn.GetItemByDataNode(slicer.mrmlScene.GetFirstNodeByName(vname)), studies[vname])
        steps[:] = [("Side to keep", side_button("Cancel"))]
        sym.run(csv_path=csv2, landmarks=False, segment=True)
        segs = {v: slicer.mrmlScene.GetFirstNodeByName(f"{v}_HeadSkull") for v in ("PRE", "POST")}
        check("segment=True: TotalSegmentator asked for PRE and POST with task craniofacial_structures, high order resampling on, "
              "normal quality, no pop-ups; its Python packages checked once",
              [c["volume"] for c in fake.calls] == ["PRE", "POST"] and fake.setups == 1 and not errors and not steps
              and all(c["task"] == "craniofacial_structures" and c["higherOrderResampling"] is True and c["quality"] == "normal"
                      and c["interactive"] is False and c["cpu"] is False and not c["subset"] for c in fake.calls), str(fake.calls))
        ids_names = {v: [(i, n.GetSegmentation().GetSegment(i).GetName()) for i in n.GetSegmentation().GetSegmentIDs()] if n else None
                     for v, n in segs.items()}
        check("one segmentation per volume (PRE_HeadSkull, POST_HeadSkull) with head, skull, mandible, lower and upper teeth; "
              "the two sinus segments removed",
              all(v is not None and sorted(i for i, _ in v) == ["head", "mandible", "skull", "teeth_lower", "teeth_upper"] for v in ids_names.values()), str(ids_names))
        check("each segmentation is in its volume's study in the Data module",
              all(segs[v] is not None and shn.GetItemParent(shn.GetItemByDataNode(segs[v])) == studies[v] for v in segs))
        check("POST_HeadSkull is under FinalAlign (with the other POST data), PRE_HeadSkull under no transform",
              parent_name(segs["POST"]) == "FinalAlign" and parent_name(segs["PRE"]) is None,
              f"{parent_name(segs['POST'])} {parent_name(segs['PRE'])}")
        steps[:] = [("Side to keep", side_button("Cancel"))]
        sym.run(csv_path=csv2, landmarks=False, segment=True)
        check("a re-run reuses both segmentations (no new TotalSegmentator run, no second package check)",
              len(fake.calls) == 2 and fake.setups == 1 and slicer.mrmlScene.GetFirstNodeByName("PRE_HeadSkull").GetID() == segs["PRE"].GetID()
              and not errors and not steps)
        steps[:] = [("Side to keep", side_button("Cancel"))]
        sym.run(csv_path=csv2, landmarks=False, segment=False)
        check("segment=False turns it off", len(fake.calls) == 2 and not errors and not steps)
        segs["PRE"].SetAttribute("SymmetrizedKeptSegments", "head,skull")      # as made before mandible/teeth were kept
        steps[:] = [("Side to keep", side_button("Cancel"))]
        sym.run(csv_path=csv2, landmarks=False, segment=True)
        pre_again = slicer.mrmlScene.GetFirstNodeByName("PRE_HeadSkull")
        check("a segmentation made with an older list of kept segments is made again (POST, up to date, is reused)",
              [c["volume"] for c in fake.calls] == ["PRE", "POST", "PRE"] and pre_again is not None
              and sorted(pre_again.GetSegmentation().GetSegmentIDs()) == ["head", "mandible", "skull", "teeth_lower", "teeth_upper"]
              and shn.GetItemParent(shn.GetItemByDataNode(pre_again)) == studies["PRE"]
              and len([n for n in slicer.util.getNodesByClass("vtkMRMLSegmentationNode") if n.GetName().startswith("PRE_HeadSkull")]) == 1
              and not errors and not steps, str([c["volume"] for c in fake.calls]))

        xvol = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode", "X")
        slicer.util.updateVolumeFromArray(xvol, np.zeros((5, 5, 5), dtype=np.int16))
        none_s, fail_prob = sym.segment_head_and_skull(xvol, logic=FakeTotalSeg(fail=True))
        check("a TotalSegmentator failure is reported and leaves no half-made segmentation",
              none_s is None and "simulated failure" in fail_prob[0] and slicer.mrmlScene.GetFirstNodeByName("X_HeadSkull") is None, str(fail_prob))
        part, part_prob = sym.segment_head_and_skull(xvol, logic=FakeTotalSeg(segments=[("skull", "Skull"), ("mandible", "Mandible")]))
        check("missing segments are reported (the ones found are kept)",
              part is not None and list(part.GetSegmentation().GetSegmentIDs()) == ["skull", "mandible"]
              and "found no head, teeth_lower, teeth_upper" in part_prob[0], str(part_prob))
        slicer.mrmlScene.RemoveNode(part)
        # on by default, and done before the landmark question (so the landmarks can be placed on it)
        for n in [n for n in slicer.util.getNodesByClass("vtkMRMLSegmentationNode") if n.GetName() in ("PRE_HeadSkull", "POST_HeadSkull")]:
            slicer.mrmlScene.RemoveNode(n)
        before_calls = len(fake.calls)
        steps[:] = [("Landmarks", side_button("Cancel"))]
        cancelled_out = sym.run(csv_path=csv2)
        check("by default the segmentation runs, first: PRE and POST segmented before the landmark question (cancelled here)",
              cancelled_out == {} and [c["volume"] for c in fake.calls[before_calls:]] == ["PRE", "POST"] and not errors and not steps,
              str([c["volume"] for c in fake.calls[before_calls:]]))

        # while a set is placed: its segmentation in 3D, the other hidden, the head (skin) hidden; all put back afterwards
        pre_s, post_s = slicer.mrmlScene.GetFirstNodeByName("PRE_HeadSkull"), slicer.mrmlScene.GetFirstNodeByName("POST_HeadSkull")
        for n in (pre_s, post_s):
            n.GetDisplayNode().SetVisibility(True)
        restore = sym.placement.show_segmentation_for_placing("PRE")
        pd_, qd_ = pre_s.GetDisplayNode(), post_s.GetDisplayNode()
        during = (pd_.GetVisibility(), pd_.GetVisibility3D(), qd_.GetVisibility(), pd_.GetSegmentVisibility3D("head"),
                  pd_.GetSegmentVisibility3D("skull"), pre_s.GetSegmentation().ContainsRepresentation("Closed surface"))
        restore()
        after = (qd_.GetVisibility(), pd_.GetSegmentVisibility3D("head"))
        check("placing PRE: PRE_HeadSkull shown in 3D as a surface with the head hidden and the skull shown, POST_HeadSkull hidden; "
              "afterwards POST_HeadSkull and the head are shown again",
              during == (True, True, False, False, True, True) and after == (True, True), f"{during} {after}")

        sym.segmentation.totalsegmentator_logic = lambda: None
        _, no_ext = sym.segment_volumes(("X",))
        check("without the TotalSegmentator extension: reported, nothing made", "not installed" in no_ext[0], str(no_ext))
        empty_made, empty_prob = sym.segment_volumes(("NoSuchVolume",))
        check("no volume to segment: a note in the terminal only, no warning", empty_made == {} and empty_prob == [])
    finally:
        sym.segmentation.totalsegmentator_logic = real_logic_fn

    calls = []
    real_start = sym.start
    sym.start = lambda **k: calls.append(k)
    seg_box_default = slicer.util.findChild(widget_m, "segmentCheckBox").checked
    slicer.util.findChild(widget_m, "segmentCheckBox").checked = True
    start_button.click()
    slicer.util.findChild(widget_m, "segmentCheckBox").checked = False
    start_button.click()
    sym.start = real_start
    check("module: the TotalSegmentator checkbox is ticked by default and passes segment=True / False",
          seg_box_default is True and calls == [{"segment": True}, {"segment": False}], str(calls))
    TotalSegmentator.TotalSegmentatorLogic = _RealTotalSegLogic

except Exception:
    ok[0] = False
    lines.append("EXCEPTION\n" + traceback.format_exc())
finally:
    lines.append("ALL PASSED" if ok[0] else "FAILURES")
    with open(RESULTS, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    slicer.app.quit()

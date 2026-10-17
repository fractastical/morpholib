#!/usr/bin/env python3
"""
measure_outputs.py — snapshot the *morphology* behind a MorphoEval audit.

The audit nanopubs pin the code (commit) and the judgement (scores); this tool pins the
form that was judged, in two ways, and writes a sidecar YAML that build_morphoeval_kg.py
merges into the nanopubs:

  1. content fingerprint of every rendered output that an evidence item cites
     (sha256, bytes, media type, pixel size, frame count)  -> schema:ImageObject/sha256
  2. morphometrics measured from the actual geometry / pixels of those outputs
     (and, for Raup, from the script's own generating function)  -> schema:PropertyValue
     evidence on the `morphometric-validation` assessment

Measurements are deliberately simple, stated with their method, and carry a reference
value only where one is actually known; "no reference recorded" is an honest state.

    python3 morphoeval/measure_outputs.py morphoeval/audits/2026-10-13-morphology-audit.yaml
      -> morphoeval/audits/2026-10-13-morphology-audit.measurements.yaml

Requires: numpy, scipy, Pillow, pyyaml.  Outputs that are not on disk are reported as
`missing` and get neither hash nor measurements (nothing is invented).
"""

from __future__ import annotations

import hashlib
import json
import mimetypes
import sys
from pathlib import Path

import numpy as np
import yaml
from PIL import Image, ImageSequence
from scipy import ndimage as ndi

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


# --------------------------------------------------------------------------- #
# 1. fingerprints
# --------------------------------------------------------------------------- #
def fingerprint(path: Path) -> dict:
    h = hashlib.sha256(path.read_bytes()).hexdigest()
    d = {"file": str(path.relative_to(ROOT)), "sha256": h, "bytes": path.stat().st_size,
         "media_type": mimetypes.guess_type(path.name)[0] or "application/octet-stream"}
    try:
        with Image.open(path) as im:
            d["width"], d["height"] = im.size
            d["frames"] = getattr(im, "n_frames", 1)
    except Exception:
        pass
    return d


def frames(path: Path, every: int = 1):
    with Image.open(path) as im:
        return [np.asarray(f.convert("RGB")) for i, f in enumerate(ImageSequence.Iterator(im)) if i % every == 0]


def M(name, value, unit, method, reference=None, source=None, note=None, derived_from="output"):
    """derived_from: 'output' (measured on the rendered file) or 'script' (computed from the
    script's own generating function / run)."""
    m = {"name": name, "value": float(round(value, 4)) if isinstance(value, (float, np.floating)) else value,
         "unit": unit, "method": method, "derived_from": derived_from}
    if reference is not None:
        m["reference_value"] = reference
    if source:
        m["reference_source"] = source
    if note:
        m["note"] = note
    return m


# --------------------------------------------------------------------------- #
# 2. per-simulation morphometrics
# --------------------------------------------------------------------------- #
def measure_planform(gif: Path) -> list:
    """Union silhouette of the superimposed morphologies per frame (central panel only,
    labels excluded): elongation and how much the silhouette changes across the timeline."""
    fs = frames(gif)
    H, W_, _ = fs[0].shape
    r0, r1, c0, c1 = int(0.22 * H), int(0.85 * H), int(0.25 * W_), int(0.75 * W_)   # exclude title/badges
    masks = []
    for a in fs:
        roi = a[r0:r1, c0:c1].astype(int)
        m = roi.min(2) < 235                                   # anything not white
        m = ndi.binary_opening(m, iterations=2)
        lab, n = ndi.label(m)
        if n == 0:
            continue
        sizes = ndi.sum(m, lab, range(1, n + 1))
        masks.append(lab == 1 + int(np.argmax(sizes)))
    ar, areas = [], []
    for m in masks:
        ys, xs = np.where(m)
        ar.append((xs.max() - xs.min() + 1) / (ys.max() - ys.min() + 1)); areas.append(m.sum())
    ious = [(masks[i] & masks[j]).sum() / (masks[i] | masks[j]).sum()
            for i in range(len(masks)) for j in range(i + 1, len(masks))]
    return [
        M("silhouette width/height (bbox, mean over frames)", float(np.mean(ar)), "ratio",
          "largest non-white connected component in the central panel of each GIF frame; bbox W/H",
          reference="≈0.15–0.3 (adult S. mediterranea is ~3–7× longer than wide, long axis = AP)",
          source="qualitative; Lobo et al. 2013 Planform renderings show elongated AP bodies",
          note=f"min {min(ar):.3f}, max {max(ar):.3f}; rendered body is wider than long"),
        M("silhouette IoU between frames (mean)", float(np.mean(ious)), "ratio",
          "pairwise intersection-over-union of the per-frame silhouettes on the fixed canvas",
          reference="distinct morphologies should give low IoU; one body plan over time should be near 1",
          note=f"min {min(ious):.3f}; frames differ mainly by added translucent flaps, not by body form"),
        M("frames with a silhouette", len(masks), "count", "frames where a component was found", note=f"of {len(fs)}"),
    ]


def measure_planform_v2(script: Path) -> list:
    """Geometry decoded by 2013-lobo-planform-morphology.py from PlanformDB: the wild-type
    silhouette against the radii encoded in the database, how many of the morphologies
    render as distinct silhouettes, and how different the canonical phenotypes are."""
    import importlib.util
    import os
    from matplotlib.path import Path as MplPath
    db = os.environ.get("PLANFORM_DB_PATH")
    if not db or not Path(db).is_file():
        raise FileNotFoundError("PLANFORM_DB_PATH not set or not a file")
    spec = importlib.util.spec_from_file_location("planform_v2", script)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod                      # dataclasses need the module registered
    spec.loader.exec_module(mod)
    morphs = mod.load_morphologies(db)
    wt = morphs[1]
    mm = mod.morphometrics(wt)
    # encoded reference: head anterior radius 80 + head→trunk 200 + trunk→tail 200 + tail posterior radius 80
    # = 560 along AP; trunk lateral radii 60 + 60 = 120 across ML (RegionParam / RegionsLink rows of morphology 1)
    enc_len, enc_wid = 80 + 200 + 200 + 80, 60 + 60
    rows = {k: mod.morphometrics(m) for k, m in morphs.items()}

    def sil_key(r):
        return (round(r["ap_length"]), round(r["ml_width"]), round(r["area"], -2), r["n_heads"], r["n_tails"])
    keys = {sil_key(r) for r in rows.values()}
    share_wt = sum(1 for r in rows.values() if sil_key(r) == sil_key(rows[1]))

    def full_key(m):                                   # every encoded number: regions, links, organs
        import dataclasses
        return (tuple((reg.type, tuple(round(p, 3) for p in reg.params)) for _, reg in sorted(m.regions.items())),
                tuple(dataclasses.astuple(l)[2:] for l in m.links),
                tuple(dataclasses.astuple(o)[:3] for o in m.organs))
    full = {full_key(m) for m in morphs.values()}

    def iou(a, b):
        pts = np.vstack(list(a.outlines.values()) + list(b.outlines.values()))
        x0, y0 = pts.min(0) - 10; x1, y1 = pts.max(0) + 10
        gx, gy = np.meshgrid(np.linspace(x0, x1, 256), np.linspace(y0, y1, 256))
        grid = np.column_stack([gx.ravel(), gy.ravel()])
        ma = np.zeros(len(grid), bool); mb = np.zeros(len(grid), bool)
        for poly in a.outlines.values():
            ma |= MplPath(poly).contains_points(grid)
        for poly in b.outlines.values():
            mb |= MplPath(poly).contains_points(grid)
        return float((ma & mb).sum() / (ma | mb).sum())

    out = [
        M("wild-type silhouette AP length", mm["ap_length"], "Planform units",
          "union of decoded region outlines of Morphology 1, rasterised; bbox extent along the AP axis",
          reference=enc_len, source="PlanformDB 2.5.0 RegionParam/RegionsLink of morphology 1 (80+200+200+80)",
          derived_from="script"),
        M("wild-type silhouette ML width", mm["ml_width"], "Planform units", "as above, across the ML axis",
          reference=enc_wid, source="trunk lateral radii 60+60", derived_from="script"),
        M("wild-type AP/ML aspect", mm["aspect"], "ratio", "AP length / ML width of the decoded silhouette",
          reference=round(enc_len / enc_wid, 3), source="encoded radii", derived_from="script",
          note="S. mediterranea in Lobo et al. 2013 Fig. 1A is of this order; the formalism, not a specimen, is the reference"),
        M("morphologies rendered", len(morphs), "count", "Morphology rows with at least one encoded region",
          note="372 rows in the database; 3 have no regions (Lethal, Midbody Buldge, Double head, side tail)",
          derived_from="script"),
        M("distinct body silhouettes", len(keys), "count",
          "unique (AP length, ML width, area to 100 units, #heads, #tails) tuples over all rendered morphologies",
          reference=len(morphs), source="upper bound: one silhouette per morphology",
          note=f"{share_wt} morphologies share the wild-type body and differ only in organs "
               "(ectopic eyes, doubled CNS, pharynx position, ...), which is how the database encodes them",
          derived_from="script"),
        M("distinct encodings (regions + links + organs)", len(full), "count",
          "unique tuples of every encoded number read from the database", reference=len(morphs),
          source="every morphology row is a distinct phenotype", derived_from="script"),
        M("IoU wild type vs double head (#24)", iou(wt, morphs[24]), "ratio",
          "intersection-over-union of the two decoded silhouettes on a shared canvas (region centres aligned by layout)",
          reference="< 1; the two bodies differ only at the posterior cap (tail -> second head), the rest of the difference is organs",
          derived_from="script"),
        M("IoU wild type vs headless (#20)", iou(wt, morphs[20]), "ratio", "as above",
          reference="≈ (trunk+tail area)/(whole-body area) if the head is simply absent", derived_from="script"),
    ]
    return out


def measure_odonata(script: Path) -> list:
    """Numbers behind 2026-eshghi-odonata-wing-cells.py, from the vendored tables and the
    script's own re-segmentation CSV (datasets/odonata/fossil_wing_resegmentation.csv):
    the scaling exponents it reports and how its independent fossil segmentation
    compares with the published table, wing by wing."""
    import importlib.util
    import pandas as pd
    spec = importlib.util.spec_from_file_location("odonata_v1", script)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    df = mod.load_wings(ROOT / "datasets" / "odonata")
    rs = pd.read_csv(ROOT / "datasets" / "odonata" / "fossil_wing_resegmentation.csv")
    out = []
    for grp in ["fossil", "extant_dragonfly", "extant_damselfly"]:
        d = df[df.group == grp]
        s_n, _, r_n = mod.ols_loglog(d.wing_length_mm, d.cell_count)
        s_c, _, _ = mod.ols_loglog(d.wing_length_mm, d.cell_len_median_mm)
        out.append(M(f"cell count ∝ wing length^b, {grp}", s_n, "exponent",
                     f"OLS slope of log10 cell count on log10 wing length, n={len(d)} wings (Eshghi et al. 2026 tables)",
                     reference=2.0, source="isometric prediction for a constant cell size (Hoffmann et al. 2018 inhibition length)",
                     note=f"r = {r_n:.3f}", derived_from="script"))
        out.append(M(f"median cell length ∝ wing length^b, {grp}", s_c, "exponent",
                     "OLS slope of log10 median cell length on log10 wing length", reference=0.0,
                     source="constant cell size", derived_from="script"))
        out.append(M(f"median cell length over wings, {grp}", float(d.cell_len_median_mm.median()), "mm",
                     "median over wings of the per-wing median cell length (Eshghi et al. 2026 tables)",
                     note=f"IQR {d.cell_len_median_mm.quantile(.25):.2f}–{d.cell_len_median_mm.quantile(.75):.2f} mm",
                     derived_from="output"))
    dn = (rs.n_cells - rs.table_cell_count).abs() / rs.table_cell_count
    dl = (rs.cell_len_median_mm - rs.table_cell_len_median_mm).abs() / rs.table_cell_len_median_mm
    dmax = (rs.cell_len_max_mm - rs.table_cell_len_max_mm).abs() / rs.table_cell_len_max_mm
    out += [
        M("fossil drawings re-segmented", len(rs), "count", "fossil venation drawings in the Zenodo archive segmented by the script",
          reference=int((df.group == "fossil").sum()), source="fossil wings in Table S1", derived_from="script"),
        M("scale bar label read from the drawing", int(rs.scale_source.str.startswith("scale bar, label read").sum()), "count",
          "drawings whose scale-bar label was read from the glyphs and found consistent with the tabulated wing length",
          reference=len(rs), note="the rest use the bar with a value inferred from the tabulated wing length, or that length alone",
          derived_from="script"),
        M("cell count, median |ΔN|/N vs Table S1", float(dn.median()), "ratio",
          "per-wing relative difference between the script's cell count and the published count, median over 69 wings",
          reference=0.0, source="Eshghi et al. 2026 Table S1", note=f"{(dn <= 0.02).mean():.0%} of wings within 2%; max {dn.max():.0%} (#7, drawing and table row disagree)",
          derived_from="script"),
        M("median cell length, median |Δ| vs Table S1", float(dl.median()), "ratio",
          "per-wing relative difference of the median cell length (maximum Feret diameter), median over wings",
          reference=0.0, source="Eshghi et al. 2026 Table S1",
          note=f"signed mean {(100 * (rs.cell_len_median_mm - rs.table_cell_len_median_mm) / rs.table_cell_len_median_mm).mean():+.1f}% (cells bounded by vein mid-lines)",
          derived_from="script"),
        M("largest cell length, median |Δ| vs Table S1", float(dmax.median()), "ratio", "as above for the largest cell of each wing",
          reference=0.0, source="Eshghi et al. 2026 Table S1", derived_from="script"),
        M("fossil median cell length, re-segmented", float(rs.cell_len_median_mm.median()), "mm",
          "median over wings of the script's own per-wing median cell length",
          reference=float(rs.table_cell_len_median_mm.median()), source="same statistic from Table S1", derived_from="script"),
    ]
    return out


FIG1_URL = "https://cdn.ncbi.nlm.nih.gov/pmc/blobs/b43e/4895304/02f6a408714c/REG2-1-37-g001.jpg"
FIG1_SHA256 = "b26e25c17bd694a828460919f762b41719f1563006fc7fcbf34eedc1943f1ba2"   # 560x855, PMC4339036, CC BY
FIG1_PANELS = {1: ("A''", 4, 181), 280: ("B''", 198, 383), 314: ("C''", 403, 548),
               490: ("D''", 579, 710), 491: ("E''", 733, 843)}                     # morphology id -> (panel, y0, y1)
FIG1_CARTOON_X0 = 400                                                             # cartoon column of the 560-px figure
FIG1_SEGMENTS = {1: (3, 4, "A′: Upper arm, Fore arm, Hand; digits 1-4"),
                 280: (3, 6, "B′: Thigh, Shank, Foot; digits 1-5 plus the prehallux node"),
                 314: (7, 0, "C′: Coxa, Basis, Ischium, Merus, Carpus, Propus, Dactyl"),
                 490: (8, 0, "D′: Coxa, Trochanter, Femur, Tibia, Tarsus 1-4 (as labelled)"),
                 491: (7, 0, "E′: Coxa, Trochanter, Femur, Patella, Tibia, Metatarsus, Tarsus")}


def _normalised_mask(mask: np.ndarray, size: int = 200) -> np.ndarray:
    """Crop a boolean mask to its bounding box and scale it (aspect kept, longer side = size)
    onto a size x size canvas, so silhouettes drawn at different scales can be overlaid."""
    ys, xs = np.where(mask)
    crop = mask[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    h, w = crop.shape
    s = (size - 2) / max(h, w)
    im = Image.fromarray((crop * 255).astype(np.uint8)).resize((max(1, round(w * s)), max(1, round(h * s))), Image.BILINEAR)
    out = np.zeros((size, size), bool)
    a = np.asarray(im) >= 128
    y0, x0 = (size - a.shape[0]) // 2, (size - a.shape[1]) // 2
    out[y0:y0 + a.shape[0], x0:x0 + a.shape[1]] = a
    return out


def _best_dihedral_iou(a: np.ndarray, b: np.ndarray):
    """IoU of two normalised masks, best over the 8 rotations/reflections of b (the paper's
    cartoons and the decoded drawing need not share an orientation)."""
    best = (0.0, "identity")
    for k in range(4):
        for flip in (False, True):
            m = np.rot90(b, k)
            if flip:
                m = np.fliplr(m)
            m = _normalised_mask(m, a.shape[0])
            iou = float((a & m).sum() / (a | m).sum())
            if iou > best[0]:
                best = (iou, f"rot{90 * k}" + ("+mirror" if flip else ""))
    return best


def measure_limbform(script: Path) -> list:
    """Geometry decoded by 2014-lobo-limbform-morphology.py from LimbformDB 2.1.0: how much of
    the database decodes, whether the joint-radius rule of the formalism holds, the segment and
    digit counts of the five Fig. 1 wild types against the labelled graphs of Fig. 1 A'-E', and
    the decoded silhouettes against the paper's own cartoons (Fig. 1 A''-E'', Regeneration
    1:37, CC BY), downloaded into the git-ignored .limbform_cache/ and SHA-256 pinned."""
    import importlib.util
    import os
    import urllib.request
    spec = importlib.util.spec_from_file_location("limbform_v1", script)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    db = mod.resolve_db_path(os.environ.get("LIMBFORM_DB_PATH"), download=False)
    morphs = mod.load_morphologies(db)
    import sqlite3
    with sqlite3.connect(db) as conn:
        n_rows = conn.execute("SELECT COUNT(*) FROM Morphology").fetchone()[0]
    ok, n_joints = mod.check_encoding(morphs)
    out = [
        M("morphologies decoded", len(morphs), "count", "Morphology rows with at least one Link (segment) row",
          reference=n_rows, source="rows in LimbformDB 2.1.0 Morphology table",
          note="the rest are outcome labels without geometry (Lethal, No limb, ...)", derived_from="script"),
        M("joints obeying the radius rule", ok, "count",
          "joints whose JointParam count equals the number of incident segments (3 for a terminal joint; "
          "a sub-joint counts its host segment twice)", reference=n_joints,
          source="all joints; the rule is the formalism's (Regeneration 1:37, Fig. 2)", derived_from="script"),
    ]
    for mid, (n_seg, n_dig, label) in FIG1_SEGMENTS.items():
        mm = mod.morphometrics(morphs[mid])
        out.append(M(f"segments, #{mid} {mm['name']}", mm["n_segments"], "count", "Link rows of the morphology",
                     reference=n_seg, source=f"Lobo et al. 2014 Regeneration Fig. 1 {label}", derived_from="script"))
        if n_dig:
            out.append(M(f"digits, #{mid} {mm['name']}", mm["n_digits"], "count", "Digit rows of the morphology",
                         reference=n_dig, source=f"Lobo et al. 2014 Regeneration Fig. 1 {label}", derived_from="script"))
    # --- silhouettes against the paper's cartoons --------------------------------------------
    fig = ROOT / ".limbform_cache" / "REG2-1-37-g001.jpg"
    try:
        if not fig.exists():
            fig.parent.mkdir(exist_ok=True)
            req = urllib.request.Request(FIG1_URL, headers={"User-Agent": "morpholib/morphoeval"})
            with urllib.request.urlopen(req, timeout=60) as r:
                fig.write_bytes(r.read())
        got = hashlib.sha256(fig.read_bytes()).hexdigest()
        if got != FIG1_SHA256:
            raise ValueError(f"Fig. 1 download sha256 {got[:12]}… != pinned {FIG1_SHA256[:12]}…")
        rgb = np.asarray(Image.open(fig).convert("RGB")).astype(int)
    except Exception as ex:
        out.append(M("silhouette IoU vs Fig. 1 cartoons", None, None,
                     f"not measured: {type(ex).__name__}: {ex}", derived_from="output"))
        return out
    for mid, (panel, y0, y1) in FIG1_PANELS.items():
        crop = rgb[y0:y1, FIG1_CARTOON_X0:]
        ink = crop.sum(axis=2) < 690                       # anything that is not paper-white
        lab, n = ndi.label(ink)
        if n == 0:
            continue
        sizes = ndi.sum(ink, lab, range(1, n + 1))
        cartoon = ndi.binary_fill_holes(lab == (1 + int(np.argmax(sizes))))
        # the script's silhouette() grid is 256x256 over a non-square extent; rasterise isotropically here
        from matplotlib.path import Path as MplPath
        x0, x1, y0, y1 = mod.extent(morphs[mid])
        nx = 400; ny = max(2, round(nx * (y1 - y0) / (x1 - x0)))
        gx, gy = np.meshgrid(np.linspace(x0, x1, nx), np.linspace(y0, y1, ny))
        grid = np.column_stack([gx.ravel(), gy.ravel()])
        ours = np.zeros(len(grid), bool)
        for poly in mod.all_polys(morphs[mid]):
            ours |= MplPath(poly).contains_points(grid)
        ours = ours.reshape(ny, nx)
        a = _normalised_mask(cartoon)
        iou, orient = _best_dihedral_iou(a, ours)
        cy, cx = np.where(cartoon); oy, ox = np.where(ours)
        asp_c = (cx.max() - cx.min() + 1) / (cy.max() - cy.min() + 1)
        asp_o = (ox.max() - ox.min() + 1) / (oy.max() - oy.min() + 1)
        name = morphs[mid].name
        out.append(M(f"silhouette IoU vs Fig. 1 {panel}, #{mid} {name}", iou, "ratio",
                     "largest ink component of the cartoon panel (holes filled) and the decoded silhouette, each "
                     "cropped to its bounding box and scaled to a 200-px canvas (aspect kept); intersection over "
                     f"union, best of 8 rotations/reflections ({orient})",
                     reference=1.0, source="the paper's own rendering of the same database entry",
                     note="bones, digits and gradient are inside the outline, so this compares outer shape only; "
                          "the cartoon is a 560-px JPEG, which bounds the achievable IoU below 1", derived_from="output"))
        out.append(M(f"bounding-box aspect, #{mid} {name}", float(max(asp_o, 1 / asp_o)), "ratio",
                     "longer / shorter side of the decoded silhouette's bounding box",
                     reference=round(float(max(asp_c, 1 / asp_c)), 3), source=f"same statistic on the Fig. 1 {panel} cartoon",
                     derived_from="script"))
    return out


def measure_celegans(script: Path) -> list:
    """Measurements taken by re-running the decoding functions of 2015-santella-wormguides-celegans.py
    on the SHA-256-pinned WormGUIDES / c302 inputs (git-ignored .celegans_cache/, no download here):
    name-grammar and parent-before-child checks, the calibrated embryo size, and the atlas against
    the cell-count / death-count / timing anchors of Sulston et al. 1983 as quoted by WormAtlas."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("celegans_v1", script)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    local = mod.fetch_inputs(download=False, write_manifest=False)
    frames_ = [mod.read_nuclei(local[f"wormguides/{mod.WG_NUCLEI.format(t=t)}"], t) for t in range(1, mod.N_FRAMES + 1)]
    offset = mod.movie_offset(local[f"wormguides/{mod.WG_FILES['production']}"])
    enc = mod.check_encoding(frames_)
    shells = {t: mod.shell_outline(local[f"wormguides/{mod.WG_SHELL.format(t=t)}"]) for t in mod.SHELL_FRAMES}
    um = mod.calibrate(shells[mod.CALIB_FRAME])
    shell_um = np.ptp(shells[mod.CALIB_FRAME], axis=0) * um
    nuc_um = mod.nuclear_cloud_scale(frames_)
    sr = mod.shell_report(shells, frames_, um)
    rows = mod.measure(frames_, offset, um)
    deaths = [ln.strip() for ln in local[f"wormguides/{mod.WG_FILES['deaths']}"].read_text().splitlines() if ln.strip()]
    seen = {n for f in frames_ for n in f.names}
    WA = "WormAtlas, Embryo introduction (Sulston et al. 1983), accessed 2026-10-15"
    out = [
        M("lineage names parsing as Sulston names", enc["parse_ok"], "count",
          "distinct names in the 360 nuclei files matching ^(AB|MS|E|C|D|EMS|P[0-4]|Z[23])[aplrdv]*$",
          reference=enc["n_names"], source="all distinct names in the atlas", derived_from="script"),
        M("cells first seen after their parent", enc["parent_ok"], "count",
          "non-founder cells whose parent name appears in an earlier frame",
          reference=enc["parent_total"], source="all cells whose parent is in the movie "
          f"({enc['parent_predates_movie']} founders present at frame 1 have parents that predate it)", derived_from="script"),
        M("calibration", um, "µm per atlas unit",
          f"x extent of the atlas's eggshell outline mesh at frame {mod.CALIB_FRAME} (obj_files/embryo_t{mod.CALIB_FRAME}.obj, "
          f"'Embryo Outline' in CellShapesConfig.csv; {sr['length_calib']:.1f} units) set to the 50 µm egg length",
          reference=None, source=None, note="assumption: Gönczy et al. 1999 JCB 144:927 ('50-µm-long cell'); the "
          "atlas coordinates are unit-less (ProductionInfo X/Y/Z_SCALE = 1); the earlier nuclear-cloud estimate "
          f"gives {nuc_um:.4f}; the sampled shell meshes give {sr['um_min']:.4f}-{sr['um_max']:.4f}", derived_from="script"),
        M("eggshell width under the calibration", float(shell_um[1]), "µm",
          f"y extent of the eggshell outline mesh at frame {mod.CALIB_FRAME} (z extent {shell_um[2]:.1f} µm)", reference=30.0,
          source="Gönczy et al. 1999 JCB 144:927 / WormAtlas: the egg is ~50 x 30 µm", derived_from="script",
          note="a consistency check of the calibration: the width is not fitted"),
        M("eggshell length spread across sampled shell meshes", float(sr["length_max"] - sr["length_min"]), "atlas units",
          f"max - min x extent of the eggshell meshes at frames {list(shells)} ({sr['length_min']:.1f}-{sr['length_max']:.1f})",
          derived_from="script", note="one static mesh is reused for frames 1-~250, re-segmented per frame from ~280; "
          "the spread bounds the scale uncertainty (~6 %)"),
        M("nuclear overrun of the shell in x", sr["overrun_max"], "atlas units",
          "largest distance by which a nucleus (centroid ± radius) lies outside the shell mesh's x range at the sampled frames",
          derived_from="script", note=f"{sr['overrun_calib']:.1f} units at frame {mod.CALIB_FRAME}; the overrun is against the "
          "static early mesh, which is why the calibration uses the re-segmented shell"),
        M("nuclear cloud width at the last frame", rows[-1]["extent_y_um"], "µm",
          "y span of nuclear centroids + one mean diameter under the calibration", derived_from="script",
          note="nuclei do not reach the eggshell; compare the eggshell width above"),
        M("nuclei at the last frame", rows[-1]["n_cells"], "count", f"valid rows of t{mod.N_FRAMES:03d}-nuclei "
          f"(movie-minute {rows[-1]['movie_min']})", reference=558, source=f"{WA}: 558 cells at hatching (671 generated, 113 deaths)",
          note="the movie ends before hatching; 553 of the 580 nuclei carry a parts-list identity", derived_from="script"),
        M("maximum nucleus count", max(r["n_cells"] for r in rows), "count", "maximum over the 360 frames",
          reference=550, source=f"{WA}: ~550 cells at the end of proliferation (330-350 min)", derived_from="script"),
        M("deaths listed by the atlas", len(deaths), "count", "names in CellDeaths.csv", reference=113,
          source=f"{WA}: 113 embryonic cell deaths", derived_from="script"),
        M("listed dying cells that occur in the movie", sum(d in seen for d in deaths), "count",
          "CellDeaths.csv names present in at least one nuclei file", reference=len(deaths), derived_from="script"),
    ]
    g = mod.gastrulation_onset(rows)
    if g:
        out.append(M("gastrulation onset, cell count", g["n_cells"], "count",
                     "first frame from which the E cells sit at < 0.8 of the median normalised radius for 5 frames",
                     reference=mod.LIT["gastrulation_cells"], source=f"{WA}: ~30-cell stage (26-28 cells when Ea/Ep ingress)",
                     derived_from="script"))
        out.append(M("gastrulation onset, movie-minute", g["movie_min"], "min", "same frame, frame index + movie start offset",
                     reference=mod.LIT["gastrulation_min"], source=f"{WA}: Ea/Ep enter the embryo at 100 min after first cleavage",
                     note="discrepancy: the atlas clock runs ~27 min ahead of the Sulston schedule at the same cell stage; "
                          "not resolved by the script (temperature or time origin)", derived_from="script"))
    at = [r for r in rows if 330 <= r["movie_min"] <= 350]
    if at:
        out.append(M("nuclei at movie-minute 330-350", f"{at[0]['n_cells']}-{at[-1]['n_cells']}", "count",
                     "nucleus count over that window", reference=550, source=f"{WA}: ~550 cells at 330-350 min",
                     derived_from="script"))
    edges = mod.read_edges(local[f"c302/{mod.C302_EDGES}"])
    nodes = {mod.canonical_cell(a) for a, b, _, _ in edges} | {mod.canonical_cell(b) for a, b, _, _ in edges}
    neurons = sorted(n for n in nodes if mod.NEURON_RE.match(n))
    out.append(M("neurons in the connectome edge list", len(neurons), "count",
                 "upper-case node names of c302 herm_full_edgelist.csv after DA01->DA1 canonicalisation",
                 reference=302, source="Cook et al. 2019 / White et al. 1986: 302 hermaphrodite neurons",
                 note="CANL/CANR are 'no synapses' in the WormGUIDES parts list and so cannot appear in an edge list",
                 derived_from="script"))
    return out


def measure_thompson_fish(script: Path) -> list:
    """Measurements taken by re-running the fitting functions of thompson/1917-thompson-scarus-pomacanthus.py
    on the vendored, SHA-256-checked Claverie & Wainwright landmark table: the Scarus -> Pomacanthus fits
    (similarity, affine, Moebius, oblique coaxial circles, thin-plate spline), the within-genus noise floor,
    a leave-one-landmark-out check, and the position of the pair among all genus pairs."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("thompson_fish_v1", script)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    names, X = mod.load_landmarks()
    Y, groups, S, T, fits, floor_T, per_T, floor_S, per_S, ctx, chaet = mod.analyse(names, X)
    em = np.array([c["explained_coaxial"] for c in ctx]); cm = np.array([c["explained_coaxial"] for c in chaet])
    loo = []
    for k in range(mod.N_LM):
        idx = [i for i in range(mod.N_LM) if i != k]
        pk, _ = mod.fit_coaxial(S[idx], T[idx]); loo.append(np.linalg.norm(mod.coaxial(pk, S[[k]])[0] - T[k]))
    loo_rms = float(np.sqrt(np.mean(np.square(loo))))
    TH = "Thompson 1917, On Growth and Form, p. 749, Figs. 375-376"
    DATA = "Friedman et al. 2019, Dryad 10.5061/dryad.fn457gr (Claverie & Wainwright 2014 landmarks), CC0"
    return [
        M("species in the landmark table", len(names), "count", f"rows of {mod.DATA.name} (sha256 checked)", reference=791,
          source=f"{DATA}: 'nearly 800 species'", derived_from="script"),
        M("Scarus species", len(groups[mod.SOURCE_GENUS]), "count", "rows whose genus is Scarus", derived_from="script"),
        M("Pomacanthus species", len(groups[mod.TARGET_GENUS]), "count", "rows whose genus is Pomacanthus", derived_from="script"),
        M("Procrustes distance, Scarus mean to Pomacanthus mean", fits["rms_similarity"], "Procrustes units (RMS per landmark)",
          "RMS landmark distance after optimal similarity superimposition of the genus means (unit centroid size)", derived_from="script"),
        M("affine residual", fits["rms_affine"], "Procrustes units", "RMS after a least-squares affine map (6 parameters)",
          note=f"explains {100*fits['explained_affine']:.1f} % of the Procrustes distance", derived_from="script"),
        M("Moebius residual (isogonal coaxial circles)", fits["rms_moebius"], "Procrustes units",
          "RMS after a least-squares Moebius map z -> (az+b)/(cz+1), the conformal map that turns the Cartesian grid into "
          "two orthogonal pencils of coaxial circles", reference=None,
          source=f"{TH}: 'a system of (approximately) coaxial circles'",
          note=f"explains {100*fits['explained_moebius']:.1f} %: a conformal map cannot change the depth/length ratio", derived_from="script"),
        M("oblique coaxial-circle residual", fits["rms_coaxial"], "Procrustes units",
          "RMS after an affine map followed by z -> z/(cz+1) (8 parameters): coaxial circles whose two pencils need not be orthogonal, "
          "the family Thompson's Fig. 376 belongs to as drawn", source=TH,
          note=f"explains {100*fits['explained_coaxial']:.1f} %", derived_from="script"),
        M("thin-plate spline bending energy", fits["bending_energy"], "dimensionless",
          "bending energy of the exact TPS from the Scarus mean to the Pomacanthus mean (Bookstein 1989)", derived_from="script"),
        M("within-Pomacanthus RMS (noise floor)", floor_T, "Procrustes units",
          "RMS of the Pomacanthus species about their own Procrustes mean", derived_from="script",
          note=f"species range {per_T.min():.4f}-{per_T.max():.4f}; within-Scarus {floor_S:.4f}"),
        M("leave-one-landmark-out RMS of the oblique coaxial family", loo_rms, "Procrustes units",
          "each landmark predicted by the 8-parameter map fitted to the other 16", derived_from="script",
          note="compare with the Procrustes distance: the family has no predictive value for held-out landmarks if it is not smaller"),
        M("depth/length, Scarus mean", float(np.ptp(S[:, 1]) / np.ptp(S[:, 0])), "ratio", "landmark extent in y over extent in x", derived_from="script"),
        M("depth/length, Pomacanthus mean", float(np.ptp(T[:, 1]) / np.ptp(T[:, 0])), "ratio", "landmark extent in y over extent in x", derived_from="script"),
        M("percentile of Scarus -> Pomacanthus among all genus pairs (oblique coaxial, explained share)",
          float(100 * (em < fits["explained_coaxial"]).mean()), "percent",
          f"share of the {len(ctx)} ordered pairs of genus means (genera with >= {mod.MIN_SPECIES} spp.) explained less well",
          note=f"median pair {100*np.median(em):.0f} %", derived_from="script"),
        M("Scarus -> Chaetodon spp., median explained by oblique coaxial circles", float(100 * np.median(cm)), "percent",
          f"median over the {len(chaet)} Chaetodon species", source=f"{TH}: 'other fishes of the same family of Chaetodonts ... small variations in the values of the constants'",
          note=f"range {100*cm.min():.0f}-{100*cm.max():.0f} %", derived_from="script"),
    ] + (measure_thompson_outlines(mod, Y, names) if hasattr(mod, "outline_analysis") else [])


def measure_thompson_outlines(mod, Y, names) -> list:
    """The real-outline test (script versions from 2026-10-17): the two Commons photographs are fetched or read
    from the cache and SHA-256-checked by the script; the silhouettes, anchors and carried outlines are the
    script's.  The IoU of each carried Scarus silhouette against the Pomacanthus silhouette is re-derived here
    with a different rasteriser (matplotlib.path point-in-polygon on a 900 x 900 grid, not skimage.draw) and
    reported beside the script's value; the areas and the anchor positions are read off the masks directly."""
    from matplotlib.path import Path as MplPath
    O = mod.outline_analysis(Y, names, allow_download=True)
    if O is None:
        return [M("real-outline test", None, None, "photographs not available (offline and not cached)")]
    BT = O["photos"]["Pomacanthus_semicirculatus"]["boundary"]
    lo = np.min([np.min(c, 0) for c in list(O["carried"].values()) + [BT]], 0) - 0.02
    hi = np.max([np.max(c, 0) for c in list(O["carried"].values()) + [BT]], 0) + 0.02
    g = 900
    gx, gy = np.meshgrid(np.linspace(lo[0], hi[0], g), np.linspace(lo[1], hi[1], g))
    pts = np.column_stack([gx.ravel(), gy.ravel()])
    inT = MplPath(BT).contains_points(pts)
    out = []
    for key, ph in O["photos"].items():
        cols = np.where(ph["mask"].any(0))[0]
        out.append(M(f"silhouette area, {key.replace('_', ' ')}", ph["area_px"], "pixels",
                     f"segmented mask at {mod.PHOTO_SCALE:g} scale; photograph SHA-256 {mod.PHOTOS[key]['sha256'][:12]}... checked by the script",
                     source=mod.PHOTOS[key]["credit"], derived_from="script"))
        out.append(M(f"caudal-peduncle anchor, {key.replace('_', ' ')}", float((ph["peduncle"][0] - cols[0]) / (cols[-1] - cols[0])), "fraction of silhouette length",
                     "x of the peduncle anchor between the anterior-most and posterior-most mask columns (fish facing left)",
                     note="plausible range for a perciform fish ~0.75-0.90; the anchor fixes the registration of the landmarks to the photograph",
                     derived_from="script"))
    for k in ["similarity", "affine", "moebius", "coaxial", "tps"]:
        inC = MplPath(O["carried"][k]).contains_points(pts)
        iou = float((inC & inT).sum() / (inC | inT).sum())
        out.append(M(f"silhouette IoU, {k}", iou, "ratio",
                     "Scarus ghobban silhouette carried by the map fitted to the 17 landmarks of S. ghobban -> P. semicirculatus, "
                     "against the P. semicirculatus silhouette; point-in-polygon on a 900 x 900 grid (independent of the script's rasteriser)",
                     reference=round(O["metrics"][k]["iou"], 4), source="the script's own IoU (skimage.draw.polygon, 1000-px grid)",
                     note=f"boundary distance {O['metrics'][k]['boundary_distance']:.4f} = {100*O['metrics'][k]['boundary_distance']/O['body_length']:.1f} % of body length; "
                          f"landmark RMS {O['landmark_rms'][k]:.4f}" + ("; the no-deformation baseline" if k == "similarity" else ""),
                     derived_from="script"))
    return out


def measure_raup(script: Path, variables: Path) -> list:
    """Read W, D and T back off meshes made by the script's own raup_shell(), independently of
    any check the script carries.  Raup's definitions: W = ratio of the generating-curve
    centre's distance from the axis over one revolution; D = inner margin / outer margin of
    the generating curve; T = axial movement of the centre per unit of its radial movement
    (dz_c/dr_c), as restated by Noshita 2014 and Okabe & Yoshimura 2017.  Each generating
    curve is a column of the mesh (rows = position on the curve); its centre is the mean over
    the curve, its outer/inner margins the max/min radial distance."""
    src = script.read_text()
    s = src.index("def raup_shell"); e = src.index("\ndef ", s + 10)
    ns = {"np": np}; exec(src[s:e], ns)                       # only the generating function
    raup_shell = ns["raup_shell"]
    out = []
    for sh in json.loads(variables.read_text())["empirical_shells"]:
        W, D, T = sh["W"], sh["D"], sh["T"]
        res = 100
        X, Y, Z = raup_shell(W=W, D=D, T=T, turns=sh.get("turns", 6), res=res)
        if X.shape[0] != res:                                  # rows must be the generating curve
            X, Y, Z = X.T, Y.T, Z.T
        R = np.hypot(X, Y)
        closed = np.allclose([X[0], Y[0], Z[0]], [X[-1], Y[-1], Z[-1]])   # seam point duplicated?
        rows = slice(0, -1) if closed else slice(None)
        rc, zc = R[rows].mean(axis=0), Z[rows].mean(axis=0)
        xc, yc = X[rows].mean(axis=0), Y[rows].mean(axis=0)
        Ro, Ri = R.max(axis=0), R.min(axis=0)
        n = rc.size
        ang = np.unwrap(np.arctan2(yc, xc))                     # coiling angle of each generating curve
        pw = int(round(2 * np.pi / np.mean(np.diff(ang))))      # columns per revolution
        i1, i0 = n - 1, n - 1 - pw
        W_rec = rc[i1] / rc[i0]
        D_rec = Ri[i1] / Ro[i1]
        T_rec = abs(zc[i1] - zc[i0]) / (rc[i1] - rc[i0]) if rc[i1] > rc[i0] else float("nan")
        width = max(X.max() - X.min(), Y.max() - Y.min())
        hw = (Z.max() - Z.min()) / width
        spire = abs(zc[-1] - zc[0]) / width
        tag = f"'{sh['name']}' (W={W}, D={D}, T={T})"
        out.append(M(f"recovered translation rate T, {tag}", round(float(T_rec), 6), "ratio",
                     "Δz_c / Δr_c of the generating-curve centre over the last revolution of raup_shell()",
                     reference=T, source="Raup's definition of T (dz_c/dr_c), Noshita 2014 / Okabe & Yoshimura 2017; should equal the input",
                     note=f"height/width = {hw:.3f}, spire/width = {spire:.3f}", derived_from="script"))
        out.append(M(f"recovered distance from axis D, {tag}", round(float(D_rec), 6), "ratio",
                     "inner margin / outer margin of the aperture (min/max radial distance of the last generating curve)",
                     reference=D, source="Raup's definition of D; should equal the input", derived_from="script"))
        out.append(M(f"recovered whorl expansion W, {tag}", round(float(W_rec), 6), "ratio",
                     "centre distance from the axis at the aperture / one revolution earlier",
                     reference=W, source="Raup's definition of W; should equal the input", derived_from="script"))
    return out


def measure_turing(gif: Path, N: int = 200) -> list:
    """Dominant pattern wavelength (radial FFT peak) in grid cells, pattern coverage, and
    whether the animation actually evolves (frame-to-frame change inside the axes box)."""
    fs = frames(gif)
    a0, aL = fs[0].astype(int), fs[-1].astype(int)
    # axes box = pixels that are viridis-coloured (not white/grey/black): saturated
    sat = aL.max(2) - aL.min(2)
    box = ndi.binary_opening(sat > 40, iterations=3)
    ys, xs = np.where(box)
    r0, r1, c0, c1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    yellow = lambda a: (a[r0:r1, c0:c1, 0] > 150) & (a[r0:r1, c0:c1, 2] < 120)     # viridis high end
    yl, y0 = yellow(aL), yellow(a0)
    g = aL[r0:r1, c0:c1].mean(2); g = g - g.mean()
    F = np.abs(np.fft.fftshift(np.fft.fft2(g))) ** 2
    h, w = F.shape; cy, cx = h // 2, w // 2
    yy, xx = np.indices(F.shape); rr = np.hypot(yy - cy, xx - cx).astype(int)
    radial = np.bincount(rr.ravel(), F.ravel()) / np.maximum(np.bincount(rr.ravel()), 1)
    radial[:4] = 0
    k = int(np.argmax(radial[: min(h, w) // 2]))
    wl_cells = (min(h, w) / k) * N / min(h, w) if k else float("nan")
    lab, ncomp = ndi.label(yl)
    change = float(np.mean(yl != y0))
    return [
        M("dominant wavelength (last frame)", float(wl_cells), "grid cells",
          f"peak of the radial power spectrum of the axes-box image, scaled to the N={N} grid",
          reference=None, source=None,
          note="Gray–Scott f=0.055,k=0.062 labyrinth; no specimen or literature wavelength recorded"),
        M("pattern coverage (last frame)", float(yl.mean()), "fraction", "fraction of axes box in the high-V (yellow) band"),
        M("connected high-V components (last frame)", int(ncomp), "count", "scipy.ndimage.label on the yellow mask"),
        M("fraction of pattern pixels changed, first→last frame", change, "fraction",
          "XOR of yellow masks of frame 0 and the last frame", reference=">0 for an evolving simulation",
          note="≈0 means the GIF shows a frozen pattern"),
    ]


def measure_hoffmann(script: Path) -> list:
    """Run the wing script (seeded, uses the cached example wing) and take the morphometrics
    it computes itself on real vs model domains: circularity c = 4πA/P² and domain counts.
    The real wing is the specimen here, so the reference values are measured, not quoted."""
    import re
    import subprocess
    r = subprocess.run([sys.executable, str(script)], cwd=ROOT, capture_output=True, text=True, timeout=1800)
    if r.returncode != 0:
        raise RuntimeError(f"{script.name} exited {r.returncode}: {r.stderr.strip()[-300:]}")
    vals = {}
    for line in r.stdout.splitlines():
        if (m := re.search(r"Real wing: (\d+) vein domains; wing area (\d+) px", line)):
            vals["domains_real"], vals["area_px"] = int(m.group(1)), int(m.group(2))
        if line.startswith("Circularity"):
            m = re.search(r"real ⟨c⟩=([0-9.]+) \(n=(\d+)\)\s+model ⟨c⟩=([0-9.]+) \(n=(\d+)\)\s+hexagon ([0-9.]+)", line)
            if m:
                vals.update(c_real=float(m[1]), n_real=int(m[2]), c_model=float(m[3]), n_model=int(m[4]), hexagon=float(m[5]))
    if "c_model" not in vals:
        raise RuntimeError("could not parse circularity line from script output")
    return [
        M("mean domain circularity, model wing", vals["c_model"], "ratio (4πA/P²)",
          "script's domain_stats() over Voronoi domains after Lloyd equilibration (seed 0)",
          reference=vals["c_real"], source="same statistic on the authors' segmented real wing (Hoffmann et al. 2018 example)",
          note=f"regular hexagon = {vals['hexagon']}; model n={vals['n_model']}, real n={vals['n_real']}",
          derived_from="script"),
        M("vein-domain count, model wing", vals["n_model"], "count", "domains large enough to measure, model",
          reference=vals["n_real"], source="same count on the real wing",
          note="equal by construction: the model is seeded with the real wing's domain count", derived_from="script"),
        M("vein-domain count, real wing (all)", vals["domains_real"], "count",
          "peaks of the distance transform inside the segmented real wing", note=f"wing area {vals['area_px']} px",
          derived_from="script"),
    ]


# --------------------------------------------------------------------------- #
def main():
    audit_path = (Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "audits" / "2026-10-13-morphology-audit.yaml").resolve()
    audit = yaml.safe_load(audit_path.read_text())
    out = {"audit": audit["audit"]["id"], "outputs": {}, "measurements": {}, "missing": []}
    for sim in audit["simulations"]:
        if sim.get("status") != "evaluated":
            continue
        sid = sim["id"]
        files = {e["file"] for d in sim["scores"].values() for e in d.get("evidence", []) if e["lines"][1] == 0}
        if sim.get("output"):
            files.add(sim["output"])
        try:   # measurements first: some re-render their output, which must then be fingerprinted
            if sid == "planform-morphology-animation" and sim["script"] == "2013-lobo-planform-morphology.py":
                out["measurements"][sid] = measure_planform_v2(ROOT / sim["script"])
            elif sid == "planform-morphology-animation" and (ROOT / sim["output"]).exists():
                out["measurements"][sid] = measure_planform(ROOT / sim["output"])
            elif sid == "raup-shell-coiling":
                out["measurements"][sid] = measure_raup(ROOT / sim["script"], ROOT / "datasets/raup/variables.json")
            elif sid == "turing-gray-scott" and (ROOT / sim["output"]).exists():
                out["measurements"][sid] = measure_turing(ROOT / sim["output"])
            elif sid == "hoffmann-wing-venation":
                out["measurements"][sid] = measure_hoffmann(ROOT / sim["script"])
            elif sid == "eshghi-odonata-wing-cells":
                out["measurements"][sid] = measure_odonata(ROOT / sim["script"])
            elif sid == "lobo-limbform-morphology":
                out["measurements"][sid] = measure_limbform(ROOT / sim["script"])
            elif sid == "santella-wormguides-celegans":
                out["measurements"][sid] = measure_celegans(ROOT / sim["script"])
            elif sid == "thompson-scarus-pomacanthus":
                out["measurements"][sid] = measure_thompson_fish(ROOT / sim["script"])
        except Exception as ex:  # a failed measurement is reported, never silently skipped
            out["measurements"][sid] = [{"name": "measurement failed", "value": None, "unit": None,
                                         "method": f"{type(ex).__name__}: {ex}"}]
        for f in sorted(files):
            p = ROOT / f
            if p.exists():
                out["outputs"].setdefault(sid, []).append(fingerprint(p))
            else:
                out["missing"].append({"simulation": sid, "file": f})
        print(f"{sid}: {len(out['outputs'].get(sid, []))} output(s) fingerprinted, "
              f"{len(out['measurements'].get(sid, []))} measurement(s)")
        for m in out["measurements"].get(sid, []):
            ref = f"   [ref {m['reference_value']}]" if m.get("reference_value") is not None else ""
            print(f"    {m['name']}: {m['value']} {m['unit'] or ''}{ref}")
    if out["missing"]:
        print("missing outputs (no hash, no measurement):", ", ".join(f"{m['simulation']}:{m['file']}" for m in out["missing"]))
    side = audit_path.with_name(audit_path.stem + ".measurements.yaml")
    side.write_text("# Generated by morphoeval/measure_outputs.py — fingerprints and morphometrics of the\n"
                    "# rendered outputs that the audit judged. Regenerate after re-rendering.\n"
                    + yaml.safe_dump(out, sort_keys=False, allow_unicode=True, width=110))
    print("->", side.relative_to(ROOT))


if __name__ == "__main__":
    main()

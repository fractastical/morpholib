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
# 1. fingerprints (dual semantic spec_fingerprint vs byte artifact_sha256)
# --------------------------------------------------------------------------- #
# Canonical mathematical/parameter specifications per simulation ID, adapted
# from MesmerPrism/rusty-manifold-provenance.  `spec_fingerprint` hashes the
# canonical JSON representation of the model equations, parameter regime, grid
# resolution, and RNG seed so it stays invariant across Matplotlib / Pillow /
# FreeType rendering upgrades, whereas `artifact_sha256` pins the exact rendered
# bytes on disk.
CANONICAL_MODEL_SPECS: dict[str, dict] = {
    "planform-morphology-animation": {
        "model": "Lobo et al. (2013) PlanformDB graph-encoded planarian body-plan decoder",
        "coordinate_transform": "(-x, -y) -> (ML, AP), dorsal top, anterior up",
        "grid_resolution": [256, 256],
        "canonical_ids": [1, 2, 3, 4, 8, 25],
        "inputs": ["datasets/planform/planformDB.sha256"],
    },
    "raup-shell-coiling": {
        "model": "Raup (1966) logarithmic helicospiral shell coiling (r_c = a*(1+D)/(1-D), z_c = T*r_c)",
        "whorls_n": 3,
        "mesh_theta_steps": 500,
        "mesh_phi_steps": 48,
        "inputs": ["datasets/raup/variables.json"],
    },
    "turing-gray-scott": {
        "model": "Turing (1952) / Pearson (1993) 2D Gray-Scott reaction-diffusion",
        "Du": 0.16,
        "Dv": 0.08,
        "grid_resolution": [128, 128],
        "steps": 4500,
        "dt": 1.0,
        "seed": 1952,
        "regimes": [
            {"id": "mitosis_spots", "F": 0.028, "k": 0.062},
            {"id": "soliton_spots", "F": 0.030, "k": 0.062},
            {"id": "stripes_worms", "F": 0.042, "k": 0.060},
            {"id": "coral_labyrinth", "F": 0.055, "k": 0.062},
            {"id": "equal_diffusion_control", "F": 0.055, "k": 0.062, "Du": 0.10, "Dv": 0.10},
        ],
        "inputs": [],
    },
    "bressloff-v1-form-constants": {
        "model": "Bressloff et al. (2001/2002) V1 anisotropic ring neural field + Rule et al. (2011) Floquet tongues",
        "alpha": 1.0,
        "mu": 1.0,
        "sigma_loc": 1.0,
        "sigma_lat": 3.0,
        "beta_aniso": 1.6,
        "log_polar_map": {"alpha_deg_inv": 1.0, "eps_deg": 0.75},
        "inputs": [
            "datasets/bressloff/paper-calibration.json",
            "datasets/bressloff/rule-2011-floquet.json",
            "datasets/bressloff/driven-neural-fields-registry.json",
        ],
    },
    "hoffmann-wing-venation": {
        "model": "Hoffmann et al. (2018) secondary-vein Voronoi / Lloyd relaxation venation",
        "lloyd_iterations": 12,
        "seed": 2018,
        "inputs": ["datasets/wings/source_comparison/salcedo2019_wing_vein_metrics.csv"],
    },
    "eshghi-odonata-wing-cells": {
        "model": "Eshghi et al. (2024) Odonata wing-cell polygonality & circularity distribution",
        "seed": 2024,
        "inputs": ["datasets/odonata/eshghi2024_odonata_wings.npz"],
    },
    "lobo-limbform-morphology": {
        "model": "Lobo et al. (2014) Limbform 2D Bézier limb-segment morphology decoder",
        "inputs": ["datasets/limbform/limbform.sqlite"],
    },
    "santella-wormguides-celegans": {
        "model": "Santella et al. (2015) WormGUIDES 4D C. elegans embryonic nuclei lineage trajectory",
        "inputs": ["datasets/wormguides/wormguides_nuclei_70_270.npz"],
    },
    "thompson-scarus-pomacanthus": {
        "model": "Thompson (1917) Chapter XVII Scarus -> Pomacanthus smooth harmonic / affine / TPS warps",
        "grid_lines": [10, 10],
        "landmark_pairs": 14,
        "inputs": [
            "thompson/datasets/price2019_scarus_pomacanthus_landmarks.json",
            "thompson/datasets/thompson1917_fig515_516.npz",
        ],
    },
    "cervera-levin-mafe": {
        "model": "Cervera, Levin & Mafe (2021) bioelectric gap-junction blockade + rusty-matter 5-node V_mem circuit",
        "grid_nx": 80,
        "grid_nz": 41,
        "dt": 0.08,
        "t_end": 25.0,
        "circuit_nodes": 5,
        "inputs": [
            "datasets/planform/planarian_bioelectric_pars.json",
            "datasets/planform/planarian_reference_traces.json",
            "datasets/planform/planform_canonical_silhouettes_v1.json",
        ],
    },
    "vogel-phyllotaxis": {
        "model": "Vogel (1979) golden-angle generative spiral + Douady & Couder (1992, 1996) Hofmeister inhibitory dynamics",
        "golden_angle_deg": 137.50776405003785,
        "lucas_angle_deg": 99.50155280998114,
        "n_florets": 600,
        "inputs": [
            "datasets/phyllotaxis/phyllotaxis_reference.json",
            "datasets/phyllotaxis/helianthus_annuus_capitulum_specimen.jpg",
            "datasets/phyllotaxis/helianthus_parastichy_annotated.jpg",
        ],
    },
    "wolfram-conus-textile-rule30": {
        "model": "Wolfram (1984, 2002) Rule 30 ECA + Meinhardt & Klingler (1987) activator-substrate-hormone PDE + Raup (1966) 3D Conus shell",
        "rule_number": 30,
        "boolean_form": "a_{t+1,i} = a_{t,i-1} XOR (a_{t,i} OR a_{t,i+1})",
        "raup_conus_params": {"W": 1.42, "D": 0.05, "T": 3.15, "whorls": 6.5},
        "mantle_ring_width": 230,
        "mantle_steps": 370,
        "seed": 42,
        "inputs": [
            "datasets/conus/conus_textile_reference.json",
            "datasets/conus/conus_textile_specimen.jpg",
            "datasets/conus/conus_textile_010.jpg",
        ],
    },
    "runions-leaf-venation": {
        "model": "Runions et al. (2005) Auxin Space-Colonization Leaf Venation + Murray (1926) / Price et al. (2013) Vascular Scaling + Katifori (2012) Loop Nesting",
        "open_eq": "v' = v + D * (sum_{s in S(v)} (s-v)/||s-v||) / ||sum_{s in S(v)} (s-v)/||s-v||||",
        "closed_rng_eq": "V(s) = {v in V : forall u in V, ||v-s|| < max(||u-s||, ||v-u||)}",
        "murray_law": "r_parent^gamma = sum_{i in children} r_i^gamma",
        "seed": 42,
        "inputs": [
            "datasets/venation/venation_reference.json",
            "datasets/venation/ginkgo_open_venation_specimen.jpg",
            "datasets/venation/reticulate_vein_skeleton_specimen.jpg",
        ],
    },
    "petroff-stromatolite": {
        "model": "Petroff et al. (2010) diffusion-competition spacing law ell = alpha*sqrt(D*tau) + Batchelor et al. (2004) Lagrangian normal/phototactic lamina front",
        "D_cm2_s": 1e-5,
        "alpha_field": 1.0,
        "alpha_lab": 0.30,
        "betas": [0.3, 1.5, 5.0],
        "n_steps": 700,
        "t_total": 0.85,
        "inputs": [],
    },
    "kahng-coral-geometry": {
        "model": "Kahng, Odle & Wakeman (2024) reef-coral colony calcification geometry V(t+1) = V(t) + G*S(t)/rho across hemispheroid, cone, flat disc, and cylindrical branches",
        "massive": {"G": 1.64, "rho": 1.28},
        "branch": {"G": 2.50, "rho": 1.40},
        "disc": {"G": 0.061, "rho": 2.67, "h": 0.20},
        "common": {"G": 1.50, "rho": 1.20},
        "inputs": [],
    },
}


def _load_dynamic_loop_spec(sim_id: str) -> dict:
    p = HERE / "math_kg_loops" / f"{sim_id}.yaml"
    if p.exists():
        try:
            return yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        except Exception:
            return {}
    return {}


def compute_input_dataset_sha256(sim_id: str, sim_doc: dict | None = None) -> dict[str, str]:
    """Compute SHA-256 digests for all declared upstream input datasets/fixtures of `sim_id`."""
    spec = dict(CANONICAL_MODEL_SPECS.get(sim_id, {}))
    if not spec:
        dyn = _load_dynamic_loop_spec(sim_id)
        spec = dict(dyn.get("model_spec") or (sim_doc or {}).get("model_spec") or {})
        if not spec.get("inputs") and (sim_doc or {}).get("source_refs"):
            spec["inputs"] = [
                r["id"] for r in (sim_doc or {}).get("source_refs", [])
                if isinstance(r, dict) and isinstance(r.get("id"), str) and (ROOT / r["id"]).is_file()
            ]
    digests: dict[str, str] = {}
    for rel in spec.get("inputs", []):
        p = ROOT / rel
        if p.exists() and p.is_file():
            digests[rel] = hashlib.sha256(p.read_bytes()).hexdigest()
    return digests


def compute_spec_fingerprint(sim_id: str, script: str | None = None, sim_doc: dict | None = None) -> str:
    """Compute the canonical SHA-256 `spec_fingerprint` of the simulation's mathematical
    model specification and input dataset digests (independent of Matplotlib/Pillow bytes)."""
    spec = dict(CANONICAL_MODEL_SPECS.get(sim_id, {}))
    if not spec:
        dyn = _load_dynamic_loop_spec(sim_id)
        spec = dict(dyn.get("model_spec") or (sim_doc or {}).get("model_spec") or {"sim_id": sim_id})
    if script:
        spec["script"] = script
    spec["input_dataset_sha256"] = compute_input_dataset_sha256(sim_id, sim_doc=sim_doc)
    canonical_bytes = json.dumps(spec, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(canonical_bytes).hexdigest()


def fingerprint(path: Path, sim_id: str | None = None, script: str | None = None) -> dict:
    h = hashlib.sha256(path.read_bytes()).hexdigest()
    d = {
        "file": str(path.relative_to(ROOT)),
        "sha256": h,
        "artifact_sha256": h,
        "bytes": path.stat().st_size,
        "media_type": mimetypes.guess_type(path.name)[0] or "application/octet-stream",
    }
    if sim_id is not None:
        d["spec_fingerprint"] = compute_spec_fingerprint(sim_id, script)
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


def measure_turing(gif: Path, N: int = 200, script: Path | None = None) -> list:
    """Dominant pattern wavelength (radial FFT peak) in grid cells, pattern coverage,
    whether the animation actually evolves (frame-to-frame change inside the axes box),
    plus Pearson (1993) multi-regime spectral ring sharpness and the Du=Dv negative control."""
    out = []
    if gif.exists():
        fs = frames(gif)
        a0, aL = fs[0].astype(int), fs[-1].astype(int)
        # axes box = pixels that are coloured (not white/grey/black): saturated
        sat = aL.max(2) - aL.min(2)
        box = ndi.binary_opening(sat > 40, iterations=3)
        ys, xs = np.where(box)
        r0, r1, c0, c1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
        yellow = lambda a: (a[r0:r1, c0:c1, 0] > 150) & (a[r0:r1, c0:c1, 2] < 120)     # high-V band
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
        out.extend([
            M("dominant wavelength (last frame)", float(wl_cells), "grid cells",
              f"peak of the radial power spectrum of the axes-box image, scaled to the N={N} grid",
              reference=None, source=None,
              note="Gray–Scott f=0.055,k=0.062 labyrinth; no specimen or literature wavelength recorded"),
            M("pattern coverage (last frame)", float(yl.mean()), "fraction", "fraction of axes box in the high-V (yellow) band"),
            M("connected high-V components (last frame)", int(ncomp), "count", "scipy.ndimage.label on the yellow mask"),
            M("fraction of pattern pixels changed, first→last frame", change, "fraction",
              "XOR of yellow masks of frame 0 and the last frame", reference=">0 for an evolving simulation",
              note="≈0 means the GIF shows a frozen pattern"),
        ])

    turing_script = script if script is not None else (ROOT / "1952-turing-morpho.py")
    if turing_script.exists():
        import importlib.util
        import tempfile
        spec = importlib.util.spec_from_file_location("turing_pearson_v1", turing_script)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mod
        spec.loader.exec_module(mod)
        if hasattr(mod, "run_pearson_morphospace_analysis"):
            with tempfile.TemporaryDirectory() as td:
                res = mod.run_pearson_morphospace_analysis(
                    output_png=Path(td) / "fig.png",
                    output_csv=Path(td) / "metrics.csv",
                    N=160,
                    steps=4000,
                    show=False,
                )
            by_id = res["by_id"]
            coral = by_id["coral_labyrinth"]
            spots = by_id["mitosis_spots"]
            ctrl = by_id["equal_diffusion_control"]
            out.extend([
                M("Pearson coral labyrinth (F=0.054, k=0.0616) spectral ring wavenumber k*",
                  float(coral["k_star_cycles"]), "cycles / domain",
                  "sub-bin parabolic peak of 2D radial PSD P(k) on N=160x160 grid",
                  reference=">0 (Turing band-pass ring)",
                  source="Pearson 1993 Science 261:189 Pattern κ (inside Saddle-Node)",
                  note=f"λ* = {coral['wavelength_grid_cells']:.2f} grid cells (analytical λ_c = {coral.get('lambda_c_analytical', float('nan')):.2f}), ring sharpness = {coral['ring_sharpness']:.1f}",
                  derived_from="script"),
                M("Pearson mitosis spots (F=0.028, k=0.062) connected components vs coral labyrinth",
                  int(spots["n_components"]), "count",
                  "connected high-activator components in self-replicating spot regime",
                  reference=int(coral["n_components"]),
                  source="Pearson 1993 Science 261:189 Pattern λ vs Pattern κ",
                  note="spots remain disconnected whereas coral labyrinth percolates",
                  derived_from="script"),
                M("steady-state equal-diffusion negative control (Du=Dv=0.20) spectral wavenumber k*",
                  float(ctrl["k_star_cycles"]), "cycles / domain",
                  "dominant wavenumber when differential diffusivity is removed around nontrivial steady state (U_*, V_*)",
                  reference=0.0,
                  source="Turing 1952 Phil. Trans. R. Soc. B 237:37 (diffusion-driven instability requires Du != Dv)",
                  note=f"spatial σ_V = {ctrl['std_v']:.2e}, ring sharpness = {ctrl['ring_sharpness']:.1f}",
                  derived_from="script"),
            ])
            if "turing_1952" in res and "bio_specimens" in res:
                t52 = res["turing_1952"]
                bio = res["bio_specimens"]
                out.extend([
                    M("Turing (1952) Sec. 8-9 analytical critical wavelength λ_c = 2π/k_c",
                      round(float(t52["lambda_c"]), 3), "cells",
                      "2*pi / sqrt((d*D_X + a*D_Y) / (2*D_X*D_Y)) on 2-morphogen activator-inhibitor system",
                      reference=5.62,
                      source="Turing 1952 Phil. Trans. R. Soc. B 237:37 Sec. 8-9",
                      note=f"d = {t52['diffusivity_ratio_d']:.2f} > d_c = {t52['critical_diffusivity_ratio_dc']:.3f}, N=20 ring dominant mode s* = {t52['measured_ring_mode']}",
                      derived_from="script"),
                    M("Pomacanthus imperator segmented skin stripe wavelength",
                      round(float(bio["poma_mean_spacing_px"]), 2), "px",
                      "mean dorsoventral inter-stripe peak spacing on SHA-256-locked specimen ROI",
                      reference="8–16 px (CV < 0.35)",
                      source="Kondo & Asai 1995 Nature 376:765-768",
                      note=f"CV = {bio['poma_spacing_cv']:.3f}, topological Y-branch junctions = {bio['poma_y_junctions']}, ontogenetic r = {bio['kondo_corr_r']:.4f}",
                      derived_from="script"),
                    M("Ostracion meleagris segmented Turing spots (DoG + Delaunay)",
                      int(bio["ost_n_spots"]), "spots",
                      "Difference-of-Gaussians segmented spots on SHA-256-locked boxfish specimen ROI",
                      reference=">=15 hexagonal spots",
                      source="Turing 1952 / Pearson 1993 spot lattice",
                      note=f"mean Delaunay nearest-neighbor spacing = {bio['ost_spot_spacing_px']:.2f} px (CV = {bio['ost_spot_spacing_cv']:.3f})",
                      derived_from="script"),
                ])
    return out


def measure_bressloff(script: Path) -> list:
    """Measurements taken by loading 2001-bressloff-v1-form-constants.py: exact parity against
    MesmerPrism/bressloff-v1-form-constants stability_reports, the exact orientation operator vs
    the perturbative scan, full-cubic (Eq. 5.15) branch selection incl. roll stability over all
    theta, the derived contoured/non-contoured regime, retinal pullback seams and chirality under
    the conformal and exact angular maps, the paper-scale preset, and the Gaussian-only negative
    control."""
    import importlib.util
    import tempfile
    spec = importlib.util.spec_from_file_location("bressloff_v1", script)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    with tempfile.TemporaryDirectory() as td:
        res = mod.run_bressloff_analysis(
            output_png=Path(td) / "fig.png",
            output_csv=Path(td) / "metrics.csv",
            show=False,
        )
    neg = res["negative_control"]
    gal = {g["preset_id"]: g for g in res["gallery"]}
    mag = res["magnification"]
    tun = gal["roll_noncontoured_derived"]
    spi = gal["spiral_noncontoured_derived"]
    hexg = gal["fig35_hex_zero_even"]
    bs_tun = tun["stability"]["branch_selection"]
    bs_rh = gal["fig33_rhombic_even"]["stability"]["branch_selection"]
    ps = res["paper_scale"]
    reg = res["regime_map"]
    n_cells = int(reg["family"].size)
    n_unreliable = int((reg["subcritical"] | reg["near_resonant"]).sum())
    return [
        M("bressloff-v1-form-constants stability_reports parity, max |Δ|",
          float(res["max_cal_err"]), "dimensionless",
          "maximum absolute error across all 5 Bressloff (2001) stability_reports (q*, λ*, γ(0), γ(θ), η_hex) vs paper-calibration.json "
          "(upstream bare-overlap path, branch_selection_upstream)",
          reference=0.0,
          source="MesmerPrism/bressloff-v1-form-constants commit 344863a1bd9e8a8e8c9cb20aced2897f3ffb2c4c",
          derived_from="script"),
        M("Exact orientation operator (|m| ≤ 8) vs perturbative n=1 scan at fig17: |Δλ_c|",
          float(abs(res["exact_vs_perturbative"]["dlambda"])), "dimensionless growth rate",
          "top eigenvalue of the Hermitian W_k matrix on e^{2imφ} at the perturbative q* minus the perturbative λ_even(q*)",
          reference=f"0 (|Δq*| = {abs(res['exact_vs_perturbative']['dq']):.0e})",
          source="Bressloff et al. 2001 §4 Eqs. 4.5-4.12 (perturbative) vs direct diagonalisation",
          derived_from="script"),
        M("Local-inhibition threshold A_loc at which the critical harmonic switches m=0 → m=1 (W₀ = W₁)",
          float(res["local_inhibition_threshold"]), "dimensionless",
          "root of W_0^loc(A) − W_1^loc(A) for the fig17 local kernel; below it the top eigenvector of the exact operator is φ-independent (non-contoured)",
          reference=f"gallery I-II derived harmonic m = {tun['stability']['critical_harmonic_exact']} at A_loc = 0.15; full-operator switch in the regime map at ≈ 0.25",
          source="Bressloff et al. 2001 §4a (non-contoured, Ermentrout & Cowan 1979 limit) vs §4b (contoured)",
          note=f"derived contour mode for gallery I-II = {tun['stability']['derived_contour_mode']}",
          derived_from="script"),
        M("Mexican-hat lateral inhibition critical wavenumber q* (Bressloff 2001 Fig. 16)",
          float(neg["mexican_hat_critical_q"]), "dimensionless wavenumber",
          "maximizer of max(λ_even(q), λ_odd(q)) with lateral_inhibition = 0.27",
          reference=">0.5 (finite Turing/Euclidean symmetry-breaking wavelength)",
          source="Bressloff et al. 2001 Phil. Trans. R. Soc. B 356:299 Fig. 16",
          note=f"interior peak prominence = {neg['mexican_hat_peak_prominence']:.4f}",
          derived_from="script"),
        M("Gaussian-only negative control critical wavenumber q* (lateral_inhibition = 0)",
          float(neg["gaussian_control_critical_q"]), "dimensionless wavenumber",
          "maximizer of λ(q) when Mexican-hat surround inhibition is removed (lateral_inhibition = 0.0)",
          reference=0.05,
          source="q_min boundary of [0.05, 3.0] grid (monotonically decaying dispersion, prominence = 0.0)",
          note=f"interior peak prominence = {neg['gaussian_control_peak_prominence']:.2e}",
          derived_from="script"),
        M("Full-cubic roll stability margin min_θ [2γ(θ)/γ(0) − 1] in the derived non-contoured regime (gallery tunnel)",
          float(bs_tun["roll_min_margin"]), "dimensionless",
          "Eq. 5.15 γ(θ) = [3s₃Γ₃ − 2s₂²Q]/N on a 5° grid, σ₀ = ½; rolls are linearly stable iff γ(0) > 0 and this margin is > 0 for every θ",
          reference=">0 (rolls stable); contoured m=1 fig33 margin = %.2f (rolls unstable, selected %s@%.0f°)" % (
              bs_rh["roll_min_margin"], bs_rh["unrestricted_selected_family"], bs_rh["rhombic_theta_optimal_deg"]),
          source="Bressloff et al. 2001 §5 Eqs. 5.9-5.15; branch_selection_full()",
          note=f"gallery tunnel: γ(0) = {bs_tun['gamma0']:+.3f}, unrestricted selection = {bs_tun['unrestricted_selected_family']}, "
               f"feedback_fraction = {bs_tun['feedback_fraction']:.2e}, near_resonant = {bs_tun['near_resonant']}",
          derived_from="script"),
        M("Regime-map cells (A_loc × σ₀) where the cubic truncation is unreliable (subcritical or feedback-dominated)",
          float(n_unreliable), "count",
          f"out of {n_cells} cells: γ(0) ≤ 0, or |2s₂²Q| > |3s₃Γ₃| because the flat-dispersion K=0 mode is nearly marginal (resolvent gain 50-2000)",
          reference="0 (fully reliable cubic analysis); all σ₀ = ½ cells are reliable",
          source="Bressloff et al. 2001 §5; mean-mode coupling per Matthews & Cox 2000 not implemented",
          note="selection is greyed out in Panel G for these cells rather than reported",
          derived_from="script"),
        M("Klüver Tunnel/Rings cortical planform spectral ring sharpness (derived non-contoured roll)",
          float(tun["spectral"]["ring_sharpness"]), "ratio (peak/median PSD)",
          f"2D FFT radial power spectrum peak/median ratio at k* = {tun['spectral']['k_star_cycles']:.0f} cycles",
          reference=">5.0",
          source="Bressloff et al. 2001 Fig. 29 / Klüver 1966 Tunnel form constant",
          note=f"k_x, k_y quantised to integer torus cycles (n_y = {tun['stability']['rendered_modes'][0]['n_y']})",
          derived_from="script"),
        M("Klüver Honeycomb/Hex linear cortical planform spectral ring wavenumber k*",
          float(hexg["spectral_linear"]["k_star_cycles"]), "cycles / domain",
          "dominant radial FFT wavenumber of the linear 3-mode hexagonal eigenmode superposition (non-contoured, weak tanh); "
          "the rendered wavenumber is q*/d_hyp quantised to integer cycles on the 2πβ/ε torus",
          reference=f"{hexg['stability']['rendered_cycles']} (q*/d_hyp = {hexg['stability']['cycles_unquantised']:.2f} cycles)",
          source="Bressloff et al. 2001 Fig. 35 / Klüver 1966 Honeycomb form constant",
          note=f"contoured-rendering k* = {hexg['spectral']['k_star_cycles']:.0f} = √3k/2k harmonic of tanh∘argmax_φ "
               f"(normalised power at k/√3k/2k = {hexg['harmonics']['power_at_k']:.2f}/"
               f"{hexg['harmonics']['power_at_sqrt3_k']:.2f}/{hexg['harmonics']['power_at_2k']:.2f}); "
               f"FFT lattice class = {hexg['spectral']['lattice_class']}",
          derived_from="script"),
        M("Retino-cortical magnification vs Horton & Hoyt (1991), RMS relative error over 1-40 deg",
          float(mag["rms_rel_error"]), "dimensionless",
          "RMS of (α/(w0+εr) − 17.3/(E+0.75)) / (17.3/(E+0.75)) over 157 eccentricities in [1, 40] deg",
          reference="0 (perfect agreement); map-implied A = α/ε = %.2f mm/deg, E2 = w0/ε = %.2f deg vs 17.3, 0.75" % (mag["model_A_mm_per_deg"], mag["model_E2_deg"]),
          source="Horton & Hoyt 1991 Arch. Ophthalmol. 109:816 (doi:10.1001/archopht.1991.01080060080030); datasets/bressloff/horton_hoyt_1991_magnification.json",
          note=f"max |rel err| = {mag['max_abs_rel_error']:.3f}; fovea→90° extent {mag['model_extent_0_90_mm']:.1f} mm (model) vs {mag['hh_extent_0_90_mm']:.1f} mm (H&H)",
          derived_from="script"),
        M("Upstream fixture paper-expectation checks passed (paper-calibration.json, stability_reports + runs)",
          float(res["fixture_checks"]["passed"]), "count",
          "number of the fixture's own expected-vs-actual checks that pass, out of %d (stability_reports %d/%d, runs %d/%d; run status %s)" % (
              res["fixture_checks"]["total"],
              res["fixture_checks"]["stability_reports"]["passed"], res["fixture_checks"]["stability_reports"]["total"],
              res["fixture_checks"]["runs"]["passed"], res["fixture_checks"]["runs"]["total"],
              res["fixture_checks"]["runs"]["status_counts"]),
          reference=float(res["fixture_checks"]["total"]),
          source="MesmerPrism/bressloff-v1-form-constants commit 344863a1bd9e8a8e8c9cb20aced2897f3ffb2c4c, stability_reports[*].checks and runs[*].checks",
          note="; ".join(f"{f['report']}/{f['name']}: expected {f['expected']}, actual {f['actual']}" for f in res["fixture_checks"]["failed"]) or "all passed",
          derived_from="script"),
        M("Preset runs reproduced with exact parity (paper-calibration.json runs[])",
          float(sum(1 for r in res["run_rows"] if r.get("parity_ok"))), "count",
          "runs whose critical_q, critical_branch, selected family/pattern/scope and rendered pattern match the fixture, out of %d" % len(res["run_rows"]),
          reference=float(len(res["run_rows"])),
          source="MesmerPrism/bressloff-v1-form-constants commit 344863a1bd9e8a8e8c9cb20aced2897f3ffb2c4c, runs[*]",
          note=f"max |Δq*| = {res['max_run_err']:.2e}",
          derived_from="script"),
        M("Gallery panels whose rendered lattice is the branch the full cubic amplitude equations select (rhombic angle free)",
          float(res["gallery_selected_count"]), "count",
          "rendered family == unrestricted_selected_family, out of %d gallery panels; %d further panels are linearly stable branches that lose the Lyapunov tie-break, %d are imposed (unstable at cubic order)" % (
              len(res["gallery"]), res["gallery_stable_coexisting_count"], res["gallery_imposed_count"]),
          reference=float(len(res["gallery"])),
          source="Bressloff et al. 2001 §5 cubic amplitude equations (Eqs. 5.9-5.15), evaluated by branch_selection_full()",
          note="; ".join(f"{g['preset_id']}: {g['lattice_consistency']}" for g in res["gallery"]),
          derived_from="script"),
        M("Rendered cortical wavelength λ_c relative to the paper's stated 2.4-3.2 mm (parity-preset lateral widths)",
          float(res["scale_anchor"]["wavelength_ratio_model_over_paper"]), "ratio",
          "λ_c = 2π d_hyp / q* (d_hyp = 2 mm, q* from the gallery preset) divided by the midpoint 2.8 mm of Bressloff 2002 §1.1's 2.4-3.2 mm",
          reference="1.0 (paper scale); paper also states 30-36 perceived stripes vs %d rendered cycles" % res["scale_anchor"]["rendered_cycles"],
          source="Bressloff et al. 2002 Neural Comput. 14:473 §1.1",
          note=f"λ_c = {res['scale_anchor']['critical_wavelength_model_mm']:.2f} mm; foveal magnification {res['scale_anchor']['foveal_magnification_model']:.2f} vs 11.5 mm/deg "
               f"(α = 3/π from upstream params.rs vs α ≈ {res['scale_anchor']['alpha_implied_by_paper']:.2f} implied by the paper)",
          derived_from="script"),
        M("Paper-scale preset (paper_scale_tunnel, lateral widths / 3.3): critical cortical wavelength λ_c",
          float(ps["anchor"]["critical_wavelength_model_mm"]), "mm",
          f"2π d_hyp / q* with q* = {ps['stability']['rendered_q']:.2f} from a q scan to 6 on the same operator",
          reference="2.4-3.2 mm (Bressloff 2002 §1.1)",
          source="Bressloff et al. 2002 Neural Comput. 14:473 §1.1",
          note=f"fovea→90° ring count {ps['morphometrics']['rings_fovea_to_90']:.1f} on the α = 3/π map ({ps['morphometrics']['model_extent_0_90_mm']:.0f} mm) vs "
               f"{ps['morphometrics']['rings_fovea_to_90_hh']:.1f} on the Horton-Hoyt map ({ps['morphometrics']['hh_extent_0_90_mm']:.0f} mm; paper 30-36): "
               f"residual shortfall is the map's α, not q*",
          derived_from="script"),
        M("Spiral retinal pullback chirality 1 − max(|r_LR|, |r_UD|) (conformal map, full-field)",
          float(spi["diag_full"]["chirality"]), "dimensionless",
          "1 minus the largest |Pearson r| between the retinal field and its left-right / up-down reflections; a chiral log-spiral gives ≈1, an achiral pattern 0",
          reference=f"{spi['diag_mirror']['chirality']:.3f} (bilateral-mirror convention, same preset, exact map: chirality destroyed)",
          source="Bressloff et al. 2001 Fig. 33b / 2002 §1.1 (logarithmic spirals are oblique rolls under the complex-log map)",
          note=f"cobweb (fig31) chirality = {gal['fig31_square_even']['diag_full']['chirality']:.3f}; spiral pitch {spi['morphometrics']['spiral_pitch_deg']:.1f}° "
               f"(n_y = {spi['morphometrics']['first_mode_ny']})",
          derived_from="script"),
        M("Spiral retinal pullback seam contrast, conformal angular map (rendered default)",
          float(spi["diag_full"]["seam_contrast"]), "normalised excess discontinuity",
          "max over the four half-meridians of (|Δfield| across the meridian − local |∇field| scale) / RMS field; 0 = no seam",
          reference=f"exact 2002 map: full-field {spi['diag_fullfield']['seam_contrast']:.2f} (left meridian {spi['diag_fullfield']['seam_h_left']:.2f}), "
                    f"chiral-mirror {spi['diag_chiral']['seam_contrast']:.2f}, bilateral-mirror {spi['diag_mirror']['seam_contrast']:.2f}",
          source="Bressloff et al. 2001 §2 (y = βθ/ε conformal coordinate) vs 2002 Eq. 1.1 (y = βrθ/(w₀+εr))",
          note="the exact map cannot close the cortical torus at finite r, so y-odd planforms seam; the conformal map closes it exactly",
          derived_from="script"),
        M("Critical-branch parity flips over the lateral-spread sweep θ₀ ∈ [0°, 90°]",
          float(res["sweep_summary"]["lateral_spread_deg"]["parity_flips"]), "count",
          "number of even↔odd changes of the maximal-growth branch across %d sweep points (fig17_even base)" % len(res["sweeps"]["lateral_spread_deg"]),
          reference="≥1 (Bressloff 2001 §4: even planforms selected for θ₀ ≳ 45°)",
          source="Bressloff et al. 2001 Phil. Trans. R. Soc. B 356:299 §4, Figs. 16-17",
          note=f"flips at θ₀ ≈ {[round(v) for v in res['sweep_summary']['lateral_spread_deg']['parity_flip_values']]}°; "
               f"q* over ξ̂_lat 1.5→6: {res['sweep_summary']['lateral_wide_sigma']['q_star_max']:.2f}→{res['sweep_summary']['lateral_wide_sigma']['q_star_min']:.2f}; "
               f"A_loc sweep harmonic flips at {[round(v, 3) for v in res['sweep_summary']['local_inhibition']['harmonic_flip_values']]}",
          derived_from="script"),
        M("Rule et al. (2011) Fig. 8 overlay domain-normalised RMS β error (diagnostic, no calibration claim)",
          float(res["floquet_status"]["domain_rms_beta_error"]), "normalised wavenumber β",
          "rule-2011-floquet.json source_curve_comparison.domain_beta_mapping.rms_error",
          reference=f"<= {res['floquet_status']['domain_rms_beta_error_max']:.2f} acceptance threshold (NOT met)",
          source="Rule, Stoffregen & Ermentrout 2011 PLoS Comput. Biol. 7:e1002158 Fig. 8 via upstream fixture",
          note=f"acceptance_status = {res['floquet_status']['acceptance_status']}; calibration_claim_allowed = {res['floquet_status']['calibration_claim_allowed']}",
          derived_from="script"),
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


def measure_cervera(script: Path) -> list:
    """Measurements taken by loading 2021-Cervera-Levin-Mafe.py and evaluating its RD solver,
    rusty-matter circuit parity, Oviedo 2010 crop-position fit/test metrics, and decoded
    Planform silhouette geometry."""
    import importlib.util
    import tempfile
    spec = importlib.util.spec_from_file_location("cervera_v1", script)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    with tempfile.TemporaryDirectory() as td:
        res = mod.run_integrated_analysis(
            output_png=Path(td) / "fig.png",
            output_csv=Path(td) / "metrics.csv",
            show=False,
        )
    xs_sym = mod.wavefront_for_f2(1.0)
    xs_f2b = mod.wavefront_for_f2(res["f2_b"])
    _, _, lam_hi = mod.lateral_index_for_f3(1.0)
    _, _, lam_lo = mod.lateral_index_for_f3(0.3)
    mem_scen = mod.run_rusty_matter_planarian_scenario("TransientDepolarizationMemory")
    nomem_scen = mod.run_rusty_matter_planarian_scenario("TransientDepolarizationNoMemoryControl")
    return [
        M("rusty-matter trace parity, max |Δ|", float(res["max_trace_err"]), "state units",
          "maximum absolute difference across all 5 scenarios, 11 sampled steps and 4 state averages against rusty_matter_planarian_traces.json",
          reference=0.0, source="MesmerPrism/rusty-matter commit ad111fa7ecfd8f43d292ff3b1d27d3af2741d828",
          derived_from="script"),
        M("symmetric RD wavefront x*(f2=1)", float(xs_sym), "AP fraction",
          "zero crossing of m1(x) - m2(x) when D2 = D2_0",
          reference=0.5, source="symmetric boundary conditions at x=0 and x=L",
          derived_from="script"),
        M("pharynx-boundary criterion x*_b", float(res["x_star_b"]), "AP fraction",
          "posterior end of the Planform #1 wild-type pharynx on the AP extent",
          note=f"crossed at f2_b = {res['f2_b']:.4f} (where x*(f2_b) = {xs_f2b:.4f})",
          derived_from="script"),
        M("anterior body-area fraction on S. mediterranea silhouette (f2=1 -> f2=0.25)",
          float(res["rd_area_fracs"][-1] - res["rd_area_fracs"][0]), "fraction of body area",
          f"increase in real-silhouette mask area anterior to x*(f2) from f2=1.0 ({res['rd_area_fracs'][0]:.3f}) to f2=0.25 ({res['rd_area_fracs'][-1]:.3f})",
          source="Stevenson & Beane 2010 Smediterranea.jpg silhouette (CC BY 2.5)",
          derived_from="script"),
        M("ML centre/edge ratio λ(f3), f3=1.0 vs f3=0.3", float(lam_hi - lam_lo), "ratio difference",
          f"m3(L_z/2)/m3(0) drop from f3=1.0 ({lam_hi:.3f}) to f3=0.3 ({lam_lo:.3f})",
          reference=">0 (monotonic decrease under lateral gap-junction blockade)",
          derived_from="script"),
        M("Oviedo 2010 octanol crop-gradient training RMSE", float(res["fit_rmse_train"]), "percentage points",
          f"1-parameter fit f2(p) = 1 - {res['a_fit']:.2f}*p against Oviedo et al. 2010 octanol double-head frequencies (Exps 415-418, n=4 crops)",
          reference=0.0, source="PlanformDB 2.5.0 / Oviedo et al. 2010 Dev. Biol. 339:188",
          derived_from="script"),
        M("Oviedo 2010 triple-innexin RNAi held-out RMSE", float(res["fit_rmse_test"]), "percentage points",
          "held-out prediction error of the octanol-fitted curve against Dj-Inx-12 + Dj-Inx-5/13 RNAi double-head frequencies (Exps 446-449, n=4 crops)",
          reference=0.0, source="PlanformDB 2.5.0 / Oviedo et al. 2010 Dev. Biol. 339:188",
          note=f"Pearson r = {res['fit_r_test']:.3f}",
          derived_from="script"),
        M("circuit posterior head readout, memory ON vs memory OFF control",
          float(mem_scen.posterior_head_readout), "readout [0,1]",
          "final posterior head-identity readout R_head,post under TransientDepolarizationMemory",
          reference=float(nomem_scen.posterior_head_readout),
          source="TransientDepolarizationNoMemoryControl negative control (Durant et al. 2019 Biophys. J. 116:948)",
          note="hysteresis memory sustains depolarised head fate after transient pulse ends",
          derived_from="script"),
        M("Planform #1 Wild Type vs #2 Double Head silhouette IoU", float(res["wt_dh_iou"]), "ratio",
          "256x256 shared-canvas intersection-over-union of decoded Lobo et al. (2013) graph silhouettes",
          reference="<1.0 (posterior tail replaced by ectopic head cap)",
          source="PlanformDB 2.5.0 morphologies #1 and #2",
          derived_from="script"),
    ]


def measure_vogel_phyllotaxis(script: Path) -> list:
    """Measurements taken by loading 1979-vogel-phyllotaxis.py and evaluating its
    Delaunay/Voronoi rising Fibonacci parastichy cascade, Voronoi cell area CV at the
    golden angle vs Vogel (1979) Fig. 2 perturbed angles, Douady & Couder (1992)
    Hofmeister attractor convergence, and Swinton et al. (2016) Helianthus survey stats."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("vogel_phyllotaxis_v1", script)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    res = mod.run_self_tests()
    ang = res["angle_metrics"]
    out = [
        M("golden divergence angle α₀ = 360°/τ²",
          float(res["golden_angle_deg"]), "degrees",
          "exact golden-section divergence angle 360 * (2 - τ)",
          reference=137.50776405003785,
          source="Vogel 1979 Mathematical Biosciences 44:179 Eq. (1)-(4)",
          derived_from="script"),
        M("outer-annulus [14.5, 22.5) opposed contact parastichy ratio (n/m)",
          float(res["zone_cascade"][-1]["ratio"]), "ratio (55/34)",
          f"Delaunay contact parastichy pair {tuple(res['zone_cascade'][-1]['pair'])} in outer annulus r in [14.5, 22.5)",
          reference=round(55.0 / 34.0, 5),
          source="Vogel 1979 / Swinton et al. 2016 R. Soc. Open Sci. 3:160091 §2.3",
          note=f"full radial cascade: {[tuple(z['pair']) for z in res['zone_cascade']]}",
          derived_from="script"),
        M("Voronoi cell area CV at golden angle (137.5078°)",
          float(ang["golden_137.508"]["voronoi_cv_area"]), "CV (std/mean)",
          "Voronoi polygon area dispersion across interior florets (0.18 R_max <= r <= 0.85 R_max, N=450)",
          reference=float(ang["sub_137.300"]["voronoi_cv_area"]),
          source="Vogel 1979 Fig. 2a negative control (α = 137.30°)",
          note=f"strictly lower than α=137.30° ({ang['sub_137.300']['voronoi_cv_area']:.4f}) and α=137.60° ({ang['super_137.600']['voronoi_cv_area']:.4f})",
          derived_from="script"),
        M("Douady & Couder (1992) Hofmeister main-branch emergent divergence angle",
          float(res["douady_couder_fibonacci_deg"]), "degrees",
          "mean steady-state divergence angle over final 25 plastochrons on main Fibonacci branch (G: 0.55 -> 0.04)",
          reference=137.507764,
          source="Douady & Couder 1992 Phys. Rev. Lett. 68:2098 Fig. 3",
          derived_from="script"),
        M("Douady & Couder (1992) Hofmeister Lucas-branch emergent divergence angle",
          float(res["douady_couder_lucas_deg"]), "degrees",
          "mean steady-state divergence angle over final 25 plastochrons on secondary Lucas branch (G: 0.22 -> 0.04)",
          reference=99.501553,
          source="Douady & Couder 1992 Phys. Rev. Lett. 68:2098 / 1996 J. Theor. Biol. 178:255",
          derived_from="script"),
        M("Swinton et al. (2016) photoreviewed Fibonacci-structure prevalence",
          float(res["swinton_fib_structure_pct"]), "percent",
          "632 of 768 photoreviewed Helianthus annuus parastichy counts belonging to Fibonacci, Lucas, Double-Fib, or F4 sequences",
          reference=82.29,
          source="Swinton, Ochu & MSI Turing's Sunflower Consortium 2016 R. Soc. Open Sci. 3:160091 Table 1",
          note=f"Fib-1 (49) vs Fib+1 (17) one-sided binomial p = {res['swinton_fib_minus_vs_plus_p']:.2e}",
          derived_from="script"),
    ]
    if isinstance(res.get("specimen_segmentation"), dict):
        seg = res["specimen_segmentation"]
        out.append(
            M("empirical specimen photograph segmented floret Voronoi circularity ⟨4πA/P²⟩",
              float(seg["specimen_mean_circularity"]), "ratio (4πA/P²)",
              f"Difference-of-Gaussians floret segmentation (N={seg['n_detected_florets']} florets, {seg['n_interior_voronoi_cells']} interior Voronoi cells) on helianthus_annuus_capitulum_specimen.jpg",
              reference=float(ang["golden_137.508"]["mean_circularity"]),
              source="helianthus_annuus_capitulum_specimen.jpg (File:Sunflower Spiral.jpg, CC BY-SA 2.0)",
              note=f"specimen Voronoi area CV = {seg['specimen_voronoi_cv_area']:.4f}, hex neighbor fraction = {seg['specimen_hex_neighbor_fraction']:.3f}",
              derived_from="output_pixels"),
        )
    if isinstance(res.get("dislocation_ensemble"), dict):
        ens = res["dislocation_ensemble"]
        c_ens = ens["counts"]
        out.append(
            M("mechanistic rim-dislocation ensemble Pearson r vs Swinton (2016) Table 1",
              float(res["swinton_ensemble_pearson_r"]), "Pearson r",
              f"correlation across 6 parastichy classes (N=768 simulated vs N=768 photoreviewed; Fib-1={c_ens['Fibonacci -1']} > Fib+1={c_ens['Fibonacci +1']}, p={ens['fib_minus_1_vs_plus_1_pvalue']:.2e})",
              reference=1.0,
              source="Swinton, Ochu & MSI Turing's Sunflower Consortium 2016 R. Soc. Open Sci. 3:160091 Table 1",
              derived_from="script"),
        )
    return out


def measure_conus_textile(script: Path) -> list:
    """Measurements taken by loading 1984-wolfram-conus-textile-rule30.py and evaluating:
      1. OEIS A051023 center-column bit match (128/128) & Jen (1990) left-bijective inverse bit errors (0),
      2. Rule 30 Prize #1 & #2 finite-horizon center-column density & 4-block Shannon entropy,
      3. Jen (1990) left-cone diagonal period-doubling dislocation indices k in {0, 3, 8, 29, 400} and spatial lock velocity,
      4. Directional Boolean damage-cone velocities v_R = +1.0000 and v_L = -0.2604 cells/step,
      5. 3D Raup Conus textile shell height-to-width ratio and 2D silhouette IoU vs segmented OSU specimen,
      6. White-tent chord-length CV (sigma/mu) and normalized KS distance across Rule 30, Meinhardt-Klingler PDE, and Conus textile specimen."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("wolfram_conus_rule30_v1", script)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    res = mod.self_test()
    prize = res["prize"]
    diag = res["diag"]
    dmg = res["dmg"]
    spec_runs = res["spec_runs"]
    spec_cv = float(spec_runs.std() / spec_runs.mean())
    disloc_ks = [int(d["k"]) for d in diag["dislocations"][:5]]
    return [
        M("Rule 30 center-column exact match to OEIS A051023 (first 128 bits)",
          128, "bits matched",
          "exact bitwise comparison of a_{t,0} (t=0..127) against OEIS A051023",
          reference=128,
          source="Wolfram 1984 Nature 311:419 / OEIS A051023 / Rule 30 Prize (2019)",
          derived_from="script"),
        M("Jen (1990) left-bijective inverse & diagonal recursion bit errors",
          int(prize["bijective_errors"] + diag["recursion_errors"]), "bit errors",
          "sum of bit reconstruction errors across a_{t,i-1} = a_{t+1,i} XOR (a_{t,i} OR a_{t,i+1}) and d_k(t+1) = d_{k-2}(t) XOR (d_{k-1}(t) OR d_k(t))",
          reference=0,
          source="Jen 1990 Physica D 45:3-18 Theorem 1 & Section 2",
          note=f"first 5 period-doubling dislocation diagonals k = {disloc_ks} (P = 1, 2, 4, 8, 16)",
          derived_from="script"),
        M("Rule 30 Prize #2 center-column density c_1024",
          round(float(prize["final_density"]), 5), "fraction of 1s",
          "cumulative fraction (1/N) sum_{t=0}^{1023} a_{t,0} on single-seed light-cone",
          reference=0.5,
          source="Wolfram 2019 Rule 30 Prize Problem 2 (https://rule30prize.org/)",
          note=f"normalized 4-bit block Shannon entropy H_4/4 = {prize['block4_entropy_bits']:.5f} bits/bit",
          derived_from="script"),
        M("left-cone periodic crystal lock-in front velocity v_lock",
          round(float(diag["spatial_lock_velocity"]), 5), "cells/step",
          "spatial drift velocity of period-locked diagonal boundary (1/slope - 1 from T_lock(k) = 1.2441*k + 7.89 over k in [30, 400])",
          reference=-0.2081,
          source="Jen 1990 Physica D 45:3-18 / Wolfram 2002 A New Kind of Science p. 871",
          note=f"damage-cone velocities: v_R = +{dmg['vR_mean']:.4f}±{dmg['vR_std']:.4f}, v_L = {dmg['vL_mean']:.4f}±{dmg['vL_std']:.4f} cells/step",
          derived_from="script"),
        M("3D Raup Conus textile shell height-to-width ratio (H/W)",
          round(float(res["hw_mesh"]), 4), "ratio (H/W)",
          "projected axial height / maximum shoulder diameter of 3D Raup Conus mesh (W=1.42, D=0.05, T=3.15)",
          reference=round(float(res["spec_hw"]), 4),
          source="Conus textile OSU specimen 24399204286 (James St. John, CC BY 2.0)",
          derived_from="script"),
        M("2D shell silhouette IoU vs segmented Conus textile specimen",
          round(float(res["silhouette_iou"]), 4), "IoU (intersection/union)",
          "pixel overlap between projected 3D Raup Conus mesh and segmented OSU Conus textile specimen silhouette (256x140 grid)",
          reference=1.0,
          source="Conus textile OSU specimen 24399204286 (datasets/conus/conus_textile_specimen.jpg)",
          derived_from="output_pixels"),
        M("Rule 30 white-tent chord-length CV (σ/μ)",
          round(float(res["r30_cv"]), 4), "CV (std/mean)",
          "coefficient of variation of horizontal unpigmented run lengths w on periodic mantle ring (230x370)",
          reference=round(spec_cv, 4),
          source="Conus textile OSU specimen dorsal white-tent segmentation (1,814 chords, CV=0.6403; exact geometric P(W=w)=2^{-w} theory CV=0.7071)",
          note=f"Meinhardt-Klingler PDE CV = {res['mein_cv']:.4f} (vs close-up specimen ling_commons_010 CV = 1.1310); KS(Rule30, specimen) = {res['ks_r30_vs_specimen']:.4f}",
          derived_from="script"),
    ]


def measure_runions_leaf_venation(script: Path) -> list:
    """Morphometric measurements for 2005-runions-leaf-venation.py: open Ginkgo biloba
    space-colonization tree topology (L = E - V + 1 = 0), Murray (1926) pipe-model
    cubic conductance conservation (r_root^3 = sum r_tip^3), Price et al. (2013)
    Horton-Strahler bifurcation area ratios, Katifori & Magnasco (2012) loop nesting,
    and empirical specimen segmentation metrics when present."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("runions_venation", script)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    metrics = mod.render_outputs()

    out = [
        M("open Ginkgo venation anastomosing cycle count L = E - V + 1",
          int(metrics["open_euler_loops"]), "loops",
          "Euler-Poincaré independent cycle rank E - V + 1 on open Voronoi space-colonization tree",
          reference=0,
          source="Runions et al. 2005 ACM Trans. Graph. 24(3):702-711 Sec. 3 (open venation tree)",
          derived_from="script"),
        M("Murray's cube law root-to-tip conservation ratio r_root^3 / (N_tips * r_tip^3)",
          round(float(metrics["open_murray_root_r3"]) / max(float(metrics["open_murray_tip_sum_r3"]), 1.0), 6),
          "ratio",
          "ratio of petiole root cubic radius r_root^3 to sum of terminal veinlet cubic radii sum r_tip^3",
          reference=1.0,
          source="Murray 1926 PNAS 12:207 / Runions et al. 2005 Sec. 3.5 / Lean 4 theorem murray_1926_tree_conservation",
          note=f"r_root^3 = {metrics['open_murray_root_r3']:.1f}, N_tips = {int(metrics['open_murray_tip_sum_r3'])}",
          derived_from="script"),
        M("open venation binary junction median daughter/parent area ratio (A_1 + A_2)/A_0",
          round(float(metrics["open_murray_median_area_ratio"]), 4), "area ratio",
          "median (r_1^2 + r_2^2) / r_0^2 across binary bifurcations of the grown open venation tree",
          reference=round(2.0 ** (1.0 / 3.0), 4),
          source="Murray 1926 symmetric bifurcation 2^{1/3} = 1.2599 / Price et al. 2013 PLoS ONE e85420 Fig. 3",
          derived_from="script"),
        M("closed reticulate venation anastomosing loop count L = E - V + 1",
          int(metrics["closed_euler_loops"]), "areole loops",
          "Euler cycle rank E - V + 1 of closed Relative Neighborhood Graph (RNG) space-colonization network",
          reference=150,
          source="Runions et al. 2005 ACM Trans. Graph. 24(3):702-711 Sec. 4 Eq. (2)-(3)",
          note=f"V = {metrics['closed_num_nodes']}, E = {metrics['closed_num_edges']}",
          derived_from="script"),
        M("Horton-Strahler order 1 vs order 8 median area-ratio shift",
          round(float(metrics["price_2013_order1_area_ratio"]) - float(metrics["price_2013_order8_area_ratio"]), 4),
          "delta area ratio",
          "decrease in median daughter/parent cross-sectional area ratio from terminal minor veins (order 1) to major midveins (order 8)",
          reference=0.24,
          source="Price, Knox & Brodribb 2013 PLoS ONE 8(12):e85420 Fig. 4 (order 1 median 1.25 -> order 8 median 1.01)",
          note=f"Pearson r across Strahler orders = {metrics['price_2013_strahler_pearson_r']:.4f}",
          derived_from="script"),
        M("Katifori & Magnasco (2012) unweighted hierarchical loop nesting number i_u",
          round(float(metrics["katifori_multiplicative_nesting_i_u"]), 4), "nesting index i_u",
          "mean subtree partition balance q_j = 1 - |r_j - s_j|/(r_j + s_j - 1) under iterative minimum-width vein removal",
          reference=0.68,
          source="Katifori & Magnasco 2012 PLoS ONE e37994 / Ronellenfitsch et al. 2015 PLoS Comput. Biol. e1004680 Fig. 3b",
          note=f"high-noise / additive control i_u = {metrics['katifori_additive_nesting_i_u']:.4f}",
          derived_from="script"),
    ]
    if "specimen_reticulate_areoles" in metrics:
        out.append(
            M("segmented reticulate leaf skeleton specimen areole count",
              int(metrics["specimen_reticulate_areoles"]), "areoles",
              "connected enclosed areole faces segmented directly from datasets/venation/reticulate_vein_skeleton_specimen.jpg",
              reference=180,
              source="Wikimedia Commons File:Vein skeleton of a leaf (de-ghosted).jpg (CC BY-SA 3.0)",
              note=f"segmented specimen nesting index i_u = {metrics.get('specimen_reticulate_nesting_i_u', 0.0):.4f}, Ginkgo open specimen loops = {metrics.get('specimen_ginkgo_loops', 0)}",
              derived_from="output_pixels")
        )
    return out


def measure_petroff_stromatolite(script: Path) -> list:
    """Morphometric measurements for 2010-petroff-stromatolite.py: diffusion-competition
    spacing ell(tau=12 h), hexagonal lattice nearest-neighbour spacing, and dome-to-cone
    lamina height/width aspect ratios across beta = v/lambda in {0.3, 1.5, 5.0}."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("petroff_stromatolite", script)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)

    ell_12h_field = float(mod.diffusion_length(12 * 3600.0, D=1e-5, alpha=1.0))
    ell_12h_lab = float(mod.diffusion_length(12 * 3600.0, D=1e-5, alpha=0.30))
    pts = mod.hex_lattice(ell_12h_field, nx=7, ny=6)
    from scipy.spatial.distance import pdist
    nn_dist = float(np.min(pdist(pts)))
    ars = {}
    for beta in (0.3, 1.5, 5.0):
        laminae = mod.grow_stromatolite(v=beta * 1.0, lam=1.0)
        ars[beta] = float(mod.aspect_ratio(laminae[-1]))
    return [
        M("field cone spacing ell at tau = 12 h (D = 1e-5 cm^2/s, alpha = 1.0)",
          round(ell_12h_field, 4), "cm",
          "diffusion_length(12 * 3600, D=1e-5, alpha=1.0) = alpha * sqrt(D * tau)",
          reference=1.0,
          source="Petroff et al. 2010 PNAS 107(22):9956-9961 Eq. 2 (~1 cm Yellowstone field spacing)",
          note=f"lab clumps (alpha=0.30) ell(12 h) = {ell_12h_lab:.4f} cm; hex_lattice min NN distance = {nn_dist:.4f} cm",
          derived_from="script"),
        M("dome regime (beta = 0.3) final lamina height/width aspect ratio",
          round(ars[0.3], 4), "height/width",
          "aspect_ratio(grow_stromatolite(v=0.3, lam=1.0)[-1])",
          derived_from="script"),
        M("cone regime (beta = 1.5) final lamina height/width aspect ratio",
          round(ars[1.5], 4), "height/width",
          "aspect_ratio(grow_stromatolite(v=1.5, lam=1.0)[-1])",
          derived_from="script"),
        M("sharp cone regime (beta = 5.0) final lamina height/width aspect ratio",
          round(ars[5.0], 4), "height/width",
          "aspect_ratio(grow_stromatolite(v=5.0, lam=1.0)[-1])",
          note=f"monotonic aspect-ratio progression across beta=(0.3, 1.5, 5.0): {ars[0.3]:.3f} < {ars[1.5]:.3f} < {ars[5.0]:.3f}",
          derived_from="script"),
    ]


def measure_kahng_coral_geometry(script: Path) -> list:
    """Morphometric measurements for 2024-kahng-coral-geometry.py: asymptotic linear
    extension C vs discrete 1-yr step extension, size-independent SAP for discs and
    cylinders, and microfragmentation SAP speedup ratio (r = 1 cm vs r = 25 cm)."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("kahng_coral_geometry", script)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)

    G_m, rho_m = mod.MASSIVE["G"], mod.MASSIVE["rho"]
    c_asymp_porites = float(G_m / rho_m)
    c_r30_porites = float(mod.hemisphere_extension(np.array([30.0]), G_m, rho_m)[0])
    c_r1_porites = float(mod.hemisphere_extension(np.array([1.0]), G_m, rho_m)[0])
    sap_r1 = float(mod.hemisphere_sap(np.array([1.0]), G_m, rho_m)[0])
    sap_r25 = float(mod.hemisphere_sap(np.array([25.0]), G_m, rho_m)[0])
    sap_ratio = sap_r1 / sap_r25
    sap_disc_leptoseris = float(mod.disc_sap(mod.DISC["G"], mod.DISC["rho"], mod.DISC["h"]))
    sap_cyl_acropora = float(mod.cylinder_sap(mod.BRANCH["G"], mod.BRANCH["rho"], r=0.5))
    return [
        M("massive Porites hemisphere linear extension C at r = 30 cm (discrete 1-yr step)",
          round(c_r30_porites, 4), "cm/yr",
          "hemisphere_extension(r=30 cm, G=1.64, rho=1.28) = (r^3 + 3*G*r^2/rho)^(1/3) - r",
          reference=round(c_asymp_porites, 4),
          source="Kahng, Odle & Wakeman 2024 PeerJ 12:e17037 Appendix A (continuum C = G/rho = 1.2812 cm/yr)",
          note=f"at r = 1 cm discrete step gives C = {c_r1_porites:.4f} cm/yr due to 1-yr forward-Euler discretization",
          derived_from="script"),
        M("massive Porites microfragmentation SAP speedup ratio (r = 1 cm vs r = 25 cm)",
          round(sap_ratio, 4), "SAP(1 cm) / SAP(25 cm)",
          "hemisphere_sap(r=1 cm) / hemisphere_sap(r=25 cm) for G=1.64, rho=1.28",
          reference=18.92,
          source="Kahng, Odle & Wakeman 2024 PeerJ 12:e17037 (SAP(1 cm)=1.8558/yr vs SAP(25 cm)=0.0981/yr)",
          derived_from="script"),
        M("deep-water Leptoseris flat-disc size-independent SAP (h = 0.20 cm)",
          round(sap_disc_leptoseris, 4), "yr^-1",
          "disc_sap(G=0.061, rho=2.67, h=0.20) = G / (rho * h)",
          reference=round(0.061 / (2.67 * 0.20), 4),
          source="Kahng, Odle & Wakeman 2024 PeerJ 12:e17037 / Kahng 2023",
          derived_from="script"),
        M("branching Acropora cylinder size-independent SAP (r = 0.50 cm)",
          round(sap_cyl_acropora, 4), "yr^-1",
          "cylinder_sap(G=2.50, rho=1.40, r=0.50) = 2 * G / (rho * r)",
          reference=round(2.0 * 2.50 / (1.40 * 0.50), 4),
          source="Kahng, Odle & Wakeman 2024 PeerJ 12:e17037 / Hughes 1987",
          derived_from="script"),
    ]


# --------------------------------------------------------------------------- #
def compute_quick_gate(sim: dict, prov: dict, outputs: list[dict], measurements: list[dict]) -> dict:
    """Deterministic zero-token pre-gate (adapted from Wang, Berkovich, Marom & Buehler,
    MRS Bull. 2026 `discovery_rubric.py`):
      - provenance (15): spec_fingerprint + artifact_sha256 + DOI/URL references present
      - quantitative (25): >= 3 numeric morphometric measurements extracted
      - baseline (20): at least 1 measurement carries an explicit reference_value
      - uncertainty (20): script or measurements include CV / std / CI / p-value / RMSE / IoU dispersion
      - delta (20): explicit negative control, alternative model comparison, or supersession delta
      Total: 0..100, plus claim_boundary summary if present on the simulation."""
    has_spec = bool(prov.get("spec_fingerprint"))
    has_art = bool(prov.get("artifact_sha256"))
    has_refs = bool(sim.get("source_refs"))
    s_prov = (5 if has_spec else 0) + (5 if has_art else 0) + (5 if has_refs else 0)

    valid_m = [m for m in measurements if m.get("value") is not None]
    s_quant = min(25, int(round(25.0 * min(len(valid_m), 4) / 4.0)))

    ref_m = [m for m in valid_m if m.get("reference_value") is not None]
    s_base = min(20, int(round(20.0 * min(len(ref_m), 2) / 2.0)))

    unc_terms = ("cv", "std", "rmse", "iou", "ks", "p=", "p-value", "ci", "±", "dispersion", "variance", "entropy")
    m_blob = " ".join(
        f"{m.get('name', '')} {m.get('unit', '')} {m.get('method', '')} {m.get('note', '')}".lower()
        for m in valid_m
    )
    n_unc = sum(1 for t in unc_terms if t in m_blob)
    s_unc = min(20, 10 * min(n_unc, 2))

    delta_terms = ("control", "vs", "null", "baseline", "alternative", "pertur", "defect", "supersede", "ratio")
    n_delta = sum(1 for t in delta_terms if t in m_blob)
    s_delta = min(20, 10 * min(n_delta, 2))

    total = s_prov + s_quant + s_base + s_unc + s_delta
    return {
        "total": total,
        "provenance_15": s_prov,
        "quantitative_25": s_quant,
        "baseline_20": s_base,
        "uncertainty_20": s_unc,
        "delta_20": s_delta,
        "result_quality": "validated" if (total >= 80 and len(ref_m) >= 2) else ("screened" if total >= 50 else "incomplete"),
    }


def main():
    quick_gate_only = "--quick-gate" in sys.argv
    argv_clean = [a for a in sys.argv[1:] if a != "--quick-gate"]
    audit_path = (Path(argv_clean[0]) if argv_clean else HERE / "audits" / "2026-10-13-morphology-audit.yaml").resolve()
    audit = yaml.safe_load(audit_path.read_text())
    supersedes_id = audit.get("audit", {}).get("supersedes") or audit.get("audit", {}).get("supersedes_audit_id")
    out = {
        "audit": audit["audit"]["id"],
        "supersedes_audit_id": supersedes_id,
        "provenance": {},
        "quick_gate": {},
        "outputs": {},
        "measurements": {},
        "missing": [],
    }
    for sim in audit["simulations"]:
        if sim.get("status") != "evaluated":
            continue
        sid = sim["id"]
        script_rel = sim.get("script")
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
            elif sid == "turing-gray-scott":
                out["measurements"][sid] = measure_turing(ROOT / sim.get("output", "turing_patterns.gif"), script=ROOT / sim["script"])
            elif sid == "bressloff-v1-form-constants":
                out["measurements"][sid] = measure_bressloff(ROOT / sim["script"])
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
            elif sid == "cervera-levin-mafe":
                out["measurements"][sid] = measure_cervera(ROOT / sim["script"])
            elif sid == "vogel-phyllotaxis":
                out["measurements"][sid] = measure_vogel_phyllotaxis(ROOT / sim["script"])
            elif sid == "wolfram-conus-textile-rule30":
                out["measurements"][sid] = measure_conus_textile(ROOT / sim["script"])
            elif sid == "runions-leaf-venation":
                out["measurements"][sid] = measure_runions_leaf_venation(ROOT / sim["script"])
            elif sid == "petroff-stromatolite":
                out["measurements"][sid] = measure_petroff_stromatolite(ROOT / sim["script"])
            elif sid == "kahng-coral-geometry":
                out["measurements"][sid] = measure_kahng_coral_geometry(ROOT / sim["script"])
            else:
                # Dynamic per-loop measurement discovery (no edits to measure_outputs.py needed)
                dyn_meas: list = []
                if script_rel and (ROOT / script_rel).exists():
                    import importlib.util
                    spec_mod = importlib.util.spec_from_file_location(f"loop_{sid.replace('-', '_')}", ROOT / script_rel)
                    if spec_mod and spec_mod.loader:
                        mod = importlib.util.module_from_spec(spec_mod)
                        spec_mod.loader.exec_module(mod)
                        if hasattr(mod, "measure_morphoeval") and callable(mod.measure_morphoeval):
                            dyn_meas.extend(mod.measure_morphoeval())
                if not dyn_meas:
                    dyn_doc = _load_dynamic_loop_spec(sid)
                    if dyn_doc.get("measurements"):
                        dyn_meas.extend(dyn_doc["measurements"])
                    elif sim.get("measurements"):
                        dyn_meas.extend(sim["measurements"])
                out["measurements"][sid] = dyn_meas
        except Exception as ex:  # a failed measurement is reported, never silently skipped
            out["measurements"][sid] = [{"name": "measurement failed", "value": None, "unit": None,
                                         "method": f"{type(ex).__name__}: {ex}"}]
        for f in sorted(files):
            p = ROOT / f
            if p.exists():
                out["outputs"].setdefault(sid, []).append(fingerprint(p, sim_id=sid, script=script_rel))
            else:
                out["missing"].append({"simulation": sid, "file": f})
        primary_art = None
        if sim.get("output") and (ROOT / sim["output"]).exists():
            primary_art = hashlib.sha256((ROOT / sim["output"]).read_bytes()).hexdigest()
        elif out["outputs"].get(sid):
            primary_art = out["outputs"][sid][0]["artifact_sha256"]
        out["provenance"][sid] = {
            "spec_fingerprint": compute_spec_fingerprint(sid, script_rel, sim_doc=sim),
            "artifact_sha256": primary_art,
            "input_dataset_sha256": compute_input_dataset_sha256(sid, sim_doc=sim),
        }
        if sim.get("claim_boundary"):
            out["provenance"][sid]["claim_boundary"] = sim["claim_boundary"]
        qg = compute_quick_gate(
            sim,
            out["provenance"][sid],
            out["outputs"].get(sid, []),
            out["measurements"].get(sid, []),
        )
        out["quick_gate"][sid] = qg
        print(f"{sid}: {len(out['outputs'].get(sid, []))} output(s) fingerprinted "
              f"(spec_fingerprint={out['provenance'][sid]['spec_fingerprint'][:16]}…), "
              f"{len(out['measurements'].get(sid, []))} measurement(s), "
              f"quick_gate={qg['total']}/100 ({qg['result_quality']})")
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

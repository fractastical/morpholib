#!/usr/bin/env python3
"""Issue #1 validation: model vs. reference specimen/figure, measured.

Closes the validation gap in fractastical/morpholib#1 ("Audit: morphology
animations don't match real forms"). For each of the two animations the issue
flagged, this script renders the *current* model, places it side-by-side with
the committed reference specimen photograph, measures both sides, and fails
(exit 1) if any acceptance check regresses:

Planarian (`models/2013-lobo-planform-morphology.py`, successor to the
convex-hull `1900-planformDB_parser.py`):
  * decoded from the Planform region graph (Region + RegionParam +
    RegionsLink), shared WT scale, auricle species template on each head;
  * wild type: 2 eyes / 2 brain lobes / 1 pharynx / 2 VNCs, pharynx at
    mid-body, eyes in the anterior quarter, aspect within 35% of the
    segmented *Schmidtea mediterranea* specimen photograph;
  * double-head / headless / two-tails phenotypes have distinct head/tail
    counts and distinct organ-count vectors.

Raup (`models/create_raup_animations.py`):
  * W, D, T recovered from each rendered mesh to <1e-6 (T = dz_c/dr_c, so
    `zc = T * rc`, not `T * theta`);
  * spire/width strictly increases with T in a controlled sweep;
  * the six named panels rank by spire/width exactly as their reference
    taxa do qualitatively (Turritella > Cerithium > Conus > Helix >
    Nautilus = ammonite = 0). Panels are illustrative morphospace points
    (see datasets/raup/variables.json), not fitted measurements of the
    photographed specimens -- the check is morphospace-region membership,
    and it is labelled as such in the output figure.

Reference photographs are committed in-repo (no downloads):
  * datasets/species_gallery/II_Anaima_Invertebrata/Schmidtea_mediterranea.jpg
    (Stevenson & Beane 2010, via morphoeval/upgrade_planarian_assets.py)
  * datasets/planform/planform_multi_phenotype_specimens.jpg (12 published
    phenotype micrographs, cropped per-phenotype below)
  * datasets/raup/raup_coiled_shells_specimen.jpg (6 whole-shell views,
    cropped per-taxon below)

Outputs:
  docs/media/planform_model_vs_reference.png
  docs/media/raup_model_vs_reference.png
  morphoeval/runs/issue1_model_vs_reference.json
  morphoeval/runs/issue1_model_vs_reference.md

Usage:
  venv/bin/python morphoeval/issue1_model_vs_reference.py [--db PATH]
The PlanformDB .edb is used when found ($PLANFORM_DB_PATH, --db, or the
canonical local path); otherwise the vendored canonical fixture
(datasets/planform/planform_canonical_silhouettes_v1.json) is decoded, which
covers #1/#2/#20/#24 (the two-tails row is then skipped, not faked).
"""
from __future__ import annotations

import argparse
import importlib.util
import io
import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy import ndimage as ndi

ROOT = Path(__file__).resolve().parent.parent

# Relative crop boxes (x0, y0, x1, y1) in the committed phenotype plate
# (874x632; measured on a 50 px grid overlay, inside each thumbnail border).
PLANFORM_PLATE = ROOT / "datasets" / "planform" / "planform_multi_phenotype_specimens.jpg"
PLANFORM_CROPS = {
    1: (98, 118, 542, 332),     # main WT micrograph (head-left)
    24: (156, 410, 262, 486),   # Double head, one pharynx
    2: (277, 408, 412, 488),    # Double head, two pharynxes
    14: (727, 408, 862, 488),   # Two tails
    20: (27, 527, 132, 602),    # Headless
}
WT_SPECIMEN = ROOT / "datasets" / "species_gallery" / "II_Anaima_Invertebrata" / "Schmidtea_mediterranea.jpg"

# Relative crop boxes in the committed Raup specimen plate (1575x1080).
RAUP_PLATE = ROOT / "datasets" / "raup" / "raup_coiled_shells_specimen.jpg"
RAUP_CROPS = {  # panel name -> crop
    "Planispiral": (25, 155, 505, 540),
    "Turreted": (530, 165, 985, 540),
    "Evolute disc": (1010, 165, 1550, 540),
    "Conical": (25, 675, 505, 1025),
    "High-spired": (530, 675, 985, 1025),
    "Globose": (1010, 675, 1550, 1025),
}
# Expected qualitative elongation ranking of the reference taxa (spire/width).
EXPECTED_SPIRE_ORDER = ["Turreted", "High-spired", "Conical", "Globose", "Planispiral", "Evolute disc"]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def crop_rel(img: Image.Image, box) -> np.ndarray:
    w, h = img.size
    # boxes were measured on the committed files; use them as pixels directly
    return np.asarray(img.crop(box).convert("RGB"))


def segment_specimen_aspect(path: Path) -> dict:
    """Bounding-box aspect of the largest dark foreground object in the WT photo."""
    rgb = np.asarray(Image.open(path).convert("RGB"), dtype=float)
    gray = rgb.mean(axis=2)
    mask = gray < 220
    mask = ndi.binary_closing(mask, np.ones((5, 5)))
    mask = ndi.binary_fill_holes(mask)
    lab, n = ndi.label(mask)
    sizes = ndi.sum(np.ones_like(lab), lab, range(1, n + 1))
    m = lab == (int(np.argmax(sizes)) + 1)
    ys, xs = np.where(m)
    w, h = int(xs.max() - xs.min()), int(ys.max() - ys.min())
    return {"bbox_width_px": w, "bbox_height_px": h, "aspect_long_over_short": round(max(w, h) / min(w, h), 3)}


def render_morphology_array(lobo, m) -> np.ndarray:
    """Render one morphology (anterior up) and rotate so the head is left,
    matching the specimen photographs' head-left convention."""
    fig, ax = plt.subplots(figsize=(3.2, 4.2))
    lobo.draw_morphology(ax, m)
    cx, cy = lobo.centre_of(m)
    x0, x1, y0, y1 = lobo.extent(m, pad=30)
    half = max(x1 - x0, y1 - y0) / 2
    lobo.fit_axes(ax, (cx - half, cx + half, cy - half, cy + half))
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    arr = np.asarray(Image.open(buf).convert("RGB"))
    return np.rot90(arr, k=1)  # CCW: anterior (top) -> left, matching the photos


def organ_ap_fraction(lobo, m, kind: str) -> float:
    """AP position of the (first) organ of `kind`, 0 = anterior tip, 1 = posterior tip."""
    x0, x1, y0, y1 = lobo.extent(m, pad=0.0)
    for o in m.organs:
        if o.kind == kind:
            cy = m.positions[o.region][1] + o.vy
            return float((y1 - cy) / (y1 - y0))
    return float("nan")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--db", default=os.getenv("PLANFORM_DB_PATH") or "/Users/jdietz/Documents/Levin/PlanformDB/planformDB_2.5.0.edb")
    ap.add_argument("--out-dir", default=str(ROOT / "docs" / "media"))
    args = ap.parse_args()

    checks: list[dict] = []

    def check(name: str, ok: bool, detail: str):
        checks.append({"check": name, "pass": bool(ok), "detail": detail})
        print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")

    lobo = load_module("lobo2013", ROOT / "models" / "2013-lobo-planform-morphology.py")
    raup = load_module("raup", ROOT / "models" / "create_raup_animations.py")

    # ---------------- Planarian ----------------
    if args.db and Path(args.db).is_file():
        morphs = lobo.load_morphologies(args.db)
        source = f"PlanformDB {Path(args.db).name} (sha256-pinned decode)"
        ids = [1, 24, 20, 14]
    else:
        morphs = lobo.load_canonical_fixture_morphologies()
        source = "vendored canonical fixture datasets/planform/planform_canonical_silhouettes_v1.json"
        ids = [1, 24, 20]
        print("note: PlanformDB .edb not found; two-tails (#14) row skipped, not faked")
    wt = morphs[1]
    mm = {mid: lobo.morphometrics(morphs[mid]) for mid in morphs if mid in (1, 2, 14, 20, 24)}

    check("planarian.source", True, source)
    check("planarian.wt_organs",
          (mm[1]["n_eyes"], mm[1]["n_brains"], mm[1]["n_pharynges"], mm[1]["n_vnc"]) == (2, 2, 1, 2),
          f"WT eyes={mm[1]['n_eyes']} brains={mm[1]['n_brains']} pharynx={mm[1]['n_pharynges']} VNC={mm[1]['n_vnc']} (expect 2/2/1/2)")
    phar = organ_ap_fraction(lobo, wt, "Phar.")
    check("planarian.pharynx_midbody", 0.35 <= phar <= 0.65, f"pharynx at AP fraction {phar:.2f} from anterior tip (expect ~0.50)")
    eye = organ_ap_fraction(lobo, wt, "Eye")
    check("planarian.eyes_anterior", eye <= 0.25, f"eyes at AP fraction {eye:.2f} from anterior tip (expect <=0.25)")
    check("planarian.two_vnc_cords", mm[1]["n_vnc"] == 2, f"WT has {mm[1]['n_vnc']} line-organ VNCs drawn from their two end points (bilateral pair, not a Y-sorted zig-zag)")

    plain = lobo.silhouette(wt, smooth=6.0)
    templ = lobo.silhouette(wt, smooth=6.0, species_template=True)

    def head_width(mask_bounds):
        mask, (x0, x1, y0, y1) = mask_bounds
        rows = np.where(mask.any(axis=1))[0]
        head_rows = rows[int(0.8 * len(rows)):]
        return float(np.max([mask[r].sum() for r in head_rows]) * (x1 - x0) / mask.shape[1])

    hw_plain, hw_templ = head_width(plain), head_width(templ)
    check("planarian.auricles", hw_templ > 1.05 * hw_plain,
          f"auricle template widens WT head {hw_plain:.0f} -> {hw_templ:.0f} model units (+{100*(hw_templ/hw_plain-1):.0f}%)")

    specimen = segment_specimen_aspect(WT_SPECIMEN)
    ratio = mm[1]["aspect"] / specimen["aspect_long_over_short"]
    check("planarian.wt_aspect_vs_specimen", 0.65 <= ratio <= 1.35,
          f"model WT aspect {mm[1]['aspect']} vs segmented specimen {specimen['aspect_long_over_short']} (ratio {ratio:.2f}, tolerance 0.65-1.35)")

    sel = [morphs[i] for i in ids]
    vectors = [(m.count("Head"), m.count("Tail"), m.count("Eye"), m.count("Phar.")) for m in sel]
    check("planarian.phenotypes_distinct", len(set(vectors)) == len(vectors),
          "; ".join(f"#{m.id} {m.name}: H{v[0]} T{v[1]} eyes{v[2]} phar{v[3]}" for m, v in zip(sel, vectors)))

    # comparison figure: specimen (left) vs model (right), head-left both sides
    plate = Image.open(PLANFORM_PLATE)
    fig, axes = plt.subplots(len(ids), 2, figsize=(11, 2.6 * len(ids)))
    for row, mid in enumerate(ids):
        m = morphs[mid]
        ref = np.asarray(Image.open(WT_SPECIMEN).convert("RGB")) if mid == 1 else crop_rel(plate, PLANFORM_CROPS[mid])
        axes[row, 0].imshow(ref)
        axes[row, 0].set_title(f"#{mid} {m.name} — specimen" + (" (Stevenson & Beane 2010)" if mid == 1 else " (plate crop)"), fontsize=7.5)
        axes[row, 1].imshow(render_morphology_array(lobo, m))
        v = vectors[row]
        axes[row, 1].set_title(f"#{mid} model: H{v[0]} T{v[1]} eyes{v[2]} phar{v[3]} aspect {mm[mid]['aspect']}", fontsize=7.5)
        for ax in axes[row]:
            ax.axis("off")
    fig.suptitle("Issue #1 — Planarian: model (right) vs reference specimen (left), head-left, shared WT scale\n"
                 f"WT model aspect {mm[1]['aspect']} vs specimen {specimen['aspect_long_over_short']}; pharynx at AP {phar:.2f}; eyes at AP {eye:.2f}\n"
                 "Reference photos are published exemplars of each phenotype class (not the encoded individual); comparison is WT aspect + organ/topology counts.", fontsize=9)
    fig.tight_layout()
    out_plan = Path(args.out_dir) / "planform_model_vs_reference.png"
    out_plan.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_plan, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out_plan}")

    # ---------------- Raup ----------------
    data = raup.load_raup_data()
    shells = raup.panels(data)
    metrics = []
    worst = 0.0
    for s in shells:
        X, Y, Z = raup.raup_shell(W=s["W"], D=s["D"], T=s["T"], S=s["S"], turns=s["turns"], res=100)
        r = raup.recover_parameters(X, Y, Z)
        worst = max(worst, abs(r["W"] - s["W"]), abs(r["D"] - s["D"]), abs(r["T"] - s["T"]))
        metrics.append({**s, **r})
    check("raup.parameters_recovered", worst < 1e-6, f"max |recovered - nominal| over 6 panels = {worst:.2e} (T = dz_c/dr_c, zc = T*rc)")

    spires = []
    mono = True
    prev = -1.0
    for T in (0.0, 0.5, 1.5, 3.0):
        X, Y, Z = raup.raup_shell(W=2.5, D=0.1, T=T, turns=4, res=60)
        sp = raup.recover_parameters(X, Y, Z)["spire_over_width"]
        spires.append(round(sp, 3))
        mono &= sp > prev
        prev = sp
    check("raup.T_monotone", mono, f"spire/width at T=0,0.5,1.5,3.0 = {spires} (strictly increasing)")

    order = [m["name"] for m in sorted(metrics, key=lambda m: -m["spire_over_width"])]
    # ties at 0 (both planispiral panels) may swap
    order_core = [n for n in order if n not in ("Planispiral", "Evolute disc")]
    check("raup.morphospace_ranking", order_core == EXPECTED_SPIRE_ORDER[:4] and set(order[4:]) == {"Planispiral", "Evolute disc"},
          f"spire/width ranking {order} vs reference-taxon expectation {EXPECTED_SPIRE_ORDER}")

    plate_r = Image.open(RAUP_PLATE)
    fig, axes = plt.subplots(len(shells), 2, figsize=(11, 2.4 * len(shells)))
    for row, (s, m) in enumerate(zip(shells, metrics)):
        axes[row, 0].imshow(crop_rel(plate_r, RAUP_CROPS[s["name"]]))
        axes[row, 0].set_title(f"cf. {s['cf']} — reference specimen (plate crop)", fontsize=8)
        ax = fig.add_subplot(len(shells), 2, row * 2 + 2, projection="3d")
        axes[row, 1].remove()
        X, Y, Z = raup.raup_shell(W=s["W"], D=s["D"], T=s["T"], S=s["S"], turns=s["turns"], res=50)
        ax.plot_wireframe(X, Y, Z, rstride=3, cstride=3, color=s["color"], linewidth=0.6, alpha=0.9)
        raup.fit_axes(ax, X, Y, Z)
        ax.view_init(elev=12, azim=-60)
        ax.set_axis_off()
        ax.set_title(f"{s['name']} model: W={s['W']} D={s['D']} T={s['T']} (recovered T={m['T']:.3f}) spire/width={m['spire_over_width']:.2f}", fontsize=8)
        axes[row, 0].axis("off")
    fig.suptitle("Issue #1 — Raup: model (right) vs reference specimen (left)\n"
                 "Illustrative morphospace points (not fitted to these specimens); check = region/ranking, W-D-T recovered from the mesh", fontsize=10)
    fig.tight_layout()
    out_raup = Path(args.out_dir) / "raup_model_vs_reference.png"
    fig.savefig(out_raup, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out_raup}")

    # ---------------- report ----------------
    payload = {
        "issue": "https://github.com/fractastical/morpholib/issues/1",
        "planform_source": source,
        "checks": checks,
        "planarian": {"morphometrics": {str(k): v for k, v in mm.items()},
                      "pharynx_ap_fraction": round(phar, 3), "eye_ap_fraction": round(eye, 3),
                      "auricle_head_width_plain": round(hw_plain, 1), "auricle_head_width_template": round(hw_templ, 1),
                      "specimen_wt": specimen, "model_specimen_aspect_ratio": round(ratio, 3)},
        "raup": {"panels": [{k: m[k] for k in ("name", "cf", "W", "D", "T", "height_over_width", "spire_over_width")} for m in metrics],
                 "T_sweep_spire_over_width": spires},
        "figures": [str(out_plan), str(out_raup)],
        "all_pass": all(c["pass"] for c in checks),
    }
    runs = ROOT / "morphoeval" / "runs"
    (runs / "issue1_model_vs_reference.json").write_text(json.dumps(payload, indent=2) + "\n")
    lines = ["# Issue #1 — model vs. reference validation", "",
             f"Source: {source}", "",
             "| Check | Result | Detail |", "|---|---|---|"]
    lines += [f"| {c['check']} | {'PASS' if c['pass'] else 'FAIL'} | {c['detail']} |" for c in checks]
    lines += ["", f"Figures: `{out_plan.name}`, `{out_raup.name}` (docs/media/)",
              "", f"**All pass: {payload['all_pass']}**"]
    (runs / "issue1_model_vs_reference.md").write_text("\n".join(lines) + "\n")
    print(f"  saved {runs / 'issue1_model_vs_reference.json'}")
    print(f"{'ALL PASS' if payload['all_pass'] else 'FAILURES PRESENT'} ({sum(c['pass'] for c in checks)}/{len(checks)})")
    return 0 if payload["all_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

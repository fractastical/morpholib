#!/usr/bin/env python3
"""
t1917-034_on-the-shapes-of-eggs-and-of-certain-other-hollo.py — Thompson (1917) claim T1917-034

Thompson, D.W. (1917) *On Growth and Form*, Ch. XV "On the Shapes of Eggs, and of Certain
Other Hollow Structures", pp. 652-669:

    "Birds' (and other hard-shelled) eggs are solids of revolution whose meridian curves form
     a definite, species-varying family of ovoids, comparable geometrically across species."

Pass 1 (v1) baseline.  The meridian family is the Preston (1953) four-parameter ovoid
    y(x) = (c0 + c1 x + c2 x^2 + c3 x^3) sqrt(1 - x^2),   -1 <= x <= 1  (half-length = 1),
as fitted to 1389 real eggs by Biggins, Thompson & Birkhead (2018, Ecol. Evol. 8:9728,
Dryad doi:10.5061/dryad.8kv2b20), with 181-species outline PCs from Deeming (2018, J. Avian
Biol. 49:jav-01528, Dryad doi:10.5061/dryad.q8rp3).  Formal theorems in
`morphoeval/lean/loops/Thompson1917_T1917_034.lean` (namespace MathOfBiology.Loop_228).

Outputs
    docs/media/t1917_034_on_the_shapes_of_eggs_and_of_c_comparison.png
    docs/media/t1917_034_on_the_shapes_of_eggs_and_of_c_growth.gif

Usage
    python3 thompson/claims/t1917-034_on-the-shapes-of-eggs-and-of-certain-other-hollo.py
    python3 thompson/claims/t1917-034_on-the-shapes-of-eggs-and-of-certain-other-hollo.py --self-test
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "datasets" / "thompson_1917" / "t1917-034"
MEDIA = ROOT / "docs" / "media"
STEM = "t1917_034_on_the_shapes_of_eggs_and_of_c"
REF_JSON = DATA / f"{STEM}_reference.json"


# --------------------------------------------------------------------------- #
# 1. Preston (1953) ovoid family — Thompson's "definite family" of meridian curves
#    (Lean: prestonPoly, poleRadius, prestonVolume315 in Thompson1917_T1917_034.lean)
# --------------------------------------------------------------------------- #
def preston_poly(c, x):
    c0, c1, c2, c3 = c
    return c0 + c1 * x + c2 * x ** 2 + c3 * x ** 3


def preston_half_breadth(c, x):
    """Meridian half-breadth y(x) of the Preston ovoid on the normalised axis x in [-1, 1]."""
    return preston_poly(c, x) * np.sqrt(np.clip(1.0 - x ** 2, 0.0, None))


def preston_indices(c, n=20001):
    """Biggins et al. (2018) shape indices: Elongation L/B, Pointedness, Polar Asymmetry.
    The polar asymmetry uses the closed form R_pole = p(+/-1)^2 (Lean Law 1)."""
    xs = np.linspace(-0.999, 0.999, n)
    y = preston_half_breadth(c, xs)
    i = int(np.argmax(y))
    x_star = xs[i]
    rp, rm = preston_poly(c, 1.0) ** 2, preston_poly(c, -1.0) ** 2
    return {
        "elongation": 1.0 / y[i],
        "pointedness": max((1.0 - x_star) / 2.0, (1.0 + x_star) / 2.0),
        "polar_asymmetry": max(rm / rp, rp / rm),
        "x_widest": x_star,
    }


def preston_volume_coefficient(c):
    """315 V / (4 pi (L/2)^3) in closed form (Lean Law 2: prestonVolume315)."""
    c0, c1, c2, c3 = c
    return 105 * c0 ** 2 + 21 * (c1 ** 2 + 2 * c0 * c2) + 9 * (c2 ** 2 + 2 * c1 * c3) + 5 * c3 ** 2


def preston_volume(c, length_mm):
    """Volume (cm^3) of the solid of revolution for an egg of length `length_mm`."""
    return np.pi * (length_mm / 20.0) ** 3 * preston_volume_coefficient(c) * 4.0 / 315.0


# --------------------------------------------------------------------------- #
# 2. Data loading (Biggins et al. 2018; Deeming 2018)
# --------------------------------------------------------------------------- #
def _read_csv(path, encoding="latin-1"):
    with open(path, newline="", encoding=encoding) as fh:
        return list(csv.DictReader(fh))


def _num(s):
    try:
        return float(s)
    except (TypeError, ValueError):
        return np.nan


def load_biggins_main():
    rows = _read_csv(DATA / "biggins2018_data_for_comparing_indices.csv")
    out = []
    for r in rows:
        out.append({
            "species": r["species"].replace(".", " ").capitalize(),
            "family": r["bird.family"],
            "c": np.array([float(r[k]) for k in ("c0", "c1", "c2", "c3")]),
            "elongation": float(r["Elongation"]),
            "pointedness": float(r["Pointedness"]),
            "polar_asymmetry": float(r["PolarAsymmetry"]),
            "mass": _num(r["mass"]),
            "adult_mass": _num(r["adult.body.mass"]),
        })
    return out


def load_biggins_range():
    rows = _read_csv(DATA / "biggins2018_range_of_egg_shapes.csv")
    return [{
        "egg": r["eggname"], "species": r["species"],
        "c": np.array([float(r[k]) for k in ("c0", "c1", "c2", "c3")]),
        "L_mm": float(r["maxlength"]), "B_mm": float(r["maxbreadth"]),
        "elongation": float(r["Elongation"]), "pointedness": float(r["Pointedness"]),
        "polar_asymmetry": float(r["PolarAsymmetry"]),
    } for r in rows]


def load_deeming():
    rows = _read_csv(DATA / "deeming2018_jab_egg_shape_composition.csv", encoding="utf-8")
    return [{
        "species": r["Species"], "order": r["Order"],
        "female_mass": _num(r["Female body mass (g)"]), "egg_mass": _num(r["Initial egg mass (g)"]),
        "pc1": _num(r["PC1"]), "pc2": _num(r["PC2"]),
    } for r in rows if r.get("Species") and _num(r["PC1"]) == _num(r["PC1"])]


def species_means(rows):
    out = {}
    for r in rows:
        out.setdefault(r["species"], []).append(r)
    table = []
    for sp, rs in out.items():
        e = np.array([r["elongation"] for r in rs])
        table.append({"species": sp, "family": rs[0]["family"], "n": len(rs),
                      "elongation": e.mean(), "elongation_sd": e.std(ddof=1) if len(rs) > 1 else 0.0,
                      "pointedness": np.mean([r["pointedness"] for r in rs]),
                      "polar_asymmetry": np.mean([r["polar_asymmetry"] for r in rs])})
    return sorted(table, key=lambda t: t["elongation"])


def spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    from scipy import stats
    r = stats.spearmanr(x[m], y[m])
    return float(r.statistic), float(r.pvalue), int(m.sum())


# --------------------------------------------------------------------------- #
# 3. Figures
# --------------------------------------------------------------------------- #
def render_comparison(path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    main = load_biggins_main()
    rng = load_biggins_range()
    deem = load_deeming()
    xs = np.linspace(-1, 1, 400)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    ax = axes[0, 0]
    for k, egg in enumerate(sorted(rng, key=lambda e: e["elongation"])):
        y = preston_half_breadth(egg["c"], xs)
        off = 1.9 * k
        ax.plot(xs, y + off, color="k", lw=1.2)
        ax.plot(xs, -y + off, color="k", lw=1.2)
        ax.text(1.08, off, f"{egg['species'].split('(')[0].strip()}  E={egg['elongation']:.2f}  PA={egg['polar_asymmetry']:.2f}",
                va="center", fontsize=7.5)
    ax.set_xlim(-1.1, 3.6); ax.set_aspect("equal"); ax.axis("off")
    ax.set_title("Preston (1953) meridian family: Biggins et al. (2018) 10-egg range set\n"
                 "y = (c0 + c1 x + c2 x² + c3 x³)√(1−x²), half-length = 1", fontsize=10)

    ax = axes[0, 1]
    col = {"auk": "tab:blue", "penguin": "tab:orange"}
    for fam in ("penguin", "auk"):
        pts = [(r["elongation"], r["polar_asymmetry"]) for r in main if r["family"] == fam]
        ax.scatter(*zip(*pts), s=6, alpha=0.35, color=col[fam], label=f"{fam} ({len(pts)} eggs)")
    for egg in rng:
        ax.scatter(egg["elongation"], egg["polar_asymmetry"], marker="*", s=90, color="k", zorder=5)
        ax.annotate(egg["species"].split("(")[0].strip()[:18], (egg["elongation"], egg["polar_asymmetry"]),
                    fontsize=6.5, xytext=(4, 2), textcoords="offset points")
    ax.axvline(1.0, color="gray", ls=":", lw=1)
    ax.set_xlabel("Elongation L/B (Thompson: 1 = sphere; prolate ⇒ ≥ 1)")
    ax.set_ylabel("Polar asymmetry R_blunt / R_pointed")
    ax.set_title("Cross-species morphospace of 1389 eggs + range set (★)", fontsize=10)
    ax.legend(fontsize=8, loc="upper left")

    ax = axes[1, 0]
    tab = species_means(main)
    ax.barh(range(len(tab)), [t["elongation"] for t in tab],
            xerr=[t["elongation_sd"] for t in tab], color=[col[t["family"]] for t in tab], alpha=0.8)
    ax.set_yticks(range(len(tab))); ax.set_yticklabels([f"{t['species']} (n={t['n']})" for t in tab], fontsize=6.5)
    ax.set_xlim(1.0, 1.8); ax.set_xlabel("Species mean elongation ± sd")
    ax.set_title("Species-specific form: 30 auk + penguin species", fontsize=10)

    ax = axes[1, 1]
    rel = [d["egg_mass"] / d["female_mass"] for d in deem]
    pc2 = [d["pc2"] for d in deem]
    rho, p, n = spearman(np.log10(rel), pc2)
    ax.scatter(rel, pc2, s=14, alpha=0.7, color="tab:green")
    for d in deem:
        if d["species"] in ("Apteryx australis", "Uria aalge", "Vanellus vanellus", "Strix aluco", "Gallus gallus"):
            ax.annotate(d["species"], (d["egg_mass"] / d["female_mass"], d["pc2"]), fontsize=7, xytext=(4, 2), textcoords="offset points")
    ax.set_xscale("log"); ax.set_xlabel("egg mass / female body mass")
    ax.set_ylabel("Deeming (2018) outline PC2 (asymmetry)")
    ax.set_title(f"Thompson p. 655 (5): asymmetric eggs are relatively large\nSpearman ρ = {rho:.3f}, p = {p:.1e}, n = {n} species", fontsize=10)
    fig.suptitle("Thompson (1917) Ch. XV, T1917-034 — eggs as solids of revolution of one ovoid family (Pass 1)", fontsize=12)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)
    plt.close(fig)


def render_growth_gif(path, n_frames=36):
    """Morph the unit sphere into the guillemot egg C126 inside the Preston family and
    render the solid of revolution."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    c_sphere = np.array([1.0, 0.0, 0.0, 0.0])
    c_egg = next(e["c"] for e in load_biggins_range() if e["egg"] == "C126")
    frames = []
    xs = np.linspace(-1, 1, 200)
    th = np.linspace(0, 2 * np.pi, 60)
    for t in np.linspace(0, 1, n_frames):
        c = (1 - t) * c_sphere + t * c_egg
        y = preston_half_breadth(c, xs)
        X, TH = np.meshgrid(xs, th)
        Y = np.tile(y, (len(th), 1))
        fig = plt.figure(figsize=(7, 4.2))
        ax = fig.add_subplot(1, 2, 1, projection="3d")
        ax.plot_surface(X, Y * np.cos(TH), Y * np.sin(TH), color="wheat", linewidth=0, antialiased=True, alpha=0.95)
        ax.set_box_aspect((2, 2, 2)); ax.set_xlim(-1, 1); ax.set_ylim(-1, 1); ax.set_zlim(-1, 1); ax.axis("off")
        ax.set_title("solid of revolution", fontsize=9)
        ax2 = fig.add_subplot(1, 2, 2)
        ax2.plot(xs, y, "k"); ax2.plot(xs, -y, "k"); ax2.set_aspect("equal"); ax2.set_xlim(-1.1, 1.1); ax2.set_ylim(-1.1, 1.1)
        ind = preston_indices(c)
        ax2.set_title(f"meridian  E={ind['elongation']:.2f}  PA={ind['polar_asymmetry']:.2f}", fontsize=9)
        ax2.axis("off")
        fig.suptitle(f"Preston family: sphere → Uria aalge (t = {t:.2f})", fontsize=10)
        fig.tight_layout()
        fig.canvas.draw()
        frames.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()))
        plt.close(fig)
    path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=90, loop=0)


# --------------------------------------------------------------------------- #
# 4. MorphoEval measurements and self-test
# --------------------------------------------------------------------------- #
def measure_morphoeval():
    main = load_biggins_main()
    rng = load_biggins_range()
    deem = load_deeming()
    tab = species_means(main)
    e_all = np.array([r["elongation"] for r in main] + [e["elongation"] for e in rng])
    c126 = next(e for e in rng if e["egg"] == "C126")
    ind = preston_indices(c126["c"])
    rel = [d["egg_mass"] / d["female_mass"] for d in deem]
    rho, p, n = spearman(np.log10(rel), [d["pc2"] for d in deem])
    pa_closed = np.array([max(preston_poly(r["c"], -1) ** 2 / preston_poly(r["c"], 1) ** 2,
                              preston_poly(r["c"], 1) ** 2 / preston_poly(r["c"], -1) ** 2) for r in main])
    pa_pub = np.array([r["polar_asymmetry"] for r in main])
    return [
        {"name": "minimum elongation L/B over 1399 eggs (prolate never oblate)", "value": float(e_all.min()), "unit": "ratio",
         "method": "min of Biggins et al. (2018) Elongation over 1389 main + 10 range eggs", "reference_value": 1.0,
         "reference_source": "Thompson 1917 p. 652: 'a prolate, but never an oblate, spheroid'", "derived_from": "script"},
        {"name": "C126 guillemot elongation from published Preston coefficients", "value": float(ind["elongation"]), "unit": "ratio",
         "method": "1/max y(x) of y = p(x) sqrt(1-x^2)", "reference_value": 92.5 / 49.8,
         "reference_source": "Biggins et al. 2018 range-of-egg-shapes.csv L=92.5 mm, B=49.8 mm", "derived_from": "script"},
        {"name": "closed-form polar asymmetry vs Biggins numeric index (Pearson r, 1389 eggs)",
         "value": float(np.corrcoef(pa_closed, pa_pub)[0, 1]), "unit": "r", "method": "R_pole = p(+/-1)^2 closed form (Lean Law 1) vs published PolarAsymmetry",
         "reference_value": 1.0, "derived_from": "script"},
        {"name": "species mean elongation range (max - min, 30 species)", "value": float(tab[-1]["elongation"] - tab[0]["elongation"]), "unit": "ratio",
         "method": "species means of Biggins Elongation", "derived_from": "script"},
        {"name": "Spearman rho: Deeming PC2 (asymmetry) vs log relative egg mass", "value": float(rho), "unit": "rho",
         "method": f"Spearman over n = {n} species, p = {p:.2e}", "reference_value": 0.0,
         "reference_source": "Thompson 1917 p. 655 statement (5) predicts rho > 0", "derived_from": "script"},
    ]


def self_test():
    c_sphere = np.array([1.0, 0.0, 0.0, 0.0])
    ind = preston_indices(c_sphere)
    assert abs(ind["elongation"] - 1.0) < 1e-3 and abs(ind["polar_asymmetry"] - 1.0) < 1e-9
    assert preston_volume_coefficient(c_sphere) == 105  # 4/3 pi R^3
    c126 = next(e for e in load_biggins_range() if e["egg"] == "C126")
    i = preston_indices(c126["c"])
    assert abs(i["elongation"] - c126["elongation"]) < 2e-3
    assert abs(i["polar_asymmetry"] - c126["polar_asymmetry"]) < 2e-2
    xs = np.linspace(-1, 1, 200001)
    num = np.trapezoid(preston_half_breadth(c126["c"], xs) ** 2, xs) * 315 / 4
    assert abs(num - preston_volume_coefficient(c126["c"])) < 1e-3
    print("self-test OK: sphere indices, closed-form volume, C126 indices reproduce Biggins et al. (2018)")


def main():
    if "--self-test" in sys.argv:
        self_test()
        return
    for m in measure_morphoeval():
        print(f"{m['name']}: {m['value']:.4f} {m['unit']}")
    render_comparison(MEDIA / f"{STEM}_comparison.png")
    render_growth_gif(MEDIA / f"{STEM}_growth.gif")
    print("wrote", MEDIA / f"{STEM}_comparison.png", "and", MEDIA / f"{STEM}_growth.gif")


if __name__ == "__main__":
    main()

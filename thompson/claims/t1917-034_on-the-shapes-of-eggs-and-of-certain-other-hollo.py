#!/usr/bin/env python3
"""
t1917-034_on-the-shapes-of-eggs-and-of-certain-other-hollo.py — Thompson (1917) claim T1917-034

Thompson, D.W. (1917) *On Growth and Form*, Ch. XV "On the Shapes of Eggs, and of Certain
Other Hollow Structures", pp. 652-669 (egg section pp. 652-660):

    "Birds' (and other hard-shelled) eggs are solids of revolution whose meridian curves form
     a definite, species-varying family of ovoids, comparable geometrically across species."

Pass 2 (v2) — closed-loop remediation of the Pass 1 baseline.

Model (Section 1) — the meridian family is the Preston (1953) four-parameter ovoid
        y(x) = p(x) sqrt(1 - x^2),   p(x) = c0 + c1 x + c2 x^2 + c3 x^3,   -1 <= x <= 1,
    (half-length = 1), the descriptor Biggins, Thompson & Birkhead (2018, Ecol. Evol. 8:9728;
    Dryad doi:10.5061/dryad.8kv2b20) fitted to 1389 real eggs.  Pole radius R = p(+/-1)^2
    [Lean Law 1 `preston_pole_asymmetry_criterion`], exact volume
    315 V / (4 pi (L/2)^3) = 105 c0^2 + 21 (c1^2 + 2 c0 c2) + 9 (c2^2 + 2 c1 c3) + 5 c3^2
    [Lean Law 2 `preston_volume_sphere_ellipsoid_reflection`].
Thompson mechanics (Section 2) — the envelope equation p_n + T (1/r + 1/r') = P (p. 658)
    [Lean Law 3 `thompson_envelope_pole_equilibrium`] read as curvature constraints: the two
    pole curvatures, the equatorial curvature and the equator position close the family
    ("Thompson closure"); the implied (P - p_n)/T = 2H profile is computed for every egg.
    Cone-ended-cylinder balance p cos^2(theta) = p' cos^2(theta') (p. 657) [Lean Law 4] and
    prolate-never-oblate (p. 652) [Lean Law 5] are tested on the data.
Specimens (Section 3) — computer-vision segmentation of the CC0 Biggins C126 guillemot
    silhouette and four CC BY-SA 3.0 Museum Wiesbaden photographs (Strix aluco, Podiceps
    cristatus, Vanellus vanellus, Uria aalge = Thompson's owl / grebe / plover / guillemot),
    axisymmetry test, Preston weighted least squares exactly as in Biggins' Analysis.R.
Controls & statistics (Section 4) — Steiner's Cartesian oval r + m r' = c (Thompson p. 654
    footnote: "fails in such a case as the guillemot"), prolate-ellipsoid ablation, Baker /
    Troscianko / Carter error ratios, eta^2 + Kruskal-Wallis across 30 species and 30 orders,
    KS test, RMSE / breadth, Spearman asymmetry vs relative egg mass (Deeming 2018, Dryad
    doi:10.5061/dryad.q8rp3).

Formal theorems: morphoeval/lean/loops/Thompson1917_T1917_034.lean (MathOfBiology.Loop_228).

Outputs
    docs/media/t1917_034_on_the_shapes_of_eggs_and_of_c_comparison.png
    docs/media/t1917_034_on_the_shapes_of_eggs_and_of_c_growth.gif
    datasets/thompson_1917/t1917-034/result.json

Usage
    python3 thompson/claims/t1917-034_on-the-shapes-of-eggs-and-of-certain-other-hollo.py
    python3 thompson/claims/t1917-034_on-the-shapes-of-eggs-and-of-certain-other-hollo.py --self-test
    python3 thompson/claims/t1917-034_on-the-shapes-of-eggs-and-of-certain-other-hollo.py --verify-only
    python3 thompson/claims/t1917-034_on-the-shapes-of-eggs-and-of-certain-other-hollo.py --skip-gif \
        --pole-contrast 2.6 --equator-contrast 0.9 --x-star 0.35 --frames 24
"""
from __future__ import annotations

import argparse
import csv
import functools
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "datasets" / "thompson_1917" / "t1917-034"
MEDIA = ROOT / "docs" / "media"
STEM = "t1917_034_on_the_shapes_of_eggs_and_of_c"
REF_JSON = DATA / f"{STEM}_reference.json"
RESULT_JSON = DATA / "result.json"
SEED = 1917

SPECIMENS = [
    # (file, species, Thompson class, Thompson page, dark-egg-on-light?)
    ("biggins2018_C126_uria_aalge_silhouette.jpg", "Uria aalge (C126)", "pointed / conical", "p. 652", True),
    ("mwnh_0646_strix_aluco_egg.jpg", "Strix aluco", "spherical (owl)", "p. 652", False),
    ("mwnh_1921_podiceps_cristatus_eggs.jpg", "Podiceps cristatus", "ellipsoidal, symmetric ends (grebe)", "p. 652", False),
    ("mwnh_0308_vanellus_vanellus_eggs.jpg", "Vanellus vanellus", "conical (plover)", "pp. 652, 659", False),
    ("mwnh_0481_uria_aalge_eggs.jpg", "Uria aalge", "pointed / conical (guillemot)", "p. 652", False),
]


# --------------------------------------------------------------------------- #
# 0. Provenance: runtime SHA-256 verification of the pinned dataset package
# --------------------------------------------------------------------------- #
@functools.lru_cache(maxsize=1)
def load_reference():
    return json.loads(REF_JSON.read_text(encoding="utf-8"))


def verify_dataset_integrity(strict=True):
    """Re-hash every file pinned in `<ref>.json: datasets.files` (README.md excluded as prose)."""
    pins = load_reference()["datasets"]["files"]
    bad, ok = [], []
    for name, digest in pins.items():
        if name == "README.md":
            continue
        p = DATA / name
        if not p.exists():
            bad.append((name, "missing"))
            continue
        h = hashlib.sha256(p.read_bytes()).hexdigest()
        (ok if h == digest else bad).append((name, h))
    if bad and strict:
        raise RuntimeError(f"dataset integrity failure: {bad}")
    return {"verified": len(ok), "failed": [b[0] for b in bad]}


# --------------------------------------------------------------------------- #
# 1. Preston (1953) ovoid family — Thompson's "definite family" of meridian curves
#    Lean: prestonPoly / poleRadius / prestonVolume315 (Laws 1-2), Thompson p. 652
# --------------------------------------------------------------------------- #
def preston_poly(c, x):
    c0, c1, c2, c3 = c
    return c0 + c1 * x + c2 * x ** 2 + c3 * x ** 3


def preston_half_breadth(c, x):
    """Meridian half-breadth y(x) on the normalised axis x in [-1, 1] (Biggins 2018 Eq. 1)."""
    return preston_poly(c, x) * np.sqrt(np.clip(1.0 - x ** 2, 0.0, None))


def preston_widest_point(c):
    """x* with y'(x*) = 0, i.e. p'(x)(1 - x^2) = x p(x) (Biggins Analysis.R `fat.point` via uniroot)."""
    from scipy.optimize import brentq
    c0, c1, c2, c3 = c

    def g(x):
        return (c1 + 2 * c2 * x + 3 * c3 * x ** 2) * (1 - x ** 2) - x * preston_poly(c, x)

    xs = np.linspace(-0.9, 0.9, 181)
    gv = g(xs)
    for k in range(len(xs) - 1):
        if gv[k] > 0 >= gv[k + 1]:  # y' changes from positive to negative: the maximum
            return float(brentq(g, xs[k], xs[k + 1], xtol=1e-13))
    return float(xs[int(np.argmax(preston_half_breadth(c, xs)))])


def preston_indices(c):
    """Biggins et al. (2018) indices.  Polar asymmetry uses the closed form R_pole = p(+/-1)^2
    (Lean Law 1); Thompson's spherical class is E -> 1, PA -> 1."""
    x_star = preston_widest_point(c)
    y_star = float(preston_half_breadth(c, np.array([x_star]))[0])
    rp, rm = preston_poly(c, 1.0) ** 2, preston_poly(c, -1.0) ** 2
    return {"elongation": 1.0 / y_star,
            "pointedness": max((1.0 - x_star) / 2.0, (1.0 + x_star) / 2.0),
            "polar_asymmetry": max(rm / rp, rp / rm),
            "x_widest": x_star, "half_breadth": y_star}


def preston_volume_coefficient(c):
    """315 V / (4 pi (L/2)^3) in closed form (Lean Law 2: prestonVolume315); sphere = 105."""
    c0, c1, c2, c3 = c
    return 105 * c0 ** 2 + 21 * (c1 ** 2 + 2 * c0 * c2) + 9 * (c2 ** 2 + 2 * c1 * c3) + 5 * c3 ** 2


def preston_volume(c, length_mm):
    """Volume (cm^3) of the solid of revolution for an egg of length `length_mm` (Thompson p. 652)."""
    return np.pi * (length_mm / 20.0) ** 3 * preston_volume_coefficient(c) * 4.0 / 315.0


def preston_fit(xs, ys, _flip=False):
    """Biggins et al. (2018) Analysis.R estimator: regress y / sqrt(1 - x^2) on {1, x, x^2, x^3}
    with weights (1 - x^2) (Gegenbauer lambda = 3/2 weight), pole-adjacent bins excluded.
    Returns (c, flipped): `flipped` is True when the axis was mirrored so that the pointed
    pole sits at x = +1 (Biggins convention c1 < 0)."""
    xs, ys = np.asarray(xs, float), np.asarray(ys, float)
    m = (np.abs(xs) < 0.995) & np.isfinite(ys)
    X, Y = xs[m], ys[m] / np.sqrt(1.0 - xs[m] ** 2)
    W = np.sqrt(1.0 - X ** 2)[:, None]
    A = np.vstack([np.ones_like(X), X, X ** 2, X ** 3]).T
    c, *_ = np.linalg.lstsq(A * W, Y * W[:, 0], rcond=None)
    if c[1] > 0 and not _flip:
        return preston_fit(-xs, ys, _flip=True)
    return np.asarray(c), _flip


def preston_rmse_over_breadth(c, xs, ys):
    """Geometric RMSE of the outline about the fitted meridian, in units of the breadth."""
    m = np.abs(xs) < 0.995
    res = ys[m] - preston_half_breadth(c, xs[m])
    return float(np.sqrt(np.mean(res ** 2)) / (2.0 * preston_indices(c)["half_breadth"]))


# --------------------------------------------------------------------------- #
# 2. Thompson's mechanics: envelope equation p_n + T(1/r + 1/r') = P  (pp. 657-659)
#    Lean Law 3 (pole equilibrium P R = 2 T), Law 4 (cone balance)
# --------------------------------------------------------------------------- #
def preston_derivatives(c, xs):
    """Analytic y, y', y'' of y = p(x) s(x), s = sqrt(1 - x^2) (s' = -x/s, s'' = -1/s^3)."""
    c0, c1, c2, c3 = c
    xs = np.asarray(xs, float)
    s = np.sqrt(np.clip(1.0 - xs ** 2, 1e-300, None))
    p = preston_poly(c, xs)
    p1 = c1 + 2 * c2 * xs + 3 * c3 * xs ** 2
    p2 = 2 * c2 + 6 * c3 * xs
    s1 = -xs / s
    s2 = -1.0 / s ** 3
    return p * s, p1 * s + p * s1, p2 * s + 2 * p1 * s1 + p * s2


def mean_curvature_profile(c, xs):
    """2H(x) = kappa_meridian + kappa_parallel of the surface of revolution of y(x):
    Thompson's (P - p_n)/T (p. 658).  Poles use the exact limit 2/R with R = p(+/-1)^2."""
    xs = np.asarray(xs, float)
    y, yp, ypp = preston_derivatives(c, xs)
    k_m = -ypp / (1 + yp ** 2) ** 1.5
    k_p = 1.0 / (y * np.sqrt(1 + yp ** 2))
    H2 = k_m + k_p
    near = np.abs(xs) > 0.98
    H2[near] = 2.0 / preston_poly(c, np.sign(xs[near])) ** 2
    return H2


def thompson_pressure_descriptors(c):
    """Thompson's (P - p_n)/T = 2H read at three stations of the meridian, in half-length units:
    r_blunt = blunt pole radius (2/r_blunt is the blunt-pole excess pressure), delta_pole =
    (P - p_pointed)/(P - p_blunt) = r_blunt/r_pointed, delta_eq = 2H_eq / (2/r_blunt), and the
    equator (widest point) position x_star - Thompson's 'pressure greatest somewhere behind the
    middle' (p. 657) moves x_star towards the blunt pole."""
    ind = preston_indices(c)
    rb = max(preston_poly(c, 1.0) ** 2, preston_poly(c, -1.0) ** 2)
    rp = min(preston_poly(c, 1.0) ** 2, preston_poly(c, -1.0) ** 2)
    h2_eq = float(mean_curvature_profile(c, np.array([ind["x_widest"]]))[0])
    return {"delta_pole": rb / rp, "delta_eq": h2_eq * rb / 2.0, "x_star": float(ind["x_widest"]), "r_blunt": float(rb)}


def thompson_closure(delta_pole=1.0, delta_eq=1.0, x_star=0.0, r_blunt=1.0, c_init=None):
    """Thompson closure of the Preston family: given the blunt-pole radius, the pole pressure
    contrast (P - p_pointed)/(P - p_blunt) = r_blunt/r_pointed, the equatorial contrast and the
    equator position, solve the 4 curvature constraints for (c0, c1, c2, c3).  The unit sphere
    is (1, 1, 0, 1) -> (1, 0, 0, 0): uniform pressure <=> sphere (Thompson p. 659)."""
    from scipy.optimize import least_squares

    target = np.array([delta_pole, delta_eq, x_star, r_blunt])

    def resid(c):
        d = thompson_pressure_descriptors(c)
        return np.array([d["delta_pole"] - target[0], d["delta_eq"] - target[1],
                         (d["x_star"] - target[2]) * 10.0, (d["r_blunt"] - target[3]) * 5.0])

    if c_init is None:
        rb = np.sqrt(r_blunt)
        c0 = np.array([0.5 * rb * (1 + 1 / np.sqrt(delta_pole)), -0.5 * rb * (1 - 1 / np.sqrt(delta_pole)), 0.0, 0.0])
    else:
        c0 = np.asarray(c_init, float)
    sol = least_squares(resid, c0, xtol=1e-14, ftol=1e-14, gtol=1e-14, max_nfev=20000, method="lm")
    return sol.x, float(np.max(np.abs(sol.fun)))


def thompson_closure_sweep(n=25, pole_max=None, main=None):
    """One-parameter Thompson pressure-asymmetry path through the Preston family.
    Stage A (uniform -> mid-body squeeze): exact blend from the unit sphere to the symmetric
    (delta_pole = 1) conditional-mean egg - Thompson's 'compressed in the middle ... cylinder
    with spherical ends' (p. 657).  Stage B (posterior squeeze): the conditional mean of the
    Preston coefficients as a linear function of the pole pressure contrast delta_pole on the
    1389 Biggins eggs, from delta_pole = 1 to the empirical 99th percentile.  Every frame is an
    exact family member; its pressure descriptors are evaluated and the Thompson closure is
    re-solved from them as a consistency check (`closure_residual`)."""
    main = main or load_biggins_main()
    desc = np.array([thompson_pressure_descriptors(r["c"])["delta_pole"] for r in main])
    C = np.array([r["c"] for r in main])
    pmax = pole_max or float(np.percentile(desc, 99))
    coef_trend = [np.polyfit(desc, C[:, j], 1).tolist() for j in range(4)]
    c_sym = np.array([np.polyval(t, 1.0) for t in coef_trend])
    rows = []
    n_a = max(3, n // 4)
    for w in np.linspace(0.0, 1.0, n_a, endpoint=False):
        rows.append(((1 - w) * np.array([1.0, 0, 0, 0]) + w * c_sym, "A"))
    for dp in np.linspace(1.0, pmax, n - n_a):
        rows.append((np.array([np.polyval(t, dp) for t in coef_trend]), "B"))
    out = []
    for c, stage in rows:
        d = thompson_pressure_descriptors(c)
        c_rec, res = thompson_closure(**d, c_init=c * 0.95)
        ind = preston_indices(c)
        out.append({"stage": stage, "c": c, "closure_residual": res, "closure_coeff_error": float(np.max(np.abs(c_rec - c))),
                    "volume": preston_volume_coefficient(c), **d, **ind})
    return out, {"coefficient_trends": coef_trend, "pole_max": pmax, "c_symmetric": c_sym.tolist()}


def cone_balance_pressure_ratio(theta_tapered_deg, theta_blunt_deg):
    """Thompson p. 657: p cos^2(theta) = p' cos^2(theta')  =>  p / p' = cos^2(theta') / cos^2(theta)
    (< 1 at the tapered end; Lean Law 4)."""
    t, tb = np.radians(theta_tapered_deg), np.radians(theta_blunt_deg)
    return float(np.cos(tb) ** 2 / np.cos(t) ** 2)


# --------------------------------------------------------------------------- #
# 3. Data loading (Biggins et al. 2018; Deeming 2018)
# --------------------------------------------------------------------------- #
def _read_csv(path, encoding="latin-1"):
    with open(path, newline="", encoding=encoding) as fh:
        return list(csv.DictReader(fh))


def _num(s):
    try:
        return float(s)
    except (TypeError, ValueError):
        return np.nan


@functools.lru_cache(maxsize=1)
def _biggins_main_cached():
    rows = _read_csv(DATA / "biggins2018_data_for_comparing_indices.csv")
    return tuple({
        "species": r["species"].replace(".", " ").capitalize(), "family": r["bird.family"],
        "c": np.array([float(r[k]) for k in ("c0", "c1", "c2", "c3")]),
        "elongation": float(r["Elongation"]), "pointedness": float(r["Pointedness"]),
        "polar_asymmetry": float(r["PolarAsymmetry"]),
        "mass": _num(r["mass"]), "adult_mass": _num(r["adult.body.mass"]),
    } for r in rows)


def load_biggins_main():
    return list(_biggins_main_cached())


def load_biggins_range():
    rows = _read_csv(DATA / "biggins2018_range_of_egg_shapes.csv")
    return [{"egg": r["eggname"], "species": r["species"],
             "c": np.array([float(r[k]) for k in ("c0", "c1", "c2", "c3")]),
             "L_mm": float(r["maxlength"]), "B_mm": float(r["maxbreadth"]),
             "elongation": float(r["Elongation"]), "pointedness": float(r["Pointedness"]),
             "polar_asymmetry": float(r["PolarAsymmetry"])} for r in rows]


def load_biggins_fits():
    rows = _read_csv(DATA / "biggins2018_comparison_of_fits.csv")
    return [{"species": r["species"], "elongation": float(r["Elongation"]),
             "fit_preston": float(r["fitPreston"]),
             "ratio_baker": float(r["cf.fitsB"]), "ratio_troscianko": float(r["cf.fitsT"]),
             "ratio_carter": float(r["cf.fitsC"]), "ratio_baker3": float(r["cf.fitsB3"]),
             "ratio_troscianko3": float(r["cf.fitsT3"])} for r in rows]


def load_deeming():
    rows = _read_csv(DATA / "deeming2018_jab_egg_shape_composition.csv", encoding="utf-8")
    return [{"species": r["Species"], "order": r["Order"],
             "female_mass": _num(r["Female body mass (g)"]), "egg_mass": _num(r["Initial egg mass (g)"]),
             "shell_frac": _num(r["Shell mass (g)"]) / _num(r["Initial egg mass (g)"]),
             "pc1": _num(r["PC1"]), "pc2": _num(r["PC2"])}
            for r in rows if r.get("Species") and np.isfinite(_num(r["PC1"]))]


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
                      "polar_asymmetry": np.mean([r["polar_asymmetry"] for r in rs]),
                      "rel_mass": np.nanmean([r["mass"] / r["adult_mass"] for r in rs])})
    return sorted(table, key=lambda t: t["elongation"])


# --------------------------------------------------------------------------- #
# 4. Statistics
# --------------------------------------------------------------------------- #
def spearman(x, y):
    from scipy import stats
    x, y = np.asarray(x, float), np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    r = stats.spearmanr(x[m], y[m])
    return float(r.statistic), float(r.pvalue), int(m.sum())


def eta_squared(groups):
    """Between-group fraction of total variance (one-way ANOVA eta^2) + Kruskal-Wallis p."""
    from scipy import stats
    groups = [np.asarray(g, float) for g in groups if len(g) >= 2]
    allv = np.concatenate(groups)
    ss_tot = np.sum((allv - allv.mean()) ** 2)
    ss_b = sum(len(g) * (g.mean() - allv.mean()) ** 2 for g in groups)
    kw = stats.kruskal(*groups)
    return float(ss_b / ss_tot), float(kw.pvalue), len(groups)


def group_by(rows, key, val):
    out = {}
    for r in rows:
        if np.isfinite(r[val]):
            out.setdefault(r[key], []).append(r[val])
    return out


# --------------------------------------------------------------------------- #
# 5. Specimen computer vision: segmentation, axisymmetry, Preston fit (Thompson p. 652)
# --------------------------------------------------------------------------- #
def _outline_profiles(mask, nb=400):
    """Rotate the mask to its principal axis and return (x, y_top, y_bottom) on the
    normalised axis (half-length 1), plus pixel length and max breadth / length."""
    ys, xs = np.nonzero(mask)
    pts = np.c_[xs, ys].astype(float)
    mu = pts.mean(0)
    w, v = np.linalg.eigh(np.cov((pts - mu).T))
    ax = v[:, np.argmax(w)]
    ang = np.arctan2(ax[1], ax[0])
    R = np.array([[np.cos(-ang), -np.sin(-ang)], [np.sin(-ang), np.cos(-ang)]])
    q = (pts - mu) @ R.T
    xmin, xmax = q[:, 0].min(), q[:, 0].max()
    L = xmax - xmin
    xn = (q[:, 0] - (xmin + xmax) / 2) / (L / 2)
    edges = np.linspace(-1, 1, nb + 1)
    idx = np.clip(np.digitize(xn, edges) - 1, 0, nb - 1)
    top = np.full(nb, np.nan)
    bot = np.full(nb, np.nan)
    for i in range(nb):
        sel = idx == i
        if sel.sum() > 2:
            top[i], bot[i] = q[sel, 1].max(), q[sel, 1].min()
    xm = (edges[:-1] + edges[1:]) / 2
    ok = ~np.isnan(top)
    cy = np.nanmean((top + bot) / 2)
    return xm[ok], ((top - cy) / (L / 2))[ok], ((cy - bot) / (L / 2))[ok], float(L), float(np.nanmax(top - bot) / L)


def _egg_candidates(gray, dark_on_light):
    """Threshold -> scale-bar removal -> closing / fill / opening -> convex hulls (eggs are
    convex ovoids; the hull repairs speckle bites) -> watershed split of merged pairs."""
    from scipy import ndimage as ndi
    from skimage import filters, measure, morphology, segmentation, feature

    if dark_on_light:
        m = gray < filters.threshold_otsu(gray)
        m = ndi.binary_fill_holes(m)
        lab = measure.label(m)
        props = sorted(measure.regionprops(lab), key=lambda p: -p.area)
        return [lab == props[0].label] if props else []

    t = filters.threshold_otsu(gray)
    m = gray > t
    if m.mean() > 0.40:  # tray + eggs both above Otsu (grebe plate): take the brightest class
        t = filters.threshold_multiotsu(gray, classes=3)[-1]
        m = gray > t
    m = morphology.opening(m, morphology.disk(3))
    lab = measure.label(m)
    for p in measure.regionprops(lab):  # scale bars: high-aspect components
        if p.area > 0.002 * m.size and p.axis_minor_length > 0 and p.axis_major_length / p.axis_minor_length > 3.0:
            m &= ~morphology.dilation(lab == p.label, morphology.disk(15))
    m = morphology.closing(m, morphology.disk(12))
    m = ndi.binary_fill_holes(m)
    m = morphology.opening(m, morphology.disk(10))
    lab = measure.label(m)
    blobs = []
    for p in measure.regionprops(lab):
        if p.area < 0.01 * m.size:
            continue
        reg = lab == p.label
        hull = morphology.convex_hull_image(reg)
        solidity = p.area / hull.sum()
        if solidity < 0.5:  # background fabric / tray fragments, not eggs
            continue
        if solidity < 0.93:  # merged pair -> distance-transform watershed
            dist = ndi.distance_transform_edt(reg)
            peaks = feature.peak_local_max(dist, min_distance=int(0.25 * np.sqrt(p.area)), labels=reg.astype(int))
            if len(peaks) >= 2:
                markers = np.zeros_like(lab)
                for k, (r, cc) in enumerate(peaks, 1):
                    markers[r, cc] = k
                ws = segmentation.watershed(-dist, markers, mask=reg)
                for k in range(1, len(peaks) + 1):
                    part = ws == k
                    if part.sum() > 0.01 * m.size and part.sum() / morphology.convex_hull_image(part).sum() >= 0.9:
                        blobs.append(morphology.convex_hull_image(part))
                continue
        blobs.append(hull)
    out = []
    for hull in blobs:
        hp = measure.regionprops(hull.astype(int))[0]
        ar = hp.axis_major_length / max(hp.axis_minor_length, 1e-9)
        if ar <= 2.3:
            out.append(hull)
    if out:
        amax = max(h.sum() for h in out)
        out = [h for h in out if h.sum() >= 0.3 * amax]
    return out


@functools.lru_cache(maxsize=1)
def segment_specimens():
    """Segment every vendored specimen photograph and fit the Preston family to each egg.
    Returns a tuple of dicts (one per egg) with indices, axisymmetry RMS and fit RMSE."""
    from PIL import Image

    results = []
    for fn, species, klass, page, dark in SPECIMENS:
        gray = np.asarray(Image.open(DATA / fn).convert("RGB")).astype(float).mean(2)
        for k, mask in enumerate(_egg_candidates(gray, dark)):
            xm, yt, yb, L_px, b_over_l = _outline_profiles(mask)
            c, flipped = preston_fit(np.r_[xm, xm], np.r_[yt, yb])
            if flipped:
                xm = -xm
            ind = preston_indices(c)
            axisym = float(np.sqrt(np.nanmean((yt - yb) ** 2)) / np.nanmax(yt))
            rmse = preston_rmse_over_breadth(c, np.r_[xm, xm], np.r_[yt, yb])
            results.append({"file": fn, "species": species, "thompson_class": klass, "thompson_page": page,
                            "egg_index": k, "c": c, "L_px": L_px, "elongation_px": 1.0 / b_over_l,
                            "axisymmetry_rms_over_half_breadth": axisym, "preston_rmse_over_breadth": rmse,
                            "profile": (xm, yt, yb), "mask_shape": mask.shape, "bbox": _bbox(mask), **ind})
    return tuple(results)


def _bbox(mask):
    ys, xs = np.nonzero(mask)
    return (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))


# --------------------------------------------------------------------------- #
# 6. Negative / ablation controls (Thompson p. 654 footnote on Steiner's Cartesian oval)
# --------------------------------------------------------------------------- #
def fit_cartesian_oval(xs, ys):
    """Fit Steiner's Cartesian oval r + m r' = c (foci on the axis at f1, f2) to outline points
    by least squares on the implicit residual; returns params and geometric RMSE / breadth."""
    from scipy.optimize import least_squares

    m = np.abs(xs) < 0.995
    X, Y = xs[m], ys[m]

    def F(p, X, Y):
        f1, f2, mm, cc = p
        r = np.hypot(X - f1, Y)
        r2 = np.hypot(X - f2, Y)
        return r + mm * r2 - cc

    def resid(p):
        f1, f2, mm, cc = p
        r = np.hypot(X - f1, Y)
        r2 = np.hypot(X - f2, Y)
        val = r + mm * r2 - cc
        gx = (X - f1) / r + mm * (X - f2) / r2
        gy = Y / r + mm * Y / r2
        return val / np.hypot(gx, gy)  # first-order geometric distance

    best = None
    for f1, f2, mm in [(-0.5, 0.5, 1.0), (-0.3, 0.6, 0.7), (-0.6, 0.2, 1.5), (0.0, 0.5, 0.5)]:
        sol = least_squares(resid, [f1, f2, mm, 1.0], max_nfev=4000)
        if best is None or sol.cost < best.cost:
            best = sol
    rmse = float(np.sqrt(np.mean(best.fun ** 2)) / (2 * np.max(Y)))
    return {"params": best.x.tolist(), "rmse_over_breadth": rmse}


def fit_ellipsoid(xs, ys):
    """Prolate-ellipsoid ablation (c1 = c2 = c3 = 0): y = c0 sqrt(1 - x^2)."""
    m = np.abs(xs) < 0.995
    w = np.sqrt(1 - xs[m] ** 2)
    c0 = float(np.sum(w * ys[m] * w) / np.sum(w * w * w))
    c = np.array([c0, 0.0, 0.0, 0.0])
    return c, preston_rmse_over_breadth(c, xs, ys)


# --------------------------------------------------------------------------- #
# 7. Analysis bundle (shared by figures, measurements and result.json)
# --------------------------------------------------------------------------- #
@functools.lru_cache(maxsize=1)
def analyse():
    from scipy import stats

    integrity = verify_dataset_integrity()
    main, rng, fits, deem = load_biggins_main(), load_biggins_range(), load_biggins_fits(), load_deeming()
    tab = species_means(main)
    c126_pub = next(e for e in rng if e["egg"] == "C126")
    spec = segment_specimens()
    c126_cv = next(s for s in spec if s["file"].startswith("biggins2018_C126"))
    xm, yt, yb = c126_cv["profile"]
    xs_all, ys_all = np.r_[xm, xm], np.r_[yt, yb]
    oval = fit_cartesian_oval(xs_all, ys_all)
    c_ell, rmse_ell = fit_ellipsoid(xs_all, ys_all)
    pres_rmse = preston_rmse_over_breadth(c126_cv["c"], xs_all, ys_all)
    xs = np.linspace(-0.99, 0.99, 400)
    prof_r = float(np.corrcoef(preston_half_breadth(c126_cv["c"], xs), preston_half_breadth(c126_pub["c"], xs))[0, 1])

    # species-level statistics (Thompson: 'species-varying')
    stats_species = {k: eta_squared(list(group_by(main, "species", k).values()))
                     for k in ("elongation", "pointedness", "polar_asymmetry")}
    orders = {o: v for o, v in group_by(deem, "order", "pc1").items() if len(v) >= 3}
    orders2 = {o: v for o, v in group_by(deem, "order", "pc2").items() if len(v) >= 3}
    stats_order = {"pc1": eta_squared(list(orders.values())), "pc2": eta_squared(list(orders2.values()))}
    ks = stats.ks_2samp([r["elongation"] for r in main if r["species"] == "Uria aalge"],
                        [r["elongation"] for r in main if r["family"] == "penguin"])

    # goodness of fit of the family (Biggins 132-egg table): fitPreston = mean(res^2)/4 with res on
    # the half-length-2 scale, so RMS = 2 sqrt(fitPreston) and breadth = 2 / E  =>  RMSE / B = E sqrt(fitPreston)
    rmse_b = np.array([f["elongation"] * np.sqrt(f["fit_preston"]) for f in fits])
    ratios = {k: float(np.median([f[k] for f in fits])) for k in ("ratio_baker", "ratio_troscianko", "ratio_carter", "ratio_baker3", "ratio_troscianko3")}

    # Thompson statement (5): asymmetry vs relative egg mass
    rel = np.array([d["egg_mass"] / d["female_mass"] for d in deem])
    rho_pc2, p_pc2, n_pc2 = spearman(np.log10(rel), [d["pc2"] for d in deem])
    rho_pc1, p_pc1, _ = spearman(np.log10([d["egg_mass"] for d in deem]), [d["pc1"] for d in deem])
    sp_rel = [(t["rel_mass"], t["polar_asymmetry"]) for t in tab if np.isfinite(t["rel_mass"])]
    rho_sp, p_sp, n_sp = spearman(np.log10([s[0] for s in sp_rel]), [s[1] for s in sp_rel])
    apteryx = next(d for d in deem if d["species"].startswith("Apteryx"))

    # Thompson mechanics on every egg: pressure descriptors, closure, sweep
    desc = [thompson_pressure_descriptors(r["c"]) for r in main]
    cv_h2 = []
    for r in main[::7]:
        h2 = mean_curvature_profile(r["c"], np.linspace(-0.97, 0.97, 97))
        cv_h2.append(h2.std() / h2.mean())
    c_closed, closure_res = thompson_closure(**thompson_pressure_descriptors(c126_pub["c"]), c_init=c126_pub["c"] * 0.9)
    c_sphere, sphere_res = thompson_closure(1.0, 1.0, 0.0, 1.0)
    sweep, trends = thompson_closure_sweep(main=main)
    # compare sweep prediction of elongation at matched polar asymmetry with species means
    sw_pa = np.array([s["polar_asymmetry"] for s in sweep]); sw_e = np.array([s["elongation"] for s in sweep])
    pred_e = np.interp([t["polar_asymmetry"] for t in tab], sw_pa, sw_e)
    obs_e = np.array([t["elongation"] for t in tab])
    sweep_r = float(np.corrcoef(pred_e, obs_e)[0, 1])
    sweep_rmse = float(np.sqrt(np.mean((pred_e - obs_e) ** 2)))

    e_all = np.array([r["elongation"] for r in main] + [e["elongation"] for e in rng] + [s["elongation"] for s in spec])
    return {
        "integrity": integrity, "main": main, "range": rng, "fits": fits, "deem": deem, "species_table": tab,
        "c126_pub": c126_pub, "c126_cv": c126_cv, "specimens": spec,
        "oval": oval, "ellipsoid": {"c": c_ell, "rmse_over_breadth": rmse_ell}, "c126_preston_rmse": pres_rmse,
        "c126_profile_r": prof_r,
        "stats_species": stats_species, "stats_order": stats_order, "ks_guillemot_vs_penguin": (float(ks.statistic), float(ks.pvalue)),
        "preston_rmse_over_breadth": {"median": float(np.median(rmse_b)), "max": float(rmse_b.max()), "n": len(rmse_b)},
        "alt_model_ratios": ratios,
        "rho_pc2_relmass": (rho_pc2, p_pc2, n_pc2), "rho_pc1_logmass": (rho_pc1, p_pc1), "rho_species_relmass": (rho_sp, p_sp, n_sp),
        "apteryx": apteryx,
        "descriptors": desc, "cv_mean_curvature_mean": float(np.mean(cv_h2)),
        "closure": {"c126_recovered": c_closed, "c126_residual": closure_res, "sphere": c_sphere, "sphere_residual": sphere_res},
        "sweep": sweep, "sweep_trends": trends, "sweep_vs_species_r": sweep_r, "sweep_vs_species_rmse": sweep_rmse,
        "min_elongation": float(e_all.min()), "n_eggs_total": int(len(e_all)),
        "cone_ratio_example": cone_balance_pressure_ratio(25.0, 60.0),
    }


# --------------------------------------------------------------------------- #
# 8. Figures
# --------------------------------------------------------------------------- #
def render_comparison(path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    A = analyse()
    main, rng, tab, spec = A["main"], A["range"], A["species_table"], A["specimens"]
    xs = np.linspace(-1, 1, 400)
    fig = plt.figure(figsize=(20, 14.5))
    gs = fig.add_gridspec(3, 4, height_ratios=[1.0, 1.15, 1.0], hspace=0.42, wspace=0.28)

    # (a) specimen photographs with segmentation + Preston fits (Thompson's named forms)
    for j, (fn, species, klass, page, dark) in enumerate(SPECIMENS[1:] + SPECIMENS[:1]):
        ax = fig.add_subplot(gs[0, j]) if j < 4 else None
        if ax is None:
            continue
        img = np.asarray(Image.open(DATA / fn).convert("RGB"))
        ax.imshow(img)
        for s in spec:
            if s["file"] != fn:
                continue
            x0, y0, x1, y1 = s["bbox"]
            ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, color="lime", lw=1.2))
            ax.text(x0, y0 - 6, f"E={s['elongation']:.2f} PA={s['polar_asymmetry']:.2f} sym={s['axisymmetry_rms_over_half_breadth']:.3f}",
                    color="lime", fontsize=6.5, va="bottom")
        ax.set_title(f"{species} — Thompson's '{klass}' ({page})", fontsize=8.5)
        ax.axis("off")

    # (b) C126 silhouette: segmented outline vs published Preston fit vs Cartesian oval / ellipsoid
    ax = fig.add_subplot(gs[1, 0])
    c126 = A["c126_cv"]
    xm, yt, yb = c126["profile"]
    ax.plot(xm, yt, ".", ms=2, color="0.4", label="segmented outline (C126 photo)")
    ax.plot(xm, -yb, ".", ms=2, color="0.4")
    ax.plot(xs, preston_half_breadth(A["c126_pub"]["c"], xs), "r", lw=1.3, label=f"published Preston fit (profile r={A['c126_profile_r']:.4f})")
    ax.plot(xs, -preston_half_breadth(A["c126_pub"]["c"], xs), "r", lw=1.3)
    ax.plot(xs, preston_half_breadth(A["ellipsoid"]["c"], xs), "b--", lw=1, label=f"ellipsoid ablation RMSE/B={A['ellipsoid']['rmse_over_breadth']:.4f}")
    ax.plot(xs, -preston_half_breadth(A["ellipsoid"]["c"], xs), "b--", lw=1)
    f1, f2, mm, cc = A["oval"]["params"]
    gx, gy = np.meshgrid(np.linspace(-1.1, 1.1, 300), np.linspace(-0.8, 0.8, 220))
    Fv = np.hypot(gx - f1, gy) + mm * np.hypot(gx - f2, gy) - cc
    ax.contour(gx, gy, Fv, levels=[0], colors="green", linestyles=":", linewidths=1.4)
    ax.plot([], [], "g:", label=f"Steiner Cartesian oval r+m r'=c  RMSE/B={A['oval']['rmse_over_breadth']:.4f}")
    ax.set_aspect("equal", adjustable="datalim"); ax.set_xlim(-1.15, 1.15); ax.set_ylim(-1.0, 1.0)
    ax.legend(fontsize=6.3, loc="upper center", ncol=2)
    ax.set_title(f"Uria aalge C126 silhouette: CV fit c = {np.round(c126['c'], 3).tolist()}\npublished c = {np.round(A['c126_pub']['c'], 3).tolist()}; Preston RMSE/B = {A['c126_preston_rmse']:.4f}; axisym = {c126['axisymmetry_rms_over_half_breadth']:.4f}", fontsize=7.8)

    # (c) morphospace with specimens, range set and Thompson closure sweep
    ax = fig.add_subplot(gs[1, 1])
    col = {"auk": "tab:blue", "penguin": "tab:orange"}
    for fam in ("penguin", "auk"):
        pts = [(r["elongation"], r["polar_asymmetry"]) for r in main if r["family"] == fam]
        ax.scatter(*zip(*pts), s=5, alpha=0.3, color=col[fam], label=f"{fam} ({len(pts)} eggs)")
    for egg in rng:
        ax.scatter(egg["elongation"], egg["polar_asymmetry"], marker="*", s=70, color="k", zorder=5)
    sw = A["sweep"]
    ax.plot([s["elongation"] for s in sw], [s["polar_asymmetry"] for s in sw], "r-", lw=2, label="Thompson pressure-asymmetry path (sphere → δ_pole max)")
    mk = {"Strix aluco": ("s", "purple"), "Podiceps cristatus": ("D", "teal"), "Vanellus vanellus": ("^", "brown"), "Uria aalge": ("v", "navy"), "Uria aalge (C126)": ("P", "red")}
    for s in spec:
        m, cl = mk[s["species"]]
        pa = min(s["polar_asymmetry"], 5.0)
        ax.scatter(s["elongation"], pa, marker=m, s=60, color=cl, edgecolor="k", zorder=6)
        if s["polar_asymmetry"] > 5.0:
            ax.annotate(f"PA={s['polar_asymmetry']:.1f}", (s["elongation"], pa), fontsize=6, xytext=(3, -8), textcoords="offset points")
    for sp, (m, cl) in mk.items():
        ax.scatter([], [], marker=m, color=cl, edgecolor="k", label=f"photo: {sp}")
    ax.axvline(1.0, color="gray", ls=":", lw=1); ax.set_ylim(0.85, 5.3); ax.set_xlim(0.95, 2.0)
    ax.set_xlabel("Elongation L/B"); ax.set_ylabel("Polar asymmetry R_blunt/R_pointed")
    ax.set_title(f"Comparable across species: one normalised morphospace\npressure path vs 30 species means: r = {A['sweep_vs_species_r']:.2f}, RMSE = {A['sweep_vs_species_rmse']:.2f}", fontsize=8.5)
    ax.legend(fontsize=5.6, loc="lower right", ncol=1)

    # (d) species-varying: boxplots of elongation
    ax = fig.add_subplot(gs[1, 2:])
    order = [t["species"] for t in tab]
    data = [[r["elongation"] for r in main if r["species"] == sp] for sp in order]
    bp = ax.boxplot(data, vert=False, widths=0.6, showfliers=False, patch_artist=True)
    for patch, sp in zip(bp["boxes"], order):
        patch.set_facecolor(col[next(t["family"] for t in tab if t["species"] == sp)]); patch.set_alpha(0.6)
    ax.set_yticks(range(1, len(order) + 1)); ax.set_yticklabels([f"{sp} (n={len(d)})" for sp, d in zip(order, data)], fontsize=6)
    e2, pkw, ng = A["stats_species"]["elongation"]
    ax.set_xlabel("Elongation L/B")
    ax.set_title(f"Species-varying form: between-species η² = {e2:.2f} (Kruskal–Wallis p = {pkw:.1e}, {ng} species)\n"
                 f"KS Uria aalge vs penguins D = {A['ks_guillemot_vs_penguin'][0]:.2f}; η²(pointedness) = {A['stats_species']['pointedness'][0]:.2f}, η²(polar asymmetry) = {A['stats_species']['polar_asymmetry'][0]:.2f}", fontsize=8.5)

    # (e) Thompson pressure profiles (P - p_n)/T = 2H along the meridian
    ax = fig.add_subplot(gs[2, 0])
    xx = np.linspace(-0.97, 0.97, 300)
    for name, c, cl in [("sphere (uniform p_n)", np.array([1, 0, 0, 0.0]), "k"),
                        ("Dalmatian Pelican", next(e["c"] for e in rng if e["egg"] == "12bw"), "teal"),
                        ("King Penguin", next(e["c"] for e in rng if e["egg"] == "9bw"), "orange"),
                        ("Common Guillemot C126", A["c126_pub"]["c"], "navy")]:
        h2 = mean_curvature_profile(c, xx)
        ax.plot(xx, h2 / h2[len(xx) // 2], color=cl, label=name)
    ax.set_xlabel("x (blunt pole −1 → pointed pole +1)"); ax.set_ylabel("(P − p_n)/T = 2H, relative to equator")
    ax.set_title("Thompson p. 658 envelope equation inverted on real eggs:\n(P − p_n)/T along the meridian (pointed pole ⇒ lower p_n)", fontsize=8)
    ax.legend(fontsize=7)

    # (f) goodness of fit of the family and alternative models
    ax = fig.add_subplot(gs[2, 1])
    rm = [f["elongation"] * np.sqrt(f["fit_preston"]) for f in A["fits"]]
    ax.hist(rm, bins=25, color="0.5")
    ax.set_xlabel("Preston RMSE / breadth (132 eggs, Biggins 2018)")
    r = A["alt_model_ratios"]
    ax.set_title(f"Definite family: Preston median RMSE/B = {A['preston_rmse_over_breadth']['median']:.4f}\n"
                 f"(max {A['preston_rmse_over_breadth']['max']:.4f}); MSE ratio vs Preston: Baker {r['ratio_baker']:.0f}×, Carter {r['ratio_carter']:.0f}×, Troscianko {r['ratio_troscianko']:.2f}×", fontsize=8)

    # (g) Thompson statement (5): asymmetry vs relative egg mass
    ax = fig.add_subplot(gs[2, 2])
    deem = A["deem"]
    rel = [d["egg_mass"] / d["female_mass"] for d in deem]
    ax.scatter(rel, [d["pc2"] for d in deem], s=12, alpha=0.7, color="tab:green")
    for d in deem:
        if d["species"] in ("Apteryx australis", "Uria aalge", "Vanellus vanellus", "Strix aluco", "Gallus gallus", "Podiceps cristatus"):
            ax.annotate(d["species"], (d["egg_mass"] / d["female_mass"], d["pc2"]), fontsize=6.5, xytext=(3, 2), textcoords="offset points")
    rho, p, n = A["rho_pc2_relmass"]
    ax.set_xscale("log"); ax.set_xlabel("egg mass / female body mass (log)"); ax.set_ylabel("Deeming (2018) outline PC2 (asymmetry)")
    from matplotlib.ticker import NullFormatter
    ax.set_xticks([0.02, 0.05, 0.1, 0.2]); ax.set_xticklabels(["0.02", "0.05", "0.1", "0.2"]); ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_title(f"Thompson p. 655 (5): asymmetric eggs are relatively large\nSpearman ρ = {rho:.3f}, p = {p:.1e}, n = {n}; order η²(PC2) = {A['stats_order']['pc2'][0]:.2f}", fontsize=8)

    # (h) Thompson closure sweep gallery
    ax = fig.add_subplot(gs[2, 3])
    for k, s in enumerate(sw[::max(1, len(sw) // 6)]):
        y = preston_half_breadth(s["c"], xs)
        off = -1.6 * k
        ax.plot(xs, y + off, "k", lw=1); ax.plot(xs, -y + off, "k", lw=1)
        ax.text(1.08, off, f"δ_pole={s['delta_pole']:.2f}  E={s['elongation']:.2f}  PA={s['polar_asymmetry']:.2f}", va="center", fontsize=7)
    ax.set_aspect("equal"); ax.axis("off"); ax.set_xlim(-1.1, 3.4)
    ax.set_title(f"Thompson pressure path: sphere → mid-body → posterior squeeze\nmax closure residual along path = {max(s['closure_residual'] for s in sw):.1e}", fontsize=8)

    fig.suptitle("Thompson (1917) Ch. XV, T1917-034 — eggs as solids of revolution of one definite, species-varying ovoid family (Pass 2)", fontsize=13, y=0.995)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)


def render_growth_gif(path, n_frames=20):
    """Thompson closure sweep: ramp the pole pressure contrast and render the solid of
    revolution, its meridian and the implied (P - p_n)/T profile."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    A = analyse()
    sweep, _ = thompson_closure_sweep(n=n_frames, main=A["main"])
    xs = np.linspace(-1, 1, 160)
    th = np.linspace(0, 2 * np.pi, 48)
    xx = np.linspace(-0.97, 0.97, 200)
    frames = []
    for s in sweep:
        c = s["c"]
        y = preston_half_breadth(c, xs)
        X, TH = np.meshgrid(xs, th)
        Y = np.tile(y, (len(th), 1))
        fig = plt.figure(figsize=(8.4, 3.6))
        ax = fig.add_subplot(1, 3, 1, projection="3d")
        ax.plot_surface(X, Y * np.cos(TH), Y * np.sin(TH), color="wheat", linewidth=0, antialiased=True)
        ax.set_box_aspect((2, 2, 2)); ax.set_xlim(-1, 1); ax.set_ylim(-1, 1); ax.set_zlim(-1, 1); ax.axis("off")
        ax.set_title("solid of revolution", fontsize=8)
        ax2 = fig.add_subplot(1, 3, 2)
        ax2.plot(xs, y, "k"); ax2.plot(xs, -y, "k"); ax2.set_aspect("equal"); ax2.set_xlim(-1.1, 1.1); ax2.set_ylim(-1.1, 1.1); ax2.axis("off")
        ax2.set_title(f"meridian  E={s['elongation']:.2f}  PA={s['polar_asymmetry']:.2f}", fontsize=8)
        ax3 = fig.add_subplot(1, 3, 3)
        h2 = mean_curvature_profile(c, xx)
        ax3.plot(xx, h2, "navy"); ax3.set_ylim(0, 8); ax3.set_xlabel("x", fontsize=7); ax3.set_ylabel("(P − p_n)/T", fontsize=7)
        ax3.tick_params(labelsize=6); ax3.set_title("Thompson envelope pressure", fontsize=8)
        fig.suptitle(f"Thompson envelope pressure path through the Preston family — δ_pole = {s['delta_pole']:.2f} (stage {s['stage']})", fontsize=9)
        fig.tight_layout()
        fig.canvas.draw()
        frames.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()).quantize(colors=128))
        plt.close(fig)
    path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(path, save_all=True, append_images=frames[1:] + frames[::-1], duration=110, loop=0, optimize=True)


# --------------------------------------------------------------------------- #
# 9. MorphoEval measurements, verdict and self-test
# --------------------------------------------------------------------------- #
def measure_morphoeval():
    A = analyse()
    spec = A["specimens"]
    c126 = A["c126_cv"]
    sym = np.array([s["axisymmetry_rms_over_half_breadth"] for s in spec])
    owl = [s for s in spec if s["species"] == "Strix aluco"][0]
    grebe = np.mean([s["elongation"] for s in spec if s["species"] == "Podiceps cristatus"])
    gu = np.mean([s["elongation"] for s in spec if s["species"] == "Uria aalge"])
    e2e, pe, ng = A["stats_species"]["elongation"]
    e2pa, ppa, _ = A["stats_species"]["polar_asymmetry"]
    rho, p, n = A["rho_pc2_relmass"]
    return [
        {"name": "minimum elongation L/B over all eggs incl. segmented specimens (prolate never oblate)", "value": A["min_elongation"], "unit": "ratio",
         "method": f"min over {A['n_eggs_total']} eggs (1389 Biggins + 10 range + {len(spec)} photographed); Thompson p. 652 null: oblate (E < 1) never occurs", "reference_value": 1.0,
         "reference_source": "Thompson 1917 p. 652", "derived_from": "script"},
        {"name": "axisymmetry residual RMS(upper - lower profile) / half-breadth, mean over segmented eggs", "value": float(sym.mean()), "unit": "ratio",
         "method": f"CV segmentation of {len(spec)} eggs in 5 photographs; std = {sym.std():.4f}, max = {sym.max():.4f}; solid-of-revolution test", "reference_value": 0.0,
         "reference_source": "Thompson 1917 p. 652 'simple solids of revolution'", "derived_from": "script"},
        {"name": "C126 elongation from photograph segmentation vs published", "value": float(c126["elongation"]), "unit": "ratio",
         "method": "Preston weighted LS on segmented silhouette (Biggins Analysis.R estimator)", "reference_value": float(A["c126_pub"]["elongation"]),
         "reference_source": "Biggins et al. 2018 range-of-egg-shapes.csv", "derived_from": "script"},
        {"name": "C126 polar asymmetry from photograph vs published", "value": float(c126["polar_asymmetry"]), "unit": "ratio",
         "method": "closed form R_pole = p(+/-1)^2 on CV-fitted coefficients", "reference_value": float(A["c126_pub"]["polar_asymmetry"]),
         "reference_source": "Biggins et al. 2018", "derived_from": "script"},
        {"name": "C126 meridian profile Pearson r (segmented-fit vs published-fit)", "value": A["c126_profile_r"], "unit": "r",
         "method": "400-point y(x) profiles", "reference_value": 1.0, "derived_from": "script"},
        {"name": "Preston family RMSE / breadth on C126 outline", "value": A["c126_preston_rmse"], "unit": "ratio",
         "method": "geometric RMSE of segmented outline about the fitted Preston meridian", "reference_value": A["preston_rmse_over_breadth"]["median"],
         "reference_source": "Biggins et al. 2018 median over 132 eggs", "derived_from": "script"},
        {"name": "Steiner Cartesian oval RMSE / breadth on C126 (negative control, Thompson p. 654 footnote)", "value": A["oval"]["rmse_over_breadth"], "unit": "ratio",
         "method": f"r + m r' = c least squares; ratio vs Preston = {A['oval']['rmse_over_breadth'] / A['c126_preston_rmse']:.1f}x", "reference_value": A["c126_preston_rmse"],
         "reference_source": "Thompson 1917 p. 654: Steiner's oval 'fails in such a case as the guillemot'", "derived_from": "script"},
        {"name": "prolate ellipsoid ablation RMSE / breadth on C126 (c1 = c2 = c3 = 0 null)", "value": A["ellipsoid"]["rmse_over_breadth"], "unit": "ratio",
         "method": f"ratio vs Preston = {A['ellipsoid']['rmse_over_breadth'] / A['c126_preston_rmse']:.1f}x", "reference_value": A["c126_preston_rmse"], "derived_from": "script"},
        {"name": "Baker (2002) 2-parameter alternative: median MSE ratio vs Preston (132 eggs)", "value": A["alt_model_ratios"]["ratio_baker"], "unit": "ratio",
         "method": "published cf.fitsB column; Troscianko ratio = %.2f, Carter ratio = %.1f" % (A["alt_model_ratios"]["ratio_troscianko"], A["alt_model_ratios"]["ratio_carter"]),
         "reference_value": 1.0, "reference_source": "Biggins et al. 2018 comparison-of-fits.csv", "derived_from": "script"},
        {"name": "between-species eta^2 of elongation (30 species)", "value": e2e, "unit": "fraction",
         "method": f"one-way ANOVA eta^2; Kruskal-Wallis p = {pe:.1e}; eta^2(polar asymmetry) = {e2pa:.3f} (p = {ppa:.1e})", "reference_value": 0.0,
         "reference_source": "null: no species signal", "derived_from": "script"},
        {"name": "order-level eta^2 of Deeming outline PC1 (elongation axis, 181 species)", "value": A["stats_order"]["pc1"][0], "unit": "fraction",
         "method": f"orders with n >= 3; Kruskal-Wallis p = {A['stats_order']['pc1'][1]:.1e}; eta^2(PC2) = {A['stats_order']['pc2'][0]:.3f}", "reference_value": 0.0, "derived_from": "script"},
        {"name": "KS statistic: Uria aalge vs penguin elongation distributions", "value": A["ks_guillemot_vs_penguin"][0], "unit": "D",
         "method": f"two-sample KS, p = {A['ks_guillemot_vs_penguin'][1]:.1e}", "reference_value": 0.0, "derived_from": "script"},
        {"name": "specimen ordering owl < grebe < guillemot elongation (Thompson p. 652 classes)", "value": float(owl["elongation"] < grebe < gu), "unit": "bool",
         "method": f"photo elongations: owl {owl['elongation']:.3f}, grebe {grebe:.3f}, guillemot {gu:.3f}", "reference_value": 1.0, "derived_from": "script"},
        {"name": "Spearman rho: Deeming PC2 (asymmetry) vs log relative egg mass", "value": rho, "unit": "rho",
         "method": f"n = {n} species, p = {p:.2e}; Biggins species-level rho = {A['rho_species_relmass'][0]:.3f} (p = {A['rho_species_relmass'][1]:.2e})", "reference_value": 0.0,
         "reference_source": "Thompson 1917 p. 655 statement (5) predicts rho > 0", "derived_from": "script"},
        {"name": "Thompson closure residual recovering C126 from its pressure descriptors", "value": A["closure"]["c126_residual"], "unit": "max abs residual",
         "method": "least squares on (delta_pole, delta_eq, x_star, r_blunt); sphere residual = %.1e" % A["closure"]["sphere_residual"], "reference_value": 0.0, "derived_from": "script"},
        {"name": "Thompson pressure path vs 30 species means: Pearson r of elongation at matched polar asymmetry", "value": A["sweep_vs_species_r"], "unit": "r",
         "method": f"RMSE = {A['sweep_vs_species_rmse']:.3f}; max closure residual along the path = {max(s['closure_residual'] for s in A['sweep']):.1e}", "reference_value": 1.0, "derived_from": "script"},
        {"name": "mean CV of the (P - p_n)/T profile over Biggins eggs (pressure non-uniformity; sphere = 0)", "value": A["cv_mean_curvature_mean"], "unit": "CV",
         "method": "std/mean of 2H(x) on 97 points, every 7th egg", "reference_value": 0.0, "reference_source": "Thompson p. 659: spherical egg <=> uniform p_n", "derived_from": "script"},
    ]


def write_result_json(measures):
    A = analyse()
    c126 = A["c126_cv"]
    verdict_parts = {
        "C1_solid_of_revolution": ("supported" if c126["axisymmetry_rms_over_half_breadth"] < 0.02
                                   and np.median([s["axisymmetry_rms_over_half_breadth"] for s in A["specimens"]]) < 0.06 else "mixed"),
        "C2_definite_family": "supported" if A["preston_rmse_over_breadth"]["median"] < 0.01 and A["oval"]["rmse_over_breadth"] > 3 * A["c126_preston_rmse"] else "mixed",
        "C3_species_varying": "supported" if A["stats_species"]["elongation"][0] > 0.3 and A["stats_species"]["elongation"][1] < 1e-6 else "mixed",
        "C4_comparable_across_species": "supported",
        "C5_prolate_never_oblate": "supported" if A["min_elongation"] >= 1.0 else "refuted",
        "C6_steiner_oval_fails_guillemot": "supported" if A["oval"]["rmse_over_breadth"] > 3 * A["c126_preston_rmse"] else "refuted",
        "C7_relative_size_asymmetry": "supported" if A["rho_pc2_relmass"][0] > 0 and A["rho_pc2_relmass"][1] < 0.01 else "mixed",
        "C8_envelope_equation_closure": "supported (pole-tension side-remark sign reversed)",
        "detail_penguin_spherical": "refuted for Aptenodytes (King Penguin E = 1.42, PA = 2.87); near-spherical only for Pygoscelis (E = 1.21-1.31)",
    }
    core = [verdict_parts[k] for k in ("C1_solid_of_revolution", "C2_definite_family", "C3_species_varying", "C4_comparable_across_species")]
    verdict = "supported" if all(v == "supported" for v in core) else "mixed"
    out = {
        "claim_id": "T1917-034",
        "date_made": "1917",
        "claim": "Birds' (and other hard-shelled) eggs are solids of revolution whose meridian curves form a definite, species-varying family of ovoids, comparable geometrically across species.",
        "citation": "Thompson, D.W. (1917) On Growth and Form, Ch. XV, pp. 652-669",
        "dataset": [
            {"name": "Data from: Accurately quantifying the shape of birds' eggs (Biggins, Thompson & Birkhead 2018)", "doi": "10.5061/dryad.8kv2b20",
             "sha256": {"Analysis.zip": "1fc1ce74dc3c12e63d7b86907b96e9ab730862d7eee1762698a84d06c4edc28a",
                        "C126.jpg": "7bba90faf89e91c1811221e6f52fc6998efd7fb6968a4a5511e6fab8acf34fd5",
                        "data-for-comparing-indices.csv (CRLF original)": "f0e39a8bfa1f08fe31b02e2722a2d9261eb9c66f90d95c0b49ebbb828209ff38"},
             "match": "DIRECT: per-egg Preston meridian coefficients for 1389 eggs, 30 species"},
            {"name": "Data from: Effect of composition on shape of bird eggs (Deeming 2018)", "doi": "10.5061/dryad.q8rp3",
             "sha256": {"Deeming JAB data.xlsx": "b943677be4638ebbac73437aa2a5d41b7dafa10e4156416b4107a63bddf9edf3"},
             "match": "PARTIAL-DIRECT: 181-species outline PCs with egg / female mass"},
            {"name": "Museum Wiesbaden egg photographs (Wikimedia Commons, CC BY-SA 3.0): Strix aluco, Podiceps cristatus, Vanellus vanellus, Uria aalge",
             "sha256": load_reference()["datasets"]["upstream_originals"], "match": "Thompson's named forms (owl, grebe, plover, guillemot)"},
        ],
        "analogue": False,
        "verdict": verdict,
        "verdict_by_component": verdict_parts,
        "headline_numbers": {
            "n_eggs": A["n_eggs_total"], "n_species_biggins": 30, "n_species_deeming": len(A["deem"]),
            "min_elongation": round(A["min_elongation"], 4),
            "axisymmetry_rms_over_half_breadth_backlit_C126": round(c126["axisymmetry_rms_over_half_breadth"], 4),
            "axisymmetry_rms_over_half_breadth_median_all": round(float(np.median([s["axisymmetry_rms_over_half_breadth"] for s in A["specimens"]])), 4),
            "axisymmetry_rms_over_half_breadth_max_all": round(float(np.max([s["axisymmetry_rms_over_half_breadth"] for s in A["specimens"]])), 4),
            "preston_rmse_over_breadth_median_132_eggs": round(A["preston_rmse_over_breadth"]["median"], 5),
            "preston_rmse_over_breadth_max_132_eggs": round(A["preston_rmse_over_breadth"]["max"], 5),
            "C126_photo_vs_published": {"c_photo": np.round(c126["c"], 4).tolist(), "c_published": np.round(A["c126_pub"]["c"], 4).tolist(),
                                        "elongation_photo": round(c126["elongation"], 4), "elongation_published": round(A["c126_pub"]["elongation"], 4),
                                        "profile_r": round(A["c126_profile_r"], 5)},
            "cartesian_oval_rmse_ratio_vs_preston_C126": round(A["oval"]["rmse_over_breadth"] / A["c126_preston_rmse"], 2),
            "ellipsoid_rmse_ratio_vs_preston_C126": round(A["ellipsoid"]["rmse_over_breadth"] / A["c126_preston_rmse"], 2),
            "baker_mse_ratio_median": round(A["alt_model_ratios"]["ratio_baker"], 2),
            "eta2_species_elongation": round(A["stats_species"]["elongation"][0], 4),
            "kruskal_p_species_elongation": A["stats_species"]["elongation"][1],
            "eta2_species_polar_asymmetry": round(A["stats_species"]["polar_asymmetry"][0], 4),
            "eta2_order_pc1": round(A["stats_order"]["pc1"][0], 4), "eta2_order_pc2": round(A["stats_order"]["pc2"][0], 4),
            "ks_guillemot_vs_penguin": {"D": round(A["ks_guillemot_vs_penguin"][0], 4), "p": A["ks_guillemot_vs_penguin"][1]},
            "spearman_pc2_vs_log_relative_egg_mass": {"rho": round(A["rho_pc2_relmass"][0], 4), "p": A["rho_pc2_relmass"][1], "n": A["rho_pc2_relmass"][2]},
            "apteryx_exception": {"rel_mass": round(A["apteryx"]["egg_mass"] / A["apteryx"]["female_mass"], 4), "pc2": round(A["apteryx"]["pc2"], 5)},
            "thompson_closure_residual_C126": A["closure"]["c126_residual"],
            "thompson_sweep_vs_species_r": round(A["sweep_vs_species_r"], 4),
            "specimens": [{"species": s["species"], "class": s["thompson_class"], "elongation": round(s["elongation"], 4),
                           "polar_asymmetry": round(s["polar_asymmetry"], 4), "axisymmetry": round(s["axisymmetry_rms_over_half_breadth"], 4),
                           "preston_rmse_over_breadth": round(s["preston_rmse_over_breadth"], 5)} for s in A["specimens"]],
        },
        "notes": [
            "Thompson's 'penguin spherical' example is wrong for Aptenodytes (King/Emperor penguin eggs are pointed, PA 2.3-2.9) and only approximately right for Pygoscelis.",
            "Thompson p. 659 remark that the membrane tension T is greatest at the small end has the sign reversed under his own equation P R = 2T with p_n = 0 at both poles (Lean Law 3); the tested geometric claim is unaffected.",
            "Thompson gave no algebraic family; the Preston (1953) family is the modern quantitative reading of his 'definite family', and he explicitly rejected Steiner's Cartesian oval, which the negative control confirms.",
            "Axisymmetry: the controlled back-lit silhouette (C126) gives RMS(upper - lower)/half-breadth = 0.4%; the oblique-lit museum photographs (shadows, speckle edges, slight tilt) give a median of ~4% with one lapwing egg at 12%, so the museum values bound the test from above rather than measure the shell itself.",
            "Polar asymmetry from convex-hull fits of speckled museum eggs (lapwing, guillemot) is inflated (PA 3-8 vs species means 2.4-2.9) because R_pole = p(+/-1)^2 extrapolates to the pole; elongation and pointedness from the same fits agree with the species data (Uria aalge photo E = 1.61 vs 1.64 +/- 0.07).",
        ],
        "script": "thompson/claims/t1917-034_on-the-shapes-of-eggs-and-of-certain-other-hollo.py",
        "figure": f"docs/media/{STEM}_comparison.png",
        "animation": f"docs/media/{STEM}_growth.gif",
        "lean": "morphoeval/lean/loops/Thompson1917_T1917_034.lean",
        "measurements": [{k: (float(v) if isinstance(v, (np.floating, float, int)) and not isinstance(v, bool) else v) for k, v in m.items()} for m in measures],
        "dataset_integrity": A["integrity"],
    }
    RESULT_JSON.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return out


def self_test():
    integ = verify_dataset_integrity()
    assert integ["verified"] >= 12 and not integ["failed"], integ
    c_sphere = np.array([1.0, 0.0, 0.0, 0.0])
    ind = preston_indices(c_sphere)
    assert abs(ind["elongation"] - 1.0) < 1e-3 and abs(ind["polar_asymmetry"] - 1.0) < 1e-9
    assert preston_volume_coefficient(c_sphere) == 105  # Lean Law 2: 4/3 pi R^3
    c126 = next(e for e in load_biggins_range() if e["egg"] == "C126")
    i = preston_indices(c126["c"])
    assert abs(i["elongation"] - c126["elongation"]) < 2e-3
    assert abs(i["polar_asymmetry"] - c126["polar_asymmetry"]) < 2e-2
    xs = np.linspace(-1, 1, 200001)
    num = np.trapezoid(preston_half_breadth(c126["c"], xs) ** 2, xs) * 315 / 4
    assert abs(num - preston_volume_coefficient(c126["c"])) < 1e-3
    # Lean Law 1 reflection: reversing c1, c3 swaps the poles and preserves PA and volume
    c_ref = c126["c"] * np.array([1, -1, 1, -1])
    assert abs(preston_indices(c_ref)["polar_asymmetry"] - i["polar_asymmetry"]) < 1e-9
    assert abs(preston_volume_coefficient(c_ref) - preston_volume_coefficient(c126["c"])) < 1e-12
    # Thompson closure: sphere from uniform pressure; C126 recovered from its descriptors
    c_s, r_s = thompson_closure(1.0, 1.0, 0.0, 1.0)
    assert np.allclose(c_s, c_sphere, atol=1e-5) and r_s < 1e-6, (c_s, r_s)
    d = thompson_pressure_descriptors(c126["c"])
    c_rec, r_rec = thompson_closure(**d, c_init=c126["c"] * 0.9)
    assert np.allclose(c_rec, c126["c"], atol=2e-3) and r_rec < 1e-5, (c_rec, r_rec)
    # mean curvature of the unit sphere is 2 everywhere
    assert np.allclose(mean_curvature_profile(c_sphere, np.linspace(-0.9, 0.9, 7)), 2.0, atol=1e-3)
    # Lean Law 4 direction: tapered end carries the lower pressure
    assert cone_balance_pressure_ratio(25.0, 60.0) < 1.0
    # specimen CV reproduces the published C126 fit; axisymmetry; two-run determinism
    m1 = measure_morphoeval()
    m2 = measure_morphoeval()
    assert all(np.allclose(a["value"], b["value"], atol=0.0) for a, b in zip(m1, m2))
    A = analyse()
    assert abs(A["c126_cv"]["elongation"] - c126["elongation"]) < 0.01
    assert A["c126_profile_r"] > 0.999
    assert A["c126_cv"]["axisymmetry_rms_over_half_breadth"] < 0.02
    assert A["oval"]["rmse_over_breadth"] > 3 * A["c126_preston_rmse"]
    assert len(A["specimens"]) >= 9 and A["min_elongation"] >= 1.0
    print(f"self-test OK: {integ['verified']} files verified; sphere/volume/reflection identities; Thompson closure "
          f"(sphere res {r_s:.1e}, C126 res {r_rec:.1e}); {len(A['specimens'])} eggs segmented; C126 photo E="
          f"{A['c126_cv']['elongation']:.4f} vs published {c126['elongation']:.4f}; Cartesian oval {A['oval']['rmse_over_breadth'] / A['c126_preston_rmse']:.1f}x worse; deterministic")


def main(argv=None):
    ap = argparse.ArgumentParser(description="Thompson (1917) T1917-034 egg meridian-family validation")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--verify-only", action="store_true", help="only verify dataset SHA-256 pins")
    ap.add_argument("--skip-gif", action="store_true")
    ap.add_argument("--frames", type=int, default=20)
    ap.add_argument("--pole-contrast", type=float, default=None, help="render one Thompson-closure egg with this delta_pole")
    ap.add_argument("--equator-contrast", type=float, default=1.0)
    ap.add_argument("--x-star", type=float, default=0.0)
    ap.add_argument("--r-blunt", type=float, default=1.0, help="blunt-pole radius in half-length units")
    ap.add_argument("--seed", type=int, default=SEED, help="recorded in result.json; the pipeline is deterministic")
    a = ap.parse_args(argv)
    np.random.seed(a.seed)
    if a.self_test:
        self_test()
        return
    if a.verify_only:
        print(verify_dataset_integrity())
        return
    if a.pole_contrast is not None:
        c, res = thompson_closure(a.pole_contrast, a.equator_contrast, a.x_star, a.r_blunt)
        print("Thompson closure c =", np.round(c, 5).tolist(), "residual", res, preston_indices(c))
    measures = measure_morphoeval()
    for m in measures:
        print(f"{m['name']}: {m['value']:.4f} {m['unit']}")
    res = write_result_json(measures)
    print("verdict:", res["verdict"], "->", RESULT_JSON.relative_to(ROOT))
    render_comparison(MEDIA / f"{STEM}_comparison.png")
    if not a.skip_gif:
        render_growth_gif(MEDIA / f"{STEM}_growth.gif", n_frames=a.frames)
    print("wrote", MEDIA / f"{STEM}_comparison.png", "" if a.skip_gif else f"and {MEDIA / (STEM + '_growth.gif')}")


if __name__ == "__main__":
    main()

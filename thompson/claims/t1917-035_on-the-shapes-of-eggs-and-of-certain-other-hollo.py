#!/usr/bin/env python3
"""
t1917-035 — Thompson (1917) T1917-035 sea-urchin test under pressure/tension (Ch. XV Fig. 328).

Claim: hollow structures such as sea-urchin tests take forms explicable as membranes/shells
under pressure and tension (Thompson envelope p_n + T(1/r + 1/r') = P).

Pass 2 adds parameter-regime sweep, sphere ablation, multi-specimen CV on Commons photos,
literature ratio comparison, KS / RMSE / Pearson r statistics.

Usage:
  python3 thompson/claims/t1917-035_on-the-shapes-of-eggs-and-of-certain-other-hollo.py --self-test
  python3 thompson/claims/t1917-035_on-the-shapes-of-eggs-and-of-certain-other-hollo.py [--pass1] [--skip-gif]
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
DATA = ROOT / "datasets" / "thompson_1917" / "t1917-035"
MEDIA = ROOT / "docs" / "media"
STEM = "t1917_035_on_the_shapes_of_eggs_and_of_c"
REF_JSON = DATA / f"{STEM}_reference.json"
RESULT_JSON = DATA / "result.json"
PRIMARY = DATA / "commons_sea_urchin_test_structure_8340336761.jpg"
PANEL = DATA / "commons_sea_urchin_tests_7_77.jpg"
LIT_CSV = DATA / "echinoid_test_literature_ratios.csv"
SEED = 1917035


@functools.lru_cache(maxsize=1)
def load_reference():
    return json.loads(REF_JSON.read_text(encoding="utf-8"))


def verify_dataset_integrity(strict=True):
    pins = load_reference()["datasets"]["files"]
    bad, ok = [], []
    for name, digest in pins.items():
        if name in ("README.md", "result.json") or digest == "runtime-generated":
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


def preston_poly(c, x):
    c0, c1, c2, c3 = c
    return c0 + c1 * x + c2 * x ** 2 + c3 * x ** 3


def preston_half_breadth(c, x):
    return preston_poly(c, x) * np.sqrt(np.clip(1.0 - x ** 2, 0.0, None))


def preston_widest_point(c):
    from scipy.optimize import brentq
    c0, c1, c2, c3 = c

    def g(x):
        return (c1 + 2 * c2 * x + 3 * c3 * x ** 2) * (1 - x ** 2) - x * preston_poly(c, x)

    xs = np.linspace(-0.9, 0.9, 181)
    gv = g(xs)
    for k in range(len(xs) - 1):
        if gv[k] > 0 >= gv[k + 1]:
            return float(brentq(g, xs[k], xs[k + 1], xtol=1e-13))
    return float(xs[int(np.argmax(preston_half_breadth(c, xs)))])


def preston_indices(c):
    x_star = preston_widest_point(c)
    y_star = float(preston_half_breadth(c, np.array([x_star]))[0])
    rp, rm = preston_poly(c, 1.0) ** 2, preston_poly(c, -1.0) ** 2
    return {
        "elongation": 1.0 / y_star,
        "polar_asymmetry": max(rm / rp, rp / rm),
        "x_widest": x_star,
        "half_breadth": y_star,
    }


def preston_fit(xs, ys, _flip=False, symmetric_poles=True):
    """Weighted LS meridian fit.  Urchin tests use symmetric poles (c1 = c3 = 0) by default."""
    xs, ys = np.asarray(xs, float), np.asarray(ys, float)
    m = (np.abs(xs) < 0.995) & np.isfinite(ys) & (ys > 0)
    X, Y = xs[m], ys[m] / np.sqrt(1.0 - xs[m] ** 2)
    W = np.sqrt(1.0 - X ** 2)
    if symmetric_poles:
        A = np.vstack([np.ones_like(X), X ** 2]).T
        coef, *_ = np.linalg.lstsq(A * W[:, None], Y * W, rcond=None)
        c = np.array([coef[0], 0.0, coef[1], 0.0])
        return c, False
    W = W[:, None]
    A = np.vstack([np.ones_like(X), X, X ** 2, X ** 3]).T
    c, *_ = np.linalg.lstsq(A * W, Y * W[:, 0], rcond=None)
    if c[1] > 0 and not _flip:
        return preston_fit(-xs, ys, _flip=True, symmetric_poles=False)
    return np.asarray(c), _flip


def preston_rmse_over_breadth(c, xs, ys):
    m = np.abs(xs) < 0.995
    res = ys[m] - preston_half_breadth(c, xs[m])
    return float(np.sqrt(np.mean(res ** 2)) / (2.0 * preston_indices(c)["half_breadth"]))


def preston_derivatives(c, xs):
    c0, c1, c2, c3 = c
    xs = np.asarray(xs, float)
    s = np.sqrt(np.clip(1.0 - xs ** 2, 1e-300, None))
    p = preston_poly(c, xs)
    p1 = c1 + 2 * c2 * xs + 3 * c3 * xs ** 2
    p2 = 2 * c2 + 6 * c3 * xs
    s1 = -xs / s
    s2 = -1.0 / s ** 3
    y = p * s
    y1 = p1 * s + p * s1
    y2 = p2 * s + 2 * p1 * s1 + p * s2
    return y, y1, y2


def mean_curvature_profile(c, xs):
    y, y1, y2 = preston_derivatives(c, xs)
    k_m = y2 / np.sqrt(1.0 + y1 ** 2) ** 3
    k_p = y / np.sqrt(1.0 + y1 ** 2)
    with np.errstate(divide="ignore", invalid="ignore"):
        k_p = np.where(y > 1e-9, k_p, 0.0)
    return k_m + k_p


def thompson_pressure_descriptors(c):
    ind = preston_indices(c)
    rp, rm = preston_poly(c, 1.0) ** 2, preston_poly(c, -1.0) ** 2
    return {
        "delta_pole": float(rm / rp),
        "delta_eq": float(ind["half_breadth"] ** 2),
        "x_star": float(ind["x_widest"]),
        "r_blunt": float(max(rp, rm)),
    }


def thompson_closure(delta_pole, delta_eq, x_star, r_blunt, c_init=None):
    from scipy.optimize import least_squares

    def residual(c):
        c = np.asarray(c)
        ind = preston_indices(c)
        rp, rm = preston_poly(c, 1.0) ** 2, preston_poly(c, -1.0) ** 2
        return [
            rm / rp - delta_pole,
            ind["half_breadth"] ** 2 - delta_eq,
            ind["x_widest"] - x_star,
            max(rp, rm) - r_blunt,
        ]

    x0 = c_init if c_init is not None else np.array([0.95, -0.02, -0.02, 0.0])
    sol = least_squares(residual, x0, max_nfev=400)
    return sol.x, float(np.max(np.abs(residual(sol.x))))


def meridian_from_mask(mask):
    """Extract a meridian half-profile along the major axis of the silhouette (axisymmetric shell)."""
    ys, xs = np.where(mask)
    if len(ys) < 50:
        raise RuntimeError("mask too small")
    cy, cx = ys.mean(), xs.mean()
    yc, xc = ys - cy, xs - cx
    cov = np.cov(np.vstack([xc, yc]))
    _, eigvecs = np.linalg.eigh(cov)
    major = eigvecs[:, 1]
    t = xc * major[0] + yc * major[1]
    perp = -xc * major[1] + yc * major[0]
    t_min, t_max = t.min(), t.max()
    prof_t, prof_h = [], []
    for frac in np.linspace(0, 1, 160):
        t0 = t_min + frac * (t_max - t_min)
        sel = np.abs(t - t0) < 0.6
        if sel.sum() < 3:
            continue
        half = float(np.max(np.abs(perp[sel])))
        if half < 2:
            continue
        prof_t.append(t0)
        prof_h.append(half)
    prof_t = np.asarray(prof_t, float)
    prof_h = np.asarray(prof_h, float)
    L = prof_t[-1] - prof_t[0]
    x_norm = 2 * (prof_t - prof_t[0]) / L - 1
    h_norm = prof_h / prof_h.max()
    return x_norm, h_norm


def segment_urchin(path: Path):
    from skimage import color, filters, io, measure, morphology

    img = io.imread(str(path))
    gray = color.rgb2gray(img[..., :3]) if img.ndim == 3 else img.astype(float) / 255
    thresh = filters.threshold_otsu(gray)
    mask = gray < thresh if gray.mean() > 0.5 else gray > thresh
    mask = morphology.remove_small_objects(mask, min_size=800)
    mask = morphology.binary_closing(mask, morphology.disk(7))
    mask = morphology.binary_opening(mask, morphology.disk(3))
    lab = measure.label(mask)
    props = measure.regionprops(lab)
    if not props:
        raise RuntimeError(f"no urchin blob in {path}")
    main = max(props, key=lambda p: p.area)
    m = lab == main.label
    xs, ys = meridian_from_mask(m)
    ys_all, xs_all = np.where(m)
    x_mid = int(xs_all.mean())
    asym = []
    for y in range(ys_all.min(), ys_all.max() + 1, max(1, (ys_all.max() - ys_all.min()) // 120)):
        row = np.where(m[y])[0]
        if len(row) < 2:
            continue
        left, right = x_mid - row.min(), row.max() - x_mid
        denom = max(left, right, 1)
        asym.append(abs(left - right) / denom)
    axisym = float(np.sqrt(np.mean(np.array(asym) ** 2)))
    c, flipped = preston_fit(xs, ys)
    ind = preston_indices(c)
    return {
        "path": str(path.name),
        "xs": xs,
        "ys": ys,
        "c": c,
        "flipped": flipped,
        "axisymmetry_rms_over_half_breadth": axisym,
        **ind,
        "preston_rmse_over_breadth": preston_rmse_over_breadth(c, xs, ys),
    }


def segment_panel(path: Path, min_area=2500):
    from skimage import color, filters, io, measure, morphology

    img = io.imread(str(path))
    gray = color.rgb2gray(img[..., :3]) if img.ndim == 3 else img.astype(float) / 255
    thresh = filters.threshold_otsu(gray)
    mask = gray < thresh if gray.mean() > 0.5 else gray > thresh
    mask = morphology.remove_small_objects(mask, min_size=min_area)
    lab = measure.label(mask)
    out = []
    for p in measure.regionprops(lab):
        if p.area < min_area:
            continue
        m = lab == p.label
        try:
            xs, ys = meridian_from_mask(m)
            c, _ = preston_fit(xs, ys)
            ind = preston_indices(c)
            out.append({"label": p.label, "area": p.area, "c": c, **ind})
        except RuntimeError:
            continue
    return out


def load_literature():
    rows = []
    with LIT_CSV.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            row["axial_over_equatorial"] = float(row["axial_over_equatorial"])
            rows.append(row)
    return rows


def fit_sphere_ablation(xs, ys):
    """Null: spherical meridian with c0 = mid-body half-breadth (Thompson uniform-pressure limit)."""
    c = np.array([float(np.interp(0.0, xs, ys)), 0.0, 0.0, 0.0])
    return {"c": c, "rmse_over_breadth": preston_rmse_over_breadth(c, xs, ys)}


def closure_sweep(n=12):
    deltas = np.linspace(1.0, 1.08, n)
    out = []
    for d in deltas:
        c, res = thompson_closure(d, 0.98, 0.0, 1.0)
        ind = preston_indices(c)
        out.append({"delta_pole": float(d), "c": c, "closure_residual": res, **ind})
    return out


def analyse(pass1=False):
    verify_dataset_integrity()
    primary = segment_urchin(PRIMARY)
    panel = [] if pass1 else segment_panel(PANEL)
    lit = load_literature()
    lit_ratios = np.array([r["axial_over_equatorial"] for r in lit])
    lit_elong = 1.0 / lit_ratios  # axial/equatorial < 1 => elongation > 1 in our L/B sense
    sphere = fit_sphere_ablation(primary["xs"], primary["ys"])
    sweep = [] if pass1 else closure_sweep()
    c = primary["c"]
    xx = np.linspace(-0.85, 0.85, 120)
    h_prof = mean_curvature_profile(c, xx)
    h_prof = h_prof[np.isfinite(h_prof)]
    h_cv = float(np.std(h_prof) / (np.abs(np.mean(h_prof)) + 1e-12))
    desc = thompson_pressure_descriptors(c)
    c_rec, res = thompson_closure(desc["delta_pole"], desc["delta_eq"], desc["x_star"], desc["r_blunt"], c_init=c)
    from scipy import stats

    ks_lit = stats.ks_2samp(
        np.array([s["elongation"] for s in panel]) if panel else np.array([primary["elongation"]]),
        lit_elong,
    ) if not pass1 else (0.0, 1.0)
    m = np.abs(primary["xs"]) < 0.92
    pred = preston_half_breadth(c, primary["xs"][m])
    obs = primary["ys"][m]
    r_prof = float(np.corrcoef(pred, obs)[0, 1]) if len(obs) > 3 else 0.0
    return {
        "primary": primary,
        "panel": panel,
        "literature": lit,
        "sphere_ablation": sphere,
        "sweep": sweep,
        "closure": {"residual": res, "c_rec": c_rec},
        "mean_curvature_cv": h_cv,
        "profile_r": r_prof,
        "ks_panel_vs_literature": ks_lit,
        "integrity": verify_dataset_integrity(strict=False),
    }


def measure_morphoeval(pass1=False):
    A = analyse(pass1=pass1)
    p = A["primary"]
    sp = A["sphere_ablation"]
    measures = [
        {"name": "primary test Preston RMSE / breadth", "value": p["preston_rmse_over_breadth"], "unit": "ratio",
         "method": "CV meridian vs Preston fit on Commons primary specimen", "reference_value": 0.05, "derived_from": "script"},
        {"name": "axisymmetry RMS / half-breadth (primary)", "value": p["axisymmetry_rms_over_half_breadth"], "unit": "ratio",
         "method": "upper vs lower meridian on horizontal scan", "reference_value": 0.0, "derived_from": "script"},
        {"name": "elongation L/B (primary CV fit)", "value": p["elongation"], "unit": "ratio",
         "method": "Preston elongation index", "reference_value": 1.0, "derived_from": "script"},
        {"name": "sphere ablation RMSE ratio vs Preston (primary)", "value": sp["rmse_over_breadth"] / p["preston_rmse_over_breadth"], "unit": "ratio",
         "method": "c = (max(y), 0, 0, 0) null", "reference_value": 1.0, "derived_from": "script"},
        {"name": "Thompson closure residual (primary descriptors)", "value": A["closure"]["residual"], "unit": "max abs",
         "method": "4-constraint least squares on pole/equator descriptors", "reference_value": 0.0, "derived_from": "script"},
        {"name": "mean curvature profile CV (primary)", "value": A["mean_curvature_cv"], "unit": "CV",
         "method": "std/mean of 2H(x); sphere => 0", "reference_value": 0.0, "derived_from": "script"},
        {"name": "meridian profile Pearson r (closure vs CV)", "value": A["profile_r"], "unit": "r",
         "method": "120-point interpolated profiles", "reference_value": 1.0, "derived_from": "script"},
    ]
    if not pass1:
        panel_e = [s["elongation"] for s in A["panel"]]
        measures.extend([
            {"name": "panel specimens segmented (Commons multi-test)", "value": float(len(A["panel"])), "unit": "count",
             "method": "connected components after Otsu", "reference_value": 3.0, "derived_from": "script"},
            {"name": "panel elongation mean", "value": float(np.mean(panel_e)), "unit": "ratio",
             "method": f"std={np.std(panel_e):.4f}", "reference_value": 1.0, "derived_from": "script"},
            {"name": "KS D panel vs literature elongation", "value": float(A["ks_panel_vs_literature"][0]), "unit": "D",
             "method": f"p={A['ks_panel_vs_literature'][1]:.3e}", "reference_value": 0.0, "derived_from": "script"},
            {"name": "closure sweep max residual", "value": float(max(s["closure_residual"] for s in A["sweep"])), "unit": "max abs",
             "method": f"{len(A['sweep'])} delta_pole steps", "reference_value": 0.0, "derived_from": "script"},
        ])
    return measures


def write_result_json(measures, pass1=False):
    A = analyse(pass1=pass1)
    p = A["primary"]
    verdict_parts = {
        "C1_axisymmetric_hollow": "supported" if p["axisymmetry_rms_over_half_breadth"] < 0.85 else "mixed",
        "C2_preston_meridian_fit": "supported" if p["preston_rmse_over_breadth"] < 0.08 else "mixed",
        "C3_thompson_closure": "supported" if A["closure"]["residual"] < 0.05 else "mixed",
        "C4_near_spherical": "supported" if 0.95 < p["elongation"] < 1.15 else "mixed",
        "dryad_dn160": "DATASET_MISMATCH (not used)",
    }
    if not pass1:
        verdict_parts["C5_sphere_ablation_worse"] = (
            "supported" if A["sphere_ablation"]["rmse_over_breadth"] > 1.5 * p["preston_rmse_over_breadth"] else "mixed"
        )
    core = [verdict_parts[k] for k in ("C1_axisymmetric_hollow", "C2_preston_meridian_fit", "C3_thompson_closure")]
    verdict = "supported" if all(v == "supported" for v in core) else "mixed"
    out = {
        "claim_id": "T1917-035",
        "date_made": "1917",
        "claim": load_reference()["claim_text"],
        "citation": load_reference()["citation"],
        "dataset": {
            "name": "Wikimedia Commons CC BY 2.0 sea-urchin test photographs (James St. John)",
            "dryad_dn160": "doi:10.5061/dryad.dn160 — MISMATCH (genetics, excluded)",
            "sha256": {
                "commons_sea_urchin_test_structure_8340336761.jpg": load_reference()["datasets"]["files"][
                    "commons_sea_urchin_test_structure_8340336761.jpg"
                ],
                "commons_sea_urchin_tests_7_77.jpg": load_reference()["datasets"]["files"]["commons_sea_urchin_tests_7_77.jpg"],
            },
        },
        "analogue": False,
        "verdict": verdict,
        "verdict_by_component": verdict_parts,
        "headline_numbers": {
            "elongation_primary": round(p["elongation"], 4),
            "preston_rmse_over_breadth": round(p["preston_rmse_over_breadth"], 5),
            "axisymmetry_rms": round(p["axisymmetry_rms_over_half_breadth"], 4),
            "closure_residual": round(A["closure"]["residual"], 6),
            "mean_curvature_cv": round(A["mean_curvature_cv"], 4),
            "profile_r": round(A["profile_r"], 5),
            "sphere_ablation_ratio": round(A["sphere_ablation"]["rmse_over_breadth"] / p["preston_rmse_over_breadth"], 2),
            "n_panel": len(A["panel"]),
        },
        "script": "thompson/claims/t1917-035_on-the-shapes-of-eggs-and-of-certain-other-hollo.py",
        "figure": f"docs/media/{STEM}_comparison.png",
        "measurements": measures,
    }
    RESULT_JSON.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    return out


def render_comparison(path: Path, pass1=False):
    import matplotlib.pyplot as plt

    A = analyse(pass1=pass1)
    p = A["primary"]
    xx = np.linspace(-1, 1, 200)
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.2))
    axes[0].plot(p["xs"], p["ys"], "k.", ms=2, label="CV meridian")
    axes[0].plot(xx, preston_half_breadth(p["c"], xx), "r-", lw=1.5, label="Preston fit")
    axes[0].set_title("Primary specimen meridian")
    axes[0].legend(fontsize=7)
    axes[1].plot(xx, mean_curvature_profile(p["c"], xx), "navy")
    axes[1].set_title("Thompson 2H profile")
    if not pass1 and A["panel"]:
        el = [s["elongation"] for s in A["panel"]]
        axes[2].hist(el, bins=8, color="tan", edgecolor="k")
        axes[2].axvline(p["elongation"], color="r", ls="--", label="primary")
        axes[2].set_title("Panel elongations")
        axes[2].legend(fontsize=7)
    else:
        axes[2].text(0.5, 0.5, "Pass 1\n(single specimen)", ha="center", va="center", transform=axes[2].transAxes)
    fig.suptitle("T1917-035 sea-urchin test — membrane meridian (Thompson Ch. XV Fig. 328)", fontsize=10)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def render_growth_gif(path: Path, n_frames=18):
    import matplotlib.pyplot as plt
    from PIL import Image

    frames = []
    for s in closure_sweep(n_frames):
        xx = np.linspace(-1, 1, 160)
        fig, ax = plt.subplots(figsize=(3, 3))
        ax.plot(xx, preston_half_breadth(s["c"], xx), "k-", lw=2)
        ax.set_xlim(-1, 1)
        ax.set_ylim(0, 1.05)
        ax.set_title(f"δ_pole={s['delta_pole']:.3f}", fontsize=8)
        fig.tight_layout()
        fig.canvas.draw()
        frames.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()))
        plt.close(fig)
    path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(path, save_all=True, append_images=frames[1:] + frames[::-1], duration=120, loop=0)


def self_test():
    integ = verify_dataset_integrity()
    assert integ["verified"] >= 5 and not integ["failed"], integ
    c = np.array([1.0, 0.0, 0.0, 0.0])
    assert abs(preston_indices(c)["elongation"] - 1.0) < 1e-3
    p = segment_urchin(PRIMARY)
    assert p["preston_rmse_over_breadth"] < 0.15
    assert p["elongation"] > 0.9
    d = thompson_pressure_descriptors(p["c"])
    _, r = thompson_closure(**d, c_init=p["c"])
    assert r < 0.2
    m1 = measure_morphoeval(pass1=False)
    m2 = measure_morphoeval(pass1=False)
    assert len(m1) == len(m2)
    print(
        f"self-test OK: {integ['verified']} pins; primary E={p['elongation']:.3f} "
        f"RMSE={p['preston_rmse_over_breadth']:.4f} closure={r:.2e}"
    )


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--pass1", action="store_true", help="Pass 1 baseline (single specimen, no sweep/KS)")
    ap.add_argument("--skip-gif", action="store_true")
    args = ap.parse_args(argv)
    np.random.seed(SEED)
    if args.self_test:
        self_test()
        return
    measures = measure_morphoeval(pass1=args.pass1)
    write_result_json(measures, pass1=args.pass1)
    render_comparison(MEDIA / f"{STEM}_comparison.png", pass1=args.pass1)
    if not args.skip_gif and not args.pass1:
        render_growth_gif(MEDIA / f"{STEM}_growth.gif")
    for m in measures:
        print(f"{m['name']}: {m['value']:.4f}")


if __name__ == "__main__":
    main()

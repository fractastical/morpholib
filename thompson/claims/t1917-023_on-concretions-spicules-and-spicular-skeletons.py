#!/usr/bin/env python3
"""
t1917-023_on-concretions-spicules-and-spicular-skeletons.py — Thompson (1917) claim T1917-023
=============================================================================================

Claim (Thompson, D.W. 1917, *On Growth and Form*, Ch. IX "On Concretions, Spicules, and
Spicular Skeletons", pp. 411-487, Figs. 194-235):

    "Concretions and calcospherites grow by radial deposition about a centre (Harting's
     artificial concretions in egg albumen mimic natural ones), giving concentric/spherical
     forms by physical deposition alone."

Pass 2 (v2, post-remediation).  Mechanistic-geometric model of radial deposition, a full
parameter-regime sweep, negative / ablation controls, computer-vision segmentation of
Thompson's own 1917 plates (IA scan `ongrowthform1917thom`, pp. 421 / 422 / 432) and of two
labelled analogue specimens, and goodness-of-fit statistics (Pearson r, RMSE, two-sample KS, CV).

Model (reference JSON `datasets/thompson_1917/t1917-023/..._reference.json`, Lean module
`morphoeval/lean/loops/Thompson1917_T1917_023.lean`, namespace `MathOfBiology.Loop_218`):

  E1  interface-controlled radial deposition  dR/dt = v           Thompson pp. 421-422, 437
      -> constant lamella thickness            (Lean lamella_thickness_constant, radial_deposition_semigroup)
  E2  diffusion-limited radial deposition     R dR/dt = kappa     Zener 1949; Mullins & Sekerka 1963
      -> lamellae thin outward                 (Lean parabolic_shell_identity, lamellae_thin_outward)
  E3  sphericity stability: uniform normal deposition keeps max rho - min rho fixed, so the
      asphericity A decays as 1/R              (Lean absolute_asphericity_conserved, asphericity_nonincreasing)
      CONTROL: anisotropic (chessboard-metric) deposition does not round off          Thompson p. 434
  E4  equal-rate impingement -> perpendicular-bisector planes -> Voronoi mosaic       Thompson pp. 422, 426
                                               (Lean impingement_locus_is_affine, midpoint_on_impingement_plane)
  E5  Liesegang supersaturation banding in d-dimensional radial geometry, r_{n+1}/r_n -> P   Thompson p. 427
                                               (Lean band_ratio_constant, geometric_band_gap_grows, band_gap_monotone)
      CONTROL: no nucleation gap (K_ss = K_sp) -> continuous deposit, no discrete bands
  E6  dielectric-breakdown supply sweep eta in {0, 0.5, 1, 1.5, 2}: compact Eden disc (eta = 0)
      -> spinous DLA (eta >= 1)                 Thompson p. 424 "scanty supply -> spinous / digitiform"

Specimen measurements (all SHA-256-locked files in `datasets/thompson_1917/t1917-023/`):
  Fig. 196  single calcospherite: outer-boundary radius CV, nucleus/outer ratio, striae count,
            radial-line index of the striated border (Hessian ridge orientation vs radial direction)
  Fig. 195  calcospherites in albumen: radial-line index around 6 manually landmarked nuclei vs a
            random-centre null
  Fig. 197  crowded calcospherites: wall segmentation -> cells, neighbour count, area CV, compared
            with an equal-rate Voronoi mosaic rendered and measured by the *same* pipeline (KS test)
  Fig. 208  concentrically striated spherocrystal: dark-ring radii on 12 rays from the nucleus,
            ratio r_{n+1}/r_n, geometric vs arithmetic spacing fit
  ooids     (ANALOGUE, Jurassic Carmel Fm.) nucleus vs outer-boundary asphericity (E3 test), cortex
            thickness, lamella count, normal-line index
  Liesegang (ANALOGUE, Ag2CrO4 in gelatine) ring radii from the azimuthal mean profile, Jablczynski P,
            geometric vs arithmetic fits (r, RMSE), monotone gap growth (Spearman), KS vs E5 simulation

Usage
-----
    python3 thompson/claims/t1917-023_on-concretions-spicules-and-spicular-skeletons.py --self-test
    python3 thompson/claims/t1917-023_on-concretions-spicules-and-spicular-skeletons.py [--quick] [--no-gif]
        [--etas 0 0.5 1 1.5 2] [--b0s 0.05 0.1 0.2] [--a0 50] [--seed 1917] [--no-result]

Outputs
-------
    docs/media/t1917_023_on_concretions_spicules_and_sp_comparison.png
    docs/media/t1917_023_on_concretions_spicules_and_sp_growth.gif
    datasets/thompson_1917/t1917-023/result.json          (claim verdict)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import warnings
from pathlib import Path

import numpy as np
from scipy import ndimage, signal, stats
from scipy.spatial import ConvexHull

ROOT = Path(__file__).resolve().parents[2]
DATASET_DIR = ROOT / "datasets" / "thompson_1917" / "t1917-023"
REFERENCE_JSON = DATASET_DIR / "t1917_023_on_concretions_spicules_and_sp_reference.json"
RESULT_JSON = DATASET_DIR / "result.json"
MEDIA_DIR = ROOT / "docs" / "media"
COMPARISON_PNG = MEDIA_DIR / "t1917_023_on_concretions_spicules_and_sp_comparison.png"
GROWTH_GIF = MEDIA_DIR / "t1917_023_on_concretions_spicules_and_sp_growth.gif"

PLATE_P421 = "thompson_1917_p421_figs195-196_harting_calcospherites.jpg"
PLATE_P422 = "thompson_1917_p422_fig197_aggregated_calcospherites.jpg"
PLATE_P432 = "thompson_1917_p432_fig208-209_spherocrystal_otoliths.jpg"
OOID_SPECIMEN = "carmel_ooids_thin_section_specimen.jpg"
LIESEGANG_SPECIMEN = "liesegang_rings_silver_chromate_gelatine_specimen.jpg"

# Pixel boxes (left, top, right, bottom) of the figures inside the full IA page scans (1904 x 3128).
PLATE_CROPS = {
    "fig195": (PLATE_P421, (230, 1140, 900, 1825)),
    "fig196": (PLATE_P421, (1085, 1355, 1565, 1825)),
    "fig197R": (PLATE_P422, (1085, 535, 1635, 1150)),
    "fig208": (PLATE_P432, (215, 1360, 500, 1700)),
    "fig209": (PLATE_P432, (255, 1905, 925, 2435)),
}
# Manual landmarks (crop pixel coordinates, x, y). The nuclei of Fig. 195 are low-contrast stipple
# ellipses that no blob detector separated from the dark lobulated rims, so their centres were
# annotated by eye once and are pinned here; every measurement around them is automated.
MANUAL_LANDMARKS = {
    "fig195_nuclei_xy": [(195, 150), (455, 135), (470, 245), (230, 395), (470, 500), (205, 655)],
    "fig195_annulus_px": (60, 115),
    "fig208_nucleus_xy": (133, 130),
}

SEED = 1917
PARAMS = {
    "front": {"grid": 241, "seed_half_width": 12, "n_frames": 24},
    "voronoi": {"n_centres": 60, "grid": 450, "wall_px": 8, "min_sep_frac": 0.6},
    "liesegang": {"D": 1.0, "a0": 50.0, "b0": 0.1, "K_sp": 0.005, "K_ss": 0.05, "k": 5.0,
                  "r0": 2.0, "r_max": 160.0, "dr": 0.5, "dt": 0.05, "t_end": 2500.0},
    "liesegang_b0_sweep": [0.05, 0.1, 0.2, 0.4],
    "liesegang_3d_a0": 100.0,
    "dbm": {"grid": 161, "n_sites": 1200, "etas": [0.0, 0.5, 1.0, 1.5, 2.0]},
}

_CACHE: dict = {}


# =============================================================================== dataset
def load_reference() -> dict:
    return json.loads(REFERENCE_JSON.read_text(encoding="utf-8"))


def verify_dataset_integrity() -> dict:
    """SHA-256 check of every vendored plate / specimen against the reference JSON."""
    ref = load_reference()
    report = {}
    for name, meta in ref["vendored_files"].items():
        p = DATASET_DIR / name
        if not p.exists():
            report[name] = "missing"
            continue
        h = hashlib.sha256(p.read_bytes()).hexdigest()
        report[name] = "ok" if h == meta["sha256"] else f"sha256 mismatch ({h[:12]}…)"
    return report


def load_plate(key: str) -> np.ndarray:
    from PIL import Image
    fname, box = PLATE_CROPS[key]
    with Image.open(DATASET_DIR / fname) as im:
        return np.asarray(im.convert("L").crop(box), dtype=float) / 255.0


def load_specimen_rgb(name: str) -> np.ndarray:
    from PIL import Image
    with Image.open(DATASET_DIR / name) as im:
        return np.asarray(im.convert("RGB"), dtype=float) / 255.0


def _disk(r: int) -> np.ndarray:
    y, x = np.ogrid[-r:r + 1, -r:r + 1]
    return (x * x + y * y) <= r * r


# =============================================================================== E1 / E2
def radial_growth_laws(R0: float = 2.0, v: float = 1.0, kappa: float = 8.0, n: int = 12) -> dict:
    """E1 (linear) and E2 (parabolic) radii at equal deposition intervals and the lamella
    thicknesses they imply (Lean: lamella_thickness_constant, lamellae_thin_outward)."""
    k = np.arange(n + 1, dtype=float)
    R_lin = R0 + v * k
    R_par = np.sqrt(R0 ** 2 + 2.0 * kappa * k)
    return {
        "k": k, "R_lin": R_lin, "R_par": R_par,
        "lamella_lin": np.diff(R_lin), "lamella_par": np.diff(R_par),
        "lamella_lin_cv": float(np.std(np.diff(R_lin)) / np.mean(np.diff(R_lin))),
        "lamella_par_monotone_decreasing": bool(np.all(np.diff(np.diff(R_par)) < 0)),
        "lamella_par_cv": float(np.std(np.diff(R_par)) / np.mean(np.diff(R_par))),
    }


# =============================================================================== E3
def simulate_front_rounding(grid: int | None = None, seed_half_width: int | None = None,
                            n_frames: int | None = None, metric: str = "euclidean") -> dict:
    """Normal deposition on a square nucleus: the front at 'radius' R is a distance-transform
    level set.  `metric='euclidean'` is uniform normal deposition (E3); `'chessboard'` is the
    anisotropic crystalline control (growth rate depends on direction, square stays square).
    Returns asphericity A(R) = (max rho - min rho)/mean rho and its log-log decay exponent."""
    g = grid or PARAMS["front"]["grid"]
    hw = seed_half_width or PARAMS["front"]["seed_half_width"]
    nf = n_frames or PARAMS["front"]["n_frames"]
    c = g // 2
    seed = np.zeros((g, g), bool)
    seed[c - hw:c + hw + 1, c - hw:c + hw + 1] = True
    if metric == "euclidean":
        dist = ndimage.distance_transform_edt(~seed)
    else:
        dist = ndimage.distance_transform_cdt(~seed, metric=metric).astype(float)
    yy, xx = np.mgrid[0:g, 0:g]
    rr = np.hypot(yy - c, xx - c)
    levels = np.linspace(0, c - hw - 3, nf)
    R_mean, A, masks = [], [], []
    for lv in levels:
        m = dist <= lv
        edge = m & ~ndimage.binary_erosion(m)
        rho = rr[edge]
        R_mean.append(rho.mean())
        A.append((rho.max() - rho.min()) / rho.mean())
        masks.append(m)
    R_mean, A = np.array(R_mean), np.array(A)
    sel = levels > 0
    slope, _, r_fit, _, _ = stats.linregress(np.log(R_mean[sel]), np.log(np.maximum(A[sel], 1e-6)))
    return {"levels": levels, "R_mean": R_mean, "asphericity": A, "masks": masks, "metric": metric,
            "decay_exponent": float(slope), "loglog_r": float(r_fit),
            "abs_asphericity_range": float(np.ptp((np.array([m.sum() for m in masks]) * 0) + 0) if False else
                                           (A * R_mean).std() / (A * R_mean).mean())}


# =============================================================================== E4
def simulate_voronoi_impingement(n_centres: int | None = None, grid: int | None = None,
                                 rng_seed: int = SEED, min_sep_frac: float | None = None) -> dict:
    """Simultaneous nucleation + equal radial growth rate = Voronoi tessellation of the centres
    (Lean impingement_locus_is_affine). `min_sep_frac` = 0 gives Poisson centres; > 0 gives a
    hard-core (sequential-inhibition) process with minimum separation min_sep_frac * mean spacing,
    i.e. spherites that had grown to similar sizes before impinging (Thompson Fig. 195 -> 197).
    Returns label map, exact per-cell areas and neighbour counts of interior cells, plus a
    rendered wall image for the same-pipeline comparison with the plate."""
    n = n_centres or PARAMS["voronoi"]["n_centres"]
    g = grid or PARAMS["voronoi"]["grid"]
    wall = PARAMS["voronoi"]["wall_px"]
    msf = PARAMS["voronoi"]["min_sep_frac"] if min_sep_frac is None else min_sep_frac
    rng = np.random.default_rng(rng_seed)
    if msf <= 0:
        pts = rng.uniform(0, g, size=(n, 2))
    else:
        dmin = msf * g / np.sqrt(n); pts = []
        for _ in range(200000):
            q = rng.uniform(0, g, size=2)
            if not pts or np.min(np.hypot(*(np.array(pts) - q).T)) >= dmin:
                pts.append(q)
            if len(pts) == n:
                break
        pts = np.array(pts)
    yy, xx = np.mgrid[0:g, 0:g]
    d2 = (xx[None] - pts[:, 0, None, None]) ** 2 + (yy[None] - pts[:, 1, None, None]) ** 2
    labels = np.argmin(d2, axis=0) + 1
    areas, neighbours = [], []
    border = set(np.unique(np.r_[labels[0], labels[-1], labels[:, 0], labels[:, -1]]))
    for lab in range(1, n + 1):
        if lab in border:
            continue
        m = labels == lab
        ring = ndimage.binary_dilation(m, iterations=2) & ~m
        neighbours.append(len(np.unique(labels[ring])))
        areas.append(m.sum())
    areas, neighbours = np.array(areas, float), np.array(neighbours, float)
    walls = ndimage.maximum_filter(labels, size=wall) != ndimage.minimum_filter(labels, size=wall)
    rendered = np.where(walls, 0.22, 0.72) + 0.02 * rng.standard_normal(labels.shape)
    return {"labels": labels, "points": pts, "areas": areas, "neighbours": neighbours, "min_sep_frac": msf,
            "mean_neighbours": float(neighbours.mean()), "area_cv": float(areas.std() / areas.mean()),
            "hexagon_fraction": float((neighbours == 6).mean()),
            "n_interior": int(len(areas)), "rendered": np.clip(rendered, 0, 1)}


# =============================================================================== E5
def simulate_liesegang_radial(dim: int = 2, threshold: bool = True, **over) -> dict:
    """Ostwald supersaturation (Keller-Rubinow type) precipitation in d-dimensional radial
    geometry: inner electrolyte reservoir a0 at r <= r0 diffuses into gel holding b0.
    Precipitation R = k (ab - K_sp)_+ on existing precipitate or where ab >= K_ss (nucleation).
    `threshold=False` removes the nucleation gap (K_ss = K_sp): the continuous-deposit control."""
    P = dict(PARAMS["liesegang"]); P.update({k: v for k, v in over.items() if v is not None})
    D, a0, b0, Ksp, Kss, k = P["D"], P["a0"], P["b0"], P["K_sp"], P["K_ss"], P["k"]
    r0, rmax, dr, dt, t_end = P["r0"], P["r_max"], P["dr"], P["dt"], P["t_end"]
    r = np.arange(dr / 2, rmax, dr); n = len(r)
    a = np.zeros(n); b = np.full(n, b0); p = np.zeros(n)
    inner = r <= r0; a[inner] = a0; b[inner] = 0.0
    rf = np.arange(0, rmax + dr, dr)[:n + 1]
    gexp = dim - 1
    rfg, rcg = rf ** gexp, r ** gexp
    nsteps = int(round(t_end / dt))
    for _ in range(nsteps):
        for c, far in ((a, 0.0), (b, None)):
            flux = np.zeros(n + 1)
            flux[1:n] = -D * (c[1:] - c[:-1]) / dr
            flux[n] = 0.0 if far is None else -D * (far - c[-1]) / dr
            c += dt * (-(rfg[1:] * flux[1:] - rfg[:-1] * flux[:-1]) / (rcg * dr))
        a[inner] = a0; b[inner] = 0.0
        prod = a * b
        grow = (prod > Ksp) & ((p > 0) | (prod >= Kss)) if threshold else (prod > Ksp)
        dep = np.minimum(k * np.where(grow, prod - Ksp, 0.0) * dt, np.minimum(a, b))
        a -= dep; b -= dep; p += dep; a[inner] = a0
    bands = _band_centres(r, p)
    ratios = bands[1:] / bands[:-1] if len(bands) > 1 else np.array([])
    out = {"r": r, "p": p, "a": a, "b": b, "bands": bands, "ratios": ratios,
           "spacing_coefficient": float(ratios.mean()) if len(ratios) else float("nan"),
           "spacing_cv": float(ratios.std() / ratios.mean()) if len(ratios) > 1 else float("nan"),
           "n_bands": int(len(bands)), "dim": dim, "threshold": threshold, "params": P,
           "deposit_fraction_of_domain": float((p > 0.1 * p.max()).mean()) if p.max() > 0 else 0.0}
    out.update(spacing_fits(bands))
    return out


def _band_centres(r: np.ndarray, p: np.ndarray, frac: float = 0.1) -> np.ndarray:
    mask = p > frac * p.max() if p.max() > 0 else np.zeros_like(p, bool)
    out, inb, start = [], False, 0
    for i in range(len(mask)):
        if mask[i] and not inb:
            start, inb = i, True
        if (not mask[i] or i == len(mask) - 1) and inb:
            seg = slice(start, i if not mask[i] else i + 1)
            w = p[seg]
            out.append(float((r[seg] * w).sum() / w.sum()))
            inb = False
    return np.array(out)


def spacing_fits(radii: np.ndarray) -> dict:
    """Geometric (ln r_n = a + n ln P; Thompson's 'logarithmic ratio') vs arithmetic (r_n = a + n d)
    fits of band radii, with Pearson r and RMSE in the units of `radii`, and the Spearman trend of
    successive gaps ('stand farther and farther apart')."""
    radii = np.asarray(radii, float)
    if len(radii) < 3:
        return {"geo_P": float("nan"), "geo_r": float("nan"), "geo_rmse": float("nan"), "geo_r0": float("nan"),
                "arith_d": float("nan"), "arith_r": float("nan"), "arith_rmse": float("nan"), "arith_r0": float("nan"),
                "gap_mean": float("nan"), "gap_cv": float("nan"),
                "gap_spearman_rho": float("nan"), "gap_spearman_p": float("nan")}
    n = np.arange(len(radii))
    sl, ic, rg, _, _ = stats.linregress(n, np.log(radii))
    pred_g = np.exp(ic + sl * n)
    sl2, ic2, ra, _, _ = stats.linregress(n, radii)
    pred_a = ic2 + sl2 * n
    gaps = np.diff(radii)
    rho, pval = stats.spearmanr(np.arange(len(gaps)), gaps) if len(gaps) >= 3 else (float("nan"), float("nan"))
    return {"geo_P": float(np.exp(sl)), "geo_r": float(rg), "geo_rmse": float(np.sqrt(np.mean((pred_g - radii) ** 2))), "geo_r0": float(np.exp(ic)),
            "arith_d": float(sl2), "arith_r": float(ra), "arith_rmse": float(np.sqrt(np.mean((pred_a - radii) ** 2))), "arith_r0": float(ic2),
            "gap_mean": float(gaps.mean()), "gap_cv": float(gaps.std() / gaps.mean()),
            "gap_spearman_rho": float(rho), "gap_spearman_p": float(pval)}


# =============================================================================== E6
def simulate_dbm_cluster(eta: float, n_sites: int | None = None, grid: int | None = None,
                         rng_seed: int = SEED, relax_iters: int = 12) -> dict:
    """Dielectric-breakdown model (Niemeyer, Pietronero & Wiesmann 1984): Laplacian supply field
    phi (0 on the cluster, 1 on a ring 8 lattice units beyond the cluster), perimeter site i grows
    with probability ∝ phi_i^eta.  eta = 0 is the Eden model (interface-controlled, compact);
    eta = 1 is DLA (diffusion-limited, branched); eta > 1 is supply-starved."""
    N = n_sites or PARAMS["dbm"]["n_sites"]
    L = grid or PARAMS["dbm"]["grid"]
    rng = np.random.default_rng(rng_seed + int(round(eta * 1000)))
    c = L // 2
    cl = np.zeros((L, L), bool); cl[c, c] = True
    phi = np.ones((L, L))
    yy, xx = np.mgrid[0:L, 0:L]
    rr = np.hypot(yy - c, xx - c)
    kern = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
    Rmax = 0.0

    def relax(nit):
        outer = rr >= Rmax + 8
        for _ in range(nit):
            new = 0.25 * (np.roll(phi, 1, 0) + np.roll(phi, -1, 0) + np.roll(phi, 1, 1) + np.roll(phi, -1, 1))
            new[cl] = 0.0; new[outer] = 1.0
            phi[:] = new

    relax(400)
    for _ in range(N - 1):
        if Rmax + 9 >= c:
            break
        relax(relax_iters)
        per = ndimage.binary_dilation(cl, kern) & ~cl
        ys, xs = np.nonzero(per)
        w = np.maximum(phi[ys, xs], 1e-12) ** eta if eta > 0 else np.ones(len(ys))
        i = rng.choice(len(ys), p=w / w.sum())
        cl[ys[i], xs[i]] = True; phi[ys[i], xs[i]] = 0.0
        Rmax = max(Rmax, rr[ys[i], xs[i]])
    return {"cluster": cl, "eta": eta, **cluster_morphometrics(cl)}


def cluster_morphometrics(cl: np.ndarray) -> dict:
    L = cl.shape[0]; c = L // 2
    ys, xs = np.nonzero(cl)
    r = np.hypot(ys - c, xs - c)
    rs = np.exp(np.linspace(np.log(3), np.log(max(0.75 * r.max(), 4)), 10))
    Nr = np.array([(r <= s).sum() for s in rs])
    Df = float(np.polyfit(np.log(rs), np.log(Nr), 1)[0])
    th = np.arctan2(ys - c, xs - c)
    bins = np.floor((th + np.pi) / (2 * np.pi) * 36).astype(int) % 36
    env = np.array([r[bins == b].max() if (bins == b).any() else 0.0 for b in range(36)])
    hull = ConvexHull(np.c_[xs, ys])
    return {"n_sites": int(len(xs)), "mass_radius_exponent": Df,
            "envelope_radius_cv": float(env.std() / env.mean()),
            "solidity": float(len(xs) / hull.volume), "R_max": float(r.max()),
            "R_gyration": float(np.sqrt((r ** 2).mean()))}


# =============================================================================== computer vision
def ridge_orientation(im: np.ndarray, sigma: float) -> tuple[np.ndarray, np.ndarray]:
    """Hessian ridge filter: strength of dark lines (largest eigenvalue, >0 for a dark line on a
    brighter ground) and the orientation *along* the line."""
    Hxx = ndimage.gaussian_filter(im, sigma, order=(0, 2))
    Hyy = ndimage.gaussian_filter(im, sigma, order=(2, 0))
    Hxy = ndimage.gaussian_filter(im, sigma, order=(1, 1))
    tmp = np.sqrt(((Hxx - Hyy) / 2) ** 2 + Hxy ** 2)
    l1 = np.clip((Hxx + Hyy) / 2 + tmp, 0, None) * sigma * sigma
    along = 0.5 * np.arctan2(2 * Hxy, Hxx - Hyy) + np.pi / 2
    return l1, along


def line_index(im: np.ndarray, mask: np.ndarray, ref_angle: np.ndarray, sigma: float = 2.0) -> float:
    """Radial-/normal-line index = <cos 2(theta_line - theta_ref)> weighted by ridge strength:
    +1 all dark lines parallel to the reference direction (radial fibres), -1 all perpendicular
    (concentric lamellae), 0 isotropic."""
    w, along = ridge_orientation(im, sigma)
    ww = w[mask]
    return float((ww * np.cos(2 * (along[mask] - ref_angle[mask]))).sum() / max(ww.sum(), 1e-12))


def _polar(shape, cx, cy):
    yy, xx = np.mgrid[0:shape[0], 0:shape[1]]
    return np.hypot(yy - cy, xx - cx), np.arctan2(yy - cy, xx - cx)


def measure_fig196() -> dict:
    """Thompson Fig. 196 (after Harting): single calcospherite on a dark ground."""
    im = load_plate("fig196")
    sm = ndimage.gaussian_filter(im, 4)
    mu = ndimage.uniform_filter(im, 9); lstd = np.sqrt(np.clip(ndimage.uniform_filter(im ** 2, 9) - mu ** 2, 0, None))
    dark = sm < 0.35
    lab, n = ndimage.label(dark); sizes = ndimage.sum(dark, lab, range(1, n + 1)); frame = lab == (1 + np.argmax(sizes))
    ys, xs = np.nonzero(frame); y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    inner = np.zeros_like(dark); inner[y0 + 8:y1 - 8, x0 + 8:x1 - 8] = True
    obj = inner & ((sm > 0.30) | (lstd > 0.12))
    obj = ndimage.binary_fill_holes(ndimage.binary_closing(obj, iterations=3))
    lab, n = ndimage.label(obj); sizes = ndimage.sum(obj, lab, range(1, n + 1)); disc = lab == (1 + np.argmax(sizes))
    disc = ndimage.binary_opening(disc, iterations=4)
    cy, cx = ndimage.center_of_mass(disc)
    rr, tr = _polar(im.shape, cx, cy)
    nb = 180; ab = np.floor((tr + np.pi) / (2 * np.pi) * nb).astype(int) % nb
    Rout = np.array([rr[disc & (ab == b)].max() for b in range(nb)])
    # nucleus / border boundary from the angular-mean texture-energy profile
    rbins = np.arange(0, Rout.mean() + 2, 2.0); ridx = np.digitize(rr, rbins)
    lprof = np.array(ndimage.mean(lstd, ridx, range(1, len(rbins))))
    half = 0.5 * (np.nanmax(lprof[:-3]) + np.nanmin(lprof[: len(lprof) // 2]))
    k = int(np.argmax(lprof > half)); R_nuc = float(rbins[max(k - 1, 0)])
    R = float(Rout.mean())
    border = disc & (rr > R_nuc * 1.08) & (rr < 0.95 * R)
    core = disc & (rr < 0.8 * R_nuc)
    rli_border = line_index(im, border, tr, 2.0)
    rli_core = line_index(im, core, tr, 2.0)
    # striae count: dominant angular harmonic of the mid-border intensity
    ann = disc & (rr > R_nuc * 1.1) & (rr < 0.93 * R); nb2 = 720
    ab2 = np.floor((tr[ann] + np.pi) / (2 * np.pi) * nb2).astype(int) % nb2
    sig = np.bincount(ab2, weights=im[ann], minlength=nb2) / np.maximum(np.bincount(ab2, minlength=nb2), 1)
    sig = sig - ndimage.uniform_filter1d(sig, 61, mode="wrap")
    F = np.abs(np.fft.rfft(sig)); striae = int(np.argmax(F[20:200]) + 20)
    edge = disc & ~ndimage.binary_erosion(disc)
    return {"centre_xy": (float(cx), float(cy)), "R_outer_px": R, "outer_radius_cv": float(Rout.std() / Rout.mean()),
            "outer_max_over_min": float(Rout.max() / Rout.min()), "R_nucleus_px": R_nuc,
            "nucleus_over_outer": R_nuc / R, "radial_line_index_border": rli_border,
            "radial_line_index_core": rli_core, "striae_count": striae,
            "mean_striae_spacing_px": float(2 * np.pi * 0.5 * (R_nuc + R) / striae),
            "circularity_4piA_P2": float(4 * np.pi * disc.sum() / max((edge.sum() * 1.0) ** 2, 1)),
            "disc": disc, "edge": edge, "image": im}


def measure_fig195(rng_seed: int = SEED, n_null: int = 100) -> dict:
    """Thompson Fig. 195: radial-line index in an annulus about each landmarked nucleus versus a
    random-centre null."""
    im = load_plate("fig195")
    r_in, r_out = MANUAL_LANDMARKS["fig195_annulus_px"]
    w, along = ridge_orientation(im, 2.0)

    def idx(cx, cy):
        rr, tr = _polar(im.shape, cx, cy); ann = (rr > r_in) & (rr < r_out); ww = w[ann]
        return float((ww * np.cos(2 * (along[ann] - tr[ann]))).sum() / ww.sum())

    vals = np.array([idx(cx, cy) for cx, cy in MANUAL_LANDMARKS["fig195_nuclei_xy"]])
    rng = np.random.default_rng(rng_seed)
    null = np.array([idx(rng.uniform(130, im.shape[1] - 130), rng.uniform(130, im.shape[0] - 130)) for _ in range(n_null)])
    z = (vals.mean() - null.mean()) / (null.std() / np.sqrt(len(vals)))
    return {"radial_line_index_per_nucleus": vals, "mean": float(vals.mean()), "sd": float(vals.std()),
            "null_mean": float(null.mean()), "null_sd": float(null.std()), "z_vs_null": float(z),
            "p_one_sided": float(stats.norm.sf(z)), "image": im}


def measure_mosaic(im: np.ndarray, thr: float = 0.42, open_r: int = 3, min_area_frac: float = 0.12) -> dict:
    """Wall segmentation of a polygonal mosaic (Fig. 197 or a rendered Voronoi image): cells are
    connected bright regions after opening; a cell's neighbours are the cells touched when it is
    dilated by twice the measured wall width (+2 px), so the identical pipeline is applied to the
    plate and to the simulated mosaic."""
    sm = ndimage.gaussian_filter(im, 2.0)
    cols = sm.mean(0) < 0.75; rows = sm.mean(1) < 0.75
    c0, c1 = int(np.argmax(cols)), len(cols) - int(np.argmax(cols[::-1]))
    r0, r1 = int(np.argmax(rows)), len(rows) - int(np.argmax(rows[::-1]))
    im = im[r0:r1, c0:c1]; sm = sm[r0:r1, c0:c1]
    cells = ndimage.binary_opening(sm >= thr, structure=_disk(open_r))
    walls = ~cells
    edt = ndimage.distance_transform_edt(walls)
    wall_width = float(2.2 * np.percentile(edt[walls], 90))
    dil_r = int(round(2.0 * wall_width)) + 2  # bridges the thicker vertex junctions as well as plain walls
    lab, n = ndimage.label(cells)
    areas = ndimage.sum(cells, lab, range(1, n + 1))
    min_area = min_area_frac * np.median(areas[areas > 50]) if (areas > 50).any() else 300
    border = set(np.unique(np.r_[lab[0], lab[-1], lab[:, 0], lab[:, -1]]))
    good = [i for i in range(1, n + 1) if areas[i - 1] > min_area and i not in border]
    nbrs, cents = [], []
    big = _disk(dil_r)
    for i in good:
        m = lab == i
        ring = ndimage.binary_dilation(m, structure=big) & ~m
        nbrs.append(len([u for u in np.unique(lab[ring]) if u != 0 and areas[u - 1] > min_area]))
        cents.append(ndimage.center_of_mass(m))
    A = np.array([areas[i - 1] for i in good], float); nbrs = np.array(nbrs, float)
    return {"n_cells": len(good), "neighbours": nbrs, "mean_neighbours": float(nbrs.mean()),
            "neighbour_sd": float(nbrs.std()), "areas": A, "area_cv": float(A.std() / A.mean()),
            "hexagon_fraction": float((nbrs == 6).mean()), "wall_width_px": wall_width, "dilation_px": dil_r,
            "cell_diameter_px": float(2 * np.sqrt(np.median(A) / np.pi)),
            "image": im, "cells": cells, "labels": lab, "good": good, "centroids": cents}


def measure_fig208() -> dict:
    """Thompson Fig. 208: dark-ring radii on 12 rays from the (landmarked) nucleus."""
    im = load_plate("fig208"); sm = ndimage.gaussian_filter(im, 1.2)
    cx, cy = MANUAL_LANDMARKS["fig208_nucleus_xy"]
    rs = np.arange(6, 175, 1.0)
    per_ray, ratios, fits = [], [], []
    for ang in np.deg2rad(np.arange(0, 360, 30)):
        P = []
        for da in np.deg2rad(np.linspace(-10, 10, 7)):
            xs = cx + rs * np.cos(ang + da); ys = cy + rs * np.sin(ang + da)
            P.append(ndimage.map_coordinates(sm, [ys, xs], order=1, mode="constant", cval=np.nan))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            prof = np.nanmean(np.array(P), 0)
        ok = ~np.isnan(prof); prof, r_ = prof[ok], rs[ok]
        mins, _ = signal.find_peaks(-prof, prominence=0.04, distance=8)
        rings = r_[mins][prof[mins] < 0.62]
        per_ray.append(rings)
        if len(rings) > 1:
            ratios.extend(list(rings[1:] / rings[:-1]))
        if len(rings) >= 4:
            fits.append(spacing_fits(rings))
    ratios = np.array(ratios)
    geo = np.array([f["geo_rmse"] for f in fits]); ari = np.array([f["arith_rmse"] for f in fits])
    gaps = np.concatenate([np.diff(r) for r in per_ray if len(r) > 1])
    outer_ratios = np.concatenate([r[2:] / r[1:-1] for r in per_ray if len(r) > 2])
    return {"rings_per_ray": per_ray, "ratios": ratios, "mean_ratio": float(ratios.mean()),
            "ratio_cv": float(ratios.std() / ratios.mean()), "n_rings_mean": float(np.mean([len(r) for r in per_ray])),
            "outer_ratio_mean": float(outer_ratios.mean()), "gaps_px": gaps, "gap_mean_px": float(gaps.mean()),
            "gap_cv": float(gaps.std() / gaps.mean()),
            "gap_cv_within_ray_mean": float(np.mean([f["gap_cv"] for f in fits])),
            "geo_rmse_mean": float(geo.mean()), "arith_rmse_mean": float(ari.mean()),
            "rays_geometric_better": int((geo < ari).sum()), "rays_fitted": int(len(fits)),
            "geo_P_mean": float(np.mean([f["geo_P"] for f in fits])), "image": im, "centre_xy": (cx, cy)}


def measure_liesegang_specimen() -> dict:
    """Silver-chromate Liesegang rings in gelatine (ANALOGUE, Thompson p. 427 experiment)."""
    rgb = load_specimen_rgb(LIESEGANG_SPECIMEN); im = rgb.mean(-1); H, W = im.shape
    sm = ndimage.gaussian_filter(im, 1.0)
    dark = sm < np.percentile(sm, 15)
    cy, cx = ndimage.center_of_mass(dark)
    yy, xx = np.mgrid[0:H, 0:W]

    def prof(cx_, cy_):
        b = np.round(np.hypot(yy - cy_, xx - cx_)).astype(int)
        return np.bincount(b.ravel(), weights=sm.ravel()) / np.maximum(np.bincount(b.ravel()), 1)

    best = None
    for dx in range(-12, 13, 2):
        for dy in range(-12, 13, 2):
            p = prof(cx + dx, cy + dy)[30:220]
            score = np.std(p - ndimage.uniform_filter1d(p, 15))
            if best is None or score > best[0]:
                best = (score, cx + dx, cy + dy)
    cx, cy = best[1], best[2]
    p = prof(cx, cy); hp = p - ndimage.uniform_filter1d(p, 21)
    mins, _ = signal.find_peaks(-hp, prominence=0.015, distance=3)
    rings = mins[(mins > 35) & (mins < 215)].astype(float)
    ratios = rings[1:] / rings[:-1]
    rr, tr = _polar(im.shape, cx, cy)
    ann = (rr > rings.min() - 5) & (rr < rings.max() + 5)
    out = {"centre_xy": (float(cx), float(cy)), "ring_radii_px": rings, "n_rings": int(len(rings)),
           "ratios": ratios, "P_mean": float(ratios.mean()), "P_cv": float(ratios.std() / ratios.mean()),
           "radial_line_index_rings": line_index(im, ann, tr, 2.0), "profile": p, "image": im}
    out.update(spacing_fits(rings))
    return out


def measure_ooid() -> dict:
    """Carmel Fm. ooid (ANALOGUE): nucleus vs outer-boundary asphericity = direct test of E3."""
    rgb = load_specimen_rgb(OOID_SPECIMEN); gray = rgb.mean(-1); H, W = gray.shape
    sm = ndimage.gaussian_filter(gray, 2.0)
    R, G, B = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    red = (R > 0.7) & (G < 0.25) & (B < 0.25); red[: int(0.75 * H)] = False; red[:, int(0.4 * W):] = False
    ys, xs = np.nonzero(red); px_per_mm = (xs.max() - xs.min() + 1) / 0.5
    m = sm < 0.50; m[ys.min() - 8:, : xs.max() + 8] = False
    m = ndimage.binary_fill_holes(ndimage.binary_closing(ndimage.binary_opening(m, structure=_disk(3)), structure=_disk(6)))
    lab, n = ndimage.label(m); areas = ndimage.sum(m, lab, range(1, n + 1)); big = lab == (np.argmax(areas) + 1)
    core = big & (sm < 0.40); core = ndimage.binary_opening(core, structure=_disk(5))
    lab, n = ndimage.label(core); areas = ndimage.sum(core, lab, range(1, n + 1))
    core = ndimage.binary_fill_holes(lab == (np.argmax(areas) + 1))
    cy, cx = ndimage.center_of_mass(core)
    yy_, xx_ = np.nonzero(core); cov = np.cov(np.vstack([xx_ - cx, yy_ - cy])); ev, evec = np.linalg.eigh(cov)
    ang = float(np.arctan2(evec[1, 1], evec[0, 1]))
    th = np.deg2rad(np.arange(0, 360, 2.0)); rs = np.arange(0, 260, 1.0)
    rho_n = np.full(len(th), np.nan); rho_o = np.full(len(th), np.nan)
    for i, t in enumerate(th):
        xs_ = cx + rs * np.cos(t); ys_ = cy + rs * np.sin(t)
        ok = (xs_ >= 0) & (xs_ < W - 1) & (ys_ >= 0) & (ys_ < H - 1); r_ = rs[ok]
        cv = ndimage.map_coordinates(core.astype(float), [ys_[ok], xs_[ok]], order=0) > 0.5
        bv = ndimage.map_coordinates(big.astype(float), [ys_[ok], xs_[ok]], order=0) > 0.5
        rho_n[i] = r_[np.argmax(~cv)] if (~cv).any() else r_[-1]
        j = int(np.argmax(~bv)) if (~bv).any() else len(r_) - 1
        rho_o[i] = min(r_[j], rho_n[i] + 95)
    med = ndimage.median_filter(rho_o, size=11, mode="wrap"); bad = np.abs(rho_o - med) > 0.12 * med
    rho_o[bad] = med[bad]

    def at(r_, t0):
        return float(r_[np.argmin(np.abs(((th - t0 + np.pi) % (2 * np.pi)) - np.pi))])

    asph = lambda r_: float((r_.max() - r_.min()) / r_.mean())
    n_maj = np.mean([at(rho_n, ang), at(rho_n, ang + np.pi)]); n_min = np.mean([at(rho_n, ang + np.pi / 2), at(rho_n, ang - np.pi / 2)])
    o_maj = np.mean([at(rho_o, ang), at(rho_o, ang + np.pi)]); o_min = np.mean([at(rho_o, ang + np.pi / 2), at(rho_o, ang - np.pi / 2)])
    off = rho_o - rho_n
    # lamellae along the cortex normal coordinate and the normal-line index
    d = ndimage.distance_transform_edt(~core); cortex = big & ~core
    bins = np.arange(0, d[cortex].max() + 3, 3); idx = np.digitize(d[cortex], bins)
    pr = np.array(ndimage.mean(gray[cortex], idx, range(1, len(bins))))
    hp = pr - ndimage.uniform_filter1d(pr, 7); mins, _ = signal.find_peaks(-hp, prominence=0.01)
    gd = ndimage.gaussian_filter(d, 3); tn = np.arctan2(ndimage.sobel(gd, 0), ndimage.sobel(gd, 1))
    return {"px_per_mm": float(px_per_mm), "centre_xy": (float(cx), float(cy)), "theta": th, "rho_nucleus": rho_n, "rho_outer": rho_o,
            "nucleus_mean_radius_mm": float(rho_n.mean() / px_per_mm), "outer_mean_radius_mm": float(rho_o.mean() / px_per_mm),
            "nucleus_asphericity": asph(rho_n), "outer_asphericity": asph(rho_o),
            "nucleus_radius_cv": float(rho_n.std() / rho_n.mean()), "outer_radius_cv": float(rho_o.std() / rho_o.mean()),
            "nucleus_aspect": float(n_maj / n_min), "outer_aspect": float(o_maj / o_min),
            "law3_predicted_outer_aspect": float((n_maj + (o_min - n_min)) / (n_min + (o_min - n_min))),
            "law3_predicted_outer_asphericity": float((rho_n.max() - rho_n.min()) / rho_o.mean()),
            "cortex_thickness_mean_px": float(off.mean()), "cortex_thickness_cv": float(off.std() / off.mean()),
            "cortex_offset_major_px": float(o_maj - n_maj), "cortex_offset_minor_px": float(o_min - n_min),
            "lamellae_count": int(len(mins)), "lamellae_d_px": bins[mins],
            "normal_line_index_cortex": line_index(gray, cortex, tn, 1.5), "image": rgb, "core": core, "big": big}


# =============================================================================== orchestration
def run_models(quick: bool = False, etas=None, b0s=None, a0=None, rng_seed: int = SEED) -> dict:
    key = (quick, tuple(etas or ()), tuple(b0s or ()), a0, rng_seed)
    if _CACHE.get("key") == key:
        return _CACHE["models"]
    etas = list(etas) if etas else PARAMS["dbm"]["etas"]
    b0s = list(b0s) if b0s else PARAMS["liesegang_b0_sweep"]
    t_end = 600.0 if quick else None
    lies = {b0: simulate_liesegang_radial(dim=2, b0=b0, a0=a0, t_end=t_end) for b0 in b0s}
    out = {
        "integrity": verify_dataset_integrity(),
        "laws": radial_growth_laws(),
        "front": simulate_front_rounding(grid=161 if quick else None, n_frames=12 if quick else None),
        "front_control": simulate_front_rounding(grid=161 if quick else None, n_frames=12 if quick else None, metric="chessboard"),
        "voronoi": simulate_voronoi_impingement(grid=300 if quick else None, rng_seed=rng_seed),
        "voronoi_poisson": simulate_voronoi_impingement(grid=300 if quick else None, rng_seed=rng_seed, min_sep_frac=0.0),
        "liesegang_sweep": lies,
        "liesegang_2d": lies[b0s[len(b0s) // 2]] if 0.1 not in lies else lies[0.1],
        "liesegang_control": simulate_liesegang_radial(dim=2, threshold=False, a0=a0, t_end=t_end),
        "liesegang_3d": simulate_liesegang_radial(dim=3, a0=PARAMS["liesegang_3d_a0"], t_end=t_end),
        "dbm": {eta: simulate_dbm_cluster(eta, n_sites=300 if quick else None, grid=121 if quick else None, rng_seed=rng_seed) for eta in etas},
        "cv": {
            "fig196": measure_fig196(), "fig195": measure_fig195(n_null=30 if quick else 100),
            "fig197": None, "fig208": measure_fig208(), "liesegang": measure_liesegang_specimen(), "ooid": measure_ooid(),
        },
    }
    out["cv"]["fig197"] = measure_mosaic(load_plate("fig197R"))
    out["cv"]["voronoi_same_pipeline"] = measure_mosaic(out["voronoi"]["rendered"])
    out["stats"] = compute_statistics(out)
    _CACHE["models"], _CACHE["key"] = out, key
    return out


def compute_statistics(M: dict) -> dict:
    cv, e = M["cv"], M["dbm"]
    etas = np.array(sorted(e)); sol = np.array([e[x]["solidity"] for x in etas]); df = np.array([e[x]["mass_radius_exponent"] for x in etas])
    r_sol = stats.pearsonr(etas, sol) if len(etas) > 2 else (float("nan"), float("nan"))
    ks_mosaic = stats.ks_2samp(cv["fig197"]["neighbours"], cv["voronoi_same_pipeline"]["neighbours"])
    ks_lies = stats.ks_2samp(cv["liesegang"]["ratios"], M["liesegang_2d"]["ratios"]) if len(M["liesegang_2d"]["ratios"]) > 1 else None
    ks_208 = stats.ks_2samp(cv["fig208"]["ratios"], M["liesegang_3d"]["ratios"]) if len(M["liesegang_3d"]["ratios"]) > 1 else None
    b0s = sorted(M["liesegang_sweep"]); Ps = [M["liesegang_sweep"][b]["spacing_coefficient"] for b in b0s]
    return {
        "eta_vs_solidity_pearson_r": float(r_sol[0]), "eta_vs_solidity_p": float(r_sol[1]),
        "eta_vs_exponent_spearman": float(stats.spearmanr(etas, df)[0]),
        "mosaic_ks_stat": float(ks_mosaic.statistic), "mosaic_ks_p": float(ks_mosaic.pvalue),
        "mosaic_neighbour_delta": float(cv["fig197"]["mean_neighbours"] - cv["voronoi_same_pipeline"]["mean_neighbours"]),
        "liesegang_ratio_ks_stat": float(ks_lies.statistic) if ks_lies else float("nan"),
        "liesegang_ratio_ks_p": float(ks_lies.pvalue) if ks_lies else float("nan"),
        "fig208_ratio_ks_p": float(ks_208.pvalue) if ks_208 else float("nan"),
        "b0_sweep_P": dict(zip([str(b) for b in b0s], Ps)),
        "b0_vs_P_spearman": float(stats.spearmanr(b0s, Ps)[0]) if len(b0s) > 2 else float("nan"),
    }


def evaluate_claim(M: dict) -> dict:
    cv, st, fr, fc = M["cv"], M["stats"], M["front"], M["front_control"]
    f196, f195, f197, vsp, f208, lsp, oo = (cv[k] for k in ("fig196", "fig195", "fig197", "voronoi_same_pipeline", "fig208", "liesegang", "ooid"))
    tests = {
        "S1_isolated_calcospherite_is_round (Fig. 196 outer-radius CV < 0.05)": f196["outer_radius_cv"] < 0.05,
        "S2_radial_fibre_texture (Fig. 196 border RLI > 0.2 and Fig. 195 nuclei z > 2 vs null)": f196["radial_line_index_border"] > 0.2 and f195["z_vs_null"] > 2,
        "S3_crowded_spherites_form_a_voronoi_like_mosaic (|delta neighbours| < 0.5 or KS p > 0.01)": abs(st["mosaic_neighbour_delta"]) < 0.5 or st["mosaic_ks_p"] > 0.01,
        "S4_concentric_zones_in_logarithmic_ratio (Liesegang: geometric RMSE < arithmetic, 1.02 < P < 1.5, gaps increase)":
            lsp["geo_rmse"] < lsp["arith_rmse"] and 1.02 < lsp["P_mean"] < 1.5 and lsp["gap_spearman_rho"] > 0,
        "S5_model_rounds_only_under_isotropic_deposition (exponent ~ -1 vs control > -0.3) and bands need a nucleation gap":
            -1.3 < fr["decay_exponent"] < -0.7 and fc["decay_exponent"] > -0.3 and M["liesegang_control"]["n_bands"] <= 2 < M["liesegang_2d"]["n_bands"],
        "S6_natural_concretion_rounds_off_outward (ooid outer asphericity < nucleus asphericity)": oo["outer_asphericity"] < oo["nucleus_asphericity"],
        "S7_spiny_forms_under_scanty_supply (solidity falls monotonically with eta, r < -0.8)": st["eta_vs_solidity_pearson_r"] < -0.8,
    }
    n_pass = sum(tests.values())
    mandatory = tests[list(tests)[0]] and tests[list(tests)[1]] and tests[list(tests)[4]]
    verdict = "supported" if (n_pass >= 6 and mandatory) else ("mixed" if n_pass >= 4 else "refuted")
    return {"tests": tests, "n_pass": int(n_pass), "n_tests": len(tests), "verdict": verdict}


def measure_morphoeval(quick: bool = False) -> list[dict]:
    """Quantitative morphometrics consumed by morphoeval/measure_outputs.py."""
    M = run_models(quick=quick)
    cv, st, laws = M["cv"], M["stats"], M["laws"]
    f196, f195, f197, vsp, f208, lsp, oo = (cv[k] for k in ("fig196", "fig195", "fig197", "voronoi_same_pipeline", "fig208", "liesegang", "ooid"))
    fr, fc, l2, l3, lc, vo, e = M["front"], M["front_control"], M["liesegang_2d"], M["liesegang_3d"], M["liesegang_control"], M["voronoi"], M["dbm"]

    def m(name, value, unit, method, reference=None, source=None, note=None, derived_from="script"):
        d = {"name": name, "value": None if value is None or (isinstance(value, float) and np.isnan(value)) else float(round(float(value), 4)),
             "unit": unit, "method": method, "derived_from": derived_from}
        if reference is not None:
            d["reference_value"] = reference
        if source:
            d["reference_source"] = source
        if note:
            d["note"] = note
        return d

    etas = sorted(e)
    return [
        # --- model laws
        m("E3 asphericity decay exponent, isotropic deposition (square nucleus)", fr["decay_exponent"], "", f"log-log slope of A vs R over {len(fr['levels'])} level sets; Pearson r = {fr['loglog_r']:.3f}", -1.0,
          "Lean asphericity_nonincreasing; A = const / R", note=f"anisotropic chessboard control exponent = {fc['decay_exponent']:.3f} (does not round off)"),
        m("E3 CONTROL asphericity decay exponent, anisotropic (chessboard) deposition", fc["decay_exponent"], "", "same level-set analysis with a chessboard metric (crystalline alternative)", 0.0, "square stays square: control"),
        m("E1 lamella-thickness CV, interface-controlled law", laws["lamella_lin_cv"], "", "std/mean of successive R_{n+1}-R_n", 0.0, "Lean lamella_thickness_constant",
          note=f"vs diffusion-limited E2 lamella CV = {laws['lamella_par_cv']:.3f} (thinning outward)"),
        m("E4 mean neighbour count, equal-rate Voronoi mosaic (exact labels, hard-core centres)", vo["mean_neighbours"], "neighbours", f"{vo['n_interior']} interior cells of a {PARAMS['voronoi']['n_centres']}-centre map, min separation {vo['min_sep_frac']} x mean spacing", 6.0, "Euler characteristic",
          note=f"area CV {vo['area_cv']:.3f} (hard-core) vs {M['voronoi_poisson']['area_cv']:.3f} (Poisson centres, exact {M['voronoi_poisson']['mean_neighbours']:.2f} neighbours); hexagon fraction {vo['hexagon_fraction']:.2f}"),
        m("E5 Liesegang spacing coefficient P (2-D, a0=50, b0=0.1)", l2["spacing_coefficient"], "", f"mean r_(n+1)/r_n over {l2['n_bands']} band centres; CV = {l2['spacing_cv']:.3f}", 1.15, "Jablczynski 1923 (1.05-1.4)",
          note=f"b0 sweep P = {st['b0_sweep_P']}; CONTROL without nucleation gap: {lc['n_bands']} band(s), deposit covers {lc['deposit_fraction_of_domain']:.2f} of domain"),
        m("E5 CONTROL band count without supersaturation gap (K_ss = K_sp)", lc["n_bands"], "bands", "precipitate maxima above 10% of peak; continuous deposit expected", 1, "null / ablation control"),
        m("E5 Liesegang band count (3-D spherical, a0=100)", l3["n_bands"], "bands", f"P = {l3['spacing_coefficient']:.3f}, geometric fit r = {l3['geo_r']:.4f}"),
        m("E6 eta-vs-solidity Pearson r over the supply sweep", st["eta_vs_solidity_pearson_r"], "", f"eta in {etas}; solidity = sites / convex-hull area; p = {st['eta_vs_solidity_p']:.3g}", -1.0, "Thompson p. 424: scanty supply -> spinous"),
        m("E6 DLA (eta=1) mass-radius exponent", e[1.0]["mass_radius_exponent"] if 1.0 in e else None, "", "log N(r) vs log r fit, r <= 0.75 R_max", 1.71, "Witten & Sander 1981",
          note=f"Eden eta=0 exponent = {e[0.0]['mass_radius_exponent']:.3f}, solidity {e[0.0]['solidity']:.3f} vs DLA {e[1.0]['solidity']:.3f}" if (0.0 in e and 1.0 in e) else None),
        # --- Thompson's plates
        m("Fig. 196 outer-boundary radius CV (single calcospherite)", f196["outer_radius_cv"], "", "180-sector polar radius of the segmented disc about its centroid", 0.0, "perfect sphere section",
          note=f"max/min = {f196['outer_max_over_min']:.3f}; R = {f196['R_outer_px']:.1f} px"),
        m("Fig. 196 radial-line index of the striated border", f196["radial_line_index_border"], "", "Hessian ridge orientation vs radial direction, sigma 2 px (+1 radial, -1 concentric)", 1.0, "ideal radial fibres",
          note=f"core index = {f196['radial_line_index_core']:.3f} (isotropic nucleus); {f196['striae_count']} striae, spacing {f196['mean_striae_spacing_px']:.1f} px"),
        m("Fig. 196 nucleus / outer radius ratio", f196["nucleus_over_outer"], "", "half-rise of the angular-mean local-std (texture) profile"),
        m("Fig. 195 mean radial-line index around 6 landmarked nuclei", f195["mean"], "", f"annulus {MANUAL_LANDMARKS['fig195_annulus_px']} px; sd {f195['sd']:.3f}; random-centre null {f195['null_mean']:.3f} +- {f195['null_sd']:.3f}", 0.0, "null: random centres",
          note=f"z = {f195['z_vs_null']:.2f}, one-sided p = {f195['p_one_sided']:.3g}"),
        m("Fig. 197 mean neighbour count of segmented cells", f197["mean_neighbours"], "neighbours", f"{f197['n_cells']} interior cells; sd {f197['neighbour_sd']:.2f}; hexagon fraction {f197['hexagon_fraction']:.2f}",
          round(vsp["mean_neighbours"], 3), "equal-rate Voronoi mosaic measured by the same pipeline",
          note=f"delta = {st['mosaic_neighbour_delta']:+.3f}; KS p = {st['mosaic_ks_p']:.3f}; area CV plate {f197['area_cv']:.3f} vs Voronoi {vsp['area_cv']:.3f}; wall width {f197['wall_width_px']:.1f} px vs {vsp['wall_width_px']:.1f} px, cell diameter {f197['cell_diameter_px']:.0f} vs {vsp['cell_diameter_px']:.0f} px"),
        m("Fig. 197 cell-area CV", f197["area_cv"], "", "std/mean of interior cell areas", round(vsp["area_cv"], 3), "same-pipeline hard-core Voronoi",
          note=f"Poisson-Voronoi exact area CV = {M['voronoi_poisson']['area_cv']:.3f} (literature 0.529): the plate is far more regular than Poisson, i.e. the nuclei were hard-core spaced spherites of similar size before impingement"),
        m("Fig. 208 lamella-gap CV along 12 rays (spherocrystal)", f208["gap_cv_within_ray_mean"], "", f"mean within-ray CV of successive dark-ring gaps; pooled gap {f208['gap_mean_px']:.1f} px, pooled CV {f208['gap_cv']:.3f}", 0.0, "E1 constant lamella thickness",
          note=f"arithmetic (constant-gap) fit beats geometric on {f208['rays_fitted'] - f208['rays_geometric_better']}/{f208['rays_fitted']} rays; mean ratio {f208['mean_ratio']:.2f} (outer rings {f208['outer_ratio_mean']:.2f}) vs E5 3-D sim P {l3['spacing_coefficient']:.3f}; KS p = {st['fig208_ratio_ks_p']:.3g}"),
        # --- analogue specimens
        m("Liesegang specimen Jablczynski P (ANALOGUE)", lsp["P_mean"], "", f"{lsp['n_rings']} rings from the azimuthal-mean profile; ratio CV {lsp['P_cv']:.4f}", 1.15, "Jablczynski 1923 (1.05-1.4)",
          note=f"geometric fit r = {lsp['geo_r']:.4f}, RMSE {lsp['geo_rmse']:.2f} px vs arithmetic r = {lsp['arith_r']:.4f}, RMSE {lsp['arith_rmse']:.2f} px; gap Spearman rho = {lsp['gap_spearman_rho']:.2f} (p = {lsp['gap_spearman_p']:.3g}); KS vs E5 sim p = {st['liesegang_ratio_ks_p']:.3f}"),
        m("Liesegang specimen geometric-fit RMSE", lsp["geo_rmse"], "px", "ln r_n = a + n ln P", round(lsp["arith_rmse"], 3), "arithmetic (constant gap) alternative RMSE"),
        m("Liesegang specimen radial-line index (concentric rings)", lsp["radial_line_index_rings"], "", "Hessian ridge orientation vs radial direction", -1.0, "ideal concentric lamellae"),
        m("Ooid outer asphericity (ANALOGUE)", oo["outer_asphericity"], "", "(max-min)/mean polar radius of the outer boundary about the nucleus centroid", round(oo["law3_predicted_outer_asphericity"], 3),
          "E3 uniform-offset prediction from the nucleus", note=f"nucleus asphericity {oo['nucleus_asphericity']:.3f} -> outer {oo['outer_asphericity']:.3f}; aspect {oo['nucleus_aspect']:.2f} -> {oo['outer_aspect']:.2f} (uniform offset predicts {oo['law3_predicted_outer_aspect']:.2f}); cortex offset major {oo['cortex_offset_major_px']:.0f} px vs minor {oo['cortex_offset_minor_px']:.0f} px"),
        m("Ooid cortex thickness CV (ANALOGUE)", oo["cortex_thickness_cv"], "", f"polar outer minus nucleus radius; mean {oo['cortex_thickness_mean_px']:.1f} px = {oo['cortex_thickness_mean_px'] / oo['px_per_mm']:.3f} mm", 0.0, "uniform normal deposition",
          note=f"{oo['lamellae_count']} dark lamellae; normal-line index {oo['normal_line_index_cortex']:.3f}; outer mean radius {oo['outer_mean_radius_mm']:.3f} mm"),
    ]


def write_result(M: dict, path: Path = RESULT_JSON) -> Path:
    ev = evaluate_claim(M)
    cv, st = M["cv"], M["stats"]
    ref = load_reference()
    res = {
        "claim_id": "T1917-023",
        "date_made": "1917",
        "claim": ref["claim"],
        "dataset": {
            "name": "Thompson (1917) On Growth and Form 1st ed., Ch. IX plates after Harting (1872): Figs. 195-197 (pp. 421-422) and 208-209 (p. 432), Internet Archive scan ongrowthform1917thom; analogues: Carmel Fm. ooids (Commons, PD) and Ag2CrO4 Liesegang rings in gelatine (Commons, CC BY-SA 3.0)",
            "doi": None,
            "ia_identifier": "ongrowthform1917thom",
            "sha256": {k: v["sha256"] for k, v in ref["vendored_files"].items()},
            "match": "MATCH — the plates are Thompson's reproductions of the very Harting specimens named in the claim; Commons files are labelled analogues",
        },
        "verdict": ev["verdict"],
        "tests": {k: bool(v) for k, v in ev["tests"].items()},
        "n_pass": ev["n_pass"], "n_tests": ev["n_tests"],
        "headline_numbers": {
            "fig196_outer_radius_cv": round(cv["fig196"]["outer_radius_cv"], 4),
            "fig196_radial_line_index_border": round(cv["fig196"]["radial_line_index_border"], 3),
            "fig195_radial_line_index_z_vs_null": round(cv["fig195"]["z_vs_null"], 2),
            "fig197_mean_neighbours": round(cv["fig197"]["mean_neighbours"], 3),
            "voronoi_same_pipeline_mean_neighbours": round(cv["voronoi_same_pipeline"]["mean_neighbours"], 3),
            "mosaic_ks_p": round(st["mosaic_ks_p"], 4),
            "fig208_mean_ring_ratio": round(cv["fig208"]["mean_ratio"], 3),
            "fig208_within_ray_gap_cv": round(cv["fig208"]["gap_cv_within_ray_mean"], 3),
            "fig208_rays_constant_gap_fits_better": f"{cv['fig208']['rays_fitted'] - cv['fig208']['rays_geometric_better']}/{cv['fig208']['rays_fitted']}",
            "liesegang_specimen_P": round(cv["liesegang"]["P_mean"], 4),
            "liesegang_geo_rmse_px": round(cv["liesegang"]["geo_rmse"], 3),
            "liesegang_arith_rmse_px": round(cv["liesegang"]["arith_rmse"], 3),
            "liesegang_sim_P_2d": round(M["liesegang_2d"]["spacing_coefficient"], 4),
            "ooid_nucleus_asphericity": round(cv["ooid"]["nucleus_asphericity"], 3),
            "ooid_outer_asphericity": round(cv["ooid"]["outer_asphericity"], 3),
            "E3_decay_exponent_isotropic": round(M["front"]["decay_exponent"], 3),
            "E3_decay_exponent_anisotropic_control": round(M["front_control"]["decay_exponent"], 3),
            "eta_vs_solidity_pearson_r": round(st["eta_vs_solidity_pearson_r"], 3),
        },
        "caveats": [
            "'Physical deposition alone' yields spherical forms in the interface-controlled (colloid-damped) regime; under diffusion-limited supply the same model gives spinous DLA forms, exactly the qualification Thompson himself makes on p. 424.",
            "The ooid rounds off faster than uniform normal deposition predicts (minor-axis cortex thicker than major-axis cortex), so natural concretions need curvature- or agitation-dependent accretion on top of E3.",
            "Fig. 195 nucleus centres and the Fig. 208 nucleus are manual landmarks (pinned in the script); all measurements about them are automated.",
            "Harting (1872) is reached through Thompson's reproductions and the 1872 QJMS abstract, not the original plates.",
        ],
        "script": "thompson/claims/t1917-023_on-concretions-spicules-and-spicular-skeletons.py",
        "figure": "docs/media/t1917_023_on_concretions_spicules_and_sp_comparison.png",
        "animation": "docs/media/t1917_023_on_concretions_spicules_and_sp_growth.gif",
        "lean": "morphoeval/lean/loops/Thompson1917_T1917_023.lean",
    }
    path.write_text(json.dumps(res, indent=2) + "\n")
    return path


# =============================================================================== rendering
def _spherite_texture(size: int, R: float, bands: np.ndarray, n_fibres: int = 48) -> np.ndarray:
    """Synthetic radial-fibre + concentric-band spherite image of radius R (pixels); the band radii
    come from the 3-D E5 simulation, the fibre count from the Fig. 196 striae measurement."""
    c = size / 2
    yy, xx = np.mgrid[0:size, 0:size]
    rr = np.hypot(yy - c + 0.5, xx - c + 0.5); th = np.arctan2(yy - c + 0.5, xx - c + 0.5)
    img = np.ones((size, size))
    inside = rr <= R
    fib = 0.5 + 0.5 * np.cos(n_fibres * th)
    tex = 0.55 + 0.35 * fib * np.clip((rr - 0.15 * R) / (0.85 * R + 1e-9), 0, 1)
    for b in bands:
        if b < R:
            tex *= 1.0 - 0.55 * np.exp(-((rr - b) ** 2) / (2 * (0.012 * size) ** 2))
    img[inside] = tex[inside]
    img[rr <= 0.12 * R] = 0.35
    return img


def render_comparison_png(M: dict, path: Path = COMPARISON_PNG) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cv, st = M["cv"], M["stats"]
    ev = evaluate_claim(M)
    plt.rcParams.update({"axes.titlesize": 9})
    fig, ax = plt.subplots(3, 4, figsize=(21, 15.5))
    fig.suptitle(f"Thompson (1917) T1917-023 — calcospherites grow by radial deposition about a centre (Harting 1872) — "
                 f"Pass 2: verdict {ev['verdict'].upper()} ({ev['n_pass']}/{ev['n_tests']} tests)", fontsize=14)
    # row 0: plates with overlays
    f196 = cv["fig196"]
    a = ax[0, 0]; a.imshow(f196["image"], cmap="gray"); ey, ex = np.nonzero(f196["edge"]); a.plot(ex, ey, "r.", ms=0.6)
    cx, cy = f196["centre_xy"]; a.add_patch(plt.Circle((cx, cy), f196["R_nucleus_px"], fill=False, color="cyan", lw=1))
    a.set_title(f"Fig. 196 (after Harting): outer-radius CV {f196['outer_radius_cv']:.4f}\nborder radial-line index {f196['radial_line_index_border']:+.2f}, {f196['striae_count']} striae")
    f195 = cv["fig195"]
    a = ax[0, 1]; a.imshow(f195["image"], cmap="gray")
    for (x, y), v in zip(MANUAL_LANDMARKS["fig195_nuclei_xy"], f195["radial_line_index_per_nucleus"]):
        a.add_patch(plt.Circle((x, y), MANUAL_LANDMARKS["fig195_annulus_px"][1], fill=False, color="red", lw=0.8))
        a.text(x, y, f"{v:+.2f}", color="red", fontsize=7, ha="center")
    a.set_title(f"Fig. 195: radial-line index {f195['mean']:+.3f} (null {f195['null_mean']:+.3f}±{f195['null_sd']:.3f}, z = {f195['z_vs_null']:.1f})")
    f197 = cv["fig197"]
    a = ax[0, 2]; a.imshow(f197["image"], cmap="gray"); edge = f197["cells"] & ~ndimage.binary_erosion(f197["cells"]); ey, ex = np.nonzero(edge); a.plot(ex, ey, "r.", ms=0.4)
    for (y, x), nb in zip(f197["centroids"], f197["neighbours"]):
        a.text(x, y, f"{int(nb)}", color="blue", fontsize=6, ha="center", va="center")
    a.set_title(f"Fig. 197 mosaic: {f197['n_cells']} cells, <n> = {f197['mean_neighbours']:.2f}\n(same-pipeline Voronoi <n> = {cv['voronoi_same_pipeline']['mean_neighbours']:.2f}, KS p = {st['mosaic_ks_p']:.2f})")
    f208 = cv["fig208"]
    a = ax[0, 3]; a.imshow(f208["image"], cmap="gray"); cx, cy = f208["centre_xy"]
    for ang, rings in zip(np.deg2rad(np.arange(0, 360, 30)), f208["rings_per_ray"]):
        a.plot(cx + rings * np.cos(ang), cy + rings * np.sin(ang), "r.", ms=4)
    a.set_title(f"Fig. 208 spherocrystal: lamella gap {f208['gap_mean_px']:.0f} px, within-ray gap CV {f208['gap_cv_within_ray_mean']:.2f}\nconstant-gap (E1) fit beats geometric on {f208['rays_fitted'] - f208['rays_geometric_better']}/{f208['rays_fitted']} rays")
    # row 1: model
    fr, fc = M["front"], M["front_control"]
    a = ax[1, 0]
    for k in np.linspace(0, len(fr["masks"]) - 1, 5).astype(int):
        a.contour(fr["masks"][k].astype(float), levels=[0.5], colors=[plt.cm.viridis(k / (len(fr["masks"]) - 1))], linewidths=1)
        a.contour(fc["masks"][k].astype(float), levels=[0.5], colors=["red"], linewidths=0.6, linestyles="dashed")
    a.set_aspect("equal"); a.invert_yaxis(); a.set_xticks([]); a.set_yticks([])
    a.set_title(f"E3 isotropic deposition rounds a square nucleus: A ∝ R^{fr['decay_exponent']:.2f}\nCONTROL anisotropic metric (red dashed): A ∝ R^{fc['decay_exponent']:.2f}")
    e = M["dbm"]; etas = sorted(e)
    a = ax[1, 1]; tiles = []
    for et in etas:
        cl = e[et]["cluster"]; ys, xs = np.nonzero(cl); c = cl.shape[0] // 2; R = int(e[et]["R_max"]) + 3
        tile = cl[max(c - R, 0):c + R + 1, max(c - R, 0):c + R + 1].astype(float)
        from PIL import Image
        tiles.append(np.asarray(Image.fromarray((tile * 255).astype(np.uint8)).resize((160, 160), Image.NEAREST)) / 255.0)
    a.imshow(np.concatenate(tiles, axis=1), cmap="gray_r", aspect="auto"); a.set_xticks([80 + 160 * i for i in range(len(etas))]); a.set_xticklabels([f"η={et:g}\nsol {e[et]['solidity']:.2f}" for et in etas], fontsize=8); a.set_yticks([])
    a.set_title(f"E6 supply sweep: Eden disc → DLA spines; r(η, solidity) = {st['eta_vs_solidity_pearson_r']:.2f}\n(Thompson p. 424: scanty supply → spinous / digitiform)")
    vo = M["voronoi"]; vsp = cv["voronoi_same_pipeline"]
    a = ax[1, 2]; a.imshow(vsp["image"], cmap="gray")
    for (y, x), nb in zip(vsp["centroids"], vsp["neighbours"]):
        a.text(x, y, f"{int(nb)}", color="blue", fontsize=6, ha="center", va="center")
    a.set_xticks([]); a.set_yticks([]); a.set_title(f"E4 equal-rate Voronoi mosaic (hard-core centres) measured like Fig. 197\nexact <n> = {vo['mean_neighbours']:.2f}, pipeline <n> = {vsp['mean_neighbours']:.2f}, area CV {vsp['area_cv']:.2f} (Poisson {M['voronoi_poisson']['area_cv']:.2f})")
    l2, lc = M["liesegang_2d"], M["liesegang_control"]
    a = ax[1, 3]
    for b0, col in zip(sorted(M["liesegang_sweep"]), ("tab:blue", "k", "tab:green", "tab:purple")):
        ls = M["liesegang_sweep"][b0]; a.plot(ls["r"], ls["p"] / max(ls["p"].max(), 1e-9) + 0.0, color=col, lw=0.8, label=f"b0={b0:g}: {ls['n_bands']} bands, P={ls['spacing_coefficient']:.3f}")
    a.plot(lc["r"], lc["p"] / max(lc["p"].max(), 1e-9), color="tab:red", lw=1.2, ls="--", label=f"CONTROL no gap: {lc['n_bands']} band")
    a.set_xlabel("r"); a.set_ylabel("precipitate (norm.)"); a.legend(fontsize=7); a.set_title("E5 Liesegang banding (2-D radial), b0 sweep + ablation control")
    # row 2: analogue specimens + fits
    oo = cv["ooid"]
    a = ax[2, 0]; a.imshow(oo["image"]); cx, cy = oo["centre_xy"]
    a.plot(cx + oo["rho_nucleus"] * np.cos(oo["theta"]), cy + oo["rho_nucleus"] * np.sin(oo["theta"]), "b-", lw=1)
    a.plot(cx + oo["rho_outer"] * np.cos(oo["theta"]), cy + oo["rho_outer"] * np.sin(oo["theta"]), "g-", lw=1)
    a.set_xticks([]); a.set_yticks([]); a.set_title(f"ANALOGUE Carmel ooid: nucleus asphericity {oo['nucleus_asphericity']:.2f} → outer {oo['outer_asphericity']:.2f}\n(E3 uniform offset predicts {oo['law3_predicted_outer_asphericity']:.2f}); {oo['lamellae_count']} lamellae")
    lsp = cv["liesegang"]
    a = ax[2, 1]; a.imshow(lsp["image"], cmap="gray"); cx, cy = lsp["centre_xy"]
    for r in lsp["ring_radii_px"]:
        a.add_patch(plt.Circle((cx, cy), r, fill=False, color="lime", lw=0.6))
    a.set_xticks([]); a.set_yticks([]); a.set_title(f"ANALOGUE Ag2CrO4 Liesegang rings: {lsp['n_rings']} rings, P = {lsp['P_mean']:.3f}\nradial-line index {lsp['radial_line_index_rings']:+.2f} (concentric)")
    a = ax[2, 2]; n = np.arange(lsp["n_rings"]); a.semilogy(n, lsp["ring_radii_px"], "ko", label="specimen rings")
    a.semilogy(n, lsp["geo_r0"] * lsp["geo_P"] ** n, "r-", label=f"geometric P={lsp['geo_P']:.3f}, RMSE {lsp['geo_rmse']:.1f} px")
    a.semilogy(n, lsp["arith_r0"] + lsp["arith_d"] * n, "b--", label=f"arithmetic d={lsp['arith_d']:.1f}, RMSE {lsp['arith_rmse']:.1f} px")
    a.set_xlabel("ring index n"); a.set_ylabel("radius (px)"); a.legend(fontsize=7)
    a.set_title(f"Thompson p. 427 'logarithmic ratio': gaps grow (Spearman ρ = {lsp['gap_spearman_rho']:.2f})\nKS specimen vs E5 ratios p = {st['liesegang_ratio_ks_p']:.2f}")
    a = ax[2, 3]
    sol = [e[et]["solidity"] for et in etas]; dfx = [e[et]["mass_radius_exponent"] for et in etas]; env = [e[et]["envelope_radius_cv"] for et in etas]
    a.plot(etas, sol, "o-", label="solidity"); a.plot(etas, env, "s-", label="envelope radius CV"); a.plot(etas, np.array(dfx) / 2, "^-", label="mass-radius exponent / 2")
    a.axhline(1.71 / 2, color="gray", ls=":", lw=0.8); a.set_xlabel("supply exponent η"); a.legend(fontsize=7)
    a.set_title(f"E6 regime sweep: Spearman(η, exponent) = {st['eta_vs_exponent_spearman']:.2f}\nE1/E2: lamella CV {M['laws']['lamella_lin_cv']:.2f} (interface) vs {M['laws']['lamella_par_cv']:.2f} (diffusion)")
    for a in ax[0]:
        a.set_xticks([]); a.set_yticks([])
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=100)
    plt.close(fig)
    return path


def render_growth_gif(M: dict, path: Path = GROWTH_GIF, n_frames: int = 24) -> Path:
    from PIL import Image
    l3 = M["liesegang_3d"]
    size = 192
    scale = 86.0 / max(l3["r"].max(), 1)
    bands = l3["bands"] * scale
    n_fib = int(M["cv"]["fig196"]["striae_count"]) if M.get("cv") else 48
    vo = M["voronoi"]
    g = vo["labels"].shape[0]
    yy, xx = np.mgrid[0:g, 0:g]
    pts = vo["points"]
    d = np.sqrt(np.min((xx[None] - pts[:, 0, None, None]) ** 2 + (yy[None] - pts[:, 1, None, None]) ** 2, axis=0))
    walls = ndimage.maximum_filter(vo["labels"], 3) != ndimage.minimum_filter(vo["labels"], 3)
    e = M["dbm"]; eta_show = 1.0 if 1.0 in e else sorted(e)[-1]
    cl = e[eta_show]["cluster"]; c = cl.shape[0] // 2; R = int(e[eta_show]["R_max"]) + 3
    cl = cl[max(c - R, 0):c + R + 1, max(c - R, 0):c + R + 1]
    cy, cx = ndimage.center_of_mass(cl); rcl = np.hypot(*np.mgrid[0:cl.shape[0], 0:cl.shape[1]] - np.array([[[cy]], [[cx]]]))
    frames = []
    for i in range(n_frames):
        f = (i + 1) / n_frames
        R_ = 6 + (86 - 6) * i / (n_frames - 1)
        left = (_spherite_texture(size, R_, bands, n_fibres=n_fib) * 255).astype(np.uint8)
        grow = d <= 1.3 * f * g / np.sqrt(len(pts))
        mid = np.where(grow, 90 + 120 * ((vo["labels"] % 5) / 4.0), 235).astype(np.uint8); mid[grow & walls] = 20
        mid = np.asarray(Image.fromarray(mid).resize((size, size), Image.BILINEAR))
        part = cl & (rcl <= f * rcl[cl].max())
        right = np.asarray(Image.fromarray(np.where(part, 30, 235).astype(np.uint8)).resize((size, size), Image.NEAREST))
        sep = np.full((size, 6), 255, np.uint8)
        frames.append(Image.fromarray(np.concatenate([left, sep, mid, sep, right], axis=1)).convert("P"))
    path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=120, loop=0)
    return path


# =============================================================================== self-test
def run_self_test() -> int:
    checks = []
    integ = verify_dataset_integrity()
    checks.append(("dataset SHA-256 integrity", all(v == "ok" for v in integ.values()), integ))
    laws = radial_growth_laws()
    checks.append(("E1 constant lamella (CV = 0)", laws["lamella_lin_cv"] < 1e-12, laws["lamella_lin_cv"]))
    checks.append(("E2 lamellae thin outward", laws["lamella_par_monotone_decreasing"], laws["lamella_par"][:3]))
    fr = simulate_front_rounding(grid=161, n_frames=10)
    checks.append(("E3 isotropic asphericity decays ~1/R (slope in [-1.3,-0.7])", -1.3 < fr["decay_exponent"] < -0.7, fr["decay_exponent"]))
    fc = simulate_front_rounding(grid=161, n_frames=10, metric="chessboard")
    checks.append(("E3 CONTROL anisotropic deposition does not round (slope > -0.3)", fc["decay_exponent"] > -0.3, fc["decay_exponent"]))
    vo = simulate_voronoi_impingement(grid=300)
    checks.append(("E4 Voronoi mean neighbours ~6", 5.5 < vo["mean_neighbours"] < 6.5, vo["mean_neighbours"]))
    l2 = simulate_liesegang_radial(dim=2, t_end=600.0)
    checks.append(("E5 >= 3 discrete bands with P in (1, 1.6)", l2["n_bands"] >= 3 and 1.0 < l2["spacing_coefficient"] < 1.6, (l2["n_bands"], l2["spacing_coefficient"])))
    lc = simulate_liesegang_radial(dim=2, t_end=600.0, threshold=False)
    checks.append(("E5 CONTROL no nucleation gap -> <= 2 bands", lc["n_bands"] <= 2, lc["n_bands"]))
    e0 = simulate_dbm_cluster(0.0, n_sites=300, grid=121); e1 = simulate_dbm_cluster(1.0, n_sites=300, grid=121)
    checks.append(("E6 Eden more solid than DLA", e0["solidity"] > e1["solidity"], (e0["solidity"], e1["solidity"])))
    sf = spacing_fits(np.array([10, 12, 14.4, 17.28, 20.74]))
    checks.append(("spacing_fits recovers P = 1.2 on a geometric sequence", abs(sf["geo_P"] - 1.2) < 1e-3 and sf["geo_rmse"] < sf["arith_rmse"], sf["geo_P"]))
    f196 = measure_fig196()
    checks.append(("Fig. 196 segmentation: round disc (radius CV < 0.05) with radial border (RLI > 0.2)", f196["outer_radius_cv"] < 0.05 and f196["radial_line_index_border"] > 0.2, (f196["outer_radius_cv"], f196["radial_line_index_border"])))
    lsp = measure_liesegang_specimen()
    checks.append(("Liesegang specimen: >= 6 rings, geometric fit beats arithmetic", lsp["n_rings"] >= 6 and lsp["geo_rmse"] < lsp["arith_rmse"], (lsp["n_rings"], lsp["P_mean"])))
    f197 = measure_mosaic(load_plate("fig197R"))
    checks.append(("Fig. 197 segmentation: >= 40 interior cells", f197["n_cells"] >= 40, f197["n_cells"]))
    oo = measure_ooid()
    checks.append(("Ooid: outer boundary rounder than nucleus", oo["outer_asphericity"] < oo["nucleus_asphericity"], (oo["nucleus_asphericity"], oo["outer_asphericity"])))
    mm = measure_morphoeval(quick=True)
    checks.append(("measure_morphoeval contract (>= 18 metrics, all named)", len(mm) >= 18 and all("name" in d and "value" in d for d in mm), len(mm)))
    mm2 = measure_morphoeval(quick=True)
    checks.append(("determinism: repeated quick run identical", [d["value"] for d in mm] == [d["value"] for d in mm2], True))
    ok = True
    for name, passed, info in checks:
        print(f"[{'PASS' if passed else 'FAIL'}] {name}: {info}")
        ok &= bool(passed)
    print("SELF-TEST", "PASSED" if ok else "FAILED", f"({sum(1 for c in checks if c[1])}/{len(checks)})")
    return 0 if ok else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--quick", action="store_true", help="smaller grids / shorter runs")
    ap.add_argument("--no-gif", action="store_true")
    ap.add_argument("--no-result", action="store_true", help="do not write result.json")
    ap.add_argument("--etas", type=float, nargs="+", default=None, help="DBM supply exponents to sweep")
    ap.add_argument("--b0s", type=float, nargs="+", default=None, help="Liesegang outer-electrolyte concentrations to sweep")
    ap.add_argument("--a0", type=float, default=None, help="Liesegang inner reservoir concentration (2-D runs)")
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args(argv)
    if args.self_test:
        return run_self_test()
    M = run_models(quick=args.quick, etas=args.etas, b0s=args.b0s, a0=args.a0, rng_seed=args.seed)
    for k, v in M["integrity"].items():
        print(f"  {k}: {v}")
    for d in measure_morphoeval(quick=args.quick):
        print(f"  {d['name']}: {d['value']} {d['unit']}" + (f"  [ref {d['reference_value']}]" if "reference_value" in d else ""))
    ev = evaluate_claim(M)
    for k, v in ev["tests"].items():
        print(f"  [{'PASS' if v else 'FAIL'}] {k}")
    print(f"  VERDICT: {ev['verdict']} ({ev['n_pass']}/{ev['n_tests']})")
    print("->", render_comparison_png(M).relative_to(ROOT))
    if not args.no_gif:
        print("->", render_growth_gif(M).relative_to(ROOT))
    if not args.no_result:
        print("->", write_result(M).relative_to(ROOT))
    return 0


if __name__ == "__main__":
    sys.exit(main())

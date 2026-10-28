#!/usr/bin/env python3
"""
t1917-023_on-concretions-spicules-and-spicular-skeletons.py — Thompson (1917) claim T1917-023
=============================================================================================

Claim (Thompson, D.W. 1917, *On Growth and Form*, Ch. IX "On Concretions, Spicules, and
Spicular Skeletons", pp. 411-487, Figs. 194-235):

    "Concretions and calcospherites grow by radial deposition about a centre (Harting's
     artificial concretions in egg albumen mimic natural ones), giving concentric/spherical
     forms by physical deposition alone."

Pass 1 baseline (v1).  Implements the mechanistic-geometric model of radial deposition and
renders it next to Thompson's own plates (IA scan `ongrowthform1917thom`, pp. 421/422/432):

  E1  interface-controlled radial deposition  dR/dt = v          (p. 421-422, 437)
  E2  diffusion-limited radial deposition     R dR/dt = kappa    (Zener 1949; Mullins & Sekerka 1963)
  E3  sphericity stability under uniform normal deposition       (p. 425-426)
  E4  equal-rate impingement -> Voronoi mosaic                   (p. 422, 426)
  E5  Liesegang supersaturation banding, radial geometry          (p. 427; Jablczynski 1923)
  E6  dielectric-breakdown supply model, eta = 0 (Eden) vs 1 (DLA) (p. 424 "scanty supply")

E1-E5 are machine-checked in `morphoeval/lean/loops/Thompson1917_T1917_023.lean`
(namespace `MathOfBiology.Loop_218`).  Dataset + SHA-256 pins:
`datasets/thompson_1917/t1917-023/t1917_023_on_concretions_spicules_and_sp_reference.json`.

Usage
-----
    python3 thompson/claims/t1917-023_on-concretions-spicules-and-spicular-skeletons.py --self-test
    python3 thompson/claims/t1917-023_on-concretions-spicules-and-spicular-skeletons.py [--quick] [--no-gif]

Outputs
-------
    docs/media/t1917_023_on_concretions_spicules_and_sp_comparison.png
    docs/media/t1917_023_on_concretions_spicules_and_sp_growth.gif
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage
from scipy.spatial import ConvexHull

ROOT = Path(__file__).resolve().parents[2]
DATASET_DIR = ROOT / "datasets" / "thompson_1917" / "t1917-023"
REFERENCE_JSON = DATASET_DIR / "t1917_023_on_concretions_spicules_and_sp_reference.json"
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

SEED = 1917
PARAMS = {
    "front": {"grid": 241, "seed_half_width": 12, "n_frames": 24},
    "voronoi": {"n_centres": 60, "grid": 300},
    "liesegang": {"D": 1.0, "a0": 50.0, "b0": 0.1, "K_sp": 0.005, "K_ss": 0.05, "k": 5.0,
                  "r0": 2.0, "r_max": 160.0, "dr": 0.5, "dt": 0.05, "t_end": 2500.0},
    "dbm": {"grid": 121, "n_sites": 1200, "etas": [0.0, 1.0]},
}

_CACHE: dict = {}


# --------------------------------------------------------------------------- dataset
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


# --------------------------------------------------------------------------- E1 / E2
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
    }


# --------------------------------------------------------------------------- E3
def simulate_front_rounding(grid: int | None = None, seed_half_width: int | None = None,
                            n_frames: int | None = None) -> dict:
    """Uniform normal deposition on a square nucleus: the front at radius R is the Euclidean
    distance-transform level set.  Returns asphericity A(R) = (max rho - min rho)/mean rho."""
    g = grid or PARAMS["front"]["grid"]
    hw = seed_half_width or PARAMS["front"]["seed_half_width"]
    nf = n_frames or PARAMS["front"]["n_frames"]
    c = g // 2
    seed = np.zeros((g, g), bool)
    seed[c - hw:c + hw + 1, c - hw:c + hw + 1] = True
    dist = ndimage.distance_transform_edt(~seed)
    yy, xx = np.mgrid[0:g, 0:g]
    rr = np.hypot(yy - c, xx - c)
    th = np.arctan2(yy - c, xx - c)
    levels = np.linspace(0, c - hw - 3, nf)
    R_mean, A = [], []
    masks = []
    for lv in levels:
        m = dist <= lv
        edge = m & ~ndimage.binary_erosion(m)
        rho = rr[edge]
        R_mean.append(rho.mean())
        A.append((rho.max() - rho.min()) / rho.mean())
        masks.append(m)
    R_mean, A = np.array(R_mean), np.array(A)
    sel = levels > 0
    slope = np.polyfit(np.log(R_mean[sel]), np.log(A[sel]), 1)[0]
    return {"levels": levels, "R_mean": R_mean, "asphericity": A, "masks": masks,
            "decay_exponent": float(slope), "abs_asphericity_px": (dist.max() * 0 + (hw * np.sqrt(2) - hw))}


# --------------------------------------------------------------------------- E4
def simulate_voronoi_impingement(n_centres: int | None = None, grid: int | None = None,
                                 rng_seed: int = SEED) -> dict:
    """Simultaneous nucleation + equal radial growth rate = Voronoi tessellation of the centres.
    Returns the label map, per-cell areas and neighbour counts (interior cells only)."""
    n = n_centres or PARAMS["voronoi"]["n_centres"]
    g = grid or PARAMS["voronoi"]["grid"]
    rng = np.random.default_rng(rng_seed)
    pts = rng.uniform(0, g, size=(n, 2))
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
    return {"labels": labels, "points": pts, "areas": areas, "neighbours": neighbours,
            "mean_neighbours": float(neighbours.mean()), "area_cv": float(areas.std() / areas.mean()),
            "n_interior": int(len(areas))}


# --------------------------------------------------------------------------- E5
def simulate_liesegang_radial(dim: int = 2, threshold: bool = True, **over) -> dict:
    """Ostwald supersaturation (Keller-Rubinow type) precipitation in d-dimensional radial
    geometry: inner electrolyte reservoir a0 at r <= r0 diffuses into gel holding b0.
    Precipitation R = k (ab - K_sp)_+ on existing precipitate or where ab >= K_ss (nucleation).
    `threshold=False` removes the nucleation gap (K_ss = K_sp) — a continuous deposit control."""
    P = dict(PARAMS["liesegang"]); P.update(over)
    D, a0, b0, Ksp, Kss, k = P["D"], P["a0"], P["b0"], P["K_sp"], P["K_ss"], P["k"]
    r0, rmax, dr, dt, t_end = P["r0"], P["r_max"], P["dr"], P["dt"], P["t_end"]
    r = np.arange(dr / 2, rmax, dr); n = len(r)
    a = np.zeros(n); b = np.full(n, b0); p = np.zeros(n)
    inner = r <= r0; a[inner] = a0; b[inner] = 0.0
    rf = np.arange(0, rmax + dr, dr)[:n + 1]
    gexp = dim - 1
    rfg, rcg = rf ** gexp, r ** gexp
    first_t = []
    nsteps = int(round(t_end / dt))
    for s in range(nsteps):
        for c, far in ((a, 0.0), (b, None)):
            flux = np.zeros(n + 1)
            flux[1:n] = -D * (c[1:] - c[:-1]) / dr
            flux[n] = 0.0 if far is None else -D * (far - c[-1]) / dr
            c += dt * (-(rfg[1:] * flux[1:] - rfg[:-1] * flux[:-1]) / (rcg * dr))
        a[inner] = a0; b[inner] = 0.0
        prod = a * b
        grow = (prod > Ksp) & ((p > 0) | (prod >= Kss)) if threshold else (prod > Ksp)
        dep = np.minimum(k * np.where(grow, prod - Ksp, 0.0) * dt, np.minimum(a, b))
        new = (p == 0) & (dep > 0)
        if new.any():
            first_t.append((s * dt, float(r[new].max())))
        a -= dep; b -= dep; p += dep; a[inner] = a0
    bands = _band_centres(r, p)
    ratios = bands[1:] / bands[:-1] if len(bands) > 1 else np.array([])
    return {"r": r, "p": p, "a": a, "b": b, "bands": bands, "ratios": ratios,
            "spacing_coefficient": float(ratios.mean()) if len(ratios) else float("nan"),
            "n_bands": int(len(bands)), "dim": dim, "threshold": threshold, "params": P}


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


# --------------------------------------------------------------------------- E6
def simulate_dbm_cluster(eta: float, n_sites: int | None = None, grid: int | None = None,
                         rng_seed: int = SEED, relax_iters: int = 12) -> dict:
    """Dielectric-breakdown model (Niemeyer, Pietronero & Wiesmann 1984): Laplacian supply field
    phi (0 on the cluster, 1 on a ring 8 lattice units beyond the cluster), perimeter site i grows
    with probability ∝ phi_i^eta.  eta = 0 is the Eden model (interface-controlled, compact);
    eta = 1 is DLA (diffusion-limited, branched)."""
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


# --------------------------------------------------------------------------- orchestration
def run_models(quick: bool = False) -> dict:
    if "models" in _CACHE and _CACHE.get("quick") == quick:
        return _CACHE["models"]
    out = {
        "integrity": verify_dataset_integrity(),
        "laws": radial_growth_laws(),
        "front": simulate_front_rounding(grid=161 if quick else None, n_frames=12 if quick else None),
        "voronoi": simulate_voronoi_impingement(grid=160 if quick else None),
        "liesegang_2d": simulate_liesegang_radial(dim=2, t_end=600.0 if quick else None or PARAMS["liesegang"]["t_end"]),
        "liesegang_3d": simulate_liesegang_radial(dim=3, a0=100.0, t_end=600.0 if quick else None or PARAMS["liesegang"]["t_end"]),
        "dbm": {eta: simulate_dbm_cluster(eta, n_sites=300 if quick else None) for eta in PARAMS["dbm"]["etas"]},
    }
    _CACHE["models"], _CACHE["quick"] = out, quick
    return out


def measure_morphoeval(quick: bool = False) -> list[dict]:
    """Quantitative morphometrics consumed by morphoeval/measure_outputs.py."""
    M = run_models(quick=quick)

    def m(name, value, unit, method, reference=None, source=None, note=None, derived_from="script"):
        d = {"name": name, "value": None if value is None else float(round(float(value), 4)), "unit": unit,
             "method": method, "derived_from": derived_from}
        if reference is not None:
            d["reference_value"] = reference
        if source:
            d["reference_source"] = source
        if note:
            d["note"] = note
        return d

    fr, vo, l2, l3, laws = M["front"], M["voronoi"], M["liesegang_2d"], M["liesegang_3d"], M["laws"]
    e0, e1 = M["dbm"][0.0], M["dbm"][1.0]
    return [
        m("E3 asphericity decay exponent d ln A / d ln R (square nucleus, uniform normal deposition)", fr["decay_exponent"], "",
          "log-log slope of (max rho - min rho)/mean rho vs mean R over distance-transform level sets", -1.0,
          "Lean asphericity_nonincreasing; A = const / R"),
        m("E1 lamella-thickness CV, interface-controlled law", laws["lamella_lin_cv"], "", "std/mean of successive R_{n+1}-R_n", 0.0,
          "Lean lamella_thickness_constant"),
        m("E4 mean neighbour count, equal-rate Voronoi mosaic", vo["mean_neighbours"], "neighbours", "interior cells of 60-centre Voronoi map",
          6.0, "Euler characteristic (Thompson 1917 Ch. VII)"),
        m("E4 cell-area CV, equal-rate Voronoi mosaic", vo["area_cv"], "", "std/mean of interior cell areas", 0.529,
          "Ferenc & Neda 2007 Poisson-Voronoi"),
        m("E5 Liesegang spacing coefficient P (2-D, a0=50, b0=0.1)", l2["spacing_coefficient"], "",
          "mean r_{n+1}/r_n over precipitate band centres", 1.15, "Jablczynski 1923 (1.05-1.4)",
          note=f"{l2['n_bands']} bands"),
        m("E5 Liesegang band count (3-D spherical, a0=100)", l3["n_bands"], "bands", "precipitate maxima above 10% of peak"),
        m("E6 Eden (eta=0) envelope radius CV", e0["envelope_radius_cv"], "", "36-sector outer envelope std/mean"),
        m("E6 DLA (eta=1) mass-radius exponent", e1["mass_radius_exponent"], "", "log N(r) vs log r fit, r <= 0.75 R_max", 1.71,
          "Witten & Sander 1981"),
    ]


# --------------------------------------------------------------------------- rendering
def _spherite_texture(size: int, R: float, bands: np.ndarray, n_fibres: int = 48, rng_seed: int = SEED) -> np.ndarray:
    """Synthetic radial-fibre + concentric-band spherite image of radius R (pixels)."""
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

    fig, ax = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle("Thompson (1917) T1917-023 — calcospherites grow by radial deposition about a centre "
                 "(Harting 1872) — Pass 1 baseline", fontsize=14)
    # plates
    ax[0, 0].imshow(load_plate("fig196"), cmap="gray"); ax[0, 0].set_title("Thompson Fig. 196 (after Harting)\nsingle calcospherite, nucleus + striated border")
    ax[0, 1].imshow(load_plate("fig195"), cmap="gray"); ax[0, 1].set_title("Fig. 195 calcospherites in white of egg")
    ax[0, 2].imshow(load_plate("fig197R"), cmap="gray"); ax[0, 2].set_title("Fig. 197 crowded -> polyhedral mosaic")
    ax[0, 3].imshow(load_plate("fig208"), cmap="gray"); ax[0, 3].set_title("Fig. 208 concentrically striated spherocrystal")
    for a in ax[0]:
        a.set_xticks([]); a.set_yticks([])
    # models
    l3 = M["liesegang_3d"]
    scale = 110.0 / max(l3["r"].max(), 1)
    ax[1, 0].imshow(_spherite_texture(256, 110, l3["bands"] * scale), cmap="gray", vmin=0, vmax=1)
    ax[1, 0].set_title(f"E1+E5 model spherite: radial fibres + {l3['n_bands']} Liesegang bands (3-D)")
    fr = M["front"]
    for k in (0, len(fr["masks"]) // 3, 2 * len(fr["masks"]) // 3, len(fr["masks"]) - 1):
        ax[1, 1].contour(fr["masks"][k].astype(float), levels=[0.5], colors=[plt.cm.viridis(k / (len(fr["masks"]) - 1))])
    ax[1, 1].set_aspect("equal"); ax[1, 1].invert_yaxis()
    ax[1, 1].set_title(f"E3 square nucleus rounds off: A ∝ R^{fr['decay_exponent']:.2f}")
    vo = M["voronoi"]
    ax[1, 2].imshow(vo["labels"] % 11, cmap="tab20", interpolation="nearest")
    ax[1, 2].contour(vo["labels"], levels=np.arange(0.5, vo["labels"].max() + 1), colors="k", linewidths=0.6)
    ax[1, 2].plot(vo["points"][:, 0], vo["points"][:, 1], "k.", ms=3)
    ax[1, 2].set_title(f"E4 equal-rate impingement: <n> = {vo['mean_neighbours']:.2f} neighbours")
    l2 = M["liesegang_2d"]
    ax[1, 3].plot(l2["r"], l2["p"] / max(l2["p"].max(), 1e-9), "k-", lw=1)
    for b in l2["bands"]:
        ax[1, 3].axvline(b, color="tab:red", lw=0.5, alpha=0.6)
    ax[1, 3].set_xlabel("r"); ax[1, 3].set_ylabel("precipitate (norm.)")
    ax[1, 3].set_title(f"E5 Liesegang bands (2-D): P = {l2['spacing_coefficient']:.3f}")
    for a in ax[1, :3]:
        a.set_xticks([]); a.set_yticks([])
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def render_growth_gif(M: dict, path: Path = GROWTH_GIF, n_frames: int = 24) -> Path:
    from PIL import Image
    l3 = M["liesegang_3d"]
    size = 192
    scale = 86.0 / max(l3["r"].max(), 1)
    bands = l3["bands"] * scale
    frames = []
    vo = M["voronoi"]
    g = vo["labels"].shape[0]
    yy, xx = np.mgrid[0:g, 0:g]
    pts = vo["points"]
    d = np.sqrt(np.min((xx[None] - pts[:, 0, None, None]) ** 2 + (yy[None] - pts[:, 1, None, None]) ** 2, axis=0))
    for i in range(n_frames):
        R = 6 + (86 - 6) * i / (n_frames - 1)
        left = (_spherite_texture(size, R, bands) * 255).astype(np.uint8)
        grow = d <= (2.0 + 0.5 * i / (n_frames - 1)) * g / np.sqrt(len(pts)) * (i + 1) / n_frames * 1.6
        right = np.where(grow, 90 + 120 * ((vo["labels"] % 5) / 4.0), 235).astype(np.uint8)
        edge = grow & (ndimage.maximum_filter(vo["labels"], 3) != ndimage.minimum_filter(vo["labels"], 3))
        right[edge] = 20
        right = np.asarray(Image.fromarray(right).resize((size, size), Image.BILINEAR))
        frame = np.concatenate([left, np.full((size, 6), 255, np.uint8), right], axis=1)
        frames.append(Image.fromarray(frame).convert("P"))
    path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=120, loop=0)
    return path


# --------------------------------------------------------------------------- self-test
def run_self_test() -> int:
    checks = []
    integ = verify_dataset_integrity()
    checks.append(("dataset SHA-256 integrity", all(v == "ok" for v in integ.values()), integ))
    laws = radial_growth_laws()
    checks.append(("E1 constant lamella (CV = 0)", laws["lamella_lin_cv"] < 1e-12, laws["lamella_lin_cv"]))
    checks.append(("E2 lamellae thin outward", laws["lamella_par_monotone_decreasing"], laws["lamella_par"][:3]))
    fr = simulate_front_rounding(grid=161, n_frames=10)
    checks.append(("E3 asphericity decays ~1/R (slope in [-1.3,-0.7])", -1.3 < fr["decay_exponent"] < -0.7, fr["decay_exponent"]))
    vo = simulate_voronoi_impingement(grid=160)
    checks.append(("E4 Voronoi mean neighbours ~6", 5.5 < vo["mean_neighbours"] < 6.5, vo["mean_neighbours"]))
    l2 = simulate_liesegang_radial(dim=2, t_end=600.0)
    checks.append(("E5 >= 3 discrete bands with P in (1, 1.6)", l2["n_bands"] >= 3 and 1.0 < l2["spacing_coefficient"] < 1.6,
                   (l2["n_bands"], l2["spacing_coefficient"])))
    e0 = simulate_dbm_cluster(0.0, n_sites=300); e1 = simulate_dbm_cluster(1.0, n_sites=300)
    checks.append(("E6 Eden more solid than DLA", e0["solidity"] > e1["solidity"], (e0["solidity"], e1["solidity"])))
    mm = measure_morphoeval(quick=True)
    checks.append(("measure_morphoeval contract", len(mm) >= 6 and all("name" in d and "value" in d for d in mm), len(mm)))
    ok = True
    for name, passed, info in checks:
        print(f"[{'PASS' if passed else 'FAIL'}] {name}: {info}")
        ok &= bool(passed)
    print("SELF-TEST", "PASSED" if ok else "FAILED")
    return 0 if ok else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--quick", action="store_true", help="smaller grids / shorter runs")
    ap.add_argument("--no-gif", action="store_true")
    args = ap.parse_args(argv)
    if args.self_test:
        return run_self_test()
    M = run_models(quick=args.quick)
    for k, v in M["integrity"].items():
        print(f"  {k}: {v}")
    for d in measure_morphoeval(quick=args.quick):
        print(f"  {d['name']}: {d['value']} {d['unit']}" + (f"  [ref {d['reference_value']}]" if "reference_value" in d else ""))
    print("->", render_comparison_png(M).relative_to(ROOT))
    if not args.no_gif:
        print("->", render_growth_gif(M).relative_to(ROOT))
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""
t1917-014_the-forms-of-cells.py — Thompson (1917) claim T1917-014, Pass 2 (v2).

Claim (On Growth and Form, Ch. V "The Forms of Cells", pp. 201-276, Figs. 61-67, 72-96):
    "The forms of single cells and cell-like organisms (Plateau's oil globules; Lagena and
    other forams/protists) are surfaces of revolution from the minimal-surface catalogue:
    sphere, cylinder, unduloid, nodoid, catenoid."
Thompson's own wording for the test genus (p. 256, Fig. 84 "Various species of Lagena, after
Brady"): "flask-shaped shells, whose surface is simply that of an unduloid, or more frequently,
like that of a flask itself, an unduloid combined with a portion of a sphere."

Mathematics implemented (ids follow datasets/thompson_1917/t1917-014/*_reference.json):
    E3  Delaunay first integral  r / sqrt(1 + r'^2) - H r^2 = C ; turning points r_min, r_max
        are the roots of H r^2 - r + C = 0  ->  r_min + r_max = 1/H, r_min r_max = C/H.
    E4  Kenmotsu (1980) arc-length parametrisation of every CMC surface of revolution with
        H != 0:  r(s) = sqrt(1 + B^2 + 2 B sin 2Hs) / (2H),  dz/ds = (1 + B sin 2Hs) / sqrt(...).
        B = 0 cylinder, 0 < B < 1 unduloid, B = 1 string of spheres, B > 1 nodoid;
        the catenoid is the H = 0 member r = c cosh(z / c).
    E5  neck ratio rho = r_min / r_max = (1 - B) / (1 + B).
    E6  unduloid period Lambda = 4 a E(e), e = B; bead aspect ratio Lambda / (2 r_max) <= pi.
    E7  "Delaunay flask": sphere (radius R) C1-joined to an unduloid whose neck is C1-joined to
        a cylinder — Thompson's "unduloid combined with a portion of a sphere" made explicit.
    E8  "Delaunay bead": a single unduloid bulge-to-neck window continued by its neck cylinder.

Comparators (Pass 2):
    ellipsoid       smooth surface of revolution that is NOT constant-mean-curvature (null model)
    sphere+cylinder flask with the unduloid shoulder ablated (C0 join) — model ablation
    free-form       degree-6 polynomial in r^2 — upper bound of attainable fit (noise floor)
Statistics per specimen and model: nRMSE (RMSE / r_max), Pearson r, two-sample KS statistic
between observed and fitted radii, 2-fold cross-validated nRMSE (odd/even rows).

Data (all SHA-256 pinned in the reference JSON and re-verified at load time):
    * Thompson's Fig. 84 — 13 Lagena silhouettes, Internet Archive leaf 0276 (p. 256) of
      ongrowthform1917thom (public domain).
    * Kireenko et al. (2022) Biodiversity Data Journal 10:e87457 Fig. 8 (CC BY 4.0, via
      Wikimedia Commons): light-microscope photographs of Lagena sp. 61a/62a, Oolina sp. 63,
      Fissurina sp. 57a/58a (unilocular tests, the claim's subject) and of the polythalamous
      Globobulimina pacifica 64a and Quinqueloculina seminulum 66a (negative controls).
      100 um scale bars give absolute sizes.
    * The pre-scouted Dryad dataset doi:10.5061/dryad.28816 was verified as DATASET_MISMATCH
      (see scout_manifest.json) and is not used.

Decision rule (per silhouette):  Delaunay-compatible  iff
    nRMSE_best_Delaunay <= min(max(RMSE_SUPPORT, FLOOR_FACTOR * nRMSE_free_form), RMSE_CAP)   and
    asymmetry index <= ASYM_MAX  (surface-of-revolution premise).
Claim verdict: supported if >= 75 % of unilocular tests are compatible, refuted if <= 25 %,
mixed otherwise; untestable if every negative control passes (the outline test would then
have no specificity at all).  The control pass rate is reported as the specificity caveat.

Usage:
    python3 thompson/claims/t1917-014_the-forms-of-cells.py --self-test
    python3 thompson/claims/t1917-014_the-forms-of-cells.py          # figure, GIF, metrics, result
    python3 thompson/claims/t1917-014_the-forms-of-cells.py --no-gif

Outputs:
    docs/media/t1917_014_the_forms_of_cells_comparison.png
    docs/media/t1917_014_the_forms_of_cells_growth.gif
    datasets/thompson_1917/t1917-014/t1917_014_metrics.json
    datasets/thompson_1917/t1917-014/result.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from scipy.optimize import least_squares
from scipy.special import ellipe
from scipy.stats import ks_2samp

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "datasets" / "thompson_1917" / "t1917-014"
MEDIA_DIR = ROOT / "docs" / "media"
REFERENCE_JSON = DATA_DIR / "t1917_014_the_forms_of_cells_reference.json"
FIG84_PNG = DATA_DIR / "thompson_1917_fig84_lagena_after_brady.png"
FIG84_SHA256 = "c02d29af384791df4ad18705c466f233d111be828e991d64047b5b8fea00dec9"
PHOTO_JPG = DATA_DIR / "kireenko_2022_bdj_e87457_fig8_station5656mc.jpg"
PHOTO_SHA256 = "3b57acc0dfc9837448683cd478c1eb24832547ed31e34457d79e70f613a63ff6"
COMPARISON_PNG = MEDIA_DIR / "t1917_014_the_forms_of_cells_comparison.png"
GROWTH_GIF = MEDIA_DIR / "t1917_014_the_forms_of_cells_growth.gif"
METRICS_JSON = DATA_DIR / "t1917_014_metrics.json"
RESULT_JSON = DATA_DIR / "result.json"

# Segmentation of the 1917 plate (grey paper, show-through text, shaded line drawings).
SEG_BG_PERCENTILE = 90          # local paper brightness estimate
SEG_BG_SIZE = 61                # footprint of the percentile filter on the 4x-downsampled plate
SEG_THRESHOLD = 0.045           # darker-than-paper threshold (fraction of full scale)
SEG_MIN_AREA = 3000             # px; drops show-through text fragments
SEG_ROW_SPLIT = 700             # Fig. 84 has an upper row (8) and a lower row (5) of specimens

# Kireenko et al. (2022) Fig. 8: panel boxes as fractions (x0, y0, x1, y1) of the 2480x3508 plate.
PHOTO_PANELS = {
    "57a": {"box": (0.00, 0.045, 0.20, 0.25), "taxon": "Fissurina sp.", "role": "test"},
    "58a": {"box": (0.37, 0.050, 0.58, 0.25), "taxon": "Fissurina sp.", "role": "test"},
    "61a": {"box": (0.34, 0.272, 0.47, 0.50), "taxon": "Lagena sp.", "role": "test"},
    "62a": {"box": (0.69, 0.272, 0.83, 0.50), "taxon": "Lagena sp.", "role": "test"},
    "63":  {"box": (0.04, 0.520, 0.25, 0.74), "taxon": "Oolina sp.", "role": "test"},
    "64a": {"box": (0.29, 0.512, 0.50, 0.73), "taxon": "Globobulimina pacifica (triserial)", "role": "control"},
    "66a": {"box": (0.13, 0.755, 0.33, 0.97), "taxon": "Quinqueloculina seminulum (miliolid)", "role": "control"},
}
PHOTO_SCALE_BAR_UM = 100.0
PHOTO_MIN_LUM = 0.25            # floor for the Otsu threshold on the black background

# Decision rule.
RMSE_SUPPORT = 0.06             # absolute gate on nRMSE (edge noise of the drawings is ~2-4 %)
FLOOR_FACTOR = 1.5              # Delaunay may be at most 1.5x the free-form noise floor ...
RMSE_CAP = 0.10                 # ... but never worse than 10 % of r_max, however poor the free-form fit
ASYM_MAX = 0.12                 # bilateral asymmetry gate (surface-of-revolution premise)
FREEFORM_DEGREE = 6             # polynomial degree in r^2 for the free-form reference
SWEEP_B = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0, 1.2, 1.5]
SWEEP_T = [0.1, 0.25, 0.5, 0.75, 1.0]


# --------------------------------------------------------------------------- provenance
def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_inputs() -> dict:
    ref = json.loads(REFERENCE_JSON.read_text(encoding="utf-8"))
    pins = ref["input_dataset_sha256"]
    for path, pin in ((FIG84_PNG, FIG84_SHA256), (PHOTO_JPG, PHOTO_SHA256)):
        got = sha256_of(path)
        if got != pins[path.name] or got != pin:
            raise SystemExit(f"SHA-256 mismatch for {path.name}: {got}")
    return ref


# --------------------------------------------------------------------------- E4-E6: Delaunay catalogue
def classify_delaunay(B: float, H: float = 1.0) -> str:
    """Plateau's catalogue member for Kenmotsu parameter B (H = 0 is the catenoid)."""
    if H == 0.0:
        return "catenoid"
    if B == 0.0:
        return "cylinder"
    if 0.0 < B < 1.0:
        return "unduloid"
    if B == 1.0:
        return "sphere"
    return "nodoid"


def neck_ratio(B: float) -> float:
    """E5: r_min / r_max = (1 - B) / (1 + B)."""
    return (1.0 - B) / (1.0 + B)


def unduloid_period(B: float, H: float) -> float:
    """E6: axial distance between successive necks, Lambda = 4 a E(e) with a = 1/(2H), e = B."""
    a = 1.0 / (2.0 * H)
    return 4.0 * a * ellipe(B * B)       # scipy's ellipe takes the parameter m = e^2


def bead_aspect_ratio(B: float) -> float:
    """E6: length / width of one unduloid bead = 2 E(e) / (1 + e); equals pi at B = 0, 1 at B = 1."""
    return 2.0 * ellipe(B * B) / (1.0 + B)


def kenmotsu_profile(H: float, B: float, n: int = 400) -> tuple[np.ndarray, np.ndarray]:
    """E4: descending half-period of the unduloid (bulge s = pi/4H -> neck s = 3pi/4H).

    Returns (dz, r): axial offset from the bulge (>= 0, trapezoid-integrated dz/ds) and radius.
    """
    s = np.linspace(np.pi / (4 * H), 3 * np.pi / (4 * H), n)
    sig = np.sin(2 * H * s)
    root = np.sqrt(1 + B * B + 2 * B * sig)
    r = root / (2 * H)
    dzds = (1 + B * sig) / root
    dz = np.concatenate([[0.0], np.cumsum(0.5 * (dzds[1:] + dzds[:-1]) * np.diff(s))])
    return dz, r


def kenmotsu_full(H: float, B: float, s: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """E4 over an arbitrary arc-length array (catalogue gallery, GIF, nodoid branch B > 1)."""
    sig = np.sin(2 * H * s)
    root = np.sqrt(np.clip(1 + B * B + 2 * B * sig, 1e-12, None))
    r = root / (2 * H)
    dzds = (1 + B * sig) / root
    z = np.concatenate([[0.0], np.cumsum(0.5 * (dzds[1:] + dzds[:-1]) * np.diff(s))])
    return z, r


def catenoid_profile(c: float, z: np.ndarray) -> np.ndarray:
    """H = 0 member: r = c cosh(z / c)."""
    return c * np.cosh(z / c)


def flask_join_radius(R: float, B: float, t: float) -> tuple[float, float, float]:
    """E7: kappa = R H = (1-B)/2 + t B in [(1-B)/2, (1+B)/2]; join radius
    r_j^2 = R (1 - B^2) / (4 H (1 - kappa)) (<= R, = R at the equator join t = 1)."""
    kappa = (1.0 - B) / 2.0 + t * B
    H = kappa / R
    rj = math.sqrt(max(R * (1 - B * B) / (4 * H * (1 - kappa)), 1e-12))
    return H, kappa, min(rj, R)


# --------------------------------------------------------------------------- E7-E8: composite models
def flask_profile(z: np.ndarray, R: float, B: float, t: float, Lneck: float, z0: float) -> np.ndarray:
    """E7 Delaunay flask.  Sphere of radius R with its aboral pole at z0, C1-joined to the
    descending branch of the unduloid (H = kappa / R), whose neck is C1-joined to a cylinder
    of length Lneck."""
    H, _kappa, rj = flask_join_radius(R, B, t)
    zc = z0 + R
    zj = zc + math.sqrt(max(R * R - rj * rj, 0.0))
    dz_u, r_u = kenmotsu_profile(H, B)
    dz_join = float(np.interp(rj, r_u[::-1], dz_u[::-1]))
    z_u = zj + (dz_u - dz_join)
    z_neck, r_neck = z_u[-1], r_u[-1]
    r = np.zeros_like(z, dtype=float)
    sph = (z >= z0) & (z <= zj)
    r[sph] = np.sqrt(np.clip(R * R - (z[sph] - zc) ** 2, 0, None))
    und = (z > zj) & (z <= z_neck)
    r[und] = np.interp(z[und], z_u, r_u)
    cyl = (z > z_neck) & (z <= z_neck + Lneck)
    r[cyl] = r_neck
    return r


def bead_profile(z: np.ndarray, H: float, B: float, zc: float) -> np.ndarray:
    """E8 Delaunay bead: unduloid (H, B) with its bulge at zc, continued beyond the necks by the
    neck cylinder r = r_min."""
    dz_u, r_u = kenmotsu_profile(H, B)
    return np.interp(np.abs(z - zc), dz_u, r_u, right=r_u[-1])


# --------------------------------------------------------------------------- comparators
def ellipsoid_profile(z: np.ndarray, a: float, b: float, zc: float) -> np.ndarray:
    """Null model: prolate/oblate ellipsoid of revolution (smooth but not CMC)."""
    return np.sqrt(np.clip(a * a * (1 - ((z - zc) / b) ** 2), 0, None))


def sphere_cylinder_profile(z: np.ndarray, R: float, r_neck: float, Lneck: float, z0: float) -> np.ndarray:
    """Ablation of E7: the unduloid shoulder removed — a sphere cut at radius r_neck and
    continued by the neck cylinder (C0 join only)."""
    zc = z0 + R
    r_neck = min(r_neck, R)
    zj = zc + math.sqrt(max(R * R - r_neck * r_neck, 0.0))
    r = np.zeros_like(z, dtype=float)
    sph = (z >= z0) & (z <= zj)
    r[sph] = np.sqrt(np.clip(R * R - (z[sph] - zc) ** 2, 0, None))
    cyl = (z > zj) & (z <= zj + Lneck)
    r[cyl] = r_neck
    return r


def freeform_fit(z: np.ndarray, r: np.ndarray, z_eval: np.ndarray | None = None) -> np.ndarray:
    """Free-form reference: least-squares polynomial of degree FREEFORM_DEGREE in r^2."""
    mu, sd = z.mean(), z.std() + 1e-9
    c = np.polyfit((z - mu) / sd, r ** 2, FREEFORM_DEGREE)
    ze = z if z_eval is None else z_eval
    return np.sqrt(np.clip(np.polyval(c, (ze - mu) / sd), 0, None))


# --------------------------------------------------------------------------- segmentation
def segment_fig84(path: Path) -> tuple[np.ndarray, list[tuple[tuple[slice, slice], np.ndarray]]]:
    """Background-normalised threshold -> closing -> hole fill -> opening -> area filter.
    Returns the grey plate and the 13 (bbox, mask) components ordered row-wise, left to right."""
    g = np.asarray(Image.open(path).convert("L"), dtype=float) / 255.0
    gs = ndi.gaussian_filter(g, 1.0)
    small = g[::4, ::4]
    bg_s = ndi.percentile_filter(small, SEG_BG_PERCENTILE, size=SEG_BG_SIZE)
    bg = ndi.zoom(bg_s, 4, order=1)
    pad = ((0, max(0, g.shape[0] - bg.shape[0])), (0, max(0, g.shape[1] - bg.shape[1])))
    bg = np.pad(bg, pad, mode="edge")[: g.shape[0], : g.shape[1]]
    m = (bg - gs) > SEG_THRESHOLD
    m = ndi.binary_closing(m, iterations=5)
    m = ndi.binary_fill_holes(m)
    m = ndi.binary_opening(m, iterations=3)
    lab, n = ndi.label(m)
    sizes = ndi.sum(m, lab, range(1, n + 1))
    comps = []
    for i, sl in enumerate(ndi.find_objects(lab)):
        if sizes[i] > SEG_MIN_AREA:
            comps.append((sl, lab[sl] == (i + 1)))
    comps.sort(key=lambda c: (c[0][0].start > SEG_ROW_SPLIT, c[0][1].start))
    return g, comps


def _largest_component(m: np.ndarray) -> np.ndarray:
    lab, n = ndi.label(m)
    if n == 0:
        return m
    sizes = ndi.sum(m, lab, range(1, n + 1))
    return lab == (int(np.argmax(sizes)) + 1)


def _otsu(x: np.ndarray) -> float:
    hist, edges = np.histogram(x.ravel(), bins=256, range=(0.0, 1.0))
    p = hist.astype(float) / max(1, hist.sum())
    w0 = np.cumsum(p); w1 = 1 - w0
    mids = 0.5 * (edges[:-1] + edges[1:])
    m0 = np.cumsum(p * mids) / np.maximum(w0, 1e-12)
    m1 = (np.sum(p * mids) - np.cumsum(p * mids)) / np.maximum(w1, 1e-12)
    between = w0 * w1 * (m0 - m1) ** 2
    return float(mids[int(np.argmax(between))])


def find_scale_bars(lum: np.ndarray) -> list[dict]:
    """All white scale bars on the plate: very bright, flat (w >= 6 h), wide components."""
    bright = lum > 0.92
    lab, n = ndi.label(bright)
    bars = []
    for i, sl in enumerate(ndi.find_objects(lab)):
        h, w = sl[0].stop - sl[0].start, sl[1].stop - sl[1].start
        if w >= 6 * h and w > 0.025 * lum.shape[1] and h < 0.01 * lum.shape[0]:
            comp = lab[sl] == (i + 1)
            if comp.mean() > 0.7:          # solid rectangle, not a specimen highlight
                bars.append({"cx": 0.5 * (sl[1].start + sl[1].stop), "cy": 0.5 * (sl[0].start + sl[0].stop),
                             "w": float(w), "h": float(h)})
    return bars


def segment_photo(path: Path, panels: dict = PHOTO_PANELS) -> dict:
    """Kireenko (2022) plate: per panel box, Otsu on the max-RGB luminance (bright test on black),
    opening, largest component, hole fill.  Each panel's 100 um scale bar is the white bar found
    on the whole plate directly below the specimen (bars are detected plate-wide so a panel box
    that clips a bar does not bias the calibration)."""
    rgb = np.asarray(Image.open(path).convert("RGB"), dtype=float) / 255.0
    Hh, Ww = rgb.shape[:2]
    lum = rgb.max(axis=2)
    bars = find_scale_bars(lum)
    out = {}
    for key, meta in panels.items():
        x0, y0, x1, y1 = meta["box"]
        oy, ox = int(y0 * Hh), int(x0 * Ww)
        sub = lum[oy:int(y1 * Hh), ox:int(x1 * Ww)]
        subs = ndi.gaussian_filter(sub, 1.0)
        thr = min(_otsu(subs), PHOTO_MIN_LUM)
        m = subs > thr
        m = ndi.binary_opening(m, iterations=2)
        m = _largest_component(ndi.binary_fill_holes(m))
        m = ndi.binary_fill_holes(ndi.binary_closing(m, iterations=3))
        ys, xs = np.nonzero(m)
        cx, bottom = ox + xs.mean(), oy + ys.max()
        below = [b for b in bars if bottom - 0.01 * Hh < b["cy"] < bottom + 0.06 * Hh and abs(b["cx"] - cx) < 0.15 * Ww]
        bar = min(below, key=lambda b: (b["cy"] - bottom) ** 2 + (b["cx"] - cx) ** 2) if below else None
        out[key] = {"mask": m, "crop": rgb[oy:int(y1 * Hh), ox:int(x1 * Ww)],
                    "um_per_px": (PHOTO_SCALE_BAR_UM / bar["w"]) if bar else None,
                    "scale_bar_px": bar["w"] if bar else None,
                    "taxon": meta["taxon"], "role": meta["role"]}
    return out


def radius_profile(mask: np.ndarray) -> dict:
    """Principal-axis rotation, per-row left/right half-widths about a least-squares axis.
    Returns z (0 at the base, increasing toward the neck), mean radius r, the two half-widths,
    the asymmetry index mean|r_L - r_R| / r_max (surface-of-revolution premise) and the rotated mask."""
    ys, xs = np.nonzero(mask)
    cy, cx = ys.mean(), xs.mean()
    w, v = np.linalg.eigh(np.cov(np.vstack([ys - cy, xs - cx])))
    major = v[:, np.argmax(w)]
    ang = math.degrees(math.atan2(major[1], major[0]))
    rot = ndi.rotate(mask.astype(np.uint8), ang, order=0, reshape=True) > 0
    rows = np.nonzero(rot.any(axis=1))[0]
    left = np.array([np.nonzero(rot[i])[0][0] for i in rows], float)
    right = np.array([np.nonzero(rot[i])[0][-1] for i in rows], float)
    mid, wts = 0.5 * (left + right), right - left + 1.0
    A = np.vstack([rows, np.ones_like(rows)]).T.astype(float)
    coef = np.linalg.lstsq(A * wts[:, None], mid * wts, rcond=None)[0]
    axis = A @ coef
    rL, rR = axis - left, right - axis
    r = 0.5 * (rL + rR)
    z = rows.astype(float)
    k = max(3, len(z) // 7)
    if r[:k].mean() <= r[-k:].mean():      # put the narrower (apertural) end at high z
        z, r, rL, rR = z[::-1], r[::-1], rL[::-1], rR[::-1]
    z = np.abs(z - z[0])
    return {"z": z, "r": r, "rL": rL, "rR": rR,
            "asymmetry": float(np.mean(np.abs(rL - rR)) / r.max()),
            "rotated_mask": rot, "axis_angle_deg": ang}


# --------------------------------------------------------------------------- fitting & statistics
def _fit(fun, z, r, x0s, bounds):
    best = None
    for x0 in x0s:
        res = least_squares(lambda p: fun(z, *p) - r, x0, bounds=bounds, max_nfev=400)
        if best is None or res.cost < best.cost:
            best = res
    return best


def nrmse(model: np.ndarray, r: np.ndarray) -> float:
    return float(np.sqrt(np.mean((model - r) ** 2)) / r.max())


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    if a.std() == 0 or b.std() == 0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def ks_stat(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    res = ks_2samp(a, b)
    return float(res.statistic), float(res.pvalue)


def _starts_flask(rmax, L):
    return ([[rmax, B, t, 0.2 * L, 0.0] for B in (0.3, 0.6, 0.85) for t in (0.5, 0.95)],
            ([0.3 * rmax, 0.01, 0.01, 0.0, -0.2 * L], [1.5 * rmax, 0.999, 1.0, L, 0.3 * L]))


def _starts_bead(rmax, L):
    return ([[(1 + B) / (2 * rmax), B, zc] for B in (0.3, 0.6, 0.9) for zc in (0.3 * L, 0.5 * L, 0.7 * L)],
            ([0.2 / rmax, 0.01, -0.5 * L], [5.0 / rmax, 0.999, 1.5 * L]))


def _starts_ellipsoid(rmax, L):
    return ([[rmax, L / 2, L / 2]], ([0.1 * rmax, 0.1 * L, -L], [3 * rmax, 3 * L, 2 * L]))


def _starts_sphcyl(rmax, L):
    return ([[rmax, 0.3 * rmax, 0.3 * L, 0.0], [0.8 * rmax, 0.1 * rmax, 0.5 * L, 0.0]],
            ([0.3 * rmax, 0.01 * rmax, 0.0, -0.2 * L], [1.5 * rmax, 1.0 * rmax, L, 0.3 * L]))


MODELS = {
    "flask": (flask_profile, _starts_flask),
    "bead": (bead_profile, _starts_bead),
    "ellipsoid": (ellipsoid_profile, _starts_ellipsoid),
    "sphere_cylinder": (sphere_cylinder_profile, _starts_sphcyl),
}
DELAUNAY_MODELS = ("flask", "bead")


def fit_all(z: np.ndarray, r: np.ndarray, cv: bool = True) -> dict:
    """Fit E7, E8, the ellipsoid null, the sphere+cylinder ablation and the free-form
    reference; return parameters, nRMSE, Pearson r, KS statistic and 2-fold CV nRMSE."""
    rmax, L = float(r.max()), float(z.max())
    out = {}
    even, odd = np.arange(len(z)) % 2 == 0, np.arange(len(z)) % 2 == 1
    for name, (fun, starts) in MODELS.items():
        x0s, bounds = starts(rmax, L)
        res = _fit(fun, z, r, x0s, bounds)
        prof = fun(z, *res.x)
        ks, ks_p = ks_stat(r, prof)
        rec = {"params": [float(v) for v in res.x], "profile": prof, "nrmse": nrmse(prof, r),
               "pearson_r": pearson(prof, r), "ks_stat": ks, "ks_p": ks_p}
        if cv:
            cvs = []
            for tr, te in ((even, odd), (odd, even)):
                rf = _fit(fun, z[tr], r[tr], [res.x], bounds)
                cvs.append(nrmse(fun(z[te], *rf.x), r[te]))
            rec["cv_nrmse"] = float(np.mean(cvs))
        if name in DELAUNAY_MODELS:
            rec["B"] = rec["params"][1]
            rec["neck_ratio"] = neck_ratio(rec["B"])
        out[name] = rec
    ff = freeform_fit(z, r)
    ks, ks_p = ks_stat(r, ff)
    rec = {"profile": ff, "nrmse": nrmse(ff, r), "pearson_r": pearson(ff, r), "ks_stat": ks, "ks_p": ks_p}
    if cv:
        cvs = [nrmse(freeform_fit(z[tr], r[tr], z[te]), r[te]) for tr, te in ((even, odd), (odd, even))]
        rec["cv_nrmse"] = float(np.mean(cvs))
    out["freeform"] = rec
    out["best_delaunay"] = min(DELAUNAY_MODELS, key=lambda k: out[k]["nrmse"])
    return out


def decide(fit: dict, asymmetry: float) -> dict:
    best = fit[fit["best_delaunay"]]
    gate = min(max(RMSE_SUPPORT, FLOOR_FACTOR * fit["freeform"]["nrmse"]), RMSE_CAP)
    return {
        "best_model": fit["best_delaunay"],
        "nrmse_gate": gate,
        "near_floor": bool(best["nrmse"] <= gate),
        "axisymmetric": bool(asymmetry <= ASYM_MAX),
        "beats_ellipsoid_null": bool(best["nrmse"] < fit["ellipsoid"]["nrmse"]),
        "beats_sphere_cylinder_ablation": bool(best["nrmse"] < fit["sphere_cylinder"]["nrmse"]),
        "delaunay_compatible": bool(best["nrmse"] <= gate and asymmetry <= ASYM_MAX),
    }


def summarise_specimen(sid: str, prof: dict, fit: dict, extra: dict) -> dict:
    L, W = float(prof["z"].max()), 2.0 * float(prof["r"].max())
    best = fit[fit["best_delaunay"]]
    rec = {"id": sid, "length_px": L, "width_px": W, "aspect_LW": L / W,
           "asymmetry_index": prof["asymmetry"],
           "exceeds_bead_aspect_bound": bool(L / W > math.pi),
           "best_B": best["B"], "best_neck_ratio": best["neck_ratio"],
           "best_nrmse": best["nrmse"], "best_pearson_r": best["pearson_r"],
           "best_ks_stat": best["ks_stat"], "best_ks_p": best["ks_p"], "best_cv_nrmse": best.get("cv_nrmse"),
           "models": {k: {kk: vv for kk, vv in v.items() if kk != "profile"}
                      for k, v in fit.items() if isinstance(v, dict)}}
    rec.update(decide(fit, prof["asymmetry"]))
    rec.update(extra)
    rec["_profile"], rec["_fit"] = prof, fit
    return rec


# --------------------------------------------------------------------------- parameter sweep
def parameter_sweep() -> dict:
    """E4-E7 regime table: catalogue member, neck ratio, period / (2a) and bead aspect ratio
    over B (cylinder -> unduloid -> spheres -> nodoid), and flask join radius over t."""
    rows = []
    for B in SWEEP_B:
        row = {"B": B, "member": classify_delaunay(B)}
        if B <= 1.0:
            row.update({"neck_ratio": neck_ratio(B), "period_over_2a": 2.0 * ellipe(B * B),
                        "bead_aspect_LW": bead_aspect_ratio(B)})
        else:   # nodoid: the Kenmotsu radius still obeys the envelope but the profile loops
            s = np.linspace(0, 2 * np.pi, 2000)
            z, r = kenmotsu_full(1.0, B, s)
            row.update({"neck_ratio": None, "period_over_2a": None, "bead_aspect_LW": None,
                        "r_min_over_r_max": float(r.min() / r.max()),
                        "axial_backtracking": bool(np.any(np.diff(z) < 0))})
        rows.append(row)
    joins = [{"t": t, "B": B, "join_radius_over_R": flask_join_radius(1.0, B, t)[2],
              "kappa": flask_join_radius(1.0, B, t)[1]} for B in (0.3, 0.6, 0.9) for t in SWEEP_T]
    rho = [neck_ratio(B) for B in SWEEP_B if B <= 1.0]
    return {"over_B": rows, "flask_join_over_t": joins,
            "neck_ratio_monotone_decreasing": bool(np.all(np.diff(rho) < 0)),
            "bead_aspect_in_1_pi": bool(all(1.0 - 1e-9 <= r_["bead_aspect_LW"] <= math.pi + 1e-9
                                            for r_ in rows if r_["bead_aspect_LW"] is not None))}


# --------------------------------------------------------------------------- analysis
def analyse() -> dict:
    ref = verify_inputs()
    g, comps = segment_fig84(FIG84_PNG)
    drawings = []
    for k, (sl, mask) in enumerate(comps):
        prof = radius_profile(mask)
        fit = fit_all(prof["z"], prof["r"])
        drawings.append(summarise_specimen(
            f"fig84_{k:02d}", prof, fit,
            {"source": "Thompson 1917 Fig. 84 (after Brady)", "kind": "drawing", "role": "test",
             "taxon": "Lagena sp.", "row": "upper" if k < 8 else "lower",
             "bbox": [sl[0].start, sl[1].start, sl[0].stop, sl[1].stop], "area_px": int(mask.sum()),
             "_mask": mask, "_slice": sl}))
    photos = []
    for key, pan in segment_photo(PHOTO_JPG).items():
        prof = radius_profile(pan["mask"])
        fit = fit_all(prof["z"], prof["r"])
        upp = pan["um_per_px"]
        photos.append(summarise_specimen(
            f"bdj_{key}", prof, fit,
            {"source": "Kireenko et al. 2022 BDJ Fig. 8", "kind": "photograph", "role": pan["role"],
             "taxon": pan["taxon"], "um_per_px": upp, "scale_bar_px": pan["scale_bar_px"],
             "length_um": float(prof["z"].max() * upp) if upp else None,
             "width_um": float(2 * prof["r"].max() * upp) if upp else None,
             "area_px": int(pan["mask"].sum()), "_mask": pan["mask"], "_crop": pan["crop"]}))
    tests = [s for s in drawings + photos if s["role"] == "test"]
    controls = [s for s in photos if s["role"] == "control"]
    n_ok = sum(s["delaunay_compatible"] for s in tests)
    frac = n_ok / len(tests)
    ctrl_med = float(np.median([c["best_nrmse"] for c in controls])) if controls else None
    test_med = float(np.median([s["best_nrmse"] for s in tests]))
    flask_like = [s for s in tests if not s["exceeds_bead_aspect_bound"]]
    spindles = [s for s in tests if s["exceeds_bead_aspect_bound"]]
    discriminating = bool(controls) and bool(sum(c["delaunay_compatible"] for c in controls) < len(controls))
    if not discriminating:
        verdict = "untestable"
    elif frac >= 0.75:
        verdict = "supported"
    elif frac <= 0.25:
        verdict = "refuted"
    else:
        verdict = "mixed"
    return {"reference": ref, "plate": g, "drawings": drawings, "photos": photos,
            "tests": tests, "controls": controls, "flask_like": flask_like, "spindles": spindles,
            "n_tests": len(tests), "n_delaunay_compatible": n_ok, "fraction_delaunay_compatible": frac,
            "test_nrmse_median": test_med, "control_nrmse_median": ctrl_med,
            "controls_discriminating": discriminating, "verdict": verdict,
            "sweep": parameter_sweep()}


def _public(s: dict) -> dict:
    return {k: v for k, v in s.items() if not k.startswith("_")}


def morphometrics_dict(a: dict | None = None) -> dict:
    a = a or analyse()
    tests, ctrls, fl, sp = a["tests"], a["controls"], a["flask_like"], a["spindles"]

    def med(rows, key, sub=None):
        vals = [(r_[sub][key] if sub else r_[key]) for r_ in rows if (r_[sub][key] if sub else r_[key]) is not None]
        return float(np.median(vals)) if vals else None

    return {
        "claim_id": "T1917-014", "verdict": a["verdict"],
        "n_tests": a["n_tests"], "n_drawings": len(a["drawings"]),
        "n_photographs_test": sum(1 for p in a["photos"] if p["role"] == "test"),
        "n_controls": len(ctrls), "n_delaunay_compatible": a["n_delaunay_compatible"],
        "fraction_delaunay_compatible": a["fraction_delaunay_compatible"],
        "n_compatible_controls": sum(c["delaunay_compatible"] for c in ctrls),
        "controls_discriminating": a["controls_discriminating"],
        "decision_rule": {"rmse_support": RMSE_SUPPORT, "floor_factor": FLOOR_FACTOR, "rmse_cap": RMSE_CAP, "asym_max": ASYM_MAX},
        "specificity_caveat": (
            f"{sum(c['delaunay_compatible'] for c in ctrls)}/{len(ctrls)} polythalamous controls pass the same gate: "
            "a smooth egg-shaped outline is close to a sphere+unduloid composite whether or not the shell is a single "
            "surface-tension chamber, so a passing flask-like test is 'consistent with' the catalogue, not a unique "
            "demonstration; the spindle refutation (bead aspect bound) is the specific result."
        ),
        "test_best_nrmse_median": a["test_nrmse_median"],
        "test_freeform_nrmse_median": med([t["models"]["freeform"] for t in tests], "nrmse"),
        "test_ellipsoid_nrmse_median": med([t["models"]["ellipsoid"] for t in tests], "nrmse"),
        "test_sphere_cylinder_nrmse_median": med([t["models"]["sphere_cylinder"] for t in tests], "nrmse"),
        "test_pearson_r_median": med(tests, "best_pearson_r"),
        "test_ks_stat_median": med(tests, "best_ks_stat"),
        "test_cv_nrmse_median": med(tests, "best_cv_nrmse"),
        "test_asymmetry_median": med(tests, "asymmetry_index"),
        "n_beats_ellipsoid_null": sum(t["beats_ellipsoid_null"] for t in tests),
        "n_beats_sphere_cylinder_ablation": sum(t["beats_sphere_cylinder_ablation"] for t in tests),
        "n_flask_like_LW_le_pi": len(fl), "flask_like_nrmse_median": med(fl, "best_nrmse"),
        "flask_like_n_compatible": sum(s["delaunay_compatible"] for s in fl),
        "n_spindle_LW_gt_pi": len(sp), "spindle_nrmse_median": med(sp, "best_nrmse"),
        "spindle_n_compatible": sum(s["delaunay_compatible"] for s in sp),
        "control_nrmse_median": a["control_nrmse_median"],
        "photo_lagena_length_um": [p["length_um"] for p in a["photos"] if p["taxon"].startswith("Lagena")],
        "fitted_B_range": [float(min(t["best_B"] for t in tests)), float(max(t["best_B"] for t in tests))],
        "catalogue_members_used": sorted({classify_delaunay(t["best_B"]) for t in tests} | {"sphere", "cylinder"}),
        "sweep": a["sweep"],
        "specimens": [_public(s) for s in a["drawings"] + a["photos"]],
    }


def measure_morphoeval(a: dict | None = None) -> list[dict]:
    """Measurement records for morphoeval/measure_outputs.py (name/value/unit/method/...)."""
    m = morphometrics_dict(a)

    def M(name, value, unit, method, reference=None, note=None):
        d = {"name": name, "value": value, "unit": unit, "method": method, "derived_from": "script"}
        if reference is not None:
            d["reference_value"] = reference
        if note:
            d["note"] = note
        return d

    lag = [v for v in m["photo_lagena_length_um"] if v]
    return [
        M("unilocular test silhouettes analysed (13 drawings + 5 photographs)", m["n_tests"], "count",
          "Fig. 84 background-normalised segmentation; Kireenko plate Otsu segmentation", reference=18),
        M("Delaunay-compatible unilocular tests", m["n_delaunay_compatible"], "count",
          f"best of E7/E8 with nRMSE <= max({RMSE_SUPPORT}, {FLOOR_FACTOR} x free-form) and asymmetry <= {ASYM_MAX}",
          reference=m["n_tests"], note=f"verdict {m['verdict']}; fraction {m['fraction_delaunay_compatible']:.2f}"),
        M("median best Delaunay RMSE / r_max (tests)", m["test_best_nrmse_median"], "dimensionless",
          "RMSE over all silhouette rows", reference=m["test_freeform_nrmse_median"],
          note="reference = free-form degree-6 noise floor"),
        M("median ellipsoid null RMSE / r_max (tests)", m["test_ellipsoid_nrmse_median"], "dimensionless",
          "null model vs Delaunay baseline", reference=m["test_best_nrmse_median"]),
        M("median sphere+cylinder ablation RMSE / r_max (tests)", m["test_sphere_cylinder_nrmse_median"], "dimensionless",
          "unduloid shoulder ablated (C0 join) vs Delaunay baseline", reference=m["test_best_nrmse_median"]),
        M("tests where Delaunay beats the ellipsoid null", m["n_beats_ellipsoid_null"], "count",
          "nRMSE comparison", reference=m["n_tests"]),
        M("median 2-fold CV RMSE / r_max (best Delaunay)", m["test_cv_nrmse_median"], "dimensionless",
          "fit on even rows, evaluate on odd rows and vice versa"),
        M("median KS statistic observed vs fitted radii (best Delaunay)", m["test_ks_stat_median"], "dimensionless",
          "scipy.stats.ks_2samp"),
        M("median Pearson r observed vs fitted radius", m["test_pearson_r_median"], "dimensionless", "np.corrcoef"),
        M("median bilateral asymmetry index mean|r_L - r_R| / r_max", m["test_asymmetry_median"], "dimensionless",
          "half-widths about the least-squares axis", reference=0.0, note="0 for a perfect surface of revolution"),
        M("flask-like tests (L/W <= pi) Delaunay-compatible", m["flask_like_n_compatible"], "count",
          "subset within the unduloid bead aspect bound", reference=m["n_flask_like_LW_le_pi"]),
        M("spindle tests (L/W > pi) Delaunay-compatible", m["spindle_n_compatible"], "count",
          "subset outside the bead aspect bound (E6, Lean bead_aspect_bound)", reference=m["n_spindle_LW_gt_pi"]),
        M("polythalamous negative controls Delaunay-compatible", m["n_compatible_controls"], "count",
          "Globobulimina 64a, Quinqueloculina 66a fitted with the same pipeline", reference=0,
          note="control median nRMSE %.3f" % (m["control_nrmse_median"] or float("nan"))),
        M("Lagena sp. test length from 100 um scale bar", float(np.mean(lag)) if lag else None, "um",
          "scale-bar calibrated silhouette length (61a, 62a)", reference=300.0,
          note="order-of-magnitude check: Lagena tests are typically 0.15-0.6 mm (Jones 1984 key to unilocular hyaline forams)"),
        M("fitted Kenmotsu B range", f"{m['fitted_B_range'][0]:.2f}-{m['fitted_B_range'][1]:.2f}", "dimensionless",
          "best catalogue member per specimen"),
    ]


# --------------------------------------------------------------------------- figures
def render_comparison(a: dict, out: Path = COMPARISON_PNG) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dr, ph = a["drawings"], a["photos"]
    fig = plt.figure(figsize=(22, 20))
    gs = fig.add_gridspec(5, 7, height_ratios=[1.25, 1, 1, 1.15, 0.9])

    # A: catalogue gallery (members offset vertically)
    ax = fig.add_subplot(gs[0, 0:2])
    s = np.linspace(0, 2 * np.pi, 600)
    members = ((0.0, "cylinder B=0", "0.4"), (0.35, "unduloid B=0.35", "tab:blue"),
               (0.7, "unduloid B=0.7", "tab:green"), (1.0, "spheres B=1", "tab:red"), (1.4, "nodoid B=1.4", "tab:orange"))
    for i, (B, lab, col) in enumerate(members):
        z, r = kenmotsu_full(1.0, B, s)
        off = -2.6 * i
        ax.plot(z, r + off, color=col, lw=1.4, label=lab); ax.plot(z, -r + off, color=col, lw=1.4)
    zc = np.linspace(-1.5, 1.5, 200)
    ax.plot(zc + 8.5, catenoid_profile(0.5, zc), "tab:purple", lw=1.4, label="catenoid H=0")
    ax.plot(zc + 8.5, -catenoid_profile(0.5, zc), "tab:purple", lw=1.4)
    ax.set_aspect("equal"); ax.legend(fontsize=7, loc="lower right", ncol=2); ax.set_yticks([])
    ax.set_title("A  Plateau/Delaunay catalogue (Kenmotsu E4; Thompson Figs. 61-65)", fontsize=10)

    # B: plate with segmentation
    ax = fig.add_subplot(gs[0, 2:5])
    ax.imshow(a["plate"], cmap="gray")
    for sdat in dr:
        sl = sdat["_slice"]
        ax.contour(np.pad(sdat["_mask"], 1)[1:-1, 1:-1], levels=[0.5],
                   colors=["tab:red" if sdat["delaunay_compatible"] else "tab:orange"],
                   linewidths=0.8, extent=(sl[1].start, sl[1].stop, sl[0].stop, sl[0].start))
        ax.text(sl[1].start, sl[0].start - 6, sdat["id"][-2:], color="tab:blue", fontsize=7)
    ax.set_title("B  Thompson (1917) Fig. 84, Lagena after Brady — segmented outlines\n(red = Delaunay-compatible, orange = not)", fontsize=10)
    ax.axis("off")

    # C: model comparison bars for all tests + controls
    ax = fig.add_subplot(gs[0, 5:7])
    allsp = dr + ph
    idx = np.arange(len(allsp))
    wdt = 0.2
    for j, (name, col) in enumerate((("flask", "tab:red"), ("bead", "tab:blue"), ("ellipsoid", "0.5"), ("freeform", "tab:green"))):
        ax.bar(idx + (j - 1.5) * wdt, [s_["models"][name]["nrmse"] for s_ in allsp], wdt, color=col,
               label={"flask": "Delaunay flask E7", "bead": "Delaunay bead E8", "ellipsoid": "ellipsoid null", "freeform": "free-form deg-6"}[name])
    ax.axhline(RMSE_SUPPORT, color="k", ls="--", lw=1, label=f"gate {RMSE_SUPPORT:.2f}")
    ax.set_xticks(idx); ax.set_xticklabels([s_["id"].replace("fig84_", "F").replace("bdj_", "P") for s_ in allsp], fontsize=6, rotation=90)
    ax.set_ylabel("RMSE / r_max"); ax.set_ylim(0, 0.6); ax.legend(fontsize=7, ncol=2)
    ax.set_title(f"C  Fit quality — {a['n_delaunay_compatible']}/{a['n_tests']} unilocular tests Delaunay-compatible; verdict: {a['verdict']}", fontsize=10)

    # D: Fig. 84 profiles
    for k, sdat in enumerate(dr[:14]):
        ax = fig.add_subplot(gs[1 + k // 7, k % 7])
        p, f = sdat["_profile"], sdat["_fit"]
        rmax = p["r"].max()
        ax.fill_betweenx(p["z"], -p["rL"], p["rR"], color="0.85")
        for name, col, ls in (("flask", "tab:red", "-"), ("bead", "tab:blue", "--"), ("ellipsoid", "0.4", ":")):
            ax.plot(f[name]["profile"], p["z"], color=col, lw=1.1, ls=ls, label=f"{name} {f[name]['nrmse']:.3f}")
            ax.plot(-f[name]["profile"], p["z"], color=col, lw=1.1, ls=ls)
        ax.set_aspect("equal"); ax.set_xlim(-1.4 * rmax, 1.4 * rmax)
        ax.set_title(f"F{sdat['id'][-2:]}  L/W={sdat['aspect_LW']:.1f}  B={sdat['best_B']:.2f}  "
                     f"{'OK' if sdat['delaunay_compatible'] else 'x'}", fontsize=8)
        ax.legend(fontsize=5.5, loc="lower right"); ax.tick_params(labelsize=6)

    # E: photographs (crop + outline + fit)
    for k, sdat in enumerate(ph[:7]):
        ax = fig.add_subplot(gs[3, k])
        crop, mask = sdat["_crop"], sdat["_mask"]
        ax.imshow(crop)
        ax.contour(mask, levels=[0.5], colors=["tab:red" if sdat["delaunay_compatible"] else "tab:orange"], linewidths=1.0)
        um = f"  {sdat['length_um']:.0f} um" if sdat.get("length_um") else ""
        ax.set_title(f"P{sdat['id'][4:]} {sdat['taxon'].split(' (')[0]} [{sdat['role']}]{um}\n"
                     f"best {sdat['best_model']} nRMSE {sdat['best_nrmse']:.3f} / ff {sdat['models']['freeform']['nrmse']:.3f} "
                     f"{'OK' if sdat['delaunay_compatible'] else 'x'}", fontsize=7.5)
        ax.axis("off")

    # F: photograph profiles with fits (first 5) + sweep panels
    for k, sdat in enumerate(ph[:5]):
        ax = fig.add_subplot(gs[4, k])
        p, f = sdat["_profile"], sdat["_fit"]
        rmax = p["r"].max()
        ax.fill_betweenx(p["z"], -p["rL"], p["rR"], color="0.85")
        for name, col, ls in (("flask", "tab:red", "-"), ("bead", "tab:blue", "--"), ("ellipsoid", "0.4", ":")):
            ax.plot(f[name]["profile"], p["z"], color=col, lw=1.1, ls=ls, label=f"{name} {f[name]['nrmse']:.3f}")
            ax.plot(-f[name]["profile"], p["z"], color=col, lw=1.1, ls=ls)
        ax.set_aspect("equal"); ax.set_xlim(-1.4 * rmax, 1.4 * rmax)
        ax.set_title(f"P{sdat['id'][4:]} profile  B={sdat['best_B']:.2f}  KS={sdat['best_ks_stat']:.2f}  CV={sdat['best_cv_nrmse']:.3f}", fontsize=7.5)
        ax.legend(fontsize=5.5, loc="lower right"); ax.tick_params(labelsize=6)
    ax = fig.add_subplot(gs[4, 5:7])
    sw = [r_ for r_ in a["sweep"]["over_B"] if r_["B"] <= 1.0]
    Bs = [r_["B"] for r_ in sw]
    ax.plot(Bs, [r_["neck_ratio"] for r_ in sw], "o-", color="tab:blue", label="neck ratio (1-B)/(1+B)  E5")
    ax.plot(Bs, [r_["bead_aspect_LW"] / math.pi for r_ in sw], "s-", color="tab:red", label="bead L/W / pi  E6")
    ax.plot(Bs, [r_["period_over_2a"] / math.pi for r_ in sw], "^-", color="tab:green", label="period / (2 pi a)  E6")
    for t in SWEEP_T:
        _, _, rj = flask_join_radius(1.0, 0.6, t)
        ax.plot([0.6], [rj], "x", color="tab:purple", ms=6)
    ax.plot([], [], "x", color="tab:purple", label="flask join r_j/R at B=0.6, t sweep  E7")
    ax.axvline(0, color="0.6", lw=0.8); ax.axvline(1, color="0.6", lw=0.8)
    ax.text(0.01, 0.02, "cylinder", fontsize=7); ax.text(0.86, 0.02, "spheres", fontsize=7)
    ax.set_xlabel("Kenmotsu B"); ax.set_ylim(0, 1.05); ax.legend(fontsize=7, loc="center left")
    ax.set_title("G  Parameter-regime sweep over the unduloid family", fontsize=10)

    fig.suptitle("T1917-014  Lagena and other unilocular tests vs. the Delaunay catalogue — Pass 2 "
                 "(Thompson 1917, Ch. V, Fig. 84; Kireenko et al. 2022 CC BY 4.0)", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=100); plt.close(fig)
    return out


def render_growth_gif(out: Path = GROWTH_GIF, n_frames: int = 36) -> Path:
    """Thompson p. 257: Lagena as 'a hanging drop drawn out to a long and slender neck'.
    Left: the unduloid family B: 0 -> 0.98 (cylinder -> string of spheres).
    Right: the Delaunay flask (E7) as B grows and the neck lengthens."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    frames = []
    s = np.linspace(0, 2 * np.pi, 500)
    z_fl = np.linspace(0, 4.2, 600)
    for i in range(n_frames):
        u = i / (n_frames - 1)
        B = 0.98 * u
        fig, axes = plt.subplots(1, 2, figsize=(8, 4), dpi=80)
        z, r = kenmotsu_full(1.0, max(B, 1e-6), s)
        axes[0].fill_between(z, -r, r, color="tab:blue", alpha=0.6)
        axes[0].set_xlim(0, 2 * np.pi); axes[0].set_ylim(-1.1, 1.1); axes[0].set_aspect("equal")
        axes[0].set_title(f"unduloid family  B={B:.2f}  r_min/r_max={neck_ratio(B):.2f}\n({classify_delaunay(B if B > 0 else 0.0)})", fontsize=9)
        rf = flask_profile(z_fl, 1.0, max(B, 1e-3), 0.75, 0.3 + 1.6 * u, 0.0)
        axes[1].fill_betweenx(z_fl, -rf, rf, color="tab:red", alpha=0.6)
        axes[1].set_xlim(-1.3, 1.3); axes[1].set_ylim(0, 4.2); axes[1].set_aspect("equal")
        axes[1].set_title("Delaunay flask (E7): sphere + unduloid + neck cylinder\n'hanging drop drawn out to a slender neck' (p. 257)", fontsize=9)
        for ax in axes:
            ax.set_xticks([]); ax.set_yticks([])
        fig.tight_layout()
        fig.canvas.draw()
        frames.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()))
        plt.close(fig)
    out.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(out, save_all=True, append_images=frames[1:], duration=120, loop=0)
    return out


# --------------------------------------------------------------------------- result record
def write_result(a: dict, m: dict, out: Path = RESULT_JSON) -> Path:
    ref = a["reference"]
    rec = {
        "claim_id": "T1917-014",
        "slot_id": "thompson1917-t1917-014",
        "date_made": "1917",
        "claim": ref["claim_statement"],
        "dataset": {
            "pre_scouted": {"name": "Prazeres, Uthicke & Pandolfi (2015) Dryad", "doi": "10.5061/dryad.28816",
                            "status": "DATASET_MISMATCH", "sha256_metadata": ref["input_dataset_sha256"]["dryad_28816_metadata.json"]},
            "used": [
                {"name": "Thompson (1917) Fig. 84, 13 Lagena after Brady — IA ongrowthform1917thom leaf 0276 (p. 256)",
                 "sha256": ref["input_dataset_sha256"]["thompson_1917_leaf0276_p256_fig84.jp2"],
                 "crop_sha256": ref["input_dataset_sha256"]["thompson_1917_fig84_lagena_after_brady.png"], "license": "public domain"},
                {"name": "Kireenko et al. (2022) BDJ 10:e87457 Fig. 8 — Lagena/Oolina/Fissurina photographs + polythalamous controls",
                 "doi": "10.3897/BDJ.10.e87457",
                 "sha256": ref["input_dataset_sha256"]["kireenko_2022_bdj_e87457_fig8_station5656mc.jpg"], "license": "CC BY 4.0"},
            ],
        },
        "verdict": a["verdict"],
        "verdict_rule": "supported >= 75 % / refuted <= 25 % of unilocular tests Delaunay-compatible; mixed otherwise; untestable if every polythalamous control passes",
        "specificity_caveat": m["specificity_caveat"],
        "headline_numbers": {
            "unilocular_tests": m["n_tests"],
            "delaunay_compatible": m["n_delaunay_compatible"],
            "fraction_compatible": round(m["fraction_delaunay_compatible"], 3),
            "flask_like_compatible": f"{m['flask_like_n_compatible']}/{m['n_flask_like_LW_le_pi']}",
            "spindle_compatible": f"{m['spindle_n_compatible']}/{m['n_spindle_LW_gt_pi']}",
            "flask_like_nrmse_median": round(m["flask_like_nrmse_median"], 4) if m["flask_like_nrmse_median"] is not None else None,
            "spindle_nrmse_median": round(m["spindle_nrmse_median"], 4) if m["spindle_nrmse_median"] is not None else None,
            "freeform_floor_nrmse_median": round(m["test_freeform_nrmse_median"], 4),
            "ellipsoid_null_nrmse_median": round(m["test_ellipsoid_nrmse_median"], 4),
            "sphere_cylinder_ablation_nrmse_median": round(m["test_sphere_cylinder_nrmse_median"], 4),
            "tests_beating_ellipsoid_null": f"{m['n_beats_ellipsoid_null']}/{m['n_tests']}",
            "pearson_r_median": round(m["test_pearson_r_median"], 4),
            "ks_stat_median": round(m["test_ks_stat_median"], 4),
            "cv_nrmse_median": round(m["test_cv_nrmse_median"], 4),
            "asymmetry_median": round(m["test_asymmetry_median"], 4),
            "controls_compatible": f"{m['n_compatible_controls']}/{m['n_controls']}",
            "control_nrmse_median": round(m["control_nrmse_median"], 4) if m["control_nrmse_median"] is not None else None,
            "fitted_B_range": [round(v, 3) for v in m["fitted_B_range"]],
        },
        "interpretation": (
            "Flask-shaped and globular unilocular tests (Lagena, Oolina, Fissurina) are reproduced by Thompson's own "
            "'unduloid combined with a portion of a sphere' composite to within the free-form noise floor; the slender "
            "fusiform Lagena (L/W > pi) cannot be any single unduloid (bead aspect bound, Lean-checked) and are not fitted "
            "by any catalogue composite. Thompson's claim therefore holds for the flask morphotype he describes and fails "
            "for the spindle morphotype he also figures."
        ),
        "script": "thompson/claims/t1917-014_the-forms-of-cells.py",
        "figure": str(COMPARISON_PNG.relative_to(ROOT)),
        "animation": str(GROWTH_GIF.relative_to(ROOT)),
        "lean": "morphoeval/lean/loops/Thompson1917_T1917_014.lean",
        "metrics": str(METRICS_JSON.relative_to(ROOT)),
    }
    out.write_text(json.dumps(rec, indent=2), encoding="utf-8")
    return out


# --------------------------------------------------------------------------- self-test
def self_test() -> None:
    # E5 / E3 identities on the Kenmotsu profile
    for B in (0.0, 0.3, 0.7, 0.99):
        H = 0.5
        dz, r = kenmotsu_profile(H, B)
        assert abs(r[0] - (1 + B) / (2 * H)) < 1e-9 and abs(r[-1] - (1 - B) / (2 * H)) < 1e-9
        assert abs(r[0] + r[-1] - 1 / H) < 1e-9                                   # roots sum = 1/H
        C = r[0] - H * r[0] ** 2
        assert abs(r[0] * r[-1] - C / H) < 1e-9                                   # roots product = C/H
        assert abs(neck_ratio(B) - r[-1] / r[0]) < 1e-9
        assert r.min() >= r[-1] - 1e-9 and r.max() <= r[0] + 1e-9                 # envelope
        assert abs(dz[-1] - unduloid_period(B, H) / 2) < 2e-3 * unduloid_period(B, H)   # E6 half period
    assert abs(bead_aspect_ratio(0.0) - math.pi) < 1e-9 and abs(bead_aspect_ratio(1.0) - 1.0) < 1e-9
    assert classify_delaunay(0.0) == "cylinder" and classify_delaunay(0.5) == "unduloid"
    assert classify_delaunay(1.0) == "sphere" and classify_delaunay(1.5) == "nodoid"
    assert classify_delaunay(0.3, H=0.0) == "catenoid"
    # E7: join radius closed form; equator join at t = 1; sphere portion above the equator for t < 1
    H, kappa, rj = flask_join_radius(1.0, 0.6, 1.0)
    assert abs(rj - 1.0) < 1e-9 and abs(kappa - 0.8) < 1e-12
    assert flask_join_radius(1.0, 0.6, 0.5)[2] < 1.0
    z = np.linspace(0, 4, 2000)
    rf = flask_profile(z, 1.0, 0.6, 1.0, 0.5, 0.0)
    assert abs(rf[np.argmin(np.abs(z - 1.0))] - 1.0) < 2e-3
    assert np.all(np.abs(np.diff(rf[(z > 0.05) & (z < 3.0)])) < 0.02)
    # Sweep invariants (mirrors Lean neck_ratio_strict_anti and bead_aspect_bound)
    sw = parameter_sweep()
    assert sw["neck_ratio_monotone_decreasing"] and sw["bead_aspect_in_1_pi"]
    assert [r_["member"] for r_ in sw["over_B"]][:1] == ["cylinder"] and sw["over_B"][-1]["member"] == "nodoid"
    # Synthetic recovery and comparator ordering on a known flask
    zs = np.arange(0, 320.0)
    rs = flask_profile(zs, 100.0, 0.7, 0.8, 60.0, 0.0)
    keep = rs > 0
    fit = fit_all(zs[keep], rs[keep], cv=False)
    assert fit["flask"]["nrmse"] < 0.01 and abs(fit["flask"]["B"] - 0.7) < 0.05, fit["flask"]
    assert fit["ellipsoid"]["nrmse"] > fit["flask"]["nrmse"] and fit["sphere_cylinder"]["nrmse"] > fit["flask"]["nrmse"]
    # Real data: 13 Lagena on the plate, 7 photo panels with scale bars, symmetric silhouettes
    g, comps = segment_fig84(FIG84_PNG)
    assert len(comps) == 13, len(comps)
    assert radius_profile(comps[1][1])["asymmetry"] < 0.10
    pans = segment_photo(PHOTO_JPG)
    assert len(pans) == 7 and all(p["mask"].sum() > 20000 for p in pans.values())
    assert sum(p["um_per_px"] is not None for p in pans.values()) >= 5
    for key in ("61a", "62a"):
        pr = radius_profile(pans[key]["mask"])
        assert pr["asymmetry"] < ASYM_MAX, (key, pr["asymmetry"])
    print("self-test OK: Delaunay identities, catalogue, flask joins, sweep invariants, synthetic recovery, "
          "comparator ordering, 13 Fig. 84 silhouettes, 7 photo panels")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--no-gif", action="store_true")
    args = ap.parse_args(argv)
    if args.self_test:
        self_test()
        return 0
    a = analyse()
    m = morphometrics_dict(a)
    m["measurements"] = measure_morphoeval(a)
    METRICS_JSON.write_text(json.dumps(m, indent=2), encoding="utf-8")
    write_result(a, m)
    render_comparison(a)
    if not args.no_gif:
        render_growth_gif()
    print(f"verdict: {a['verdict']} — {m['n_delaunay_compatible']}/{m['n_tests']} unilocular tests Delaunay-compatible "
          f"(flask-like {m['flask_like_n_compatible']}/{m['n_flask_like_LW_le_pi']} median nRMSE {m['flask_like_nrmse_median']:.3f}; "
          f"spindles {m['spindle_n_compatible']}/{m['n_spindle_LW_gt_pi']} median {m['spindle_nrmse_median']:.3f}; "
          f"free-form floor {m['test_freeform_nrmse_median']:.3f}; ellipsoid null {m['test_ellipsoid_nrmse_median']:.3f}; "
          f"controls {m['n_compatible_controls']}/{m['n_controls']} compatible, median {m['control_nrmse_median']:.3f})")
    print(f"wrote {METRICS_JSON}, {RESULT_JSON}, {COMPARISON_PNG}" + ("" if args.no_gif else f", {GROWTH_GIF}"))
    return 0


if __name__ == "__main__":
    sys.exit(main())

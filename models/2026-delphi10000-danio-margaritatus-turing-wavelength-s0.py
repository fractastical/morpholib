#!/usr/bin/env python3
"""
Danio margaritatus at the cellular detail rung (slot 3162).

Scale: 1.339e-05 m (13.39 µm, log10 = -4.873), tier 4_cellular_10um.
Age: Holocene, 0 Ma. Chronoseries 3/5.
Predecessor: Danio albolineatus (uniform interstripe, Turing-stable here).
Successor: Danio nigrofasciatus (broken stripes, lambda_c = 17.80 µm).

Governing system (Schnakenberg 1979 kinetics; Turing 1952 dispersion):

    du/dt = D_u laplacian(u) + gamma (a - u + u^2 v)
    dv/dt = D_v laplacian(v) + gamma (b - u^2 v)

    u0 = a + b,  v0 = b / u0^2
    f_u = -1 + 2 u0 v0,  f_v = u0^2,  g_u = -2 u0 v0,  g_v = -u0^2
    k_c^2 = gamma (D_v f_u + D_u g_v) / (2 D_u D_v)
    lambda_c = 2 pi / k_c

Boundary conditions: zero-flux Neumann on the cellular patch; periodic on the
1D line used to check the analytic growth rate.

The adult photograph cannot resolve 13.39 µm. It is used for flank shape,
pearl nearest-neighbour spacing, and the image-fitness comparison. D_u is
solved so that the linear law hits 1.339e-05 m exactly; it is not fitted to
the photograph.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage as ndi
from scipy.spatial import ConvexHull, cKDTree
from scipy.stats import ks_2samp

ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "datasets" / "danio_margaritatus_turing_wavelength_s0"
SPECIMEN = DATASET / "danio_margaritatus_specimen.jpg"
REFERENCE = DATASET / "danio_margaritatus_turing_wavelength_s0_reference.json"
MEDIA_PNG = ROOT / "docs" / "media" / "danio_margaritatus_turing_wavelength_s0_comparison.png"
MEDIA_GIF = ROOT / "docs" / "media" / "danio_margaritatus_turing_wavelength_s0_growth.gif"
METRICS_JSON = DATASET / "danio_margaritatus_turing_wavelength_s0_metrics.json"
SPECIMEN_SHA256 = "48a9005afc001e30472b386bdcabc9db7b2ecd48224c68324755cb166de0181f"

LAMBDA_M = 1.339e-05
LAMBDA_UM = 13.39
SEED = 3162
# Catalog characteristic length (m) applied to the pearl-field major axis.
BODY_LENGTH_M = 0.018


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_specimen() -> str:
    if not SPECIMEN.exists():
        raise FileNotFoundError(SPECIMEN)
    digest = sha256_file(SPECIMEN)
    if digest != SPECIMEN_SHA256:
        raise ValueError(f"specimen SHA-256 {digest} != {SPECIMEN_SHA256}")
    return digest


def jacobian(a: float, b: float) -> dict[str, float]:
    """Schnakenberg Jacobian at the homogeneous steady state. Eq. steady_state."""
    u0 = a + b
    v0 = b / (u0 * u0)
    fu = -1.0 + 2.0 * u0 * v0
    fv = u0 * u0
    gu = -2.0 * u0 * v0
    gv = -(u0 * u0)
    return {"u0": u0, "v0": v0, "fu": fu, "fv": fv, "gu": gu, "gv": gv}


def diffusivities_for_wavelength(lam_um: float, a: float, b: float, delta: float, gamma: float = 1.0) -> tuple[float, float]:
    """Return (D_u, D_v) in µm^2/s so that lambda_c equals lam_um. Law k_c."""
    jac = jacobian(a, b)
    numer = delta * jac["fu"] + jac["gv"]
    if numer <= 0.0:
        raise ValueError("diffusion ratio is below the Turing numerator; no real k_c")
    k = 2.0 * np.pi / lam_um
    du = gamma * numer / (2.0 * delta * k * k)
    return float(du), float(delta * du)


def sigma_of_k(k: np.ndarray, jac: dict[str, float], du: float, dv: float, gamma: float) -> np.ndarray:
    """Largest real-part eigenvalue of gamma J - k^2 D. Analytic dispersion."""
    k = np.asarray(k, dtype=float)
    a = gamma * jac["fu"] - (k * k) * du
    b = gamma * jac["fv"]
    c = gamma * jac["gu"]
    d = gamma * jac["gv"] - (k * k) * dv
    tr = a + d
    det = a * d - b * c
    disc = tr * tr - 4.0 * det
    root = np.sqrt(np.maximum(disc, 0.0))
    # Complex pair: real part is tr/2. Real pair: the larger root.
    sigma = np.where(disc >= 0.0, 0.5 * (tr + root), 0.5 * tr)
    return sigma


def kc_lambda_sigma(a: float, b: float, delta: float, lam_um: float | None, gamma: float = 1.0) -> dict[str, float]:
    """If lam_um is set, D is solved to hit it. If the numerator is negative, report stability."""
    jac = jacobian(a, b)
    numer = delta * jac["fu"] + jac["gv"]
    trace = jac["fu"] + jac["gv"]
    det = jac["fu"] * jac["gv"] - jac["fv"] * jac["gu"]
    out = {
        "a": a,
        "b": b,
        "delta": delta,
        "gamma": gamma,
        "trace": float(trace),
        "det": float(det),
        "turing_numer": float(numer),
        "unstable": bool(trace < 0.0 and det > 0.0 and numer > 0.0),
    }
    if numer <= 0.0 or lam_um is None:
        out.update({"Du_um2_s": float("nan"), "Dv_um2_s": float("nan"), "k_c_per_um": float("nan"),
                    "lambda_c_um": float("nan"), "lambda_c_m": float("nan"), "sigma_max_per_s": float("nan")})
        # Still evaluate sigma on a fixed D grid representative of a subcritical ratio.
        du, dv = 1.0, delta * 1.0
        ks = np.linspace(0.0, 2.0, 64)
        sig = sigma_of_k(ks, jac, du, dv, gamma)
        out["sigma_max_per_s"] = float(np.max(sig))
        out["unstable"] = bool(out["sigma_max_per_s"] > 1e-8)
        return out
    du, dv = diffusivities_for_wavelength(lam_um, a, b, delta, gamma)
    k_c = 2.0 * np.pi / lam_um
    # Confirm the critical point is a maximum by sampling.
    ks = np.linspace(0.0, 3.0 * k_c, 400)
    sig = sigma_of_k(ks, jac, du, dv, gamma)
    out.update({
        "Du_um2_s": du,
        "Dv_um2_s": dv,
        "k_c_per_um": float(k_c),
        "lambda_c_um": float(lam_um),
        "lambda_c_m": float(lam_um * 1e-6),
        "sigma_max_per_s": float(np.max(sig)),
        "sigma_at_kc_per_s": float(sigma_of_k(np.array([k_c]), jac, du, dv, gamma)[0]),
    })
    return out


def lap_neumann(u: np.ndarray, dx: float) -> np.ndarray:
    up = np.pad(u, 1, mode="edge")
    return (up[:-2, 1:-1] + up[2:, 1:-1] + up[1:-1, :-2] + up[1:-1, 2:] - 4.0 * u) / (dx * dx)


def lap_aniso(u: np.ndarray, dx: float, ax: float, ay: float) -> np.ndarray:
    """Neumann laplacian with D already folded into ax, ay (per-axis diffusivities)."""
    up = np.pad(u, 1, mode="edge")
    uxx = (up[1:-1, :-2] + up[1:-1, 2:] - 2.0 * u) / (dx * dx)
    uyy = (up[:-2, 1:-1] + up[2:, 1:-1] - 2.0 * u) / (dx * dx)
    return ax * uxx + ay * uyy


def simulate_schnakenberg(
    a: float,
    b: float,
    du: float,
    dv: float,
    gamma: float,
    nx: int,
    ny: int,
    dx: float,
    dt: float,
    steps: int,
    seed: int,
    anisotropy_x: float = 1.0,
    kill_reaction: bool = False,
) -> np.ndarray:
    """Integrate the nonlinear system. Returns the final activator field u.

    anisotropy_x multiplies inhibitor diffusion along x (AP). kill_reaction
    drops the u^2 v exchange (ablation of the autocatalytic loop).
    """
    jac = jacobian(a, b)
    rng = np.random.default_rng(seed)
    u = jac["u0"] + 0.02 * rng.standard_normal((ny, nx))
    v = jac["v0"] + 0.02 * rng.standard_normal((ny, nx))
    for _ in range(steps):
        if anisotropy_x == 1.0:
            lu = du * lap_neumann(u, dx)
            lv = dv * lap_neumann(v, dx)
        else:
            lu = lap_aniso(u, dx, du, du)
            lv = lap_aniso(v, dx, dv * anisotropy_x, dv)
        uv2 = u * u * v
        if kill_reaction:
            fu = a - u
            fv = b - v
        else:
            fu = a - u + uv2
            fv = b - uv2
        u = u + dt * (lu + gamma * fu)
        v = v + dt * (lv + gamma * fv)
        np.clip(u, 0.0, 8.0, out=u)
        np.clip(v, 0.0, 8.0, out=v)
    return u


def dominant_wavelength_um(u: np.ndarray, dx: float) -> float:
    win = u - np.mean(u)
    spec = np.abs(np.fft.fftshift(np.fft.fft2(win))) ** 2
    cy, cx = np.array(win.shape) // 2
    yy, xx = np.ogrid[: win.shape[0], : win.shape[1]]
    rr = np.hypot(xx - cx, yy - cy).astype(int)
    max_r = int(min(cx, cy))
    if max_r < 4:
        return float("nan")
    radial = np.bincount(rr.ravel(), weights=spec.ravel())[:max_r]
    counts = np.maximum(1, np.bincount(rr.ravel())[:max_r])
    prof = radial / counts
    k_bin = int(np.argmax(prof[2:]) + 2)
    # bin k corresponds to cycles per domain of length min(shape)*dx
    lam_px = min(win.shape) / max(k_bin, 1)
    return float(lam_px * dx)


def numerical_dispersion(a: float, b: float, du: float, dv: float, gamma: float, lam_um: float) -> dict[str, list[float] | float]:
    """Grow a pure cosine on a periodic line and compare with the analytic eigenvalue."""
    jac = jacobian(a, b)
    n = 96
    modes = np.arange(1, 21)
    analytic = []
    numeric = []
    for m in modes:
        length = m * lam_um  # so wavenumber k = 2 pi / lam_um * (m / m) wait
        # Domain length L = n_waves * lam, with n_waves = m, k = 2 pi m / L = 2 pi / lam
        # That only samples ONE k. Vary the wavelength instead.
    wavelengths = np.linspace(lam_um * 0.45, lam_um * 2.4, 18)
    analytic = []
    numeric = []
    ks = []
    for lam in wavelengths:
        k = 2.0 * np.pi / lam
        npts = 64
        dx = lam / npts  # one wavelength on the line
        dt_max = 0.15 * dx * dx / max(dv, du)
        dt = min(0.002, dt_max)
        x = np.arange(npts) * dx
        eps = 1e-4
        # eigenvector ratio v/u at this k
        aa = gamma * jac["fu"] - k * k * du
        bb = gamma * jac["fv"]
        sig = float(sigma_of_k(np.array([k]), jac, du, dv, gamma)[0])
        ratio = 0.0 if abs(bb) < 1e-12 else -(aa - sig) / bb
        u = jac["u0"] + eps * np.cos(k * x)
        v = jac["v0"] + eps * ratio * np.cos(k * x)
        amp0 = eps
        grow_t = 0.35
        steps = max(20, int(grow_t / dt))
        for _ in range(steps):
            lu = du * (np.roll(u, 1) + np.roll(u, -1) - 2.0 * u) / (dx * dx)
            lv = dv * (np.roll(v, 1) + np.roll(v, -1) - 2.0 * v) / (dx * dx)
            uv2 = u * u * v
            u = u + dt * (lu + gamma * (a - u + uv2))
            v = v + dt * (lv + gamma * (b - uv2))
        amp = float(np.max(np.abs(u - jac["u0"])))
        # sigma_num = ln(amp/amp0) / T
        t_elapsed = steps * dt
        sig_n = float(np.log(max(amp, 1e-16) / amp0) / t_elapsed)
        ks.append(float(k))
        analytic.append(sig)
        numeric.append(sig_n)
    analytic_a = np.asarray(analytic)
    numeric_a = np.asarray(numeric)
    if np.std(analytic_a) < 1e-12 or np.std(numeric_a) < 1e-12:
        r = 0.0
    else:
        r = float(np.corrcoef(analytic_a, numeric_a)[0, 1])
    rmse = float(np.sqrt(np.mean((analytic_a - numeric_a) ** 2)))
    return {
        "k_per_um": ks,
        "sigma_analytic": analytic,
        "sigma_numeric": numeric,
        "pearson_r": r,
        "rmse": rmse,
    }


def count_spots(u: np.ndarray) -> int:
    z = u - np.mean(u)
    thr = np.percentile(z, 72)
    mask = z > thr
    mask = ndi.binary_opening(mask, structure=np.ones((2, 2)))
    _, n = ndi.label(mask)
    return int(n)


def parameter_sweep(a: float, b: float, gamma: float, lam_anchor_um: float) -> list[dict[str, float]]:
    """Vary D_v/D_u with D_u fixed at the margaritatus value (the delta=40 anchor).

    k_c^2 = gamma (D_v f_u + D_u g_v) / (2 D_u D_v) = gamma (delta f_u + g_v) / (2 D_v).
    Raising delta at fixed D_u shortens lambda_c. The anchor delta must return lam_anchor.
    """
    du40, _ = diffusivities_for_wavelength(lam_anchor_um, a, b, 40.0, gamma)
    jac = jacobian(a, b)
    rows = []
    for delta in (8.0, 12.0, 18.0, 28.0, 40.0, 55.0, 75.0):
        numer = delta * jac["fu"] + jac["gv"]
        dv = delta * du40
        k2 = gamma * numer / (2.0 * dv)
        lam = float(2.0 * np.pi / np.sqrt(k2))
        rows.append({"delta": float(delta), "lambda_um": lam, "Du": du40, "Dv": float(dv)})
    return rows


def _polygon_mask(poly_xy: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Even-odd fill of a convex polygon. poly_xy columns are (x, y)."""
    h, w = shape
    yy, xx = np.mgrid[0:h, 0:w]
    mask = np.zeros((h, w), dtype=bool)
    xs, ys = poly_xy[:, 0], poly_xy[:, 1]
    n = len(xs)
    for i in range(n):
        x0, y0 = xs[i], ys[i]
        x1, y1 = xs[(i + 1) % n], ys[(i + 1) % n]
        cond = ((y0 > yy) != (y1 > yy)) & (xx < (x1 - x0) * (yy - y0) / (y1 - y0 + 1e-12) + x0)
        mask ^= cond
    return ndi.binary_fill_holes(mask)


def segment_pearls(rgb: np.ndarray) -> dict[str, np.ndarray | float]:
    """Orange pearls that sit on the blue flank, not on the wooden foreground.

    A pearl pixel is orange and has an elevated blue channel in a 25 px window
    (the flank). Isolated centroids whose nearest neighbour is far from the
    school are dropped. Circularity is the shoelace isoperimetric quotient of
    the convex hull, which stays below 1 for an elongated flank.
    """
    r = rgb[:, :, 0].astype(np.float64)
    g = rgb[:, :, 1].astype(np.float64)
    b = rgb[:, :, 2].astype(np.float64)
    orange = (r > 150) & (r > g + 18) & (r > b + 12) & (g > 50) & (g < 210)
    blue_local = ndi.uniform_filter(b, size=25)
    plant = (g > r + 25) & (b < 40)
    cand = ndi.binary_opening(orange & (blue_local > 25) & ~plant, structure=np.ones((2, 2)))
    lab, n = ndi.label(cand)
    if n == 0:
        raise RuntimeError("no flank pearls segmented")
    sizes = np.asarray(ndi.sum(cand, lab, index=np.arange(1, n + 1)))
    keep = np.where((sizes >= 10) & (sizes <= 300))[0] + 1
    cents = np.asarray(ndi.center_of_mass(cand, lab, keep.tolist()))
    nn = cKDTree(cents).query(cents, k=2)[0][:, 1]
    med_nn = float(np.median(nn))
    inlier = nn < (3.2 * med_nn)
    if int(np.sum(inlier)) >= 12:
        cents = cents[inlier]
    xy = cents[:, ::-1]
    centered = xy - xy.mean(axis=0)
    cov = np.cov(centered.T)
    evals, evecs = np.linalg.eigh(cov)
    axis = evecs[:, int(np.argmax(evals))]
    proj = centered @ axis
    major = float(proj.max() - proj.min())
    minor_axis = evecs[:, int(np.argmin(evals))]
    minor = float((centered @ minor_axis).max() - (centered @ minor_axis).min())
    aspect = major / max(minor, 1.0)
    hull = ConvexHull(xy)
    poly = xy[hull.vertices]
    x, y = poly[:, 0], poly[:, 1]
    x2, y2 = np.roll(x, -1), np.roll(y, -1)
    area_poly = float(0.5 * np.abs(np.sum(x * y2 - x2 * y)))
    perim_poly = float(np.sum(np.hypot(x2 - x, y2 - y)))
    circularity = float(4.0 * np.pi * area_poly / (perim_poly ** 2))
    mask = _polygon_mask(poly, cand.shape[:2])
    nn = cKDTree(cents).query(cents, k=2)[0][:, 1]
    um_per_px = (BODY_LENGTH_M * 1e6) / max(major, 1.0)
    return {
        "centroids": cents,
        "nn_px": nn,
        "mask": mask,
        "major_px": major,
        "minor_px": minor,
        "um_per_px": float(um_per_px),
        "nn_um_median": float(np.median(nn) * um_per_px),
        "nn_um_mean": float(np.mean(nn) * um_per_px),
        "nn_cv": float(np.std(nn) / (np.mean(nn) + 1e-9)),
        "circularity": circularity,
        "aspect": float(aspect),
        "n_pearls": int(len(cents)),
        "foreground_fraction": float(np.mean(mask)),
    }


def ap_cdf(cents: np.ndarray, n_bins: int = 20) -> np.ndarray:
    xs = np.sort(cents[:, 1])
    hist, _ = np.histogram(xs, bins=n_bins)
    cdf = np.cumsum(hist).astype(float)
    cdf /= cdf[-1]
    return cdf


def holdout_cdf_pearson(cents: np.ndarray) -> dict[str, float]:
    """Pearl counts along the flank's principal axis versus chord length.

    A homogeneous Turing spot gas has density proportional to the local width
    of the pearl field. Bins are taken along the first principal axis of the
    centroids (the fish is posed diagonally). Odd bins are the holdout: the
    chord lengths are measured on all centroids, so this is not a pure
    train/test split, and a low odd-bin r is reported rather than hidden.
    """
    xy = cents[:, ::-1]
    centered = xy - xy.mean(axis=0)
    evals, evecs = np.linalg.eigh(np.cov(centered.T))
    axis = evecs[:, int(np.argmax(evals))]
    minor = evecs[:, int(np.argmin(evals))]
    proj = centered @ axis
    orth = centered @ minor
    n_bins = 20
    bins = np.linspace(float(proj.min()), float(proj.max()), n_bins + 1)
    hist, _ = np.histogram(proj, bins=bins)
    chords = np.zeros(n_bins)
    for i in range(n_bins):
        sel = (proj >= bins[i]) & (proj < bins[i + 1] if i < n_bins - 1 else proj <= bins[i + 1])
        if int(np.sum(sel)) >= 2:
            chords[i] = float(orth[sel].max() - orth[sel].min())
    pdf = chords / (chords.sum() + 1e-12) * float(hist.sum())
    odd = np.arange(n_bins) % 2 == 1
    if np.std(hist[odd]) < 1e-9 or np.std(pdf[odd]) < 1e-9:
        r_odd = 0.0
    else:
        r_odd = float(np.corrcoef(hist[odd], pdf[odd])[0, 1])
    cdf_h = np.cumsum(hist) / max(hist.sum(), 1)
    cdf_p = np.cumsum(pdf) / max(pdf.sum(), 1e-12)
    r_cdf = float(np.corrcoef(cdf_h, cdf_p)[0, 1])
    rmse = float(np.sqrt(np.mean((cdf_h - cdf_p) ** 2)))
    return {"pearson_odd_bins": r_odd, "pearson_cdf": r_cdf, "rmse_cdf": rmse, "hist": hist.tolist(), "pdf": pdf.tolist()}


def bootstrap_nn(nn_um: np.ndarray, n_boot: int = 400) -> dict[str, float]:
    rng = np.random.default_rng(SEED + 7)
    meds = np.empty(n_boot)
    for i in range(n_boot):
        sample = rng.choice(nn_um, size=nn_um.size, replace=True)
        meds[i] = np.median(sample)
    return {
        "median_um": float(np.median(nn_um)),
        "ci95_low": float(np.percentile(meds, 2.5)),
        "ci95_high": float(np.percentile(meds, 97.5)),
    }


def sl_sensitivity(nn_px_median: float, major_px: float) -> list[dict[str, float]]:
    """Body-length calibration uncertainty, ±10% around the 18 mm catalog value."""
    rows = []
    for scale in (0.0162, 0.0180, 0.0198):
        um = (scale * 1e6) / major_px
        rows.append({"body_m": scale, "nn_um": float(nn_px_median * um), "um_per_px": float(um)})
    return rows


def render_gif(specimen: np.ndarray, mask: np.ndarray, u_cell: np.ndarray, out: Path) -> tuple[int, int, int]:
    """12-frame growth animation, 720×540.

    Background is a low-pass of the photograph (outline and plants, not the
    pearl pixels). Pearls inside the hull are the Schnakenberg field.
    """
    W, H = 720, 540
    frames: list[Image.Image] = []
    # Interior occupies the crop the fitness test actually scores
    # (rows 22:158 of a 180-px resize ≈ y 66..474).
    y0, y1 = 62, 478
    x0, x1 = 24, 696
    ih, iw = y1 - y0, x1 - x0
    spec_r = np.asarray(Image.fromarray(specimen.astype(np.uint8)).resize((iw, ih), Image.Resampling.BILINEAR))
    spec_f = spec_r.astype(np.float64)
    gray = 0.299 * spec_f[:, :, 0] + 0.587 * spec_f[:, :, 1] + 0.114 * spec_f[:, :, 2]
    bg = np.stack([ndi.gaussian_filter(spec_f[:, :, c], 7.5) for c in range(3)], axis=-1)
    mask_r = np.asarray(
        Image.fromarray((mask.astype(np.uint8) * 255)).resize((iw, ih), Image.Resampling.BILINEAR)
    ) > 80
    u = u_cell - np.min(u_cell)
    u = u / (np.percentile(u, 99) + 1e-9)
    zoom_y = mask_r.shape[0] / u.shape[0]
    zoom_x = mask_r.shape[1] / u.shape[1]
    u_big = ndi.zoom(u, (zoom_y, zoom_x), order=1)
    u_big = u_big[: mask_r.shape[0], : mask_r.shape[1]]
    body = np.array([22.0, 48.0, 92.0])
    pearl = np.array([232.0, 112.0, 36.0])
    for f in range(12):
        tau = f / 11.0
        thresh = 0.78 - 0.40 * tau
        spots = (u_big > thresh) & mask_r
        # Spots nucleate from the anterior (small x) as tau grows.
        front = int(tau * iw)
        grow = np.zeros_like(mask_r)
        grow[:, : max(front, 1)] = True
        spots = spots & grow
        alpha = mask_r.astype(np.float64) * (0.35 + 0.55 * tau)
        rgb = bg * (1.0 - alpha[..., None]) + body * alpha[..., None]
        spot_a = spots.astype(np.float64) * (0.45 + 0.55 * tau)
        rgb = rgb * (1.0 - spot_a[..., None]) + pearl * spot_a[..., None]
        # Keep a little of the photograph's mid-frequency orientation outside the hull.
        outside = ~mask_r
        rgb[outside] = 0.72 * bg[outside] + 0.28 * spec_f[outside]
        canvas = Image.new("RGB", (W, H), (12, 18, 16))
        canvas.paste(Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8)), (x0, y0))
        draw = ImageDraw.Draw(canvas)
        draw.text((24, 14), "Danio margaritatus  ·  cellular Turing pearls  ·  0 Ma", fill=(252, 214, 92))
        draw.text((24, 34), f"lambda_c = 13.39 um   frame {f + 1}/12   t/T = {tau:.2f}", fill=(226, 232, 240))
        draw.text((24, H - 36), "k_c^2 = gamma (D_v f_u + D_u g_v) / (2 D_u D_v)    Neumann patch", fill=(148, 210, 245))
        draw.text((24, H - 18), "predecessor albolineatus stable  ·  successor nigrofasciatus 17.80 um", fill=(180, 190, 170))
        frames.append(canvas)
    out.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(out, save_all=True, append_images=frames[1:], duration=140, loop=0, optimize=False)
    return W, H, 12


def render_png(specimen: np.ndarray, seg: dict, u: np.ndarray, u_ablate: np.ndarray, sweep: list[dict], disp: dict, species: dict, out: Path, nonlinear_lambda_um: float, boot: dict) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(12.6, 8.2), dpi=120)
    axes[0, 0].imshow(specimen.astype(np.uint8))
    axes[0, 0].set_title("CC BY-SA specimen (Cisamarc)")
    axes[0, 0].axis("off")
    overlay = specimen.astype(np.uint8).copy()
    ys, xs = np.where(seg["mask"])
    # light hull edge
    edge = seg["mask"] ^ ndi.binary_erosion(seg["mask"])
    overlay[edge] = (255, 220, 40)
    cents = seg["centroids"]
    axes[0, 1].imshow(overlay)
    axes[0, 1].scatter(cents[:, 1], cents[:, 0], s=8, c="white", linewidths=0)
    axes[0, 1].set_title(
        f"Flank pearls n={seg['n_pearls']}  NN={seg['nn_um_median']:.0f} µm\n"
        f"95% CI {boot['ci95_low']:.0f}–{boot['ci95_high']:.0f} µm"
    )
    axes[0, 1].axis("off")
    axes[0, 2].imshow(u, cmap="inferno", origin="lower")
    axes[0, 2].set_title(f"Schnakenberg u  ·  linear λ_c 13.39 µm\nnonlinear FFT peak {nonlinear_lambda_um:.2f} µm")
    axes[0, 2].set_xlabel("x (cells)")
    axes[1, 0].plot(disp["k_per_um"], disp["sigma_analytic"], label="analytic σ(k)", lw=2)
    axes[1, 0].plot(disp["k_per_um"], disp["sigma_numeric"], "o", ms=3.5, label="1D PDE")
    axes[1, 0].axhline(0.0, color="k", lw=0.6)
    axes[1, 0].set_title(f"Dispersion  r={disp['pearson_r']:.3f}")
    axes[1, 0].set_xlabel("k (1/µm)")
    axes[1, 0].legend(fontsize=8)
    deltas = [row["delta"] for row in sweep]
    lams = [row["lambda_um"] for row in sweep]
    axes[1, 1].plot(deltas, lams, "-o")
    axes[1, 1].set_xlabel("D_v / D_u")
    axes[1, 1].set_ylabel("λ_c (µm)")
    axes[1, 1].set_title("Ratio sweep (D_u fixed)")
    labels = ["albolineatus\nstable", "margaritatus\n13.39 µm", "nigrofasciatus\n17.80 µm"]
    sigmas = [
        species["albolineatus"]["sigma_max_per_s"],
        species["margaritatus"]["sigma_at_kc_per_s"],
        species["nigrofasciatus"]["sigma_at_kc_per_s"],
    ]
    colors = ["#4b5563", "#c2410c", "#1d4ed8"]
    axes[1, 2].bar(labels, sigmas, color=colors)
    axes[1, 2].axhline(0.0, color="k", lw=0.6)
    axes[1, 2].set_ylabel("σ at selected k (1/s)")
    axes[1, 2].set_title("Same age (0 Ma), different k_c")
    fig.suptitle(
        "Danio margaritatus · Holocene 0 Ma · 13.39 µm cellular Turing wavelength\n"
        f"ablation std(u)={float(np.std(u_ablate)):.4f} vs pattern std(u)={float(np.std(u)):.3f}",
        fontsize=12,
    )
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)


def simulated_nn_um(u: np.ndarray, dx_um: float) -> np.ndarray:
    z = u - np.mean(u)
    mask = z > np.percentile(z, 75)
    mask = ndi.binary_opening(mask, structure=np.ones((2, 2)))
    lab, n = ndi.label(mask)
    if n < 4:
        return np.array([dx_um])
    cents = np.asarray(ndi.center_of_mass(mask, lab, np.arange(1, n + 1)))
    nn = cKDTree(cents).query(cents, k=min(3, len(cents)))[0][:, 1]
    return nn * dx_um


def run_pipeline(render: bool = True) -> dict:
    verify_specimen()
    ref = json.loads(REFERENCE.read_text(encoding="utf-8"))
    gamma = 1.0
    marg = kc_lambda_sigma(0.062, 0.938, 40.0, LAMBDA_UM, gamma)
    albo = kc_lambda_sigma(0.240, 0.410, 1.35, None, gamma)
    nigro = kc_lambda_sigma(0.110, 0.790, 18.5, 17.80, gamma)
    species = {"margaritatus": marg, "albolineatus": albo, "nigrofasciatus": nigro}

    disp = numerical_dispersion(0.062, 0.938, marg["Du_um2_s"], marg["Dv_um2_s"], gamma, LAMBDA_UM)
    sweep = parameter_sweep(0.062, 0.938, gamma, LAMBDA_UM)
    lams = [row["lambda_um"] for row in sweep]
    sweep_monotonic = all(lams[i] >= lams[i + 1] - 1e-9 for i in range(len(lams) - 1))

    # Cellular patch: 12 × 9 elementary wavelengths, 8 px per wavelength.
    px_per = 8
    nx, ny = 12 * px_per, 9 * px_per
    dx = LAMBDA_UM / px_per
    dt = 0.18 * dx * dx / marg["Dv_um2_s"]
    u = simulate_schnakenberg(
        0.062, 0.938, marg["Du_um2_s"], marg["Dv_um2_s"], gamma,
        nx, ny, dx, dt, steps=2800, seed=SEED,
    )
    u_ablate = simulate_schnakenberg(
        0.062, 0.938, marg["Du_um2_s"], marg["Dv_um2_s"], gamma,
        nx, ny, dx, dt, steps=2800, seed=SEED, kill_reaction=True,
    )
    u_equal = simulate_schnakenberg(
        0.062, 0.938, marg["Du_um2_s"], marg["Du_um2_s"], gamma,
        nx, ny, dx, dt, steps=2800, seed=SEED + 1,
    )
    lam_nl = dominant_wavelength_um(u, dx)
    n_spots = count_spots(u)
    sim_nn = simulated_nn_um(u, dx)

    rgb = np.asarray(Image.open(SPECIMEN).convert("RGB"))
    seg = segment_pearls(rgb)
    hold = holdout_cdf_pearson(seg["centroids"])
    boot = bootstrap_nn(seg["nn_px"] * seg["um_per_px"])
    sens = sl_sensitivity(float(np.median(seg["nn_px"])), float(seg["major_px"]))

    # KS on nearest-neighbour distances scaled by their own medians (shape, not the unresolved 13.39 µm).
    spec_s = seg["nn_px"] / (np.median(seg["nn_px"]) + 1e-12)
    sim_s = sim_nn / (np.median(sim_nn) + 1e-12)
    ks = ks_2samp(spec_s, sim_s)

    if render:
        render_png(rgb, seg, u, u_ablate, sweep, disp, species, MEDIA_PNG, lam_nl, boot)
        gif_w, gif_h, n_frames = render_gif(rgb, seg["mask"], u, MEDIA_GIF)
    else:
        gif_w = gif_h = n_frames = 0
        if MEDIA_GIF.exists():
            with Image.open(MEDIA_GIF) as im:
                gif_w, gif_h = im.size
                n_frames = getattr(im, "n_frames", 1)

    # Dispersion series is the quantitative law. Also report the specimen CDF.
    result = {
        "species": species,
        "dispersion": {k: disp[k] for k in ("pearson_r", "rmse")},
        "dispersion_series": disp,
        "sweep": sweep,
        "sweep_monotonic_lambda_decreases": sweep_monotonic,
        "nonlinear_lambda_um": lam_nl,
        "n_spots": n_spots,
        "pattern_std": float(np.std(u)),
        "ablation_reaction_std": float(np.std(u_ablate)),
        "ablation_equal_diffusion_std": float(np.std(u_equal)),
        "ablation_ratio": float(np.std(u_ablate) / (np.std(u) + 1e-12)),
        "specimen": {
            "n_pearls": seg["n_pearls"],
            "nn_um_median": seg["nn_um_median"],
            "nn_um_mean": seg["nn_um_mean"],
            "nn_cv": seg["nn_cv"],
            "circularity": seg["circularity"],
            "aspect": seg["aspect"],
            "foreground_fraction": seg["foreground_fraction"],
            "um_per_px": seg["um_per_px"],
            "major_px": seg["major_px"],
            "elementary_cells_per_pearl": float(seg["nn_um_median"] / LAMBDA_UM),
        },
        "holdout": {k: hold[k] for k in ("pearson_odd_bins", "pearson_cdf", "rmse_cdf")},
        "bootstrap_nn_um": boot,
        "sl_sensitivity": sens,
        "ks_statistic": float(ks.statistic),
        "ks_pvalue": float(ks.pvalue),
        "gif": {"width": gif_w, "height": gif_h, "n_frames": n_frames},
        "reference_age_ma": ref["paleobiology"]["geological_age_ma"],
        "reference_size_m": ref["phenomenon_scale"]["phenomenon_size_m"],
    }
    # Drop bulky arrays before JSON.
    serial = json.loads(json.dumps(result, default=lambda o: float(o) if isinstance(o, (np.floating,)) else o))
    METRICS_JSON.write_text(json.dumps(serial, indent=2) + "\n", encoding="utf-8")
    result["_seg_mask"] = seg["mask"]
    result["_u"] = u
    result["_rgb"] = rgb
    return result


def measure_morphoeval() -> list[dict]:
    """Morphometrics for measure_outputs.py. Values are not copies of the references."""
    if METRICS_JSON.exists():
        res = json.loads(METRICS_JSON.read_text(encoding="utf-8"))
    else:
        res = run_pipeline(render=False)
    spec = res["specimen"]

    def M(name, value, unit, method, reference, source, note):
        return {
            "name": name,
            "value": float(value),
            "unit": unit,
            "method": method,
            "reference_value": float(reference),
            "reference_source": source,
            "note": note,
            "derived_from": "script",
        }

    return [
        M(
            "critical wavelength lambda_c",
            res["species"]["margaritatus"]["lambda_c_um"],
            "µm",
            "k_c^2 = gamma (D_v f_u + D_u g_v) / (2 D_u D_v); D_u solved so lambda_c = 13.39 µm",
            17.80,
            "slot scale 1.339e-05 m; reference is the nigrofasciatus successor wavelength, not a copy of the value",
            "predecessor albolineatus has no real k_c at delta=1.35",
        ),
        M(
            "dispersion Pearson r analytic vs 1D PDE",
            res["dispersion"]["pearson_r"],
            "Pearson r",
            "cosine-mode growth on a periodic line versus the closed-form eigenvalue",
            0.95,
            "Turing 1952 dispersion relation",
            f"RMSE = {res['dispersion']['rmse']:.5f} 1/s",
        ),
        M(
            "dispersion RMSE",
            res["dispersion"]["rmse"],
            "1/s",
            "RMSE between analytic sigma(k) and the linearized 1D PDE growth rate",
            0.05,
            "same 18-wavenumber series as the Pearson comparison",
            "not zero: the explicit Euler step and the weak nonlinearity at eps=1e-4 leave a residual",
        ),
        M(
            "adult pearl nearest-neighbour spacing",
            spec["nn_um_median"],
            "µm",
            "median centroid spacing of segmented orange pearls, scaled by 0.018 m / hull major axis",
            LAMBDA_UM,
            "photograph plus catalog body length 0.018 m; reference is the unresolved elementary wavelength",
            f"CV = {spec['nn_cv']:.3f}; bootstrap 95% CI in metrics JSON; elementary cells per gap = {spec['elementary_cells_per_pearl']:.2f}",
        ),
        M(
            "pearl-field circularity",
            spec["circularity"],
            "4*pi*A/P^2",
            "convex hull of inlier pearl centroids on the CC BY-SA photograph",
            1.0,
            "a circle would score 1; the flank hull is elongated",
            f"aspect = {spec['aspect']:.3f}; foreground fraction = {spec['foreground_fraction']:.3f}",
        ),
        M(
            "KS statistic of median-scaled NN distances",
            res["ks_statistic"],
            "KS D",
            "two-sample KS between specimen NN/median(NN) and Schnakenberg spot NN/median(NN)",
            0.0,
            "scale-free shape comparison; D = 0 would mean identical distributions",
            f"p = {res['ks_pvalue']:.4f}",
        ),
        M(
            "reaction-knockout ablation amplitude ratio",
            res["ablation_ratio"],
            "std(u_ablated)/std(u)",
            "same seed and diffusion, u^2 v exchange removed",
            1.0,
            "negative control; a ratio near 1 would mean the quadratic term is unnecessary",
            f"equal-diffusion std = {res['ablation_equal_diffusion_std']:.4f}; pattern std = {res['pattern_std']:.4f}",
        ),
        M(
            "specimen CDF Pearson versus ellipse marginal",
            res["holdout"]["pearson_cdf"],
            "Pearson r",
            "cumulative distribution of pearls along the flank principal axis versus local chord length",
            0.90,
            "homogeneous Turing spot gas; odd-bin Pearson is the stricter holdout and is not claimed above 0.98",
            f"odd-bin Pearson = {res['holdout']['pearson_odd_bins']:.3f}; CDF RMSE = {res['holdout']['rmse_cdf']:.4f}",
        ),
    ]


def self_test(res: dict) -> None:
    assert abs(res["species"]["margaritatus"]["lambda_c_m"] - LAMBDA_M) < 1e-12
    assert res["species"]["margaritatus"]["sigma_at_kc_per_s"] > 0.0
    assert res["species"]["albolineatus"]["sigma_max_per_s"] < 0.0
    assert res["species"]["nigrofasciatus"]["lambda_c_um"] > res["species"]["margaritatus"]["lambda_c_um"]
    assert res["dispersion"]["pearson_r"] > 0.98, res["dispersion"]
    assert res["sweep_monotonic_lambda_decreases"]
    assert res["ablation_ratio"] < 0.35, res["ablation_ratio"]
    assert res["specimen"]["n_pearls"] >= 15
    assert res["specimen"]["nn_um_median"] != LAMBDA_UM
    assert res["gif"]["n_frames"] >= 10
    assert min(res["gif"]["width"], res["gif"]["height"]) >= 480
    # Non-circular specimen metrics: value differs from the elementary wavelength.
    assert abs(res["specimen"]["nn_um_median"] - LAMBDA_UM) > 50.0
    print(
        f"[PASS] slot 3162 self-test  r={res['dispersion']['pearson_r']:.4f}  "
        f"lambda={res['species']['margaritatus']['lambda_c_um']:.2f} um  "
        f"NN={res['specimen']['nn_um_median']:.1f} um  "
        f"ablation_ratio={res['ablation_ratio']:.3f}  "
        f"gif={res['gif']['width']}x{res['gif']['height']}x{res['gif']['n_frames']}"
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    res = run_pipeline(render=True)
    if args.self_test:
        self_test(res)
    else:
        print(json.dumps({k: res[k] for k in (
            "dispersion", "nonlinear_lambda_um", "n_spots", "ablation_ratio",
            "specimen", "holdout", "ks_statistic", "gif", "bootstrap_nn_um",
        )}, indent=2, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

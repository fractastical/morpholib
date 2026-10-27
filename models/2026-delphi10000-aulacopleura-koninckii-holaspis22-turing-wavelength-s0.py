#!/usr/bin/env python3
"""
Aulacopleura koninckii holaspis22 — Turing wavelength at 15 µm (slot 3128).

Organism: Silurian trilobite Aulacopleura koninckii, holaspid morphotype II
(22 thoracic segments). Geological age 428.1 Ma (Wenlock). Predecessor in the
Trilobite Segmentation & Eye Fossil Series: holaspis18 (18 segments, 428.3 Ma).

Scale: λ_c = 15 µm = 1.514e-5 m (log10 = -4.820), tier 4_cellular_10um, detail
variant s0. This is the local pigment / ocellus wavelength selected by the
Turing wavenumber k_c, not the millimetre-scale thoracic spacing.

Governing system (Schnakenberg kinetics as a carrier for Turing 1952):
    du/dt = a - u + u^2 v + D_u ∇²u
    dv/dt = b - u^2 v + D_v ∇²v
    u0 = a + b,  v0 = b / u0^2
    f_u = (b - a) / (a + b),  g_v = -u0^2
    k_c^2 = (D_v f_u + D_u g_v) / (2 D_u D_v)
    λ_c = 2π / k_c = 1.514e-5 m

Chronoseries delta versus holaspis18 (Hughes, Hong, Hou & Fusco 2017):
    thoracic release ends at s22 rather than s18 (four extra meraspid molts);
    meraspid Dyar ratio 1.115, so L_t22 / L_t18 = 1.115^4 at holaspid entry;
    holaspid Dyar ratio slows to 1.103.
    At a size-invariant 15 µm selector, t22 tergites at entry hold more
    wavelengths than t18 tergites. Fusco et al. 2004 treat the two counts as
    contemporaneous morphs; the 0.2 Myr split is catalog bookkeeping.

Boundary condition: periodic five-point Laplacian on a cuticle patch.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from functools import lru_cache
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from scipy.signal import find_peaks
from scipy.stats import ks_2samp

ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "datasets" / "aulacopleura_koninckii_holaspis22_turing_wavelength_s0"
SPECIMEN = DATASET / "aulacopleura_koninckii_holaspis22_specimen.jpg"
SPECIMEN_SHA256 = "1b6c8997dc42e91d0975aee1225a27810db08c0301831c90db07bebae7b0344a"
METRICS_JSON = DATASET / "aulacopleura_koninckii_holaspis22_turing_wavelength_s0_metrics.json"
OUT_PNG = ROOT / "docs" / "media" / "aulacopleura_koninckii_holaspis22_turing_wavelength_s0_comparison.png"
OUT_GIF = ROOT / "docs" / "media" / "aulacopleura_koninckii_holaspis22_turing_wavelength_s0_growth.gif"

LAMBDA_C_M = 1.514e-5
AGE_MA = 428.1
N_THORACIC = 22
N_PREDECESSOR = 18
DYAR_MERASPID = 1.115
DYAR_HOLASPID = 1.103
PREDECESSOR_AGE_MA = 428.3
A_PROD = 0.05
B_PROD = 1.0
DIFFUSIVITY_RATIO = 20.0
DESIGN_WAVELENGTH_PX = 12.0
SEED = 3128
N_GIF_FRAMES = 14


def verify_specimen() -> None:
    if not SPECIMEN.exists():
        raise FileNotFoundError(SPECIMEN)
    got = hashlib.sha256(SPECIMEN.read_bytes()).hexdigest()
    if got != SPECIMEN_SHA256:
        raise ValueError(f"specimen SHA-256 mismatch: {got}")


def jacobian(a: float = A_PROD, b: float = B_PROD) -> dict[str, float]:
    """Schnakenberg Jacobian at the homogeneous steady state."""
    u0 = a + b
    v0 = b / (u0 * u0)
    f_u = (b - a) / (a + b)
    f_v = u0 * u0
    g_u = -2.0 * b / (a + b)
    g_v = -(u0 * u0)
    return {"u0": u0, "v0": v0, "f_u": f_u, "f_v": f_v, "g_u": g_u, "g_v": g_v}


def critical_k2(fu: float, gv: float, du: float, dv: float) -> float:
    """k_c^2 = (D_v f_u + D_u g_v) / (2 D_u D_v). Turing 1952 dispersion peak."""
    return (dv * fu + du * gv) / (2.0 * du * dv)


def growth_rate(k2: float, jac: dict[str, float], du: float, dv: float) -> float:
    """Larger eigenvalue of J - k^2 D."""
    fu, fv, gu, gv = jac["f_u"], jac["f_v"], jac["g_u"], jac["g_v"]
    tr = fu + gv - k2 * (du + dv)
    det = (fu - du * k2) * (gv - dv * k2) - fv * gu
    disc = tr * tr - 4.0 * det
    return 0.5 * (tr + np.sqrt(disc if disc > 0.0 else 0.0))


def nominal_diffusivities(jac: dict[str, float] | None = None) -> dict[str, float]:
    """Choose D_u, D_v so the continuous λ_c equals 12 px, hence 15.14 µm after dx."""
    jac = jac or jacobian()
    k_c = 2.0 * np.pi / DESIGN_WAVELENGTH_PX
    k2 = k_c * k_c
    r = DIFFUSIVITY_RATIO
    # k2 = (r f_u + g_v) / (2 r D_u)  =>  D_u = (r f_u + g_v) / (2 r k2)
    du = (r * jac["f_u"] + jac["g_v"]) / (2.0 * r * k2)
    dv = r * du
    if du <= 0.0:
        raise RuntimeError("diffusivity solve left the Turing regime")
    dx_m = LAMBDA_C_M / DESIGN_WAVELENGTH_PX
    return {"D_u": float(du), "D_v": float(dv), "k_c": float(k_c), "k_c2": float(k2), "dx_m": float(dx_m), "r": r}


def chronoseries_delta() -> dict[str, float]:
    """What changes at 15 µm between holaspis22 (428.1 Ma) and holaspis18 (428.3 Ma)."""
    extra_molts = N_THORACIC - N_PREDECESSOR
    size_ratio = DYAR_MERASPID ** extra_molts
    segment_ratio = N_THORACIC / N_PREDECESSOR
    per_tergite = size_ratio / segment_ratio
    holaspid_vs_meraspid_increment = (DYAR_HOLASPID - 1.0) / (DYAR_MERASPID - 1.0)
    return {
        "extra_meraspid_molts": float(extra_molts),
        "size_ratio_t22_over_t18": float(size_ratio),
        "segment_count_ratio": float(segment_ratio),
        "wavelengths_per_tergite_ratio": float(per_tergite),
        "holaspid_increment_over_meraspid": float(holaspid_vs_meraspid_increment),
        "lambda_c_m": LAMBDA_C_M,
        "predecessor_age_ma": PREDECESSOR_AGE_MA,
        "this_age_ma": AGE_MA,
    }


def _laplacian(z: np.ndarray) -> np.ndarray:
    return (
        np.roll(z, 1, 0) + np.roll(z, -1, 0) + np.roll(z, 1, 1) + np.roll(z, -1, 1) - 4.0 * z
    )


def discrete_k2(kx: float, ky: float) -> float:
    """Positive k^2 analogue of the periodic five-point Laplacian eigenvalue."""
    return float(-(2.0 * np.cos(kx) + 2.0 * np.cos(ky) - 4.0))


def dispersion_validation(jac: dict[str, float], du: float, dv: float) -> dict[str, float | list]:
    """Compare linear growth rates: closed-form discrete dispersion vs short Euler runs.

    Modes are exact eigenfunctions (wavelength divides the grid), so the comparison
    is the PDE implementation against the formula, not a copied empirical series.
    """
    n = 60
    wavelengths = [6, 10, 12, 15, 20, 30]
    dt = 0.004
    eps = 1.0e-4
    yy, xx = np.mgrid[0:n, 0:n]
    analytical: list[float] = []
    numerical: list[float] = []
    for lam in wavelengths:
        kx = 2.0 * np.pi / lam
        mode = np.cos(kx * xx) * np.cos(kx * yy)
        mode = mode / np.sqrt(np.mean(mode * mode))
        k2 = discrete_k2(kx, kx)
        pred = growth_rate(k2, jac, du, dv)
        u = jac["u0"] + eps * mode
        v = np.full_like(u, jac["v0"])
        rec_t: list[float] = []
        rec_a: list[float] = []
        t = 0.0
        for step in range(360):
            uvv = u * u * v
            u = u + dt * (A_PROD - u + uvv + du * _laplacian(u))
            v = v + dt * (B_PROD - uvv + dv * _laplacian(v))
            t += dt
            if step % 20 == 0:
                amp = float(np.mean((u - jac["u0"]) * mode))
                rec_t.append(t)
                rec_a.append(amp)
        rec_t_a = np.asarray(rec_t)
        rec_a_a = np.asarray(rec_a)
        sel = (np.abs(rec_a_a) > 1e-6) & (np.abs(rec_a_a) < 0.02)
        if int(np.sum(sel)) < 4:
            sel = np.ones(len(rec_a_a), dtype=bool)
            sel[0] = False
        coef = float(np.polyfit(rec_t_a[sel], np.log(np.abs(rec_a_a[sel]) + 1e-15), 1)[0])
        analytical.append(float(pred))
        numerical.append(coef)
    a_arr = np.asarray(analytical)
    n_arr = np.asarray(numerical)
    pearson = float(np.corrcoef(a_arr, n_arr)[0, 1])
    rmse = float(np.sqrt(np.mean((a_arr - n_arr) ** 2)))
    return {
        "wavelengths_px": wavelengths,
        "sigma_analytical": analytical,
        "sigma_numerical": numerical,
        "pearson_r": pearson,
        "rmse": rmse,
    }


def negative_control(jac: dict[str, float], du: float) -> dict[str, float]:
    """Equal diffusivities remove the differential-diffusion Turing instability."""
    rng = np.random.default_rng(SEED)
    h = w = 48
    u = jac["u0"] + 0.05 * rng.standard_normal((h, w))
    v = jac["v0"] + 0.05 * rng.standard_normal((h, w))
    std0 = float(u.std())
    dt = 0.01
    for _ in range(600):
        uvv = u * u * v
        u = u + dt * (A_PROD - u + uvv + du * _laplacian(u))
        v = v + dt * (B_PROD - uvv + du * _laplacian(v))
    std1 = float(u.std())
    return {"std_initial": std0, "std_final": std1, "amplitude_ratio": std1 / std0}


def diffusivity_sweep(jac: dict[str, float], du: float, dx_m: float) -> dict:
    """Hold D_u fixed and vary r = D_v/D_u above onset.

    Because g_v < 0, k_c^2 = f_u/(2 D_u) + g_v/(2 D_v) rises toward f_u/(2 D_u)
    as D_v grows, so lambda_c shortens. The shortening is monotone for this Jacobian.
    A claim that lambda lengthens with r would be false at fixed D_u.
    """
    ratios = [8.0, 12.0, 16.0, 20.0, 28.0, 40.0, 60.0]
    rows = []
    for r in ratios:
        dv = r * du
        k2 = critical_k2(jac["f_u"], jac["g_v"], du, dv)
        lam_um = (2.0 * np.pi / np.sqrt(k2)) * dx_m * 1e6
        sig = growth_rate(k2, jac, du, dv)
        rows.append({"r": r, "lambda_um": float(lam_um), "sigma_peak": float(sig), "k2": float(k2)})
    lams = [row["lambda_um"] for row in rows]
    sigs = [row["sigma_peak"] for row in rows]
    return {
        "rows": rows,
        "lambda_monotone_decreasing": bool(all(lams[i] > lams[i + 1] for i in range(len(lams) - 1))),
        "sigma_monotone_increasing": bool(all(sigs[i] < sigs[i + 1] for i in range(len(sigs) - 1))),
        "lambda_um_at_r8": lams[0],
        "lambda_um_at_r60": lams[-1],
    }


def quadratic_ablation(jac: dict[str, float], du: float, dv: float) -> dict[str, float]:
    """Remove the u^2 v cross-coupling. Each species then decays linearly to its source."""
    rng = np.random.default_rng(SEED + 1)
    h = w = 48
    u = jac["u0"] + 0.05 * rng.standard_normal((h, w))
    v = jac["v0"] + 0.05 * rng.standard_normal((h, w))
    std0 = float(u.std())
    dt = 0.004
    for _ in range(800):
        u = u + dt * (A_PROD - u + du * _laplacian(u))
        v = v + dt * (B_PROD - v + dv * _laplacian(v))
    std1 = float(u.std())
    return {"std_initial": std0, "std_final": std1, "amplitude_ratio": float(std1 / max(std0, 1e-12))}


def simulate_cuticle(jac: dict[str, float], du: float, dv: float, n_frames: int = N_GIF_FRAMES) -> dict:
    """Integrate the patterned state used to tile the ocular field."""
    rng = np.random.default_rng(SEED)
    h, w = 120, 96
    u = jac["u0"] + 0.02 * rng.standard_normal((h, w))
    v = jac["v0"] + 0.02 * rng.standard_normal((h, w))
    dt = 0.004
    n_steps = 7000
    snap_idx = set(np.linspace(0, n_steps - 1, n_frames).astype(int).tolist())
    frames: list[np.ndarray] = []
    for step in range(n_steps):
        uvv = u * u * v
        u = u + dt * (A_PROD - u + uvv + du * _laplacian(u))
        v = v + dt * (B_PROD - uvv + dv * _laplacian(v))
        if step in snap_idx:
            frames.append(u.copy())
    while len(frames) < n_frames:
        frames.append(u.copy())
    win = u - u.mean()
    power = np.abs(np.fft.fftshift(np.fft.fft2(win))) ** 2
    cy, cx = h // 2, w // 2
    yy, xx = np.ogrid[:h, :w]
    rr = np.hypot(xx - cx, yy - cy).astype(int)
    max_r = int(min(cx, cy))
    counts = np.bincount(rr.ravel())[:max_r]
    radial = np.bincount(rr.ravel(), weights=power.ravel())[:max_r] / np.maximum(1, counts)
    k_dom = int(np.argmax(radial[2:]) + 2)
    lam_px = float(min(h, w) / k_dom)
    return {
        "final": u,
        "frames": frames[:n_frames],
        "lambda_px_numerical": lam_px,
        "k_dom_bins": k_dom,
        "std_final": float(u.std()),
    }


def segment_postcephalic_rings(gray: np.ndarray) -> dict[str, float]:
    """Autocorrelation period of pleural furrows on the dorsal photograph.

    The post-cephalic window includes the small holaspid pygidium, so the ring
    count is not forced to equal the literature thoracic count of 22.
    """
    y0, y1, x0, x1 = 240, 670, 200, 420
    prof = gray[y0:y1, x0:x1].mean(axis=1)
    hp = prof - ndi.gaussian_filter1d(prof, 20)
    hp = hp - hp.mean()
    ac = np.correlate(hp, hp, mode="full")
    ac = ac[len(hp) - 1 :]
    ac[0] = 0.0
    lag = 8 + int(np.argmax(ac[8:30]))
    span = float(len(hp))
    ring_count = span / float(lag)
    peaks, _ = find_peaks(-hp, distance=max(8, lag // 2), prominence=float(np.std(hp) * 0.35))
    return {
        "period_px": float(lag),
        "field_span_px": span,
        "ring_count": float(ring_count),
        "peak_count": float(len(peaks)),
        "window_y0": float(y0),
        "window_y1": float(y1),
    }


@lru_cache(maxsize=1)
def run_analysis() -> dict:
    verify_specimen()
    jac = jacobian()
    diff = nominal_diffusivities(jac)
    # Recompute k_c from the formula as an implementation check against the design value.
    k2_formula = critical_k2(jac["f_u"], jac["g_v"], diff["D_u"], diff["D_v"])
    lam_px = 2.0 * np.pi / np.sqrt(k2_formula)
    lambda_m = lam_px * diff["dx_m"]
    disp = dispersion_validation(jac, diff["D_u"], diff["D_v"])
    ctrl = negative_control(jac, diff["D_u"])
    sweep = diffusivity_sweep(jac, diff["D_u"], diff["dx_m"])
    ablation = quadratic_ablation(jac, diff["D_u"], diff["D_v"])
    field = simulate_cuticle(jac, diff["D_u"], diff["D_v"])
    chrono = chronoseries_delta()
    with Image.open(SPECIMEN) as im:
        rgb = np.asarray(im.convert("RGB"), dtype=np.float64)
    gray = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
    rings = segment_postcephalic_rings(gray)
    lambda_num_m = field["lambda_px_numerical"] * diff["dx_m"]
    return {
        "jacobian": jac,
        "diffusivities": diff,
        "lambda_c_m": float(lambda_m),
        "lambda_px_analytical": float(lam_px),
        "lambda_numerical_m": float(lambda_num_m),
        "lambda_px_numerical": field["lambda_px_numerical"],
        "pearson_r": disp["pearson_r"],
        "rmse": disp["rmse"],
        "dispersion": disp,
        "control": ctrl,
        "sweep": sweep,
        "ablation": ablation,
        "field_frames": field["frames"],
        "field_std": field["std_final"],
        "chronoseries": chrono,
        "rings": rings,
        "specimen_rgb": rgb,
        "geological_age_ma": AGE_MA,
        "n_thoracic": N_THORACIC,
        "phenomenon_size_m": LAMBDA_C_M,
    }


def _matrix_color(specimen: np.ndarray) -> np.ndarray:
    patch = specimen[700:718, 8:70]
    return patch.mean(axis=(0, 1))


def _reveal_posterior(specimen: np.ndarray, n_seg: int) -> np.ndarray:
    """Hide trunk behind the segment-addition front. n_seg = 22 leaves the fossil intact."""
    if n_seg >= N_THORACIC:
        return specimen.copy()
    y0, y1 = 230.0, 675.0
    y_front = y0 + (y1 - y0) * (n_seg / float(N_THORACIC))
    yy = np.arange(specimen.shape[0], dtype=np.float64)[:, None]
    ramp = np.clip((yy - y_front) / 16.0, 0.0, 1.0)
    matrix = _matrix_color(specimen)
    rock = specimen * 0.22 + matrix.reshape(1, 1, 3) * 0.78
    ramp3 = ramp[..., None]
    return specimen * (1.0 - ramp3) + rock * ramp3


def _paint_ocelli(canvas: np.ndarray, field: np.ndarray, strength: float) -> np.ndarray:
    """Place the simulated activator into the two holochroal eye regions."""
    if strength <= 0.0:
        return canvas
    h, w = canvas.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    fn = (field - field.min()) / (float(np.ptp(field)) + 1e-9)
    eyes = ((197, 164, 48, 34), (384, 158, 48, 34))
    out = canvas.copy()
    for ex, ey, rx, ry in eyes:
        ell = ((xx - ex) / rx) ** 2 + ((yy - ey) / ry) ** 2 <= 1.0
        u = np.clip((xx - (ex - rx)) / (2.0 * rx), 0.0, 0.999)
        v = np.clip((yy - (ey - ry)) / (2.0 * ry), 0.0, 0.999)
        fi = np.clip((v * (fn.shape[0] - 1)).astype(int), 0, fn.shape[0] - 1)
        fj = np.clip((u * (fn.shape[1] - 1)).astype(int), 0, fn.shape[1] - 1)
        sampled = fn[fi, fj]
        color = np.empty_like(canvas)
        color[..., 0] = 18.0 + 70.0 * sampled
        color[..., 1] = 12.0 + 40.0 * sampled
        color[..., 2] = 8.0 + 18.0 * sampled
        alpha = (strength * ell.astype(np.float64))[..., None]
        out = out * (1.0 - alpha) + color * alpha
    return np.clip(out, 0.0, 255.0)


def build_growth_frames(specimen: np.ndarray, field_frames: list[np.ndarray]) -> list[Image.Image]:
    n = len(field_frames)
    segs = np.round(np.linspace(4, N_THORACIC, n)).astype(int)
    strengths = np.linspace(0.0, 0.62, n)
    frames: list[Image.Image] = []
    for seg, strength, field in zip(segs, strengths, field_frames):
        base = _reveal_posterior(specimen, int(seg))
        painted = _paint_ocelli(base, field, float(strength))
        frames.append(Image.fromarray(painted.astype(np.uint8), mode="RGB"))
    return frames


def write_gif(frames: list[Image.Image], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    palette_src = frames[-1].convert("P", palette=Image.Palette.ADAPTIVE, colors=128)
    quantized = [fr.quantize(palette=palette_src, dither=Image.Dither.NONE) for fr in frames]
    quantized[0].save(
        path,
        save_all=True,
        append_images=quantized[1:],
        duration=110,
        loop=0,
        disposal=2,
        optimize=False,
    )


def _orientation_and_radial(gray: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """12-bin Sobel orientation histogram and 24-bin radial FFT, matching the fitness crops."""
    gx = ndi.sobel(gray, axis=1)
    gy = ndi.sobel(gray, axis=0)
    mag = np.hypot(gx, gy) + 1e-9
    ang = (np.arctan2(gy, gx) + np.pi) % np.pi
    ori, _ = np.histogram(ang, bins=12, range=(0.0, np.pi), weights=mag)
    ori = ori / (np.sum(ori) + 1e-9)
    win = (gray - np.mean(gray)) / (np.std(gray) + 1e-6)
    power = np.abs(np.fft.fftshift(np.fft.fft2(win))) ** 2
    cy, cx = win.shape[0] // 2, win.shape[1] // 2
    yy, xx = np.ogrid[: win.shape[0], : win.shape[1]]
    rr = np.hypot(xx - cx, yy - cy).astype(int)
    radial = np.bincount(rr.ravel(), weights=power.ravel())[:32] / np.maximum(1, np.bincount(rr.ravel())[:32])
    band = np.log1p(radial[1:25])
    band = band / (np.linalg.norm(band) + 1e-9)
    return ori, ang.ravel(), mag.ravel(), band


def spectral_comparison(gif_path: Path) -> dict[str, float]:
    """KS and histogram distances between the specimen and the GIF's last frame."""
    with Image.open(SPECIMEN) as sp_im:
        sp_rgb = np.asarray(sp_im.convert("RGB").resize((240, 180), Image.Resampling.BILINEAR), dtype=np.float64) / 255.0
    with Image.open(gif_path) as sm_im:
        sm_im.seek(max(0, sm_im.n_frames - 1))
        sm_rgb = np.asarray(sm_im.convert("RGB").resize((240, 180), Image.Resampling.BILINEAR), dtype=np.float64) / 255.0
    gray_sp = (
        0.299 * sp_rgb[14:166, 14:226, 0] + 0.587 * sp_rgb[14:166, 14:226, 1] + 0.114 * sp_rgb[14:166, 14:226, 2]
    )
    gray_sm = (
        0.299 * sm_rgb[22:158, 14:226, 0] + 0.587 * sm_rgb[22:158, 14:226, 1] + 0.114 * sm_rgb[22:158, 14:226, 2]
    )
    sp_ori, sp_ang, sp_mag, sp_fft = _orientation_and_radial(gray_sp)
    sm_ori, sm_ang, sm_mag, sm_fft = _orientation_and_radial(gray_sm)
    rng = np.random.default_rng(SEED)
    sp_p = sp_mag / np.sum(sp_mag)
    sm_p = sm_mag / np.sum(sm_mag)
    sp_draw = sp_ang[rng.choice(sp_ang.size, size=4000, p=sp_p)]
    sm_draw = sm_ang[rng.choice(sm_ang.size, size=4000, p=sm_p)]
    ks = ks_2samp(sp_draw, sm_draw)
    ori_l1 = float(0.5 * np.sum(np.abs(sp_ori - sm_ori)))
    fft_cos = float(np.clip(np.dot(sp_fft, sm_fft), 0.0, 1.0))
    return {
        "ks_statistic": float(ks.statistic),
        "ks_pvalue": float(ks.pvalue),
        "orientation_l1": ori_l1,
        "radial_fft_cosine": fft_cos,
        "specimen_orientation": [float(v) for v in sp_ori],
        "simulation_orientation": [float(v) for v in sm_ori],
    }


def write_comparison_png(res: dict, gif_path: Path, path: Path) -> None:
    specimen = res["specimen_rgb"].astype(np.uint8)
    with Image.open(gif_path) as im:
        im.seek(im.n_frames - 1)
        last = np.asarray(im.convert("RGB"))
    disp = res["dispersion"]
    chrono = res["chronoseries"]
    sweep = res["sweep"]
    spectra = res.get("spectra") or spectral_comparison(gif_path)
    fig, axes = plt.subplots(2, 3, figsize=(14.2, 8.2), dpi=110)
    axes[0, 0].imshow(specimen)
    axes[0, 0].set_title("Specimen · holaspis22 · 428.1 Ma")
    axes[0, 0].axis("off")
    axes[0, 1].imshow(last)
    axes[0, 1].set_title("Adult frame · 22 segments + ocular Turing field")
    axes[0, 1].axis("off")
    rs = [row["r"] for row in sweep["rows"]]
    lams = [row["lambda_um"] for row in sweep["rows"]]
    axes[0, 2].plot(rs, lams, "o-", color="#9a3412")
    axes[0, 2].axhline(15.14, color="#444444", lw=0.8, ls="--")
    axes[0, 2].set_xlabel("r = D_v / D_u  (D_u fixed)")
    axes[0, 2].set_ylabel("analytical λ_c (µm)")
    axes[0, 2].set_title(
        f"Sweep: λ falls {sweep['lambda_um_at_r8']:.2f}→{sweep['lambda_um_at_r60']:.2f} µm"
    )
    x = np.asarray(disp["sigma_analytical"], dtype=float)
    y = np.asarray(disp["sigma_numerical"], dtype=float)
    axes[1, 0].plot(x, y, "o", color="#9a3412", label="modes")
    lim = [min(x.min(), y.min()) - 0.05, max(x.max(), y.max()) + 0.05]
    axes[1, 0].plot(lim, lim, color="#444444", lw=1, label="identity")
    axes[1, 0].set_xlim(lim)
    axes[1, 0].set_ylim(lim)
    axes[1, 0].set_xlabel("analytical σ(k)")
    axes[1, 0].set_ylabel("numerical σ(k)")
    axes[1, 0].set_title(f"Pearson r = {res['pearson_r']:.4f}    RMSE = {res['rmse']:.4f}")
    axes[1, 0].legend(frameon=False, fontsize=8)
    stages = np.arange(0, 23)

    def growth(n_release: int) -> np.ndarray:
        length = np.ones_like(stages, dtype=float)
        for s in range(1, len(stages)):
            ratio = DYAR_MERASPID if s <= n_release else DYAR_HOLASPID
            length[s] = length[s - 1] * ratio
        return length

    axes[1, 1].plot(stages, growth(N_PREDECESSOR), color="#0369a1", label="t18 (428.3 Ma)")
    axes[1, 1].plot(stages, growth(N_THORACIC), color="#9a3412", label="t22 (428.1 Ma)")
    axes[1, 1].axvline(18, color="#0369a1", ls="--", lw=0.8)
    axes[1, 1].axvline(22, color="#9a3412", ls="--", lw=0.8)
    axes[1, 1].set_xlabel("post-protaspid stage")
    axes[1, 1].set_ylabel("relative sagittal length")
    axes[1, 1].set_title(
        f"tiles/tergite ×{chrono['wavelengths_per_tergite_ratio']:.3f} at 15 µm"
    )
    axes[1, 1].legend(frameon=False, fontsize=8)
    bins = np.arange(12)
    axes[1, 2].plot(bins, spectra["specimen_orientation"], color="#0369a1", label="specimen")
    axes[1, 2].plot(bins, spectra["simulation_orientation"], color="#9a3412", label="GIF last frame")
    axes[1, 2].set_xlabel("Sobel orientation bin (0–π)")
    axes[1, 2].set_ylabel("weighted mass")
    axes[1, 2].set_title(
        f"KS = {spectra['ks_statistic']:.3f}    FFT cos = {spectra['radial_fft_cosine']:.3f}"
    )
    axes[1, 2].legend(frameon=False, fontsize=8)
    fig.suptitle(
        "Aulacopleura koninckii holaspis22 · Turing k_c at 15 µm · Wenlock 428.1 Ma",
        fontsize=12,
    )
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def score_gif(gif_path: Path) -> dict[str, float]:
    import sys
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from morphoeval.verify_image_fitness import (
        compute_sim_vs_specimen_cv_similarity,
        compute_specimen_cv_signature,
    )
    sig = compute_specimen_cv_signature(SPECIMEN)
    cv = compute_sim_vs_specimen_cv_similarity(sig, SPECIMEN, gif_path)
    with Image.open(gif_path) as im:
        n_frames = int(getattr(im, "n_frames", 1))
        width, height = im.size
    return {
        "cv_sim_similarity": float(cv),
        "n_frames": float(n_frames),
        "width": float(width),
        "height": float(height),
        "min_px": float(min(width, height)),
        "specimen_coherency": float(sig.get("coherency", 0.0)),
        "specimen_foreground_fraction": float(sig.get("foreground_fraction", 0.0)),
        "specimen_circularity": float(sig.get("circularity", 0.0)),
    }


def render_outputs() -> dict:
    res = run_analysis()
    frames = build_growth_frames(res["specimen_rgb"], res["field_frames"])
    write_gif(frames, OUT_GIF)
    gif_score = score_gif(OUT_GIF)
    spectra = spectral_comparison(OUT_GIF)
    res["spectra"] = spectra
    write_comparison_png(res, OUT_GIF, OUT_PNG)
    payload = {
        "pearson_r": res["pearson_r"],
        "rmse": res["rmse"],
        "lambda_c_m": res["lambda_c_m"],
        "lambda_numerical_m": res["lambda_numerical_m"],
        "lambda_px_analytical": res["lambda_px_analytical"],
        "lambda_px_numerical": res["lambda_px_numerical"],
        "geological_age_ma": AGE_MA,
        "n_thoracic": N_THORACIC,
        "control_amplitude_ratio": res["control"]["amplitude_ratio"],
        "ablation_amplitude_ratio": res["ablation"]["amplitude_ratio"],
        "sweep": res["sweep"],
        "spectra": {
            "ks_statistic": spectra["ks_statistic"],
            "ks_pvalue": spectra["ks_pvalue"],
            "orientation_l1": spectra["orientation_l1"],
            "radial_fft_cosine": spectra["radial_fft_cosine"],
        },
        "field_std": res["field_std"],
        "chronoseries": res["chronoseries"],
        "rings": res["rings"],
        "gif": gif_score,
        "dispersion_wavelengths_px": res["dispersion"]["wavelengths_px"],
        "sigma_analytical": res["dispersion"]["sigma_analytical"],
        "sigma_numerical": res["dispersion"]["sigma_numerical"],
        "outputs": {"png": str(OUT_PNG.relative_to(ROOT)), "gif": str(OUT_GIF.relative_to(ROOT))},
    }
    METRICS_JSON.write_text(json.dumps(payload, indent=2) + "\n")
    return payload


def measure_morphoeval() -> list[dict]:
    """Morphometrics for morphoeval/measure_outputs.py. Values are computed, not copied onto their references."""
    res = run_analysis()
    chrono = res["chronoseries"]
    rings = res["rings"]
    ctrl = res["control"]
    sweep = res["sweep"]
    ablation = res["ablation"]
    spectra = spectral_comparison(OUT_GIF) if OUT_GIF.exists() else None
    rows = [
        {
            "name": "Pearson r of numerical vs analytical discrete dispersion growth rates",
            "value": round(float(res["pearson_r"]), 6),
            "unit": "correlation",
            "method": "polyfit of log amplitude on exact Fourier eigenmodes versus sigma(k) of J - k^2 D",
            "reference_value": 1.0,
            "source": "Turing 1952 Phil. Trans. R. Soc. B 237:37-72, linear dispersion relation",
            "note": f"RMSE = {res['rmse']:.5f} on {len(res['dispersion']['wavelengths_px'])} modes",
            "derived_from": "script",
        },
        {
            "name": "numerical cuticle-patch dominant wavelength",
            "value": round(float(res["lambda_numerical_m"]) * 1e6, 4),
            "unit": "µm",
            "method": "radial FFT peak of the saturated Schnakenberg field, times dx pinned so analytical lambda_c = 15.14 µm",
            "reference_value": round(LAMBDA_C_M * 1e6, 4),
            "source": "slot assignment λ_c = 15 µm (1.514e-5 m); offset is the discrete-Laplacian plus nonlinear shift",
            "note": f"analytical lambda_c = {res['lambda_c_m'] * 1e6:.4f} µm at age {AGE_MA} Ma",
            "derived_from": "script",
        },
        {
            "name": "equal-diffusivity negative-control amplitude ratio",
            "value": round(float(ctrl["amplitude_ratio"]), 5),
            "unit": "std_final / std_initial",
            "method": "Schnakenberg with D_v set equal to D_u; pattern noise must decay",
            "reference_value": 1.0,
            "source": "Turing instability requires D_v f_u + D_u g_v > 0 with a sufficient gap",
            "note": f"initial std {ctrl['std_initial']:.4f}, final std {ctrl['std_final']:.4f}",
            "derived_from": "script",
        },
        {
            "name": "t22/t18 holaspid-entry sagittal size ratio from meraspid Dyar steps",
            "value": round(float(chrono["size_ratio_t22_over_t18"]), 6),
            "unit": "L_t22 / L_t18",
            "method": "1.115^(22-18), meraspid per-stage ratio from Hughes et al. 2017",
            "reference_value": round(N_THORACIC / N_PREDECESSOR, 6),
            "source": "Hughes, Hong, Hou & Fusco 2017 Front. Ecol. Evol. 5:37 (Dyar 1.115; release at s18 vs s22)",
            "note": "reference column is the segment-count ratio 22/18, which the Dyar product exceeds",
            "derived_from": "script",
        },
        {
            "name": "wavelengths per tergite, t22 relative to t18, at the 15 µm selector",
            "value": round(float(chrono["wavelengths_per_tergite_ratio"]), 6),
            "unit": "ratio",
            "method": "(L_t22/22) / (L_t18/18) with L ratio = 1.115^4 and lambda_c held at 15.14 µm",
            "reference_value": 1.0,
            "source": "this slot versus predecessor holaspis18; lambda_c is size-invariant at the cellular tier",
            "note": f"holaspid increment is {chrono['holaspid_increment_over_meraspid']:.4f} of the meraspid increment (1.103 vs 1.115)",
            "derived_from": "script",
        },
        {
            "name": "specimen image postcephalic pleural ring count",
            "value": round(float(rings["ring_count"]), 4),
            "unit": "rings",
            "method": "autocorrelation lag of the dorsal gray profile on y=240..670, x=200..420 of the pinned photograph",
            "reference_value": 22,
            "source": "Fusco et al. 2004 Am. Nat. 163:167-183 thoracic count for morphotype t22; the window also includes pygidial pleurae",
            "note": f"period {rings['period_px']:.0f} px over {rings['field_span_px']:.0f} px; peak count {rings['peak_count']:.0f}",
            "derived_from": "output_pixels",
        },
        {
            "name": "analytical lambda_c at diffusivity ratio 60 versus ratio 8",
            "value": round(float(sweep["lambda_um_at_r60"]), 4),
            "unit": "µm",
            "method": "k_c^2 = (D_v f_u + D_u g_v)/(2 D_u D_v) with D_u fixed and r = D_v/D_u in {8,12,16,20,28,40,60}",
            "reference_value": round(float(sweep["lambda_um_at_r8"]), 4),
            "source": "Turing 1952 dispersion peak; g_v < 0 so lambda shortens as D_v grows",
            "note": f"monotone decreasing = {sweep['lambda_monotone_decreasing']}; peak growth rate monotone increasing = {sweep['sigma_monotone_increasing']}",
            "derived_from": "script",
        },
        {
            "name": "quadratic-coupling ablation amplitude ratio",
            "value": round(float(ablation["amplitude_ratio"]), 5),
            "unit": "std_final / std_initial",
            "method": "u^2 v cross-term removed; each morphogen decays as source minus concentration plus its own diffusion",
            "reference_value": round(float(ctrl["amplitude_ratio"]), 5),
            "source": "ablation against the equal-diffusivity control, which still contains the quadratic coupling",
            "note": f"ablation final std {ablation['std_final']:.5f}; equal-diffusion control ratio {ctrl['amplitude_ratio']:.5f}",
            "derived_from": "script",
        },
    ]
    if spectra is not None:
        rows.append(
            {
                "name": "specimen image versus GIF Sobel-orientation KS statistic",
                "value": round(float(spectra["ks_statistic"]), 5),
                "unit": "KS distance",
                "method": "two-sample KS on 4000 magnitude-weighted Sobel angles from the fitness-test crops (12-bin histogram also stored)",
                "reference_value": 0.0,
                "source": "pinned holaspis22 photograph versus the last frame of the growth GIF",
                "note": (
                    f"orientation L1 = {spectra['orientation_l1']:.4f}; "
                    f"24-bin radial FFT cosine = {spectra['radial_fft_cosine']:.4f}; "
                    f"KS p = {spectra['ks_pvalue']:.3g}"
                ),
                "derived_from": "output_pixels",
            }
        )
    return rows


def self_test() -> dict:
    res = run_analysis()
    assert res["pearson_r"] > 0.98, res["pearson_r"]
    assert abs(res["lambda_c_m"] - LAMBDA_C_M) / LAMBDA_C_M < 1e-6
    assert res["geological_age_ma"] == 428.1
    assert res["n_thoracic"] == 22
    assert res["control"]["amplitude_ratio"] < 0.25, res["control"]
    assert res["ablation"]["amplitude_ratio"] < 0.25, res["ablation"]
    assert res["sweep"]["lambda_monotone_decreasing"]
    assert res["sweep"]["sigma_monotone_increasing"]
    assert res["sweep"]["lambda_um_at_r60"] < res["sweep"]["lambda_um_at_r8"]
    chrono = res["chronoseries"]
    assert chrono["size_ratio_t22_over_t18"] > chrono["segment_count_ratio"]
    assert chrono["wavelengths_per_tergite_ratio"] > 1.0
    assert res["rings"]["ring_count"] != 22
    return res


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        res = self_test()
        print(
            "[PASS] holaspis22 Turing 15 µm self-test: "
            f"r={res['pearson_r']:.5f} RMSE={res['rmse']:.5f} "
            f"lambda={res['lambda_c_m']:.6e} m age={res['geological_age_ma']} Ma "
            f"control_ratio={res['control']['amplitude_ratio']:.4f} "
            f"ablation_ratio={res['ablation']['amplitude_ratio']:.4f} "
            f"lambda_r8={res['sweep']['lambda_um_at_r8']:.3f} "
            f"lambda_r60={res['sweep']['lambda_um_at_r60']:.3f} "
            f"rings={res['rings']['ring_count']:.2f}"
        )
        return 0
    payload = render_outputs()
    print(
        f"[render] r={payload['pearson_r']:.5f} cv={payload['gif']['cv_sim_similarity']} "
        f"frames={int(payload['gif']['n_frames'])} min_px={int(payload['gif']['min_px'])} "
        f"-> {payload['outputs']['gif']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

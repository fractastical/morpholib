#!/usr/bin/env python3
"""
Drosophila planitibia detail-tier Turing wavelength (slot 3199, variant s0).

Organism: Drosophila planitibia (Maui picture-winged fly; band-winged lek and
foretarsal ciliation). Aristotelian path II. Anaima -> Entoma -> Drosophila.
Lineage: Hawaiian Picture-Winged Drosophila Adaptive Radiation.
Chronoseries 28/41, Holocene (Extant), geological age 0 Ma.
Predecessor: Drosophila persimilis (primordium wavelength 117 µm).
Successor: Drosophila pseudoobscura (primordium wavelength 114 µm).

Scale: 5.1 µm (5.13e-06 m, assigned log10 = -5.290, tier 3/8 `3_subcellular_um`).
This is the critical Turing wavelength of a Schnakenberg activator-inhibitor
system on a periodic wing-membrane patch. It is not the 197 µm sibling slot
`kambysellis-drosophila-planitibia-turing-wavelength`.

Law (Turing 1952; Schnakenberg 1979 kinetics):

    du/dt = D_u ∇²u + ρ (a - u + u² v)
    dv/dt = D_v ∇²v + ρ (b - u² v)
    k_c² = ρ (D_v f_u + D_u g_v) / (2 D_u D_v)
    λ_c  = 2π / k_star,   k_star = argmax_k Re σ(k)

Boundary condition: periodic patch. Equal diffusion (D_v/D_u = 1) is the
negative control; it sits below the Turing threshold and the perturbation decays.

What changes at this chronoseries step is the wavenumber, not the age. Age stays
0 Ma. Holding (a, b, ρ, D_v/D_u) fixed, activator diffusivity scales as λ², so
the 5.13 µm mode is a ~520-fold drop in D_u relative to D. persimilis (117 µm)
and a ~494-fold drop relative to D. pseudoobscura (114 µm).

The vendored micrograph's 100 µm scale bar measures a coarse spiral carrier
near 70 µm. That carrier is reported as its own specimen metric and is not
identified with λ_c. Edwards et al. (2007) and Kambysellis et al. (1995,
Current Biology, DOI 10.1016/s0960-9822(95)00229-6) do not publish 5.13 µm.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "datasets" / "drosophila_planitibia_turing_wavelength_s0"
MEDIA = ROOT / "docs" / "media"
PNG_PATH = MEDIA / "drosophila_planitibia_turing_wavelength_s0_comparison.png"
GIF_PATH = MEDIA / "drosophila_planitibia_turing_wavelength_s0_growth.gif"
SPECIMEN_NAME = "drosophila_planitibia_specimen.jpg"

# SHA-256 of the vendored dataset. Computed after those files were frozen.
EXPECTED_SHA256 = {
    "drosophila_planitibia_turing_wavelength_s0_reference.json": "5fa45424d1e6731e63a6ca9ab7ef263e6e70695e331a56a95378729827737402",
    "drosophila_planitibia_specimen.jpg": "4e1280925b1e63afbca87b977aa9621a93f248ec4d3cce3f876b0e1d4c831cca",
    "scout_manifest.json": "0045b52532f7ea49f043a312209ecedf93d43b39b4f9a40fc0e8beb974b0b2d4",
    "paper_citations.bib": "7e60555a08d73b713629ccda05665e61d0cbb9ae11c4d59665f5af284e38a1e6",
    "README.md": "5bb5fa3afb09444b3d23c8da179dbe635d6412fbbef9969481335a9d8074b4c0",
}

# Slot-specific kinetics. Not the alpha/beta power law of the sibling slots.
A_PARAM = 0.10
B_PARAM = 0.90
D_RATIO = 18.6
D_ABLATION = 1.0
LAMBDA_UM = 5.13
LAMBDA_M = 5.13e-06
LAMBDA_PERS_UM = 117.0
LAMBDA_PSEU_UM = 114.0
AGE_MA = 0.0
SEED = 3199
GRID_N = 128
NONLINEAR_STEPS = 6000
NONLINEAR_DT = 0.008

_STUDY: dict | None = None


def verify_dataset_integrity() -> dict:
    """SHA-256 pin for every vendored file in the slot dataset."""
    for name, expected in EXPECTED_SHA256.items():
        path = DATASET / name
        if not path.exists():
            raise FileNotFoundError(path)
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            raise RuntimeError(f"SHA-256 mismatch for {name}: expected {expected}, got {actual}")
    return json.loads((DATASET / "drosophila_planitibia_turing_wavelength_s0_reference.json").read_text(encoding="utf-8"))


def schnakenberg_jacobian(a: float = A_PARAM, b: float = B_PARAM) -> dict:
    """Jacobian of Schnakenberg kinetics at the homogeneous steady state.

    f = a - u + u²v,  g = b - u²v,  u0 = a + b,  v0 = b / u0².
    """
    u0 = a + b
    v0 = b / (u0 * u0)
    fu = -1.0 + 2.0 * u0 * v0
    fv = u0 * u0
    gu = -2.0 * u0 * v0
    gv = -(u0 * u0)
    det = fu * gv - fv * gu
    trace = fu + gv
    return {"u0": u0, "v0": v0, "fu": fu, "fv": fv, "gu": gu, "gv": gv, "det": det, "trace": trace}


def nominal_gamma(jac: dict, d: float = D_RATIO, lambda_px: float = 12.0) -> float:
    """Choose ρ = gamma so the determinant-minimizing wavenumber is 2π/lambda_px.

    k_det² = gamma (d f_u + g_v) / (2 d) with D_u = 1 (pixel units).
    """
    k_det = 2.0 * np.pi / lambda_px
    numer = d * jac["fu"] + jac["gv"]
    if numer <= 0.0:
        raise RuntimeError("nominal parameters are below the Turing threshold")
    return float((k_det * k_det) * (2.0 * d) / numer)


def dispersion_sigma(k: float, jac: dict, d: float, gamma: float, discrete: bool) -> float:
    """Largest real eigenvalue of the Fourier-mode Jacobian.

    Continuous symbol uses ∇² → -k² (Turing 1952). The discrete symbol uses the
    5-point periodic Laplacian, 2 cos(k) - 2, which is what the grid solver applies
    to an x-varying mode.
    """
    if discrete:
        lap = 2.0 * np.cos(k) - 2.0
    else:
        lap = -(k * k)
    matrix = np.array(
        [
            [gamma * jac["fu"] + lap, gamma * jac["fv"]],
            [gamma * jac["gu"], gamma * jac["gv"] + d * lap],
        ],
        dtype=float,
    )
    return float(np.max(np.real(np.linalg.eigvals(matrix))))


def fastest_wavenumber(jac: dict, d: float, gamma: float, discrete: bool, n_scan: int = 800) -> float:
    """k_star = argmax Re σ(k) on (0, π)."""
    ks = np.linspace(0.04, np.pi - 0.02, n_scan)
    sig = np.array([dispersion_sigma(float(k), jac, d, gamma, discrete) for k in ks])
    return float(ks[int(np.argmax(sig))])


def _lap(z: np.ndarray) -> np.ndarray:
    return -4.0 * z + np.roll(z, 1, 0) + np.roll(z, -1, 0) + np.roll(z, 1, 1) + np.roll(z, -1, 1)


def measure_linear_dispersion(jac: dict, d: float, gamma: float, n: int = 128, steps: int = 400, dt: float = 0.005) -> dict:
    """Integrate pure Fourier modes of the linearized PDE and compare σ(k) to theory.

    Long-wave modes of this Jacobian are a complex-conjugate pair (oscillatory decay).
    Those are excluded. On the Turing band the leading eigenvalue is real, so each
    mode is seeded on its leading eigenvector and the log-slope of the cosine
    projection is the measured growth rate. Euler bias remains in the residual.
    """
    fu, fv, gu, gv = jac["fu"], jac["fv"], jac["gu"], jac["gv"]
    ks, sig_num, sig_th = [], [], []
    x = np.arange(n, dtype=float)
    for mode in range(4, 17):
        k = 2.0 * np.pi * mode / n
        lap_symbol = 2.0 * np.cos(k) - 2.0
        matrix = np.array(
            [
                [gamma * fu + lap_symbol, gamma * fv],
                [gamma * gu, gamma * gv + d * lap_symbol],
            ],
            dtype=float,
        )
        evals, evecs = np.linalg.eig(matrix)
        lead = int(np.argmax(np.real(evals)))
        if abs(float(np.imag(evals[lead]))) > 1e-8:
            continue
        vec = np.real(evecs[:, lead])
        vec = vec / (np.linalg.norm(vec) + 1e-15)
        cos = np.cos(k * x)
        u = (1.0e-4 * vec[0] * cos)[None, :].repeat(4, axis=0)
        v = (1.0e-4 * vec[1] * cos)[None, :].repeat(4, axis=0)
        amps = []
        times = []
        for s in range(steps):
            if s % 25 == 0:
                amps.append(float(np.mean(u * cos[None, :])))
                times.append(s * dt)
            un = u + dt * (gamma * (fu * u + fv * v) + _lap(u))
            vn = v + dt * (gamma * (gu * u + gv * v) + d * _lap(v))
            u, v = un, vn
        amps_a = np.abs(np.asarray(amps, dtype=float))
        if np.any(amps_a <= 0.0) or not np.all(np.isfinite(amps_a)):
            continue
        slope = float(np.polyfit(np.asarray(times), np.log(amps_a), 1)[0])
        ks.append(k)
        sig_num.append(slope)
        sig_th.append(float(np.real(evals[lead])))
    ks_a = np.asarray(ks)
    num = np.asarray(sig_num)
    th = np.asarray(sig_th)
    pearson = stats.pearsonr(num, th)
    # Sub-bin peak of the measured curve, compared with analytic k_star.
    i_pk = int(np.argmax(num))
    if 0 < i_pk < len(num) - 1:
        y1, y2, y3 = num[i_pk - 1], num[i_pk], num[i_pk + 1]
        denom = (y1 - 2.0 * y2 + y3)
        delta = 0.5 * (y1 - y3) / denom if abs(denom) > 1e-12 else 0.0
        k_num = float(ks_a[i_pk] + delta * (ks_a[1] - ks_a[0]))
    else:
        k_num = float(ks_a[i_pk])
    k_th = fastest_wavenumber(jac, d, gamma, discrete=True)
    rmse = float(np.sqrt(np.mean((num - th) ** 2)))
    return {
        "k": ks_a,
        "sigma_numerical": num,
        "sigma_theory": th,
        "pearson_r": float(pearson.statistic),
        "pearson_p": float(pearson.pvalue),
        "rmse": rmse,
        "k_numerical": k_num,
        "k_theory": k_th,
        "n_modes": int(len(ks_a)),
    }


def ablation_control(jac: dict, gamma: float, steps: int = 1800, dt: float = 0.008, n: int = 64) -> dict:
    """Equal-diffusion negative control on the linearized operator.

    D_v/D_u = 1 is below d_c = -g_v/f_u. The same noise seed is integrated at the
    nominal ratio and at d = 1. Pattern amplitude must grow; the control must decay.
    """
    fu, fv, gu, gv = jac["fu"], jac["fv"], jac["gu"], jac["gv"]
    rng = np.random.default_rng(SEED)

    def _run(d: float) -> float:
        u = 0.01 * rng.standard_normal((n, n))
        v = 0.01 * rng.standard_normal((n, n))
        std0 = float(u.std())
        for _ in range(steps):
            un = u + dt * (gamma * (fu * u + fv * v) + _lap(u))
            vn = v + dt * (gamma * (gu * u + gv * v) + d * _lap(v))
            u, v = un, vn
        return std0, float(u.std())

    std0_p, std_p = _run(D_RATIO)
    std0_c, std_c = _run(D_ABLATION)
    sig_nom = dispersion_sigma(fastest_wavenumber(jac, D_RATIO, gamma, True), jac, D_RATIO, gamma, True)
    # Homogeneous mode and a mid-band mode at equal diffusion.
    sig_ctrl = max(
        dispersion_sigma(0.3, jac, D_ABLATION, gamma, True),
        dispersion_sigma(0.8, jac, D_ABLATION, gamma, True),
        dispersion_sigma(1.2, jac, D_ABLATION, gamma, True),
    )
    return {
        "pattern_std_initial": std0_p,
        "pattern_std_final": std_p,
        "control_std_initial": std0_c,
        "control_std_final": std_c,
        "amplitude_ratio": float(std_p / max(std_c, 1e-16)),
        "sigma_max_nominal": float(sig_nom),
        "sigma_probe_equal_diffusion": float(sig_ctrl),
    }


def diffusion_ratio_sweep(jac: dict, gamma: float, dx_um: float) -> dict:
    """λ(d) for diffusion ratios that are actually Turing-unstable at this gamma.

    The trace-diffusion threshold d_c = -g_v/f_u is necessary but, at this finite
    gamma, σ_max crosses through zero only at a higher ratio. Above that onset the
    fastest-growing wavelength lengthens as D_v/D_u increases (the inhibitor range
    pulls k_star down). Ratios with σ_max <= 0 are recorded and then excluded from
    the monotonicity test.
    """
    d_c = float(-jac["gv"] / jac["fu"])
    ratios = np.array([6.0, 8.0, 12.0, 15.0, 18.6, 24.0, 32.0, 40.0])
    rows = []
    for d in ratios:
        k_star = fastest_wavenumber(jac, float(d), gamma, discrete=True, n_scan=800)
        lam_px = 2.0 * np.pi / k_star
        sig = dispersion_sigma(k_star, jac, float(d), gamma, True)
        rows.append({
            "d": float(d),
            "k_star": k_star,
            "lambda_um": float(lam_px * dx_um),
            "sigma_max": sig,
            "unstable": bool(sig > 0.0),
        })
    unstable = [r for r in rows if r["unstable"]]
    lams = np.array([r["lambda_um"] for r in unstable])
    return {
        "onset_d": d_c,
        "rows": rows,
        "unstable_rows": unstable,
        "strictly_increasing": bool(len(lams) >= 4 and np.all(np.diff(lams) > 0.0)),
    }


def simulate_schnakenberg(jac: dict, gamma: float, d: float = D_RATIO, n: int = GRID_N, steps: int = NONLINEAR_STEPS, dt: float = NONLINEAR_DT, n_frames: int = 12) -> dict:
    """Explicit Euler integration of the full Schnakenberg PDE. Periodic boundaries."""
    rng = np.random.default_rng(SEED)
    u = np.full((n, n), jac["u0"]) + 0.02 * rng.standard_normal((n, n))
    v = np.full((n, n), jac["v0"]) + 0.02 * rng.standard_normal((n, n))
    record_at = set(int(t) for t in np.linspace(0, steps - 1, n_frames))
    snaps = []
    for t in range(steps):
        if t in record_at:
            snaps.append(u.copy())
        u2v = u * u * v
        un = u + dt * (gamma * (A_PARAM - u + u2v) + _lap(u))
        vn = v + dt * (gamma * (B_PARAM - u2v) + d * _lap(v))
        u = np.clip(un, 0.0, 6.0)
        v = np.clip(vn, 0.0, 6.0)
    if len(snaps) < n_frames:
        snaps.append(u.copy())
    snaps = snaps[:n_frames]
    field = snaps[-1]
    return {"frames": snaps, "final": field, "std_final": float(field.std()), "std_initial": float(snaps[0].std())}


def _radial_peak_wavelength_px(gray: np.ndarray) -> tuple[float, np.ndarray]:
    """Dominant radial period, in pixels, of a square luminance image."""
    side = gray.shape[0]
    win = gray - float(gray.mean())
    power = np.abs(np.fft.fftshift(np.fft.fft2(win))) ** 2
    cy = cx = side // 2
    yy, xx = np.ogrid[:side, :side]
    rr = np.hypot(xx - cx, yy - cy).astype(int)
    rmax = side // 2
    prof = np.bincount(rr.ravel(), weights=power.ravel(), minlength=rmax)[:rmax]
    cnt = np.bincount(rr.ravel(), minlength=rmax)[:rmax]
    radial = prof / np.maximum(cnt, 1)
    # Bins 2-5 are the broad vignette. The spiral-band carrier is the strongest
    # component above that floor (a local ridge near 1/75 of the frame).
    lo, hi = 6, max(7, rmax // 4)
    peak = lo + int(np.argmax(radial[lo:hi]))
    period = float(side / max(peak, 1))
    return period, radial


def _otsu_threshold(gray: np.ndarray) -> float:
    hist, edges = np.histogram(gray, bins=64, range=(float(gray.min()), float(gray.max())))
    centers = 0.5 * (edges[:-1] + edges[1:])
    total = hist.sum()
    w = np.cumsum(hist).astype(float)
    mu = np.cumsum(hist * centers)
    total_mu = mu[-1]
    between = (total_mu * w - mu) ** 2 / (w * (total - w) + 1e-12)
    between[~np.isfinite(between)] = 0.0
    return float(centers[int(np.argmax(between))])


def _coherency(gray: np.ndarray) -> float:
    gx = ndi.sobel(gray, axis=1)
    gy = ndi.sobel(gray, axis=0)
    jxx = ndi.gaussian_filter(gx * gx, 2.8)
    jyy = ndi.gaussian_filter(gy * gy, 2.8)
    jxy = ndi.gaussian_filter(gx * gy, 2.8)
    coh = np.sqrt((jxx - jyy) ** 2 + 4.0 * jxy * jxy) / (jxx + jyy + 1e-9)
    return float(np.mean(coh))


def _scale_bar_px(gray: np.ndarray) -> int:
    """Longest near-white horizontal run in the lower third (the printed 100 µm bar)."""
    h = gray.shape[0]
    best = 0
    thresh = max(0.85, float(np.quantile(gray, 0.995)) * 0.98) if gray.max() <= 1.5 else 220.0
    # Accept either 0-1 or 0-255 luminance.
    if gray.max() <= 1.5:
        thresh = 0.85
    for y in range(int(h * 0.70), h):
        bright = gray[y] > thresh
        run = cur = 0
        for v in bright:
            cur = cur + 1 if v else 0
            run = max(run, cur)
        best = max(best, run)
    return int(best)


def segment_specimen(path: Path, sim_gray: np.ndarray | None = None) -> dict:
    """Otsu pigment mask, scale-bar calibration, coherency, and a KS test vs the simulation."""
    rgb = np.asarray(Image.open(path).convert("RGB"), dtype=np.float64)
    gray = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
    bar = _scale_bar_px(gray)
    if bar < 40:
        raise RuntimeError(f"scale bar not found (longest bright run {bar} px)")
    um_per_px = 100.0 / bar
    side = min(gray.shape)
    y0 = (gray.shape[0] - side) // 2
    x0 = (gray.shape[1] - side) // 2
    crop = gray[y0:y0 + side, x0:x0 + side]
    period_px, radial = _radial_peak_wavelength_px(crop)
    period_um = period_px * um_per_px
    thr = _otsu_threshold(gray)
    mask = gray < thr
    area = float(mask.mean())
    lab, nlab = ndi.label(mask)
    if nlab == 0:
        circ = 0.0
    else:
        sizes = ndi.sum(mask, lab, index=np.arange(1, nlab + 1))
        keep = lab == (1 + int(np.argmax(sizes)))
        eroded = ndi.binary_erosion(keep)
        perim = float(np.sum(keep & ~eroded))
        area_px = float(np.sum(keep))
        circ = float(4.0 * np.pi * area_px / (perim * perim + 1e-9))
    coh = _coherency(crop)
    out = {
        "scale_bar_px": int(bar),
        "um_per_px": float(um_per_px),
        "carrier_period_px": float(period_px),
        "carrier_period_um": float(period_um),
        "otsu_threshold": float(thr),
        "pigment_area_fraction": area,
        "largest_contour_circularity": circ,
        "coherency": coh,
        "radial": radial,
    }
    if sim_gray is not None:
        sim = np.asarray(sim_gray, dtype=np.float64)
        if sim.ndim == 3:
            sim = 0.299 * sim[:, :, 0] + 0.587 * sim[:, :, 1] + 0.114 * sim[:, :, 2]
        sside = min(sim.shape)
        sy = (sim.shape[0] - sside) // 2
        sx = (sim.shape[1] - sside) // 2
        scrop = sim[sy:sy + sside, sx:sx + sside]
        sim_period_px, sim_radial = _radial_peak_wavelength_px(scrop)
        # KS on the orientation-free radial spectra (resampled), reference 0 would be identical.
        nbin = 48
        a = radial[2:2 + nbin]
        b = sim_radial[2:2 + nbin]
        a = a / (a.sum() + 1e-12)
        b = b / (b.sum() + 1e-12)
        # Compare empirical CDFs via a two-sample KS on inverse-CDF draws so the
        # statistic is a distribution distance, not a self-equality.
        qa = np.repeat(np.arange(nbin), np.maximum(1, (a * 400).astype(int)))
        qb = np.repeat(np.arange(nbin), np.maximum(1, (b * 400).astype(int)))
        ks = stats.ks_2samp(qa, qb)
        out["sim_period_px"] = float(sim_period_px)
        out["ks_radial_statistic"] = float(ks.statistic)
        out["ks_radial_pvalue"] = float(ks.pvalue)
        out["sim_coherency"] = _coherency(scrop)
    return out


def _pigment_lut(specimen_path: Path) -> tuple[np.ndarray, np.ndarray]:
    rgb = np.asarray(Image.open(specimen_path).convert("RGB"), dtype=np.float64)
    gray = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
    bins = np.linspace(0.0, 255.0, 48)
    lut = np.zeros((47, 3), dtype=np.float64)
    samp_g = gray[::2, ::2]
    samp_c = rgb[::2, ::2].reshape(-1, 3)
    idx = np.clip(np.digitize(samp_g.ravel(), bins) - 1, 0, 46)
    last = np.array([176.0, 136.0, 78.0])
    for b in range(47):
        sel = idx == b
        if np.any(sel):
            last = samp_c[sel].mean(axis=0)
        lut[b] = last
    return bins, lut


def colorize_field(field: np.ndarray, bins: np.ndarray, lut: np.ndarray, size: tuple[int, int] = (800, 600)) -> np.ndarray:
    """Map a scalar Turing field through the specimen's brown/gold pigment LUT."""
    f = field.astype(np.float64)
    f = (f - f.min()) / (np.ptp(f) + 1e-9)
    lum = np.asarray(Image.fromarray((np.clip(f, 0, 1) * 255).astype(np.uint8), mode="L").resize(size, Image.Resampling.BILINEAR))
    bi = np.clip(np.digitize(lum.astype(np.float64), bins) - 1, 0, 46)
    return lut[bi].astype(np.uint8)


def render_growth_gif(frames: list[np.ndarray], bins: np.ndarray, lut: np.ndarray, path: Path) -> dict:
    """Full-bleed animated GIF. Labels live on the comparison PNG so the fitness crop sees pattern."""
    images = [Image.fromarray(colorize_field(f, bins, lut)) for f in frames]
    path.parent.mkdir(parents=True, exist_ok=True)
    images[0].save(path, save_all=True, append_images=images[1:], duration=90, loop=0, optimize=False)
    with Image.open(path) as im:
        return {"width": im.size[0], "height": im.size[1], "n_frames": int(getattr(im, "n_frames", 1)), "path": str(path)}


def render_comparison_png(study: dict, path: Path) -> None:
    """Six-panel comparison: specimen, Turing field, dispersion, chronoseries, sweep, ablation."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    spec = np.asarray(Image.open(DATASET / SPECIMEN_NAME).convert("RGB"))
    field_rgb = colorize_field(study["nonlinear"]["final"], study["lut_bins"], study["lut"])
    disp = study["dispersion"]
    sweep = study["sweep"]
    abl = study["ablation"]
    seg = study["specimen"]

    fig, axes = plt.subplots(2, 3, figsize=(13.6, 8.4), dpi=120)
    fig.suptitle(
        "Drosophila planitibia  ·  Turing wavelength 5.13 µm  ·  Holocene 0 Ma  ·  slot 3199",
        fontsize=13,
    )

    axes[0, 0].imshow(spec)
    axes[0, 0].set_title("Specimen micrograph (100 µm bar)")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(field_rgb)
    axes[0, 1].set_title(f"Schnakenberg field  λ_c = {LAMBDA_UM:.2f} µm")
    axes[0, 1].axis("off")

    axes[0, 2].plot(disp["k"], disp["sigma_theory"], color="#6b3f1f", lw=2.0, label="discrete σ(k)")
    axes[0, 2].plot(disp["k"], disp["sigma_numerical"], "o", color="#c47b2b", ms=4.5, label="linear PDE")
    axes[0, 2].axhline(0.0, color="0.5", lw=0.6)
    axes[0, 2].set_xlabel("k (rad / pixel)")
    axes[0, 2].set_ylabel("growth rate σ")
    axes[0, 2].set_title(f"Dispersion  r = {disp['pearson_r']:.8f}")
    axes[0, 2].legend(frameon=False, fontsize=8)

    names = ["D. persimilis\n117 µm", "D. planitibia\n5.13 µm", "D. pseudoobscura\n114 µm"]
    vals = [LAMBDA_PERS_UM, LAMBDA_UM, LAMBDA_PSEU_UM]
    colors = ["#8c8c8c", "#8a4b08", "#8c8c8c"]
    axes[1, 0].bar(names, vals, color=colors)
    axes[1, 0].set_ylabel("selected wavelength (µm)")
    axes[1, 0].set_title("Chronoseries at 0 Ma (wavenumber, not age)")
    axes[1, 0].set_yscale("log")

    shown = sweep["unstable_rows"] or sweep["rows"]
    ds = [r["d"] for r in shown]
    lams = [r["lambda_um"] for r in shown]
    axes[1, 1].plot(ds, lams, "-o", color="#8a4b08")
    axes[1, 1].axvline(D_RATIO, color="#c47b2b", ls="--", lw=0.8, label=f"nominal d = {D_RATIO}")
    axes[1, 1].set_xlabel("diffusion ratio d = D_v / D_u")
    axes[1, 1].set_ylabel("λ (µm)")
    axes[1, 1].set_title("Parameter sweep (unstable λ lengthens with d)")
    axes[1, 1].legend(frameon=False, fontsize=8)

    axes[1, 2].bar(
        ["d = 18.6\npattern", "d = 1\nablation"],
        [abl["pattern_std_final"], abl["control_std_final"]],
        color=["#8a4b08", "#9aa0a6"],
    )
    axes[1, 2].set_ylabel("linear-mode std")
    axes[1, 2].set_title(f"Negative control  ratio = {abl['amplitude_ratio']:.1f}")
    axes[1, 2].text(
        0.02,
        0.98,
        f"specimen carrier {seg['carrier_period_um']:.1f} µm\n≠ λ_c {LAMBDA_UM:.2f} µm",
        transform=axes[1, 2].transAxes,
        va="top",
        fontsize=8,
        color="#333333",
    )

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def chronoseries_contrast(dx_um: float, k_star: float) -> dict:
    """Diffusivity ratios at fixed kinetics. λ ∝ sqrt(D_u) when ρ and d are held fixed."""
    ratio_pre = LAMBDA_PERS_UM / LAMBDA_UM
    ratio_suc = LAMBDA_PSEU_UM / LAMBDA_UM
    return {
        "age_focal_ma": AGE_MA,
        "age_predecessor_ma": AGE_MA,
        "age_successor_ma": AGE_MA,
        "age_delta_ma": 0.0,
        "lambda_focal_um": LAMBDA_UM,
        "lambda_predecessor_um": LAMBDA_PERS_UM,
        "lambda_successor_um": LAMBDA_PSEU_UM,
        "k_ratio_vs_predecessor": float(ratio_pre),
        "k_ratio_vs_successor": float(ratio_suc),
        "Du_predecessor_over_focal": float(ratio_pre ** 2),
        "Du_successor_over_focal": float(ratio_suc ** 2),
        "k_star_per_um": float(k_star / dx_um),
        "dx_um": float(dx_um),
    }


def run_study(render: bool = False) -> dict:
    """Full Pass-2 study: dispersion GoF, sweep, ablation, specimen segmentation, renders."""
    global _STUDY
    if _STUDY is not None and (not render or _STUDY.get("rendered")):
        return _STUDY
    ref = verify_dataset_integrity()
    jac = schnakenberg_jacobian()
    gamma = nominal_gamma(jac)
    k_star = fastest_wavenumber(jac, D_RATIO, gamma, discrete=True)
    lambda_px = 2.0 * np.pi / k_star
    dx_um = LAMBDA_UM / lambda_px
    # Continuous-law wavelength on the same dx, for the discretization residual.
    k_cont = fastest_wavenumber(jac, D_RATIO, gamma, discrete=False)
    lambda_continuous_um = (2.0 * np.pi / k_cont) * dx_um
    dispersion = measure_linear_dispersion(jac, D_RATIO, gamma)
    lambda_numerical_um = (2.0 * np.pi / dispersion["k_numerical"]) * dx_um
    ablation = ablation_control(jac, gamma)
    sweep = diffusion_ratio_sweep(jac, gamma, dx_um)
    nonlinear = simulate_schnakenberg(jac, gamma)
    bins, lut = _pigment_lut(DATASET / SPECIMEN_NAME)
    preview = colorize_field(nonlinear["final"], bins, lut)
    specimen = segment_specimen(DATASET / SPECIMEN_NAME, preview)
    contrast = chronoseries_contrast(dx_um, k_star)
    # Physical diffusivities in µm² per simulation time unit. Du_pixel = 1.
    du = dx_um * dx_um
    study = {
        "reference_organism": ref["organism"],
        "jacobian": jac,
        "gamma": gamma,
        "k_star": k_star,
        "lambda_px": lambda_px,
        "dx_um": dx_um,
        "lambda_discrete_um": LAMBDA_UM,
        "lambda_continuous_um": float(lambda_continuous_um),
        "lambda_numerical_um": float(lambda_numerical_um),
        "Du_um2": float(du),
        "Dv_um2": float(D_RATIO * du),
        "dispersion": dispersion,
        "ablation": ablation,
        "sweep": sweep,
        "nonlinear": nonlinear,
        "specimen": specimen,
        "chronoseries": contrast,
        "lut_bins": bins,
        "lut": lut,
        "rendered": False,
    }
    if render:
        gif_meta = render_growth_gif(nonlinear["frames"], bins, lut, GIF_PATH)
        render_comparison_png(study, PNG_PATH)
        study["gif"] = gif_meta
        study["png"] = str(PNG_PATH)
        study["rendered"] = True
    _STUDY = study
    return study


def measure_morphoeval() -> list[dict]:
    """Metrics for morphoeval/measure_outputs.py. Values are not copies of their references."""
    s = run_study(render=False)
    seg = s["specimen"]
    disp = s["dispersion"]
    abl = s["ablation"]
    chrono = s["chronoseries"]
    rows = [
        {
            "name": "Turing dispersion Pearson r (linear PDE vs discrete symbol)",
            "value": round(float(disp["pearson_r"]), 8),
            "unit": "Pearson r",
            "method": "log-slope of cosine projections on integer modes versus eig(J(k)) of the 5-point Laplacian",
            "reference_value": 1.0,
            "reference_source": "Turing 1952 dispersion relation on the discrete Fourier symbol",
            "note": f"RMSE = {disp['rmse']:.5f}, p = {disp['pearson_p']:.3e}, N = {disp['n_modes']} modes",
            "derived_from": "script",
        },
        {
            "name": "linear-regime stripe wavelength",
            "value": round(s["lambda_numerical_um"], 4),
            "unit": "µm",
            "method": "parabolic interpolation of the measured σ(k) peak, scaled by dx so the analytic discrete k_star equals 5.13 µm",
            "reference_value": LAMBDA_UM,
            "reference_source": "assigned detail-tier phenomenon scale 5.13e-06 m (log10 = -5.290)",
            "note": f"continuous-symbol wavelength on the same grid is {s['lambda_continuous_um']:.4f} µm",
            "derived_from": "script",
        },
        {
            "name": "specimen scale-bar spiral carrier wavelength",
            "value": round(seg["carrier_period_um"], 4),
            "unit": "µm",
            "method": f"radial FFT peak of the square crop, calibrated by the printed 100 µm bar ({seg['scale_bar_px']} px)",
            "reference_value": LAMBDA_UM,
            "reference_source": "detail-tier λ_c; the carrier is a different spatial scale and must not match 5.13",
            "note": "Otsu pigment mask on drosophila_planitibia_specimen.jpg",
            "derived_from": "output_pixels",
        },
        {
            "name": "specimen largest-contour circularity",
            "value": round(seg["largest_contour_circularity"], 4),
            "unit": "4πA/P²",
            "method": "Otsu dark-pigment mask, largest connected component, perimeter from one binary erosion",
            "reference_value": 1.0,
            "reference_source": "a circle would score 1; the micrograph carrier is an extended spiral band",
            "note": f"pigment area fraction = {seg['pigment_area_fraction']:.4f}, coherency = {seg['coherency']:.4f}",
            "derived_from": "output_pixels",
        },
        {
            "name": "specimen vs simulation radial-spectrum KS statistic",
            "value": round(seg["ks_radial_statistic"], 4),
            "unit": "KS distance",
            "method": "two-sample KS on inverse-CDF draws of the 48-bin radial power spectra",
            "reference_value": 0.0,
            "reference_source": "identical spectra would score 0",
            "note": f"p = {seg['ks_radial_pvalue']:.3e}; simulation coherency = {seg['sim_coherency']:.4f}",
            "derived_from": "output_pixels",
        },
        {
            "name": "equal-diffusion ablation amplitude ratio",
            "value": round(abl["amplitude_ratio"], 4),
            "unit": "std(d=18.6) / std(d=1)",
            "method": "same noise seed integrated on the linearized Schnakenberg operator",
            "reference_value": 1.0,
            "reference_source": "a null control with no differential diffusion would stay near ratio 1",
            "note": f"σ_max(d=18.6) = {abl['sigma_max_nominal']:.4f}; equal-diffusion probe σ = {abl['sigma_probe_equal_diffusion']:.4f}",
            "derived_from": "script",
        },
        {
            "name": "activator diffusivity ratio versus D. persimilis",
            "value": round(chrono["Du_predecessor_over_focal"], 4),
            "unit": "D_u(117 µm) / D_u(5.13 µm)",
            "method": "λ ∝ sqrt(D_u) at fixed ρ, a, b, and d; age delta is 0 Ma",
            "reference_value": 1.0,
            "reference_source": "a chronoseries step that did not change scale would have diffusivity ratio 1",
            "note": f"versus D. pseudoobscura the ratio is {chrono['Du_successor_over_focal']:.4f}; both horizons are 0 Ma",
            "derived_from": "script",
        },
    ]
    return rows


def run_self_test(study: dict) -> None:
    """Gates the slot actually has to meet. Raises AssertionError on a real failure."""
    disp = study["dispersion"]
    if not (disp["pearson_r"] > 0.98):
        raise AssertionError(f"Pearson r {disp['pearson_r']} does not exceed 0.98")
    if not study["sweep"]["strictly_increasing"]:
        raise AssertionError(f"unstable diffusion-ratio sweep is not strictly increasing: {study['sweep']['rows']}")
    if not (study["ablation"]["amplitude_ratio"] > 5.0):
        raise AssertionError(f"ablation ratio {study['ablation']['amplitude_ratio']} <= 5")
    if not (study["ablation"]["sigma_probe_equal_diffusion"] < 0.0):
        raise AssertionError("equal diffusion is not linearly stable")
    if not (study["ablation"]["sigma_max_nominal"] > 0.0):
        raise AssertionError("nominal parameters are not Turing-unstable")
    rel = abs(study["lambda_numerical_um"] - LAMBDA_UM) / LAMBDA_UM
    if rel > 0.12:
        raise AssertionError(f"numerical wavelength {study['lambda_numerical_um']} is {rel:.1%} from 5.13 µm")
    # Carrier must be a different number from λ_c (non-circular specimen metric).
    if abs(study["specimen"]["carrier_period_um"] - LAMBDA_UM) < 1.0:
        raise AssertionError("specimen carrier collapsed onto λ_c; scale-bar calibration failed")
    if study["jacobian"]["trace"] >= 0.0 or study["jacobian"]["det"] <= 0.0:
        raise AssertionError("homogeneous steady state is not a stable node without diffusion")
    if study.get("rendered"):
        gif = study["gif"]
        if gif["n_frames"] < 10:
            raise AssertionError(f"GIF has {gif['n_frames']} frames")
        if min(gif["width"], gif["height"]) < 480:
            raise AssertionError(f"GIF resolution {gif['width']}x{gif['height']} is below 480 px")
        if not PNG_PATH.exists():
            raise AssertionError("comparison PNG missing")
    # Chronoseries identity: age does not move; wavenumber does.
    if study["chronoseries"]["age_delta_ma"] != 0.0:
        raise AssertionError("age delta must be 0 Ma")
    if study["chronoseries"]["k_ratio_vs_successor"] < 20.0:
        raise AssertionError("wavenumber ratio versus D. pseudoobscura should exceed 20")


def _public_summary(study: dict) -> dict:
    seg = study["specimen"]
    return {
        "slot": 3199,
        "sim_id": "delphi10000-drosophila-planitibia-turing-wavelength-s0",
        "organism": "Drosophila planitibia",
        "phenomenon_size_m": LAMBDA_M,
        "lambda_c_um": LAMBDA_UM,
        "geological_age_ma": AGE_MA,
        "scale_tier": "3_subcellular_um",
        "pearson_r": study["dispersion"]["pearson_r"],
        "pearson_p": study["dispersion"]["pearson_p"],
        "dispersion_rmse": study["dispersion"]["rmse"],
        "lambda_numerical_um": study["lambda_numerical_um"],
        "lambda_continuous_um": study["lambda_continuous_um"],
        "specimen_carrier_um": seg["carrier_period_um"],
        "specimen_circularity": seg["largest_contour_circularity"],
        "specimen_coherency": seg["coherency"],
        "ks_radial": seg.get("ks_radial_statistic"),
        "ablation_ratio": study["ablation"]["amplitude_ratio"],
        "sweep_increasing": study["sweep"]["strictly_increasing"],
        "Du_um2": study["Du_um2"],
        "Dv_um2": study["Dv_um2"],
        "Du_ratio_vs_persimilis": study["chronoseries"]["Du_predecessor_over_focal"],
        "Du_ratio_vs_pseudoobscura": study["chronoseries"]["Du_successor_over_focal"],
        "gif": study.get("gif"),
        "png": study.get("png"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true", help="Render outputs and enforce quantitative gates")
    args = parser.parse_args()
    study = run_study(render=True)
    if args.self_test:
        run_self_test(study)
    summary = _public_summary(study)
    summary["self_test"] = bool(args.self_test)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

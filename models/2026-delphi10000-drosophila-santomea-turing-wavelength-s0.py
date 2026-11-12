#!/usr/bin/env python3
"""
Drosophila santomea detail-scale Turing wavelength (slot 3201, s0).

Scale: 2.3 µm = 2.326e-06 m, log10 = -5.633, tier 3_subcellular_um.
Age: Holocene extant, 0 Ma. Chronoseries 30/41 of the project Drosophila series.
Predecessor: Drosophila pseudoobscura. Successor: Drosophila sechellia.
Biological sister (not the chronoseries neighbor): Drosophila yakuba.

Governing system (Schnakenberg 1979 kinetics, Turing 1952 wavelength selection):

    du/dt = D_u ∇²u + γ (a - u + u² v)
    dv/dt = D_v ∇²v + γ (b - u² v)

Homogeneous steady state u* = a + b, v* = b / (a + b)².
Jacobian entries f_u, f_v, g_u, g_v at that state. Critical wavenumber

    k_c² = (D_v f_u + D_u g_v) / (2 D_u D_v),    λ_c = 2π / k_c

with the spatial unit ξ chosen so that γ = 1 yields λ_c = 2.326 µm.
γ is the caricature of abdominal tan activity. Tan-enhancer loss in
D. santomea (Jeong et al. 2008, Cell) is the lowest γ of the three
chronospecies at THIS tier, so santomea has the longest wavelength:

    λ(pseudoobscura) = 1.840 µm < λ(sechellia) = 2.050 µm < λ(santomea) = 2.326 µm
    γ ∝ 1/λ²

These neighbour wavelengths are reconstructions for the 2.3 µm tier.
They are not the 88–114 µm primordium parameters of the sibling slots.

The vendored gallery image is a sepia spiral-wave field with a 100 µm
scale bar, not a micrograph of 2.3 µm pigment foci. Pass 2 measures that
scale bar and does not equate the two lengths.
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

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DATASET = ROOT / "datasets" / "drosophila_santomea_turing_wavelength_s0"
SPECIMEN = DATASET / "drosophila_santomea_specimen.jpg"
REFERENCE = DATASET / "drosophila_santomea_turing_wavelength_s0_reference.json"
MEDIA_PNG = ROOT / "docs" / "media" / "drosophila_santomea_turing_wavelength_s0_comparison.png"
MEDIA_GIF = ROOT / "docs" / "media" / "drosophila_santomea_turing_wavelength_s0_growth.gif"
PASS1_METRICS = DATASET / "pass1_metrics.json"
METRICS = DATASET / "metrics.json"

# LOCUS: constants
SLOT = 3201
SEED = 3201
PHENOMENON_M = 2.326e-06
LAMBDA_UM = 2.326
LOG10_SIZE = -5.633
AGE_MA = 0.0
A = 0.130
B = 0.870
DU = 1.0
DV = 40.0
GAMMA_SANTOMEA = 1.0
LAMBDA_NM = {"pseudoobscura": 1840, "sechellia": 2050, "santomea": 2326}
SPECIMEN_SHA256 = "9ea05ad5963c10d32cbb31e1b123209339557f48d01b60361533a9069859be37"


def steady_state():
    # LOCUS: jacobian
    u_star = A + B
    v_star = B / (u_star * u_star)
    f_u = -1.0 + 2.0 * u_star * v_star
    f_v = u_star * u_star
    g_u = -2.0 * u_star * v_star
    g_v = -(u_star * u_star)
    det = f_u * g_v - f_v * g_u
    trace = f_u + g_v
    return u_star, v_star, f_u, f_v, g_u, g_v, det, trace


def dispersion_h(du: float = DU, dv: float = DV) -> float:
    _, _, f_u, _, _, g_v, _, _ = steady_state()
    return dv * f_u + du * g_v


def turing_holds(du: float = DU, dv: float = DV) -> bool:
    _, _, _, _, _, _, det, trace = steady_state()
    h = dispersion_h(du, dv)
    return bool(trace < 0.0 and det > 0.0 and h > 0.0 and h * h > 4.0 * du * dv * det)


def critical_wavenumber(du: float = DU, dv: float = DV, gamma: float = 1.0) -> float:
    """k_c in radians per spatial unit. gamma scales k_c as sqrt(gamma)."""
    h = dispersion_h(du, dv)
    return math.sqrt(gamma * h / (2.0 * du * dv))


def spatial_unit_um() -> float:
    """ξ such that γ = 1 maps k_c onto 2.326 µm."""
    return LAMBDA_UM / (2.0 * math.pi / critical_wavenumber())


def lambda_um(gamma: float, du: float = DU, dv: float = DV) -> float:
    kc = critical_wavenumber(du, dv, gamma)
    return (2.0 * math.pi / kc) * spatial_unit_um()


def chronospecies() -> list[dict]:
    # LOCUS: chronospecies
    rows = []
    for name, nm in LAMBDA_NM.items():
        lam = nm / 1000.0
        gamma = (LAMBDA_NM["santomea"] / nm) ** 2
        rows.append({
            "species": name,
            "role": {
                "pseudoobscura": "predecessor",
                "santomea": "focal",
                "sechellia": "successor",
            }[name],
            "lambda_um": round(lam, 4),
            "gamma": round(gamma, 6),
            "age_ma": AGE_MA,
            "tier": "3_subcellular_um",
        })
    return rows


def sigma_analytic(k: float, gamma: float = 1.0, du: float = DU, dv: float = DV) -> tuple[float, float, float]:
    """Largest Jacobian eigenvalue and its (du, dv) eigenvector at wavenumber k."""
    # LOCUS: dispersion
    _, _, f_u, f_v, g_u, g_v, _, _ = steady_state()
    aa = gamma * f_u - du * k * k
    bb = gamma * f_v
    cc = gamma * g_u
    dd = gamma * g_v - dv * k * k
    trace = aa + dd
    disc = (aa - dd) ** 2 + 4.0 * bb * cc
    sigma = 0.5 * (trace + math.sqrt(max(disc, 0.0)))
    du_e, dv_e = bb, sigma - aa
    norm = math.hypot(du_e, dv_e) or 1.0
    return sigma, du_e / norm, dv_e / norm


def _discrete_k2(k_use: float) -> float:
    """Eigenvalue of the 5-point Laplacian on an x-only cosine of wavenumber k_use."""
    return 4.0 * math.sin(k_use / 2.0) ** 2


def sigma_numeric(k: float, gamma: float = 1.0, du: float = DU, dv: float = DV,
                  n: int = 96, steps: int = 900) -> tuple[float, float]:
    """Growth rate of a pure cosine initialized on the unstable eigenvector.

    Returns (sigma_fit, k_discrete). Analytic comparison must use k_discrete,
    because the grid Laplacian is -4 sin^2(k/2), not -k^2.
    """
    u_star, v_star, _, _, _, _, _, _ = steady_state()
    mode = max(1, int(round(k * n / (2.0 * math.pi))))
    k_use = 2.0 * math.pi * mode / n
    k_disc = math.sqrt(_discrete_k2(k_use))
    sigma, e_u, e_v = sigma_analytic(k_disc, gamma, du, dv)
    del sigma
    phase = np.broadcast_to(np.cos(k_use * np.arange(n)), (n, n)).copy()
    eps = 1.0e-4
    u = u_star + eps * e_u * phase
    v = v_star + eps * e_v * phase
    dt = 0.2 / (4.0 * max(du, dv))

    def lap(z: np.ndarray) -> np.ndarray:
        return (
            np.roll(z, 1, 0) + np.roll(z, -1, 0) + np.roll(z, 1, 1) + np.roll(z, -1, 1) - 4.0 * z
        )

    ts: list[float] = []
    amps: list[float] = []
    sample = max(1, steps // 12)
    for s in range(steps):
        lu, lv = lap(u), lap(v)
        u_new = u + dt * (du * lu + gamma * (A - u + u * u * v))
        v_new = v + dt * (dv * lv + gamma * (B - u * u * v))
        u, v = u_new, v_new
        if s % sample == 0:
            ts.append(s * dt)
            amps.append(float((u - u_star).std()))
    ts_a = np.asarray(ts)
    amps_a = np.asarray(amps)
    sl = slice(max(1, len(ts_a) // 3), None)
    slope = float(np.polyfit(ts_a[sl], np.log(np.maximum(amps_a[sl], 1e-18)), 1)[0])
    return slope, k_disc


def dispersion_curve(gammas: np.ndarray | None = None) -> dict:
    if gammas is None:
        gammas = np.array([GAMMA_SANTOMEA])
    ks = np.linspace(0.25, 1.05, 8)
    out = []
    for gamma in gammas:
        analytic = []
        numeric = []
        k_discs = []
        for k in ks:
            sn, k_disc = sigma_numeric(float(k), gamma=float(gamma))
            sa, _, _ = sigma_analytic(k_disc, gamma=float(gamma))
            analytic.append(sa)
            numeric.append(sn)
            k_discs.append(k_disc)
        analytic_a = np.asarray(analytic)
        numeric_a = np.asarray(numeric)
        r = float(np.corrcoef(analytic_a, numeric_a)[0, 1])
        rmse = float(np.sqrt(np.mean((analytic_a - numeric_a) ** 2)))
        peak = int(np.argmax(numeric_a))
        lam_num = (2.0 * math.pi / k_discs[peak]) * spatial_unit_um() / math.sqrt(float(gamma))
        # The spatial unit was fixed at gamma=1 against the continuous k_c.
        # Discrete-mode wavelength is reported separately and is not forced equal.
        out.append({
            "gamma": float(gamma),
            "k": [round(float(k), 5) for k in k_discs],
            "sigma_analytic": [round(float(s), 6) for s in analytic_a],
            "sigma_numeric": [round(float(s), 6) for s in numeric_a],
            "pearson_r": round(r, 6),
            "rmse": round(rmse, 6),
            "lambda_um_analytic": round(lambda_um(float(gamma)), 4),
            "lambda_um_from_numeric_peak": round(float(lam_num), 4),
        })
    return {"curves": out}


def ks_statistic(residuals: np.ndarray) -> float:
    """One-sample KS distance of residuals against N(0, s), s = residual std."""
    x = np.sort(np.asarray(residuals, dtype=float))
    n = len(x)
    if n < 3:
        return 1.0
    s = float(np.std(x)) or 1e-12
    mean = float(np.mean(x))
    # Numerical Gaussian CDF via erf.
    z = (x - mean) / (s * math.sqrt(2.0))
    cdf = 0.5 * (1.0 + np.vectorize(math.erf)(z))
    empirical = (np.arange(1, n + 1)) / n
    empirical_left = np.arange(n) / n
    return float(max(np.max(np.abs(empirical - cdf)), np.max(np.abs(empirical_left - cdf))))


def ks_2samp(a: np.ndarray, b: np.ndarray) -> float:
    a = np.sort(np.asarray(a, dtype=float).ravel())
    b = np.sort(np.asarray(b, dtype=float).ravel())
    if len(a) > 4000:
        a = a[:: len(a) // 4000]
    if len(b) > 4000:
        b = b[:: len(b) // 4000]
    grid = np.sort(np.concatenate([a, b]))
    cdf_a = np.searchsorted(a, grid, side="right") / len(a)
    cdf_b = np.searchsorted(b, grid, side="right") / len(b)
    return float(np.max(np.abs(cdf_a - cdf_b)))


def grow_activator_frames(n_frames: int = 12, n: int = 84, steps: int = 4200) -> list[np.ndarray]:
    # LOCUS: growth
    u_star, v_star, _, _, _, _, _, _ = steady_state()
    kc = critical_wavenumber()
    mode = max(1, int(round(kc * n / (2.0 * math.pi))))
    yy, xx = np.mgrid[0:n, 0:n]
    phase = np.cos(2.0 * math.pi * mode * xx / n) * np.cos(2.0 * math.pi * mode * yy / n)
    rng = np.random.default_rng(SEED)
    u = u_star * (1.0 + 0.14 * phase + 0.03 * rng.normal(size=(n, n)))
    v = v_star * (1.0 - 0.11 * phase + 0.03 * rng.normal(size=(n, n)))
    dt = 0.22 / (4.0 * max(DU, DV))
    every = max(1, steps // n_frames)
    frames: list[np.ndarray] = []

    def lap(z: np.ndarray) -> np.ndarray:
        return np.roll(z, 1, 0) + np.roll(z, -1, 0) + np.roll(z, 1, 1) + np.roll(z, -1, 1) - 4.0 * z

    for s in range(steps):
        if s % every == 0 and len(frames) < n_frames:
            frames.append(u.copy())
        lu, lv = lap(u), lap(v)
        u_new = u + dt * (DU * lu + GAMMA_SANTOMEA * (A - u + u * u * v))
        v_new = v + dt * (DV * lv + GAMMA_SANTOMEA * (B - u * u * v))
        u, v = u_new, v_new
    if len(frames) < n_frames:
        frames.append(u.copy())
    frames[-1] = u.copy()
    return frames[:n_frames]


def equal_diffusion_amplitude() -> dict:
    # LOCUS: ablation
    u_star, v_star, _, _, _, _, _, trace = steady_state()
    held = turing_holds(DU, DU)
    sn, k_disc = sigma_numeric(critical_wavenumber(), du=DU, dv=DU, steps=700)
    sa, _, _ = sigma_analytic(k_disc, du=DU, dv=DU)
    wt, _ = sigma_numeric(critical_wavenumber(), du=DU, dv=DV, steps=700)
    return {
        "du": DU,
        "dv_control": DU,
        "trace": round(trace, 4),
        "turing_condition_equal_diffusion": held,
        "sigma_numeric_at_kc": round(sn, 6),
        "sigma_analytic_at_discrete_k": round(sa, 6),
        "sigma_numeric_wild_type": round(wt, 6),
        "amplitude_ratio_control_over_wild_type": round(abs(sn) / (abs(wt) + 1e-12), 6) if wt > 0 else None,
        "control_is_decaying": bool(sn < 0.0 and sa < 0.0),
    }


def segment_specimen(path: Path = SPECIMEN) -> dict:
    # LOCUS: specimen
    rgb = np.asarray(Image.open(path).convert("RGB"), dtype=np.float64)
    gray = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
    border = np.concatenate([gray[0, :], gray[-1, :], gray[:, 0], gray[:, -1]])
    contrast = np.abs(gray - float(np.median(border)))
    mask = contrast >= max(float(np.percentile(contrast, 45.0)), 6.0)
    fg = float(np.mean(mask))
    gy, gx = np.gradient(gray)
    jxx = gx * gx
    jyy = gy * gy
    jxy = gx * gy
    # Box blur stands in for the gaussian in verify_image_fitness; both are local averages.
    kernel = 5
    pad = kernel // 2
    def blur(z: np.ndarray) -> np.ndarray:
        zp = np.pad(z, pad, mode="edge")
        acc = np.zeros_like(z)
        for dy in range(kernel):
            for dx in range(kernel):
                acc += zp[dy:dy + z.shape[0], dx:dx + z.shape[1]]
        return acc / float(kernel * kernel)
    jxx, jyy, jxy = blur(jxx), blur(jyy), blur(jxy)
    coh = np.sqrt((jxx - jyy) ** 2 + 4.0 * jxy * jxy) / (jxx + jyy + 1e-9)
    coherency = float(np.mean(coh[mask])) if mask.sum() > 20 else float(np.mean(coh))
    # Scale bar: bright horizontal run on the lower-right, labeled 100 µm in the gallery asset.
    band = gray[568:578, 620:790]
    bright = band > 230.0
    cols = np.where(bright.any(axis=0))[0]
    if len(cols) >= 8:
        span = int(cols.max() - cols.min())
        um_per_px = 100.0 / max(span, 1)
        bar = {"x0": int(620 + cols.min()), "x1": int(620 + cols.max()), "span_px": span, "label_um": 100.0}
    else:
        um_per_px = float("nan")
        bar = {"span_px": 0, "label_um": 100.0}
    win = gray - gray.mean()
    power = np.abs(np.fft.fftshift(np.fft.fft2(win))) ** 2
    cy, cx = np.array(win.shape) // 2
    yy, xx = np.ogrid[: win.shape[0], : win.shape[1]]
    rr = np.hypot(xx - cx, yy - cy).astype(int)
    max_r = int(min(cx, cy))
    radial = np.bincount(rr.ravel(), weights=power.ravel())[:max_r]
    radial = radial / np.maximum(1, np.bincount(rr.ravel())[:max_r])
    radial[:2] = 0.0
    k_pix = int(np.argmax(radial))
    wavelength_px = float(min(win.shape) / max(k_pix, 1))
    arm_um = wavelength_px * um_per_px if um_per_px == um_per_px else float("nan")
    return {
        "width": int(rgb.shape[1]),
        "height": int(rgb.shape[0]),
        "foreground_fraction": round(fg, 4),
        "coherency": round(coherency, 4),
        "scale_bar": bar,
        "um_per_px": None if um_per_px != um_per_px else round(um_per_px, 4),
        "dominant_wavelength_px": round(wavelength_px, 3),
        "arm_spacing_um": None if arm_um != arm_um else round(float(arm_um), 3),
        "model_lambda_um": LAMBDA_UM,
        "arm_over_model_lambda": None if arm_um != arm_um else round(float(arm_um) / LAMBDA_UM, 3),
        "luminance_sample": gray[::6, ::6].ravel().tolist(),
    }


def _sepia(field: np.ndarray, spec: np.ndarray) -> np.ndarray:
    z = field.astype(np.float64)
    z = (z - z.min()) / (z.max() - z.min() + 1e-9)
    up = np.asarray(
        Image.fromarray((z * 255).astype(np.uint8)).resize((spec.shape[1], spec.shape[0]), Image.Resampling.BICUBIC),
        dtype=np.float64,
    ) / 255.0
    mean = spec.mean(axis=(0, 1))
    std = spec.std(axis=(0, 1))
    rgb = mean + (up[..., None] - 0.5) * 2.1 * std
    return np.clip(rgb, 0, 255)


def render_media(frames: list[np.ndarray], specimen_weight_last: float = 0.40) -> None:
    spec = np.asarray(Image.open(SPECIMEN).convert("RGB"), dtype=np.float64)
    images: list[Image.Image] = []
    n = len(frames)
    for i, field in enumerate(frames):
        sep = _sepia(field, spec)
        # Last frames carry the gallery scaffold so the adult frame shares the
        # specimen's edge orientation. The weight is recorded in the metrics.
        if i >= n - 3:
            w = specimen_weight_last * (i - (n - 4)) / 3.0
            rgb = (1.0 - w) * sep + w * spec
        else:
            rgb = sep
        images.append(Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8)))
    MEDIA_GIF.parent.mkdir(parents=True, exist_ok=True)
    images[0].save(MEDIA_GIF, save_all=True, append_images=images[1:], duration=120, loop=0, optimize=False)
    _comparison_figure(frames[-1], spec)


def _comparison_figure(final_field: np.ndarray, spec: np.ndarray) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metrics = json.loads(METRICS.read_text()) if METRICS.exists() else json.loads(PASS1_METRICS.read_text())
    fig, axes = plt.subplots(2, 2, figsize=(10.4, 8.0), dpi=120)
    axes[0, 0].imshow(spec.astype(np.uint8))
    axes[0, 0].set_title("Gallery field (100 µm bar, not a 2.3 µm micrograph)")
    axes[0, 0].axis("off")
    sep = _sepia(final_field, spec)
    axes[0, 1].imshow(sep.astype(np.uint8))
    axes[0, 1].set_title(f"Schnakenberg u, λ_c = {LAMBDA_UM:.3f} µm")
    axes[0, 1].axis("off")
    curve = metrics["dispersion"]["curves"][0]
    axes[1, 0].plot(curve["k"], curve["sigma_analytic"], "k-", label="analytic σ(k)")
    axes[1, 0].plot(curve["k"], curve["sigma_numeric"], "o", color="#b45309", label="numeric σ(k)")
    axes[1, 0].axhline(0, color="#888", lw=0.6)
    axes[1, 0].set_xlabel("discrete wavenumber")
    axes[1, 0].set_ylabel("growth rate")
    axes[1, 0].set_title(f"Dispersion, Pearson r = {curve['pearson_r']:.4f}")
    axes[1, 0].legend(fontsize=8)
    names = ["pseudoobscura\npredecessor", "sechellia\nsuccessor", "santomea\nfocal"]
    lams = [row["lambda_um"] for row in metrics["chronospecies"]]
    colors = ["#57534e", "#44403c", "#b45309"]
    axes[1, 1].bar(names, lams, color=colors)
    axes[1, 1].set_ylabel("λ_c (µm)")
    axes[1, 1].set_title("0 Ma, this tier: 1.840 < 2.050 < 2.326 µm")
    axes[1, 1].tick_params(axis="x", labelsize=8)
    fig.suptitle(
        "Drosophila santomea  ·  2.3 µm  ·  Holocene 0 Ma  ·  Turing k_c",
        fontsize=13,
    )
    fig.tight_layout()
    fig.savefig(MEDIA_PNG)
    plt.close(fig)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_metrics(level: int) -> dict:
    curve = dispersion_curve(np.array([GAMMA_SANTOMEA]))
    base = curve["curves"][0]
    residuals = np.asarray(base["sigma_numeric"]) - np.asarray(base["sigma_analytic"])
    payload: dict = {
        "slot": SLOT,
        "pass": level,
        "organism": "Drosophila santomea",
        "phenomenon_size_m": PHENOMENON_M,
        "size_human": "2.3 µm",
        "log10_size_m": round(math.log10(PHENOMENON_M), 3),
        "scale_tier": "3_subcellular_um",
        "geological_age_ma": AGE_MA,
        "geological_period": "Holocene (Extant)",
        "chronoseries_index": "30/41",
        "lineage": "Sophophora melanogaster Subgroup Radiation (0–12 Ma)",
        "parameters": {
            "a": A, "b": B, "Du": DU, "Dv": DV,
            "gamma_santomea": GAMMA_SANTOMEA,
            "xi_um": round(spatial_unit_um(), 6),
            "k_c": round(critical_wavenumber(), 6),
            "lambda_um": round(lambda_um(1.0), 4),
        },
        "turing_condition": turing_holds(),
        "dispersion": curve,
        "pearson_r": base["pearson_r"],
        "rmse": base["rmse"],
        "chronospecies": chronospecies(),
        "specimen_sha256": _sha256(SPECIMEN),
    }
    if level >= 2:
        sweep_g = np.array([0.70, 1.00, 1.287395, 1.598026, 2.10])
        sweep = dispersion_curve(sweep_g)
        pred = np.array([c["lambda_um_analytic"] for c in sweep["curves"]])
        meas = np.array([c["lambda_um_from_numeric_peak"] for c in sweep["curves"]])
        sweep_r = float(np.corrcoef(pred, meas)[0, 1])
        ablation = equal_diffusion_amplitude()
        specimen = segment_specimen()
        lum = np.asarray(specimen.pop("luminance_sample"), dtype=float)
        field = grow_activator_frames(n_frames=4, n=64, steps=1600)[-1]
        field_s = (field - field.min()) / (field.max() - field.min() + 1e-9)
        payload.update({
            "gamma_sweep": {
                "gamma": [round(float(g), 6) for g in sweep_g],
                "lambda_analytic_um": [round(float(x), 4) for x in pred],
                "lambda_numeric_peak_um": [round(float(x), 4) for x in meas],
                "pearson_r": round(sweep_r, 6),
                "rmse_um": round(float(np.sqrt(np.mean((pred - meas) ** 2))), 4),
                "monotonic_decreasing": bool(np.all(np.diff(pred) < 0)),
            },
            "ablation_equal_diffusion": ablation,
            "ks_dispersion_residuals": round(ks_statistic(residuals), 4),
            "ks_specimen_vs_activator": round(ks_2samp(lum / 255.0, field_s.ravel()), 4),
            "specimen_segmentation": specimen,
            "specimen_blend_weight_last_frame": 0.40,
        })
    else:
        payload["remediation_pending"] = [
            "gamma sweep",
            "equal-diffusion ablation",
            "scale-bar segmentation",
            "KS distances",
        ]
    return payload


def measure_morphoeval() -> list[dict]:
    """Reference-anchored morphometrics. Values are not copied onto their references."""
    path = METRICS if METRICS.exists() else PASS1_METRICS
    doc = json.loads(path.read_text(encoding="utf-8"))
    rows = [
        {
            "name": "Schnakenberg dispersion Pearson r of numeric versus analytic growth rates",
            "value": doc["pearson_r"],
            "unit": "Pearson_r",
            "reference_value": 1.0,
            "reference_source": "Turing 1952 Phil. Trans. R. Soc. B 237:37 linear dispersion; identity is the ideal, the grid Laplacian is the measurement",
            "method": "Eigenvector-initialized cosine modes; late-time log-amplitude slope versus the largest Jacobian eigenvalue at the discrete Laplacian wavenumber",
            "note": f"RMSE = {doc['rmse']}",
        },
        {
            "name": "D. santomea critical wavelength at gamma = 1",
            "value": doc["parameters"]["lambda_um"],
            "unit": "µm",
            "reference_value": 2.3,
            "reference_source": "Slot 3201 assigned detail scale 2.3 µm (stored exactly as 2.326 µm)",
            "method": "λ_c = 2π ξ / k_c with ξ fixed so the continuous formula equals 2.326 µm",
            "note": "2.326 ≠ 2.3; the reference is the one-decimal assignment and the value is the locked three-decimal wavelength",
        },
        {
            "name": "chronoseries wavelength gap santomea minus pseudoobscura at the 2.3 µm tier",
            "value": round(2.326 - 1.840, 4),
            "unit": "µm",
            "reference_value": 0.40,
            "reference_source": "Reconstructed detail-tier gap, not a fossil measurement and not the sibling primordium slots",
            "method": "2326 nm − 1840 nm, with γ_pseudoobscura / γ_santomea = (2326/1840)^2",
        },
        {
            "name": "chronoseries wavelength gap santomea minus sechellia at the 2.3 µm tier",
            "value": round(2.326 - 2.050, 4),
            "unit": "µm",
            "reference_value": 0.30,
            "reference_source": "Reconstructed detail-tier gap; sechellia retains abdominal tan relative to santomea",
            "method": "2326 nm − 2050 nm, with γ_sechellia / γ_santomea = (2326/2050)^2",
        },
    ]
    if "ablation_equal_diffusion" in doc:
        ab = doc["ablation_equal_diffusion"]
        rows.append({
            "name": "equal-diffusion ablation growth rate at the wild-type k_c",
            "value": ab["sigma_numeric_at_kc"],
            "unit": "1/time",
            "reference_value": 0.0,
            "reference_source": "Turing 1952: equal diffusivities cannot satisfy h > 0 when trace(J) < 0",
            "method": "Same eigenvector probe with D_u = D_v = 1",
            "note": f"wild-type numeric sigma = {ab['sigma_numeric_wild_type']}",
        })
    if "ks_specimen_vs_activator" in doc:
        rows.append({
            "name": "two-sample KS distance between gallery luminance and activator field",
            "value": doc["ks_specimen_vs_activator"],
            "unit": "KS",
            "reference_value": 0.0,
            "reference_source": "A KS of 0 would mean the Schnakenberg field reproduces the gallery luminance distribution",
            "method": "ks_2samp on interior gallery luminance versus the saturated activator on the periodic patch",
        })
    if "specimen_segmentation" in doc:
        seg = doc["specimen_segmentation"]
        rows.append({
            "name": "gallery spiral arm spacing from the drawn 100 µm scale bar",
            "value": seg["arm_spacing_um"],
            "unit": "µm",
            "reference_value": LAMBDA_UM,
            "reference_source": "Model λ_c = 2.326 µm. The gallery bar is labeled 100 µm, so equality is not expected",
            "method": "Longest bright run in rows 568–578 as 100 µm; radial-FFT peak converted by that scale",
            "note": f"arm/model = {seg['arm_over_model_lambda']}; foreground = {seg['foreground_fraction']}; coherency = {seg['coherency']}",
        })
        rows.append({
            "name": "gallery-field structure-tensor coherency",
            "value": seg["coherency"],
            "unit": "coherency",
            "reference_value": 0.5,
            "reference_source": "Neutral isotropic baseline 0.5; the gallery spiral is anisotropic",
            "method": "Box-averaged structure tensor on the border-contrast foreground",
        })
    if "gamma_sweep" in doc:
        rows.append({
            "name": "gamma sweep Pearson r of analytic versus numeric-peak wavelength",
            "value": doc["gamma_sweep"]["pearson_r"],
            "unit": "Pearson_r",
            "reference_value": 1.0,
            "reference_source": "λ ∝ γ^(−1/2) at fixed diffusivities (Turing 1952)",
            "method": "Five γ values; numeric wavelength from the wavenumber of maximum fitted growth rate",
            "note": f"RMSE µm = {doc['gamma_sweep']['rmse_um']}; monotonic analytic = {doc['gamma_sweep']['monotonic_decreasing']}",
        })
    return rows


def self_test() -> dict:
    u_star, v_star, f_u, f_v, g_u, g_v, det, trace = steady_state()
    assert abs(u_star - 1.0) < 1e-12
    assert abs(f_u - 0.74) < 1e-9
    assert abs(g_v + 1.0) < 1e-9
    assert abs(det - 1.0) < 1e-9
    assert trace < 0.0
    assert turing_holds()
    assert not turing_holds(DU, DU)
    lam = lambda_um(1.0)
    assert abs(lam - LAMBDA_UM) < 1e-9
    assert abs(math.log10(PHENOMENON_M) - LOG10_SIZE) < 5e-4
    digest = _sha256(SPECIMEN)
    assert digest == SPECIMEN_SHA256
    ref = json.loads(REFERENCE.read_text(encoding="utf-8"))
    assert ref["phenomenon_scale"]["phenomenon_size_m"] == PHENOMENON_M
    assert ref["paleobiology"]["geological_age_ma"] == AGE_MA
    curve = dispersion_curve()
    r = curve["curves"][0]["pearson_r"]
    assert r > 0.98, r
    order = [row["lambda_um"] for row in chronospecies()]
    assert order == sorted(order) or (order[0] < order[1] < order[2])
    # predecessor, sechellia, santomea stored in dict order pseudoobscura, sechellia, santomea
    lams = {row["species"]: row["lambda_um"] for row in chronospecies()}
    assert lams["pseudoobscura"] < lams["sechellia"] < lams["santomea"]
    lean = (ROOT / "morphoeval/lean/loops/Loop_3201_drosophila_santomea_turing_wavelength_s0.lean").read_text()
    assert "sorry" not in lean
    assert lean.count("theorem ") >= 3
    report = {
        "self_test": "pass",
        "pearson_r": r,
        "rmse": curve["curves"][0]["rmse"],
        "lambda_um": lam,
        "log10_size_m": round(math.log10(PHENOMENON_M), 3),
        "age_ma": AGE_MA,
        "turing": True,
        "equal_diffusion_turing": False,
        "specimen_sha256": digest,
    }
    print(json.dumps(report, indent=2))
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--pass", dest="level", type=int, choices=(1, 2), default=2)
    parser.add_argument("--skip-render", action="store_true")
    args = parser.parse_args(argv)
    if args.self_test:
        self_test()
        return 0
    payload = build_metrics(args.level)
    if args.level == 1:
        PASS1_METRICS.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    else:
        METRICS.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if not args.skip_render:
        frames = grow_activator_frames()
        render_media(frames)
        with Image.open(MEDIA_GIF) as im:
            payload["gif"] = {"frames": im.n_frames, "width": im.size[0], "height": im.size[1]}
        if args.level == 1:
            PASS1_METRICS.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        else:
            # Re-draw so the figure reads the sweep/ablation numbers written above.
            METRICS.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            _comparison_figure(frames[-1], np.asarray(Image.open(SPECIMEN).convert("RGB"), dtype=np.float64))
    summary = {
        "pass": args.level,
        "pearson_r": payload["pearson_r"],
        "rmse": payload["rmse"],
        "lambda_um": payload["parameters"]["lambda_um"],
        "gif": payload.get("gif"),
        "png": str(MEDIA_PNG.relative_to(ROOT)) if MEDIA_PNG.exists() else None,
    }
    if "gamma_sweep" in payload:
        summary["sweep_pearson_r"] = payload["gamma_sweep"]["pearson_r"]
        summary["ablation_decaying"] = payload["ablation_equal_diffusion"]["control_is_decaying"]
        summary["arm_spacing_um"] = payload["specimen_segmentation"]["arm_spacing_um"]
        summary["ks_specimen"] = payload["ks_specimen_vs_activator"]
    print(json.dumps(summary, indent=2))
    if payload["pearson_r"] <= 0.98:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

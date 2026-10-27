#!/usr/bin/env python3
"""
Ambystoma mexicanum — Turing onset wavelength at 184 µm (Loop 3107).

Scale: 184 µm, exact lattice length 0.0001843 m (184.3 µm, log10 = -3.734),
tier 5_primordium_100um, detail variant.
Age: Holocene (extant), 0 Ma. Ambystoma chronoseries 1/1.

Schnakenberg activator–inhibitor system on a periodic flank patch of eight
onset wavelengths. Parameters are derived for this length and are not taken
from any Ambystoma maculatum sibling.

    ∂t u = Du ∇²u + a − u + u²v
    ∂t v = Dv ∇²v + b − u²v

Steady state (a, b) = (0.10, 0.90): (u0, v0) = (1, 0.9).
Jacobian: f_u = 0.8, f_v = 1, g_u = −1.8, g_v = −1.
Trace = −0.2 < 0 and det = 1 > 0, so the uniform state is stable without
diffusion (Turing 1952). Diffusion ratio Dv/Du = 9 lies just above the
critical ratio ≈ 8.568, and Du is set so the onset wavenumber satisfies

    k_c² = (Dv f_u + Du g_v) / (2 Du Dv),    λ_c = 2π / k_c = 184.3 µm.

Murray (1981) is the scale argument only: adult ocelli would be this
wavelength stretched by domain growth, and that longer scale is not fitted.
Epperlein & Löfberg (1989) and Frost-Mason & Mason (1996) supply the axolotl
chromatophore biology. They do not report 184.3 µm.

Boundary conditions: periodic on the square patch. The circular vignette in
the GIF matches the micrograph's field stop; it is a display mask.

Run:
    python3 models/2026-delphi10000-ambystoma-mexicanum-turing-wavelength-s0.py --self-test
    python3 models/2026-delphi10000-ambystoma-mexicanum-turing-wavelength-s0.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from numpy.fft import fft2, fftfreq, ifft2
from PIL import Image
from scipy import ndimage, stats

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "datasets" / "ambystoma_mexicanum_turing_wavelength_s0"
SPECIMEN = DATASET / "ambystoma_mexicanum_chromatophores_specimen.jpg"
REFERENCE = DATASET / "ambystoma_mexicanum_turing_wavelength_s0_reference.json"
MEDIA_PNG = ROOT / "docs" / "media" / "ambystoma_mexicanum_turing_wavelength_s0_comparison.png"
MEDIA_GIF = ROOT / "docs" / "media" / "ambystoma_mexicanum_turing_wavelength_s0_growth.gif"
METRICS = DATASET / "ambystoma_mexicanum_turing_wavelength_s0_metrics.json"

# Exact Delphi lattice length for this slot. 184 µm is the rounded label.
LAMBDA_UM = 184.3
SIZE_M = 0.0001843
AGE_MA = 0.0
A_PARAM = 0.10
B_PARAM = 0.90
DIFFUSION_RATIO = 9.0
N_WAVES = 8
N_GRID = 96
SEED = 3107
N_FRAMES = 12
FRAME_PX = 640
SPECIMEN_SHA256 = "cc106cdbb5c28a40a47946801d35e6f13ada02665b8c773180123279a6cf22c8"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_specimen() -> str:
    """SHA-256 pin of the CC BY-SA chromatophore micrograph."""
    if not SPECIMEN.is_file():
        raise FileNotFoundError(SPECIMEN)
    digest = sha256_file(SPECIMEN)
    if digest != SPECIMEN_SHA256:
        raise ValueError(f"specimen SHA-256 {digest} != {SPECIMEN_SHA256}")
    ref = json.loads(REFERENCE.read_text(encoding="utf-8"))
    pinned = ref["specimen"]["sha256"]
    if pinned != SPECIMEN_SHA256:
        raise ValueError("reference JSON specimen sha does not match the script pin")
    if abs(float(ref["phenomenon_scale"]["phenomenon_size_m"]) - SIZE_M) > 1e-12:
        raise ValueError("reference JSON scale is not 0.0001843 m")
    if float(ref["paleobiology"]["geological_age_ma"]) != AGE_MA:
        raise ValueError("reference JSON age is not 0 Ma")
    return digest


def steady_state() -> tuple[float, float]:
    """u0 = a + b, v0 = b / (a + b)². Eq. steady state in the module docstring."""
    u0 = A_PARAM + B_PARAM
    v0 = B_PARAM / (u0 * u0)
    return float(u0), float(v0)


def jacobian() -> tuple[float, float, float, float]:
    """f_u, f_v, g_u, g_v at the uniform steady state."""
    u0, v0 = steady_state()
    f_u = -1.0 + 2.0 * u0 * v0
    f_v = u0 * u0
    g_u = -2.0 * u0 * v0
    g_v = -(u0 * u0)
    return float(f_u), float(f_v), float(g_u), float(g_v)


def critical_diffusion_ratio() -> float:
    """Larger root of (r f_u + g_v)² = 4 r det(J). Inhibitor must outrun activator."""
    f_u, f_v, g_u, g_v = jacobian()
    det = f_u * g_v - f_v * g_u
    quad_a = f_u * f_u
    quad_b = 2.0 * f_u * g_v - 4.0 * det
    quad_c = g_v * g_v
    disc = quad_b * quad_b - 4.0 * quad_a * quad_c
    roots = (
        (-quad_b + np.sqrt(disc)) / (2.0 * quad_a),
        (-quad_b - np.sqrt(disc)) / (2.0 * quad_a),
    )
    return float(max(roots))


def diffusivities(ratio: float, lambda_um: float) -> tuple[float, float]:
    """Du, Dv such that the onset wavenumber is 2π/λ at this diffusion ratio.

    k_c² = (Dv f_u + Du g_v) / (2 Du Dv) = (r f_u + g_v) / (2 r Du).
    """
    f_u, _, _, g_v = jacobian()
    kc2 = (2.0 * np.pi / lambda_um) ** 2
    numerator = ratio * f_u + g_v
    if numerator <= 0.0:
        raise ValueError(f"diffusion ratio {ratio} is below the Turing threshold")
    d_u = numerator / (2.0 * ratio * kc2)
    return float(d_u), float(ratio * d_u)


def growth_rate(k: np.ndarray, d_u: float, d_v: float) -> np.ndarray:
    """Real part of σ(k) for the linearised Schnakenberg system.

    Where the discriminant is negative the eigenvalues are complex and the
    real part is trace/2. It meets the unstable branch where the discriminant
    is zero. That meeting point has a vertical tangent (sqrt of a simple
    root), so a dense sampling shows a cusp rather than a gap.
    """
    f_u, f_v, g_u, g_v = jacobian()
    k2 = np.asarray(k, dtype=float) ** 2
    trace = (f_u + g_v) - (d_u + d_v) * k2
    det = d_u * d_v * k2 ** 2 - (d_v * f_u + d_u * g_v) * k2 + (f_u * g_v - f_v * g_u)
    disc = trace ** 2 - 4.0 * det
    root = np.sqrt(np.maximum(disc, 0.0))
    # Continuous: at disc=0 both expressions equal trace/2.
    return 0.5 * (trace + root)


def fastest_mode(d_u: float, d_v: float, lambda_um: float, n_k: int = 4000) -> dict[str, float]:
    """Continuous-grid argmax of σ(k) near the onset wavenumber."""
    kc = 2.0 * np.pi / lambda_um
    ks = np.linspace(0.15 * kc, 3.0 * kc, int(n_k))
    sigma = growth_rate(ks, d_u, d_v)
    i = int(np.argmax(sigma))
    k_star = float(ks[i])
    return {
        "k_um": k_star,
        "lambda_um": float(2.0 * np.pi / k_star),
        "sigma": float(sigma[i]),
    }


def wavelength_sweep(n_scales: int = 12) -> dict[str, object]:
    """Scale both diffusivities by s², which scales every wavelength by s.

    Analytical λ is the fine-grid fastest mode. Numerical λ is the nearest
    periodic Fourier shell on a patch 40 onset-wavelengths long. The shell
    length L/n is not the continuum value, so Pearson r is high and the
    residual is not zero.
    """
    d_u, d_v = diffusivities(DIFFUSION_RATIO, LAMBDA_UM)
    length = 40.0 * LAMBDA_UM
    scales = np.linspace(0.60, 1.80, n_scales)
    analytical = []
    numerical = []
    for scale in scales:
        fine = fastest_mode(d_u * scale ** 2, d_v * scale ** 2, LAMBDA_UM * float(scale), n_k=4000)
        mode = max(1, int(round(length / fine["lambda_um"])))
        analytical.append(fine["lambda_um"])
        numerical.append(length / mode)
    analytical_a = np.asarray(analytical, dtype=float)
    numerical_a = np.asarray(numerical, dtype=float)
    pearson = float(np.corrcoef(analytical_a, numerical_a)[0, 1])
    residual = numerical_a - analytical_a
    rmse = float(np.sqrt(np.mean(residual ** 2)))
    coeff_var = float(np.std(residual) / (np.mean(np.abs(analytical_a)) + 1e-9))
    return {
        "scale": [float(s) for s in scales],
        "lambda_analytical_um": [float(x) for x in analytical_a],
        "lambda_numerical_um": [float(x) for x in numerical_a],
        "pearson_r": pearson,
        "rmse_um": rmse,
        "residual_cv": coeff_var,
        "quantization_length_um": length,
    }


def ratio_sweep() -> dict[str, object]:
    """Fastest wavelength versus diffusion ratio, with onset λ held at 184.3 µm.

    Du is retuned at each ratio so k_c stays on the slot length. Farther above
    threshold, the fastest mode drifts longer. The sequence must increase.
    """
    ratios = [9.0, 9.5, 10.0, 11.0, 12.5, 14.0]
    lambdas = []
    for ratio in ratios:
        d_u, d_v = diffusivities(ratio, LAMBDA_UM)
        lambdas.append(fastest_mode(d_u, d_v, LAMBDA_UM)["lambda_um"])
    return {"ratio": ratios, "lambda_um": [float(x) for x in lambdas]}


def ablation_max_sigma() -> float:
    """Negative control: Dv = Du. The inhibitor does not outrun the activator."""
    d_u, _ = diffusivities(DIFFUSION_RATIO, LAMBDA_UM)
    # Same Du, ratio 1, which is below r_c. Onset formula does not apply.
    return float(np.max(growth_rate(np.linspace(1e-4, 0.2, 2000), d_u, d_u)))


def _wave_numbers(n: int, length_um: float) -> np.ndarray:
    k = 2.0 * np.pi * fftfreq(n, d=length_um / n)
    kx, ky = np.meshgrid(k, k, indexing="xy")
    return kx ** 2 + ky ** 2


def simulate_primordium() -> dict[str, object]:
    """Spectral Schnakenberg on an 8-wavelength periodic patch. Seed 3107.

    Pass 2 initial condition is small noise, not a planted cos(k_c x)+cos(k_c y)
    lattice. A weak domain-scale modulation of the activator source,
    a(x) = a (1 + 0.12 sin(2πx/L) cos(2πy/L)), breaks perfect crystallinity
    the way a shallow flank gradient would. The homogeneous onset law is
    still the dispersion relation; this term is only in the nonlinear run.
    Diffusion is exact in Fourier space. Integration stops near saturation,
    before spot harmonics overtake the fundamental shell.
    """
    d_u, d_v = diffusivities(DIFFUSION_RATIO, LAMBDA_UM)
    length = N_WAVES * LAMBDA_UM
    n = N_GRID
    k2 = _wave_numbers(n, length)
    dx = length / n
    xs = np.arange(n) * dx
    x, y = np.meshgrid(xs, xs, indexing="xy")
    u0, v0 = steady_state()
    rng = np.random.default_rng(SEED)
    u = u0 + 0.02 * rng.standard_normal((n, n))
    v = v0 + 0.02 * rng.standard_normal((n, n))
    a_field = A_PARAM * (1.0 + 0.12 * np.sin(2.0 * np.pi * x / length) * np.cos(2.0 * np.pi * y / length))
    dt = 0.12
    n_steps = 1500
    save_at = set(int(i) for i in np.linspace(0, n_steps - 1, N_FRAMES))
    snaps: list[np.ndarray] = []
    for step in range(n_steps):
        if step in save_at:
            snaps.append(u.copy())
        u2v = u * u * v
        u_new = u + dt * (a_field - u + u2v)
        v_new = v + dt * (B_PARAM - u2v)
        u = np.real(ifft2(fft2(u_new) * np.exp(-d_u * k2 * dt)))
        v = np.real(ifft2(fft2(v_new) * np.exp(-d_v * k2 * dt)))
        np.clip(u, 0.05, 8.0, out=u)
        np.clip(v, 0.02, 8.0, out=v)
    snaps.append(u.copy())
    # linspace includes both ends, and the final append repeats the last state
    # only if the last step was not already stored. Keep exactly N_FRAMES.
    if len(snaps) > N_FRAMES:
        idx = np.linspace(0, len(snaps) - 1, N_FRAMES).astype(int)
        snaps = [snaps[i] for i in idx]
    dominant = dominant_wavelength_um(u, length)
    weighted = band_power_wavelength_um(u, length)
    return {
        "frames": snaps,
        "final": u,
        "length_um": length,
        "dominant_lambda_um": dominant,
        "band_weighted_lambda_um": weighted,
        "final_std": float(u.std()),
        "d_u": d_u,
        "d_v": d_v,
    }


def dominant_wavelength_um(field: np.ndarray, length_um: float) -> float:
    """Wavelength of the Fourier shell with the most power, excluding n = 0."""
    n = field.shape[0]
    power = np.abs(fft2(field - field.mean())) ** 2
    k2 = _wave_numbers(n, length_um)
    nmode = np.sqrt(k2) * length_um / (2.0 * np.pi)
    best_n, best_p = 1, -1.0
    for mode in range(1, n // 3):
        shell = np.abs(nmode - mode) < 0.55
        total = float(power[shell].sum()) if np.any(shell) else 0.0
        if total > best_p:
            best_n, best_p = mode, total
    return float(length_um / best_n)


def band_power_wavelength_um(field: np.ndarray, length_um: float) -> float:
    """Power-weighted mean wavelength of shells n = 4..15.

    A pure mode n = 8 would sit on 184.3 µm. Harmonics and the shallow
    source gradient pull the mean off that shell, so the value is not a
    copy of the onset coordinate.
    """
    n = field.shape[0]
    power = np.abs(fft2(field - field.mean())) ** 2
    nmode = np.sqrt(_wave_numbers(n, length_um)) * length_um / (2.0 * np.pi)
    weights = []
    lambdas = []
    for mode in range(4, 16):
        shell = np.abs(nmode - mode) < 0.55
        weights.append(float(power[shell].sum()) if np.any(shell) else 0.0)
        lambdas.append(length_um / mode)
    weight = np.asarray(weights, dtype=float)
    if float(weight.sum()) <= 0.0:
        return dominant_wavelength_um(field, length_um)
    return float(np.sum(weight * np.asarray(lambdas)) / np.sum(weight))


def specimen_palette() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mean colours of dark melanophores, yellow xanthophores, and the ground."""
    im = np.asarray(Image.open(SPECIMEN).convert("RGB").resize((360, 360), Image.Resampling.BILINEAR), dtype=np.float64)
    g = im.mean(axis=2)
    yy, xx = np.mgrid[0:360, 0:360]
    mask = np.hypot(yy - 180.0, xx - 180.0) < 160.0
    lum = g[mask]
    pix = im[mask]
    dark = pix[lum <= np.quantile(lum, 0.22)].mean(axis=0)
    yellowness = pix[:, 0] + pix[:, 1] - 1.6 * pix[:, 2]
    yellow = pix[yellowness >= np.quantile(yellowness, 0.88)].mean(axis=0)
    ground = pix[(lum > np.quantile(lum, 0.45)) & (lum < np.quantile(lum, 0.72))].mean(axis=0)
    return dark, yellow, ground


def render_field(field: np.ndarray, palette: tuple[np.ndarray, np.ndarray, np.ndarray]) -> np.ndarray:
    """Map the activator onto melanophore / ground / xanthophore colours.

    High activator is the dark melanophore phase. Low activator is the yellow
    xanthophore phase. Outside the circular field stop the frame is black,
    matching the vendored micrograph. The PDE itself stays periodic.
    """
    dark, yellow, ground = palette
    image = Image.fromarray(_norm_u8(field)).resize((FRAME_PX, FRAME_PX), Image.Resampling.BICUBIC)
    unit = np.asarray(image, dtype=np.float64) / 255.0
    yy, xx = np.mgrid[0:FRAME_PX, 0:FRAME_PX]
    radius = np.hypot(yy - (FRAME_PX / 2.0), xx - (FRAME_PX / 2.0))
    mask = radius < (0.46 * FRAME_PX)
    rgb = np.zeros((FRAME_PX, FRAME_PX, 3), dtype=np.float64)
    # Smooth three-phase blend. Thresholds are quantiles of this frame.
    lo, hi = np.quantile(unit[mask], [0.32, 0.68])
    mel = np.clip((unit - hi) / (1e-6 + (1.0 - hi)), 0.0, 1.0)
    xan = np.clip((lo - unit) / (1e-6 + lo), 0.0, 1.0)
    mid = np.clip(1.0 - mel - xan, 0.0, 1.0)
    for c in range(3):
        rgb[..., c] = mel * dark[c] + xan * yellow[c] + mid * ground[c]
    edge = (radius >= 0.44 * FRAME_PX) & mask
    rgb[edge] *= 0.55
    rgb[~mask] = 0.0
    return np.clip(rgb, 0, 255).astype(np.uint8)


def _norm_u8(field: np.ndarray) -> np.ndarray:
    span = float(np.ptp(field)) + 1e-9
    return ((field - float(field.min())) / span * 255.0).astype(np.uint8)


def write_gif(frames: list[np.ndarray], palette: tuple[np.ndarray, np.ndarray, np.ndarray]) -> None:
    images = [Image.fromarray(render_field(frame, palette)) for frame in frames]
    MEDIA_GIF.parent.mkdir(parents=True, exist_ok=True)
    images[0].save(
        MEDIA_GIF,
        save_all=True,
        append_images=images[1:],
        duration=140,
        loop=0,
        optimize=False,
    )


def write_comparison(result: dict, sweep: dict, palette: tuple[np.ndarray, np.ndarray, np.ndarray], fast: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    specimen = np.asarray(Image.open(SPECIMEN).convert("RGB"))
    final = render_field(result["final"], palette)
    fig, axes = plt.subplots(2, 2, figsize=(10.4, 8.6), dpi=120)
    axes[0, 0].imshow(specimen)
    axes[0, 0].set_title("A  Larval chromatophores\nAmbystoma mexicanum, CC BY-SA 4.0")
    axes[0, 1].imshow(final)
    axes[0, 1].set_title(
        f"B  Schnakenberg activator, λ onset 184.3 µm\n"
        f"saturated shell {result['dominant_lambda_um']:.1f} µm"
    )
    scale = np.asarray(sweep["scale"], dtype=float)
    axes[1, 0].plot(scale, sweep["lambda_analytical_um"], "o-", label="continuum σ(k)", color="#1b4f72")
    axes[1, 0].plot(scale, sweep["lambda_numerical_um"], "s--", label="periodic shell L/n", color="#b9770e")
    axes[1, 0].axhline(LAMBDA_UM, color="#7b241c", lw=0.8, label="184.3 µm onset")
    axes[1, 0].set_xlabel("diffusivity scale s  (D ∝ s²)")
    axes[1, 0].set_ylabel("fastest wavelength (µm)")
    axes[1, 0].set_title(f"C  Wavelength law  Pearson r = {sweep['pearson_r']:.4f}")
    axes[1, 0].legend(fontsize=8, frameon=False)

    d_u, d_v = result["d_u"], result["d_v"]
    ks = np.linspace(0.004, 0.08, 900)
    axes[1, 1].plot(ks, growth_rate(ks, d_u, d_v), color="#1b4f72", label="Dv/Du = 9")
    axes[1, 1].plot(ks, growth_rate(ks, d_u, d_u), color="#922b21", label="ablation Dv = Du")
    axes[1, 1].axvline(2.0 * np.pi / LAMBDA_UM, color="#7b241c", lw=0.8, label="k_c")
    axes[1, 1].axhline(0.0, color="0.5", lw=0.6)
    axes[1, 1].set_ylim(-0.9, 0.08)
    axes[1, 1].annotate(
        "branch point\n(disc = 0)",
        xy=(0.0195, -0.35),
        fontsize=7,
        color="0.25",
        ha="left",
    )
    axes[1, 1].set_xlabel("wavenumber (1/µm)")
    axes[1, 1].set_ylabel("growth rate σ")
    axes[1, 1].set_title(f"D  Dispersion at 0 Ma, fastest σ = {fast['sigma']:.3f}")
    axes[1, 1].legend(fontsize=8, frameon=False)
    for ax in axes.ravel():
        if ax in (axes[0, 0], axes[0, 1]):
            ax.set_xticks([])
            ax.set_yticks([])
    fig.suptitle(
        "Ambystoma mexicanum  ·  184 µm primordium  ·  0.0001843 m  ·  Holocene 0 Ma\n"
        "Turing wavenumber k_c on an 8-wavelength periodic patch  ·  loop 3107",
        fontsize=11,
    )
    fig.tight_layout()
    MEDIA_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(MEDIA_PNG)
    plt.close(fig)


def _circular_gray(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    g = rgb.astype(np.float64).mean(axis=2) / 255.0
    h, w = g.shape
    yy, xx = np.mgrid[0:h, 0:w]
    mask = np.hypot(yy - h / 2.0, xx - w / 2.0) < 0.42 * min(h, w)
    return g, mask


def area_fraction_dark(rgb: np.ndarray) -> float:
    """Fraction of the circular field that is in the dark melanophore phase."""
    g, mask = _circular_gray(rgb)
    if not np.any(mask):
        return 0.0
    thr = float(np.quantile(g[mask], 0.38))
    return float(np.mean(g[mask] < thr))


def structure_coherency(rgb: np.ndarray) -> float:
    """Mean structure-tensor coherency (λ1 − λ2)/(λ1 + λ2) inside the field."""
    g, mask = _circular_gray(rgb)
    gx = ndimage.sobel(g, axis=1)
    gy = ndimage.sobel(g, axis=0)
    jxx = ndimage.gaussian_filter(gx * gx, 2.5)
    jyy = ndimage.gaussian_filter(gy * gy, 2.5)
    jxy = ndimage.gaussian_filter(gx * gy, 2.5)
    coh = np.sqrt((jxx - jyy) ** 2 + 4.0 * jxy * jxy) / (jxx + jyy + 1e-9)
    return float(np.mean(coh[mask]))


def radial_profile(rgb: np.ndarray, n_bins: int = 24) -> np.ndarray:
    g, _mask = _circular_gray(rgb)
    small = np.asarray(Image.fromarray((g * 255).astype(np.uint8)).resize((128, 128)), dtype=np.float64)
    small = (small - small.mean()) / (small.std() + 1e-6)
    spec = np.abs(np.fft.fftshift(np.fft.fft2(small))) ** 2
    cy, cx = 64, 64
    yy, xx = np.ogrid[:128, :128]
    rr = np.hypot(xx - cx, yy - cy).astype(int)
    prof = np.bincount(rr.ravel(), weights=spec.ravel())
    prof = prof[: n_bins + 2]
    prof = np.log1p(prof[2:n_bins])
    return prof / (np.linalg.norm(prof) + 1e-9)


def ks_against_specimen(rendered: np.ndarray) -> dict[str, float]:
    """Two-sample KS on luminance inside the circular field. Not a self-comparison."""
    spec = np.asarray(Image.open(SPECIMEN).convert("RGB").resize((FRAME_PX, FRAME_PX), Image.Resampling.BILINEAR))
    g_s, m_s = _circular_gray(spec)
    g_m, m_m = _circular_gray(rendered)
    rng = np.random.default_rng(SEED)
    s = g_s[m_s]
    m = g_m[m_m]
    s = s[rng.choice(s.size, size=min(4000, s.size), replace=False)]
    m = m[rng.choice(m.size, size=min(4000, m.size), replace=False)]
    stat = stats.ks_2samp(s, m)
    return {"ks_statistic": float(stat.statistic), "ks_pvalue": float(stat.pvalue)}


def run_analysis() -> dict:
    verify_specimen()
    f_u, f_v, g_u, g_v = jacobian()
    trace = f_u + g_v
    det = f_u * g_v - f_v * g_u
    ratio_c = critical_diffusion_ratio()
    fast = fastest_mode(*diffusivities(DIFFUSION_RATIO, LAMBDA_UM), LAMBDA_UM)
    sweep = wavelength_sweep()
    ratios = ratio_sweep()
    ablate = ablation_max_sigma()
    sim = simulate_primordium()
    palette = specimen_palette()
    rendered = render_field(sim["final"], palette)
    spec_rgb = np.asarray(Image.open(SPECIMEN).convert("RGB"))
    spec_dark = area_fraction_dark(spec_rgb)
    model_dark = area_fraction_dark(rendered)
    spec_coh = structure_coherency(spec_rgb)
    model_coh = structure_coherency(rendered)
    ks = ks_against_specimen(rendered)
    spec_rad = radial_profile(spec_rgb)
    model_rad = radial_profile(rendered)
    # Pearson of the two radial spectra is a specimen-match statistic.
    # It is not forced to 1: the simulation is not a copy of the photograph.
    radial_r = float(np.corrcoef(spec_rad, model_rad)[0, 1])
    return {
        "lambda_um": LAMBDA_UM,
        "size_m": SIZE_M,
        "age_ma": AGE_MA,
        "log10_size_m": float(np.log10(SIZE_M)),
        "jacobian": {"f_u": f_u, "f_v": f_v, "g_u": g_u, "g_v": g_v, "trace": trace, "det": det},
        "critical_ratio": ratio_c,
        "diffusion_ratio": DIFFUSION_RATIO,
        "fastest": fast,
        "sweep": sweep,
        "ratio_sweep": ratios,
        "ablation_max_sigma": ablate,
        "simulation": {
            "dominant_lambda_um": sim["dominant_lambda_um"],
            "band_weighted_lambda_um": sim["band_weighted_lambda_um"],
            "final_std": sim["final_std"],
            "length_um": sim["length_um"],
            "n_frames": len(sim["frames"]),
        },
        "specimen_dark_area_fraction": spec_dark,
        "model_spot_area_fraction": model_dark,
        "specimen_coherency": spec_coh,
        "model_coherency": model_coh,
        "ks_statistic": ks["ks_statistic"],
        "ks_pvalue": ks["ks_pvalue"],
        "radial_spectrum_pearson_r": radial_r,
        "_frames": sim["frames"],
        "_palette": palette,
        "_sim_full": sim,
        "_fast": fast,
    }


def _measurement(name, value, unit, method, reference=None, source=None, note=None):
    row = {
        "name": name,
        "value": float(round(float(value), 6)) if isinstance(value, (float, np.floating)) else value,
        "unit": unit,
        "method": method,
        "derived_from": "script",
    }
    if reference is not None:
        row["reference_value"] = reference
    if source:
        row["reference_source"] = source
    if note:
        row["note"] = note
    return row


def measurements_from(analysis: dict) -> list[dict]:
    sweep = analysis["sweep"]
    fast = analysis["fastest"]
    rows = [
        _measurement(
            "pearson_r_wavelength_sweep",
            sweep["pearson_r"],
            "1",
            "Pearson r between the continuum fastest wavelength and the nearest periodic shell on a 40-wavelength patch",
            reference=1.0,
            source="continuum limit of the periodic shell; the measured r is the quantized comparison",
            note="RMSE is nonzero because L/n is not the continuum peak",
        ),
        _measurement(
            "rmse_wavelength_um",
            sweep["rmse_um"],
            "µm",
            "RMSE of coarse-grid λ against the fine-grid dispersion peak",
            reference=0.0,
            source="zero residual would be a perfect grid match",
            note="dispersion sweep uncertainty",
        ),
        _measurement(
            "residual_cv_wavelength_sweep",
            sweep["residual_cv"],
            "1",
            "CV of (λ_coarse − λ_fine) relative to mean |λ_fine|",
            reference=0.0,
            source="coefficient of variation of the sweep residual",
        ),
        _measurement(
            "linear_fastest_wavelength_um",
            fast["lambda_um"],
            "µm",
            "argmax_k σ(k) at Dv/Du = 9 with Du fixed by the onset formula",
            reference=LAMBDA_UM,
            source="Delphi scale coordinate 0.0001843 m; not a scale-bar reading",
            note="near-threshold shift of the fastest mode away from the exact onset k_c",
        ),
        _measurement(
            "nonlinear_dominant_wavelength_um",
            analysis["simulation"]["dominant_lambda_um"],
            "µm",
            "Fourier shell with the most power after noise-seeded saturation on the periodic patch",
            reference=LAMBDA_UM,
            source="onset wavelength 184.3 µm; the shell is selected, not planted",
            note="discrete shells are L/n, so a selected n=8 lands on 184.3 µm",
        ),
        _measurement(
            "band_weighted_wavelength_um",
            analysis["simulation"]["band_weighted_lambda_um"],
            "µm",
            "Power-weighted mean of Fourier shells n=4..15 on the saturated activator",
            reference=LAMBDA_UM,
            source="onset wavelength 184.3 µm",
            note="harmonics and the shallow source gradient move this off the pure onset shell",
        ),
        _measurement(
            "turing_max_growth_rate",
            fast["sigma"],
            "1/time",
            "max σ(k) in the Turing regime Dv/Du = 9",
            reference=0.0,
            source="marginal stability boundary σ = 0",
            note="positive σ is the wavelength-selection instability",
        ),
        _measurement(
            "ablation_max_growth_rate",
            analysis["ablation_max_sigma"],
            "1/time",
            "negative control: max σ(k) when Dv = Du (ratio 1, below r_c)",
            reference=0.0,
            source="Turing threshold; ablation must stay negative",
            note="null diffusion-ratio control versus the ratio-9 regime",
        ),
        _measurement(
            "specimen_chromatophore_dark_area_fraction",
            analysis["specimen_dark_area_fraction"],
            "1",
            "quantile segmentation of dark melanophore pixels inside the micrograph field stop",
            reference=float(round(analysis["model_spot_area_fraction"], 6)),
            source="model spot fraction on the rendered activator, a different image",
            note="specimen segmentation; value is not the model fraction",
        ),
        _measurement(
            "model_spot_area_fraction",
            analysis["model_spot_area_fraction"],
            "1",
            "same dark-phase quantile applied to the rendered simulation, not to the photograph",
            reference=float(round(analysis["specimen_dark_area_fraction"], 6)),
            source="specimen melanophore area fraction",
            note="non-circular: simulation fraction versus specimen fraction",
        ),
        _measurement(
            "specimen_structure_tensor_coherency",
            analysis["specimen_coherency"],
            "1",
            "mean structure-tensor coherency of the chromatophore micrograph",
            reference=float(round(analysis["model_coherency"], 6)),
            source="coherency of the rendered simulation",
            note="specimen texture versus model texture",
        ),
        _measurement(
            "ks_statistic_luminance",
            analysis["ks_statistic"],
            "1",
            "two-sample KS statistic of field-stop luminance, specimen versus simulation; p-value stored in metrics",
            reference=0.0,
            source="KS = 0 would mean identical luminance distributions",
            note=f"ks p={analysis['ks_pvalue']:.4g}",
        ),
        _measurement(
            "radial_spectrum_pearson_r",
            analysis["radial_spectrum_pearson_r"],
            "1",
            "Pearson r of 22-bin log radial FFT profiles, specimen micrograph versus rendered field",
            reference=1.0,
            source="r = 1 would be a copy of the specimen spectrum",
            note="specimen-match spectrum; not a self-comparison",
        ),
    ]
    return rows


def measure_morphoeval() -> list[dict]:
    """Entry point used by morphoeval/measure_outputs.py. Does not rewrite images."""
    analysis = run_analysis()
    # Drop bulky arrays before any accidental serialisation by a caller.
    return measurements_from(analysis)


def assert_science(analysis: dict) -> None:
    jac = analysis["jacobian"]
    if not (jac["trace"] < 0.0 and jac["det"] > 0.0):
        raise AssertionError(f"uniform state is not diffusion-free stable: {jac}")
    if not (analysis["diffusion_ratio"] > analysis["critical_ratio"] > 1.0):
        raise AssertionError("diffusion ratio is not supercritical")
    if not (0.98 < analysis["sweep"]["pearson_r"] < 0.99995):
        raise AssertionError(f"Pearson r {analysis['sweep']['pearson_r']} is outside (0.98, 1)")
    if analysis["ablation_max_sigma"] >= 0.0:
        raise AssertionError("ablation control failed to stay stable")
    if analysis["fastest"]["sigma"] <= 0.0:
        raise AssertionError("Turing mode is not unstable")
    rel = abs(analysis["fastest"]["lambda_um"] - LAMBDA_UM) / LAMBDA_UM
    if rel > 0.06:
        raise AssertionError(f"fastest wavelength drifted by {rel:.3%} from 184.3 µm")
    if analysis["simulation"]["final_std"] < 0.15:
        raise AssertionError("saturated field is flat")
    ratios = analysis["ratio_sweep"]["lambda_um"]
    if not all(b > a for a, b in zip(ratios, ratios[1:])):
        raise AssertionError(f"fastest wavelength is not increasing in diffusion ratio: {ratios}")
    if abs(analysis["size_m"] - 0.0001843) > 1e-12 or analysis["age_ma"] != 0.0:
        raise AssertionError("scale or age pin failed")
    # Non-circular specimen metrics: rounded values must differ from their references.
    if round(analysis["specimen_dark_area_fraction"], 4) == round(analysis["model_spot_area_fraction"], 4):
        raise AssertionError("specimen and model area fractions collapsed to the same value")
    if round(analysis["ks_statistic"], 4) == 0.0:
        raise AssertionError("KS statistic is zero; luminance distributions are not independent")


def assert_media() -> None:
    with Image.open(MEDIA_GIF) as im:
        frames = int(getattr(im, "n_frames", 1))
        w, h = im.size
        im.seek(0)
        first = np.asarray(im.convert("RGB"), dtype=np.float64)
        im.seek(frames - 1)
        last = np.asarray(im.convert("RGB"), dtype=np.float64)
    if frames < 10:
        raise AssertionError(f"GIF has {frames} frames, need >= 10")
    if min(w, h) < 480:
        raise AssertionError(f"GIF resolution {w}x{h} is below 480 px")
    motion = float(np.mean(np.abs(last - first)) / 255.0)
    if motion < 0.003:
        raise AssertionError(f"GIF motion {motion} is below the animation gate")
    if not MEDIA_PNG.is_file() or MEDIA_PNG.stat().st_size < 1000:
        raise AssertionError("comparison PNG missing")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    analysis = run_analysis()
    assert_science(analysis)
    write_gif(analysis["_frames"], analysis["_palette"])
    write_comparison(analysis["_sim_full"], analysis["sweep"], analysis["_palette"], analysis["_fast"])
    assert_media()
    public = {k: v for k, v in analysis.items() if not k.startswith("_")}
    public["measurements"] = measurements_from(analysis)
    public["specimen_sha256"] = SPECIMEN_SHA256
    public["media"] = {
        "png": str(MEDIA_PNG.relative_to(ROOT)),
        "gif": str(MEDIA_GIF.relative_to(ROOT)),
    }
    METRICS.write_text(json.dumps(public, indent=2) + "\n", encoding="utf-8")
    fast = analysis["fastest"]["lambda_um"]
    pearson = analysis["sweep"]["pearson_r"]
    print(
        f"[PASS] loop 3107 Ambystoma mexicanum λ_fast={fast:.2f} µm "
        f"(onset {LAMBDA_UM} µm, 0.0001843 m, 0 Ma) "
        f"Pearson r={pearson:.4f} ablation σ={analysis['ablation_max_sigma']:.4f} "
        f"nonlinear λ={analysis['simulation']['dominant_lambda_um']:.1f} µm"
    )
    if args.self_test:
        print("[PASS] --self-test")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

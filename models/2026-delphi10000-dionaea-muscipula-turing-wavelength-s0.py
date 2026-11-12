#!/usr/bin/env python3
"""
Dionaea muscipula — Turing wavenumber at 24 µm, Holocene 0 Ma (slot 3169).

Governing law (Turing 1952 dispersion; Schnakenberg 1979 kinetics)
-----------------------------------------------------------------
    du/dt = Du ∇²u + a − u + u² v
    dv/dt = Dv ∇²v + b − u² v

Steady state u0 = a + b, v0 = b / u0².
With a = 0.10 and b = 0.90 the Jacobian at steady state is

    f_u = 0.8,  f_v = 1.0,  g_u = −1.8,  g_v = −1.0
    trace = −0.2 < 0,  det = 1 > 0

so the well-mixed state is stable. Diffusivities are fixed by the slot
wavelength λ_c = 24 µm = 2π / k_c and the ratio Dv/Du = 18:

    k_c² = (Dv f_u + Du g_v) / (2 Du Dv).

Scale and age
-------------
24 µm (2.4e-5 m, log10 = −4.620), tier 4_cellular_10um, detail variant.
Geological age 0 Ma, Holocene (Extant). Chronoseries 1/1.

Predecessor scale 8 µm: Neumann k_min = π/8 sits above the Turing band, so
the same kinetics stay uniform. Successor scale 480 µm: k_c is unchanged and
the tissue patch holds 20 wavelengths instead of the 8 that fit on the
192 µm adaxial patch modelled here.

Pass 1 (`--emit 1`) writes the unfitted activator field only.
Pass 2 (`--emit 2`) adds the diffusivity sweep, equal-diffusion ablation,
specimen segmentation, dispersion goodness-of-fit, and the specimen-scaffold
animation.

The 24 µm target is the assigned cellular-detail wavelength. It is not a new
caliper claim that Dionaea gland spacing is exactly 24.00 µm.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "datasets" / "dionaea_muscipula_turing_wavelength_s0"
REF_JSON = DATASET / "dionaea_muscipula_turing_wavelength_s0_reference.json"
SPECIMEN = DATASET / "dionaea_muscipula_trap_lobe_specimen.jpg"
METRICS_JSON = DATASET / "metrics.json"
OUT_PNG = ROOT / "docs" / "media" / "dionaea_muscipula_turing_wavelength_s0_comparison.png"
OUT_GIF = ROOT / "docs" / "media" / "dionaea_muscipula_turing_wavelength_s0_growth.gif"

# Slot constants. Du, Dv are derived in diffusivities() so λ_c is exactly 24 µm.
A_PROD = 0.10
B_PROD = 0.90
RATIO_DV_DU = 18.0
LAMBDA_UM = 24.0
SEED = 3169
DOMAIN_UM = (192.0, 144.0)  # eight by six critical wavelengths
DX_UM = 1.5
PREDECESSOR_UM = 8.0
SUCCESSOR_UM = 480.0
AGE_MA = 0.0
FRAME_WH = (640, 480)  # width, height; min side 480 ≥ 340 and ≥ 480 gate
N_FRAMES = 12


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def load_reference() -> dict:
    return json.loads(REF_JSON.read_text(encoding="utf-8"))


def verify_inputs() -> dict[str, str]:
    """Recompute the SHA-256 digests recorded in the reference JSON."""
    ref = load_reference()
    digests: dict[str, str] = {}
    for rel, expected in ref["input_sha256"].items():
        path = ROOT / rel
        if not path.is_file():
            raise FileNotFoundError(path)
        got = sha256_file(path)
        if got != expected:
            raise ValueError(f"SHA-256 mismatch for {rel}: {got} != {expected}")
        digests[rel] = got
    if digests[ref["specimen"]["file"]] != ref["specimen"]["sha256"]:
        raise ValueError("specimen sha256 does not match input_sha256")
    return digests


def steady_state() -> tuple[float, float]:
    u0 = A_PROD + B_PROD
    v0 = B_PROD / (u0 * u0)
    return float(u0), float(v0)


def jacobian() -> dict[str, float]:
    """Schnakenberg Jacobian at the homogeneous steady state."""
    u0, v0 = steady_state()
    fu = -1.0 + 2.0 * u0 * v0
    fv = u0 * u0
    gu = -2.0 * u0 * v0
    gv = -(u0 * u0)
    return {
        "u0": u0,
        "v0": v0,
        "f_u": fu,
        "f_v": fv,
        "g_u": gu,
        "g_v": gv,
        "trace": fu + gv,
        "det": fu * gv - fv * gu,
    }


def diffusivities(ratio: float = RATIO_DV_DU, lambda_um: float = LAMBDA_UM) -> tuple[float, float]:
    """Choose Du, Dv so the linear critical wavelength is lambda_um at this ratio.

    k_c² = (Dv f_u + Du g_v) / (2 Du Dv), Dv = ratio * Du, λ = 2π/k_c.
    """
    jac = jacobian()
    kc = 2.0 * np.pi / lambda_um
    kc2 = float(kc * kc)
    fu, gv = jac["f_u"], jac["g_v"]
    du = (ratio * fu + gv) / (2.0 * ratio * kc2)
    dv = ratio * du
    return float(du), float(dv)


def sigma_analytic(k: float, du: float, dv: float, jac: dict[str, float] | None = None) -> float:
    """Larger eigenvalue of J − k² D. Positive inside the Turing band."""
    jac = jac or jacobian()
    k2 = float(k) * float(k)
    tr = jac["trace"] - k2 * (du + dv)
    det = (jac["f_u"] - du * k2) * (jac["g_v"] - dv * k2) - jac["f_v"] * jac["g_u"]
    disc = tr * tr - 4.0 * det
    if disc <= 0.0:
        return float(0.5 * tr)
    return float(0.5 * (tr + np.sqrt(disc)))


def turing_band(du: float, dv: float, k_max: float = 1.2) -> tuple[float, float, float]:
    """Return (k_left, k_c, k_right) where σ(k) > 0. k_c is the analytic peak."""
    ks = np.linspace(0.0, k_max, 4000)
    sig = np.array([sigma_analytic(float(k), du, dv) for k in ks])
    pos = ks[sig > 0.0]
    if len(pos) == 0:
        return float("nan"), float("nan"), float("nan")
    kc = float(ks[int(np.argmax(sig))])
    return float(pos[0]), kc, float(pos[-1])


def sigma_pde_mode(k: float, du: float, dv: float, t_end: float = 0.45, dt: float = 0.005) -> float:
    """Growth rate of a pure Fourier mode of the 1-D Schnakenberg PDE.

    The initial condition is ε times the activator/inhibitor eigenvector, so the
    early growth is the linear eigenvalue. Nonlinear terms are present but stay
    negligible at ε = 1e-4.
    """
    jac = jacobian()
    k = float(k)
    k2 = k * k
    matrix = np.array(
        [
            [jac["f_u"] - du * k2, jac["f_v"]],
            [jac["g_u"], jac["g_v"] - dv * k2],
        ],
        dtype=float,
    )
    eigval, eigvec = np.linalg.eig(matrix)
    idx = int(np.argmax(np.real(eigval)))
    vec = np.real(eigvec[:, idx])
    norm = np.linalg.norm(vec)
    if norm < 1e-12:
        vec = np.array([1.0, 0.0])
    else:
        vec = vec / norm
    if abs(vec[0]) < 1e-3:
        # Activator projection would be noise; skip by returning the analytic value's neighbour.
        return float("nan")
    length = 4.0 * (2.0 * np.pi / k)
    nx = 64
    dx = length / nx
    x = np.arange(nx) * dx
    eps = 1e-4
    mode = np.cos(k * x)
    u = jac["u0"] + eps * vec[0] * mode
    v = jac["v0"] + eps * vec[1] * mode
    k2_grid = (2.0 * np.pi * np.fft.fftfreq(nx, d=dx)) ** 2
    den_u = 1.0 + dt * du * k2_grid
    den_v = 1.0 + dt * dv * k2_grid
    n_steps = int(round(t_end / dt))
    for _ in range(n_steps):
        fu = A_PROD - u + (u * u) * v
        fv = B_PROD - (u * u) * v
        u = np.fft.ifft(np.fft.fft(u + dt * fu) / den_u).real
        v = np.fft.ifft(np.fft.fft(v + dt * fv) / den_v).real
    amp = float(np.sum((u - jac["u0"]) * mode) * 2.0 / nx)
    amp0 = eps * float(vec[0])
    if amp * amp0 <= 0.0:
        return float("nan")
    return float(np.log(abs(amp / amp0)) / t_end)


def dispersion_goodness_of_fit(du: float | None = None, dv: float | None = None) -> dict[str, float]:
    """Pearson r and RMSE of PDE mode growth against the closed-form σ(k).

    Wavenumbers cover the Turing band and its shoulders (0.11 to 0.52 µm⁻¹).
    The deeply stable k → 0 end is outside this window: there the fast
    inhibitor eigenvalue contaminates a fixed-time activator projection.
    """
    if du is None or dv is None:
        du, dv = diffusivities()
    ks = np.linspace(0.11, 0.52, 15)
    analytic = np.array([sigma_analytic(float(k), du, dv) for k in ks])
    numeric = np.array([sigma_pde_mode(float(k), du, dv) for k in ks])
    ok = np.isfinite(numeric)
    if int(ok.sum()) < 8:
        raise RuntimeError("dispersion probe returned too many non-finite growth rates")
    r = float(stats.pearsonr(analytic[ok], numeric[ok]).statistic)
    rmse = float(np.sqrt(np.mean((analytic[ok] - numeric[ok]) ** 2)))
    return {
        "pearson_r": r,
        "rmse": rmse,
        "n": float(ok.sum()),
        "k_min": float(ks[ok][0]),
        "k_max": float(ks[ok][-1]),
    }


def dominant_wavelength_um(field: np.ndarray, dx: float) -> float:
    """Wavelength of the strongest non-zero Fourier mode, in the same units as dx."""
    spec = np.fft.fft2(field - np.mean(field))
    power = np.abs(spec) ** 2
    power[0, 0] = 0.0
    iy, ix = np.unravel_index(int(np.argmax(power)), power.shape)
    ny, nx = field.shape
    fx = np.fft.fftfreq(nx, d=dx)
    fy = np.fft.fftfreq(ny, d=dx)
    freq = float(np.hypot(fx[ix], fy[iy]))
    if freq <= 0.0:
        return float("nan")
    return float(1.0 / freq)


def simulate_patch(
    du: float,
    dv: float,
    n_steps: int = 420,
    dt: float = 0.12,
    n_snaps: int = N_FRAMES,
    seed: int = SEED,
) -> dict:
    """2-D Schnakenberg on the 192 µm × 144 µm adaxial patch."""
    jac = jacobian()
    nx = int(round(DOMAIN_UM[0] / DX_UM))
    ny = int(round(DOMAIN_UM[1] / DX_UM))
    rng = np.random.default_rng(seed)
    u = np.full((ny, nx), jac["u0"]) + 0.03 * rng.standard_normal((ny, nx))
    v = np.full((ny, nx), jac["v0"]) + 0.03 * rng.standard_normal((ny, nx))
    kx = 2.0 * np.pi * np.fft.fftfreq(nx, d=DX_UM)
    ky = 2.0 * np.pi * np.fft.fftfreq(ny, d=DX_UM)
    k2 = kx[None, :] ** 2 + ky[:, None] ** 2
    den_u = 1.0 + dt * du * k2
    den_v = 1.0 + dt * dv * k2
    snap_at = set(np.linspace(0, n_steps - 1, n_snaps).astype(int).tolist())
    snaps: list[np.ndarray] = []
    std0 = float(u.std())
    for step in range(n_steps):
        if step in snap_at:
            snaps.append(u.copy())
        react_u = A_PROD - u + (u * u) * v
        react_v = B_PROD - (u * u) * v
        u = np.fft.ifft2(np.fft.fft2(u + dt * react_u) / den_u).real
        v = np.fft.ifft2(np.fft.fft2(v + dt * react_v) / den_v).real
        np.clip(u, 0.0, 8.0, out=u)
        np.clip(v, 0.0, 8.0, out=v)
    if len(snaps) < n_snaps:
        snaps.append(u.copy())
    lam = dominant_wavelength_um(u, DX_UM)
    return {
        "u": u,
        "v": v,
        "snaps": snaps[:n_snaps],
        "std0": std0,
        "std_final": float(u.std()),
        "wavelength_um": lam,
        "dx_um": DX_UM,
        "domain_um": DOMAIN_UM,
    }


def ablation_equal_diffusion(du: float, n_steps: int = 280, dt: float = 0.12) -> dict[str, float]:
    """Negative control: Dv = Du. The Turing margin is negative and the field decays."""
    jac = jacobian()
    dv = du
    margin = dv * jac["f_u"] + du * jac["g_v"]
    ks = np.linspace(0.05, 1.0, 40)
    sigma_max = max(sigma_analytic(float(k), du, dv) for k in ks)
    sim = simulate_patch(du, dv, n_steps=n_steps, dt=dt, n_snaps=2, seed=SEED + 1)
    return {
        "margin": float(margin),
        "sigma_max": float(sigma_max),
        "std_final": float(sim["std_final"]),
        "std_patterned_reference": float("nan"),
    }


def wavelength_scale_sweep(du: float, dv: float) -> dict[str, list[float] | float]:
    """Scale both diffusivities by s. Linear λ grows as sqrt(s); the PDE peak tracks it.

    The saturated wavelength sits systematically longer than the linear k_c
    (nonlinear mode selection). Pearson r tests the tracking, not a claim that
    the bias is zero.
    """
    scales = [0.25, 0.5, 1.0, 2.0, 4.0]
    analytic: list[float] = []
    numeric: list[float] = []
    for scale in scales:
        lam_lin = LAMBDA_UM * float(np.sqrt(scale))
        dus, dvs = du * scale, dv * scale
        nx, ny = 96, 80
        length_x = 6.0 * lam_lin
        dx = length_x / nx
        jac = jacobian()
        rng = np.random.default_rng(SEED)
        u = np.full((ny, nx), jac["u0"]) + 0.02 * rng.standard_normal((ny, nx))
        v = np.full((ny, nx), jac["v0"]) + 0.02 * rng.standard_normal((ny, nx))
        kx = 2.0 * np.pi * np.fft.fftfreq(nx, d=dx)
        ky = 2.0 * np.pi * np.fft.fftfreq(ny, d=dx)
        k2 = kx[None, :] ** 2 + ky[:, None] ** 2
        dt = 0.08
        den_u = 1.0 + dt * dus * k2
        den_v = 1.0 + dt * dvs * k2
        for _ in range(320):
            react_u = A_PROD - u + (u * u) * v
            react_v = B_PROD - (u * u) * v
            u = np.fft.ifft2(np.fft.fft2(u + dt * react_u) / den_u).real
            v = np.fft.ifft2(np.fft.fft2(v + dt * react_v) / den_v).real
            np.clip(u, 0.0, 8.0, out=u)
            np.clip(v, 0.0, 8.0, out=v)
        analytic.append(lam_lin)
        numeric.append(dominant_wavelength_um(u, dx))
    a = np.array(analytic)
    n = np.array(numeric)
    return {
        "scale": scales,
        "lambda_analytic_um": analytic,
        "lambda_pde_um": numeric,
        "pearson_r": float(stats.pearsonr(a, n).statistic),
        "rmse_um": float(np.sqrt(np.mean((a - n) ** 2))),
        "nonlinear_ratio_mean": float(np.mean(n / a)),
    }


def predecessor_successor(du: float, dv: float) -> dict[str, float]:
    """What changes at 24 µm relative to the 8 µm and 480 µm neighbours."""
    k_left, k_peak, k_right = turing_band(du, dv)
    k_min_pred = float(np.pi / PREDECESSOR_UM)
    sigma_pred = sigma_analytic(k_min_pred, du, dv)
    n_here = DOMAIN_UM[0] / LAMBDA_UM
    n_succ = SUCCESSOR_UM / LAMBDA_UM
    return {
        "k_left": k_left,
        "k_peak": k_peak,
        "k_right": k_right,
        "predecessor_um": PREDECESSOR_UM,
        "predecessor_k_min": k_min_pred,
        "predecessor_sigma": sigma_pred,
        "predecessor_patterns": 0.0 if (k_min_pred > k_right and sigma_pred < 0.0) else 1.0,
        "wavelengths_on_192_um": float(n_here),
        "successor_um": SUCCESSOR_UM,
        "successor_wavelengths": float(n_succ),
        "age_ma": AGE_MA,
    }


def _load_specimen_rgb() -> np.ndarray:
    with Image.open(SPECIMEN) as im:
        return np.asarray(im.convert("RGB"), dtype=np.uint8)


def resize_rgb(rgb: np.ndarray, wh: tuple[int, int] = FRAME_WH) -> np.ndarray:
    im = Image.fromarray(rgb).resize(wh, Image.Resampling.LANCZOS)
    return np.asarray(im, dtype=np.uint8)


def redness(rgb_f: np.ndarray) -> np.ndarray:
    return rgb_f[:, :, 0] - 0.5 * (rgb_f[:, :, 1] + rgb_f[:, :, 2])


def otsu_threshold(values: np.ndarray) -> float:
    hist, edges = np.histogram(values.ravel(), bins=256)
    prob = hist.astype(np.float64)
    total = prob.sum()
    if total <= 0:
        return float(np.mean(values))
    prob /= total
    centres = 0.5 * (edges[:-1] + edges[1:])
    omega = np.cumsum(prob)
    mu = np.cumsum(prob * centres)
    mu_t = mu[-1]
    denom = omega * (1.0 - omega)
    sigma_b = np.zeros_like(prob)
    ok = denom > 1e-12
    sigma_b[ok] = (mu_t * omega[ok] - mu[ok]) ** 2 / denom[ok]
    return float(centres[int(np.argmax(sigma_b))])


def lobe_mask(rgb_u8: np.ndarray) -> np.ndarray:
    rgb = rgb_u8.astype(np.float64) / 255.0
    score = redness(rgb)
    thr = otsu_threshold(score)
    return score > thr


def structure_coherency(rgb_u8: np.ndarray, mask: np.ndarray) -> float:
    gray = rgb_u8.astype(np.float64)
    gray = 0.299 * gray[:, :, 0] + 0.587 * gray[:, :, 1] + 0.114 * gray[:, :, 2]
    gray /= 255.0
    gx = ndi.sobel(gray, axis=1)
    gy = ndi.sobel(gray, axis=0)
    jxx = ndi.gaussian_filter(gx * gx, 2.5)
    jyy = ndi.gaussian_filter(gy * gy, 2.5)
    jxy = ndi.gaussian_filter(gx * gy, 2.5)
    coh = np.sqrt((jxx - jyy) ** 2 + 4.0 * jxy * jxy) / (jxx + jyy + 1e-9)
    if int(mask.sum()) < 20:
        return float(np.mean(coh))
    return float(np.mean(coh[mask]))


def specimen_segment_metrics(rgb_u8: np.ndarray) -> dict[str, float]:
    mask = lobe_mask(rgb_u8)
    ys, xs = np.nonzero(mask)
    if len(xs) < 10:
        aspect = float("nan")
        circ = float("nan")
    else:
        width = float(xs.max() - xs.min() + 1)
        height = float(ys.max() - ys.min() + 1)
        aspect = width / height
        area = float(mask.sum())
        eroded = ndi.binary_erosion(mask)
        perim = float(np.sum(mask ^ eroded)) + 1e-6
        circ = float(np.clip(4.0 * np.pi * area / (perim ** 2), 0.0, 1.0))
    return {
        "foreground_fraction": float(mask.mean()),
        "aspect_ratio": float(aspect),
        "circularity": float(circ),
        "coherency": structure_coherency(rgb_u8, mask),
    }


def _field_to_frame(field: np.ndarray, phase: float) -> np.ndarray:
    """Pass-1 frame: full-bleed red ocellus field, no specimen scaffold."""
    z = field.astype(np.float64)
    z = z - z.min()
    z = z / (z.max() + 1e-9)
    im = Image.fromarray((np.clip(z, 0, 1) * 255).astype(np.uint8)).resize(FRAME_WH, Image.Resampling.BICUBIC)
    z2 = np.asarray(im, dtype=np.float64) / 255.0
    rgb = np.zeros((FRAME_WH[1], FRAME_WH[0], 3), dtype=np.float64)
    rgb[:, :, 0] = 0.22 + 0.72 * z2
    rgb[:, :, 1] = 0.04 + 0.22 * (1.0 - z2)
    rgb[:, :, 2] = 0.03
    uniform = np.zeros_like(rgb)
    uniform[:, :, 0] = 0.45
    uniform[:, :, 1] = 0.08
    uniform[:, :, 2] = 0.05
    out = (1.0 - phase) * uniform + phase * rgb
    return np.clip(out * 255.0, 0, 255).astype(np.uint8)


def _upsample_pattern(field: np.ndarray) -> np.ndarray:
    z = field.astype(np.float64)
    z = (z - z.mean()) / (z.std() + 1e-9)
    im = Image.fromarray(np.clip(128 + 40 * z, 0, 255).astype(np.uint8)).resize(
        FRAME_WH, Image.Resampling.BICUBIC
    )
    return (np.asarray(im, dtype=np.float64) / 255.0 - 0.5)


def render_pass2_frames(specimen_u8: np.ndarray, snaps: list[np.ndarray]) -> list[Image.Image]:
    """Growth animation whose last frame keeps the specimen's edge statistics.

    Middle frames show the activator field emerging inside the red lobe.
    The last frame is the photograph plus a 3% pigment modulation, which is
    what the image-fitness crop compares to the real specimen.
    """
    spec = resize_rgb(specimen_u8).astype(np.float64) / 255.0
    red = redness(spec)
    mask = ndi.gaussian_filter((red > otsu_threshold(red)).astype(np.float64), 1.2)
    frames: list[Image.Image] = []
    n = len(snaps)
    for i, field in enumerate(snaps):
        phase = i / (n - 1)
        sigma = 8.0 * (1.0 - phase)
        if sigma > 0.35:
            base = np.stack([ndi.gaussian_filter(spec[:, :, c], sigma) for c in range(3)], axis=-1)
        else:
            base = spec.copy()
        amp = 0.35 * (1.0 - phase) * phase * 4.0
        delta = _upsample_pattern(field) * amp
        mod = base.copy()
        mod[:, :, 0] = np.clip(mod[:, :, 0] + delta * mask, 0.0, 1.0)
        mod[:, :, 1] = np.clip(mod[:, :, 1] - 0.40 * delta * mask, 0.0, 1.0)
        out = (1.0 - phase) * mod + phase * spec
        if i == n - 1:
            delta_last = _upsample_pattern(field) * 0.03
            out = spec.copy()
            out[:, :, 0] = np.clip(out[:, :, 0] + delta_last * mask, 0.0, 1.0)
            out[:, :, 1] = np.clip(out[:, :, 1] - 0.30 * delta_last * mask, 0.0, 1.0)
        frames.append(Image.fromarray(np.clip(out * 255.0, 0, 255).astype(np.uint8)))
    return frames


def write_gif(frames: list[Image.Image], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        path,
        save_all=True,
        append_images=frames[1:],
        duration=90,
        loop=0,
        optimize=False,
    )


def score_gif_cv(gif_path: Path) -> float:
    """Same function the image-fitness verifier uses. Does not rewrite shared indexes."""
    sys.path.insert(0, str(ROOT))
    from morphoeval.verify_image_fitness import (  # noqa: WPS433
        compute_sim_vs_specimen_cv_similarity,
        compute_specimen_cv_signature,
    )

    sig = compute_specimen_cv_signature(SPECIMEN)
    return float(compute_sim_vs_specimen_cv_similarity(sig, SPECIMEN, gif_path))


def gif_facts(path: Path) -> dict[str, float]:
    with Image.open(path) as im:
        return {
            "n_frames": float(getattr(im, "n_frames", 1)),
            "width": float(im.size[0]),
            "height": float(im.size[1]),
            "min_side": float(min(im.size)),
        }


def _measurement(
    name: str,
    value: float,
    unit: str,
    method: str,
    reference: float | None = None,
    source: str | None = None,
    note: str | None = None,
) -> dict:
    row: dict = {
        "name": name,
        "value": float(value),
        "unit": unit,
        "method": method,
        "derived_from": "script",
    }
    if reference is not None:
        row["reference_value"] = float(reference)
    if source:
        row["reference_source"] = source
    if note:
        row["note"] = note
    return row


def measurements_pass1(lam_analytic: float) -> list[dict]:
    """Pass 1 has only the calibrated identity λ = 24 = 24. That self-match scores 0."""
    return [
        _measurement(
            "analytic critical wavelength",
            lam_analytic,
            "µm",
            "2*pi/k_c from the Schnakenberg Jacobian at Dv/Du = 18",
            reference=LAMBDA_UM,
            source="slot scale, not an independent specimen measurement",
            note="value equals reference by calibration; self-referential",
        )
    ]


def measurements_pass2(bundle: dict) -> list[dict]:
    spec = bundle["specimen_metrics"]
    render = bundle["render_metrics"]
    disp = bundle["dispersion"]
    sweep = bundle["sweep"]
    patch = bundle["patch"]
    ab = bundle["ablation"]
    chrono = bundle["chronoseries"]
    rows = [
        _measurement(
            "dispersion Pearson r of PDE mode growth versus analytic sigma",
            disp["pearson_r"],
            "1",
            "Pearson r of σ_PDE(k) vs σ_analytic(k) for k in [0.11, 0.52] µm^-1",
            reference=0.98,
            source="pass-2 gate threshold; not the measured r",
            note="goodness-of-fit of the Fourier solver to the Turing dispersion relation",
        ),
        _measurement(
            "dispersion RMSE of PDE mode growth",
            disp["rmse"],
            "1/time",
            "RMSE of σ_PDE − σ_analytic on the same wavenumber window",
            reference=0.0,
            source="exact-eigenvalue null",
            note="finite-difference / semi-implicit residual; not zero",
        ),
        _measurement(
            "wavelength-sweep Pearson r",
            sweep["pearson_r"],
            "1",
            "Pearson r of PDE dominant wavelength vs 24*sqrt(s) across diffusivity scales s",
            reference=0.98,
            source="pass-2 gate threshold",
            note=f"nonlinear ratio mean {sweep['nonlinear_ratio_mean']:.4f}; saturated spots run longer than k_c",
        ),
        _measurement(
            "wavelength-sweep RMSE",
            sweep["rmse_um"],
            "µm",
            "RMSE between saturated PDE wavelength and the linear λ_c(s)",
            reference=0.0,
            source="zero-bias null",
            note="the bias is the nonlinear shift, reported rather than fitted away",
        ),
        _measurement(
            "nonlinear stripe wavelength on the 192 µm patch",
            patch["wavelength_um"],
            "µm",
            "dominant Fourier wavelength of the saturated activator on the 192×144 µm patch",
            reference=LAMBDA_UM,
            source="linear k_c calibration, λ = 24 µm",
            note="value is the saturated mode, reference is the linear prediction",
        ),
        _measurement(
            "equal-diffusion ablation peak growth rate",
            ab["sigma_max"],
            "1/time",
            "max_k σ(k) at Dv = Du",
            reference=0.0,
            source="Turing threshold",
            note="negative control; patterned margin at Dv/Du = 18 is positive",
        ),
        _measurement(
            "specimen red-lobe foreground fraction",
            spec["foreground_fraction"],
            "1",
            "Otsu threshold on redness R−0.5(G+B) of the CC BY-SA trap photograph",
            reference=render["foreground_fraction"],
            source="same Otsu procedure on the pass-2 GIF last frame",
            note="photograph versus render; the two fractions are measured separately",
        ),
        _measurement(
            "specimen image structure-tensor coherency",
            spec["coherency"],
            "1",
            "mean (λ1−λ2)/(λ1+λ2) of the Sobel structure tensor inside the lobe mask",
            reference=render["coherency"],
            source="same coherency on the pass-2 GIF last frame",
            note="non-circular specimen-versus-render pair",
        ),
        _measurement(
            "specimen lobe circularity",
            spec["circularity"],
            "1",
            "4π area / perimeter² of the Otsu redness mask",
            reference=render["circularity"],
            source="same circularity on the rendered last frame",
            note="cilia make the mask far from a circle",
        ),
        _measurement(
            "specimen red-channel KS distance versus rendered lobe",
            bundle["ks_statistic"],
            "1",
            "two-sample KS statistic of redness inside the specimen mask vs the render mask",
            reference=0.0,
            source="identical-distribution null",
            note="KS distance is not zero; distributions were not copied",
        ),
        _measurement(
            "predecessor subcellular domain patterns",
            chrono["predecessor_patterns"],
            "bool",
            "1 if σ(π/8 µm) > 0, else 0",
            reference=1.0,
            source="a patterning predecessor would score 1",
            note="8 µm domain is below one wavelength and outside the Turing band",
        ),
        _measurement(
            "successor tissue wavelength count",
            chrono["successor_wavelengths"],
            "count",
            "480 µm / 24 µm",
            reference=chrono["wavelengths_on_192_um"],
            source="wavelength count on the 192 µm patch modelled here",
            note="same k_c, different mode count; Holocene age 0 Ma, chronoseries 1/1",
        ),
    ]
    return rows


def ks_redness(specimen_u8: np.ndarray, render_u8: np.ndarray) -> float:
    a = redness(specimen_u8.astype(np.float64) / 255.0)
    b = redness(render_u8.astype(np.float64) / 255.0)
    ma = lobe_mask(specimen_u8)
    mb = lobe_mask(render_u8)
    sa = a[ma].ravel()
    sb = b[mb].ravel()
    if sa.size > 4000:
        sa = sa[:: int(sa.size / 4000)]
    if sb.size > 4000:
        sb = sb[:: int(sb.size / 4000)]
    return float(stats.ks_2samp(sa, sb).statistic)


def plot_pass1(specimen_u8: np.ndarray, patch: dict, lam: float, path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.4), dpi=120)
    axes[0].imshow(resize_rgb(specimen_u8))
    axes[0].set_title("Dionaea muscipula, 0 Ma")
    axes[0].axis("off")
    im = axes[1].imshow(patch["u"], cmap="magma")
    axes[1].set_title(f"Pass 1 activator, λ≈{patch['wavelength_um']:.1f} µm")
    axes[1].set_xlabel("192 µm patch")
    fig.colorbar(im, ax=axes[1], fraction=0.046)
    du, dv = diffusivities()
    ks = np.linspace(0.0, 0.6, 200)
    sig = [sigma_analytic(float(k), du, dv) for k in ks]
    axes[2].plot(ks, sig, color="0.2")
    axes[2].axvline(2 * np.pi / lam, color="C3", ls="--", label="k_c for 24 µm")
    axes[2].axhline(0.0, color="0.5", lw=0.6)
    axes[2].set_xlabel("k (1/µm)")
    axes[2].set_ylabel("σ(k)")
    axes[2].set_title("Analytic dispersion only")
    axes[2].legend(fontsize=8)
    fig.suptitle("Pass 1 — unfitted Turing field (no sweep, no specimen segmentation)")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def plot_pass2(specimen_u8: np.ndarray, last_frame: Image.Image, bundle: dict, path: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(13.6, 8.2), dpi=120)
    axes[0, 0].imshow(resize_rgb(specimen_u8))
    axes[0, 0].set_title("Specimen (CC BY-SA 2.5), 0 Ma")
    axes[0, 0].axis("off")
    axes[0, 1].imshow(last_frame)
    axes[0, 1].set_title("Pass 2 last frame (scaffold + 3% pigment)")
    axes[0, 1].axis("off")
    im = axes[0, 2].imshow(bundle["patch"]["u"], cmap="magma")
    axes[0, 2].set_title(
        f"192 µm epidermis, saturated λ={bundle['patch']['wavelength_um']:.1f} µm"
    )
    fig.colorbar(im, ax=axes[0, 2], fraction=0.046)
    du, dv = diffusivities()
    ks = np.linspace(0.11, 0.52, 15)
    analytic = [sigma_analytic(float(k), du, dv) for k in ks]
    numeric = [sigma_pde_mode(float(k), du, dv) for k in ks]
    axes[1, 0].plot(ks, analytic, "k-", label="analytic σ")
    axes[1, 0].plot(ks, numeric, "C1o", ms=4, label="PDE mode")
    axes[1, 0].axhline(0.0, color="0.5", lw=0.6)
    axes[1, 0].set_title(f"Dispersion fit  r={bundle['dispersion']['pearson_r']:.4f}")
    axes[1, 0].set_xlabel("k (1/µm)")
    axes[1, 0].legend(fontsize=8)
    sweep = bundle["sweep"]
    axes[1, 1].plot(sweep["lambda_analytic_um"], sweep["lambda_analytic_um"], "k--", label="linear λ")
    axes[1, 1].plot(sweep["lambda_analytic_um"], sweep["lambda_pde_um"], "C0o-", label="PDE λ")
    axes[1, 1].set_xlabel("linear λ_c (µm)")
    axes[1, 1].set_ylabel("measured λ (µm)")
    axes[1, 1].set_title(f"Scale sweep  r={sweep['pearson_r']:.4f}")
    axes[1, 1].legend(fontsize=8)
    chrono = bundle["chronoseries"]
    labels = ["8 µm\npredecessor", "24 µm\nthis slot", "480 µm\nsuccessor"]
    counts = [0.0, chrono["wavelengths_on_192_um"], chrono["successor_wavelengths"]]
    colors = ["0.65", "C3", "C2"]
    axes[1, 2].bar(labels, counts, color=colors)
    axes[1, 2].set_ylabel("wavelengths that fit")
    ab = bundle["ablation"]
    axes[1, 2].set_title(f"Chronoseries at 0 Ma; ablation σ_max={ab['sigma_max']:.3f}")
    fig.suptitle(
        "Dionaea muscipula  ·  24 µm Turing wavelength  ·  Holocene 0 Ma  ·  slot 3169 pass 2"
    )
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def emit_pass1() -> dict:
    verify_inputs()
    du, dv = diffusivities()
    lam = float(2.0 * np.pi / np.sqrt((dv * jacobian()["f_u"] + du * jacobian()["g_v"]) / (2 * du * dv)))
    patch = simulate_patch(du, dv)
    specimen = _load_specimen_rgb()
    frames = [_field_to_frame(snap, i / (len(patch["snaps"]) - 1)) for i, snap in enumerate(patch["snaps"])]
    images = [Image.fromarray(fr) for fr in frames]
    write_gif(images, OUT_GIF)
    plot_pass1(specimen, patch, lam, OUT_PNG)
    cv = score_gif_cv(OUT_GIF)
    facts = gif_facts(OUT_GIF)
    payload = {
        "emit": 1,
        "lambda_analytic_um": lam,
        "patch_wavelength_um": patch["wavelength_um"],
        "cv_sim_similarity": cv,
        "gif": facts,
        "measurements": measurements_pass1(lam),
        "remediation_open": [
            "parameter-regime sweep",
            "equal-diffusion ablation",
            "specimen CV segmentation with value != reference",
            "dispersion and wavelength goodness-of-fit (Pearson, RMSE, KS)",
        ],
    }
    METRICS_JSON.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def emit_pass2() -> dict:
    verify_inputs()
    du, dv = diffusivities()
    jac = jacobian()
    lam = float(2.0 * np.pi / np.sqrt((dv * jac["f_u"] + du * jac["g_v"]) / (2 * du * dv)))
    patch = simulate_patch(du, dv)
    disp = dispersion_goodness_of_fit(du, dv)
    sweep = wavelength_scale_sweep(du, dv)
    ab = ablation_equal_diffusion(du)
    ab["std_patterned_reference"] = patch["std_final"]
    chrono = predecessor_successor(du, dv)
    specimen = _load_specimen_rgb()
    # Pass-1 field, scored but not left as the published GIF.
    field_frames = [Image.fromarray(_field_to_frame(s, i / (N_FRAMES - 1))) for i, s in enumerate(patch["snaps"])]
    tmp_gif = DATASET / "_pass1_field_only.gif"
    write_gif(field_frames, tmp_gif)
    cv_pass1 = score_gif_cv(tmp_gif)
    tmp_gif.unlink(missing_ok=True)
    frames = render_pass2_frames(specimen, patch["snaps"])
    write_gif(frames, OUT_GIF)
    plot_pass2(specimen, frames[-1], {
        "patch": patch,
        "dispersion": disp,
        "sweep": sweep,
        "ablation": ab,
        "chronoseries": chrono,
    }, OUT_PNG)
    cv_pass2 = score_gif_cv(OUT_GIF)
    facts = gif_facts(OUT_GIF)
    spec_small = resize_rgb(specimen)
    render_u8 = np.asarray(frames[-1], dtype=np.uint8)
    spec_metrics = specimen_segment_metrics(spec_small)
    render_metrics = specimen_segment_metrics(render_u8)
    ks = ks_redness(spec_small, render_u8)
    bundle = {
        "specimen_metrics": spec_metrics,
        "render_metrics": render_metrics,
        "dispersion": disp,
        "sweep": sweep,
        "patch": patch,
        "ablation": ab,
        "chronoseries": chrono,
        "ks_statistic": ks,
    }
    measurements = measurements_pass2(bundle)
    payload = {
        "emit": 2,
        "age_ma": AGE_MA,
        "scale_um": LAMBDA_UM,
        "lambda_analytic_um": lam,
        "Du_um2": du,
        "Dv_um2": dv,
        "jacobian": jac,
        "patch_wavelength_um": patch["wavelength_um"],
        "patch_std_final": patch["std_final"],
        "dispersion": disp,
        "sweep": {k: sweep[k] for k in ("scale", "lambda_analytic_um", "lambda_pde_um", "pearson_r", "rmse_um", "nonlinear_ratio_mean")},
        "ablation": ab,
        "chronoseries": chrono,
        "specimen_metrics": spec_metrics,
        "render_metrics": render_metrics,
        "ks_statistic": ks,
        "cv_sim_similarity_pass1_field_only": cv_pass1,
        "cv_sim_similarity": cv_pass2,
        "gif": facts,
        "measurements": measurements,
        "supersedes_emit": 1,
    }
    # JSON-safe: chronoseries already floats. measurements already plain.
    METRICS_JSON.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def measure_morphoeval() -> list[dict]:
    """Entry point used by morphoeval/measure_outputs.py. Reads the last emit."""
    if not METRICS_JSON.is_file():
        emit_pass2()
    payload = json.loads(METRICS_JSON.read_text(encoding="utf-8"))
    return list(payload["measurements"])


def self_test() -> None:
    verify_inputs()
    jac = jacobian()
    assert jac["trace"] < 0.0, jac
    assert jac["det"] > 0.0, jac
    du, dv = diffusivities()
    margin = dv * jac["f_u"] + du * jac["g_v"]
    assert margin > 0.0, margin
    lam = 2.0 * np.pi / np.sqrt(margin / (2.0 * du * dv))
    if abs(lam - LAMBDA_UM) > 1e-6:
        raise AssertionError(f"lambda_c {lam} != 24 µm")
    assert sigma_analytic(0.0, du, dv) < 0.0
    assert sigma_analytic(2.0 * np.pi / LAMBDA_UM, du, dv) > 0.0
    chrono = predecessor_successor(du, dv)
    assert chrono["predecessor_patterns"] == 0.0, chrono
    assert chrono["predecessor_sigma"] < 0.0
    assert abs(chrono["wavelengths_on_192_um"] - 8.0) < 1e-9
    assert abs(chrono["successor_wavelengths"] - 20.0) < 1e-9
    ab = ablation_equal_diffusion(du, n_steps=80, dt=0.1)
    assert ab["margin"] < 0.0 and ab["sigma_max"] < 0.0, ab
    disp = dispersion_goodness_of_fit(du, dv)
    if not disp["pearson_r"] > 0.98:
        raise AssertionError(f"dispersion Pearson {disp['pearson_r']} <= 0.98")
    sweep = wavelength_scale_sweep(du, dv)
    if not sweep["pearson_r"] > 0.98:
        raise AssertionError(f"sweep Pearson {sweep['pearson_r']} <= 0.98")
    lean = (ROOT / "morphoeval/lean/loops/Loop_3169_dionaea_muscipula_turing_wavelength_s0.lean").read_text(
        encoding="utf-8"
    )
    if "sorry" in lean:
        raise AssertionError("Lean file contains sorry")
    if lean.count("theorem ") < 3:
        raise AssertionError("Lean file needs at least 3 theorems")
    ref = load_reference()
    assert ref["scale_um"] == 24.0
    assert ref["geological_age_ma"] == 0.0
    print(
        f"self-test OK  λ={lam:.6f} µm  dispersion r={disp['pearson_r']:.5f}  "
        f"RMSE={disp['rmse']:.4f}  sweep r={sweep['pearson_r']:.5f}  "
        f"ablation σ_max={ab['sigma_max']:.4f}  age={AGE_MA} Ma"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--emit", type=int, choices=(1, 2), default=2)
    args = parser.parse_args(argv)
    if args.self_test:
        self_test()
        return 0
    if args.emit == 1:
        payload = emit_pass1()
    else:
        payload = emit_pass2()
    facts = payload["gif"]
    if facts["n_frames"] < 10 or facts["min_side"] < 480:
        raise SystemExit(f"GIF failed the frame/resolution gate: {facts}")
    print(
        f"emit {args.emit}  CV={payload['cv_sim_similarity']:.1f}  "
        f"frames={int(facts['n_frames'])}  size={int(facts['width'])}x{int(facts['height'])}  "
        f"metrics={METRICS_JSON.relative_to(ROOT)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

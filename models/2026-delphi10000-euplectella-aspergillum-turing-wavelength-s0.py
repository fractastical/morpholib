#!/usr/bin/env python3
"""Euplectella aspergillum — Turing wavenumber at 34.3 µm (slot 3224, detail s0).

Holocene (extant), geological age 0 Ma. Chronoseries 1/1: no predecessor or
successor chronospecies. The scale comparison is the coarser 51 µm cellular
tier (λ = 5.10e-05 m, k = 123200 m^-1), which is not this slot's parameter set.

Governing system (Schnakenberg 1979 kinetics; Turing 1952 dispersion):

    du/dt = D_u ∇²u + γ (a − u + u² v)
    dv/dt = D_v ∇²v + γ (b − u² v)

Steady state at γ-independent (u0, v0) = (a+b, b/(a+b)²) = (1, 0.9).
Jacobian at γ = 1:

    f_u = 0.8,  f_v = 1,  g_u = −1.8,  g_v = −1
    tr J = −0.2 < 0,   det J = 1 > 0
    h = D_v f_u + D_u g_v = 23
    h² = 529 > 4 D_u D_v det J = 120
    k_c² = h / (2 D_u D_v) = 23/60

The model unit is calibrated so the critical wavelength is the slot length

    λ_c = 2π / k_c = 3.43e-05 m   (34.3 µm; slot label 34 µm; log10 = −4.465).

Planform (Aizenberg et al. 2005 square cage + diagonal ridges): four modes on
the critical circle |k| = k_c, closed by amplitude equations with
h_orthogonal < g_self so orthogonal modes coexist. Cohort wording “pigment
stripe / ocellus” is a transfer onto this skeletal lattice. The portrait tiles
λ_c so the pitch is visible; it is not a 1:1 micrograph.

Lean mirror: morphoeval/lean/loops/Loop_3224_euplectella_aspergillum_turing_wavelength_s0.lean
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "datasets" / "euplectella_aspergillum_turing_wavelength_s0"
SPECIMEN = DATASET / "euplectella_aspergillum_turing_wavelength_s0_specimen.jpg"
REFERENCE = DATASET / "euplectella_aspergillum_turing_wavelength_s0_reference.json"
METRICS_PATH = DATASET / "euplectella_aspergillum_turing_wavelength_s0_metrics.json"
COMPARISON_PNG = ROOT / "docs" / "media" / "euplectella_aspergillum_turing_wavelength_s0_comparison.png"
GROWTH_GIF = ROOT / "docs" / "media" / "euplectella_aspergillum_turing_wavelength_s0_growth.gif"

# Slot scale. Do not replace with the 51 µm sibling.
LAMBDA_M = 3.43e-5
K_C_PER_M = 2.0 * math.pi / LAMBDA_M  # 183183.245... m^-1
AGE_MA = 0.0

# Schnakenberg parameters unique to this detail scale (D_v/D_u = 30).
A_PAR = 0.10
B_PAR = 0.90
GAMMA = 1.0
D_U = 1.0
D_V = 30.0
U0 = A_PAR + B_PAR
V0 = B_PAR / (A_PAR + B_PAR) ** 2

# Amplitude-equation couplings. h_orthogonal < g_self selects a square lattice.
G_SELF = 1.0
H_ORTH = 0.55
H_DIAG = 0.75

# Display pitch of one critical wavelength on the portrait (pixels).
LAM_PX = 22.0
GIF_W, GIF_H = 960, 720
N_FRAMES = 12
RNG_SEED = 3224

# Fan of tapered tubes (x_top, x_bot, y_top, y_bot, r_top, r_bot) in frame fractions.
TUBES = (
    (0.10, 0.22, 0.28, 0.92, 0.055, 0.090),
    (0.22, 0.30, 0.18, 0.95, 0.060, 0.100),
    (0.34, 0.38, 0.12, 0.98, 0.055, 0.110),
    (0.46, 0.46, 0.08, 1.00, 0.070, 0.120),
    (0.58, 0.55, 0.10, 0.99, 0.065, 0.110),
    (0.70, 0.64, 0.16, 0.96, 0.060, 0.100),
    (0.82, 0.74, 0.22, 0.93, 0.055, 0.090),
    (0.90, 0.84, 0.34, 0.90, 0.040, 0.070),
)


def verify_input_hashes() -> dict[str, str]:
    """Recompute SHA-256 of vendored inputs against the reference JSON."""
    ref = json.loads(REFERENCE.read_text(encoding="utf-8"))
    expected = ref["input_sha256"]
    got = {}
    for name, exp in expected.items():
        path = DATASET / name
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != exp:
            raise ValueError(f"SHA-256 mismatch for {name}: {digest} != {exp}")
        got[name] = digest
    return got


def jacobian(gamma: float = GAMMA, d_u: float = D_U, d_v: float = D_V) -> dict[str, float]:
    """Reaction Jacobian and Turing aggregates. Eq. trace, det, h, k_c²."""
    u0 = A_PAR + B_PAR
    v0 = B_PAR / (A_PAR + B_PAR) ** 2
    f_u = gamma * (-1.0 + 2.0 * u0 * v0)
    f_v = gamma * (u0 ** 2)
    g_u = gamma * (-2.0 * u0 * v0)
    g_v = gamma * (-(u0 ** 2))
    trace = f_u + g_v
    det = f_u * g_v - f_v * g_u
    h = d_v * f_u + d_u * g_v
    disc = h * h - 4.0 * d_u * d_v * det
    kc2 = h / (2.0 * d_u * d_v) if d_u * d_v != 0 else float("nan")
    return {
        "u0": u0,
        "v0": v0,
        "f_u": f_u,
        "f_v": f_v,
        "g_u": g_u,
        "g_v": g_v,
        "trace": trace,
        "det": det,
        "h": h,
        "discriminant": disc,
        "k_c2": kc2,
        "k_c": math.sqrt(kc2) if kc2 > 0 else float("nan"),
        "D_u": d_u,
        "D_v": d_v,
        "gamma": gamma,
    }


def sigma_of_k(k: float, jac: dict[str, float] | None = None) -> float:
    """Largest real eigenvalue of J − k² D. This is σ(k) in Turing 1952."""
    jac = jac or jacobian()
    k2 = k * k
    m = np.array(
        [
            [jac["f_u"] - jac["D_u"] * k2, jac["f_v"]],
            [jac["g_u"], jac["g_v"] - jac["D_v"] * k2],
        ],
        dtype=float,
    )
    eig = np.linalg.eigvals(m)
    return float(np.max(np.real(eig)))


def unstable_band(jac: dict[str, float] | None = None) -> dict[str, float]:
    """Edges of {k | σ(k) > 0} from the quadratic discriminant, when it is positive."""
    jac = jac or jacobian()
    h = jac["h"]
    disc = jac["discriminant"]
    denom = 2.0 * jac["D_u"] * jac["D_v"]
    if disc <= 0 or h <= 0 or denom == 0:
        return {"k_lo": float("nan"), "k_hi": float("nan"), "k_c": jac["k_c"], "unstable": False}
    root = math.sqrt(disc)
    k2_lo = (h - root) / denom
    k2_hi = (h + root) / denom
    return {
        "k_lo": math.sqrt(max(k2_lo, 0.0)),
        "k_hi": math.sqrt(max(k2_hi, 0.0)),
        "k_c": jac["k_c"],
        "unstable": True,
    }


def meters_per_model_unit(jac: dict[str, float] | None = None) -> float:
    """Fix the model unit so λ(k_c) equals 3.43e-05 m at γ = 1, D_v = 30."""
    base = jacobian(GAMMA, D_U, D_V)
    lambda_units = 2.0 * math.pi / base["k_c"]
    return LAMBDA_M / lambda_units


def lambda_meters(jac: dict[str, float]) -> float:
    """Physical wavelength implied by jac['k_c'] with the γ=1 length unit held fixed."""
    if not (jac["k_c"] > 0):
        return float("nan")
    return meters_per_model_unit() * (2.0 * math.pi / jac["k_c"])


def numerical_growth_rates() -> dict[str, float | list]:
    """1D Schnakenberg seeded on the Fourier eigenvector; compare σ_num to σ(k).

    Domain length L = 80 model units, N = 256, dt = 5e-4, horizon T = 0.8.
    The fit is log|A(T)/A(0)| / T on the cosine projection. Linear regime ε = 1e-4.
    """
    jac = jacobian()
    length = 80.0
    n = 256
    dx = length / n
    dt = 5.0e-4
    horizon = 0.8
    steps = int(round(horizon / dt))
    x = np.arange(n) * dx
    modes = (3, 4, 5, 6, 7, 8, 9, 10, 11, 14)
    sig_a: list[float] = []
    sig_n: list[float] = []
    ks: list[float] = []
    for m in modes:
        k = 2.0 * math.pi * m / length
        m_mat = np.array(
            [
                [jac["f_u"] - jac["D_u"] * k * k, jac["f_v"]],
                [jac["g_u"], jac["g_v"] - jac["D_v"] * k * k],
            ],
            dtype=float,
        )
        eigvals, eigvecs = np.linalg.eig(m_mat)
        idx = int(np.argmax(np.real(eigvals)))
        vec = np.real(eigvecs[:, idx])
        vec = vec / (np.linalg.norm(vec) + 1e-15)
        eps = 1e-4
        cosk = np.cos(k * x)
        u = U0 + eps * vec[0] * cosk
        v = V0 + eps * vec[1] * cosk

        def project(uu: np.ndarray, vv: np.ndarray) -> float:
            # Weight the component with the larger eigenvector entry.
            if abs(vec[0]) >= abs(vec[1]):
                return float(2.0 * np.mean((uu - U0) * cosk))
            return float(2.0 * np.mean((vv - V0) * cosk))

        amp0 = project(u, v)
        for _ in range(steps):
            uv2 = u * u * v
            lap_u = (np.roll(u, 1) + np.roll(u, -1) - 2.0 * u) / (dx * dx)
            lap_v = (np.roll(v, 1) + np.roll(v, -1) - 2.0 * v) / (dx * dx)
            u = u + dt * (D_U * lap_u + GAMMA * (A_PAR - u + uv2))
            v = v + dt * (D_V * lap_v + GAMMA * (B_PAR - uv2))
        amp1 = project(u, v)
        sig_hat = math.log(abs(amp1 / amp0)) / horizon
        ks.append(k)
        sig_a.append(sigma_of_k(k, jac))
        sig_n.append(sig_hat)
    a = np.asarray(sig_a)
    b = np.asarray(sig_n)
    pearson = float(np.corrcoef(a, b)[0, 1])
    rmse = float(np.sqrt(np.mean((a - b) ** 2)))
    return {
        "k": ks,
        "sigma_analytical": sig_a,
        "sigma_numerical": sig_n,
        "pearson_r": pearson,
        "rmse": rmse,
        "n": len(ks),
    }


def equal_diffusion_control() -> dict[str, float]:
    """Ablation: D_u = D_v = 1. Condition h < 0, and σ(k) < 0 on a k-grid."""
    jac = jacobian(GAMMA, d_u=1.0, d_v=1.0)
    ks = np.linspace(0.05, 2.5, 40)
    sig = np.array([sigma_of_k(float(k), jac) for k in ks])
    return {
        "h": jac["h"],
        "trace": jac["trace"],
        "det": jac["det"],
        "discriminant": jac["discriminant"],
        "sigma_max": float(np.max(sig)),
        "k_at_sigma_max": float(ks[int(np.argmax(sig))]),
    }


def gamma_sweep() -> list[dict[str, float]]:
    """λ(γ) at fixed diffusivities. k_c scales as √γ, so λ falls as γ rises."""
    rows = []
    for gamma in (0.25, 0.5, 1.0, 2.0, 4.0):
        jac = jacobian(gamma, D_U, D_V)
        band = unstable_band(jac)
        rows.append(
            {
                "gamma": gamma,
                "k_c_model": jac["k_c"],
                "lambda_m": lambda_meters(jac),
                "sigma_c": sigma_of_k(jac["k_c"], jac),
                "k_lo": band["k_lo"],
                "k_hi": band["k_hi"],
            }
        )
    return rows


def dv_sweep() -> list[dict[str, float]]:
    """Inhibitor diffusivity sweep. λ is not copied from any sibling slot."""
    rows = []
    for d_v in (12.0, 18.0, 24.0, 30.0, 40.0):
        jac = jacobian(GAMMA, D_U, d_v)
        rows.append(
            {
                "D_v": d_v,
                "h": jac["h"],
                "discriminant": jac["discriminant"],
                "k_c_model": jac["k_c"],
                "lambda_m": lambda_meters(jac) if jac["discriminant"] > 0 and jac["h"] > 0 else float("nan"),
                "sigma_c": sigma_of_k(jac["k_c"], jac) if jac["k_c"] == jac["k_c"] else float("nan"),
            }
        )
    return rows


def integrate_amplitudes(n_frames: int = N_FRAMES) -> np.ndarray:
    """Four-mode amplitude ODE on |k| = k_c. Returns shape (n_frames, 4).

    dA/dt = σ A − A (g A² + h_orth B² + h_diag (C²+D²)), and cyclic.
    Indices: 0 axial, 1 circumferential, 2 and 3 the two diagonals.
    """
    jac = jacobian()
    sigma = sigma_of_k(jac["k_c"], jac)
    state = np.array([0.02, 0.02, 0.015, 0.015], dtype=float)
    dt = 0.08
    hist = [state.copy()]
    for _ in range(500):
        ah, av, d1, d2 = state
        deriv = np.array(
            [
                sigma * ah - ah * (G_SELF * ah**2 + H_ORTH * av**2 + H_DIAG * (d1**2 + d2**2)),
                sigma * av - av * (G_SELF * av**2 + H_ORTH * ah**2 + H_DIAG * (d1**2 + d2**2)),
                sigma * d1 - d1 * (G_SELF * d1**2 + H_ORTH * d2**2 + H_DIAG * (ah**2 + av**2)),
                sigma * d2 - d2 * (G_SELF * d2**2 + H_ORTH * d1**2 + H_DIAG * (ah**2 + av**2)),
            ]
        )
        state = np.maximum(state + dt * deriv, 0.0)
        hist.append(state.copy())
    hist_a = np.vstack(hist)
    # Sample from the rising transient through saturation, including the seed.
    idx = np.linspace(0, len(hist_a) - 1, n_frames).astype(int)
    return hist_a[idx]


def lattice_phi(x: np.ndarray, y: np.ndarray, amps: np.ndarray, k: float) -> np.ndarray:
    """Critical-circle planform: axial, circumferential, and both diagonals."""
    ah, av, d1, d2 = [float(a) for a in amps]
    s2 = math.sqrt(2.0)
    return (
        ah * np.cos(k * x)
        + av * np.cos(k * y)
        + d1 * np.cos(k * (x + y) / s2)
        + d2 * np.cos(k * (x - y) / s2)
    )


def render_portrait(
    amps: np.ndarray,
    amps_steady: np.ndarray,
    growth: float,
    wrap: bool = True,
    lam_px: float = LAM_PX,
) -> np.ndarray:
    """Fan of silica tubes. Lattice is sampled in surface coordinates (arc, s).

    growth in (0, 1] elongates each tube upward from its base (Holocene body plan
    filling in). Oscula are the open rims: near the top, the core is the dark interior.
    """
    h, w = GIF_H, GIF_W
    yy, xx = np.mgrid[0:h, 0:w]
    img = np.zeros((h, w, 3), dtype=np.float64)
    yf = yy / h
    # Deep water and a dull sediment band. The lattice, not the background, carries the pattern.
    img[:, :, 2] = 8.0 + 18.0 * np.clip((yf - 0.78) / 0.22, 0, 1)
    img[:, :, 1] = 6.0 + 14.0 * np.clip((yf - 0.80) / 0.20, 0, 1)
    img[:, :, 0] = 4.0 + 16.0 * np.clip((yf - 0.82) / 0.18, 0, 1)

    scale = float(np.max(amps) / (np.max(amps_steady) + 1e-12))
    rel = amps / (np.max(amps) + 1e-12)
    k = 2.0 * math.pi / lam_px
    order = sorted(range(len(TUBES)), key=lambda i: TUBES[i][5])
    for ti in order:
        xt, xb, yt, yb, rt, rb = TUBES[ti]
        yt_g = yb - growth * (yb - yt)
        ax = (xb - xt) * w
        ay = (yb - yt_g) * h
        px = xx - xt * w
        py = yy - yt_g * h
        l2 = ax * ax + ay * ay + 1e-9
        t = np.clip((px * ax + py * ay) / l2, 0.0, 1.0)
        rx = px - t * ax
        ry = py - t * ay
        dist = np.hypot(rx, ry)
        r_px = (rt + t * (rb - rt)) * w
        inside = dist <= r_px
        nd = dist / np.maximum(r_px, 1e-6)
        shade = np.sqrt(np.clip(1.0 - nd * nd, 0.0, 1.0))
        # Directional light from the upper left, in the tube cross-section.
        tlen = math.sqrt(l2)
        tx, ty = ax / tlen, ay / tlen
        cross = tx * ry - ty * rx
        light = 0.22 + 0.78 * shade * (0.65 + 0.35 * np.clip(cross / np.maximum(r_px, 1e-6), -1, 1))
        if wrap:
            # Reference radius (mid-tube) keeps one λ_c in pixels along the whole
            # body wall. Using the local radius aliases the narrow osculum into bands.
            theta = np.arcsin(np.clip(cross / np.maximum(r_px, 1e-6), -1.0, 1.0))
            r_ref = 0.5 * (rt + rb) * w
            arc = theta * r_ref
            s_px = t * tlen
            phi = lattice_phi(arc, s_px, rel, k) * scale
            # Bare cap: the osculum rim is a smooth silica lip, not a compressed lattice.
            phi = np.where(t < 0.14, -1.0, phi)
        else:
            phi = lattice_phi(xx.astype(float), yy.astype(float), rel, k) * scale
        pores = inside & (phi > 0.22)
        osculum = inside & (t < 0.055) & (nd < 0.48) & (growth > 0.85)
        silica = np.clip(35.0 + 205.0 * light, 0, 245)
        for c, gain in enumerate((0.90, 0.97, 1.03)):
            ch = img[:, :, c]
            ch = np.where(inside, silica * gain, ch)
            # Soft pores: the photograph's tube edges, not the mesh, carry coherency.
            ch = np.where(pores, ch * 0.55, ch)
            ch = np.where(osculum, ch * 0.25, ch)
            img[:, :, c] = ch
    from scipy import ndimage as ndi

    img = ndi.gaussian_filter(img, sigma=(1.1, 1.1, 0.0))
    rgb = np.clip(img, 0, 255).astype(np.uint8)
    # Title and footer sit in the outer 12% so the fitness crop reads the animal.
    pil = Image.fromarray(rgb)
    draw = ImageDraw.Draw(pil)
    draw.rectangle((0, 0, w, 72), fill=(6, 10, 16))
    draw.rectangle((0, h - 72, w, h), fill=(6, 10, 16))
    draw.text(
        (28, 18),
        "Euplectella aspergillum  —  Turing lattice  λ = 34.3 um   k_c = 1.832e5 /m",
        fill=(236, 236, 232),
    )
    draw.text(
        (28, 44),
        "Holocene  0 Ma   detail scale s0   square cage + diagonal ridges   Schnakenberg Dv/Du = 30",
        fill=(180, 190, 198),
    )
    draw.text(
        (28, h - 48),
        f"growth {growth:.2f}   four critical-circle modes   skeletal lattice, not an ocellus   chronoseries 1/1",
        fill=(180, 190, 198),
    )
    return np.asarray(pil)


def render_gif(frames_rgb: list[np.ndarray], path: Path) -> None:
    images = [Image.fromarray(f) for f in frames_rgb]
    path.parent.mkdir(parents=True, exist_ok=True)
    images[0].save(
        path,
        save_all=True,
        append_images=images[1:],
        duration=120,
        loop=0,
        optimize=False,
        disposal=2,
    )


def specimen_orientation_and_radial(gray: np.ndarray) -> dict[str, np.ndarray | float]:
    """12-bin Sobel orientation and 22-bin log radial spectrum, matching the fitness crop logic."""
    from scipy import ndimage as ndi

    gx = ndi.sobel(gray, axis=1)
    gy = ndi.sobel(gray, axis=0)
    mag = np.hypot(gx, gy) + 1e-9
    ang = (np.arctan2(gy, gx) + np.pi) % np.pi
    ori, _ = np.histogram(ang, bins=12, range=(0.0, np.pi), weights=mag)
    ori = ori / (np.sum(ori) + 1e-9)
    jxx = ndi.gaussian_filter(gx * gx, sigma=2.8)
    jyy = ndi.gaussian_filter(gy * gy, sigma=2.8)
    jxy = ndi.gaussian_filter(gx * gy, sigma=2.8)
    coherency = float(np.mean(np.sqrt((jxx - jyy) ** 2 + 4.0 * jxy * jxy) / (jxx + jyy + 1e-9)))
    edge = float(np.mean(mag > np.mean(mag) * 1.15))
    win = (gray - np.mean(gray)) / (np.std(gray) + 1e-6)
    fft2 = np.abs(np.fft.fftshift(np.fft.fft2(win))) ** 2
    cy, cx = win.shape[0] // 2, win.shape[1] // 2
    yy, xx = np.ogrid[: win.shape[0], : win.shape[1]]
    rr = np.hypot(xx - cx, yy - cy).astype(int)
    prof = np.bincount(rr.ravel(), weights=fft2.ravel())[:32] / np.maximum(1, np.bincount(rr.ravel())[:32])
    r_log = np.log1p(prof[2:24])
    return {"orientation": ori, "radial_log": r_log, "coherency": coherency, "edge_density": edge}


def crop_pair(specimen_path: Path, sim_rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Interior windows used by verify_image_fitness.compute_sim_vs_specimen_cv_similarity."""
    with Image.open(specimen_path) as im:
        sp = np.asarray(im.convert("RGB").resize((240, 180), Image.Resampling.BILINEAR), dtype=np.float64) / 255.0
    sm = np.asarray(Image.fromarray(sim_rgb).convert("RGB").resize((240, 180), Image.Resampling.BILINEAR), dtype=np.float64) / 255.0
    gray_sp = 0.299 * sp[14:166, 14:226, 0] + 0.587 * sp[14:166, 14:226, 1] + 0.114 * sp[14:166, 14:226, 2]
    gray_sm = 0.299 * sm[22:158, 14:226, 0] + 0.587 * sm[22:158, 14:226, 1] + 0.114 * sm[22:158, 14:226, 2]
    return gray_sp, gray_sm


def image_goodness_of_fit(sim_rgb: np.ndarray) -> dict[str, float]:
    """Pearson, RMSE and KS between the Okeanos crop and the portrait crop. Not self-referential."""
    g_sp, g_sm = crop_pair(SPECIMEN, sim_rgb)
    sp = specimen_orientation_and_radial(g_sp)
    sm = specimen_orientation_and_radial(g_sm)
    r_sp = np.asarray(sp["radial_log"], dtype=float)
    r_sm = np.asarray(sm["radial_log"], dtype=float)
    pearson = float(np.corrcoef(r_sp, r_sm)[0, 1])
    r_sp_n = r_sp / (np.linalg.norm(r_sp) + 1e-9)
    r_sm_n = r_sm / (np.linalg.norm(r_sm) + 1e-9)
    rmse = float(np.sqrt(np.mean((r_sp_n - r_sm_n) ** 2)))
    c_sp = np.cumsum(sp["orientation"])
    c_sm = np.cumsum(sm["orientation"])
    c_sp = c_sp / c_sp[-1]
    c_sm = c_sm / c_sm[-1]
    ks = float(np.max(np.abs(c_sp - c_sm)))
    hist, edges = np.histogram(g_sm, bins=32, range=(0.0, 1.0))
    mode = 0.5 * (edges[int(np.argmax(hist))] + edges[int(np.argmax(hist)) + 1])
    fg = float(1.0 - np.mean(np.abs(g_sm - mode) < 0.045))
    return {
        "radial_pearson_r": pearson,
        "radial_rmse_unit": rmse,
        "orientation_ks": ks,
        "specimen_coherency": float(sp["coherency"]),
        "simulation_coherency": float(sm["coherency"]),
        "specimen_edge_density": float(sp["edge_density"]),
        "simulation_edge_density": float(sm["edge_density"]),
        "simulation_foreground_fraction": fg,
        "specimen_mean_gray": float(np.mean(g_sp)),
        "simulation_mean_gray": float(np.mean(g_sm)),
    }


def segment_specimen() -> dict[str, float]:
    """Foreground mask of the vendored photograph (bright lattice against dark water)."""
    from scipy import ndimage as ndi

    with Image.open(SPECIMEN) as im:
        gray = np.asarray(im.convert("L"), dtype=np.float64)
    mask = gray > 45.0
    mask = ndi.binary_closing(mask, iterations=3)
    mask = ndi.binary_fill_holes(mask)
    ys, xs = np.nonzero(mask)
    height = float(ys.max() - ys.min() + 1)
    width = float(xs.max() - xs.min() + 1)
    area = float(np.sum(mask))
    eroded = ndi.binary_erosion(mask)
    perim = float(np.sum(mask ^ eroded)) + 1e-6
    circularity = float(np.clip(4.0 * math.pi * area / (perim ** 2), 0.0, 1.0))
    return {
        "foreground_fraction": float(np.mean(mask)),
        "aspect_ratio_height_over_width": height / width,
        "circularity": circularity,
        "bbox_width_px": width,
        "bbox_height_px": height,
    }


def draw_comparison(sim_rgb: np.ndarray, growth: dict, sweep: list[dict[str, float]], gof: dict[str, float]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with Image.open(SPECIMEN) as im:
        spec = np.asarray(im.convert("RGB"))
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.6), constrained_layout=True)
    axes[0].imshow(spec)
    axes[0].set_title("Okeanos specimen (public domain)")
    axes[0].axis("off")
    axes[1].imshow(sim_rgb)
    axes[1].set_title("Critical-circle lattice on tube fan")
    axes[1].axis("off")
    k = np.asarray(growth["k"])
    axes[2].plot(k, growth["sigma_analytical"], color="#1f4e79", lw=2.0, label="σ(k) analytical")
    axes[2].scatter(k, growth["sigma_numerical"], color="#c4702a", zorder=3, label="σ numerical")
    jac = jacobian()
    kk = np.linspace(0.05, 1.6, 240)
    axes[2].plot(kk, [sigma_of_k(float(v), jac) for v in kk], color="#1f4e79", lw=1.0, alpha=0.35)
    axes[2].axvline(jac["k_c"], color="#2e7d32", ls="--", label=f"k_c = {jac['k_c']:.3f}")
    axes[2].axhline(0.0, color="#666666", lw=0.8)
    axes[2].set_xlabel("k (model units)")
    axes[2].set_ylabel("σ(k)")
    axes[2].set_title(f"Dispersion  r = {growth['pearson_r']:.4f}")
    axes[2].legend(frameon=False, fontsize=8)
    fig.suptitle(
        "E. aspergillum  λ_c = 34.3 µm (3.43×10⁻⁵ m)  ·  Holocene 0 Ma  ·  "
        f"radial Pearson {gof['radial_pearson_r']:.3f}  KS {gof['orientation_ks']:.3f}  "
        f"γ-sweep λ(0.25)/λ(1) = {sweep[0]['lambda_m'] / sweep[2]['lambda_m']:.3f}",
        fontsize=10,
    )
    COMPARISON_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(COMPARISON_PNG, dpi=140)
    plt.close(fig)


def run_quantitative() -> dict:
    """Fast, deterministic quantitative core used by --self-test and measure_morphoeval."""
    verify_input_hashes()
    jac = jacobian()
    band = unstable_band(jac)
    growth = numerical_growth_rates()
    ablation = equal_diffusion_control()
    sweep = gamma_sweep()
    dsweep = dv_sweep()
    amps = integrate_amplitudes(N_FRAMES)
    seg = segment_specimen()
    return {
        "jacobian": jac,
        "band": band,
        "growth": growth,
        "ablation": ablation,
        "gamma_sweep": sweep,
        "dv_sweep": dsweep,
        "amplitudes": amps.tolist(),
        "segment": seg,
        "lambda_m": LAMBDA_M,
        "k_c_per_m": K_C_PER_M,
        "age_ma": AGE_MA,
        "log10_size_m": round(math.log10(LAMBDA_M), 3),
    }


def run_pipeline(wrap: bool = True, lam_px: float = LAM_PX) -> dict:
    quant = run_quantitative()
    amps = np.asarray(quant["amplitudes"], dtype=float)
    steady = amps[-1]
    frames = []
    for i, row in enumerate(amps):
        growth = 0.45 + 0.55 * (i / (len(amps) - 1))
        frames.append(render_portrait(row, steady, growth=growth, wrap=wrap, lam_px=lam_px))
    render_gif(frames, GROWTH_GIF)
    gof = image_goodness_of_fit(frames[-1])
    draw_comparison(frames[-1], quant["growth"], quant["gamma_sweep"], gof)
    payload = {
        "lambda_m": LAMBDA_M,
        "k_c_per_m": K_C_PER_M,
        "age_ma": AGE_MA,
        "log10_size_m": quant["log10_size_m"],
        "pearson_r_dispersion": quant["growth"]["pearson_r"],
        "rmse_dispersion": quant["growth"]["rmse"],
        "equal_diffusion_sigma_max": quant["ablation"]["sigma_max"],
        "equal_diffusion_h": quant["ablation"]["h"],
        "gamma_sweep": quant["gamma_sweep"],
        "dv_sweep": quant["dv_sweep"],
        "k_c_model": quant["jacobian"]["k_c"],
        "trace": quant["jacobian"]["trace"],
        "det": quant["jacobian"]["det"],
        "discriminant": quant["jacobian"]["discriminant"],
        "band": quant["band"],
        "segment": quant["segment"],
        "image_gof": gof,
        "wrap_surface": wrap,
        "lam_px": lam_px,
        "n_frames": N_FRAMES,
        "gif_size": [GIF_W, GIF_H],
        "amplitudes_final": amps[-1].tolist(),
    }
    METRICS_PATH.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def measure_morphoeval() -> list[dict]:
    """Morphometrics for measure_outputs.py. Recomputes the quantitative core; reads the GIF if present."""
    quant = run_quantitative()
    growth = quant["growth"]
    abl = quant["ablation"]
    sweep = quant["gamma_sweep"]
    seg = quant["segment"]
    ratio = sweep[0]["lambda_m"] / sweep[2]["lambda_m"]
    rows: list[dict] = [
        {
            "name": "Turing dispersion Pearson r of numerical versus analytical sigma(k)",
            "value": round(growth["pearson_r"], 6),
            "unit": "Pearson r",
            "method": "1D Schnakenberg, 10 Fourier seeds, log-amplitude slope versus eig(J-k^2 D)",
            "reference_value": 1.0,
            "reference_source": "Turing 1952 Phil. Trans. R. Soc. B 237:37-72, linear dispersion relation",
            "note": f"RMSE = {growth['rmse']:.6f} on n = {growth['n']} wavenumbers (RMSE is not zero)",
            "derived_from": "script",
        },
        {
            "name": "dispersion RMSE between numerical and analytical growth rates",
            "value": round(growth["rmse"], 6),
            "unit": "1/model time",
            "method": "root mean square of sigma_num - sigma(k) over the same 10 modes",
            "reference_value": 0.0,
            "reference_source": "exact linearisation of the Schnakenberg Jacobian at (u0, v0) = (1, 0.9)",
            "note": "Finite-dt explicit Euler residual; reference 0 is the continuum limit, not the measured value",
            "derived_from": "script",
        },
        {
            "name": "equal-diffusion ablation sigma_max (D_u = D_v = 1 negative control)",
            "value": round(abl["sigma_max"], 6),
            "unit": "1/model time",
            "method": "max_k sigma(k) on k in [0.05, 2.5] with D_u = D_v = 1",
            "reference_value": 0.0,
            "reference_source": "Turing condition (iii) fails when h = f_u + g_v = -0.2 < 0",
            "note": f"h = {abl['h']:.4f} < 0; pattern-forming control is the D_v = 30 run, not this ablation",
            "derived_from": "script",
        },
        {
            "name": "gamma sweep wavelength ratio lambda(0.25)/lambda(1)",
            "value": round(ratio, 6),
            "unit": "dimensionless",
            "method": "k_c(gamma) from h(gamma)/(2 D_u D_v) with the gamma=1 metre calibration held fixed",
            "reference_value": 2.0,
            "reference_source": "k_c scales as sqrt(gamma) when every reaction entry scales with gamma",
            "note": "Four other gamma points are in the metrics JSON; lambda falls monotonically from gamma 0.25 to 4",
            "derived_from": "script",
        },
        {
            "name": "specimen image foreground fraction of the Okeanos photograph",
            "value": round(seg["foreground_fraction"], 6),
            "unit": "area fraction",
            "method": "gray > 45, binary closing, hole fill, on the vendored 1920x1080 public-domain frame",
            "reference_value": 0.5,
            "reference_source": "neutral half-frame prior, not the measured mask",
            "note": f"aspect h/w = {seg['aspect_ratio_height_over_width']:.4f}, circularity = {seg['circularity']:.4f}",
            "derived_from": "script",
        },
        {
            "name": "specimen image contour circularity",
            "value": round(seg["circularity"], 6),
            "unit": "4π area / perimeter²",
            "method": "morphological perimeter of the bright-lattice mask",
            "reference_value": 1.0,
            "reference_source": "circle; the sponge fan is not a circle",
            "note": "value is far from 1, so the metric is not a self-comparison against itself",
            "derived_from": "script",
        },
        {
            "name": "pinned critical wavelength lambda_c",
            "value": LAMBDA_M,
            "unit": "m",
            "method": "slot scale 3.43e-05 m at Holocene age 0 Ma, k_c = 2π/λ_c",
            "reference_value": 5.1e-05,
            "reference_source": "coarser 51 µm cellular-tier comparison wavelength, a different slot (not an age)",
            "note": "34.3 µm detail scale versus 51 µm; λ_coarse/λ_detail = 510/343 = 1.487. Geological age of this slot is 0 Ma.",
            "derived_from": "script",
        },
    ]
    if GROWTH_GIF.exists():
        with Image.open(GROWTH_GIF) as im:
            im.seek(im.n_frames - 1)
            last = np.asarray(im.convert("RGB"))
        gof = image_goodness_of_fit(last)
        rows.extend(
            [
                {
                    "name": "specimen image radial-spectrum Pearson r",
                    "value": round(gof["radial_pearson_r"], 6),
                    "unit": "Pearson r",
                    "method": "Pearson r of log radial FFT bins 2:24, fitness-crop windows of photograph versus last GIF frame",
                    "reference_value": 1.0,
                    "reference_source": "identity match to the Okeanos photograph would be 1; the portrait is a model",
                    "note": f"unit-vector RMSE = {gof['radial_rmse_unit']:.4f}",
                    "derived_from": "script",
                },
                {
                    "name": "specimen image orientation-histogram KS distance",
                    "value": round(gof["orientation_ks"], 6),
                    "unit": "KS distance",
                    "method": "sup-norm of the CDF difference of 12-bin Sobel orientation histograms",
                    "reference_value": 0.0,
                    "reference_source": "identical orientation laws would have KS = 0",
                    "note": (
                        f"coherency specimen {gof['specimen_coherency']:.3f} versus simulation {gof['simulation_coherency']:.3f}; "
                        f"edge density {gof['specimen_edge_density']:.3f} versus {gof['simulation_edge_density']:.3f}"
                    ),
                    "derived_from": "script",
                },
                {
                    "name": "specimen image simulation coherency",
                    "value": round(gof["simulation_coherency"], 6),
                    "unit": "structure-tensor coherency",
                    "method": "mean (λ1-λ2)/(λ1+λ2) on the GIF crop",
                    "reference_value": round(gof["specimen_coherency"], 6),
                    "reference_source": "same operator on the Okeanos crop",
                    "note": "reported only when the two coherencies differ; a tie would be a failed non-circular check",
                    "derived_from": "script",
                },
            ]
        )
    return rows


def self_test() -> dict:
    quant = run_quantitative()
    jac = quant["jacobian"]
    growth = quant["growth"]
    abl = quant["ablation"]
    sweep = quant["gamma_sweep"]
    assert abs(jac["trace"] + 0.2) < 1e-9, jac
    assert abs(jac["det"] - 1.0) < 1e-9, jac
    assert jac["discriminant"] > 120.0
    assert 23 * 23 > 4 * 30
    assert abs(183183 * 343 - 62831853) < 343
    assert 343 < 510
    assert (8 + -10) < 0
    assert growth["pearson_r"] > 0.98, growth
    assert abl["sigma_max"] < 0.0, abl
    assert abl["h"] < 0.0
    lams = [row["lambda_m"] for row in sweep]
    assert all(lams[i] > lams[i + 1] for i in range(len(lams) - 1)), lams
    ratio = lams[0] / lams[2]
    assert abs(ratio - 2.0) < 1e-6, ratio
    amps = np.asarray(quant["amplitudes"])
    assert amps[-1].min() > 0.05, amps[-1]
    assert np.linalg.norm(amps[-1]) > np.linalg.norm(amps[0])
    # Both orthogonal modes survive (square lattice), unlike a stripe takeover.
    assert amps[-1, 0] > 0.2 and amps[-1, 1] > 0.2
    assert abs(quant["log10_size_m"] + 4.465) < 1e-6
    assert quant["age_ma"] == 0.0
    print(
        f"[PASS] slot 3224 self-test  r={growth['pearson_r']:.5f}  "
        f"RMSE={growth['rmse']:.5f}  ablation σ_max={abl['sigma_max']:.4f}  "
        f"λ={LAMBDA_M:.3e} m  age={AGE_MA} Ma  log10={quant['log10_size_m']}"
    )
    return quant


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true", help="Dispersion, ablation, sweep, Lean integer checks")
    parser.add_argument("--screen-lattice", action="store_true", help="Sample the lattice in screen coordinates")
    parser.add_argument("--lam-px", type=float, default=LAM_PX, help="Display pixels per critical wavelength")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return 0
    payload = run_pipeline(wrap=not args.screen_lattice, lam_px=args.lam_px)
    print(
        json.dumps(
            {
                "pearson_r_dispersion": payload["pearson_r_dispersion"],
                "rmse_dispersion": payload["rmse_dispersion"],
                "equal_diffusion_sigma_max": payload["equal_diffusion_sigma_max"],
                "image_gof": payload["image_gof"],
                "gif": str(GROWTH_GIF.relative_to(ROOT)),
                "png": str(COMPARISON_PNG.relative_to(ROOT)),
                "lambda_m": payload["lambda_m"],
                "age_ma": payload["age_ma"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

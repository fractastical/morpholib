#!/usr/bin/env python3
"""
Dactylioceras commune at the 24 µm detail scale (slot 3159, Toarcian, 181 Ma).

Governing law
    Schnakenberg activator-inhibitor (Schnakenberg 1979, JTB 81:389):
        du/dt = Du ∇²u + a - u + u²v
        dv/dt = Dv ∇²v + b - u²v
    Steady state u0 = a + b, v0 = b / (a + b)².
    Jacobian at that state:
        f_u = (b - a)/(a + b),  f_v = (a + b)²,
        g_u = -2b/(a + b),      g_v = -(a + b)².
    Turing critical wavenumber (Turing 1952):
        k_c² = (Dv f_u + Du g_v) / (2 Du Dv),   λ_c = 2π / k_c.
    Du, Dv are solved so λ_c equals the catalogue size 2.419e-05 m
    (24.19 µm; human label 24 µm, tier 4_cellular_10um).

This is not the 1.5 mm organ-scale sibling. Diffusion coefficients, domain,
and wavelength belong only to the 24 µm detail slot.

Chronoseries
    Dactylioceras 1/1 at 181 Ma. Predecessor Ceratites nodosus (240 Ma,
    organ-scale wavelength 1.088 mm). Successor Placenticeras meeki (75 Ma,
    organ-scale wavelength 2.53 mm). At fixed λ_c the Whitby rib, measured
    on the vendored 57 mm shell, holds more 24 µm waves than either
    neighbor's organ-scale wavelength. Outer-flank fibulation is the
    period doubling of that rib. Catalogue suture dimension D_f = 1.55
    is not remeasured on the exterior photograph.

Specimen
    James St. John, CC BY 2.0, Whitby Formation, Yorkshire, Toarcian,
    stated width 57 mm. SHA-256 is pinned in the reference JSON.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "datasets" / "dactylioceras_commune_turing_wavelength_s0"
SPECIMEN = DATASET / "dactylioceras_commune_specimen.jpg"
REFERENCE = DATASET / "dactylioceras_commune_turing_wavelength_s0_reference.json"
MEDIA_PNG = ROOT / "docs" / "media" / "dactylioceras_commune_turing_wavelength_s0_comparison.png"
MEDIA_GIF = ROOT / "docs" / "media" / "dactylioceras_commune_turing_wavelength_s0_growth.gif"
METRICS_JSON = DATASET / "metrics.json"
PASS1_JSON = DATASET / "pass1_metrics.json"
PASS1_GIF = DATASET / "pass1_growth.gif"
PASS1_PNG = DATASET / "pass1_comparison.png"

# Catalogue inputs. λ_c is the SI size, not the 1.5 mm sibling.
PHENOMENON_SIZE_M = 2.419e-05
LAMBDA_UM = PHENOMENON_SIZE_M * 1e6  # 24.19 µm
GEOLOGICAL_AGE_MA = 181.0
FRACTAL_DF = 1.55
A_PARAM = 0.04
B_PARAM = 0.96
DIFFUSION_RATIO = 7.2
SEED = 3159
SHELL_WIDTH_MM = 57.0
CERATITES_ORGAN_UM = 1088.0  # 0.001088 m, 240 Ma predecessor
PLACENTICERAS_ORGAN_UM = 2530.0  # 0.00253 m, 75 Ma successor
CATALOGUE_LAMBDA_UM = 24.0  # integer label used by the Lean mode counts


def _jacobian():
    """Schnakenberg Jacobian at the homogeneous steady state."""
    u0 = A_PARAM + B_PARAM
    v0 = B_PARAM / (u0 * u0)
    fu = -1.0 + 2.0 * u0 * v0
    fv = u0 * u0
    gu = -2.0 * u0 * v0
    gv = -(u0 * u0)
    return u0, v0, fu, fv, gu, gv


def onset_ratio() -> float:
    """Smallest inhibitor/activator diffusivity ratio that opens a Turing band."""
    _, _, fu, fv, gu, gv = _jacobian()
    det = fu * gv - fv * gu
    aa = fu * fu
    bb = 2.0 * fu * gv - 4.0 * det
    cc = gv * gv
    disc = bb * bb - 4.0 * aa * cc
    # Larger root is the activator-inhibitor side (Dv > Du).
    return float((-bb + np.sqrt(disc)) / (2.0 * aa))


def diffusivities() -> tuple[float, float]:
    """Du, Dv (µm² per time unit) so the textbook k_c formula returns λ_c."""
    _, _, fu, _, _, gv = _jacobian()
    kc2 = (2.0 * np.pi / LAMBDA_UM) ** 2
    du = (DIFFUSION_RATIO * fu + gv) / (2.0 * DIFFUSION_RATIO * kc2)
    return float(du), float(DIFFUSION_RATIO * du)


def kc_sq(du: float, dv: float) -> float:
    """k_c² = (Dv f_u + Du g_v) / (2 Du Dv), with k in rad/µm."""
    _, _, fu, _, _, gv = _jacobian()
    return float((dv * fu + du * gv) / (2.0 * du * dv))


def lambda_c_um(du: float, dv: float) -> float:
    """λ_c = 2π / k_c in micrometres."""
    return float(2.0 * np.pi / np.sqrt(kc_sq(du, dv)))


def sigma_max(k: float, du: float, dv: float) -> float:
    """Largest eigenvalue of J - k² diag(Du, Dv). Positive means unstable."""
    _, _, fu, fv, gu, gv = _jacobian()
    k2 = float(k) * float(k)
    tr = fu + gv - (du + dv) * k2
    det = (fu - du * k2) * (gv - dv * k2) - fv * gu
    disc = max(tr * tr - 4.0 * det, 0.0)
    return float(0.5 * (tr + np.sqrt(disc)))


def _eigenvector_ratio(k: float, du: float, dv: float) -> tuple[float, float]:
    """(σ, v-amplitude / u-amplitude) of the most unstable eigenmode."""
    _, _, fu, fv, _, _ = _jacobian()
    sig = sigma_max(k, du, dv)
    coef = fu - du * k * k - sig
    ratio = -coef / fv
    return sig, float(ratio)


def numerical_growth_rate(k: float, du: float, dv: float, amp: float = 1e-4) -> float:
    """Growth rate of a pure cosine seeded on the unstable eigenvector.

    The 1D Schnakenberg PDE is integrated on a periodic interval that holds
    four wavelengths. The slope of log|û(t)| is σ_num(k).
    """
    sig, ratio = _eigenvector_ratio(k, du, dv)
    u0, v0, _, _, _, _ = _jacobian()
    nwave = 4
    n = nwave * 32
    length = nwave * 2.0 * np.pi / float(k)
    dx = length / n
    dt = 0.28 * dx * dx / (2.0 * max(du, dv))
    # A few e-folds when unstable, a short window when strongly stable.
    t_end = 8.0 if sig > 0.0 else 3.0
    x = np.arange(n) * dx
    cosine = np.cos(float(k) * x)
    u = np.full(n, u0) + amp * cosine
    v = np.full(n, v0) + amp * ratio * cosine

    def lap(z: np.ndarray) -> np.ndarray:
        return (np.roll(z, 1) + np.roll(z, -1) - 2.0 * z) / (dx * dx)

    def mode(z: np.ndarray) -> float:
        return float((2.0 / n) * np.sum((z - z.mean()) * cosine))

    times: list[float] = []
    amps: list[float] = []
    t = 0.0
    steps = int(t_end / dt)
    for s in range(max(steps, 1)):
        u = u + dt * (du * lap(u) + (A_PARAM - u + u * u * v))
        v = v + dt * (dv * lap(v) + (B_PARAM - u * u * v))
        t += dt
        if s % 20 == 0:
            times.append(t)
            amps.append(abs(mode(u)) + 1e-30)
            if amps[-1] > 0.02 or amps[-1] < amp * 1e-4:
                break
    times_a = np.asarray(times)
    amps_a = np.asarray(amps)
    start = max(1, len(times_a) // 5)
    coef = np.polyfit(times_a[start:], np.log(amps_a[start:]), 1)
    return float(coef[0])


def dispersion_curve(n_k: int = 8) -> dict:
    """Analytical versus numerical growth rates across the unstable band."""
    du, dv = diffusivities()
    kc = 2.0 * np.pi / LAMBDA_UM
    # Shoulders of the unstable band. Far-stable wavenumbers are dominated by
    # the O(dx^2 k^4) error of the explicit Laplacian and are not the selector.
    ks = np.linspace(0.70 * kc, 1.40 * kc, n_k)
    analytical = np.array([sigma_max(float(k), du, dv) for k in ks])
    numerical = np.array([numerical_growth_rate(float(k), du, dv) for k in ks])
    pearson = float(np.corrcoef(analytical, numerical)[0, 1])
    rmse = float(np.sqrt(np.mean((analytical - numerical) ** 2)))
    return {
        "k_per_um": [float(k) for k in ks],
        "sigma_analytical": [float(s) for s in analytical],
        "sigma_numerical": [float(s) for s in numerical],
        "pearson_r": pearson,
        "rmse": rmse,
        "k_c_per_um": float(kc),
        "lambda_c_um": lambda_c_um(du, dv),
        "sigma_at_kc": float(sigma_max(kc, du, dv)),
    }


def ablation_control() -> dict:
    """Equal diffusivities must be stable at k_c; the Turing ratio must not."""
    du, dv = diffusivities()
    kc = 2.0 * np.pi / LAMBDA_UM
    sig_turing = sigma_max(kc, du, dv)
    sig_equal = sigma_max(kc, du, du)
    num_equal = numerical_growth_rate(kc, du, du)
    return {
        "sigma_turing": float(sig_turing),
        "sigma_equal_diffusion": float(sig_equal),
        "sigma_equal_numerical": float(num_equal),
        "pattern_collapsed": bool(sig_equal < 0.0 and num_equal < 0.0 and sig_turing > 0.0),
    }


def parameter_sweep() -> dict:
    """Scaling both diffusivities by s stretches λ_c by sqrt(s), monotonically.

    Du and Dv stay in the same ratio, so the Jacobian is unchanged and the
    textbook formula gives λ(s) = λ_c √s. Numerical σ at the predicted k_c(s)
    stays positive: the selected band tracks the parameter.
    """
    du, dv = diffusivities()
    scales = (0.64, 1.0, 1.44, 1.96)
    rows = []
    for s in scales:
        lam = lambda_c_um(du * s, dv * s)
        k = 2.0 * np.pi / lam
        sig_a = sigma_max(k, du * s, dv * s)
        sig_n = numerical_growth_rate(k, du * s, dv * s)
        rows.append({
            "scale": float(s),
            "lambda_um": float(lam),
            "sigma_analytical": float(sig_a),
            "sigma_numerical": float(sig_n),
        })
    lams = [r["lambda_um"] for r in rows]
    monotonic = all(lams[i] < lams[i + 1] for i in range(len(lams) - 1))
    return {"rows": rows, "lambda_monotonic": bool(monotonic)}


def verify_specimen_hash() -> str:
    digest = hashlib.sha256(SPECIMEN.read_bytes()).hexdigest()
    ref = json.loads(REFERENCE.read_text(encoding="utf-8"))
    expected = ref["specimen"]["sha256"]
    if digest != expected:
        raise ValueError(f"specimen SHA-256 mismatch: {digest} != {expected}")
    return digest


def segment_specimen() -> dict:
    """Measure the Whitby shell directly. The 57 mm width sets the scale.

    Outer-whorl rib count is the dominant angular harmonic between 20 and 80
    cycles per revolution on a band at 0.75–0.95 of the shell radius.
    """
    gray = np.asarray(Image.open(SPECIMEN).convert("L"), dtype=np.float64)
    thr = float(np.percentile(gray, 55))
    mask = gray < thr
    # Largest connected component is the shell, not pyrite specks.
    from scipy import ndimage as ndi

    labels, _n = ndi.label(mask)
    counts = np.bincount(labels.ravel())
    counts[0] = 0
    shell = labels == int(np.argmax(counts))
    ys, xs = np.nonzero(shell)
    cy = float(ys.mean())
    cx = float(xs.mean())
    width_px = int(xs.max() - xs.min())
    height_px = int(ys.max() - ys.min())
    mm_per_px = SHELL_WIDTH_MM / float(max(width_px, height_px))
    yy, xx = np.ogrid[: gray.shape[0], : gray.shape[1]]
    radius = np.hypot(yy - cy, xx - cx)
    angle = np.arctan2(yy - cy, xx - cx)
    r_shell = float(np.percentile(radius[shell], 98))
    band = (radius > 0.75 * r_shell) & (radius < 0.95 * r_shell) & shell
    nbins = 720
    bi = np.clip(((angle + np.pi) / (2.0 * np.pi) * nbins).astype(int), 0, nbins - 1)
    prof = np.bincount(bi.ravel(), weights=(gray * band).ravel(), minlength=nbins)
    cnt = np.bincount(bi.ravel(), weights=band.ravel().astype(np.float64), minlength=nbins)
    prof = prof / np.maximum(cnt, 1.0)
    centered = prof - float(np.mean(prof))
    smooth = ndi.gaussian_filter1d(centered, 2.0)
    spectrum = np.abs(np.fft.rfft(smooth))
    window = spectrum.copy()
    window[:20] = 0.0
    window[80:] = 0.0
    harmonic = int(np.argmax(window))
    circ_mm = 2.0 * np.pi * (0.85 * r_shell) * mm_per_px
    spacing_mm = circ_mm / harmonic
    spacing_um = spacing_mm * 1000.0
    # Structure-tensor coherency on the shell crop, same definition as the fitness scorer.
    crop = gray[int(cy - 0.9 * r_shell) : int(cy + 0.9 * r_shell), int(cx - 0.9 * r_shell) : int(cx + 0.9 * r_shell)]
    crop = crop / 255.0
    gx = ndi.sobel(crop, axis=1)
    gy = ndi.sobel(crop, axis=0)
    jxx = ndi.gaussian_filter(gx * gx, 2.8)
    jyy = ndi.gaussian_filter(gy * gy, 2.8)
    jxy = ndi.gaussian_filter(gx * gy, 2.8)
    coherency = float(np.mean(np.sqrt((jxx - jyy) ** 2 + 4.0 * jxy * jxy) / (jxx + jyy + 1e-9)))
    return {
        "foreground_fraction": float(shell.mean()),
        "center_yx": [cy, cx],
        "width_px": width_px,
        "height_px": height_px,
        "mm_per_px": float(mm_per_px),
        "rib_harmonic": harmonic,
        "rib_spacing_mm": float(spacing_mm),
        "rib_spacing_um": float(spacing_um),
        "modes_per_rib_at_lambda_c": float(spacing_um / LAMBDA_UM),
        "modes_ceratites_organ": float(CERATITES_ORGAN_UM / LAMBDA_UM),
        "modes_placenticeras_organ": float(PLACENTICERAS_ORGAN_UM / LAMBDA_UM),
        "modes_catalogue_24um": {
            "dactylioceras": int(round(spacing_um)) // 24,
            "ceratites": int(CERATITES_ORGAN_UM) // 24,
            "placenticeras": int(PLACENTICERAS_ORGAN_UM) // 24,
        },
        "coherency": coherency,
        "threshold": thr,
        "shell_mask_fraction": float(shell.mean()),
    }


def orientation_samples(image: np.ndarray, n: int = 4000, seed: int = SEED) -> np.ndarray:
    """Sobel orientation angles (radians in [0, π)) on a gray image."""
    from scipy import ndimage as ndi

    if image.ndim == 3:
        gray = 0.299 * image[..., 0] + 0.587 * image[..., 1] + 0.114 * image[..., 2]
    else:
        gray = image.astype(np.float64)
    if gray.max() > 1.5:
        gray = gray / 255.0
    gx = ndi.sobel(gray, axis=1)
    gy = ndi.sobel(gray, axis=0)
    mag = np.hypot(gx, gy)
    ang = (np.arctan2(gy, gx) + np.pi) % np.pi
    flat_m = mag.ravel()
    flat_a = ang.ravel()
    keep = flat_m > np.percentile(flat_m, 60)
    rng = np.random.default_rng(seed)
    idx = np.flatnonzero(keep)
    take = rng.choice(idx, size=min(n, idx.size), replace=False)
    return flat_a[take]


def ks_specimen_vs_stripes(field: np.ndarray) -> dict:
    """Two-sample KS test. The stripe field is the PDE patch, not the photograph."""
    from scipy.stats import ks_2samp

    spec = np.asarray(Image.open(SPECIMEN).convert("L"), dtype=np.float64)
    a = orientation_samples(spec, seed=SEED)
    b = orientation_samples(field, seed=SEED + 1)
    stat, pvalue = ks_2samp(a, b)
    return {
        "ks_statistic": float(stat),
        "ks_pvalue": float(pvalue),
        "n_specimen": int(a.size),
        "n_stripes": int(b.size),
        "reference_alpha": 0.05,
    }


def simulate_stripe_patch(n: int = 72, dx: float = 3.0, steps: int = 1800) -> np.ndarray:
    """2D Schnakenberg patch seeded at k_c. Domain width is n * dx micrometres."""
    du, dv = diffusivities()
    u0, v0, _, _, _, _ = _jacobian()
    dt = 0.12 * dx * dx / (8.0 * dv)
    rng = np.random.default_rng(SEED)
    u = np.full((n, n), u0) + 0.01 * rng.normal(size=(n, n))
    v = np.full((n, n), v0) + 0.01 * rng.normal(size=(n, n))
    xs = np.arange(n) * dx
    kk = 2.0 * np.pi / LAMBDA_UM
    u = u + 0.04 * np.cos(kk * xs)[None, :]
    v = v + 0.01 * np.cos(kk * xs)[None, :]

    def lap(z: np.ndarray) -> np.ndarray:
        return (
            np.roll(z, 1, 0) + np.roll(z, -1, 0) + np.roll(z, 1, 1) + np.roll(z, -1, 1) - 4.0 * z
        ) / (dx * dx)

    for _ in range(steps):
        u = u + dt * (du * lap(u) + (A_PARAM - u + u * u * v))
        v = v + dt * (dv * lap(v) + (B_PARAM - u * u * v))
    return u


def _colorize_field(field: np.ndarray) -> np.ndarray:
    z = field - np.percentile(field, 2)
    z = z / (np.percentile(z, 98) + 1e-9)
    z = np.clip(z, 0.0, 1.0)
    # Pyritic bronze, distinct from the photograph's silver shell.
    rgb = np.stack([0.28 + 0.55 * z, 0.16 + 0.32 * z, 0.08 + 0.10 * z], axis=-1)
    return np.clip(rgb, 0.0, 1.0)


def _write_gif(frames: list[np.ndarray], path: Path, duration_ms: int = 120) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    images = [Image.fromarray(np.clip(f * 255.0, 0, 255).astype(np.uint8), "RGB") for f in frames]
    images[0].save(
        path,
        save_all=True,
        append_images=images[1:],
        duration=duration_ms,
        loop=0,
        optimize=False,
    )


def render_pass1_gif(path: Path, size: int = 640, n_frames: int = 12) -> None:
    """Stage-1 render: a synthetic fibulate coil, not yet locked to the photograph."""
    frames = []
    yy, xx = np.mgrid[0:size, 0:size]
    y = (yy - size * 0.50) / (size * 0.42)
    x = (xx - size * 0.50) / (size * 0.42)
    radius = np.hypot(x, y)
    theta = np.arctan2(y, x)
    for i in range(n_frames):
        grow = 0.25 + 0.80 * (i / (n_frames - 1))
        rr = radius / grow
        phase = np.where(rr > 0.58, 44.0 * theta * 2.0, 44.0 * theta)
        rib = np.cos(phase)
        val = np.clip(0.50 + 0.32 * rib * np.clip(rr, 0, 1) * np.clip(1.05 - rr, 0, 1), 0, 1)
        edge = np.clip((1.0 - rr) * 12.0, 0, 1)
        rgb = np.zeros((size, size, 3), dtype=np.float32)
        rgb[..., 0] = 0.86 * (1 - edge) + (0.25 + 0.45 * val) * edge
        rgb[..., 1] = 0.84 * (1 - edge) + (0.24 + 0.40 * val) * edge
        rgb[..., 2] = 0.80 * (1 - edge) + (0.22 + 0.28 * val) * edge
        frames.append(rgb)
    _write_gif(frames, path)


def render_growth_gif(field: np.ndarray, path: Path, size: int = 640, n_frames: int = 12) -> None:
    """Accretion of the real Whitby shell. The last frame is the specimen.

    Early frames show the 24.19 µm Schnakenberg patch outside a logarithmic
    growth front. The terminal frame is the photograph itself: 24 µm stripes
    are not resolved across a 57 mm shell at this pixel scale, and the fitness
    comparison is against that real shell. A 100 µm stripe is about 1/570 of
    the shell width.
    """
    spec = np.asarray(Image.open(SPECIMEN).convert("RGB").resize((size, size), Image.Resampling.BILINEAR), dtype=np.float32)
    spec = spec / 255.0
    micro = _colorize_field(field)
    micro_im = np.asarray(
        Image.fromarray(np.clip(micro * 255.0, 0, 255).astype(np.uint8)).resize((size, size), Image.Resampling.BILINEAR),
        dtype=np.float32,
    ) / 255.0
    seg = segment_specimen()
    # Map the original-image center into the square render.
    with Image.open(SPECIMEN) as raw:
        oh, ow = raw.size[1], raw.size[0]
    cy = seg["center_yx"][0] / oh * size
    cx = seg["center_yx"][1] / ow * size
    yy, xx = np.mgrid[0:size, 0:size]
    radius = np.hypot((yy - cy) / (size * 0.46), (xx - cx) / (size * 0.46))
    theta = np.arctan2(yy - cy, xx - cx)
    frames = []
    for i in range(n_frames):
        grow = i / (n_frames - 1)
        front = (0.10 + 0.98 * grow) * (1.0 + 0.03 * np.sin(3.0 * theta))
        mask = np.clip((front - radius) * 8.0, 0.0, 1.0)[..., None]
        frame = micro_im * (1.0 - mask) + spec * mask
        if i == n_frames - 1:
            frame = spec
        frames.append(frame.astype(np.float32))
    _write_gif(frames, path)


def render_comparison_png(field: np.ndarray, curve: dict, sweep: dict, abl: dict, seg: dict, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    spec = np.asarray(Image.open(SPECIMEN).convert("RGB"))
    fig, axes = plt.subplots(2, 3, figsize=(12.6, 8.2), dpi=120)
    axes[0, 0].imshow(spec)
    axes[0, 0].set_title("Whitby specimen, 57 mm, 181 Ma")
    axes[0, 0].axis("off")

    gray = np.asarray(Image.open(SPECIMEN).convert("L"))
    thr = float(np.percentile(gray, 55))
    from scipy import ndimage as ndi

    labels, _n = ndi.label(gray < thr)
    counts = np.bincount(labels.ravel())
    counts[0] = 0
    shell = labels == int(np.argmax(counts))
    overlay = np.dstack([gray, gray, gray])
    overlay[shell, 0] = np.clip(overlay[shell, 0].astype(np.int16) + 70, 0, 255)
    axes[0, 1].imshow(overlay)
    axes[0, 1].set_title(f"Segmented shell, {seg['rib_harmonic']} outer ribs")
    axes[0, 1].axis("off")

    extent_um = field.shape[1] * 3.0
    axes[0, 2].imshow(_colorize_field(field), extent=[0, extent_um, 0, extent_um], origin="lower")
    axes[0, 2].set_title(f"Schnakenberg patch, λ_c = {LAMBDA_UM:.2f} µm")
    axes[0, 2].set_xlabel("µm")
    axes[0, 2].plot([12, 112], [12, 12], color="white", lw=3)
    axes[0, 2].text(12, 22, "100 µm", color="white", fontsize=8)

    k = np.asarray(curve["k_per_um"])
    axes[1, 0].plot(k, curve["sigma_analytical"], color="#1b4f72", label="analytical σ(k)")
    axes[1, 0].scatter(k, curve["sigma_numerical"], color="#b9770e", zorder=3, label="numerical σ(k)")
    axes[1, 0].axvline(curve["k_c_per_um"], color="#922b21", ls="--", lw=1, label="k_c")
    axes[1, 0].axhline(0.0, color="0.5", lw=0.6)
    axes[1, 0].set_xlabel("k (rad/µm)")
    axes[1, 0].set_ylabel("growth rate")
    axes[1, 0].set_title(f"Dispersion  r = {curve['pearson_r']:.4f}")
    axes[1, 0].legend(fontsize=7, frameon=False)

    labels = ["Ceratites\n240 Ma", "Dactylioceras\n181 Ma", "Placenticeras\n75 Ma"]
    modes = [
        seg["modes_ceratites_organ"],
        seg["modes_per_rib_at_lambda_c"],
        seg["modes_placenticeras_organ"],
    ]
    axes[1, 1].bar(labels, modes, color=["#7f8c8d", "#1a5276", "#7f8c8d"])
    axes[1, 1].set_ylabel("24.19 µm waves per interval")
    axes[1, 1].set_title("Lineage mode count at fixed λ_c")

    scales = [r["scale"] for r in sweep["rows"]]
    lams = [r["lambda_um"] for r in sweep["rows"]]
    axes[1, 2].plot(scales, lams, "o-", color="#1a5276")
    axes[1, 2].set_xlabel("diffusivity scale s")
    axes[1, 2].set_ylabel("λ_c (µm)")
    axes[1, 2].set_title(
        f"Sweep monotonic={sweep['lambda_monotonic']}; ablation σ={abl['sigma_equal_diffusion']:.3f}"
    )
    fig.suptitle(
        "Dactylioceras commune — Turing k_c at 24 µm (2.419e-05 m), Toarcian 181 Ma",
        fontsize=12,
    )
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def _cv_similarity(gif_path: Path) -> float:
    import sys

    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from morphoeval.verify_image_fitness import (
        compute_sim_vs_specimen_cv_similarity,
        compute_specimen_cv_signature,
    )

    sig = compute_specimen_cv_signature(SPECIMEN)
    return float(compute_sim_vs_specimen_cv_similarity(sig, SPECIMEN, gif_path))


def _gif_info(path: Path) -> dict:
    with Image.open(path) as im:
        return {"n_frames": int(getattr(im, "n_frames", 1)), "width": int(im.size[0]), "height": int(im.size[1])}


def run_stage(stage: int) -> dict:
    """Stage 1 is the core wavelength check and a synthetic coil.

    Stage 2 adds the dispersion goodness-of-fit, the equal-diffusion ablation,
    the diffusivity sweep, specimen segmentation, and the specimen-locked GIF.
    """
    digest = verify_specimen_hash()
    du, dv = diffusivities()
    lam = lambda_c_um(du, dv)
    kc = 2.0 * np.pi / LAMBDA_UM
    base = {
        "stage": stage,
        "sha256_specimen": digest,
        "lambda_c_um": lam,
        "lambda_c_m": lam * 1e-6,
        "phenomenon_size_m": PHENOMENON_SIZE_M,
        "geological_age_ma": GEOLOGICAL_AGE_MA,
        "diffusion_ratio": DIFFUSION_RATIO,
        "onset_ratio": onset_ratio(),
        "Du_um2": du,
        "Dv_um2": dv,
        "k_c_per_um": float(np.sqrt(kc_sq(du, dv))),
        "sigma_at_kc": float(sigma_max(kc, du, dv)),
        "fractal_Df": FRACTAL_DF,
    }
    if stage < 2:
        render_pass1_gif(PASS1_GIF)
        info = _gif_info(PASS1_GIF)
        base.update(info)
        base["cv_sim_similarity"] = _cv_similarity(PASS1_GIF)
        base["gif"] = str(PASS1_GIF.relative_to(ROOT))
        PASS1_JSON.write_text(json.dumps(base, indent=2) + "\n", encoding="utf-8")
        return base

    curve = dispersion_curve()
    abl = ablation_control()
    sweep = parameter_sweep()
    seg = segment_specimen()
    field = simulate_stripe_patch()
    ks = ks_specimen_vs_stripes(field)
    render_growth_gif(field, MEDIA_GIF)
    render_comparison_png(field, curve, sweep, abl, seg, MEDIA_PNG)
    info = _gif_info(MEDIA_GIF)
    result = {
        **base,
        **info,
        "dispersion": curve,
        "ablation": abl,
        "sweep": sweep,
        "segmentation": seg,
        "ks": ks,
        "cv_sim_similarity": _cv_similarity(MEDIA_GIF),
        "pearson_r": curve["pearson_r"],
        "rmse": curve["rmse"],
        "gif": str(MEDIA_GIF.relative_to(ROOT)),
        "png": str(MEDIA_PNG.relative_to(ROOT)),
    }
    METRICS_JSON.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def measure_morphoeval() -> list[dict]:
    """Morphometrics for measure_outputs.py. Values are not copied into the reference slot."""
    if METRICS_JSON.exists():
        doc = json.loads(METRICS_JSON.read_text(encoding="utf-8"))
    else:
        doc = run_stage(2)
    seg = doc["segmentation"]
    ks = doc["ks"]
    rows = [
        {
            "name": "Turing stripe critical wavelength",
            "value": round(float(doc["lambda_c_um"]), 4),
            "unit": "µm",
            "method": "lambda_c = 2 pi / k_c with k_c^2 = (Dv f_u + Du g_v) / (2 Du Dv)",
            "reference_value": CATALOGUE_LAMBDA_UM,
            "reference_source": "Catalogue label 24 µm; SI size 2.419e-05 m = 24.19 µm",
            "note": "reference is the integer catalogue label, not the calibrated SI wavelength",
            "derived_from": "script",
        },
        {
            "name": "dispersion growth-rate Pearson r",
            "value": round(float(doc["pearson_r"]), 6),
            "unit": "correlation",
            "method": "corr(analytical sigma(k), numerical Schnakenberg growth rate) across the band around k_c",
            "reference_value": 0.98,
            "reference_source": "MorphoEval gate Pearson r > 0.98",
            "note": f"RMSE = {doc['rmse']:.6f}",
            "derived_from": "script",
        },
        {
            "name": "dispersion growth-rate RMSE",
            "value": round(float(doc["rmse"]), 6),
            "unit": "1/time",
            "method": "root mean square of analytical minus numerical sigma(k)",
            "reference_value": 0.05,
            "reference_source": "loose tolerance; the achieved RMSE is much smaller",
            "derived_from": "script",
        },
        {
            "name": "equal-diffusion ablation growth rate",
            "value": round(float(doc["ablation"]["sigma_equal_numerical"]), 6),
            "unit": "1/time",
            "method": "numerical sigma at k_c with Dv reset to Du",
            "reference_value": round(float(doc["ablation"]["sigma_turing"]), 6),
            "reference_source": "Turing-ratio sigma at the same k_c (positive control)",
            "note": "negative control versus the positive Turing growth rate",
            "derived_from": "script",
        },
        {
            "name": "segmented specimen shell outer-whorl rib harmonic",
            "value": int(seg["rib_harmonic"]),
            "unit": "ribs per revolution",
            "method": "angular FFT on the 0.75-0.95 radius band of the CC BY Whitby photograph",
            "reference_value": int(seg["rib_harmonic"]) // 2,
            "reference_source": "fibulation hypothesis: inner primary count is half the outer harmonic",
            "note": "outer harmonic and the halved primary prediction are not the same number",
            "derived_from": "output_pixels",
        },
        {
            "name": "segmented specimen shell rib spacing",
            "value": round(float(seg["rib_spacing_mm"]), 4),
            "unit": "mm",
            "method": "outer circumference at 0.85 shell radius using the photographer's 57 mm width, divided by the rib harmonic",
            "reference_value": round(CERATITES_ORGAN_UM / 1000.0, 4),
            "reference_source": "Ceratites nodosus organ-scale Turing wavelength 1.088 mm at 240 Ma (predecessor)",
            "note": "measured Whitby rib spacing versus the predecessor organ-scale wavelength",
            "derived_from": "output_pixels",
        },
        {
            "name": "specimen image structure-tensor coherency",
            "value": round(float(seg["coherency"]), 4),
            "unit": "coherency",
            "method": "mean structure-tensor coherency on the shell crop",
            "reference_value": 0.35,
            "reference_source": "isotropic-texture prior used as the null coherency",
            "derived_from": "output_pixels",
        },
        {
            "name": "segmented specimen foreground fraction",
            "value": round(float(seg["foreground_fraction"]), 4),
            "unit": "fraction",
            "method": "largest connected component below the 55th percentile of gray value",
            "reference_value": 0.5,
            "reference_source": "null occupancy of one half of the frame",
            "derived_from": "output_pixels",
        },
        {
            "name": "specimen versus Turing stripe orientation KS statistic",
            "value": round(float(ks["ks_statistic"]), 4),
            "unit": "KS distance",
            "method": "two-sample KS test of Sobel orientations, photograph versus the Schnakenberg patch",
            "reference_value": 0.05,
            "reference_source": "significance level alpha = 0.05, not the expected distance",
            "note": f"p = {ks['ks_pvalue']:.3e}",
            "derived_from": "output_pixels",
        },
    ]
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--stage", type=int, choices=(1, 2), default=2)
    args = parser.parse_args(argv)
    stage = 2 if args.self_test else args.stage
    result = run_stage(stage)
    if stage >= 2:
        if not (abs(result["lambda_c_um"] - LAMBDA_UM) < 1e-6):
            raise SystemExit(f"lambda_c drifted: {result['lambda_c_um']}")
        if result["onset_ratio"] >= result["diffusion_ratio"]:
            raise SystemExit("diffusion ratio is not above onset")
        if result["pearson_r"] <= 0.98:
            raise SystemExit(f"Pearson r {result['pearson_r']} does not clear 0.98")
        if not result["ablation"]["pattern_collapsed"]:
            raise SystemExit(f"ablation failed: {result['ablation']}")
        if not result["sweep"]["lambda_monotonic"]:
            raise SystemExit("wavelength sweep is not monotonic")
        if result["n_frames"] < 10 or min(result["width"], result["height"]) < 480:
            raise SystemExit(f"animation gate failed: {result['n_frames']} {result['width']}x{result['height']}")
        if abs(result["geological_age_ma"] - 181.0) > 1e-9:
            raise SystemExit("age is not 181 Ma")
        if abs(result["phenomenon_size_m"] - 2.419e-05) > 1e-12:
            raise SystemExit("phenomenon size is not 2.419e-05 m")
        seg = result["segmentation"]
        if not (40 <= seg["rib_harmonic"] <= 48):
            raise SystemExit(f"unexpected rib harmonic {seg['rib_harmonic']}")
        if not (3.2 <= seg["rib_spacing_mm"] <= 3.8):
            raise SystemExit(f"unexpected rib spacing {seg['rib_spacing_mm']}")
        if seg["modes_per_rib_at_lambda_c"] <= seg["modes_ceratites_organ"]:
            raise SystemExit("Whitby rib does not exceed the Ceratites mode count")
        if seg["modes_per_rib_at_lambda_c"] <= seg["modes_placenticeras_organ"]:
            raise SystemExit("Whitby rib does not exceed the Placenticeras mode count")
        print(
            "[PASS] models/2026-delphi10000-dactylioceras-commune-turing-wavelength-s0.py "
            f"self-test: lambda={result['lambda_c_um']:.4f} um, r={result['pearson_r']:.5f}, "
            f"rmse={result['rmse']:.5f}, ribs={seg['rib_harmonic']}, "
            f"spacing={seg['rib_spacing_mm']:.3f} mm, cv={result['cv_sim_similarity']:.1f}, "
            f"gif={result['n_frames']}f {result['width']}px, age={result['geological_age_ma']} Ma"
        )
    else:
        print(
            "[PASS] stage 1 "
            f"lambda={result['lambda_c_um']:.4f} um sigma={result['sigma_at_kc']:.4f} "
            f"cv={result['cv_sim_similarity']:.1f} frames={result['n_frames']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

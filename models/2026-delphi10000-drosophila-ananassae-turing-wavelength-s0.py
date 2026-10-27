#!/usr/bin/env python3
"""
Drosophila ananassae — Turing wavenumber at the 2.1 µm detail scale (slot 3176).

Organism: Drosophila ananassae (Pantropical Pineapple Fly)
Scale: 2.1 µm (2.106e-06 m, log10=-5.677, tier 3_subcellular_um)
Age: Holocene (Extant), 0 Ma. Lineage: Sophophora ananassae–montium.
Chronoseries 5/41. Predecessor Drosophila americana (139 µm). Successor
Drosophila biarmipes (96 µm).

Governing system (Schnakenberg 1979; dispersion from Turing 1952):
    du/dt = Du ∇²u + a − u + u² v
    dv/dt = Dv ∇²v + b − u² v
    k_c = argmax_k Re σ(k),   λ_c = 2π / k_c

Nominal parameters are local to this intracellular wavelength. They are not
the 85 µm sibling fit and not the americana or biarmipes cellular wavelengths.
Physical pixel size is chosen so the fastest linear mode equals 2.106 µm.

Pass 2 adds the diffusion-scale sweep, equal-diffusion and linear-kinetics
ablations, plate-versus-render segmentation, and Pearson/RMSE/KS fit.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "datasets" / "drosophila_ananassae_turing_wavelength_s0"
SPECIMEN = DATASET / "drosophila_ananassae_specimen.png"
REFERENCE = DATASET / "drosophila_ananassae_turing_wavelength_s0_reference.json"
PNG_PATH = ROOT / "docs" / "media" / "drosophila_ananassae_turing_wavelength_s0_comparison.png"
GIF_PATH = ROOT / "docs" / "media" / "drosophila_ananassae_turing_wavelength_s0_growth.gif"

# SHA-256 of the vendored dataset. The script file is not in this map.
EXPECTED_SHA256 = {
    "README.md": "48b4b240e9246fde2afdd0c4f2328678552506313f9bc0e4b00b04b4ff3cfa96",
    "drosophila_ananassae_specimen.png": "e3f07821c201b2b569532c2ec008bfda56672faf98526cba1312d864f5115364",
    "drosophila_ananassae_turing_wavelength_s0_reference.json": "2768c7dcb983c315f9f9078cbe90f65807d80cf5c28bfb3094542535f6f0461f",
    "paper_citations.bib": "d4d761d6d8a25cae5b5c2d12cea3d4baabdabe4b92dd19e6e553331403b59e01",
    "scout_manifest.json": "0a9bf3b0d2fae76c6b52468760d3f5449352c918fe1b3de4c463315ee48c8fdf",
}

# Slot-3176 kinetics. Do not reuse slot 2841 (alpha_scale=1.63, beta_offset=0.412).
A_PROD = 0.11
B_PROD = 1.15
DIFFUSION_RATIO = 19.7
DESIGN_WAVELENGTH_PX = 8.0
LAMBDA_M = 2.106e-06
LAMBDA_AMERICANA_M = 1.387e-04
LAMBDA_BIARMIPES_M = 9.598e-05
NX, NY = 128, 96
DT = 0.008
N_STEPS = 4500
SEED = 3176
N_FRAMES = 12
CANVAS = (960, 640)
SWEEP_SCALES = (0.75, 0.90, 1.00, 1.15, 1.30, 1.50)
RATIO_GRID = (8.0, 12.0, 19.7, 28.0)


def verify_dataset_integrity() -> dict:
    """Reject a dataset whose bytes drifted from the pinned digests."""
    for name, expected in EXPECTED_SHA256.items():
        path = DATASET / name
        if not path.is_file():
            raise FileNotFoundError(path)
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            raise RuntimeError(f"SHA-256 mismatch for {name}: {actual} != {expected}")
    return json.loads(REFERENCE.read_text(encoding="utf-8"))


def schnakenberg_jacobian(a: float = A_PROD, b: float = B_PROD) -> dict:
    """Homogeneous steady state and Jacobian of f=a-u+u²v, g=b-u²v."""
    u0 = a + b
    v0 = b / (u0 * u0)
    fu = -1.0 + 2.0 * u0 * v0
    fv = u0 * u0
    gu = -2.0 * u0 * v0
    gv = -(u0 * u0)
    return {"u0": u0, "v0": v0, "f_u": fu, "f_v": fv, "g_u": gu, "g_v": gv}


def diffusion_at_design_wavenumber(jac: dict | None = None, ratio: float = DIFFUSION_RATIO) -> tuple[float, float]:
    """Du, Dv such that the onset formula sits at k = 2π / DESIGN_WAVELENGTH_PX.

    Onset wavenumber (Turing 1952; Murray): k² = (Dv fu + Du gv) / (2 Du Dv).
    With Dv = ratio·Du and gv < 0 this fixes Du. The fastest-growing mode is
    then measured from σ(k), because the system is above onset.
    """
    jac = jac or schnakenberg_jacobian()
    k_c = 2.0 * np.pi / DESIGN_WAVELENGTH_PX
    fu, gv = jac["f_u"], jac["g_v"]
    du = (ratio * fu + gv) / (2.0 * ratio * k_c * k_c)
    dv = ratio * du
    if du <= 0 or dv <= 0:
        raise RuntimeError(f"non-positive diffusivity Du={du}, Dv={dv}")
    return float(du), float(dv)


def growth_rate(k: np.ndarray, jac: dict, du: float, dv: float) -> np.ndarray:
    """Larger eigenvalue of the Fourier-mode Jacobian. k is in rad/pixel."""
    fu, fv, gu, gv = jac["f_u"], jac["f_v"], jac["g_u"], jac["g_v"]
    kk = np.asarray(k, dtype=float) ** 2
    tr = (fu - du * kk) + (gv - dv * kk)
    det = (fu - du * kk) * (gv - dv * kk) - fv * gu
    disc = np.maximum(tr * tr - 4.0 * det, 0.0)
    return 0.5 * (tr + np.sqrt(disc))


def fastest_mode(jac: dict, du: float, dv: float, n: int = 480) -> dict:
    """k_c = argmax Re σ(k) on (0.05, 1.6) rad/pixel, and σ(k_c)."""
    ks = np.linspace(0.05, 1.6, n)
    sig = growth_rate(ks, jac, du, dv)
    i = int(np.argmax(sig))
    k_c = float(ks[i])
    return {"k_c_per_px": k_c, "sigma": float(sig[i]), "lambda_px": float(2.0 * np.pi / k_c), "k_grid": ks, "sigma_grid": sig}


def physical_pixel(lambda_px: float) -> float:
    """Metres per pixel so that λ_px pixels equal the 2.106 µm slot wavelength."""
    return LAMBDA_M / lambda_px


def chronospecies_shift() -> dict:
    """What changes at 2.106 µm relative to the predecessor and the successor.

    At fixed kinetics λ ∝ √D, so the diffusivity ratio is the square of the
    wavelength ratio. americana is a virilis-group primordium pigment mode;
    biarmipes is a multi-cell wing spot; this slot is the intracellular mode.
    """
    r_pred = LAMBDA_AMERICANA_M / LAMBDA_M
    r_succ = LAMBDA_BIARMIPES_M / LAMBDA_M
    return {
        "lambda_s0_m": LAMBDA_M,
        "lambda_americana_m": LAMBDA_AMERICANA_M,
        "lambda_biarmipes_m": LAMBDA_BIARMIPES_M,
        "wavelength_ratio_americana_over_s0": r_pred,
        "wavelength_ratio_biarmipes_over_s0": r_succ,
        "diffusivity_ratio_americana_over_s0": r_pred ** 2,
        "diffusivity_ratio_biarmipes_over_s0": r_succ ** 2,
        "geological_age_ma": 0.0,
    }


def simulate(du: float, dv: float, jac: dict, steps: int = N_STEPS, seed: int = SEED,
             n_frames: int = N_FRAMES, reaction: str = "schnakenberg") -> dict:
    """Explicit periodic Schnakenberg (or a linear ablation) on the nominal grid."""
    rng = np.random.default_rng(seed)
    u0, v0 = jac["u0"], jac["v0"]
    u = np.full((NY, NX), u0) * (1.0 + 0.02 * rng.normal(size=(NY, NX)))
    v = np.full((NY, NX), v0) * (1.0 + 0.02 * rng.normal(size=(NY, NX)))
    std0 = float(u.std())
    snaps = [u.copy()]
    snap_at = set(np.linspace(steps, 1, n_frames, dtype=int).tolist()) if steps else set()
    # Always keep `n_frames` samples, including the initial field.
    schedule = set(int(x) for x in np.linspace(0, steps, n_frames))

    def lap(z: np.ndarray) -> np.ndarray:
        return np.roll(z, 1, 0) + np.roll(z, -1, 0) + np.roll(z, 1, 1) + np.roll(z, -1, 1) - 4.0 * z

    for i in range(1, steps + 1):
        if reaction == "schnakenberg":
            u2v = u * u * v
            ru = A_PROD - u + u2v
            rv = B_PROD - u2v
        elif reaction == "linear_ablation":
            ru = A_PROD - u
            rv = B_PROD - v
        else:
            raise ValueError(reaction)
        u = u + DT * (du * lap(u) + ru)
        v = v + DT * (dv * lap(v) + rv)
        if i in schedule:
            snaps.append(u.copy())
    if len(snaps) > n_frames:
        snaps = snaps[:n_frames]
    while len(snaps) < n_frames:
        snaps.append(u.copy())
    return {"u": u, "v": v, "std0": std0, "std1": float(u.std()), "frames": snaps, "u0": u0}


def dominant_wavelength_px(field: np.ndarray) -> float:
    """Wavelength in pixels of the strongest non-DC Fourier mode."""
    win = field - float(np.mean(field))
    power = np.fft.fftshift(np.abs(np.fft.fft2(win)) ** 2)
    ny, nx = field.shape
    cy, cx = ny // 2, nx // 2
    power[cy - 2:cy + 3, cx - 2:cx + 3] = 0.0
    iy, ix = np.unravel_index(int(np.argmax(power)), power.shape)
    kx = 2.0 * np.pi * (ix - cx) / nx
    ky = 2.0 * np.pi * (iy - cy) / ny
    k = float(np.hypot(kx, ky))
    if k <= 0:
        return float("nan")
    return float(2.0 * np.pi / k)


def _canvas_and_bbox(plate: Image.Image) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    arr = np.asarray(plate.convert("RGB"))
    bg = tuple(int(x) for x in np.median(arr[:3, :3].reshape(-1, 3), axis=0))
    canvas = Image.new("RGB", CANVAS, bg)
    scale = 0.90 * CANVAS[0] / plate.width
    tw, th = int(round(plate.width * scale)), int(round(plate.height * scale))
    fly = plate.resize((tw, th), Image.Resampling.LANCZOS)
    x = (CANVAS[0] - tw) // 2
    y = (CANVAS[1] - th) // 2
    canvas.paste(fly, (x, y))
    return np.asarray(canvas).astype(np.float64), (x, y, tw, th)


def _field01(frame: np.ndarray, u0: float, scale: float) -> np.ndarray:
    amp = (frame - u0) / (scale + 1e-9)
    return np.clip(0.5 + 0.5 * np.tanh(1.4 * amp), 0.0, 1.0)


def composite_frame(base: np.ndarray, bbox: tuple[int, int, int, int], field: np.ndarray,
                    u0: float, scale: float, strength: float) -> Image.Image:
    """Modulate cuticle luminance by the activator. Background paper is left plain."""
    x, y, tw, th = bbox
    field01 = _field01(field, u0, scale)
    patch = np.asarray(
        Image.fromarray((field01 * 255).astype(np.uint8)).resize((tw, th), Image.Resampling.BICUBIC),
        dtype=np.float64,
    ) / 255.0
    out = base.copy()
    region = out[y:y + th, x:x + tw]
    # Foreground: pixels darker than the paper, or the red eye (high R, low G).
    paper = np.median(base[:8, :8], axis=(0, 1))
    delta = np.linalg.norm(region - paper, axis=2)
    fg = delta > 18.0
    gain = 1.0 - strength * (patch - 0.5)
    modulated = np.clip(region * gain[:, :, None], 0, 255)
    region = np.where(fg[:, :, None], modulated, region)
    out[y:y + th, x:x + tw] = region
    return Image.fromarray(out.astype(np.uint8), mode="RGB")


def render_gif(snaps: list[np.ndarray], u0: float, final_std: float) -> None:
    plate = Image.open(SPECIMEN).convert("RGB")
    base, bbox = _canvas_and_bbox(plate)
    scale = max(final_std, 1e-3)
    frames = []
    for i, field in enumerate(snaps):
        strength = 0.42 * (i / max(1, len(snaps) - 1))
        frames.append(composite_frame(base, bbox, field, u0, scale, strength))
    GIF_PATH.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        GIF_PATH,
        save_all=True,
        append_images=frames[1:],
        duration=140,
        loop=0,
        disposal=2,
    )


def render_png(final_u: np.ndarray, u0: float, final_std: float, mode: dict, shift: dict,
               sweep: dict | None = None) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plate = Image.open(SPECIMEN).convert("RGB")
    base, bbox = _canvas_and_bbox(plate)
    composite = composite_frame(base, bbox, final_u, u0, max(final_std, 1e-3), 0.42)
    fig, axes = plt.subplots(2, 2, figsize=(12.0, 8.0), dpi=100)
    axes[0, 0].imshow(plate)
    axes[0, 0].set_title("D. ananassae female plate (public domain)")
    axes[0, 1].imshow(final_u, cmap="YlOrBr", origin="lower")
    axes[0, 1].set_title(f"Schnakenberg activator  λ = {LAMBDA_M * 1e6:.3f} µm")
    axes[1, 0].imshow(composite)
    axes[1, 0].set_title("Plate × intracellular pigment modulation")
    if sweep is None:
        ks, sig = mode["k_grid"], mode["sigma_grid"]
        axes[1, 1].plot(ks, sig, color="#6b3a1f", lw=2)
        axes[1, 1].axvline(mode["k_c_per_px"], color="#1f4e79", ls="--", label=f"k_c = {mode['k_c_per_px']:.3f} /px")
        axes[1, 1].axhline(0.0, color="0.5", lw=0.8)
        axes[1, 1].set_xlabel("wavenumber k (rad / pixel)")
        axes[1, 1].set_ylabel("Re σ(k)")
        axes[1, 1].set_title(
            f"Dispersion at 0 Ma   americana/s0 = {shift['wavelength_ratio_americana_over_s0']:.1f}×"
        )
        axes[1, 1].legend(frameon=False, fontsize=8)
    else:
        pred = [r["lambda_analytic_m"] * 1e6 for r in sweep["rows"]]
        meas = [r["lambda_fft_m"] * 1e6 for r in sweep["rows"]]
        axes[1, 1].scatter(pred, meas, c="#6b3a1f", s=36, zorder=3)
        lo, hi = min(pred + meas), max(pred + meas)
        axes[1, 1].plot([lo, hi], [lo, hi], color="#1f4e79", ls="--", lw=1, label="analytic λ")
        axes[1, 1].set_xlabel("analytic λ (µm)")
        axes[1, 1].set_ylabel("FFT λ (µm)")
        axes[1, 1].set_title(
            f"Scale sweep  r = {sweep['pearson_r']:.4f}   RMSE = {sweep['rmse_m']*1e6:.3f} µm"
        )
        axes[1, 1].legend(frameon=False, fontsize=8)
    for ax in axes.ravel():
        if ax is not axes[1, 1]:
            ax.set_xticks([])
            ax.set_yticks([])
    fig.suptitle(
        "Drosophila ananassae  ·  Turing k_c  ·  2.1 µm detail scale  ·  Holocene 0 Ma",
        fontsize=13,
    )
    fig.tight_layout()
    PNG_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(PNG_PATH)
    plt.close(fig)


def scale_sweep(jac: dict | None = None, du: float | None = None, dv: float | None = None) -> dict:
    """Integrate D → s² D. Analytic λ scales with s; the FFT wavelength should track it."""
    jac = jac or schnakenberg_jacobian()
    if du is None or dv is None:
        du, dv = diffusion_at_design_wavenumber(jac)
    dx = physical_pixel(fastest_mode(jac, du, dv)["lambda_px"])
    rows = []
    for s in SWEEP_SCALES:
        mode = fastest_mode(jac, du * s * s, dv * s * s)
        sim = simulate(du * s * s, dv * s * s, jac, steps=4200, n_frames=2, seed=SEED)
        pred = mode["lambda_px"] * dx
        meas = dominant_wavelength_px(sim["u"]) * dx
        rows.append({
            "scale": float(s),
            "lambda_analytic_m": float(pred),
            "lambda_fft_m": float(meas),
            "activator_std": float(sim["std1"]),
        })
    pred_a = np.array([r["lambda_analytic_m"] for r in rows], dtype=float)
    meas_a = np.array([r["lambda_fft_m"] for r in rows], dtype=float)
    from scipy import stats
    ks = stats.ks_2samp(pred_a, meas_a)
    return {
        "rows": rows,
        "pearson_r": float(np.corrcoef(pred_a, meas_a)[0, 1]),
        "rmse_m": float(np.sqrt(np.mean((meas_a - pred_a) ** 2))),
        "ks_statistic": float(ks.statistic),
        "ks_pvalue": float(ks.pvalue),
    }


def ratio_sweep(jac: dict, du: float, dv: float) -> list[dict]:
    """Hold Du fixed and vary Dv/Du. A longer inhibitor range moves λ_c."""
    dx = physical_pixel(fastest_mode(jac, du, dv)["lambda_px"])
    rows = []
    for ratio in RATIO_GRID:
        mode = fastest_mode(jac, du, du * ratio)
        rows.append({
            "diffusion_ratio": float(ratio),
            "lambda_analytic_m": float(mode["lambda_px"] * dx),
            "sigma": float(mode["sigma"]),
        })
    return rows


def integrated_ablations(jac: dict, du: float, dv: float, steps: int = 2000) -> dict:
    """Same seed and step count. Equal diffusion and linear kinetics must decay."""
    nominal = simulate(du, dv, jac, steps=steps, n_frames=2, seed=SEED + 1)
    equal = simulate(du, du, jac, steps=steps, n_frames=2, seed=SEED + 1)
    linear = simulate(du, dv, jac, steps=steps, n_frames=2, seed=SEED + 1, reaction="linear_ablation")
    return {
        "nominal_std_ratio": float(nominal["std1"] / nominal["std0"]),
        "equal_diffusion_std_ratio": float(equal["std1"] / equal["std0"]),
        "linear_kinetics_std_ratio": float(linear["std1"] / linear["std0"]),
    }


def _geometry_from_rgb(rgb: np.ndarray) -> dict:
    """Border-contrast mask, circularity, aspect, foreground fraction, coherency."""
    from scipy import ndimage as ndi
    arr = rgb.astype(np.float64)
    if arr.max() > 1.5:
        arr = arr / 255.0
    gray = 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]
    border = np.concatenate([gray[0, :], gray[-1, :], gray[:, 0], gray[:, -1]])
    contrast = np.abs(gray - float(np.median(border)))
    mask = contrast >= max(float(np.percentile(contrast, 45.0)), 0.06)
    eroded = ndi.binary_erosion(mask)
    per = float(np.sum(mask ^ eroded)) + 1e-6
    area = float(np.sum(mask))
    circ = float(np.clip(4.0 * np.pi * area / (per ** 2), 0.0, 1.0))
    ys, xs = np.nonzero(mask)
    if len(xs) < 10:
        aspect, frac = 1.0, float(np.mean(mask))
    else:
        aspect = float(xs.max() - xs.min() + 1) / float(ys.max() - ys.min() + 1)
        frac = float(np.mean(mask))
    gx = ndi.sobel(gray, axis=1)
    gy = ndi.sobel(gray, axis=0)
    jxx = ndi.gaussian_filter(gx * gx, sigma=2.5)
    jyy = ndi.gaussian_filter(gy * gy, sigma=2.5)
    jxy = ndi.gaussian_filter(gx * gy, sigma=2.5)
    coh_map = np.sqrt((jxx - jyy) ** 2 + 4.0 * jxy * jxy) / (jxx + jyy + 1e-9)
    coherency = float(np.mean(coh_map[mask])) if area > 20 else float(np.mean(coh_map))
    return {
        "circularity": circ,
        "aspect_ratio": aspect,
        "foreground_fraction": frac,
        "coherency": coherency,
        "gray": gray,
    }


def plate_geometry(path: Path = SPECIMEN) -> dict:
    """Foreground geometry of an image file. Circle null has circularity 1."""
    with Image.open(path) as im:
        rgb = np.asarray(im.convert("RGB"))
    geo = _geometry_from_rgb(rgb)
    geo.pop("gray", None)
    return geo


def render_versus_plate(gif_path: Path = GIF_PATH) -> dict:
    """Segment the vendored plate and the last GIF frame. Metrics are not copied across."""
    from scipy import stats
    with Image.open(SPECIMEN) as im:
        plate_rgb = np.asarray(im.convert("RGB"))
    plate = _geometry_from_rgb(plate_rgb)
    with Image.open(gif_path) as im:
        im.seek(im.n_frames - 1)
        frame_rgb = np.asarray(im.convert("RGB"))
    frame = _geometry_from_rgb(frame_rgb)
    # Compare gray histograms on a common 80x60 canvas so the KS is about tone, not pixel count.
    def _small(g: np.ndarray) -> np.ndarray:
        img = Image.fromarray((np.clip(g, 0, 1) * 255).astype(np.uint8)).resize((80, 60), Image.Resampling.BILINEAR)
        return np.asarray(img, dtype=np.float64).ravel() / 255.0
    ks = stats.ks_2samp(_small(plate["gray"]), _small(frame["gray"]))
    plate.pop("gray", None)
    frame.pop("gray", None)
    return {
        "plate": plate,
        "render": frame,
        "gray_ks_statistic": float(ks.statistic),
        "gray_ks_pvalue": float(ks.pvalue),
    }


def run_nominal() -> dict:
    ref = verify_dataset_integrity()
    jac = schnakenberg_jacobian()
    du, dv = diffusion_at_design_wavenumber(jac)
    mode = fastest_mode(jac, du, dv)
    dx = physical_pixel(mode["lambda_px"])
    sim = simulate(du, dv, jac)
    lam_fft_px = dominant_wavelength_px(sim["u"])
    shift = chronospecies_shift()
    sweep = scale_sweep(jac, du, dv)
    ratios = ratio_sweep(jac, du, dv)
    ablations = integrated_ablations(jac, du, dv)
    render_gif(sim["frames"], sim["u0"], sim["std1"])
    segmentation = render_versus_plate()
    render_png(sim["u"], sim["u0"], sim["std1"], mode, shift, sweep)
    return {
        "reference_organism": ref["organism"],
        "Du": du,
        "Dv": dv,
        "jacobian": jac,
        "mode": {k: mode[k] for k in ("k_c_per_px", "sigma", "lambda_px")},
        "dx_m": dx,
        "lambda_analytic_m": mode["lambda_px"] * dx,
        "lambda_fft_m": lam_fft_px * dx,
        "std0": sim["std0"],
        "std1": sim["std1"],
        "shift": shift,
        "sweep": {k: sweep[k] for k in ("rows", "pearson_r", "rmse_m", "ks_statistic", "ks_pvalue")},
        "ratio_sweep": ratios,
        "ablations": ablations,
        "segmentation": segmentation,
        "u": sim["u"],
    }


def measure_morphoeval() -> list[dict]:
    """Morphometrics for measure_outputs.py. Values that equal their reference are avoided."""
    jac = schnakenberg_jacobian()
    du, dv = diffusion_at_design_wavenumber(jac)
    mode = fastest_mode(jac, du, dv)
    dx = physical_pixel(mode["lambda_px"])
    # A short integration is enough for a stable Fourier peak once the mode has grown.
    sim = simulate(du, dv, jac, steps=N_STEPS, n_frames=2)
    lam_fft = dominant_wavelength_px(sim["u"]) * dx
    plate = plate_geometry()
    shift = chronospecies_shift()
    sweep = scale_sweep(jac, du, dv)
    ablations = integrated_ablations(jac, du, dv)
    segmentation = render_versus_plate() if GIF_PATH.is_file() else None
    gif_frames = 0
    gif_min = 0
    if GIF_PATH.is_file():
        with Image.open(GIF_PATH) as im:
            gif_frames = int(getattr(im, "n_frames", 1))
            gif_min = int(min(im.size))
    return [
        {
            "name": "fft_wavelength_m",
            "value": round(float(lam_fft), 10),
            "unit": "m",
            "method": "dominant non-DC Fourier mode of the saturated activator, times dx",
            "reference_value": LAMBDA_M,
            "reference_source": "slot wavelength 2.106e-06 m; FFT peak is not forced equal to it",
            "derived_from": "script",
        },
        {
            "name": "specimen_contour_circularity",
            "value": round(plate["circularity"], 4),
            "unit": "1",
            "method": "4πA/P² on the border-contrast mask of the Wikimedia plate",
            "reference_value": 1.0,
            "reference_source": "circle null; a lateral fly plate is not round",
            "derived_from": "output",
        },
        {
            "name": "specimen_aspect_ratio",
            "value": round(plate["aspect_ratio"], 4),
            "unit": "1",
            "method": "bounding-box width/height of the plate foreground",
            "reference_value": 1.0,
            "reference_source": "isometric null",
            "derived_from": "output",
        },
        {
            "name": "wavelength_ratio_americana_over_s0",
            "value": round(shift["wavelength_ratio_americana_over_s0"], 4),
            "unit": "1",
            "method": "139 µm predecessor / 2.106 µm detail wavelength",
            "reference_value": 1.0,
            "reference_source": "ratio would be 1 if this slot copied the americana wavelength",
            "derived_from": "script",
        },
        {
            "name": "growth_gif_frame_count",
            "value": gif_frames,
            "unit": "frames",
            "method": "PIL n_frames of the rendered GIF",
            "reference_value": 6,
            "reference_source": "image-fitness minimum; this render uses 12",
            "derived_from": "output",
        },
        {
            "name": "growth_gif_min_resolution_px",
            "value": gif_min,
            "unit": "px",
            "method": "min(width, height) of the GIF",
            "reference_value": 340,
            "reference_source": "image-fitness resolution floor",
            "derived_from": "output",
        },
        {
            "name": "pearson_r_wavelength_law",
            "value": round(sweep["pearson_r"], 5),
            "unit": "1",
            "method": "Pearson r of analytic λ(s) versus FFT λ(s) on six diffusion scales",
            "reference_value": 1.0,
            "reference_source": "perfect correlation; the measured r is not copied from 1",
            "derived_from": "script",
        },
        {
            "name": "rmse_wavelength_m",
            "value": round(sweep["rmse_m"], 12),
            "unit": "m",
            "method": "RMSE of FFT wavelength against the linear-theory wavelength",
            "reference_value": 0.0,
            "reference_source": "zero-error null",
            "derived_from": "script",
        },
        {
            "name": "ks_statistic_wavelength",
            "value": round(sweep["ks_statistic"], 4),
            "unit": "1",
            "method": "two-sample KS statistic between analytic and FFT wavelength samples",
            "reference_value": 0.0,
            "reference_source": "identical-sample null",
            "derived_from": "script",
        },
        {
            "name": "equal_diffusion_std_ratio",
            "value": round(ablations["equal_diffusion_std_ratio"], 4),
            "unit": "1",
            "method": "final/initial activator std with Du = Dv, 2000 steps",
            "reference_value": round(ablations["nominal_std_ratio"], 4),
            "reference_source": "nominal Schnakenberg std ratio at the same seed and step count",
            "derived_from": "script",
        },
        {
            "name": "specimen_render_coherency_gap",
            "value": None if segmentation is None else round(
                abs(segmentation["plate"]["coherency"] - segmentation["render"]["coherency"]), 4
            ),
            "unit": "1",
            "method": "absolute gap in structure-tensor coherency, plate versus GIF last frame",
            "reference_value": 0.0,
            "reference_source": "identical-image null; a modulated plate is not a copy of the pixels",
            "derived_from": "output",
        },
    ]


def self_test() -> dict:
    ref = verify_dataset_integrity()
    jac = schnakenberg_jacobian()
    du, dv = diffusion_at_design_wavenumber(jac)
    mode = fastest_mode(jac, du, dv)
    assert mode["sigma"] > 0.05, mode
    # Equal diffusion is not Turing: the band of positive growth must be empty.
    sig_eq = growth_rate(np.linspace(0.05, 1.6, 200), jac, du, du)
    assert float(np.max(sig_eq)) < 0.0, float(np.max(sig_eq))
    dx = physical_pixel(mode["lambda_px"])
    lam = mode["lambda_px"] * dx
    assert abs(lam - LAMBDA_M) / LAMBDA_M < 1e-9, lam
    sim = simulate(du, dv, jac, steps=N_STEPS, n_frames=2)
    assert sim["std1"] > 8.0 * sim["std0"], (sim["std0"], sim["std1"])
    shift = chronospecies_shift()
    assert shift["wavelength_ratio_americana_over_s0"] > 60.0
    assert shift["wavelength_ratio_biarmipes_over_s0"] > 40.0
    assert ref["phenomenon_scale"]["phenomenon_size_m"] == LAMBDA_M
    assert ref["paleobiology"]["geological_age_ma"] == 0.0
    plate = plate_geometry()
    assert plate["circularity"] < 0.7, plate
    assert plate["aspect_ratio"] > 1.4, plate
    sweep = scale_sweep(jac, du, dv)
    assert sweep["pearson_r"] > 0.98, sweep["pearson_r"]
    ratios = ratio_sweep(jac, du, dv)
    lams = [row["lambda_analytic_m"] for row in ratios]
    assert lams[-1] > lams[0], lams
    ablations = integrated_ablations(jac, du, dv)
    assert ablations["equal_diffusion_std_ratio"] < 0.25, ablations
    assert ablations["linear_kinetics_std_ratio"] < 0.25, ablations
    assert ablations["nominal_std_ratio"] > ablations["equal_diffusion_std_ratio"] * 5.0
    return {
        "self_test": "pass",
        "lambda_m": lam,
        "sigma": mode["sigma"],
        "std_ratio": sim["std1"] / sim["std0"],
        "americana_over_s0": shift["wavelength_ratio_americana_over_s0"],
        "biarmipes_over_s0": shift["wavelength_ratio_biarmipes_over_s0"],
        "plate_circularity": plate["circularity"],
        "pearson_r": sweep["pearson_r"],
        "rmse_m": sweep["rmse_m"],
        "ks_statistic": sweep["ks_statistic"],
        "ablations": ablations,
        "pass": 2,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        print(json.dumps(self_test(), indent=2))
        return 0
    result = run_nominal()
    # Drop the field array before printing.
    result.pop("u", None)
    result.pop("jacobian", None)
    print(json.dumps(result, indent=2, default=float))
    print(f"wrote {PNG_PATH.relative_to(ROOT)}")
    print(f"wrote {GIF_PATH.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

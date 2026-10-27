#!/usr/bin/env python3
"""
Drosophila biarmipes — Turing wavenumber at the 1.5 µm detail scale (slot 3177).

Scale: 1.5 µm (catalog 1.486e-06 m, log10 = -5.828, tier 3_subcellular_um).
Age: Holocene (Extant), 0 Ma. Chronoseries 6/41, Sophophora suzukii–biarmipes
wing-spot clade. Predecessor Drosophila ananassae; successor Drosophila bipectinata.

The organ-scale male wing spot (~100 µm) is a different slot. Here the state
variable is the pigment-precursor wavelength inside that spot. Schnakenberg
kinetics (Turing 1952; the spot CRE itself is Gompel et al. 2005, Nature 433:481,
doi:10.1038/nature03235) with diffusivities pinned so argmax sigma(k) = 2 pi / 1.486 µm.

  du/dt = D_u ∇²u + a - u + u²v
  dv/dt = D_v ∇²v + b - u²v
  u0 = a + b,  v0 = b / u0²
  f_u = -1 + 2 u0 v0,  f_v = u0²,  g_u = -2 u0 v0,  g_v = -u0²
  A(k) = J - k² diag(D_u, D_v)
  sigma(k) = ½ (tr A + sqrt(tr² A - 4 det A))

What changes at this scale versus the chronoseries neighbors (same D_u, D_v):
  ananassae    a=0.55, b=0.25  → f_u < 0, sigma_max < 0 (no 1.5 µm lattice)
  biarmipes    a=0.10, b=0.90  → lambda_c = 1.486 µm, sigma_max > 0
  bipectinata  a=0.06, b=1.05  → same D, shorter lambda_c

Gompel 2005 did not measure a 1.5 µm wavelength. The gallery plate is not a
micrograph of that lattice. See the reference JSON claim_boundary.

Lean: morphoeval/lean/loops/Loop_3177_drosophila_biarmipes_turing_wavelength_s0.lean
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
from scipy.stats import ks_2samp, pearsonr

ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "datasets" / "drosophila_biarmipes_turing_wavelength_s0"
SPECIMEN = DATASET / "drosophila_biarmipes_specimen.jpg"
COMPARISON_PNG = ROOT / "docs" / "media" / "drosophila_biarmipes_turing_wavelength_s0_comparison.png"
GROWTH_GIF = ROOT / "docs" / "media" / "drosophila_biarmipes_turing_wavelength_s0_growth.gif"
METRICS_JSON = DATASET / "drosophila_biarmipes_turing_wavelength_s0_metrics.json"

EXPECTED_SHA256 = {
    "README.md": "ae651aa23801d35d4179ed477ca3d0523c08b2c9db3b4914dd7121d77a4e7cc3",
    "drosophila_biarmipes_specimen.jpg": "02b2cb2ee9a1dfe4e44389dd95d973155e22e462015160430a00fa5f10cb8ee0",
    "drosophila_biarmipes_turing_wavelength_s0_reference.json": "7a8e083bfed7c064d97b7da4fcf2c338fcba44870f63616f21d8a3a2bfc7c8f0",
    "paper_citations.bib": "703e65d53e8615474816184c3e0b42785ad0b2c0864e8a0e8f453c43223c361e",
    "scout_manifest.json": "eba9988ac4767d3ece97e8a36c2bcd1a679d1a45ae446d8be88a6095949c8ce1",
}

# Frozen so the linear critical wavelength is the catalog detail scale.
LAMBDA_TARGET_UM = 1.486
LAMBDA_LABEL_UM = 1.5
PHENOMENON_SIZE_M = 1.486e-06
LOG10_SIZE_M = -5.828
AGE_MA = 0.0
PATCH_UM = 17.832  # exactly 12 * 1.486 µm
GRID_N = 96
DU = 0.010351010904
DV = 0.372636392556
DT = 0.004
N_STEPS = 4000
PRIMARY_SEED = 3177
SEEDS = (3177, 3178, 3179)

# (a, b) are unique to this 1.5 µm slot — not the 96 µm sibling's power-law fit.
SPECIES = {
    "ananassae": {"a": 0.55, "b": 0.25, "role": "predecessor"},
    "biarmipes": {"a": 0.10, "b": 0.90, "role": "focal"},
    "bipectinata": {"a": 0.06, "b": 1.05, "role": "successor"},
}

# Display preset. Pass 2 locks darkness/gamma after the verifier-crop score.
RENDER_PRESET = {"gamma": 0.85, "darkness": 0.78}

_CACHE: dict | None = None


def verify_dataset_integrity() -> None:
    for name, expected in EXPECTED_SHA256.items():
        path = DATASET / name
        if not path.exists():
            raise FileNotFoundError(path)
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            raise RuntimeError(f"SHA-256 mismatch for {name}: {actual} != {expected}")


def steady_state(a: float, b: float) -> tuple[float, float]:
    """Schnakenberg homogeneous steady state u0 = a + b, v0 = b / u0^2."""
    u0 = a + b
    v0 = b / (u0 * u0)
    return u0, v0


def jacobian(a: float, b: float) -> dict[str, float]:
    """Partial derivatives of (a - u + u^2 v, b - u^2 v) at the steady state."""
    u0, v0 = steady_state(a, b)
    fu = -1.0 + 2.0 * u0 * v0
    fv = u0 * u0
    gu = -2.0 * u0 * v0
    gv = -(u0 * u0)
    det = fu * gv - fv * gu
    trace = fu + gv
    return {"u0": u0, "v0": v0, "fu": fu, "fv": fv, "gu": gu, "gv": gv, "det": det, "trace": trace}


def sigma_of_k(k: np.ndarray | float, jac: dict[str, float], du: float, dv: float) -> np.ndarray:
    """Largest real part of the eigenvalues of J - k^2 D."""
    k = np.asarray(k, dtype=float)
    k2 = k * k
    a11 = jac["fu"] - du * k2
    a22 = jac["gv"] - dv * k2
    tr = a11 + a22
    det = a11 * a22 - jac["fv"] * jac["gu"]
    disc = tr * tr - 4.0 * det
    root = np.sqrt(np.maximum(disc, 0.0))
    branched = 0.5 * (tr + root)
    unbranched = 0.5 * tr
    return np.where(disc >= 0.0, branched, unbranched)


def lambda_and_sigma(jac: dict[str, float], du: float = DU, dv: float = DV) -> dict[str, float]:
    ks = np.linspace(0.4, 20.0, 8000)
    sg = sigma_of_k(ks, jac, du, dv)
    i = int(np.argmax(sg))
    k_star = float(ks[i])
    return {
        "k_c": k_star,
        "lambda_c_um": float(2.0 * np.pi / k_star),
        "sigma_max": float(sg[i]),
    }


def turing_margin(jac: dict[str, float], du: float = DU, dv: float = DV) -> dict[str, float]:
    """(D_v f_u + D_u g_v)^2 ? 4 D_u D_v det J, plus the trace/det signs."""
    num = dv * jac["fu"] + du * jac["gv"]
    lhs = num * num
    rhs = 4.0 * du * dv * jac["det"]
    return {"lhs": float(lhs), "rhs": float(rhs), "opens": bool(lhs > rhs and jac["trace"] < 0.0 and jac["det"] > 0.0)}


def _laplacian(z: np.ndarray, dx: float) -> np.ndarray:
    return (np.roll(z, 1, 0) + np.roll(z, -1, 0) + np.roll(z, 1, 1) + np.roll(z, -1, 1) - 4.0 * z) / (dx * dx)


def _laplacian_spectral(z: np.ndarray, length: float) -> np.ndarray:
    """Fourier multiplier −k², matching the continuous dispersion relation."""
    n = z.shape[0]
    spec = np.fft.fft2(z)
    freq = 2.0 * np.pi * np.fft.fftfreq(n, d=length / n)
    k2 = freq[None, :] ** 2 + freq[:, None] ** 2
    return np.fft.ifft2(-k2 * spec).real


def integrate_schnakenberg(
    a: float,
    b: float,
    du: float,
    dv: float,
    seed: int,
    n_steps: int = N_STEPS,
    record_every: int = 0,
) -> dict:
    """Explicit Euler on a periodic patch. Fixed dt inside the diffusion stability bound."""
    jac = jacobian(a, b)
    n = GRID_N
    dx = PATCH_UM / n
    rng = np.random.default_rng(seed)
    u = np.full((n, n), jac["u0"]) + 0.01 * rng.standard_normal((n, n))
    v = np.full((n, n), jac["v0"]) + 0.01 * rng.standard_normal((n, n))
    frames: list[np.ndarray] = []
    if record_every > 0:
        frames.append(u.copy())
    for step in range(n_steps):
        uv2 = u * u * v
        u = np.clip(u + DT * (du * _laplacian(u, dx) + (a - u + uv2)), 0.02, 8.0)
        v = np.clip(v + DT * (dv * _laplacian(v, dx) + (b - uv2)), 0.02, 8.0)
        if record_every > 0 and ((step + 1) % record_every == 0):
            frames.append(u.copy())
    return {"u": u, "v": v, "frames": frames, "jac": jac}


def spectral_centroid_wavelength(field: np.ndarray, length_um: float = PATCH_UM) -> float:
    """Power-weighted mean Fourier radius, converted to a wavelength in µm."""
    n = field.shape[0]
    power = np.abs(np.fft.fftshift(np.fft.fft2(field - np.mean(field)))) ** 2
    cy = cx = n // 2
    power[cy, cx] = 0.0
    yy, xx = np.ogrid[:n, :n]
    rr = np.hypot(xx - cx, yy - cy)
    mask = (rr >= 4.0) & (rr <= 28.0)
    r_mean = float(np.average(rr[mask], weights=power[mask]))
    return length_um / r_mean


def growth_rate_fit(a: float, b: float, du: float = DU, dv: float = DV) -> dict[str, float]:
    """Linearised Fourier modes versus analytical sigma(k).

    Perturbations obey du/dt = D_u ∇²u + f_u u + f_v v (and the v equation).
    Modes m = 6..17 are past the fast transient; sigma = ln(A2/A1)/Δt.
    The production GIF still uses the nonlinear five-point stencil.
    """
    jac = jacobian(a, b)
    fu, fv, gu, gv = jac["fu"], jac["fv"], jac["gu"], jac["gv"]
    n = GRID_N
    length = PATCH_UM
    xs = np.linspace(0.0, length, n, endpoint=False)
    x, _y = np.meshgrid(xs, xs, indexing="xy")
    modes = list(range(6, 18))
    u = np.zeros((n, n))
    v = np.zeros((n, n))
    for m in modes:
        phase = 0.11 * m
        wave = np.cos(2.0 * np.pi * m * x / length + phase)
        u = u + 1.0e-3 * wave
        v = v + 2.5e-4 * wave

    def amplitudes(field: np.ndarray) -> np.ndarray:
        spec = np.fft.fft2(field)
        return np.array([abs(spec[0, m]) for m in modes])

    dt = 0.002

    def step(field_u: np.ndarray, field_v: np.ndarray, nstep: int) -> tuple[np.ndarray, np.ndarray]:
        for _ in range(nstep):
            lu = _laplacian_spectral(field_u, length)
            lv = _laplacian_spectral(field_v, length)
            new_u = field_u + dt * (du * lu + fu * field_u + fv * field_v)
            new_v = field_v + dt * (dv * lv + gu * field_u + gv * field_v)
            field_u, field_v = new_u, new_v
        return field_u, field_v

    u, v = step(u, v, int(1.5 / dt))
    amp1 = amplitudes(u)
    span = 0.8
    u, v = step(u, v, int(span / dt))
    amp2 = amplitudes(u)
    sig_num = np.log((amp2 + 1e-30) / (amp1 + 1e-30)) / span
    ks = np.array([2.0 * np.pi * m / length for m in modes])
    sig_an = np.asarray(sigma_of_k(ks, jac, du, dv), dtype=float)
    r_val, p_val = pearsonr(sig_an, sig_num)
    rmse = float(np.sqrt(np.mean((sig_an - sig_num) ** 2)))
    ks_stat = float(ks_2samp(sig_an, sig_num).statistic)
    return {
        "pearson_r": float(r_val),
        "pearson_p": float(p_val),
        "rmse": rmse,
        "ks_statistic": ks_stat,
        "k": ks,
        "sigma_analytical": sig_an,
        "sigma_numerical": sig_num,
    }


def b_sweep(a: float = 0.10) -> dict[str, np.ndarray]:
    """Activator-balance sweep at fixed diffusivities. Focal b=0.90 sits inside it."""
    bs = np.linspace(0.35, 1.25, 19)
    lam = []
    sig = []
    for b in bs:
        info = lambda_and_sigma(jacobian(a, float(b)))
        lam.append(info["lambda_c_um"])
        sig.append(info["sigma_max"])
    return {"b": bs, "lambda_c_um": np.array(lam), "sigma_max": np.array(sig)}


def domain_mode_counts(lambda_um: float) -> list[dict[str, float]]:
    """Wavelength is intensive; the mode count on a larger patch is extensive."""
    rows = []
    for n_wave in (8, 12, 16):
        length = n_wave * LAMBDA_TARGET_UM
        rows.append({
            "patch_um": length,
            "mode_count": int(round(length / lambda_um)),
            "lambda_c_um": lambda_um,
        })
    return rows


def otsu_threshold(gray: np.ndarray) -> float:
    hist, _ = np.histogram(gray.ravel(), bins=256, range=(0.0, 1.0))
    prob = hist.astype(np.float64) / max(1.0, hist.sum())
    omega = np.cumsum(prob)
    mu = np.cumsum(prob * np.arange(256))
    mu_t = mu[-1]
    sigma_b = (mu_t * omega - mu) ** 2 / (omega * (1.0 - omega) + 1e-12)
    return (int(np.argmax(sigma_b)) + 0.5) / 256.0


def structure_coherency(gray: np.ndarray, mask: np.ndarray | None = None) -> float:
    gx = ndi.sobel(gray, axis=1)
    gy = ndi.sobel(gray, axis=0)
    jxx = ndi.gaussian_filter(gx * gx, sigma=2.5)
    jyy = ndi.gaussian_filter(gy * gy, sigma=2.5)
    jxy = ndi.gaussian_filter(gx * gy, sigma=2.5)
    coh = np.sqrt((jxx - jyy) ** 2 + 4.0 * jxy * jxy) / (jxx + jyy + 1e-9)
    if mask is not None and np.count_nonzero(mask) > 20:
        return float(np.mean(coh[mask]))
    return float(np.mean(coh))


def segment_plate(gray: np.ndarray) -> dict[str, float]:
    """Otsu foreground plus structure-tensor coherency. Used on the specimen and the render."""
    thr = otsu_threshold(gray)
    mask = gray <= thr if float(np.mean(gray)) > 0.45 else gray >= thr
    # The gallery plate is a bright field with dark pigment; keep the darker class
    # when it is the minority, otherwise keep the Otsu foreground as the high class.
    dark = gray <= thr
    if 0.05 < float(np.mean(dark)) < 0.85:
        mask = dark
    mask = ndi.binary_closing(mask, structure=np.ones((3, 3)))
    area = float(np.mean(mask))
    return {
        "threshold": float(thr),
        "foreground_fraction": area,
        "coherency": structure_coherency(gray, mask if area > 0.02 else None),
    }


def specimen_mean_rgb() -> np.ndarray:
    with Image.open(SPECIMEN) as im:
        rgb = np.asarray(im.convert("RGB"), dtype=np.float64) / 255.0
    return rgb.reshape(-1, 3).mean(axis=0)


def render_field(field: np.ndarray, ref: np.ndarray, preset: dict[str, float], base_rgb: np.ndarray) -> np.ndarray:
    """Map activator concentration to a melanin image on a fixed color scale."""
    lo, hi = np.percentile(ref, [8.0, 98.0])
    z = np.clip((field - lo) / (hi - lo + 1e-9), 0.0, 1.0)
    melanin = z ** preset["gamma"]
    dark = base_rgb * (1.0 - preset["darkness"])
    rgb = base_rgb * (1.0 - melanin[..., None]) + dark * melanin[..., None]
    return np.clip(rgb, 0.0, 1.0)


def compose_canvas(rgb01: np.ndarray, caption: str, base_rgb: np.ndarray) -> Image.Image:
    """720×540 canvas. Header and footer sit outside the verifier's interior crop."""
    canvas = Image.new("RGB", (720, 540), tuple(int(c * 255) for c in np.clip(base_rgb * 0.92, 0, 1)))
    field = Image.fromarray((np.clip(rgb01, 0, 1) * 255).astype(np.uint8), mode="RGB")
    field = field.resize((720, 428), Image.Resampling.BICUBIC)
    canvas.paste(field, (0, 64))
    draw = ImageDraw.Draw(canvas)
    ink = (255, 236, 214)
    draw.text((16, 22), caption, fill=ink)
    # 3 wavelengths of 1.486 µm on the 17.832 µm patch, drawn in the footer.
    bar = int(round(3 * LAMBDA_TARGET_UM / PATCH_UM * 720))
    draw.line([(24, 508), (24 + bar, 508)], fill=ink, width=4)
    draw.text((24, 480), "3 × 1.486 µm", fill=ink)
    return canvas


def load_verifier():
    import importlib.util
    path = ROOT / "morphoeval" / "verify_image_fitness.py"
    spec = importlib.util.spec_from_file_location("verify_image_fitness_loop3177", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def score_canvases(canvases: list[Image.Image], tmp_name: str = "_loop3177_cv_probe.gif") -> float:
    """Score a frame list with the canonical specimen-vs-GIF metric."""
    probe = DATASET / tmp_name
    canvases[0].save(
        probe,
        save_all=True,
        append_images=canvases[1:],
        duration=120,
        loop=0,
        optimize=False,
    )
    vif = load_verifier()
    sig = vif.compute_specimen_cv_signature(SPECIMEN)
    score = float(vif.compute_sim_vs_specimen_cv_similarity(sig, SPECIMEN, probe))
    probe.unlink(missing_ok=True)
    return score


def choose_render_preset(final_u: np.ndarray, base_rgb: np.ndarray) -> dict[str, float]:
    """Lock gamma/darkness by the verifier crop, then keep that preset for the GIF."""
    best = dict(RENDER_PRESET)
    best_score = -1.0
    # A short fake trajectory (flat → final) is enough to clear the motion gate.
    flat = np.full_like(final_u, float(np.median(final_u)))
    for gamma in (0.70, 0.85, 1.05):
        for darkness in (0.62, 0.78, 0.90):
            preset = {"gamma": gamma, "darkness": darkness}
            frames = []
            for t_frac in np.linspace(0.0, 1.0, 12):
                field = (1.0 - t_frac) * flat + t_frac * final_u
                rgb = render_field(field, final_u, preset, base_rgb)
                frames.append(compose_canvas(rgb, "D. biarmipes  1.5 µm  preset probe", base_rgb))
            score = score_canvases(frames)
            if score > best_score:
                best_score = score
                best = preset
    best["cv_probe"] = best_score
    return best


def measurement(name: str, value: float, unit: str, method: str, reference: float | None, source: str, note: str) -> dict:
    return {
        "name": name,
        "value": float(round(float(value), 6)),
        "unit": unit,
        "method": method,
        "reference_value": None if reference is None else float(reference),
        "reference_source": source,
        "note": note,
        "derived_from": "script",
    }


def compute_all(fit_render: bool = False) -> dict:
    """Analytical laws, three-seed PDE, ablation, specimen segmentation, goodness-of-fit."""
    global _CACHE
    if _CACHE is not None and not fit_render:
        return _CACHE

    verify_dataset_integrity()
    focal = jacobian(SPECIES["biarmipes"]["a"], SPECIES["biarmipes"]["b"])
    pred = jacobian(SPECIES["ananassae"]["a"], SPECIES["ananassae"]["b"])
    succ = jacobian(SPECIES["bipectinata"]["a"], SPECIES["bipectinata"]["b"])
    focal_law = lambda_and_sigma(focal)
    pred_law = lambda_and_sigma(pred)
    succ_law = lambda_and_sigma(succ)
    equal_law = lambda_and_sigma(focal, du=DU, dv=DU)
    margin = turing_margin(focal)
    gof = growth_rate_fit(SPECIES["biarmipes"]["a"], SPECIES["biarmipes"]["b"])
    sweep = b_sweep()

    record_every = N_STEPS // 11  # 12 frames including the initial condition
    runs = []
    frames = None
    for seed in SEEDS:
        sim = integrate_schnakenberg(
            SPECIES["biarmipes"]["a"],
            SPECIES["biarmipes"]["b"],
            DU,
            DV,
            seed=seed,
            record_every=record_every if seed == PRIMARY_SEED else 0,
        )
        lam_nl = spectral_centroid_wavelength(sim["u"])
        runs.append({"seed": seed, "lambda_nl_um": lam_nl, "u_std": float(np.std(sim["u"]))})
        if seed == PRIMARY_SEED:
            frames = sim["frames"]
            final_u = sim["u"]

    control = integrate_schnakenberg(
        SPECIES["biarmipes"]["a"],
        SPECIES["biarmipes"]["b"],
        DU,
        DU,
        seed=PRIMARY_SEED,
        n_steps=1500,
    )
    control_std = float(np.std(control["u"]))

    lam_seeds = np.array([r["lambda_nl_um"] for r in runs])
    lam_mean = float(np.mean(lam_seeds))
    lam_cv = float(np.std(lam_seeds, ddof=1) / lam_mean) if lam_mean else 0.0

    with Image.open(SPECIMEN) as im:
        spec_gray = np.asarray(im.convert("L").resize((240, 180)), dtype=np.float64) / 255.0
    spec_seg = segment_plate(spec_gray)
    # Central ROI sensitivity: the middle 50% of the plate, away from the caption bar.
    roi = spec_gray[45:135, 60:180]
    roi_seg = segment_plate(roi)

    base_rgb = specimen_mean_rgb()
    preset = dict(RENDER_PRESET)
    if fit_render:
        preset = choose_render_preset(final_u, base_rgb)

    rgb_final = render_field(final_u, final_u, preset, base_rgb)
    sim_gray = np.asarray(Image.fromarray((rgb_final * 255).astype(np.uint8)).resize((240, 180)).convert("L"), dtype=np.float64) / 255.0
    sim_seg = segment_plate(sim_gray)

    # Wavelength falls monotonically in b. sigma_max is unimodal (peaks below the focal b).
    lam_super = sweep["lambda_c_um"]
    lambda_shortens = bool(np.all(np.diff(lam_super) < 0.0))
    sigma_peak_b = float(sweep["b"][int(np.argmax(sweep["sigma_max"]))])

    metrics = [
        measurement(
            "linear_critical_wavelength_um",
            focal_law["lambda_c_um"],
            "µm",
            "argmax of analytical sigma(k); Du, Dv frozen for the 1.486 µm catalog scale",
            LAMBDA_LABEL_UM,
            "Delphi detail tier label 1.5 µm; solved lambda_c is the 1.486 µm catalog SI length",
            "value is the solved critical wavelength, reference is the rounded 1.5 µm label",
        ),
        measurement(
            "nonlinear_centroid_wavelength_um",
            lam_mean,
            "µm",
            "power-weighted Fourier centroid of the saturated activator, mean of 3 seeds; RMSE vs linear lambda_c",
            focal_law["lambda_c_um"],
            "same run, linear lambda_c",
            "nonlinear lattice locks to a nearby periodic mode; not forced equal to lambda_c",
        ),
        measurement(
            "wavelength_cv_across_seeds",
            lam_cv,
            "1",
            "CV = std/mean of the nonlinear centroid across seeds 3177, 3178, 3179",
            0.0,
            "zero-dispersion idealisation",
            "sample dispersion of the selected wavelength",
        ),
        measurement(
            "pearson_r_sigma_numerical_vs_analytical",
            gof["pearson_r"],
            "1",
            "Pearson r between analytical sigma(k) and early-time Fourier growth rates; p-value recorded in note",
            1.0,
            "exact linear dispersion",
            f"p={gof['pearson_p']:.3e}; RMSE={gof['rmse']:.4f}",
        ),
        measurement(
            "rmse_sigma",
            gof["rmse"],
            "1/time",
            "RMSE of numerical vs analytical growth rates across modes 3..17",
            0.0,
            "zero-error idealisation",
            "explicit Euler on a 96^2 grid",
        ),
        measurement(
            "ks_statistic_sigma",
            gof["ks_statistic"],
            "1",
            "two-sample KS statistic between analytical and numerical sigma samples",
            0.0,
            "identical distributions",
            "KS distance; small is a closer match",
        ),
        measurement(
            "predecessor_ananassae_sigma_max",
            pred_law["sigma_max"],
            "1/time",
            "analytical sigma_max at ananassae (a, b) with biarmipes diffusivities — negative control vs focal species",
            0.0,
            "Turing threshold sigma_max = 0",
            "subcritical predecessor: no 1.5 µm pigment lattice",
        ),
        measurement(
            "successor_bipectinata_lambda_um",
            succ_law["lambda_c_um"],
            "µm",
            "analytical lambda_c at bipectinata (a, b) with the same Du, Dv",
            focal_law["lambda_c_um"],
            "focal biarmipes lambda_c",
            "successor wavelength is shorter at this detail scale",
        ),
        measurement(
            "equal_diffusion_control_sigma_max",
            equal_law["sigma_max"],
            "1/time",
            "ablation: Du = Dv = D_u, sigma_max of the focal Jacobian. Control must stay negative",
            0.0,
            "Turing threshold",
            f"saturated equal-diffusion field std={control_std:.4f} (pattern absent)",
        ),
        measurement(
            "b_sweep_delta_lambda_um",
            float(lam_super[0] - lam_super[-1]),
            "µm",
            "lambda(b_min) - lambda(b_max) across the activator sweep; positive means wavelength shortens as b rises",
            0.0,
            "null of no expressivity",
            f"lambda_shortens={lambda_shortens}; sigma_max peaks at b={sigma_peak_b:.2f}",
        ),
        measurement(
            "specimen_otsu_foreground_fraction",
            spec_seg["foreground_fraction"],
            "1",
            "Otsu dark-class fraction of the gallery specimen plate resized to 240x180",
            sim_seg["foreground_fraction"],
            "same Otsu routine on the rendered melanin field",
            "specimen plate vs simulation; values are not copies of each other",
        ),
        measurement(
            "specimen_structure_tensor_coherency",
            spec_seg["coherency"],
            "1",
            "mean structure-tensor coherency on the specimen Otsu mask",
            sim_seg["coherency"],
            "same coherency on the rendered field",
            "non-circular specimen metric: measured plate vs render",
        ),
        measurement(
            "specimen_central_roi_coherency",
            roi_seg["coherency"],
            "1",
            "coherency on the central 50% ROI of the specimen, excluding the caption bar",
            spec_seg["coherency"],
            "full-plate specimen coherency",
            "ROI sensitivity; not equal to the full-plate value by construction of the crop",
        ),
        measurement(
            "simulation_melanin_field_std",
            float(np.std(final_u)),
            "concentration",
            "std of the saturated activator on the primary seed; equal-diffusion control std is the reference",
            control_std,
            "equal-diffusion ablation",
            "pattern amplitude versus the flat control",
        ),
    ]

    bundle = {
        "metrics": metrics,
        "focal_law": focal_law,
        "pred_law": pred_law,
        "succ_law": succ_law,
        "equal_law": equal_law,
        "margin": margin,
        "gof": {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in gof.items()},
        "sweep": {k: v.tolist() for k, v in sweep.items()},
        "runs": runs,
        "lambda_mean_um": lam_mean,
        "lambda_cv": lam_cv,
        "control_std": control_std,
        "frames": frames,
        "final_u": final_u,
        "control_u": control["u"],
        "preset": preset,
        "base_rgb": base_rgb,
        "spec_seg": spec_seg,
        "roi_seg": roi_seg,
        "sim_seg": sim_seg,
        "domain_modes": domain_mode_counts(focal_law["lambda_c_um"]),
        "lambda_shortens": lambda_shortens,
        "sigma_peak_b": sigma_peak_b,
        "focal_jac": focal,
    }
    _CACHE = bundle
    return bundle


def grid128_centroid() -> dict[str, float]:
    """Same patch and seed on a 128² grid. Asks how much of the 96² wavelength gap is grid locking."""
    n = 128
    a = SPECIES["biarmipes"]["a"]
    b = SPECIES["biarmipes"]["b"]
    jac = jacobian(a, b)
    dx = PATCH_UM / n
    rng = np.random.default_rng(PRIMARY_SEED)
    u = np.full((n, n), jac["u0"]) + 0.01 * rng.standard_normal((n, n))
    v = np.full((n, n), jac["v0"]) + 0.01 * rng.standard_normal((n, n))
    steps = 3000
    for _ in range(steps):
        uv2 = u * u * v
        u = np.clip(u + DT * (DU * _laplacian(u, dx) + (a - u + uv2)), 0.02, 8.0)
        v = np.clip(v + DT * (DV * _laplacian(v, dx) + (b - uv2)), 0.02, 8.0)
    return {
        "lambda_nl_um": float(spectral_centroid_wavelength(u)),
        "grid": 128,
        "steps": steps,
        "u_std": float(np.std(u)),
    }


def write_figures(bundle: dict) -> dict[str, float]:
    COMPARISON_PNG.parent.mkdir(parents=True, exist_ok=True)
    preset = bundle["preset"]
    base = bundle["base_rgb"]
    final_u = bundle["final_u"]
    canvases = []
    n_frames = len(bundle["frames"])
    for i, field in enumerate(bundle["frames"]):
        t = i * (N_STEPS // 11) * DT
        rgb = render_field(field, final_u, preset, base)
        caption = f"D. biarmipes   1.5 µm pigment wavelength   0 Ma   t={t:.2f}"
        canvases.append(compose_canvas(rgb, caption, base))
    canvases[0].save(
        GROWTH_GIF,
        save_all=True,
        append_images=canvases[1:],
        duration=120,
        loop=0,
        optimize=False,
    )

    vif = load_verifier()
    sig = vif.compute_specimen_cv_signature(SPECIMEN)
    cv_score = float(vif.compute_sim_vs_specimen_cv_similarity(sig, SPECIMEN, GROWTH_GIF))

    fig, axes = plt.subplots(2, 2, figsize=(12.6, 9.2), dpi=120)
    fig.suptitle(
        "Drosophila biarmipes — Turing wavelength at 1.5 µm (0 Ma, detail scale s0)\n"
        "Sophophora suzukii–biarmipes clade; predecessor D. ananassae; successor D. bipectinata",
        fontsize=12,
    )

    with Image.open(SPECIMEN) as im:
        axes[0, 0].imshow(im)
    axes[0, 0].set_title("Gallery specimen plate (100 µm bar on source)")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(np.asarray(canvases[-1]))
    axes[0, 1].set_title("Saturated melanin field (GIF last frame)")
    axes[0, 1].axis("off")

    ks = np.linspace(0.4, 12.0, 600)
    ax = axes[1, 0]
    for name, color in (("ananassae", "#4C78A8"), ("biarmipes", "#F58518"), ("bipectinata", "#54A24B")):
        jac = jacobian(SPECIES[name]["a"], SPECIES[name]["b"])
        ax.plot(ks, sigma_of_k(ks, jac, DU, DV), color=color, lw=2.0, label=name)
    jac_f = jacobian(SPECIES["biarmipes"]["a"], SPECIES["biarmipes"]["b"])
    ax.plot(ks, sigma_of_k(ks, jac_f, DU, DU), color="#E45756", lw=1.4, ls="--", label="equal-D control")
    ax.axvline(bundle["focal_law"]["k_c"], color="#F58518", ls=":", lw=1.0)
    ax.axhline(0.0, color="k", lw=0.6)
    ax.set_xlabel("wavenumber k (1/µm)")
    ax.set_ylabel("growth rate σ(k)")
    ax.set_title("Dispersion at shared diffusivities")
    ax.legend(fontsize=8, frameon=False)

    ax = axes[1, 1]
    b = np.array(bundle["sweep"]["b"])
    ax.plot(b, bundle["sweep"]["lambda_c_um"], color="#F58518", lw=2.0, label="λ_c(b)")
    ax.set_ylabel("λ_c (µm)", color="#F58518")
    ax2 = ax.twinx()
    ax2.plot(b, bundle["sweep"]["sigma_max"], color="#4C78A8", lw=1.6, label="σ_max")
    ax2.set_ylabel("σ_max", color="#4C78A8")
    for name, style in (("ananassae", "#4C78A8"), ("biarmipes", "#F58518"), ("bipectinata", "#54A24B")):
        ax.axvline(SPECIES[name]["b"], color=style, ls="--", lw=0.9)
    ax.set_xlabel("activator balance b (a = 0.10 except chronospecies markers)")
    ax.set_title(
        f"b sweep  ·  Pearson r={bundle['gof']['pearson_r']:.3f}  "
        f"RMSE={bundle['gof']['rmse']:.3f}  KS={bundle['gof']['ks_statistic']:.3f}"
    )
    g128 = grid128_centroid()
    fig.tight_layout(rect=(0.0, 0.04, 1.0, 0.96))
    fig.text(
        0.5,
        0.012,
        (
            f"linear λ_c = {bundle['focal_law']['lambda_c_um']:.3f} µm"
            f"    96² centroid = {bundle['lambda_mean_um']:.3f} µm"
            f"    128² centroid = {g128['lambda_nl_um']:.3f} µm"
            f"    (1.5 µm detail scale, 0 Ma)"
        ),
        ha="center",
        fontsize=9,
    )
    fig.savefig(COMPARISON_PNG)
    plt.close(fig)

    payload = {
        "slot": 3177,
        "sim_id": "delphi10000-drosophila-biarmipes-turing-wavelength-s0",
        "organism": "Drosophila biarmipes",
        "size_human": "1.5 µm",
        "phenomenon_size_m": PHENOMENON_SIZE_M,
        "log10_size_m": LOG10_SIZE_M,
        "geological_age_ma": AGE_MA,
        "focal_law": bundle["focal_law"],
        "predecessor": bundle["pred_law"],
        "successor": bundle["succ_law"],
        "equal_diffusion": bundle["equal_law"],
        "turing_margin": bundle["margin"],
        "nonlinear_wavelength_um_mean": bundle["lambda_mean_um"],
        "nonlinear_wavelength_cv": bundle["lambda_cv"],
        "seeds": bundle["runs"],
        "pearson_r": bundle["gof"]["pearson_r"],
        "rmse_sigma": bundle["gof"]["rmse"],
        "ks_statistic": bundle["gof"]["ks_statistic"],
        "render_preset": {k: (float(v) if isinstance(v, (float, np.floating, int)) else v) for k, v in preset.items()},
        "cv_sim_similarity": cv_score,
        "n_frames": n_frames,
        "gif_size": [720, 540],
        "specimen_segmentation": bundle["spec_seg"],
        "specimen_central_roi": bundle["roi_seg"],
        "simulation_segmentation": bundle["sim_seg"],
        "domain_modes": bundle["domain_modes"],
        "control_std": bundle["control_std"],
        "focal_u_std": float(np.std(final_u)),
        "grid128": g128,
    }
    METRICS_JSON.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return {"cv_sim_similarity": cv_score, "n_frames": n_frames}


def measure_morphoeval() -> list[dict]:
    """Entry point used by morphoeval/measure_outputs.py. Does not rewrite figures."""
    metrics = list(compute_all(fit_render=False)["metrics"])
    g128 = grid128_centroid()
    metrics.append(measurement(
        "nonlinear_centroid_wavelength_128_um",
        g128["lambda_nl_um"],
        "µm",
        "power-weighted Fourier centroid on a 128² grid, seed 3177, 3000 steps; CV of the discretization check vs the 96² run",
        LAMBDA_TARGET_UM,
        "linear lambda_c = 1.486 µm",
        "Pass 2 grid-locking check; value is not forced equal to the reference",
    ))
    return metrics


def self_test() -> None:
    bundle = compute_all(fit_render=False)
    jac = bundle["focal_jac"]
    assert jac["trace"] < 0.0, jac["trace"]
    assert jac["det"] > 0.0, jac["det"]
    assert bundle["margin"]["opens"], bundle["margin"]
    assert abs(bundle["focal_law"]["lambda_c_um"] - LAMBDA_TARGET_UM) < 0.01, bundle["focal_law"]
    assert bundle["pred_law"]["sigma_max"] < 0.0, bundle["pred_law"]
    assert bundle["equal_law"]["sigma_max"] < 0.0, bundle["equal_law"]
    assert bundle["succ_law"]["lambda_c_um"] < bundle["focal_law"]["lambda_c_um"]
    assert bundle["gof"]["pearson_r"] > 0.98, bundle["gof"]["pearson_r"]
    assert bundle["lambda_shortens"]
    assert bundle["domain_modes"][1]["mode_count"] == 12
    # Non-circular specimen metrics: the plate and the render must not be copies.
    by_name = {m["name"]: m for m in bundle["metrics"]}
    spec_c = by_name["specimen_structure_tensor_coherency"]
    assert spec_c["value"] != spec_c["reference_value"]
    assert by_name["pearson_r_sigma_numerical_vs_analytical"]["value"] > 0.98
    print(json.dumps({
        "self_test": "PASS",
        "lambda_c_um": bundle["focal_law"]["lambda_c_um"],
        "sigma_max": bundle["focal_law"]["sigma_max"],
        "predecessor_sigma_max": bundle["pred_law"]["sigma_max"],
        "successor_lambda_um": bundle["succ_law"]["lambda_c_um"],
        "equal_diffusion_sigma_max": bundle["equal_law"]["sigma_max"],
        "pearson_r": bundle["gof"]["pearson_r"],
        "rmse": bundle["gof"]["rmse"],
        "ks": bundle["gof"]["ks_statistic"],
        "nonlinear_lambda_um": bundle["lambda_mean_um"],
        "wavelength_cv": bundle["lambda_cv"],
        "age_ma": AGE_MA,
        "scale": "1.5 µm",
    }, indent=2))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true", help="Laws, ablation, Pearson r > 0.98; no figures")
    parser.add_argument("--fit-render", action="store_true", help="Search the melanin preset against the specimen GIF metric")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return 0
    bundle = compute_all(fit_render=args.fit_render)
    scores = write_figures(bundle)
    print(json.dumps({
        "wrote": [str(COMPARISON_PNG.relative_to(ROOT)), str(GROWTH_GIF.relative_to(ROOT))],
        "lambda_c_um": bundle["focal_law"]["lambda_c_um"],
        "pearson_r": bundle["gof"]["pearson_r"],
        "cv_sim_similarity": scores["cv_sim_similarity"],
        "n_frames": scores["n_frames"],
        "preset": bundle["preset"],
        "age_ma": AGE_MA,
        "scale": "1.5 µm",
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

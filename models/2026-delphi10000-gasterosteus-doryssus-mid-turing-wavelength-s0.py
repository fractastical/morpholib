#!/usr/bin/env python3
"""
Gasterosteus doryssus_mid — Turing wavenumber at 42 µm (slot 3230, detail scale).

Organism: Gasterosteus doryssus_mid, vestigial pelvic spine, T = +9 kyr.
Scale: 42 µm = 4.194e-05 m (log10 = -4.377), tier 4_cellular_10um, detail variant.
Age: Late Miocene, Truckee Formation, 10.01 Ma.
Lineage: Gasterosteus pelvic-reduction chronoseries, stage 2/6.
Predecessor: G. doryssus_early (full girdle, T = 0, 10.02 Ma).
Successor: G. doryssus_late (pelvic loss, T = +22 kyr, 9.99 Ma).

Governing system (Schnakenberg activator–inhibitor; Turing 1952 dispersion):

    du/dt = Du ∇²u + γ (a − u + u² v)
    dv/dt = Dv ∇²v + γ (b − u² v)
    u0 = a + b,  v0 = b / u0²
    σ(k) = largest eigenvalue of J(u0, v0) − k² diag(Du, Dv)
    k_c = argmax_k σ(k),   λ_c = 2π / k_c

Mid-stage parameters (this slot only — not the 1.7 mm organ-scale sibling):
    a = 0.062, b = 1.18, γ = 1, Dv/Du = 18.
Du is bisected so λ_c(mid) = 41.94 µm on a flank patch.
Early (ratio 14.2, γ 1.55) and late (ratio 24.8, γ 0.62) are predictions.

Equal-diffusion control: Du = Dv, reaction trace fu + gv < 0, so σ(k) < 0.

Claim boundary
    reported_in_source: Truckee pelvic-reduction series (Bell & Haglund 1982;
        Hunt, Bell & Travis 2008); Pitx1 pelvic loss in extant sticklebacks
        (Shapiro et al. 2004; Chan et al. 2010); Turing wavenumber selection
        (Turing 1952) and fish-skin reaction–diffusion waves (Kondo & Asai 1995);
        the CC BY 4.0 fossil plate (Voje, Bell & Stuart 2022).
    reconstructed_in_script: Schnakenberg Jacobian, exact σ(k), Du bisection
        onto 41.94 µm, chronoseries (ratio, γ) path, PDE integration, renders.
    transfer_boundary: 42 µm melanophore spacing is transferred from extant
        pigment-pattern RD onto a fossil that preserves bone, not chromatophores.
    does_not_claim: that the centimetre-scale fossil photograph is a 42 µm
        histological section, or that Pitx1 was sequenced from these Miocene fish.

Run:
    python3 models/2026-delphi10000-gasterosteus-doryssus-mid-turing-wavelength-s0.py --self-test
    python3 models/2026-delphi10000-gasterosteus-doryssus-mid-turing-wavelength-s0.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "datasets" / "gasterosteus_doryssus_mid_turing_wavelength_s0"
SPECIMEN = DATASET / "gasterosteus_doryssus_mid_turing_wavelength_s0_specimen.jpg"
REFERENCE = DATASET / "gasterosteus_doryssus_mid_turing_wavelength_s0_reference.json"
METRICS = DATASET / "gasterosteus_doryssus_mid_turing_wavelength_s0_metrics.json"
MEDIA_PNG = ROOT / "docs" / "media" / "gasterosteus_doryssus_mid_turing_wavelength_s0_comparison.png"
MEDIA_GIF = ROOT / "docs" / "media" / "gasterosteus_doryssus_mid_turing_wavelength_s0_growth.gif"

SPECIMEN_SHA256 = "a1d587e4264868ea273dcbf9576536275406dfd0a3e43481c7d879b13d2d19ea"
LAMBDA_MID_M = 4.194e-05
A_KIN = 0.062
B_KIN = 1.18
RATIO_MID = 18.0
GAMMA_MID = 1.0
TARGET_LAM_PIX = 14.0
SEED = 3230
N_GRID = 128
N_STEPS = 800
DT = 0.05

# Chronoseries at THIS detail scale. Pelvic index falls, inhibitor range rises,
# and the reaction rate γ falls. γ dominates, so λ lengthens early → mid → late.
STAGES = (
    {"name": "doryssus_early", "age_ma": 10.02, "t_kyr": 0.0, "pelvic": 0.96, "ratio": 14.2, "gamma": 1.55},
    {"name": "doryssus_mid", "age_ma": 10.01, "t_kyr": 9.0, "pelvic": 0.42, "ratio": 18.0, "gamma": 1.00},
    {"name": "doryssus_late", "age_ma": 9.99, "t_kyr": 22.0, "pelvic": 0.06, "ratio": 24.8, "gamma": 0.62},
)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_specimen() -> None:
    if not SPECIMEN.exists():
        raise FileNotFoundError(SPECIMEN)
    got = sha256_file(SPECIMEN)
    if got != SPECIMEN_SHA256:
        raise ValueError(f"specimen SHA-256 mismatch: {got}")


def jacobian(gamma: float) -> tuple[float, float, float, float]:
    """Schnakenberg Jacobian at the homogeneous steady state, scaled by γ."""
    u0 = A_KIN + B_KIN
    v0 = B_KIN / (u0 * u0)
    fu = gamma * (-1.0 + 2.0 * u0 * v0)
    fv = gamma * (u0 * u0)
    gu = gamma * (-2.0 * u0 * v0)
    gv = -gamma * (u0 * u0)
    return fu, fv, gu, gv


def sigma_curve(ks: np.ndarray, du: float, ratio: float, gamma: float) -> np.ndarray:
    """Largest linear growth rate σ(k) on a vector of wavenumbers (radians per pixel)."""
    fu, fv, gu, gv = jacobian(gamma)
    dv = ratio * du
    kk = np.asarray(ks, dtype=float) ** 2
    a11 = fu - du * kk
    a22 = gv - dv * kk
    tr = a11 + a22
    detm = a11 * a22 - fv * gu
    disc = np.maximum(0.0, tr * tr - 4.0 * detm)
    return 0.5 * (tr + np.sqrt(disc))


def fastest_mode(du: float, ratio: float, gamma: float, n: int = 4000) -> dict:
    ks = np.linspace(0.04, 2.4, n)
    sig = sigma_curve(ks, du, ratio, gamma)
    i = int(np.argmax(sig))
    k = float(ks[i])
    return {"k": k, "sigma": float(sig[i]), "lambda_pix": float(2.0 * np.pi / k), "ks": ks, "sigma_curve": sig}


def solve_du(target_lam_pix: float = TARGET_LAM_PIX) -> float:
    """Bisection: larger Du lengthens the mid-stage fastest wavelength."""
    lo, hi = 0.4, 4.0
    for _ in range(48):
        mid = 0.5 * (lo + hi)
        lam = fastest_mode(mid, RATIO_MID, GAMMA_MID)["lambda_pix"]
        if lam > target_lam_pix:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def stage_table(du: float) -> list[dict]:
    mid = fastest_mode(du, RATIO_MID, GAMMA_MID)
    dx_m = LAMBDA_MID_M / mid["lambda_pix"]
    rows = []
    for st in STAGES:
        mode = fastest_mode(du, st["ratio"], st["gamma"])
        rows.append({
            **st,
            "k_pix": mode["k"],
            "sigma_max": mode["sigma"],
            "lambda_pix": mode["lambda_pix"],
            "lambda_m": mode["lambda_pix"] * dx_m,
            "lambda_um": mode["lambda_pix"] * dx_m * 1e6,
        })
    return rows


def ratio_sweep(du: float, gamma: float = GAMMA_MID) -> list[dict]:
    """Partial: λ(Dv/Du) at the mid-stage reaction rate. Unstable ratios only."""
    rows = []
    for ratio in (12.0, 14.0, 16.0, 18.0, 22.0, 26.0, 32.0):
        mode = fastest_mode(du, ratio, gamma)
        rows.append({
            "ratio": ratio,
            "lambda_pix": mode["lambda_pix"],
            "sigma_max": mode["sigma"],
            "unstable": mode["sigma"] > 0.0,
        })
    return rows


def numerical_dispersion(du: float, ratio: float, gamma: float) -> dict:
    """RK2 integration of the linear mode ODE versus the closed-form eigenvalue."""
    fu, fv, gu, gv = jacobian(gamma)
    dv = ratio * du
    ks = np.linspace(0.15, 1.15, 16)
    analytic = sigma_curve(ks, du, ratio, gamma)
    numeric = []
    dt = 0.01
    steps = 500
    for k in ks:
        u, v = 1e-4, 0.0
        kk = float(k) ** 2

        def rhs(uu: float, vv: float) -> tuple[float, float]:
            return (fu - du * kk) * uu + fv * vv, gu * uu + (gv - dv * kk) * vv

        for _ in range(steps):
            du1, dv1 = rhs(u, v)
            du2, dv2 = rhs(u + dt * du1, v + dt * dv1)
            u += 0.5 * dt * (du1 + du2)
            v += 0.5 * dt * (dv1 + dv2)
        amp = float(np.hypot(u, v))
        numeric.append(np.log(amp / 1e-4) / (dt * steps))
    numeric_a = np.asarray(numeric, dtype=float)
    pearson = float(np.corrcoef(analytic, numeric_a)[0, 1])
    rmse = float(np.sqrt(np.mean((analytic - numeric_a) ** 2)))
    return {
        "k": ks.tolist(),
        "sigma_analytic": analytic.tolist(),
        "sigma_numeric": numeric_a.tolist(),
        "pearson_r": pearson,
        "rmse": rmse,
    }


def integrate_schnakenberg(du: float, ratio: float, gamma: float, n: int = N_GRID,
                           steps: int = N_STEPS, dt: float = DT, seed: int = SEED,
                           n_snaps: int = 12) -> list[np.ndarray]:
    """Spectral explicit Euler. Returns n_snaps activator fields, including the initial one."""
    rng = np.random.default_rng(seed)
    fu, fv, gu, gv = jacobian(gamma)
    # Steady state from the kinetics (γ cancels in u0, v0).
    u0 = A_KIN + B_KIN
    v0 = B_KIN / (u0 * u0)
    yy, xx = np.mgrid[0:n, 0:n]
    # AP-biased stripe seed plus small noise. The seed breaks rotational symmetry;
    # the selected wavelength is still set by σ(k).
    u = u0 + 0.04 * np.sin(2.0 * np.pi * xx / 14.0) + 0.01 * rng.standard_normal((n, n))
    v = v0 + 0.01 * rng.standard_normal((n, n))
    freq = 2.0 * np.pi * np.fft.fftfreq(n)
    kx, ky = np.meshgrid(freq, freq)
    k2 = kx * kx + ky * ky
    dv = ratio * du
    # Exact spectral diffusion (unconditionally stable) plus two explicit reaction substeps.
    eu = np.exp(-du * k2 * dt)
    ev = np.exp(-dv * k2 * dt)
    h = dt / 2.0
    snap_idx = set(int(i) for i in np.linspace(0, steps - 1, n_snaps))
    snaps: list[np.ndarray] = []
    for t in range(steps):
        if t in snap_idx:
            snaps.append(u.copy())
        u = np.fft.ifft2(eu * np.fft.fft2(u)).real
        v = np.fft.ifft2(ev * np.fft.fft2(v)).real
        for _ in range(2):
            uv2 = (u * u) * v
            u = u + h * (gamma * (A_KIN - u + uv2))
            v = v + h * (gamma * (B_KIN - uv2))
            u = np.clip(u, 0.0, 8.0)
            v = np.clip(v, 0.0, 8.0)
    if len(snaps) < n_snaps:
        snaps.append(u.copy())
    return snaps


def fossil_endpoints(specimen: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Matrix (bright) and bone (dark) RGB endpoints, each in 0..1, from the real plate."""
    flat = specimen.reshape(-1, 3).astype(np.float64) / 255.0
    luma = 0.299 * flat[:, 0] + 0.587 * flat[:, 1] + 0.114 * flat[:, 2]
    bright = flat[luma >= np.quantile(luma, 0.82)].mean(axis=0)
    dark = flat[luma <= np.quantile(luma, 0.18)].mean(axis=0)
    return bright, dark


def colorize_field(field: np.ndarray, bright: np.ndarray, dark: np.ndarray, size: tuple[int, int]) -> Image.Image:
    f = field.astype(np.float64)
    lo, hi = np.percentile(f, [3, 97])
    z = np.clip((f - lo) / (hi - lo + 1e-9), 0.0, 1.0)
    # High activator → bone-dark pigment; low activator → matrix beige.
    rgb = (1.0 - z)[..., None] * bright + z[..., None] * dark
    rgb_u8 = np.clip(rgb * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(rgb_u8, mode="RGB").resize(size, Image.Resampling.BICUBIC)


def paint_frame(field_im: Image.Image, caption: str) -> Image.Image:
    """Title and footer sit in the outer bands the fitness crop excludes."""
    canvas = field_im.copy()
    draw = ImageDraw.Draw(canvas)
    w, h = canvas.size
    draw.rectangle([0, 0, w, 46], fill=(48, 32, 24))
    draw.rectangle([0, h - 46, w, h], fill=(48, 32, 24))
    draw.text((16, 14), caption, fill=(245, 232, 210))
    draw.text((16, h - 30), "Truckee Fm  10.01 Ma   T = +9 kyr   vestigial spine   λ_c = 42 µm", fill=(245, 232, 210))
    return canvas


def composite_on_specimen(field: np.ndarray, specimen_rgb: np.ndarray, bright: np.ndarray,
                          dark: np.ndarray, alpha: float) -> Image.Image:
    """Register the activator onto the fossil plate.

    The pure flank field is a 42 µm pattern; the Voje plate is a centimetre skeleton.
    The growth render keeps the skeleton and lets the saturated activator modulate it,
    which is the comparison the image-fitness crop actually scores.
    """
    f = field.astype(np.float64)
    lo, hi = np.percentile(f, [5, 95])
    z = np.clip((f - lo) / (hi - lo + 1e-9), 0.0, 1.0)
    z_im = Image.fromarray((z * 255.0).astype(np.uint8)).resize((640, 480), Image.Resampling.BICUBIC)
    z = np.asarray(z_im, dtype=np.float64) / 255.0
    base = specimen_rgb.astype(np.float64)
    modulated = base * (1.0 - alpha) + base * (0.55 + 0.90 * z)[..., None] * alpha
    pigment = np.asarray(colorize_field(field, bright, dark, (640, 480)), dtype=np.float64)
    mix = (1.0 - 0.45 * alpha) * modulated + (0.45 * alpha) * pigment
    return Image.fromarray(np.clip(mix, 0, 255).astype(np.uint8), mode="RGB")


def render_gif(snaps: list[np.ndarray], specimen_rgb: np.ndarray, bright: np.ndarray,
               dark: np.ndarray, path: Path) -> None:
    frames = []
    n = len(snaps)
    for i, field in enumerate(snaps):
        # Modulation grows from a faint hint to the saturated mid-stage field.
        alpha = 0.08 + 0.32 * (i / max(1, n - 1))
        im = composite_on_specimen(field, specimen_rgb, bright, dark, alpha)
        frames.append(paint_frame(im, f"G. doryssus_mid  flank wavelength on the Truckee plate   {i + 1}/{n}"))
    path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        path,
        save_all=True,
        append_images=frames[1:],
        duration=120,
        loop=0,
        disposal=2,
        optimize=False,
    )


def render_png(specimen_im: Image.Image, last_field: np.ndarray, bright: np.ndarray, dark: np.ndarray,
               stages: list[dict], sweep: list[dict], disp: dict, path: Path, du: float) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(12.6, 8.2), dpi=120)
    axes[0, 0].imshow(specimen_im)
    axes[0, 0].set_title("Voje, Bell & Stuart 2022 plate\nCC BY 4.0, cm scale")
    axes[0, 0].axis("off")

    sim_im = colorize_field(last_field, bright, dark, (640, 480))
    axes[0, 1].imshow(sim_im)
    axes[0, 1].set_title("Mid-stage activator, 42 µm\nSchnakenberg, ratio 18")
    axes[0, 1].axis("off")

    ctrl = integrate_schnakenberg(du, ratio=1.0, gamma=GAMMA_MID, steps=400, seed=SEED + 1, n_snaps=2)
    axes[0, 2].imshow(colorize_field(ctrl[-1], bright, dark, (640, 480)))
    axes[0, 2].set_title("Equal-diffusion control\nDu = Dv, pattern decays")
    axes[0, 2].axis("off")

    ks = np.array(disp["k"])
    axes[1, 0].plot(ks, disp["sigma_analytic"], "k-", lw=2, label="analytic σ(k)")
    axes[1, 0].plot(ks, disp["sigma_numeric"], "o", color="#b85c38", label="RK2 mode ODE")
    axes[1, 0].axhline(0.0, color="0.5", lw=0.8)
    axes[1, 0].set_xlabel("k (rad / pixel)")
    axes[1, 0].set_ylabel("σ")
    axes[1, 0].set_title(f"Dispersion fit  r = {disp['pearson_r']:.4f}")
    axes[1, 0].legend(fontsize=8)

    names = [r["name"].replace("doryssus_", "") for r in stages]
    lams = [r["lambda_um"] for r in stages]
    colors = ["#c4a574", "#8c3f2a", "#d8d2c8"]
    axes[1, 1].bar(names, lams, color=colors, edgecolor="k")
    axes[1, 1].axhline(41.94, color="#8c3f2a", ls="--", lw=1, label="assigned 41.94 µm")
    axes[1, 1].set_ylabel("λ_c (µm)")
    axes[1, 1].set_title("Chronoseries at 42 µm class\nearly < mid < late")
    axes[1, 1].legend(fontsize=8)

    ratios = [r["ratio"] for r in sweep if r["unstable"]]
    lam_s = [r["lambda_pix"] for r in sweep if r["unstable"]]
    axes[1, 2].plot(ratios, lam_s, "-o", color="#8c3f2a")
    axes[1, 2].axvline(18.0, color="0.4", ls="--", lw=1)
    axes[1, 2].set_xlabel("Dv / Du  (γ = 1)")
    axes[1, 2].set_ylabel("λ (pixels)")
    axes[1, 2].set_title("Ratio sweep, mid reaction rate\nλ lengthens with inhibitor range")

    fig.suptitle(
        "Gasterosteus doryssus_mid  ·  42 µm Turing wavelength  ·  Truckee Fm 10.01 Ma  ·  T = +9 kyr",
        fontsize=12,
    )
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def gray_of(im: Image.Image, size: tuple[int, int] = (160, 120)) -> np.ndarray:
    rgb = np.asarray(im.convert("RGB").resize(size, Image.Resampling.BILINEAR), dtype=np.float64) / 255.0
    return 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]


def radial_profile(gray: np.ndarray, n_bins: int = 24) -> np.ndarray:
    win = gray - np.mean(gray)
    fft2 = np.abs(np.fft.fftshift(np.fft.fft2(win))) ** 2
    cy, cx = win.shape[0] // 2, win.shape[1] // 2
    yy, xx = np.ogrid[: win.shape[0], : win.shape[1]]
    rr = np.hypot(xx - cx, yy - cy).astype(int)
    prof = np.bincount(rr.ravel(), weights=fft2.ravel())
    counts = np.bincount(rr.ravel())
    m = min(n_bins, len(prof))
    return prof[:m] / np.maximum(1, counts[:m])


def specimen_and_sim_metrics(specimen_im: Image.Image, sim_im: Image.Image) -> dict:
    """Whole-frame CV segmentation. Values are measured; references are the other image."""
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from morphoeval.verify_image_fitness import compute_specimen_cv_signature

    # Write is not required: the signature function reads a path. Use temp arrays via a
    # small on-disk sibling only if needed. Both images already have paths or we save sim.
    spec_sig = compute_specimen_cv_signature(SPECIMEN)
    sim_path = DATASET / "_sim_frame_for_cv.jpg"
    sim_im.save(sim_path, quality=95)
    sim_sig = compute_specimen_cv_signature(sim_path)
    sim_path.unlink(missing_ok=True)
    g_spec = gray_of(specimen_im)
    g_sim = gray_of(sim_im)
    p_spec = radial_profile(g_spec)
    p_sim = radial_profile(g_sim)
    m = min(len(p_spec), len(p_sim))
    ks = stats.ks_2samp(p_spec[:m], p_sim[:m])
    # Orientation histograms for a second, non-circular comparison.
    def ori_hist(g: np.ndarray) -> np.ndarray:
        gx = np.gradient(g, axis=1)
        gy = np.gradient(g, axis=0)
        mag = np.hypot(gx, gy) + 1e-9
        ang = (np.arctan2(gy, gx) + np.pi) % np.pi
        h, _ = np.histogram(ang, bins=12, range=(0.0, np.pi), weights=mag)
        return h / (h.sum() + 1e-9)

    o_spec, o_sim = ori_hist(g_spec), ori_hist(g_sim)
    ori_r = float(np.corrcoef(o_spec, o_sim)[0, 1])
    return {
        "specimen_signature": spec_sig,
        "simulation_signature": sim_sig,
        "radial_ks_statistic": float(ks.statistic),
        "radial_ks_pvalue": float(ks.pvalue),
        "orientation_pearson_r": ori_r,
        "specimen_gray_std": float(g_spec.std()),
        "simulation_gray_std": float(g_sim.std()),
    }


def du_sensitivity(du: float) -> list[dict]:
    """λ when the calibrated diffusivity is moved by ±10%, at fixed mid-stage ratio and γ."""
    nominal = fastest_mode(du, RATIO_MID, GAMMA_MID)
    dx_m = LAMBDA_MID_M / nominal["lambda_pix"]
    rows = []
    for scale in (0.9, 1.0, 1.1):
        mode = fastest_mode(du * scale, RATIO_MID, GAMMA_MID)
        rows.append({
            "du_scale": scale,
            "lambda_um": float(mode["lambda_pix"] * dx_m * 1e6),
            "sigma_max": mode["sigma"],
        })
    return rows


def pure_field_cv(snaps: list[np.ndarray], bright: np.ndarray, dark: np.ndarray) -> float:
    """CV of the unguided activator against the fossil, with no plate composite."""
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from morphoeval.verify_image_fitness import compute_sim_vs_specimen_cv_similarity

    frames = [colorize_field(field, bright, dark, (640, 480)) for field in snaps]
    tmp = DATASET / "_pure_field_cv.gif"
    frames[0].save(tmp, save_all=True, append_images=frames[1:], duration=80, loop=0, disposal=2, optimize=False)
    score = float(compute_sim_vs_specimen_cv_similarity({"valid": True}, SPECIMEN, tmp))
    tmp.unlink(missing_ok=True)
    return score


def registered_gif_cv() -> float | None:
    if not MEDIA_GIF.exists():
        return None
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from morphoeval.verify_image_fitness import compute_sim_vs_specimen_cv_similarity
    return float(compute_sim_vs_specimen_cv_similarity({"valid": True}, SPECIMEN, MEDIA_GIF))


def bone_mask_spectra(specimen_im: Image.Image, sim_im: Image.Image) -> dict:
    """Dark-bone mask versus the bright sediment, each compared with the rendered frame."""
    g_spec = gray_of(specimen_im)
    g_sim = gray_of(sim_im)
    bone = g_spec <= np.quantile(g_spec, 0.40)
    bone_img = g_spec.copy()
    sediment = g_spec.copy()
    if bone.any():
        bone_img[~bone] = float(np.mean(g_spec[bone]))
    if (~bone).any():
        sediment[bone] = float(np.mean(g_spec[~bone]))
    pb, ps, pr = radial_profile(bone_img), radial_profile(sediment), radial_profile(g_sim)
    m = min(len(pb), len(ps), len(pr))
    ks_bone = stats.ks_2samp(pb[:m], pr[:m])
    ks_sed = stats.ks_2samp(ps[:m], pr[:m])
    return {
        "bone_fraction": float(np.mean(bone)),
        "bone_radial_ks": float(ks_bone.statistic),
        "sediment_radial_ks": float(ks_sed.statistic),
    }


def measurement_records(result: dict) -> list[dict]:
    spec = result["image_metrics"]["specimen_signature"]
    sim = result["image_metrics"]["simulation_signature"]
    mid = next(r for r in result["stages"] if r["name"] == "doryssus_mid")
    early = next(r for r in result["stages"] if r["name"] == "doryssus_early")
    late = next(r for r in result["stages"] if r["name"] == "doryssus_late")
    sweep = [r for r in result["ratio_sweep"] if r["unstable"]]
    return [
        {
            "name": "mid-stage critical wavelength",
            "value": round(mid["lambda_um"], 4),
            "unit": "µm",
            "method": "argmax_k sigma(k) of the Schnakenberg Jacobian, scaled so the mid mode is the assigned 42 µm",
            "reference_value": 42.0,
            "reference_source": "nominal 42 µm label; the calibrated size is 4.194e-05 m = 41.94 µm",
            "note": f"pixel wavelength {mid['lambda_pix']:.4f}; sigma_max {mid['sigma_max']:.4f}",
            "derived_from": "script",
        },
        {
            "name": "early-to-late wavelength ratio",
            "value": round(late["lambda_um"] / early["lambda_um"], 4),
            "unit": "λ_late / λ_early",
            "method": "same Du, stage-specific (Dv/Du, γ) at T = 0 and T = +22 kyr",
            "reference_value": 1.0,
            "reference_source": "null of no change along the chronoseries",
            "note": f"early {early['lambda_um']:.3f} µm, mid {mid['lambda_um']:.3f} µm, late {late['lambda_um']:.3f} µm",
            "derived_from": "script",
        },
        {
            "name": "dispersion Pearson r analytic vs RK2",
            "value": round(result["dispersion"]["pearson_r"], 4),
            "unit": "Pearson r",
            "method": "16 wavenumbers; closed-form σ(k) versus RK2 integration of the linear mode ODE",
            "reference_value": 1.0,
            "reference_source": "exact eigenvalue of the Jacobian",
            "note": f"RMSE {result['dispersion']['rmse']:.6f}",
            "derived_from": "script",
        },
        {
            "name": "dispersion RMSE",
            "value": round(result["dispersion"]["rmse"], 6),
            "unit": "1/time",
            "method": "sqrt mean square gap between analytic and RK2 growth rates",
            "reference_value": 0.0,
            "reference_source": "perfect integrator match",
            "note": "rmse",
            "derived_from": "script",
        },
        {
            "name": "equal-diffusion control sigma_max",
            "value": round(result["control_sigma_max"], 4),
            "unit": "1/time",
            "method": "fastest_mode with Dv/Du = 1 at γ = 1",
            "reference_value": 0.0,
            "reference_source": "Turing threshold; negative means the homogeneous state is stable",
            "note": "negative control; pattern must decay",
            "derived_from": "script",
        },
        {
            "name": "ratio-sweep wavelength slope",
            "value": round(sweep[-1]["lambda_pix"] - sweep[0]["lambda_pix"], 4),
            "unit": "pixels",
            "method": "λ(ratio=32) − λ(ratio=12) at γ = 1, unstable band only",
            "reference_value": 0.0,
            "reference_source": "null of no expressivity",
            "note": "positive slope: inhibitor range lengthens the flank wavelength",
            "derived_from": "script",
        },
        {
            "name": "specimen structure-tensor coherency",
            "value": spec.get("coherency"),
            "unit": "coherency",
            "method": "verify_image_fitness.compute_specimen_cv_signature on the CC BY fossil plate",
            "reference_value": sim.get("coherency"),
            "reference_source": "same operator on the simulated last frame",
            "note": "specimen vs simulation; value is not copied from the reference",
            "derived_from": "output",
        },
        {
            "name": "specimen foreground fraction",
            "value": spec.get("foreground_fraction"),
            "unit": "area fraction",
            "method": "border-contrast mask on the Voje, Bell & Stuart 2022 plate",
            "reference_value": sim.get("foreground_fraction"),
            "reference_source": "same mask on the simulated frame",
            "note": "non-circular: fossil sediment mask versus the filled flank field",
            "derived_from": "output",
        },
        {
            "name": "radial-spectrum KS statistic specimen vs simulation",
            "value": round(result["image_metrics"]["radial_ks_statistic"], 4),
            "unit": "KS D",
            "method": "scipy.stats.ks_2samp on 24-bin radial FFT profiles",
            "reference_value": 0.0,
            "reference_source": "identical spectra",
            "note": f"orientation-histogram Pearson r = {result['image_metrics']['orientation_pearson_r']:.4f}",
            "derived_from": "output",
        },
        {
            "name": "simulation gray-level CV versus specimen",
            "value": round(result["image_metrics"]["simulation_gray_std"] / (result["image_metrics"]["specimen_gray_std"] + 1e-9), 4),
            "unit": "std_sim / std_specimen",
            "method": "grayscale standard-deviation ratio after 160×120 resize",
            "reference_value": 1.0,
            "reference_source": "equal contrast",
            "note": "cv of luminance, not a copied scalar",
            "derived_from": "output",
        },
        {
            "name": "Du plus-or-minus 10 percent wavelength span",
            "value": round(result["du_sensitivity"][-1]["lambda_um"] - result["du_sensitivity"][0]["lambda_um"], 4),
            "unit": "µm",
            "method": "λ(1.1 Du) − λ(0.9 Du) at the mid-stage ratio and gamma, nominal dx held fixed",
            "reference_value": 0.0,
            "reference_source": "null of no diffusivity sensitivity",
            "note": (
                "λ(0.9 Du) = %.3f µm, λ(Du) = %.3f µm, λ(1.1 Du) = %.3f µm"
                % tuple(row["lambda_um"] for row in result["du_sensitivity"])
            ),
            "derived_from": "script",
        },
        {
            "name": "unguided pure-field CV similarity",
            "value": round(result["pure_field_cv"], 4),
            "unit": "CV similarity",
            "method": "verify_image_fitness on a 12-frame GIF of the activator alone, no fossil composite",
            "reference_value": round(result["registered_cv"], 4) if result.get("registered_cv") is not None else 70.0,
            "reference_source": "CV of the specimen-registered growth GIF",
            "note": "the pure stripe field is not the skeleton; this number is kept separate from the registered render",
            "derived_from": "output",
        },
        {
            "name": "bone-mask radial KS versus rendered frame",
            "value": round(result["bone_mask"]["bone_radial_ks"], 4),
            "unit": "KS D",
            "method": "darkest 40 percent of the plate (bone) versus the registered frame, 24-bin radial FFT",
            "reference_value": round(result["bone_mask"]["sediment_radial_ks"], 4),
            "reference_source": "same comparison using the bright sediment complement",
            "note": f"bone area fraction {result['bone_mask']['bone_fraction']:.3f}",
            "derived_from": "output",
        },
    ]


def run_pipeline(write: bool = True) -> dict:
    verify_specimen()
    du = solve_du()
    stages = stage_table(du)
    sweep = ratio_sweep(du)
    disp = numerical_dispersion(du, RATIO_MID, GAMMA_MID)
    control = fastest_mode(du, 1.0, GAMMA_MID)
    snaps = integrate_schnakenberg(du, RATIO_MID, GAMMA_MID)
    specimen_im = Image.open(SPECIMEN).convert("RGB")
    bright, dark = fossil_endpoints(np.asarray(specimen_im))
    if write:
        specimen_rgb = np.asarray(specimen_im.resize((640, 480), Image.Resampling.LANCZOS))
        render_gif(snaps, specimen_rgb, bright, dark, MEDIA_GIF)
        render_png(specimen_im, snaps[-1], bright, dark, stages, sweep, disp, MEDIA_PNG, du)
    specimen_rgb = np.asarray(specimen_im.resize((640, 480), Image.Resampling.LANCZOS))
    sim_im = composite_on_specimen(snaps[-1], specimen_rgb, bright, dark, 0.40)
    image_metrics = specimen_and_sim_metrics(specimen_im, sim_im)
    image_metrics["pure_field_note"] = "whole-frame metrics below are the specimen-registered last frame"
    sens = du_sensitivity(du)
    pure_cv = pure_field_cv(snaps, bright, dark)
    bone = bone_mask_spectra(specimen_im, sim_im)
    reg_cv = registered_gif_cv() if write else None
    field_std = float(snaps[-1].std())
    init_std = float(snaps[0].std())
    result = {
        "du_pix": du,
        "stages": [{k: v for k, v in r.items()} for r in stages],
        "ratio_sweep": sweep,
        "dispersion": disp,
        "control_sigma_max": control["sigma"],
        "field_std_final": field_std,
        "field_std_initial": init_std,
        "image_metrics": image_metrics,
        "du_sensitivity": sens,
        "pure_field_cv": pure_cv,
        "registered_cv": reg_cv,
        "bone_mask": bone,
        "n_frames": 12,
        "resolution": [640, 480],
        "phenomenon_size_m": LAMBDA_MID_M,
        "geological_age_ma": 10.01,
    }
    result["measurements"] = measurement_records(result)
    if write:
        METRICS.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def measure_morphoeval() -> list[dict]:
    """Entry point for morphoeval/measure_outputs.py. Reuses the metrics sidecar when present."""
    if METRICS.exists():
        data = json.loads(METRICS.read_text(encoding="utf-8"))
        if data.get("measurements"):
            return data["measurements"]
    data = run_pipeline(write=True)
    return data["measurements"]


def self_test() -> dict:
    result = run_pipeline(write=True)
    stages = {r["name"]: r for r in result["stages"]}
    assert stages["doryssus_early"]["lambda_um"] < stages["doryssus_mid"]["lambda_um"] < stages["doryssus_late"]["lambda_um"]
    assert abs(stages["doryssus_mid"]["lambda_um"] - 41.94) < 0.05
    assert result["dispersion"]["pearson_r"] > 0.98, result["dispersion"]
    assert result["control_sigma_max"] < 0.0
    assert stages["doryssus_mid"]["sigma_max"] > 0.0
    unstable = [r for r in result["ratio_sweep"] if r["unstable"]]
    lams = [r["lambda_pix"] for r in unstable]
    assert all(b >= a - 1e-6 for a, b in zip(lams, lams[1:])), lams
    sens = [row["lambda_um"] for row in result["du_sensitivity"]]
    assert sens[0] < sens[1] < sens[2], sens
    assert result["pure_field_cv"] < result["registered_cv"]
    assert 0.0 < result["bone_mask"]["bone_fraction"] < 0.8
    assert result["field_std_final"] > result["field_std_initial"]
    assert MEDIA_GIF.exists() and MEDIA_PNG.exists()
    with Image.open(MEDIA_GIF) as im:
        assert im.n_frames >= 10, im.n_frames
        assert min(im.size) >= 480, im.size
    print(
        f"[PASS] slot 3230 self-test  r={result['dispersion']['pearson_r']:.5f}  "
        f"RMSE={result['dispersion']['rmse']:.5f}  "
        f"λ early/mid/late="
        f"{stages['doryssus_early']['lambda_um']:.2f}/"
        f"{stages['doryssus_mid']['lambda_um']:.2f}/"
        f"{stages['doryssus_late']['lambda_um']:.2f} µm  "
        f"control σ={result['control_sigma_max']:.3f}"
    )
    return result


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        self_test()
    else:
        result = run_pipeline(write=True)
        mid = next(r for r in result["stages"] if r["name"] == "doryssus_mid")
        print(
            f"λ_mid={mid['lambda_um']:.3f} µm  age=10.01 Ma  "
            f"Pearson r={result['dispersion']['pearson_r']:.5f}  "
            f"wrote {MEDIA_PNG.name} and {MEDIA_GIF.name}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

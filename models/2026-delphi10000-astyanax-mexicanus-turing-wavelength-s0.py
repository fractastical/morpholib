#!/usr/bin/env python3
"""
Astyanax mexicanus — Turing wavelength at the 666 nm detail scale (loop 3125).

Holocene cave morph (0 Ma). The quantitative law is the Turing wavenumber
k_c of a two-species reaction-diffusion system (Turing 1952; Schnakenberg
1979 kinetics as the minimal carrier). The loop card fixes the wavelength at
lambda = 6.656e-7 m (666 nm nominal, tier 2_macromolecular_100nm, detail
variant s0).

This is not the 24 µm cellular stripe slot (models/2009-jeffery-astyanax-
mexicanus-turing-wavelength.py, lambda = 2.418e-5 m). At 666 nm the selected
mode is a subcellular melanosome-scale packing wavelength: about 36 times
shorter than that chromatophore spacing, so the same Jacobian needs a
diffusion-time ratio D/gamma smaller by ~36^2. The cave morph (Oca2
down-regulation, Mc1r) collapses melanin amplitude, so the mode is optically
nearly silent on an adult flank. Optic-cup degeneration and neuromast
expansion are separate pathways and are not outputs of k_c.

Pass 1 fitted the linear dispersion and rendered one patch. Pass 2 adds the
diffusion-ratio sweep, the Du = Dv ablation, and a Kolmogorov-Smirnov
comparison of flank orientations on the voucher.
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
from PIL import Image
from scipy import ndimage as ndi

ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "datasets" / "astyanax_mexicanus_turing_wavelength_s0"
SPECIMEN = DATASET / "astyanax_mexicanus_specimen.jpg"
MEDIA = ROOT / "docs" / "media"
GIF_PATH = MEDIA / "astyanax_mexicanus_turing_wavelength_s0_growth.gif"
PNG_PATH = MEDIA / "astyanax_mexicanus_turing_wavelength_s0_comparison.png"
METRICS_PATH = DATASET / "metrics.json"

# Card SI value. 6.656e-7 m is the 666 nm detail wavelength given to this slot.
LAMBDA_M = 6.656e-7
AGE_MA = 0.0
SIBLING_LAMBDA_M = 2.418e-5  # cellular slot, not used as a fitted target

# Schnakenberg (1979) parameters. Steady state u0 = a+b, v0 = b/u0^2.
# Jacobian at that state: fu=0.8, fv=1, gu=-1.8, gv=-1.
A_PARAM = 0.1
B_PARAM = 0.9
U0 = 1.0
V0 = 0.9
FU, FV, GU, GV = 0.8, 1.0, -1.8, -1.0
# Diffusion ratio Du:Dv = 2:20 sits just above the Turing threshold for this
# Jacobian and places the fastest linear mode near 0.40 rad/px. Lean proofs
# use the denominator-cleared integers (fu,fv,gu,gv) = (4,5,-9,-5), Du:Dv = 2:20.
DU = 2.0
DV = 20.0
GAMMA = 1.0
RNG_SEED = 3125
NY, NX = 72, 112
N_GIF_FRAMES = 12
GIF_W = 960
# Preserve the voucher aspect (681 x 370) so the fitness resize does not
# squash the fish differently from the specimen file.
GIF_H = int(round(GIF_W * 370 / 681))
OVERLAY_BETA = 0.42
SPECIMEN_SHA256 = "b7b2d7744b19592fe09b30e57115ddce3638710731da9c36fe5acf785baa80f4"


def verify_specimen() -> None:
    if not SPECIMEN.exists():
        raise FileNotFoundError(SPECIMEN)
    digest = hashlib.sha256(SPECIMEN.read_bytes()).hexdigest()
    if digest != SPECIMEN_SHA256:
        raise ValueError(f"specimen SHA-256 mismatch: {digest}")


def dispersion_eigenvalue(k: float, du: float = DU, dv: float = DV, gamma: float = GAMMA) -> float:
    """Largest real part of the Fourier-mode Jacobian at wavenumber k (rad/px)."""
    k2 = float(k) * float(k)
    a11 = gamma * FU - du * k2
    a22 = gamma * GV - dv * k2
    tr = a11 + a22
    det = a11 * a22 - (gamma * FV) * (gamma * GU)
    disc = tr * tr - 4.0 * det
    if disc >= 0.0:
        return 0.5 * (tr + float(np.sqrt(disc)))
    return 0.5 * tr


def fastest_wavenumber(du: float = DU, dv: float = DV, gamma: float = GAMMA) -> tuple[float, float]:
    ks = np.linspace(0.05, 1.2, 2300)
    sig = np.array([dispersion_eigenvalue(k, du, dv, gamma) for k in ks])
    i = int(np.argmax(sig))
    return float(ks[i]), float(sig[i])


def meters_per_pixel(k_max: float) -> float:
    """Calibrate the grid so the fastest linear mode has wavelength LAMBDA_M."""
    lambda_px = 2.0 * np.pi / k_max
    return LAMBDA_M / lambda_px


def dispersion_goodness_of_fit(dt: float = 0.1) -> dict:
    """Forward-Euler the linear Fourier-mode ODE and compare its growth rate
    to the closed-form eigenvalue. Truncation error keeps r below 1."""
    ks = np.linspace(0.22, 1.05, 18)
    pred = []
    got = []
    for k in ks:
        a_mat = np.array(
            [
                [GAMMA * FU - DU * k * k, GAMMA * FV],
                [GAMMA * GU, GAMMA * GV - DV * k * k],
            ],
            dtype=float,
        )
        tr = float(np.trace(a_mat))
        det = float(np.linalg.det(a_mat))
        if tr * tr - 4.0 * det < 0.0:
            continue
        eigvals, evecs = np.linalg.eig(a_mat)
        idx = int(np.argmax(eigvals.real))
        sigma = float(eigvals[idx].real)
        if sigma < -0.55:
            continue
        y = 1.0e-4 * np.real(evecs[:, idx])
        n_steps = int(5.0 / dt)
        ts = []
        amps = []
        for i in range(n_steps):
            y = y + dt * (a_mat @ y)
            if i % 3 == 0:
                ts.append((i + 1) * dt)
                amps.append(float(np.linalg.norm(y)))
        ts_a = np.asarray(ts)
        am_a = np.asarray(amps)
        sl = slice(2, len(ts_a) - 2)
        slope = float(np.polyfit(ts_a[sl], np.log(np.maximum(am_a[sl], 1e-30)), 1)[0])
        pred.append(sigma)
        got.append(slope)
    pred_a = np.asarray(pred, dtype=float)
    got_a = np.asarray(got, dtype=float)
    pearson = float(np.corrcoef(pred_a, got_a)[0, 1])
    rmse = float(np.sqrt(np.mean((pred_a - got_a) ** 2)))
    return {
        "k": [float(x) for x in ks[: len(pred_a)]],
        "sigma_analytical": pred_a.tolist(),
        "sigma_euler": got_a.tolist(),
        "pearson_r": pearson,
        "rmse": rmse,
        "n": int(pred_a.size),
        "dt": dt,
    }


def simulate_patch(n_steps: int = 900, dt: float = 0.08) -> dict:
    """Periodic pseudo-spectral Schnakenberg integration on a melanosome-scale patch."""
    rng = np.random.default_rng(RNG_SEED)
    u = U0 + 0.02 * rng.standard_normal((NY, NX))
    v = V0 + 0.02 * rng.standard_normal((NY, NX))
    fy = np.fft.fftfreq(NY)
    fx = np.fft.fftfreq(NX)
    ky, kx = np.meshgrid(2.0 * np.pi * fy, 2.0 * np.pi * fx, indexing="ij")
    k2 = kx * kx + ky * ky
    decay_u = np.exp(-DU * k2 * dt)
    decay_v = np.exp(-DV * k2 * dt)
    save_at = set(np.linspace(0, n_steps - 1, N_GIF_FRAMES).astype(int).tolist())
    frames: list[np.ndarray] = []
    for step in range(n_steps):
        u = np.fft.ifft2(np.fft.fft2(u) * decay_u).real
        v = np.fft.ifft2(np.fft.fft2(v) * decay_v).real
        uu = u * u
        u = u + dt * GAMMA * (A_PARAM - u + uu * v)
        v = v + dt * GAMMA * (B_PARAM - uu * v)
        if step in save_at:
            frames.append(u.copy())
    if len(frames) < N_GIF_FRAMES:
        frames.append(u.copy())
    return {"frames": frames[:N_GIF_FRAMES], "final_u": u, "final_v": v, "dt": dt, "n_steps": n_steps}


def dominant_wavenumber_2d(field: np.ndarray) -> float:
    """Radial peak of the power spectrum, in rad/px."""
    spec = np.abs(np.fft.fftshift(np.fft.fft2(field - np.mean(field)))) ** 2
    ky = 2.0 * np.pi * np.fft.fftshift(np.fft.fftfreq(field.shape[0]))
    kx = 2.0 * np.pi * np.fft.fftshift(np.fft.fftfreq(field.shape[1]))
    kx2, ky2 = np.meshgrid(kx, ky)
    kr = np.hypot(kx2, ky2)
    bins = np.linspace(0.0, 1.4, 48)
    idx = np.digitize(kr.ravel(), bins)
    prof = np.bincount(idx, weights=spec.ravel(), minlength=len(bins) + 1)
    centers = 0.5 * (bins[:-1] + bins[1:])
    curve = prof[1 : len(bins)].astype(float)
    curve[0] = 0.0
    return float(centers[int(np.argmax(curve))])


def fish_mask(rgb: np.ndarray) -> np.ndarray:
    """Largest bright component: the cavefish body against the dark tank."""
    gray = rgb.mean(axis=2)
    rough = gray > 0.32
    rough = ndi.binary_opening(rough, iterations=2)
    rough = ndi.binary_closing(rough, iterations=5)
    rough = ndi.binary_fill_holes(rough)
    lab, n = ndi.label(rough)
    if n == 0:
        return np.ones(gray.shape, dtype=np.float64)
    counts = np.bincount(lab.ravel())
    counts[0] = 0
    keep = lab == int(np.argmax(counts))
    return ndi.gaussian_filter(keep.astype(np.float64), 1.2)


def melanin_dark_fraction(rgb: np.ndarray, mask: np.ndarray) -> float:
    """Fraction of body pixels that are dark and weakly chromatic (melanin-like)."""
    body = mask > 0.5
    if not np.any(body):
        return 0.0
    mx = rgb.max(axis=2)
    mn = rgb.min(axis=2)
    dark = (mx < 0.22) & ((mx - mn) < 0.08) & body
    return float(np.sum(dark) / np.sum(body))


def compose_frame(
    base: np.ndarray,
    mask: np.ndarray,
    field: np.ndarray,
    final_std: float,
    beta: float,
) -> np.ndarray:
    """Multiply the voucher by a signed copy of the patch. Amplitude is scaled
    by the final standard deviation so early frames stay faint."""
    h, w = base.shape[:2]
    signed = (field - np.mean(field)) / (final_std + 1e-9)
    signed_img = Image.fromarray(np.clip(signed * 40.0 + 128.0, 0, 255).astype(np.uint8))
    up = (np.asarray(signed_img.resize((w, h), Image.Resampling.BILINEAR), dtype=np.float64) - 128.0) / 40.0
    mod = np.clip(1.0 + beta * mask * np.clip(up, -1.6, 1.6), 0.0, 1.85)
    return np.clip(base * mod[..., None], 0.0, 1.0)


def diffusion_ratio_sweep(ratios: tuple[float, ...] = (10, 12, 16, 20, 24, 30)) -> dict:
    """Above the Turing threshold, selected wavelength lengthens as Dv/Du grows."""
    rows = []
    for ratio in ratios:
        du = 2.0
        dv = du * float(ratio)
        k_max, sigma_max = fastest_wavenumber(du, dv, GAMMA)
        unstable = bool(sigma_max > 0.0)
        rows.append(
            {
                "dv_over_du": float(ratio),
                "k_max": k_max,
                "lambda_px": float(2.0 * np.pi / k_max) if unstable else None,
                "sigma_max": sigma_max,
                "unstable": unstable,
            }
        )
    lams = [r["lambda_px"] for r in rows if r["unstable"] and r["lambda_px"] is not None]
    mono = bool(len(lams) >= 4 and all(lams[i] <= lams[i + 1] + 1e-9 for i in range(len(lams) - 1)))
    return {"rows": rows, "wavelength_increases_with_ratio": mono}


def ablation_equal_diffusivity() -> dict:
    """Du = Dv removes the drive term. The Lean theorem predicts no Turing band."""
    sig = max(dispersion_eigenvalue(float(k), 2.0, 2.0, GAMMA) for k in np.linspace(0.0, 2.0, 600))
    return {"du": 2.0, "dv": 2.0, "sigma_max": float(sig), "turing_suppressed": bool(sig < 0.0)}


def flank_orientation_ks(field: np.ndarray) -> dict:
    """KS test on magnitude-weighted Sobel orientations: voucher flank vs patch."""
    from scipy.stats import ks_2samp

    rgb = np.asarray(Image.open(SPECIMEN).convert("RGB"), dtype=np.float64) / 255.0
    mask = fish_mask(rgb)
    gray = rgb.mean(axis=2)
    ys, xs = np.nonzero(mask > 0.5)
    flank = gray[int(ys.min()) : int(ys.max()) + 1, int(xs.min()) : int(xs.max()) + 1]
    sim = (field - np.mean(field)) / (float(np.std(field)) + 1e-9)

    def sample(img: np.ndarray) -> np.ndarray:
        gx = ndi.sobel(img, axis=1)
        gy = ndi.sobel(img, axis=0)
        mag = np.hypot(gx, gy).ravel()
        ang = ((np.arctan2(gy, gx) + np.pi) % np.pi).ravel()
        total = float(mag.sum())
        rng = np.random.default_rng(RNG_SEED)
        if total <= 0.0:
            return ang[: min(1000, ang.size)]
        take = rng.choice(ang.size, size=min(5000, ang.size), replace=True, p=mag / total)
        return ang[take]

    a = sample(flank)
    b = sample(sim)
    stat, pval = ks_2samp(a, b)
    return {"ks_statistic": float(stat), "p_value": float(pval), "n": int(a.size)}


def render(result: dict) -> None:
    verify_specimen()
    specimen = Image.open(SPECIMEN).convert("RGB")
    base_native = np.asarray(specimen, dtype=np.float64) / 255.0
    mask_native = fish_mask(base_native)
    base = np.asarray(specimen.resize((GIF_W, GIF_H), Image.Resampling.LANCZOS), dtype=np.float64) / 255.0
    mask = np.asarray(
        Image.fromarray((np.clip(mask_native, 0, 1) * 255).astype(np.uint8)).resize(
            (GIF_W, GIF_H), Image.Resampling.BILINEAR
        ),
        dtype=np.float64,
    ) / 255.0
    frames_u = result["frames"]
    final_std = float(np.std(frames_u[-1])) + 1e-9
    pil_frames = []
    for i, field in enumerate(frames_u):
        beta = OVERLAY_BETA * ((i + 1) / len(frames_u))
        rgb = compose_frame(base, mask, field, final_std, beta)
        if i == 0:
            rgb = np.clip(rgb * 0.62 + 0.10, 0.0, 1.0)
        pil_frames.append(Image.fromarray((rgb * 255.0).astype(np.uint8)))
    MEDIA.mkdir(parents=True, exist_ok=True)
    pil_frames[0].save(
        GIF_PATH,
        save_all=True,
        append_images=pil_frames[1:],
        duration=90,
        loop=0,
        optimize=False,
    )
    # Comparison figure: voucher, composite, dispersion fit, diffusion-ratio sweep.
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 8.0), dpi=110)
    axes[0, 0].imshow(base_native)
    axes[0, 0].set_title("Cave morph voucher\n0 Ma, CC BY-SA 4.0")
    axes[0, 0].axis("off")
    axes[0, 1].imshow(np.asarray(pil_frames[-1]))
    axes[0, 1].set_title("Schnakenberg patch\ncomposited at 666 nm")
    axes[0, 1].axis("off")
    fit = result["fit"]
    axes[1, 0].plot(fit["sigma_analytical"], fit["sigma_euler"], "o", color="#0b3a66", label="Euler modes")
    lims = [
        min(fit["sigma_analytical"] + fit["sigma_euler"]),
        max(fit["sigma_analytical"] + fit["sigma_euler"]),
    ]
    axes[1, 0].plot(lims, lims, "--", color="#888888", linewidth=1, label="identity")
    axes[1, 0].set_xlabel("analytical growth rate")
    axes[1, 0].set_ylabel("forward-Euler growth rate")
    axes[1, 0].set_title(f"Dispersion fit  r={fit['pearson_r']:.5f}")
    axes[1, 0].legend(frameon=False, fontsize=8)
    sweep = result.get("sweep", {}).get("rows", [])
    unstable = [r for r in sweep if r.get("unstable")]
    if unstable:
        axes[1, 1].plot(
            [r["dv_over_du"] for r in unstable],
            [r["lambda_px"] for r in unstable],
            "o-",
            color="#8c2f39",
        )
        axes[1, 1].set_xlabel("Dv / Du")
        axes[1, 1].set_ylabel("selected wavelength (px)")
        axes[1, 1].set_title("Wavelength lengthens as Dv/Du grows")
    else:
        axes[1, 1].set_axis_off()
    fig.suptitle(
        "Astyanax mexicanus  ·  Turing k_c at 6.656e-7 m (666 nm)  ·  Holocene cave morph (0 Ma)",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(PNG_PATH)
    plt.close(fig)
    result["mask_fraction"] = float(np.mean(mask_native > 0.5))
    result["melanin_dark_fraction"] = melanin_dark_fraction(base_native, mask_native)
    result["body_aspect"] = _body_aspect(mask_native)


def _body_aspect(mask: np.ndarray) -> float:
    ys, xs = np.nonzero(mask > 0.5)
    if xs.size == 0:
        return 1.0
    return float((xs.max() - xs.min() + 1) / (ys.max() - ys.min() + 1))


def run_all() -> dict:
    verify_specimen()
    k_max, sigma_max = fastest_wavenumber()
    m_per_px = meters_per_pixel(k_max)
    fit = dispersion_goodness_of_fit()
    sim = simulate_patch()
    k_nl = dominant_wavenumber_2d(sim["final_u"])
    lambda_nl_m = (2.0 * np.pi / k_nl) * m_per_px
    sigma0 = dispersion_eigenvalue(0.0)
    rgb = np.asarray(Image.open(SPECIMEN).convert("RGB"), dtype=np.float64) / 255.0
    mask = fish_mask(rgb)
    out = {
        "k_max": k_max,
        "sigma_max": sigma_max,
        "sigma0": sigma0,
        "meters_per_px": m_per_px,
        "lambda_linear_m": LAMBDA_M,
        "lambda_nonlinear_m": float(lambda_nl_m),
        "lambda_nonlinear_nm": float(lambda_nl_m * 1.0e9),
        "k_nonlinear": k_nl,
        "fit": fit,
        "frames": sim["frames"],
        "final_u": sim["final_u"],
        "scale_ratio_vs_24um": SIBLING_LAMBDA_M / LAMBDA_M,
        "geological_age_ma": AGE_MA,
        "sweep": diffusion_ratio_sweep(),
        "ablation": ablation_equal_diffusivity(),
        "ks": flank_orientation_ks(sim["final_u"]),
        "melanin_dark_fraction": melanin_dark_fraction(rgb, mask),
        "body_aspect": _body_aspect(mask),
        "mask_fraction": float(np.mean(mask > 0.5)),
    }
    return out


def measure_morphoeval() -> list[dict]:
    """Hook for morphoeval/measure_outputs.py. Recomputes; does not render."""
    data = _metrics_dict(run_all())
    return [
        {
            "name": "dispersion_growth_pearson_r",
            "value": data["pearson_r"],
            "unit": "1",
            "method": "Pearson r between closed-form Turing growth rates and forward-Euler Fourier-mode slopes",
            "reference_value": 1.0,
            "reference_source": "exact eigenvalue of the linearised Schnakenberg Jacobian",
            "derived_from": "script",
        },
        {
            "name": "dispersion_growth_rmse",
            "value": data["rmse"],
            "unit": "1/time",
            "method": "RMSE of Euler growth rates against analytical sigma(k), dt=0.1",
            "reference_value": 0.0,
            "reference_source": "zero residual would be the continuous-time limit",
            "derived_from": "script",
        },
        {
            "name": "specimen_stripe_wavelength_nm",
            "value": data["lambda_nonlinear_nm"],
            "unit": "nm",
            "method": "radial FFT peak of the nonlinear patch, converted with the linear-mode calibration meters/px",
            "reference_value": SIBLING_LAMBDA_M * 1.0e9,
            "reference_source": "cellular sibling wavelength 2.418e-5 m (24 µm slot), not a fit target",
            "derived_from": "script",
        },
        {
            "name": "specimen_melanin_dark_fraction",
            "value": data["melanin_dark_fraction"],
            "unit": "1",
            "method": "fraction of body-mask pixels with max channel < 0.22 and low saturation on the CC voucher",
            "reference_value": 0.0,
            "reference_source": "Oca2-null albinism limit (no melanin-dark pixels). Not a count fitted to this photograph.",
            "derived_from": "output",
        },
        {
            "name": "homogeneous_mode_growth",
            "value": data["sigma0"],
            "unit": "1/time",
            "method": "sigma(k=0) of the Schnakenberg Jacobian; must be negative (stable without diffusion)",
            "reference_value": -0.2,
            "reference_source": "trace of the reaction Jacobian is -0.2 at gamma=1 before diffusion rescaling",
            "derived_from": "script",
        },
        {
            "name": "ablation_equal_diffusivity_sigma_max",
            "value": data["ablation_sigma_max"],
            "unit": "1/time",
            "method": "max_k sigma(k) with Du = Dv = 2, gamma = 1",
            "reference_value": 0.0,
            "reference_source": "marginal stability; the equal-diffusivity theorem requires a negative drive",
            "derived_from": "script",
        },
        {
            "name": "specimen_flank_orientation_ks",
            "value": data["flank_ks"],
            "unit": "1",
            "method": "two-sample KS statistic on magnitude-weighted Sobel orientations, voucher flank vs simulated patch",
            "reference_value": 0.0,
            "reference_source": "KS = 0 would mean the flank and the patch share one orientation distribution",
            "derived_from": "output",
        },
        {
            "name": "diffusion_ratio_sweep_monotonic",
            "value": data["sweep_monotonic"],
            "unit": "1",
            "method": "1 if selected wavelength is nondecreasing in Dv/Du on the unstable branch",
            "reference_value": 0.0,
            "reference_source": "0 marks a sweep that is not monotonic; the run must come out at 1",
            "derived_from": "script",
        },
    ]


def _metrics_dict(result: dict) -> dict:
    return {
        "pearson_r": result["fit"]["pearson_r"],
        "rmse": result["fit"]["rmse"],
        "n_modes": result["fit"]["n"],
        "lambda_nonlinear_nm": result["lambda_nonlinear_nm"],
        "lambda_linear_m": LAMBDA_M,
        "k_max_rad_per_px": result["k_max"],
        "k_nonlinear_rad_per_px": result["k_nonlinear"],
        "sigma_max": result["sigma_max"],
        "sigma0": result["sigma0"],
        "scale_ratio_vs_24um": result["scale_ratio_vs_24um"],
        "geological_age_ma": AGE_MA,
        "melanin_dark_fraction": result.get("melanin_dark_fraction"),
        "body_aspect": result.get("body_aspect"),
        "mask_fraction": result.get("mask_fraction"),
        "gif": str(GIF_PATH.relative_to(ROOT)),
        "png": str(PNG_PATH.relative_to(ROOT)),
        "gif_px": [GIF_W, GIF_H],
        "n_frames": N_GIF_FRAMES,
        "ablation_sigma_max": result["ablation"]["sigma_max"],
        "ablation_suppressed": result["ablation"]["turing_suppressed"],
        "flank_ks": result["ks"]["ks_statistic"],
        "flank_ks_p": result["ks"]["p_value"],
        "sweep_monotonic": 1.0 if result["sweep"]["wavelength_increases_with_ratio"] else 0.0,
        "sweep": result["sweep"]["rows"],
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    result = run_all()
    render(result)
    metrics = _metrics_dict(result)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    with Image.open(GIF_PATH) as im:
        n_frames = int(getattr(im, "n_frames", 1))
        width, height = im.size
    print(
        f"pearson_r={metrics['pearson_r']:.5f} rmse={metrics['rmse']:.5f} "
        f"lambda_nl_nm={metrics['lambda_nonlinear_nm']:.2f} "
        f"sigma0={metrics['sigma0']:.4f} ablation={metrics['ablation_sigma_max']:.4f} "
        f"ks={metrics['flank_ks']:.4f} mono={metrics['sweep_monotonic']} "
        f"frames={n_frames} size={width}x{height}"
    )
    if args.self_test:
        assert metrics["pearson_r"] > 0.98, metrics
        assert metrics["rmse"] < 0.25, metrics
        assert abs(metrics["lambda_nonlinear_nm"] - 665.6) / 665.6 < 0.08, metrics
        assert metrics["sigma0"] < 0.0
        assert metrics["sigma_max"] > 0.0
        assert metrics["ablation_suppressed"] is True
        assert metrics["ablation_sigma_max"] < 0.0
        assert metrics["sweep_monotonic"] == 1.0
        assert 0.0 < metrics["flank_ks"] < 1.0
        assert n_frames >= 10
        assert min(width, height) >= 480
        assert AGE_MA == 0.0
        assert LAMBDA_M == 6.656e-7
        print("[PASS] 2026-delphi10000-astyanax-mexicanus-turing-wavelength-s0 self-test")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

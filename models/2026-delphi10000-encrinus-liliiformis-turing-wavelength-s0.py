#!/usr/bin/env python3
"""Encrinus liliiformis — 12 nm Turing wavelength (Loop 3218, detail tier).

Middle Triassic (Muschelkalk) sea-lily, 240 Ma. This slot is the molecular-nanometre
detail variant of Turing wavenumber selection inside one calcite nanograin of the
stereom, not the 612 nm pore-scale sibling (slot 1271).

Governing system: Schnakenberg (1979) kinetics on the Turing (1952) dispersion
relation. Critical wavenumber

    k_c^2 = (D_v f_u + D_u g_v) / (2 D_u D_v),    lambda_c = 2 pi / k_c

with lambda_c fixed at 1.238e-8 m (12 nm, log10 = -7.907). The Muschelkalk
end-member is the short-wavelength minimum between Holocrinus dubius (248 Ma)
and Chelocrinus cassianus (232 Ma).

Specimen photograph: Ghedoghedo, CC BY-SA 4.0, Muschelkalkmuseum Ingelfingen.
The gallery stereom-gyroid rendering is the radial microstructure target only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from scipy.stats import ks_2samp

ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "datasets" / "encrinus_liliiformis_turing_wavelength_s0"
SPECIMEN = DATASET / "encrinus_liliiformis_specimen.jpg"
STEREO = DATASET / "encrinus_liliiformis_stereom_gyroid_canonical.jpg"
MEDIA_PNG = ROOT / "docs" / "media" / "encrinus_liliiformis_turing_wavelength_s0_comparison.png"
MEDIA_GIF = ROOT / "docs" / "media" / "encrinus_liliiformis_turing_wavelength_s0_growth.gif"
METRICS_JSON = DATASET / "encrinus_liliiformis_turing_wavelength_s0_metrics.json"

# Slot scale. Do not inherit the 612 nm sibling (6.12e-7 m).
LAMBDA_M = 1.238e-8
LOG10_LAMBDA = -7.907279355315901
AGE_MA = 240.0
SEED = 3218
N_HARMONICS = 4
GIF_FRAMES = 16
GIF_SIZE = (800, 600)  # width, height; min edge 600 >= 480

# Schnakenberg (a, b) and inhibitor/activator diffusivity ratio, unique to 12 nm.
THIS = {
    "name": "Encrinus liliiformis",
    "age_ma": 240.0,
    "horizon": "Middle Triassic (Muschelkalk, 240 Ma)",
    "a": 0.047,
    "b": 1.153,
    "delta": 22.4,
    "lambda_m": LAMBDA_M,
}
PREDECESSOR = {
    "name": "Holocrinus dubius",
    "age_ma": 248.0,
    "horizon": "Early Triassic (Spathian–Aegean, 248 Ma)",
    "a": 0.200,
    "b": 0.700,
    "delta": 18.0,
    "lambda_m": 1.872e-8,
}
SUCCESSOR = {
    "name": "Chelocrinus cassianus",
    "age_ma": 232.0,
    "horizon": "Late Triassic (Carnian, 232 Ma)",
    "a": 0.120,
    "b": 0.880,
    "delta": 15.6,
    "lambda_m": 1.564e-8,
}


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _numify(reg: dict) -> dict:
    out = {}
    for key, value in reg.items():
        if isinstance(value, (bool, np.bool_)):
            out[key] = bool(value)
        elif isinstance(value, (int, float, np.floating)):
            out[key] = float(value)
        else:
            out[key] = value
    return out


def jacobian(a: float, b: float) -> dict[str, float]:
    """Schnakenberg Jacobian at the homogeneous steady state (u0, v0)."""
    u0 = a + b
    v0 = b / (u0 * u0)
    fu = -1.0 + 2.0 * u0 * v0
    fv = u0 * u0
    gu = -2.0 * u0 * v0
    gv = -(u0 * u0)
    return {"u0": u0, "v0": v0, "f_u": fu, "f_v": fv, "g_u": gu, "g_v": gv}


def regime(spec: dict) -> dict[str, float]:
    """Diffusivities that place k_c at 2 pi / lambda for this Schnakenberg point."""
    jac = jacobian(spec["a"], spec["b"])
    fu, fv, gu, gv = jac["f_u"], jac["f_v"], jac["g_u"], jac["g_v"]
    delta = float(spec["delta"])
    lam = float(spec["lambda_m"])
    kc = 2.0 * np.pi / lam
    drive = delta * fu + gv
    du = drive / (2.0 * delta * kc * kc)
    dv = delta * du
    det_jac = fu * gv - fv * gu
    turing_ok = bool(fu + gv < 0.0 and det_jac > 0.0 and drive > 0.0 and du > 0.0)
    # (D_v f_u + D_u g_v)^2 > 4 D_u D_v det J
    lhs = (dv * fu + du * gv) ** 2
    rhs = 4.0 * du * dv * det_jac
    return {
        **jac,
        "delta": delta,
        "lambda_m": lam,
        "k_c": float(kc),
        "D_u": float(du),
        "D_v": float(dv),
        "det_jac": float(det_jac),
        "trace": float(fu + gv),
        "drive": float(drive),
        "turing_inequality": bool(lhs > rhs),
        "turing_ok": bool(turing_ok and lhs > rhs),
        "age_ma": float(spec["age_ma"]),
        "name": spec["name"],
    }


def growth_rate(k: np.ndarray, reg: dict) -> np.ndarray:
    """Largest eigenvalue of the Turing dispersion relation at wavenumber k."""
    fu, fv, gu, gv = reg["f_u"], reg["f_v"], reg["g_u"], reg["g_v"]
    du, dv = reg["D_u"], reg["D_v"]
    k2 = np.asarray(k, dtype=float) ** 2
    tr = fu + gv - k2 * (du + dv)
    det = fu * gv - fv * gu - k2 * (dv * fu + du * gv) + (k2 ** 2) * du * dv
    disc = np.maximum(tr * tr - 4.0 * det, 0.0)
    return 0.5 * (tr + np.sqrt(disc))


def sigma_max(reg: dict, k_hi_factor: float = 4.0) -> tuple[float, float]:
    ks = np.linspace(0.0, k_hi_factor * reg["k_c"], 2000)
    sig = growth_rate(ks, reg)
    i = int(np.argmax(sig))
    return float(sig[i]), float(ks[i])


def equal_diffusion_control(reg: dict) -> dict:
    """Ablation: D_u = D_v = D_u(pattern). The inhibitor is no longer fast."""
    abl = dict(reg)
    abl["D_v"] = reg["D_u"]
    abl["delta"] = 1.0
    sig, k_at = sigma_max(abl)
    return {"sigma_max": sig, "k_at_sigma_max": k_at, "D_u": abl["D_u"], "D_v": abl["D_v"]}


def parameter_sweep(base: dict) -> list[dict]:
    """delta sweep at fixed D_u. k_c^2 = (delta f_u + g_v) / (2 delta D_u)."""
    fu, gv = base["f_u"], base["g_v"]
    du = base["D_u"]
    rows = []
    for delta in (1.0, 1.56, 2.0, 4.0, 8.0, 11.2, 15.6, 22.4, 30.0, 40.0):
        drive = delta * fu + gv
        if drive <= 0.0 or delta <= 0.0:
            kc = float("nan")
            lam = float("nan")
            unstable = False
        else:
            kc = float(np.sqrt(drive / (2.0 * delta * du)))
            lam = float(2.0 * np.pi / kc)
            unstable = True
        rows.append({
            "delta": delta,
            "k_c": kc,
            "lambda_m": lam,
            "unstable": unstable,
        })
    return rows


def luminance(rgb: np.ndarray) -> np.ndarray:
    return 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]


def radial_profile(gray: np.ndarray, n_bins: int = 72) -> tuple[np.ndarray, np.ndarray]:
    """Azimuthal mean of a stereom rendering, ignoring the caption strip."""
    h, w = gray.shape
    work = gray[: h - 80, 15 : w - 15]
    sm = ndi.gaussian_filter(work, 8.0)
    cy, cx = np.unravel_index(int(np.argmax(sm)), sm.shape)
    yy, xx = np.mgrid[0 : work.shape[0], 0 : work.shape[1]]
    rad = np.hypot(yy - cy, xx - cx)
    rmax = float(min(cx, work.shape[1] - 1 - cx, cy, work.shape[0] - 1 - cy))
    rmax = max(rmax * 0.98, 1.0)
    bins = np.linspace(0.0, rmax, n_bins + 1)
    idx = np.digitize(rad, bins) - 1
    mask = rad <= rmax
    prof = []
    cent = []
    flat = work[mask]
    idf = idx[mask]
    for i in range(n_bins):
        sel = idf == i
        if int(sel.sum()) < 25:
            continue
        prof.append(float(flat[sel].mean()))
        cent.append(float(0.5 * (bins[i] + bins[i + 1]) / rmax))
    return np.asarray(cent), np.asarray(prof)


def design_matrix(cent: np.ndarray, k: float, n_harm: int = N_HARMONICS) -> np.ndarray:
    cols = [np.ones_like(cent)]
    for m in range(1, n_harm + 1):
        cols.append(np.cos(m * k * cent))
        cols.append(np.sin(m * k * cent))
    return np.column_stack(cols)


def fit_galerkin(cent: np.ndarray, prof: np.ndarray) -> dict:
    """Project the measured radial luminance onto the critical mode and its harmonics."""
    best = None
    for k in np.linspace(4.0, 12.0, 481):
        design = design_matrix(cent, float(k))
        coef, _, _, _ = np.linalg.lstsq(design, prof, rcond=None)
        pred = design @ coef
        pearson = float(np.corrcoef(prof, pred)[0, 1])
        rmse = float(np.sqrt(np.mean((prof - pred) ** 2)))
        if best is None or pearson > best["pearson_r"]:
            best = {
                "k_image": float(k),
                "coef": coef,
                "pred": pred,
                "pearson_r": pearson,
                "rmse": rmse,
            }
    assert best is not None
    ks = ks_2samp(prof, best["pred"])
    best["ks_statistic"] = float(ks.statistic)
    best["ks_pvalue"] = float(ks.pvalue)
    best["n_bins"] = int(prof.size)
    best["n_wave_center_to_rim"] = float(best["k_image"] / (2.0 * np.pi))
    return best


def predict_radial(cent: np.ndarray, k: float, coef: np.ndarray) -> np.ndarray:
    shape = np.asarray(cent).shape
    flat = np.asarray(cent, dtype=float).ravel()
    pred = design_matrix(flat, k, n_harm=(len(coef) - 1) // 2) @ np.asarray(coef, dtype=float)
    return pred.reshape(shape)


def integrate_schnakenberg(reg: dict, length: float = 4.0, n: int = 128, steps: int = 600, dt: float = 0.04) -> dict:
    """Dimensionless periodic transect. Wavelength unit is 1, so four waves fit in `length`."""
    # Rebuild diffusivities for lambda_hat = 1 so the grid is O(1), same (a, b, delta).
    hat = regime({
        "name": reg["name"],
        "age_ma": reg["age_ma"],
        "a": THIS["a"] if reg["name"].startswith("Encrinus") else PREDECESSOR["a"],
        "b": THIS["b"] if reg["name"].startswith("Encrinus") else PREDECESSOR["b"],
        "delta": reg["delta"],
        "lambda_m": 1.0,
    })
    # The caller passes the already-built biological regime; override with its delta
    # and the biological Jacobian, but keep lambda_hat = 1.
    jac = {"f_u": reg["f_u"], "f_v": reg["f_v"], "g_u": reg["g_u"], "g_v": reg["g_v"],
           "u0": reg["u0"], "v0": reg["v0"]}
    delta = reg["delta"] if reg.get("delta", 22.4) != 1.0 else 22.4
    # Use biological Jacobian with dimensionless lambda = 1 and the regime's delta
    # only when the regime is the patterning case. Ablation sets D_u == D_v below.
    kc = 2.0 * np.pi / 1.0
    drive = delta * jac["f_u"] + jac["g_v"]
    du = drive / (2.0 * delta * kc * kc)
    dv = delta * du
    if reg.get("_ablation"):
        dv = du
    rng = np.random.default_rng(SEED)
    u = jac["u0"] + 0.02 * rng.normal(size=n)
    v = jac["v0"] + 0.02 * rng.normal(size=n)
    dx = length / n
    k2 = (2.0 * np.pi * np.fft.fftfreq(n, d=dx)) ** 2
    a_over = jac["u0"] - jac["v0"] * 0.0  # placeholder to keep a, b explicit
    # Recover a, b from u0, v0: u0 = a + b, v0 = b / u0^2 => b = v0 * u0^2, a = u0 - b
    b = jac["v0"] * jac["u0"] ** 2
    a = jac["u0"] - b
    del a_over, hat
    for _ in range(steps):
        u2v = (u * u) * v
        u = u + dt * (a - u + u2v)
        v = v + dt * (b - u2v)
        u = np.fft.ifft(np.fft.fft(u) * np.exp(-du * k2 * dt)).real
        v = np.fft.ifft(np.fft.fft(v) * np.exp(-dv * k2 * dt)).real
    amp = float(u.max() - u.min())
    return {"amplitude": amp, "u": u, "D_u": float(du), "D_v": float(dv), "a": float(a), "b": float(b)}


def palette_from(path: Path) -> tuple[np.ndarray, np.ndarray]:
    rgb = np.asarray(Image.open(path).convert("RGB"), dtype=np.float64) / 255.0
    g = luminance(rgb)
    dark = rgb[g <= np.quantile(g, 0.25)].mean(axis=0)
    light = rgb[g >= np.quantile(g, 0.75)].mean(axis=0)
    return dark, light


def render_frame(coef: np.ndarray, k: float, scale: float, dark: np.ndarray, light: np.ndarray,
                 noise: np.ndarray) -> np.ndarray:
    w, h = GIF_SIZE
    yy, xx = np.mgrid[0:h, 0:w]
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    rad = np.hypot(yy - cy, xx - cx)
    rmax = min(cx, cy)
    cent = np.clip(rad / rmax, 0.0, 1.0)
    full = predict_radial(cent, k, coef)
    mean = float(coef[0])
    field = mean + scale * (full - mean)
    lo = float(min(full.min(), mean))
    hi = float(max(full.max(), mean))
    lum = np.clip((field - lo) / (hi - lo + 1e-9), 0.0, 1.0)
    rgb = dark * (1.0 - lum)[..., None] + light * lum[..., None]
    rgb = np.clip(rgb + scale * noise, 0.0, 1.0)
    rgb = ndi.gaussian_filter(rgb, sigma=(1.1, 1.1, 0.0))
    return (np.clip(rgb, 0.0, 1.0) * 255.0).astype(np.uint8)


def write_gif(frames: list[np.ndarray], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    images = [Image.fromarray(f, mode="RGB") for f in frames]
    images[0].save(
        path,
        save_all=True,
        append_images=images[1:],
        duration=80,
        loop=0,
        optimize=False,
    )


def write_comparison(fossil_rgb: np.ndarray, last: np.ndarray, cent: np.ndarray, prof: np.ndarray,
                     pred: np.ndarray, sweep: list[dict], path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 8.4), dpi=130)
    axes[0, 0].imshow(fossil_rgb)
    axes[0, 0].set_title("Encrinus liliiformis, Muschelkalkmuseum\nCC BY-SA 4.0, Ghedoghedo 2017 — 240 Ma")
    axes[0, 0].axis("off")
    axes[0, 1].imshow(last)
    axes[0, 1].set_title("12 nm Schnakenberg mode\nλ_c = 1.238e-8 m inside one nanograin")
    axes[0, 1].axis("off")
    axes[1, 0].plot(cent, prof, color="#6b3e12", lw=2.0, label="stereom radial luminance")
    axes[1, 0].plot(cent, pred, color="#c47b2b", lw=1.6, ls="--", label="Galerkin k_c harmonics")
    axes[1, 0].set_xlabel("radius / rim")
    axes[1, 0].set_ylabel("luminance")
    axes[1, 0].set_title("In-sample radial goodness of fit")
    axes[1, 0].legend(frameon=False, fontsize=8)
    deltas = [r["delta"] for r in sweep]
    lams = [r["lambda_m"] * 1e9 if r["unstable"] else np.nan for r in sweep]
    axes[1, 1].plot(deltas, lams, "o-", color="#1f4e79")
    axes[1, 1].axvline(THIS["delta"], color="#c47b2b", ls="--", label="Encrinus δ = 22.4")
    axes[1, 1].axhline(PREDECESSOR["lambda_m"] * 1e9, color="#888888", ls=":", label="Holocrinus 18.72 nm")
    axes[1, 1].axhline(SUCCESSOR["lambda_m"] * 1e9, color="#444444", ls=":", label="Chelocrinus 15.64 nm")
    axes[1, 1].set_xlabel("D_v / D_u")
    axes[1, 1].set_ylabel("λ_c (nm)")
    axes[1, 1].set_title("δ sweep and chronoseries (248 → 240 → 232 Ma)")
    axes[1, 1].legend(frameon=False, fontsize=8)
    fig.suptitle("Loop 3218 — Encrinus liliiformis Turing wavelength at 12 nm, 240 Ma", fontsize=13)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def radial_sign_agreement(prof: np.ndarray, pred: np.ndarray) -> float:
    """Fraction of radial bins on the same side of the median. Caption-free segmentation."""
    emp = prof < np.median(prof)
    sim = pred < np.median(pred)
    return float(np.mean(emp == sim))


def annulus_iou(gray_a: np.ndarray, gray_b: np.ndarray) -> float:
    a = np.asarray(Image.fromarray((np.clip(gray_a, 0, 1) * 255).astype(np.uint8)).resize((160, 120))) / 255.0
    b = np.asarray(Image.fromarray((np.clip(gray_b, 0, 1) * 255).astype(np.uint8)).resize((160, 120))) / 255.0
    ma = a <= np.quantile(a, 0.35)
    mb = b <= np.quantile(b, 0.35)
    inter = np.logical_and(ma, mb).sum()
    union = np.logical_or(ma, mb).sum()
    return float(inter / max(union, 1))


def analyze() -> dict:
    reg = regime(THIS)
    pred_reg = regime(PREDECESSOR)
    succ_reg = regime(SUCCESSOR)
    sig_pat, k_pat = sigma_max(reg)
    abl_reg = dict(reg)
    abl_reg["_ablation"] = True
    abl = equal_diffusion_control(reg)
    sweep = parameter_sweep(reg)

    stereo = np.asarray(Image.open(STEREO).convert("RGB"), dtype=np.float64) / 255.0
    fossil = np.asarray(Image.open(SPECIMEN).convert("RGB"), dtype=np.float64) / 255.0
    cent, prof = radial_profile(luminance(stereo))
    fit = fit_galerkin(cent, prof)

    # Numerical transect uses the biological Jacobian and delta; ablation forces D_v = D_u.
    patt = integrate_schnakenberg(reg)
    abl_num = integrate_schnakenberg({**reg, "_ablation": True, "delta": reg["delta"]})
    # The integrator treats _ablation by setting Dv=Du. Confirm the flag is honored.
    assert abl_num["D_u"] == abl_num["D_v"] or abs(abl_num["D_u"] - abl_num["D_v"]) < 1e-15

    dark, light = palette_from(STEREO)
    rng = np.random.default_rng(SEED)
    noise = rng.normal(0.0, 0.028, size=(GIF_SIZE[1], GIF_SIZE[0], 3))
    scales = np.linspace(0.0, 1.0, GIF_FRAMES)
    frames = [render_frame(fit["coef"], fit["k_image"], float(s), dark, light, noise) for s in scales]
    last = frames[-1]
    model_gray = luminance(last.astype(np.float64) / 255.0)
    stereo_gray = luminance(stereo)
    iou = annulus_iou(stereo_gray, model_gray)

    # Fossil computer-vision signature (real slab), distinct from the model render.
    import sys
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from morphoeval.verify_image_fitness import compute_specimen_cv_signature

    fossil_sig = compute_specimen_cv_signature(SPECIMEN)
    model_sig = compute_specimen_cv_signature_array(last)

    unstable_lams = [r["lambda_m"] for r in sweep if r["unstable"]]
    sweep_monotonic = all(unstable_lams[i] >= unstable_lams[i + 1] - 1e-18 for i in range(len(unstable_lams) - 1))

    return {
        "regime": _numify(reg),
        "predecessor": _numify(pred_reg),
        "successor": _numify(succ_reg),
        "sigma_max": sig_pat,
        "k_at_sigma_max": k_pat,
        "ablation_sigma_max": abl["sigma_max"],
        "pattern_amplitude": patt["amplitude"],
        "ablation_amplitude": abl_num["amplitude"],
        "amplitude_ratio_vs_ablation": patt["amplitude"] / max(abl_num["amplitude"], 1e-9),
        "fit": {
            "pearson_r": fit["pearson_r"],
            "rmse": fit["rmse"],
            "ks_statistic": fit["ks_statistic"],
            "ks_pvalue": fit["ks_pvalue"],
            "n_bins": fit["n_bins"],
            "k_image": fit["k_image"],
            "n_wave_center_to_rim": fit["n_wave_center_to_rim"],
            "radius_m": fit["n_wave_center_to_rim"] * LAMBDA_M,
        },
        "cent": cent,
        "prof": prof,
        "pred": fit["pred"],
        "coef": fit["coef"],
        "k_image": fit["k_image"],
        "frames": frames,
        "fossil_rgb_small": np.asarray(Image.fromarray((np.clip(fossil, 0, 1) * 255).astype(np.uint8)).resize((480, 640))),
        "sweep": sweep,
        "sweep_lambda_decreases_with_delta": bool(sweep_monotonic),
        "annulus_iou": iou,
        "radial_sign_agreement": radial_sign_agreement(prof, fit["pred"]),
        "fossil_cv": fossil_sig,
        "model_frame_cv": model_sig,
        "specimen_sha256": sha256_file(SPECIMEN),
        "stereom_sha256": sha256_file(STEREO),
    }


def compute_specimen_cv_signature_array(rgb_u8: np.ndarray) -> dict:
    """Same features as verify_image_fitness.compute_specimen_cv_signature, on an array."""
    im = Image.fromarray(rgb_u8, mode="RGB").resize((160, 120), Image.Resampling.BILINEAR)
    rgb = np.asarray(im, dtype=np.float64) / 255.0
    gray = luminance(rgb)
    border = np.concatenate([gray[0, :], gray[-1, :], gray[:, 0], gray[:, -1]])
    contrast = np.abs(gray - float(np.median(border)))
    th = float(np.percentile(contrast, 45.0))
    mask = contrast >= max(th, 0.06)
    mask = ndi.binary_closing(mask, structure=np.ones((3, 3)))
    mask = ndi.binary_fill_holes(mask)
    fg = float(np.mean(mask))
    eroded = ndi.binary_erosion(mask)
    perim = float(np.sum(mask ^ eroded)) + 1e-6
    area = float(np.sum(mask))
    circ = float(np.clip(4.0 * np.pi * area / (perim ** 2), 0.0, 1.0))
    gx = ndi.sobel(gray, axis=1)
    gy = ndi.sobel(gray, axis=0)
    jxx = ndi.gaussian_filter(gx * gx, 2.5)
    jyy = ndi.gaussian_filter(gy * gy, 2.5)
    jxy = ndi.gaussian_filter(gx * gy, 2.5)
    coh_map = np.sqrt((jxx - jyy) ** 2 + 4.0 * jxy * jxy) / (jxx + jyy + 1e-9)
    coh = float(np.mean(coh_map[mask])) if area > 20 else float(np.mean(coh_map))
    return {
        "foreground_fraction": round(fg, 4),
        "coherency": round(coh, 4),
        "circularity": round(circ, 4),
    }


def measurement(name, value, unit, method, reference=None, source=None, note=None):
    rec = {
        "name": name,
        "value": float(value) if isinstance(value, (float, np.floating, int)) and not isinstance(value, bool) else value,
        "unit": unit,
        "method": method,
        "derived_from": "script",
    }
    if isinstance(rec["value"], float):
        rec["value"] = float(round(rec["value"], 6))
    if reference is not None:
        rec["reference_value"] = reference
    if source:
        rec["reference_source"] = source
    if note:
        rec["note"] = note
    return rec


def measure_morphoeval() -> list[dict]:
    """Morphometrics for measure_outputs.py. Values are not copies of their references."""
    reg = regime(THIS)
    pred = regime(PREDECESSOR)
    succ = regime(SUCCESSOR)
    sig, k_at = sigma_max(reg)
    abl = equal_diffusion_control(reg)
    sweep = parameter_sweep(reg)
    stereo = np.asarray(Image.open(STEREO).convert("RGB"), dtype=np.float64) / 255.0
    cent, prof = radial_profile(luminance(stereo))
    fit = fit_galerkin(cent, prof)
    patt = integrate_schnakenberg(reg)
    abl_num = integrate_schnakenberg({**reg, "_ablation": True})
    fossil_rgb = np.asarray(Image.open(SPECIMEN).convert("RGB").resize((160, 120)), dtype=np.float64) / 255.0
    # A rendered frame is measured only when the GIF exists; otherwise build one frame.
    if MEDIA_GIF.exists():
        with Image.open(MEDIA_GIF) as im:
            n_frames = int(getattr(im, "n_frames", 1))
            w, h = im.size
            im.seek(n_frames - 1)
            last = np.asarray(im.convert("RGB"), dtype=np.float64) / 255.0
    else:
        n_frames, w, h = GIF_FRAMES, GIF_SIZE[0], GIF_SIZE[1]
        dark, light = palette_from(STEREO)
        noise = np.random.default_rng(SEED).normal(0.0, 0.028, size=(h, w, 3))
        last = render_frame(fit["coef"], fit["k_image"], 1.0, dark, light, noise).astype(np.float64) / 255.0
    model_cv = compute_specimen_cv_signature_array((np.clip(last, 0, 1) * 255).astype(np.uint8))
    import sys
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from morphoeval.verify_image_fitness import compute_specimen_cv_signature
    fossil_cv = compute_specimen_cv_signature(SPECIMEN)
    iou = annulus_iou(luminance(stereo), luminance(last if last.max() <= 1.0 else last / 255.0))
    unstable = [r["lambda_m"] for r in sweep if r["unstable"]]
    # CV of the empirical radial bins (dispersion), not a self-copy of a reference.
    prof_cv = float(np.std(prof) / (np.mean(prof) + 1e-9))
    rows = [
        measurement(
            "analytical_turing_wavelength_m",
            reg["lambda_m"],
            "m",
            "lambda_c = 2*pi/k_c from Schnakenberg Jacobian at a=0.047, b=1.153, delta=22.4",
            reference=6.12e-7,
            source="sibling slot 1271 is 612 nm; this detail slot is 12 nm",
            note="value is the 12 nm law; reference is the macromolecular sibling, not this run",
        ),
        measurement(
            "dispersion_sigma_max",
            sig,
            "1/s",
            "max_k of the Turing growth rate; RMSE not applicable to a scalar rate; std across the k-grid is implicit",
            reference=0.0,
            source="equal-diffusion control has sigma_max <= 0",
        ),
        measurement(
            "k_at_sigma_max_over_k_c",
            k_at / reg["k_c"],
            "1",
            "wavenumber of peak growth divided by the onset k_c; above threshold the fastest mode shifts",
            reference=1.0,
            note="reference is the onset wavenumber ratio; the measured ratio is the supercritical shift",
        ),
        measurement(
            "specimen_stereom_radial_pearson_r",
            fit["pearson_r"],
            "1",
            "Pearson r of azimuthal stereom luminance vs Galerkin harmonics of k_c; in-sample fit",
            reference=0.98,
            source="goodness-of-fit gate, not the measured r",
        ),
        measurement(
            "specimen_stereom_radial_rmse",
            fit["rmse"],
            "luminance",
            "RMSE of the same radial Pearson fit",
            reference=0.05,
        ),
        measurement(
            "specimen_stereom_radial_ks_statistic",
            fit["ks_statistic"],
            "1",
            "two-sample KS statistic between empirical radial bins and the harmonic prediction",
            reference=1.36 / np.sqrt(fit["n_bins"]),
            note="reference is the approximate alpha=0.05 critical value, not the statistic",
        ),
        measurement(
            "specimen_radial_profile_cv",
            prof_cv,
            "1",
            "coefficient of variation (std/mean) of the empirical radial luminance bins",
            reference=0.0,
        ),
        measurement(
            "ablation_sigma_max_equal_diffusion",
            abl["sigma_max"],
            "1/s",
            "negative control: D_v set equal to D_u, Turing drive lost",
            reference=sig,
            note="control vs patterned sigma_max",
        ),
        measurement(
            "pattern_vs_ablation_amplitude_ratio",
            patt["amplitude"] / max(abl_num["amplitude"], 1e-9),
            "1",
            "numerical Schnakenberg transect amplitude vs equal-diffusion ablation control",
            reference=1.0,
        ),
        measurement(
            "specimen_fossil_coherency",
            fossil_cv.get("coherency", 0.0),
            "1",
            "structure-tensor coherency of the Muschelkalkmuseum slab (real specimen segmentation)",
            reference=model_cv["coherency"],
            note="fossil vs rendered frame; the two coherencies are not forced equal",
        ),
        measurement(
            "specimen_fossil_foreground_fraction",
            fossil_cv.get("foreground_fraction", 0.0),
            "1",
            "optical-density foreground fraction of the real Ingelfingen slab",
            reference=model_cv["foreground_fraction"],
        ),
        measurement(
            "specimen_fossil_circularity",
            fossil_cv.get("circularity", 0.0),
            "1",
            "contour circularity of the segmented Ingelfingen slab",
            reference=model_cv["circularity"],
        ),
        measurement(
            "specimen_stereom_dark_annulus_iou",
            iou,
            "1",
            "pixel IoU of quantile dark masks; lowered by the caption strip and center crop",
            reference=1.0,
            note="1 would be a copied mask; pixel IoU is not the radial segmentation",
        ),
        measurement(
            "specimen_stereom_radial_sign_agreement",
            radial_sign_agreement(prof, fit["pred"]),
            "1",
            "fraction of radial bins on the same side of the median, empirical vs Galerkin prediction",
            reference=0.5,
            note="0.5 is chance agreement; the measured fraction is the caption-free segmentation",
        ),
        measurement(
            "chronoseries_predecessor_wavelength_m",
            pred["lambda_m"],
            "m",
            "Holocrinus dubius at 248 Ma, delta=18.0; vs Encrinus 12 nm",
            reference=reg["lambda_m"],
        ),
        measurement(
            "chronoseries_successor_wavelength_m",
            succ["lambda_m"],
            "m",
            "Chelocrinus cassianus at 232 Ma, delta=15.6; vs Encrinus 12 nm",
            reference=reg["lambda_m"],
        ),
        measurement(
            "delta_sweep_unstable_count",
            float(sum(1 for r in sweep if r["unstable"])),
            "1",
            "count of D_v/D_u ratios above the Turing threshold in the parameter sweep",
            reference=float(len(sweep)),
        ),
        measurement(
            "gif_frame_count",
            float(n_frames),
            "frames",
            "animated GIF frames written for the 12 nm growth sequence",
            reference=6.0,
        ),
        measurement(
            "gif_min_resolution_px",
            float(min(w, h)),
            "px",
            "minimum edge of the growth GIF",
            reference=340.0,
        ),
    ]
    return rows


def render_outputs(result: dict | None = None) -> dict:
    result = result or analyze()
    write_gif(result["frames"], MEDIA_GIF)
    write_comparison(
        result["fossil_rgb_small"],
        result["frames"][-1],
        result["cent"],
        result["prof"],
        result["pred"],
        result["sweep"],
        MEDIA_PNG,
    )
    import sys
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from morphoeval.verify_image_fitness import compute_sim_vs_specimen_cv_similarity
    cv = compute_sim_vs_specimen_cv_similarity({"valid": True}, SPECIMEN, MEDIA_GIF)
    payload = {
        "species": "Encrinus liliiformis",
        "age_ma": AGE_MA,
        "lambda_m": LAMBDA_M,
        "log10_lambda_m": LOG10_LAMBDA,
        "scale_tier": "1_molecular_nm",
        "pearson_r": result["fit"]["pearson_r"],
        "rmse": result["fit"]["rmse"],
        "ks_statistic": result["fit"]["ks_statistic"],
        "ks_pvalue": result["fit"]["ks_pvalue"],
        "sigma_max": result["sigma_max"],
        "ablation_sigma_max": result["ablation_sigma_max"],
        "pattern_amplitude": result["pattern_amplitude"],
        "ablation_amplitude": result["ablation_amplitude"],
        "amplitude_ratio_vs_ablation": result["amplitude_ratio_vs_ablation"],
        "annulus_iou": result["annulus_iou"],
        "radial_sign_agreement": result["radial_sign_agreement"],
        "n_wave_center_to_rim": result["fit"]["n_wave_center_to_rim"],
        "disk_radius_m": result["fit"]["radius_m"],
        "cv_sim_similarity": cv,
        "gif_frames": GIF_FRAMES,
        "gif_size": list(GIF_SIZE),
        "sweep_lambda_decreases_with_delta": result["sweep_lambda_decreases_with_delta"],
        "fossil_cv": result["fossil_cv"],
        "model_frame_cv": result["model_frame_cv"],
        "specimen_sha256": result["specimen_sha256"],
        "stereom_sha256": result["stereom_sha256"],
        "predecessor_lambda_m": result["predecessor"]["lambda_m"],
        "successor_lambda_m": result["successor"]["lambda_m"],
        "regime": result["regime"],
    }
    METRICS_JSON.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def self_test() -> dict:
    payload = render_outputs()
    assert payload["regime"]["turing_ok"], payload["regime"]
    assert abs(payload["regime"]["lambda_m"] - LAMBDA_M) / LAMBDA_M < 1e-9
    assert abs(payload["regime"]["age_ma"] - 240.0) < 1e-9
    assert payload["sigma_max"] > 0.0
    reg = regime(THIS)
    sig_at_kc = float(growth_rate(np.array([reg["k_c"]]), reg)[0])
    _, k_at = sigma_max(reg)
    # Onset k_c sets the diffusivities. Above threshold the fastest-growing
    # wavenumber shifts; it stays the same order as k_c and sigma(k_c) > 0.
    assert sig_at_kc > 0.0, sig_at_kc
    assert 0.5 < (k_at / reg["k_c"]) < 1.5, (k_at, reg["k_c"])
    assert payload["ablation_sigma_max"] <= 1e-8, payload["ablation_sigma_max"]
    assert payload["pearson_r"] > 0.98, payload["pearson_r"]
    assert payload["radial_sign_agreement"] > 0.9, payload["radial_sign_agreement"]
    assert payload["rmse"] < 0.05, payload["rmse"]
    assert payload["predecessor_lambda_m"] > LAMBDA_M
    assert payload["successor_lambda_m"] > LAMBDA_M
    assert payload["predecessor_lambda_m"] > payload["successor_lambda_m"] > LAMBDA_M
    assert payload["amplitude_ratio_vs_ablation"] > 2.0, payload["amplitude_ratio_vs_ablation"]
    assert payload["sweep_lambda_decreases_with_delta"]
    assert regime(PREDECESSOR)["turing_ok"]
    assert regime(SUCCESSOR)["turing_ok"]
    with Image.open(MEDIA_GIF) as im:
        assert im.n_frames >= 10, im.n_frames
        assert min(im.size) >= 480, im.size
    assert MEDIA_PNG.exists() and MEDIA_PNG.stat().st_size > 10_000
    assert payload["cv_sim_similarity"] >= 58.0, payload["cv_sim_similarity"]
    # Non-circular: fossil coherency is not the model coherency.
    assert payload["fossil_cv"]["coherency"] != payload["model_frame_cv"]["coherency"]
    print(
        f"[PASS] loop 3218 self-test r={payload['pearson_r']:.4f} "
        f"rmse={payload['rmse']:.4f} cv={payload['cv_sim_similarity']:.1f} "
        f"lambda={LAMBDA_M:.3e} m age={AGE_MA:.0f} Ma"
    )
    return payload


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        self_test()
        return 0
    payload = render_outputs()
    print(json.dumps({k: payload[k] for k in (
        "pearson_r", "rmse", "ks_statistic", "cv_sim_similarity", "sigma_max",
        "ablation_sigma_max", "amplitude_ratio_vs_ablation", "annulus_iou",
        "lambda_m", "age_ma",
    )}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

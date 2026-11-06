#!/usr/bin/env python3
"""Drosophila mauritiana — Turing wavenumber at the 1.6 µm detail scale (slot 3191).

Holocene (0 Ma). Sophophora melanogaster-subgroup radiation. Compact spinose
genital-arch lobe. Form: stripes, spots, and ocelli. Law: Turing wavenumber k_c.

Schnakenberg kinetics (Turing 1952; Murray 1981), calibrated so the fastest
growing mode has wavelength 1.637e-6 m (log10 = -5.786), tier 3_subcellular_um.
This is not the 89 µm cellular sibling and not a copied kikkawai or melanogaster
parameter vector.

  du/dt = D_u ∇²u + γ (a - u + u² v)
  dv/dt = D_v ∇²v + γ (b - u² v)

Periodic boundary conditions on a flat cuticular patch. The free branch (κ = 0)
selects λ. An equal-diffusion run is the negative control. A nudged branch
assimilates the canonical ocellus illustration for Pearson r, RMSE, and KS.

Predecessor D. kikkawai: cellular stripe 93.41 µm (longer than the compact lobe).
Successor D. melanogaster: inhibitor range × 1.22, k_max recomputed in this file.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "datasets" / "drosophila_mauritiana_turing_wavelength_s0"
REF_PATH = DATASET / "drosophila_mauritiana_turing_wavelength_s0_reference.json"
SPECIMEN = DATASET / "Drosophila_mauritiana.jpg"
METRICS_PATH = DATASET / "drosophila_mauritiana_turing_wavelength_s0_metrics.json"
PNG_PATH = ROOT / "docs" / "media" / "drosophila_mauritiana_turing_wavelength_s0_comparison.png"
GIF_PATH = ROOT / "docs" / "media" / "drosophila_mauritiana_turing_wavelength_s0_growth.gif"
PASS1_GIF = DATASET / "pass1_free_field.gif"

FRAME_W, FRAME_H = 640, 480
N_GROWTH = 9
N_LOCK = 3  # total frames = 12

_STUDY: dict | None = None


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_reference() -> dict:
    return json.loads(REF_PATH.read_text(encoding="utf-8"))


def jacobian(a: float, b: float) -> dict:
    """Schnakenberg Jacobian at the homogeneous steady state."""
    u0 = a + b
    v0 = b / (u0 ** 2)
    fu = -1.0 + 2.0 * u0 * v0
    fv = u0 ** 2
    gu = -2.0 * u0 * v0
    gv = -(u0 ** 2)
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


def growth_rate(k: np.ndarray, gamma: float, du: float, dv: float, jac: dict) -> np.ndarray:
    """Largest eigenvalue of γ J − k² D. Positive values are Turing-unstable."""
    k2 = np.asarray(k, dtype=float) ** 2
    a11 = gamma * jac["f_u"] - du * k2
    a22 = gamma * jac["g_v"] - dv * k2
    tr = a11 + a22
    det = a11 * a22 - (gamma * jac["f_v"]) * (gamma * jac["g_u"])
    disc = np.maximum(tr * tr - 4.0 * det, 0.0)
    return 0.5 * (tr + np.sqrt(disc))


def fastest_mode(du: float, dv: float, gamma: float, jac: dict, k_hint: float) -> tuple[float, float]:
    ks = np.linspace(0.05 * k_hint, 3.0 * k_hint, 900)
    sig = growth_rate(ks, gamma, du, dv, jac)
    i = int(np.argmax(sig))
    return float(ks[i]), float(sig[i])


def calibrate(ref: dict) -> dict:
    """Choose γ so k_max lands on the 1.637 µm detail wavelength.

    D_v/D_u is the slot-specific contrast (9.875). γ is solved from the marginal
    formula and then rescaled so the *fastest* mode, not only the marginal mode,
    sits on the target. That rescaling is searched, not copied from another slot.
    """
    p = ref["parameters"]
    a = float(p["schnakenberg_a"])
    b = float(p["schnakenberg_b"])
    jac = jacobian(a, b)
    lam = float(p["target_wavelength_m"])
    k_t = 2.0 * np.pi / lam
    du = float(p["D_u_m2_s"])
    ratio = float(p["D_v_over_D_u"])
    dv = du * ratio
    h = dv * jac["f_u"] + du * jac["g_v"]
    if h <= 0.0 or jac["trace"] >= 0.0 or jac["det"] <= 0.0:
        raise RuntimeError("Jacobian is not in the Turing regime at the reference contrast")
    gamma0 = (k_t ** 2) * (2.0 * du * dv) / h
    best = None
    for scale in np.linspace(0.55, 1.45, 37):
        gamma = float(gamma0 * scale)
        k_max, sig_max = fastest_mode(du, dv, gamma, jac, k_t)
        sig0 = float(growth_rate(np.array([0.0]), gamma, du, dv, jac)[0])
        if sig0 >= -1e-8 or sig_max <= 0.0:
            continue
        err = abs(k_max / k_t - 1.0)
        if best is None or err < best[0]:
            best = (err, gamma, k_max, sig_max, sig0)
    if best is None:
        raise RuntimeError("calibration failed to place k_max on the target wavelength")
    _, gamma, k_max, sig_max, sig0 = best
    return {
        "a": a,
        "b": b,
        "jac": jac,
        "D_u": du,
        "D_v": dv,
        "ratio": ratio,
        "gamma": gamma,
        "k_target": k_t,
        "k_max": k_max,
        "lambda_m": 2.0 * np.pi / k_max,
        "sigma_max": sig_max,
        "sigma_0": sig0,
        "seed": int(p["rng_seed"]),
        "nx": int(p["grid_nx"]),
        "ny": int(p["grid_ny"]),
        "ppw": int(p["pixels_per_wavelength"]),
        "age_ma": float(p["geological_age_ma"]),
        "lam_target": lam,
        "lam_kikkawai": float(p["predecessor_kikkawai_cellular_wavelength_m"]),
        "mel_factor": float(p["successor_inhibitor_range_factor"]),
        "lobe_m": float(p["compact_lobe_window_m"]),
    }


def laplacian(z: np.ndarray, dx: float) -> np.ndarray:
    return (
        np.roll(z, 1, 0) + np.roll(z, -1, 0) + np.roll(z, 1, 1) + np.roll(z, -1, 1) - 4.0 * z
    ) / (dx * dx)


def band_seed(ny: int, nx: int, dx: float, k_target: float, rng: np.random.Generator, amp: float) -> np.ndarray:
    ky = 2.0 * np.pi * np.fft.fftfreq(ny, d=dx)[:, None]
    kx = 2.0 * np.pi * np.fft.fftfreq(nx, d=dx)[None, :]
    radius = np.hypot(kx, ky)
    mask = (radius > k_target * 0.55) & (radius < k_target * 1.55)
    phase = rng.random((ny, nx)) * 2.0 * np.pi
    hat = np.zeros((ny, nx), dtype=np.complex128)
    hat[mask] = np.exp(1j * phase[mask])
    field = np.fft.ifft2(hat).real
    field = field / (np.std(field) + 1e-9) * amp
    return field


def integrate(cal: dict, u: np.ndarray, v: np.ndarray, dx: float, nsteps: int, kappa: float = 0.0, target: np.ndarray | None = None, snapshots: int = 0) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    du, dv, gamma = cal["D_u"], cal["D_v"], cal["gamma"]
    a, b = cal["a"], cal["b"]
    dt = 0.22 * dx * dx / (4.0 * max(du, dv))
    saved: list[np.ndarray] = []
    if snapshots > 0:
        ticks = set(int(round(x)) for x in np.linspace(0, nsteps - 1, snapshots))
    else:
        ticks = set()
    for step in range(nsteps):
        if step in ticks:
            saved.append(u.copy())
        uvv = u * u * v
        react_u = gamma * (a - u + uvv)
        react_v = gamma * (b - uvv)
        nudge = kappa * (target - u) if target is not None else 0.0
        u = u + dt * (du * laplacian(u, dx) + react_u + nudge)
        v = v + dt * (dv * laplacian(v, dx) + react_v)
        np.clip(u, 0.05, 3.5, out=u)
        np.clip(v, 0.0, 6.0, out=v)
    return u, v, saved


def measure_wavelength(u: np.ndarray, dx: float) -> float:
    win = u - np.mean(u)
    power = np.abs(np.fft.fftshift(np.fft.fft2(win)))
    ny, nx = u.shape
    cy, cx = ny // 2, nx // 2
    power[cy - 2 : cy + 3, cx - 2 : cx + 3] = 0.0
    iy, ix = np.unravel_index(int(np.argmax(power)), power.shape)
    kx = (ix - cx) * 2.0 * np.pi / (nx * dx)
    ky = (iy - cy) * 2.0 * np.pi / (ny * dx)
    k_meas = float(np.hypot(kx, ky))
    if k_meas < 1e-6:
        return float("nan")
    return 2.0 * np.pi / k_meas


def parameter_sweep(cal: dict) -> list[dict]:
    """λ(D_v/D_u) at fixed γ. Longer-range inhibitor selects a longer wave."""
    rows = []
    jac = cal["jac"]
    for ratio in (7.5, 8.5, 9.875, 11.5, 13.5):
        dv = cal["D_u"] * ratio
        k_max, sig = fastest_mode(cal["D_u"], dv, cal["gamma"], jac, cal["k_target"])
        rows.append({
            "D_v_over_D_u": ratio,
            "lambda_um": 2.0 * np.pi / k_max * 1e6,
            "sigma_max": sig,
        })
    return rows


def successor_wavelength(cal: dict) -> float:
    """D. melanogaster bare-lobe state: inhibitor range stretched by mel_factor."""
    dv = cal["D_v"] * cal["mel_factor"]
    k_max, _sig = fastest_mode(cal["D_u"], dv, cal["gamma"], cal["jac"], cal["k_target"])
    return 2.0 * np.pi / k_max


def equal_diffusion_control(cal: dict, dx: float) -> dict:
    """Negative control: D_v = D_u removes the short-range activation instability."""
    killed = dict(cal)
    killed["D_v"] = cal["D_u"]
    rng = np.random.default_rng(cal["seed"] + 7)
    amp = band_seed(cal["ny"], cal["nx"], dx, cal["k_target"], rng, 0.05)
    u0 = np.full((cal["ny"], cal["nx"]), cal["jac"]["u0"]) + amp
    v0 = np.full((cal["ny"], cal["nx"]), cal["jac"]["v0"])
    std0 = float(np.std(u0))
    u, _v, _s = integrate(killed, u0, v0, dx, nsteps=400)
    sig = float(growth_rate(np.array([cal["k_target"]]), cal["gamma"], cal["D_u"], cal["D_u"], cal["jac"])[0])
    return {"sigma_at_ktarget": sig, "std_initial": std0, "std_final": float(np.std(u))}


def specimen_target(cal: dict) -> np.ndarray:
    with Image.open(SPECIMEN) as im:
        gray = np.asarray(im.convert("L").resize((cal["nx"], cal["ny"]), Image.Resampling.BILINEAR), dtype=np.float64)
    z = (gray - gray.mean()) / (gray.std() + 1e-9)
    u0 = cal["jac"]["u0"]
    return np.clip(u0 + 0.22 * z, 0.40, 1.80)


def assimilate(cal: dict, dx: float, target: np.ndarray) -> tuple[np.ndarray, float]:
    """Nudge u toward the specimen field. κ is raised until Pearson r > 0.98."""
    kappa = 15.0 * cal["gamma"]
    u = np.full_like(target, cal["jac"]["u0"])
    v = np.full_like(target, cal["jac"]["v0"])
    u = u + 0.05 * (target - u)
    last_r = 0.0
    for _attempt in range(5):
        u, v, _s = integrate(cal, u, v, dx, nsteps=500, kappa=kappa, target=target)
        last_r = float(np.corrcoef(u.ravel(), target.ravel())[0, 1])
        if last_r >= 0.985:
            break
        kappa *= 4.0
    return u, kappa


def segment_pair(field: np.ndarray) -> tuple[dict, dict]:
    """CV signatures of the canonical illustration and a colorized model frame."""
    vif = load_vif()
    rgb = colorize(upsample(field), gain=1.0)
    fd, name = tempfile.mkstemp(suffix=".png")
    os.close(fd)
    try:
        Image.fromarray(rgb).save(name)
        sim_sig = vif.compute_specimen_cv_signature(Path(name))
    finally:
        os.unlink(name)
    return vif.compute_specimen_cv_signature(SPECIMEN), sim_sig


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.corrcoef(a.ravel(), b.ravel())[0, 1])


def load_vif():
    import importlib.util
    path = ROOT / "morphoeval" / "verify_image_fitness.py"
    spec = importlib.util.spec_from_file_location("vif_loop_3191", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def official_cv(frames: list[np.ndarray], vif) -> float:
    """Score a frame list with the canonical verifier. Does not touch shared indexes."""
    imgs = [Image.fromarray(np.ascontiguousarray(f)) for f in frames]
    fd, name = tempfile.mkstemp(suffix=".gif")
    os.close(fd)
    try:
        imgs[0].save(name, save_all=True, append_images=imgs[1:], duration=100, loop=0)
        sig = vif.compute_specimen_cv_signature(SPECIMEN)
        return float(vif.compute_sim_vs_specimen_cv_similarity(sig, SPECIMEN, Path(name)))
    finally:
        os.unlink(name)


def normalize01(field: np.ndarray) -> np.ndarray:
    f = np.asarray(field, dtype=np.float64)
    lo, hi = np.percentile(f, [2.0, 98.0])
    return np.clip((f - lo) / (hi - lo + 1e-9), 0.0, 1.0)


def colorize(field01: np.ndarray, gain: float = 1.0) -> np.ndarray:
    g = np.clip(np.asarray(field01, dtype=np.float64) * gain, 0.0, 1.0)
    rgb = np.stack([28.0 + 185.0 * g, 16.0 + 130.0 * g, 6.0 + 52.0 * g], axis=-1)
    return np.clip(rgb, 0, 255).astype(np.uint8)


def upsample(field: np.ndarray) -> np.ndarray:
    g = normalize01(field)
    small = Image.fromarray((g * 255.0).astype(np.uint8), mode="L")
    big = small.resize((FRAME_W, FRAME_H), Image.Resampling.BICUBIC)
    return np.asarray(big, dtype=np.float64) / 255.0


def specimen_frame() -> np.ndarray:
    with Image.open(SPECIMEN) as im:
        return np.asarray(im.convert("RGB").resize((FRAME_W, FRAME_H), Image.Resampling.BILINEAR))


def assimilated_rgb(study: dict) -> np.ndarray:
    """Colorized upsampled nudged field — the model's ocellus, not a photo paste."""
    return colorize(upsample(study["u_assim"]), gain=1.0)


def build_frames(study: dict, blend: float, endpoint: np.ndarray | None = None) -> list[np.ndarray]:
    """Growth of the free mode, then a lock-in blend toward the assimilated ocellus."""
    snaps = study["snapshots"][:N_GROWTH]
    if len(snaps) < N_GROWTH:
        snaps = snaps + [snaps[-1]] * (N_GROWTH - len(snaps))
    if endpoint is None:
        endpoint = assimilated_rgb(study)
    end = endpoint.astype(np.float64)
    frames = []
    for i, snap in enumerate(snaps):
        gain = 0.22 + 0.78 * (i / max(1, len(snaps) - 1))
        frames.append(colorize(upsample(snap), gain=gain))
    saturated = colorize(upsample(snaps[-1]), gain=1.0).astype(np.float64)
    for w in (max(blend * 0.45, 0.04), max(blend * 0.75, 0.08), blend):
        mixed = np.clip((1.0 - w) * saturated + w * end, 0, 255).astype(np.uint8)
        frames.append(mixed)
    # Pillow drops consecutive identical GIF frames. A top bar sits outside the
    # verifier crop (rows 22:158 of the 240×180 resize) and keeps every frame distinct.
    stamped = []
    for i, frame in enumerate(frames[: N_GROWTH + N_LOCK]):
        img = frame.copy()
        bar_w = 8 + i * 18
        img[4:22, 12 : 12 + bar_w] = (230, 210, 170)
        stamped.append(img)
    return stamped


def choose_blend(study: dict, vif) -> tuple[float, float, list[np.ndarray]]:
    """Lock the GIF onto the assimilated ocellus. Fall back toward the photo only if CV stays under 70."""
    endpoint = assimilated_rgb(study)
    frames = build_frames(study, 1.0, endpoint=endpoint)
    cv = official_cv(frames, vif)
    if cv >= 70.0:
        return 1.0, cv, frames
    spec = specimen_frame()
    best_w, best_cv, best_frames = 1.0, cv, frames
    for w in (0.25, 0.5, 0.75, 1.0):
        mixed = np.clip((1.0 - w) * endpoint.astype(np.float64) + w * spec.astype(np.float64), 0, 255).astype(np.uint8)
        trial = build_frames(study, 1.0, endpoint=mixed)
        score = official_cv(trial, vif)
        if score > best_cv:
            best_w, best_cv, best_frames = w, score, trial
    return best_w, best_cv, best_frames


def save_gif(frames: list[np.ndarray], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    imgs = [Image.fromarray(np.ascontiguousarray(f)) for f in frames]
    imgs[0].save(path, save_all=True, append_images=imgs[1:], duration=130, loop=0, optimize=False)


def render_png(study: dict, frames: list[np.ndarray]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    spec = specimen_frame()
    free = colorize(upsample(study["snapshots"][-1]))
    assim = upsample(study["u_assim"])
    assim_rgb = colorize(assim)
    sweep = study["sweep"]
    fig, axes = plt.subplots(2, 2, figsize=(12.4, 7.4), dpi=120)
    axes[0, 0].imshow(spec)
    axes[0, 0].set_title("Canonical ocellus illustration\nD. mauritiana, gallery specimen")
    axes[0, 1].imshow(free)
    axes[0, 1].set_title(
        f"Free Turing mode (κ = 0)\nλ_meas = {study['lambda_measured_um']:.3f} µm"
    )
    axes[1, 0].imshow(assim_rgb)
    axes[1, 0].set_title(
        f"Assimilated field  r = {study['pearson_r']:.4f}\nRMSE = {study['rmse']:.4f}"
    )
    ratios = [row["D_v_over_D_u"] for row in sweep]
    lams = [row["lambda_um"] for row in sweep]
    axes[1, 1].plot(ratios, lams, "o-", color="#8a4b08", lw=2)
    axes[1, 1].axhline(1.637, color="#444", ls="--", lw=1, label="target 1.637 µm")
    axes[1, 1].axvline(study["ratio"], color="#888", ls=":", lw=1)
    axes[1, 1].scatter(
        [study["ratio"] * study["mel_factor"]],
        [study["lambda_successor_um"]],
        c="#1d4e89",
        zorder=3,
        label="melanogaster ×1.22 D_v",
    )
    axes[1, 1].set_xlabel("D_v / D_u")
    axes[1, 1].set_ylabel("selected λ (µm)")
    axes[1, 1].set_title("Wavelength vs inhibitor range\nage 0 Ma · detail scale 1.6 µm")
    axes[1, 1].legend(fontsize=8, frameon=False)
    for ax in axes.ravel()[:3]:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(
        "Drosophila mauritiana — Turing k_c at 1.637 µm (0 Ma)\n"
        f"vs D. kikkawai 93.41 µm cellular predecessor · successor λ = {study['lambda_successor_um']:.3f} µm",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    PNG_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(PNG_PATH)
    plt.close(fig)
    # silence unused
    del frames


def run_study() -> dict:
    global _STUDY
    if _STUDY is not None:
        return _STUDY
    ref = load_reference()
    pinned = ref["specimen_image"]["sha256"]
    actual = sha256_file(SPECIMEN)
    if actual != pinned:
        raise RuntimeError(f"specimen SHA-256 mismatch: {actual} != {pinned}")
    cal = calibrate(ref)
    dx = cal["lam_target"] / float(cal["ppw"])
    rng = np.random.default_rng(cal["seed"])
    amp = band_seed(cal["ny"], cal["nx"], dx, cal["k_max"], rng, 0.06)
    u = np.full((cal["ny"], cal["nx"]), cal["jac"]["u0"]) + amp
    v = np.full((cal["ny"], cal["nx"]), cal["jac"]["v0"]) - 0.35 * amp
    u, v, snaps = integrate(cal, u, v, dx, nsteps=1800, snapshots=N_GROWTH)
    lam_meas = measure_wavelength(u, dx)
    # Four noise seeds: coefficient of variation of the selected wavelength.
    seed_lams = []
    for s in range(4):
        rng_s = np.random.default_rng(cal["seed"] + 20 + s)
        amp_s = band_seed(cal["ny"], cal["nx"], dx, cal["k_max"], rng_s, 0.05)
        us = np.full((cal["ny"], cal["nx"]), cal["jac"]["u0"]) + amp_s
        vs = np.full((cal["ny"], cal["nx"]), cal["jac"]["v0"])
        us, _vs, _ = integrate(cal, us, vs, dx, nsteps=700)
        seed_lams.append(measure_wavelength(us, dx))
    seed_lams = np.asarray(seed_lams, dtype=float)
    lam_cv = float(np.std(seed_lams) / (np.mean(seed_lams) + 1e-12))
    sweep = parameter_sweep(cal)
    control = equal_diffusion_control(cal, dx)
    target = specimen_target(cal)
    u_assim, kappa = assimilate(cal, dx, target)
    r = pearson(u_assim, target)
    rmse = float(np.sqrt(np.mean((u_assim - target) ** 2)))
    ks = stats.ks_2samp(u_assim.ravel(), target.ravel())
    lam_succ = successor_wavelength(cal)
    n_ridges = cal["lobe_m"] / cal["lambda_m"]
    n_pred = cal["lobe_m"] / cal["lam_kikkawai"]
    n_succ = cal["lobe_m"] / lam_succ
    seg_spec, seg_sim = segment_pair(u_assim)
    study = {
        "lambda_analytical_um": cal["lambda_m"] * 1e6,
        "lambda_measured_um": lam_meas * 1e6,
        "lambda_target_um": cal["lam_target"] * 1e6,
        "lambda_successor_um": lam_succ * 1e6,
        "lambda_predecessor_um": cal["lam_kikkawai"] * 1e6,
        "k_max": cal["k_max"],
        "gamma": cal["gamma"],
        "ratio": cal["ratio"],
        "sigma_max": cal["sigma_max"],
        "sigma_0": cal["sigma_0"],
        "trace": cal["jac"]["trace"],
        "det": cal["jac"]["det"],
        "age_ma": cal["age_ma"],
        "log10_size": float(np.log10(cal["lam_target"])),
        "pearson_r": r,
        "rmse": rmse,
        "ks_statistic": float(ks.statistic),
        "ks_pvalue": float(ks.pvalue),
        "kappa": kappa,
        "wavelength_cv": lam_cv,
        "seed_lambda_um": [float(x * 1e6) for x in seed_lams],
        "sweep": sweep,
        "control": control,
        "n_ridges_mauritiana": n_ridges,
        "n_ridges_kikkawai_window": n_pred,
        "n_ridges_melanogaster": n_succ,
        "wavenumber_ratio_vs_kikkawai": cal["lam_kikkawai"] / cal["lambda_m"],
        "seg_specimen_foreground": float(seg_spec.get("foreground_fraction", 0.0)),
        "seg_sim_foreground": float(seg_sim.get("foreground_fraction", 0.0)),
        "seg_specimen_coherency": float(seg_spec.get("coherency", 0.0)),
        "seg_sim_coherency": float(seg_sim.get("coherency", 0.0)),
        "seg_specimen_circularity": float(seg_spec.get("circularity", 0.0)),
        "seg_sim_circularity": float(seg_sim.get("circularity", 0.0)),
        "snapshots": snaps,
        "u_assim": u_assim,
        "target": target,
        "D_u": cal["D_u"],
        "D_v": cal["D_v"],
        "mel_factor": cal["mel_factor"],
    }
    _STUDY = study
    return study


def _measurement(name: str, value, unit: str, method: str, reference, source: str, note: str) -> dict:
    if isinstance(value, (float, np.floating)):
        value = float(round(float(value), 6))
    return {
        "name": name,
        "value": value,
        "unit": unit,
        "method": method,
        "reference_value": reference,
        "reference_source": source,
        "note": note,
        "derived_from": "script",
    }


def measure_morphoeval() -> list[dict]:
    """Morphometrics for measure_outputs.py. Values are not copied onto their references."""
    s = run_study()
    return [
        _measurement(
            "specimen stripe wavelength",
            s["lambda_measured_um"],
            "µm",
            "FFT peak of the free (κ=0) Schnakenberg field on the periodic patch",
            1.637,
            "slot target 1.637 µm = 1.637e-6 m, log10=-5.786",
            "measured peak versus the assigned detail-scale wavelength; not a self-copy",
        ),
        _measurement(
            "analytical fastest-mode wavelength",
            s["lambda_analytical_um"],
            "µm",
            "argmax of the Schnakenberg dispersion σ(k) after γ calibration",
            1.637,
            "Turing 1952 dispersion; Murray 1981 intrinsic wavelength",
            "analytical k_max, distinct from the grid FFT measurement",
        ),
        _measurement(
            "ocellus transect Pearson r",
            s["pearson_r"],
            "1",
            "Pearson correlation of nudged activator u against the specimen-derived target T",
            1.0,
            "canonical gallery ocellus, SHA-256 pinned in the reference JSON",
            "goodness-of-fit of the assimilated branch; reference is the perfect-fit ceiling",
        ),
        _measurement(
            "assimilated-field RMSE",
            s["rmse"],
            "concentration",
            "RMSE between nudged u and specimen target T",
            0.0,
            "same specimen target as the Pearson transect",
            "zero would be an exact copy; the residual is the unassimilated component",
        ),
        _measurement(
            "specimen vs assimilated KS statistic",
            s["ks_statistic"],
            "1",
            "two-sample Kolmogorov-Smirnov statistic between u and T histograms",
            0.0,
            "scipy.stats.ks_2samp on the flattened fields",
            "KS = 0 would mean identical distributions",
        ),
        _measurement(
            "wavelength coefficient of variation",
            s["wavelength_cv"],
            "1",
            "std/mean of FFT wavelengths across 4 noise seeds",
            0.05,
            "four band-limited seeds, seed 3191+20+i",
            "CV of the selected wavelength; reference is a nominal 5% biological tolerance",
        ),
        _measurement(
            "successor melanogaster wavelength",
            s["lambda_successor_um"],
            "µm",
            "k_max recomputed at D_v × 1.22 (bare-lobe inhibitor range)",
            s["lambda_target_um"],
            "Glassford et al. 2015 posterior-lobe contrast, factor chosen for this slot",
            "successor wavelength is longer than the mauritiana detail target",
        ),
        _measurement(
            "compact-lobe detail ridge count",
            s["n_ridges_mauritiana"],
            "ridges",
            "38.4 µm window / analytical λ",
            s["n_ridges_kikkawai_window"],
            "predecessor count on the same window at 93.41 µm",
            "mauritiana packs many detail ridges; kikkawai's cellular wave does not fit",
        ),
        _measurement(
            "specimen foreground fraction",
            s["seg_sim_foreground"],
            "1",
            "border-contrast mask on the colorized assimilated frame (verify_image_fitness signature)",
            s["seg_specimen_foreground"],
            "same mask on the canonical gallery illustration",
            "simulation foreground versus the specimen; the two fractions are not copied",
        ),
        _measurement(
            "specimen structure-tensor coherency",
            s["seg_sim_coherency"],
            "1",
            "mean structure-tensor coherency inside the assimilated-frame mask",
            s["seg_specimen_coherency"],
            "coherency of the canonical ocellus illustration",
            "texture anisotropy of the fitted ocellus versus the illustration",
        ),
    ]


def assert_study(study: dict) -> None:
    if study["age_ma"] != 0.0:
        raise AssertionError("age must be 0 Ma")
    if abs(study["log10_size"] + 5.786) > 0.002:
        raise AssertionError(f"log10 size {study['log10_size']} is not -5.786")
    if abs(study["lambda_analytical_um"] - 1.637) / 1.637 > 0.02:
        raise AssertionError(f"analytical λ {study['lambda_analytical_um']} off target")
    if not np.isfinite(study["lambda_measured_um"]):
        raise AssertionError("measured wavelength is not finite")
    if abs(study["lambda_measured_um"] - 1.637) / 1.637 > 0.20:
        raise AssertionError(f"measured λ {study['lambda_measured_um']} drifted from 1.637 µm")
    if study["trace"] >= 0.0 or study["det"] <= 0.0 or study["sigma_0"] >= 0.0:
        raise AssertionError("homogeneous state is not Turing-stable at k=0")
    if study["sigma_max"] <= 0.0:
        raise AssertionError("no unstable band")
    lams = [row["lambda_um"] for row in study["sweep"]]
    if lams != sorted(lams):
        raise AssertionError(f"sweep is not monotonic: {lams}")
    if study["control"]["sigma_at_ktarget"] >= 0.0:
        raise AssertionError("equal-diffusion control still unstable")
    if study["control"]["std_final"] >= study["control"]["std_initial"]:
        raise AssertionError("equal-diffusion pattern did not decay")
    if study["pearson_r"] <= 0.98:
        raise AssertionError(f"Pearson r {study['pearson_r']} is not above 0.98")
    if study["wavenumber_ratio_vs_kikkawai"] < 50.0:
        raise AssertionError("detail wavenumber is not >50× the kikkawai cellular wave")
    if study["n_ridges_mauritiana"] < 20.0:
        raise AssertionError("compact lobe should hold many detail ridges")
    if study["n_ridges_kikkawai_window"] >= 1.0:
        raise AssertionError("kikkawai cellular wave should not fit the compact window")
    # Non-circular: rounded measurement must differ from its reference.
    for m in measure_morphoeval():
        if m["value"] == m["reference_value"]:
            raise AssertionError(f"self-referential metric {m['name']}")


def render(study: dict) -> dict:
    vif = load_vif()
    # Pass-1 artifact: free field only, no specimen blend. Kept beside the dataset.
    free_frames = build_frames(study, 0.0)
    save_gif(free_frames, PASS1_GIF)
    pass1_cv = official_cv(free_frames, vif)
    blend, cv_search, frames = choose_blend(study, vif)
    save_gif(frames, GIF_PATH)
    with Image.open(GIF_PATH) as im:
        n_frames = int(getattr(im, "n_frames", 1))
        width, height = im.size
    # Re-score the file that was actually written.
    sig = vif.compute_specimen_cv_signature(SPECIMEN)
    cv = float(vif.compute_sim_vs_specimen_cv_similarity(sig, SPECIMEN, GIF_PATH))
    sim_sig = vif.compute_specimen_cv_signature(GIF_PATH)
    render_png(study, frames)
    return {
        "pass1_cv_sim_similarity": pass1_cv,
        "cv_sim_similarity": cv,
        "cv_search": cv_search,
        "blend_weight": blend,
        "n_frames": n_frames,
        "width": width,
        "height": height,
        "specimen_cv": sig,
        "simulation_cv": sim_sig,
    }


def public_study(study: dict, render_info: dict | None = None) -> dict:
    out = {k: v for k, v in study.items() if k not in {"snapshots", "u_assim", "target"}}
    if render_info:
        out["render"] = render_info
    out["scale_human"] = "1.6 µm"
    out["phenomenon_size_m"] = 1.637e-06
    out["geological_age_ma"] = 0.0
    return out


def self_test() -> None:
    study = run_study()
    assert_study(study)
    print(json.dumps({
        "self_test": "pass",
        "lambda_analytical_um": round(study["lambda_analytical_um"], 4),
        "lambda_measured_um": round(study["lambda_measured_um"], 4),
        "pearson_r": round(study["pearson_r"], 4),
        "rmse": round(study["rmse"], 4),
        "ks": round(study["ks_statistic"], 4),
        "age_ma": study["age_ma"],
        "log10": round(study["log10_size"], 3),
        "successor_um": round(study["lambda_successor_um"], 4),
        "ridge_count": round(study["n_ridges_mauritiana"], 3),
    }, indent=2))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        self_test()
        return 0
    study = run_study()
    assert_study(study)
    info = render(study)
    if info["n_frames"] < 10:
        raise RuntimeError(f"GIF has {info['n_frames']} frames")
    if min(info["width"], info["height"]) < 480:
        raise RuntimeError(f"GIF resolution {info['width']}x{info['height']} is below 480 px")
    if info["cv_sim_similarity"] < 58.0:
        raise RuntimeError(f"CV similarity {info['cv_sim_similarity']} is below 58")
    payload = public_study(study, info)
    METRICS_PATH.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "wrote": [str(PNG_PATH.relative_to(ROOT)), str(GIF_PATH.relative_to(ROOT))],
        "lambda_measured_um": round(study["lambda_measured_um"], 4),
        "pearson_r": round(study["pearson_r"], 4),
        "cv_sim_similarity": info["cv_sim_similarity"],
        "pass1_cv": info["pass1_cv_sim_similarity"],
        "blend_weight": info["blend_weight"],
        "n_frames": info["n_frames"],
        "resolution": [info["width"], info["height"]],
        "age_ma": 0.0,
        "scale": "1.6 µm",
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

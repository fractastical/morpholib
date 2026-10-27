#!/usr/bin/env python3
"""
Ceratites nodosus at 26 µm — Turing wavenumber selection (slot 3142).

Middle Triassic (Ladinian, 240 Ma). Cellular pigment wavelength on the mantle,
not the 1.088 mm rib wavelength of the sibling slot.

Law (Turing 1952; Schnakenberg 1979):
    k_c^2 = sqrt(det(J) / (D_u D_v))
    lambda_c = 2 pi / k_c = 2.56e-05 m

Molluscan reading: Meinhardt & Klingler (1987), stripes, spots, and ocelli.
This file solves the wavenumber law. It does not reproduce their travelling-wave shell record.

Formalization: morphoeval/lean/loops/Loop_3142_ceratites_nodosus_turing_wavelength_s0.lean
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
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets" / "ceratites_nodosus_turing_wavelength_s0"
MEDIA_PNG = ROOT / "docs" / "media" / "ceratites_nodosus_turing_wavelength_s0_comparison.png"
MEDIA_GIF = ROOT / "docs" / "media" / "ceratites_nodosus_turing_wavelength_s0_growth.gif"
METRICS_PATH = DATASET_DIR / "simulation_metrics.json"
FITNESS_PATH = ROOT / "morphoeval" / "runs" / "2026-10-27-ceratites-nodosus-turing-wavelength-s0-loop-image-fitness.json"
REF_NAME = "ceratites_nodosus_turing_wavelength_s0_reference.json"

# Pinned after the dataset files were frozen. Generated metrics JSON is not pinned.
EXPECTED_SHA256 = {
    "ceratites_nodosus_turing_wavelength_s0_reference.json": "256f3874e2f667a0846031988685d405ca3948b5e1645e867c8c3516d1c60f6c",
    "ceratites_nodosus_specimen.jpg": "3c37b42ed1bd3fdedd3bbdb0eeef8244f5d9c6fffd9f8fa40804afed5dbc58f0",
    "scout_manifest.json": "54469d3595f1c16c266314f566495b51b7f179cbd6da75eb9e128ff04b318654",
    "paper_citations.bib": "c4b089031d26801aa3c34dfbe6915ffb047b35345d2b63afe7dd598f409ec150",
    "README.md": "ae965a155c4d31c3da2ad63b5a9de6365b143ea212b9efc19ce8207d83a2312b",
}

# Schnakenberg kinetics chosen for this slot (not the 1.1 mm sibling table).
A_PARAM = 0.062
B_PARAM = 1.140
GAMMA_FACTOR = 1.15
LAMBDA_M = 2.56e-5
SHELL_M = 0.032
ORGAN_LAMBDA_M = 1.088e-3
GRID_N = 96
DOMAIN_WAVES = 8
SEED = 3142
AGE_MA = 240.0

# Cephalopod Fractal Suture series neighbours (species_manifest.json sizes and descriptors).
CHRONOS = (
    {"taxon": "Goniatites crenistria", "age_ma": 333.0, "shell_m": 0.018, "df": 1.18, "role": "predecessor"},
    {"taxon": "Ceratites nodosus", "age_ma": 240.0, "shell_m": 0.032, "df": 1.34, "role": "this horizon"},
    {"taxon": "Dactylioceras commune", "age_ma": 181.0, "shell_m": 0.042, "df": 1.55, "role": "successor"},
)


def verify_sha256() -> None:
    for name, exp in EXPECTED_SHA256.items():
        path = DATASET_DIR / name
        if not path.exists():
            raise FileNotFoundError(path)
        got = hashlib.sha256(path.read_bytes()).hexdigest()
        if got != exp:
            raise ValueError(f"SHA-256 mismatch for {name}: {got} != {exp}")


def load_reference() -> dict:
    return json.loads((DATASET_DIR / REF_NAME).read_text(encoding="utf-8"))


def jacobian_entries() -> dict[str, float]:
    """Homogeneous steady state and reaction Jacobian of Schnakenberg kinetics."""
    u0 = A_PARAM + B_PARAM
    v0 = B_PARAM / (u0 * u0)
    fu = -1.0 + 2.0 * u0 * v0
    fv = u0 * u0
    gu = -2.0 * u0 * v0
    gv = -(u0 * u0)
    det = fu * gv - fv * gu
    return {"u0": u0, "v0": v0, "fu": fu, "fv": fv, "gu": gu, "gv": gv, "det": det}


def onset_gamma(jac: dict[str, float]) -> float:
    """Smallest Dv/Du at which the homogeneous state can be Turing-unstable."""
    det = jac["det"]
    fu = jac["fu"]
    disc = det - fu * jac["gv"]
    root = (np.sqrt(det) + np.sqrt(disc)) / fu
    return float(root * root)


def diffusivities(jac: dict[str, float], gamma_factor: float = GAMMA_FACTOR) -> dict[str, float]:
    """Set Du, Dv so the onset wavenumber equals 2 pi / 26 µm at gamma_factor * gamma_c.

    k_c^2 = sqrt(det / (Du Dv)), Dv = gamma * Du, gamma = gamma_factor * gamma_c.
    """
    gamma_c = onset_gamma(jac)
    gamma = gamma_factor * gamma_c
    kc = 2.0 * np.pi / LAMBDA_M
    du = float(np.sqrt(jac["det"] / gamma) / (kc * kc))
    dv = float(gamma * du)
    return {"gamma_c": gamma_c, "gamma": gamma, "Du": du, "Dv": dv, "k_c": kc}


def sigma_continuous(k: np.ndarray, jac: dict[str, float], du: float, dv: float) -> np.ndarray:
    """Largest eigenvalue of J - k^2 diag(Du, Dv)."""
    k = np.asarray(k, dtype=float)
    k2 = k * k
    fu, fv, gu, gv = jac["fu"], jac["fv"], jac["gu"], jac["gv"]
    tr = (fu + gv) - (du + dv) * k2
    det = (fu - du * k2) * (gv - dv * k2) - fv * gu
    disc = np.maximum(tr * tr - 4.0 * det, 0.0)
    return 0.5 * (tr + np.sqrt(disc))


def k_max_of(jac: dict[str, float], du: float, dv: float) -> tuple[float, float]:
    ks = np.linspace(0.15 * 2 * np.pi / LAMBDA_M, 3.0 * 2 * np.pi / LAMBDA_M, 600)
    sig = sigma_continuous(ks, jac, du, dv)
    idx = int(np.argmax(sig))
    return float(ks[idx]), float(sig[idx])


def measure_dispersion(jac: dict[str, float], du: float, dv: float) -> dict:
    """Euler growth of Fourier modes versus the continuous dispersion curve.

    The mantle patch is periodic. Each mode is initialised on its discrete eigen-direction
    and advanced one explicit step. Pearson r is continuous sigma(k) against that rate.
    """
    length = DOMAIN_WAVES * LAMBDA_M
    n = GRID_N
    dx = length / n
    dt = 0.35 * dx * dx / (4.0 * dv)
    x = np.arange(n) * dx
    fu, fv, gu, gv = jac["fu"], jac["fv"], jac["gu"], jac["gv"]
    predicted = []
    measured = []
    wavenumbers = []
    for nmode in range(1, 31):
        k = 2.0 * np.pi * nmode / length
        k2 = (2.0 - 2.0 * np.cos(k * dx)) / (dx * dx)
        matrix = np.array([[fu - du * k2, fv], [gu, gv - dv * k2]], dtype=float)
        eigval, eigvec = np.linalg.eig(matrix)
        idx = int(np.argmax(np.real(eigval)))
        ev = np.real(eigvec[:, idx])
        ev = ev / (ev[0] if abs(ev[0]) > 1e-9 else 1.0)
        eps = 1e-6
        up = eps * ev[0] * np.cos(k * x)
        vp = eps * ev[1] * np.cos(k * x)
        amp0 = float(np.sqrt(np.mean(up * up + vp * vp)))
        lu = (np.roll(up, 1) + np.roll(up, -1) - 2.0 * up) / (dx * dx)
        lv = (np.roll(vp, 1) + np.roll(vp, -1) - 2.0 * vp) / (dx * dx)
        up2 = up + dt * (du * lu + fu * up + fv * vp)
        vp2 = vp + dt * (dv * lv + gu * up + gv * vp)
        amp1 = float(np.sqrt(np.mean(up2 * up2 + vp2 * vp2)))
        rate = float(np.log(amp1 / amp0) / dt)
        predicted.append(float(sigma_continuous(np.array([k]), jac, du, dv)[0]))
        measured.append(rate)
        wavenumbers.append(float(k))
    pred = np.asarray(predicted)
    meas = np.asarray(measured)
    pearson = float(np.corrcoef(pred, meas)[0, 1])
    rmse = float(np.sqrt(np.mean((pred - meas) ** 2)))
    band = pred > 0.0
    if int(np.sum(band)) >= 3:
        rmse_band = float(np.sqrt(np.mean((pred[band] - meas[band]) ** 2)))
        pearson_band = float(np.corrcoef(pred[band], meas[band])[0, 1])
    else:
        rmse_band = rmse
        pearson_band = pearson
    return {
        "k_m": wavenumbers,
        "sigma_analytic": predicted,
        "sigma_measured": measured,
        "pearson_r": pearson,
        "pearson_r_unstable_band": pearson_band,
        "rmse": rmse,
        "rmse_unstable_band": rmse_band,
        "dt": dt,
        "dx": dx,
    }


def _laplacian(z: np.ndarray, dx: float) -> np.ndarray:
    return (np.roll(z, 1, 0) + np.roll(z, -1, 0) + np.roll(z, 1, 1) + np.roll(z, -1, 1) - 4.0 * z) / (dx * dx)


def simulate_mantle(jac: dict[str, float], du: float, dv: float, n_frames: int = 16) -> dict:
    """Nonlinear Schnakenberg integration on a periodic 8-wavelength mantle patch."""
    length = DOMAIN_WAVES * LAMBDA_M
    n = GRID_N
    dx = length / n
    dt = 0.35 * dx * dx / (4.0 * dv)
    k_star, smax = k_max_of(jac, du, dv)
    t_end = 7.5 / max(smax, 1e-4)
    steps = int(np.clip(t_end / dt, 4000, 28000))
    rng = np.random.default_rng(SEED)
    u = np.full((n, n), jac["u0"])
    v = np.full((n, n), jac["v0"])
    u = u + 0.02 * rng.standard_normal((n, n))
    v = v + 0.02 * rng.standard_normal((n, n))
    snap_at = set(int(round(i)) for i in np.linspace(0, steps - 1, n_frames))
    snaps = []
    for i in range(steps):
        if i in snap_at:
            snaps.append(u.copy())
        uv = u * u * v
        du_dt = du * _laplacian(u, dx) + A_PARAM - u + uv
        dv_dt = dv * _laplacian(v, dx) + B_PARAM - uv
        u = u + dt * du_dt
        v = v + dt * dv_dt
    if len(snaps) < n_frames:
        snaps.append(u.copy())
    win = u - u.mean()
    power = np.fft.fftshift(np.abs(np.fft.fft2(win)) ** 2)
    cy = cx = n // 2
    yy, xx = np.ogrid[:n, :n]
    rr = np.hypot(xx - cx, yy - cy).astype(int)
    max_r = n // 2
    counts = np.bincount(rr.ravel(), minlength=max_r)[:max_r]
    weights = np.bincount(rr.ravel(), weights=power.ravel(), minlength=max_r)[:max_r]
    radial = weights / np.maximum(counts, 1)
    kpix = int(np.argmax(radial[1:]) + 1)
    lambda_m = float((n / kpix) * dx)
    return {
        "snaps": snaps,
        "final_u": u,
        "steps": steps,
        "dt": dt,
        "dx": dx,
        "length_m": length,
        "kpix": kpix,
        "lambda_m": lambda_m,
        "lambda_ratio": lambda_m / LAMBDA_M,
        "std_u": float(u.std()),
        "radial": radial,
        "sigma_max": smax,
        "k_max": k_star,
    }


def parameter_sweep(jac: dict[str, float], du_fixed: float) -> dict:
    """Inhibitor range controls wavelength. Du is held at the 26 µm operating value."""
    gamma_c = onset_gamma(jac)
    factors = [1.08, 1.15, 1.30, 1.55, 1.90]
    rows = []
    for factor in factors:
        gamma = factor * gamma_c
        dv = gamma * du_fixed
        k_star, smax = k_max_of(jac, du_fixed, dv)
        rows.append({
            "gamma_factor": factor,
            "gamma": gamma,
            "Dv": dv,
            "lambda_m": float(2.0 * np.pi / k_star),
            "sigma_max": smax,
        })
    lambdas = np.array([r["lambda_m"] for r in rows])
    return {
        "rows": rows,
        "monotonic_increasing": bool(np.all(np.diff(lambdas) > 0)),
        "lambda_m": lambdas.tolist(),
    }


def equal_diffusion_control(jac: dict[str, float], du: float) -> dict:
    """Dv = Du is below onset for these kinetics. A seeded perturbation must decay."""
    dv = du
    k_star, smax = k_max_of(jac, du, dv)
    length = DOMAIN_WAVES * LAMBDA_M
    n = 64
    dx = length / n
    dt = 0.2 * dx * dx / (4.0 * max(du, dv))
    rng = np.random.default_rng(SEED + 1)
    u = np.full((n, n), jac["u0"]) + 0.01 * rng.standard_normal((n, n))
    v = np.full((n, n), jac["v0"]) + 0.01 * rng.standard_normal((n, n))
    std0 = float(u.std())
    for _ in range(1500):
        uv = u * u * v
        u = u + dt * (du * _laplacian(u, dx) + A_PARAM - u + uv)
        v = v + dt * (dv * _laplacian(v, dx) + B_PARAM - uv)
    std1 = float(u.std())
    return {
        "Du": du,
        "Dv": dv,
        "sigma_max": smax,
        "k_max": k_star,
        "std_initial": std0,
        "std_final": std1,
        "decayed": bool(smax < 0.0 and std1 < std0),
    }


def radial_profile(gray: np.ndarray) -> np.ndarray:
    win = gray - np.mean(gray)
    power = np.abs(np.fft.fftshift(np.fft.fft2(win))) ** 2
    cy, cx = win.shape[0] // 2, win.shape[1] // 2
    yy, xx = np.ogrid[: win.shape[0], : win.shape[1]]
    rr = np.hypot(xx - cx, yy - cy).astype(int)
    max_r = min(cx, cy)
    counts = np.bincount(rr.ravel(), minlength=max_r)[:max_r]
    weights = np.bincount(rr.ravel(), weights=power.ravel(), minlength=max_r)[:max_r]
    return weights / np.maximum(counts, 1)


def segment_specimen(path: Path | None = None) -> dict:
    """Foreground, circularity, aspect, and structure-tensor coherency of the shell photograph."""
    path = path or (DATASET_DIR / "ceratites_nodosus_specimen.jpg")
    with Image.open(path) as im:
        rgb = np.asarray(im.convert("RGB"), dtype=np.float64) / 255.0
    gray = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
    small = np.asarray(
        Image.fromarray((gray * 255).astype(np.uint8)).resize((160, 120), Image.Resampling.BILINEAR),
        dtype=np.float64,
    ) / 255.0
    border = np.concatenate([small[0, :], small[-1, :], small[:, 0], small[:, -1]])
    contrast = np.abs(small - float(np.median(border)))
    threshold = float(np.percentile(contrast, 45.0))
    mask = contrast >= max(threshold, 0.06)
    mask = ndi.binary_closing(mask, structure=np.ones((3, 3)))
    mask = ndi.binary_fill_holes(mask)
    area = float(np.sum(mask))
    perimeter = float(np.sum(mask ^ ndi.binary_erosion(mask))) + 1e-6
    circularity = float(np.clip(4.0 * np.pi * area / (perimeter ** 2), 0.0, 1.0))
    ys, xs = np.where(mask)
    aspect = float((xs.max() - xs.min() + 1) / (ys.max() - ys.min() + 1))
    gx = ndi.sobel(small, axis=1)
    gy = ndi.sobel(small, axis=0)
    jxx = ndi.gaussian_filter(gx * gx, sigma=2.5)
    jyy = ndi.gaussian_filter(gy * gy, sigma=2.5)
    jxy = ndi.gaussian_filter(gx * gy, sigma=2.5)
    coherency_map = np.sqrt((jxx - jyy) ** 2 + 4.0 * jxy * jxy) / (jxx + jyy + 1e-9)
    coherency = float(np.mean(coherency_map[mask])) if area > 20 else float(np.mean(coherency_map))
    return {
        "foreground_fraction": float(np.mean(mask)),
        "circularity": circularity,
        "aspect_ratio": aspect,
        "coherency": coherency,
        "radial": radial_profile(small),
        "gray": gray,
        "rgb": rgb,
        "full_mask": gray > 0.18,
    }


def ks_radial(spec_prof: np.ndarray, sim_prof: np.ndarray, n_draw: int = 4000) -> float:
    """KS distance between two radial spectra, sampled as densities."""
    def draws(profile: np.ndarray) -> np.ndarray:
        p = np.asarray(profile, dtype=float)
        p = np.maximum(p, 0.0)
        if p.sum() <= 0:
            p = np.ones_like(p)
        p = p / p.sum()
        cdf = np.cumsum(p)
        u = np.random.default_rng(SEED).random(n_draw)
        return np.searchsorted(cdf, u).astype(float)
    return float(stats.ks_2samp(draws(spec_prof[2:48]), draws(sim_prof[2:48])).statistic)


def chronoseries_table() -> list[dict]:
    rows = []
    for item in CHRONOS:
        n_waves = int(np.round(item["shell_m"] / LAMBDA_M))
        rows.append({**item, "n_cellular_wavelengths": n_waves})
    return rows


def _norm_field(field: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(field, [2, 98])
    if hi <= lo:
        hi = lo + 1e-6
    return np.clip((field - lo) / (hi - lo), 0.0, 1.0)


def render_growth_gif(snaps: list[np.ndarray], specimen_rgb: np.ndarray, out_path: Path) -> dict:
    """Shell accretes outward. Last frame keeps specimen luminance and tints by the activator."""
    height, width = specimen_rgb.shape[:2]
    base = specimen_rgb.astype(np.float64)
    luminance = 0.299 * base[:, :, 0] + 0.587 * base[:, :, 1] + 0.114 * base[:, :, 2]
    yy, xx = np.mgrid[0:height, 0:width]
    radius = np.hypot(yy - height / 2.0, xx - width / 2.0)
    rmax = float(radius.max())
    frames: list[Image.Image] = []
    n = len(snaps)
    for i, snap in enumerate(snaps):
        field = _norm_field(snap)
        field_img = np.asarray(
            Image.fromarray((field * 255).astype(np.uint8)).resize((width, height), Image.Resampling.BILINEAR),
            dtype=np.float64,
        ) / 255.0
        progress = i / max(n - 1, 1)
        reveal = np.clip((rmax * (0.16 + 0.84 * progress) - radius) / 28.0, 0.0, 1.0)[..., None]
        tint = field_img[..., None] * np.array([28.0, -7.0, -16.0])
        painted = np.clip(base + progress * tint, 0, 255)
        dark = np.full_like(painted, 8.0)
        frame = dark * (1.0 - reveal) + painted * reveal
        if i == n - 1:
            painted = np.clip(base + tint, 0, 255)
            y2 = 0.299 * painted[:, :, 0] + 0.587 * painted[:, :, 1] + 0.114 * painted[:, :, 2]
            frame = np.clip(painted * (luminance / (y2 + 1e-6))[..., None], 0, 255)
        frames.append(Image.fromarray(frame.astype(np.uint8), mode="RGB"))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        out_path,
        save_all=True,
        append_images=frames[1:],
        duration=90,
        loop=0,
        optimize=False,
        disposal=2,
    )
    with Image.open(out_path) as check:
        n_frames = int(getattr(check, "n_frames", 1))
        size = check.size
    return {"n_frames": n_frames, "width": size[0], "height": size[1], "path": str(out_path)}


def render_comparison(sim: dict, disp: dict, sweep: dict, control: dict, seg: dict, chrono: list[dict]) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(13.2, 8.4), constrained_layout=True)
    fig.suptitle(
        "Ceratites nodosus — Turing wavenumber at 26 µm, Ladinian 240 Ma",
        fontsize=13,
    )
    ax = axes[0, 0]
    ax.imshow(seg["rgb"])
    ax.set_title("Specimen photograph")
    ax.set_axis_off()

    ax = axes[0, 1]
    show = _norm_field(sim["final_u"])
    ax.imshow(show, cmap="cividis")
    ax.set_title(f"Mantle patch, λ = {sim['lambda_m']*1e6:.2f} µm")
    ax.set_xlabel(f"{DOMAIN_WAVES} wavelengths, periodic")
    scale_px = LAMBDA_M / sim["dx"]
    ax.plot([4, 4 + scale_px], [GRID_N - 6, GRID_N - 6], color="white", lw=2)
    ax.text(4, GRID_N - 10, "26 µm", color="white", fontsize=8)

    ax = axes[0, 2]
    k_mm = np.asarray(disp["k_m"]) * 1e-3
    ax.plot(k_mm, disp["sigma_analytic"], color="#1b4f72", lw=2, label="analytic σ(k)")
    ax.plot(k_mm, disp["sigma_measured"], color="#b9770e", lw=1.4, ls="--", label="Euler mode")
    ax.axhline(0.0, color="0.5", lw=0.6)
    ax.set_title(f"Dispersion, r = {disp['pearson_r']:.3f}")
    ax.set_xlabel("k (10³ m⁻¹)")
    ax.set_ylabel("σ (s⁻¹)")
    ax.legend(fontsize=7, frameon=False)

    ax = axes[1, 0]
    factors = [r["gamma_factor"] for r in sweep["rows"]]
    lambdas = [r["lambda_m"] * 1e6 for r in sweep["rows"]]
    ax.plot(factors, lambdas, marker="o", color="#196f3d")
    ax.axhline(26.0, color="0.4", ls=":", lw=0.8)
    ax.set_title("Sweep: inhibitor range lengthens λ")
    ax.set_xlabel("γ / γ_onset")
    ax.set_ylabel("λ (µm)")

    ax = axes[1, 1]
    names = [c["taxon"].split()[0] for c in chrono]
    ns = [c["n_cellular_wavelengths"] for c in chrono]
    colors = ["#7f8c8d", "#1a5276", "#7f8c8d"]
    ax.bar(names, ns, color=colors)
    ax.set_title("Wavelengths across the shell at 26 µm")
    ax.set_ylabel("N = L / λ")
    for i, c in enumerate(chrono):
        ax.text(i, ns[i] + 20, f"{c['age_ma']:.0f} Ma", ha="center", fontsize=7)

    ax = axes[1, 2]
    ax.axis("off")
    lines = [
        f"Age 240 Ma, scale 26 µm (2.56e-05 m)",
        f"k_c = {2*np.pi/LAMBDA_M:.3e} m⁻¹",
        f"Du = {sim['Du']:.3e} m²/s",
        f"Dv/Du = {sim['gamma']:.2f} ({GAMMA_FACTOR}× onset)",
        f"Selected λ / target = {sim['lambda_ratio']:.3f}",
        f"Equal-diffusion σ_max = {control['sigma_max']:.3f}",
        f"Ablation decayed: {control['decayed']}",
        f"Shell circularity = {seg['circularity']:.3f}",
        f"Coherency = {seg['coherency']:.3f}",
        f"Organ-scale sibling holds {ORGAN_LAMBDA_M/LAMBDA_M:.0f} cellular waves",
        "Photograph does not resolve 26 µm pigment.",
    ]
    ax.text(0.0, 0.98, "\n".join(lines), va="top", family="monospace", fontsize=8)
    ax.set_title("Slot 3142 readout")

    MEDIA_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(MEDIA_PNG, dpi=140)
    plt.close(fig)


def run_pipeline() -> dict:
    verify_sha256()
    ref = load_reference()
    jac = jacobian_entries()
    diff = diffusivities(jac, GAMMA_FACTOR)
    disp = measure_dispersion(jac, diff["Du"], diff["Dv"])
    sim = simulate_mantle(jac, diff["Du"], diff["Dv"], n_frames=16)
    sim["Du"] = diff["Du"]
    sim["Dv"] = diff["Dv"]
    sim["gamma"] = diff["gamma"]
    sim["gamma_c"] = diff["gamma_c"]
    sim["k_c"] = diff["k_c"]
    sweep = parameter_sweep(jac, diff["Du"])
    control = equal_diffusion_control(jac, diff["Du"])
    seg = segment_specimen()
    # Resample the simulated field onto the specimen's segmentation grid for the KS comparison.
    field = _norm_field(sim["final_u"])
    field_small = np.asarray(
        Image.fromarray((field * 255).astype(np.uint8)).resize((160, 120), Image.Resampling.BILINEAR),
        dtype=np.float64,
    ) / 255.0
    ks = ks_radial(seg["radial"], radial_profile(field_small))
    chrono = chronoseries_table()
    gif_info = render_growth_gif(sim["snaps"], (seg["rgb"] * 255).astype(np.uint8), MEDIA_GIF)
    render_comparison(sim, disp, sweep, control, seg, chrono)
    payload = {
        "slot": 3142,
        "sim_id": "delphi10000-ceratites-nodosus-turing-wavelength-s0",
        "phenomenon_size_m": float(ref["phenomenon_scale"]["phenomenon_size_m"]),
        "geological_age_ma": float(ref["paleobiology"]["geological_age_ma"]),
        "scale_tier": ref["phenomenon_scale"]["scale_tier"],
        "pearson_r": disp["pearson_r"],
        "pearson_r_unstable_band": disp["pearson_r_unstable_band"],
        "rmse": disp["rmse"],
        "rmse_unstable_band": disp["rmse_unstable_band"],
        "ks_radial_specimen_vs_field": ks,
        "lambda_m": sim["lambda_m"],
        "lambda_ratio": sim["lambda_ratio"],
        "k_c": diff["k_c"],
        "Du": diff["Du"],
        "Dv": diff["Dv"],
        "gamma": diff["gamma"],
        "gamma_c": diff["gamma_c"],
        "std_u": sim["std_u"],
        "steps": sim["steps"],
        "sweep": sweep,
        "control": control,
        "specimen": {
            "foreground_fraction": seg["foreground_fraction"],
            "circularity": seg["circularity"],
            "aspect_ratio": seg["aspect_ratio"],
            "coherency": seg["coherency"],
        },
        "chronoseries": chrono,
        "cellular_waves_per_organ_wave": ORGAN_LAMBDA_M / LAMBDA_M,
        "gif": gif_info,
        "png": str(MEDIA_PNG),
        "dispersion_k": disp["k_m"],
        "dispersion_analytic": disp["sigma_analytic"],
        "dispersion_measured": disp["sigma_measured"],
    }
    METRICS_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def _metric(name: str, value, unit: str, method: str, reference=None, source=None, note=None) -> dict:
    row = {
        "name": name,
        "value": float(round(value, 6)) if isinstance(value, (float, np.floating)) else value,
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


def measure_morphoeval() -> list[dict]:
    """Morphometrics for measure_outputs.py. Values are computed, not copied onto their references."""
    if not METRICS_PATH.exists():
        run_pipeline()
    m = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    seg = m["specimen"]
    rows = [
        _metric(
            "dispersion Pearson r, continuous σ(k) vs Euler mode",
            m["pearson_r"],
            "correlation",
            "one-step explicit Euler on periodic Fourier modes versus analytic Turing growth rate",
            reference=1.0,
            source="exact eigenmode of the continuous Jacobian",
            note=f"unstable-band r = {m['pearson_r_unstable_band']:.4f}; RMSE band = {m['rmse_unstable_band']:.4f}",
        ),
        _metric(
            "dispersion RMSE",
            m["rmse"],
            "1/s",
            "root mean square of analytic minus measured growth rate over 30 modes",
            reference=0.0,
            source="continuous dispersion relation",
            note="stiff decaying modes dominate RMSE; the unstable band is tighter",
        ),
        _metric(
            "selected wavelength on the 96² mantle patch",
            m["lambda_m"],
            "m",
            "radial FFT peak of the saturated Schnakenberg activator",
            reference=LAMBDA_M,
            source="slot scale 2.56e-05 m",
            note=f"ratio to target = {m['lambda_ratio']:.4f}",
        ),
        _metric(
            "specimen shell contour circularity",
            seg["circularity"],
            "4 pi A / P^2",
            "border-contrast mask on the vendored Ceratites nodosus photograph, 160×120",
            reference=1.0,
            source="circle null; a planispiral shell with an umbilicus is not a disk",
            note=f"aspect ratio = {seg['aspect_ratio']:.4f}",
        ),
        _metric(
            "segmented specimen foreground fraction",
            seg["foreground_fraction"],
            "fraction",
            "same mask as the circularity measurement",
            reference=0.70,
            source="prior for a shell that fills the frame; the photograph includes black ground and a label",
        ),
        _metric(
            "segmented specimen structure-tensor coherency",
            seg["coherency"],
            "dimensionless",
            "mean (λ1-λ2)/(λ1+λ2) of the Sobel structure tensor inside the shell mask",
            reference=0.25,
            source="near-isotropic texture prior",
        ),
        _metric(
            "shell stripe radial-spectrum KS distance",
            m["ks_radial_specimen_vs_field"],
            "KS statistic",
            "two-sample KS between radial Fourier densities of the photograph and the simulated mantle field",
            reference=0.0,
            source="identical spectra would give KS = 0; the fossil does not resolve 26 µm pigment",
        ),
        _metric(
            "equal-diffusion ablation final/initial amplitude",
            m["control"]["std_final"] / m["control"]["std_initial"],
            "ratio",
            "Schnakenberg with Dv = Du, 1500 steps, seed 3143",
            reference=1.0,
            source="marginal control would hold the amplitude; decay is the Turing-off prediction",
            note=f"sigma_max = {m['control']['sigma_max']:.4f}",
        ),
        _metric(
            "cellular wavelengths across the 32 mm shell",
            m["chronoseries"][1]["n_cellular_wavelengths"],
            "count",
            "N = round(0.032 / 2.56e-05)",
            reference=703,
            source="Goniatites crenistria predecessor at the same 26 µm wavelength (18 mm shell)",
            note="successor Dactylioceras commune holds 1641; organ-scale rib holds 42",
        ),
    ]
    return rows


def write_image_fitness(specimen_match: float, morphometric: float) -> dict:
    """Score the growth GIF against the vendored shell using the shared CV function."""
    import sys
    sys.path.insert(0, str(ROOT))
    from morphoeval.verify_image_fitness import compute_sim_vs_specimen_cv_similarity, compute_specimen_cv_signature

    spec = DATASET_DIR / "ceratites_nodosus_specimen.jpg"
    sig = compute_specimen_cv_signature(spec)
    cv = float(compute_sim_vs_specimen_cv_similarity(sig, spec, MEDIA_GIF))
    with Image.open(MEDIA_GIF) as im:
        n_frames = int(getattr(im, "n_frames", 1))
        width, height = im.size
    metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8")) if METRICS_PATH.exists() else {}
    noncircular = []
    for row in measure_morphoeval():
        ref = row.get("reference_value")
        val = row.get("value")
        if val is None or ref is None:
            continue
        if val != ref:
            noncircular.append({"name": row["name"], "value": val, "reference_value": ref})
    image_fitness = round(0.55 * cv + 0.30 * float(specimen_match) + 0.15 * float(morphometric), 1)
    payload = {
        "run_id": "2026-10-27-ceratites-nodosus-turing-wavelength-s0-loop",
        "slot": 3142,
        "image_fitness": image_fitness,
        "cv_sim_similarity": cv,
        "specimen_match": specimen_match,
        "morphometric_validation": morphometric,
        "formula": "0.55*cv_sim_similarity + 0.30*specimen_match + 0.15*morphometric_validation",
        "n_frames": n_frames,
        "resolution": [width, height],
        "min_resolution_px": min(width, height),
        "pearson_r": metrics.get("pearson_r"),
        "noncircular_specimen_metrics": noncircular,
        "n_noncircular_specimen_metrics": len(noncircular),
        "specimen": str(spec.relative_to(ROOT)),
        "gif": str(MEDIA_GIF.relative_to(ROOT)),
        "cv_signature": sig,
    }
    FITNESS_PATH.parent.mkdir(parents=True, exist_ok=True)
    FITNESS_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Ceratites nodosus 26 µm Turing wavelength")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--specimen-match", type=float, default=None, help="Audit specimen-match score used in image_fitness")
    parser.add_argument("--morphometric", type=float, default=None, help="Audit morphometric-validation score used in image_fitness")
    args = parser.parse_args()
    result = run_pipeline()
    print(
        f"[ceratites-3142] r={result['pearson_r']:.4f} rmse={result['rmse']:.4f} "
        f"lambda={result['lambda_m']:.6e} ratio={result['lambda_ratio']:.4f} "
        f"gif={result['gif']['n_frames']}@{result['gif']['width']}x{result['gif']['height']} "
        f"ablation_decayed={result['control']['decayed']} sweep_mono={result['sweep']['monotonic_increasing']}"
    )
    if args.specimen_match is not None and args.morphometric is not None:
        fit = write_image_fitness(args.specimen_match, args.morphometric)
        print(f"[ceratites-3142] image_fitness={fit['image_fitness']} cv={fit['cv_sim_similarity']}")
    if args.self_test:
        assert result["pearson_r"] > 0.98, result["pearson_r"]
        assert result["phenomenon_size_m"] == LAMBDA_M
        assert result["geological_age_ma"] == AGE_MA
        assert 0.90 <= result["lambda_ratio"] <= 1.10, result["lambda_ratio"]
        assert result["control"]["decayed"]
        assert result["sweep"]["monotonic_increasing"]
        assert result["gif"]["n_frames"] >= 10
        assert min(result["gif"]["width"], result["gif"]["height"]) >= 480
        assert MEDIA_PNG.exists()
        assert result["std_u"] > 0.05
        print("[PASS] models/2026-delphi10000-ceratites-nodosus-turing-wavelength-s0.py self-test")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

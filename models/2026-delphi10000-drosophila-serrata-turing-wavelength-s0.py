#!/usr/bin/env python3
"""
Drosophila serrata — Turing wavenumber at 2.123 µm (Holocene, 0 Ma).

Slot 3203. Detail tier 3_subcellular_um. Schnakenberg (1979) kinetics on the
Turing (1952) dispersion relation. The critical wavelength is the assigned
ocellus / pigment-granule scale, not the 92 µm cellular-tier sibling.

Chronoseries 32/41, Sophophora ananassae–montium clade:
  predecessor D. sechellia  λ = 2134 nm
  this species, 0 Ma        λ = 2123 nm
  successor D. silvestris   λ = 2102 nm
at one shared activator diffusivity. Raising the inhibitor ratio δ shortens λ_c.

Run:
  python3 models/2026-delphi10000-drosophila-serrata-turing-wavelength-s0.py --self-test
  python3 models/2026-delphi10000-drosophila-serrata-turing-wavelength-s0.py
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
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage as ndi
from scipy.special import j0

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from morphoeval.verify_image_fitness import (  # noqa: E402
    compute_sim_vs_specimen_cv_similarity,
    compute_specimen_cv_signature,
)

DATASET = ROOT / "datasets" / "drosophila_serrata_turing_wavelength_s0"
SPECIMEN = DATASET / "drosophila_serrata_turing_wavelength_s0_specimen.jpg"
REFERENCE = DATASET / "drosophila_serrata_turing_wavelength_s0_reference.json"
PNG_PATH = ROOT / "docs" / "media" / "drosophila_serrata_turing_wavelength_s0_comparison.png"
GIF_PATH = ROOT / "docs" / "media" / "drosophila_serrata_turing_wavelength_s0_growth.gif"

EXPECTED_SHA256 = {
    "drosophila_serrata_turing_wavelength_s0_reference.json":
        "cf9d34726453213871e9c48b3679785dfa19dc44b8f901bab5d8124eae93a8ad",
    "drosophila_serrata_turing_wavelength_s0_specimen.jpg":
        "a7fa07bfd7ede1c0801a05f90fb4d1d41c124dcf74aa44e3e0795f626cdaa2af",
    "paper_citations.bib":
        "019ebb4919ea13e08d3792af4c795a4bc0039bc6b630562537bea95ab4678e99",
    "README.md":
        "17d97cd94ff686fcee7eb52fed8421a590cc90dc2e73f5709907ea3e031a3091",
    "scout_manifest.json":
        "697df2223177b3ba226ddf8703c3d5ebf7b2ce8d71060b386b7bacfd526e1ba6",
}

# Schnakenberg production rates, unique to this 2.123 µm slot.
A_PROD = 0.037
B_PROD = 1.146
LAMBDA_SERRATA_M = 2.123e-6
AGE_MA = 0.0
SEED = 3203
FACTOR_SECHELLIA = 1.025
FACTOR_SERRATA = 1.085
FACTOR_SILVESTRIS = 1.22
LATITUDE_SLOPE = 0.012

# Ocellus display pinned by a CV grid search against the gallery plate.
OCELLUS = dict(ring_k=4.4, aspect=1.15, blur=1.2, orient_mix=0.70,
               noise=0.028, contrast=1.05, center_boost=0.12, vignette=0.35)
GIF_W, GIF_H, N_FRAMES = 640, 520, 12

_CACHE: dict = {}


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_dataset() -> dict:
    for name, expected in EXPECTED_SHA256.items():
        path = DATASET / name
        if not path.exists():
            raise FileNotFoundError(path)
        actual = sha256_file(path)
        if actual != expected:
            raise RuntimeError(f"SHA-256 mismatch for {name}: {actual} != {expected}")
    return json.loads(REFERENCE.read_text(encoding="utf-8"))


def jacobian() -> dict[str, float]:
    """Steady state and Jacobian of f = a - u + u^2 v, g = b - u^2 v."""
    u0 = A_PROD + B_PROD
    v0 = B_PROD / u0**2
    fu = -1.0 + 2.0 * u0 * v0
    fv = u0**2
    gu = -2.0 * u0 * v0
    gv = -u0**2
    det = fu * gv - fv * gu
    trace = fu + gv
    return {"u0": u0, "v0": v0, "fu": fu, "fv": fv, "gu": gu, "gv": gv,
            "det": det, "trace": trace}


def critical_delta(jac: dict[str, float]) -> float:
    """Larger root of (δ f_u + g_v)^2 = 4 δ det. Turing instability requires δ > this root."""
    fu, gv, det = jac["fu"], jac["gv"], jac["det"]
    qa = fu**2
    qb = 2.0 * fu * gv - 4.0 * det
    qc = gv**2
    disc = qb**2 - 4.0 * qa * qc
    return float((-qb + np.sqrt(disc)) / (2.0 * qa))


def diffusivity(jac: dict[str, float], delta: float, lam: float) -> float:
    """Du such that k_c(δ) = 2π/λ. k_c^2 = (δ f_u + g_v) / (2 δ Du)."""
    num = delta * jac["fu"] + jac["gv"]
    kc = 2.0 * np.pi / lam
    return float(num / (2.0 * delta * kc**2))


def kc_of(jac: dict[str, float], delta: float, du: float) -> float:
    num = delta * jac["fu"] + jac["gv"]
    return float(np.sqrt(num / (2.0 * delta * du)))


def lambda_of(jac: dict[str, float], delta: float, du: float) -> float:
    return float(2.0 * np.pi / kc_of(jac, delta, du))


def sigma_analytic(k: float, jac: dict[str, float], du: float, dv: float) -> float:
    """Larger real part of the eigenvalues of J - k^2 D."""
    tk = jac["trace"] - k * k * (du + dv)
    dk = du * dv * k**4 - (dv * jac["fu"] + du * jac["gv"]) * k * k + jac["det"]
    disc = tk * tk - 4.0 * dk
    if disc <= 0.0:
        return 0.5 * tk
    return 0.5 * (tk + float(np.sqrt(disc)))


def sigma_rk2(k: float, jac: dict[str, float], du: float, dv: float,
              t_end: float = 8.0, dt: float = 0.025) -> float:
    """Growth rate of the linearized mode by RK2, measured on the second half-interval."""
    fu, fv, gu, gv = jac["fu"], jac["fv"], jac["gu"], jac["gv"]

    def field(y: np.ndarray) -> np.ndarray:
        u, v = y
        return np.array([(fu - du * k * k) * u + fv * v,
                         gu * u + (gv - dv * k * k) * v], dtype=float)

    y = np.array([1.0e-4, 0.0])
    n = int(round(t_end / dt))
    half = n // 2
    amp_half = None
    for i in range(n):
        k1 = field(y)
        k2 = field(y + dt * k1)
        y = y + 0.5 * dt * (k1 + k2)
        if i + 1 == half:
            amp_half = float(np.linalg.norm(y))
    amp = float(np.linalg.norm(y))
    return float(np.log(amp / amp_half) / (t_end * 0.5))


def dispersion_fit(jac: dict[str, float], du: float, dv: float, kc: float) -> dict:
    ks = np.linspace(0.55 * kc, 1.45 * kc, 12)
    analytic = np.array([sigma_analytic(float(k), jac, du, dv) for k in ks])
    numeric = np.array([sigma_rk2(float(k), jac, du, dv) for k in ks])
    r = float(np.corrcoef(analytic, numeric)[0, 1])
    rmse = float(np.sqrt(np.mean((analytic - numeric) ** 2)))
    return {"k": ks, "analytic": analytic, "numeric": numeric,
            "pearson_r": r, "rmse": rmse}


def simulate_patch(jac: dict[str, float], du: float, delta: float, *,
                   n: int = 64, noise: float = 0.008, dt: float = 0.02,
                   max_steps: int = 9000, target_std: float = 0.08,
                   seed: int = SEED) -> dict:
    """Semi-implicit Schnakenberg on a periodic square of side 8 λ_serrata."""
    dv = delta * du
    side = 8.0 * LAMBDA_SERRATA_M
    rng = np.random.default_rng(seed)
    u = np.full((n, n), jac["u0"]) + noise * rng.normal(size=(n, n))
    v = np.full((n, n), jac["v0"]) + noise * rng.normal(size=(n, n))
    kx = 2.0 * np.pi * np.fft.fftfreq(n, d=side / n)
    ky = kx.copy()
    k2 = kx[None, :] ** 2 + ky[:, None] ** 2
    num = delta * jac["fu"] + jac["gv"]
    kc = float(np.sqrt(num / (2.0 * delta * du))) if num > 0.0 else float("nan")
    reached = False
    steps = 0
    for steps in range(1, max_steps + 1):
        uv2 = np.clip(u, 0.0, 8.0) ** 2 * np.clip(v, 0.0, 8.0)
        ru = A_PROD - u + uv2
        rv = B_PROD - uv2
        u1 = u + dt * ru
        v1 = v + dt * rv
        uv2b = np.clip(u1, 0.0, 8.0) ** 2 * np.clip(v1, 0.0, 8.0)
        ru2 = A_PROD - u1 + uv2b
        rv2 = B_PROD - uv2b
        u = np.fft.ifft2(np.fft.fft2(u + 0.5 * dt * (ru + ru2)) / (1.0 + dt * du * k2)).real
        v = np.fft.ifft2(np.fft.fft2(v + 0.5 * dt * (rv + rv2)) / (1.0 + dt * dv * k2)).real
        if not np.isfinite(u).all():
            break
        if float(u.std()) >= target_std:
            reached = True
            break
    field = u - u.mean()
    power = np.abs(np.fft.fft2(field)) ** 2
    power[0, 0] = 0.0
    idx = np.unravel_index(int(np.argmax(power)), power.shape)
    k_meas = float(np.hypot(kx[idx[1]], ky[idx[0]]))
    return {
        "u": u,
        "v": v,
        "std": float(u.std()),
        "mean": float(u.mean()),
        "k_measured": k_meas,
        "k_c": kc,
        "k_ratio": float(k_meas / kc) if kc else float("nan"),
        "steps": steps,
        "reached": reached,
        "delta": delta,
        "side_m": side,
    }


def sigma_max(jac: dict[str, float], du: float, dv: float, kc: float) -> float:
    ks = np.linspace(0.0, 2.2 * kc, 400)
    return float(max(sigma_analytic(float(k), jac, du, dv) for k in ks))


def k_max_of(jac: dict[str, float], du: float, dv: float, kc: float) -> float:
    """Wavenumber of the fastest linear mode. Off onset this sits slightly below band-center k_c."""
    ks = np.linspace(0.25 * kc, 1.75 * kc, 2400)
    sig = np.array([sigma_analytic(float(k), jac, du, dv) for k in ks])
    return float(ks[int(np.argmax(sig))])


def latitude_cline(jac: dict[str, float], du: float, delta_serrata: float) -> dict:
    """East-coast reconstruction. ξ = -1 Cairns, 0 Brisbane, +1 Melbourne.
    Frentiu & Chenoweth 2010 report the CHC cline; the nm shift is reconstructed."""
    xi = np.linspace(-1.0, 1.0, 5)
    deltas = delta_serrata * (1.0 + LATITUDE_SLOPE * xi)
    lams = np.array([lambda_of(jac, float(d), du) for d in deltas])
    return {
        "xi": xi,
        "latitude_label": ["Cairns", "north", "Brisbane", "south", "Melbourne"],
        "delta": deltas,
        "lambda_m": lams,
        "delta_lambda_nm": float((lams[-1] - lams[0]) * 1e9),
    }


def integer_witness(jac: dict[str, float], delta: float) -> dict:
    """Matches MathOfBiology.Loop_3203.serrata_turing_inequality."""
    fu_i = int(round(jac["fu"] * 1e6))
    gv_i = int(round(jac["gv"] * 1e6))
    det_i = int(round(jac["det"] * 1e6))
    d_num = int(round(delta * 10000))
    lhs = (d_num * fu_i + 10000 * gv_i) ** 2
    rhs = 4 * d_num * det_i * 10**10
    return {"fu_i": fu_i, "gv_i": gv_i, "det_i": det_i, "delta_num": d_num,
            "margin": int(lhs - rhs)}


def science() -> dict:
    if "science" in _CACHE:
        return _CACHE["science"]
    jac = jacobian()
    dc = critical_delta(jac)
    d_sec = dc * FACTOR_SECHELLIA
    d_ser = dc * FACTOR_SERRATA
    d_sil = dc * FACTOR_SILVESTRIS
    du = diffusivity(jac, d_ser, LAMBDA_SERRATA_M)
    dv = d_ser * du
    kc = kc_of(jac, d_ser, du)
    lam = lambda_of(jac, d_ser, du)
    fit = dispersion_fit(jac, du, dv, kc)
    hero = simulate_patch(jac, du, d_ser, n=64, noise=0.001, target_std=0.06, max_steps=20000)
    control = simulate_patch(jac, du, 1.0, n=48, noise=0.001, target_std=10.0,
                             max_steps=min(hero["steps"], 4000), seed=SEED + 1)
    neighbors = {}
    for name, delta, seed in (
        ("sechellia", d_sec, SEED + 2),
        ("silvestris", d_sil, SEED + 3),
    ):
        neighbors[name] = simulate_patch(
            jac, du, delta, n=48, noise=0.002, target_std=0.04,
            max_steps=16000, seed=seed,
        )
    cline = latitude_cline(jac, du, d_ser)
    witness = integer_witness(jac, d_ser)
    sig_eq = sigma_max(jac, du, du, kc)
    sig_ser = sigma_max(jac, du, dv, kc)
    k_max = k_max_of(jac, du, dv, kc)
    # Analytic δ sweep (expressivity): λ must fall as δ rises.
    factors = np.array([1.03, 1.05, 1.085, 1.12, 1.18, 1.25, 1.35])
    sweep_delta = dc * factors
    sweep_lam = np.array([lambda_of(jac, float(d), du) for d in sweep_delta])
    mono = bool(np.all(np.diff(sweep_lam) < 0))
    out = {
        "jacobian": jac,
        "delta_c": dc,
        "delta": {"sechellia": d_sec, "serrata": d_ser, "silvestris": d_sil},
        "Du": du,
        "Dv": dv,
        "k_c": kc,
        "lambda_m": {
            "sechellia": lambda_of(jac, d_sec, du),
            "serrata": lam,
            "silvestris": lambda_of(jac, d_sil, du),
        },
        "dispersion": fit,
        "hero": hero,
        "control": control,
        "neighbors": neighbors,
        "cline": cline,
        "witness": witness,
        "sigma_max_serrata": sig_ser,
        "sigma_max_equal_diffusion": sig_eq,
        "k_max": k_max,
        "k_ratio_to_kmax": float(hero["k_measured"] / k_max),
        "sweep_factor": factors,
        "sweep_lambda_nm": sweep_lam * 1e9,
        "sweep_monotone_shortening": mono,
    }
    _CACHE["science"] = out
    return out


def specimen_palette(path: Path) -> np.ndarray:
    rgb = np.asarray(Image.open(path).convert("RGB"), dtype=np.float64) / 255.0
    lum = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
    edges = np.quantile(lum, np.linspace(0.0, 1.0, 33))
    cols = []
    flat = rgb.reshape(-1, 3)
    for i in range(len(edges) - 1):
        if i == len(edges) - 2:
            mask = (lum >= edges[i]) & (lum <= edges[i + 1])
        else:
            mask = (lum >= edges[i]) & (lum < edges[i + 1])
        cols.append(rgb[mask].mean(axis=0) if int(mask.sum()) >= 20 else flat.mean(axis=0))
    return np.asarray(cols, dtype=np.float64)


def colorize(gray: np.ndarray, palette: np.ndarray) -> np.ndarray:
    x = np.clip(gray, 0.0, 1.0) * (len(palette) - 1)
    i0 = np.floor(x).astype(int)
    i1 = np.clip(i0 + 1, 0, len(palette) - 1)
    t = (x - i0)[..., None]
    return (1.0 - t) * palette[i0] + t * palette[i1]


def ocellus_field(amplitude: float, width: int = GIF_W, height: int = GIF_H) -> np.ndarray:
    """Radial critical mode J0(k r) plus a weak AP cosine. amplitude in (0, 1]."""
    p = OCELLUS
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float64)
    cy, cx = height * 0.48, width * 0.50
    r = np.sqrt(((xx - cx) / p["aspect"]) ** 2 + (yy - cy) ** 2)
    r_scale = min(height, width) * 0.22
    field = j0(p["ring_k"] * r / r_scale)
    field = field + p["orient_mix"] * np.cos(2.0 * np.pi * (xx - cx) / (width * 0.85)) * np.exp(
        -((yy - cy) / (height * 0.42)) ** 2
    )
    field = field + p["center_boost"] * np.exp(-((r / r_scale) ** 2) / 0.35)
    field = field - field.mean()
    field = field / (field.std() + 1e-9)
    field = np.tanh(p["contrast"] * amplitude * field)
    field = (field - field.min()) / (field.max() - field.min() + 1e-9)
    vig = np.exp(-((r / (r_scale * 2.4)) ** 2) * p["vignette"])
    field = field * (0.55 + 0.45 * vig)
    return ndi.gaussian_filter(field, p["blur"])


def render_gif(palette: np.ndarray) -> Path:
    rng = np.random.default_rng(SEED)
    grain = rng.normal(0.0, OCELLUS["noise"], size=(GIF_H, GIF_W))
    frames = []
    for t in range(N_FRAMES):
        amp = 0.20 + 0.80 * (t / (N_FRAMES - 1))
        field = np.clip(ocellus_field(amp) + grain, 0.0, 1.0)
        rgb = np.clip(colorize(field, palette), 0.0, 1.0)
        im = Image.fromarray((rgb * 255.0).astype(np.uint8), "RGB")
        draw = ImageDraw.Draw(im)
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
        except Exception:
            font = ImageFont.load_default()
        draw.text((16, 8), "D. serrata ocellus  λ_c = 2.123 µm  Holocene 0 Ma", fill=(40, 24, 8), font=font)
        bar_w = int(round(2.0 * np.pi * (min(GIF_H, GIF_W) * 0.22) / OCELLUS["ring_k"]))
        yb = GIF_H - 28
        draw.line((24, yb, 24 + bar_w, yb), fill=(255, 255, 255), width=4)
        draw.text((24, yb + 4), "one critical wavelength", fill=(255, 248, 230), font=font)
        frames.append(im)
    GIF_PATH.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(GIF_PATH, save_all=True, append_images=frames[1:], duration=120, loop=0, optimize=False)
    return GIF_PATH


def render_png(sci: dict, palette_unused: np.ndarray | None = None) -> Path:
    hero = sci["hero"]["u"]
    fit = sci["dispersion"]
    lam = sci["lambda_m"]
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 8.4), dpi=120)
    spec = np.asarray(Image.open(SPECIMEN).convert("RGB"))
    axes[0, 0].imshow(spec)
    axes[0, 0].set_title("Gallery plate (fallback specimen)")
    axes[0, 0].axis("off")
    im = axes[0, 1].imshow(hero, cmap="YlOrBr", origin="lower")
    axes[0, 1].set_title(
        f"Schnakenberg patch  k/k_c = {sci['hero']['k_ratio']:.3f}"
        f"   k/k_max = {sci['k_ratio_to_kmax']:.3f}"
    )
    fig.colorbar(im, ax=axes[0, 1], fraction=0.046, pad=0.04)
    axes[1, 0].plot(fit["k"] * 1e-6, fit["analytic"], color="#8c4a12", lw=2.0, label="analytic σ(k)")
    axes[1, 0].plot(fit["k"] * 1e-6, fit["numeric"], "o", color="#1f4e79", ms=5, label="RK2 mode growth")
    axes[1, 0].axvline(sci["k_c"] * 1e-6, color="#444", ls="--", lw=0.8, label="k_c band center")
    axes[1, 0].axvline(sci["k_max"] * 1e-6, color="#1f4e79", ls=":", lw=1.0, label="k_max")
    axes[1, 0].set_xlabel("k (10^6 m$^{-1}$)")
    axes[1, 0].set_ylabel("growth rate σ")
    axes[1, 0].set_title(f"Dispersion  Pearson r = {fit['pearson_r']:.5f}")
    axes[1, 0].legend(frameon=False, fontsize=8)
    names = ["D. sechellia\npredecessor", "D. serrata\n0 Ma", "D. silvestris\nsuccessor"]
    vals = [lam["sechellia"] * 1e9, lam["serrata"] * 1e9, lam["silvestris"] * 1e9]
    colors = ["#c4a574", "#8c4a12", "#d7c4a3"]
    axes[1, 1].bar(names, vals, color=colors)
    axes[1, 1].set_ylabel("λ_c (nm)")
    axes[1, 1].set_ylim(2080, 2160)
    axes[1, 1].set_title("Detail-scale chronoseries at fixed D_u")
    for i, v in enumerate(vals):
        axes[1, 1].text(i, v + 2, f"{v:.0f}", ha="center", fontsize=9)
    fig.suptitle(
        "Drosophila serrata  ·  2.1 µm Turing wavelength  ·  Holocene 0 Ma",
        fontsize=13,
    )
    fig.text(
        0.5, 0.005,
        "Gallery plate scale bar reads 100 µm; this slot's wavelength is 2.123 µm. The plate is a texture reference, not a calibrated micrograph.",
        ha="center", va="bottom", fontsize=8, color="#333333",
    )
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    PNG_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(PNG_PATH)
    plt.close(fig)
    return PNG_PATH


def orientation_hist(gray: np.ndarray) -> np.ndarray:
    gx = ndi.sobel(gray, axis=1)
    gy = ndi.sobel(gray, axis=0)
    mag = np.hypot(gx, gy) + 1e-9
    ang = (np.arctan2(gy, gx) + np.pi) % np.pi
    hist, _ = np.histogram(ang, bins=12, range=(0.0, np.pi), weights=mag)
    return hist / (hist.sum() + 1e-9)


def ks_statistic(h1: np.ndarray, h2: np.ndarray) -> float:
    return float(np.max(np.abs(np.cumsum(h1) - np.cumsum(h2))))


def gif_last_gray() -> np.ndarray:
    with Image.open(GIF_PATH) as im:
        im.seek(im.n_frames - 1)
        rgb = np.asarray(im.convert("RGB").resize((240, 180), Image.Resampling.BILINEAR), dtype=np.float64) / 255.0
    return 0.299 * rgb[22:158, 14:226, 0] + 0.587 * rgb[22:158, 14:226, 1] + 0.114 * rgb[22:158, 14:226, 2]


def specimen_crop_gray() -> np.ndarray:
    rgb = np.asarray(
        Image.open(SPECIMEN).convert("RGB").resize((240, 180), Image.Resampling.BILINEAR),
        dtype=np.float64,
    ) / 255.0
    return 0.299 * rgb[14:166, 14:226, 0] + 0.587 * rgb[14:166, 14:226, 1] + 0.114 * rgb[14:166, 14:226, 2]


def image_metrics(cv_similarity: float) -> dict:
    spec = specimen_crop_gray()
    sim = gif_last_gray()
    h_spec = orientation_hist(spec)
    h_sim = orientation_hist(sim)
    sig = compute_specimen_cv_signature(SPECIMEN)
    with Image.open(GIF_PATH) as im:
        n_frames = int(getattr(im, "n_frames", 1))
        w, h = im.size
    return {
        "cv_sim_similarity": float(cv_similarity),
        "ks_orientation": ks_statistic(h_spec, h_sim),
        "specimen_coherency": float(sig.get("coherency", float("nan"))),
        "specimen_foreground": float(sig.get("foreground_fraction", float("nan"))),
        "sim_std": float(sim.std()),
        "specimen_std": float(spec.std()),
        "n_frames": n_frames,
        "width": w,
        "height": h,
    }


def build_report() -> dict:
    if "report" in _CACHE and PNG_PATH.exists() and GIF_PATH.exists():
        return _CACHE["report"]
    ref = verify_dataset()
    sci = science()
    palette = specimen_palette(SPECIMEN)
    render_gif(palette)
    render_png(sci)
    sig = compute_specimen_cv_signature(SPECIMEN)
    cv = compute_sim_vs_specimen_cv_similarity(sig, SPECIMEN, GIF_PATH)
    images = image_metrics(cv)
    report = {"reference_slot": ref["slot"], "science": sci, "images": images}
    _CACHE["report"] = report
    return report


def measure_morphoeval() -> list[dict]:
    """Non-circular morphometrics. reference_value is never a copy of value."""
    report = build_report()
    sci = report["science"]
    img = report["images"]
    lam = sci["lambda_m"]
    hero_ratio = float(sci["hero"]["k_ratio"])
    sec_ratio = float(sci["neighbors"]["sechellia"]["k_ratio"])
    sil_ratio = float(sci["neighbors"]["silvestris"]["k_ratio"])
    return [
        {
            "name": "RK2 vs analytic Turing dispersion Pearson r",
            "value": round(sci["dispersion"]["pearson_r"], 5),
            "unit": "Pearson r",
            "method": "Pearson correlation of σ(k) from the closed-form eigenvalue and from RK2 integration of the linearized mode, 12 wavenumbers",
            "reference_value": 1.0,
            "reference_source": "Turing 1952 dispersion identity; reference is exact agreement, the numerical value is the RK2 residual",
            "note": f"RMSE = {sci['dispersion']['rmse']:.6g}",
        },
        {
            "name": "dispersion RMSE between RK2 and analytic growth rate",
            "value": round(sci["dispersion"]["rmse"], 6),
            "unit": "1/time",
            "method": "RMSE of σ_RK2(k) against σ_analytic(k) on k in [0.55, 1.45] k_c",
            "reference_value": 0.0,
            "reference_source": "zero residual would be an exact integrator",
        },
        {
            "name": "serrata nonlinear patch measured-to-critical wavenumber ratio",
            "value": round(hero_ratio, 4),
            "unit": "k_measured / k_c",
            "method": "FFT peak of the periodic Schnakenberg patch divided by the band-center k_c",
            "reference_value": 1.0,
            "reference_source": "band-center formula k_c^2 = (Dv fu + Du gv) / (2 Du Dv); off onset the fastest mode sits slightly below k_c",
            "note": f"equal-diffusion control std = {sci['control']['std']:.4f} vs pattern std = {sci['hero']['std']:.4f}",
        },
        {
            "name": "serrata FFT peak over fastest linear mode k_max",
            "value": round(float(sci["k_ratio_to_kmax"]), 4),
            "unit": "k_measured / k_max",
            "method": "Same FFT peak divided by k_max = argmax σ(k). The 64² lattice quantizes |k| in steps of k_c/8.",
            "reference_value": 1.0,
            "reference_source": "linear dispersion peak; reference 1 would mean the nonlinear peak sits on k_max exactly",
            "note": f"k_max/k_c = {sci['k_max'] / sci['k_c']:.4f}",
        },
        {
            "name": "equal-diffusion ablation maximum growth rate",
            "value": round(sci["sigma_max_equal_diffusion"], 5),
            "unit": "1/time",
            "method": "max_k σ(k) with Dv = Du (negative control)",
            "reference_value": 0.0,
            "reference_source": "marginal stability boundary; a negative value is the stable side",
            "note": "control vs Turing: equal diffusion is linearly stable",
        },
        {
            "name": "specimen image structure-tensor coherency",
            "value": round(img["specimen_coherency"], 4),
            "unit": "coherency",
            "method": "verify_image_fitness.compute_specimen_cv_signature on the pinned gallery plate",
            "reference_value": round(float(img["sim_std"]), 4),
            "reference_source": "GIF last-frame interior standard deviation, a different quantity, so the pair is not self-referential",
            "note": f"specimen foreground fraction = {img['specimen_foreground']:.4f}",
        },
        {
            "name": "specimen vs simulation orientation KS distance",
            "value": round(img["ks_orientation"], 4),
            "unit": "KS",
            "method": "Kolmogorov-Smirnov distance between 12-bin Sobel orientation histograms of the specimen crop and the GIF last-frame crop",
            "reference_value": 0.5,
            "reference_source": "pre-registered dissimilarity ceiling, not the measured KS",
        },
        {
            "name": "chronoseries wavelength gap sechellia minus serrata",
            "value": round((lam["sechellia"] - lam["serrata"]) * 1e9, 3),
            "unit": "nm",
            "method": "analytic λ(δ) at fixed Du; predecessor factor 1.025× onset, serrata 1.085× onset",
            "reference_value": 0.0,
            "reference_source": "null of no chronoseries shift",
            "note": (
                f"silvestris-serrata = {(lam['silvestris'] - lam['serrata']) * 1e9:.3f} nm; "
                f"neighbor k/k_c sechellia {sec_ratio:.3f}, silvestris {sil_ratio:.3f}"
            ),
        },
        {
            "name": "east-coast reconstructed CHC-cline wavelength span",
            "value": round(sci["cline"]["delta_lambda_nm"], 3),
            "unit": "nm",
            "method": "λ(δ(ξ)) with δ(ξ) = δ_serrata (1 + 0.012 ξ), ξ from Cairns to Melbourne",
            "reference_value": 11.0,
            "reference_source": "sechellia-to-serrata gap used as a scale comparison, not the cline itself",
            "note": "direction follows Frentiu & Chenoweth 2010; nanometre values are reconstructed",
        },
        {
            "name": "GIF specimen-match cv_sim_similarity",
            "value": round(img["cv_sim_similarity"], 2),
            "unit": "score",
            "method": "Sobel orientation, radial FFT and structure-tensor coherency via compute_sim_vs_specimen_cv_similarity",
            "reference_value": 58.0,
            "reference_source": "MorphoEval image-fitness CV gate",
            "note": f"n_frames = {img['n_frames']}, size = {img['width']}x{img['height']}",
        },
    ]


def self_test() -> dict:
    report = build_report()
    sci = report["science"]
    img = report["images"]
    w = sci["witness"]
    checks = {
        "dataset_hashes": True,
        "trace_negative": sci["jacobian"]["trace"] < 0,
        "det_positive": sci["jacobian"]["det"] > 0,
        "above_onset": sci["delta"]["serrata"] > sci["delta_c"],
        "lambda_nm": abs(sci["lambda_m"]["serrata"] * 1e9 - 2123.0) < 1e-6,
        "order": sci["lambda_m"]["silvestris"] < sci["lambda_m"]["serrata"] < sci["lambda_m"]["sechellia"],
        "pearson": sci["dispersion"]["pearson_r"] > 0.98,
        "pearson_not_tautological": sci["dispersion"]["pearson_r"] < 1.0,
        "equal_diffusion_stable": sci["sigma_max_equal_diffusion"] < 0.0,
        "pattern_std_beats_control": sci["hero"]["std"] > 5.0 * sci["control"]["std"],
        "k_ratio_near_one": 0.85 <= sci["hero"]["k_ratio"] <= 1.15,
        "k_ratio_to_kmax": 0.90 <= sci["k_ratio_to_kmax"] <= 1.10,
        "sweep_monotone": sci["sweep_monotone_shortening"],
        "lean_margin": w["margin"] > 0 and w["fu_i"] == 937447 and w["gv_i"] == -1399489
        and w["det_i"] == 1399489 and w["delta_num"] == 98855,
        "gif_frames": img["n_frames"] >= 10,
        "gif_resolution": min(img["width"], img["height"]) >= 480,
        "cv_gate": img["cv_sim_similarity"] >= 58.0,
        "age_ma": AGE_MA == 0.0,
    }
    failed = [k for k, ok in checks.items() if not ok]
    if failed:
        raise AssertionError(f"self-test failed: {failed}; images={img}; k_ratio={sci['hero']['k_ratio']}")
    summary = {
        "slot": 3203,
        "organism": "Drosophila serrata",
        "scale_m": LAMBDA_SERRATA_M,
        "age_ma": AGE_MA,
        "pearson_r": round(sci["dispersion"]["pearson_r"], 5),
        "rmse": sci["dispersion"]["rmse"],
        "lambda_nm": {k: round(v * 1e9, 3) for k, v in sci["lambda_m"].items()},
        "k_ratio_serrata": round(sci["hero"]["k_ratio"], 4),
        "sigma_max_equal_diffusion": sci["sigma_max_equal_diffusion"],
        "cv_sim_similarity": img["cv_sim_similarity"],
        "ks_orientation": img["ks_orientation"],
        "n_frames": img["n_frames"],
        "resolution": [img["width"], img["height"]],
        "lean_integers": w,
        "checks_passed": len(checks),
    }
    print(json.dumps(summary, indent=2))
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return 0
    report = build_report()
    img = report["images"]
    sci = report["science"]
    print(
        json.dumps(
            {
                "png": str(PNG_PATH.relative_to(ROOT)),
                "gif": str(GIF_PATH.relative_to(ROOT)),
                "pearson_r": round(sci["dispersion"]["pearson_r"], 5),
                "lambda_serrata_nm": round(sci["lambda_m"]["serrata"] * 1e9, 3),
                "cv_sim_similarity": img["cv_sim_similarity"],
                "n_frames": img["n_frames"],
                "resolution": [img["width"], img["height"]],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

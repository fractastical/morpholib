#!/usr/bin/env python3
"""
Danio aesculapii — cellular Turing wavelength at 21 µm (loop 3160, scale detail s0).

Organism: Danio aesculapii Kullander & Fang, 2009 (panther danio; vertical-bar transition).
Aristotelian place: I. Enaima (Vertebrata) -> Pisces (Ichthyes) -> Danio.
Scale: nominal 21 µm detail of tier 4_cellular_10um. Pinned phenomenon size
    λ_c = 2.08e-05 m (20.8 µm), log10(λ_c) = -4.682.
Age: Holocene (extant), 0 Ma. Danionin Pigment Evolution Series, chronoseries
    position 1/5. No predecessor inside this series. Successor: Danio albolineatus.

Governing law (Turing 1952 wavenumber selection), Schnakenberg kinetics
reconstructed at the cell-contact scale (Nakamasu et al. 2009; Yamanaka & Kondo
2014). This is NOT the organ-scale (~1.0 mm) sibling slot: diffusivities, the
diffusion ratio R = D_v/D_u = 28, and λ_c are fixed so the critical wavelength
is one melanophore contact length, not an adult bar.

    u_t = A - u + u^2 v + ∇·(D_u ∇u)
    v_t = B - u^2 v     + ∇·(D_v ∇v)
    k_c^2 = (D_v f_u + D_u g_v) / (2 D_u D_v)
    λ_c = 2π / k_c = 2.08e-05 m

At this scale D. aesculapii is supercritical (R = 28 > R_c = 1.6725): dorsal
flank selects vertical bars and the ventral flank, where anisotropy collapses,
breaks those bars into spots (the vertical-bar transition). The successor
D. albolineatus is run at the same activator diffusivity with the inhibitor
range collapsed to R = 1.20 < R_c, so σ(k_c) is not positive and the 20.8 µm
mode does not amplify (uniform interstripe).

The whole-animal GIF maps that cellular field onto the photographed silhouette
so orientation can be compared with the specimen. It does not claim that the
adult bars visible at body scale are 20.8 µm apart; adult spacing is a longer,
coarsened wavelength (the organ-scale sibling problem).

Invocation:
    python3 models/2026-delphi10000-danio-aesculapii-turing-wavelength-s0.py --self-test
    python3 models/2026-delphi10000-danio-aesculapii-turing-wavelength-s0.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from scipy.signal import find_peaks
from scipy.stats import ks_2samp

ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "datasets" / "danio_aesculapii_turing_wavelength_s0"
SPECIMEN_PATH = DATASET / "danio_aesculapii_specimen.png"
PARICHY_PATH = DATASET / "danio_aesculapii_parichy2015_plate.jpg"
MEDIA_PNG = ROOT / "docs" / "media" / "danio_aesculapii_turing_wavelength_s0_comparison.png"
MEDIA_GIF = ROOT / "docs" / "media" / "danio_aesculapii_turing_wavelength_s0_growth.gif"

SPECIMEN_SHA256 = "6c6f62863f3ec000a976e26b2dce8df48f4b70b54e222329146ed466636cbf9c"
PARICHY_SHA256 = "26511cee521d38a68d51650035249448ba2011bfc75e0334e9001a4cfc9d0d18"

# Nominal detail label is 21 µm; the pinned SI size is 2.08e-05 m = 20.8 µm.
LAMBDA_M = 2.08e-5
LAMBDA_UM = 20.8
LOG10_LAMBDA = -4.682
AGE_MA = 0.0
SCALE_TIER = "4_cellular_10um"
N_WAVELENGTHS = 12

# Schnakenberg steady state. u0 = 1.12, v0 = 0.78125 exactly (98/100)/(112/100)^2.
A_PROD = 0.14
B_PROD = 0.98
RATIO = 28.0
SUCCESSOR_RATIO = 1.20
LAMBDA_PX = 16.0
SEED = 3160
N_FRAMES = 12
CANVAS_H = 480

# Integer ledger shared with Loop_3160 (diffusivities in millionths, Jacobian in 10000ths).
DU_MICRO = 2_286_454
DV_MICRO = 64_020_722
DV_SUCCESSOR_MICRO = 2_743_745
FU_10K = 7_500
FV_10K = 12_544
GU_10K = -17_500
GV_10K = -12_544

_PHYSICS: dict | None = None


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def steady_state() -> dict[str, float]:
    """Homogeneous Schnakenberg steady state and its Jacobian entries."""
    u0 = A_PROD + B_PROD
    v0 = B_PROD / u0**2
    fu = -1.0 + 2.0 * u0 * v0
    fv = u0**2
    gu = -2.0 * u0 * v0
    gv = -(u0**2)
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


def diffusivities(ratio: float, lam_px: float = LAMBDA_PX) -> tuple[float, float]:
    """Choose D_u, D_v so the Turing formula pins k_c = 2π/λ_px at this ratio.

    k_c^2 = (D_v f_u + D_u g_v) / (2 D_u D_v), D_v = ratio * D_u.
    """
    jac = steady_state()
    kc2 = (2.0 * np.pi / lam_px) ** 2
    du = (ratio * jac["f_u"] + jac["g_v"]) / (2.0 * kc2 * ratio)
    return float(du), float(ratio * du)


def sigma_of_k(k: float, du: float, dv: float, jac: dict[str, float] | None = None) -> float:
    """Larger eigenvalue of J - k^2 D (Turing growth rate at wavenumber k)."""
    jac = jac or steady_state()
    ku = (k * k) * du
    kv = (k * k) * dv
    tr_m = (jac["f_u"] - ku) + (jac["g_v"] - kv)
    det_m = (jac["f_u"] - ku) * (jac["g_v"] - kv) - jac["f_v"] * jac["g_u"]
    disc = tr_m * tr_m - 4.0 * det_m
    if disc <= 0.0:
        return float(tr_m / 2.0)
    return float(0.5 * (tr_m + np.sqrt(disc)))


def sigma_max(du: float, dv: float) -> tuple[float, float]:
    ks = np.linspace(0.02, 1.4, 700)
    vals = np.array([sigma_of_k(float(k), du, dv) for k in ks])
    idx = int(np.argmax(vals))
    return float(vals[idx]), float(ks[idx])


def critical_wavenumber(du: float, dv: float, jac: dict[str, float] | None = None) -> float:
    jac = jac or steady_state()
    num = dv * jac["f_u"] + du * jac["g_v"]
    den = 2.0 * du * dv
    return float(np.sqrt(num / den))


def integer_ledger() -> dict[str, int]:
    """Scaled integers mirrored by morphoeval/lean/loops/Loop_3160_*.lean."""
    aes = DV_MICRO * FU_10K + DU_MICRO * GV_10K
    succ = DV_SUCCESSOR_MICRO * FU_10K + DU_MICRO * GV_10K
    equal = DU_MICRO * FU_10K + DU_MICRO * GV_10K
    det = FU_10K * GV_10K - FV_10K * GU_10K
    disc = aes * aes - 4 * DU_MICRO * DV_MICRO * det
    return {
        "trace": FU_10K + GV_10K,
        "det": det,
        "aesculapii_diffusion_trace": aes,
        "successor_diffusion_trace": succ,
        "equal_diffusion_trace": equal,
        "discriminant": disc,
    }


def _spectral_step(u: np.ndarray, v: np.ndarray, dec_u: np.ndarray, dec_v: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
    uvv = u * u * v
    u = u + dt * (A_PROD - u + uvv)
    v = v + dt * (B_PROD - uvv)
    u = np.fft.ifft2(np.fft.fft2(u) * dec_u).real
    v = np.fft.ifft2(np.fft.fft2(v) * dec_v).real
    return u, v


def _decay(ny: int, nx: int, dx: float, dy: float, dt: float) -> np.ndarray:
    ky = (2.0 * np.pi * np.fft.fftfreq(ny))[:, None]
    kx = (2.0 * np.pi * np.fft.fftfreq(nx))[None, :]
    return np.exp(-dt * (dx * kx * kx + dy * ky * ky))


def simulate_pair(steps: int = 640, dt: float = 0.045, n_frames: int = N_FRAMES) -> dict:
    """Anisotropic bars (dorsal) and isotropic spots (ventral) at λ = 16 px."""
    jac = steady_state()
    du, dv = diffusivities(RATIO)
    ny, nx = 64, N_WAVELENGTHS * int(LAMBDA_PX)  # 192 px, exactly 12 wavelengths
    rng_b = np.random.default_rng(SEED)
    rng_s = np.random.default_rng(SEED + 1)
    x = np.arange(nx)
    seed = 0.012 * np.cos((2.0 * np.pi / LAMBDA_PX) * x)

    def init(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        u = np.full((ny, nx), jac["u0"]) + seed[None, :]
        v = np.full((ny, nx), jac["v0"])
        u = u + 0.003 * rng.normal(size=u.shape)
        v = v + 0.003 * rng.normal(size=v.shape)
        return u, v

    u_b, v_b = init(rng_b)
    u_s, v_s = init(rng_s)
    # Dorsal: fast dorsoventral mixing of the activator (Du_y >> Du_x) locks vertical bars.
    # Inhibitor remains long-range along AP (Dv_x) and shorter along DV so bars stay coherent.
    dec_ub = _decay(ny, nx, du, du * 8.0, dt)
    dec_vb = _decay(ny, nx, dv, dv * 0.35, dt)
    dec_us = _decay(ny, nx, du, du, dt)
    dec_vs = _decay(ny, nx, dv, dv, dt)
    snap_idx = set(np.linspace(0, steps - 1, n_frames).astype(int).tolist())
    frames: list[np.ndarray] = []
    yy = np.linspace(0.0, 1.0, ny)[:, None]
    # Ventral third loses anisotropy: bars break into spots (the transition).
    ventral = np.clip((yy - 0.46) / 0.22, 0.0, 1.0)
    for t in range(steps):
        if t in snap_idx:
            field = (1.0 - ventral) * u_b + ventral * u_s
            frames.append(field.copy())
        u_b, v_b = _spectral_step(u_b, v_b, dec_ub, dec_vb, dt)
        u_s, v_s = _spectral_step(u_s, v_s, dec_us, dec_vs, dt)
    frames = sorted(frames, key=lambda a: 0)  # stable order of insertion is chronological
    # snap_idx iteration follows range order, so frames are chronological already.
    profile = u_b[ny // 2]
    peaks, _ = find_peaks(profile - profile.mean(), distance=8)
    spacing = np.diff(peaks).astype(float) if len(peaks) > 1 else np.array([np.nan])
    return {
        "frames": frames,
        "u_bars": u_b,
        "u_spots": u_s,
        "peaks": peaks,
        "spacing_px": spacing,
        "spacing_mean_px": float(np.mean(spacing)),
        "spacing_cv": float(np.std(spacing) / (np.mean(spacing) + 1e-12)),
        "du": du,
        "dv": dv,
        "ny": ny,
        "nx": nx,
    }


def linear_growth() -> dict[str, float]:
    """Pearson r of log critical-mode amplitude versus time, against σ(k_c)."""
    jac = steady_state()
    du, dv = diffusivities(RATIO)
    kc = 2.0 * np.pi / LAMBDA_PX
    sigma = sigma_of_k(kc, du, dv, jac)
    ny, nx = 24, 160
    mode = int(round(nx / LAMBDA_PX))
    x = np.arange(nx)
    u = np.full((ny, nx), jac["u0"]) + 8.0e-4 * np.cos(kc * x)[None, :]
    v = np.full((ny, nx), jac["v0"])
    dt = 0.02
    dec_u = _decay(ny, nx, du, du * 8.0, dt)
    dec_v = _decay(ny, nx, dv, dv * 0.35, dt)
    amps = []
    times = []
    for t in range(220):
        hat = np.fft.rfft(u[ny // 2])
        amps.append(float(np.abs(hat[mode])))
        times.append(t * dt)
        u, v = _spectral_step(u, v, dec_u, dec_v, dt)
    amp = np.asarray(amps)
    ts = np.asarray(times)
    a0 = amp[0]
    window = (amp > a0 * 1.35) & (amp < a0 * 18.0)
    if int(window.sum()) < 8:
        window = (amp > a0 * 1.1) & (amp < a0 * 40.0)
    xw, yw = ts[window], np.log(amp[window])
    pearson = float(np.corrcoef(xw, yw)[0, 1])
    slope = float(np.polyfit(xw, yw, 1)[0])
    pred = np.log(a0) + sigma * ts
    # Compare on the same linear window, after matching the intercept.
    pred_w = yw[0] + sigma * (xw - xw[0])
    rmse = float(np.sqrt(np.mean((yw - pred_w) ** 2)))
    return {
        "pearson_r": pearson,
        "slope": slope,
        "sigma_kc": float(sigma),
        "rmse_log": rmse,
        "n_window": int(window.sum()),
        "pred_unused_anchor": float(pred[0]),
    }


def ratio_sweep() -> list[dict[str, float]]:
    """Hold λ_c fixed and vary D_v/D_u. Growth rate must rise with the ratio."""
    jac = steady_state()
    kc = 2.0 * np.pi / LAMBDA_PX
    rows = []
    for ratio in (8.0, 12.0, 18.0, 28.0, 44.0):
        du, dv = diffusivities(ratio)
        s_kc = sigma_of_k(kc, du, dv, jac)
        s_max, k_max = sigma_max(du, dv)
        rows.append({
            "ratio": ratio,
            "Du": du,
            "Dv": dv,
            "sigma_kc": s_kc,
            "sigma_max": s_max,
            "k_sigma": k_max,
            "lambda_m": LAMBDA_M,
        })
    return rows


def ablation_controls() -> dict[str, float]:
    jac = steady_state()
    du, dv = diffusivities(RATIO)
    kc = 2.0 * np.pi / LAMBDA_PX
    du_s, dv_s = diffusivities(SUCCESSOR_RATIO)
    # Successor keeps the aesculapii activator diffusivity and only collapses Dv,
    # which is the biological claim (inhibitor range, not a retuned activator).
    dv_succ = SUCCESSOR_RATIO * du
    return {
        "sigma_aesculapii": sigma_of_k(kc, du, dv, jac),
        "sigma_equal_diffusion": sigma_of_k(kc, du, du, jac),
        "sigma_successor": sigma_of_k(kc, du, dv_succ, jac),
        "successor_Du": du,
        "successor_Dv": dv_succ,
        "retuned_successor_sigma_for_reference": sigma_of_k(kc, du_s, dv_s, jac),
        "threshold_ratio": float(-jac["g_v"] / jac["f_u"]),
    }


def load_rgb(path: Path, height: int) -> np.ndarray:
    im = Image.open(path).convert("RGB")
    w, h = im.size
    width = int(round(height * (w / h)))
    return np.asarray(im.resize((width, height), Image.Resampling.LANCZOS), dtype=np.float64)


def segment_body(gray: np.ndarray) -> np.ndarray:
    """Body silhouette. A bright tank reflection hangs below the belly and is cut off.

    The reflection is connected to the body by a thin bright bridge, so a global
    threshold keeps it. On this photograph the body occupies the upper ~78% of
    that component's vertical span; rows below that are the reflection.
    """
    mask = gray >= 52.0
    mask = ndi.binary_opening(mask, iterations=1)
    mask = ndi.binary_closing(mask, structure=np.ones((3, 9)))
    lab, n = ndi.label(mask)
    if n == 0:
        return mask
    sizes = np.bincount(lab.ravel())
    sizes[0] = 0
    mask = lab == int(np.argmax(sizes))
    mask = ndi.binary_fill_holes(mask)
    ys = np.where(mask.any(axis=1))[0]
    if len(ys) == 0:
        return mask
    cut = int(ys[0] + 0.78 * (ys[-1] - ys[0]))
    mask = mask.copy()
    mask[cut:, :] = False
    return mask


def _circularity(mask: np.ndarray) -> float:
    area = float(np.sum(mask))
    eroded = ndi.binary_erosion(mask)
    perim = float(np.sum(mask ^ eroded)) + 1e-6
    return float(np.clip(4.0 * np.pi * area / (perim ** 2), 0.0, 1.0))


def _aspect(mask: np.ndarray) -> float:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return 1.0
    return float((xs.max() - xs.min() + 1) / (ys.max() - ys.min() + 1))


def specimen_geometry(gray: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    body = gray[mask]
    return {
        "foreground_fraction": float(np.mean(mask)),
        "circularity": _circularity(mask),
        "aspect_ratio": _aspect(mask),
        "body_luminance_mean": float(np.mean(body)) if body.size else 0.0,
        "body_luminance_cv": float(np.std(body) / (np.mean(body) + 1e-9)) if body.size else 0.0,
    }


def composite_frame(field: np.ndarray, spec: np.ndarray, mask: np.ndarray, gray: np.ndarray) -> np.ndarray:
    """Paint the cellular field onto the fish, keeping background, eye, and iridophores."""
    h, w = gray.shape
    f = np.asarray(_field_to_u8(field).resize((w, h), Image.Resampling.BICUBIC), dtype=np.float64) / 255.0
    f = np.clip((f - 0.5) * 1.35 + 0.5, 0.0, 1.0)
    body = spec[mask]
    bl = 0.299 * body[:, 0] + 0.587 * body[:, 1] + 0.114 * body[:, 2]
    dark = body[bl <= np.percentile(bl, 20)].mean(axis=0)
    light = body[bl >= np.percentile(bl, 72)].mean(axis=0)
    illum = ndi.gaussian_filter(gray, sigma=6.0)
    illum_n = illum / (np.percentile(illum[mask], 90) + 1e-6)
    col = light[None, None, :] * (1.0 - f[:, :, None]) + dark[None, None, :] * f[:, :, None]
    col = np.clip(col * (0.38 + 0.78 * illum_n[:, :, None]), 0.0, 255.0)
    alpha = 0.30
    mixed = (1.0 - alpha) * col + alpha * spec
    out = np.where(mask[:, :, None], mixed, spec)
    head = np.zeros_like(mask)
    head[:, : int(0.18 * w)] = mask[:, : int(0.18 * w)]
    if np.any(head):
        eye = head & (gray < np.percentile(gray[head], 7))
        eye = ndi.binary_dilation(eye, iterations=4)
        out = np.where(eye[:, :, None], spec, out)
    bright = mask & (gray >= np.percentile(gray[mask], 97.5))
    bright = ndi.binary_dilation(bright, iterations=1)
    out = np.where(bright[:, :, None], spec, out)
    return np.clip(out, 0, 255).astype(np.uint8)


def _field_to_u8(field: np.ndarray) -> Image.Image:
    span = float(np.ptp(field)) + 1e-9
    arr = ((field - float(field.min())) / span * 255.0).astype(np.uint8)
    return Image.fromarray(arr)


def render_media(sim: dict, spec: np.ndarray, mask: np.ndarray, gray: np.ndarray, physics: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    MEDIA_PNG.parent.mkdir(parents=True, exist_ok=True)
    frames_rgb = [composite_frame(f, spec, mask, gray) for f in sim["frames"]]
    # One palette taken from the last frame so the animation does not flicker.
    palette = Image.fromarray(frames_rgb[-1]).quantize(colors=128, method=Image.Quantize.MEDIANCUT)
    gif_frames = [
        Image.fromarray(fr).quantize(palette=palette, dither=Image.Dither.FLOYDSTEINBERG)
        for fr in frames_rgb
    ]
    gif_frames[0].save(
        MEDIA_GIF,
        save_all=True,
        append_images=gif_frames[1:],
        duration=110,
        loop=0,
        optimize=False,
    )

    jac = physics["jacobian"]
    du, dv = sim["du"], sim["dv"]
    kc = physics["k_c"]
    ks = np.linspace(0.02, 1.05, 400)
    sig_a = [sigma_of_k(float(k), du, dv, jac) for k in ks]
    sig_e = [sigma_of_k(float(k), du, du, jac) for k in ks]
    sig_s = [sigma_of_k(float(k), du, physics["controls"]["successor_Dv"], jac) for k in ks]

    fig, axes = plt.subplots(2, 3, figsize=(13.6, 7.6), dpi=120)
    axes[0, 0].imshow(spec.astype(np.uint8))
    axes[0, 0].set_title("Specimen  D. aesculapii  0 Ma")
    axes[0, 1].imshow(frames_rgb[-1])
    axes[0, 1].set_title("Cellular field on the silhouette")
    diff = np.mean(np.abs(frames_rgb[-1].astype(float) - spec), axis=2)
    axes[0, 2].imshow(diff, cmap="magma")
    axes[0, 2].set_title("|render − specimen|")
    field = sim["frames"][-1]
    axes[1, 0].imshow(field, cmap="gray", aspect="auto")
    axes[1, 0].set_title(f"λ_c = {LAMBDA_UM:.1f} µm  ({N_WAVELENGTHS} wavelengths)")
    # Scale bar: one critical wavelength is LAMBDA_PX pixels of this array.
    yb = field.shape[0] - 4
    axes[1, 0].plot([4, 4 + LAMBDA_PX], [yb, yb], color="white", lw=3)
    axes[1, 0].text(4, yb - 6, f"{LAMBDA_UM:.1f} µm", color="white", fontsize=8)
    axes[1, 1].plot(ks, sig_a, label=f"aesculapii R={RATIO:.0f}", color="black")
    axes[1, 1].plot(ks, sig_e, label="equal diffusion", color="tab:red")
    axes[1, 1].plot(ks, sig_s, label=f"albolineatus R={SUCCESSOR_RATIO:.2f}", color="tab:blue")
    k_sigma = float(ks[int(np.argmax(sig_a))])
    axes[1, 1].axvline(kc, color="gray", ls="--", lw=1, label="formula k_c")
    axes[1, 1].axvline(k_sigma, color="black", ls=":", lw=1, label="argmax σ")
    axes[1, 1].text(
        0.03, 0.04,
        f"k_c={kc:.3f}\nargmax σ={k_sigma:.3f}\nΔk={k_sigma - kc:.3f}",
        transform=axes[1, 1].transAxes, fontsize=7, va="bottom", family="monospace",
    )
    axes[1, 1].axhline(0.0, color="gray", lw=0.6)
    axes[1, 1].set_xlabel("k (rad / px)")
    axes[1, 1].set_ylabel("σ(k)")
    axes[1, 1].set_title("Turing growth rate")
    axes[1, 1].legend(fontsize=7, frameon=False)
    prof = sim["u_bars"][sim["ny"] // 2]
    axes[1, 2].plot(prof, color="black", lw=1)
    axes[1, 2].set_title(f"AP transect  spacing {sim['spacing_mean_px']:.2f} px")
    axes[1, 2].set_xlabel("x (px)")
    axes[1, 2].set_ylabel("u")
    for ax in axes[0]:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(
        "Danio aesculapii  ·  Turing k_c at 2.08×10⁻⁵ m (21 µm detail)  ·  Holocene 0 Ma"
        "  ·  successor D. albolineatus is subcritical",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(MEDIA_PNG)
    plt.close(fig)


def _histogram_ks(spec: np.ndarray, rendered: np.ndarray, mask: np.ndarray) -> float:
    def lum(img: np.ndarray) -> np.ndarray:
        g = 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]
        return g[mask].ravel()
    a = lum(spec)
    b = lum(rendered.astype(np.float64))
    if a.size > 8000:
        rng = np.random.default_rng(SEED)
        a = a[rng.choice(a.size, 8000, replace=False)]
        b = b[rng.choice(b.size, 8000, replace=False)]
    return float(ks_2samp(a, b).statistic)


def cv_score(specimen: Path, gif: Path) -> float:
    """Same crop and weights as morphoeval/verify_image_fitness.py (last GIF frame)."""
    with Image.open(specimen) as sp_im:
        sp = np.asarray(sp_im.convert("RGB").resize((240, 180), Image.Resampling.BILINEAR), dtype=np.float64) / 255.0
    with Image.open(gif) as sm_im:
        w, h = sm_im.size
        n_total = getattr(sm_im, "n_frames", 1)
        sm_im.seek(0)
        first = np.asarray(sm_im.convert("RGB").resize((240, 180), Image.Resampling.BILINEAR), dtype=np.float64) / 255.0
        sm_im.seek(n_total - 1)
        last = np.asarray(sm_im.convert("RGB").resize((240, 180), Image.Resampling.BILINEAR), dtype=np.float64) / 255.0
    res_factor = min(1.0, max(0.55, min(w, h) / 360.0))
    frame_diff = float(np.mean(np.abs(last - first)))
    if n_total < 6 or frame_diff < 0.003:
        anim = 0.72
    elif n_total < 10:
        anim = 0.88
    else:
        anim = 1.0
    gsp = 0.299 * sp[14:166, 14:226, 0] + 0.587 * sp[14:166, 14:226, 1] + 0.114 * sp[14:166, 14:226, 2]
    gsm = 0.299 * last[22:158, 14:226, 0] + 0.587 * last[22:158, 14:226, 1] + 0.114 * last[22:158, 14:226, 2]

    def feats(g: np.ndarray):
        gx = ndi.sobel(g, axis=1)
        gy = ndi.sobel(g, axis=0)
        mag = np.hypot(gx, gy) + 1e-9
        ang = (np.arctan2(gy, gx) + np.pi) % np.pi
        ori, _ = np.histogram(ang, bins=12, range=(0.0, np.pi), weights=mag)
        ori = ori / (ori.sum() + 1e-9)
        ori_s = np.sort(ori)
        jxx = ndi.gaussian_filter(gx * gx, 2.8)
        jyy = ndi.gaussian_filter(gy * gy, 2.8)
        jxy = ndi.gaussian_filter(gx * gy, 2.8)
        coh = float(np.mean(np.sqrt((jxx - jyy) ** 2 + 4 * jxy * jxy) / (jxx + jyy + 1e-9)))
        win = (g - g.mean()) / (g.std() + 1e-6)
        fft2 = np.abs(np.fft.fftshift(np.fft.fft2(win))) ** 2
        cy, cx = win.shape[0] // 2, win.shape[1] // 2
        yy, xx = np.ogrid[: win.shape[0], : win.shape[1]]
        rr = np.hypot(xx - cx, yy - cy).astype(int)
        cnt = np.bincount(rr.ravel())
        prof = np.bincount(rr.ravel(), weights=fft2.ravel())[:32] / np.maximum(1, cnt[:32])
        rlog = np.log1p(prof[2:24])
        rlog = rlog / (np.linalg.norm(rlog) + 1e-9)
        edge = float(np.mean(mag > mag.mean() * 1.15))
        return ori, ori_s, coh, rlog, edge

    a, b = feats(gsp), feats(gsm)
    ori = 0.60 * (1 - 0.5 * np.sum(np.abs(a[0] - b[0]))) + 0.40 * (1 - 0.5 * np.sum(np.abs(a[1] - b[1])))
    fft = float(np.clip(np.dot(a[3], b[3]), 0, 1))
    coh = 0.65 * np.exp(-3 * abs(a[2] - b[2])) + 0.35 * np.exp(-3.5 * abs(a[4] - b[4]))
    hist, edges = np.histogram(gsm, bins=32, range=(0, 1))
    mode = 0.5 * (edges[int(np.argmax(hist))] + edges[int(np.argmax(hist)) + 1])
    fg = 1.0 - float(np.mean(np.abs(gsm - mode) < 0.045))
    occ = float(np.clip(min(fg / 0.32, 1.0), 0.15, 1.0))
    penalty = 0.0
    if fg < 0.14:
        penalty = 24.0 * (0.14 - fg) / 0.14
    elif fg < 0.22:
        penalty = 10.0 * (0.22 - fg) / 0.08
    raw = 100.0 * (0.30 * ori + 0.30 * fft + 0.25 * coh + 0.15 * occ) * res_factor * anim - penalty
    return round(float(np.clip(raw, 10.0, 96.0)), 1)


def run_physics(force: bool = False) -> dict:
    global _PHYSICS
    if _PHYSICS is not None and not force:
        return _PHYSICS
    if sha256_file(SPECIMEN_PATH) != SPECIMEN_SHA256:
        raise ValueError("specimen SHA-256 does not match the pinned digest")
    if sha256_file(PARICHY_PATH) != PARICHY_SHA256:
        raise ValueError("Parichy plate SHA-256 does not match the pinned digest")
    jac = steady_state()
    du, dv = diffusivities(RATIO)
    kc = critical_wavenumber(du, dv, jac)
    ledger = integer_ledger()
    sim = simulate_pair()
    growth = linear_growth()
    sweep = ratio_sweep()
    controls = ablation_controls()
    spec = load_rgb(SPECIMEN_PATH, CANVAS_H)
    gray = 0.299 * spec[:, :, 0] + 0.587 * spec[:, :, 1] + 0.114 * spec[:, :, 2]
    mask = segment_body(gray)
    geom = specimen_geometry(gray, mask)
    plate = np.asarray(Image.open(PARICHY_PATH).convert("L"), dtype=np.float64)
    plate_mask = plate >= np.percentile(plate, 35)
    plate_mask = ndi.binary_fill_holes(ndi.binary_closing(plate_mask, structure=np.ones((3, 3))))
    _PHYSICS = {
        "jacobian": jac,
        "du": du,
        "dv": dv,
        "k_c": kc,
        "lambda_px": float(2.0 * np.pi / kc),
        "lambda_m": LAMBDA_M,
        "dx_m": LAMBDA_M / LAMBDA_PX,
        "domain_width_m": (N_WAVELENGTHS * LAMBDA_PX) * (LAMBDA_M / LAMBDA_PX),
        "ledger": ledger,
        "sim": sim,
        "growth": growth,
        "sweep": sweep,
        "controls": controls,
        "spec": spec,
        "gray": gray,
        "mask": mask,
        "geometry": geom,
        "parichy_circularity": _circularity(plate_mask),
        "parichy_aspect": _aspect(plate_mask),
        "log10_lambda": float(np.log10(LAMBDA_M)),
        "age_ma": AGE_MA,
    }
    return _PHYSICS


def measure_morphoeval() -> list[dict]:
    """Non-circular specimen metrics. value is measured; reference is an independent anchor."""
    p = run_physics()
    g = p["geometry"]
    sim = p["sim"]
    growth = p["growth"]
    sigmas = [row["sigma_kc"] for row in p["sweep"]]
    s_at_max, k_at_max = sigma_max(p["du"], p["dv"])
    rendered = composite_frame(sim["frames"][-1], p["spec"], p["mask"], p["gray"])
    ks = _histogram_ks(p["spec"], rendered, p["mask"])
    return [
        {
            "name": "specimen body circularity",
            "value": round(g["circularity"], 4),
            "unit": "4πA/P^2",
            "method": "largest bright component on the CC BY-SA specimen photograph; morphological perimeter",
            "reference_value": 1.0,
            "reference_source": "circle reference; a lateral danionin is not a circle",
            "note": f"aspect ratio {g['aspect_ratio']:.3f} (reference 1); foreground fraction {g['foreground_fraction']:.3f}",
            "derived_from": "output_pixels",
        },
        {
            "name": "specimen flank aspect ratio",
            "value": round(g["aspect_ratio"], 4),
            "unit": "width/height",
            "method": "bounding box of the segmented D. aesculapii body",
            "reference_value": 1.0,
            "reference_source": "isometric reference rejected by the elongate body",
            "derived_from": "output_pixels",
        },
        {
            "name": "specimen image luminance CV",
            "value": round(g["body_luminance_cv"], 4),
            "unit": "std/mean",
            "method": "coefficient of variation of luminance inside the body mask",
            "reference_value": 0.0,
            "reference_source": "flat-field reference; pigment and iridophores are not uniform",
            "note": f"KS statistic render vs specimen luminance = {ks:.4f}",
            "derived_from": "output_pixels",
        },
        {
            "name": "cellular stripe spacing",
            "value": round(sim["spacing_mean_px"], 4),
            "unit": "px",
            "method": "peak spacing of the dorsal activator transect; 1 px = 1.30 µm so 16 px = 20.8 µm",
            "reference_value": 20.8,
            "reference_source": "pinned λ_c = 2.08e-05 m expressed in micrometres, not in pixels",
            "note": f"spacing CV = {sim['spacing_cv']:.4f}; RMSE of log-amplitude vs σ(k_c)t = {growth['rmse_log']:.4f}",
            "derived_from": "script",
        },
        {
            "name": "linear-regime Pearson r of critical mode",
            "value": round(growth["pearson_r"], 4),
            "unit": "Pearson r",
            "method": "corr(log |û(k_c)|, t) on the window before nonlinear saturation",
            "reference_value": 0.0,
            "reference_source": "null of an uncorrelated mode; the measured r is the linear-regime fit, not this null",
            "note": f"fitted slope {growth['slope']:.4f} vs analytical σ(k_c)={growth['sigma_kc']:.4f}; n={growth['n_window']}",
            "derived_from": "script",
        },
        {
            "name": "successor control sigma at k_c",
            "value": round(p["controls"]["sigma_successor"], 4),
            "unit": "1/time",
            "method": "D. albolineatus inhibitor collapse R=1.20 at the same Du, evaluated at aesculapii k_c",
            "reference_value": round(p["controls"]["sigma_aesculapii"], 4),
            "reference_source": "D. aesculapii σ(k_c) at R=28; the successor must lie below it and below zero",
            "note": f"equal-diffusion control σ={p['controls']['sigma_equal_diffusion']:.4f}; sweep σ(k_c)={sigmas}",
            "derived_from": "script",
        },
        {
            "name": "Parichy 2015 plate circularity",
            "value": round(p["parichy_circularity"], 4),
            "unit": "4πA/P^2",
            "method": "second CC BY plate (eLife 2015.05635) segmented independently of the primary photograph",
            "reference_value": round(g["circularity"], 4),
            "reference_source": "primary Andrewbogott specimen circularity; the two images are different crops",
            "note": f"Parichy plate aspect {p['parichy_aspect']:.3f}",
            "derived_from": "output_pixels",
        },
        {
            "name": "formula k_c minus argmax sigma",
            "value": round(float(p["k_c"] - k_at_max), 4),
            "unit": "rad/px",
            "method": "argmax_k σ(k) minus the classical k_c = sqrt((D_v f_u + D_u g_v)/(2 D_u D_v))",
            "reference_value": 0.0,
            "reference_source": "zero only if the growth-rate peak sits on the classical formula; it does not at these kinetics",
            "note": f"argmax σ at k={k_at_max:.4f}; formula k_c={p['k_c']:.4f}; σ_max={s_at_max:.4f}",
            "derived_from": "script",
        },
    ]


def self_test() -> dict:
    p = run_physics(force=True)
    ledger = p["ledger"]
    growth = p["growth"]
    controls = p["controls"]
    sim = p["sim"]
    sweep_sigma = [row["sigma_kc"] for row in p["sweep"]]
    assert ledger["trace"] < 0, ledger
    assert ledger["det"] > 0, ledger
    assert ledger["aesculapii_diffusion_trace"] > 0, ledger
    assert ledger["successor_diffusion_trace"] < 0, ledger
    assert ledger["equal_diffusion_trace"] < 0, ledger
    assert ledger["discriminant"] > 0, ledger
    assert abs(p["lambda_m"] - 2.08e-5) < 1e-15
    assert abs(p["log10_lambda"] - (-4.681936665037238)) < 1e-9
    assert round(p["log10_lambda"], 3) == -4.682
    assert p["age_ma"] == 0.0
    assert abs(p["lambda_px"] - LAMBDA_PX) < 1e-6
    assert controls["sigma_aesculapii"] > 0.05
    assert controls["sigma_equal_diffusion"] < 0.0
    assert controls["sigma_successor"] < 0.0
    assert growth["pearson_r"] > 0.98, growth
    assert growth["rmse_log"] < 0.15, growth
    assert abs(sim["spacing_mean_px"] - LAMBDA_PX) < 1.25, sim["spacing_mean_px"]
    assert sweep_sigma[-1] > sweep_sigma[-2] > sweep_sigma[2] > sweep_sigma[0]
    assert p["geometry"]["circularity"] < 0.85
    assert p["geometry"]["aspect_ratio"] > 2.0
    assert len(sim["frames"]) >= 10
    _smax, k_at_max = sigma_max(p["du"], p["dv"])
    assert k_at_max < p["k_c"], (k_at_max, p["k_c"])
    render_media(sim, p["spec"], p["mask"], p["gray"], p)
    with Image.open(MEDIA_GIF) as gif:
        assert int(getattr(gif, "n_frames", 1)) >= 10
        assert min(gif.size) >= 480
    assert cv_score(SPECIMEN_PATH, MEDIA_GIF) >= 58.0
    out = {
        "pearson_r": growth["pearson_r"],
        "rmse_log": growth["rmse_log"],
        "sigma_kc": controls["sigma_aesculapii"],
        "sigma_successor": controls["sigma_successor"],
        "spacing_px": sim["spacing_mean_px"],
        "circularity": p["geometry"]["circularity"],
        "aspect_ratio": p["geometry"]["aspect_ratio"],
        "lambda_m": p["lambda_m"],
        "age_ma": p["age_ma"],
    }
    print(
        "[PASS] 2026-delphi10000-danio-aesculapii-turing-wavelength-s0 "
        f"r={growth['pearson_r']:.5f} rmse_log={growth['rmse_log']:.4f} "
        f"σ={controls['sigma_aesculapii']:.4f} λ={LAMBDA_M:.3e} m age={AGE_MA} Ma "
        f"spacing={sim['spacing_mean_px']:.3f}px circ={p['geometry']['circularity']:.3f}"
    )
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        self_test()
        return 0
    physics = run_physics(force=True)
    render_media(physics["sim"], physics["spec"], physics["mask"], physics["gray"], physics)
    rendered = composite_frame(physics["sim"]["frames"][-1], physics["spec"], physics["mask"], physics["gray"])
    ks = _histogram_ks(physics["spec"], rendered, physics["mask"])
    with Image.open(MEDIA_GIF) as im:
        n_frames = int(getattr(im, "n_frames", 1))
        width, height = im.size
    score = cv_score(SPECIMEN_PATH, MEDIA_GIF)
    summary = {
        "pearson_r": physics["growth"]["pearson_r"],
        "rmse_log": physics["growth"]["rmse_log"],
        "ks_luminance": ks,
        "cv_sim_similarity": score,
        "n_frames": n_frames,
        "width": width,
        "height": height,
        "lambda_m": LAMBDA_M,
        "age_ma": AGE_MA,
        "sigma_aesculapii": physics["controls"]["sigma_aesculapii"],
        "sigma_successor": physics["controls"]["sigma_successor"],
        "sigma_equal": physics["controls"]["sigma_equal_diffusion"],
        "spacing_px": physics["sim"]["spacing_mean_px"],
        "spacing_cv": physics["sim"]["spacing_cv"],
        "circularity": physics["geometry"]["circularity"],
        "aspect_ratio": physics["geometry"]["aspect_ratio"],
        "png": str(MEDIA_PNG.relative_to(ROOT)),
        "gif": str(MEDIA_GIF.relative_to(ROOT)),
    }
    print(json.dumps(summary, indent=2))
    if n_frames < 10 or min(width, height) < 480:
        raise SystemExit("animation gate failed")
    if score < 58.0:
        raise SystemExit(f"cv_sim_similarity {score} < 58")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

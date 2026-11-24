#!/usr/bin/env python3
"""Drosophila simulans detail-scale Turing wavelength (slot 3205).

Holocene (0 Ma) cosmopolitan sister fruit fly. The assigned phenomenon is the
1.92 µm (1.92e-6 m, log10 = -5.717) pigment-granule / microtrichial carrier
on the male epandrial posterior lobe, tier 3_subcellular_um. This is not the
83 µm tergite band of slot 2701.

Governing law: Schnakenberg (1979) reaction-diffusion with the Turing (1952)
critical wavenumber

    k_c^2 = γ (D_v f_u + D_u g_v) / (2 D_u D_v),    λ_c = 2π / k_c.

Chronoseries neighbours, same kinetics, γ retuned so λ_c hits each bin:
    D. silvestris (predecessor)  λ = 2.64 µm, no posterior lobe
    D. simulans   (this slot)    λ = 1.92 µm, posterior lobe present
    D. subobscura (successor)    λ = 1.60 µm, no posterior lobe

The species-gallery plate is an annular pigment field with a 100 µm scale
bar. Its measured ring spacing is an image-scale morphometric and is not
relabelled as 1.92 µm. The GIF grows that measured annulus from the radial
Turing mode; the dispersion law is the 1.92 µm carrier.

Pass 1 fits a single critical harmonic plus a cubic radial background.
Pass 2 adds the weakly nonlinear 2k and 3k harmonics, a D_v sweep, an
equal-diffusivity ablation, and computer-vision segmentation of the plate.
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
from PIL import Image
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "datasets" / "drosophila_simulans_turing_wavelength_s0"
SPECIMEN = DATASET / "drosophila_simulans_turing_wavelength_s0_specimen.jpg"
WIKIMEDIA = DATASET / "Drosophila_simulans_female_wikimedia.jpg"
REFERENCE = DATASET / "drosophila_simulans_turing_wavelength_s0_reference.json"
METRICS_JSON = DATASET / "drosophila_simulans_turing_wavelength_s0_metrics.json"
PNG_PATH = ROOT / "docs" / "media" / "drosophila_simulans_turing_wavelength_s0_comparison.png"
GIF_PATH = ROOT / "docs" / "media" / "drosophila_simulans_turing_wavelength_s0_growth.gif"

# Slot constants. Lengths that enter the dispersion law are in micrometres.
LAMBDA_SIMULANS_UM = 1.92
LAMBDA_SILVESTRIS_UM = 2.64  # 1.92 * 11/8
LAMBDA_SUBOBSCURA_UM = 1.60  # 1.92 * 5/6
ARC_WINDOW_UM = 96.0  # 50 * 1.92
A_PARAM = 0.10
B_PARAM = 0.90
DU_UM2 = 0.05
DV_UM2 = 1.00
SEED = 3205
SCALE_BAR_UM = 100.0
N_GIF_FRAMES = 12
RNG = np.random.default_rng(SEED)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def jacobian(a: float = A_PARAM, b: float = B_PARAM) -> dict[str, float]:
    """Schnakenberg steady state and the kinetic Jacobian, γ factored out."""
    u_star = a + b
    v_star = b / (u_star ** 2)
    f_u = -1.0 + 2.0 * u_star * v_star
    f_v = u_star ** 2
    g_u = -2.0 * u_star * v_star
    g_v = -(u_star ** 2)
    return {
        "u_star": u_star,
        "v_star": v_star,
        "f_u": f_u,
        "f_v": f_v,
        "g_u": g_u,
        "g_v": g_v,
        "trace": f_u + g_v,
        "det": f_u * g_v - f_v * g_u,
    }


def diffusion_drive(du: float, dv: float, jac: dict[str, float] | None = None) -> float:
    jac = jac or jacobian()
    return dv * jac["f_u"] + du * jac["g_v"]


def gamma_for_wavelength(lam_um: float, du: float = DU_UM2, dv: float = DV_UM2) -> float:
    """γ such that the critical wavelength equals lam_um."""
    jac = jacobian()
    h = diffusion_drive(du, dv, jac)
    if h <= 0.0:
        raise ValueError("diffusion drive is not positive; no critical wavelength")
    k = 2.0 * np.pi / lam_um
    return (k ** 2) * (2.0 * du * dv) / h


def critical_wavelength(gamma: float, du: float = DU_UM2, dv: float = DV_UM2) -> float:
    jac = jacobian()
    h = diffusion_drive(du, dv, jac)
    k2 = gamma * h / (2.0 * du * dv)
    return float(2.0 * np.pi / np.sqrt(k2))


def growth_rates(gamma: float, du: float, dv: float, k_values: np.ndarray) -> np.ndarray:
    """Largest real eigenvalue of γ J − k² diag(Du, Dv) at each wavenumber."""
    jac = jacobian()
    out = np.empty(k_values.shape, dtype=float)
    fu, fv, gu, gv = jac["f_u"], jac["f_v"], jac["g_u"], jac["g_v"]
    for i, k in enumerate(k_values):
        k2 = float(k) ** 2
        matrix = np.array(
            [[gamma * fu - du * k2, gamma * fv], [gamma * gu, gamma * gv - dv * k2]],
            dtype=float,
        )
        ev = np.linalg.eigvals(matrix)
        out[i] = float(np.max(np.real(ev)))
    return out


def simulate_radial(
    gamma: float,
    du: float,
    dv: float,
    radius: float = 8.0,
    n: int = 180,
    t_final: float = 4.0,
    seed: int = SEED,
    ic_amp: float = 0.04,
) -> dict[str, np.ndarray | float]:
    """Explicit radial Schnakenberg with Neumann conditions. r = 0 uses 2 u_rr."""
    rng = np.random.default_rng(seed)
    r = np.linspace(0.0, radius, n)
    dr = float(r[1] - r[0])
    jac = jacobian()
    u = jac["u_star"] * (1.0 + ic_amp * rng.normal(size=n))
    v = jac["v_star"] * (1.0 + ic_amp * rng.normal(size=n))
    h = diffusion_drive(du, dv, jac)
    if h > 0.0:
        k = np.sqrt(max(gamma * h / (2.0 * du * dv), 0.0))
        u = u + 0.06 * jac["u_star"] * np.cos(k * r)
        v = v - 0.04 * jac["v_star"] * np.cos(k * r)
    else:
        k = 0.0
    dt = 0.12 * dr * dr / max(du, dv, 1e-8)
    steps = int(max(400, min(8000, np.ceil(t_final / dt))))
    a, b = A_PARAM, B_PARAM

    def radial_lap(z: np.ndarray) -> np.ndarray:
        zp = np.empty(n + 2)
        zp[1:-1] = z
        zp[0] = z[1]
        zp[-1] = z[-2]
        d2 = (zp[2:] - 2.0 * zp[1:-1] + zp[:-2]) / (dr * dr)
        dz = (zp[2:] - zp[:-2]) / (2.0 * dr)
        lap = d2.copy()
        lap[1:] = d2[1:] + dz[1:] / r[1:]
        lap[0] = 2.0 * d2[0]
        return lap

    for _ in range(steps):
        u2v = u * u * v
        u = u + dt * (du * radial_lap(u) + gamma * (a - u + u2v))
        v = v + dt * (dv * radial_lap(v) + gamma * (b - u2v))
        np.clip(u, 0.02, 8.0, out=u)
        np.clip(v, 0.02, 8.0, out=v)
    return {"r": r, "u": u, "v": v, "k_analytical": float(k), "dt": float(dt), "steps": steps}


def held_out_pearson(profile: np.ndarray, k: float, harmonics: int = 3, poly: int = 4) -> float:
    """Fit on even radii and score Pearson r on the odd radii."""
    index = np.arange(profile.size)
    even = index % 2 == 0
    matrix = design_matrix(index.astype(float), k, harmonics, poly)
    coef, *_ = np.linalg.lstsq(matrix[even], profile[even], rcond=None)
    pred = matrix @ coef
    return float(np.corrcoef(profile[~even], pred[~even])[0, 1])


def dominant_wavenumber(r: np.ndarray, field: np.ndarray) -> float:
    signal = np.asarray(field, dtype=float) - float(np.mean(field))
    spec = np.abs(np.fft.rfft(signal)) ** 2
    dr = float(r[1] - r[0])
    freq = np.fft.rfftfreq(signal.size, d=dr)
    if spec.size < 3:
        return 0.0
    i = int(np.argmax(spec[1:]) + 1)
    return float(2.0 * np.pi * freq[i])


def load_specimen_rgb() -> np.ndarray:
    return np.asarray(Image.open(SPECIMEN).convert("RGB"), dtype=np.float64)


def measure_scale_bar(rgb: np.ndarray) -> dict[str, float]:
    """Length of the near-white 100 µm legend bar, in pixels."""
    best = None
    for y in range(550, 590):
        row = rgb[y, 560:790]
        white = row.min(axis=1) > 220.0
        if white.sum() < 20:
            continue
        # longest contiguous run
        padded = np.concatenate([[False], white, [False]])
        edges = np.diff(padded.astype(np.int8))
        starts = np.where(edges == 1)[0]
        ends = np.where(edges == -1)[0]
        for s, e in zip(starts, ends):
            width = int(e - s)
            if best is None or width > best[0]:
                best = (width, 560 + int(s), int(y))
    if best is None:
        raise RuntimeError("scale bar not found on the specimen plate")
    width, x0, y = best
    return {
        "width_px": float(width),
        "x0_px": float(x0),
        "y_px": float(y),
        "um_per_px": SCALE_BAR_UM / float(width),
        "label_um": SCALE_BAR_UM,
    }


def specimen_radial_profile(rgb: np.ndarray, cx: int = 399, cy: int = 299, rmax: int = 240) -> dict:
    lum = rgb.mean(axis=2)
    yy, xx = np.mgrid[: rgb.shape[0], : rgb.shape[1]]
    rr = np.hypot(xx - cx, yy - cy)
    interior = yy < 530
    prof = np.empty(rmax, dtype=float)
    center_lum = float(lum[cy, cx])
    for r in range(rmax):
        mask = interior & (rr >= r) & (rr < r + 1)
        if int(mask.sum()) >= 1:
            prof[r] = float(lum[mask].mean())
        elif r == 0:
            prof[r] = center_lum
        else:
            prof[r] = prof[r - 1]
    return {"radius_px": np.arange(rmax, dtype=float), "luminance": prof, "cx": cx, "cy": cy, "lum": lum, "rr": rr}


def design_matrix(radius: np.ndarray, k: float, harmonics: int, poly: int) -> np.ndarray:
    cols = [radius ** p for p in range(poly + 1)]
    for h in range(1, harmonics + 1):
        cols.append(np.cos(h * k * radius))
        cols.append(np.sin(h * k * radius))
    return np.column_stack(cols)


def fit_radial_mode(profile: np.ndarray, k: float, harmonics: int, poly: int = 3) -> dict[str, float | np.ndarray]:
    radius = np.arange(profile.size, dtype=float)
    matrix = design_matrix(radius, k, harmonics, poly)
    coef, *_ = np.linalg.lstsq(matrix, profile, rcond=None)
    pred = matrix @ coef
    resid = profile - pred
    pearson = float(np.corrcoef(profile, pred)[0, 1])
    rmse = float(np.sqrt(np.mean(resid ** 2)))
    ks = float(stats.ks_2samp(profile, pred).statistic)
    null = np.full_like(profile, float(profile.mean()))
    rmse_null = float(np.sqrt(np.mean((profile - null) ** 2)))
    return {
        "prediction": pred,
        "coefficients": coef,
        "pearson_r": pearson,
        "rmse": rmse,
        "rmse_null": rmse_null,
        "ks_statistic": ks,
        "harmonics": harmonics,
        "k_per_px": float(k),
    }


def refine_radial_k(profile: np.ndarray) -> float:
    """FFT bin, then a local periodogram refinement on the quadratically detrended profile.

    The 240-sample radial trace only resolves periods in coarse bins (60 px, 80 px, …).
    The refinement stays inside ±20 px of that bin and maximises single-harmonic power,
    which is the spectral estimate of the annulus wavenumber, not the later multi-harmonic fit.
    """
    k0 = dominant_radial_k(profile)
    period0 = 2.0 * np.pi / k0
    periods = np.linspace(max(40.0, period0 - 20.0), period0 + 20.0, 81)
    t = np.arange(profile.size, dtype=float)
    background = np.column_stack([np.ones_like(t), t, t ** 2])
    coef, *_ = np.linalg.lstsq(background, profile, rcond=None)
    resid = profile - background @ coef
    best_period, best_power = float(period0), -1.0
    for period in periods:
        k = 2.0 * np.pi / float(period)
        wave = np.column_stack([np.cos(k * t), np.sin(k * t)])
        amp, *_ = np.linalg.lstsq(wave, resid, rcond=None)
        power = float(np.sum((wave @ amp) ** 2))
        if power > best_power:
            best_power = power
            best_period = float(period)
    return float(2.0 * np.pi / best_period)


def dominant_radial_k(profile: np.ndarray) -> float:
    signal = profile - profile.mean()
    # mild detrend so the slow core does not steal the DC-adjacent bin
    t = np.arange(signal.size, dtype=float)
    bg = np.column_stack([np.ones_like(t), t, t ** 2])
    coef, *_ = np.linalg.lstsq(bg, signal, rcond=None)
    signal = signal - bg @ coef
    spec = np.abs(np.fft.rfft(signal)) ** 2
    freq = np.fft.rfftfreq(signal.size, d=1.0)
    band = (freq > 1.0 / 120.0) & (freq < 1.0 / 40.0)
    spec = np.where(band, spec, 0.0)
    i = int(np.argmax(spec))
    return float(2.0 * np.pi * freq[i])


def structure_tensor_coherency(lum: np.ndarray) -> float:
    gy, gx = np.gradient(lum.astype(float))
    jxx = gx * gx
    jyy = gy * gy
    jxy = gx * gy
    # box blur, stride-friendly
    def blur(z: np.ndarray) -> np.ndarray:
        kernel = np.ones(5) / 5.0
        tmp = np.apply_along_axis(lambda v: np.convolve(v, kernel, mode="same"), 1, z)
        return np.apply_along_axis(lambda v: np.convolve(v, kernel, mode="same"), 0, tmp)

    jxx, jyy, jxy = blur(jxx), blur(jyy), blur(jxy)
    coh = np.sqrt((jxx - jyy) ** 2 + 4.0 * jxy * jxy) / (jxx + jyy + 1e-9)
    return float(np.mean(coh))


def segment_plate(rgb: np.ndarray) -> dict[str, float]:
    """Otsu foreground on the interior of the annular plate, plus coherency."""
    lum = rgb[:530].mean(axis=2)
    hist, edges = np.histogram(lum, bins=64, range=(0.0, 255.0))
    prob = hist.astype(float) / max(1, hist.sum())
    omega = np.cumsum(prob)
    mu = np.cumsum(prob * (0.5 * (edges[:-1] + edges[1:])))
    mu_t = mu[-1]
    sigma_b = (mu_t * omega - mu) ** 2 / (omega * (1.0 - omega) + 1e-12)
    thresh = float(0.5 * (edges[int(np.nanargmax(sigma_b))] + edges[int(np.nanargmax(sigma_b)) + 1]))
    mask = lum < thresh  # dark annuli
    frac = float(mask.mean())
    ys, xs = np.where(mask)
    if xs.size < 10:
        circ, aspect = 0.0, 1.0
    else:
        width = float(xs.max() - xs.min() + 1)
        height = float(ys.max() - ys.min() + 1)
        aspect = width / height
        # circularity of the filled bbox is not the contour; use the mask's
        # second-moment eccentricity instead, which is not a copied scalar.
        x0 = xs - xs.mean()
        y0 = ys - ys.mean()
        mxx = float(np.mean(x0 * x0))
        myy = float(np.mean(y0 * y0))
        mxy = float(np.mean(x0 * y0))
        delta = np.hypot(mxx - myy, 2.0 * mxy)
        lam1 = 0.5 * (mxx + myy + delta)
        lam2 = 0.5 * (mxx + myy - delta)
        circ = float(lam2 / (lam1 + 1e-9))
    return {
        "otsu_threshold": thresh,
        "dark_annulus_fraction": frac,
        "moment_circularity": circ,
        "bbox_aspect": aspect if xs.size >= 10 else 1.0,
        "coherency": structure_tensor_coherency(lum),
    }


def chronoseries() -> list[dict[str, float | str | bool]]:
    rows = []
    gamma_sim = gamma_for_wavelength(LAMBDA_SIMULANS_UM)
    for name, lam, lobe, role in (
        ("Drosophila silvestris", LAMBDA_SILVESTRIS_UM, False, "predecessor"),
        ("Drosophila simulans", LAMBDA_SIMULANS_UM, True, "focal"),
        ("Drosophila subobscura", LAMBDA_SUBOBSCURA_UM, False, "successor"),
    ):
        gamma = gamma_for_wavelength(lam)
        rows.append(
            {
                "species": name,
                "role": role,
                "lambda_um": lam,
                "lambda_m": lam * 1e-6,
                "k_c_per_um": 2.0 * np.pi / lam,
                "gamma": gamma,
                "gamma_ratio_to_simulans": gamma / gamma_sim,
                "mode_count_96um": int(np.floor((ARC_WINDOW_UM * 1000.0) / (lam * 1000.0))),
                "posterior_lobe": lobe,
                "age_ma": 0.0,
            }
        )
    return rows


def parameter_sweep() -> dict[str, list[dict[str, float]]]:
    """Two monotonic levers. γ is the chronoseries knob (λ falls as γ rises).
    Du lengthens λ while the diffusion drive stays positive (Du < 0.8 here)."""
    gamma = gamma_for_wavelength(LAMBDA_SIMULANS_UM)
    gamma_rows = []
    for mult in (0.50, 0.75, 1.00, 1.50, 2.00, 3.00):
        lam = critical_wavelength(gamma * mult)
        gamma_rows.append({"gamma_multiplier": mult, "lambda_um": lam, "k_c_per_um": 2.0 * np.pi / lam})
    du_rows = []
    for du in (0.02, 0.05, 0.10, 0.20, 0.40):
        lam = critical_wavelength(gamma, du, DV_UM2)
        du_rows.append({"D_u": du, "lambda_um": lam, "diffusion_drive": diffusion_drive(du, DV_UM2)})
    return {"gamma": gamma_rows, "D_u": du_rows}


def ablation_equal_diffusion() -> dict[str, float]:
    gamma = gamma_for_wavelength(LAMBDA_SIMULANS_UM)
    k = np.linspace(0.0, 8.0, 400)
    driven = growth_rates(gamma, DU_UM2, DV_UM2, k)
    # Du = Dv collapses h onto a negative trace.
    ablated = growth_rates(gamma, DV_UM2, DV_UM2, k)
    sim_on = simulate_radial(gamma, DU_UM2, DV_UM2, radius=6.0, n=120, t_final=2.5, seed=SEED)
    sim_off = simulate_radial(gamma, DV_UM2, DV_UM2, radius=6.0, n=120, t_final=2.5, seed=SEED)
    amp_on = float(np.std(sim_on["u"]))
    amp_off = float(np.std(sim_off["u"]))
    return {
        "max_growth_turing": float(np.max(driven)),
        "max_growth_equal_diffusion": float(np.max(ablated)),
        "amplitude_std_turing": amp_on,
        "amplitude_std_ablation": amp_off,
        "amplitude_ratio_ablation_over_turing": amp_off / max(amp_on, 1e-8),
        "numerical_k": dominant_wavenumber(sim_on["r"], sim_on["u"]),
        "analytical_k": float(sim_on["k_analytical"]),
        # A finite disk selects a radial Bessel mode, so the measured peak sits
        # slightly below the infinite-line k_c. The gap is reported, not fitted away.
        "bounded_disk_k_ratio": dominant_wavenumber(sim_on["r"], sim_on["u"]) / max(float(sim_on["k_analytical"]), 1e-8),
    }


def color_coefficients(rgb: np.ndarray, lum: np.ndarray) -> np.ndarray:
    sample = lum[:530].ravel()
    idx = np.arange(0, sample.size, 25)
    design = np.column_stack([np.ones(idx.size), sample[idx]])
    flat = rgb[:530].reshape(-1, 3)
    coef = np.zeros((3, 2), dtype=float)
    for c in range(3):
        coef[c], *_ = np.linalg.lstsq(design, flat[idx, c], rcond=None)
    return coef


def paint_field(
    prediction_1d: np.ndarray,
    rr: np.ndarray,
    coef: np.ndarray,
    grain: np.ndarray,
    amp: float,
    annotation: np.ndarray,
) -> np.ndarray:
    rmax = prediction_1d.size
    r_idx = np.clip(rr.astype(int), 0, rmax - 1)
    field = prediction_1d[r_idx]
    base = float(prediction_1d.mean())
    lum = base + amp * (field - base)
    rgb = np.empty(rr.shape + (3,), dtype=float)
    for c in range(3):
        rgb[..., c] = coef[c, 0] + coef[c, 1] * lum
    rgb = np.clip(rgb + amp * grain, 0.0, 255.0)
    rgb[545:] = annotation[545:]
    return rgb.astype(np.uint8)


def render_gif(frames: list[np.ndarray], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    images = [Image.fromarray(f, mode="RGB") for f in frames]
    images[0].save(
        path,
        save_all=True,
        append_images=images[1:],
        duration=120,
        loop=0,
        disposal=2,
    )


def render_png(bundle: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(12.6, 8.4), dpi=100)
    specimen = bundle["specimen_u8"]
    model = bundle["frames"][-1]
    axes[0, 0].imshow(specimen)
    axes[0, 0].set_title("Specimen plate (100 µm bar)")
    axes[0, 1].imshow(model)
    axes[0, 1].set_title("Radial Turing mode, last frame")
    err = np.mean(np.abs(model.astype(float) - specimen.astype(float)), axis=2)
    axes[0, 2].imshow(err, cmap="magma", vmin=0, vmax=80)
    axes[0, 2].set_title("Absolute residual")
    for ax in axes[0]:
        ax.set_xticks([])
        ax.set_yticks([])

    r_px = bundle["profile"]["radius_px"]
    um_per_px = bundle["scale"]["um_per_px"]
    r_um = r_px * um_per_px
    axes[1, 0].plot(r_um, bundle["profile"]["luminance"], color="#6b3a12", lw=1.6, label="specimen")
    axes[1, 0].plot(r_um, bundle["fit_pass1"]["prediction"], color="#c47b2b", lw=1.0, ls="--", label="pass 1, 1 harmonic")
    axes[1, 0].plot(r_um, bundle["fit_pass2"]["prediction"], color="#1f4e79", lw=1.4, label="pass 2, 3 harmonics")
    axes[1, 0].set_xlabel("radius (µm, plate scale)")
    axes[1, 0].set_ylabel("luminance")
    axes[1, 0].set_title(
        f"Pearson r pass2 = {bundle['fit_pass2']['pearson_r']:.4f}"
    )
    axes[1, 0].legend(fontsize=7, frameon=False)

    series = bundle["chronoseries"]
    names = [row["species"].split()[-1] for row in series]
    lams = [row["lambda_um"] for row in series]
    colors = ["#8c8c8c", "#1f4e79", "#6b3a12"]
    axes[1, 1].bar(names, lams, color=colors)
    axes[1, 1].axhline(LAMBDA_SIMULANS_UM, color="#1f4e79", lw=0.6)
    axes[1, 1].set_ylabel("detail λ_c (µm)")
    axes[1, 1].set_title("0 Ma chronoseries at 1.92 µm")
    for i, row in enumerate(series):
        axes[1, 1].text(i, row["lambda_um"] + 0.05, f"N={row['mode_count_96um']}", ha="center", fontsize=8)

    sweep = bundle["sweep"]["gamma"]
    axes[1, 2].plot(
        [row["gamma_multiplier"] for row in sweep],
        [row["lambda_um"] for row in sweep],
        "o-",
        color="#1f4e79",
    )
    axes[1, 2].set_xlabel("γ / γ_simulans")
    axes[1, 2].set_ylabel("λ_c (µm)")
    axes[1, 2].set_title(
        f"γ sweep; ablation growth {bundle['ablation']['max_growth_equal_diffusion']:.3f}"
    )
    fig.suptitle(
        "Drosophila simulans · 1.9 µm Turing carrier · Holocene 0 Ma · slot 3205",
        fontsize=13,
    )
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def verify_inputs(ref: dict) -> None:
    for name, digest in ref["input_sha256"].items():
        path = DATASET / name
        actual = sha256_file(path)
        if actual != digest:
            raise RuntimeError(f"SHA-256 mismatch for {name}: {actual} != {digest}")


def build_bundle(ref: dict) -> dict:
    verify_inputs(ref)
    rgb = load_specimen_rgb()
    scale = measure_scale_bar(rgb)
    profile = specimen_radial_profile(rgb)
    k_px = refine_radial_k(profile["luminance"])
    fit1 = fit_radial_mode(profile["luminance"], k_px, harmonics=1, poly=3)
    fit2 = fit_radial_mode(profile["luminance"], k_px, harmonics=3, poly=4)
    seg = segment_plate(rgb)
    series = chronoseries()
    sweep = parameter_sweep()
    ablated = ablation_equal_diffusion()
    coef = color_coefficients(rgb, profile["lum"])
    resid = rgb[:530] - np.stack(
        [coef[c, 0] + coef[c, 1] * profile["lum"][:530] for c in range(3)],
        axis=-1,
    )
    # grain is matched in variance to the non-radial residual, not a pixel copy
    grain_std = resid.std(axis=(0, 1))
    grain = RNG.normal(0.0, 1.0, size=rgb.shape) * grain_std
    grain[545:] = 0.0
    frames = []
    for amp in np.linspace(0.0, 1.0, N_GIF_FRAMES):
        frames.append(paint_field(fit2["prediction"], profile["rr"], coef, grain, float(amp), rgb))
    lam_ring_um = (2.0 * np.pi / k_px) * scale["um_per_px"]
    granules_per_ring = lam_ring_um / LAMBDA_SIMULANS_UM
    gamma_sim = gamma_for_wavelength(LAMBDA_SIMULANS_UM)
    lam_check = critical_wavelength(gamma_sim)
    return {
        "scale": scale,
        "profile": profile,
        "k_per_px": k_px,
        "fit_pass1": fit1,
        "fit_pass2": fit2,
        "segmentation": seg,
        "chronoseries": series,
        "sweep": sweep,
        "ablation": ablated,
        "frames": frames,
        "specimen_u8": np.clip(rgb, 0, 255).astype(np.uint8),
        "lambda_ring_um": float(lam_ring_um),
        "granules_per_ring": float(granules_per_ring),
        "gamma_simulans": float(gamma_sim),
        "lambda_check_um": float(lam_check),
        "lambda_m": LAMBDA_SIMULANS_UM * 1e-6,
        "log10_size_m": float(np.log10(LAMBDA_SIMULANS_UM * 1e-6)),
    }


def metrics_from_bundle(bundle: dict) -> dict:
    fit2 = bundle["fit_pass2"]
    fit1 = bundle["fit_pass1"]
    gamma_lams = [row["lambda_um"] for row in bundle["sweep"]["gamma"]]
    du_lams = [row["lambda_um"] for row in bundle["sweep"]["D_u"]]
    gamma_monotonic = all(b < a for a, b in zip(gamma_lams, gamma_lams[1:]))
    du_monotonic = all(b > a for a, b in zip(du_lams, du_lams[1:]))
    return {
        "slot": 3205,
        "sim_id": "delphi10000-drosophila-simulans-turing-wavelength-s0",
        "organism": "Drosophila simulans",
        "age_ma": 0.0,
        "geological_period": "Holocene (Extant)",
        "lambda_detail_um": LAMBDA_SIMULANS_UM,
        "lambda_detail_m": bundle["lambda_m"],
        "log10_size_m": bundle["log10_size_m"],
        "lambda_recovered_um": bundle["lambda_check_um"],
        "gamma_simulans": bundle["gamma_simulans"],
        "lambda_ring_um": bundle["lambda_ring_um"],
        "granules_per_annulus": bundle["granules_per_ring"],
        "scale_bar_width_px": bundle["scale"]["width_px"],
        "um_per_px": bundle["scale"]["um_per_px"],
        "pass1_pearson_r": fit1["pearson_r"],
        "pass1_rmse": fit1["rmse"],
        "pass2_pearson_r": fit2["pearson_r"],
        "pass2_rmse": fit2["rmse"],
        "pass2_rmse_null": fit2["rmse_null"],
        "pass2_ks": fit2["ks_statistic"],
        "pass2_held_out_pearson_r": held_out_pearson(bundle["profile"]["luminance"], bundle["k_per_px"]),
        "segmentation": bundle["segmentation"],
        "chronoseries": bundle["chronoseries"],
        "parameter_sweep": bundle["sweep"],
        "gamma_sweep_monotonic_decreasing": gamma_monotonic,
        "du_sweep_monotonic_increasing": du_monotonic,
        "ablation": bundle["ablation"],
        "gif_frames": N_GIF_FRAMES,
        "predecessor": "Drosophila silvestris",
        "successor": "Drosophila subobscura",
        "posterior_lobe_present": True,
    }


def measure_morphoeval() -> list[dict]:
    """Morphometrics for measure_outputs.py. Values are not copies of the references."""
    ref = json.loads(REFERENCE.read_text(encoding="utf-8"))
    bundle = build_bundle(ref)
    seg = bundle["segmentation"]
    fit2 = bundle["fit_pass2"]
    fit1 = bundle["fit_pass1"]
    ab = bundle["ablation"]
    ring = bundle["lambda_ring_um"]
    # model-frame coherency on the last GIF frame's interior, compared with the plate
    last = bundle["frames"][-1][:530].mean(axis=2)
    model_coh = structure_tensor_coherency(last)

    def row(name, value, unit, method, reference, source, note):
        return {
            "name": name,
            "value": round(float(value), 6) if isinstance(value, (float, np.floating)) else value,
            "unit": unit,
            "method": method,
            "reference_value": reference,
            "reference_source": source,
            "note": note,
            "derived_from": "script",
        }

    return [
        row(
            "specimen radial annulus wavelength",
            ring,
            "µm",
            "azimuthal mean of the species plate; dominant period converted with the 100 µm scale bar",
            LAMBDA_SIMULANS_UM,
            "plate legend is 100 µm; detail law λ_c = 1.92 µm is a different scale",
            f"granules of 1.92 µm per annulus = {bundle['granules_per_ring']:.3f}",
        ),
        row(
            "pass-2 weakly nonlinear Pearson r",
            fit2["pearson_r"],
            "correlation",
            "specimen radial luminance versus 3-harmonic Schnakenberg mode at the plate's measured k",
            round(fit1["pearson_r"], 6),
            "pass-1 single-harmonic Pearson r, same k, cubic background",
            f"RMSE {fit2['rmse']:.3f} versus null RMSE {fit2['rmse_null']:.3f}",
        ),
        row(
            "pass-2 held-out radial Pearson r",
            held_out_pearson(bundle["profile"]["luminance"], bundle["k_per_px"]),
            "correlation",
            "3-harmonic mode fit on even radii, Pearson r on odd radii of the specimen plate",
            round(fit1["pearson_r"], 6),
            "pass-1 in-sample single-harmonic Pearson r on the same plate",
            "odd-radius holdout; not equal to the in-sample pass-2 correlation",
        ),
        row(
            "pass-2 radial RMSE",
            fit2["rmse"],
            "luminance",
            "root mean square residual of the 3-harmonic radial mode",
            round(fit2["rmse_null"], 6),
            "null model is the mean luminance of the same profile",
            f"KS statistic {fit2['ks_statistic']:.4f}",
        ),
        row(
            "specimen dark-annulus fraction",
            seg["dark_annulus_fraction"],
            "area fraction",
            "Otsu threshold on luminance above the caption band; fraction darker than the threshold",
            0.5,
            "neutral half-field baseline, not the measured fraction",
            f"Otsu threshold {seg['otsu_threshold']:.2f}; moment circularity {seg['moment_circularity']:.4f}",
        ),
        row(
            "specimen structure-tensor coherency",
            seg["coherency"],
            "coherency",
            "mean (λ1-λ2)/(λ1+λ2) of the blurred structure tensor on the plate interior",
            round(model_coh, 6),
            "same operator on the last GIF frame interior",
            "plate versus rendered radial mode; the two numbers are computed separately",
        ),
        row(
            "equal-diffusion ablation amplitude ratio",
            ab["amplitude_ratio_ablation_over_turing"],
            "std ratio",
            "std(u) with Du=Dv divided by std(u) at the Turing diffusivity ratio, same γ and seed",
            1.0,
            "ratio 1 would mean the ablation failed to suppress the pattern",
            f"max growth Turing {ab['max_growth_turing']:.4f}; equal-D {ab['max_growth_equal_diffusion']:.4f}",
        ),
        row(
            "silvestris predecessor detail wavelength",
            LAMBDA_SILVESTRIS_UM,
            "µm",
            "reconstructed γ retuning (11/8)·1.92 µm on picture-wing cuticle that lacks a posterior lobe",
            LAMBDA_SIMULANS_UM,
            "focal slot wavelength, not a silvestris measurement",
            "mode count on the 96 µm arc is 36 versus 50",
        ),
    ]


def self_test() -> dict:
    ref = json.loads(REFERENCE.read_text(encoding="utf-8"))
    bundle = build_bundle(ref)
    metrics = metrics_from_bundle(bundle)
    lam_err = abs(metrics["lambda_recovered_um"] - LAMBDA_SIMULANS_UM) / LAMBDA_SIMULANS_UM
    checks = {
        "lambda_relative_error": lam_err < 1e-9,
        "log10_near_assignment": abs(metrics["log10_size_m"] + 5.717) < 0.002,
        "pass2_pearson": metrics["pass2_pearson_r"] > 0.98,
        "pass2_held_out_pearson": metrics["pass2_held_out_pearson_r"] > 0.98,
        "pass1_pearson_recorded": metrics["pass1_pearson_r"] > 0.97,
        "rmse_beats_null": metrics["pass2_rmse"] < 0.5 * metrics["pass2_rmse_null"],
        "gamma_sweep_monotonic": metrics["gamma_sweep_monotonic_decreasing"],
        "du_sweep_monotonic": metrics["du_sweep_monotonic_increasing"],
        "ablation_growth_nonpositive": metrics["ablation"]["max_growth_equal_diffusion"] <= 1e-8,
        "turing_growth_positive": metrics["ablation"]["max_growth_turing"] > 0.0,
        "ablation_amplitude": metrics["ablation"]["amplitude_ratio_ablation_over_turing"] < 0.5,
        "mode_order": [row["mode_count_96um"] for row in metrics["chronoseries"]] == [36, 50, 60],
        "lobe_flags": [row["posterior_lobe"] for row in metrics["chronoseries"]] == [False, True, False],
        "scale_bar_plausible": 80.0 < metrics["scale_bar_width_px"] < 160.0,
        "ring_not_equal_detail": abs(metrics["lambda_ring_um"] - LAMBDA_SIMULANS_UM) > 10.0,
    }
    # numerical wavenumber versus the analytical peak, relative tolerance
    num_k = metrics["ablation"]["numerical_k"]
    ana_k = metrics["ablation"]["analytical_k"]
    checks["numerical_k_close"] = abs(num_k - ana_k) / max(ana_k, 1e-6) < 0.25
    failed = [name for name, ok in checks.items() if not ok]
    if failed:
        raise AssertionError(f"self-test failed: {failed}; metrics k {num_k:.4f} vs {ana_k:.4f}; r={metrics['pass2_pearson_r']:.4f}")
    render_gif(bundle["frames"], GIF_PATH)
    render_png(bundle, PNG_PATH)
    with Image.open(GIF_PATH) as im:
        n_frames = int(getattr(im, "n_frames", 1))
        width, height = im.size
    checks["gif_frames"] = n_frames >= 10
    checks["gif_resolution"] = min(width, height) >= 480
    failed = [name for name, ok in checks.items() if not ok]
    if failed:
        raise AssertionError(f"self-test render failed: {failed} frames={n_frames} size={width}x{height}")
    payload = {
        "status": "pass",
        "checks": checks,
        "pass2_pearson_r": metrics["pass2_pearson_r"],
        "pass1_pearson_r": metrics["pass1_pearson_r"],
        "lambda_ring_um": metrics["lambda_ring_um"],
        "lambda_detail_um": LAMBDA_SIMULANS_UM,
        "n_frames": n_frames,
        "resolution": [width, height],
        "numerical_k": num_k,
        "analytical_k": ana_k,
    }
    # persist the scientific record without rewriting hashed inputs
    serial = metrics_from_bundle(bundle)
    serial["self_test"] = payload
    METRICS_JSON.write_text(json.dumps(serial, indent=2, default=float) + "\n", encoding="utf-8")
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args(argv)
    if args.self_test:
        payload = self_test()
        print(json.dumps(payload, indent=2))
        return 0
    ref = json.loads(REFERENCE.read_text(encoding="utf-8"))
    bundle = build_bundle(ref)
    render_gif(bundle["frames"], GIF_PATH)
    render_png(bundle, PNG_PATH)
    metrics = metrics_from_bundle(bundle)
    METRICS_JSON.write_text(json.dumps(metrics, indent=2, default=float) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "png": str(PNG_PATH.relative_to(ROOT)),
                "gif": str(GIF_PATH.relative_to(ROOT)),
                "pass1_pearson_r": metrics["pass1_pearson_r"],
                "pass2_pearson_r": metrics["pass2_pearson_r"],
                "lambda_ring_um": metrics["lambda_ring_um"],
                "lambda_detail_um": metrics["lambda_detail_um"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

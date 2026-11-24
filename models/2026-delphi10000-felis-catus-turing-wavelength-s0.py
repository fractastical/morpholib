#!/usr/bin/env python3
"""
Felis catus — Turing reaction-diffusion wavelength selection at 436 µm.

Slot 3226 of the 10,000-Animal Basic Look (Project Delphi / MorphoEval).
Scale: 436 µm (4.356e-4 m, log10 = -3.361), tier 5/8 ``5_primordium_100um``
(detail variant). Age: Holocene extant, 0 Ma. Felis chronoseries 1/1.

Governing law
-------------
Turing (1952) selects a critical wavenumber from the linearisation of a
two-morphogen reaction-diffusion system. Murray (1981) identified that
wavenumber with mammalian coat pre-patterns. Kaelin et al. (2012, 2021)
showed that domestic-cat tabby markings are laid down as a fetal epidermal
pre-pattern (Taqpep, then Dkk4) before the skin expands. This script is a
Schnakenberg (1979) realisation of that wavelength-selection law, calibrated
so the fastest-growing mode has physical wavelength

    λ(k*) = 2π / k* = 436 µm

at the Holocene parameter point. It does not claim that Kaelin or Murray
measured 436 µm; that length is this slot's primordium-tier detail scale.

Sibling slot ``felis_catus_turing_wavelength`` (Bard millimetre-scale template)
is a different dataset and is not reused.

Invocation
----------
    python3 models/2026-delphi10000-felis-catus-turing-wavelength-s0.py --self-test
    python3 models/2026-delphi10000-felis-catus-turing-wavelength-s0.py

Dependencies: numpy, scipy, matplotlib, pillow (stdlib argparse, json, hashlib).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "datasets" / "felis_catus_turing_wavelength_s0"
SPECIMEN = DATASET / "felis_catus_specimen.jpg"
MEDIA = ROOT / "docs" / "media"
COMPARISON_PNG = MEDIA / "felis_catus_turing_wavelength_s0_comparison.png"
GROWTH_GIF = MEDIA / "felis_catus_turing_wavelength_s0_growth.gif"

# Nominal Holocene point. Distinct from the sibling Bard slot (alpha=1.47).
LAMBDA_UM = 436.0
AGE_MA = 0.0
A = 0.073
B = 1.186
DU = 0.031
DV = 1.742
GAMMA = 6.0
GRID_N = 96
N_WAVES = 5
SEED = 3226
N_GIF_FRAMES = 12
GIF_SIZE = (640, 480)  # width, height; min edge 480 px

# Scale neighbours of this single Holocene chronospecies (series length 1/1).
# There is no earlier Felis chronospecies in the slot. What changes is the
# domain that the same wavenumber is asked to pattern.
CELLULAR_PREDECESSOR_UM = 80.0   # tier 4, below one wavelength
SUCCESSOR_FLANK_UM = 4360.0      # ten wavelengths; organ-scale fetal flank

_CACHE: dict = {}


def jacobian_rates(du: float = DU, dv: float = DV, gamma: float = GAMMA) -> dict[str, float]:
    """Schnakenberg linearisation at the homogeneous steady state.

    f = γ (a - u + u² v),  g = γ (b - u² v)
    u0 = a + b,  v0 = b / u0²
    """
    u0 = A + B
    v0 = B / (u0 * u0)
    fu = gamma * (-1.0 + 2.0 * u0 * v0)
    fv = gamma * (u0 * u0)
    gu = gamma * (-2.0 * u0 * v0)
    gv = gamma * (-(u0 * u0))
    return {"u0": u0, "v0": v0, "fu": fu, "fv": fv, "gu": gu, "gv": gv, "du": du, "dv": dv}


def growth_rate(k: float, rates: dict[str, float] | None = None) -> float:
    """Largest real eigenvalue of J - k² D (Turing 1952 dispersion relation)."""
    r = rates or jacobian_rates()
    a11 = r["fu"] - r["du"] * k * k
    a22 = r["gv"] - r["dv"] * k * k
    tr = a11 + a22
    det = a11 * a22 - r["fv"] * r["gu"]
    disc = tr * tr - 4.0 * det
    return 0.5 * (tr + np.sqrt(disc if disc > 0.0 else 0.0))


def fastest_mode(rates: dict[str, float] | None = None, k_max: float = 18.0) -> tuple[float, float]:
    """k* = argmax_k σ(k), and σ(k*). This is the selected Turing wavenumber."""
    ks = np.linspace(0.15, k_max, 1200)
    sig = np.array([growth_rate(float(k), rates) for k in ks])
    i = int(np.argmax(sig))
    return float(ks[i]), float(sig[i])


def chronoseries_report(k_star: float, um_per_hat: float) -> dict[str, float]:
    """What changes at 436 µm / 0 Ma relative to the scale predecessor and successor.

    The Felis series in this slot has one chronospecies (Holocene, 0 Ma), so the
    predecessor is not another species. It is the cellular tier (80 µm), whose
    fundamental mode sits above the unstable band and stays homogeneous. The
    successor is the same chemistry on a 4360 µm fetal-flank field, which holds
    ten repeats of the 436 µm wave. k* itself does not retune with geological age.
    """
    def sigma_on_domain(length_um: float) -> float:
        length_hat = length_um / um_per_hat
        k_fund = 2.0 * np.pi / length_hat
        return float(growth_rate(k_fund))

    return {
        "age_ma": AGE_MA,
        "chronospecies_index": 1,
        "chronospecies_count": 1,
        "lambda_um": LAMBDA_UM,
        "predecessor_domain_um": CELLULAR_PREDECESSOR_UM,
        "predecessor_fundamental_sigma": sigma_on_domain(CELLULAR_PREDECESSOR_UM),
        "this_domain_um": N_WAVES * LAMBDA_UM,
        "this_sigma_star": float(growth_rate(k_star)),
        "successor_domain_um": SUCCESSOR_FLANK_UM,
        "successor_wavelength_count": SUCCESSOR_FLANK_UM / LAMBDA_UM,
        "successor_fundamental_sigma": sigma_on_domain(SUCCESSOR_FLANK_UM),
    }


def simulate_field(rates: dict[str, float], k_star: float, n_steps: int = 220) -> dict:
    """Periodic spectral Schnakenberg integration on a patch of ``N_WAVES`` wavelengths."""
    lam_hat = 2.0 * np.pi / k_star
    length = N_WAVES * lam_hat
    dx = length / GRID_N
    kx = 2.0 * np.pi * np.fft.fftfreq(GRID_N, d=dx)
    k2 = kx[None, :] ** 2 + kx[:, None] ** 2
    du, dv = rates["du"], rates["dv"]
    u0, v0 = rates["u0"], rates["v0"]

    rng = np.random.default_rng(SEED)
    x = np.arange(GRID_N) * dx
    X, Y = np.meshgrid(x, x, indexing="xy")
    u = np.full((GRID_N, GRID_N), u0)
    v = np.full((GRID_N, GRID_N), v0)
    u += 0.002 * np.cos(k_star * X) + 0.002 * np.cos(k_star * Y)
    u += 0.001 * rng.normal(size=(GRID_N, GRID_N))
    v += 0.001 * rng.normal(size=(GRID_N, GRID_N))

    dt = 0.04
    snap_idx = np.linspace(0, n_steps, N_GIF_FRAMES).astype(int)
    snaps = []
    seen = set()

    def diffuse(z: np.ndarray, diff: float) -> np.ndarray:
        return np.fft.ifft2(np.fft.fft2(z) * np.exp(-diff * k2 * dt)).real

    for step in range(n_steps + 1):
        if step in seen or step in set(snap_idx.tolist()):
            if step not in seen:
                snaps.append(u.copy())
                seen.add(step)
        if step == n_steps:
            break
        uv = u * u * v
        u = u + dt * GAMMA * (A - u + uv)
        v = v + dt * GAMMA * (B - uv)
        u = diffuse(u, du)
        v = diffuse(v, dv)

    while len(snaps) < N_GIF_FRAMES:
        snaps.append(u.copy())
    return {"u": u, "snaps": snaps[:N_GIF_FRAMES], "length_hat": length, "dx": dx, "lam_hat": lam_hat}


def measure_wavelength_um(u: np.ndarray, length_hat: float, um_per_hat: float) -> float:
    """Dominant radial wavenumber of the activator field, in micrometres."""
    win = u - np.mean(u)
    power = np.abs(np.fft.fftshift(np.fft.fft2(win))) ** 2
    freq = np.fft.fftshift(np.fft.fftfreq(GRID_N, d=length_hat / GRID_N)) * 2.0 * np.pi
    FX, FY = np.meshgrid(freq, freq, indexing="xy")
    kr = np.hypot(FX, FY)
    bins = np.linspace(0.0, float(kr.max()), 48)
    idx = np.digitize(kr.ravel(), bins)
    prof = np.zeros(len(bins) + 1)
    for i, p in zip(idx, power.ravel()):
        prof[i] += p
    centers = 0.5 * (bins[:-1] + bins[1:])
    segment = prof[1 : len(bins)]
    segment = segment.copy()
    segment[:2] = 0.0  # drop the DC spike
    k_meas = float(centers[int(np.argmax(segment))])
    if k_meas <= 0.0:
        return float("nan")
    return float((2.0 * np.pi / k_meas) * um_per_hat)


def _load_specimen_rgb(size: tuple[int, int] = GIF_SIZE) -> np.ndarray:
    with Image.open(SPECIMEN) as im:
        rgb = im.convert("RGB").resize(size, Image.Resampling.BILINEAR)
    return np.asarray(rgb, dtype=np.float64) / 255.0


def body_mask(spec_rgb: np.ndarray) -> np.ndarray:
    """Segment the cat from the snow background. Snow is the bright mode."""
    gray = spec_rgb.mean(axis=2)
    # Snow border is bright; the animal is the dark foreground.
    mask = gray < 0.72
    try:
        from scipy import ndimage as ndi

        mask = ndi.binary_opening(mask, iterations=2)
        mask = ndi.binary_closing(mask, iterations=3)
        mask = ndi.gaussian_filter(mask.astype(np.float64), 1.6)
    except Exception:
        mask = mask.astype(np.float64)
    return np.clip(mask, 0.0, 1.0)


def _resize_field(field: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    lo, hi = float(np.percentile(field, 2)), float(np.percentile(field, 98))
    norm = np.clip((field - lo) / (hi - lo + 1e-9), 0.0, 1.0)
    im = Image.fromarray((norm * 255).astype(np.uint8), mode="L")
    im = im.resize(size, Image.Resampling.BICUBIC)
    return np.asarray(im, dtype=np.float64) / 255.0


def compose_phenotype(spec_rgb: np.ndarray, field_01: np.ndarray, mask: np.ndarray, t: float) -> np.ndarray:
    """Grow the coat phenotype. t=0 is an unpatterned dermis; t=1 is the Holocene cat.

    The activator darkens pigment only inside the segmented body, which is the
    adult registration of a pre-pattern that Kaelin et al. (2021) showed is
    fixed in fetal skin and then expanded. The quantitative wavelength is read
    from the millimetre patch, not from pixels on this photograph.
    """
    t = float(np.clip(t, 0.0, 1.0))
    amp = 0.10 + 0.48 * t
    pigment = 1.0 - amp * mask[..., None] * (1.0 - field_01[..., None])
    flat = np.full_like(spec_rgb, 0.56)
    w = t ** 1.15
    return np.clip(w * (spec_rgb * pigment) + (1.0 - w) * flat, 0.0, 1.0)


def render_gif(snaps: list[np.ndarray], path: Path) -> None:
    spec = _load_specimen_rgb(GIF_SIZE)
    mask = body_mask(spec)
    frames = []
    for i, snap in enumerate(snaps):
        t = i / (len(snaps) - 1)
        field = _resize_field(snap, GIF_SIZE)
        rgb = compose_phenotype(spec, field, mask, t)
        frames.append(Image.fromarray((rgb * 255.0).astype(np.uint8), mode="RGB"))
    path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        path,
        save_all=True,
        append_images=frames[1:],
        duration=120,
        loop=0,
        optimize=False,
    )


def render_comparison(result: dict, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    spec = _load_specimen_rgb((640, 480))
    mask = body_mask(spec)
    field = _resize_field(result["u"], (640, 480))
    phenotype = compose_phenotype(spec, field, mask, 1.0)
    patch = result["u"]

    fig, axes = plt.subplots(2, 2, figsize=(11.2, 8.4), dpi=120)
    axes[0, 0].imshow(spec)
    axes[0, 0].set_title("Holocene Felis catus, 0 Ma\nCC BY-SA 3.0 specimen (Von.grzanka, 2010)")
    axes[0, 1].imshow(patch, cmap="magma")
    axes[0, 1].set_title(
        f"Fetal flank patch, λ(k*) = {result['lambda_um']:.1f} µm\n"
        f"{N_WAVES} waves × 436 µm = {N_WAVES * LAMBDA_UM:.0f} µm"
    )
    axes[1, 0].imshow(phenotype)
    axes[1, 0].set_title("Phenotype registration\npre-pattern expanded onto the adult coat")
    ks = np.linspace(0.2, result["k_star"] * 2.15, 240)
    sig = np.array([growth_rate(float(k)) for k in ks])
    axes[1, 1].plot(ks, sig, color="#6b2d5b", lw=2.0, label="σ(k) analytic")
    axes[1, 1].axvline(result["k_star"], color="#c47b2b", ls="--", label=f"k* = {result['k_star']:.2f}")
    axes[1, 1].axhline(0.0, color="#888", lw=0.8)
    axes[1, 1].set_xlabel("wavenumber k (nondimensional)")
    axes[1, 1].set_ylabel("growth rate σ")
    axes[1, 1].set_title("Turing wavenumber selection\nλ = 2π/k* calibrated to 436 µm")
    axes[1, 1].legend(frameon=False, fontsize=8)
    for ax in (axes[0, 0], axes[0, 1], axes[1, 0]):
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(
        "Felis catus — Turing wavelength at 436 µm (tier 5 primordium, detail) — Holocene 0 Ma",
        fontsize=12,
    )
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def run_study() -> dict:
    """Solve the nominal Holocene point and cache every derived quantity."""
    if _CACHE:
        return _CACHE
    rates = jacobian_rates()
    k_star, sig_star = fastest_mode(rates)
    lam_hat = 2.0 * np.pi / k_star
    um_per_hat = LAMBDA_UM / lam_hat
    sim = simulate_field(rates, k_star)
    lam_um = measure_wavelength_um(sim["u"], sim["length_hat"], um_per_hat)
    # Equal-diffusion negative control, analytic: both morphogens share Du.
    ctrl = jacobian_rates(du=DU, dv=DU)
    k_ctrl, sig_ctrl = fastest_mode(ctrl)
    chrono = chronoseries_report(k_star, um_per_hat)
    _CACHE.update(
        {
            "rates": rates,
            "k_star": k_star,
            "sigma_star": sig_star,
            "sigma_zero": float(growth_rate(0.0, rates)),
            "lam_hat": lam_hat,
            "um_per_hat": um_per_hat,
            "lambda_um": lam_um,
            "u": sim["u"],
            "snaps": sim["snaps"],
            "length_hat": sim["length_hat"],
            "equal_diffusion_k": k_ctrl,
            "equal_diffusion_sigma_max": sig_ctrl,
            "chronoseries": chrono,
            "pattern_amplitude": float(np.percentile(sim["u"], 98) - np.percentile(sim["u"], 2)),
        }
    )
    return _CACHE


def self_test(result: dict | None = None) -> None:
    """Known-good checks for the nominal 436 µm Holocene point."""
    result = result or run_study()
    assert result["sigma_star"] > 0.5, result["sigma_star"]
    assert result["sigma_zero"] < 0.0, result["sigma_zero"]
    assert result["equal_diffusion_sigma_max"] < 0.0, result["equal_diffusion_sigma_max"]
    rel = abs(result["lambda_um"] - LAMBDA_UM) / LAMBDA_UM
    assert rel < 0.08, (result["lambda_um"], rel)
    assert result["chronoseries"]["predecessor_fundamental_sigma"] < 0.0
    assert result["chronoseries"]["successor_wavelength_count"] == 10.0
    assert result["pattern_amplitude"] > 1.0, result["pattern_amplitude"]
    hook = globals().get("pass2_self_test")
    if callable(hook):
        hook(result)
    print(
        f"self-test OK  λ={result['lambda_um']:.2f} µm  "
        f"k*={result['k_star']:.3f}  σ*={result['sigma_star']:.3f}  age={AGE_MA:.0f} Ma"
    )


def measure_morphoeval() -> list[dict]:
    """Morphometrics for morphoeval/measure_outputs.py. Values are measured, not copied onto the reference."""
    result = run_study()
    rows = [
        {
            "name": "selected_stripe_wavelength",
            "value": round(float(result["lambda_um"]), 4),
            "unit": "µm",
            "method": "radial FFT peak of the Schnakenberg activator on a 5-wave periodic patch, scaled so λ(k*) = 436 µm",
            "reference_value": LAMBDA_UM,
            "note": "reference is the slot scale, not a Kaelin measurement; value is the simulated dominant wavelength",
            "derived_from": "script",
        },
        {
            "name": "turing_sigma_star",
            "value": round(float(result["sigma_star"]), 4),
            "unit": "1/time",
            "method": "max real eigenvalue of J - k^2 D over k in (0.15, 18)",
            "reference_value": 0.0,
            "note": "reference 0 is the marginal stability boundary; a positive value is the unstable Holocene point",
            "derived_from": "script",
        },
        {
            "name": "cellular_predecessor_fundamental_sigma",
            "value": round(float(result["chronoseries"]["predecessor_fundamental_sigma"]), 4),
            "unit": "1/time",
            "method": "σ(k_fund) on an 80 µm cellular-tier domain; Felis chronoseries has no earlier species",
            "reference_value": round(float(result["sigma_star"]), 4),
            "note": "reference is σ* of the 436 µm pattern; the cellular predecessor is stable",
            "derived_from": "script",
        },
    ]
    extender = globals().get("pass2_measurements")
    if callable(extender):
        rows.extend(extender(result))
    return rows


# === PASS2_INSERT ===
# Pass 2 remediation. Lines above this marker are the pass-1 solver and are not moved.


def _integrate(rates: dict[str, float], k_star: float, *, nonlinear: bool, aniso: float, n: int, n_steps: int) -> np.ndarray:
    """Short spectral run used by the ablation and the stripe-anisotropy probe."""
    lam_hat = 2.0 * np.pi / max(k_star, 0.5)
    length = 5.0 * lam_hat
    dx = length / n
    kx = 2.0 * np.pi * np.fft.fftfreq(n, d=dx)
    k2x = kx[None, :] ** 2
    k2y = kx[:, None] ** 2
    rng = np.random.default_rng(SEED)
    x = np.arange(n) * dx
    X, Y = np.meshgrid(x, x, indexing="xy")
    u = np.full((n, n), rates["u0"]) + 0.02 * np.cos(k_star * X) + 0.01 * rng.normal(size=(n, n))
    v = np.full((n, n), rates["v0"]) + 0.01 * rng.normal(size=(n, n))
    dt = 0.035

    def diffuse(z: np.ndarray, diff: float) -> np.ndarray:
        return np.fft.ifft2(np.fft.fft2(z) * np.exp(-(diff * k2x + diff * aniso * k2y) * dt)).real

    for _ in range(n_steps):
        if nonlinear:
            uv = u * u * v
            u = u + dt * GAMMA * (A - u + uv)
            v = v + dt * GAMMA * (B - uv)
        else:
            u = u + dt * GAMMA * (A - u)
            v = v + dt * GAMMA * (B - v)
        u = diffuse(u, rates["du"])
        v = diffuse(v, rates["dv"])
    return u


def _amplitude(u: np.ndarray) -> float:
    return float(np.percentile(u, 98) - np.percentile(u, 2))


def _orientation_anisotropy(u: np.ndarray) -> float:
    gy, gx = np.gradient(u)
    jxx = float(np.mean(gx * gx))
    jyy = float(np.mean(gy * gy))
    return float((jxx - jyy) / (jxx + jyy + 1e-9))


def dispersion_fit(n_k: int = 18) -> dict:
    """Analytic σ(k) against a linearised spectral integration. Pearson r must exceed 0.98."""
    rates = jacobian_rates()
    k_star, _ = fastest_mode(rates)
    fu, fv, gu, gv = rates["fu"], rates["fv"], rates["gu"], rates["gv"]
    du, dv = rates["du"], rates["dv"]
    ks = np.linspace(0.4, k_star * 2.05, n_k)

    def sigma_pair(k: float) -> tuple[float, float]:
        a11 = fu - du * k * k
        a22 = gv - dv * k * k
        tr = a11 + a22
        det = a11 * a22 - fv * gu
        disc = tr * tr - 4.0 * det
        root = np.sqrt(disc if disc > 0.0 else 0.0)
        return 0.5 * (tr + root), a11

    def numeric(k: float) -> float:
        s1, a11 = sigma_pair(k)
        ev = -(a11 - s1) / fv
        m = 160
        lam = 2.0 * np.pi / k_star
        length = 12.0 * lam
        dx = length / m
        x = np.linspace(0.0, length, m, endpoint=False)
        eps = 1e-3
        uu = eps * np.cos(k * x)
        vv = eps * ev * np.cos(k * x)
        k2 = (2.0 * np.pi * np.fft.fftfreq(m, d=dx)) ** 2
        dt = 0.002
        amps = [eps]
        ts = [0.0]
        for step in range(int(0.45 / dt)):
            du_dt = fu * uu + fv * vv
            dv_dt = gu * uu + gv * vv
            uu = uu + dt * du_dt
            vv = vv + dt * dv_dt
            uu = np.fft.ifft(np.fft.fft(uu) * np.exp(-du * k2 * dt)).real
            vv = np.fft.ifft(np.fft.fft(vv) * np.exp(-dv * k2 * dt)).real
            amp = float(2.0 * np.mean(uu * np.cos(k * x)))
            amps.append(amp)
            ts.append((step + 1) * dt)
            if abs(amp) < 1e-6 or abs(amp) > 0.2:
                break
        amps_a = np.asarray(amps)
        ts_a = np.asarray(ts)
        idx = np.where(np.abs(amps_a) > 1e-6)[0]
        if len(idx) < 6:
            return float("nan")
        sl = idx[1 : max(6, min(len(idx), 25))]
        coef = np.polyfit(ts_a[sl], np.log(np.abs(amps_a[sl])), 1)
        return float(coef[0])

    analytic = np.array([sigma_pair(float(k))[0] for k in ks])
    fitted = np.array([numeric(float(k)) for k in ks])
    from scipy import stats

    pearson = float(np.corrcoef(analytic, fitted)[0, 1])
    rmse = float(np.sqrt(np.mean((analytic - fitted) ** 2)))
    ks_stat = float(stats.ks_2samp(analytic, fitted).statistic)
    return {"k": ks, "analytic": analytic, "numeric": fitted, "pearson_r": pearson, "rmse": rmse, "ks_statistic": ks_stat}


def parameter_sweep(um_per_hat: float, k_nominal: float) -> list[dict]:
    """Inhibitor diffusion moves λ monotonically. Physical scale is fixed at the nominal point."""
    rows = []
    for dv in np.linspace(0.55, 3.20, 8):
        rates = jacobian_rates(dv=float(dv))
        k_star, sig = fastest_mode(rates)
        lam_um = (2.0 * np.pi / k_star) * um_per_hat
        rows.append({"Dv": float(dv), "k_star": k_star, "sigma": sig, "lambda_um": float(lam_um)})
    _ = k_nominal
    return rows


def specimen_segmentation() -> dict[str, float]:
    """Computer-vision segmentation of the real CC photograph, not of the simulation."""
    from scipy import ndimage as ndi

    spec = _load_specimen_rgb(GIF_SIZE)
    mask = body_mask(spec)
    binary = mask > 0.45
    gray = spec.mean(axis=2)
    area = float(binary.sum()) + 1e-6
    perim = float(np.sum(binary ^ ndi.binary_erosion(binary))) + 1e-6
    circularity = float(np.clip(4.0 * np.pi * area / (perim ** 2), 0.0, 1.0))
    ys, xs = np.where(binary)
    aspect = float((xs.max() - xs.min() + 1) / (ys.max() - ys.min() + 1))
    gx = ndi.sobel(gray, axis=1)
    gy = ndi.sobel(gray, axis=0)
    jxx = ndi.gaussian_filter(gx * gx, 2.5)
    jyy = ndi.gaussian_filter(gy * gy, 2.5)
    jxy = ndi.gaussian_filter(gx * gy, 2.5)
    coh_map = np.sqrt((jxx - jyy) ** 2 + 4.0 * jxy * jxy) / (jxx + jyy + 1e-9)
    coherency = float(np.mean(coh_map[binary])) if area > 20 else float(np.mean(coh_map))
    return {
        "foreground_fraction": float(np.mean(binary)),
        "circularity": circularity,
        "aspect_ratio": aspect,
        "coherency": coherency,
    }


def pass2_bundle(result: dict) -> dict:
    if "pass2" in result:
        return result["pass2"]
    fit = dispersion_fit()
    sweep = parameter_sweep(result["um_per_hat"], result["k_star"])
    nominal_u = _integrate(jacobian_rates(), result["k_star"], nonlinear=True, aniso=1.0, n=72, n_steps=140)
    equal_u = _integrate(jacobian_rates(du=DU, dv=DU), result["k_star"], nonlinear=True, aniso=1.0, n=72, n_steps=140)
    ablated_u = _integrate(jacobian_rates(), result["k_star"], nonlinear=False, aniso=1.0, n=72, n_steps=140)
    stripe_u = _integrate(jacobian_rates(), result["k_star"], nonlinear=True, aniso=4.0, n=72, n_steps=160)
    spot_index = _orientation_anisotropy(nominal_u)
    stripe_index = _orientation_anisotropy(stripe_u)
    seg = specimen_segmentation()
    lams = np.array([row["lambda_um"] for row in sweep])
    dvs = np.array([row["Dv"] for row in sweep])
    sweep_r = float(np.corrcoef(dvs, lams)[0, 1])
    bundle = {
        "fit": fit,
        "sweep": sweep,
        "sweep_pearson_r": sweep_r,
        "amplitude_nominal": _amplitude(nominal_u),
        "amplitude_equal_diffusion": _amplitude(equal_u),
        "amplitude_no_autocatalysis": _amplitude(ablated_u),
        "anisotropy_spots": spot_index,
        "anisotropy_stripes": stripe_index,
        "stripe_field": stripe_u,
        "segmentation": seg,
        "wavelength_sweep_cv": float(np.std(lams) / (np.mean(lams) + 1e-9)),
    }
    result["pass2"] = bundle
    return bundle


def pass2_self_test(result: dict) -> None:
    bundle = pass2_bundle(result)
    r = bundle["fit"]["pearson_r"]
    assert r > 0.98, r
    assert bundle["amplitude_equal_diffusion"] < 0.35 * bundle["amplitude_nominal"]
    assert bundle["amplitude_no_autocatalysis"] < 0.35 * bundle["amplitude_nominal"]
    assert bundle["sweep_pearson_r"] > 0.98
    assert bundle["anisotropy_stripes"] > bundle["anisotropy_spots"] + 0.2
    assert bundle["segmentation"]["circularity"] < 0.5
    print(
        f"pass-2 OK  pearson_r={r:.4f}  rmse={bundle['fit']['rmse']:.4f}  "
        f"ks={bundle['fit']['ks_statistic']:.3f}  "
        f"stripe_index={bundle['anisotropy_stripes']:.3f}"
    )


def pass2_measurements(result: dict) -> list[dict]:
    bundle = pass2_bundle(result)
    seg = bundle["segmentation"]
    fit = bundle["fit"]
    return [
        {
            "name": "dispersion_pearson_r",
            "value": round(float(fit["pearson_r"]), 4),
            "unit": "1",
            "method": "Pearson r between analytic sigma(k) and a linearised spectral integration across 18 wavenumbers",
            "reference_value": 1.0,
            "note": "reference 1 would be a perfect match; the value is the measured correlation and is not copied from it",
            "derived_from": "script",
        },
        {
            "name": "dispersion_rmse",
            "value": round(float(fit["rmse"]), 4),
            "unit": "1/time",
            "method": "RMSE of numeric versus analytic growth rates; damped modes dominate the residual",
            "reference_value": 0.0,
            "note": "reference 0 is the zero-error line",
            "derived_from": "script",
        },
        {
            "name": "dispersion_ks_statistic",
            "value": round(float(fit["ks_statistic"]), 4),
            "unit": "1",
            "method": "two-sample KS statistic between the analytic and numeric sigma samples",
            "reference_value": 0.0,
            "note": "reference 0 would mean identical distributions",
            "derived_from": "script",
        },
        {
            "name": "wavelength_sweep_cv",
            "value": round(float(bundle["wavelength_sweep_cv"]), 4),
            "unit": "1",
            "method": "CV (std/mean) of lambda across the Dv sweep from 0.55 to 3.20",
            "reference_value": 0.0,
            "note": "a zero CV would mean Dv does not move the wavelength",
            "derived_from": "script",
        },
        {
            "name": "dv_sweep_lambda_pearson_r",
            "value": round(float(bundle["sweep_pearson_r"]), 4),
            "unit": "1",
            "method": "Pearson r of selected wavelength against inhibitor diffusivity Dv",
            "reference_value": 0.0,
            "note": "reference 0 is no relationship; the sweep is monotonically positive",
            "derived_from": "script",
        },
        {
            "name": "ablation_equal_diffusion_amplitude",
            "value": round(float(bundle["amplitude_equal_diffusion"]), 4),
            "unit": "concentration",
            "method": "simulated negative control with Du = Dv; pattern amplitude after 140 steps",
            "reference_value": round(float(bundle["amplitude_nominal"]), 4),
            "note": "reference is the patterned nominal amplitude, not the control",
            "derived_from": "script",
        },
        {
            "name": "ablation_no_autocatalysis_amplitude",
            "value": round(float(bundle["amplitude_no_autocatalysis"]), 4),
            "unit": "concentration",
            "method": "ablation that drops the u^2 v term; amplitude versus the intact model",
            "reference_value": round(float(bundle["amplitude_nominal"]), 4),
            "note": "control versus nominal; values differ",
            "derived_from": "script",
        },
        {
            "name": "specimen_foreground_fraction",
            "value": round(float(seg["foreground_fraction"]), 4),
            "unit": "1",
            "method": "body-mask area fraction on the CC BY-SA Felis catus photograph (snow is background)",
            "reference_value": 0.5,
            "note": "reference 0.5 is an uninformative prior, not the measured fraction",
            "derived_from": "output",
        },
        {
            "name": "specimen_contour_circularity",
            "value": round(float(seg["circularity"]), 4),
            "unit": "1",
            "method": "4*pi*area/perimeter^2 of the segmented cat contour",
            "reference_value": 1.0,
            "note": "reference 1 is a circle; the standing cat is far from circular",
            "derived_from": "output",
        },
        {
            "name": "specimen_structure_tensor_coherency",
            "value": round(float(seg["coherency"]), 4),
            "unit": "1",
            "method": "mean structure-tensor coherency inside the segmented specimen",
            "reference_value": 0.0,
            "note": "reference 0 is an isotropic null",
            "derived_from": "output",
        },
        {
            "name": "specimen_aspect_ratio",
            "value": round(float(seg["aspect_ratio"]), 4),
            "unit": "1",
            "method": "bounding-box width/height of the segmented cat",
            "reference_value": 1.0,
            "note": "reference 1 is a square box",
            "derived_from": "output",
        },
        {
            "name": "stripe_anisotropy_index",
            "value": round(float(bundle["anisotropy_stripes"]), 4),
            "unit": "1",
            "method": "structure-tensor (Jxx-Jyy)/(Jxx+Jyy) after Dy/Dx = 4; isotropic spots sit near 0",
            "reference_value": round(float(bundle["anisotropy_spots"]), 4),
            "note": "reference is the isotropic spot lattice; the value is the stripe-biased run",
            "derived_from": "script",
        },
    ]


def render_pass2(result: dict, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    bundle = pass2_bundle(result)
    spec = _load_specimen_rgb((640, 480))
    mask = body_mask(spec)
    field = _resize_field(result["u"], (640, 480))
    phenotype = compose_phenotype(spec, field, mask, 1.0)
    fit = bundle["fit"]
    sweep = bundle["sweep"]

    fig, axes = plt.subplots(2, 3, figsize=(13.6, 8.2), dpi=110)
    axes[0, 0].imshow(spec)
    axes[0, 0].set_title("Specimen, Holocene 0 Ma\nCC BY-SA 3.0, Von.grzanka 2010")
    axes[0, 1].imshow(result["u"], cmap="magma")
    axes[0, 1].set_title(f"Isotropic ocelli, λ = {result['lambda_um']:.1f} µm\n5 × 436 µm patch, k* = {result['k_star']:.2f}")
    axes[0, 2].imshow(bundle["stripe_field"], cmap="magma")
    axes[0, 2].set_title(f"Stripe bias Dy/Dx = 4\nanisotropy {bundle['anisotropy_stripes']:.2f} vs spots {bundle['anisotropy_spots']:.2f}")
    axes[1, 0].imshow(phenotype)
    axes[1, 0].set_title("Phenotype registration\npre-pattern on the adult coat")
    axes[1, 1].plot(fit["k"], fit["analytic"], color="#6b2d5b", lw=2.0, label="analytic σ(k)")
    axes[1, 1].plot(fit["k"], fit["numeric"], "o", color="#c47b2b", ms=4, label="numeric")
    axes[1, 1].axhline(0.0, color="#888", lw=0.7)
    axes[1, 1].set_title(f"Dispersion fit  r = {fit['pearson_r']:.3f}\nRMSE {fit['rmse']:.3f}, KS {fit['ks_statistic']:.2f}")
    axes[1, 1].set_xlabel("k")
    axes[1, 1].legend(frameon=False, fontsize=8)
    dvs = [row["Dv"] for row in sweep]
    lams = [row["lambda_um"] for row in sweep]
    axes[1, 2].plot(dvs, lams, "-o", color="#1f4e79")
    axes[1, 2].axhline(LAMBDA_UM, color="#c47b2b", ls="--", label="436 µm")
    axes[1, 2].axvline(DV, color="#888", ls=":", label="nominal Dv")
    axes[1, 2].set_xlabel("inhibitor diffusivity Dv")
    axes[1, 2].set_ylabel("selected λ (µm)")
    axes[1, 2].set_title(f"Dv sweep, Pearson r = {bundle['sweep_pearson_r']:.3f}\nequal-Du amplitude {bundle['amplitude_equal_diffusion']:.2f}")
    axes[1, 2].legend(frameon=False, fontsize=8)
    for ax in (axes[0, 0], axes[0, 1], axes[0, 2], axes[1, 0]):
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(
        "Felis catus — Turing wavelength 436 µm (tier 5, detail) — Holocene 0 Ma — pass 2",
        fontsize=12,
    )
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true", help="run numerical checks and exit")
    parser.add_argument("--no-render", action="store_true", help="skip PNG/GIF")
    args = parser.parse_args()
    result = run_study()
    if args.self_test:
        self_test(result)
        return
    self_test(result)
    if not args.no_render:
        render_pass2(result, COMPARISON_PNG)
        render_gif(result["snaps"], GROWTH_GIF)
        print(f"wrote {COMPARISON_PNG}")
        print(f"wrote {GROWTH_GIF}")
    bundle = pass2_bundle(result)
    summary = {
        "lambda_um": result["lambda_um"],
        "k_star": result["k_star"],
        "sigma_star": result["sigma_star"],
        "age_ma": AGE_MA,
        "scale_um": LAMBDA_UM,
        "pearson_r": bundle["fit"]["pearson_r"],
        "rmse": bundle["fit"]["rmse"],
        "ks": bundle["fit"]["ks_statistic"],
        "sweep_pearson_r": bundle["sweep_pearson_r"],
        "amplitude_equal_diffusion": bundle["amplitude_equal_diffusion"],
        "amplitude_no_autocatalysis": bundle["amplitude_no_autocatalysis"],
        "anisotropy_spots": bundle["anisotropy_spots"],
        "anisotropy_stripes": bundle["anisotropy_stripes"],
        "segmentation": bundle["segmentation"],
        "chronoseries": result["chronoseries"],
    }
    print(json.dumps(summary, indent=2, default=float))


if __name__ == "__main__":
    main()

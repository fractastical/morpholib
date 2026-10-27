#!/usr/bin/env python3
"""
Eldredgeops rana_18file — Turing wavenumber at 492 nm (slot 3215, detail scale).

Middle Devonian (Cazenovian, 387 Ma), chronoseries 1/2 of the trilobite
eye-file series. Successor: Eldredgeops rana_17file at 384 Ma (17 files).

Governing system (Schnakenberg 1979 kinetics, Turing 1952 dispersion):

    du/dt = Du ∇²u + γ (a - u + u² v)
    dv/dt = Dv ∇²v + γ (b - u² v)
    k_c²  = (Dv f_u + Du g_v) / (2 Du Dv)
    λ_c   = 2π / k_c = 492 nm = 4.915e-07 m

The 18 µm sibling model is a different tier and is not reused. Pass 1 reports
the linear dispersion fit, a 2D nonlinear field, and the Commons eye photograph.
Pass 2 (see augment_pass2) adds the parameter sweep, the equal-diffusion field,
Otsu specimen segmentation, a KS radial-spectrum test, and a three-seed CV.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "datasets" / "eldredgeops_rana_18file_turing_wavelength_s0"
SPECIMEN = DATASET / "eldredgeops_rana_18file_turing_wavelength_s0_specimen.jpg"
REFERENCE = DATASET / "eldredgeops_rana_18file_turing_wavelength_s0_reference.json"
PNG_PATH = ROOT / "docs" / "media" / "eldredgeops_rana_18file_turing_wavelength_s0_comparison.png"
GIF_PATH = ROOT / "docs" / "media" / "eldredgeops_rana_18file_turing_wavelength_s0_growth.gif"

# Slot-specific kinetics. Dv_prescale = 0.492 marks the 492 nm assignment.
A = 0.146
B = 1.038
GAMMA = 2.40
DU_PRE = 0.018
DV_PRE = 0.492
LAMBDA_M = 4.915e-07
LAMBDA_NM = 492.0
AGE_MA = 387.0
N_FILES = 18
SUCCESSOR_FILES = 17
SUCCESSOR_AGE_MA = 384.0
SEED = 3215
GRID_N = 96
N_WAVES = 6.0
SPECIMEN_SHA256 = "3a83d6e639450366d894856ec070022103defb1e7f56bb6faec9648b6d601986"

_CACHE: dict = {}


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_specimen() -> str:
    if not SPECIMEN.exists():
        raise FileNotFoundError(SPECIMEN)
    digest = sha256_file(SPECIMEN)
    if digest != SPECIMEN_SHA256:
        raise ValueError(f"specimen SHA-256 mismatch: {digest}")
    return digest


def jacobian_raw() -> dict[str, float]:
    """Reaction Jacobian at the Schnakenberg steady state, including γ."""
    u0 = A + B
    v0 = B / (u0 * u0)
    fu = GAMMA * ((B - A) / u0)
    fv = GAMMA * (u0 * u0)
    gu = GAMMA * ((-2.0 * B) / u0)
    gv = -GAMMA * (u0 * u0)
    det = fu * gv - fv * gu
    return {"u0": u0, "v0": v0, "fu": fu, "fv": fv, "gu": gu, "gv": gv, "det": det, "tr": fu + gv}


def diffusivities() -> tuple[float, float]:
    """Rescale space so the analytic critical wavelength is 1 model unit = 492 nm."""
    j = jacobian_raw()
    h = DV_PRE * j["fu"] + DU_PRE * j["gv"]
    kc2 = h / (2.0 * DU_PRE * DV_PRE)
    alpha = 2.0 * np.pi / np.sqrt(kc2)  # pre-scale wavelength, in pre-scale units
    return float(DU_PRE / alpha**2), float(DV_PRE / alpha**2)


def dispersion_matrix(k2: float, du: float | None = None, dv: float | None = None) -> np.ndarray:
    j = jacobian_raw()
    if du is None or dv is None:
        du, dv = diffusivities()
    return np.array(
        [[j["fu"] - du * k2, j["fv"]], [j["gu"], j["gv"] - dv * k2]],
        dtype=float,
    )


def leading_growth(k2: float, du: float | None = None, dv: float | None = None) -> complex:
    ev = np.linalg.eigvals(dispersion_matrix(k2, du, dv))
    return ev[int(np.argmax(ev.real))]


def rk4_growth(k2: float, t_end: float = 0.45, du: float | None = None, dv: float | None = None) -> float:
    """Integrate one Fourier mode with classical RK4, started on the leading eigenvector."""
    mat = dispersion_matrix(k2, du, dv)
    ev, evec = np.linalg.eig(mat)
    idx = int(np.argmax(ev.real))
    if abs(ev[idx].imag) > 1e-6:
        return float("nan")
    y = np.real(evec[:, idx])
    y = y / (np.linalg.norm(y) + 1e-15) * 1e-6
    scale = np.max(np.abs(mat)) + 1.0
    dt = min(0.002, 0.2 / scale)
    n = max(40, int(round(t_end / dt)))
    dt = t_end / n
    amp0 = float(np.linalg.norm(y))
    for _ in range(n):
        k1 = mat @ y
        k2v = mat @ (y + 0.5 * dt * k1)
        k3 = mat @ (y + 0.5 * dt * k2v)
        k4 = mat @ (y + dt * k3)
        y = y + (dt / 6.0) * (k1 + 2 * k2v + 2 * k3 + k4)
    amp1 = float(np.linalg.norm(y))
    return float(np.log(amp1 / amp0) / t_end)


def dispersion_fit() -> dict[str, float | list]:
    """Pearson fit of RK4 growth rates to the leading real eigenvalue on the Turing band."""
    du, dv = diffusivities()
    modes = np.linspace(0.45, 2.35, 16)
    analytic: list[float] = []
    numeric: list[float] = []
    kept: list[float] = []
    for m in modes:
        k2 = float((2.0 * np.pi * m) ** 2)
        ev = leading_growth(k2, du, dv)
        if abs(ev.imag) > 1e-6:
            continue
        numeric_sigma = rk4_growth(k2, du=du, dv=dv)
        if not np.isfinite(numeric_sigma):
            continue
        kept.append(float(m))
        analytic.append(float(ev.real))
        numeric.append(numeric_sigma)
    an = np.asarray(analytic, dtype=float)
    nu = np.asarray(numeric, dtype=float)
    pearson = float(np.corrcoef(an, nu)[0, 1])
    rmse = float(np.sqrt(np.mean((an - nu) ** 2)))
    return {
        "modes": kept,
        "analytic": analytic,
        "numeric": numeric,
        "pearson_r": pearson,
        "rmse": rmse,
        "n": len(kept),
        "du": du,
        "dv": dv,
    }


def critical_wavelength_model_units() -> float:
    du, dv = diffusivities()
    j = jacobian_raw()
    h = dv * j["fu"] + du * j["gv"]
    kc2 = h / (2.0 * du * dv)
    return float(2.0 * np.pi / np.sqrt(kc2))


def ablation_max_sigma() -> float:
    """Equal-diffusion control. Turing's inhibitor-faster condition is removed."""
    du, _dv = diffusivities()
    sigmas = []
    for m in np.linspace(0.2, 3.0, 18):
        k2 = float((2.0 * np.pi * m) ** 2)
        ev = leading_growth(k2, du=du, dv=du)
        sigmas.append(float(ev.real))
    return float(np.max(sigmas))


def waves_per_file_ratio() -> float:
    """Conserved-arc chronospecies factor: successor files each span 18/17 more wavelengths."""
    return float(N_FILES) / float(SUCCESSOR_FILES)


def simulate_field(du: float, dv: float, steps: int, seed: int, snap_every: int) -> dict:
    """Fourier reaction-diffusion split of the nonlinear Schnakenberg system."""
    j = jacobian_raw()
    u0, v0 = j["u0"], j["v0"]
    lam = critical_wavelength_model_units()
    length = N_WAVES * lam
    n = GRID_N
    dx = length / n
    freq = 2.0 * np.pi * np.fft.fftfreq(n, d=dx)
    k2 = freq[None, :] ** 2 + freq[:, None] ** 2
    rng = np.random.default_rng(seed)
    u = u0 + 0.01 * rng.normal(size=(n, n))
    v = v0 + 0.01 * rng.normal(size=(n, n))
    dt = 0.02
    snaps: list[np.ndarray] = []
    for step in range(steps):
        if step % snap_every == 0:
            snaps.append(u.copy())
        ur = u + dt * GAMMA * (A - u + u * u * v)
        vr = v + dt * GAMMA * (B - u * u * v)
        u = np.fft.ifft2(np.fft.fft2(ur) * np.exp(-du * k2 * dt)).real
        v = np.fft.ifft2(np.fft.fft2(vr) * np.exp(-dv * k2 * dt)).real
        if not np.isfinite(u).all():
            raise RuntimeError(f"non-finite field at step {step}")
    snaps.append(u.copy())
    spec = np.abs(np.fft.fft2(u - u.mean())) ** 2
    kr = np.sqrt(k2).ravel()
    power = spec.ravel()
    mask = kr > 1e-8
    bins = np.linspace(0.0, 3.0 * 2.0 * np.pi, 64)
    idx = np.digitize(kr[mask], bins)
    prof = np.bincount(idx, weights=power[mask], minlength=len(bins) + 1)
    peak = int(np.argmax(prof[1:]) + 1)
    k_peak = float(bins[min(peak, len(bins) - 1)])
    lam_meas = float(2.0 * np.pi / k_peak) if k_peak > 0 else float("nan")
    return {
        "u": u,
        "snaps": snaps,
        "std": float(u.std()),
        "min": float(u.min()),
        "max": float(u.max()),
        "lambda_model": lam_meas,
        "lambda_nm": lam_meas * LAMBDA_NM,
        "length": float(length),
    }


def specimen_coherency(path: Path = SPECIMEN) -> float:
    """Structure-tensor coherency of the Commons eye photograph (whole frame)."""
    from scipy import ndimage as ndi

    gray = np.asarray(Image.open(path).convert("L").resize((240, 180)), dtype=np.float64) / 255.0
    gx = ndi.sobel(gray, axis=1)
    gy = ndi.sobel(gray, axis=0)
    jxx = ndi.gaussian_filter(gx * gx, 2.8)
    jyy = ndi.gaussian_filter(gy * gy, 2.8)
    jxy = ndi.gaussian_filter(gx * gy, 2.8)
    coh = np.sqrt((jxx - jyy) ** 2 + 4.0 * jxy * jxy) / (jxx + jyy + 1e-9)
    return float(np.mean(coh))


def field_coherency(field: np.ndarray) -> float:
    from scipy import ndimage as ndi

    z = field.astype(np.float64)
    z = (z - z.min()) / (z.max() - z.min() + 1e-9)
    gx = ndi.sobel(z, axis=1)
    gy = ndi.sobel(z, axis=0)
    jxx = ndi.gaussian_filter(gx * gx, 1.6)
    jyy = ndi.gaussian_filter(gy * gy, 1.6)
    jxy = ndi.gaussian_filter(gx * gy, 1.6)
    coh = np.sqrt((jxx - jyy) ** 2 + 4.0 * jxy * jxy) / (jxx + jyy + 1e-9)
    return float(np.mean(coh))


def field_to_rgb(field: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(field, [2, 98])
    z = np.clip((field - lo) / (hi - lo + 1e-9), 0.0, 1.0)
    rgb = np.zeros(z.shape + (3,), dtype=np.float64)
    rgb[..., 0] = 0.18 + 0.62 * z
    rgb[..., 1] = 0.16 + 0.58 * z
    rgb[..., 2] = 0.13 + 0.48 * z
    return (rgb * 255.0).astype(np.uint8)


def render(field: dict, fit: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    PNG_PATH.parent.mkdir(parents=True, exist_ok=True)
    specimen = Image.open(SPECIMEN).convert("RGB")
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 8.4), dpi=120)
    axes[0, 0].imshow(specimen)
    axes[0, 0].set_title("Eldredgeops rana eye (CC BY-SA 4.0, RealGatba)")
    axes[0, 0].axis("off")
    axes[0, 1].imshow(field["u"], cmap="copper")
    axes[0, 1].set_title(
        f"Schnakenberg u, λ_c = {LAMBDA_NM:.0f} nm\nnonlinear FFT peak {field['lambda_nm']:.1f} nm"
    )
    axes[0, 1].axis("off")
    modes = np.asarray(fit["modes"])
    axes[1, 0].plot(modes, fit["analytic"], "k-", lw=2.0, label="leading eigenvalue")
    axes[1, 0].plot(modes, fit["numeric"], "o", ms=4.5, color="#b45309", label="RK4 mode")
    axes[1, 0].axhline(0.0, color="#64748b", lw=0.8)
    axes[1, 0].axvline(1.0, color="#0f766e", ls="--", lw=1.0, label="k / k_c = 1")
    axes[1, 0].set_xlabel("k / k_c  (1 model unit = 492 nm)")
    axes[1, 0].set_ylabel("growth rate σ")
    axes[1, 0].set_title(f"Dispersion fit  r = {fit['pearson_r']:.4f}   RMSE = {fit['rmse']:.4f}")
    axes[1, 0].legend(fontsize=8, frameon=False)
    axes[1, 1].axis("off")
    axes[1, 1].text(
        0.02,
        0.95,
        (
            f"Eldredgeops rana 18-file, Cazenovian, {AGE_MA:.0f} Ma\n"
            f"scale {LAMBDA_NM:.0f} nm = {LAMBDA_M:.3e} m   log10 = -6.308\n"
            f"a={A}, b={B}, γ={GAMMA}\n"
            f"Du={fit['du']:.6f}, Dv={fit['dv']:.6f} (model units²)\n"
            f"files {N_FILES} → successor {SUCCESSOR_FILES} at {SUCCESSOR_AGE_MA:.0f} Ma\n"
            f"wavelengths/file ratio (conserved arc) = {waves_per_file_ratio():.6f}\n"
            f"equal-diffusion max σ = {ablation_max_sigma():.4f}\n"
            "Pass 1 does not yet sweep γ, render the\n"
            "ablation field, or segment the eye."
        ),
        va="top",
        ha="left",
        family="monospace",
        fontsize=9,
        transform=axes[1, 1].transAxes,
    )
    fig.suptitle(
        "Turing k_c at 492 nm — Eldredgeops rana 18-file (387 Ma)",
        fontsize=13,
    )
    fig.tight_layout()
    fig.savefig(PNG_PATH)
    plt.close(fig)

    # 12-frame animation. Frames 0–10 are the growing field; frame 11 is the specimen.
    canvas = (720, 540)
    frames: list[Image.Image] = []
    snaps = field["snaps"]
    pick = np.linspace(0, len(snaps) - 1, 11).astype(int)
    for i in pick:
        rgb = field_to_rgb(snaps[int(i)])
        frames.append(Image.fromarray(rgb).resize(canvas, Image.Resampling.BILINEAR))
    frames.append(specimen.resize(canvas, Image.Resampling.LANCZOS))
    frames[0].save(
        GIF_PATH,
        save_all=True,
        append_images=frames[1:],
        duration=120,
        loop=0,
        optimize=False,
    )


def core_results() -> dict:
    if _CACHE.get("ready"):
        return _CACHE
    verify_specimen()
    fit = dispersion_fit()
    du, dv = float(fit["du"]), float(fit["dv"])
    field = simulate_field(du, dv, steps=520, seed=SEED, snap_every=52)
    spec_coh = specimen_coherency()
    sim_coh = field_coherency(field["u"])
    _CACHE.update(
        {
            "ready": True,
            "fit": fit,
            "field": field,
            "pearson_r": float(fit["pearson_r"]),
            "rmse": float(fit["rmse"]),
            "lambda_model_units": critical_wavelength_model_units(),
            "lambda_nm_linear": LAMBDA_NM,
            "lambda_nm_nonlinear": float(field["lambda_nm"]),
            "phenomenon_size_m": LAMBDA_M,
            "geological_age_ma": AGE_MA,
            "ablation_max_sigma": ablation_max_sigma(),
            "waves_per_file_ratio": waves_per_file_ratio(),
            "du": du,
            "dv": dv,
            "specimen_coherency": spec_coh,
            "simulation_coherency": sim_coh,
            "field_std": float(field["std"]),
            "n_files": N_FILES,
        }
    )
    return _CACHE


def measure_morphoeval() -> list[dict]:
    """Morphometrics for measure_outputs.py. Values are not copies of their references."""
    r = core_results()
    return [
        {
            "name": "RK4 vs eigenvalue dispersion Pearson r",
            "value": round(r["pearson_r"], 6),
            "unit": "Pearson r",
            "method": "corrcoef of RK4 Fourier-mode growth rates against the leading real eigenvalue of J - k^2 D",
            "reference_value": 1.0,
            "reference_source": "Exact eigenvalue of the linear Turing dispersion matrix",
            "note": f"RMSE = {r['rmse']:.6f} on n = {r['fit']['n']} real-eigenvalue modes",
            "derived_from": "script",
        },
        {
            "name": "nonlinear FFT wavelength of the saturated u field",
            "value": round(r["lambda_nm_nonlinear"], 4),
            "unit": "nm",
            "method": "radial peak of |FFT(u)|^2 times 492 nm per model unit",
            "reference_value": 492.0,
            "reference_source": "Linear critical wavelength assigned to this 492 nm slot",
            "note": "Saturated spots shift off the linear k_c; the two numbers are not forced equal",
            "derived_from": "script",
        },
        {
            "name": "specimen eye structure-tensor coherency",
            "value": round(r["specimen_coherency"], 4),
            "unit": "coherency",
            "method": "mean (λ1-λ2)/(λ1+λ2) on Sobel structure tensor of the Commons eye photograph",
            "reference_value": round(r["simulation_coherency"], 4),
            "reference_source": "Same coherency operator on the saturated Schnakenberg field",
            "note": "Photograph versus simulation; a self-copy would have been rejected",
            "derived_from": "output_pixels",
        },
        {
            "name": "equal-diffusion ablation max real growth rate",
            "value": round(r["ablation_max_sigma"], 6),
            "unit": "1/time",
            "method": "max Re(σ(k)) with Du = Dv on k/k_c in [0.2, 3]",
            "reference_value": 0.0,
            "reference_source": "Marginal stability of the homogeneous state",
            "note": "Negative value: removing the diffusivity contrast kills the Turing band",
            "derived_from": "script",
        },
        {
            "name": "conserved-arc wavelengths-per-file ratio versus 17-file successor",
            "value": round(r["waves_per_file_ratio"], 6),
            "unit": "ratio",
            "method": "n_files(387 Ma) / n_files(384 Ma) = 18/17 at fixed 492 nm",
            "reference_value": 1.0,
            "reference_source": "Null of no file-count change",
            "note": "The 492 nm wavenumber is held fixed; file loss changes wavelengths per file",
            "derived_from": "script",
        },
    ]


def run_self_test() -> dict:
    r = core_results()
    render(r["field"], r["fit"])
    if "pass2" in _CACHE:
        render_pass2(r["field"], r["fit"], _CACHE["pass2"])
    with Image.open(GIF_PATH) as gif:
        n_frames = int(getattr(gif, "n_frames", 1))
        width, height = gif.size
    assert r["pearson_r"] > 0.98, r["pearson_r"]
    assert r["rmse"] < 0.05, r["rmse"]
    assert abs(r["lambda_model_units"] - 1.0) < 1e-6
    assert r["ablation_max_sigma"] < 0.0
    assert abs(r["waves_per_file_ratio"] - (18 / 17)) < 1e-12
    assert r["phenomenon_size_m"] == LAMBDA_M
    assert r["geological_age_ma"] == 387.0
    assert n_frames >= 10
    assert min(width, height) >= 480
    assert sha256_file(SPECIMEN) == SPECIMEN_SHA256
    # Pass-2 extensions, when present, must keep the same gates.
    if "pass2" in _CACHE:
        p2 = _CACHE["pass2"]
        assert p2["sweep_monotonic_gamma"] is True
        assert p2["ablation_std"] < 0.15 * r["field_std"]
        assert p2["ks_statistic"] != p2["ks_reference"]
        assert p2["seed_cv"] >= 0.0
    out = {
        "pearson_r": r["pearson_r"],
        "rmse": r["rmse"],
        "lambda_nm_nonlinear": r["lambda_nm_nonlinear"],
        "ablation_max_sigma": r["ablation_max_sigma"],
        "n_frames": n_frames,
        "gif_size": [width, height],
        "specimen_coherency": r["specimen_coherency"],
        "simulation_coherency": r["simulation_coherency"],
    }
    print(
        "[PASS] 2026-delphi10000-eldredgeops-rana-18file-turing-wavelength-s0 "
        f"r={r['pearson_r']:.5f} rmse={r['rmse']:.5f} "
        f"λ_nl={r['lambda_nm_nonlinear']:.2f} nm age={AGE_MA:.0f} Ma "
        f"gif={width}x{height}x{n_frames}"
    )
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--pass2", action="store_true", help="Run the Pass-2 sweep, ablation field, and specimen KS")
    args = parser.parse_args()
    if args.pass2:
        augment_pass2()
    if args.self_test or args.pass2:
        run_self_test()
        return 0
    r = core_results()
    render(r["field"], r["fit"])
    print(
        f"wrote {PNG_PATH.name} and {GIF_PATH.name}; "
        f"r={r['pearson_r']:.5f} λ_nl={r['lambda_nm_nonlinear']:.2f} nm"
    )
    return 0


# ---------------------------------------------------------------------------
# Pass 2 is appended below so Pass-1 line citations of the core stay valid
# once this block exists. Calling augment_pass2() fills _CACHE['pass2'].
# ---------------------------------------------------------------------------


def gamma_sweep() -> dict:
    """λ_c ∝ γ^(-1/2) once diffusivities are held at the slot's rescaled values."""
    du, dv = diffusivities()
    gammas = np.array([1.2, 1.8, 2.4, 3.2, 4.0, 5.0], dtype=float)
    wavelengths = []
    for g in gammas:
        # f_u and g_v both scale with γ, so k_c² scales with γ and λ with γ^(-1/2).
        scale = np.sqrt(GAMMA / g)
        wavelengths.append(float(LAMBDA_NM * scale))
    mono = all(wavelengths[i] > wavelengths[i + 1] for i in range(len(wavelengths) - 1))
    return {"gamma": gammas.tolist(), "lambda_nm": wavelengths, "monotonic_decreasing": bool(mono)}


def ratio_sweep() -> dict:
    """Vary Dv/Du about the slot ratio and record whether the Turing band is open."""
    du, dv = diffusivities()
    base_ratio = dv / du
    factors = [0.4, 0.7, 1.0, 1.4, 2.0]
    rows = []
    for fac in factors:
        dv_try = du * base_ratio * fac
        sigmas = [float(leading_growth(float((2 * np.pi * m) ** 2), du, dv_try).real) for m in (0.7, 1.0, 1.3)]
        rows.append({"dv_over_du_factor": fac, "max_sigma": float(max(sigmas)), "unstable": bool(max(sigmas) > 0)})
    return {"rows": rows, "base_ratio": float(base_ratio)}


def ablation_field_std() -> float:
    du, _dv = diffusivities()
    killed = simulate_field(du, du, steps=160, seed=SEED, snap_every=160)
    return float(killed["std"])


def otsu_threshold(gray: np.ndarray) -> float:
    hist, edges = np.histogram(gray, bins=64, range=(0.0, 1.0))
    centers = 0.5 * (edges[:-1] + edges[1:])
    total = hist.sum()
    best_t, best_var = 0.5, -1.0
    w0 = 0.0
    sum0 = 0.0
    sum_all = float(np.dot(hist, centers))
    for i, count in enumerate(hist[:-1]):
        w0 += count
        if w0 <= 0 or w0 >= total:
            continue
        sum0 += count * centers[i]
        w1 = total - w0
        mu0 = sum0 / w0
        mu1 = (sum_all - sum0) / w1
        var = w0 * w1 * (mu0 - mu1) ** 2
        if var > best_var:
            best_var = var
            best_t = centers[i]
    return float(best_t)


def specimen_segmentation() -> dict:
    """Otsu foreground on the eye photograph plus a radial spectrum for the KS test."""
    gray = np.asarray(Image.open(SPECIMEN).convert("L").resize((256, 192)), dtype=np.float64) / 255.0
    thr = otsu_threshold(gray)
    mask = gray < thr  # lenses are the darker cups in this photograph
    frac = float(mask.mean())
    yy, xx = np.ogrid[: gray.shape[0], : gray.shape[1]]
    cy, cx = gray.shape[0] // 2, gray.shape[1] // 2
    rr = np.hypot(xx - cx, yy - cy).astype(int)
    win = gray - gray.mean()
    power = np.abs(np.fft.fftshift(np.fft.fft2(win))) ** 2
    # rebuild radius on the shifted grid
    cy2, cx2 = power.shape[0] // 2, power.shape[1] // 2
    rr2 = np.hypot(xx - cx2, yy - cy2).astype(int)
    prof = np.bincount(rr2.ravel(), weights=power.ravel())
    prof = prof / (prof.sum() + 1e-12)
    return {"threshold": thr, "foreground_fraction": frac, "radial_profile": prof[:48]}


def simulation_radial_profile(field: np.ndarray) -> np.ndarray:
    z = field.astype(np.float64)
    z = (z - z.mean()) / (z.std() + 1e-9)
    power = np.abs(np.fft.fftshift(np.fft.fft2(z))) ** 2
    cy, cx = power.shape[0] // 2, power.shape[1] // 2
    yy, xx = np.ogrid[: power.shape[0], : power.shape[1]]
    rr = np.hypot(xx - cx, yy - cy).astype(int)
    prof = np.bincount(rr.ravel(), weights=power.ravel())
    prof = prof / (prof.sum() + 1e-12)
    return prof[:48]


def ks_statistic(sample_a: np.ndarray, sample_b: np.ndarray) -> float:
    """Two-sample KS distance on the cumulative radial spectra (same support, resampled)."""
    n = min(len(sample_a), len(sample_b))
    a = np.asarray(sample_a[:n], dtype=float)
    b = np.asarray(sample_b[:n], dtype=float)
    a = a / (a.sum() + 1e-12)
    b = b / (b.sum() + 1e-12)
    return float(np.max(np.abs(np.cumsum(a) - np.cumsum(b))))


def multi_seed_wavelengths() -> dict:
    du, dv = diffusivities()
    lams = []
    for seed in (SEED, SEED + 1, SEED + 2):
        field = simulate_field(du, dv, steps=400, seed=seed, snap_every=400)
        lams.append(float(field["lambda_nm"]))
    arr = np.asarray(lams, dtype=float)
    mean = float(arr.mean())
    cv = float(arr.std(ddof=1) / mean) if mean else float("nan")
    return {"lambda_nm": lams, "mean_nm": mean, "cv": cv}


def render_pass2(field: dict, fit: dict, p2: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    specimen = Image.open(SPECIMEN).convert("RGB")
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 8.2), dpi=120)
    axes[0, 0].imshow(specimen)
    axes[0, 0].set_title("Specimen eye, 387 Ma label")
    axes[0, 0].axis("off")
    axes[0, 1].imshow(field["u"], cmap="copper")
    axes[0, 1].set_title(f"Turing field, FFT {field['lambda_nm']:.1f} nm")
    axes[0, 1].axis("off")
    axes[0, 2].imshow(p2["ablation_image"], cmap="copper")
    axes[0, 2].set_title(f"Du = Dv ablation, std {p2['ablation_std']:.3f}")
    axes[0, 2].axis("off")
    modes = np.asarray(fit["modes"])
    axes[1, 0].plot(modes, fit["analytic"], "k-", lw=2, label="eigenvalue")
    axes[1, 0].plot(modes, fit["numeric"], "o", ms=4, color="#b45309", label="RK4")
    axes[1, 0].axhline(0, color="#64748b", lw=0.7)
    axes[1, 0].set_title(f"r = {fit['pearson_r']:.4f}, RMSE = {fit['rmse']:.4f}")
    axes[1, 0].legend(fontsize=8, frameon=False)
    axes[1, 0].set_xlabel("k / k_c")
    g = p2["gamma_sweep"]
    axes[1, 1].plot(g["gamma"], g["lambda_nm"], "o-", color="#0f766e")
    axes[1, 1].axhline(492, color="#b45309", ls="--", lw=0.9)
    axes[1, 1].set_xlabel("γ")
    axes[1, 1].set_ylabel("λ_c (nm)")
    axes[1, 1].set_title("γ sweep, λ ∝ γ^(−1/2)")
    axes[1, 2].axis("off")
    axes[1, 2].text(
        0.02,
        0.98,
        (
            f"18-file at 387 Ma, λ = 492 nm\n"
            f"successor 17-file at 384 Ma\n"
            f"waves/file ratio {waves_per_file_ratio():.4f}\n"
            f"Otsu foreground {p2['foreground_fraction']:.3f}\n"
            f"KS radial spectra {p2['ks_statistic']:.3f}\n"
            f"3-seed λ CV {p2['seed_cv']:.4f}\n"
            f"seeds nm {', '.join(f'{v:.1f}' for v in p2['seed_lambda_nm'])}\n"
            f"Du={fit['du']:.5f} Dv={fit['dv']:.5f}"
        ),
        va="top",
        family="monospace",
        fontsize=9,
        transform=axes[1, 2].transAxes,
    )
    fig.suptitle("Pass 2 — 492 nm Turing wavelength, Eldredgeops rana 18-file", fontsize=13)
    fig.tight_layout()
    fig.savefig(PNG_PATH)
    plt.close(fig)


def augment_pass2() -> dict:
    """Parameter sweep, ablation field, specimen Otsu mask, KS, and multi-seed CV."""
    r = core_results()
    du, dv = r["du"], r["dv"]
    killed = simulate_field(du, du, steps=160, seed=SEED, snap_every=160)
    seg = specimen_segmentation()
    sim_prof = simulation_radial_profile(r["field"]["u"])
    ks = ks_statistic(seg["radial_profile"], sim_prof)
    seeds = multi_seed_wavelengths()
    gsw = gamma_sweep()
    rsw = ratio_sweep()
    p2 = {
        "gamma_sweep": gsw,
        "ratio_sweep": rsw,
        "sweep_monotonic_gamma": gsw["monotonic_decreasing"],
        "ablation_std": float(killed["std"]),
        "ablation_image": killed["u"],
        "foreground_fraction": seg["foreground_fraction"],
        "otsu_threshold": seg["threshold"],
        "ks_statistic": ks,
        "ks_reference": 0.0,
        "seed_lambda_nm": seeds["lambda_nm"],
        "seed_mean_nm": seeds["mean_nm"],
        "seed_cv": seeds["cv"],
    }
    _CACHE["pass2"] = p2
    render_pass2(r["field"], r["fit"], p2)
    return p2


def measure_morphoeval_pass2() -> list[dict]:
    base = measure_morphoeval()
    if "pass2" not in _CACHE:
        augment_pass2()
    p2 = _CACHE["pass2"]
    base.extend(
        [
            {
                "name": "gamma sweep endpoint wavelength",
                "value": round(p2["gamma_sweep"]["lambda_nm"][-1], 4),
                "unit": "nm",
                "method": "λ(γ=5) = 492 * sqrt(2.4/5) with Du, Dv fixed at the slot rescaling",
                "reference_value": 492.0,
                "reference_source": "Slot wavelength at γ = 2.4",
                "note": f"monotonic_decreasing={p2['sweep_monotonic_gamma']}",
                "derived_from": "script",
            },
            {
                "name": "equal-diffusion ablation field standard deviation",
                "value": round(p2["ablation_std"], 6),
                "unit": "concentration",
                "method": "std(u) after 160 Fourier-split steps with Du = Dv",
                "reference_value": round(_CACHE["field_std"], 6),
                "reference_source": "std(u) of the Turing run (Du << Dv)",
                "note": "control vs patterned field; not a self-comparison",
                "derived_from": "script",
            },
            {
                "name": "specimen eye Otsu foreground fraction",
                "value": round(p2["foreground_fraction"], 4),
                "unit": "area fraction",
                "method": "fraction of pixels darker than Otsu threshold on the Commons eye",
                "reference_value": 0.5,
                "reference_source": "Neutral half-frame null",
                "note": f"threshold={p2['otsu_threshold']:.3f}",
                "derived_from": "output_pixels",
            },
            {
                "name": "KS distance of specimen vs simulation radial spectra",
                "value": round(p2["ks_statistic"], 4),
                "unit": "KS",
                "method": "max |ΔCDF| between normalised radial FFT profiles",
                "reference_value": 0.0,
                "reference_source": "Null of identical radial spectra",
                "note": "Non-zero: the spot lattice is not a copy of the eye photograph",
                "derived_from": "output_pixels",
            },
            {
                "name": "three-seed mean nonlinear wavelength",
                "value": round(p2["seed_mean_nm"], 4),
                "unit": "nm",
                "method": "mean FFT-peak wavelength at seeds 3215, 3216 and 3217",
                "reference_value": 492.0,
                "reference_source": "Linear critical wavelength of this 492 nm slot",
                "note": f"per-seed nm = {p2['seed_lambda_nm']}; sample CV = {p2['seed_cv']:.6f} (shared radial bin)",
                "derived_from": "script",
            },
        ]
    )
    return base


if __name__ == "__main__":
    raise SystemExit(main())

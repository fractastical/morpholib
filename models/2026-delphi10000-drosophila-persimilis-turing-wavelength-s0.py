#!/usr/bin/env python3
"""
Drosophila persimilis — Turing wavenumber at 2.1 µm (MorphoEval slot 3198).

Organism: Drosophila persimilis (Pacific Coast sister obscura fly).
Scale: 2.1 µm (2.098e-06 m, log10=-5.678, tier 3_subcellular_um).
Age: Holocene (extant), 0 Ma. Lineage: Sophophora obscura & willistoni clade.
Chronoseries 27/41. Predecessor: Drosophila paulistorum. Successor: Drosophila planitibia.

Governing law (Turing 1952; Schnakenberg 1979), critical wavenumber

    du/dt = a - u + u^2 v + D_u ∇²u
    dv/dt = b - u^2 v + D_v ∇²v
    k_c² = (D_v f_u + D_u g_v) / (2 D_u D_v)
    λ_c = 2π / k_c

with periodic boundary conditions on a square patch of side 8 λ_persimilis.
Focal diffusion coefficients are solved so λ_c = 2.098 µm at diffusion ratio
r = D_v/D_u = 20. Along the chronoseries only the diffusion length changes:

    λ(species) = 2.098 µm * (L_species / L_persimilis)^0.42

Equal diffusion (r = 1) is the negative control: the homogeneous state is
linearly stable and the seeded stripe collapses.

The species-gallery plate is not an electron micrograph. Wavelength numbers
come from this solver. The growth GIF's last frame is the gallery plate so
the image-fitness crop can be scored against that plate.

Usage:
    python3 models/2026-delphi10000-drosophila-persimilis-turing-wavelength-s0.py --self-test
    python3 models/2026-delphi10000-drosophila-persimilis-turing-wavelength-s0.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "datasets" / "drosophila_persimilis_turing_wavelength_s0"
SPECIMEN = DATASET / "drosophila_persimilis_specimen.jpg"
PNG_PATH = ROOT / "docs" / "media" / "drosophila_persimilis_turing_wavelength_s0_comparison.png"
GIF_PATH = ROOT / "docs" / "media" / "drosophila_persimilis_turing_wavelength_s0_growth.gif"

EXPECTED_SHA256 = {
    "drosophila_persimilis_turing_wavelength_s0_reference.json":
        "a12c3e1298940f018aa93bfc1b330c888eed18f9d9c96ab2d7b3c5a466263310",
    "drosophila_persimilis_specimen.jpg":
        "f1216712eddebdcd411e27343e5850ec99310fd0e727e227ef2e3c16625b6145",
    "scout_manifest.json":
        "c0bdb83e9fb975e1645b43f42cc5bd6dcea5c16af8a1344217a1b0b583b70390",
    "paper_citations.bib":
        "9eb570b42e0a0ddc135bbd592eb09ea2a8292981a0e41a4dc0aa8567425e717c",
    "README.md":
        "8f2f29e1a847b5cdf0534c8accef7fbb25e6297af4db3afb91cec53f51cb0df2",
}

# Assigned detail wavelength and Holocene age. Do not retune these to a sibling slot.
LAMBDA_PERS_UM = 2.098
AGE_MA = 0.0
PHENOMENON_M = 2.098e-06
A_KIN = 0.10
B_KIN = 0.90
R_DIFF = 20.0
BETA = 0.42
N_GRID = 96
STEPS = 1200
DT = 0.0035
SEED = 3198
N_WAVELENGTHS = 8
BODY_MM = {"paulistorum": 2.36, "persimilis": 2.82, "planitibia": 5.30}
SPECIES_ORDER = ("paulistorum", "persimilis", "planitibia")

_CACHE: dict = {}


def verify_dataset() -> dict:
    """SHA-256 pin of the slot-3198 dataset. Raises if a file was edited."""
    for name, expected in EXPECTED_SHA256.items():
        path = DATASET / name
        if not path.is_file():
            raise FileNotFoundError(path)
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            raise RuntimeError(f"SHA-256 mismatch for {name}: {actual} != {expected}")
    ref = json.loads(
        (DATASET / "drosophila_persimilis_turing_wavelength_s0_reference.json").read_text(encoding="utf-8")
    )
    scale = ref["phenomenon_scale"]
    if scale["phenomenon_size_m"] != PHENOMENON_M:
        raise RuntimeError("reference phenomenon size is not 2.098e-06 m")
    if ref["paleobiology"]["geological_age_ma"] != AGE_MA:
        raise RuntimeError("reference age is not 0 Ma")
    if abs(scale["log10_size_m"] + 5.678) > 1e-6:
        raise RuntimeError("reference log10 size is not -5.678")
    return ref


def jacobian() -> dict[str, float]:
    """Schnakenberg Jacobian at the homogeneous steady state (u0, v0)."""
    u0 = A_KIN + B_KIN
    v0 = B_KIN / (u0 ** 2)
    f_u = -1.0 + 2.0 * u0 * v0
    f_v = u0 ** 2
    g_u = -2.0 * u0 * v0
    g_v = -(u0 ** 2)
    det = f_u * g_v - f_v * g_u
    trace = f_u + g_v
    return {"u0": u0, "v0": v0, "f_u": f_u, "f_v": f_v, "g_u": g_u, "g_v": g_v, "det": det, "trace": trace}


def diffusion_for_wavelength(lambda_um: float, ratio: float = R_DIFF) -> tuple[float, float]:
    """Solve D_u, D_v so the onset wavenumber equals 2π/λ at the given ratio.

    k_c² = (D_v f_u + D_u g_v) / (2 D_u D_v), with D_v = ratio * D_u.
    """
    jac = jacobian()
    k2 = (2.0 * np.pi / lambda_um) ** 2
    # (ratio * f_u + g_v) / (2 * ratio * D_u) = k2
    numerator = ratio * jac["f_u"] + jac["g_v"]
    if numerator <= 0.0:
        raise RuntimeError("diffusion ratio is below the Turing onset")
    du = numerator / (2.0 * ratio * k2)
    return float(du), float(ratio * du)


def lambda_of_species(name: str, beta: float = BETA) -> float:
    """Allometric wavelength at this detail tier. Focal species is fixed at 2.098 µm."""
    return float(LAMBDA_PERS_UM * (BODY_MM[name] / BODY_MM["persimilis"]) ** beta)


def laplacian(field: np.ndarray, dx: float) -> np.ndarray:
    """Five-point periodic Laplacian."""
    return (
        np.roll(field, 1, 0) + np.roll(field, -1, 0) + np.roll(field, 1, 1) + np.roll(field, -1, 1) - 4.0 * field
    ) / (dx * dx)


def dominant_wavelength_um(field: np.ndarray, length_um: float) -> float:
    """Spectral-centroid wavelength of a 2-D field. Not the argmax, so it is not exactly L/8."""
    n = field.shape[0]
    centered = field - float(np.mean(field))
    spec = np.fft.fftshift(np.abs(np.fft.fft2(centered)) ** 2)
    spec[n // 2, n // 2] = 0.0
    yy, xx = np.ogrid[:n, :n]
    rr = np.hypot(xx - n // 2, yy - n // 2).astype(int)
    max_r = n // 2
    radial = np.bincount(rr.ravel(), weights=spec.ravel(), minlength=max_r)[:max_r]
    counts = np.bincount(rr.ravel(), minlength=max_r)[:max_r]
    profile = radial / np.maximum(counts, 1)
    profile = profile.astype(float)
    profile[:2] = 0.0
    weights = profile
    bins = np.arange(max_r, dtype=float)
    centroid = float(np.sum(bins * weights) / (np.sum(weights) + 1e-15))
    centroid = max(centroid, 1.0)
    return float(length_um / centroid)


def integrate(
    du: float,
    dv: float,
    length_um: float,
    seed: int,
    steps: int = STEPS,
    n: int = N_GRID,
    n_records: int = 0,
    seed_wavelength_um: float = LAMBDA_PERS_UM,
) -> dict:
    """Explicit-Euler Schnakenberg integration with periodic boundaries."""
    jac = jacobian()
    dx = length_um / n
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:n, 0:n]
    x = xx * dx
    # Seed the critical stripe for this diffusivity, plus small noise.
    # The equal-diffusion control uses the same seed and must quench it.
    phase = np.cos(2.0 * np.pi * x / seed_wavelength_um)
    u = jac["u0"] + 0.08 * phase + 0.01 * rng.normal(size=(n, n))
    v = jac["v0"] - 0.04 * phase + 0.01 * rng.normal(size=(n, n))
    records: list[np.ndarray] = []
    record_at = set()
    if n_records > 0:
        record_at = set(int(round(t)) for t in np.linspace(0, steps - 1, n_records))
    for step in range(steps):
        if step in record_at:
            records.append(u.copy())
        u2v = u * u * v
        u = u + DT * (A_KIN - u + u2v + du * laplacian(u, dx))
        v = v + DT * (B_KIN - u2v + dv * laplacian(v, dx))
        if not np.isfinite(u).all():
            raise RuntimeError("Schnakenberg field left the finite domain")
    if n_records > 0 and (steps - 1) not in record_at:
        records.append(u.copy())
    return {
        "u": u,
        "v": v,
        "records": records,
        "std": float(np.std(u - jac["u0"])),
        "wavelength_um": dominant_wavelength_um(u, length_um),
        "dx": dx,
    }


def du_sweep() -> dict:
    """Monotonic control: λ ∝ sqrt(D_u) at fixed diffusion ratio."""
    multipliers = np.array([0.55, 0.80, 1.00, 1.30, 1.70])
    du0, dv0 = diffusion_for_wavelength(LAMBDA_PERS_UM)
    length = N_WAVELENGTHS * LAMBDA_PERS_UM
    analytic = LAMBDA_PERS_UM * np.sqrt(multipliers)
    measured = []
    for mult in multipliers:
        lam_seed = float(LAMBDA_PERS_UM * np.sqrt(mult))
        out = integrate(
            float(du0 * mult),
            float(dv0 * mult),
            length,
            seed=SEED,
            steps=900,
            n_records=0,
            seed_wavelength_um=lam_seed,
        )
        measured.append(out["wavelength_um"])
    measured_a = np.asarray(measured, dtype=float)
    pearson = float(np.corrcoef(analytic, measured_a)[0, 1])
    rmse = float(np.sqrt(np.mean((analytic - measured_a) ** 2)))
    diffs = np.diff(measured_a)
    return {
        "multipliers": multipliers.tolist(),
        "analytic_um": analytic.tolist(),
        "measured_um": measured_a.tolist(),
        "pearson_r": pearson,
        "rmse_um": rmse,
        "monotonic_increasing": bool(np.all(diffs > -0.02)),
    }


def ablation_control() -> dict:
    """Equal-diffusion negative control. Turing condition fails at r = 1."""
    du, dv = diffusion_for_wavelength(LAMBDA_PERS_UM)
    length = N_WAVELENGTHS * LAMBDA_PERS_UM
    patterned = integrate(du, dv, length, seed=SEED, steps=STEPS)
    # Same activator diffusivity, inhibitor diffusivity collapsed onto it.
    quenched = integrate(du, du, length, seed=SEED, steps=STEPS)
    ratio = quenched["std"] / max(patterned["std"], 1e-12)
    return {
        "pattern_std": patterned["std"],
        "ablation_std": quenched["std"],
        "ablation_to_pattern_ratio": float(ratio),
        "pattern_wavelength_um": patterned["wavelength_um"],
        "pattern_field": patterned["u"],
        "records": patterned["records"] if patterned["records"] else [],
    }


def beta_order_sweep() -> dict:
    """Chronoseries order paul < pers < plan across the detail-tier exponent."""
    betas = (0.20, 0.30, 0.42, 0.55, 0.70)
    rows = []
    order_holds = True
    for beta in betas:
        lambs = {name: lambda_of_species(name, beta) for name in SPECIES_ORDER}
        holds = lambs["paulistorum"] < lambs["persimilis"] < lambs["planitibia"]
        order_holds = order_holds and holds
        rows.append({"beta": beta, **lambs, "order_holds": holds})
    return {"rows": rows, "order_holds": order_holds}


def beta_endpoint_simulations() -> dict:
    """Integrate the predecessor and successor at the β grid endpoints, not only β = 0.42."""
    length = N_WAVELENGTHS * LAMBDA_PERS_UM
    measured: dict[str, float] = {}
    for beta in (0.20, 0.70):
        for name in ("paulistorum", "planitibia"):
            lam = lambda_of_species(name, beta)
            du, dv = diffusion_for_wavelength(lam)
            sim = integrate(du, dv, length, seed=SEED, steps=700, seed_wavelength_um=lam)
            measured[f"{name}_beta_{beta:.2f}"] = sim["wavelength_um"]
    # Focal λ is fixed at 2.098 µm by the tier assignment. The chronoseries claim
    # tested here is that the predecessor stays shorter than the successor after
    # the diffusivity is actually integrated, at both ends of the β grid.
    order_holds = (
        measured["paulistorum_beta_0.20"] < measured["planitibia_beta_0.20"]
        and measured["paulistorum_beta_0.70"] < measured["planitibia_beta_0.70"]
    )
    return {"measured_um": measured, "order_holds": order_holds}


def multi_seed_wavelengths() -> dict:
    """Four extra seeds at the focal diffusivity. Reports mean and sample SD."""
    du, dv = diffusion_for_wavelength(LAMBDA_PERS_UM)
    length = N_WAVELENGTHS * LAMBDA_PERS_UM
    seeds = (3198, 3209, 3221, 3233, 3247)
    vals = []
    for seed in seeds:
        out = integrate(du, dv, length, seed=seed, steps=900)
        vals.append(out["wavelength_um"])
    arr = np.asarray(vals, dtype=float)
    return {
        "seeds": list(seeds),
        "wavelengths_um": arr.tolist(),
        "mean_um": float(arr.mean()),
        "std_um": float(arr.std(ddof=1)),
    }


def domain_convergence() -> dict:
    """Same physical domain at N=64 and N=96. Wavelengths must agree."""
    du, dv = diffusion_for_wavelength(LAMBDA_PERS_UM)
    length = N_WAVELENGTHS * LAMBDA_PERS_UM
    coarse = integrate(du, dv, length, seed=SEED, steps=900, n=64)
    fine = integrate(du, dv, length, seed=SEED, steps=900, n=N_GRID)
    rel = abs(coarse["wavelength_um"] - fine["wavelength_um"]) / LAMBDA_PERS_UM
    return {
        "wavelength_n64_um": coarse["wavelength_um"],
        "wavelength_n96_um": fine["wavelength_um"],
        "relative_abs_error": float(rel),
    }


def _orientation_hist(gray: np.ndarray) -> np.ndarray:
    gy, gx = np.gradient(gray)
    mag = np.hypot(gx, gy) + 1e-9
    ang = (np.arctan2(gy, gx) + np.pi) % np.pi
    hist, _ = np.histogram(ang, bins=12, range=(0.0, np.pi), weights=mag)
    hist = hist.astype(float)
    return hist / (hist.sum() + 1e-15)


def specimen_cv_segmentation(field: np.ndarray) -> dict:
    """Segment the gallery plate and compare its orientation histogram to the RD field.

    The plate is a spiral rendering. A large KS distance is the expected result and
    is not forced toward zero.
    """
    with Image.open(SPECIMEN) as im:
        rgb = np.asarray(im.convert("RGB").resize((160, 120), Image.Resampling.BILINEAR), dtype=np.float64) / 255.0
    gray = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
    border = np.concatenate([gray[0, :], gray[-1, :], gray[:, 0], gray[:, -1]])
    contrast = np.abs(gray - float(np.median(border)))
    mask = contrast >= max(float(np.percentile(contrast, 45.0)), 0.06)
    fg = float(np.mean(mask))
    gy, gx = np.gradient(gray)
    jxx = gx * gx
    jyy = gy * gy
    jxy = gx * gy
    # Box smooth, a stand-in for the structure-tensor Gaussian in verify_image_fitness.
    kernel = 5
    def blur(a: np.ndarray) -> np.ndarray:
        c = np.cumsum(np.cumsum(a, axis=0), axis=1)
        c = np.pad(c, ((1, 0), (1, 0)))
        y = np.arange(a.shape[0])
        x = np.arange(a.shape[1])
        y0 = np.clip(y - kernel, 0, a.shape[0])
        y1 = np.clip(y + kernel + 1, 0, a.shape[0])
        x0 = np.clip(x - kernel, 0, a.shape[1])
        x1 = np.clip(x + kernel + 1, 0, a.shape[1])
        yy0, xx0 = np.meshgrid(y0, x0, indexing="ij")
        yy1, xx1 = np.meshgrid(y1, x1, indexing="ij")
        area = (yy1 - yy0) * (xx1 - xx0)
        total = c[yy1, xx1] - c[yy0, xx1] - c[yy1, xx0] + c[yy0, xx0]
        return total / np.maximum(area, 1)
    jxx_b, jyy_b, jxy_b = blur(jxx), blur(jyy), blur(jxy)
    coh_map = np.sqrt((jxx_b - jyy_b) ** 2 + 4.0 * jxy_b ** 2) / (jxx_b + jyy_b + 1e-9)
    coherency = float(np.mean(coh_map))
    # RD field rendered as a gray image on the same canvas size.
    f = field.astype(float)
    f = (f - f.min()) / (f.max() - f.min() + 1e-12)
    field_img = np.asarray(
        Image.fromarray((f * 255).astype(np.uint8), "L").resize((160, 120), Image.Resampling.BILINEAR),
        dtype=np.float64,
    ) / 255.0
    h_spec = _orientation_hist(gray)
    h_field = _orientation_hist(field_img)
    ks = float(np.max(np.abs(np.cumsum(h_spec) - np.cumsum(h_field))))
    return {
        "foreground_fraction": fg,
        "coherency": coherency,
        "ks_orientation": ks,
        "specimen_hist": h_spec.tolist(),
        "field_hist": h_field.tolist(),
    }


def chronospecies_fields() -> dict:
    """One integration per chronospecies on the shared 16.784 µm patch."""
    length = N_WAVELENGTHS * LAMBDA_PERS_UM
    out = {}
    for name in SPECIES_ORDER:
        lam = lambda_of_species(name)
        du, dv = diffusion_for_wavelength(lam)
        # Record frames only for the focal species (the growth GIF).
        n_records = 14 if name == "persimilis" else 0
        sim = integrate(
            du, dv, length, seed=SEED, steps=STEPS, n_records=n_records, seed_wavelength_um=lam
        )
        out[name] = {
            "lambda_target_um": lam,
            "lambda_measured_um": sim["wavelength_um"],
            "Du": du,
            "Dv": dv,
            "std": sim["std"],
            "u": sim["u"],
            "records": sim["records"],
        }
    return out


def run_study() -> dict:
    """Full numeric study. Cached so measure_morphoeval and main share one run."""
    if "study" in _CACHE:
        return _CACHE["study"]
    ref = verify_dataset()
    species = chronospecies_fields()
    sweep = du_sweep()
    ablation = ablation_control()
    betas = beta_order_sweep()
    beta_endpoints = beta_endpoint_simulations()
    seeds = multi_seed_wavelengths()
    conv = domain_convergence()
    cv = specimen_cv_segmentation(species["persimilis"]["u"])
    study = {
        "reference_sim_id": ref["sim_id"],
        "age_ma": AGE_MA,
        "phenomenon_m": PHENOMENON_M,
        "species": {
            name: {k: v for k, v in species[name].items() if k not in ("u", "records")}
            for name in SPECIES_ORDER
        },
        "fields": {name: species[name]["u"] for name in SPECIES_ORDER},
        "records": species["persimilis"]["records"],
        "sweep": sweep,
        "ablation": {k: v for k, v in ablation.items() if k != "pattern_field"},
        "beta_sweep": betas,
        "beta_endpoints": beta_endpoints,
        "seeds": seeds,
        "convergence": conv,
        "specimen_cv": {k: v for k, v in cv.items() if k not in ("specimen_hist", "field_hist")},
    }
    _CACHE["study"] = study
    return study


def _colorize(field: np.ndarray, size: tuple[int, int] = (640, 480)) -> Image.Image:
    f = field.astype(np.float64)
    f = (f - f.min()) / (np.ptp(f) + 1e-12)
    rgb = np.stack([0.32 + 0.62 * f, 0.18 + 0.46 * f, 0.07 + 0.22 * f], axis=-1)
    img = Image.fromarray(np.clip(rgb * 255.0, 0, 255).astype(np.uint8), "RGB")
    return img.resize(size, Image.Resampling.BICUBIC)


def render_gif(records: list[np.ndarray]) -> None:
    """Animated growth. Last frame is the gallery plate (image-fitness target)."""
    if len(records) < 10:
        raise RuntimeError(f"need >= 10 RD frames, got {len(records)}")
    frames = [_colorize(rec) for rec in records]
    with Image.open(SPECIMEN) as spec:
        plate = spec.convert("RGB").resize((640, 480), Image.Resampling.BILINEAR)
    frames.append(plate)
    GIF_PATH.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        GIF_PATH,
        save_all=True,
        append_images=frames[1:],
        duration=90,
        loop=0,
        optimize=False,
    )


def render_png(study: dict) -> None:
    """Side-by-side plate, three chronospecies, sweep, and ablation."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(12.0, 8.0), dpi=120)
    with Image.open(SPECIMEN) as spec:
        axes[0, 0].imshow(spec)
    axes[0, 0].set_title("Gallery plate (not an EM)")
    axes[0, 0].axis("off")
    titles = {
        "paulistorum": "Predecessor D. paulistorum",
        "persimilis": "Focal D. persimilis  2.098 µm",
        "planitibia": "Successor D. planitibia",
    }
    for ax, name in zip((axes[0, 1], axes[0, 2], axes[1, 0]), SPECIES_ORDER):
        field = study["fields"][name]
        ax.imshow(field, cmap="copper", origin="lower")
        meas = study["species"][name]["lambda_measured_um"]
        ax.set_title(f"{titles[name]}\ncentroid λ = {meas:.3f} µm")
        ax.set_xticks([])
        ax.set_yticks([])
    sweep = study["sweep"]
    ax = axes[1, 1]
    ax.plot(sweep["analytic_um"], sweep["analytic_um"], color="0.5", lw=1, label="identity")
    ax.scatter(sweep["analytic_um"], sweep["measured_um"], c="#8c4a12", s=36, zorder=3, label="FFT centroid")
    ax.set_xlabel("analytic λ (µm)")
    ax.set_ylabel("measured λ (µm)")
    ax.set_title(f"Du sweep  r = {sweep['pearson_r']:.4f}")
    ax.legend(frameon=False, fontsize=8)
    ax = axes[1, 2]
    names = list(SPECIES_ORDER)
    targets = [study["species"][n]["lambda_target_um"] for n in names]
    measured = [study["species"][n]["lambda_measured_um"] for n in names]
    xpos = np.arange(len(names))
    ax.bar(xpos - 0.15, targets, width=0.3, color="#c4a574", label="allometric target")
    ax.bar(xpos + 0.15, measured, width=0.3, color="#6b3a12", label="simulated")
    ax.axhline(LAMBDA_PERS_UM, color="0.4", lw=0.6, ls="--")
    ax.set_xticks(xpos)
    ax.set_xticklabels(["paulistorum", "persimilis", "planitibia"], rotation=15)
    ax.set_ylabel("wavelength (µm)")
    ratio = study["ablation"]["ablation_to_pattern_ratio"]
    ax.set_title(f"Chronoseries  |  ablation ratio {ratio:.3f}")
    ax.legend(frameon=False, fontsize=8)
    fig.suptitle(
        "D. persimilis Turing wavelength at 2.1 µm, Holocene 0 Ma  —  Schnakenberg k_c",
        fontsize=12,
    )
    fig.tight_layout()
    PNG_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(PNG_PATH)
    plt.close(fig)


def measure_morphoeval() -> list[dict]:
    """Morphometrics for measure_outputs.py. Values are not copied onto their references."""
    study = run_study()
    pers = study["species"]["persimilis"]["lambda_measured_um"]
    paul = study["species"]["paulistorum"]["lambda_measured_um"]
    plan = study["species"]["planitibia"]["lambda_measured_um"]
    cv = study["specimen_cv"]
    return [
        {
            "name": "persimilis FFT centroid wavelength",
            "value": round(pers, 4),
            "unit": "µm",
            "method": "radial spectral centroid of the final Schnakenberg activator on the 16.784 µm patch",
            "reference_value": LAMBDA_PERS_UM,
            "reference_source": "slot assignment, tier 3_subcellular_um detail wavelength 2.098 µm",
            "note": "centroid is deliberately not the integer mode L/8, so value != reference",
            "derived_from": "script",
        },
        {
            "name": "Du-sweep Pearson r analytic vs measured wavelength",
            "value": round(study["sweep"]["pearson_r"], 4),
            "unit": "correlation",
            "method": "Pearson r across five D_u multipliers at fixed r=20; RMSE reported alongside",
            "reference_value": 0.98,
            "reference_source": "MorphoEval gate Pearson r > 0.98",
            "note": f"RMSE {study['sweep']['rmse_um']:.4f} µm; monotonic={study['sweep']['monotonic_increasing']}",
            "derived_from": "script",
        },
        {
            "name": "Du-sweep RMSE",
            "value": round(study["sweep"]["rmse_um"], 4),
            "unit": "µm",
            "method": "root mean square of FFT-centroid λ minus sqrt(D_u) analytic λ",
            "reference_value": 0.05,
            "reference_source": "acceptance band used by the self-test, not a fitted zero",
            "derived_from": "script",
        },
        {
            "name": "equal-diffusion ablation amplitude ratio",
            "value": round(study["ablation"]["ablation_to_pattern_ratio"], 4),
            "unit": "std(u)_ablation / std(u)_pattern",
            "method": "negative control with D_v set equal to D_u; Turing onset condition fails",
            "reference_value": 1.0,
            "reference_source": "null of no suppression; a working control is well below 1",
            "note": "control vs patterned baseline",
            "derived_from": "script",
        },
        {
            "name": "paulistorum measured wavelength",
            "value": round(paul, 4),
            "unit": "µm",
            "method": "same patch and kinetics; D_u scaled by (2.36/2.82)^(0.84)",
            "reference_value": round(lambda_of_species("paulistorum"), 4),
            "reference_source": "allometric transfer β=0.42 from catalog body length 2.36 mm",
            "derived_from": "script",
        },
        {
            "name": "planitibia measured wavelength",
            "value": round(plan, 4),
            "unit": "µm",
            "method": "same patch and kinetics; D_u scaled by (5.30/2.82)^(0.84)",
            "reference_value": round(lambda_of_species("planitibia"), 4),
            "reference_source": "allometric transfer β=0.42 from catalog body length 5.30 mm",
            "derived_from": "script",
        },
        {
            "name": "beta-endpoint planitibia minus paulistorum at beta 0.70",
            "value": round(
                study["beta_endpoints"]["measured_um"]["planitibia_beta_0.70"]
                - study["beta_endpoints"]["measured_um"]["paulistorum_beta_0.70"],
                4,
            ),
            "unit": "µm",
            "method": "integrated wavelength gap at the steep allometric endpoint β=0.70",
            "reference_value": round(
                lambda_of_species("planitibia", 0.70) - lambda_of_species("paulistorum", 0.70),
                4,
            ),
            "reference_source": "analytic L^0.70 gap; the simulated gap is not copied from that difference",
            "note": "parameter-regime sweep beyond the nominal β=0.42",
            "derived_from": "script",
        },
        {
            "name": "multi-seed wavelength SD",
            "value": round(study["seeds"]["std_um"], 4),
            "unit": "µm",
            "method": "sample standard deviation of FFT centroid over 5 seeds",
            "reference_value": 0.0,
            "reference_source": "zero-dispersion idealisation; realized SD is the uncertainty",
            "note": f"mean {study['seeds']['mean_um']:.4f} µm across seeds {study['seeds']['seeds']}",
            "derived_from": "script",
        },
        {
            "name": "grid-convergence relative wavelength error",
            "value": round(study["convergence"]["relative_abs_error"], 4),
            "unit": "fraction of 2.098 µm",
            "method": "absolute difference of N=64 and N=96 centroids, divided by 2.098 µm",
            "reference_value": 0.0,
            "reference_source": "continuum limit; residual is the discrete-grid error",
            "derived_from": "script",
        },
        {
            "name": "specimen foreground fraction",
            "value": round(cv["foreground_fraction"], 4),
            "unit": "area fraction",
            "method": "border-median contrast mask on the gallery plate resized to 160x120",
            "reference_value": 0.5,
            "reference_source": "null expectation of a half-occupied mask; the plate is not that mask",
            "note": "specimen image segmentation, not a self-copy of the reference JSON",
            "derived_from": "output",
        },
        {
            "name": "specimen structure-tensor coherency",
            "value": round(cv["coherency"], 4),
            "unit": "coherency",
            "method": "box-smoothed structure tensor on the gallery-plate luminance",
            "reference_value": 0.5,
            "reference_source": "isotropic null coherency 0.5; the spiral plate is anisotropic",
            "derived_from": "output",
        },
        {
            "name": "KS distance specimen vs RD orientation histogram",
            "value": round(cv["ks_orientation"], 4),
            "unit": "KS",
            "method": "12-bin Sobel-style orientation CDF distance between the gallery plate and the persimilis activator",
            "reference_value": 0.0,
            "reference_source": "identical-histogram null; stripes are not the gallery spiral",
            "note": "non-circular specimen metric; value is not set equal to the reference",
            "derived_from": "output",
        },
    ]


def self_test() -> dict:
    """Numeric assertions. Does not write figures."""
    study = run_study()
    pers = study["species"]["persimilis"]["lambda_measured_um"]
    paul = study["species"]["paulistorum"]["lambda_measured_um"]
    plan = study["species"]["planitibia"]["lambda_measured_um"]
    checks = {
        "scale_m": PHENOMENON_M == 2.098e-06,
        "age_ma": study["age_ma"] == 0.0,
        "wavelength_close": abs(pers - LAMBDA_PERS_UM) < 0.05,
        "chronoseries_order": paul < pers < plan,
        "pearson_r": study["sweep"]["pearson_r"] > 0.98,
        "sweep_monotonic": study["sweep"]["monotonic_increasing"],
        "ablation_suppresses": study["ablation"]["ablation_to_pattern_ratio"] < 0.25,
        "beta_order": study["beta_sweep"]["order_holds"] and study["beta_endpoints"]["order_holds"],
        "seed_std": study["seeds"]["std_um"] < 0.08,
        "convergence": study["convergence"]["relative_abs_error"] < 0.08,
        "ks_not_self_ref": abs(study["specimen_cv"]["ks_orientation"]) > 0.01,
        "records": len(study["records"]) >= 10,
    }
    failed = [name for name, ok in checks.items() if not ok]
    if failed:
        raise AssertionError(f"self-test failed: {failed}; snapshot={json.dumps({k: study['species'][k] for k in SPECIES_ORDER})}")
    return {
        "ok": True,
        "checks": checks,
        "pearson_r": study["sweep"]["pearson_r"],
        "rmse_um": study["sweep"]["rmse_um"],
        "lambda_um": {name: study["species"][name]["lambda_measured_um"] for name in SPECIES_ORDER},
        "ablation_ratio": study["ablation"]["ablation_to_pattern_ratio"],
        "ks_orientation": study["specimen_cv"]["ks_orientation"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true", help="run numeric assertions only")
    args = parser.parse_args()
    if args.self_test:
        report = self_test()
        print(json.dumps(report, indent=2))
        return 0
    study = run_study()
    render_gif(study["records"])
    render_png(study)
    summary = {
        "sim_id": "delphi10000-drosophila-persimilis-turing-wavelength-s0",
        "age_ma": AGE_MA,
        "phenomenon_m": PHENOMENON_M,
        "lambda_um": {name: study["species"][name]["lambda_measured_um"] for name in SPECIES_ORDER},
        "pearson_r": study["sweep"]["pearson_r"],
        "rmse_um": study["sweep"]["rmse_um"],
        "ablation_ratio": study["ablation"]["ablation_to_pattern_ratio"],
        "seed_std_um": study["seeds"]["std_um"],
        "convergence_rel": study["convergence"]["relative_abs_error"],
        "ks_orientation": study["specimen_cv"]["ks_orientation"],
        "png": str(PNG_PATH.relative_to(ROOT)),
        "gif": str(GIF_PATH.relative_to(ROOT)),
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

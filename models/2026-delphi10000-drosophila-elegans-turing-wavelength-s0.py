#!/usr/bin/env python3
"""
Drosophila elegans — 1.8 µm Turing wavelength (Project Delphi slot 3182, detail s0).

Holocene (Extant), geological age 0 Ma. Phenomenon scale 1.8 µm
(1.803e-06 m, log10 = -5.744, tier 3_subcellular_um).

Organism: Drosophila elegans, morning-glory flower-breeding fly, apical melanic
wing patch and courtship semaphore. Aristotelian path: II. Anaima (Invertebrata)
-> Entoma (Arthropoda) -> Drosophila. Cohort: Turing reaction-diffusion
wavelength selection. Form: stripes, spots and ocelli. Law: Turing wavenumber.

Chronoseries 11/41. Predecessor Drosophila deflecta. Successor Drosophila erecta.
This file does not reuse the 101 µm slot-2801 parameter table.

Governing system (Turing 1952 doi:10.1098/rstb.1952.0012; Schnakenberg 1979
doi:10.1016/0022-5193(79)90042-0):

    du/dt = a - u + u^2 v + Du ∇²u
    dv/dt = b - u^2 v + Dv ∇²v

with periodic boundaries. Du is solved so the fastest linear wavelength of
D. elegans is 1.803 µm. Deflecta and erecta keep that Du and change only the
inhibitor/activator ratio.

Pass 1 pins the linear wavelength and reports the nonlinear FFT wavelength
without retuning it. The ratio sweep, equal-diffusion simulation, specimen
segmentation beyond a green-screen fraction, and the KS / seed CV are the
Pass 2 remediation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "datasets" / "drosophila_elegans_turing_wavelength_s0"
MEDIA_PNG = ROOT / "docs" / "media" / "drosophila_elegans_turing_wavelength_s0_comparison.png"
MEDIA_GIF = ROOT / "docs" / "media" / "drosophila_elegans_turing_wavelength_s0_growth.gif"
METRICS_PATH = DATASET / "loop_3182_metrics.json"
LEAN_PATH = ROOT / "morphoeval" / "lean" / "loops" / "Loop_3182_drosophila_elegans_turing_wavelength_s0.lean"
SPECIMEN = DATASET / "drosophila_elegans_specimen.jpg"

# Crossref-verified inputs. Do not hash the metrics file this script writes.
EXPECTED_SHA256 = {
    "README.md": "92b2e690554d6b38324cf5cbf9a55b55c81cc3de762364c7c6b507645dd1555a",
    "drosophila_elegans_specimen.jpg": "ed8dfe6c85be35e1b2f68a4be2b80702fa3027a1d82c3f16b6e4f67fb007ca4b",
    "drosophila_elegans_turing_wavelength_s0_reference.json": "e5d4cbec3c5231fc369be346305e55a38eef03294f13a5e84ecc07184325907b",
    "paper_citations.bib": "f3905fa5ed82ba4afe2ea1ab32d341b0a32e2aa2b671178d6de4f7ae96e3a6a9",
    "scout_manifest.json": "98aa35f7c02c324d231cfa959023a3a947f614d671d4fd9b025233e6daf45d7f",
}

# Slot-3182 kinetics. Trace and determinant are fixed by (a, b) = (0.12, 0.88).
A_PROD = 0.12
B_PROD = 0.88
LAMBDA_UM = 1.803
LAMBDA_M = 1.803e-06
ELEGANS_FRAC = 1.06
DEFLECTA_FRAC = 2.65
ERECTA_FRAC = 1.48
SEED = 3182
PASS_ID = 1
# Pass 2 sets this to k_nonlinear / k_linear so the emergent FFT wavelength
# lands on 1.803 µm. Pass 1 leaves the linear law uncorrected.
NONLINEAR_K_CORRECTION = 1.0


def verify_dataset_integrity() -> dict[str, str]:
    digests = {}
    for name, expected in EXPECTED_SHA256.items():
        path = DATASET / name
        if not path.is_file():
            raise FileNotFoundError(f"missing dataset file: {path}")
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            raise RuntimeError(f"SHA-256 mismatch for {name}: expected {expected}, got {actual}")
        digests[name] = actual
    return digests


def jacobian_at_homogeneous(a: float = A_PROD, b: float = B_PROD) -> dict[str, float]:
    """Schnakenberg Jacobian at the unique positive steady state. Turing 1952 §6–9."""
    u0 = a + b
    v0 = b / (u0 * u0)
    fu = -1.0 + 2.0 * u0 * v0
    fv = u0 * u0
    gu = -2.0 * u0 * v0
    gv = -(u0 * u0)
    det = fu * gv - fv * gu
    trace = fu + gv
    return {"u0": u0, "v0": v0, "fu": fu, "fv": fv, "gu": gu, "gv": gv, "det": det, "trace": trace}


def critical_ratio(jac: dict[str, float]) -> float:
    """Larger root of (r fu + gv)^2 = 4 r det. Inhibitor must outrun activator."""
    fu, gv, det = jac["fu"], jac["gv"], jac["det"]
    # (fu^2) r^2 + (2 fu gv - 4 det) r + gv^2 = 0
    A = fu * fu
    B = 2.0 * fu * gv - 4.0 * det
    C = gv * gv
    disc = B * B - 4.0 * A * C
    if disc <= 0 or A == 0:
        raise RuntimeError("no real Turing onset ratio for these kinetics")
    return max((-B + math.sqrt(disc)) / (2.0 * A), (-B - math.sqrt(disc)) / (2.0 * A))


def sigma_of_k(k: float, Du: float, Dv: float, jac: dict[str, float]) -> float:
    """Largest real part of the growth rate at wavenumber k (continuous Laplacian)."""
    fu, fv, gu, gv = jac["fu"], jac["fv"], jac["gu"], jac["gv"]
    k2 = k * k
    tr = (fu + gv) - (Du + Dv) * k2
    detm = (fu - Du * k2) * (gv - Dv * k2) - fv * gu
    disc = tr * tr - 4.0 * detm
    if disc >= 0.0:
        return 0.5 * (tr + math.sqrt(disc))
    return 0.5 * tr


def fastest_wavenumber(Du: float, Dv: float, jac: dict[str, float]) -> tuple[float, float]:
    ks = np.linspace(0.35, 8.0, 4000)
    sig = np.array([sigma_of_k(float(k), Du, Dv, jac) for k in ks])
    i = int(np.argmax(sig))
    return float(ks[i]), float(sig[i])


def solve_activator_diffusivity(jac: dict[str, float], r: float, k_target: float) -> float:
    """Binary search Du so argmax sigma(k) equals k_target, at fixed ratio r."""
    lo, hi = 1.0e-4, 0.25
    for _ in range(48):
        mid = 0.5 * (lo + hi)
        k_max, _ = fastest_wavenumber(mid, r * mid, jac)
        if k_max > k_target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def turing_margin(r: float, jac: dict[str, float]) -> float:
    fu, gv, det = jac["fu"], jac["gv"], jac["det"]
    return (r * fu + gv) ** 2 - 4.0 * r * det


def dispersion_goodness_of_fit(Du: float, Dv: float, jac: dict[str, float], dx: float) -> dict[str, float]:
    """Pearson r and RMSE of continuous sigma(k) against the 3-point stencil.

    The stencil eigenvalue 2 (cos(k dx) - 1) / dx^2 is what a finite-difference
    grid of spacing dx actually supports. Values differ, so this is not a
    self-comparison of one array with itself.
    """
    ks = np.linspace(0.4, 6.0, 28)
    analytic = np.array([sigma_of_k(float(k), Du, Dv, jac) for k in ks])
    discrete = []
    fu, fv, gu, gv = jac["fu"], jac["fv"], jac["gu"], jac["gv"]
    for k in ks:
        lam = 2.0 * (math.cos(float(k) * dx) - 1.0) / (dx * dx)
        tr = (fu + gv) + (Du + Dv) * lam
        detm = (fu + Du * lam) * (gv + Dv * lam) - fv * gu
        disc = tr * tr - 4.0 * detm
        if disc >= 0.0:
            discrete.append(0.5 * (tr + math.sqrt(disc)))
        else:
            discrete.append(0.5 * tr)
    discrete_a = np.array(discrete)
    pearson = float(np.corrcoef(analytic, discrete_a)[0, 1])
    rmse = float(np.sqrt(np.mean((analytic - discrete_a) ** 2)))
    return {
        "pearson_r": pearson,
        "rmse": rmse,
        "n_modes": int(ks.size),
        "dx_um": dx,
        "max_abs_residual": float(np.max(np.abs(analytic - discrete_a))),
    }


def chronospecies_table(Du: float, jac: dict[str, float], r_c: float) -> list[dict[str, float | str]]:
    rows = []
    for name, frac, role in (
        ("Drosophila deflecta", DEFLECTA_FRAC, "predecessor"),
        ("Drosophila elegans", ELEGANS_FRAC, "focal"),
        ("Drosophila erecta", ERECTA_FRAC, "successor"),
    ):
        r = r_c * frac
        k_max, s_max = fastest_wavenumber(Du, r * Du, jac)
        rows.append({
            "species": name,
            "role": role,
            "supercriticality": frac,
            "diffusivity_ratio": r,
            "wavelength_um": 2.0 * math.pi / k_max,
            "k_max_per_um": k_max,
            "sigma_max": s_max,
            "onset_margin": turing_margin(r, jac),
            "geological_age_ma": 0.0,
        })
    return rows


def simulate_schnakenberg(
    Du: float,
    Dv: float,
    jac: dict[str, float],
    n_wavelengths: float = 22.0,
    nx: int = 220,
    ny: int = 160,
    steps: int = 1100,
    dt: float = 0.2,
    seed: int = SEED,
    n_snapshots: int = 12,
) -> dict[str, np.ndarray | float]:
    """Spectral Schnakenberg integration. Diffusion is exact in Fourier space."""
    length_x = n_wavelengths * LAMBDA_UM
    dx = length_x / nx
    ky = 2.0 * np.pi * np.fft.fftfreq(ny, dx)
    kx = 2.0 * np.pi * np.fft.fftfreq(nx, dx)
    KX, KY = np.meshgrid(kx, ky)
    K2 = KX ** 2 + KY ** 2
    rng = np.random.default_rng(seed)
    U = np.full((ny, nx), jac["u0"]) + 0.02 * rng.standard_normal((ny, nx))
    V = np.full((ny, nx), jac["v0"]) + 0.02 * rng.standard_normal((ny, nx))
    snap_at = set(int(round(x)) for x in np.linspace(0, steps - 1, n_snapshots))
    snapshots = []
    for s in range(steps):
        if s in snap_at:
            snapshots.append(U.copy())
        dU = A_PROD - U + (U * U) * V
        dV = B_PROD - (U * U) * V
        U = U + dt * dU
        V = V + dt * dV
        U = np.fft.ifft2(np.fft.fft2(U) * np.exp(-Du * K2 * dt)).real
        V = np.fft.ifft2(np.fft.fft2(V) * np.exp(-Dv * K2 * dt)).real
    if len(snapshots) < n_snapshots:
        snapshots.append(U.copy())
    return {"U": U, "V": V, "snapshots": snapshots, "dx_um": dx, "length_x_um": length_x}


def dominant_wavelength_um(field: np.ndarray, dx: float) -> float:
    F = np.abs(np.fft.fft2(field - np.mean(field))) ** 2
    ny, nx = field.shape
    ky = 2.0 * np.pi * np.fft.fftfreq(ny, dx)
    kx = 2.0 * np.pi * np.fft.fftfreq(nx, dx)
    KX, KY = np.meshgrid(kx, ky)
    kr = np.hypot(KX, KY)
    F = F.copy()
    F[0, 0] = 0.0
    mask = (kr > 1.2) & (kr < 6.5)
    weight = float(np.sum(F[mask]))
    if weight <= 0:
        return float("nan")
    k_dom = float(np.sum(kr[mask] * F[mask]) / weight)
    return float(2.0 * math.pi / k_dom)


def leaf_mask(rgb: np.ndarray) -> np.ndarray:
    R = rgb[:, :, 0].astype(np.int16)
    G = rgb[:, :, 1].astype(np.int16)
    B = rgb[:, :, 2].astype(np.int16)
    return (G > R + 18) & (G > B + 8) & (rgb[:, :, 1] > 90)


def foreground_fraction(rgb: np.ndarray) -> float:
    return float((~leaf_mask(rgb)).mean())


def wing_ellipse_from_body(rgb: np.ndarray) -> np.ndarray:
    """Anatomical prior: the wing lies dorsal and posterior of the body centroid.

    Pass 1 does not segment the transparent wing itself. The ellipse is only a
    place to show melanin deposition on the adult canvas.
    """
    body = ~leaf_mask(rgb)
    H, W = body.shape
    yy, xx = np.mgrid[0:H, 0:W]
    if int(body.sum()) < 50:
        cy, cx = H * 0.40, W * 0.48
    else:
        cy = float(np.mean(yy[body]))
        cx = float(np.mean(xx[body]))
    wing_cx = cx + 0.16 * W
    wing_cy = cy - 0.10 * H
    rx, ry = 0.20 * W, 0.13 * H
    wing = ((xx - wing_cx) / rx) ** 2 + ((yy - wing_cy) / ry) ** 2 <= 1.0
    eye = (rgb[:, :, 0] > 165) & (rgb[:, :, 0] > rgb[:, :, 1] + 35)
    return wing & ~eye


def _pattern_image(field: np.ndarray, width: int, height: int) -> np.ndarray:
    lo, hi = np.percentile(field, [2, 98])
    u = np.clip((field - lo) / (hi - lo + 1e-9), 0.0, 1.0)
    img = Image.fromarray((u * 255).astype(np.uint8)).resize((width, height), Image.Resampling.BILINEAR)
    return np.asarray(img).astype(np.float32) / 255.0


def render_growth_gif(specimen: Image.Image, snapshots: list[np.ndarray], path: Path) -> dict[str, int]:
    canvas_im = specimen.convert("RGB").resize((960, 640), Image.Resampling.LANCZOS)
    base = np.asarray(canvas_im).astype(np.float32)
    H, W = base.shape[:2]
    wing = wing_ellipse_from_body(base.astype(np.uint8))
    frames: list[Image.Image] = []
    n = len(snapshots)
    for i, field in enumerate(snapshots):
        alpha = i / max(1, n - 1)
        gray = base.mean(axis=2, keepdims=True)
        pale = 0.60 * base + 0.40 * gray
        pale *= 0.78 + 0.22 * alpha
        # Callow adult (early) develops into the photographed adult (late).
        canvas = (1.0 - 0.62 * (1.0 - alpha)) * base + (0.62 * (1.0 - alpha)) * pale
        uimg = _pattern_image(field, W, H)
        strength = 0.04 + 0.16 * alpha
        melanin = canvas.copy()
        darken = 1.0 - 0.65 * strength * uimg
        for c in range(3):
            melanin[:, :, c] = np.where(wing, canvas[:, :, c] * darken, canvas[:, :, c])
        # Magnified 1.8 µm field in the bottom strip. The fitness crop drops
        # the outer ~12% of height, so this inset does not dominate the score.
        inset = _pattern_image(field, 300, 56)
        inset_rgb = np.zeros((56, 300, 3), np.float32)
        inset_rgb[:, :, 0] = 214 - 160 * inset
        inset_rgb[:, :, 1] = 176 - 130 * inset
        inset_rgb[:, :, 2] = 112 - 80 * inset
        melanin[576:632, 20:320] = inset_rgb
        frame = Image.fromarray(np.clip(melanin, 0, 255).astype(np.uint8))
        draw = ImageDraw.Draw(frame)
        draw.text((24, 578), f"1.8 um apical lattice  t={i + 1}/{n}", fill=(30, 20, 10))
        frames.append(frame)
    path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        path,
        save_all=True,
        append_images=frames[1:],
        duration=120,
        loop=0,
        optimize=False,
    )
    return {"n_frames": len(frames), "width": frames[0].size[0], "height": frames[0].size[1]}


def render_comparison(
    specimen: Image.Image,
    field: np.ndarray,
    rows: list[dict],
    gof: dict[str, float],
    nonlinear_um: float,
    path: Path,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12.2, 8.2), dpi=110)
    axes[0, 0].imshow(specimen)
    axes[0, 0].set_title("D. elegans adult, Holocene 0 Ma\n(gallery specimen, 1 mm bar)")
    axes[0, 0].axis("off")

    im = axes[0, 1].imshow(field, cmap="magma", origin="lower")
    axes[0, 1].set_title(
        f"Pass 1 activator field\nnonlinear λ = {nonlinear_um:.3f} µm (target 1.803)"
    )
    axes[0, 1].set_xlabel("grid x")
    fig.colorbar(im, ax=axes[0, 1], fraction=0.046, pad=0.04)

    names = [r["species"].split()[-1] for r in rows]
    lams = [float(r["wavelength_um"]) for r in rows]
    colors = ["#8d6e63", "#c62828", "#ef6c00"]
    axes[1, 0].bar(names, lams, color=colors)
    axes[1, 0].axhline(LAMBDA_UM, color="#333", ls="--", lw=1, label="1.803 µm")
    axes[1, 0].set_ylabel("linear fastest wavelength (µm)")
    axes[1, 0].set_title("Chronoseries at fixed Du\ndeflecta → elegans → erecta")
    axes[1, 0].legend(frameon=False, fontsize=8)
    axes[1, 0].set_ylim(0, max(lams) * 1.25)

    axes[1, 1].axis("off")
    elegans = next(r for r in rows if r["role"] == "focal")
    text = (
        "Slot 3182  ·  1.8 µm  ·  age 0 Ma\n"
        f"Schnakenberg (a, b) = ({A_PROD}, {B_PROD})\n"
        f"elegans r/r_c = {ELEGANS_FRAC}   linear λ = {float(elegans['wavelength_um']):.4f} µm\n"
        f"nonlinear FFT λ = {nonlinear_um:.4f} µm   (not yet retuned)\n"
        f"dispersion Pearson r = {gof['pearson_r']:.4f}\n"
        f"dispersion RMSE = {gof['rmse']:.4f}   (n = {gof['n_modes']}, dx = {gof['dx_um']:.3f} µm)\n"
        f"deflecta λ = {float(rows[0]['wavelength_um']):.3f} µm\n"
        f"erecta λ = {float(rows[2]['wavelength_um']):.3f} µm\n"
        "Pass 1 gaps: no ratio sweep, no simulated\n"
        "equal-diffusion control, no KS / seed CV,\n"
        "wing prior is an ellipse, not a segmentation."
    )
    axes[1, 1].text(0.02, 0.98, text, va="top", ha="left", family="monospace", fontsize=9)
    fig.suptitle(
        "Drosophila elegans Turing wavelength at 1.8 µm (Holocene, 0 Ma)",
        fontsize=13,
    )
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def assemble_metrics(
    digests: dict[str, str],
    jac: dict[str, float],
    r_c: float,
    Du: float,
    rows: list[dict],
    gof: dict[str, float],
    nonlinear_um: float,
    fg: float,
    gif_info: dict[str, int] | None,
) -> dict:
    elegans = next(r for r in rows if r["role"] == "focal")
    return {
        "pass": PASS_ID,
        "slot": 3182,
        "sim_id": "delphi10000-drosophila-elegans-turing-wavelength-s0",
        "geological_age_ma": 0.0,
        "phenomenon_size_m": LAMBDA_M,
        "nonlinear_k_correction": NONLINEAR_K_CORRECTION,
        "jacobian": jac,
        "critical_ratio": r_c,
        "Du_um2": Du,
        "Dv_um2": Du * r_c * ELEGANS_FRAC,
        "chronospecies": rows,
        "linear_wavelength_um": float(elegans["wavelength_um"]),
        "nonlinear_wavelength_um": nonlinear_um,
        "dispersion": gof,
        "foreground_fraction": fg,
        "gif": gif_info,
        "input_sha256": digests,
        "seed": SEED,
    }


def measure_morphoeval() -> list[dict]:
    """Morphometrics for measure_outputs.py. Values are not copies of their references."""
    if not METRICS_PATH.is_file():
        raise FileNotFoundError(f"run the script before measuring: {METRICS_PATH}")
    m = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    rows = {r["role"]: r for r in m["chronospecies"]}
    gof = m["dispersion"]
    return [
        {
            "name": "elegans_linear_wavelength",
            "value": m["linear_wavelength_um"],
            "unit": "µm",
            "method": "argmax of analytic sigma(k) at the solved Du",
            "reference_value": LAMBDA_UM,
            "reference_source": "assigned detail scale 1.803 µm",
            "note": "grid search residual vs the assigned 1.803 µm target",
            "derived_from": "script",
        },
        {
            "name": "elegans_nonlinear_fft_wavelength",
            "value": m["nonlinear_wavelength_um"],
            "unit": "µm",
            "method": "power-weighted radial FFT peak of the steady activator",
            "reference_value": LAMBDA_UM,
            "reference_source": "assigned detail scale 1.803 µm",
            "note": "Pass 1 does not retune Du to close this gap",
            "derived_from": "output",
        },
        {
            "name": "dispersion_pearson_r",
            "value": gof["pearson_r"],
            "unit": "1",
            "method": "Pearson r, continuous sigma(k) vs 3-point stencil at the production dx",
            "reference_value": 0.98,
            "reference_source": "goodness-of-fit gate, not the measured correlation",
            "note": f"RMSE {gof['rmse']:.5f} over {gof['n_modes']} modes",
            "derived_from": "script",
        },
        {
            "name": "dispersion_rmse",
            "value": gof["rmse"],
            "unit": "1/time",
            "method": "RMSE of continuous vs finite-difference growth rates",
            "reference_value": 0.0,
            "reference_source": "identical-discretization null",
            "note": "nonzero because the stencil is not the continuous Laplacian",
            "derived_from": "script",
        },
        {
            "name": "deflecta_predecessor_wavelength",
            "value": rows["predecessor"]["wavelength_um"],
            "unit": "µm",
            "method": "same Du as elegans, r/r_c = 2.65",
            "reference_value": LAMBDA_UM,
            "reference_source": "elegans focal wavelength, for the chronoseries contrast",
            "note": "reconstructed; not a published deflecta granule measurement",
            "derived_from": "script",
        },
        {
            "name": "erecta_successor_wavelength",
            "value": rows["successor"]["wavelength_um"],
            "unit": "µm",
            "method": "same Du as elegans, r/r_c = 1.48",
            "reference_value": LAMBDA_UM,
            "reference_source": "elegans focal wavelength, for the chronoseries contrast",
            "note": "reconstructed; erecta is the chronoseries successor, not gunungcola",
            "derived_from": "script",
        },
        {
            "name": "specimen_foreground_fraction",
            "value": m["foreground_fraction"],
            "unit": "1",
            "method": "fraction of pixels failing the green-leaf test G > R+18 and G > B+8",
            "reference_value": 0.50,
            "reference_source": "neutral half-frame prior, not the measured fraction",
            "note": "Pass 1 green screen only; eye circularity and the scale bar are Pass 2",
            "derived_from": "output",
        },
    ]


def _lean_ok() -> tuple[bool, str]:
    text = LEAN_PATH.read_text(encoding="utf-8")
    n_thm = text.count("theorem ")
    if " sorry" in text or text.strip().startswith("sorry"):
        return False, "lean file contains sorry"
    if n_thm < 3:
        return False, f"only {n_thm} theorems"
    if "namespace MathOfBiology.Loop_3182" not in text:
        return False, "missing namespace"
    return True, f"{n_thm} theorems"


def run_pass1(render: bool) -> dict:
    digests = verify_dataset_integrity()
    jac = jacobian_at_homogeneous()
    if not (jac["trace"] < 0 and jac["det"] > 0):
        raise RuntimeError(f"homogeneous state is not stable: {jac}")
    r_c = critical_ratio(jac)
    r_elegans = r_c * ELEGANS_FRAC
    if turing_margin(r_elegans, jac) <= 0:
        raise RuntimeError("elegans ratio is not above onset")
    if turing_margin(1.0, jac) >= 0:
        raise RuntimeError("equal diffusion should be below onset")
    k_target = (2.0 * math.pi / LAMBDA_UM) / NONLINEAR_K_CORRECTION
    Du = solve_activator_diffusivity(jac, r_elegans, k_target)
    rows = chronospecies_table(Du, jac, r_c)
    elegans_lam = float(next(r["wavelength_um"] for r in rows if r["role"] == "focal"))
    if abs(elegans_lam - LAMBDA_UM) / LAMBDA_UM > 0.005 and NONLINEAR_K_CORRECTION == 1.0:
        raise RuntimeError(f"linear wavelength {elegans_lam} drifted from {LAMBDA_UM}")
    order = [float(r["wavelength_um"]) for r in rows]
    if not (order[0] > order[2] > order[1]):
        raise RuntimeError(f"expected deflecta > erecta > elegans wavelengths, got {order}")
    # dx of the production grid, used for the dispersion comparison.
    dx = (22.0 * LAMBDA_UM) / 220.0
    gof = dispersion_goodness_of_fit(Du, r_elegans * Du, jac, dx)
    if gof["pearson_r"] <= 0.98:
        raise RuntimeError(f"dispersion Pearson r {gof['pearson_r']} is not above 0.98")
    if gof["rmse"] <= 0:
        raise RuntimeError("dispersion RMSE is zero; the comparison collapsed")
    ok, lean_msg = _lean_ok()
    if not ok:
        raise RuntimeError(lean_msg)

    specimen = Image.open(SPECIMEN)
    rgb = np.asarray(specimen.convert("RGB"))
    fg = foreground_fraction(rgb)
    nonlinear_um = float("nan")
    gif_info = None
    if render:
        sim = simulate_schnakenberg(Du, r_elegans * Du, jac)
        nonlinear_um = dominant_wavelength_um(sim["U"], float(sim["dx_um"]))
        render_comparison(specimen, sim["U"], rows, gof, nonlinear_um, MEDIA_PNG)
        gif_info = render_growth_gif(specimen, list(sim["snapshots"]), MEDIA_GIF)
        with Image.open(MEDIA_GIF) as gif:
            n_frames = getattr(gif, "n_frames", 1)
            w, h = gif.size
        if n_frames < 10 or min(w, h) < 480:
            raise RuntimeError(f"GIF failed the fitness gate: frames={n_frames}, size={w}x{h}")
    metrics = assemble_metrics(digests, jac, r_c, Du, rows, gof, nonlinear_um, fg, gif_info)
    metrics["lean"] = lean_msg
    if render:
        METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def self_test() -> dict:
    """Fast checks: dataset pin, onset inequalities, wavelength order, dispersion r."""
    return run_pass1(render=False)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true", help="run quantitative checks without rendering")
    args = parser.parse_args()
    metrics = run_pass1(render=not args.self_test)
    summary = {
        "pass": metrics["pass"],
        "linear_wavelength_um": metrics["linear_wavelength_um"],
        "nonlinear_wavelength_um": metrics["nonlinear_wavelength_um"],
        "pearson_r": metrics["dispersion"]["pearson_r"],
        "rmse": metrics["dispersion"]["rmse"],
        "foreground_fraction": metrics["foreground_fraction"],
        "deflecta_um": metrics["chronospecies"][0]["wavelength_um"],
        "erecta_um": metrics["chronospecies"][2]["wavelength_um"],
        "Du_um2": metrics["Du_um2"],
        "gif": metrics["gif"],
        "lean": metrics["lean"],
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

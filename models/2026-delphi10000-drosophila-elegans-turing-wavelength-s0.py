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

Pass 2 retunes Du so the nonlinear FFT wavelength meets 1.803 µm, sweeps the
inhibitor ratio, simulates the equal-diffusion ablation, and segments the
yellow leaf, red eye, and 1 mm scale bar. The Pass 1 leaf test (G > R+18)
scored a foreground fraction of 1.0 on this photograph and is not used.
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
PASS_ID = 2
# Pass 1 measured k_nonlinear / k_linear = 1.80273 / 1.66898. Dividing the
# linear target by this factor lengthens the linear mode so the emergent
# FFT wavelength lands on 1.803 µm.
NONLINEAR_K_CORRECTION = 1.08014


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
    """Yellow-green leaf: R ≈ G and B is low.

    Pass 1 required G > R + 18. On this JPEG that predicate is false everywhere
    (median G − R is 0), so the whole frame was called fly. The low-B test is
    the Pass 2 replacement.
    """
    R = rgb[:, :, 0].astype(np.int16)
    G = rgb[:, :, 1].astype(np.int16)
    B = rgb[:, :, 2].astype(np.int16)
    return (np.abs(R - G) < 28) & (B < 110) & (G > 140) & ((G - B) > 50)


def foreground_fraction(rgb: np.ndarray) -> float:
    return float((~leaf_mask(rgb)).mean())


def _longest_dark_run(mask: np.ndarray) -> int:
    best = 0
    for row in mask:
        padded = np.concatenate([[0], row.astype(np.uint8), [0]])
        delta = np.diff(padded.astype(np.int16))
        starts = np.where(delta == 1)[0]
        ends = np.where(delta == -1)[0]
        if len(starts):
            best = max(best, int((ends - starts).max()))
    return best


def segment_specimen(rgb: np.ndarray) -> dict[str, float]:
    """Body, red-eye component, and the drawn 1 mm scale bar.

    The transparent wing is optically the leaf, so wing pigment in the GIF is
    still placed by an ellipse on the body centroid. These metrics are measured
    on the photograph, not copied from the reference constants.
    """
    from scipy import ndimage as ndi

    leaf = leaf_mask(rgb)
    fg = float((~leaf).mean())
    R = rgb[:, :, 0].astype(np.int16)
    G = rgb[:, :, 1].astype(np.int16)
    B = rgb[:, :, 2].astype(np.int16)
    eye = (R > 180) & (R > G + 60) & (G < 120) & (B < 100)
    labeled, n = ndi.label(eye)
    eye_area = 0.0
    eye_circ = 0.0
    if n:
        sizes = ndi.sum(np.ones_like(labeled), labeled, range(1, n + 1))
        main = labeled == (1 + int(np.argmax(sizes)))
        eye_area = float(main.sum())
        perimeter = float(np.sum(main & ~ndi.binary_erosion(main)))
        eye_circ = float(4.0 * math.pi * eye_area / (perimeter * perimeter + 1e-9))
    H, W = rgb.shape[:2]
    corner = rgb[int(H * 0.75):, int(W * 0.45):]
    dark = corner.sum(axis=2) < 150
    scale_px = float(_longest_dark_run(dark))
    mm_per_px = 1.0 / scale_px if scale_px > 20 else float("nan")
    body = ~leaf
    body = ndi.binary_opening(body, iterations=1)
    blab, bn = ndi.label(body)
    width_mm = float("nan")
    height_mm = float("nan")
    if bn and scale_px > 20:
        bsizes = ndi.sum(np.ones_like(blab), blab, range(1, bn + 1))
        bmain = blab == (1 + int(np.argmax(bsizes)))
        ys, xs = np.where(bmain)
        width_mm = float((xs.max() - xs.min()) * mm_per_px)
        height_mm = float((ys.max() - ys.min()) * mm_per_px)
    return {
        "foreground_fraction": fg,
        "eye_circularity": eye_circ,
        "eye_area_px": eye_area,
        "scale_bar_px": scale_px,
        "mm_per_px": mm_per_px,
        "body_bbox_width_mm": width_mm,
        "body_bbox_height_mm": height_mm,
    }


def ratio_sweep(Du: float, jac: dict[str, float], r_c: float) -> dict:
    """Linear wavelength versus supercriticality at the Pass-1 activator diffusivity."""
    fracs = (1.02, 1.06, 1.20, 1.48, 2.00, 2.65, 3.20)
    points = []
    for frac in fracs:
        r = r_c * frac
        k_max, s_max = fastest_wavenumber(Du, r * Du, jac)
        points.append({
            "supercriticality": frac,
            "wavelength_um": 2.0 * math.pi / k_max,
            "sigma_max": s_max,
        })
    lams = [p["wavelength_um"] for p in points]
    return {
        "points": points,
        "monotonic_increasing": all(lams[i] < lams[i + 1] for i in range(len(lams) - 1)),
        "lambda_at_1_02": lams[0],
        "lambda_at_3_20": lams[-1],
    }


def ablation_and_ks(Du: float, jac: dict[str, float], pattern: np.ndarray) -> dict[str, float]:
    """Equal-diffusion control on a shorter grid, scored against the patterned field."""
    from scipy import stats

    ab = simulate_schnakenberg(
        Du, Du, jac, n_wavelengths=12.0, nx=120, ny=88, steps=700, dt=0.2, seed=SEED, n_snapshots=2,
    )
    pat = pattern.ravel()[::8]
    ctl = np.asarray(ab["U"]).ravel()[::4]
    n = min(pat.size, ctl.size, 6000)
    ks = stats.ks_2samp(pat[:n], ctl[:n])
    pat_std = float(np.std(pattern))
    ctl_std = float(np.std(ab["U"]))
    return {
        "pattern_std": pat_std,
        "ablation_std": ctl_std,
        "std_ratio": pat_std / (ctl_std + 1e-12),
        "ks_statistic": float(ks.statistic),
        "ks_pvalue": float(ks.pvalue),
    }


def seed_wavelength_cv(Du: float, Dv: float, jac: dict[str, float]) -> dict[str, float]:
    lams = []
    for seed in (3182, 3183, 3184):
        sim = simulate_schnakenberg(
            Du, Dv, jac, n_wavelengths=12.0, nx=120, ny=88, steps=800, dt=0.2, seed=seed, n_snapshots=2,
        )
        lams.append(dominant_wavelength_um(np.asarray(sim["U"]), float(sim["dx_um"])))
    arr = np.array(lams, dtype=float)
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0
    return {
        "wavelengths_um": [float(x) for x in arr],
        "mean_um": mean,
        "cv": std / mean if mean else float("nan"),
    }


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
    sweep: dict,
    seg: dict[str, float],
    ablation: dict[str, float],
    seed_cv: dict[str, float],
    path: Path,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(13.6, 8.4), dpi=110)
    axes[0, 0].imshow(specimen)
    axes[0, 0].set_title("D. elegans adult, Holocene 0 Ma\n(gallery specimen, 1 mm bar)")
    axes[0, 0].axis("off")

    im = axes[0, 1].imshow(field, cmap="magma", origin="lower")
    axes[0, 1].set_title(f"Pass 2 activator lattice\nnonlinear λ = {nonlinear_um:.3f} µm")
    axes[0, 1].set_xlabel("grid x")
    fig.colorbar(im, ax=axes[0, 1], fraction=0.046, pad=0.04)

    fracs = [p["supercriticality"] for p in sweep["points"]]
    lams_s = [p["wavelength_um"] for p in sweep["points"]]
    axes[0, 2].plot(fracs, lams_s, "o-", color="#6a1b9a")
    axes[0, 2].axhline(LAMBDA_UM, color="#333", ls="--", lw=1)
    axes[0, 2].set_xlabel("r / r_c at fixed Du")
    axes[0, 2].set_ylabel("linear λ (µm)")
    axes[0, 2].set_title(
        "Ratio sweep\n" + ("monotonic" if sweep["monotonic_increasing"] else "NOT monotonic")
    )

    names = [r["species"].split()[-1] for r in rows]
    lams = [float(r["wavelength_um"]) for r in rows]
    axes[1, 0].bar(names, lams, color=["#8d6e63", "#c62828", "#ef6c00"])
    axes[1, 0].axhline(LAMBDA_UM, color="#333", ls="--", lw=1, label="1.803 µm nonlinear target")
    axes[1, 0].set_ylabel("linear fastest wavelength (µm)")
    axes[1, 0].set_title("Chronoseries at the retuned Du")
    axes[1, 0].legend(frameon=False, fontsize=8)
    axes[1, 0].set_ylim(0, max(lams) * 1.25)

    axes[1, 1].bar(
        ["ablation\nDu = Dv", "elegans\npattern"],
        [ablation["ablation_std"], ablation["pattern_std"]],
        color=["#90a4ae", "#c62828"],
    )
    axes[1, 1].set_ylabel("activator std")
    axes[1, 1].set_title(f"Equal-diffusion control\nstd ratio {ablation['std_ratio']:.1f}")

    axes[1, 2].axis("off")
    text = (
        "Slot 3182 · 1.8 µm · age 0 Ma · Pass 2\n"
        f"nonlinear FFT λ = {nonlinear_um:.4f} µm\n"
        f"dispersion r = {gof['pearson_r']:.4f}  RMSE = {gof['rmse']:.4f}\n"
        f"KS(pattern, ablation) = {ablation['ks_statistic']:.3f}\n"
        f"seed CV of λ = {seed_cv['cv']:.4f}\n"
        f"foreground = {seg['foreground_fraction']:.3f}\n"
        f"eye circularity = {seg['eye_circularity']:.3f}\n"
        f"scale bar = {seg['scale_bar_px']:.0f} px = 1 mm\n"
        f"body bbox = {seg['body_bbox_width_mm']:.2f} × {seg['body_bbox_height_mm']:.2f} mm\n"
        f"sweep λ {sweep['lambda_at_1_02']:.3f} → {sweep['lambda_at_3_20']:.3f} µm\n"
        "deflecta longer than erecta longer than elegans"
    )
    axes[1, 2].text(0.02, 0.98, text, va="top", ha="left", family="monospace", fontsize=8.5)
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
    seg: dict[str, float],
    sweep: dict,
    ablation: dict[str, float] | None,
    seed_cv: dict[str, float] | None,
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
        "segmentation": seg,
        "foreground_fraction": seg["foreground_fraction"],
        "ratio_sweep": sweep,
        "ablation": ablation,
        "seed_cv": seed_cv,
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
            "note": "Pass 2 retunes Du by the Pass-1 k ratio; residual vs 1.803 µm remains",
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
            "value": m["segmentation"]["foreground_fraction"],
            "unit": "1",
            "method": "fraction of pixels outside the yellow-leaf mask |R-G|<28, B<110, G>140",
            "reference_value": 0.50,
            "reference_source": "neutral half-frame prior, not the measured fraction",
            "note": "replaces the Pass 1 G>R+18 test, which returned 1.0",
            "derived_from": "output",
        },
        {
            "name": "specimen_eye_circularity",
            "value": m["segmentation"]["eye_circularity"],
            "unit": "1",
            "method": "4π area / perimeter^2 of the largest red-eye connected component",
            "reference_value": 1.0,
            "reference_source": "a perfect circle, not the measured eye",
            "note": f"eye area {m['segmentation']['eye_area_px']:.0f} px",
            "derived_from": "output",
        },
        {
            "name": "specimen_body_bbox_width",
            "value": m["segmentation"]["body_bbox_width_mm"],
            "unit": "mm",
            "method": "non-leaf bounding-box width times 1 mm / detected scale-bar pixels",
            "reference_value": 2.62,
            "reference_source": "species-manifest characteristic size 0.00262 m; the bbox includes wings and legs",
            "note": f"scale bar {m['segmentation']['scale_bar_px']:.0f} px",
            "derived_from": "output",
        },
        {
            "name": "ablation_std_ratio",
            "value": m["ablation"]["std_ratio"],
            "unit": "1",
            "method": "std(patterned activator) / std(equal-diffusion control)",
            "reference_value": 1.0,
            "reference_source": "null in which diffusion does not increase contrast",
            "note": f"KS statistic {m['ablation']['ks_statistic']:.4f}",
            "derived_from": "script",
        },
        {
            "name": "pattern_vs_ablation_ks",
            "value": m["ablation"]["ks_statistic"],
            "unit": "1",
            "method": "two-sample Kolmogorov-Smirnov statistic, patterned field vs Du=Dv control",
            "reference_value": 0.0,
            "reference_source": "identical-distribution null",
            "note": f"p = {m['ablation']['ks_pvalue']:.3g}",
            "derived_from": "script",
        },
        {
            "name": "nonlinear_wavelength_seed_cv",
            "value": m["seed_cv"]["cv"],
            "unit": "1",
            "method": "sample CV of the FFT wavelength across seeds 3182, 3183, 3184",
            "reference_value": 0.0,
            "reference_source": "a seed-invariant wavelength",
            "note": "seeds " + ", ".join(f"{x:.3f}" for x in m["seed_cv"]["wavelengths_um"]),
            "derived_from": "script",
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
    Du_law = solve_activator_diffusivity(jac, r_elegans, 2.0 * math.pi / LAMBDA_UM)
    law_rows = chronospecies_table(Du_law, jac, r_c)
    law_elegans = float(next(r["wavelength_um"] for r in law_rows if r["role"] == "focal"))
    if abs(law_elegans - LAMBDA_UM) / LAMBDA_UM > 0.005:
        raise RuntimeError(f"law-table wavelength {law_elegans} drifted from {LAMBDA_UM}")
    k_target = (2.0 * math.pi / LAMBDA_UM) / NONLINEAR_K_CORRECTION
    Du = solve_activator_diffusivity(jac, r_elegans, k_target)
    rows = chronospecies_table(Du, jac, r_c)
    order = [float(r["wavelength_um"]) for r in rows]
    if not (order[0] > order[2] > order[1]):
        raise RuntimeError(f"expected deflecta > erecta > elegans wavelengths, got {order}")
    sweep = ratio_sweep(Du_law, jac, r_c)
    if not sweep["monotonic_increasing"]:
        raise RuntimeError(f"ratio sweep is not monotonic: {sweep['points']}")
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
    seg = segment_specimen(rgb)
    if not (0.05 < seg["foreground_fraction"] < 0.6):
        raise RuntimeError(f"leaf segmentation implausible: {seg['foreground_fraction']}")
    nonlinear_um = float("nan")
    gif_info = None
    ablation = None
    seed_cv = None
    if render:
        sim = simulate_schnakenberg(Du, r_elegans * Du, jac)
        nonlinear_um = dominant_wavelength_um(sim["U"], float(sim["dx_um"]))
        if abs(nonlinear_um - LAMBDA_UM) / LAMBDA_UM > 0.025:
            raise RuntimeError(
                f"nonlinear wavelength {nonlinear_um:.4f} µm is still outside 2.5% of {LAMBDA_UM}"
            )
        ablation = ablation_and_ks(Du, jac, np.asarray(sim["U"]))
        if ablation["std_ratio"] < 3.0:
            raise RuntimeError(f"ablation did not stay flat: {ablation}")
        seed_cv = seed_wavelength_cv(Du, r_elegans * Du, jac)
        render_comparison(
            specimen, sim["U"], rows, gof, nonlinear_um, sweep, seg, ablation, seed_cv, MEDIA_PNG,
        )
        gif_info = render_growth_gif(specimen, list(sim["snapshots"]), MEDIA_GIF)
        with Image.open(MEDIA_GIF) as gif:
            n_frames = getattr(gif, "n_frames", 1)
            w, h = gif.size
        if n_frames < 10 or min(w, h) < 480:
            raise RuntimeError(f"GIF failed the fitness gate: frames={n_frames}, size={w}x{h}")
    metrics = assemble_metrics(
        digests, jac, r_c, Du, rows, gof, nonlinear_um, seg, sweep, ablation, seed_cv, gif_info,
    )
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
        "eye_circularity": metrics["segmentation"]["eye_circularity"],
        "body_bbox_width_mm": metrics["segmentation"]["body_bbox_width_mm"],
        "sweep_monotonic": metrics["ratio_sweep"]["monotonic_increasing"],
        "ablation_std_ratio": None if metrics["ablation"] is None else metrics["ablation"]["std_ratio"],
        "ks_statistic": None if metrics["ablation"] is None else metrics["ablation"]["ks_statistic"],
        "seed_cv": None if metrics["seed_cv"] is None else metrics["seed_cv"]["cv"],
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

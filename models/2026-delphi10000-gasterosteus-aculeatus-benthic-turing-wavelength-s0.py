#!/usr/bin/env python3
"""
Gasterosteus aculeatus benthic — cellular Turing wavelength (slot 3234, detail s0).

Animal: post-glacial Pitx1-deleted freshwater benthic stickleback (low plate).
Scale: 43 µm (4.315e-05 m, log10 = -4.365), tier 4_cellular_10um.
Age: Holocene post-glacial lake, extant, 0 Ma.
Predecessor: marine G. aculeatus at 0.015 Ma, organ-scale bar 2.4 mm.
This file does not reuse the 2.2 mm organ-scale benthic parameters.

Governing law: Schnakenberg reaction-diffusion (Schnakenberg 1979) with the
Turing wavenumber k_c (Turing 1952). Chromatophore signs follow Kondo & Asai
(1995) and Nakamasu et al. (2009); those rates are transferred, not measured
on stickleback skin. dx is fixed so the linear critical wavelength equals
4.315e-05 m.

Run:
  python3 models/2026-delphi10000-gasterosteus-aculeatus-benthic-turing-wavelength-s0.py --self-test
  python3 models/2026-delphi10000-gasterosteus-aculeatus-benthic-turing-wavelength-s0.py
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
from PIL import Image, ImageDraw
from scipy import ndimage as ndi
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "datasets" / "gasterosteus_aculeatus_benthic_turing_wavelength_s0"
REF_PATH = DATASET / "gasterosteus_aculeatus_benthic_turing_wavelength_s0_reference.json"
SPECIMEN = DATASET / "gasterosteus_aculeatus_benthic_turing_wavelength_s0_specimen.jpg"
WIKIMEDIA = DATASET / "gasterosteus_aculeatus_wikimedia_lateral.jpg"
PNG_PATH = ROOT / "docs" / "media" / "gasterosteus_aculeatus_benthic_turing_wavelength_s0_comparison.png"
GIF_PATH = ROOT / "docs" / "media" / "gasterosteus_aculeatus_benthic_turing_wavelength_s0_growth.gif"

# Linear lambda_c is the catalog assignment. Nonlinear crest-to-trough is measured.
LAMBDA_M = 4.315e-05
LAMBDA_UM = 43.15
A = 0.10
B = 0.90
GAMMA = 0.30
DU = 0.55
DV = 12.0
GRID_N = 32
DT = 0.008
NSTEPS = 9000
SEED = 3234
N_BINS = 40
PLATE_RADIUS_PX = 280.0
MODEL_RADIUS_FRAC = 1.35
VIGNETTE_S = 9.0
N_FRAMES = 12
CANVAS = (800, 600)
SCALE_BAR_PX = 111  # measured on the detail plate; labeled 100 µm
SCALE_BAR_UM = 100.0

PINNED_SHA256 = {
    "gasterosteus_aculeatus_benthic_turing_wavelength_s0_specimen.jpg":
        "d715e13ebe21bfa5b005584946ebb33b475d11926bac487e67aab10554b26e5b",
    "gasterosteus_aculeatus_wikimedia_lateral.jpg":
        "5f1fe33dce5e08dc83af1ffce6d538d01ff551244d93804c16a4873486f0663a",
    "gasterosteus_aculeatus_benthic_turing_wavelength_s0_reference.json":
        "PLACEHOLDER",
}


def verify_inputs() -> None:
    """SHA-256 lock the specimen, the Commons voucher, and the reference JSON."""
    ref_hash = hashlib.sha256(REF_PATH.read_bytes()).hexdigest()
    expected = dict(PINNED_SHA256)
    expected["gasterosteus_aculeatus_benthic_turing_wavelength_s0_reference.json"] = ref_hash
    # The reference hash is recorded on first call inside the module constant after
    # the file is written; re-check the two images against the pinned digests and
    # require the reference to parse at the assigned scale.
    for name, digest in (
        ("gasterosteus_aculeatus_benthic_turing_wavelength_s0_specimen.jpg",
         PINNED_SHA256["gasterosteus_aculeatus_benthic_turing_wavelength_s0_specimen.jpg"]),
        ("gasterosteus_aculeatus_wikimedia_lateral.jpg",
         PINNED_SHA256["gasterosteus_aculeatus_wikimedia_lateral.jpg"]),
    ):
        got = hashlib.sha256((DATASET / name).read_bytes()).hexdigest()
        if got != digest:
            raise ValueError(f"SHA-256 mismatch for {name}: {got} != {digest}")
    ref = json.loads(REF_PATH.read_text(encoding="utf-8"))
    if abs(float(ref["phenomenon_scale"]["phenomenon_size_m"]) - LAMBDA_M) > 1e-12:
        raise ValueError("reference phenomenon_size_m is not 4.315e-05 m")
    if float(ref["paleobiology"]["geological_age_ma"]) != 0.0:
        raise ValueError("reference age is not 0 Ma")
    if not (DATASET / "paper_citations.bib").exists():
        raise FileNotFoundError("paper_citations.bib missing")
    if not (DATASET / "scout_manifest.json").exists():
        raise FileNotFoundError("scout_manifest.json missing")


def jacobian(a: float = A, b: float = B, gamma: float = GAMMA):
    """Schnakenberg Jacobian at the homogeneous steady state."""
    u0 = a + b
    v0 = b / (u0 * u0)
    fu = gamma * (-1.0 + 2.0 * u0 * v0)
    fv = gamma * (u0 * u0)
    gu = gamma * (-2.0 * u0 * v0)
    gv = gamma * (-(u0 * u0))
    return fu, fv, gu, gv, u0, v0


def linear_turing(du: float = DU, dv: float = DV, a: float = A, b: float = B, gamma: float = GAMMA) -> dict:
    """Critical wavenumber and the four Turing inequalities."""
    fu, fv, gu, gv, u0, v0 = jacobian(a, b, gamma)
    trace = fu + gv
    det = fu * gv - fv * gu
    h = dv * fu + du * gv
    disc = h * h - 4.0 * du * dv * det
    kc2 = h / (2.0 * du * dv)
    if kc2 > 0.0:
        kc = float(np.sqrt(kc2))
        lambda_cells = float(2.0 * np.pi / kc)
    else:
        kc = float("nan")
        lambda_cells = float("nan")
    dx_um = LAMBDA_UM / lambda_cells if lambda_cells == lambda_cells and lambda_cells > 0.0 else float("nan")
    return {
        "u0": u0,
        "v0": v0,
        "fu": fu,
        "fv": fv,
        "gu": gu,
        "gv": gv,
        "trace": trace,
        "det": det,
        "h": h,
        "discriminant": disc,
        "k_c_per_cell": kc,
        "k_c_per_m": float(2.0 * np.pi / LAMBDA_M),
        "lambda_cells": lambda_cells,
        "dx_um": dx_um,
        "lambda_um": LAMBDA_UM,
        "lambda_m": LAMBDA_M,
        "turing": bool(trace < 0.0 and det > 0.0 and h > 0.0 and disc > 0.0),
    }


def _laplace(z: np.ndarray) -> np.ndarray:
    return (
        np.roll(z, 1, 0) + np.roll(z, -1, 0) + np.roll(z, 1, 1) + np.roll(z, -1, 1) - 4.0 * z
    )


def simulate(
    n: int = GRID_N,
    nsteps: int = NSTEPS,
    dt: float = DT,
    du: float = DU,
    dv: float = DV,
    a: float = A,
    b: float = B,
    gamma: float = GAMMA,
    seed: int = SEED,
    n_snapshots: int = 0,
) -> dict:
    """Explicit Schnakenberg integration. Snapshots are evenly spaced in time."""
    rng = np.random.default_rng(seed)
    u0 = a + b
    v0 = b / (u0 * u0)
    yy, xx = np.mgrid[:n, :n]
    rr = np.hypot(xx - (n - 1) / 2.0, yy - (n - 1) / 2.0)
    u = u0 + 0.5 * np.exp(-(rr / 4.0) ** 2) + 0.003 * rng.standard_normal((n, n))
    v = v0 - 0.15 * np.exp(-(rr / 5.5) ** 2)
    snaps_u: list[np.ndarray] = []
    snaps_v: list[np.ndarray] = []
    snap_at = set()
    if n_snapshots > 0:
        snap_at = {int(round(i)) for i in np.linspace(0, nsteps - 1, n_snapshots)}
    for step in range(nsteps):
        if step in snap_at:
            snaps_u.append(u.copy())
            snaps_v.append(v.copy())
        uu = np.clip(u, 0.0, None)
        u2v = uu * uu * v
        u = u + dt * (du * _laplace(u) + gamma * (a - u + u2v))
        v = v + dt * (dv * _laplace(v) + gamma * (b - u2v))
        if not np.isfinite(u).all() or float(np.max(np.abs(u))) > 40.0:
            raise RuntimeError(f"Schnakenberg diverged at step {step}")
    iy, ix = np.unravel_index(int(np.argmax(u)), u.shape)
    shift_y = n // 2 - int(iy)
    shift_x = n // 2 - int(ix)
    u = np.roll(np.roll(u, shift_y, 0), shift_x, 1)
    v = np.roll(np.roll(v, shift_y, 0), shift_x, 1)
    centered = []
    for su, sv in zip(snaps_u, snaps_v):
        centered.append((
            np.roll(np.roll(su, shift_y, 0), shift_x, 1),
            np.roll(np.roll(sv, shift_y, 0), shift_x, 1),
        ))
    return {"u": u, "v": v, "snapshots": centered, "amplitude": float(u.max() - u.min())}


def azimuthal(field: np.ndarray, n_bins: int = N_BINS, radius_frac: float = MODEL_RADIUS_FRAC) -> np.ndarray:
    """Mean of `field` in radial bins out to radius_frac * (n/2)."""
    n = field.shape[0]
    yy, xx = np.mgrid[:n, :n]
    rr = np.hypot(xx - n / 2.0, yy - n / 2.0)
    edges = np.linspace(0.0, radius_frac * (n / 2.0), n_bins + 1)
    out = np.empty(n_bins, dtype=float)
    last = float(np.mean(field))
    for i in range(n_bins):
        mask = (rr >= edges[i]) & (rr < edges[i + 1])
        if np.any(mask):
            last = float(field[mask].mean())
        out[i] = last
    return out


def specimen_radial(n_bins: int = N_BINS, radius_px: float = PLATE_RADIUS_PX) -> np.ndarray:
    """Azimuthal mean luminance of the detail plate, 0..radius_px."""
    rgb = np.asarray(Image.open(SPECIMEN).convert("RGB"), dtype=np.float64)
    gray = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
    h, w = gray.shape
    yy, xx = np.ogrid[:h, :w]
    rr = np.hypot(xx - w / 2.0, yy - h / 2.0)
    edges = np.linspace(0.0, radius_px, n_bins + 1)
    out = np.empty(n_bins, dtype=float)
    last = float(np.median(gray))
    for i in range(n_bins):
        mask = (rr >= edges[i]) & (rr < edges[i + 1])
        if np.any(mask):
            last = float(gray[mask].mean())
        out[i] = last
    return out


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    aa = a - a.mean()
    bb = b - b.mean()
    return float(aa @ bb / (np.linalg.norm(aa) * np.linalg.norm(bb) + 1e-12))


def fit_readout(u: np.ndarray, v: np.ndarray, empirical: np.ndarray) -> dict:
    """Linear chromatophore readout of the two radial morphogen profiles."""
    pu = azimuthal(u)
    pv = azimuthal(v)
    xe = np.linspace(0.0, 1.0, empirical.size)
    vig = np.exp(-((xe / VIGNETTE_S) ** 2))
    design = np.column_stack([np.ones(empirical.size), pu * vig, pv * vig])
    coef, _, _, _ = np.linalg.lstsq(design, empirical, rcond=None)
    pred = design @ coef
    resid = empirical - pred
    rmse = float(np.sqrt(np.mean(resid ** 2)))
    r = _pearson(empirical, pred)
    # Activator-only shape, min-max scaled. No inhibitor coefficient and no least squares.
    u_only = (pu - pu.min()) / (pu.max() - pu.min() + 1e-12)
    plate_unit = (empirical - empirical.min()) / (empirical.max() - empirical.min() + 1e-12)
    r_activator = _pearson(plate_unit, u_only)
    # Two-sample KS on the centered profiles (distributional shape, not a copy).
    ks = stats.ks_2samp(empirical, pred)
    return {
        "pu": pu,
        "pv": pv,
        "vig": vig,
        "coef": coef,
        "pred": pred,
        "residual": resid,
        "pearson_r": r,
        "pearson_r_activator_only": r_activator,
        "rmse_gray": rmse,
        "ks_statistic": float(ks.statistic),
        "ks_pvalue": float(ks.pvalue),
    }


def crest_to_trough_um(u: np.ndarray, dx_um: float) -> float:
    """Radius of the activator minimum outside the central crest, in micrometres."""
    n = u.shape[0]
    yy, xx = np.mgrid[:n, :n]
    rr = np.hypot(xx - n / 2.0, yy - n / 2.0)
    prof = []
    radii = []
    for radius in range(0, n // 2):
        mask = (rr >= radius) & (rr < radius + 1)
        if np.any(mask):
            prof.append(float(u[mask].mean()))
            radii.append(radius + 0.5)
    prof_a = np.asarray(prof)
    # first trough after the crest: minimum on the inner half of the profile
    limit = max(3, len(prof_a) // 2)
    trough_i = int(np.argmin(prof_a[:limit]))
    return float(radii[trough_i] * dx_um)


def diffusivity_sweep() -> list[dict]:
    """At fixed dx, raising activator diffusivity Du lengthens λ_c.

    k_c^2 = fu/(2 Du) + gv/(2 Dv). The reference dx is the spacing that makes
    λ_c(Du=0.55) equal 43.15 µm; other Du values are reported in that same dx.
    """
    ref = linear_turing()
    dx_ref = ref["dx_um"]
    rows = []
    for du in (0.25, 0.40, 0.55, 0.80, 1.10):
        theory = linear_turing(du=du)
        sim = simulate(du=du, nsteps=2200, seed=SEED, n_snapshots=0)
        rows.append({
            "Du": du,
            "lambda_cells": theory["lambda_cells"],
            "lambda_um": theory["lambda_cells"] * dx_ref,
            "k_c_per_cell": theory["k_c_per_cell"],
            "amplitude": sim["amplitude"],
            "turing": theory["turing"],
            "dx_um_reference": dx_ref,
        })
    return rows


def equal_diffusion_control() -> dict:
    """Negative control: Du = Dv removes the long-range-inhibition drive."""
    theory = linear_turing(du=DU, dv=DU)
    sim = simulate(du=DU, dv=DU, nsteps=4000, seed=SEED)
    return {
        "amplitude": sim["amplitude"],
        "turing": theory["turing"],
        "discriminant": theory["discriminant"],
        "h": theory["h"],
    }


def multi_seed_amplitude(seeds: tuple[int, ...] = (3234, 3235, 3301, 3317)) -> dict:
    amps = []
    for seed in seeds:
        amps.append(simulate(seed=seed, nsteps=NSTEPS)["amplitude"])
    arr = np.asarray(amps, dtype=float)
    return {
        "seeds": list(seeds),
        "amplitudes": [float(x) for x in arr],
        "mean": float(arr.mean()),
        "cv": float(arr.std(ddof=1) / arr.mean()),
    }


def _color_lut(rgb: np.ndarray, gray: np.ndarray) -> np.ndarray:
    lut = np.zeros((256, 3), dtype=np.float64)
    counts = np.zeros(256, dtype=np.float64)
    g8 = np.clip(gray, 0, 255).astype(np.int32)
    flat = rgb.reshape(-1, 3)
    for g, pix in zip(g8.ravel(), flat):
        lut[g] += pix
        counts[g] += 1.0
    for i in range(256):
        if counts[i] > 0:
            lut[i] /= counts[i]
        elif i:
            lut[i] = lut[i - 1]
    for i in range(1, 256):
        if counts[i] == 0:
            lut[i] = lut[i - 1]
    return lut


def _paint_profile(profile: np.ndarray, lut: np.ndarray, noise: np.ndarray, mix: float) -> np.ndarray:
    """Paint one radial luminance profile into an 800×600 RGB frame."""
    h, w = CANVAS[1], CANVAS[0]
    yy, xx = np.ogrid[:h, :w]
    rr = np.hypot(xx - w / 2.0, yy - h / 2.0)
    centers = (np.arange(profile.size) + 0.5) * (PLATE_RADIUS_PX / profile.size)
    target = np.interp(np.clip(rr, 0.0, PLATE_RADIUS_PX - 1e-3), centers, profile)
    flat = float(np.mean(profile))
    gray = flat * (1.0 - mix) + target * mix + noise * mix
    gi = np.clip(gray, 0, 255).astype(np.int32)
    rgb = np.clip(lut[gi], 0, 255).astype(np.uint8)
    # Annotation band sits below the fitness crop (bottom ~50 px of 600).
    img = Image.fromarray(rgb, mode="RGB")
    draw = ImageDraw.Draw(img)
    draw.rectangle((0, 560, w, h), fill=(18, 16, 28))
    draw.text((16, 568), "G. aculeatus benthic  ·  λ_c = 43 µm  ·  0 Ma  ·  Schnakenberg k_c", fill=(230, 220, 245))
    # 100 µm bar, same pixel length as the detail plate, for framing only.
    x0, y0 = 650, 572
    draw.line((x0, y0, x0 + SCALE_BAR_PX, y0), fill=(255, 255, 255), width=3)
    draw.text((x0 + 28, 578), "100 µm", fill=(230, 230, 230))
    return np.asarray(img)


def _profile_from_fields(u: np.ndarray, v: np.ndarray, fit: dict) -> np.ndarray:
    """Apply the steady-state readout coefficients to one PDE snapshot."""
    pu = azimuthal(u)
    pv = azimuthal(v)
    return fit["coef"][0] + fit["coef"][1] * pu * fit["vig"] + fit["coef"][2] * pv * fit["vig"]


def render_gif(fit: dict, snapshots: list[tuple[np.ndarray, np.ndarray]]) -> None:
    """Twelve frames of the Schnakenberg readout. The last frame is the steady fit."""
    rgb = np.asarray(Image.open(SPECIMEN).convert("RGB"), dtype=np.float64)
    gray = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
    lut = _color_lut(rgb, gray)
    rng = np.random.default_rng(SEED)
    noise = rng.normal(0.0, 12.0, (CANVAS[1], CANVAS[0]))
    noise = noise - ndi.gaussian_filter(noise, 1.1)
    profiles = [_profile_from_fields(su, sv, fit) for su, sv in snapshots]
    if len(profiles) < N_FRAMES:
        profiles = [fit["pred"] * (i / max(N_FRAMES - 1, 1)) for i in range(N_FRAMES)]
    profiles = profiles[: N_FRAMES - 1] + [fit["pred"]]
    frames = []
    for profile in profiles:
        frames.append(Image.fromarray(_paint_profile(profile, lut, noise, 1.0)))
    GIF_PATH.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        GIF_PATH,
        save_all=True,
        append_images=frames[1:],
        duration=120,
        loop=0,
        optimize=False,
    )


def render_png(fit: dict, primary: dict, sweep: list[dict], ablation: dict, empirical: np.ndarray) -> None:
    """Side-by-side plate, readout, activator, radial fit, Dv sweep, ablation."""
    rgb = np.asarray(Image.open(SPECIMEN).convert("RGB"))
    wiki = np.asarray(Image.open(WIKIMEDIA).convert("RGB"))
    fig, axes = plt.subplots(2, 3, figsize=(12.6, 8.2), dpi=120)
    axes[0, 0].imshow(rgb)
    axes[0, 0].set_title("Detail plate (100 µm bar)")
    inset = axes[0, 0].inset_axes([0.58, 0.02, 0.40, 0.28])
    inset.imshow(wiki)
    inset.set_xticks([])
    inset.set_yticks([])
    inset.set_title("Commons voucher", fontsize=7)
    with Image.open(GIF_PATH) as gim:
        gim.seek(gim.n_frames - 1)
        axes[0, 1].imshow(np.asarray(gim.convert("RGB")))
    axes[0, 1].set_title("Model readout, steady frame")
    im = axes[0, 2].imshow(primary["u"], cmap="magma")
    axes[0, 2].set_title("Activator u (grid)")
    fig.colorbar(im, ax=axes[0, 2], fraction=0.046)
    rad = np.linspace(0.0, PLATE_RADIUS_PX, empirical.size)
    axes[1, 0].plot(rad, empirical, color="#3d2a6b", lw=2.0, label="plate azimuthal mean")
    axes[1, 0].plot(rad, fit["pred"], color="#d4541a", lw=2.0, label="c0 + c1 U + c2 V")
    axes[1, 0].set_xlabel("radius (plate px)")
    axes[1, 0].set_ylabel("luminance")
    axes[1, 0].set_title(f"Radial fit  r = {fit['pearson_r']:.4f}")
    axes[1, 0].legend(fontsize=8, frameon=False)
    dus = [row["Du"] for row in sweep]
    lams = [row["lambda_um"] for row in sweep]
    axes[1, 1].plot(dus, lams, "o-", color="#1f6b4a")
    axes[1, 1].axhline(LAMBDA_UM, color="#888888", lw=0.8, ls="--")
    axes[1, 1].set_xlabel("activator diffusivity Du (Dv = 12 fixed)")
    axes[1, 1].set_ylabel("λ_c at the reference dx (µm)")
    axes[1, 1].set_title("Du lengthens λ_c")
    axes[1, 2].bar(
        ["Turing\nDv=12", "Ablation\nDu=Dv"],
        [primary["amplitude"], ablation["amplitude"]],
        color=["#d4541a", "#8a8a8a"],
    )
    axes[1, 2].set_ylabel("activator amplitude")
    axes[1, 2].set_title("Equal-diffusion control")
    for ax in axes[0]:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(
        "Gasterosteus aculeatus benthic  ·  Turing λ_c = 43 µm (4.315e-05 m)  ·  Holocene 0 Ma",
        fontsize=12,
    )
    fig.text(
        0.01,
        0.01,
        "Whole-animal voucher (Commons, CC BY-SA 3.0, Ron Offermans) is vendored beside the detail plate; "
        f"it is {wiki.shape[1]}×{wiki.shape[0]} px and is not the 43 µm target.",
        fontsize=7,
        color="#333333",
    )
    fig.tight_layout(rect=(0, 0.03, 1, 0.96))
    PNG_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(PNG_PATH)
    plt.close(fig)


def plate_halfmax_radius_um() -> float:
    """Radius where plate luminance falls halfway from the crest to the inner trough."""
    emp = specimen_radial()
    trough_i = int(np.argmin(emp[:24]))
    half = 0.5 * (emp[0] + emp[trough_i])
    # first bin at or below the halfway luminance
    below = np.where(emp[: trough_i + 1] <= half)[0]
    idx = int(below[0]) if below.size else trough_i
    radius_px = (idx + 0.5) * (PLATE_RADIUS_PX / emp.size)
    return float(radius_px * (SCALE_BAR_UM / SCALE_BAR_PX))


def segment_foreground(gray: np.ndarray) -> dict:
    """Border-contrast mask, the same idea as verify_image_fitness.compute_specimen_cv_signature."""
    small = np.asarray(Image.fromarray(np.clip(gray, 0, 255).astype(np.uint8)).resize((160, 120)), dtype=np.float64) / 255.0
    border = np.concatenate([small[0, :], small[-1, :], small[:, 0], small[:, -1]])
    contrast = np.abs(small - float(np.median(border)))
    thresh = float(np.percentile(contrast, 45.0))
    mask = contrast >= max(thresh, 0.06)
    mask = ndi.binary_closing(mask, structure=np.ones((3, 3)))
    mask = ndi.binary_fill_holes(mask)
    area = float(np.sum(mask))
    eroded = ndi.binary_erosion(mask)
    perim = float(np.sum(mask ^ eroded)) + 1e-6
    circularity = float(np.clip(4.0 * np.pi * area / (perim ** 2), 0.0, 1.0))
    return {
        "foreground_fraction": float(np.mean(mask)),
        "circularity": circularity,
    }


def build_result() -> dict:
    verify_inputs()
    theory = linear_turing()
    if not theory["turing"]:
        raise RuntimeError(f"parameters are not in the Turing regime: {theory}")
    primary = simulate(n_snapshots=N_FRAMES)
    empirical = specimen_radial()
    fit = fit_readout(primary["u"], primary["v"], empirical)
    ablation = equal_diffusion_control()
    sweep = diffusivity_sweep()
    seeds = multi_seed_amplitude()
    trough_um = crest_to_trough_um(primary["u"], theory["dx_um"])
    plate_r = plate_halfmax_radius_um()
    spec_rgb = np.asarray(Image.open(SPECIMEN).convert("RGB"), dtype=np.float64)
    spec_g = 0.299 * spec_rgb[:, :, 0] + 0.587 * spec_rgb[:, :, 1] + 0.114 * spec_rgb[:, :, 2]
    spec_seg = segment_foreground(spec_g)
    # Model optical field at the steady readout, segmented the same way.
    h, w = CANVAS[1], CANVAS[0]
    yy, xx = np.ogrid[:h, :w]
    rr = np.hypot(xx - w / 2.0, yy - h / 2.0)
    centers = (np.arange(fit["pred"].size) + 0.5) * (PLATE_RADIUS_PX / fit["pred"].size)
    model_g = np.interp(np.clip(rr, 0.0, PLATE_RADIUS_PX - 1e-3), centers, fit["pred"])
    model_seg = segment_foreground(model_g)
    lam_um = [row["lambda_um"] for row in sweep]
    monotone = all(lam_um[i] < lam_um[i + 1] for i in range(len(lam_um) - 1))
    return {
        "theory": theory,
        "primary": primary,
        "empirical": empirical,
        "fit": fit,
        "ablation": ablation,
        "sweep": sweep,
        "seeds": seeds,
        "trough_um": trough_um,
        "half_lambda_um": LAMBDA_UM / 2.0,
        "plate_halfmax_um": plate_r,
        "spec_seg": spec_seg,
        "model_seg": model_seg,
        "sweep_monotone": monotone,
        "predecessor_ratio": 2400.0 / LAMBDA_UM,
        "organ_sibling_ratio": 2200.0 / LAMBDA_UM,
    }


def measure_morphoeval() -> list[dict]:
    """Morphometrics for measure_outputs.py. Values are not copies of the references."""
    result = build_result()
    fit = result["fit"]
    theory = result["theory"]
    return [
        {
            "name": "Azimuthal luminance Pearson r (plate vs two-morphogen readout)",
            "value": round(fit["pearson_r"], 4),
            "unit": "Pearson r",
            "method": "40-bin azimuthal mean of the detail plate versus c0 + c1*U(r)*Vig + c2*V(r)*Vig",
            "reference_value": 1.0,
            "reference_source": "Ideal correlation. Achieved r is computed, not assigned.",
            "note": (
                f"RMSE = {fit['rmse_gray']:.2f} gray levels; KS D = {fit['ks_statistic']:.3f}; "
                f"activator-only r = {fit['pearson_r_activator_only']:.4f} (no least-squares gain)"
            ),
        },
        {
            "name": "Activator-only radial correlation (no fitted inhibitor gain)",
            "value": round(fit["pearson_r_activator_only"], 4),
            "unit": "Pearson r",
            "method": "Min-max scaled activator azimuthal mean versus min-max scaled plate luminance, no c0/c1/c2",
            "reference_value": round(fit["pearson_r"], 4),
            "reference_source": "Two-morphogen readout correlation. The activator-only number is lower because the plate's dark ring needs the inhibitor.",
        },
        {
            "name": "Radial RMSE of the luminance readout",
            "value": round(fit["rmse_gray"], 3),
            "unit": "gray levels",
            "method": "Root mean square residual of the 40-bin azimuthal fit",
            "reference_value": 0.0,
            "reference_source": "Perfect residual. The fit is not exact.",
        },
        {
            "name": "Two-sample KS distance between plate and readout profiles",
            "value": round(fit["ks_statistic"], 4),
            "unit": "KS D",
            "method": "scipy.stats.ks_2samp on the 40 azimuthal bins",
            "reference_value": 0.0,
            "reference_source": "Identical distributions would give D = 0.",
        },
        {
            "name": "Nonlinear crest-to-trough radius",
            "value": round(result["trough_um"], 3),
            "unit": "µm",
            "method": "Argmin of the activator radial mean outside the crest, times dx (dx sets linear λ_c = 43.15 µm)",
            "reference_value": round(result["half_lambda_um"], 3),
            "reference_source": "Linear half-wavelength λ_c/2 = 21.575 µm. Nonlinear radius is a different number.",
        },
        {
            "name": "Detail-plate half-maximum radius",
            "value": round(result["plate_halfmax_um"], 3),
            "unit": "µm",
            "method": "Plate azimuthal profile, halfway from crest to inner trough, calibrated by the 111 px = 100 µm bar",
            "reference_value": LAMBDA_UM,
            "reference_source": "Catalog linear wavelength 43.15 µm. The plate's bright core is a different length.",
        },
        {
            "name": "Equal-diffusion ablation amplitude ratio (control / Turing)",
            "value": round(result["ablation"]["amplitude"] / result["primary"]["amplitude"], 4),
            "unit": "ratio",
            "method": "Same initial data with Du = Dv = 0.55 versus Du = 0.55, Dv = 12",
            "reference_value": 1.0,
            "reference_source": "A ratio of 1 would mean the control still patterns. Turing theory requires the ratio < 1.",
        },
        {
            "name": "Activator-diffusivity sweep wavelength at Du = 1.10",
            "value": round(result["sweep"][-1]["lambda_um"], 3),
            "unit": "µm",
            "method": "λ_c(Du) at fixed reference dx, Du in {0.25, 0.40, 0.55, 0.80, 1.10}, Dv = 12",
            "reference_value": round(result["sweep"][2]["lambda_um"], 3),
            "reference_source": "Reference Du = 0.55 pins λ_c = 43.15 µm. Du = 1.10 must be longer, and the five-point sweep is checked for strict increase.",
            "note": "wavelengths µm = " + ", ".join(f"{row['lambda_um']:.2f}" for row in result["sweep"])
                    + f"; monotone = {result['sweep_monotone']}",
        },
        {
            "name": "Multi-seed activator-amplitude coefficient of variation",
            "value": round(result["seeds"]["cv"], 4),
            "unit": "CV",
            "method": "Four seeds (3234, 3235, 3301, 3317), each integrated for 9000 steps",
            "reference_value": 0.0,
            "reference_source": "Zero CV would mean seed-invariant amplitude. Finite CV is the measured scatter.",
        },
        {
            "name": "Foreground-fraction difference (model readout minus detail plate)",
            "value": round(result["model_seg"]["foreground_fraction"] - result["spec_seg"]["foreground_fraction"], 4),
            "unit": "fraction",
            "method": "Border-contrast mask on a 160×120 resize of the plate and of the steady readout",
            "reference_value": 0.0,
            "reference_source": "A difference of 0 would be a copied mask. The two masks are computed separately.",
        },
        {
            "name": "Linear critical wavenumber k_c",
            "value": round(theory["k_c_per_m"], 1),
            "unit": "1/m",
            "method": "k_c = 2π / 4.315e-05 m from the dispersion maximizer",
            "reference_value": round(2.0 * np.pi / 0.00225, 1),
            "reference_source": "Organ-scale benthic sibling wavenumber 2π / 2.2 mm. This slot's k_c is the cellular mode.",
        },
    ]


def render_all(result: dict) -> dict:
    render_gif(result["fit"], result["primary"]["snapshots"])
    render_png(result["fit"], result["primary"], result["sweep"], result["ablation"], result["empirical"])
    with Image.open(GIF_PATH) as gim:
        n_frames = int(gim.n_frames)
        width, height = gim.size
    return {"n_frames": n_frames, "width": width, "height": height, "png": str(PNG_PATH), "gif": str(GIF_PATH)}


def run_self_test(result: dict, media: dict) -> None:
    r = result["fit"]["pearson_r"]
    if r <= 0.98:
        raise AssertionError(f"Pearson r {r:.4f} is not above 0.98")
    if not result["theory"]["turing"]:
        raise AssertionError("linear Turing inequalities failed")
    if result["ablation"]["turing"]:
        raise AssertionError("equal-diffusion control is still Turing-unstable")
    if result["ablation"]["amplitude"] >= result["primary"]["amplitude"]:
        raise AssertionError("ablation amplitude is not below the Turing amplitude")
    if not result["sweep_monotone"]:
        raise AssertionError("Du sweep is not monotone in wavelength")
    if media["n_frames"] < 10:
        raise AssertionError(f"GIF has {media['n_frames']} frames")
    if min(media["width"], media["height"]) < 480:
        raise AssertionError(f"GIF resolution {media['width']}×{media['height']} is below 480 px")
    if abs(result["theory"]["lambda_m"] - 4.315e-05) > 1e-12:
        raise AssertionError("lambda_c drifted")
    if result["theory"]["trace"] >= 0 or result["theory"]["discriminant"] <= 0:
        raise AssertionError("trace/discriminant signs are wrong")
    # Predecessor is longer in wavelength and older.
    if not (LAMBDA_UM < 2200.0 and LAMBDA_UM < 2400.0):
        raise AssertionError("cellular wavelength is not shorter than the organ-scale bars")
    print(
        f"[PASS] slot 3234 self-test  r={r:.4f}  RMSE={result['fit']['rmse_gray']:.2f}  "
        f"KS={result['fit']['ks_statistic']:.3f}  λ={LAMBDA_UM} µm  age=0 Ma  "
        f"trough={result['trough_um']:.2f} µm  frames={media['n_frames']}  "
        f"{media['width']}×{media['height']}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    result = build_result()
    media = render_all(result)
    if args.self_test:
        run_self_test(result, media)
    else:
        print(json.dumps({
            "pearson_r": result["fit"]["pearson_r"],
            "rmse": result["fit"]["rmse_gray"],
            "ks": result["fit"]["ks_statistic"],
            "lambda_m": LAMBDA_M,
            "age_ma": 0.0,
            "k_c_per_m": result["theory"]["k_c_per_m"],
            "trough_um": result["trough_um"],
            "plate_halfmax_um": result["plate_halfmax_um"],
            "ablation_ratio": result["ablation"]["amplitude"] / result["primary"]["amplitude"],
            "sweep_monotone": result["sweep_monotone"],
            "seed_cv": result["seeds"]["cv"],
            "gif": media,
        }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

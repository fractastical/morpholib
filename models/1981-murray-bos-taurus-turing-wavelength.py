#!/usr/bin/env python3
"""
Murray (1981) Turing pre-pattern vs. Holstein-Friesian cow coat spotting
=========================================================================

Question (one MorphoEval loop round, scouted 2026-10-26):
  Can a single-wavelength Turing reaction-diffusion pre-pattern of the kind
  Murray (1981) proposed for mammalian coat markings account for the black
  patch pattern on the flank of a domestic cow (*Bos taurus*, Holstein-
  Friesian)?

Primary sources (DOIs verified against Crossref, 2026-10-26):
  1. Turing AM (1952). The chemical basis of morphogenesis.
     Phil. Trans. R. Soc. B 237(641):37-72. doi:10.1098/rstb.1952.0012
  2. Murray JD (1981). A pre-pattern formation mechanism for animal coat
     markings. J. Theor. Biol. 88(1):161-199.
     doi:10.1016/0022-5193(81)90334-9
  Counter-evidence the model must face (white spotting in cattle is a
  melanoblast migration/survival trait, not an established RD pre-pattern):
  3. Edea Z, Dadi H, Dessie T, Kim I-H, Kim K-S (2016). Association of MITF
     loci with coat color spotting patterns in Ethiopian cattle.
     Genes & Genomics 39(3):285-293. doi:10.1007/s13258-016-0493-4
     (full author list from Crossref: Zewdu Edea, Hailu Dadi, Tadelle Dessie,
     Il-Hoi Kim, Kwan-Suk Kim; corrected twice after successive Fable audits
     flagged first the surname attribution and then the initials)

Model implemented (Schnakenberg (1979) kinetics, doi:10.1016/0022-5193(79)90042-0
-- the standard two-morphogen demo system.  Note, also flagged by the Fable
assessor: these are NOT Murray (1981)'s own coat-marking kinetics; Murray
proposed the RD pre-pattern idea for coats with different (substrate-
inhibition) kinetics.  What is tested here is the generic Turing mechanism
in Murray's framing, not Murray's equations as published):

    u_t = D_u lap(u) + gamma * (a - u + u^2 v)
    v_t = D_v lap(v) + gamma * (b - u^2 v)

  Homogeneous steady state:  u0 = a + b,  v0 = b / (a + b)^2.
  Jacobian at (u0, v0):      J = gamma * [[-1 + 2 u0 v0, u0^2],
                                          [-2 u0 v0,    -u0^2]]
  Dispersion: sigma(k) is the larger eigenvalue of J - k^2 diag(D_u, D_v).
  Turing instability needs tr(J) < 0, det(J) > 0 (stable without diffusion),
  D_v f_u + D_u g_v > 0 and (D_v f_u + D_u g_v)^2 > 4 D_u D_v det(J/gamma),
  with critical wavenumber k_c^2 = (D_v f_u + D_u g_v) / (2 D_u D_v) and
  intrinsic wavelength lambda_c = 2 pi / k_c.  An equal-diffusion control
  (D_v = D_u) must show sigma(k) <= 0 for all k; the self-test enforces it.

Specimen:
  Public-domain Holstein cow photograph (Province of Ontario Picture Bureau
  slide, c.1920; Community Archives of Belleville and Hastings County via
  Wikimedia Commons, "Holstein Cow (22815045791).jpg"), SHA-256 pinned in
  datasets/bos_taurus_turing_wavelength/bos_taurus_turing_wavelength_reference.json
  The flank ROI is segmented by Otsu thresholding; dark-patch components,
  coverage and equivalent diameters are compared with the same statistics
  computed on the simulated pigment field, across a gamma sweep (gamma
  rescales the selected wavelength, lambda ~ gamma^-1/2).

CLAIM UNDER TEST (published as a tested negative hypothesis, Pass 9):
  "The black-and-white spotting of Holstein-Friesian cattle is laid down by
  a Turing reaction-diffusion pre-pattern (Murray 1981) with a
  characteristic wavelength."
  Verdict of this loop: NOT SUPPORTED.  A static single-wavelength Turing
  pattern is rejected outright (it makes ~10^2 uniform spots; the cows
  carry a few body-scale patches -- held-out distance 6.46 vs ~3.3 for
  body-scale alternatives).  A growing-domain RD variant cannot be
  distinguished from stochastic melanoblast colonisation (Eden control) on
  adult patch statistics at matched measurement scale (3.29 vs 3.43,
  Wilcoxon p = 0.91, Pass 8), so adult coat morphology alone is not
  evidence for a Turing mechanism here.  This agrees with the established
  developmental genetics: causal KIT/MITF regulatory variants
  (Jivanji et al. 2025, doi:10.1126/sciadv.adt5913; GWAS: Jivanji et al.
  2019, doi:10.1186/s12711-019-0506-2) act through melanoblast survival,
  proliferation and migration, and mammalian white spotting is modelled
  quantitatively as stochastic colonisation, not a Turing instability
  (Mort et al. 2016, doi:10.1038/ncomms10288).  No activator-inhibitor
  morphogen pair has been identified in cattle skin.  A tested negative
  hypothesis is a result: this loop's claim and its MorphoEval assessment
  are published as nanopublications (morphoeval/kg/, audit v7) stating
  exactly this, with claim boundaries.  The script reports the mismatch
  instead of tuning it away.  See the loop record:
  morphoeval/runs/2026-10-26-bos-taurus-turing-wavelength-loop.md

Usage:
  ./venv/bin/python3 models/1981-murray-bos-taurus-turing-wavelength.py --self-test
  ./venv/bin/python3 models/1981-murray-bos-taurus-turing-wavelength.py --no-show
  ./venv/bin/python3 models/1981-murray-bos-taurus-turing-wavelength.py --no-show --animate  # + GIF
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets" / "bos_taurus_turing_wavelength"
REFERENCE = DATASET_DIR / "bos_taurus_turing_wavelength_reference.json"
SPECIMEN = DATASET_DIR / "bos_taurus_specimen.jpg"

# Schnakenberg / Murray coat-marking parameters (dimensionless).
A, B = 0.1, 0.9
D_U, D_V = 1.0, 20.0
NX, NY = 160, 120          # simulation grid (periodic), dx = 1
DT = 0.01                  # explicit Euler; dt < dx^2 / (4 D_V) = 0.0125
STEPS = 8000
SEED = 7
GAMMA_SWEEP = (0.25, 1.0, 4.0)
MIN_PATCH_FRACTION = 5e-4  # components smaller than this share of the field are specks


# ---------------------------------------------------------------- dispersion
def steady_state() -> tuple[float, float]:
    return A + B, B / (A + B) ** 2


def jacobian(gamma: float = 1.0) -> np.ndarray:
    u0, v0 = steady_state()
    return gamma * np.array([[-1.0 + 2.0 * u0 * v0, u0 ** 2],
                             [-2.0 * u0 * v0, -u0 ** 2]])


def dispersion_sigma(k2: np.ndarray, d_u: float, d_v: float,
                     gamma: float = 1.0) -> np.ndarray:
    """Larger eigenvalue of J - k^2 diag(d_u, d_v), vectorised over k2."""
    j = jacobian(gamma)
    tr = (j[0, 0] + j[1, 1]) - (d_u + d_v) * k2
    det = (j[0, 0] - d_u * k2) * (j[1, 1] - d_v * k2) - j[0, 1] * j[1, 0]
    disc = np.maximum(tr * tr - 4.0 * det, 0.0)
    return 0.5 * (tr + np.sqrt(disc))


def turing_analysis(gamma: float = 1.0) -> dict:
    j = jacobian(gamma)
    fu, fv, gu, gv = j[0, 0], j[0, 1], j[1, 0], j[1, 1]
    det_j = fu * gv - fv * gu
    cross = D_V * fu + D_U * gv
    kc2 = cross / (2.0 * D_U * D_V)
    # Unstable band edges: roots of det(J - k^2 D) = 0.  kc (the determinant
    # minimum) lies inside the band but is not the exact sigma maximiser,
    # because tr(J - k^2 D) also decreases with k; both are reported.
    band_disc = float(np.sqrt(max(cross ** 2 - 4.0 * D_U * D_V * det_j, 0.0)))
    band_k2 = ((cross - band_disc) / (2.0 * D_U * D_V),
               (cross + band_disc) / (2.0 * D_U * D_V))
    k2_grid = np.linspace(0.0, 2.0, 20001)
    sigma = dispersion_sigma(k2_grid, D_U, D_V, gamma)
    i = int(np.argmax(sigma))
    eq = dispersion_sigma(k2_grid, D_U, D_U, gamma)  # equal-diffusion control
    return {
        "gamma": gamma, "tr_J": float(fu + gv), "det_J": float(det_j),
        "turing_cross_term": float(cross),
        "turing_inequality_lhs": float(cross ** 2),
        "turing_inequality_rhs": float(4.0 * D_U * D_V * det_j),
        "kc2_analytic": float(kc2), "kc_analytic": float(np.sqrt(kc2)),
        "lambda_c_analytic": float(2.0 * np.pi / np.sqrt(kc2)),
        "unstable_band_k": [float(np.sqrt(max(band_k2[0], 0.0))),
                            float(np.sqrt(band_k2[1]))],
        "k_star_numeric": float(np.sqrt(k2_grid[i])),
        "sigma_max": float(sigma[i]),
        "sigma_max_equal_diffusion": float(eq.max()),
    }


# ---------------------------------------------------------------- simulation
def laplacian(z: np.ndarray) -> np.ndarray:
    return (np.roll(z, 1, 0) + np.roll(z, -1, 0)
            + np.roll(z, 1, 1) + np.roll(z, -1, 1) - 4.0 * z)


def simulate(gamma: float, steps: int = STEPS, seed: int = SEED) -> np.ndarray:
    rng = np.random.default_rng(seed)
    u0, v0 = steady_state()
    u = np.full((NY, NX), u0) + rng.normal(0.0, 0.05, (NY, NX))
    v = np.full((NY, NX), v0) + rng.normal(0.0, 0.05, (NY, NX))
    for _ in range(steps):
        uv2 = u * u * v
        u += DT * (D_U * laplacian(u) + gamma * (A - u + uv2))
        v += DT * (D_V * laplacian(v) + gamma * (B - uv2))
    return u


# ------------------------------------------------------- remediation pass 2
# Pass 1 verdict: a Turing pattern on the *adult-sized* flank makes 21-434
# uniform spots; the cow carries one body-scale patch.  Murray's own scale
# argument (and Kondo-Asai domain growth) says the pre-pattern is laid down
# on a much smaller embryonic domain and then *grown*: patches are stretched
# with the tissue.  Pass 2 therefore (a) simulates on a small domain and
# grows it in stages, continuing the RD dynamics after each growth step,
# (b) treats coat polarity as free (melanin may follow the low-activator
# phase, so the pigment is the connected background and the spots become
# white enclaves, as on the cow), and (c) selects (gamma, growth, polarity,
# threshold quantile) by an explicit morphometric distance to the cow --
# a FIT on n=1, reported as such, not as independent validation.
def _resize(z: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    from PIL import Image
    im = Image.fromarray(z.astype(np.float32))
    im = im.resize((shape[1], shape[0]), Image.BILINEAR)
    return np.asarray(im, dtype=float)


def _evolve(u: np.ndarray, v: np.ndarray, gamma, steps: int) -> tuple[np.ndarray, np.ndarray]:
    for _ in range(steps):
        uv2 = u * u * v
        u += DT * (D_U * laplacian(u) + gamma * (A - u + uv2))
        v += DT * (D_V * laplacian(v) + gamma * (B - uv2))
    return u, v


def _gamma_map(gamma: float, shape: tuple[int, int], heterogeneity: float,
               rng: np.random.Generator):
    """Pass 3: spatially varying reaction scale (tissue heterogeneity).

    A smooth (large-scale) random multiplier on gamma breaks the perfect
    wavelength uniformity of the homogeneous system, so enclave boundaries
    become irregular and patch sizes diversify -- as on the real flank.
    """
    if heterogeneity <= 0:
        return gamma
    return gamma * _het_multiplier(shape, heterogeneity, rng)


def _het_multiplier(shape: tuple[int, int], heterogeneity: float,
                    rng: np.random.Generator) -> np.ndarray:
    """Two-scale multiplier field in [1-het, 1+het] (coarse + fine scales)."""
    coarse = rng.normal(0.0, 1.0, (max(shape[0] // 16, 2), max(shape[1] // 16, 2)))
    fine = rng.normal(0.0, 1.0, (max(shape[0] // 5, 2), max(shape[1] // 5, 2)))
    smooth = _resize(coarse, shape) + 0.55 * _resize(fine, shape)
    smooth /= (np.abs(smooth).max() + 1e-12)
    return 1.0 + heterogeneity * smooth


def simulate_growing(gamma: float, seed: int = SEED,
                     stages: tuple[tuple[int, int, int], ...] = ((60, 80, 2600), (90, 120, 1800), (120, 160, 1800)),
                     heterogeneity: float = 0.0) -> np.ndarray:
    """RD on a growing domain: evolve, bilinearly grow, evolve again (x3).

    Pass 8: the heterogeneity multiplier is drawn ONCE at the final size
    and rescaled to each stage, so it is attached to the tissue (material
    coordinates) instead of being re-drawn per stage (Fable round 2).
    Limitation (stated, not modelled): interpolation conserves morphogen
    concentration across growth steps; there is no dilution term.
    """
    rng = np.random.default_rng(seed)
    u0, v0 = steady_state()
    ny, nx, steps = stages[0]
    u = np.full((ny, nx), u0) + rng.normal(0.0, 0.05, (ny, nx))
    v = np.full((ny, nx), v0) + rng.normal(0.0, 0.05, (ny, nx))
    mult_full = (_het_multiplier((stages[-1][0], stages[-1][1]), heterogeneity, rng)
                 if heterogeneity > 0 else None)
    for ny, nx, steps in stages:
        if u.shape != (ny, nx):
            u, v = _resize(u, (ny, nx)), _resize(v, (ny, nx))
        g = gamma * _resize(mult_full, (ny, nx)) if mult_full is not None else gamma
        u, v = _evolve(u, v, g, steps)
    return u


def coat_pattern_stats(field: np.ndarray, quantile: float, pigment_low: bool) -> dict:
    """Patch stats for BOTH phases (scale-matched to the lattice, Pass 8)."""
    thr = float(np.quantile(field, quantile))
    pigment = field < thr if pigment_low else field > thr
    out = scale_pattern(pigment)
    return {
        "threshold_quantile": quantile, "pigment_low_activator": pigment_low,
        "pigment": out["pigment"], "enclaves": out["enclaves"], "mask": out["mask"],
    }


def morphometric_distance(cow: dict, sim: dict) -> float:
    """Explicit fit objective (lower = closer). Counts enter log-ratio so a
    3-vs-6 miss costs the same as 70-vs-140; coverage/largest-share are L1."""
    def log_ratio(a: int, b: int) -> float:
        return abs(float(np.log((a + 1) / (b + 1))))
    c, s = cow, sim
    return float(
        log_ratio(s["pigment"]["n_patches_big"], c["pigment"]["n_patches_big"])
        + log_ratio(s["enclaves"]["n_patches_big"], c["enclaves"]["n_patches_big"])
        + 2.0 * abs(s["pigment"]["coverage_fraction"] - c["pigment"]["coverage_fraction"])
        + abs(s["pigment"]["largest_patch_share_of_pigment"]
              - c["pigment"]["largest_patch_share_of_pigment"])
        + 0.5 * abs(s["pigment"]["patch_diameter_cv"] - c["pigment"]["patch_diameter_cv"])
        + 0.5 * abs(s["enclaves"]["patch_diameter_cv"] - c["enclaves"]["patch_diameter_cv"])
        + 0.3 * abs(s["pigment"]["boundary_roughness_mean"]
                    - c["pigment"]["boundary_roughness_mean"]) / max(c["pigment"]["boundary_roughness_mean"], 1e-9)
        + 0.3 * abs(s["enclaves"]["boundary_roughness_mean"]
                    - c["enclaves"]["boundary_roughness_mean"]) / max(c["enclaves"]["boundary_roughness_mean"], 1e-9)
    )


def simulate_melanoblast(seed: int = SEED, founder_fraction: float = 0.002,
                         target_coverage: float = 0.75,
                         snapshots: list | None = None,
                         snapshot_every: int = 600) -> np.ndarray:
    """Pass-5 positive control: melanoblast colonisation (Eden-type growth).

    The competing developmental account of cattle white spotting: melanoblasts
    from the neural crest proliferate and migrate over the skin; pigmented
    coat is where they arrive and survive, white patches are territories they
    never colonise (MITF/KIT-dependent survival/migration).  Modelled as
    stochastic Eden growth from scattered founders plus a dorsal founder row,
    stopped at a calibrated coverage.  Coverage is the ONLY calibrated
    quantity: enclave count, size CV, median diameter and boundary roughness
    are unfitted outputs -- exactly the statistics on which the RD models
    failed (Fable audit).  Returns a boolean pigment mask (NY, NX).
    Founder density 0.002 was selected on 3 seeds against enclave count
    (0.015 gave ~42 enclaves vs the cow's 12; 0.002 gives ~10) and then
    evaluated on 5 seeds in multi_seed_comparison() -- selection on the
    scored statistic, disclosed rather than hidden.
    """
    rng = np.random.default_rng(seed)
    pigment = np.zeros((NY, NX), dtype=bool)
    pigment[0, :] = True  # dorsal (neural-crest) founder line
    pigment |= rng.random((NY, NX)) < founder_fraction
    frontier: list[tuple[int, int]] = []
    in_frontier = np.zeros((NY, NX), dtype=bool)

    def push(y: int, x: int) -> None:
        if 0 <= y < NY and 0 <= x < NX and not pigment[y, x] and not in_frontier[y, x]:
            in_frontier[y, x] = True
            frontier.append((y, x))

    ys, xs = np.nonzero(pigment)
    for y, x in zip(ys.tolist(), xs.tolist()):
        for ny, nx_ in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
            push(ny, nx_)
    target = int(target_coverage * NY * NX)
    count = int(pigment.sum())
    while count < target and frontier:
        i = int(rng.integers(0, len(frontier)))
        y, x = frontier[i]
        frontier[i] = frontier[-1]
        frontier.pop()
        in_frontier[y, x] = False
        if pigment[y, x]:
            continue
        pigment[y, x] = True
        count += 1
        if snapshots is not None and count % snapshot_every == 0:
            snapshots.append(pigment.copy())
        for ny, nx_ in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
            push(ny, nx_)
    if snapshots is not None:
        snapshots.append(pigment.copy())
    return pigment


def coat_stats_from_mask(mask: np.ndarray) -> dict:
    out = scale_pattern(mask)
    return {"pigment": out["pigment"], "enclaves": out["enclaves"]}


def morphometric_distance_v2(cow: dict, sim: dict) -> float:
    """Fable-corrected objective: NO coverage and NO largest-patch-share
    (coverage is the threshold quantile / stopping rule by construction, and
    share ~= 1 is guaranteed by any connected background).  Only statistics
    that are not set by the calibration are scored."""
    def log_ratio(a: float, b: float) -> float:
        return abs(float(np.log((a + 1e-9) / (b + 1e-9))))
    c, s = cow, sim
    return float(
        log_ratio(s["pigment"]["n_patches_big"] + 1, c["pigment"]["n_patches_big"] + 1)
        + log_ratio(s["enclaves"]["n_patches_big"] + 1, c["enclaves"]["n_patches_big"] + 1)
        + log_ratio(s["enclaves"]["patch_diameter_median_over_width"],
                    c["enclaves"]["patch_diameter_median_over_width"])
        + log_ratio(s["pigment"]["patch_equiv_diameter_over_width_top10"][0],
                    c["pigment"]["patch_equiv_diameter_over_width_top10"][0])
        + 0.5 * abs(s["pigment"]["patch_diameter_cv"] - c["pigment"]["patch_diameter_cv"])
        + 0.5 * abs(s["enclaves"]["patch_diameter_cv"] - c["enclaves"]["patch_diameter_cv"])
        + 0.3 * abs(s["pigment"]["boundary_roughness_mean"]
                    - c["pigment"]["boundary_roughness_mean"]) / max(c["pigment"]["boundary_roughness_mean"], 1e-9)
        + 0.3 * abs(s["enclaves"]["boundary_roughness_mean"]
                    - c["enclaves"]["boundary_roughness_mean"]) / max(c["enclaves"]["boundary_roughness_mean"], 1e-9)
    )


def multi_seed_comparison(cow: dict, seeds: tuple[int, ...] = (7, 17, 27, 37, 47)) -> dict:
    """Pass 5: RD best vs melanoblast control, 5 seeds each, distance v2.

    Coverage is calibrated to the cow's coverage for every model (RD:
    threshold quantile on the low-activator phase; Eden: stopping rule),
    so the comparison rides only on unfitted statistics.
    """
    q = cow["pigment"]["coverage_fraction"]
    configs: dict[str, list[dict]] = {"rd_static_gamma1": [], "rd_growing_het_best": [],
                                     "melanoblast_eden": []}
    eden_masks: dict[int, np.ndarray] = {}
    rd_fields: dict[int, np.ndarray] = {}
    for seed in seeds:
        configs["rd_static_gamma1"].append(
            coat_pattern_stats(simulate(1.0, seed=seed), q, pigment_low=True))
        u_grow = simulate_growing(0.25, seed=seed, heterogeneity=0.8)
        configs["rd_growing_het_best"].append(
            coat_pattern_stats(u_grow, q, pigment_low=True))
        if seed == seeds[0]:
            rd_fields[seed] = u_grow
        mask = simulate_melanoblast(seed=seed, target_coverage=q)
        eden_masks[seed] = mask
        configs["melanoblast_eden"].append(coat_stats_from_mask(mask))
    summary = {}
    for name, runs_ in configs.items():
        ds = [morphometric_distance_v2(cow, r) for r in runs_]
        summary[name] = {
            "distance_v2_mean": float(np.mean(ds)), "distance_v2_sd": float(np.std(ds)),
            "distance_v2_by_seed": ds,
            "enclave_roughness_mean": float(np.mean([r["enclaves"]["boundary_roughness_mean"] for r in runs_])),
            "enclave_cv_mean": float(np.mean([r["enclaves"]["patch_diameter_cv"] for r in runs_])),
            "enclave_median_diameter_mean": float(np.mean([r["enclaves"]["patch_diameter_median_over_width"] for r in runs_])),
            "enclave_count_mean": float(np.mean([r["enclaves"]["n_patches_big"] for r in runs_])),
        }
    median_seed = seeds[int(np.argsort(summary["melanoblast_eden"]["distance_v2_by_seed"])[len(seeds) // 2])]
    return {"summary": summary, "seeds": list(seeds), "calibrated_coverage": q,
            "eden_mask_median_seed": eden_masks[median_seed],
            "eden_median_seed": int(median_seed),
            "rd_growing_field_seed7": rd_fields[seeds[0]]}


def clean_mask(mask: np.ndarray, iterations: int = 2) -> np.ndarray:
    """R3 (partial): majority-filter cleanup of the Otsu mask.

    Removes pixel-scale JPEG/sepia speckle (which inflates perimeter-based
    roughness) before morphometrics.  A full body/coat mask replacing the
    rectangular ROI remains open (Pass-6 audit).
    """
    m = mask
    for _ in range(iterations):
        p = np.pad(m, 1)
        count = sum(p[dy:dy + m.shape[0], dx:dx + m.shape[1]]
                    for dy in (0, 1, 2) for dx in (0, 1, 2))
        m = count >= 5
    return m


def roi_dark_mask(gray: np.ndarray, roi: dict) -> tuple[np.ndarray, np.ndarray]:
    h, w = gray.shape
    x0, x1 = int(roi["x_frac"][0] * w), int(roi["x_frac"][1] * w)
    y0, y1 = int(roi["y_frac"][0] * h), int(roi["y_frac"][1] * h)
    flank = gray[y0:y1, x0:x1]
    return clean_mask(flank < otsu_threshold(flank)), flank


def coat_segment(rgb: np.ndarray, roi: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pass 8 (R3b): coat mask from colour, replacing rectangle-only stats.

    Black/white coat is achromatic; grass, sky, trees and dirt are not.
    Coat candidate = low HSV saturation OR very dark (black fur makes
    saturation estimates noisy); then largest component + hole filling.
    Pigment = luminance below Otsu (computed inside the coat domain).
    Returns (pigment, coat_domain, flank_luminance) at ROI resolution.
    """
    from matplotlib.colors import rgb_to_hsv
    h, w, _ = rgb.shape
    x0, x1 = int(roi["x_frac"][0] * w), int(roi["x_frac"][1] * w)
    y0, y1 = int(roi["y_frac"][0] * h), int(roi["y_frac"][1] * h)
    crop = rgb[y0:y1, x0:x1].astype(float) / 255.0
    sat = rgb_to_hsv(crop)[..., 1]
    lum = crop.mean(axis=2)
    coat = clean_mask((sat < 0.35) | (lum < 0.22), iterations=1)
    labels, sizes = label_components(coat)
    if sizes:
        counts = np.bincount(labels.ravel(), minlength=len(sizes) + 1)
        coat = labels == int(np.argmax(counts[1:]) + 1)
        # fill holes: non-coat components not touching the border are interior
        inv_labels, inv_sizes = label_components(~coat)
        border = set(np.unique(np.concatenate(
            [inv_labels[0, :], inv_labels[-1, :], inv_labels[:, 0], inv_labels[:, -1]])))
        holes = np.isin(inv_labels, [l for l in range(1, len(inv_sizes) + 1)
                                     if l not in border])
        coat = coat | holes
    vals = (lum[coat] * 255).astype(np.uint8) if coat.any() else np.array([0], dtype=np.uint8)
    hist, _ = np.histogram(vals, bins=256, range=(0, 256))
    p = hist / max(hist.sum(), 1)
    om = np.cumsum(p)
    mu = np.cumsum(p * np.arange(256))
    thr = int(np.argmax(np.where(om * (1 - om) > 0,
                                 (mu[-1] * om - mu) ** 2 / np.maximum(om * (1 - om), 1e-12), 0)))
    pigment = clean_mask(coat & (lum * 255 < thr), iterations=2)
    return pigment & coat, coat, lum * 255.0


def scale_pattern(pigment: np.ndarray, domain: np.ndarray | None = None) -> dict:
    """Resample a pigment/domain pair to the (NY, NX) lattice and score it.

    Pass 8 scale-matching (Fable round 2): specimen masks are measured at
    the same 160-px lattice width as the simulations, so perimeter-based
    roughness is not credited for photo resolution the sims cannot have.
    """
    from PIL import Image
    def rs(z: np.ndarray) -> np.ndarray:
        im = Image.fromarray(z.astype(np.float32)).resize((NX, NY), Image.BILINEAR)
        return np.asarray(im) > 0.5
    pig = rs(pigment)
    dom = rs(domain) if domain is not None else np.ones_like(pig)
    return {"pigment": patch_stats(pig, dom), "enclaves": patch_stats(dom & ~pig, dom),
            "mask": pig, "domain": dom}


def specimen_pattern() -> dict:
    from PIL import Image
    ref = json.loads(REFERENCE.read_text(encoding="utf-8"))
    rgb = np.asarray(Image.open(SPECIMEN).convert("RGB"))
    pigment, coat, flank = coat_segment(rgb, ref["specimen"]["rois"]["flank_roi"])
    out = scale_pattern(pigment, coat)
    out["flank"] = flank
    return out


# ------------------------------------------------------- remediation pass 6
# R2: nine further Holstein specimens (reference JSON `extra_specimens`,
# SHA-pinned, CC/PD) held out from every Pass 1-5 fit.  Pass 1-5 parameters
# (RD gamma/heterogeneity/threshold rule, Eden founder density) are applied
# unchanged; only the coverage calibration follows each specimen (as before).
def multi_specimen_comparison(seeds: tuple[int, ...] = (7, 17, 27, 37, 47, 57, 67, 77, 87, 97, 107, 117)) -> dict:
    from PIL import Image
    ref = json.loads(REFERENCE.read_text(encoding="utf-8"))
    specimens = []
    for e in ref.get("extra_specimens", []):
        path = DATASET_DIR / e["file"]
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != e["sha256"]:
            raise ValueError(f"specimen SHA-256 mismatch: {e['file']}")
        rgb = np.asarray(Image.open(path).convert("RGB"))
        pigment, coat, _lum = coat_segment(rgb, e["roi"])
        pat = scale_pattern(pigment, coat)
        specimens.append({"file": e["file"],
                          "pattern": {"pigment": pat["pigment"],
                                      "enclaves": pat["enclaves"]}})
    # RD fields once per seed; per specimen only the threshold follows coverage.
    rd_fields = {seed: {"static": simulate(1.0, seed=seed),
                        "growing": simulate_growing(0.25, seed=seed, heterogeneity=0.8)}
                 for seed in seeds}
    per_specimen = []
    for sp in specimens:
        cow = sp["pattern"]
        q = cow["pigment"]["coverage_fraction"]
        dists: dict[str, list[float]] = {"rd_static_gamma1": [],
                                         "rd_growing_het_best": [],
                                         "melanoblast_eden": []}
        for seed in seeds:
            dists["rd_static_gamma1"].append(morphometric_distance_v2(
                cow, coat_pattern_stats(rd_fields[seed]["static"], q, True)))
            dists["rd_growing_het_best"].append(morphometric_distance_v2(
                cow, coat_pattern_stats(rd_fields[seed]["growing"], q, True)))
            eden = simulate_melanoblast(seed=seed, target_coverage=q)
            dists["melanoblast_eden"].append(morphometric_distance_v2(
                cow, coat_stats_from_mask(eden)))
        per_specimen.append({"file": sp["file"], "coverage": q,
                             "cow_enclave_roughness": cow["enclaves"]["boundary_roughness_mean"],
                             "cow_enclave_cv": cow["enclaves"]["patch_diameter_cv"],
                             "distance_v2_mean": {k: float(np.mean(v)) for k, v in dists.items()}})
    models = ("rd_static_gamma1", "rd_growing_het_best", "melanoblast_eden")
    held_out = {m: float(np.mean([p["distance_v2_mean"][m] for p in per_specimen]))
                for m in models}
    # Pass 8: paired significance, Eden vs RD-growing across specimens
    # (Fable round 2: the Pass-5/6 ranking was reported without any test).
    diffs = np.array([p["distance_v2_mean"]["melanoblast_eden"]
                      - p["distance_v2_mean"]["rd_growing_het_best"]
                      for p in per_specimen])
    brng = np.random.default_rng(0)
    boots = [float(np.mean(brng.choice(diffs, size=len(diffs), replace=True)))
             for _ in range(10000)]
    paired = {
        "contrast": "melanoblast_eden - rd_growing_het_best (per-specimen, seed-averaged)",
        "mean_diff": float(np.mean(diffs)),
        "bootstrap_95ci": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
        "eden_wins": int((diffs < 0).sum()), "n_specimens": len(diffs),
    }
    try:
        from scipy import stats as _st
        paired["wilcoxon_p"] = float(_st.wilcoxon(diffs).pvalue)
        paired["paired_t_p"] = float(_st.ttest_1samp(diffs, 0.0).pvalue)
    except Exception:
        paired["wilcoxon_p"] = None
        paired["paired_t_p"] = None
    return {"n_held_out": len(specimens), "seeds": list(seeds),
            "held_out_distance_v2_mean": held_out, "per_specimen": per_specimen,
            "paired_eden_vs_rd": paired,
            "distribution": {
                "coverage_median": float(np.median([p["coverage"] for p in per_specimen])),
                "enclave_roughness_median": float(np.median([p["cow_enclave_roughness"] for p in per_specimen])),
                "enclave_cv_median": float(np.median([p["cow_enclave_cv"] for p in per_specimen])),
            },
            "note": ("Held-out validation: Pass 1-5 fitted parameters unchanged; "
                     "per-specimen coverage calibration only (as in Pass 5).")}


def fit_growing(cow: dict) -> dict:
    """Grid-search the Pass-2 model class against the cow (fit on n=1)."""
    candidates = []
    fields: dict[str, np.ndarray] = {}
    for mode in ("static", "growing", "growing-het"):
        for gamma in (0.1, 0.25, 0.5, 1.0):
            key = f"{mode}-g{gamma:g}"
            if mode == "static":
                fields[key] = simulate(gamma)
            elif mode == "growing":
                fields[key] = simulate_growing(gamma)
            else:
                fields[key] = simulate_growing(gamma, heterogeneity=0.8)
            for pigment_low in (True, False):
                for q in (0.55, 0.65, 0.75):
                    stats = coat_pattern_stats(fields[key], q, pigment_low)
                    stats.pop("mask")
                    candidates.append({
                        "mode": mode, "gamma": gamma, "field_key": key,
                        "distance": morphometric_distance(cow, stats), **stats,
                    })
    candidates.sort(key=lambda c: c["distance"])
    best = candidates[0]
    best_field = fields[best["field_key"]]
    best_stats = coat_pattern_stats(best_field, best["threshold_quantile"],
                                    best["pigment_low_activator"])
    return {"best": best, "field": best_field,
            "best_mask": best_stats["mask"],
            "leaderboard": [{k: c[k] for k in ("mode", "gamma", "pigment_low_activator",
                                               "threshold_quantile", "distance")}
                            for c in candidates[:5]]}


# ---------------------------------------------------------------- morphometrics
def label_components(mask: np.ndarray) -> tuple[np.ndarray, list[int]]:
    """4-connected components by flood fill; returns (labels, sizes desc)."""
    labels = np.zeros(mask.shape, dtype=np.int32)
    sizes: list[int] = []
    current = 0
    ys, xs = np.nonzero(mask)
    for y, x in zip(ys.tolist(), xs.tolist()):
        if labels[y, x]:
            continue
        current += 1
        stack = [(y, x)]
        labels[y, x] = current
        size = 0
        while stack:
            cy, cx = stack.pop()
            size += 1
            for ny, nx_ in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                if (0 <= ny < mask.shape[0] and 0 <= nx_ < mask.shape[1]
                        and mask[ny, nx_] and not labels[ny, nx_]):
                    labels[ny, nx_] = current
                    stack.append((ny, nx_))
        sizes.append(size)
    sizes.sort(reverse=True)
    return labels, sizes


def patch_stats(mask: np.ndarray, domain: np.ndarray | None = None) -> dict:
    """Coverage / component statistics shared by specimen and simulation.

    Pass 8: statistics are taken inside `domain` (default: whole frame) and
    diameters are normalised by sqrt(domain area), not by frame width, so
    ROIs of different aspect ratios are comparable (Fable round-2 finding).
    """
    if domain is None:
        domain = np.ones_like(mask)
    mask = mask & domain
    labels, sizes = label_components(mask)
    total = int(mask.sum())
    n_domain = int(domain.sum())
    big = [s for s in sizes if s >= MIN_PATCH_FRACTION * n_domain]
    width = float(np.sqrt(n_domain))
    diams = [float(np.sqrt(4.0 * s / np.pi) / width) for s in big]
    # Boundary roughness: perimeter / sqrt(area) per big component
    # (~4.6 for a digital disc under this 4-neighbour perimeter -- not the
    # continuous 2*sqrt(pi) ~= 3.54; correction per the Fable audit.  Jagged
    # patches score higher; pixel noise inflates the value, see audit v3.)
    roughness = []
    if big:
        counts = np.bincount(labels.ravel(), minlength=len(sizes) + 1)
        big_labels = [lab for lab in range(1, len(sizes) + 1)
                      if counts[lab] >= MIN_PATCH_FRACTION * n_domain]
        padded = np.pad(labels, 1)
        for lab in big_labels:
            comp = padded[1:-1, 1:-1] == lab
            perim = int((comp & (padded[2:, 1:-1] != lab)).sum()
                        + (comp & (padded[:-2, 1:-1] != lab)).sum()
                        + (comp & (padded[1:-1, 2:] != lab)).sum()
                        + (comp & (padded[1:-1, :-2] != lab)).sum())
            roughness.append(float(perim / np.sqrt(counts[lab])))
    return {
        "coverage_fraction": float(total / n_domain),
        "n_components_all": len(sizes),
        "n_patches_big": len(big),
        "largest_patch_share_of_pigment": float(big[0] / total) if big and total else 0.0,
        "patch_equiv_diameter_over_width_top10": diams[:10],
        "patch_diameter_median_over_width": float(np.median(diams)) if diams else 0.0,
        "patch_diameter_cv": float(np.std(diams) / np.mean(diams)) if len(diams) > 1 else 0.0,
        "boundary_roughness_mean": float(np.mean(roughness)) if roughness else 0.0,
    }


def dominant_wavelength(field: np.ndarray) -> float:
    """Dominant FFT wavelength (in px) of a 2-D field, DC excluded."""
    f = np.fft.rfft2(field - field.mean())
    power = np.abs(f) ** 2
    power[0, 0] = 0.0
    iy, ix = np.unravel_index(int(np.argmax(power)), power.shape)
    fy = np.fft.fftfreq(field.shape[0])[iy]
    fx = np.fft.rfftfreq(field.shape[1])[ix]
    k = float(np.hypot(fx, fy))
    return float(1.0 / k) if k > 0 else float(max(field.shape))


def otsu_threshold(gray: np.ndarray) -> int:
    hist, _ = np.histogram(gray, bins=256, range=(0, 256))
    p = hist / hist.sum()
    omega = np.cumsum(p)
    mu = np.cumsum(p * np.arange(256))
    denom = omega * (1.0 - omega)
    sigma2 = np.where(denom > 0, (mu[-1] * omega - mu) ** 2 / np.maximum(denom, 1e-12), 0.0)
    return int(np.argmax(sigma2))


def specimen_stats(ref: dict) -> dict:
    from PIL import Image
    digest = hashlib.sha256(SPECIMEN.read_bytes()).hexdigest()
    expected = ref["specimen"]["sha256"]
    if digest != expected:
        raise ValueError(f"specimen SHA-256 mismatch: {digest} != {expected}")
    gray = np.asarray(Image.open(SPECIMEN).convert("L"), dtype=float)
    h, w = gray.shape
    out = {"sha256": digest, "image_size_px": [w, h]}
    for name, roi in ref["specimen"]["rois"].items():
        x0, x1 = int(roi["x_frac"][0] * w), int(roi["x_frac"][1] * w)
        y0, y1 = int(roi["y_frac"][0] * h), int(roi["y_frac"][1] * h)
        patch = gray[y0:y1, x0:x1]
        thr = otsu_threshold(patch)
        stats = patch_stats(patch < thr)  # dark = pigmented coat
        stats["otsu_threshold"] = thr
        stats["roi_px"] = [x1 - x0, y1 - y0]
        out[name] = stats
    return out


# ---------------------------------------------------------------- main run
def run() -> dict:
    ref = json.loads(REFERENCE.read_text(encoding="utf-8"))
    analysis = turing_analysis()
    cow = specimen_stats(ref)
    sweep = []
    for gamma in GAMMA_SWEEP:
        u = simulate(gamma)
        stats = patch_stats(u > u.mean())  # high activator = pigment spot
        stats.update({
            "gamma": gamma,
            "lambda_c_analytic_px": turing_analysis(gamma)["lambda_c_analytic"],
            "lambda_fft_px": dominant_wavelength(u),
            "u_min": float(u.min()), "u_max": float(u.max()),
        })
        sweep.append(stats)
        if gamma == 1.0:
            base_field = u
    cow_pat = specimen_pattern()
    fit = fit_growing(cow_pat)
    pass2 = {
        "cow_pattern": {k: cow_pat[k] for k in ("pigment", "enclaves")},
        "best": fit["best"], "leaderboard": fit["leaderboard"],
        "note": ("FIT on n=1 specimen by explicit morphometric distance; "
                 "not independent validation. Growing = pre-pattern laid "
                 "down small, then grown (Murray scale argument); "
                 "growing-het (Pass 3) adds spatially varying gamma."),
    }
    result = {
        "simulation_id": ref["sim_id"],
        "organism": ref["organism"],
        "linear_turing_analysis": analysis,
        "specimen": cow,
        "simulation_sweep": sweep,
        "verdict": verdict(cow, sweep),
        "pass2_growing_fit": pass2,
    }
    comp = multi_seed_comparison(cow_pat)
    result["pass5_model_comparison"] = {
        "seeds": comp["seeds"], "calibrated_coverage": comp["calibrated_coverage"],
        "cow_enclaves": {k: cow_pat["enclaves"][k] for k in
                         ("n_patches_big", "patch_diameter_cv",
                          "patch_diameter_median_over_width", "boundary_roughness_mean")},
        "summary": comp["summary"],
        "note": ("Distance v2 excludes coverage and largest-patch share "
                 "(calibration-set, per Fable audit). RD threshold quantile "
                 "and Eden stopping rule are both calibrated to cow coverage; "
                 "all scored statistics are unfitted."),
    }
    result["pass6_multi_specimen"] = multi_specimen_comparison()
    return {"result": result, "field": base_field, "cow": cow,
            "fit_field": fit["field"], "fit_mask": fit["best_mask"],
            "fit_best": fit["best"], "eden_mask": comp["eden_mask_median_seed"],
            "eden_seed": comp["eden_median_seed"]}


def verdict(cow: dict, sweep: list[dict]) -> dict:
    flank = cow["flank_roi"]
    best = min(sweep, key=lambda s: abs(s["n_patches_big"] - flank["n_patches_big"]))
    return {
        "cow_flank_big_patches": flank["n_patches_big"],
        "cow_flank_coverage": flank["coverage_fraction"],
        "cow_largest_patch_share": flank["largest_patch_share_of_pigment"],
        "closest_sim_gamma": best["gamma"],
        "closest_sim_big_patches": best["n_patches_big"],
        "closest_sim_coverage": best["coverage_fraction"],
        "single_wavelength_turing_matches_flank": bool(
            best["n_patches_big"] <= 2 * max(flank["n_patches_big"], 1)
            and abs(best["coverage_fraction"] - flank["coverage_fraction"]) < 0.15),
        "interpretation": (
            "A single-wavelength Turing pre-pattern predicts many quasi-uniform "
            "patches; the Holstein flank carries essentially one body-scale patch. "
            "Matching the count by lowering gamma destroys the wavelength "
            "selection the mechanism exists to provide. Cow white spotting is "
            "therefore not supported as a simple Turing pattern by this test; "
            "it is consistent with the melanoblast-migration (MITF/KIT) account."
        ),
    }


def render(result: dict, field: np.ndarray, fit_mask: np.ndarray | None = None,
           fit_best: dict | None = None, eden_mask: np.ndarray | None = None) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image
    gray = np.asarray(Image.open(SPECIMEN).convert("L"), dtype=float)
    h, w = gray.shape
    roi = result["specimen"]["flank_roi"]["roi_px"]
    thr = result["specimen"]["flank_roi"]["otsu_threshold"]
    x1 = int(0.17 * w) + roi[0]
    y1 = int(0.22 * h) + roi[1]
    flank = gray[int(0.22 * h):y1, int(0.17 * w):x1]
    n_panels = (4 if fit_mask is not None else 3) + (1 if eden_mask is not None else 0)
    fig, axes = plt.subplots(1, n_panels, figsize=(4.4 * n_panels, 4.2))
    axes[0].imshow(flank, cmap="gray", vmin=0, vmax=255)
    axes[0].set_title("Cow flank ROI (specimen, c.1920)")
    axes[1].imshow(np.where(flank < thr, 0, 255).astype("uint8"), cmap="gray", vmin=0, vmax=255)
    axes[1].set_title(f"Otsu mask: {result['specimen']['flank_roi']['n_patches_big']} big dark patches,\n"
                      f"coverage {result['specimen']['flank_roi']['coverage_fraction']:.2f}")
    axes[2].imshow(field, cmap="viridis")
    axes[2].set_title(f"Pass 1 Turing (gamma=1): {result['simulation_sweep'][1]['n_patches_big']} spots,\n"
                      f"lambda_c={result['linear_turing_analysis']['lambda_c_analytic']:.1f} px")
    if fit_mask is not None and fit_best is not None:
        axes[3].imshow(np.where(fit_mask, 0, 255).astype("uint8"), cmap="gray",
                       vmin=0, vmax=255)  # pigment black, enclaves white
        axes[3].set_title(
            f"Remediated fit ({fit_best['mode']}, gamma={fit_best['gamma']:g}): "
            f"{fit_best['pigment']['n_patches_big']} patches,\n"
            f"coverage {fit_best['pigment']['coverage_fraction']:.2f}, "
            f"distance {fit_best['distance']:.2f}")
    if eden_mask is not None:
        comp = result["pass5_model_comparison"]["summary"]
        axes[-1].imshow(np.where(eden_mask, 0, 255).astype("uint8"), cmap="gray",
                        vmin=0, vmax=255)
        axes[-1].set_title(
            f"Pass 5 melanoblast (Eden) control:\n"
            f"distance v2 {comp['melanoblast_eden']['distance_v2_mean']:.2f} "
            f"vs RD {comp['rd_growing_het_best']['distance_v2_mean']:.2f} (5 seeds)")
    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    out = DATASET_DIR / "bos_taurus_turing_wavelength_comparison.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def self_test() -> None:
    ref = json.loads(REFERENCE.read_text(encoding="utf-8"))
    a = turing_analysis()
    assert a["tr_J"] < 0 and a["det_J"] > 0, a                      # stable at k=0
    assert a["turing_inequality_lhs"] > a["turing_inequality_rhs"], a
    lo, hi = a["unstable_band_k"]
    assert lo < a["k_star_numeric"] < hi, a                         # sigma max inside band
    assert lo < a["kc_analytic"] < hi, a
    assert a["sigma_max"] > 0, a                                    # unstable band exists
    assert a["sigma_max_equal_diffusion"] < 0, a                    # control: no Turing
    u = simulate(1.0, steps=3000)
    stats = patch_stats(u > u.mean())
    assert stats["n_patches_big"] > 20, stats                       # many spots, not one patch
    lam = dominant_wavelength(u)
    lam_star = 2.0 * np.pi / a["k_star_numeric"]
    assert 0.5 * lam_star < lam < 2.0 * lam_star, (lam, lam_star)
    cow = specimen_stats(ref)
    flank = cow["flank_roi"]
    assert flank["n_patches_big"] <= 5, flank
    assert flank["largest_patch_share_of_pigment"] > 0.9, flank
    # Pass-2/3/5 coverage (Fable audit: self-test previously covered none of it)
    g = simulate_growing(0.25, heterogeneity=0.8)
    assert g.shape == (NY, NX) and np.isfinite(g).all()
    gs = coat_pattern_stats(g, 0.75, pigment_low=True)
    assert gs["pigment"]["n_patches_big"] <= 10, gs["pigment"]
    eden = simulate_melanoblast(seed=SEED, target_coverage=0.75)
    assert abs(float(eden.mean()) - 0.75) < 0.02, float(eden.mean())
    es = coat_stats_from_mask(eden)
    assert es["enclaves"]["boundary_roughness_mean"] > gs["enclaves"]["boundary_roughness_mean"], (
        "Eden enclaves should be rougher than RD enclaves")
    print(f"[PASS] self-test: kc={a['kc_analytic']:.3f}, lambda_c={a['lambda_c_analytic']:.2f}px, "
          f"sigma_max={a['sigma_max']:.4f}, equal-diff sigma_max={a['sigma_max_equal_diffusion']:.2e}; "
          f"sim spots={stats['n_patches_big']}, FFT lambda={lam:.1f}px; "
          f"cow flank big patches={flank['n_patches_big']}, "
          f"largest share={flank['largest_patch_share_of_pigment']:.3f}")


def animate(gamma: float = 0.25, quantile: float = 0.75, heterogeneity: float = 0.8,
            frames_every: int = 180) -> Path:
    """GIF (Pass 2): cow flank mask (left) vs. growing-domain Turing coat (right).

    The pre-pattern forms on a small domain, which then grows in two stages
    with RD continuing after each growth step; pigment is the low-activator
    phase, rendered black, so the spots read as white enclaves on black --
    the cow's polarity.  Gamma/quantile default to the Pass-2 fit region;
    `main --animate` does not re-run the grid search, so check the metrics
    leaderboard if these defaults are changed.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    ref = json.loads(REFERENCE.read_text(encoding="utf-8"))
    cow = specimen_stats(ref)
    gray = np.asarray(Image.open(SPECIMEN).convert("L"), dtype=float)
    h, w = gray.shape
    roi = ref["specimen"]["rois"]["flank_roi"]
    x0, x1 = int(roi["x_frac"][0] * w), int(roi["x_frac"][1] * w)
    y0, y1 = int(roi["y_frac"][0] * h), int(roi["y_frac"][1] * h)
    flank = gray[y0:y1, x0:x1]
    mask = flank < cow["flank_roi"]["otsu_threshold"]
    mask_img = np.where(mask, 0, 255).astype("uint8")

    rng = np.random.default_rng(SEED)
    u0, v0 = steady_state()
    stages = ((60, 80, 2600), (90, 120, 1800), (120, 160, 1800))
    ny, nx, _ = stages[0]
    u = np.full((ny, nx), u0) + rng.normal(0.0, 0.05, (ny, nx))
    v = np.full((ny, nx), v0) + rng.normal(0.0, 0.05, (ny, nx))
    frames: list[Image.Image] = []

    # Pass 5: melanoblast (Eden) trajectory, computed first so its frames can
    # be shown beside the RD trajectory (progress-matched by frame index).
    eden_snaps: list[np.ndarray] = []
    simulate_melanoblast(seed=SEED, target_coverage=cow["flank_roi"]["coverage_fraction"],
                         snapshots=eden_snaps, snapshot_every=500)
    frame_idx = [0]

    def snap(u_now: np.ndarray, label: str) -> None:
        coat = np.where(u_now < np.quantile(u_now, quantile), 0, 255).astype("uint8")
        coat_big = np.asarray(Image.fromarray(coat).resize((640, 480)), dtype="uint8")
        frac = frame_idx[0] / 34.0  # approx. total RD frames, for progress matching
        eden = eden_snaps[min(int(frac * (len(eden_snaps) - 1)), len(eden_snaps) - 1)]
        eden_big = np.asarray(
            Image.fromarray(np.where(eden, 0, 255).astype("uint8")).resize((640, 480)),
            dtype="uint8")
        frame_idx[0] += 1
        fig, axes = plt.subplots(1, 3, figsize=(14, 3.6))
        axes[0].imshow(mask_img, cmap="gray", vmin=0, vmax=255)
        axes[0].set_title(f"Cow flank mask: {cow['flank_roi']['n_patches_big']} big patches, "
                          f"coverage {cow['flank_roi']['coverage_fraction']:.2f}")
        axes[1].imshow(coat_big, cmap="gray", vmin=0, vmax=255)
        axes[1].set_title(f"RD growing Turing coat (gamma={gamma:g}, het={heterogeneity:g})\n{label}")
        axes[2].imshow(eden_big, cmap="gray", vmin=0, vmax=255)
        axes[2].set_title("Melanoblast colonisation (Eden)\nwhite = never colonised")
        for ax in axes:
            ax.set_xticks([]); ax.set_yticks([])
        fig.tight_layout()
        fig.canvas.draw()
        frames.append(Image.frombuffer(
            "RGBA", fig.canvas.get_width_height(),
            fig.canvas.buffer_rgba()).convert("RGB"))
        plt.close(fig)

    snap(u, "embryonic domain 80x60, t=0")
    for si, (ny, nx, steps) in enumerate(stages):
        if u.shape != (ny, nx):
            u, v = _resize(u, (ny, nx)), _resize(v, (ny, nx))
            snap(u, f"grown to {nx}x{ny} -- pattern stretches with tissue")
        g = _gamma_map(gamma, (ny, nx), heterogeneity, rng)
        for step in range(steps):
            u, v = _evolve(u, v, g, 1)
            if (step + 1) % frames_every == 0:
                snap(u, f"stage {si + 1}/3 on {nx}x{ny}, step {step + 1}/{steps}")
    snap(u, "adult size 160x120 -- few large patches, white enclaves")
    out = DATASET_DIR / "bos_taurus_turing_wavelength_animation.gif"
    frames[0].save(out, save_all=True, append_images=frames[1:], duration=120, loop=0)
    return out


def render_specimens() -> Path:
    """Montage of the 9 held-out specimens: flank ROI (top) + cleaned mask."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image
    ref = json.loads(REFERENCE.read_text(encoding="utf-8"))
    fig, axes = plt.subplots(2, 9, figsize=(22, 5))
    for j, e in enumerate(ref.get("extra_specimens", [])):
        rgb = np.asarray(Image.open(DATASET_DIR / e["file"]).convert("RGB"))
        pigment, coat, flank = coat_segment(rgb, e["roi"])
        axes[0, j].imshow(flank, cmap="gray", vmin=0, vmax=255)
        axes[0, j].set_title(e["file"][-6:-4], fontsize=8)
        disp = np.where(pigment, 0, np.where(coat, 255, 128)).astype("uint8")
        axes[1, j].imshow(disp, cmap="gray", vmin=0, vmax=255)
    for ax in axes.ravel():
        ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    out = DATASET_DIR / "bos_taurus_specimens_montage.png"
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out


def animate_full() -> Path:
    """Full-story GIF: all 10 specimens + RD coat + Eden colonisation + scores.

    Top row cycles through the 10 specimens (flank ROI + cleaned mask,
    labelled with coverage); bottom row shows the growing-domain RD coat
    and the melanoblast (Eden) colonisation evolving; the final frames show
    the held-out distance-v2 comparison from the metrics JSON (if present).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    ref = json.loads(REFERENCE.read_text(encoding="utf-8"))
    # --- specimens (cow 01 + 9 held out) ---
    specimens: list[dict] = []
    rgb1 = np.asarray(Image.open(SPECIMEN).convert("RGB"))
    dark1, coat1, flank1 = coat_segment(rgb1, ref["specimen"]["rois"]["flank_roi"])
    specimens.append({"label": "01 (fit specimen, c.1920)", "flank": flank1,
                      "mask": dark1, "coat": coat1})
    for e in ref.get("extra_specimens", []):
        rgb = np.asarray(Image.open(DATASET_DIR / e["file"]).convert("RGB"))
        dark, coat, flank = coat_segment(rgb, e["roi"])
        specimens.append({"label": f"{e['file'][-6:-4]} held-out ({e['license']})",
                          "flank": flank, "mask": dark, "coat": coat})
    for sp in specimens:
        cov = patch_stats(sp["mask"])["coverage_fraction"]
        sp["title"] = f"{sp['label']}  coverage {cov:.2f}"

    # --- model trajectories ---
    metrics_path = DATASET_DIR / "bos_taurus_turing_wavelength_metrics.json"
    eden_snaps: list[np.ndarray] = []
    simulate_melanoblast(seed=SEED, target_coverage=0.65, snapshots=eden_snaps,
                         snapshot_every=500)
    rng = np.random.default_rng(SEED)
    u0, v0 = steady_state()
    stages = ((60, 80, 2600), (90, 120, 1800), (120, 160, 1800))
    ny, nx, _ = stages[0]
    u = np.full((ny, nx), u0) + rng.normal(0.0, 0.05, (ny, nx))
    v = np.full((ny, nx), v0) + rng.normal(0.0, 0.05, (ny, nx))

    frames: list[Image.Image] = []
    frame_no = [0]

    def snap(label: str) -> None:
        i = frame_no[0]
        sp = specimens[(i // 4) % len(specimens)]
        coat = np.where(u < np.quantile(u, 0.75), 0, 255).astype("uint8")
        eden = eden_snaps[min(int(i / 34 * (len(eden_snaps) - 1)), len(eden_snaps) - 1)]
        fig, axes = plt.subplots(2, 2, figsize=(11, 6.4))
        axes[0, 0].imshow(sp["flank"], cmap="gray", vmin=0, vmax=255)
        axes[0, 0].set_title(f"Specimen {sp['title']}", fontsize=9)
        axes[0, 1].imshow(np.where(sp["mask"], 0,
                                   np.where(sp["coat"], 255, 128)).astype("uint8"),
                          cmap="gray", vmin=0, vmax=255)
        axes[0, 1].set_title("Coat mask (black = pigment, grey = not coat)", fontsize=9)
        axes[1, 0].imshow(np.asarray(Image.fromarray(coat).resize((640, 480))),
                          cmap="gray", vmin=0, vmax=255)
        axes[1, 0].set_title(f"RD growing Turing coat\n{label}", fontsize=9)
        axes[1, 1].imshow(np.asarray(
            Image.fromarray(np.where(eden, 0, 255).astype("uint8")).resize((640, 480))),
            cmap="gray", vmin=0, vmax=255)
        axes[1, 1].set_title("Melanoblast colonisation (Eden)\nwhite = never colonised", fontsize=9)
        for ax in axes.ravel():
            ax.set_xticks([]); ax.set_yticks([])
        fig.suptitle("Bos taurus coat spotting: 10 specimens, two mechanisms", fontsize=11)
        fig.tight_layout()
        fig.canvas.draw()
        frames.append(Image.frombuffer("RGBA", fig.canvas.get_width_height(),
                                       fig.canvas.buffer_rgba()).convert("RGB"))
        plt.close(fig)
        frame_no[0] += 1

    snap("embryonic domain 80x60, t=0")
    for si, (ny, nx, steps) in enumerate(stages):
        if u.shape != (ny, nx):
            u, v = _resize(u, (ny, nx)), _resize(v, (ny, nx))
            snap(f"grown to {nx}x{ny}")
        g = _gamma_map(0.25, (ny, nx), 0.8, rng)
        for step in range(steps):
            u, v = _evolve(u, v, g, 1)
            if (step + 1) % 180 == 0:
                snap(f"stage {si + 1}/3 on {nx}x{ny}, step {step + 1}/{steps}")
    snap("adult size -- few large patches, white enclaves")

    # --- final score chart frames (from committed metrics, if available) ---
    if metrics_path.exists():
        m = json.loads(metrics_path.read_text(encoding="utf-8"))
        if "pass6_multi_specimen" in m:
            held = m["pass6_multi_specimen"]["held_out_distance_v2_mean"]
            per = m["pass6_multi_specimen"]["per_specimen"]
            names = ["rd_static_gamma1", "rd_growing_het_best", "melanoblast_eden"]
            labels = ["RD static", "RD growing-het", "Melanoblast Eden"]
            fig, ax = plt.subplots(figsize=(11, 6.4))
            vals = [held[n] for n in names]
            ax.bar(labels, vals, color=["#b0b0b0", "#4c78a8", "#222222"])
            for j, n in enumerate(names):
                pts = [p["distance_v2_mean"][n] for p in per]
                ax.plot([j] * len(pts), pts, "o", color="red", ms=4, alpha=0.7)
            ax.set_ylabel("distance v2 on held-out cows (lower = closer)")
            pr = m["pass6_multi_specimen"].get("paired_eden_vs_rd", {})
            if pr:
                ax.set_title("Held-out validation, 9 unseen Holsteins (red dots = per-cow), coat-mask + scale-matched\n"
                             f"Eden wins {pr.get('eden_wins', '?')}/9, mean diff {pr.get('mean_diff', float('nan')):+.2f}, "
                             f"Wilcoxon p={pr.get('wilcoxon_p') if pr.get('wilcoxon_p') is not None else float('nan'):.2f} -- models indistinguishable at matched scale")
            else:
                ax.set_title("Held-out validation, 9 unseen Holsteins (red dots = per-cow)")
            fig.tight_layout()
            fig.canvas.draw()
            chart = Image.frombuffer("RGBA", fig.canvas.get_width_height(),
                                     fig.canvas.buffer_rgba()).convert("RGB")
            plt.close(fig)
            frames.extend([chart] * 10)  # hold ~1.2 s
    out = DATASET_DIR / "bos_taurus_turing_wavelength_full_animation.gif"
    frames[0].save(out, save_all=True, append_images=frames[1:], duration=120, loop=0)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--no-show", action="store_true", help="write outputs, do not display")
    ap.add_argument("--animate", action="store_true", help="also write the animation GIF")
    ap.add_argument("--animate-full", action="store_true",
                    help="write only the full-story GIF (10 specimens + models + scores)")
    args = ap.parse_args()
    if args.self_test:
        self_test()
        return 0
    if args.animate_full:
        print(f"full animation: {animate_full()}")
        return 0
    bundle = run()
    result = bundle["result"]
    (DATASET_DIR / "bos_taurus_turing_wavelength_metrics.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8")
    out = render(result, bundle["field"], bundle["fit_mask"], bundle["fit_best"],
                 bundle["eden_mask"])
    print(f"specimens: {render_specimens()}")
    print(json.dumps(result["verdict"], indent=2))
    print(f"figure: {out}")
    if args.animate:
        print(f"animation: {animate()}")
    print(f"metrics: {DATASET_DIR / 'bos_taurus_turing_wavelength_metrics.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

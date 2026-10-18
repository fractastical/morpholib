#!/usr/bin/env python3
"""
Wolfram's Rule 30 cellular automaton (1984, 1986) and Meinhardt & Klingler's reaction-diffusion
model (1987, 1995) of Conus textile shell pigmentation, mapped onto a 3D Raup helicospiral shell
and compared quantitatively against real Conus textile specimens and the Wolfram Rule 30 Prizes.

Primary references:
  1. S. Wolfram, "Cellular automata as models of complexity,"
     Nature 311(5985):419-424 (1984). doi:10.1038/311419a0
  2. S. Wolfram, "Random sequence generation by cellular automata,"
     Adv. Appl. Math. 7(2):123-169 (1986). doi:10.1016/0196-8858(86)90029-1
  3. E. Jen, "Aperiodicity in one-dimensional cellular automata,"
     Physica D 45(1-3):3-18 (1990). doi:10.1016/0167-2789(90)90169-P
  4. G. B. Ermentrout, J. Campbell, G. F. Oster, "A model for shell patterns based on neural
     activity," The Veliger 28(4):369-388 (1986).
  5. H. Meinhardt, M. Klingler, "A model for pattern formation on the shells of molluscs,"
     J. Theor. Biol. 126(1):63-89 (1987). doi:10.1016/S0022-5193(87)80101-7
  6. H. Meinhardt, The Algorithmic Beauty of Sea Shells, Springer (1995).
     doi:10.1007/978-3-662-03125-4
  7. D. M. Raup, "Geometric analysis of shell coiling: general problems,"
     J. Paleontol. 40(5):1178-1190 (1966). doi:10.2307/1301992
  8. Wolfram Rule 30 Prizes: https://rule30prize.org/ (OEIS A051023)

Models implemented:
  1. Elementary Cellular Automaton Rule 30 (and additive Rule 90 control):
       a_{t+1,i} = a_{t,i-1} XOR (a_{t,i} OR a_{t,i+1})
                 = (a_{t,i-1} + a_{t,i} + a_{t,i+1} + a_{t,i} * a_{t,i+1}) mod 2
     Includes the exact left-bijective spatial inverse (Jen 1990):
       a_{t,i-1} = a_{t+1,i} XOR (a_{t,i} OR a_{t,i+1}),
     tracking of the left-periodic / right-chaotic phase boundary across the center column c_t = a_{t,0},
     and verification of the geometric white-triangle ("tent") base-width law P(W = w) = 2^{-w}.
  2. Meinhardt-Klingler (1987, Eq. 5 / 1995, Ch. 5-6) continuous 1D mantle-edge PDE:
       da/dt = rho * s * (a^2 / (1 + kappa * a^2) + rho_0) / b - mu_a * a + D_a * d^2a/dx^2
       ds/dt = sigma_s - rho * s * (a^2 / (1 + kappa * a^2) + rho_0) - mu_s * s + D_s * d^2s/dx^2
       db/dt = mu_b * (a - b) + D_b * d^2b/dx^2
     where traveling activator waves branch when depleted substrate s recovers and annihilate
     at the vertices of downward-pointing white triangles when opposing fronts collide.
  3. 3D Raup (1966) biconical helicospiral Conus shell surface onto which the 2D mantle-edge
     space-time kymographs (t, x) are mapped and compared against segmented Conus textile
     specimens (datasets/conus/conus_textile_reference.json).

Usage:
  python3 1984-wolfram-conus-textile-rule30.py [--no-gif] [--self-test-only] [--download]
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pathlib
import sys
import urllib.request

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
from PIL import Image
from scipy import ndimage, stats

HERE = pathlib.Path(__file__).resolve().parent
DATA_DIR = HERE / "datasets" / "conus"
REF_JSON = DATA_DIR / "conus_textile_reference.json"
CACHE_DIR = HERE / ".conus_cache"

EXPECTED_REF_SHA256 = "dfcb500d87a393907a5e862cb5167f1d79591f24da0f22b446ea10fc98a2afa5"

# Authentic Conus textile color palette: creamy calcified white (0) -> golden-ochre -> chestnut brown (1)
CONUS_CMAP = LinearSegmentedColormap.from_list(
    "conus_textile",
    [
        (0.00, "#fdfaf2"),  # creamy white interior of triangles ("tents")
        (0.35, "#f7ebd2"),  # warm ivory
        (0.62, "#d88b28"),  # golden amber margin
        (0.85, "#7c3214"),  # dark chestnut-brown line
        (1.00, "#3b1306"),  # deep umber apex intersection
    ],
)


# --------------------------------------------------------------------------- #
# 0. Fixture & Specimen Loading (SHA-256 verified)
# --------------------------------------------------------------------------- #
def verify_sha256(path: pathlib.Path, expected_hex: str) -> str:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != expected_hex:
        raise ValueError(f"SHA-256 mismatch for {path}: got {digest}, expected {expected_hex}")
    return digest


def load_reference_fixture() -> dict:
    if not REF_JSON.exists():
        raise FileNotFoundError(f"Missing reference fixture: {REF_JSON}")
    verify_sha256(REF_JSON, EXPECTED_REF_SHA256)
    return json.loads(REF_JSON.read_text())


def download_specimens(fixture: dict) -> dict[str, pathlib.Path]:
    """Download original Wikimedia Commons Conus textile photographs to .conus_cache/ and verify SHA-256."""
    CACHE_DIR.mkdir(exist_ok=True)
    filenames = {
        "stjohn_osu_24399204286": "conus_textile_stjohn_ccby2.jpg",
        "ling_commons_010": "conus_textile_010_pd.jpg",
    }
    paths = {}
    for spec in fixture["specimens"]:
        sid = spec["id"]
        target = CACHE_DIR / filenames[sid]
        if not target.exists():
            req = urllib.request.Request(
                spec["url"],
                headers={"User-Agent": "morpholib/0.1 (https://github.com/fractastical/morpholib)"},
            )
            with urllib.request.urlopen(req, timeout=25) as resp:
                target.write_bytes(resp.read())
        verify_sha256(target, spec["sha256"])
        paths[sid] = target
    return paths


def unpack_specimen_masks(fixture: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (silhouette_256x140, body_whorl_white_tents_148x92, stjohn_chord_runs_px)."""
    sil = np.zeros((256, 140), dtype=bool)
    for r, (c0, c1) in enumerate(fixture["silhouette_256x140_row_bounds"]):
        if c0 >= 0 and c1 >= c0:
            sil[r, c0 : c1 + 1] = True
    bitstrings = fixture["body_whorl_tents_148x92_bitstrings"]
    tents = np.array([[ch == "1" for ch in row] for row in bitstrings], dtype=bool)
    runs = np.array(fixture["stjohn_white_chord_runs_px"], dtype=float)
    return sil, tents, runs


# --------------------------------------------------------------------------- #
# 1. Wolfram Elementary Cellular Automata & Rule 30 Prize Diagnostics
# --------------------------------------------------------------------------- #
def rule30_step_boolean(row: np.ndarray) -> np.ndarray:
    """Exact Boolean formulation: a_{t+1,i} = a_{t,i-1} XOR (a_{t,i} OR a_{t,i+1})."""
    left = np.roll(row, 1)
    center = row
    right = np.roll(row, -1)
    return (left ^ (center | right)).astype(np.uint8)


def rule30_step_gf2_anf(row: np.ndarray) -> np.ndarray:
    """Algebraic Normal Form (ANF) over GF(2): a_{t+1,i} = (a_{t,i-1} + a_{t,i} + a_{t,i+1} + a_{t,i}*a_{t,i+1}) mod 2."""
    left = np.roll(row, 1).astype(np.uint8)
    center = row.astype(np.uint8)
    right = np.roll(row, -1).astype(np.uint8)
    return ((left + center + right + center * right) & 1).astype(np.uint8)


def run_eca(rule_num: int, width: int, steps: int, init: str = "single", seed: int = 42) -> np.ndarray:
    """Evolve any 1D elementary cellular automaton (0..255) on a periodic ring of given width."""
    rng = np.random.default_rng(seed)
    grid = np.zeros((steps, width), dtype=np.uint8)
    if init == "single":
        grid[0, width // 2] = 1
    elif init == "random":
        grid[0] = rng.integers(0, 2, size=width, dtype=np.uint8)
    else:
        raise ValueError(f"Unknown init mode: {init}")
    for t in range(steps - 1):
        left = np.roll(grid[t], 1)
        center = grid[t]
        right = np.roll(grid[t], -1)
        neighborhood = (left << 2) | (center << 1) | right
        grid[t + 1] = (rule_num >> neighborhood) & 1
    return grid


def reconstruct_left_column(col_i: np.ndarray, col_ip1: np.ndarray) -> np.ndarray:
    """Erica Jen's (1990) left-bijective spatial inverse for Rule 30:
        a_{t,i-1} = a_{t+1,i} XOR (a_{t,i} OR a_{t,i+1})  for t = 0 .. T-2.
    Because Rule 30 is linear in its left input a_{t,i-1}, any two adjacent time-columns
    uniquely determine all columns to their left, which is why no two adjacent columns
    in the light-cone can both be periodic without forcing every column to their left to be periodic.
    """
    return (col_i[1:] ^ (col_i[:-1] | col_ip1[:-1])).astype(np.uint8)


def rule30_prize_diagnostics(steps: int = 1024) -> dict:
    """Compute quantitative diagnostics for the three Wolfram Rule 30 Prize problems (rule30prize.org)."""
    width = 2 * steps + 5
    mid = width // 2
    cone = run_eca(30, width=width, steps=steps, init="single")
    center_col = cone[:, mid]

    # 1. Problem 1 (Non-periodicity): test whether center_col[t0:] is periodic for any period p <= steps // 4
    max_p = steps // 4
    periodic_found = False
    for p in range(1, max_p + 1):
        # Check if the tail of length 3*p is strictly periodic with period p
        tail = center_col[-3 * p :]
        if np.array_equal(tail[:-p], tail[p:]):
            periodic_found = True
            break

    # 2. Left-bijective spatial reconstruction check (Jen 1990)
    col_0 = cone[:, mid]
    col_1 = cone[:, mid + 1]
    recon_m1 = reconstruct_left_column(col_0, col_1)
    true_m1 = cone[:-1, mid - 1]
    bijective_errors = int(np.sum(recon_m1 != true_m1))

    # 3. Track left-regular vs right-chaotic phase boundary:
    # Along left diagonals d_k(t) = a_{t, mid - t + k} (k >= 0), d_0(t) = 1 for all t >= 0,
    # d_1(t) = 1 for t >= 1, and low-k diagonals lock into period dividing 16 (2^m).
    front_offsets = []
    for t in range(64, steps):
        k_front = 0
        for k in range(0, 2 * t):
            ok = True
            for tau in range(t - 15, t + 1):
                c1 = mid - tau + k
                c0 = c1 + 16  # mid - (tau - 16) + k
                if c0 >= width or cone[tau, c1] != cone[tau - 16, c0]:
                    ok = False
                    break
            if ok:
                k_front = k
            else:
                break
        front_offsets.append((t, k_front - t))
    ts_f = np.array([x[0] for x in front_offsets], dtype=float)
    pos_f = np.array([x[1] for x in front_offsets], dtype=float)
    slope_front, intercept_front = np.polyfit(ts_f, pos_f, 1)

    # 4. Problem 2 (Equidistribution): cumulative density of 1s and k-block Shannon entropy (k = 4)
    cum_density = np.cumsum(center_col) / np.arange(1, steps + 1)
    k_block = 4
    weights = 1 << np.arange(k_block - 1, -1, -1)
    blocks = np.convolve(center_col, weights, mode="valid")
    counts = np.bincount(blocks, minlength=1 << k_block)
    probs = counts[counts > 0] / len(blocks)
    block_entropy_per_bit = float(-np.sum(probs * np.log2(probs)) / k_block)

    # 5. Problem 3 (Sensitivity / Boolean avalanche width on periodic ring)
    ring_w = 256
    ring_steps = 120
    base_ring = run_eca(30, width=ring_w, steps=ring_steps, init="random", seed=17)
    pert_init = base_ring[0].copy()
    pert_init[ring_w // 2] ^= 1
    pert_ring = np.zeros_like(base_ring)
    pert_ring[0] = pert_init
    for t in range(ring_steps - 1):
        pert_ring[t + 1] = rule30_step_boolean(pert_ring[t])
    diff_cone = base_ring ^ pert_ring
    avalanche_widths = np.array([np.ptp(np.where(diff_cone[t])[0]) + 1 if np.any(diff_cone[t]) else 0 for t in range(ring_steps)])
    lyapunov_speed = float(np.polyfit(np.arange(10, 80), avalanche_widths[10:80], 1)[0])

    return {
        "cone": cone,
        "center_col": center_col,
        "cum_density": cum_density,
        "final_density": float(cum_density[-1]),
        "density_z_score": float(abs(cum_density[-1] - 0.5) / (0.5 / np.sqrt(steps))),
        "block4_entropy_bits": block_entropy_per_bit,
        "periodic_found": periodic_found,
        "bijective_errors": bijective_errors,
        "front_ts": ts_f,
        "front_pos": pos_f,
        "front_velocity": float(slope_front),
        "front_intercept": float(intercept_front),
        "lyapunov_damage_spreading_speed": lyapunov_speed,
    }


# --------------------------------------------------------------------------- #
# 2. Meinhardt-Klingler (1987 / 1995) Continuous Reaction-Diffusion PDE
# --------------------------------------------------------------------------- #
def run_meinhardt_conus(nx: int = 230, nt_out: int = 370, substeps: int = 12, seed: int = 42) -> np.ndarray:
    """Meinhardt & Klingler (1987, J. Theor. Biol. 126:63-89, Eq. 5; Meinhardt 1995, Ch. 5-6)
    1D mantle-edge activator (a) - depleted substrate (s) - local quenching inhibitor (b) PDE.
    """
    rng = np.random.default_rng(seed)
    a = np.full(nx, 0.10, dtype=float)
    s = np.full(nx, 1.00, dtype=float)
    b = np.full(nx, 0.10, dtype=float)
    init_sites = rng.choice(nx, size=14, replace=False)
    a[init_sites] = 1.50

    dt = 0.15
    Da, Ds, Db = 0.22, 0.08, 0.01
    rho, rho0, kappa = 0.08, 0.002, 0.25
    mu_a, sigma_s, mu_s, mu_b = 0.075, 0.032, 0.012, 0.018

    kymograph = np.zeros((nt_out, nx), dtype=float)
    for step in range(nt_out):
        for _ in range(substeps):
            lap_a = np.roll(a, 1) + np.roll(a, -1) - 2.0 * a
            lap_s = np.roll(s, 1) + np.roll(s, -1) - 2.0 * s
            lap_b = np.roll(b, 1) + np.roll(b, -1) - 2.0 * b
            autocat = rho * s * (a * a / (1.0 + kappa * a * a) + rho0)
            da = autocat / np.maximum(b, 0.08) - mu_a * a + Da * lap_a
            ds = sigma_s - autocat - mu_s * s + Ds * lap_s
            db = mu_b * (a - b) + Db * lap_b
            # Wave branching when unpigmented clearing accumulates high substrate s
            if rng.random() < 0.18:
                high_s = np.where((s > 1.45) & (a < 0.15))[0]
                if len(high_s) > 0:
                    a[rng.choice(high_s)] += 0.90
            a = np.clip(a + dt * da, 0.0, 5.0)
            s = np.clip(s + dt * ds, 0.01, 3.0)
            b = np.clip(b + dt * db, 0.05, 5.0)
        kymograph[step] = a
    return kymograph


# --------------------------------------------------------------------------- #
# 3. 3D Raup (1966) Biconical Conus textile Shell Geometry & Morphometrics
# --------------------------------------------------------------------------- #
def raup_conus_mesh(
    W: float = 1.28,
    D: float = 0.42,
    T: float = 0.36,
    S: float = 5.99,
    turns: float = 5.5,
    n_theta: int = 360,
    n_phi: int = 160,
    azim_deg: float = 195.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate 3D (X, Y, Z) surface mesh of a Conus textile shell using Raup's (1966)
    logarithmic helicospiral parameters (W, D, T, S) with a biconical Conus aperture profile.
    """
    azim = np.deg2rad(azim_deg)
    theta = np.linspace(0.0, 2.0 * np.pi * turns, n_theta) + azim
    phi = np.linspace(0.0, 2.0 * np.pi, n_phi)
    TH, PH = np.meshgrid(theta, phi)
    scale = W ** ((TH - azim) / (2.0 * np.pi))
    a = scale
    rc = a * (1.0 + D) / (1.0 - D)
    zc = T * rc
    b = a * S

    s_pos = np.maximum(0.0, np.sin(PH))
    s_neg = np.maximum(0.0, -np.sin(PH))
    c_phi = np.cos(PH)
    envelope = np.where(
        np.sin(PH) >= 0.0,
        np.cos(0.468 * np.pi * (s_pos ** 1.28)),
        1.0 - 0.78 * (s_neg ** 1.05),
    )
    axial_ext = np.where(np.sin(PH) >= 0.0, 1.55 * s_pos, -0.42 * s_neg)
    R = np.maximum(0.02 * scale, (rc + a * c_phi) * envelope)
    X = R * np.cos(TH)
    Y = R * np.sin(TH)
    Z = -(zc + b * axial_ext)
    return X, Y, Z


def project_mesh_silhouette(X: np.ndarray, Z: np.ndarray, shape: tuple[int, int] = (256, 140)) -> tuple[np.ndarray, float]:
    """Project 3D shell mesh onto coronal (X, Z) plane and rasterize binary silhouette."""
    H_grid, W_grid = shape
    z_min, z_max = float(Z.min()), float(Z.max())
    x_min, x_max = float(X.min()), float(X.max())
    hw_ratio = (z_max - z_min) / (x_max - x_min)
    z_norm = (Z - z_min) / (z_max - z_min)
    x_norm = (X - x_min) / (x_max - x_min)
    sil = np.zeros(shape, dtype=bool)
    for k in range(H_grid):
        zk = 1.0 - (k + 0.5) / H_grid
        band = np.abs(z_norm - zk) < 0.018
        if np.any(band):
            xlo, xhi = float(x_norm[band].min()), float(x_norm[band].max())
            c0 = max(0, int(round(xlo * (W_grid - 1))))
            c1 = min(W_grid - 1, int(round(xhi * (W_grid - 1))))
            sil[k, c0 : c1 + 1] = True
    return sil, hw_ratio


def white_chord_runs(white_mask: np.ndarray) -> np.ndarray:
    """Extract interior horizontal contiguous runs of unpigmented (white tent) cells."""
    runs = []
    for row in white_mask:
        padded = np.concatenate(([False], row, [False]))
        starts = np.where(padded[1:] & ~padded[:-1])[0]
        ends = np.where(~padded[1:] & padded[:-1])[0]
        for s, e in zip(starts, ends):
            if s > 0 and e < len(row):
                runs.append(int(e - s))
    return np.array(runs, dtype=float)


def box_counting_dimension(white_mask: np.ndarray) -> float:
    """Compute 2D box-counting fractal dimension D_box of the tent boundary network."""
    edges = white_mask ^ ndimage.binary_erosion(white_mask)
    scales = np.array([2, 4, 8, 16, 32], dtype=int)
    H, W = edges.shape
    scales = scales[(scales <= H // 2) & (scales <= W // 2)]
    counts = []
    for s in scales:
        h = (H // s) * s
        w = (W // s) * s
        blocks = edges[:h, :w].reshape(h // s, s, w // s, s)
        counts.append(max(1, int(np.sum(blocks.any(axis=(1, 3))))))
    slope, _ = np.polyfit(np.log(1.0 / scales.astype(float)), np.log(np.array(counts, dtype=float)), 1)
    return float(slope)


def map_kymograph_to_shell_rgb(
    X: np.ndarray,
    Y: np.ndarray,
    Z: np.ndarray,
    kymograph: np.ndarray,
    light_dir: tuple[float, float, float] = (-0.45, 0.55, 0.70),
    nearest: bool = False,
) -> np.ndarray:
    """Map a 2D mantle-edge space-time kymograph (nt, nx) onto the 3D Raup shell mesh (n_phi, n_theta)
    with Lambertian surface normal shading.
    """
    n_phi, n_theta = X.shape
    # Resample kymograph (time = coiling angle theta, space = aperture position phi)
    k_norm = (kymograph - kymograph.min()) / max(1e-9, float(kymograph.max() - kymograph.min()))
    k_img = Image.fromarray((k_norm * 255).astype(np.uint8))
    interp = Image.NEAREST if nearest else Image.BILINEAR
    # Transpose so rows = phi (mantle margin), cols = theta (growth time)
    tex = np.asarray(k_img.resize((n_phi, n_theta), interp), dtype=float).T / 255.0
    rgb = CONUS_CMAP(tex)[..., :3]

    # Surface normals for Lambertian + specular gloss shading
    dX_dth, dX_dph = np.gradient(X)
    dY_dth, dY_dph = np.gradient(Y)
    dZ_dth, dZ_dph = np.gradient(Z)
    nx = dY_dth * dZ_dph - dZ_dth * dY_dph
    ny = dZ_dth * dX_dph - dX_dth * dZ_dph
    nz = dX_dth * dY_dph - dY_dth * dX_dph
    norm = np.maximum(1e-9, np.sqrt(nx * nx + ny * ny + nz * nz))
    nx /= norm
    ny /= norm
    nz /= norm

    lx, ly, lz = light_dir
    lnorm = np.sqrt(lx * lx + ly * ly + lz * lz)
    cos_inc = np.abs(nx * (lx / lnorm) + ny * (ly / lnorm) + nz * (lz / lnorm))
    shade = 0.42 + 0.58 * cos_inc + 0.14 * (cos_inc ** 6)
    shaded_rgb = np.clip(rgb * shade[..., None], 0.0, 1.0)
    return shaded_rgb


# --------------------------------------------------------------------------- #
# 4. Self-Tests (Algebraic, Geometric, and Empirical Verification)
# --------------------------------------------------------------------------- #
def self_test() -> dict:
    fixture = load_reference_fixture()
    spec_sil, spec_tents, spec_runs = unpack_specimen_masks(fixture)
    prize = rule30_prize_diagnostics(steps=1024)

    # Check 1: Boolean XOR/OR vs GF(2) ANF polynomial equivalence across all 8 neighborhoods
    test_row = np.array([0, 0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 0], dtype=np.uint8)
    assert np.array_equal(rule30_step_boolean(test_row), rule30_step_gf2_anf(test_row)), "Boolean vs GF(2) ANF mismatch"

    # Check 2: Exact match of Rule 30 center column against first 128 bits of OEIS A051023
    oeis_128 = np.array(fixture["rule30_prize_benchmarks"]["oeis_a051023_first_128"], dtype=np.uint8)
    assert np.array_equal(prize["center_col"][:128], oeis_128), "Center column disagrees with OEIS A051023"

    # Check 3: Erica Jen's (1990) left-bijective spatial inverse has 0 bit errors
    assert prize["bijective_errors"] == 0, f"Expected 0 left-bijective inverse errors, got {prize['bijective_errors']}"

    # Check 4: Rule 30 Prize Problem 1 & 2 finite-horizon verification
    assert not prize["periodic_found"], "Unexpected periodicity found in Rule 30 center column"
    assert 0.46 <= prize["final_density"] <= 0.54, f"Center column density out of bounds: {prize['final_density']:.4f}"
    assert prize["block4_entropy_bits"] > 0.98, f"4-block entropy too low: {prize['block4_entropy_bits']:.4f}"
    assert -0.40 <= prize["front_velocity"] <= -0.15, f"Unexpected phase-boundary velocity: {prize['front_velocity']:.4f}"

    # Check 5: Rule 30 white-triangle base-width geometric law P(W = w) = 2^{-w} (mean = 2.0, CV = 1/sqrt(2) = 0.7071)
    r30_grid = run_eca(30, width=230, steps=370, init="random", seed=42)
    r30_runs = white_chord_runs(r30_grid == 0)
    r30_mean = float(r30_runs.mean())
    r30_cv = float(r30_runs.std() / r30_mean)
    assert abs(r30_mean - 2.0) < 0.08, f"Rule 30 mean white run {r30_mean:.3f} != 2.0"
    assert abs(r30_cv - (1.0 / np.sqrt(2.0))) < 0.04, f"Rule 30 white run CV {r30_cv:.4f} != 0.7071"

    # Check 6: Meinhardt-Klingler (1987) PDE generates self-extinguishing white triangles
    mein_kymo = run_meinhardt_conus(nx=230, nt_out=370, seed=42)
    mein_pigment = mein_kymo > np.percentile(mein_kymo, 55)
    mein_runs = white_chord_runs(~mein_pigment)
    mein_cv = float(mein_runs.std() / mein_runs.mean())
    assert len(mein_runs) > 800 and 0.60 <= mein_cv <= 1.35, f"Meinhardt tent runs invalid: n={len(mein_runs)}, CV={mein_cv:.3f}"

    # Check 7: 3D Raup Conus textile shell geometry vs segmented OSU specimen
    X, Y, Z = raup_conus_mesh()
    sim_sil, hw_mesh = project_mesh_silhouette(X, Z, shape=spec_sil.shape)
    iou = float(np.logical_and(spec_sil, sim_sil).sum() / np.logical_or(spec_sil, sim_sil).sum())
    spec_hw = float(fixture["specimens"][0]["height_to_width"])
    assert abs(hw_mesh - spec_hw) / spec_hw < 0.015, f"Mesh H/W {hw_mesh:.4f} deviates >1.5% from specimen {spec_hw:.4f}"
    assert iou > 0.82, f"Mesh silhouette IoU {iou:.4f} < 0.82"

    # Check 8: Normalized chord-length distribution comparison (Rule 30, Meinhardt, Rule 90 vs Specimen)
    r90_grid = run_eca(90, width=230, steps=370, init="random", seed=42)
    r90_runs = white_chord_runs(r90_grid == 0)

    ks_r30 = float(stats.ks_2samp(r30_runs / r30_runs.mean(), spec_runs / spec_runs.mean()).statistic)
    ks_mein = float(stats.ks_2samp(mein_runs / mein_runs.mean(), spec_runs / spec_runs.mean()).statistic)
    ks_r90 = float(stats.ks_2samp(r90_runs / r90_runs.mean(), spec_runs / spec_runs.mean()).statistic)

    d_box_spec = box_counting_dimension(spec_tents)
    d_box_r30 = box_counting_dimension(r30_grid[:148, :92] == 0)
    d_box_mein = box_counting_dimension(~mein_pigment[:148, :92])

    return {
        "fixture": fixture,
        "spec_sil": spec_sil,
        "spec_tents": spec_tents,
        "spec_runs": spec_runs,
        "prize": prize,
        "r30_grid": r30_grid,
        "r30_runs": r30_runs,
        "r30_mean": r30_mean,
        "r30_cv": r30_cv,
        "mein_kymo": mein_kymo,
        "mein_pigment": mein_pigment,
        "mein_runs": mein_runs,
        "mein_cv": mein_cv,
        "r90_runs": r90_runs,
        "X": X,
        "Y": Y,
        "Z": Z,
        "sim_sil": sim_sil,
        "hw_mesh": hw_mesh,
        "spec_hw": spec_hw,
        "silhouette_iou": iou,
        "ks_r30_vs_specimen": ks_r30,
        "ks_meinhardt_vs_specimen": ks_mein,
        "ks_r90_vs_specimen": ks_r90,
        "d_box_specimen": d_box_spec,
        "d_box_r30": d_box_r30,
        "d_box_meinhardt": d_box_mein,
    }


# --------------------------------------------------------------------------- #
# 5. CSV Metrics, 6-Panel Summary Figure & 1x2 Animated GIF
# --------------------------------------------------------------------------- #
def write_metrics_csv(res: dict, out_csv: pathlib.Path) -> None:
    spec_runs = res["spec_runs"]
    rows = [
        {
            "system": "Conus textile specimen (St. John OSU, CC BY 2.0)",
            "kind": "biological_specimen",
            "pigment_fraction": round(float(1.0 - res["spec_tents"].mean()), 4),
            "white_chord_mean": round(float(spec_runs.mean()), 4),
            "white_chord_cv": round(float(spec_runs.std() / spec_runs.mean()), 4),
            "ks_vs_specimen_norm_chords": 0.0,
            "box_counting_dim": round(res["d_box_specimen"], 4),
            "shell_hw_ratio": round(res["spec_hw"], 4),
            "silhouette_iou_vs_specimen": 1.0,
        },
        {
            "system": "Wolfram Rule 30 (a_{t+1,i} = l XOR (c OR r))",
            "kind": "discrete_cellular_automaton",
            "pigment_fraction": round(float(res["r30_grid"].mean()), 4),
            "white_chord_mean": round(res["r30_mean"], 4),
            "white_chord_cv": round(res["r30_cv"], 4),
            "ks_vs_specimen_norm_chords": round(res["ks_r30_vs_specimen"], 4),
            "box_counting_dim": round(res["d_box_r30"], 4),
            "shell_hw_ratio": round(res["hw_mesh"], 4),
            "silhouette_iou_vs_specimen": round(res["silhouette_iou"], 4),
        },
        {
            "system": "Meinhardt-Klingler (1987) activator-substrate-inhibitor PDE",
            "kind": "continuous_reaction_diffusion",
            "pigment_fraction": round(float(res["mein_pigment"].mean()), 4),
            "white_chord_mean": round(float(res["mein_runs"].mean()), 4),
            "white_chord_cv": round(res["mein_cv"], 4),
            "ks_vs_specimen_norm_chords": round(res["ks_meinhardt_vs_specimen"], 4),
            "box_counting_dim": round(res["d_box_meinhardt"], 4),
            "shell_hw_ratio": round(res["hw_mesh"], 4),
            "silhouette_iou_vs_specimen": round(res["silhouette_iou"], 4),
        },
        {
            "system": "Wolfram Rule 90 (additive XOR Sierpinski control)",
            "kind": "linear_ca_control",
            "pigment_fraction": round(float(1.0 - (res["r90_runs"].sum() / (230 * 370))), 4),
            "white_chord_mean": round(float(res["r90_runs"].mean()), 4),
            "white_chord_cv": round(float(res["r90_runs"].std() / res["r90_runs"].mean()), 4),
            "ks_vs_specimen_norm_chords": round(res["ks_r90_vs_specimen"], 4),
            "box_counting_dim": round(box_counting_dimension(run_eca(90, 92, 148, init="random", seed=42) == 0), 4),
            "shell_hw_ratio": round(res["hw_mesh"], 4),
            "silhouette_iou_vs_specimen": round(res["silhouette_iou"], 4),
        },
    ]
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def render_comparison_figure(res: dict, out_png: pathlib.Path) -> None:
    prize = res["prize"]
    fig = plt.figure(figsize=(16.5, 10.2), dpi=160, facecolor="#0e1117")
    gs = fig.add_gridspec(2, 3, hspace=0.30, wspace=0.24, left=0.05, right=0.97, top=0.90, bottom=0.06)

    fig.suptitle(
        "Conus textile Pigmentation & The Wolfram Rule 30 Prizes: Cellular Automata (1984) vs. Meinhardt PDE (1987) on a 3D Raup Shell",
        color="#f0f6fc",
        fontsize=13.5,
        fontweight="bold",
        y=0.96,
    )
    fig.text(
        0.5,
        0.928,
        f"OEIS A051023 verified (128/128 bits) · Left-bijective inverse (Jen 1990): 0 errors · Center density c̄_1024 = {prize['final_density']:.4f} · "
        f"3D Raup shell vs. OSU Conus textile specimen: H/W = {res['hw_mesh']:.3f} (ref {res['spec_hw']:.3f}), Silhouette IoU = {res['silhouette_iou']:.3f}",
        color="#8b949e",
        fontsize=9.5,
        ha="center",
    )

    # Panel A: Rule 30 single-seed light-cone + phase boundary + center column
    ax_a = fig.add_subplot(gs[0, 0], facecolor="#161b22")
    n_show = 180
    mid = prize["cone"].shape[1] // 2
    sub_cone = prize["cone"][:n_show, mid - n_show : mid + n_show + 1]
    ax_a.imshow(
        sub_cone,
        cmap=CONUS_CMAP,
        aspect="auto",
        extent=[-n_show, n_show, n_show, 0],
        interpolation="nearest",
    )
    ax_a.axvline(0, color="#58a6ff", lw=1.3, ls="--", label="Center column $c_t = a_{t,0}$ (Prize #1–3)")
    t_line = np.linspace(15, n_show, 100)
    ax_a.plot(
        prize["front_velocity"] * t_line + prize["front_intercept"],
        t_line,
        color="#3fb950",
        lw=1.8,
        label=f"Period-$2^k$ / chaos front ($v = {prize['front_velocity']:.3f}$)",
    )
    ax_a.set_xlim(-n_show, n_show)
    ax_a.set_ylim(n_show, 0)
    ax_a.set_title("A. Rule 30 Light-Cone & Phase Boundary", color="#f0f6fc", fontsize=11, fontweight="bold")
    ax_a.set_xlabel("Spatial cell index $i$ relative to seed", color="#c9d1d9", fontsize=9)
    ax_a.set_ylabel("Time step $t$ (shell growth)", color="#c9d1d9", fontsize=9)
    ax_a.tick_params(colors="#8b949e", labelsize=8)
    ax_a.legend(loc="lower left", fontsize=7.5, facecolor="#0d1117", edgecolor="#30363d", labelcolor="#f0f6fc")

    # Panel B: Rule 30 Prize Diagnostics (Equidistribution & Left-Bijective Reconstruction)
    ax_b = fig.add_subplot(gs[0, 1], facecolor="#161b22")
    t_steps = np.arange(1, len(prize["cum_density"]) + 1)
    env_95 = 1.96 * 0.5 / np.sqrt(t_steps)
    ax_b.fill_between(t_steps, 0.5 - env_95, 0.5 + env_95, color="#58a6ff", alpha=0.18, label="Binomial $95\\%$ envelope ($1/2 \\pm 1.96/\\sqrt{4t}$)")
    ax_b.axhline(0.5, color="#f0f6fc", lw=1.0, ls=":")
    ax_b.plot(t_steps, prize["cum_density"], color="#58a6ff", lw=1.6, label=f"Center-column ones density $\\bar{{c}}_t$ ($\\bar{{c}}_{{1024}}={prize['final_density']:.4f}$)")
    ax_b.set_xscale("log")
    ax_b.set_ylim(0.32, 0.72)
    ax_b.set_title("B. Rule 30 Prize #1–3 Diagnostics (OEIS A051023)", color="#f0f6fc", fontsize=11, fontweight="bold")
    ax_b.set_xlabel("Time horizon $T$ (log scale)", color="#c9d1d9", fontsize=9)
    ax_b.set_ylabel("Cumulative fraction of 1s $\\bar{c}_T$", color="#c9d1d9", fontsize=9)
    ax_b.tick_params(colors="#8b949e", labelsize=8)
    ax_b.legend(loc="upper right", fontsize=7.5, facecolor="#0d1117", edgecolor="#30363d", labelcolor="#f0f6fc")
    ax_b.text(
        0.04,
        0.06,
        f"• Problem 1 (aperiodicity): no period $p \\leq {len(t_steps)//4}$ found\n"
        f"• Problem 2 (equidistribution): $H_4/4 = {prize['block4_entropy_bits']:.4f}$ bits/bit\n"
        f"• Problem 3 (irreducibility): avalanche speed $\\lambda = {prize['lyapunov_damage_spreading_speed']:.2f}$ cells/step\n"
        f"• Jen (1990) left-inverse $a_{{t,i-1}} = a_{{t+1,i}} \\oplus (a_{{t,i}} \\vee a_{{t,i+1}})$: 0 errors",
        transform=ax_b.transAxes,
        color="#c9d1d9",
        fontsize=7.8,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#0d1117", edgecolor="#30363d", alpha=0.92),
    )

    # Panel C: Specimen Silhouette vs 3D Raup Mesh + Dorsal Tent Lattice
    ax_c = fig.add_subplot(gs[0, 2], facecolor="#161b22")
    spec_sil = res["spec_sil"]
    sim_sil = res["sim_sil"]
    overlay = np.zeros((*spec_sil.shape, 3), dtype=float)
    overlay[:] = [0.086, 0.106, 0.133]
    both = spec_sil & sim_sil
    only_spec = spec_sil & ~sim_sil
    only_sim = sim_sil & ~spec_sil
    overlay[both] = [0.95, 0.76, 0.42]       # warm gold overlap
    overlay[only_spec] = [0.25, 0.65, 0.95]  # blue specimen-only margin
    overlay[only_sim] = [0.92, 0.35, 0.30]   # red model-only margin
    # Composite specimen dorsal tent crop next to silhouette overlay
    tents_rgb = CONUS_CMAP(1.0 - res["spec_tents"].astype(float))[..., :3]
    tents_resized = np.asarray(Image.fromarray((tents_rgb * 255).astype(np.uint8)).resize((115, 256), Image.NEAREST), dtype=float) / 255.0
    sep = np.full((256, 6, 3), 0.15)
    comp_c = np.concatenate([overlay, sep, tents_resized], axis=1)
    ax_c.imshow(comp_c, aspect="auto")
    ax_c.set_title(
        f"C. Real Conus textile Specimen vs. Raup Mesh (IoU = {res['silhouette_iou']:.3f})",
        color="#f0f6fc",
        fontsize=11,
        fontweight="bold",
    )
    ax_c.set_xticks([70, 200])
    ax_c.set_xticklabels(["Shell Silhouette\n(Gold = Overlap)", "Segmented Dorsal\nWhite Tents (OSU)"], color="#c9d1d9", fontsize=8.5)
    ax_c.set_yticks([])

    # Panel D: Mantle-Edge Space-Time Kymographs (Rule 30 vs Meinhardt-Klingler PDE)
    ax_d = fig.add_subplot(gs[1, 0], facecolor="#161b22")
    # Upsample a 74 x 55 Rule 30 patch by 2x (nearest) so each secretory cell is 2x2 px alongside Meinhardt PDE
    r30_patch = np.kron(res["r30_grid"][:74, :55].astype(float), np.ones((2, 2)))
    mein_crop = res["mein_kymo"][:148, :110]
    mein_norm = (mein_crop - mein_crop.min()) / max(1e-9, float(mein_crop.max() - mein_crop.min()))
    divider = np.ones((148, 4), dtype=float) * 0.5
    dual_kymo = np.concatenate([r30_patch, divider, mein_norm], axis=1)
    ax_d.imshow(dual_kymo, cmap=CONUS_CMAP, aspect="auto", interpolation="nearest")
    ax_d.axvline(111.5, color="#58a6ff", lw=1.5)
    ax_d.set_title("D. 1D Mantle Space-Time Kymographs: Rule 30 vs. Meinhardt PDE", color="#f0f6fc", fontsize=11, fontweight="bold")
    ax_d.set_xticks([55, 169])
    ax_d.set_xticklabels(["Wolfram Rule 30 (1984)\nDiscrete CA ($a_{t+1,i}$)", "Meinhardt–Klingler (1987)\nActivator–Substrate PDE"], color="#c9d1d9", fontsize=8.5)
    ax_d.set_ylabel("Mantle secretion time $t \\downarrow$", color="#c9d1d9", fontsize=9)
    ax_d.tick_params(colors="#8b949e", labelsize=8)

    # Panel E: 3D Pigmented Raup Conus Shells (Rule 30 & Meinhardt mapped onto 3D surface)
    ax_e = fig.add_subplot(gs[1, 1], projection="3d", facecolor="#161b22")
    X, Y, Z = res["X"], res["Y"], res["Z"]
    # Map coarse secretory-zone Rule 30 (90 steps x 54 mantle zones) and Meinhardt PDE onto 3D Conus surface
    rgb_r30 = map_kymograph_to_shell_rgb(X, Y, Z, res["r30_grid"][:90, :54].astype(float), nearest=True)
    rgb_mein = map_kymograph_to_shell_rgb(X, Y, Z, res["mein_kymo"][:180, :110])
    span_x = float(X.max() - X.min())
    ax_e.plot_surface(
        X - 0.62 * span_x,
        Y,
        Z,
        facecolors=rgb_r30,
        rstride=2,
        cstride=2,
        antialiased=True,
        shade=False,
    )
    ax_e.plot_surface(
        X + 0.62 * span_x,
        Y,
        Z,
        facecolors=rgb_mein,
        rstride=2,
        cstride=2,
        antialiased=True,
        shade=False,
    )
    ax_e.view_init(elev=14, azim=-72)
    ax_e.set_axis_off()
    ax_e.set_title("E. 3D Raup Conus Shells: Rule 30 (left) & Meinhardt (right)", color="#f0f6fc", fontsize=11, fontweight="bold")

    # Panel F: White-Triangle Chord Distribution (CCDF P(W >= w/<w>)) & Fractal Box Dimension
    ax_f = fig.add_subplot(gs[1, 2], facecolor="#161b22")
    for label, runs, color, ls, lw in [
        (f"Conus textile specimen (CV={res['spec_runs'].std()/res['spec_runs'].mean():.2f}, $D_{{box}}={res['d_box_specimen']:.2f}$)", res["spec_runs"], "#f0f6fc", "-", 2.3),
        (f"Wolfram Rule 30 (CV={res['r30_cv']:.2f}, $D_{{box}}={res['d_box_r30']:.2f}$, KS={res['ks_r30_vs_specimen']:.2f})", res["r30_runs"], "#58a6ff", "--", 1.9),
        (f"Meinhardt PDE (CV={res['mein_cv']:.2f}, $D_{{box}}={res['d_box_meinhardt']:.2f}$, KS={res['ks_meinhardt_vs_specimen']:.2f})", res["mein_runs"], "#f0883e", "-.", 1.9),
        (f"Rule 90 Sierpiński control (KS={res['ks_r90_vs_specimen']:.2f})", res["r90_runs"], "#f85149", ":", 1.5),
    ]:
        norm_r = np.sort(runs / runs.mean())
        ccdf = 1.0 - np.arange(len(norm_r)) / len(norm_r)
        ax_f.step(norm_r, ccdf, where="post", color=color, ls=ls, lw=lw, label=label)

    ax_f.set_yscale("log")
    ax_f.set_xlim(0.0, 5.5)
    ax_f.set_ylim(1e-3, 1.05)
    ax_f.set_title("F. White-Triangle Base-Width Tail $P(W \\geq w / \\langle w \\rangle)$", color="#f0f6fc", fontsize=11, fontweight="bold")
    ax_f.set_xlabel("Normalized white-tent chord width $w / \\langle w \\rangle$", color="#c9d1d9", fontsize=9)
    ax_f.set_ylabel("Complementary CDF $P(W \\geq w / \\langle w \\rangle)$", color="#c9d1d9", fontsize=9)
    ax_f.tick_params(colors="#8b949e", labelsize=8)
    ax_f.legend(loc="upper right", fontsize=7.2, facecolor="#0d1117", edgecolor="#30363d", labelcolor="#f0f6fc")

    fig.savefig(out_png, dpi=160)
    plt.close(fig)


def make_gif(res: dict, out_gif: pathlib.Path, frames: int = 36, fps: int = 10) -> None:
    """Render a clean 1x2 side-by-side animation (640x340):
      - Left panel: 1D mantle-edge space-time kymograph progressively depositing rows of triangles
        (first 18 frames: Wolfram Rule 30; next 18 frames: Meinhardt-Klingler PDE).
      - Right panel: 3D Raup Conus textile shell rotating 360 deg with the mapped pigmentation.
    """
    X, Y, Z = res["X"], res["Y"], res["Z"]
    Xs, Ys, Zs = X[::2, ::2], Y[::2, ::2], Z[::2, ::2]
    rgb_r30 = map_kymograph_to_shell_rgb(Xs, Ys, Zs, res["r30_grid"][:90, :54].astype(float), nearest=True)
    rgb_mein = map_kymograph_to_shell_rgb(Xs, Ys, Zs, res["mein_kymo"][:180, :110])

    # Use 2x2 cellular blocks for Rule 30 so every white triangle tent is crisp at thumbnail scale
    r30_kymo = np.kron(res["r30_grid"][:80, :60].astype(float), np.ones((2, 2)))
    mein_raw = res["mein_kymo"][:160, :120]
    mein_kymo = (mein_raw - mein_raw.min()) / max(1e-9, float(mein_raw.max() - mein_raw.min()))

    fig = plt.figure(figsize=(6.4, 3.4), dpi=100, facecolor="#0e1117")
    gs = fig.add_gridspec(1, 2, width_ratios=[1.05, 1.0], wspace=0.12, left=0.08, right=0.98, top=0.83, bottom=0.13)
    ax_kymo = fig.add_subplot(gs[0, 0], facecolor="#161b22")
    ax_3d = fig.add_subplot(gs[0, 1], projection="3d", facecolor="#161b22")

    half = frames // 2

    def update(frame: int):
        ax_kymo.clear()
        ax_3d.clear()
        is_mein = frame >= half
        local_f = frame if not is_mein else (frame - half)
        # Start at 96 rows on frame 0 so the thumbnail immediately shows rich triangular tents
        prog_rows = int(np.clip(96 + (64 * local_f) // max(1, half - 1), 96, 160))

        kymo_full = mein_kymo if is_mein else r30_kymo
        canvas = np.full_like(kymo_full, 0.12)
        canvas[:prog_rows] = kymo_full[:prog_rows]
        ax_kymo.imshow(canvas, cmap=CONUS_CMAP, aspect="auto", vmin=0.0, vmax=1.0, interpolation="nearest")
        ax_kymo.axhline(prog_rows - 1, color="#58a6ff", lw=1.8)
        mode_title = (
            "1/2 Wolfram Rule 30 CA ($l \\oplus (c \\vee r)$)"
            if not is_mein
            else "2/2 Meinhardt–Klingler (1987) PDE"
        )
        ax_kymo.set_title(mode_title, color="#f0f6fc", fontsize=9.2, fontweight="bold")
        ax_kymo.set_xlabel("Mantle edge position $x$", color="#c9d1d9", fontsize=8)
        ax_kymo.set_ylabel("Growth time $t \\downarrow$", color="#c9d1d9", fontsize=8)
        ax_kymo.tick_params(colors="#8b949e", labelsize=7)

        # Rotate 3D Conus shell
        az = -80.0 + (360.0 * frame) / frames
        rad = np.deg2rad(az)
        Xr = Xs * np.cos(rad) - Ys * np.sin(rad)
        Yr = Xs * np.sin(rad) + Ys * np.cos(rad)
        face_rgb = rgb_mein if is_mein else rgb_r30
        ax_3d.plot_surface(Xr, Yr, Zs, facecolors=face_rgb, rstride=1, cstride=1, antialiased=True, shade=False)
        ax_3d.view_init(elev=15, azim=-70)
        ax_3d.set_axis_off()
        ax_3d.set_title(
            f"3D Raup Conus textile (IoU = {res['silhouette_iou']:.2f})",
            color="#f0f6fc",
            fontsize=9.2,
            fontweight="bold",
        )
        fig.suptitle(
            "Conus textile Shell Pigmentation: 1D Mantle-Edge Space-Time → 3D Coiled Shell",
            color="#f0f6fc",
            fontsize=10.0,
            fontweight="bold",
            y=0.95,
        )
        return []

    anim = FuncAnimation(fig, update, frames=frames, interval=1000 // fps, blit=False)
    anim.save(out_gif, writer=PillowWriter(fps=fps))
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--no-gif", action="store_true", help="Skip GIF animation generation")
    ap.add_argument("--self-test-only", action="store_true", help="Run self-tests and exit")
    ap.add_argument("--download", action="store_true", help="Download & SHA-256 verify Wikimedia Commons Conus textile photos")
    ap.add_argument("--frames", type=int, default=36, help="Number of animation frames")
    ap.add_argument("--fps", type=int, default=10, help="Animation frames per second")
    args = ap.parse_args(argv)

    res = self_test()
    if args.download:
        paths = download_specimens(res["fixture"])
        print(f"[download] Verified {len(paths)} Wikimedia Commons Conus textile specimen files in {CACHE_DIR}")

    prize = res["prize"]
    print(
        f"[self_test] OK | OEIS A051023 128/128 bits | Jen (1990) left-inverse errors={prize['bijective_errors']} | "
        f"center density c_1024={prize['final_density']:.4f} (H4/4={prize['block4_entropy_bits']:.4f} bits) | "
        f"front velocity={prize['front_velocity']:.3f} cells/step"
    )
    print(
        f"[morphometrics] Conus textile H/W={res['hw_mesh']:.4f} (specimen {res['spec_hw']:.4f}), "
        f"Silhouette IoU={res['silhouette_iou']:.4f} | "
        f"White-chord CV: specimen={res['spec_runs'].std()/res['spec_runs'].mean():.3f}, "
        f"Rule30={res['r30_cv']:.3f} (theory 0.707), Meinhardt={res['mein_cv']:.3f}"
    )
    if args.self_test_only:
        return 0

    out_csv = HERE / "conus_textile_rule30_metrics.csv"
    out_png = HERE / "conus_textile_rule30_comparison.png"
    write_metrics_csv(res, out_csv)
    render_comparison_figure(res, out_png)
    print(f"Wrote {out_csv.name} and {out_png.name}")

    if not args.no_gif:
        out_gif = HERE / "conus_textile_rule30.gif"
        media_gif = HERE / "docs" / "media" / "conus_textile_rule30.gif"
        make_gif(res, out_gif, frames=args.frames, fps=args.fps)
        if media_gif.parent.exists():
            media_gif.write_bytes(out_gif.read_bytes())
        print(f"Wrote {out_gif.name} and {media_gif} ({out_gif.stat().st_size // 1024} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

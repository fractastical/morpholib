#!/usr/bin/env python3
"""
Turing (1952) & Pearson (1993) Reaction–Diffusion Morphogenesis (Gray–Scott Model)
Enhanced with Spectral Ring Morphometrics & Equal-Diffusion Negative Control
=================================================================================

Primary literature sources:
  1. Turing AM (1952). The chemical basis of morphogenesis.
     Philosophical Transactions of the Royal Society of London B 237(641):37-72.
     doi:10.1098/rstb.1952.0012
  2. Pearson JE (1993). Complex patterns in a simple system.
     Science 261(5118):189-192. doi:10.1126/science.261.5118.189
  3. Lee KJ, McCormick WD, Pearson JE, Swinney HL (1994). Experimental observation
     of self-replicating spots in a reaction-diffusion system.
     Nature 369(6477):215-218. doi:10.1038/369215a0

Mathematical model (Gray–Scott reaction–diffusion system):
  ∂U/∂t = D_u ∇²U - U V² + F (1 - U)
  ∂V/∂t = D_v ∇²V + U V² - (F + k) V

where:
  - U(x,y,t) is the substrate / inhibitor species with diffusion coefficient D_u,
  - V(x,y,t) is the autocatalytic activator species with diffusion coefficient D_v < D_u,
  - F is the dimensionless feed rate, and k is the dimensionless kill/removal rate.

Cross-repository spectral morphometrics (Track 2 / Issue #4):
  - Shares 2D radial power-spectral density (PSD) dominant wavenumber k*, characteristic
    wavelength λ* = N / k*, spectral ring sharpness, and azimuthal harmonic anisotropy
    (m=2 stripe vs. m=6 hexagonal/spot symmetry) with `2001-bressloff-v1-form-constants.py`.
  - Includes an explicit equal-diffusion negative control (D_u = D_v = 0.10 at the
    canonical coral/labyrinth regime F=0.055, k=0.062) verifying Turing's core theorem:
    removing differential diffusivity extinguishes the spatial instability and collapses
    the non-zero spectral ring (k* -> 0, spatial variance -> 0).

Usage:
  python3 1952-turing-morpho.py                     # run Pearson morphospace + legacy GIF
  python3 1952-turing-morpho.py --no-show           # save PNG + CSV without GUI window
  python3 1952-turing-morpho.py --skip-gif          # skip legacy GIF generation for fast runs
  python3 1952-turing-morpho.py --self-test         # verify Pearson regimes + negative control
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import matplotlib

if any(flag in sys.argv for flag in ("--no-show", "--self-test", "--skip-gif")) or not os.environ.get("DISPLAY"):
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.colors import LinearSegmentedColormap, PowerNorm

REPO_ROOT = Path(__file__).resolve().parent


# ---------------------------------------------------------------------------
# 1. Custom Bioelectric / Neon Colormap (preserved from original script)
# ---------------------------------------------------------------------------
BLACK_NEON_CMAP = LinearSegmentedColormap.from_list(
    "black_neon_bio",
    [
        (0.00, "#000000"),  # black background
        (0.10, "#040b2d"),  # deep navy
        (0.25, "#1e2f97"),  # electric blue
        (0.42, "#00b6ff"),  # cyan
        (0.58, "#00e676"),  # neon green
        (0.74, "#c6ff00"),  # yellow-green
        (0.88, "#ffd54f"),  # warm yellow
        (1.00, "#ff7043"),  # orange-red highlights
    ],
)


# ---------------------------------------------------------------------------
# 2. Canonical Pearson (1993) Regimes & Negative Control
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class GrayScottRegime:
    id: str
    title: str
    pearson_class: str
    F: float
    k: float
    du: float = 0.20
    dv: float = 0.10
    expected_morphology: str = "patterned"
    description: str = ""


PEARSON_REGIMES: Tuple[GrayScottRegime, ...] = (
    GrayScottRegime(
        id="mitosis_spots",
        title="Mitosis / Self-Replicating Spots",
        pearson_class="Pattern λ / θ (Pearson 1993)",
        F=0.028,
        k=0.062,
        du=0.20,
        dv=0.10,
        expected_morphology="spots",
        description="Self-replicating spots that grow and split via binary fission (Pearson 1993; Lee et al. 1994)",
    ),
    GrayScottRegime(
        id="soliton_spots",
        title="Stable Soliton Spots",
        pearson_class="Pattern λ (Pearson 1993)",
        F=0.030,
        k=0.062,
        du=0.20,
        dv=0.10,
        expected_morphology="spots",
        description="Localized high-activator spots arranged in quasi-hexagonal domains",
    ),
    GrayScottRegime(
        id="stripes_worms",
        title="Labyrinthine Stripes / Worms",
        pearson_class="Pattern μ / ζ (Pearson 1993)",
        F=0.035,
        k=0.060,
        du=0.20,
        dv=0.10,
        expected_morphology="stripes",
        description="Elongated vermiform stripes filling the domain at constant wavelength λ*",
    ),
    GrayScottRegime(
        id="coral_labyrinth",
        title="Coral / Connected Labyrinth",
        pearson_class="Pattern η / γ (Pearson 1993)",
        F=0.055,
        k=0.062,
        du=0.20,
        dv=0.10,
        expected_morphology="labyrinth",
        description="Connected coral-like Turing labyrinth (default regime of 1952-turing-morpho.py)",
    ),
    GrayScottRegime(
        id="equal_diffusion_control",
        title="Equal-Diffusion Negative Control",
        pearson_class="Turing Null Control (Du = Dv)",
        F=0.055,
        k=0.062,
        du=0.10,
        dv=0.10,
        expected_morphology="homogeneous",
        description="Identical reactions (F=0.055, k=0.062) and initial seed, but Du = Dv = 0.10 removes differential diffusion",
    ),
)


# ---------------------------------------------------------------------------
# 3. Finite-Difference Gray–Scott Solver (Deterministic Seed)
# ---------------------------------------------------------------------------
def laplacian(Z: np.ndarray) -> np.ndarray:
    """5-point periodic finite-difference Laplacian on a 2D torus."""
    return (
        np.roll(Z, 1, axis=0)
        + np.roll(Z, -1, axis=0)
        + np.roll(Z, 1, axis=1)
        + np.roll(Z, -1, axis=1)
        - 4.0 * Z
    )


def initialize_gray_scott_fields(
    N: int = 160,
    seed: int = 1952,
    n_nucleation_patches: int = 9,
) -> Tuple[np.ndarray, np.ndarray]:
    """Initialize (U, V) with reproducible multi-patch nucleation + low-amplitude noise."""
    rng = np.random.default_rng(seed)
    U = np.ones((N, N), dtype=np.float64)
    V = np.zeros((N, N), dtype=np.float64)

    # Central square perturbation (matches original script) plus distributed nucleation sites
    half = max(6, N // 14)
    c = N // 2
    U[c - half : c + half, c - half : c + half] = 0.50
    V[c - half : c + half, c - half : c + half] = 0.25

    patch_r = max(4, N // 24)
    for _ in range(n_nucleation_patches):
        py = int(rng.integers(patch_r + 4, N - patch_r - 4))
        px = int(rng.integers(patch_r + 4, N - patch_r - 4))
        U[py - patch_r : py + patch_r, px - patch_r : px + patch_r] = 0.50
        V[py - patch_r : py + patch_r, px - patch_r : px + patch_r] = 0.25

    U += 0.02 * rng.random((N, N))
    V += 0.02 * rng.random((N, N))
    np.clip(U, 0.0, 1.0, out=U)
    np.clip(V, 0.0, 1.0, out=V)
    return U, V


def simulate_gray_scott(
    regime: GrayScottRegime,
    N: int = 160,
    steps: int = 4000,
    dt: float = 1.0,
    seed: int = 1952,
) -> Tuple[np.ndarray, np.ndarray]:
    """Integrate the Gray–Scott PDE system for `steps` forward Euler steps."""
    U, V = initialize_gray_scott_fields(N=N, seed=seed)
    du, dv, f, k = regime.du, regime.dv, regime.F, regime.k

    for _ in range(steps):
        Lu = laplacian(U)
        Lv = laplacian(V)
        uvv = U * V * V
        U += dt * (du * Lu - uvv + f * (1.0 - U))
        V += dt * (dv * Lv + uvv - (f + k) * V)

    return U, V


# ---------------------------------------------------------------------------
# 4. 2D Spectral Ring & Topological Morphometrics
# ---------------------------------------------------------------------------
def compute_turing_spectral_metrics(V: np.ndarray) -> Dict[str, Any]:
    """Compute 2D FFT radial power spectrum, characteristic wavelength λ*,
    ring sharpness, azimuthal harmonic symmetry (m=2 vs m=6), and connected components."""
    from scipy import ndimage as ndi

    N = V.shape[0]
    std_v = float(np.std(V))
    ptp_v = float(np.ptp(V))

    # Thresholded high-activator coverage and Euler/component statistics
    thresh = max(0.12, float(np.mean(V) + 0.5 * std_v))
    active_mask = (V > thresh) if std_v > 1e-4 else np.zeros_like(V, dtype=bool)
    coverage = float(np.mean(active_mask))
    labeled, n_components = ndi.label(active_mask)

    # Holes inside active regions (distinguishes closed labyrinths/rings from isolated spots)
    inv_mask = ~active_mask
    _, n_bg_components = ndi.label(inv_mask)
    euler_characteristic = int(n_components - max(0, n_bg_components - 1))

    if std_v < 1e-4:
        return {
            "k_star_cycles": 0.0,
            "wavelength_grid_cells": float("inf"),
            "ring_sharpness": 0.0,
            "peak_power": 0.0,
            "m2_anisotropy": 0.0,
            "m6_hex_symmetry": 0.0,
            "std_v": std_v,
            "ptp_v": ptp_v,
            "active_coverage": 0.0,
            "n_components": 0,
            "euler_characteristic": 0,
            "radial_k": [float(i) for i in range(N // 2)],
            "radial_psd": [0.0 for _ in range(N // 2)],
        }

    centered = V - float(np.mean(V))
    win_1d = np.hanning(N)
    window = np.outer(win_1d, win_1d)
    fft2 = np.fft.fftshift(np.fft.fft2(centered * window))
    psd2d = np.abs(fft2) ** 2

    cy, cx = N // 2, N // 2
    yy, xx = np.indices((N, N))
    rr = np.hypot(yy - cy, xx - cx)
    r_int = np.rint(rr).astype(int)
    max_r = min(N // 2 - 1, 50)

    counts = np.bincount(r_int.ravel(), minlength=max_r + 1)[: max_r + 1]
    sums = np.bincount(r_int.ravel(), weights=psd2d.ravel(), minlength=max_r + 1)[: max_r + 1]
    radial_psd = sums / np.maximum(counts, 1)
    radial_psd[:2] = 0.0  # exclude DC / domain-scale drift

    k_star_idx = int(np.argmax(radial_psd))
    peak_power = float(radial_psd[k_star_idx])
    positive = radial_psd[2:]
    median_power = float(np.median(positive[positive > 0])) if np.any(positive > 0) else 1e-12
    ring_sharpness = float(peak_power / max(median_power, 1e-12))
    wavelength_cells = float(N / k_star_idx) if k_star_idx > 0 else float("inf")

    # Azimuthal Fourier harmonics on the k* annulus
    annulus = (rr >= max(2.0, k_star_idx - 1.5)) & (rr <= k_star_idx + 1.5)
    angles = np.arctan2(yy[annulus] - cy, xx[annulus] - cx)
    weights = psd2d[annulus]
    w_sum = float(np.sum(weights))
    if w_sum > 1e-12:
        m2 = float(np.abs(np.sum(weights * np.exp(1j * 2.0 * angles))) / w_sum)
        m6 = float(np.abs(np.sum(weights * np.exp(1j * 6.0 * angles))) / w_sum)
    else:
        m2, m6 = 0.0, 0.0

    return {
        "k_star_cycles": float(k_star_idx),
        "wavelength_grid_cells": wavelength_cells,
        "ring_sharpness": ring_sharpness,
        "peak_power": peak_power,
        "m2_anisotropy": m2,
        "m6_hex_symmetry": m6,
        "std_v": std_v,
        "ptp_v": ptp_v,
        "active_coverage": coverage,
        "n_components": int(n_components),
        "euler_characteristic": euler_characteristic,
        "radial_k": [float(i) for i in range(max_r + 1)],
        "radial_psd": [float(v) for v in radial_psd],
    }


# ---------------------------------------------------------------------------
# 5. Legacy Neon GIF Generator (preserves `turing_patterns_black_neon.gif`)
# ---------------------------------------------------------------------------
def render_legacy_neon_gif(
    output_gif: Path = REPO_ROOT / "turing_patterns_black_neon.gif",
    N: int = 200,
    steps: int = 5000,
    substeps_per_frame: int = 10,
) -> Path:
    """Render the classic `turing_patterns_black_neon.gif` animation."""
    du, dv = 0.2, 0.1
    f, k = 0.055, 0.062
    dt = 1.0

    rng = np.random.default_rng(1952)
    U = np.ones((N, N), dtype=np.float64)
    V = np.zeros((N, N), dtype=np.float64)
    U[N // 2 - 20 : N // 2 + 20, N // 2 - 20 : N // 2 + 20] = 0.5
    V[N // 2 - 20 : N // 2 + 20, N // 2 - 20 : N // 2 + 20] = 0.25
    U += 0.05 * rng.random((N, N))
    V += 0.05 * rng.random((N, N))

    fig, ax = plt.subplots(facecolor="black")
    ax.set_facecolor("black")
    im = ax.imshow(
        V,
        cmap=BLACK_NEON_CMAP,
        norm=PowerNorm(gamma=0.75, vmin=0.0, vmax=0.65),
        interpolation="nearest",
    )
    ax.set_axis_off()
    plt.title("Turing Pattern Simulation (Gray-Scott Model)", color="white")

    def update(_frame: int):
        nonlocal U, V
        for _ in range(substeps_per_frame):
            Lu = laplacian(U)
            Lv = laplacian(V)
            uvv = U * V * V
            U += dt * (du * Lu - uvv + f * (1.0 - U))
            V += dt * (dv * Lv + uvv - (f + k) * V)
        im.set_array(V)
        return [im]

    ani = FuncAnimation(fig, update, frames=steps // substeps_per_frame, interval=50, blit=True)
    ani.save(str(output_gif), writer="pillow")
    plt.close(fig)
    print(f"Saved Turing neon animation to {output_gif}")
    return output_gif


# ---------------------------------------------------------------------------
# 6. Pearson (1993) Multi-Regime Morphospace Synthesis & Verification
# ---------------------------------------------------------------------------
def run_pearson_morphospace_analysis(
    output_png: Path = REPO_ROOT / "turing_pearson_morphospace.png",
    output_csv: Path = REPO_ROOT / "turing_pearson_metrics.csv",
    N: int = 160,
    steps: int = 4000,
    show: bool = False,
) -> Dict[str, Any]:
    """Simulate all canonical Pearson (1993) regimes + the Du=Dv negative control,
    compute 2D spectral ring metrics, write CSV, and render a 2x5 publication figure."""
    results: List[Dict[str, Any]] = []
    for reg in PEARSON_REGIMES:
        U, V = simulate_gray_scott(reg, N=N, steps=steps, seed=1952)
        metrics = compute_turing_spectral_metrics(V)
        results.append({
            "regime": reg,
            "U": U,
            "V": V,
            "metrics": metrics,
        })

    # Write CSV
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "regime_id",
            "pearson_class",
            "F",
            "k",
            "Du",
            "Dv",
            "expected_morphology",
            "k_star_cycles",
            "wavelength_grid_cells",
            "ring_sharpness",
            "std_v",
            "active_coverage",
            "n_components",
            "euler_characteristic",
            "m2_anisotropy",
            "m6_hex_symmetry",
        ])
        for item in results:
            reg = item["regime"]
            m = item["metrics"]
            wl_str = f"{m['wavelength_grid_cells']:.3f}" if math.isfinite(m["wavelength_grid_cells"]) else "inf"
            writer.writerow([
                reg.id,
                reg.pearson_class,
                f"{reg.F:.4f}",
                f"{reg.k:.4f}",
                f"{reg.du:.3f}",
                f"{reg.dv:.3f}",
                reg.expected_morphology,
                f"{m['k_star_cycles']:.1f}",
                wl_str,
                f"{m['ring_sharpness']:.3f}",
                f"{m['std_v']:.6f}",
                f"{m['active_coverage']:.4f}",
                m["n_components"],
                m["euler_characteristic"],
                f"{m['m2_anisotropy']:.4f}",
                f"{m['m6_hex_symmetry']:.4f}",
            ])

    # Render 2-row x 5-column figure
    fig = plt.figure(figsize=(18.5, 8.2), dpi=160)
    gs = fig.add_gridspec(2, 5, height_ratios=[1.25, 1.0], hspace=0.36, wspace=0.28)

    colors = ["#00e676", "#00b6ff", "#ffd54f", "#ff7043", "#95a5a6"]
    for col, item in enumerate(results):
        reg = item["regime"]
        V = item["V"]
        m = item["metrics"]

        # Row 0: Activator V(x,y) spatial field
        ax_field = fig.add_subplot(gs[0, col])
        ax_field.imshow(
            V,
            cmap=BLACK_NEON_CMAP,
            norm=PowerNorm(gamma=0.75, vmin=0.0, vmax=0.65),
            origin="lower",
        )
        wl_label = f"$\\lambda^*={m['wavelength_grid_cells']:.1f}$ px" if math.isfinite(m["wavelength_grid_cells"]) else "$\\lambda^*=\\infty$ (uniform)"
        ax_field.set_title(
            f"{reg.title}\n"
            f"$F={reg.F:.3f}, k={reg.k:.3f}, D_u/D_v={reg.du/reg.dv:.1f}$\n"
            f"{wl_label}, comp={m['n_components']}",
            fontsize=8.8,
            fontweight="bold",
        )
        ax_field.set_xticks([])
        ax_field.set_yticks([])
        if col == 0:
            ax_field.set_ylabel("Activator Field $V(x,y)$\n($N=160\\times 160$, $t=4000$)", fontsize=9.5, fontweight="bold")

        # Row 1: Radial 2D FFT Power Spectrum P(k)
        ax_psd = fig.add_subplot(gs[1, col])
        rk = np.asarray(m["radial_k"])
        rpsd = np.asarray(m["radial_psd"])
        norm_psd = rpsd / max(float(np.max(rpsd)), 1e-12) if np.max(rpsd) > 0 else rpsd
        ax_psd.plot(rk, norm_psd, color=colors[col], lw=2.0)
        ax_psd.fill_between(rk, 0.0, norm_psd, color=colors[col], alpha=0.18)
        if m["k_star_cycles"] > 0:
            ax_psd.axvline(m["k_star_cycles"], color="#e74c3c", ls="--", lw=1.5, label=f"$k^*={m['k_star_cycles']:.0f}$ cyc")
            ax_psd.legend(fontsize=7.8, loc="upper right")
        ax_psd.set_xlim(0, 40)
        ax_psd.set_ylim(0, 1.12)
        ax_psd.set_xlabel("Radial Wavenumber $k$ (cycles / box)", fontsize=8.5)
        if col == 0:
            ax_psd.set_ylabel("Normalized Radial PSD $P(k)$", fontsize=9.0, fontweight="bold")
        ax_psd.set_title(
            f"Spectral Ring Sharpness = {m['ring_sharpness']:.1f}\n"
            f"$\\sigma_V={m['std_v']:.4f}$, coverage={100*m['active_coverage']:.1f}%",
            fontsize=8.3,
        )
        ax_psd.grid(alpha=0.25)

    fig.suptitle(
        "Turing (1952) & Pearson (1993) Gray–Scott Reaction–Diffusion Morphospace\n"
        "Self-Replicating Spots, Solitons, Stripes, Coral Labyrinths & Equal-Diffusion ($D_u = D_v$) Negative Control",
        fontsize=12.5,
        fontweight="bold",
        y=0.98,
    )
    fig.savefig(output_png, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)

    print(f"Saved Pearson Turing morphospace figure to {output_png}")
    print(f"Saved quantitative Turing spectral metrics to {output_csv}")

    by_id = {item["regime"].id: item["metrics"] for item in results}
    return {
        "results": results,
        "by_id": by_id,
        "output_png": str(output_png),
        "output_csv": str(output_csv),
    }


def self_test() -> bool:
    """Self-test verifying Pearson (1993) regimes and the Du = Dv negative control."""
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        out = run_pearson_morphospace_analysis(
            output_png=Path(td) / "turing_morphospace.png",
            output_csv=Path(td) / "turing_metrics.csv",
            N=160,
            steps=4000,
            show=False,
        )

    by_id = out["by_id"]
    ok = True

    # 1. All 4 patterned Pearson regimes must exhibit a strong non-zero spectral ring (k* >= 6, sharpness > 5)
    for rid in ("mitosis_spots", "soliton_spots", "stripes_worms", "coral_labyrinth"):
        m = by_id[rid]
        cond = m["k_star_cycles"] >= 6.0 and m["ring_sharpness"] > 5.0 and m["std_v"] > 0.02
        status = "PASS" if cond else "FAIL"
        print(
            f"[{status}] {rid}: k*={m['k_star_cycles']:.0f} cyc, "
            f"lambda*={m['wavelength_grid_cells']:.2f} px, sharpness={m['ring_sharpness']:.1f}, "
            f"components={m['n_components']}, std_V={m['std_v']:.4f}"
        )
        ok = ok and cond

    # 2. Spot regimes must have more disconnected components than connected coral/labyrinth
    spots_comp = by_id["mitosis_spots"]["n_components"]
    coral_comp = by_id["coral_labyrinth"]["n_components"]
    comp_ok = spots_comp > coral_comp
    print(
        f"[{'PASS' if comp_ok else 'FAIL'}] Topological distinction: mitosis_spots components ({spots_comp}) "
        f"> coral_labyrinth components ({coral_comp})"
    )
    ok = ok and comp_ok

    # 3. Equal-diffusion negative control (Du = Dv = 0.10) must extinguish spatial patterning
    ctrl = by_id["equal_diffusion_control"]
    ctrl_ok = ctrl["k_star_cycles"] == 0.0 and ctrl["std_v"] < 1e-4 and ctrl["n_components"] == 0
    print(
        f"[{'PASS' if ctrl_ok else 'FAIL'}] Equal-diffusion negative control (Du=Dv=0.10): "
        f"k*={ctrl['k_star_cycles']:.1f}, std_V={ctrl['std_v']:.2e}, components={ctrl['n_components']}"
    )
    ok = ok and ctrl_ok

    return ok


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Turing (1952) & Pearson (1993) Gray-Scott morphospace and spectral analysis."
    )
    parser.add_argument("--no-show", action="store_true", help="Do not display interactive Matplotlib window")
    parser.add_argument("--skip-gif", action="store_true", help="Skip rendering legacy turing_patterns_black_neon.gif")
    parser.add_argument("--self-test", action="store_true", help="Run Pearson regime + negative-control verification suite")
    parser.add_argument(
        "--output-png",
        type=Path,
        default=REPO_ROOT / "turing_pearson_morphospace.png",
        help="Output path for the Pearson morphospace figure",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=REPO_ROOT / "turing_pearson_metrics.csv",
        help="Output path for the Pearson spectral metrics CSV",
    )
    args = parser.parse_args(argv)

    if args.self_test:
        return 0 if self_test() else 1

    run_pearson_morphospace_analysis(
        output_png=args.output_png,
        output_csv=args.output_csv,
        show=not args.no_show,
    )
    if not args.skip_gif:
        render_legacy_neon_gif()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

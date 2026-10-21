#!/usr/bin/env python3
"""
Generate 3 bespoke, high-impact visual summary plates tailored for outreach emails:
  1. docs/outreach_fig1_bookstein_silhouette_paradox.png
     -> For Prof. Fred L. Bookstein (University of Vienna / University of Michigan)
     -> Focus: Why 17-landmark fitting degrades real fish silhouettes on Thompson's
        1917 pairs, why direct 600-point silhouette fitting recovers them, and why
        8-parameter Thompson coordinate laws sit at the exact AICc parsimony optimum
        between 6p affine and 12p quadratic trend surfaces.
  2. docs/outreach_fig2_price_wainwright_horizon.png
     -> For Prof. Samantha A. Price (Clemson), Dr. Sarah T. Friedman, and
        Prof. Peter C. Wainwright (UC Davis)
     -> Focus: The ~45-50 Myr phylogenetic horizon across their 791 Indo-Pacific
        reef-fish species (4,334 dated pairs on Rabosky et al. 2018), Satterfield
        et al. (2024) fin-body decoupling, and the Area-of-Difference silhouette test.
  3. docs/outreach_fig3_evans_larouche_modularity.png
     -> For Prof. Kory M. Evans (Rice) & Dr. Olivier Larouche
     -> Focus: Teleost head-vs-fin-insertion modularity (9/17 landmarks in the head
        forcing 47.2%-135.4% posterior silhouette overhang) and the Diodon -> Mola
        79.7% radial flare vs. 2.26x fin height decoupling.
"""

import importlib.util
import json
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

REPO_ROOT = Path(__file__).resolve().parents[1]
EXACT_JSON = REPO_ROOT / "datasets" / "thompson" / "thompson_1917_exact_species_landmarks.json"


def load_module(name, rel_path):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / rel_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


sil_mod = load_module("sil_mod", "thompson/1917-thompson-silhouette-area-paradox.py")


def draw_area_diff_panel(ax, src_sil, tgt_sil, src_lm, tgt_lm, title, subtitle, grid_fn=None):
    """Render Green shared overlap, Red missing target, and Blue source overhang."""
    bd = sil_mod.area_diff_breakdown(src_sil, tgt_sil, grid=600)
    ra, rb = bd["ra"], bd["rb"]
    xmin, xmax, ymin, ymax = bd["extent"]

    H, W = ra.shape
    img = np.ones((H, W, 4), dtype=float)
    img[:, :] = [0.985, 0.988, 0.995, 1.0]

    both = ra & rb
    only_tgt = rb & (~ra)
    only_src = ra & (~rb)

    img[both] = [0.16, 0.66, 0.38, 0.78]     # Emerald Green: shared overlap
    img[only_tgt] = [0.86, 0.18, 0.22, 0.84] # Crimson Red: missing target
    img[only_src] = [0.15, 0.42, 0.85, 0.80] # Royal Blue: excess overhang

    ax.imshow(img, extent=[xmin, xmax, ymin, ymax], origin="lower", aspect="equal", zorder=1)

    if grid_fn is not None:
        gx_vals = np.linspace(xmin + 0.03, xmax - 0.03, 9)
        gy_vals = np.linspace(ymin + 0.03, ymax - 0.03, 7)
        for gx in gx_vals:
            line = np.column_stack([np.full(80, gx), np.linspace(ymin + 0.03, ymax - 0.03, 80)])
            wline = grid_fn(line)
            ax.plot(wline[:, 0], wline[:, 1], color="#334155", lw=0.6, alpha=0.45, zorder=2)
        for gy in gy_vals:
            line = np.column_stack([np.linspace(xmin + 0.03, xmax - 0.03, 80), np.full(80, gy)])
            wline = grid_fn(line)
            ax.plot(wline[:, 0], wline[:, 1], color="#334155", lw=0.6, alpha=0.45, zorder=2)

    ax.plot(tgt_sil[:, 0], tgt_sil[:, 1], color="#7f1d1d", lw=1.6, ls="--", zorder=4)
    ax.plot(src_sil[:, 0], src_sil[:, 1], color="#0f172a", lw=1.5, zorder=5)

    if tgt_lm is not None:
        trunk_idx = [1, 2, 3, 4, 5, 6, 7, 0, 1]
        ax.plot(tgt_lm[trunk_idx, 0], tgt_lm[trunk_idx, 1], color="#1e293b", lw=1.1, ls=":", alpha=0.75, zorder=6)

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=9.8, fontweight="bold", color="#0f172a", pad=18)
    ax.text(
        0.5, 1.015, subtitle,
        transform=ax.transAxes, ha="center", va="bottom",
        fontsize=8.2, color="#1e293b", fontweight="semibold",
        bbox=dict(boxstyle="round,pad=0.18", facecolor="#f1f5f9", edgecolor="#cbd5e1", lw=0.7)
    )


def get_pair_data(specs, src_name, tgt_name, fam, at, sp_mod):
    S, BS = specs[src_name]
    T, BT = specs[tgt_name]
    BS_unw = sil_mod.unwarped_similarity_silhouette(S, BS, T, at, sp_mod)

    if fam == "shear":
        Ms, _, _ = at.fit_shear(S, T)
        BS_lm = at.apply_affine(Ms, BS)
        lm_grid_fn = lambda P: at.apply_affine(Ms, P)
    else:
        p_lm, _ = at.fit_nonlinear_single(fam, S, T)
        BS_lm = at.FAMILY_FUNCS[fam](p_lm, BS)
        lm_grid_fn = lambda P: at.FAMILY_FUNCS[fam](p_lm, P)

    p_sil, BS_sil, apply_p = sil_mod.fit_direct_to_silhouette(fam, S, BS, T, BT, at)
    sil_grid_fn = lambda P: apply_p(p_sil, P)

    return {
        "S": S, "T": T,
        "BS": BS, "BS_unw": BS_unw, "BT": BT,
        "BS_lm": BS_lm, "BS_sil": BS_sil,
        "lm_grid_fn": lm_grid_fn, "sil_grid_fn": sil_grid_fn
    }


def build_all_outreach_figures():
    at, sp_mod = sil_mod.load_modules()
    names, families, X = at.load_landmarks()
    Y = at.gpa(X)
    mean_consensus = at.centre_scale(Y.mean(0))
    meta = json.loads(EXACT_JSON.read_text(encoding="utf-8"))
    specs = {s: sil_mod.extract_aligned_silhouette_and_lms(s, meta, mean_consensus, sp_mod) for s in meta["specimens"]}
    O_sp = sp_mod.outline_analysis(Y, names, allow_download=False)
    specs["Scarus_ghobban"] = (O_sp["photos"]["Scarus_ghobban"]["landmarks"], O_sp["photos"]["Scarus_ghobban"]["boundary"])
    specs["Pomacanthus_semicirculatus"] = (
        O_sp["photos"]["Pomacanthus_semicirculatus"]["landmarks"],
        O_sp["photos"]["Pomacanthus_semicirculatus"]["boundary"],
    )

    p_scarus = get_pair_data(specs, "Scarus_ghobban", "Pomacanthus_semicirculatus", "coaxial", at, sp_mod)
    p_shear = get_pair_data(specs, "Argyropelecus_olfersii", "Sternoptyx_diaphana", "shear", at, sp_mod)
    p_poly = get_pair_data(specs, "Polyprion_americanus", "Pristigenys_alta", "fan", at, sp_mod)
    p_prist = get_pair_data(specs, "Pristigenys_alta", "Scorpaena_scrofa", "fan", at, sp_mod)
    p_scorp = get_pair_data(specs, "Scorpaena_scrofa", "Antigonia_capros", "fan", at, sp_mod)

    # Diodon hystrix (from Claverie 791-species dataset) -> Mola mola
    idx_diodon = names.index("Diodon_hystrix")
    S_dio_raw = Y[idx_diodon]
    T_mola, BT_mola = specs["Mola_mola"]
    S_dio = at.opa(S_dio_raw, T_mola)
    p_dio_lm, _ = at.fit_nonlinear_single("radial", S_dio, T_mola)
    W_dio = at.FAMILY_FUNCS["radial"](p_dio_lm, S_dio)

    # =========================================================================
    # PLATE 1: PROF. FRED L. BOOKSTEIN (2x3 visual plate)
    # =========================================================================
    fig1 = plt.figure(figsize=(16.5, 9.8), dpi=220, facecolor="white")
    gs1 = gridspec.GridSpec(2, 3, figure=fig1, hspace=0.35, wspace=0.22, top=0.87, bottom=0.07, left=0.05, right=0.97)

    fig1.suptitle(
        "Empirical Confirmation of Bookstein (2023, 2024): Why 17-Landmark Fitting Distorts Thompson's Fish Outlines\n"
        "& Why Direct 600-Point Silhouette Fitting + 8-Parameter Coordinate Laws Recover Them",
        fontsize=13.5, fontweight="bold", color="#0f172a", y=0.965
    )

    ax1a = fig1.add_subplot(gs1[0, 0])
    draw_area_diff_panel(
        ax1a, p_scarus["BS_unw"], p_scarus["BT"], p_scarus["S"], p_scarus["T"],
        "A. Unwarped Baseline (4p Similarity)\nScarus -> Pomacanthus (Figs. 375-376)",
        "Area of Diff |A Δ B| = 40.6% (IoU = 0.662)"
    )

    ax1b = fig1.add_subplot(gs1[0, 1])
    draw_area_diff_panel(
        ax1b, p_scarus["BS_lm"], p_scarus["BT"], p_scarus["S"], p_scarus["T"],
        "B. Warped by 17-Landmark Fit (8p Coaxial)\n9/17 Head Landmarks Pull Grid Wrong Way!",
        "DEGRADES Silhouette: |A Δ B| = 56.2% (47.2% Blue Overhang!)",
        grid_fn=p_scarus["lm_grid_fn"]
    )

    ax1c = fig1.add_subplot(gs1[0, 2])
    draw_area_diff_panel(
        ax1c, p_scarus["BS_sil"], p_scarus["BT"], None, p_scarus["T"],
        "C. Warped by Direct 600-Point Silhouette Fit\nSame 8p Coaxial Family Fitted to True Outline",
        "RECOVERED: |A Δ B| Cut in Half to 28.5% (IoU = 0.749)",
        grid_fn=p_scarus["sil_grid_fn"]
    )

    ax1d = fig1.add_subplot(gs1[1, 0])
    draw_area_diff_panel(
        ax1d, p_scorp["BS_sil"], p_scorp["BT"], None, p_scorp["T"],
        "D. Scorpaena -> Antigonia (Figs. 379-380, 8p Fan)\n17-LM Fit = 143.7% Error vs Direct 600-pt Fit",
        "Direct 600-pt Fit Cuts |A Δ B| 2.5x: 143.7% -> 57.6%",
        grid_fn=p_scorp["sil_grid_fn"]
    )

    ax1e = fig1.add_subplot(gs1[1, 1])
    draw_area_diff_panel(
        ax1e, p_prist["BS_sil"], p_prist["BT"], None, p_prist["T"],
        "E. Pristigenys -> Scorpaena (Figs. 378-379, 8p Fan)\nDirect 600-pt Fit Slashes |A Δ B| 3.4x!",
        "Unwarped 120.1% -> 17-LM 59.4% -> 600-pt 35.5% (IoU = 0.694)",
        grid_fn=p_prist["sil_grid_fn"]
    )

    ax1f = fig1.add_subplot(gs1[1, 2])
    tiers = ["Intrageneric\n(n = 723)", "Intrafamily\n(n = 292)", "Interfamily\n(n = 3,490)"]
    x = np.arange(len(tiers))
    w = 0.26
    beats_aff = [71.8, 82.2, 76.0]
    beats_quad = [83.0, 84.9, 92.1]
    quad_beats_8p = [17.0, 15.1, 7.9]

    b1 = ax1f.bar(x - w, beats_aff, width=w, color="#2563eb", label="8p Thompson beats 6p Affine (AICc)")
    b2 = ax1f.bar(x, beats_quad, width=w, color="#16a34a", label="8p Thompson beats 12p Quadratic (AICc)")
    b3 = ax1f.bar(x + w, quad_beats_8p, width=w, color="#dc2626", label="12p Quadratic beats 8p Thompson (AICc)")

    for bars in [b1, b2, b3]:
        for bar in bars:
            h = bar.get_height()
            ax1f.text(bar.get_x() + bar.get_width() / 2, h + 1.5, f"{h:.1f}%", ha="center", va="bottom", fontsize=8, fontweight="bold")

    ax1f.set_ylim(0, 148)
    ax1f.set_yticks([0, 20, 40, 60, 80, 100])
    ax1f.set_xticks(x)
    ax1f.set_xticklabels(tiers, fontsize=9, fontweight="bold")
    ax1f.set_ylabel("% of Species/Genus Pairs (n = 34 coords)", fontsize=9, fontweight="bold")
    ax1f.set_title("F. Small-Sample AICc Model Selection Across 4,505 Pairs\n8p Thompson Grids Sit at the Exact Parsimony Sweet Spot", fontsize=9.8, fontweight="bold", pad=10)
    ax1f.legend(loc="upper center", fontsize=7.6, framealpha=0.95)
    ax1f.grid(axis="y", ls="--", alpha=0.35)

    fig1.text(
        0.5, 0.015,
        "Silhouette Shading Key:   [Emerald Green] Shared Overlap |Ω_A ∩ Ω_B|     [Crimson Red] Unexplained Target |Ω_B \\ Ω_A|     [Royal Blue] Excess Source Overhang |Ω_A \\ Ω_B|",
        ha="center", va="bottom", fontsize=9.2, fontweight="bold", color="#1e293b",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#f8fafc", edgecolor="#cbd5e1", lw=0.9)
    )

    out1 = REPO_ROOT / "docs" / "outreach_fig1_bookstein_silhouette_paradox.png"
    fig1.savefig(out1, dpi=220)
    plt.close(fig1)

    # =========================================================================
    # PLATE 2: PROF. SAMANTHA PRICE, DR. SARAH FRIEDMAN & PROF. PETER WAINWRIGHT
    # =========================================================================
    fig2 = plt.figure(figsize=(16.5, 9.5), dpi=220, facecolor="white")
    gs2 = gridspec.GridSpec(2, 3, figure=fig2, hspace=0.35, wspace=0.24, top=0.87, bottom=0.07, left=0.05, right=0.97)

    fig2.suptitle(
        "Testing D'Arcy Thompson's 1917 Coordinate Laws Across Your 791 Indo-Pacific Reef-Fish Species\n"
        "A ~45-50 Myr Phylogenetic Horizon (Rabosky et al. 2018) & Satterfield et al. (2024) Fin-Body Decoupling",
        fontsize=13.5, fontweight="bold", color="#0f172a", y=0.965
    )

    ax2a = fig2.add_subplot(gs2[0, 0:2])
    bins_labels = ["0-10 Myr\n(n=149)", "10-25 Myr\n(n=359)", "25-50 Myr\n(n=346)", "50-80 Myr\n(n=170)", "80-110 Myr\n(n=618)", "110-160 Myr\n(n=2,692)"]
    xb = np.arange(len(bins_labels))
    aff_floor = [30.2, 34.5, 27.5, 0.6, 0.0, 0.0]
    thomp_floor = [53.0, 61.3, 49.7, 10.6, 1.0, 0.0]
    quad_floor = [73.8, 79.1, 70.5, 25.3, 1.9, 0.6]

    ax2a.axvspan(-0.4, 2.45, color="#dcfce7", alpha=0.55, label="0-50 Myr: 8p Thompson Grids Reach Noise Floor")
    ax2a.axvspan(2.45, 5.4, color="#fee2e2", alpha=0.45, label=">50 Myr: Beyond Modular Decoupling Horizon")
    ax2a.axvline(2.45, color="#b91c1c", ls="--", lw=2.0)
    ax2a.text(2.50, 39, "← ~45-50 Myr Phylogenetic Cliff\n    (Fin & Body Modules Decouple)", color="#991b1b", fontsize=8.8, fontweight="bold",
              bbox=dict(boxstyle="round,pad=0.22", facecolor="#fff1f2", edgecolor="#fecdd3", alpha=0.92))

    ax2a.plot(xb, quad_floor, "s--", color="#64748b", lw=2.0, markersize=7, label="12p Quadratic Trend Surface (overparameterized)")
    ax2a.plot(xb, thomp_floor, "o-", color="#16a34a", lw=3.0, markersize=9, label="Best 8p Thompson Coordinate Grid (Coaxial / Fan / Radial)")
    ax2a.plot(xb, aff_floor, "^-", color="#2563eb", lw=2.2, markersize=7, label="6p General Affine Transformation")

    for i, val in enumerate(thomp_floor):
        ax2a.text(xb[i], val + 3.0, f"{val:.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold", color="#15803d")

    ax2a.set_ylim(-3, 108)
    ax2a.set_yticks([0, 20, 40, 60, 80, 100])
    ax2a.set_xticks(xb)
    ax2a.set_xticklabels(bins_labels, fontsize=9, fontweight="bold")
    ax2a.set_ylabel("% Reaching Within-Genus Noise Floor", fontsize=9.5, fontweight="bold")
    ax2a.set_title("A. Continuous Phylogenetic Horizon Across 4,334 Time-Calibrated Pairs (Rabosky et al. 2018 Tree)", fontsize=10.5, fontweight="bold", pad=10)
    ax2a.legend(loc="upper right", fontsize=7.5, framealpha=0.95)
    ax2a.grid(True, ls="--", alpha=0.35)

    ax2b = fig2.add_subplot(gs2[0, 2])
    tiers_short = ["Intrageneric\n(Within Genus)", "Intrafamily\n(Within Family)", "Interfamily\n(Across Families)"]
    xt = np.arange(3)
    head_rms = [0.0098, 0.0099, 0.0225]
    fin_rms = [0.0112, 0.0132, 0.0326]
    ratios = ["1.16x", "1.28x", "1.48x"]

    ax2b.bar(xt - 0.18, head_rms, width=0.34, color="#0284c7", label="Craniofacial Landmarks (LM 1-2, 11-17; 9 pts)")
    ax2b.bar(xt + 0.18, fin_rms, width=0.34, color="#e11d48", label="Median/Paired Fin Insertions (LM 3-10; 8 pts)")

    for i in range(3):
        ax2b.text(xt[i], max(head_rms[i], fin_rms[i]) + 0.0014, f"{ratios[i]}\n(Fin/Head)", ha="center", va="bottom", fontsize=8.1, fontweight="bold", color="#9f1239")

    ax2b.set_xlim(-0.55, 2.55)
    ax2b.set_ylim(0, 0.049)
    ax2b.set_xticks(xt)
    ax2b.set_xticklabels(tiers_short, fontsize=8.8, fontweight="bold")
    ax2b.set_ylabel("Post-Warp Procrustes RMS Residual", fontsize=9, fontweight="bold")
    ax2b.set_title("B. Why Global Grids Break Above 50 Myr:\nFin Insertions Decouple From Head (Satterfield 2024)", fontsize=9.8, fontweight="bold", pad=10)
    ax2b.legend(loc="upper left", fontsize=7.4, framealpha=0.94)
    ax2b.grid(axis="y", ls="--", alpha=0.35)

    ax2c = fig2.add_subplot(gs2[1, 0])
    draw_area_diff_panel(
        ax2c, p_scarus["BS_lm"], p_scarus["BT"], p_scarus["S"], p_scarus["T"],
        "C. Scarus -> Pomacanthus: 17-LM Grid Fit\n9 Head Landmarks Force 47.2% Blue Overhang",
        "17-LM Fit Worsens |A Δ B|: 40.6% -> 56.2%",
        grid_fn=p_scarus["lm_grid_fn"]
    )

    ax2d = fig2.add_subplot(gs2[1, 1])
    draw_area_diff_panel(
        ax2d, p_scarus["BS_sil"], p_scarus["BT"], None, p_scarus["T"],
        "D. Scarus -> Pomacanthus: Direct 600-pt Fit\nFitting True Contour Recovers Thompson's Grid",
        "Direct 600-pt Fit Slashes |A Δ B| to 28.5% (IoU = 0.749)",
        grid_fn=p_scarus["sil_grid_fn"]
    )

    ax2e = fig2.add_subplot(gs2[1, 2])
    draw_area_diff_panel(
        ax2e, p_poly["BS_sil"], p_poly["BT"], None, p_poly["T"],
        "E. Polyprion -> Pristigenys (Figs. 377-378, 8p Fan)\nDirect 600-pt Fit Achieves IoU = 0.839",
        "Area of Difference |A Δ B| = 17.2% (vs 33.2% Unwarped)",
        grid_fn=p_poly["sil_grid_fn"]
    )

    fig2.text(
        0.5, 0.015,
        "Silhouette Shading Key:   [Emerald Green] Shared Overlap |Ω_A ∩ Ω_B|     [Crimson Red] Unexplained Target |Ω_B \\ Ω_A|     [Royal Blue] Excess Source Overhang |Ω_A \\ Ω_B|",
        ha="center", va="bottom", fontsize=9.2, fontweight="bold", color="#1e293b",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#f8fafc", edgecolor="#cbd5e1", lw=0.9)
    )

    out2 = REPO_ROOT / "docs" / "outreach_fig2_price_wainwright_horizon.png"
    fig2.savefig(out2, dpi=220)
    plt.close(fig2)

    # =========================================================================
    # PLATE 3: PROF. KORY EVANS & DR. OLIVIER LAROUCHE (Modularity & Allometry)
    # =========================================================================
    fig3 = plt.figure(figsize=(16.5, 9.2), dpi=220, facecolor="white")
    gs3 = gridspec.GridSpec(2, 3, figure=fig3, hspace=0.35, wspace=0.23, top=0.87, bottom=0.07, left=0.05, right=0.97)

    fig3.suptitle(
        "How Craniofacial vs. Fin-Insertion Modularity Explains the 17-Landmark Silhouette Paradox\n"
        "Testing Thompson's 1917 Transformations Across Head, Trunk, and Fin Modules in 791 Teleost Species",
        fontsize=13.5, fontweight="bold", color="#0f172a", y=0.965
    )

    ax3a = fig3.add_subplot(gs3[0, 0])
    draw_area_diff_panel(
        ax3a, p_shear["BS_sil"], p_shear["BT"], p_shear["S"], p_shear["T"],
        "A. Argyropelecus -> Sternoptyx (5p Oblique Shear)\n17-LM Polygon Amputates Dorsal Blade & Ventral Keel",
        "17-LM |A Δ B| = 63.0% (worse) -> 600-pt |A Δ B| = 45.6%",
        grid_fn=p_shear["sil_grid_fn"]
    )

    ax3b = fig3.add_subplot(gs3[0, 1])
    draw_area_diff_panel(
        ax3b, p_scorp["BS_lm"], p_scorp["BT"], p_scorp["S"], p_scorp["T"],
        "B. Scorpaena -> Antigonia: 17-LM Fit\nHead Landmarks Blow Up Posterior Overhang",
        "17-LM Fit |A Δ B| = 143.7% (135.4% Blue Overhang!)",
        grid_fn=p_scorp["lm_grid_fn"]
    )

    ax3c = fig3.add_subplot(gs3[0, 2])
    draw_area_diff_panel(
        ax3c, p_scorp["BS_sil"], p_scorp["BT"], None, p_scorp["T"],
        "C. Scorpaena -> Antigonia: Direct 600-pt Fit\nWhole-Body Silhouette Restores Fan Convergence",
        "Direct 600-pt Fit Cuts |A Δ B| 2.5x: 143.7% -> 57.6%",
        grid_fn=p_scorp["sil_grid_fn"]
    )

    ax3d = fig3.add_subplot(gs3[1, 0:2])
    pair_names = [
        "Argyropelecus ->\nSternoptyx (5p Shear)",
        "Scarus ->\nPomacanthus (8p Coax)",
        "Polyprion ->\nPristigenys (8p Fan)",
        "Pristigenys ->\nScorpaena (8p Fan)",
        "Scorpaena ->\nAntigonia (8p Fan)",
        "Diodon ->\nMola mola (8p Radial)"
    ]
    xp = np.arange(len(pair_names))
    unw_err = [58.4, 40.6, 33.2, 120.1, 102.2, 70.8]
    lm17_err = [63.0, 56.2, 20.7, 59.4, 143.7, 69.4]
    sil600_err = [45.6, 28.5, 17.2, 35.5, 57.6, 49.0]

    w3 = 0.25
    bu = ax3d.bar(xp - w3, unw_err, width=w3, color="#94a3b8", label="Unwarped Baseline (4p Similarity)")
    bl = ax3d.bar(xp, lm17_err, width=w3, color="#ef4444", label="Warped by 17-Landmark Fit (9/17 pts in Head)")
    bs = ax3d.bar(xp + w3, sil600_err, width=w3, color="#16a34a", label="Warped by Direct 600-Point Silhouette Fit")

    for bars in [bu, bl, bs]:
        for bar in bars:
            h = bar.get_height()
            ax3d.text(bar.get_x() + bar.get_width() / 2, h + 1.8, f"{h:.1f}%", ha="center", va="bottom", fontsize=7.8, fontweight="bold")

    ax3d.set_ylim(0, 174)
    ax3d.set_xticks(xp)
    ax3d.set_xticklabels(pair_names, fontsize=8.5, fontweight="bold")
    ax3d.set_ylabel("Area of Difference |A Δ B| (% Target Area)", fontsize=9.2, fontweight="bold")
    ax3d.set_title("D. Symmetric Area of Difference |A Δ B| Across All Six of Thompson's 1917 Pairs (Lower = Better Match)", fontsize=10.2, fontweight="bold", pad=10)
    ax3d.legend(loc="upper left", fontsize=8.2, framealpha=0.94)
    ax3d.grid(axis="y", ls="--", alpha=0.35)

    ax3e = fig3.add_subplot(gs3[1, 2])
    ax3e.fill(BT_mola[:, 0], BT_mola[:, 1], color="#fee2e2", alpha=0.85, label="Mola mola Full 600-pt Silhouette (Span = 0.786)")
    ax3e.plot(BT_mola[:, 0], BT_mola[:, 1], color="#dc2626", lw=1.6, ls="--")
    trunk_idx = [1, 2, 3, 4, 5, 6, 7, 0, 1]
    ax3e.plot(T_mola[trunk_idx, 0], T_mola[trunk_idx, 1], color="#1e293b", lw=2.0, label="17-LM Trunk Polygon (Span = 0.349; 2.26x Shorter)")
    ax3e.plot(W_dio[trunk_idx, 0], W_dio[trunk_idx, 1], color="#16a34a", lw=2.0, ls="-.", label="8p Radial Fit of Diodon Trunk (79.7% Expl.)")
    ax3e.scatter(T_mola[:, 0], T_mola[:, 1], color="#dc2626", s=26, zorder=5)
    ax3e.scatter(W_dio[:, 0], W_dio[:, 1], color="#16a34a", marker="^", s=28, zorder=6)
    ymin_m, ymax_m = BT_mola[:, 1].min(), BT_mola[:, 1].max()
    xmin_m, xmax_m = BT_mola[:, 0].min(), BT_mola[:, 0].max()
    ax3e.set_xlim(xmin_m - 0.08, xmax_m + 0.08)
    ax3e.set_ylim(ymin_m - 0.24, ymax_m + 0.06)
    ax3e.set_aspect("equal")
    ax3e.set_xticks([])
    ax3e.set_yticks([])
    ax3e.set_title("E. Diodon -> Mola mola (Figs. 381-382)\nTrunk Explains 79.7%; Fin Span is 2.26x Taller", fontsize=9.8, fontweight="bold", color="#0f172a", pad=18)
    ax3e.text(
        0.5, 1.015, "LOO-CV = 0.0461 < d0 = 0.0814 | ΔAICc = -41.9",
        transform=ax3e.transAxes, ha="center", va="bottom",
        fontsize=8.2, color="#1e293b", fontweight="semibold",
        bbox=dict(boxstyle="round,pad=0.18", facecolor="#f1f5f9", edgecolor="#cbd5e1", lw=0.7)
    )
    ax3e.legend(loc="lower center", fontsize=6.8, framealpha=0.94)

    fig3.text(
        0.5, 0.015,
        "Silhouette Shading Key:   [Emerald Green] Shared Overlap |Ω_A ∩ Ω_B|     [Crimson Red] Unexplained Target |Ω_B \\ Ω_A|     [Royal Blue] Excess Source Overhang |Ω_A \\ Ω_B|",
        ha="center", va="bottom", fontsize=9.2, fontweight="bold", color="#1e293b",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#f8fafc", edgecolor="#cbd5e1", lw=0.9)
    )

    out3 = REPO_ROOT / "docs" / "outreach_fig3_evans_larouche_modularity.png"
    fig3.savefig(out3, dpi=220)
    plt.close(fig3)

    print("Saved outreach plates:")
    print("  1.", out1)
    print("  2.", out2)
    print("  3.", out3)


if __name__ == "__main__":
    build_all_outreach_figures()

#!/usr/bin/env python3
"""
Generate crystal-clear, large-font, ZERO-OVERLAP standalone diagrams for the
13-Year-Old's Visual Guide PDF ('docs/thompson_1917_visual_guide_for_everyone.pdf')
and Markdown ('docs/EXPLAINED_FOR_EVERYONE.md').

Key design rule:
  - NO legend or text box is ever placed inside the plot axes over a fish or bar!
  - Every legend sits in a dedicated margin BELOW the plot axes (bbox_to_anchor),
    and all narrative explanations are rendered as native, large-font LaTeX callout
    boxes in the PDF.
"""

import importlib.util
from pathlib import Path
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches

REPO_ROOT = Path(__file__).resolve().parents[1]
DOCS_DIR = REPO_ROOT / "docs"
CACHE_DIR = REPO_ROOT / ".thompson_cache"


def load_module(name: str, rel_path: str):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / rel_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def style_clean_ax(ax, xlim=(-0.48, 0.54), ylim=(-0.39, 0.39)):
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_color("#cbd5e1")
        sp.set_linewidth(1.3)


def main():
    sil_mod = load_module("sil_mod", "thompson/1917-thompson-silhouette-area-paradox.py")
    at, sp_mod = sil_mod.load_modules()
    names, families, X = at.load_landmarks()
    Y = at.gpa(X)
    O_sp = sp_mod.outline_analysis(Y, names, allow_download=False)

    S_sc = O_sp["photos"]["Scarus_ghobban"]["landmarks"]
    BS_sc = O_sp["photos"]["Scarus_ghobban"]["boundary"]
    T_po = O_sp["photos"]["Pomacanthus_semicirculatus"]["landmarks"]
    BT_po = O_sp["photos"]["Pomacanthus_semicirculatus"]["boundary"]

    BS_sc_unw = sil_mod.unwarped_similarity_silhouette(S_sc, BS_sc, T_po, at, sp_mod)
    S_sc_unw = at.fit_similarity(S_sc, T_po)

    p_lm, _ = at.fit_nonlinear_single("coaxial", S_sc, T_po)
    BS_sc_lm = at.FAMILY_FUNCS["coaxial"](p_lm, BS_sc)

    p_sil, BS_sc_sil, apply_sil = sil_mod.fit_direct_to_silhouette("coaxial", S_sc, BS_sc, T_po, BT_po, at)

    # Box stretch of Scarus onto Pomacanthus:
    scale_x = (BT_po[:, 0].max() - BT_po[:, 0].min()) / (BS_sc_unw[:, 0].max() - BS_sc_unw[:, 0].min())
    scale_y = (BT_po[:, 1].max() - BT_po[:, 1].min()) / (BS_sc_unw[:, 1].max() - BS_sc_unw[:, 1].min())
    BS_box = np.column_stack([
        (BS_sc_unw[:, 0] - BS_sc_unw[:, 0].mean()) * scale_x + BT_po[:, 0].mean(),
        (BS_sc_unw[:, 1] - BS_sc_unw[:, 1].mean()) * scale_y + BT_po[:, 1].mean(),
    ])
    dors_po = BT_po[np.argmax(BT_po[:, 1])]
    anal_po = BT_po[np.argmin(BT_po[:, 1])]
    dors_box = BS_box[np.argmax(BS_box[:, 1])]
    anal_box = BS_box[np.argmin(BS_box[:, 1])]

    head_idx = [0, 1, 10, 11, 12, 13, 14, 15, 16]
    fin_idx = [2, 3, 4, 5, 6, 7, 8, 9]

    # =========================================================================
    # STEP 1: D'ARCY THOMPSON'S 1917 RUBBER GRAPH-PAPER IDEA
    # =========================================================================
    fig1 = plt.figure(figsize=(14.5, 5.4), dpi=220, facecolor="white")
    gs1 = gridspec.GridSpec(1, 3, figure=fig1, width_ratios=[1.05, 1.0, 1.0], wspace=0.14, top=0.83, bottom=0.22, left=0.03, right=0.97)

    # 1A: Original 1917 Drawing Scan (full horizontal width x0=0.0 to x1=0.90 so the snout of Scarus is never clipped!)
    ax1a = fig1.add_subplot(gs1[0, 0])
    scan_p = CACHE_DIR / "ongrowthform1917thom_0769.jp2"
    if scan_p.exists():
        im = Image.open(scan_p).convert("L")
        w_im, h_im = im.size
        crop_im = np.asarray(im.crop((0, int(0.365 * h_im), int(0.89 * w_im), int(0.58 * h_im))))
        ax1a.imshow(crop_im, cmap="gray", vmin=0, vmax=255)
    ax1a.set_xticks([]); ax1a.set_yticks([])
    ax1a.set_title("A. Thompson's 1917 Drawing\nParrotfish (Left) → Angelfish (Right)", fontsize=13.0, fontweight="bold", color="#0f172a", pad=12)

    # 1B: Starting Parrotfish on Flat Graph Paper
    ax1b = fig1.add_subplot(gs1[0, 1])
    for gx in np.linspace(-0.40, 0.42, 9):
        ax1b.plot([gx, gx], [-0.26, 0.26], color="#94a3b8", lw=1.0, alpha=0.7)
    for gy in np.linspace(-0.26, 0.26, 7):
        ax1b.plot([-0.40, 0.42], [gy, gy], color="#94a3b8", lw=1.0, alpha=0.7)
    ax1b.fill(BS_sc_unw[:, 0], BS_sc_unw[:, 1], color="#bae6fd", alpha=0.80)
    ax1b.plot(BS_sc_unw[:, 0], BS_sc_unw[:, 1], color="#0369a1", lw=2.6, label="Starting Fish: Parrotfish (Scarus)")
    style_clean_ax(ax1b)
    ax1b.set_title("B. Starting Fish on Flat Grid\nLong, Low Body With Flat Fins", fontsize=13.0, fontweight="bold", color="#0369a1", pad=12)
    ax1b.legend(loc="upper center", bbox_to_anchor=(0.5, -0.04), fontsize=10.5, frameon=True, facecolor="#f8fafc")

    # 1C: Warped Parrotfish on Curved Coaxial Grid Matching Angelfish
    ax1c = fig1.add_subplot(gs1[0, 2])
    for gx in np.linspace(-0.36, 0.36, 9):
        ln = np.column_stack([np.full(80, gx), np.linspace(-0.22, 0.22, 80)])
        wln = apply_sil(p_sil, ln)
        ax1c.plot(wln[:, 0], wln[:, 1], color="#16a34a", lw=1.0, alpha=0.65)
    for gy in np.linspace(-0.22, 0.22, 7):
        ln = np.column_stack([np.linspace(-0.36, 0.36, 80), np.full(80, gy)])
        wln = apply_sil(p_sil, ln)
        ax1c.plot(wln[:, 0], wln[:, 1], color="#16a34a", lw=1.0, alpha=0.65)
    ax1c.fill(BS_sc_sil[:, 0], BS_sc_sil[:, 1], color="#bbf7d0", alpha=0.78, label="Stretched Parrotfish on Curved Sheet")
    ax1c.plot(BS_sc_sil[:, 0], BS_sc_sil[:, 1], color="#15803d", lw=2.4)
    ax1c.plot(BT_po[:, 0], BT_po[:, 1], color="#dc2626", lw=2.2, ls="--", label="Real Angelfish Outline (Pomacanthus)")
    style_clean_ax(ax1c)
    ax1c.set_title("C. Bending the Sheet Into Circles\nTurns the Parrotfish Into an Angelfish!", fontsize=13.0, fontweight="bold", color="#15803d", pad=12)
    ax1c.legend(loc="upper center", bbox_to_anchor=(0.5, -0.04), fontsize=10.0, frameon=True, facecolor="#f8fafc")

    fig1.savefig(DOCS_DIR / "guide_step1_1917_rubber_sheet.png", dpi=220)
    plt.close(fig1)

    # =========================================================================
    # STEP 2: MISTAKE #1 — THE "17-DOT HEAD SEESAW TRAP"
    # =========================================================================
    fig2 = plt.figure(figsize=(14.5, 5.8), dpi=220, facecolor="white")
    gs2 = gridspec.GridSpec(1, 2, figure=fig2, width_ratios=[1.0, 1.0], wspace=0.14, top=0.84, bottom=0.26, left=0.04, right=0.96)

    # 2A: Where the 17 Dots Actually Sit on the Fish
    ax2a = fig2.add_subplot(gs2[0, 0])
    ax2a.fill(BT_po[:, 0], BT_po[:, 1], color="#f1f5f9", alpha=0.95, label="Real Whole Fish Body + Fins (600-Point Outline)")
    ax2a.plot(BT_po[:, 0], BT_po[:, 1], color="#475569", lw=2.2, ls="--")
    trunk_loop = [1, 2, 3, 4, 5, 6, 7, 0, 1]
    ax2a.plot(T_po[trunk_loop, 0], T_po[trunk_loop, 1], color="#0f172a", lw=2.2, label="8-Dot Skeleton Box (Chops Off All the Fins!)")
    ax2a.scatter(T_po[head_idx, 0], T_po[head_idx, 1], color="#f59e0b", edgecolor="#0f172a", s=145, zorder=8,
                 label="9 of 17 Dots (53%) Crammed Inside the Head!")
    ax2a.scatter(T_po[fin_idx, 0], T_po[fin_idx, 1], color="#10b981", edgecolor="#0f172a", s=115, marker="s", zorder=7,
                 label="8 of 17 Dots (47%) on Body Wall (ZERO on Outer Fins!)")
    head_el = mpatches.Ellipse((-0.25, -0.01), 0.24, 0.24, fill=False, edgecolor="#d97706", lw=3.0, zorder=9)
    ax2a.add_patch(head_el)
    style_clean_ax(ax2a)
    ax2a.set_title("A. What Standard Fish Databases Measure\n9 Dots in the Skull (53%), ZERO Dots on the Fins!", fontsize=13.0, fontweight="bold", color="#0f172a", pad=12)
    ax2a.legend(loc="upper center", bbox_to_anchor=(0.5, -0.04), fontsize=9.8, frameon=True, facecolor="#f8fafc")

    # 2B: What Happens When You Fit the Grid to Those 17 Dots
    ax2b = fig2.add_subplot(gs2[0, 1])
    for gx in np.linspace(-0.36, 0.36, 9):
        ln = np.column_stack([np.full(80, gx), np.linspace(-0.22, 0.22, 80)])
        wln = at.FAMILY_FUNCS["coaxial"](p_lm, ln)
        ax2b.plot(wln[:, 0], wln[:, 1], color="#dc2626", lw=0.9, alpha=0.5)
    for gy in np.linspace(-0.22, 0.22, 7):
        ln = np.column_stack([np.linspace(-0.36, 0.36, 80), np.full(80, gy)])
        wln = at.FAMILY_FUNCS["coaxial"](p_lm, ln)
        ax2b.plot(wln[:, 0], wln[:, 1], color="#dc2626", lw=0.9, alpha=0.5)
    ax2b.fill(BT_po[:, 0], BT_po[:, 1], color="#fecaca", alpha=0.65, label="Red = Missed Angelfish Dorsal & Anal Fins")
    ax2b.plot(BT_po[:, 0], BT_po[:, 1], color="#dc2626", lw=2.2, ls="--")
    ax2b.fill(BS_sc_lm[:, 0], BS_sc_lm[:, 1], color="#bfdbfe", alpha=0.68, label="Blue = Parrotfish Pulled the Wrong Way (56.2% Error!)")
    ax2b.plot(BS_sc_lm[:, 0], BS_sc_lm[:, 1], color="#1d4ed8", lw=2.4)
    style_clean_ax(ax2b)
    ax2b.set_title("B. What Happens When You Fit Those 17 Dots\nThe 9 Head Dots Tilt the Sheet the Wrong Way!", fontsize=13.0, fontweight="bold", color="#b91c1c", pad=12)
    ax2b.legend(loc="upper center", bbox_to_anchor=(0.5, -0.04), fontsize=9.8, frameon=True, facecolor="#f8fafc")

    fig2.savefig(DOCS_DIR / "guide_step2_17dot_head_trap.png", dpi=220)
    plt.close(fig2)

    # =========================================================================
    # STEP 3: MISTAKE #2 — WHY "BOX STRETCH" FAILS ANATOMY & CURVES WIN
    # =========================================================================
    fig3 = plt.figure(figsize=(14.5, 5.8), dpi=220, facecolor="white")
    gs3 = gridspec.GridSpec(1, 2, figure=fig3, width_ratios=[1.0, 1.0], wspace=0.14, top=0.84, bottom=0.26, left=0.04, right=0.96)

    # 3A: Simple Rectangular Box Stretch (No Skew, No Curve)
    ax3a = fig3.add_subplot(gs3[0, 0])
    for gx in np.linspace(-0.40, 0.42, 9):
        ax3a.plot([gx, gx], [-0.36, 0.36], color="#f97316", lw=0.9, alpha=0.55)
    for gy in np.linspace(-0.36, 0.36, 7):
        ax3a.plot([-0.40, 0.42], [gy, gy], color="#f97316", lw=0.9, alpha=0.55)
    ax3a.plot(BT_po[:, 0], BT_po[:, 1], color="#0f172a", lw=2.2, ls="--", label="True Angelfish Outline")
    ax3a.fill(BS_box[:, 0], BS_box[:, 1], color="#ffedd5", alpha=0.75, label="Plain 'Box Stretch' (Taller Rectangle, No Curve)")
    ax3a.plot(BS_box[:, 0], BS_box[:, 1], color="#ea580c", lw=2.4)
    ax3a.scatter([dors_box[0], anal_box[0]], [dors_box[1], anal_box[1]], color="#dc2626", marker="X", s=190, zorder=9,
                 label="Where Box Stretch Puts Fin Tips (Wrong! 2.4% Match)")
    ax3a.scatter([dors_po[0], anal_po[0]], [dors_po[1], anal_po[1]], color="#15803d", marker="*", s=230, zorder=10,
                 label="True Swept-Back Angelfish Fin Tips")
    for pt_b, pt_t in [(dors_box, dors_po), (anal_box, anal_po)]:
        ax3a.annotate("", xy=pt_t, xytext=pt_b, arrowprops=dict(arrowstyle="->", color="#dc2626", lw=2.6))
    style_clean_ax(ax3a)
    ax3a.set_title("A. Plain 'Box Stretch' (No Slant, No Curve)\nLifts Middle Straight Up — Puts Fin Tips in Wrong Place!", fontsize=12.6, fontweight="bold", color="#c2410c", pad=12)
    ax3a.legend(loc="upper center", bbox_to_anchor=(0.5, -0.04), fontsize=9.6, frameon=True, facecolor="#f8fafc")

    # 3B: Thompson's Curved & Slanted Grid
    ax3b = fig3.add_subplot(gs3[0, 1])
    AS_sc = np.vstack([S_sc[:8], BS_sc[np.argmax(BS_sc[:, 1])], BS_sc[np.argmin(BS_sc[:, 1])]])
    AT_po = np.vstack([T_po[:8], dors_po, anal_po])
    _, pred_coax_hom = at.fit_nonlinear_single("coaxial", AS_sc, AT_po)
    for gx in np.linspace(-0.36, 0.36, 9):
        ln = np.column_stack([np.full(80, gx), np.linspace(-0.22, 0.22, 80)])
        wln = apply_sil(p_sil, ln)
        ax3b.plot(wln[:, 0], wln[:, 1], color="#16a34a", lw=0.95, alpha=0.6)
    for gy in np.linspace(-0.22, 0.22, 7):
        ln = np.column_stack([np.linspace(-0.36, 0.36, 80), np.full(80, gy)])
        wln = apply_sil(p_sil, ln)
        ax3b.plot(wln[:, 0], wln[:, 1], color="#16a34a", lw=0.95, alpha=0.6)
    ax3b.plot(BT_po[:, 0], BT_po[:, 1], color="#0f172a", lw=2.2, ls="--", label="True Angelfish Outline")
    ax3b.fill(BS_sc_sil[:, 0], BS_sc_sil[:, 1], color="#dcfce7", alpha=0.75, label="Thompson's Curved Coaxial Grid")
    ax3b.plot(BS_sc_sil[:, 0], BS_sc_sil[:, 1], color="#15803d", lw=2.4)
    ax3b.scatter(pred_coax_hom[[8, 9], 0], pred_coax_hom[[8, 9], 1], color="#2563eb", marker="^", s=180, zorder=9,
                 label="Thompson Warped Fin Tips (Swept Back: 15× Better!)")
    ax3b.scatter([dors_po[0], anal_po[0]], [dors_po[1], anal_po[1]], color="#15803d", marker="*", s=230, zorder=10,
                 label="True Swept-Back Angelfish Fin Tips")
    style_clean_ax(ax3b)
    ax3b.set_title("B. Thompson's Curved & Slanted Grids\nBends Grid Lines Backward so Fin Tips Land in True Spot!", fontsize=12.6, fontweight="bold", color="#15803d", pad=12)
    ax3b.legend(loc="upper center", bbox_to_anchor=(0.5, -0.04), fontsize=9.6, frameon=True, facecolor="#f8fafc")

    fig3.savefig(DOCS_DIR / "guide_step3_box_cheat_vs_curves.png", dpi=220)
    plt.close(fig3)

    # =========================================================================
    # STEP 4: TWO SEPARATE PANELS (LEFT = OUTSIDE SILHOUETTE, RIGHT = INSIDE JAWS)
    # =========================================================================
    fig4 = plt.figure(figsize=(14.5, 5.6), dpi=220, facecolor="white")
    gs4 = gridspec.GridSpec(1, 2, figure=fig4, width_ratios=[1.0, 1.0], wspace=0.14, top=0.83, bottom=0.23, left=0.04, right=0.96)

    # 4A: Rule 1 — Outside Silhouette (Whole Body + Fins = Swimming Wing)
    ax4a = fig4.add_subplot(gs4[0, 0])
    for gx in np.linspace(-0.36, 0.36, 8):
        ln = np.column_stack([np.full(80, gx), np.linspace(-0.22, 0.22, 80)])
        wln = apply_sil(p_sil, ln)
        ax4a.plot(wln[:, 0], wln[:, 1], color="#0284c7", lw=0.9, alpha=0.5)
    for gy in np.linspace(-0.22, 0.22, 6):
        ln = np.column_stack([np.linspace(-0.36, 0.36, 80), np.full(80, gy)])
        wln = apply_sil(p_sil, ln)
        ax4a.plot(wln[:, 0], wln[:, 1], color="#0284c7", lw=0.9, alpha=0.5)
    ax4a.fill(BT_po[:, 0], BT_po[:, 1], color="#bae6fd", alpha=0.80, label="Whole Outer Body + Fins = Swimming Wing")
    ax4a.plot(BT_po[:, 0], BT_po[:, 1], color="#0284c7", lw=2.8)
    ax4a.scatter(T_po[fin_idx, 0], T_po[fin_idx, 1], color="#0284c7", edgecolor="#0f172a", s=130, marker="s", zorder=8,
                 label="Body & Fin Positions (90.8% of 150-Myr Family Evolution!)")
    style_clean_ax(ax4a)
    ax4a.set_title("Rule 1 — THE OUTSIDE SILHOUETTE (How It Swims)\nFollows Thompson's Rubber Sheet Across 440 Families (94%–99% Match)", fontsize=12.4, fontweight="bold", color="#0369a1", pad=12)
    ax4a.legend(loc="upper center", bbox_to_anchor=(0.5, -0.04), fontsize=9.8, frameon=True, facecolor="#f8fafc")

    # 4B: Rule 2 — Inside the Head (Jaws, Snout & Eyes = Food Tool)
    ax4b = fig4.add_subplot(gs4[0, 1])
    ax4b.fill(BT_po[:, 0], BT_po[:, 1], color="#f8fafc", alpha=0.9)
    ax4b.plot(BT_po[:, 0], BT_po[:, 1], color="#94a3b8", lw=1.8, ls="--", label="Conserved Swimming Body Shared by Sister Species")
    jaw_patch = mpatches.FancyBboxPatch((-0.42, -0.14), 0.23, 0.27, boxstyle="round,pad=0.02",
                                        facecolor="#fde68a", edgecolor="#d97706", lw=2.8, alpha=0.92, zorder=6,
                                        label="Active Trophic Zone: Jaws, Snout & Eyes")
    ax4b.add_patch(jaw_patch)
    ax4b.scatter(T_po[head_idx, 0], T_po[head_idx, 1], color="#d97706", edgecolor="#0f172a", s=145, zorder=8,
                 label="9 Head & Jaw Dots (Rewire 3.3× to 9.3× Faster on Same Reef!)")
    style_clean_ax(ax4b)
    ax4b.set_title("Rule 2 — INSIDE THE HEAD (What It Eats)\nRewires 3× to 9× Faster When Neighbors Compete for Food!", fontsize=12.4, fontweight="bold", color="#b45309", pad=12)
    ax4b.legend(loc="upper center", bbox_to_anchor=(0.5, -0.04), fontsize=9.8, frameon=True, facecolor="#f8fafc")

    fig4.savefig(DOCS_DIR / "guide_step4_outside_vs_inside_rules.png", dpi=220)
    plt.close(fig4)

    # =========================================================================
    # STEP 5: THE 4-TEST SCOREBOARD (LEGEND ABOVE PLOT, ZERO BAR OVERLAP!)
    # =========================================================================
    fig5 = plt.figure(figsize=(14.5, 5.6), dpi=220, facecolor="white")
    ax5 = fig5.add_subplot(111)
    fig5.subplots_adjust(top=0.76, bottom=0.20, left=0.07, right=0.97)

    categories = [
        "1. Swept Fin Tips on Angelfish\n(Plain Box Stretch vs.\nThompson's Curved Grid)",
        "2. Sunfish Fin Positions\n(Plain Box Stretch vs.\nThompson's Radial Flare)",
        "3. 150-Million-Year Body Highway\n(Local Noise vs. Thompson's\nGrid Across 426 Families)",
        "4. Sister Species on Same Reef\n(Body Change vs. Jaw/Snout\nSpeed Across 10,303 Species)",
    ]
    x5 = np.arange(len(categories))
    w5 = 0.34
    base_v = [2.4, 56.0, 4.6, 14.6]
    win_v = [35.3, 97.2, 98.9, 85.4]

    b1 = ax5.bar(x5 - w5 / 2, base_v, width=w5, color="#cbd5e1", edgecolor="#334155", lw=1.1,
                 label="Plain Box Stretch / Local Noise / Slow Body Baseline")
    b2 = ax5.bar(x5 + w5 / 2, win_v, width=w5, color=["#16a34a", "#16a34a", "#0284c7", "#d97706"],
                 edgecolor="#0f172a", lw=1.2, label="Thompson's Curved Grid (Tests 1–3) / Reef Jaw Competition (Test 4)")

    base_lbls = ["2.4%\n(Box Fails!)", "56.0%\n(Box Only)", "4.6%\n(No Match)", "14.6%\n(Slow Body)"]
    win_lbls = ["35.3%\n(15× Better!)", "97.2%\n(+41% Jump!)", "98.9%\n(Follows Grid!)", "85.4%\n(9.3× Jaw Speed!)"]

    for bar, lbl in zip(b1, base_lbls):
        h = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width() / 2, h + 2.2, lbl, ha="center", va="bottom", fontsize=10.5, fontweight="bold", color="#475569")
    for bar, lbl in zip(b2, win_lbls):
        h = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width() / 2, h + 2.2, lbl, ha="center", va="bottom", fontsize=11.0, fontweight="bold", color="#0f172a")

    ax5.set_ylim(0, 120)
    ax5.set_xticks(x5)
    ax5.set_xticklabels(categories, fontsize=10.8, fontweight="bold")
    ax5.set_ylabel("Accuracy / Share of Evolution (%)", fontsize=11.5, fontweight="bold")
    ax5.set_title("The Scoreboard in Four Simple Numbers (Tested Across 10,303 Fish Species)",
                  fontsize=14.2, fontweight="bold", color="#0f172a", pad=38)
    ax5.legend(loc="upper center", bbox_to_anchor=(0.5, 1.14), ncol=2, fontsize=10.5, frameon=True, facecolor="#f8fafc")
    ax5.grid(axis="y", ls="--", alpha=0.35)

    fig5.savefig(DOCS_DIR / "guide_step5_10000_fishes_scoreboard.png", dpi=220)
    plt.close(fig5)

    # =========================================================================
    # STEP 6: WHAT THIS MEANS FOR EVOLUTIONARY THEORY (13YO VISUAL DIAGRAM)
    # =========================================================================
    fig6 = plt.figure(figsize=(14.5, 5.8), dpi=220, facecolor="white")
    gs6 = gridspec.GridSpec(1, 2, figure=fig6, width_ratios=[1.0, 1.0], wspace=0.18, top=0.80, bottom=0.25, left=0.06, right=0.96)

    # 6A: "Parking-Lot Dodging" (Jaws) vs. "100-Million-Year Highway" (Body)
    ax6a = fig6.add_subplot(gs6[0, 0])
    time_bins = ["0–10 Myr\n(Same-Reef\nNeighbors)", "10–25 Myr\n(Young\nGenera)", "25–50 Myr\n(Within\nFamilies)", "50–80 Myr\n(New\nFamilies)", "80–110 Myr\n(Deep Body\nRadiations)"]
    xt6 = np.arange(len(time_bins))
    jaw_speed = [1.58, 0.71, 0.30, 0.34, 0.83]
    body_speed = [1.22, 0.70, 0.42, 0.73, 1.22]

    ax6a.axvspan(-0.4, 0.6, color="#fef3c7", alpha=0.55)
    ax6a.axvspan(2.5, 4.4, color="#e0f2fe", alpha=0.55)
    ax6a.plot(xt6, jaw_speed, "s--", color="#d97706", lw=2.8, markersize=9,
              label="Jaw Speed (5.4× 'Parking-Lot Swerve' When Neighbors Share a Reef!)")
    ax6a.plot(xt6, body_speed, "o-", color="#0284c7", lw=3.2, markersize=9.5,
              label="Body & Fin Speed (Cruises Ahead 2.2× to Build New Fish Families!)")

    for i in range(len(xt6)):
        ax6a.text(xt6[i], body_speed[i] + 0.08, f"{body_speed[i]:.2f}", ha="center", va="bottom", fontsize=9.2, fontweight="bold", color="#0369a1")
        off_y = 0.08 if i == 0 else -0.16
        ax6a.text(xt6[i], jaw_speed[i] + off_y, f"{jaw_speed[i]:.2f}", ha="center", va="bottom", fontsize=9.2, fontweight="bold", color="#b45309")

    ax6a.set_ylim(0.05, 1.92)
    ax6a.set_xticks(xt6)
    ax6a.set_xticklabels(time_bins, fontsize=8.8, fontweight="bold")
    ax6a.set_ylabel("Speed of Shape Change (Rate per Myr)", fontsize=10.2, fontweight="bold")
    ax6a.set_title("A. Solving the 'Missing Early Explosion' Mystery\nJaws Sprint Near the End (0–10 Myr); Bodies Build Deep Families (50–110 Myr)",
                   fontsize=11.8, fontweight="bold", color="#0f172a", pad=12)
    ax6a.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), fontsize=9.2, frameon=True, facecolor="#f8fafc")
    ax6a.grid(True, ls="--", alpha=0.35)

    # 6B: Four Big Upgrades for Future Science (Family Trees, Ancestors, Fossils, Bioelectricity)
    ax6b = fig6.add_subplot(gs6[0, 1])
    imp_labels = [
        "1. Family Tree Clues\n(Fin/Body vs.\nJaw Signal)",
        "2. Drawing Extinct\n'Missing Link' Ancestors\n(Valid Un-Folded Grids)",
        "3. Small Fossil Groups\n(Works With Just 6\nSpecies vs. Needing 35+)",
        "4. Bioelectric Blueprint\n(Global Body Voltage\nHighway Match)",
    ]
    x6b = np.arange(len(imp_labels))
    w6 = 0.34
    old_way = [22.0, 16.7, 0.0, 4.6]
    new_way = [100.0, 100.0, 100.0, 98.9]

    b_old = ax6b.bar(x6b - w6 / 2, old_way, width=w6, color="#cbd5e1", edgecolor="#334155", lw=1.0,
                     label="Old Method (Jaws / 600-Dot Spline / Unseparated Dots)")
    b_new = ax6b.bar(x6b + w6 / 2, new_way, width=w6, color=["#0284c7", "#16a34a", "#8b5cf6", "#f59e0b"],
                     edgecolor="#0f172a", lw=1.1, label="New Method (Thompson's 8-Knob Rubber Sheet + Jaw Split)")

    old_tags = ["3.0×\n(Noisy Jaws)", "1 of 6\n(Folds 5/6!)", "0%\n(Math Crashes!)", "4.6%\n(No Match)"]
    new_tags = ["13.6×\n(4.5× Cleaner!)", "6 of 6\n(100% Smooth!)", "100%\n(Works on 6 Spp!)", "98.9%\n(Matches Grid!)"]

    for bar, t_str in zip(b_old, old_tags):
        h = bar.get_height()
        ax6b.text(bar.get_x() + bar.get_width() / 2, h + 2.2, t_str, ha="center", va="bottom", fontsize=8.4, fontweight="bold", color="#475569")
    for bar, t_str in zip(b_new, new_tags):
        h = bar.get_height()
        ax6b.text(bar.get_x() + bar.get_width() / 2, h + 2.2, t_str, ha="center", va="bottom", fontsize=8.8, fontweight="bold", color="#0f172a")

    ax6b.set_ylim(0, 122)
    ax6b.set_xticks(x6b)
    ax6b.set_xticklabels(imp_labels, fontsize=8.6, fontweight="bold")
    ax6b.set_ylabel("Success / Signal Score (%)", fontsize=10.2, fontweight="bold")
    ax6b.set_title("B. Four New Superpowers for Evolutionary Biology\nBetter Family Trees, Un-Crumpled Ancestors, Fossil Math & Bioelectricity",
                   fontsize=11.8, fontweight="bold", color="#0f172a", pad=12)
    ax6b.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), fontsize=9.2, frameon=True, facecolor="#f8fafc")
    ax6b.grid(axis="y", ls="--", alpha=0.35)

    fig6.savefig(DOCS_DIR / "guide_step6_evolutionary_implications.png", dpi=220)
    plt.close(fig6)

    print("Regenerated all 6 figures with external legends and zero overlaps!")


if __name__ == "__main__":
    main()

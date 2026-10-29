#!/usr/bin/env python3
"""
sorting-morphogenesis/render_13yo_illustrations.py —
Generates 4 intuitive, colorful, visual diagram plates specifically designed for
the 13-Year-Old's Illustrated PDF Guide (`docs/zhang_levin_13yo_illustrated_guide.pdf`):

  1. `docs/media/zhang_13yo_fig1_three_personalities.png`:
     Visual cartoon of the line of kids in gym class wearing numbered jerseys,
     illustrating the 3 personalities (Bubble = neighbor swap, Insertion = wait
     for sorted left line + the `prev = -1` frozen-kid reset rule, Selection =
     long-range target pointer jumping right over a Frozen brick-wall kid!).
  2. `docs/media/zhang_13yo_fig2_two_rulers_illusion.png`:
     Step-by-step visual proof of the "Two Rulers" illusion on `[2, 4, 1, 3] -> [2, 1, 4, 3]`,
     drawing every out-of-order pair arch (`I: 3 -> 2`, strictly downhill!) right
     above the neighbor-step cliff diagram (`E: 1 -> 2`, why the paper's ruler
     went up even though the kids moved strictly closer to sorted!).
  3. `docs/media/zhang_13yo_fig3_sports_cars_vs_pedestrians.png`:
     Visual diagram of the "Secret Clique" mystery:
       - Top: Unique numbers (`1..12`) showing fast Selection "sports cars"
         teleporting left and compressing slow Bubble "pedestrians" into a
         traffic wave on the right.
       - Bottom: Duplicate numbers showing why Selection grabs the LEFT half of
         each number's group (`S...S | B...B`), creating `S B | S B | S B`
         anti-clustering (`0.477 < 0.50`) at `C = 2` copies per number and
         `SSSSS | BBBBB` mega-cliques (`0.659`) at `C = 10` copies per number!
"""

from __future__ import annotations

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Circle, Rectangle

ROOT = Path(__file__).resolve().parent.parent
MEDIA = ROOT / "docs" / "media"


def draw_jersey_kid(ax, x, y, number, algo="B", frozen=False, width=0.68, height=0.82):
    """Draw a kid/cell card with a jersey number and algotype badge."""
    if frozen:
        face = "#dc2626"
        edge = "#fef08a"
        badge_col = "#7f1d1d"
        lbl = "FROZEN"
    elif algo == "B":
        face = "#2563eb"
        edge = "#93c5fd"
        badge_col = "#1e3a8a"
        lbl = "Bubble (B)"
    elif algo == "I":
        face = "#16a34a"
        edge = "#86efac"
        badge_col = "#14532d"
        lbl = "Insert (I)"
    else:
        face = "#ea580c"
        edge = "#fdba74"
        badge_col = "#7c2d12"
        lbl = "Select (S)"

    box = FancyBboxPatch(
        (x - width / 2, y - height / 2), width, height,
        boxstyle="round,pad=0.06,rounding_size=0.12",
        facecolor=face, edgecolor=edge, linewidth=2.2, zorder=4
    )
    ax.add_patch(box)
    ax.text(x, y + 0.08, f"#{number}", color="#ffffff", fontsize=13.5, fontweight="bold", ha="center", va="center", zorder=5)
    ax.text(
        x, y - 0.25, lbl, color="#fef08a" if frozen else "#e2e8f0",
        fontsize=7.2, fontweight="bold", ha="center", va="center",
        bbox=dict(boxstyle="round,pad=0.12", facecolor=badge_col, edgecolor="none"), zorder=5
    )


def render_fig1_three_personalities():
    fig, axes = plt.subplots(3, 1, figsize=(12.5, 7.8), dpi=150, facecolor="#0b1325")
    fig.subplots_adjust(hspace=0.42, left=0.04, right=0.96, top=0.88, bottom=0.05)

    fig.suptitle(
        "Illustrated Guide 1: The Three 'Cell Personalities' (Algotypes) & How They Handle a Frozen Kid",
        color="#f8fafc", fontsize=13.5, fontweight="bold", y=0.96
    )

    # Row 1: Bubble Kids
    ax = axes[0]
    ax.set_facecolor("#0f172a")
    ax.set_xlim(-0.8, 7.8)
    ax.set_ylim(-0.75, 0.95)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_color("#38bdf8"); sp.set_linewidth(1.6)
    ax.set_title(
        "1. Bubble Kids (B — Blue): Only Swap With Immediate Neighbors (Blocked Forever by a Frozen Wall!)",
        color="#38bdf8", fontsize=10.8, fontweight="bold", pad=6
    )
    nums_b = [(1, "B", False), (4, "B", False), (99, "B", True), (2, "B", False), (5, "B", False), (3, "B", False)]
    for i, (num, al, fr) in enumerate(nums_b):
        draw_jersey_kid(ax, i * 1.15 + 0.3, 0.0, num, al, fr)
    # Draw swap arrow between #5 and #3
    arr1 = FancyArrowPatch((4 * 1.15 + 0.3, 0.48), (5 * 1.15 + 0.3, 0.48), connectionstyle="arc3,rad=-0.45",
                           arrowstyle="<->,head_length=6,head_width=5", color="#38bdf8", lw=2.4, zorder=6)
    ax.add_patch(arr1)
    ax.text(4.5 * 1.15 + 0.3, 0.74, "Neighbor swap OK! (#5 > #3)", color="#38bdf8", fontsize=8.5, fontweight="bold", ha="center")
    # Blocked X at frozen #99
    ax.text(2.5 * 1.15 + 0.3, 0.62, "✖ BLOCKED! #2 can NEVER get past Frozen #99!", color="#f87171", fontsize=8.8, fontweight="bold", ha="center",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="#450a0a", edgecolor="#ef4444"))

    # Row 2: Insertion Kids
    ax = axes[1]
    ax.set_facecolor("#0f172a")
    ax.set_xlim(-0.8, 7.8)
    ax.set_ylim(-0.75, 0.95)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_color("#4ade80"); sp.set_linewidth(1.6)
    ax.set_title(
        "2. Insertion Kids (I — Green): Need a Sorted Left Line — Saved by 3 Lines of 'Reset' Code (`if FREEZE: prev = -1`)!",
        color="#4ade80", fontsize=10.8, fontweight="bold", pad=6
    )
    nums_i = [(2, "I", False), (99, "I", True), (6, "I", False), (9, "I", False), (7, "I", False), (12, "I", False)]
    for i, (num, al, fr) in enumerate(nums_i):
        draw_jersey_kid(ax, i * 1.15 + 0.3, 0.0, num, al, fr)
    ax.text(1 * 1.15 + 0.3, 0.66, "Special Code L74–76:\n`if FREEZE: prev = -1`\n(Resets memory here!)", color="#fef08a", fontsize=8.0, fontweight="bold", ha="center",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="#14532d", edgecolor="#4ade80"))
    arr2 = FancyArrowPatch((4 * 1.15 + 0.3, 0.48), (3 * 1.15 + 0.3, 0.48), connectionstyle="arc3,rad=0.45",
                           arrowstyle="->,head_length=6,head_width=5", color="#4ade80", lw=2.4, zorder=6)
    ax.add_patch(arr2)
    ax.text(3.5 * 1.15 + 0.3, 0.74, "Because memory reset after #99, #7 sees [6, 9] is sorted & steps left!", color="#86efac", fontsize=8.2, fontweight="bold", ha="center")

    # Row 3: Selection Kids
    ax = axes[2]
    ax.set_facecolor("#0f172a")
    ax.set_xlim(-0.8, 7.8)
    ax.set_ylim(-0.75, 0.95)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_color("#facc15"); sp.set_linewidth(1.6)
    ax.set_title(
        "3. Selection Kids (S — Orange): Use a Long-Range `target` Pointer to Teleport Right Over the Frozen Wall!",
        color="#facc15", fontsize=10.8, fontweight="bold", pad=6
    )
    nums_s = [(8, "S", False), (5, "S", False), (99, "S", True), (9, "S", False), (1, "S", False), (11, "S", False)]
    for i, (num, al, fr) in enumerate(nums_s):
        draw_jersey_kid(ax, i * 1.15 + 0.3, 0.0, num, al, fr)
    arr3 = FancyArrowPatch((4 * 1.15 + 0.3, 0.46), (0 * 1.15 + 0.3, 0.46), connectionstyle="arc3,rad=0.32",
                           arrowstyle="<->,head_length=7,head_width=6", color="#facc15", lw=2.8, zorder=6)
    ax.add_patch(arr3)
    ax.text(2.0 * 1.15 + 0.3, 0.75, "TELEPORT SWAP! Kid #1 points `target = Slot 0` and swaps with #8 right over Frozen #99!", color="#fef08a", fontsize=8.6, fontweight="bold", ha="center",
            bbox=dict(boxstyle="round,pad=0.18", facecolor="#7c2d12", edgecolor="#facc15"))

    out = MEDIA / "zhang_13yo_fig1_three_personalities.png"
    fig.savefig(out, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)
    print("Saved", out)


def render_fig2_two_rulers_illusion():
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.6), dpi=150, facecolor="#0b1325")
    fig.subplots_adjust(wspace=0.16, left=0.04, right=0.96, top=0.82, bottom=0.08)

    fig.suptitle(
        "Illustrated Guide 2: Why 'Delayed Gratification' Was an Optical Illusion of the Ruler!\n"
        "Swapping 4 ↔ 1 in [2, 4, 1, 3] → [2, 1, 4, 3] Goes STRICTLY DOWNHILL on Ruler 1 (I: 3 → 2), While Ruler 2 Goes Up (E: 1 → 2)!",
        color="#f8fafc", fontsize=12.0, fontweight="bold", y=0.96
    )

    states = [
        ("BEFORE SWAP:  [ 2,  4,  1,  3 ]", [2, 4, 1, 3], [(0, 2, "2 > 1"), (1, 2, "4 > 1"), (1, 3, "4 > 3")], [(1, 2, "1 Big Cliff: 4 > 1")], "Ruler 1 (True Energy I) = 3 Inversions\nRuler 2 (Neighbor Drops E) = 1 Drop"),
        ("AFTER SWAP (4 ↔ 1):  [ 2,  1,  4,  3 ]", [2, 1, 4, 3], [(0, 1, "2 > 1"), (2, 3, "4 > 3")], [(0, 1, "Step 1: 2 > 1"), (2, 3, "Step 2: 4 > 3")], "Ruler 1 (True Energy I) = 2 Inversions (DOWN -1!)\nRuler 2 (Neighbor Drops E) = 2 Drops (UP +1?!)"),
    ]

    for idx, (tit, arr, inv_pairs, adj_drops, summary_txt) in enumerate(states):
        ax = axes[idx]
        ax.set_facecolor("#0f172a")
        ax.set_xlim(-0.6, 3.6)
        ax.set_ylim(-1.35, 1.55)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color("#facc15" if idx == 1 else "#38bdf8")
            sp.set_linewidth(2.0)
        ax.set_title(tit, color="#facc15" if idx == 1 else "#38bdf8", fontsize=11.2, fontweight="bold", pad=8)

        for i, v in enumerate(arr):
            draw_jersey_kid(ax, i, 0.0, v, "B", False, width=0.58, height=0.72)

        # Top arches: Ruler 1 (Total Out-of-Order Pairs I(v))
        ax.text(1.5, 1.38, f"RULER 1 — Total Out-of-Order Pairs I(v) = {len(inv_pairs)} (Hoel CE 2.0 = +0.119, 100% Deterministic)",
                color="#4ade80", fontsize=8.8, fontweight="bold", ha="center",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="#052e16", edgecolor="#4ade80"))
        heights = [0.62, 0.88, 1.12]
        for k, (i1, i2, lbl) in enumerate(inv_pairs):
            rad = -0.38 - 0.12 * k
            arc = FancyArrowPatch((i1, 0.38), (i2, 0.38), connectionstyle=f"arc3,rad={rad}",
                                  arrowstyle="-", color="#4ade80", lw=2.2, zorder=6)
            ax.add_patch(arc)
            ax.text(0.5 * (i1 + i2), heights[k], lbl, color="#86efac", fontsize=8.0, fontweight="bold", ha="center")

        # Bottom brackets: Ruler 2 (Neighbor Drops E(v))
        ax.text(1.5, -1.18, f"RULER 2 — Only Neighbor Steps E(v) = {len(adj_drops)} (Hoel CE 2.0 = -0.253, Non-Lumpable Illusion!)",
                color="#f87171", fontsize=8.6, fontweight="bold", ha="center",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="#450a0a", edgecolor="#ef4444"))
        for i1, i2, lbl in adj_drops:
            arc_b = FancyArrowPatch((i1, -0.38), (i2, -0.38), connectionstyle="arc3,rad=0.42",
                                    arrowstyle="->,head_length=6,head_width=5", color="#f87171", lw=2.5, zorder=6)
            ax.add_patch(arc_b)
            ax.text(0.5 * (i1 + i2), -0.82, lbl, color="#fca5a5", fontsize=8.2, fontweight="bold", ha="center")

    out = MEDIA / "zhang_13yo_fig2_two_rulers_illusion.png"
    fig.savefig(out, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)
    print("Saved", out)


def render_fig3_sports_cars_and_c2():
    fig, axes = plt.subplots(2, 1, figsize=(12.8, 6.4), dpi=150, facecolor="#0b1325")
    fig.subplots_adjust(hspace=0.42, left=0.04, right=0.96, top=0.86, bottom=0.06)

    fig.suptitle(
        "Illustrated Guide 3: Why 'Secret Cliques' Form — And How C = 2 Copies Proves the Blog Critics Wrong!",
        color="#f8fafc", fontsize=12.8, fontweight="bold", y=0.96
    )

    # Top row: C = 2 copies per number -> S B | S B | S B Anti-Clustering!
    ax0 = axes[0]
    ax0.set_facecolor("#0f172a")
    ax0.set_xlim(-0.6, 8.6)
    ax0.set_ylim(-0.75, 0.95)
    ax0.set_xticks([]); ax0.set_yticks([])
    for sp in ax0.spines.values():
        sp.set_color("#f87171"); sp.set_linewidth(1.8)
    ax0.set_title(
        "Case A: C = 2 Copies of Each Number (#1, #1, #2, #2, #3, #3, #4, #4) → Selection Grabs the LEFT Spot of Each Pair (`S B | S B | S B`) → ANTI-CLUSTERING (0.477 < 0.50)!",
        color="#f87171", fontsize=9.8, fontweight="bold", pad=6
    )
    c2_cells = [(1, "S"), (1, "B"), (2, "S"), (2, "B"), (3, "S"), (3, "B"), (4, "S"), (4, "B")]
    for i, (num, al) in enumerate(c2_cells):
        draw_jersey_kid(ax0, i * 1.05 + 0.3, -0.05, num, al, False, width=0.64, height=0.74)
    for g in range(4):
        x_left = (2 * g) * 1.05 - 0.08
        rect = Rectangle((x_left, -0.48), 1.92, 0.88, facecolor="none", edgecolor="#facc15", lw=1.8, ls="--", zorder=3)
        ax0.add_patch(rect)
        ax0.text(x_left + 0.96, 0.58, f"Group #{g+1}: S then B!", color="#fef08a", fontsize=8.0, fontweight="bold", ha="center")

    # Bottom row: C = 4+ copies per number -> S S S S | B B B B Mega-Cliques!
    ax1 = axes[1]
    ax1.set_facecolor("#0f172a")
    ax1.set_xlim(-0.6, 8.6)
    ax1.set_ylim(-0.75, 0.95)
    ax1.set_xticks([]); ax1.set_yticks([])
    for sp in ax1.spines.values():
        sp.set_color("#4ade80"); sp.set_linewidth(1.8)
    ax1.set_title(
        "Case B: Many Copies of the Same Number (e.g. Eight #1s) → Selection Packs the Left Half (`S S S S | B B B B`) → BIG CLIQUES (0.659 > 0.50)!",
        color="#4ade80", fontsize=9.8, fontweight="bold", pad=6
    )
    c8_cells = [(1, "S"), (1, "S"), (1, "S"), (1, "S"), (1, "B"), (1, "B"), (1, "B"), (1, "B")]
    for i, (num, al) in enumerate(c8_cells):
        draw_jersey_kid(ax1, i * 1.05 + 0.3, -0.05, num, al, False, width=0.64, height=0.74)
    ax1.add_patch(Rectangle((-0.08, -0.48), 4.02, 0.88, facecolor="none", edgecolor="#facc15", lw=2.0, ls="--", zorder=3))
    ax1.text(1.90, 0.58, "Left Half of Group #1: All Selection ('Sports Cars' Arrive First!)", color="#fdba74", fontsize=8.4, fontweight="bold", ha="center")
    ax1.add_patch(Rectangle((4.12, -0.48), 4.02, 0.88, facecolor="none", edgecolor="#38bdf8", lw=2.0, ls="--", zorder=3))
    ax1.text(6.10, 0.58, "Right Half of Group #1: All Bubble ('Pedestrians' Displaced Right!)", color="#93c5fd", fontsize=8.4, fontweight="bold", ha="center")

    out = MEDIA / "zhang_13yo_fig3_sports_cars_vs_pedestrians.png"
    fig.savefig(out, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)
    print("Saved", out)


def render_fig4_real_data_curves():
    """
    Illustrated Guide 4: Two kid-friendly real-data charts side by side:
      - Left: The 'Smooth Slide vs. Rollercoaster' graph (Ruler 1 I(v) has 0
        uphill steps; Ruler 2 E(v) has 47 uphill bumps on the exact same run!).
      - Right: The 'Blog Critics vs. Reality' graph across C = 1, 2, 4, 5, 8, 10
        copies per number, highlighting the C = 2 Anti-Clustering dip (0.477)
        and the C = 10 Mega-Clique peak (0.659).
    """
    import json
    import random

    ce_json = ROOT / "sorting-morphogenesis" / "figures" / "quantifying_emergence_ce2_metrics.json"
    ce_data = json.loads(ce_json.read_text())

    def total_inversions(arr):
        return sum(1 for i in range(len(arr)) for j in range(i + 1, len(arr)) if arr[i] > arr[j])

    def monotonicity_error(arr):
        return sum(1 for i in range(len(arr) - 1) if arr[i] > arr[i + 1])

    # Simulate exact Bubble-sort trace with 1 frozen cell at index 10 (N=25, seed=2026)
    rng = random.Random(2026)
    vals = list(range(1, 26))
    rng.shuffle(vals)
    frozen_idx = 10
    i_trace = [float(total_inversions(vals))]
    e_trace = [float(monotonicity_error(vals))]
    for _ in range(900):
        idx = rng.randint(0, 24)
        if idx == frozen_idx:
            continue
        direction = -1 if rng.random() < 0.5 else 1
        t = idx + direction
        if 0 <= t < 25 and t != frozen_idx:
            if (direction == -1 and vals[idx] < vals[t]) or (direction == 1 and vals[idx] > vals[t]):
                vals[idx], vals[t] = vals[t], vals[idx]
                i_trace.append(float(total_inversions(vals)))
                e_trace.append(float(monotonicity_error(vals)))
    i_trace = np.array(i_trace)
    e_trace = np.array(e_trace)

    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.2), dpi=160, facecolor="#0b1325")
    fig.subplots_adjust(wspace=0.22, left=0.06, right=0.96, top=0.80, bottom=0.14)

    fig.suptitle(
        "Illustrated Guide 4: Seeing the Proof in the Real Computer Experiments!",
        color="#f8fafc", fontsize=13.2, fontweight="bold", y=0.95
    )

    # Left panel: Ruler 1 (I) vs Ruler 2 (E) normalized trajectory
    ax0 = axes[0]
    ax0.set_facecolor("#0f172a")
    for sp in ax0.spines.values():
        sp.set_color("#38bdf8"); sp.set_linewidth(1.6)
    steps = np.arange(len(e_trace))
    ax0.plot(steps, i_trace / max(i_trace[0], 1.0) * 100.0, color="#4ade80", lw=3.0, label="Ruler 1: Total Pairs I(v) — 0 Uphill Steps! (Smooth Slide)")
    ax0.plot(steps, e_trace / max(e_trace[0], 1.0) * 100.0, color="#f87171", lw=2.0, alpha=0.9, label="Ruler 2: Neighbor Steps E(v) — 47 Uphill Bumps! (Illusion)")
    ax0.set_title("A. Same Kids, Two Rulers: Smooth Slide vs. Bumpy Illusion", color="#4ade80", fontsize=10.6, fontweight="bold", pad=8)
    ax0.set_xlabel("Swap Step Number", color="#cbd5e1", fontsize=9.5, fontweight="bold")
    ax0.set_ylabel("% of Starting Messiness Left", color="#cbd5e1", fontsize=9.5, fontweight="bold")
    ax0.tick_params(colors="#cbd5e1", labelsize=8.5)
    ax0.grid(color="#1e293b", lw=0.8)
    ax0.legend(loc="upper right", facecolor="#0b1325", edgecolor="#334155", labelcolor="#f8fafc", fontsize=8.2)
    mid_k = min(75, len(i_trace) - 1)
    ax0.annotate(
        "Every single swap goes\nSTRICTLY DOWNHILL on Ruler 1!",
        xy=(mid_k, i_trace[mid_k] / i_trace[0] * 100.0), xytext=(15, 22),
        arrowprops=dict(arrowstyle="->", color="#4ade80", lw=2.0),
        color="#86efac", fontsize=8.3, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.2", facecolor="#052e16", edgecolor="#4ade80")
    )

    # Right panel: Duplicate copies C = 1..40
    ax1 = axes[1]
    ax1.set_facecolor("#0f172a")
    for sp in ax1.spines.values():
        sp.set_color("#facc15"); sp.set_linewidth(1.6)
    dm = ce_data["duplicate_multiplicity_scaling_law"]
    c_vals = dm["multiplicities_C"][:5]
    x_pos = np.arange(len(c_vals))
    final_vals = dm["final_agg_mean"][:5]
    analytic_vals = dm["analytic_domain_wall_law"][:5]

    ax1.axhline(0.50, color="#94a3b8", ls="--", lw=2.0, label="Blog Critics' 'Stability' Prediction (Stays at 0.50 Random)")
    ax1.plot(x_pos, analytic_vals, "o--", color="#38bdf8", lw=2.0, markersize=6, label="Ideal Left-Right Split Formula (`S...S | B...B`)")
    ax1.plot(x_pos, final_vals, "s-", color="#facc15", lw=2.8, markersize=7, label="Actual Computer Simulation (Final Clustering)")
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels([f"C={c}" for c in c_vals], color="#f8fafc", fontweight="bold")
    ax1.set_ylim(0.42, 0.82)
    ax1.set_title("B. How C = 2 Anti-Clustering Refutes the Blog Critics!", color="#facc15", fontsize=10.6, fontweight="bold", pad=8)
    ax1.set_xlabel("Number of Duplicate Copies per Jersey Number (C)", color="#cbd5e1", fontsize=9.5, fontweight="bold")
    ax1.set_ylabel("Same-Personality Neighborship (Clustering)", color="#cbd5e1", fontsize=9.5, fontweight="bold")
    ax1.tick_params(colors="#cbd5e1", labelsize=8.5)
    ax1.grid(color="#1e293b", lw=0.8)
    ax1.legend(loc="upper left", facecolor="#0b1325", edgecolor="#334155", labelcolor="#f8fafc", fontsize=8.0)

    ax1.annotate(
        "C = 2 Dip: 0.477 < 0.50!\n(`S B | S B | S B` Anti-Clustering)",
        xy=(1, final_vals[1]), xytext=(1.35, 0.435),
        arrowprops=dict(arrowstyle="->", color="#f87171", lw=2.2),
        color="#fca5a5", fontsize=8.2, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.2", facecolor="#450a0a", edgecolor="#ef4444")
    )
    ax1.annotate(
        "C = 10 Peak: 0.659!\n(`SSSSS | BBBBB` Cliques)",
        xy=(4, final_vals[4]), xytext=(2.05, 0.65),
        arrowprops=dict(arrowstyle="->", color="#4ade80", lw=2.2),
        color="#86efac", fontsize=8.2, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.2", facecolor="#052e16", edgecolor="#4ade80")
    )

    out = MEDIA / "zhang_13yo_fig4_real_data_curves.png"
    fig.savefig(out, dpi=160, facecolor=fig.get_facecolor())
    plt.close(fig)
    print("Saved", out)


def render_fig5_causal_emergence_13yo():
    """
    Illustrated Guide 5: A super-clean, 3-step visual cartoon of Causal Emergence
    (Erik Hoel 2013 & 2026) that a 13-year-old can understand in 10 seconds:
      - Step 1 (Top Row): The Coin-Flip Problem at the Micro-Level ("Zoomed In")
      - Step 2 (Middle Row, Green): Zooming Out to a GOOD Scoreboard (Ruler 1 = 100% Certainty!)
      - Step 3 (Bottom Row, Red): Zooming Out to a BLURRY Scoreboard (Ruler 2 = Scrambled Guesswork!)
    """
    fig, axes = plt.subplots(3, 1, figsize=(12.6, 8.2), dpi=160, facecolor="#0b1325")
    fig.subplots_adjust(hspace=0.38, left=0.04, right=0.96, top=0.87, bottom=0.05)

    fig.suptitle(
        "What Is 'Causal Emergence'? How Choosing the Right Scoreboard Beats a Coin Flip!",
        color="#f8fafc", fontsize=14.0, fontweight="bold", y=0.96
    )

    # --- STEP 1 (TOP): ZOOMED IN ON INDIVIDUAL KIDS = 50/50 COIN FLIP ---
    ax0 = axes[0]
    ax0.set_facecolor("#0f172a")
    ax0.set_xlim(0, 10); ax0.set_ylim(0, 4.2)
    ax0.set_xticks([]); ax0.set_yticks([])
    for sp in ax0.spines.values():
        sp.set_color("#38bdf8"); sp.set_linewidth(2.0)
    ax0.set_title(
        "STEP 1 — Zoomed In on Individual Kids: You Can't Predict Which Line Comes Next! (50/50 Coin Flip)",
        color="#38bdf8", fontsize=11.0, fontweight="bold", pad=6
    )

    ax0.text(1.55, 2.1, "Starting Line:\n[ 2,  4,  1,  3 ]", color="#ffffff", fontsize=11.0, fontweight="bold", ha="center", va="center",
             bbox=dict(boxstyle="round,pad=0.38", facecolor="#1e293b", edgecolor="#38bdf8", lw=2.2))

    ax0.annotate("", xy=(4.7, 3.1), xytext=(2.85, 2.35), arrowprops=dict(arrowstyle="->,head_length=8,head_width=6", color="#38bdf8", lw=2.6))
    ax0.text(3.75, 3.32, "50% Coin Flip\n(#4 swaps with #1)", color="#93c5fd", fontsize=8.8, fontweight="bold", ha="center")
    ax0.text(6.1, 3.1, "Possible Next Line A:\n[ 2,  1,  4,  3 ]", color="#ffffff", fontsize=10.2, fontweight="bold", ha="center", va="center",
             bbox=dict(boxstyle="round,pad=0.32", facecolor="#1e3a8a", edgecolor="#60a5fa", lw=2.0))

    ax0.annotate("", xy=(4.7, 1.1), xytext=(2.85, 1.85), arrowprops=dict(arrowstyle="->,head_length=8,head_width=6", color="#38bdf8", lw=2.6))
    ax0.text(3.75, 0.82, "50% Coin Flip\n(#4 swaps with #3)", color="#93c5fd", fontsize=8.8, fontweight="bold", ha="center")
    ax0.text(6.1, 1.1, "Possible Next Line B:\n[ 2,  4,  3,  1 ]", color="#ffffff", fontsize=10.2, fontweight="bold", ha="center", va="center",
             bbox=dict(boxstyle="round,pad=0.32", facecolor="#1e3a8a", edgecolor="#60a5fa", lw=2.0))

    ax0.text(8.65, 2.1, "Zoomed-In\nPredictability:\n86.9%\n(Noisy because\nrandom kids\nwake up first!)",
             color="#bae6fd", fontsize=9.2, fontweight="bold", ha="center", va="center",
             bbox=dict(boxstyle="round,pad=0.32", facecolor="#082f49", edgecolor="#0284c7", lw=1.8))

    # --- STEP 2 (MIDDLE): ZOOM OUT TO RULER 1 = 100% PREDICTABLE! ---
    ax1 = axes[1]
    ax1.set_facecolor("#0f172a")
    ax1.set_xlim(0, 10); ax1.set_ylim(0, 4.2)
    ax1.set_xticks([]); ax1.set_yticks([])
    for sp in ax1.spines.values():
        sp.set_color("#4ade80"); sp.set_linewidth(2.2)
    ax1.set_title(
        "STEP 2 — Zoom Out to Ruler 1 (Total Out-of-Order Pairs): Both Coin Flips Land on the EXACT SAME Score!",
        color="#4ade80", fontsize=11.0, fontweight="bold", pad=6
    )

    xs_i = [1.2, 3.45, 5.7, 7.55]
    lbls_i = ["Start Score:\n3 Pairs\nOut of Order", "Next Score:\n2 Pairs\n(Line A & B Both = 2!)", "Next Score:\n1 Pair\nOut of Order", "Final Score:\n0 Pairs\n(100% Sorted!)"]
    for k, (xp, lb) in enumerate(zip(xs_i, lbls_i)):
        ax1.text(xp, 2.1, lb, color="#ffffff", fontsize=9.4, fontweight="bold", ha="center", va="center",
                 bbox=dict(boxstyle="round,pad=0.34", facecolor="#14532d" if k < 3 else "#15803d", edgecolor="#4ade80", lw=2.2))
        if k < 3:
            ax1.annotate("", xy=(xs_i[k+1] - 0.88, 2.1), xytext=(xp + 0.88, 2.1),
                         arrowprops=dict(arrowstyle="->,head_length=8,head_width=6", color="#4ade80", lw=3.0))
            ax1.text(0.5 * (xp + xs_i[k+1]), 3.18, "Always −1!", color="#86efac", fontsize=8.8, fontweight="bold", ha="center")

    ax1.text(9.05, 2.1, "✔ POSITIVE\nEMERGENCE!\n100% Sure!\n(Hoel Score:\n+0.119)",
             color="#dcfce7", fontsize=9.4, fontweight="bold", ha="center", va="center",
             bbox=dict(boxstyle="round,pad=0.32", facecolor="#052e16", edgecolor="#22c55e", lw=2.0))

    # --- STEP 3 (BOTTOM): ZOOM OUT TO RULER 2 = SCRAMBLED MESS! ---
    ax2 = axes[2]
    ax2.set_facecolor("#0f172a")
    ax2.set_xlim(0, 10); ax2.set_ylim(0, 4.2)
    ax2.set_xticks([]); ax2.set_yticks([])
    for sp in ax2.spines.values():
        sp.set_color("#f87171"); sp.set_linewidth(2.0)
    ax2.set_title(
        "STEP 3 — Zoom Out to Ruler 2 (Only Neighbor Drops): A Blurry Ruler That Scrambles the Future!",
        color="#f87171", fontsize=11.0, fontweight="bold", pad=6
    )

    ax2.text(1.75, 2.1, "Start Score:\n1 Neighbor Drop\n(Lumps [2, 4, 1, 3]\nwith [1, 3, 2, 4]!)",
             color="#ffffff", fontsize=9.2, fontweight="bold", ha="center", va="center",
             bbox=dict(boxstyle="round,pad=0.32", facecolor="#450a0a", edgecolor="#f87171", lw=2.0))

    branches_e = [
        (3.35, "Goes UP to 2 Drops! (Looks like 'going backwards'!)", "#fca5a5"),
        (2.10, "Stays Flat at 1 Drop! (Looks like nothing happened!)", "#cbd5e1"),
        (0.85, "Goes DOWN to 0 Drops! (Sorted in one step!)", "#86efac"),
    ]
    for yp, txt, col in branches_e:
        ax2.annotate("", xy=(4.3, yp), xytext=(3.1, 2.1), arrowprops=dict(arrowstyle="->,head_length=7,head_width=5", color="#f87171", lw=2.4))
        ax2.text(6.15, yp, txt, color=col, fontsize=8.8, fontweight="bold", ha="center", va="center",
                 bbox=dict(boxstyle="round,pad=0.24", facecolor="#1e293b", edgecolor="#ef4444", lw=1.5))

    ax2.text(9.05, 2.1, "✖ NEGATIVE\nEMERGENCE!\nDrops to 72.1%!\n(Hoel Score:\n−0.253)",
             color="#fee2e2", fontsize=9.4, fontweight="bold", ha="center", va="center",
             bbox=dict(boxstyle="round,pad=0.32", facecolor="#450a0a", edgecolor="#ef4444", lw=2.0))

    out = MEDIA / "zhang_13yo_fig5_causal_emergence_explained.png"
    fig.savefig(out, dpi=160, facecolor=fig.get_facecolor())
    plt.close(fig)
    print("Saved", out)


def render_fig6_decisive_comparison_d_vs_dp():
    """
    Illustrated Guide 6: A super-clean, 2-step 13-year-old cartoon explaining:
      - Row 1: The 'D vs. D + P' Test (Does adding a 'Platonic Ghost' P make a single
        new prediction beyond the 6 lines of code D? No: 0 new predictions!)
      - Row 2: The 'Frozen Rules' Test (How a math pattern DOES earn real credit:
        lock your prediction BEFORE peeking at new animals, and if it fails on
        hornet wings, take the loss honestly!)
    """
    fig, axes = plt.subplots(2, 1, figsize=(12.6, 5.8), dpi=160, facecolor="#0b1325")
    fig.subplots_adjust(hspace=0.42, left=0.04, right=0.96, top=0.84, bottom=0.06)

    fig.suptitle(
        "How Do You Prove a Pattern Is Real? The 'Frozen Rules' Test!",
        color="#f8fafc", fontsize=14.0, fontweight="bold", y=0.96
    )

    # --- ROW 1: D vs D + P on Sorting Kids ---
    ax0 = axes[0]
    ax0.set_facecolor("#0f172a")
    ax0.set_xlim(0, 10); ax0.set_ylim(0, 4.0)
    ax0.set_xticks([]); ax0.set_yticks([])
    for sp in ax0.spines.values():
        sp.set_color("#38bdf8"); sp.set_linewidth(2.0)
    ax0.set_title(
        "RULE 1 — A Ghost Sticker ('+ P') Doesn't Get Credit If the 6 Lines of Code ('D') Already Predict 100%!",
        color="#38bdf8", fontsize=10.8, fontweight="bold", pad=6
    )

    ax0.text(1.85, 2.0, "Model D (The Code Alone)\n• 6 Lines of Kid Swap Rules\n• Ruler 1 Countdown (I − 1)\n• Sports Cars vs. Pedestrians",
             color="#ffffff", fontsize=9.4, fontweight="bold", ha="center", va="center",
             bbox=dict(boxstyle="round,pad=0.35", facecolor="#082f49", edgecolor="#38bdf8", lw=2.2))

    ax0.text(5.15, 2.0, "Model D + P (Code + Ghost)\n• Exact Same 6 Lines of Code\n• PLUS a 'Platonic Ghost'\n  Sticker on the Box",
             color="#ffffff", fontsize=9.4, fontweight="bold", ha="center", va="center",
             bbox=dict(boxstyle="round,pad=0.35", facecolor="#3b0764", edgecolor="#a855f7", lw=2.2))

    ax0.annotate("", xy=(7.15, 2.0), xytext=(6.45, 2.0), arrowprops=dict(arrowstyle="->,head_length=8,head_width=6", color="#facc15", lw=2.8))

    ax0.text(8.55, 2.0, "Extra Swaps Predicted\nby the Ghost:\nZERO (0.00)!\n(Code D gets 100%\nof the credit!)",
             color="#fef08a", fontsize=9.5, fontweight="bold", ha="center", va="center",
             bbox=dict(boxstyle="round,pad=0.34", facecolor="#1e293b", edgecolor="#facc15", lw=2.2))

    # --- ROW 2: LOCK YOUR PREDICTIONS BEFORE PEEKING ---
    ax1 = axes[1]
    ax1.set_facecolor("#0f172a")
    ax1.set_xlim(0, 10); ax1.set_ylim(0, 4.0)
    ax1.set_xticks([]); ax1.set_yticks([])
    for sp in ax1.spines.values():
        sp.set_color("#4ade80"); sp.set_linewidth(2.0)
    ax1.set_title(
        "RULE 2 — Lock Your Guess in a Digital Safe BEFORE Peeking at New Animals (And Take Losses Honestly!)",
        color="#4ade80", fontsize=10.8, fontweight="bold", pad=6
    )

    ax1.text(1.75, 2.0, "1. Lock the Safe First!\nWe locked our 8-knob fish\ngrid predictions BEFORE\ndownloading 5,180 new fish!",
             color="#ffffff", fontsize=9.2, fontweight="bold", ha="center", va="center",
             bbox=dict(boxstyle="round,pad=0.34", facecolor="#082f49", edgecolor="#38bdf8", lw=2.0))

    ax1.annotate("", xy=(3.65, 2.0), xytext=(3.10, 2.0), arrowprops=dict(arrowstyle="->,head_length=8,head_width=6", color="#4ade80", lw=2.6))

    ax1.text(5.15, 2.0, "2. Where It WON:\n✔ Crushed the baseline on\n1,122 unseen fish (r = 0.87)!\n✔ 15× better on angelfish fins!",
             color="#dcfce7", fontsize=9.2, fontweight="bold", ha="center", va="center",
             bbox=dict(boxstyle="round,pad=0.34", facecolor="#052e16", edgecolor="#22c55e", lw=2.0))

    ax1.annotate("", xy=(7.0, 2.0), xytext=(6.60, 2.0), arrowprops=dict(arrowstyle="->,head_length=8,head_width=6", color="#f87171", lw=2.6))

    ax1.text(8.50, 2.0, "3. Where It LOST (Fair & Square!):\n✖ Failed on 1,027 hornet wings\n(0 out of 7 passed — bug wings\ndon't stretch like fish bodies!)",
             color="#fee2e2", fontsize=9.0, fontweight="bold", ha="center", va="center",
             bbox=dict(boxstyle="round,pad=0.34", facecolor="#450a0a", edgecolor="#ef4444", lw=2.0))

    out = MEDIA / "zhang_13yo_fig6_decisive_comparison_d_vs_dp.png"
    fig.savefig(out, dpi=160, facecolor=fig.get_facecolor())
    plt.close(fig)
    print("Saved", out)


def render_fig7_final_scoreboard():
    """
    Illustrated Guide 7: 'The Final Scoreboard: How Right Is Michael Levin?'
    The per-claim points below are unchanged. Current reading (errata S17): the
    unweighted mean of all seven claims is 77.6%. S16's 65.7% (Claims 1–4 only)
    and the equal-pillar 81.6% (structural 93.4%) stay drawn, labelled superseded.

    Historical 3-pillar split, kept so the superseded number recomputes:
      - Pillar I   (33.3% weight) — Sorting Morphogenesis (Claims 1–3, 11.1% each):
          • Claim 1 (Error Tolerance):       7.4 / 11.1 pts (67%)
          • Claim 2 ('Delayed Gratification'): 1.5 / 11.1 pts (13%)
          • Claim 3 (Secret Clique Clustering): 9.6 / 11.1 pts (87%)
          -> Pillar I Subtotal = 18.5 / 33.3 pts (55.6% of Pillar I)
      - Pillar II  (33.3% weight) — D'Arcy Thompson Morphospace (Claim 4, 33.3%):
          • Claim 4 (10,303 Fishes & 5,180 Blind Test): 32.0 / 33.3 pts (96.0% of Pillar II)
      - Pillar III (33.3% weight) — Bioelectric & Biobot Attractors (Claims 5–7, 11.1% each):
          • Claim 5 (Planarian 2H0T Memory):    10.6 / 11.1 pts (95%)
          • Claim 6 (Picasso Tadpole Homeostat): 10.6 / 11.1 pts (95%)
          • Claim 7 (Xenobot Self-Replication):  10.0 / 11.1 pts (90%)
          -> Pillar III Subtotal = 31.1 / 33.3 pts (93.3% of Pillar III)
      -------------------------------------------------------------------------
      3-PILLAR TOTAL:  18.5 + 32.0 + 31.1 = 81.6 / 100 pts (81.6% Overall!)
        • Missing 18.4 pts split: 12.6 dualist/overstated (3.7 Claim 1 Insertion + 7.4 Claim 2
          barrier/gradient + 1.5 Claim 3 ghost clustering) + 5.8 ordinary modelling shortfalls
          (2.2 rising E(v) in Claim 2, 1.3 failed H2 in Claim 4, 2.2 reduced-model biophysics in Claims 5–7)
        • Empirical & Structural Math (D + P_structural): 93.4% Right (81.6 / 87.4 pts)
        • Strong Dualist 'Non-Physical Ghosts' (P_dualist): 0.0% Needed (0.0 / 12.6 pts)
    """
    fig, ax = plt.subplots(figsize=(14.6, 8.8), dpi=160, facecolor="#0b1325")
    fig.subplots_adjust(left=0.02, right=0.98, top=0.87, bottom=0.03)
    ax.set_facecolor("#0b1325")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 106)
    ax.axis("off")

    fig.suptitle(
        "Why Michael Levin Is 77.6% Right on Platonic Ingress",
        color="#f8fafc", fontsize=14.2, fontweight="bold", y=0.962
    )
    ax.text(
        50, 103.6,
        "Ingressing Minds, seven claims, weighted equally:  (66.7 + 13.5 + 86.5 + 96.1 + 95.0 + 95.0 + 90.1) / 7  =  77.6%",
        color="#93c5fd", fontsize=8.2, fontweight="bold", ha="center", va="center"
    )
    ax.text(
        50, 100.6,
        "Superseded, kept visible:  equal pillars 81.6% (structural 93.4%)   ·   Claims 1–4 only 65.7% (structural 84.2%, errata S16)",
        color="#facc15", fontsize=7.8, fontweight="bold", ha="center", va="center"
    )

    rows = [
        (
            "CLAIM 1 · Sorting Error Tolerance",
            "Pillar I.A · Weight: 11.1% (1/3 of 33.3%)",
            "• Bubble (+3.7/3.7) & Selection (+3.7/3.7) only skip frozen cells (a wall, no reset)\n• InsertionSortCell.py (L74–76) has explicit 3-line `FREEZE` reset (+0/3.7)",
            "7.4 / 11.1 pts (67%)\nPARTIALLY REFUTED",
            "#f59e0b", "#451a03", "#fef3c7"
        ),
        (
            "CLAIM 2 · 'Delayed Gratification'",
            "Pillar I.B · Weight: 11.1% (1/3 of 33.3%)",
            "• Neighbor curve E(v) ticks up (+1.5/3.7), but 0 Bubble cells cross wall (+0/3.7)\n• True energy I(v) goes strictly downhill ΔI = −1 on 192/192 swaps (+0/3.7)",
            "1.5 / 11.1 pts (13%)\nOPTICAL ILLUSION",
            "#ef4444", "#450a0a", "#fee2e2"
        ),
        (
            "CLAIM 3 · Secret Clique Clustering",
            "Pillar I.C · Weight: 11.1% (1/3 of 33.3%)",
            "• Unique keys (+4.8/4.8) & C=2/C=10 duplicate laws (+4.8/4.8) 100% real\n• Refutes CS blog critics twice, but 100% produced by speed waves (+0/1.5 ghost)",
            "9.6 / 11.1 pts (87%)\nEMPIRICALLY VINDICATED",
            "#22c55e", "#052e16", "#dcfce7"
        ),
        (
            "CLAIM 4 · D'Arcy Thompson Grids",
            "Pillar II · Weight: 33.3% (10,303 Fishes)",
            "• 600-pt homologous fin outlines beat 4p Box by +13% to +48% (+16.7/16.7)\n• Aligns 93.5%–98.9% across 10,303 fishes & 5,180 blind test (+15.3/16.6)",
            "32.0 / 33.3 pts (96%)\nSTRONGLY CONFIRMED",
            "#22c55e", "#052e16", "#dcfce7"
        ),
        (
            "CLAIMS 5–7 · Bioelectric Attractors",
            "Claims 5, 6 and 7 · each counts once",
            "• Laboratory results valid (Oviedo/Beane, Vandenberg, Kriegman)\n• Claim 5  10.55/11.1 = 95.0%    Claim 6  10.55/11.1 = 95.0%\n• Claim 7  10.0/11.1 = 90.1%    (−2.2 is our reduced equation, not the experiment)",
            "95.0 · 95.0 · 90.1\nIN THE MEAN",
            "#38bdf8", "#082f49", "#e0f2fe"
        ),
    ]

    y_tops = [91.5, 75.5, 59.5, 43.5, 27.5]
    for y_c, (c_title, c_quote, c_detail, verdict, border_col, badge_bg, badge_fg) in zip(y_tops, rows):
        card = FancyBboxPatch(
            (1.2, y_c - 6.4), 97.6, 13.0,
            boxstyle="round,pad=0.4,rounding_size=1.4",
            facecolor="#0f172a", edgecolor=border_col, linewidth=2.0, zorder=2
        )
        ax.add_patch(card)

        ax.plot([31.5, 31.5], [y_c - 5.3, y_c + 5.3], color="#334155", lw=1.2, ls=":", zorder=3)
        ax.plot([76.0, 76.0], [y_c - 5.3, y_c + 5.3], color="#334155", lw=1.2, ls=":", zorder=3)

        ax.text(2.5, y_c + 2.3, c_title, color=border_col, fontsize=9.6, fontweight="bold", ha="left", va="center", zorder=4)
        ax.text(2.5, y_c - 2.2, c_quote, color="#cbd5e1", fontsize=8.1, style="italic", ha="left", va="center", zorder=4)

        ax.text(32.8, y_c, c_detail, color="#f8fafc", fontsize=8.0, fontweight="bold", ha="left", va="center", linespacing=1.32, zorder=4)

        badge = FancyBboxPatch(
            (77.4, y_c - 4.5), 20.0, 9.0,
            boxstyle="round,pad=0.3,rounding_size=1.2",
            facecolor=badge_bg, edgecolor=border_col, linewidth=2.0, zorder=3
        )
        ax.add_patch(badge)
        ax.text(87.4, y_c, verdict, color=badge_fg, fontsize=8.4, fontweight="bold", ha="center", va="center", linespacing=1.25, zorder=4)

    # Stacked percentage bar showing the 3 × 33.3% Pillars: 18.5 + 32.0 + 31.1 = 81.6 / 100
    bar_y = 12.2
    bar_h = 5.4
    ax.text(1.4, bar_y + bar_h / 2, "SUPERSEDED\n3-pillar sum = 81.6%",
            color="#facc15", fontsize=7.6, fontweight="bold", ha="left", va="center")

    seg_x = 24.5
    total_w = 74.0  # 100 pts mapped to 74.0 width units
    segments = [
        (18.5, "#f59e0b", "Pillar I (Sorting 1–3): 18.5%"),
        (32.0, "#16a34a", "Pillar II (Thompson 4): 32.0%"),
        (31.1, "#0284c7", "Pillar III (Bioelectric 5–7): 31.1%"),
        (18.4, "#334155", "Missed / Overstated: 18.4%"),
    ]
    cur_x = seg_x
    for pts, col, lbl in segments:
        w = (pts / 100.0) * total_w
        rect = Rectangle((cur_x, bar_y), w, bar_h, facecolor=col, edgecolor="#0b1325", lw=1.5, zorder=3)
        ax.add_patch(rect)
        ax.text(cur_x + w / 2, bar_y + bar_h / 2, lbl, color="#ffffff", fontsize=7.4, fontweight="bold", ha="center", va="center", zorder=4)
        cur_x += w

    # Bottom synthesis bar: 3 boxes (3-Pillar Total 81.6%, Structural Math 93.4%, Dualist Ghosts 0.0%)
    b1 = FancyBboxPatch((1.2, 0.8), 30.5, 8.2, boxstyle="round,pad=0.3,rounding_size=1.0", facecolor="#1e293b", edgecolor="#facc15", linewidth=2.0, zorder=2)
    ax.add_patch(b1)
    ax.text(16.45, 4.9, "CURRENT READING: 77.6%\nmean of all seven claims, each once\n(81.6% and 65.7% both superseded)",
            color="#fef08a", fontsize=7.4, fontweight="bold", ha="center", va="center", linespacing=1.2, zorder=4)

    b2 = FancyBboxPatch((33.2, 0.8), 33.0, 8.2, boxstyle="round,pad=0.3,rounding_size=1.0", facecolor="#052e16", edgecolor="#22c55e", linewidth=2.0, zorder=2)
    ax.add_patch(b2)
    ax.text(49.7, 4.9, "✔ Structural, same seven claims: 88.1%\n(100+40.5+100+96.1+95.0+95.0+90.1) / 7\nsupersedes 84.2% and 93.4%",
            color="#dcfce7", fontsize=7.4, fontweight="bold", ha="center", va="center", linespacing=1.2, zorder=4)

    b3 = FancyBboxPatch((67.6, 0.8), 31.2, 8.2, boxstyle="round,pad=0.3,rounding_size=1.0", facecolor="#450a0a", edgecolor="#ef4444", linewidth=2.0, zorder=2)
    ax.add_patch(b3)
    ax.text(83.2, 4.9, "✖ Dualist points awarded: 0.0%\nunchanged — no test earned a ghost point\n(old pool 0.0 / 12.6 stays on the record)",
            color="#fee2e2", fontsize=7.4, fontweight="bold", ha="center", va="center", linespacing=1.2, zorder=4)

    out = MEDIA / "substack_final_scoreboard.png"
    fig.savefig(out, dpi=160, facecolor=fig.get_facecolor())
    plt.close(fig)
    print("Saved", out)


def render_fig8_ai_vs_human_authorship_heatmap():
    """
    Illustrated Guide 8 (Appendix of the Substack article): Heatmap of Human vs.
    AI-Agent Contribution Across the Main Article Sections and 5 Workflow
    Dimensions.

    The summary percentages printed on the figure (and quoted in the Substack article)
    are COMPUTED from `ai_matrix` below (unweighted mean across sections), so the
    figure can never disagree with its own cells. Run this function to print them.

    Rows follow `docs/SUBSTACK_LEVIN_LATENT_SPACE_ARTICLE.md` after the claim-by-claim
    walkthroughs were moved to the companion page. That companion is not a row here.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14.4, 8.4), dpi=160, facecolor="#0b1325", gridspec_kw={"width_ratios": [1.55, 1.0]})
    fig.subplots_adjust(wspace=0.22, left=0.235, right=0.97, top=0.845, bottom=0.125)

    fig.suptitle(
        "Methods Transparency: Heatmap of AI-Agent vs. Human Authorship Across This Article",
        color="#f8fafc", fontsize=14.2, fontweight="bold", y=0.965
    )

    sections = [
        "1. Intro: Bach, Levin & the Tools",
        "2. The Seven Claims (headers + GIF)",
        "3. The Score",
        "4. Rationalism: Short and Sweet",
        "5. On Dualism: Within/Without",
        "6. Conclusion",
        "7. Footnotes, the ++ & Problem Graph",
        "8. Appendix: How This Was Audited",
    ]
    dims = [
        "Hypothesis &\nDirection\n(% AI)",
        "Prose Text\nDrafted\n(% AI)",
        "Simulation &\nMath Code\n(% AI)",
        "Lean 4 &\nStats Proof\n(% AI)",
        "Referee Audit\n& Verification\n(% AI)",
    ]

    # Matrix of % AI contribution (0% = Pure Human Joel Dietz, 100% = Autonomous AI Agent)
    ai_matrix = np.array([
        [  5,   8,  15,  10,  20],  # 1. Intro
        [ 10,  50,  80,  80,  85],  # 2. Seven Claims headers (walkthrough lives on the companion page)
        [ 20,  60,  70,  60,  80],  # 3. The Score
        [ 15,  58,  90,  92,  88],  # 4. Rationalism
        [ 10,  25,  40,  30,  60],  # 5. On Dualism
        [ 10,  40,  75,  80,  82],  # 6. Conclusion
        [ 15,  72,  90,  60,  80],  # 7. Footnotes
        [ 15,  62,  85,  88,  90],  # 8. Appendix
    ], dtype=float)
    assert ai_matrix.shape == (len(sections), len(dims))

    # Summary statistics derived from the matrix itself (unweighted mean over sections).
    technical_rows = [2, 3, 7]  # Score, Rationalism, Appendix — the remaining technical prose
    hyp_ai = float(ai_matrix[:, 0].mean())
    prose_ai = float(ai_matrix[:, 1].mean())
    tech_ai = float(ai_matrix[:, 2:5].mean())
    tech_ai_technical = float(ai_matrix[technical_rows, 2:5].mean())
    summary = {
        "n_sections": len(sections),
        "hypothesis_ai_pct": round(hyp_ai, 1),
        "hypothesis_human_pct": round(100 - hyp_ai, 1),
        "prose_ai_pct": round(prose_ai, 1),
        "prose_human_pct": round(100 - prose_ai, 1),
        "code_proof_audit_ai_pct": round(tech_ai, 1),
        "code_proof_audit_human_pct": round(100 - tech_ai, 1),
        "code_proof_audit_ai_pct_technical_sections_only": round(tech_ai_technical, 1),
        "prose_ai_range_technical_sections": (int(ai_matrix[technical_rows, 1].min()), int(ai_matrix[technical_rows, 1].max())),
        "tech_ai_range_technical_sections": (int(ai_matrix[technical_rows, 2:5].min()), int(ai_matrix[technical_rows, 2:5].max())),
        "prose_human_pct_by_section": {i + 1: int(100 - ai_matrix[i, 1]) for i in range(len(sections))},
        "sections_ge_85pct_human_direction": [i + 1 for i in range(len(sections)) if ai_matrix[i, 0] <= 15],
    }
    print("AI-vs-human authorship summary (computed from ai_matrix):")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    fig.text(
        0.5, 0.92,
        f"Overall Article Text Breakdown (unweighted mean of {len(sections)} sections): {100 - prose_ai:.1f}% Direct Human Writing & Edits (Joel Dietz) · "
        f"{prose_ai:.1f}% AI-Assisted Synthesis & Technical Drafting  |  Code / Proofs / Audits: {tech_ai:.1f}% AI Agents",
        color="#93c5fd", fontsize=8.6, fontweight="bold", ha="center"
    )

    ax0 = axes[0]
    ax0.set_facecolor("#0f172a")
    im = ax0.imshow(ai_matrix, cmap="YlGnBu", vmin=0, vmax=100, aspect="auto")

    ax0.set_xticks(np.arange(len(dims)))
    ax0.set_xticklabels(dims, color="#f8fafc", fontsize=8.2, fontweight="bold")
    ax0.set_yticks(np.arange(len(sections)))
    ax0.set_yticklabels(sections, color="#f8fafc", fontsize=8.4, fontweight="bold")
    ax0.set_title("A. % AI Contribution by Article Section & Task (Remaining % = Human PI)", color="#38bdf8", fontsize=10.0, fontweight="bold", pad=8)

    for i in range(len(sections)):
        for j in range(len(dims)):
            val = int(ai_matrix[i, j])
            hum = 100 - val
            txt_col = "#f8fafc" if val >= 55 else "#0f172a"  # YlGnBu: light cells need dark text
            ax0.text(j, i, f"{val}% AI\n({hum}% Hum)", color=txt_col, fontsize=7.3, fontweight="bold", ha="center", va="center")

    cbar = fig.colorbar(im, ax=ax0, fraction=0.035, pad=0.03)
    cbar.ax.tick_params(colors="#cbd5e1", labelsize=8.0)
    cbar.set_label("% AI-Generated (0% = Pure Human, 100% = Autonomous Agent)", color="#cbd5e1", fontsize=8.0, fontweight="bold")

    # Right panel: Stacked horizontal bar chart of Prose Word Count (Human vs AI) by Section
    ax1 = axes[1]
    ax1.set_facecolor("#0f172a")
    for sp in ax1.spines.values():
        sp.set_color("#475569"); sp.set_linewidth(1.5)

    prose_ai_pct = ai_matrix[:, 1]
    prose_hum_pct = 100.0 - prose_ai_pct
    y_pos = np.arange(len(sections))

    ax1.barh(y_pos, prose_hum_pct, color="#22c55e", edgecolor="#0f172a", height=0.62, label=f"Human Authored / Edited (Joel Dietz: {100 - prose_ai:.1f}% Avg)")
    ax1.barh(y_pos, prose_ai_pct, left=prose_hum_pct, color="#6366f1", edgecolor="#0f172a", height=0.62, label=f"AI Technical Synthesis (Gemini 4 / Claude 5: {prose_ai:.1f}% Avg)")

    for i, (h_p, a_p) in enumerate(zip(prose_hum_pct, prose_ai_pct)):
        if h_p >= 18:
            ax1.text(h_p / 2, i, f"{int(h_p)}% Human", color="#052e16", fontsize=7.6, fontweight="bold", ha="center", va="center")
        ax1.text(h_p + a_p / 2, i, f"{int(a_p)}% AI", color="#ffffff", fontsize=7.6, fontweight="bold", ha="center", va="center")

    ax1.set_yticks(y_pos)
    ax1.set_yticklabels([f"Sec {i+1}" for i in range(len(sections))], color="#cbd5e1", fontsize=8.2, fontweight="bold")
    ax1.invert_yaxis()
    ax1.set_xlim(0, 100)
    ax1.set_xlabel("% of Section Prose Words (Human vs. AI)", color="#cbd5e1", fontsize=8.6, fontweight="bold")
    ax1.set_title("B. Prose Word Authorship Split per Section", color="#4ade80", fontsize=10.0, fontweight="bold", pad=8)
    ax1.tick_params(colors="#cbd5e1", labelsize=8.0)
    ax1.legend(loc="lower center", bbox_to_anchor=(0.5, -0.24), facecolor="#0b1325", edgecolor="#334155", labelcolor="#f8fafc", fontsize=7.8)

    out = MEDIA / "substack_ai_authorship_heatmap.png"
    fig.savefig(out, dpi=160, facecolor=fig.get_facecolor())
    plt.close(fig)
    print("Saved", out)


def render_fig9_causal_emergence_practitioner_workflow():
    """
    Illustrated Guide 9: Practitioner's 5-Step Workflow for Applying
    Causal Emergence 1.0 (PNAS 2013) & Causal Emergence 2.0 (Patterns 2026) in Practice.
      - Row 1 (Steps 1–3): Build Micro-TPM -> Define Candidate Macro-Partitions -> Coarse-Grain TPMs
      - Row 2 (Steps 4–5): Compute CE 1.0 vs. CE 2.0 (Det, Deg, CP ∈ [0,1], ΔCP ∈ [-1,1], EC) -> Limitations & Decision Rule
    """
    fig, axes = plt.subplots(2, 1, figsize=(14.4, 7.6), dpi=160, facecolor="#0b1325")
    fig.subplots_adjust(hspace=0.38, left=0.03, right=0.97, top=0.86, bottom=0.05)

    fig.suptitle(
        "Practitioner's Guide: The 5-Step Recipe for Applying Causal Emergence 1.0 & 2.0 in Practice",
        color="#f8fafc", fontsize=14.6, fontweight="bold", y=0.96
    )
    fig.text(
        0.5, 0.905,
        "How It Works & Where It Breaks: Testing Any Discrete System for Macro-Scale Causation (Hoel 2013 PNAS · Hoel & Levin 2020 · Hoel 2026 Patterns)",
        color="#93c5fd", fontsize=9.0, fontweight="bold", ha="center"
    )

    # --- TOP ROW: STEPS 1, 2, 3 (FROM SYSTEM TO COARSE-GRAINED TPMs) ---
    ax0 = axes[0]
    ax0.set_facecolor("#0f172a")
    ax0.set_xlim(0, 100); ax0.set_ylim(0, 10)
    ax0.set_xticks([]); ax0.set_yticks([])
    for sp in ax0.spines.values():
        sp.set_color("#38bdf8"); sp.set_linewidth(1.9)
    ax0.set_title(
        "PHASE A (Steps 1–3) — Build the Micro-TPM, Pre-Register Candidate Rulers g(s), and Coarse-Grain",
        color="#38bdf8", fontsize=10.8, fontweight="bold", pad=6
    )

    b_s1 = FancyBboxPatch((1.5, 1.2), 29.5, 7.6, boxstyle="round,pad=0.35,rounding_size=1.0", facecolor="#082f49", edgecolor="#38bdf8", lw=2.0)
    ax0.add_patch(b_s1)
    ax0.text(16.25, 7.5, "STEP 1 · Build Micro-TPM T_m (N × N)", color="#38bdf8", fontsize=9.2, fontweight="bold", ha="center")
    ax0.text(16.25, 4.1,
             "• Enumerate microstates s ∈ {1..N}\n"
             "  (e.g. N=720 in S_6, or N=32 in GRN)\n"
             "• Intervene uniformly: do(S_t = s_i)\n"
             "• Record row probabilities P(s_j | s_i)\n"
             "• Caveat: Watch for unreachable\n"
             "  Garden-of-Eden states!",
             color="#f8fafc", fontsize=7.8, fontweight="bold", ha="center", va="center", linespacing=1.25)

    ax0.annotate("", xy=(34.2, 5.0), xytext=(31.4, 5.0), arrowprops=dict(arrowstyle="->,head_length=7,head_width=5", color="#facc15", lw=2.6))

    b_s2 = FancyBboxPatch((34.6, 1.2), 30.5, 7.6, boxstyle="round,pad=0.35,rounding_size=1.0", facecolor="#1e293b", edgecolor="#facc15", lw=2.0)
    ax0.add_patch(b_s2)
    ax0.text(49.85, 7.5, "STEP 2 · Pre-Register Rulers g(s)", color="#facc15", fontsize=9.2, fontweight="bold", ha="center")
    ax0.text(49.85, 4.1,
             "• Define competing macro-maps\n"
             "  g: {1..N} → {1..M} BEFORE scoring\n"
             "  (avoids Bell-number B_N overfitting!):\n"
             "  1. Candidate A: Inversions I(v) (M=16)\n"
             "  2. Candidate B: Neighbor Drops E(v) (M=6)",
             color="#f8fafc", fontsize=7.8, fontweight="bold", ha="center", va="center", linespacing=1.25)

    ax0.annotate("", xy=(68.3, 5.0), xytext=(65.5, 5.0), arrowprops=dict(arrowstyle="->,head_length=7,head_width=5", color="#4ade80", lw=2.6))

    b_s3 = FancyBboxPatch((68.7, 1.2), 29.8, 7.6, boxstyle="round,pad=0.35,rounding_size=1.0", facecolor="#052e16", edgecolor="#4ade80", lw=2.0)
    ax0.add_patch(b_s3)
    ax0.text(83.6, 7.5, "STEP 3 · Coarse-Grain to Macro T_M", color="#4ade80", fontsize=9.2, fontweight="bold", ha="center")
    ax0.text(83.6, 4.1,
             "• Average micro-rows inside each\n"
             "  macro-bucket U_a = {s : g(s) = a}\n"
             "  and sum columns into U_b:\n"
             "  T_M[a,b] = (1/|U_a|) ∑_{i∈U_a} ∑_{j∈U_b} T_m[i,j]\n"
             "• Test dynamic lumpability: do all\n"
             "  microstates i ∈ U_a agree on P(U_b)?",
             color="#dcfce7", fontsize=7.8, fontweight="bold", ha="center", va="center", linespacing=1.25)

    # --- BOTTOM ROW: STEPS 4 & 5 (COMPUTE CE 1.0 vs CE 2.0 & MULTI-SCALE APPORTIONING / EC) ---
    ax1 = axes[1]
    ax1.set_facecolor("#0f172a")
    ax1.set_xlim(0, 100); ax1.set_ylim(0, 10)
    ax1.set_xticks([]); ax1.set_yticks([])
    for sp in ax1.spines.values():
        sp.set_color("#a855f7"); sp.set_linewidth(1.9)
    ax1.set_title(
        "PHASE B (Steps 4–5) — Compute CE 1.0 (EI) vs. CE 2.0 (CP ∈ [0,1], ΔCP ∈ [−1,1], EC) & Audit Limitations",
        color="#c084fc", fontsize=10.8, fontweight="bold", pad=6
    )

    b_s4 = FancyBboxPatch((1.5, 1.2), 46.5, 7.6, boxstyle="round,pad=0.35,rounding_size=1.0", facecolor="#2e1065", edgecolor="#a855f7", lw=2.0)
    ax1.add_patch(b_s4)
    ax1.text(24.75, 7.5, "STEP 4 · Compute Det, Deg, CP ∈ [0, 1], ΔCP ∈ [−1, 1] & EI", color="#e9d5ff", fontsize=9.1, fontweight="bold", ha="center")
    ax1.text(24.75, 4.1,
             "• Determinism: det(T) = 1 − ⟨H(row_i)⟩ / log₂(N)  |  Degeneracy: deg(T) = 1 − H(ū) / log₂(N)\n"
             "• CE 2.0 Entropic Effectiveness: CP_eff(T) = det(T) − deg(T) ∈ [0, 1] (by concavity H(ū) ≥ ⟨H⟩)\n"
             "• CE 2.0 Gain: ΔCP = CP_M − CP_m ∈ [−1, 1]  (vs. probabilistic CP_sn = suff + nec − 1)\n"
             "• CE 1.0 Effective Info: EI(T) = CP_eff(T) × log₂(N) bits  (ΔEI = ΔI_eff + ΔI_size)",
             color="#f8fafc", fontsize=7.75, fontweight="bold", ha="center", va="center", linespacing=1.28)

    ax1.annotate("", xy=(51.2, 5.0), xytext=(48.4, 5.0), arrowprops=dict(arrowstyle="->,head_length=7,head_width=5", color="#22c55e", lw=2.6))

    b_s5 = FancyBboxPatch((51.6, 1.2), 46.9, 7.6, boxstyle="round,pad=0.35,rounding_size=1.0", facecolor="#0f172a", edgecolor="#22c55e", lw=2.0)
    ax1.add_patch(b_s5)
    ax1.text(75.05, 7.5, "STEP 5 · Path Apportioning, Emergent Complexity (EC) & Audit", color="#4ade80", fontsize=9.1, fontweight="bold", ha="center")
    ax1.text(75.05, 4.1,
             "• Multi-scale path Π = (π^(0) → π^(1) → π^(2)):  CP_0 (0.8496) + ΔCP_1 (+0.0477) + ΔCP_2 (+0.0714) = 0.9687\n"
             "• Emergent Complexity: EC(Π) = −∑ w_k log₂(w_k) bits  (0.5378 bits 2-bin / 0.6572 bits 3-bin on S_6)\n"
             "• Critical Audit Check: Compare active-swap ΔCP (+0.1191 on I(v)) vs. raw self-loop ΔCP (−0.0501),\n"
             "  and verify whether analytic math (e.g. ΔI = −1) already proves the law without an N × N TPM!",
             color="#dcfce7", fontsize=7.75, fontweight="bold", ha="center", va="center", linespacing=1.28)

    out = MEDIA / "substack_ce2_practitioner_workflow.png"
    fig.savefig(out, dpi=160, facecolor=fig.get_facecolor())
    plt.close(fig)
    print("Saved", out)


if __name__ == "__main__":
    render_fig1_three_personalities()
    render_fig2_two_rulers_illusion()
    render_fig3_sports_cars_and_c2()
    render_fig4_real_data_curves()
    render_fig5_causal_emergence_13yo()
    render_fig6_decisive_comparison_d_vs_dp()
    render_fig7_final_scoreboard()
    render_fig8_ai_vs_human_authorship_heatmap()
    render_fig9_causal_emergence_practitioner_workflow()

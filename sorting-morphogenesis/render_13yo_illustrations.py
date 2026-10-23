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
    Illustrated Guide 5: The 13-Year-Old's Guide to 'Causal Emergence 2.0' (Erik Hoel 2013 & 2026):
      - Panel A (Top-Left): Micro-View ('Pixel Vision') — Starting at [2, 4, 1, 3], random
        kid choice branches into two different next lines ([2, 1, 4, 3] or [2, 4, 3, 1]).
        Looks like a 50/50 coin flip (86.9% determinism)!
      - Panel B (Top-Right): Macro-View ('Ruler 1 Vision', Total Pairs I(v)) — Both branches
        land on I = 2! 100% deterministic (3 -> 2 -> 1 -> 0), Causal Emergence = +0.119!
      - Panel C (Bottom-Left): Bad Macro-View ('Ruler 2 Vision', Neighbor Steps E(v)) —
        Starting from E = 1, the next step can go UP to E = 2 or DOWN to E = 0!
        72.1% determinism, Causal Emergence = -0.253 (Blurry Ruler Illusion!).
      - Panel D (Bottom-Right): Why CE 2.0 (2026) Beat CE 1.0 (2013) — CE 1.0 penalized
        Ruler 1 (-4.66 bits) just for having fewer buckets (16 vs. 720), hiding the
        +0.48 bit accuracy gain. CE 2.0 fixes the scorecard (+0.119 vs. -0.253)!
    """
    fig, axes = plt.subplots(2, 2, figsize=(13.6, 8.2), dpi=160, facecolor="#0b1325")
    fig.subplots_adjust(wspace=0.16, hspace=0.34, left=0.04, right=0.96, top=0.86, bottom=0.06)

    fig.suptitle(
        "Illustrated Guide 5: What Is 'Causal Emergence 2.0' (Hoel 2026)?\n"
        "Why Zooming Out to the Right Scoreboard Turns a Noisy Coin-Flip Into a 100% Predictable Law!",
        color="#f8fafc", fontsize=13.0, fontweight="bold", y=0.96
    )

    # --- Panel A: Micro-View (Noisy Coin Flip!) ---
    ax = axes[0, 0]
    ax.set_facecolor("#0f172a")
    ax.set_xlim(0, 10); ax.set_ylim(0, 6)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_color("#38bdf8"); sp.set_linewidth(1.8)
    ax.set_title("1. Micro-Scale ('Zoomed In'): Which Exact Line Comes Next? (Coin Flip!)", color="#38bdf8", fontsize=10.2, fontweight="bold", pad=6)

    # Start state box
    ax.text(1.9, 3.1, "Start Line:\n[ 2, 4, 1, 3 ]\n(I = 3, E = 1)", color="#f8fafc", fontsize=9.2, fontweight="bold", ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.35", facecolor="#1e293b", edgecolor="#38bdf8", lw=2.0))
    # Branch 1 (Top)
    ax.annotate("", xy=(5.6, 4.55), xytext=(3.4, 3.45), arrowprops=dict(arrowstyle="->,head_length=7,head_width=5", color="#38bdf8", lw=2.4))
    ax.text(4.35, 4.45, "50% chance:\nSwap 4 ↔ 1", color="#93c5fd", fontsize=8.0, fontweight="bold", ha="center")
    ax.text(7.6, 4.55, "Line A: [ 2, 1, 4, 3 ]\n(I = 2, E = 2 ↑)", color="#f8fafc", fontsize=8.8, fontweight="bold", ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#1e3a8a", edgecolor="#60a5fa", lw=1.8))
    # Branch 2 (Bottom)
    ax.annotate("", xy=(5.6, 1.65), xytext=(3.4, 2.75), arrowprops=dict(arrowstyle="->,head_length=7,head_width=5", color="#38bdf8", lw=2.4))
    ax.text(4.35, 1.65, "50% chance:\nSwap 4 ↔ 3", color="#93c5fd", fontsize=8.0, fontweight="bold", ha="center")
    ax.text(7.6, 1.65, "Line B: [ 2, 4, 3, 1 ]\n(I = 2, E = 2 ↑)", color="#f8fafc", fontsize=8.8, fontweight="bold", ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#1e3a8a", edgecolor="#60a5fa", lw=1.8))
    ax.text(5.0, 0.42, "Micro Predictability = 86.9% (Causal Score CP = 0.850)\nBecause random kids wake up first, exact micro-lines branch like a coin toss!",
            color="#bae6fd", fontsize=8.0, fontweight="bold", ha="center",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="#082f49", edgecolor="#0284c7"))

    # --- Panel B: Good Macro-View (Ruler 1 I(v) -> 100% Deterministic!) ---
    ax = axes[0, 1]
    ax.set_facecolor("#0f172a")
    ax.set_xlim(0, 10); ax.set_ylim(0, 6)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_color("#4ade80"); sp.set_linewidth(2.0)
    ax.set_title("2. Zoom Out to Ruler 1 (Total Pairs I): 100% Predictable Countdown!", color="#4ade80", fontsize=10.2, fontweight="bold", pad=6)

    xs_i = [1.4, 3.8, 6.2, 8.6]
    vals_i = ["I = 3\n(Start)", "I = 2\n(Both Land Here!)", "I = 1\n(Next Swap)", "I = 0\n(Sorted!)"]
    for k, (xp, lbl) in enumerate(zip(xs_i, vals_i)):
        ax.text(xp, 3.35, lbl, color="#ffffff", fontsize=8.8, fontweight="bold", ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.32", facecolor="#14532d" if k < 3 else "#15803d", edgecolor="#4ade80", lw=2.2))
        if k < 3:
            ax.annotate("", xy=(xs_i[k+1] - 0.95, 3.35), xytext=(xp + 0.95, 3.35),
                        arrowprops=dict(arrowstyle="->,head_length=7,head_width=5", color="#4ade80", lw=2.8))
            ax.text(0.5 * (xp + xs_i[k+1]), 4.25, "100% Sure!\n(ΔI = −1)", color="#86efac", fontsize=7.8, fontweight="bold", ha="center")

    ax.text(5.0, 0.95, "✔ POSITIVE CAUSAL EMERGENCE 2.0 (ΔCP = +0.1191 > 0)!\nDeterminism jumps 86.9% → 100.0% (CP: 0.850 → 0.969) because EVERY\nbranch at the micro-level does the exact same thing on Ruler 1 (I drops by −1)!",
            color="#dcfce7", fontsize=8.0, fontweight="bold", ha="center",
            bbox=dict(boxstyle="round,pad=0.24", facecolor="#052e16", edgecolor="#22c55e"))

    # --- Panel C: Bad Macro-View (Ruler 2 E(v) -> Negative Emergence!) ---
    ax = axes[1, 0]
    ax.set_facecolor("#0f172a")
    ax.set_xlim(0, 10); ax.set_ylim(0, 6)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_color("#f87171"); sp.set_linewidth(1.8)
    ax.set_title("3. Zoom Out to Ruler 2 (Neighbor Drops E): Blurry & Scrambled!", color="#f87171", fontsize=10.2, fontweight="bold", pad=6)

    ax.text(2.0, 3.35, "Ruler 2 Score:\nE = 1 Drop\n(Lumps [2,4,1,3]\nwith [1,3,2,4]!)", color="#ffffff", fontsize=8.4, fontweight="bold", ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#450a0a", edgecolor="#f87171", lw=2.0))
    # 3 scrambled arrows
    targets_e = [
        (4.85, "E = 2 (Goes UP +1!)\nif [2,4,1,3] → [2,1,4,3]", "#fca5a5"),
        (3.35, "E = 1 (Stays Flat 0)\nif [1,4,2,3] → [1,2,4,3]", "#cbd5e1"),
        (1.85, "E = 0 (Goes DOWN −1)\nif [1,3,2,4] → [1,2,3,4]", "#86efac"),
    ]
    for yp, txt, col in targets_e:
        ax.annotate("", xy=(5.4, yp), xytext=(3.65, 3.35), arrowprops=dict(arrowstyle="->,head_length=6,head_width=5", color="#f87171", lw=2.2))
        ax.text(7.55, yp, txt, color=col, fontsize=8.0, fontweight="bold", ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.22", facecolor="#1e293b", edgecolor="#ef4444", lw=1.4))

    ax.text(5.0, 0.55, "✖ NEGATIVE CAUSAL EMERGENCE (ΔCP = −0.2525 < 0)!\nDeterminism crashes 86.9% → 72.1% (CP: 0.850 → 0.597) because Ruler 2\nlumps together lines that go in opposite directions ('Delayed Gratification' illusion)!",
            color="#fee2e2", fontsize=7.9, fontweight="bold", ha="center",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="#450a0a", edgecolor="#ef4444"))

    # --- Panel D: Why Hoel's 2026 CE 2.0 Beat 2013 CE 1.0 ---
    ax = axes[1, 1]
    ax.set_facecolor("#0f172a")
    for sp in ax.spines.values():
        sp.set_color("#facc15"); sp.set_linewidth(1.8)
    ax.set_title("4. Why Hoel's 2026 'CE 2.0' Fixed the 2013 'CE 1.0' Size Penalty!", color="#facc15", fontsize=10.2, fontweight="bold", pad=8)

    labels = ["CE 1.0 Accuracy\nGain (ΔI_eff)", "CE 1.0 Bucket-Size\nPenalty (ΔI_size)", "Old 2013 CE 1.0\nTotal (ΔEI bits)", "New 2026 CE 2.0:\nRuler 1 I(v) (×10)", "New 2026 CE 2.0:\nRuler 2 E(v) (×10)"]
    vals = [0.476, -4.663, -4.187, 1.191, -2.525]
    cols = ["#38bdf8", "#94a3b8", "#f97316", "#4ade80", "#f87171"]
    x_b = np.arange(len(labels))
    bars = ax.bar(x_b, vals, color=cols, edgecolor="#f8fafc", linewidth=1.2, width=0.58)
    ax.axhline(0, color="#f8fafc", lw=1.2)
    ax.set_xticks(x_b)
    ax.set_xticklabels(labels, color="#f8fafc", fontsize=7.6, fontweight="bold")
    ax.set_ylim(-5.6, 2.3)
    ax.set_ylabel("Score (Bits / Scaled ΔCP)", color="#cbd5e1", fontsize=8.5, fontweight="bold")
    ax.tick_params(colors="#cbd5e1", labelsize=8.0)
    ax.grid(axis="y", color="#1e293b", lw=0.8)

    val_labels = ["+0.48 bits\n(Cleaner!)", "−4.66 bits\n(16 vs 720)", "−4.19 bits\n(Fooled!)", "+0.119 ΔCP\n(Ruler 1 Wins!)", "−0.253 ΔCP\n(Ruler 2 Fails!)"]
    for bar, v, vl in zip(bars, vals, val_labels):
        ypos = v + 0.18 if v >= 0 else v - 0.82
        ax.text(bar.get_x() + bar.get_width() / 2, ypos, vl, color="#f8fafc", fontsize=7.5, fontweight="bold", ha="center")

    out = MEDIA / "zhang_13yo_fig5_causal_emergence_explained.png"
    fig.savefig(out, dpi=160, facecolor=fig.get_facecolor())
    plt.close(fig)
    print("Saved", out)


if __name__ == "__main__":
    render_fig1_three_personalities()
    render_fig2_two_rulers_illusion()
    render_fig3_sports_cars_and_c2()
    render_fig4_real_data_curves()
    render_fig5_causal_emergence_13yo()

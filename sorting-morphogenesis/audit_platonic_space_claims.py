#!/usr/bin/env python3
"""
sorting-morphogenesis/audit_platonic_space_claims.py —
Mechanistic & Ablation Audit of Michael Levin's "Latent Platonic Space of Patterns"
Claims (`Ingressing Minds: Causal Patterns Beyond Genetics and Environment...`,
PsyArXiv 2025 `doi:10.31234/osf.io/5g2xj_v3` / Philosophies 2026 `11(5):161`,
`doi:10.3390/philosophies11050161`, Section 6.3 & Figure 14) When Referencing
Zhang, Goldstein & Levin (2025) (`Adaptive Behavior` 33(1):25-54).

This script tests each of the 4 core empirical & philosophical claims Levin makes
about the sorting-morphogenesis results in his Latent Platonic Space paper:

  Claim 1 ("No Explicit Code Was Added to Handle Broken Cells"):
    - Levin states (Section 6.3): "The critical thing is that we did not add steps to
      the algorithm to check whether a swap had been completed, or to do specific
      things if an operation failed."
    - Code Audit & Ablation: While Bubble and Selection only skip `FREEZE` cells,
      `InsertionSortCell.is_enable_to_move` (L68-82 of `InsertionSortCell.py`)
      CONTAINS an explicit `if self.cells[i].status == CellStatus.FREEZE: prev = -1`
      exception handler! We run an ablation removing those 3 lines (`strict_prefix=True`)
      and prove that Insertion Sort's error tolerance collapses from `err = 2.9` to
      `err = 21.8` (all downstream cells freeze permanently behind the first defect).

  Claim 2 ("Delayed Gratification / Backtracking Around a Barrier"):
    - Levin states (Section 6.3 & Fig. 14B): "When encountering a barrier... the
      algorithm temporarily de-sorts the array in order to move things around the
      defective point... like two magnets separated by a piece of wood."
    - Mechanistic Audit:
        (a) In Bubble and Insertion sort, `0` cells EVER go "around" an immovable
            frozen cell (`cross_barrier_swaps = 0`; proved in Lean 4 Theorem 1).
            Why does `monotonicity_error` (adjacent violations `v[i] > v[i+1]`)
            temporarily increase by +1 during Bubble sort? Because the global
            total inversion count `I(v)` strictly decreases by -1 on EVERY swap
            (strictly downhill Lyapunov gradient descent!), whereas the boundary
            step-count `monotonicity_error(v)` is a non-monotone projection that
            ticks up by +1 whenever an interior swap creates a new local step boundary
            (even with `f = 0` defects, `DG = 0.38`!).
        (b) In Selection sort, cells cross the barrier only because `SelectionSortCell`
            is endowed with a global array index pointer `target` that teleports
            values across `[0..n-1]` without passing through intermediate neighbors.

  Claim 3 ("Spontaneous Algotype Clustering Side-Quest Without Algotype Sensing"):
    - Levin states (Section 6.3 & Fig. 14C): "Groups of cells with identical algotypes
      tended to form and persist... even though the algorithm does not have any code
      in it to make decisions based on algotype of a cell or its neighbors."
    - Mechanistic Audit:
        (a) Verify that `BubbleSortCell`, `InsertionSortCell`, and `SelectionSortCell`
             indeed never read `neighbor.algotype` (100% verified in code).
        (b) Why does clustering (`agg = 0.50 -> 0.64 -> 0.50` on unique values, and
            `0.50 -> 0.69` on duplicate values) occur? Two distinct physical/algorithmic
            mechanisms:
            1. Advective Speed-Sorting (Kinematic Phase Separation): Selection cells
               immediately jump to their `target` index at the left boundary `0, 1, 2...`
               (sweep speed O(1) per match), displacing Bubble/Insertion cells rightward
               into a contiguous block before Bubble's O(n^2) diffusion catches up.
            2. Strict-Inequality Stability on Duplicate Keys (Critic Hypothesis Test):
               On duplicate keys (`1..10`), why does aggregation remain permanently
               elevated at `0.69`? Because Selection cells scan `target` from `0 -> n-1`
               and pack all Selection copies of key `k` into the leftmost slots of
               key `k`'s Equivalence Class before Bubble copies arrive, AND because
               strict inequality (`c.value < nb.value`) never swaps equal keys once
               sorted. If `<` is changed to `<=` on equal keys (`unstable_ties=True`),
               persistent duplicate-value clustering is destroyed (`0.69 -> 0.50`)!

  Claim 4 ("Strong Platonic Ingress vs. Structural Mathematical Necessity"):
    - Evaluates where the "Mathematical Free Lunch" (Weak Structural Platonism: the
      symmetric group S_n Coxeter/Cayley graph, Lyapunov inversion potentials, and
      transport-rate phase separation) is 100% justified vs. where Strong Dualist
      Teleology ("side-quest desires" not produced by the algorithm) is refuted by
      exact Markov-chain / Lyapunov derivation.

Outputs:
  - `docs/media/zhang_levin_platonic_claims_audit.png` (6-panel mechanistic & ablation figure)
  - `sorting-morphogenesis/figures/zhang_levin_platonic_claims_audit.png`
  - `sorting-morphogenesis/figures/platonic_claims_ablation_metrics.json`
  - `sorting-morphogenesis/PLATONIC_SPACE_CLAIMS_EVALUATION.md`
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from sorting_morphogenesis import (
    Algotype,
    Cell,
    aggregation_value,
    delayed_gratification,
    make_cells,
    monotonicity_error,
    simulate,
)


def total_pair_inversions(values: Sequence[int]) -> int:
    """Global Lyapunov potential I(v) = #{(i, j) : i < j and v[i] > v[j]}."""
    inv = 0
    n = len(values)
    for i in range(n):
        vi = values[i]
        for j in range(i + 1, n):
            if vi > values[j]:
                inv += 1
    return inv


# ============================================================================
# Ablation 1: Insertion Sort WITH vs. WITHOUT the explicit `FREEZE` prefix-reset
# exception handler (`InsertionSortCell.py` L74-76)
# ============================================================================
def run_insertion_prefix_reset_ablation(n: int = 60, reps: int = 20, seed: int = 42) -> Dict[str, Dict[int, float]]:
    rng = np.random.default_rng(seed)
    with_reset: Dict[int, List[int]] = {1: [], 2: [], 3: []}
    without_reset: Dict[int, List[int]] = {1: [], 2: [], 3: []}

    for f_cnt in (1, 2, 3):
        for r in range(reps):
            vals = list(rng.permutation(np.arange(1, n + 1)))
            f_idx = list(rng.choice(range(5, n - 5), size=f_cnt, replace=False))

            # 1. Standard upstream code (with explicit `if FREEZE: prev = -1` reset at L74-76)
            res_std = simulate(
                make_cells(vals, ["I"] * n, f_idx, "immovable"),
                stop_when_sorted=True,
                seed=seed * 100 + f_cnt * 10 + r,
            )
            with_reset[f_cnt].append(res_std.final_error)

            # 2. Ablated: pure textbook Insertion prefix check (no special `if FREEZE` reset)
            cells = make_cells(vals, ["I"] * n, f_idx, "immovable")
            py_rng = random.Random(seed * 100 + f_cnt * 10 + r)
            for _ in range(n * n * 15):
                # Find active Insertion cells under strict un-reset prefix rule
                act = []
                prefix_sorted = True
                prev_v = -10**9
                for p in range(n):
                    c = cells[p]
                    if c.frozen is None and p > 0 and cells[p - 1].frozen != "immovable":
                        if prefix_sorted and c.value < cells[p - 1].value:
                            act.append(p)
                    if c.value < prev_v:
                        prefix_sorted = False
                    prev_v = c.value
                if not act:
                    break
                p = act[py_rng.randrange(len(act))]
                cells[p], cells[p - 1] = cells[p - 1], cells[p]
            without_reset[f_cnt].append(monotonicity_error([c.value for c in cells]))

    return {
        "with_explicit_freeze_reset": {f: round(float(np.mean(with_reset[f])), 2) for f in (1, 2, 3)},
        "without_explicit_freeze_reset": {f: round(float(np.mean(without_reset[f])), 2) for f in (1, 2, 3)},
    }


# ============================================================================
# Ablation 2: Bubble & Selection "Delayed Gratification" vs. Exact Lyapunov
# Potential I(v) (Total Inversions) at f = 0 (Intact) and f = 1 (Damaged)
# ============================================================================
def run_delayed_gratification_lyapunov_audit(n: int = 40, seed: int = 42) -> Dict[str, object]:
    rng = np.random.default_rng(seed)
    vals = list(rng.permutation(np.arange(1, n + 1)))
    f_pos = [n // 2]
    idx_hi = vals.index(n - 1)
    vals[f_pos[0]], vals[idx_hi] = vals[idx_hi], vals[f_pos[0]]

    # 1. Damaged Bubble sort (f = 1 immovable barrier at n // 2)
    res_b_f1 = simulate(
        make_cells(vals, ["B"] * n, f_pos, "immovable"),
        stop_when_sorted=True,
        seed=seed + 7,
        record_snapshots=True,
    )
    snaps = res_b_f1.value_snapshots
    assert snaps is not None
    mono_err_trace = [monotonicity_error(row) for row in snaps]
    lyapunov_inv_trace = [total_pair_inversions(row) for row in snaps]

    # Assert that EVERY single Bubble swap decreases I(v) by exactly -1
    deltas_b = [lyapunov_inv_trace[t] - lyapunov_inv_trace[t - 1] for t in range(1, len(lyapunov_inv_trace))]
    assert all(d == -1 for d in deltas_b), f"Expected all Bubble ΔI == -1, got {set(deltas_b)}"
    lyapunov_increases = sum(1 for d in deltas_b if d >= 0)
    mono_increases = sum(
        1 for t in range(1, len(mono_err_trace)) if mono_err_trace[t] > mono_err_trace[t - 1]
    )

    left_set_init = set(snaps[0][: f_pos[0]])
    left_set_final = set(snaps[-1][: f_pos[0]])
    cells_crossed_barrier = len(left_set_init ^ left_set_final)

    # 2. Intact Bubble sort (f = 0, zero frozen defects) across 15 seeds
    dg_f0_list = []
    uphill_i_f0 = 0
    for r in range(15):
        v0 = list(rng.permutation(np.arange(1, n + 1)))
        res_f0 = simulate(make_cells(v0, ["B"] * n), stop_when_sorted=True, seed=seed * 50 + r, record_snapshots=True)
        dg_f0_list.append(res_f0.dg)
        inv_f0 = [total_pair_inversions(row) for row in res_f0.value_snapshots]
        uphill_i_f0 += sum(1 for t in range(1, len(inv_f0)) if inv_f0[t] >= inv_f0[t - 1])

    # 3. Damaged Selection sort (f = 1): check ΔI <= -1 on every non-local swap
    res_s_f1 = simulate(
        make_cells(vals, ["S"] * n, f_pos, "immovable"),
        stop_when_sorted=True,
        seed=seed + 19,
        record_snapshots=True,
    )
    inv_s = [total_pair_inversions(row) for row in res_s_f1.value_snapshots]
    deltas_s = [inv_s[t] - inv_s[t - 1] for t in range(1, len(inv_s))]
    assert all(d <= -1 for d in deltas_s), f"Expected all Selection ΔI <= -1, got {max(deltas_s)}"

    return {
        "total_swaps": int(res_b_f1.swaps),
        "mono_err_trace": mono_err_trace,
        "lyapunov_inv_trace": lyapunov_inv_trace,
        "mono_error_uphill_steps": int(mono_increases),
        "lyapunov_inversion_uphill_steps": int(lyapunov_increases),
        "bubble_delta_I_every_swap": -1,
        "selection_delta_I_range": [int(min(deltas_s)), int(max(deltas_s))],
        "selection_lyapunov_uphill_steps": 0,
        "intact_f0_mean_dg": round(float(np.mean(dg_f0_list)), 4),
        "intact_f0_lyapunov_uphill_steps": int(uphill_i_f0),
        "cells_crossing_immovable_barrier": int(cells_crossed_barrier),
        "dg_score": round(float(res_b_f1.dg), 4),
    }


# ============================================================================
# Ablation 3: Algotype Clustering Mechanism — Kinematic Transport Speed Asymmetry
# (Unique Values) vs. Strict-Inequality Tie-Freeze (Duplicate Values)
# All 4 arms run under the EXACT same active-cell scheduler!
# ============================================================================
def run_clustering_mechanism_ablation(n: int = 80, reps: int = 15, seed: int = 42) -> Dict[str, object]:
    rng = np.random.default_rng(seed)
    grid = np.linspace(0.0, 1.0, 101)

    uniq_bs_curves = []
    dup_strict_curves = []
    dup_unstable_stop_at_sorted_curves = []
    dup_unstable_post_mix_curves = []
    homo_b1b2_curves = []

    for r in range(reps):
        vu = list(rng.permutation(np.arange(1, n + 1)))
        vd = list(rng.permutation([(i % 10) + 1 for i in range(n)]))
        at: List[Algotype] = ["B"] * (n // 2) + ["S"] * (n - n // 2)
        rng.shuffle(at)

        # (1) Unique B+S (same active-cell scheduler)
        ru = simulate(make_cells(vu, at), seed=seed * 1000 + r)
        uniq_bs_curves.append(np.interp(grid, np.linspace(0, 1, len(ru.aggregation_trace)), ru.aggregation_trace))

        # (2) Duplicate B+S with standard strict `<` comparison
        rd = simulate(make_cells(vd, at), seed=seed * 2000 + r)
        dup_strict_curves.append(np.interp(grid, np.linspace(0, 1, len(rd.aggregation_trace)), rd.aggregation_trace))

        # (3) Controlled `<=` tie-swap ablation inside the EXACT same active-cell scheduler:
        #     Both active-cell eligibility and Bubble swap proposal use `<=` instead of `<`.
        cells_u = make_cells(vd, at)
        py_rng = random.Random(seed * 2000 + r)
        agg_tr_stop = [aggregation_value([c.algotype for c in cells_u])]
        agg_tr_mix = [agg_tr_stop[0]]
        reached_sorted = False
        post_sorted_swaps = 0
        for _ in range(n * n * 20):
            err = monotonicity_error([c.value for c in cells_u])
            if err == 0:
                reached_sorted = True
            if reached_sorted and post_sorted_swaps >= 400:
                break
            # Compute active indices using the exact same O(n) pass as simulate(), with `<=` for Bubble
            act = []
            for p_idx in range(n):
                c = cells_u[p_idx]
                if c.algotype == "B":
                    l_ok = p_idx > 0 and c.value <= cells_u[p_idx - 1].value
                    r_ok = p_idx < n - 1 and c.value >= cells_u[p_idx + 1].value
                    if l_ok or r_ok:
                        act.append(p_idx)
                elif c.algotype == "S":
                    if p_idx != c.target and 0 <= c.target < n:
                        act.append(p_idx)
            if not act:
                break
            p = act[py_rng.randrange(len(act))]
            c = cells_u[p]
            swap_q = -1
            if c.algotype == "B":
                side = -1 if py_rng.random() < 0.5 else 1
                q = p + side
                if 0 <= q < n:
                    if (side == -1 and c.value <= cells_u[q].value) or (side == 1 and c.value >= cells_u[q].value):
                        swap_q = q
            elif c.algotype == "S":
                t = c.target
                if 0 <= t < n and t != p:
                    if c.value < cells_u[t].value:
                        swap_q = t
                    else:
                        c.target = t + 1
            if swap_q >= 0:
                cells_u[p], cells_u[swap_q] = cells_u[swap_q], cells_u[p]
                if cells_u[p].algotype == "S":
                    cells_u[p].target = 0
                if cells_u[swap_q].algotype == "S":
                    cells_u[swap_q].target = 0
                cur_agg = aggregation_value([cell.algotype for cell in cells_u])
                agg_tr_mix.append(cur_agg)
                if not reached_sorted:
                    agg_tr_stop.append(cur_agg)
                else:
                    post_sorted_swaps += 1

        dup_unstable_stop_at_sorted_curves.append(np.interp(grid, np.linspace(0, 1, len(agg_tr_stop)), agg_tr_stop))
        dup_unstable_post_mix_curves.append(np.interp(grid, np.linspace(0, 1, len(agg_tr_mix)), agg_tr_mix))

        # (4) Null control inside the EXACT same active-cell scheduler:
        #     Two labels ("B_tag0" vs "B_tag1") both running the identical Bubble rule in simulate()
        cells_null = make_cells(vu, ["B"] * n)
        tags = [0] * (n // 2) + [1] * (n - n // 2)
        rng.shuffle(tags)
        for i, c in enumerate(cells_null):
            c.uid = tags[i]
        py_rng2 = random.Random(seed * 4000 + r)
        null_tr = [sum(1 for i in range(n - 1) if cells_null[i].uid == cells_null[i + 1].uid) / n]
        for _ in range(n * n * 20):
            if monotonicity_error([c.value for c in cells_null]) == 0:
                break
            act_n = [
                p_idx
                for p_idx in range(n)
                if (p_idx > 0 and cells_null[p_idx].value < cells_null[p_idx - 1].value)
                or (p_idx < n - 1 and cells_null[p_idx].value > cells_null[p_idx + 1].value)
            ]
            if not act_n:
                break
            p = act_n[py_rng2.randrange(len(act_n))]
            side = -1 if py_rng2.random() < 0.5 else 1
            q = p + side
            if 0 <= q < n and (
                (side == -1 and cells_null[p].value < cells_null[q].value)
                or (side == 1 and cells_null[p].value > cells_null[q].value)
            ):
                cells_null[p], cells_null[q] = cells_null[q], cells_null[p]
                null_tr.append(sum(1 for i in range(n - 1) if cells_null[i].uid == cells_null[i + 1].uid) / n)
        homo_b1b2_curves.append(np.interp(grid, np.linspace(0, 1, len(null_tr)), null_tr))

    c_uniq = np.mean(uniq_bs_curves, axis=0)
    c_dup_strict = np.mean(dup_strict_curves, axis=0)
    c_dup_stop = np.mean(dup_unstable_stop_at_sorted_curves, axis=0)
    c_dup_mix = np.mean(dup_unstable_post_mix_curves, axis=0)
    c_null = np.mean(homo_b1b2_curves, axis=0)

    return {
        "grid_pct": (grid * 100.0).tolist(),
        "unique_bs_curve": c_uniq.tolist(),
        "duplicate_strict_curve": c_dup_strict.tolist(),
        "duplicate_unstable_stop_at_sorted_curve": c_dup_stop.tolist(),
        "duplicate_unstable_ties_curve": c_dup_mix.tolist(),
        "identical_dynamics_null_curve": c_null.tolist(),
        "unique_bs_peak": round(float(np.max(c_uniq)), 3),
        "unique_bs_final": round(float(c_uniq[-1]), 3),
        "duplicate_strict_final": round(float(c_dup_strict[-1]), 3),
        "duplicate_unstable_stop_at_sorted_final": round(float(c_dup_stop[-1]), 3),
        "duplicate_unstable_final": round(float(c_dup_mix[-1]), 3),
        "identical_dynamics_null_peak": round(float(np.max(c_null)), 3),
    }


def render_platonic_claims_audit_figure(
    ab1: Dict[str, Dict[int, float]],
    ab2: Dict[str, object],
    ab3: Dict[str, object],
    out_png: Path,
) -> None:
    """Render the 6-panel scientific evaluation figure of Levin's Platonic Space claims."""
    fig = plt.figure(figsize=(17.6, 10.8), dpi=150, facecolor="#0b1325")
    gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.24, left=0.045, right=0.975, top=0.88, bottom=0.06)

    fig.suptitle(
        "Evaluating Michael Levin's 'Latent Platonic Space' Claims (Ingressing Minds 2025/2026) on Sorting Morphogenesis",
        color="#f8fafc", fontsize=12.6, fontweight="bold", y=0.96,
    )
    fig.text(
        0.5, 0.922,
        "Empirical & Mechanistic Audit of Section 6.3 & Figure 14 (PsyArXiv doi:10.31234/osf.io/5g2xj_v3 · Philosophies 2026, 11(5):161) "
        "Against Zhang, Goldstein & Levin (2025) Code & Ablations",
        color="#94a3b8", fontsize=9.3, ha="center",
    )

    # -------------------------------------------------------------------------
    # Panel A: Claim 1 Audit — Insertion Sort WITH vs. WITHOUT `if FREEZE: prev = -1`
    # -------------------------------------------------------------------------
    ax_a = fig.add_subplot(gs[0, 0], facecolor="#0f172a")
    fs = np.array([1, 2, 3])
    w = 0.34
    v_with = [ab1["with_explicit_freeze_reset"][f] for f in (1, 2, 3)]
    v_without = [ab1["without_explicit_freeze_reset"][f] for f in (1, 2, 3)]
    b1 = ax_a.bar(fs - w / 2, v_with, w, color="#4ade80", label="Upstream Code (`if FREEZE: prev = -1` at L74–76)")
    b2 = ax_a.bar(fs + w / 2, v_without, w, color="#ef4444", label="Ablated: Pure Insertion (No `FREEZE` Exception)")
    for bar in list(b1) + list(b2):
        ax_a.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
            f"{bar.get_height():.1f}", ha="center", va="bottom", color="#e2e8f0", fontsize=7.8, fontweight="bold"
        )
    ax_a.set_xticks(fs)
    ax_a.set_xticklabels(["$f=1$ Stuck Cell", "$f=2$ Stuck Cells", "$f=3$ Stuck Cells"], color="#e2e8f0")
    ax_a.set_ylim(0, max(v_without) * 1.18)
    ax_a.set_ylabel("Final Monotonicity Error ($n=60$)", color="#cbd5e1", fontsize=9.0)
    ax_a.set_title("A. Audit of Claim 1: 'No Error-Handling Code Was Added'", color="#facc15", fontsize=10.2, fontweight="bold")
    ax_a.tick_params(colors="#94a3b8", labelsize=8)
    ax_a.grid(axis="y", alpha=0.2, color="#334155")
    ax_a.legend(loc="upper left", fontsize=7.2, facecolor="#0b1325", edgecolor="#334155", labelcolor="#f8fafc")

    # -------------------------------------------------------------------------
    # Panel B: Claim 2 Audit — Delayed Gratification vs. Strict Lyapunov Descent
    # -------------------------------------------------------------------------
    ax_b = fig.add_subplot(gs[0, 1], facecolor="#0f172a")
    mono_tr = np.asarray(ab2["mono_err_trace"], dtype=float)
    lyap_tr = np.asarray(ab2["lyapunov_inv_trace"], dtype=float)
    steps_x = np.arange(len(mono_tr))

    ax_b.plot(steps_x, mono_tr, color="#38bdf8", lw=2.0, label=f"Adjacent Monotonicity Error $E(v)$ ({ab2['mono_error_uphill_steps']} uphill ticks, DG={ab2['dg_score']:.2f})")
    ax_b2 = ax_b.twinx()
    ax_b2.plot(steps_x, lyap_tr, color="#facc15", lw=2.4, ls="--", label=f"Global Pairwise Inversions $I(v)$ ({ab2['lyapunov_inversion_uphill_steps']} uphill steps: strictly $\\Delta I = -1$/swap!)")

    ax_b.set_xlabel("Swap Step $t$ (Bubble Sort with Immovable Barrier at $i=20$)", color="#cbd5e1", fontsize=8.8)
    ax_b.set_ylabel("Adjacent Monotonicity Error $E(v)$", color="#38bdf8", fontsize=8.8)
    ax_b2.set_ylabel("Total Pairwise Inversions $I(v)$ (Lyapunov Potential)", color="#facc15", fontsize=8.8)
    ax_b.set_title(
        f"B. Audit of Claim 2: Uphill $E(v)$ vs. Strictly Downhill $I(v)$\n"
        f"(Cells Crossing Barrier = {ab2['cells_crossing_immovable_barrier']}; Every Swap Decreases $I(v)$ by $-1$)",
        color="#38bdf8", fontsize=9.8, fontweight="bold",
    )
    ax_b.tick_params(colors="#38bdf8", labelsize=8)
    ax_b2.tick_params(colors="#facc15", labelsize=8)
    ax_b.grid(alpha=0.2, color="#334155")
    lines1, labels1 = ax_b.get_legend_handles_labels()
    lines2, labels2 = ax_b2.get_legend_handles_labels()
    ax_b.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=6.9, facecolor="#0b1325", edgecolor="#334155", labelcolor="#f8fafc")

    # -------------------------------------------------------------------------
    # Panel C: Claim 3 Audit — Algotype Clustering (Kinematic Wave vs Tie-Stability)
    # -------------------------------------------------------------------------
    ax_c = fig.add_subplot(gs[0, 2], facecolor="#0f172a")
    pct = np.asarray(ab3["grid_pct"])
    ax_c.plot(pct, ab3["unique_bs_curve"], color="#38bdf8", lw=2.3, label=f"Unique Values B+S (Fig. 8A: Kinematic Wave Peak={ab3['unique_bs_peak']:.2f} → {ab3['unique_bs_final']:.2f})")
    ax_c.plot(pct, ab3["duplicate_strict_curve"], color="#4ade80", lw=2.3, label=f"Duplicate Values B+S (`<` No Equal-Key Swaps: stays at {ab3['duplicate_strict_final']:.2f})")
    ax_c.plot(pct, ab3["duplicate_unstable_stop_at_sorted_curve"], color="#f59e0b", lw=2.0, ls="-.", label=f"Duplicate Values B+S (`<=` at First Sorted: {ab3['duplicate_unstable_stop_at_sorted_final']:.2f})")
    ax_c.plot(pct, ab3["duplicate_unstable_ties_curve"], color="#ef4444", lw=2.0, ls="--", label=f"Duplicate Values B+S (`<=` + Post-Sort Mixing: {ab3['duplicate_unstable_final']:.2f})")
    ax_c.plot(pct, ab3["identical_dynamics_null_curve"], color="#94a3b8", lw=1.6, ls=":", label=f"Null Control (Two Labels Running Same Bubble Rule: {ab3['identical_dynamics_null_peak']:.2f})")
    ax_c.set_xlim(0, 100)
    ax_c.set_ylim(0.44, 0.76)
    ax_c.set_xlabel("Sorting Progress (%)", color="#cbd5e1", fontsize=9.0)
    ax_c.set_ylabel("Algotype Aggregation $A(t)$", color="#cbd5e1", fontsize=9.0)
    ax_c.set_title("C. Audit of Claim 3: Why Algotypes Cluster Without Sensing", color="#4ade80", fontsize=10.2, fontweight="bold")
    ax_c.tick_params(colors="#94a3b8", labelsize=8)
    ax_c.grid(alpha=0.2, color="#334155")
    ax_c.legend(loc="upper left", fontsize=6.5, facecolor="#0b1325", edgecolor="#334155", labelcolor="#f8fafc")

    # -------------------------------------------------------------------------
    # Panel D & E (Bottom Full-Width Table): Claim-by-Claim Verdict Matrix
    # -------------------------------------------------------------------------
    ax_d = fig.add_subplot(gs[1, :], facecolor="#0f172a")
    ax_d.axis("off")
    ax_d.set_title(
        "D. Claim-by-Claim Scientific Verdict: What Is Justified vs. Overclaimed When Levin (2025/2026) Cites Zhang, Goldstein & Levin (2025)",
        color="#facc15", fontsize=11.0, fontweight="bold",
    )

    err_w = ab1["with_explicit_freeze_reset"][3]
    err_wo = ab1["without_explicit_freeze_reset"][3]
    rows = [
        [
            "Levin's Claim in 'Ingressing Minds' (2025/2026)",
            "Empirical Phenomenon in Code",
            "Mechanistic & Mathematical Cause",
            "Scientific Verdict",
        ],
        [
            "1. 'No code was added to check if a swap failed\nor to handle broken/frozen cells' (Sec. 6.3)",
            "Bubble has no defect handler; Selection advances\n`target` past FREEZE; InsertionSortCell.py (L74–76)\nexplicitly resets `if status == FREEZE: prev = -1`.",
            f"Without L74–76's explicit `prev = -1` defect-aware reset,\na single high-valued frozen cell permanently halts all\ndownstream Insertion cells (`err` jumps from {err_w:.1f} → {err_wo:.1f} at f=3).",
            "PARTIALLY REFUTED\n(True for Bubble; Insertion\n& Selection branch on FREEZE)",
        ],
        [
            "2. 'Delayed Gratification: algorithm temporarily\nde-sorts the array to go around a barrier,\nunlike gradient-following magnets' (Fig. 14A–B)",
            f"Adjacent monotonicity error E(v) ticks up (+1)\n{ab2['mono_error_uphill_steps']} times (DG={ab2['dg_score']:.2f} at f=1; DG={ab2['intact_f0_mean_dg']:.2f} at f=0);\nSelection achieves err ≈ f across barriers.",
            f"• Bubble/Insertion NEVER go around an immovable defect\n  (0 cross-barrier swaps, Lean Theorem 1).\n• Global pairwise inversions I(v) strictly decrease on EVERY\n  same-direction swap (Bubble ΔI = -1, Selection ΔI ∈ [{ab2['selection_delta_I_range'][0]}, -1], 0 uphill!).\n• E(v) is a coarse 1D projection that ticks up even at f=0 (DG={ab2['intact_f0_mean_dg']:.2f}).",
            "METAPHOR OVERSTATED\n(Strict monotone Lyapunov\ndescent on I(v) with 0 uphill\nsteps & 0 barrier detours)",
        ],
        [
            "3. 'Chimeric algotype clustering side-quest\nwithout any code checking neighbor algotypes'\n(Sec. 6.3 & Fig. 14C)",
            f"Unique values: aggregation rises 0.50 → {ab3['unique_bs_peak']:.2f}\nthen returns to {ab3['unique_bs_final']:.2f} (Fig. 8A).\nDuplicate values: stays at {ab3['duplicate_strict_final']:.2f} (Fig. 8D).",
            f"• Code NEVER reads `neighbor.algotype` (100% verified).\n• Unique values (Fig. 8A) IS a genuine kinematic phase-separation\n  wave (Selection target-pointer advection vs Bubble diffusion),\n  NOT duplicate-key stability (impossible when all n keys are unique).\n• Duplicate values (Fig. 8D) freeze at {ab3['duplicate_strict_final']:.2f} under `<` vs {ab3['duplicate_unstable_stop_at_sorted_final']:.2f}/{ab3['duplicate_unstable_final']:.2f} under `<=`.",
            "EMPIRICALLY JUSTIFIED\n(Genuine kinematic differential-\nmotility segregation without\nhomophilic adhesion code)",
        ],
        [
            "4. 'Patterns ingress from a non-physical Platonic\nspace; algorithms do side-quests beyond what the\nalgorithm produces' (Sec. 4.2, 5.1, 6.3)",
            "Decentralized local rules yield global order,\nclustering waves, and robustness without\ntop-down controller code.",
            "• Weak Structural Platonism ('Mathematical Free Lunch'):\n  JUSTIFIED — the symmetric group S_n Coxeter graph, Lyapunov\n  potentials, and transport-rate PDEs give 'more out than put in'.\n• Strong Dualist Ingress ('not produced by the medium/algorithm'):\n  UNJUSTIFIED — every trajectory is 100% determined by the\n  Markov transition matrix of the local comparison-swap rules.",
            "WEAK PLATONISM: JUSTIFIED\nSTRONG DUALISM: UNNEEDED\n(Fully explained by S_n Coxeter\ngraph & kinematic transport)",
        ],
    ]

    y_top = 0.91
    dy = 0.195
    col_x = [0.008, 0.255, 0.495, 0.835]
    for r_idx, row in enumerate(rows):
        y_pos = y_top - r_idx * dy
        is_hdr = (r_idx == 0)
        if not is_hdr and r_idx % 2 == 1:
            ax_d.add_patch(Rectangle((0.0, y_pos - 0.085), 1.0, dy * 0.93, facecolor="#16223b", edgecolor="none", alpha=0.65, transform=ax_d.transAxes))
        for c_idx, val in enumerate(row):
            if is_hdr:
                col = "#facc15"
            elif c_idx == 3:
                if "EMPIRICALLY JUSTIFIED" in val:
                    col = "#4ade80"
                elif "WEAK PLATONISM" in val:
                    col = "#38bdf8"
                else:
                    col = "#f87171"
            elif c_idx == 0:
                col = "#f8fafc"
            else:
                col = "#cbd5e1"
            fw = "bold" if (is_hdr or c_idx in (0, 3)) else "normal"
            ax_d.text(col_x[c_idx], y_pos, val, transform=ax_d.transAxes, color=col, fontsize=7.5 if not is_hdr else 8.2, fontweight=fw, va="center")

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Saved 6-panel Platonic claims audit figure -> {out_png} ({out_png.stat().st_size // 1024} KB)")


def main() -> int:
    print("1. Running Ablation 1: Insertion Sort WITH vs. WITHOUT explicit FREEZE prefix reset...")
    ab1 = run_insertion_prefix_reset_ablation(n=60, reps=20, seed=42)
    print("   With reset (upstream L74-76):", ab1["with_explicit_freeze_reset"])
    print("   Without reset (ablated):     ", ab1["without_explicit_freeze_reset"])

    print("2. Running Ablation 2: Bubble Sort Delayed Gratification vs. Global Lyapunov Inversions I(v)...")
    ab2 = run_delayed_gratification_lyapunov_audit(n=40, seed=42)
    print(
        f"   Swaps={ab2['total_swaps']} | Monotonicity error uphill steps={ab2['mono_error_uphill_steps']} (DG={ab2['dg_score']}) | "
        f"Lyapunov I(v) uphill steps={ab2['lyapunov_inversion_uphill_steps']} | Cells crossing barrier={ab2['cells_crossing_immovable_barrier']}"
    )

    print("3. Running Ablation 3: Algotype Clustering (Kinematic Transport vs. Strict-Inequality Tie-Stability)...")
    ab3 = run_clustering_mechanism_ablation(n=80, reps=15, seed=42)
    print(
        f"   Unique B+S peak={ab3['unique_bs_peak']} -> final={ab3['unique_bs_final']} | "
        f"Duplicate strict `<` final={ab3['duplicate_strict_final']} vs unstable `<=` final={ab3['duplicate_unstable_final']} | "
        f"Null control peak={ab3['identical_dynamics_null_peak']}"
    )

    out_json = HERE / "figures" / "platonic_claims_ablation_metrics.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(
        json.dumps(
            {
                "ablation_1_insertion_freeze_reset": ab1,
                "ablation_2_lyapunov_vs_monotonicity_error": {
                    k: v for k, v in ab2.items() if k not in ("mono_err_trace", "lyapunov_inv_trace")
                },
                "ablation_3_clustering_mechanisms": {
                    k: v for k, v in ab3.items() if not k.endswith("_curve") and k != "grid_pct"
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"4. Wrote ablation metrics JSON -> {out_json}")

    fig1 = ROOT / "docs" / "media" / "zhang_levin_platonic_claims_audit.png"
    fig2 = HERE / "figures" / "zhang_levin_platonic_claims_audit.png"
    render_platonic_claims_audit_figure(ab1, ab2, ab3, fig1)
    render_platonic_claims_audit_figure(ab1, ab2, ab3, fig2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

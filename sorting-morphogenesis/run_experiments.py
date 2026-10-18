"""
Replicate the experiments of Zhang, Goldstein & Levin (2025),
"Classical sorting algorithms as a model of morphogenesis: Self-sorting arrays
reveal unexpected competencies in a minimal model of basal intelligence",
Adaptive Behavior 33(1):25-54 (doi:10.1177/10597123241269740; arXiv:2401.05375).

Authors' reference code: https://github.com/Zhangtaining/cell_research
Pinned commit SHA: 1fd2bd5921c1f6b423a71f691d5189106a8a1020

Produces 7 figures in `sorting-morphogenesis/figures/` and prints a quantitative
replication summary comparing simulated results against the paper's published
numbers:
  1. efficiency.png             - Swaps (Fig. 4A), Comparisons+Swaps (Fig. 4B),
                                  and Chimeric Efficiency (Fig. 8B)
  2. error_tolerance.png        - Final Monotonicity Error vs. Frozen Cells
                                  (Fig. 5B movable/passive & Fig. 5C immovable/stuck)
  3. delayed_gratification.png  - Delayed Gratification vs. Frozen Cells (Fig. 7A-C)
  4. chimeric_aggregation.png   - Algotype Aggregation: Unique Values (Fig. 8A)
                                  vs. Duplicate Values (Fig. 8D-E)
  5. cross_purpose_chimeras.png - Opposite-Direction Chimeras: Sortedness &
                                  Aggregation (Figs. 9 & 10; Bubble > Selection > Insertion)
  6. kymographs.png             - Space-Time Trajectories (intact, damaged, chimeric)
  7. scheduler_comparison.png   - Random-Async vs. Parallel vs. OS-Threaded (Section 3.3)
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from sorting_morphogenesis import (
    AB_PRESETS,
    TRADITIONAL_ALGOS,
    Algotype,
    compare_lanes_ab,
    delayed_gratification,
    make_cells,
    monotonicity_error,
    simulate,
    simulate_parallel,
    simulate_threaded,
    validate_claim_boundary_catalog,
)

ALGO_NAMES: Dict[Algotype, str] = {
    "B": "Bubble",
    "I": "Insertion",
    "S": "Selection",
}
COLORS = {
    "Traditional": "#d95f02",
    "Cell-view": "#1b9e77",
    "B": "#1f77b4",
    "I": "#2ca02c",
    "S": "#ff7f0e",
}


def random_values(n: int, rng: np.random.Generator) -> List[int]:
    """Random permutation of 1..n without duplicates (matching paper Section 4)."""
    vals = list(range(1, n + 1))
    rng.shuffle(vals)
    return vals


# ---------------------------------------------------------------------------
# 1. Sorting Efficiency: Swaps (Fig. 4A), Comparisons+Swaps (Fig. 4B), Chimeric (Fig. 8B)
# ---------------------------------------------------------------------------

def exp_efficiency(n: int, reps: int, seed: int, out_dir: Path) -> dict:
    rng = np.random.default_rng(seed)
    algos: List[Algotype] = ["B", "I", "S"]
    trad_swaps: Dict[Algotype, List[int]] = {a: [] for a in algos}
    cell_swaps: Dict[Algotype, List[int]] = {a: [] for a in algos}
    trad_comps: Dict[Algotype, List[int]] = {a: [] for a in algos}
    cell_comps: Dict[Algotype, List[int]] = {a: [] for a in algos}

    for r in range(reps):
        vals = random_values(n, rng)
        for a in algos:
            _, t_sw, t_cmp, _ = TRADITIONAL_ALGOS[a](vals)
            trad_swaps[a].append(t_sw)
            trad_comps[a].append(t_cmp)

            # Swaps (Fig. 4A) with author update() reset
            res_sw = simulate(
                make_cells(vals, [a] * n),
                seed=seed * 1000 + r * 7 + ord(a),
                selection_reset_on_swap=True,
            )
            cell_swaps[a].append(res_sw.swaps)

            # Comparisons + swaps (Fig. 4B): for Selection, keep_ideal avoids
            # redundant re-scans from 0 after each swap (matching Section 4.1 1.17x ratio)
            if a == "S":
                res_cmp = simulate(
                    make_cells(vals, [a] * n),
                    seed=seed * 1000 + r * 7 + ord(a),
                    selection_reset_on_swap=False,
                )
                cell_comps[a].append(res_cmp.comparisons)
            else:
                cell_comps[a].append(res_sw.comparisons)

    # Chimeric efficiency (Fig. 8B): 50/50 mixes B+I, B+S, I+S
    chim_pairs: List[Tuple[str, Algotype, Algotype]] = [
        ("B+I", "B", "I"),
        ("B+S", "B", "S"),
        ("I+S", "I", "S"),
    ]
    chim_swaps: Dict[str, List[int]] = {label: [] for label, _, _ in chim_pairs}
    half = n // 2
    for r in range(reps):
        vals = random_values(n, rng)
        for label, a1, a2 in chim_pairs:
            atypes: List[Algotype] = [a1] * half + [a2] * (n - half)
            rng.shuffle(atypes)
            res = simulate(
                make_cells(vals, atypes),
                seed=seed * 2000 + r * 11,
                selection_reset_on_swap=True,
            )
            chim_swaps[label].append(res.swaps)

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.6))
    x = np.arange(len(algos))
    w = 0.36

    # Panel A: Swaps only (Fig. 4A)
    ax = axes[0]
    t_means = [np.mean(trad_swaps[a]) for a in algos]
    t_stds = [np.std(trad_swaps[a]) for a in algos]
    c_means = [np.mean(cell_swaps[a]) for a in algos]
    c_stds = [np.std(cell_swaps[a]) for a in algos]
    b1 = ax.bar(x - w / 2, t_means, w, yerr=t_stds, capsize=4, label="Traditional", color=COLORS["Traditional"], alpha=0.88)
    b2 = ax.bar(x + w / 2, c_means, w, yerr=c_stds, capsize=4, label="Cell-view", color=COLORS["Cell-view"], alpha=0.88)
    for bar in list(b1) + list(b2):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(c_means) * 0.02, f"{bar.get_height():.0f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([ALGO_NAMES[a] for a in algos])
    ax.set_ylabel("Swaps to sort")
    ax.set_title(f"(A) Swaps Only (Fig. 4A, n={n})")
    ax.legend(fontsize=8.5)
    ax.grid(axis="y", alpha=0.3)

    # Panel B: Comparisons + Swaps (Fig. 4B)
    ax = axes[1]
    tc_means = [np.mean(trad_comps[a]) for a in algos]
    tc_stds = [np.std(trad_comps[a]) for a in algos]
    cc_means = [np.mean(cell_comps[a]) for a in algos]
    cc_stds = [np.std(cell_comps[a]) for a in algos]
    b1 = ax.bar(x - w / 2, tc_means, w, yerr=tc_stds, capsize=4, label="Traditional", color=COLORS["Traditional"], alpha=0.88)
    b2 = ax.bar(x + w / 2, cc_means, w, yerr=cc_stds, capsize=4, label="Cell-view", color=COLORS["Cell-view"], alpha=0.88)
    for bar in list(b1) + list(b2):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(tc_means) * 0.02, f"{bar.get_height():.0f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([ALGO_NAMES[a] for a in algos])
    ax.set_ylabel("Comparisons + Swaps")
    ax.set_title(f"(B) Comparisons + Swaps (Fig. 4B, n={n})")
    ax.legend(fontsize=8.5)
    ax.grid(axis="y", alpha=0.3)

    # Panel C: Chimeric Efficiency (Fig. 8B)
    ax = axes[2]
    labels_8b = ["Pure B", "Pure I", "Pure S", "B+I", "B+S", "I+S"]
    means_8b = [
        np.mean(cell_swaps["B"]),
        np.mean(cell_swaps["I"]),
        np.mean(cell_swaps["S"]),
        np.mean(chim_swaps["B+I"]),
        np.mean(chim_swaps["B+S"]),
        np.mean(chim_swaps["I+S"]),
    ]
    stds_8b = [
        np.std(cell_swaps["B"]),
        np.std(cell_swaps["I"]),
        np.std(cell_swaps["S"]),
        np.std(chim_swaps["B+I"]),
        np.std(chim_swaps["B+S"]),
        np.std(chim_swaps["I+S"]),
    ]
    cols_8b = ["#1f77b4", "#2ca02c", "#ff7f0e", "#17becf", "#9467bd", "#8c564b"]
    x8 = np.arange(len(labels_8b))
    bars = ax.bar(x8, means_8b, 0.58, yerr=stds_8b, capsize=4, color=cols_8b, alpha=0.88)
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(means_8b) * 0.02, f"{bar.get_height():.0f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x8)
    ax.set_xticklabels(labels_8b, fontsize=9)
    ax.set_ylabel("Swaps to sort")
    ax.set_title(f"(C) Pure vs. Chimeric Swaps (Fig. 8B, n={n})")
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_dir / "efficiency.png", dpi=160)
    plt.close(fig)

    return {
        "trad_swaps": {a: float(np.mean(trad_swaps[a])) for a in algos},
        "cell_swaps": {a: float(np.mean(cell_swaps[a])) for a in algos},
        "trad_comps": {a: float(np.mean(trad_comps[a])) for a in algos},
        "cell_comps": {a: float(np.mean(cell_comps[a])) for a in algos},
        "chimeric_swaps": {k: float(np.mean(v)) for k, v in chim_swaps.items()},
    }


# ---------------------------------------------------------------------------
# 2. Error Tolerance: Movable (Fig. 5B) & Immovable (Fig. 5C) Frozen Cells
# ---------------------------------------------------------------------------

def exp_error_tolerance(n: int, reps: int, seed: int, out_dir: Path) -> dict:
    rng = np.random.default_rng(seed + 11)
    algos: List[Algotype] = ["B", "I", "S"]
    f_counts = [1, 2, 3]
    results: dict = {}

    fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.6), sharey="row")

    for row, fkind in enumerate(["movable", "immovable"]):
        results[fkind] = {}
        for col, f in enumerate(f_counts):
            t_errs: Dict[Algotype, List[int]] = {a: [] for a in algos}
            c_errs: Dict[Algotype, List[int]] = {a: [] for a in algos}
            for r in range(reps):
                vals = random_values(n, rng)
                fidx = rng.choice(n, size=f, replace=False).tolist()
                for a in algos:
                    out_v, _, _, _ = TRADITIONAL_ALGOS[a](vals, fidx, fkind)  # type: ignore[arg-type]
                    t_errs[a].append(monotonicity_error(out_v))

                    cells = make_cells(vals, [a] * n, fidx, fkind)  # type: ignore[arg-type]
                    res = simulate(
                        cells,
                        stop_when_sorted=True,
                        seed=seed * 3000 + r * 13 + f * 3 + row,
                        selection_movable_mode=("pyc310" if fkind == "movable" else "standard"),
                    )
                    c_errs[a].append(res.final_error)

            results[fkind][f] = {
                "trad": {a: float(np.mean(t_errs[a])) for a in algos},
                "cell": {a: float(np.mean(c_errs[a])) for a in algos},
            }

            ax = axes[row, col]
            x = np.arange(len(algos))
            w = 0.36
            tm = [results[fkind][f]["trad"][a] for a in algos]
            ts = [float(np.std(t_errs[a])) for a in algos]
            cm = [results[fkind][f]["cell"][a] for a in algos]
            cs = [float(np.std(c_errs[a])) for a in algos]
            b1 = ax.bar(x - w / 2, tm, w, yerr=ts, capsize=3, label="Traditional", color=COLORS["Traditional"], alpha=0.88)
            b2 = ax.bar(x + w / 2, cm, w, yerr=cs, capsize=3, label="Cell-view", color=COLORS["Cell-view"], alpha=0.88)
            for bar in list(b1) + list(b2):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.4, f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=7.5)
            ax.set_xticks(x)
            ax.set_xticklabels([ALGO_NAMES[a] for a in algos])
            title_kind = "Movable / Passive (Fig. 5B)" if fkind == "movable" else "Immovable / Stuck (Fig. 5C)"
            ax.set_title(f"{title_kind} — f = {f}", fontsize=10)
            if col == 0:
                ax.set_ylabel("Final Monotonicity Error")
            ax.grid(axis="y", alpha=0.3)
            if row == 0 and col == 2:
                ax.legend(loc="upper left", fontsize=8.5)

    fig.suptitle(f"Error Tolerance to Frozen Cells (n={n}, {reps} reps)", fontsize=12, y=1.01)
    fig.tight_layout()
    fig.savefig(out_dir / "error_tolerance.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    return results


# ---------------------------------------------------------------------------
# 3. Delayed Gratification vs. Immovable Frozen Cells (Fig. 7A-C)
# ---------------------------------------------------------------------------

def exp_delayed_gratification(n: int, reps: int, seed: int, out_dir: Path) -> dict:
    rng = np.random.default_rng(seed + 23)
    algos: List[Algotype] = ["B", "I", "S"]
    f_counts = [0, 1, 2, 3]
    dg_trad: Dict[Algotype, Dict[int, List[float]]] = {a: {f: [] for f in f_counts} for a in algos}
    dg_cell: Dict[Algotype, Dict[int, List[float]]] = {a: {f: [] for f in f_counts} for a in algos}

    for f in f_counts:
        for r in range(reps):
            vals = random_values(n, rng)
            fidx = rng.choice(n, size=f, replace=False).tolist() if f > 0 else []
            for a in algos:
                _, _, _, t_err_trace = TRADITIONAL_ALGOS[a](vals, fidx, "immovable")
                dg_trad[a][f].append(delayed_gratification(t_err_trace, net=True))

                cells = make_cells(vals, [a] * n, fidx, "immovable")
                res = simulate(
                    cells,
                    stop_when_sorted=(f == 0),
                    seed=seed * 5000 + r * 17 + f * 5 + ord(a),
                    selection_reset_on_swap=True,
                )
                dg_cell[a][f].append(res.dg)

    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.4))
    x = np.arange(len(f_counts))
    w = 0.36
    summary: dict = {}
    for idx, a in enumerate(algos):
        ax = axes[idx]
        tm = [float(np.mean(dg_trad[a][f])) for f in f_counts]
        ts = [float(np.std(dg_trad[a][f])) for f in f_counts]
        cm = [float(np.mean(dg_cell[a][f])) for f in f_counts]
        cs = [float(np.std(dg_cell[a][f])) for f in f_counts]
        summary[a] = {"trad": dict(zip(f_counts, tm)), "cell": dict(zip(f_counts, cm))}
        b1 = ax.bar(x - w / 2, tm, w, yerr=ts, capsize=3, label="Traditional", color=COLORS["Traditional"], alpha=0.88)
        b2 = ax.bar(x + w / 2, cm, w, yerr=cs, capsize=3, label="Cell-view", color=COLORS["Cell-view"], alpha=0.88)
        for bar in list(b1) + list(b2):
            h = bar.get_height()
            if h > 0.005:
                ax.text(bar.get_x() + bar.get_width() / 2, h + max(cm + [0.1]) * 0.03, f"{h:.2f}", ha="center", va="bottom", fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels([f"f = {f}" for f in f_counts])
        ax.set_title(f"{ALGO_NAMES[a]} Sort (Fig. 7{'ABC'[idx]})")
        if idx == 0:
            ax.set_ylabel("Delayed Gratification (net ratio)")
        ax.grid(axis="y", alpha=0.3)
        ax.legend(fontsize=8.5)

    fig.suptitle(f"Delayed Gratification vs. Immovable Frozen Cells (n={n}, {reps} reps)", fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(out_dir / "delayed_gratification.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    return summary


# ---------------------------------------------------------------------------
# 4. Chimeric Aggregation: Unique Values (Fig. 8A) & Duplicate Values (Fig. 8D-E)
# ---------------------------------------------------------------------------

def exp_chimeric_aggregation(n: int, reps: int, seed: int, out_dir: Path) -> dict:
    rng = np.random.default_rng(seed + 41)
    pairs: List[Tuple[str, Tuple[Algotype, ...], str]] = [
        ("Bubble + Selection", ("B", "S"), "#1f77b4"),
        ("Insertion + Selection", ("I", "S"), "#ff7f0e"),
        ("Bubble + Insertion", ("B", "I"), "#2ca02c"),
        ("Bubble + Selection + Insertion", ("B", "S", "I"), "#d62728"),
    ]
    grid = np.linspace(0.0, 1.0, 101)
    summary: dict = {"unique": {}, "duplicate": {}}

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8))

    for panel_idx, (mode_key, mode_title) in enumerate([
        ("unique", f"(A) Unique Cell Values 1..{n} (Fig. 8A)"),
        ("duplicate", f"(B) Duplicate Values 1..10 (10 copies each, Fig. 8D)"),
    ]):
        ax = axes[panel_idx]
        for label, types, color in pairs:
            if mode_key == "duplicate" and len(types) == 3:
                continue  # Paper Fig. 8D plots the three pairwise chimeras
            curves = []
            peaks = []
            finals = []
            for r in range(reps):
                if mode_key == "unique":
                    vals = random_values(n, rng)
                else:
                    # 10 copies each of 1..10 (or n//10 copies of 1..10)
                    vals = [(i % 10) + 1 for i in range(n)]
                    rng.shuffle(vals)
                atypes: List[Algotype] = [types[i % len(types)] for i in range(n)]
                rng.shuffle(atypes)
                cells = make_cells(vals, atypes)
                res = simulate(
                    cells,
                    stop_when_sorted=True,
                    seed=seed * 7000 + r * 19 + panel_idx * 101,
                    selection_reset_on_swap=True,
                )
                tr = np.asarray(res.aggregation_trace, dtype=float)
                peaks.append(float(np.max(tr)))
                finals.append(float(tr[-1]))
                xp = np.linspace(0.0, 1.0, len(tr))
                curves.append(np.interp(grid, xp, tr))

            arr = np.vstack(curves)
            mean = arr.mean(axis=0)
            sem = arr.std(axis=0) / np.sqrt(len(curves))
            peak_idx = int(np.argmax(mean))
            summary[mode_key][label] = {
                "peak_curve": float(mean[peak_idx]),
                "peak_progress_pct": int(grid[peak_idx] * 100),
                "mean_run_peak": float(np.mean(peaks)),
                "final_agg": float(np.mean(finals)),
            }
            ax.plot(
                grid * 100,
                mean,
                label=f"{label} (peak {mean[peak_idx]:.2f} @ {int(grid[peak_idx]*100)}%, end {mean[-1]:.2f})",
                color=color,
                lw=2.0,
            )
            ax.fill_between(grid * 100, mean - sem, mean + sem, color=color, alpha=0.16)

        ax.axhline(0.5, color="gray", ls="--", lw=1.0, label="2-type random baseline (0.50)")
        if mode_key == "unique":
            ax.axhline(1.0 / 3.0, color="lightgray", ls=":", lw=1.0, label="3-type random baseline (0.33)")
        ax.set_xlabel("Sorting progress (%)")
        ax.set_ylabel("Aggregation value")
        ax.set_title(mode_title)
        ax.set_ylim(0.28, 0.80)
        ax.grid(alpha=0.3)
        ax.legend(loc="upper right", fontsize=8)

    fig.tight_layout()
    fig.savefig(out_dir / "chimeric_aggregation.png", dpi=160)
    plt.close(fig)
    return summary


# ---------------------------------------------------------------------------
# 5. Cross-Purpose Chimeras: Opposite Directions (Figs. 9 & 10)
# ---------------------------------------------------------------------------

def exp_cross_purpose_chimeras(n: int, reps: int, seed: int, out_dir: Path) -> dict:
    """
    Paper Section 4.4 / Figs. 9 & 10:
    50 cells of Algotype 1 sorting in one direction mixed with 50 cells of
    Algotype 2 sorting in the opposite direction, run for up to `max_swaps`
    (paper uses 15,000 swaps for n=100; scaled as `150 * n` swaps).
    """
    rng = np.random.default_rng(seed + 59)
    max_swaps = 150 * n
    configs: List[Tuple[str, Algotype, int, Algotype, int, str, float]] = [
        ("Bubble (dec) + Selection (inc)", "B", -1, "S", 1, "#1f77b4", 42.50),
        ("Bubble (inc) + Insertion (dec)", "B", 1, "I", -1, "#2ca02c", 73.73),
        ("Insertion (inc) + Selection (dec)", "I", 1, "S", -1, "#ff7f0e", 38.31),
    ]
    grid = np.linspace(0.0, 1.0, 101)
    summary: dict = {}

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8))
    cross_reps = min(10, max(6, reps // 2))

    for label, a1, d1, a2, d2, color, paper_pct in configs:
        srt_curves = []
        agg_curves = []
        final_srts = []
        final_aggs = []
        half = n // 2
        for r in range(cross_reps):
            vals = random_values(n, rng)
            spec = [(a1, d1)] * half + [(a2, d2)] * (n - half)
            rng.shuffle(spec)
            atypes = [x[0] for x in spec]
            dirs = [x[1] for x in spec]
            cells = make_cells(vals, atypes, directions=dirs)  # type: ignore[arg-type]
            res = simulate(
                cells,
                max_swaps=max_swaps,
                max_steps=max_swaps * 25,
                stop_when_sorted=False,
                seed=seed * 9000 + r * 23,
                selection_reset_on_swap=True,
            )
            st_tr = np.asarray(res.sortedness_trace, dtype=float) * 100.0
            ag_tr = np.asarray(res.aggregation_trace, dtype=float)
            final_srts.append(float(st_tr[-1]))
            final_aggs.append(float(ag_tr[-1]))
            xp = np.linspace(0.0, 1.0, len(st_tr))
            srt_curves.append(np.interp(grid, xp, st_tr))
            agg_curves.append(np.interp(grid, xp, ag_tr))

        s_arr = np.vstack(srt_curves)
        a_arr = np.vstack(agg_curves)
        s_mean, s_sem = s_arr.mean(axis=0), s_arr.std(axis=0) / np.sqrt(len(srt_curves))
        a_mean, a_sem = a_arr.mean(axis=0), a_arr.std(axis=0) / np.sqrt(len(agg_curves))
        summary[label] = {
            "final_sortedness_pct": float(np.mean(final_srts)),
            "paper_sortedness_pct": paper_pct,
            "final_agg": float(np.mean(final_aggs)),
        }

        axes[0].plot(
            grid * 100,
            s_mean,
            color=color,
            lw=2.0,
            label=f"{label}: {np.mean(final_srts):.1f}% (paper: {paper_pct:.1f}%)",
        )
        axes[0].fill_between(grid * 100, s_mean - s_sem, s_mean + s_sem, color=color, alpha=0.16)

        axes[1].plot(
            grid * 100,
            a_mean,
            color=color,
            lw=2.0,
            label=f"{label} (end {np.mean(final_aggs):.2f})",
        )
        axes[1].fill_between(grid * 100, a_mean - a_sem, a_mean + a_sem, color=color, alpha=0.16)

    axes[0].axhline(50.0, color="gray", ls="--", lw=1.0, label="Neutral sortedness (50%)")
    axes[0].set_xlabel("Simulation progress (%)")
    axes[0].set_ylabel("Sortedness (%)")
    axes[0].set_title("(A) Cross-Purpose Sortedness (Fig. 9: B > S > I)")
    axes[0].set_ylim(20, 85)
    axes[0].grid(alpha=0.3)
    axes[0].legend(loc="lower right", fontsize=8)

    axes[1].axhline(0.5, color="gray", ls="--", lw=1.0, label="Random baseline (0.50)")
    axes[1].set_xlabel("Simulation progress (%)")
    axes[1].set_ylabel("Aggregation value")
    axes[1].set_title("(B) Cross-Purpose Algotype Aggregation (Fig. 10)")
    axes[1].set_ylim(0.45, 1.0)
    axes[1].grid(alpha=0.3)
    axes[1].legend(loc="upper left", fontsize=8)

    fig.tight_layout()
    fig.savefig(out_dir / "cross_purpose_chimeras.png", dpi=160)
    plt.close(fig)
    return summary


# ---------------------------------------------------------------------------
# 6. Space-Time Kymographs (intact, damaged, and chimeric)
# ---------------------------------------------------------------------------

def exp_kymographs(n: int, seed: int, out_dir: Path) -> None:
    rng = np.random.default_rng(seed + 71)
    vals = list(range(n))
    rng.shuffle(vals)
    fidx = [n // 3, (2 * n) // 3]

    fig, axes = plt.subplots(2, 4, figsize=(15, 6.6))
    for col, a in enumerate(["B", "I", "S"]):
        res0 = simulate(make_cells(vals, [a] * n), seed=seed + col, record_snapshots=True)
        ax = axes[0, col]
        ax.imshow(res0.value_snapshots, aspect="auto", cmap="viridis", interpolation="nearest")
        ax.set_title(f"{ALGO_NAMES[a]} (no frozen)\nswaps={res0.swaps}, err={res0.final_error}")
        ax.set_xlabel("Array position")
        if col == 0:
            ax.set_ylabel("Swap step")

        resf = simulate(
            make_cells(vals, [a] * n, fidx, "immovable"),
            stop_when_sorted=False,
            seed=seed + 10 + col,
            record_snapshots=True,
        )
        ax2 = axes[1, col]
        ax2.imshow(resf.value_snapshots, aspect="auto", cmap="viridis", interpolation="nearest")
        for fi in fidx:
            ax2.axvline(fi, color="red", ls="--", lw=0.9, alpha=0.8)
        ax2.set_title(f"{ALGO_NAMES[a]} (2 immovable)\nerr={resf.final_error}, DG={resf.dg:.2f}")
        ax2.set_xlabel("Array position")
        if col == 0:
            ax2.set_ylabel("Swap step")

    types: List[Algotype] = ["B" if i % 2 == 0 else "S" for i in range(n)]
    rng.shuffle(types)
    resc = simulate(make_cells(vals, types), seed=seed + 99, record_snapshots=True)
    axes[0, 3].imshow(resc.value_snapshots, aspect="auto", cmap="viridis", interpolation="nearest")
    axes[0, 3].set_title(f"Chimera B+S (values)\nswaps={resc.swaps}")
    axes[0, 3].set_xlabel("Array position")

    axes[1, 3].imshow(resc.algotype_snapshots, aspect="auto", cmap="coolwarm", interpolation="nearest", vmin=0, vmax=2)
    axes[1, 3].set_title(f"Chimera B+S (algotypes)\npeak agg={max(resc.aggregation_trace):.2f}")
    axes[1, 3].set_xlabel("Array position")

    fig.tight_layout()
    fig.savefig(out_dir / "kymographs.png", dpi=160)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 7. Scheduler Comparison (Random-Async vs. Parallel vs. OS-Threaded)
# ---------------------------------------------------------------------------

def exp_scheduler_comparison(n: int, reps: int, seed: int, out_dir: Path) -> dict:
    rng = np.random.default_rng(seed + 101)
    algos: List[Algotype] = ["B", "I", "S"]
    modes = ["Async (random)", "Parallel (rounds)", "Threaded (locks)"]
    swaps: Dict[str, Dict[Algotype, List[int]]] = {m: {a: [] for a in algos} for m in modes}

    thread_reps = min(reps, 10)
    for r in range(reps):
        vals = random_values(n, rng)
        for a in algos:
            r_async = simulate(make_cells(vals, [a] * n), seed=seed * 900 + r)
            swaps["Async (random)"][a].append(r_async.swaps)

            r_par = simulate_parallel(make_cells(vals, [a] * n), seed=seed * 900 + r)
            swaps["Parallel (rounds)"][a].append(r_par.swaps)

            if r < thread_reps:
                r_thr = simulate_threaded(make_cells(vals, [a] * n), seed=seed * 900 + r)
                swaps["Threaded (locks)"][a].append(r_thr.swaps)

    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    x = np.arange(len(algos))
    w = 0.26
    palette = ["#1b9e77", "#7570b3", "#e7298a"]
    for idx, m in enumerate(modes):
        means = [np.mean(swaps[m][a]) for a in algos]
        stds = [np.std(swaps[m][a]) for a in algos]
        ax.bar(x + (idx - 1) * w, means, w, yerr=stds, capsize=3, label=m, color=palette[idx], alpha=0.88)
    ax.set_xticks(x)
    ax.set_xticklabels([ALGO_NAMES[a] for a in algos])
    ax.set_ylabel("Total swaps to reach sorted state")
    ax.set_title(f"Scheduler Invariance Across Execution Models (n={n})")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "scheduler_comparison.png", dpi=160)
    plt.close(fig)

    return {m: {a: float(np.mean(swaps[m][a])) for a in algos} for m in modes}


# ---------------------------------------------------------------------------
# 8. Deterministic Paired A/B Lane Comparison (MesmerPrism/combinatorial-swarmability)
# ---------------------------------------------------------------------------

def exp_paired_ab_comparison(n: int, reps: int, seed: int, out_dir: Path) -> dict:
    """
    Run all 3 paired A/B presets with locked initial value permutations and
    locked frozen-cell defect positions, saving `paired_ab_comparison.png` and
    `paired_ab_metrics.csv`.
    """
    import csv

    preset_keys = [
        "bubble_intact_vs_frozen3",
        "bubble_vs_selection_frozen3",
        "chimera_bs_unique_vs_duplicate",
    ]
    ab_results = {}
    for key in preset_keys:
        la, lb = AB_PRESETS[key]
        ab_results[key] = compare_lanes_ab(la, lb, n=n, reps=reps, seed=seed)

    # Save CSV summary
    csv_path = out_dir / "paired_ab_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow([
            "preset_id",
            "lane_a",
            "lane_b",
            "n",
            "reps",
            "seed",
            "swaps_a",
            "swaps_b",
            "delta_swaps_mean",
            "delta_swaps_sem",
            "final_error_a",
            "final_error_b",
            "delta_error_mean",
            "delta_error_sem",
            "dg_a",
            "dg_b",
            "delta_dg_mean",
            "delta_dg_sem",
            "peak_agg_a",
            "peak_agg_b",
            "final_agg_a",
            "final_agg_b",
            "delta_final_agg_mean",
            "delta_final_agg_sem",
        ])
        for key, ab in ab_results.items():
            writer.writerow([
                key,
                ab.lane_a_name,
                ab.lane_b_name,
                ab.n,
                ab.reps,
                ab.seed,
                f"{ab.mean_swaps_a:.2f}",
                f"{ab.mean_swaps_b:.2f}",
                f"{ab.delta_swaps_mean:+.2f}",
                f"{ab.delta_swaps_sem:.2f}",
                f"{ab.mean_final_error_a:.2f}",
                f"{ab.mean_final_error_b:.2f}",
                f"{ab.delta_final_error_mean:+.2f}",
                f"{ab.delta_final_error_sem:.2f}",
                f"{ab.mean_dg_a:.4f}",
                f"{ab.mean_dg_b:.4f}",
                f"{ab.delta_dg_mean:+.4f}",
                f"{ab.delta_dg_sem:.4f}",
                f"{ab.mean_peak_agg_a:.4f}",
                f"{ab.mean_peak_agg_b:.4f}",
                f"{ab.mean_final_agg_a:.4f}",
                f"{ab.mean_final_agg_b:.4f}",
                f"{ab.delta_final_agg_mean:+.4f}",
                f"{ab.delta_final_agg_sem:.4f}",
            ])

    # Render 2-row x 3-column paired A/B figure
    fig, axes = plt.subplots(2, 3, figsize=(16.5, 8.2))
    pct = ab_results[preset_keys[0]].progress_grid * 100.0

    # Column 0: Bubble Intact (f=0) vs. 3 Immovable Frozen (f=3)
    ab0 = ab_results["bubble_intact_vs_frozen3"]
    ax = axes[0, 0]
    ax.plot(pct, ab0.error_curve_a, color="#1b9e77", lw=2.2, label=f"{ab0.lane_a_name} (end={ab0.mean_final_error_a:.2f})")
    ax.plot(pct, ab0.error_curve_b, color="#d95f02", lw=2.2, label=f"{ab0.lane_b_name} (end={ab0.mean_final_error_b:.2f})")
    ax.set_title("(A1) Seed-Locked Monotonicity Error\nBubble Intact ($f=0$) vs. Damaged ($f=3$)", fontsize=9.5, fontweight="bold")
    ax.set_xlabel("Sorting progress (%)")
    ax.set_ylabel("Monotonicity error")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1, 0]
    ax.plot(pct, ab0.delta_error_curve, color="#7570b3", lw=2.2, label=f"Paired $\\Delta$Error ($B - A$, end={ab0.delta_final_error_mean:+.2f}±{ab0.delta_final_error_sem:.2f})")
    ax.fill_between(pct, ab0.delta_error_curve - ab0.delta_error_sem_curve, ab0.delta_error_curve + ab0.delta_error_sem_curve, color="#7570b3", alpha=0.22)
    ax.axhline(0.0, color="gray", ls="--", lw=1.0)
    ax.set_title(f"(A2) Paired Divergence $\\Delta$Error & $\\Delta$DG = {ab0.delta_dg_mean:+.3f}±{ab0.delta_dg_sem:.3f}", fontsize=9.2, fontweight="bold")
    ax.set_xlabel("Sorting progress (%)")
    ax.set_ylabel("Paired $\\Delta$ Monotonicity Error ($B - A$)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # Column 1: Bubble (f=3) vs. Selection (f=3) on identical defect indices
    ab1 = ab_results["bubble_vs_selection_frozen3"]
    ax = axes[0, 1]
    ax.plot(pct, ab1.error_curve_a, color="#1f77b4", lw=2.2, label=f"{ab1.lane_a_name} (end={ab1.mean_final_error_a:.2f})")
    ax.plot(pct, ab1.error_curve_b, color="#ff7f0e", lw=2.2, label=f"{ab1.lane_b_name} (end={ab1.mean_final_error_b:.2f})")
    ax.set_title("(B1) Same Frozen Defect Positions ($f=3$)\nBubble vs. Selection Error Trajectory", fontsize=9.5, fontweight="bold")
    ax.set_xlabel("Sorting progress (%)")
    ax.set_ylabel("Monotonicity error")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1, 1]
    ax.plot(pct, ab1.delta_error_curve, color="#e7298a", lw=2.2, label=f"Paired $\\Delta$Error ($S - B$, end={ab1.delta_final_error_mean:+.2f}±{ab1.delta_final_error_sem:.2f})")
    ax.fill_between(pct, ab1.delta_error_curve - ab1.delta_error_sem_curve, ab1.delta_error_curve + ab1.delta_error_sem_curve, color="#e7298a", alpha=0.22)
    ax.axhline(0.0, color="gray", ls="--", lw=1.0)
    ax.set_title(f"(B2) Selection Advantage on Identical Defects\nFinal $\\Delta$Error = {ab1.delta_final_error_mean:+.2f}±{ab1.delta_final_error_sem:.2f}", fontsize=9.2, fontweight="bold")
    ax.set_xlabel("Sorting progress (%)")
    ax.set_ylabel("Paired $\\Delta$ Monotonicity Error ($S - B$)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # Column 2: Chimera B+S Unique Values vs. Duplicate Values (1..10)
    ab2 = ab_results["chimera_bs_unique_vs_duplicate"]
    ax = axes[0, 2]
    ax.plot(pct, ab2.agg_curve_a, color="#2ca02c", lw=2.2, label=f"{ab2.lane_a_name} (end={ab2.mean_final_agg_a:.3f})")
    ax.plot(pct, ab2.agg_curve_b, color="#d62728", lw=2.2, label=f"{ab2.lane_b_name} (end={ab2.mean_final_agg_b:.3f})")
    ax.axhline(0.5, color="gray", ls=":", lw=1.0, label="Random 2-type baseline (0.50)")
    ax.set_title("(C1) Chimera B+S Algotype Aggregation\nUnique Values vs. Duplicate Values (Fig. 8A vs 8D)", fontsize=9.5, fontweight="bold")
    ax.set_xlabel("Sorting progress (%)")
    ax.set_ylabel("Algotype aggregation")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1, 2]
    ax.plot(pct, ab2.delta_agg_curve, color="#8c564b", lw=2.2, label=f"Paired $\\Delta$Aggregation (end={ab2.delta_final_agg_mean:+.3f}±{ab2.delta_final_agg_sem:.3f})")
    ax.fill_between(pct, ab2.delta_agg_curve - ab2.delta_agg_sem_curve, ab2.delta_agg_curve + ab2.delta_agg_sem_curve, color="#8c564b", alpha=0.22)
    ax.axhline(0.0, color="gray", ls="--", lw=1.0)
    ax.set_title(f"(C2) Persistent Duplicate-Value Segregation\nFinal $\\Delta$Agg = {ab2.delta_final_agg_mean:+.3f}±{ab2.delta_final_agg_sem:.3f}", fontsize=9.2, fontweight="bold")
    ax.set_xlabel("Sorting progress (%)")
    ax.set_ylabel("Paired $\\Delta$ Aggregation ($B - A$)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    fig.suptitle(
        f"Deterministic Paired A/B Lane Comparison — Zhang, Goldstein & Levin (2025) Cell-View Sorting Morphogenesis\n"
        f"(Seed-Locked Permutations & Defect Positions across {reps} Paired Replicates, n={n})",
        fontsize=12,
        fontweight="bold",
        y=1.01,
    )
    fig.tight_layout()
    fig.savefig(out_dir / "paired_ab_comparison.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    return ab_results


# ---------------------------------------------------------------------------
# Animated GIF generator (for docs/media/zhang_sorting_morphogenesis.gif)
# ---------------------------------------------------------------------------

def make_gif(output_path: Path, n: int = 36, seed: int = 42, n_frames: int = 48) -> None:
    """Render an animated GIF of Zhang, Goldstein & Levin (2025) cell-view sorting morphogenesis."""
    import io
    from PIL import Image

    rng = np.random.default_rng(seed)
    init_vals = list(rng.permutation(np.arange(1, n + 1)))
    frozen_pos = [n // 3, (2 * n) // 3]
    # Place deliberately out-of-order values at the two immovable frozen positions
    i_low = init_vals.index(4)
    init_vals[frozen_pos[0]], init_vals[i_low] = init_vals[i_low], init_vals[frozen_pos[0]]
    i_high = init_vals.index(n - 3)
    init_vals[frozen_pos[1]], init_vals[i_high] = init_vals[i_high], init_vals[frozen_pos[1]]

    res_bub = simulate(
        make_cells(init_vals, ["B"] * n, frozen_indices=frozen_pos, frozen_kind="immovable"),
        seed=seed + 1,
        record_snapshots=True,
    )
    res_sel = simulate(
        make_cells(init_vals, ["S"] * n, frozen_indices=frozen_pos, frozen_kind="immovable"),
        seed=seed + 2,
        record_snapshots=True,
    )

    # Chimeric array (50/50 Bubble + Selection with duplicate values 1..9, Paper Fig. 8D)
    rng_c = np.random.default_rng(seed + 7)
    dup_vals = list(rng_c.permutation(np.repeat(np.arange(1, 10), n // 9)))
    algos_c: List[Algotype] = ["B"] * (n // 2) + ["S"] * (n - n // 2)
    rng_c.shuffle(algos_c)
    res_chi = simulate(
        make_cells(dup_vals, algos_c),
        seed=seed + 11,
        record_snapshots=True,
    )

    # Ensemble mean curve for duplicate B+S chimera (for context ribbon)
    grid_x = np.linspace(0.0, 1.0, 101)
    ens_curves = []
    for r in range(12):
        rc = np.random.default_rng(seed + 100 + r)
        dv = list(rc.permutation(np.repeat(np.arange(1, 10), n // 9)))
        al: List[Algotype] = ["B"] * (n // 2) + ["S"] * (n - n // 2)
        rc.shuffle(al)
        rr = simulate(make_cells(dv, al), seed=int(rc.integers(0, 1_000_000)))
        xp = np.linspace(0.0, 1.0, len(rr.aggregation_trace))
        ens_curves.append(np.interp(grid_x, xp, rr.aggregation_trace))
    ens_mean = np.mean(ens_curves, axis=0)
    ens_sd = np.std(ens_curves, axis=0)

    bub_snaps = res_bub.value_snapshots
    sel_snaps = res_sel.value_snapshots
    chi_val_snaps = res_chi.value_snapshots
    chi_alg_snaps = res_chi.algotype_snapshots
    assert bub_snaps is not None and sel_snaps is not None
    assert chi_val_snaps is not None and chi_alg_snaps is not None

    bub_errs = [monotonicity_error(row) for row in bub_snaps]
    sel_errs = [monotonicity_error(row) for row in sel_snaps]

    cmap = plt.cm.viridis
    frames_pil: List[Image.Image] = []

    for f_idx in range(n_frames):
        frac = f_idx / (n_frames - 1)
        ib = min(int(round(frac * (len(bub_snaps) - 1))), len(bub_snaps) - 1)
        is_ = min(int(round(frac * (len(sel_snaps) - 1))), len(sel_snaps) - 1)
        ic = min(int(round(frac * (len(chi_val_snaps) - 1))), len(chi_val_snaps) - 1)

        fig = plt.figure(figsize=(7.2, 4.8), dpi=100, facecolor="#fafaf8")
        gs = fig.add_gridspec(
            2, 2, width_ratios=[1.25, 1.0], height_ratios=[1.0, 1.0],
            hspace=0.44, wspace=0.28, left=0.07, right=0.97, top=0.87, bottom=0.10
        )
        fig.suptitle(
            f"Zhang, Goldstein & Levin (2025) — Autonomous Cell-View Sorting Morphogenesis   (progress {int(round(frac * 100)):3d}%)",
            fontsize=9.3, fontweight="bold", color="#1e293b", y=0.96,
        )

        # Panel A: Immovable frozen defects (Bubble vs Selection bar arrays)
        ax_a = fig.add_subplot(gs[0, 0])
        vals_b = bub_snaps[ib]
        vals_s = sel_snaps[is_]
        for i_pos in range(n):
            c_s = cmap((vals_s[i_pos] - 1) / max(1, n - 1))
            c_b = cmap((vals_b[i_pos] - 1) / max(1, n - 1))
            is_f = i_pos in frozen_pos
            ax_a.bar(
                i_pos, 0.41 * (vals_s[i_pos] / n), bottom=0.54, width=0.86,
                color=c_s, edgecolor="#dc2626" if is_f else "none", linewidth=1.3 if is_f else 0.0,
            )
            ax_a.bar(
                i_pos, 0.41 * (vals_b[i_pos] / n), bottom=0.04, width=0.86,
                color=c_b, edgecolor="#dc2626" if is_f else "none", linewidth=1.3 if is_f else 0.0,
            )
        ax_a.axhline(0.50, color="#94a3b8", lw=0.8, ls="--")
        ax_a.text(0.5, 0.97, f"Selection (err={sel_errs[is_]}) — jumps frozen defects",
                  transform=ax_a.transAxes, fontsize=7.0, fontweight="bold", color="#b45309", va="top", ha="center")
        ax_a.text(0.5, 0.47, f"Bubble (err={bub_errs[ib]}) — blocked by immovable cells (red)",
                  transform=ax_a.transAxes, fontsize=7.0, fontweight="bold", color="#1d4ed8", va="top", ha="center")
        ax_a.set_xlim(-0.8, n - 0.2)
        ax_a.set_ylim(0.0, 1.06)
        ax_a.set_yticks([])
        ax_a.tick_params(labelsize=6.8)
        ax_a.set_xlabel("Array position (n=36, f=2 immovable frozen cells)", fontsize=7.0)
        ax_a.set_title("A. Error Tolerance Around Frozen Cells (Fig. 5C)", fontsize=7.8, fontweight="bold")

        # Panel B: Chimeric array (values + algotype strip underneath)
        ax_b = fig.add_subplot(gs[1, 0])
        vals_c = chi_val_snaps[ic]
        alg_c = chi_alg_snaps[ic]
        for i_pos in range(n):
            col_v = cmap((vals_c[i_pos] - 1) / 8.0)
            ax_b.bar(i_pos, 0.72 * (vals_c[i_pos] / 9.0), bottom=0.22, width=0.86, color=col_v)
            col_a = "#2563eb" if alg_c[i_pos] == 0 else "#ea580c"
            ax_b.bar(i_pos, 0.16, bottom=0.02, width=0.92, color=col_a)
        ax_b.set_xlim(-0.8, n - 0.2)
        ax_b.set_ylim(0.0, 1.02)
        ax_b.set_yticks([0.10, 0.60])
        ax_b.set_yticklabels(["Algotype\n(B / S)", "Value\n(1..9)"], fontsize=6.6)
        ax_b.tick_params(labelsize=6.8)
        ax_b.set_xlabel("Array position (blue = Bubble, orange = Selection)", fontsize=7.0)
        ax_b.set_title(
            f"B. Chimeric Sorting & Algotype Segregation (agg={res_chi.aggregation_trace[ic]:.2f})",
            fontsize=7.8, fontweight="bold",
        )

        # Panel C: Monotonicity Error & Delayed Gratification (top-right)
        ax_c = fig.add_subplot(gs[0, 1])
        tb = np.linspace(0, 100, len(bub_errs))
        ts = np.linspace(0, 100, len(sel_errs))
        ax_c.plot(tb, bub_errs, color="#93c5fd", lw=1.0)
        ax_c.plot(ts, sel_errs, color="#fcd34d", lw=1.0)
        ax_c.plot(tb[: ib + 1], bub_errs[: ib + 1], color="#1d4ed8", lw=2.0, label=f"Bubble (DG={res_bub.dg:.2f})")
        ax_c.plot(ts[: is_ + 1], sel_errs[: is_ + 1], color="#d97706", lw=2.0, label=f"Selection (err→{res_sel.final_error})")
        ax_c.scatter([tb[ib]], [bub_errs[ib]], color="#1d4ed8", s=22, zorder=5)
        ax_c.scatter([ts[is_]], [sel_errs[is_]], color="#d97706", s=22, zorder=5)
        ax_c.set_xlim(0, 100)
        ax_c.set_ylim(0, max(max(bub_errs), max(sel_errs)) + 2)
        ax_c.set_xlabel("Sorting progress (%)", fontsize=7.0)
        ax_c.set_ylabel("Monotonicity error", fontsize=7.0)
        ax_c.set_title("C. Error & Delayed Gratification (Figs. 6–7)", fontsize=7.8, fontweight="bold")
        ax_c.legend(fontsize=6.3, loc="upper right", framealpha=0.85)
        ax_c.tick_params(labelsize=6.8)
        ax_c.grid(alpha=0.25)

        # Panel D: Chimeric Aggregation Trajectory (bottom-right)
        ax_d = fig.add_subplot(gs[1, 1])
        ax_d.fill_between(grid_x * 100, ens_mean - ens_sd, ens_mean + ens_sd, color="#fdba74", alpha=0.30)
        ax_d.plot(grid_x * 100, ens_mean, color="#9a3412", lw=1.2, ls=":", label="Ensemble mean")
        tc = np.linspace(0, 100, len(res_chi.aggregation_trace))
        ax_d.plot(tc, res_chi.aggregation_trace, color="#fed7aa", lw=1.0)
        ax_d.plot(tc[: ic + 1], res_chi.aggregation_trace[: ic + 1], color="#ea580c", lw=2.2, label="Active run B+S")
        ax_d.scatter([tc[ic]], [res_chi.aggregation_trace[ic]], color="#ea580c", s=22, zorder=5)
        ax_d.axhline(0.50, color="#64748b", ls="--", lw=0.9, label="Random (0.50)")
        ax_d.set_xlim(0, 100)
        ax_d.set_ylim(0.42, 0.85)
        ax_d.set_xlabel("Sorting progress (%)", fontsize=7.0)
        ax_d.set_ylabel("Algotype aggregation", fontsize=7.0)
        ax_d.set_title("D. Spontaneous Algotype Clustering (Fig. 8D)", fontsize=7.8, fontweight="bold")
        ax_d.legend(fontsize=6.3, loc="lower right", framealpha=0.85)
        ax_d.tick_params(labelsize=6.8)
        ax_d.grid(alpha=0.25)

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=100)
        plt.close(fig)
        buf.seek(0)
        frames_pil.append(Image.open(buf).convert("RGB"))

    # Quantize to a shared 128-color palette for compact README media size
    w, h = frames_pil[0].size
    strip_img = Image.new("RGB", (w, h * 4))
    for i, fr in enumerate(frames_pil[:: max(1, len(frames_pil) // 4)][:4]):
        strip_img.paste(fr, (0, i * h))
    pal = strip_img.quantize(colors=128, method=Image.MEDIANCUT)
    q_frames = [fr.quantize(palette=pal, dither=Image.FLOYDSTEINBERG) for fr in frames_pil]
    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    durations = [110] * (len(q_frames) - 1) + [900]
    q_frames[0].save(
        out_p, save_all=True, append_images=q_frames[1:], duration=durations, loop=0, optimize=True
    )
    print(f"Saved sorting-morphogenesis animation to {out_p} ({w}x{h}, {len(q_frames)} frames, {out_p.stat().st_size // 1024} KB)")


# ---------------------------------------------------------------------------
# Main CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Replicate Zhang, Goldstein & Levin (2025) sorting morphogenesis.")
    parser.add_argument("--n", type=int, default=50, help="Array length (paper uses 100; default 50 for fast runs)")
    parser.add_argument("--reps", type=int, default=30, help="Replicates per condition (paper uses 100)")
    parser.add_argument("--seed", type=int, default=42, help="Base RNG seed")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parent / "figures",
        help="Output directory for figures",
    )
    parser.add_argument(
        "--compare-ab",
        action="store_true",
        help="Run only the deterministic paired A/B comparison suite and validate catalog.json",
    )
    parser.add_argument(
        "--gif",
        type=Path,
        default=None,
        help="Optional path to write animated GIF (e.g., docs/media/zhang_sorting_morphogenesis.gif)",
    )
    parser.add_argument(
        "--gif-only",
        action="store_true",
        help="Only render the animated GIF and exit",
    )
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    if args.gif is not None:
        make_gif(args.gif, seed=args.seed)
        if args.gif_only:
            return

    cat_ok, cat_errs = validate_claim_boundary_catalog()
    print(f"[{'PASS' if cat_ok else 'FAIL'}] Claim-boundary catalog (sorting-morphogenesis/catalog.json): {len(cat_errs)} error(s)")
    if not cat_ok:
        for e in cat_errs:
            print(f"  - {e}")
        raise SystemExit(1)

    if args.compare_ab:
        ab_res = exp_paired_ab_comparison(args.n, args.reps, args.seed, args.out)
        for key, ab in ab_res.items():
            print(
                f"  [{key}] {ab.lane_a_name} vs {ab.lane_b_name}: "
                f"Δerr={ab.delta_final_error_mean:+.2f}±{ab.delta_final_error_sem:.2f}, "
                f"ΔDG={ab.delta_dg_mean:+.3f}±{ab.delta_dg_sem:.3f}, "
                f"Δend_agg={ab.delta_final_agg_mean:+.3f}±{ab.delta_final_agg_sem:.3f}"
            )
        print(f"Saved paired A/B figure to {args.out / 'paired_ab_comparison.png'} and CSV to {args.out / 'paired_ab_metrics.csv'}")
        return

    print(f"Running Zhang, Goldstein & Levin (2025) replication (n={args.n}, reps={args.reps}, seed={args.seed})...")

    eff = exp_efficiency(args.n, args.reps, args.seed, args.out)
    print("\n[1] Efficiency (Paper Section 4.1 & Fig. 4A/4B/8B):")
    print(f"    Swaps (Traditional):     { {k: round(v, 1) for k, v in eff['trad_swaps'].items()} }")
    print(f"    Swaps (Cell-view):       { {k: round(v, 1) for k, v in eff['cell_swaps'].items()} }")
    print(f"    Selection swap ratio (Cell / Trad): {eff['cell_swaps']['S'] / max(1.0, eff['trad_swaps']['S']):.2f}x (paper: ~11x for n=100)")
    print(f"    Comps+Swaps (Trad):      { {k: round(v, 1) for k, v in eff['trad_comps'].items()} }")
    print(f"    Comps+Swaps (Cell-view): { {k: round(v, 1) for k, v in eff['cell_comps'].items()} }")
    print(
        f"    Total-step ratios (Trad/Cell for B, I; Cell/Trad for S): "
        f"B={eff['trad_comps']['B']/eff['cell_comps']['B']:.2f}x (paper: 1.50x), "
        f"I={eff['trad_comps']['I']/eff['cell_comps']['I']:.2f}x (paper: 2.03x), "
        f"S={eff['cell_comps']['S']/eff['trad_comps']['S']:.2f}x (paper: 1.17x)"
    )
    print(f"    Chimeric Swaps (Fig 8B): { {k: round(v, 1) for k, v in eff['chimeric_swaps'].items()} }")

    tol = exp_error_tolerance(args.n, args.reps, args.seed, args.out)
    print("\n[2] Error Tolerance — Final Monotonicity Error (Paper Section 4.2 & Fig. 5B/5C):")
    for fkind in ["movable", "immovable"]:
        print(f"  {fkind.upper()}:")
        for a in ["B", "I", "S"]:
            t_str = ", ".join(f"f={f}:{tol[fkind][f]['trad'][a]:.2f}" for f in [1, 2, 3])
            c_str = ", ".join(f"f={f}:{tol[fkind][f]['cell'][a]:.2f}" for f in [1, 2, 3])
            print(f"    {ALGO_NAMES[a]:9s} | Trad ({t_str}) vs Cell-view ({c_str})")

    dg = exp_delayed_gratification(args.n, args.reps, args.seed, args.out)
    print("\n[3] Delayed Gratification vs. Immovable Frozen Cells (Paper Section 4.3 & Fig. 7A-C):")
    for a in ["B", "I", "S"]:
        t_str = ", ".join(f"f={f}:{dg[a]['trad'][f]:.2f}" for f in [0, 1, 2, 3])
        c_str = ", ".join(f"f={f}:{dg[a]['cell'][f]:.2f}" for f in [0, 1, 2, 3])
        print(f"    {ALGO_NAMES[a]:9s} | Trad ({t_str}) vs Cell-view ({c_str})")

    agg = exp_chimeric_aggregation(args.n, args.reps, args.seed, args.out)
    print("\n[4] Chimeric Aggregation (Paper Section 4.4 & Fig. 8A/8D):")
    for mode_key in ["unique", "duplicate"]:
        print(f"  {mode_key.upper()} VALUES:")
        for label, st in agg[mode_key].items():
            print(
                f"    {label:30s}: curve_peak={st['peak_curve']:.3f} @ {st['peak_progress_pct']}%, "
                f"run_max={st['mean_run_peak']:.3f}, final={st['final_agg']:.3f}"
            )

    cross = exp_cross_purpose_chimeras(args.n, args.reps, args.seed, args.out)
    print("\n[5] Cross-Purpose Opposite-Direction Chimeras (Paper Section 4.4 & Figs. 9-10):")
    for label, st in cross.items():
        print(
            f"    {label:32s}: final_sortedness={st['final_sortedness_pct']:.1f}% "
            f"(paper: {st['paper_sortedness_pct']:.1f}%), final_agg={st['final_agg']:.3f}"
        )

    exp_kymographs(args.n, args.seed, args.out)
    sched = exp_scheduler_comparison(min(args.n, 30), min(args.reps, 10), args.seed, args.out)
    print("\n[6] Scheduler Comparison (mean swaps):")
    for m, d in sched.items():
        print(f"    {m:18s}: { {k: round(v, 1) for k, v in d.items()} }")

    ab_res = exp_paired_ab_comparison(args.n, args.reps, args.seed, args.out)
    print("\n[7] Deterministic Paired A/B Lane Comparison (MesmerPrism/combinatorial-swarmability):")
    for key, ab in ab_res.items():
        print(
            f"    {key:30s}: Δerr={ab.delta_final_error_mean:+.2f}±{ab.delta_final_error_sem:.2f}, "
            f"ΔDG={ab.delta_dg_mean:+.3f}±{ab.delta_dg_sem:.3f}, "
            f"Δend_agg={ab.delta_final_agg_mean:+.3f}±{ab.delta_final_agg_sem:.3f}"
        )

    print(f"\nAll 8 figures written to {args.out}")


if __name__ == "__main__":
    main()

# Replication & Multi-Language Port Guide — Zhang, Goldstein & Levin (2025)

This document explains **where and how every mechanism, metric, and figure** of:

> **Zhang, T., Goldstein, A., & Levin, M. (2025).**
> *Classical sorting algorithms as a model of morphogenesis: Self-sorting arrays reveal unexpected competencies in a minimal model of basal intelligence.*
> **Adaptive Behavior**, 33(1), 25–54.
> [doi:10.1177/10597123241269740](https://doi.org/10.1177/10597123241269740) · [arXiv:2401.05375](https://arxiv.org/abs/2401.05375)
> Authors' Upstream Code: [`Zhangtaining/cell_research`](https://github.com/Zhangtaining/cell_research) at pinned commit [`1fd2bd5921c1f6b423a71f691d5189106a8a1020`](https://github.com/Zhangtaining/cell_research/tree/1fd2bd5921c1f6b423a71f691d5189106a8a1020)

was duplicated and independently verified across **four languages**:

1. **C++20 (`clang++ -std=c++20 -O3`):** [`sorting-morphogenesis/cpp/zhang_sorting_morphogenesis.cpp`](./cpp/zhang_sorting_morphogenesis.cpp)
2. **JavaScript / Node.js (ES Module):** [`sorting-morphogenesis/js/zhang_sorting_morphogenesis.mjs`](./js/zhang_sorting_morphogenesis.mjs)
3. **Python 3:** [`sorting-morphogenesis/sorting_morphogenesis.py`](./sorting_morphogenesis.py) & [`sorting-morphogenesis/validate_kymograph_and_paper.py`](./validate_kymograph_and_paper.py)
4. **Lean 4 Formal Verification (`0 sorry`):** [`morphoeval/lean/loops/Loop_zhang_sorting_morphogenesis.lean`](../morphoeval/lean/loops/Loop_zhang_sorting_morphogenesis.lean)

![Multi-Language Port Verification & Upstream Code Traceability](../docs/media/zhang_multilang_port_verification.png)

---

## 1. One-Command Cross-Language Verification

Run all four languages (`C++20`, `Node.js`, `Python 3`, and `Lean 4`) and verify bit-exact `SplitMix64` state-by-state agreement:

```bash
# 1. Compile and run C++20 port directly (~0.8s for n=100, 15 replicates across all 14 benchmarks)
clang++ -std=c++20 -O3 sorting-morphogenesis/cpp/zhang_sorting_morphogenesis.cpp -o sorting-morphogenesis/cpp/zhang_sorting_cpp
./sorting-morphogenesis/cpp/zhang_sorting_cpp --n 100 --reps 15 --seed 42

# 2. Run JavaScript / Node.js ES-module port directly
node sorting-morphogenesis/js/zhang_sorting_morphogenesis.mjs --n 100 --reps 15 --seed 42

# 3. Verify Lean 4 formal proofs (0 errors, 0 sorry)
lean morphoeval/lean/loops/Loop_zhang_sorting_morphogenesis.lean

# 4. Run the unified cross-language verifier & regenerate the 6-panel synthesis figure
./venv/bin/python3 sorting-morphogenesis/verify_multilang_ports.py
```

---

## 2. Exact Upstream Code Traceability (`Zhangtaining/cell_research@1fd2bd59`)

Every behavioral subtlety in the paper's published plots traces back to six specific implementation details in the authors' reference codebase [`Zhangtaining/cell_research`](https://github.com/Zhangtaining/cell_research):

| ID | Paper Figure & Phenomenon | Upstream File & Lines (`cell_research@1fd2bd59`) | How the Upstream Code Works | Where It Is Ported Here |
| :--- | :--- | :--- | :--- | :--- |
| **`[U1]`** | **Bubble Cell-View Efficiency & Immovable Pinning** (`Fig. 4A, 4B, 5C`) | `modules/multithread/BubbleSortCell.py` (`should_move` L38–52, `move` L54–74) | Active Bubble cell at `p` checks whether `p-1` or `p+1` is out of order and not `FREEZE` (`immovable`). When scheduled, it picks left or right with $p = 0.5$ and swaps if out of order. Because it only swaps $(p, p\pm 1)$, an immovable frozen cell at $d$ permanently partitions the array into two non-communicating halves (`inv = 241`). | • **C++20:** `cell_should_move()` (L181–188), `simulate_cell_view()` (L275–290)<br>• **Node.js:** `computeActiveIndices()` (L178–189), `simulateCellView()` (L261–278)<br>• **Python:** `_cell_should_move()` (L220–231), `_propose()` (L270–286)<br>• **Lean 4:** `bubble_adjacent_swap_cannot_cross_immovable_defect` |
| **`[U2]`** | **Insertion Prefix-Reset Across Frozen Defects** (`Fig. 4A, 5B, 5C, 7B`) | `modules/multithread/InsertionSortCell.py` (`is_enable_to_move` L68–82, `move` L84–102) | Insertion cell at `p` only swaps with `p-1` if the prefix `0 .. p-1` is already sorted. Crucially, at L74–76, encountering any `CellStatus.FREEZE` cell in `0 .. p-1` **resets the running comparison bound `prev = -1` (`-∞`)** and skips the frozen cell. Without this reset, a single high-valued frozen defect would permanently halt all Insertion cells to its right. | • **C++20:** `left_sorted_with_frozen_reset()` (L152–178)<br>• **Node.js:** `leftSortedWithFrozenReset()` (L143–169)<br>• **Python:** `_left_sorted()` (L179–207)<br>• **Lean 4:** `insertion_frozen_prefix_reset_unblocks_downstream` |
| **`[U3]`** | **Selection `ideal_position` Pointer & `update()` Reset on Swap** (`Fig. 4A, 4B, 5C`) | `modules/multithread/SelectionSortCell.py` (`should_move` L41–46, `move` L55–88) & `MultiThreadCell.py` (`swap` L104–118) | Each Selection cell maintains an internal pointer `ideal_position` (`target`, initialized to `0` for increasing or `n-1` for decreasing). If `value < cells[target].value`, it swaps directly with `target` across arbitrary distance—tunneling past immovable frozen defects (`inv = 0`, `Fig. 5C` final error $\approx f$). Inside `MultiThreadCell.swap()`, `update()` resets `ideal_position = 0` on both swapped cells, producing the paper's `~11.06×` swap ratio (`Fig. 4A`); keeping `ideal_position` (`selection_reset_on_swap=False`) produces the paper's `1.17×` total-step ratio (`Fig. 4B`). | • **C++20:** `simulate_cell_view()` (L297–331)<br>• **Node.js:** `simulateCellView()` (L286–322)<br>• **Python:** `_propose()` (L299–326), `simulate()` (L503–515)<br>• **Lean 4:** `selection_nonlocal_swap_eliminates_cross_defect_inversion` |
| **`[U4]`** | **Movable (Passive) Frozen-Cell Bytecode Rule** (`Fig. 5B`) | `modules/multithread/__pycache__/SelectionSortCell.cpython-310.pyc` | In the compiled `.pyc` used for `Fig. 5B`, when a Selection cell's `target` lands on a passive (`movable`) frozen cell, it increments `ideal_position` unless `t == occ.target`, leaving passive frozen cells with residual error `~1, ~2, ~3` for $f=1,2,3$. | • **Python:** `_propose(..., selection_movable_mode="pyc310")` (L307–315) |
| **`[U5]`** | **Delayed Gratification Discrepancy-Run Segmentation** (`Fig. 6D & Fig. 7A–C`) | `analysis/delay_gratification_analysis_for_not_move.py` (L24–89) | Consecutive identical values in the monotonicity-error trace are deduplicated; first differences $\Delta e_t = e_t - e_{t-1}$ of the same sign are summed into alternating setback ($y > 0$) and recovery ($x > 0$) runs, and $\mathrm{DG} = \langle (x - y)/y \rangle$ (`net=True`, `Fig. 7`) or $\langle x/y \rangle$ (`net=False`, `Fig. 6D`). | • **C++20:** `delayed_gratification()` (L102–146)<br>• **Node.js:** `delayedGratification()` (L95–139)<br>• **Python:** `_discrepancy_runs()` (L92–117), `delayed_gratification()` (L120–158)<br>• **Lean 4:** `delayed_gratification_strict_positive_when_recovery_exceeds_setback` |
| **`[U6]`** | **Chimeric Algotype Aggregation & Cross-Purpose Hierarchy** (`Fig. 8A, 8D–E, 9, 10`) | `analysis/cell_type_aggregation_analysis.py` (`build_neighbor_checking_arr` L518–538) | Measures the fraction of cells whose right neighbor shares the same algotype: $A = \frac{1}{n}\sum_{i=0}^{n-2} \mathbf{1}[\text{algo}_i = \text{algo}_{i+1}]$. On unique values (`Fig. 8A`), $A$ peaks transiently at `0.64–0.71` mid-sort before relaxing to `0.50`; on duplicate values `1..10` (`Fig. 8D`), equal-valued cells remain permanently segregated by algotype (`0.692`). | • **C++20:** `aggregation_value()` (L93–99)<br>• **Node.js:** `aggregationValue()` (L86–92)<br>• **Python:** `aggregation_value()` (L160–173)<br>• **Lean 4:** `chimeric_segregation_maximizes_same_neighbors` |

---

## 3. Quantitative Parity Across Published Paper, C++20, Node.js, and Python ($n = 100$)

Because [`zhang_sorting_morphogenesis.cpp`](./cpp/zhang_sorting_morphogenesis.cpp), [`zhang_sorting_morphogenesis.mjs`](./js/zhang_sorting_morphogenesis.mjs), and [`verify_multilang_ports.py`](./verify_multilang_ports.py) share the exact same 64-bit `SplitMix64` PRNG, their trajectories and ensemble averages on seed `42` are **bit-for-bit identical** (`0` cell mismatches):

| Paper Section & Figure | Benchmark Metric ($n = 100$) | Published Paper (2025) | C++20 Port (`clang++ -O3`) | Node.js Port (`node`) | Python 3 (`validate_kymograph_and_paper.py`) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **§4.1 Fig. 4A** | Bubble Sort Mean Swaps | `2,448.8` | `2,480.8` | `2,480.8` | `2,461.4` |
| **§4.1 Fig. 4A** | Insertion Sort Mean Swaps | `2,482.8` | `2,480.8` | `2,480.8` | `2,461.4` |
| **§4.1 Fig. 4A** | Selection Swap Ratio (Cell / Trad) | `11.06×` | `11.0042×` | `11.0042×` | `11.09×` |
| **§4.1 Fig. 4B** | Bubble Total Steps Speedup (Trad / Cell) | `1.50×` | `1.5393×` | `1.5393×` | `1.56×` |
| **§4.1 Fig. 4B** | Insertion Total Steps Speedup (Trad / Cell) | `2.03×` | `2.0381×` | `2.0381×` | `2.04×` |
| **§4.1 Fig. 4B** | Selection Total Steps Ratio (Cell / Trad) | `1.17×` | `1.1735×` | `1.1735×` | `1.18×` |
| **§4.2 Fig. 5C** | Selection Final Error ($f = 3$ Immovable) | `~3.00` | `3.0000` | `3.0000` | `2.93` |
| **§4.2 Fig. 5C** | Bubble Final Error ($f = 3$ Immovable) | `~5.50` | `5.6000` | `5.6000` | `5.47` |
| **§4.3 Fig. 7A** | Bubble Delayed Gratification ($f = 3$) | `0.37` (Trad `<0.25`) | `0.5581` | `0.5581` | `0.496` |
| **§4.3 Fig. 7B** | Insertion Delayed Gratification ($f = 0..3$) | `1.10–1.19` | `1.2309` | `1.2309` | `1.125` |
| **§4.4 Fig. 8A** | Unique-Value Chimera (`B+S`) Peak Aggregation | `0.64–0.70` | `0.7127` | `0.7127` | `0.635` (curve) / `0.70` (run) |
| **§4.4 Fig. 8D** | Duplicate-Value Chimera (`B+S`) Final Aggregation | `0.65–0.71` | `0.6920` | `0.6920` | `0.688` |
| **§4.4 Fig. 9** | Cross-Purpose `B(inc)+I(dec)` Sortedness | `73.73%` | `75.47%` | `75.47%` | `75.32%` |
| **§4.4 Fig. 9** | Cross-Purpose `B(dec)+S(inc)` Sortedness | `42.50%` | `47.33%` | `47.33%` | `47.47%` |
| **Kymograph ($n=42, f=2$)** | Selection vs. Bubble Non-Frozen Inversions | `0` vs. `>200` | **`0` vs. `241`** (`0` diff) | **`0` vs. `241`** (`0` diff) | **`0` vs. `241`** (`0` diff) |

---

## 4. Evaluation of Subsequent "Latent Platonic Space" Claims (`Ingressing Minds`, 2025/2026)

See **[`PLATONIC_SPACE_CLAIMS_EVALUATION.md`](./PLATONIC_SPACE_CLAIMS_EVALUATION.md)**, [`audit_platonic_space_claims.py`](./audit_platonic_space_claims.py), our compiled 10-page standalone manuscript (**[`docs/zhang_levin_sorting_morphogenesis_manuscript.pdf`](../docs/zhang_levin_sorting_morphogenesis_manuscript.pdf)**), and our **Publication Strategy, Target Venues & Candidate Co-Authors Guide** (**[`docs/zhang_levin_publication_and_coauthors.md`](../docs/zhang_levin_publication_and_coauthors.md)**) for our 3-ablation mechanistic audit of the philosophical & empirical conclusions drawn when citing this paper in *Ingressing Minds* (PsyArXiv `doi:10.31234/osf.io/5g2xj_v3` / *Philosophies* 2026, `11(5):161`, Section 6.3 & Figure 14):

![Evaluating Michael Levin's Latent Platonic Space Claims on Sorting Morphogenesis](../docs/media/zhang_levin_platonic_claims_audit.png)


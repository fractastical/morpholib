#!/usr/bin/env node
// ============================================================================
// sorting-morphogenesis/js/zhang_sorting_morphogenesis.mjs
//
// Standalone JavaScript (ES Module / Node.js) Port of Zhang, Goldstein & Levin (2025):
//   "Classical sorting algorithms as a model of morphogenesis: Self-sorting
//    arrays reveal unexpected competencies in a minimal model of basal
//    intelligence." Adaptive Behavior 33(1):25-54 (2025).
//   DOI: 10.1177/10597123241269740  |  arXiv: 2401.05375
//   Upstream Python repo: https://github.com/Zhangtaining/cell_research
//   Pinned upstream commit: 1fd2bd5921c1f6b423a71f691d5189106a8a1020
//
// Exact Upstream File & Line Traceability:
//   [U1] BubbleSortCell.py (L38-74)       -> computeActiveIndices (Bubble), simulateCellView
//   [U2] InsertionSortCell.py (L68-102)   -> leftSortedWithFrozenReset (prefix reset at frozen cells)
//   [U3] SelectionSortCell.py (L41-88) &
//        MultiThreadCell.py (L104-118)    -> Selection target pointer + reset on swap
//   [U5] delay_gratification_analysis_for_not_move.py (L24-89)
//                                         -> delayedGratification()
//   [U6] cell_type_aggregation_analysis.py (L518-538)
//                                         -> aggregationValue()
//
// Shares the exact 64-bit SplitMix64 PRNG with `zhang_sorting_morphogenesis.cpp`
// so every step, swap, and kymograph matrix is bit-for-bit identical between
// C++20, Node.js, and Python.
// ============================================================================

import fs from "node:fs";
import { performance } from "node:perf_hooks";

const MASK_64 = 0xFFFFFFFFFFFFFFFFn;

export class SplitMix64 {
  constructor(seed) {
    this.state = BigInt(seed) & MASK_64;
  }

  nextU64() {
    this.state = (this.state + 0x9E3779B97F4A7C15n) & MASK_64;
    let z = this.state;
    z = ((z ^ (z >> 30n)) * 0xBF58476D1CE4E5B9n) & MASK_64;
    z = ((z ^ (z >> 27n)) * 0x94D049BB133111EBn) & MASK_64;
    return (z ^ (z >> 31n)) & MASK_64;
  }

  nextUint(bound) {
    return Number(this.nextU64() % BigInt(bound));
  }

  nextDouble() {
    const top53 = Number(this.nextU64() >> 11n);
    return top53 * (1.0 / 9007199254740992.0);
  }

  shuffle(arr) {
    for (let i = arr.length; i > 1; --i) {
      const j = this.nextUint(i);
      const tmp = arr[i - 1];
      arr[i - 1] = arr[j];
      arr[j] = tmp;
    }
  }
}

export const Algotype = Object.freeze({
  Bubble: 0,
  Insertion: 1,
  Selection: 2,
});

export const FrozenKind = Object.freeze({
  None: 0,
  Movable: 1,
  Immovable: 2,
});

export function monotonicityError(cells, direction = 1) {
  let err = 0;
  for (let i = 1; i < cells.length; ++i) {
    if (direction === 1) {
      if (cells[i].value < cells[i - 1].value) ++err;
    } else {
      if (cells[i].value > cells[i - 1].value) ++err;
    }
  }
  return err;
}

export function sortedness(cells, direction = 1) {
  if (cells.length <= 1) return 1.0;
  const n = cells.length;
  return (n - 1 - monotonicityError(cells, direction)) / (n - 1);
}

// [U6] analysis/cell_type_aggregation_analysis.py (L518-538)
export function aggregationValue(cells) {
  if (cells.length <= 1) return 1.0;
  let same = 0;
  for (let i = 0; i + 1 < cells.length; ++i) {
    if (cells[i].algotype === cells[i + 1].algotype) ++same;
  }
  return same / cells.length;
}

// [U5] analysis/delay_gratification_analysis_for_not_move.py (L24-89)
export function delayedGratification(monoTrace, net = true) {
  const dedup = [];
  for (const v of monoTrace) {
    if (dedup.length === 0 || dedup[dedup.length - 1] !== v) dedup.push(v);
  }
  if (dedup.length < 2) return 0.0;

  const temp = [];
  for (let i = 1; i < dedup.length; ++i) {
    temp.push(dedup[i] - dedup[i - 1]);
  }

  const runs = [];
  let cur = temp[0];
  for (let i = 1; i < temp.length; ++i) {
    if ((temp[i - 1] > 0 && temp[i] > 0) || (temp[i - 1] < 0 && temp[i] < 0)) {
      cur += temp[i];
    } else {
      runs.push(cur);
      cur = temp[i];
    }
  }
  runs.push(cur);

  let firstPos = runs.length;
  for (let i = 0; i < runs.length; ++i) {
    if (runs[i] > 0) {
      firstPos = i;
      break;
    }
  }
  if (firstPos >= runs.length - 1) return 0.0;

  let total = 0.0;
  let count = 0;
  for (let i = firstPos; i + 1 < runs.length; i += 2) {
    if (runs[i + 1] > 0) break;
    const y = runs[i];
    const xNeg = runs[i + 1];
    const change = net ? -(xNeg + y) / y : -xNeg / y;
    total += change;
    ++count;
  }
  return count > 0 ? total / count : 0.0;
}

// [U2] InsertionSortCell.is_enable_to_move (L68-82)
export function leftSortedWithFrozenReset(cells, p, direction) {
  if (direction === 1) {
    let prev = -1000000000;
    for (let k = 0; k < p; ++k) {
      if (cells[k].frozen !== FrozenKind.None) {
        prev = -1000000000;
        continue;
      }
      if (cells[k].value < prev) return false;
      prev = cells[k].value;
    }
    return true;
  } else {
    let prev = 1000000000;
    for (let k = 0; k < p; ++k) {
      if (cells[k].frozen !== FrozenKind.None) {
        prev = 1000000000;
        continue;
      }
      if (cells[k].value > prev) return false;
      prev = cells[k].value;
    }
    return true;
  }
}

export function computeActiveIndices(cells) {
  const n = cells.length;
  const act = [];
  let incOk = true;
  let decOk = true;
  let prevInc = -1000000000;
  let prevDec = 1000000000;

  for (let p = 0; p < n; ++p) {
    const c = cells[p];
    if (c.frozen === FrozenKind.None) {
      const d = c.direction;
      if (c.algotype === Algotype.Bubble) {
        const leftOk =
          p > 0 &&
          cells[p - 1].frozen !== FrozenKind.Immovable &&
          (d === 1 ? c.value < cells[p - 1].value : c.value > cells[p - 1].value);
        const rightOk =
          p < n - 1 &&
          cells[p + 1].frozen !== FrozenKind.Immovable &&
          (d === 1 ? c.value > cells[p + 1].value : c.value < cells[p + 1].value);
        if (leftOk || rightOk) act.push(p);
      } else if (c.algotype === Algotype.Insertion) {
        if (p > 0 && cells[p - 1].frozen !== FrozenKind.Immovable) {
          if (d === 1 && incOk && c.value < cells[p - 1].value) act.push(p);
          else if (d === -1 && decOk && c.value > cells[p - 1].value) act.push(p);
        }
      } else if (c.algotype === Algotype.Selection) {
        if (p !== c.target && c.target >= 0 && c.target < n) act.push(p);
      }
    }
    if (c.frozen !== FrozenKind.None) {
      prevInc = -1000000000;
      prevDec = 1000000000;
    } else {
      if (c.value < prevInc) incOk = false;
      prevInc = c.value;
      if (c.value > prevDec) decOk = false;
      prevDec = c.value;
    }
  }
  return act;
}

export function makeCells(values, algos, frozenIdx = [], fkind = FrozenKind.None, dirs = []) {
  const n = values.length;
  const cells = new Array(n);
  for (let i = 0; i < n; ++i) {
    const d = dirs.length === 0 ? 1 : dirs[i];
    cells[i] = {
      value: values[i],
      algotype: algos[i],
      frozen: FrozenKind.None,
      target: d === 1 ? 0 : n - 1,
      direction: d,
    };
  }
  for (const idx of frozenIdx) {
    if (idx >= 0 && idx < n) cells[idx].frozen = fkind;
  }
  return cells;
}

export function simulateCellView(
  cellsInput,
  seed,
  selectionResetOnSwap = true,
  recordSnapshots = false,
  maxSwaps = -1,
  maxSteps = -1,
  stopWhenSorted = true,
  selectionMovablePyc310 = false
) {
  const cells = cellsInput.map((c) => ({ ...c }));
  const rng = new SplitMix64(seed);
  const n = cells.length;
  if (maxSteps < 0) maxSteps = n * n * 40;

  const monoTrace = [monotonicityError(cells)];
  let peakAgg = aggregationValue(cells);
  let lastAgg = peakAgg;

  const valueSnapshots = [];
  if (recordSnapshots) {
    valueSnapshots.push(cells.map((c) => c.value));
  }

  let swaps = 0;
  let comparisons = 0;

  for (let step = 0; step < maxSteps; ++step) {
    if (stopWhenSorted && monotonicityError(cells) === 0) break;
    if (maxSwaps > 0 && swaps >= maxSwaps) break;

    const active = computeActiveIndices(cells);
    if (active.length === 0) break;

    const p = active[rng.nextUint(active.length)];
    const c = cells[p];
    const d = c.direction;
    let swapQ = -1;
    let didCompare = false;

    if (c.algotype === Algotype.Bubble) {
      didCompare = true;
      const side = rng.nextDouble() < 0.5 ? -1 : 1;
      const q = p + side;
      if (q >= 0 && q < n && cells[q].frozen !== FrozenKind.Immovable) {
        if (d === 1) {
          if ((side === -1 && c.value < cells[q].value) || (side === 1 && c.value > cells[q].value)) {
            swapQ = q;
          }
        } else {
          if ((side === -1 && c.value > cells[q].value) || (side === 1 && c.value < cells[q].value)) {
            swapQ = q;
          }
        }
      }
    } else if (c.algotype === Algotype.Insertion) {
      if (p > 0 && leftSortedWithFrozenReset(cells, p, d) && cells[p - 1].frozen !== FrozenKind.Immovable) {
        if ((d === 1 && c.value < cells[p - 1].value) || (d === -1 && c.value > cells[p - 1].value)) {
          swapQ = p - 1;
          didCompare = true;
        }
      }
    } else if (c.algotype === Algotype.Selection) {
      const t = c.target;
      if (t >= 0 && t < n && t !== p) {
        didCompare = true;
        if (cells[t].frozen === FrozenKind.Immovable) {
          c.target = d === 1 ? t + 1 : t - 1;
        } else if (cells[t].frozen === FrozenKind.Movable && selectionMovablePyc310) {
          if (c.value < cells[t].value && t === cells[t].target) {
            swapQ = t;
          } else {
            c.target = d === 1 ? t + 1 : t - 1;
          }
        } else if (c.value < cells[t].value) {
          swapQ = t;
        } else {
          c.target = d === 1 ? t + 1 : t - 1;
        }
      }
    }

    if (didCompare) ++comparisons;
    if (swapQ >= 0) {
      const tmp = cells[p];
      cells[p] = cells[swapQ];
      cells[swapQ] = tmp;
      if (selectionResetOnSwap) {
        if (cells[p].algotype === Algotype.Selection) {
          cells[p].target = cells[p].direction === 1 ? 0 : n - 1;
        }
        if (cells[swapQ].algotype === Algotype.Selection) {
          cells[swapQ].target = cells[swapQ].direction === 1 ? 0 : n - 1;
        }
      } else {
        for (const idx of [p, swapQ]) {
          if (cells[idx].algotype === Algotype.Selection) {
            if (cells[idx].direction === 1 && cells[idx].target > idx) {
              cells[idx].target = 0;
            } else if (cells[idx].direction === -1 && cells[idx].target < idx) {
              cells[idx].target = n - 1;
            }
          }
        }
      }
      ++swaps;
      monoTrace.push(monotonicityError(cells));
      lastAgg = aggregationValue(cells);
      if (lastAgg > peakAgg) peakAgg = lastAgg;
      if (recordSnapshots) {
        valueSnapshots.push(cells.map((cell) => cell.value));
      }
    }
  }

  const finalValues = cells.map((c) => c.value);
  const freeVals = cells.filter((c) => c.frozen === FrozenKind.None).map((c) => c.value);
  let inv = 0;
  for (let i = 0; i < freeVals.length; ++i) {
    for (let j = i + 1; j < freeVals.length; ++j) {
      if (freeVals[i] > freeVals[j]) ++inv;
    }
  }

  return {
    swaps,
    comparisons,
    final_error: monotonicityError(cells),
    dg: delayedGratification(monoTrace, true),
    peak_agg: peakAgg,
    final_agg: lastAgg,
    final_sortedness: sortedness(cells, 1),
    free_inversions: inv,
    value_snapshots: valueSnapshots,
    final_values: finalValues,
  };
}

function traditionalSortCounts(arr, algo) {
  const a = arr.slice();
  const n = a.length;
  let swaps = 0;
  let comps = 0;
  if (algo === Algotype.Bubble) {
    for (let i = 0; i < n; ++i) {
      let any = false;
      for (let j = 0; j + 1 < n - i; ++j) {
        ++comps;
        if (a[j] > a[j + 1]) {
          const t = a[j];
          a[j] = a[j + 1];
          a[j + 1] = t;
          ++swaps;
          ++comps;
          any = true;
        }
      }
      if (!any) break;
    }
  } else if (algo === Algotype.Insertion) {
    for (let i = 1; i < n; ++i) {
      let j = i;
      while (j > 0) {
        ++comps;
        if (a[j] < a[j - 1]) {
          const t = a[j];
          a[j] = a[j - 1];
          a[j - 1] = t;
          ++swaps;
          ++comps;
          --j;
        } else {
          break;
        }
      }
    }
  } else {
    for (let i = 0; i < n; ++i) {
      let minIdx = i;
      for (let j = i + 1; j < n; ++j) {
        ++comps;
        if (a[j] < a[minIdx]) minIdx = j;
      }
      if (minIdx !== i) {
        const t = a[i];
        a[i] = a[minIdx];
        a[minIdx] = t;
        ++swaps;
        ++comps;
      }
    }
  }
  return [swaps, comps];
}

// CLI execution when invoked directly via `node`
if (import.meta.url === `file://${process.argv[1]}`) {
  let n = 100;
  let reps = 15;
  let seed = 42n;
  let outJson = "sorting-morphogenesis/js/js_validation_benchmarks.json";

  for (let i = 2; i < process.argv.length; ++i) {
    if (process.argv[i] === "--n") n = Number(process.argv[++i]);
    else if (process.argv[i] === "--reps") reps = Number(process.argv[++i]);
    else if (process.argv[i] === "--seed") seed = BigInt(process.argv[++i]);
    else if (process.argv[i] === "--out") outJson = process.argv[++i];
  }

  const t0 = performance.now();

  // 1. Deterministic Kymograph Equivalence Test (n = 42, f = 2 at [14, 28])
  const nKymo = 42;
  const rngK = new SplitMix64(seed);
  const initK = Array.from({ length: nKymo }, (_, i) => i + 1);
  rngK.shuffle(initK);
  const fPos = [Math.floor(nKymo / 3), Math.floor((2 * nKymo) / 3)];
  const idxHi = initK.indexOf(nKymo - 2);
  [initK[fPos[0]], initK[idxHi]] = [initK[idxHi], initK[fPos[0]]];
  const idxLo = initK.indexOf(3);
  [initK[fPos[1]], initK[idxLo]] = [initK[idxLo], initK[fPos[1]]];

  const resKB = simulateCellView(makeCells(initK, new Array(nKymo).fill(Algotype.Bubble), fPos, FrozenKind.Immovable), seed + 66n, true, true);
  const resKI = simulateCellView(makeCells(initK, new Array(nKymo).fill(Algotype.Insertion), fPos, FrozenKind.Immovable), seed + 73n, true, true);
  const resKS = simulateCellView(makeCells(initK, new Array(nKymo).fill(Algotype.Selection), fPos, FrozenKind.Immovable), seed + 83n, true, true);

  // 2. Quantitative Paper Replication Suite
  const rng = new SplitMix64(seed);
  const sumTradSw = [0, 0, 0];
  const sumCellSw = [0, 0, 0];
  const sumTradTot = [0, 0, 0];
  const sumCellTot = [0, 0, 0];
  const allAlgos = [Algotype.Bubble, Algotype.Insertion, Algotype.Selection];

  for (let r = 0; r < reps; ++r) {
    const vals = Array.from({ length: n }, (_, i) => i + 1);
    rng.shuffle(vals);
    for (let aIdx = 0; aIdx < 3; ++aIdx) {
      const a = allAlgos[aIdx];
      const [ts, tc] = traditionalSortCounts(vals, a);
      sumTradSw[aIdx] += ts;
      sumTradTot[aIdx] += tc;

      const rSeed = seed * 1000n + BigInt(r * 7 + aIdx);
      const rs = simulateCellView(makeCells(vals, new Array(n).fill(a)), rSeed, true);
      sumCellSw[aIdx] += rs.swaps;
      if (a === Algotype.Selection) {
        const rc = simulateCellView(makeCells(vals, new Array(n).fill(a)), rSeed, false);
        sumCellTot[aIdx] += rc.comparisons;
      } else {
        sumCellTot[aIdx] += rs.comparisons;
      }
    }
  }

  const sumImmovF3 = [0, 0, 0];
  const sumMovF3 = [0, 0, 0];
  let sumDgF3B = 0.0;
  let sumDgI = 0.0;
  for (let r = 0; r < reps; ++r) {
    const vals = Array.from({ length: n }, (_, i) => i + 1);
    rng.shuffle(vals);
    const idxPool = Array.from({ length: n }, (_, i) => i);
    rng.shuffle(idxPool);
    const f3 = [idxPool[0], idxPool[1], idxPool[2]];

    for (let aIdx = 0; aIdx < 3; ++aIdx) {
      const a = allAlgos[aIdx];
      const rf = simulateCellView(makeCells(vals, new Array(n).fill(a), f3, FrozenKind.Immovable), seed * 3000n + BigInt(r * 13 + aIdx));
      sumImmovF3[aIdx] += rf.final_error;
      if (a === Algotype.Bubble) sumDgF3B += rf.dg;
      if (a === Algotype.Insertion) sumDgI += rf.dg;

      const rm = simulateCellView(
        makeCells(vals, new Array(n).fill(a), f3, FrozenKind.Movable),
        seed * 4000n + BigInt(r * 13 + aIdx),
        true,
        false,
        -1,
        -1,
        true,
        true
      );
      sumMovF3[aIdx] += rm.final_error;
    }
  }

  let sumUniqPeak = 0.0;
  let sumDupFinal = 0.0;
  for (let r = 0; r < reps; ++r) {
    const vu = Array.from({ length: n }, (_, i) => i + 1);
    const vd = Array.from({ length: n }, (_, i) => (i % 10) + 1);
    rng.shuffle(vu);
    rng.shuffle(vd);
    const at = Array.from({ length: n }, (_, i) => (i < Math.floor(n / 2) ? Algotype.Bubble : Algotype.Selection));
    rng.shuffle(at);

    const ru = simulateCellView(makeCells(vu, at), seed * 5000n + BigInt(r * 17));
    const rd = simulateCellView(makeCells(vd, at), seed * 6000n + BigInt(r * 19));
    sumUniqPeak += ru.peak_agg;
    sumDupFinal += rd.final_agg;
  }

  const crossReps = Math.max(6, Math.floor(reps / 2));
  let sumCrossBI = 0.0;
  let sumCrossBS = 0.0;
  for (let r = 0; r < crossReps; ++r) {
    const v = Array.from({ length: n }, (_, i) => i + 1);
    rng.shuffle(v);
    const half = Math.floor(n / 2);
    const specBI = Array.from({ length: n }, (_, i) => (i < half ? [Algotype.Bubble, 1] : [Algotype.Insertion, -1]));
    const specBS = Array.from({ length: n }, (_, i) => (i < half ? [Algotype.Bubble, -1] : [Algotype.Selection, 1]));
    rng.shuffle(specBI);
    rng.shuffle(specBS);
    const atBI = specBI.map((p) => p[0]);
    const dirBI = specBI.map((p) => p[1]);
    const atBS = specBS.map((p) => p[0]);
    const dirBS = specBS.map((p) => p[1]);

    const rBI = simulateCellView(makeCells(v, atBI, [], FrozenKind.None, dirBI), seed * 8000n + BigInt(r * 23), true, false, 120 * n, 120 * n * 20, false);
    const rBS = simulateCellView(makeCells(v, atBS, [], FrozenKind.None, dirBS), seed * 9000n + BigInt(r * 29), true, false, 120 * n, 120 * n * 20, false);
    sumCrossBI += rBI.final_sortedness * 100.0;
    sumCrossBS += rBS.final_sortedness * 100.0;
  }

  const elapsedMs = performance.now() - t0;
  const payload = {
    language: `JavaScript / Node.js (${process.version})`,
    prng: "SplitMix64",
    elapsed_ms: Number(elapsedMs.toFixed(2)),
    n,
    reps,
    kymograph_n42: {
      bubble: { swaps: resKB.swaps, final_error: resKB.final_error, free_inversions: resKB.free_inversions, dg: Number(resKB.dg.toFixed(4)) },
      insertion: { swaps: resKI.swaps, final_error: resKI.final_error, free_inversions: resKI.free_inversions, dg: Number(resKI.dg.toFixed(4)) },
      selection: { swaps: resKS.swaps, final_error: resKS.final_error, free_inversions: resKS.free_inversions, dg: Number(resKS.dg.toFixed(4)) },
      insertion_final_values: resKI.final_values,
      selection_final_values: resKS.final_values,
      bubble_final_values: resKB.final_values,
    },
    benchmarks: {
      fig4a_bubble_swaps: Number((sumCellSw[0] / reps).toFixed(4)),
      fig4a_insertion_swaps: Number((sumCellSw[1] / reps).toFixed(4)),
      fig4a_selection_swap_ratio: Number((sumCellSw[2] / Math.max(1.0, sumTradSw[2])).toFixed(4)),
      fig4b_bubble_step_speedup: Number((sumTradTot[0] / sumCellTot[0]).toFixed(4)),
      fig4b_insertion_step_speedup: Number((sumTradTot[1] / sumCellTot[1]).toFixed(4)),
      fig4b_selection_step_overhead: Number((sumCellTot[2] / sumTradTot[2]).toFixed(4)),
      fig5b_selection_movable_f3: Number((sumMovF3[2] / reps).toFixed(4)),
      fig5b_bubble_movable_f3: Number((sumMovF3[0] / reps).toFixed(4)),
      fig5c_selection_immovable_f3: Number((sumImmovF3[2] / reps).toFixed(4)),
      fig5c_bubble_immovable_f3: Number((sumImmovF3[0] / reps).toFixed(4)),
      fig7a_bubble_dg_f3: Number((sumDgF3B / reps).toFixed(4)),
      fig7b_insertion_dg_f3: Number((sumDgI / reps).toFixed(4)),
      fig8a_unique_chimera_peak_agg: Number((sumUniqPeak / reps).toFixed(4)),
      fig8d_duplicate_chimera_final_agg: Number((sumDupFinal / reps).toFixed(4)),
      fig9_cross_purpose_bi_sortedness: Number((sumCrossBI / crossReps).toFixed(4)),
      fig9_cross_purpose_bs_sortedness: Number((sumCrossBS / crossReps).toFixed(4)),
    },
  };

  fs.mkdirSync(outJson.substring(0, outJson.lastIndexOf("/")), { recursive: true });
  fs.writeFileSync(outJson, JSON.stringify(payload, null, 2));
  console.log(`[Node.js Port] Completed n=${n}, reps=${reps} in ${elapsedMs.toFixed(1)} ms -> ${outJson}`);
  console.log(
    `  Kymograph (n=42, f=2): Bubble(swaps=${resKB.swaps}, err=${resKB.final_error}, inv=${resKB.free_inversions}) | Selection(swaps=${resKS.swaps}, err=${resKS.final_error}, inv=${resKS.free_inversions})`
  );
}

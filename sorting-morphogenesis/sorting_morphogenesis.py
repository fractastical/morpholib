"""
Cell-view sorting morphogenesis — replication of Zhang, Goldstein & Levin (2025).

Reference paper:
    Zhang, T., Goldstein, A., & Levin, M. (2025).
    "Classical sorting algorithms as a model of morphogenesis: Self-sorting
    arrays reveal unexpected competencies in a minimal model of basal
    intelligence." Adaptive Behavior, 33(1), 25-54.
    https://doi.org/10.1177/10597123241269740 (preprint: arXiv:2401.05375)

Authors' reference implementation:
    https://github.com/Zhangtaining/cell_research
    Pinned commit SHA: 1fd2bd5921c1f6b423a71f691d5189106a8a1020 (2024-10-30)

Key implementation details verified against Zhangtaining/cell_research:
1. Insertion sort prefix check (`InsertionSortCell.is_enable_to_move`, L68-82):
   When scanning `0 .. p - 1` to verify the left prefix is sorted, frozen cells
   (`CellStatus.FREEZE`) reset the running comparison value (`prev = -1` for
   increasing, `+inf` for decreasing) and are skipped. Without this rule, a
   single out-of-order frozen cell permanently blocks all Insertion cells to
   its right.
2. Selection sort target pointer (`SelectionSortCell.py` + `MultiThreadCell.swap`):
   - `should_move()` is `current_position != ideal_position`. When a cell lands
     at `p == target`, it occupies its target slot and does not increment
     `target` unless displaced or outvoted (`value >= cells[target].value`).
   - In `MultiThreadCell.swap`, `update()` resets `ideal_position` to
     `left_boundary` (`0`, or `n - 1` for decreasing) when `reset_on_swap=True`,
     producing ~1,085 swaps on n=100 (paper: 1,095.5) and Delayed Gratification
     ~3.8 (paper: ~3.9). Setting `reset_on_swap=False` keeps displaced cells'
     scan progress and yields ~5,970 total comparisons+swaps on n=100 (1.18x
     traditional Selection's 5,049 steps; paper Section 4.1: 1.17x).
   - In the earlier bytecode (`SelectionSortCell.cpython-310.pyc`) used for the
     passive/movable frozen-cell plot (Paper Fig. 5B), encountering a `FREEZE`
     cell at `target` incremented `ideal_position` before swapping if
     `value < cells[target].value`, which leaves passive frozen cells only
     partially sorted (`selection_movable_mode="pyc310"`).
3. Delayed Gratification (`analysis/delay_gratification_analysis_for_not_move.py`):
   Consecutive non-zero changes in monotonicity error are grouped into
   alternating runs `d_arr` (`d_arr[i] > 0` = contiguous error increase `y`,
   followed by `d_arr[i+1] < 0` = contiguous error decrease `x`). Each
   setback-recovery pair contributes `(x - y) / y` (net gain per unit setback,
   used in Fig. 7) or `x / y` (gross ratio, shown in Fig. 6D).
4. Aggregation Value (`analysis/cell_type_aggregation_analysis.py`, L518-538):
   Fraction of cells whose adjacent neighbor has the same algotype (`0.5` for a
   random 50/50 two-algotype array; `0.33` for a random three-algotype array).
"""

from __future__ import annotations

import random
import threading
import time
from dataclasses import dataclass
from typing import Callable, List, Literal, Optional, Sequence, Tuple

import numpy as np

Algotype = Literal["B", "I", "S"]
FrozenKind = Optional[Literal["movable", "immovable"]]
SortDirection = Literal[1, -1]  # +1 = increasing, -1 = decreasing


@dataclass
class Cell:
    value: int
    algotype: Algotype
    frozen: FrozenKind = None
    target: int = 0  # Selection-cell ideal_position pointer
    direction: SortDirection = 1  # +1 increasing, -1 decreasing
    uid: int = 0


# ---------------------------------------------------------------------------
# Metrics (Paper Section 3.4 & Zhangtaining/cell_research/analysis/)
# ---------------------------------------------------------------------------

def monotonicity_error(values: Sequence[int], direction: SortDirection = 1) -> int:
    """Count adjacent violations of the target ordering."""
    if direction == 1:
        return sum(1 for i in range(1, len(values)) if values[i] < values[i - 1])
    return sum(1 for i in range(1, len(values)) if values[i] > values[i - 1])


def sortedness(values: Sequence[int], direction: SortDirection = 1) -> float:
    """Fraction of adjacent pairs in non-decreasing (or non-increasing) order."""
    n = len(values)
    if n <= 1:
        return 1.0
    return (n - 1 - monotonicity_error(values, direction=direction)) / (n - 1)


def _discrepancy_runs(mono_trace: Sequence[int]) -> List[int]:
    """
    Exact discrepancy-run segmentation from
    `Zhangtaining/cell_research/analysis/delay_gratification_analysis_for_not_move.py`:
    1. Deduplicate consecutive identical monotonicity-error values.
    2. Compute first differences `temp[i] = arr[i] - arr[i-1]`.
    3. Sum consecutive differences of the same sign into alternating runs.
    """
    dedup: List[int] = []
    for x in mono_trace:
        if not dedup or dedup[-1] != x:
            dedup.append(int(x))
    if len(dedup) < 2:
        return [0]
    temp = [dedup[i] - dedup[i - 1] for i in range(1, len(dedup))]
    res: List[int] = []
    x = temp[0]
    for i in range(1, len(temp)):
        if temp[i - 1] * temp[i] > 0:
            x += temp[i]
        else:
            res.append(x)
            x = temp[i]
    res.append(x)
    return res


def delayed_gratification(
    mono_trace: Sequence[int],
    net: bool = True,
) -> float:
    """
    Paper Section 3.4 / Fig. 7 Delayed Gratification metric from
    `analysis/delay_gratification_analysis_for_not_move.py` (`avg_wandering_range`).

    Parameters
    ----------
    mono_trace : Sequence[int]
        Monotonicity error recorded after each swap step.
    net : bool, default True
        If True (matches `delay_gratification_analysis_for_not_move.py` used for
        Paper Fig. 7 and Section 4.3 numbers 1.10 -> 1.13 -> 1.15 -> 1.19),
        each setback `y = d_arr[i] > 0` followed by recovery `x = -d_arr[i+1] > 0`
        contributes `(x - y) / y`.
        If False (matches `delay_gratification_analysis.py` / Fig. 6D schematic),
        each pair contributes `x / y = 1 + (x - y) / y`.
    """
    d_arr = _discrepancy_runs(mono_trace)
    first_pos = next((i for i, v in enumerate(d_arr) if v > 0), len(d_arr))
    if first_pos >= len(d_arr) - 1:
        return 0.0
    total = 0.0
    count = 0
    i = first_pos
    while i < len(d_arr):
        j = i + 1
        if j >= len(d_arr) or d_arr[j] > 0:
            break
        if net:
            change = -(d_arr[j] + d_arr[i]) / d_arr[i]
        else:
            change = -d_arr[j] / d_arr[i]
        total += change
        count += 1
        i += 2
    return total / count if count > 0 else 0.0


def aggregation_value(algotypes: Sequence[Algotype]) -> float:
    """
    Paper Section 3.4 & `build_neighbor_checking_arr` in
    `analysis/cell_type_aggregation_analysis.py` (L518-538):
    Fraction of cells whose adjacent right neighbor has the same algotype.
    (For a random 50/50 two-algotype array of length n=100, expectation is ~0.495;
    for a random 1/3:1/3:1/3 three-algotype array, expectation is ~0.33.)
    """
    n = len(algotypes)
    if n <= 1:
        return 1.0
    same = sum(1 for i in range(n - 1) if algotypes[i] == algotypes[i + 1])
    return same / n


# ---------------------------------------------------------------------------
# Local cell-view rules (`modules/multithread/*SortCell.py`)
# ---------------------------------------------------------------------------

def _left_sorted(cells: Sequence[Cell], p: int, direction: SortDirection = 1) -> bool:
    """
    Check whether `cells[0 .. p-1]` is sorted according to `direction`,
    matching `InsertionSortCell.is_enable_to_move()` in
    `Zhangtaining/cell_research/modules/multithread/InsertionSortCell.py` (L68-82):
    frozen cells reset `prev` and are skipped so an out-of-order frozen cell
    does not permanently block all Insertion cells to its right.
    """
    if direction == 1:
        prev = -10**18
        for k in range(p):
            if cells[k].frozen is not None:
                prev = -10**18
                continue
            if cells[k].value < prev:
                return False
            prev = cells[k].value
        return True
    else:
        prev = 10**18
        for k in range(p):
            if cells[k].frozen is not None:
                prev = 10**18
                continue
            if cells[k].value > prev:
                return False
            prev = cells[k].value
        return True


def _cell_should_move(cells: List[Cell], p: int) -> bool:
    """
    Match `should_move()` in `BubbleSortCell.py`, `InsertionSortCell.py`,
    and `SelectionSortCell.py` (used by `StatusProbe.record_compare_and_swap`
    and termination detection `no_cells_should_move`).
    """
    c = cells[p]
    if c.frozen is not None:
        return False
    n = len(cells)
    d = c.direction
    if c.algotype == "B":
        left_ok = (
            p > 0
            and cells[p - 1].frozen != "immovable"
            and ((c.value < cells[p - 1].value) if d == 1 else (c.value > cells[p - 1].value))
        )
        right_ok = (
            p < n - 1
            and cells[p + 1].frozen != "immovable"
            and ((c.value > cells[p + 1].value) if d == 1 else (c.value < cells[p + 1].value))
        )
        return left_ok or right_ok
    elif c.algotype == "I":
        if p <= 0 or cells[p - 1].frozen == "immovable":
            return False
        if not _left_sorted(cells, p, direction=d):
            return False
        return (c.value < cells[p - 1].value) if d == 1 else (c.value > cells[p - 1].value)
    elif c.algotype == "S":
        return p != c.target and 0 <= c.target < n
    return False


def _reset_selection_target(cell: Cell, n: int) -> None:
    """Match `SelectionSortCell.update()` called inside `MultiThreadCell.swap()`."""
    if cell.algotype == "S":
        cell.target = 0 if cell.direction == 1 else (n - 1)


def _propose(
    cells: List[Cell],
    p: int,
    rng: random.Random,
    selection_reset_on_swap: bool = True,
    selection_movable_mode: Literal["standard", "pyc310"] = "standard",
) -> Tuple[Optional[Tuple[int, int]], bool]:
    """
    Let the cell at index `p` execute one local perception-action cycle.

    Returns `(swap_pair, did_compare)` where `swap_pair` is `(p, q)` if a swap
    should be performed (or `None` if no swap), and `did_compare` is True if the
    cell had an active move/comparison opportunity (`should_move()` was True,
    matching `StatusProbe.record_compare_and_swap()`).
    """
    c = cells[p]
    if c.frozen is not None:
        return None, False
    n = len(cells)
    d = c.direction

    if c.algotype == "B":
        can_move = _cell_should_move(cells, p)
        side = -1 if rng.random() < 0.5 else 1
        q = p + side
        if q < 0 or q >= n:
            return None, can_move
        nb = cells[q]
        if nb.frozen == "immovable":
            return None, can_move
        if d == 1:
            if (side == -1 and c.value < nb.value) or (side == 1 and c.value > nb.value):
                return (p, q), True
        else:
            if (side == -1 and c.value > nb.value) or (side == 1 and c.value < nb.value):
                return (p, q), True
        return None, can_move

    if c.algotype == "I":
        if p == 0:
            return None, False
        if not _left_sorted(cells, p, direction=d):
            return None, False
        nb = cells[p - 1]
        if nb.frozen == "immovable":
            return None, False
        if (d == 1 and c.value < nb.value) or (d == -1 and c.value > nb.value):
            return (p, p - 1), True
        return None, False

    if c.algotype == "S":
        t = c.target
        if t < 0 or t >= n or t == p:
            return None, False
        occ = cells[t]
        if occ.frozen == "immovable":
            c.target = (t + 1) if d == 1 else (t - 1)
            return None, True
        if occ.frozen == "movable" and selection_movable_mode == "pyc310":
            # In SelectionSortCell.cpython-310.pyc (used for Paper Fig. 5B),
            # encountering a FREEZE cell at target increments ideal_position,
            # and swaps only if value < occ.value while occ is passive.
            want_swap = c.value < occ.value
            if want_swap and t == occ.target:
                return (p, t), True
            c.target = (t + 1) if d == 1 else (t - 1)
            return None, True

        # Note: SelectionSortCell.py always seeks the smallest remaining value
        # (`c.value < occ.value`); for increasing order (d == 1) `target` scans
        # 0 -> n-1, whereas for decreasing order (d == -1) `target` scans
        # n-1 -> 0 (placing smallest values at the right end).
        want_swap = c.value < occ.value
        if want_swap:
            return (p, t), True
        c.target = (t + 1) if d == 1 else (t - 1)
        return None, True

    raise ValueError(f"Unknown algotype: {c.algotype}")


def _compute_active_indices(cells: List[Cell]) -> List[int]:
    """
    Compute all cell indices `p` for which `_cell_should_move(cells, p)` is True
    in a single O(n) pass (tracking running prefix sortedness for both
    increasing and decreasing Insertion cells).
    """
    n = len(cells)
    act: List[int] = []
    inc_ok = True
    dec_ok = True
    prev_inc = -10**18
    prev_dec = 10**18

    for p in range(n):
        c = cells[p]
        if c.frozen is None:
            d = c.direction
            if c.algotype == "B":
                left_ok = (
                    p > 0
                    and cells[p - 1].frozen != "immovable"
                    and ((c.value < cells[p - 1].value) if d == 1 else (c.value > cells[p - 1].value))
                )
                right_ok = (
                    p < n - 1
                    and cells[p + 1].frozen != "immovable"
                    and ((c.value > cells[p + 1].value) if d == 1 else (c.value < cells[p + 1].value))
                )
                if left_ok or right_ok:
                    act.append(p)
            elif c.algotype == "I":
                pref_ok = inc_ok if d == 1 else dec_ok
                if pref_ok and p > 0 and cells[p - 1].frozen != "immovable":
                    if (d == 1 and c.value < cells[p - 1].value) or (
                        d == -1 and c.value > cells[p - 1].value
                    ):
                        act.append(p)
            elif c.algotype == "S":
                if p != c.target and 0 <= c.target < n:
                    act.append(p)

        # Advance prefix-sorted state past index p (frozen cells reset prev)
        if c.frozen is not None:
            prev_inc = -10**18
            prev_dec = 10**18
        else:
            if c.value < prev_inc:
                inc_ok = False
            prev_inc = c.value
            if c.value > prev_dec:
                dec_ok = False
            prev_dec = c.value

    return act


def _any_cell_can_act(cells: List[Cell]) -> bool:
    """Exact termination check `not no_cells_should_move(cells)`."""
    return bool(_compute_active_indices(cells))


# ---------------------------------------------------------------------------
# Simulation engines (asynchronous random, parallel synchronous, OS-threaded)
# ---------------------------------------------------------------------------

@dataclass
class SimResult:
    swaps: int
    steps: int
    comparisons: int  # comparisons + swaps (Paper Fig. 4B metric)
    final_values: List[int]
    final_algotypes: List[Algotype]
    final_error: int
    final_sortedness: float
    dg: float
    dg_gross: float
    error_trace: List[int]
    sortedness_trace: List[float]
    aggregation_trace: List[float]
    value_snapshots: Optional[np.ndarray] = None
    algotype_snapshots: Optional[np.ndarray] = None


def make_cells(
    values: Sequence[int],
    algotypes: Sequence[Algotype],
    frozen_indices: Sequence[int] = (),
    frozen_kind: FrozenKind = "immovable",
    directions: Optional[Sequence[SortDirection]] = None,
) -> List[Cell]:
    n = len(values)
    fset = set(frozen_indices)
    if directions is None:
        directions = [1] * n
    return [
        Cell(
            value=int(v),
            algotype=a,
            frozen=(frozen_kind if i in fset else None),
            target=(0 if d == 1 else n - 1),
            direction=d,
            uid=i,
        )
        for i, (v, a, d) in enumerate(zip(values, algotypes, directions))
    ]


def simulate(
    cells: List[Cell],
    max_steps: Optional[int] = None,
    max_swaps: Optional[int] = None,
    stop_when_sorted: bool = True,
    seed: Optional[int] = None,
    record_snapshots: bool = False,
    snapshot_every: int = 1,
    selection_reset_on_swap: bool = True,
    selection_movable_mode: Literal["standard", "pyc310"] = "standard",
) -> SimResult:
    """
    Asynchronous cell-view simulation.

    At each step an active cell that `should_move()` (or a random cell from
    the array) is selected and executes one perception-action step under a
    shared array lock, matching `MultiThreadCell.run()` in
    `Zhangtaining/cell_research`.
    """
    rng = random.Random(seed)
    n = len(cells)
    if max_steps is None:
        max_steps = n * n * 40

    vals = [c.value for c in cells]
    err0 = monotonicity_error(vals)
    error_trace: List[int] = []
    sortedness_trace: List[float] = [(n - 1 - err0) / (n - 1) if n > 1 else 1.0]
    aggregation_trace: List[float] = [aggregation_value([c.algotype for c in cells])]

    val_snaps: List[List[int]] = [[c.value for c in cells]] if record_snapshots else []
    algo_snaps: List[List[int]] = (
        [[0 if c.algotype == "B" else 1 if c.algotype == "I" else 2 for c in cells]]
        if record_snapshots
        else []
    )

    swaps = 0
    steps = 0
    comparisons = 0
    act_indices = _compute_active_indices(cells)

    while steps < max_steps:
        if stop_when_sorted and err0 == 0:
            break
        if max_swaps is not None and swaps >= max_swaps:
            break
        if not act_indices:
            break

        pick = rng.randrange(len(act_indices))
        p = act_indices[pick]
        steps += 1
        proposal, did_cmp = _propose(
            cells,
            p,
            rng,
            selection_reset_on_swap=selection_reset_on_swap,
            selection_movable_mode=selection_movable_mode,
        )
        if did_cmp:
            comparisons += 1

        if proposal is not None:
            i, j = proposal
            cells[i], cells[j] = cells[j], cells[i]
            if selection_reset_on_swap:
                _reset_selection_target(cells[i], n)
                _reset_selection_target(cells[j], n)
            else:
                # If a displaced Selection cell was moved to the left of its target,
                # clamp/reset its target so target <= current_position holds.
                for idx in (i, j):
                    if cells[idx].algotype == "S":
                        if cells[idx].direction == 1 and cells[idx].target > idx:
                            cells[idx].target = 0
                        elif cells[idx].direction == -1 and cells[idx].target < idx:
                            cells[idx].target = n - 1
            swaps += 1
            vals = [c.value for c in cells]
            err0 = monotonicity_error(vals)
            error_trace.append(err0)
            sortedness_trace.append((n - 1 - err0) / (n - 1) if n > 1 else 1.0)
            aggregation_trace.append(aggregation_value([c.algotype for c in cells]))
            if record_snapshots and (swaps % snapshot_every == 0):
                val_snaps.append(vals)
                algo_snaps.append(
                    [0 if c.algotype == "B" else 1 if c.algotype == "I" else 2 for c in cells]
                )
            act_indices = _compute_active_indices(cells)
        elif cells[p].algotype == "S":
            t = cells[p].target
            if t == p or t < 0 or t >= n:
                act_indices[pick] = act_indices[-1]
                act_indices.pop()

    final_vals = [c.value for c in cells]
    final_algos = [c.algotype for c in cells]
    final_err = monotonicity_error(final_vals)
    return SimResult(
        swaps=swaps,
        steps=steps,
        comparisons=comparisons,
        final_values=final_vals,
        final_algotypes=final_algos,
        final_error=final_err,
        final_sortedness=sortedness(final_vals),
        dg=delayed_gratification(error_trace, net=True),
        dg_gross=delayed_gratification(error_trace, net=False),
        error_trace=error_trace,
        sortedness_trace=sortedness_trace,
        aggregation_trace=aggregation_trace,
        value_snapshots=np.asarray(val_snaps) if record_snapshots else None,
        algotype_snapshots=np.asarray(algo_snaps) if record_snapshots else None,
    )


def simulate_parallel(
    cells: List[Cell],
    max_rounds: Optional[int] = None,
    seed: Optional[int] = None,
) -> SimResult:
    """
    Synchronous parallel update round: every active cell proposes a move
    simultaneously; a random maximal matching of disjoint swaps is committed.
    """
    rng = random.Random(seed)
    n = len(cells)
    if max_rounds is None:
        max_rounds = n * n * 4

    vals = [c.value for c in cells]
    err0 = monotonicity_error(vals)
    error_trace: List[int] = []
    sortedness_trace: List[float] = [(n - 1 - err0) / (n - 1) if n > 1 else 1.0]
    aggregation_trace: List[float] = [aggregation_value([c.algotype for c in cells])]

    swaps = 0
    rounds = 0
    comparisons = 0

    while rounds < max_rounds and err0 > 0:
        if not _any_cell_can_act(cells):
            break
        rounds += 1
        order = list(range(n))
        rng.shuffle(order)
        proposals: List[Tuple[int, int]] = []
        for p in order:
            prop, did_cmp = _propose(cells, p, rng)
            if did_cmp:
                comparisons += 1
            if prop is not None:
                proposals.append(prop)
        used: set[int] = set()
        for i, j in proposals:
            if i not in used and j not in used:
                cells[i], cells[j] = cells[j], cells[i]
                _reset_selection_target(cells[i], n)
                _reset_selection_target(cells[j], n)
                used.add(i)
                used.add(j)
                swaps += 1
                vals = [c.value for c in cells]
                err0 = monotonicity_error(vals)
                error_trace.append(err0)
                sortedness_trace.append((n - 1 - err0) / (n - 1) if n > 1 else 1.0)
                aggregation_trace.append(aggregation_value([c.algotype for c in cells]))

    final_vals = [c.value for c in cells]
    final_algos = [c.algotype for c in cells]
    return SimResult(
        swaps=swaps,
        steps=rounds,
        comparisons=comparisons,
        final_values=final_vals,
        final_algotypes=final_algos,
        final_error=monotonicity_error(final_vals),
        final_sortedness=sortedness(final_vals),
        dg=delayed_gratification(error_trace, net=True),
        dg_gross=delayed_gratification(error_trace, net=False),
        error_trace=error_trace,
        sortedness_trace=sortedness_trace,
        aggregation_trace=aggregation_trace,
    )


def simulate_threaded(
    cells: List[Cell],
    max_steps: Optional[int] = None,
    seed: Optional[int] = None,
) -> SimResult:
    """
    Literal OS-threaded execution (`threading.Thread` per cell + shared `Lock`),
    matching `MultiThreadCell` in `Zhangtaining/cell_research`.
    """
    n = len(cells)
    if max_steps is None:
        max_steps = n * n * 40
    lock = threading.Lock()
    stop_evt = threading.Event()

    vals0 = [c.value for c in cells]
    error_trace: List[int] = []
    sortedness_trace: List[float] = [sortedness(vals0)]
    aggregation_trace: List[float] = [aggregation_value([c.algotype for c in cells])]
    counters = {"swaps": 0, "steps": 0, "comparisons": 0}

    uid_to_idx = {c.uid: i for i, c in enumerate(cells)}
    active_set = set(_compute_active_indices(cells))

    def worker(cell_uid: int, worker_seed: int) -> None:
        nonlocal active_set
        local_rng = random.Random(worker_seed)
        while not stop_evt.is_set():
            with lock:
                if stop_evt.is_set():
                    break
                if not active_set:
                    stop_evt.set()
                    break
                p = uid_to_idx[cell_uid]
                if p in active_set:
                    counters["steps"] += 1
                    prop, did_cmp = _propose(cells, p, local_rng)
                    if did_cmp:
                        counters["comparisons"] += 1
                    if prop is not None:
                        i, j = prop
                        uid_i, uid_j = cells[i].uid, cells[j].uid
                        cells[i], cells[j] = cells[j], cells[i]
                        _reset_selection_target(cells[i], n)
                        _reset_selection_target(cells[j], n)
                        uid_to_idx[uid_i] = j
                        uid_to_idx[uid_j] = i
                        counters["swaps"] += 1
                        vals = [c.value for c in cells]
                        err = monotonicity_error(vals)
                        error_trace.append(err)
                        sortedness_trace.append((n - 1 - err) / (n - 1) if n > 1 else 1.0)
                        aggregation_trace.append(aggregation_value([c.algotype for c in cells]))
                        if err == 0:
                            stop_evt.set()
                            break
                        active_set = set(_compute_active_indices(cells))
                    elif cells[p].algotype == "S":
                        t = cells[p].target
                        if t == p or t < 0 or t >= n:
                            active_set.discard(p)
                    if counters["steps"] >= max_steps:
                        stop_evt.set()
                        break
            time.sleep(0)

    base_rng = random.Random(seed)
    threads = [
        threading.Thread(target=worker, args=(c.uid, base_rng.randint(0, 10**9)), daemon=True)
        for c in cells
        if c.frozen is None
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30.0)
    stop_evt.set()

    final_vals = [c.value for c in cells]
    final_algos = [c.algotype for c in cells]
    return SimResult(
        swaps=counters["swaps"],
        steps=counters["steps"],
        comparisons=counters["comparisons"],
        final_values=final_vals,
        final_algotypes=final_algos,
        final_error=monotonicity_error(final_vals),
        final_sortedness=sortedness(final_vals),
        dg=delayed_gratification(error_trace, net=True),
        dg_gross=delayed_gratification(error_trace, net=False),
        error_trace=error_trace,
        sortedness_trace=sortedness_trace,
        aggregation_trace=aggregation_trace,
    )


# ---------------------------------------------------------------------------
# Traditional top-down sorting baselines (with optional frozen indices)
# ---------------------------------------------------------------------------

def traditional_bubble(
    values: Sequence[int],
    frozen_indices: Sequence[int] = (),
    frozen_kind: FrozenKind = "immovable",
) -> Tuple[List[int], int, int, List[int]]:
    """
    Top-down Bubble Sort with frozen defects.
    Returns `(sorted_values, swaps, comparisons_plus_swaps, error_trace)`.
    """
    a = list(values)
    fset = set(frozen_indices)
    n = len(a)
    swaps = 0
    comparisons = 0
    err_trace: List[int] = []
    for i in range(n):
        swapped = False
        for j in range(0, n - i - 1):
            comparisons += 1
            if j in fset or (j + 1) in fset:
                if frozen_kind == "immovable":
                    continue
                if j in fset and (j + 1) in fset:
                    continue
            if a[j] > a[j + 1]:
                a[j], a[j + 1] = a[j + 1], a[j]
                if frozen_kind == "movable":
                    j_in, jp_in = (j in fset), ((j + 1) in fset)
                    fset.discard(j)
                    fset.discard(j + 1)
                    if j_in:
                        fset.add(j + 1)
                    if jp_in:
                        fset.add(j)
                swaps += 1
                comparisons += 1
                swapped = True
                err_trace.append(monotonicity_error(a))
        if not swapped and not fset:
            break
    if not err_trace:
        err_trace.append(monotonicity_error(a))
    return a, swaps, comparisons, err_trace


def traditional_insertion(
    values: Sequence[int],
    frozen_indices: Sequence[int] = (),
    frozen_kind: FrozenKind = "immovable",
) -> Tuple[List[int], int, int, List[int]]:
    """
    Top-down Insertion Sort (adjacent-swap formulation) with frozen defects.
    Returns `(sorted_values, swaps, comparisons_plus_swaps, error_trace)`.
    """
    a = list(values)
    fset = set(frozen_indices)
    n = len(a)
    swaps = 0
    comparisons = 0
    err_trace: List[int] = []
    for i in range(1, n):
        j = i
        while j > 0:
            comparisons += 1
            if a[j - 1] <= a[j]:
                break
            if j in fset or (j - 1) in fset:
                if frozen_kind == "immovable":
                    break
                if j in fset:
                    break  # passive frozen key does not initiate leftward insertion
            a[j], a[j - 1] = a[j - 1], a[j]
            if frozen_kind == "movable":
                jm_in = (j - 1) in fset
                fset.discard(j - 1)
                if jm_in:
                    fset.add(j)
            swaps += 1
            comparisons += 1
            err_trace.append(monotonicity_error(a))
            j -= 1
    if not err_trace:
        err_trace.append(monotonicity_error(a))
    return a, swaps, comparisons, err_trace


def traditional_selection(
    values: Sequence[int],
    frozen_indices: Sequence[int] = (),
    frozen_kind: FrozenKind = "immovable",
) -> Tuple[List[int], int, int, List[int]]:
    """
    Top-down Selection Sort with frozen defects.
    Returns `(sorted_values, swaps, comparisons_plus_swaps, error_trace)`.
    """
    a = list(values)
    fset = set(frozen_indices)
    n = len(a)
    swaps = 0
    comparisons = 0
    err_trace: List[int] = []
    for i in range(n):
        if i in fset and frozen_kind == "immovable":
            continue
        min_idx = i
        for j in range(i + 1, n):
            comparisons += 1
            if j in fset and frozen_kind == "immovable":
                continue
            if a[j] < a[min_idx]:
                min_idx = j
        if min_idx != i:
            if (i in fset or min_idx in fset) and frozen_kind == "immovable":
                continue
            if i in fset and frozen_kind == "movable":
                # Passive frozen cell at slot i does not initiate a selection swap
                continue
            a[i], a[min_idx] = a[min_idx], a[i]
            if frozen_kind == "movable":
                m_in = min_idx in fset
                fset.discard(min_idx)
                if m_in:
                    fset.add(i)
            swaps += 1
            comparisons += 1
            err_trace.append(monotonicity_error(a))
    if not err_trace:
        err_trace.append(monotonicity_error(a))
    return a, swaps, comparisons, err_trace


TRADITIONAL_ALGOS: dict[Algotype, Callable[..., Tuple[List[int], int, int, List[int]]]] = {
    "B": traditional_bubble,
    "I": traditional_insertion,
    "S": traditional_selection,
}


# ---------------------------------------------------------------------------
# Deterministic Paired A/B Lane Comparison & Claim-Boundary Catalog (Track 3)
# Ported from MesmerPrism/combinatorial-swarmability (`comparison.rs` & `catalog.v1.json`)
# ---------------------------------------------------------------------------

@dataclass
class LaneSpec:
    """Configuration for one lane (Lane A or Lane B) in a paired comparison."""
    name: str
    algotypes: Tuple[Algotype, ...] = ("B",)
    frozen_count: int = 0
    frozen_kind: FrozenKind = "immovable"
    duplicate_values: bool = False
    directions: Optional[Tuple[SortDirection, ...]] = None
    selection_reset_on_swap: bool = True
    selection_movable_mode: Literal["standard", "pyc310"] = "standard"


@dataclass
class PairedABResult:
    """Summary and step-normalized delta curves for a seed-locked Lane A vs. Lane B run."""
    lane_a_name: str
    lane_b_name: str
    n: int
    reps: int
    seed: int
    progress_grid: np.ndarray  # 0.0 .. 1.0 (101 points)
    # Lane A means
    mean_swaps_a: float
    mean_comparisons_a: float
    mean_final_error_a: float
    mean_final_sortedness_a: float
    mean_dg_a: float
    mean_peak_agg_a: float
    mean_final_agg_a: float
    # Lane B means
    mean_swaps_b: float
    mean_comparisons_b: float
    mean_final_error_b: float
    mean_final_sortedness_b: float
    mean_dg_b: float
    mean_peak_agg_b: float
    mean_final_agg_b: float
    # Paired differences (B - A) across identical initial permutations & defect positions
    delta_swaps_mean: float
    delta_swaps_sem: float
    delta_final_error_mean: float
    delta_final_error_sem: float
    delta_dg_mean: float
    delta_dg_sem: float
    delta_peak_agg_mean: float
    delta_peak_agg_sem: float
    delta_final_agg_mean: float
    delta_final_agg_sem: float
    # Trajectories over normalized progress (mean across paired seeds)
    error_curve_a: np.ndarray
    error_curve_b: np.ndarray
    delta_error_curve: np.ndarray
    delta_error_sem_curve: np.ndarray
    sortedness_curve_a: np.ndarray
    sortedness_curve_b: np.ndarray
    delta_sortedness_curve: np.ndarray
    agg_curve_a: np.ndarray
    agg_curve_b: np.ndarray
    delta_agg_curve: np.ndarray
    delta_agg_sem_curve: np.ndarray


def validate_claim_boundary_catalog(catalog_path: Optional[ str ] = None) -> Tuple[bool, List[str]]:
    """
    Validate `sorting-morphogenesis/catalog.json` against the 4-tier claim-boundary
    schema from `MesmerPrism/combinatorial-swarmability` (`reported`, `reconstruction`,
    `transfer_boundary`, `does_not_claim`).
    """
    import json
    from pathlib import Path

    path = Path(catalog_path) if catalog_path else (Path(__file__).resolve().parent / "catalog.json")
    errors: List[str] = []
    if not path.exists():
        return False, [f"Catalog file not found: {path}"]

    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("schema_version") != "combinatorial-swarmability-catalog.v1":
        errors.append(f"Unexpected schema_version: {data.get('schema_version')}")

    entries = data.get("entries", [])
    if not entries:
        errors.append("Catalog contains no entries.")

    required_tiers = ("reported", "reconstruction", "transfer_boundary", "does_not_claim")
    for idx, entry in enumerate(entries):
        eid = entry.get("id", f"entry_{idx}")
        cb = entry.get("claim_boundary", {})
        for tier in required_tiers:
            items = cb.get(tier)
            if not isinstance(items, list) or len(items) == 0 or not all(isinstance(s, str) and s.strip() for s in items):
                errors.append(f"Entry '{eid}' missing non-empty claim_boundary.{tier} list")

    return len(errors) == 0, errors


def compare_lanes_ab(
    lane_a: LaneSpec,
    lane_b: LaneSpec,
    n: int = 100,
    reps: int = 20,
    seed: int = 42,
) -> PairedABResult:
    """
    Execute Lane A and Lane B side-by-side from locked initial permutations,
    locked frozen-cell defect positions, and locked per-replicate RNG seeds.
    """
    grid = np.linspace(0.0, 1.0, 101)
    max_frozen = max(lane_a.frozen_count, lane_b.frozen_count)

    swaps_a, swaps_b = [], []
    comps_a, comps_b = [], []
    ferr_a, ferr_b = [], []
    fsrt_a, fsrt_b = [], []
    dg_a, dg_b = [], []
    pagg_a, pagg_b = [], []
    fagg_a, fagg_b = [], []

    err_curves_a, err_curves_b, delta_err_curves = [], [], []
    srt_curves_a, srt_curves_b, delta_srt_curves = [], [], []
    agg_curves_a, agg_curves_b, delta_agg_curves = [], [], []

    for r in range(reps):
        rep_rng = np.random.default_rng(seed * 10000 + r * 97)
        # Shared base permutation order
        perm = rep_rng.permutation(n).tolist()
        vals_unique = [p + 1 for p in perm]
        vals_dup = [((p % 10) + 1) for p in perm]

        # Shared pool of defect indices so Lane A and Lane B share exact defect locations
        shared_fidx = rep_rng.choice(n, size=max_frozen, replace=False).tolist() if max_frozen > 0 else []

        # Shared algotype shuffle order
        algo_perm = rep_rng.permutation(n).tolist()

        def _build_lane_cells(spec: LaneSpec) -> List[Cell]:
            vals = vals_dup if spec.duplicate_values else vals_unique
            base_algos = [spec.algotypes[i % len(spec.algotypes)] for i in range(n)]
            atypes: List[Algotype] = [base_algos[idx] for idx in algo_perm]
            if spec.directions is not None:
                base_dirs = [spec.directions[i % len(spec.directions)] for i in range(n)]
                dirs: Optional[List[SortDirection]] = [base_dirs[idx] for idx in algo_perm]
            else:
                dirs = None
            fidx = shared_fidx[: spec.frozen_count]
            return make_cells(vals, atypes, frozen_indices=fidx, frozen_kind=spec.frozen_kind, directions=dirs)

        sim_seed = seed * 50000 + r * 31
        res_a = simulate(
            _build_lane_cells(lane_a),
            stop_when_sorted=(lane_a.frozen_count == 0 and lane_a.directions is None),
            seed=sim_seed,
            selection_reset_on_swap=lane_a.selection_reset_on_swap,
            selection_movable_mode=lane_a.selection_movable_mode,
        )
        res_b = simulate(
            _build_lane_cells(lane_b),
            stop_when_sorted=(lane_b.frozen_count == 0 and lane_b.directions is None),
            seed=sim_seed,
            selection_reset_on_swap=lane_b.selection_reset_on_swap,
            selection_movable_mode=lane_b.selection_movable_mode,
        )

        swaps_a.append(res_a.swaps)
        swaps_b.append(res_b.swaps)
        comps_a.append(res_a.comparisons)
        comps_b.append(res_b.comparisons)
        ferr_a.append(res_a.final_error)
        ferr_b.append(res_b.final_error)
        fsrt_a.append(res_a.final_sortedness)
        fsrt_b.append(res_b.final_sortedness)
        dg_a.append(res_a.dg)
        dg_b.append(res_b.dg)

        ag_tr_a = np.asarray(res_a.aggregation_trace, dtype=float)
        ag_tr_b = np.asarray(res_b.aggregation_trace, dtype=float)
        pagg_a.append(float(np.max(ag_tr_a)))
        pagg_b.append(float(np.max(ag_tr_b)))
        fagg_a.append(float(ag_tr_a[-1]))
        fagg_b.append(float(ag_tr_b[-1]))

        # Interpolate error, sortedness, and aggregation traces onto common progress grid
        err_tr_a = np.asarray([monotonicity_error([c for c in (vals_dup if lane_a.duplicate_values else vals_unique)])] + res_a.error_trace, dtype=float)
        err_tr_b = np.asarray([monotonicity_error([c for c in (vals_dup if lane_b.duplicate_values else vals_unique)])] + res_b.error_trace, dtype=float)
        srt_tr_a = np.asarray(res_a.sortedness_trace, dtype=float)
        srt_tr_b = np.asarray(res_b.sortedness_trace, dtype=float)

        ec_a = np.interp(grid, np.linspace(0.0, 1.0, len(err_tr_a)), err_tr_a)
        ec_b = np.interp(grid, np.linspace(0.0, 1.0, len(err_tr_b)), err_tr_b)
        sc_a = np.interp(grid, np.linspace(0.0, 1.0, len(srt_tr_a)), srt_tr_a)
        sc_b = np.interp(grid, np.linspace(0.0, 1.0, len(srt_tr_b)), srt_tr_b)
        ac_a = np.interp(grid, np.linspace(0.0, 1.0, len(ag_tr_a)), ag_tr_a)
        ac_b = np.interp(grid, np.linspace(0.0, 1.0, len(ag_tr_b)), ag_tr_b)

        err_curves_a.append(ec_a)
        err_curves_b.append(ec_b)
        delta_err_curves.append(ec_b - ec_a)

        srt_curves_a.append(sc_a)
        srt_curves_b.append(sc_b)
        delta_srt_curves.append(sc_b - sc_a)

        agg_curves_a.append(ac_a)
        agg_curves_b.append(ac_b)
        delta_agg_curves.append(ac_b - ac_a)

    sw_a_arr, sw_b_arr = np.asarray(swaps_a, dtype=float), np.asarray(swaps_b, dtype=float)
    fe_a_arr, fe_b_arr = np.asarray(ferr_a, dtype=float), np.asarray(ferr_b, dtype=float)
    dg_a_arr, dg_b_arr = np.asarray(dg_a, dtype=float), np.asarray(dg_b, dtype=float)
    pa_a_arr, pa_b_arr = np.asarray(pagg_a, dtype=float), np.asarray(pagg_b, dtype=float)
    fa_a_arr, fa_b_arr = np.asarray(fagg_a, dtype=float), np.asarray(fagg_b, dtype=float)

    def _mean_sem(diff: np.ndarray) -> Tuple[float, float]:
        return float(np.mean(diff)), float(np.std(diff) / max(1.0, np.sqrt(len(diff))))

    d_sw_m, d_sw_s = _mean_sem(sw_b_arr - sw_a_arr)
    d_fe_m, d_fe_s = _mean_sem(fe_b_arr - fe_a_arr)
    d_dg_m, d_dg_s = _mean_sem(dg_b_arr - dg_a_arr)
    d_pa_m, d_pa_s = _mean_sem(pa_b_arr - pa_a_arr)
    d_fa_m, d_fa_s = _mean_sem(fa_b_arr - fa_a_arr)

    dec_mat = np.vstack(delta_err_curves)
    dac_mat = np.vstack(delta_agg_curves)

    return PairedABResult(
        lane_a_name=lane_a.name,
        lane_b_name=lane_b.name,
        n=n,
        reps=reps,
        seed=seed,
        progress_grid=grid,
        mean_swaps_a=float(np.mean(sw_a_arr)),
        mean_comparisons_a=float(np.mean(comps_a)),
        mean_final_error_a=float(np.mean(fe_a_arr)),
        mean_final_sortedness_a=float(np.mean(fsrt_a)),
        mean_dg_a=float(np.mean(dg_a_arr)),
        mean_peak_agg_a=float(np.mean(pa_a_arr)),
        mean_final_agg_a=float(np.mean(fa_a_arr)),
        mean_swaps_b=float(np.mean(sw_b_arr)),
        mean_comparisons_b=float(np.mean(comps_b)),
        mean_final_error_b=float(np.mean(fe_b_arr)),
        mean_final_sortedness_b=float(np.mean(fsrt_b)),
        mean_dg_b=float(np.mean(dg_b_arr)),
        mean_peak_agg_b=float(np.mean(pa_b_arr)),
        mean_final_agg_b=float(np.mean(fa_b_arr)),
        delta_swaps_mean=d_sw_m,
        delta_swaps_sem=d_sw_s,
        delta_final_error_mean=d_fe_m,
        delta_final_error_sem=d_fe_s,
        delta_dg_mean=d_dg_m,
        delta_dg_sem=d_dg_s,
        delta_peak_agg_mean=d_pa_m,
        delta_peak_agg_sem=d_pa_s,
        delta_final_agg_mean=d_fa_m,
        delta_final_agg_sem=d_fa_s,
        error_curve_a=np.mean(err_curves_a, axis=0),
        error_curve_b=np.mean(err_curves_b, axis=0),
        delta_error_curve=np.mean(dec_mat, axis=0),
        delta_error_sem_curve=np.std(dec_mat, axis=0) / max(1.0, np.sqrt(reps)),
        sortedness_curve_a=np.mean(srt_curves_a, axis=0),
        sortedness_curve_b=np.mean(srt_curves_b, axis=0),
        delta_sortedness_curve=np.mean(delta_srt_curves, axis=0),
        agg_curve_a=np.mean(agg_curves_a, axis=0),
        agg_curve_b=np.mean(agg_curves_b, axis=0),
        delta_agg_curve=np.mean(dac_mat, axis=0),
        delta_agg_sem_curve=np.std(dac_mat, axis=0) / max(1.0, np.sqrt(reps)),
    )


AB_PRESETS: dict[str, Tuple[LaneSpec, LaneSpec]] = {
    "bubble_intact_vs_frozen3": (
        LaneSpec(name="Lane A: Bubble (Intact f=0)", algotypes=("B",), frozen_count=0),
        LaneSpec(name="Lane B: Bubble (3 Immovable Frozen f=3)", algotypes=("B",), frozen_count=3, frozen_kind="immovable"),
    ),
    "bubble_vs_selection_frozen3": (
        LaneSpec(name="Lane A: Bubble (f=3 Immovable)", algotypes=("B",), frozen_count=3, frozen_kind="immovable"),
        LaneSpec(name="Lane B: Selection (f=3 Immovable)", algotypes=("S",), frozen_count=3, frozen_kind="immovable"),
    ),
    "chimera_bs_unique_vs_duplicate": (
        LaneSpec(name="Lane A: Chimera B+S (Unique Values)", algotypes=("B", "S"), duplicate_values=False),
        LaneSpec(name="Lane B: Chimera B+S (Duplicate Values 1..10)", algotypes=("B", "S"), duplicate_values=True),
    ),
}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Cell-view sorting morphogenesis with deterministic paired A/B lane comparison."
    )
    parser.add_argument(
        "--compare-ab",
        choices=list(AB_PRESETS.keys()) + ["all"],
        default=None,
        help="Run deterministic seed-locked paired A/B lane comparison preset",
    )
    parser.add_argument("--n", type=int, default=60, help="Array length (default: 60)")
    parser.add_argument("--reps", type=int, default=15, help="Paired seed replicates (default: 15)")
    parser.add_argument("--seed", type=int, default=42, help="Base RNG seed")
    parser.add_argument("--validate-catalog", action="store_true", help="Validate catalog.json claim-boundary schema")
    parser.add_argument("--self-test", action="store_true",
                        help="Run deterministic checks (catalog schema + a small seed-locked sorting run) and exit")
    args = parser.parse_args()

    if args.self_test:
        ok, errs = validate_claim_boundary_catalog()
        print(f"[{'PASS' if ok else 'FAIL'}] Claim-boundary catalog schema validation ({len(errs)} errors)")
        for e in errs:
            print(f"  - {e}")
        # Small deterministic cell-view run: a reversed array of one algotype
        # must sort fully under the shared seed.
        import random as _random
        rng = _random.Random(42)
        values = list(range(12, 0, -1))
        cells = make_cells(values, ["B"] * len(values))
        res = simulate(cells, seed=42)
        sorted_ok = res.final_values == sorted(values) and res.final_error == 0
        print(f"[{'PASS' if sorted_ok else 'FAIL'}] deterministic Bubble run: "
              f"final_error={res.final_error}, swaps={res.swaps}, steps={res.steps}")
        raise SystemExit(0 if (ok and sorted_ok) else 1)

    if args.validate_catalog or args.compare_ab:
        ok, errs = validate_claim_boundary_catalog()
        print(f"[{'PASS' if ok else 'FAIL'}] Claim-boundary catalog schema validation ({len(errs)} errors)")
        for e in errs:
            print(f"  - {e}")
        if not ok:
            raise SystemExit(1)

    if args.compare_ab:
        presets = list(AB_PRESETS.keys()) if args.compare_ab == "all" else [args.compare_ab]
        for key in presets:
            la, lb = AB_PRESETS[key]
            ab = compare_lanes_ab(la, lb, n=args.n, reps=args.reps, seed=args.seed)
            print(f"\n=== Paired A/B Comparison: {key} (n={ab.n}, reps={ab.reps}, seed={ab.seed}) ===")
            print(f"  {ab.lane_a_name}: swaps={ab.mean_swaps_a:.1f}, err={ab.mean_final_error_a:.2f}, DG={ab.mean_dg_a:.3f}, peak_agg={ab.mean_peak_agg_a:.3f}, end_agg={ab.mean_final_agg_a:.3f}")
            print(f"  {ab.lane_b_name}: swaps={ab.mean_swaps_b:.1f}, err={ab.mean_final_error_b:.2f}, DG={ab.mean_dg_b:.3f}, peak_agg={ab.mean_peak_agg_b:.3f}, end_agg={ab.mean_final_agg_b:.3f}")
            print(
                f"  Paired Delta (B - A): "
                f"Δswaps={ab.delta_swaps_mean:+.1f}±{ab.delta_swaps_sem:.1f}, "
                f"Δerr={ab.delta_final_error_mean:+.2f}±{ab.delta_final_error_sem:.2f}, "
                f"ΔDG={ab.delta_dg_mean:+.3f}±{ab.delta_dg_sem:.3f}, "
                f"Δend_agg={ab.delta_final_agg_mean:+.3f}±{ab.delta_final_agg_sem:.3f}"
            )


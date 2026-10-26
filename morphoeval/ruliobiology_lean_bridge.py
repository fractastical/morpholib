#!/usr/bin/env python3
"""
Bridge to the Lean 4 ruliobiology engine (`morphoeval/lean/Ruliobiology.lean`).

Lean owns the rule, the developmental step, lifetime fitness, excision and the
adaptive-evolution loop, and proves the structural laws about those same
functions.  Python here does only three things:

  1. call the Lean engine and parse its JSON;
  2. cross-check Lean's phenotypes against the NumPy reference implementation in
     `models/2026-wolfram-ruliobiology.py` (same genome -> identical space-time
     diagram, lifetime, width and coding-orbit set);
  3. draw figures from the Lean output.

Usage:
  python3 morphoeval/ruliobiology_lean_bridge.py --check            # kernel-check + cross-check
  python3 morphoeval/ruliobiology_lean_bridge.py --figure           # evolve in Lean, render PNG
  python3 morphoeval/ruliobiology_lean_bridge.py --check --figure
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import pathlib
import subprocess
import sys
import time

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent.parent
LEAN_FILE = HERE / "morphoeval" / "lean" / "Ruliobiology.lean"
REF_SCRIPT = HERE / "models" / "2026-wolfram-ruliobiology.py"
OUT_PNG = HERE / "docs" / "media" / "wolfram_ruliobiology_lean_organism.png"


# --------------------------------------------------------------------------- #
# Lean engine
# --------------------------------------------------------------------------- #
def _lean(*args: str) -> dict:
    cmd = ["lean", "--run", str(LEAN_FILE), *args]
    out = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout
    # `lean --run` prints one JSON object on the first line.
    return json.loads(out.splitlines()[0])


def kernel_check() -> float:
    t0 = time.perf_counter()
    subprocess.run(["lean", str(LEAN_FILE)], capture_output=True, text=True, check=True)
    return time.perf_counter() - t0


def lean_develop(genome: str, cap: int) -> dict:
    return _lean("develop", genome, str(cap))


def lean_excise(genome: str, cap: int, t0: int, lo: int, hi: int, k: int) -> dict:
    return _lean("excise", genome, str(cap), str(t0), str(lo), str(hi), str(k))


def lean_evolve(seed: int, n_mut: int, cap: int) -> dict:
    return _lean("evolve", str(seed), str(n_mut), str(cap))


def stages_to_grid(stages: list[dict]) -> np.ndarray:
    """Exact space-time grid from (offset, cells) pairs; no re-centring needed."""
    lo = min(s["offset"] for s in stages)
    hi = max(s["offset"] + len(s["cells"]) for s in stages)
    grid = np.zeros((len(stages), hi - lo), dtype=np.int8)
    for t, s in enumerate(stages):
        a = s["offset"] - lo
        grid[t, a : a + len(s["cells"])] = [int(ch) for ch in s["cells"]]
    return grid


# --------------------------------------------------------------------------- #
# Cross-check against the NumPy reference implementation
# --------------------------------------------------------------------------- #
def _load_reference():
    spec = importlib.util.spec_from_file_location("ruliobio_ref", REF_SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _ref_rows(sim: dict) -> list[str]:
    """Trim each row of the reference grid to its live extent (it is palindromic)."""
    rows = []
    for row in sim["grid"]:
        nz = np.flatnonzero(row)
        rows.append("".join(str(int(v)) for v in row[nz[0] : nz[-1] + 1]) if nz.size else "")
    return rows


def cross_check(n_random: int, cap: int, rng_seed: int) -> dict:
    ref = _load_reference()
    orbits = ref.build_symmetric_k5_orbits(5)
    t2o = orbits["triple_to_orbit"]
    rng = np.random.default_rng(rng_seed)

    genomes = ["0" * 75]
    for _ in range(n_random):
        g = rng.integers(0, 5, size=75)
        g[0] = 0
        genomes.append("".join(map(str, g)))

    # Also every breakthrough genome of a lineage evolved *in Lean* (all finite-lifetime):
    # Python must agree on each phenotype.
    lineage = lean_evolve(seed=3, n_mut=400, cap=cap)
    genomes.extend(b["genome"] for b in lineage["breakthroughs"][1:])
    genomes.append(lineage["genome"])

    results = {"compared": 0, "mismatches": [], "terminated": 0, "lean_evolved_fitness": lineage["fitness"]}
    for gs in genomes:
        L = lean_develop(gs, cap)
        orbit_rule = np.array([int(ch) for ch in gs], dtype=np.int8)
        P = ref.simulate_rulioorganism(orbit_rule, t2o, max_steps=cap, record_grid=True)

        lean_rows = [s["cells"] for s in L["stages"]]
        ref_rows = _ref_rows(P) if P["terminated"] or P["lifetime"] else None

        issues = []
        if L["terminated"] != bool(P["terminated"]):
            # Python also aborts early on exact state repetition ("periodic trap"); both
            # report fitness 0 in that case, so compare fitness rather than the flag.
            if L["fitness"] != P["fitness"]:
                issues.append(f"fitness lean={L['fitness']} ref={P['fitness']}")
        else:
            if L["fitness"] != P["fitness"]:
                issues.append(f"fitness lean={L['fitness']} ref={P['fitness']}")
            if L["terminated"]:
                results["terminated"] += 1
                if L["max_width"] != P["max_width"]:
                    issues.append(f"max_width lean={L['max_width']} ref={P['max_width']}")
                if ref_rows is not None and lean_rows != ref_rows:
                    first = next(i for i, (a, b) in enumerate(zip(lean_rows, ref_rows)) if a != b)
                    issues.append(f"rows differ at t={first}: lean={lean_rows[first]} ref={ref_rows[first]}")
                lean_used = set(L["used_orbits"]) - {0}
                ref_used = set(P["used_orbits"]) - {0}
                if lean_used != ref_used:
                    issues.append(f"used orbits differ: {sorted(lean_used ^ ref_used)}")
        results["compared"] += 1
        if issues:
            results["mismatches"].append({"genome": gs, "issues": issues})
    return results


# --------------------------------------------------------------------------- #
# Drawing: large cells, hairline borders, quiet palette
# --------------------------------------------------------------------------- #
PAPER = "#f7f3ea"
PALETTE = ["#f7f3ea", "#b23a2f", "#4f7f52", "#5a4a7a", "#c9962b"]   # void, red, green, purple, gold
BORDER = "#f7f3ea"


def draw_cells(ax, grid: np.ndarray, border: float = 0.12) -> None:
    """Draw a space-time grid as bordered square cells (void cells are not drawn)."""
    from matplotlib.collections import PatchCollection
    from matplotlib.patches import Rectangle

    h, w = grid.shape
    patches, colors = [], []
    ys, xs = np.nonzero(grid)
    for y, x in zip(ys, xs):
        patches.append(Rectangle((x + border / 2, y + border / 2), 1 - border, 1 - border))
        colors.append(PALETTE[int(grid[y, x])])
    ax.add_collection(PatchCollection(patches, facecolors=colors, edgecolors="none"))
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.set_aspect("equal")
    ax.set_facecolor(PAPER)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def render_showcase(lineage: dict, cap: int, out: pathlib.Path, n_details: int = 3,
                    detail_steps: int = 110) -> pathlib.Path:
    """One organism: whole body on the left, `n_details` zoomed windows on the right,
    fitness trace across the top."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    champ = lean_develop(lineage["genome"], cap)
    grid = stages_to_grid(champ["stages"])
    h, w = grid.shape

    fig = plt.figure(figsize=(16, 15), facecolor=PAPER)
    gs = fig.add_gridspec(1 + n_details, 2, width_ratios=[1, 1.9],
                          height_ratios=[0.9] + [3.0] * n_details, hspace=0.12, wspace=0.04,
                          left=0.04, right=0.98, top=0.95, bottom=0.02)

    ax = fig.add_subplot(gs[0, :])
    trace = lineage["trace"]
    ax.step(range(1, len(trace) + 1), trace, where="post", color="#333", lw=1.2)
    for b in lineage["breakthroughs"][1:]:
        ax.plot(b["step"], b["fitness"], "o", color=PALETTE[1], ms=3.5)
    ax.set_facecolor(PAPER)
    ax.set_xlabel("point mutation", fontsize=9)
    ax.set_ylabel("lifetime", fontsize=9)
    ax.tick_params(labelsize=8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_title(f"lifetime {champ['lifetime']}, width {w}, {len(lineage['breakthroughs']) - 1} breakthroughs "
                 f"in {len(trace)} mutations from the null rule\ngenome {lineage['genome']}",
                 fontsize=9, loc="left", family="monospace")

    # whole body
    ax_body = fig.add_subplot(gs[1:, 0])
    draw_cells(ax_body, grid, border=0.0)
    ax_body.set_title("whole organism", fontsize=10)

    # detail windows, evenly spaced through life
    starts = [int(h * f) for f in np.linspace(0.08, 0.82, n_details)]
    for i, t0 in enumerate(starts):
        t1 = min(h, t0 + detail_steps)
        sub = grid[t0:t1]
        cols = np.flatnonzero(sub.any(axis=0))
        sub = sub[:, cols[0] : cols[-1] + 1]
        ax = fig.add_subplot(gs[1 + i, 1])
        draw_cells(ax, sub)
        ax.set_title(f"steps {t0}–{t1}", fontsize=10, loc="left")
        ax_body.axhspan(t0, t1, color="#333", alpha=0.07, lw=0)

    fig.savefig(out, dpi=110, facecolor=PAPER)
    plt.close(fig)
    return out


# --------------------------------------------------------------------------- #
# Figure
# --------------------------------------------------------------------------- #
def render_figure(seed: int, n_mut: int, cap: int) -> pathlib.Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lineage = lean_evolve(seed, n_mut, cap)
    champ = lean_develop(lineage["genome"], cap)
    grid = stages_to_grid(champ["stages"])

    # Excise the inner half of the live cells of a wide row (same rule as the reference
    # script: a row with >= 15 live cells, one third of the way through the wide rows).
    live_counts = [sum(ch != "0" for ch in s["cells"]) for s in champ["stages"]]
    wide = [t for t, n in enumerate(live_counts) if n >= 15] or [int(np.argmax(live_counts))]
    t0 = wide[len(wide) // 3]
    live = [i for i, ch in enumerate(champ["stages"][t0]["cells"]) if ch != "0"]
    lo, hi = live[len(live) // 4], live[(3 * len(live)) // 4] + 1
    ex = lean_excise(lineage["genome"], cap, t0, lo, hi, 24)
    iso = stages_to_grid(ex["isolated"]) if ex["isolated"] else np.zeros((1, 1), dtype=np.int8)
    situ = stages_to_grid(ex["in_situ"])

    fig = plt.figure(figsize=(14, 10), facecolor=PAPER)
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 2.2], width_ratios=[1.2, 1, 1])

    ax = fig.add_subplot(gs[0, :])
    trace = lineage["trace"]
    ax.step(range(1, len(trace) + 1), trace, where="post", color="#333", lw=1.4)
    for b in lineage["breakthroughs"][1:]:
        ax.plot(b["step"], b["fitness"], "o", color=PALETTE[1], ms=4)
    ax.set_facecolor(PAPER)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_xlabel("accepted or rejected point mutation")
    ax.set_ylabel("lifetime fitness")
    ax.set_title(
        f"Adaptive evolution in Lean (`evolve`, SplitMix64 seed {seed}, {n_mut} mutations, cap {cap}): "
        f"null rule → lifetime {lineage['fitness']}; ratchet_ge proves the trace is monotone"
    )

    ax = fig.add_subplot(gs[1, 0])
    draw_cells(ax, grid)
    ax.axhline(t0, color="#333", lw=0.8, ls="--")
    ax.set_title(f"`develop` from one seed: lifetime {champ['lifetime']}, max width {champ['max_width']}\n"
                 f"(exact offsets; step_reverse ⇒ every row is a palindrome)", fontsize=10)

    ax = fig.add_subplot(gs[1, 1])
    draw_cells(ax, situ)
    ax.set_title(f"organ cells [{lo},{hi}) at t={t0}, in situ\n({len(ex['in_situ'])} steps after excision)", fontsize=10)

    ax = fig.add_subplot(gs[1, 2])
    draw_cells(ax, iso)
    ax.set_title("same cells grown in isolation (`excise`)\nwin_interior: only 2 cells/side/step feel the context",
                 fontsize=10)

    fig.suptitle("Wolfram (2026) rulioorganism, computed by morphoeval/lean/Ruliobiology.lean", y=0.995)
    fig.tight_layout()
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=130, facecolor=PAPER)
    plt.close(fig)
    return OUT_PNG


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="kernel-check the Lean file and cross-check against the NumPy reference")
    ap.add_argument("--figure", action="store_true", help="evolve a lineage in Lean and render the figure")
    ap.add_argument("--showcase", metavar="LINEAGE_JSON_OR_SEED",
                    help="render one organism at full size from a saved `evolve` JSON (or evolve it now from this seed)")
    ap.add_argument("--showcase-out", type=pathlib.Path,
                    default=HERE / "docs" / "media" / "wolfram_ruliobiology_lean_showcase.png")
    ap.add_argument("--random", type=int, default=40, help="random genomes to cross-check")
    ap.add_argument("--cap", type=int, default=400)
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--mutations", type=int, default=1000)
    args = ap.parse_args()
    if not (args.check or args.figure or args.showcase):
        ap.error("pass --check, --figure and/or --showcase")

    if args.check:
        dt = kernel_check()
        print(f"[lean] {LEAN_FILE.relative_to(HERE)} kernel-checked in {dt:.2f}s (0 sorry)")
        res = cross_check(args.random, args.cap, rng_seed=0)
        print(f"[check] {res['compared']} genomes compared against the NumPy reference "
              f"({res['terminated']} finite-lifetime), Lean-evolved genome fitness {res['lean_evolved_fitness']}")
        if res["mismatches"]:
            for m in res["mismatches"]:
                print(f"  MISMATCH {m['genome'][:20]}…: {m['issues']}")
            sys.exit(1)
        print("[check] Lean and NumPy agree on every lifetime, width, row and coding-orbit set")

    if args.figure:
        path = render_figure(args.seed, args.mutations, args.cap)
        print(f"[figure] {path.relative_to(HERE)}")

    if args.showcase:
        src = pathlib.Path(args.showcase)
        if src.exists():
            lineage = json.loads(src.read_text().splitlines()[0])
        else:
            lineage = lean_evolve(int(args.showcase), args.mutations, args.cap)
        path = render_showcase(lineage, args.cap, args.showcase_out)
        print(f"[showcase] lifetime {lineage['fitness']} -> {path}")


if __name__ == "__main__":
    main()

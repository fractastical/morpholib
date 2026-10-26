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
PAPER = "#ffffff"
PALETTE = ["#ffffff", "#c8553d", "#4f8a6b", "#6c5b7b", "#e0a93c"]   # void, red, green, purple, gold
INK = "#2b2b2b"
INK_SOFT = "#8a8a8a"
RULE = "#d9d9d9"


def _rgb(grid: np.ndarray) -> np.ndarray:
    from matplotlib.colors import to_rgb
    lut = np.array([to_rgb(c) for c in PALETTE])
    return lut[grid]


def _quiet(ax) -> None:
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_facecolor(PAPER)
    for s in ax.spines.values():
        s.set_visible(False)


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
    _quiet(ax)


def render_showcase(lineage: dict, cap: int, out: pathlib.Path, n_tiles: int = 4,
                    tile: int = 40, cell_px: float = 3.0, tile_cell_px: float = 8.5,
                    dpi: int = 150) -> pathlib.Path:
    """A plate: the whole organism in vertical strips at a fixed cell size, a row of
    equal square detail tiles at a larger cell size, a sparkline of the lineage's
    fitness, and a one-line caption. No axes, ticks or frames."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    champ = lean_develop(lineage["genome"], cap)
    stages = champ["stages"]
    grid = stages_to_grid(stages)
    h, w = grid.shape
    axis_col = -min(s["offset"] for s in stages)          # column of the seed cell

    # --- geometry (inches) ---------------------------------------------------
    cell = cell_px / dpi
    tcell = tile_cell_px / dpi
    n_strips = max(1, int(round(np.sqrt(1.6 * h / w))))  # strip grid ≈ 16:10
    strip_h = -(-h // n_strips)
    gap, margin = 0.30, 0.55
    strips_w_in = n_strips * w * cell + (n_strips - 1) * gap
    strips_h_in = strip_h * cell
    tile_in = tile * tcell
    tiles_w_in = n_tiles * tile_in + (n_tiles - 1) * gap
    content_w = max(strips_w_in, tiles_w_in)
    title_h, spark_h, label_h, caption_h = 0.50, 0.55, 0.22, 0.30
    fig_w = content_w + 2 * margin
    fig_h = (margin + title_h + spark_h + 0.35 + label_h + strips_h_in + 0.45
             + label_h + tile_in + caption_h + margin)

    fig = plt.figure(figsize=(fig_w, fig_h), facecolor=PAPER)
    X = lambda x_in: x_in / fig_w                         # inches -> figure fraction
    Y = lambda y_in: 1 - y_in / fig_h                     # from the top
    y = margin

    # --- title ----------------------------------------------------------------
    n_breaks = len(lineage["breakthroughs"]) - 1
    fig.text(X(margin), Y(y), "Rulioorganism", fontsize=13, color=INK, ha="left", va="top")
    fig.text(X(margin), Y(y + 0.26),
             f"symmetric 5-colour cellular automaton · {len(lineage['trace'])} point mutations from the null rule · "
             f"{n_breaks} breakthroughs · lifetime {champ['lifetime']} · width {w}",
             fontsize=8.5, color=INK_SOFT, ha="left", va="top")
    y += title_h

    # --- sparkline --------------------------------------------------------------
    ax = fig.add_axes([X(margin), Y(y + spark_h), X(content_w), spark_h / fig_h])
    trace = lineage["trace"]
    ax.step(range(1, len(trace) + 1), trace, where="post", color=INK, lw=0.9)
    for b in lineage["breakthroughs"][1:]:
        ax.plot(b["step"], b["fitness"], "o", color=PALETTE[1], ms=3, mec=PAPER, mew=0.6)
    ax.set_xlim(0, len(trace) * 1.04)
    ax.set_ylim(-0.04 * max(trace), 1.15 * max(trace))
    ax.text(len(trace) * 1.005, trace[-1], f"{trace[-1]}", fontsize=7.5, color=INK, va="center")
    ax.text(0, 1.12 * max(trace), "lifetime fitness along the lineage, one step per point mutation",
            fontsize=7.5, color=INK_SOFT, va="top")
    ax.axhline(0, color=RULE, lw=0.6)
    _quiet(ax)
    y += spark_h + 0.35

    # --- body in strips --------------------------------------------------------------
    x0 = margin + (content_w - strips_w_in) / 2
    for i in range(n_strips):
        t0, t1 = i * strip_h, min(h, (i + 1) * strip_h)
        sub = np.zeros((strip_h, w), dtype=np.int8)
        sub[: t1 - t0] = grid[t0:t1]
        left = x0 + i * (w * cell + gap)
        fig.text(X(left), Y(y), f"steps {t0}–{t1}", fontsize=7.5, color=INK_SOFT, ha="left", va="top")
        ax = fig.add_axes([X(left), Y(y + label_h + strips_h_in), (w * cell) / fig_w, strips_h_in / fig_h])
        ax.imshow(_rgb(sub), interpolation="nearest", aspect="auto")
        _quiet(ax)
    y += label_h + strips_h_in + 0.45

    # --- detail tiles ---------------------------------------------------------------
    x0 = margin + (content_w - tiles_w_in) / 2
    fig.text(X(x0), Y(y), f"detail · {tile} × {tile} cells, the densest tissue in each quarter of life",
             fontsize=7.5, color=INK_SOFT, ha="left", va="top")
    live = (grid > 0).astype(np.int32)
    # summed-area table -> live-cell count of any tile window in O(1)
    S = np.zeros((h + 1, w + 1), dtype=np.int64)
    S[1:, 1:] = live.cumsum(0).cumsum(1)
    def count(t0: int, c0: int) -> int:
        t1, c1 = min(h, t0 + tile), min(w, c0 + tile)
        return int(S[t1, c1] - S[t0, c1] - S[t1, c0] + S[t0, c0])
    bands = np.linspace(0, max(1, h - tile), n_tiles + 1).astype(int)
    for i in range(n_tiles):
        best, best_tc = -1, (bands[i], max(0, axis_col - tile // 2))
        for t0 in range(bands[i], max(bands[i] + 1, bands[i + 1]), 4):
            for c0 in range(0, max(1, w - tile + 1), 2):
                n = count(t0, c0)
                if n > best:
                    best, best_tc = n, (t0, c0)
        t0, c0 = best_tc
        sub = np.zeros((tile, tile), dtype=np.int8)
        src = grid[t0 : t0 + tile, c0 : c0 + tile]
        sub[: src.shape[0], : src.shape[1]] = src
        left = x0 + i * (tile_in + gap)
        ax = fig.add_axes([X(left), Y(y + label_h + tile_in), tile_in / fig_w, tile_in / fig_h])
        draw_cells(ax, sub, border=0.16)
        ax.text(0, tile + 1.2, f"t = {t0}–{t0 + tile}", fontsize=7, color=INK_SOFT, va="top")
    y += label_h + tile_in + 0.08

    # --- caption ----------------------------------------------------------------------
    fig.text(X(margin), Y(y + caption_h), f"genome {lineage['genome']}", fontsize=7, color=INK_SOFT,
             family="monospace", ha="left", va="bottom")
    fig.text(X(margin + content_w), Y(y + caption_h),
             "computed and proved in morphoeval/lean/Ruliobiology.lean", fontsize=7, color=INK_SOFT,
             ha="right", va="bottom")

    fig.savefig(out, dpi=dpi, facecolor=PAPER)
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

#!/usr/bin/env python3
"""
t1917-019_the-forms-of-tissues-or-cell-aggregates.py — Thompson (1917) claim T1917-019

    "Tissue cells tessellate approximately as hexagons / as a froth, the configuration
     that partitions area with least perimeter."
    D'Arcy W. Thompson, On Growth and Form (1917), Ch. VII "The Forms of Tissues, or
    Cell-Aggregates", pp. 293-345, Figs. 120-127 (IA scan ongrowthform1917thom).

Pass 1 (baseline v1).  The claim is decomposed into three testable parts (see
datasets/thompson_1917/t1917-019/t1917_019_the_forms_of_tissues_or_cell_a_reference.json):

  C1  froth / mean six      — trivalent network, <n> = 6 - 6 chi / F        (E1, Thompson p. 325)
  C2  least perimeter       — honeycomb bound  P_6 = 2 (12)^(1/4) sqrt(A)   (E3, Thompson p. 330; Hales 2001)
  C3  approximately hexagons — p6 is modal and >= 0.5 (strict reading)      (Thompson p. 321)

What v1 implements
  * a periodic (torus) Voronoi froth morphospace: hexagonal seed lattice + positional
    disorder delta, optional Lloyd (centroidal-Voronoi) relaxation = "froth relaxing
    toward least perimeter";
  * exact polygon statistics on the froth: p_n, <n>, mu2, shape index P/sqrt(A),
    excess perimeter over the isoperimetric honeycomb bound (E2, E3), Lewis's law (E4)
    and Aboav-Weaire (E5) fits;
  * a reconstruction of the Gibson et al. (2006) division-topology Markov chain (E6);
  * side-by-side figure: the vendored human corneal endothelium specimen, Thompson's
    own Fig. 125 soap froth (p. 322), and the model froth; plus a relaxation GIF.
  * measure_morphoeval() for morphoeval/measure_outputs.py.

Usage
  python3 thompson/claims/t1917-019_the-forms-of-tissues-or-cell-aggregates.py --self-test
  python3 thompson/claims/t1917-019_the-forms-of-tissues-or-cell-aggregates.py

Dependencies: numpy, scipy, matplotlib, Pillow (requirements.txt).  Deterministic
(seed 20261028).  All outputs are written under docs/media/ and
datasets/thompson_1917/t1917-019/.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
from scipy.spatial import Voronoi

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.collections import PolyCollection  # noqa: E402
from PIL import Image  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
DATA = ROOT / "datasets" / "thompson_1917" / "t1917-019"
MEDIA = ROOT / "docs" / "media"
SLUG = "t1917_019_the_forms_of_tissues_or_cell_a"
SIM_ID = "thompson1917-t1917-019"
SEED = 20261028

# E3: regular-polygon shape index P / sqrt(A) = 2 sqrt(n tan(pi/n))
HEX_SHAPE_INDEX = 2.0 * 12.0 ** 0.25              # 3.7224, Hales honeycomb constant x 2
SHAPE_INDEX = {n: 2.0 * math.sqrt(n * math.tan(math.pi / n)) for n in range(3, 13)}

# Literature reference values (reference.json -> reference_values)
GIBSON_2006_MARKOV = {4: 0.029, 5: 0.289, 6: 0.464, 7: 0.208, 8: 0.010}
CORNEA_NORM_P6 = 0.616          # Sheng & Bullimore 2007, age 40, no CL, Rx 0
CORNEA_NORM_CV_AREA = 0.315     # idem (CoV 31.5 %)
POISSON_VORONOI_P6 = 0.2947     # Weaire & Rivier 1984
POISSON_VORONOI_MU2 = 1.78

SPECIMEN = DATA / "human_corneal_endothelium_specimen.jpg"
THOMPSON_P322 = DATA / "thompson_1917_p322_fig125_126_froth_elodea.jpg"
# Fig. 125 panels on the vendored p. 322 scan (1904 x 3128 px), inside the glass-plate rules
FIG125_A_BOX = (560, 1096, 1400, 1306)
FIG125_B_BOX = (560, 1456, 1400, 1662)


# --------------------------------------------------------------------------- #
# 1. Periodic Voronoi froth morphospace (E1-E5 computed exactly on polygons)
# --------------------------------------------------------------------------- #
def hex_lattice(nx: int, ny: int) -> tuple[np.ndarray, float, float]:
    """Regular hexagonal (triangular) seed lattice on a torus; ny must be even."""
    if ny % 2:
        raise ValueError("ny must be even for a periodic hexagonal lattice")
    a = 1.0
    h = math.sqrt(3.0) / 2.0 * a
    pts = []
    for j in range(ny):
        for i in range(nx):
            pts.append(((i + 0.5 * (j % 2)) * a, j * h))
    return np.array(pts), nx * a, ny * h


def periodic_voronoi(points: np.ndarray, Lx: float, Ly: float) -> dict:
    """Voronoi tessellation of a periodic box via 3x3 tiling.

    Returns per-cell neighbour lists (indices mod N), polygon vertices, areas,
    perimeters and side counts for the N central cells."""
    N = len(points)
    shifts = [(dx, dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1)]
    tiled = np.concatenate([points + np.array([dx * Lx, dy * Ly]) for dx, dy in shifts])
    # central copy is shift (0,0) -> index 4 in the shifts list
    central_offset = shifts.index((0, 0)) * N
    vor = Voronoi(tiled)
    neighbours: list[set] = [set() for _ in range(N)]
    for p, q in vor.ridge_points:
        for a, b in ((p, q), (q, p)):
            if central_offset <= a < central_offset + N:
                neighbours[a - central_offset].add(int(b % N))
    polys, areas, perims = [], np.zeros(N), np.zeros(N)
    for i in range(N):
        region = vor.regions[vor.point_region[central_offset + i]]
        poly = vor.vertices[region]
        polys.append(poly)
        x, y = poly[:, 0], poly[:, 1]
        areas[i] = 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
        perims[i] = np.sum(np.hypot(np.diff(x, append=x[0]), np.diff(y, append=y[0])))
    sides = np.array([len(nb) for nb in neighbours])
    return {"neighbours": [sorted(nb) for nb in neighbours], "polys": polys,
            "areas": areas, "perims": perims, "sides": sides, "Lx": Lx, "Ly": Ly}


def lloyd_step(points: np.ndarray, Lx: float, Ly: float) -> np.ndarray:
    """One centroidal-Voronoi (Lloyd) relaxation step on the torus."""
    vor = periodic_voronoi(points, Lx, Ly)
    new = np.empty_like(points)
    for i, poly in enumerate(vor["polys"]):
        x, y = poly[:, 0], poly[:, 1]
        cross = x * np.roll(y, -1) - np.roll(x, -1) * y
        A = 0.5 * cross.sum()
        cx = np.sum((x + np.roll(x, -1)) * cross) / (6.0 * A)
        cy = np.sum((y + np.roll(y, -1)) * cross) / (6.0 * A)
        new[i] = (cx % Lx, cy % Ly)
    return new


def froth(delta: float, lloyd_iters: int = 0, nx: int = 20, ny: int = 20,
          seed: int = SEED) -> dict:
    """Froth morphospace point: hexagonal lattice jittered by Gaussian displacement of
    s.d. `delta` lattice spacings (delta = 0 honeycomb; delta >~ 0.7 ~ Poisson-Voronoi),
    then `lloyd_iters` relaxation steps.  Returns periodic_voronoi() dict + metadata."""
    rng = np.random.default_rng(seed)
    pts, Lx, Ly = hex_lattice(nx, ny)
    pts = (pts + rng.normal(0.0, delta, size=pts.shape)) % (Lx, Ly)
    for _ in range(lloyd_iters):
        pts = lloyd_step(pts, Lx, Ly)
    vor = periodic_voronoi(pts, Lx, Ly)
    vor.update({"points": pts, "delta": delta, "lloyd_iters": lloyd_iters, "seed": seed})
    return vor


# --------------------------------------------------------------------------- #
# 2. Polygon statistics (E1 mean six, E2 mu2, E3 excess perimeter, E4 Lewis, E5 Aboav-Weaire)
# --------------------------------------------------------------------------- #
def polygon_statistics(sides: np.ndarray, areas: np.ndarray, perims: np.ndarray,
                       neighbours: list[list[int]], keep: np.ndarray | None = None) -> dict:
    """Topological and isoperimetric statistics of a cell tessellation.

    `keep` selects the cells that enter the statistics (interior cells); neighbour
    side numbers for Aboav-Weaire are taken from the full tessellation."""
    sides = np.asarray(sides); areas = np.asarray(areas, float); perims = np.asarray(perims, float)
    if keep is None:
        keep = np.ones(len(sides), bool)
    idx = np.flatnonzero(keep)
    n = sides[idx]
    F = len(idx)
    mean_n = float(n.mean())
    mu2 = float(np.mean((n - 6.0) ** 2))
    classes = list(range(3, 13))
    p_n = {k: float(np.mean(n == k)) for k in classes}
    # E3 — excess perimeter over the honeycomb bound for the SAME cell areas
    A = areas[idx]; P = perims[idx]
    p6_bound = HEX_SHAPE_INDEX * np.sqrt(A)
    excess_perimeter = float(P.sum() / p6_bound.sum() - 1.0)
    shape_index = P / np.sqrt(A)
    # E4 — Lewis's law: <A_n>/<A> = 1 + lambda (n - 6)
    lewis_x, lewis_y = [], []
    for k in classes:
        sel = n == k
        if sel.sum() >= 3:
            lewis_x.append(k); lewis_y.append(float(A[sel].mean() / A.mean()))
    lewis = _linfit(lewis_x, lewis_y)
    # E5 — Aboav-Weaire: n m(n) = (6 - a) n + (6a + mu2)
    m_i = np.array([np.mean(sides[neighbours[i]]) if len(neighbours[i]) else np.nan for i in idx])
    aw_x, aw_y = [], []
    for k in classes:
        sel = (n == k) & np.isfinite(m_i)
        if sel.sum() >= 3:
            aw_x.append(k); aw_y.append(float(k * m_i[sel].mean()))
    aw = _linfit(aw_x, aw_y)
    a_aboav = 6.0 - aw["slope"] if aw["slope"] is not None else None
    sum_rule_lhs = float(np.nanmean(n * m_i))         # (1/F) sum n_i m(n_i)
    sum_rule_rhs = 36.0 + mu2                          # Law 5 (holds exactly when <n> = 6)
    return {
        "F": int(F), "mean_n": mean_n, "mu2": mu2, "p_n": p_n, "p6": p_n[6],
        "modal_class": int(max(p_n, key=p_n.get)),
        "excess_perimeter": excess_perimeter,
        "shape_index_mean": float(shape_index.mean()),
        "shape_index_median": float(np.median(shape_index)),
        "area_cv": float(A.std() / A.mean()),
        "lewis": {"lambda": lewis["slope"], "intercept": lewis["intercept"], "r": lewis["r"],
                  "classes": lewis_x, "rel_area": lewis_y},
        "aboav_weaire": {"a": a_aboav, "slope": aw["slope"], "intercept": aw["intercept"], "r": aw["r"],
                         "classes": aw_x, "n_m_n": aw_y,
                         "sum_rule_lhs": sum_rule_lhs, "sum_rule_rhs": sum_rule_rhs,
                         "sum_rule_residual": sum_rule_lhs - sum_rule_rhs},
    }


def _linfit(x, y) -> dict:
    x = np.asarray(x, float); y = np.asarray(y, float)
    if len(x) < 2:
        return {"slope": None, "intercept": None, "r": None}
    slope, intercept = np.polyfit(x, y, 1)
    r = float(np.corrcoef(x, y)[0, 1]) if len(x) > 2 else 1.0
    return {"slope": float(slope), "intercept": float(intercept), "r": r}


# --------------------------------------------------------------------------- #
# 3. Gibson et al. (2006) division-topology Markov chain (E6) — reconstruction
# --------------------------------------------------------------------------- #
def gibson_markov_fixed_point(n_min: int = 3, n_max: int = 16, p_cut: float = 1.0 / 3.0,
                              cleavage: str = "through_centre") -> dict:
    """Mean-field reconstruction of the Gibson 2006 polygon-class Markov chain.

    One generation: an s-gon first gains k ~ Binomial(s, p_cut) sides from its dividing
    neighbours (a neighbour's cleavage plane cuts the shared side with probability
    2/s' ~ 1/3 for s' = 6), then divides along a new wall joining two of its m = s + k
    sides; with side gap g the daughters have g + 2 and m - g + 2 sides (Lean Law 6:
    the daughters sum to m + 4, so the mean of six is preserved for ANY cleavage rule).
      cleavage = "through_centre": the wall passes through the cell centre, so the two
                 cut sides are as nearly opposite as possible, g in {floor(m/2), ceil(m/2)}
                 (Gibson's assumption of equal daughters);
      cleavage = "uniform_random": the two cut sides are chosen uniformly among all
                 C(m, 2) pairs — an ablation that produces many 3- and 4-gons.
    Returns the stationary distribution (power iteration)."""
    from math import comb
    states = list(range(n_min, n_max + 1))
    T = np.zeros((len(states), len(states)))
    for si, s in enumerate(states):
        for k in range(0, s + 1):
            pk = comb(s, k) * p_cut ** k * (1 - p_cut) ** (s - k)
            m = s + k
            if cleavage == "through_centre":
                gaps = sorted({m // 2, (m + 1) // 2})
                weights = {g: 1.0 / len(gaps) for g in gaps}
            elif cleavage == "uniform_random":
                pairs = comb(m, 2)
                weights = {g: (m - g) / pairs for g in range(1, m)}
            else:
                raise ValueError(cleavage)
            for g, wg in weights.items():
                for d in (g + 2, m - g + 2):
                    d = min(max(d, n_min), n_max)
                    T[si, states.index(d)] += pk * wg * 0.5
    T /= T.sum(1, keepdims=True)
    pi = np.ones(len(states)) / len(states)
    for _ in range(4000):
        pi = pi @ T
    pi = pi / pi.sum()
    dist = {s: float(p) for s, p in zip(states, pi)}
    mean_n = sum(s * p for s, p in dist.items())
    mu2 = sum((s - 6.0) ** 2 * p for s, p in dist.items())
    return {"distribution": dist, "mean_n": float(mean_n), "mu2": float(mu2), "p6": dist[6],
            "p_cut": p_cut, "cleavage": cleavage,
            "transition_matrix_rows_sum_to_one": bool(np.allclose(T.sum(1), 1.0))}


# --------------------------------------------------------------------------- #
# 4. Rendering
# --------------------------------------------------------------------------- #
POLY_COLORS = {3: "#7b3294", 4: "#c2a5cf", 5: "#f4a582", 6: "#f7f7f7", 7: "#92c5de", 8: "#0571b0", 9: "#023858"}


def draw_froth(ax, vor: dict, title: str = "", keep: np.ndarray | None = None) -> None:
    polys = vor["polys"]; sides = vor["sides"]
    cols = [POLY_COLORS.get(int(min(max(s, 3), 9)), "#333333") for s in sides]
    if keep is not None:
        cols = [c if k else "#dddddd" for c, k in zip(cols, keep)]
    pc = PolyCollection(polys, facecolors=cols, edgecolors="k", linewidths=0.6)
    ax.add_collection(pc)
    ax.set_xlim(0, vor["Lx"]); ax.set_ylim(0, vor["Ly"]); ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=9)


def polygon_legend(ax):
    from matplotlib.patches import Patch
    handles = [Patch(facecolor=POLY_COLORS[k], edgecolor="k", label=f"{k}-gon") for k in (4, 5, 6, 7, 8)]
    ax.legend(handles=handles, loc="lower center", ncol=5, fontsize=7, frameon=False, bbox_to_anchor=(0.5, -0.12))


def render_comparison(model: dict, model_stats: dict, gibson: dict, out_png: Path) -> None:
    fig = plt.figure(figsize=(14, 9))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.15, 1.0])
    # (a) specimen
    ax = fig.add_subplot(gs[0, 0])
    if SPECIMEN.exists():
        ax.imshow(Image.open(SPECIMEN))
    ax.set_title("Human corneal endothelium in situ (alizarin red)\nZhiguo He, Wikimedia Commons CC BY-SA 4.0", fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])
    # (b) Thompson Fig. 125 A/B
    ax = fig.add_subplot(gs[0, 1])
    if THOMPSON_P322.exists():
        page = Image.open(THOMPSON_P322)
        a = page.crop(FIG125_A_BOX); b = page.crop(FIG125_B_BOX)
        canvas = Image.new("RGB", (a.width, a.height + b.height + 20), "white")
        canvas.paste(a, (0, 0)); canvas.paste(b, (0, a.height + 20))
        ax.imshow(canvas)
    ax.set_title("Thompson (1917) Fig. 125, p. 322 — soap-froth under pressure\n(after Rhumbler): A compressed, B relaxed", fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])
    # (c) model froth
    ax = fig.add_subplot(gs[0, 2])
    draw_froth(ax, model, f"Periodic Voronoi froth, delta = {model['delta']:.2f}, Lloyd = {model['lloyd_iters']}\n"
                          f"<n> = {model_stats['mean_n']:.3f}, mu2 = {model_stats['mu2']:.2f}, p6 = {model_stats['p6']:.2f}, "
                          f"excess perimeter = {100 * model_stats['excess_perimeter']:.1f} %")
    polygon_legend(ax)
    # (d) polygon-class distributions
    ax = fig.add_subplot(gs[1, 0])
    ks = [4, 5, 6, 7, 8]
    w = 0.27
    ax.bar([k - w for k in ks], [1.0 if k == 6 else 0.0 for k in ks], w, color="#f7f7f7", edgecolor="k", label="honeycomb (Thompson strict)")
    ax.bar(ks, [model_stats["p_n"][k] for k in ks], w, color="#92c5de", edgecolor="k", label=f"froth model delta={model['delta']:.2f}")
    ax.bar([k + w for k in ks], [GIBSON_2006_MARKOV[k] for k in ks], w, color="#f4a582", edgecolor="k", label="Gibson 2006 wing disc (Markov)")
    ax.plot(ks, [gibson["distribution"].get(k, 0.0) for k in ks], "k.--", lw=1, label=f"division-topology chain (reconstruction), p6={gibson['p6']:.2f}")
    ax.axhline(CORNEA_NORM_P6, color="#b2182b", ls=":", lw=1); ax.text(8.4, CORNEA_NORM_P6 + 0.01, "cornea norm p6", color="#b2182b", fontsize=7, ha="right")
    ax.set_xlabel("polygon class n"); ax.set_ylabel("fraction p_n"); ax.set_ylim(0, 1.05)
    ax.legend(fontsize=7, frameon=False); ax.set_title("Polygon-class distributions (E1/E2)", fontsize=9)
    # (e) Lewis's law
    ax = fig.add_subplot(gs[1, 1])
    lw_ = model_stats["lewis"]
    ax.plot(lw_["classes"], lw_["rel_area"], "o", color="#0571b0", label="froth model <A_n>/<A>")
    xs = np.array([4, 8]); ax.plot(xs, 1 + 0.25 * (xs - 6), "k--", lw=1, label="Lewis 1928: 1 + (n-6)/4")
    if lw_["lambda"] is not None:
        ax.plot(xs, lw_["intercept"] + lw_["lambda"] * xs, "-", color="#0571b0", lw=1, label=f"fit lambda = {lw_['lambda']:.3f}, r = {lw_['r']:.2f}")
    ax.set_xlabel("n"); ax.set_ylabel("relative mean area"); ax.legend(fontsize=7, frameon=False)
    ax.set_title("Lewis's law (E4)", fontsize=9)
    # (f) Aboav-Weaire
    ax = fig.add_subplot(gs[1, 2])
    aw = model_stats["aboav_weaire"]
    ax.plot(aw["classes"], aw["n_m_n"], "s", color="#7b3294", label="froth model n m(n)")
    if aw["slope"] is not None:
        ax.plot(xs, aw["intercept"] + aw["slope"] * xs, "-", color="#7b3294", lw=1, label=f"fit: a = {aw['a']:.2f}, r = {aw['r']:.2f}")
    ax.plot(xs, (6 - 1.2) * xs + 6 * 1.2 + model_stats["mu2"], "k--", lw=1, label="Aboav-Weaire a = 1.2 (froths)")
    ax.set_xlabel("n"); ax.set_ylabel("n m(n)"); ax.legend(fontsize=7, frameon=False)
    ax.set_title(f"Aboav-Weaire (E5); sum rule residual = {aw['sum_rule_residual']:+.3f}", fontsize=9)
    fig.suptitle("T1917-019  Thompson (1917) Ch. VII: tissue cells as hexagons / froth — Pass 1 baseline", fontsize=12)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=110)
    plt.close(fig)


def render_relaxation_gif(out_gif: Path, delta: float = 0.6, iters: int = 24, nx: int = 16, ny: int = 16) -> list[dict]:
    """Lloyd relaxation of a disordered froth: p6 rises, excess perimeter falls."""
    rng = np.random.default_rng(SEED + 1)
    pts, Lx, Ly = hex_lattice(nx, ny)
    pts = (pts + rng.normal(0.0, delta, size=pts.shape)) % (Lx, Ly)
    frames, trace = [], []
    for it in range(iters + 1):
        vor = periodic_voronoi(pts, Lx, Ly)
        st = polygon_statistics(vor["sides"], vor["areas"], vor["perims"], vor["neighbours"])
        trace.append({"iter": it, "p6": st["p6"], "mu2": st["mu2"], "excess_perimeter": st["excess_perimeter"]})
        fig, ax = plt.subplots(figsize=(5, 5.4))
        vor.update({"Lx": Lx, "Ly": Ly})
        draw_froth(ax, vor, f"Froth relaxation (Lloyd step {it}/{iters})\n"
                            f"p6 = {st['p6']:.2f}   mu2 = {st['mu2']:.2f}   excess perimeter = {100 * st['excess_perimeter']:.1f} %")
        fig.tight_layout()
        fig.canvas.draw()
        frames.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()))
        plt.close(fig)
        pts = lloyd_step(pts, Lx, Ly)
    out_gif.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(out_gif, save_all=True, append_images=frames[1:], duration=160, loop=0, optimize=False)
    return trace


# --------------------------------------------------------------------------- #
# 5. Pipeline, metrics, measure_morphoeval
# --------------------------------------------------------------------------- #
def sha256(path: Path) -> str | None:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None


def run(canonical_delta: float = 0.35) -> dict:
    t0 = time.time()
    model = froth(canonical_delta, 0)
    model_stats = polygon_statistics(model["sides"], model["areas"], model["perims"], model["neighbours"])
    honeycomb = froth(0.0, 0)
    honey_stats = polygon_statistics(honeycomb["sides"], honeycomb["areas"], honeycomb["perims"], honeycomb["neighbours"])
    gibson = gibson_markov_fixed_point()
    out_png = MEDIA / f"{SLUG}_comparison.png"
    out_gif = MEDIA / f"{SLUG}_growth.gif"
    render_comparison(model, model_stats, gibson, out_png)
    trace = render_relaxation_gif(out_gif)
    metrics = {
        "sim_id": SIM_ID, "claim_id": "T1917-019", "pass": 1, "seed": SEED,
        "model": {"delta": canonical_delta, "lloyd_iters": 0, "n_cells": int(model["sides"].size), **model_stats},
        "honeycomb_control": {"delta": 0.0, **honey_stats},
        "gibson_division_chain": gibson,
        "relaxation_trace": trace,
        "specimen_sha256": sha256(SPECIMEN),
        "outputs": {"comparison_png": str(out_png.relative_to(ROOT)), "growth_gif": str(out_gif.relative_to(ROOT))},
        "runtime_s": round(time.time() - t0, 2),
    }
    (DATA / f"{SLUG}_metrics.json").write_text(json.dumps(metrics, indent=1, default=float))
    return metrics


def measure_morphoeval() -> list[dict]:
    """Quantitative morphometrics for morphoeval/measure_outputs.py (dynamic discovery)."""
    mpath = DATA / f"{SLUG}_metrics.json"
    m = json.loads(mpath.read_text()) if mpath.exists() else run()
    md = m["model"]; hc = m["honeycomb_control"]; gb = m["gibson_division_chain"]

    def M(name, value, unit, method, reference=None, source=None, note=None, derived_from="script"):
        d = {"name": name, "value": None if value is None else float(round(value, 4)), "unit": unit,
             "method": method, "derived_from": derived_from}
        if reference is not None: d["reference_value"] = reference
        if source: d["reference_source"] = source
        if note: d["note"] = note
        return d

    return [
        M("froth model mean polygon side number <n> (torus)", md["mean_n"], "sides",
          f"periodic Voronoi, delta={md['delta']}, {md['n_cells']} cells; Lean Law 1 torus_mean_six", reference=6.0,
          source="Thompson 1917 p. 325; Weaire & Rivier 1984 Contemp. Phys. 25:59 Eq. 2.1"),
        M("froth model hexagon fraction p6", md["p6"], "fraction", "polygon_statistics on model froth",
          reference=GIBSON_2006_MARKOV[6], source="Gibson et al. 2006 Nature 442:1038 Fig. 3 (Drosophila wing disc Markov fixed point)"),
        M("froth model second moment mu2", md["mu2"], "sides^2", "sum p_n (n-6)^2; honeycomb control = 0",
          reference=0.0, source="honeycomb (Thompson strict reading); Lean Law 2 honeycomb_iff_zero_second_moment",
          note=f"honeycomb control mu2 = {hc['mu2']:.3f}; Poisson-Voronoi null ~ {POISSON_VORONOI_MU2}"),
        M("froth model excess perimeter over honeycomb bound", 100 * md["excess_perimeter"], "%",
          "sum P_i / sum 2*12^(1/4) sqrt(A_i) - 1 (E3, Hales 2001)", reference=0.0,
          source="Hales 2001 Discrete Comput. Geom. 25:1 (honeycomb is the least-perimeter equal-area partition)",
          note=f"honeycomb control = {100 * hc['excess_perimeter']:.2f} %"),
        M("froth model Lewis slope lambda", md["lewis"]["lambda"], "1/side", "linear fit of <A_n>/<A> vs n",
          reference=0.25, source="Lewis 1928 Anat. Rec. 38:341 (A_n proportional to n-2)"),
        M("froth model Aboav-Weaire a", md["aboav_weaire"]["a"], "dimensionless", "fit n m(n) = (6-a) n + (6a + mu2)",
          reference=1.2, source="Aboav 1970 Metallography 3:383; Weaire & Rivier 1984"),
        M("Aboav-Weaire sum-rule residual (1/F) sum n m(n) - (36 + mu2)", md["aboav_weaire"]["sum_rule_residual"], "sides^2",
          "Lean Law 5 aboav_weaire_sum_rule consistency check", reference=0.0, source="Weaire & Rivier 1984 Eq. 2.7"),
        M("division-topology chain fixed-point p6 (reconstruction)", gb["p6"], "fraction",
          "mean-field Gibson 2006 Markov chain, p_cut = 1/3", reference=GIBSON_2006_MARKOV[6],
          source="Gibson et al. 2006 Nature 442:1038"),
    ]


def self_test() -> int:
    t0 = time.time()
    # Law 1: torus mean six for several disorders (Euler, exact)
    for d in (0.0, 0.2, 0.5, 1.0):
        v = froth(d, 0, nx=12, ny=12)
        assert abs(v["sides"].mean() - 6.0) < 1e-9, f"torus mean {v['sides'].mean()} != 6 at delta={d}"
    # Law 2: honeycomb has mu2 = 0, p6 = 1 and (Law 3) sits on the Hales bound
    hc = froth(0.0, 0, nx=12, ny=12)
    st = polygon_statistics(hc["sides"], hc["areas"], hc["perims"], hc["neighbours"])
    assert st["mu2"] == 0.0 and st["p6"] == 1.0
    assert abs(st["excess_perimeter"]) < 1e-6, st["excess_perimeter"]
    assert abs(st["shape_index_mean"] - HEX_SHAPE_INDEX) < 1e-6
    # Law 3: regular-polygon shape indices ordered hexagon < pentagon < square < triangle
    assert SHAPE_INDEX[6] < SHAPE_INDEX[5] < SHAPE_INDEX[4] < SHAPE_INDEX[3]
    assert abs(SHAPE_INDEX[6] ** 4 - 192.0) < 1e-9 and abs(SHAPE_INDEX[4] ** 4 - 256.0) < 1e-9 and abs(SHAPE_INDEX[3] ** 4 - 432.0) < 1e-6
    # disorder monotonicity: p6 falls, mu2 and excess perimeter rise
    stats_ = [polygon_statistics(*(lambda v: (v["sides"], v["areas"], v["perims"], v["neighbours"]))(froth(d, 0, nx=14, ny=14)))
              for d in (0.0, 0.15, 0.4, 1.0)]
    assert all(a["p6"] >= b["p6"] for a, b in zip(stats_, stats_[1:]))
    assert all(a["excess_perimeter"] <= b["excess_perimeter"] + 1e-9 for a, b in zip(stats_, stats_[1:]))
    assert stats_[-1]["mu2"] > 1.0 and stats_[-1]["p6"] < 0.45   # ~Poisson-Voronoi limit
    # Lloyd relaxation lowers excess perimeter
    v0 = froth(0.6, 0, nx=12, ny=12); v1 = froth(0.6, 8, nx=12, ny=12)
    s0 = polygon_statistics(v0["sides"], v0["areas"], v0["perims"], v0["neighbours"])
    s1 = polygon_statistics(v1["sides"], v1["areas"], v1["perims"], v1["neighbours"])
    assert s1["excess_perimeter"] < s0["excess_perimeter"] and s1["p6"] >= s0["p6"]
    # Law 5 sum rule holds exactly on the torus
    assert abs(s0["aboav_weaire"]["sum_rule_residual"]) < 1e-9
    # Law 6: division chain is stochastic, mean six, p6 near Gibson's 0.46
    g = gibson_markov_fixed_point()
    assert g["transition_matrix_rows_sum_to_one"] and abs(g["mean_n"] - 6.0) < 0.05, g["mean_n"]
    assert 0.35 < g["p6"] < 0.55, g["p6"]
    # determinism
    a = froth(0.35, 0, nx=8, ny=8)["sides"]; b = froth(0.35, 0, nx=8, ny=8)["sides"]
    assert np.array_equal(a, b)
    print(f"self-test OK ({time.time() - t0:.1f}s): torus <n>=6 exact; honeycomb mu2=0, p6=1, excess=0; "
          f"P^4 = 192/256/432; disorder monotone; Lloyd relaxes; AW sum rule exact; Gibson chain p6={g['p6']:.3f}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--delta", type=float, default=0.35, help="canonical froth disorder (lattice spacings)")
    args = ap.parse_args(argv)
    if args.self_test:
        return self_test()
    m = run(args.delta)
    md = m["model"]
    print(f"[{SIM_ID}] froth delta={md['delta']}: <n>={md['mean_n']:.3f} mu2={md['mu2']:.3f} p6={md['p6']:.3f} "
          f"excess perimeter={100 * md['excess_perimeter']:.2f}%  Lewis lambda={md['lewis']['lambda']:.3f}  "
          f"Aboav a={md['aboav_weaire']['a']:.2f}")
    print(f"[{SIM_ID}] division chain p6={m['gibson_division_chain']['p6']:.3f} (Gibson 2006: {GIBSON_2006_MARKOV[6]})")
    print(f"[{SIM_ID}] wrote {m['outputs']['comparison_png']}, {m['outputs']['growth_gif']} in {m['runtime_s']} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())

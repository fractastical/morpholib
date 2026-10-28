#!/usr/bin/env python3
"""
t1917-019_the-forms-of-tissues-or-cell-aggregates.py — Thompson (1917) claim T1917-019

    "Tissue cells tessellate approximately as hexagons / as a froth, the configuration
     that partitions area with least perimeter."
    D'Arcy W. Thompson, On Growth and Form (1917), Ch. VII "The Forms of Tissues, or
    Cell-Aggregates", pp. 293-345, Figs. 120-127 (IA scan ongrowthform1917thom).

The claim is decomposed into three testable parts (see
datasets/thompson_1917/t1917-019/t1917_019_the_forms_of_tissues_or_cell_a_reference.json):

  C1  froth / mean six       — trivalent network, <n> = 6 - 6 chi / F         (E1, Thompson p. 325)
  C2  least perimeter        — honeycomb bound  P_6 = 2 (12)^(1/4) sqrt(A)    (E3, Thompson p. 330; Hales 2001)
  C3  approximately hexagons — p6 is modal and >= 0.5 (strict reading)       (Thompson p. 321)

Pass 2 (v2) — what the script does
  1. Froth morphospace: periodic (torus) Voronoi tessellation of a hexagonal seed lattice
     with positional disorder `delta` and optional Lloyd (centroidal-Voronoi) relaxation
     — "a froth relaxing toward least perimeter".  Exact polygon statistics: p_n, <n>,
     mu2, shape index P/sqrt(A), excess perimeter over the isoperimetric honeycomb
     bound (E2, E3), Lewis's law (E4), Aboav-Weaire (E5) and its sum rule.
  2. Division topology: mean-field reconstruction of the Gibson et al. (2006) Markov
     chain (E6) with through-centre cleavage; uniform-random cleavage as an ablation.
  3. Specimen computer vision: the vendored human corneal endothelium in situ
     (alizarin-red cell outlines, Zhiguo He, CC BY-SA 4.0) is segmented with a
     nucleus-marker watershed; every interior cell's neighbour count (= polygon class),
     area, Crofton perimeter and junction valence are measured.  Thompson's own Fig. 125
     soap froth (p. 322, panels A and B) is digitised through the same pipeline.
  4. Parameter-regime sweep delta x Lloyd; best-fit delta to the specimen polygon
     distribution (total-variation distance).
  5. Controls: honeycomb positive control, Poisson-Voronoi null, rasterised-froth
     pipeline control (pixel perimeter bias calibration), Aboav-Weaire neighbour-shuffle
     control, uniform-random cleavage ablation.
  6. Statistics: Pearson r and RMSE between specimen and model / Gibson / honeycomb
     p_n, total variation and KS-type CDF distance, chi-square tests, bootstrap 95 % CI
     on p6 and <n>, CV of cell area vs the clinical norm (Sheng & Bullimore 2007).
  7. Verdict -> datasets/thompson_1917/t1917-019/result.json; figures ->
     docs/media/t1917_019_the_forms_of_tissues_or_cell_a_{comparison.png,growth.gif};
     measure_morphoeval() for morphoeval/measure_outputs.py.

Usage
  python3 thompson/claims/t1917-019_the-forms-of-tissues-or-cell-aggregates.py --self-test
  python3 thompson/claims/t1917-019_the-forms-of-tissues-or-cell-aggregates.py [--delta 0.35] [--lloyd 0] [--nx 20]

Dependencies: numpy, scipy, matplotlib, Pillow, scikit-image (requirements.txt).
Deterministic (seed 20261028).
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
from scipy import ndimage as ndi
from scipy import stats as sstats
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
HEX_SHAPE_INDEX = 2.0 * 12.0 ** 0.25              # 3.7224 = 2 x Hales honeycomb constant 12^(1/4)
SHAPE_INDEX = {n: 2.0 * math.sqrt(n * math.tan(math.pi / n)) for n in range(3, 13)}
CLASSES = list(range(3, 13))
FIT_CLASSES = [4, 5, 6, 7, 8]

# Literature reference values (reference.json -> reference_values)
GIBSON_2006_MARKOV = {4: 0.029, 5: 0.289, 6: 0.464, 7: 0.208, 8: 0.010}
CORNEA_NORM_P6 = 0.616          # Sheng & Bullimore 2007 Cornea 26:520 (age 40, no CL, Rx 0)
CORNEA_NORM_P6_RANGE = (0.55, 0.70)
CORNEA_NORM_CV_AREA = 0.315     # idem, CoV 31.5 %
POISSON_VORONOI_P6 = 0.2947     # Weaire & Rivier 1984
POISSON_VORONOI_MU2 = 1.78

SPECIMEN = DATA / "human_corneal_endothelium_specimen.jpg"
ELODEA = DATA / "elodea_canadensis_leaf_specimen.jpg"
THOMPSON_P322 = DATA / "thompson_1917_p322_fig125_126_froth_elodea.jpg"
# Fig. 125 panels on the vendored p. 322 scan (1904 x 3128 px), cropped inside the two
# horizontal "glass plate" rules located by row ink density (rules at y ~ 1090/1312 and
# ~ 1448/1670); x-range covers the froth blob (x ~ 600-1360) with margin.
FIG125_A_BOX = (560, 1096, 1400, 1306)
FIG125_B_BOX = (560, 1456, 1400, 1662)

# verdict thresholds (reference.json -> claim_decomposition -> pass_criterion)
C1_MEAN_TOL = 0.15
C1_TRIVALENT_MIN = 0.90
C2_EXCESS_MAX = 0.10
C3_P6_STRICT = 0.50
C3_P6_FROTH_RANGE = (0.40, 0.70)


# --------------------------------------------------------------------------- #
# 1. Periodic Voronoi froth morphospace (E1-E5 computed exactly on polygons)
# --------------------------------------------------------------------------- #
def hex_lattice(nx: int, ny: int) -> tuple[np.ndarray, float, float]:
    """Regular hexagonal (triangular) seed lattice on a torus; ny must be even."""
    if ny % 2:
        raise ValueError("ny must be even for a periodic hexagonal lattice")
    a = 1.0
    h = math.sqrt(3.0) / 2.0 * a
    pts = [((i + 0.5 * (j % 2)) * a, j * h) for j in range(ny) for i in range(nx)]
    return np.array(pts), nx * a, ny * h


def periodic_voronoi(points: np.ndarray, Lx: float, Ly: float) -> dict:
    """Voronoi tessellation of a periodic box via 3x3 tiling.

    Returns per-cell neighbour lists (indices mod N), polygon vertices, areas,
    perimeters and side counts for the N central cells (E1: every vertex of a generic
    Voronoi diagram is trivalent, so sum sides = 6 F exactly on the torus)."""
    N = len(points)
    shifts = [(dx, dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1)]
    tiled = np.concatenate([points + np.array([dx * Lx, dy * Ly]) for dx, dy in shifts])
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
    """One centroidal-Voronoi (Lloyd) relaxation step on the torus — a proxy for the
    surface-tension relaxation of a froth toward the least-perimeter configuration."""
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
          seed: int = SEED, poisson: bool = False) -> dict:
    """Froth morphospace point: hexagonal lattice jittered by Gaussian displacement of
    s.d. `delta` lattice spacings (delta = 0 honeycomb), then `lloyd_iters` relaxation
    steps.  `poisson=True` replaces the lattice by uniformly random seeds (the
    Poisson-Voronoi null tessellation, p6 ~ 0.29, mu2 ~ 1.78)."""
    rng = np.random.default_rng(seed)
    pts, Lx, Ly = hex_lattice(nx, ny)
    if poisson:
        pts = rng.uniform((0.0, 0.0), (Lx, Ly), size=pts.shape)
    else:
        pts = (pts + rng.normal(0.0, delta, size=pts.shape)) % (Lx, Ly)
    for _ in range(lloyd_iters):
        pts = lloyd_step(pts, Lx, Ly)
    vor = periodic_voronoi(pts, Lx, Ly)
    vor.update({"points": pts, "delta": (None if poisson else delta), "lloyd_iters": lloyd_iters,
                "seed": seed, "poisson": poisson})
    return vor


# --------------------------------------------------------------------------- #
# 2. Polygon statistics (E1 mean six, E2 mu2, E3 excess perimeter, E4 Lewis, E5 Aboav-Weaire)
# --------------------------------------------------------------------------- #
def _linfit(x, y) -> dict:
    x = np.asarray(x, float); y = np.asarray(y, float)
    if len(x) < 2:
        return {"slope": None, "intercept": None, "r": None}
    slope, intercept = np.polyfit(x, y, 1)
    r = float(np.corrcoef(x, y)[0, 1]) if len(x) > 2 else 1.0
    return {"slope": float(slope), "intercept": float(intercept), "r": r}


def polygon_statistics(sides: np.ndarray, areas: np.ndarray, perims: np.ndarray,
                       neighbours: list[list[int]], keep: np.ndarray | None = None,
                       shuffle_seed: int | None = None) -> dict:
    """Topological and isoperimetric statistics of a cell tessellation.

    `keep` selects the cells that enter the statistics (interior cells); neighbour
    side numbers for Aboav-Weaire are taken from the full tessellation.
    `shuffle_seed` permutes the side numbers among cells before computing m(n) — the
    neighbour-shuffle control that destroys the Aboav-Weaire correlation."""
    sides = np.asarray(sides); areas = np.asarray(areas, float); perims = np.asarray(perims, float)
    if keep is None:
        keep = np.ones(len(sides), bool)
    idx = np.flatnonzero(keep)
    n = sides[idx]
    F = len(idx)
    mean_n = float(n.mean())
    mu2 = float(np.mean((n - 6.0) ** 2))
    p_n = {k: float(np.mean(n == k)) for k in CLASSES}
    counts = {k: int(np.sum(n == k)) for k in CLASSES}
    # E3 — excess perimeter over the honeycomb bound for the SAME cell areas
    A = areas[idx]; P = perims[idx]
    p6_bound = HEX_SHAPE_INDEX * np.sqrt(A)
    excess_perimeter = float(P.sum() / p6_bound.sum() - 1.0)
    shape_index = P / np.sqrt(A)
    # E4 — Lewis's law: <A_n>/<A> = 1 + lambda (n - 6)
    lewis_x, lewis_y, lewis_counts = [], [], []
    for k in CLASSES:
        sel = n == k
        if sel.sum() >= 3:
            lewis_x.append(k); lewis_y.append(float(A[sel].mean() / A.mean())); lewis_counts.append(int(sel.sum()))
    lewis = _linfit(lewis_x, lewis_y)
    # E5 — Aboav-Weaire: n m(n) = (6 - a) n + (6a + mu2)
    side_lookup = sides.copy()
    if shuffle_seed is not None:
        side_lookup = np.random.default_rng(shuffle_seed).permutation(side_lookup)
    m_i = np.array([np.mean(side_lookup[neighbours[i]]) if len(neighbours[i]) else np.nan for i in idx])
    aw_x, aw_y = [], []
    for k in CLASSES:
        sel = (n == k) & np.isfinite(m_i)
        if sel.sum() >= 3:
            aw_x.append(k); aw_y.append(float(k * m_i[sel].mean()))
    aw = _linfit(aw_x, aw_y)
    a_aboav = 6.0 - aw["slope"] if aw["slope"] is not None else None
    sum_rule_lhs = float(np.nanmean(n * m_i))         # (1/F) sum n_i m(n_i)
    sum_rule_rhs = 36.0 + mu2                          # Lean Law 5 (exact when <n> = 6)
    return {
        "F": int(F), "mean_n": mean_n, "mu2": mu2, "p_n": p_n, "counts": counts, "p6": p_n[6],
        "modal_class": int(max(p_n, key=p_n.get)),
        "excess_perimeter": excess_perimeter,
        "shape_index_mean": float(shape_index.mean()),
        "shape_index_median": float(np.median(shape_index)),
        "area_cv": float(A.std() / A.mean()),
        "lewis": {"lambda": lewis["slope"], "intercept": lewis["intercept"], "r": lewis["r"],
                  "classes": lewis_x, "rel_area": lewis_y, "class_counts": lewis_counts},
        "aboav_weaire": {"a": a_aboav, "slope": aw["slope"], "intercept": aw["intercept"], "r": aw["r"],
                         "classes": aw_x, "n_m_n": aw_y,
                         "sum_rule_lhs": sum_rule_lhs, "sum_rule_rhs": sum_rule_rhs,
                         "sum_rule_residual": sum_rule_lhs - sum_rule_rhs},
    }


def stats_of(vor: dict, **kw) -> dict:
    return polygon_statistics(vor["sides"], vor["areas"], vor["perims"], vor["neighbours"], **kw)


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
# 4. Specimen computer vision: label image -> tessellation
# --------------------------------------------------------------------------- #
def tessellation_from_labels(lab: np.ndarray, background_area_factor: float | None = 2.5,
                             min_contact_px: int = 4) -> dict:
    """Turn a watershed label image into the same tessellation record as periodic_voronoi():
    neighbour lists (4-connected label adjacency with at least `min_contact_px` shared
    boundary pixel pairs, so that cells merely touching at a vertex are not neighbours),
    pixel areas, Crofton perimeters, side counts, and an interior mask (cells that do not
    touch the image border and are not adjacent to an oversized border-touching
    'background' region).  Junction valence is read from 2x2 pixel windows: 3 distinct
    labels = trivalent, >= 4 = four-fold."""
    from collections import Counter
    from skimage.measure import regionprops
    n = int(lab.max())
    contact: Counter = Counter()
    for a, c in ((lab[:, :-1], lab[:, 1:]), (lab[:-1, :], lab[1:, :])):
        m = a != c
        lo = np.minimum(a[m], c[m]); hi = np.maximum(a[m], c[m])
        contact.update(zip(lo.tolist(), hi.tolist()))
    neighbours = [set() for _ in range(n + 1)]
    for (p, q), cnt in contact.items():
        if p > 0 and q > 0 and cnt >= min_contact_px:
            neighbours[p].add(q); neighbours[q].add(p)
    props = {p.label: p for p in regionprops(lab)}
    areas = np.array([props[i].area if i in props else 0.0 for i in range(1, n + 1)], float)
    perims = np.array([props[i].perimeter_crofton if i in props else 0.0 for i in range(1, n + 1)], float)
    centroids = np.array([props[i].centroid if i in props else (np.nan, np.nan) for i in range(1, n + 1)])
    border = set(np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]])).tolist()) - {0}
    med = float(np.median(areas[areas > 0]))
    background = {i for i in border if background_area_factor and areas[i - 1] > background_area_factor * med}
    # border-touching fragments (cells cut by the image edge to < half a cell) carry
    # unreliable adjacency, so cells touching them are not scored either
    fragments = {i for i in border if areas[i - 1] < 0.5 * med}
    interior = np.array([(i not in border) and not (neighbours[i] & (background | fragments)) and areas[i - 1] > 0
                         for i in range(1, n + 1)])
    # junction valence
    w = np.stack([lab[:-1, :-1], lab[:-1, 1:], lab[1:, :-1], lab[1:, 1:]], -1)
    s = np.sort(w, -1)
    distinct = 1 + np.sum(s[..., 1:] != s[..., :-1], -1)
    n3 = int(np.sum(distinct == 3)); n4 = int(np.sum(distinct >= 4))
    nb0 = [sorted(i - 1 for i in neighbours[k]) for k in range(1, n + 1)]      # 0-based like the model
    sides = np.array([len(x) for x in nb0])
    return {"neighbours": nb0, "areas": areas, "perims": perims, "sides": sides, "interior": interior,
            "centroids": centroids, "n_cells": n, "n_interior": int(interior.sum()),
            "n_border": len(border), "n_background": len(background), "median_area_px": med,
            "junctions_trivalent": n3, "junctions_fourfold": n4,
            "trivalent_fraction": (n3 / (n3 + n4)) if (n3 + n4) else None, "labels": lab}


def segment_tissue(img_rgb: np.ndarray, nucleus_blueness: float = 40.0, nucleus_min_px: int = 80,
                   sigma: float = 2.0, border_level: float = 110.0) -> dict:
    """Nucleus-marker watershed for alizarin-red / nuclear-counterstained epithelia.

    Cell borders are dark in every channel; nuclei are blue (B >> R, G); cytoplasm is
    mid-grey.  Markers = connected components of the blueness map (one per cell);
    elevation = -min(R, G, B) smoothed (borders are ridges) with the nuclei filled to
    cytoplasm level so they are not mistaken for borders; cells lacking a detected
    nucleus get a marker from the distance-transform peaks of the cytoplasm mask."""
    from skimage.feature import peak_local_max
    from skimage.segmentation import watershed
    im = img_rgb.astype(float)
    R, G, B = im[..., 0], im[..., 1], im[..., 2]
    blue = ndi.gaussian_filter(B - 0.5 * (R + G), sigma)
    nuc = ndi.binary_opening(blue > nucleus_blueness, iterations=2)
    nl, nn = ndi.label(nuc)
    sizes = ndi.sum(nuc, nl, range(1, nn + 1))
    markers = np.zeros(B.shape, int)
    k = 0
    for i, sz in enumerate(sizes, 1):
        if sz > nucleus_min_px:
            k += 1; markers[nl == i] = k
    gray = ndi.gaussian_filter(np.minimum(np.minimum(R, G), B), sigma)
    gray[ndi.binary_dilation(nuc, iterations=6)] = np.percentile(gray, 70)
    mask = gray > border_level
    dist = ndi.distance_transform_edt(mask)
    dnuc = ndi.distance_transform_edt(markers == 0)
    extra = 0
    for y, x in peak_local_max(dist, min_distance=30, threshold_abs=14):
        if dnuc[y, x] > 45:
            k += 1; markers[y, x] = k; extra += 1
    lab = watershed(-gray, markers)
    out = tessellation_from_labels(lab, background_area_factor=None)
    out.update({"n_nucleus_markers": k - extra, "n_extra_markers": extra, "method": "nucleus-marker watershed"})
    return out


def segment_line_drawing(gray: np.ndarray, sigma: float = 1.0, min_distance: int = 9,
                         threshold_abs: float = 4.0) -> dict:
    """Watershed segmentation of a line drawing / rasterised froth (dark walls on a light
    ground): Otsu wall mask -> distance-transform peaks as markers -> watershed on the
    smoothed intensity.  Used for Thompson's Fig. 125 and for the pipeline control."""
    from skimage.feature import peak_local_max
    from skimage.filters import threshold_otsu
    from skimage.segmentation import watershed
    g = ndi.gaussian_filter(gray.astype(float), sigma)
    mask = g > threshold_otsu(g)
    dist = ndi.distance_transform_edt(mask)
    lab_mask, _ = ndi.label(mask)
    peaks = peak_local_max(dist, min_distance=min_distance, threshold_abs=threshold_abs, labels=lab_mask)
    markers = np.zeros(g.shape, int)
    for i, (y, x) in enumerate(peaks, 1):
        markers[y, x] = i
    lab = watershed(-g, markers)
    out = tessellation_from_labels(lab, background_area_factor=2.5)
    out["method"] = "distance-peak watershed (line drawing)"
    return out


def rasterize_froth(vor: dict, px: int = 1000, linewidth_pt: float = 2.0) -> np.ndarray:
    """Render the central copy of a periodic froth as a dark-wall line drawing (grayscale
    uint8) — the pipeline control: exact polygon statistics are known for this image."""
    fig = plt.figure(figsize=(px / 100, px / 100 * vor["Ly"] / vor["Lx"]), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.add_collection(PolyCollection(vor["polys"], facecolors="white", edgecolors="black", linewidths=linewidth_pt))
    ax.set_xlim(0, vor["Lx"]); ax.set_ylim(0, vor["Ly"]); ax.axis("off")
    fig.canvas.draw()
    arr = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    return np.asarray(Image.fromarray(arr).convert("L"))


# --------------------------------------------------------------------------- #
# 5. Statistics: goodness of fit, bootstrap, sweep and fit
# --------------------------------------------------------------------------- #
def distribution_distance(p: dict, q: dict, classes=FIT_CLASSES) -> dict:
    """Pearson r, RMSE, total variation and KS-type max-CDF distance between two
    polygon-class distributions restricted to `classes`."""
    pv = np.array([p.get(k, 0.0) for k in classes]); qv = np.array([q.get(k, 0.0) for k in classes])
    r = float(np.corrcoef(pv, qv)[0, 1]) if pv.std() > 0 and qv.std() > 0 else None
    return {"r": r, "rmse": float(np.sqrt(np.mean((pv - qv) ** 2))),
            "tv": float(0.5 * np.abs(pv - qv).sum()),
            "ks_D": float(np.max(np.abs(np.cumsum(pv) - np.cumsum(qv))))}


def chi_square_vs(counts: dict, q: dict, classes=FIT_CLASSES) -> dict:
    """Chi-square test of observed class counts against expected fractions q (tails
    merged into n <= 4 and n >= 8; expected probabilities renormalised; classes with
    zero expectation dropped)."""
    obs = np.array([sum(v for k, v in counts.items() if k <= 4), counts.get(5, 0), counts.get(6, 0), counts.get(7, 0),
                    sum(v for k, v in counts.items() if k >= 8)], float)
    exp_p = np.array([sum(v for k, v in q.items() if k <= 4), q.get(5, 0.0), q.get(6, 0.0), q.get(7, 0.0),
                      sum(v for k, v in q.items() if k >= 8)], float)
    keep = exp_p > 0
    obs = obs[keep]; exp_p = exp_p[keep] / exp_p[keep].sum()
    N = obs.sum()
    if len(obs) < 2 or N == 0:
        return {"chi2": None, "dof": None, "p_value": None}
    chi2, p = sstats.chisquare(obs, exp_p * N)
    return {"chi2": float(chi2), "dof": int(len(obs) - 1), "p_value": float(p)}


def bootstrap_ci(values: np.ndarray, fn, n_boot: int = 2000, seed: int = SEED) -> dict:
    rng = np.random.default_rng(seed)
    values = np.asarray(values)
    boots = np.array([fn(values[rng.integers(0, len(values), len(values))]) for _ in range(n_boot)])
    return {"point": float(fn(values)), "ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
            "se": float(boots.std()), "n": int(len(values)), "n_boot": n_boot}


def parameter_sweep(deltas=(0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0), lloyds=(0, 2, 5, 10, 20),
                    nx: int = 20, ny: int = 20, seeds=(SEED,)) -> list[dict]:
    rows = []
    for d in deltas:
        for L in lloyds:
            for s in seeds:
                st = stats_of(froth(d, L, nx, ny, seed=s))
                rows.append({"delta": d, "lloyd": L, "seed": s, "p6": st["p6"], "mu2": st["mu2"],
                             "excess_perimeter": st["excess_perimeter"], "mean_n": st["mean_n"],
                             "area_cv": st["area_cv"], "p_n": {k: st["p_n"][k] for k in FIT_CLASSES}})
    return rows


def fit_to_specimen(spec: dict, sweep: list[dict]) -> dict:
    """Best froth morphospace points for the specimen:
      best            — min total-variation distance between polygon distributions (topology only)
      best_unrelaxed  — the same restricted to Lloyd = 0
      best_joint      — min of TV + |area CV difference| + |excess-perimeter difference|
                        (topology AND geometry); a large gap between `best` and `best_joint`
                        means the froth cannot match both at once."""
    spec_p_n = spec["p_n"]

    def tv(r):
        return distribution_distance(spec_p_n, r["p_n"])["tv"]

    def joint(r):
        return tv(r) + abs(r["area_cv"] - spec["area_cv"]) + abs(r["excess_perimeter"] - spec["excess_perimeter_corrected"])

    best = min(sweep, key=tv)
    best0 = min((r for r in sweep if r["lloyd"] == 0), key=tv)
    bestj = min(sweep, key=joint)
    return {"best": {**best, **distribution_distance(spec_p_n, best["p_n"])},
            "best_unrelaxed": {**best0, **distribution_distance(spec_p_n, best0["p_n"])},
            "best_joint": {**bestj, **distribution_distance(spec_p_n, bestj["p_n"]), "joint_objective": joint(bestj),
                           "area_cv_diff": bestj["area_cv"] - spec["area_cv"],
                           "excess_perimeter_diff": bestj["excess_perimeter"] - spec["excess_perimeter_corrected"]}}


# --------------------------------------------------------------------------- #
# 6. Rendering
# --------------------------------------------------------------------------- #
POLY_COLORS = {3: "#7b3294", 4: "#c2a5cf", 5: "#f4a582", 6: "#f7f7f7", 7: "#92c5de", 8: "#0571b0", 9: "#023858"}


def draw_froth(ax, vor: dict, title: str = "", keep: np.ndarray | None = None) -> None:
    cols = [POLY_COLORS.get(int(min(max(s, 3), 9)), "#333333") for s in vor["sides"]]
    if keep is not None:
        cols = [c if k else "#dddddd" for c, k in zip(cols, keep)]
    ax.add_collection(PolyCollection(vor["polys"], facecolors=cols, edgecolors="k", linewidths=0.6))
    ax.set_xlim(0, vor["Lx"]); ax.set_ylim(0, vor["Ly"]); ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=8.5)


def polygon_legend(ax, loc="lower center", anchor=(0.5, -0.1)):
    from matplotlib.patches import Patch
    handles = [Patch(facecolor=POLY_COLORS[k], edgecolor="k", label=f"{k}-gon") for k in (4, 5, 6, 7, 8)]
    ax.legend(handles=handles, loc=loc, ncol=5, fontsize=7, frameon=False, bbox_to_anchor=anchor)


def draw_segmentation(ax, img, tess: dict, title: str, label_numbers: bool = True, alpha: float = 0.45) -> None:
    """Specimen image with interior cells filled by polygon class and side numbers printed."""
    from skimage.segmentation import find_boundaries
    lab = tess["labels"]
    base = np.asarray(img, float) / 255.0
    if base.ndim == 2:
        base = np.stack([base] * 3, -1)
    overlay = base.copy()
    cmap = np.zeros((lab.max() + 1, 3))
    for i in range(1, lab.max() + 1):
        if tess["interior"][i - 1]:
            cmap[i] = matplotlib.colors.to_rgb(POLY_COLORS.get(int(min(max(tess["sides"][i - 1], 3), 9)), "#333333"))
        else:
            cmap[i] = (0.85, 0.85, 0.85)
    fill = cmap[lab]
    overlay = (1 - alpha) * base + alpha * fill
    overlay[find_boundaries(lab, mode="inner")] = (0.85, 0.0, 0.0)
    ax.imshow(np.clip(overlay, 0, 1))
    if label_numbers:
        for i in range(1, lab.max() + 1):
            if tess["interior"][i - 1]:
                y, x = tess["centroids"][i - 1]
                ax.text(x, y, str(tess["sides"][i - 1]), color="k", fontsize=6, ha="center", va="center")
    ax.set_title(title, fontsize=8.5); ax.set_xticks([]); ax.set_yticks([])


def render_comparison(m: dict, model: dict, spec_tess: dict, spec_img, fig125_tess: dict, fig125_img: dict,
                      out_png: Path) -> None:
    sp = m["specimen"]; ms = m["model_fit"]["stats"]; sw = m["sweep"]; gb = m["gibson_division_chain"]
    fig = plt.figure(figsize=(16, 13))
    gs = fig.add_gridspec(3, 3, height_ratios=[1.25, 1.0, 1.0])
    # (a) segmented specimen
    ax = fig.add_subplot(gs[0, 0])
    draw_segmentation(ax, spec_img, spec_tess,
                      f"Human corneal endothelium in situ (Zhiguo He, CC BY-SA 4.0)\n"
                      f"{sp['n_interior']} interior cells: <n> = {sp['mean_n']:.2f}, p6 = {sp['p6']:.2f}, "
                      f"mu2 = {sp['mu2']:.2f}, trivalent {100 * sp['trivalent_fraction']:.0f} %")
    # (b) Thompson Fig. 125 A and B digitised
    sub = gs[0, 1].subgridspec(2, 1, hspace=0.35)
    fa = m["thompson_fig125"]["A"]; fb = m["thompson_fig125"]["B"]
    for row, (name, st) in enumerate((("A", fa), ("B", fb))):
        ax = fig.add_subplot(sub[row, 0])
        draw_segmentation(ax, fig125_img[name], fig125_tess[name],
                          (f"Thompson (1917) Fig. 125 {name}, p. 322 — soap froth (after Rhumbler), digitised\n" if row == 0 else "")
                          + f"{name}: {st['n_interior']} interior cells, <n> = {st['mean_n']:.2f}, p6 = {st['p6']:.2f}, mu2 = {st['mu2']:.2f}",
                          label_numbers=False)
    # (c) best-fit model froth
    ax = fig.add_subplot(gs[0, 2])
    bf = m["model_fit"]["best"]
    draw_froth(ax, model, f"Best-fit periodic Voronoi froth: delta = {bf['delta']:.2f}, Lloyd = {bf['lloyd']}\n"
                          f"<n> = {ms['mean_n']:.2f}, p6 = {ms['p6']:.2f}, mu2 = {ms['mu2']:.2f}, "
                          f"excess perimeter = {100 * ms['excess_perimeter']:.1f} %  (TV to specimen {bf['tv']:.3f})")
    polygon_legend(ax)
    # (d) polygon-class distributions
    ax = fig.add_subplot(gs[1, 0])
    ks = FIT_CLASSES; w = 0.2
    series = [("specimen (cornea)", {k: sp["p_n"][k] for k in ks}, "#b2182b"),
              (f"froth model delta={bf['delta']:.2f}", {k: ms["p_n"][k] for k in ks}, "#92c5de"),
              ("Gibson 2006 wing disc (Markov)", GIBSON_2006_MARKOV, "#f4a582"),
              ("Thompson Fig. 125 A (digitised)", {k: fa["p_n"][k] for k in ks}, "#bdbdbd")]
    for j, (lab_, d, col) in enumerate(series):
        ax.bar([k + (j - 1.5) * w for k in ks], [d.get(k, 0.0) for k in ks], w, color=col, edgecolor="k", label=lab_)
    ax.plot([6 - 2 * w, 6 + 2 * w], [1.0, 1.0], "k-", lw=2, label="honeycomb p6 = 1 (strict reading)")
    ax.plot(ks, [gb["distribution"].get(k, 0.0) for k in ks], "k.--", lw=1, label=f"division chain (reconstr.) p6={gb['p6']:.2f}")
    ax.axhspan(*CORNEA_NORM_P6_RANGE, color="#b2182b", alpha=0.08, lw=0)
    ax.text(8.5, CORNEA_NORM_P6_RANGE[1] + 0.01, "cornea norm p6 0.55-0.70", color="#b2182b", fontsize=6.5, ha="right")
    ax.set_xlabel("polygon class n"); ax.set_ylabel("fraction p_n"); ax.set_ylim(0, 1.05)
    ax.legend(fontsize=6.5, frameon=False, loc="upper left", bbox_to_anchor=(0.0, 0.93))
    ax.set_title("Polygon-class distributions (E1/E2)", fontsize=8.5)
    # (e) sweep: p6 and excess perimeter vs delta
    ax = fig.add_subplot(gs[1, 1])
    for L, col in zip((0, 2, 5, 20), ("#0571b0", "#5ab4ac", "#8c510a", "#762a83")):
        rows = sorted([r for r in sw if r["lloyd"] == L], key=lambda r: r["delta"])
        ax.plot([r["delta"] for r in rows], [r["p6"] for r in rows], "-o", ms=3, color=col, label=f"p6, Lloyd={L}")
    ax.axhline(sp["p6"], color="#b2182b", ls="--", lw=1.2, label=f"specimen p6 = {sp['p6']:.2f}")
    ax.axhline(POISSON_VORONOI_P6, color="k", ls=":", lw=1, label="Poisson-Voronoi null 0.29")
    ax.set_xlabel("positional disorder delta (lattice spacings)"); ax.set_ylabel("hexagon fraction p6"); ax.set_ylim(0, 1.05)
    ax.legend(fontsize=6.5, frameon=False); ax.set_title("Parameter-regime sweep: p6(delta, Lloyd)", fontsize=8.5)
    ax2 = ax.twinx()
    rows = sorted([r for r in sw if r["lloyd"] == 0], key=lambda r: r["delta"])
    ax2.plot([r["delta"] for r in rows], [100 * r["excess_perimeter"] for r in rows], "s--", ms=3, color="#d6604d", alpha=0.7)
    ax2.axhline(100 * sp["excess_perimeter_corrected"], color="#d6604d", ls="-.", lw=1)
    ax2.set_ylabel("excess perimeter % (Lloyd=0, dashed; specimen dash-dot)", color="#d6604d", fontsize=7)
    # (f) mu2 vs p6 morphospace with all tessellations
    ax = fig.add_subplot(gs[1, 2])
    ax.scatter([r["p6"] for r in sw], [r["mu2"] for r in sw], c=[r["delta"] for r in sw], cmap="viridis", s=14, label="froth sweep (colour = delta)")
    pts = [("cornea", sp["p6"], sp["mu2"], "#b2182b", "o"), ("Fig.125 A", fa["p6"], fa["mu2"], "#636363", "s"),
           ("Fig.125 B", m["thompson_fig125"]["B"]["p6"], m["thompson_fig125"]["B"]["mu2"], "#969696", "D"),
           ("Gibson 2006", GIBSON_2006_MARKOV[6], sum((k - 6) ** 2 * v for k, v in GIBSON_2006_MARKOV.items()), "#f4a582", "^"),
           ("Poisson null", m["controls"]["poisson_voronoi"]["p6"], m["controls"]["poisson_voronoi"]["mu2"], "k", "X"),
           ("honeycomb", 1.0, 0.0, "k", "*")]
    for lab_, x, y, col, mk in pts:
        ax.scatter([x], [y], color=col, marker=mk, s=70, edgecolor="k", zorder=5)
        ax.annotate(lab_, (x, y), textcoords="offset points", xytext=(5, 4), fontsize=7)
    ax.set_xlabel("p6"); ax.set_ylabel("mu2"); ax.set_title("Froth morphospace (p6, mu2): tissue vs froth vs honeycomb", fontsize=8.5)
    ax.legend(fontsize=6.5, frameon=False, loc="upper right")
    # (g) Lewis's law on specimen + model
    ax = fig.add_subplot(gs[2, 0])
    lw_s = sp["lewis"]; lw_m = ms["lewis"]
    ax.plot(lw_s["classes"], lw_s["rel_area"], "o", color="#b2182b", label="specimen <A_n>/<A>")
    ax.plot(lw_m["classes"], lw_m["rel_area"], "s", color="#0571b0", mfc="none", label="froth model")
    xs = np.array([4, 8]); ax.plot(xs, 1 + 0.25 * (xs - 6), "k--", lw=1, label="Lewis 1928: 1 + (n-6)/4")
    if lw_s["lambda"] is not None:
        ax.plot(xs, lw_s["intercept"] + lw_s["lambda"] * xs, "-", color="#b2182b", lw=1,
                label=f"specimen fit lambda = {lw_s['lambda']:.3f}, r = {lw_s['r']:.2f}")
    ax.set_xlabel("n"); ax.set_ylabel("relative mean area"); ax.legend(fontsize=6.5, frameon=False)
    ax.set_title("Lewis's law (E4)", fontsize=8.5)
    # (h) Aboav-Weaire on specimen + shuffle control
    ax = fig.add_subplot(gs[2, 1])
    aw_s = sp["aboav_weaire"]; aw_c = m["controls"]["aboav_shuffle_specimen"]
    ax.plot(aw_s["classes"], aw_s["n_m_n"], "o", color="#b2182b", label="specimen n m(n)")
    if aw_s["slope"] is not None:
        ax.plot(xs, aw_s["intercept"] + aw_s["slope"] * xs, "-", color="#b2182b", lw=1, label=f"fit a = {aw_s['a']:.2f}, r = {aw_s['r']:.2f}")
    ax.plot(aw_c["classes"], aw_c["n_m_n"], "x", color="#636363", label=f"neighbour-shuffle control (a = {aw_c['a']:.2f})")
    ax.plot(xs, (6 - 1.2) * xs + 6 * 1.2 + sp["mu2"], "k--", lw=1, label="Aboav-Weaire a = 1.2 (froths)")
    ax.set_xlabel("n"); ax.set_ylabel("n m(n)"); ax.legend(fontsize=6.5, frameon=False)
    ax.set_title(f"Aboav-Weaire (E5); specimen sum-rule residual {aw_s['sum_rule_residual']:+.2f}", fontsize=8.5)
    # (i) verdict text
    ax = fig.add_subplot(gs[2, 2]); ax.axis("off")
    v = m["verdict"]
    lines = [f"VERDICT: {v['verdict'].upper()}", ""]
    for c in ("C1", "C2", "C3"):
        lines.append(f"{c} {'PASS' if v[c]['pass'] else 'FAIL'} — {v[c]['summary']}")
    lines += ["", f"p6 = {sp['p6']:.3f} [95% CI {sp['p6_ci'][0]:.2f}, {sp['p6_ci'][1]:.2f}]  vs cornea norm {CORNEA_NORM_P6}",
              f"<n> = {sp['mean_n']:.3f} [CI {sp['mean_n_ci'][0]:.2f}, {sp['mean_n_ci'][1]:.2f}]",
              f"area CV = {sp['area_cv']:.3f} vs norm {CORNEA_NORM_CV_AREA}",
              f"excess perimeter raw {100 * sp['excess_perimeter']:.1f} % -> bias-corrected {100 * sp['excess_perimeter_corrected']:.1f} %",
              f"fit vs Gibson 2006: r = {m['gof']['vs_gibson']['r']:.2f}, RMSE = {m['gof']['vs_gibson']['rmse']:.3f}, chi2 p = {m['gof']['vs_gibson']['chi2']['p_value']:.2g}",
              f"fit vs best froth: r = {m['gof']['vs_best_froth']['r']:.2f}, RMSE = {m['gof']['vs_best_froth']['rmse']:.3f}, chi2 p = {m['gof']['vs_best_froth']['chi2']['p_value']:.2g}",
              f"vs honeycomb: TV = {m['gof']['vs_honeycomb']['tv']:.2f}  (1 - p6)",
              f"joint topology+geometry fit: delta = {m['model_fit']['best_joint']['delta']:.2f}, Lloyd = {m['model_fit']['best_joint']['lloyd']}, "
              f"TV = {m['model_fit']['best_joint']['tv']:.2f}, dCV = {m['model_fit']['best_joint']['area_cv_diff']:+.2f}, "
              f"dexcess = {100 * m['model_fit']['best_joint']['excess_perimeter_diff']:+.1f} %"]
    ax.text(0.0, 1.0, "\n".join(lines), va="top", ha="left", fontsize=7.6, family="monospace", wrap=True)
    fig.suptitle("T1917-019  Thompson (1917) Ch. VII: tissue cells as hexagons / froth, least-perimeter partition — Pass 2", fontsize=12)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=100)
    plt.close(fig)


def render_relaxation_gif(out_gif: Path, delta: float = 0.6, iters: int = 24, nx: int = 16, ny: int = 16,
                          target_p6: float | None = None) -> list[dict]:
    """Lloyd relaxation of a disordered froth: p6 rises, excess perimeter falls toward the
    honeycomb bound.  The specimen's p6 is drawn as the target line."""
    rng = np.random.default_rng(SEED + 1)
    pts, Lx, Ly = hex_lattice(nx, ny)
    pts = (pts + rng.normal(0.0, delta, size=pts.shape)) % (Lx, Ly)
    frames, trace = [], []
    for it in range(iters + 1):
        vor = periodic_voronoi(pts, Lx, Ly)
        st = stats_of(vor)
        trace.append({"iter": it, "p6": st["p6"], "mu2": st["mu2"], "excess_perimeter": st["excess_perimeter"]})
        fig, axes = plt.subplots(1, 2, figsize=(9, 4.8), gridspec_kw={"width_ratios": [1.1, 1]})
        draw_froth(axes[0], vor, f"Froth relaxation (Lloyd step {it}/{iters})\n"
                                 f"p6 = {st['p6']:.2f}   mu2 = {st['mu2']:.2f}   excess perimeter = {100 * st['excess_perimeter']:.1f} %")
        ax = axes[1]
        ax.plot([t["iter"] for t in trace], [t["p6"] for t in trace], "-o", ms=3, color="#0571b0", label="p6")
        ax.plot([t["iter"] for t in trace], [10 * t["excess_perimeter"] for t in trace], "-s", ms=3, color="#d6604d", label="10 x excess perimeter")
        if target_p6 is not None:
            ax.axhline(target_p6, color="#b2182b", ls="--", lw=1, label=f"corneal endothelium p6 = {target_p6:.2f}")
        ax.set_xlim(0, iters); ax.set_ylim(0, 1.05); ax.set_xlabel("Lloyd step"); ax.legend(fontsize=7, frameon=False)
        ax.set_title("toward the least-perimeter froth", fontsize=9)
        fig.tight_layout()
        fig.canvas.draw()
        frames.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()))
        plt.close(fig)
        pts = lloyd_step(pts, Lx, Ly)
    out_gif.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(out_gif, save_all=True, append_images=frames[1:], duration=160, loop=0, optimize=False)
    return trace


# --------------------------------------------------------------------------- #
# 7. Pipeline, verdict, metrics, measure_morphoeval
# --------------------------------------------------------------------------- #
def sha256(path: Path) -> str | None:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None


def _tess_stats(tess: dict, **kw) -> dict:
    st = polygon_statistics(tess["sides"], tess["areas"], tess["perims"], tess["neighbours"], keep=tess["interior"], **kw)
    st.update({k: tess[k] for k in ("n_cells", "n_interior", "n_border", "n_background", "median_area_px",
                                    "junctions_trivalent", "junctions_fourfold", "trivalent_fraction", "method")})
    return st


def analyse_specimen() -> tuple[dict, dict, np.ndarray]:
    img = np.asarray(Image.open(SPECIMEN).convert("RGB"))
    tess = segment_tissue(img)
    st = _tess_stats(tess)
    n_int = tess["sides"][tess["interior"]]
    st["p6_ci"] = bootstrap_ci(n_int, lambda v: np.mean(v == 6))["ci95"]
    st["mean_n_ci"] = bootstrap_ci(n_int, np.mean)["ci95"]
    st["mu2_ci"] = bootstrap_ci(n_int, lambda v: np.mean((v - 6.0) ** 2))["ci95"]
    st["n_nucleus_markers"] = tess["n_nucleus_markers"]; st["n_extra_markers"] = tess["n_extra_markers"]
    st["image_px"] = [int(img.shape[1]), int(img.shape[0])]
    return st, tess, img


def analyse_fig125() -> tuple[dict, dict, dict]:
    """Digitise Thompson's Fig. 125 A (compressed) and B (relaxed) soap froth panels."""
    page = Image.open(THOMPSON_P322).convert("L")
    out, tess_d, img_d = {}, {}, {}
    for name, box in (("A", FIG125_A_BOX), ("B", FIG125_B_BOX)):
        g = np.asarray(page.crop(box))
        tess = segment_line_drawing(g)
        st = _tess_stats(tess); st["crop_box"] = list(box)
        out[name] = st; tess_d[name] = tess; img_d[name] = g
    return out, tess_d, img_d


def pipeline_control(delta: float, lloyd: int) -> dict:
    """Rasterise a froth with known exact statistics, run the line-drawing pipeline on it
    and report recovery errors (p6, <n>, mu2) and the pixel perimeter bias of the
    excess-perimeter estimate (used to correct the specimen value)."""
    vor = froth(delta, lloyd, nx=16, ny=16, seed=SEED + 7)
    exact = stats_of(vor)
    img = rasterize_froth(vor, px=900, linewidth_pt=2.0)
    tess = segment_line_drawing(img, min_distance=12, threshold_abs=5)
    rec = _tess_stats(tess)
    honey = froth(0.0, 0, nx=16, ny=16)
    himg = rasterize_froth(honey, px=900, linewidth_pt=2.0)
    hrec = _tess_stats(segment_line_drawing(himg, min_distance=12, threshold_abs=5))
    return {"delta": delta, "lloyd": lloyd,
            "exact": {k: exact[k] for k in ("p6", "mean_n", "mu2", "excess_perimeter", "F")},
            "recovered": {k: rec[k] for k in ("p6", "mean_n", "mu2", "excess_perimeter", "n_interior", "trivalent_fraction")},
            "errors": {"p6": rec["p6"] - exact["p6"], "mean_n": rec["mean_n"] - exact["mean_n"], "mu2": rec["mu2"] - exact["mu2"]},
            "perimeter_bias": rec["excess_perimeter"] - exact["excess_perimeter"],
            "honeycomb_raster": {k: hrec[k] for k in ("p6", "mean_n", "mu2", "excess_perimeter", "n_interior", "trivalent_fraction")},
            "honeycomb_perimeter_bias": hrec["excess_perimeter"] - 0.0}


def verdict_from(sp: dict, gibson_gof: dict) -> dict:
    c1_ok = abs(sp["mean_n"] - 6.0) <= C1_MEAN_TOL and (sp["trivalent_fraction"] or 0) >= C1_TRIVALENT_MIN
    c2_ok = sp["excess_perimeter_corrected"] <= C2_EXCESS_MAX
    strict = sp["modal_class"] == 6 and sp["p6"] >= C3_P6_STRICT
    froth_ok = C3_P6_FROTH_RANGE[0] <= sp["p6"] <= C3_P6_FROTH_RANGE[1]
    if sp["n_interior"] < 40:
        v = "untestable"
    elif c1_ok and c2_ok and strict:
        v = "supported"
    elif c1_ok and (c2_ok or froth_ok):
        v = "mixed"
    else:
        v = "refuted"
    return {
        "verdict": v,
        "C1": {"pass": bool(c1_ok), "summary": f"<n> = {sp['mean_n']:.2f} (|dev| {abs(sp['mean_n'] - 6):.2f} <= {C1_MEAN_TOL}), "
                                               f"trivalent junctions {100 * (sp['trivalent_fraction'] or 0):.0f} % >= {100 * C1_TRIVALENT_MIN:.0f} %"},
        "C2": {"pass": bool(c2_ok), "summary": f"excess perimeter {100 * sp['excess_perimeter_corrected']:.1f} % (bias-corrected) <= {100 * C2_EXCESS_MAX:.0f} % of the honeycomb bound"},
        "C3": {"pass": bool(strict), "strict_pass": bool(strict), "froth_reading_pass": bool(froth_ok),
               "summary": f"p6 = {sp['p6']:.2f} modal={sp['modal_class'] == 6}; strict (>= {C3_P6_STRICT}) {'PASS' if strict else 'FAIL'}; "
                          f"froth reading ({C3_P6_FROTH_RANGE[0]}-{C3_P6_FROTH_RANGE[1]}) {'PASS' if froth_ok else 'FAIL'}"},
        "qualification": ("Thompson's exemplar — a non-proliferating adult ocular monolayer — is the hexagon-rich corner of the froth "
                          "morphospace; proliferating epithelia (Gibson 2006: p6 = 0.46) fail the strict hexagon reading "
                          f"(chi-square p vs Gibson = {gibson_gof['chi2']['p_value']:.2g}), so the strict reading is tissue-state dependent."),
    }


def run(canonical_delta: float = 0.35, lloyd: int = 0, nx: int = 20) -> dict:
    t0 = time.time()
    ny = nx if nx % 2 == 0 else nx + 1
    # --- specimen and 1917 plate
    sp, spec_tess, spec_img = analyse_specimen()
    fig125, fig125_tess, fig125_img = analyse_fig125()
    # --- pipeline control first: its perimeter bias corrects the specimen value used by the fit
    pipe = pipeline_control(canonical_delta, lloyd)
    sp["excess_perimeter_corrected"] = sp["excess_perimeter"] - pipe["perimeter_bias"]
    # finite-window bias of <n>: interior cells next to border cells lose neighbours; the
    # rasterised froth (exact <n> = 6) measures the same bias, which is reported, not used
    sp["mean_n_window_bias"] = pipe["errors"]["mean_n"]
    sp["mean_n_window_corrected"] = sp["mean_n"] - pipe["errors"]["mean_n"]
    # --- froth morphospace and fit
    sweep = parameter_sweep(nx=nx, ny=ny)
    fit = fit_to_specimen(sp, sweep)
    best = fit["best"]
    model = froth(best["delta"], best["lloyd"], nx, ny)
    model_stats = stats_of(model)
    joint_model_stats = stats_of(froth(fit["best_joint"]["delta"], fit["best_joint"]["lloyd"], nx, ny))
    canonical = stats_of(froth(canonical_delta, lloyd, nx, ny))
    # --- controls
    honeycomb = stats_of(froth(0.0, 0, nx, ny))
    poisson = stats_of(froth(0.0, 0, nx, ny, poisson=True))
    shuffle = polygon_statistics(spec_tess["sides"], spec_tess["areas"], spec_tess["perims"], spec_tess["neighbours"],
                                 keep=spec_tess["interior"], shuffle_seed=SEED)["aboav_weaire"]
    gibson = gibson_markov_fixed_point()
    gibson_ablation = gibson_markov_fixed_point(cleavage="uniform_random")
    # --- goodness of fit
    honey_p = {k: (1.0 if k == 6 else 0.0) for k in CLASSES}
    gof = {
        "vs_gibson": {**distribution_distance(sp["p_n"], GIBSON_2006_MARKOV), "chi2": chi_square_vs(sp["counts"], GIBSON_2006_MARKOV)},
        "vs_best_froth": {**distribution_distance(sp["p_n"], model_stats["p_n"]), "chi2": chi_square_vs(sp["counts"], model_stats["p_n"])},
        "vs_division_chain": {**distribution_distance(sp["p_n"], gibson["distribution"]), "chi2": chi_square_vs(sp["counts"], gibson["distribution"])},
        "vs_honeycomb": {**distribution_distance(sp["p_n"], honey_p), "chi2": chi_square_vs(sp["counts"], {**honey_p, 5: 1e-9, 7: 1e-9})},
        "vs_poisson_null": {**distribution_distance(sp["p_n"], poisson["p_n"]), "chi2": chi_square_vs(sp["counts"], poisson["p_n"])},
        "fig125A_vs_specimen": distribution_distance(fig125["A"]["p_n"], sp["p_n"]),
        "specimen_p6_vs_cornea_norm": {"specimen": sp["p6"], "norm": CORNEA_NORM_P6, "norm_range": list(CORNEA_NORM_P6_RANGE),
                                       "within_norm": bool(CORNEA_NORM_P6_RANGE[0] <= sp["p6"] <= CORNEA_NORM_P6_RANGE[1])},
        "specimen_area_cv_vs_cornea_norm": {"specimen": sp["area_cv"], "norm": CORNEA_NORM_CV_AREA, "abs_diff": abs(sp["area_cv"] - CORNEA_NORM_CV_AREA)},
    }
    verdict = verdict_from(sp, gof["vs_gibson"])
    # --- figures
    out_png = MEDIA / f"{SLUG}_comparison.png"
    out_gif = MEDIA / f"{SLUG}_growth.gif"
    metrics = {
        "sim_id": SIM_ID, "claim_id": "T1917-019", "pass": 2, "seed": SEED,
        "specimen": sp, "thompson_fig125": fig125,
        "model_fit": {"best": best, "best_unrelaxed": fit["best_unrelaxed"], "best_joint": fit["best_joint"],
                      "stats": model_stats, "joint_stats": joint_model_stats},
        "canonical_model": {"delta": canonical_delta, "lloyd": lloyd, **canonical},
        "sweep": sweep,
        "controls": {"honeycomb": honeycomb, "poisson_voronoi": poisson, "pipeline_raster": pipe,
                     "aboav_shuffle_specimen": shuffle, "gibson_uniform_random_cleavage_ablation": gibson_ablation},
        "gibson_division_chain": gibson, "gof": gof, "verdict": verdict,
        "input_sha256": {p.name: sha256(p) for p in (SPECIMEN, ELODEA, THOMPSON_P322)},
        "outputs": {"comparison_png": str(out_png.relative_to(ROOT)), "growth_gif": str(out_gif.relative_to(ROOT))},
    }
    render_comparison(metrics, model, spec_tess, spec_img, fig125_tess, fig125_img, out_png)
    metrics["relaxation_trace"] = render_relaxation_gif(out_gif, target_p6=sp["p6"])
    metrics["runtime_s"] = round(time.time() - t0, 2)
    (DATA / f"{SLUG}_metrics.json").write_text(json.dumps(metrics, indent=1, default=_json_default))
    write_result(metrics)
    return metrics


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def write_result(m: dict) -> Path:
    sp = m["specimen"]; v = m["verdict"]; fa = m["thompson_fig125"]["A"]; fb = m["thompson_fig125"]["B"]
    res = {
        "claim_id": "T1917-019",
        "date_made": "1917",
        "claim": "Tissue cells tessellate approximately as hexagons / as a froth, the configuration that partitions area with least perimeter.",
        "source": "Thompson, D.W. (1917) On Growth and Form, Ch. VII, pp. 293-345, Figs. 120-127 (IA ongrowthform1917thom)",
        "dataset": {
            "name": "Human corneal endothelium in situ, alizarin red (Zhiguo He, Wikimedia Commons, CC BY-SA 4.0) + Thompson 1917 Fig. 125 A/B digitised (IA leaf n341 = p. 322)",
            "doi": None,
            "url": "https://commons.wikimedia.org/wiki/File:Morphology_of_human_corneal_endothelium_in_situ.jpg",
            "sha256": m["input_sha256"][SPECIMEN.name],
            "sha256_thompson_p322": m["input_sha256"][THOMPSON_P322.name],
            "match": "direct instance (Thompson p. 321 names the hexagonal epithelium cells of the eye); not an analogue",
        },
        "verdict": v["verdict"],
        "verdict_parts": {k: v[k] for k in ("C1", "C2", "C3")},
        "qualification": v["qualification"],
        "headline_numbers": {
            "specimen_interior_cells": sp["n_interior"],
            "specimen_mean_sides": round(sp["mean_n"], 3), "specimen_mean_sides_ci95": [round(x, 3) for x in sp["mean_n_ci"]],
            "specimen_mean_sides_window_bias_corrected": round(sp["mean_n_window_corrected"], 3),
            "specimen_p6": round(sp["p6"], 3), "specimen_p6_ci95": [round(x, 3) for x in sp["p6_ci"]],
            "specimen_p5": round(sp["p_n"][5], 3), "specimen_p7": round(sp["p_n"][7], 3),
            "specimen_mu2": round(sp["mu2"], 3),
            "specimen_trivalent_junction_fraction": round(sp["trivalent_fraction"], 3),
            "specimen_excess_perimeter_raw": round(sp["excess_perimeter"], 4),
            "specimen_excess_perimeter_bias_corrected": round(sp["excess_perimeter_corrected"], 4),
            "specimen_shape_index_median": round(sp["shape_index_median"], 3), "regular_hexagon_shape_index": round(HEX_SHAPE_INDEX, 4),
            "specimen_area_cv": round(sp["area_cv"], 3), "cornea_norm_area_cv": CORNEA_NORM_CV_AREA,
            "cornea_norm_p6": CORNEA_NORM_P6,
            "specimen_lewis_lambda": None if sp["lewis"]["lambda"] is None else round(sp["lewis"]["lambda"], 3),
            "specimen_aboav_a": None if sp["aboav_weaire"]["a"] is None else round(sp["aboav_weaire"]["a"], 3),
            "thompson_fig125A_p6": round(fa["p6"], 3), "thompson_fig125A_mean_sides": round(fa["mean_n"], 3),
            "thompson_fig125B_p6": round(fb["p6"], 3), "thompson_fig125B_mean_sides": round(fb["mean_n"], 3),
            "best_fit_froth_delta": m["model_fit"]["best"]["delta"], "best_fit_froth_lloyd": m["model_fit"]["best"]["lloyd"],
            "best_fit_froth_tv_distance": round(m["model_fit"]["best"]["tv"], 4),
            "best_fit_froth_excess_perimeter": round(m["model_fit"]["stats"]["excess_perimeter"], 4),
            "joint_fit_froth_delta": m["model_fit"]["best_joint"]["delta"], "joint_fit_froth_lloyd": m["model_fit"]["best_joint"]["lloyd"],
            "joint_fit_froth_tv_distance": round(m["model_fit"]["best_joint"]["tv"], 4),
            "joint_fit_area_cv_diff": round(m["model_fit"]["best_joint"]["area_cv_diff"], 4),
            "joint_fit_excess_perimeter_diff": round(m["model_fit"]["best_joint"]["excess_perimeter_diff"], 4),
            "gibson2006_p6": GIBSON_2006_MARKOV[6],
            "r_vs_gibson": round(m["gof"]["vs_gibson"]["r"], 3), "rmse_vs_gibson": round(m["gof"]["vs_gibson"]["rmse"], 4),
            "chi2_p_vs_gibson": m["gof"]["vs_gibson"]["chi2"]["p_value"],
            "r_vs_best_froth": round(m["gof"]["vs_best_froth"]["r"], 3), "rmse_vs_best_froth": round(m["gof"]["vs_best_froth"]["rmse"], 4),
            "chi2_p_vs_best_froth": m["gof"]["vs_best_froth"]["chi2"]["p_value"],
            "tv_vs_honeycomb": round(m["gof"]["vs_honeycomb"]["tv"], 4),
            "poisson_null_p6": round(m["controls"]["poisson_voronoi"]["p6"], 3),
            "pipeline_control_p6_error": round(m["controls"]["pipeline_raster"]["errors"]["p6"], 4),
            "pipeline_perimeter_bias": round(m["controls"]["pipeline_raster"]["perimeter_bias"], 4),
        },
        "script": "thompson/claims/t1917-019_the-forms-of-tissues-or-cell-aggregates.py",
        "figure": m["outputs"]["comparison_png"],
        "animation": m["outputs"]["growth_gif"],
        "lean": "morphoeval/lean/loops/Thompson1917_T1917_019.lean",
        "seed": SEED,
    }
    p = DATA / "result.json"
    p.write_text(json.dumps(res, indent=1, default=_json_default))
    return p


def measure_morphoeval() -> list[dict]:
    """Quantitative morphometrics for morphoeval/measure_outputs.py (dynamic discovery)."""
    mpath = DATA / f"{SLUG}_metrics.json"
    m = json.loads(mpath.read_text()) if mpath.exists() else None
    if m is None or m.get("pass") != 2:
        m = run()
    sp = m["specimen"]; ms = m["model_fit"]["stats"]; bf = m["model_fit"]["best"]; gb = m["gibson_division_chain"]
    ctl = m["controls"]; fa = m["thompson_fig125"]["A"]; gof = m["gof"]

    def M(name, value, unit, method, reference=None, source=None, note=None, derived_from="output"):
        d = {"name": name, "value": None if value is None else float(round(value, 4)), "unit": unit,
             "method": method, "derived_from": derived_from}
        if reference is not None: d["reference_value"] = reference
        if source: d["reference_source"] = source
        if note: d["note"] = note
        return d

    return [
        M("specimen mean polygon side number <n> (interior cells)", sp["mean_n"], "sides",
          f"nucleus-marker watershed on human corneal endothelium, {sp['n_interior']} interior cells; bootstrap 95% CI "
          f"[{sp['mean_n_ci'][0]:.2f}, {sp['mean_n_ci'][1]:.2f}]; finite-window bias from raster control {sp['mean_n_window_bias']:+.3f} "
          f"(window-corrected <n> = {sp['mean_n_window_corrected']:.3f})", reference=6.0,
          source="Thompson 1917 p. 325 'on the average hexagonal'; Euler (Lean Law 1 torus_mean_six); Weaire & Rivier 1984 Eq. 2.1"),
        M("specimen hexagon fraction p6", sp["p6"], "fraction",
          f"interior-cell neighbour count = 6; bootstrap 95% CI [{sp['p6_ci'][0]:.2f}, {sp['p6_ci'][1]:.2f}]",
          reference=CORNEA_NORM_P6, source="Sheng & Bullimore 2007 Cornea 26:520 (%Hex regression, age 40): normal range 0.55-0.70",
          note=f"honeycomb (Thompson strict) = 1.0; Gibson 2006 proliferating wing disc = {GIBSON_2006_MARKOV[6]}; Poisson-Voronoi null = {ctl['poisson_voronoi']['p6']:.2f}"),
        M("specimen second moment mu2 of polygon distribution", sp["mu2"], "sides^2",
          f"sum p_n (n-6)^2 over interior cells; CI [{sp['mu2_ci'][0]:.2f}, {sp['mu2_ci'][1]:.2f}]", reference=0.0,
          source="honeycomb mu2 = 0 (Lean Law 2 honeycomb_iff_zero_second_moment); Weaire & Rivier 1984 Eq. 2.3",
          note=f"Poisson-Voronoi null control mu2 = {ctl['poisson_voronoi']['mu2']:.2f}; best-fit froth mu2 = {ms['mu2']:.2f}"),
        M("specimen trivalent junction fraction", sp["trivalent_fraction"], "fraction",
          "2x2 label windows with 3 distinct labels / (3 or >=4)", reference=1.0,
          source="Thompson 1917 p. 309, 323 (three partitions meet at a point, Lamarle/Plateau)"),
        M("specimen excess perimeter over honeycomb bound (bias-corrected)", 100 * sp["excess_perimeter_corrected"], "%",
          f"sum P_i / sum 2*12^(1/4) sqrt(A_i) - 1 on Crofton perimeters, minus rasterised-froth pipeline bias "
          f"{100 * ctl['pipeline_raster']['perimeter_bias']:.2f} % (raw {100 * sp['excess_perimeter']:.2f} %)", reference=0.0,
          source="Hales 2001 Discrete Comput. Geom. 25:1 (honeycomb is the least-perimeter partition; Lean Law 3)",
          note=f"best-fit froth model excess perimeter = {100 * ms['excess_perimeter']:.2f} %"),
        M("specimen median shape index P/sqrt(A)", sp["shape_index_median"], "dimensionless", "Crofton perimeter / sqrt(pixel area), interior cells",
          reference=round(HEX_SHAPE_INDEX, 4), source="regular hexagon 2*(12)^(1/4) = 3.7224; pentagon 3.812, square 4.0"),
        M("specimen coefficient of variation of cell area (polymegethism)", sp["area_cv"], "CV",
          "std/mean of interior pixel areas", reference=CORNEA_NORM_CV_AREA,
          source="Sheng & Bullimore 2007 Cornea 26:520 (CoV regression, age 40, no contact lenses)"),
        M("specimen Lewis slope lambda (<A_n>/<A> = 1 + lambda (n-6))", sp["lewis"]["lambda"], "1/side",
          f"linear fit over classes {sp['lewis']['classes']}, r = {sp['lewis']['r']}", reference=0.25,
          source="Lewis 1928 Anat. Rec. 38:341 (Lean Law 4 lewis_quarter_slope_classes)"),
        M("specimen Aboav-Weaire parameter a", sp["aboav_weaire"]["a"], "dimensionless",
          f"fit n m(n) = (6-a) n + (6a+mu2), r = {sp['aboav_weaire']['r']}; neighbour-shuffle control a = {ctl['aboav_shuffle_specimen']['a']:.2f}",
          reference=1.2, source="Aboav 1970 Metallography 3:383; Weaire & Rivier 1984 (froths a = 1.0-1.5)"),
        M("specimen Aboav-Weaire sum-rule residual (1/F) sum n m(n) - (36 + mu2)", sp["aboav_weaire"]["sum_rule_residual"], "sides^2",
          "Lean Law 5 aboav_weaire_sum_rule consistency check (exact on a torus; finite-window specimen)", reference=0.0,
          source="Weaire & Rivier 1984 Eq. 2.7"),
        M("Pearson r specimen vs Gibson 2006 polygon distribution (n = 4..8)", gof["vs_gibson"]["r"], "r",
          f"RMSE = {gof['vs_gibson']['rmse']:.3f}, TV = {gof['vs_gibson']['tv']:.3f}, KS D = {gof['vs_gibson']['ks_D']:.3f}, "
          f"chi-square p = {gof['vs_gibson']['chi2']['p_value']:.3g}", reference=1.0,
          source="Gibson et al. 2006 Nature 442:1038 Fig. 3 (Drosophila wing disc Markov fixed point)"),
        M("Pearson r specimen vs best-fit froth model polygon distribution", gof["vs_best_froth"]["r"], "r",
          f"delta = {bf['delta']}, Lloyd = {bf['lloyd']}; RMSE = {gof['vs_best_froth']['rmse']:.3f}, TV = {gof['vs_best_froth']['tv']:.3f}, "
          f"chi-square p = {gof['vs_best_froth']['chi2']['p_value']:.3g}", reference=1.0,
          source="periodic Voronoi froth sweep (this script); Sanchez-Gutierrez et al. 2016 EMBO J 35:77 CVT path"),
        M("total-variation distance specimen vs honeycomb (= 1 - p6)", gof["vs_honeycomb"]["tv"], "TV",
          "0.5 sum |p_n - delta_n6|; Thompson strict reading baseline", reference=0.0, source="honeycomb (Thompson 1917 p. 321 strict reading)"),
        M("Thompson Fig. 125 A (compressed soap froth) hexagon fraction, digitised", fa["p6"], "fraction",
          f"line-drawing watershed on IA leaf n341 crop {fa['crop_box']}, {fa['n_interior']} interior cells, <n> = {fa['mean_n']:.2f}",
          reference=CORNEA_NORM_P6, source="Thompson 1917 Fig. 125 p. 322 (after Rhumbler) vs cornea norm; TV to specimen = "
                                           f"{gof['fig125A_vs_specimen']['tv']:.3f}"),
        M("pipeline control: p6 recovery error on rasterised froth", ctl["pipeline_raster"]["errors"]["p6"], "fraction",
          f"recovered - exact p6 for delta = {ctl['pipeline_raster']['delta']}, {ctl['pipeline_raster']['recovered']['n_interior']} interior cells; "
          f"honeycomb raster recovered p6 = {ctl['pipeline_raster']['honeycomb_raster']['p6']:.3f}", reference=0.0,
          source="positive control (exact Voronoi statistics known); Lean Law 1", derived_from="script"),
        M("division-topology chain fixed-point p6 (through-centre cleavage reconstruction)", gb["p6"], "fraction",
          f"mean-field Gibson 2006 Markov chain, p_cut = 1/3; uniform-random cleavage ablation p6 = {ctl['gibson_uniform_random_cleavage_ablation']['p6']:.3f}",
          reference=GIBSON_2006_MARKOV[6], source="Gibson et al. 2006 Nature 442:1038 (Lean Law 6 division_preserves_mean_six)", derived_from="script"),
    ]


def self_test() -> int:
    t0 = time.time()
    # Law 1: torus mean six for several disorders (Euler, exact)
    for d in (0.0, 0.2, 0.5, 1.0):
        v = froth(d, 0, nx=12, ny=12)
        assert abs(v["sides"].mean() - 6.0) < 1e-9, f"torus mean {v['sides'].mean()} != 6 at delta={d}"
    assert abs(froth(0.0, 0, nx=12, ny=12, poisson=True)["sides"].mean() - 6.0) < 1e-9
    # Law 2 / 3: honeycomb has mu2 = 0, p6 = 1 and sits on the Hales bound
    st = stats_of(froth(0.0, 0, nx=12, ny=12))
    assert st["mu2"] == 0.0 and st["p6"] == 1.0
    assert abs(st["excess_perimeter"]) < 1e-6, st["excess_perimeter"]
    assert abs(st["shape_index_mean"] - HEX_SHAPE_INDEX) < 1e-6
    assert SHAPE_INDEX[6] < SHAPE_INDEX[5] < SHAPE_INDEX[4] < SHAPE_INDEX[3]
    assert abs(SHAPE_INDEX[6] ** 4 - 192.0) < 1e-9 and abs(SHAPE_INDEX[4] ** 4 - 256.0) < 1e-9 and abs(SHAPE_INDEX[3] ** 4 - 432.0) < 1e-6
    # disorder monotonicity: p6 falls, excess perimeter rises; Poisson null is froth-like
    seq = [stats_of(froth(d, 0, nx=14, ny=14)) for d in (0.0, 0.15, 0.4, 1.0)]
    assert all(a["p6"] >= b["p6"] for a, b in zip(seq, seq[1:]))
    assert all(a["excess_perimeter"] <= b["excess_perimeter"] + 1e-9 for a, b in zip(seq, seq[1:]))
    pz = stats_of(froth(0.0, 0, nx=14, ny=14, poisson=True))
    assert 0.2 < pz["p6"] < 0.4 and 1.3 < pz["mu2"] < 2.4, (pz["p6"], pz["mu2"])
    # Lloyd relaxation lowers excess perimeter; Law 5 sum rule exact on the torus
    s0 = stats_of(froth(0.6, 0, nx=12, ny=12)); s1 = stats_of(froth(0.6, 8, nx=12, ny=12))
    assert s1["excess_perimeter"] < s0["excess_perimeter"] and s1["p6"] >= s0["p6"]
    assert abs(s0["aboav_weaire"]["sum_rule_residual"]) < 1e-9
    # shuffle control destroys the Aboav-Weaire correlation (a -> ~6, i.e. slope ~0)
    sh = stats_of(froth(0.6, 0, nx=12, ny=12), shuffle_seed=1)["aboav_weaire"]
    assert abs(sh["slope"] - 6.0) < 1.0 and s0["aboav_weaire"]["slope"] < 5.5, (sh["slope"], s0["aboav_weaire"]["slope"])
    # Law 6: division chain stochastic, mean six for both cleavage rules, p6 near Gibson for through-centre
    g = gibson_markov_fixed_point(); ga = gibson_markov_fixed_point(cleavage="uniform_random")
    assert g["transition_matrix_rows_sum_to_one"] and abs(g["mean_n"] - 6.0) < 0.01 and abs(ga["mean_n"] - 6.0) < 0.2
    assert 0.40 < g["p6"] < 0.50 and ga["p6"] < g["p6"] and ga["mu2"] > 2 * g["mu2"], (g["p6"], ga["p6"])
    # statistics helpers
    dd = distribution_distance(GIBSON_2006_MARKOV, GIBSON_2006_MARKOV)
    assert dd["tv"] == 0.0 and dd["rmse"] == 0.0 and abs(dd["r"] - 1.0) < 1e-12
    ci = bootstrap_ci(np.array([6] * 50 + [5] * 25 + [7] * 25), lambda v: np.mean(v == 6), n_boot=300)
    assert ci["ci95"][0] <= 0.5 <= ci["ci95"][1]
    # pipeline control: rasterised froth recovers exact statistics
    pc = pipeline_control(0.2, 0)
    assert pc["recovered"]["n_interior"] >= 100, pc["recovered"]["n_interior"]
    assert abs(pc["errors"]["p6"]) < 0.08 and abs(pc["errors"]["mean_n"]) < 0.15, pc["errors"]
    assert pc["honeycomb_raster"]["p6"] > 0.95 and abs(pc["honeycomb_perimeter_bias"]) < 0.08, pc["honeycomb_raster"]
    # specimen segmentation sanity (real image)
    if SPECIMEN.exists():
        sp, tess, _ = analyse_specimen()
        assert 100 <= sp["n_cells"] <= 200 and sp["n_interior"] >= 60, (sp["n_cells"], sp["n_interior"])
        assert 5.6 < sp["mean_n"] < 6.4 and sp["modal_class"] == 6, (sp["mean_n"], sp["modal_class"])
        assert sp["trivalent_fraction"] > 0.8
        fig125, _, _ = analyse_fig125()
        assert fig125["A"]["n_interior"] >= 40 and fig125["B"]["n_interior"] >= 40
        assert fig125["B"]["p6"] > 0.85 and 5.7 < fig125["A"]["mean_n"] < 6.3
    # determinism
    assert np.array_equal(froth(0.35, 0, nx=8, ny=8)["sides"], froth(0.35, 0, nx=8, ny=8)["sides"])
    print(f"self-test OK ({time.time() - t0:.1f}s): torus <n>=6 exact; honeycomb mu2=0, p6=1, excess=0; P^4 = 192/256/432; "
          f"disorder monotone; Poisson null p6={pz['p6']:.2f}; Lloyd relaxes; AW sum rule exact; shuffle control; "
          f"Gibson chain p6={g['p6']:.3f} (ablation {ga['p6']:.3f}); raster control p6 err={pc['errors']['p6']:+.3f}; "
          f"specimen + Fig. 125 segmentation sane")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--delta", type=float, default=0.35, help="canonical froth disorder (lattice spacings)")
    ap.add_argument("--lloyd", type=int, default=0, help="Lloyd relaxation steps for the canonical froth")
    ap.add_argument("--nx", type=int, default=20, help="lattice cells per side (ny = nx, made even)")
    args = ap.parse_args(argv)
    if args.self_test:
        return self_test()
    m = run(args.delta, args.lloyd, args.nx)
    sp = m["specimen"]; v = m["verdict"]; bf = m["model_fit"]["best"]
    print(f"[{SIM_ID}] specimen: {sp['n_cells']} cells, {sp['n_interior']} interior; <n>={sp['mean_n']:.3f} "
          f"[{sp['mean_n_ci'][0]:.2f},{sp['mean_n_ci'][1]:.2f}] p6={sp['p6']:.3f} [{sp['p6_ci'][0]:.2f},{sp['p6_ci'][1]:.2f}] "
          f"mu2={sp['mu2']:.3f} trivalent={sp['trivalent_fraction']:.3f} CV(A)={sp['area_cv']:.3f} "
          f"excess perimeter raw={100 * sp['excess_perimeter']:.2f}% corrected={100 * sp['excess_perimeter_corrected']:.2f}%")
    print(f"[{SIM_ID}] Fig.125 A p6={m['thompson_fig125']['A']['p6']:.3f} <n>={m['thompson_fig125']['A']['mean_n']:.3f}; "
          f"B p6={m['thompson_fig125']['B']['p6']:.3f}")
    print(f"[{SIM_ID}] best froth fit delta={bf['delta']} lloyd={bf['lloyd']} TV={bf['tv']:.3f} r={bf['r']:.3f}; "
          f"vs Gibson r={m['gof']['vs_gibson']['r']:.3f} chi2 p={m['gof']['vs_gibson']['chi2']['p_value']:.3g}")
    print(f"[{SIM_ID}] VERDICT: {v['verdict']}  C1={v['C1']['pass']} C2={v['C2']['pass']} C3(strict)={v['C3']['pass']} "
          f"C3(froth)={v['C3']['froth_reading_pass']}")
    print(f"[{SIM_ID}] wrote {m['outputs']['comparison_png']}, {m['outputs']['growth_gif']}, "
          f"datasets/thompson_1917/t1917-019/result.json in {m['runtime_s']} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())

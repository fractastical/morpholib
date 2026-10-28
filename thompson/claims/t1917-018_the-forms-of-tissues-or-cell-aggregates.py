#!/usr/bin/env python3
"""
t1917-018_the-forms-of-tissues-or-cell-aggregates.py — Thompson (1917) claim T1917-018 (Pass 2, v2)

Claim under test (On Growth and Form, Ch. VII "The Forms of Tissues, or Cell-Aggregates",
pp. 293-345, Figs. 100-135; Internet Archive `ongrowthform1917thom`, jp2 leaf = page + 20):

    "Cells in contact, like soap films, meet three-at-a-time at ~120 degrees and in other
     respects obey Plateau's laws of film configuration."

Model (datasets/thompson_1917/t1917-018/t1917_018_the_forms_of_tissues_or_cell_a_reference.json):
  E1  Lami / Thompson Fig. 113 (pp. 296-297, 303): tensions T1,T2,T3 at a tri-junction balance iff
      T_i / sin(alpha_i) is constant;  cos(alpha_i) = (T_i^2 - T_j^2 - T_k^2) / (2 T_j T_k).
      Equal tensions -> 120/120/120; cases (1)-(3) of Fig. 113 = 90 / ~100 / 120 degrees.
      `tensions_from_angles` is the inverse map used to read tension ratios off a measured vertex.
  E2  Plateau / Lamarle (pp. 309-310; Taylor 1976): minimal partitions have only three-fold vertices
      at 120 degrees; the unit-square X partition (2 sqrt 2) is longer than the two-Y Steiner one
      (1 + sqrt 3), so four-fold film vertices are unstable.  Scored as the three-fold fraction f3.
  E3  alpha + beta + gamma = 360 at every planar tri-junction: the MEAN is 120 by identity, so the
      test statistics are dispersions (SD, RMS deviation, median |theta - 120|, fraction within the
      a-priori +/-15 degree tolerance) and never the mean.
  E5  Periodic vertex model  E = sum_c [ (K/2)(A_c - A0)^2 + (Gamma/2) P_c^2 ] + sum_e Lambda(phi_e) l_e
      with Lambda(phi) = Lambda0 (1 + eps cos 2 phi)  (Farhadifar et al. 2007 Eq. 1 + orientation-
      dependent tension, Alt et al. 2017), relaxed by gradient descent WITH T1 edge exchanges from a
      Poisson-Voronoi seed.  Gamma = eps = 0 is the equal-tension soap froth (Plateau limit);
      eps > 0 is Thompson's "asymmetrical component arising from growth or traction" (Fig. 124).

Specimens (all SHA-256-verified at start-up, see reference.json):
  * Homo sapiens corneal endothelium in situ (Wikimedia Commons, CC BY-SA 4.0) - PRIMARY, isotropic
  * Allium cepa bulb epidermis (Wikimedia Commons, CC BY-SA 4.0)              - anisotropic (Fig. 124 regime)
  * Thompson Fig. 103 Zea mays parenchyma, Fig. 124 Girardia epidermis (IA leaves 318 / 341)
  The pre-scouted Dryad set doi:10.5061/dryad.mn8cs is a DATASET_MISMATCH (Ca2+ ratio imaging) and is not used.

Junction-angle estimators (both reported):
  chord   - angles between the straight chords from a vertex to its three neighbouring vertices
            (standard vertex-model geometry; parameter-free; primary estimator)
  branch  - angles between the mean offset directions of the three wall segments within R = L_e/3
            of the vertex (L_e = median chord length; follows curved walls; secondary estimator)

Lean 4 skeleton: morphoeval/lean/loops/Thompson1917_T1917_018.lean (MathOfBiology.Loop_213).

Usage:
    python3 thompson/claims/t1917-018_the-forms-of-tissues-or-cell-aggregates.py              # full run
    python3 thompson/claims/t1917-018_the-forms-of-tissues-or-cell-aggregates.py --self-test
    python3 thompson/claims/t1917-018_the-forms-of-tissues-or-cell-aggregates.py --quick       # fewer steps/boots
    options: --eps-sweep 0,0.1,...  --lambda-bar 0.12  --gamma-bar 0.0  --seed 1917  --steps 3000
             --n-cells 96  --bootstrap 400  --tolerance 15  --no-figures

Outputs:
    docs/media/t1917_018_the_forms_of_tissues_or_cell_a_comparison.png
    docs/media/t1917_018_the_forms_of_tissues_or_cell_a_growth.gif
    datasets/thompson_1917/t1917-018/result.json           (verdict + headline numbers)
    datasets/thompson_1917/t1917-018/t1917_018_metrics.json (every number behind the figure)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "datasets" / "thompson_1917" / "t1917-018"
REFERENCE_JSON = DATA_DIR / "t1917_018_the_forms_of_tissues_or_cell_a_reference.json"
RESULT_JSON = DATA_DIR / "result.json"
METRICS_JSON = DATA_DIR / "t1917_018_metrics.json"
MEDIA_DIR = ROOT / "docs" / "media"
STEM = "t1917_018_the_forms_of_tissues_or_cell_a"
COMPARISON_PNG = MEDIA_DIR / f"{STEM}_comparison.png"
GROWTH_GIF = MEDIA_DIR / f"{STEM}_growth.gif"

CLAIM_ID = "T1917-018"
PLATEAU_DEG = 120.0
TOLERANCE_DEG = 15.0           # a-priori (reference.json: tolerance_deg_stated_a_priori)
F3_THRESHOLD = 0.90            # a-priori three-fold fraction for Plateau's second law
SEED = 1917
N_CELLS = 96
LAMBDA_BAR = 0.12              # Farhadifar et al. (2007) Case I line tension
GAMMA_BAR_FARHADIFAR = 0.04    # perimeter contractility of the wing-disc fit
K_AREA = 1.0
RELAX_STEPS = 3000
RELAX_DT = 0.05
T1_LENGTH = 0.04               # edges shorter than this (in units of sqrt(A0)) undergo a T1 exchange
EPS_SWEEP_DEFAULT = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6)

SPECIMENS = {
    # key: (file, kind, label, regime, analogue flag)
    "human_corneal_endothelium": ("human_corneal_endothelium_specimen.jpg", "cornea",
                                  "Homo sapiens corneal endothelium (Commons, CC BY-SA 4.0)",
                                  "isotropic monolayer (Plateau regime)", "modern specimen, not a Thompson figure"),
    "allium_cepa_epidermis": ("allium_cepa_epidermis_specimen.jpg", "onion",
                              "Allium cepa epidermis (Commons, CC BY-SA 4.0)",
                              "anisotropically grown epidermis (Fig. 124 regime)", "modern specimen, not a Thompson figure"),
    "thompson_fig103_maize_parenchyma": ("thompson1917_fig103_maize_parenchyma.png", "drawing:9",
                                         "Thompson 1917 Fig. 103 Zea mays parenchyma (p. 298)",
                                         "Thompson's own 120-degree exemplar (line drawing)", "Thompson figure"),
    "thompson_fig124_girardia_epidermis": ("thompson1917_fig124_girardia_epidermis.png", "drawing",
                                           "Thompson 1917 Fig. 124 Girardia epidermis (p. 321)",
                                           "Thompson's own growth-distorted counter-regime (line drawing)", "Thompson figure"),
}


# =============================================================================== provenance
def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_reference() -> dict:
    with open(REFERENCE_JSON, encoding="utf-8") as fh:
        return json.load(fh)


def verify_inputs(ref: dict) -> Dict[str, str]:
    """SHA-256-verify every pinned specimen (and the IA leaves) against reference.json; exit on mismatch."""
    digests = {}
    for key, spec in ref["specimens"].items():
        for fkey, hkey in (("file", "sha256"), ("leaf_file", "leaf_sha256")):
            if fkey not in spec:
                continue
            p = ROOT / spec[fkey]
            got = sha256_of(p)
            if got != spec[hkey]:
                sys.exit(f"{p}: sha256 {got} != pinned {spec[hkey]} ({key})")
            digests[spec[fkey]] = got
    return digests


# =============================================================================== E1: Lami / Fig. 113
def lami_angles_from_tensions(T1: float, T2: float, T3: float) -> Optional[Tuple[float, float, float]]:
    """Interior angles (deg) opposite films 1, 2, 3 at a balanced tri-junction (Thompson pp. 296-297).

    None when no triangle of forces exists (Thompson's 'no triangle having sides of these relative
    magnitudes is possible', p. 296)."""
    T = (T1, T2, T3)
    if min(T) <= 0:
        return None
    out = []
    for i in range(3):
        j, k = (i + 1) % 3, (i + 2) % 3
        c = (T[i] ** 2 - T[j] ** 2 - T[k] ** 2) / (2.0 * T[j] * T[k])
        if c < -1.0 - 1e-12 or c > 1.0 + 1e-12:
            return None
        out.append(math.degrees(math.acos(max(-1.0, min(1.0, c)))))
    return out[0], out[1], out[2]


def tensions_from_angles(alpha: float, beta: float, gamma: float) -> Tuple[float, float, float]:
    """Inverse Lami: tension ratios (T1:T2:T3, normalised to mean 1) from the three interior angles."""
    s = np.array([math.sin(math.radians(a)) for a in (alpha, beta, gamma)])
    s = np.clip(s, 1e-9, None)
    return tuple((s / s.mean()).tolist())


def thompson_fig113_cases() -> Dict[str, Tuple[float, float, float]]:
    """The three worked cases of Fig. 113 (p. 303): T = T' with t negligible / small / equal."""
    return {
        "case1_t_negligible": lami_angles_from_tensions(1.0, 1.0, 1e-6),
        "case2_t_small": lami_angles_from_tensions(1.0, 1.0, 0.3473),   # alpha = beta = 100 deg
        "case3_equal": lami_angles_from_tensions(1.0, 1.0, 1.0),
    }


def steiner_vs_cross_lengths() -> Tuple[float, float]:
    """Unit-square film lengths: X partition (four-fold vertex) vs two-Y Steiner partition (E2)."""
    return 2.0 * math.sqrt(2.0), 1.0 + math.sqrt(3.0)


# =============================================================================== E3: statistics
def plateau_statistics(angles_deg: np.ndarray, tol: float = TOLERANCE_DEG) -> Dict[str, float]:
    a = np.asarray(angles_deg, float)
    dev = a - PLATEAU_DEG
    if a.size == 0:
        return {"n_angles": 0, "mean_deg": float("nan"), "sd_deg": float("nan"), "rms_dev_deg": float("nan"),
                "median_abs_dev_deg": float("nan"), "frac_within_tol": float("nan"), "cv": float("nan")}
    return {
        "n_angles": int(a.size),
        "mean_deg": float(a.mean()),
        "sd_deg": float(a.std(ddof=1)) if a.size > 1 else float("nan"),
        "rms_dev_deg": float(np.sqrt(np.mean(dev ** 2))),
        "median_abs_dev_deg": float(np.median(np.abs(dev))),
        "frac_within_tol": float(np.mean(np.abs(dev) <= tol)),
        "cv": float(a.std(ddof=1) / a.mean()) if a.size > 1 else float("nan"),
    }


def bootstrap_vertex_ci(vertex_angles: List[List[float]], n_boot: int, seed: int,
                        tol: float = TOLERANCE_DEG) -> Dict[str, Tuple[float, float]]:
    """Percentile bootstrap (resampling vertices, not angles) of median |dev|, SD and fraction within tol."""
    rng = np.random.default_rng(seed)
    V = len(vertex_angles)
    if V < 3:
        return {}
    med, sd, frac = [], [], []
    arr = [np.asarray(v) for v in vertex_angles]
    for _ in range(n_boot):
        idx = rng.integers(0, V, V)
        a = np.concatenate([arr[i] for i in idx])
        d = np.abs(a - PLATEAU_DEG)
        med.append(np.median(d))
        sd.append(a.std(ddof=1))
        frac.append(np.mean(d <= tol))
    pct = lambda x: (float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5)))
    return {"median_abs_dev_deg": pct(med), "sd_deg": pct(sd), "frac_within_tol": pct(frac)}


def two_sample_ks(a: np.ndarray, b: np.ndarray) -> Tuple[float, float]:
    from scipy import stats
    if len(a) < 2 or len(b) < 2:
        return float("nan"), float("nan")
    r = stats.ks_2samp(a, b)
    return float(r.statistic), float(r.pvalue)


def histogram_fit(a: np.ndarray, b: np.ndarray, bins: np.ndarray) -> Tuple[float, float]:
    """Pearson r and RMSE between the normalised angle histograms of two samples."""
    ha, _ = np.histogram(a, bins=bins, density=True)
    hb, _ = np.histogram(b, bins=bins, density=True)
    if ha.std() == 0 or hb.std() == 0:
        return float("nan"), float(np.sqrt(np.mean((ha - hb) ** 2)))
    return float(np.corrcoef(ha, hb)[0, 1]), float(np.sqrt(np.mean((ha - hb) ** 2)))


def angles_from_directions(dirs: Sequence[float]) -> List[float]:
    az = np.sort(np.asarray(dirs, float))
    gaps = np.diff(np.concatenate([az, [az[0] + 2 * math.pi]]))
    return np.degrees(gaps).tolist()


# =============================================================================== E5: periodic vertex model
class PeriodicNetwork:
    """Periodic 2D vertex network. `cells[c]` is a CCW list of (vertex_id, wrap_offset) pairs; the
    vertex position seen from cell c is pos[v] + offset."""

    def __init__(self, pos: np.ndarray, cells: List[List[Tuple[int, np.ndarray]]], L: float):
        self.pos = np.array(pos, float)
        self.cells = [list(c) for c in cells]
        self.L = float(L)
        self._build()

    # ---- flat slot arrays for vectorised energy / gradient
    def _build(self):
        vid, off, cid, nxt, prv = [], [], [], [], []
        start = 0
        for c, cell in enumerate(self.cells):
            n = len(cell)
            for k, (v, o) in enumerate(cell):
                vid.append(v); off.append(o); cid.append(c)
                nxt.append(start + (k + 1) % n); prv.append(start + (k - 1) % n)
            start += n
        self.s_vid = np.array(vid, int)
        self.s_off = np.array(off, float).reshape(-1, 2)
        self.s_cid = np.array(cid, int)
        self.s_nxt = np.array(nxt, int)
        self.s_prv = np.array(prv, int)
        self.n_cells = len(self.cells)

    def copy(self) -> "PeriodicNetwork":
        return PeriodicNetwork(self.pos.copy(), [[(v, o.copy()) for v, o in c] for c in self.cells], self.L)

    def cell_polygon(self, c: int) -> np.ndarray:
        return np.array([self.pos[v] + off for v, off in self.cells[c]])

    def slot_positions(self) -> np.ndarray:
        return self.pos[self.s_vid] + self.s_off

    def edge_table(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Per slot: edge vector to the next slot, its length and orientation angle."""
        P = self.slot_positions()
        d = P[self.s_nxt] - P
        l = np.linalg.norm(d, axis=1)
        return d, l, np.arctan2(d[:, 1], d[:, 0])

    def energy_gradient(self, lambda_bar: float = LAMBDA_BAR, gamma_bar: float = 0.0, K: float = K_AREA,
                        A0: float = 1.0, eps: float = 0.0, phi0: float = 0.0) -> Tuple[float, np.ndarray]:
        """E = sum_c [K/2 (A_c - A0)^2 + Gamma/2 P_c^2] + sum_e Lambda(phi_e) l_e  (each edge shared by 2 cells)."""
        P = self.slot_positions()
        x, y = P[:, 0], P[:, 1]
        xn, yn = x[self.s_nxt], y[self.s_nxt]
        cross = x * yn - xn * y
        A = 0.5 * np.bincount(self.s_cid, cross, minlength=self.n_cells)
        d, l, phi = self.edge_table()
        l = np.maximum(l, 1e-12)
        perim = np.bincount(self.s_cid, l, minlength=self.n_cells)
        lam = lambda_bar * (1.0 + eps * np.cos(2.0 * (phi - phi0)))
        E = float(np.sum(0.5 * K * (A - A0) ** 2 + 0.5 * gamma_bar * perim ** 2) + 0.5 * np.sum(lam * l))
        # area gradient (shoelace): dA/dP_k = 0.5 * (y_{k+1} - y_{k-1}, x_{k-1} - x_{k+1})
        dA = 0.5 * np.stack([y[self.s_nxt] - y[self.s_prv], x[self.s_prv] - x[self.s_nxt]], axis=1)
        unit = d / l[:, None]
        # perimeter gradient: edge k contributes -unit_k to slot k and +unit_k to slot k+1
        dP = -unit + unit[self.s_prv]
        # anisotropic line tension: d(lam l)/dP_{k+1} = lam * unit + dlam/dphi * (-dy, dx)/l ; slot k gets the negative
        dlam_dphi = -2.0 * lambda_bar * eps * np.sin(2.0 * (phi - phi0))
        perp = np.stack([-d[:, 1], d[:, 0]], axis=1) / l[:, None]
        f_edge = lam[:, None] * unit + dlam_dphi[:, None] * perp          # force on slot k+1 from edge k
        dLine = 0.5 * (-f_edge + f_edge[self.s_prv])                       # half: edge shared by two cells
        g_slot = (K * (A - A0))[self.s_cid, None] * dA + (gamma_bar * perim)[self.s_cid, None] * dP + dLine
        grad = np.zeros_like(self.pos)
        np.add.at(grad, self.s_vid, g_slot)
        return E, grad

    # ---- topology
    def vertex_directions(self) -> Dict[int, List[np.ndarray]]:
        dirs: Dict[int, List[np.ndarray]] = {}
        P = self.slot_positions()
        d = P[self.s_nxt] - P
        for i in range(len(self.s_vid)):
            v1, v2 = self.s_vid[i], self.s_vid[self.s_nxt[i]]
            for v, vec in ((v1, d[i]), (v2, -d[i])):
                lst = dirs.setdefault(int(v), [])
                if not any(np.linalg.norm(vec - u) < 1e-9 for u in lst):
                    lst.append(vec)
        return dirs

    def junction_angles(self) -> Tuple[List[List[float]], Dict[int, int]]:
        """Per-vertex interior angles (deg, chord definition: straight edges) and the valence histogram."""
        per_vertex, valence = [], {}
        for v, vecs in self.vertex_directions().items():
            valence[len(vecs)] = valence.get(len(vecs), 0) + 1
            if len(vecs) >= 3:
                per_vertex.append(angles_from_directions([math.atan2(u[1], u[0]) for u in vecs]))
        return per_vertex, valence

    def polygon_classes(self) -> Dict[int, int]:
        out: Dict[int, int] = {}
        for cell in self.cells:
            out[len(cell)] = out.get(len(cell), 0) + 1
        return out

    def min_edge_length(self) -> float:
        _, l, _ = self.edge_table()
        return float(l.min())

    # ---- T1 exchange
    def t1_shortest_edge(self, l_t1: float, l_new: float) -> bool:
        """Perform one T1 on the shortest edge below l_t1 (if any). Returns True if an exchange happened."""
        _, l, _ = self.edge_table()
        order = np.argsort(l)
        for i in order:
            if l[i] >= l_t1:
                return False
            v1, v2 = int(self.s_vid[i]), int(self.s_vid[self.s_nxt[i]])
            if v1 == v2:
                continue
            both = [c for c, cell in enumerate(self.cells) if any(v == v1 for v, _ in cell) and any(v == v2 for v, _ in cell)]
            only1 = [c for c, cell in enumerate(self.cells) if any(v == v1 for v, _ in cell) and not any(v == v2 for v, _ in cell)]
            only2 = [c for c, cell in enumerate(self.cells) if any(v == v2 for v, _ in cell) and not any(v == v1 for v, _ in cell)]
            if len(both) != 2 or len(only1) != 1 or len(only2) != 1:
                continue
            A = int(self.s_cid[i]); B = [c for c in both if c != A][0]; C = only1[0]; D = only2[0]
            if len(self.cells[A]) <= 3 or len(self.cells[B]) <= 3:
                continue                                                   # no triangles below 3 sides
            offA = {v: o for v, o in self.cells[A]}
            P1, P2 = self.pos[v1] + offA[v1], self.pos[v2] + offA[v2]
            mid = 0.5 * (P1 + P2)
            dvec = P2 - P1
            n = np.array([-dvec[1], dvec[0]]) / max(np.linalg.norm(dvec), 1e-12)
            cA = self.cell_polygon(A).mean(axis=0)
            if np.dot(n, cA - mid) < 0:
                n = -n
            P1n, P2n = mid + n * (l_new / 2), mid - n * (l_new / 2)
            self.pos[v1] = P1n - offA[v1]
            self.pos[v2] = P2n - offA[v2]
            offC1 = [o for v, o in self.cells[C] if v == v1][0]
            offD2 = [o for v, o in self.cells[D] if v == v2][0]
            idsA = {v for v, _ in self.cells[A]}
            idsB = {v for v, _ in self.cells[B]}

            def insert_next_to(cell_list, anchor, new_entry, other_ids):
                # the new vertex goes between `anchor` and the anchor's neighbour that lies on the boundary
                # with the cell that is about to lose `anchor` (combinatorial T1; no geometric re-sorting)
                k = [j for j, (v, _) in enumerate(cell_list) if v == anchor][0]
                n = len(cell_list)
                nxt, prv = cell_list[(k + 1) % n][0], cell_list[(k - 1) % n][0]
                if nxt in other_ids and prv not in other_ids:
                    cell_list.insert(k + 1, new_entry)
                elif prv in other_ids and nxt not in other_ids:
                    cell_list.insert(k, new_entry)
                else:
                    return False
                return True

            newC = list(self.cells[C]); newD = list(self.cells[D])
            okC = insert_next_to(newC, v1, (v2, offA[v2] + (offC1 - offA[v1])), idsB)
            okD = insert_next_to(newD, v2, (v1, offA[v1] + (offD2 - offA[v2])), idsA)
            if not (okC and okD):
                self.pos[v1] = P1 - offA[v1]
                self.pos[v2] = P2 - offA[v2]
                continue
            self.cells[A] = [(v, o) for v, o in self.cells[A] if v != v2]
            self.cells[B] = [(v, o) for v, o in self.cells[B] if v != v1]
            self.cells[C], self.cells[D] = newC, newD
            self._build()
            return True
        return False


def periodic_voronoi_network(n_cells: int = N_CELLS, seed: int = SEED) -> PeriodicNetwork:
    """Poisson-Voronoi seed tessellation on a periodic square of side sqrt(n) (so <A> = 1)."""
    from scipy.spatial import Voronoi

    rng = np.random.default_rng(seed)
    L = math.sqrt(n_cells)
    pts = rng.uniform(0, L, size=(n_cells, 2))
    shifts = np.array([[i, j] for i in (-1, 0, 1) for j in (-1, 0, 1)], float) * L
    tiled = np.concatenate([pts + s for s in shifts])
    vor = Voronoi(tiled)
    centre = 4
    vid: Dict[Tuple[int, int], int] = {}
    pos: List[np.ndarray] = []
    cells: List[List[Tuple[int, np.ndarray]]] = []
    for i in range(n_cells):
        region = vor.regions[vor.point_region[centre * n_cells + i]]
        verts = vor.vertices[region]
        c = pts[i]
        order = np.argsort(np.arctan2(verts[:, 1] - c[1], verts[:, 0] - c[0]))
        cell = []
        for vtx in verts[order]:
            wrapped = np.mod(vtx, L)
            off = vtx - wrapped
            key = (int(round(wrapped[0] * 1e6)), int(round(wrapped[1] * 1e6)))
            if key not in vid:
                vid[key] = len(pos)
                pos.append(wrapped)
            cell.append((vid[key], off))
        cells.append(cell)
    return PeriodicNetwork(np.array(pos), cells, L)


def relax_froth(net: PeriodicNetwork, steps: int = RELAX_STEPS, dt: float = RELAX_DT, record_every: int = 0,
                t1: bool = True, grad_tol: float = 3e-3, **kw) -> Tuple[PeriodicNetwork, Dict[str, object]]:
    """Over-damped gradient descent on the vertex energy with T1 exchanges on collapsing edges."""
    energies, snapshots, n_t1 = [], [], 0
    gmax = float("nan")
    for s in range(steps):
        E, g = net.energy_gradient(**kw)
        energies.append(E)
        gmax = float(np.abs(g).max())
        if record_every and s % record_every == 0:
            snapshots.append(net.copy())
        if gmax < grad_tol:
            break
        net.pos = net.pos - dt * g
        if t1:
            for _ in range(4):
                if not net.t1_shortest_edge(T1_LENGTH, 1.5 * T1_LENGTH):
                    break
                n_t1 += 1
    E, g = net.energy_gradient(**kw)
    energies.append(E)
    snapshots.append(net.copy())
    return net, {"energies": energies, "snapshots": snapshots, "n_t1": n_t1, "steps_run": len(energies) - 1,
                 "final_max_grad": float(np.abs(g).max()), "converged": bool(np.abs(g).max() < grad_tol),
                 "min_edge": net.min_edge_length()}


def brick_lattice_control(nx: int = 8, ny: int = 8, stagger: float = 0.5) -> List[List[float]]:
    """Analytic control: a running-bond brick wall has three-fold vertices with angles (90, 90, 180);
    stagger = 0 is the square lattice whose vertices are four-fold (90 x 4). Returns per-vertex angles."""
    out = []
    if stagger == 0.0:
        return [[90.0, 90.0, 90.0, 90.0] for _ in range((nx - 1) * (ny - 1))]
    for _ in range(nx * (ny - 1)):
        out.append([90.0, 90.0, 180.0])
    return out


# =============================================================================== specimen segmentation
def load_rgb(path: Path) -> np.ndarray:
    from PIL import Image
    return np.asarray(Image.open(path).convert("RGB")).astype(float) / 255.0


def _drop_small_components(mask: np.ndarray, min_px: int) -> np.ndarray:
    """Remove connected components with fewer than min_px pixels (version-independent helper)."""
    from skimage import measure
    lab = measure.label(mask, connectivity=1)
    areas = np.bincount(lab.ravel())
    keep = areas >= min_px
    keep[0] = False
    return keep[lab]


def _robust_cell_area(areas: np.ndarray) -> float:
    """Median area of the components that make up 95 % of the non-wall area (ignores specks)."""
    a = np.sort(areas[areas > 0])[::-1]
    if a.size == 0:
        return 0.0
    cum = np.cumsum(a) / a.sum()
    cells = a[cum <= 0.95]
    return float(np.median(cells if cells.size else a))


def components_segmentation(wall: np.ndarray, fov: np.ndarray, min_area_frac: float = 0.3,
                            drop_page_regions: bool = False) -> np.ndarray:
    """Cells = connected components of non-wall pixels; fragments smaller than min_area_frac x the robust
    median cell area are returned to the wall; labels are then grown into the wall pixels (distance watershed).
    With drop_page_regions (line drawings) components that touch the image border AND span more than 60 % of
    the image width or height (the page background / the bands above and below a drawn strip) are removed."""
    from scipy import ndimage as ndi
    from skimage import measure, segmentation

    H, W = wall.shape
    lab = measure.label(~wall & fov, connectivity=1)
    if drop_page_regions:
        edge_ids = set(np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))) - {0}
        for p in measure.regionprops(lab):
            if p.label in edge_ids:
                y0, x0, y1, x1 = p.bbox
                if (x1 - x0) > 0.6 * W or (y1 - y0) > 0.6 * H:
                    lab[lab == p.label] = 0
    areas = np.bincount(lab.ravel()).astype(float)
    areas[0] = 0
    med = _robust_cell_area(areas)
    small = np.where((areas > 0) & (areas < min_area_frac * med))[0]
    lab[np.isin(lab, small)] = 0
    lab, _, _ = segmentation.relabel_sequential(lab)
    dist = ndi.distance_transform_edt(lab == 0)
    return segmentation.watershed(dist, lab, mask=fov)


def segment_specimen(im: np.ndarray, kind: str) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], dict]:
    """Return (labels, field-of-view mask, wall/ink mask or None, notes) for one specimen kind."""
    from scipy import ndimage as ndi
    from skimage import filters, measure, morphology, segmentation

    notes: dict = {"kind": kind}
    if kind == "cornea":
        ch = im.max(axis=2)                                   # blue nuclei -> bright; dark borders stay dark
        fov = np.ones(ch.shape, bool)
        blue = im[:, :, 2] - 0.5 * (im[:, :, 0] + im[:, :, 1])
        nuc = _drop_small_components(morphology.opening(blue > 0.18, morphology.disk(2)), 60)
        markers = measure.label(nuc)
        ridge = filters.sato(1.0 - ch, sigmas=[2, 3, 4], black_ridges=False)
        lab = segmentation.watershed(filters.gaussian(ridge, 1), markers, mask=fov)
        notes.update(method="one nucleus marker per cell (blue excess > 0.18) + watershed on Sato ridge of dark borders",
                     n_markers=int(markers.max()))
        return lab, fov, None, notes
    if kind == "onion":
        ch = im[:, :, 1]                                      # magenta walls are dark in green
        fov = ndi.binary_erosion(im.mean(axis=2) > 0.25, iterations=30)
        ridge = filters.gaussian(filters.sato(1.0 - ch, sigmas=[3, 4, 5, 6], black_ridges=False), 1)
        wall = ridge > 0.6 * filters.threshold_otsu(ridge[fov])
        wall = morphology.closing(wall, morphology.disk(2))
        lab = components_segmentation(wall, fov, 0.3)
        notes.update(method="Sato ridge wall mask (green channel) -> connected components -> grow into walls",
                     n_markers=int(lab.max()))
        return lab, fov, wall, notes
    # line drawing: `kind` is "drawing" or "drawing:<closing_px>" (Fig. 103 draws every wall as a double line
    # with a white channel between; a 9-px closing merges the pair into one solid wall)
    closing_px = int(kind.split(":")[1]) if ":" in kind else 2
    ch = im.mean(axis=2)
    fov = np.ones(ch.shape, bool)
    ink = ch < filters.threshold_otsu(ch)
    wall = morphology.closing(ink, morphology.disk(closing_px))
    wall[:2, :] = wall[-2:, :] = True                      # 2-px frame closes the cells cut by the figure edge
    wall[:, :2] = wall[:, -2:] = True
    lab = components_segmentation(wall, fov, 0.3, drop_page_regions=True)
    notes.update(method=f"Otsu ink, {closing_px}-px closing, 2-px frame -> connected components (page/band regions and "
                        "bourrelet fragments removed) -> grow into ink", n_markers=int(lab.max()))
    return lab, fov, ink, notes


def find_vertices(lab: np.ndarray, fov: np.ndarray, merge_px: float = 6.0,
                  max_ink_dist: Optional[float] = None, ink: Optional[np.ndarray] = None) -> List[dict]:
    """Vertices = connected clusters of 2x2 windows containing >= 3 labels; clusters with identical label
    sets closer than merge_px are merged (they are one physical vertex split by pixelisation)."""
    from scipy import ndimage as ndi
    from skimage import measure

    H, W = lab.shape
    a = lab[:-1, :-1]; b = lab[:-1, 1:]; c = lab[1:, :-1]; d = lab[1:, 1:]
    ndist = 1 + (b != a) + ((c != a) & (c != b)) + ((d != a) & (d != b) & (d != c))
    jl = measure.label(ndist >= 3, connectivity=2)
    dist_ink = ndi.distance_transform_edt(~ink) if (ink is not None and max_ink_dist) else None
    raw = []
    for p in measure.regionprops(jl):
        cy, cx = p.centroid
        cy += 0.5; cx += 0.5
        iy, ix = int(cy), int(cx)
        y0, y1, x0, x1 = max(0, iy - 2), min(H, iy + 3), max(0, ix - 2), min(W, ix + 3)
        labs = set(np.unique(lab[y0:y1, x0:x1]).tolist()) - {0}
        if len(labs) < 3 or not fov[y0:y1, x0:x1].all():
            continue
        if dist_ink is not None and dist_ink[min(H - 1, iy), min(W - 1, ix)] > max_ink_dist:
            continue
        raw.append({"x": cx, "y": cy, "labels": frozenset(labs)})
    merged: List[dict] = []
    for v in raw:
        for m in merged:
            if m["labels"] == v["labels"] and math.hypot(m["x"] - v["x"], m["y"] - v["y"]) < merge_px:
                m["x"] = 0.5 * (m["x"] + v["x"]); m["y"] = 0.5 * (m["y"] + v["y"])
                break
        else:
            merged.append(dict(v))
    return merged


def chord_angles(verts: List[dict]) -> Tuple[List[dict], float]:
    """Chord estimator: for each vertex, the neighbour along each label-pair boundary is the nearest other
    vertex sharing that pair; angles between the chords. Returns (scored vertices, median chord length)."""
    pair_index = defaultdict(list)
    for i, v in enumerate(verts):
        L = sorted(v["labels"])
        for p in range(len(L)):
            for q in range(p + 1, len(L)):
                pair_index[(L[p], L[q])].append(i)
    out, chord_lengths = [], []
    for i, v in enumerate(verts):
        L = sorted(v["labels"])
        dirs, ok = [], True
        for p in range(len(L)):
            for q in range(p + 1, len(L)):
                cands = [j for j in pair_index[(L[p], L[q])] if j != i]
                if not cands:
                    ok = False
                    break
                j = min(cands, key=lambda j: (verts[j]["x"] - v["x"]) ** 2 + (verts[j]["y"] - v["y"]) ** 2)
                dx, dy = verts[j]["x"] - v["x"], verts[j]["y"] - v["y"]
                if math.hypot(dx, dy) < 1e-6:
                    ok = False
                    break
                dirs.append(math.atan2(dy, dx))
                chord_lengths.append(math.hypot(dx, dy))
            if not ok:
                break
        if not ok or len(dirs) != len(L) * (len(L) - 1) // 2 or len(dirs) < 3:
            continue
        out.append({"x": v["x"], "y": v["y"], "valence": len(L), "angles": angles_from_directions(dirs)})
    med_chord = float(np.median(chord_lengths)) if chord_lengths else float("nan")
    return out, med_chord


def branch_angles(lab: np.ndarray, fov: np.ndarray, verts: List[dict], R: float) -> List[dict]:
    """Branch estimator: direction of each wall = mean offset (from the vertex) of the boundary pixels between
    the corresponding label pair within radius R (core r < 2 px excluded)."""
    H, W = lab.shape
    out = []
    for v in verts:
        cy, cx = v["y"], v["x"]
        if cy < R + 2 or cx < R + 2 or cy > H - R - 3 or cx > W - R - 3:
            continue
        y0, y1, x0, x1 = int(cy - R), int(cy + R + 1), int(cx - R), int(cx + R + 1)
        sub = lab[y0:y1, x0:x1]
        if not fov[y0:y1, x0:x1].all() or (sub == 0).any():
            continue
        yy, xx = np.mgrid[y0:y1, x0:x1]
        ys, xs = yy + 0.5 - cy, xx + 0.5 - cx
        r2 = ys ** 2 + xs ** 2
        inside = (r2 <= R ** 2) & (r2 > 4.0)
        L = sorted(v["labels"])
        dirs = []
        for p in range(len(L)):
            for q in range(p + 1, len(L)):
                mA, mB = sub == L[p], sub == L[q]
                adj = np.zeros_like(mA)
                adj[:-1, :] |= mA[:-1, :] & mB[1:, :]; adj[1:, :] |= mA[1:, :] & mB[:-1, :]
                adj[:, :-1] |= mA[:, :-1] & mB[:, 1:]; adj[:, 1:] |= mA[:, 1:] & mB[:, :-1]
                adj &= inside
                if adj.sum() < 3:
                    continue
                vx, vy = xs[adj].mean(), ys[adj].mean()
                if math.hypot(vx, vy) > 1e-6:
                    dirs.append(math.atan2(vy, vx))
        if len(dirs) != len(L) * (len(L) - 1) // 2 or len(dirs) < 3:
            continue
        out.append({"x": cx, "y": cy, "valence": len(L), "angles": angles_from_directions(dirs)})
    return out


def skeleton_graph_analysis(wall: np.ndarray, spur_px: float = 12.0, cluster_px: int = 3,
                            short_edge_frac: float = 0.15) -> dict:
    """Line-drawing pipeline: skeletonise the (closed) ink, cluster skeleton pixels with >= 3 neighbours into
    junction clusters (vertices), label the skeleton branches between clusters, prune dangling spurs shorter
    than spur_px, and return per-vertex valence, chord angles (to the neighbouring junction along each branch)
    and branch-direction angles (mean offset of the first R px of each branch, R = median chord / 3)."""
    from scipy import ndimage as ndi
    from skimage import measure, morphology

    # fill enclosed holes smaller than 1 % of the image (Thompson's bourrelet triangles at the angles, p. 298)
    wall = ~_drop_small_components(~wall, int(0.01 * wall.size))
    skel = morphology.skeletonize(wall)
    nb = ndi.convolve(skel.astype(int), np.ones((3, 3), int), mode="constant") - skel.astype(int)
    jpx = skel & (nb >= 3)
    jlab = measure.label(morphology.dilation(jpx, morphology.disk(cluster_px)), connectivity=2)
    branches = skel & (jlab == 0)
    blab = measure.label(branches, connectivity=2)
    yy, xx = np.mgrid[0:skel.shape[0], 0:skel.shape[1]]
    # junction centroids from the junction pixels themselves
    cents = {}
    for j in range(1, jlab.max() + 1):
        m = jpx & (jlab == j)
        if not m.any():
            m = (jlab == j) & skel
        cents[j] = (float(xx[m].mean()), float(yy[m].mean()))
    # branch -> adjacent junction clusters, pixel lists
    adj: Dict[int, set] = {}
    bpix: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
    for b in range(1, blab.max() + 1):
        m = blab == b
        ring = morphology.dilation(m, morphology.disk(2)) & (jlab > 0)
        adj[b] = set(np.unique(jlab[ring]).tolist()) - {0}
        bpix[b] = (xx[m], yy[m])
    # prune short dangling spurs
    for b in list(adj):
        if len(adj[b]) <= 1 and bpix[b][0].size < spur_px:
            del adj[b]
    # collapse very short junction-to-junction branches (one physical vertex split by the thick-wall skeleton)
    lengths = [bpix[b][0].size for b, js in adj.items() if len(js) == 2]
    med_len = float(np.median(lengths)) if lengths else 0.0
    parent = {j: j for j in cents}

    def find(j):
        while parent[j] != j:
            parent[j] = parent[parent[j]]
            j = parent[j]
        return j

    for b in list(adj):
        js = sorted(adj[b])
        if len(js) == 2 and bpix[b][0].size < short_edge_frac * med_len:
            parent[find(js[1])] = find(js[0])
            del adj[b]
    groups: Dict[int, List[int]] = defaultdict(list)
    for j in cents:
        groups[find(j)].append(j)
    for root, members in groups.items():
        if len(members) > 1:
            cents[root] = (float(np.mean([cents[j][0] for j in members])), float(np.mean([cents[j][1] for j in members])))
    adj = {b: {find(j) for j in js} for b, js in adj.items()}
    j_branches: Dict[int, List[int]] = defaultdict(list)
    for b, js in adj.items():
        for j in js:
            j_branches[j].append(b)
    verts = []
    chord_lengths = []
    for j, bl in j_branches.items():
        cx, cy = cents[j]
        if len(bl) < 3:
            continue
        chords = []
        for b in bl:
            others = adj[b] - {j}
            if len(others) == 1:
                ox, oy = cents[next(iter(others))]
                chords.append(math.atan2(oy - cy, ox - cx))
                chord_lengths.append(math.hypot(ox - cx, oy - cy))
        verts.append({"x": cx, "y": cy, "valence": len(bl), "branches": bl, "chord_dirs": chords})
    med_chord = float(np.median(chord_lengths)) if chord_lengths else float("nan")
    R = max(6.0, round(med_chord / 3.0)) if chord_lengths else 10.0
    chord_out, branch_out = [], []
    for v in verts:
        if len(v["chord_dirs"]) == v["valence"]:
            chord_out.append({"x": v["x"], "y": v["y"], "valence": v["valence"], "angles": angles_from_directions(v["chord_dirs"])})
        dirs = []
        for b in v["branches"]:
            px, py = bpix[b]
            d2 = (px + 0.5 - v["x"]) ** 2 + (py + 0.5 - v["y"]) ** 2
            sel = (d2 <= R ** 2) & (d2 > 4.0)
            if sel.sum() >= 2:
                vx, vy = (px[sel] + 0.5 - v["x"]).mean(), (py[sel] + 0.5 - v["y"]).mean()
                if math.hypot(vx, vy) > 1e-6:
                    dirs.append(math.atan2(vy, vx))
        if len(dirs) == v["valence"] and len(dirs) >= 3:
            branch_out.append({"x": v["x"], "y": v["y"], "valence": v["valence"], "angles": angles_from_directions(dirs)})
    return {"vertices": verts, "chord": chord_out, "branch": branch_out, "median_chord_px": med_chord, "R": R,
            "skeleton": skel, "n_branches": len(adj)}


def polygon_class_distribution(lab: np.ndarray, verts: List[dict]) -> Dict[int, int]:
    """Sides per cell = number of vertices whose label set contains the cell (interior cells only:
    cells touching the image border are skipped)."""
    H, W = lab.shape
    border = set(np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]])).tolist())
    count: Dict[int, int] = defaultdict(int)
    for v in verts:
        for l in v["labels"]:
            count[l] += 1
    out: Dict[int, int] = defaultdict(int)
    for l, n in count.items():
        if l in border or n < 3:
            continue
        out[n] += 1
    return dict(sorted(out.items()))


def analyse_specimen(key: str, n_boot: int, seed: int, tol: float) -> dict:
    fname, kind, label, regime, flag = SPECIMENS[key]
    path = DATA_DIR / fname
    im = load_rgb(path)
    lab, fov, ink, notes = segment_specimen(im, kind)
    is_drawing = kind.startswith("drawing")
    if is_drawing:
        # Thompson's drawings: partial cells are open to the page, so vertices come from the skeleton graph
        from skimage import filters, morphology
        closing_px = int(kind.split(":")[1]) if ":" in kind else 2
        wall = morphology.closing(ink, morphology.disk(closing_px))
        sk = skeleton_graph_analysis(wall)
        verts = [{"x": v["x"], "y": v["y"], "labels": frozenset(range(v["valence"]))} for v in sk["vertices"]]
        chord, med_chord, R, branch = sk["chord"], sk["median_chord_px"], sk["R"], sk["branch"]
        notes.update(vertex_method="skeleton graph of the closed ink (junction clusters, branch tracing, spur pruning)",
                     n_branches=sk["n_branches"])
        valence_list = [v["valence"] for v in sk["vertices"]]
        lab_for_plot = sk["skeleton"].astype(int)
        polygon_classes = {}
    else:
        verts = find_vertices(lab, fov, ink=ink)
        chord, med_chord = chord_angles(verts)
        R = max(6.0, round(med_chord / 3.0))
        branch = branch_angles(lab, fov, verts, R)
        notes.update(vertex_method="label image: 2x2 windows with >= 3 labels, chords to neighbouring vertices")
        valence_list = [len(v["labels"]) for v in verts]
        lab_for_plot = lab
        polygon_classes = polygon_class_distribution(lab, verts)
    valence_all = defaultdict(int)
    for k_ in valence_list:
        valence_all[k_] += 1
    n_all = len(valence_list)
    f3 = valence_all.get(3, 0) / n_all if n_all else float("nan")
    res = {"key": key, "file": str(path.relative_to(ROOT)), "label": label, "regime": regime, "analogue_flag": flag,
           "segmentation": notes, "n_cells": int(lab.max()) if not is_drawing else None, "n_vertices_detected": n_all,
           "valence_histogram": dict(sorted(valence_all.items())), "three_fold_fraction": f3,
           "median_chord_length_px": med_chord, "branch_radius_px": R,
           "polygon_classes": polygon_classes}
    for name, scored in (("chord", chord), ("branch", branch)):
        v3 = [s["angles"] for s in scored if s["valence"] == 3]
        pooled = np.concatenate(v3) if v3 else np.array([])
        st = plateau_statistics(pooled, tol)
        st["n_vertices_scored"] = len(v3)
        st["bootstrap_95ci"] = bootstrap_vertex_ci(v3, n_boot, seed)
        if pooled.size:
            # inferred tension ratios (inverse Lami): spread of T_i / <T> across vertices
            ratios = np.array([tensions_from_angles(*a) for a in v3])
            st["inferred_tension_ratio_sd"] = float(ratios.std())
            st["inferred_tension_ratio_range_median"] = float(np.median(ratios.max(axis=1) - ratios.min(axis=1)))
        res[name] = st
        res[name + "_vertex_angles"] = [[round(x, 3) for x in a] for a in v3]
    # sensitivity of the branch estimator to R (micrographs only; drawings use the fixed skeleton R)
    sens = {}
    if not is_drawing:
        for fac in (0.25, 1 / 3, 0.5):
            Rf = max(6.0, round(med_chord * fac))
            b = branch_angles(lab, fov, verts, Rf)
            pooled = np.concatenate([s["angles"] for s in b if s["valence"] == 3]) if b else np.array([])
            sens[f"R={Rf:.0f}px (L_e x {fac:.2f})"] = plateau_statistics(pooled, tol)
    res["branch_radius_sensitivity"] = sens
    res["_lab"] = lab_for_plot
    res["_chord"] = chord
    res["_im"] = im
    return res


# =============================================================================== model runs
def run_model_suite(eps_sweep: Sequence[float], lambda_bar: float, gamma_bar: float, seed: int, steps: int,
                    n_cells: int, tol: float, record_gif: bool = True) -> dict:
    seed_net = periodic_voronoi_network(n_cells, seed)
    vor_pv, vor_val = seed_net.junction_angles()
    out = {"poisson_voronoi": {"stats": plateau_statistics(np.concatenate(vor_pv), tol), "valence": vor_val,
                               "vertex_angles": vor_pv, "polygon_classes": seed_net.polygon_classes()}}
    # Plateau limit (eps = 0, Gamma = 0) with GIF snapshots
    net, info = relax_froth(seed_net.copy(), steps=steps, record_every=max(1, steps // 60) if record_gif else 0,
                            lambda_bar=lambda_bar, gamma_bar=gamma_bar, eps=0.0)
    pv, val = net.junction_angles()
    out["plateau_froth"] = {"stats": plateau_statistics(np.concatenate(pv), tol), "valence": val, "vertex_angles": pv,
                            "polygon_classes": net.polygon_classes(), "n_t1": info["n_t1"], "steps_run": info["steps_run"],
                            "final_max_grad": info["final_max_grad"], "converged": info["converged"],
                            "min_edge": info["min_edge"], "energy_start": info["energies"][0], "energy_end": info["energies"][-1]}
    out["_snapshots"] = info["snapshots"]
    out["_seed_cells"] = seed_net
    out["_plateau_net"] = net
    # Farhadifar Case I (active-mechanics comparator)
    netF, infoF = relax_froth(seed_net.copy(), steps=steps, lambda_bar=LAMBDA_BAR, gamma_bar=GAMMA_BAR_FARHADIFAR, eps=0.0)
    pvF, valF = netF.junction_angles()
    out["farhadifar_case1"] = {"stats": plateau_statistics(np.concatenate(pvF), tol), "valence": valF, "vertex_angles": pvF,
                               "polygon_classes": netF.polygon_classes(), "n_t1": infoF["n_t1"], "converged": infoF["converged"]}
    # anisotropy sweep
    sweep = []
    for eps in eps_sweep:
        n_e, inf_e = relax_froth(seed_net.copy(), steps=steps, lambda_bar=lambda_bar, gamma_bar=gamma_bar, eps=float(eps))
        pv_e, val_e = n_e.junction_angles()
        st = plateau_statistics(np.concatenate(pv_e), tol)
        sweep.append({"eps": float(eps), "stats": st, "valence": val_e, "n_t1": inf_e["n_t1"], "converged": inf_e["converged"],
                      "vertex_angles": pv_e})
    out["eps_sweep"] = sweep
    # Lami partition-tension sweep (E1)
    ratios = np.linspace(0.0, 2.0, 201)
    lami = []
    for r in ratios:
        a = lami_angles_from_tensions(1.0, 1.0, max(float(r), 1e-9))
        lami.append({"t_over_T": float(r), "alpha": a[0] if a else None, "gamma": a[2] if a else None})
    out["lami_sweep"] = lami
    # controls
    brick = brick_lattice_control(stagger=0.5)
    square = brick_lattice_control(stagger=0.0)
    out["brick_control"] = {"stats": plateau_statistics(np.concatenate(brick), tol), "three_fold_fraction": 1.0}
    out["square_control"] = {"stats": plateau_statistics(np.concatenate(square), tol), "three_fold_fraction": 0.0}
    return out


def goodness_of_fit(specimen: dict, model: dict, tol: float) -> dict:
    """Fit eps by KS distance between the specimen chord angles and the eps-sweep froths; GoF vs references."""
    bins = np.arange(0.0, 360.1, 5.0)
    spec = np.concatenate(specimen["chord_vertex_angles"]) if specimen["chord_vertex_angles"] else np.array([])
    res = {}
    if spec.size < 3:
        return {"note": "too few scored vertices"}
    for name in ("plateau_froth", "poisson_voronoi", "farhadifar_case1"):
        ref = np.concatenate(model[name]["vertex_angles"])
        D, p = two_sample_ks(spec, ref)
        r, rmse = histogram_fit(spec, ref, bins)
        res[name] = {"ks_D": D, "ks_p": p, "hist_pearson_r": r, "hist_rmse": rmse}
    best = None
    for e in model["eps_sweep"]:
        ref = np.concatenate(e["vertex_angles"])
        D, p = two_sample_ks(spec, ref)
        r, rmse = histogram_fit(spec, ref, bins)
        rec = {"eps": e["eps"], "ks_D": D, "ks_p": p, "hist_pearson_r": r, "hist_rmse": rmse, "model_sd_deg": e["stats"]["sd_deg"]}
        if best is None or D < best["ks_D"]:
            best = rec
    res["best_eps_fit"] = best
    res["controls"] = {"brick_median_abs_dev_deg": model["brick_control"]["stats"]["median_abs_dev_deg"],
                       "square_three_fold_fraction": model["square_control"]["three_fold_fraction"]}
    return res


# =============================================================================== verdict
def decide_verdict(spec_results: Dict[str, dict], model: dict, tol: float) -> dict:
    prim = spec_results["human_corneal_endothelium"]
    sec = spec_results["allium_cepa_epidermis"]
    c = prim["chord"]
    three_fold_ok = prim["three_fold_fraction"] >= F3_THRESHOLD and sec["three_fold_fraction"] >= F3_THRESHOLD
    angle_ok = c["median_abs_dev_deg"] <= tol
    ci = c["bootstrap_95ci"].get("median_abs_dev_deg", (float("nan"), float("nan")))
    below_voronoi = c["sd_deg"] < model["poisson_voronoi"]["stats"]["sd_deg"]
    if three_fold_ok and angle_ok and below_voronoi:
        verdict = "supported"
    elif three_fold_ok and (angle_ok or below_voronoi):
        verdict = "mixed"
    elif three_fold_ok:
        verdict = "mixed"
    else:
        verdict = "refuted"
    b = prim["branch"]
    branch_ok = b["median_abs_dev_deg"] <= tol
    return {"verdict": verdict, "three_fold_law_passes": bool(three_fold_ok), "angle_tolerance_passes": bool(angle_ok),
            "dispersion_below_poisson_voronoi_null": bool(below_voronoi),
            "primary_median_abs_dev_deg": c["median_abs_dev_deg"], "primary_median_abs_dev_95ci": ci,
            "robustness": {
                "branch_estimator_median_abs_dev_deg": b["median_abs_dev_deg"],
                "branch_estimator_passes_tolerance": bool(branch_ok),
                "verdict_under_branch_estimator": verdict if branch_ok else ("mixed" if three_fold_ok else "refuted"),
                "note": "The branch estimator follows the wall within R = L_e/3 of the vertex and is inflated by wall "
                        "thickness / segmentation jitter; the chord estimator (straight edges between vertices) is the "
                        "pre-declared primary. Both are reported; the straight-edge froth model itself has an irreducible "
                        "chord-angle SD of ~10 deg because a straight-sided pentagon cannot have three 120-degree angles."},
            "secondary_specimen": {"key": "allium_cepa_epidermis", "f3": sec["three_fold_fraction"],
                                   "chord_median_abs_dev_deg": sec["chord"]["median_abs_dev_deg"],
                                   "angle_tolerance_passes": bool(sec["chord"]["median_abs_dev_deg"] <= tol),
                                   "reading": "Thompson's Fig. 124 regime: three-fold meeting holds, angles are spread by growth anisotropy"},
            "criteria": {"f3_threshold": F3_THRESHOLD, "tolerance_deg": tol, "estimator": "chord (primary)",
                         "primary_specimen": "human_corneal_endothelium"}}


# =============================================================================== figures
def _draw_network(ax, net: PeriodicNetwork, title: str):
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection

    polys = [net.cell_polygon(c) for c in range(net.n_cells)]
    sides = np.array([len(c) for c in net.cells])
    cmap = plt.get_cmap("coolwarm")
    colors = [cmap((s - 4) / 4.0) for s in sides]
    ax.add_collection(PolyCollection(polys, facecolors=colors, edgecolors="k", linewidths=0.8, alpha=0.9))
    ax.set_xlim(0, net.L); ax.set_ylim(0, net.L); ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([]); ax.set_title(title, fontsize=8)


def render_figures(spec_results: Dict[str, dict], model: dict, gof: Dict[str, dict], verdict: dict, tol: float):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image
    from skimage import segmentation

    MEDIA_DIR.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(20, 10.5))
    gs = fig.add_gridspec(2, 4, height_ratios=[1.15, 1.0])
    keys = list(SPECIMENS)
    for i, key in enumerate(keys):
        r = spec_results[key]
        ax = fig.add_subplot(gs[0, i])
        im = r["_im"]
        ax.imshow(im)
        bnd = (r["_lab"] > 0) if r["n_cells"] is None else segmentation.find_boundaries(r["_lab"], mode="inner")
        ov = np.zeros((*bnd.shape, 4)); ov[bnd] = (1.0, 0.95, 0.0, 0.9)
        ax.imshow(ov)
        for v in r["_chord"]:
            ax.plot(v["x"], v["y"], "o", ms=3, mfc="red" if v["valence"] == 3 else "cyan", mec="k", mew=0.3)
        c = r["chord"]
        ax.set_title(f"{r['label']}\n{r['regime']}\nN cells={r['n_cells'] if r['n_cells'] is not None else 'n/a (skeleton)'}, vertices={r['n_vertices_detected']}, "
                     f"f3={r['three_fold_fraction']:.2f}, median|θ−120|={c['median_abs_dev_deg']:.1f}°, "
                     f"within ±{tol:.0f}°={c['frac_within_tol']:.2f}", fontsize=7.5)
        ax.set_xticks([]); ax.set_yticks([])
    # (e) angle distributions
    ax = fig.add_subplot(gs[1, 0])
    bins = np.arange(45, 196, 5)
    def pooled(pv): return np.concatenate(pv) if pv else np.array([])
    series = [
        (pooled(spec_results["human_corneal_endothelium"]["chord_vertex_angles"]), "tab:red", "corneal endothelium (chord)"),
        (pooled(spec_results["allium_cepa_epidermis"]["chord_vertex_angles"]), "tab:purple", "onion epidermis (chord)"),
        (pooled(model["plateau_froth"]["vertex_angles"]), "tab:blue", "Plateau froth (Γ=ε=0, T1 on)"),
        (pooled(model["poisson_voronoi"]["vertex_angles"]), "grey", "Poisson–Voronoi null"),
    ]
    for a, col, lab in series:
        st = plateau_statistics(a, tol)
        ax.hist(a, bins=bins, density=True, histtype="step", lw=1.8, color=col,
                label=f"{lab}  SD={st['sd_deg']:.1f}°, med|dev|={st['median_abs_dev_deg']:.1f}°")
    ax.axvline(120, color="k", ls="--", lw=0.8)
    ax.axvspan(120 - tol, 120 + tol, color="gold", alpha=0.15)
    ax.set_xlabel("tri-junction angle (deg)"); ax.set_ylabel("density")
    ax.set_title("E3  chord-angle distributions (mean ≡ 120°; dispersion is the test)", fontsize=8.5)
    ax.legend(fontsize=6.5, loc="upper left")
    # (f) eps sweep
    ax = fig.add_subplot(gs[1, 1])
    eps = [e["eps"] for e in model["eps_sweep"]]
    ax.plot(eps, [e["stats"]["sd_deg"] for e in model["eps_sweep"]], "o-", color="tab:blue", label="froth model SD(θ)")
    ax.plot(eps, [e["stats"]["median_abs_dev_deg"] for e in model["eps_sweep"]], "s--", color="tab:cyan", label="froth model median|θ−120|")
    for key, col in (("human_corneal_endothelium", "tab:red"), ("allium_cepa_epidermis", "tab:purple"),
                     ("thompson_fig103_maize_parenchyma", "tab:green"), ("thompson_fig124_girardia_epidermis", "tab:olive")):
        sd = spec_results[key]["chord"]["sd_deg"]
        ax.axhline(sd, color=col, lw=1.0, ls=":", label=f"{key.split('_')[0]} SD={sd:.1f}°")
    ax.axhline(model["poisson_voronoi"]["stats"]["sd_deg"], color="grey", lw=1.2, label="Poisson–Voronoi SD")
    ax.axhline(model["farhadifar_case1"]["stats"]["sd_deg"], color="tab:orange", lw=1.0, ls="-.", label="Farhadifar Case I SD")
    ax.set_xlabel("tension anisotropy ε  [Λ(φ)=Λ₀(1+ε cos 2φ)]"); ax.set_ylabel("deg")
    ax.set_title("E5  anisotropy sweep (Fig. 124 regime) vs specimens & nulls", fontsize=8.5)
    ax.legend(fontsize=6, loc="upper left", ncol=2)
    # (g) Lami curve with inferred tension spread
    ax = fig.add_subplot(gs[1, 2])
    rr = [d["t_over_T"] for d in model["lami_sweep"]]
    ax.plot(rr, [d["alpha"] if d["alpha"] is not None else np.nan for d in model["lami_sweep"]], "b-", label="α=β (opposite T, T′)")
    ax.plot(rr, [d["gamma"] if d["gamma"] is not None else np.nan for d in model["lami_sweep"]], "r-", label="γ (opposite partition t)")
    ax.axhline(120, color="k", ls="--", lw=0.8); ax.axvline(1.0, color="k", ls=":", lw=0.8)
    for r_, lab in ((1e-6, "case (1) 90°"), (0.3473, "case (2) 100°"), (1.0, "case (3) 120°")):
        a = lami_angles_from_tensions(1.0, 1.0, max(r_, 1e-6))[0]
        ax.plot([r_], [a], "ko", ms=4); ax.annotate(lab, (r_, a), textcoords="offset points", xytext=(6, -14), fontsize=7)
    c = spec_results["human_corneal_endothelium"]["chord"]
    ax.set_xlabel("partition tension ratio t / T"); ax.set_ylabel("junction angle (deg)")
    ax.set_title(f"E1  Lami / Thompson Fig. 113; cornea inverse-Lami tension-ratio SD = {c.get('inferred_tension_ratio_sd', float('nan')):.3f}", fontsize=8)
    ax.legend(fontsize=7, loc="center right")
    # (h) text summary
    ax = fig.add_subplot(gs[1, 3]); ax.axis("off")
    lines = [f"VERDICT (pre-registered criteria): {verdict['verdict'].upper()}",
             f"  three-fold law (f3 ≥ {F3_THRESHOLD:.2f}): {'pass' if verdict['three_fold_law_passes'] else 'fail'}",
             f"  cornea median|θ−120| ≤ {tol:.0f}°: {verdict['primary_median_abs_dev_deg']:.1f}° "
             f"[95% CI {verdict['primary_median_abs_dev_95ci'][0]:.1f}, {verdict['primary_median_abs_dev_95ci'][1]:.1f}] "
             f"→ {'pass' if verdict['angle_tolerance_passes'] else 'fail'}",
             f"  dispersion < Poisson–Voronoi null: {'yes' if verdict['dispersion_below_poisson_voronoi_null'] else 'no'}", ""]
    for key in keys:
        r = spec_results[key]; cc = r["chord"]; bb = r["branch"]; g = gof[key]
        lines.append(f"{key}:")
        lines.append(f"  f3={r['three_fold_fraction']:.3f} ({r['valence_histogram']}), N3={cc['n_vertices_scored']}")
        lines.append(f"  chord: SD={cc['sd_deg']:.1f}° med|dev|={cc['median_abs_dev_deg']:.1f}° in±{tol:.0f}°={cc['frac_within_tol']:.2f} CV={cc['cv']:.3f}")
        lines.append(f"  branch(R={r['branch_radius_px']:.0f}px): SD={bb['sd_deg']:.1f}° med|dev|={bb['median_abs_dev_deg']:.1f}°")
        if "best_eps_fit" in g:
            b = g["best_eps_fit"]
            lines.append(f"  KS vs Plateau D={g['plateau_froth']['ks_D']:.2f}, vs Voronoi D={g['poisson_voronoi']['ks_D']:.2f}; "
                         f"best ε={b['eps']:.1f} (D={b['ks_D']:.2f}, r={b['hist_pearson_r']:.2f}, RMSE={b['hist_rmse']:.4f})")
    pf = model["plateau_froth"]
    lines += ["", f"Plateau froth: SD={pf['stats']['sd_deg']:.2f}°, T1s={pf['n_t1']}, converged={pf['converged']}, "
                  f"max|∇E|={pf['final_max_grad']:.1e}",
              f"Poisson–Voronoi: SD={model['poisson_voronoi']['stats']['sd_deg']:.1f}°; brick control med|dev|="
              f"{model['brick_control']['stats']['median_abs_dev_deg']:.0f}°; square lattice f3=0",
              "Dryad mn8cs: DATASET_MISMATCH (Ca²⁺ ratio imaging, no junction label) — not used"]
    ax.text(0.0, 1.0, "\n".join(lines), va="top", ha="left", fontsize=6.6, family="monospace")
    fig.suptitle("Thompson (1917) T1917-018 — \"cells in contact meet three-at-a-time at ~120°\" (Pass 2): Thompson's figures, "
                 "real specimens, Plateau froth model, nulls", fontsize=11)
    fig.tight_layout()
    fig.savefig(COMPARISON_PNG, dpi=105)
    plt.close(fig)

    # GIF: Plateau-froth relaxation with T1s
    frames = []
    bins_g = np.arange(60, 181, 5)
    snaps = model["_snapshots"]
    for i, net_i in enumerate(snaps):
        pv, _ = net_i.junction_angles()
        ang = np.concatenate(pv)
        f, (a1, a2) = plt.subplots(1, 2, figsize=(8, 4))
        _draw_network(a1, net_i, f"Plateau froth relaxation (Γ=ε=0, T1 on)  frame {i}/{len(snaps) - 1}")
        a2.hist(ang, bins=bins_g, color="tab:blue", alpha=0.7)
        a2.axvline(120, color="k", ls="--"); a2.axvspan(120 - tol, 120 + tol, color="gold", alpha=0.15)
        a2.set_xlim(60, 180); a2.set_ylim(0, 300)
        a2.set_title(f"junction angles  SD = {ang.std(ddof=1):.1f}°", fontsize=9)
        f.tight_layout(); f.canvas.draw()
        frames.append(Image.fromarray(np.asarray(f.canvas.buffer_rgba())[:, :, :3].copy()))
        plt.close(f)
    frames[0].save(GROWTH_GIF, save_all=True, append_images=frames[1:], duration=200, loop=0)


# =============================================================================== driver
def _strip_private(d):
    if isinstance(d, dict):
        return {str(k): _strip_private(v) for k, v in d.items() if not (isinstance(k, str) and k.startswith("_"))}
    if isinstance(d, list):
        return [_strip_private(x) for x in d]
    if isinstance(d, (np.floating, np.integer)):
        return d.item()
    if isinstance(d, np.ndarray):
        return d.tolist()
    if isinstance(d, tuple):
        return list(d)
    return d


def run_all(args) -> dict:
    t0 = time.time()
    ref = load_reference()
    digests = verify_inputs(ref)
    tol = float(args.tolerance)
    spec_results = {k: analyse_specimen(k, args.bootstrap, args.seed, tol) for k in SPECIMENS}
    model = run_model_suite(args.eps_sweep, args.lambda_bar, args.gamma_bar, args.seed, args.steps, args.n_cells, tol,
                            record_gif=not args.no_figures)
    gof = {k: goodness_of_fit(spec_results[k], model, tol) for k in SPECIMENS}
    verdict = decide_verdict(spec_results, model, tol)
    if not args.no_figures:
        render_figures(spec_results, model, gof, verdict, tol)
    metrics = {"claim_id": CLAIM_ID, "generated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               "parameters": {"seed": args.seed, "n_cells": args.n_cells, "steps": args.steps, "lambda_bar": args.lambda_bar,
                              "gamma_bar": args.gamma_bar, "eps_sweep": list(args.eps_sweep), "tolerance_deg": tol,
                              "f3_threshold": F3_THRESHOLD, "t1_length": T1_LENGTH, "bootstrap": args.bootstrap},
               "input_sha256": digests, "specimens": _strip_private(spec_results), "model": _strip_private(model),
               "goodness_of_fit": gof, "verdict": verdict, "runtime_s": round(time.time() - t0, 1)}
    # keep the metrics file compact: drop raw per-vertex angle lists from the model block
    for k in ("poisson_voronoi", "plateau_froth", "farhadifar_case1"):
        metrics["model"][k].pop("vertex_angles", None)
    for e in metrics["model"]["eps_sweep"]:
        e.pop("vertex_angles", None)
    METRICS_JSON.write_text(json.dumps(metrics, indent=1, default=float), encoding="utf-8")
    write_result_json(ref, spec_results, model, gof, verdict, tol, digests)
    return metrics


def write_result_json(ref, spec_results, model, gof, verdict, tol, digests):
    prim = spec_results["human_corneal_endothelium"]; sec = spec_results["allium_cepa_epidermis"]
    f103 = spec_results["thompson_fig103_maize_parenchyma"]; f124 = spec_results["thompson_fig124_girardia_epidermis"]
    sp = ref["specimens"]
    result = {
        "claim_id": CLAIM_ID,
        "date_made": "1917",
        "claim": ref["claim"]["text"],
        "citation": "Thompson, D.W. (1917) On Growth and Form, Ch. VII, pp. 293-345, Figs. 100-135",
        "dataset": {
            "pre_scouted_dryad": {"name": "Data from: Sources of variability in cytosolic calcium transients ... uro-epithelial cell monolayers",
                                  "doi": "10.5061/dryad.mn8cs", "status": "DATASET_MISMATCH",
                                  "reason": "Fura-2 Ca2+ ratio JPEG sequences of cultured human urothelial monolayers carry no junction label; byte download also blocked by Dryad's anti-bot challenge",
                                  "metadata_sha256": ref["pre_scouted_dataset_not_used"]["metadata_sha256"]},
            "used_for_verdict": [
                {"name": "Thompson (1917) Fig. 103 Parenchyma of Maize, p. 298 (IA leaf 318)", "doi": None,
                 "sha256": sp["thompson_fig103_maize_parenchyma"]["sha256"], "leaf_sha256": sp["thompson_fig103_maize_parenchyma"]["leaf_sha256"], "role": "Thompson's own figure"},
                {"name": "Thompson (1917) Fig. 124 Epidermis of Girardia, p. 321 (IA leaf 341)", "doi": None,
                 "sha256": sp["thompson_fig124_girardia_epidermis"]["sha256"], "leaf_sha256": sp["thompson_fig124_girardia_epidermis"]["leaf_sha256"], "role": "Thompson's own figure (growth-distorted regime)"},
                {"name": "Wikimedia Commons: Morphology of human corneal endothelium in situ (Zhiguo.he, CC BY-SA 4.0)", "doi": None,
                 "sha256": sp["human_corneal_endothelium"]["sha256"], "role": "PRIMARY real specimen (modern; claim is species-agnostic)"},
                {"name": "Wikimedia Commons: Onion epidermis (microscopic image) 007533276M (J. C. Fonseca Mata, CC BY-SA 4.0)", "doi": None,
                 "sha256": sp["allium_cepa_epidermis"]["sha256"], "original_sha256": sp["allium_cepa_epidermis"]["original_sha256"], "role": "secondary real specimen (anisotropic regime)"},
            ],
        },
        "verdict": verdict["verdict"],
        "verdict_detail": verdict,
        "headline_numbers": {
            "a_priori_tolerance_deg": tol,
            "a_priori_three_fold_threshold": F3_THRESHOLD,
            "corneal_endothelium": {"n_cells": prim["n_cells"], "n_vertices": prim["n_vertices_detected"], "f3": prim["three_fold_fraction"],
                                    "chord_median_abs_dev_deg": prim["chord"]["median_abs_dev_deg"], "chord_sd_deg": prim["chord"]["sd_deg"],
                                    "chord_frac_within_tol": prim["chord"]["frac_within_tol"], "chord_95ci": prim["chord"]["bootstrap_95ci"],
                                    "branch_median_abs_dev_deg": prim["branch"]["median_abs_dev_deg"], "branch_sd_deg": prim["branch"]["sd_deg"],
                                    "ks_D_vs_plateau_froth": gof["human_corneal_endothelium"]["plateau_froth"]["ks_D"],
                                    "ks_D_vs_poisson_voronoi": gof["human_corneal_endothelium"]["poisson_voronoi"]["ks_D"],
                                    "best_eps": gof["human_corneal_endothelium"]["best_eps_fit"]},
            "onion_epidermis": {"n_cells": sec["n_cells"], "n_vertices": sec["n_vertices_detected"], "f3": sec["three_fold_fraction"],
                                "chord_median_abs_dev_deg": sec["chord"]["median_abs_dev_deg"], "chord_sd_deg": sec["chord"]["sd_deg"],
                                "chord_frac_within_tol": sec["chord"]["frac_within_tol"], "best_eps": gof["allium_cepa_epidermis"]["best_eps_fit"]},
            "thompson_fig103": {"n_vertices": f103["n_vertices_detected"], "f3": f103["three_fold_fraction"],
                                "chord_median_abs_dev_deg": f103["chord"]["median_abs_dev_deg"], "chord_sd_deg": f103["chord"]["sd_deg"]},
            "thompson_fig124": {"n_vertices": f124["n_vertices_detected"], "f3": f124["three_fold_fraction"],
                                "chord_median_abs_dev_deg": f124["chord"]["median_abs_dev_deg"], "chord_sd_deg": f124["chord"]["sd_deg"]},
            "plateau_froth_model_sd_deg": model["plateau_froth"]["stats"]["sd_deg"],
            "poisson_voronoi_null_sd_deg": model["poisson_voronoi"]["stats"]["sd_deg"],
            "farhadifar_case1_sd_deg": model["farhadifar_case1"]["stats"]["sd_deg"],
        },
        "script": "thompson/claims/t1917-018_the-forms-of-tissues-or-cell-aggregates.py",
        "figure": str(COMPARISON_PNG.relative_to(ROOT)),
        "animation": str(GROWTH_GIF.relative_to(ROOT)),
        "metrics": str(METRICS_JSON.relative_to(ROOT)),
        "lean": "morphoeval/lean/loops/Thompson1917_T1917_018.lean",
        "notes": [
            "The mean tri-junction angle is 120 degrees by identity (alpha+beta+gamma=360); the verdict rests on the dispersion about 120 and on the three-fold fraction.",
            "Modern Commons specimens are not Thompson's own figures; the claim names no species, so they are specimens of the claim, not analogues of a named species, and are labelled as modern.",
            "Thompson's Figs. 103/124 are hand drawings: angles measured on them test his depiction, not a photograph.",
        ],
    }
    RESULT_JSON.write_text(json.dumps(result, indent=1, default=float), encoding="utf-8")


# =============================================================================== MorphoEval hooks
def measure_morphoeval() -> List[Dict[str, object]]:
    """Quantitative morphometrics for morphoeval/measure_outputs.py. Reads the metrics written by the last
    full run (datasets/thompson_1917/t1917-018/t1917_018_metrics.json); recomputes a quick run if absent."""
    if not METRICS_JSON.exists():
        run_all(parse_args(["--quick", "--no-figures"]))
    m = json.loads(METRICS_JSON.read_text(encoding="utf-8"))
    tol = m["parameters"]["tolerance_deg"]

    def M(name, value, unit, method, reference=None, source=None, derived_from="script"):
        d = {"name": name, "value": float(round(value, 4)) if isinstance(value, (int, float)) else value, "unit": unit,
             "method": method, "derived_from": derived_from}
        if reference is not None:
            d["reference_value"] = reference
        if source:
            d["reference_source"] = source
        return d

    out = []
    for key, short in (("human_corneal_endothelium", "corneal endothelium"), ("allium_cepa_epidermis", "onion epidermis"),
                       ("thompson_fig103_maize_parenchyma", "Thompson Fig. 103"), ("thompson_fig124_girardia_epidermis", "Thompson Fig. 124")):
        s = m["specimens"][key]
        out.append(M(f"{short}: three-fold vertex fraction f3", s["three_fold_fraction"], "fraction",
                     f"vertices with exactly 3 cells / all detected vertices (N={s['n_vertices_detected']})", 1.0, "Plateau's second law; Thompson p. 310"))
        out.append(M(f"{short}: chord median |theta - 120|", s["chord"]["median_abs_dev_deg"], "deg",
                     f"median absolute deviation of chord angles at 3-fold vertices (N3={s['chord']['n_vertices_scored']})", f"<= {tol}", "a-priori tolerance (reference.json)"))
        out.append(M(f"{short}: chord angle SD", s["chord"]["sd_deg"], "deg", "pooled SD of chord angles at 3-fold vertices", 0.0, "Plateau limit (exact 120)"))
        out.append(M(f"{short}: fraction within +/-{tol:.0f} deg", s["chord"]["frac_within_tol"], "fraction", "chord estimator", None, None))
        out.append(M(f"{short}: branch-estimator median |theta - 120|", s["branch"]["median_abs_dev_deg"], "deg",
                     f"mean-offset wall directions within R={s['branch_radius_px']:.0f}px", f"<= {tol}", "a-priori tolerance"))
    g = m["goodness_of_fit"]["human_corneal_endothelium"]
    out += [
        M("corneal endothelium: KS D vs Plateau froth", g["plateau_froth"]["ks_D"], "dimensionless", "two-sample KS, chord angles vs relaxed Gamma=eps=0 froth"),
        M("corneal endothelium: KS D vs Poisson-Voronoi null", g["poisson_voronoi"]["ks_D"], "dimensionless", "two-sample KS"),
        M("corneal endothelium: best-fit tension anisotropy eps", g["best_eps_fit"]["eps"], "dimensionless", "argmin KS D over the eps sweep"),
        M("corneal endothelium: histogram Pearson r at best eps", g["best_eps_fit"]["hist_pearson_r"], "dimensionless", "5-degree bins"),
        M("corneal endothelium: histogram RMSE at best eps", g["best_eps_fit"]["hist_rmse"], "density", "5-degree bins"),
        M("Plateau froth model: junction-angle SD", m["model"]["plateau_froth"]["stats"]["sd_deg"], "deg",
          f"relaxed Gamma=eps=0 froth, {m['model']['plateau_froth']['n_t1']} T1s, converged={m['model']['plateau_froth']['converged']}", 0.0, "Lami/Plateau"),
        M("Poisson-Voronoi null: junction-angle SD", m["model"]["poisson_voronoi"]["stats"]["sd_deg"], "deg", "unrelaxed periodic Voronoi seed"),
        M("Farhadifar Case I: junction-angle SD", m["model"]["farhadifar_case1"]["stats"]["sd_deg"], "deg", "Lambda_bar=0.12, Gamma_bar=0.04 relaxed"),
        M("eps sweep: SD at eps=0.6 minus SD at eps=0", m["model"]["eps_sweep"][-1]["stats"]["sd_deg"] - m["model"]["eps_sweep"][0]["stats"]["sd_deg"], "deg",
          "monotone dispersion gain with tension anisotropy"),
        M("Lami equal-tension junction angle", thompson_fig113_cases()["case3_equal"][0], "deg", "E1 with T1=T2=T3", 120.0, "Thompson p. 297"),
        M("Steiner / cross length ratio (unit square)", steiner_vs_cross_lengths()[1] / steiner_vs_cross_lengths()[0], "dimensionless", "(1+sqrt3)/(2 sqrt2)", 0.9659, "Plateau/Lamarle; Taylor 1976"),
        M("verdict (supported=1, mixed=0.5, refuted=0)", {"supported": 1.0, "mixed": 0.5, "refuted": 0.0, "untestable": None}[m["verdict"]["verdict"]], "code",
          "pre-registered criteria in reference.json"),
    ]
    return out


# =============================================================================== self-test
def self_test() -> int:
    a = lami_angles_from_tensions(1.0, 1.0, 1.0)
    assert a is not None and all(abs(x - 120.0) < 1e-9 for x in a), a
    b = lami_angles_from_tensions(1.0, 1.0, 1e-9)
    assert b is not None and abs(b[0] - 90.0) < 1e-5 and abs(b[2] - 180.0) < 1e-4, b
    assert lami_angles_from_tensions(1.0, 1.0, 2.5) is None
    c = thompson_fig113_cases()["case2_t_small"]
    assert abs(c[0] - 100.0) < 0.05 and abs(sum(c) - 360.0) < 1e-6, c
    T = tensions_from_angles(*lami_angles_from_tensions(1.0, 1.2, 0.9))
    assert np.allclose(np.array(T) / T[0], np.array([1.0, 1.2, 0.9]), atol=1e-6), T
    cross, steiner = steiner_vs_cross_lengths()
    assert steiner < cross
    # SHA-256 pins
    ref = load_reference()
    d = verify_inputs(ref)
    assert len(d) >= 6, d
    # periodic network: E3 identity, three-valence, analytic gradient vs finite differences
    net = periodic_voronoi_network(40, 7)
    pv, val = net.junction_angles()
    assert set(val) == {3}, val
    assert all(abs(sum(v) - 360.0) < 1e-6 for v in pv)
    kw = dict(lambda_bar=0.12, gamma_bar=0.02, eps=0.3)
    E0, g = net.energy_gradient(**kw)
    rng = np.random.default_rng(0)
    for _ in range(5):
        v = rng.integers(0, len(net.pos)); k = rng.integers(0, 2)
        h = 1e-6
        net.pos[v, k] += h; Ep, _ = net.energy_gradient(**kw); net.pos[v, k] -= 2 * h
        Em, _ = net.energy_gradient(**kw); net.pos[v, k] += h
        fd = (Ep - Em) / (2 * h)
        assert abs(fd - g[v, k]) < 1e-5 * max(1.0, abs(fd)), (fd, g[v, k])
    # relaxation lowers energy and dispersion; T1s keep three-valence
    rel, info = relax_froth(net.copy(), steps=600, lambda_bar=0.12, gamma_bar=0.0, eps=0.0)
    pv2, val2 = rel.junction_angles()
    assert info["energies"][-1] < info["energies"][0]
    assert set(val2) == {3}, val2
    sd0, sd1 = np.concatenate(pv).std(), np.concatenate(pv2).std()
    assert sd1 < 0.5 * sd0, (sd0, sd1)
    assert sum(len(c) for c in rel.cells) == 6 * rel.n_cells          # Euler: mean 6 sides preserved by T1s
    # controls
    assert plateau_statistics(np.concatenate(brick_lattice_control())) ["median_abs_dev_deg"] == 30.0
    # determinism
    net2 = periodic_voronoi_network(40, 7)
    assert np.allclose(net2.pos, net.pos)
    # segmentation smoke test on Thompson Fig. 124 (fast): all vertices three-fold, angles sum to 360
    r = analyse_specimen("thompson_fig124_girardia_epidermis", n_boot=20, seed=1, tol=TOLERANCE_DEG)
    assert r["three_fold_fraction"] == 1.0 and r["chord"]["n_vertices_scored"] >= 10, (r["valence_histogram"], r["chord"])
    assert all(abs(sum(v) - 360.0) < 1e-2 for v in r["chord_vertex_angles"])      # stored rounded to 3 dp
    print("self-test OK: Lami 120/90/100 cases + inverse Lami, Steiner < cross, SHA-256 pins, analytic gradient = finite "
          f"difference, Plateau relaxation SD {sd0:.1f} -> {sd1:.1f} deg with {info['n_t1']} T1s (three-valent, mean 6 sides), "
          f"Fig. 124 segmentation: {r['chord']['n_vertices_scored']} three-fold vertices")
    return 0


def parse_args(argv: Optional[Sequence[str]] = None):
    ap = argparse.ArgumentParser(description="Thompson (1917) T1917-018: cells meet three-at-a-time at ~120 degrees")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--quick", action="store_true", help="fewer relaxation steps / bootstraps")
    ap.add_argument("--no-figures", action="store_true")
    ap.add_argument("--eps-sweep", type=lambda s: tuple(float(x) for x in s.split(",")), default=EPS_SWEEP_DEFAULT)
    ap.add_argument("--lambda-bar", type=float, default=LAMBDA_BAR)
    ap.add_argument("--gamma-bar", type=float, default=0.0, help="perimeter contractility (0 = soap-froth Plateau limit)")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--steps", type=int, default=RELAX_STEPS)
    ap.add_argument("--n-cells", type=int, default=N_CELLS)
    ap.add_argument("--bootstrap", type=int, default=400)
    ap.add_argument("--tolerance", type=float, default=TOLERANCE_DEG)
    args = ap.parse_args(argv)
    if args.quick:
        args.steps = min(args.steps, 600)
        args.bootstrap = min(args.bootstrap, 50)
    return args


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if args.self_test:
        return self_test()
    m = run_all(args)
    v = m["verdict"]
    print(f"wrote {COMPARISON_PNG.relative_to(ROOT)}, {GROWTH_GIF.relative_to(ROOT)}, {RESULT_JSON.relative_to(ROOT)}, "
          f"{METRICS_JSON.relative_to(ROOT)}  ({m['runtime_s']} s)")
    for k, s in m["specimens"].items():
        print(f"{k}: cells={s['n_cells']} vertices={s['n_vertices_detected']} f3={s['three_fold_fraction']:.3f} "
              f"chord SD={s['chord']['sd_deg']:.1f} med|dev|={s['chord']['median_abs_dev_deg']:.1f} within={s['chord']['frac_within_tol']:.2f} | "
              f"branch SD={s['branch']['sd_deg']:.1f} med|dev|={s['branch']['median_abs_dev_deg']:.1f}")
    pf = m["model"]["plateau_froth"]
    print(f"Plateau froth SD={pf['stats']['sd_deg']:.2f} (T1s={pf['n_t1']}, converged={pf['converged']}); "
          f"Voronoi SD={m['model']['poisson_voronoi']['stats']['sd_deg']:.1f}; Farhadifar SD={m['model']['farhadifar_case1']['stats']['sd_deg']:.1f}")
    print("eps sweep SD:", [(e["eps"], round(e["stats"]["sd_deg"], 1)) for e in m["model"]["eps_sweep"]])
    print(f"VERDICT: {v['verdict']}  (f3 pass={v['three_fold_law_passes']}, angle pass={v['angle_tolerance_passes']}, "
          f"below Voronoi={v['dispersion_below_poisson_voronoi_null']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""
t1917-018_the-forms-of-tissues-or-cell-aggregates.py — Thompson (1917) claim T1917-018 (Pass 1, v1)

Claim under test (On Growth and Form, Ch. VII "The Forms of Tissues, or Cell-Aggregates",
pp. 293-345, Figs. 100-135; Internet Archive `ongrowthform1917thom`, jp2 leaf = page + 20):

    "Cells in contact, like soap films, meet three-at-a-time at ~120 degrees and in other
     respects obey Plateau's laws of film configuration."

Model implemented (datasets/thompson_1917/t1917-018/*_reference.json, equations E1-E5):
  E1  Lami / Thompson Fig. 113 (p. 303): three films of tensions T1, T2, T3 balance at a point
      iff T_i / sin(alpha_i) is constant; cos(alpha_i) = (T_i^2 - T_j^2 - T_k^2) / (2 T_j T_k).
      Equal tensions -> 120, 120, 120.  Cases (1)-(3) of Fig. 113 are reproduced numerically.
  E2  Plateau / Lamarle (pp. 309-310): minimal partitions have only three-fold vertices at 120
      degrees; the unit-square X partition (2 sqrt 2) is longer than the two-Y Steiner one (1 + sqrt 3).
  E3  alpha + beta + gamma = 360 at every planar tri-junction, so the MEAN angle is 120 by
      identity; the test statistic is the dispersion D_v = sqrt(mean (theta - 120)^2).
  E5  Periodic vertex model E = sum_c [ (K/2)(A_c - A0)^2 + (Gamma/2) P_c^2 ] + sum_e Lambda l_e
      (Farhadifar et al. 2007 Eq. 1) relaxed from a Poisson-Voronoi seed: the isotropic
      soap-froth limit in which Thompson predicts 120 degrees.

Lean 4 skeleton: morphoeval/lean/loops/Thompson1917_T1917_018.lean (MathOfBiology.Loop_213).

Usage:
    python3 thompson/claims/t1917-018_the-forms-of-tissues-or-cell-aggregates.py            # render figures
    python3 thompson/claims/t1917-018_the-forms-of-tissues-or-cell-aggregates.py --self-test

Outputs:
    docs/media/t1917_018_the_forms_of_tissues_or_cell_a_comparison.png
    docs/media/t1917_018_the_forms_of_tissues_or_cell_a_growth.gif
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "datasets" / "thompson_1917" / "t1917-018"
MEDIA_DIR = ROOT / "docs" / "media"
STEM = "t1917_018_the_forms_of_tissues_or_cell_a"
COMPARISON_PNG = MEDIA_DIR / f"{STEM}_comparison.png"
GROWTH_GIF = MEDIA_DIR / f"{STEM}_growth.gif"

PLATEAU_DEG = 120.0
SEED = 1917
N_CELLS = 96
LAMBDA_BAR = 0.12          # Farhadifar et al. (2007) Case I wing-disc line tension
GAMMA_BAR = 0.04           # perimeter contractility
K_AREA = 1.0
RELAX_STEPS = 400
RELAX_DT = 0.05


# ----------------------------------------------------------------------------- E1: Lami / Fig. 113
def lami_angles_from_tensions(T1: float, T2: float, T3: float) -> Optional[Tuple[float, float, float]]:
    """Interior angles (deg) opposite films 1, 2, 3 at a balanced tri-junction (Thompson pp. 296-297).

    Returns None when no triangle of forces exists (Thompson's water/oil/air example, p. 296)."""
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


def thompson_fig113_cases() -> Dict[str, Tuple[float, float, float]]:
    """The three worked cases of Fig. 113 (p. 303): T = T' with t negligible / small / equal."""
    return {
        "case1_t_negligible": lami_angles_from_tensions(1.0, 1.0, 1e-6),
        "case2_t_small": lami_angles_from_tensions(1.0, 1.0, 0.3473),   # alpha = beta = 100 deg
        "case3_equal": lami_angles_from_tensions(1.0, 1.0, 1.0),
    }


def steiner_vs_cross_lengths() -> Tuple[float, float]:
    """Unit-square film lengths: X partition (four-fold) vs two-Y Steiner partition (E2)."""
    return 2.0 * math.sqrt(2.0), 1.0 + math.sqrt(3.0)


# ----------------------------------------------------------------------------- E3: statistics
def plateau_statistics(angles_deg: np.ndarray) -> Dict[str, float]:
    """Dispersion statistics about 120 deg for a pooled array of junction angles."""
    a = np.asarray(angles_deg, float)
    dev = a - PLATEAU_DEG
    return {
        "n_angles": int(a.size),
        "mean_deg": float(a.mean()) if a.size else float("nan"),
        "sd_deg": float(a.std(ddof=1)) if a.size > 1 else float("nan"),
        "rms_dev_deg": float(np.sqrt(np.mean(dev ** 2))) if a.size else float("nan"),
        "median_abs_dev_deg": float(np.median(np.abs(dev))) if a.size else float("nan"),
        "frac_within_15": float(np.mean(np.abs(dev) <= 15.0)) if a.size else float("nan"),
    }


# ----------------------------------------------------------------------------- E5: periodic froth
class PeriodicNetwork:
    """Periodic 2D vertex network: cells are lists of (vertex_id, wrap_offset) in CCW order."""

    def __init__(self, pos: np.ndarray, cells: List[List[Tuple[int, np.ndarray]]], L: float):
        self.pos = pos.astype(float)
        self.cells = cells
        self.L = L

    def cell_polygon(self, c: int) -> np.ndarray:
        return np.array([self.pos[v] + off for v, off in self.cells[c]])

    def vertex_directions(self) -> Dict[int, List[np.ndarray]]:
        dirs: Dict[int, List[np.ndarray]] = {}
        for cell in self.cells:
            n = len(cell)
            for k in range(n):
                v1, o1 = cell[k]
                v2, o2 = cell[(k + 1) % n]
                d = (self.pos[v2] + o2) - (self.pos[v1] + o1)
                for v, vec in ((v1, d), (v2, -d)):
                    lst = dirs.setdefault(v, [])
                    if not any(np.linalg.norm(vec - u) < 1e-9 for u in lst):
                        lst.append(vec)
        return dirs

    def junction_angles(self) -> Tuple[np.ndarray, Dict[int, int]]:
        """Pooled interior angles (deg) at every vertex and the vertex-valence histogram."""
        angles, valence = [], {}
        for v, vecs in self.vertex_directions().items():
            valence[len(vecs)] = valence.get(len(vecs), 0) + 1
            az = np.sort(np.array([math.atan2(d[1], d[0]) for d in vecs]))
            gaps = np.diff(np.concatenate([az, [az[0] + 2 * math.pi]]))
            angles.extend(np.degrees(gaps).tolist())
        return np.array(angles), valence

    def polygon_classes(self) -> Dict[int, int]:
        out: Dict[int, int] = {}
        for cell in self.cells:
            out[len(cell)] = out.get(len(cell), 0) + 1
        return out


def periodic_voronoi_network(n_cells: int = N_CELLS, seed: int = SEED) -> PeriodicNetwork:
    """Poisson-Voronoi seed tessellation on a periodic square of side sqrt(n) (so <A> = 1)."""
    from scipy.spatial import Voronoi

    rng = np.random.default_rng(seed)
    L = math.sqrt(n_cells)
    pts = rng.uniform(0, L, size=(n_cells, 2))
    shifts = np.array([[i, j] for i in (-1, 0, 1) for j in (-1, 0, 1)], float) * L
    tiled = np.concatenate([pts + s for s in shifts])
    vor = Voronoi(tiled)
    centre = 4                                     # shift index of (0, 0)
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


def froth_energy_and_gradient(net: PeriodicNetwork, lambda_bar: float = LAMBDA_BAR,
                              gamma_bar: float = GAMMA_BAR, K: float = K_AREA,
                              A0: float = 1.0) -> Tuple[float, np.ndarray]:
    """E = sum_c [K/2 (A_c - A0)^2 + Gamma/2 P_c^2] + sum_e Lambda l_e, and dE/dr for every vertex."""
    grad = np.zeros_like(net.pos)
    E = 0.0
    for cell in net.cells:
        n = len(cell)
        P = np.array([net.pos[v] + off for v, off in cell])
        x, y = P[:, 0], P[:, 1]
        A = 0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)
        edges = np.roll(P, -1, axis=0) - P
        lens = np.linalg.norm(edges, axis=1)
        perim = lens.sum()
        E += 0.5 * K * (A - A0) ** 2 + 0.5 * gamma_bar * perim ** 2 + 0.5 * lambda_bar * perim
        dA = 0.5 * np.stack([np.roll(y, -1) - np.roll(y, 1), np.roll(x, 1) - np.roll(x, -1)], axis=1)
        unit = edges / lens[:, None]
        dP = -unit + np.roll(unit, 1, axis=0)       # d perimeter / d vertex_k
        g = K * (A - A0) * dA + (gamma_bar * perim + 0.5 * lambda_bar) * dP
        for k, (v, _) in enumerate(cell):
            grad[v] += g[k]
    return E, grad


def relax_froth(net: PeriodicNetwork, steps: int = RELAX_STEPS, dt: float = RELAX_DT,
                record_every: int = 0, **kw) -> Tuple[PeriodicNetwork, List[float], List[np.ndarray]]:
    """Over-damped gradient descent on the froth energy (no T1 transitions: small-strain regime)."""
    energies, snapshots = [], []
    for s in range(steps):
        E, g = froth_energy_and_gradient(net, **kw)
        energies.append(E)
        if record_every and s % record_every == 0:
            snapshots.append(net.pos.copy())
        net.pos = net.pos - dt * g
    E, _ = froth_energy_and_gradient(net, **kw)
    energies.append(E)
    snapshots.append(net.pos.copy())
    return net, energies, snapshots


# ----------------------------------------------------------------------------- figures
def _draw_network(ax, net: PeriodicNetwork, title: str, color_by_sides: bool = True):
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection

    polys = [net.cell_polygon(c) for c in range(len(net.cells))]
    if color_by_sides:
        sides = np.array([len(c) for c in net.cells])
        cmap = plt.get_cmap("coolwarm")
        colors = [cmap((s - 4) / 4.0) for s in sides]
    else:
        colors = ["#dddddd"] * len(polys)
    ax.add_collection(PolyCollection(polys, facecolors=colors, edgecolors="k", linewidths=0.9, alpha=0.9))
    ax.set_xlim(0, net.L)
    ax.set_ylim(0, net.L)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=9)


def render_outputs(seed: int = SEED) -> Dict[str, object]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    MEDIA_DIR.mkdir(parents=True, exist_ok=True)
    seed_net = periodic_voronoi_network(N_CELLS, seed)
    voronoi_angles, voronoi_valence = seed_net.junction_angles()
    relaxed = PeriodicNetwork(seed_net.pos.copy(), seed_net.cells, seed_net.L)
    relaxed, energies, snaps = relax_froth(relaxed, record_every=20)
    froth_angles, froth_valence = relaxed.junction_angles()
    s_vor, s_froth = plateau_statistics(voronoi_angles), plateau_statistics(froth_angles)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    ratios = np.linspace(0.0, 2.0, 201)
    alpha = [lami_angles_from_tensions(1.0, 1.0, r)[0] if lami_angles_from_tensions(1.0, 1.0, r) else np.nan for r in ratios]
    gamma = [lami_angles_from_tensions(1.0, 1.0, r)[2] if lami_angles_from_tensions(1.0, 1.0, r) else np.nan for r in ratios]
    ax = axes[0]
    ax.plot(ratios, alpha, "b-", label=r"$\alpha=\beta$ (opposite $T, T'$)")
    ax.plot(ratios, gamma, "r-", label=r"$\gamma$ (opposite partition $t$)")
    ax.axhline(120, color="k", ls="--", lw=0.8)
    ax.axvline(1.0, color="k", ls=":", lw=0.8)
    for r, lab in ((1e-6, "case (1)\n90°"), (0.3473, "case (2)\n100°"), (1.0, "case (3)\n120°")):
        a = lami_angles_from_tensions(1.0, 1.0, max(r, 1e-6))[0]
        ax.plot([r], [a], "ko", ms=4)
        ax.annotate(lab, (r, a), textcoords="offset points", xytext=(6, -22), fontsize=7)
    ax.set_xlabel("partition tension ratio  t / T")
    ax.set_ylabel("junction angle (deg)")
    ax.set_title("E1  Lami balance — Thompson Fig. 113 (p. 303)", fontsize=9)
    ax.legend(fontsize=7, loc="center right")
    _draw_network(axes[1], relaxed, f"E5  relaxed periodic froth (N={N_CELLS}, Λ̄={LAMBDA_BAR}, Γ̄={GAMMA_BAR})\n"
                                    f"cells coloured by side number 4..8")
    ax = axes[2]
    bins = np.arange(60, 181, 5)
    ax.hist(voronoi_angles, bins=bins, alpha=0.45, color="grey", label=f"Poisson-Voronoi seed  SD={s_vor['sd_deg']:.1f}°")
    ax.hist(froth_angles, bins=bins, alpha=0.6, color="tab:blue", label=f"relaxed froth  SD={s_froth['sd_deg']:.1f}°")
    ax.axvline(120, color="k", ls="--")
    ax.axvspan(105, 135, color="gold", alpha=0.15, label="±15° tolerance")
    ax.set_xlabel("tri-junction angle (deg)")
    ax.set_ylabel("count")
    ax.set_title("E3  junction angles: mean is 120 by identity, dispersion is the test", fontsize=9)
    ax.legend(fontsize=7)
    fig.suptitle("Thompson (1917) T1917-018 — cells meet three-at-a-time at ~120° (Pass 1 baseline: model only)", fontsize=11)
    fig.tight_layout()
    fig.savefig(COMPARISON_PNG, dpi=110)
    plt.close(fig)

    frames = []
    for i, p in enumerate(snaps):
        net_i = PeriodicNetwork(p, seed_net.cells, seed_net.L)
        ang, _ = net_i.junction_angles()
        f, (a1, a2) = plt.subplots(1, 2, figsize=(8, 4))
        _draw_network(a1, net_i, f"froth relaxation step {min(i * 20, RELAX_STEPS)}")
        a2.hist(ang, bins=bins, color="tab:blue", alpha=0.7)
        a2.axvline(120, color="k", ls="--")
        a2.set_xlim(60, 180)
        a2.set_ylim(0, 140)
        a2.set_title(f"junction angles  SD = {np.std(ang, ddof=1):.1f}°", fontsize=9)
        f.tight_layout()
        f.canvas.draw()
        frames.append(Image.fromarray(np.asarray(f.canvas.buffer_rgba())[:, :, :3].copy()))
        plt.close(f)
    frames[0].save(GROWTH_GIF, save_all=True, append_images=frames[1:], duration=250, loop=0)
    return {"voronoi": s_vor, "froth": s_froth, "voronoi_valence": voronoi_valence,
            "froth_valence": froth_valence, "energies": energies, "polygon_classes": relaxed.polygon_classes()}


# ----------------------------------------------------------------------------- MorphoEval hooks
def measure_morphoeval() -> List[Dict[str, object]]:
    """Quantitative morphometrics for morphoeval/measure_outputs.py (Pass 1: model-only)."""
    seed_net = periodic_voronoi_network(N_CELLS, SEED)
    vor_angles, _ = seed_net.junction_angles()
    relaxed, _, _ = relax_froth(PeriodicNetwork(seed_net.pos.copy(), seed_net.cells, seed_net.L))
    froth_angles, valence = relaxed.junction_angles()
    sv, sf = plateau_statistics(vor_angles), plateau_statistics(froth_angles)
    cross, steiner = steiner_vs_cross_lengths()
    f113 = thompson_fig113_cases()

    def M(name, value, unit, method, reference=None, source=None, derived_from="script"):
        m = {"name": name, "value": float(round(value, 4)), "unit": unit, "method": method, "derived_from": derived_from}
        if reference is not None:
            m["reference_value"] = reference
        if source:
            m["reference_source"] = source
        return m

    return [
        M("Lami equal-tension junction angle", f113["case3_equal"][0], "deg",
          "cos(alpha) = (T^2 - T^2 - T^2)/(2 T T) with T1 = T2 = T3 (E1)", 120.0, "Thompson 1917 p. 297 / p. 303 case (3)"),
        M("Lami negligible-partition angle", f113["case1_t_negligible"][0], "deg",
          "E1 with t -> 0 (Fig. 113 case 1)", 90.0, "Thompson 1917 p. 303 case (1)"),
        M("Steiner / cross length ratio (unit square)", steiner / cross, "dimensionless",
          "(1 + sqrt 3) / (2 sqrt 2) (E2)", 0.9659, "Plateau/Lamarle; Taylor 1976"),
        M("relaxed froth junction-angle SD", sf["sd_deg"], "deg",
          f"pooled SD of all tri-junction angles after {RELAX_STEPS} gradient steps of E5 (seed {SEED})", None, None),
        M("Poisson-Voronoi seed junction-angle SD (negative control)", sv["sd_deg"], "deg",
          "pooled SD of tri-junction angles of the unrelaxed Voronoi seed", None, None),
        M("relaxed froth fraction of angles within +/-15 deg of 120", sf["frac_within_15"], "fraction",
          "|theta - 120| <= 15 over all junction angles", None, "a-priori tolerance, reference.json"),
        M("relaxed froth three-fold vertex fraction", valence.get(3, 0) / max(1, sum(valence.values())), "fraction",
          "vertices with exactly three incident edges / all vertices", 1.0, "Plateau's second law"),
    ]


def self_test() -> int:
    a = lami_angles_from_tensions(1.0, 1.0, 1.0)
    assert a is not None and all(abs(x - 120.0) < 1e-9 for x in a), a
    b = lami_angles_from_tensions(1.0, 1.0, 1e-9)
    assert b is not None and abs(b[0] - 90.0) < 1e-5 and abs(b[2] - 180.0) < 1e-4, b
    assert lami_angles_from_tensions(1.0, 1.0, 2.5) is None                    # violates triangle inequality
    c = thompson_fig113_cases()["case2_t_small"]
    assert abs(c[0] - 100.0) < 0.05 and abs(sum(c) - 360.0) < 1e-6, c
    cross, steiner = steiner_vs_cross_lengths()
    assert steiner < cross
    net = periodic_voronoi_network(48, 7)
    angles, valence = net.junction_angles()
    assert set(valence) == {3}, valence
    assert abs(np.mean(angles) - 120.0) < 1e-6                                 # E3 identity
    relaxed, energies, _ = relax_froth(PeriodicNetwork(net.pos.copy(), net.cells, net.L), steps=150)
    assert energies[-1] < energies[0], (energies[0], energies[-1])
    r_angles, _ = relaxed.junction_angles()
    assert np.std(r_angles) < np.std(angles), (np.std(r_angles), np.std(angles))
    # determinism
    net2 = periodic_voronoi_network(48, 7)
    assert np.allclose(net2.pos, net.pos)
    print("self-test OK: Lami 120/90 cases, Fig. 113 case (2) = 100 deg, Steiner < cross, "
          f"froth SD {np.std(r_angles):.1f} < Voronoi SD {np.std(angles):.1f} deg, all vertices three-fold")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args(argv)
    if args.self_test:
        return self_test()
    res = render_outputs()
    print(f"wrote {COMPARISON_PNG.relative_to(ROOT)} and {GROWTH_GIF.relative_to(ROOT)}")
    print(f"Voronoi seed: SD {res['voronoi']['sd_deg']:.2f} deg, within 15: {res['voronoi']['frac_within_15']:.3f}")
    print(f"relaxed froth: SD {res['froth']['sd_deg']:.2f} deg, within 15: {res['froth']['frac_within_15']:.3f}, "
          f"valence {res['froth_valence']}, polygon classes {res['polygon_classes']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

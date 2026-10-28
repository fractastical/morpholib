#!/usr/bin/env python3
"""
Thompson (1917) T1917-032 — the shapes of teeth or tusks, with a note on torsion
(On Growth and Form, Ch. XIII, pp. 612-634, Figs. 318-324).

Claim under test
----------------
"Teeth and tusks grow by the same spiral/torsion geometry (with St Venant torsion curves
describing the twisted prismatic forms)."

Model (Thompson Ch. XIII; algebraic forms reconstructed in
datasets/thompson_1917/t1917-032/t1917_032_the_shapes_of_horns_and_of_tee_reference.json)
  E1  r(θ) = r₀ exp(θ cot ψ)                 equiangular spiral, constant angle ψ
  E2  W = exp(2π cot ψ)                      whorl expansion (gnomonic magnification)
  E3  κ = sin ψ / r                          curvature of the log spiral
  E4  κ = (v_max − v_min)/(w v̄) = Δ/w        differential accretion across a section of width w
  E5  w = c r ⇒ sin ψ = Δ/c ;  w = const ⇒ circle of radius w/Δ  (rodent incisor limit)
  E6  Ω = κ b + τ n                           screw accretion (bend + twist) ⇒ helix / helico-spiral
  E7  tan φ = ρ τ ;  tan α = π/(τ d)          St Venant uniform-torsion helix; Kiladze ratio
  E8  u_z ∝ 3x²y − y³                         St Venant warping of the twisted equilateral prism (Fig. 321)
  E9  u_z ∝ (b² − a²) x y                     elliptic prism; circular section a = b does not warp

Specimens (SHA-256 pinned in the reference JSON, verified at import)
  * Mammuthus primigenius tusk, Museum Wiesbaden (Wikimedia Commons, CC BY-SA 4.0): species named
    by Thompson p. 634 — log-spiral / accretion-law test on the segmented centreline.
  * Monodon monoceros tusk close-up (Wikimedia Commons, CC BY 2.0): TORSION ANALOGUE (the narwhal
    is not named in Thompson's chapter) — groove helix angle vs Kiladze & Chernova (2018).
  * Thompson's own Fig. 321 (IA scan leaf n642, public domain) — St Venant torsion curves.

Lean 4 formalisation of E1/E2 (gnomonic self-similarity), E4/E5 (accretion bending law),
E8/E9 (St Venant anticlastic warping) and E7 (uniform-torsion helix) in
morphoeval/lean/loops/Thompson1917_T1917_032.lean (namespace MathOfBiology.Loop_226).

Usage
  python3 thompson/claims/t1917-032_the-shapes-of-horns-and-of-teeth-or-tusks.py --self-test
  python3 thompson/claims/t1917-032_the-shapes-of-horns-and-of-teeth-or-tusks.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: E402
from PIL import Image  # noqa: E402
from scipy import ndimage as ndi  # noqa: E402
from scipy import stats  # noqa: E402
from skimage import color, measure, morphology  # noqa: E402
from skimage.feature import structure_tensor  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "datasets" / "thompson_1917" / "t1917-032"
REF_JSON = DATA_DIR / "t1917_032_the_shapes_of_horns_and_of_tee_reference.json"
MAMMOTH = DATA_DIR / "mammuthus_primigenius_tusk_specimen.jpg"
NARWHAL = DATA_DIR / "monodon_monoceros_tusk_spiral_specimen.jpg"
FIG321 = DATA_DIR / "thompson1917_p623_fig321_322_st_venant_torsion_curves.jpg"
KILADZE_CSV = DATA_DIR / "kiladze2018_narwhal_spiral_table1.csv"
OUT_PNG = ROOT / "docs" / "media" / "t1917_032_the_shapes_of_horns_and_of_tee_comparison.png"
OUT_GIF = ROOT / "docs" / "media" / "t1917_032_the_shapes_of_horns_and_of_tee_growth.gif"
RESULT_JSON = DATA_DIR / "result.json"

SEED = 1917032
RNG = np.random.default_rng(SEED)


# --------------------------------------------------------------------------- #
# Provenance
# --------------------------------------------------------------------------- #
def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_reference(verify: bool = True) -> dict:
    """Load the reference JSON and verify every pinned dataset digest (fail closed)."""
    ref = json.loads(REF_JSON.read_text())
    if verify:
        for name, meta in ref["datasets"].items():
            got = sha256_file(DATA_DIR / name)
            if got != meta["sha256"]:
                sys.exit(f"{name}: sha256 mismatch (expected {meta['sha256'][:16]}…, got {got[:16]}…)")
    return ref


def load_kiladze_table() -> dict[str, dict[str, float]]:
    """Table 1 of Kiladze & Chernova (2018), doi:10.1016/j.dib.2018.09.029 (CC BY 4.0)."""
    out: dict[str, dict[str, float]] = {}
    for line in KILADZE_CSV.read_text().splitlines():
        if not line or line.startswith("#") or line.startswith("parameter"):
            continue
        p, unit, mean, sem, lo, hi, n = line.split(",")
        out[p] = {"mean": float(mean), "sem": float(sem), "min": float(lo), "max": float(hi), "n": int(n),
                  "sd": float(sem) * math.sqrt(int(n))}
    return out


# --------------------------------------------------------------------------- #
# E1-E3  Equiangular spiral (Lean Law 1: gnomon_radius / gnomon_radius_self_similar)
# --------------------------------------------------------------------------- #
def log_spiral_r(r0: float, psi_deg: float, theta: np.ndarray) -> np.ndarray:
    """E1: r(θ) = r₀ exp(θ cot ψ)."""
    k = 1.0 / math.tan(math.radians(psi_deg))
    return r0 * np.exp(k * theta)


def whorl_expansion(psi_deg: float) -> float:
    """E2: W = exp(2π cot ψ) (Raup 1966 restatement)."""
    return float(math.exp(2 * math.pi / math.tan(math.radians(psi_deg))))


def gnomon_radius(r0: float, W: float, n: int) -> float:
    """Discrete gnomon r_n = r₀ W^n — the quantity proved self-similar in Lean."""
    return r0 * W ** n


def log_spiral_curvature(r: np.ndarray, psi_deg: float) -> np.ndarray:
    """E3: κ = sin ψ / r (radius of curvature r/sin ψ)."""
    return math.sin(math.radians(psi_deg)) / r


# --------------------------------------------------------------------------- #
# E4-E6  Differential accretion & screw growth (Lean Law 2: bend_moment / accretion law)
# --------------------------------------------------------------------------- #
def bend_moment(vmax: float, vmin: float) -> float:
    """Lean `bend_moment`: growth-rate contrast across the basal zone."""
    return vmax - vmin


def accretion_curvature(vmax: float, vmin: float, w: float) -> float:
    """E4: κ = (v_max − v_min)/(w v̄) = Δ/w with Δ = 2(1−λ)/(1+λ)."""
    return bend_moment(vmax, vmin) / (w * 0.5 * (vmax + vmin))


def grow_tooth(lam: float, taper_c: float, tau: float, n_steps: int = 600, ds: float = 1.0,
               w_tip: float = 2.0) -> dict[str, np.ndarray]:
    """E4-E6 screw-accretion growth of a tooth/tusk from its tip.

    Each slab of arc ds is added at the growth zone with width w(s) = w_tip + c·s, bends the
    axis by dθ = Δ/w ds about the in-section bending axis (differential growth, E4) and twists
    the body frame about the tangent by τ ds (oblique fibres / torsional strain, E6).
      c = 0, τ = 0 -> circular arc (rodent incisor);  c > 0, τ = 0 -> plane log spiral (tusk);
      τ > 0 -> helix / conical log helico-spiral ("curve of double curvature", p. 634).
    """
    delta = 2 * (1 - lam) / (1 + lam)
    t = np.array([1.0, 0.0, 0.0])      # tangent (growth direction)
    b = np.array([0.0, 1.0, 0.0])      # bending axis (in-section, ⟂ growth gradient)
    n_vec = np.cross(t, b)             # section normal completing the frame
    pos = np.zeros(3)
    P = [pos.copy()]
    widths: list[float] = []
    kappas: list[float] = []
    for i in range(n_steps):
        s = i * ds
        w = w_tip + taper_c * s
        kappa = delta / w
        widths.append(w)
        kappas.append(kappa)
        dth = kappa * ds
        # bend: rotate (t, n) about b by dθ
        t, n_vec = (t * math.cos(dth) + n_vec * math.sin(dth), -t * math.sin(dth) + n_vec * math.cos(dth))
        # twist: rotate (b, n) about t by τ ds
        dph = tau * ds
        b, n_vec = (b * math.cos(dph) + n_vec * math.sin(dph), -b * math.sin(dph) + n_vec * math.cos(dph))
        pos = pos + t * ds
        P.append(pos.copy())
    widths.append(w_tip + taper_c * n_steps * ds)
    kappas.append(delta / widths[-1])
    return {"xyz": np.array(P), "w": np.array(widths), "kappa": np.array(kappas), "delta": delta}


# --------------------------------------------------------------------------- #
# E8-E9  St Venant torsion curves (Lean Law 3: tri_warp / ellipse_warp)
# --------------------------------------------------------------------------- #
def tri_warp(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """E8 warping of the twisted equilateral prism, centroid at O, vertex on +x (Fig. 321)."""
    return 3 * x * x * y - y ** 3


def ellipse_warp(a: float, b: float, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """E9 warping of the twisted elliptic prism; a = b (circle) gives identically zero."""
    return (b * b - a * a) * x * y


def st_venant_triangle_field(n: int = 301, a: float = 1.0) -> dict[str, Any]:
    """Evaluate u_z on the equilateral triangle of Thompson's Fig. 321 (OX = 2a/3, side at
    x = −a/3) and quantify the anticlastic structure: number of alternating sign sectors around
    the centroid and sign changes along each face (Thompson: half convex, half concave)."""
    xs = np.linspace(-a / 3 - 0.05, 2 * a / 3 + 0.05, n)
    ys = np.linspace(-a / math.sqrt(3) - 0.05, a / math.sqrt(3) + 0.05, n)
    X, Y = np.meshgrid(xs, ys)
    inside = (X >= -a / 3) & (Y <= (2 * a / 3 - X) / math.sqrt(3)) & (Y >= -(2 * a / 3 - X) / math.sqrt(3))
    U = tri_warp(X, Y)
    U = np.where(inside, U, np.nan)
    # sign sectors on a circle of radius a/4 about the centroid
    th = np.linspace(0, 2 * math.pi, 720, endpoint=False) + math.pi / 720.0   # avoid sampling the nodal rays exactly
    sgn = np.sign(tri_warp(0.25 * a * np.cos(th), 0.25 * a * np.sin(th)))
    sgn = sgn[sgn != 0]
    sectors = int(np.sum(sgn != np.roll(sgn, 1)))
    # sign changes along each face (parametrised by the face coordinate)
    verts = [np.array([2 * a / 3, 0.0]), np.array([-a / 3, a / math.sqrt(3)]), np.array([-a / 3, -a / math.sqrt(3)])]
    face_changes = []
    for i in range(3):
        p0, p1 = verts[i], verts[(i + 1) % 3]
        tt = np.linspace(0.02, 0.98, 400)
        fx = p0[0] + (p1[0] - p0[0]) * tt
        fy = p0[1] + (p1[1] - p0[1]) * tt
        s = np.sign(tri_warp(fx, fy))
        face_changes.append(int(np.sum(s[1:] != s[:-1])))
    return {"X": X, "Y": Y, "U": U, "inside": inside, "sign_sectors": sectors, "face_sign_changes": face_changes,
            "circle_warp_max": float(np.nanmax(np.abs(ellipse_warp(1.0, 1.0, X, Y))))}


# --------------------------------------------------------------------------- #
# E7  Uniform-torsion helix (Lean Law 4: helix_slope)
# --------------------------------------------------------------------------- #
def helix_slope(rho: float, tau: float) -> float:
    """E7: tan φ = ρ τ, inclination of a surface fibre from the axis."""
    return rho * tau


def kiladze_tan_alpha_from_phi(phi_from_axis_deg: float) -> float:
    """Convert a locally measured groove inclination φ (from the axis) at the silhouette centre
    line into Kiladze & Chernova's chord ratio tan α = BC/AC = P/(2d) = π/(2 tan φ)."""
    return math.pi / (2 * math.tan(math.radians(phi_from_axis_deg)))


def kiladze_tan_alpha_from_twist(tau: float, d: float) -> float:
    """E7: tan α = π/(τ d) for a uniform twist rate τ on a tusk of diameter d."""
    return math.pi / (tau * d)


# --------------------------------------------------------------------------- #
# Computer vision: mammoth tusk centreline, width and curvature
# --------------------------------------------------------------------------- #
def segment_mammoth_tusk(img: np.ndarray) -> np.ndarray:
    """Segment the ivory against the dark display background (HSV value & saturation),
    drop the white label and the broken root fragments, keep the largest component."""
    hsv = color.rgb2hsv(img)
    S, V = hsv[..., 1], hsv[..., 2]
    mask = (V > 0.25) & (S > 0.25)
    mask = morphology.remove_small_objects(mask, max_size=500)
    mask = ndi.binary_fill_holes(morphology.closing(mask, morphology.disk(5)))
    mask = morphology.opening(mask, morphology.disk(9))
    lab = measure.label(mask)
    props = sorted(measure.regionprops(lab), key=lambda p: -p.area)
    return lab == props[0].label


def skeleton_longest_path(skel: np.ndarray) -> np.ndarray:
    """Longest geodesic path through a skeleton (two BFS passes); returns (x, y) pixels."""
    from collections import deque

    ys, xs = np.nonzero(skel)
    idx = {(int(y), int(x)): i for i, (y, x) in enumerate(zip(ys, xs))}
    nbrs: list[list[int]] = [[] for _ in range(len(ys))]
    for i, (y, x) in enumerate(zip(ys, xs)):
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                j = idx.get((int(y) + dy, int(x) + dx))
                if j is not None:
                    nbrs[i].append(j)

    def bfs(src: int) -> tuple[np.ndarray, np.ndarray]:
        d = -np.ones(len(ys), int)
        par = -np.ones(len(ys), int)
        d[src] = 0
        q = deque([src])
        while q:
            u = q.popleft()
            for v in nbrs[u]:
                if d[v] < 0:
                    d[v] = d[u] + 1
                    par[v] = u
                    q.append(v)
        return d, par

    d0, _ = bfs(0)
    a = int(np.argmax(d0))
    d1, par = bfs(a)
    b = int(np.argmax(d1))
    path = []
    u = b
    while u != -1:
        path.append(u)
        u = par[u]
    path = path[::-1]
    return np.array([(xs[i], ys[i]) for i in path], float)


def tusk_centreline_profile(mask: np.ndarray, step: float = 2.0, sigma: float = 50.0, trim: float = 60.0) -> dict[str, np.ndarray]:
    """Centreline (root -> tip), arc length s, local width w(s) from the distance transform and
    signed curvature κ(s) after Gaussian smoothing of the resampled centreline."""
    dist = ndi.distance_transform_edt(mask)
    skel = morphology.skeletonize(mask)
    P = skeleton_longest_path(skel)
    # orient root -> tip: the root (pulp-cavity end) is the wider end
    hw0 = float(np.mean(ndi.map_coordinates(dist, [P[:40, 1], P[:40, 0]], order=1)))
    hw1 = float(np.mean(ndi.map_coordinates(dist, [P[-40:, 1], P[-40:, 0]], order=1)))
    if hw1 > hw0:
        P = P[::-1]
    seg = np.hypot(*np.diff(P, axis=0).T)
    s_raw = np.concatenate([[0.0], np.cumsum(seg)])
    s = np.arange(0.0, s_raw[-1], step)
    X = np.interp(s, s_raw, P[:, 0])
    Y = np.interp(s, s_raw, P[:, 1])
    Xs = ndi.gaussian_filter1d(X, sigma / step, mode="nearest")
    Ys = ndi.gaussian_filter1d(Y, sigma / step, mode="nearest")
    dx, dy = np.gradient(Xs, s), np.gradient(Ys, s)
    ddx, ddy = np.gradient(dx, s), np.gradient(dy, s)
    kappa = (dx * ddy - dy * ddx) / np.power(dx * dx + dy * dy, 1.5)
    w = 2.0 * ndi.map_coordinates(dist, [Y, X], order=1)
    keep = (s > trim) & (s < s[-1] - trim)
    return {"x": X, "y": Y, "xs": Xs, "ys": Ys, "s": s, "w": w, "kappa": kappa, "keep": keep, "mask": mask, "dist": dist}


# --------------------------------------------------------------------------- #
# Fits: log spiral about a searched pole vs constant-curvature circle (null)
# --------------------------------------------------------------------------- #
def fit_log_spiral_pole(x: np.ndarray, y: np.ndarray, pole: tuple[float, float]) -> dict[str, float]:
    """OLS of ln r on unwrapped θ about a given pole: returns k, r0, Pearson r, RMSE (px), ψ."""
    cx, cy = pole
    th = np.unwrap(np.arctan2(y - cy, x - cx))
    r = np.hypot(x - cx, y - cy)
    if np.any(r <= 0):
        return {"k": 0.0, "r0": 1.0, "pearson_r": 0.0, "rmse_px": float("inf"), "psi_deg": 90.0}
    lr = np.log(r)
    A = np.column_stack([np.ones_like(th), th])
    coef, _, _, _ = np.linalg.lstsq(A, lr, rcond=None)
    ln_r0, k = coef
    pred = np.exp(A @ coef)
    rmse = float(np.sqrt(np.mean((r - pred) ** 2)))
    pr = float(abs(np.corrcoef(th, lr)[0, 1])) if np.std(th) > 0 and np.std(lr) > 0 else 0.0
    psi = math.degrees(math.atan2(1.0, abs(k))) if k != 0 else 90.0
    return {"k": float(k), "r0": float(math.exp(ln_r0)), "pearson_r": pr, "rmse_px": rmse, "psi_deg": psi,
            "theta_span_deg": float(math.degrees(th.max() - th.min()))}


def search_log_spiral_pole(x: np.ndarray, y: np.ndarray, n_grid: int = 25, refine: bool = True) -> tuple[tuple[float, float], dict[str, float]]:
    """Grid (+ optional refinement) search of the pole minimising the radial RMSE of E1."""
    cx0, cy0 = float(np.mean(x)), float(np.mean(y))
    span = float(max(np.ptp(x), np.ptp(y)))
    best_pole, best = (cx0, cy0), fit_log_spiral_pole(x, y, (cx0, cy0))
    for scale in ([1.5, 0.5, 0.15] if refine else [1.5]):
        gx = np.linspace(best_pole[0] - scale * span, best_pole[0] + scale * span, n_grid)
        gy = np.linspace(best_pole[1] - scale * span, best_pole[1] + scale * span, n_grid)
        for cx in gx:
            for cy in gy:
                f = fit_log_spiral_pole(x, y, (cx, cy))
                if f["rmse_px"] < best["rmse_px"] and f["k"] > 0:
                    best_pole, best = (float(cx), float(cy)), f
    return best_pole, best


def fit_circle(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    """Kasa algebraic circle fit — the constant-curvature (ψ = 90°) null model (E5, w = const)."""
    A = np.column_stack([2 * x, 2 * y, np.ones_like(x)])
    bvec = x * x + y * y
    (cx, cy, c), _, _, _ = np.linalg.lstsq(A, bvec, rcond=None)
    R = math.sqrt(max(c + cx * cx + cy * cy, 1e-12))
    resid = np.hypot(x - cx, y - cy) - R
    return {"cx": float(cx), "cy": float(cy), "R": float(R), "rmse_px": float(np.sqrt(np.mean(resid ** 2)))}


def accretion_law_statistics(prof: dict[str, np.ndarray]) -> dict[str, float]:
    """T2: Thompson's accretion law κ·w = Δ along the tusk. Reports CV(κ w) vs CV(κ) (circle null),
    the sign structure of κ (plane spiral predicts a single sign) and Spearman ρ(κ, 1/w)."""
    k = prof["kappa"][prof["keep"]]
    w = prof["w"][prof["keep"]]
    kw = k * w
    cv_k = float(np.std(k) / abs(np.mean(k))) if np.mean(k) != 0 else float("inf")
    cv_kw = float(np.std(kw) / abs(np.mean(kw))) if np.mean(kw) != 0 else float("inf")
    sign_changes = int(np.sum(np.sign(k[1:]) != np.sign(k[:-1])))
    rho, p = stats.spearmanr(np.abs(k), 1.0 / w)
    frac_dominant = float(max(np.mean(k > 0), np.mean(k < 0)))
    return {"cv_kappa": cv_k, "cv_kappa_w": cv_kw, "kappa_sign_changes": sign_changes,
            "spearman_abs_kappa_vs_inv_w": float(rho), "spearman_p": float(p), "dominant_sign_fraction": frac_dominant,
            "delta_estimate": float(np.median(np.abs(kw))), "n_points": int(k.size)}


# --------------------------------------------------------------------------- #
# Computer vision: narwhal groove helix angle (structure tensor)
# --------------------------------------------------------------------------- #
def narwhal_groove_angles(img: np.ndarray, n_bands: int = 12, strip_frac: float = 0.35) -> dict[str, Any]:
    """Structure-tensor orientation of the spiral grooves in the low-foreshortening central strip
    of the tusk, per axial band: φ (from the axis), coherence, local silhouette width."""
    g = color.rgb2gray(img)
    H, W = g.shape
    mask = g > 0.35
    mask = morphology.remove_small_objects(mask, max_size=2000)
    mask = ndi.binary_fill_holes(mask)
    lab = measure.label(mask)
    props = sorted(measure.regionprops(lab), key=lambda p: -p.area)
    mask = lab == props[0].label
    widths = mask.sum(1)
    cols = np.arange(W)
    xc = np.array([cols[mask[yy]].mean() if widths[yy] > 0 else np.nan for yy in range(H)])
    hp = g - ndi.gaussian_filter(g, 8)
    Arr, Arc, Acc = structure_tensor(hp, sigma=6, order="rc")
    strip = mask.copy()
    for yy in range(H):
        if widths[yy] > 0:
            strip[yy] &= np.abs(cols - xc[yy]) < strip_frac * widths[yy] / 2
    bands = np.linspace(int(0.04 * H), int(0.96 * H), n_bands + 1).astype(int)
    rows = []
    for y0, y1 in zip(bands[:-1], bands[1:]):
        sub = strip[y0:y1]
        if sub.sum() < 50:
            continue
        J = np.array([[Arr[y0:y1][sub].sum(), Arc[y0:y1][sub].sum()], [Arc[y0:y1][sub].sum(), Acc[y0:y1][sub].sum()]])
        evals, evecs = np.linalg.eigh(J)
        gvec = evecs[:, 1]                      # dominant gradient direction (r, c)
        line = np.array([-gvec[1], gvec[0]])    # groove direction ⟂ gradient
        phi = math.degrees(math.atan2(abs(line[1]), abs(line[0])))
        coh = float((evals[1] - evals[0]) / (evals[1] + evals[0])) if (evals[1] + evals[0]) > 0 else 0.0
        rows.append({"y0": int(y0), "y1": int(y1), "phi_from_axis_deg": phi, "coherence": coh,
                     "width_px": float(np.median(widths[y0:y1])), "signed_line_deg": math.degrees(math.atan2(line[1], line[0]))})
    phis = np.array([r["phi_from_axis_deg"] for r in rows])
    cohs = np.array([r["coherence"] for r in rows])
    wts = cohs / cohs.sum()
    phi_mean = float(np.sum(wts * phis))
    phi_sd = float(np.sqrt(np.sum(wts * (phis - phi_mean) ** 2)))
    return {"bands": rows, "phi_mean_deg": phi_mean, "phi_sd_deg": phi_sd, "mask": mask, "strip": strip,
            "tan_alpha_equiv": kiladze_tan_alpha_from_phi(phi_mean),
            "alpha_equiv_deg": math.degrees(math.atan(kiladze_tan_alpha_from_phi(phi_mean))),
            "width_median_px": float(np.median(widths[widths > 0]))}


# --------------------------------------------------------------------------- #
# Analysis driver
# --------------------------------------------------------------------------- #
def run_analysis(full: bool = True) -> dict[str, Any]:
    ref = load_reference()
    kil = load_kiladze_table()
    mam = np.asarray(Image.open(MAMMOTH).convert("RGB")).astype(float) / 255.0
    nar = np.asarray(Image.open(NARWHAL).convert("RGB")).astype(float) / 255.0

    mask = segment_mammoth_tusk(mam)
    prof = tusk_centreline_profile(mask)
    keep = prof["keep"]
    x, y = prof["xs"][keep], prof["ys"][keep]
    pole, spiral = search_log_spiral_pole(x, y, refine=full)
    circle = fit_circle(x, y)
    accr = accretion_law_statistics(prof)
    # KS test: are the log-spiral radial residuals distributed like the circle residuals?
    th = np.unwrap(np.arctan2(y - pole[1], x - pole[0]))
    r_obs = np.hypot(x - pole[0], y - pole[1])
    r_fit = spiral["r0"] * np.exp(spiral["k"] * th)
    res_spiral = r_obs - r_fit
    res_circle = np.hypot(x - circle["cx"], y - circle["cy"]) - circle["R"]
    ks = stats.ks_2samp(res_spiral, res_circle)

    narw = narwhal_groove_angles(nar)
    alpha_ref, tan_ref = kil["alpha"], kil["tan_alpha"]
    z_alpha = (narw["alpha_equiv_deg"] - alpha_ref["mean"]) / alpha_ref["sd"]
    z_tan = (narw["tan_alpha_equiv"] - tan_ref["mean"]) / tan_ref["sd"]
    within_range = bool(tan_ref["min"] <= narw["tan_alpha_equiv"] <= tan_ref["max"])
    # inferred uniform twist (E7) in units of 1/diameter: τ d = π / tan α
    tau_d = math.pi / narw["tan_alpha_equiv"]
    pitch_over_d = 2 * math.pi / tau_d

    sv = st_venant_triangle_field()

    # model regimes (Thompson's spiral-angle table + narwhal twist) for the morphospace panel
    regimes = [
        {"label": "rodent incisor (w const, ψ=90°)", "lam": 0.90, "c": 0.0, "tau": 0.0},
        {"label": "Mastodon angustidens (low angle)", "lam": 0.97, "c": 0.25, "tau": 0.0},
        {"label": "mammoth tusk (plane log spiral)", "lam": 0.85, "c": 0.12, "tau": 0.0},
        {"label": "mammoth tusk + twist (double curvature)", "lam": 0.85, "c": 0.12, "tau": 0.0025},
        {"label": "Babirussa canine (strong twist)", "lam": 0.75, "c": 0.08, "tau": 0.006},
    ]
    for rg in regimes:
        g = grow_tooth(rg["lam"], rg["c"], rg["tau"], n_steps=700)
        rg["xyz"] = g["xyz"]
        rg["w"] = g["w"]
        rg["delta"] = g["delta"]
        rg["kappa_w_cv"] = float(np.std(g["kappa"] * g["w"]) / np.mean(g["kappa"] * g["w"]))
        rg["psi_deg_implied"] = math.degrees(math.asin(min(1.0, g["delta"] / rg["c"]))) if rg["c"] > 0 else 90.0

    verdict_parts = {
        "T1_log_spiral_beats_circle": bool(spiral["rmse_px"] < circle["rmse_px"]),
        "T1_single_sign_curvature": bool(accr["kappa_sign_changes"] == 0),
        "T2_accretion_law_cv_improves": bool(accr["cv_kappa_w"] < accr["cv_kappa"]),
        "T3_narwhal_within_kiladze_2sd": bool(abs(z_alpha) < 2.0),
        "T4_st_venant_six_sectors": bool(sv["sign_sectors"] == 6 and all(c == 1 for c in sv["face_sign_changes"])),
    }
    n_ok = sum(verdict_parts.values())
    verdict = "supported" if n_ok == len(verdict_parts) else ("mixed" if n_ok >= 2 else "refuted")

    headline = {
        "log_spiral_pearson_r": round(spiral["pearson_r"], 4),
        "log_spiral_rmse_px": round(spiral["rmse_px"], 2),
        "log_spiral_psi_deg": round(spiral["psi_deg"], 2),
        "log_spiral_W": round(whorl_expansion(spiral["psi_deg"]), 3),
        "circle_rmse_px": round(circle["rmse_px"], 2),
        "circle_R_px": round(circle["R"], 1),
        "ks_pvalue_spiral_vs_circle_residuals": round(float(ks.pvalue), 4),
        "cv_kappa": round(accr["cv_kappa"], 3),
        "cv_kappa_w": round(accr["cv_kappa_w"], 3),
        "kappa_sign_changes": accr["kappa_sign_changes"],
        "spearman_abs_kappa_vs_inv_w": round(accr["spearman_abs_kappa_vs_inv_w"], 3),
        "narwhal_phi_from_axis_deg": round(narw["phi_mean_deg"], 2),
        "narwhal_alpha_equiv_deg": round(narw["alpha_equiv_deg"], 2),
        "narwhal_tan_alpha_equiv": round(narw["tan_alpha_equiv"], 3),
        "kiladze_alpha_ref_deg": alpha_ref["mean"],
        "kiladze_alpha_sd_deg": round(alpha_ref["sd"], 2),
        "narwhal_z_alpha": round(z_alpha, 2),
        "narwhal_tan_alpha_within_published_range": within_range,
        "narwhal_pitch_over_diameter": round(pitch_over_d, 2),
        "st_venant_sign_sectors": sv["sign_sectors"],
        "st_venant_face_sign_changes": sv["face_sign_changes"],
        "st_venant_circle_warp_max": sv["circle_warp_max"],
    }
    result = {
        "claim_id": "T1917-032",
        "date_made": "1917",
        "dataset": {
            "name": "Mammuthus primigenius tusk (Commons) + Monodon monoceros tusk (Commons, torsion analogue) + Kiladze & Chernova 2018 Table 1",
            "doi": "10.1016/j.dib.2018.09.029",
            "sha256": {k: v["sha256"] for k, v in ref["datasets"].items()},
            "analogue_note": "narwhal specimen is a labelled analogue for the torsion component; Thompson 1917 names mammoth/elephant/Babirussa/beaver, not the narwhal",
        },
        "verdict": verdict,
        "verdict_components": verdict_parts,
        "headline_numbers": headline,
        "script": "thompson/claims/t1917-032_the-shapes-of-horns-and-of-teeth-or-tusks.py",
        "figure": "docs/media/t1917_032_the_shapes_of_horns_and_of_tee_comparison.png",
        "animation": "docs/media/t1917_032_the_shapes_of_horns_and_of_tee_growth.gif",
        "lean": "morphoeval/lean/loops/Thompson1917_T1917_032.lean",
        "pass": "full" if full else "pass1-baseline",
    }
    return {"ref": ref, "kil": kil, "mam": mam, "nar": nar, "mask": mask, "prof": prof, "pole": pole, "spiral": spiral,
            "circle": circle, "accr": accr, "res_spiral": res_spiral, "res_circle": res_circle, "narw": narw, "sv": sv,
            "regimes": regimes, "result": result, "full": full}


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #
def _project(xyz: np.ndarray, elev_deg: float = 25.0, azim_deg: float = -35.0) -> tuple[np.ndarray, np.ndarray]:
    e, a = math.radians(elev_deg), math.radians(azim_deg)
    Rz = np.array([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1]])
    Rx = np.array([[1, 0, 0], [0, math.cos(e), -math.sin(e)], [0, math.sin(e), math.cos(e)]])
    p = xyz @ Rz.T @ Rx.T
    return p[:, 0], p[:, 2]


def render_comparison(state: dict[str, Any]) -> None:
    prof, pole, spiral, circle = state["prof"], state["pole"], state["spiral"], state["circle"]
    h = state["result"]["headline_numbers"]
    fig, axes = plt.subplots(2, 3, figsize=(20, 11.5))
    fig.patch.set_facecolor("#0e1117")
    for ax in axes.ravel():
        ax.set_facecolor("#161b22")
        for sp in ax.spines.values():
            sp.set_color("#444")
        ax.tick_params(colors="#ccc", labelsize=8)
        ax.xaxis.label.set_color("#ddd")
        ax.yaxis.label.set_color("#ddd")
        ax.title.set_color("#fff")

    # (a) specimen + centreline + fits
    ax = axes[0, 0]
    ax.imshow(state["mam"])
    keep = prof["keep"]
    ax.plot(prof["xs"][keep], prof["ys"][keep], color="#00e5ff", lw=1.8, label="segmented centreline")
    th = np.unwrap(np.arctan2(prof["ys"][keep] - pole[1], prof["xs"][keep] - pole[0]))
    tt = np.linspace(th.min() - 0.6, th.max() + 0.6, 400)
    rr = spiral["r0"] * np.exp(spiral["k"] * tt)
    ax.plot(pole[0] + rr * np.cos(tt), pole[1] + rr * np.sin(tt), color="#ffd166", lw=1.5, ls="--",
            label=f"log spiral ψ={spiral['psi_deg']:.1f}°, RMSE={spiral['rmse_px']:.1f}px")
    cc = np.linspace(0, 2 * math.pi, 400)
    ax.plot(circle["cx"] + circle["R"] * np.cos(cc), circle["cy"] + circle["R"] * np.sin(cc), color="#ef476f", lw=1.0, ls=":",
            label=f"circle null R={circle['R']:.0f}px, RMSE={circle['rmse_px']:.1f}px")
    ax.plot(*pole, marker="+", color="#ffd166", ms=14, mew=2)
    ax.set_xlim(0, state["mam"].shape[1])
    ax.set_ylim(state["mam"].shape[0], 0)
    ax.set_title("(a) Mammuthus primigenius tusk (CC BY-SA 4.0) — E1 log spiral vs circle null")
    ax.legend(fontsize=7, loc="lower right", facecolor="#222", labelcolor="#eee")
    ax.set_xticks([])
    ax.set_yticks([])

    # (b) ln r vs theta
    ax = axes[0, 1]
    x, y = prof["xs"][keep], prof["ys"][keep]
    r = np.hypot(x - pole[0], y - pole[1])
    ax.scatter(np.degrees(th), np.log(r), s=4, color="#00e5ff", alpha=0.7, label="centreline ln r")
    ax.plot(np.degrees(th), math.log(spiral["r0"]) + spiral["k"] * th, color="#ffd166", lw=2,
            label=f"OLS: k={spiral['k']:.3f}, Pearson r={spiral['pearson_r']:.3f}")
    ax.set_xlabel("polar angle θ about fitted pole (deg)")
    ax.set_ylabel("ln r (px)")
    ax.set_title(f"(b) E1 linearity test — KS(spiral vs circle residuals) p={h['ks_pvalue_spiral_vs_circle_residuals']:.3f}")
    ax.legend(fontsize=8, facecolor="#222", labelcolor="#eee")
    ax.grid(alpha=0.2)

    # (c) accretion law
    ax = axes[0, 2]
    s = prof["s"][keep]
    k_ = prof["kappa"][keep]
    w_ = prof["w"][keep]
    ax.plot(s, k_ / np.max(np.abs(k_)), color="#ef476f", lw=1.5, label=f"κ(s)/max|κ|  (CV={h['cv_kappa']:.2f}, sign changes={h['kappa_sign_changes']})")
    ax.plot(s, (k_ * w_) / np.max(np.abs(k_ * w_)), color="#06d6a0", lw=1.5, label=f"κ·w / max  (CV={h['cv_kappa_w']:.2f})")
    ax.plot(s, w_ / np.max(w_), color="#aaa", lw=1.0, ls="--", label="width w(s)/max")
    ax.axhline(0, color="#666", lw=0.8)
    ax.set_xlabel("arc length s from root (px)")
    ax.set_title("(c) E4 accretion law κ·w = Δ along the tusk (root → tip)")
    ax.legend(fontsize=7, facecolor="#222", labelcolor="#eee", loc="lower left")
    ax.grid(alpha=0.2)

    # (d) narwhal
    ax = axes[1, 0]
    nar = state["nar"]
    narw = state["narw"]
    ax.imshow(nar)
    for b in narw["bands"]:
        yc = 0.5 * (b["y0"] + b["y1"])
        xc = nar.shape[1] / 2
        L = 0.25 * b["width_px"]
        ang = math.radians(b["signed_line_deg"])
        dr, dc = math.cos(ang) * L, math.sin(ang) * L
        ax.plot([xc - dc, xc + dc], [yc - dr, yc + dr], color="#ffd166", lw=2)
        ax.text(xc + 0.3 * b["width_px"], yc, f"{b['phi_from_axis_deg']:.0f}°", color="#fff", fontsize=7, va="center")
    ax.set_title(f"(d) Monodon monoceros tusk (CC BY 2.0, torsion analogue)\nφ={h['narwhal_phi_from_axis_deg']:.1f}° from axis → α≡{h['narwhal_alpha_equiv_deg']:.1f}° vs Kiladze 66.9±{h['kiladze_alpha_sd_deg']:.1f}° (z={h['narwhal_z_alpha']:+.2f})", fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])

    # (e) St Venant
    ax = axes[1, 1]
    sv = state["sv"]
    lv = np.linspace(-0.3, 0.3, 25)
    ax.contour(sv["X"], sv["Y"], np.nan_to_num(sv["U"], nan=0.0), levels=lv[lv > 0], colors="#ffd166", linewidths=0.8)
    ax.contour(sv["X"], sv["Y"], np.nan_to_num(sv["U"], nan=0.0), levels=lv[lv < 0], colors="#00e5ff", linewidths=0.8, linestyles="dotted")
    tri = np.array([[2 / 3, 0], [-1 / 3, 1 / math.sqrt(3)], [-1 / 3, -1 / math.sqrt(3)], [2 / 3, 0]])
    ax.plot(tri[:, 0], tri[:, 1], color="#fff", lw=1.5)
    ax.set_aspect("equal")
    ax.set_title(f"(e) E8 St Venant torsion curves u_z∝3x²y−y³ — {sv['sign_sectors']} sectors, face sign changes {sv['face_sign_changes']}", fontsize=9)
    # inset: Thompson's own Fig. 321
    try:
        plate = np.asarray(Image.open(FIG321).convert("L"))
        H, W = plate.shape
        crop = plate[int(0.40 * H):int(0.64 * H), int(0.02 * W):int(0.47 * W)]
        ins = ax.inset_axes([0.60, 0.02, 0.40, 0.40])
        ins.imshow(crop, cmap="gray")
        ins.set_xticks([])
        ins.set_yticks([])
        ins.set_title("Thompson 1917 Fig. 321", fontsize=7, color="#fff")
    except Exception:
        pass

    # (f) morphospace
    ax = axes[1, 2]
    cols = ["#aaa", "#06d6a0", "#ffd166", "#ef476f", "#c77dff"]
    for rg, cl in zip(state["regimes"], cols):
        px, py = _project(rg["xyz"])
        ax.plot(px, py, color=cl, lw=2, label=f"{rg['label']}  (λ={rg['lam']}, c={rg['c']}, τ={rg['tau']})")
    ax.set_aspect("equal")
    ax.set_title("(f) E4-E6 screw-accretion morphospace (tip at origin)")
    ax.legend(fontsize=7, facecolor="#222", labelcolor="#eee", loc="upper left")
    ax.set_xticks([])
    ax.set_yticks([])

    fig.suptitle(f"Thompson (1917) T1917-032 — teeth & tusks by spiral/torsion geometry — verdict: {state['result']['verdict'].upper()}"
                 f"  [{state['result']['pass']}]", color="#fff", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=110, facecolor=fig.get_facecolor())
    plt.close(fig)


def render_growth(state: dict[str, Any], n_frames: int = 36) -> None:
    """Animate screw-accretion growth (E4-E6) of a plane log-spiral tusk and a twisted tusk."""
    plane = grow_tooth(0.85, 0.12, 0.0, n_steps=700)
    twist = grow_tooth(0.85, 0.12, 0.0025, n_steps=700)
    incisor = grow_tooth(0.90, 0.0, 0.0, n_steps=700)
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.5))
    fig.patch.set_facecolor("#0e1117")
    for ax in axes:
        ax.set_facecolor("#161b22")
        ax.set_xticks([])
        ax.set_yticks([])
    extents = []
    for g in (plane, twist, incisor):
        px, py = _project(g["xyz"])
        extents.append((px.min(), px.max(), py.min(), py.max()))
    xmin = min(e[0] for e in extents) - 20
    xmax = max(e[1] for e in extents) + 20
    ymin = min(e[2] for e in extents) - 20
    ymax = max(e[3] for e in extents) + 20

    def draw(frame: int) -> None:
        n = int(10 + (len(plane["xyz"]) - 10) * (frame + 1) / n_frames)
        for ax in axes:
            ax.cla()
            ax.set_facecolor("#161b22")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlim(xmin, xmax)
            ax.set_ylim(ymin, ymax)
            ax.set_aspect("equal")
        ax = axes[0]
        for g, cl, lab in ((incisor, "#aaa", "incisor: w const → circle"), (plane, "#ffd166", "tusk: w ∝ s → log spiral")):
            px, py = _project(g["xyz"][:n])
            ax.plot(px, py, color=cl, lw=2 + 6 * g["w"][n - 1] / g["w"].max(), label=lab, solid_capstyle="round")
        ax.set_title(f"E4 differential accretion, step {n}/{len(plane['xyz'])}", color="#fff", fontsize=10)
        ax.legend(fontsize=8, facecolor="#222", labelcolor="#eee", loc="upper left")
        ax = axes[1]
        px, py = _project(twist["xyz"][:n])
        ax.plot(px, py, color="#ef476f", lw=2.5, label="tusk + twist τ → double curvature")
        px, py = _project(plane["xyz"][:n])
        ax.plot(px, py, color="#ffd166", lw=1, ls="--", alpha=0.6, label="plane spiral (τ = 0)")
        ax.set_title("E6 screw accretion Ω = κ b + τ n (Thompson p. 634)", color="#fff", fontsize=10)
        ax.legend(fontsize=8, facecolor="#222", labelcolor="#eee", loc="upper left")

    anim = FuncAnimation(fig, draw, frames=n_frames, interval=120)
    OUT_GIF.parent.mkdir(parents=True, exist_ok=True)
    anim.save(OUT_GIF, writer=PillowWriter(fps=8))
    plt.close(fig)


# --------------------------------------------------------------------------- #
# MorphoEval measurement hook & self-test
# --------------------------------------------------------------------------- #
def measure_morphoeval() -> list[dict[str, Any]]:
    state = run_analysis(full=True)
    h = state["result"]["headline_numbers"]
    kil = state["kil"]
    return [
        {"name": "log_spiral_pearson_r", "value": h["log_spiral_pearson_r"], "unit": "1", "method": "OLS ln r vs θ about searched pole (E1)"},
        {"name": "log_spiral_rmse_px", "value": h["log_spiral_rmse_px"], "unit": "px", "method": "radial RMSE of E1 fit vs circle null RMSE", "note": f"circle null RMSE = {h['circle_rmse_px']} px"},
        {"name": "log_spiral_psi_deg", "value": h["log_spiral_psi_deg"], "unit": "deg", "method": "atan(1/k); Thompson: mammoth 'high' spiral angle, Ovis ammon ~75"},
        {"name": "circle_null_rmse_px", "value": h["circle_rmse_px"], "unit": "px", "method": "Kasa circle fit (constant curvature control)"},
        {"name": "cv_kappa_w_accretion_law", "value": h["cv_kappa_w"], "unit": "1", "method": "CV of κ·w along centreline (E4) vs CV κ alone", "note": f"CV kappa = {h['cv_kappa']}"},
        {"name": "kappa_sign_changes", "value": h["kappa_sign_changes"], "unit": "count", "method": "sign changes of smoothed curvature root→tip (plane spiral predicts 0)"},
        {"name": "narwhal_alpha_equiv_deg", "value": h["narwhal_alpha_equiv_deg"], "unit": "deg", "method": "structure-tensor groove inclination → Kiladze chord angle π/(2 tan φ)",
         "reference_value": kil["alpha"]["mean"], "reference_source": "Kiladze & Chernova (2018) Data in Brief 21:1200, Table 1 (n=30, SD %.2f)" % kil["alpha"]["sd"]},
        {"name": "narwhal_tan_alpha_equiv", "value": h["narwhal_tan_alpha_equiv"], "unit": "1", "method": "π/(2 tan φ)", "reference_value": kil["tan_alpha"]["mean"],
         "reference_source": "Kiladze & Chernova (2018) Table 1 (range 1.65-3.22)"},
        {"name": "narwhal_z_alpha", "value": h["narwhal_z_alpha"], "unit": "SD", "method": "(α_measured − α_ref)/SD_ref"},
        {"name": "st_venant_sign_sectors", "value": h["st_venant_sign_sectors"], "unit": "count", "method": "sign alternations of 3x²y−y³ on a circle about the centroid", "reference_value": 6,
         "reference_source": "St Venant (1856) / Thompson Fig. 321: six alternating sectors"},
        {"name": "ks_pvalue_spiral_vs_circle_residuals", "value": h["ks_pvalue_spiral_vs_circle_residuals"], "unit": "1", "method": "two-sample KS test"},
    ]


def run_self_test() -> None:
    load_reference()
    # E1/E2 gnomonic self-similarity (mirrors Lean Law 1)
    W = whorl_expansion(75.0)
    assert abs(gnomon_radius(3.0, W, 2) * 3.0 - gnomon_radius(3.0, W, 1) * gnomon_radius(3.0, W, 1)) < 1e-9
    th = np.linspace(0, 2 * math.pi, 5)
    rr = log_spiral_r(1.0, 75.0, th)
    assert abs(rr[-1] / rr[0] - W) < 1e-9
    # E4: equal rates straight, swap mirrors (Lean Law 2)
    assert bend_moment(1.0, 1.0) == 0 and bend_moment(1.0, 0.8) == -bend_moment(0.8, 1.0)
    g_circ = grow_tooth(0.9, 0.0, 0.0, n_steps=400)
    assert np.std(g_circ["kappa"]) < 1e-12, "constant width must give constant curvature"
    g_spir = grow_tooth(0.9, 0.1, 0.0, n_steps=400)
    assert np.all(np.diff(g_spir["kappa"]) < 0), "gnomonic widening must make curvature fall"
    assert np.std(g_spir["kappa"] * g_spir["w"]) < 1e-12, "κ·w = Δ must hold exactly"
    # plane growth stays planar; twisted growth leaves the plane
    assert np.max(np.abs(g_spir["xyz"][:, 1])) < 1e-9
    g_tw = grow_tooth(0.9, 0.1, 0.005, n_steps=400)
    assert np.max(np.abs(g_tw["xyz"][:, 1])) > 1.0
    # E8/E9 St Venant (Lean Law 3)
    xs = np.array([0.3, -0.2, 0.5])
    ys = np.array([0.1, 0.4, -0.3])
    assert np.allclose(tri_warp(xs, -ys), -tri_warp(xs, ys)) and np.allclose(tri_warp(-xs, ys), tri_warp(xs, ys))
    assert np.allclose(tri_warp(xs, np.zeros(3)), 0) and np.allclose(ellipse_warp(1.0, 1.0, xs, ys), 0)
    sv = st_venant_triangle_field(n=121)
    assert sv["sign_sectors"] == 6 and sv["face_sign_changes"] == [1, 1, 1]
    # E7 helix (Lean Law 4)
    assert helix_slope(1.0, 0.5) < helix_slope(2.0, 0.5) and helix_slope(1.0, -0.5) == -helix_slope(1.0, 0.5)
    assert abs(kiladze_tan_alpha_from_twist(math.pi / (2.39 * 1.0), 1.0) - 2.39) < 1e-9
    # synthetic log spiral recovered by the pole search
    tt = np.linspace(0.3, 2.2 * math.pi, 300)
    k = 1 / math.tan(math.radians(78.0))
    xs_, ys_ = 100 + 20 * np.exp(k * tt) * np.cos(tt), 80 + 20 * np.exp(k * tt) * np.sin(tt)
    xs_ += RNG.normal(0, 0.2, xs_.shape)
    ys_ += RNG.normal(0, 0.2, ys_.shape)
    pole, f = search_log_spiral_pole(xs_, ys_, n_grid=15)
    assert abs(f["psi_deg"] - 78.0) < 1.5 and f["pearson_r"] > 0.99, f
    # specimen pipeline + determinism
    m1 = measure_morphoeval()
    m2 = measure_morphoeval()
    assert m1 == m2, "measure_morphoeval must be deterministic"
    assert all(m["value"] is not None for m in m1)
    print(f"self-test OK ({len(m1)} measurements; log-spiral r={m1[0]['value']}, narwhal α≡{m1[6]['value']}°)")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--pass1", action="store_true", help="Pass 1 baseline: coarse pole grid only")
    args = ap.parse_args(argv)
    if args.self_test:
        run_self_test()
        return
    state = run_analysis(full=not args.pass1)
    render_comparison(state)
    render_growth(state)
    RESULT_JSON.write_text(json.dumps(state["result"], indent=2) + "\n")
    print(f"Wrote {OUT_PNG.relative_to(ROOT)}, {OUT_GIF.relative_to(ROOT)}, {RESULT_JSON.relative_to(ROOT)}")
    print(f"Verdict: {state['result']['verdict']}")
    for k, v in state["result"]["headline_numbers"].items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()

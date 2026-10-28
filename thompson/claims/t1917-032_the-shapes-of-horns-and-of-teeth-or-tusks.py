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
  E5  w = c_r r ⇒ sin ψ = Δ/c_r  (⇔ w = c_s s ⇒ tan ψ = Δ/c_s, c_r = c_s/cos ψ);  w = const ⇒ circle w/Δ
  E6  Ω = κ b + τ n                           screw accretion (bend + twist) ⇒ helix / helico-spiral
  E7  tan φ = ρ τ ;  tan α = π/(τ d)          St Venant uniform-torsion helix; Kiladze chord ratio
  E8  u_z ∝ 3x²y − y³                         St Venant warping of the twisted equilateral prism (Fig. 321)
  E9  u_z ∝ (b² − a²) x y                     elliptic prism; circular section a = b does not warp

Specimens (SHA-256 pinned in the reference JSON, verified at import)
  * Mammuthus primigenius tusk, Museum Wiesbaden (Wikimedia Commons, CC BY-SA 4.0): species named
    by Thompson p. 634 — centreline / width profile; 2-D log-spiral vs circle fits; 2 × 2 screw-accretion
    model comparison {constant width, measured width w(s)} × {τ = 0 plane, τ free twisted} fitted to the
    projected centreline (Pass 2).
  * Monodon monoceros tusk close-up (Wikimedia Commons, CC BY 2.0): TORSION ANALOGUE (the narwhal
    is not named in Thompson's chapter) — groove helix angle vs Kiladze & Chernova (2018) and the
    uniform-torsion prediction tan α ∝ 1/d along the shaft.
  * Thompson's own Fig. 321 (IA scan leaf n642, public domain) — St Venant torsion curves.

Pass 2 additions (closed-loop remediation of audit 2026-10-28-t1917-032-…-audit):
  * 3-D screw-accretion fits (E4-E6) with Procrustes alignment of the projected model to the
    specimen centreline; nested comparison τ = 0 vs τ ≠ 0 and κ = Δ/w(s) vs κ = Δ/w̄ (ΔRMSE, ΔAIC)
  * systematic parameter-regime sweep (ψ, λ, taper c, twist τ) checking the Lean-proved monotonicities
  * negative / ablation controls: Archimedean spiral, shuffled-width permutation test for κ·w,
    τ = 0 ablation, circular-section St Venant null
  * statistics: block-bootstrap CIs (ψ, Pearson r, groove angle), one-sample KS on standardised
    residuals, tan α vs 1/d regression along the narwhal shaft
  * width-scaled smoothing, CLI overrides, library versions recorded in result.json

Lean 4 formalisation of E1/E2 (gnomonic self-similarity), E4/E5 (accretion bending law),
E8/E9 (St Venant anticlastic warping) and E7 (uniform-torsion helix) in
morphoeval/lean/loops/Thompson1917_T1917_032.lean (namespace MathOfBiology.Loop_226).

Usage
  python3 thompson/claims/t1917-032_the-shapes-of-horns-and-of-teeth-or-tusks.py --self-test
  python3 thompson/claims/t1917-032_the-shapes-of-horns-and-of-teeth-or-tusks.py            # Pass 2 (full)
  python3 thompson/claims/t1917-032_the-shapes-of-horns-and-of-teeth-or-tusks.py --pass1    # Pass 1 baseline
  python3 thompson/claims/t1917-032_the-shapes-of-horns-and-of-teeth-or-tusks.py --sigma-frac 0.8 --n-trials 12
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import scipy  # noqa: E402
import skimage  # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: E402
from PIL import Image  # noqa: E402
from scipy import ndimage as ndi  # noqa: E402
from scipy import stats  # noqa: E402
from scipy.optimize import minimize  # noqa: E402
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

DEFAULTS = {
    "sigma_frac": 0.65,     # centreline smoothing σ as a fraction of the median tusk width (Pass 2)
    "sigma_px_pass1": 50.0,  # Pass 1 hand-set smoothing scale
    "n_trials": 8,           # multi-start Nelder-Mead restarts per screw-model fit
    "ds_fit": 4.0,           # arc-length step (px) of the fitted screw model
    "n_boot": 200,           # bootstrap replicates
    "n_perm": 300,           # permutations for the shuffled-width control
}


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


def environment_versions() -> dict[str, str]:
    return {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
            "matplotlib": matplotlib.__version__, "Pillow": Image.__version__ if hasattr(Image, "__version__") else "n/a",
            "scikit-image": skimage.__version__}


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


def growth_contrast(lam: float) -> float:
    """Δ = 2(1−λ)/(1+λ) from the growth-rate ratio λ = v_min/v_max (E4)."""
    return 2 * (1 - lam) / (1 + lam)


def lambda_from_contrast(delta: float) -> float:
    """Inverse of `growth_contrast`: λ = (2 − Δ)/(2 + Δ)."""
    return (2 - delta) / (2 + delta)


def grow_tooth(lam: float, taper_c: float, tau: float, n_steps: int = 600, ds: float = 1.0,
               w_tip: float = 2.0, w_profile: np.ndarray | None = None) -> dict[str, np.ndarray]:
    """E4-E6 screw-accretion growth of a tooth/tusk from its tip.

    Each slab of arc ds is added at the growth zone with width w(s) = w_tip + c·s (or the supplied
    measured profile `w_profile`), bends the axis by dθ = Δ/w ds about the in-section bending axis
    (differential growth, E4) and twists the body frame about the tangent by τ ds (oblique fibres /
    torsional strain, E6).
      c = 0, τ = 0 -> circular arc (rodent incisor);  c > 0, τ = 0 -> plane log spiral (tusk);
      τ ≠ 0 -> helix / conical log helico-spiral ("curve of double curvature", p. 634).
    """
    delta = growth_contrast(lam)
    t = np.array([1.0, 0.0, 0.0])      # tangent (growth direction)
    b = np.array([0.0, 1.0, 0.0])      # bending axis (in-section, ⟂ growth gradient)
    n_vec = np.cross(t, b)             # section normal completing the frame
    pos = np.zeros(3)
    P = [pos.copy()]
    widths: list[float] = []
    kappas: list[float] = []
    for i in range(n_steps):
        w = float(w_profile[i]) if w_profile is not None else w_tip + taper_c * i * ds
        kappa = delta / w
        widths.append(w)
        kappas.append(kappa)
        dth = kappa * ds
        t, n_vec = (t * math.cos(dth) + n_vec * math.sin(dth), -t * math.sin(dth) + n_vec * math.cos(dth))
        dph = tau * ds
        b, n_vec = (b * math.cos(dph) + n_vec * math.sin(dph), -b * math.sin(dph) + n_vec * math.cos(dph))
        pos = pos + t * ds
        P.append(pos.copy())
    w_last = float(w_profile[min(n_steps, len(w_profile) - 1)]) if w_profile is not None else w_tip + taper_c * n_steps * ds
    widths.append(w_last)
    kappas.append(delta / w_last)
    return {"xyz": np.array(P), "w": np.array(widths), "kappa": np.array(kappas), "delta": delta}


def implied_psi_deg(delta: float, taper_c: float) -> float:
    """E5 gnomonic closure for an arc-length taper w = c_s·s: κ = Δ/(c_s s) and, on a log spiral,
    κ = tan ψ / s, hence tan ψ = Δ/c_s (equivalently sin ψ = Δ/c_r with the radial taper c_r = c_s/cos ψ).
    ψ = 90° (circle) when the section does not widen."""
    if taper_c <= 0:
        return 90.0
    return math.degrees(math.atan2(delta, taper_c))


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
    th = np.linspace(0, 2 * math.pi, 720, endpoint=False) + math.pi / 720.0   # avoid sampling the nodal rays exactly
    sgn = np.sign(tri_warp(0.25 * a * np.cos(th), 0.25 * a * np.sin(th)))
    sgn = sgn[sgn != 0]
    sectors = int(np.sum(sgn != np.roll(sgn, 1)))
    verts = [np.array([2 * a / 3, 0.0]), np.array([-a / 3, a / math.sqrt(3)]), np.array([-a / 3, -a / math.sqrt(3)])]
    face_changes = []
    for i in range(3):
        p0, p1 = verts[i], verts[(i + 1) % 3]
        tt = np.linspace(0.02, 0.98, 400)
        fx = p0[0] + (p1[0] - p0[0]) * tt
        fy = p0[1] + (p1[1] - p0[1]) * tt
        s = np.sign(tri_warp(fx, fy))
        face_changes.append(int(np.sum(s[1:] != s[:-1])))
    # ablation / null: the circular section (a = b) and a mildly elliptic one
    circ = float(np.nanmax(np.abs(ellipse_warp(1.0, 1.0, X, Y))))
    ell = float(np.nanmax(np.abs(ellipse_warp(1.0, 1.3, X, Y))))
    return {"X": X, "Y": Y, "U": U, "inside": inside, "sign_sectors": sectors, "face_sign_changes": face_changes,
            "circle_warp_max": circ, "ellipse_1p3_warp_max": ell}


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


def tusk_centreline_profile(mask: np.ndarray, step: float = 2.0, sigma: float | None = None,
                            sigma_frac: float = DEFAULTS["sigma_frac"], trim: float = 60.0) -> dict[str, Any]:
    """Centreline (root -> tip), arc length s, local width w(s) from the distance transform and
    signed curvature κ(s) after Gaussian smoothing of the resampled centreline.  The smoothing
    scale is `sigma` px if given (Pass 1) or `sigma_frac` × median width (Pass 2, width-scaled)."""
    dist = ndi.distance_transform_edt(mask)
    skel = morphology.skeletonize(mask)
    P = skeleton_longest_path(skel)
    hw0 = float(np.mean(ndi.map_coordinates(dist, [P[:40, 1], P[:40, 0]], order=1)))
    hw1 = float(np.mean(ndi.map_coordinates(dist, [P[-40:, 1], P[-40:, 0]], order=1)))
    if hw1 > hw0:
        P = P[::-1]
    seg = np.hypot(*np.diff(P, axis=0).T)
    s_raw = np.concatenate([[0.0], np.cumsum(seg)])
    s = np.arange(0.0, s_raw[-1], step)
    X = np.interp(s, s_raw, P[:, 0])
    Y = np.interp(s, s_raw, P[:, 1])
    w = 2.0 * ndi.map_coordinates(dist, [Y, X], order=1)
    sigma_px = float(sigma) if sigma is not None else float(sigma_frac * np.median(w))
    Xs = ndi.gaussian_filter1d(X, sigma_px / step, mode="nearest")
    Ys = ndi.gaussian_filter1d(Y, sigma_px / step, mode="nearest")
    dx, dy = np.gradient(Xs, s), np.gradient(Ys, s)
    ddx, ddy = np.gradient(dx, s), np.gradient(dy, s)
    kappa = (dx * ddy - dy * ddx) / np.power(dx * dx + dy * dy, 1.5)
    keep = (s > trim) & (s < s[-1] - trim)
    return {"x": X, "y": Y, "xs": Xs, "ys": Ys, "s": s, "w": w, "kappa": kappa, "keep": keep, "mask": mask,
            "dist": dist, "sigma_px": sigma_px}


# --------------------------------------------------------------------------- #
# 2-D fits: log spiral about a searched pole, circle null, Archimedean control
# --------------------------------------------------------------------------- #
def fit_log_spiral_pole(x: np.ndarray, y: np.ndarray, pole: tuple[float, float]) -> dict[str, float]:
    """OLS of ln r on unwrapped θ about a given pole: returns k, r0, Pearson r, RMSE (px), ψ."""
    cx, cy = pole
    th = np.unwrap(np.arctan2(y - cy, x - cx))
    r = np.hypot(x - cx, y - cy)
    if np.any(r <= 0):
        return {"k": 0.0, "r0": 1.0, "pearson_r": 0.0, "rmse_px": float("inf"), "psi_deg": 90.0, "theta_span_deg": 0.0}
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


def fit_archimedean(x: np.ndarray, y: np.ndarray, pole: tuple[float, float]) -> dict[str, float]:
    """Negative control: Archimedean spiral r = a + b θ about the same pole (fails gnomonic
    self-similarity, cf. Lean `archimedean_not_self_similar`)."""
    th = np.unwrap(np.arctan2(y - pole[1], x - pole[0]))
    r = np.hypot(x - pole[0], y - pole[1])
    A = np.column_stack([np.ones_like(th), th])
    coef, _, _, _ = np.linalg.lstsq(A, r, rcond=None)
    pred = A @ coef
    return {"a": float(coef[0]), "b": float(coef[1]), "rmse_px": float(np.sqrt(np.mean((r - pred) ** 2)))}


def block_bootstrap_log_spiral(x: np.ndarray, y: np.ndarray, pole: tuple[float, float], n_boot: int, rng: np.random.Generator,
                               block: int = 40) -> dict[str, Any]:
    """Moving-block bootstrap (correlated centreline points) of ψ and Pearson r at a fixed pole."""
    n = len(x)
    n_blocks = max(1, int(math.ceil(n / block)))
    psis, rs = [], []
    for _ in range(n_boot):
        starts = rng.integers(0, max(1, n - block), size=n_blocks)
        idx = np.concatenate([np.arange(s0, min(s0 + block, n)) for s0 in starts])[:n]
        f = fit_log_spiral_pole(x[idx], y[idx], pole)
        psis.append(f["psi_deg"])
        rs.append(f["pearson_r"])
    psis, rs = np.array(psis), np.array(rs)
    return {"psi_ci95": [float(np.percentile(psis, 2.5)), float(np.percentile(psis, 97.5))],
            "r_ci95": [float(np.percentile(rs, 2.5)), float(np.percentile(rs, 97.5))], "n_boot": n_boot}


def residual_normality(res: np.ndarray) -> dict[str, float]:
    """One-sample KS test of standardised residuals against N(0, 1) (replaces the Pass 1 two-sample misuse)."""
    z = (res - np.mean(res)) / (np.std(res) + 1e-12)
    ks = stats.kstest(z, "norm")
    return {"ks_stat": float(ks.statistic), "ks_pvalue": float(ks.pvalue)}


def accretion_law_statistics(prof: dict[str, Any], n_perm: int = 0, rng: np.random.Generator | None = None) -> dict[str, float]:
    """T2: Thompson's accretion law κ·w = Δ along the tusk. Reports CV(κ w) vs CV(κ) (circle null),
    the sign structure of κ (plane spiral predicts a single sign), Spearman ρ(|κ|, 1/w) and, when
    n_perm > 0, a shuffled-width permutation control: P(CV(κ·w_shuffled) ≤ CV(κ·w))."""
    k = prof["kappa"][prof["keep"]]
    w = prof["w"][prof["keep"]]
    kw = k * w
    cv_k = float(np.std(k) / abs(np.mean(k))) if np.mean(k) != 0 else float("inf")
    cv_kw = float(np.std(kw) / abs(np.mean(kw))) if np.mean(kw) != 0 else float("inf")
    sign_changes = int(np.sum(np.sign(k[1:]) != np.sign(k[:-1])))
    rho, p = stats.spearmanr(np.abs(k), 1.0 / w)
    out = {"cv_kappa": cv_k, "cv_kappa_w": cv_kw, "kappa_sign_changes": sign_changes,
           "spearman_abs_kappa_vs_inv_w": float(rho), "spearman_p": float(p),
           "dominant_sign_fraction": float(max(np.mean(k > 0), np.mean(k < 0))),
           "delta_estimate": float(np.median(np.abs(kw))), "n_points": int(k.size)}
    if n_perm > 0 and rng is not None:
        # block-shuffle the width profile so that the κ(s) ↔ w(s) coupling is destroyed but the width
        # marginal distribution and short-range autocorrelation are preserved
        # the accretion law predicts ρ(|κ|, 1/w) > 0 (thin where tightly curved): one-sided p = P(ρ_shuffled ≥ ρ_obs)
        block = 40
        cvs, rhos = [], []
        for _ in range(n_perm):
            n_blocks = int(math.ceil(len(w) / block))
            order = rng.permutation(n_blocks)
            w_sh = np.concatenate([w[b * block:(b + 1) * block] for b in order])[:len(w)]
            kw_sh = k * w_sh
            cvs.append(np.std(kw_sh) / abs(np.mean(kw_sh)) if np.mean(kw_sh) != 0 else np.inf)
            rhos.append(stats.spearmanr(np.abs(k), 1.0 / w_sh)[0])
        cvs, rhos = np.array(cvs), np.array(rhos)
        out["shuffled_width_cv_median"] = float(np.median(cvs))
        out["shuffled_width_cv_p_value"] = float(np.mean(cvs <= cv_kw))
        out["shuffled_width_spearman_p_value"] = float(np.mean(rhos >= rho))
        out["n_perm"] = int(n_perm)
    return out


# --------------------------------------------------------------------------- #
# 3-D screw-accretion fits to the projected centreline (Pass 2, E4-E6)
# --------------------------------------------------------------------------- #
def project_view(xyz: np.ndarray, polar: float, azim: float) -> np.ndarray:
    """Orthographic projection along the body-frame view direction (polar, azim)."""
    v = np.array([math.sin(polar) * math.cos(azim), math.sin(polar) * math.sin(azim), math.cos(polar)])
    a = np.array([1.0, 0.0, 0.0]) if abs(v[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e1 = np.cross(v, a)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(v, e1)
    return np.column_stack([xyz @ e1, xyz @ e2])


def procrustes_align(P: np.ndarray, Q: np.ndarray) -> tuple[float, np.ndarray]:
    """Rigid (rotation + translation, no scale) alignment of P onto Q by index correspondence;
    returns RMSE and the aligned P."""
    Pc, Qc = P - P.mean(0), Q - Q.mean(0)
    U, _, Vt = np.linalg.svd(Pc.T @ Qc)
    d = np.sign(np.linalg.det(U @ Vt))
    R = U @ np.diag([1.0, d]) @ Vt
    Pa = Pc @ R + Q.mean(0)
    return float(np.sqrt(np.mean(np.sum((Pa - Q) ** 2, axis=1)))), Pa


def prepare_fit_target(prof: dict[str, Any], ds: float) -> dict[str, Any]:
    """Resample the observed centreline (tip -> root) and width at the model's arc-length grid."""
    keep = prof["keep"]
    X, Y, s, w = prof["xs"][keep][::-1], prof["ys"][keep][::-1], prof["s"][keep], prof["w"][keep][::-1]
    s_tip = s[-1] - s[::-1]
    n_steps = int(s_tip[-1] / ds)
    s_model = np.arange(n_steps + 1) * ds
    obs = np.column_stack([np.interp(s_model, s_tip, X), np.interp(s_model, s_tip, Y)])
    w_meas = ndi.gaussian_filter1d(np.interp(s_model, s_tip, w), 5)
    return {"obs": obs, "w_meas": w_meas, "w_const": np.full_like(w_meas, float(np.mean(w_meas))), "n_steps": n_steps,
            "ds": ds, "s_model": s_model}


def fit_screw_model(target: dict[str, Any], w_profile: np.ndarray, with_tau: bool, n_trials: int,
                    rng: np.random.Generator) -> dict[str, Any]:
    """Multi-start Nelder-Mead fit of (Δ[, τ], view polar, view azim) of the screw-accretion model
    whose curvature follows κ(s) = Δ / w_profile(s); the projected model is Procrustes-aligned to
    the observed centreline. Returns the best fit, RMSE and an AIC-style score."""
    obs, n_steps, ds = target["obs"], target["n_steps"], target["ds"]

    def model_xyz(delta: float, tau: float) -> np.ndarray:
        return grow_tooth(lambda_from_contrast(delta), 0.0, tau, n_steps=n_steps, ds=ds, w_profile=w_profile)["xyz"]

    def cost(params: np.ndarray) -> float:
        if with_tau:
            delta, tau, polar, azim = params
        else:
            delta, polar, azim = params
            tau = 0.0
        if delta <= 0 or delta > 1.9:
            return 1e6
        return procrustes_align(project_view(model_xyz(delta, tau), polar, azim), obs)[0]

    best: tuple[float, np.ndarray] | None = None
    for _ in range(n_trials):
        x0 = [rng.uniform(0.02, 0.6)] + ([rng.uniform(-0.01, 0.01)] if with_tau else []) + [rng.uniform(0, math.pi), rng.uniform(0, 2 * math.pi)]
        res = minimize(cost, np.array(x0), method="Nelder-Mead", options={"maxiter": 500, "xatol": 1e-4, "fatol": 1e-3})
        if best is None or res.fun < best[0]:
            best = (float(res.fun), res.x)
    assert best is not None
    rmse, params = best
    if with_tau:
        delta, tau, polar, azim = params
    else:
        (delta, polar, azim), tau = params, 0.0
    xyz = model_xyz(float(delta), float(tau))
    _, aligned = procrustes_align(project_view(xyz, float(polar), float(azim)), obs)
    n = len(obs)
    k_par = 4 if with_tau else 3
    aic = n * math.log(max(rmse, 1e-9) ** 2) + 2 * k_par
    return {"rmse_px": rmse, "delta": float(delta), "lambda": lambda_from_contrast(float(delta)), "tau": float(tau),
            "polar": float(polar), "azim": float(azim), "aic": aic, "k_params": k_par, "aligned_xy": aligned, "xyz": xyz,
            "out_of_plane_px": float(np.ptp(xyz @ np.array([0.0, 1.0, 0.0])))}


def compare_screw_models(prof: dict[str, Any], n_trials: int, ds: float, rng: np.random.Generator) -> dict[str, Any]:
    """2 × 2 nested comparison: curvature law {κ = Δ/w̄ (constant width), κ = Δ/w(s) (measured width)}
    × twist {τ = 0 (plane), τ free (helico-spiral)}."""
    target = prepare_fit_target(prof, ds)
    fits: dict[str, dict[str, Any]] = {}
    for wname, wprof in (("const_width", target["w_const"]), ("measured_width", target["w_meas"])):
        for tname, with_tau in (("plane", False), ("twisted", True)):
            fits[f"{wname}__{tname}"] = fit_screw_model(target, wprof, with_tau, n_trials, rng)
    best_key = min(fits, key=lambda k: fits[k]["rmse_px"])
    summary = {k: {kk: (round(vv, 4) if isinstance(vv, float) else vv) for kk, vv in v.items() if kk not in ("aligned_xy", "xyz")}
               for k, v in fits.items()}
    twist_gain = {
        "const_width": fits["const_width__plane"]["rmse_px"] / fits["const_width__twisted"]["rmse_px"],
        "measured_width": fits["measured_width__plane"]["rmse_px"] / fits["measured_width__twisted"]["rmse_px"],
    }
    width_law_gain = {
        "plane": fits["const_width__plane"]["rmse_px"] / fits["measured_width__plane"]["rmse_px"],
        "twisted": fits["const_width__twisted"]["rmse_px"] / fits["measured_width__twisted"]["rmse_px"],
    }
    return {"fits": fits, "summary": summary, "best": best_key, "target": target,
            "twist_rmse_ratio": {k: round(v, 3) for k, v in twist_gain.items()},
            "width_law_rmse_ratio": {k: round(v, 3) for k, v in width_law_gain.items()},
            "delta_aic_twist_measured": round(fits["measured_width__plane"]["aic"] - fits["measured_width__twisted"]["aic"], 1),
            "delta_aic_twist_const": round(fits["const_width__plane"]["aic"] - fits["const_width__twisted"]["aic"], 1)}


# --------------------------------------------------------------------------- #
# Parameter-regime sweep (Pass 2): numerical check of the Lean-proved monotonicities
# --------------------------------------------------------------------------- #
def integrate_accretion_spiral(delta: float, c_s: float, s0: float = 1e-3, s_max: float = 1.0, n: int = 6000) -> np.ndarray:
    """Plane accretion ODE with a purely gnomonic section w = c_s·s (E4-E5): dθ/ds = Δ/(c_s s), so
    θ(s) = (Δ/c_s) ln(s/s0); the centreline is integrated by cumulative trapezoids on a log-spaced grid
    (fine near the tip where the curvature diverges).  Returns (x, y) with the tip at the origin."""
    s = np.geomspace(s0, s_max, n)
    th = (delta / c_s) * np.log(s / s0)
    x = np.concatenate([[0.0], np.cumsum(0.5 * (np.cos(th[1:]) + np.cos(th[:-1])) * np.diff(s))])
    y = np.concatenate([[0.0], np.cumsum(0.5 * (np.sin(th[1:]) + np.sin(th[:-1])) * np.diff(s))])
    return np.column_stack([x, y])


def frenet_torsion(xyz: np.ndarray, stride: int = 10) -> float:
    """Median discrete Frenet torsion tau = (r1 x r2) . r3 / |r1 x r2|^2 of a sampled space curve
    (r1, r2, r3 = first, second, third finite-difference derivatives)."""
    P = xyz[::stride]
    d1 = np.gradient(P, axis=0)
    d2 = np.gradient(d1, axis=0)
    d3 = np.gradient(d2, axis=0)
    cr = np.cross(d1, d2)
    denom = np.sum(cr * cr, axis=1)
    ok = denom > 1e-12
    if not np.any(ok):
        return 0.0
    return float(np.median(np.sum(cr[ok] * d3[ok], axis=1) / denom[ok]))


def parameter_sweep(ref: dict) -> dict[str, Any]:
    """Sweep ψ (via Δ/c), λ, taper c and twist τ through `grow_tooth` and verify:
      S1 κ(s) strictly decreasing whenever c > 0 (Lean `accretion_curvature_falls_with_width`),
      S2 κ(s) constant when c = 0 (Lean `accretion_constant_width_is_circle`),
      S3 τ = 0 is planar, τ > 0 is not, and the discrete Frenet torsion recovers τ (E6 helix),
      S4 ψ fitted on the generated plane curve tracks the E5 prediction tan ψ = Δ/c_s,
      S5 helix slope ρτ strictly increasing in ρ (Lean `helix_slope_monotone_in_radius`)."""
    sw = ref["parameter_regimes"]["sweep_used_in_script"]
    lams = sw["lambda_growth_ratio"]
    tapers = sw["taper_c"]
    taus = sw["twist_tau_per_diameter"]
    rows = []
    s1_ok, s2_ok = True, True
    for lam in lams:
        for c in tapers:
            g = grow_tooth(lam, c, 0.0, n_steps=400, ds=1.0, w_tip=4.0)
            dk = np.diff(g["kappa"])
            if c > 0:
                s1_ok &= bool(np.all(dk < 0))
            else:
                s2_ok &= bool(np.all(np.abs(dk) < 1e-12))
            rows.append({"lambda": lam, "taper_c": c, "delta": round(g["delta"], 4), "kappa_tip": round(float(g["kappa"][0]), 5),
                         "kappa_root": round(float(g["kappa"][-1]), 5), "psi_implied_deg": round(implied_psi_deg(g["delta"], c), 2)})
    # S3: τ = 0 is exactly planar; every τ > 0 leaves the plane, and the discrete Frenet torsion of the
    # generated curve recovers the imposed τ (E6: constant (κ, τ) screw accretion is a helix)
    excursions, torsion_est, planar_ok = [], [], True
    for tau in taus:
        tau_px = tau / 40.0                                                   # τ given per diameter (~40 px)
        g = grow_tooth(0.85, 0.0, tau_px, n_steps=600, ds=1.0, w_tip=40.0)   # constant width -> exact helix
        xyz = g["xyz"]
        excursions.append(float(np.ptp(xyz[:, 1])))
        planar_ok &= bool((np.ptp(xyz[:, 1]) < 1e-9) if tau == 0 else (np.ptp(xyz[:, 1]) > 1e-3))
        torsion_est.append(frenet_torsion(xyz, stride=10))
    tau_px_values = [t / 40.0 for t in taus]
    s3_ok = bool(planar_ok and np.allclose(torsion_est, tau_px_values, rtol=0.02, atol=1e-5))
    # S4: ψ recovered from the accretion ODE dθ/ds = Δ/(c_s s) integrated on a log-spaced arc grid from
    # s0 = 1e-3 (pole at the tip) vs the E5 prediction tan ψ = Δ/c_s
    psi_pred, psi_fit = [], []
    for lam in (0.5, 0.7, 0.8, 0.9):
        delta = growth_contrast(lam)
        for c in (0.05, 0.1, 0.2, 0.4):
            pts = integrate_accretion_spiral(delta, c)
            far = pts[len(pts) // 2:]
            f = fit_log_spiral_pole(far[:, 0], far[:, 1], (0.0, 0.0))
            psi_pred.append(implied_psi_deg(delta, c))
            psi_fit.append(f["psi_deg"])
    r_psi = float(np.corrcoef(psi_pred, psi_fit)[0, 1]) if len(psi_pred) > 2 else float("nan")
    rmse_psi = float(np.sqrt(np.mean((np.array(psi_pred) - np.array(psi_fit)) ** 2))) if psi_pred else float("nan")
    # S5: helix slope monotone in ρ
    rhos = np.linspace(1, 40, 20)
    s5_ok = bool(np.all(np.diff([helix_slope(r, 0.05) for r in rhos]) > 0))
    psi_table = {k: v for k, v in ref["parameter_regimes"]["spiral_angle_psi_deg_thompson"].items()}
    return {"rows": rows, "S1_curvature_falls_with_taper": s1_ok, "S2_constant_width_constant_curvature": s2_ok,
            "S3_twist_is_helical_torsion": s3_ok, "tau_values": taus, "tau_per_px": [round(t, 5) for t in tau_px_values],
            "torsion_recovered_per_px": [round(t, 5) for t in torsion_est], "out_of_plane_px": [round(e, 2) for e in excursions],
            "S4_psi_pred_vs_fit_pearson_r": round(r_psi, 4), "S4_psi_rmse_deg": round(rmse_psi, 2),
            "S4_pairs": [[round(a, 2), round(b, 2)] for a, b in zip(psi_pred, psi_fit)],
            "S5_helix_slope_monotone": s5_ok, "thompson_psi_table": psi_table}


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


def narwhal_torsion_regression(narw: dict[str, Any], n_boot: int, rng: np.random.Generator) -> dict[str, Any]:
    """E7 along-shaft test. Uniform twist τ predicts tan α = π/(τ d) ∝ 1/d (grooves more axial where the
    shaft is thinner); a constant helix angle predicts slope 0. Band-level OLS of tan α on 1/d with a
    bootstrap CI on the slope and on the coherence-weighted mean groove angle.  NOTE: camera obliquity
    along the shaft confounds this test in a single photograph; it is reported, not used for the verdict."""
    bands = narw["bands"]
    inv_d = np.array([1.0 / b["width_px"] for b in bands])
    tan_a = np.array([kiladze_tan_alpha_from_phi(b["phi_from_axis_deg"]) for b in bands])
    phis = np.array([b["phi_from_axis_deg"] for b in bands])
    cohs = np.array([b["coherence"] for b in bands])
    slope, intercept, r, p, se = stats.linregress(inv_d, tan_a)
    slopes, means = [], []
    n = len(bands)
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        if np.ptp(inv_d[idx]) < 1e-9:
            continue
        sl = stats.linregress(inv_d[idx], tan_a[idx]).slope
        slopes.append(sl)
        wts = cohs[idx] / cohs[idx].sum()
        means.append(float(np.sum(wts * phis[idx])))
    return {"slope_tan_alpha_vs_inv_d": float(slope), "slope_p": float(p), "slope_r": float(r),
            "slope_ci95": [float(np.percentile(slopes, 2.5)), float(np.percentile(slopes, 97.5))],
            "phi_mean_ci95_deg": [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))],
            "uniform_torsion_sign_expected": "positive", "confounded_by_camera_obliquity": True}


# --------------------------------------------------------------------------- #
# Analysis driver
# --------------------------------------------------------------------------- #
def run_analysis(full: bool = True, sigma_frac: float = DEFAULTS["sigma_frac"], n_trials: int = DEFAULTS["n_trials"],
                 n_boot: int = DEFAULTS["n_boot"], n_perm: int = DEFAULTS["n_perm"]) -> dict[str, Any]:
    rng = np.random.default_rng(SEED)
    ref = load_reference()
    kil = load_kiladze_table()
    mam = np.asarray(Image.open(MAMMOTH).convert("RGB")).astype(float) / 255.0
    nar = np.asarray(Image.open(NARWHAL).convert("RGB")).astype(float) / 255.0

    mask = segment_mammoth_tusk(mam)
    prof = tusk_centreline_profile(mask, sigma=None if full else DEFAULTS["sigma_px_pass1"], sigma_frac=sigma_frac)
    keep = prof["keep"]
    x, y = prof["xs"][keep], prof["ys"][keep]
    pole, spiral = search_log_spiral_pole(x, y, refine=full)
    circle = fit_circle(x, y)
    archi = fit_archimedean(x, y, pole)
    accr = accretion_law_statistics(prof, n_perm=n_perm if full else 0, rng=rng)
    th = np.unwrap(np.arctan2(y - pole[1], x - pole[0]))
    r_obs = np.hypot(x - pole[0], y - pole[1])
    res_spiral = r_obs - spiral["r0"] * np.exp(spiral["k"] * th)
    res_circle = np.hypot(x - circle["cx"], y - circle["cy"]) - circle["R"]
    ks2 = stats.ks_2samp(res_spiral, res_circle)              # kept for continuity with Pass 1 (not used for the verdict)
    norm_spiral = residual_normality(res_spiral)
    boot = block_bootstrap_log_spiral(x, y, pole, n_boot, rng) if full else None
    screw = compare_screw_models(prof, n_trials=n_trials, ds=DEFAULTS["ds_fit"], rng=rng) if full else None
    sweep = parameter_sweep(ref) if full else None

    narw = narwhal_groove_angles(nar)
    tors = narwhal_torsion_regression(narw, n_boot, rng) if full else None
    alpha_ref, tan_ref = kil["alpha"], kil["tan_alpha"]
    z_alpha = (narw["alpha_equiv_deg"] - alpha_ref["mean"]) / alpha_ref["sd"]
    within_range = bool(tan_ref["min"] <= narw["tan_alpha_equiv"] <= tan_ref["max"])
    tau_d = math.pi / narw["tan_alpha_equiv"]
    pitch_over_d = 2 * math.pi / tau_d

    sv = st_venant_triangle_field()

    regimes = [
        {"label": "rodent incisor (w const, ψ=90°)", "lam": 0.90, "c": 0.0, "tau": 0.0},
        {"label": "Mastodon angustidens (low angle)", "lam": 0.97, "c": 0.25, "tau": 0.0},
        {"label": "mammoth tusk (plane log spiral)", "lam": 0.85, "c": 0.12, "tau": 0.0},
        {"label": "mammoth tusk + twist (double curvature)", "lam": 0.85, "c": 0.12, "tau": 0.0025},
        {"label": "Babirussa canine (strong twist)", "lam": 0.75, "c": 0.08, "tau": 0.006},
    ]
    for rg in regimes:
        g = grow_tooth(rg["lam"], rg["c"], rg["tau"], n_steps=700)
        rg["xyz"], rg["w"], rg["delta"] = g["xyz"], g["w"], g["delta"]
        rg["psi_deg_implied"] = implied_psi_deg(g["delta"], rg["c"])

    verdict_parts: dict[str, bool] = {
        "T1_log_spiral_beats_circle": bool(spiral["rmse_px"] < circle["rmse_px"]),
        "T1_single_sign_curvature": bool(accr["kappa_sign_changes"] == 0),
        "T2_curvature_rises_where_width_falls": bool(accr["spearman_abs_kappa_vs_inv_w"] > 0 and accr["spearman_p"] < 0.05),
        "T3_narwhal_within_kiladze_2sd": bool(abs(z_alpha) < 2.0),
        "T4_st_venant_six_sectors": bool(sv["sign_sectors"] == 6 and all(c == 1 for c in sv["face_sign_changes"])),
    }
    if full and screw is not None and sweep is not None:
        f = screw["fits"]
        verdict_parts["T1b_log_spiral_beats_archimedean"] = bool(spiral["rmse_px"] < archi["rmse_px"])
        verdict_parts["T2b_kappa_width_coupling_beats_shuffled_width"] = bool(accr.get("shuffled_width_spearman_p_value", 1.0) < 0.05)
        verdict_parts["T2c_measured_width_law_beats_const_width_3d"] = bool(f["measured_width__twisted"]["rmse_px"] < f["const_width__twisted"]["rmse_px"])
        verdict_parts["T3b_twist_improves_3d_fit"] = bool(screw["delta_aic_twist_measured"] > 10 and screw["delta_aic_twist_const"] > 10)
        verdict_parts["T5_sweep_monotonicities_hold"] = bool(sweep["S1_curvature_falls_with_taper"] and sweep["S2_constant_width_constant_curvature"]
                                                             and sweep["S3_twist_is_helical_torsion"] and sweep["S5_helix_slope_monotone"]
                                                             and sweep["S4_psi_pred_vs_fit_pearson_r"] > 0.95)
    n_ok = sum(verdict_parts.values())
    frac = n_ok / len(verdict_parts)
    verdict = "supported" if frac >= 0.85 else ("mixed" if frac >= 0.4 else "refuted")

    headline: dict[str, Any] = {
        "log_spiral_pearson_r": round(spiral["pearson_r"], 4),
        "log_spiral_rmse_px": round(spiral["rmse_px"], 2),
        "log_spiral_psi_deg": round(spiral["psi_deg"], 2),
        "log_spiral_W": round(whorl_expansion(spiral["psi_deg"]), 3),
        "circle_rmse_px": round(circle["rmse_px"], 2),
        "circle_R_px": round(circle["R"], 1),
        "archimedean_rmse_px": round(archi["rmse_px"], 2),
        "ks_pvalue_spiral_vs_circle_residuals": round(float(ks2.pvalue), 4),
        "ks_pvalue_spiral_residual_normality": round(norm_spiral["ks_pvalue"], 4),
        "cv_kappa": round(accr["cv_kappa"], 3),
        "cv_kappa_w": round(accr["cv_kappa_w"], 3),
        "kappa_sign_changes": accr["kappa_sign_changes"],
        "spearman_abs_kappa_vs_inv_w": round(accr["spearman_abs_kappa_vs_inv_w"], 3),
        "centreline_sigma_px": round(prof["sigma_px"], 1),
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
        "st_venant_ellipse_1p3_warp_max": round(sv["ellipse_1p3_warp_max"], 4),
    }
    if full and screw is not None and boot is not None and sweep is not None and tors is not None:
        f = screw["fits"]
        headline.update({
            "psi_ci95_deg": [round(v, 2) for v in boot["psi_ci95"]],
            "pearson_r_ci95": [round(v, 4) for v in boot["r_ci95"]],
            "shuffled_width_cv_median": round(accr["shuffled_width_cv_median"], 3),
            "shuffled_width_cv_p_value": round(accr["shuffled_width_cv_p_value"], 4),
            "shuffled_width_spearman_p_value": round(accr["shuffled_width_spearman_p_value"], 4),
            "spearman_p": round(accr["spearman_p"], 4),
            "screw_rmse_const_width_plane_px": round(f["const_width__plane"]["rmse_px"], 2),
            "screw_rmse_const_width_twisted_px": round(f["const_width__twisted"]["rmse_px"], 2),
            "screw_rmse_measured_width_plane_px": round(f["measured_width__plane"]["rmse_px"], 2),
            "screw_rmse_measured_width_twisted_px": round(f["measured_width__twisted"]["rmse_px"], 2),
            "screw_best_model": screw["best"],
            "screw_best_delta": round(f[screw["best"]]["delta"], 4),
            "screw_best_lambda": round(f[screw["best"]]["lambda"], 4),
            "screw_best_tau_per_px": round(f[screw["best"]]["tau"], 5),
            "screw_best_pitch_px": round(abs(2 * math.pi / f[screw["best"]]["tau"]), 1) if f[screw["best"]]["tau"] != 0 else None,
            "twist_rmse_ratio": screw["twist_rmse_ratio"],
            "width_law_rmse_ratio": screw["width_law_rmse_ratio"],
            "delta_aic_twist_measured_width": screw["delta_aic_twist_measured"],
            "delta_aic_twist_const_width": screw["delta_aic_twist_const"],
            "narwhal_phi_ci95_deg": [round(v, 2) for v in tors["phi_mean_ci95_deg"]],
            "narwhal_tan_alpha_vs_inv_d_slope": round(tors["slope_tan_alpha_vs_inv_d"], 2),
            "narwhal_tan_alpha_vs_inv_d_slope_ci95": [round(v, 2) for v in tors["slope_ci95"]],
            "narwhal_tan_alpha_vs_inv_d_p": round(tors["slope_p"], 4),
            "sweep_S4_psi_pred_vs_fit_r": sweep["S4_psi_pred_vs_fit_pearson_r"],
            "sweep_S4_psi_rmse_deg": sweep["S4_psi_rmse_deg"],
        })
    result: dict[str, Any] = {
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
        "verdict_rule": "supported if >= 85% of components pass, mixed if >= 40%, else refuted",
        "headline_numbers": headline,
        "interpretation": (
            "Twist (helical component) is strongly supported on the mammoth specimen and the narwhal groove angle matches the "
            "published value; the accretion coupling kappa = Delta / w(s) is NOT supported on this 2-D projection (|kappa| does not rise where the "
            "section thins; the constant-width helix fits better), so Thompson's claim holds for the torsion/double-curvature part "
            "and the St Venant torsion curves, and only weakly for the gnomonic log-spiral part of a single tusk photograph." if full else
            "Pass 1 baseline: 2-D plane-spiral and circle fits only."),
        "script": "thompson/claims/t1917-032_the-shapes-of-horns-and-of-teeth-or-tusks.py",
        "figure": "docs/media/t1917_032_the_shapes_of_horns_and_of_tee_comparison.png",
        "animation": "docs/media/t1917_032_the_shapes_of_horns_and_of_tee_growth.gif",
        "lean": "morphoeval/lean/loops/Thompson1917_T1917_032.lean",
        "pass": "pass2-full" if full else "pass1-baseline",
        "environment": environment_versions(),
        "parameters": {"seed": SEED, "sigma_frac": sigma_frac if full else None, "sigma_px_pass1": None if full else DEFAULTS["sigma_px_pass1"],
                       "n_trials": n_trials if full else 0, "ds_fit": DEFAULTS["ds_fit"], "n_boot": n_boot if full else 0, "n_perm": n_perm if full else 0},
    }
    if full and screw is not None and sweep is not None:
        result["screw_model_comparison"] = screw["summary"]
        result["regime_sweep"] = {k: v for k, v in sweep.items() if k != "rows"}
        result["regime_sweep"]["rows"] = sweep["rows"]
    return {"ref": ref, "kil": kil, "mam": mam, "nar": nar, "mask": mask, "prof": prof, "pole": pole, "spiral": spiral,
            "circle": circle, "archi": archi, "accr": accr, "res_spiral": res_spiral, "res_circle": res_circle, "boot": boot,
            "screw": screw, "sweep": sweep, "narw": narw, "tors": tors, "sv": sv, "regimes": regimes, "result": result, "full": full}


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #
def _project(xyz: np.ndarray, elev_deg: float = 25.0, azim_deg: float = -35.0) -> tuple[np.ndarray, np.ndarray]:
    e, a = math.radians(elev_deg), math.radians(azim_deg)
    Rz = np.array([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1]])
    Rx = np.array([[1, 0, 0], [0, math.cos(e), -math.sin(e)], [0, math.sin(e), math.cos(e)]])
    p = xyz @ Rz.T @ Rx.T
    return p[:, 0], p[:, 2]


def _style(ax: plt.Axes) -> None:
    ax.set_facecolor("#161b22")
    for sp in ax.spines.values():
        sp.set_color("#444")
    ax.tick_params(colors="#ccc", labelsize=8)
    ax.xaxis.label.set_color("#ddd")
    ax.yaxis.label.set_color("#ddd")
    ax.title.set_color("#fff")


def _legend(ax: plt.Axes, **kw: Any) -> None:
    kw.setdefault("fontsize", 7)
    ax.legend(facecolor="#222", labelcolor="#eee", **kw)


def render_comparison(state: dict[str, Any]) -> None:
    prof, pole, spiral, circle = state["prof"], state["pole"], state["spiral"], state["circle"]
    h = state["result"]["headline_numbers"]
    full = state["full"]
    nrows = 3 if full else 2
    fig, axes = plt.subplots(nrows, 3, figsize=(21, 6.2 * nrows))
    fig.patch.set_facecolor("#0e1117")
    for ax in axes.ravel():
        _style(ax)
    keep = prof["keep"]
    x, y = prof["xs"][keep], prof["ys"][keep]
    th = np.unwrap(np.arctan2(y - pole[1], x - pole[0]))

    # (a) specimen + 2-D fits
    ax = axes[0, 0]
    ax.imshow(state["mam"])
    ax.plot(x, y, color="#00e5ff", lw=1.8, label="segmented centreline")
    tt = np.linspace(th.min() - 0.6, th.max() + 0.6, 400)
    rr = spiral["r0"] * np.exp(spiral["k"] * tt)
    ax.plot(pole[0] + rr * np.cos(tt), pole[1] + rr * np.sin(tt), color="#ffd166", lw=1.5, ls="--",
            label=f"log spiral ψ={spiral['psi_deg']:.1f}°, RMSE={spiral['rmse_px']:.1f}px")
    cc = np.linspace(0, 2 * math.pi, 400)
    ax.plot(circle["cx"] + circle["R"] * np.cos(cc), circle["cy"] + circle["R"] * np.sin(cc), color="#ef476f", lw=1.0, ls=":",
            label=f"circle null R={circle['R']:.0f}px, RMSE={circle['rmse_px']:.1f}px")
    if full and state["screw"] is not None:
        best = state["screw"]["fits"][state["screw"]["best"]]
        ax.plot(best["aligned_xy"][:, 0], best["aligned_xy"][:, 1], color="#06d6a0", lw=2.2,
                label=f"3-D screw fit [{state['screw']['best']}] RMSE={best['rmse_px']:.1f}px, τ={best['tau']:.4f}/px")
    ax.plot(*pole, marker="+", color="#ffd166", ms=14, mew=2)
    ax.set_xlim(0, state["mam"].shape[1])
    ax.set_ylim(state["mam"].shape[0], 0)
    ax.set_title("(a) Mammuthus primigenius tusk (CC BY-SA 4.0) — E1 log spiral vs circle null" + (" + E6 screw fit" if full else ""), fontsize=10)
    _legend(ax, loc="lower right")
    ax.set_xticks([])
    ax.set_yticks([])

    # (b) ln r vs θ
    ax = axes[0, 1]
    r = np.hypot(x - pole[0], y - pole[1])
    ax.scatter(np.degrees(th), np.log(r), s=4, color="#00e5ff", alpha=0.7, label="centreline ln r")
    lab = f"OLS: k={spiral['k']:.3f}, Pearson r={spiral['pearson_r']:.3f}"
    if full and state["boot"] is not None:
        lab += f"\nbootstrap 95% CI ψ∈[{h['psi_ci95_deg'][0]:.1f}, {h['psi_ci95_deg'][1]:.1f}]°, r∈[{h['pearson_r_ci95'][0]:.3f}, {h['pearson_r_ci95'][1]:.3f}]"
    ax.plot(np.degrees(th), math.log(spiral["r0"]) + spiral["k"] * th, color="#ffd166", lw=2, label=lab)
    ax.set_xlabel("polar angle θ about fitted pole (deg)")
    ax.set_ylabel("ln r (px)")
    ttl = f"(b) E1 linearity — residual normality KS p={h['ks_pvalue_spiral_residual_normality']:.3f}; Archimedean control RMSE={h['archimedean_rmse_px']:.1f}px"
    ax.set_title(ttl, fontsize=9)
    _legend(ax, fontsize=8)
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
    ttl = f"(c) E4 accretion law κ·w = Δ along the tusk (σ={h['centreline_sigma_px']:.0f}px)"
    if full:
        ttl += (f"\nSpearman ρ(|κ|, 1/w)={h['spearman_abs_kappa_vs_inv_w']:+.3f} (accretion law predicts > 0); "
                f"block-shuffled-width control p={h['shuffled_width_spearman_p_value']:.3f}")
    ax.set_title(ttl, fontsize=9)
    _legend(ax, loc="lower left")
    ax.grid(alpha=0.2)

    # (d) narwhal
    ax = axes[1, 0]
    nar, narw = state["nar"], state["narw"]
    ax.imshow(nar)
    for b in narw["bands"]:
        yc = 0.5 * (b["y0"] + b["y1"])
        xc = nar.shape[1] / 2
        L = 0.25 * b["width_px"]
        ang = math.radians(b["signed_line_deg"])
        dr, dc = math.cos(ang) * L, math.sin(ang) * L
        ax.plot([xc - dc, xc + dc], [yc - dr, yc + dr], color="#ffd166", lw=2)
        ax.text(xc + 0.3 * b["width_px"], yc, f"{b['phi_from_axis_deg']:.0f}°", color="#fff", fontsize=7, va="center")
    ci = f" (95% CI {h['narwhal_phi_ci95_deg'][0]:.1f}-{h['narwhal_phi_ci95_deg'][1]:.1f}°)" if full else ""
    ax.set_title(f"(d) Monodon monoceros tusk (CC BY 2.0, torsion analogue)\nφ={h['narwhal_phi_from_axis_deg']:.1f}° from axis{ci} → α≡{h['narwhal_alpha_equiv_deg']:.1f}° "
                 f"vs Kiladze 66.9±{h['kiladze_alpha_sd_deg']:.1f}° (z={h['narwhal_z_alpha']:+.2f})", fontsize=9)
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
    ax.set_title(f"(e) E8 St Venant torsion curves u_z∝3x²y−y³ — {sv['sign_sectors']} sectors, face sign changes {sv['face_sign_changes']}\n"
                 f"circular-section null max|u_z|={sv['circle_warp_max']:.0f}; ellipse b/a=1.3 max|u_z|={sv['ellipse_1p3_warp_max']:.2f}", fontsize=9)
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

    # (f) morphospace regimes
    ax = axes[1, 2]
    cols = ["#aaa", "#06d6a0", "#ffd166", "#ef476f", "#c77dff"]
    for i, (rg, cl) in enumerate(zip(state["regimes"], cols)):
        px, py = _project(rg["xyz"])
        ext = max(np.ptp(px), np.ptp(py), 1e-9)
        px, py = (px - px.min()) / ext + 1.15 * i, (py - py.min()) / ext   # each form normalised to unit extent, laid out side by side
        ax.plot(px, py, color=cl, lw=2, label=f"{rg['label']}  (λ={rg['lam']}, c={rg['c']}, τ={rg['tau']}, ψ_E5={rg['psi_deg_implied']:.0f}°)")
        ax.plot(px[0], py[0], marker="o", color=cl, ms=4)
    ax.set_aspect("equal")
    ax.set_ylim(-0.1, 1.6)
    ax.set_title("(f) E4-E6 screw-accretion regimes (each normalised to unit extent; dot = tip)")
    _legend(ax, loc="upper left")
    ax.set_xticks([])
    ax.set_yticks([])

    if full and state["screw"] is not None and state["sweep"] is not None and state["tors"] is not None:
        screw, sweep, tors = state["screw"], state["sweep"], state["tors"]
        # (g) 2x2 screw model comparison
        ax = axes[2, 0]
        keys = ["const_width__plane", "const_width__twisted", "measured_width__plane", "measured_width__twisted"]
        vals = [screw["fits"][k]["rmse_px"] for k in keys]
        colors = ["#ef476f", "#ff9f1c", "#118ab2", "#06d6a0"]
        ax.bar(range(4), vals, color=colors)
        ax.axhline(circle["rmse_px"], color="#aaa", ls=":", lw=1, label=f"2-D circle fit {circle['rmse_px']:.1f}px")
        ax.axhline(spiral["rmse_px"], color="#ffd166", ls="--", lw=1, label=f"2-D log spiral {spiral['rmse_px']:.1f}px")
        ax.set_xticks(range(4))
        ax.set_xticklabels(["κ=Δ/w̄\nτ=0", "κ=Δ/w̄\nτ free", "κ=Δ/w(s)\nτ=0", "κ=Δ/w(s)\nτ free"], fontsize=8)
        for i, v in enumerate(vals):
            ax.text(i, v + 0.5, f"{v:.1f}", ha="center", color="#fff", fontsize=8)
        ax.set_ylabel("Procrustes RMSE (px)")
        ax.set_title(f"(g) E4-E6 3-D screw fits: twist ΔAIC={h['delta_aic_twist_measured_width']:.0f} (w(s)), {h['delta_aic_twist_const_width']:.0f} (w̄);\n"
                     f"best = {screw['best']}  λ={h['screw_best_lambda']:.3f}, pitch={h['screw_best_pitch_px']} px", fontsize=9)
        _legend(ax, loc="upper right")
        ax.grid(alpha=0.2, axis="y")

        # (h) parameter sweep
        ax = axes[2, 1]
        pairs = np.array(sweep["S4_pairs"])
        if len(pairs):
            ax.scatter(pairs[:, 0], pairs[:, 1], color="#ffd166", s=30, label=f"S4 ψ fit vs E5 prediction (r={sweep['S4_psi_pred_vs_fit_pearson_r']:.3f}, RMSE={sweep['S4_psi_rmse_deg']:.1f}°)")
            lim = [min(pairs.min(), 0), 90]
            ax.plot(lim, lim, color="#666", lw=1)
        ax2 = ax.twinx()
        ax2.plot([1e3 * t for t in sweep["tau_per_px"]], [1e3 * t for t in sweep["torsion_recovered_per_px"]], color="#ef476f", marker="o", lw=1.5,
                 label="S3 Frenet torsion of generated curve vs imposed τ (×10⁻³/px)")
        ax2.tick_params(colors="#ccc", labelsize=8)
        ax2.set_ylabel("recovered torsion (×10⁻³ px⁻¹)", color="#ef476f")
        ax.set_xlabel("E5 predicted ψ = atan(Δ/c_s) (deg)  |  imposed τ (×10⁻³ px⁻¹)")
        ax.set_ylabel("fitted ψ (deg)")
        ax.set_title(f"(h) regime sweep — S1 κ↓ with taper: {sweep['S1_curvature_falls_with_taper']}, S2 const-width circle: {sweep['S2_constant_width_constant_curvature']},\n"
                     f"S3 twist = helical torsion: {sweep['S3_twist_is_helical_torsion']}, S5 helix slope↑ρ: {sweep['S5_helix_slope_monotone']}  ({len(sweep['rows'])} (λ, c) regimes)", fontsize=9)
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, fontsize=7, facecolor="#222", labelcolor="#eee", loc="lower right")
        ax.grid(alpha=0.2)

        # (i) narwhal torsion regression
        ax = axes[2, 2]
        bands = narw["bands"]
        inv_d = np.array([1.0 / b["width_px"] for b in bands])
        tan_a = np.array([kiladze_tan_alpha_from_phi(b["phi_from_axis_deg"]) for b in bands])
        ax.scatter(1e3 * inv_d, tan_a, color="#00e5ff", s=30, label="per-band tan α (centre strip)")
        xx = np.linspace(inv_d.min(), inv_d.max(), 50)
        ax.plot(1e3 * xx, tors["slope_tan_alpha_vs_inv_d"] * xx + (tan_a.mean() - tors["slope_tan_alpha_vs_inv_d"] * inv_d.mean()), color="#ffd166", lw=1.5,
                label=f"OLS slope={tors['slope_tan_alpha_vs_inv_d']:.0f} (95% CI {tors['slope_ci95'][0]:.0f}..{tors['slope_ci95'][1]:.0f}), p={tors['slope_p']:.3f}")
        kil = state["kil"]["tan_alpha"]
        ax.axhspan(kil["mean"] - kil["sd"], kil["mean"] + kil["sd"], color="#06d6a0", alpha=0.15, label=f"Kiladze 2018 tan α = {kil['mean']} ± {kil['sd']:.2f} (SD, n=30)")
        ax.axhline(kil["mean"], color="#06d6a0", lw=1)
        ax.set_xlabel("1/d (×10⁻³ px⁻¹) along the shaft")
        ax.set_ylabel("tan α (Kiladze chord ratio)")
        ax.set_title("(i) E7 uniform-torsion test: tan α = π/(τ d) predicts slope > 0\n(camera obliquity confounds the along-shaft trend; not used for the verdict)", fontsize=9)
        _legend(ax, loc="upper left")
        ax.grid(alpha=0.2)

    fig.suptitle(f"Thompson (1917) T1917-032 — teeth & tusks by spiral/torsion geometry — verdict: {state['result']['verdict'].upper()}"
                 f"  [{state['result']['pass']}]", color="#fff", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=105, facecolor=fig.get_facecolor())
    plt.close(fig)


def render_growth(state: dict[str, Any], n_frames: int = 36) -> None:
    """Animate screw-accretion growth (E4-E6): incisor circle vs log-spiral tusk, plane vs twisted
    tusk and — in Pass 2 — the fitted twisted model growing over the mammoth specimen."""
    plane = grow_tooth(0.85, 0.12, 0.0, n_steps=700)
    twist = grow_tooth(0.85, 0.12, 0.0025, n_steps=700)
    incisor = grow_tooth(0.90, 0.0, 0.0, n_steps=700)
    full = state["full"] and state["screw"] is not None
    ncols = 3 if full else 2
    fig, axes = plt.subplots(1, ncols, figsize=(5.5 * ncols, 5.5))
    fig.patch.set_facecolor("#0e1117")
    extents = []
    for g in (plane, twist, incisor):
        px, py = _project(g["xyz"])
        extents.append((px.min(), px.max(), py.min(), py.max()))
    xmin = min(e[0] for e in extents) - 20
    xmax = max(e[1] for e in extents) + 20
    ymin = min(e[2] for e in extents) - 20
    ymax = max(e[3] for e in extents) + 20
    if full:
        best = state["screw"]["fits"][state["screw"]["best"]]
        aligned = best["aligned_xy"]

    def draw(frame: int) -> None:
        f = (frame + 1) / n_frames
        n = int(10 + (len(plane["xyz"]) - 10) * f)
        for ax in axes:
            ax.cla()
            ax.set_facecolor("#161b22")
            ax.set_xticks([])
            ax.set_yticks([])
        for ax in axes[:2]:
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
        if full:
            ax = axes[2]
            ax.imshow(state["mam"])
            m = int(2 + (len(aligned) - 2) * f)
            ax.plot(aligned[:m, 0], aligned[:m, 1], color="#06d6a0", lw=3)
            ax.set_xlim(0, state["mam"].shape[1])
            ax.set_ylim(state["mam"].shape[0], 0)
            ax.set_title(f"fitted screw model growing tip→root (RMSE {best['rmse_px']:.1f} px)", color="#fff", fontsize=10)

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
        {"name": "log_spiral_pearson_r", "value": h["log_spiral_pearson_r"], "unit": "1", "method": f"OLS ln r vs θ about searched pole (E1); block-bootstrap 95% CI {h['pearson_r_ci95']}"},
        {"name": "log_spiral_rmse_px", "value": h["log_spiral_rmse_px"], "unit": "px", "method": "radial RMSE of E1 fit vs circle null and Archimedean control", "note": f"circle null RMSE = {h['circle_rmse_px']} px; Archimedean control RMSE = {h['archimedean_rmse_px']} px"},
        {"name": "log_spiral_psi_deg", "value": h["log_spiral_psi_deg"], "unit": "deg", "method": f"atan(1/k); block-bootstrap 95% CI {h['psi_ci95_deg']} deg; Thompson: mammoth high angle, Ovis ammon ~75"},
        {"name": "circle_null_rmse_px", "value": h["circle_rmse_px"], "unit": "px", "method": "Kasa circle fit (constant-curvature control)"},
        {"name": "cv_kappa_w_accretion_law", "value": h["cv_kappa_w"], "unit": "1", "method": "CV of κ·w along centreline (E4) vs CV κ alone; shuffled-width permutation control", "note": f"CV kappa = {h['cv_kappa']}; shuffled-width median CV = {h['shuffled_width_cv_median']}, CV p = {h['shuffled_width_cv_p_value']}"},
        {"name": "spearman_abs_kappa_vs_inv_width", "value": h["spearman_abs_kappa_vs_inv_w"], "unit": "1", "method": f"Spearman ρ(|κ(s)|, 1/w(s)) along the tusk; E4 predicts > 0; block-shuffled-width permutation p = {h['shuffled_width_spearman_p_value']} (null control)"},
        {"name": "kappa_sign_changes", "value": h["kappa_sign_changes"], "unit": "count", "method": "sign changes of width-scaled smoothed curvature root→tip (plane spiral predicts 0)"},
        {"name": "screw_fit_rmse_measured_width_twisted_px", "value": h["screw_rmse_measured_width_twisted_px"], "unit": "px", "method": "3-D screw-accretion fit κ=Δ/w(s), τ free, Procrustes-aligned projection (E4-E6)", "note": f"plane τ=0 RMSE = {h['screw_rmse_measured_width_plane_px']} px; ΔAIC twist = {h['delta_aic_twist_measured_width']}"},
        {"name": "screw_fit_rmse_const_width_twisted_px", "value": h["screw_rmse_const_width_twisted_px"], "unit": "px", "method": "3-D helix fit κ=Δ/w̄, τ free (constant-width ablation of the accretion law)", "note": f"plane RMSE = {h['screw_rmse_const_width_plane_px']} px; width-law RMSE ratio = {h['width_law_rmse_ratio']}"},
        {"name": "twist_rmse_ratio_measured_width", "value": h["twist_rmse_ratio"]["measured_width"], "unit": "1", "method": "RMSE(τ=0)/RMSE(τ free) for the measured-width accretion model (>1 means twist improves the fit; τ=0 ablation)"},
        {"name": "narwhal_alpha_equiv_deg", "value": h["narwhal_alpha_equiv_deg"], "unit": "deg", "method": f"structure-tensor groove inclination → Kiladze chord angle π/(2 tan φ); bootstrap 95% CI of φ {h['narwhal_phi_ci95_deg']} deg",
         "reference_value": kil["alpha"]["mean"], "reference_source": f"Kiladze & Chernova (2018) Data in Brief 21:1200, Table 1 (n=30, SD {kil['alpha']['sd']:.2f})"},
        {"name": "narwhal_tan_alpha_equiv", "value": h["narwhal_tan_alpha_equiv"], "unit": "1", "method": "π/(2 tan φ)", "reference_value": kil["tan_alpha"]["mean"],
         "reference_source": "Kiladze & Chernova (2018) Table 1 (range 1.65-3.22)"},
        {"name": "narwhal_z_alpha", "value": h["narwhal_z_alpha"], "unit": "SD", "method": "(α_measured − α_ref)/SD_ref"},
        {"name": "narwhal_tan_alpha_vs_inv_d_slope", "value": h["narwhal_tan_alpha_vs_inv_d_slope"], "unit": "px", "method": f"OLS slope of tan α on 1/d along the shaft (E7 uniform torsion predicts > 0); bootstrap 95% CI {h['narwhal_tan_alpha_vs_inv_d_slope_ci95']}; confounded by camera obliquity"},
        {"name": "st_venant_sign_sectors", "value": h["st_venant_sign_sectors"], "unit": "count", "method": f"sign alternations of 3x²y−y³ on a circle about the centroid; circular-section null max|u_z| = {h['st_venant_circle_warp_max']}", "reference_value": 6,
         "reference_source": "St Venant (1856) / Thompson Fig. 321: six alternating sectors"},
        {"name": "sweep_psi_pred_vs_fit_pearson_r", "value": h["sweep_S4_psi_pred_vs_fit_r"], "unit": "1", "method": f"parameter sweep S4: ψ fitted on generated plane curves vs E5 prediction atan(Δ/c_s); RMSE {h['sweep_S4_psi_rmse_deg']} deg"},
        {"name": "ks_pvalue_spiral_residual_normality", "value": h["ks_pvalue_spiral_residual_normality"], "unit": "1", "method": "one-sample KS of standardised E1 residuals vs N(0,1)"},
    ]


def run_self_test() -> None:
    load_reference()
    # E1/E2 gnomonic self-similarity (mirrors Lean Law 1)
    W = whorl_expansion(75.0)
    assert abs(gnomon_radius(3.0, W, 2) * 3.0 - gnomon_radius(3.0, W, 1) * gnomon_radius(3.0, W, 1)) < 1e-9
    th = np.linspace(0, 2 * math.pi, 5)
    rr = log_spiral_r(1.0, 75.0, th)
    assert abs(rr[-1] / rr[0] - W) < 1e-9
    # E4: equal rates straight, swap mirrors (Lean Law 2); Δ ↔ λ inversion
    assert bend_moment(1.0, 1.0) == 0 and bend_moment(1.0, 0.8) == -bend_moment(0.8, 1.0)
    assert abs(lambda_from_contrast(growth_contrast(0.83)) - 0.83) < 1e-12
    g_circ = grow_tooth(0.9, 0.0, 0.0, n_steps=400)
    assert np.std(g_circ["kappa"]) < 1e-12, "constant width must give constant curvature"
    g_spir = grow_tooth(0.9, 0.1, 0.0, n_steps=400)
    assert np.all(np.diff(g_spir["kappa"]) < 0), "gnomonic widening must make curvature fall"
    assert np.std(g_spir["kappa"] * g_spir["w"]) < 1e-12, "κ·w = Δ must hold exactly"
    assert np.max(np.abs(g_spir["xyz"][:, 1])) < 1e-9
    g_tw = grow_tooth(0.9, 0.1, 0.005, n_steps=400)
    assert np.max(np.abs(g_tw["xyz"][:, 1])) > 1.0
    g_prof = grow_tooth(0.9, 0.0, 0.0, n_steps=50, w_profile=np.linspace(5, 10, 51))
    assert np.allclose(g_prof["kappa"] * g_prof["w"], g_prof["delta"]), "measured-width profile must obey κ·w = Δ"
    # E8/E9 St Venant (Lean Law 3)
    xs = np.array([0.3, -0.2, 0.5])
    ys = np.array([0.1, 0.4, -0.3])
    assert np.allclose(tri_warp(xs, -ys), -tri_warp(xs, ys)) and np.allclose(tri_warp(-xs, ys), tri_warp(xs, ys))
    assert np.allclose(tri_warp(xs, np.zeros(3)), 0) and np.allclose(ellipse_warp(1.0, 1.0, xs, ys), 0)
    sv = st_venant_triangle_field(n=121)
    assert sv["sign_sectors"] == 6 and sv["face_sign_changes"] == [1, 1, 1] and sv["circle_warp_max"] == 0.0 and sv["ellipse_1p3_warp_max"] > 0
    # E7 helix (Lean Law 4)
    assert helix_slope(1.0, 0.5) < helix_slope(2.0, 0.5) and helix_slope(1.0, -0.5) == -helix_slope(1.0, 0.5)
    assert abs(kiladze_tan_alpha_from_twist(math.pi / (2.39 * 1.0), 1.0) - 2.39) < 1e-9
    # synthetic log spiral recovered by the pole search; Archimedean control worse
    rng = np.random.default_rng(SEED)
    tt = np.linspace(0.3, 2.2 * math.pi, 300)
    k = 1 / math.tan(math.radians(78.0))
    xs_, ys_ = 100 + 20 * np.exp(k * tt) * np.cos(tt), 80 + 20 * np.exp(k * tt) * np.sin(tt)
    xs_ += rng.normal(0, 0.2, xs_.shape)
    ys_ += rng.normal(0, 0.2, ys_.shape)
    pole, f = search_log_spiral_pole(xs_, ys_, n_grid=15)
    assert abs(f["psi_deg"] - 78.0) < 1.5 and f["pearson_r"] > 0.99, f
    assert fit_archimedean(xs_, ys_, pole)["rmse_px"] > f["rmse_px"]
    # synthetic twisted tusk recovered by the screw fit (τ ≠ 0 must beat τ = 0)
    truth = grow_tooth(0.8, 0.0, 0.004, n_steps=150, ds=4.0, w_profile=np.full(151, 60.0))
    obs = project_view(truth["xyz"], 1.1, 0.7)
    target = {"obs": obs, "w_meas": np.full(151, 60.0), "w_const": np.full(151, 60.0), "n_steps": 150, "ds": 4.0}
    f_plane = fit_screw_model(target, target["w_const"], False, 4, np.random.default_rng(1))
    f_twist = fit_screw_model(target, target["w_const"], True, 6, np.random.default_rng(2))
    assert f_twist["rmse_px"] < f_plane["rmse_px"], (f_twist["rmse_px"], f_plane["rmse_px"])
    assert abs(abs(f_twist["tau"]) - 0.004) < 0.0015, f_twist["tau"]
    # Procrustes alignment is exact for a rigidly moved copy
    Q = np.column_stack([np.linspace(0, 10, 20), np.sin(np.linspace(0, 3, 20))])
    ang = 0.4
    P = Q @ np.array([[math.cos(ang), -math.sin(ang)], [math.sin(ang), math.cos(ang)]]) + np.array([3.0, -2.0])
    assert procrustes_align(P, Q)[0] < 1e-9
    # specimen pipeline + determinism (two independent full runs)
    m1 = measure_morphoeval()
    m2 = measure_morphoeval()
    assert m1 == m2, "measure_morphoeval must be deterministic"
    assert all(m["value"] is not None for m in m1)
    by_name = {m["name"]: m["value"] for m in m1}
    print(f"self-test OK ({len(m1)} measurements; log-spiral r={by_name['log_spiral_pearson_r']}, "
          f"screw fit RMSE={by_name['screw_fit_rmse_measured_width_twisted_px']} px, narwhal α≡{by_name['narwhal_alpha_equiv_deg']}°)")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--pass1", action="store_true", help="Pass 1 baseline: coarse pole grid, fixed σ, no screw fits / sweep / controls")
    ap.add_argument("--sigma-frac", type=float, default=DEFAULTS["sigma_frac"], help="centreline smoothing σ as a fraction of median width")
    ap.add_argument("--n-trials", type=int, default=DEFAULTS["n_trials"], help="Nelder-Mead restarts per screw-model fit")
    ap.add_argument("--n-boot", type=int, default=DEFAULTS["n_boot"], help="bootstrap replicates")
    ap.add_argument("--n-perm", type=int, default=DEFAULTS["n_perm"], help="shuffled-width permutations")
    args = ap.parse_args(argv)
    if args.self_test:
        run_self_test()
        return
    state = run_analysis(full=not args.pass1, sigma_frac=args.sigma_frac, n_trials=args.n_trials, n_boot=args.n_boot, n_perm=args.n_perm)
    render_comparison(state)
    render_growth(state)
    RESULT_JSON.write_text(json.dumps(state["result"], indent=2) + "\n")
    print(f"Wrote {OUT_PNG.relative_to(ROOT)}, {OUT_GIF.relative_to(ROOT)}, {RESULT_JSON.relative_to(ROOT)}")
    print(f"Verdict: {state['result']['verdict']}  components: {state['result']['verdict_components']}")
    for k, v in state["result"]["headline_numbers"].items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()

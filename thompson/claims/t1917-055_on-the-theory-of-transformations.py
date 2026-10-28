#!/usr/bin/env python3
"""
T1917-055 — Thompson (1917) Ch. XVII, pp. 760-764, Figs. 394-400:
rhinoceros and allied skulls (after Osborn) compared by deforming a common Cartesian
grid — Hyrachyus, Aceratherium, Titanotherium, tapir, horse, and rabbit panels.

Claim as registered (thompson/thompson_1917_claims.json, T1917-055)
  "The skulls of rhinoceroses and their allies (Hyrachyus, Aceratherium, Titanotherium,
  tapir, horse, rabbit comparisons after Osborn) differ by coordinate transformation of
  a common skull diagram."

What Thompson prints (pp. 760-764)
  Figs. 394-395 superpose the coordinate nets of Hyrachyus and Aceratherium; Figs. 396-400
  extend the same method to occipital views and to more distant skull shapes (titanothere,
  tapir, horse, rabbit).  The claim is methodological: a smooth grid map phi carrying one
  inscribed diagram onto another should carry the whole outline, not merely affine strain.

Equations (datasets/thompson_1917/t1917-055/t1917_055_on_the_theory_of_transformatio_reference.json)
  E1  phi : R^2 -> R^2, phi(grid_A) = grid_B, det(D phi) > 0           -> tps_fit / check_no_fold
  E2  phi_t(x) = (1 - t) x + t phi(x), t in [0, 1]                     -> interp_stage
  E3  Omega_t = phi_t(Omega_A)                                         -> predicted outlines
  E4  thin-plate spline realisation of phi (Bookstein 1989)            -> tps_fit / tps_apply
  E5  affine special case, M_t = (1 - t) I + t A                       -> affine_fit / affine_interp_matrix
  E6  phi_t(a_i) = (1 - t) a_i + t b_i for corresponding points        -> interp_stage on configurations
  E7  t*_k = <D_k - A, B - A> / |B - A|^2 ; residual |t*_k - k/4|      -> best_fit_t (secondary stages)
  Lean mirror: morphoeval/lean/loops/Thompson1917_T1917_055.lean.

Data (SHA-256 pinned in the reference JSON; checked in verify_inputs)
  Internet Archive item ongrowthform1917thom, pp. 760-764 (leaves n780-n784), public domain.
  Grid intersections and skull outlines are read from the printed figures (percentile threshold,
  morphological opening, row/column peak picking on the grid ink, largest skull component).
  Wikimedia Commons Ceratotherium simum (MNHN 1400) is vendored for a descriptive bone-vs-matrix
  segmentation only.  thompson/portfolio/results/leporidae_skulls.json is a modern rabbit-clade
  analogue (DATASET_MISMATCH vs Thompson's perissodactyl series) and is not used in the verdict.

What is computed
  1. Automatic grid landmarks and arc-length skull boundaries per Fig. 394-400 panel.
  2. Correspondence of grid nodes by normalized-bbox nearest-neighbour matching (max distance 0.12).
  3. Primary pair Hyrachyus (394) -> Aceratherium (395): TPS phi on matched grids, phi applied to
     the Hyrachyus boundary, compared to the drawn Aceratherium boundary (Procrustes, Pearson r,
     RMSE/centroid size, IoU, symmetric boundary RMSE) vs identity, affine-only, and shuffled-grid null.
  4. Secondary pairs: occipital bottom -> top-left / top-right (Fig. 396); Hyrachyus -> Titanotherium,
     tapir, horse, rabbit (Figs. 397-400).
  5. Parameter sweep over maxd, TPS lambda, boundary point count; E1 no-fold check on the primary TPS.
  6. Figures: docs/media/t1917_055_on_the_theory_of_transformatio_comparison.png and ..._growth.gif;
     verdict: datasets/thompson_1917/t1917-055/result.json.

Usage
  python3 thompson/claims/t1917-055_on-the-theory-of-transformations.py --self-test
  python3 thompson/claims/t1917-055_on-the-theory-of-transformations.py [--lam 0.0] [--maxd 0.12]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage as ndi
from scipy.signal import find_peaks
from scipy.spatial import cKDTree
from skimage import measure

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
DATA = ROOT / "datasets" / "thompson_1917" / "t1917-055"
MEDIA = ROOT / "docs" / "media"
SLUG = "t1917_055_on_the_theory_of_transformatio"
REF_JSON = DATA / f"{SLUG}_reference.json"
RESULT_JSON = DATA / "result.json"
COMPARISON_PNG = MEDIA / f"{SLUG}_comparison.png"
GROWTH_GIF = MEDIA / f"{SLUG}_growth.gif"
SCRIPT_REL = "thompson/claims/t1917-055_on-the-theory-of-transformations.py"

SEED = 1917
SPECIMEN_JPG = "ceratotherium_simum_skull_mnhn_1400.jpg"
VDIST = 14
HDIST = 12
DEFAULT_MAXD = 0.12
DEFAULT_LAM = 0.0
DEFAULT_N_BOUNDARY = 160

CROPS: dict[str, tuple[str, tuple[float, float, float, float]]] = {
    "hyrackyus_394": ("ongrowthform1917thom_p761_leaf0781_figs394-395.jpg", (0.08, 0.22, 0.48, 0.52)),
    "aceratherium_395": ("ongrowthform1917thom_p761_leaf0781_figs394-395.jpg", (0.52, 0.22, 0.92, 0.52)),
    "occip_bottom_396": ("ongrowthform1917thom_p762_leaf0782_fig396.jpg", (0.10, 0.55, 0.90, 0.92)),
    "occip_tl_396": ("ongrowthform1917thom_p762_leaf0782_fig396.jpg", (0.08, 0.08, 0.48, 0.48)),
    "occip_tr_396": ("ongrowthform1917thom_p762_leaf0782_fig396.jpg", (0.52, 0.08, 0.92, 0.48)),
    "titanotherium_397": ("ongrowthform1917thom_p763_leaf0783_figs397-398.jpg", (0.08, 0.10, 0.48, 0.55)),
    "tapir_398": ("ongrowthform1917thom_p763_leaf0783_figs397-398.jpg", (0.52, 0.10, 0.92, 0.55)),
    "horse_399": ("ongrowthform1917thom_p764_leaf0784_figs399-400.jpg", (0.10, 0.12, 0.90, 0.48)),
    "rabbit_400": ("ongrowthform1917thom_p764_leaf0784_figs399-400.jpg", (0.10, 0.52, 0.90, 0.88)),
}


# --------------------------------------------------------------------------- #
# provenance
# --------------------------------------------------------------------------- #
def load_reference() -> dict:
    return json.loads(REF_JSON.read_text(encoding="utf-8"))


def sha256_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def verify_inputs(ref: dict) -> dict[str, str]:
    """E0: every vendored input must hash to the pinned digest (fail closed)."""
    digests = {}
    for name, want in ref["inputs_sha256"].items():
        p = DATA / name
        if not p.exists():
            raise FileNotFoundError(f"pinned input missing: {p}")
        got = sha256_file(p)
        if got != want:
            raise ValueError(f"SHA-256 mismatch for {name}: {got} != {want}")
        digests[name] = got
    return digests


# --------------------------------------------------------------------------- #
# geometry helpers (shared with T1917-054)
# --------------------------------------------------------------------------- #
def crop_page(page: Path, frac: tuple[float, float, float, float]) -> np.ndarray:
    im = Image.open(page).convert("L")
    w, h = im.size
    box = (int(frac[0] * w), int(frac[1] * h), int(frac[2] * w), int(frac[3] * h))
    return np.asarray(im.crop(box)).astype(float)


def rc_to_xy(rc: np.ndarray) -> np.ndarray:
    """image (row, col) -> Cartesian (x right, y up)."""
    return np.c_[rc[:, 1], -rc[:, 0]]


def cr_to_xy(cr: np.ndarray) -> np.ndarray:
    """crop pixel (col, row) -> Cartesian."""
    return np.c_[cr[:, 0], -cr[:, 1]]


def centroid_size(X: np.ndarray) -> float:
    return float(np.sqrt(((X - X.mean(0)) ** 2).sum()))


def procrustes_align(X: np.ndarray, Y: np.ndarray) -> tuple[np.ndarray, float, dict]:
    Xc = X - X.mean(0)
    Yc = Y - Y.mean(0)
    zx = Xc[:, 0] + 1j * Xc[:, 1]
    zy = Yc[:, 0] + 1j * Yc[:, 1]
    a = np.vdot(zx, zy)
    beta = a / np.vdot(zx, zx)
    Xa = zx * beta
    sx, sy = np.sqrt(np.vdot(zx, zx).real), np.sqrt(np.vdot(zy, zy).real)
    rho = abs(a) / (sx * sy)
    d_full = float(np.sqrt(max(0.0, 1.0 - rho ** 2)))
    out = np.c_[Xa.real, Xa.imag] + Y.mean(0)
    return out, d_full, {"scale": float(abs(beta)), "rotation_deg": float(np.degrees(np.angle(beta)))}


def similarity_params(X: np.ndarray, Y: np.ndarray) -> tuple[complex, np.ndarray, np.ndarray]:
    Xc = X - X.mean(0)
    Yc = Y - Y.mean(0)
    zx = Xc[:, 0] + 1j * Xc[:, 1]
    zy = Yc[:, 0] + 1j * Yc[:, 1]
    beta = np.vdot(zx, zy) / np.vdot(zx, zx)
    return beta, X.mean(0), Y.mean(0)


def apply_similarity(P: np.ndarray, beta: complex, mx: np.ndarray, my: np.ndarray) -> np.ndarray:
    z = (P[:, 0] - mx[0]) + 1j * (P[:, 1] - mx[1])
    z = z * beta
    return np.c_[z.real + my[0], z.imag + my[1]]


def _U(r2: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(r2 > 0, r2 * np.log(r2), 0.0)


def tps_fit(P: np.ndarray, Q: np.ndarray, lam: float = 0.0) -> dict:
    n = len(P)
    r2 = ((P[:, None, :] - P[None, :, :]) ** 2).sum(-1)
    K = _U(r2) + lam * np.eye(n)
    Pm = np.c_[np.ones(n), P]
    L = np.zeros((n + 3, n + 3))
    L[:n, :n] = K
    L[:n, n:] = Pm
    L[n:, :n] = Pm.T
    Y = np.zeros((n + 3, 2))
    Y[:n] = Q
    sol = np.linalg.solve(L, Y)
    return {"P": P.copy(), "W": sol[:n], "a": sol[n:], "lam": lam}


def tps_apply(m: dict, X: np.ndarray) -> np.ndarray:
    r2 = ((X[:, None, :] - m["P"][None, :, :]) ** 2).sum(-1)
    return np.c_[np.ones(len(X)), X] @ m["a"] + _U(r2) @ m["W"]


def affine_fit(P: np.ndarray, Q: np.ndarray) -> np.ndarray:
    Pm = np.c_[np.ones(len(P)), P]
    M, *_ = np.linalg.lstsq(Pm, Q, rcond=None)
    return M


def affine_apply(M: np.ndarray, X: np.ndarray) -> np.ndarray:
    return np.c_[np.ones(len(X)), X] @ M


def affine_interp_matrix(M: np.ndarray, t: float) -> np.ndarray:
    Mt = M.copy()
    Mt[1:, :] = (1 - t) * np.eye(2) + t * M[1:, :]
    Mt[0, :] = t * M[0, :]
    return Mt


def interp_stage(X: np.ndarray, phiX: np.ndarray, t: float) -> np.ndarray:
    return (1.0 - t) * X + t * phiX


def best_fit_t(D: np.ndarray, A: np.ndarray, B: np.ndarray) -> float:
    v = (B - A).ravel()
    return float(np.dot((D - A).ravel(), v) / np.dot(v, v))


def make_grid(X: np.ndarray, nv: int = 9, nh: int = 8, per_line: int = 50, pad: float = 0.08) -> list[np.ndarray]:
    lo, hi = X.min(0), X.max(0)
    span = hi - lo
    lo, hi = lo - pad * span, hi + pad * span
    xs, ys = np.linspace(lo[0], hi[0], nv), np.linspace(lo[1], hi[1], nh)
    lines = []
    for x in xs:
        lines.append(np.c_[np.full(per_line, x), np.linspace(lo[1], hi[1], per_line)])
    for y in ys:
        lines.append(np.c_[np.linspace(lo[0], hi[0], per_line), np.full(per_line, y)])
    return lines


def rasterize(poly: np.ndarray, lo: np.ndarray, scale: float, size: int) -> np.ndarray:
    pts = [(float((x - lo[0]) * scale), float((y - lo[1]) * scale)) for x, y in poly]
    im = Image.new("1", (size, size), 0)
    ImageDraw.Draw(im).polygon(pts, fill=1)
    return np.asarray(im, dtype=bool)


def outline_iou(P: np.ndarray, Q: np.ndarray, size: int = 400) -> float:
    lo = np.minimum(P.min(0), Q.min(0))
    hi = np.maximum(P.max(0), Q.max(0))
    scale = (size - 1) / max(hi - lo)
    mp, mq = rasterize(P, lo, scale, size), rasterize(Q, lo, scale, size)
    return float((mp & mq).sum() / max((mp | mq).sum(), 1))


def chamfer_rmse(P: np.ndarray, Q: np.ndarray) -> float:
    d1, _ = cKDTree(Q).query(P)
    d2, _ = cKDTree(P).query(Q)
    return float(np.sqrt(0.5 * (np.mean(d1 ** 2) + np.mean(d2 ** 2))))


def compare_outlines(pred_cfg: np.ndarray, pred_dense: np.ndarray, drawn_cfg: np.ndarray, drawn_dense: np.ndarray) -> dict:
    aligned, d_full, _ = procrustes_align(pred_cfg, drawn_cfg)
    beta, mx, my = similarity_params(pred_cfg, drawn_cfg)
    pred_dense_al = apply_similarity(pred_dense, beta, mx, my)
    cs = centroid_size(drawn_cfg)
    resid = aligned - drawn_cfg
    r = float(np.corrcoef(aligned.ravel(), drawn_cfg.ravel())[0, 1])
    return {
        "procrustes": d_full,
        "pearson_r": r,
        "rmse_cs": float(np.sqrt((resid ** 2).sum(1).mean()) / cs),
        "iou": outline_iou(pred_dense_al, drawn_dense),
        "chamfer_rmse_cs": chamfer_rmse(pred_dense_al, drawn_dense) / cs,
    }


def jacobian_min_det(tps: dict, X: np.ndarray, t: float, h: float = 1e-4) -> float:
    def phi_t(P):
        return interp_stage(P, tps_apply(tps, P), t)

    ex, ey = np.array([h, 0.0]), np.array([0.0, h])
    dx = (phi_t(X + ex) - phi_t(X - ex)) / (2 * h)
    dy = (phi_t(X + ey) - phi_t(X - ey)) / (2 * h)
    det = dx[:, 0] * dy[:, 1] - dx[:, 1] * dy[:, 0]
    return float(det.min())


def check_no_fold(tps: dict, A: np.ndarray, dense_A: np.ndarray | None = None, n_t: int = 21) -> dict:
    from matplotlib.path import Path as MplPath

    pts = np.vstack(make_grid(A, per_line=25))
    ts = np.linspace(0.0, 1.0, n_t)
    mins = [jacobian_min_det(tps, pts, t) for t in ts]
    out = {"t": ts.tolist(), "min_det_grid": mins, "min_over_stages_grid": float(min(mins)), "no_fold_grid": bool(min(mins) > 0)}
    if dense_A is not None:
        inside = pts[MplPath(dense_A).contains_points(pts)]
        mins_in = [jacobian_min_det(tps, inside, t) for t in ts] if len(inside) else [float("nan")]
        out.update({"n_inside": int(len(inside)), "min_det_inside": mins_in,
                    "min_over_stages_inside": float(min(mins_in)), "no_fold_inside": bool(min(mins_in) > 0)})
    return out


# --------------------------------------------------------------------------- #
# digitisation: grid + skull boundary
# --------------------------------------------------------------------------- #
def _largest_skull_label(mask: np.ndarray, max_frac: float = 0.55, min_px: int = 8000) -> tuple[np.ndarray, int]:
    lab, n = ndi.label(mask)
    if n == 0:
        raise RuntimeError("empty mask after threshold")
    cand: list[tuple[int, float]] = []
    for i in range(1, n + 1):
        area = float(ndi.sum(mask, lab, i))
        if area >= min_px and area <= max_frac * mask.size:
            cand.append((i, area))
    if not cand:
        li = int(np.argmax(ndi.sum(mask, lab, range(1, n + 1)))) + 1
    else:
        cand.sort(key=lambda x: -x[1])
        li = cand[0][0]
    return lab, li


def grid_landmarks(crop: np.ndarray) -> np.ndarray:
    """Grid-node candidates as (col, row) in crop pixels.

    Percentile threshold + opening; row/column sum peaks (vdist~14, hdist~12); keep
    intersections inside the largest filled skull component (with a small fallback search
    when a panel's ink connects to the page border).
    """
    settings = ((28, 0.12), (28, 0.10), (25, 0.08), (32, 0.10), (35, 0.08))
    best = np.zeros((0, 2))
    for pct, prom in settings:
        thr = np.percentile(crop, pct)
        mask = ndi.binary_opening(crop < thr, iterations=1)
        lab, li = _largest_skull_label(mask)
        skull = ndi.binary_fill_holes(lab == li)
        row, col = mask.sum(1), mask.sum(0)
        rp, _ = find_peaks(row, distance=VDIST, prominence=max(float(row.max()) * prom, 1.0))
        cp, _ = find_peaks(col, distance=HDIST, prominence=max(float(col.max()) * prom, 1.0))
        pts = [[float(c), float(r)] for r in rp for c in cp if skull[max(0, r - 3):r + 4, max(0, c - 3):c + 4].any()]
        arr = np.asarray(pts, dtype=float)
        if len(arr) > len(best):
            best = arr
        if len(arr) >= 12:
            break
    return best


def norm_bbox(P: np.ndarray) -> np.ndarray:
    lo, hi = P.min(0), P.max(0)
    return (P - lo) / np.maximum(hi - lo, 1e-9)


def match_grids(A: np.ndarray, B: np.ndarray, maxd: float = DEFAULT_MAXD) -> tuple[np.ndarray, np.ndarray, list[tuple[int, int]]]:
    """Greedy one-to-one matching on normalized bbox coordinates; distance <= maxd."""
    if len(A) == 0 or len(B) == 0:
        return np.zeros((0, 2)), np.zeros((0, 2)), []
    An, Bn = norm_bbox(A), norm_bbox(B)
    d, j = cKDTree(Bn).query(An)
    cand = [(i, int(ji), float(di)) for i, (di, ji) in enumerate(zip(d, j)) if di <= maxd]
    cand.sort(key=lambda x: x[2])
    used_b: set[int] = set()
    pairs: list[tuple[int, int]] = []
    for i, jb, _ in cand:
        if jb not in used_b:
            used_b.add(jb)
            pairs.append((i, jb))
    PA = np.array([A[i] for i, _ in pairs], dtype=float)
    PB = np.array([B[j] for _, j in pairs], dtype=float)
    return PA, PB, pairs


def skull_boundary(skull: np.ndarray, n_boundary: int = DEFAULT_N_BOUNDARY) -> np.ndarray:
    smooth = ndi.binary_opening(ndi.binary_closing(skull, iterations=2), iterations=2)
    contours = measure.find_contours(smooth.astype(float), 0.5)
    if not contours:
        raise RuntimeError("no skull contour found")
    c = max(contours, key=len)
    dl = np.hypot(*np.diff(c, axis=0).T)
    s = np.r_[0.0, np.cumsum(dl)]
    ss = np.linspace(0.0, s[-1], n_boundary + 1)[:-1]
    rc = np.c_[np.interp(ss, s, c[:, 0]), np.interp(ss, s, c[:, 1])]
    return rc_to_xy(rc)


def load_panel(key: str, n_boundary: int = DEFAULT_N_BOUNDARY) -> dict:
    page_name, frac = CROPS[key]
    crop = crop_page(DATA / page_name, frac)
    grid_cr = grid_landmarks(crop)
    thr = np.percentile(crop, 28)
    mask = ndi.binary_opening(crop < thr, iterations=1)
    lab, li = _largest_skull_label(mask)
    skull = ndi.binary_fill_holes(lab == li)
    boundary_xy = skull_boundary(skull, n_boundary=n_boundary)
    return {
        "key": key,
        "page": page_name,
        "frac": frac,
        "crop": crop,
        "grid_cr": grid_cr,
        "grid_xy": cr_to_xy(grid_cr) if len(grid_cr) else np.zeros((0, 2)),
        "skull": skull,
        "boundary_xy": boundary_xy,
        "n_grid": int(len(grid_cr)),
    }


# --------------------------------------------------------------------------- #
# pair analysis
# --------------------------------------------------------------------------- #
def analyze_pair(
    source_key: str,
    target_key: str,
    panels: dict[str, dict],
    maxd: float = DEFAULT_MAXD,
    lam: float = DEFAULT_LAM,
    n_boundary: int = DEFAULT_N_BOUNDARY,
    rng: np.random.Generator | None = None,
) -> dict:
    src, tgt = panels[source_key], panels[target_key]
    if n_boundary != len(src["boundary_xy"]):
        src = {**src, "boundary_xy": skull_boundary(src["skull"], n_boundary)}
    if n_boundary != len(tgt["boundary_xy"]):
        tgt = {**tgt, "boundary_xy": skull_boundary(tgt["skull"], n_boundary)}

    PA, PB, pairs = match_grids(src["grid_cr"], tgt["grid_cr"], maxd=maxd)
    n_match = len(pairs)
    out: dict = {
        "source": source_key,
        "target": target_key,
        "n_grid_source": src["n_grid"],
        "n_grid_target": tgt["n_grid"],
        "n_matched": n_match,
        "maxd": maxd,
        "lam": lam,
        "n_boundary": n_boundary,
    }
    if n_match < 4:
        out["error"] = "too few matched grid landmarks"
        return out

    P, Q = cr_to_xy(PA), cr_to_xy(PB)
    b_src, b_tgt = src["boundary_xy"], tgt["boundary_xy"]

    tps = tps_fit(P, Q, lam=lam)
    pred_b = tps_apply(tps, b_src)
    fit_tps = compare_outlines(pred_b, pred_b, b_tgt, b_tgt)

    fit_identity = compare_outlines(b_src, b_src, b_tgt, b_tgt)

    aff = affine_fit(P, Q)
    pred_aff = affine_apply(aff, b_src)
    fit_aff = compare_outlines(pred_aff, pred_aff, b_tgt, b_tgt)

    rng = rng or np.random.default_rng(SEED)
    Q_shuf = Q[rng.permutation(len(Q))]
    tps_null = tps_fit(P, Q_shuf, lam=lam)
    pred_null = tps_apply(tps_null, b_src)
    fit_null = compare_outlines(pred_null, pred_null, b_tgt, b_tgt)

    out.update({
        "tps": fit_tps,
        "identity": fit_identity,
        "affine": fit_aff,
        "shuffled_grid_null": fit_null,
        "tps_beats_identity_procrustes": bool(fit_tps["procrustes"] < fit_identity["procrustes"]),
        "tps_beats_identity_iou": bool(fit_tps["iou"] > fit_identity["iou"]),
        "tps_beats_affine_procrustes": bool(fit_tps["procrustes"] < fit_aff["procrustes"]),
        "tps_beats_affine_iou": bool(fit_tps["iou"] > fit_aff["iou"]),
        "tps_beats_shuffled_procrustes": bool(fit_tps["procrustes"] < fit_null["procrustes"]),
        "tps_model": tps,
        "grid_source_xy": P,
        "grid_target_xy": Q,
        "pred_boundary_xy": pred_b,
    })
    return out


def run_analysis(
    maxd: float = DEFAULT_MAXD,
    lam: float = DEFAULT_LAM,
    n_boundary: int = DEFAULT_N_BOUNDARY,
    verbose: bool = True,
) -> dict:
    t0 = time.time()
    ref = load_reference()
    digests = verify_inputs(ref)

    panels = {k: load_panel(k, n_boundary=n_boundary) for k in CROPS}
    primary = analyze_pair("hyrackyus_394", "aceratherium_395", panels, maxd=maxd, lam=lam, n_boundary=n_boundary)
    secondary_specs = [
        ("occip_bottom_396", "occip_tl_396"),
        ("occip_bottom_396", "occip_tr_396"),
        ("hyrackyus_394", "titanotherium_397"),
        ("hyrackyus_394", "tapir_398"),
        ("hyrackyus_394", "horse_399"),
        ("hyrackyus_394", "rabbit_400"),
    ]
    secondary = [analyze_pair(a, b, panels, maxd=maxd, lam=lam, n_boundary=n_boundary) for a, b in secondary_specs]

    no_fold = None
    if primary.get("tps_model") is not None:
        no_fold = check_no_fold(primary["tps_model"], primary["grid_source_xy"], primary["pred_boundary_xy"])

    hl = _headline(primary, secondary, no_fold)
    res = {
        "maxd": maxd,
        "lam": lam,
        "n_boundary": n_boundary,
        "seed": SEED,
        "digests": digests,
        "panels": panels,
        "primary": primary,
        "secondary": secondary,
        "headline": hl,
        "no_fold_primary": no_fold,
        "elapsed_s": time.time() - t0,
    }
    if verbose:
        print(f"[t1917-055] primary matched={primary.get('n_matched')} "
              f"Procrustes TPS={primary.get('tps', {}).get('procrustes', float('nan')):.4f} "
              f"identity={primary.get('identity', {}).get('procrustes', float('nan')):.4f} "
              f"IoU={primary.get('tps', {}).get('iou', float('nan')):.3f}")
    return res


def _headline(primary: dict, secondary: list[dict], no_fold: dict | None) -> dict:
    pt = primary.get("tps", {})
    return {
        "primary_n_matched": primary.get("n_matched", 0),
        "primary_procrustes_tps": pt.get("procrustes"),
        "primary_procrustes_identity": primary.get("identity", {}).get("procrustes"),
        "primary_procrustes_affine": primary.get("affine", {}).get("procrustes"),
        "primary_iou_tps": pt.get("iou"),
        "primary_iou_affine": primary.get("affine", {}).get("iou"),
        "primary_pearson_r": pt.get("pearson_r"),
        "primary_rmse_cs": pt.get("rmse_cs"),
        "primary_chamfer_cs": pt.get("chamfer_rmse_cs"),
        "primary_tps_beats_identity_procrustes": primary.get("tps_beats_identity_procrustes"),
        "primary_tps_beats_identity_iou": primary.get("tps_beats_identity_iou"),
        "primary_tps_beats_affine_procrustes": primary.get("tps_beats_affine_procrustes"),
        "primary_tps_beats_affine_iou": primary.get("tps_beats_affine_iou"),
        "secondary_n_pairs": len(secondary),
        "secondary_tps_beats_identity_count": sum(1 for s in secondary if s.get("tps_beats_identity_procrustes")),
        "min_jacobian_det_primary_inside": None if no_fold is None else no_fold.get("min_over_stages_inside"),
    }


SWEEP_MAXD = (0.08, 0.10, 0.12, 0.15)
SWEEP_LAM = (0.0, 0.001, 0.01)
SWEEP_N_BOUNDARY = (80, 120, 160)


def parameter_sweep(verbose: bool = True) -> dict:
    rows = []
    for maxd in SWEEP_MAXD:
        for lam in SWEEP_LAM:
            for nb in SWEEP_N_BOUNDARY:
                r = run_analysis(maxd=maxd, lam=lam, n_boundary=nb, verbose=False)
                hl = r["headline"]
                rows.append({
                    "maxd": maxd,
                    "lam": lam,
                    "n_boundary": nb,
                    "n_matched": hl["primary_n_matched"],
                    "procrustes_tps": hl["primary_procrustes_tps"],
                    "procrustes_identity": hl["primary_procrustes_identity"],
                    "iou_tps": hl["primary_iou_tps"],
                    "tps_beats_identity": hl["primary_tps_beats_identity_procrustes"],
                    "tps_beats_affine": hl["primary_tps_beats_affine_procrustes"],
                })
    beats_id = [row["tps_beats_identity"] for row in rows if row["n_matched"] >= 40]
    out = {"rows": rows, "n_settings": len(rows), "all_beat_identity_when_matched": bool(beats_id and all(beats_id))}
    if verbose:
        print(f"[t1917-055] sweep {len(rows)} settings; primary beats identity in {sum(beats_id)}/{len(beats_id)} "
              f"settings with >=40 matches")
    return out


def otsu_threshold(a: np.ndarray) -> int:
    hist, _ = np.histogram(a, bins=256, range=(0, 256))
    p = hist / max(hist.sum(), 1)
    w = np.cumsum(p)
    m = np.cumsum(p * np.arange(256))
    sb = (m[-1] * w - m) ** 2 / np.maximum(w * (1 - w), 1e-12)
    return int(np.argmax(sb))


def segment_specimen_bone(path: Path) -> dict:
    """Descriptive bone-vs-matrix on the Ceratotherium photograph (not in verdict)."""
    im = Image.open(path).convert("RGB")
    a = np.asarray(im).astype(float) / 255.0
    mx, mn = a.max(2), a.min(2)
    light = a.mean(2)
    sat = np.where(mx > 0, (mx - mn) / np.maximum(mx, 1e-9), 0.0)
    score = np.clip(light - sat, 0, 1)
    thr = otsu_threshold((score * 255).astype(np.uint8))
    bone = ndi.binary_opening(score * 255 > thr, iterations=2)
    lab, n = ndi.label(bone)
    sizes = np.sort(ndi.sum(bone, lab, range(1, n + 1)))[::-1] if n else np.array([0])
    return {"file": path.name, "size_px": list(im.size), "otsu": int(thr), "bone_fraction": float(bone.mean()),
            "n_components": int(n), "largest_component_fraction": float(sizes[0] / max(bone.sum(), 1)),
            "mask": bone}


# --------------------------------------------------------------------------- #
# figures
# --------------------------------------------------------------------------- #
def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _closed(X: np.ndarray) -> np.ndarray:
    return np.vstack([X, X[:1]])


def render_comparison(res: dict, specimen: dict | None = None, out: Path = COMPARISON_PNG) -> Path:
    plt = _plt()
    primary = res["primary"]
    panels = res["panels"]
    src, tgt = panels["hyrackyus_394"], panels["aceratherium_395"]
    fig = plt.figure(figsize=(18, 10), dpi=110)
    gs = fig.add_gridspec(2, 3, width_ratios=[1.0, 1.0, 1.1])

    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(src["crop"], cmap="gray", vmin=0, vmax=255)
    if len(src["grid_cr"]):
        ax.plot(src["grid_cr"][:, 0], src["grid_cr"][:, 1], "c.", ms=3)
    ax.set_title("Fig. 394 — Hyrachyus (grid landmarks)", fontsize=9)
    ax.axis("off")

    ax = fig.add_subplot(gs[0, 1])
    ax.imshow(tgt["crop"], cmap="gray", vmin=0, vmax=255)
    if len(tgt["grid_cr"]):
        ax.plot(tgt["grid_cr"][:, 0], tgt["grid_cr"][:, 1], "c.", ms=3)
    ax.set_title("Fig. 395 — Aceratherium", fontsize=9)
    ax.axis("off")

    ax = fig.add_subplot(gs[0, 2])
    if primary.get("pred_boundary_xy") is not None:
        pred = primary["pred_boundary_xy"]
        drawn = tgt["boundary_xy"]
        beta, mx, my = similarity_params(pred, drawn)
        drawn_al = apply_similarity(drawn, beta, mx, my)
        ax.fill(*_closed(drawn_al).T, color="0.55", alpha=0.35, label="drawn Aceratherium")
        ax.plot(*_closed(pred).T, color="#d62728", lw=1.5, label="TPS(Hyrachyus outline)")
        ax.set_aspect("equal")
        pt = primary["tps"]
        ax.set_title(f"primary pair: Procrustes {pt['procrustes']:.3f} · IoU {pt['iou']:.2f} · r {pt['pearson_r']:.3f}", fontsize=9)
        ax.legend(fontsize=7, loc="lower left")
    ax.axis("off")

    ax = fig.add_subplot(gs[1, 0])
    keys = [f"{s['source']}→{s['target']}" for s in res["secondary"]]
    vals = [s.get("tps", {}).get("procrustes", float("nan")) for s in res["secondary"]]
    ax.barh(range(len(keys)), vals, color="#1f77b4")
    ax.set_yticks(range(len(keys)), keys, fontsize=7)
    ax.set_xlabel("Procrustes distance (TPS prediction vs target boundary)")
    ax.set_title("secondary pairs", fontsize=9)

    ax = fig.add_subplot(gs[1, 1])
    labels = ["TPS", "identity", "affine", "shuffled grid"]
    vals = [
        primary.get("tps", {}).get("procrustes", float("nan")),
        primary.get("identity", {}).get("procrustes", float("nan")),
        primary.get("affine", {}).get("procrustes", float("nan")),
        primary.get("shuffled_grid_null", {}).get("procrustes", float("nan")),
    ]
    ax.bar(labels, vals, color=["#d62728", "0.5", "#aec7e8", "0.7"])
    ax.set_ylabel("Procrustes distance (lower better)")
    ax.set_title("primary controls", fontsize=9)

    ax = fig.add_subplot(gs[1, 2])
    if specimen is not None:
        im = Image.open(DATA / specimen["file"]).convert("RGB")
        ax.imshow(im)
        ax.contour(specimen["mask"].astype(float), levels=[0.5], colors=["#00e5ff"], linewidths=0.4)
        ax.set_title(f"Ceratotherium simum (MNHN) — bone fraction {100 * specimen['bone_fraction']:.1f}%\n(descriptive; not in verdict)", fontsize=8)
    ax.axis("off")

    fig.suptitle("T1917-055 · Thompson (1917) Figs. 394-400: perissodactyl skull grid transformations (Osborn)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    return out


def render_growth_gif(res: dict, out: Path = GROWTH_GIF, n_frames: int = 33) -> Path:
    plt = _plt()
    primary = res["primary"]
    if primary.get("tps_model") is None:
        raise RuntimeError("primary pair has no TPS model")
    tps = primary["tps_model"]
    P = primary["grid_source_xy"]
    b_src = res["panels"]["hyrackyus_394"]["boundary_xy"]
    b_tgt = res["panels"]["aceratherium_395"]["boundary_xy"]
    phi_b = tps_apply(tps, b_src)
    lo = np.minimum(b_src.min(0), b_tgt.min(0))
    hi = np.maximum(b_src.max(0), b_tgt.max(0))
    pad = 0.1 * (hi - lo).max()
    lo, hi = lo - pad, hi + pad
    frames = []
    for t in np.linspace(0, 1, n_frames):
        fig, ax = plt.subplots(figsize=(6.4, 4.8), dpi=90)
        for line in make_grid(P, per_line=40, pad=0.06):
            gl = interp_stage(line, tps_apply(tps, line), t)
            ax.plot(gl[:, 0], gl[:, 1], color="0.78", lw=0.6)
        pred = interp_stage(b_src, phi_b, t)
        ax.fill(*_closed(pred).T, color="#555555", alpha=0.85)
        ax.set_xlim(lo[0], hi[0])
        ax.set_ylim(lo[1], hi[1])
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(f"T1917-055 · Hyrachyus → Aceratherium  φ_t = (1−t)·id + t·φ   t = {t:.2f}", fontsize=8.5)
        fig.canvas.draw()
        frames.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()))
        plt.close(fig)
    out.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(out, save_all=True, append_images=frames[1:], duration=120, loop=0)
    return out


# --------------------------------------------------------------------------- #
# verdict + MorphoEval
# --------------------------------------------------------------------------- #
def verdict_label(hl: dict) -> str:
    beats_id = bool(hl.get("primary_tps_beats_identity_procrustes") and hl.get("primary_tps_beats_identity_iou"))
    beats_aff = bool(hl.get("primary_tps_beats_affine_procrustes") and hl.get("primary_tps_beats_affine_iou"))
    if beats_id and beats_aff:
        return "supported"
    if beats_id or beats_aff or hl.get("primary_tps_beats_identity_procrustes") or hl.get("primary_tps_beats_affine_procrustes"):
        return "mixed"
    return "refuted"


def write_result(res: dict, ref: dict, sweep: dict | None = None, specimen: dict | None = None, out: Path = RESULT_JSON) -> Path:
    hl = res["headline"]
    v = verdict_label(hl)
    doc = {
        "claim_id": "T1917-055",
        "date_made": "1917",
        "claim": ref.get("claim_text", ref.get("claim", "")),
        "dataset": {
            "name": "Thompson (1917) On Growth and Form, Figs. 394-400 (Osborn), Internet Archive ongrowthform1917thom pp. 760-764",
            "DOI": None,
            "url": "https://archive.org/details/ongrowthform1917thom",
            "sha256": res["digests"],
            "pre_scouted_repo_cache": {
                "path": "thompson/portfolio/results/leporidae_skulls.json",
                "status": "DATASET_MISMATCH",
                "reason": "Modern Leporidae GM analogue; Thompson's series is perissodactyl/rhinoceros — not used in verdict.",
            },
        },
        "verdict": v,
        "headline_numbers": {k: (round(v, 4) if isinstance(v, float) else v) for k, v in hl.items()},
        "primary_pair": "hyrackyus_394->aceratherium_395",
        "script": SCRIPT_REL,
        "figure": str(COMPARISON_PNG.relative_to(ROOT)),
        "animation": str(GROWTH_GIF.relative_to(ROOT)),
        "lean": "morphoeval/lean/loops/Thompson1917_T1917_055.lean",
        "notes": [
            "Grid correspondences are automatic (normalized-bbox nearest neighbour), not Thompson's hand-labelled nodes.",
            "Secondary pairs to tapir/horse/rabbit test cross-family analogues; rabbit panel is especially coarse.",
            "Leporidae portfolio cache is an analogue mismatch and is excluded from scoring.",
        ],
    }
    if sweep is not None:
        doc["parameter_sweep"] = {k: v for k, v in sweep.items() if k != "rows"}
        doc["parameter_sweep"]["rows"] = sweep["rows"]
    if specimen is not None:
        doc["specimen_cv_descriptive_only"] = {k: v for k, v in specimen.items() if k != "mask"}
    out.write_text(json.dumps(doc, indent=2), encoding="utf-8")
    return out


def measure_morphoeval() -> list[dict]:
    res = run_analysis(verbose=False)
    sweep = parameter_sweep(verbose=False)
    hl = res["headline"]
    primary = res["primary"]

    def M(name, value, unit, method, note=None):
        return {"name": name, "value": value, "unit": unit, "method": method, "derived_from": "script", **({"note": note} if note else {})}

    return [
        M("matched grid landmarks (Hyrachyus -> Aceratherium)", hl["primary_n_matched"], "count",
          f"normalized-bbox greedy NN, maxd={DEFAULT_MAXD}"),
        M("Procrustes distance, TPS outline vs Aceratherium", round(hl["primary_procrustes_tps"] or 0, 4), "Procrustes distance",
          "TPS on matched grids applied to Hyrachyus boundary", note=f"identity control {hl['primary_procrustes_identity']:.4f}"),
        M("IoU filled outlines, TPS vs Aceratherium", round(hl["primary_iou_tps"] or 0, 4), "IoU",
          "similarity-aligned dense boundaries, raster 400²"),
        M("parameter sweep settings", sweep["n_settings"], "count", "maxd x lam x n_boundary grid"),
    ]


# --------------------------------------------------------------------------- #
# self-test
# --------------------------------------------------------------------------- #
def self_test() -> int:
    t0 = time.time()
    ref = load_reference()
    verify_inputs(ref)
    print("ok  SHA-256 of all pinned inputs")

    rng = np.random.default_rng(SEED)
    X, Y = rng.normal(size=(7, 2)), rng.normal(size=(7, 2))
    assert np.allclose(interp_stage(X, Y, 0.0), X) and np.allclose(interp_stage(X, Y, 1.0), Y)
    steps = [interp_stage(X, Y, (k + 1) / 4) - interp_stage(X, Y, k / 4) for k in range(4)]
    assert all(np.allclose(s, (Y - X) / 4) for s in steps)
    print("ok  E2/E6 endpoints and equidistant steps")

    m = tps_fit(X, Y)
    assert np.allclose(tps_apply(m, X), Y, atol=1e-8)
    Maff = affine_fit(X, Y)
    assert np.allclose(affine_interp_matrix(Maff, 0.0)[1:], np.eye(2)) and np.allclose(affine_interp_matrix(Maff, 1.0), Maff)
    print("ok  E4 TPS interpolates; E5 affine endpoints")

    for k in range(5):
        assert abs(best_fit_t(interp_stage(X, Y, k / 4), X, Y) - k / 4) < 1e-12
    print("ok  E7 best-fit stage on exact interpolant")

    sq = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], float)
    assert abs(outline_iou(sq, sq) - 1.0) < 0.02
    assert chamfer_rmse(sq, sq) == 0.0
    print("ok  IoU and chamfer sanity")

    res = run_analysis(verbose=False)
    primary = res["primary"]
    assert primary.get("n_matched", 0) >= 40, primary.get("n_matched")
    print(f"ok  primary pair matched grid landmarks = {primary['n_matched']}")
    assert primary.get("tps_beats_identity_procrustes"), primary.get("tps", {}).get("procrustes")
    print("ok  primary TPS beats identity on Procrustes distance")

    sp = segment_specimen_bone(DATA / SPECIMEN_JPG)
    assert 0.02 < sp["bone_fraction"] < 0.95
    print(f"ok  Ceratotherium specimen segmentation (bone fraction {sp['bone_fraction']:.3f})")

    if "digitisation" in ref:
        want = ref["digitisation"].get("min_primary_matched", 40)
        assert primary["n_matched"] >= want

    print(f"self-test passed in {time.time() - t0:.1f} s")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--maxd", type=float, default=DEFAULT_MAXD)
    ap.add_argument("--lam", type=float, default=DEFAULT_LAM)
    ap.add_argument("--n-boundary", type=int, default=DEFAULT_N_BOUNDARY)
    ap.add_argument("--no-sweep", action="store_true")
    ap.add_argument("--no-specimen", action="store_true")
    ap.add_argument("--no-gif", action="store_true")
    args = ap.parse_args(argv)
    if args.self_test:
        return self_test()
    ref = load_reference()
    res = run_analysis(maxd=args.maxd, lam=args.lam, n_boundary=args.n_boundary)
    sweep = None if args.no_sweep else parameter_sweep()
    specimen = None if args.no_specimen else segment_specimen_bone(DATA / SPECIMEN_JPG)
    print("->", render_comparison(res, specimen=specimen).relative_to(ROOT))
    if not args.no_gif:
        print("->", render_growth_gif(res).relative_to(ROOT))
    print("->", write_result(res, ref, sweep=sweep, specimen=specimen).relative_to(ROOT))
    return 0


if __name__ == "__main__":
    sys.exit(main())

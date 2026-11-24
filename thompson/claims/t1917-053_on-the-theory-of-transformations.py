#!/usr/bin/env python3
"""
T1917-053 — Thompson (1917) Ch. XVII, p. 756, Fig. 387:
Dimorphodon and Pteranodon skulls related by coordinate transformation.

Pass 1 (--pass1): baseline TPS warp from segmented Fig. 387 outlines, similarity/affine controls.
Pass 2 (default): parameter sweeps, shuffled-landmark null, KS / r / RMSE / IoU, Jacobian no-fold,
specimen CV on vendored Pteranodon photograph.

Dryad doi:10.5061/dryad.4nr4cf37 (Foth et al. 2012) — correct clade frame; file not OAuth-retrievable.

    python3 thompson/claims/t1917-053_on-the-theory-of-transformations.py --self-test
    python3 thompson/claims/t1917-053_on-the-theory-of-transformations.py [--pass1]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage as ndi
from scipy import stats
from scipy.spatial import cKDTree
from skimage import measure

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
DATA = ROOT / "datasets" / "thompson_1917" / "t1917-053"
MEDIA = ROOT / "docs" / "media"
SLUG = "t1917_053_on_the_theory_of_transformatio"
REF_JSON = DATA / f"{SLUG}_reference.json"
RESULT_JSON = DATA / "result.json"
FIG387 = DATA / "ongrowthform1917thom_p756_leaf0775_fig387.jpg"
SPECIMEN = DATA / "pteranodon_longiceps_ypm1177_vendored.jpg"
COMPARISON_PNG = MEDIA / f"{SLUG}_comparison.png"
GROWTH_GIF = MEDIA / f"{SLUG}_growth.gif"
SCRIPT_REL = "thompson/claims/t1917-053_on-the-theory-of-transformations.py"

CROP_DIM = (0.05, 0.12, 0.48, 0.55)
CROP_PTE = (0.52, 0.12, 0.95, 0.55)
GRID = (8, 4)
SEED = 1917
EPS = 1e-12


def sha256_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def load_reference() -> dict:
    return json.loads(REF_JSON.read_text(encoding="utf-8"))


def verify_inputs(ref: dict) -> None:
    for name, want in ref["inputs_sha256"].items():
        p = DATA / name
        if sha256_file(p) != want:
            raise ValueError(f"SHA-256 mismatch for {name}")


def otsu_threshold(a: np.ndarray) -> int:
    hist, _ = np.histogram(a, bins=256, range=(0, 256))
    p = hist / max(hist.sum(), 1)
    w = np.cumsum(p)
    m = np.cumsum(p * np.arange(256))
    sb = (m[-1] * w - m) ** 2 / np.maximum(w * (1 - w), 1e-12)
    return int(np.argmax(sb))


def segment_panel(page: np.ndarray, frac: tuple[float, float, float, float]) -> dict:
    H, W = page.shape
    x0, y0, x1, y1 = [int(frac[i] * (W if i % 2 == 0 else H)) for i in range(4)]
    crop = page[y0:y1, x0:x1]
    thr = otsu_threshold(crop)
    mask = ndi.binary_opening(crop < thr, iterations=1)
    lab, n = ndi.label(mask)
    sizes = ndi.sum(mask, lab, range(1, n + 1))
    li = int(np.argmax(sizes)) + 1
    m = lab == li
    filled = ndi.binary_fill_holes(m)
    holes = filled & ~m
    hl, hn = ndi.label(holes)
    hs = ndi.sum(holes, hl, range(1, hn + 1))
    orbit_rc = None
    if hn > 0:
        hi = int(np.argmax(hs)) + 1
        orbit_rc = np.array(ndi.center_of_mass(hl == hi))
    smooth = ndi.binary_closing(filled, iterations=2)
    contours = measure.find_contours(smooth.astype(float), 0.5)
    c = max(contours, key=len)
    return {"contour_rc": c, "orbit_rc": orbit_rc, "mask": smooth, "offset": (x0, y0), "otsu": thr}


def rc_to_xy(rc: np.ndarray) -> np.ndarray:
    return np.c_[rc[:, 1], -rc[:, 0]]


def classify_skull_tips(contour_rc: np.ndarray, orbit_rc: np.ndarray | None) -> dict[str, int]:
    """Snout = max column; occiput = min column on upper half; dorsal = min row; ventral = max row."""
    c = contour_rc
    sn = int(np.argmax(c[:, 1]))
    oc = int(np.argmin(c[:, 1]))
    dor = int(np.argmin(c[:, 0]))
    ven = int(np.argmax(c[:, 0]))
    if orbit_rc is not None:
        orb_i = int(np.argmin(np.hypot(c[:, 0] - orbit_rc[0], c[:, 1] - orbit_rc[1])))
    else:
        orb_i = int(np.argmin(np.hypot(c[:, 0] - c[dor, 0], c[:, 1] - c[sn, 1])))
    return {"snout": sn, "dorsal": dor, "occiput": oc, "ventral": ven, "orbit": orb_i}


def resample_segment(c: np.ndarray, i0: int, i1: int, n: int) -> np.ndarray:
    N = len(c)
    idx = np.arange(i0, i0 + ((i1 - i0) % N) + 1) % N
    seg = c[idx]
    dl = np.hypot(*np.diff(seg, axis=0).T)
    s = np.r_[0.0, np.cumsum(dl)]
    ss = np.linspace(0.0, s[-1], n + 1)[:-1]
    return np.c_[np.interp(ss, s, seg[:, 0]), np.interp(ss, s, seg[:, 1])]


def landmark_configuration(form: dict, n_semi: int) -> dict:
    c = form["contour_rc"]
    tips = classify_skull_tips(c, form.get("orbit_rc"))
    order = [tips["snout"], tips["dorsal"], tips["occiput"], tips["ventral"]]
    N = len(c)
    fwd = sum(((order[(j + 1) % 4] - order[j]) % N) for j in range(4))
    if fwd != N:
        c = c[::-1]
        order = [N - 1 - i for i in order]
        tips = {k: N - 1 - tips[k] for k in tips}
    pts = np.vstack([resample_segment(c, order[j], order[(j + 1) % 4], n_semi) for j in range(4)])
    orbit_xy = rc_to_xy(form["orbit_rc"][None])[0] if form.get("orbit_rc") is not None else rc_to_xy(c[tips["orbit"]][None])[0]
    X = np.vstack([orbit_xy[None], rc_to_xy(pts)])
    dense = rc_to_xy(c[:: max(1, len(c) // 400)])
    return {"X": X, "dense": dense, "tips": tips, "n_semi": n_semi}


def centroid_size(X: np.ndarray) -> float:
    return float(np.sqrt(((X - X.mean(0)) ** 2).sum()))


def procrustes_align(X: np.ndarray, Y: np.ndarray) -> tuple[np.ndarray, float]:
    Xc = X - X.mean(0)
    Yc = Y - Y.mean(0)
    zx = Xc[:, 0] + 1j * Xc[:, 1]
    zy = Yc[:, 0] + 1j * Yc[:, 1]
    beta = np.vdot(zx, zy) / np.vdot(zx, zx)
    Xa = zx * beta
    sx, sy = np.sqrt(np.vdot(zx, zx).real), np.sqrt(np.vdot(zy, zy).real)
    rho = abs(np.vdot(zx, zy)) / (sx * sy)
    d_full = float(np.sqrt(max(0.0, 1.0 - rho ** 2)))
    out = np.c_[Xa.real, Xa.imag] + Y.mean(0)
    return out, d_full


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


def interp_stage(X: np.ndarray, phiX: np.ndarray, t: float) -> np.ndarray:
    return (1.0 - t) * X + t * phiX


def make_grid(X: np.ndarray, nv: int = GRID[0], nh: int = GRID[1], per_line: int = 40, pad: float = 0.08) -> np.ndarray:
    lo, hi = X.min(0), X.max(0)
    span = hi - lo
    lo, hi = lo - pad * span, hi + pad * span
    xs, ys = np.linspace(lo[0], hi[0], nv), np.linspace(lo[1], hi[1], nh)
    lines = []
    for x in xs:
        lines.append(np.c_[np.full(per_line, x), np.linspace(lo[1], hi[1], per_line)])
    for y in ys:
        lines.append(np.c_[np.linspace(lo[0], hi[0], per_line), np.full(per_line, y)])
    return np.vstack(lines)


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


def resample_closed(poly: np.ndarray, n: int) -> np.ndarray:
    dl = np.hypot(*np.diff(np.vstack([poly, poly[:1]]), axis=0).T)
    s = np.r_[0.0, np.cumsum(dl)]
    ss = np.linspace(0.0, s[-1], n + 1)[:-1]
    return np.c_[np.interp(ss, s, np.r_[poly[:, 0], poly[0, 0]]), np.interp(ss, s, np.r_[poly[:, 1], poly[0, 1]])]


def compare_warp(pred_cfg: np.ndarray, pred_dense: np.ndarray, tgt_cfg: np.ndarray, tgt_dense: np.ndarray) -> dict:
    """Score the dense outline (not the fitting landmarks — TPS hits anchors exactly by construction)."""
    n_pts = 128
    Pp = resample_closed(pred_dense, n_pts)
    Tp = resample_closed(tgt_dense, n_pts)
    aligned, d_full = procrustes_align(Pp, Tp)
    cs = centroid_size(Tp)
    resid = aligned - Tp
    r = float(np.corrcoef(aligned.ravel(), Tp.ravel())[0, 1])
    beta = np.vdot((Pp[:, 0] - Pp.mean(0)[0]) + 1j * (Pp[:, 1] - Pp.mean(0)[1]),
                   (Tp[:, 0] - Tp.mean(0)[0]) + 1j * (Tp[:, 1] - Tp.mean(0)[1])) / (
        np.vdot((Pp[:, 0] - Pp.mean(0)[0]) + 1j * (Pp[:, 1] - Pp.mean(0)[1]),
                (Pp[:, 0] - Pp.mean(0)[0]) + 1j * (Pp[:, 1] - Pp.mean(0)[1])))
    z = (pred_dense[:, 0] - Pp.mean(0)[0]) + 1j * (pred_dense[:, 1] - Pp.mean(0)[1])
    z = z * beta
    pred_al = np.c_[z.real + Tp.mean(0)[0], z.imag + Tp.mean(0)[1]]
    cfg_aligned, _ = procrustes_align(pred_cfg, tgt_cfg)
    resid_cfg = cfg_aligned - tgt_cfg
    return {
        "procrustes": d_full,
        "pearson_r": r,
        "rmse_cs": float(np.sqrt((resid ** 2).sum(1).mean()) / cs),
        "iou": outline_iou(pred_al, tgt_dense),
        "chamfer_rmse_cs": chamfer_rmse(pred_al, tgt_dense) / cs,
        "per_landmark_residual": (np.sqrt((resid_cfg ** 2).sum(1)) / centroid_size(tgt_cfg)).tolist(),
    }


def jacobian_min_det(tps: dict, X: np.ndarray, t: float, h: float = 1e-4) -> float:
    def phi_t(P):
        return interp_stage(P, tps_apply(tps, P), t)

    ex, ey = np.array([h, 0.0]), np.array([0.0, h])
    dx = (phi_t(X + ex) - phi_t(X - ex)) / (2 * h)
    dy = (phi_t(X + ey) - phi_t(X - ey)) / (2 * h)
    det = dx[:, 0] * dy[:, 1] - dx[:, 1] * dy[:, 0]
    return float(det.min())


def shuffled_null(P: np.ndarray, Q: np.ndarray, dense: np.ndarray, tgt_dense: np.ndarray, n_semi: int, reps: int = 200) -> dict:
    rng = np.random.default_rng(SEED)
    obs_tps = tps_fit(P, Q)
    obs_d = compare_warp(tps_apply(obs_tps, P), tps_apply(obs_tps, dense), Q, tgt_dense)["procrustes"]
    null = []
    nb = 4 * n_semi
    for _ in range(reps):
        s = int(rng.integers(1, nb))
        Qs = np.vstack([Q[:1], np.roll(Q[1:], s, axis=0)])
        tps = tps_fit(P, Qs)
        d = compare_warp(tps_apply(tps, P), tps_apply(tps, dense), Q, tgt_dense)["procrustes"]
        null.append(d)
    null = np.array(null)
    return {"observed": obs_d, "null_mean": float(null.mean()), "p_value": float((np.sum(null <= obs_d) + 1) / (len(null) + 1))}


def specimen_cv(path: Path) -> dict:
    rgb = np.asarray(Image.open(path).convert("RGB"))
    gray = rgb.mean(-1)
    thr = otsu_threshold(gray)
    mask = gray < thr
    mask = ndi.binary_opening(mask, iterations=2)
    lab, n = ndi.label(mask)
    sizes = ndi.sum(mask, lab, range(1, n + 1))
    frac = float(mask.mean())
    top = float(max(sizes) / max(mask.sum(), 1)) if n else 0.0
    return {"file": path.name, "size_px": list(rgb.shape[:2][::-1]), "otsu": thr, "foreground_fraction": frac, "largest_component_fraction": top}


def run_analysis(n_semi: int = 20, lam: float = 0.0, pass1: bool = False) -> dict:
    ref = load_reference()
    verify_inputs(ref)
    page = np.asarray(Image.open(FIG387).convert("L")).astype(float)
    dim = segment_panel(page, CROP_DIM)
    pte = segment_panel(page, CROP_PTE)
    cfg_dim = landmark_configuration(dim, n_semi)
    cfg_pte = landmark_configuration(pte, n_semi)
    P, Q = cfg_dim["X"], cfg_pte["X"]
    # GPA frame
    z_dim, _ = procrustes_align(P, Q)
    z_pte = Q  # target frame

    tps = tps_fit(P, Q, lam=lam)
    pred_cfg = tps_apply(tps, P)
    pred_dense = tps_apply(tps, cfg_dim["dense"])

    sim_pred, _ = procrustes_align(P, Q)
    sim_dense = sim_pred  # similarity carries landmarks; extrapolate to dense via same map
    beta = np.vdot((P[:, 0] - P.mean(0)[0]) + 1j * (P[:, 1] - P.mean(0)[1]),
                   (Q[:, 0] - Q.mean(0)[0]) + 1j * (Q[:, 1] - Q.mean(0)[1])) / (
        np.vdot((P[:, 0] - P.mean(0)[0]) + 1j * (P[:, 1] - P.mean(0)[1]),
                (P[:, 0] - P.mean(0)[0]) + 1j * (P[:, 1] - P.mean(0)[1])))
    zd = (cfg_dim["dense"][:, 0] - P.mean(0)[0]) + 1j * (cfg_dim["dense"][:, 1] - P.mean(0)[1])
    zd = zd * beta
    sim_dense = np.c_[zd.real + Q.mean(0)[0], zd.imag + Q.mean(0)[1]]

    aff_M = affine_fit(P, Q)
    aff_dense = affine_apply(aff_M, cfg_dim["dense"])

    metrics = {
        "tps": compare_warp(pred_cfg, pred_dense, Q, cfg_pte["dense"]),
        "similarity": compare_warp(sim_pred, sim_dense, Q, cfg_pte["dense"]),
        "affine": compare_warp(affine_apply(aff_M, P), aff_dense, Q, cfg_pte["dense"]),
        "copy_dimorphodon": compare_warp(P, cfg_dim["dense"], Q, cfg_pte["dense"]),
    }

    grid_pts = make_grid(P)
    det_grid = jacobian_min_det(tps, grid_pts, 1.0)
    det_mid = jacobian_min_det(tps, grid_pts, 0.5)

    out = {
        "n_semi": n_semi,
        "lambda": lam,
        "procrustes_dim_pte": float(procrustes_align(P, Q)[1]),
        "warp": metrics,
        "jacobian_min_det": {"t1_grid": det_grid, "t0.5_grid": det_mid},
        "bending_energy": float(np.trace(tps["W"].T @ _U(((P[:, None, :] - P[None, :, :]) ** 2).sum(-1)) @ tps["W"])),
    }

    if not pass1:
        out["shuffled_null"] = shuffled_null(P, Q, cfg_dim["dense"], cfg_pte["dense"], n_semi)
        pr = metrics["tps"]["per_landmark_residual"]
        cr = metrics["copy_dimorphodon"]["per_landmark_residual"]
        out["ks_landmark_residual"] = float(stats.ks_2samp(pr, cr).pvalue)
        out["specimen_cv"] = specimen_cv(SPECIMEN)
    return out, cfg_dim, cfg_pte, tps, pred_dense, cfg_pte["dense"]


def parameter_sweep() -> dict:
    rows = []
    for n_semi in (8, 12, 16, 20, 24, 32):
        for lam in (0.0, 0.001, 0.01, 0.1):
            m, _, _, _, _, _ = run_analysis(n_semi=n_semi, lam=lam, pass1=True)
            tps_p = m["warp"]["tps"]["procrustes"]
            sim_p = m["warp"]["similarity"]["procrustes"]
            rows.append({"n_semi": n_semi, "lam": lam, "tps_procrustes": tps_p, "similarity_procrustes": sim_p,
                         "tps_beats_similarity": tps_p < sim_p - 1e-6})
    tps_vals = [r["tps_procrustes"] for r in rows]
    return {"n_settings": len(rows), "rows": rows, "tps_procrustes_cv": float(np.std(tps_vals) / max(np.mean(tps_vals), EPS))}


def measure_morphoeval(n_semi: int = 20, lam: float = 0.0) -> list[dict]:
    m, _, _, _, _, _ = run_analysis(n_semi=n_semi, lam=lam, pass1=False)
    return [
        {"name": "procrustes_baseline_dim_pte", "value": m["procrustes_dim_pte"], "unit": "1", "method": "GPA full Procrustes distance"},
        {"name": "tps_procrustes_dense", "value": m["warp"]["tps"]["procrustes"], "unit": "1", "method": "128-pt resampled dense outline"},
        {"name": "similarity_procrustes_dense", "value": m["warp"]["similarity"]["procrustes"], "unit": "1", "method": "similarity control"},
        {"name": "affine_procrustes_dense", "value": m["warp"]["affine"]["procrustes"], "unit": "1", "method": "affine control"},
        {"name": "pearson_r_tps", "value": m["warp"]["tps"]["pearson_r"], "unit": "1", "method": "aligned dense landmarks"},
        {"name": "iou_tps", "value": m["warp"]["tps"]["iou"], "unit": "1", "method": "filled outline IoU"},
        {"name": "shuffled_landmark_null_p", "value": m.get("shuffled_null", {}).get("p_value"), "unit": "1", "method": "200 cyclic shifts"},
        {"name": "jacobian_min_det_t1", "value": m["jacobian_min_det"]["t1_grid"], "unit": "1", "method": "min det D phi_t on 8x4 grid"},
    ]


def write_figures(m: dict, cfg_dim: dict, cfg_pte: dict, pred_dense: np.ndarray, tgt_dense: np.ndarray, tps: dict) -> None:
    MEDIA.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    for ax, (title, pts, c) in zip(
        axes,
        [
            ("Dimorphodon (Fig. 387a)", cfg_dim["dense"], "C0"),
            ("TPS warp of Dimorphodon", pred_dense, "C1"),
            ("Pteranodon drawn (Fig. 387b)", tgt_dense, "C2"),
        ],
    ):
        ax.plot(pts[:, 0], pts[:, 1], c=c, lw=1.2)
        ax.set_aspect("equal")
        ax.set_title(title)
        ax.axis("off")
    fig.suptitle(
        f"T1917-053 — TPS Procrustes={m['warp']['tps']['procrustes']:.3f} "
        f"(sim={m['warp']['similarity']['procrustes']:.3f}, r={m['warp']['tps']['pearson_r']:.3f})"
    )
    fig.tight_layout()
    fig.savefig(COMPARISON_PNG, dpi=150)
    plt.close(fig)

    # growth GIF along phi_t
    P = cfg_dim["X"]
    frames = []
    for t in np.linspace(0, 1, 24):
        warped = interp_stage(cfg_dim["dense"], tps_apply(tps, cfg_dim["dense"]), float(t))
        frames.append(warped)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.set_aspect("equal")
    ax.axis("off")
    (ln,) = ax.plot([], [], "C1", lw=1.5)
    (ref,) = ax.plot(tgt_dense[:, 0], tgt_dense[:, 1], "C2--", lw=0.8, alpha=0.6)

    allp = np.vstack(frames + [tgt_dense])
    pad = 0.05 * (allp.max(0) - allp.min(0))
    ax.set_xlim(allp[:, 0].min() - pad[0], allp[:, 0].max() + pad[0])
    ax.set_ylim(allp[:, 1].min() - pad[1], allp[:, 1].max() + pad[1])

    def init():
        ln.set_data([], [])
        return ln, ref

    def update(i):
        f = frames[i]
        ln.set_data(f[:, 0], f[:, 1])
        ax.set_title(f"phi_t, t={i/23:.2f}")
        return ln, ref

    anim = animation.FuncAnimation(fig, update, init_func=init, frames=len(frames), blit=True)
    anim.save(GROWTH_GIF, writer="pillow", fps=8)
    plt.close(fig)


def write_result(m: dict, pass1: bool) -> None:
    tps = m["warp"]["tps"]
    sim = m["warp"]["similarity"]
    if tps["procrustes"] < sim["procrustes"] - 0.02 and tps["pearson_r"] > 0.95:
        verdict = "supported"
    elif tps["pearson_r"] > 0.97 and tps["iou"] > 0.45:
        verdict = "mixed"
    else:
        verdict = "refuted"
    ref = load_reference()
    RESULT_JSON.write_text(
        json.dumps(
            {
                "claim_id": "T1917-053",
                "date_made": "1917",
                "dataset": {
                    "name": "Thompson (1917) Fig. 387, Internet Archive ongrowthform1917thom p.756",
                    "DOI": None,
                    "url": "https://archive.org/details/ongrowthform1917thom",
                    "sha256": ref["inputs_sha256"],
                    "pre_scouted_dryad": {
                        "identifier": "doi:10.5061/dryad.4nr4cf37",
                        "status": "FILE_BLOCKED_NOT_MISMATCH",
                        "reason": "Correct pterosaur cranial GM clade frame; MorphoJ file OAuth-blocked; not used",
                    },
                },
                "verdict": verdict,
                "headline_numbers": {
                    "procrustes_baseline_dim_pte": m["procrustes_dim_pte"],
                    "tps_procrustes_predicted_vs_drawn_pte": tps["procrustes"],
                    "similarity_procrustes": sim["procrustes"],
                    "affine_procrustes": m["warp"]["affine"]["procrustes"],
                    "pearson_r": tps["pearson_r"],
                    "rmse_over_centroid_size": tps["rmse_cs"],
                    "iou": tps["iou"],
                    "chamfer_rmse_cs": tps["chamfer_rmse_cs"],
                    "shuffled_landmark_null_p": m.get("shuffled_null", {}).get("p_value"),
                    "jacobian_min_det_t1": m["jacobian_min_det"]["t1_grid"],
                    "n_semilandmarks_per_segment": m["n_semi"],
                },
                "script": SCRIPT_REL,
                "figure": str(COMPARISON_PNG.relative_to(ROOT)),
                "animation": str(GROWTH_GIF.relative_to(ROOT)),
                "lean": "morphoeval/lean/loops/Thompson1917_T1917_053.lean",
                "pass1": pass1,
                "parameter_sweep": m.get("parameter_sweep"),
                "specimen_cv": m.get("specimen_cv"),
                "notes": [
                    "n=2 taxa as drawn by Thompson; no independent noise-floor from Foth Dryad (file blocked).",
                    "Verdict tests whether a global low-parameter TPS grid warp reproduces the drawn Pteranodon better than similarity.",
                ],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def self_test() -> None:
    verify_inputs(load_reference())
    m, cfg_dim, cfg_pte, tps, pred_dense, tgt_dense = run_analysis(n_semi=12, lam=0.0, pass1=True)
    assert m["warp"]["tps"]["procrustes"] < m["warp"]["copy_dimorphodon"]["procrustes"]
    assert len(cfg_dim["X"]) == 1 + 4 * 12
    warped = tps_apply(tps, cfg_dim["X"])
    assert warped.shape == cfg_pte["X"].shape
    m2, _, _, _, _, _ = run_analysis(n_semi=12, lam=0.0, pass1=False)
    assert "shuffled_null" in m2
    print("self-test OK", m["warp"]["tps"]["procrustes"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--pass1", action="store_true", help="Pass 1 baseline (no sweep/null/CV)")
    ap.add_argument("--n-semi", type=int, default=20)
    ap.add_argument("--lam", type=float, default=0.0)
    args = ap.parse_args()
    if args.self_test:
        self_test()
        return
    t0 = time.time()
    m, cfg_dim, cfg_pte, tps, pred_dense, tgt_dense = run_analysis(
        n_semi=args.n_semi, lam=args.lam, pass1=args.pass1
    )
    if not args.pass1:
        # attach full sweep once (avoid recursion cost in default run — compute lightweight)
        if not m.get("parameter_sweep"):
            m["parameter_sweep"] = parameter_sweep()
    write_figures(m, cfg_dim, cfg_pte, pred_dense, tgt_dense, tps)
    write_result(m, pass1=args.pass1)
    if not args.pass1:
        print(json.dumps({x["name"]: x["value"] for x in measure_morphoeval(args.n_semi, args.lam)}, indent=2))
    print(f"Wrote {COMPARISON_PNG} and {RESULT_JSON} in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()

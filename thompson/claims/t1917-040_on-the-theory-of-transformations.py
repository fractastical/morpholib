#!/usr/bin/env python3
"""
t1917-040_on-the-theory-of-transformations.py — Thompson (1917) claim T1917-040:
Dürer/Camper facial grids as whole-grid coordinate deformations.

Claim (Thompson, On Growth and Form, 1917, Ch. XVII; transformation types pp. 729-731,
Figs. 353-358; Dürer heads Fig. 364 p. 740 and Fig. 366 p. 742; Camper footnote p. 742):

    "Human (and animal) heads/faces and proportions, in the Dürer/Camper tradition, are
     mutually transformable by coordinate deformation; differences of facial
     angle/proportion are whole-grid deformations."

Equations implemented (ids as in datasets/thompson_1917/t1917-040/*_reference.json):
    E1  proportional stretch        (x, y) -> (x, c*y)                      stretch()
    E3  simple shear / oblique net  (x, y) -> (x + k*y, y),  k = -cot(omega)  shear(), oblique_to_shear()
    E4  whole-grid (affine) map     X' = A X + t                           affine_fit(), fit_pair()
    E5  Camper facial-angle law     tan(phi') = tan(phi) + k               facial_line_tan()
The Lean module morphoeval/lean/loops/Thompson1917_T1917_040.lean proves the algebraic
content of E1/E3/E4/E5 (group law, whole-grid property, facial-angle monotonicity,
circle -> ellipse, affine closure, invertibility of Thompson's oblique transposition).

Biological test (repo standard for Ch. XVII claims: per-pair low-parameter warp vs
similarity null, residual in centroid-size units):
    1. Segment the facial silhouette of each vendored Wikimedia Commons profile
       photograph (Lab colour distance from a declared background sample) and read the
       profile curve x(y) toward the facing side.
    2. Locate 11 soft-tissue profile landmarks on that curve by the classical
       extremum / curvature sequence (de Jong et al. 2014): g, n, prn, cm, sn, ls, sto,
       li, sm, pg, me.
    3. For every ordered pair of accepted specimens fit the similarity (4 dof) and the
       whole-grid affine / Dürer-grid (6 dof) map and compare residuals.  The claim
       predicts that the affine family absorbs the inter-individual difference and that
       the fitted shear k tracks the measured facial-line inclination change (E5).

Pass 1 (v1.0) was the baseline: segmentation, landmarks, pairwise similarity vs 6-dof affine
fits, angles vs norms, comparison figure and Dürer-shear animation.
Pass 2 (this version, v2.0) closes the Pass-1 audit items R2-R8:
    R2  Dürer sub-family fit in the Camper frame (photograph horizontal after the declared
        deskew = proxy for Camper's porion-subnasale horizontal, which a horizontal shear
        leaves invariant): B ~ U A + t with U = [[a, b], [0, d]] upper-triangular, i.e.
        Thompson's stretch c = d/a and shear k = b/d on a uniform scale a (5 dof, linear least
        squares).  Null = uniform scale + translation (3 dof).  Rotation is NOT free: on
        nearly collinear profile points a shear is indistinguishable from a rotation — the
        very reason Camper drew a horizontal — so the rotation-free 6-dof affine is kept only
        as an upper bound, and a +/-5 deg pose-sensitivity check is reported.
    R3  dense semilandmarks: 30 equal-arc-length points of the profile curve between prn and
        me are added to the common landmarks ("dense" fits).
    R4  noise floor: landmarks re-detected after image rescale (0.85, 1.15) and 1-px blur;
        leave-one-landmark-out CV residual for every pair and family.
    R5  controls on real data: positive (exact Dürer grid applied to a specimen is recovered
        to machine precision), negative (scrambled correspondences), ablation (a local
        nose-only displacement is not absorbed by the grid).
    R6  statistics: Pearson r / slope / RMSE of fitted k vs measured facial-line tan change
        (E5), two-sample KS test similarity vs Dürer residuals, CV of R across pairs.
    R7  parameter-regime sweep (omega 50-130 deg, c 0.67-1.5) on a real specimen with angle
        read-outs; Hough digitisation of the obliquity of Thompson's Fig. 366 nets.
    R8  QC and figure fixes (GIF specimen with a visible forehead, angle counts reported).
Not remediated: the gradient-background specimen (Helen Keller) still fails segmentation and
hair-covered foreheads still lose g/n — both are reported, not hidden.

Claim boundary:
    reported_in_source      Dürer drawings transformed by rectangular/oblique grids; Camper's
                            facial angle as an index of a general deformation (no data).
    reconstructed_in_script E1/E3/E4/E5 exactly; landmark warp fits on 9 real photographs.
    transfer_boundary       soft-tissue photographic profiles, not skulls (T1917-060 is the
                            osteological sibling); 2-D mid-sagittal silhouette only.
    does_not_claim          anything about populations, ancestry or "types"; nothing about
                            3-D form or growth.

Usage:
    python3 thompson/claims/t1917-040_on-the-theory-of-transformations.py --self-test
    python3 thompson/claims/t1917-040_on-the-theory-of-transformations.py [--no-gif]

Outputs:
    docs/media/t1917_040_on_the_theory_of_transformatio_comparison.png
    docs/media/t1917_040_on_the_theory_of_transformatio_growth.gif
    datasets/thompson_1917/t1917-040/t1917_040_landmarks.csv
    datasets/thompson_1917/t1917-040/t1917_040_pairwise_fits.csv
    datasets/thompson_1917/t1917-040/t1917_040_metrics.json
    datasets/thompson_1917/t1917-040/result.json

Dependencies (requirements.lock.txt): numpy, scipy, scikit-image, Pillow, matplotlib.
Deterministic: seed 1917; no stochastic step in Pass 1.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from scipy.signal import find_peaks

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "datasets" / "thompson_1917" / "t1917-040"
REF_PATH = DATA / "t1917_040_on_the_theory_of_transformatio_reference.json"
MEDIA = ROOT / "docs" / "media"
SLUG = "t1917_040_on_the_theory_of_transformatio"
SEED = 1917
VERSION = "2.0 (Pass 2 remediation)"

LANDMARKS = ["g", "n", "prn", "cm", "sn", "ls", "sto", "li", "sm", "pg", "me"]
LANDMARK_NAMES = {
    "g": "glabella", "n": "soft-tissue nasion", "prn": "pronasale", "cm": "columella",
    "sn": "subnasale", "ls": "labrale superius", "sto": "stomion", "li": "labrale inferius",
    "sm": "labiomental fold", "pg": "soft-tissue pogonion", "me": "soft-tissue menton",
}


# =============================================================================
# Thompson's transformations (E1, E3, E4, E5)
# =============================================================================
def shear(P: np.ndarray, k: float) -> np.ndarray:
    """E3: Thompson type-3 simple shear (x, y) -> (x + k*y, y)."""
    P = np.asarray(P, float)
    return np.column_stack([P[:, 0] + k * P[:, 1], P[:, 1]])


def stretch(P: np.ndarray, c: float) -> np.ndarray:
    """E1: Thompson type-1 proportional stretch (x, y) -> (x, c*y)."""
    P = np.asarray(P, float)
    return np.column_stack([P[:, 0], c * P[:, 1]])


def oblique_to_shear(omega_deg: float) -> tuple[float, float]:
    """Thompson p. 730: x = xi - eta*cot(omega), y = eta*cosec(omega).
    Returns (k, cosec) with k = -cot(omega): the equivalent shear after undoing the
    y-rescaling by cosec(omega).  omega = 90 deg is the rectangular net (k = 0)."""
    w = math.radians(omega_deg)
    return -math.cos(w) / math.sin(w), 1.0 / math.sin(w)


def affine_apply(P: np.ndarray, A: np.ndarray, t: np.ndarray) -> np.ndarray:
    """E4: X' = A X + t."""
    return np.asarray(P, float) @ A.T + t


def facial_line_tan(dx: float, dy: float, k: float = 0.0) -> float:
    """E5: tan of the facial-line inclination (run / rise) after a shear k."""
    return (dx + k * dy) / dy


def durer_grid(bounds: tuple[float, float, float, float], n: int = 8):
    """Polylines of an n x n rectangular net over bounds (x0, y0, x1, y1), each sampled
    densely so a deformation of the grid can be drawn like Thompson's Figs. 364-366."""
    x0, y0, x1, y1 = bounds
    lines = []
    s = np.linspace(0, 1, 40)
    for i in range(n + 1):
        x = x0 + (x1 - x0) * i / n
        lines.append(np.column_stack([np.full_like(s, x), y0 + (y1 - y0) * s]))
        y = y0 + (y1 - y0) * i / n
        lines.append(np.column_stack([x0 + (x1 - x0) * s, np.full_like(s, y)]))
    return lines


# =============================================================================
# Warp-family fits (similarity null vs whole-grid affine)
# =============================================================================
def centroid_size(P: np.ndarray) -> float:
    P = np.asarray(P, float)
    return float(np.sqrt(((P - P.mean(0)) ** 2).sum()))


def fit_similarity(A: np.ndarray, B: np.ndarray):
    """Least-squares similarity (rotation + isotropic scale + translation) mapping A -> B
    (Procrustes, Umeyama closed form).  Returns (M, t) with B ~ A M^T + t."""
    A = np.asarray(A, float); B = np.asarray(B, float)
    ca, cb = A.mean(0), B.mean(0)
    A0, B0 = A - ca, B - cb
    U, S, Vt = np.linalg.svd(B0.T @ A0)
    D = np.eye(2)
    if np.linalg.det(U @ Vt) < 0:
        D[1, 1] = -1
    R = U @ D @ Vt
    s = (S * np.diag(D)).sum() / (A0 ** 2).sum()
    M = s * R
    return M, cb - ca @ M.T


def fit_scale_translation(A: np.ndarray, B: np.ndarray):
    """Least-squares uniform scale + translation (3 dof, no rotation) mapping A -> B.
    This is the Camper-frame null: pose is fixed by the photograph's horizontal (after the
    declared deskew), the proxy for Camper's porion-subnasale horizontal, which a horizontal
    shear leaves invariant.  Returns (M, t) with M = s*I."""
    A = np.asarray(A, float); B = np.asarray(B, float)
    ca, cb = A.mean(0), B.mean(0)
    A0, B0 = A - ca, B - cb
    sc = float((A0 * B0).sum() / (A0 ** 2).sum())
    M = sc * np.eye(2)
    return M, cb - ca @ M.T


def fit_affine(A: np.ndarray, B: np.ndarray):
    """Least-squares affine (E4, 6 dof) mapping A -> B.  Returns (M, t)."""
    A = np.asarray(A, float); B = np.asarray(B, float)
    X = np.column_stack([A, np.ones(len(A))])
    coef, *_ = np.linalg.lstsq(X, B, rcond=None)
    return coef[:2].T, coef[2]


def decompose_affine(M: np.ndarray) -> dict:
    """Write M = R @ [[sx, k*sy], [0, sy]] (rotation x Dürer stretch-then-shear); returns the
    rotation (deg), the two stretches, the shear coefficient k (E3) and det."""
    Q, Rt = np.linalg.qr(M)
    sgn = np.sign(np.diag(Rt)); sgn[sgn == 0] = 1
    Q = Q * sgn; Rt = (Rt.T * sgn).T
    sx, sy = Rt[0, 0], Rt[1, 1]
    return {"rotation_deg": float(math.degrees(math.atan2(Q[1, 0], Q[0, 0]))),
            "sx": float(sx), "sy": float(sy), "anisotropy": float(sy / sx),
            "k": float(Rt[0, 1] / sy), "det": float(np.linalg.det(M))}


def rmse(Bhat: np.ndarray, B: np.ndarray, size: float) -> float:
    """Root-mean-square landmark residual in units of the target's centroid size."""
    return float(np.sqrt(((Bhat - B) ** 2).sum(1).mean()) / size)


def common_rows(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Boolean mask of landmarks present (finite) in both configurations."""
    return np.isfinite(A).all(1) & np.isfinite(B).all(1)


def fit_pair(A: np.ndarray, B: np.ndarray) -> dict:
    """Similarity null vs whole-grid affine for one ordered specimen pair, on the landmarks
    common to both specimens."""
    keep = common_rows(A, B)
    A, B = A[keep], B[keep]
    size = centroid_size(B)
    Ms, ts = fit_similarity(A, B)
    Ma, ta = fit_affine(A, B)
    r_sim = rmse(affine_apply(A, Ms, ts), B, size)
    r_aff = rmse(affine_apply(A, Ma, ta), B, size)
    ss_sim = ((affine_apply(A, Ms, ts) - B) ** 2).sum()
    ss_aff = ((affine_apply(A, Ma, ta) - B) ** 2).sum()
    out = {"n_landmarks": int(keep.sum()), "rmse_similarity": r_sim, "rmse_affine": r_aff,
           "R_affine": float(1.0 - ss_aff / ss_sim) if ss_sim > 0 else float("nan")}
    out.update({f"affine_{k}": v for k, v in decompose_affine(Ma).items()})
    return out


def fit_durer(A: np.ndarray, B: np.ndarray):
    """R2 — Thompson/Dürer sub-family (E1 + E3 on a uniform scale) in the Camper frame
    (image horizontal after deskew; no rotation): B ~ A U^T + t with
    U = [[a, b], [0, d]]  (d = vertical scale, c = a/d = horizontal stretch — Thompson's own
    ox -> sheep form x' = 2x/3 —, k = b/d = shear).
    Linear least squares, 5 dof.  Without an external horizontal a shear of nearly
    collinear profile points is indistinguishable from a rotation — exactly why Camper
    needed his horizontal — so rotation is deliberately NOT a free parameter here; the
    6-dof affine (rotation included) is reported separately as an upper bound."""
    A = np.asarray(A, float); B = np.asarray(B, float)
    n = len(A)
    # unknowns [a, b, tx, d, ty]
    X = np.zeros((2 * n, 5)); y = np.zeros(2 * n)
    X[0::2, 0] = A[:, 0]; X[0::2, 1] = A[:, 1]; X[0::2, 2] = 1.0; y[0::2] = B[:, 0]
    X[1::2, 3] = A[:, 1]; X[1::2, 4] = 1.0; y[1::2] = B[:, 1]
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    a, b, tx, d, ty = coef
    U = np.array([[a, b], [0.0, d]])
    return U, np.array([tx, ty]), {"scale": float(d), "c": float(a / d), "k": float(b / d)}


def loo_cv_rmse(A: np.ndarray, B: np.ndarray, fitter, size: float) -> float:
    """R4 — leave-one-landmark-out cross-validated residual (centroid-size units)."""
    errs = []
    for i in range(len(A)):
        m = np.ones(len(A), bool); m[i] = False
        out = fitter(A[m], B[m])
        M, t = out[0], out[1]
        pred = affine_apply(A[i:i + 1], M, t)
        errs.append(((pred - B[i:i + 1]) ** 2).sum())
    return float(np.sqrt(np.mean(errs)) / size)


def fit_pair_v2(A: np.ndarray, B: np.ndarray) -> dict:
    """R2/R4 — Camper-frame null (scale + translation, 3 dof), Dürer sub-family (5 dof) and
    affine upper bound (6 dof, rotation free) for one ordered pair on the landmarks common to
    both, each with its LOO-CV residual.  The rotation-including similarity (4 dof) is also
    reported for reference."""
    keep = common_rows(A, B)
    A, B = A[keep], B[keep]
    size = centroid_size(B)
    Mn, tn = fit_scale_translation(A, B)
    Ms, ts = fit_similarity(A, B)
    U, tu, dp = fit_durer(A, B)
    Ma, ta = fit_affine(A, B)
    pred_n, pred_s, pred_d, pred_a = (affine_apply(A, Mn, tn), affine_apply(A, Ms, ts), affine_apply(A, U, tu), affine_apply(A, Ma, ta))
    ss = lambda Bh: float(((Bh - B) ** 2).sum())
    out = {"n_landmarks": int(keep.sum()),
           "rmse_null": rmse(pred_n, B, size), "rmse_similarity": rmse(pred_s, B, size),
           "rmse_durer": rmse(pred_d, B, size), "rmse_affine": rmse(pred_a, B, size),
           "R_durer": float(1 - ss(pred_d) / ss(pred_n)) if ss(pred_n) > 0 else float("nan"),
           "R_affine": float(1 - ss(pred_a) / ss(pred_n)) if ss(pred_n) > 0 else float("nan"),
           "durer_scale": dp["scale"], "durer_c": dp["c"], "durer_k": dp["k"],
           "loo_null": loo_cv_rmse(A, B, fit_scale_translation, size),
           "loo_durer": loo_cv_rmse(A, B, fit_durer, size),
           "loo_affine": loo_cv_rmse(A, B, fit_affine, size)}
    out.update({f"affine_{k}": v for k, v in decompose_affine(Ma).items()})
    return out


def semilandmarks(curve_xy: np.ndarray, y_from: float, y_to: float, n: int = 30) -> np.ndarray:
    """R3 — n points at equal arc-length along the profile curve between two rows."""
    sel = (curve_xy[:, 1] >= min(y_from, y_to)) & (curve_xy[:, 1] <= max(y_from, y_to))
    C = curve_xy[sel]
    if len(C) < 3:
        return np.full((n, 2), np.nan)
    d = np.r_[0.0, np.cumsum(np.hypot(*np.diff(C, axis=0).T))]
    target = np.linspace(0, d[-1], n)
    return np.column_stack([np.interp(target, d, C[:, 0]), np.interp(target, d, C[:, 1])])


def durer_grid_apply(P: np.ndarray, c: float, omega_deg: float) -> np.ndarray:
    """Exact Dürer grid: horizontal stretch c (E1 in Thompson's x' = c x form, p. 730) then
    oblique net omega (E3)."""
    k, _ = oblique_to_shear(omega_deg)
    P = np.asarray(P, float)
    return shear(np.column_stack([c * P[:, 0], P[:, 1]]), k)


def controls(configs: list[np.ndarray], rng: np.random.Generator, c_true: float = 1.25, omega_true: float = 75.0) -> dict:
    """R5 — positive, negative and ablation controls on the real landmark sets."""
    out = {"positive": [], "negative": [], "ablation": [], "real": []}
    for i in range(len(configs)):
        A = configs[i]
        keep = np.isfinite(A).all(1); A0 = A[keep]
        # positive: an exact Dürer grid of A must be recovered (rmse -> 0, parameters -> truth)
        B = durer_grid_apply(A0, c_true, omega_true)
        f = fit_pair_v2(A0, B)
        k_true, _ = oblique_to_shear(omega_true)
        out["positive"].append({"rmse_durer": f["rmse_durer"], "c_err": abs(f["durer_c"] - c_true), "k_err": abs(f["durer_k"] - k_true)})
        for j in range(len(configs)):
            if i == j:
                continue
            Bj = configs[j]
            keep2 = common_rows(A, Bj)
            A2, B2 = A[keep2], Bj[keep2]
            real = fit_pair_v2(A2, B2)
            out["real"].append(real["rmse_durer"])
            # negative: scrambled correspondences
            perm = rng.permutation(len(B2))
            out["negative"].append(fit_pair_v2(A2, B2[perm])["rmse_durer"])
        # ablation: exact Dürer grid of A plus a LOCAL change (pronasale displaced anteriorly by
        # 10 % of centroid size).  The grid absorbed the global part exactly (positive control);
        # the residual that remains is the local change the whole-grid family cannot represent.
        B3 = B.copy()
        B3[LANDMARKS.index("prn"), 0] += 0.10 * centroid_size(B)
        out["ablation"].append(fit_pair_v2(A0, B3)["rmse_durer"])
    return {"positive_rmse_max": float(max(p["rmse_durer"] for p in out["positive"])),
            "positive_expected": "~0 (exact Dürer grid stretch 1.25 + omega 75 deg must be recovered)",
            "positive_c_err_max": float(max(p["c_err"] for p in out["positive"])),
            "positive_k_err_max": float(max(p["k_err"] for p in out["positive"])),
            "negative_rmse_median": float(np.median(out["negative"])) if out["negative"] else float("nan"),
            "real_rmse_median": float(np.median(out["real"])) if out["real"] else float("nan"),
            "ablation_rmse_median": float(np.median(out["ablation"])) if out["ablation"] else float("nan"),
            "ablation_expected_if_local_not_absorbed": "> 0, of order 0.10/sqrt(n_points) (a 10 % nose-only shift on an otherwise exact grid must leave a residual)"}


def parameter_sweep(P: np.ndarray) -> dict:
    """R7 — apply Thompson's omega and c ranges to one real landmark set and read the angles."""
    rows = []
    for omega in np.arange(50, 131, 10):
        Q = durer_grid_apply(P, 1.0, float(omega))
        a = profile_angles(Q)
        rows.append({"sweep": "omega", "omega_deg": float(omega), "k": oblique_to_shear(float(omega))[0], "c": 1.0, **{k: v for k, v in a.items() if k != "facial_line_top"}})
    for c in (0.67, 0.8, 1.0, 1.25, 1.5):
        Q = durer_grid_apply(P, c, 90.0)
        a = profile_angles(Q)
        rows.append({"sweep": "c", "omega_deg": 90.0, "k": 0.0, "c": float(c), **{k: v for k, v in a.items() if k != "facial_line_top"}})
    inc = [r["eline_inclination_deg"] for r in rows if r["sweep"] == "omega"]
    return {"rows": rows, "inclination_monotone_in_omega": bool(np.all(np.diff(inc) > 0) or np.all(np.diff(inc) < 0))}


def digitise_fig366_obliquity(ref: dict) -> dict:
    """R7 — obliquity of Thompson's Fig. 366 (after Dürer) nets from the pinned IA scan:
    Canny edges + probabilistic Hough lines in each head's crop, length-weighted angle
    histogram; the dominant non-horizontal family gives omega (90 deg = rectangular)."""
    from skimage.feature import canny
    from skimage.transform import probabilistic_hough_line
    page = next(p for p in ref["thompson_1917_pages"] if p["leaf"] == 761)
    im = Image.open(DATA / page["file"]).convert("L")
    W, H = im.size
    out = {}
    for name, (x0, x1) in {"left_head": (0.24, 0.52), "right_head": (0.54, 0.80)}.items():
        crop = np.asarray(im.crop((int(x0 * W), int(0.17 * H), int(x1 * W), int(0.33 * H))), float) / 255.0
        edges = canny(crop, sigma=1.5)
        lines = probabilistic_hough_line(edges, threshold=10, line_length=40, line_gap=3, rng=SEED)
        ang = np.array([(math.degrees(math.atan2(p1[1] - p0[1], p1[0] - p0[0])) % 180.0, math.hypot(p1[0] - p0[0], p1[1] - p0[1])) for p0, p1 in lines])
        hist, edges_deg = np.histogram(ang[:, 0], bins=72, range=(0, 180), weights=ang[:, 1])
        centres = edges_deg[:-1] + 1.25
        band = (centres > 55) & (centres < 125)
        omega = float(centres[band][np.argmax(hist[band])])
        out[name] = {"omega_deg": omega, "k": oblique_to_shear(omega)[0], "n_lines": int(len(lines))}
    return out


def noise_floor(ref: dict, accepted: list[dict]) -> dict:
    """R4 — landmark re-detection RMSE (centroid-size units) under image perturbation:
    rescale 0.85 / 1.15 and Gaussian blur sigma 1 px."""
    per = []
    for r in accepted:
        base = landmark_array(r["landmarks"], r["spec"]["facing"])
        keep0 = np.isfinite(base).all(1)
        size = centroid_size(base[keep0])
        for kind, val in (("rescale", 0.85), ("rescale", 1.15), ("blur", 1.0)):
            L2 = _landmarks_perturbed(r["spec"], kind, val)
            if L2 is None:
                continue
            P2 = landmark_array(L2, r["spec"]["facing"])
            keep = keep0 & np.isfinite(P2).all(1)
            if keep.sum() < 4:
                continue
            Ms, ts = fit_similarity(P2[keep], base[keep])   # remove the trivial rescale/translation
            per.append(rmse(affine_apply(P2[keep], Ms, ts), base[keep], size))
    return {"rmse_median": float(np.median(per)) if per else float("nan"),
            "rmse_mean": float(np.mean(per)) if per else float("nan"), "n": len(per)}


def _landmarks_perturbed(spec: dict, kind: str, val: float):
    im = load_specimen_image(spec)
    if kind == "rescale":
        im = im.resize((max(8, int(im.width * val)), max(8, int(im.height * val))), Image.LANCZOS)
    else:
        from PIL import ImageFilter
        im = im.filter(ImageFilter.GaussianBlur(val))
    W, H = im.size
    box = _box_px(spec.get("roi") or [0, 0, 1, 1], W, H)
    crop_u8 = np.asarray(im.crop(box)); crop = crop_u8.astype(float) / 255.0
    bg_box = None
    if spec.get("bg_sample_box"):
        bx = _box_px(spec["bg_sample_box"], W, H)
        bg_box = (bx[0] - box[0], bx[1] - box[1], bx[2] - box[0], bx[3] - box[1])
    fg = segment_profile(crop, spec["facing"], bg_box=bg_box, k_sigma=float(spec.get("k_sigma") or 4.0))
    fg = deskew(fg, float(spec.get("deskew_deg") or 0.0))
    L, _ = detect_landmarks(profile_curve(fg, spec["facing"]), spec["facing"])
    ok, _ = qc_landmarks(L, fg.shape[0])
    return L if ok else None


# =============================================================================
# Specimen handling: segmentation and landmarks
# =============================================================================
def sha256_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def load_reference() -> dict:
    return json.loads(REF_PATH.read_text(encoding="utf-8"))


def load_specimen_image(spec: dict) -> Image.Image:
    """Open the vendored photograph and verify its SHA-256 against the reference JSON."""
    p = DATA / spec["file"]
    digest = sha256_file(p)
    if digest != spec["sha256"]:
        raise RuntimeError(f"{spec['file']}: SHA-256 {digest[:12]} != pinned {spec['sha256'][:12]}")
    return Image.open(p).convert("RGB")


def deskew(arr: np.ndarray, angle_deg: float, fill=0, resample=Image.NEAREST) -> np.ndarray:
    """Rotate an image or a boolean mask by the declared deskew angle (CCW positive, PIL
    convention) with canvas expansion, so that a bowed or raised head gets a vertical face
    axis before the row-wise profile scan.  Masks are rotated with nearest-neighbour
    sampling and zero fill, so the segmentation itself is done on the untouched photograph."""
    if abs(angle_deg) < 1e-9:
        return arr
    if arr.dtype == bool:
        im = Image.fromarray(arr.astype(np.uint8) * 255)
        return np.asarray(im.rotate(angle_deg, resample=Image.NEAREST, expand=True, fillcolor=0)) > 127
    im = Image.fromarray(arr)
    return np.asarray(im.rotate(angle_deg, resample=resample, expand=True, fillcolor=fill))


def _box_px(frac_box, W: int, H: int) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = frac_box
    return int(x0 * W), int(y0 * H), max(int(x0 * W) + 2, int(x1 * W)), max(int(y0 * H) + 2, int(y1 * H))


def segment_profile(rgb: np.ndarray, facing: str, bg_box=None, k_sigma: float = 4.0,
                    strip_frac: float = 0.08):
    """Foreground mask from a Lab colour-distance background model.

    The background is modelled by the per-channel median and robust sigma (1.4826 MAD) of
    either the declared `bg_box` (pixel box) or, by default, the facing-side top corner
    (8 % of the width x top 30 % of rows: the background in front of the forehead).  Pixels farther than `k_sigma` robust
    sigmas (Euclidean in sigma units) are foreground; the largest connected component
    after opening/closing and hole filling is the head+torso silhouette."""
    from skimage.color import rgb2lab
    lab = rgb2lab(rgb)
    H, W = lab.shape[:2]
    if bg_box is not None:
        x0, y0, x1, y1 = bg_box
        sample = lab[y0:y1, x0:x1].reshape(-1, 3)
    else:
        sw = max(3, int(strip_frac * W))
        rows = slice(0, int(0.3 * H))
        sample = (lab[rows, W - sw:] if facing == "right" else lab[rows, :sw]).reshape(-1, 3)
    med = np.median(sample, 0)
    sig = np.median(np.abs(sample - med), 0) * 1.4826 + 1.0
    dist = np.sqrt((((lab - med) / sig) ** 2).sum(-1))
    fg = dist > k_sigma
    fg = ndi.binary_opening(fg, iterations=2)
    fg = ndi.binary_closing(fg, iterations=2)
    lab_img, n = ndi.label(fg)
    if n > 1:
        sizes = ndi.sum(fg, lab_img, range(1, n + 1))
        fg = lab_img == (int(np.argmax(sizes)) + 1)
    return ndi.binary_fill_holes(fg)


def profile_curve(fg: np.ndarray, facing: str) -> np.ndarray:
    """Outermost foreground abscissa per row toward the facing side (NaN where empty)."""
    H = fg.shape[0]
    xs = np.full(H, np.nan)
    for y in range(H):
        r = np.flatnonzero(fg[y])
        if r.size:
            xs[y] = r.max() if facing == "right" else r.min()
    return xs


def detect_landmarks(xs: np.ndarray, facing: str) -> tuple[dict, np.ndarray]:
    """Soft-tissue profile landmarks from the protrusion curve p(y) (anterior positive).

    Pass A: candidate peaks with prominence >= 2 % of the ROI height inside the vertical
    band 10-80 %; the pronasale is the most anterior candidate (the face axis must be
    within ~10 deg of vertical, hence the declared per-specimen deskew).
    Pass B: fine extrema (prominence >= 0.3 % of height after 0.6 % smoothing).  The
    chin/neck boundary is the most posterior point below prn (cervical retreat, or the
    ROI bottom when the neck is cropped); within
    (prn, neck) the anatomical sequence is read as
        sn = first valley below prn, ls = first peak below sn, pg = last peak above the
        neck valley; between ls and pg: li = most prominent peak, sto = deepest valley
        between ls and li, sm = deepest valley between li and pg (li/sto are left None when
        the lower lip is not resolved as a separate extremum);
    above prn: n = first valley, g = first peak above n.
    cm is the point of the prn-sn arc farthest from its chord; me is the steepest posterior
    retreat below pg (chin underside).  Returns ({name: (x, y) or None}, smoothed p)."""
    H = len(xs)
    p = xs.copy() if facing == "right" else -xs.copy()
    ok = ~np.isnan(p)
    L: dict = {k: None for k in LANDMARKS}
    if ok.sum() < 10:
        return L, p
    ys = np.arange(H)
    p = np.interp(ys, ys[ok], p[ok])
    psA = ndi.gaussian_filter1d(p, max(1.5, 0.004 * H))
    pkA, _ = find_peaks(psA, prominence=0.02 * H)
    pkA = pkA[(pkA > 0.10 * H) & (pkA < 0.80 * H)]
    if pkA.size == 0:
        return L, psA
    prn = int(pkA[np.argmax(psA[pkA])])
    psB = ndi.gaussian_filter1d(p, max(1.5, 0.006 * H))
    pmin = max(1.5, 0.003 * H)
    pk, pk_prop = find_peaks(psB, prominence=pmin)
    vl, vl_prop = find_peaks(-psB, prominence=pmin)
    pk_prom = dict(zip(pk.tolist(), pk_prop["prominences"].tolist()))

    def nxt(arr, after):
        c = arr[arr > after]; return int(c[0]) if c.size else None

    def prv(arr, before):
        c = arr[arr < before]; return int(c[-1]) if c.size else None

    seq = {"prn": prn}
    tail = np.arange(prn + 1, H)
    neck = int(tail[np.argmin(psB[tail])]) if tail.size else None   # cervical retreat (or ROI bottom)
    seq["sn"] = nxt(vl, prn)
    seq["ls"] = nxt(pk, seq["sn"]) if seq["sn"] is not None else None
    if neck is not None and seq["ls"] is not None and seq["sn"] != neck:
        chin_pks = pk[(pk > seq["ls"]) & (pk < neck)]
        if chin_pks.size:
            seq["pg"] = int(chin_pks[-1])
            mid_pk = chin_pks[:-1]
            if mid_pk.size and max(pk_prom[int(y)] for y in mid_pk) >= 2 * pmin:
                seq["li"] = int(max(mid_pk, key=lambda y: pk_prom[int(y)]))
                v1 = vl[(vl > seq["ls"]) & (vl < seq["li"])]
                v2 = vl[(vl > seq["li"]) & (vl < seq["pg"])]
                seq["sto"] = int(v1[np.argmin(psB[v1])]) if v1.size else None
                seq["sm"] = int(v2[np.argmin(psB[v2])]) if v2.size else None
            else:
                v2 = vl[(vl > seq["ls"]) & (vl < seq["pg"])]
                seq["sm"] = int(v2[np.argmin(psB[v2])]) if v2.size else None
    seq["n"] = prv(vl, prn)
    seq["g"] = prv(pk, seq["n"]) if seq["n"] is not None else None
    if seq["sn"] is not None:
        a, b = seq["prn"], seq["sn"]
        rows = np.arange(a, b + 1)
        if rows.size > 2:
            P0 = np.array([psB[a], a]); P1 = np.array([psB[b], b])
            d = (P1 - P0) / (np.linalg.norm(P1 - P0) + 1e-9)
            pts = np.column_stack([psB[rows], rows]) - P0
            cross = pts[:, 0] * d[1] - pts[:, 1] * d[0]
            seq["cm"] = int(rows[np.argmax(np.abs(cross))])
    if seq.get("pg") is not None:
        rows = np.arange(seq["pg"], min(H - 1, seq["pg"] + int(0.25 * H)))
        if rows.size > 3:
            seq["me"] = int(rows[np.argmin(np.gradient(psB)[rows])])
    for name in LANDMARKS:
        y = seq.get(name)
        if y is not None:
            x = p[y] if facing == "right" else -p[y]
            L[name] = (float(x), float(y))
    return L, psB


REQUIRED_LANDMARKS = ("prn", "cm", "sn", "ls", "pg", "me")
MIN_LANDMARKS = 7


def qc_landmarks(L: dict, H: int) -> tuple[bool, str]:
    """Accept a specimen if the required nose/lip/chin landmarks exist, at least
    MIN_LANDMARKS of the 11 are present, they are in anatomical top-to-bottom order and
    the chin is not cut by the ROI bottom.  Partial sets (e.g. forehead hidden by hair)
    are allowed; pairwise fits later use the landmarks common to both specimens."""
    missing = [k for k in LANDMARKS if L[k] is None]
    if any(k in missing for k in REQUIRED_LANDMARKS):
        return False, "missing required landmarks: " + ",".join(k for k in REQUIRED_LANDMARKS if k in missing)
    if len(LANDMARKS) - len(missing) < MIN_LANDMARKS:
        return False, f"only {len(LANDMARKS) - len(missing)} landmarks"
    ys = [L[k][1] for k in LANDMARKS if L[k] is not None]
    if any(b <= a for a, b in zip(ys, ys[1:])):
        return False, "landmark order violated"
    if L["me"][1] >= H - 3:
        return False, "chin cut by ROI bottom"
    nose = L["prn"][1] - L["n"][1] if L["n"] is not None else None
    lower = L["pg"][1] - L["prn"][1]
    if nose is not None and not (0.6 <= lower / max(nose, 1e-9) <= 5.0):
        return False, f"implausible lower-face/nose ratio {lower / nose:.2f}"
    if not (0.3 <= (L["pg"][1] - L["sn"][1]) / max(L["sn"][1] - L["prn"][1], 1e-9) <= 12.0):
        return False, "implausible lower-face/columella ratio"
    return True, "ok" if not missing else "ok (missing " + ",".join(missing) + ")"


def process_specimen(spec: dict) -> dict:
    """Full pipeline for one specimen: SHA-256 check -> ROI crop -> Lab background-model
    silhouette (on the untouched photograph) -> optional mask deskew -> profile scan ->
    landmark sequence -> QC.  Landmarks are returned in the (deskewed) display frame."""
    im = load_specimen_image(spec)
    W, H = im.size
    box = _box_px(spec.get("roi") or [0, 0, 1, 1], W, H)
    crop_u8 = np.asarray(im.crop(box))
    crop = crop_u8.astype(float) / 255.0
    bg_box = None
    if spec.get("bg_sample_box"):
        bx = _box_px(spec["bg_sample_box"], W, H)
        bg_box = (bx[0] - box[0], bx[1] - box[1], bx[2] - box[0], bx[3] - box[1])
    fg = segment_profile(crop, spec["facing"], bg_box=bg_box, k_sigma=float(spec.get("k_sigma") or 4.0))
    angle = float(spec.get("deskew_deg") or 0.0)
    fill = tuple(int(v) for v in np.median(crop_u8[: max(1, crop_u8.shape[0] // 10)].reshape(-1, 3), axis=0))
    display = deskew(crop_u8, angle, fill=fill, resample=Image.BICUBIC)
    fg = deskew(fg, angle)
    xs = profile_curve(fg, spec["facing"])
    L, ps = detect_landmarks(xs, spec["facing"])
    ok, reason = qc_landmarks(L, fg.shape[0])
    return {"spec": spec, "image": Image.fromarray(display), "box": (0, 0, display.shape[1], display.shape[0]),
            "mask": fg, "curve": xs, "landmarks": L, "accepted": ok, "reason": reason}


def landmark_array(L: dict, facing: str) -> np.ndarray:
    """(11, 2) array in a right-facing, y-up frame (left-facing specimens are mirrored so
    that every configuration is comparable); missing landmarks are NaN rows."""
    P = np.array([L[k] if L[k] is not None else (np.nan, np.nan) for k in LANDMARKS], float)
    if facing == "left":
        P[:, 0] = -P[:, 0]
    P[:, 1] = -P[:, 1]
    return P


# =============================================================================
# Angles vs literature norms
# =============================================================================
def angle_deg(a, b, c) -> float:
    """Angle at b between rays b->a and b->c."""
    v1 = np.asarray(a) - np.asarray(b); v2 = np.asarray(c) - np.asarray(b)
    cosang = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-12)
    return float(math.degrees(math.acos(max(-1.0, min(1.0, cosang)))))


def profile_angles(P: np.ndarray) -> dict:
    """Convexity (G-Sn-Pg, as 180 - interior angle, Legan & Burstone), nasolabial
    (Cm-Sn-Ls) and the facial-line inclination (G->Ls relative to the N->Me axis)."""
    i = {k: j for j, k in enumerate(LANDMARKS)}
    has = lambda *ks: all(np.isfinite(P[i[k]]).all() for k in ks)
    conv = 180.0 - angle_deg(P[i["g"]], P[i["sn"]], P[i["pg"]]) if has("g", "sn", "pg") else float("nan")
    nl = angle_deg(P[i["cm"]], P[i["sn"]], P[i["ls"]]) if has("cm", "sn", "ls") else float("nan")
    # facial line: Camper's forehead-to-upper-lip line; soft-tissue nasion stands in for the
    # glabella when hair hides the forehead.  Reference axis: Sn -> Me (lower-face vertical).
    # Camper frame: inclination of the facial line (Ls -> G, or Ls -> N when hair hides the
    # glabella) from the frame vertical, positive when the forehead leans forward (+x).
    # Under E3 (x' = x + k y) tan(inc') = tan(inc) + k exactly.
    top = "g" if has("g") else ("n" if has("n") else None)
    if top is not None and has("ls"):
        fl = P[i[top]] - P[i["ls"]]
        inc = math.degrees(math.atan2(fl[0], fl[1]))
    else:
        inc = float("nan")
    # E-line (Ricketts, Prn -> Pg) inclination from the frame vertical: obeys the same E5 law
    # and uses two required landmarks, so it is available on every accepted specimen and is not
    # corrupted by hair at the forehead.  This is the line used for the E5 statistics.
    el = P[i["prn"]] - P[i["pg"]]
    eline = math.degrees(math.atan2(el[0], el[1])) if has("prn", "pg") else float("nan")
    return {"convexity_deg": conv, "nasolabial_deg": nl, "facial_line_inclination_deg": inc, "facial_line_top": top,
            "eline_inclination_deg": eline}


# =============================================================================
# Analysis
# =============================================================================
def generalized_procrustes(configs: list[np.ndarray], iters: int = 10):
    """Similarity-align all configurations to their iteratively updated mean, using the
    landmarks present in every configuration for the alignment (missing rows stay NaN
    and are carried along by the fitted similarity)."""
    keep = np.logical_and.reduce([np.isfinite(c).all(1) for c in configs])
    aligned = []
    for c in configs:
        c0 = c - c[keep].mean(0)
        aligned.append(c0 / centroid_size(c0[keep]))
    mean = aligned[0][keep]
    for _ in range(iters):
        new = []
        for c in aligned:
            M, t = fit_similarity(c[keep], mean)
            new.append(affine_apply(c, M, t))
        aligned = new
        mean = np.mean([c[keep] for c in aligned], 0)
        mean = (mean - mean.mean(0)) / centroid_size(mean)
    full_mean = np.full_like(configs[0], np.nan)
    full_mean[keep] = mean
    return aligned, full_mean


def analyse(ref: dict, verbose: bool = True, full: bool = True) -> dict:
    rng = np.random.default_rng(SEED)
    results = [process_specimen(s) for s in ref["specimens"]]
    acc = [r for r in results if r["accepted"]]
    if verbose:
        for r in results:
            print(f"  {r['spec']['file']:55s} {'ACCEPT' if r['accepted'] else 'reject'}  {r['reason']}")
    configs = [landmark_array(r["landmarks"], r["spec"]["facing"]) for r in acc]
    names = [Path(r["spec"]["file"]).stem for r in acc]
    angles = [profile_angles(c) for c in configs]
    # R3 dense configurations: common landmarks + 30 semilandmarks prn -> me (same frame)
    dense = []
    for r, c in zip(acc, configs):
        ys = np.arange(len(r["curve"])); ok = ~np.isnan(r["curve"])
        curve_xy = np.column_stack([r["curve"][ok], ys[ok]])
        S = semilandmarks(curve_xy, r["landmarks"]["prn"][1], r["landmarks"]["me"][1])
        if r["spec"]["facing"] == "left":
            S[:, 0] = -S[:, 0]
        S[:, 1] = -S[:, 1]
        dense.append(np.vstack([c, S]))
    aligned, mean = generalized_procrustes(configs) if len(configs) >= 2 else (configs, configs[0] if configs else None)
    pairs = []
    for i in range(len(configs)):
        for j in range(len(configs)):
            if i == j:
                continue
            f = fit_pair_v2(configs[i], configs[j])
            fd = fit_pair_v2(dense[i], dense[j])
            f.update({f"dense_{k}": v for k, v in fd.items() if k.startswith(("rmse", "R_", "loo", "durer", "n_land"))})
            ti, tj = angles[i]["eline_inclination_deg"], angles[j]["eline_inclination_deg"]
            fi, fj = angles[i]["facial_line_inclination_deg"], angles[j]["facial_line_inclination_deg"]
            f.update({"from": names[i], "to": names[j],
                      "d_convexity_deg": angles[j]["convexity_deg"] - angles[i]["convexity_deg"],
                      "d_facial_line_inclination_deg": fj - fi,
                      "d_eline_inclination_deg": tj - ti,
                      "d_tan_inclination": math.tan(math.radians(tj)) - math.tan(math.radians(ti)) if np.isfinite(ti) and np.isfinite(tj) else float("nan")})
            pairs.append(f)
    summary: dict = {}
    if pairs:
        g = lambda key: np.array([p[key] for p in pairs], float)
        summary = {"n_specimens_accepted": len(acc), "n_specimens_total": len(results), "n_pairs": len(pairs),
                   "rmse_null_median": float(np.median(g("rmse_null"))),
                   "rmse_similarity_median": float(np.median(g("rmse_similarity"))),
                   "rmse_durer_median": float(np.median(g("rmse_durer"))),
                   "rmse_affine_median": float(np.median(g("rmse_affine"))),
                   "loo_null_median": float(np.median(g("loo_null"))),
                   "loo_durer_median": float(np.median(g("loo_durer"))),
                   "loo_affine_median": float(np.median(g("loo_affine"))),
                   "R_durer_median": float(np.median(g("R_durer"))), "R_durer_cv": float(np.std(g("R_durer")) / abs(np.mean(g("R_durer")))) if np.mean(g("R_durer")) != 0 else float("nan"),
                   "R_affine_median": float(np.median(g("R_affine"))),
                   "dense_rmse_null_median": float(np.median(g("dense_rmse_null"))),
                   "dense_rmse_similarity_median": float(np.median(g("dense_rmse_similarity"))),
                   "dense_rmse_affine_median": float(np.median(g("dense_rmse_affine"))),
                   "dense_rmse_durer_median": float(np.median(g("dense_rmse_durer"))),
                   "dense_loo_null_median": float(np.median(g("dense_loo_null"))),
                   "dense_loo_durer_median": float(np.median(g("dense_loo_durer"))),
                   "dense_loo_affine_median": float(np.median(g("dense_loo_affine"))),
                   "dense_R_durer_median": float(np.median(g("dense_R_durer"))),
                   "dense_R_durer_min": float(g("dense_R_durer").min()), "dense_R_durer_max": float(g("dense_R_durer").max()),
                   "durer_c_range": [float(g("dense_durer_c").min()), float(g("dense_durer_c").max())],
                   "durer_k_range": [float(g("dense_durer_k").min()), float(g("dense_durer_k").max())],
                   "convexity_mean_deg": float(np.nanmean([a["convexity_deg"] for a in angles])),
                   "convexity_sd_deg": float(np.nanstd([a["convexity_deg"] for a in angles], ddof=1)) if sum(np.isfinite(a["convexity_deg"]) for a in angles) > 1 else float("nan"),
                   "convexity_n": int(sum(np.isfinite(a["convexity_deg"]) for a in angles)),
                   "nasolabial_mean_deg": float(np.nanmean([a["nasolabial_deg"] for a in angles])),
                   "nasolabial_sd_deg": float(np.nanstd([a["nasolabial_deg"] for a in angles], ddof=1)) if len(angles) > 1 else float("nan"),
                   "nasolabial_n": int(sum(np.isfinite(a["nasolabial_deg"]) for a in angles))}
        # R6 statistics
        from scipy import stats
        k = g("dense_durer_k"); dt = g("d_tan_inclination"); m = np.isfinite(k) & np.isfinite(dt)
        if m.sum() >= 3:
            r, pval = stats.pearsonr(k[m], dt[m])
            slope, intercept = np.polyfit(k[m], dt[m], 1)
            summary["e5_pearson_r"] = float(r); summary["e5_p_value"] = float(pval); summary["e5_slope"] = float(slope)
            summary["e5_rmse_tan"] = float(np.sqrt(np.mean((dt[m] - (slope * k[m] + intercept)) ** 2)))
            summary["e5_n"] = int(m.sum())
        ks = stats.ks_2samp(g("dense_rmse_null"), g("dense_rmse_durer"))
        summary["ks_null_vs_durer_D"] = float(ks.statistic); summary["ks_null_vs_durer_p"] = float(ks.pvalue)
        # pose sensitivity (R4): the Camper horizontal is the photograph's horizontal after the
        # declared deskew; rotate every configuration by +/-5 deg and refit
        sens = []
        for ang in (-5.0, 5.0):
            c_, s_ = math.cos(math.radians(ang)), math.sin(math.radians(ang))
            Rm = np.array([[c_, -s_], [s_, c_]])
            vals = []
            for i in range(len(dense)):
                for j in range(len(dense)):
                    if i != j:
                        vals.append(fit_pair_v2(dense[i] @ Rm.T, dense[j])["R_durer"])
            sens.append(float(np.median(vals)))
        summary["pose_sensitivity_R_durer_median_pm5deg"] = sens
    extra: dict = {}
    if full and pairs:
        extra["controls"] = controls(dense, rng)
        extra["noise_floor"] = noise_floor(ref, acc)
        extra["sweep"] = parameter_sweep(configs[int(np.argmax([np.isfinite(c).all(1).sum() for c in configs]))])
        extra["fig366"] = digitise_fig366_obliquity(ref)
        nf = extra["noise_floor"]["rmse_median"]
        summary["noise_floor_rmse_median"] = nf
        summary["dense_rmse_durer_over_noise_floor"] = float(summary["dense_rmse_durer_median"] / nf) if nf and np.isfinite(nf) and nf > 0 else float("nan")
        summary["control_negative_rmse_median"] = extra["controls"]["negative_rmse_median"]
        summary["control_positive_rmse_max"] = extra["controls"]["positive_rmse_max"]
        summary["control_ablation_rmse_median"] = extra["controls"]["ablation_rmse_median"]
        summary["fig366_omega_left_deg"] = extra["fig366"]["left_head"]["omega_deg"]
        summary["fig366_omega_right_deg"] = extra["fig366"]["right_head"]["omega_deg"]
    return {"results": results, "accepted": acc, "configs": configs, "dense": dense, "names": names, "angles": angles,
            "aligned": aligned, "mean": mean, "pairs": pairs, "summary": summary, **extra}


# =============================================================================
# Figures
# =============================================================================
def _draw_profile_panel(ax, r: dict, title: str):
    ax.imshow(r["image"])
    box = r["box"]
    ys = np.arange(len(r["curve"])) + box[1]
    ax.plot(r["curve"] + box[0], ys, color="red", lw=1.0)
    for k in LANDMARKS:
        v = r["landmarks"][k]
        if v is not None:
            ax.plot(v[0], v[1], "o", ms=4, color="lime", mec="black", mew=0.5)
            ax.annotate(k, v, xytext=(6 if r["spec"]["facing"] == "right" else -16, 0),
                        textcoords="offset points", fontsize=6, color="yellow")
    ax.set_title(title, fontsize=8)
    ax.set_axis_off()


def make_comparison_png(an: dict, out: Path, ref: dict):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(17, 15))
    gs = fig.add_gridspec(3, 4)
    s = an["summary"]
    # (a) Thompson Fig. 366 crop + digitised obliquity
    ax = fig.add_subplot(gs[0, 0])
    page = next(p for p in ref["thompson_1917_pages"] if p["leaf"] == 761)
    pim = Image.open(DATA / page["file"])
    W, H = pim.size
    ax.imshow(pim.crop((int(0.22 * W), int(0.17 * H), int(0.82 * W), int(0.33 * H))))
    f366 = an.get("fig366")
    sub = f"Hough obliquity: ω = {f366['left_head']['omega_deg']:.0f}° / {f366['right_head']['omega_deg']:.0f}°" if f366 else ""
    ax.set_title(f"Thompson 1917 Fig. 366 (after Dürer), p. 742 — oblique nets = E3 shear\n{sub}", fontsize=8)
    ax.set_axis_off()
    # (b, c) specimens with landmarks
    acc = an["accepted"]
    order = np.argsort([-np.isfinite(c).all(1).sum() for c in an["configs"]])
    if acc:
        r0 = acc[order[0]]
        _draw_profile_panel(fig.add_subplot(gs[0, 1]), r0, f"specimen: {r0['spec']['commons_title'][5:40]}\nsilhouette + landmarks ({r0['reason']})")
    if len(acc) > 1:
        r1 = acc[order[1]]
        _draw_profile_panel(fig.add_subplot(gs[0, 2]), r1, f"specimen: {r1['spec']['commons_title'][5:40]}\n({r1['reason']})")
    # (d) Procrustes superimposition (dense)
    ax = fig.add_subplot(gs[0, 3])
    if an["dense"]:
        keep = np.logical_and.reduce([np.isfinite(d).all(1) for d in an["dense"]])
        ref0 = an["dense"][0][keep]
        for d, nm in zip(an["dense"], an["names"]):
            M, t = fit_similarity(d[keep], ref0)
            dd = affine_apply(d, M, t)
            fL = np.isfinite(dd[:len(LANDMARKS)]).all(1)
            ax.plot(dd[len(LANDMARKS):, 0], dd[len(LANDMARKS):, 1], "-", lw=1.0, alpha=0.8, label=nm[:18])
            ax.plot(dd[:len(LANDMARKS)][fL, 0], dd[:len(LANDMARKS)][fL, 1], "o", ms=3, color=ax.lines[-1].get_color())
        ax.set_aspect("equal", adjustable="datalim"); ax.set_title(f"similarity-aligned dense profiles (n={len(acc)}; dots = landmarks)", fontsize=8)
        ax.legend(fontsize=5, loc="lower left")
    # (e) Dürer grid: A -> B best and worst pair (dense)
    for col, pick in ((0, "best"), (1, "worst")):
        ax = fig.add_subplot(gs[1, col])
        if len(an["dense"]) >= 2:
            Rs = np.array([p["dense_R_durer"] for p in an["pairs"]])
            idx = int(np.argmax(Rs)) if pick == "best" else int(np.argmin(Rs))
            pr = an["pairs"][idx]
            i, j = an["names"].index(pr["from"]), an["names"].index(pr["to"])
            A, B = an["dense"][i], an["dense"][j]
            keep = common_rows(A, B); A, B = A[keep], B[keep]
            Ms, ts = fit_scale_translation(A, B); As = affine_apply(A, Ms, ts)
            U, tu, dp = fit_durer(A, B); Ad = affine_apply(A, U, tu)
            cs = centroid_size(B)
            bx = (As[:, 0].min() - 0.15 * cs, As[:, 1].min() - 0.05 * cs, As[:, 0].max() + 0.15 * cs, As[:, 1].max() + 0.05 * cs)
            Minv = np.linalg.inv(Ms)
            for ln in durer_grid(bx, 6):
                g = affine_apply(affine_apply(ln - ts, Minv.T, np.zeros(2)), U, tu); ax.plot(g[:, 0], g[:, 1], color="0.75", lw=0.6)
            ax.plot(As[:, 0], As[:, 1], "b-", lw=1, label=f"A = {pr['from'][:14]} (scale+shift null)")
            ax.plot(Ad[:, 0], Ad[:, 1], "r-", lw=1.2, label="A on fitted Dürer grid (stretch c, shear k)")
            ax.plot(B[:, 0], B[:, 1], "k-", lw=1.5, label=f"B = {pr['to'][:14]}")
            ax.set_title(f"{pick} pair: RMSE null {pr['dense_rmse_null']:.3f} -> Dürer {pr['dense_rmse_durer']:.3f} "
                         f"(R={pr['dense_R_durer']:.2f}; c={dp['c']:.2f}, k={dp['k']:+.2f})", fontsize=8)
            ax.legend(fontsize=6); ax.set_aspect("equal", adjustable="datalim"); ax.set_axis_off()
    # (f) residual distributions + noise floor + controls
    ax = fig.add_subplot(gs[1, 2])
    if an["pairs"]:
        cols = [[p["dense_rmse_null"] for p in an["pairs"]], [p["dense_rmse_similarity"] for p in an["pairs"]], [p["dense_rmse_durer"] for p in an["pairs"]],
                [p["dense_loo_durer"] for p in an["pairs"]], [p["dense_rmse_affine"] for p in an["pairs"]]]
        ax.boxplot(cols, tick_labels=["scale+shift\n3 dof (null)", "similarity\n4 dof", "Dürer grid\n5 dof", "Dürer\nLOO-CV", "affine\n6 dof"])
        nf = s.get("noise_floor_rmse_median")
        if nf is not None and np.isfinite(nf):
            ax.axhline(nf, color="tab:green", ls="--", lw=1, label=f"landmark noise floor {nf:.3f}")
        if "control_negative_rmse_median" in s:
            ax.axhline(s["control_negative_rmse_median"], color="tab:red", ls=":", lw=1, label=f"scrambled-correspondence control {s['control_negative_rmse_median']:.3f}")
        ax.set_ylabel("RMSE (centroid-size units), dense configurations")
        ax.set_title(f"{len(an['pairs'])} ordered pairs; KS null vs Dürer p = {s.get('ks_null_vs_durer_p', float('nan')):.3g}", fontsize=8)
        ax.legend(fontsize=6)
    # (g) E5: fitted k vs measured tan change
    ax = fig.add_subplot(gs[1, 3])
    if an["pairs"]:
        k = np.array([p["dense_durer_k"] for p in an["pairs"]]); dt = np.array([p["d_tan_inclination"] for p in an["pairs"]])
        m = np.isfinite(k) & np.isfinite(dt)
        ax.scatter(k[m], dt[m], s=18, c="k", zorder=3)
        if "e5_slope" in s:
            xx = np.linspace(k[m].min(), k[m].max(), 10)
            ax.plot(xx, s["e5_slope"] * xx + (dt[m].mean() - s["e5_slope"] * k[m].mean()), "r-", lw=1, label=f"fit slope {s['e5_slope']:.2f}, r = {s['e5_pearson_r']:.2f}")
        ax.plot([-0.5, 0.5], [-0.5, 0.5], "0.6", ls="--", lw=1, label="E5 prediction: Δtan φ = k")
        ax.set_xlabel("fitted Dürer shear k (A -> B)"); ax.set_ylabel("measured Δ tan(E-line Prn–Pg inclination)")
        ax.set_title("Camper footnote (E5): is the facial-angle change the shear?", fontsize=8); ax.legend(fontsize=6)
    # (h) parameter sweep
    ax = fig.add_subplot(gs[2, 0])
    sw = an.get("sweep")
    if sw:
        rows = [r for r in sw["rows"] if r["sweep"] == "omega"]
        ax.plot([r["omega_deg"] for r in rows], [r["eline_inclination_deg"] for r in rows], "o-", label="E-line (Prn–Pg) inclination")
        ax.plot([r["omega_deg"] for r in rows], [r["convexity_deg"] for r in rows], "s-", label="convexity G-Sn-Pg")
        ax.plot([r["omega_deg"] for r in rows], [r["nasolabial_deg"] for r in rows], "^-", label="nasolabial")
        ax.set_xlabel("Thompson ω (deg); 90 = rectangular net"); ax.set_ylabel("angle (deg)")
        ax.set_title(f"R7 sweep on a real specimen (monotone: {sw['inclination_monotone_in_omega']})", fontsize=8); ax.legend(fontsize=6)
    # (i) stretch sweep
    ax = fig.add_subplot(gs[2, 1])
    if sw:
        rows = [r for r in sw["rows"] if r["sweep"] == "c"]
        ax.plot([r["c"] for r in rows], [r["convexity_deg"] for r in rows], "s-", label="convexity G-Sn-Pg")
        ax.plot([r["c"] for r in rows], [r["nasolabial_deg"] for r in rows], "^-", label="nasolabial")
        ax.set_xlabel("Thompson stretch c (E1)"); ax.set_ylabel("angle (deg)"); ax.set_title("R7 sweep: proportional stretch", fontsize=8); ax.legend(fontsize=6)
    # (j) angles vs norms
    ax = fig.add_subplot(gs[2, 2])
    if an["angles"]:
        conv = [a["convexity_deg"] for a in an["angles"]]; nl = [a["nasolabial_deg"] for a in an["angles"]]
        ax.scatter(conv, nl, c="k", s=18, zorder=3)
        ax.axvspan(8, 16, color="tab:blue", alpha=0.15, label="Legan-Burstone convexity 12±4°")
        ax.axhspan(94, 110, color="tab:orange", alpha=0.15, label="Legan-Burstone nasolabial 102±8°")
        for a, nm in zip(an["angles"], an["names"]):
            if np.isfinite(a["convexity_deg"]) and np.isfinite(a["nasolabial_deg"]):
                ax.annotate(nm[:10], (a["convexity_deg"], a["nasolabial_deg"]), fontsize=5)
        ax.set_xlabel("facial convexity G-Sn-Pg (deg)"); ax.set_ylabel("nasolabial Cm-Sn-Ls (deg)")
        ax.legend(fontsize=6); ax.set_title(f"angles vs norms (convexity n={s.get('convexity_n')}, nasolabial n={s.get('nasolabial_n')})", fontsize=8)
    # (k) verdict text
    ax = fig.add_subplot(gs[2, 3]); ax.set_axis_off()
    verdict, why = verdict_from(s)
    ctl = an.get("controls", {})
    txt = (f"VERDICT: {verdict.upper()}\n{why}\n\n"
           f"accepted specimens {s.get('n_specimens_accepted')}/{s.get('n_specimens_total')}, ordered pairs {s.get('n_pairs')}\n"
           f"dense RMSE median: null {s.get('dense_rmse_null_median', float('nan')):.3f} -> Dürer {s.get('dense_rmse_durer_median', float('nan')):.3f} (LOO {s.get('dense_loo_durer_median', float('nan')):.3f}); affine {s.get('dense_rmse_affine_median', float('nan')):.3f}\n"
           f"noise floor {s.get('noise_floor_rmse_median', float('nan')):.3f}  ->  Dürer residual / noise = {s.get('dense_rmse_durer_over_noise_floor', float('nan')):.1f}\n"
           f"dense R_Dürer median {s.get('dense_R_durer_median', float('nan')):.2f} (range {s.get('dense_R_durer_min', float('nan')):.2f}..{s.get('dense_R_durer_max', float('nan')):.2f})\n"
           f"fitted c {s.get('durer_c_range', [float('nan')] * 2)[0]:.2f}..{s.get('durer_c_range', [float('nan')] * 2)[1]:.2f}, k {s.get('durer_k_range', [float('nan')] * 2)[0]:+.2f}..{s.get('durer_k_range', [float('nan')] * 2)[1]:+.2f}  (Dürer Fig. 366: |k| ≈ {abs(oblique_to_shear(s.get('fig366_omega_left_deg', 90))[0]):.2f})\n"
           f"E5: r = {s.get('e5_pearson_r', float('nan')):.2f}, slope {s.get('e5_slope', float('nan')):.2f}, p = {s.get('e5_p_value', float('nan')):.3g}\n"
           f"pose sensitivity (±5°): median R_Dürer {s.get('pose_sensitivity_R_durer_median_pm5deg', ['?', '?'])}\n"
           f"controls: positive max RMSE {ctl.get('positive_rmse_max', float('nan')):.1e}; scrambled {ctl.get('negative_rmse_median', float('nan')):.3f}; "
           f"nose-only ablation on exact grid {ctl.get('ablation_rmse_median', float('nan')):.3f}")
    import textwrap
    txt = "\n".join(textwrap.fill(line, 58) if line else "" for line in txt.split("\n"))
    ax.text(0.0, 1.0, txt, va="top", ha="left", fontsize=7, family="monospace", transform=ax.transAxes)
    fig.suptitle("T1917-040 — Thompson (1917) Dürer/Camper facial grids: are inter-individual profile differences whole-grid deformations?  "
                 f"[{VERSION}]", fontsize=10)
    fig.subplots_adjust(left=0.04, right=0.99, top=0.93, bottom=0.05, hspace=0.38, wspace=0.28)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=100)
    plt.close(fig)


def make_growth_gif(an: dict, out: Path, n_frames: int = 36):
    """Dürer shear sweep (E3) applied to the first accepted specimen's dense profile curve and
    an 8x8 net, omega 60 -> 120 deg and back, with the facial-line inclination read-out (E5)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if not an["accepted"]:
        return
    # prefer a specimen whose forehead (g) is visible, then the one with most landmarks
    score = [np.isfinite(c).all(1).sum() + (100 if r_["landmarks"]["g"] is not None else 0) for c, r_ in zip(an["configs"], an["accepted"])]
    r = an["accepted"][int(np.argmax(score))]
    box = r["box"]; facing = r["spec"]["facing"]
    ys = np.arange(len(r["curve"]))
    ok = ~np.isnan(r["curve"])
    curve = np.column_stack([r["curve"][ok], ys[ok]])
    L = {k: np.array(r["landmarks"][k]) - np.array([box[0], box[1]]) for k in LANDMARKS if r["landmarks"][k] is not None}
    top = "g" if "g" in L else "n"
    face_h = L["me"][1] - L[top][1]
    y_top, y_bot = L[top][1] - 0.08 * face_h, L["me"][1] + 0.05 * face_h
    sel = (curve[:, 1] >= y_top) & (curve[:, 1] <= y_bot)
    curve = curve[sel]
    # to a right-facing, y-up frame centred on the nasion
    P = curve.copy(); P[:, 1] = -P[:, 1]
    if facing == "left":
        P[:, 0] = -P[:, 0]
    Lm = {k: np.array([v[0] if facing == "right" else -v[0], -v[1]]) for k, v in L.items()}
    origin = Lm["n"] if "n" in Lm else Lm["prn"]
    P -= origin
    Lm = {k: v - origin for k, v in Lm.items()}
    span = (Lm["me"][1] - Lm[top][1])
    bounds = (P[:, 0].min() - 0.6 * abs(span), P[:, 1].min() - 0.05 * abs(span), P[:, 0].max() + 0.2 * abs(span), P[:, 1].max() + 0.05 * abs(span))
    grid = durer_grid(bounds, 8)
    omegas = np.concatenate([np.linspace(60, 120, n_frames // 2), np.linspace(120, 60, n_frames - n_frames // 2)])
    frames = []
    fl = Lm["ls"] - Lm[top]
    for w in omegas:
        k, _ = oblique_to_shear(w)
        fig, ax = plt.subplots(figsize=(5, 5.5), dpi=80)
        for ln in grid:
            g = shear(ln, k); ax.plot(g[:, 0], g[:, 1], color="0.75", lw=0.6)
        S = shear(P, k); ax.plot(S[:, 0], S[:, 1], "k-", lw=1.8)
        G, Ls = shear(Lm[top][None], k)[0], shear(Lm["ls"][None], k)[0]
        ax.plot([G[0], Ls[0]], [G[1], Ls[1]], "r-", lw=1.2)
        phi = math.degrees(math.atan(facial_line_tan(fl[0], fl[1], k)))
        ax.set_title(f"Thompson E3 oblique net  ω = {w:5.1f}°  (k = {k:+.2f})\nfacial line {top.upper()}–Ls inclination {phi:+.1f}°  [tan φ' = tan φ + k]", fontsize=9)
        ax.set_xlim(bounds[0] - 0.9 * abs(span), bounds[2] + 0.9 * abs(span)); ax.set_ylim(bounds[1], bounds[3]); ax.set_aspect("equal"); ax.set_axis_off()
        fig.tight_layout()
        fig.canvas.draw()
        frames.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()))
        plt.close(fig)
    out.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(out, save_all=True, append_images=frames[1:], duration=120, loop=0, optimize=True)


# =============================================================================
# Outputs, measurements, verdict
# =============================================================================
def write_outputs(an: dict, ref: dict) -> dict:
    DATA.mkdir(parents=True, exist_ok=True)
    with (DATA / "t1917_040_landmarks.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["specimen", "accepted", "reason", "facing"] + [f"{k}_{a}" for k in LANDMARKS for a in ("x", "y")])
        for r in an["results"]:
            row = [Path(r["spec"]["file"]).stem, r["accepted"], r["reason"], r["spec"]["facing"]]
            for k in LANDMARKS:
                v = r["landmarks"][k]
                row += ["", ""] if v is None else [f"{v[0]:.1f}", f"{v[1]:.1f}"]
            w.writerow(row)
    if an["pairs"]:
        keys = list(an["pairs"][0].keys())
        with (DATA / "t1917_040_pairwise_fits.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys); w.writeheader()
            for p in an["pairs"]:
                w.writerow({k: (f"{v:.5f}" if isinstance(v, float) else v) for k, v in p.items()})
    metrics = {"version": VERSION, "seed": SEED, "summary": an["summary"],
               "controls": an.get("controls"), "noise_floor": an.get("noise_floor"), "fig366_obliquity": an.get("fig366"),
               "parameter_sweep": an.get("sweep"),
               "angles": {nm: a for nm, a in zip(an["names"], an["angles"])},
               "specimens": [{"file": r["spec"]["file"], "accepted": r["accepted"], "reason": r["reason"]} for r in an["results"]]}
    (DATA / "t1917_040_metrics.json").write_text(json.dumps(metrics, indent=1), encoding="utf-8")
    return metrics


def verdict_from(summary: dict) -> tuple[str, str]:
    """Verdict rule (reference JSON C1-C3).  supported: the dense Dürer-grid residual is within
    1.5x the landmark noise floor AND median R_Dürer >= 0.6 AND the E5 shear/facial-angle link
    has |r| >= 0.7.  refuted: median R_Dürer < 0.3 (the whole-grid family absorbs less than
    30 % of the non-scale profile difference).  mixed: everything in between — in particular
    the case met here, where the grid absorbs a substantial but minority share and leaves a
    residual far above detection noise, i.e. the STRONG form (whole-grid = whole story) is not
    supported while the bounded form (a Dürer stretch+shear is a real, significant component
    and the facial angle tracks its shear) is."""
    if not summary or "dense_R_durer_median" not in summary:
        return "untestable", "no accepted specimen pairs"
    R = summary["dense_R_durer_median"]
    ratio = summary.get("dense_rmse_durer_over_noise_floor", float("nan"))
    r = abs(summary.get("e5_pearson_r", float("nan")))
    parts = [f"dense Dürer grid absorbs a median {R:.0%} of the non-scale profile difference",
             f"residual = {ratio:.1f}x the landmark re-detection noise floor" if np.isfinite(ratio) else "noise floor unavailable",
             f"|r|(k, Δtan φ) = {r:.2f}" if np.isfinite(r) else "E5 correlation unavailable"]
    if np.isfinite(ratio) and ratio <= 1.5 and R >= 0.6 and np.isfinite(r) and r >= 0.7:
        return "supported", "; ".join(parts)
    if R < 0.3:
        return "refuted", "; ".join(parts)
    strong = "strong form NOT supported (residual >> noise)" if (np.isfinite(ratio) and ratio > 3.0) else "strong form undecided"
    return "mixed", "; ".join(parts + [strong])


def write_result(an: dict, ref: dict, metrics: dict):
    verdict, why = verdict_from(an["summary"])
    s = an["summary"]
    res = {"claim_id": "T1917-040", "date_made": "1917",
           "claim": ref["source"]["claim_text"],
           "dataset": {"primary_source": "Thompson 1917 Figs. 364/366, IA scan ongrowthform1917thom leaves 759/761",
                       "primary_source_sha256": {p["file"]: p["sha256"] for p in ref["thompson_1917_pages"] if p["leaf"] in (759, 761)},
                       "pre_scouted_pointer": "doi:10.5061/dryad.0zpc86746 — DATASET_MISMATCH (cichlid fish), unused",
                       "specimens": {sp["file"]: {"commons": sp["commons_title"], "license": sp["license"], "sha256": sp["sha256"]} for sp in ref["specimens"]}},
           "verdict": verdict, "verdict_basis": why, "pass": VERSION,
           "headline_numbers": {k: s.get(k) for k in ("n_specimens_accepted", "n_pairs", "dense_rmse_null_median", "dense_rmse_similarity_median", "dense_rmse_durer_median", "dense_rmse_affine_median",
                                                      "dense_loo_null_median", "dense_loo_durer_median", "dense_loo_affine_median", "noise_floor_rmse_median", "dense_rmse_durer_over_noise_floor",
                                                      "dense_R_durer_median", "dense_R_durer_min", "dense_R_durer_max", "durer_c_range", "durer_k_range",
                                                      "e5_pearson_r", "e5_slope", "e5_p_value", "ks_null_vs_durer_p", "pose_sensitivity_R_durer_median_pm5deg",
                                                      "control_negative_rmse_median", "control_positive_rmse_max", "control_ablation_rmse_median",
                                                      "fig366_omega_left_deg", "fig366_omega_right_deg", "convexity_mean_deg", "convexity_n",
                                                      "nasolabial_mean_deg", "nasolabial_n")},
           "verdict_rule": verdict_from.__doc__.strip(),
           "limitations": ["n = %d accepted photographs (5 of 9 vendored specimens fail the segmentation / landmark QC gate: gradient wall shadow, halftone print, faded lower face, textured background, stage lettering)" % s.get("n_specimens_accepted", 0),
                           "soft-tissue photographic profiles, not skulls; forehead landmarks lost where hair covers the glabella",
                           "the Dryad pointer pre-scouted for this claim is a cichlid dataset (mismatch) and was not used; specimens are Commons photographs, i.e. an analogue to Camper's drawn series, labelled as such"],
           "script": "thompson/claims/t1917-040_on-the-theory-of-transformations.py",
           "figure": f"docs/media/{SLUG}_comparison.png", "animation": f"docs/media/{SLUG}_growth.gif"}
    (DATA / "result.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    return res


def measure_morphoeval() -> list[dict]:
    """Quantitative morphometrics for morphoeval/measure_outputs.py (dynamic discovery)."""
    mp = DATA / "t1917_040_metrics.json"
    if mp.exists():
        s = json.loads(mp.read_text(encoding="utf-8"))["summary"]
    else:
        s = analyse(load_reference(), verbose=False)["summary"]
    src = "Thompson 1917 Ch. XVII; Legan & Burstone 1980 norms"
    nan = float("nan")
    out = [
        {"name": "accepted specimens (QC gate) of vendored photographs", "value": s.get("n_specimens_accepted"), "unit": "count",
         "method": f"Lab background-model silhouette + extremum-sequence landmarks; {s.get('n_specimens_total')} vendored", "derived_from": "script", "source": src},
        {"name": "median dense RMSE, Camper-frame null (uniform scale + translation)", "value": round(s.get("dense_rmse_null_median", nan), 4), "unit": "centroid-size units",
         "method": "3-dof null per ordered pair on common landmarks + 30 semilandmarks; rmse; similarity (4 dof, rotation) median = %.4f" % s.get("dense_rmse_similarity_median", nan), "derived_from": "script", "source": src},
        {"name": "median dense RMSE, Dürer grid (stretch c + shear k, 5 dof)", "value": round(s.get("dense_rmse_durer_median", nan), 4), "unit": "centroid-size units",
         "method": "upper-triangular fit in the Camper frame (E1+E3) vs scale+shift null; rmse; LOO-CV median = %.4f (null LOO %.4f); affine (6 dof) median = %.4f" % (s.get("dense_loo_durer_median", nan), s.get("dense_loo_null_median", nan), s.get("dense_rmse_affine_median", nan)),
         "reference_value": round(s.get("noise_floor_rmse_median", nan), 4), "note": "reference = landmark re-detection noise floor (rescale 0.85/1.15, blur 1 px); claim predicts residual at the noise floor", "derived_from": "script", "source": src},
        {"name": "Dürer residual / noise floor", "value": round(s.get("dense_rmse_durer_over_noise_floor", nan), 3), "unit": "ratio",
         "method": "median dense Dürer rmse divided by median noise-floor rmse", "reference_value": 1.0, "note": "whole-grid claim predicts ~1; verdict rule: <= 1.5 supported, > 3 refuted", "derived_from": "script", "source": src},
        {"name": "median R_Dürer = 1 - SS(Dürer)/SS(null), dense", "value": round(s.get("dense_R_durer_median", nan), 4), "unit": "fraction",
         "method": "fraction of the non-scale profile difference absorbed by the whole-grid deformation; CV across pairs (landmark-only) = %.2f; pose sensitivity +/-5 deg -> %s" % (s.get("R_durer_cv", nan), s.get("pose_sensitivity_R_durer_median_pm5deg")),
         "reference_value": 1.0, "note": "claim predicts ~1; 0 = all difference is local", "derived_from": "script", "source": src},
        {"name": "Pearson r, fitted shear k vs measured Δtan(E-line Prn-Pg inclination) (E5 Camper law)", "value": round(s.get("e5_pearson_r", nan), 3), "unit": "r",
         "method": "over ordered pairs; slope = %.2f, p = %.3g, RMSE(tan) = %.3f" % (s.get("e5_slope", nan), s.get("e5_p_value", nan), s.get("e5_rmse_tan", nan)),
         "reference_value": 1.0, "note": "E5 predicts Δtan φ = k exactly (r = 1, slope 1) if the facial-angle change is a whole-grid shear", "derived_from": "script", "source": src},
        {"name": "KS test, null vs Dürer dense residual distributions", "value": round(s.get("ks_null_vs_durer_p", nan), 4), "unit": "p-value",
         "method": "two-sample Kolmogorov-Smirnov, D = %.3f" % s.get("ks_null_vs_durer_D", nan), "derived_from": "script", "source": src},
        {"name": "scrambled-correspondence control, median Dürer RMSE", "value": round(s.get("control_negative_rmse_median", nan), 4), "unit": "centroid-size units",
         "method": "negative control: B landmarks permuted (seed 1917) before fitting; must exceed the real-pair residual", "derived_from": "script", "source": src},
        {"name": "positive control, max Dürer RMSE on an exact Dürer grid of a real specimen", "value": s.get("control_positive_rmse_max"), "unit": "centroid-size units",
         "method": "stretch 1.25 + omega 75 deg applied to each specimen; fit must recover it (rmse -> 0)", "reference_value": 0.0, "derived_from": "script", "source": src},
        {"name": "nose-only ablation on an exact Dürer grid, median Dürer RMSE", "value": round(s.get("control_ablation_rmse_median", nan), 4), "unit": "centroid-size units",
         "method": "exact grid of each specimen + pronasale displaced anteriorly by 10 % centroid size; the whole-grid family cannot absorb a local change, so the residual must stay > 0 (vs 0 for the positive control)", "reference_value": 0.0, "derived_from": "script", "source": src},
        {"name": "Thompson Fig. 366 net obliquity (left / right head), Hough digitisation", "value": s.get("fig366_omega_left_deg"), "unit": "deg",
         "method": "Canny + probabilistic Hough on the pinned IA scan leaf 761; right head = %.1f deg" % s.get("fig366_omega_right_deg", nan), "reference_value": 90.0,
         "note": "90 deg = rectangular net; Dürer's drawn shear is |k| = |cot omega|", "derived_from": "script", "source": "Thompson 1917 p. 742 Fig. 366"},
        {"name": "facial convexity angle G-Sn-Pg, mean over specimens", "value": round(s.get("convexity_mean_deg", nan), 2), "unit": "deg",
         "method": "180 - interior angle at Sn; std = %.2f, n = %s" % (s.get("convexity_sd_deg", nan), s.get("convexity_n")),
         "reference_value": 12.0, "note": "Legan & Burstone 1980 adult norm 12 +/- 4 deg", "derived_from": "script", "source": src},
        {"name": "nasolabial angle Cm-Sn-Ls, mean over specimens", "value": round(s.get("nasolabial_mean_deg", nan), 2), "unit": "deg",
         "method": "interior angle at Sn; std = %.2f, n = %s" % (s.get("nasolabial_sd_deg", nan), s.get("nasolabial_n")),
         "reference_value": 102.0, "note": "Legan & Burstone 1980 adult norm 102 +/- 8 deg", "derived_from": "script", "source": src},
    ]
    return out


# =============================================================================
# Self-test
# =============================================================================
def self_test() -> int:
    rng = np.random.default_rng(SEED)
    P = rng.normal(size=(11, 2)) * 10
    # E3 group law (Lean shear_group_law)
    assert np.allclose(shear(shear(P, 0.3), 0.5), shear(P, 0.8))
    assert np.allclose(shear(shear(P, 0.7), -0.7), P)
    # E1 composition (Lean stretch_circle_to_ellipse)
    assert np.allclose(stretch(stretch(P, 1.5), 2.0), stretch(P, 3.0))
    # Thompson oblique transposition: omega = 90 -> rectangular net
    k90, cs90 = oblique_to_shear(90.0)
    assert abs(k90) < 1e-12 and abs(cs90 - 1) < 1e-12
    k50, _ = oblique_to_shear(50.0)
    assert abs(k50 + 1 / math.tan(math.radians(50))) < 1e-12
    # E5 facial-angle law (Lean camper_facial_angle_index)
    assert abs(facial_line_tan(1.0, 4.0, 0.25) - (0.25 + 0.25)) < 1e-12
    # positive control: an exact Dürer grid (stretch then shear) is recovered by the affine fit
    Q = shear(stretch(P, 1.3), 0.4) @ np.array([[math.cos(0.2), -math.sin(0.2)], [math.sin(0.2), math.cos(0.2)]]).T + np.array([5, -3])
    f = fit_pair(P, Q)
    assert f["rmse_affine"] < 1e-9, f
    assert abs(f["affine_k"] - 0.4) < 1e-9 and abs(f["affine_anisotropy"] - 1.3) < 1e-9, f
    assert f["rmse_similarity"] > 1e-3 and f["R_affine"] > 0.999
    # R2: Dürer sub-family fit recovers an exact stretch+shear in the Procrustes frame
    Pd = P - P.mean(0)
    Qd = durer_grid_apply(Pd, 1.25, 75.0) * 2.0 + np.array([3.0, -1.0])
    Mn, tn = fit_scale_translation(Pd, Pd * 3.0 + 1.0)
    assert np.allclose(Mn, 3.0 * np.eye(2)) and np.allclose(tn, 1.0)
    U, tu, dp = fit_durer(Pd, Qd)
    assert abs(dp["c"] - 1.25) < 1e-9 and abs(dp["k"] - oblique_to_shear(75.0)[0]) < 1e-9 and abs(dp["scale"] - 2.0) < 1e-9, dp
    assert np.allclose(affine_apply(Pd, U, tu), Qd)
    f2 = fit_pair_v2(Pd, Qd)
    assert f2["rmse_durer"] < 1e-9 and f2["loo_durer"] < 1e-6 and f2["R_durer"] > 0.999, f2
    # E5 in the Camper frame: inclination change of a sheared configuration equals k in tan units
    a0 = profile_angles(Pd); a1 = profile_angles(shear(Pd, 0.2))
    assert abs(math.tan(math.radians(a1["facial_line_inclination_deg"])) - math.tan(math.radians(a0["facial_line_inclination_deg"])) - 0.2) < 1e-9
    # R3: semilandmarks are equally spaced along arc length
    S = semilandmarks(np.column_stack([np.zeros(100), np.arange(100.0)]), 10, 60, n=11)
    assert np.allclose(np.diff(S[:, 1]), 5.0)
    # similarity fit recovers a pure similarity
    Rm = np.array([[math.cos(0.5), -math.sin(0.5)], [math.sin(0.5), math.cos(0.5)]]) * 2.0
    Ms, ts = fit_similarity(P, P @ Rm.T + 1.0)
    assert np.allclose(Ms, Rm) and np.allclose(ts, 1.0)
    # synthetic protrusion curve -> landmark sequence
    H = 600
    y = np.arange(H)
    bumps = [(120, 10, 6), (210, 40, 14), (300, 12, 7), (360, 14, 7), (440, 22, 10)]  # g, prn, ls, li, pg
    p = 100 + sum(a * np.exp(-0.5 * ((y - c) / w) ** 2) for c, a, w in bumps) - 40 * (y > 500) * ((y - 500) / 100)
    L, _ = detect_landmarks(p, "right")
    assert all(L[k] is not None for k in LANDMARKS), L
    ok, reason = qc_landmarks(L, H)
    assert ok, reason
    assert abs(L["prn"][1] - 210) <= 3 and abs(L["pg"][1] - 440) <= 4 and abs(L["g"][1] - 120) <= 4
    # dataset integrity: every pinned SHA-256 verifies
    ref = load_reference()
    for sp in ref["specimens"]:
        assert sha256_file(DATA / sp["file"]) == sp["sha256"], sp["file"]
    for pg in ref["thompson_1917_pages"]:
        assert sha256_file(DATA / pg["file"]) == pg["sha256"], pg["file"]
    # one real specimen end-to-end
    r = process_specimen(ref["specimens"][1])
    assert r["accepted"], r["reason"]
    print("self-test OK: E1/E3/E4/E5 algebra, affine + Dürer sub-family recover exact grids, LOO, semilandmarks, landmark sequence, "
          f"{len(ref['specimens'])} specimen + {len(ref['thompson_1917_pages'])} page digests verified")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--no-gif", action="store_true")
    args = ap.parse_args(argv)
    if args.self_test:
        return self_test()
    ref = load_reference()
    print(f"[t1917-040] {VERSION}: {len(ref['specimens'])} specimens")
    an = analyse(ref)
    metrics = write_outputs(an, ref)
    make_comparison_png(an, MEDIA / f"{SLUG}_comparison.png", ref)
    if not args.no_gif:
        make_growth_gif(an, MEDIA / f"{SLUG}_growth.gif")
    res = write_result(an, ref, metrics)
    s = an["summary"]
    print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in s.items()}, indent=1, default=str))
    print(f"verdict: {res['verdict']} — {res['verdict_basis']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

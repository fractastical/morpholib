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

Pass 1 (this version) is the baseline: segmentation, landmarks, pairwise similarity vs
affine fits, convexity / nasolabial angles vs literature norms, comparison figure and
Dürer-shear animation.  Controls, parameter sweeps, noise floor, LOO-CV and the
statistical goodness-of-fit battery are Pass 2 remediation items.

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
VERSION = "1.0 (Pass 1 baseline)"

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
    top = "g" if has("g") else ("n" if has("n") else None)
    if top is not None and has("ls", "sn", "me"):
        axis = P[i["me"]] - P[i["sn"]]
        fl = P[i["ls"]] - P[i[top]]
        inc = math.degrees(math.atan2(fl[0] * axis[1] - fl[1] * axis[0], fl @ axis))
    else:
        inc = float("nan")
    return {"convexity_deg": conv, "nasolabial_deg": nl, "facial_line_inclination_deg": inc, "facial_line_top": top}


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


def analyse(ref: dict, verbose: bool = True) -> dict:
    results = [process_specimen(s) for s in ref["specimens"]]
    acc = [r for r in results if r["accepted"]]
    if verbose:
        for r in results:
            print(f"  {r['spec']['file']:55s} {'ACCEPT' if r['accepted'] else 'reject'}  {r['reason']}")
    configs = [landmark_array(r["landmarks"], r["spec"]["facing"]) for r in acc]
    names = [Path(r["spec"]["file"]).stem for r in acc]
    angles = [profile_angles(c) for c in configs]
    aligned, mean = generalized_procrustes(configs) if len(configs) >= 2 else (configs, configs[0] if configs else None)
    pairs = []
    for i in range(len(configs)):
        for j in range(len(configs)):
            if i == j:
                continue
            f = fit_pair(configs[i], configs[j])
            f.update({"from": names[i], "to": names[j],
                      "d_convexity_deg": angles[j]["convexity_deg"] - angles[i]["convexity_deg"],
                      "d_inclination_deg": angles[j]["facial_line_inclination_deg"] - angles[i]["facial_line_inclination_deg"]})
            pairs.append(f)
    summary = {}
    if pairs:
        rs = np.array([p["rmse_similarity"] for p in pairs]); ra = np.array([p["rmse_affine"] for p in pairs])
        R = np.array([p["R_affine"] for p in pairs])
        summary = {"n_specimens_accepted": len(acc), "n_specimens_total": len(results), "n_pairs": len(pairs),
                   "rmse_similarity_median": float(np.median(rs)), "rmse_affine_median": float(np.median(ra)),
                   "R_affine_median": float(np.median(R)), "R_affine_min": float(R.min()), "R_affine_max": float(R.max()),
                   "convexity_mean_deg": float(np.nanmean([a["convexity_deg"] for a in angles])),
                   "convexity_sd_deg": float(np.nanstd([a["convexity_deg"] for a in angles], ddof=1)) if sum(np.isfinite(a["convexity_deg"]) for a in angles) > 1 else float("nan"),
                   "convexity_n": int(sum(np.isfinite(a["convexity_deg"]) for a in angles)),
                   "nasolabial_mean_deg": float(np.nanmean([a["nasolabial_deg"] for a in angles])),
                   "nasolabial_sd_deg": float(np.nanstd([a["nasolabial_deg"] for a in angles], ddof=1)) if len(angles) > 1 else float("nan"),
                   "nasolabial_n": int(sum(np.isfinite(a["nasolabial_deg"]) for a in angles))}
    return {"results": results, "accepted": acc, "configs": configs, "names": names, "angles": angles,
            "aligned": aligned, "mean": mean, "pairs": pairs, "summary": summary}


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
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 4)
    # (a) Thompson Fig. 366 crop
    ax = fig.add_subplot(gs[0, 0])
    page = next(p for p in ref["thompson_1917_pages"] if p["leaf"] == 761)
    pim = Image.open(DATA / page["file"])
    W, H = pim.size
    ax.imshow(pim.crop((int(0.22 * W), int(0.17 * H), int(0.82 * W), int(0.33 * H))))
    ax.set_title("Thompson 1917 Fig. 366 (after Dürer), p. 742 —\noblique nets = E3 shear", fontsize=8)
    ax.set_axis_off()
    # (b) specimen with landmarks
    acc = an["accepted"]
    if acc:
        _draw_profile_panel(fig.add_subplot(gs[0, 1]), acc[0], f"specimen: {acc[0]['spec']['commons_title'][5:40]}\nsegmentation + 11 landmarks")
    if len(acc) > 1:
        _draw_profile_panel(fig.add_subplot(gs[0, 2]), acc[1], f"specimen: {acc[1]['spec']['commons_title'][5:40]}")
    # (c) Procrustes superimposition
    ax = fig.add_subplot(gs[0, 3])
    for a, nm in zip(an["aligned"], an["names"]):
        f = np.isfinite(a).all(1)
        ax.plot(a[f, 0], a[f, 1], "-o", ms=2, lw=0.8, alpha=0.7, label=nm[:18])
    if an["mean"] is not None:
        f = np.isfinite(an["mean"]).all(1)
        ax.plot(an["mean"][f, 0], an["mean"][f, 1], "k-", lw=2, label="mean (common landmarks)")
    ax.set_aspect("equal"); ax.set_title(f"similarity-aligned profiles (n={len(acc)})", fontsize=8)
    ax.legend(fontsize=5, loc="lower left")
    # (d) Dürer grid: A warped onto B by the fitted affine
    ax = fig.add_subplot(gs[1, 0:2])
    if len(an["configs"]) >= 2:
        A, B = an["configs"][0], an["configs"][1]
        keep = common_rows(A, B); A, B = A[keep], B[keep]
        Ms, ts = fit_similarity(A, B); Ma, ta = fit_affine(A, B)
        As = affine_apply(A, Ms, ts); Aa = affine_apply(A, Ma, ta)
        bx = (As[:, 0].min() - 0.1 * centroid_size(B), As[:, 1].min() - 0.1 * centroid_size(B),
              As[:, 0].max() + 0.1 * centroid_size(B), As[:, 1].max() + 0.1 * centroid_size(B))
        Ainv_s = np.linalg.inv(Ms)
        for ln in durer_grid(bx, 6):
            # grid drawn in A's similarity frame, then carried by the residual affine Ma Ms^-1
            g = affine_apply(affine_apply(ln - ts, Ainv_s.T, np.zeros(2)), Ma, ta)
            ax.plot(g[:, 0], g[:, 1], color="0.7", lw=0.6)
        ax.plot(As[:, 0], As[:, 1], "b.-", lw=1, label=f"A ({an['names'][0][:14]}) similarity-fitted")
        ax.plot(Aa[:, 0], Aa[:, 1], "r.-", lw=1, label="A under fitted Dürer grid (affine)")
        ax.plot(B[:, 0], B[:, 1], "k.-", lw=1.5, label=f"B ({an['names'][1][:14]})")
        f = an["pairs"][0]
        ax.set_title(f"whole-grid fit A->B: RMSE sim {f['rmse_similarity']:.3f} -> affine {f['rmse_affine']:.3f} "
                     f"(R_affine={f['R_affine']:.2f}, k={f['affine_k']:+.2f}, sy/sx={f['affine_anisotropy']:.2f})", fontsize=8)
        ax.legend(fontsize=6); ax.set_aspect("equal")
    # (e) residuals per pair
    ax = fig.add_subplot(gs[1, 2])
    if an["pairs"]:
        rs = [p["rmse_similarity"] for p in an["pairs"]]; ra = [p["rmse_affine"] for p in an["pairs"]]
        ax.boxplot([rs, ra], tick_labels=["similarity\n(4 dof)", "affine / Dürer grid\n(6 dof)"])
        ax.set_ylabel("landmark RMSE (centroid-size units)")
        ax.set_title(f"{len(an['pairs'])} ordered pairs; median R_affine = {an['summary']['R_affine_median']:.2f}", fontsize=8)
    # (f) angles vs norms
    ax = fig.add_subplot(gs[1, 3])
    if an["angles"]:
        conv = [a["convexity_deg"] for a in an["angles"]]; nl = [a["nasolabial_deg"] for a in an["angles"]]
        ax.scatter(conv, nl, c="k", s=18, zorder=3)
        ax.axvspan(8, 16, color="tab:blue", alpha=0.15, label="Legan-Burstone convexity 12±4°")
        ax.axhspan(94, 110, color="tab:orange", alpha=0.15, label="Legan-Burstone nasolabial 102±8°")
        for a, nm in zip(an["angles"], an["names"]):
            if np.isfinite(a["convexity_deg"]) and np.isfinite(a["nasolabial_deg"]):
                ax.annotate(nm[:10], (a["convexity_deg"], a["nasolabial_deg"]), fontsize=5)
        ax.set_xlabel("facial convexity G-Sn-Pg (deg)"); ax.set_ylabel("nasolabial Cm-Sn-Ls (deg)")
        ax.legend(fontsize=6); ax.set_title("profile angles vs literature norms", fontsize=8)
    fig.suptitle("T1917-040 — Thompson (1917) Dürer/Camper facial grids: are inter-individual profile differences whole-grid deformations?  "
                 f"[{VERSION}]", fontsize=10)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110)
    plt.close(fig)


def make_growth_gif(an: dict, out: Path, n_frames: int = 36):
    """Dürer shear sweep (E3) applied to the first accepted specimen's dense profile curve and
    an 8x8 net, omega 60 -> 120 deg and back, with the facial-line inclination read-out (E5)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if not an["accepted"]:
        return
    r = an["accepted"][0]
    box = r["box"]; facing = r["spec"]["facing"]
    ys = np.arange(len(r["curve"]))
    ok = ~np.isnan(r["curve"])
    curve = np.column_stack([r["curve"][ok], ys[ok]])
    L = {k: np.array(r["landmarks"][k]) - np.array([box[0], box[1]]) for k in LANDMARKS if r["landmarks"][k] is not None}
    top = "g" if "g" in L else "n"
    face_h = L["me"][1] - L[top][1]
    y_top, y_bot = L[top][1] - 0.35 * face_h, L["me"][1] + 0.1 * face_h
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
               "angles": {nm: a for nm, a in zip(an["names"], an["angles"])},
               "specimens": [{"file": r["spec"]["file"], "accepted": r["accepted"], "reason": r["reason"]} for r in an["results"]]}
    (DATA / "t1917_040_metrics.json").write_text(json.dumps(metrics, indent=1), encoding="utf-8")
    return metrics


def verdict_from(summary: dict) -> tuple[str, str]:
    if not summary:
        return "untestable", "no accepted specimen pairs"
    R = summary["R_affine_median"]
    if R >= 0.6:
        return "supported", f"whole-grid affine absorbs a median {R:.0%} of the non-similarity shape difference"
    if R >= 0.3:
        return "mixed", f"whole-grid affine absorbs only a median {R:.0%} of the non-similarity shape difference"
    return "refuted", f"whole-grid affine absorbs a median {R:.0%} of the non-similarity shape difference"


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
           "headline_numbers": {k: s.get(k) for k in ("n_specimens_accepted", "n_pairs", "rmse_similarity_median", "rmse_affine_median",
                                                      "R_affine_median", "convexity_mean_deg", "nasolabial_mean_deg")},
           "script": "thompson/claims/t1917-040_on-the-theory-of-transformations.py",
           "figure": f"docs/media/{SLUG}_comparison.png", "animation": f"docs/media/{SLUG}_growth.gif"}
    (DATA / "result.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    return res


def measure_morphoeval() -> list[dict]:
    """Quantitative morphometrics for morphoeval/measure_outputs.py (dynamic discovery)."""
    mp = DATA / "t1917_040_metrics.json"
    if mp.exists():
        m = json.loads(mp.read_text(encoding="utf-8"))
        s = m["summary"]
    else:
        s = analyse(load_reference(), verbose=False)["summary"]
    src = "Thompson 1917 Ch. XVII; Legan & Burstone 1980 norms"
    out = [
        {"name": "accepted specimens with 11 profile landmarks", "value": s.get("n_specimens_accepted"), "unit": "count",
         "method": "Lab background-model silhouette + extremum-sequence landmarks, QC gate", "derived_from": "script", "source": src},
        {"name": "median landmark RMSE, similarity null", "value": round(s["rmse_similarity_median"], 4) if s else None, "unit": "centroid-size units",
         "method": "Procrustes similarity fit per ordered pair (4 dof), rmse", "derived_from": "script", "source": src},
        {"name": "median landmark RMSE, whole-grid affine (Dürer grid)", "value": round(s["rmse_affine_median"], 4) if s else None, "unit": "centroid-size units",
         "method": "least-squares affine fit per ordered pair (6 dof) vs similarity null, rmse", "derived_from": "script", "source": src},
        {"name": "median R_affine = 1 - SS(affine)/SS(similarity)", "value": round(s["R_affine_median"], 4) if s else None, "unit": "fraction",
         "method": "fraction of non-similarity shape difference absorbed by the whole-grid deformation (claim statistic)",
         "reference_value": 1.0, "note": "claim predicts ~1 (whole-grid); 0 = all difference is local", "derived_from": "script", "source": src},
        {"name": "facial convexity angle G-Sn-Pg, mean over specimens", "value": round(s["convexity_mean_deg"], 2) if s else None, "unit": "deg",
         "method": "180 - interior angle at Sn; std over specimens = %.2f" % s.get("convexity_sd_deg", float("nan")) if s else "",
         "reference_value": 12.0, "note": "Legan & Burstone 1980 adult norm 12 +/- 4 deg", "derived_from": "script", "source": src},
        {"name": "nasolabial angle Cm-Sn-Ls, mean over specimens", "value": round(s["nasolabial_mean_deg"], 2) if s else None, "unit": "deg",
         "method": "interior angle at Sn; std over specimens = %.2f" % s.get("nasolabial_sd_deg", float("nan")) if s else "",
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
    print("self-test OK: E1/E3/E4/E5 algebra, affine recovers exact Dürer grid, landmark sequence, "
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
    print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in s.items()}, indent=1))
    print(f"verdict: {res['verdict']} — {res['verdict_basis']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

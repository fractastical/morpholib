#!/usr/bin/env python3
"""
T1917-054 — Thompson (1917) Ch. XVII, pp. 757-759, Figs. 388-393:
the pelves of Archaeopteryx and Apatornis (after Heilmann) and the three
"intermediate and equidistant co-ordinate systems ... interpolated between them".

Claim as registered (thompson/thompson_1917_claims.json, T1917-054)
  "The pelves of Archaeopteryx and Apatornis (after Heilmann) and intermediate
  configurations can be found by interpolation between coordinate-transformed forms -
  i.e. interpolated grids generate biologically real intermediate forms."

What Thompson prints (p. 759)
  "In Fig. 390 the outlines of these two co-ordinate systems are superposed upon one
  another, and those of three intermediate and equidistant co-ordinate systems are
  interpolated between them.  From each of these latter systems, so determined by direct
  interpolation, a complete co-ordinate diagram is drawn, and the corresponding outline
  of a pelvis is found from each of these systems of co-ordinates, as in Figs. 391, 392.
  Finally, in Fig. 393 the complete series is represented, beginning with the known
  pelvis of Archaeopteryx, and leading up by our three intermediate hypothetical types
  to the known pelvis of Apatornis."

So the claim has two parts.  (A, methodological) three equidistant interpolations of the
Archaeopteryx -> Apatornis grid yield the intermediate outlines of Figs. 391-393; this is
testable from Thompson's own figures and is what this script tests.  (B, biological
reality) those intermediates match real intermediate Mesozoic bird pelves; Thompson calls
them "hypothetical types", no landmark data of intermediate taxa is pinned here, and the
pre-scouted Dryad dataset (doi:10.5061/dryad.k1v34cf, Deeming & Mayr 2018) is a variable
mismatch (1-D pelvic-canal width), so part B is reported as untestable, not fitted.

Equations (datasets/thompson_1917/t1917-054/t1917_054_on_the_theory_of_transformatio_reference.json)
  E1  phi : R^2 -> R^2, phi(grid_A) = grid_B, det(D phi) > 0           -> tps_fit / check_no_fold
  E2  phi_t(x) = (1 - t) x + t phi(x), t in {0, 1/4, 1/2, 3/4, 1}       -> interp_stage
  E3  Omega_t = phi_t(Omega_A)                                         -> predicted outlines
  E4  thin-plate spline realisation of phi (Bookstein 1989)            -> tps_fit / tps_apply
  E5  affine special case, M_t = (1 - t) I + t A                       -> affine_fit / affine_interp_matrix
  E6  phi_t(a_i) = (1 - t) a_i + t b_i for corresponding points        -> interp_stage on configurations
  E7  t*_k = <D_k - A, B - A> / |B - A|^2 ; residual |t*_k - k/4|      -> best_fit_t
  Lean 4 mirror: morphoeval/lean/loops/Thompson1917_T1917_054.lean (MathOfBiology.Loop_243).

Data (all SHA-256 pinned in the reference JSON and checked here)
  Internet Archive item ongrowthform1917thom, pages 757-760 (leaves n776-n779), public
  domain.  Fig. 393 (p. 760) carries the five wash silhouettes - Archaeopteryx, three
  interpolated stages, Apatornis - and is the quantitative surface: the silhouettes are
  segmented (Otsu threshold, opening, connected components), the acetabulum (interior
  hole) and the four process tips (anterior/posterior ilium, ischium, pubis) are found
  automatically, and arc-length semilandmarks are placed between the tips.
  Wikimedia Commons: Archaeopteryx lithographica, Berlin specimen (H. Raab, CC BY-SA 3.0,
  downscaled derivative) and the Apatornis celer holotype synsacrum (Marsh 1880, PD) are
  vendored as the real-specimen anchors; neither yields an articulated pelvis outline, so
  they document the taxa and do not enter the verdict.

What is computed
  1. Segmentation of the five Fig. 393 silhouettes; landmark configuration per form
     (acetabulum + 4 x n_semi semilandmarks).
  2. Generalised Procrustes superimposition of the five configurations.
  3. The Thompson-Heilmann construction: phi fitted from Archaeopteryx (stage 0) to
     Apatornis (stage 4) as a thin-plate spline on the full configuration, phi_t for
     t = 1/4, 1/2, 3/4, applied to the Archaeopteryx outline and to a 9 x 8 grid.
  4. Comparison of each predicted stage with Heilmann's drawn stage: full Procrustes
     distance, versus the distance from the drawn stage to either endpoint (the trivial
     "copy an endpoint" prediction); best-fit stage t*_k and equidistance residual.
  5. Shape descriptors along the series (pubis and ischium angles to the ilium axis,
     pre-/post-acetabular ilium ratio, pubis/ilium length, centroid size).
  6. Figures: docs/media/t1917_054_on_the_theory_of_transformatio_comparison.png and
     ..._growth.gif; verdict: datasets/thompson_1917/t1917-054/result.json.

Claim boundary
  reported_in_source: the five drawn forms and the statement that the three middle ones
    are equidistant interpolations of the two grids.
  reconstructed_in_script: the grid as a thin-plate spline on automatically placed
    landmarks/semilandmarks (Thompson's grid in Fig. 389 is not digitised line by line),
    the stage parameter t, and all statistics.
  does_not_claim: anything about real intermediate taxa (part B); any particular
    phylogenetic position of Apatornis; that Heilmann's wash drawings are to one scale.

Usage
  python3 thompson/claims/t1917-054_on-the-theory-of-transformations.py --self-test
  python3 thompson/claims/t1917-054_on-the-theory-of-transformations.py [--n-semi 24] [--lam 0.0]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from skimage import measure

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
DATA = ROOT / "datasets" / "thompson_1917" / "t1917-054"
MEDIA = ROOT / "docs" / "media"
SLUG = "t1917_054_on_the_theory_of_transformatio"
REF_JSON = DATA / f"{SLUG}_reference.json"
RESULT_JSON = DATA / "result.json"
COMPARISON_PNG = MEDIA / f"{SLUG}_comparison.png"
GROWTH_GIF = MEDIA / f"{SLUG}_growth.gif"
SCRIPT_REL = "thompson/claims/t1917-054_on-the-theory-of-transformations.py"

SEED = 1917
STAGES = (0.0, 0.25, 0.5, 0.75, 1.0)
STAGE_NAMES = ("Archaeopteryx (Fig. 388)", "stage 1 (Fig. 391)", "stage 2 (Fig. 392 top)",
               "stage 3 (Fig. 392 bottom)", "Apatornis (Fig. 389)")
FIG393_PAGE = "ongrowthform1917thom_p760_leaf0779_fig393.jpg"
FIG388_PAGE = "ongrowthform1917thom_p757_leaf0776_figs388-389.jpg"
FIG393_CROP = (0.25, 0.17, 0.80, 0.70)          # fraction of page (x0, y0, x1, y1)
FIG388_CROP = (0.22, 0.25, 0.72, 0.50)
FIG389_CROP = (0.12, 0.52, 0.80, 0.72)
GRID_LINES = (9, 8)                               # Fig. 388: vertical 1..9, horizontal a..h
MIN_COMPONENT_PX = 50_000                         # at 1904 x 3128 page resolution

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
# 1. segmentation of Fig. 393 and landmarking
# --------------------------------------------------------------------------- #
def otsu_threshold(a: np.ndarray) -> int:
    hist, _ = np.histogram(a, bins=256, range=(0, 256))
    p = hist / max(hist.sum(), 1)
    w = np.cumsum(p)
    m = np.cumsum(p * np.arange(256))
    sb = (m[-1] * w - m) ** 2 / np.maximum(w * (1 - w), 1e-12)
    return int(np.argmax(sb))


def crop_page(page: Path, frac: tuple[float, float, float, float]) -> np.ndarray:
    im = Image.open(page).convert("L")
    W, H = im.size
    box = (int(frac[0] * W), int(frac[1] * H), int(frac[2] * W), int(frac[3] * H))
    return np.asarray(im.crop(box)).astype(float)


def segment_fig393(a: np.ndarray) -> list[dict]:
    """Five largest dark components (top -> bottom) = Archaeopteryx, stages 1-3, Apatornis.
    Each form: filled mask (halftone speckle closed), acetabulum = largest interior hole."""
    thr = otsu_threshold(a)
    mask = ndi.binary_opening(a < thr, iterations=1)
    lab, n = ndi.label(mask)
    sizes = ndi.sum(mask, lab, range(1, n + 1))
    big = [i + 1 for i in np.argsort(sizes)[::-1][:5] if sizes[i] >= MIN_COMPONENT_PX]
    if len(big) != 5:
        raise RuntimeError(f"expected 5 silhouettes, found {len(big)} components >= {MIN_COMPONENT_PX} px")
    objs = ndi.find_objects(lab)
    big.sort(key=lambda li: objs[li - 1][0].start)   # top-to-bottom
    forms = []
    for li in big:
        m = lab == li
        filled = ndi.binary_fill_holes(m)
        holes = filled & ~m
        hl, hn = ndi.label(holes)
        hs = ndi.sum(holes, hl, range(1, hn + 1))
        hi = int(np.argmax(hs)) + 1
        acet_rc = np.array(ndi.center_of_mass(hl == hi))
        smooth = ndi.binary_opening(ndi.binary_closing(filled, iterations=2), iterations=2)
        contours = measure.find_contours(smooth.astype(float), 0.5)
        c = max(contours, key=len)
        forms.append({"mask": smooth, "raw_area_px": int(m.sum()), "area_px": int(smooth.sum()),
                      "acetabulum_rc": acet_rc, "acetabulum_area_px": int(hs[hi - 1]),
                      "contour_rc": c, "otsu": thr})
    return forms


def radial_peaks(contour_rc: np.ndarray, center_rc: np.ndarray, sep_frac: float = 0.08) -> tuple[list[int], np.ndarray]:
    """Indices of local maxima of distance-from-acetabulum along the closed contour,
    strongest first (non-maximum suppression over +/- sep_frac of the perimeter)."""
    d = np.hypot(contour_rc[:, 0] - center_rc[0], contour_rc[:, 1] - center_rc[1])
    n = len(d)
    win = max(3, int(n * sep_frac))
    idx = np.arange(n)
    is_peak = np.ones(n, bool)
    for off in range(1, win + 1):
        is_peak &= d >= d[(idx + off) % n]
        is_peak &= d >= d[(idx - off) % n]
    peaks = list(np.flatnonzero(is_peak))
    merged: list[list[int]] = []
    for p in peaks:
        if merged and p - merged[-1][-1] <= 2:
            merged[-1].append(p)
        else:
            merged.append([p])
    peaks = [int(np.mean(g)) for g in merged]
    peaks.sort(key=lambda i: -d[i])
    return peaks, d


def classify_tips(contour_rc: np.ndarray, acet_rc: np.ndarray) -> dict[str, int]:
    """The four strongest radial peaks are the process tips.  Pubis = lowest point on the
    page (largest row), posterior ilium = right-most, anterior ilium = the dorsal one of the
    remaining two (smallest row), ischium = the other.  Holds for all five panels of
    Fig. 393 because Heilmann drew them in one orientation."""
    peaks, _ = radial_peaks(contour_rc, acet_rc)
    if len(peaks) < 4:
        raise RuntimeError("fewer than 4 process tips found")
    P = peaks[:4]
    pub = max(P, key=lambda i: contour_rc[i, 0])
    rest = [i for i in P if i != pub]
    post = max(rest, key=lambda i: contour_rc[i, 1])
    rest = [i for i in rest if i != post]
    ant = min(rest, key=lambda i: contour_rc[i, 0])
    isc = [i for i in rest if i != ant][0]
    return {"post_ilium": post, "ischium": isc, "pubis": pub, "ant_ilium": ant}


def resample_segment(c: np.ndarray, i0: int, i1: int, n: int) -> np.ndarray:
    N = len(c)
    idx = np.arange(i0, i0 + ((i1 - i0) % N) + 1) % N
    seg = c[idx]
    dl = np.hypot(*np.diff(seg, axis=0).T)
    s = np.r_[0.0, np.cumsum(dl)]
    ss = np.linspace(0.0, s[-1], n + 1)[:-1]
    return np.c_[np.interp(ss, s, seg[:, 0]), np.interp(ss, s, seg[:, 1])]


def rc_to_xy(rc: np.ndarray) -> np.ndarray:
    """image (row, col) -> Cartesian (x right, y up)."""
    return np.c_[rc[:, 1], -rc[:, 0]]


def landmark_configuration(form: dict, n_semi: int) -> dict:
    """Row 0 = acetabulum; then 4 segments x n_semi arc-length semilandmarks, starting at the
    posterior ilium tip and running post_ilium -> ischium -> pubis -> ant_ilium -> back."""
    c = form["contour_rc"]
    tips = classify_tips(c, form["acetabulum_rc"])
    order = [tips["post_ilium"], tips["ischium"], tips["pubis"], tips["ant_ilium"]]
    N = len(c)
    fwd = sum(((order[(j + 1) % 4] - order[j]) % N) for j in range(4))
    if fwd != N:                       # contour runs the other way: reverse it
        c = c[::-1]
        order = [N - 1 - i for i in order]
    pts = np.vstack([resample_segment(c, order[j], order[(j + 1) % 4], n_semi) for j in range(4)])
    X = np.vstack([rc_to_xy(form["acetabulum_rc"][None])[0][None], rc_to_xy(pts)])
    tip_rows = {"post_ilium": 1, "ischium": 1 + n_semi, "pubis": 1 + 2 * n_semi, "ant_ilium": 1 + 3 * n_semi}
    return {"X": X, "tip_rows": tip_rows, "contour_xy": rc_to_xy(c), "n_semi": n_semi}


# --------------------------------------------------------------------------- #
# 2. Procrustes
# --------------------------------------------------------------------------- #
def centroid_size(X: np.ndarray) -> float:
    return float(np.sqrt(((X - X.mean(0)) ** 2).sum()))


def procrustes_align(X: np.ndarray, Y: np.ndarray) -> tuple[np.ndarray, float, dict]:
    """Full Procrustes fit of X onto Y (translation, rotation, scale; no reflection).
    Returns X aligned in Y's frame, the full Procrustes distance between the unit-size
    shapes, and the similarity parameters."""
    Xc = X - X.mean(0)
    Yc = Y - Y.mean(0)
    zx = Xc[:, 0] + 1j * Xc[:, 1]
    zy = Yc[:, 0] + 1j * Yc[:, 1]
    a = np.vdot(zx, zy)                     # sum conj(zx) * zy
    beta = a / np.vdot(zx, zx)              # complex scale*rotation, least squares
    Xa = zx * beta
    sx, sy = np.sqrt(np.vdot(zx, zx).real), np.sqrt(np.vdot(zy, zy).real)
    rho = abs(a) / (sx * sy)
    d_full = float(np.sqrt(max(0.0, 1.0 - rho ** 2)))
    out = np.c_[Xa.real, Xa.imag] + Y.mean(0)
    return out, d_full, {"scale": float(abs(beta)), "rotation_deg": float(np.degrees(np.angle(beta))),
                         "translation": (Y.mean(0) - X.mean(0)).tolist()}


def gpa(configs: list[np.ndarray], iters: int = 10) -> tuple[list[np.ndarray], np.ndarray]:
    """Generalised Procrustes analysis (unit centroid size, consensus iteration)."""
    Z = [(X - X.mean(0)) / centroid_size(X) for X in configs]
    mean = Z[0].copy()
    for _ in range(iters):
        Z = [procrustes_align(X, mean)[0] for X in Z]
        new = np.mean(Z, axis=0)
        new = (new - new.mean(0)) / centroid_size(new)
        if np.allclose(new, mean, atol=1e-12):
            break
        mean = new
    return Z, mean


# --------------------------------------------------------------------------- #
# 3. the transformation: TPS (E4), affine (E5), interpolation (E2, E6)
# --------------------------------------------------------------------------- #
def _U(r2: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(r2 > 0, r2 * np.log(r2), 0.0)


def tps_fit(P: np.ndarray, Q: np.ndarray, lam: float = 0.0) -> dict:
    """Bookstein (1989) thin-plate spline phi with phi(P) = Q (lam = 0) or smoothed (lam > 0)."""
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


def tps_bending_energy(m: dict) -> float:
    r2 = ((m["P"][:, None, :] - m["P"][None, :, :]) ** 2).sum(-1)
    K = _U(r2)
    return float(np.trace(m["W"].T @ K @ m["W"]))


def affine_fit(P: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """Least-squares affine map Q ~ [1 P] M, M is 3 x 2 (E5)."""
    Pm = np.c_[np.ones(len(P)), P]
    M, *_ = np.linalg.lstsq(Pm, Q, rcond=None)
    return M


def affine_apply(M: np.ndarray, X: np.ndarray) -> np.ndarray:
    return np.c_[np.ones(len(X)), X] @ M


def affine_interp_matrix(M: np.ndarray, t: float) -> np.ndarray:
    """M_t = (1 - t) I + t A for the linear part, t b for the translation (E5)."""
    Mt = M.copy()
    Mt[1:, :] = (1 - t) * np.eye(2) + t * M[1:, :]
    Mt[0, :] = t * M[0, :]
    return Mt


def interp_stage(X: np.ndarray, phiX: np.ndarray, t: float) -> np.ndarray:
    """E2 / E6: phi_t(x) = (1 - t) x + t phi(x)."""
    return (1.0 - t) * X + t * phiX


def best_fit_t(D: np.ndarray, A: np.ndarray, B: np.ndarray) -> float:
    """E7: least-squares stage of D on the segment A -> B (all in one Procrustes frame)."""
    v = (B - A).ravel()
    return float(np.dot((D - A).ravel(), v) / np.dot(v, v))


def make_grid(X: np.ndarray, nv: int = GRID_LINES[0], nh: int = GRID_LINES[1], per_line: int = 60, pad: float = 0.06):
    """Thompson's reference grid (nv vertical, nh horizontal lines) over the form's bbox."""
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


# --------------------------------------------------------------------------- #
# 4. descriptors
# --------------------------------------------------------------------------- #
def _angle_deg(u: np.ndarray, v: np.ndarray) -> float:
    cu, cv = u / np.linalg.norm(u), v / np.linalg.norm(v)
    return float(np.degrees(np.arctan2(cu[0] * cv[1] - cu[1] * cv[0], np.dot(cu, cv))))


def descriptors(cfg: dict) -> dict:
    X, r = cfg["X"], cfg["tip_rows"]
    ac = X[0]
    ant, post, isc, pub = X[r["ant_ilium"]], X[r["post_ilium"]], X[r["ischium"]], X[r["pubis"]]
    axis = post - ant
    return {
        "pubis_angle_deg": abs(_angle_deg(axis, pub - ac)),
        "ischium_angle_deg": abs(_angle_deg(axis, isc - ac)),
        "pubis_over_ilium": float(np.linalg.norm(pub - ac) / np.linalg.norm(axis)),
        "ischium_over_ilium": float(np.linalg.norm(isc - ac) / np.linalg.norm(axis)),
        "preacetabular_over_postacetabular": float(np.linalg.norm(ant - ac) / np.linalg.norm(post - ac)),
        "centroid_size_px": centroid_size(X),
    }


# --------------------------------------------------------------------------- #
# 5. the analysis
# --------------------------------------------------------------------------- #
def run_analysis(n_semi: int = 24, lam: float = 0.0, verbose: bool = True) -> dict:
    t0 = time.time()
    ref = load_reference()
    digests = verify_inputs(ref)
    page = crop_page(DATA / FIG393_PAGE, FIG393_CROP)
    forms = segment_fig393(page)
    cfgs = [landmark_configuration(f, n_semi) for f in forms]
    raw = [c["X"] for c in cfgs]
    Z, consensus = gpa(raw)
    A, B = Z[0], Z[4]

    # E4/E1: the grid as a TPS on the Archaeopteryx -> Apatornis configurations (consensus frame)
    tps = tps_fit(A, B, lam=lam)
    phiA = tps_apply(tps, A)                     # = B when lam = 0 (interpolating spline)
    aff = affine_fit(A, B)
    affA = affine_apply(aff, A)

    stages = []
    for k, t in enumerate(STAGES):
        pred_tps = interp_stage(A, phiA, t)       # E2/E6
        pred_aff = interp_stage(A, affA, t)
        D = Z[k]
        _, d_pred, _ = procrustes_align(pred_tps, D)
        _, d_aff, _ = procrustes_align(pred_aff, D)
        _, d_A, _ = procrustes_align(A, D)
        _, d_B, _ = procrustes_align(B, D)
        tstar = best_fit_t(D, A, B)
        stages.append({
            "k": k, "t_nominal": t, "name": STAGE_NAMES[k],
            "procrustes_pred_tps_vs_drawn": d_pred,
            "procrustes_pred_affine_vs_drawn": d_aff,
            "procrustes_drawn_vs_archaeopteryx": d_A,
            "procrustes_drawn_vs_apatornis": d_B,
            "nearest_endpoint_distance": min(d_A, d_B),
            "t_star": tstar, "equidistance_residual": abs(tstar - t),
            "descriptors": descriptors(cfgs[k]),
            "area_px": forms[k]["area_px"], "raw_area_px": forms[k]["raw_area_px"],
            "acetabulum_area_px": forms[k]["acetabulum_area_px"],
        })
    _, d_AB, _ = procrustes_align(A, B)
    mids = stages[1:4]
    headline = {
        "procrustes_distance_archaeopteryx_apatornis": d_AB,
        "t_star": [s["t_star"] for s in mids],
        "t_nominal": [s["t_nominal"] for s in mids],
        "equidistance_residual_max": max(s["equidistance_residual"] for s in mids),
        "equidistance_residual_mean": float(np.mean([s["equidistance_residual"] for s in mids])),
        "t_star_monotone": bool(mids[0]["t_star"] < mids[1]["t_star"] < mids[2]["t_star"]),
        "t_star_strictly_inside_0_1": bool(all(0.0 < s["t_star"] < 1.0 for s in mids)),
        "pred_beats_nearest_endpoint_all": bool(all(s["procrustes_pred_tps_vs_drawn"] < s["nearest_endpoint_distance"] for s in mids)),
        "pred_tps_vs_drawn_mean": float(np.mean([s["procrustes_pred_tps_vs_drawn"] for s in mids])),
        "pred_affine_vs_drawn_mean": float(np.mean([s["procrustes_pred_affine_vs_drawn"] for s in mids])),
        "nearest_endpoint_mean": float(np.mean([s["nearest_endpoint_distance"] for s in mids])),
        "tps_bending_energy": tps_bending_energy(tps),
        "tps_endpoint_residual_rms": float(np.sqrt(((phiA - B) ** 2).sum(1).mean())),
        "affine_endpoint_residual_rms": float(np.sqrt(((affA - B) ** 2).sum(1).mean())),
        "centroid_size_px": [s["descriptors"]["centroid_size_px"] for s in stages],
        "pubis_angle_deg": [s["descriptors"]["pubis_angle_deg"] for s in stages],
    }
    res = {"n_semi": n_semi, "lam": lam, "seed": SEED, "otsu_threshold": forms[0]["otsu"],
           "digests": digests, "stages": stages, "headline": headline,
           "consensus": consensus, "Z": Z, "cfgs": cfgs, "forms": forms, "tps": tps, "phiA": phiA,
           "affine": aff, "affA": affA, "elapsed_s": time.time() - t0}
    if verbose:
        print(f"[t1917-054] Otsu={forms[0]['otsu']}  areas(px)={[f['area_px'] for f in forms]}  n_semi={n_semi} lam={lam}")
        print(f"[t1917-054] Procrustes d(A,B) = {d_AB:.4f}")
        for s in mids:
            print(f"  stage {s['k']}: t*={s['t_star']:.3f} (nominal {s['t_nominal']}), "
                  f"d(pred_tps, drawn)={s['procrustes_pred_tps_vs_drawn']:.4f}, d(pred_aff, drawn)={s['procrustes_pred_affine_vs_drawn']:.4f}, "
                  f"d(drawn, A)={s['procrustes_drawn_vs_archaeopteryx']:.4f}, d(drawn, B)={s['procrustes_drawn_vs_apatornis']:.4f}")
        print(f"[t1917-054] centroid sizes px: {np.round(headline['centroid_size_px'], 1).tolist()}")
        print(f"[t1917-054] pubis angle deg: {np.round(headline['pubis_angle_deg'], 1).tolist()}")
    return res


# --------------------------------------------------------------------------- #
# 6. figures
# --------------------------------------------------------------------------- #
def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _closed(X: np.ndarray) -> np.ndarray:
    return np.vstack([X, X[:1]])


def render_comparison(res: dict, out: Path = COMPARISON_PNG) -> Path:
    plt = _plt()
    Z, cfgs, forms = res["Z"], res["cfgs"], res["forms"]
    A, B, phiA = Z[0], Z[4], res["phiA"]
    tps = res["tps"]
    fig = plt.figure(figsize=(18, 11), dpi=110)
    gs = fig.add_gridspec(2, 4, height_ratios=[1.0, 1.0])

    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(crop_page(DATA / FIG388_PAGE, FIG388_CROP), cmap="gray", vmin=0, vmax=255)
    ax.set_title("Thompson 1917 Fig. 388 — Archaeopteryx\n(Cartesian grid, p. 757)", fontsize=10)
    ax.axis("off")
    ax = fig.add_subplot(gs[1, 0])
    ax.imshow(crop_page(DATA / FIG388_PAGE, FIG389_CROP), cmap="gray", vmin=0, vmax=255)
    ax.set_title("Fig. 389 — Apatornis (transformed grid)", fontsize=10)
    ax.axis("off")

    ax = fig.add_subplot(gs[:, 1])
    page = crop_page(DATA / FIG393_PAGE, FIG393_CROP)
    ax.imshow(page, cmap="gray", vmin=0, vmax=255)
    cols = ["#1f77b4", "#2ca02c", "#9467bd", "#ff7f0e", "#d62728"]
    for k, (f, c) in enumerate(zip(forms, cfgs)):
        cont = f["contour_rc"]
        ax.plot(cont[:, 1], cont[:, 0], color=cols[k], lw=1.0)
        X = c["X"]
        ax.plot(X[1:, 0], -X[1:, 1], ".", color=cols[k], ms=2.5)
        ax.plot(X[0, 0], -X[0, 1], "o", mfc="none", mec="k", ms=8)
        for nm, r in c["tip_rows"].items():
            ax.plot(X[r, 0], -X[r, 1], "s", color="k", ms=4)
        ax.text(X[:, 0].max() + 15, -X[:, 1].mean(), f"t={STAGES[k]:.2f}", color=cols[k], fontsize=9, va="center")
    ax.set_title("Fig. 393 (p. 760) segmented: outlines, acetabulum (o),\n4 process tips (■), arc-length semilandmarks (·)", fontsize=10)
    ax.axis("off")

    # predicted stages with interpolated grid vs drawn stage
    for j, k in enumerate((1, 2, 3)):
        ax = fig.add_subplot(gs[0, 2] if j == 0 else (gs[0, 3] if j == 1 else gs[1, 2]))
        t = STAGES[k]
        pred = interp_stage(A, phiA, t)
        D = Z[k]
        Dal, d, _ = procrustes_align(D, pred)
        for line in make_grid(A):
            gl = interp_stage(line, tps_apply(tps, line), t)
            ax.plot(gl[:, 0], gl[:, 1], color="0.75", lw=0.6)
        ax.plot(*_closed(pred[1:]).T, color=cols[k], lw=1.8, label=f"interpolated grid, t={t}")
        ax.plot(*_closed(Dal[1:]).T, color="k", lw=1.2, ls="--", label="Heilmann's drawn stage")
        ax.plot(pred[0, 0], pred[0, 1], "o", color=cols[k], mfc="none")
        ax.set_aspect("equal")
        ax.set_title(f"stage {k}: t*={res['stages'][k]['t_star']:.3f} (nominal {t}); "
                     f"d_P={d:.3f} vs nearest endpoint {res['stages'][k]['nearest_endpoint_distance']:.3f}", fontsize=9)
        ax.legend(fontsize=7, loc="lower left")
        ax.axis("off")

    ax = fig.add_subplot(gs[1, 3])
    ks = [1, 2, 3]
    ax.plot([0, 1], [0, 1], color="0.6", lw=1, label="t* = t (perfect equidistance)")
    ax.plot([STAGES[k] for k in ks], [res["stages"][k]["t_star"] for k in ks], "o-", color="k", label="best-fit stage t* (E7)")
    ax2 = ax.twinx()
    w = 0.04
    ax2.bar([STAGES[k] - w for k in ks], [res["stages"][k]["procrustes_pred_tps_vs_drawn"] for k in ks], width=w, color="#1f77b4", label="d(interpolated, drawn)")
    ax2.bar([STAGES[k] for k in ks], [res["stages"][k]["procrustes_pred_affine_vs_drawn"] for k in ks], width=w, color="#aec7e8", label="d(affine interp., drawn)")
    ax2.bar([STAGES[k] + w for k in ks], [res["stages"][k]["nearest_endpoint_distance"] for k in ks], width=w, color="0.5", label="d(nearest endpoint, drawn)")
    ax.set_xlabel("nominal stage t")
    ax.set_ylabel("t*")
    ax2.set_ylabel("full Procrustes distance")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=7, loc="upper left")
    hl = res["headline"]
    ax.set_title(f"equidistance residual max {hl['equidistance_residual_max']:.3f}; "
                 f"d(A,B)={hl['procrustes_distance_archaeopteryx_apatornis']:.3f}", fontsize=9)

    fig.suptitle("T1917-054 · Thompson (1917) Ch. XVII Figs. 388-393: Archaeopteryx → Apatornis pelvis by equidistant grid interpolation "
                 "(part A, methodological) — part B 'biologically real' untestable with pinned data", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    return out


def render_growth_gif(res: dict, out: Path = GROWTH_GIF, n_frames: int = 33) -> Path:
    plt = _plt()
    Z, phiA, tps = res["Z"], res["phiA"], res["tps"]
    A = Z[0]
    frames = []
    lo = np.minimum(A.min(0), Z[4].min(0)) - 0.15
    hi = np.maximum(A.max(0), Z[4].max(0)) + 0.15
    for t in np.linspace(0, 1, n_frames):
        fig, ax = plt.subplots(figsize=(6.4, 4.2), dpi=90)
        for line in make_grid(A):
            gl = interp_stage(line, tps_apply(tps, line), t)
            ax.plot(gl[:, 0], gl[:, 1], color="0.78", lw=0.6)
        pred = interp_stage(A, phiA, t)
        ax.fill(*_closed(pred[1:]).T, color="#555555", alpha=0.85)
        ax.plot(pred[0, 0], pred[0, 1], "o", color="w", ms=5)
        k = int(round(t * 4))
        if abs(t - STAGES[k]) < 0.5 / (n_frames - 1):
            Dal, _, _ = procrustes_align(Z[k], pred)
            ax.plot(*_closed(Dal[1:]).T, color="#d62728", lw=1.2, ls="--")
            ax.text(lo[0] + 0.02, hi[1] - 0.05, f"drawn {STAGE_NAMES[k]} (red dashed)", fontsize=8, color="#d62728")
        ax.set_xlim(lo[0], hi[0])
        ax.set_ylim(lo[1], hi[1])
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(f"T1917-054 · φ_t = (1−t)·id + t·φ,  t = {t:.2f}   (Archaeopteryx → Apatornis)", fontsize=9)
        fig.canvas.draw()
        frames.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()))
        plt.close(fig)
    out.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(out, save_all=True, append_images=frames[1:], duration=120, loop=0)
    return out


# --------------------------------------------------------------------------- #
# 7. verdict and MorphoEval measurements
# --------------------------------------------------------------------------- #
def write_result(res: dict, ref: dict, out: Path = RESULT_JSON) -> Path:
    hl = res["headline"]
    part_a = hl["t_star_monotone"] and hl["t_star_strictly_inside_0_1"] and hl["pred_beats_nearest_endpoint_all"]
    verdict = "mixed"   # part A testable and (if part_a) supported; part B untestable with pinned data
    doc = {
        "claim_id": "T1917-054",
        "date_made": "1917",
        "claim": ref["claim_text"],
        "dataset": {
            "name": "Thompson (1917) On Growth and Form, Figs. 388-393 (Heilmann 1915), Internet Archive scan ongrowthform1917thom pp. 757-760",
            "DOI": None,
            "url": "https://archive.org/details/ongrowthform1917thom",
            "sha256": res["digests"],
            "pre_scouted_dryad": {
                "identifier": "doi:10.5061/dryad.k1v34cf",
                "status": "DATASET_MISMATCH",
                "reason": "1-D pelvic-canal width / body mass for 21 Mesozoic birds, not 2-D pelvis shape; file not script-retrievable (OAuth token / browser challenge). Labelled analogue; NOT used.",
            },
        },
        "verdict": verdict,
        "verdict_by_part": {
            "A_methodological_interpolation_yields_equidistant_intermediates": "supported" if part_a else "refuted",
            "B_intermediates_are_biologically_real": "untestable",
        },
        "headline_numbers": {
            "procrustes_distance_archaeopteryx_apatornis": round(hl["procrustes_distance_archaeopteryx_apatornis"], 4),
            "t_star_stages_1_3": [round(v, 4) for v in hl["t_star"]],
            "t_nominal": hl["t_nominal"],
            "equidistance_residual_max": round(hl["equidistance_residual_max"], 4),
            "procrustes_interpolated_vs_drawn_mean": round(hl["pred_tps_vs_drawn_mean"], 4),
            "procrustes_affine_interpolated_vs_drawn_mean": round(hl["pred_affine_vs_drawn_mean"], 4),
            "procrustes_nearest_endpoint_vs_drawn_mean": round(hl["nearest_endpoint_mean"], 4),
            "centroid_size_px": [round(v, 1) for v in hl["centroid_size_px"]],
            "pubis_angle_deg": [round(v, 1) for v in hl["pubis_angle_deg"]],
            "n_semilandmarks_per_segment": res["n_semi"],
        },
        "script": SCRIPT_REL,
        "figure": str(COMPARISON_PNG.relative_to(ROOT)),
        "animation": str(GROWTH_GIF.relative_to(ROOT)),
        "lean": "morphoeval/lean/loops/Thompson1917_T1917_054.lean",
        "notes": [
            "Thompson's own words are 'three intermediate hypothetical types'; the registered claim's 'biologically real' is stronger than the source.",
            "Part A is scored against Heilmann's drawn intermediates (Fig. 393), i.e. it tests that the construction is well-defined, equidistant and reproducible, not that it predicts fossils.",
        ],
    }
    out.write_text(json.dumps(doc, indent=2), encoding="utf-8")
    return out


def measure_morphoeval() -> list[dict]:
    """MorphoEval dynamic measurements (consumed by morphoeval/measure_outputs.py)."""
    res = run_analysis(verbose=False)
    hl = res["headline"]
    st = res["stages"]

    def M(name, value, unit, method, reference=None, source=None, note=None, derived_from="script"):
        m = {"name": name, "value": float(round(value, 4)) if isinstance(value, (float, np.floating)) else value,
             "unit": unit, "method": method, "derived_from": derived_from}
        if reference is not None:
            m["reference_value"] = reference
        if source:
            m["reference_source"] = source
        if note:
            m["note"] = note
        return m

    out = [
        M("best-fit stage t* of Heilmann's first intermediate (Fig. 391)", st[1]["t_star"], "t (0 = Archaeopteryx, 1 = Apatornis)",
          "E7 least-squares projection of the GPA-aligned drawn stage onto the Archaeopteryx->Apatornis segment", reference=0.25,
          source="Thompson 1917 p. 759: 'three intermediate and equidistant co-ordinate systems'", derived_from="script"),
        M("best-fit stage t* of the second intermediate (Fig. 392 top)", st[2]["t_star"], "t", "E7 projection", reference=0.5,
          source="Thompson 1917 p. 759", derived_from="script"),
        M("best-fit stage t* of the third intermediate (Fig. 392 bottom)", st[3]["t_star"], "t", "E7 projection", reference=0.75,
          source="Thompson 1917 p. 759", derived_from="script"),
        M("max equidistance residual |t* - k/4| over the three stages", hl["equidistance_residual_max"], "t units",
          "max over k of |t*_k - k/4|", reference=0.0, source="exact equidistance (Lean: interp_endpoints_and_equidistant)",
          note=f"mean residual {hl['equidistance_residual_mean']:.4f}; d(A,B) = {hl['procrustes_distance_archaeopteryx_apatornis']:.4f} so one stage = {hl['procrustes_distance_archaeopteryx_apatornis']/4:.4f} Procrustes units"),
        M("mean full Procrustes distance, interpolated grid prediction vs drawn intermediate", hl["pred_tps_vs_drawn_mean"], "Procrustes distance",
          "E2/E4: TPS phi_t applied to the Archaeopteryx configuration, full Procrustes fit onto the drawn stage, mean over stages 1-3",
          reference=round(hl["nearest_endpoint_mean"], 4), source="control: copy the nearest endpoint (same metric)",
          note=f"affine-only interpolation control: {hl['pred_affine_vs_drawn_mean']:.4f}; ratio interpolated/endpoint = {hl['pred_tps_vs_drawn_mean']/hl['nearest_endpoint_mean']:.3f}"),
        M("Archaeopteryx -> Apatornis full Procrustes distance (Fig. 393 endpoints)", hl["procrustes_distance_archaeopteryx_apatornis"], "Procrustes distance",
          f"GPA of acetabulum + 4 x {res['n_semi']} semilandmarks, unit centroid size", derived_from="output"),
        M("pubis angle to ilium axis along the series (A, 1, 2, 3, B)", [round(v, 1) for v in hl["pubis_angle_deg"]], "deg",
          "angle between acetabulum->pubis tip and anterior->posterior ilium tip vectors, per segmented form", derived_from="output"),
        M("centroid size of the five drawn forms", [round(v, 1) for v in hl["centroid_size_px"]], "px (page scan)",
          "sqrt of summed squared deviations from centroid of the landmark configuration", derived_from="output",
          note="not monotone: Heilmann's Fig. 393 series is not drawn to the interpolated grid's scale"),
    ]
    return out


# --------------------------------------------------------------------------- #
# 8. self-test
# --------------------------------------------------------------------------- #
def self_test() -> int:
    t0 = time.time()
    ref = load_reference()
    verify_inputs(ref)
    print("ok  SHA-256 of all pinned inputs")
    # E2 / E6 on random points: endpoints and equidistance
    rng = np.random.default_rng(SEED)
    X, Y = rng.normal(size=(7, 2)), rng.normal(size=(7, 2))
    assert np.allclose(interp_stage(X, Y, 0.0), X) and np.allclose(interp_stage(X, Y, 1.0), Y)
    steps = [interp_stage(X, Y, (k + 1) / 4) - interp_stage(X, Y, k / 4) for k in range(4)]
    assert all(np.allclose(s, (Y - X) / 4) for s in steps)
    print("ok  E2/E6 endpoints and equidistant steps")
    # E4: interpolating TPS reproduces the targets; E5: affine interpolation endpoints
    m = tps_fit(X, Y)
    assert np.allclose(tps_apply(m, X), Y, atol=1e-8)
    Maff = affine_fit(X, Y)
    assert np.allclose(affine_interp_matrix(Maff, 0.0)[1:], np.eye(2)) and np.allclose(affine_interp_matrix(Maff, 1.0), Maff)
    print("ok  E4 TPS interpolates its landmarks; E5 affine interpolation endpoints")
    # E7 soundness (Lean best_fit_stage_sound): exact interpolant recovers k/4
    for k in range(5):
        assert abs(best_fit_t(interp_stage(X, Y, k / 4), X, Y) - k / 4) < 1e-12
    print("ok  E7 best-fit stage recovers nominal stage for an exact interpolant")
    # Procrustes: invariance to similarity
    c, th = 2.3, 0.7
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    _, d0, _ = procrustes_align(X, Y)
    _, d1, _ = procrustes_align(c * X @ R.T + 5.0, Y)
    assert abs(d0 - d1) < 1e-10
    print("ok  Procrustes distance invariant to similarity transforms")
    # real data
    res = run_analysis(verbose=False)
    areas = [f["raw_area_px"] for f in res["forms"]]
    want = ref["digitisation"]["expected_component_area_px"]
    for a, w in zip(areas, [want["archaeopteryx"], want["stage_1"], want["stage_2"], want["stage_3"], want["apatornis"]]):
        assert abs(a - w) / w < 0.05, (a, w)
    print(f"ok  five silhouettes segmented, areas within 5% of the pinned reference: {areas}")
    for c in res["cfgs"]:
        assert c["X"].shape == (1 + 4 * res["n_semi"], 2)
    print("ok  landmark configurations (acetabulum + 4 segments) for all five forms")
    hl = res["headline"]
    assert hl["t_star_monotone"] and hl["t_star_strictly_inside_0_1"], hl["t_star"]
    print(f"ok  t* monotone and inside (0,1): {np.round(hl['t_star'], 3).tolist()}")
    assert hl["pred_beats_nearest_endpoint_all"]
    print("ok  interpolated prediction closer to each drawn stage than the nearest endpoint")
    res2 = run_analysis(verbose=False)
    assert res2["headline"]["t_star"] == hl["t_star"]
    print("ok  deterministic (two runs identical)")
    print(f"self-test passed in {time.time() - t0:.1f} s")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--n-semi", type=int, default=24, help="semilandmarks per boundary segment (default 24)")
    ap.add_argument("--lam", type=float, default=0.0, help="TPS regularisation (0 = interpolating)")
    ap.add_argument("--no-gif", action="store_true")
    args = ap.parse_args(argv)
    if args.self_test:
        return self_test()
    ref = load_reference()
    res = run_analysis(n_semi=args.n_semi, lam=args.lam)
    print("->", render_comparison(res).relative_to(ROOT))
    if not args.no_gif:
        print("->", render_growth_gif(res).relative_to(ROOT))
    print("->", write_result(res, ref).relative_to(ROOT))
    return 0


if __name__ == "__main__":
    sys.exit(main())

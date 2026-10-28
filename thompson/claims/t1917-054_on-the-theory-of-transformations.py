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
     distance, Pearson r and RMSE of corresponding coordinates, and two correspondence-free
     statistics (IoU of the filled dense outlines, symmetric boundary RMSE) - each against
     three controls: the nearest endpoint ("copy an endpoint"), an affine-only interpolation
     (Thompson's simple strains, E5) and a 9-anchor coarse grid (acetabulum, 4 tips, 4
     midpoints: a Thompson-like grid that is fold-free); best-fit stage t*_k and the
     equidistance residual |t*_k - k/4|.
  5. Nulls and stability (Pass 2): all 3! pairings of predicted and drawn stages; the
     correspondence-shift null (every cyclic shift of the Apatornis boundary semilandmarks,
     exact p); KS test of per-landmark residuals against the endpoint control; a 6 x 4 sweep of
     semilandmark density and TPS regularisation; the E1 no-fold check (min det D phi_t on
     the 9 x 8 grid and inside the form, dense vs coarse grid).
  6. Shape descriptors along the series (pubis and ischium angles to the ilium axis,
     pre-/post-acetabular ilium ratio, pubis/ilium length, centroid size); a labelled,
     non-verdict bone-vs-matrix colour segmentation of the Berlin Archaeopteryx photograph.
  7. Figures: docs/media/t1917_054_on_the_theory_of_transformatio_comparison.png and
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
# 4b. correspondence-free and statistical goodness of fit (Pass 2, R4)
# --------------------------------------------------------------------------- #
def similarity_params(X: np.ndarray, Y: np.ndarray) -> tuple[complex, np.ndarray, np.ndarray]:
    """Least-squares similarity carrying X onto Y: y ~ beta * (x - mean X) + mean Y (complex beta)."""
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


def rasterize(poly: np.ndarray, lo: np.ndarray, scale: float, size: int) -> np.ndarray:
    """Scan-line fill of a closed polygon (PIL) into a size x size boolean mask."""
    from PIL import ImageDraw
    pts = [(float((x - lo[0]) * scale), float((y - lo[1]) * scale)) for x, y in poly]
    im = Image.new("1", (size, size), 0)
    ImageDraw.Draw(im).polygon(pts, fill=1)
    return np.asarray(im, dtype=bool)


def outline_iou(P: np.ndarray, Q: np.ndarray, size: int = 400) -> float:
    """IoU of the filled outlines P and Q (already superimposed, same frame)."""
    lo = np.minimum(P.min(0), Q.min(0))
    hi = np.maximum(P.max(0), Q.max(0))
    scale = (size - 1) / max(hi - lo)
    mp, mq = rasterize(P, lo, scale, size), rasterize(Q, lo, scale, size)
    return float((mp & mq).sum() / max((mp | mq).sum(), 1))


def chamfer_rmse(P: np.ndarray, Q: np.ndarray) -> float:
    """Symmetric nearest-point boundary RMSE (correspondence-free) in the frame's units."""
    from scipy.spatial import cKDTree
    d1, _ = cKDTree(Q).query(P)
    d2, _ = cKDTree(P).query(Q)
    return float(np.sqrt(0.5 * (np.mean(d1 ** 2) + np.mean(d2 ** 2))))


def compare_outlines(pred_cfg: np.ndarray, pred_dense: np.ndarray, drawn_cfg: np.ndarray, drawn_dense: np.ndarray) -> dict:
    """Superimpose the prediction on the drawn stage (similarity from the semilandmark
    configurations), then score: full Procrustes distance (correspondence-based), Pearson r
    and RMSE of corresponding coordinates, IoU of the filled dense outlines and the symmetric
    boundary RMSE (both correspondence-free).  Distances are in units of the drawn stage's
    centroid size (GPA frame: 1.0)."""
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
        "per_landmark_residual": np.sqrt((resid ** 2).sum(1)) / cs,
        "iou": outline_iou(pred_dense_al, drawn_dense),
        "chamfer_rmse_cs": chamfer_rmse(pred_dense_al, drawn_dense) / cs,
    }


def jacobian_min_det(tps: dict, X: np.ndarray, t: float, h: float = 1e-4) -> float:
    """Numerical det(D phi_t) on the points X (E1 at stage t; Lean interp_no_fold is the 1-D form)."""
    def phi_t(P):
        return interp_stage(P, tps_apply(tps, P), t)
    ex, ey = np.array([h, 0.0]), np.array([0.0, h])
    dx = (phi_t(X + ex) - phi_t(X - ex)) / (2 * h)
    dy = (phi_t(X + ey) - phi_t(X - ey)) / (2 * h)
    det = dx[:, 0] * dy[:, 1] - dx[:, 1] * dy[:, 0]
    return float(det.min())


def check_no_fold(tps: dict, A: np.ndarray, dense_A: np.ndarray | None = None, n_t: int = 21) -> dict:
    """R5: minimum Jacobian determinant of phi_t for t in [0, 1], (a) over the padded 9 x 8
    reference grid of Fig. 388 and (b) over the interior of the Archaeopteryx form only
    (points of the grid that fall inside the dense outline)."""
    pts = np.vstack(make_grid(A, per_line=25))
    ts = np.linspace(0.0, 1.0, n_t)
    mins = [jacobian_min_det(tps, pts, t) for t in ts]
    out = {"t": ts.tolist(), "min_det_grid": mins, "min_over_stages_grid": float(min(mins)), "no_fold_grid": bool(min(mins) > 0)}
    if dense_A is not None:
        from matplotlib.path import Path as MplPath
        inside = pts[MplPath(dense_A).contains_points(pts)]
        mins_in = [jacobian_min_det(tps, inside, t) for t in ts] if len(inside) else [float("nan")]
        out.update({"n_inside": int(len(inside)), "min_det_inside": mins_in,
                    "min_over_stages_inside": float(min(mins_in)), "no_fold_inside": bool(min(mins_in) > 0)})
    return out


def sparse_anchor_rows(n_semi: int) -> list[int]:
    """Rows of the configuration that act as Thompson-like coarse-grid anchors: acetabulum (0), the
    four process tips (segment starts) and the four segment midpoints."""
    return [0] + [1 + j * n_semi for j in range(4)] + [1 + j * n_semi + n_semi // 2 for j in range(4)]


def pairing_permutation_test(pred_mid: list[np.ndarray], drawn_mid: list[np.ndarray]) -> dict:
    """R3a: all 3! pairings of predicted stages with drawn intermediates; the nominal pairing
    should minimise the total Procrustes distance.  Exact p = (#pairings <= observed) / 6."""
    from itertools import permutations
    totals = {}
    for perm in permutations(range(3)):
        totals[perm] = float(sum(procrustes_align(pred_mid[i], drawn_mid[perm[i]])[1] for i in range(3)))
    obs = totals[(0, 1, 2)]
    p = sum(1 for v in totals.values() if v <= obs + 1e-12) / len(totals)
    return {"observed_total": obs, "all_totals": {"".join(str(i + 1) for i in k): v for k, v in totals.items()},
            "p_value": p, "nominal_is_best": bool(obs <= min(totals.values()) + 1e-12)}


def correspondence_shift_null(A: np.ndarray, B: np.ndarray, drawn_mid: list[np.ndarray], n_semi: int) -> dict:
    """R3b: destroy the anatomical correspondence by cyclically shifting the Apatornis boundary
    semilandmarks (acetabulum kept), rebuild the interpolation, and score the prediction against
    the drawn stages.  All 4*n_semi - 1 non-trivial shifts form an exact null distribution for
    'the interpolation works because the correspondence is anatomical'."""
    nb = 4 * n_semi
    obs = float(np.mean([procrustes_align(interp_stage(A, B, STAGES[k]), drawn_mid[k - 1])[1] for k in (1, 2, 3)]))
    null = []
    for s in range(1, nb):
        Bs = np.vstack([B[:1], np.roll(B[1:], s, axis=0)])
        null.append(float(np.mean([procrustes_align(interp_stage(A, Bs, STAGES[k]), drawn_mid[k - 1])[1] for k in (1, 2, 3)])))
    null = np.array(null)
    return {"observed_mean_procrustes": obs, "null_mean": float(null.mean()), "null_min": float(null.min()),
            "null_sd": float(null.std()), "n_null": int(len(null)),
            "p_value": float((np.sum(null <= obs) + 1) / (len(null) + 1)),
            "z_score": float((obs - null.mean()) / max(null.std(), 1e-12))}


def ks_residuals(pred_res: np.ndarray, ctrl_res: np.ndarray) -> dict:
    from scipy.stats import ks_2samp
    ks = ks_2samp(pred_res, ctrl_res, alternative="greater")   # H1: prediction residuals stochastically smaller
    return {"D": float(ks.statistic), "p_value": float(ks.pvalue)}


def segment_specimen_bone(path: Path) -> dict:
    """R6 (labelled, non-verdict): bone vs matrix on the Berlin Archaeopteryx photograph.  Bone
    reads lighter and less saturated than the tan limestone; threshold the lightness-minus-
    saturation image by Otsu, open, and report the bone fraction and component count."""
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
# 5. the analysis
# --------------------------------------------------------------------------- #
def dense_in_frame(cfg: dict, Zk: np.ndarray) -> np.ndarray:
    """Dense contour of a form carried into the GPA frame by the similarity its configuration received."""
    beta, mx, my = similarity_params(cfg["X"], Zk)
    return apply_similarity(cfg["contour_xy"], beta, mx, my)


def run_analysis(n_semi: int = 24, lam: float = 0.0, verbose: bool = True, controls: bool = True) -> dict:
    t0 = time.time()
    ref = load_reference()
    digests = verify_inputs(ref)
    page = crop_page(DATA / FIG393_PAGE, FIG393_CROP)
    forms = segment_fig393(page)
    cfgs = [landmark_configuration(f, n_semi) for f in forms]
    raw = [c["X"] for c in cfgs]
    Z, consensus = gpa(raw)
    A, B = Z[0], Z[4]
    dense = [dense_in_frame(c, z) for c, z in zip(cfgs, Z)]

    # E4/E1: the grid as a TPS on the Archaeopteryx -> Apatornis configurations (consensus frame)
    tps = tps_fit(A, B, lam=lam)
    phiA = tps_apply(tps, A)                     # = B when lam = 0 (interpolating spline)
    phiA_dense = tps_apply(tps, dense[0])        # E3: the whole Archaeopteryx outline through the grid
    aff = affine_fit(A, B)
    affA = affine_apply(aff, A)
    affA_dense = affine_apply(aff, dense[0])
    # Thompson-like coarse grid: TPS on the 9 anatomical anchors only (acetabulum, 4 tips, 4 segment
    # midpoints); the rest of the outline follows the smooth grid (E3) instead of point-wise correspondence
    sparse_rows = sparse_anchor_rows(n_semi)
    tps_sparse = tps_fit(A[sparse_rows], B[sparse_rows], lam=lam)
    phiA_sparse = tps_apply(tps_sparse, A)
    phiA_sparse_dense = tps_apply(tps_sparse, dense[0])

    stages = []
    for k, t in enumerate(STAGES):
        pred_tps = interp_stage(A, phiA, t)       # E2/E6
        pred_aff = interp_stage(A, affA, t)
        pred_sparse = interp_stage(A, phiA_sparse, t)
        pred_dense = interp_stage(dense[0], phiA_dense, t)
        pred_aff_dense = interp_stage(dense[0], affA_dense, t)
        pred_sparse_dense = interp_stage(dense[0], phiA_sparse_dense, t)
        D, Dd = Z[k], dense[k]
        fit = compare_outlines(pred_tps, pred_dense, D, Dd)
        fit_aff = compare_outlines(pred_aff, pred_aff_dense, D, Dd)
        fit_sp = compare_outlines(pred_sparse, pred_sparse_dense, D, Dd)
        fit_A = compare_outlines(A, dense[0], D, Dd)
        fit_B = compare_outlines(B, dense[4], D, Dd)
        near = fit_A if fit_A["procrustes"] <= fit_B["procrustes"] else fit_B
        tstar = best_fit_t(D, A, B)
        st = {
            "k": k, "t_nominal": t, "name": STAGE_NAMES[k],
            "procrustes_pred_tps_vs_drawn": fit["procrustes"],
            "procrustes_pred_affine_vs_drawn": fit_aff["procrustes"],
            "procrustes_pred_sparse_vs_drawn": fit_sp["procrustes"],
            "procrustes_drawn_vs_archaeopteryx": fit_A["procrustes"],
            "procrustes_drawn_vs_apatornis": fit_B["procrustes"],
            "nearest_endpoint_distance": near["procrustes"],
            "iou_pred_vs_drawn": fit["iou"], "iou_affine_vs_drawn": fit_aff["iou"], "iou_sparse_vs_drawn": fit_sp["iou"], "iou_nearest_endpoint_vs_drawn": near["iou"],
            "chamfer_rmse_pred_cs": fit["chamfer_rmse_cs"], "chamfer_rmse_affine_cs": fit_aff["chamfer_rmse_cs"], "chamfer_rmse_sparse_cs": fit_sp["chamfer_rmse_cs"], "chamfer_rmse_nearest_endpoint_cs": near["chamfer_rmse_cs"],
            "pearson_r_pred": fit["pearson_r"], "pearson_r_nearest_endpoint": near["pearson_r"],
            "rmse_pred_cs": fit["rmse_cs"], "rmse_nearest_endpoint_cs": near["rmse_cs"],
            "t_star": tstar, "equidistance_residual": abs(tstar - t),
            "descriptors": descriptors(cfgs[k]),
            "area_px": forms[k]["area_px"], "raw_area_px": forms[k]["raw_area_px"],
            "acetabulum_area_px": forms[k]["acetabulum_area_px"],
        }
        if 1 <= k <= 3:
            st["ks_pred_vs_nearest_endpoint"] = ks_residuals(fit["per_landmark_residual"], near["per_landmark_residual"])
            st["ks_pred_vs_affine"] = ks_residuals(fit["per_landmark_residual"], fit_aff["per_landmark_residual"])
        stages.append(st)
    _, d_AB, _ = procrustes_align(A, B)
    mids = stages[1:4]

    def mean_of(key):
        return float(np.mean([s[key] for s in mids]))

    headline = {
        "procrustes_distance_archaeopteryx_apatornis": d_AB,
        "t_star": [s["t_star"] for s in mids],
        "t_nominal": [s["t_nominal"] for s in mids],
        "equidistance_residual_max": max(s["equidistance_residual"] for s in mids),
        "equidistance_residual_mean": float(np.mean([s["equidistance_residual"] for s in mids])),
        "t_star_monotone": bool(mids[0]["t_star"] < mids[1]["t_star"] < mids[2]["t_star"]),
        "t_star_strictly_inside_0_1": bool(all(0.0 < s["t_star"] < 1.0 for s in mids)),
        "pred_beats_nearest_endpoint_all": bool(all(s["procrustes_pred_tps_vs_drawn"] < s["nearest_endpoint_distance"] for s in mids)),
        "pred_beats_affine_all": bool(all(s["procrustes_pred_tps_vs_drawn"] < s["procrustes_pred_affine_vs_drawn"] for s in mids)),
        "pred_tps_vs_drawn_mean": mean_of("procrustes_pred_tps_vs_drawn"),
        "pred_affine_vs_drawn_mean": mean_of("procrustes_pred_affine_vs_drawn"),
        "pred_sparse_vs_drawn_mean": mean_of("procrustes_pred_sparse_vs_drawn"),
        "nearest_endpoint_mean": mean_of("nearest_endpoint_distance"),
        "iou_pred_mean": mean_of("iou_pred_vs_drawn"), "iou_affine_mean": mean_of("iou_affine_vs_drawn"), "iou_sparse_mean": mean_of("iou_sparse_vs_drawn"), "iou_nearest_endpoint_mean": mean_of("iou_nearest_endpoint_vs_drawn"),
        "chamfer_pred_mean_cs": mean_of("chamfer_rmse_pred_cs"), "chamfer_affine_mean_cs": mean_of("chamfer_rmse_affine_cs"), "chamfer_sparse_mean_cs": mean_of("chamfer_rmse_sparse_cs"), "chamfer_nearest_endpoint_mean_cs": mean_of("chamfer_rmse_nearest_endpoint_cs"),
        "sparse_endpoint_residual_rms": float(np.sqrt(((phiA_sparse - B) ** 2).sum(1).mean())),
        "pearson_r_pred_mean": mean_of("pearson_r_pred"), "pearson_r_nearest_endpoint_mean": mean_of("pearson_r_nearest_endpoint"),
        "rmse_pred_mean_cs": mean_of("rmse_pred_cs"), "rmse_nearest_endpoint_mean_cs": mean_of("rmse_nearest_endpoint_cs"),
        "ks_p_values_vs_nearest_endpoint": [s["ks_pred_vs_nearest_endpoint"]["p_value"] for s in mids],
        "tps_bending_energy": tps_bending_energy(tps),
        "tps_endpoint_residual_rms": float(np.sqrt(((phiA - B) ** 2).sum(1).mean())),
        "affine_endpoint_residual_rms": float(np.sqrt(((affA - B) ** 2).sum(1).mean())),
        "centroid_size_px": [s["descriptors"]["centroid_size_px"] for s in stages],
        "pubis_angle_deg": [s["descriptors"]["pubis_angle_deg"] for s in stages],
        "ischium_angle_deg": [s["descriptors"]["ischium_angle_deg"] for s in stages],
        "preacetabular_over_postacetabular": [s["descriptors"]["preacetabular_over_postacetabular"] for s in stages],
    }
    res = {"n_semi": n_semi, "lam": lam, "seed": SEED, "otsu_threshold": forms[0]["otsu"],
           "digests": digests, "stages": stages, "headline": headline,
           "consensus": consensus, "Z": Z, "dense": dense, "cfgs": cfgs, "forms": forms, "tps": tps, "phiA": phiA,
           "phiA_dense": phiA_dense, "tps_sparse": tps_sparse, "phiA_sparse": phiA_sparse, "phiA_sparse_dense": phiA_sparse_dense,
           "sparse_rows": sparse_rows, "affine": aff, "affA": affA, "elapsed_s": time.time() - t0}
    if controls:
        pred_mid = [interp_stage(A, phiA, STAGES[k]) for k in (1, 2, 3)]
        res["permutation"] = pairing_permutation_test(pred_mid, [Z[1], Z[2], Z[3]])
        res["correspondence_null"] = correspondence_shift_null(A, B, [Z[1], Z[2], Z[3]], n_semi)
        res["no_fold"] = check_no_fold(tps, A, dense[0])
        res["no_fold_sparse"] = check_no_fold(tps_sparse, A, dense[0])
        headline["permutation_p"] = res["permutation"]["p_value"]
        headline["correspondence_null_p"] = res["correspondence_null"]["p_value"]
        headline["correspondence_null_z"] = res["correspondence_null"]["z_score"]
        headline["min_jacobian_det_dense_grid"] = res["no_fold"]["min_over_stages_grid"]
        headline["min_jacobian_det_dense_inside"] = res["no_fold"]["min_over_stages_inside"]
        headline["min_jacobian_det_sparse_grid"] = res["no_fold_sparse"]["min_over_stages_grid"]
        headline["min_jacobian_det_sparse_inside"] = res["no_fold_sparse"]["min_over_stages_inside"]
    res["elapsed_s"] = time.time() - t0
    if verbose:
        print(f"[t1917-054] Otsu={forms[0]['otsu']}  areas(px)={[f['area_px'] for f in forms]}  n_semi={n_semi} lam={lam}")
        print(f"[t1917-054] Procrustes d(A,B) = {d_AB:.4f}")
        for s in mids:
            print(f"  stage {s['k']}: t*={s['t_star']:.3f} (nominal {s['t_nominal']}) | Procrustes dense {s['procrustes_pred_tps_vs_drawn']:.4f} "
                  f"sparse {s['procrustes_pred_sparse_vs_drawn']:.4f} affine {s['procrustes_pred_affine_vs_drawn']:.4f} nearest-endpoint {s['nearest_endpoint_distance']:.4f} | "
                  f"IoU dense {s['iou_pred_vs_drawn']:.3f} sparse {s['iou_sparse_vs_drawn']:.3f} affine {s['iou_affine_vs_drawn']:.3f} endpoint {s['iou_nearest_endpoint_vs_drawn']:.3f} | "
                  f"r={s['pearson_r_pred']:.4f} RMSE/cs={s['rmse_pred_cs']:.4f} KS p={s['ks_pred_vs_nearest_endpoint']['p_value']:.2e}")
        if controls:
            print(f"[t1917-054] pairing permutation: nominal best={res['permutation']['nominal_is_best']} p={res['permutation']['p_value']:.3f}; "
                  f"correspondence-shift null: obs={res['correspondence_null']['observed_mean_procrustes']:.4f} null mean={res['correspondence_null']['null_mean']:.4f} "
                  f"min={res['correspondence_null']['null_min']:.4f} p={res['correspondence_null']['p_value']:.4f} z={res['correspondence_null']['z_score']:.1f}")
            print(f"[t1917-054] no-fold min det(D phi_t): dense grid {res['no_fold']['min_over_stages_grid']:.3f} / inside form {res['no_fold']['min_over_stages_inside']:.3f}; "
                  f"sparse grid {res['no_fold_sparse']['min_over_stages_grid']:.3f} / inside form {res['no_fold_sparse']['min_over_stages_inside']:.3f}")
        print(f"[t1917-054] centroid sizes px: {np.round(headline['centroid_size_px'], 1).tolist()}")
        print(f"[t1917-054] pubis angle deg: {np.round(headline['pubis_angle_deg'], 1).tolist()}")
    return res


SWEEP_N_SEMI = (8, 12, 16, 24, 32, 48)
SWEEP_LAM = (0.0, 1e-3, 1e-2, 1e-1)


def parameter_sweep(n_semis=SWEEP_N_SEMI, lams=SWEEP_LAM, verbose: bool = True) -> dict:
    """R2: stability of t*, the prediction/endpoint ratio and IoU across semilandmark density
    and TPS regularisation.  lam > 0 makes phi a smoothing spline, so phi_1(A) != B and the
    endpoint residual grows; that is the price of a smoother grid and is reported."""
    rows = []
    for n in n_semis:
        for lam in lams:
            r = run_analysis(n_semi=n, lam=lam, verbose=False, controls=False)
            hl = r["headline"]
            rows.append({"n_semi": n, "lam": lam, "t_star": hl["t_star"],
                         "equidistance_residual_max": hl["equidistance_residual_max"],
                         "pred_over_endpoint": hl["pred_tps_vs_drawn_mean"] / hl["nearest_endpoint_mean"],
                         "pred_over_affine": hl["pred_tps_vs_drawn_mean"] / hl["pred_affine_vs_drawn_mean"],
                         "iou_pred_mean": hl["iou_pred_mean"], "iou_nearest_endpoint_mean": hl["iou_nearest_endpoint_mean"],
                         "tps_endpoint_residual_rms": hl["tps_endpoint_residual_rms"],
                         "pred_beats_nearest_endpoint_all": hl["pred_beats_nearest_endpoint_all"],
                         "t_star_monotone": hl["t_star_monotone"]})
    T = np.array([row["t_star"] for row in rows])
    out = {"rows": rows,
           "t_star_mean": T.mean(0).tolist(), "t_star_sd": T.std(0).tolist(),
           "t_star_cv": (T.std(0) / T.mean(0)).tolist(),
           "pred_over_endpoint_range": [float(min(r["pred_over_endpoint"] for r in rows)), float(max(r["pred_over_endpoint"] for r in rows))],
           "all_monotone": bool(all(r["t_star_monotone"] for r in rows)),
           "all_beat_endpoint": bool(all(r["pred_beats_nearest_endpoint_all"] for r in rows)),
           "n_settings": len(rows)}
    if verbose:
        print(f"[t1917-054] sweep {len(rows)} settings: t* mean {np.round(out['t_star_mean'], 3).tolist()} sd {np.round(out['t_star_sd'], 3).tolist()} "
              f"CV {np.round(out['t_star_cv'], 3).tolist()}; pred/endpoint in [{out['pred_over_endpoint_range'][0]:.3f}, {out['pred_over_endpoint_range'][1]:.3f}]; "
              f"all monotone={out['all_monotone']} all beat endpoint={out['all_beat_endpoint']}")
    return out


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


def _form_limits(*configs, pad: float = 0.08):
    P = np.vstack([c for c in configs])
    lo, hi = P.min(0), P.max(0)
    span = (hi - lo).max()
    return lo - pad * span, hi + pad * span


def render_comparison(res: dict, sweep: dict | None = None, specimen: dict | None = None, out: Path = COMPARISON_PNG) -> Path:
    plt = _plt()
    Z, cfgs, forms, dense = res["Z"], res["cfgs"], res["forms"], res["dense"]
    A, B, phiA, phiA_dense = Z[0], Z[4], res["phiA"], res["phiA_dense"]
    tps_sparse, phiA_sparse_dense = res["tps_sparse"], res["phiA_sparse_dense"]
    hl = res["headline"]
    cols = ["#1f77b4", "#2ca02c", "#9467bd", "#ff7f0e", "#d62728"]
    fig = plt.figure(figsize=(22, 13.5), dpi=100)
    gs = fig.add_gridspec(3, 5, width_ratios=[1.0, 0.9, 1.0, 1.0, 1.0])

    # --- column 0: source figures and the real specimen ----------------------------------------
    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(crop_page(DATA / FIG388_PAGE, FIG388_CROP), cmap="gray", vmin=0, vmax=255)
    ax.set_title("Thompson 1917 Fig. 388 — Archaeopteryx pelvis\nin Cartesian coordinates (p. 757)", fontsize=9)
    ax.axis("off")
    ax = fig.add_subplot(gs[1, 0])
    ax.imshow(crop_page(DATA / FIG388_PAGE, FIG389_CROP), cmap="gray", vmin=0, vmax=255)
    ax.set_title("Fig. 389 — Apatornis pelvis in the transformed grid", fontsize=9)
    ax.axis("off")
    ax = fig.add_subplot(gs[2, 0])
    if specimen is not None:
        im = Image.open(DATA / specimen["file"]).convert("RGB")
        ax.imshow(im)
        ax.contour(specimen["mask"].astype(float), levels=[0.5], colors=["#00e5ff"], linewidths=0.4)
        ax.set_title(f"real specimen: Archaeopteryx, Berlin (MfN; CC BY-SA 3.0 H. Raab)\n"
                     f"bone-vs-matrix CV mask: {100*specimen['bone_fraction']:.1f}% bone, {specimen['n_components']} comps — descriptive only, not in verdict", fontsize=8)
    ax.axis("off")

    # --- column 1: segmented Fig. 393 -----------------------------------------------------------
    ax = fig.add_subplot(gs[:, 1])
    ax.imshow(crop_page(DATA / FIG393_PAGE, FIG393_CROP), cmap="gray", vmin=0, vmax=255)
    for k, (f, c) in enumerate(zip(forms, cfgs)):
        cont = f["contour_rc"]
        ax.plot(cont[:, 1], cont[:, 0], color=cols[k], lw=1.0)
        X = c["X"]
        ax.plot(X[1:, 0], -X[1:, 1], ".", color=cols[k], ms=2.5)
        ax.plot(X[0, 0], -X[0, 1], "o", mfc="none", mec="k", ms=8)
        for nm, r in c["tip_rows"].items():
            ax.plot(X[r, 0], -X[r, 1], "s", color="k", ms=4)
        ax.text(X[:, 0].max() + 15, -X[:, 1].mean(), f"t={STAGES[k]:.2f}\nt*={res['stages'][k]['t_star']:.3f}" if 0 < k < 4 else f"t={STAGES[k]:.2f}",
                color=cols[k], fontsize=8, va="center")
    ax.set_title("Fig. 393 (p. 760) segmented: outlines, acetabulum (o),\n4 process tips (■), arc-length semilandmarks (·)", fontsize=9)
    ax.axis("off")

    # --- row 0, columns 2-4: predicted vs drawn stages ------------------------------------------
    for j, k in enumerate((1, 2, 3)):
        ax = fig.add_subplot(gs[0, 2 + j])
        t = STAGES[k]
        pred_cfg = interp_stage(A, phiA, t)
        pred_dense = interp_stage(dense[0], phiA_dense, t)
        pred_sparse_dense = interp_stage(dense[0], phiA_sparse_dense, t)
        beta, mx, my = similarity_params(Z[k], pred_cfg)
        drawn_al = apply_similarity(dense[k], beta, mx, my)
        for line in make_grid(A, per_line=40, pad=0.02):
            gl = interp_stage(line, tps_apply(tps_sparse, line), t)
            ax.plot(gl[:, 0], gl[:, 1], color="0.82", lw=0.5)
        ax.fill(*_closed(drawn_al).T, color="0.55", alpha=0.35, lw=0, label="Heilmann's drawn stage (Fig. 393)")
        ax.plot(*_closed(pred_dense).T, color=cols[k], lw=1.6, label="dense-correspondence interpolation")
        ax.plot(*_closed(pred_sparse_dense).T, color="k", lw=0.9, ls=":", label="9-anchor coarse grid (fold-free)")
        lo, hi = _form_limits(pred_dense, drawn_al)
        ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1])
        ax.set_aspect("equal")
        st = res["stages"][k]
        ax.set_title(f"stage {k} (t={t}): t*={st['t_star']:.3f}\nProcrustes dense {st['procrustes_pred_tps_vs_drawn']:.3f} · sparse {st['procrustes_pred_sparse_vs_drawn']:.3f} · "
                     f"affine {st['procrustes_pred_affine_vs_drawn']:.3f} · endpoint {st['nearest_endpoint_distance']:.3f}\n"
                     f"IoU dense {st['iou_pred_vs_drawn']:.2f} · sparse {st['iou_sparse_vs_drawn']:.2f} · endpoint {st['iou_nearest_endpoint_vs_drawn']:.2f}; r={st['pearson_r_pred']:.3f}", fontsize=8)
        if j == 0:
            ax.legend(fontsize=7, loc="lower left")
        ax.axis("off")

    # --- row 1: t* (with sweep spread), Procrustes bars, IoU bars ------------------------------
    ks = [1, 2, 3]
    ax = fig.add_subplot(gs[1, 2])
    ax.plot([0, 1], [0, 1], color="0.6", lw=1, label="t* = t (exact equidistance)")
    yerr = None
    if sweep is not None:
        yerr = np.array(sweep["t_star_sd"]) * 2
    ax.errorbar([STAGES[k] for k in ks], hl["t_star"], yerr=yerr, fmt="o-", color="k", capsize=3,
                label=f"best-fit stage t* (E7){'  ±2 SD over ' + str(sweep['n_settings']) + ' sweep settings' if sweep else ''}")
    for k in ks:
        ax.annotate(f"{hl['t_star'][k-1]:.3f}", (STAGES[k], hl["t_star"][k - 1]), textcoords="offset points", xytext=(6, -12), fontsize=8)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel("nominal stage t (Thompson: 1/4, 1/2, 3/4)"); ax.set_ylabel("t*")
    ax.set_title(f"equidistance: max |t*−t| = {hl['equidistance_residual_max']:.3f} (one stage = 0.25)\n"
                 f"pairing permutation p = {hl.get('permutation_p', float('nan')):.3f} (nominal pairing best of 3!)", fontsize=9)
    ax.legend(fontsize=7, loc="upper left")

    ax = fig.add_subplot(gs[1, 3])
    w = 0.05
    keys = [("procrustes_pred_tps_vs_drawn", cols[0], "dense interpolation"), ("procrustes_pred_sparse_vs_drawn", "#17becf", "9-anchor coarse grid"),
            ("procrustes_pred_affine_vs_drawn", "#aec7e8", "affine interpolation"), ("nearest_endpoint_distance", "0.5", "nearest endpoint (null)")]
    for i, (key, col, lab) in enumerate(keys):
        ax.bar([STAGES[k] + (i - 1.5) * w for k in ks], [res["stages"][k][key] for k in ks], width=w, color=col, label=lab)
    ax.set_xticks([STAGES[k] for k in ks]); ax.set_xlabel("stage"); ax.set_ylabel("full Procrustes distance to drawn stage")
    ax.set_title(f"prediction error vs controls (lower is better)\nmean dense {hl['pred_tps_vs_drawn_mean']:.3f} · sparse {hl['pred_sparse_vs_drawn_mean']:.3f} · "
                 f"affine {hl['pred_affine_vs_drawn_mean']:.3f} · endpoint {hl['nearest_endpoint_mean']:.3f}", fontsize=9)
    ax.legend(fontsize=7)

    ax = fig.add_subplot(gs[1, 4])
    keys = [("iou_pred_vs_drawn", cols[0], "dense interpolation"), ("iou_sparse_vs_drawn", "#17becf", "9-anchor coarse grid"),
            ("iou_affine_vs_drawn", "#aec7e8", "affine interpolation"), ("iou_nearest_endpoint_vs_drawn", "0.5", "nearest endpoint (null)")]
    for i, (key, col, lab) in enumerate(keys):
        ax.bar([STAGES[k] + (i - 1.5) * w for k in ks], [res["stages"][k][key] for k in ks], width=w, color=col, label=lab)
    ax.set_xticks([STAGES[k] for k in ks]); ax.set_xlabel("stage"); ax.set_ylabel("IoU of filled outlines (correspondence-free)")
    ax.set_ylim(0, 1)
    ax.set_title(f"IoU vs controls (higher is better)\nmean dense {hl['iou_pred_mean']:.3f} · sparse {hl['iou_sparse_mean']:.3f} · affine {hl['iou_affine_mean']:.3f} · endpoint {hl['iou_nearest_endpoint_mean']:.3f}", fontsize=9)
    ax.legend(fontsize=7, loc="lower right")

    # --- row 2: correspondence null, no-fold, descriptors ---------------------------------------
    ax = fig.add_subplot(gs[2, 2])
    if "correspondence_null" in res:
        cn = res["correspondence_null"]
        A_, B_ = Z[0], Z[4]
        nb = 4 * res["n_semi"]
        null = [float(np.mean([procrustes_align(interp_stage(A_, np.vstack([B_[:1], np.roll(B_[1:], s_, axis=0)]), STAGES[k]), Z[k])[1] for k in (1, 2, 3)])) for s_ in range(1, nb)]
        ax.hist(null, bins=24, color="0.6", label=f"{cn['n_null']} cyclic shifts of the Apatornis correspondence")
        ax.axvline(cn["observed_mean_procrustes"], color=cols[0], lw=2, label=f"anatomical correspondence: {cn['observed_mean_procrustes']:.3f}")
        ax.set_xlabel("mean Procrustes distance, prediction vs drawn stages 1-3"); ax.set_ylabel("count")
        ax.set_title(f"negative control: destroyed correspondence\nexact p = {cn['p_value']:.4f}, z = {cn['z_score']:.1f} (null min {cn['null_min']:.3f})", fontsize=9)
        ax.legend(fontsize=7)

    ax = fig.add_subplot(gs[2, 3])
    if "no_fold" in res:
        nf, nfs = res["no_fold"], res["no_fold_sparse"]
        ax.plot(nf["t"], nf["min_det_grid"], color=cols[0], lw=1.5, label="dense TPS, 9x8 grid")
        ax.plot(nf["t"], nf["min_det_inside"], color=cols[0], lw=1.5, ls="--", label="dense TPS, inside form")
        ax.plot(nfs["t"], nfs["min_det_grid"], color="#17becf", lw=1.5, label="9-anchor grid, 9x8 grid")
        ax.plot(nfs["t"], nfs["min_det_inside"], color="#17becf", lw=1.5, ls="--", label="9-anchor grid, inside form")
        ax.axhline(0, color="k", lw=0.8)
        ax.set_yscale("symlog", linthresh=1.0)
        ax.set_xlabel("stage t"); ax.set_ylabel("min det(D φ_t)  [symlog]")
        ax.set_title(f"E1 no-fold check (Lean interp_no_fold, 1-D)\ndense grid folds (min {hl['min_jacobian_det_dense_inside']:.1f} inside form); "
                     f"coarse grid stays a grid (min {hl['min_jacobian_det_sparse_inside']:.2f})", fontsize=9)
        ax.legend(fontsize=7, loc="lower left")

    ax = fig.add_subplot(gs[2, 4])
    ts = list(STAGES)
    for key, col, lab in [("pubis_angle_deg", cols[4], "pubis angle to ilium axis (deg)"), ("ischium_angle_deg", cols[3], "ischium angle to ilium axis (deg)")]:
        v = hl[key]
        ax.plot(ts, v, "o-", color=col, label=lab)
        ax.plot([0, 1], [v[0], v[4]], color=col, lw=0.8, ls=":")
    ax.set_xlabel("stage t"); ax.set_ylabel("angle (deg)")
    ax2 = ax.twinx()
    v = hl["preacetabular_over_postacetabular"]
    ax2.plot(ts, v, "s-", color="0.3", label="pre-/post-acetabular ilium ratio")
    ax2.plot([0, 1], [v[0], v[4]], color="0.3", lw=0.8, ls=":")
    ax2.set_ylabel("ratio")
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=7, loc="upper left")
    ax.set_title("descriptors along Heilmann's series (dotted: linear interpolation of endpoints)\n"
                 f"centroid size px {[int(round(x)) for x in hl['centroid_size_px']]} (not monotone: not drawn to grid scale)", fontsize=9)

    fig.suptitle("T1917-054 · Thompson (1917) Ch. XVII Figs. 388-393: Archaeopteryx → Apatornis pelvis by equidistant grid interpolation — "
                 "part A (interpolation yields equidistant intermediates) tested on Heilmann's own figures; part B ('biologically real') untestable with pinned data",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    return out


def render_growth_gif(res: dict, out: Path = GROWTH_GIF, n_frames: int = 33) -> Path:
    plt = _plt()
    Z, dense, phiA_dense, tps_sparse = res["Z"], res["dense"], res["phiA_dense"], res["tps_sparse"]
    A = Z[0]
    frames = []
    lo, hi = _form_limits(dense[0], dense[4], interp_stage(dense[0], phiA_dense, 0.5), pad=0.12)
    for t in np.linspace(0, 1, n_frames):
        fig, ax = plt.subplots(figsize=(6.4, 4.4), dpi=90)
        for line in make_grid(A, per_line=40, pad=0.02):
            gl = interp_stage(line, tps_apply(tps_sparse, line), t)
            ax.plot(gl[:, 0], gl[:, 1], color="0.78", lw=0.6)
        pred = interp_stage(dense[0], phiA_dense, t)
        ax.fill(*_closed(pred).T, color="#555555", alpha=0.85)
        k = int(round(t * 4))
        if abs(t - STAGES[k]) < 0.5 / (n_frames - 1):
            beta, mx, my = similarity_params(Z[k], interp_stage(A, res["phiA"], t))
            ax.plot(*_closed(apply_similarity(dense[k], beta, mx, my)).T, color="#d62728", lw=1.2, ls="--")
            ax.text(lo[0] + 0.02, hi[1] - 0.05, f"red dashed: Heilmann's drawn {STAGE_NAMES[k]}", fontsize=8, color="#d62728")
        ax.set_xlim(lo[0], hi[0])
        ax.set_ylim(lo[1], hi[1])
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(f"T1917-054 · φ_t = (1−t)·id + t·φ,  t = {t:.2f}   (Archaeopteryx → Apatornis; grid = 9-anchor coarse TPS)", fontsize=8.5)
        fig.canvas.draw()
        frames.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()))
        plt.close(fig)
    out.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(out, save_all=True, append_images=frames[1:], duration=120, loop=0)
    return out


# --------------------------------------------------------------------------- #
# 7. verdict and MorphoEval measurements
# --------------------------------------------------------------------------- #
def part_a_supported(hl: dict) -> bool:
    return bool(hl["t_star_monotone"] and hl["t_star_strictly_inside_0_1"] and hl["pred_beats_nearest_endpoint_all"]
                and hl["pred_beats_affine_all"] and hl.get("correspondence_null_p", 1.0) < 0.05)


def write_result(res: dict, ref: dict, sweep: dict | None = None, specimen: dict | None = None, out: Path = RESULT_JSON) -> Path:
    hl = res["headline"]
    part_a = part_a_supported(hl)
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
        "verdict": "mixed",
        "verdict_by_part": {
            "A_methodological_interpolation_yields_equidistant_intermediates": "supported (approximately equidistant)" if part_a else "refuted",
            "B_intermediates_are_biologically_real": "untestable",
        },
        "headline_numbers": {
            "procrustes_distance_archaeopteryx_apatornis": round(hl["procrustes_distance_archaeopteryx_apatornis"], 4),
            "t_star_stages_1_3": [round(v, 4) for v in hl["t_star"]],
            "t_nominal": hl["t_nominal"],
            "equidistance_residual_max": round(hl["equidistance_residual_max"], 4),
            "procrustes_vs_drawn_mean": {"dense_interpolation": round(hl["pred_tps_vs_drawn_mean"], 4), "coarse_9_anchor_grid": round(hl["pred_sparse_vs_drawn_mean"], 4),
                                         "affine_interpolation": round(hl["pred_affine_vs_drawn_mean"], 4), "nearest_endpoint_null": round(hl["nearest_endpoint_mean"], 4)},
            "iou_vs_drawn_mean": {"dense_interpolation": round(hl["iou_pred_mean"], 4), "coarse_9_anchor_grid": round(hl["iou_sparse_mean"], 4),
                                  "affine_interpolation": round(hl["iou_affine_mean"], 4), "nearest_endpoint_null": round(hl["iou_nearest_endpoint_mean"], 4)},
            "boundary_rmse_over_centroid_size_mean": {"dense_interpolation": round(hl["chamfer_pred_mean_cs"], 5), "nearest_endpoint_null": round(hl["chamfer_nearest_endpoint_mean_cs"], 5)},
            "pearson_r_pred_vs_drawn_mean": round(hl["pearson_r_pred_mean"], 4),
            "pearson_r_nearest_endpoint_mean": round(hl["pearson_r_nearest_endpoint_mean"], 4),
            "ks_p_pred_residuals_smaller_than_endpoint": [float(f"{v:.3g}") for v in hl["ks_p_values_vs_nearest_endpoint"]],
            "pairing_permutation_p": hl.get("permutation_p"),
            "correspondence_shift_null": {"p": hl.get("correspondence_null_p"), "z": round(hl.get("correspondence_null_z", float("nan")), 2)},
            "min_jacobian_det": {"dense_tps_inside_form": round(hl.get("min_jacobian_det_dense_inside", float("nan")), 3),
                                 "coarse_9_anchor_grid_inside_form": round(hl.get("min_jacobian_det_sparse_inside", float("nan")), 3)},
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
            "Part A is scored against Heilmann's drawn intermediates (Fig. 393), i.e. it tests that the construction is well-defined, approximately equidistant and reproducible, not that it predicts fossils.",
            "The dense semilandmark TPS reproduces the drawn stages best but folds (det < 0) as a grid; the 9-anchor coarse grid is fold-free (a genuine Thompson grid) and beats the nearest endpoint on stages 1-2 but not stage 3.",
        ],
    }
    if sweep is not None:
        doc["parameter_sweep"] = {k: v for k, v in sweep.items() if k != "rows"}
        doc["parameter_sweep"]["rows"] = [{k: (round(v, 4) if isinstance(v, float) else ([round(x, 4) for x in v] if isinstance(v, list) else v)) for k, v in r.items()} for r in sweep["rows"]]
    if specimen is not None:
        doc["specimen_cv_descriptive_only"] = {k: v for k, v in specimen.items() if k != "mask"}
    out.write_text(json.dumps(doc, indent=2), encoding="utf-8")
    return out


def measure_morphoeval() -> list[dict]:
    """MorphoEval dynamic measurements (consumed by morphoeval/measure_outputs.py)."""
    res = run_analysis(verbose=False)
    sweep = parameter_sweep(verbose=False)
    hl = res["headline"]
    st = res["stages"]
    d_AB = hl["procrustes_distance_archaeopteryx_apatornis"]

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
          source="Thompson 1917 p. 759: 'three intermediate and equidistant co-ordinate systems'",
          note=f"sweep over {sweep['n_settings']} settings: mean {sweep['t_star_mean'][0]:.3f} ± SD {sweep['t_star_sd'][0]:.3f} (CV {sweep['t_star_cv'][0]:.3f})"),
        M("best-fit stage t* of the second intermediate (Fig. 392 top)", st[2]["t_star"], "t", "E7 projection", reference=0.5,
          source="Thompson 1917 p. 759", note=f"sweep mean {sweep['t_star_mean'][1]:.3f} ± SD {sweep['t_star_sd'][1]:.3f} (CV {sweep['t_star_cv'][1]:.3f})"),
        M("best-fit stage t* of the third intermediate (Fig. 392 bottom)", st[3]["t_star"], "t", "E7 projection", reference=0.75,
          source="Thompson 1917 p. 759", note=f"sweep mean {sweep['t_star_mean'][2]:.3f} ± SD {sweep['t_star_sd'][2]:.3f} (CV {sweep['t_star_cv'][2]:.3f})"),
        M("max equidistance residual |t* - k/4| over the three stages", hl["equidistance_residual_max"], "t units",
          "max over k of |t*_k - k/4|", reference=0.0, source="exact equidistance (Lean: interp_endpoints_and_equidistant)",
          note=f"mean residual {hl['equidistance_residual_mean']:.4f}; d(A,B) = {d_AB:.4f} so one stage = {d_AB/4:.4f} Procrustes units; pairing permutation p = {hl['permutation_p']:.3f}"),
        M("mean full Procrustes distance, dense interpolation vs drawn intermediate", hl["pred_tps_vs_drawn_mean"], "Procrustes distance",
          "E2/E4: TPS phi_t applied to the Archaeopteryx configuration, full Procrustes fit onto the drawn stage, mean over stages 1-3",
          reference=round(hl["nearest_endpoint_mean"], 4), source="negative control: copy the nearest endpoint (same metric)",
          note=f"ratio vs endpoint control {hl['pred_tps_vs_drawn_mean']/hl['nearest_endpoint_mean']:.3f} (sweep range {sweep['pred_over_endpoint_range'][0]:.3f}-{sweep['pred_over_endpoint_range'][1]:.3f}); "
               f"affine interpolation {hl['pred_affine_vs_drawn_mean']:.4f}; 9-anchor coarse grid {hl['pred_sparse_vs_drawn_mean']:.4f}; "
               f"correspondence-shift null p = {hl['correspondence_null_p']:.4f}, z = {hl['correspondence_null_z']:.1f}"),
        M("mean IoU of filled outlines, dense interpolation vs drawn intermediate", hl["iou_pred_mean"], "IoU (intersection/union)",
          "dense Archaeopteryx outline carried through phi_t, similarity-superimposed on the drawn stage, rasterised 400x400", reference=round(hl["iou_nearest_endpoint_mean"], 4),
          source="negative control: nearest endpoint outline (same metric)",
          note=f"affine {hl['iou_affine_mean']:.3f}; 9-anchor coarse grid {hl['iou_sparse_mean']:.3f}; symmetric boundary RMSE/centroid size {hl['chamfer_pred_mean_cs']:.4f} vs endpoint {hl['chamfer_nearest_endpoint_mean_cs']:.4f}"),
        M("Pearson r, predicted vs drawn landmark coordinates (mean over stages 1-3)", hl["pearson_r_pred_mean"], "r",
          "corrcoef of flattened Procrustes-aligned coordinates", reference=round(hl["pearson_r_nearest_endpoint_mean"], 4), source="nearest-endpoint control",
          note=f"RMSE/centroid size {hl['rmse_pred_mean_cs']:.4f} vs endpoint {hl['rmse_nearest_endpoint_mean_cs']:.4f}; KS (pred residuals < endpoint residuals) p = "
               + ", ".join(f"{v:.1e}" for v in hl["ks_p_values_vs_nearest_endpoint"])),
        M("min det(D phi_t) inside the Archaeopteryx form, 9-anchor coarse grid, t in [0, 1]", hl["min_jacobian_det_sparse_inside"], "det (1 = identity)",
          "central-difference Jacobian of the TPS phi_t on reference-grid points inside the dense outline, min over 21 stages", reference=0.0,
          source="E1 / Lean interp_no_fold: a valid grid needs det > 0 at every stage",
          note=f"dense semilandmark TPS folds: min det {hl['min_jacobian_det_dense_inside']:.2f} inside the form ({hl['min_jacobian_det_dense_grid']:.1f} on the padded 9x8 grid)"),
        M("Archaeopteryx -> Apatornis full Procrustes distance (Fig. 393 endpoints)", d_AB, "Procrustes distance",
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
    # Procrustes: invariance to similarity; similarity_params round trip
    c, th = 2.3, 0.7
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    Xs = c * X @ R.T + 5.0
    _, d0, _ = procrustes_align(X, Y)
    _, d1, _ = procrustes_align(Xs, Y)
    assert abs(d0 - d1) < 1e-10
    beta, mx, my = similarity_params(X, Xs)
    assert np.allclose(apply_similarity(X, beta, mx, my), Xs, atol=1e-9)
    print("ok  Procrustes distance invariant to similarity transforms; similarity round trip")
    # E1 no-fold on a pure shear and a pure scaling (Lean shear_interp_preserves_area / scaling_interp_orientation_preserving)
    sq = np.array([[0, 0], [1, 0], [1, 1], [0, 1], [0.5, 0.5]], float)
    shear = tps_fit(sq, sq @ np.array([[1, 0], [0.7, 1]]))
    scal = tps_fit(sq, sq * np.array([2.0, 1.5]))
    assert abs(jacobian_min_det(shear, sq, 0.5) - 1.0) < 1e-6
    assert jacobian_min_det(scal, sq, 0.5) > 0 and abs(jacobian_min_det(scal, sq, 1.0) - 3.0) < 1e-6
    print("ok  E1/E5 numerical Jacobian: shear keeps det = 1 at mid-stage, scaling det = 3 at t = 1")
    # IoU / chamfer sanity
    sq_poly = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], float)
    assert abs(outline_iou(sq_poly, sq_poly) - 1.0) < 0.01 and abs(outline_iou(sq_poly, sq_poly + [0.5, 0]) - 1 / 3) < 0.02
    assert chamfer_rmse(sq_poly, sq_poly) == 0.0
    print("ok  IoU and boundary RMSE sanity (identity = 1 / 0; half-shifted square IoU = 1/3)")
    # real data
    res = run_analysis(verbose=False)
    areas = [f["raw_area_px"] for f in res["forms"]]
    want = ref["digitisation"]["expected_component_area_px"]
    for a, w in zip(areas, [want["archaeopteryx"], want["stage_1"], want["stage_2"], want["stage_3"], want["apatornis"]]):
        assert abs(a - w) / w < 0.05, (a, w)
    print(f"ok  five silhouettes segmented, areas within 5% of the pinned reference: {areas}")
    for cfg in res["cfgs"]:
        assert cfg["X"].shape == (1 + 4 * res["n_semi"], 2)
    print("ok  landmark configurations (acetabulum + 4 segments) for all five forms")
    hl = res["headline"]
    assert hl["t_star_monotone"] and hl["t_star_strictly_inside_0_1"], hl["t_star"]
    print(f"ok  t* monotone and inside (0,1): {np.round(hl['t_star'], 3).tolist()}")
    assert hl["pred_beats_nearest_endpoint_all"] and hl["pred_beats_affine_all"]
    print("ok  dense interpolation closer to each drawn stage than the nearest endpoint and than the affine interpolation")
    assert res["permutation"]["nominal_is_best"]
    assert res["correspondence_null"]["p_value"] < 0.05 and res["correspondence_null"]["observed_mean_procrustes"] < res["correspondence_null"]["null_min"]
    print(f"ok  nominal pairing is the best of 3!; anatomical correspondence beats all {res['correspondence_null']['n_null']} cyclic shifts (p={res['correspondence_null']['p_value']:.4f})")
    assert all(p < 0.01 for p in hl["ks_p_values_vs_nearest_endpoint"])
    print("ok  KS: prediction residuals stochastically smaller than endpoint-control residuals (p < 0.01, all stages)")
    assert res["no_fold_sparse"]["no_fold_grid"] and res["no_fold_sparse"]["no_fold_inside"]
    assert abs(res["no_fold"]["min_det_grid"][0] - 1.0) < 1e-6
    print(f"ok  9-anchor coarse grid fold-free at all 21 stages (min det {res['no_fold_sparse']['min_over_stages_inside']:.3f}); identity at t = 0")
    res2 = run_analysis(verbose=False, controls=False)
    assert res2["headline"]["t_star"] == hl["t_star"]
    print("ok  deterministic (two runs identical)")
    sp = segment_specimen_bone(DATA / "archaeopteryx_lithographica_berlin_specimen_mfn.jpg")
    assert 0.05 < sp["bone_fraction"] < 0.8
    print(f"ok  Berlin specimen bone-vs-matrix segmentation runs (bone fraction {sp['bone_fraction']:.3f}; descriptive only)")
    print(f"self-test passed in {time.time() - t0:.1f} s")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--n-semi", type=int, default=24, help="semilandmarks per boundary segment (default 24)")
    ap.add_argument("--lam", type=float, default=0.0, help="TPS regularisation (0 = interpolating)")
    ap.add_argument("--no-sweep", action="store_true", help="skip the n_semi x lambda parameter sweep")
    ap.add_argument("--no-specimen", action="store_true", help="skip the Berlin specimen CV pass")
    ap.add_argument("--no-gif", action="store_true")
    args = ap.parse_args(argv)
    if args.self_test:
        return self_test()
    ref = load_reference()
    res = run_analysis(n_semi=args.n_semi, lam=args.lam)
    sweep = None if args.no_sweep else parameter_sweep()
    specimen = None if args.no_specimen else segment_specimen_bone(DATA / "archaeopteryx_lithographica_berlin_specimen_mfn.jpg")
    if specimen is not None:
        print(f"[t1917-054] Berlin specimen CV (descriptive, not in verdict): bone fraction {specimen['bone_fraction']:.3f}, {specimen['n_components']} components")
    print("->", render_comparison(res, sweep=sweep, specimen=specimen).relative_to(ROOT))
    if not args.no_gif:
        print("->", render_growth_gif(res).relative_to(ROOT))
    print("->", write_result(res, ref, sweep=sweep, specimen=specimen).relative_to(ROOT))
    return 0


if __name__ == "__main__":
    sys.exit(main())

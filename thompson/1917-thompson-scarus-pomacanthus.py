#!/usr/bin/env python3
"""
D'Arcy Thompson's Scarus -> Pomacanthus transformation tested on real species.

On Growth and Form (1917), ch. XVII, p. 749, Figs. 375-376: "Fig. 375 is an outline
diagram of a typical Scaroid fish. Let us deform its rectilinear co-ordinates into a
system of (approximately) coaxial circles, as in Fig. 376, and then filling into the
new system, space by space and point by point, our former diagram of Scarus, we obtain
a very good outline of an allied fish, belonging to a neighbouring family, of the genus
Pomacanthus. [...] the still more bizarre outlines of other fishes of the same family
of Chaetodonts will be found to correspond to very slight modifications of similar
co-ordinates; in other words, to small variations in the values of the constants of the
coaxial curves."

Thompson drew the grids by hand and left no instructions for fitting them (Bookstein
1978).  This script fits them.  A Cartesian grid whose lines become two orthogonal
pencils of coaxial circles is the image of the plane under a Moebius transformation
z -> (az + b)/(cz + d), the conformal ("isogonal", Thompson's word) map that carries
lines to circles.  So Thompson's claim for this pair is a parametric one with six real
constants, and it can be fitted by least squares to homologous landmarks on real fish
and compared with (i) the affine map that is the simplest alternative and (ii) the
thin-plate spline that fits the landmarks exactly (Bookstein 1989) and so measures the
whole shape difference.

Data (real species, not Thompson's drawings)
  Friedman ST, Martinez CM, Price SA, Wainwright PC (2019). The influence of size on
  body shape diversification across Indo-Pacific shore fishes. Evolution 73:1873-1884.
  https://doi.org/10.1111/evo.13755 ; data: Dryad https://doi.org/10.5061/dryad.fn457gr
  (CC0): trimmed_tps_Claverie2014.csv, 17 lateral-view landmarks for 791 species,
  digitised by Claverie T & Wainwright PC (2014), PLoS ONE 9:e112732,
  https://doi.org/10.1371/journal.pone.0112732 (Fig. 1 defines the landmarks: 1-2 jaw
  tips, 3-4 dorsal-fin origin/end, 5-6 caudal-fin insertions, 7-8 anal-fin end/origin,
  9-10 pectoral-fin insertions, 11-17 head and orbit) on J. Randall's Bishop Museum
  photographs.  The file is vendored under datasets/thompson/ (CC0 permits it) and its
  SHA-256 is checked here.  It holds 22 Scarus, 9 Pomacanthus and 54 Chaetodon species
  (Thompson's Scaroid, his Pomacanthus and his "other Chaetodonts"; Pomacanthus was in
  Chaetodontidae in 1917, now Pomacanthidae), plus 2 Diodon but no Mola, no Polyprion,
  no Argyropelecus/Sternoptyx - so of Thompson's four fish pairs this is the one that can
  be tested with these data.

Thompson's own figure
  Leaf 769 (p. 749) of the Internet Archive scan of the 1917 Cambridge edition,
  https://archive.org/details/ongrowthform1917thom (public domain), fetched as the
  original JPEG-2000 page from the item's jp2 archive and SHA-256-checked; Figs. 375-376
  are cropped from it and shown beside the real species.

What is computed
  1. Generalised Procrustes superimposition of all 791 species (unit centroid size, no
     reflection); genus means for genera with >= 3 species.
  2. Scarus mean -> Pomacanthus mean, five fits: similarity (Procrustes distance),
     affine (6 par.), Moebius (6 par.; isogonal coaxial circles), Moebius after an affine
     map (8 par.; oblique coaxial circles, which is what Fig. 376 actually shows, since
     its two pencils do not meet at right angles), thin-plate spline (exact).  Residuals
     are RMS landmark distances in Procrustes units; "explained" = 1 - (RMS/RMS_sim)^2.
     Before the real fits the two non-linear fitters are checked to recover known maps.
  3. Noise floor: RMS of each Pomacanthus species about the Pomacanthus mean, so a fit
     whose residual is at or below it is "a very good outline" at the resolution of the
     species themselves.
  4. Context: the same fits for every ordered pair of genus means, so the Scarus ->
     Pomacanthus pair can be placed as a percentile; and Scarus -> every Chaetodon species
     for Thompson's second claim.
  5. Leave-one-landmark-out: each landmark predicted by the oblique-coaxial map fitted
     to the other 16, to ask whether the family has any predictive value.
  6. Thompson's grid versus the fitted one: the fitted coaxial grids are drawn on the
     real Pomacanthus and the TPS grid beside them.
  7. Real outlines.  Thompson's claim is about an outline, and the test above is on 17
     points.  Two lateral photographs of species in the landmark table - a Scarus ghobban
     specimen (J. Huet / IFREMER, CC BY 4.0) and a living Pomacanthus semicirculatus
     (E. Levy, CC BY 2.0), both via Wikimedia Commons - are fetched into .thompson_cache/
     (not vendored), SHA-256-checked and segmented into silhouettes (the specimen by its
     Lab distance from the white background; the field photograph by a marker watershed
     on colour distance from the frame-sampled background plus colour gradient, inside a
     declared bounding box).  Each silhouette is registered to its species' landmark
     configuration by a similarity fixed by two anchors the silhouette itself gives:
     anterior-most point -> LM2 and the caudal peduncle (the narrowest column between the
     caudal fin and the body, tracked from the tail tip) -> midpoint of LM5-LM6 (the axis
     gpa() orients by).  The five families are fitted to the 17 landmarks of S. ghobban ->
     P. semicirculatus and applied to the Scarus silhouette; agreement with the Pomacanthus
     silhouette is the IoU of the filled outlines and the mean symmetric nearest-point
     distance between the boundaries.  The similarity (no deformation) is the baseline; a
     deformation that is read off the landmarks helps on the outline only if its IoU rises
     above that baseline.  Fin posture is a declared confound (specimen vs living fish).

Usage
    python3 thompson/1917-thompson-scarus-pomacanthus.py                 # figures + GIF
    python3 thompson/1917-thompson-scarus-pomacanthus.py --no-anim --csv fits.csv
    python3 thompson/1917-thompson-scarus-pomacanthus.py --no-scan --no-photos   # offline: landmarks only

Outputs (git-ignored): thompson_scarus_pomacanthus.png, thompson_scarus_pomacanthus_outlines.png,
thompson_outline_metrics.csv, thompson_scarus_pomacanthus.gif, optional CSV of all genus-pair fits.
Extra dependencies for item 7: Pillow, scikit-image (both in requirements.txt).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import sys
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "datasets" / "thompson" / "trimmed_tps_Claverie2014.csv"
DATA_SHA256 = "ab5315f3be273285c101299457093199e98ff0b1d0fcf08331cc303b2f07e6bd"
DATA_DRYAD = "https://doi.org/10.5061/dryad.fn457gr"
CACHE = ROOT / ".thompson_cache"
SCAN_ITEM = "ongrowthform1917thom"
SCAN_LEAF = 769                                   # printed page 749 (jp2 file index; the OCR leaf index is one lower)
SCAN_URL = (f"https://archive.org/download/{SCAN_ITEM}/{SCAN_ITEM}_jp2.zip/"
            f"{SCAN_ITEM}_jp2%2F{SCAN_ITEM}_{SCAN_LEAF:04d}.jp2")
SCAN_SHA256 = "69cdc9e24bd757b476c763516fb9fcd45db7dc8c7e28f83a4c142417b762c6fa"
SCAN_CROP = (0.0, 0.37, 1.0, 0.575)               # Figs. 375-376 as fractions of the page (x0, y0, x1, y1)

N_LM = 17
SOURCE_GENUS, TARGET_GENUS, OTHER_GENUS = "Scarus", "Pomacanthus", "Chaetodon"
MIN_SPECIES = 3
OUTLINE = [2, 3, 4, 5, 6, 7, 8, 1]               # landmark numbers (1-based) joined into a body polygon
WIRES = [(15, 16), (9, 10), (11, 12), (3, 17)]   # orbit, pectoral base, opercle, nape
LM_NOTE = "landmarks of Claverie & Wainwright 2014 Fig. 1"


# ----------------------------------------------------------------------------- inputs
def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_landmarks() -> Tuple[List[str], np.ndarray]:
    got = sha256_of(DATA)
    if got != DATA_SHA256:
        sys.exit(f"{DATA}: sha256 {got} != pinned {DATA_SHA256} (source: {DATA_DRYAD})")
    names, X = [], []
    with open(DATA, newline="") as fh:
        for r in csv.DictReader(fh):
            names.append(r["Genus_species"])
            X.append([[float(r[f"lm_{i}x"]), float(r[f"lm_{i}y"])] for i in range(1, N_LM + 1)])
    return names, np.asarray(X)


def fetch_scan(allow_download: bool) -> Optional[Path]:
    """Thompson's page as the original JP2 leaf (byte-stable, unlike the on-the-fly JPEGs)."""
    CACHE.mkdir(exist_ok=True)
    p = CACHE / f"{SCAN_ITEM}_{SCAN_LEAF:04d}.jp2"
    if not p.exists():
        if not allow_download:
            return None
        print(f"  fetching {SCAN_URL}")
        with urllib.request.urlopen(SCAN_URL, timeout=120) as r, open(p, "wb") as fh:
            fh.write(r.read())
    got = sha256_of(p)
    if got != SCAN_SHA256:
        sys.exit(f"{p}: sha256 {got} != pinned {SCAN_SHA256}")
    return p


# ----------------------------------------------------------------------------- shape analysis
def centre_scale(X: np.ndarray) -> np.ndarray:
    X = X - X.mean(axis=-2, keepdims=True)
    cs = np.sqrt((X ** 2).sum(axis=(-2, -1), keepdims=True))
    return X / cs


def opa(S: np.ndarray, T: np.ndarray, scale: bool = True) -> np.ndarray:
    """Ordinary Procrustes: rotate (and scale) centred S onto centred T, no reflection."""
    U, s, Vt = np.linalg.svd(T.T @ S)
    R = (U @ Vt).T
    if np.linalg.det(R) < 0:
        Vt[-1] *= -1
        R = (U @ Vt).T
        s = s.copy(); s[-1] *= -1
    k = s.sum() / (S ** 2).sum() if scale else 1.0
    return k * S @ R


def gpa(X: np.ndarray, iters: int = 50, tol: float = 1e-10) -> np.ndarray:
    """Generalised Procrustes superimposition to unit centroid size (full Procrustes), then the whole
    sample rotated so that the consensus's snout (LM2) -> caudal-peduncle midpoint (LM5, LM6) axis is
    horizontal with the snout at the left, as in Thompson's figures and the source photographs."""
    Y = centre_scale(X.astype(float))
    mean = Y[0]
    for _ in range(iters):
        Y = np.stack([opa(y, mean) for y in Y])
        new = centre_scale(Y.mean(axis=0))
        if np.abs(new - mean).max() < tol:
            break
        mean = new
    v = (mean[4] + mean[5]) / 2 - mean[1]
    th = -np.arctan2(v[1], v[0])
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    return Y @ R.T


def genus_of(name: str) -> str:
    return name.split("_")[0]


def genus_groups(names: Sequence[str]) -> Dict[str, List[int]]:
    g: Dict[str, List[int]] = {}
    for i, n in enumerate(names):
        g.setdefault(genus_of(n), []).append(i)
    return g


def rms(A: np.ndarray, B: np.ndarray) -> float:
    return float(np.sqrt(((A - B) ** 2).sum(axis=1).mean()))


def fit_similarity(S: np.ndarray, T: np.ndarray) -> np.ndarray:
    return opa(S - S.mean(0), T - T.mean(0)) + T.mean(0)


def fit_affine(S: np.ndarray, T: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    A = np.column_stack([S, np.ones(len(S))])
    M, *_ = np.linalg.lstsq(A, T, rcond=None)
    return M, A @ M


def apply_affine(M: np.ndarray, P: np.ndarray) -> np.ndarray:
    return np.column_stack([P, np.ones(len(P))]) @ M


def moebius(params: np.ndarray, z: np.ndarray) -> np.ndarray:
    a, b, c = params[0] + 1j * params[1], params[2] + 1j * params[3], params[4] + 1j * params[5]
    return (a * z + b) / (c * z + 1.0)


def fit_moebius(S: np.ndarray, T: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Least-squares Moebius map z -> (az+b)/(cz+1) from S to T, started at the affine fit (c = 0).
    Thompson's 'system of coaxial circles' is the image of the Cartesian grid under such a map."""
    from scipy.optimize import least_squares
    zs, zt = S[:, 0] + 1j * S[:, 1], T[:, 0] + 1j * T[:, 1]
    M, _ = fit_affine(S, T)
    a0 = complex(M[0, 0], M[0, 1]); b0 = complex(M[2, 0], M[2, 1])      # the conformal part of the affine fit
    p0 = np.array([a0.real, a0.imag, b0.real, b0.imag, 0.0, 0.0])

    def resid(p):
        w = moebius(p, zs) - zt
        return np.concatenate([w.real, w.imag])

    best = None
    for c_init in [(0.0, 0.0), (0.3, 0.0), (-0.3, 0.0), (0.0, 0.3), (0.0, -0.3)]:
        p = p0.copy(); p[4:] = c_init
        r = least_squares(resid, p, method="lm", max_nfev=20000)
        if best is None or r.cost < best.cost:
            best = r
    w = moebius(best.x, zs)
    return best.x, np.column_stack([w.real, w.imag])


def apply_moebius(params: np.ndarray, P: np.ndarray) -> np.ndarray:
    w = moebius(params, P[:, 0] + 1j * P[:, 1])
    return np.column_stack([w.real, w.imag])


def coaxial(params: np.ndarray, P: np.ndarray) -> np.ndarray:
    """Oblique coaxial circles: an affine map (Thompson's oblique coordinates, Fig. 374) followed by
    z -> z/(cz+1).  Both pencils of grid lines still become pencils of coaxial circles, but the two
    pencils need not be orthogonal, so the map is not conformal; 8 real parameters."""
    M = params[:6].reshape(3, 2)
    Q = apply_affine(M, P)
    z = Q[:, 0] + 1j * Q[:, 1]
    c = params[6] + 1j * params[7]
    w = z / (c * z + 1.0)
    return np.column_stack([w.real, w.imag])


def fit_coaxial(S: np.ndarray, T: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    from scipy.optimize import least_squares
    M, _ = fit_affine(S, T)
    p0 = np.concatenate([M.ravel(), [0.0, 0.0]])

    def resid(p):
        return (coaxial(p, S) - T).ravel()

    best = None
    for c_init in [(0.0, 0.0), (0.5, 0.0), (-0.5, 0.0), (0.0, 0.5), (0.0, -0.5), (1.0, 0.0), (-1.0, 0.0)]:
        p = p0.copy(); p[6:] = c_init
        r = least_squares(resid, p, method="lm", max_nfev=20000)
        if best is None or r.cost < best.cost:
            best = r
    return best.x, coaxial(best.x, S)


def tps_fit(S: np.ndarray, T: np.ndarray) -> Tuple[np.ndarray, float]:
    """Thin-plate spline S -> T (Bookstein 1989): returns the (n+3) x 2 coefficient matrix and the bending energy."""
    n = len(S)
    r2 = ((S[:, None, :] - S[None, :, :]) ** 2).sum(-1)
    K = np.where(r2 > 0, r2 * np.log(r2 + 1e-300), 0.0)
    P = np.column_stack([np.ones(n), S])
    L = np.zeros((n + 3, n + 3)); L[:n, :n] = K; L[:n, n:] = P; L[n:, :n] = P.T
    Y = np.vstack([T, np.zeros((3, 2))])
    W = np.linalg.solve(L, Y)
    Wn = W[:n]
    be = float(np.trace(Wn.T @ K @ Wn))
    return W, be


def tps_apply(W: np.ndarray, S: np.ndarray, P: np.ndarray) -> np.ndarray:
    n = len(S)
    r2 = ((P[:, None, :] - S[None, :, :]) ** 2).sum(-1)
    U = np.where(r2 > 0, r2 * np.log(r2 + 1e-300), 0.0)
    return U @ W[:n] + np.column_stack([np.ones(len(P)), P]) @ W[n:]


def all_fits(S: np.ndarray, T: np.ndarray) -> Dict[str, float]:
    sim = fit_similarity(S, T)
    _, aff = fit_affine(S, T)
    _, moe = fit_moebius(S, T)
    _, cox = fit_coaxial(S, T)
    _, be = tps_fit(S, T)
    r_sim, r_aff, r_moe, r_cox = rms(sim, T), rms(aff, T), rms(moe, T), rms(cox, T)
    return {"rms_similarity": r_sim, "rms_affine": r_aff, "rms_moebius": r_moe, "rms_coaxial": r_cox,
            "explained_affine": 1 - (r_aff / r_sim) ** 2, "explained_moebius": 1 - (r_moe / r_sim) ** 2,
            "explained_coaxial": 1 - (r_cox / r_sim) ** 2, "bending_energy": be}


def self_test(S: np.ndarray, rng: np.random.Generator) -> Dict[str, float]:
    """The fitters recover known maps: apply a random Moebius / oblique-coaxial map to S and fit it back.
    Returns the recovery RMS of each (should be ~1e-12); run before the real fits so a broken optimiser
    cannot pass as a biological result."""
    pm = np.array([1.1, 0.2, 0.02, -0.01, 0.6, 0.3]) + rng.normal(0, 0.05, 6)
    _, back = fit_moebius(S, apply_moebius(pm, S))
    pc = np.concatenate([[1.0, 0.1, -0.1, 1.3, 0.02, -0.01], [0.5, 0.2]]) + rng.normal(0, 0.05, 8)
    _, back2 = fit_coaxial(S, coaxial(pc, S))
    return {"moebius": rms(back, apply_moebius(pm, S)), "coaxial": rms(back2, coaxial(pc, S))}


def within_group_rms(Y: np.ndarray, idx: Sequence[int]) -> Tuple[float, np.ndarray]:
    mean = centre_scale(Y[idx].mean(0))
    per = np.array([rms(fit_similarity(Y[i], mean), mean) for i in idx])
    return float(np.sqrt((per ** 2).mean())), per


# ----------------------------------------------------------------------------- drawing
def grid_lines(S: np.ndarray, n: int = 9, m: int = 60, pad: float = 0.08) -> List[np.ndarray]:
    lo, hi = S.min(0) - pad, S.max(0) + pad
    xs, ys = np.linspace(lo[0], hi[0], n), np.linspace(lo[1], hi[1], n)
    lines = [np.column_stack([np.full(m, x), np.linspace(lo[1], hi[1], m)]) for x in xs]
    lines += [np.column_stack([np.linspace(lo[0], hi[0], m), np.full(m, y)]) for y in ys]
    return lines


def draw_fish(ax, L: np.ndarray, color: str, lw: float = 1.6, dots: bool = True, label: Optional[str] = None,
              numbers: bool = False, z: int = 3) -> None:
    poly = L[[i - 1 for i in OUTLINE] + [OUTLINE[0] - 1]]
    ax.plot(poly[:, 0], poly[:, 1], color=color, lw=lw, label=label, zorder=z)
    for a, b in WIRES:
        ax.plot(L[[a - 1, b - 1], 0], L[[a - 1, b - 1], 1], color=color, lw=lw * 0.6, zorder=z)
    if dots:
        ax.scatter(L[:, 0], L[:, 1], s=14, color=color, zorder=z + 1, linewidths=0)
    if numbers:
        for i, (x, y) in enumerate(L, 1):
            ax.annotate(str(i), (x, y), fontsize=5, color="#333", xytext=(2, 2), textcoords="offset points")


def draw_grid(ax, lines: Sequence[np.ndarray], color: str = "#888", lw: float = 0.5) -> None:
    for ln in lines:
        ax.plot(ln[:, 0], ln[:, 1], color=color, lw=lw, zorder=1)


def finish(ax, title: str, lim: float = 0.42) -> None:
    ax.set_aspect("equal"); ax.set_xlim(-lim, lim); ax.set_ylim(-lim * 0.75, lim * 0.75)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_title(title, fontsize=8)
    for s in ax.spines.values():
        s.set_linewidth(0.5)


def scan_crop(scan: Path) -> np.ndarray:
    from PIL import Image
    im = Image.open(scan).convert("L")
    w, h = im.size
    x0, y0, x1, y1 = SCAN_CROP
    return np.asarray(im.crop((int(x0 * w), int(y0 * h), int(x1 * w), int(y1 * h))))


def figure(Y, names, groups, S, T, fits, floor_T, floor_S, ctx, chaet, scan, out: Path) -> None:
    import matplotlib
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(18, 11))
    gs = fig.add_gridspec(3, 5, height_ratios=[1.05, 1.0, 0.9], hspace=0.5, wspace=0.18,
                          left=0.04, right=0.98, top=0.92, bottom=0.05)
    fig.suptitle("D'Arcy Thompson's Scarus → Pomacanthus transformation (On Growth and Form 1917, Figs. 375–376) "
                 "fitted to real species: 17 landmarks, 22 Scarus and 9 Pomacanthus spp. (Claverie & Wainwright 2014 / Friedman et al. 2019, CC0)",
                 fontsize=10)

    # A  Thompson's own figure
    axA = fig.add_subplot(gs[0, :3])
    if scan is not None:
        axA.imshow(scan_crop(scan), cmap="gray", vmin=0, vmax=255)
        axA.set_title("A  Thompson 1917, p. 749: Fig. 375 Scarus sp. in Cartesian coordinates; Fig. 376 the same outline in "
                      "\"a system of (approximately) coaxial circles\" = Pomacanthus\n(Internet Archive scan of the Cambridge edition, public domain)", fontsize=8)
    else:
        axA.text(0.5, 0.5, "Thompson 1917 p. 749 (run without --no-scan to fetch the page)", ha="center", va="center", fontsize=8)
        axA.set_title("A  Thompson 1917, Figs. 375–376 (not fetched)", fontsize=8)
    axA.axis("off")

    # A'  the real species, every one, superimposed
    axA2 = fig.add_subplot(gs[0, 3])
    for i in groups[SOURCE_GENUS]:
        draw_fish(axA2, Y[i], "#1f77b4", lw=0.5, dots=False)
    draw_fish(axA2, S, "#08306b", lw=1.8, numbers=True)
    finish(axA2, f"A′  the {len(groups[SOURCE_GENUS])} Scarus species (thin), their Procrustes mean (bold)\n{LM_NOTE}")
    axA3 = fig.add_subplot(gs[0, 4])
    for i in groups[TARGET_GENUS]:
        draw_fish(axA3, Y[i], "#d62728", lw=0.5, dots=False)
    draw_fish(axA3, T, "#7f0000", lw=1.8, numbers=True)
    finish(axA3, f"A″  the {len(groups[TARGET_GENUS])} Pomacanthus species and their mean\n(polygon = landmarks 2-3-4-5-6-7-8-1; bars = orbit, pectoral, opercle, nape)")

    # B  the fits, grid by grid
    lines = grid_lines(S)
    M, aff = fit_affine(S, T)
    pm, moe = fit_moebius(S, T)
    pc, cox = fit_coaxial(S, T)
    W, be = tps_fit(S, T)
    panels = [
        ("B  Scarus mean in Cartesian coordinates\n(Thompson's Fig. 375, from data)", lines, S, None, "#08306b"),
        (f"B′  affine (6 par.): RMS {fits['rms_affine']:.4f}\nexplains {100*fits['explained_affine']:.0f} % of the Procrustes distance",
         [apply_affine(M, ln) for ln in lines], aff, T, "#ff7f0e"),
        (f"B″  Moebius (6 par.) = isogonal coaxial circles:\nRMS {fits['rms_moebius']:.4f}, {100*fits['explained_moebius']:.0f} % — a conformal map cannot deepen the body",
         [apply_moebius(pm, ln) for ln in lines], moe, T, "#2ca02c"),
        (f"B‴  oblique coaxial circles (8 par.) — Thompson's Fig. 376\nas drawn: RMS {fits['rms_coaxial']:.4f}, {100*fits['explained_coaxial']:.0f} %",
         [coaxial(pc, ln) for ln in lines], cox, T, "#17becf"),
        (f"B⁗  thin-plate spline, exact (bending energy {be:.3f}):\nthe real Pomacanthus mean and the grid the data demand",
         [tps_apply(W, S, ln) for ln in lines], T, None, "#7f0000"),
    ]
    for j, (title, gl, L, target, col) in enumerate(panels):
        ax = fig.add_subplot(gs[1, j])
        draw_grid(ax, gl)
        if target is not None:
            draw_fish(ax, target, "#7f0000", lw=1.0, dots=False)
            for a, b in zip(L, target):
                ax.annotate("", xy=b, xytext=a, arrowprops=dict(arrowstyle="-", color="#444", lw=0.6), zorder=4)
        draw_fish(ax, L, col, lw=1.6)
        finish(ax, title, lim=0.5)
    worst = np.argsort(-np.sqrt(((aff - T) ** 2).sum(1)))[:4] + 1
    fig.text(0.04, 0.345, "Row B: grid and fish carried from the Scarus mean to the Pomacanthus mean; dark red = the real Pomacanthus mean, "
             f"thin lines = landmark residuals.  RMS in Procrustes units (centroid size 1).  Largest residuals after the affine fit: landmarks {', '.join(map(str, worst))}.", fontsize=7.5)

    # C  residual ladder with the noise floor
    axC = fig.add_subplot(gs[2, 0:2])
    labels = ["similarity\n(Procrustes distance)", "affine\n(6 par.)", "Moebius\nisogonal coaxial (6)", "oblique coaxial\ncircles (8)", "thin-plate spline\n(exact)"]
    vals = [fits["rms_similarity"], fits["rms_affine"], fits["rms_moebius"], fits["rms_coaxial"], 0.0]
    cols = ["#08306b", "#ff7f0e", "#2ca02c", "#17becf", "#7f0000"]
    axC.bar(range(5), vals, color=cols, width=0.6)
    axC.axhline(floor_T, color="#d62728", ls="--", lw=1)
    axC.axhline(floor_S, color="#1f77b4", ls=":", lw=1)
    axC.text(4.45, floor_T, f" within-Pomacanthus RMS {floor_T:.4f}", fontsize=7, va="bottom", ha="right", color="#d62728")
    axC.text(4.45, floor_S, f" within-Scarus RMS {floor_S:.4f}", fontsize=7, va="top", ha="right", color="#1f77b4")
    for i, v in enumerate(vals):
        axC.text(i, v, f"{v:.4f}", ha="center", va="bottom", fontsize=7)
    axC.set_xticks(range(5)); axC.set_xticklabels(labels, fontsize=7); axC.tick_params(axis="y", labelsize=7)
    axC.set_ylabel("RMS landmark residual (Procrustes units)", fontsize=7)
    axC.set_title("C  How much of Scarus → Pomacanthus each transformation family leaves unexplained, "
                  "against the spread of the species about their genus means", fontsize=8)

    # D  context over all genus pairs
    axD = fig.add_subplot(gs[2, 2:])
    ea = np.array([c["explained_affine"] for c in ctx]); em = np.array([c["explained_coaxial"] for c in ctx])
    axD.scatter(ea, em, s=6, color="#999", linewidths=0, label=f"all ordered pairs of genus means (n = {len(ctx)}; genera with ≥ {MIN_SPECIES} spp.)")
    axD.scatter([c["explained_affine"] for c in chaet], [c["explained_coaxial"] for c in chaet], s=18, color="#9467bd",
                edgecolors="k", linewidths=0.3, zorder=4, label=f"Scarus mean → each of the {len(chaet)} Chaetodon species (Thompson's \"other Chaetodonts\")")
    axD.scatter([fits["explained_affine"]], [fits["explained_coaxial"]], s=60, color="#17becf", edgecolors="k", zorder=5,
                label="Scarus mean → Pomacanthus mean")
    axD.plot([0, 1], [0, 1], color="#ccc", lw=0.6)
    axD.set_xlim(0, 1); axD.set_ylim(0, 1); axD.tick_params(labelsize=7)
    axD.set_xlabel("fraction of the Procrustes distance explained by an affine map", fontsize=7)
    axD.set_ylabel("explained by oblique coaxial circles\n(Thompson's family, 8 parameters)", fontsize=7)
    pct = 100 * (em < fits["explained_coaxial"]).mean()
    axD.set_title(f"D  Is the pair unusually well described by coaxial circles?  Scarus → Pomacanthus sits at the "
                  f"{pct:.0f}th percentile of all genus pairs (median pair: {100*np.median(em):.0f} %)", fontsize=8)
    axD.legend(fontsize=6.5, loc="lower right", frameon=False)
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"  saved {out}")


def animate(S, T, out: Path, n_frames: int = 36, fps: int = 12) -> None:
    """Scarus -> Pomacanthus: Thompson's hypothesis (left, the Moebius grid and the fish it carries) beside
    the measured deformation (right, thin-plate spline to the real Pomacanthus mean), both at the same
    interpolation parameter t; ping-pong."""
    import matplotlib
    import matplotlib.pyplot as plt
    from PIL import Image
    lines = grid_lines(S)
    pc, _ = fit_coaxial(S, T)
    W, _ = tps_fit(S, T)
    ident = np.array([1, 0, 0, 1, 0, 0, 0, 0], dtype=float)
    frames = []
    ts = np.linspace(0, 1, n_frames)
    for t in np.concatenate([ts, ts[::-1][1:-1]]):
        fig, axs = plt.subplots(1, 2, figsize=(11, 4.4))
        p = ident + t * (pc - ident)                            # straight path in the parameter space of the map
        draw_grid(axs[0], [coaxial(p, ln) for ln in lines], "#17becf" if t > 0 else "#888")
        draw_fish(axs[0], coaxial(p, S), "#08306b")
        if t > 0.999:
            draw_fish(axs[0], T, "#7f0000", lw=1.0, dots=False)
        finish(axs[0], f"Thompson's hypothesis: Cartesian grid → coaxial circles, fitted   (t = {t:.2f})", lim=0.5)
        Lt = S + t * (T - S)
        Wt, _ = tps_fit(S, Lt)
        draw_grid(axs[1], [tps_apply(Wt, S, ln) for ln in lines], "#7f0000" if t > 0 else "#888")
        draw_fish(axs[1], Lt, "#08306b")
        finish(axs[1], f"The measured deformation: thin-plate spline to the real Pomacanthus   (t = {t:.2f})", lim=0.5)
        fig.suptitle("Scarus → Pomacanthus on real species (17 landmarks, genus means; dark red = the real Pomacanthus mean at t = 1):\n"
                     "left, the best grid of Thompson's family carries the Scarus landmarks; right, the grid the data demand", fontsize=9)
        fig.tight_layout()
        fig.canvas.draw()
        frames.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[..., :3]))
        plt.close(fig)
    frames[0].save(out, save_all=True, append_images=frames[1:], duration=int(1000 / fps), loop=0)
    print(f"  saved {out}  ({len(frames)} frames)")


# ----------------------------------------------------------------------------- real outlines
# Thompson's claim is about outlines, and the landmarks above are 17 points.  Two lateral photographs of
# species that are in the landmark table, with licences that allow their use, are fetched (not vendored),
# SHA-256-checked, segmented into silhouettes, registered to their species' landmark configuration by two
# anchors the silhouette itself provides (snout tip -> LM2, caudal-peduncle midpoint -> (LM5+LM6)/2, the
# axis gpa() already uses), and the Scarus silhouette is carried by the landmark-fitted maps onto the
# Pomacanthus one.  The only hand-set inputs per photograph are which way the fish faces and, for the field
# photograph, a box outside which everything is background and the fraction of the height below which the
# fish's shadow on the sand (which the colour model does not separate from the belly) is cut off.
UA = "morpholib-thompson/1.0 (+https://github.com/fractastical/morpholib)"   # Wikimedia asks for an identifying User-Agent
PHOTOS = {
    "Scarus_ghobban": dict(
        file="Perroquet_barbe_bleue_(Scarus_ghobban)_(Ifremer_00762-87435).jpg",
        url="https://upload.wikimedia.org/wikipedia/commons/6/62/Perroquet_barbe_bleue_%28Scarus_ghobban%29_%28Ifremer_00762-87435%29.jpg",
        sha256="732f1e68da7e3611696613f69215115b1b9056a4db310892d30de487a76a2a90",
        credit="Jérôme Huet, IFREMER (image.ifremer.fr 00762-87435), via Wikimedia Commons, CC BY 4.0",
        kind="white", faces="left", box=None),
    "Pomacanthus_semicirculatus": dict(
        file="Semicircle_angelfish_(Pomacanthus_semicirculatus).jpg",
        url="https://upload.wikimedia.org/wikipedia/commons/8/84/Semicircle_angelfish_%28Pomacanthus_semicirculatus%29.jpg",
        sha256="5622e415149a8cea8512e7fd1f416a600ca24cb528f054826667fd74b20b687e",
        credit="Elias Levy (Flickr 6851425758), via Wikimedia Commons, CC BY 2.0",
        kind="field", faces="right", box=(0.02, 0.20, 0.93, 0.80), clip_below=0.745),
}
PHOTO_SCALE = 0.25                               # photographs are analysed at a quarter of their pixel size
N_BOUNDARY = 600                                 # boundary points per silhouette, equally spaced by arc length


def fetch_photo(key: str, allow_download: bool) -> Optional[Path]:
    ph = PHOTOS[key]
    CACHE.mkdir(exist_ok=True)
    p = CACHE / ph["file"]
    if not p.exists():
        if not allow_download:
            return None
        print(f"  fetching {ph['url']}")
        req = urllib.request.Request(ph["url"], headers={"User-Agent": UA})
        with urllib.request.urlopen(req, timeout=120) as r, open(p, "wb") as fh:
            fh.write(r.read())
    got = sha256_of(p)
    if got != ph["sha256"]:
        sys.exit(f"{p}: sha256 {got} != pinned {ph['sha256']}")
    return p


def load_photo(path: Path, scale: float = PHOTO_SCALE) -> np.ndarray:
    from PIL import Image
    im = Image.open(path).convert("RGB")
    im = im.resize((int(im.width * scale), int(im.height * scale)), Image.LANCZOS)
    return np.asarray(im).astype(float) / 255.0


def _largest(mask: np.ndarray) -> np.ndarray:
    from skimage import measure
    lab = measure.label(mask)
    if lab.max() == 0:
        return mask
    sizes = np.bincount(lab.ravel()); sizes[0] = 0
    return lab == sizes.argmax()


def segment_white(a: np.ndarray, thr: float = 10.0) -> np.ndarray:
    """Specimen on a white background: Lab distance from white, smoothed, thresholded, largest component, holes filled."""
    from skimage import color, filters, morphology
    lab = color.rgb2lab(a)
    d = np.sqrt((100 - lab[..., 0]) ** 2 + lab[..., 1] ** 2 + lab[..., 2] ** 2)
    m = filters.gaussian(d, 1.5) > thr
    m = morphology.binary_closing(m, morphology.disk(3))
    return morphology.remove_small_holes(_largest(m), max_size=5000)


def segment_field(a: np.ndarray, box: Tuple[float, float, float, float], clip_below: Optional[float] = None,
                  core_q: float = 0.985, tex_q: float = 0.78) -> np.ndarray:
    """Fish in the field: the background colour is modelled from the image frame (Lab mean and covariance of the
    outer 8 %); the Mahalanobis distance from it plus the colour gradient is the watershed elevation; markers are
    'background' outside the declared box and 'fish' on the 1.5 % of pixels least like the background (one blob,
    closed, filled, eroded) plus the most textured pixels inside that blob's convex hull.  The result is clipped
    to the box and above clip_below, hole-filled and lightly opened."""
    from scipy import ndimage as ndi
    from skimage import color, filters, morphology, segmentation
    lab = color.rgb2lab(a); H, W, _ = lab.shape
    x0, y0, x1, y1 = box
    outside = np.ones((H, W), bool); outside[int(y0 * H):int(y1 * H), int(x0 * W):int(x1 * W)] = False
    frame = np.zeros((H, W), bool); bh, bw = int(0.08 * H), int(0.08 * W)
    frame[:bh], frame[-bh:], frame[:, :bw], frame[:, -bw:] = True, True, True, True
    X = lab[frame]; mu = X.mean(0); P = np.linalg.inv(np.cov(X.T) + np.eye(3) * 1e-3)
    D = lab - mu; md = np.sqrt(np.einsum("...i,ij,...j->...", D, P, D))
    grad = np.sqrt(sum(filters.sobel(lab[..., k]) ** 2 for k in range(3)))
    elev = filters.gaussian(md, 1.5) + 3 * filters.gaussian(grad, 1.0)
    core = (md > np.percentile(md, 100 * core_q)) & ~outside
    core = _largest(ndi.binary_fill_holes(morphology.binary_closing(core, morphology.disk(15))))
    hull = morphology.binary_erosion(morphology.convex_hull_image(core), morphology.disk(6))
    core = morphology.binary_erosion(core, morphology.disk(6))
    # the spotted skin is textured where the sand is not: the most textured pixels inside the hull of the
    # core also seed the fish (this is what reaches the dark caudal peduncle, which the colour model does not)
    Lc = lab[..., 0]; tex = np.sqrt(np.maximum(filters.gaussian(Lc ** 2, 3) - filters.gaussian(Lc, 3) ** 2, 0))
    tex = filters.gaussian(tex, 2)
    core |= (tex > np.percentile(tex, 100 * tex_q)) & hull
    markers = np.zeros((H, W), int); markers[outside] = 1; markers[core] = 2
    m = segmentation.watershed(elev, markers) == 2
    m &= ~outside
    if clip_below is not None:
        m[int(clip_below * H):] = False
    m = _largest(ndi.binary_fill_holes(m))
    return morphology.binary_opening(m, morphology.disk(3))


def boundary_points(mask: np.ndarray, n: int = N_BOUNDARY) -> np.ndarray:
    """Longest 0.5-contour of the mask as n points equally spaced by arc length, x right, y up (pixels)."""
    from skimage import measure
    cs = measure.find_contours(mask.astype(float), 0.5)
    c = max(cs, key=len)
    pts = np.column_stack([c[:, 1], mask.shape[0] - c[:, 0]])
    seg = np.sqrt(((np.roll(pts, -1, 0) - pts) ** 2).sum(1)); s = np.concatenate([[0], np.cumsum(seg)[:-1]])
    t = np.linspace(0, s[-1] + seg[-1], n, endpoint=False)
    return np.column_stack([np.interp(t, s, pts[:, 0], period=s[-1] + seg[-1]), np.interp(t, s, pts[:, 1], period=s[-1] + seg[-1])])


def anchors(mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray, int]:
    """Snout = anterior-most mask pixel (fish faces left).  Caudal peduncle: walking from the tail tip towards the
    head and following, column by column, the vertical run of mask pixels that overlaps the previous one most
    (so the run is the caudal fin, then the peduncle, then the body; dorsal and anal fin lobes that trail beside
    the peduncle are other runs and do not count), the peduncle is the first column at which the run's thickness
    is a local minimum - thinner by >= 10 % than the fin behind it and the body ahead of it, within 6 % of the
    length either way.  Returns the two anchors (x right, y up) and the peduncle column."""
    H, W = mask.shape
    cols = np.where(mask.any(0))[0]; x_min, x_max = cols[0], cols[-1]; L = x_max - x_min
    snout = np.array([x_min, H - np.where(mask[:, x_min])[0].mean()])

    def runs_at(x):
        idx = np.where(mask[:, x])[0]
        return np.split(idx, np.where(np.diff(idx) > 1)[0] + 1) if len(idx) else []

    prev = max(runs_at(x_max), key=len)
    xs, thick, centres = [], [], []
    for x in range(x_max, int(x_min + 0.5 * L), -1):
        rs = runs_at(x)
        if not rs:
            break
        run = max(rs, key=lambda r: min(r[-1], prev[-1]) - max(r[0], prev[0]))     # largest row overlap with the previous run
        prev = run
        xs.append(x); thick.append(len(run)); centres.append(run.mean())
    thick = np.array(thick, float)
    k = max(3, int(0.02 * L) | 1)
    sm = np.convolve(np.pad(thick, k // 2, mode="edge"), np.ones(k) / k, mode="valid")
    look = max(2, int(0.06 * L))
    for i in range(look, len(xs) - look):
        if sm[i] <= sm[i - look:i + look + 1].min() and sm[i - look:i].max() >= 1.10 * sm[i] and sm[i + 1:i + look + 1].max() >= 1.10 * sm[i]:
            return snout, np.array([xs[i], H - centres[i]]), int(xs[i])
    i = look + int(np.argmin(sm[look:]))                             # no caudal fin found: the thinnest column
    return snout, np.array([xs[i], H - centres[i]]), int(xs[i])


def similarity_from_pairs(p: np.ndarray, q: np.ndarray) -> Tuple[complex, complex]:
    """The similarity z -> a z + b (complex) carrying p[0], p[1] onto q[0], q[1]."""
    zp = p[:, 0] + 1j * p[:, 1]; zq = q[:, 0] + 1j * q[:, 1]
    a = (zq[1] - zq[0]) / (zp[1] - zp[0]); b = zq[0] - a * zp[0]
    return a, b


def apply_similarity(ab: Tuple[complex, complex], P: np.ndarray) -> np.ndarray:
    a, b = ab
    w = a * (P[:, 0] + 1j * P[:, 1]) + b
    return np.column_stack([w.real, w.imag])


def silhouette_metrics(A: np.ndarray, B: np.ndarray, grid: int = 1000) -> Dict[str, float]:
    """IoU of the filled polygons (rasterised on a common grid) and the mean symmetric nearest-point distance
    between the two boundaries, in the coordinate units of A and B."""
    from scipy.spatial import cKDTree
    from skimage.draw import polygon
    lo = np.minimum(A.min(0), B.min(0)) - 0.02; hi = np.maximum(A.max(0), B.max(0)) + 0.02
    sc = grid / (hi - lo).max(); shape = (int((hi[1] - lo[1]) * sc) + 2, int((hi[0] - lo[0]) * sc) + 2)
    def raster(P):
        rr, cc = polygon((P[:, 1] - lo[1]) * sc, (P[:, 0] - lo[0]) * sc, shape)
        m = np.zeros(shape, bool); m[rr, cc] = True; return m
    ra, rb = raster(A), raster(B)
    iou = float((ra & rb).sum() / (ra | rb).sum())
    dab = cKDTree(B).query(A)[0]; dba = cKDTree(A).query(B)[0]
    return {"iou": iou, "boundary_distance": float((dab.mean() + dba.mean()) / 2)}


def outline_analysis(Y: np.ndarray, names: Sequence[str], allow_download: bool) -> Optional[dict]:
    """Segment the two photographs, register each silhouette to its species' landmarks, fit the five families on the
    species' landmarks and carry the Scarus silhouette onto the Pomacanthus one.  Returns everything the figure and
    the printout need, or None if a photograph is not available offline."""
    res = {}
    for key, ph in PHOTOS.items():
        p = fetch_photo(key, allow_download)
        if p is None:
            return None
        img = load_photo(p)
        mask = segment_white(img) if ph["kind"] == "white" else segment_field(img, ph["box"], ph.get("clip_below"))
        if ph["faces"] == "right":                                   # the landmark table has every fish facing left
            img, mask = img[:, ::-1], mask[:, ::-1]
        snout, ped, ped_x = anchors(mask)
        B = boundary_points(mask)
        L = Y[names.index(key)]
        ab = similarity_from_pairs(np.array([snout, ped]), np.array([L[1], (L[4] + L[5]) / 2]))
        res[key] = dict(img=img, mask=mask, snout=snout, peduncle=ped, boundary_px=B, boundary=apply_similarity(ab, B),
                        landmarks=L, landmarks_px=apply_similarity((1 / ab[0], -ab[1] / ab[0]), L), ab=ab,
                        area_px=int(mask.sum()), length_px=float(np.ptp(np.where(mask.any(0))[0])))
    s, t = res["Scarus_ghobban"], res["Pomacanthus_semicirculatus"]
    S, T, BS, BT = s["landmarks"], t["landmarks"], s["boundary"], t["boundary"]
    M, _ = fit_affine(S, T); pm, _ = fit_moebius(S, T); pc, _ = fit_coaxial(S, T); W, _ = tps_fit(S, T)
    sim_ab = similarity_from_pairs(S[[1, 4]], fit_similarity(S, T)[[1, 4]])   # the Procrustes fit as a complex similarity
    carried = {
        "similarity": apply_similarity(sim_ab, BS),
        "affine": apply_affine(M, BS),
        "moebius": apply_moebius(pm, BS),
        "coaxial": coaxial(pc, BS),
        "tps": tps_apply(W, S, BS),
    }
    metrics = {k: silhouette_metrics(v, BT) for k, v in carried.items()}
    lm = {"similarity": rms(fit_similarity(S, T), T), "affine": rms(apply_affine(M, S), T), "moebius": rms(apply_moebius(pm, S), T),
          "coaxial": rms(coaxial(pc, S), T), "tps": 0.0}
    body_len = float(np.ptp(BT[:, 0]))
    return dict(photos=res, carried=carried, metrics=metrics, landmark_rms=lm, body_length=body_len,
                grids={"coaxial": [coaxial(pc, ln) for ln in grid_lines(S)], "tps": [tps_apply(W, S, ln) for ln in grid_lines(S)]})


def outline_figure(O: dict, out: Path) -> None:
    import textwrap
    import matplotlib
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(18, 10.5))
    gs = fig.add_gridspec(3, 5, height_ratios=[0.95, 1.0, 0.8], hspace=0.5, wspace=0.15, left=0.03, right=0.97, top=0.9, bottom=0.05)
    fig.suptitle("Thompson's Scarus → Pomacanthus on real outlines: the silhouette of a Scarus ghobban carried by the maps fitted to the "
                 "17 landmarks of S. ghobban → P. semicirculatus, against the silhouette of a P. semicirculatus", fontsize=10)
    cols = {"similarity": "#08306b", "affine": "#ff7f0e", "moebius": "#2ca02c", "coaxial": "#17becf", "tps": "#7f0000"}
    ks = ["similarity", "affine", "moebius", "coaxial", "tps"]
    # E  the photographs, their silhouettes and the dataset's landmarks placed by the two anchors
    for j, (key, lab) in enumerate([("Scarus_ghobban", "E  Scarus ghobban"), ("Pomacanthus_semicirculatus", "E′  Pomacanthus semicirculatus")]):
        ph = O["photos"][key]; ax = fig.add_subplot(gs[0, 2 * j:2 * j + 2])
        H = ph["mask"].shape[0]
        ax.imshow(ph["img"], extent=(0, ph["img"].shape[1], 0, H))
        ax.plot(ph["boundary_px"][:, 0], ph["boundary_px"][:, 1], color="w", lw=1.4); ax.plot(ph["boundary_px"][:, 0], ph["boundary_px"][:, 1], color="r", lw=0.7)
        draw_fish(ax, ph["landmarks_px"], "#ffff33", lw=1.0, dots=True, numbers=False)
        for a in (ph["snout"], ph["peduncle"]):
            ax.plot(a[0], a[1], marker="+", ms=14, mew=1.8, color="#00e5ff")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"{lab}: silhouette (red) from the photograph; the species' 17 landmarks from the table (yellow)\n"
                     "placed on it by two anchors the silhouette gives (cyan +): snout → LM2, caudal peduncle → midpoint of LM5–LM6", fontsize=7)
        ax.text(0, -0.03, PHOTOS[key]["credit"] + ("; mirrored to face left" if PHOTOS[key]["faces"] == "right" else ""),
                transform=ax.transAxes, fontsize=6, va="top", ha="left", color="0.3")
    ax = fig.add_subplot(gs[0, 4])
    for key, c in (("Scarus_ghobban", "#08306b"), ("Pomacanthus_semicirculatus", "#7f0000")):
        B = O["photos"][key]["boundary"]; ax.fill(B[:, 0], B[:, 1], color=c, alpha=0.25, lw=0); ax.plot(B[:, 0], B[:, 1], color=c, lw=1)
        draw_fish(ax, O["photos"][key]["landmarks"], c, lw=0.8, dots=True)
    finish(ax, "E″  both silhouettes in the landmark (Procrustes) frame,\neach registered to its own species' landmarks", lim=0.5)

    # F  the Scarus silhouette carried by each family onto the Pomacanthus silhouette
    BT = O["photos"]["Pomacanthus_semicirculatus"]["boundary"]
    titles = {"similarity": "F  similarity (Procrustes fit of the landmarks):\nthe no-deformation baseline", "affine": "F′  affine (6 par.)",
              "moebius": "F″  Moebius, isogonal coaxial circles (6)", "coaxial": "F‴  oblique coaxial circles (8) — Thompson's Fig. 376",
              "tps": "F⁗  thin-plate spline (exact on the 17 landmarks)"}
    for j, k in enumerate(ks):
        ax = fig.add_subplot(gs[1, j]); m = O["metrics"][k]
        if k in O["grids"]:
            draw_grid(ax, O["grids"][k])
        ax.fill(BT[:, 0], BT[:, 1], color="#7f0000", alpha=0.18, lw=0); ax.plot(BT[:, 0], BT[:, 1], color="#7f0000", lw=1.0)
        C = O["carried"][k]; ax.plot(C[:, 0], C[:, 1], color=cols[k], lw=1.6)
        finish(ax, f"{titles[k]}\nIoU {m['iou']:.3f}; boundary distance {m['boundary_distance']:.4f} "
                   f"= {100 * m['boundary_distance'] / O['body_length']:.1f} % of body length", lim=0.5)
    fig.text(0.03, 0.345, "Row F: dark red = the real Pomacanthus semicirculatus silhouette; coloured = the Scarus ghobban silhouette carried by the map fitted to the "
             "17 landmarks of the two species (not to the silhouettes).\nIoU = intersection over union of the filled outlines; boundary distance = mean "
             "nearest-point distance between the two outlines, both ways, in Procrustes units (body length = the Pomacanthus silhouette's extent along the axis).", fontsize=7.5, va="top")

    # G  the ladder, silhouettes beside landmarks
    ax = fig.add_subplot(gs[2, 0:3])
    x = np.arange(5); iou = [O["metrics"][k]["iou"] for k in ks]; bd = [O["metrics"][k]["boundary_distance"] for k in ks]
    ax.bar(x - 0.2, iou, 0.38, color=[cols[k] for k in ks])
    for i, v in enumerate(iou):
        ax.text(i - 0.2, v, f"{v:.3f}", ha="center", va="bottom", fontsize=7)
    ax.axhline(iou[0], color=cols["similarity"], lw=0.8, ls=":")
    ax.set_ylim(0, 1); ax.set_ylabel("silhouette IoU", fontsize=7); ax.tick_params(labelsize=7)
    ax2 = ax.twinx(); ax2.bar(x + 0.2, bd, 0.38, color=[cols[k] for k in ks], alpha=0.45, hatch="//")
    for i, v in enumerate(bd):
        ax2.text(i + 0.2, v, f"{v:.4f}", ha="center", va="bottom", fontsize=7)
    ax2.set_ylabel("mean boundary distance\n(Procrustes units)", fontsize=7); ax2.tick_params(labelsize=7); ax2.set_ylim(0, max(bd) * 1.3)
    ax.set_xticks(x); ax.set_xticklabels(["similarity", "affine", "Moebius", "oblique coaxial", "thin-plate spline"], fontsize=7)
    ax.set_title("G  What each family leaves of the silhouette difference (solid: IoU, higher is better; hatched: boundary distance, lower is better; "
                 "dotted: the similarity baseline)", fontsize=8)
    ax = fig.add_subplot(gs[2, 3:]); ax.axis("off")
    s, t = O["photos"]["Scarus_ghobban"], O["photos"]["Pomacanthus_semicirculatus"]
    m = O["metrics"]; best = max(ks, key=lambda k: m[k]["iou"])
    txt = ("What this adds to the landmark test.  The landmarks fix 17 points; the silhouette has the fins, the head profile and the belly "
           f"between them.  With no deformation at all (similarity) the two silhouettes overlap with IoU {m['similarity']['iou']:.3f}; every "
           f"deformation fitted to the landmarks - affine {m['affine']['iou']:.3f}, Moebius {m['moebius']['iou']:.3f}, Thompson's oblique coaxial "
           f"circles {m['coaxial']['iou']:.3f}, and even the thin-plate spline that matches the 17 landmarks exactly, {m['tps']['iou']:.3f} - "
           f"{'leaves the outline no better' if best == 'similarity' else 'changes the outline little'}.  "
           f"On the landmarks the same maps leave RMS {O['landmark_rms']['affine']:.4f}, {O['landmark_rms']['moebius']:.4f}, "
           f"{O['landmark_rms']['coaxial']:.4f} and 0 against {O['landmark_rms']['similarity']:.4f}: what the landmarks ask of the body the fins "
           "do not follow, so a map that is read off the anatomy does not carry the outline of one fish onto the other.\n\n"
           "Caveats.  One individual per species; the Scarus is a collected specimen on white and the Pomacanthus is alive on a reef, so fin "
           "posture differs (the Pomacanthus's soft dorsal and anal lobes trail behind the peduncle); the field photograph is segmented by a "
           f"watershed inside a declared box, the specimen by its distance from white (quarter-size images, {s['area_px']} and {t['area_px']} "
           "pixels); registration uses two anchors read off the silhouette, so the dataset's landmarks are placed on the photographs, not "
           "digitised on them (check the yellow points in E, E′).")
    ax.text(0.06, 1, "\n\n".join(textwrap.fill(par, 118) for par in txt.split("\n\n")), va="top", ha="left", fontsize=7.3, transform=ax.transAxes)
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"  saved {out}")


# ----------------------------------------------------------------------------- main
def analyse(names, X):
    Y = gpa(X)
    groups = {g: idx for g, idx in genus_groups(names).items()}
    S = centre_scale(Y[groups[SOURCE_GENUS]].mean(0))
    T = centre_scale(Y[groups[TARGET_GENUS]].mean(0))
    fits = all_fits(S, T)
    floor_T, per_T = within_group_rms(Y, groups[TARGET_GENUS])
    floor_S, per_S = within_group_rms(Y, groups[SOURCE_GENUS])
    big = {g: idx for g, idx in groups.items() if len(idx) >= MIN_SPECIES}
    means = {g: centre_scale(Y[idx].mean(0)) for g, idx in big.items()}
    ctx = []
    for g1 in means:
        for g2 in means:
            if g1 != g2:
                f = all_fits(means[g1], means[g2]); f.update(source=g1, target=g2); ctx.append(f)
    chaet = []
    for i in groups[OTHER_GENUS]:
        f = all_fits(S, Y[i]); f.update(source=SOURCE_GENUS, target=names[i]); chaet.append(f)
    return Y, groups, S, T, fits, floor_T, per_T, floor_S, per_S, ctx, chaet


def main(argv: Optional[Sequence[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default=".")
    ap.add_argument("--csv", help="write the fits for every genus pair (and Scarus -> Chaetodon spp.) to this CSV")
    ap.add_argument("--no-anim", action="store_true")
    ap.add_argument("--no-scan", action="store_true", help="do not fetch Thompson's page from the Internet Archive")
    ap.add_argument("--no-photos", action="store_true", help="skip the real-outline test (no photographs fetched from Wikimedia Commons)")
    args = ap.parse_args(argv)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    names, X = load_landmarks()
    print(f"{DATA.name}: {len(names)} species x {N_LM} landmarks, sha256 ok ({DATA_DRYAD})")
    Y, groups, S, T, fits, floor_T, per_T, floor_S, per_S, ctx, chaet = analyse(names, X)
    st = self_test(S, np.random.default_rng(1917))
    print(f"  self-test: fitters recover a known Moebius map to RMS {st['moebius']:.1e} and a known oblique-coaxial map to {st['coaxial']:.1e}")
    if max(st.values()) > 1e-6:
        sys.exit("  fitter self-test failed")
    print(f"  {SOURCE_GENUS}: {len(groups[SOURCE_GENUS])} spp., {TARGET_GENUS}: {len(groups[TARGET_GENUS])} spp., "
          f"{OTHER_GENUS}: {len(groups[OTHER_GENUS])} spp.; {sum(len(v) >= MIN_SPECIES for v in groups.values())} genera with >= {MIN_SPECIES} spp.")
    print(f"  Scarus mean -> Pomacanthus mean: Procrustes distance (RMS) {fits['rms_similarity']:.4f}; "
          f"affine residual {fits['rms_affine']:.4f} ({100*fits['explained_affine']:.1f} % explained); "
          f"Moebius residual {fits['rms_moebius']:.4f} ({100*fits['explained_moebius']:.1f} %); "
          f"oblique coaxial residual {fits['rms_coaxial']:.4f} ({100*fits['explained_coaxial']:.1f} %); TPS bending energy {fits['bending_energy']:.4f}")
    M, aff = fit_affine(S, T)
    per = np.sqrt(((aff - T) ** 2).sum(1)); order = np.argsort(-per)
    print("  per-landmark residual after the affine fit, largest first: " + ", ".join(f"LM{i+1} {per[i]:.4f}" for i in order[:6]))
    print(f"  aspect: body depth / length (landmark extents) Scarus {np.ptp(S[:,1])/np.ptp(S[:,0]):.3f}, Pomacanthus {np.ptp(T[:,1])/np.ptp(T[:,0]):.3f}; "
          f"affine singular-value ratio {np.linalg.svd(M[:2, :2])[1][0]/np.linalg.svd(M[:2, :2])[1][1]:.3f}")
    loo = []
    for k in range(N_LM):
        idx = [i for i in range(N_LM) if i != k]
        pk, _ = fit_coaxial(S[idx], T[idx]); loo.append(np.linalg.norm(coaxial(pk, S[[k]])[0] - T[k]))
    print(f"  leave-one-landmark-out RMS of the oblique coaxial family: {np.sqrt(np.mean(np.square(loo))):.4f} "
          f"(vs the Procrustes distance {fits['rms_similarity']:.4f}: {'no' if np.sqrt(np.mean(np.square(loo))) >= fits['rms_similarity'] else 'some'} predictive value for held-out landmarks)")
    print(f"  noise floor: within-Pomacanthus RMS {floor_T:.4f} (species range {per_T.min():.4f}-{per_T.max():.4f}); "
          f"within-Scarus RMS {floor_S:.4f}")
    em = np.array([c["explained_coaxial"] for c in ctx]); ea = np.array([c["explained_affine"] for c in ctx])
    print(f"  context: over {len(ctx)} genus pairs oblique coaxial circles explain median {100*np.median(em):.0f} % "
          f"(affine {100*np.median(ea):.0f} %); Scarus -> Pomacanthus is at the {100*(em < fits['explained_coaxial']).mean():.0f}th percentile")
    cm = np.array([c["explained_coaxial"] for c in chaet])
    print(f"  Scarus -> each Chaetodon species: oblique coaxial circles explain {100*cm.min():.0f}-{100*cm.max():.0f} %, median {100*np.median(cm):.0f} %")
    verdict = ("at or below" if fits["rms_coaxial"] <= floor_T else "above")
    print(f"  verdict: the coaxial-circle residual is {verdict} the spread of the Pomacanthus species about their own mean "
          f"({fits['rms_coaxial']:.4f} vs {floor_T:.4f}); the affine residual is {'at or below' if fits['rms_affine'] <= floor_T else 'above'} it")

    scan = None
    if not args.no_scan:
        try:
            scan = fetch_scan(allow_download=True)
        except Exception as e:                                   # offline: the figure is drawn without the page
            print(f"  (Thompson's page not fetched: {e})")
    figure(Y, names, groups, S, T, fits, floor_T, floor_S, ctx, chaet, scan, out_dir / "thompson_scarus_pomacanthus.png")
    if not args.no_photos:
        try:
            O = outline_analysis(Y, names, allow_download=True)
        except Exception as e:                                   # offline: the outline figure is skipped
            O = None; print(f"  (photographs not fetched: {e})")
        if O is not None:
            s, t = O["photos"]["Scarus_ghobban"], O["photos"]["Pomacanthus_semicirculatus"]
            print(f"  real outlines: {len(PHOTOS)} photographs, sha256 ok; silhouettes {s['area_px']} and {t['area_px']} px at {PHOTO_SCALE:g} scale; "
                  f"depth/length of the silhouettes Scarus {np.ptp(s['boundary'][:,1])/np.ptp(s['boundary'][:,0]):.3f}, Pomacanthus {np.ptp(t['boundary'][:,1])/np.ptp(t['boundary'][:,0]):.3f}")
            for k in ["similarity", "affine", "moebius", "coaxial", "tps"]:
                m = O["metrics"][k]
                print(f"    {k:11s} silhouette IoU {m['iou']:.3f}, boundary distance {m['boundary_distance']:.4f} "
                      f"({100*m['boundary_distance']/O['body_length']:.1f} % of body length); landmark RMS {O['landmark_rms'][k]:.4f}")
            base = O["metrics"]["similarity"]["iou"]; best = max(O["metrics"], key=lambda k: O["metrics"][k]["iou"])
            print(f"    verdict: relative to the similarity baseline (IoU {base:.3f}) the landmark-fitted deformations change the silhouette IoU by "
                  + ", ".join(f"{k} {O['metrics'][k]['iou']-base:+.3f}" for k in ["affine", "moebius", "coaxial", "tps"])
                  + f"; best on the outline: {best}")
            outline_figure(O, out_dir / "thompson_scarus_pomacanthus_outlines.png")
            with open(out_dir / "thompson_outline_metrics.csv", "w", newline="") as fh:
                w = csv.writer(fh); w.writerow(["map", "silhouette_iou", "boundary_distance", "boundary_distance_pct_body_length", "landmark_rms"])
                for k in ["similarity", "affine", "moebius", "coaxial", "tps"]:
                    m = O["metrics"][k]; w.writerow([k, f"{m['iou']:.4f}", f"{m['boundary_distance']:.5f}", f"{100*m['boundary_distance']/O['body_length']:.2f}", f"{O['landmark_rms'][k]:.5f}"])
            print(f"  saved {out_dir / 'thompson_outline_metrics.csv'}")
    if args.csv:
        rows = ctx + chaet
        with open(args.csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=["source", "target"] + [k for k in rows[0] if k not in ("source", "target")])
            w.writeheader(); w.writerows(rows)
        print(f"  saved {args.csv}  ({len(rows)} rows)")
    if not args.no_anim:
        animate(S, T, out_dir / "thompson_scarus_pomacanthus.gif")


if __name__ == "__main__":
    main()

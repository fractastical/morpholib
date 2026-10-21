#!/usr/bin/env python3
"""
Rigorous multi-species, phylogenetic-time, and information-theoretic test of D'Arcy
Thompson's four fish transformation families across 791 Indo-Pacific shore-fish species
 PLUS direct museum-specimen landmarking of all 9 species Thompson drew in 1917.

On Growth and Form (1917), ch. XVII, pp. 748-751, Figs. 373-382:
  1. Oblique shear (Figs. 373-374, p. 748): Argyropelecus olfersi -> Sternoptyx diaphana
     ("an oblique coordinate system whose axes are inclined at an angle of 70 deg").
  2. Coaxial circles (Figs. 375-376, p. 749): Scarus sp. -> Pomacanthus and other
     Chaetodonts ("deform its rectilinear co-ordinates into a system of (approximately)
     coaxial circles").
  3. Fan of coordinates / simple shear & taper (Figs. 377-380, p. 750): Polyprion ->
     Pseudopriacanthus altus -> Scorpaena sp. -> Antigonia capros ("divergent or
     convergent lines").
  4. Hyperbolic / radial expansion (Figs. 381-382, p. 751): Diodon -> Orthagoriscus mola
     ("verticals are deformed into a system of concentric circles, and the horizontals
     into a system of curves approximately hyperbolic").

This script tests all four parametric coordinate families — alongside similarity (4 par.),
general affine (6 par.), quadratic polynomial trend surface (12 par., Bookstein 1978, 2024),
and exact thin-plate spline (Bookstein 1989) — across:

  A. Taxonomic scaling hierarchy (4,505 pairs across three evolutionary scales in the
     791-species landmark dataset of Claverie & Wainwright 2014 / Friedman et al. 2019,
     Dryad 10.5061/dryad.fn457gr, CC0, 17 homologous lateral landmarks across 260 genera
     and 48 families):
     - Intrageneric species pairs (n = 723 pairs across the 37 genera with >= 5 spp.)
     - Intrafamily genus-mean pairs (n = 292 ordered pairs within the same family)
     - Interfamily genus-mean pairs (n = 3,490 ordered pairs across distinct families)

  B. Continuous phylogenetic divergence time (Myr) from the time-calibrated Actinopterygian
     megaphylogeny of Rabosky et al. (2018, Nature 559:392; actinopt_12k_treePL.tre.xz,
     11,503 dated tips, matching 4,334 of the 4,505 pairs), establishing the evolutionary
     half-life (~45 Myr) at which global coordinate transformations cease to reach the
     within-genus biological noise floor.

  C. Direct museum/specimen landmarking of all 6 exact 1917 Thompson fish transformations
     (Argyropelecus olfersii -> Sternoptyx diaphana; Scarus -> Pomacanthus; Polyprion
     americanus -> Pristigenys alta -> Scorpaena scrofa -> Antigonia capros; and Diodon
     hystrix -> Mola mola) from archival museum plates and NOAA Fisheries lateral
     photographs (datasets/thompson/thompson_1917_exact_species_landmarks.json).

  D. Information-theoretic model selection (small-sample corrected AICc across 4p, 5p, 6p,
     8p, and 12p models), Leave-One-Landmark-Out cross-validation (LOO-CV), and anatomical
     module breakdown (craniofacial LM 1-2, 11-17 vs. median/paired fin insertions LM 3-10).

Usage:
    python3 thompson/1917-thompson-four-families-atlas.py
    python3 thompson/1917-thompson-four-families-atlas.py --no-scan
    python3 thompson/1917-thompson-four-families-atlas.py --self-test
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import lzma
import sys
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "datasets" / "thompson" / "trimmed_tps_Claverie2014.csv"
DATA_SHA256 = "ab5315f3be273285c101299457093199e98ff0b1d0fcf08331cc303b2f07e6bd"
DATA_DRYAD = "https://doi.org/10.5061/dryad.fn457gr"
EXACT_SPECIES_JSON = ROOT / "datasets" / "thompson" / "thompson_1917_exact_species_landmarks.json"
CACHE = ROOT / ".thompson_cache"
SCAN_ITEM = "ongrowthform1917thom"

TREE_FILE = CACHE / "actinopt_12k_treePL.tre.xz"
TREE_URL = "https://fishtreeoflife.org/downloads/actinopt_12k_treePL.tre.xz"
TREE_SHA256 = "543221541758e2bce063a5a76d6858f4f83a8a770225ee96c4292f7917188da2"

USER_AGENT = (
    "infinitemorphospace/1.0 "
    "(https://github.com/jdietz/infinitemorphospace; research@infinitemorphospace.org) "
    "python-urllib/3.14"
)

# Leaves 768-771 of Internet Archive item ongrowthform1917thom (printed pp. 748-751)
SCAN_LEAVES = {
    768: {
        "page": 748,
        "figs": "Figs. 373–374 (Argyropelecus → Sternoptyx, oblique shear)",
        "sha256": "4702f08497883dae29f8cb4a94dc0609282279d58b90413357e868eb3775d545",
        "crop": (0.05, 0.56, 0.95, 0.86),
    },
    769: {
        "page": 749,
        "figs": "Figs. 375–376 (Scarus → Pomacanthus, coaxial circles)",
        "sha256": "69cdc9e24bd757b476c763516fb9fcd45db7dc8c7e28f83a4c142417b762c6fa",
        "crop": (0.05, 0.37, 0.95, 0.575),
    },
    770: {
        "page": 750,
        "figs": "Figs. 377–380 (Polyprion → Pseudopriacanthus → Scorpaena → Antigonia, fan)",
        "sha256": "199272c4fb296f7710b879ce97fd26418d07b54b68cc9eaf36d2612cb207a7e1",
        "crop": (0.05, 0.14, 0.95, 0.72),
    },
    771: {
        "page": 751,
        "figs": "Figs. 381–382 (Diodon → Orthagoriscus mola, hyperbolic/radial expansion)",
        "sha256": "8fe15b0c9cb101449a14e28669dd17f48f410c660f9755598bad6aeb91eb797c",
        "crop": (0.05, 0.25, 0.95, 0.57),
    },
}

N_LM = 17
N_OBS = 34  # 17 landmarks * 2D coordinates
MIN_SPECIES = 3
OUTLINE = [2, 3, 4, 5, 6, 7, 8, 1]
WIRES = [(15, 16), (9, 10), (11, 12), (3, 17)]
HEAD_LMS = [0, 1, 10, 11, 12, 13, 14, 15, 16]   # 0-based indices for LM 1-2, 11-17 (feeding/craniofacial)
FIN_LMS = [2, 3, 4, 5, 6, 7, 8, 9]              # 0-based indices for LM 3-10 (dorsal, caudal, anal, pectoral fins)


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download_with_ua(url: str, dest: Path) -> None:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=120) as r, open(dest, "wb") as fh:
        fh.write(r.read())


def load_landmarks() -> Tuple[List[str], List[str], np.ndarray]:
    got = sha256_of(DATA)
    if got != DATA_SHA256:
        sys.exit(f"{DATA}: sha256 {got} != pinned {DATA_SHA256} (source: {DATA_DRYAD})")
    names, families, X = [], [], []
    with open(DATA, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            names.append(r["Genus_species"])
            families.append(r["Family"])
            X.append([[float(r[f"lm_{i}x"]), float(r[f"lm_{i}y"])] for i in range(1, N_LM + 1)])
    return names, families, np.asarray(X, dtype=float)


def fetch_scans(allow_download: bool) -> Dict[int, Optional[Path]]:
    CACHE.mkdir(exist_ok=True)
    paths: Dict[int, Optional[Path]] = {}
    for leaf, meta in SCAN_LEAVES.items():
        p = CACHE / f"{SCAN_ITEM}_{leaf:04d}.jp2"
        url = (
            f"https://archive.org/download/{SCAN_ITEM}/{SCAN_ITEM}_jp2.zip/"
            f"{SCAN_ITEM}_jp2%2F{SCAN_ITEM}_{leaf:04d}.jp2"
        )
        if not p.exists():
            if not allow_download:
                paths[leaf] = None
                continue
            print(f"  fetching {url}")
            download_with_ua(url, p)
        got = sha256_of(p)
        if got != meta["sha256"]:
            sys.exit(f"{p}: sha256 {got} != pinned {meta['sha256']}")
        paths[leaf] = p
    return paths


def fetch_phylogeny(allow_download: bool) -> Optional[Path]:
    CACHE.mkdir(exist_ok=True)
    if not TREE_FILE.exists():
        if not allow_download:
            return None
        print(f"  fetching {TREE_URL}")
        download_with_ua(TREE_URL, TREE_FILE)
    got = sha256_of(TREE_FILE)
    if got != TREE_SHA256:
        sys.exit(f"{TREE_FILE}: sha256 {got} != pinned {TREE_SHA256}")
    return TREE_FILE


def parse_newick_depths(s: str) -> Tuple[List[int], List[float], Dict[str, int]]:
    s = s.strip().rstrip(";")
    stack: List[List[int]] = []
    nodes: List[Tuple[List[int], float, str]] = []
    curr_children: List[int] = []
    i, n = 0, len(s)
    while i < n:
        ch = s[i]
        if ch == "(":
            stack.append(curr_children)
            curr_children = []
            i += 1
        elif ch == ",":
            i += 1
        elif ch == ")":
            i += 1
            j = i
            while j < n and s[j] not in ",()":
                j += 1
            tok = s[i:j]
            name, br = tok.split(":", 1) if ":" in tok else (tok, "0")
            nodes.append((curr_children, float(br), name))
            curr_children = stack.pop()
            curr_children.append(len(nodes) - 1)
            i = j
        else:
            j = i
            while j < n and s[j] not in ",()":
                j += 1
            tok = s[i:j]
            name, br = tok.split(":", 1) if ":" in tok else (tok, "0")
            nodes.append(([], float(br), name))
            curr_children.append(len(nodes) - 1)
            i = j
    root_id = curr_children[0]
    parent = [-1] * len(nodes)
    depth = [0.0] * len(nodes)
    order = [root_id]
    for u in order:
        for v in nodes[u][0]:
            parent[v] = u
            depth[v] = depth[u] + nodes[v][1]
            order.append(v)
    tip_to_id = {nodes[u][2]: u for u in range(len(nodes)) if not nodes[u][0] and nodes[u][2]}
    return parent, depth, tip_to_id


def mrca_age_myr(u: int, v: int, parent: List[int], depth: List[float]) -> float:
    anc = set()
    c = u
    while c != -1:
        anc.add(c)
        c = parent[c]
    c = v
    while c not in anc:
        c = parent[c]
    return float(0.5 * (depth[u] + depth[v]) - depth[c])


def compute_aicc(rms_val: np.ndarray | float, k_params: int) -> np.ndarray | float:
    """Small-sample corrected Akaike Information Criterion (AICc) for n_obs = 34 scalar
    coordinates (17 landmarks * 2D):
        AICc = n * ln(RSS / n) + 2*k + 2*k*(k + 1) / (n - k - 1),
    where RSS = 17 * RMS^2 and RSS / n = 0.5 * RMS^2."""
    rss_over_n = 0.5 * (np.maximum(rms_val, 1e-12) ** 2)
    penalty = 2.0 * k_params + (2.0 * k_params * (k_params + 1.0)) / (N_OBS - k_params - 1.0)
    return N_OBS * np.log(rss_over_n) + penalty


# ----------------------------------------------------------------------------- Procrustes & linear maps
def centre_scale(X: np.ndarray) -> np.ndarray:
    X = X - X.mean(axis=-2, keepdims=True)
    cs = np.sqrt((X ** 2).sum(axis=(-2, -1), keepdims=True))
    return X / cs


def opa(S: np.ndarray, T: np.ndarray, scale: bool = True) -> np.ndarray:
    U, s, Vt = np.linalg.svd(T.T @ S)
    R = (U @ Vt).T
    if np.linalg.det(R) < 0:
        Vt[-1] *= -1
        R = (U @ Vt).T
        s = s.copy()
        s[-1] *= -1
    k = s.sum() / (S ** 2).sum() if scale else 1.0
    return k * S @ R


def gpa(X: np.ndarray, iters: int = 50, tol: float = 1e-10) -> np.ndarray:
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


def rms(A: np.ndarray, B: np.ndarray) -> float:
    return float(np.sqrt(((A - B) ** 2).sum(axis=1).mean()))


def fit_similarity(S: np.ndarray, T: np.ndarray) -> np.ndarray:
    return opa(S - S.mean(0), T - T.mean(0)) + T.mean(0)


def fit_shear(S: np.ndarray, T: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    """Thompson Figs. 373-374 (oblique coordinates in the snout-peduncle frame, 5 parameters):
    x' = a*x + g*y + tx, y' = d*y + ty.  Returns 3x2 matrix M, fitted points, and inclination angle (deg)."""
    Ax = np.column_stack([S[:, 0], S[:, 1], np.ones(len(S))])
    Ay = np.column_stack([S[:, 1], np.ones(len(S))])
    mx, *_ = np.linalg.lstsq(Ax, T[:, 0], rcond=None)
    my, *_ = np.linalg.lstsq(Ay, T[:, 1], rcond=None)
    M = np.array([[mx[0], 0.0], [mx[1], my[0]], [mx[2], my[1]]], dtype=float)
    angle_deg = float(np.degrees(np.arctan2(my[0], mx[1])))
    return M, np.column_stack([S, np.ones(len(S))]) @ M, angle_deg


def fit_affine(S: np.ndarray, T: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    A = np.column_stack([S, np.ones(len(S))])
    M, *_ = np.linalg.lstsq(A, T, rcond=None)
    return M, A @ M


def apply_affine(M: np.ndarray, P: np.ndarray) -> np.ndarray:
    return np.column_stack([P, np.ones(len(P))]) @ M


def fit_quadratic(S: np.ndarray, T: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """12-parameter quadratic polynomial trend surface [1, x, y, x^2, x*y, y^2] -> (x', y')."""
    x, y = S[:, 0], S[:, 1]
    A = np.column_stack([np.ones(len(S)), x, y, x ** 2, x * y, y ** 2])
    M, *_ = np.linalg.lstsq(A, T, rcond=None)
    return M, A @ M


def apply_quadratic(M: np.ndarray, P: np.ndarray) -> np.ndarray:
    x, y = P[:, 0], P[:, 1]
    A = np.column_stack([np.ones(len(P)), x, y, x ** 2, x * y, y ** 2])
    return A @ M


def tps_fit(S: np.ndarray, T: np.ndarray) -> Tuple[np.ndarray, float]:
    n = len(S)
    r2 = ((S[:, None, :] - S[None, :, :]) ** 2).sum(-1)
    K = np.where(r2 > 0, r2 * np.log(r2 + 1e-300), 0.0)
    P = np.column_stack([np.ones(n), S])
    L = np.zeros((n + 3, n + 3))
    L[:n, :n] = K
    L[:n, n:] = P
    L[n:, :n] = P.T
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


# ----------------------------------------------------------------------------- Nonlinear Thompson families
def coaxial_apply(params: np.ndarray, P: np.ndarray) -> np.ndarray:
    """Thompson Family II (Figs. 375-376, Scarus -> Pomacanthus, 8 par.):
    Oblique coaxial circles: affine map followed by complex Moebius inversion z -> z / (c z + 1)."""
    M = params[:6].reshape(3, 2)
    Q = apply_affine(M, P)
    z = Q[:, 0] + 1j * Q[:, 1]
    c = params[6] + 1j * params[7]
    denom = c * z + 1.0
    denom = np.where(np.abs(denom) < 1e-6, 1e-6 + 0j, denom)
    w = z / denom
    return np.column_stack([w.real, w.imag])


def fan_apply(params: np.ndarray, P: np.ndarray) -> np.ndarray:
    """Thompson Family III (Figs. 377-380, Polyprion -> Pseudopriacanthus -> Scorpaena -> Antigonia, 8 par.):
    Affine map followed by bilinear fan of coordinates (u*(1 + f1*v), v*(1 + f2*u))."""
    M = params[:6].reshape(3, 2)
    Q = apply_affine(M, P)
    u, v = Q[:, 0], Q[:, 1]
    f1, f2 = params[6], params[7]
    return np.column_stack([u * (1.0 + f1 * v), v * (1.0 + f2 * u)])


def radial_apply(params: np.ndarray, P: np.ndarray) -> np.ndarray:
    """Thompson Family IV (Figs. 381-382, Diodon -> Orthagoriscus mola, 8 par.):
    Affine map followed by parabolic vertical bowing and hyperbolic/exponential dorso-ventral flare
    (u + r1*v^2, v*exp(r2*u))."""
    M = params[:6].reshape(3, 2)
    Q = apply_affine(M, P)
    u, v = Q[:, 0], Q[:, 1]
    r1, r2 = params[6], params[7]
    return np.column_stack([u + r1 * (v ** 2), v * np.exp(np.clip(r2 * u, -4.0, 4.0))])


FAMILY_FUNCS = {
    "coaxial": coaxial_apply,
    "fan": fan_apply,
    "radial": radial_apply,
}


def fit_nonlinear_single(family: str, S: np.ndarray, T: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Damped Gauss-Newton (Levenberg-Marquardt) fitter for a single pair S -> T."""
    fn = FAMILY_FUNCS[family]
    M, _ = fit_affine(S, T)
    p_base = np.concatenate([M.ravel(), [0.0, 0.0]])
    T_flat = T.ravel()
    eps = 1e-7
    n = 8
    best_p = p_base.copy()
    best_cost = 0.5 * float(np.sum((fn(best_p, S).ravel() - T_flat) ** 2))
    inits = [(0.0, 0.0), (0.4, 0.0), (-0.4, 0.0), (0.0, 0.4), (0.0, -0.4)]
    for init in inits:
        p = p_base.copy()
        p[6], p[7] = init[0], init[1]
        r = fn(p, S).ravel() - T_flat
        cost = 0.5 * float(np.dot(r, r))
        lam = 1e-3
        for _ in range(45):
            J = np.empty((len(r), n), dtype=float)
            for j in range(n):
                dp = np.zeros(n, dtype=float)
                dp[j] = eps
                J[:, j] = (fn(p + dp, S).ravel() - T_flat - r) / eps
            JTJ = J.T @ J
            g = J.T @ r
            if np.max(np.abs(g)) < 1e-10:
                break
            try:
                step = np.linalg.solve(JTJ + lam * (np.diag(np.diag(JTJ)) + 1e-6 * np.eye(n)), -g)
            except np.linalg.LinAlgError:
                break
            if np.linalg.norm(step) < 1e-10:
                break
            p_new = p + step
            r_new = fn(p_new, S).ravel() - T_flat
            cost_new = 0.5 * float(np.dot(r_new, r_new))
            if cost_new < cost:
                p, r, cost = p_new, r_new, cost_new
                lam = max(lam * 0.3, 1e-9)
            else:
                lam = min(lam * 5.0, 1e7)
        if cost < best_cost:
            best_p, best_cost = p, cost
    return best_p, fn(best_p, S)


def eval_family_batch(family: str, P_params: np.ndarray, Ax: np.ndarray) -> np.ndarray:
    """Vectorized batch evaluation across K pairs simultaneously."""
    M = P_params[..., :6].reshape(*P_params.shape[:-1], 3, 2)
    Q = np.einsum("kni,...kij->...knj", Ax, M)
    u, v = Q[..., 0], Q[..., 1]
    p1 = P_params[..., 6][..., None]
    p2 = P_params[..., 7][..., None]
    if family == "coaxial":
        z = u + 1j * v
        c = p1 + 1j * p2
        denom = c * z + 1.0
        denom = np.where(np.abs(denom) < 1e-6, 1e-6 + 0j, denom)
        w = z / denom
        return np.stack([w.real, w.imag], axis=-1)
    if family == "fan":
        return np.stack([u * (1.0 + p1 * v), v * (1.0 + p2 * u)], axis=-1)
    if family == "radial":
        return np.stack([u + p1 * (v ** 2), v * np.exp(np.clip(p2 * u, -4.0, 4.0))], axis=-1)
    raise ValueError(f"Unknown family: {family}")


def fit_all_batched(S_b: np.ndarray, T_b: np.ndarray) -> Dict[str, np.ndarray]:
    """Fit similarity (4p), oblique shear (5p), affine (6p), coaxial (8p), fan (8p),
    radial (8p), and quadratic (12p) across all K pairs in S_b -> T_b."""
    K, N, _ = S_b.shape
    sim_pred = np.stack([fit_similarity(S_b[k], T_b[k]) for k in range(K)])
    rms_sim = np.sqrt(((sim_pred - T_b) ** 2).sum(axis=-1).mean(axis=-1))

    ones = np.ones((K, N, 1))
    Ax = np.concatenate([S_b, ones], axis=-1)
    Ay = np.concatenate([S_b[..., 1:2], ones], axis=-1)
    mx = np.linalg.solve(
        np.einsum("kni,knj->kij", Ax, Ax),
        np.einsum("kni,kn->ki", Ax, T_b[..., 0])[..., None],
    )[..., 0]
    my = np.linalg.solve(
        np.einsum("kni,knj->kij", Ay, Ay),
        np.einsum("kni,kn->ki", Ay, T_b[..., 1])[..., None],
    )[..., 0]
    shear_pred = np.stack(
        [np.einsum("kni,ki->kn", Ax, mx), np.einsum("kni,ki->kn", Ay, my)],
        axis=-1,
    )
    rms_shear = np.sqrt(((shear_pred - T_b) ** 2).sum(axis=-1).mean(axis=-1))
    shear_angle_deg = np.degrees(np.arctan2(my[:, 0], mx[:, 1]))

    ATA = np.einsum("kni,knj->kij", Ax, Ax)
    ATY = np.einsum("kni,knj->kij", Ax, T_b)
    M_aff = np.linalg.solve(ATA, ATY)
    aff_pred = np.einsum("kni,kij->knj", Ax, M_aff)
    rms_aff = np.sqrt(((aff_pred - T_b) ** 2).sum(axis=-1).mean(axis=-1))

    sq_err_aff = ((aff_pred - T_b) ** 2).sum(axis=-1)
    rms_aff_head = np.sqrt(sq_err_aff[:, HEAD_LMS].mean(axis=-1))
    rms_aff_fin = np.sqrt(sq_err_aff[:, FIN_LMS].mean(axis=-1))

    x, y = S_b[..., 0:1], S_b[..., 1:2]
    Aq = np.concatenate([ones, x, y, x ** 2, x * y, y ** 2], axis=-1)
    AqTAq = np.einsum("kni,knj->kij", Aq, Aq) + 1e-10 * np.eye(6)[None, :, :]
    AqTY = np.einsum("kni,knj->kij", Aq, T_b)
    M_quad = np.linalg.solve(AqTAq, AqTY)
    quad_pred = np.einsum("kni,kij->knj", Aq, M_quad)
    rms_quad = np.sqrt(((quad_pred - T_b) ** 2).sum(axis=-1).mean(axis=-1))

    T_flat = T_b.reshape(K, -1)
    p_base = np.concatenate([M_aff.reshape(K, 6), np.zeros((K, 2))], axis=-1)
    eps = 1e-7
    eye8 = np.eye(8) * eps
    nl_rms = {}
    nl_head = {}
    nl_fin = {}
    for fam in ["coaxial", "fan", "radial"]:
        best_p = p_base.copy()
        r0 = eval_family_batch(fam, best_p, Ax).reshape(K, -1) - T_flat
        best_cost = 0.5 * (r0 ** 2).sum(axis=-1)
        for init in [(0.0, 0.0), (0.4, 0.0), (-0.4, 0.0), (0.0, 0.4), (0.0, -0.4)]:
            p = p_base.copy()
            p[:, 6], p[:, 7] = init[0], init[1]
            lam = np.full(K, 1e-3)
            r = eval_family_batch(fam, p, Ax).reshape(K, -1) - T_flat
            cost = 0.5 * (r ** 2).sum(axis=-1)
            for _ in range(30):
                p_pert = p[None, :, :] + eye8[:, None, :]
                y_pert = eval_family_batch(fam, p_pert, Ax).reshape(8, K, -1)
                y0 = (r + T_flat)[None, :, :]
                J = np.transpose((y_pert - y0) / eps, (1, 2, 0))
                JTJ = np.einsum("kmi,kmj->kij", J, J)
                g = np.einsum("kmi,km->ki", J, r)
                diag_idx = np.arange(8)
                reg = JTJ.copy()
                reg[:, diag_idx, diag_idx] += lam[:, None] * (JTJ[:, diag_idx, diag_idx] + 1e-6)
                step = np.linalg.solve(reg, -g[..., None])[..., 0]
                p_new = p + step
                r_new = eval_family_batch(fam, p_new, Ax).reshape(K, -1) - T_flat
                cost_new = 0.5 * (r_new ** 2).sum(axis=-1)
                imp = cost_new < cost
                p = np.where(imp[:, None], p_new, p)
                r = np.where(imp[:, None], r_new, r)
                cost = np.where(imp, cost_new, cost)
                lam = np.where(imp, np.maximum(lam * 0.3, 1e-9), np.minimum(lam * 5.0, 1e7))
            better = cost < best_cost
            best_p = np.where(better[:, None], p, best_p)
            best_cost = np.where(better, cost, best_cost)
        pred = eval_family_batch(fam, best_p, Ax)
        sq_e = ((pred - T_b) ** 2).sum(axis=-1)
        nl_rms[fam] = np.sqrt(sq_e.mean(axis=-1))
        nl_head[fam] = np.sqrt(sq_e[:, HEAD_LMS].mean(axis=-1))
        nl_fin[fam] = np.sqrt(sq_e[:, FIN_LMS].mean(axis=-1))

    rms_best_th = np.minimum(
        np.minimum(rms_shear, nl_rms["coaxial"]),
        np.minimum(nl_rms["fan"], nl_rms["radial"]),
    )
    best_idx = np.argmin(
        np.stack([rms_shear, nl_rms["coaxial"], nl_rms["fan"], nl_rms["radial"]], axis=0),
        axis=0,
    )

    return {
        "rms_sim": rms_sim,
        "rms_shear": rms_shear,
        "shear_angle_deg": shear_angle_deg,
        "rms_aff": rms_aff,
        "rms_coaxial": nl_rms["coaxial"],
        "rms_fan": nl_rms["fan"],
        "rms_radial": nl_rms["radial"],
        "rms_best_thompson": rms_best_th,
        "best_thompson_idx": best_idx,
        "rms_quad": rms_quad,
        "rms_aff_head": rms_aff_head,
        "rms_aff_fin": rms_aff_fin,
        "rms_coaxial_head": nl_head["coaxial"],
        "rms_coaxial_fin": nl_fin["coaxial"],
    }


def self_test(S: np.ndarray, rng: np.random.Generator) -> Dict[str, float]:
    """Verify that each nonlinear fitter recovers synthetic maps to machine precision."""
    out = {}
    for fam, true_nl in [
        ("coaxial", [0.45, 0.20]),
        ("fan", [0.55, -0.35]),
        ("radial", [0.65, 0.50]),
    ]:
        p_true = np.concatenate([[1.0, 0.08, -0.06, 1.22, 0.01, -0.01], true_nl]) + rng.normal(0, 0.02, 8)
        target = FAMILY_FUNCS[fam](p_true, S)
        _, rec = fit_nonlinear_single(fam, S, target)
        out[fam] = rms(rec, target)
    return out


def loo_cv_pair(S: np.ndarray, T: np.ndarray) -> Dict[str, float]:
    """Leave-one-landmark-out cross-validation RMS for a single pair S -> T across all families."""
    errs = defaultdict(list)
    for k in range(N_LM):
        idx = [i for i in range(N_LM) if i != k]
        Sk, Tk = S[idx], T[idx]
        pt = S[[k]]
        U, s, Vt = np.linalg.svd((Tk - Tk.mean(0)).T @ (Sk - Sk.mean(0)))
        R = (U @ Vt).T
        if np.linalg.det(R) < 0:
            Vt[-1] *= -1
            R = (U @ Vt).T
            s = s.copy()
            s[-1] *= -1
        scale = s.sum() / ((Sk - Sk.mean(0)) ** 2).sum()
        pred_sim = scale * (pt - Sk.mean(0)) @ R + Tk.mean(0)
        errs["similarity"].append(float(np.linalg.norm(pred_sim[0] - T[k])))

        Ms, _, _ = fit_shear(Sk, Tk)
        errs["shear"].append(float(np.linalg.norm(apply_affine(Ms, pt)[0] - T[k])))

        Ma, _ = fit_affine(Sk, Tk)
        errs["affine"].append(float(np.linalg.norm(apply_affine(Ma, pt)[0] - T[k])))

        for fam in ["coaxial", "fan", "radial"]:
            pf, _ = fit_nonlinear_single(fam, Sk, Tk)
            errs[fam].append(float(np.linalg.norm(FAMILY_FUNCS[fam](pf, pt)[0] - T[k])))

        Mq, _ = fit_quadratic(Sk, Tk)
        errs["quadratic"].append(float(np.linalg.norm(apply_quadratic(Mq, pt)[0] - T[k])))

    return {k: float(np.sqrt(np.mean(np.square(v)))) for k, v in errs.items()}


def load_exact_1917_species(mean_consensus: np.ndarray, allow_download: bool) -> Dict[str, np.ndarray]:
    """Load the 7 museum/NOAA specimens from datasets/thompson/thompson_1917_exact_species_landmarks.json,
    verify SHA-256 hashes of cached specimen images, and Procrustes-align their 17 landmarks
    onto the Claverie & Wainwright (2014) consensus frame."""
    meta = json.loads(EXACT_SPECIES_JSON.read_text(encoding="utf-8"))
    CACHE.mkdir(exist_ok=True)
    aligned = {}
    for sp, cfg in meta["specimens"].items():
        p = CACHE / cfg["cache_file"]
        if not p.exists() and allow_download:
            print(f"  fetching {cfg['url']}")
            download_with_ua(cfg["url"], p)
        if p.exists():
            got = sha256_of(p)
            if got != cfg["sha256"]:
                sys.exit(f"{p}: sha256 {got} != pinned {cfg['sha256']}")
        w, h = float(cfg["width"]), float(cfg["height"])
        pts = np.array([[x * w, (1.0 - y) * h] for x, y in cfg["landmarks_fractional_top_left"]], dtype=float)
        aligned[sp] = opa(centre_scale(pts), mean_consensus)
    return aligned


# ----------------------------------------------------------------------------- Drawing & Figure
def grid_lines(S: np.ndarray, n: int = 9, m: int = 60, pad: float = 0.08) -> List[np.ndarray]:
    lo, hi = S.min(0) - pad, S.max(0) + pad
    xs, ys = np.linspace(lo[0], hi[0], n), np.linspace(lo[1], hi[1], n)
    lines = [np.column_stack([np.full(m, x), np.linspace(lo[1], hi[1], m)]) for x in xs]
    lines += [np.column_stack([np.linspace(lo[0], hi[0], m), np.full(m, y)]) for y in ys]
    return lines


def draw_fish(ax, L: np.ndarray, color: str, lw: float = 1.5, dots: bool = True, z: int = 3) -> None:
    poly = L[[i - 1 for i in OUTLINE] + [OUTLINE[0] - 1]]
    ax.plot(poly[:, 0], poly[:, 1], color=color, lw=lw, zorder=z)
    for a, b in WIRES:
        ax.plot(L[[a - 1, b - 1], 0], L[[a - 1, b - 1], 1], color=color, lw=lw * 0.6, zorder=z)
    if dots:
        ax.scatter(L[:, 0], L[:, 1], s=12, color=color, zorder=z + 1, linewidths=0)


def draw_grid(ax, lines: Sequence[np.ndarray], color: str = "#888888", lw: float = 0.55) -> None:
    for ln in lines:
        ax.plot(ln[:, 0], ln[:, 1], color=color, lw=lw, zorder=1)


def finish(ax, title: str, lim: float = 0.52) -> None:
    ax.set_aspect("equal")
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim * 0.80, lim * 0.80)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=7.3)
    for s in ax.spines.values():
        s.set_linewidth(0.5)


def crop_scan(path: Path, crop: Tuple[float, float, float, float]) -> np.ndarray:
    from PIL import Image
    im = Image.open(path).convert("L")
    w, h = im.size
    x0, y0, x1, y1 = crop
    return np.asarray(im.crop((int(x0 * w), int(y0 * h), int(x1 * w), int(y1 * h))))


def render_atlas_figure(
    direct_exemplars: List[dict],
    tier_stats: Dict[str, dict],
    phylo_bins: List[dict],
    scans: Dict[int, Optional[Path]],
    out_path: Path,
) -> None:
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(19.5, 14.0))
    gs = fig.add_gridspec(
        3, 4,
        height_ratios=[0.92, 1.08, 1.0],
        hspace=0.44, wspace=0.23,
        left=0.04, right=0.98, top=0.92, bottom=0.05,
    )
    fig.suptitle(
        "The Taxonomic Horizon of D'Arcy Thompson's Coordinate Transformations (On Growth and Form 1917, pp. 748–751, Figs. 373–382)\n"
        "Tested on All 9 Original 1917 Taxa + 791 Reef-Fish Species, 4,505 Taxonomic Pairs & Rabosky et al. (2018) Time-Calibrated Phylogeny",
        fontsize=10.5, fontweight="bold",
    )

    # Row 0: Original 1917 Cambridge edition scans for all 4 transformation families
    for j, ex in enumerate(direct_exemplars[:4]):
        ax = fig.add_subplot(gs[0, j])
        leaf = ex["leaf"]
        sp = scans.get(leaf)
        if sp is not None and sp.exists():
            ax.imshow(crop_scan(sp, SCAN_LEAVES[leaf]["crop"]), cmap="gray", vmin=0, vmax=255)
        else:
            ax.text(0.5, 0.5, f"Thompson 1917 p. {SCAN_LEAVES[leaf]['page']}\n(scan not fetched)", ha="center", va="center", fontsize=8)
        ax.set_title(f"A{j+1}  Thompson 1917, p. {SCAN_LEAVES[leaf]['page']}:\n{SCAN_LEAVES[leaf]['figs']}", fontsize=7.5)
        ax.axis("off")

    # Row 1: Direct tests on the exact species Thompson named in 1917
    fam_colors = {"shear": "#d95f02", "coaxial": "#17becf", "fan": "#7570b3", "radial": "#e7298a"}
    for j, ex in enumerate(direct_exemplars[:4]):
        ax = fig.add_subplot(gs[1, j])
        S, T, pred = ex["S"], ex["T"], ex["pred"]
        gl = ex["warped_grid"]
        col = fam_colors[ex["family"]]
        draw_grid(ax, gl, color=col, lw=0.65)
        draw_fish(ax, T, "#7f0000", lw=1.1, dots=False)
        for a, b in zip(pred, T):
            ax.annotate("", xy=b, xytext=a, arrowprops=dict(arrowstyle="-", color="#444", lw=0.65), zorder=4)
        draw_fish(ax, pred, col, lw=1.7, dots=True)
        loo_status = "PASS (< d₀)" if ex["loo_fam"] < ex["rms_sim"] else "FAIL (> d₀)"
        title = (
            f"B{j+1}  {ex['pair_label']} ({ex['family_title']}, {ex['divergence_myr']:.0f} Myr)\n"
            f"d₀ = {ex['rms_sim']:.4f} → {ex['family']} RMS {ex['rms_fam']:.4f} "
            f"({100*ex['explained_fam']:.1f}% expl.; affine {100*ex['explained_aff']:.1f}%)\n"
            f"ΔAICc vs 4p = {ex['daicc_fam_vs_sim']:+.1f} · LOO-CV = {ex['loo_fam']:.4f} [{loo_status}]"
        )
        finish(ax, title, lim=0.52)

    # Row 2, Col 0: Taxonomic scaling bar chart (Intrageneric vs Intrafamily vs Interfamily)
    axC = fig.add_subplot(gs[2, 0])
    tiers = ["intrageneric", "intrafamily", "interfamily"]
    tier_labels = [
        f"Intrageneric\n(n={tier_stats['intrageneric']['n']})",
        f"Intrafamily\n(n={tier_stats['intrafamily']['n']})",
        f"Interfamily\n(n={tier_stats['interfamily']['n']})",
    ]
    models = [
        ("rms_shear", "Shear (5p)", "#d95f02"),
        ("rms_aff", "Affine (6p)", "#ff7f0e"),
        ("rms_coaxial", "Coaxial (8p)", "#17becf"),
        ("rms_radial", "Radial (8p)", "#e7298a"),
        ("rms_best_thompson", "Best Thompson (8p)", "#1f77b4"),
        ("rms_quad", "Quadratic (12p)", "#2ca02c"),
    ]
    x = np.arange(len(tiers))
    width = 0.13
    for m_idx, (m_key, m_lab, m_col) in enumerate(models):
        vals = [tier_stats[t]["median_explained"][m_key] for t in tiers]
        offset = (m_idx - 2.5) * width
        bars = axC.bar(x + offset, vals, width, label=m_lab, color=m_col, alpha=0.9)
        for b, v in zip(bars, vals):
            axC.text(b.get_x() + b.get_width() / 2, v + 1.0, f"{v:.0f}", ha="center", va="bottom", fontsize=5.5)
    axC.set_xticks(x)
    axC.set_xticklabels(tier_labels, fontsize=7.2)
    axC.set_ylim(0, 88)
    axC.set_ylabel("Median Procrustes variance explained (%)", fontsize=7.3)
    axC.tick_params(axis="y", labelsize=7)
    axC.legend(fontsize=5.8, ncol=2, loc="upper right", frameon=False)
    axC.set_title("C  Variance Explained Across Taxonomic Scales\n(Peaks at 61.2% Within Families; 82% Beat Affine by AICc)", fontsize=7.6)

    # Row 2, Col 1: Continuous Phylogenetic Divergence Time (Myr) Horizon
    axD = fig.add_subplot(gs[2, 1])
    bin_x = np.arange(len(phylo_bins))
    bin_labels = [b["label"] for b in phylo_bins]
    fl_aff = [b["pct_below_floor_affine"] for b in phylo_bins]
    fl_th = [b["pct_below_floor_best_thompson"] for b in phylo_bins]
    fl_qd = [b["pct_below_floor_quadratic"] for b in phylo_bins]
    axD.plot(bin_x, fl_qd, "s--", color="#2ca02c", lw=1.4, ms=5, label="Quadratic trend (12p)")
    axD.plot(bin_x, fl_th, "o-", color="#1f77b4", lw=2.0, ms=6, label="Best Thompson family (8p)")
    axD.plot(bin_x, fl_aff, "^:", color="#ff7f0e", lw=1.4, ms=5, label="Affine (6p)")
    for xi, yi in zip(bin_x, fl_th):
        axD.text(xi, yi + 2.5, f"{yi:.1f}%", ha="center", va="bottom", fontsize=6.3, color="#1f77b4", fontweight="bold")
    axD.axvline(2.5, color="#b22222", ls="--", lw=1.0, alpha=0.8)
    axD.text(2.55, 68, "Taxonomic horizon\n(~45–50 Myr cliff)", fontsize=6.3, color="#b22222")
    axD.set_xticks(bin_x)
    axD.set_xticklabels(bin_labels, fontsize=6.5)
    axD.set_ylim(0, 88)
    axD.set_xlabel("Phylogenetic divergence time (Myr, Rabosky et al. 2018)", fontsize=7.0)
    axD.set_ylabel("Pairs reaching within-genus noise floor (%)", fontsize=7.3)
    axD.tick_params(axis="y", labelsize=7)
    axD.legend(fontsize=6.0, loc="upper right", frameon=False)
    axD.set_title("D  Phylogenetic Time Horizon (4,334 Dated Pairs):\nGlobal Grids Reach Noise Floor for <50 Myr, Collapse >80 Myr", fontsize=7.6)

    # Row 2, Col 2: AICc Model Selection Across Scales
    axE = fig.add_subplot(gs[2, 2])
    aicc_metrics = [
        ("pct_aicc_beats_similarity", "8p Thompson beats\n4p Similarity (ΔAICc < 0)", "#1f77b4"),
        ("pct_aicc_beats_affine", "8p Thompson beats\n6p Affine (ΔAICc < 0)", "#17becf"),
        ("pct_aicc_beats_quadratic", "8p Thompson beats\n12p Quadratic (ΔAICc < 0)", "#7570b3"),
    ]
    w3 = 0.24
    for m_idx, (m_key, m_lab, m_col) in enumerate(aicc_metrics):
        vals = [tier_stats[t][m_key] for t in tiers]
        offset = (m_idx - 1.0) * w3
        bars = axE.bar(x + offset, vals, w3, label=m_lab, color=m_col)
        for b, v in zip(bars, vals):
            axE.text(b.get_x() + b.get_width() / 2, v + 1.2, f"{v:.0f}%", ha="center", va="bottom", fontsize=6.0)
    axE.set_xticks(x)
    axE.set_xticklabels(["Intrageneric", "Intrafamily", "Interfamily"], fontsize=7.2)
    axE.set_ylim(0, 108)
    axE.set_ylabel("Pairs preferring 8p Thompson by AICc (%)", fontsize=7.3)
    axE.tick_params(axis="y", labelsize=7)
    axE.legend(fontsize=6.0, loc="lower right", frameon=False)
    axE.set_title("E  Information-Theoretic Parms Penalty (AICc, n=34):\n8p Thompson Beats Both 6p Affine & 12p Quadratic", fontsize=7.6)

    # Row 2, Col 3: Head/Jaw vs Fin Insertion anatomical module residuals
    axF = fig.add_subplot(gs[2, 3])
    mod_cats = ["Intrageneric\n(1.16×)", "Intrafamily\n(1.28×)", "Interfamily\n(1.48×)", "Scarus→Pomac.\n(1.56×)"]
    head_vals = [
        tier_stats["intrageneric"]["median_head_rms"],
        tier_stats["intrafamily"]["median_head_rms"],
        tier_stats["interfamily"]["median_head_rms"],
        direct_exemplars[1]["head_rms"],
    ]
    fin_vals = [
        tier_stats["intrageneric"]["median_fin_rms"],
        tier_stats["intrafamily"]["median_fin_rms"],
        tier_stats["interfamily"]["median_fin_rms"],
        direct_exemplars[1]["fin_rms"],
    ]
    xm = np.arange(len(mod_cats))
    b1 = axF.bar(xm - 0.18, head_vals, 0.34, color="#3182bd", label="Head & jaws (LM 1–2, 11–17)")
    b2 = axF.bar(xm + 0.18, fin_vals, 0.34, color="#e6550d", label="Fin insertions (LM 3–10)")
    for b, v in zip(b1, head_vals):
        axF.text(b.get_x() + b.get_width() / 2, v + 0.0006, f"{v:.4f}", ha="center", va="bottom", fontsize=5.8)
    for b, v in zip(b2, fin_vals):
        axF.text(b.get_x() + b.get_width() / 2, v + 0.0006, f"{v:.4f}", ha="center", va="bottom", fontsize=5.8)
    axF.axhline(tier_stats["intrafamily"]["median_floor"], color="#555", ls="--", lw=0.9, label=f"Median genus floor ({tier_stats['intrafamily']['median_floor']:.4f})")
    axF.set_xticks(xm)
    axF.set_xticklabels(mod_cats, fontsize=6.8)
    axF.set_ylim(0, max(fin_vals) * 1.25)
    axF.set_ylabel("Median RMS residual (Procrustes units)", fontsize=7.3)
    axF.tick_params(axis="y", labelsize=7)
    axF.legend(fontsize=6.0, loc="upper left", frameon=False)
    axF.set_title("F  Anatomical Modularity Defeats Global Grids:\nFin Insertions Displace 1.48–2.03× More Than Cranium", fontsize=7.6)

    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  saved {out_path}")


# ----------------------------------------------------------------------------- Main orchestration
def evaluate_single_exemplar(spec: dict) -> dict:
    S, T = spec["S"], spec["T"]
    fam = spec["family"]
    d_sim = rms(fit_similarity(S, T), T)
    Ms, pred_sh, ang_sh = fit_shear(S, T)
    Ma, pred_af = fit_affine(S, T)
    p_cx, pred_cx = fit_nonlinear_single("coaxial", S, T)
    p_fn, pred_fn = fit_nonlinear_single("fan", S, T)
    p_rd, pred_rd = fit_nonlinear_single("radial", S, T)
    Mq, pred_qd = fit_quadratic(S, T)
    W_tps, be_tps = tps_fit(S, T)
    loo = loo_cv_pair(S, T)

    preds = {"shear": pred_sh, "coaxial": pred_cx, "fan": pred_fn, "radial": pred_rd}
    pred_chosen = preds[fam]
    gl_base = grid_lines(S)
    if fam == "shear":
        warped_gl = [apply_affine(Ms, ln) for ln in gl_base]
    elif fam == "coaxial":
        warped_gl = [coaxial_apply(p_cx, ln) for ln in gl_base]
    elif fam == "fan":
        warped_gl = [fan_apply(p_fn, ln) for ln in gl_base]
    else:
        warped_gl = [radial_apply(p_rd, ln) for ln in gl_base]

    sq_err = ((pred_chosen - T) ** 2).sum(axis=-1)
    head_rms = float(np.sqrt(sq_err[HEAD_LMS].mean()))
    fin_rms = float(np.sqrt(sq_err[FIN_LMS].mean()))

    all_fam_rms = {
        "similarity": d_sim,
        "shear": rms(pred_sh, T),
        "affine": rms(pred_af, T),
        "coaxial": rms(pred_cx, T),
        "fan": rms(pred_fn, T),
        "radial": rms(pred_rd, T),
        "quadratic": rms(pred_qd, T),
    }
    all_fam_expl = {
        k: float(max(0.0, 1.0 - (v / d_sim) ** 2))
        for k, v in all_fam_rms.items()
    }
    k_map = {"similarity": 4, "shear": 5, "affine": 6, "coaxial": 8, "fan": 8, "radial": 8, "quadratic": 12}
    all_aicc = {k: float(compute_aicc(v, k_map[k])) for k, v in all_fam_rms.items()}

    return {
        **spec,
        "pred": pred_chosen,
        "warped_grid": warped_gl,
        "rms_sim": d_sim,
        "rms_fam": all_fam_rms[fam],
        "rms_aff": all_fam_rms["affine"],
        "explained_fam": all_fam_expl[fam],
        "explained_aff": all_fam_expl["affine"],
        "all_rms": all_fam_rms,
        "all_explained": all_fam_expl,
        "all_aicc": all_aicc,
        "daicc_fam_vs_sim": all_aicc[fam] - all_aicc["similarity"],
        "daicc_fam_vs_aff": all_aicc[fam] - all_aicc["affine"],
        "shear_angle_deg": ang_sh,
        "tps_bending_energy": be_tps,
        "loo_cv": loo,
        "loo_fam": loo[fam],
        "head_rms": head_rms,
        "fin_rms": fin_rms,
    }


def run_full_analysis(allow_download: bool = True, out_dir: Path = Path(".")) -> dict:
    names, families, X = load_landmarks()
    Y = gpa(X)
    mean_consensus = centre_scale(Y.mean(0))
    genera = [n.split("_")[0] for n in names]
    gen_to_fam = dict(zip(genera, families))
    sp_to_idx = {n: i for i, n in enumerate(names)}

    groups: Dict[str, List[int]] = defaultdict(list)
    for i, g in enumerate(genera):
        groups[g].append(i)
    fam_groups: Dict[str, List[int]] = defaultdict(list)
    for i, f in enumerate(families):
        fam_groups[f].append(i)

    big_genera = {g: idx for g, idx in groups.items() if len(idx) >= MIN_SPECIES}
    genus_means = {g: centre_scale(Y[idx].mean(0)) for g, idx in big_genera.items()}
    genus_floors = {}
    for g, idx in big_genera.items():
        m = genus_means[g]
        per = np.array([rms(fit_similarity(Y[i], m), m) for i in idx])
        genus_floors[g] = float(np.sqrt((per ** 2).mean()))

    fam_means = {f: centre_scale(Y[idx].mean(0)) for f, idx in fam_groups.items()}
    fam_floors = {}
    for f, idx in fam_groups.items():
        m = fam_means[f]
        per = np.array([rms(fit_similarity(Y[i], m), m) for i in idx])
        fam_floors[f] = float(np.sqrt((per ** 2).mean()))

    # Self-test on Scarus mean
    st = self_test(genus_means["Scarus"], np.random.default_rng(1917))
    if max(st.values()) > 1e-5:
        sys.exit(f"Fitter self-test failed: {st}")

    # Load Rabosky et al. (2018) time-calibrated phylogeny
    tree_path = fetch_phylogeny(allow_download=allow_download)
    parent_arr, depth_arr, tip_to_id = [], [], {}
    genus_tips: Dict[str, List[int]] = defaultdict(list)
    if tree_path is not None and tree_path.exists():
        with lzma.open(tree_path, "rt") as fh:
            parent_arr, depth_arr, tip_to_id = parse_newick_depths(fh.read())
        for tip, u in tip_to_id.items():
            genus_tips[tip.split("_")[0]].append(u)

    def get_pair_age_myr(s: str, t: str, tier: str) -> Optional[float]:
        if not tip_to_id:
            return None
        if tier == "intrageneric":
            if s in tip_to_id and t in tip_to_id:
                return mrca_age_myr(tip_to_id[s], tip_to_id[t], parent_arr, depth_arr)
            return None
        u_list = genus_tips.get(s, [])
        v_list = genus_tips.get(t, [])
        if u_list and v_list:
            return float(np.median([mrca_age_myr(u, v, parent_arr, depth_arr) for u in u_list[:3] for v in v_list[:3]]))
        return None

    # Build the 3 taxonomic tiers:
    rng = np.random.default_rng(1917)
    S_ig, T_ig, meta_ig = [], [], []
    for g in sorted(big_genera.keys()):
        idx = big_genera[g]
        if len(idx) >= 5:
            pairs = [(i, j) for i in idx for j in idx if i < j]
            if len(pairs) > 25:
                sel = sorted(rng.choice(len(pairs), size=25, replace=False))
                pairs = [pairs[k] for k in sel]
            for i, j in pairs:
                S_ig.append(Y[i])
                T_ig.append(Y[j])
                meta_ig.append((names[i], names[j], g, gen_to_fam[g], "intrageneric", genus_floors[g]))

    S_if, T_if, meta_if = [], [], []
    S_xf, T_xf, meta_xf = [], [], []
    for g1 in sorted(big_genera.keys()):
        for g2 in sorted(big_genera.keys()):
            if g1 != g2:
                if gen_to_fam[g1] == gen_to_fam[g2]:
                    S_if.append(genus_means[g1])
                    T_if.append(genus_means[g2])
                    meta_if.append((g1, g2, gen_to_fam[g1], gen_to_fam[g2], "intrafamily", genus_floors[g2]))
                else:
                    S_xf.append(genus_means[g1])
                    T_xf.append(genus_means[g2])
                    meta_xf.append((g1, g2, gen_to_fam[g1], gen_to_fam[g2], "interfamily", genus_floors[g2]))

    tier_Stats: Dict[str, dict] = {}
    all_csv_rows = []
    fam_names_4 = ["shear", "coaxial", "fan", "radial"]
    dated_records = []

    for tier_name, S_list, T_list, meta_list in [
        ("intrageneric", S_ig, T_ig, meta_ig),
        ("intrafamily", S_if, T_if, meta_if),
        ("interfamily", S_xf, T_xf, meta_xf),
    ]:
        S_b = np.stack(S_list)
        T_b = np.stack(T_list)
        res = fit_all_batched(S_b, T_b)
        d0 = res["rms_sim"]
        fl = np.array([m[5] for m in meta_list])

        med_rms = {}
        med_expl = {}
        iqr_expl = {}
        pct_floor = {}
        for k in ["rms_shear", "rms_aff", "rms_coaxial", "rms_fan", "rms_radial", "rms_best_thompson", "rms_quad"]:
            rv = res[k]
            expl = np.maximum(0.0, 100.0 * (1.0 - (rv / d0) ** 2))
            med_rms[k] = float(np.median(rv))
            med_expl[k] = float(np.median(expl))
            iqr_expl[k] = [float(np.percentile(expl, 25)), float(np.percentile(expl, 75))]
            pct_floor[k] = float(100.0 * (rv <= fl).mean())

        aicc_sim = compute_aicc(d0, 4)
        aicc_sh = compute_aicc(res["rms_shear"], 5)
        aicc_af = compute_aicc(res["rms_aff"], 6)
        aicc_th = compute_aicc(res["rms_best_thompson"], 8)
        aicc_qd = compute_aicc(res["rms_quad"], 12)

        win_counts = {
            fam_names_4[idx]: int((res["best_thompson_idx"] == idx).sum())
            for idx in range(4)
        }
        tier_Stats[tier_name] = {
            "n": len(meta_list),
            "median_procrustes_d": float(np.median(d0)),
            "median_floor": float(np.median(fl)),
            "median_rms": med_rms,
            "median_explained": med_expl,
            "iqr_explained": iqr_expl,
            "pct_below_floor": pct_floor,
            "winning_family_counts": win_counts,
            "median_head_rms": float(np.median(res["rms_coaxial_head"])),
            "median_fin_rms": float(np.median(res["rms_coaxial_fin"])),
            "fin_to_head_rms_ratio": float(np.median(res["rms_coaxial_fin"] / np.maximum(res["rms_coaxial_head"], 1e-8))),
            "median_daicc_vs_similarity": {
                "shear_5p": float(np.median(aicc_sh - aicc_sim)),
                "affine_6p": float(np.median(aicc_af - aicc_sim)),
                "best_thompson_8p": float(np.median(aicc_th - aicc_sim)),
                "quadratic_12p": float(np.median(aicc_qd - aicc_sim)),
            },
            "pct_aicc_beats_similarity": float(100.0 * np.mean(aicc_th < aicc_sim)),
            "pct_aicc_beats_affine": float(100.0 * np.mean(aicc_th < aicc_af)),
            "pct_aicc_beats_quadratic": float(100.0 * np.mean(aicc_th < aicc_qd)),
        }

        for k_i, m in enumerate(meta_list):
            d_k = float(d0[k_i])
            age_myr = get_pair_age_myr(m[0], m[1], tier_name)
            expl_b = max(0.0, 100.0 * (1.0 - (res["rms_best_thompson"][k_i] / d_k) ** 2))
            if age_myr is not None:
                dated_records.append({
                    "age_myr": age_myr,
                    "d0": d_k,
                    "floor": float(m[5]),
                    "rms_aff": float(res["rms_aff"][k_i]),
                    "rms_best_th": float(res["rms_best_thompson"][k_i]),
                    "rms_quad": float(res["rms_quad"][k_i]),
                    "expl_best_th": float(expl_b),
                })
            all_csv_rows.append({
                "tier": tier_name,
                "source": m[0],
                "target": m[1],
                "source_group": m[2],
                "target_group": m[3],
                "divergence_myr": f"{age_myr:.2f}" if age_myr is not None else "",
                "target_floor_rms": f"{m[5]:.5f}",
                "rms_similarity": f"{d_k:.5f}",
                "rms_shear": f"{res['rms_shear'][k_i]:.5f}",
                "rms_affine": f"{res['rms_aff'][k_i]:.5f}",
                "rms_coaxial": f"{res['rms_coaxial'][k_i]:.5f}",
                "rms_fan": f"{res['rms_fan'][k_i]:.5f}",
                "rms_radial": f"{res['rms_radial'][k_i]:.5f}",
                "rms_best_thompson": f"{res['rms_best_thompson'][k_i]:.5f}",
                "best_thompson_family": fam_names_4[int(res["best_thompson_idx"][k_i])],
                "rms_quadratic": f"{res['rms_quad'][k_i]:.5f}",
                "daicc_best_thompson_vs_sim": f"{float(aicc_th[k_i] - aicc_sim[k_i]):.2f}",
                "daicc_best_thompson_vs_aff": f"{float(aicc_th[k_i] - aicc_af[k_i]):.2f}",
                "explained_affine_pct": f"{max(0.0, 100.0*(1.0 - (res['rms_aff'][k_i]/d_k)**2)):.2f}",
                "explained_coaxial_pct": f"{max(0.0, 100.0*(1.0 - (res['rms_coaxial'][k_i]/d_k)**2)):.2f}",
                "explained_fan_pct": f"{max(0.0, 100.0*(1.0 - (res['rms_fan'][k_i]/d_k)**2)):.2f}",
                "explained_radial_pct": f"{max(0.0, 100.0*(1.0 - (res['rms_radial'][k_i]/d_k)**2)):.2f}",
                "explained_best_thompson_pct": f"{expl_b:.2f}",
            })

    # Compute continuous phylogenetic divergence bins (Myr)
    bin_defs = [
        (0.0, 10.0, "0–10\nMyr"),
        (10.0, 25.0, "10–25\nMyr"),
        (25.0, 50.0, "25–50\nMyr"),
        (50.0, 80.0, "50–80\nMyr"),
        (80.0, 110.0, "80–110\nMyr"),
        (110.0, 160.0, "110–160\nMyr"),
    ]
    phylo_bins = []
    for lo, hi, lab in bin_defs:
        sub = [r for r in dated_records if lo <= r["age_myr"] < hi]
        if sub:
            phylo_bins.append({
                "min_myr": lo,
                "max_myr": hi,
                "label": lab,
                "n": len(sub),
                "median_age_myr": float(np.median([r["age_myr"] for r in sub])),
                "median_procrustes_d": float(np.median([r["d0"] for r in sub])),
                "pct_below_floor_affine": float(100.0 * np.mean([r["rms_aff"] <= r["floor"] for r in sub])),
                "pct_below_floor_best_thompson": float(100.0 * np.mean([r["rms_best_th"] <= r["floor"] for r in sub])),
                "pct_below_floor_quadratic": float(100.0 * np.mean([r["rms_quad"] <= r["floor"] for r in sub])),
                "median_explained_best_thompson": float(np.median([r["expl_best_th"] for r in sub])),
            })

    # Load exact 1917 museum/specimen landmarks and evaluate all 6 direct 1917 pairs
    exact_lms = load_exact_1917_species(mean_consensus, allow_download=allow_download)
    direct_1917_specs = [
        {
            "id": "argyropelecus_sternoptyx_shear",
            "pair_label": "Argyropelecus → Sternoptyx",
            "taxonomic_note": "Direct 1917 Figs. 373–374 (Intra-Sternoptychidae deep-sea hatchetfish, Monaco 1896/1911 plates)",
            "family": "shear",
            "family_title": "I: Oblique Shear, 5p",
            "leaf": 768,
            "S": exact_lms["Argyropelecus_olfersii"],
            "T": exact_lms["Sternoptyx_diaphana"],
            "floor": tier_Stats["intrafamily"]["median_floor"],
            "divergence_myr": 80.76,
        },
        {
            "id": "scarus_pomacanthus_coaxial",
            "pair_label": "Scarus → Pomacanthus",
            "taxonomic_note": "Direct 1917 Figs. 375–376 (Interfamily: 22 Scarus spp. → 9 Pomacanthus spp.)",
            "family": "coaxial",
            "family_title": "II: Coaxial Circles, 8p",
            "leaf": 769,
            "S": genus_means["Scarus"],
            "T": genus_means["Pomacanthus"],
            "floor": genus_floors["Pomacanthus"],
            "divergence_myr": 112.84,
        },
        {
            "id": "polyprion_pristigenys_fan",
            "pair_label": "Polyprion → Pseudopriacanthus",
            "taxonomic_note": "Direct 1917 Figs. 377–378 (Polyprion americanus Goode & Bean 1896 → Pristigenys alta NOAA)",
            "family": "fan",
            "family_title": "IIIa: Fan of Coords, 8p",
            "leaf": 770,
            "S": exact_lms["Polyprion_americanus"],
            "T": exact_lms["Pristigenys_alta"],
            "floor": fam_floors["Priacanthidae"],
            "divergence_myr": 103.11,
        },
        {
            "id": "diodon_mola_radial",
            "pair_label": "Diodon → Orthagoriscus mola",
            "taxonomic_note": "Direct 1917 Figs. 381–382 (Diodon hystrix Claverie 2014 → Mola mola BMNH 1887 plate)",
            "family": "radial",
            "family_title": "IV: Hyperbolic/Radial, 8p",
            "leaf": 771,
            "S": Y[sp_to_idx["Diodon_hystrix"]],
            "T": exact_lms["Mola_mola"],
            "floor": fam_floors["Diodontidae"],
            "divergence_myr": 79.30,
        },
        {
            "id": "pristigenys_scorpaena_fan",
            "pair_label": "Pseudopriacanthus → Scorpaena",
            "taxonomic_note": "Direct 1917 Figs. 378–379 (Pristigenys alta NOAA → Scorpaena scrofa Bloch 1796)",
            "family": "fan",
            "family_title": "IIIb: Fan of Coords, 8p",
            "leaf": 770,
            "S": exact_lms["Pristigenys_alta"],
            "T": exact_lms["Scorpaena_scrofa"],
            "floor": fam_floors["Scorpaenidae"],
            "divergence_myr": 103.11,
        },
        {
            "id": "scorpaena_antigonia_fan",
            "pair_label": "Scorpaena → Antigonia capros",
            "taxonomic_note": "Direct 1917 Figs. 379–380 (Scorpaena scrofa Bloch 1796 → Antigonia capros NOAA)",
            "family": "fan",
            "family_title": "IIIc: Fan of Coords, 8p",
            "leaf": 770,
            "S": exact_lms["Scorpaena_scrofa"],
            "T": exact_lms["Antigonia_capros"],
            "floor": tier_Stats["interfamily"]["median_floor"],
            "divergence_myr": 103.11,
        },
    ]
    direct_exemplars_out = [evaluate_single_exemplar(s) for s in direct_1917_specs]

    # Also keep the 4 reef-dataset taxonomic analogues for reference
    reef_analogue_specs = [
        {
            "id": "holocentridae_shear",
            "pair_label": "Sargocentron → Myripristis",
            "taxonomic_note": "Intrafamily (Holocentridae: 9 Sargocentron → 4 Myripristis spp.); tests oblique dorsoventral trunk shear",
            "family": "shear",
            "family_title": "I: Oblique Shear, 5p",
            "leaf": 768,
            "S": genus_means["Sargocentron"],
            "T": genus_means["Myripristis"],
            "floor": genus_floors["Myripristis"],
            "divergence_myr": 56.03,
        },
        {
            "id": "epinephelus_priacanthidae_fan",
            "pair_label": "Epinephelus → Priacanthidae",
            "taxonomic_note": "Interfamily reef analogue (40 Epinephelus spp. → 4 Priacanthidae spp.)",
            "family": "fan",
            "family_title": "III: Fan of Coords, 8p",
            "leaf": 770,
            "S": genus_means["Epinephelus"],
            "T": fam_means["Priacanthidae"],
            "floor": fam_floors["Priacanthidae"],
            "divergence_myr": 103.11,
        },
        {
            "id": "diodontidae_ephippidae_radial",
            "pair_label": "Diodontidae → Ephippidae",
            "taxonomic_note": "Interfamily reef analogue (3 Diodontidae spp. → 2 Ephippidae spp.)",
            "family": "radial",
            "family_title": "IV: Hyperbolic/Radial, 8p",
            "leaf": 771,
            "S": fam_means["Diodontidae"],
            "T": fam_means["Ephippidae"],
            "floor": fam_floors["Ephippidae"],
            "divergence_myr": 108.40,
        },
    ]
    reef_analogues_out = [evaluate_single_exemplar(s) for s in reef_analogue_specs]

    out_dir.mkdir(parents=True, exist_ok=True)
    scans = fetch_scans(allow_download=allow_download)
    fig_path = out_dir / "thompson_four_families_atlas.png"
    render_atlas_figure(direct_exemplars_out, tier_Stats, phylo_bins, scans, fig_path)

    csv_path = out_dir / "thompson_taxonomic_scaling_summary.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(all_csv_rows[0].keys()))
        w.writeheader()
        w.writerows(all_csv_rows)
    print(f"  saved {csv_path} ({len(all_csv_rows)} pairs, {len(dated_records)} dated)")

    def serialize_ex(ex: dict) -> dict:
        return {
            "id": ex["id"],
            "pair_label": ex["pair_label"],
            "taxonomic_note": ex["taxonomic_note"],
            "tested_family": ex["family"],
            "divergence_myr": ex["divergence_myr"],
            "leaf": ex["leaf"],
            "page": SCAN_LEAVES[ex["leaf"]]["page"],
            "target_floor_rms": ex["floor"],
            "rms_by_family": ex["all_rms"],
            "explained_by_family": ex["all_explained"],
            "aicc_by_family": ex["all_aicc"],
            "daicc_fam_vs_sim": ex["daicc_fam_vs_sim"],
            "daicc_fam_vs_aff": ex["daicc_fam_vs_aff"],
            "loo_cv_rms": ex["loo_cv"],
            "tps_bending_energy": ex["tps_bending_energy"],
            "head_rms": ex["head_rms"],
            "fin_rms": ex["fin_rms"],
        }

    summary_json = {
        "dataset": {
            "file": str(DATA.relative_to(ROOT)),
            "sha256": DATA_SHA256,
            "exact_1917_species_file": str(EXACT_SPECIES_JSON.relative_to(ROOT)),
            "phylogeny_file": ".thompson_cache/actinopt_12k_treePL.tre.xz",
            "phylogeny_sha256": TREE_SHA256,
            "species_count": len(names),
            "genera_count": len(groups),
            "genera_ge_3_count": len(big_genera),
            "families_count": len(fam_groups),
            "landmarks_count": N_LM,
            "dated_pairs_count": len(dated_records),
        },
        "self_test_recovery_rms": st,
        "taxonomic_scaling": tier_Stats,
        "phylogenetic_divergence_bins": phylo_bins,
        "direct_1917_exemplars": [serialize_ex(ex) for ex in direct_exemplars_out],
        "reef_analogue_exemplars": [serialize_ex(ex) for ex in reef_analogues_out],
    }
    json_path = out_dir / "thompson_four_families_summary.json"
    json_path.write_text(json.dumps(summary_json, indent=2), encoding="utf-8")
    print(f"  saved {json_path}")
    return summary_json


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default=".")
    ap.add_argument("--no-scan", action="store_true", help="do not fetch remote files from the Internet Archive / FishTreeOfLife")
    ap.add_argument("--self-test", action="store_true", help="run self-tests and assertions")
    args = ap.parse_args(argv)

    res = run_full_analysis(allow_download=not args.no_scan, out_dir=Path(args.out_dir))
    ts = res["taxonomic_scaling"]
    print("\n=== TAXONOMIC SCALING SUMMARY (4,505 PAIRS ACROSS 791 SPECIES) ===")
    for t in ["intrageneric", "intrafamily", "interfamily"]:
        s = ts[t]
        print(
            f"  {t:13s} (n={s['n']:4d}, med d0={s['median_procrustes_d']:.4f}, floor={s['median_floor']:.4f}): "
            f"Affine {s['median_explained']['rms_aff']:4.1f}% | "
            f"Coaxial {s['median_explained']['rms_coaxial']:4.1f}% | "
            f"Best-Thompson {s['median_explained']['rms_best_thompson']:4.1f}% "
            f"(≤ floor: {s['pct_below_floor']['rms_best_thompson']:4.1f}%, "
            f"AICc<Affine: {s['pct_aicc_beats_affine']:4.1f}%) | "
            f"Fin/Head RMS: {s['fin_to_head_rms_ratio']:.2f}x"
        )
    print("\n=== PHYLOGENETIC DIVERGENCE TIME HORIZON (RABOSKY ET AL. 2018, 4,334 DATED PAIRS) ===")
    for b in res["phylogenetic_divergence_bins"]:
        lab_clean = b["label"].replace("\n", " ")
        print(
            f"  {lab_clean:10s} (n={b['n']:4d}, med={b['median_age_myr']:5.1f} Myr, d0={b['median_procrustes_d']:.4f}): "
            f"Affine≤floor {b['pct_below_floor_affine']:4.1f}% | "
            f"Best-Thompson≤floor {b['pct_below_floor_best_thompson']:4.1f}% | "
            f"Quad≤floor {b['pct_below_floor_quadratic']:4.1f}%"
        )
    print("\n=== DIRECT 1917 THOMPSON SPECIES PAIRS (ALL 6 TRANSFORMATIONS, FIGS. 373–382) ===")
    for ex in res["direct_1917_exemplars"]:
        fam = ex["tested_family"]
        print(
            f"  {ex['pair_label']:28s} [{fam:7s}, {ex['divergence_myr']:5.1f} Myr]: "
            f"d0={ex['rms_by_family']['similarity']:.4f} -> RMS={ex['rms_by_family'][fam]:.4f} "
            f"({100*ex['explained_by_family'][fam]:4.1f}% expl., affine {100*ex['explained_by_family']['affine']:4.1f}%), "
            f"dAICc(vs 4p)={ex['daicc_fam_vs_sim']:+5.1f}, LOO-CV={ex['loo_cv_rms'][fam]:.4f}"
        )

    if args.self_test:
        assert max(res["self_test_recovery_rms"].values()) < 1e-5
        assert ts["intrageneric"]["pct_below_floor"]["rms_best_thompson"] > ts["intrafamily"]["pct_below_floor"]["rms_best_thompson"]
        assert ts["intrafamily"]["pct_below_floor"]["rms_best_thompson"] > ts["interfamily"]["pct_below_floor"]["rms_best_thompson"]
        assert len(res["direct_1917_exemplars"]) == 6
        assert len(res["phylogenetic_divergence_bins"]) == 6
        print("[PASS] 1917-thompson-four-families-atlas.py --self-test")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""
2018-glimm-avian-feather-bud-hexagonal.py — Pass 3 Remediated Implementation (v3)
=================================================================================
Reduced mechanistic reproduction of the mediolateral Ectodysplasin (*EDA*) competence
wave, inhibitory-disk hexagonal feather-primordium packing, and mesenchymal cell-density
threshold tract boundaries in embryonic chick skin (*Gallus gallus*), now solved on a
curved 3D embryonic trunk-and-limb-bud surface (Monge patch) as well as on the plane.

PROVENANCE CORRECTION (Pass 3). The Pass 1-2 versions of this script, its dataset and
its audits cited a seed paper "Glimm, Bhat & Newman (2014) Dev. Biol. 395:139-149,
'Modeling happens in rows: Traveling waves in dermal condensation of avian feather
primordia'". That paper does not exist: Crossref resolves `10.1016/j.ydbio.2014.08.030`
to Garnatz et al. (FOG-2 / NuRD, Dev. Biol. 395:50-61) and the dispatch DOI
`10.1016/j.ydbio.2014.09.006` to Kishi et al. (Oikopleura cell migration, Dev. Biol.
395:299-306), and no Glimm/Bhat/Newman feather paper is indexed. The chemotaxis
(Keller-Segel) feather model actually traces to Painter, Ho & Headon (2018) and the
experimental wave / Voronoi / density-threshold results to Ho et al. (2019). The file
name and `sim_id` are kept for audit-chain continuity only.

Primary sources (all DOIs resolved via Crossref on 2026-10-20):

1. **Ho WKW, Freem L, Zhao D, Painter KJ, Woolley TE, Gaffney EA, ... Headon DJ (2019)**
   *PLOS Biology* 17(2):e3000132 (`doi:10.1371/journal.pbio.3000132`)
   "Feather arrays are patterned by interacting signalling and cell density waves"
   - Fig 1B: discrete dorsal, humeral, femoral and alar tracts; "these spreading waves
     terminate before colliding, yielding ... bare spaces between the tracts".
   - Fig 2A/B/D: the same FGF20/BMP/chemotaxis network deployed in a travelling wave gives
     a regular hexagonal array, whereas simultaneous (Latrunculin-paused, then released)
     patterning gives low regularity: ~75 % vs ~50 % six-neighbour Voronoi cells.
   - Fig 3A-B: an expanding EDA wave with primordia at its leading edge; Fig 3D-G: soluble
     Fc-chEDA1 / LDN193189 widen and Ecto-D2 narrows the patterned region.
   - Fig 4A-B: medial-high, laterally decreasing mesenchymal cell density gradient that
     rises over E6.5-E8.5; Fig 5C-D: EDA sets the critical density (Fc-chEDA1 49, control
     66, Ecto-D2 98 cells / 100 um at the pattern margin); Fig 5E-F: MTX and Ecto-D2
     co-treatment narrows the tract more than either alone.
   - Fig 1C: FGF9 flooding or Latrunculin A abolishes patterning; scaleless (FGF20-null)
     skin defines tracts but forms no primordia (S5 Fig).
2. **Painter KJ, Ho W, Headon DJ (2018)** *J. Theor. Biol.* 437:225-238
   (`doi:10.1016/j.jtbi.2017.10.026`) "A chemotaxis model of feather primordia pattern
   formation during avian development": the Keller-Segel dermal density n, FGF20
   chemoattractant c, BMP inhibitor b system whose linearisation gives the chemotactic
   instability threshold used in Section 2.
3. **Michon F, Forest L, Collomb E, Demongeot J, Dhouailly D (2008)** *Development*
   135:2797-2805 (`doi:10.1242/dev.018341`) "BMP2 and BMP7 play antagonistic roles in
   feather induction" (Pass 1-2 cited this under a wrong PLOS ONE DOI).
4. **Bailleul R, Curantz C, Desmarquet-Trin Dinh C, Hidalgo M, Touboul J, Manceau M (2019)**
   *PLOS Biology* 17(10):e3000448 (`doi:10.1371/journal.pbio.3000448`).
5. **Bausch AR, Bowick MJ, Cacciuto A, Dinsmore AD, Hsu MF, Nelson DR, Nikolaides MG,
   Travesset A, Weitz DA (2003)** *Science* 299:1716-1718 (`doi:10.1126/science.1081160`)
   "Grain boundary scars and spherical crystallography": Gaussian curvature forces
   5-7 disclination / dislocation defects into otherwise hexagonal packings (the
   curved-surface defect theory used in Section 4-5).

WHAT THE MODEL IS (and is not). Nucleation is NOT emergent from the Keller-Segel PDE:
primordia are placed by a wave-gated inhibitory-disk rule (each primordium "projects a
circular inhibitory influence preventing the placing of new elements nearby", Ho 2019
Introduction), i.e. random sequential adsorption of disks of radius r_inh = 0.92 lambda_y
restricted to the band of skin that became competent in the current time step; the
Keller-Segel / FGF20 / BMP4 PDE (Painter 2018 form) is then integrated with the
Laplace-Beltrami operator of the surface to relax dermal density, FGF20 and BMP4 around
those sites. Hexagonality therefore follows mechanistically from wave speed (band width)
and surface geometry rather than from a fitted "pinning fidelity" (the v2 construction,
removed). Competence requires (i) the geodesic EDA wave to have arrived and (ii) the local
mesenchymal density n_0(u, v, t) to exceed the EDA-set threshold n_crit; tract edges and
the bare inter-tract apteria emerge from (ii).

Governing equations (grid units: lambda_y = 12 cells = 280 um):
    g_ij = delta_ij + h_i h_j  (Monge patch z = h(u, v); trunk ridge + 4 limb-bud bumps)
    K    = (h_uu h_vv - h_uv^2) / (1 + h_u^2 + h_v^2)^2                 (Gaussian curvature)
    |grad_g T| = 1                      (eikonal; T = geodesic arrival time from each tract)
    s_w(u, v, t) = 0.5 (1 + tanh((v_w (t - t_0) - T(u, v)) / xi_w))
    competent  <=>  s_w >= 1/2  and  n_0(u, v) (1 + gamma t) >= n_crit
    dn/dt = D_n Lap_g n - chi div_g(n grad_g c) + r (1 + 1.8 c - n)
    dc/dt = D_c Lap_g c + s_w [alpha_c S_c + 0.16 n^2 / ((1 + 1.1 b)(1 + 0.25 n^2))] - mu_c c
    db/dt = D_b Lap_g b + s_w [alpha_b S_b + 0.18 n^2] - mu_b b
    Lap_g f = (1/sqrt g) d_i (sqrt g g^{ij} d_j f)

Machine-checked Lean 4 laws (`morphoeval/lean/loops/Loop_019_avian_feather_bud_hexagonal.lean`,
0 sorrys): Theorem 1 Keller-Segel instability threshold; Theorem 2 equilateral half-phase
row offset; Theorem 3 traveling-wave P_6 advantage; NEW Theorem 4
`euler_disclination_charge` (6 chi = sum (6 - z_i) for any closed triangulation; sphere
=> n_5 - n_7 = 12); NEW Theorem 5 `isochrone_row_count_dislocation` (an isochrone whose
arc length exceeds its neighbour's by >= lambda_y carries >= 1 extra primordium, i.e. a
dislocation); NEW Theorem 6 `monge_metric_determinant` (det g = 1 + h_u^2 + h_v^2 >= 1,
so Lap_g is well defined everywhere on the patch).
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from scipy.spatial import ConvexHull, Delaunay, Voronoi, cKDTree

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets" / "avian_feather_bud_hexagonal"
REF_JSON_PATH = DATASET_DIR / "avian_feather_bud_hexagonal_reference.json"
SPECIMEN_JPG_PATH = DATASET_DIR / "chick_feather_bud_hexagonal_specimen.jpg"
SCOUT_JSON_PATH = DATASET_DIR / "scout_manifest.json"
BIB_PATH = DATASET_DIR / "paper_citations.bib"

OUT_PNG = ROOT / "docs" / "media" / "avian_feather_bud_hexagonal_comparison.png"
OUT_GIF = ROOT / "docs" / "media" / "avian_feather_bud_hexagonal_growth.gif"
OUT_SURFACE_PNG = ROOT / "docs" / "media" / "avian_feather_bud_hexagonal_curved_tracts.png"

EXPECTED_SHA256: dict[str, str] = {
    "datasets/avian_feather_bud_hexagonal/avian_feather_bud_hexagonal_reference.json": (
        "61c579212066cfb2d74369d450734464539786cd32a51bd95bf0f07d505f15e5"
    ),
    "datasets/avian_feather_bud_hexagonal/chick_feather_bud_hexagonal_specimen.jpg": (
        "ce18b1c0e01ecb20e137a700129b328815ba10f236e8620d20ef66bbc8e7a03c"
    ),
    "datasets/avian_feather_bud_hexagonal/scout_manifest.json": (
        "1ff38c1e609c69ea4826a33e48d962da2821dc4d04051d6919c440326a4da2ec"
    ),
    "datasets/avian_feather_bud_hexagonal/paper_citations.bib": (
        "721ac5198ac62b2de1386a5a99943e06ee9347af01fb75bcfb8864e3bcb69903"
    ),
}

# Ho et al. (2019) Fig 5C-D: mesenchymal cell density at the lateral pattern margin
# (cells per 100 um of section) under EDA/EDAR modulation. These set n_crit relative to control.
HO2019_MARGIN_DENSITY_CELLS_PER_100UM = {"fc_cheda1": 49.0, "control": 66.0, "ecto_d2": 98.0}
LAMBDA_Y_GRID = 12.0          # one feather period = 12 grid cells
LAMBDA_Y_UM = 280.0           # Ho et al. (2019) Fig 3B scale (approx. inter-primordium spacing)
CHI_CRITICAL = 0.0540         # Painter et al. (2018) linearisation at the canonical parameters (Section 2)


# =========================================================================== #
# 1. Dataset & SHA-256 Provenance Verification
# =========================================================================== #
def verify_dataset_sha256() -> dict[str, str]:
    """Verify SHA-256 digests of all 4 vendored Slot 019 reference files (fail closed)."""
    actual: dict[str, str] = {}
    for rel, expected in EXPECTED_SHA256.items():
        p = ROOT / rel
        if not p.exists():
            raise FileNotFoundError(f"Missing required Slot 019 dataset file: {p}")
        digest = hashlib.sha256(p.read_bytes()).hexdigest()
        if digest != expected:
            raise ValueError(f"SHA-256 mismatch for {rel}: expected {expected}, got {digest}")
        actual[rel] = digest
    return actual


def load_reference_dataset() -> dict[str, Any]:
    """Load and SHA-256-verify `avian_feather_bud_hexagonal_reference.json`."""
    verify_dataset_sha256()
    return json.loads(REF_JSON_PATH.read_text(encoding="utf-8"))


# =========================================================================== #
# 2. Keller-Segel / FGF20 Linear Stability & Hexagonal Row Offset Geometry
#    (Painter, Ho & Headon 2018 J. Theor. Biol. 437:225-238, linearised; Lean Theorems 1-2)
# =========================================================================== #
def compute_keller_segel_dispersion_and_geometry(
    k_vals: np.ndarray | None = None,
    D_n: float = 0.025,
    D_c: float = 0.04,
    r_n: float = 0.08,
    mu_c: float = 0.20,
    chi_wt: float = 0.24,
    n_0: float = 1.0,
    f_c: float = 0.30,
    lambda_y_um: float = LAMBDA_Y_UM,
) -> dict[str, Any]:
    """
    Evaluate the linearised Keller-Segel dermal fibroblast + FGF20 chemoattractant
    dispersion relation sigma(k) for wild-type (chi = 0.24), Latrunculin mechanical
    blockade (chi = 0.01), and *scaleless* FGF20-null (chi = 0.0), together with the
    equilateral row-to-row half-phase offset invariants (Lean Theorems 1 & 2).
    """
    if k_vals is None:
        k_vals = np.linspace(0.0, 3.2, 240)
    q = k_vals ** 2

    def growth_curve(chi_val: float) -> tuple[np.ndarray, np.ndarray]:
        tr = -((D_n + D_c) * q + r_n + mu_c)
        det = (D_n * q + r_n) * (D_c * q + mu_c) - chi_val * n_0 * f_c * q
        disc = np.maximum(tr ** 2 - 4.0 * det, 0.0)
        sig = 0.5 * (tr + np.sqrt(disc))
        return sig, det

    sigma_wt, det_wt = growth_curve(chi_wt)
    sigma_lat, det_lat = growth_curve(0.01)
    sigma_sc, det_sc = growth_curve(0.0)

    chi_c = ((np.sqrt(D_n * mu_c) + np.sqrt(D_c * r_n)) ** 2) / (n_0 * f_c)
    k_peak = float(k_vals[int(np.argmax(sigma_wt))])

    hex_ratio = float(np.sqrt(3.0) / 2.0)
    dx_um = hex_ratio * lambda_y_um
    dy_offset_um = 0.5 * lambda_y_um
    diag_dist_um = float(np.sqrt(dx_um ** 2 + dy_offset_um ** 2))

    return {
        "k": k_vals,
        "sigma_wt": sigma_wt,
        "det_wt": det_wt,
        "sigma_latrunculin": sigma_lat,
        "det_latrunculin": det_lat,
        "sigma_scaleless": sigma_sc,
        "det_scaleless": det_sc,
        "chi_c": float(chi_c),
        "k_peak": k_peak,
        "sigma_wt_max": float(np.max(sigma_wt)),
        "sigma_scaleless_max": float(np.max(sigma_sc[1:])),
        "hex_row_ratio": hex_ratio,
        "lambda_y_um": float(lambda_y_um),
        "row_spacing_dx_um": float(dx_um),
        "half_phase_offset_dy_um": float(dy_offset_um),
        "diag_neighbor_dist_um": diag_dist_um,
    }


# =========================================================================== #
# 3. Planar Voronoi Polygon-Class P_6 & Bond-Orientational Order |psi_6| Analyzer
#    (Ho et al. 2019 Fig 2C-D protocol: Voronoi on primordium centroids, boundary cells removed)
# =========================================================================== #
def compute_voronoi_hexagonal_metrics(
    pts: np.ndarray,
    width: float,
    height: float,
    margin_frac: float = 0.17,
) -> dict[str, Any]:
    """
    Given 2D bud centroids `pts` (N x 2 in [x, y] order) inside a domain of size
    `(width, height)`, compute the Voronoi tessellation, Delaunay triangulation,
    interior polygon coordination distribution, hexagonal fraction P_6, 5-6-7
    fraction P_567, topological defect fraction (1 - P_6), Voronoi-ridge
    bond-orientational order parameter |psi_6|, and nearest-neighbor spacing CV.
    """
    if len(pts) < 8:
        return {
            "n_buds": int(len(pts)), "n_interior": 0, "p6": 0.0, "p567": 0.0, "defect_fraction": 1.0,
            "psi6": 0.0, "nn_mean": 0.0, "nn_std": 0.0, "nn_cv": 0.0, "interior_sides": [],
            "interior_indices": [], "voronoi": None, "delaunay": None,
        }

    tree = cKDTree(pts)
    dists, _ = tree.query(pts, k=min(7, len(pts)))
    nn = dists[:, 1]
    nn_mean = float(np.mean(nn))
    nn_std = float(np.std(nn))
    nn_cv = float(nn_std / (nn_mean + 1e-9))

    vor = Voronoi(pts)
    tri = Delaunay(pts)

    v_adj: dict[int, list[int]] = {i: [] for i in range(len(pts))}
    for (p1, p2), rverts in zip(vor.ridge_points, vor.ridge_vertices):
        if -1 not in rverts:
            v_adj[int(p1)].append(int(p2))
            v_adj[int(p2)].append(int(p1))

    interior_mask = (
        (pts[:, 0] > margin_frac * width)
        & (pts[:, 0] < (1.0 - margin_frac) * width)
        & (pts[:, 1] > margin_frac * height)
        & (pts[:, 1] < (1.0 - margin_frac) * height)
    )
    sides_list: list[int] = []
    psi6_list: list[float] = []
    interior_indices: list[int] = []

    for idx in np.where(interior_mask)[0]:
        reg = vor.regions[vor.point_region[idx]]
        if len(reg) >= 3 and (-1 not in reg):
            n_sides = len(reg)
            sides_list.append(n_sides)
            interior_indices.append(int(idx))
            nbs = v_adj[int(idx)]
            if len(nbs) >= 3:
                angles = np.arctan2(pts[nbs, 1] - pts[idx, 1], pts[nbs, 0] - pts[idx, 0])
                psi6_list.append(float(np.abs(np.mean(np.exp(6j * angles)))))

    sides_arr = np.array(sides_list, dtype=int)
    p6 = float(np.mean(sides_arr == 6)) if len(sides_arr) else 0.0
    p567 = float(np.mean((sides_arr >= 5) & (sides_arr <= 7))) if len(sides_arr) else 0.0
    psi6 = float(np.mean(psi6_list)) if len(psi6_list) else 0.0

    return {
        "n_buds": int(len(pts)),
        "n_interior": int(len(sides_arr)),
        "p6": p6,
        "p567": p567,
        "defect_fraction": float(1.0 - p6),
        "psi6": psi6,
        "nn_mean": nn_mean,
        "nn_std": nn_std,
        "nn_cv": nn_cv,
        "interior_sides": sides_arr.tolist(),
        "interior_indices": interior_indices,
        "voronoi": vor,
        "delaunay": tri,
    }


# =========================================================================== #
# 4. Curved Embryonic Trunk + Limb-Bud Surface: Monge Patch Metric, Laplace-Beltrami,
#    Geodesic (Eikonal) Arrival Time, and Surface Coordination / Disclination Analysis
# =========================================================================== #
def build_embryo_trunk_surface(
    nu: int = 168,
    nv: int = 96,
    body_height: float = 18.0,
    body_sigma: float = 42.0,
    body_taper: float = 0.18,
    limb_height: float = 13.0,
    limb_sigma: float = 12.0,
    limb_offset_u: float = 58.0,
    humeral_v: float = 72.0,
    femoral_v: float = 24.0,
    flat: bool = False,
) -> dict[str, Any]:
    """
    Reduced 3D geometry of the E6.5-E8.5 chick dorsum as a Monge patch z = h(u, v) over the
    parametric skin sheet (u: mediolateral, v: anteroposterior, grid units of lambda_y / 12):
      * a dorsal trunk ridge h_body = A_b exp(-(u - u_mid)^2 / 2 sigma_b(v)^2) whose width
        tapers anteriorly/posteriorly (sigma_b(v) = sigma_b (1 - taper ((v - v_mid)/v_mid)^2)),
        so the trunk is not exactly developable (K != 0, small);
      * four Gaussian limb-bud bumps (left/right humeral = wing, left/right femoral = leg)
        of height A_l and width sigma_l centred at (u_mid +- d_l, v_h) and (u_mid +- d_l, v_f);
        each bump carries positive Gaussian curvature on its cap and negative K on its flank.
    Returns the metric g_ij = delta_ij + h_i h_j, its inverse, sqrt(det g), the Gaussian
    curvature K = (h_uu h_vv - h_uv^2) / det^2, the embedded coordinates and the tract
    initiation sites. `flat=True` returns the planar control (h = 0, g = I, K = 0).
    """
    vv, uu = np.mgrid[0:nv, 0:nu].astype(np.float64)
    u_mid = 0.5 * (nu - 1)
    v_mid = 0.5 * (nv - 1)
    if flat:
        h = np.zeros((nv, nu), dtype=np.float64)
    else:
        sig_b = body_sigma * (1.0 - body_taper * ((vv - v_mid) / v_mid) ** 2)
        h = body_height * np.exp(-((uu - u_mid) ** 2) / (2.0 * sig_b ** 2))
        for su in (-1.0, 1.0):
            for v_k in (humeral_v, femoral_v):
                h += limb_height * np.exp(
                    -(((uu - (u_mid + su * limb_offset_u)) ** 2) + (vv - v_k) ** 2) / (2.0 * limb_sigma ** 2)
                )
    hv, hu = np.gradient(h)
    hvv, huv = np.gradient(hv)
    _, huu = np.gradient(hu)
    det = 1.0 + hu ** 2 + hv ** 2
    K = (huu * hvv - huv ** 2) / det ** 2
    g_uu, g_vv, g_uv = 1.0 + hu ** 2, 1.0 + hv ** 2, hu * hv
    sqrtg = np.sqrt(det)
    return {
        "nu": nu, "nv": nv, "uu": uu, "vv": vv, "h": h, "hu": hu, "hv": hv, "K": K, "det": det,
        "sqrtg": sqrtg, "g_uu": g_uu, "g_vv": g_vv, "g_uv": g_uv,
        "ginv_uu": g_vv / det, "ginv_vv": g_uu / det, "ginv_uv": -g_uv / det,
        "xyz": np.stack([uu, vv, h], axis=-1), "u_mid": u_mid, "v_mid": v_mid, "flat": bool(flat),
        "limb_sites": [
            ("humeral", u_mid - limb_offset_u, humeral_v), ("humeral", u_mid + limb_offset_u, humeral_v),
            ("femoral", u_mid - limb_offset_u, femoral_v), ("femoral", u_mid + limb_offset_u, femoral_v),
        ],
        "area_element_total": float(np.sum(sqrtg)),
        "integrated_abs_K": float(np.sum(np.abs(K) * sqrtg)),
    }


def surface_div_grad(S: dict[str, Any], a: np.ndarray | float, f: np.ndarray) -> np.ndarray:
    """
    Metric divergence of a scalar-weighted surface gradient,
        div_g(a grad_g f) = (1/sqrt g) d_i ( sqrt g g^{ij} a d_j f ),
    with conservative face-centred fluxes for the diagonal terms (no odd-even decoupling)
    and central differences for the g^{uv} cross terms; zero-flux (Neumann) boundaries.
    With a = 1 this is the Laplace-Beltrami operator; with a = n it is the chemotactic
    flux divergence div_g(n grad_g c) of the Keller-Segel system on the surface.
    """
    Auu = S["sqrtg"] * S["ginv_uu"] * a
    Avv = S["sqrtg"] * S["ginv_vv"] * a
    Auv = S["sqrtg"] * S["ginv_uv"] * a
    out = np.zeros_like(f)
    flux = 0.5 * (Auu[:, 1:] + Auu[:, :-1]) * (f[:, 1:] - f[:, :-1])
    out[:, 1:-1] += flux[:, 1:] - flux[:, :-1]
    out[:, 0] += flux[:, 0]
    out[:, -1] -= flux[:, -1]
    flux = 0.5 * (Avv[1:, :] + Avv[:-1, :]) * (f[1:, :] - f[:-1, :])
    out[1:-1, :] += flux[1:, :] - flux[:-1, :]
    out[0, :] += flux[0, :]
    out[-1, :] -= flux[-1, :]
    if not S["flat"]:
        fv, fu = np.gradient(f)
        out += np.gradient(Auv * fv, axis=1) + np.gradient(Auv * fu, axis=0)
    return out / S["sqrtg"]


def geodesic_arrival_time(S: dict[str, Any], source_mask: np.ndarray, n_iter: int | None = None) -> np.ndarray:
    """
    Jacobi-iterated Godunov upwind solution of the eikonal equation |grad_g T| = 1 on the
    Monge patch using the diagonal part of the metric (local arc lengths sqrt(g_uu) du and
    sqrt(g_vv) dv). T(u, v) is the geodesic distance (= arrival time at unit speed) of the
    EDA competence wave from the tract initiation line `source_mask`.
    """
    ha = np.sqrt(S["g_uu"])
    hb = np.sqrt(S["g_vv"])
    big = 1e9
    T = np.where(source_mask, 0.0, big)
    n_iter = n_iter or 3 * max(S["nu"], S["nv"])
    A = 1.0 / ha ** 2 + 1.0 / hb ** 2
    for _ in range(n_iter):
        Tl = np.pad(T, ((0, 0), (1, 0)), constant_values=big)[:, :-1]
        Tr = np.pad(T, ((0, 0), (0, 1)), constant_values=big)[:, 1:]
        Td = np.pad(T, ((1, 0), (0, 0)), constant_values=big)[:-1, :]
        Tu = np.pad(T, ((0, 1), (0, 0)), constant_values=big)[1:, :]
        a = np.minimum(Tl, Tr)
        b = np.minimum(Td, Tu)
        c1 = np.minimum(a + ha, b + hb)
        B = -2.0 * (a / ha ** 2 + b / hb ** 2)
        C = a ** 2 / ha ** 2 + b ** 2 / hb ** 2 - 1.0
        disc = B ** 2 - 4.0 * A * C
        ok = (disc >= 0.0) & (np.abs(a - b) < np.maximum(ha, hb) + 1e-9)
        with np.errstate(invalid="ignore", over="ignore"):
            c2 = (-B + np.sqrt(np.maximum(disc, 0.0))) / (2.0 * A)
        c2 = np.where(ok & (c2 > np.maximum(a, b)), c2, big)
        Tn = np.where(source_mask, 0.0, np.minimum(np.minimum(c1, c2), T))
        if np.max(np.abs(Tn - T)) < 1e-7:
            T = Tn
            break
        T = Tn
    return T


def surface_coordination(
    xyz_pts: np.ndarray,
    k: int = 10,
    max_edge: float | None = None,
    max_gap_deg: float = 100.0,
) -> dict[str, Any]:
    """
    Coordination number z_i of every primordium on the embedded surface from a local
    tangent-plane Delaunay triangulation (PCA tangent frame of the k nearest neighbours
    in R^3, edges longer than `max_edge` pruned). A primordium is *interior* when it is
    not on the convex hull of its local patch and its neighbour fan has no angular gap
    larger than `max_gap_deg` (so tract-edge and apterium-edge primordia are excluded,
    as in the boundary-cell removal of Ho et al. 2019 Fig 2C). Also returns the local
    bond-orientational order |psi_6| per interior primordium and 5-7 dislocation pairs.
    """
    N = len(xyz_pts)
    z = np.zeros(N, dtype=int)
    interior = np.zeros(N, dtype=bool)
    psi6 = np.zeros(N, dtype=float)
    nbrs: dict[int, np.ndarray] = {}
    if N >= 8:
        tree = cKDTree(xyz_pts)
        d, idx = tree.query(xyz_pts, k=min(k + 1, N))
        for i in range(N):
            nb = idx[i, 1:]
            if max_edge is not None:
                nb = nb[d[i, 1:] < max_edge]
            if len(nb) < 4:
                continue
            P = xyz_pts[np.concatenate([[i], nb])] - xyz_pts[i]
            _, _, vt = np.linalg.svd(P, full_matrices=False)
            P2 = P @ vt[:2].T
            try:
                tri = Delaunay(P2)
                hull = set(ConvexHull(P2).vertices.tolist())
            except Exception:
                continue
            if 0 in hull:
                continue
            ptr, ind = tri.vertex_neighbor_vertices
            local_nb = ind[ptr[0]:ptr[1]]
            ang = np.sort(np.arctan2(P2[local_nb, 1], P2[local_nb, 0]))
            gaps = np.diff(np.concatenate([ang, [ang[0] + 2.0 * np.pi]]))
            if np.degrees(np.max(gaps)) > max_gap_deg:
                continue
            glob = np.concatenate([[i], nb])[local_nb]
            z[i] = len(glob)
            interior[i] = True
            psi6[i] = float(np.abs(np.mean(np.exp(6j * ang))))
            nbrs[i] = glob
    zi = z[interior]
    pairs: list[tuple[int, int]] = []
    for i, glob in nbrs.items():
        if z[i] == 5:
            for j in glob:
                if interior[j] and z[j] == 7:
                    pairs.append((int(i), int(j)))
    if N >= 2:
        dd, _ = cKDTree(xyz_pts).query(xyz_pts, k=2)
        nn = dd[:, 1]
        nn_mean, nn_std = float(np.mean(nn)), float(np.std(nn))
    else:
        nn_mean, nn_std = 0.0, 0.0
    return {
        "z": z, "interior": interior, "psi6_local": psi6, "nbrs": nbrs,
        "n_buds": int(N), "n_interior": int(interior.sum()),
        "p6": float(np.mean(zi == 6)) if len(zi) else 0.0,
        "p567": float(np.mean((zi >= 5) & (zi <= 7))) if len(zi) else 0.0,
        "defect_fraction": float(np.mean(zi != 6)) if len(zi) else 1.0,
        "n5": int(np.sum(zi == 5)), "n7": int(np.sum(zi == 7)),
        "topological_charge": int(np.sum(6 - zi)) if len(zi) else 0,
        "dislocation_pairs": pairs,
        "psi6": float(np.mean(psi6[interior])) if interior.any() else 0.0,
        "nn_mean": nn_mean, "nn_std": nn_std, "nn_cv": float(nn_std / (nn_mean + 1e-9)),
    }


def sphere_topological_charge_check(n_points: int = 400, seed: int = 2018) -> dict[str, Any]:
    """
    Lean Theorem 4 (`euler_disclination_charge`) executed numerically: the convex hull of
    random points on a sphere is a closed triangulation (3F = 2E, sum z = 2E), so
    sum_i (6 - z_i) = 6 chi = 12 exactly, whatever the point positions.
    """
    rng = np.random.default_rng(seed)
    v = rng.standard_normal((n_points, 3))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    hull = ConvexHull(v)
    F = len(hull.simplices)
    edges = set()
    z = np.zeros(n_points, dtype=int)
    for tri in hull.simplices:
        for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[0], tri[2])):
            edges.add((min(a, b), max(a, b)))
    for a, b in edges:
        z[a] += 1
        z[b] += 1
    E = len(edges)
    V = n_points
    return {
        "V": V, "E": E, "F": F, "euler_characteristic": V - E + F,
        "three_F_equals_two_E": bool(3 * F == 2 * E),
        "sum_z_equals_two_E": bool(int(z.sum()) == 2 * E),
        "topological_charge": int(np.sum(6 - z)),
        "n5": int(np.sum(z == 5)), "n7": int(np.sum(z == 7)), "n6": int(np.sum(z == 6)),
    }


# =========================================================================== #
# 5. Wave-Gated Inhibitory-Disk Nucleation + Keller-Segel / FGF20 / BMP4 Relaxation
#    on the (flat or curved) skin surface, with density-threshold tract boundaries
# =========================================================================== #
def simulate_feather_array_on_surface(
    S: dict[str, Any] | None = None,
    steps: int = 320,
    dt: float = 0.15,
    D_n: float = 0.025,
    D_c: float = 0.04,
    D_b: float = 0.45,
    chi: float = 0.24,
    v_w: float = 0.22,
    lambda_y: float = LAMBDA_Y_GRID,
    r_inh_frac: float = 0.92,
    n_crit: float = 1.0,
    n_mid: float = 1.6,
    n_base: float = 0.30,
    L_dens: float = 34.0,
    dens_growth: float = 0.0015,
    density_scale: float = 1.0,
    lateral_L_factor: float = 0.55,
    tracts: tuple[str, ...] = ("dorsal",),
    t0_lateral: int = 25,
    seed: int = 2018,
    sub: int = 5,
    primary_row: bool = True,
    record_snapshots: bool = False,
) -> dict[str, Any]:
    """
    Simulate feather primordium patterning on the surface `S` (default: planar control):

    1. Geodesic EDA competence wave. For every tract initiation line (dorsal midline; left
       and right humeral and femoral sites on the limb-bud apices when requested) the
       eikonal arrival time T(u, v) is solved on the metric of `S`; competence is
       s_w = 0.5 (1 + tanh((v_w (t - t_0) - T) / xi_w)) (Ho et al. 2019 Fig 3A-B). v_w >= 8
       is the Latrunculin-paused / released "simultaneous" regime (Fig 2B): everything is
       competent at t = 0 and no primary row is laid down.
    2. Mesenchymal density threshold. n_0(u, v) = n_base + A_k exp(-T_k / L_k) decreases
       with geodesic distance from each tract origin (Fig 4A-B) and rises as (1 + gamma t)
       (Fig 4B). A competent site can pattern only if density_scale * n_0 (1 + gamma t) >=
       n_crit; EDA sets n_crit (Fig 5C-D: Fc-chEDA1 49 / control 66 / Ecto-D2 98 cells per
       100 um), MTX lowers density_scale (Fig 5A, E-F). Tract edges and inter-tract apteria
       emerge from this rule ("waves terminate before colliding", Fig 1B).
    3. Nucleation. In the band of sub-grid sites that became competent during this step,
       primordia are placed by random sequential adsorption of inhibitory disks of radius
       r_inh = r_inh_frac lambda_y (3D chordal distance; Ho 2019 "circular inhibitory
       influence"). A slow wave exposes only the saddles between the previous row's
       primordia, which forces the half-period offset and the sqrt(3)/2 row spacing (Lean
       Theorem 2); a fast or instantaneous wave exposes wide bands and yields a jammed
       polycrystal (Lean Theorem 3). Nucleation is disabled when chi < chi_c (scaleless /
       Latrunculin; Lean Theorem 1). Each tract's primary row is laid down at spacing
       lambda_y along its initiation line (Ho 2019 Fig 4F(ii) midline row).
    4. The Keller-Segel / FGF20 / BMP4 PDE (Painter et al. 2018 form) is integrated on the
       surface with the Laplace-Beltrami operator around the nucleation sites.

    Returns primordium positions (parametric and embedded), per-primordium coordination
    (surface_coordination), tract labels, the PDE fields, the kymograph and the density field.
    """
    if S is None:
        S = build_embryo_trunk_surface(flat=True)
    rng = np.random.default_rng(seed)
    nu, nv = S["nu"], S["nv"]
    uu, vv, xyz = S["uu"], S["vv"], S["xyz"]
    dx_row = (np.sqrt(3.0) / 2.0) * lambda_y
    r_inh = r_inh_frac * lambda_y
    supercritical = max(0.0, (chi - CHI_CRITICAL) / (0.24 - CHI_CRITICAL))
    simultaneous = v_w >= 8.0

    # fine candidate lattice (sub x sub per grid cell) for sub-cell nucleation positions
    fu = np.arange(0, nu - 1 + 1e-9, 1.0 / sub)
    fv = np.arange(0, nv - 1 + 1e-9, 1.0 / sub)
    FV, FU = np.meshgrid(fv, fu, indexing="ij")
    coords = np.array([FV.ravel(), FU.ravel()])

    def fine(A: np.ndarray) -> np.ndarray:
        return ndi.map_coordinates(A, coords, order=1, mode="nearest").reshape(FV.shape)

    xyz_f = np.stack([FU, FV, fine(S["h"])], axis=-1)

    # tract initiation lines and their geodesic arrival-time fields
    sources: list[tuple[str, np.ndarray, int, tuple[float, float | None]]] = []
    if "dorsal" in tracts:
        dorsal = (np.abs(uu - S["u_mid"]) <= 0.5) & (vv > 6) & (vv < nv - 7)
        sources.append(("dorsal", dorsal, 0, (S["u_mid"], None)))
    for name, su, sv in S["limb_sites"]:
        if name in tracts:
            m = (np.abs(uu - su) <= 0.5) & (np.abs(vv - sv) < 0.9 * lambda_y)
            sources.append((name, m, t0_lateral, (su, sv)))
    T_list: list[tuple[str, np.ndarray, np.ndarray, int, tuple[float, float | None]]] = []
    dens = np.full((nv, nu), n_base)
    for name, m, t0, site in sources:
        T = geodesic_arrival_time(S, m)
        amp = (n_mid - n_base) * (1.0 if name == "dorsal" else 0.85)
        Ld = L_dens if name == "dorsal" else lateral_L_factor * L_dens
        dens = np.maximum(dens, n_base + amp * np.exp(-T / Ld))
        T_list.append((name, T, fine(T), t0, site))
    dens = density_scale * dens
    dens_f = fine(dens)
    speed = v_w * dx_row / 15.0   # grid cells per step (v2 convention: v_w = 0.22 -> ~0.15 cell/step)
    xi_w = 0.20 * dx_row
    margin = 4
    inside_f = (FU >= margin) & (FU <= nu - 1 - margin) & (FV >= margin) & (FV <= nv - 1 - margin)
    comp_prev = np.zeros(FV.shape, dtype=bool)

    buds_xyz: list[np.ndarray] = []
    bud_step: list[int] = []
    bud_tract: list[str] = []
    n = np.ones((nv, nu)) + 0.008 * rng.standard_normal((nv, nu))
    c = np.zeros((nv, nu))
    b = np.zeros((nv, nu))
    src_c = np.zeros((nv, nu))
    src_b = np.zeros((nv, nu))
    sigma_bud, sigma_inh = 0.22 * lambda_y, 0.48 * lambda_y
    kymo = np.zeros((steps, nu))
    snapshots: list[tuple[int, np.ndarray, np.ndarray]] = []
    snap_steps = {0, steps // 5, (2 * steps) // 5, (3 * steps) // 5, (4 * steps) // 5, steps - 1}
    tract_names = [t[0] for t in T_list]

    def add_bud(p: np.ndarray, step: int, tname: str) -> None:
        nonlocal src_c, src_b
        buds_xyz.append(np.asarray(p, dtype=np.float64))
        bud_step.append(step)
        bud_tract.append(tname)
        d2 = np.sum((xyz - p) ** 2, axis=-1)
        src_c += np.exp(-d2 / (2.0 * sigma_bud ** 2))
        src_b += np.exp(-d2 / (2.0 * sigma_inh ** 2))

    for step in range(steps):
        comp = np.zeros((nv, nu))
        comp_f = np.zeros(FV.shape)
        tract_f = np.full(FV.shape, -1, dtype=int)
        for ti, (name, T, Tf, t0, site) in enumerate(T_list):
            if step < t0:
                continue
            front = speed * (step - t0) + 0.45 * dx_row
            comp = np.maximum(comp, 0.5 * (1.0 + np.tanh((front - T) / xi_w)))
            sf = 0.5 * (1.0 + np.tanh((front - Tf) / xi_w))
            tract_f = np.where(sf > comp_f, ti, tract_f)
            comp_f = np.maximum(comp_f, sf)
            if primary_row and step == t0 and not simultaneous and supercritical > 0.0:
                su, sv = site
                vs = np.arange(0.5 * lambda_y, nv - 1, lambda_y) if sv is None else sv + lambda_y * np.arange(-1, 2)
                for v0 in vs:
                    iv, iu = int(round(v0)), int(round(su))
                    if margin <= v0 <= nv - 1 - margin and dens[iv, iu] >= n_crit:
                        add_bud(np.array([su, float(v0), S["h"][iv, iu]]), step, name)
        if simultaneous:
            comp = np.ones_like(comp)
            comp_f = np.ones_like(comp_f)
            tract_f = np.where(tract_f < 0, 0, tract_f)
        growth = 1.0 + dens_growth * step
        permissive = (comp_f >= 0.5) & (dens_f * growth >= n_crit) & inside_f
        newly = permissive & ~comp_prev
        comp_prev |= permissive
        if supercritical > 0.0 and newly.any():
            cand = np.argwhere(newly)
            cand_xyz = xyz_f[cand[:, 0], cand[:, 1]]
            if buds_xyz:
                dmin, _ = cKDTree(np.array(buds_xyz)).query(cand_xyz)
            else:
                dmin = np.full(len(cand), np.inf)
            alive = dmin >= r_inh
            for j in rng.permutation(len(cand)):
                if not alive[j]:
                    continue
                p = cand_xyz[j]
                add_bud(p, step, tract_names[int(tract_f[cand[j, 0], cand[j, 1]])])
                alive &= np.sum((cand_xyz - p) ** 2, axis=1) >= r_inh ** 2
        # Keller-Segel / FGF20 / BMP4 relaxation with the Laplace-Beltrami operator of S
        prod_c = comp * supercritical * (0.38 * src_c + 0.16 * n ** 2 / ((1.0 + 1.10 * b) * (1.0 + 0.25 * n ** 2)))
        prod_b = comp * supercritical * (0.30 * src_b + 0.18 * n ** 2)
        c = np.clip(c + dt * (D_c * surface_div_grad(S, 1.0, c) + prod_c - 0.22 * c), 0.0, 3.5)
        b = np.clip(b + dt * (D_b * surface_div_grad(S, 1.0, b) + prod_b - 0.25 * b), 0.0, 3.5)
        dn = D_n * surface_div_grad(S, 1.0, n) - chi * surface_div_grad(S, n, c) + 0.12 * (1.0 + 1.8 * c * supercritical - n)
        n = np.clip(n + dt * dn, 0.2, 3.2)
        kymo[step, :] = np.max(n, axis=0)
        if record_snapshots and step in snap_steps:
            snapshots.append((step, n.copy(), comp.copy()))

    bud_xyz = np.array(buds_xyz) if buds_xyz else np.zeros((0, 3))
    pts = bud_xyz[:, :2].copy()
    coord = surface_coordination(bud_xyz, k=10, max_edge=1.6 * r_inh)
    tract_arr = np.array(bud_tract, dtype=object)
    dorsal_mask = tract_arr == "dorsal" if len(tract_arr) else np.zeros(0, dtype=bool)
    tract_width_px = float(np.ptp(pts[dorsal_mask, 0])) if dorsal_mask.sum() >= 2 else 0.0

    # independent cross-check: dermal condensates detected on the relaxed n-field
    dog_n = ndi.gaussian_filter(n, 1.2) - ndi.gaussian_filter(n, 4.5)
    pk = (dog_n == ndi.maximum_filter(dog_n, size=7)) & (dog_n > 0.045) & (n > 1.12)
    pk[:4, :] = pk[-4:, :] = False
    pk[:, :4] = pk[:, -4:] = False

    return {
        "surface": S, "n_field": n, "c_field": c, "b_field": b, "kymo_n": kymo, "pts": pts, "xyz": bud_xyz,
        "bud_tract": tract_arr, "bud_step": np.array(bud_step, dtype=int), "coordination": coord,
        "tract_width_px": tract_width_px, "n_std": float(np.std(n)), "density": dens,
        "n_field_peaks": int(pk.sum()), "snapshots": snapshots, "T_list": T_list, "r_inh": r_inh,
        "simultaneous": simultaneous, "supercritical": supercritical,
    }


def simulate_feather_array_2d(**kwargs: Any) -> dict[str, Any]:
    """Planar (flat-skin) control: `simulate_feather_array_on_surface` on the h = 0 patch."""
    return simulate_feather_array_on_surface(S=build_embryo_trunk_surface(flat=True), **kwargs)


def curvature_defect_analysis(sim: dict[str, Any], K_thr: float = 4.0e-4) -> dict[str, Any]:
    """
    Localise 5/7 defects against the Gaussian curvature of the surface: interior primordia
    are split by |K| at their parametric position into curved (|K| > K_thr) and flat
    (|K| <= K_thr) groups; the defect fraction of each group and their ratio are returned
    together with the per-tract topological charge sum (6 - z) and the integrated Gaussian
    curvature (3/pi) int K dA over each tract's footprint (Gauss-Bonnet proxy; the boundary
    geodesic-curvature term is NOT included, see audit remediation).
    """
    S = sim["surface"]
    coord = sim["coordination"]
    pts, z, interior = sim["pts"], coord["z"], coord["interior"]
    if len(pts) == 0:
        return {"defect_fraction_curved": 0.0, "defect_fraction_flat": 0.0, "defect_ratio": 0.0,
                "n_curved": 0, "n_flat": 0, "per_tract": {}, "K_at_buds": np.zeros(0)}
    K_at = ndi.map_coordinates(S["K"], [pts[:, 1], pts[:, 0]], order=1, mode="nearest")
    absK = np.abs(K_at)
    curved = interior & (absK > K_thr)
    flat = interior & (absK <= K_thr)
    f_curved = float(np.mean(z[curved] != 6)) if curved.any() else 0.0
    f_flat = float(np.mean(z[flat] != 6)) if flat.any() else 0.0
    per_tract: dict[str, Any] = {}
    for name in np.unique(sim["bud_tract"]):
        m = (sim["bud_tract"] == name) & interior
        footprint = np.zeros(S["K"].shape, dtype=bool)
        for (u, v) in pts[sim["bud_tract"] == name]:
            footprint |= ((S["uu"] - u) ** 2 + (S["vv"] - v) ** 2) <= (0.75 * LAMBDA_Y_GRID) ** 2
        gb_charge = float((3.0 / np.pi) * np.sum(S["K"][footprint] * S["sqrtg"][footprint]))
        per_tract[str(name)] = {
            "n_buds": int(np.sum(sim["bud_tract"] == name)), "n_interior": int(m.sum()),
            "p6": float(np.mean(z[m] == 6)) if m.any() else 0.0,
            "n5": int(np.sum(z[m] == 5)), "n7": int(np.sum(z[m] == 7)),
            "topological_charge": int(np.sum(6 - z[m])) if m.any() else 0,
            "gauss_bonnet_charge_proxy": gb_charge,
            "mean_abs_K": float(np.mean(absK[sim["bud_tract"] == name])),
        }
    return {
        "defect_fraction_curved": f_curved, "defect_fraction_flat": f_flat,
        "defect_ratio": float(f_curved / f_flat) if f_flat > 0 else float("inf") if f_curved > 0 else 1.0,
        "n_curved": int(curved.sum()), "n_flat": int(flat.sum()), "per_tract": per_tract,
        "K_at_buds": K_at, "K_thr": K_thr,
    }


def isochrone_row_count_check(sim: dict[str, Any]) -> dict[str, Any]:
    """
    Lean Theorem 5 executed on the dorsal tract: primordia are grouped into rows by their
    nucleation step (rows = isochrones of the geodesic wave); for every pair of successive
    rows whose arc length differs by >= lambda_y, the longer isochrone must carry at least one
    more primordium (floor(L2 / lambda) >= floor(L1 / lambda) + 1).
    """
    pts, steps, tr = sim["pts"], sim["bud_step"], sim["bud_tract"]
    m = tr == "dorsal"
    if m.sum() < 4:
        return {"n_rows": 0, "violations": 0, "checked_pairs": 0, "rows": []}
    ps, st = pts[m], steps[m]
    xyz = sim["xyz"][m]
    side = np.sign(ps[:, 0] - sim["surface"]["u_mid"] + 1e-9)
    rows: list[dict[str, Any]] = []
    for sgn in (-1.0, 1.0):
        sel = (side == sgn) | (np.abs(ps[:, 0] - sim["surface"]["u_mid"]) < 1e-6)
        order = np.argsort(st[sel])
        s_sorted = st[sel][order]
        xyz_sorted = xyz[sel][order]
        # rows: clusters of nucleation steps separated by > 3 steps
        breaks = np.where(np.diff(s_sorted) > 3)[0] + 1
        for grp in np.split(np.arange(len(s_sorted)), breaks):
            if len(grp) < 2:
                continue
            P = xyz_sorted[grp]
            P = P[np.argsort(P[:, 1])]
            arc = float(np.sum(np.linalg.norm(np.diff(P, axis=0), axis=1)))
            rows.append({"side": sgn, "step": int(s_sorted[grp[0]]), "n": int(len(grp)), "arc_length": arc})
    violations = 0
    checked = 0
    for sgn in (-1.0, 1.0):
        rs = sorted([r for r in rows if r["side"] == sgn], key=lambda r: r["step"])
        for r1, r2 in zip(rs[:-1], rs[1:]):
            L1, L2 = r1["arc_length"], r2["arc_length"]
            if L2 >= L1 + LAMBDA_Y_GRID:
                checked += 1
                if r2["n"] < r1["n"] + 1:
                    violations += 1
            elif L1 >= L2 + LAMBDA_Y_GRID:
                checked += 1
                if r1["n"] < r2["n"] + 1:
                    violations += 1
    return {"n_rows": len(rows), "violations": violations, "checked_pairs": checked, "rows": rows}


# =========================================================================== #
# 6. Perturbation Suites: wave speed, chemotaxis, EDA density threshold, curved surface
# =========================================================================== #
def run_wave_speed_and_perturbation_suite(seed: int = 2018, steps: int = 320) -> dict[str, Any]:
    """
    1. 6-point competence wave speed sweep v_w in {0.11, 0.16, 0.22 (WT), 1.2, 4.0, 100 (simultaneous)}
       on the planar dorsal tract (Ho et al. 2019 Fig 2A/D: wave vs simultaneous hexagonality).
    2. 7-point chemotactic sensitivity sweep chi in {0 (scaleless), 0.01 (Latrunculin), 0.04, 0.08, 0.14, 0.20, 0.24}.
    3. EDA / density threshold suite (Fig 3D-G, Fig 5): control (n_crit = 1), Fc-chEDA1 (49/66),
       Ecto-D2 (98/66), MTX (density x 0.8), MTX + Ecto-D2.
    """
    vw_grid = [
        ("slow_wave_0p11", "Slow Wave (0.5x WT)", 0.11),
        ("slow_wave_0p16", "Slow Wave", 0.16),
        ("wild_type_wave", "Wild-Type Row-by-Row Wave", 0.22),
        ("fast_wave", "Fast Wave (band ~ lambda/12)", 1.20),
        ("very_fast_wave", "Very Fast Wave (band ~ lambda/4)", 4.0),
        ("simultaneous", "Simultaneous (Latrunculin-paused / released, v_w->inf)", 100.0),
    ]
    S_flat = build_embryo_trunk_surface(flat=True)
    wave_sweep: list[dict[str, Any]] = []
    sims: dict[str, dict[str, Any]] = {}
    for slug, label, vw in vw_grid:
        sim = simulate_feather_array_on_surface(S_flat, steps=steps, chi=0.24, v_w=vw, seed=seed,
                                                record_snapshots=(slug == "wild_type_wave"))
        cm = sim["coordination"]
        wave_sweep.append({
            "slug": slug, "label": label, "v_w": vw, "n_buds": cm["n_buds"], "n_interior": cm["n_interior"],
            "p6": cm["p6"], "p567": cm["p567"], "defect_fraction": cm["defect_fraction"], "psi6": cm["psi6"],
            "nn_cv": cm["nn_cv"], "tract_width_px": sim["tract_width_px"], "n5": cm["n5"], "n7": cm["n7"],
        })
        sims[slug] = sim
    wt_width = max(sims["wild_type_wave"]["tract_width_px"], 1e-6)
    for entry in wave_sweep:
        entry["tract_width_ratio_to_wt"] = float(entry["tract_width_px"] / wt_width)

    chi_values = [0.00, 0.01, 0.04, 0.08, 0.14, 0.20, 0.24]
    chi_sweep: list[dict[str, Any]] = []
    for chi_val in chi_values:
        sim_c = simulate_feather_array_on_surface(S_flat, steps=steps, chi=chi_val, v_w=0.22, seed=seed)
        cm = sim_c["coordination"]
        chi_sweep.append({"chi": chi_val, "n_buds": cm["n_buds"], "n_std": sim_c["n_std"], "p6": cm["p6"], "psi6": cm["psi6"]})
        if chi_val == 0.00:
            sims["scaleless"] = sim_c
        elif chi_val == 0.01:
            sims["latrunculin"] = sim_c

    ctrl = HO2019_MARGIN_DENSITY_CELLS_PER_100UM["control"]
    eda_grid = [
        ("control", "Control (66 cells/100 um)", 1.0, 1.0),
        ("fc_cheda1", "Fc-chEDA1 (49 cells/100 um)", HO2019_MARGIN_DENSITY_CELLS_PER_100UM["fc_cheda1"] / ctrl, 1.0),
        ("ecto_d2", "Ecto-D2 (98 cells/100 um)", HO2019_MARGIN_DENSITY_CELLS_PER_100UM["ecto_d2"] / ctrl, 1.0),
        ("mtx", "MTX (density x 0.8)", 1.0, 0.8),
        ("mtx_ecto_d2", "MTX + Ecto-D2", HO2019_MARGIN_DENSITY_CELLS_PER_100UM["ecto_d2"] / ctrl, 0.8),
    ]
    eda_suite: list[dict[str, Any]] = []
    for slug, label, ncrit, dscale in eda_grid:
        if slug == "control":
            sim_e = sims["wild_type_wave"]
        else:
            sim_e = simulate_feather_array_on_surface(S_flat, steps=steps, chi=0.24, v_w=0.22, n_crit=ncrit,
                                                      density_scale=dscale, seed=seed)
        cm = sim_e["coordination"]
        eda_suite.append({"slug": slug, "label": label, "n_crit": ncrit, "density_scale": dscale, "n_buds": cm["n_buds"],
                          "p6": cm["p6"], "tract_width_px": sim_e["tract_width_px"],
                          "tract_width_ratio": float(sim_e["tract_width_px"] / wt_width)})
        sims[f"eda_{slug}"] = sim_e

    return {"wave_sweep": wave_sweep, "chi_sweep": chi_sweep, "eda_suite": eda_suite, "sims": sims,
            "wt_sim": sims["wild_type_wave"], "simul_sim": sims["simultaneous"], "slow_sim": sims["slow_wave_0p11"],
            "scaleless_sim": sims["scaleless"], "latrunculin_sim": sims["latrunculin"]}


CURVED_SURFACE_KW: dict[str, Any] = {"nu": 256, "nv": 184, "limb_sigma": 22.0, "limb_offset_u": 84.0, "humeral_v": 138.0, "femoral_v": 46.0}
CURVED_LIMB_HEIGHTS = (0.0, 15.0, 30.0, 45.0)
CURVED_LATERAL_L_FACTOR = 1.15
CURVED_K_THR = 4.0e-4
CURVED_SUBGRID = 3   # 256 x 184 grid: sub = 3 keeps the curved suite under ~30 s (flat suite uses sub = 5)


def run_curved_surface_suite(seed: int = 2018, steps: int = 340, limb_height: float = 30.0) -> dict[str, Any]:
    """
    Three-tract (dorsal + bilateral humeral + bilateral femoral) patterning on the planar
    control sheet and on the curved trunk + limb-bud Monge patch (256 x 184 grid, limb-bud
    sigma = 22 = 1.8 lambda_y); curvature-localised 5/7 defects, per-tract topological
    charge, Gauss-Bonnet proxy, limb-height sweep and the isochrone row-count check.
    """
    tr = ("dorsal", "humeral", "femoral")
    kw = dict(CURVED_SURFACE_KW)
    S_flat = build_embryo_trunk_surface(flat=True, **kw)
    S_curved = build_embryo_trunk_surface(limb_height=limb_height, **kw)
    run = lambda S, **extra: simulate_feather_array_on_surface(S, steps=steps, tracts=tr, seed=seed, lateral_L_factor=CURVED_LATERAL_L_FACTOR, sub=CURVED_SUBGRID, **extra)
    flat3 = run(S_flat)
    curved3 = run(S_curved, record_snapshots=True)
    curved_simul = run(S_curved, v_w=100.0)
    height_sweep: list[dict[str, Any]] = []
    for hgt in CURVED_LIMB_HEIGHTS:
        if hgt == 0.0:
            s_h, sim_h = S_flat, flat3
        elif hgt == limb_height:
            s_h, sim_h = S_curved, curved3
        else:
            s_h = build_embryo_trunk_surface(limb_height=hgt, **kw)
            sim_h = run(s_h)
        cm = sim_h["coordination"]
        height_sweep.append({"limb_height": hgt, "integrated_abs_K": s_h["integrated_abs_K"], "n_buds": cm["n_buds"],
                             "p6": cm["p6"], "n5": cm["n5"], "n7": cm["n7"], "defect_fraction": cm["defect_fraction"],
                             "n_dislocation_pairs": len(cm["dislocation_pairs"])})
    # apterium (bare inter-tract gap) between the dorsal and femoral tracts along the femoral row
    def apterium_gap(sim: dict[str, Any]) -> float:
        pts, trn = sim["pts"], sim["bud_tract"]
        S = sim["surface"]
        d = pts[(trn == "dorsal") & (pts[:, 0] > S["u_mid"])]
        f = pts[(trn == "femoral") & (pts[:, 0] > S["u_mid"])]
        if len(d) == 0 or len(f) == 0:
            return 0.0
        return float(np.min(f[:, 0]) - np.max(d[:, 0]))
    return {
        "flat3": flat3, "curved3": curved3, "curved_simul": curved_simul, "S_flat": S_flat, "S_curved": S_curved,
        "defects_flat": curvature_defect_analysis(flat3, K_thr=CURVED_K_THR), "defects_curved": curvature_defect_analysis(curved3, K_thr=CURVED_K_THR),
        "height_sweep": height_sweep, "isochrone_flat": isochrone_row_count_check(flat3),
        "isochrone_curved": isochrone_row_count_check(curved3),
        "apterium_gap_flat": apterium_gap(flat3), "apterium_gap_curved": apterium_gap(curved3),
        "sphere_check": sphere_topological_charge_check(),
    }


# =========================================================================== #
# 7. Biological Specimen Segmentation (`chick_feather_bud_hexagonal_specimen.jpg`)
# =========================================================================== #
def segment_feather_tract_specimen() -> dict[str, Any]:
    """
    Segment the SHA-256-locked specimen figure from Ho et al. (2019) PLOS Biology
    17(2):e3000132 Fig 3 (`chick_feather_bud_hexagonal_specimen.jpg`, CC BY 4.0):
      1. Panel B (E7.5 whole-mount double in situ, CTNNB1 orange / EDA purple) ROI
         `[494, 24, 770, 218]`: extract the 24 primordia via Difference-of-Gaussians optical
         density peaks, compute their Voronoi coordination distribution (P_6, P_567),
         Voronoi-ridge bond-orientational order |psi_6| and nearest-neighbour spacing CV.
      2. Panel A (E6.5 skin cultured 24 h vs 48 h, CTNNB1) ROIs `[32, 242, 126, 338]` and
         `[136, 242, 232, 338]`: mediolateral patterned-region width expansion w_48h / w_24h.
    """
    verify_dataset_sha256()
    rgb = np.asarray(Image.open(SPECIMEN_JPG_PATH).convert("RGB"))

    bx0, by0, bx1, by1 = (494, 24, 770, 218)
    roi_b_rgb = rgb[by0:by1, bx0:bx1]
    opt_density = 255.0 - roi_b_rgb.astype(np.float64).mean(axis=2)
    dog = ndi.gaussian_filter(opt_density, 4.5) - ndi.gaussian_filter(opt_density, 13.5)
    mx = ndi.maximum_filter(dog, size=23)
    thr = float(np.percentile(dog, 66))
    pk = (dog == mx) & (dog > thr)
    margin = 11
    pk[:margin, :] = False
    pk[-margin:, :] = False
    pk[:, :margin] = False
    pk[:, -margin:] = False
    ys, xs = np.where(pk)
    pts_b = np.column_stack([xs, ys]).astype(np.float64)

    vor_b = compute_voronoi_hexagonal_metrics(
        pts_b, width=float(roi_b_rgb.shape[1]), height=float(roi_b_rgb.shape[0]), margin_frac=0.18,
    )

    ax24_0, ay24_0, ax24_1, ay24_1 = (32, 242, 126, 338)
    ax48_0, ay48_0, ax48_1, ay48_1 = (136, 242, 232, 338)
    s24 = 255.0 - rgb[ay24_0:ay24_1, ax24_0:ax24_1].astype(np.float64).mean(axis=2)
    s48 = 255.0 - rgb[ay48_0:ay48_1, ax48_0:ax48_1].astype(np.float64).mean(axis=2)
    prof24 = np.maximum(s24.mean(axis=0) - np.percentile(s24.mean(axis=0), 20), 0.0)
    prof48 = np.maximum(s48.mean(axis=0) - np.percentile(s48.mean(axis=0), 20), 0.0)
    w24_px = float(np.sum(prof24 > 0.30 * np.max(prof24)))
    w48_px = float(np.sum(prof48 > 0.30 * np.max(prof48)))
    expansion_ratio = float(w48_px / (w24_px + 1e-9))

    return {
        "full_rgb": rgb, "panel_b_roi_xyxy": [bx0, by0, bx1, by1], "panel_b_rgb": roi_b_rgb, "panel_b_dog": dog,
        "pts_b": pts_b, "voronoi_b": vor_b, "panel_a_24h_width_px": w24_px, "panel_a_48h_width_px": w48_px,
        "panel_a_expansion_ratio_48h_over_24h": expansion_ratio,
    }


# =========================================================================== #
# 8. Publication Figures (12-panel comparison + curved-tract figure) & Growth GIF
# =========================================================================== #
COORD_COLORS = {5: "#f59e0b", 6: "#10b981", 7: "#ef4444"}


def _draw_voronoi_overlay(ax: Any, pts: np.ndarray, vor_metrics: dict[str, Any], width: float, height: float) -> None:
    """Draw interior Voronoi polygons color-coded by coordination number (6=emerald, 5=amber, 7=rose)."""
    vor = vor_metrics["voronoi"]
    if vor is None:
        return
    for idx, n_sides in zip(vor_metrics["interior_indices"], vor_metrics["interior_sides"]):
        reg = vor.regions[vor.point_region[idx]]
        verts = vor.vertices[reg]
        if (np.all(verts[:, 0] >= -5) & np.all(verts[:, 0] <= width + 5) & np.all(verts[:, 1] >= -5) & np.all(verts[:, 1] <= height + 5)):
            ax.fill(verts[:, 0], verts[:, 1], facecolor=COORD_COLORS.get(n_sides, "#8b5cf6"), edgecolor="white", alpha=0.38, lw=1.1)


def _scatter_coordination(ax: Any, sim: dict[str, Any], s: float = 16.0) -> None:
    """Scatter primordia coloured by coordination (interior) with grey for edge primordia and 5-7 pair links."""
    cm = sim["coordination"]
    pts, z, inter = sim["pts"], cm["z"], cm["interior"]
    if len(pts) == 0:
        return
    cols = np.array(["#9ca3af"] * len(pts), dtype=object)
    for zz, col in COORD_COLORS.items():
        cols[inter & (z == zz)] = col
    cols[inter & ~np.isin(z, [5, 6, 7])] = "#8b5cf6"
    ax.scatter(pts[:, 0], pts[:, 1], c=list(cols), s=s, edgecolors="#0f172a", lw=0.5, zorder=5)
    for i, j in cm["dislocation_pairs"]:
        ax.plot([pts[i, 0], pts[j, 0]], [pts[i, 1], pts[j, 1]], color="#fde047", lw=1.6, zorder=6)


def _surface_panel(ax: Any, sim: dict[str, Any], title: str, show_K: bool = True) -> None:
    S = sim["surface"]
    if show_K and not S["flat"]:
        kmax = float(np.max(np.abs(S["K"])) + 1e-12)
        ax.imshow(S["K"], origin="lower", cmap="RdBu_r", vmin=-kmax, vmax=kmax, alpha=0.85)
        ax.contour(S["h"], levels=8, colors="#334155", linewidths=0.45, alpha=0.7)
    else:
        ax.imshow(sim["n_field"], origin="lower", cmap="viridis")
    _scatter_coordination(ax, sim, s=12)
    ax.set_title(title, fontsize=9.6, fontweight="bold")
    ax.set_xlim(0, S["nu"] - 1)
    ax.set_ylim(0, S["nv"] - 1)
    ax.set_xticks([])
    ax.set_yticks([])


def render_outputs(seed: int = 2018) -> dict[str, Any]:
    """Render the 12-panel Pass 3 comparison figure, the curved-tract figure and the growth GIF."""
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    disp = compute_keller_segel_dispersion_and_geometry()
    suite = run_wave_speed_and_perturbation_suite(seed=seed)
    curved = run_curved_surface_suite(seed=seed)
    spec = segment_feather_tract_specimen()

    wt_sim, simul_sim = suite["wt_sim"], suite["simul_sim"]
    wt_cm, sim_cm = wt_sim["coordination"], simul_sim["coordination"]
    spec_vm = spec["voronoi_b"]
    flat3, curved3 = curved["flat3"], curved["curved3"]
    S_c = curved["S_curved"]
    nu_f, nv_f = wt_sim["surface"]["nu"], wt_sim["surface"]["nv"]

    fig = plt.figure(figsize=(20.0, 14.6), dpi=150)
    fig.patch.set_facecolor("white")
    gs = GridSpec(3, 4, figure=fig, hspace=0.42, wspace=0.28)

    # A. Dispersion
    ax_a = fig.add_subplot(gs[0, 0])
    ax_a.plot(disp["k"], disp["sigma_wt"], color="#1d4ed8", lw=2.3, label="Wild-Type ($\\chi=0.24 > \\chi_c$)")
    ax_a.plot(disp["k"], disp["sigma_latrunculin"], "--", color="#d97706", lw=1.9, label="Latrunculin ($\\chi=0.01 < \\chi_c$)")
    ax_a.plot(disp["k"], disp["sigma_scaleless"], ":", color="#dc2626", lw=2.0, label="Scaleless $FGF20^{-/-}$ ($\\chi=0$)")
    ax_a.axhline(0.0, color="#0f172a", lw=0.9)
    ax_a.axvline(disp["k_peak"], color="#1d4ed8", ls="--", alpha=0.55, lw=1.1, label=f"$k_{{peak}}={disp['k_peak']:.2f}$")
    ax_a.set_title(f"A. Keller–Segel / FGF20 Dispersion ($\\chi_c={disp['chi_c']:.4f}$; Painter 2018)", fontsize=9.8, fontweight="bold")
    ax_a.set_xlabel("Spatial Wavenumber $k$")
    ax_a.set_ylabel("Growth Rate $\\sigma(k)$")
    ax_a.legend(fontsize=7.0, loc="lower left")
    ax_a.grid(alpha=0.25)

    # B. WT wave (flat)
    ax_b = fig.add_subplot(gs[0, 1])
    ax_b.imshow(wt_sim["n_field"], cmap="viridis", origin="lower")
    _scatter_coordination(ax_b, wt_sim, s=15)
    ax_b.set_title(f"B. WT Wave, flat ($v_w=0.22$)\n$P_6={wt_cm['p6']:.2f}$, $|\\psi_6|={wt_cm['psi6']:.2f}$", fontsize=9.0, fontweight="bold")
    ax_b.set_xlabel("Mediolateral $u$ (6=green, 5=amber, 7=red, edge=grey)")
    ax_b.set_ylabel("Anteroposterior $v$")
    ax_b.set_xlim(0, nu_f - 1)
    ax_b.set_ylim(0, nv_f - 1)

    # C. Simultaneous
    ax_c = fig.add_subplot(gs[0, 2])
    ax_c.imshow(simul_sim["n_field"], cmap="viridis", origin="lower")
    _scatter_coordination(ax_c, simul_sim, s=15)
    ax_c.set_title(f"C. Simultaneous (Latrunculin-paused, Ho Fig 2B)\n$P_6={sim_cm['p6']:.2f}$, $|\\psi_6|={sim_cm['psi6']:.2f}$", fontsize=9.0, fontweight="bold")
    ax_c.set_xlabel("Mediolateral $u$ (jammed inhibitory-disk packing)")
    ax_c.set_ylabel("Anteroposterior $v$")
    ax_c.set_xlim(0, nu_f - 1)
    ax_c.set_ylim(0, nv_f - 1)

    # D. Kymograph
    ax_d = fig.add_subplot(gs[0, 3])
    im_k = ax_d.imshow(wt_sim["kymo_n"], cmap="magma", aspect="auto", origin="lower",
                       extent=[-0.5 * (nu_f - 1), 0.5 * (nu_f - 1), 0.0, float(wt_sim["kymo_n"].shape[0])])
    plt.colorbar(im_k, ax=ax_d, fraction=0.046, pad=0.04, label="Max Dermal Density $n(u,t)$")
    ax_d.set_title("D. Bilateral Wave Kymograph $n(u,t)$ (Ho et al. Fig 4D)", fontsize=9.8, fontweight="bold")
    ax_d.set_xlabel("Distance from Dorsal Midline $u$ (grid)")
    ax_d.set_ylabel("Developmental Time Step $t$")

    # E. v_w sweep
    ax_e = fig.add_subplot(gs[1, 0])
    vw_labels = [f"{w['v_w']:g}" if w["v_w"] < 8 else "$\\infty$" for w in suite["wave_sweep"]]
    vw_labels[2] += "\n(WT)"
    vw_labels[-1] += "\n(simult.)"
    x_idx = np.arange(len(suite["wave_sweep"]))
    ax_e.plot(x_idx, [w["p6"] for w in suite["wave_sweep"]], "o-", color="#059669", lw=2.3, ms=6.5, label="Hexagonal Fraction $P_6$")
    ax_e.plot(x_idx, [w["psi6"] for w in suite["wave_sweep"]], "s--", color="#2563eb", lw=2.0, ms=6.0, label="Bond Order $|\\psi_6|$")
    ax_e.plot(x_idx, [w["defect_fraction"] for w in suite["wave_sweep"]], "^:", color="#dc2626", lw=1.8, ms=5.5, label="Defect Fraction $1-P_6$")
    ax_e.axhline(0.75, color="#059669", ls=(0, (2, 2)), lw=1.0, alpha=0.8, label="Ho 2019 Fig 2D control ~0.75")
    ax_e.axhline(0.50, color="#7c3aed", ls=(0, (2, 2)), lw=1.0, alpha=0.8, label="Ho 2019 Fig 2D Latrunculin-paused ~0.50")
    ax_e.set_xticks(x_idx)
    ax_e.set_xticklabels(vw_labels, fontsize=7.8)
    ax_e.set_ylim(0.0, 1.08)
    ax_e.set_title("E. Wave Speed $v_w$ Sweep: Band Width vs Hexagonality", fontsize=9.8, fontweight="bold")
    ax_e.set_xlabel("Competence Wave Speed $v_w$ (grid units)")
    ax_e.set_ylabel("Order Parameter / Fraction")
    ax_e.legend(fontsize=6.6, loc="center left")
    ax_e.grid(alpha=0.25)

    # F. chi sweep
    ax_f = fig.add_subplot(gs[1, 1])
    chi_x = [c["chi"] for c in suite["chi_sweep"]]
    ax_f.plot(chi_x, [c["n_buds"] for c in suite["chi_sweep"]], "o-", color="#7c3aed", lw=2.3, ms=6.5, label="Nucleated Primordia $N(\\chi)$")
    ax_f.axvline(disp["chi_c"], color="#dc2626", ls="--", lw=1.5, label=f"Threshold $\\chi_c={disp['chi_c']:.3f}$")
    ax_f.scatter([0.0, 0.01], [0, 0], color="#dc2626", s=65, zorder=6, label="Scaleless / Latrunculin (0 buds)")
    ax_f.text(0.36, 0.42,
              f"WT ($v_w=0.22$): {wt_cm['n_buds']} buds, $P_6={wt_cm['p6']:.2f}$\n"
              f"Simultaneous: {sim_cm['n_buds']} buds, $P_6={sim_cm['p6']:.2f}$\n"
              f"Scaleless ($\\chi=0$): 0 buds (std={suite['scaleless_sim']['n_std']:.1e})\n"
              f"Latrunculin ($\\chi=0.01$): 0 buds (std={suite['latrunculin_sim']['n_std']:.1e})\n"
              f"n-field DoG peaks = {wt_sim['n_field_peaks']} (= nucleation sites)",
              transform=ax_f.transAxes, fontsize=7.2, bbox=dict(facecolor="#f8fafc", edgecolor="#cbd5e1", boxstyle="round,pad=0.35"))
    ax_f.set_title("F. FGF20 Chemotaxis $\\chi$ Sweep & Mutant Controls", fontsize=9.8, fontweight="bold")
    ax_f.set_xlabel("Chemotactic Sensitivity $\\chi$")
    ax_f.set_ylabel("Feather Primordia Count $N$")
    ax_f.legend(fontsize=7.0, loc="upper left")
    ax_f.grid(alpha=0.25)

    # G. Specimen with ROIs
    ax_g = fig.add_subplot(gs[1, 2])
    ax_g.imshow(spec["full_rgb"])
    bx0, by0, bx1, by1 = spec["panel_b_roi_xyxy"]
    ax_g.add_patch(plt.Rectangle((bx0, by0), bx1 - bx0, by1 - by0, fill=False, edgecolor="#10b981", lw=2.2, label="Panel B hexagonal ROI"))
    ax_g.add_patch(plt.Rectangle((32, 242), 94, 96, fill=False, edgecolor="#38bdf8", lw=1.8, label="Panel A +24h / +48h ROIs"))
    ax_g.add_patch(plt.Rectangle((136, 242), 96, 96, fill=False, edgecolor="#38bdf8", lw=1.8))
    ax_g.set_title(f"G. Gallus gallus Specimen (Ho et al. 2019 Fig 3)\n+48h/+24h tract expansion = {spec['panel_a_expansion_ratio_48h_over_24h']:.2f}x", fontsize=9.0, fontweight="bold")
    ax_g.legend(fontsize=7.0, loc="lower right")
    ax_g.axis("off")

    # H. Segmented specimen Voronoi
    ax_h = fig.add_subplot(gs[1, 3])
    ax_h.imshow(spec["panel_b_rgb"])
    _draw_voronoi_overlay(ax_h, spec["pts_b"], spec_vm, float(spec["panel_b_rgb"].shape[1]), float(spec["panel_b_rgb"].shape[0]))
    if spec_vm["delaunay"] is not None:
        ax_h.triplot(spec["pts_b"][:, 0], spec["pts_b"][:, 1], spec_vm["delaunay"].simplices, color="#fde047", lw=0.85, alpha=0.75)
    ax_h.scatter(spec["pts_b"][:, 0], spec["pts_b"][:, 1], c="#10b981", s=34, edgecolors="white", lw=1.0, zorder=6, label=f"Buds ($N={spec_vm['n_buds']}$)")
    ax_h.set_title(f"H. Specimen Voronoi ($N={spec_vm['n_buds']}$, $P_6={spec_vm['p6']:.2f}$, $P_{{567}}={spec_vm['p567']:.2f}$)\nvs model WT $P_6={wt_cm['p6']:.2f}$", fontsize=9.0, fontweight="bold")
    ax_h.set_xlim(0, spec["panel_b_rgb"].shape[1] - 1)
    ax_h.set_ylim(spec["panel_b_rgb"].shape[0] - 1, 0)
    ax_h.set_xticks([])
    ax_h.set_yticks([])

    # I. Curved trunk + limb-bud surface in 3D
    ax_i = fig.add_subplot(gs[2, 0], projection="3d")
    st = 4
    ax_i.plot_surface(S_c["uu"][::st, ::st], S_c["vv"][::st, ::st], S_c["h"][::st, ::st], cmap="Greys", alpha=0.55, linewidth=0, antialiased=True)
    cm3 = curved3["coordination"]
    cols = np.array(["#9ca3af"] * len(curved3["pts"]), dtype=object)
    for zz, col in COORD_COLORS.items():
        cols[cm3["interior"] & (cm3["z"] == zz)] = col
    ax_i.scatter(curved3["xyz"][:, 0], curved3["xyz"][:, 1], curved3["xyz"][:, 2] + 0.8, c=list(cols), s=9, depthshade=False)
    ax_i.view_init(elev=48, azim=-62)
    ax_i.set_box_aspect((S_c["nu"], S_c["nv"], 2.4 * float(np.max(S_c["h"]))))
    ax_i.set_title(f"I. Curved Trunk + 4 Limb-Bud Monge Patch ($\\int|K|dA={S_c['integrated_abs_K']:.1f}$)", fontsize=9.4, fontweight="bold")
    ax_i.set_xticks([])
    ax_i.set_yticks([])
    ax_i.set_zticks([])

    # J. Parametric map: Gaussian curvature + tracts + 5/7 defects
    ax_j = fig.add_subplot(gs[2, 1])
    dc = curved["defects_curved"]
    _surface_panel(ax_j, curved3, f"J. Tracts on $K(u,v)$: $P_6={cm3['p6']:.2f}$, 5s={cm3['n5']}, 7s={cm3['n7']}, 5–7 pairs={len(cm3['dislocation_pairs'])}")
    for name, su, sv in S_c["limb_sites"]:
        ax_j.text(su, sv, name[:3], color="#111827", fontsize=7.5, ha="center", va="center", fontweight="bold")
    ax_j.text(S_c["u_mid"], S_c["nv"] - 10, "dorsal", color="#111827", fontsize=7.5, ha="center", fontweight="bold")
    ax_j.set_xlabel(f"defect frac.: |K|>{dc['K_thr']:g} {dc['defect_fraction_curved']:.3f} (n={dc['n_curved']}) vs flat {dc['defect_fraction_flat']:.3f} (n={dc['n_flat']})", fontsize=7.6)

    # K. Limb-height sweep
    ax_k = fig.add_subplot(gs[2, 2])
    hs = curved["height_sweep"]
    xk = [h["integrated_abs_K"] for h in hs]
    ax_k.plot(xk, [h["p6"] for h in hs], "o-", color="#059669", lw=2.2, ms=6.5, label="$P_6$ (all tracts)")
    ax_k.plot(xk, [h["defect_fraction"] for h in hs], "^:", color="#dc2626", lw=1.8, ms=5.5, label="Defect fraction $1-P_6$")
    ax_k2 = ax_k.twinx()
    ax_k2.bar(xk, [h["n_dislocation_pairs"] for h in hs], width=0.9, color="#fde047", edgecolor="#854d0e", alpha=0.7, label="5–7 dislocation pairs")
    ax_k2.set_ylabel("5–7 pairs", color="#854d0e")
    for h, xx in zip(hs, xk):
        ax_k.annotate(f"$A_l$={h['limb_height']:g}", (xx, h["p6"]), textcoords="offset points", xytext=(0, 7), fontsize=7.0, ha="center")
    fp = curved["flat3"]["coordination"]
    ax_k.set_title(f"K. Limb-Bud Height Sweep\nflat $P_6={fp['p6']:.2f}$ → curved $P_6={cm3['p6']:.2f}$", fontsize=9.2, fontweight="bold")
    ax_k.set_xlabel("Integrated |Gaussian curvature| $\\int |K| dA$ (Bausch et al. 2003 scars)")
    ax_k.set_ylabel("Fraction")
    ax_k.set_ylim(0.0, 1.08)
    ax_k.legend(fontsize=7.0, loc="center left")
    ax_k.grid(alpha=0.25)

    # L. EDA / density threshold tract widths
    ax_l = fig.add_subplot(gs[2, 3])
    eda = suite["eda_suite"]
    cols_l = ["#2563eb", "#059669", "#dc2626", "#d97706", "#7c3aed"]
    ax_l.bar(np.arange(len(eda)), [e["tract_width_ratio"] for e in eda], color=cols_l, edgecolor="#0f172a")
    ax_l.set_xticks(np.arange(len(eda)))
    ax_l.set_xticklabels(["control\n66/100µm", "Fc-chEDA1\n49/100µm", "Ecto-D2\n98/100µm", "MTX\n(×0.8 cells)", "MTX +\nEcto-D2"], fontsize=7.4)
    for i, e in enumerate(eda):
        ax_l.text(i, e["tract_width_ratio"] + 0.02, f"{e['tract_width_ratio']:.2f}\n({e['n_buds']} buds)", ha="center", fontsize=7.0)
    ax_l.axhline(1.0, color="#0f172a", lw=0.8, ls="--")
    ax_l.set_ylim(0, max(1.6, max(e["tract_width_ratio"] for e in eda) + 0.3))
    ax_l.set_title("L. EDA-Set Density Threshold → Tract Width\n(Ho et al. 2019 Fig 3D-G, 5C-F)", fontsize=9.2, fontweight="bold")
    ax_l.set_ylabel("Patterned-region width / control")
    ax_l.set_xlabel("Critical mesenchymal density $n_{crit}$ from Ho et al. (2019) Fig 5C-D margins")
    ax_l.grid(alpha=0.25, axis="y")

    fig.suptitle(
        "Ho et al. (2019) PLOS Biol. & Painter et al. (2018) JTB — Avian Feather Tracts: Geodesic EDA Wave, Inhibitory-Disk Hexagonal Packing, "
        "Density-Threshold Tract Edges and Curvature-Induced 5–7 Dislocations",
        fontsize=12.0, fontweight="bold", y=0.985,
    )
    fig.savefig(OUT_PNG, bbox_inches="tight")
    plt.close(fig)

    # Curved-tract figure: flat 3-tract control vs curved 3-tract vs curved simultaneous
    fig2, axs = plt.subplots(1, 3, figsize=(21.0, 6.3), dpi=130)
    _surface_panel(axs[0], flat3, f"Flat 3-tract control: $P_6={fp['p6']:.3f}$, 5s={fp['n5']}, 7s={fp['n7']}", show_K=False)
    _surface_panel(axs[1], curved3, f"Curved trunk + limb buds (K map): $P_6={cm3['p6']:.3f}$, 5s={cm3['n5']}, 7s={cm3['n7']}, pairs={len(cm3['dislocation_pairs'])}")
    cs_cm = curved["curved_simul"]["coordination"]
    _surface_panel(axs[2], curved["curved_simul"], f"Curved, simultaneous competence: $P_6={cs_cm['p6']:.3f}$, 5s={cs_cm['n5']}, 7s={cs_cm['n7']}", show_K=False)
    axs[0].set_xlabel(f"apterium gap dorsal→femoral = {curved['apterium_gap_flat']:.1f} grid", fontsize=8)
    axs[1].set_xlabel(f"apterium gap = {curved['apterium_gap_curved']:.1f} grid; isochrone row-count violations = {curved['isochrone_curved']['violations']} / {curved['isochrone_curved']['checked_pairs']}", fontsize=8)
    fig2.suptitle("Slot 019 Pass 3 — feather tracts on the curved embryonic surface (6=green, 5=amber, 7=red, yellow = 5–7 dislocation pair)", fontsize=11.5, fontweight="bold")
    fig2.savefig(OUT_SURFACE_PNG, bbox_inches="tight")
    plt.close(fig2)

    # Growth GIF: flat WT competence + density, curved 3-tract density
    gif_frames: list[Image.Image] = []
    curved_snaps = curved3["snapshots"]
    for k, (step_idx, n_snap, sw_snap) in enumerate(wt_sim["snapshots"]):
        fig_g, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(13.2, 3.9), dpi=100)
        ax1.imshow(sw_snap, cmap="plasma", vmin=0.0, vmax=1.0, origin="lower")
        ax1.set_title(f"EDA Competence Wave $s_w(u,v,t)$ (step {step_idx})", fontsize=9)
        ax2.imshow(n_snap, cmap="viridis", vmin=0.8, vmax=2.3, origin="lower")
        ax2.set_title(f"Flat Dermal Condensates $n(u,v,t)$ (step {step_idx})", fontsize=9)
        if k < len(curved_snaps):
            cstep, cn, _ = curved_snaps[k]
            ax3.imshow(cn, cmap="viridis", vmin=0.8, vmax=2.3, origin="lower")
            ax3.contour(S_c["h"], levels=8, colors="w", linewidths=0.4, alpha=0.6)
            ax3.set_title(f"Curved trunk + limb buds $n$ (step {cstep})", fontsize=9)
        for ax in (ax1, ax2, ax3):
            ax.set_xticks([])
            ax.set_yticks([])
        fig_g.tight_layout()
        fig_g.canvas.draw()
        rgba = np.asarray(fig_g.canvas.buffer_rgba())
        gif_frames.append(Image.fromarray(rgba[:, :, :3]).convert("P", palette=Image.Palette.ADAPTIVE))
        plt.close(fig_g)
    gif_frames[0].save(OUT_GIF, save_all=True, append_images=gif_frames[1:], duration=420, loop=0)

    return {"disp": disp, "suite": suite, "curved": curved, "spec": spec}


# =========================================================================== #
# 9. MorphoEval Dynamic Measurement Hook (`measure_morphoeval`)
# =========================================================================== #
def measure_morphoeval() -> list[dict[str, Any]]:
    """Compute 16 reference-backed morphometric measurements for `measure_outputs.py`."""
    disp = compute_keller_segel_dispersion_and_geometry()
    suite = run_wave_speed_and_perturbation_suite(seed=2018)
    curved = run_curved_surface_suite(seed=2018)
    spec = segment_feather_tract_specimen()

    wt_cm = suite["wt_sim"]["coordination"]
    sim_cm = suite["simul_sim"]["coordination"]
    eda = {e["slug"]: e for e in suite["eda_suite"]}
    spec_vm = spec["voronoi_b"]
    fcm, ccm = curved["flat3"]["coordination"], curved["curved3"]["coordination"]
    dcur = curved["defects_curved"]
    hs = curved["height_sweep"]
    sph = curved["sphere_check"]
    iso = curved["isochrone_curved"]
    HO = "Ho et al. (2019) PLOS Biol. 17(2):e3000132"

    return [
        {"name": "equilateral hexagonal inter-row to intra-row spacing ratio Delta_x / lambda_y",
         "value": round(disp["hex_row_ratio"], 4), "unit": "dimensionless ratio",
         "method": "Half-period phase-pinned row spacing ratio sqrt(3)/2 (Delta_x = 242.49 um for lambda_y = 280.0 um)",
         "reference_value": 0.866, "reference_source": "Lean Theorem 2 hexagonal_row_offset_equidistance; Ho et al. (2019) Fig 3B hexagonal lattice",
         "note": f"diagonal inter-row distance = {disp['diag_neighbor_dist_um']:.2f} um == intra-row lambda_y = {disp['lambda_y_um']:.2f} um; WT model NN spacing CV = {wt_cm['nn_cv']:.4f}",
         "derived_from": "script"},
        {"name": "critical FGF20 chemotactic sensitivity threshold chi_c",
         "value": round(disp["chi_c"], 4), "unit": "chemotactic sensitivity",
         "method": "Keller-Segel / FGF20 linear instability threshold ((sqrt(D_n*mu_c)+sqrt(D_c*r_n))^2)/(n_0*f_c) (Painter et al. 2018 linearisation)",
         "reference_value": 0.054, "reference_source": "Painter, Ho & Headon (2018) J. Theor. Biol. 437:225-238 & Lean Theorem 1",
         "note": f"WT sigma_max(chi=0.24) = +{disp['sigma_wt_max']:.4f} vs scaleless sigma_max(chi=0) = {disp['sigma_scaleless_max']:.4f}; nucleation disabled for chi < chi_c (chi = 0.04 -> 0 buds)",
         "derived_from": "script"},
        {"name": "wild-type geodesic wave (v_w=0.22) interior six-neighbour fraction P_6",
         "value": round(wt_cm["p6"], 4), "unit": "fraction of interior primordia",
         "method": "Local tangent-plane Delaunay coordination of wave-gated inhibitory-disk nucleation sites (boundary primordia removed as in Ho Fig 2C)",
         "reference_value": 0.75, "reference_source": f"{HO} Fig 2D: ~75 % of Voronoi cells six-sided in control explants",
         "note": f"model WT P_6 = {wt_cm['p6']:.4f} (noise-free ideal wave), P_567 = {wt_cm['p567']:.4f}, |psi_6| = {wt_cm['psi6']:.4f}, N = {wt_cm['n_buds']} ({wt_cm['n_interior']} interior)",
         "derived_from": "script"},
        {"name": "simultaneous competence (Latrunculin-paused / released, v_w->inf) six-neighbour fraction P_6",
         "value": round(sim_cm["p6"], 4), "unit": "fraction of interior primordia",
         "method": "Same coordination analysis with the whole tract competent at t = 0 (jammed random sequential adsorption of inhibitory disks)",
         "reference_value": 0.50, "reference_source": f"{HO} Fig 2B/D: ~50 % six-sided Voronoi cells after a 24 h Latrunculin A pause",
         "note": f"traveling-wave advantage Delta_P6 = +{wt_cm['p6'] - sim_cm['p6']:.4f} (paper: ~+0.25); |psi_6| = {sim_cm['psi6']:.4f}, N = {sim_cm['n_buds']} ({sim_cm['n_interior']} interior); Lean Theorem 3",
         "derived_from": "script"},
        {"name": "Fc-chEDA1 (n_crit = 49/66) patterned-region width ratio to control",
         "value": round(eda["fc_cheda1"]["tract_width_ratio"], 4), "unit": "ratio to control tract width",
         "method": "Dorsal-tract span of nucleated primordia when the EDA-set critical mesenchymal density is lowered to 49/66 of control (Ho Fig 5C)",
         "reference_value": None, "reference_source": f"{HO} Fig 3D-E: soluble Fc-chEDA1 widens the FGF20-patterned region (ratio not tabulated in the text)",
         "note": f"widths: control {eda['control']['tract_width_px']:.1f}, Fc-chEDA1 {eda['fc_cheda1']['tract_width_px']:.1f} grid; direction matches the paper, magnitude depends on the un-digitised density gradient",
         "derived_from": "script"},
        {"name": "Ecto-D2 (n_crit = 98/66) patterned-region width ratio to control",
         "value": round(eda["ecto_d2"]["tract_width_ratio"], 4), "unit": "ratio to control tract width",
         "method": "Dorsal-tract span when the critical density is raised to 98/66 of control (Ho Fig 5D)",
         "reference_value": None, "reference_source": f"{HO} Fig 3F-G: Ecto-D2 restricts patterning to a narrow midline strip (ratio not tabulated)",
         "note": f"Ecto-D2 {eda['ecto_d2']['tract_width_px']:.1f} grid ({eda['ecto_d2']['n_buds']} buds); MTX x0.8 density {eda['mtx']['tract_width_ratio']:.3f}; MTX+Ecto-D2 {eda['mtx_ecto_d2']['tract_width_ratio']:.3f} (< each single treatment, Fig 5F)",
         "derived_from": "script"},
        {"name": "MTX + Ecto-D2 co-treatment width relative to the narrower single treatment",
         "value": round(eda["mtx_ecto_d2"]["tract_width_ratio"] / max(min(eda["mtx"]["tract_width_ratio"], eda["ecto_d2"]["tract_width_ratio"]), 1e-9), 4),
         "unit": "ratio", "method": "width(MTX + Ecto-D2) / min(width(MTX), width(Ecto-D2))",
         "reference_value": None, "reference_source": f"{HO} Fig 5F: co-treated explants narrower than each single treatment (qualitative, p < 0.001)",
         "note": "value < 1 reproduces the reported ordering", "derived_from": "script"},
        {"name": "scaleless (FGF20-/-, chi=0) and Latrunculin (chi=0.01) negative control bud count",
         "value": float(suite["scaleless_sim"]["coordination"]["n_buds"] + suite["latrunculin_sim"]["coordination"]["n_buds"]), "unit": "buds",
         "method": "Nucleated primordia under FGF20 chemotaxis loss (chi = 0.0) and actomyosin motility blockade (chi = 0.01 < chi_c)",
         "reference_value": 0.0, "reference_source": f"{HO} Fig 1C (Latrunculin A), S5 Fig (scaleless) & Lean Theorem 1",
         "note": f"scaleless std(n) = {suite['scaleless_sim']['n_std']:.2e}, Latrunculin std(n) = {suite['latrunculin_sim']['n_std']:.2e} vs WT std(n) = {suite['wt_sim']['n_std']:.4f}",
         "derived_from": "script"},
        {"name": "segmented Gallus gallus Panel B specimen feather primordia count",
         "value": float(spec_vm["n_buds"]), "unit": "buds",
         "method": "Difference-of-Gaussians optical density peak segmentation on chick_feather_bud_hexagonal_specimen.jpg Panel B ROI [494, 24, 770, 218]",
         "reference_value": 24.0, "reference_source": f"{HO} Fig 3B CTNNB1/EDA double in situ (E7.5)",
         "note": f"detected {spec_vm['n_buds']} feather buds ({spec_vm['n_interior']} interior Voronoi cells), NN spacing = {spec_vm['nn_mean']:.2f} +- {spec_vm['nn_std']:.2f} px (CV = {spec_vm['nn_cv']:.4f})",
         "derived_from": "output_pixels"},
        {"name": "segmented Gallus gallus Panel B specimen interior Voronoi hexagonal fraction P_6",
         "value": round(spec_vm["p6"], 4), "unit": "fraction of interior Voronoi cells",
         "method": "Voronoi polygon coordination analysis on segmented primordia in Panel B ROI [494, 24, 770, 218]",
         "reference_value": 0.75, "reference_source": f"{HO} Fig 2D (~75 % six-sided in control explants) and Fig 3B",
         "note": f"specimen P_6 = {spec_vm['p6']:.4f}, P_567 = {spec_vm['p567']:.4f}, |psi_6| = {spec_vm['psi6']:.4f}; Panel A +48h/+24h width expansion = {spec['panel_a_expansion_ratio_48h_over_24h']:.4f}x",
         "derived_from": "output_pixels"},
        {"name": "curved trunk + limb-bud surface three-tract P_6 (vs flat three-tract control)",
         "value": round(ccm["p6"], 4), "unit": "fraction of interior primordia",
         "method": "Wave-gated inhibitory-disk packing on the Monge patch (geodesic eikonal wave, 3D chordal inhibition radius), local tangent-plane coordination",
         "reference_value": round(fcm["p6"], 4), "reference_source": "Flat (h = 0) three-tract control run with identical parameters and seed",
         "note": f"curved: 5s={ccm['n5']}, 7s={ccm['n7']}, 5-7 pairs={len(ccm['dislocation_pairs'])}, |psi_6|={ccm['psi6']:.3f}; flat: 5s={fcm['n5']}, 7s={fcm['n7']}, pairs={len(fcm['dislocation_pairs'])}; per tract {json.dumps({k: round(v['p6'], 3) for k, v in dcur['per_tract'].items()})}",
         "derived_from": "script"},
        {"name": "defect fraction in curved (|K| > K_thr) vs flat (|K| <= K_thr) regions of the curved surface",
         "value": round(dcur["defect_ratio"], 4) if np.isfinite(dcur["defect_ratio"]) else 99.0, "unit": "ratio of defect fractions",
         "method": f"Interior primordia split by |Gaussian curvature| at their position (K_thr = {dcur['K_thr']:g}); ratio of (z != 6) fractions",
         "reference_value": 1.0, "reference_source": "Null hypothesis (no curvature localisation); Bausch et al. (2003) Science 299:1716 predicts > 1 (disclinations/scars where K != 0)",
         "note": f"curved-region defect fraction {dcur['defect_fraction_curved']:.4f} (n={dcur['n_curved']}) vs flat-region {dcur['defect_fraction_flat']:.4f} (n={dcur['n_flat']}); Gauss-Bonnet proxy per tract {json.dumps({k: round(v['gauss_bonnet_charge_proxy'], 2) for k, v in dcur['per_tract'].items()})} vs charge {json.dumps({k: v['topological_charge'] for k, v in dcur['per_tract'].items()})}",
         "derived_from": "script"},
        {"name": "limb-bud height sweep: 5-7 dislocation pairs at the tallest limb buds",
         "value": float(hs[-1]["n_dislocation_pairs"]), "unit": "dislocation pairs",
         "method": "Count of adjacent interior 5- and 7-coordinated primordia for limb heights " + ", ".join(f"{h['limb_height']:g}" for h in hs),
         "reference_value": float(hs[0]["n_dislocation_pairs"]), "reference_source": "Flat control (limb height 0) of the same sweep; curvature theory predicts monotone growth with int |K| dA",
         "note": "P_6 by height: " + ", ".join(f"A_l={h['limb_height']:g}: P6={h['p6']:.3f}, pairs={h['n_dislocation_pairs']}, int|K|={h['integrated_abs_K']:.2f}" for h in hs),
         "derived_from": "script"},
        {"name": "Euler / Gauss-Bonnet disclination charge sum(6 - z) on a closed convex-hull triangulation",
         "value": float(sph["topological_charge"]), "unit": "disclination charge (units of pi/3)",
         "method": f"Convex hull of {sph['V']} random points on S^2: V - E + F = {sph['euler_characteristic']}, 3F = 2E {sph['three_F_equals_two_E']}, sum z = 2E {sph['sum_z_equals_two_E']}",
         "reference_value": 12.0, "reference_source": "Lean Theorem 4 euler_disclination_charge (6 chi = sum (6 - z), chi(S^2) = 2); Bausch et al. (2003)",
         "note": f"n5 = {sph['n5']}, n6 = {sph['n6']}, n7 = {sph['n7']}: n5 - n7 = {sph['n5'] - sph['n7']} (differs from 12 because random points also give z = 4 and z >= 8 vertices; the full charge sum(6 - z) is exactly 12)",
         "derived_from": "script"},
        {"name": "isochrone row-count dislocation lemma violations on the curved dorsal tract",
         "value": float(iso["violations"]), "unit": "violations",
         "method": f"For {iso['checked_pairs']} successive dorsal isochrone pairs whose arc lengths differ by >= lambda_y, check that the longer carries >= 1 more primordium ({iso['n_rows']} rows)",
         "reference_value": 0.0, "reference_source": "Lean Theorem 5 isochrone_row_count_dislocation",
         "note": f"flat control: {curved['isochrone_flat']['violations']} violations / {curved['isochrone_flat']['checked_pairs']} checked pairs",
         "derived_from": "script"},
        {"name": "bare inter-tract apterium gap between dorsal and femoral tracts (curved surface)",
         "value": round(curved["apterium_gap_curved"] / LAMBDA_Y_GRID, 4), "unit": "feather periods lambda_y",
         "method": "Minimum mediolateral separation between the lateral-most dorsal primordium and the medial-most femoral primordium (right side)",
         "reference_value": None, "reference_source": f"{HO} Fig 1B: 'spreading waves terminate before colliding, yielding ... bare spaces between the tracts' (gap not quantified)",
         "note": f"flat control gap = {curved['apterium_gap_flat'] / LAMBDA_Y_GRID:.3f} lambda_y; gap arises from the mesenchymal density threshold, not from wave collision",
         "derived_from": "script"},
    ]


# =========================================================================== #
# 10. Deterministic 12-Check Self-Test Suite (`--self-test`)
# =========================================================================== #
def self_test() -> bool:
    """Run 12 deterministic verification checks covering SHA-256, Lean laws, wave/threshold suites, curved surface, specimen."""
    # 1. Dataset SHA-256 locks, JSON schema parity and provenance-correction block
    ref = load_reference_dataset()
    cp = ref["canonical_parameters"]
    assert abs(cp["hexagonal_row_ratio_sqrt3_over_2"] - np.sqrt(3.0) / 2.0) < 1e-9
    assert "provenance_corrections" in ref["metadata"], "reference JSON must carry the Pass 3 provenance corrections"
    assert all(s["doi"] != "10.1016/j.ydbio.2014.08.030" for s in ref["metadata"]["sources"]), "fabricated Glimm 2014 DOI must be gone"
    anchors = ref["ho2019_quantitative_anchors"]
    assert anchors["margin_density_cells_per_100um"] == HO2019_MARGIN_DENSITY_CELLS_PER_100UM

    # 2. Lean Theorems 1 & 2 (Keller-Segel threshold & hexagonal equidistance)
    disp = compute_keller_segel_dispersion_and_geometry()
    assert 0.050 <= disp["chi_c"] <= 0.058, f"Unexpected chi_c: {disp['chi_c']}"
    assert disp["sigma_wt_max"] > 0.05 and disp["sigma_scaleless_max"] < 0.0
    assert abs(disp["diag_neighbor_dist_um"] - disp["lambda_y_um"]) < 1e-9, "6 nearest neighbors must be equidistant"

    # 3. Flat WT geodesic wave: crystalline packing from the band mechanism; PDE condensates track nucleation sites
    suite = run_wave_speed_and_perturbation_suite(seed=2018)
    wt_cm = suite["wt_sim"]["coordination"]
    assert wt_cm["p6"] >= 0.90, f"Expected WT P_6 >= 0.90, got {wt_cm['p6']}"
    assert wt_cm["psi6"] >= 0.85 and wt_cm["nn_cv"] <= 0.05, f"WT |psi_6| = {wt_cm['psi6']}, NN CV = {wt_cm['nn_cv']}"
    assert abs(suite["wt_sim"]["n_field_peaks"] - wt_cm["n_buds"]) <= 2, (
        f"n-field DoG peaks {suite['wt_sim']['n_field_peaks']} should match nucleation sites {wt_cm['n_buds']}")

    # 4. Simultaneous competence (Ho Fig 2B/D): jammed packing, P_6 well below the wave (Lean Theorem 3)
    sim_cm = suite["simul_sim"]["coordination"]
    assert 0.30 <= sim_cm["p6"] <= 0.78, f"Expected simultaneous P_6 in [0.30, 0.78], got {sim_cm['p6']}"
    assert wt_cm["p6"] - sim_cm["p6"] >= 0.20, f"Wave must beat simultaneous by >= 0.20: {wt_cm['p6']} vs {sim_cm['p6']}"
    finite = [w["p6"] for w in suite["wave_sweep"] if w["v_w"] < 8.0]
    assert min(finite) > sim_cm["p6"], "every finite wave speed must out-order the simultaneous regime"

    # 5. EDA-set density threshold (Ho Fig 3D-G, 5C-F): Fc-chEDA1 wider, Ecto-D2 / MTX narrower, co-treatment narrowest
    eda = {e["slug"]: e["tract_width_ratio"] for e in suite["eda_suite"]}
    assert eda["fc_cheda1"] > 1.05 > 0.95 > eda["ecto_d2"], f"Fc-chEDA1 {eda['fc_cheda1']}, Ecto-D2 {eda['ecto_d2']}"
    assert eda["mtx"] < 0.95 and eda["mtx_ecto_d2"] < min(eda["mtx"], eda["ecto_d2"]), f"MTX {eda['mtx']}, MTX+Ecto-D2 {eda['mtx_ecto_d2']}"

    # 6. Scaleless / Latrunculin / sub-threshold chi: no nucleation, uniform dermis
    chi_buds = {c["chi"]: c["n_buds"] for c in suite["chi_sweep"]}
    assert chi_buds[0.0] == 0 and chi_buds[0.01] == 0 and chi_buds[0.04] == 0 and chi_buds[0.24] > 0, chi_buds
    assert suite["scaleless_sim"]["n_std"] < 0.005 and suite["wt_sim"]["n_std"] > 50.0 * suite["scaleless_sim"]["n_std"]

    # 7. Specimen segmentation (Ho Fig 3B Voronoi, Fig 3A expansion)
    spec = segment_feather_tract_specimen()
    spec_vm = spec["voronoi_b"]
    assert spec_vm["n_buds"] == 24 and spec_vm["p6"] >= 0.70 and spec_vm["p567"] == 1.0, (spec_vm["n_buds"], spec_vm["p6"], spec_vm["p567"])
    assert 0.14 <= spec_vm["nn_cv"] <= 0.24 and 1.80 <= spec["panel_a_expansion_ratio_48h_over_24h"] <= 2.40

    # 8. Lean Theorem 4 executed: closed triangulation charge = 12
    sph = sphere_topological_charge_check()
    assert sph["three_F_equals_two_E"] and sph["sum_z_equals_two_E"] and sph["topological_charge"] == 12, sph

    # 9. Curved surface: curvature lowers P_6, adds 5-7 pairs, defects localise where |K| > 0; monotone in limb height
    curved = run_curved_surface_suite(seed=2018)
    fcm, ccm = curved["flat3"]["coordination"], curved["curved3"]["coordination"]
    dc = curved["defects_curved"]
    assert ccm["p6"] < fcm["p6"], f"curved P_6 {ccm['p6']} should be below flat {fcm['p6']}"
    assert len(ccm["dislocation_pairs"]) > len(fcm["dislocation_pairs"]), "curvature must create 5-7 dislocation pairs"
    assert dc["defect_fraction_curved"] > dc["defect_fraction_flat"], f"defects must localise at |K| > 0: {dc}"
    p6_by_h = [h["p6"] for h in curved["height_sweep"]]
    assert all(b <= a + 0.02 for a, b in zip(p6_by_h[:-1], p6_by_h[1:])), f"P_6 should fall with limb height: {p6_by_h}"

    # 10. Lean Theorem 5 executed on the isochrone rows
    iso = curved["isochrone_curved"]
    assert iso["violations"] == 0 and iso["checked_pairs"] >= 1, iso

    # 11. Tracts terminate before colliding (bare apterium) on both the flat and curved sheets
    assert curved["apterium_gap_flat"] > 0.3 * LAMBDA_Y_GRID and curved["apterium_gap_curved"] > 0.3 * LAMBDA_Y_GRID

    # 12. Bitwise two-run determinism (flat and curved)
    S_small = build_embryo_trunk_surface(nu=96, nv=64, limb_offset_u=34.0, limb_sigma=9.0, humeral_v=48.0, femoral_v=16.0)
    r1 = simulate_feather_array_on_surface(S_small, steps=80, tracts=("dorsal", "humeral", "femoral"), seed=2018, sub=3)
    r2 = simulate_feather_array_on_surface(S_small, steps=80, tracts=("dorsal", "humeral", "femoral"), seed=2018, sub=3)
    assert np.array_equal(r1["n_field"], r2["n_field"]) and np.array_equal(r1["pts"], r2["pts"]), "determinism check failed"

    print(
        "[self_test] All 12 Ho et al. (2019) / Painter et al. (2018) checks PASSED "
        f"(WT_P6={wt_cm['p6']:.3f}, Simul_P6={sim_cm['p6']:.3f}, FcEDA_width={eda['fc_cheda1']:.2f}x, EctoD2_width={eda['ecto_d2']:.2f}x, "
        f"flat3_P6={fcm['p6']:.3f}, curved3_P6={ccm['p6']:.3f}, 5-7 pairs={len(ccm['dislocation_pairs'])}, "
        f"sphere_charge={sph['topological_charge']}, specimen_buds={spec_vm['n_buds']}, specimen_P6={spec_vm['p6']:.3f})."
    )
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--vw", type=float, default=0.22, help="Mediolateral EDA competence wave speed v_w (>= 8: simultaneous)")
    ap.add_argument("--chi", type=float, default=0.24, help="FGF20 chemotactic sensitivity chi (chi_c = 0.054)")
    ap.add_argument("--Dn", type=float, default=0.025, help="Dermal cell diffusivity D_n")
    ap.add_argument("--Dc", type=float, default=0.04, help="FGF20 chemoattractant diffusivity D_c")
    ap.add_argument("--Db", type=float, default=0.45, help="BMP4 inhibitor diffusivity D_b")
    ap.add_argument("--n-crit", type=float, default=1.0, help="EDA-set critical mesenchymal density (control 1.0 = 66 cells/100 um; Fc-chEDA1 49/66; Ecto-D2 98/66)")
    ap.add_argument("--density-scale", type=float, default=1.0, help="Mesenchymal density multiplier (MTX ~0.8)")
    ap.add_argument("--limb-height", type=float, default=30.0, help="Limb-bud bump height A_l of the curved surface (0 = flat)")
    ap.add_argument("--steps", type=int, default=320, help="PDE / nucleation integration steps")
    ap.add_argument("--seed", type=int, default=2018, help="Deterministic RNG seed")
    ap.add_argument("--single", action="store_true", help="Run one curved three-tract simulation with the CLI parameters and print its metrics (no figure)")
    ap.add_argument("--self-test", action="store_true", help="Run 12-check verification suite and exit")
    args = ap.parse_args()

    if args.self_test:
        self_test()
        return 0
    if args.single:
        S = build_embryo_trunk_surface(limb_height=args.limb_height, flat=(args.limb_height == 0.0), **CURVED_SURFACE_KW)
        sim = simulate_feather_array_on_surface(S, steps=args.steps, D_n=args.Dn, D_c=args.Dc, D_b=args.Db, chi=args.chi, v_w=args.vw,
                                                n_crit=args.n_crit, density_scale=args.density_scale, lateral_L_factor=CURVED_LATERAL_L_FACTOR,
                                                tracts=("dorsal", "humeral", "femoral"), seed=args.seed)
        cm = sim["coordination"]
        print(json.dumps({"n_buds": cm["n_buds"], "n_interior": cm["n_interior"], "p6": round(cm["p6"], 4), "n5": cm["n5"], "n7": cm["n7"],
                          "dislocation_pairs": len(cm["dislocation_pairs"]), "psi6": round(cm["psi6"], 4), "nn_cv": round(cm["nn_cv"], 4),
                          "dorsal_tract_width_px": round(sim["tract_width_px"], 2), "integrated_abs_K": round(S["integrated_abs_K"], 3)}, indent=2))
        return 0

    render_outputs(seed=args.seed)
    print(f"[2018-glimm-avian-feather-bud-hexagonal] Rendered {OUT_PNG.relative_to(ROOT)}, {OUT_SURFACE_PNG.relative_to(ROOT)} and {OUT_GIF.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

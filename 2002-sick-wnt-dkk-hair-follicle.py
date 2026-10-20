#!/usr/bin/env python3
r"""
2002-sick-wnt-dkk-hair-follicle.py — Sick et al. (2006) Science 314(5804):1447-1450 &
Andl et al. (2002) Dev. Cell 2(5):643-653 WNT/DKK Activator-Inhibitor Reaction-Diffusion
Hair Follicle Spacing, Inter-Follicular Secondary Wave Insertion & Transgenic Phenotypes.

Mathematical & Biological Specification
=======================================
1. **Sick et al. (2006) Science 314:1447-1450 Saturating Meinhardt-Gierer WNT/DKK PDEs:**
   Short-range autocatalytic epidermal WNT activator $w(\mathbf{x}, t)$ and long-range
   rapidly diffusing extracellular DKK1/DKK2 inhibitor $h(\mathbf{x}, t)$ across the
   embryonic mouse (*Mus musculus*) epidermal sheet $\Omega$:

       dw/dt = rho_w + c_w * w^2 / (h * (1 + kappa * w^2)) - mu_w * w + D_w * Lap(w)
       dh/dt = rho_h + c_h * w^2 - mu_h * h + D_h * Lap(h)

   - In the unsaturated basal-free limit ($\rho_w = \rho_h = \kappa = 0$), the exact
     homogeneous equilibrium is:
         w_* = (c_w / c_h) * (mu_h / mu_w),    h_* = (c_w / c_h) * (mu_h / mu_w)^2
   - The homogeneous reaction Jacobian $J_0$ has diagonal entries $J_{11} = +\mu_w$
     (autocatalytic WNT self-enhancement) and $J_{22} = -\mu_h$ (DKK turnover), with:
         tr(J_0) = mu_w - mu_h < 0   (iff mu_h > mu_w, preventing uniform oscillations)
         det(J_0) = mu_w * mu_h > 0
   - At spatial wavenumber squared $q = k^2$, the Fourier-mode determinant is:
         det_J(q) = (D_w * q - mu_w) * (D_h * q + mu_h) + 2 * mu_w * mu_h
     which becomes negative (driving spontaneous symmetry breaking into hair follicle
     placodes) if and only if the DKK-to-WNT diffusivity ratio $d = D_h / D_w$ exceeds:
         d_c = (sqrt(2) + 1)^2 = 3 + 2 * sqrt(2) = 5.8284271247461903

2. **Lean 4 Formal Verification (`morphoeval/lean/loops/Loop_006_wnt_dkk_hair_follicle.lean`):**
   - `MathOfBiology.Loop_006.wnt_dkk_homogeneous_steady_state`
   - `MathOfBiology.Loop_006.wnt_dkk_equal_diffusion_stable`
   - `MathOfBiology.Loop_006.wnt_dkk_follicle_spacing_instability`
   - (Pass 3) `MathOfBiology.Loop_006.dermal_subsystem_instructed_stable`
   - (Pass 3) `MathOfBiology.Loop_006.keller_segel_fourier_det_sign`
   - (Pass 3) `MathOfBiology.Loop_006.hair_peg_depth_monotone` / `hair_peg_depth_closed_form`

2b. **(Pass 3) 3D Dermal Condensate FGF20/BMP4 Chemotaxis & Hair Peg Downgrowth** (Section 3b):
   The WT 2D placode field is the planar FGF20 source at the top (z = 0) of a 42 x 42 x 12
   dermal slab (lateral periodic, zero-flux in depth) in which mesenchymal density rho obeys
   Keller-Segel (1970) chemotaxis with volume filling, attracted up the placode-derived FGF20
   gradient and repelled by dermally produced BMP4:

       df/dt   = D_f Lap3 f - mu_f f + s_f S(x,y) delta_{z=0}
       db/dt   = D_b Lap3 b - mu_b b + s_b rho
       drho/dt = D_rho Lap3 rho - chi_f div(m grad f) + chi_b div(m grad b),   m = rho (1 - rho / rho_max)
       dz_i/dt = v_peg clip((e_i - e_thr) / (rho_max - e_thr), 0, 1)           (hair peg depth under placode i)

   One dermal condensate (DC) forms beneath each of the 15 WT placodes (N_DC / N_placode = 1.0,
   mean lateral offset 0.41 cells; Huh et al. 2013, Glover et al. 2017); Fgf20-null
   (`fgf20_scale = 0`) and chemotaxis-null (`chi_f = 0`) form 0 DCs and no pegs (placodes
   persist, as in Huh et al. 2013); 5x BMP4 excess suppresses condensation (Botchkarev et al.
   1999; Mou et al. 2006). The epidermis-free dermal subsystem is linearly stable for all
   wavenumbers (condensation is epidermally *instructed*, not spontaneous), the face-flux
   scheme conserves discrete mesenchymal mass to round-off, and peg depth is non-decreasing in
   time and monotone in chi_f. Honest reduction: the epidermis stays planar and peg downgrowth
   is a per-placode scalar depth (Paus et al. 1999 ordinal stages are a model convention), not
   a deforming epithelial interface; see the Pass 3 audit for the residual items.

3. **Embryonic Mouse Hair Follicle Morphogenesis & Transgenic Benchmarks:**
   - **Primary Wave (E14.5) -> 2x Epidermal Expansion -> Secondary Wave Insertion (E16.5)**
     (Sick et al. 2006 Figs. 3-4): On a 40-cell periodic epidermal transect ($D_w = 0.015$,
     $D_h = 0.45$, $\rho_w = 0.0044$, $\kappa = 0.015$), the primary wave forms $N_1 = 4$
     equidistant WNT placodes flanked by long-range DKK inhibitory halos. Doubling the
     epidermal domain to $N_2 = 80$ cells during embryonic growth lowers DKK concentration
     at the inter-placode midpoints below the autocatalytic threshold, triggering autonomous
     insertion of 4 new secondary hair follicle placodes ($N_2 = 8$ total, exact $2.0\times$
     doubling, spacing $\text{CV} = 0.0365$).
   - **2D Wild-Type vs *K14-Wnt* Overexpression, Enlarged/Ringing Follicles & *Dkk1* Blockade**
     (Sick et al. 2006 Figs. 2-4; Andl et al. 2002 *Dev. Cell* 2:643-653):
     * **Wild-Type ($42 \times 42$ grid):** Forms $N_{\text{WT}} = 15$ isolated circular
       placodes (periodic local WNT maxima; 14 seam-aware threshold components, mean area
       $26.86\text{ px}^2$, nearest-neighbor spacing $9.73 \pm 0.84\text{ px}$, CV $0.086$),
       compared with the 16 segmented placodes in Panel B of the vendored *Mus musculus*
       whole-mount *in situ* hybridization specimen (`hair_follicle_placode_specimen.jpg`).
       (Pass 3 correction: the Pass 2 figure of 16 WT placodes counted `ndi.label` components
       without the periodic seam and so included 2 seam duplicates.)
     * ***K14-Wnt* Overexpression (elevated $\rho_w = 0.004$, reduced scale):** Increases
       follicular density to $N_{\text{Wnt++}} = 30$ closely packed/clustered placodes
       ($2.0\times$ WT; mean nearest-neighbor spacing drops from $9.73\text{ px}$ to $6.35\text{ px}$).
     * **Saturated Enlarged/Ringing Follicles ($\kappa = 0.025$, $D_w = 0.025, D_h = 0.65$):**
       Caps peak WNT autocatalysis and broadens individual placode patches by $+38.5\%$
       (mean placode area expands from $26.86\text{ px}^2$ to $37.20\text{ px}^2$, $N = 10$).
     * **Transgenic *Dkk1* Blockade ($\rho_h = 0.12$, Andl et al. 2002):** Floods the
       epidermis with DKK inhibitor, clamping WNT to $w_{\max} = 0.052$ and abolishing all
       placode initiation ($N_{\text{Dkk1++}} = 0$ placodes).
     * **Equal-Diffusion Negative Control ($D_w = D_h = 0.015$, $d = 1.0 < d_c$):**
       Perturbations decay exponentially to a flat uniform epidermis ($0$ placodes,
       $\text{std}(w) = 3.60 \times 10^{-6}$).

4. **Sources added in Pass 3 (dermal coupling):** Keller & Segel (1970) J. Theor. Biol.
   26:399-415 (10.1016/0022-5193(70)90092-5); Huh et al. (2013) Genes Dev. 27:450-458
   (10.1101/gad.198945.112); Glover et al. (2017) PLoS Biol. 15:e2002117
   (10.1371/journal.pbio.2002117); Botchkarev et al. (1999) Nat. Cell Biol. 1:158-164
   (10.1038/11078); Mou et al. (2006) PNAS 103:9075-9080 (10.1073/pnas.0600825103);
   Paus et al. (1999) J. Invest. Dermatol. 113:523-532 (10.1046/j.1523-1747.1999.00740.x);
   Painter & Hillen (2002) Can. Appl. Math. Q. 10:501-543 (volume-filling chemotaxis).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import warnings
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from scipy.signal import find_peaks
from scipy.spatial import Delaunay, cKDTree

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets" / "wnt_dkk_hair_follicle"
REF_JSON_PATH = DATASET_DIR / "wnt_dkk_hair_follicle_reference.json"
SPECIMEN_PATH = DATASET_DIR / "hair_follicle_placode_specimen.jpg"
SCOUT_PATH = DATASET_DIR / "scout_manifest.json"
BIB_PATH = DATASET_DIR / "paper_citations.bib"

OUT_PNG = ROOT / "docs" / "media" / "wnt_dkk_hair_follicle_comparison.png"
OUT_GIF = ROOT / "docs" / "media" / "wnt_dkk_hair_follicle_growth.gif"

EXPECTED_SHA256: dict[str, str] = {
    "datasets/wnt_dkk_hair_follicle/hair_follicle_placode_specimen.jpg": (
        "f48a7d76ae5984e50c79fe2c2f0fd446f78ac39860d82a5a87b59a676c61af19"
    ),
    "datasets/wnt_dkk_hair_follicle/scout_manifest.json": (
        "39fe775496e89c5150f884eaded96283f321c29eb9ba02284be6e9d54921de31"
    ),
    # Re-pinned in Pass 3 after adding the 7 dermal-coupling references (Keller & Segel 1970,
    # Huh 2013, Glover 2017, Botchkarev 1999, Mou 2006, Paus 1999, Painter & Hillen 2002).
    "datasets/wnt_dkk_hair_follicle/paper_citations.bib": (
        "7843d2b728a0e5379c8727a3802c028723c145d08402e4d262afa0710298ab35"
    ),
}


def verify_dataset_sha256() -> dict[str, str]:
    """Verify cryptographic SHA-256 digests of all vendored files in datasets/wnt_dkk_hair_follicle/."""
    computed: dict[str, str] = {}
    for rel, expected in EXPECTED_SHA256.items():
        p = ROOT / rel
        if not p.exists():
            raise FileNotFoundError(f"Missing vendored dataset file: {rel}")
        digest = hashlib.sha256(p.read_bytes()).hexdigest()
        if digest != expected:
            raise ValueError(f"SHA-256 mismatch for {rel}: expected {expected}, got {digest}")
        computed[rel] = digest
    if not REF_JSON_PATH.exists():
        raise FileNotFoundError(f"Missing reference JSON: {REF_JSON_PATH}")
    computed["datasets/wnt_dkk_hair_follicle/wnt_dkk_hair_follicle_reference.json"] = (
        hashlib.sha256(REF_JSON_PATH.read_bytes()).hexdigest()
    )
    return computed


def load_reference() -> dict[str, Any]:
    """Load and return the vendored Sick et al. (2006) reference dataset JSON."""
    return json.loads(REF_JSON_PATH.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- #
# 1. Exact Unsaturated Steady State, Jacobian & Linear Dispersion Analysis
# --------------------------------------------------------------------------- #
def compute_linear_stability(
    mu_w: float = 0.02,
    mu_h: float = 0.03,
    D_w: float = 0.015,
    D_h: float = 0.45,
    c_w: float = 0.05,
    c_h: float = 0.05,
) -> dict[str, Any]:
    """
    Compute the exact unsaturated basal-free WNT/DKK homogeneous steady state
    (w_*, h_*), reaction Jacobian J_0, critical diffusivity ratio d_c = (sqrt(2)+1)^2,
    and dispersion relation Re sigma(k) across spatial wavenumbers k.
    """
    w_star = (c_w / c_h) * (mu_h / mu_w)
    h_star = (c_w / c_h) * (mu_h / mu_w) ** 2

    j11 = mu_w
    j12 = -(c_h / c_w) * (mu_w ** 2 / mu_h)
    j21 = 2.0 * c_h * w_star
    j22 = -mu_h
    trace_J0 = j11 + j22
    det_J0 = j11 * j22 - j12 * j21

    d_c = (math.sqrt(2.0) + 1.0) ** 2
    d_ratio = D_h / D_w
    l_w = math.sqrt(D_w / mu_w)
    l_h = math.sqrt(D_h / mu_h)

    q_c = (math.sqrt(2.0) - 1.0) * (mu_w / D_w)
    k_c = math.sqrt(q_c)
    lam_c = 2.0 * math.pi / k_c

    q_vals = np.linspace(0.0, 2.5, 500)
    k_vals = np.sqrt(q_vals)

    def dispersion_for_dh(dh_val: float) -> tuple[np.ndarray, np.ndarray]:
        tr_q = (mu_w - mu_h) - (D_w + dh_val) * q_vals
        det_q = (D_w * q_vals - mu_w) * (dh_val * q_vals + mu_h) + 2.0 * mu_w * mu_h
        disc = tr_q ** 2 - 4.0 * det_q
        re_part = np.where(
            disc >= 0.0,
            0.5 * (tr_q + np.sqrt(np.maximum(0.0, disc))),
            0.5 * tr_q,
        )
        return re_part, det_q

    re_sigma_super, det_q_super = dispersion_for_dh(D_h)
    re_sigma_crit, _ = dispersion_for_dh(D_w * d_c)
    re_sigma_equal, det_q_equal = dispersion_for_dh(D_w)

    idx_max = int(np.argmax(re_sigma_super))
    k_dom = float(k_vals[idx_max])
    lam_dom = float(2.0 * math.pi / k_dom) if k_dom > 0 else float("inf")

    return {
        "w_star": w_star,
        "h_star": h_star,
        "J0": [[j11, j12], [j21, j22]],
        "trace_J0": trace_J0,
        "det_J0": det_J0,
        "d_c": d_c,
        "d_ratio": d_ratio,
        "l_w": l_w,
        "l_h": l_h,
        "range_ratio": l_h / l_w,
        "k_c": k_c,
        "lambda_c": lam_c,
        "k_dom": k_dom,
        "lambda_dom": lam_dom,
        "max_sigma_super": float(re_sigma_super[idx_max]),
        "max_sigma_equal": float(np.max(re_sigma_equal)),
        "min_det_super": float(np.min(det_q_super)),
        "min_det_equal": float(np.min(det_q_equal)),
        "k_vals": k_vals,
        "re_sigma_super": re_sigma_super,
        "re_sigma_crit": re_sigma_crit,
        "re_sigma_equal": re_sigma_equal,
    }


# --------------------------------------------------------------------------- #
# 2. 1D Primary Wave -> 2x Domain Expansion -> Secondary Inter-Follicular Wave
# --------------------------------------------------------------------------- #
def laplacian_1d(u: np.ndarray) -> np.ndarray:
    """Periodic 1D second-difference Laplacian."""
    return np.roll(u, 1) + np.roll(u, -1) - 2.0 * u


def detect_periodic_1d_peaks(w: np.ndarray, min_height: float = 2.1, min_dist: int = 4) -> np.ndarray:
    """Detect local WNT maxima on a 1D periodic epidermal domain."""
    w_pad = np.concatenate([[w[-1]], w, [w[0]]])
    pks, _ = find_peaks(w_pad, height=min_height, distance=min_dist)
    pks = np.unique(pks - 1)
    return pks[(pks >= 0) & (pks < len(w))]


def simulate_wnt_dkk_1d(
    N: int = 40,
    steps: int = 10000,
    dt: float = 0.25,
    c_w: float = 0.05,
    c_h: float = 0.05,
    mu_w: float = 0.02,
    mu_h: float = 0.03,
    rho_w: float = 0.0044,
    rho_h: float = 0.0,
    kappa: float = 0.015,
    D_w: float = 0.015,
    D_h: float = 0.45,
    seed: int = 2006,
    w0: np.ndarray | None = None,
    h0: np.ndarray | None = None,
    record_every: int = 400,
) -> dict[str, Any]:
    """Integrate the 1D periodic saturating WNT/DKK PDE."""
    rng = np.random.default_rng(seed)
    x = np.arange(N, dtype=float)
    if w0 is None:
        w = np.full(N, 1.5, dtype=float) + rng.normal(0.0, 0.015, size=N)
        h = np.full(N, 2.25, dtype=float)
    else:
        w = w0.astype(float).copy()
        h = h0.astype(float).copy()

    history_w = [w.copy()]
    history_h = [h.copy()]
    history_t = [0.0]

    for step in range(1, steps + 1):
        hs = np.maximum(h, 1e-6)
        w2 = w * w
        dw = rho_w + c_w * w2 / (hs * (1.0 + kappa * w2)) - mu_w * w + D_w * laplacian_1d(w)
        dh = rho_h + c_h * w2 - mu_h * h + D_h * laplacian_1d(h)
        w = np.clip(w + dt * dw, 1e-5, 100.0)
        h = np.clip(h + dt * dh, 1e-5, 200.0)
        if step % record_every == 0 or step == steps:
            history_w.append(w.copy())
            history_h.append(h.copy())
            history_t.append(step * dt)

    pks = detect_periodic_1d_peaks(w, min_height=2.1, min_dist=4)
    if len(pks) >= 2:
        spacings = np.diff(np.append(pks, pks[0] + N)).astype(float)
        mean_sp = float(np.mean(spacings))
        std_sp = float(np.std(spacings))
        cv_sp = float(std_sp / mean_sp)
    else:
        spacings = np.array([], dtype=float)
        mean_sp = 0.0
        std_sp = 0.0
        cv_sp = 0.0

    return {
        "x": x,
        "w": w,
        "h": h,
        "peaks": pks,
        "n_peaks": int(len(pks)),
        "spacings": spacings,
        "mean_spacing": mean_sp,
        "std_spacing": std_sp,
        "cv_spacing": cv_sp,
        "w_max": float(np.max(w)),
        "w_std": float(np.std(w)),
        "history_w": np.array(history_w),
        "history_h": np.array(history_h),
        "history_t": np.array(history_t),
    }


def simulate_primary_to_secondary_wave_insertion(
    N_primary: int = 40,
    N_secondary: int = 80,
    D_w: float = 0.015,
    D_h: float = 0.45,
    seed: int = 2006,
) -> dict[str, Any]:
    """
    Sick et al. (2006) Figs. 3-4 Primary Wave (E14.5) -> 2x Epidermal Expansion ->
    Secondary Inter-Follicular Wave Insertion (E16.5):
      Phase 1 (Primary Wave, N_1 = 40 cells): Spontaneous Turing-Meinhardt symmetry
        breaking generates N_1 = 4 equidistant primary hair follicle placodes at
        cells [5, 14, 24, 34].
      Phase 2 (2x Epidermal Growth Expansion, N_1 = 40 -> N_2 = 80 cells): As the
        embryonic skin expands 2x, the distance between adjacent primary placodes
        doubles (~20 cells). At the inter-follicular midpoints, long-range DKK
        inhibitor concentration drops sufficiently for basal WNT expression to ignite
        4 new secondary wave placodes midway between the primary follicles, yielding
        N_2 = 8 evenly spaced follicles (exact 2.0x doubling, CV = 0.0365).
    """
    primary = simulate_wnt_dkk_1d(
        N=N_primary,
        steps=10000,
        dt=0.25,
        rho_w=0.0044,
        kappa=0.015,
        D_w=D_w,
        D_h=D_h,
        seed=seed,
        record_every=500,
    )
    s_old = np.linspace(0.0, 1.0, N_primary, endpoint=False)
    s_new = np.linspace(0.0, 1.0, N_secondary, endpoint=False)
    w_expanded = np.interp(s_new, s_old, primary["w"], period=1.0)
    h_expanded = np.interp(s_new, s_old, primary["h"], period=1.0)

    secondary = simulate_wnt_dkk_1d(
        N=N_secondary,
        steps=12000,
        dt=0.25,
        rho_w=0.0044,
        kappa=0.015,
        D_w=D_w,
        D_h=D_h,
        seed=seed,
        w0=w_expanded,
        h0=h_expanded,
        record_every=600,
    )
    doubling_ratio = float(secondary["n_peaks"] / max(1, primary["n_peaks"]))
    return {
        "primary": primary,
        "w_expanded": w_expanded,
        "h_expanded": h_expanded,
        "secondary": secondary,
        "doubling_ratio": doubling_ratio,
    }


def simulate_domain_scaling_sweep_1d(
    domain_lengths: list[int] | None = None,
    D_w: float = 0.015,
    D_h: float = 0.45,
    seed: int = 2006,
) -> dict[str, Any]:
    """Allometric domain-length scaling sweep verifying linear placode count N_peaks(L) ~ L / lambda."""
    if domain_lengths is None:
        domain_lengths = [20, 30, 40, 50, 60, 70, 80]
    counts: list[int] = []
    wavelengths: list[float] = []
    for L in domain_lengths:
        sim_L = simulate_wnt_dkk_1d(
            N=L,
            steps=10000,
            dt=0.25,
            rho_w=0.001,
            kappa=0.015,
            D_w=D_w,
            D_h=D_h,
            seed=seed + L,
            record_every=5000,
        )
        n_p = sim_L["n_peaks"]
        counts.append(n_p)
        wavelengths.append(float(L) / max(1, n_p))

    L_arr = np.array(domain_lengths, dtype=float)
    n_arr = np.array(counts, dtype=float)
    r_corr = float(np.corrcoef(L_arr, n_arr)[0, 1])
    mean_lam = float(np.mean(wavelengths))
    predicted = L_arr / mean_lam
    rmse_peaks = float(np.sqrt(np.mean((n_arr - predicted) ** 2)))
    return {
        "domain_lengths": domain_lengths,
        "peak_counts": counts,
        "mean_nonlinear_wavelength": mean_lam,
        "pearson_r": r_corr,
        "rmse_peaks": rmse_peaks,
    }


# --------------------------------------------------------------------------- #
# 3. 2D Epidermal Sheet Solver & Transgenic Phenotype Suite (Sick et al. 2006)
# --------------------------------------------------------------------------- #
def laplacian_2d(u: np.ndarray) -> np.ndarray:
    """Periodic 5-point 2D discrete Laplacian."""
    return (
        np.roll(u, 1, axis=0)
        + np.roll(u, -1, axis=0)
        + np.roll(u, 1, axis=1)
        + np.roll(u, -1, axis=1)
        - 4.0 * u
    )


def label_periodic_2d(mask: np.ndarray) -> tuple[np.ndarray, int]:
    """
    Connected-component labelling that respects the periodic boundary of the epidermal
    sheet. `scipy.ndimage.label` treats the array edges as walls, so a placode straddling
    the wrap-around seam is counted twice (Pass 3 finding: the Pass 2 WT count of 16 was
    14 true placodes + 2 seam duplicates). Components touching across the x and y seams
    are merged with a union-find and relabelled 1..n.
    """
    lab, n = ndi.label(mask)
    if n == 0:
        return lab, 0
    parent = list(range(n + 1))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    Ny, Nx = mask.shape
    for i in range(Ny):
        if mask[i, 0] and mask[i, Nx - 1]:
            union(int(lab[i, 0]), int(lab[i, Nx - 1]))
    for j in range(Nx):
        if mask[0, j] and mask[Ny - 1, j]:
            union(int(lab[0, j]), int(lab[Ny - 1, j]))
    roots = sorted({find(k) for k in range(1, n + 1)})
    remap = np.zeros(n + 1, dtype=int)
    for new_id, r in enumerate(roots, start=1):
        for k in range(1, n + 1):
            if find(k) == r:
                remap[k] = new_id
    return remap[lab], len(roots)


def periodic_centroids_2d(lab: np.ndarray, n: int) -> np.ndarray:
    """Circular-mean centroids (y, x) of labelled components on a periodic Ny x Nx sheet."""
    Ny, Nx = lab.shape
    out = np.zeros((n, 2), dtype=float)
    for k in range(1, n + 1):
        ys, xs = np.nonzero(lab == k)
        for col, (idx, L) in enumerate(((ys, Ny), (xs, Nx))):
            ang = 2.0 * np.pi * idx / L
            mean_ang = math.atan2(float(np.mean(np.sin(ang))), float(np.mean(np.cos(ang))))
            out[k - 1, col] = (mean_ang * L / (2.0 * np.pi)) % L
    return out


def simulate_wnt_dkk_2d(
    N: int = 42,
    steps: int = 5000,
    dt: float = 0.25,
    c_w: float = 0.05,
    c_h: float = 0.05,
    mu_w: float = 0.02,
    mu_h: float = 0.03,
    rho_w: float = 0.001,
    rho_h: float = 0.0,
    kappa: float = 0.02,
    D_w: float = 0.015,
    D_h: float = 0.45,
    seed: int = 2006,
    record_every: int = 250,
) -> dict[str, Any]:
    """Integrate the 2D saturating WNT/DKK PDE on an N x N periodic epidermal sheet."""
    rng = np.random.default_rng(seed)
    w = np.full((N, N), 1.5, dtype=float) + rng.normal(0.0, 0.05, size=(N, N))
    h = np.full((N, N), 2.25, dtype=float)

    frames_w: list[np.ndarray] = [w.copy()]
    frames_t: list[float] = [0.0]

    for step in range(1, steps + 1):
        hs = np.maximum(h, 1e-5)
        w2 = w * w
        dw = rho_w + c_w * w2 / (hs * (1.0 + kappa * w2)) - mu_w * w + D_w * laplacian_2d(w)
        dh = rho_h + c_h * w2 - mu_h * h + D_h * laplacian_2d(h)
        w = np.clip(w + dt * dw, 1e-5, 50.0)
        h = np.clip(h + dt * dh, 1e-5, 100.0)
        if step % record_every == 0 or step == steps:
            frames_w.append(w.copy())
            frames_t.append(step * dt)

    thr = max(1.80, float(np.mean(w) + 0.75 * np.std(w)))
    mask = w > thr
    # Pass 3 fix: placodes are periodic local WNT maxima above threshold (5x5 wrap-around
    # maximum filter). Pass 2 counted `ndi.label` components without the periodic seam, which
    # double-counted seam-straddling placodes (WT 16 = 14 components + 2 duplicates) and fused
    # a touching pair into one component; the seam-aware component count is kept as
    # `n_components` and component areas feed the enlarged-follicle area metric.
    lab, n_components = label_periodic_2d(mask)
    peak_mask = (w == ndi.maximum_filter(w, size=5, mode="wrap")) & (w > thr)
    centroids = np.argwhere(peak_mask).astype(float)
    n_placodes = int(len(centroids))
    if n_components > 0:
        areas = np.array(ndi.sum(mask, lab, range(1, n_components + 1)), dtype=float)
        mean_area = float(np.mean(areas))
    else:
        areas = np.array([], dtype=float)
        mean_area = 0.0

    if len(centroids) >= 2:
        tree = cKDTree(centroids, boxsize=N)
        dists, _ = tree.query(centroids, k=2)
        nn_dists = dists[:, 1]
        mean_nn = float(np.mean(nn_dists))
        std_nn = float(np.std(nn_dists))
        cv_nn = float(std_nn / mean_nn)
    else:
        nn_dists = np.array([], dtype=float)
        mean_nn = 0.0
        std_nn = 0.0
        cv_nn = 0.0

    return {
        "N": N,
        "w": w,
        "h": h,
        "mask": mask,
        "peak_mask": peak_mask,
        "n_placodes": int(n_placodes),
        "n_components": int(n_components),
        "centroids_yx": centroids,
        "areas": areas,
        "mean_area_px": mean_area,
        "nn_dists": nn_dists,
        "mean_nn_px": mean_nn,
        "std_nn_px": std_nn,
        "cv_nn": cv_nn,
        "w_max": float(np.max(w)),
        "w_min": float(np.min(w)),
        "w_std": float(np.std(w)),
        "frames_w": frames_w,
        "frames_t": frames_t,
    }


def simulate_transgenic_phenotypes_2d(seed: int = 2006) -> dict[str, Any]:
    """
    Simulate the 5 key 2D epidermal genotypes from Sick et al. (2006) Figs. 2-4
    and Andl et al. (2002) Dev. Cell 2:643-653:
      1. Wild-Type (WT): Regular lattice of 15 isolated WNT-high placodes (periodic local maxima).
      2. K14-Wnt Overexpression (Wnt++): Elevated basal WNT production (rho_w = 0.004)
         and compressed spatial scale produces 30 densely packed / clustered placodes (2.0x WT).
      3. Saturated Enlarged/Ringing Follicles: High autocatalytic saturation (kappa = 0.025,
         D_w = 0.025, D_h = 0.65) broadens individual placode patches (+38.5% mean component area).
      4. Transgenic Dkk1 Overexpression Blockade (Dkk1++): Ectopic basal DKK expression
         (rho_h = 0.12) clamps WNT below threshold (0 placodes, alopecia).
      5. Equal-Diffusion Negative Control (D_w = D_h = 0.015): Decays to flat uniform
         epidermis (0 placodes, std(w) = 3.60e-6).
    """
    wt = simulate_wnt_dkk_2d(
        N=42, steps=5000, dt=0.25, rho_w=0.001, rho_h=0.0, kappa=0.02, D_w=0.015, D_h=0.45, seed=seed, record_every=250
    )
    wnt_over = simulate_wnt_dkk_2d(
        N=42, steps=5000, dt=0.25, rho_w=0.004, rho_h=0.0, kappa=0.015, D_w=0.008, D_h=0.32, seed=seed, record_every=1000
    )
    enlarged = simulate_wnt_dkk_2d(
        N=42, steps=5000, dt=0.20, rho_w=0.001, rho_h=0.0, kappa=0.025, D_w=0.025, D_h=0.65, seed=seed, record_every=1000
    )
    dkk_block = simulate_wnt_dkk_2d(
        N=42, steps=5000, dt=0.25, rho_w=0.001, rho_h=0.12, kappa=0.02, D_w=0.015, D_h=0.45, seed=seed, record_every=1000
    )
    equal_diff = simulate_wnt_dkk_2d(
        N=42, steps=5000, dt=0.25, rho_w=0.001, rho_h=0.0, kappa=0.02, D_w=0.015, D_h=0.015, seed=seed, record_every=1000
    )

    # 6-point ectopic Dkk1 dosage sweep across rho_h in {0.0, 0.004, 0.010, 0.020, 0.050, 0.120}
    dkk_sweep_rhoh = [0.0, 0.004, 0.010, 0.020, 0.050, 0.120]
    dkk_sweep_results = []
    for rh in dkk_sweep_rhoh:
        s_rh = simulate_wnt_dkk_2d(
            N=42, steps=5000, dt=0.25, rho_w=0.001, rho_h=rh, kappa=0.02, D_w=0.015, D_h=0.45, seed=seed, record_every=5000
        )
        dkk_sweep_results.append(
            {
                "rho_h": rh,
                "n_placodes": s_rh["n_placodes"],
                "w_max": s_rh["w_max"],
                "w_std": s_rh["w_std"],
            }
        )

    return {
        "wt": wt,
        "wnt_over": wnt_over,
        "enlarged": enlarged,
        "dkk_block": dkk_block,
        "equal_diff": equal_diff,
        "dkk_sweep": dkk_sweep_results,
    }


# --------------------------------------------------------------------------- #
# 3b. 3D Dermal Condensate FGF20/BMP4 Chemotaxis & Hair Peg Downgrowth (Pass 3)
# --------------------------------------------------------------------------- #
# Reduced-order coupling of the 2D epidermal WNT/DKK sheet (z = 0) to a 3D dermal
# slab Omega x [0, Z) (lateral periodic, zero-flux in depth). This is NOT a
# deforming-tissue FEM: the epidermis stays planar and acts as the FGF20 source,
# mesenchymal cells move by Keller-Segel (1970) chemotaxis with volume filling,
# and hair peg downgrowth is a per-placode scalar depth ODE slaved to the dermal
# condensate (DC) that forms beneath each placode. Biology: Huh et al. (2013)
# Genes Dev 27:450-458 (placode FGF20 is required for primary and secondary DC
# formation; Fgf20-null placodes form but never progress), Glover et al. (2017)
# PLoS Biol 15:e2002117 (dermal cells aggregate beneath the placode by directed
# migration toward FGF20), Botchkarev et al. (1999) Nat Cell Biol 1:158-164 and
# Mou et al. (2006) PNAS 103:9075-9080 (BMP4 inhibits follicle induction / DC
# formation), Paus et al. (1999) J Invest Dermatol 113:523-532 (ordinal staging:
# placode -> hair germ -> hair peg).
#
#   df/dt   = D_f Lap3(f) - mu_f f + s_f * S(x,y) delta_{z=0}        (FGF20, epidermal source)
#   db/dt   = D_b Lap3(b) - mu_b b + s_b * rho                        (BMP4, dermally produced)
#   drho/dt = D_rho Lap3(rho) - chi_f div(m grad f) + chi_b div(m grad b),  m = rho (1 - rho/rho_max)
#   dz_i/dt = v_peg * clip((e_i - e_thr) / (rho_max - e_thr), 0, 1)   (peg depth under placode i)
#
# where S(x,y) = clip((w - w_thr) / (w_max - w_thr), 0, 1) is the normalized placode
# WNT field and e_i(t) is the mean upper-dermal rho in the 3x3 column under placode i.
DERMAL_DEFAULTS: dict[str, float] = {
    "Z": 12,
    "steps": 3200,
    "dt": 0.25,
    "D_f": 0.25,
    "mu_f": 0.04,
    "s_f": 0.5,
    "D_b": 0.12,
    "mu_b": 0.05,
    "s_b": 0.03,
    "D_rho": 0.05,
    "chi_f": 0.25,
    "chi_b": 0.4,
    "rho_0": 1.0,
    "rho_max": 4.0,
    "e_thr": 1.6,
    "v_peg": 0.03,
    "noise": 0.01,
}


def laplacian_3d_slab(u: np.ndarray) -> np.ndarray:
    """7-point Laplacian on an N x N x Z slab: periodic in x, y (axes 0, 1), zero-flux in z (axis 2)."""
    up = np.concatenate([u[:, :, :1], u, u[:, :, -1:]], axis=2)
    lz = up[:, :, 2:] + up[:, :, :-2] - 2.0 * u
    lxy = np.roll(u, 1, 0) + np.roll(u, -1, 0) + np.roll(u, 1, 1) + np.roll(u, -1, 1) - 4.0 * u
    return lxy + lz


def chemotactic_divergence_3d(rho: np.ndarray, c: np.ndarray, chi: float, rho_max: float) -> np.ndarray:
    """
    Conservative face-flux discretisation of chi * div( m grad c ) with volume-filling
    mobility m = rho (1 - rho / rho_max) (Keller & Segel 1970; Painter & Hillen 2002
    volume filling). Lateral faces are periodic; the two outer z faces carry zero flux,
    so the discrete total mass sum(rho) is conserved exactly up to round-off.
    """
    m = rho * np.clip(1.0 - rho / rho_max, 0.0, 1.0)
    out = np.zeros_like(rho)
    for ax in (0, 1):
        flux = 0.5 * (m + np.roll(m, -1, ax)) * (np.roll(c, -1, ax) - c)  # face i+1/2
        out += flux - np.roll(flux, 1, ax)
    flux_z = 0.5 * (m[:, :, :-1] + m[:, :, 1:]) * (c[:, :, 1:] - c[:, :, :-1])
    zero = np.zeros_like(flux_z[:, :, :1])
    flux_z = np.concatenate([zero, flux_z, zero], axis=2)
    out += flux_z[:, :, 1:] - flux_z[:, :, :-1]
    return chi * out


def compute_dermal_linear_stability(
    D_rho: float = DERMAL_DEFAULTS["D_rho"],
    D_b: float = DERMAL_DEFAULTS["D_b"],
    mu_b: float = DERMAL_DEFAULTS["mu_b"],
    s_b: float = DERMAL_DEFAULTS["s_b"],
    chi_b: float = DERMAL_DEFAULTS["chi_b"],
    chi_f: float = DERMAL_DEFAULTS["chi_f"],
    D_f: float = DERMAL_DEFAULTS["D_f"],
    mu_f: float = DERMAL_DEFAULTS["mu_f"],
    s_f: float = DERMAL_DEFAULTS["s_f"],
    rho_0: float = DERMAL_DEFAULTS["rho_0"],
    rho_max: float = DERMAL_DEFAULTS["rho_max"],
) -> dict[str, Any]:
    """
    Linear stability of the epidermis-free dermal subsystem (rho, b) about the uniform
    state rho_0 and of the hypothetical autonomous Keller-Segel system in which the
    dermis itself secreted its chemoattractant.

    Instructed (actual) subsystem, Fourier mode q = k^2, m_0 = rho_0 (1 - rho_0/rho_max):
        M(q) = [[-D_rho q, -chi_b m_0 q], [s_b, -(D_b q + mu_b)]]
        -tr M = (D_rho + D_b) q + mu_b > 0,   det M = D_rho q (D_b q + mu_b) + chi_b m_0 s_b q >= 0
    so no spontaneous dermal condensation occurs without an epidermal FGF20 cue
    (Lean: `dermal_subsystem_instructed_stable`). Autonomous Keller-Segel comparison:
        det_KS(q) = D_rho q (D_f q + mu_f) - chi_f m_0 s_f q < 0  iff  chi_f m_0 s_f > D_rho (D_f q + mu_f)
    (Lean: `keller_segel_fourier_det_sign`).
    """
    m_0 = rho_0 * (1.0 - rho_0 / rho_max)
    q_vals = np.linspace(0.0, 4.0, 401)
    neg_trace = (D_rho + D_b) * q_vals + mu_b
    det_instructed = D_rho * q_vals * (D_b * q_vals + mu_b) + chi_b * m_0 * s_b * q_vals
    det_ks = D_rho * q_vals * (D_f * q_vals + mu_f) - chi_f * m_0 * s_f * q_vals
    ks_threshold_q0 = chi_f * m_0 * s_f / (D_rho * mu_f)  # > 1 means autonomous KS would be unstable at q -> 0
    return {
        "m_0": m_0,
        "q_vals": q_vals,
        "neg_trace": neg_trace,
        "det_instructed": det_instructed,
        "min_det_instructed": float(np.min(det_instructed[1:])),
        "min_neg_trace": float(np.min(neg_trace)),
        "det_ks": det_ks,
        "min_det_ks": float(np.min(det_ks[1:])),
        "ks_autonomy_ratio": float(ks_threshold_q0),
    }


def simulate_dermal_condensate_3d(
    w_epidermis: np.ndarray,
    placode_centroids_yx: np.ndarray,
    Z: int = int(DERMAL_DEFAULTS["Z"]),
    steps: int = int(DERMAL_DEFAULTS["steps"]),
    dt: float = DERMAL_DEFAULTS["dt"],
    D_f: float = DERMAL_DEFAULTS["D_f"],
    mu_f: float = DERMAL_DEFAULTS["mu_f"],
    s_f: float = DERMAL_DEFAULTS["s_f"],
    D_b: float = DERMAL_DEFAULTS["D_b"],
    mu_b: float = DERMAL_DEFAULTS["mu_b"],
    s_b: float = DERMAL_DEFAULTS["s_b"],
    D_rho: float = DERMAL_DEFAULTS["D_rho"],
    chi_f: float = DERMAL_DEFAULTS["chi_f"],
    chi_b: float = DERMAL_DEFAULTS["chi_b"],
    rho_0: float = DERMAL_DEFAULTS["rho_0"],
    rho_max: float = DERMAL_DEFAULTS["rho_max"],
    e_thr: float = DERMAL_DEFAULTS["e_thr"],
    v_peg: float = DERMAL_DEFAULTS["v_peg"],
    fgf20_scale: float = 1.0,
    bmp4_scale: float = 1.0,
    seed: int = 2006,
    record_every: int = 200,
) -> dict[str, Any]:
    """
    Integrate the 3D dermal FGF20/BMP4 chemotaxis slab beneath a fixed 2D epidermal WNT
    placode field and the per-placode hair peg downgrowth ODE.

    `fgf20_scale = 0` is the Fgf20-null genotype (Huh et al. 2013), `bmp4_scale > 1`
    is BMP4 excess (Botchkarev 1999 / Mou 2006 inhibition), `chi_f = 0` is the
    chemotaxis-null ablation. Returns the DC segmentation of the upper dermis, the
    DC-placode registration, per-placode peg depths and the Paus (1999) ordinal stage.
    """
    N = int(w_epidermis.shape[0])
    rng = np.random.default_rng(seed)
    w_thr = max(1.80, float(np.mean(w_epidermis) + 0.75 * np.std(w_epidermis)))
    S = np.clip((w_epidermis - w_thr) / max(1e-9, float(np.max(w_epidermis)) - w_thr), 0.0, 1.0) * fgf20_scale

    f = np.zeros((N, N, Z), dtype=float)
    b = np.zeros((N, N, Z), dtype=float)
    rho = np.full((N, N, Z), rho_0, dtype=float) + rng.normal(0.0, DERMAL_DEFAULTS["noise"], size=(N, N, Z))
    mass_0 = float(np.sum(rho))

    n_pl = int(len(placode_centroids_yx))
    ci = np.rint(placode_centroids_yx).astype(int) % N if n_pl > 0 else np.empty((0, 2), dtype=int)
    offs = np.array([(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 0), (0, 1), (1, -1), (1, 0), (1, 1)])
    col_y = (ci[:, None, 0] + offs[None, :, 0]) % N if n_pl > 0 else None
    col_x = (ci[:, None, 1] + offs[None, :, 1]) % N if n_pl > 0 else None

    peg_depth = np.zeros(n_pl, dtype=float)
    enrich_hist: list[np.ndarray] = []
    peg_hist: list[np.ndarray] = []
    t_hist: list[float] = []
    xz_row = int(ci[0, 0]) if n_pl > 0 else N // 2
    # choose the row that passes through the most placode centroids (for the kymograph / GIF)
    if n_pl > 0:
        row_counts = np.bincount(ci[:, 0], minlength=N)
        xz_row = int(np.argmax(row_counts))
    xz_frames: list[np.ndarray] = [rho[xz_row, :, :].T.copy()]
    top_frames: list[np.ndarray] = [rho[:, :, :2].mean(axis=2)]
    xz_t: list[float] = [0.0]

    for step in range(1, steps + 1):
        df = D_f * laplacian_3d_slab(f) - mu_f * f
        df[:, :, 0] += s_f * S
        db = D_b * laplacian_3d_slab(b) - mu_b * b + bmp4_scale * s_b * rho
        drho = (
            D_rho * laplacian_3d_slab(rho)
            - chemotactic_divergence_3d(rho, f, chi_f, rho_max)
            + chemotactic_divergence_3d(rho, b, chi_b, rho_max)
        )
        f = f + dt * df
        b = b + dt * db
        rho = np.clip(rho + dt * drho, 0.0, rho_max)
        if n_pl > 0:
            e_i = rho[col_y, col_x, :2].mean(axis=(1, 2))
            peg_depth = np.minimum(Z - 1.0, peg_depth + dt * v_peg * np.clip((e_i - e_thr) / (rho_max - e_thr), 0.0, 1.0))
        else:
            e_i = np.array([], dtype=float)
        if step % record_every == 0 or step == steps:
            enrich_hist.append(e_i.copy())
            peg_hist.append(peg_depth.copy())
            t_hist.append(step * dt)
            xz_frames.append(rho[xz_row, :, :].T.copy())
            top_frames.append(rho[:, :, :2].mean(axis=2))
            xz_t.append(step * dt)

    mass_1 = float(np.sum(rho))
    top = rho[:, :, :2].mean(axis=2)
    dc_mask = top > e_thr
    lab, n_dc = label_periodic_2d(dc_mask)
    if n_dc > 0:
        dc_centroids = periodic_centroids_2d(lab, n_dc)
    else:
        dc_centroids = np.empty((0, 2), dtype=float)

    # DC <-> placode registration (periodic nearest neighbour)
    if n_dc > 0 and n_pl > 0:
        d = placode_centroids_yx[:, None, :] - dc_centroids[None, :, :]
        d = (d + N / 2.0) % N - N / 2.0
        dist = np.sqrt(np.sum(d * d, axis=2))
        nearest = dist.min(axis=1)
        placode_has_dc = nearest < 2.5
        offsets = nearest[placode_has_dc]
        mean_offset = float(np.mean(offsets)) if len(offsets) > 0 else 0.0
        max_offset = float(np.max(offsets)) if len(offsets) > 0 else 0.0
    else:
        placode_has_dc = np.zeros(n_pl, dtype=bool)
        mean_offset = 0.0
        max_offset = 0.0
    frac_with_dc = float(np.mean(placode_has_dc)) if n_pl > 0 else 0.0

    excess = np.clip(rho - rho_0, 0.0, None)
    z_idx = np.arange(Z, dtype=float)
    dc_depth_com = float(np.sum(excess * z_idx[None, None, :]) / max(1e-12, np.sum(excess)))

    frac_depth = peg_depth / float(max(1, Z - 1))
    stage = np.where(
        ~placode_has_dc, 1,
        np.where(frac_depth < 0.25, 2, np.where(frac_depth < 0.5, 3, 4))
    ) if n_pl > 0 else np.array([], dtype=int)

    return {
        "N": N,
        "Z": Z,
        "rho": rho,
        "f": f,
        "b": b,
        "top_density": top,
        "dc_mask": dc_mask,
        "n_dc": int(n_dc),
        "dc_centroids_yx": dc_centroids,
        "n_placodes": n_pl,
        "placode_has_dc": placode_has_dc,
        "frac_placodes_with_dc": frac_with_dc,
        "dc_placode_ratio": float(n_dc / max(1, n_pl)),
        "mean_dc_offset_px": mean_offset,
        "max_dc_offset_px": max_offset,
        "dc_depth_com": dc_depth_com,
        "peak_enrichment": float(np.max(top) / rho_0),
        "mass_drift": (mass_1 - mass_0) / mass_0,
        "peg_depth": peg_depth,
        "peg_depth_frac": frac_depth,
        "mean_peg_depth_frac": float(np.mean(frac_depth)) if n_pl > 0 else 0.0,
        "stage": stage,
        "enrichment_history": np.array(enrich_hist),
        "peg_history": np.array(peg_hist),
        "t_history": np.array(t_hist),
        "xz_row": xz_row,
        "xz_frames": xz_frames,
        "top_frames": top_frames,
        "xz_t": xz_t,
        "fgf20_source": S,
        "params": {"Z": Z, "steps": steps, "dt": dt, "chi_f": chi_f, "chi_b": chi_b, "fgf20_scale": fgf20_scale, "bmp4_scale": bmp4_scale},
    }


def simulate_dermal_condensate_genotypes(wt_2d: dict[str, Any], seed: int = 2006, chi_f: float = DERMAL_DEFAULTS["chi_f"]) -> dict[str, Any]:
    """
    Run the 3D dermal coupling beneath the Wild-Type 2D placode lattice for four genotypes
    plus a 5-point FGF20 chemotactic-sensitivity sweep:
      1. WT: one DC beneath each of the 15 placodes, hair pegs descend.
      2. Fgf20-null (fgf20_scale = 0; Huh et al. 2013): placodes persist, 0 DCs, 0 downgrowth.
      3. BMP4 excess (bmp4_scale = 5; Botchkarev 1999 / Mou 2006): DC enrichment stays
         below threshold, downgrowth suppressed.
      4. Chemotaxis-null (chi_f = 0): FGF20 present but unread, 0 DCs.
      5. chi_f sweep in {0.0, 0.1, 0.15, 0.2, 0.25}: DC count and mean peg depth monotone.
    """
    w = wt_2d["w"]
    c_yx = wt_2d["centroids_yx"]
    wt = simulate_dermal_condensate_3d(w, c_yx, chi_f=chi_f, seed=seed)
    fgf20_ko = simulate_dermal_condensate_3d(w, c_yx, chi_f=chi_f, fgf20_scale=0.0, seed=seed, record_every=3200)
    bmp4_excess = simulate_dermal_condensate_3d(w, c_yx, chi_f=chi_f, bmp4_scale=5.0, seed=seed, record_every=3200)
    chemo_null = simulate_dermal_condensate_3d(w, c_yx, chi_f=0.0, seed=seed, record_every=3200)
    sweep_chi = [0.0, 0.1, 0.15, 0.2, 0.25]
    sweep: list[dict[str, Any]] = []
    for cf in sweep_chi:
        if abs(cf - chi_f) < 1e-12:
            s = wt
        elif cf == 0.0:
            s = chemo_null
        else:
            s = simulate_dermal_condensate_3d(w, c_yx, chi_f=cf, seed=seed, record_every=3200)
        sweep.append({"chi_f": cf, "n_dc": s["n_dc"], "mean_peg_depth_frac": s["mean_peg_depth_frac"], "peak_enrichment": s["peak_enrichment"]})
    depths = np.array([s["mean_peg_depth_frac"] for s in sweep])
    chis = np.array(sweep_chi)
    r_sweep = float(np.corrcoef(chis, depths)[0, 1]) if np.std(depths) > 0 else 0.0
    monotone = bool(np.all(np.diff(depths) >= -1e-12))
    return {
        "wt": wt,
        "fgf20_ko": fgf20_ko,
        "bmp4_excess": bmp4_excess,
        "chemo_null": chemo_null,
        "chi_sweep": sweep,
        "chi_sweep_pearson_r": r_sweep,
        "chi_sweep_monotone": monotone,
        "linear": compute_dermal_linear_stability(chi_f=chi_f),
    }


# --------------------------------------------------------------------------- #
# 4. Biological Specimen Computer-Vision Segmentation (Mus musculus E12.5-E13.5)
# --------------------------------------------------------------------------- #
def segment_follicle_specimen(specimen_path: Path = SPECIMEN_PATH) -> dict[str, Any]:
    """
    Segment the embryonic mouse (Mus musculus, E12.5-E13.5) whole-mount in situ
    hybridization hair/vibrissa follicle placode specimen
    (`datasets/wnt_dkk_hair_follicle/hair_follicle_placode_specimen.jpg`, Maddox & Condie
    2004 BMC Dev. Biol. 4:4, CC BY 2.5):
      1. Crop the high-magnification regular placode field in Panel B
         (`x in [625, 1165], y in [30, 460]`).
      2. Compute the background-subtracted in situ hybridization optical density
         `S(x,y) = G_sigma=22(L) - G_sigma=2(L)` to isolate the dark purple-blue
         stained follicle placodes while masking out black/white arrow annotations.
      3. Threshold at the 94th percentile, apply morphological binary opening, and
         extract connected components (`area in [25, 900]` px^2), yielding N = 16
         discrete follicle placodes.
      4. Build the Delaunay triangulation and cKDTree nearest-neighbor graph across the
         16 placode centroids to quantify inter-follicular spacing (`70.86 +- 22.92` px,
         `CV = 0.3234`).
    """
    img = np.asarray(Image.open(specimen_path).convert("RGB"))
    x0, y0, x1, y1 = 625, 30, 1165, 460
    roi_rgb = img[y0:y1, x0:x1]
    gray = 0.299 * roi_rgb[:, :, 0] + 0.587 * roi_rgb[:, :, 1] + 0.114 * roi_rgb[:, :, 2]

    bg = ndi.gaussian_filter(gray, sigma=22.0)
    stain = bg - ndi.gaussian_filter(gray, sigma=2.0)
    valid = (gray > 28.0) & (gray < 235.0)
    thr = float(np.percentile(stain[valid], 94))
    mask_raw = (stain > thr) & valid
    mask_open = ndi.binary_opening(mask_raw, iterations=1)
    lab, n_comp = ndi.label(mask_open)
    areas = ndi.sum(mask_open, lab, range(1, n_comp + 1))

    keep_labels = [i + 1 for i, a in enumerate(areas) if 25 <= a <= 900]
    clean_mask = np.isin(lab, keep_labels)
    centroids_yx = np.array([ndi.center_of_mass(mask_open, lab, k) for k in keep_labels], dtype=float)
    centroids_xy = centroids_yx[:, [1, 0]] if len(centroids_yx) > 0 else np.empty((0, 2), dtype=float)
    kept_areas = np.array([areas[k - 1] for k in keep_labels], dtype=float)

    if len(centroids_xy) >= 4:
        tree = cKDTree(centroids_xy)
        dists, _ = tree.query(centroids_xy, k=2)
        nn_dists = dists[:, 1]
        mean_nn = float(np.mean(nn_dists))
        std_nn = float(np.std(nn_dists))
        cv_nn = float(std_nn / mean_nn)
        tri = Delaunay(centroids_xy)
        simplices = tri.simplices
    else:
        nn_dists = np.array([], dtype=float)
        mean_nn = 0.0
        std_nn = 0.0
        cv_nn = 0.0
        simplices = np.empty((0, 3), dtype=int)

    return {
        "roi_rgb": roi_rgb,
        "roi_xyxy": [x0, y0, x1, y1],
        "clean_mask": clean_mask,
        "n_placodes_segmented": int(len(keep_labels)),
        "centroids_xy": centroids_xy,
        "placode_areas_px": kept_areas,
        "mean_placode_area_px": float(np.mean(kept_areas)) if len(kept_areas) > 0 else 0.0,
        "nn_dists_px": nn_dists,
        "mean_nn_px": mean_nn,
        "std_nn_px": std_nn,
        "cv_nn": cv_nn,
        "delaunay_simplices": simplices,
    }


# --------------------------------------------------------------------------- #
# 5. 12-Panel Publication Figure & Animated Growth/Secondary-Insertion/Dermal GIF
# --------------------------------------------------------------------------- #
def render_outputs(
    c_w: float = 0.05,
    c_h: float = 0.05,
    mu_w: float = 0.02,
    mu_h: float = 0.03,
    D_w: float = 0.015,
    D_h: float = 0.45,
    seed: int = 2006,
    skip_gif: bool = False,
    chi_f: float = DERMAL_DEFAULTS["chi_f"],
) -> dict[str, Any]:
    """Render the 12-panel comparison synthesis figure and animated growth GIF."""
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    verify_dataset_sha256()

    stab = compute_linear_stability(mu_w=mu_w, mu_h=mu_h, D_w=D_w, D_h=D_h, c_w=c_w, c_h=c_h)
    waves_1d = simulate_primary_to_secondary_wave_insertion(D_w=D_w, D_h=D_h, seed=seed)
    dom_sweep = simulate_domain_scaling_sweep_1d(D_w=D_w, D_h=D_h, seed=seed)
    pheno_2d = simulate_transgenic_phenotypes_2d(seed=seed)
    spec_seg = segment_follicle_specimen(SPECIMEN_PATH)
    dermal = simulate_dermal_condensate_genotypes(pheno_2d["wt"], seed=seed, chi_f=chi_f)

    fig, axes = plt.subplots(4, 3, figsize=(16.5, 18.4), dpi=135)
    fig.patch.set_facecolor("#0b0f17")
    for ax in axes.flat:
        ax.set_facecolor("#131b29")
        ax.tick_params(colors="#cbd5e1", labelsize=8.5)
        for sp in ax.spines.values():
            sp.set_color("#334155")

    # Panel A1: WNT/DKK Linear Dispersion Relation sigma(k) across d = D_h / D_w
    ax = axes[0, 0]
    ax.plot(stab["k_vals"], stab["re_sigma_super"], color="#38bdf8", lw=2.3, label=rf"$d = {stab['d_ratio']:.0f} > d_c$ (rapid DKK diffusion)")
    ax.plot(stab["k_vals"], stab["re_sigma_crit"], color="#f59e0b", lw=1.8, ls="--", label=rf"$d = d_c = (\sqrt{{2}}+1)^2 \approx {stab['d_c']:.2f}$")
    ax.plot(stab["k_vals"], stab["re_sigma_equal"], color="#94a3b8", lw=1.8, ls=":", label=r"$d = 1.0$ ($D_w = D_h$ control)")
    ax.axhline(0.0, color="#64748b", lw=1.0, ls="--")
    ax.axvline(stab["k_dom"], color="#38bdf8", lw=1.2, ls=":", label=rf"$k_{{\mathrm{{dom}}}} = {stab['k_dom']:.2f}$ ($\lambda = {stab['lambda_dom']:.1f}$)")
    ax.set_title("A1. WNT/DKK Linear Dispersion $\\mathrm{Re}\\,\\sigma(k)$", color="#f8fafc", fontsize=10.5, fontweight="bold")
    ax.set_xlabel("Spatial wavenumber $k$ (rad/cell)", color="#cbd5e1", fontsize=9)
    ax.set_ylabel("Growth rate $\\mathrm{Re}\\,\\sigma(k)$", color="#cbd5e1", fontsize=9)
    ax.legend(facecolor="#0f172a", edgecolor="#334155", labelcolor="#f8fafc", fontsize=7.5, loc="lower left")

    # Panel A2: Primary Wave (E14.5, N=40) vs Secondary Inter-Follicular Wave Insertion (E16.5, N=80)
    ax = axes[0, 1]
    prim = waves_1d["primary"]
    sec = waves_1d["secondary"]
    s_prim = prim["x"] / float(len(prim["x"]))
    s_sec = sec["x"] / float(len(sec["x"]))
    ax.plot(s_prim, prim["w"], color="#f59e0b", lw=2.0, ls="--", label=f"E14.5 Primary wave $w(x)$ ($N_1={prim['n_peaks']}$ peaks)")
    ax.plot(s_sec, sec["w"], color="#f43f5e", lw=2.4, label=f"E16.5 Expanded $2\\times$ wave $w(x)$ ($N_2={sec['n_peaks']}$ peaks)")
    ax.plot(s_sec, sec["h"] * 0.65, color="#a855f7", lw=1.6, ls=":", label="Long-range DKK inhibitor $0.65\\,h(x)$")
    if len(sec["peaks"]) > 0:
        ax.scatter(s_sec[sec["peaks"]], sec["w"][sec["peaks"]], color="#38bdf8", s=40, zorder=5)
    ax.set_title(
        f"A2. Inter-Follicular Secondary Wave Insertion ({prim['n_peaks']} $\\to$ {sec['n_peaks']} Placodes)",
        color="#f8fafc",
        fontsize=10.5,
        fontweight="bold",
    )
    ax.set_xlabel("Normalized epidermal coordinate $x / L \\in [0, 1)$", color="#cbd5e1", fontsize=9)
    ax.set_ylabel("Morphogen concentration", color="#cbd5e1", fontsize=9)
    ax.legend(facecolor="#0f172a", edgecolor="#334155", labelcolor="#f8fafc", fontsize=7.3, loc="upper right")

    # Panel A3: Space-Time Kymograph of Primary Wave -> 2x Expansion -> Secondary Insertion
    ax = axes[0, 2]
    hist_p = prim["history_w"]
    hist_s = sec["history_w"]
    s_grid = np.linspace(0.0, 1.0, len(sec["x"]), endpoint=False)
    s_old = np.linspace(0.0, 1.0, len(prim["x"]), endpoint=False)
    p_interp = np.array([np.interp(s_grid, s_old, row, period=1.0) for row in hist_p])
    kymo = np.vstack([p_interp, hist_s])
    im_k = ax.imshow(kymo, aspect="auto", origin="lower", cmap="magma", extent=[0.0, 1.0, 0, len(kymo) - 1])
    ax.axhline(len(p_interp) - 0.5, color="#38bdf8", lw=1.7, ls="--", label="$2\\times$ Epidermal expansion (E14.5$\\to$E16.5)")
    ax.set_title("A3. Kymograph: Primary Wave $\\to$ Secondary Insertion", color="#f8fafc", fontsize=10.5, fontweight="bold")
    ax.set_xlabel("Normalized epidermal coordinate $x / L$", color="#cbd5e1", fontsize=9)
    ax.set_ylabel("Developmental time frame", color="#cbd5e1", fontsize=9)
    ax.legend(facecolor="#0f172a", edgecolor="#334155", labelcolor="#f8fafc", fontsize=7.5, loc="upper right")
    cb = fig.colorbar(im_k, ax=ax, fraction=0.046, pad=0.04)
    cb.ax.tick_params(colors="#cbd5e1", labelsize=7.5)

    # Panel B1: 2D Wild-Type Epidermal Placode Lattice (N = 16 placodes)
    ax = axes[1, 0]
    wt = pheno_2d["wt"]
    im_wt = ax.imshow(wt["w"], cmap="magma", origin="lower")
    if len(wt["centroids_yx"]) > 0:
        ax.scatter(wt["centroids_yx"][:, 1], wt["centroids_yx"][:, 0], color="#38bdf8", edgecolor="#0f172a", s=38, zorder=5, label=f"WT placodes ($N={wt['n_placodes']}$)")
    ax.set_title(
        f"B1. Wild-Type 2D WNT Placodes ($N={wt['n_placodes']}$, NN={wt['mean_nn_px']:.1f}px)",
        color="#f8fafc",
        fontsize=10.5,
        fontweight="bold",
    )
    ax.set_xlabel("Epidermal $x$ (cells)", color="#cbd5e1", fontsize=9)
    ax.set_ylabel("Epidermal $y$ (cells)", color="#cbd5e1", fontsize=9)
    ax.legend(facecolor="#0f172a", edgecolor="#334155", labelcolor="#f8fafc", fontsize=7.5, loc="upper right")
    cb1 = fig.colorbar(im_wt, ax=ax, fraction=0.046, pad=0.04)
    cb1.ax.tick_params(colors="#cbd5e1", labelsize=7.5)

    # Panel B2: Transgenic K14-Wnt Overexpression High-Density Clustering (Sick et al. 2006 Fig. 2B-D)
    ax = axes[1, 1]
    wnt_o = pheno_2d["wnt_over"]
    im_wo = ax.imshow(wnt_o["w"], cmap="magma", origin="lower")
    if len(wnt_o["centroids_yx"]) > 0:
        ax.scatter(wnt_o["centroids_yx"][:, 1], wnt_o["centroids_yx"][:, 0], color="#22c55e", edgecolor="#0f172a", s=38, zorder=5, label=f"K14-Wnt++ ($N={wnt_o['n_placodes']}$)")
    ax.set_title(
        f"B2. Transgenic K14-Wnt Overexpression ($N={wnt_o['n_placodes']}$, NN={wnt_o['mean_nn_px']:.1f}px)",
        color="#f8fafc",
        fontsize=10.5,
        fontweight="bold",
    )
    ax.set_xlabel("Epidermal $x$ (cells)", color="#cbd5e1", fontsize=9)
    ax.set_ylabel("Epidermal $y$ (cells)", color="#cbd5e1", fontsize=9)
    ax.legend(facecolor="#0f172a", edgecolor="#334155", labelcolor="#f8fafc", fontsize=7.5, loc="upper right")
    cb2 = fig.colorbar(im_wo, ax=ax, fraction=0.046, pad=0.04)
    cb2.ax.tick_params(colors="#cbd5e1", labelsize=7.5)

    # Panel B3: Saturated Enlarged Follicles vs Dkk1 Blockade Dose-Response Sweep
    ax = axes[1, 2]
    enl = pheno_2d["enlarged"]
    im_en = ax.imshow(enl["w"], cmap="magma", origin="lower")
    ax.contour(enl["mask"], levels=[0.5], colors=["#eab308"], linewidths=1.3)
    ax.set_title(
        f"B3. Saturated Enlarged Follicles ($N={enl['n_placodes']}$, Area={enl['mean_area_px']:.1f}px$^2$)",
        color="#f8fafc",
        fontsize=10.5,
        fontweight="bold",
    )
    ax.set_xlabel(f"Dkk1++ blockade ($\\rho_h=0.12$) $\\to$ {pheno_2d['dkk_block']['n_placodes']} placodes | Equal-diff $\\to$ {pheno_2d['equal_diff']['n_placodes']} placodes", color="#cbd5e1", fontsize=8.3)
    ax.set_ylabel("Epidermal $y$ (cells)", color="#cbd5e1", fontsize=9)
    cb3 = fig.colorbar(im_en, ax=ax, fraction=0.046, pad=0.04)
    cb3.ax.tick_params(colors="#cbd5e1", labelsize=7.5)

    # Panel C1: Mus musculus E12.5-E13.5 In Situ Hybridization Specimen Segmentation (N = 16 placodes)
    ax = axes[2, 0]
    ax.imshow(spec_seg["roi_rgb"])
    ax.contour(spec_seg["clean_mask"], levels=[0.5], colors=["#38bdf8"], linewidths=1.2)
    pts_xy = spec_seg["centroids_xy"]
    if len(spec_seg["delaunay_simplices"]) > 0:
        ax.triplot(pts_xy[:, 0], pts_xy[:, 1], spec_seg["delaunay_simplices"], color="#f59e0b", lw=0.9, alpha=0.75, label="Delaunay neighbor graph")
    if len(pts_xy) > 0:
        ax.scatter(pts_xy[:, 0], pts_xy[:, 1], color="#f43f5e", edgecolor="#f8fafc", s=48, zorder=6, label=f"Segmented placodes ($N={spec_seg['n_placodes_segmented']}$)")
    ax.set_title(
        f"C1. Mus musculus Placode Specimen Segmentation ($N={spec_seg['n_placodes_segmented']}$)",
        color="#f8fafc",
        fontsize=10.5,
        fontweight="bold",
    )
    ax.legend(facecolor="#0f172a", edgecolor="#334155", labelcolor="#f8fafc", fontsize=7.2, loc="lower right")
    ax.axis("off")

    # Panel C2: Normalized Nearest-Neighbor Spacing Comparison & Ectopic Dkk1 Suppression Curve
    ax = axes[2, 1]
    rh_vals = [r["rho_h"] for r in pheno_2d["dkk_sweep"]]
    np_vals = [r["n_placodes"] for r in pheno_2d["dkk_sweep"]]
    ax.plot(rh_vals, np_vals, "o-", color="#a855f7", lw=2.3, markersize=6.5, label="2D PDE placode count $N(\\rho_h)$")
    ax.axhline(spec_seg["n_placodes_segmented"], color="#38bdf8", lw=1.6, ls="--", label=f"Specimen Panel B ($N={spec_seg['n_placodes_segmented']}$ placodes)")
    ax.axhline(wnt_o["n_placodes"], color="#22c55e", lw=1.5, ls=":", label=f"K14-Wnt++ overexpression ($N={wnt_o['n_placodes']}$)")
    ax.axhline(0, color="#f43f5e", lw=1.2, ls="-.", label="Andl et al. (2002) Dkk1++ alopecia ($N=0$)")
    ax.set_title("C2. Ectopic Dkk1 Dose-Response & Genotype Comparison", color="#f8fafc", fontsize=10.5, fontweight="bold")
    ax.set_xlabel("Basal DKK inhibitor expression rate $\\rho_h$", color="#cbd5e1", fontsize=9)
    ax.set_ylabel("Initiated hair follicle placodes $N$", color="#cbd5e1", fontsize=9)
    ax.legend(facecolor="#0f172a", edgecolor="#334155", labelcolor="#f8fafc", fontsize=7.3, loc="center right")

    # Panel C3: Allometric Domain-Length Scaling Sweep N_peaks(L)
    ax = axes[2, 2]
    L_arr = np.array(dom_sweep["domain_lengths"], dtype=float)
    n_arr = np.array(dom_sweep["peak_counts"], dtype=float)
    ax.plot(L_arr, n_arr, "o-", color="#38bdf8", lw=2.2, markersize=6.5, label="PDE placode count $N_{\\mathrm{peaks}}(L)$")
    ax.plot(
        L_arr,
        L_arr / dom_sweep["mean_nonlinear_wavelength"],
        "--",
        color="#f59e0b",
        lw=1.8,
        label=rf"Linear wavelength law $L / \bar{{\lambda}}$ ($\bar{{\lambda}}={dom_sweep['mean_nonlinear_wavelength']:.2f}$)",
    )
    ax.set_title(
        f"C3. Epidermal Domain Scaling ($r = {dom_sweep['pearson_r']:.4f}$, RMSE = {dom_sweep['rmse_peaks']:.3f})",
        color="#f8fafc",
        fontsize=10.5,
        fontweight="bold",
    )
    ax.set_xlabel("Epidermal transect length $L$ (cells)", color="#cbd5e1", fontsize=9)
    ax.set_ylabel("Hair follicle placode count $N_{\\mathrm{peaks}}$", color="#cbd5e1", fontsize=9)
    ax.legend(facecolor="#0f172a", edgecolor="#334155", labelcolor="#f8fafc", fontsize=7.5, loc="upper left")

    # ----- Row D (Pass 3): 3D dermal condensate FGF20/BMP4 chemotaxis & hair peg downgrowth -----
    dwt = dermal["wt"]
    Zd = dwt["Z"]
    Nd = dwt["N"]

    # Panel D1: 3D view — epidermal WNT placode plane (z = 0) + DC voxels + hair pegs
    axes[3, 0].remove()
    ax3 = fig.add_subplot(4, 3, 10, projection="3d")
    ax3.set_facecolor("#131b29")
    xx, yy = np.meshgrid(np.arange(Nd), np.arange(Nd))
    w_plane = wt["w"]
    w_norm = (w_plane - w_plane.min()) / max(1e-9, w_plane.max() - w_plane.min())
    ax3.plot_surface(xx, yy, np.zeros_like(xx, dtype=float), facecolors=plt.cm.magma(w_norm), rstride=1, cstride=1, shade=False, alpha=0.38, linewidth=0)
    vox = np.argwhere(dwt["rho"] > DERMAL_DEFAULTS["e_thr"])
    if len(vox) > 0:
        ax3.scatter(vox[:, 1], vox[:, 0], -vox[:, 2] - 0.5, c=dwt["rho"][vox[:, 0], vox[:, 1], vox[:, 2]], cmap="viridis", s=9, alpha=0.75, depthshade=False, label=f"DC voxels $\\rho > {DERMAL_DEFAULTS['e_thr']}$ ($N_{{DC}}={dwt['n_dc']}$)")
    for i, (cy, cx) in enumerate(wt["centroids_yx"]):
        ax3.plot([cx, cx], [cy, cy], [0.0, -dwt["peg_depth"][i]], color="#f43f5e", lw=2.0, alpha=0.95, label="Hair peg downgrowth $z_i$" if i == 0 else None)
    ax3.set_zlim(-(Zd - 1), 0.5)
    ax3.set_xlim(0, Nd)
    ax3.set_ylim(0, Nd)
    ax3.view_init(elev=16, azim=-58)
    ax3.set_xlabel("$x$ (cells)", color="#cbd5e1", fontsize=8, labelpad=2)
    ax3.set_ylabel("$y$ (cells)", color="#cbd5e1", fontsize=8, labelpad=2)
    ax3.set_zlabel("dermal depth $-z$", color="#cbd5e1", fontsize=8, labelpad=2)
    ax3.tick_params(colors="#cbd5e1", labelsize=7)
    for pane in (ax3.xaxis.pane, ax3.yaxis.pane, ax3.zaxis.pane):
        pane.set_facecolor("#131b29")
        pane.set_edgecolor("#334155")
    ax3.set_title(
        f"D1. 3D Dermal Condensates + Pegs ($N_{{DC}}/N_{{pl}}={dwt['n_dc']}/{dwt['n_placodes']}$, offset {dwt['mean_dc_offset_px']:.2f}px)",
        color="#f8fafc", fontsize=9.8, fontweight="bold", pad=8,
    )
    ax3.legend(facecolor="#0f172a", edgecolor="#334155", labelcolor="#f8fafc", fontsize=7.2, loc="upper left")

    # Panel D2: x-z cross-section through the placode row: mesenchymal density + FGF20 + pegs
    ax = axes[3, 1]
    xz = dwt["xz_frames"][-1]
    im_xz = ax.imshow(xz, cmap="viridis", origin="upper", aspect="auto", extent=[0, Nd, Zd - 0.5, -0.5], vmin=0.0, vmax=max(2.0, float(xz.max())))
    f_xz = dwt["f"][dwt["xz_row"], :, :].T
    ax.contour(np.arange(Nd) + 0.5, np.arange(Zd), f_xz, levels=4, colors=["#f59e0b"], linewidths=0.9, alpha=0.9)
    row_pl = [i for i, (cy, cx) in enumerate(wt["centroids_yx"]) if int(np.rint(cy)) % Nd == dwt["xz_row"]]
    for i in row_pl:
        cx = wt["centroids_yx"][i][1]
        ax.plot([cx, cx], [-0.5, dwt["peg_depth"][i]], color="#f43f5e", lw=3.2, solid_capstyle="butt", label="Hair peg $z_i(t_{end})$" if i == row_pl[0] else None)
        ax.scatter([cx], [-0.5], marker="v", color="#38bdf8", s=48, zorder=6, label="Epidermal placode (z=0)" if i == row_pl[0] else None)
    ax.axhline(-0.5, color="#38bdf8", lw=1.2)
    ax.set_title(
        f"D2. Dermal $x$-$z$ Section (y={dwt['xz_row']}): $\\rho$, FGF20, Pegs ({dwt['mean_peg_depth_frac']*100:.0f}% depth)",
        color="#f8fafc", fontsize=9.8, fontweight="bold",
    )
    ax.set_xlabel("Epidermal $x$ (cells)", color="#cbd5e1", fontsize=9)
    ax.set_ylabel("Dermal depth $z$ (layers below epidermis)", color="#cbd5e1", fontsize=9)
    ax.legend(facecolor="#0f172a", edgecolor="#334155", labelcolor="#f8fafc", fontsize=7.2, loc="lower right")
    cbd = fig.colorbar(im_xz, ax=ax, fraction=0.046, pad=0.04)
    cbd.ax.tick_params(colors="#cbd5e1", labelsize=7.5)
    cbd.set_label("mesenchymal density $\\rho/\\rho_0$", color="#cbd5e1", fontsize=8)

    # Panel D3: hair peg downgrowth time courses by genotype + chi_f sweep
    ax = axes[3, 2]
    t_h = dwt["t_history"]
    ph = dwt["peg_history"] / float(max(1, Zd - 1))
    ax.plot(t_h, ph.mean(axis=1), color="#f43f5e", lw=2.4, label=f"WT (FGF20$^+$, $N_{{DC}}={dwt['n_dc']}$)")
    ax.fill_between(t_h, ph.mean(axis=1) - ph.std(axis=1), ph.mean(axis=1) + ph.std(axis=1), color="#f43f5e", alpha=0.18, label="WT $\\pm 1$ s.d. over placodes")
    for key, col, lab in (("bmp4_excess", "#a855f7", "BMP4 excess ($5\\times s_b$)"), ("fgf20_ko", "#94a3b8", "Fgf20$^{-/-}$ (Huh 2013)"), ("chemo_null", "#38bdf8", "chemotaxis-null ($\\chi_f=0$)")):
        g = dermal[key]
        ax.plot([0.0, t_h[-1]], [g["mean_peg_depth_frac"]] * 2, color=col, lw=1.8, ls="--", label=f"{lab}: $N_{{DC}}={g['n_dc']}$, depth {g['mean_peg_depth_frac']*100:.0f}%")
    ax.axhline(0.25, color="#64748b", lw=0.8, ls=":")
    ax.axhline(0.5, color="#64748b", lw=0.8, ls=":")
    ax.text(t_h[-1] * 0.01, 0.26, "stage 3 (early peg) — model convention", color="#94a3b8", fontsize=6.8)
    ax.text(t_h[-1] * 0.01, 0.51, "stage 4 (hair peg)", color="#94a3b8", fontsize=6.8)
    ax.set_ylim(-0.02, 1.0)
    ax.set_title("D3. Hair Peg Downgrowth by Genotype & $\\chi_f$ Sweep", color="#f8fafc", fontsize=9.8, fontweight="bold")
    ax.set_xlabel("Dermal time after placode specification", color="#cbd5e1", fontsize=9)
    ax.set_ylabel("Peg depth / dermal depth", color="#cbd5e1", fontsize=9)
    ax.legend(facecolor="#0f172a", edgecolor="#334155", labelcolor="#f8fafc", fontsize=6.9, loc="upper left")
    ins = ax.inset_axes([0.60, 0.07, 0.37, 0.34])
    ins.set_facecolor("#0f172a")
    sw_chi = [s["chi_f"] for s in dermal["chi_sweep"]]
    sw_dep = [s["mean_peg_depth_frac"] for s in dermal["chi_sweep"]]
    sw_ndc = [s["n_dc"] for s in dermal["chi_sweep"]]
    ins.plot(sw_chi, sw_dep, "o-", color="#f43f5e", lw=1.6, markersize=4)
    ins.set_ylabel("peg depth", color="#cbd5e1", fontsize=6.5)
    ins.set_xlabel("$\\chi_f$", color="#cbd5e1", fontsize=6.5)
    ins.tick_params(colors="#cbd5e1", labelsize=6)
    ins2 = ins.twinx()
    ins2.plot(sw_chi, sw_ndc, "s--", color="#22c55e", lw=1.2, markersize=3.5)
    ins2.set_ylabel("$N_{DC}$", color="#22c55e", fontsize=6.5)
    ins2.tick_params(colors="#22c55e", labelsize=6)
    ins.set_title(f"$\\chi_f$ sweep ($r={dermal['chi_sweep_pearson_r']:.3f}$)", color="#f8fafc", fontsize=7)
    for sp in list(ins.spines.values()) + list(ins2.spines.values()):
        sp.set_color("#334155")

    fig.suptitle(
        "Sick et al. (2006) Science & Andl et al. (2002) WNT/DKK Reaction-Diffusion Hair Follicle Spacing, Secondary Wave Insertion, Transgenic Phenotypes\n"
        "& Pass 3: 3D Dermal Condensate FGF20/BMP4 Chemotaxis + Hair Peg Downgrowth (Huh 2013 / Glover 2017 / Paus 1999)",
        color="#f8fafc",
        fontsize=12.0,
        fontweight="bold",
        y=0.995,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)  # 3D axes are not tight_layout-aware
        fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.972])
    fig.savefig(OUT_PNG, dpi=135)
    plt.close(fig)

    if not skip_gif:
        frames_pil: list[Image.Image] = []
        hist_p_w = prim["history_w"]
        hist_p_h = prim["history_h"]
        hist_s_w = sec["history_w"]
        hist_s_h = sec["history_h"]
        wt_frames = wt["frames_w"]

        p_indices = np.linspace(0, len(hist_p_w) - 1, 15, dtype=int)
        s_indices = np.linspace(0, len(hist_s_w) - 1, 15, dtype=int)

        for k_i, idx in enumerate(p_indices):
            f_fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.8, 3.8), dpi=90)
            f_fig.patch.set_facecolor("#0b0f17")
            for a_sub in (ax1, ax2):
                a_sub.set_facecolor("#131b29")
                a_sub.tick_params(colors="#cbd5e1", labelsize=8)
                for sp in a_sub.spines.values():
                    sp.set_color("#334155")
            ax1.plot(s_prim, hist_p_w[idx], color="#f43f5e", lw=2.3, label="WNT $w(x,t)$")
            ax1.plot(s_prim, hist_p_h[idx] * 0.65, color="#a855f7", lw=1.8, ls="--", label="DKK $0.65\\,h(x,t)$")
            ax1.set_ylim(0, 3.8)
            ax1.set_xlim(0, 1.0)
            ax1.set_title(f"Phase 1 (E14.5): Primary Placode Wave (N={prim['n_peaks']})", color="#f8fafc", fontsize=9.2)
            ax1.set_xlabel("Epidermal coordinate $x/L_0$", color="#cbd5e1", fontsize=8.5)
            ax1.legend(loc="upper right", facecolor="#0f172a", edgecolor="#334155", labelcolor="#f8fafc", fontsize=7.5)

            w2d_idx = min(k_i, len(wt_frames) - 1)
            ax2.imshow(wt_frames[w2d_idx], cmap="magma", origin="lower", vmin=0.0, vmax=4.2)
            ax2.set_title(f"2D Wild-Type Epidermal Sheet (step {w2d_idx * 250})", color="#f8fafc", fontsize=9.2)
            ax2.axis("off")
            f_fig.tight_layout()
            f_fig.canvas.draw()
            rgba = np.asarray(f_fig.canvas.buffer_rgba())
            frames_pil.append(Image.fromarray(rgba[:, :, :3]))
            plt.close(f_fig)

        for k_i, idx in enumerate(s_indices):
            f_fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.8, 3.8), dpi=90)
            f_fig.patch.set_facecolor("#0b0f17")
            for a_sub in (ax1, ax2):
                a_sub.set_facecolor("#131b29")
                a_sub.tick_params(colors="#cbd5e1", labelsize=8)
                for sp in a_sub.spines.values():
                    sp.set_color("#334155")
            ax1.plot(s_sec, hist_s_w[idx], color="#f43f5e", lw=2.3, label="WNT $w(x,t)$ (2x domain)")
            ax1.plot(s_sec, hist_s_h[idx] * 0.65, color="#a855f7", lw=1.8, ls="--", label="DKK $0.65\\,h(x,t)$")
            ax1.plot(s_sec, waves_1d["w_expanded"], color="#94a3b8", lw=1.3, ls=":", label="Post-expansion $w(x,0)$")
            ax1.set_ylim(0, 3.8)
            ax1.set_xlim(0, 1.0)
            ax1.set_title(f"Phase 2 (E16.5): Inter-Follicular Insertion (4->{sec['n_peaks']} peaks)", color="#f8fafc", fontsize=9.2)
            ax1.set_xlabel("Expanded coordinate $x/(2L_0)$", color="#cbd5e1", fontsize=8.5)
            ax1.legend(loc="upper right", facecolor="#0f172a", edgecolor="#334155", labelcolor="#f8fafc", fontsize=7.2)

            w2d_idx = min(10 + k_i, len(wt_frames) - 1)
            ax2.imshow(wt_frames[w2d_idx], cmap="magma", origin="lower", vmin=0.0, vmax=4.2)
            ax2.set_title(f"2D Wild-Type Lattice ($N={wt['n_placodes']}$ Placodes)", color="#f8fafc", fontsize=9.2)
            ax2.axis("off")
            f_fig.tight_layout()
            f_fig.canvas.draw()
            rgba = np.asarray(f_fig.canvas.buffer_rgba())
            frames_pil.append(Image.fromarray(rgba[:, :, :3]))
            plt.close(f_fig)

        # Phase 3 (Pass 3): 3D dermal condensation beneath the placodes and hair peg downgrowth
        d_indices = np.linspace(0, len(dwt["xz_frames"]) - 1, 15, dtype=int)
        vmax_xz = max(2.0, float(dwt["xz_frames"][-1].max()))
        for k_i, idx in enumerate(d_indices):
            f_fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.8, 3.8), dpi=90)
            f_fig.patch.set_facecolor("#0b0f17")
            for a_sub in (ax1, ax2):
                a_sub.set_facecolor("#131b29")
                a_sub.tick_params(colors="#cbd5e1", labelsize=8)
                for sp in a_sub.spines.values():
                    sp.set_color("#334155")
            ax1.imshow(dwt["xz_frames"][idx], cmap="viridis", origin="upper", aspect="auto", extent=[0, Nd, Zd - 0.5, -0.5], vmin=0.0, vmax=vmax_xz)
            h_idx = min(max(0, idx - 1), len(dwt["peg_history"]) - 1)
            peg_now = dwt["peg_history"][h_idx] if idx > 0 else np.zeros(dwt["n_placodes"])
            for i in row_pl:
                cx = wt["centroids_yx"][i][1]
                ax1.plot([cx, cx], [-0.5, peg_now[i]], color="#f43f5e", lw=3.0, solid_capstyle="butt")
                ax1.scatter([cx], [-0.5], marker="v", color="#38bdf8", s=40, zorder=6)
            ax1.axhline(-0.5, color="#38bdf8", lw=1.2)
            ax1.set_title(f"Phase 3: Dermal condensation + peg downgrowth (t={dwt['xz_t'][idx]:.0f})", color="#f8fafc", fontsize=9.0)
            ax1.set_xlabel("Epidermal $x$ (cells)", color="#cbd5e1", fontsize=8.5)
            ax1.set_ylabel("Dermal depth $z$", color="#cbd5e1", fontsize=8.5)
            ax2.imshow(dwt["top_frames"][idx], cmap="viridis", origin="lower", vmin=0.0, vmax=vmax_xz)
            ax2.scatter(wt["centroids_yx"][:, 1], wt["centroids_yx"][:, 0], facecolors="none", edgecolors="#f43f5e", s=60, lw=1.1)
            ax2.set_title(f"Upper-dermal $\\rho$ beneath {wt['n_placodes']} placodes ($N_{{DC}}$ final = {dwt['n_dc']})", color="#f8fafc", fontsize=9.0)
            ax2.axis("off")
            f_fig.tight_layout()
            f_fig.canvas.draw()
            rgba = np.asarray(f_fig.canvas.buffer_rgba())
            frames_pil.append(Image.fromarray(rgba[:, :, :3]))
            plt.close(f_fig)

        frames_pil[0].save(
            OUT_GIF,
            save_all=True,
            append_images=frames_pil[1:],
            duration=85,
            loop=0,
        )

    return {
        "stability": stab,
        "waves_1d": waves_1d,
        "domain_sweep": dom_sweep,
        "phenotypes_2d": pheno_2d,
        "specimen": spec_seg,
        "dermal": dermal,
    }


# --------------------------------------------------------------------------- #
# 6. MorphoEval Sidecar Quantitative Measurements (`measure_morphoeval()`)
# --------------------------------------------------------------------------- #
def measure_morphoeval() -> list[dict[str, Any]]:
    """Compute 13 quantitative morphometric measurements for `morphoeval/measure_outputs.py`
    (7 epidermal WNT/DKK measurements from Pass 2 + 6 dermal condensate / hair peg measurements added in Pass 3)."""
    stab = compute_linear_stability()
    waves_1d = simulate_primary_to_secondary_wave_insertion(seed=2006)
    dom_sweep = simulate_domain_scaling_sweep_1d(seed=2006)
    pheno_2d = simulate_transgenic_phenotypes_2d(seed=2006)
    spec_seg = segment_follicle_specimen(SPECIMEN_PATH)
    dermal = simulate_dermal_condensate_genotypes(pheno_2d["wt"], seed=2006)

    wt = pheno_2d["wt"]
    wnt_o = pheno_2d["wnt_over"]
    enl = pheno_2d["enlarged"]
    dkk_b = pheno_2d["dkk_block"]
    eq_ctl = pheno_2d["equal_diff"]
    dwt = dermal["wt"]
    dko = dermal["fgf20_ko"]
    dbmp = dermal["bmp4_excess"]
    dchi0 = dermal["chemo_null"]
    dlin = dermal["linear"]

    return [
        {
            "name": "Dermal condensate per placode ratio N_DC / N_placode (WT, 3D FGF20 chemotaxis)",
            "value": round(dwt["dc_placode_ratio"], 4),
            "unit": "N_DC / N_placode",
            "method": "3D Keller-Segel FGF20/BMP4 chemotaxis slab (42x42x12, periodic x,y / zero-flux z) beneath the WT 2D WNT placode field; DCs = connected components of upper-dermal rho > e_thr",
            "reference_value": 1.0,
            "reference_source": "Huh et al. (2013) Genes Dev 27:450-458 (one primary dermal condensate beneath every primary placode); Glover et al. (2017) PLoS Biol 15:e2002117",
            "note": f"WT: {dwt['n_dc']} DCs beneath {dwt['n_placodes']} placodes (fraction of placodes with a DC = {dwt['frac_placodes_with_dc']:.2f}); peak upper-dermal enrichment rho_max/rho_0 = {dwt['peak_enrichment']:.2f}; DC excess-density centre of mass {dwt['dc_depth_com']:.2f} layers below the epidermis",
            "derived_from": "script",
        },
        {
            "name": "Fgf20-null + chemotaxis-null dermal condensate count (placodes persist, no DC)",
            "value": int(dko["n_dc"] + dchi0["n_dc"]),
            "unit": "dermal condensates",
            "method": "Same 3D slab with fgf20_scale = 0 (Fgf20-/-) and with chi_f = 0 (FGF20 present but unread); epidermal placode field unchanged (15 placodes)",
            "reference_value": 0,
            "reference_source": "Huh et al. (2013) Genes Dev 27:450-458 Figs. 1-3 (Fgf20-/- skin forms placodes but no primary or secondary dermal condensates) + Lean 4 dermal_subsystem_instructed_stable",
            "note": f"Fgf20-/-: {dko['n_dc']} DCs, peak enrichment {dko['peak_enrichment']:.3f}, peg depth {dko['mean_peg_depth_frac']:.3f}; chi_f=0: {dchi0['n_dc']} DCs; epidermis-free dermal subsystem min det(q>0) = {dlin['min_det_instructed']:.3e} >= 0 (no spontaneous condensation)",
            "derived_from": "script",
        },
        {
            "name": "Mean lateral DC-placode centroid offset (WT, periodic nearest neighbour)",
            "value": round(dwt["mean_dc_offset_px"], 4),
            "unit": "cells (px)",
            "method": "Periodic nearest-neighbour distance between each WT epidermal placode centroid and the upper-dermal DC centroid beneath it",
            "reference_value": 0.0,
            "reference_source": "Glover et al. (2017) PLoS Biol 15:e2002117 (dermal cells aggregate directly beneath the placode by migration toward placode FGF20)",
            "note": f"mean offset {dwt['mean_dc_offset_px']:.3f} px, max {dwt['max_dc_offset_px']:.3f} px over {int(np.sum(dwt['placode_has_dc']))} registered placode/DC pairs (WT 2D placode NN spacing {wt['mean_nn_px']:.2f} px)",
            "derived_from": "script",
        },
        {
            "name": "Hair peg downgrowth: Fgf20-null / WT mean peg-depth ratio (arrest at placode stage)",
            "value": round(dko["mean_peg_depth_frac"] / max(1e-12, dwt["mean_peg_depth_frac"]), 4),
            "unit": "ratio",
            "method": "Per-placode peg depth ODE dz_i/dt = v_peg clip((e_i - e_thr)/(rho_max - e_thr), 0, 1) slaved to the upper-dermal density e_i beneath placode i; depth normalised by dermal slab depth (Z-1)",
            "reference_value": 0.0,
            "reference_source": "Huh et al. (2013) Genes Dev 27:450-458 (Fgf20-/- follicles arrest at the placode stage); Paus et al. (1999) J Invest Dermatol 113:523-532 (ordinal stages: placode -> hair germ -> hair peg)",
            "note": f"WT mean peg depth = {dwt['mean_peg_depth_frac']*100:.1f}% of dermis (stage histogram 1..4 = {np.bincount(dwt['stage'], minlength=5)[1:].tolist()}, model stage convention); Fgf20-/- = {dko['mean_peg_depth_frac']*100:.1f}%; BMP4 excess (5x s_b) = {dbmp['mean_peg_depth_frac']*100:.1f}% with {dbmp['n_dc']} DCs (peak enrichment {dbmp['peak_enrichment']:.2f})",
            "derived_from": "script",
        },
        {
            "name": "FGF20 chemotactic-sensitivity sweep correlation r(chi_f, mean peg depth)",
            "value": round(dermal["chi_sweep_pearson_r"], 4),
            "unit": "Pearson r",
            "method": "5-point sweep chi_f in {0, 0.1, 0.15, 0.2, 0.25}; Pearson r between chi_f and mean normalised peg depth; monotonicity checked separately",
            "reference_value": 1.0,
            "reference_source": "Keller & Segel (1970) J Theor Biol 26:399-415 (chemotactic sensitivity controls aggregation strength); Glover et al. (2017) PLoS Biol 15:e2002117",
            "note": f"DC counts {[s['n_dc'] for s in dermal['chi_sweep']]} and peg depths {[round(s['mean_peg_depth_frac'], 3) for s in dermal['chi_sweep']]} across chi_f = {[s['chi_f'] for s in dermal['chi_sweep']]}; monotone = {dermal['chi_sweep_monotone']}",
            "derived_from": "script",
        },
        {
            "name": "Discrete mesenchymal mass conservation of the chemotaxis scheme (WT relative drift)",
            "value": float(f"{abs(dwt['mass_drift']):.3e}"),
            "unit": "relative",
            "method": "|sum(rho_end) - sum(rho_0)| / sum(rho_0) over 3200 explicit steps of the conservative face-flux chemotaxis + diffusion scheme (periodic x,y, zero-flux z)",
            "reference_value": 0.0,
            "reference_source": "Keller & Segel (1970) J Theor Biol 26:399-415 conservation form; Painter & Hillen (2002) Can Appl Math Q 10:501-543 volume filling",
            "note": f"WT drift {dwt['mass_drift']:.3e}, Fgf20-/- drift {dko['mass_drift']:.3e}, BMP4-excess drift {dbmp['mass_drift']:.3e} (non-zero drift would indicate rho hitting the clip bounds [0, rho_max])",
            "derived_from": "script",
        },
        {
            "name": "WNT/DKK critical diffusivity ratio d_c = (sqrt(2)+1)^2",
            "value": round(stab["d_c"], 4),
            "unit": "D_h / D_w",
            "method": "Exact analytical Turing-Meinhardt threshold (3 + 2*sqrt(2)) at mu_h > mu_w",
            "reference_value": 5.8284,
            "reference_source": "Sick et al. (2006) Science 314:1447-1450 & Gierer & Meinhardt (1972)",
            "note": f"canonical d = {stab['d_ratio']:.1f} > d_c (max Re sigma = +{stab['max_sigma_super']:.5f}); equal-diffusion control d = 1.0 has max Re sigma = {stab['max_sigma_equal']:.5f}",
            "derived_from": "script",
        },
        {
            "name": "Unsaturated steady-state WNT activator concentration w_*",
            "value": round(stab["w_star"], 4),
            "unit": "concentration units",
            "method": "Exact steady-state formula w_* = (c_w / c_h) * (mu_h / mu_w) with h_* = 2.25",
            "reference_value": 1.5,
            "reference_source": "Sick et al. (2006) Science 314:1447-1450",
            "note": f"homogeneous Jacobian tr(J_0) = {stab['trace_J0']:.4f} < 0, det(J_0) = {stab['det_J0']:.5f} > 0",
            "derived_from": "script",
        },
        {
            "name": "Secondary inter-follicular wave placode doubling ratio upon 2x epidermal expansion",
            "value": round(waves_1d["doubling_ratio"], 4),
            "unit": "N_secondary / N_primary",
            "method": "1D periodic WNT/DKK PDE expanded 2x from N_1=40 (E14.5 primary wave) to N_2=80 (E16.5 secondary inter-follicular insertion)",
            "reference_value": 2.0,
            "reference_source": "Sick et al. (2006) Science 314:1447-1450 Figs. 3-4 (inter-follicular secondary wave insertion)",
            "note": f"primary wave forms {waves_1d['primary']['n_peaks']} peaks (CV={waves_1d['primary']['cv_spacing']:.4f}) -> 2x expanded epidermis inserts 4 midpoint secondary placodes to reach {waves_1d['secondary']['n_peaks']} peaks (spacing {waves_1d['secondary']['mean_spacing']:.2f} ± {waves_1d['secondary']['std_spacing']:.2f} cells, CV={waves_1d['secondary']['cv_spacing']:.4f})",
            "derived_from": "script",
        },
        {
            "name": "Wild-type 2D hair follicle placode count vs Mus musculus in situ specimen (Panel B)",
            "value": int(wt["n_placodes"]),
            "unit": "placodes",
            "method": "Connected components of 2D WNT activator field on 42x42 grid vs optical-density segmentation of Mus musculus E12.5-E13.5 specimen Panel B",
            "reference_value": int(spec_seg["n_placodes_segmented"]),
            "reference_source": "Sick et al. (2006) Science 314:1447-1450 Fig. 2A & Maddox & Condie (2004) BMC Dev. Biol. 4:4 (16 placodes)",
            "note": f"WT 2D PDE forms {wt['n_placodes']} placodes (NN spacing = {wt['mean_nn_px']:.2f} ± {wt['std_nn_px']:.2f} px, CV = {wt['cv_nn']:.4f}, area = {wt['mean_area_px']:.2f} px^2); specimen Panel B segmented {spec_seg['n_placodes_segmented']} placodes (NN spacing = {spec_seg['mean_nn_px']:.2f} ± {spec_seg['std_nn_px']:.2f} px, CV = {spec_seg['cv_nn']:.4f})",
            "derived_from": "script",
        },
        {
            "name": "Transgenic K14-Wnt overexpression vs Wild-Type placode density ratio",
            "value": round(float(wnt_o["n_placodes"]) / float(max(1, wt["n_placodes"])), 4),
            "unit": "N_Wnt++ / N_WT",
            "method": "Ratio of 2D placode count (periodic local WNT maxima) under elevated K14-Wnt expression (rho_w=0.004) to Wild-Type (rho_w=0.001)",
            "reference_value": None,
            "reference_source": "Sick et al. (2006) Science 314:1447-1450 Fig. 2B-D (K14-Wnt transgenic hyper-density & clustering; qualitative 'increased density' only - no numeric ratio is reported, so no reference_value is claimed)",
            "note": f"K14-Wnt++ raises placode count from {wt['n_placodes']} to {wnt_o['n_placodes']} ({wnt_o['n_placodes'] / max(1, wt['n_placodes']):.2f}x) and reduces mean nearest-neighbor spacing from {wt['mean_nn_px']:.2f} px to {wnt_o['mean_nn_px']:.2f} px; saturated regime increases mean placode component area from {wt['mean_area_px']:.2f} to {enl['mean_area_px']:.2f} px^2 ({(enl['mean_area_px'] / max(1e-9, wt['mean_area_px']) - 1.0) * 100:+.1f}%); the Pass 2 reference ratio 2.0 is a qualitative 'increased density' benchmark, not a value reported by Sick et al.",
            "derived_from": "script",
        },
        {
            "name": "Transgenic Dkk1 overexpression blockade & equal-diffusion negative control placode count",
            "value": int(dkk_b["n_placodes"] + eq_ctl["n_placodes"]),
            "unit": "placodes",
            "method": "2D WNT/DKK PDE under ectopic Dkk1 overexpression (rho_h=0.12) and equal-diffusion control (D_w=D_h=0.015)",
            "reference_value": 0,
            "reference_source": "Andl et al. (2002) Dev. Cell 2:643-653 & Sick et al. (2006) Fig. 4 (Dkk1 alopecia blockade) + Lean 4 wnt_dkk_equal_diffusion_stable",
            "note": f"Dkk1++ blockade clamps w_max = {dkk_b['w_max']:.4f} (0 placodes, std = {dkk_b['w_std']:.2e}); equal-diffusion control forms 0 placodes (std = {eq_ctl['w_std']:.2e}) vs {wt['n_placodes']} WT placodes",
            "derived_from": "script",
        },
        {
            "name": "Allometric epidermal domain-length scaling correlation r(L, N_peaks)",
            "value": round(dom_sweep["pearson_r"], 4),
            "unit": "Pearson r",
            "method": "Linear correlation between epidermal transect length L in [20..80] cells and placode count N_peaks(L)",
            "reference_value": 1.0,
            "reference_source": "Sick et al. (2006) Science 314:1447-1450 intrinsic reaction-diffusion wavelength invariance",
            "note": f"placode counts {dom_sweep['peak_counts']} across L={dom_sweep['domain_lengths']}; mean nonlinear wavelength = {dom_sweep['mean_nonlinear_wavelength']:.2f} cells, RMSE = {dom_sweep['rmse_peaks']:.4f} peaks",
            "derived_from": "script",
        },
    ]


# --------------------------------------------------------------------------- #
# 7. Comprehensive 12-Check Deterministic Self-Test Suite (`--self-test`)
# --------------------------------------------------------------------------- #
def self_test() -> int:
    """Run 12 deterministic verification checks covering SHA-256 locks, theory,
    primary->secondary wave insertion, 2D transgenic phenotypes, specimen segmentation,
    bitwise two-run reproducibility, and (Pass 3) the 3D dermal condensate FGF20/BMP4
    chemotaxis + hair peg downgrowth coupling (checks 9-12)."""
    # 1. Verify SHA-256 digests of all vendored dataset files
    digests = verify_dataset_sha256()
    ref = load_reference()
    assert len(digests) == 4, "Expected 4 verified dataset files"

    # 2. Verify JSON-to-code parameter and algebraic parity
    canon = ref["canonical_parameters"]
    stab = compute_linear_stability(
        mu_w=canon["mu_w"],
        mu_h=canon["mu_h"],
        D_w=canon["D_w"],
        D_h=canon["D_h"],
        c_w=canon["c_w"],
        c_h=canon["c_h"],
    )
    assert abs(stab["d_c"] - canon["critical_diffusivity_ratio_dc"]) < 1e-12, "d_c parity failure"
    assert abs(stab["w_star"] - canon["unsaturated_w_star"]) < 1e-12, "w_* parity failure"
    assert abs(stab["h_star"] - canon["unsaturated_h_star"]) < 1e-12, "h_* parity failure"
    assert stab["trace_J0"] < 0.0 and stab["det_J0"] > 0.0, "Homogeneous Jacobian must be stable"
    assert stab["max_sigma_super"] > 0.0 and stab["max_sigma_equal"] < 0.0, "Dispersion sign check failed"

    # 3. Verify 1D primary wave (4 peaks) -> 2x domain expansion -> secondary wave insertion (8 peaks)
    waves_1d = simulate_primary_to_secondary_wave_insertion(seed=2006)
    assert waves_1d["primary"]["n_peaks"] == 4, f"Expected 4 primary wave peaks, got {waves_1d['primary']['n_peaks']}"
    assert waves_1d["secondary"]["n_peaks"] == 8, f"Expected 8 secondary wave peaks after 2x expansion, got {waves_1d['secondary']['n_peaks']}"
    assert abs(waves_1d["doubling_ratio"] - 2.0) < 1e-9, "Expected exact 2.0x placode doubling"
    assert waves_1d["secondary"]["cv_spacing"] < 0.08, f"Expected regular secondary spacing CV < 0.08, got {waves_1d['secondary']['cv_spacing']:.4f}"

    # 4. Verify 2D Wild-Type, K14-Wnt overexpression, Saturated enlarged, Dkk1 blockade & Equal-diffusion control
    pheno_2d = simulate_transgenic_phenotypes_2d(seed=2006)
    assert pheno_2d["wt"]["n_placodes"] == ref["experimental_phenotypes"]["wild_type_primary_wave"]["expected_placode_count"] == 15, f"Expected 15 WT 2D placodes (periodic local maxima), got {pheno_2d['wt']['n_placodes']}"
    assert pheno_2d["wt"]["n_components"] == 14, f"Expected 14 seam-aware WT threshold components, got {pheno_2d['wt']['n_components']}"
    assert pheno_2d["wnt_over"]["n_placodes"] == ref["experimental_phenotypes"]["k14_wnt_overexpression"]["expected_placode_count"], "K14-Wnt++ placode count parity with JSON failed"
    assert pheno_2d["enlarged"]["n_placodes"] == ref["experimental_phenotypes"]["enlarged_ringing_follicles_saturated"]["expected_placode_count"], "Enlarged placode count parity with JSON failed"
    assert pheno_2d["wnt_over"]["n_placodes"] > pheno_2d["wt"]["n_placodes"], "K14-Wnt++ must increase placode count over WT"
    assert pheno_2d["wnt_over"]["mean_nn_px"] < pheno_2d["wt"]["mean_nn_px"], "K14-Wnt++ must reduce inter-placode spacing"
    assert pheno_2d["enlarged"]["mean_area_px"] > pheno_2d["wt"]["mean_area_px"] * 1.30, "Saturated regime must enlarge placode area by >30%"
    assert pheno_2d["dkk_block"]["n_placodes"] == 0, "Dkk1++ blockade must abolish all placode initiation (0 placodes)"
    assert pheno_2d["equal_diff"]["n_placodes"] == 0 and pheno_2d["equal_diff"]["w_std"] < 1e-4, "Equal-diffusion control must decay to flat state"

    # 5. Verify ectopic Dkk1 dosage sweep monotonic suppression to 0
    assert pheno_2d["dkk_sweep"][0]["n_placodes"] == 15 and pheno_2d["dkk_sweep"][-1]["n_placodes"] == 0
    sweep_counts = [r["n_placodes"] for r in pheno_2d["dkk_sweep"]]
    assert sweep_counts == sorted(sweep_counts, reverse=True), f"Dkk1 dosage sweep must be monotone non-increasing: {sweep_counts}"

    # 6. Verify allometric domain-length scaling
    dom_sweep = simulate_domain_scaling_sweep_1d(seed=2006)
    assert dom_sweep["pearson_r"] >= 0.98, f"Expected domain scaling r >= 0.98, got {dom_sweep['pearson_r']:.4f}"

    # 7. Verify Mus musculus specimen segmentation (16 placodes in Panel B)
    spec_seg = segment_follicle_specimen(SPECIMEN_PATH)
    assert spec_seg["n_placodes_segmented"] == 16, f"Expected 16 segmented placodes in specimen Panel B, got {spec_seg['n_placodes_segmented']}"
    assert 0.15 < spec_seg["cv_nn"] < 0.45, f"Unexpected specimen nearest-neighbor CV: {spec_seg['cv_nn']:.4f}"

    # 8. Verify bitwise two-run determinism (2D epidermis and 3D dermis)
    run1 = simulate_wnt_dkk_2d(N=24, steps=400, dt=0.25, seed=2006)
    run2 = simulate_wnt_dkk_2d(N=24, steps=400, dt=0.25, seed=2006)
    assert np.allclose(run1["w"], run2["w"], atol=0.0, rtol=0.0), "Bitwise determinism check failed"
    d1 = simulate_dermal_condensate_3d(run1["w"], run1["centroids_yx"], Z=6, steps=200, seed=2006, record_every=200)
    d2 = simulate_dermal_condensate_3d(run2["w"], run2["centroids_yx"], Z=6, steps=200, seed=2006, record_every=200)
    assert np.array_equal(d1["rho"], d2["rho"]) and np.array_equal(d1["peg_depth"], d2["peg_depth"]), "3D dermal determinism check failed"

    # 9. (Pass 3) 3D dermal condensate: one DC beneath every WT placode, laterally registered
    dermal = simulate_dermal_condensate_genotypes(pheno_2d["wt"], seed=2006)
    dwt = dermal["wt"]
    dref = ref["dermal_condensate_model"]
    assert dwt["n_dc"] == pheno_2d["wt"]["n_placodes"] == dref["expected_wt_dc_count"], f"Expected {dref['expected_wt_dc_count']} WT DCs, got {dwt['n_dc']}"
    assert dwt["frac_placodes_with_dc"] == 1.0, "Every WT placode must acquire a dermal condensate"
    assert dwt["mean_dc_offset_px"] < 1.0 and dwt["max_dc_offset_px"] < 1.5, f"DC must sit beneath its placode (mean offset {dwt['mean_dc_offset_px']:.3f} px)"
    assert dwt["dc_depth_com"] < 1.5, "DC excess density must be concentrated in the upper dermis"
    for k_par in ("chi_f", "chi_b", "D_f", "mu_f", "s_f", "D_b", "mu_b", "s_b", "D_rho", "rho_max", "e_thr", "v_peg"):
        assert abs(DERMAL_DEFAULTS[k_par] - dref["parameters"][k_par]) < 1e-12, f"dermal parameter parity failure for {k_par}"

    # 10. (Pass 3) Fgf20-null, chemotaxis-null and BMP4-excess genotypes
    assert dermal["fgf20_ko"]["n_dc"] == 0 and dermal["fgf20_ko"]["mean_peg_depth_frac"] == 0.0, "Fgf20-/- must form no DC and no peg (Huh et al. 2013)"
    assert dermal["chemo_null"]["n_dc"] == 0, "chi_f = 0 must form no DC"
    assert dermal["bmp4_excess"]["n_dc"] < dwt["n_dc"] and dermal["bmp4_excess"]["peak_enrichment"] < dwt["peak_enrichment"], "BMP4 excess must suppress dermal condensation"
    assert dermal["bmp4_excess"]["mean_peg_depth_frac"] < dwt["mean_peg_depth_frac"], "BMP4 excess must suppress peg downgrowth"
    assert int(np.min(dwt["stage"])) >= 2 and int(np.max(dwt["stage"])) == 4, "WT follicles must reach hair germ/peg stages"
    assert np.all(dermal["fgf20_ko"]["stage"] == 1), "Fgf20-/- follicles must arrest at the placode stage"

    # 11. (Pass 3) Mass conservation of the conservative chemotaxis scheme + peg-depth monotonicity in time
    for key in ("wt", "fgf20_ko", "bmp4_excess", "chemo_null"):
        assert abs(dermal[key]["mass_drift"]) < 1e-9, f"Mesenchymal mass drift too large in {key}: {dermal[key]['mass_drift']:.3e}"
    assert np.all(np.diff(dwt["peg_history"], axis=0) >= -1e-12), "Peg depth must be non-decreasing in time (Lean hair_peg_depth_monotone)"

    # 12. (Pass 3) Dermal linear stability & chi_f sweep monotonicity
    lin = dermal["linear"]
    assert lin["min_det_instructed"] >= 0.0 and lin["min_neg_trace"] > 0.0, "Epidermis-free dermal subsystem must be linearly stable (instructed condensation)"
    assert dermal["chi_sweep_monotone"], "Mean peg depth must be monotone non-decreasing in chi_f"
    assert dermal["chi_sweep_pearson_r"] >= 0.85, f"Expected chi_f sweep r >= 0.85, got {dermal['chi_sweep_pearson_r']:.4f}"
    sweep_ndc = [s["n_dc"] for s in dermal["chi_sweep"]]
    assert sweep_ndc == sorted(sweep_ndc), f"DC count must be monotone in chi_f: {sweep_ndc}"

    print(
        "[2002-sick-wnt-dkk-hair-follicle] SELF-TEST PASSED (12/12 checks: SHA-256 dataset locks, "
        f"d_c={stab['d_c']:.4f}, primary->secondary wave={waves_1d['primary']['n_peaks']}->{waves_1d['secondary']['n_peaks']} peaks, "
        f"2D WT={pheno_2d['wt']['n_placodes']} vs specimen={spec_seg['n_placodes_segmented']} placodes, "
        f"K14-Wnt++={pheno_2d['wnt_over']['n_placodes']}, enlarged area={pheno_2d['enlarged']['mean_area_px']:.1f}px^2, "
        f"Dkk1++={pheno_2d['dkk_block']['n_placodes']}, domain scaling r={dom_sweep['pearson_r']:.4f}; "
        f"3D dermal: WT DCs={dwt['n_dc']}/{dwt['n_placodes']} (offset {dwt['mean_dc_offset_px']:.2f}px, peg depth {dwt['mean_peg_depth_frac']*100:.0f}%), "
        f"Fgf20-/- DCs={dermal['fgf20_ko']['n_dc']}, BMP4++ DCs={dermal['bmp4_excess']['n_dc']}, mass drift={dwt['mass_drift']:.1e}, chi_f sweep r={dermal['chi_sweep_pearson_r']:.3f})."
    )
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true", help="Run 12-check deterministic verification suite and exit")
    ap.add_argument("--skip-gif", action="store_true", help="Skip rendering animated GIF")
    ap.add_argument("--cw", type=float, default=0.05, help="Autocatalytic WNT production rate c_w")
    ap.add_argument("--ch", type=float, default=0.05, help="DKK production rate c_h")
    ap.add_argument("--mu-w", type=float, default=0.02, help="WNT turnover rate mu_w")
    ap.add_argument("--mu-h", type=float, default=0.03, help="DKK turnover rate mu_h")
    ap.add_argument("--Dw", type=float, default=0.015, help="WNT diffusivity D_w")
    ap.add_argument("--Dh", type=float, default=0.45, help="DKK diffusivity D_h")
    ap.add_argument("--seed", type=int, default=2006, help="Deterministic RNG seed")
    ap.add_argument("--chi-f", type=float, default=DERMAL_DEFAULTS["chi_f"], help="Dermal FGF20 chemotactic sensitivity chi_f (Pass 3 3D dermal condensate model)")
    args = ap.parse_args()

    if args.self_test:
        return self_test()

    res = render_outputs(
        c_w=args.cw,
        c_h=args.ch,
        mu_w=args.mu_w,
        mu_h=args.mu_h,
        D_w=args.Dw,
        D_h=args.Dh,
        seed=args.seed,
        skip_gif=args.skip_gif,
        chi_f=args.chi_f,
    )
    dwt = res["dermal"]["wt"]
    print(
        f"[2002-sick-wnt-dkk-hair-follicle] Rendered 12-panel comparison -> {OUT_PNG.relative_to(ROOT)} "
        f"and animation -> {OUT_GIF.relative_to(ROOT)} "
        f"(primary->secondary={res['waves_1d']['primary']['n_peaks']}->{res['waves_1d']['secondary']['n_peaks']}, "
        f"WT 2D={res['phenotypes_2d']['wt']['n_placodes']} vs specimen={res['specimen']['n_placodes_segmented']}, "
        f"K14-Wnt++={res['phenotypes_2d']['wnt_over']['n_placodes']}, Dkk1++={res['phenotypes_2d']['dkk_block']['n_placodes']}; "
        f"3D dermal DCs={dwt['n_dc']}/{dwt['n_placodes']}, peg depth={dwt['mean_peg_depth_frac']*100:.0f}%, "
        f"Fgf20-/- DCs={res['dermal']['fgf20_ko']['n_dc']})."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

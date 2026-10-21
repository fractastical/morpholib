#!/usr/bin/env python3
"""
1976-cooke-clock-and-wavefront.py — Clock-and-Wavefront Somite Segmentation Model
==================================================================================

Faithful mathematical, numerical, and morphometric reproduction of:
  1. Cooke, J. & Zeeman, E. C. (1976). "A clock and wavefront model for control
     of the number of repeated structures during animal morphogenesis."
     Journal of Theoretical Biology 58(2):455-476.
     DOI: 10.1016/S0022-5193(76)80131-2
  2. Cooke, J. (1975). "Control of somite number during morphogenesis of a
     vertebrate, Xenopus laevis." Nature 254(5497):196-199.
     DOI: 10.1038/254196a0
  3. Palmeirim, I., Henrique, D., Ish-Horowicz, D. & Pourquie, O. (1997).
     "Avian hairy gene expression identifies a molecular clock linked to
     vertebrate segmentation and somitogenesis." Cell 91(5):639-648.
     DOI: 10.1016/S0092-8674(00)80451-1
  4. Goldbeter, A., Gonze, D. & Pourquie, O. (2007). "Sharp developmental
     thresholds defined through bistability by antagonistic gradients of
     retinoic acid and FGF signaling." Developmental Dynamics 236(6):1495-1508.
     DOI: 10.1002/dvdy.21193
  5. Dubrulle, J. & Pourquie, O. (2004). "fgf8 mRNA decay establishes a gradient
     that couples axial elongation to patterning in the vertebrate embryo."
     Nature 427(6973):419-422.  DOI: 10.1038/nature02216
  6. Dubrulle, J., McGrew, M. J. & Pourquie, O. (2001). "FGF signaling controls
     somite boundary position and regulates segmentation clock control of
     spatiotemporal Hox gene activation." Cell 106(2):219-232.
     DOI: 10.1016/S0092-8674(01)00437-8
  7. Farhadifar, R., Roper, J.-C., Aigouy, B., Eaton, S. & Julicher, F. (2007).
     "The influence of cell mechanics, cell-cell interactions, and proliferation
     on epithelial packing." Current Biology 17(24):2095-2104.
     DOI: 10.1016/j.cub.2007.11.049
  8. Bi, D., Lopez, J. H., Schwarz, J. M. & Manning, M. L. (2015). "A density-
     independent rigidity transition in biological tissues." Nature Physics
     11:1074-1079.  DOI: 10.1038/nphys3471
  9. Saga, Y. & Takeda, H. (2001). "The making of the somite: molecular events
     in vertebrate segmentation." Nature Reviews Genetics 2(11):835-845.
     DOI: 10.1038/35098552

Governing Mathematical Laws (verified in `morphoeval/lean/loops/Loop_021_clock_and_wavefront.lean`):
---------------------------------------------------------------------------------------------------
1. Cooke & Zeeman (1976) Somite Wavelength Law (`cooke_zeeman_1976_wavelength_period_scaling`):
       S = v * T_c
   where S is anteroposterior somite length, v is the velocity of the anteriorly-to-
   posteriorly regressing determination wavefront x_w(t) = v * t, and T_c is the
   period of the autonomous intracellular segmentation oscillator.

2. Cooke (1975) & Cooke-Zeeman (1976) Embryo Size-Invariance Law
   (`cooke_1975_miniature_embryo_size_invariance`):
       N = L / S = L / (v * T_c),   with   v = L / T_total  =>  N = T_total / T_c
   Surgically reduced miniature Xenopus laevis embryos (L' = alpha * L_0, alpha = 0.60)
   scale their positional wavefront gradient and regression speed proportionally
   (v' = alpha * v_0), yielding scaled somites S' = alpha * S_0 while preserving an
   invariant total somite number N ~ 40.

3. Thom-Zeeman Cusp Catastrophe Wavefront Dynamics (`zeeman_1976_cusp_wavefront_bifurcation`):
       tau_chi * dchi(x,t)/dt = -(chi^3 - a(x,t) * chi - b_osc(x,t))
       a(x,t) = tanh((v * t - x) / sigma_w)                        [kinematic wavefront]
       b_osc(x,t) = b_0 + A_c * cos(2 * pi * t / T_c - k_phi * x)
   Prior to wavefront arrival (x > v*t, a < 0), cells reside near the monostable
   presomitic mesoderm branch chi ~ 0. As the wavefront x_w(t) = v*t sweeps past x
   (a -> +1), the cusp manifold bifurcates into upper (chi -> +1, somite core) and
   lower (chi -> -1, inter-somitic boundary cleft) attractors, gated periodically by
   the intracellular segmentation clock phase phi(x,t).

4. Goldbeter-Gonze-Pourquie (2007) FGF8/RA Bistable Determination Front
   (`goldbeter_2007_front_velocity_equals_tailbud_elongation`), Pass 3 remediation:
       M_F(x,t) = exp(-(x_tb(t) - x) / lambda_F),   x_tb(t) = x_tb0 + v_tb * t,
       lambda_F = v_tb / k_dM                       [Dubrulle & Pourquie 2004 mRNA decay gradient]
       tau_FR * dF/dt = alpha_F * M_F * K^n / (K^n + R^n) - F
       tau_FR * dR/dt = alpha_R       * K^n / (K^n + F^n) - R
       a(x,t) = tanh((R - F) / kappa_a),   m(x,t) = F^4 / (F^4 + K_mob^4)
   The mutually repressive FGF8 (F) / retinoic-acid (R) toggle is bistable for
   M_F in [M_lo, M_hi]; cells leaving the tailbud switch irreversibly to the RA-high
   state when their fgf8 mRNA decays below M_lo, so the determination front sits at
       x_w(t) = x_tb(t) - lambda_F * ln(1 / M_lo)
   and regresses at exactly the tailbud elongation speed, dx_w/dt = v_tb. The
   cusp unfolding parameter a and the commitment mobility m are read off the toggle
   state, so the wavefront velocity v in Law 1 is no longer prescribed but emerges
   from axis elongation. (Reduced two-variable form of Goldbeter et al. 2007; the
   Cyp26 / Raldh2 intermediates are lumped into the two Hill repression terms.)

5. Meinhardt / Saga-Takeda Anterior-Posterior Compartment Cleft Rule & Farhadifar
   Vertex-Model Paraxial Mesoderm Sheet (`meinhardt_1986_half_somite_cleft_count`),
   Pass 3 remediation:
       E = sum_cells [ K_A/2 (A_c - A_0)^2 + Gamma/2 P_c^2 ] + sum_edges Lambda_e l_e
       Lambda_e = Lambda_0 + dLambda * [chi(anterior cell) < 0 < chi(posterior cell)]
   The alternating chi = +1 / -1 compartments of Law 3 are interpreted as anterior
   (+1) and posterior (-1) half-somites; an inter-somitic cleft forms only at a
   posterior(-1) -> anterior(+1) interface, so 2N half-somite compartments yield N
   somites and N-1 internal clefts. Differential line tension on those interfaces
   in a 2D vertex model of the paraxial mesoderm sheet rotates the initially
   zig-zagging lattice cleft toward the mediolateral axis and straightens it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection, PolyCollection
import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from scipy import signal, stats

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets" / "clock_and_wavefront"
REFERENCE_JSON = DATASET_DIR / "clock_and_wavefront_reference.json"
SPECIMEN_JPG = DATASET_DIR / "somite_embryo_specimen.jpg"
SCOUT_MANIFEST_JSON = DATASET_DIR / "scout_manifest.json"
FGF_RA_REFERENCE_JSON = DATASET_DIR / "fgf_ra_vertex_reference.json"
OUTPUT_PNG = ROOT / "docs" / "media" / "clock_and_wavefront_comparison.png"
OUTPUT_GIF = ROOT / "docs" / "media" / "clock_and_wavefront_growth.gif"
OUTPUT_SHEET_PNG = ROOT / "docs" / "media" / "clock_and_wavefront_vertex_sheet.png"

EXPECTED_SHA256 = {
    "datasets/clock_and_wavefront/somite_embryo_specimen.jpg": (
        "8b80971796cb29e290c10886c0026092bc30f3f624d2a75fa72f9aaf05187d5f"
    ),
    "datasets/clock_and_wavefront/scout_manifest.json": (
        "0beabd3277c705e62b894794a2448ec9572377059c72a185a6148e2f17f0b327"
    ),
}

# Reduced Goldbeter-Gonze-Pourquie (2007) FGF8/RA toggle parameters (Law 4).
# F, R are normalised protein levels; M_F is fgf8 mRNA normalised to its tailbud value.
FGF_RA_PARAMS: dict[str, float] = {
    "alpha_F": 3.0,          # maximal FGF8 synthesis per unit fgf8 mRNA
    "alpha_R": 3.0,          # maximal RA synthesis (Raldh2) when FGF is absent
    "K_RF": 1.0,             # RA threshold repressing fgf8 (Goldbeter 2007 RA -| Fgf8)
    "K_FR": 1.0,             # FGF threshold repressing RA (Raldh2 repression + Cyp26 induction lumped)
    "n_hill": 4.0,           # Hill cooperativity (n >= 2 required for bistability)
    "tau_FR_min": 12.0,      # protein turnover time (min)
    "k_dM_per_min": 0.0074,  # fgf8 mRNA decay rate (Dubrulle & Pourquie 2004 mechanism; lambda_F = v_tb / k_dM)
    "kappa_a": 0.5,          # cusp unfolding sharpness a = tanh((R - F)/kappa_a)
    "K_mob": 0.3,            # FGF level below which cusp commitment mobility is lost
    "n_mob": 4.0,
}

# Farhadifar et al. (2007) vertex-model parameters (case I: Lambda_bar = 0.12, Gamma_bar = 0.04).
VERTEX_PARAMS: dict[str, float] = {
    "K_A": 1.0,
    "Gamma": 0.04,
    "Lambda_0": 0.12,
    "delta_Lambda_cleft": 0.9,   # extra line tension on posterior(-1)->anterior(+1) interfaces
    "step": 0.05,
    "steps_per_stage": 150,
    "cell_pitch_um": 20.0,       # AP pitch of PSM cells => 5 cells per 100-um Xenopus somite
    "n_rows": 6,
    "sheet_length_mm": 1.2,
    "commit_threshold": 0.5,
}


# ============================================================================
# Section 1: Cryptographic Dataset Verification & Reference Loading
# ============================================================================
def verify_dataset_integrity() -> dict[str, str]:
    """Verify SHA-256 digests of vendored specimen and scout manifest files."""
    digests: dict[str, str] = {}
    for rel, expected in EXPECTED_SHA256.items():
        p = ROOT / rel
        if not p.exists():
            raise FileNotFoundError(f"Missing required dataset file: {p}")
        actual = hashlib.sha256(p.read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(
                f"SHA-256 mismatch for {rel}: expected {expected}, got {actual}"
            )
        digests[rel] = actual
    if not REFERENCE_JSON.exists():
        raise FileNotFoundError(f"Missing reference JSON: {REFERENCE_JSON}")
    digests["datasets/clock_and_wavefront/clock_and_wavefront_reference.json"] = (
        hashlib.sha256(REFERENCE_JSON.read_bytes()).hexdigest()
    )
    if FGF_RA_REFERENCE_JSON.exists():
        digests["datasets/clock_and_wavefront/fgf_ra_vertex_reference.json"] = (
            hashlib.sha256(FGF_RA_REFERENCE_JSON.read_bytes()).hexdigest()
        )
    return digests


def load_reference_data() -> dict[str, Any]:
    """Load `datasets/clock_and_wavefront/clock_and_wavefront_reference.json`."""
    return json.loads(REFERENCE_JSON.read_text(encoding="utf-8"))


def load_fgf_ra_reference() -> dict[str, Any]:
    """Load the Pass 3 FGF/RA + vertex-sheet reference table (empty dict if absent)."""
    if FGF_RA_REFERENCE_JSON.exists():
        return json.loads(FGF_RA_REFERENCE_JSON.read_text(encoding="utf-8"))
    return {}


# ============================================================================
# Section 2: Cooke & Zeeman (1976) Analytical Clock-and-Wavefront Kinematics
#            (Cooke & Zeeman 1976 J. Theor. Biol. 58:455-476 Sections 2-3)
# ============================================================================
def analytical_somite_wavelength(v_um_per_min: float, T_c_min: float) -> float:
    """Law 1 (`cooke-clock-and-wavefront-1`):
    Compute somite anteroposterior length S = v * T_c (in um).
    Ref: Cooke & Zeeman (1976) Section 2; Lean 4 theorem `cooke_zeeman_1976_wavelength_period_scaling`.
    """
    return float(v_um_per_min * T_c_min)


def analytical_somite_count(L_mm: float, v_um_per_min: float, T_c_min: float) -> float:
    """Law 2 (`cooke-clock-and-wavefront-2`):
    Compute total segmented somite count N = (L * 1000) / (v * T_c).
    Ref: Cooke (1975) Nature 254:196-199; Cooke & Zeeman (1976) Section 4;
    Lean 4 theorem `cooke_1975_miniature_embryo_size_invariance`.
    """
    S_um = analytical_somite_wavelength(v_um_per_min, T_c_min)
    return float((L_mm * 1000.0) / S_um)


# ============================================================================
# Section 3: Cooke (1975) Size-Invariance Scaling & Ablation Controls
#            (Cooke 1975 Nature 254:196-199 Table 1; Cooke & Zeeman 1976 Sec. 4)
# ============================================================================
def evaluate_size_regulation_regimes(ref: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """Evaluate all 5 experimental regimes from Cooke (1975) & Cooke-Zeeman (1976)
    plus an unscaled-wavefront negative control (constant v across reduced L).
    """
    if ref is None:
        ref = load_reference_data()
    regimes = ref["cooke_1975_xenopus_size_regulation_table"]["regimes"]
    results: list[dict[str, Any]] = []
    for r in regimes:
        L_mm = float(r["axial_length_mm"])
        T_c = float(r["clock_period_min"])
        v_um = float(r["wavefront_speed_um_per_min"])
        S_pred = analytical_somite_wavelength(v_um, T_c)
        N_pred = analytical_somite_count(L_mm, v_um, T_c)
        N_obs = float(r["observed_somite_count_mean"])
        results.append(
            {
                "regime_id": r["regime_id"],
                "label": r["label"],
                "L_mm": L_mm,
                "T_c_min": T_c,
                "v_um_per_min": v_um,
                "S_pred_um": round(S_pred, 3),
                "N_pred": round(N_pred, 3),
                "N_obs": N_obs,
                "N_obs_sd": float(r["observed_somite_count_sd"]),
                "abs_err_N": round(abs(N_pred - N_obs), 4),
            }
        )
    return results


# ============================================================================
# Section 4a: Goldbeter-Gonze-Pourquie (2007) FGF8/RA Bistable Toggle Analysis
#             (Dev. Dyn. 236:1495-1508; Law 4)
# ============================================================================
def _toggle_R_of_F(F: np.ndarray, p: dict[str, float]) -> np.ndarray:
    Kn = p["K_FR"] ** p["n_hill"]
    return p["alpha_R"] * Kn / (Kn + np.power(F, p["n_hill"]))


def _toggle_F_rhs(F: np.ndarray, M: float, p: dict[str, float]) -> np.ndarray:
    """Steady-state FGF synthesis given R at its own steady state for that F."""
    Kn = p["K_RF"] ** p["n_hill"]
    R = _toggle_R_of_F(F, p)
    return p["alpha_F"] * M * Kn / (Kn + np.power(R, p["n_hill"]))


def fgf_ra_bistability_scan(
    params: dict[str, float] | None = None,
    M_max: float = 1.6,
    nM: int = 641,
    nF: int = 6001,
) -> dict[str, Any]:
    """Scan fgf8 mRNA level M_F and locate the steady states of the reduced
    Goldbeter et al. (2007) FGF/RA toggle (fixed points of F = f(F; M)).

    Returns the low/middle/high FGF branches (NaN where absent), the bistable
    window [M_lo, M_hi] bounded by the two saddle-node bifurcations, and the
    determination-front threshold M_lo (below which only the RA-high state
    exists, so FGF-high tailbud-derived cells must switch).
    """
    p = FGF_RA_PARAMS if params is None else params
    M_grid = np.linspace(0.0, M_max, nM)
    F_grid = np.linspace(0.0, p["alpha_F"] * M_max * 1.05 + 1e-6, nF)
    F_low = np.full(nM, np.nan)
    F_mid = np.full(nM, np.nan)
    F_high = np.full(nM, np.nan)
    n_roots = np.zeros(nM, dtype=int)
    for i, M in enumerate(M_grid):
        g = _toggle_F_rhs(F_grid, float(M), p) - F_grid
        s = np.sign(g)
        idx = np.where((s[:-1] * s[1:]) < 0)[0]
        roots = []
        for j in idx:
            f0, f1 = F_grid[j], F_grid[j + 1]
            g0, g1 = g[j], g[j + 1]
            roots.append(float(f0 - g0 * (f1 - f0) / (g1 - g0)))
        if g[0] == 0.0:
            roots.insert(0, 0.0)
        roots = sorted(roots)
        n_roots[i] = len(roots)
        if len(roots) == 1:
            if roots[0] < 0.5 * p["alpha_F"] * max(M, 1e-9) + 0.25:
                F_low[i] = roots[0]
            else:
                F_high[i] = roots[0]
        elif len(roots) >= 3:
            F_low[i], F_mid[i], F_high[i] = roots[0], roots[1], roots[-1]
    bistable = n_roots >= 3
    if bistable.any():
        M_lo = float(M_grid[bistable][0])
        M_hi = float(M_grid[bistable][-1]) if not bistable[-1] else float("inf")
    else:
        M_lo, M_hi = float("nan"), float("nan")
    return {
        "M_grid": M_grid,
        "F_low": F_low,
        "F_mid": F_mid,
        "F_high": F_high,
        "n_roots": n_roots,
        "M_lo": M_lo,
        "M_hi": M_hi,
        "bistable_window_width": (M_hi - M_lo) if np.isfinite(M_hi) else float("inf"),
        "tailbud_M": 1.0,
        "tailbud_is_FGF_high_or_bistable": bool(1.0 >= M_lo),
    }


def _front_position_from_a(a_xt: np.ndarray, x: np.ndarray, L_mm: float) -> float:
    """Anterior-most zero crossing of the cusp unfolding parameter a(x) (a > 0 anterior)."""
    neg = a_xt < 0.0
    if not neg.any():
        return float(L_mm)
    i = int(np.argmax(neg))
    if i == 0:
        return 0.0
    a0, a1 = float(a_xt[i - 1]), float(a_xt[i])
    if a0 == a1:
        return float(x[i])
    return float(x[i - 1] + (x[i] - x[i - 1]) * a0 / (a0 - a1))


# ============================================================================
# Section 4b: Thom-Zeeman Cusp Catastrophe Spatiotemporal PDE/ODE Solver
#             (Cooke & Zeeman 1976 J. Theor. Biol. 58:455-476 Figs. 2-4), with
#             either the kinematic wavefront x_w = v t or the Goldbeter FGF/RA front
# ============================================================================
def simulate_cusp_clock_wavefront(
    L_mm: float = 4.0,
    T_c_min: float = 45.0,
    T_total_min: float = 1800.0,
    v_um_per_min: float | None = None,
    nx: int = 800,
    nt: int = 900,
    sigma_w_mm: float | None = None,
    tau_chi_min: float = 4.0,
    A_clock: float = 0.52,
    b0_bias: float = 0.04,
    k_phi_per_mm: float = 0.0,
    clock_enabled: bool = True,
    seed: int = 1976,
    wavefront_mode: str = "kinematic",
    v_tailbud_um_per_min: float | None = None,
    x_tailbud0_mm: float | None = None,
    fgf_bead: dict[str, float] | None = None,
    fgf_ra_params: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Integrate the Cooke & Zeeman (1976) Thom-Zeeman cusp catastrophe clock-and-wavefront
    system along a 1D anteroposterior axis x in [0, L_mm]:

        tau_chi * dchi(x,t)/dt = -(chi^3 - a(x,t) * chi - b_osc(x,t))
        b_osc(x,t) = b0_bias + A_clock * cos(2*pi*t / T_c_min - k_phi * x)

    `wavefront_mode="kinematic"` (Cooke & Zeeman 1976 Figs. 2-4): the determination
    front is prescribed, a(x,t) = tanh((v t - x)/sigma_w), with a commitment mobility
    window trailing the front.

    `wavefront_mode="fgf_ra"` (Goldbeter et al. 2007 + Dubrulle & Pourquie 2004, Law 4):
    the front is NOT prescribed. The tailbud elongates at v_tailbud, fgf8 mRNA decays in
    the cells it leaves behind (M_F = exp(-(x_tb - x)/lambda_F), lambda_F = v_tb/k_dM),
    and each cell runs the bistable FGF/RA toggle; a(x,t) = tanh((R - F)/kappa_a) and the
    mobility m = F^4/(F^4 + K_mob^4) are read off the toggle, so the front position and
    its velocity emerge from axis elongation. An optional ectopic FGF8 bead
    (Dubrulle et al. 2001) adds M_bead = A_b exp(-t/tau_b) exp(-(x - x_b)^2 / 2 w_b^2).

    Because cellular morphogenetic state locks irreversibly once committed to epithelioid
    somite morphogenesis behind the determination front, the commitment rate mobility
    m(x,t) is concentrated across the active wavefront transition zone (Cooke & Zeeman
    1976 Fig. 3).
    """
    if wavefront_mode not in ("kinematic", "fgf_ra"):
        raise ValueError(f"unknown wavefront_mode {wavefront_mode!r}")
    rng = np.random.default_rng(seed)
    if v_um_per_min is None:
        v_mm_per_min = L_mm / T_total_min
        v_um_per_min = v_mm_per_min * 1000.0
    else:
        v_mm_per_min = v_um_per_min / 1000.0
    if sigma_w_mm is None:
        sigma_w_mm = 0.0045 * L_mm

    x = np.linspace(0.0, L_mm, nx)
    dx_um = float((L_mm * 1000.0) / (nx - 1))
    t_grid = np.linspace(0.0, T_total_min, nt)
    dt = float(t_grid[1] - t_grid[0])

    chi = rng.normal(0.0, 1e-3, size=nx)
    n_snap = 45
    snap_indices = np.linspace(0, nt - 1, n_snap, dtype=int)
    snap_set = set(int(i) for i in snap_indices)
    chi_kymograph = np.zeros((n_snap, nx), dtype=float)
    xw_snaps = np.zeros(n_snap, dtype=float)
    t_snaps = np.zeros(n_snap, dtype=float)
    xw_track = np.zeros(nt, dtype=float)
    snap_ptr = 0

    # ---- Goldbeter (2007) FGF/RA toggle state (Law 4) --------------------------------
    fgf = wavefront_mode == "fgf_ra"
    if fgf:
        p = FGF_RA_PARAMS if fgf_ra_params is None else fgf_ra_params
        if v_tailbud_um_per_min is None:
            v_tailbud_um_per_min = v_um_per_min
        v_tb_mm = v_tailbud_um_per_min / 1000.0
        # In the coupled model the kinematic prediction is v = v_tb (Law 4); analytical
        # comparisons below therefore use the tailbud elongation speed.
        v_um_per_min = v_tailbud_um_per_min
        v_mm_per_min = v_tb_mm
        lam_mm = v_tb_mm / p["k_dM_per_min"]
        scan = fgf_ra_bistability_scan(p)
        M_lo = scan["M_lo"]
        front_offset_mm = lam_mm * math.log(1.0 / M_lo)
        if x_tailbud0_mm is None:
            x_tailbud0_mm = front_offset_mm  # quasi-static front starts at x = 0
        n_h = p["n_hill"]
        KRFn = p["K_RF"] ** n_h
        KFRn = p["K_FR"] ** n_h
        n_mob = p["n_mob"]
        Kmobn = p["K_mob"] ** n_mob
        tau_FR = p["tau_FR_min"]
        kappa_a = p["kappa_a"]

        def _mRNA(t: float) -> np.ndarray:
            x_tb = x_tailbud0_mm + v_tb_mm * t
            M = np.exp(-np.clip(x_tb - x, 0.0, None) / lam_mm)
            if fgf_bead is not None:
                M = M + fgf_bead["amplitude"] * math.exp(-t / fgf_bead["tau_min"]) * np.exp(
                    -0.5 * ((x - fgf_bead["x_mm"]) / fgf_bead["width_mm"]) ** 2
                )
            return M

        M0 = _mRNA(0.0)
        high = M0 >= M_lo
        F = np.where(high, p["alpha_F"] * M0, 0.0)
        R = np.where(high, 0.0, p["alpha_R"])
        for _ in range(120):  # relax onto the branch-appropriate steady state
            F = p["alpha_F"] * M0 * KRFn / (KRFn + R**n_h)
            R = p["alpha_R"] * KFRn / (KFRn + F**n_h)
        F_kymo = np.zeros((n_snap, nx))
        R_kymo = np.zeros((n_snap, nx))
        M_kymo = np.zeros((n_snap, nx))
        xtb_snaps = np.zeros(n_snap)
    else:
        F = R = None  # type: ignore[assignment]

    amp = A_clock if clock_enabled else 0.0
    for it, t in enumerate(t_grid):
        if not fgf:
            x_w = v_mm_per_min * t
            u = (x_w - x) / sigma_w_mm
            a_xt = np.tanh(u)
            # Plasticity window: cells undergo rapid cusp commitment as the wavefront passes
            # and lock in their differentiated somite/cleft state behind the front
            mobility = 0.5 * (1.0 - np.tanh((x_w - x - 2.2 * sigma_w_mm) / (0.8 * sigma_w_mm)))
        else:
            M = _mRNA(float(t))
            dF = (p["alpha_F"] * M * KRFn / (KRFn + R**n_h) - F) / tau_FR
            dR = (p["alpha_R"] * KFRn / (KFRn + F**n_h) - R) / tau_FR
            F = np.clip(F + dt * dF, 0.0, None)
            R = np.clip(R + dt * dR, 0.0, None)
            a_xt = np.tanh((R - F) / kappa_a)
            Fn = F**n_mob
            mobility = Fn / (Fn + Kmobn)
            x_w = _front_position_from_a(a_xt, x, L_mm)
        phi = (2.0 * math.pi * t / T_c_min) - k_phi_per_mm * x
        b_xt = (b0_bias + amp * np.cos(phi)) * (0.5 * (1.0 + a_xt))
        dchi_dt = -(chi**3 - a_xt * chi - b_xt) / tau_chi_min
        chi = np.clip(chi + dt * mobility * dchi_dt, -1.6, 1.6)
        xw_track[it] = x_w

        if it in snap_set:
            chi_kymograph[snap_ptr] = chi
            xw_snaps[snap_ptr] = x_w
            t_snaps[snap_ptr] = t
            if fgf:
                F_kymo[snap_ptr] = F
                R_kymo[snap_ptr] = R
                M_kymo[snap_ptr] = M
                xtb_snaps[snap_ptr] = x_tailbud0_mm + v_tb_mm * t
            snap_ptr += 1

    # Detect segmented somite blocks (chi > 0 domains separated by chi < 0 clefts)
    cleft_minima, _ = signal.find_peaks(
        -chi,
        distance=max(4, int(0.55 * (v_um_per_min * T_c_min) / dx_um)),
        prominence=0.35,
    )
    somite_peaks, _ = signal.find_peaks(
        chi,
        distance=max(4, int(0.55 * (v_um_per_min * T_c_min) / dx_um)),
        prominence=0.35,
    )
    if len(somite_peaks) >= 2:
        spacings_um = np.diff(x[somite_peaks]) * 1000.0
        mean_S_um = float(np.mean(spacings_um))
        sd_S_um = float(np.std(spacings_um))
        cv_S = float(sd_S_um / mean_S_um) if mean_S_um > 0 else 0.0
    else:
        spacings_um = np.array([], dtype=float)
        mean_S_um = 0.0
        sd_S_um = 0.0
        cv_S = 0.0

    # Estimated somite count along the full axis [0, L_mm]
    n_somites_sim = int(round((L_mm * 1000.0) / mean_S_um)) if mean_S_um > 0 else 0
    S_analytical_um = analytical_somite_wavelength(v_um_per_min, T_c_min)
    N_analytical = analytical_somite_count(L_mm, v_um_per_min, T_c_min)

    out: dict[str, Any] = {
        "wavefront_mode": wavefront_mode,
        "x_mm": x,
        "dx_um": dx_um,
        "t_grid_min": t_grid,
        "t_snaps_min": t_snaps,
        "xw_snaps_mm": xw_snaps,
        "xw_track_mm": xw_track,
        "chi_final": chi,
        "chi_kymograph": chi_kymograph,
        "somite_peaks_idx": somite_peaks,
        "cleft_minima_idx": cleft_minima,
        "spacings_um": spacings_um,
        "n_peaks_detected": int(len(somite_peaks)),
        "n_somites_sim": n_somites_sim,
        "mean_S_sim_um": round(mean_S_um, 4),
        "sd_S_sim_um": round(sd_S_um, 4),
        "cv_S_sim": round(cv_S, 4),
        "S_analytical_um": round(S_analytical_um, 4),
        "N_analytical": round(N_analytical, 4),
        "v_um_per_min": float(v_um_per_min),
        "wavelength_rel_error_pct": round(
            100.0 * abs(mean_S_um - S_analytical_um) / S_analytical_um, 4
        )
        if S_analytical_um > 0 and mean_S_um > 0
        else 100.0,
    }

    if fgf:
        # Endogenous front velocity: linear fit of x_w(t) in the steady regime
        lo, hi = 0.25 * L_mm, 0.9 * L_mm
        sel = (xw_track >= lo) & (xw_track <= hi) & (t_grid > 60.0)
        if sel.sum() >= 10:
            fit = stats.linregress(t_grid[sel], xw_track[sel] * 1000.0)
            v_fit = float(fit.slope)
            r2 = float(fit.rvalue**2)
            quasi_static = (x_tailbud0_mm + v_tb_mm * t_grid[sel]) - front_offset_mm
            lag_mm = float(np.mean(quasi_static - xw_track[sel]))
        else:
            v_fit, r2, lag_mm = float("nan"), float("nan"), float("nan")
        # Front sharpness: 10-90 % width of a(x) at the final snapshot
        a_final = np.tanh((R - F) / kappa_a)
        inside = (a_final < 0.8) & (a_final > -0.8)
        front_width_um = float(inside.sum() * dx_um)
        out.update(
            {
                "F_kymograph": F_kymo,
                "R_kymograph": R_kymo,
                "M_kymograph": M_kymo,
                "F_final": F,
                "R_final": R,
                "xtb_snaps_mm": xtb_snaps,
                "v_tailbud_um_per_min": float(v_tailbud_um_per_min),
                "lambda_F_mm": round(lam_mm, 5),
                "M_lo": round(M_lo, 5),
                "M_hi": scan["M_hi"] if np.isfinite(scan["M_hi"]) else None,
                "front_offset_mm": round(front_offset_mm, 5),
                "x_tailbud0_mm": float(x_tailbud0_mm),
                "v_front_fit_um_per_min": round(v_fit, 5),
                "v_front_fit_r2": round(r2, 6),
                "v_front_rel_error_pct": round(
                    100.0 * abs(v_fit - v_tailbud_um_per_min) / v_tailbud_um_per_min, 4
                ),
                "front_dynamic_lag_mm": round(lag_mm, 5),
                "front_width_10_90_um": round(front_width_um, 2),
                "fgf_bead": dict(fgf_bead) if fgf_bead else None,
            }
        )
    return out


def evaluate_endogenous_front_sweep(
    v_tb_values_um_per_min: tuple[float, ...] = (1.3333, 1.6667, 2.2222, 3.3333),
    T_c_min: float = 45.0,
) -> list[dict[str, Any]]:
    """Parameter sweep over tailbud elongation speed v_tb (Law 4): the determination
    front velocity and the somite wavelength S must follow v_front = v_tb and
    S = v_tb * T_c with no wavefront parameter prescribed."""
    rows = []
    for v_tb in v_tb_values_um_per_min:
        sim = simulate_cusp_clock_wavefront(
            L_mm=4.0, T_c_min=T_c_min, T_total_min=1800.0,
            wavefront_mode="fgf_ra", v_tailbud_um_per_min=v_tb,
        )
        rows.append(
            {
                "v_tb_um_per_min": v_tb,
                "v_front_fit_um_per_min": sim["v_front_fit_um_per_min"],
                "S_sim_um": sim["mean_S_sim_um"],
                "S_pred_um": round(v_tb * T_c_min, 4),
                "cv_S": sim["cv_S_sim"],
                "n_peaks": sim["n_peaks_detected"],
                "lambda_F_mm": sim["lambda_F_mm"],
            }
        )
    return rows


# ============================================================================
# Section 4c: 2D Vertex-Model Paraxial Mesoderm Sheet with Differential Cleft Tension
#             (Farhadifar et al. 2007 Curr. Biol.; Bi et al. 2015 Nat. Phys.; Law 5)
# ============================================================================
def build_hexagonal_psm_sheet(
    sheet_length_mm: float = 1.2,
    n_rows: int = 6,
    cell_pitch_um: float = 20.0,
    seed: int = 1976,
) -> dict[str, Any]:
    """Build a pointy-top hexagonal cell packing (AP axis = x) with deduplicated
    vertices, cell->vertex incidence (CCW), edge list and edge->cell incidence."""
    rng = np.random.default_rng(seed)
    r = cell_pitch_um / math.sqrt(3.0)  # circumradius
    w = cell_pitch_um
    n_cols = int(round(sheet_length_mm * 1000.0 / w))
    centers = []
    for row in range(n_rows):
        for col in range(n_cols):
            cx = col * w + (0.5 * w if row % 2 else 0.0) + 0.5 * w
            cy = row * 1.5 * r + r
            centers.append((cx, cy))
    centers = np.asarray(centers, dtype=float)
    ang = np.deg2rad(30.0 + 60.0 * np.arange(6))
    offs = r * np.stack([np.cos(ang), np.sin(ang)], axis=-1)
    verts_all = centers[:, None, :] + offs[None, :, :]
    key = np.round(verts_all.reshape(-1, 2) / (0.02 * r)).astype(np.int64)
    uniq, inv = np.unique(key, axis=0, return_inverse=True)
    inv = inv.reshape(-1)
    V = np.zeros((len(uniq), 2), dtype=float)
    V[inv] = verts_all.reshape(-1, 2)
    cells = inv.reshape(-1, 6)
    he = np.stack([cells, np.roll(cells, -1, axis=1)], axis=-1).reshape(-1, 2)
    he_sorted = np.sort(he, axis=1)
    edges, e_inv = np.unique(he_sorted, axis=0, return_inverse=True)
    e_inv = e_inv.reshape(-1)
    edge_cells = -np.ones((len(edges), 2), dtype=int)
    cell_of_he = np.repeat(np.arange(len(cells)), 6)
    for e_id, c_id in zip(e_inv.tolist(), cell_of_he.tolist()):
        if edge_cells[e_id, 0] < 0:
            edge_cells[e_id, 0] = c_id
        else:
            edge_cells[e_id, 1] = c_id
    boundary_edge = edge_cells[:, 1] < 0
    pinned = np.zeros(len(V), dtype=bool)
    pinned[edges[boundary_edge].ravel()] = True
    V = V + rng.normal(0.0, 0.03 * r, size=V.shape) * (~pinned)[:, None]
    A0 = 1.5 * math.sqrt(3.0) * r * r
    return {
        "V0_um": V,
        "cells": cells,
        "edges": edges,
        "edge_cells": edge_cells,
        "boundary_edge": boundary_edge,
        "pinned": pinned,
        "centers0_um": centers,
        "A0_um2": A0,
        "l0_um": math.sqrt(A0),
        "n_cols": n_cols,
        "n_rows": n_rows,
        "sheet_length_mm": sheet_length_mm,
        "sheet_height_um": float(V[:, 1].max() - V[:, 1].min()),
    }


def _vertex_model_gradient(
    V: np.ndarray,
    cells: np.ndarray,
    edges: np.ndarray,
    Lambda_e: np.ndarray,
    A0: float,
    K_A: float,
    Gamma: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Analytic gradient of the Farhadifar et al. (2007) vertex energy
    E = sum_c K_A/2 (A_c - A0)^2 + Gamma/2 P_c^2 + sum_e Lambda_e l_e."""
    P = V[cells]                      # (nc, 6, 2)
    Pn = np.roll(P, -1, axis=1)
    Pp = np.roll(P, 1, axis=1)
    A = 0.5 * np.sum(P[..., 0] * Pn[..., 1] - Pn[..., 0] * P[..., 1], axis=1)
    dA = 0.5 * np.stack([Pn[..., 1] - Pp[..., 1], Pp[..., 0] - Pn[..., 0]], axis=-1)
    seg = Pn - P
    seg_len = np.linalg.norm(seg, axis=-1) + 1e-12
    Per = seg_len.sum(axis=1)
    unit = seg / seg_len[..., None]
    dP = -unit + np.roll(unit, 1, axis=1)
    grad_cell = (K_A * (A - A0))[:, None, None] * dA + (Gamma * Per)[:, None, None] * dP
    grad = np.zeros_like(V)
    np.add.at(grad, cells.ravel(), grad_cell.reshape(-1, 2))
    d = V[edges[:, 0]] - V[edges[:, 1]]
    l = np.linalg.norm(d, axis=1) + 1e-12
    u = d / l[:, None]
    np.add.at(grad, edges[:, 0], Lambda_e[:, None] * u)
    np.add.at(grad, edges[:, 1], -Lambda_e[:, None] * u)
    return grad, A, Per


def _cleft_metrics(
    V_um: np.ndarray, edges: np.ndarray, cleft_mask: np.ndarray, sheet_height_um: float
) -> dict[str, float]:
    """Length-weighted mean deviation of cleft edges from the mediolateral (y) axis,
    cleft tortuosity (total cleft length / n_clefts * sheet height) and 2D somite pitch."""
    if cleft_mask.sum() == 0:
        return {"angle_deg": float("nan"), "tortuosity": float("nan"), "n_clefts": 0, "S_2d_um": float("nan")}
    d = V_um[edges[cleft_mask, 0]] - V_um[edges[cleft_mask, 1]]
    l = np.linalg.norm(d, axis=1)
    ang = np.degrees(np.arctan2(np.abs(d[:, 0]), np.abs(d[:, 1]) + 1e-12))  # 0 deg = along ML axis
    angle = float(np.sum(ang * l) / np.sum(l))
    mid_x = 0.5 * (V_um[edges[cleft_mask, 0], 0] + V_um[edges[cleft_mask, 1], 0])
    # group cleft edges into individual clefts by 1D clustering on x
    order = np.argsort(mid_x)
    xs = mid_x[order]
    gaps = np.where(np.diff(xs) > 25.0)[0]
    groups = np.split(order, gaps + 1)
    n_clefts = len(groups)
    cleft_x = np.array([float(np.mean(mid_x[g])) for g in groups])
    S_2d = float(np.mean(np.diff(np.sort(cleft_x)))) if n_clefts >= 2 else float("nan")
    # per-cleft tortuosity = path length / end-to-end mediolateral span of that cleft
    cleft_edges = edges[cleft_mask]
    torts = []
    for g in groups:
        vid = np.unique(cleft_edges[g].ravel())
        span = float(V_um[vid, 1].max() - V_um[vid, 1].min())
        if span > 1e-6:
            torts.append(float(np.sum(l[g]) / span))
    tort = float(np.mean(torts)) if torts else float("nan")
    return {"angle_deg": angle, "tortuosity": tort, "n_clefts": int(n_clefts), "S_2d_um": S_2d}


def simulate_vertex_sheet_segmentation(
    sim: dict[str, Any],
    params: dict[str, float] | None = None,
    delta_lambda: float | None = None,
    seed: int = 1976,
) -> dict[str, Any]:
    """Couple the 1D cusp field chi(x,t) (kinematic or FGF/RA-driven) to a 2D vertex
    model of the anterior paraxial mesoderm sheet (Law 5).

    At each kymograph snapshot, every cell whose centroid has |chi| above the commitment
    threshold is assigned its half-somite identity (+1 anterior / -1 posterior); each
    edge separating a posterior(-1) cell (anterior side) from an anterior(+1) cell
    (posterior side) is an inter-somitic cleft and receives extra line tension
    dLambda (Farhadifar et al. 2007 energy, gradient descent, no T1 transitions).
    Cleft orientation relative to the mediolateral axis and tortuosity are tracked
    stage by stage to quantify cleft rotation/straightening.
    """
    vp = dict(VERTEX_PARAMS if params is None else params)
    if delta_lambda is not None:
        vp["delta_Lambda_cleft"] = delta_lambda
    mesh = build_hexagonal_psm_sheet(
        vp["sheet_length_mm"], int(vp["n_rows"]), vp["cell_pitch_um"], seed=seed
    )
    l0 = mesh["l0_um"]
    V = mesh["V0_um"] / l0
    cells, edges, edge_cells = mesh["cells"], mesh["edges"], mesh["edge_cells"]
    pinned = mesh["pinned"]
    interior_edge = edge_cells[:, 1] >= 0
    x_mm = sim["x_mm"]
    kymo = sim["chi_kymograph"]
    t_snaps = sim["t_snaps_min"]
    xw_snaps = sim["xw_snaps_mm"]
    L_sheet = vp["sheet_length_mm"]
    # stages: snapshots until the front has cleared the sheet, plus extra relaxation
    last = int(np.argmax(xw_snaps > L_sheet + 0.15)) if (xw_snaps > L_sheet + 0.15).any() else len(xw_snaps) - 1
    stage_snaps = list(range(0, last + 1)) + [last] * 4
    centers = np.array([V[c].mean(axis=0) for c in cells]) * l0  # um
    cell_x_mm = centers[:, 0] / 1000.0
    c0, c1 = edge_cells[:, 0], edge_cells[:, 1]
    # orient each interior edge so that cell "ant" is the anterior (smaller x) one
    swap = (centers[np.where(interior_edge, c1, c0), 0] < centers[np.where(interior_edge, c0, c0), 0])
    ant = np.where(swap, c1, c0)
    post = np.where(swap, c0, c1)

    history: list[dict[str, Any]] = []
    first_commit_angle: dict[int, float] = {}
    state = np.zeros(len(cells), dtype=int)
    cleft_mask = np.zeros(len(edges), dtype=bool)
    for k, s_idx in enumerate(stage_snaps):
        chi_c = np.interp(cell_x_mm, x_mm, kymo[s_idx])
        # a cell commits once |chi| clears the threshold, or unconditionally (by sign) once it
        # lies > 0.1 mm behind the front, so cells sitting on a chi zero-crossing still resolve
        committed = (np.abs(chi_c) > vp["commit_threshold"]) | (cell_x_mm < xw_snaps[s_idx] - 0.1)
        new_state = np.where(committed, np.sign(chi_c).astype(int), 0)
        state = np.where(state == 0, new_state, state)  # irreversible commitment
        cleft_mask = interior_edge & (state[ant] == -1) & (state[post] == 1)
        Lambda_e = np.full(len(edges), vp["Lambda_0"])
        Lambda_e[cleft_mask] += vp["delta_Lambda_cleft"]
        if k == 0 or len(history) == 0:
            m_pre = _cleft_metrics(V * l0, edges, cleft_mask, mesh["sheet_height_um"])
        for _ in range(int(vp["steps_per_stage"])):
            grad, A, Per = _vertex_model_gradient(V, cells, edges, Lambda_e, 1.0, vp["K_A"], vp["Gamma"])
            grad[pinned] = 0.0
            V = V - vp["step"] * grad
        m = _cleft_metrics(V * l0, edges, cleft_mask, mesh["sheet_height_um"])
        history.append(
            {
                "stage": k,
                "t_min": float(t_snaps[s_idx]),
                "x_w_mm": float(xw_snaps[s_idx]),
                "n_committed_cells": int(committed.sum()),
                "cleft_angle_deg": m["angle_deg"],
                "cleft_tortuosity": m["tortuosity"],
                "n_clefts": m["n_clefts"],
            }
        )
    # geometry of the undeformed lattice cleft (same mask, V0) = angle at commitment
    m0 = _cleft_metrics(mesh["V0_um"], edges, cleft_mask, mesh["sheet_height_um"])
    mF = _cleft_metrics(V * l0, edges, cleft_mask, mesh["sheet_height_um"])
    grad, A, Per = _vertex_model_gradient(V, cells, edges, np.full(len(edges), vp["Lambda_0"]), 1.0, vp["K_A"], vp["Gamma"])
    shape_index = Per / np.sqrt(np.abs(A) + 1e-12)
    # boundary (cleft-adjacent) vs core cell AP elongation
    cleft_cells = np.unique(np.concatenate([ant[cleft_mask], post[cleft_mask]])) if cleft_mask.any() else np.array([], int)
    Pv = V[cells] * l0
    ext_x = Pv[..., 0].max(axis=1) - Pv[..., 0].min(axis=1)
    ext_y = Pv[..., 1].max(axis=1) - Pv[..., 1].min(axis=1)
    ap_aspect = ext_x / ext_y
    core_mask = np.ones(len(cells), bool)
    core_mask[cleft_cells] = False
    return {
        "mesh": mesh,
        "V_final_um": V * l0,
        "cell_state": state,
        "cleft_mask": cleft_mask,
        "stage_history": history,
        "cleft_angle_lattice_deg": round(m0["angle_deg"], 3),
        "cleft_angle_final_deg": round(mF["angle_deg"], 3),
        "cleft_tortuosity_lattice": round(m0["tortuosity"], 4),
        "cleft_tortuosity_final": round(mF["tortuosity"], 4),
        "n_clefts": int(mF["n_clefts"]),
        "n_somite_blocks_2d": int(mF["n_clefts"] + 1),
        "N_sheet_analytical": round(L_sheet * 1000.0 / sim["S_analytical_um"], 3) if sim["S_analytical_um"] > 0 else float("nan"),
        "S_2d_um": round(mF["S_2d_um"], 3),
        "shape_index_mean": round(float(np.mean(shape_index)), 4),
        "shape_index_sd": round(float(np.std(shape_index)), 4),
        "ap_aspect_cleft_cells": round(float(np.mean(ap_aspect[cleft_cells])), 4) if len(cleft_cells) else float("nan"),
        "ap_aspect_core_cells": round(float(np.mean(ap_aspect[core_mask])), 4),
        "delta_Lambda_cleft": vp["delta_Lambda_cleft"],
        "n_cells": int(len(cells)),
        "n_vertices": int(len(V)),
        "n_edges": int(len(edges)),
    }


# ============================================================================
# Section 5: Biological Specimen Computer-Vision Segmentation
#            (`datasets/clock_and_wavefront/somite_embryo_specimen.jpg`)
# ============================================================================
def segment_vertebrate_somite_specimen(
    img_path: Path = SPECIMEN_JPG,
) -> dict[str, Any]:
    """Segment the dorsal vertebrate embryo SEM specimen (`somite_embryo_specimen.jpg`,
    CC BY 4.0) to extract:
      - Upper and lower bilateral paraxial somite rows flanking the central neural tube
      - Individual somite dome centroids and inter-somitic clefts along the AP axis
      - Empirical somite wavelength S_obs (px), spacing CV, and bilateral symmetry ratio
      - Goodness-of-fit (Pearson r and normalized RMSE) between the empirical AP
        somite profile and the Cooke-Zeeman cusp wavefront spatial profile.
    """
    im = np.asarray(Image.open(img_path).convert("RGB"), dtype=float)
    R, G, B = im[..., 0], im[..., 1], im[..., 2]
    lum = 0.299 * R + 0.587 * G + 0.114 * B
    magenta_idx = (R - G) + 0.8 * (B - G)

    x0, x1 = 130, 550
    upper_y0, upper_y1 = 208, 252
    lower_y0, lower_y1 = 312, 356

    def _extract_row(y0: int, y1: int) -> dict[str, Any]:
        strip_L = lum[y0:y1, x0:x1].mean(axis=0)
        sig = ndi.gaussian_filter1d(strip_L, sigma=3.5) - ndi.gaussian_filter1d(
            strip_L, sigma=22.0
        )
        peaks, _ = signal.find_peaks(sig, distance=32, prominence=1.8)
        clefts, _ = signal.find_peaks(-sig, distance=32, prominence=1.8)
        dx = np.diff(peaks).astype(float)
        mean_s = float(np.mean(dx))
        sd_s = float(np.std(dx))
        return {
            "signal": sig,
            "peaks_x": (peaks + x0).tolist(),
            "clefts_x": (clefts + x0).tolist(),
            "n_somites": int(len(peaks)),
            "spacings_px": dx.tolist(),
            "mean_spacing_px": round(mean_s, 4),
            "sd_spacing_px": round(sd_s, 4),
            "cv_spacing": round(sd_s / mean_s, 4),
        }

    upper = _extract_row(upper_y0, upper_y1)
    lower = _extract_row(lower_y0, lower_y1)
    all_spacings = np.array(upper["spacings_px"] + lower["spacings_px"], dtype=float)
    bilateral_mean_px = float(np.mean(all_spacings))
    bilateral_sd_px = float(np.std(all_spacings))
    bilateral_cv = float(bilateral_sd_px / bilateral_mean_px)
    symmetry_ratio = float(upper["mean_spacing_px"] / lower["mean_spacing_px"])

    # Compare normalized empirical somite profile against a phase-aligned Cooke-Zeeman wave
    x_px = np.arange(x1 - x0, dtype=float)
    emp_norm = lower["signal"] / (np.std(lower["signal"]) + 1e-12)
    # Fit fundamental phase at wavelength = lower["mean_spacing_px"]
    k_emp = 2.0 * math.pi / lower["mean_spacing_px"]
    c_cos = np.cos(k_emp * x_px)
    c_sin = np.sin(k_emp * x_px)
    phi_fit = math.atan2(float(np.dot(emp_norm, c_sin)), float(np.dot(emp_norm, c_cos)))
    model_wave = np.tanh(1.8 * np.cos(k_emp * x_px - phi_fit))
    model_norm = model_wave / (np.std(model_wave) + 1e-12)
    r_val, p_val = stats.pearsonr(emp_norm, model_norm)

    # Pass 3: empirical inter-somitic cleft orientation on the specimen. For each lower-row
    # cleft, locate the luminance-minimum column in thin horizontal bands across the row and
    # fit a line; the deviation of that line from the image's ML (vertical) axis is the
    # observed cleft angle to which the vertex-sheet cleft angle is compared.
    cleft_angles: list[float] = []
    for cx in lower["clefts_x"]:
        ys, xs = [], []
        for yb in range(lower_y0, lower_y1 - 3, 4):
            band = ndi.gaussian_filter1d(lum[yb:yb + 4, max(cx - 12, 0):cx + 13].mean(axis=0), 1.5)
            xs.append(max(cx - 12, 0) + int(np.argmin(band)))
            ys.append(yb + 2.0)
        if len(xs) >= 4:
            slope = stats.linregress(ys, xs).slope  # dx/dy
            cleft_angles.append(float(np.degrees(np.arctan(abs(slope)))))
    cleft_angle_mean = float(np.mean(cleft_angles)) if cleft_angles else float("nan")

    return {
        "image_shape": list(im.shape),
        "ap_roi_x_px": [x0, x1],
        "upper_row": upper,
        "lower_row": lower,
        "bilateral_mean_spacing_px": round(bilateral_mean_px, 4),
        "bilateral_sd_spacing_px": round(bilateral_sd_px, 4),
        "bilateral_cv_spacing": round(bilateral_cv, 4),
        "bilateral_symmetry_ratio": round(symmetry_ratio, 4),
        "magenta_somite_mask_fraction": round(
            float(np.mean(magenta_idx[200:365, x0:x1] > 18.0)), 4
        ),
        "profile_phase_correlation_r": round(float(r_val), 4),
        "profile_phase_correlation_p": float(p_val),
        "lower_row_cleft_angles_deg": [round(a, 2) for a in cleft_angles],
        "lower_row_cleft_angle_mean_deg": round(cleft_angle_mean, 3),
    }


# ============================================================================
# Section 6: MorphoEval Dynamic Measurement Hook (`measure_morphoeval()`)
# ============================================================================
def run_pass3_models() -> dict[str, Any]:
    """Run the Pass 3 coupled models once and return every derived quantity."""
    sim_fgf = simulate_cusp_clock_wavefront(
        L_mm=4.0, T_c_min=45.0, T_total_min=1800.0, wavefront_mode="fgf_ra"
    )
    sim_fgf_mini = simulate_cusp_clock_wavefront(
        L_mm=2.4, T_c_min=45.0, T_total_min=1800.0, wavefront_mode="fgf_ra",
        v_tailbud_um_per_min=1.3333333333333333,
    )
    sim_bead = simulate_cusp_clock_wavefront(
        L_mm=4.0, T_c_min=45.0, T_total_min=1800.0, wavefront_mode="fgf_ra",
        fgf_bead={"x_mm": 0.6, "amplitude": 1.0, "width_mm": 0.06, "tau_min": 300.0},
    )
    sweep = evaluate_endogenous_front_sweep()
    scan = fgf_ra_bistability_scan()
    sheet = simulate_vertex_sheet_segmentation(sim_fgf)
    sheet_ctrl = simulate_vertex_sheet_segmentation(sim_fgf, delta_lambda=0.0)
    # Bead-adjacent somite size ratio (Dubrulle et al. 2001): spacings whose midpoint lies
    # within +/- 0.3 mm of the bead vs the control wavelength
    xb = sim_bead["fgf_bead"]["x_mm"]
    pk_x = sim_bead["x_mm"][sim_bead["somite_peaks_idx"]]
    mids = 0.5 * (pk_x[1:] + pk_x[:-1])
    near = np.abs(mids - xb) <= 0.3
    local = sim_bead["spacings_um"][near] if near.any() else np.array([])
    bead_ratio_min = float(np.min(local) / sim_fgf["mean_S_sim_um"]) if len(local) else float("nan")
    bead_ratio_max = float(np.max(local) / sim_fgf["mean_S_sim_um"]) if len(local) else float("nan")
    s_sim = np.array([r["S_sim_um"] for r in sweep])
    s_pred = np.array([r["S_pred_um"] for r in sweep])
    v_fit = np.array([r["v_front_fit_um_per_min"] for r in sweep])
    v_tb = np.array([r["v_tb_um_per_min"] for r in sweep])
    return {
        "sim_fgf": sim_fgf,
        "sim_fgf_mini": sim_fgf_mini,
        "sim_bead": sim_bead,
        "sweep": sweep,
        "scan": scan,
        "sheet": sheet,
        "sheet_ctrl": sheet_ctrl,
        "bead_local_min_ratio": round(bead_ratio_min, 4),
        "bead_local_max_ratio": round(bead_ratio_max, 4),
        "bead_n_local_spacings": int(len(local)),
        "sweep_S_rmse_um": round(float(np.sqrt(np.mean((s_sim - s_pred) ** 2))), 4),
        "sweep_S_r": round(float(stats.pearsonr(s_sim, s_pred)[0]), 6),
        "sweep_v_rmse_um_per_min": round(float(np.sqrt(np.mean((v_fit - v_tb) ** 2))), 5),
        "sweep_v_max_rel_err_pct": round(float(np.max(np.abs(v_fit - v_tb) / v_tb) * 100.0), 4),
    }


def measure_morphoeval() -> list[dict[str, Any]]:
    """Return quantitative morphometric measurements for `morphoeval/measure_outputs.py`."""
    verify_dataset_integrity()
    ref = load_reference_data()
    sim_norm = simulate_cusp_clock_wavefront(L_mm=4.0, T_c_min=45.0, T_total_min=1800.0)
    sim_mini = simulate_cusp_clock_wavefront(L_mm=2.4, T_c_min=45.0, T_total_min=1800.0)
    sim_slow = simulate_cusp_clock_wavefront(
        L_mm=4.0, T_c_min=90.0, T_total_min=1800.0, v_um_per_min=2.2222222222222223
    )
    sim_null = simulate_cusp_clock_wavefront(
        L_mm=4.0, T_c_min=45.0, T_total_min=1800.0, clock_enabled=False
    )
    spec = segment_vertebrate_somite_specimen()
    regimes = evaluate_size_regulation_regimes(ref)
    n_pred_vec = np.array([r["N_pred"] for r in regimes], dtype=float)
    n_obs_vec = np.array([r["N_obs"] for r in regimes], dtype=float)
    r_cooke, p_cooke = stats.pearsonr(n_pred_vec, n_obs_vec)
    rmse_cooke = float(np.sqrt(np.mean((n_pred_vec - n_obs_vec) ** 2)))
    p3 = run_pass3_models()
    sf, sh, shc, scan = p3["sim_fgf"], p3["sheet"], p3["sheet_ctrl"], p3["scan"]
    ref3 = load_fgf_ra_reference()
    p0_ref = float(
        ref3.get("farhadifar_2007_vertex_sheet", {})
        .get("reference_values", {})
        .get("shape_index_rigidity_transition", {})
        .get("value", 3.81)
    )

    return [
        {
            "name": "normal Xenopus somite wavelength S_sim vs analytical S = v*T_c",
            "value": round(sim_norm["mean_S_sim_um"], 4),
            "unit": "um",
            "method": f"cusp clock-and-wavefront ODE solver peak-to-peak somite spacing (CV={sim_norm['cv_S_sim']:.4f}, n={sim_norm['n_peaks_detected']} somites)",
            "reference_value": 100.0,
            "reference_source": "Cooke & Zeeman (1976) J. Theor. Biol. 58:455-476 (L=4.0 mm, T_c=45 min, v=2.222 um/min => S=100.0 um)",
            "note": f"relative error = {sim_norm['wavelength_rel_error_pct']:.3f}%, spacing CV = {sim_norm['cv_S_sim']:.4f}",
            "derived_from": "script",
        },
        {
            "name": "miniature operated Xenopus embryo (60% length) somite count N_mini vs normal N_norm (size-invariance ratio)",
            "value": round(sim_mini["n_somites_sim"] / max(1, sim_norm["n_somites_sim"]), 4),
            "unit": "N_mini / N_norm ratio",
            "method": f"proportional wavefront scaling v' = 0.60*v_0 in L'=2.4 mm miniature embryo (S_mini={sim_mini['mean_S_sim_um']:.2f} um vs S_norm={sim_norm['mean_S_sim_um']:.2f} um)",
            "reference_value": 0.9851,
            "reference_source": "Cooke (1975) Nature 254:196-199 Table 1 (39.6 / 40.2 = 0.9851 somite count invariance ratio; theoretical = 1.0000)",
            "note": f"N_mini={sim_mini['n_somites_sim']} vs N_norm={sim_norm['n_somites_sim']}; across 5 Cooke regimes r={r_cooke:.4f} (p={p_cooke:.2e}), RMSE={rmse_cooke:.3f} somites",
            "derived_from": "script",
        },
        {
            "name": "clock-doubled perturbation (2x T_c) vs clock-arrested ablation control somite count delta",
            "value": round(float(sim_slow["n_somites_sim"] - sim_null["n_peaks_detected"]), 4),
            "unit": "somites (slow_clock vs null_control)",
            "method": "2x clock period perturbation (T_c=90 min -> N=20, S=200 um) vs clock-arrested negative control (A_clock=0 -> 0 somites)",
            "reference_value": 20.0,
            "reference_source": "Cooke & Zeeman (1976) J. Theor. Biol. 58:455-476 Section 3 (doubling T_c halves N from 40 to 20; arresting clock abolishes segmentation N=0)",
            "note": f"slow_clock S={sim_slow['mean_S_sim_um']:.2f} um (CV={sim_slow['cv_S_sim']:.4f}), null_control peaks={sim_null['n_peaks_detected']}",
            "derived_from": "script",
        },
        {
            "name": "vertebrate embryo SEM specimen bilateral somite spacing S_obs",
            "value": round(spec["bilateral_mean_spacing_px"], 4),
            "unit": "px",
            "method": f"bilateral dorsal somite row DoG peak segmentation on somite_embryo_specimen.jpg (upper={spec['upper_row']['n_somites']} somites, lower={spec['lower_row']['n_somites']} somites, CV={spec['bilateral_cv_spacing']:.4f})",
            "reference_value": 48.47,
            "reference_source": "Vendored CC BY 4.0 vertebrate embryo SEM specimen (datasets/clock_and_wavefront/somite_embryo_specimen.jpg)",
            "note": f"bilateral symmetry ratio = {spec['bilateral_symmetry_ratio']:.4f}, std = {spec['bilateral_sd_spacing_px']:.2f} px, phase correlation r = {spec['profile_phase_correlation_r']:.4f}",
            "derived_from": "output",
        },
        {
            "name": "Cooke (1975) 5-regime somite count prediction RMSE",
            "value": round(rmse_cooke, 4),
            "unit": "somites",
            "method": f"RMSE across 5 Cooke (1975) / Cooke & Zeeman (1976) size-regulation and clock/wavefront perturbation regimes (Pearson r={r_cooke:.4f}, p={p_cooke:.2e})",
            "reference_value": 0.0,
            "reference_source": "Cooke (1975) Nature 254:196-199 Table 1",
            "note": f"max regime error = {max(r['abs_err_N'] for r in regimes):.3f} somites across normal, 75%, 60% miniature, 2x clock, and 1.5x wavefront",
            "derived_from": "script",
        },
        # ---------------- Pass 3: Goldbeter (2007) FGF/RA endogenous determination front ----
        {
            "name": "endogenous FGF8/RA determination-front velocity v_front vs tailbud elongation speed v_tb",
            "value": sf["v_front_fit_um_per_min"],
            "unit": "um/min",
            "method": f"linear fit of the a(x,t)=0 front locus x_w(t) over x in [1.0, 3.6] mm in the coupled Goldbeter FGF/RA toggle + cusp solver (R^2={sf['v_front_fit_r2']:.6f}; no wavefront velocity prescribed)",
            "reference_value": 2.2222,
            "reference_source": "Goldbeter, Gonze & Pourquie (2007) Dev. Dyn. 236:1495-1508 (front regresses with axis elongation) and Cooke & Zeeman (1976) kinematic v = L/T_total = 2.222 um/min",
            "note": f"rel. error {sf['v_front_rel_error_pct']:.3f}%; lambda_F={sf['lambda_F_mm']:.3f} mm, front offset behind tailbud={sf['front_offset_mm']:.3f} mm, dynamic lag={sf['front_dynamic_lag_mm']*1000:.1f} um, bistable window M_F in [{scan['M_lo']:.3f}, {scan['M_hi'] if np.isfinite(scan['M_hi']) else 'inf'}]",
            "derived_from": "script",
        },
        {
            "name": "somite wavelength with endogenous FGF/RA front S_sim(fgf_ra) vs S = v_tb*T_c",
            "value": sf["mean_S_sim_um"],
            "unit": "um",
            "method": f"peak-to-peak somite spacing of the FGF/RA-driven cusp solver (CV={sf['cv_S_sim']:.4f}, n={sf['n_peaks_detected']}, N_sim={sf['n_somites_sim']}; miniature v_tb=1.333 um/min gives S={p3['sim_fgf_mini']['mean_S_sim_um']:.2f} um, N={p3['sim_fgf_mini']['n_somites_sim']})",
            "reference_value": 100.0,
            "reference_source": "Cooke & Zeeman (1976) S = v*T_c with v supplied by tailbud elongation (Goldbeter et al. 2007; Dubrulle & Pourquie 2004)",
            "note": f"rel. error {sf['wavelength_rel_error_pct']:.3f}%; front 10-90% width = {sf['front_width_10_90_um']:.1f} um (kinematic sigma_w = 18 um)",
            "derived_from": "script",
        },
        {
            "name": "v_tb sweep (1.33-3.33 um/min): RMSE of S_sim(fgf_ra) against S = v_tb*T_c",
            "value": p3["sweep_S_rmse_um"],
            "unit": "um",
            "method": f"4-point sweep of tailbud elongation speed in the coupled model; Pearson r={p3['sweep_S_r']:.6f}; front-velocity RMSE={p3['sweep_v_rmse_um_per_min']:.4f} um/min (max rel. err {p3['sweep_v_max_rel_err_pct']:.2f}%)",
            "reference_value": 0.0,
            "reference_source": "Cooke & Zeeman (1976) linear law S = v*T_c; Goldbeter et al. (2007) v = elongation speed",
            "note": "; ".join(f"v_tb={r['v_tb_um_per_min']:.3f}: v_front={r['v_front_fit_um_per_min']:.3f}, S={r['S_sim_um']:.1f}/{r['S_pred_um']:.1f} um" for r in p3["sweep"]),
            "derived_from": "script",
        },
        {
            "name": "ectopic FGF8 bead: smallest adjacent somite length / control S (Dubrulle 2001)",
            "value": p3["bead_local_min_ratio"],
            "unit": "ratio",
            "method": f"decaying Gaussian fgf8 source (A=1.0, w=60 um, tau=300 min) at x=0.6 mm in the coupled model; min/max somite spacing within +/-0.3 mm of the bead over control S ({p3['bead_n_local_spacings']} spacings; max ratio={p3['bead_local_max_ratio']:.3f})",
            "reference_value": 1.0,
            "reference_source": "control ratio; Dubrulle, McGrew & Pourquie (2001) Cell 106:219-232 report somites formed at the level of an FGF8 bead smaller than contralateral controls (ratio < 1), reproduced by Goldbeter et al. (2007)",
            "note": "directional comparison only: the 2001 paper gives no numerical size ratio, so the model is tested for ratio < 1 (bead stalls the front, extra clock cycles elapse per unit length)",
            "derived_from": "script",
        },
        # ---------------- Pass 3: 2D vertex-model paraxial mesoderm sheet -----------------
        {
            "name": "2D vertex-sheet inter-somitic cleft angle from the mediolateral axis after differential tension",
            "value": sh["cleft_angle_final_deg"],
            "unit": "deg",
            "method": f"Farhadifar vertex model ({sh['n_cells']} cells, {sh['n_edges']} edges, Lambda_0=0.12, Gamma=0.04, dLambda_cleft={sh['delta_Lambda_cleft']}) coupled to the FGF/RA cusp field; length-weighted cleft edge angle, lattice value {sh['cleft_angle_lattice_deg']:.1f} deg -> final {sh['cleft_angle_final_deg']:.1f} deg (dLambda=0 control: {shc['cleft_angle_final_deg']:.1f} deg)",
            "reference_value": spec["lower_row_cleft_angle_mean_deg"],
            "reference_source": "inter-somitic cleft angle measured on the vendored SEM specimen lower somite row (luminance-minimum line fit per cleft; straight transverse clefts = 0 deg)",
            "note": f"rotation = {sh['cleft_angle_lattice_deg'] - sh['cleft_angle_final_deg']:.1f} deg; no T1 transitions, so the residual zig-zag cannot fully resolve",
            "derived_from": "script",
        },
        {
            "name": "2D vertex-sheet cleft tortuosity (cleft length / straight mediolateral span)",
            "value": sh["cleft_tortuosity_final"],
            "unit": "ratio",
            "method": f"total cleft edge length over n_clefts * sheet height; hexagonal lattice zig-zag value {sh['cleft_tortuosity_lattice']:.3f} -> {sh['cleft_tortuosity_final']:.3f} (dLambda=0 control {shc['cleft_tortuosity_final']:.3f})",
            "reference_value": 1.0,
            "reference_source": "straight inter-somitic boundary (Cooke & Zeeman 1976 Fig. 1; Saga & Takeda 2001 Nat. Rev. Genet. 2:835-845)",
            "note": f"{sh['n_clefts']} clefts -> {sh['n_somite_blocks_2d']} somite blocks on the 1.2 mm sheet vs analytical {sh['N_sheet_analytical']:.1f}; 2D cleft pitch S_2d={sh['S_2d_um']:.1f} um",
            "derived_from": "script",
        },
        {
            "name": "2D vertex-sheet somite block pitch S_2d vs S = v*T_c",
            "value": sh["S_2d_um"],
            "unit": "um",
            "method": "mean AP distance between consecutive cleft centroids in the relaxed vertex sheet (posterior(-1)->anterior(+1) interfaces only, Law 5)",
            "reference_value": 100.0,
            "reference_source": "Cooke & Zeeman (1976) S = v*T_c = 100 um for L=4 mm, T_c=45 min",
            "note": f"cleft-adjacent cells AP aspect {sh['ap_aspect_cleft_cells']:.3f} vs core cells {sh['ap_aspect_core_cells']:.3f}",
            "derived_from": "script",
        },
        {
            "name": "2D vertex-sheet mean cell shape index p = P/sqrt(A)",
            "value": sh["shape_index_mean"],
            "unit": "dimensionless",
            "method": f"mean +/- sd ({sh['shape_index_sd']:.3f}) over all cells of the relaxed Farhadifar sheet (case I parameters)",
            "reference_value": p0_ref,
            "reference_source": "Bi, Lopez, Schwarz & Manning (2015) Nat. Phys. 11:1074-1079 solid-fluid rigidity transition p0* = 3.81 (regular hexagon 3.72); vendored in datasets/clock_and_wavefront/fgf_ra_vertex_reference.json",
            "note": "p < 3.81 => solid-like epithelioid sheet, consistent with the absence of spontaneous rearrangement in the committed somites",
            "derived_from": "script",
        },
    ]


# ============================================================================
# Section 7: Publication Figure & Animated GIF Rendering
# ============================================================================
def _draw_vertex_sheet(
    ax: plt.Axes,
    sheet: dict[str, Any],
    sim: dict[str, Any],
    title: str,
    x_window_um: tuple[float, float] | None = None,
) -> None:
    V = sheet["V_final_um"]
    cells = sheet["cells"] if "cells" in sheet else sheet["mesh"]["cells"]
    edges = sheet["mesh"]["edges"]
    polys = [V[c] for c in cells]
    state = sheet["cell_state"]
    colors = np.where(state[:, None] == 1, np.array([[0.85, 0.33, 0.31]]), np.where(state[:, None] == -1, np.array([[0.25, 0.47, 0.85]]), np.array([[0.45, 0.50, 0.58]])))
    pc = PolyCollection(polys, facecolors=colors, edgecolors="#0b0f19", linewidths=0.4, alpha=0.95)
    ax.add_collection(pc)
    cm = sheet["cleft_mask"]
    segs = np.stack([V[edges[cm, 0]], V[edges[cm, 1]]], axis=1)
    ax.add_collection(LineCollection(segs, colors="#facc15", linewidths=2.2))
    if x_window_um is None:
        ax.set_xlim(V[:, 0].min() - 5, V[:, 0].max() + 5)
    else:
        ax.set_xlim(x_window_um[0], x_window_um[1])
    ax.set_ylim(V[:, 1].min() - 5, V[:, 1].max() + 5)
    ax.set_aspect("equal")
    ax.set_title(title, fontweight="bold")
    ax.set_xlabel("Anteroposterior $x$ ($\\mu$m)")
    ax.set_ylabel("Mediolateral $y$ ($\\mu$m)")


def render_comparison_figure(out_png: Path = OUTPUT_PNG) -> Path:
    """Render the 9-panel dark-theme publication synthesis figure
    `docs/media/clock_and_wavefront_comparison.png`.
    """
    out_png.parent.mkdir(parents=True, exist_ok=True)
    ref = load_reference_data()
    sim_norm = simulate_cusp_clock_wavefront(L_mm=4.0, T_c_min=45.0, T_total_min=1800.0)
    sim_mini = simulate_cusp_clock_wavefront(L_mm=2.4, T_c_min=45.0, T_total_min=1800.0)
    sim_slow = simulate_cusp_clock_wavefront(
        L_mm=4.0, T_c_min=90.0, T_total_min=1800.0, v_um_per_min=2.2222222222222223
    )
    sim_null = simulate_cusp_clock_wavefront(
        L_mm=4.0, T_c_min=45.0, T_total_min=1800.0, clock_enabled=False
    )
    spec = segment_vertebrate_somite_specimen()
    regimes = evaluate_size_regulation_regimes(ref)
    p3 = run_pass3_models()
    sf, sb, sheet, sheet_ctrl, scan = p3["sim_fgf"], p3["sim_bead"], p3["sheet"], p3["sheet_ctrl"], p3["scan"]

    plt.rcParams.update(
        {
            "figure.facecolor": "#0b0f19",
            "axes.facecolor": "#131b2e",
            "axes.edgecolor": "#3b4c6b",
            "axes.labelcolor": "#e2e8f0",
            "xtick.color": "#cbd5e1",
            "ytick.color": "#cbd5e1",
            "text.color": "#f8fafc",
            "font.size": 9.5,
        }
    )
    fig, axes = plt.subplots(3, 3, figsize=(16.5, 14.6), dpi=140)
    fig.suptitle(
        "Cooke & Zeeman (1976) Clock-and-Wavefront: Cusp Wavefront, Goldbeter FGF8/RA "
        "Endogenous Front, Vertex-Sheet Clefts & Specimen Morphometrics",
        fontsize=12.5,
        fontweight="bold",
        color="#38bdf8",
        y=0.985,
    )

    # Panel A: Spatiotemporal Kymograph chi(x, t) and wavefront trajectory x_w(t) = v*t
    ax = axes[0, 0]
    im_kymo = ax.imshow(
        sim_norm["chi_kymograph"],
        aspect="auto",
        origin="lower",
        extent=[0.0, 4.0, 0.0, 1800.0],
        cmap="coolwarm",
        vmin=-1.2,
        vmax=1.2,
    )
    ax.plot(
        sim_norm["xw_snaps_mm"],
        sim_norm["t_snaps_min"],
        color="#facc15",
        lw=2.2,
        ls="--",
        label=r"Kinematic wavefront $x_w(t) = v t$",
    )
    ax.plot(
        sf["xw_snaps_mm"],
        sf["t_snaps_min"],
        color="#34d399",
        lw=1.6,
        label=f"FGF8/RA toggle front ($v_{{fit}}={sf['v_front_fit_um_per_min']:.3f}\\,\\mu$m/min)",
    )
    ax.set_title("A. Kymograph $\\chi(x,t)$: kinematic vs FGF8/RA front", fontweight="bold")
    ax.set_xlabel("Anteroposterior Axis $x$ (mm)")
    ax.set_ylabel("Developmental Time $t$ (min)")
    ax.legend(loc="upper left", framealpha=0.85, facecolor="#0f172a", fontsize=8.0)
    fig.colorbar(im_kymo, ax=ax, fraction=0.045, pad=0.03, label="Order parameter $\\chi$")

    # Panel B: Thom-Zeeman Cusp Catastrophe Manifold & Spatial Somite Train
    ax = axes[0, 1]
    x_sub = sim_norm["x_mm"] <= 1.2
    ax.plot(
        sim_norm["x_mm"][x_sub],
        sim_norm["chi_final"][x_sub],
        color="#38bdf8",
        lw=1.8,
        label=f"Normal ($L=4.0$ mm, $S={sim_norm['mean_S_sim_um']:.1f}\\,\\mu$m, $N={sim_norm['n_somites_sim']}$)",
    )
    ax.plot(
        sim_mini["x_mm"][sim_mini["x_mm"] <= 1.2],
        sim_mini["chi_final"][sim_mini["x_mm"] <= 1.2] - 2.3,
        color="#34d399",
        lw=1.6,
        label=f"Miniature 60% ($L=2.4$ mm, $S={sim_mini['mean_S_sim_um']:.1f}\\,\\mu$m, $N={sim_mini['n_somites_sim']}$)",
    )
    ax.plot(
        sim_slow["x_mm"][sim_slow["x_mm"] <= 1.2],
        sim_slow["chi_final"][sim_slow["x_mm"] <= 1.2] - 4.6,
        color="#f472b6",
        lw=1.6,
        label=f"Slow Clock $2 T_c$ ($T_c=90$ min, $S={sim_slow['mean_S_sim_um']:.1f}\\,\\mu$m, $N={sim_slow['n_somites_sim']}$)",
    )
    ax.plot(
        sim_null["x_mm"][sim_null["x_mm"] <= 1.2],
        sim_null["chi_final"][sim_null["x_mm"] <= 1.2] - 6.9,
        color="#94a3b8",
        lw=1.5,
        ls="--",
        label=f"Clock-Arrested Control ($A_c=0$, $N={sim_null['n_peaks_detected']}$ somites)",
    )
    ax.plot(
        sf["x_mm"][sf["x_mm"] <= 1.2],
        sf["chi_final"][sf["x_mm"] <= 1.2] - 9.2,
        color="#fbbf24",
        lw=1.6,
        label=f"FGF8/RA endogenous front ($S={sf['mean_S_sim_um']:.1f}\\,\\mu$m, $N={sf['n_somites_sim']}$)",
    )
    ax.plot(
        sb["x_mm"][sb["x_mm"] <= 1.2],
        sb["chi_final"][sb["x_mm"] <= 1.2] - 11.5,
        color="#c084fc",
        lw=1.6,
        label=f"FGF8 bead @0.6 mm (min local $S$ ratio {p3['bead_local_min_ratio']:.2f})",
    )
    ax.axvline(0.6, ymin=0.0, ymax=0.14, color="#c084fc", lw=1.0, ls=":")
    ax.set_title("B. Anterior Somite Trains ($x \\leq 1.2$ mm), Controls & Perturbations", fontweight="bold", fontsize=9.2)
    ax.set_xlabel("Anteroposterior Position $x$ (mm)")
    ax.set_ylabel("Offset Morphogenetic State $\\chi(x, T_{\\mathrm{final}})$")
    ax.legend(loc="lower right", fontsize=7.2, framealpha=0.88, facecolor="#0f172a")

    # Panel C: Continuous Parameter-Space Surface S(v, T_c) = v * T_c
    ax = axes[0, 2]
    v_grid = np.linspace(1.0, 3.6, 80)
    Tc_grid = np.linspace(25.0, 100.0, 80)
    VV, TT = np.meshgrid(v_grid, Tc_grid)
    SS = VV * TT
    cs = ax.contourf(VV, TT, SS, levels=16, cmap="viridis")
    ax.contour(VV, TT, SS, levels=[60.0, 75.0, 100.0, 150.0, 200.0], colors="white", linewidths=0.9, alpha=0.7)
    for r in regimes:
        ax.scatter(
            r["v_um_per_min"],
            r["T_c_min"],
            s=65,
            c="#f43f5e",
            edgecolors="white",
            zorder=5,
        )
    for r in p3["sweep"]:
        ax.scatter(r["v_front_fit_um_per_min"], 45.0, s=80, marker="D", c="#fbbf24", edgecolors="black", zorder=6,
                   label="FGF8/RA-derived $v_{front}$ ($v_{tb}$ sweep)" if r is p3["sweep"][0] else None)
    ax.set_title("C. Morphospace $S(v, T_c) = v T_c$ ($\\mu$m)", fontweight="bold")
    ax.set_xlabel("Wavefront Velocity $v$ ($\\mu$m / min)")
    ax.set_ylabel("Clock Period $T_c$ (min)")
    ax.legend(loc="upper right", fontsize=8.0, framealpha=0.88, facecolor="#0f172a")
    fig.colorbar(cs, ax=ax, fraction=0.045, pad=0.03, label="Somite length $S$ ($\\mu$m)")

    # Panel D: Cooke (1975) Size-Invariance Scaling vs Unscaled Wavefront Control
    ax = axes[1, 0]
    L_sweep = np.linspace(1.6, 4.4, 50)
    N_scaled = np.full_like(L_sweep, 40.0)
    N_unscaled = (L_sweep * 1000.0) / (2.2222222222222223 * 45.0)
    ax.plot(L_sweep, N_scaled, color="#38bdf8", lw=2.3, label="Proportional Wavefront $v \\propto L$ ($N = 40$ invariant)")
    ax.plot(L_sweep, N_unscaled, color="#f97316", lw=2.0, ls="--", label="Unscaled Wavefront $v = v_0$ ($N \\propto L$ failure)")
    for r in regimes[:3]:
        ax.errorbar(
            r["L_mm"],
            r["N_obs"],
            yerr=r["N_obs_sd"],
            fmt="o",
            color="#34d399",
            ecolor="#a7f3d0",
            capsize=4,
            markersize=7,
            label="Cooke (1975) Xenopus empirical" if r["regime_id"] == "normal_diploid_control" else None,
        )
    ax.scatter([4.0, 2.4], [sf["n_somites_sim"], p3["sim_fgf_mini"]["n_somites_sim"]], marker="D", s=70, c="#fbbf24",
               edgecolors="black", zorder=6, label="FGF8/RA endogenous front ($v_{tb} \\propto L$)")
    ax.set_title("D. Cooke (1975) Xenopus Embryo Size-Invariance Scaling", fontweight="bold")
    ax.set_xlabel("Embryo Axial Length $L$ (mm)")
    ax.set_ylabel("Total Somite Count $N$")
    ax.set_ylim(12, 52)
    ax.legend(loc="lower right", fontsize=8.2, framealpha=0.88, facecolor="#0f172a")

    # Panel E: Biological Specimen Segmentation (`somite_embryo_specimen.jpg`)
    ax = axes[1, 1]
    spec_im = np.asarray(Image.open(SPECIMEN_JPG).convert("RGB"))
    ax.imshow(spec_im)
    x0, x1 = spec["ap_roi_x_px"]
    ax.axhline(230, xmin=x0 / 736.0, xmax=x1 / 736.0, color="#38bdf8", lw=1.2, ls=":")
    ax.axhline(334, xmin=x0 / 736.0, xmax=x1 / 736.0, color="#34d399", lw=1.2, ls=":")
    ax.scatter(
        spec["upper_row"]["peaks_x"],
        [230] * len(spec["upper_row"]["peaks_x"]),
        c="#38bdf8",
        s=42,
        edgecolors="black",
        label=f"Upper row ({spec['upper_row']['n_somites']} somites, $\\bar{{S}}={spec['upper_row']['mean_spacing_px']:.1f}$ px)",
    )
    ax.scatter(
        spec["lower_row"]["peaks_x"],
        [334] * len(spec["lower_row"]["peaks_x"]),
        c="#34d399",
        s=42,
        edgecolors="black",
        label=f"Lower row ({spec['lower_row']['n_somites']} somites, $\\bar{{S}}={spec['lower_row']['mean_spacing_px']:.1f}$ px)",
    )
    for cx in spec["lower_row"]["clefts_x"]:
        ax.plot([cx, cx], [312, 356], color="#facc15", lw=1.0, alpha=0.9)
    ax.set_title(f"E. Vertebrate Embryo SEM Somite Segmentation (CC BY 4.0); cleft angle {spec['lower_row_cleft_angle_mean_deg']:.1f}°", fontweight="bold", fontsize=9.5)
    ax.set_xlabel("Anterior-Posterior Pixel Coordinate $x$ (px)")
    ax.set_ylabel("Dorsoventral Coordinate $y$ (px)")
    ax.legend(loc="lower left", fontsize=8.0, framealpha=0.88, facecolor="#0f172a")

    # Panel F: Empirical Somite Profile vs Phase-Aligned Cooke-Zeeman Wave
    ax = axes[1, 2]
    x_roi = np.arange(x0, x1, dtype=float)
    emp_sig = np.array(spec["lower_row"]["signal"], dtype=float)
    emp_norm = emp_sig / (np.std(emp_sig) + 1e-12)
    k_emp = 2.0 * math.pi / spec["lower_row"]["mean_spacing_px"]
    x_rel = x_roi - x0
    phi_fit = math.atan2(
        float(np.dot(emp_norm, np.sin(k_emp * x_rel))),
        float(np.dot(emp_norm, np.cos(k_emp * x_rel))),
    )
    mod_wave = np.tanh(1.8 * np.cos(k_emp * x_rel - phi_fit))
    mod_norm = mod_wave / (np.std(mod_wave) + 1e-12)
    ax.plot(x_roi, emp_norm, color="#34d399", lw=1.8, label="Empirical SEM somite band-pass profile")
    ax.plot(
        x_roi,
        mod_norm,
        color="#38bdf8",
        lw=1.6,
        ls="--",
        label=f"Cooke-Zeeman cusp wave ($r={spec['profile_phase_correlation_r']:.3f}$)",
    )
    ax.set_title(
        f"F. Specimen vs Model Periodicity ($\\bar{{S}}={spec['bilateral_mean_spacing_px']:.2f}$ px, CV={spec['bilateral_cv_spacing']:.3f})",
        fontweight="bold",
    )
    ax.set_xlabel("Anterior-Posterior Coordinate $x$ (px)")
    ax.set_ylabel("Normalized Somite Elevation $z$-score")
    ax.legend(loc="upper right", fontsize=8.2, framealpha=0.88, facecolor="#0f172a")

    # Panel G: Goldbeter (2007) FGF/RA bistability diagram
    ax = axes[2, 0]
    Mg = scan["M_grid"]
    ax.plot(Mg, scan["F_high"], color="#f87171", lw=2.2, label="FGF-high (tailbud / posterior PSM) stable branch")
    ax.plot(Mg, scan["F_low"], color="#60a5fa", lw=2.2, label="RA-high (determined) stable branch")
    ax.plot(Mg, scan["F_mid"], color="#e2e8f0", lw=1.3, ls="--", label="unstable saddle branch")
    M_hi_plot = scan["M_hi"] if np.isfinite(scan["M_hi"]) else Mg[-1]
    ax.axvspan(scan["M_lo"], M_hi_plot, color="#fbbf24", alpha=0.12, label=f"bistable window $M_F \\in [{scan['M_lo']:.3f}, {M_hi_plot:.2f}{'+' if not np.isfinite(scan['M_hi']) else ''}]$")
    ax.axvline(1.0, color="#34d399", lw=1.2, ls=":", label="tailbud $M_F = 1$")
    ax.annotate("determination\nfront $M_F = M_{lo}$", xy=(scan["M_lo"], 0.2 + float(np.nanmax(scan["F_high"])) * 0.1), xytext=(scan["M_lo"] + 0.25, float(np.nanmax(scan["F_high"])) * 0.55),
                arrowprops=dict(arrowstyle="->", color="#fbbf24"), color="#fbbf24", fontsize=8.5)
    ax.set_title("G. Goldbeter (2007) FGF8/RA Toggle: $F^*$ vs fgf8 mRNA $M_F$", fontweight="bold", fontsize=9.2)
    ax.set_xlabel("fgf8 mRNA $M_F$ (tailbud = 1; decays as $e^{-(x_{tb}-x)/\\lambda_F}$)")
    ax.set_ylabel("Steady-state FGF8 level $F^*$")
    ax.legend(loc="upper left", fontsize=7.6, framealpha=0.88, facecolor="#0f172a")

    # Panel H: Endogenous front — FGF/RA profiles and x_w(t) vs tailbud
    ax = axes[2, 1]
    snaps = [10, 22, 34]
    cols = ["#fbbf24", "#f472b6", "#34d399"]
    for s_i, c in zip(snaps, cols):
        ax.plot(sf["x_mm"], sf["F_kymograph"][s_i] / FGF_RA_PARAMS["alpha_F"], color=c, lw=1.5, label=f"FGF8 $F/\\alpha_F$, t={sf['t_snaps_min'][s_i]:.0f} min")
        ax.plot(sf["x_mm"], sf["R_kymograph"][s_i] / FGF_RA_PARAMS["alpha_R"], color=c, lw=1.5, ls="--")
        ax.plot(sf["x_mm"], sf["M_kymograph"][s_i], color=c, lw=0.9, ls=":")
        ax.axvline(sf["xw_snaps_mm"][s_i], color=c, lw=0.9, alpha=0.7)
        ax.axvline(sf["xtb_snaps_mm"][s_i], color=c, lw=0.9, alpha=0.4, ls="-.")
    ax.plot([], [], color="#e2e8f0", ls="--", label="RA $R/\\alpha_R$")
    ax.plot([], [], color="#e2e8f0", ls=":", label="fgf8 mRNA $M_F$")
    ax.plot([], [], color="#e2e8f0", ls="-.", alpha=0.5, label="tailbud $x_{tb}(t)$")
    ax.set_xlim(0.0, 4.0)
    ax.set_ylim(-0.05, 1.15)
    ax.set_title(f"H. Endogenous Front: $v_{{front}}={sf['v_front_fit_um_per_min']:.3f}$ vs $v_{{tb}}={sf['v_tailbud_um_per_min']:.3f}\\,\\mu$m/min", fontweight="bold", fontsize=9.2)
    ax.set_xlabel("Anteroposterior Axis $x$ (mm)")
    ax.set_ylabel("Normalised level")
    ax.legend(loc="center left", fontsize=7.4, framealpha=0.88, facecolor="#0f172a")
    ins = ax.inset_axes([0.63, 0.55, 0.35, 0.40])
    ins.set_facecolor("#0f172a")
    vt = [r["v_tb_um_per_min"] for r in p3["sweep"]]
    ins.plot([1.0, 3.6], [45.0, 162.0], color="#94a3b8", lw=1.0, ls="--")
    ins.scatter(vt, [r["S_sim_um"] for r in p3["sweep"]], c="#fbbf24", edgecolors="black", s=36, zorder=5)
    ins.set_xlabel("$v_{tb}$ ($\\mu$m/min)", fontsize=7)
    ins.set_ylabel("$S_{sim}$ ($\\mu$m)", fontsize=7)
    ins.tick_params(labelsize=6.5)
    ins.set_title(f"S vs $v_{{tb}}T_c$: RMSE {p3['sweep_S_rmse_um']:.2f} $\\mu$m", fontsize=7, color="#f8fafc")

    # Panel I: Vertex-model sheet
    ax = axes[2, 2]
    _draw_vertex_sheet(
        ax, sheet, sf,
        f"I. Vertex-Model PSM Sheet (0.45-0.8 mm): cleft {sheet['cleft_angle_lattice_deg']:.1f}°→{sheet['cleft_angle_final_deg']:.1f}° (ctrl {sheet_ctrl['cleft_angle_final_deg']:.1f}°)",
        x_window_um=(450.0, 800.0),
    )
    ax.set_aspect("auto")
    ax.title.set_fontsize(8.6)
    ins2 = ax.inset_axes([0.66, 0.60, 0.33, 0.37])
    ins2.set_facecolor("#0f172a")
    hist = sheet["stage_history"]
    hist_c = sheet_ctrl["stage_history"]
    ins2.plot([h["t_min"] for h in hist], [h["cleft_angle_deg"] for h in hist], color="#facc15", lw=1.5, label="$\\Delta\\Lambda$ clefts")
    ins2.plot([h["t_min"] for h in hist_c], [h["cleft_angle_deg"] for h in hist_c], color="#94a3b8", lw=1.2, ls="--", label="$\\Delta\\Lambda=0$")
    ins2.axhline(spec["lower_row_cleft_angle_mean_deg"], color="#34d399", lw=1.0, ls=":", label="SEM clefts")
    ins2.set_xlabel("t (min)", fontsize=7)
    ins2.set_ylabel("cleft angle (°)", fontsize=7)
    ins2.tick_params(labelsize=6.5)
    ins2.legend(fontsize=6, facecolor="#0f172a", framealpha=0.8, loc="upper right")
    ax.text(0.01, 0.02, f"red = anterior half-somite ($\\chi>0$), blue = posterior ($\\chi<0$), yellow = cleft\ntortuosity {sheet['cleft_tortuosity_lattice']:.2f}→{sheet['cleft_tortuosity_final']:.2f}; $S_{{2D}}$={sheet['S_2d_um']:.0f} µm; shape index $p$={sheet['shape_index_mean']:.2f}; full sheet: clock_and_wavefront_vertex_sheet.png",
            transform=ax.transAxes, fontsize=6.8, color="#f8fafc",
            bbox=dict(facecolor="#0f172a", alpha=0.75, edgecolor="none"))

    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.965])
    fig.savefig(out_png, dpi=140)
    plt.close(fig)
    return out_png


def render_vertex_sheet_figure(p3: dict[str, Any], out_png: Path = OUTPUT_SHEET_PNG) -> Path:
    """Render a dedicated two-panel vertex-sheet figure (differential cleft tension vs
    dLambda = 0 control) at a legible scale."""
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(16.0, 6.4), dpi=130)
    fig.patch.set_facecolor("#0b0f19")
    for a in axes:
        a.set_facecolor("#131b2e")
    _draw_vertex_sheet(axes[0], p3["sheet"], p3["sim_fgf"],
                       f"Vertex-model paraxial mesoderm sheet with differential cleft tension ΔΛ={p3['sheet']['delta_Lambda_cleft']}: cleft angle {p3['sheet']['cleft_angle_lattice_deg']:.1f}°→{p3['sheet']['cleft_angle_final_deg']:.1f}°, tortuosity {p3['sheet']['cleft_tortuosity_lattice']:.3f}→{p3['sheet']['cleft_tortuosity_final']:.3f}")
    _draw_vertex_sheet(axes[1], p3["sheet_ctrl"], p3["sim_fgf"],
                       f"Control ΔΛ=0 (no differential tension): cleft angle {p3['sheet_ctrl']['cleft_angle_final_deg']:.1f}°, tortuosity {p3['sheet_ctrl']['cleft_tortuosity_final']:.3f}")
    fig.tight_layout()
    fig.savefig(out_png, dpi=130)
    plt.close(fig)
    return out_png


def render_growth_gif(out_gif: Path = OUTPUT_GIF) -> Path:
    """Render an animated GIF (`docs/media/clock_and_wavefront_growth.gif`) showing the
    tailbud elongating, the fgf8 mRNA / FGF8 / RA profiles of the Goldbeter toggle, the
    emergent determination front and the cusp state chi budding 40 somites sequentially
    from anterior to posterior (coupled `wavefront_mode="fgf_ra"` model).
    """
    out_gif.parent.mkdir(parents=True, exist_ok=True)
    sim = simulate_cusp_clock_wavefront(L_mm=4.0, T_c_min=45.0, T_total_min=1800.0, wavefront_mode="fgf_ra")
    x = sim["x_mm"]
    kymo = sim["chi_kymograph"]
    t_snaps = sim["t_snaps_min"]
    xw_snaps = sim["xw_snaps_mm"]

    frames_pil: list[Image.Image] = []
    step_indices = np.linspace(0, len(t_snaps) - 1, 24, dtype=int)
    for idx in step_indices:
        fig, (ax0, ax1, ax2) = plt.subplots(3, 1, figsize=(9.2, 6.6), dpi=100, gridspec_kw={"height_ratios": [1.0, 1.4, 0.7]})
        fig.patch.set_facecolor("#0b0f19")
        for a in (ax0, ax1, ax2):
            a.set_facecolor("#131b2e")
            for sp in a.spines.values():
                sp.set_color("#3b4c6b")
            a.tick_params(colors="#cbd5e1")

        chi_t = kymo[idx]
        xw_t = xw_snaps[idx]
        t_min = t_snaps[idx]
        n_formed = int(np.sum((chi_t[1:] > 0) & (chi_t[:-1] <= 0) & (x[1:] <= xw_t)))

        ax0.plot(x, sim["M_kymograph"][idx], color="#94a3b8", lw=1.2, ls=":", label="fgf8 mRNA $M_F$")
        ax0.plot(x, sim["F_kymograph"][idx] / FGF_RA_PARAMS["alpha_F"], color="#f87171", lw=1.7, label="FGF8 $F$")
        ax0.plot(x, sim["R_kymograph"][idx] / FGF_RA_PARAMS["alpha_R"], color="#60a5fa", lw=1.7, label="RA $R$")
        ax0.axvline(min(sim["xtb_snaps_mm"][idx], 4.0), color="#34d399", lw=1.6, ls="-.", label=f"tailbud $x_{{tb}}={sim['xtb_snaps_mm'][idx]:.2f}$ mm")
        ax0.axvline(xw_t, color="#facc15", lw=2.0, ls="--")
        ax0.set_xlim(0.0, 4.0)
        ax0.set_ylim(-0.05, 1.1)
        ax0.set_ylabel("Goldbeter toggle", color="#e2e8f0", fontsize=9)
        ax0.set_title(
            f"Cooke-Zeeman (1976) clock + Goldbeter (2007) FGF8/RA front — t = {t_min:6.1f} min ({n_formed}/40 somites)",
            color="#f8fafc",
            fontweight="bold",
            fontsize=10.5,
        )
        ax0.legend(loc="upper left", fontsize=7.5, facecolor="#0f172a", labelcolor="#f8fafc", ncol=4)

        ax1.plot(x, chi_t, color="#38bdf8", lw=1.7, label="Morphogenetic state $\\chi(x,t)$")
        ax1.axvline(xw_t, color="#facc15", lw=2.0, ls="--", label=f"Emergent front $x_w(t) = {xw_t:.2f}$ mm")
        ax1.fill_between(x, -1.3, 1.3, where=(x <= xw_t), color="#38bdf8", alpha=0.10)
        ax1.set_xlim(0.0, 4.0)
        ax1.set_ylim(-1.35, 1.35)
        ax1.set_ylabel("State $\\chi(x,t)$", color="#e2e8f0")
        ax1.legend(loc="upper right", fontsize=8.5, facecolor="#0f172a", labelcolor="#f8fafc")

        # 2D dorsal embryo somite strip visualization
        ribbon = np.tile(chi_t, (28, 1))
        ax2.imshow(
            ribbon,
            aspect="auto",
            extent=[0.0, 4.0, 0.0, 1.0],
            cmap="coolwarm",
            vmin=-1.1,
            vmax=1.1,
        )
        ax2.axvline(xw_t, color="#facc15", lw=2.0, ls="--")
        ax2.set_xlabel("Anteroposterior Embryo Axis $x$ (mm) [Anterior -> Posterior]", color="#e2e8f0")
        ax2.set_yticks([])
        ax2.set_ylabel("Somite Row", color="#e2e8f0")

        fig.tight_layout()
        fig.canvas.draw()
        rgba = np.asarray(fig.canvas.buffer_rgba())
        frames_pil.append(Image.fromarray(rgba[..., :3]).convert("P", palette=Image.Palette.ADAPTIVE))
        plt.close(fig)

    frames_pil[0].save(
        out_gif,
        save_all=True,
        append_images=frames_pil[1:],
        duration=110,
        loop=0,
        optimize=True,
    )
    return out_gif


# ============================================================================
# Section 8: Deterministic Self-Test Suite (`--self-test`)
# ============================================================================
def run_self_test() -> int:
    """Run deterministic mathematical, morphometric, and reproducibility assertions."""
    print("[1976-cooke-clock-and-wavefront] Running --self-test suite...")
    digests = verify_dataset_integrity()
    assert len(digests) >= 3, "Expected >=3 SHA-256-verified dataset files"

    # 1. Analytical wavelength and size-invariance laws
    s_norm = analytical_somite_wavelength(2.2222222222222223, 45.0)
    s_mini = analytical_somite_wavelength(1.3333333333333333, 45.0)
    n_norm = analytical_somite_count(4.0, 2.2222222222222223, 45.0)
    n_mini = analytical_somite_count(2.4, 1.3333333333333333, 45.0)
    assert abs(s_norm - 100.0) < 1e-6, f"Expected S_norm=100 um, got {s_norm}"
    assert abs(s_mini - 60.0) < 1e-6, f"Expected S_mini=60 um, got {s_mini}"
    assert abs(n_norm - 40.0) < 1e-6 and abs(n_mini - 40.0) < 1e-6

    # 2. Spatiotemporal cusp catastrophe solver accuracy & bitwise determinism
    sim1 = simulate_cusp_clock_wavefront(L_mm=4.0, T_c_min=45.0, T_total_min=1800.0, seed=1976)
    sim2 = simulate_cusp_clock_wavefront(L_mm=4.0, T_c_min=45.0, T_total_min=1800.0, seed=1976)
    assert np.allclose(sim1["chi_final"], sim2["chi_final"], atol=0.0), "Bitwise determinism failed"
    assert sim1["wavelength_rel_error_pct"] < 3.0, (
        f"Expected <3% wavelength error vs S=v*T_c, got {sim1['wavelength_rel_error_pct']}%"
    )
    assert 39 <= sim1["n_somites_sim"] <= 41, f"Expected ~40 somites, got {sim1['n_somites_sim']}"

    # 3. Miniature embryo size invariance & clock-arrested negative control
    sim_mini = simulate_cusp_clock_wavefront(L_mm=2.4, T_c_min=45.0, T_total_min=1800.0, seed=1976)
    sim_null = simulate_cusp_clock_wavefront(
        L_mm=4.0, T_c_min=45.0, T_total_min=1800.0, clock_enabled=False, seed=1976
    )
    assert sim_mini["n_somites_sim"] == sim1["n_somites_sim"], (
        f"Size invariance violated: N_mini={sim_mini['n_somites_sim']} vs N_norm={sim1['n_somites_sim']}"
    )
    assert sim_null["n_peaks_detected"] == 0, (
        f"Clock-arrested negative control must form 0 somites, got {sim_null['n_peaks_detected']}"
    )

    # 4. Goldbeter (2007) FGF/RA toggle: bistability, endogenous front velocity, S = v_tb*T_c
    scan = fgf_ra_bistability_scan()
    assert np.isfinite(scan["M_lo"]) and 0.0 < scan["M_lo"] < 1.0, f"Toggle must be bistable below the tailbud level, M_lo={scan['M_lo']}"
    assert scan["tailbud_is_FGF_high_or_bistable"], "Tailbud (M_F = 1) must support the FGF-high state"
    sim_f1 = simulate_cusp_clock_wavefront(L_mm=4.0, T_c_min=45.0, T_total_min=1800.0, wavefront_mode="fgf_ra", seed=1976)
    sim_f2 = simulate_cusp_clock_wavefront(L_mm=4.0, T_c_min=45.0, T_total_min=1800.0, wavefront_mode="fgf_ra", seed=1976)
    assert np.allclose(sim_f1["chi_final"], sim_f2["chi_final"], atol=0.0), "FGF/RA-coupled determinism failed"
    assert sim_f1["v_front_rel_error_pct"] < 2.0, (
        f"Endogenous front velocity must track tailbud elongation within 2%, got {sim_f1['v_front_rel_error_pct']}%"
    )
    assert sim_f1["v_front_fit_r2"] > 0.999, f"Front trajectory must be linear (R^2 > 0.999), got {sim_f1['v_front_fit_r2']}"
    assert sim_f1["wavelength_rel_error_pct"] < 3.0, (
        f"FGF/RA-driven somite wavelength must match v_tb*T_c within 3%, got {sim_f1['wavelength_rel_error_pct']}%"
    )
    assert 39 <= sim_f1["n_somites_sim"] <= 41, f"Expected ~40 somites with endogenous front, got {sim_f1['n_somites_sim']}"
    sim_f_fast = simulate_cusp_clock_wavefront(L_mm=4.0, T_c_min=45.0, T_total_min=1800.0, wavefront_mode="fgf_ra", v_tailbud_um_per_min=3.3333)
    assert abs(sim_f_fast["mean_S_sim_um"] - 150.0) / 150.0 < 0.03, (
        f"1.5x tailbud elongation must give S ~ 150 um with no wavefront parameter changed, got {sim_f_fast['mean_S_sim_um']}"
    )
    sim_bead = simulate_cusp_clock_wavefront(
        L_mm=4.0, T_c_min=45.0, T_total_min=1800.0, wavefront_mode="fgf_ra",
        fgf_bead={"x_mm": 0.6, "amplitude": 1.0, "width_mm": 0.06, "tau_min": 300.0},
    )
    pk = sim_bead["x_mm"][sim_bead["somite_peaks_idx"]]
    mids = 0.5 * (pk[1:] + pk[:-1])
    local = sim_bead["spacings_um"][np.abs(mids - 0.6) <= 0.3]
    assert len(local) >= 2 and float(local.min()) < 0.9 * sim_f1["mean_S_sim_um"], (
        f"FGF8 bead must produce smaller adjacent somites (Dubrulle 2001), got local spacings {local}"
    )

    # 5. 2D vertex-model sheet: cleft rotation toward the ML axis and straightening
    sheet = simulate_vertex_sheet_segmentation(sim_f1)
    sheet_ctrl = simulate_vertex_sheet_segmentation(sim_f1, delta_lambda=0.0)
    assert sheet["n_clefts"] >= 9, f"Expected >=9 clefts on the 1.2 mm sheet, got {sheet['n_clefts']}"
    assert abs(sheet["n_somite_blocks_2d"] - sheet["N_sheet_analytical"]) <= 1.5, (
        f"2D block count {sheet['n_somite_blocks_2d']} must match analytical {sheet['N_sheet_analytical']}"
    )
    assert sheet["cleft_angle_final_deg"] < sheet["cleft_angle_lattice_deg"] - 5.0, (
        f"Differential tension must rotate clefts toward the ML axis: {sheet['cleft_angle_lattice_deg']} -> {sheet['cleft_angle_final_deg']} deg"
    )
    assert sheet["cleft_angle_final_deg"] < sheet_ctrl["cleft_angle_final_deg"] - 5.0, (
        f"dLambda=0 control must not rotate clefts: treated {sheet['cleft_angle_final_deg']} vs control {sheet_ctrl['cleft_angle_final_deg']} deg"
    )
    assert sheet["cleft_tortuosity_final"] < sheet["cleft_tortuosity_lattice"], "Clefts must straighten under differential tension"
    assert abs(sheet["S_2d_um"] - 100.0) / 100.0 < 0.06, f"2D cleft pitch must match S=100 um within 6%, got {sheet['S_2d_um']}"

    # 6. Specimen computer-vision segmentation & measure_morphoeval()
    spec = segment_vertebrate_somite_specimen()
    assert spec["upper_row"]["n_somites"] >= 7 and spec["lower_row"]["n_somites"] >= 8
    assert 40.0 <= spec["bilateral_mean_spacing_px"] <= 56.0
    assert len(spec["lower_row_cleft_angles_deg"]) >= 5 and 0.0 <= spec["lower_row_cleft_angle_mean_deg"] <= 45.0
    meas = measure_morphoeval()
    assert len(meas) >= 13, f"Expected >=13 MorphoEval measurements, got {len(meas)}"
    assert all(("reference_value" in m and "unit" in m) for m in meas)

    print(
        f"[1976-cooke-clock-and-wavefront] SELF-TEST PASSED: "
        f"S_sim={sim1['mean_S_sim_um']:.2f} um (err={sim1['wavelength_rel_error_pct']:.2f}%), "
        f"N_norm={sim1['n_somites_sim']}, N_mini={sim_mini['n_somites_sim']}, "
        f"N_null={sim_null['n_peaks_detected']}, specimen S_obs={spec['bilateral_mean_spacing_px']:.2f} px; "
        f"FGF/RA front v={sim_f1['v_front_fit_um_per_min']:.4f} um/min (err={sim_f1['v_front_rel_error_pct']:.2f}%), "
        f"S_fgf={sim_f1['mean_S_sim_um']:.2f} um, bead min ratio={float(local.min())/sim_f1['mean_S_sim_um']:.3f}; "
        f"vertex sheet clefts={sheet['n_clefts']}, angle {sheet['cleft_angle_lattice_deg']:.1f}->{sheet['cleft_angle_final_deg']:.1f} deg "
        f"(ctrl {sheet_ctrl['cleft_angle_final_deg']:.1f}), tortuosity {sheet['cleft_tortuosity_lattice']:.3f}->{sheet['cleft_tortuosity_final']:.3f}; "
        f"{len(meas)} measurements."
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Cooke & Zeeman (1976) Clock-and-Wavefront Somite Segmentation Model"
    )
    ap.add_argument("--self-test", action="store_true", help="Run deterministic self-test suite and exit")
    ap.add_argument("--L-mm", type=float, default=4.0, help="Embryo axial length L in mm (default: 4.0)")
    ap.add_argument("--Tc-min", type=float, default=45.0, help="Clock period T_c in min (default: 45.0)")
    ap.add_argument("--T-total-min", type=float, default=1800.0, help="Total segmentation time in min (default: 1800.0)")
    ap.add_argument("--v-tailbud", type=float, default=None, help="Tailbud elongation speed v_tb (um/min) for the FGF/RA mode (default: L/T_total)")
    ap.add_argument("--wavefront-mode", choices=["kinematic", "fgf_ra"], default="fgf_ra", help="Wavefront mechanism for --summary runs (default: fgf_ra)")
    ap.add_argument("--seed", type=int, default=1976, help="Deterministic RNG seed (default: 1976)")
    ap.add_argument("--skip-gif", action="store_true", help="Skip rendering animated GIF")
    ap.add_argument("--summary", action="store_true", help="Print a JSON summary of one run with the given parameters and exit (no rendering)")
    args = ap.parse_args(argv)

    if args.self_test:
        return run_self_test()

    verify_dataset_integrity()
    if args.summary:
        sim = simulate_cusp_clock_wavefront(
            L_mm=args.L_mm, T_c_min=args.Tc_min, T_total_min=args.T_total_min, seed=args.seed,
            wavefront_mode=args.wavefront_mode, v_tailbud_um_per_min=args.v_tailbud,
        )
        keys = ["wavefront_mode", "v_um_per_min", "mean_S_sim_um", "cv_S_sim", "n_somites_sim", "S_analytical_um",
                "N_analytical", "wavelength_rel_error_pct", "v_front_fit_um_per_min", "v_front_fit_r2",
                "front_offset_mm", "lambda_F_mm", "M_lo", "front_width_10_90_um"]
        print(json.dumps({k: sim[k] for k in keys if k in sim}, indent=2))
        return 0
    png_path = render_comparison_figure(OUTPUT_PNG)
    print(f"[1976-cooke-clock-and-wavefront] Rendered figure -> {png_path.relative_to(ROOT)}")
    sheet_png = render_vertex_sheet_figure(run_pass3_models(), OUTPUT_SHEET_PNG)
    print(f"[1976-cooke-clock-and-wavefront] Rendered vertex sheet -> {sheet_png.relative_to(ROOT)}")
    if not args.skip_gif:
        gif_path = render_growth_gif(OUTPUT_GIF)
        print(f"[1976-cooke-clock-and-wavefront] Rendered animation -> {gif_path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

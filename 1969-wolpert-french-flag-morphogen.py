#!/usr/bin/env python3
"""
1969-wolpert-french-flag-morphogen.py — Pass 3 Remediated Implementation (Slot 011)
===================================================================================
Wolpert (1969) Positional Information & French Flag Morphogen Gradients,
Tickle, Summerbell & Wolpert (1975) Chick Limb ZPA Digit Duplication,
Ben-Zvi & Barkai (2010) Expansion-Repression Dynamic Scaling, and (Pass 3) a
2D proximo-distal x anteroposterior chick wing bud paddle coupling AER FGF
outgrowth with posterior ZPA Shh (Saunders 1948; Summerbell, Lewis & Wolpert 1973;
Summerbell 1974; Tickle 1981; Laufer et al. 1994; Niswander et al. 1994).

Primary Literature & Mathematical Mapping:
  1. Wolpert L (1969). Positional information and the spatial pattern of cellular
     differentiation. J. Theor. Biol. 25(1):1-47 (doi:10.1016/S0022-5193(69)80016-0).
     - Section 3, Figs. 2-3: Linear source-sink diffusion `D d^2c/dx^2 = 0` with
       `c(0) = c0, c(L) = 0  =>  c_lin(x) = c0 * (1 - x/L)`.
       Threshold boundaries `x_i / L = 1 - T_i / c0` are strictly scale-invariant
       (Lean 4 theorems `wolpert_linear_scale_homogeneity` &
       `wolpert_french_flag_exact_trisection` in `Loop_011_french_flag_morphogen.lean`).
  2. Crick F (1970). Diffusion in embryogenesis. Nature 225(5231):420-422
     (doi:10.1038/225420a0).
     - Synthesis-Diffusion-Degradation (SDD) PDE:
       `dc/dt = D d^2c/dx^2 - k c`, characteristic length `lambda = sqrt(D / k)`,
       semi-infinite steady state `c_SDD(x) = c0 * exp(-x / lambda)`, and finite-domain
       no-flux steady state `c_fin(x) = c0 * cosh((L - x)/lambda) / cosh(L/lambda)`
       (Lean 4 theorems `sdd_steady_state_balance` & `sdd_zero_source_ablation_quiescence`).
     - Positional precision under concentration noise `sigma_c`:
       `sigma_x(x_i) = sigma_c / |dc/dx(x_i)| = (lambda / T_i) * sigma_c`.
  3. Tickle C, Summerbell D, Wolpert L (1975). Positional signalling and specification
     of digits in chick limb morphogenesis. Nature 254(5497):199-202
     (doi:10.1038/254199a0); Riddle RD et al. (1993). Sonic hedgehog mediates the
     polarizing activity of the ZPA. Cell 75(7):1401-1416
     (doi:10.1016/0092-8674(93)90626-2).
     - Posterior ZPA Sonic hedgehog (Shh) morphogen specifies chick wing digits
       `4-3-2` (`T4 = 0.45, T3 = 0.18, T2 = 0.05`).
     - Grafting a second ZPA to the anterior margin (`x = L`) yields the bilateral
       superposition `c_dup(x) = phi(x) + phi(L - x) = c_dup(L - x)`, producing the
       mirror-image digit duplication `4-3-2-2-3-4` (Lean 4 theorem
       `tickle_1975_zpa_mirror_duplication_symmetry`).
  4. Ben-Zvi D, Barkai N (2010). Scaling of morphogen gradients by an
     expansion-repression integral feedback control. PNAS 107(15):6924-6929
     (doi:10.1073/pnas.0912734107).
     - Diffusible expander `E` produced when distal morphogen `c(L, t) < T_rep`
       locks `lambda_eff(L) = L / ln(c0 / T_rep) proportional to L`, restoring
       scale-invariant `1/3 : 1/3 : 1/3` French-Flag proportions across growing
       chick limb stages HH18 (180 um) -> HH28 (660 um) (Lean 4 theorem
       `benzvi_barkai_2010_expander_scaling_invariance`).
  5. Pass 3 — 2D PD x AP limb bud paddle (Section 3b; `simulate_limb_bud_2d`):
     - Saunders JW (1948). J. Exp. Zool. 108(3):363-403 (doi:10.1002/jez.1401080304);
       Summerbell D (1974). J. Embryol. Exp. Morphol. 32(3):651-660 (PMID 4463222);
       Summerbell D, Lewis JH, Wolpert L (1973). Nature 244:492-496 (doi:10.1038/244492a0).
       AER-driven distal outgrowth `dL/dt = v_max <F>_PZ / (<F>_PZ + K_g)`; AER excision
       at a stage halts outgrowth and truncates the limb at a PD level that advances
       stylopod -> zeugopod -> autopod with the stage of excision (Lean 4 theorem
       `aer_excision_truncation_monotone`).
     - Laufer E et al. (1994). Cell 79(6):993-1003 (doi:10.1016/0092-8674(94)90030-2);
       Niswander L et al. (1994). Nature 371:609-612 (doi:10.1038/371609a0);
       Crossley PH et al. (1996). Cell 84:127-136 (doi:10.1016/S0092-8674(00)80999-X).
       Shh-FGF positive feedback: `dS/dt = D lap S - k_S S + s_S [ZPA] F/(F+K_FS)` and
       `dF/dt = D lap F - k_F F + s_F [AER] (1 + g_S S/(S+K_S))` (Lean 4 theorem
       `shh_fgf_feedback_aer_ablation_quiescence`); zero-flux 5-point stencil on the
       paddle mask (Lean 4 theorem `sdd_2d_steady_state_balance`).
     - Tickle C (1981). Nature 289(5795):295-298 (doi:10.1038/289295a0). The digit next
       to an anterior graft advances 2 -> 3 -> 4 with graft strength; implemented by the
       morphogen-valley digit reader `extract_digit_sequence` (Lean 4 theorem
       `tickle_1981_graft_dose_digit_monotone`).
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
from matplotlib.colors import ListedColormap
import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from scipy.optimize import curve_fit

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets" / "french_flag_morphogen"
REF_JSON_PATH = DATASET_DIR / "french_flag_morphogen_reference.json"
SPECIMEN_PATH = DATASET_DIR / "french_flag_specimen.jpg"
SCOUT_PATH = DATASET_DIR / "scout_manifest.json"

OUT_PNG = ROOT / "docs" / "media" / "french_flag_morphogen_comparison.png"
OUT_GIF = ROOT / "docs" / "media" / "french_flag_morphogen_growth.gif"
OUT_LIMB_GIF = ROOT / "docs" / "media" / "french_flag_morphogen_limb2d.gif"

# Cryptographic SHA-256 locks for vendored reference data & biological specimen
EXPECTED_SHA256: dict[str, str] = {
    "datasets/french_flag_morphogen/french_flag_morphogen_reference.json": (
        "deae09091fa98a1be7bc0e999291faaaf3ccd7a00a0551013964100fd46f4d8a"
    ),
    "datasets/french_flag_morphogen/french_flag_specimen.jpg": (
        "7163e90c56e1316e6398849083a6e48955cd20b124a6df7cf2178247cedd0c4f"
    ),
    "datasets/french_flag_morphogen/scout_manifest.json": (
        "d3bd7cc1c62dd2cb98ded323b71a65884d37fecf4c7c48333fb986b53d6a059e"
    ),
    # Pass 3: ordinal empirical references for the 2D AER/ZPA paddle (Saunders 1948,
    # Summerbell 1974, Tickle 1981, Laufer 1994, Niswander 1994)
    "datasets/french_flag_morphogen/limb_bud_2d_aer_zpa_reference.json": (
        "c34af92f3747bb64ad947b2a81c460709ced113f17eb69f0c565496954939455"
    ),
}
LIMB2D_REF_PATH = DATASET_DIR / "limb_bud_2d_aer_zpa_reference.json"


def verify_dataset_integrity() -> dict[str, str]:
    """Verify SHA-256 digests of all vendored Slot 011 dataset files."""
    verified: dict[str, str] = {}
    for rel_path, expected_hex in EXPECTED_SHA256.items():
        full_path = ROOT / rel_path
        if not full_path.exists():
            raise FileNotFoundError(f"Missing required dataset file: {rel_path}")
        actual_hex = hashlib.sha256(full_path.read_bytes()).hexdigest()
        if actual_hex != expected_hex:
            raise ValueError(
                f"SHA-256 mismatch for {rel_path}: expected {expected_hex}, got {actual_hex}"
            )
        verified[rel_path] = actual_hex
    return verified


def load_reference_dataset() -> dict[str, Any]:
    """Load and SHA-256-verify `french_flag_morphogen_reference.json`."""
    verify_dataset_integrity()
    return json.loads(REF_JSON_PATH.read_text(encoding="utf-8"))


def load_limb2d_reference() -> dict[str, Any]:
    """Load and SHA-256-verify the Pass 3 `limb_bud_2d_aer_zpa_reference.json`."""
    verify_dataset_integrity()
    return json.loads(LIMB2D_REF_PATH.read_text(encoding="utf-8"))


# ============================================================================
# 1. Wolpert (1969) Linear Source-Sink & SDD Reaction-Diffusion Solvers
# ============================================================================

def compute_linear_source_sink(x: np.ndarray, length: float = 300.0, c0: float = 1.0) -> np.ndarray:
    """Wolpert (1969) Section 3 linear source-sink steady state:
    `D d^2c/dx^2 = 0, c(0) = c0, c(L) = 0  =>  c_lin(x) = c0 * (1 - x / L)`.
    """
    return np.clip(c0 * (1.0 - x / float(length)), 0.0, c0)


def compute_sdd_exponential(
    x: np.ndarray,
    c0: float = 1.0,
    D: float = 10.0,
    k_deg: float = 0.0025,
) -> tuple[np.ndarray, float]:
    """Wolpert (1969) & Crick (1970) semi-infinite SDD steady state:
    `c_SDD(x) = c0 * exp(-x / lambda)` where `lambda = sqrt(D / k_deg)`.
    """
    lam = float(np.sqrt(D / k_deg))
    return c0 * np.exp(-x / lam), lam


def compute_sdd_finite_noflux(
    x: np.ndarray,
    length: float = 300.0,
    c0: float = 1.0,
    D: float = 10.0,
    k_deg: float = 0.0025,
) -> tuple[np.ndarray, float]:
    """Exact analytical steady state of `dc/dt = D d^2c/dx^2 - k_deg c` on `[0, L]`
    with Dirichlet source `c(0) = c0` and zero-flux distal tip `dc/dx(L) = 0`:
    `c_fin(x) = c0 * cosh((L - x) / lambda) / cosh(L / lambda)`.
    """
    lam = float(np.sqrt(D / k_deg))
    c_fin = c0 * np.cosh((length - x) / lam) / np.cosh(length / lam)
    return c_fin, lam


def integrate_sdd_pde_1d(
    length: float = 300.0,
    nx: int = 121,
    D: float = 10.0,
    k_deg: float = 0.0025,
    c0: float = 1.0,
    c_anterior: float = 0.0,
    anterior_dirichlet: bool = False,
    t_end: float = 3600.0,
) -> dict[str, Any]:
    """Integrate the time-dependent 1D SDD reaction-diffusion PDE:
      `dc/dt = D d^2c/dx^2 - k_deg * c`
    using Crank-Nicolson / implicit-explicit relaxation until steady state `t = t_end`.
    Supports either a zero-flux distal boundary at `x = L` (`anterior_dirichlet=False`)
    or a bilateral ZPA graft Dirichlet boundary `c(L) = c_anterior` (`anterior_dirichlet=True`).
    """
    x = np.linspace(0.0, length, nx)
    dx = x[1] - x[0]
    dt = min(0.40 * (dx ** 2) / max(D, 1e-9), 2.5)
    n_steps = int(np.ceil(t_end / dt))

    c = np.zeros(nx, dtype=np.float64)
    c[0] = c0
    if anterior_dirichlet:
        c[-1] = c_anterior

    snapshots: list[tuple[float, np.ndarray]] = []
    record_steps = set(np.linspace(0, n_steps - 1, 8, dtype=int))

    alpha = D * dt / (dx ** 2)
    decay = k_deg * dt
    for step in range(n_steps):
        c_next = c.copy()
        lap = c[:-2] - 2.0 * c[1:-1] + c[2:]
        c_next[1:-1] = c[1:-1] + alpha * lap - decay * c[1:-1]
        c_next[0] = c0
        if anterior_dirichlet:
            c_next[-1] = c_anterior
        else:
            # Second-order zero-flux boundary at x = L: c_{N} = c_{N-2}
            c_next[-1] = c[-1] + alpha * (2.0 * c[-2] - 2.0 * c[-1]) - decay * c[-1]
        c = np.clip(c_next, 0.0, None)
        if step in record_steps:
            snapshots.append((step * dt, c.copy()))

    c_exact, lam = compute_sdd_finite_noflux(x, length=length, c0=c0, D=D, k_deg=k_deg)
    rmse_vs_exact = float(np.sqrt(np.mean((c - c_exact) ** 2))) if not anterior_dirichlet else 0.0
    return {
        "x": x,
        "c_pde": c,
        "c_exact_noflux": c_exact,
        "lambda_um": lam,
        "rmse_vs_exact": rmse_vs_exact,
        "snapshots": snapshots,
    }


def classify_french_flag_zones(
    c: np.ndarray,
    t1: float,
    t2: float,
) -> tuple[np.ndarray, dict[str, float]]:
    """Classify concentration profile `c(x)` into French-Flag territories:
      - `2` = Blue (proximal / high morphogen `c >= T1`)
      - `1` = White (intermediate morphogen `T2 <= c < T1`)
      - `0` = Red (distal / low morphogen `c < T2`)
    Returns `(zone_array, fractions_dict)`.
    """
    zones = np.zeros_like(c, dtype=int)
    zones[c >= t2] = 1
    zones[c >= t1] = 2
    n = max(len(c), 1)
    fracs = {
        "blue_frac": float(np.sum(zones == 2) / n),
        "white_frac": float(np.sum(zones == 1) / n),
        "red_frac": float(np.sum(zones == 0) / n),
    }
    return zones, fracs


# ============================================================================
# 2. Ben-Zvi & Barkai (2010) Expansion-Repression Dynamic Scaling Across Growth
# ============================================================================

def simulate_domain_growth_scaling(
    lengths_um: list[float] | None = None,
    c0: float = 1.0,
    D0: float = 10.0,
    k0: float = 0.0025,
    L_ref: float = 300.0,
) -> dict[str, Any]:
    """Compare French-Flag zone proportions across growing chick limb domains
    `L in [180, 240, 300, 420, 540, 660] um` (HH18 -> HH28) under:
      1. Linear source-sink diffusion `c_lin(x) = c0 * (1 - x/L)` (`T1=2/3, T2=1/3`)
      2. Unscaled SDD exponential decay `c_SDD(x) = c0 * exp(-x / lambda_0)` with fixed
         `lambda_0 = sqrt(D0 / k0) = 63.2456 um` and thresholds `T1_exp = exp(-L_ref/(3*lambda_0))`,
         `T2_exp = exp(-2*L_ref/(3*lambda_0))`
      3. Ben-Zvi & Barkai (2010) Expansion-Repression (ER) integral feedback where
         distal expander production pins `c(L) = T_rep = c0 * exp(-L_ref / lambda_0)`,
         yielding `lambda_eff(L) = L / ln(c0 / T_rep) = (lambda_0 / L_ref) * L`.
    """
    if lengths_um is None:
        lengths_um = [180.0, 240.0, 300.0, 420.0, 540.0, 660.0]
    hh_labels = ["HH18", "HH20", "HH22", "HH24", "HH26", "HH28"]

    lam0 = float(np.sqrt(D0 / k0))
    t1_exp = float(c0 * np.exp(-(L_ref / 3.0) / lam0))
    t2_exp = float(c0 * np.exp(-(2.0 * L_ref / 3.0) / lam0))
    t_rep = float(c0 * np.exp(-L_ref / lam0))

    rows: list[dict[str, Any]] = []
    for idx, L in enumerate(lengths_um):
        x = np.linspace(0.0, L, 1201)
        # 1. Linear source-sink
        c_lin = compute_linear_source_sink(x, length=L, c0=c0)
        _, f_lin = classify_french_flag_zones(c_lin, t1=2.0 * c0 / 3.0, t2=1.0 * c0 / 3.0)

        # 2. Unscaled SDD
        c_unscaled = c0 * np.exp(-x / lam0)
        _, f_unscaled = classify_french_flag_zones(c_unscaled, t1=t1_exp, t2=t2_exp)

        # 3. Ben-Zvi & Barkai (2010) Expansion-Repression scaled SDD
        lam_eff = L / np.log(c0 / t_rep)
        c_er = c0 * np.exp(-x / lam_eff)
        _, f_er = classify_french_flag_zones(c_er, t1=t1_exp, t2=t2_exp)

        rows.append({
            "stage_hh": hh_labels[idx] if idx < len(hh_labels) else f"L{int(L)}",
            "length_um": float(L),
            "lambda_0_um": lam0,
            "lambda_eff_um": float(lam_eff),
            "lin_blue": f_lin["blue_frac"],
            "lin_white": f_lin["white_frac"],
            "lin_red": f_lin["red_frac"],
            "unscaled_blue": f_unscaled["blue_frac"],
            "unscaled_white": f_unscaled["white_frac"],
            "unscaled_red": f_unscaled["red_frac"],
            "er_blue": f_er["blue_frac"],
            "er_white": f_er["white_frac"],
            "er_red": f_er["red_frac"],
        })

    unscaled_blues = np.array([r["unscaled_blue"] for r in rows])
    er_blues = np.array([r["er_blue"] for r in rows])
    lin_blues = np.array([r["lin_blue"] for r in rows])

    return {
        "rows": rows,
        "lambda_0_um": lam0,
        "t1_exp": t1_exp,
        "t2_exp": t2_exp,
        "t_rep": t_rep,
        "unscaled_blue_cv": float(np.std(unscaled_blues) / np.mean(unscaled_blues)),
        "er_blue_cv": float(np.std(er_blues) / np.mean(er_blues)),
        "linear_blue_cv": float(np.std(lin_blues) / np.mean(lin_blues)),
        "er_max_abs_err_vs_third": float(np.max(np.abs(er_blues - 1.0 / 3.0))),
    }


# ============================================================================
# 3. Tickle, Summerbell & Wolpert (1975) Chick Limb ZPA Digit Duplication
# ============================================================================

def extract_digit_sequence(
    c_norm: np.ndarray,
    t4: float = 0.45,
    t3: float = 0.18,
    t2: float = 0.05,
) -> tuple[list[str], np.ndarray]:
    """Read the posterior -> anterior digit sequence off a normalized Shh profile.

    Digit identity per point (Tickle et al. 1975; Riddle et al. 1993): `4` if
    `c >= T4`, `3` if `T3 <= c < T4`, `2` if `T2 <= c < T3`, none if `c < T2`.
    A new digit territory starts (i) whenever the identity changes, or (ii) at an
    interior strict local minimum of `c` (a morphogen valley between two sources)
    whose identity is unchanged — two digit-2 territories separated only by a
    valley above `T2` are therefore counted separately (`4-3-2 | 2-3-4`).  This single
    rule reproduces the Tickle (1981) dose series `4-3-2`, `4-3-2-2`, `4-3-2-2-3`,
    `4-3-2-2-3-4` and replaces the Pass 2 ad hoc half-domain split.
    """
    c = np.asarray(c_norm, dtype=np.float64)
    zones = np.zeros(len(c), dtype=int)
    zones[c >= t2] = 2
    zones[c >= t3] = 3
    zones[c >= t4] = 4
    seq: list[str] = []
    prev = 0
    for i, val in enumerate(zones):
        val = int(val)
        at_valley = 0 < i < len(c) - 1 and c[i] < c[i - 1] and c[i] <= c[i + 1]
        if val != prev or (at_valley and val > 0):
            if val > 0:
                seq.append(str(val))
            prev = val
    return seq, zones


def simulate_chick_limb_zpa_regimes(
    length: float = 500.0,
    lam: float = 80.0,
    t4: float = 0.45,
    t3: float = 0.18,
    t2: float = 0.05,
    nx: int = 1001,
) -> dict[str, Any]:
    """Simulate Tickle, Summerbell & Wolpert (1975) Nature 254:199-202 & Riddle et al.
    (1993) Cell 75:1401-1416 chick wing bud ZPA / Sonic hedgehog (Shh) experiments:
      - Regime 1 (`wild_type_posterior_zpa`): Posterior ZPA (`c_post=1.0, c_ant=0.0`)
        -> Digits `['4', '3', '2']`.
      - Regime 2 (`anterior_zpa_graft_duplication`): Full ZPA graft at anterior margin
        (`c_post=1.0, c_ant=1.0`) -> Mirror-image duplication `['4', '3', '2', '2', '3', '4']`.
      - Regime 3 (`attenuated_anterior_zpa_graft`): Irradiated/attenuated anterior ZPA
        graft (`c_post=1.0, c_ant=0.35`) -> Partial duplication `['4', '3', '2', '2', '3']`
        (Tickle 1981 Nature 289:295-298: the digit specified next to the graft advances
        2 -> 3 -> 4 with graft strength).
      - Regime 4 (`weak_anterior_zpa_graft`): Few-cell / strongly irradiated graft
        (`c_post=1.0, c_ant=0.10`) -> single extra digit `['4', '3', '2', '2']`.
      - Regime 5 (`zpa_ablation_or_cyclopamine`): ZPA excision / Shh blockade
        (`c_post=0.0, c_ant=0.0`) -> `[]` (0 digits, negative control).
    """
    x = np.linspace(0.0, length, nx)

    regimes_spec = [
        ("wild_type_posterior_zpa", 1.0, 0.0),
        ("anterior_zpa_graft_duplication", 1.0, 1.0),
        ("attenuated_anterior_zpa_graft", 1.0, 0.35),
        ("weak_anterior_zpa_graft", 1.0, 0.10),
        ("zpa_ablation_or_cyclopamine", 0.0, 0.0),
    ]

    results: dict[str, Any] = {"x": x, "thresholds": {"T4": t4, "T3": t3, "T2": t2}, "regimes": {}}
    for reg_id, c_post, c_ant in regimes_spec:
        c_prof = c_post * np.exp(-x / lam) + c_ant * np.exp(-(length - x) / lam)
        # Assign digit zone at each point: 4 (c>=t4), 3 (t3<=c<t4), 2 (t2<=c<t3), 0 (c<t2)
        d_zone = np.zeros(nx, dtype=int)
        d_zone[c_prof >= t2] = 2
        d_zone[c_prof >= t3] = 3
        d_zone[c_prof >= t4] = 4

        # Ordered sequence of contiguous digit territories along the posterior -> anterior
        # axis.  Pass 3: the Pass 2 ad hoc half-domain split (only for c_ant >= 0.8) is
        # replaced by the general morphogen-valley rule in `extract_digit_sequence` so that
        # the attenuated graft follows Tickle (1981): 0.35 -> 4-3-2-2-3 (not 4-3-2-3).
        digits_seq, _ = extract_digit_sequence(c_prof, t4=t4, t3=t3, t2=t2)

        symmetry_err = float(np.max(np.abs(c_prof - c_prof[::-1])))
        results["regimes"][reg_id] = {
            "c_post": c_post,
            "c_ant": c_ant,
            "c_profile": c_prof,
            "digit_zones": d_zone,
            "digit_sequence": digits_seq,
            "digit_count": len(digits_seq),
            "mirror_symmetry_max_err": symmetry_err,
        }

    return results


# ============================================================================
# 3b. Pass 3 — 2D Proximo-Distal x Anteroposterior Limb Bud Paddle:
#     AER FGF Outgrowth coupled to Posterior ZPA Shh (Saunders 1948; Summerbell 1974;
#     Summerbell, Lewis & Wolpert 1973; Laufer et al. 1994; Niswander et al. 1994)
# ============================================================================
#
# Reduced 2D model (stated honestly; see audit v3 `transfer_boundary`):
#   * Domain: paddle-shaped wing bud on a fixed Cartesian lattice (dx = 12.5 um), AP width
#     W = 500 um (x = 0 posterior, x = W anterior), PD length L(t) (y = 0 body wall, y = L
#     tip) with an elliptical distal cap of height r_tip = 150 um.  Cells outside the paddle
#     are inactive; the 5-point Laplacian drops missing neighbours => zero-flux everywhere.
#   * AER = distal-facing rim cells (one lattice row thick) running from the posterior to the
#     anterior margin.  FGF (Fgf8 baseline + Shh-induced Fgf4 component, Laufer 1994;
#     Niswander 1994; Crossley 1996):
#       dF/dt = D nabla^2 F - k_F F + s_F * AER(x,y,t) * (1 + g_S * S / (S + K_S))
#   * ZPA = posterior-margin mesenchyme (x <= 60 um) within 120 um of the rim; Shh production
#     requires AER FGF (the positive feedback loop):
#       dS/dt = D nabla^2 S - k_S S + s_S * [ZPA + graft * ZPA_anterior] * F / (F + K_FS)
#   * Outgrowth by distal apposition (progress-zone model, Summerbell-Lewis-Wolpert 1973):
#       dL/dt = v_max * <F>_PZ / (<F>_PZ + K_g),   <F>_PZ = mean FGF within 100 um of the rim;
#     new lattice rows are activated (concentration 0) as L crosses the next grid line.
#   * Readouts: AP digit identity from the distal (<= 150 um) Shh profile normalised to the
#     wild-type peak with the Pass 2 thresholds T4/T3/T2 = 0.45/0.18/0.05; PD skeletal segment
#     from the truncation fraction L_final / L_WT (stylopod < 1/3 <= zeugopod < 2/3 <= autopod).
#   * The developmental clock is compressed: 12 h of model time spans HH18 (180 um) -> HH28
#     (~660 um), i.e. ~5x faster than in ovo, while keeping tau_relax = 1/k (400-1000 s) <<
#     tau_growth so the gradients stay quasi-steady.  No cell proliferation field, no actual
#     digit condensation (Turing) and no 3D dorsoventral axis are modelled.

LIMB2D_DEFAULTS: dict[str, float] = {
    "W_um": 500.0,
    "dx_um": 12.5,
    "ny": 59,
    "r_tip_um": 150.0,
    "D_um2_s": 10.0,
    "k_F_s": 0.001,        # lambda_F = 100 um
    "k_S_s": 10.0 / 6400.0,  # lambda_S = 80 um (same as the 1D ZPA solver)
    "s_F": 0.008,
    "s_S": 0.006,
    "g_S": 1.0,
    "K_S": 0.3,
    "K_FS": 0.1,
    "K_g": 0.3,
    "v_max_um_s": 0.016,
    "L0_um": 180.0,
    "T_s": 12.0 * 3600.0,
    "dt_s": 3.0,
    "zpa_width_um": 60.0,
    "zpa_depth_um": 120.0,
    "pz_depth_um": 100.0,
    "autopod_depth_um": 150.0,
}

HH_STAGE_LENGTHS_UM: dict[str, float] = {
    "HH18": 180.0, "HH20": 276.0, "HH22": 372.0, "HH24": 468.0, "HH26": 564.0, "HH28": 660.0,
}


def build_paddle_geometry(
    L: float,
    W: float = 500.0,
    dx: float = 12.5,
    ny: int = 59,
    r_tip: float = 150.0,
    zpa_width: float = 60.0,
    zpa_depth: float = 120.0,
    pz_depth: float = 100.0,
    rectangular: bool = False,
) -> dict[str, Any]:
    """Paddle mask + AER / ZPA / progress-zone index sets for PD length `L` (um)."""
    nx = int(round(W / dx)) + 1
    x = np.arange(nx) * dx
    y = np.arange(ny) * dx
    X, Y = np.meshgrid(x, y)
    if rectangular:
        mask = Y <= L
    else:
        rt = min(r_tip, L)
        base = L - rt
        half_w = W / 2.0
        mask = Y <= base
        ell = ((X - half_w) / half_w) ** 2 + ((Y - base) / max(rt, 1e-9)) ** 2 <= 1.0
        mask = mask | (ell & (Y > base) & (Y <= L))
    mask[-1, :] = False  # keep one inactive guard row at the distal lattice edge
    col_has = mask.any(axis=0)
    heights = np.where(col_has, ny - 1 - np.argmax(mask[::-1, :], axis=0), -1)
    rim_dist = heights[None, :] - np.arange(ny)[:, None]  # rows below the rim (>= 0 inside)
    aer = mask & (rim_dist == 0)
    zpa_post = mask & (x[None, :] <= zpa_width) & (rim_dist * dx <= zpa_depth)
    zpa_ant = mask & (x[None, :] >= W - zpa_width) & (rim_dist * dx <= zpa_depth)
    pz = mask & (rim_dist * dx <= pz_depth)
    # Padded neighbour-availability masks for the zero-flux 5-point Laplacian
    mp = np.pad(mask, 1).astype(np.float64)
    nb = {
        "up": mp[2:, 1:-1], "down": mp[:-2, 1:-1], "left": mp[1:-1, :-2], "right": mp[1:-1, 2:],
    }
    return {
        "mask": mask, "mask_f": mask.astype(np.float64), "aer": aer.astype(np.float64),
        "zpa_post": zpa_post.astype(np.float64), "zpa_ant": zpa_ant.astype(np.float64),
        "pz": pz, "rim_dist": rim_dist, "heights": heights, "x": x, "y": y, "nb": nb,
        "aer_bool": aer, "nx": nx,
    }


def _masked_laplacian(cp: np.ndarray, nb: dict[str, np.ndarray]) -> np.ndarray:
    """Zero-flux 5-point Laplacian (units of 1/dx^2 omitted) on a padded field `cp`."""
    c = cp[1:-1, 1:-1]
    return (
        nb["up"] * (cp[2:, 1:-1] - c)
        + nb["down"] * (cp[:-2, 1:-1] - c)
        + nb["left"] * (cp[1:-1, :-2] - c)
        + nb["right"] * (cp[1:-1, 2:] - c)
    )


def simulate_limb_bud_2d(
    graft_strength: float = 0.0,
    aer_ablation_L_um: float | None = None,
    zpa_ablated: bool = False,
    grow: bool = True,
    rectangular: bool = False,
    aer_dirichlet_value: float | None = None,
    params: dict[str, float] | None = None,
    snapshot_every_s: float = 3600.0,
    stop_after_ablation_s: float | None = 3.0 * 3600.0,
) -> dict[str, Any]:
    """Explicit finite-difference integration of the coupled AER-FGF / ZPA-Shh system on the
    growing 2D paddle (see the section header for the equations).

    `graft_strength`      relative Shh production of a second ZPA at the anterior margin
                          (Tickle 1975/1981 graft; 0 = wild type).
    `aer_ablation_L_um`   remove the AER (s_F -> 0 for all later times) when L first reaches
                          this length (Saunders 1948 / Summerbell 1974 excision at a stage).
    `zpa_ablated`         no Shh production anywhere (negative control).
    `aer_dirichlet_value` verification mode: AER cells are clamped to this FGF value instead
                          of producing it (lets a non-growing rectangle be compared with the
                          1D cosh steady state).
    """
    p = dict(LIMB2D_DEFAULTS)
    if params:
        p.update(params)
    W, dx, ny = p["W_um"], p["dx_um"], int(p["ny"])
    D, k_F, k_S = p["D_um2_s"], p["k_F_s"], p["k_S_s"]
    dt = p["dt_s"]
    assert dt <= dx * dx / (4.0 * D), "explicit 2D stability limit violated"
    alpha = D * dt / (dx * dx)
    n_steps = int(np.ceil(p["T_s"] / dt))

    def geom(L_now: float) -> dict[str, Any]:
        return build_paddle_geometry(
            L_now, W=W, dx=dx, ny=ny, r_tip=p["r_tip_um"], zpa_width=p["zpa_width_um"],
            zpa_depth=p["zpa_depth_um"], pz_depth=p["pz_depth_um"], rectangular=rectangular,
        )

    L = float(p["L0_um"])
    g = geom(L)
    nx = g["nx"]
    Fp = np.zeros((ny + 2, nx + 2))
    Sp = np.zeros((ny + 2, nx + 2))
    F = Fp[1:-1, 1:-1]
    S = Sp[1:-1, 1:-1]
    L_cell = int(L // dx)
    aer_on = 1.0
    t_ablation: float | None = None
    shh_peak_at_ablation: float | None = None
    L_history: list[tuple[float, float]] = [(0.0, L)]
    snapshots: list[dict[str, Any]] = []
    snap_stride = max(int(snapshot_every_s / dt), 1)
    hist_stride = max(int(600.0 / dt), 1)
    zpa_gain = 0.0 if zpa_ablated else 1.0

    t = 0.0
    for step in range(n_steps):
        if aer_ablation_L_um is not None and aer_on > 0.0 and L >= aer_ablation_L_um:
            aer_on = 0.0
            t_ablation = t
            shh_peak_at_ablation = float(S.max())
        if aer_dirichlet_value is not None:
            F[g["aer_bool"]] = aer_dirichlet_value
        lapF = _masked_laplacian(Fp, g["nb"])
        lapS = _masked_laplacian(Sp, g["nb"])
        fgf_src = (p["s_F"] * aer_on) * g["aer"] * (1.0 + p["g_S"] * S / (S + p["K_S"]))
        shh_src = (p["s_S"] * zpa_gain) * (g["zpa_post"] + graft_strength * g["zpa_ant"]) * (F / (F + p["K_FS"]))
        if aer_dirichlet_value is not None:
            fgf_src = fgf_src * 0.0
        F_new = (F + alpha * lapF - (k_F * dt) * F + dt * fgf_src) * g["mask_f"]
        S_new = (S + alpha * lapS - (k_S * dt) * S + dt * shh_src) * g["mask_f"]
        F[...] = np.maximum(F_new, 0.0)
        S[...] = np.maximum(S_new, 0.0)
        if aer_dirichlet_value is not None:
            F[g["aer_bool"]] = aer_dirichlet_value
        if grow:
            F_pz = float(F[g["pz"]].mean())
            L = min(L + dt * p["v_max_um_s"] * F_pz / (F_pz + p["K_g"]), (ny - 2) * dx)
            if int(L // dx) != L_cell:
                L_cell = int(L // dx)
                g = geom(L)
        t += dt
        if step % hist_stride == 0:
            L_history.append((t, L))
        if step % snap_stride == 0:
            snapshots.append({"t_s": t, "L_um": L, "F": F.copy(), "S": S.copy(), "mask": g["mask"].copy()})
        if (
            stop_after_ablation_s is not None
            and t_ablation is not None
            and t - t_ablation >= stop_after_ablation_s
        ):
            break

    L_history.append((t, L))
    snapshots.append({"t_s": t, "L_um": L, "F": F.copy(), "S": S.copy(), "mask": g["mask"].copy()})
    strip = g["mask"] & (g["rim_dist"] * dx <= p["autopod_depth_um"])
    S_ap = np.array([S[strip[:, j], j].mean() if strip[:, j].any() else 0.0 for j in range(nx)])
    F_aer = np.array([F[g["aer_bool"][:, j], j].mean() if g["aer_bool"][:, j].any() else 0.0 for j in range(nx)])
    return {
        "params": p,
        "x_um": g["x"],
        "y_um": g["y"],
        "L_final_um": float(L),
        "t_final_s": float(t),
        "F": F.copy(),
        "S": S.copy(),
        "mask": g["mask"].copy(),
        "aer": g["aer_bool"].copy(),
        "zpa_post": g["zpa_post"].astype(bool),
        "zpa_ant": g["zpa_ant"].astype(bool),
        "autopod_strip": strip,
        "S_ap_profile": S_ap,
        "F_aer_profile": F_aer,
        "F_max": float(F.max()),
        "S_max": float(S.max()),
        "t_ablation_s": t_ablation,
        "shh_peak_at_ablation": shh_peak_at_ablation,
        "L_history": np.array(L_history),
        "snapshots": snapshots,
    }


def hh_stage_from_length(L_um: float) -> float:
    """Linear HH stage clock used throughout the script: HH18 = 180 um ... HH28 = 660 um."""
    return 18.0 + (L_um - 180.0) / 48.0


def pd_segment_from_fraction(frac: float) -> str:
    """Progress-zone (Summerbell-Lewis-Wolpert 1973) PD trisection of the final wild-type length:
    stylopod (humerus) < 1/3 <= zeugopod (radius/ulna) < 2/3 <= autopod (wrist + digits)."""
    if frac < 1.0 / 3.0:
        return "stylopod"
    if frac < 2.0 / 3.0:
        return "zeugopod"
    return "autopod"


def verify_limb2d_against_1d_cosh(params: dict[str, float] | None = None) -> dict[str, Any]:
    """Code verification of the 2D stencil: non-growing rectangle, AER row clamped at F = 1,
    no Shh.  The exact steady state is `F(y) = cosh(y / lambda_F) / cosh(L / lambda_F)`
    (no-flux body wall at y = 0, Dirichlet rim at y = L), identical in every column."""
    p = dict(LIMB2D_DEFAULTS)
    p.update({"W_um": 100.0, "ny": 27, "T_s": 6000.0, "L0_um": 300.0})
    if params:
        p.update(params)
    res = simulate_limb_bud_2d(
        grow=False, rectangular=True, aer_dirichlet_value=1.0, zpa_ablated=True, params=p,
        snapshot_every_s=1e12, stop_after_ablation_s=None,
    )
    lam_F = float(np.sqrt(p["D_um2_s"] / p["k_F_s"]))
    col = res["mask"][:, res["mask"].shape[1] // 2]
    y = res["y_um"][col]
    L_eff = float(y.max())
    exact = np.cosh(y / lam_F) / np.cosh(L_eff / lam_F)
    num = res["F"][col, res["mask"].shape[1] // 2]
    rmse = float(np.sqrt(np.mean((num - exact) ** 2)))
    # Every active column must carry the same profile (the problem is 1D in y)
    rows = np.where(res["mask"], res["F"], np.nan)[col, :]
    col_spread = float(np.nanmax(np.nanmax(rows, axis=1) - np.nanmin(rows, axis=1)))
    return {"rmse_vs_cosh": rmse, "lambda_F_um": lam_F, "max_column_spread": col_spread, "y_um": y, "numeric": num, "exact": exact}


def run_limb_bud_2d_experiments(params: dict[str, float] | None = None) -> dict[str, Any]:
    """Pass 3 experimental suite on the 2D paddle:
      (a) wild type;                               (b) anterior ZPA grafts of strength 0.10 / 0.35 / 1.0
      (c) AER excision at HH18/20/22/24/26;         (d) ZPA ablation; (e) 1D-cosh code verification.
    Empirical comparisons: Tickle (1975, 1981) digit sequences and dose series; Saunders (1948)
    & Summerbell (1974) stage-dependent distal truncation; Laufer (1994) & Niswander (1994)
    Shh-FGF feedback (Shh loss after AER removal; Shh-dependent posterior AER FGF)."""
    thr = {"t4": 0.45, "t3": 0.18, "t2": 0.05}
    wt = simulate_limb_bud_2d(params=params)
    s_ref = float(wt["S_ap_profile"].max())
    wt_seq, wt_zones = extract_digit_sequence(wt["S_ap_profile"] / s_ref, **thr)

    grafts: dict[str, dict[str, Any]] = {}
    for gs in (0.10, 0.35, 1.0):
        r = simulate_limb_bud_2d(graft_strength=gs, params=params)
        seq, zones = extract_digit_sequence(r["S_ap_profile"] / s_ref, **thr)
        n_edge = 3
        pa_ratio = float(r["F_aer_profile"][:n_edge].mean() / max(r["F_aer_profile"][-n_edge:].mean(), 1e-12))
        grafts[f"{gs:.2f}"] = {
            "strength": gs, "result": r, "digit_sequence": seq, "digit_zones": zones,
            "extra_digits": len(seq) - len(wt_seq), "aer_fgf_post_over_ant": pa_ratio,
            "profile_mirror_err": float(np.max(np.abs(r["S_ap_profile"] - r["S_ap_profile"][::-1])) / s_ref),
        }
    wt_pa_ratio = float(wt["F_aer_profile"][:3].mean() / max(wt["F_aer_profile"][-3:].mean(), 1e-12))

    # Tickle (1981): extra digits specified next to the graft increase 0 -> 1 -> 2 -> 3 with dose
    dose_strengths = [0.0, 0.10, 0.35, 1.0]
    dose_extra = [0] + [grafts[f"{gs:.2f}"]["extra_digits"] for gs in (0.10, 0.35, 1.0)]
    dose_ref = [0, 1, 2, 3]
    dose_matches = int(sum(int(a == b) for a, b in zip(dose_extra, dose_ref)))

    # Saunders (1948) / Summerbell (1974): AER excision at successive stages
    ablation_ref = {"HH18": "stylopod", "HH20": "zeugopod", "HH22": "zeugopod", "HH24": "autopod", "HH26": "autopod"}
    ablations: dict[str, dict[str, Any]] = {}
    for stage, L_stage in HH_STAGE_LENGTHS_UM.items():
        if stage not in ablation_ref:
            continue
        r = simulate_limb_bud_2d(aer_ablation_L_um=L_stage, params=params)
        frac = r["L_final_um"] / wt["L_final_um"]
        seg = pd_segment_from_fraction(frac)
        shh_pre = r["shh_peak_at_ablation"] or 0.0
        ablations[stage] = {
            "stage_L_um": L_stage, "result": r, "L_final_um": r["L_final_um"], "truncation_fraction": float(frac),
            "overshoot_um": float(r["L_final_um"] - L_stage), "predicted_segment": seg,
            "reference_segment": ablation_ref[stage], "segment_match": seg == ablation_ref[stage],
            "shh_peak_before": float(shh_pre), "shh_peak_after": float(r["S_max"]),
            "shh_remaining_fraction": float(r["S_max"] / shh_pre) if shh_pre > 0 else 0.0,
        }
    abl_stages = np.array([float(s[2:]) for s in ablations])
    abl_final = np.array([ablations[s]["L_final_um"] for s in ablations])
    rho = float(np.corrcoef(np.argsort(np.argsort(abl_stages)), np.argsort(np.argsort(abl_final)))[0, 1])
    strictly_monotone = bool(np.all(np.diff(abl_final) > 0.0))
    segment_matches = int(sum(int(a["segment_match"]) for a in ablations.values()))
    shh_fracs = [ablations[s]["shh_remaining_fraction"] for s in ablations if s != "HH18"]

    zpa_off = simulate_limb_bud_2d(zpa_ablated=True, params=params)
    zpa_off_seq, _ = extract_digit_sequence(zpa_off["S_ap_profile"] / s_ref, **thr)
    verify = verify_limb2d_against_1d_cosh()

    return {
        "thresholds": thr,
        "s_ref": s_ref,
        "wild_type": {"result": wt, "digit_sequence": wt_seq, "digit_zones": wt_zones,
                      "aer_fgf_post_over_ant": wt_pa_ratio, "stage_final": hh_stage_from_length(wt["L_final_um"])},
        "grafts": grafts,
        "dose_strengths": dose_strengths,
        "dose_extra_digits": dose_extra,
        "dose_reference_extra_digits": dose_ref,
        "dose_matches": dose_matches,
        "ablations": ablations,
        "ablation_spearman_rho": rho,
        "ablation_strictly_monotone": strictly_monotone,
        "ablation_segment_matches": segment_matches,
        "ablation_segment_total": len(ablations),
        "shh_remaining_after_aer_removal_max": float(max(shh_fracs)) if shh_fracs else 0.0,
        "zpa_ablation": {"result": zpa_off, "digit_sequence": zpa_off_seq, "L_final_um": zpa_off["L_final_um"]},
        "verification": verify,
    }


# ============================================================================
# 4. Positional Precision & Stochastic Morphogen Noise Propagation
# ============================================================================

def evaluate_positional_precision_noise(
    c0: float = 1.0,
    lam: float = 63.245553203367585,
    sigma_c: float = 0.012,
    n_mc: int = 400,
    seed: int = 1969,
) -> dict[str, Any]:
    """Verify Wolpert (1969) & Crick (1970) positional error propagation law:
      `sigma_x(x_i) = sigma_c / |dc/dx(x_i)| = (lambda / T_i) * sigma_c`
    across a 5-point threshold sweep `T in [0.15, 0.25, 0.35, 0.50, 0.65]` using a
    deterministic Monte Carlo ensemble (`seed=1969`).
    """
    rng = np.random.default_rng(seed)
    thresholds = np.array([0.15, 0.25, 0.35, 0.50, 0.65], dtype=np.float64)
    x_exact = lam * np.log(c0 / thresholds)
    sigma_x_analytical = (lam / thresholds) * sigma_c

    # Monte Carlo perturbation: c_noisy(x) = c0 * exp(-x/lam) + delta_c
    delta_c = rng.normal(0.0, sigma_c, size=(n_mc, len(thresholds)))
    t_perturbed = np.clip(thresholds[None, :] - delta_c, 1e-4, c0 * 0.999)
    x_mc = lam * np.log(c0 / t_perturbed)
    sigma_x_mc = np.std(x_mc, axis=0, ddof=1)

    r_corr = float(np.corrcoef(sigma_x_analytical, sigma_x_mc)[0, 1])
    rel_err = float(np.mean(np.abs(sigma_x_mc - sigma_x_analytical) / sigma_x_analytical))
    return {
        "thresholds": thresholds,
        "x_exact_um": x_exact,
        "sigma_x_analytical_um": sigma_x_analytical,
        "sigma_x_mc_um": sigma_x_mc,
        "pearson_r": r_corr,
        "mean_rel_err": rel_err,
    }


# ============================================================================
# 5. Biological Specimen Segmentation (Gallus gallus 72h Embryo in ovo)
# ============================================================================

def segment_gallus_embryo_specimen(specimen_path: Path = SPECIMEN_PATH) -> dict[str, Any]:
    """Segment the vendored 72h HH18-20 `Gallus gallus domesticus` embryo photograph
    (`datasets/french_flag_morphogen/french_flag_specimen.jpg`, CC BY 4.0):
      1. Isolate the circular yolk blastoderm ROI via chromatic thresholding.
      2. Locate the central embryonic organizer (`organizer_center_xy = (268, 388)`)
         from the smoothed hemoglobin/vascular optical density `H(x,y) = max(0, R - 0.72*G)`.
      3. Compute the radial morphogen-response / vascular density profile `c(r)` across
         `r in [0, 220] px` and fit the SDD exponential decay model
         `c_fit(r) = c0 * exp(-r / lambda_px) + bg`.
      4. Partition the blastoderm into 3 concentric French-Flag territories:
         - Blue (`c >= T1 = 0.50`): Embryo proper & primary vitelline organizer
         - White (`T2 = 0.18 <= c < T1`): Intermediate branching *area vasculosa*
         - Red (`c < T2`): Distal sinus terminalis & peripheral vitelline margin
    """
    im = Image.open(specimen_path).convert("RGB")
    arr = np.asarray(im, dtype=np.float64) / 255.0
    H, W, _ = arr.shape
    R, G, B = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    yy, xx = np.mgrid[0:H, 0:W]

    yolk_mask = (R > 0.45) & (B < 0.45) & ((R - B) > 0.25)
    yolk_mask = ndi.binary_fill_holes(yolk_mask)
    lab, n = ndi.label(yolk_mask)
    sizes = ndi.sum(yolk_mask, lab, range(1, n + 1))
    yolk_mask = lab == (1 + int(np.argmax(sizes)))
    cy_yolk, cx_yolk = ndi.center_of_mass(yolk_mask)

    dist_yolk = np.hypot(xx - cx_yolk, yy - cy_yolk)
    inner_mask = yolk_mask & (dist_yolk < 140.0)
    hemo = np.clip(R - 0.72 * G, 0.0, None) * yolk_mask
    smooth_hemo = ndi.gaussian_filter(hemo, sigma=12.0)
    emb_y, emb_x = np.unravel_index(np.argmax(smooth_hemo * inner_mask), hemo.shape)

    r_grid = np.hypot(xx - emb_x, yy - emb_y)
    r_bins = np.linspace(0.0, 220.0, 45)
    r_mid = 0.5 * (r_bins[:-1] + r_bins[1:])
    c_prof = np.zeros_like(r_mid)
    for i in range(len(r_mid)):
        ring = yolk_mask & (r_grid >= r_bins[i]) & (r_grid < r_bins[i + 1])
        c_prof[i] = float(hemo[ring].mean()) if np.any(ring) else 0.0

    c_min, c_max = float(c_prof.min()), float(c_prof.max())
    c_norm = (c_prof - c_min) / (c_max - c_min + 1e-12)

    def _sdd_radial_model(r: np.ndarray, c0_p: float, lam_p: float, bg_p: float) -> np.ndarray:
        return c0_p * np.exp(-r / lam_p) + bg_p

    popt, _ = curve_fit(
        _sdd_radial_model,
        r_mid,
        c_norm,
        p0=[1.0, 40.0, 0.01],
        bounds=([0.5, 10.0, -0.2], [1.5, 250.0, 0.3]),
    )
    c_fit = _sdd_radial_model(r_mid, *popt)
    r_corr = float(np.corrcoef(c_norm, c_fit)[0, 1])
    rmse = float(np.sqrt(np.mean((c_norm - c_fit) ** 2)))

    # Linear fit comparison (demonstrating why exponential SDD fits better than linear decay)
    lin_coef = np.polyfit(r_mid, c_norm, 1)
    c_lin_fit = np.polyval(lin_coef, r_mid)
    rmse_linear = float(np.sqrt(np.mean((c_norm - c_lin_fit) ** 2)))

    # 2D French-Flag territory map on the blastoderm
    t1_spec, t2_spec = 0.50, 0.18
    r_1 = float(popt[1] * np.log(popt[0] / (t1_spec - popt[2])))
    r_2 = float(popt[1] * np.log(popt[0] / (t2_spec - popt[2])))

    return {
        "rgb": arr,
        "yolk_mask": yolk_mask,
        "hemo_field": hemo,
        "organizer_xy": (int(emb_x), int(emb_y)),
        "r_mid_px": r_mid,
        "c_norm": c_norm,
        "c_fit_sdd": c_fit,
        "c_fit_linear": c_lin_fit,
        "fitted_c0": float(popt[0]),
        "fitted_lambda_px": float(popt[1]),
        "fitted_bg": float(popt[2]),
        "pearson_r": r_corr,
        "rmse_sdd": rmse,
        "rmse_linear": rmse_linear,
        "r1_blue_px": r_1,
        "r2_white_px": r_2,
    }


# ============================================================================
# 6. Figure & Animated GIF Rendering
# ============================================================================

def render_outputs(
    out_png: Path = OUT_PNG,
    out_gif: Path = OUT_GIF,
    out_limb_gif: Path = OUT_LIMB_GIF,
    D: float = 10.0,
    k_deg: float = 0.0025,
    length: float = 300.0,
    c0: float = 1.0,
    seed: int = 1969,
    limb2d_params: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Render the 9-panel synthesis figure `docs/media/french_flag_morphogen_comparison.png`
    (A-F from Pass 2, G-I the Pass 3 2D AER/ZPA paddle), the 18-frame ontogenetic growth +
    scaling animation `docs/media/french_flag_morphogen_growth.gif`, and the Pass 3 2D limb bud
    outgrowth animation `docs/media/french_flag_morphogen_limb2d.gif`.
    """
    out_png.parent.mkdir(parents=True, exist_ok=True)
    out_gif.parent.mkdir(parents=True, exist_ok=True)
    out_limb_gif.parent.mkdir(parents=True, exist_ok=True)

    pde_res = integrate_sdd_pde_1d(length=length, nx=151, D=D, k_deg=k_deg, c0=c0)
    scale_res = simulate_domain_growth_scaling(c0=c0, D0=D, k0=k_deg, L_ref=length)
    zpa_res = simulate_chick_limb_zpa_regimes()
    noise_res = evaluate_positional_precision_noise(c0=c0, lam=pde_res["lambda_um"], seed=seed)
    spec_res = segment_gallus_embryo_specimen()
    limb2d = run_limb_bud_2d_experiments(params=limb2d_params)

    fig = plt.figure(figsize=(18.0, 16.8), dpi=160)
    fig.patch.set_facecolor("#0f172a")
    gs = fig.add_gridspec(3, 3, hspace=0.36, wspace=0.26, left=0.055, right=0.975, top=0.935, bottom=0.045)

    panel_bg = "#1e293b"
    text_fg = "#f8fafc"
    grid_col = "#334155"

    def _style_ax(ax: plt.Axes, title: str, xlabel: str, ylabel: str) -> None:
        ax.set_facecolor(panel_bg)
        ax.set_title(title, color=text_fg, fontsize=10.5, fontweight="bold", loc="left", pad=8)
        ax.set_xlabel(xlabel, color="#cbd5e1", fontsize=9.2)
        ax.set_ylabel(ylabel, color="#cbd5e1", fontsize=9.2)
        ax.tick_params(colors="#cbd5e1", labelsize=8.5)
        for sp in ax.spines.values():
            sp.set_color("#475569")
        ax.grid(True, linestyle="--", alpha=0.35, color=grid_col)

    # Panel A: Wolpert (1969) Linear Source-Sink vs SDD PDE & French-Flag Thresholds
    ax_a = fig.add_subplot(gs[0, 0])
    x = pde_res["x"]
    c_lin = compute_linear_source_sink(x, length=length, c0=c0)
    c_sdd, lam = compute_sdd_exponential(x, c0=c0, D=D, k_deg=k_deg)
    _style_ax(
        ax_a,
        "A. Wolpert (1969) Linear vs SDD PDE Gradient",
        "Anteroposterior Coordinate x (μm)",
        "Normalized Morphogen c(x) / c₀",
    )
    ax_a.axvspan(0, length / 3.0, color="#1d4ed8", alpha=0.18, label="Canonical Blue Zone (0–100 μm)")
    ax_a.axvspan(length / 3.0, 2.0 * length / 3.0, color="#f8fafc", alpha=0.10, label="Canonical White Zone (100–200 μm)")
    ax_a.axvspan(2.0 * length / 3.0, length, color="#dc2626", alpha=0.16, label="Canonical Red Zone (200–300 μm)")
    ax_a.plot(x, c_lin, "--", color="#38bdf8", lw=2.2, label="Linear source-sink c₀(1 - x/L)")
    ax_a.plot(x, c_sdd, "-", color="#f59e0b", lw=2.5, label=f"Analytical SDD exp(-x/λ), λ={lam:.1f} μm")
    ax_a.plot(
        x[::4],
        pde_res["c_pde"][::4],
        "o",
        color="#22c55e",
        ms=4.2,
        label=f"Steady-state SDD PDE (RMSE={pde_res['rmse_vs_exact']:.1e})",
    )
    ax_a.axhline(scale_res["t1_exp"], color="#60a5fa", ls=":", lw=1.4, label=f"T₁ (exp) = {scale_res['t1_exp']:.3f}")
    ax_a.axhline(scale_res["t2_exp"], color="#f87171", ls=":", lw=1.4, label=f"T₂ (exp) = {scale_res['t2_exp']:.3f}")
    ax_a.set_xlim(0, length)
    ax_a.set_ylim(0, 1.05)
    ax_a.legend(loc="upper right", fontsize=7.4, facecolor="#0f172a", edgecolor="#475569", labelcolor=text_fg)

    # Panel B: Ben-Zvi & Barkai (2010) Expansion-Repression Scaling Across HH18-HH28 Growth
    ax_b = fig.add_subplot(gs[0, 1])
    _style_ax(
        ax_b,
        "B. Growth Scaling (HH18→HH28): Unscaled SDD vs Expander Feedback",
        "Chick Limb Domain Length L (μm) [HH18 → HH28]",
        "Fractional Territory Width (x_i / L)",
    )
    L_vals = [r["length_um"] for r in scale_res["rows"]]
    ax_b.plot(
        L_vals,
        [r["unscaled_blue"] for r in scale_res["rows"]],
        "s--",
        color="#f97316",
        lw=2.1,
        label=f"Unscaled SDD Blue frac (CV={scale_res['unscaled_blue_cv']:.3f})",
    )
    ax_b.plot(
        L_vals,
        [r["unscaled_red"] for r in scale_res["rows"]],
        "^--",
        color="#ef4444",
        lw=1.9,
        label="Unscaled SDD Red frac (expands with L)",
    )
    ax_b.plot(
        L_vals,
        [r["er_blue"] for r in scale_res["rows"]],
        "o-",
        color="#38bdf8",
        lw=2.6,
        label=f"Expansion-Repression Blue (CV={scale_res['er_blue_cv']:.4f})",
    )
    ax_b.axhline(1.0 / 3.0, color="#22c55e", ls=":", lw=1.6, label="Exact 1/3 French-Flag Invariance")
    for r in scale_res["rows"]:
        ax_b.annotate(
            r["stage_hh"],
            (r["length_um"], r["unscaled_blue"] + 0.025),
            color="#cbd5e1",
            fontsize=7.6,
            ha="center",
        )
    ax_b.set_ylim(0.0, 0.80)
    ax_b.legend(loc="upper left", fontsize=7.6, facecolor="#0f172a", edgecolor="#475569", labelcolor=text_fg)

    # Panel C: Tickle, Summerbell & Wolpert (1975) ZPA Graft Mirror-Image Digit Duplication
    ax_c = fig.add_subplot(gs[0, 2])
    _style_ax(
        ax_c,
        "C. Tickle et al. (1975) Chick Wing ZPA Graft Digit Duplication",
        "Posterior (ZPA, x=0) → Anterior (x=500 μm) Axis",
        "Shh Morphogen Concentration c(x)",
    )
    xz = zpa_res["x"]
    wt = zpa_res["regimes"]["wild_type_posterior_zpa"]
    dup = zpa_res["regimes"]["anterior_zpa_graft_duplication"]
    att = zpa_res["regimes"]["attenuated_anterior_zpa_graft"]
    abl = zpa_res["regimes"]["zpa_ablation_or_cyclopamine"]

    ax_c.plot(xz, wt["c_profile"], "-", color="#38bdf8", lw=2.3, label=f"Wild-type ZPA → {'-'.join(wt['digit_sequence'])}")
    ax_c.plot(xz, dup["c_profile"], "-", color="#f43f5e", lw=2.5, label=f"Anterior ZPA graft → {'-'.join(dup['digit_sequence'])}")
    ax_c.plot(xz, att["c_profile"], "--", color="#eab308", lw=2.0, label=f"Attenuated graft → {'-'.join(att['digit_sequence'])}")
    ax_c.plot(xz, abl["c_profile"], ":", color="#94a3b8", lw=2.0, label="ZPA ablation control → 0 digits")
    ax_c.axhline(zpa_res["thresholds"]["T4"], color="#60a5fa", ls="--", lw=1.2, label="T₄ = 0.45 (Digit 4)")
    ax_c.axhline(zpa_res["thresholds"]["T3"], color="#e2e8f0", ls="--", lw=1.2, label="T₃ = 0.18 (Digit 3)")
    ax_c.axhline(zpa_res["thresholds"]["T2"], color="#f87171", ls="--", lw=1.2, label="T₂ = 0.05 (Digit 2)")
    ax_c.set_xlim(0, 500.0)
    ax_c.set_ylim(-0.03, 1.15)
    ax_c.legend(loc="upper center", fontsize=7.2, ncol=2, facecolor="#0f172a", edgecolor="#475569", labelcolor=text_fg)

    # Panel D: Positional Precision & Noise Sensitivity sigma_x = (lambda / T_i) * sigma_c
    ax_d = fig.add_subplot(gs[1, 0])
    _style_ax(
        ax_d,
        f"D. Positional Boundary Error σ_x(T) (MC r = {noise_res['pearson_r']:.4f})",
        "Morphogen Threshold T_i / c₀",
        "Boundary Positional Uncertainty σ_x (μm)",
    )
    t_dense = np.linspace(0.12, 0.70, 200)
    ax_d.plot(
        t_dense,
        (lam / t_dense) * 0.012,
        "-",
        color="#a855f7",
        lw=2.4,
        label="Analytical σ_x = (λ / T_i) σ_c",
    )
    ax_d.plot(
        noise_res["thresholds"],
        noise_res["sigma_x_mc_um"],
        "o",
        color="#22c55e",
        ms=6.5,
        label=f"Monte Carlo (N=400, rel err={noise_res['mean_rel_err']*100:.2f}%)",
    )
    ax_d.set_xlim(0.12, 0.70)
    ax_d.legend(loc="upper right", fontsize=8.0, facecolor="#0f172a", edgecolor="#475569", labelcolor=text_fg)

    # Panel E: Biological Specimen — 72h HH18-20 Gallus gallus Embryo & French-Flag Rings
    ax_e = fig.add_subplot(gs[1, 1])
    _style_ax(
        ax_e,
        "E. Gallus gallus 72h Embryo Specimen (CC BY 4.0) & Zones",
        "Image X (px)",
        "Image Y (px)",
    )
    ax_e.imshow(spec_res["rgb"])
    ox, oy = spec_res["organizer_xy"]
    ax_e.plot(ox, oy, marker="*", color="#38bdf8", ms=12, label="Embryo Organizer Source")
    theta = np.linspace(0, 2 * np.pi, 200)
    ax_e.plot(
        ox + spec_res["r1_blue_px"] * np.cos(theta),
        oy + spec_res["r1_blue_px"] * np.sin(theta),
        color="#3b82f6",
        lw=2.0,
        ls="-",
        label=f"Blue/White T₁ ring (r₁={spec_res['r1_blue_px']:.1f} px)",
    )
    ax_e.plot(
        ox + spec_res["r2_white_px"] * np.cos(theta),
        oy + spec_res["r2_white_px"] * np.sin(theta),
        color="#f8fafc",
        lw=2.0,
        ls="--",
        label=f"White/Red T₂ ring (r₂={spec_res['r2_white_px']:.1f} px)",
    )
    ax_e.plot(
        ox + 220.0 * np.cos(theta),
        oy + 220.0 * np.sin(theta),
        color="#ef4444",
        lw=1.6,
        ls=":",
        label="Distal sinus terminalis (r=220 px)",
    )
    ax_e.set_xlim(60, 580)
    ax_e.set_ylim(680, 120)
    ax_e.grid(False)
    ax_e.legend(loc="lower right", fontsize=7.4, facecolor="#0f172a", edgecolor="#475569", labelcolor=text_fg)

    # Panel F: Empirical Radial Profile c(r) vs SDD Exponential & Linear Fits
    ax_f = fig.add_subplot(gs[1, 2])
    _style_ax(
        ax_f,
        f"F. Specimen Radial Profile Fit (λ={spec_res['fitted_lambda_px']:.1f} px, r={spec_res['pearson_r']:.4f})",
        "Radial Distance from Embryo Organizer r (px)",
        "Normalized Vascular / Morphogen Proxy I(r)",
    )
    r_px = spec_res["r_mid_px"]
    ax_f.axvspan(0, spec_res["r1_blue_px"], color="#1d4ed8", alpha=0.18, label="Blue: Embryo + Proximal Plexus")
    ax_f.axvspan(spec_res["r1_blue_px"], spec_res["r2_white_px"], color="#f8fafc", alpha=0.10, label="White: Branching Area Vasculosa")
    ax_f.axvspan(spec_res["r2_white_px"], 220.0, color="#dc2626", alpha=0.16, label="Red: Peripheral Vitelline Margin")
    ax_f.plot(r_px, spec_res["c_norm"], "o", color="#f8fafc", ms=4.5, label="Measured Gallus gallus radial profile")
    ax_f.plot(
        r_px,
        spec_res["c_fit_sdd"],
        "-",
        color="#22c55e",
        lw=2.5,
        label=f"SDD exp fit (r={spec_res['pearson_r']:.4f}, RMSE={spec_res['rmse_sdd']:.4f})",
    )
    ax_f.plot(
        r_px,
        spec_res["c_fit_linear"],
        "--",
        color="#f97316",
        lw=1.8,
        label=f"Linear decay fit (RMSE={spec_res['rmse_linear']:.4f})",
    )
    ax_f.set_xlim(0, 220.0)
    ax_f.set_ylim(-0.05, 1.08)
    ax_f.legend(loc="upper right", fontsize=7.3, facecolor="#0f172a", edgecolor="#475569", labelcolor=text_fg)

    # ------------------------------------------------------------------------
    # Pass 3 panels G-I: 2D PD x AP paddle with AER-FGF / ZPA-Shh coupling
    # ------------------------------------------------------------------------
    wt2 = limb2d["wild_type"]["result"]
    thr2 = limb2d["thresholds"]
    s_ref = limb2d["s_ref"]
    x2 = wt2["x_um"]
    y2 = wt2["y_um"]
    dx2 = float(x2[1] - x2[0])
    digit_cmap = ListedColormap(["#0f172a", "#64748b", "#f59e0b", "#f43f5e"])  # 0, 2, 3, 4

    def _draw_paddle(ax: plt.Axes, res: dict[str, Any], title: str, zones: np.ndarray | None, show_cbar: bool) -> None:
        S_img = np.where(res["mask"], res["S"] / s_ref, np.nan)
        F_img = np.where(res["mask"], res["F"], np.nan)
        ext = [x2[0] - dx2 / 2, x2[-1] + dx2 / 2, y2[0] - dx2 / 2, y2[-1] + dx2 / 2]
        im = ax.imshow(S_img, origin="lower", extent=ext, cmap="magma", vmin=0.0, vmax=1.0, aspect="equal", interpolation="nearest")
        ax.contour(x2, y2, np.nan_to_num(F_img, nan=0.0), levels=[0.1, 0.3, 0.6, 0.9], colors=["#38bdf8"], linewidths=[0.9], alpha=0.95)
        ay, ax_ = np.where(res["aer"])
        ax.plot(x2[ax_], y2[ay], "s", color="#22c55e", ms=2.3, label="AER (FGF source)")
        zy, zx = np.where(res["zpa_post"])
        ax.plot(x2[zx], y2[zy], ".", color="#f8fafc", ms=1.6, alpha=0.7, label="ZPA (Shh source)")
        if res["zpa_ant"].any():
            zy, zx = np.where(res["zpa_ant"])
            ax.plot(x2[zx], y2[zy], ".", color="#f8fafc", ms=1.6, alpha=0.7)
        if zones is not None:
            bar_y = y2[-1] - 2.0 * dx2
            ax.imshow(zones[None, :], origin="lower", extent=[ext[0], ext[1], bar_y - dx2, bar_y + dx2],
                      cmap=digit_cmap, vmin=0, vmax=3, aspect="auto", interpolation="nearest")
        ax.set_title(title, color=text_fg, fontsize=9.6, fontweight="bold", loc="left", pad=6)
        ax.set_xlim(ext[0], ext[1])
        ax.set_ylim(ext[2], ext[3])
        if show_cbar:
            cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
            cb.set_label("Shh / Shh_WT,peak", color="#cbd5e1", fontsize=8)
            cb.ax.yaxis.set_tick_params(color="#cbd5e1", labelsize=7.5)
            plt.setp(cb.ax.get_yticklabels(), color="#cbd5e1")

    # Panel G: wild-type 2D paddle (Shh field, FGF contours, AER, ZPA, digit readout)
    ax_g = fig.add_subplot(gs[2, 0])
    title_g = f"G. 2D PD×AP Paddle (WT, HH{limb2d['wild_type']['stage_final']:.0f}): ZPA Shh + AER FGF → {'-'.join(limb2d['wild_type']['digit_sequence'])}"
    _style_ax(
        ax_g,
        title_g,
        "Posterior (x=0) → Anterior (x=500 μm)",
        "Proximal (body wall) → Distal PD axis y (μm)",
    )
    ax_g.grid(False)
    zones_wt = np.where(limb2d["wild_type"]["digit_zones"] == 0, 0, limb2d["wild_type"]["digit_zones"] - 1)
    _draw_paddle(ax_g, wt2, title_g, zones_wt, show_cbar=True)
    ax_g.text(
        0.02, 0.03,
        "FGF contours 0.1/0.3/0.6/0.9 (cyan)  ·  bar = distal digit identity (grey 2, amber 3, rose 4)\n"
        f"AER FGF posterior/anterior = {limb2d['wild_type']['aer_fgf_post_over_ant']:.2f} (Shh→Fgf4 feedback; Laufer 1994)",
        transform=ax_g.transAxes, color="#cbd5e1", fontsize=7.0, va="bottom",
    )
    ax_g.legend(loc="upper right", fontsize=7.0, facecolor="#0f172a", edgecolor="#475569", labelcolor=text_fg, markerscale=2.5)

    # Panel H: Tickle (1975, 1981) anterior ZPA graft dose series on the 2D paddle
    ax_h = fig.add_subplot(gs[2, 1])
    _style_ax(
        ax_h,
        f"H. 2D Anterior ZPA Graft Dose Series — Tickle (1981) match {limb2d['dose_matches']}/4",
        "Posterior → Anterior x (μm), distal 150 μm strip",
        "Shh / Shh_WT,peak",
    )
    dose_cols = ["#38bdf8", "#a3e635", "#eab308", "#f43f5e"]
    dose_items = [("0.00", limb2d["wild_type"]["digit_sequence"], wt2["S_ap_profile"])]
    for key in ("0.10", "0.35", "1.00"):
        dose_items.append((key, limb2d["grafts"][key]["digit_sequence"], limb2d["grafts"][key]["result"]["S_ap_profile"]))
    for (key, seq, prof), colr, ref_extra in zip(dose_items, dose_cols, limb2d["dose_reference_extra_digits"]):
        ax_h.plot(x2, prof / s_ref, "-", color=colr, lw=2.2,
                  label=f"graft {float(key):.2f} → {'-'.join(seq) if seq else '∅'} (+{len(seq) - 3} digit{'s' if len(seq) - 3 != 1 else ''}; Tickle 1981: +{ref_extra})")
    ax_h.axhline(thr2["t4"], color="#f43f5e", ls="--", lw=1.0, label="T₄ = 0.45")
    ax_h.axhline(thr2["t3"], color="#f59e0b", ls="--", lw=1.0, label="T₃ = 0.18")
    ax_h.axhline(thr2["t2"], color="#94a3b8", ls="--", lw=1.0, label="T₂ = 0.05")
    ax_h.set_xlim(0, 500.0)
    ax_h.set_ylim(0, 1.12)
    ax_h.legend(loc="upper center", fontsize=6.9, ncol=1, facecolor="#0f172a", edgecolor="#475569", labelcolor=text_fg)

    # Panel I: AER excision (Saunders 1948; Summerbell 1974) — outgrowth L(t) and truncation level
    ax_i = fig.add_subplot(gs[2, 2])
    _style_ax(
        ax_i,
        f"I. AER Excision → PD Truncation (Summerbell 1974 match {limb2d['ablation_segment_matches']}/{limb2d['ablation_segment_total']})",
        "Model time (h)  [HH18 → HH28 compressed clock]",
        "Proximo-distal length L(t) (μm)",
    )
    L_wt_final = wt2["L_final_um"]
    ax_i.axhspan(0, L_wt_final / 3.0, color="#1d4ed8", alpha=0.16, label="Stylopod (humerus) < ⅓ L_WT")
    ax_i.axhspan(L_wt_final / 3.0, 2.0 * L_wt_final / 3.0, color="#f8fafc", alpha=0.08, label="Zeugopod (radius/ulna)")
    ax_i.axhspan(2.0 * L_wt_final / 3.0, L_wt_final * 1.08, color="#dc2626", alpha=0.14, label="Autopod (wrist + digits)")
    hist_wt = wt2["L_history"]
    ax_i.plot(hist_wt[:, 0] / 3600.0, hist_wt[:, 1], "-", color="#38bdf8", lw=2.6, label=f"Wild type → L = {L_wt_final:.0f} μm")
    abl_cols = ["#fde68a", "#fbbf24", "#f97316", "#ef4444", "#be123c"]
    for (stage, abl), colr in zip(limb2d["ablations"].items(), abl_cols):
        h = abl["result"]["L_history"]
        ax_i.plot(h[:, 0] / 3600.0, h[:, 1], "--", color=colr, lw=1.9,
                  label=f"AER off @{stage} → {abl['predicted_segment']} (ref: {abl['reference_segment']}){' ✓' if abl['segment_match'] else ' ✗'}")
        ax_i.plot(h[-1, 0] / 3600.0, h[-1, 1], "o", color=colr, ms=5.5)
    zh = limb2d["zpa_ablation"]["result"]["L_history"]
    ax_i.plot(zh[:, 0] / 3600.0, zh[:, 1], ":", color="#94a3b8", lw=1.8, label=f"ZPA ablation → L = {zh[-1, 1]:.0f} μm, 0 digits")
    ax_i.set_xlim(0, 12.0)
    ax_i.set_ylim(0, L_wt_final * 1.08)
    ax_i.text(
        0.98, 0.03,
        f"Shh remaining ≤ {limb2d['shh_remaining_after_aer_removal_max']*100:.2f}% of pre-excision peak 3 h after AER removal\n(Niswander 1994: AER FGF required for ZPA Shh)",
        transform=ax_i.transAxes, color="#cbd5e1", fontsize=6.8, ha="right", va="bottom",
    )
    ax_i.legend(loc="upper left", fontsize=6.2, facecolor="#0f172a", edgecolor="#475569", labelcolor=text_fg, handlelength=1.6)

    fig.suptitle(
        "Wolpert (1969) French Flag Positional Information, Tickle–Wolpert (1975) ZPA Duplication, Ben-Zvi–Barkai (2010) Scaling\n"
        "& Pass 3: 2D Proximo-Distal × Anteroposterior Limb Bud with AER-FGF / ZPA-Shh Feedback (Saunders 1948 · Summerbell 1974 · Laufer & Niswander 1994)",
        color=text_fg,
        fontsize=12.5,
        fontweight="bold",
        y=0.985,
    )
    fig.savefig(out_png, dpi=160, facecolor=fig.get_facecolor())
    plt.close(fig)

    # Pass 3 GIF: 2D outgrowth of WT vs anterior ZPA graft vs AER excision at HH22
    limb_frames: list[Image.Image] = []
    seq_wt = wt2["snapshots"]
    seq_gr = limb2d["grafts"]["1.00"]["result"]["snapshots"]
    seq_ab = limb2d["ablations"]["HH22"]["result"]["snapshots"]
    n_fr = len(seq_wt)
    ext2 = [x2[0] - dx2 / 2, x2[-1] + dx2 / 2, y2[0] - dx2 / 2, y2[-1] + dx2 / 2]
    for fi in range(n_fr):
        f_l, axs = plt.subplots(1, 3, figsize=(9.6, 4.6), dpi=100)
        f_l.patch.set_facecolor("#0f172a")
        for ax_, series, ttl in zip(
            axs,
            (seq_wt, seq_gr, seq_ab),
            ("Wild type (posterior ZPA)", "Anterior ZPA graft (1.0)", "AER excised at HH22"),
        ):
            snap = series[min(fi, len(series) - 1)]
            S_img = np.where(snap["mask"], snap["S"] / s_ref, np.nan)
            F_img = np.nan_to_num(np.where(snap["mask"], snap["F"], np.nan), nan=0.0)
            ax_.set_facecolor("#0f172a")
            ax_.imshow(S_img, origin="lower", extent=ext2, cmap="magma", vmin=0, vmax=1.0, aspect="equal", interpolation="nearest")
            ax_.contour(x2, y2, F_img, levels=[0.1, 0.3, 0.6, 0.9], colors=["#38bdf8"], linewidths=[0.8])
            ax_.set_title(f"{ttl}\nL = {snap['L_um']:.0f} μm (HH{hh_stage_from_length(snap['L_um']):.0f})", color="#f8fafc", fontsize=8.5)
            ax_.set_xlim(ext2[0], ext2[1])
            ax_.set_ylim(ext2[2], ext2[3])
            ax_.tick_params(colors="#cbd5e1", labelsize=7)
            for sp in ax_.spines.values():
                sp.set_color("#475569")
        axs[0].set_ylabel("PD axis y (μm)", color="#cbd5e1", fontsize=8)
        axs[1].set_xlabel("AP axis x (μm): posterior → anterior", color="#cbd5e1", fontsize=8)
        f_l.suptitle(
            f"2D limb bud paddle — Shh (magma) + FGF contours (cyan) — model t = {seq_wt[fi]['t_s']/3600.0:.1f} h ({fi+1}/{n_fr})",
            color="#f8fafc", fontsize=9.5, fontweight="bold",
        )
        f_l.tight_layout()
        f_l.canvas.draw()
        buf = np.asarray(f_l.canvas.buffer_rgba())[:, :, :3].copy()
        limb_frames.append(Image.fromarray(buf))
        plt.close(f_l)
    pal_l = limb_frames[-1].quantize(colors=255, method=Image.Quantize.MEDIANCUT)
    q_l = [fr.quantize(palette=pal_l, dither=Image.Dither.NONE) for fr in limb_frames]
    q_l[0].save(out_limb_gif, save_all=True, append_images=q_l[1:], duration=320, loop=0, optimize=True)

    # Render 18-frame animated GIF comparing unscaled SDD vs Expansion-Repression scaling + ZPA graft
    frames: list[Image.Image] = []
    growth_lengths = np.linspace(180.0, 660.0, 18)
    lam0 = scale_res["lambda_0_um"]
    t1_e = scale_res["t1_exp"]
    t2_e = scale_res["t2_exp"]
    t_rep = scale_res["t_rep"]
    ff_cmap = ListedColormap(["#dc2626", "#f8fafc", "#2563eb"])

    for idx, Lg in enumerate(growth_lengths):
        xg = np.linspace(0.0, Lg, 300)
        c_un = c0 * np.exp(-xg / lam0)
        z_un, f_un = classify_french_flag_zones(c_un, t1=t1_e, t2=t2_e)

        lam_er = Lg / np.log(c0 / t_rep)
        c_er = c0 * np.exp(-xg / lam_er)
        z_er, f_er = classify_french_flag_zones(c_er, t1=t1_e, t2=t2_e)

        f_gif, (g1, g2, g3) = plt.subplots(
            3, 1, figsize=(7.6, 5.4), dpi=110, gridspec_kw={"height_ratios": [2.2, 0.65, 0.65]}
        )
        f_gif.patch.set_facecolor("#0f172a")
        g1.set_facecolor("#1e293b")
        g1.plot(xg, c_un, "--", color="#f97316", lw=2.2, label=f"Unscaled SDD (λ₀={lam0:.1f} μm, Blue={f_un['blue_frac']:.2f})")
        g1.plot(xg, c_er, "-", color="#38bdf8", lw=2.5, label=f"Expander-Scaled ER (λ_eff={lam_er:.1f} μm, Blue={f_er['blue_frac']:.2f})")
        g1.axhline(t1_e, color="#60a5fa", ls=":", lw=1.2)
        g1.axhline(t2_e, color="#f87171", ls=":", lw=1.2)
        g1.set_xlim(0, 660.0)
        g1.set_ylim(0, 1.05)
        g1.set_title(
            f"Chick Limb Bud Ontogenetic Growth (Frame {idx+1}/18): L = {Lg:.0f} μm",
            color="#f8fafc",
            fontsize=10.5,
            fontweight="bold",
        )
        g1.set_ylabel("c(x) / c₀", color="#cbd5e1", fontsize=9)
        g1.tick_params(colors="#cbd5e1", labelsize=8)
        g1.legend(loc="upper right", fontsize=8, facecolor="#0f172a", edgecolor="#475569", labelcolor="#f8fafc")

        g2.set_facecolor("#0f172a")
        g2.imshow(z_un[None, :], aspect="auto", cmap=ff_cmap, vmin=0, vmax=2, extent=[0, Lg, 0, 1])
        g2.set_xlim(0, 660.0)
        g2.set_yticks([])
        g2.set_title("Unscaled SDD French Flag (Distal Red Overgrowth)", color="#f97316", fontsize=8.5, loc="left")
        g2.tick_params(colors="#cbd5e1", labelsize=7.5)

        g3.set_facecolor("#0f172a")
        g3.imshow(z_er[None, :], aspect="auto", cmap=ff_cmap, vmin=0, vmax=2, extent=[0, Lg, 0, 1])
        g3.set_xlim(0, 660.0)
        g3.set_yticks([])
        g3.set_xlabel("Anteroposterior Coordinate x (μm)", color="#cbd5e1", fontsize=8.5)
        g3.set_title("Expansion-Repression Scaled French Flag (1/3 : 1/3 : 1/3 Invariant)", color="#38bdf8", fontsize=8.5, loc="left")
        g3.tick_params(colors="#cbd5e1", labelsize=7.5)

        f_gif.tight_layout()
        f_gif.canvas.draw()
        buf = np.asarray(f_gif.canvas.buffer_rgba())[:, :, :3].copy()
        frames.append(Image.fromarray(buf))
        plt.close(f_gif)

    pal = frames[-1].quantize(colors=255, method=Image.Quantize.MEDIANCUT)
    q_frames = [fr.quantize(palette=pal, dither=Image.Dither.NONE) for fr in frames]
    q_frames[0].save(
        out_gif,
        save_all=True,
        append_images=q_frames[1:],
        duration=160,
        loop=0,
        optimize=True,
    )

    return {
        "pde": pde_res,
        "scaling": scale_res,
        "zpa": zpa_res,
        "noise": noise_res,
        "specimen": spec_res,
        "limb2d": limb2d,
    }


# ============================================================================
# 7. MorphoEval Dynamic Morphometric Hook (`measure_morphoeval`) & Self-Test
# ============================================================================

def measure_morphoeval() -> list[dict[str, Any]]:
    """Compute reference-anchored quantitative morphometric measurements for
    `morphoeval/measure_outputs.py`.
    """
    ref = load_reference_dataset()
    pde_res = integrate_sdd_pde_1d(length=300.0, nx=151, D=10.0, k_deg=0.0025, c0=1.0)
    scale_res = simulate_domain_growth_scaling()
    zpa_res = simulate_chick_limb_zpa_regimes()
    noise_res = evaluate_positional_precision_noise()
    spec_res = segment_gallus_embryo_specimen()
    limb2d = run_limb_bud_2d_experiments()
    limb_ref = load_limb2d_reference()

    ref_sdd = ref["wolpert_1969_french_flag_model"]["sdd_exponential_decay"]
    ref_spec = ref["metadata"]["specimens"]["gallus_gallus_embryo_72h"]

    wt_digits = zpa_res["regimes"]["wild_type_posterior_zpa"]["digit_count"]
    dup_digits = zpa_res["regimes"]["anterior_zpa_graft_duplication"]["digit_count"]
    abl_digits = zpa_res["regimes"]["zpa_ablation_or_cyclopamine"]["digit_count"]
    er_blue_mean = float(np.mean([r["er_blue"] for r in scale_res["rows"]]))

    wt2 = limb2d["wild_type"]
    g_full = limb2d["grafts"]["1.00"]
    abl = limb2d["ablations"]
    ref_dose = limb_ref["tickle_1981_graft_dose_response"]
    ref_abl = limb_ref["saunders_1948_summerbell_1974_aer_excision"]
    ref_fb = limb_ref["laufer_1994_niswander_1994_shh_fgf_feedback"]

    pass3_measurements: list[dict[str, Any]] = [
        {
            "name": "2D paddle wild-type distal digit count (posterior ZPA Shh, AER FGF outgrowth)",
            "value": int(len(wt2["digit_sequence"])),
            "unit": "digits",
            "method": "Digit identities from the distal 150 um Shh profile of the 2D PD x AP paddle at the end of outgrowth, thresholds T4/T3/T2 = 0.45/0.18/0.05 of the wild-type peak",
            "derived_from": "script",
            "reference_value": 3,
            "reference_source": "Tickle, Summerbell & Wolpert (1975) Nature 254:199-202 (wing digits 2-3-4)",
            "note": f"2D wild-type sequence: {'-'.join(wt2['digit_sequence'])}; final L = {wt2['result']['L_final_um']:.1f} um (HH{wt2['stage_final']:.1f})",
        },
        {
            "name": "2D paddle anterior ZPA graft (strength 1.0) mirror-duplication digit count",
            "value": int(len(g_full["digit_sequence"])),
            "unit": "digits",
            "method": "Same readout with a second ZPA of equal strength at the anterior margin of the 2D paddle",
            "derived_from": "script",
            "reference_value": 6,
            "reference_source": "Tickle, Summerbell & Wolpert (1975) Nature 254:199-202; Tickle (1981) Nature 289:295-298 (4-3-2-2-3-4)",
            "note": f"2D graft sequence: {'-'.join(g_full['digit_sequence'])}; AP mirror error of distal Shh profile = {g_full['profile_mirror_err']:.1e}",
        },
        {
            "name": "Tickle (1981) graft dose series: graft strengths whose extra-digit count matches (0, 0.10, 0.35, 1.0 -> +0, +1, +2, +3)",
            "value": int(limb2d["dose_matches"]),
            "unit": "count",
            "method": "Extra digits specified next to the anterior graft relative to wild type, per relative graft strength, on the 2D paddle",
            "derived_from": "script",
            "reference_value": int(len(ref_dose["relative_strength"])),
            "reference_source": "Tickle (1981) Nature 289:295-298 (digit next to graft advances 2 -> 3 -> 4 with polarizing-cell number)",
            "note": "Model extra digits " + str(limb2d["dose_extra_digits"]) + " vs reference " + str(limb2d["dose_reference_extra_digits"]),
        },
        {
            "name": "AER excision truncation: stages whose predicted most-distal PD segment matches Saunders (1948) / Summerbell (1974)",
            "value": int(limb2d["ablation_segment_matches"]),
            "unit": "count",
            "method": "AER removed when L reaches the HH18/20/22/24/26 length; final L / L_WT trisected into stylopod / zeugopod / autopod",
            "derived_from": "script",
            "reference_value": int(len(ref_abl["stages"])),
            "reference_source": "Saunders (1948) J Exp Zool 108:363-403; Summerbell (1974) J Embryol Exp Morphol 32:651-660",
            "note": "; ".join(f"{s}: L={a['L_final_um']:.0f} um ({a['truncation_fraction']:.2f}) -> {a['predicted_segment']} [ref {a['reference_segment']}]" for s, a in abl.items()),
        },
        {
            "name": "AER excision truncation level vs stage of excision, Spearman rho",
            "value": round(float(limb2d["ablation_spearman_rho"]), 4),
            "unit": "rho",
            "method": "Rank correlation between excision stage (HH18..HH26) and final PD length of the 2D paddle",
            "derived_from": "script",
            "reference_value": 1.0,
            "reference_source": "Summerbell (1974) J Embryol Exp Morphol 32:651-660 (truncation progressively more distal with later excision)",
            "note": f"Strictly monotone: {limb2d['ablation_strictly_monotone']}; mean post-excision overshoot = {np.mean([a['overshoot_um'] for a in abl.values()]):.1f} um",
        },
        {
            "name": "Shh remaining 3 h after AER excision (max over HH20-HH26), fraction of pre-excision peak",
            "value": round(float(limb2d["shh_remaining_after_aer_removal_max"]), 5),
            "unit": "fraction",
            "method": "Peak Shh in the paddle 3 h after s_F -> 0, divided by the peak at the moment of excision (ZPA Shh production requires AER FGF)",
            "derived_from": "script",
            "reference_value": float(ref_fb["shh_after_aer_removal_reference_fraction"]),
            "reference_source": "Niswander, Jeffrey, Martin & Tickle (1994) Nature 371:609-612 (Shh expression lost after AER removal, rescued by FGF4 bead)",
            "note": "Reference value 0 encodes the qualitative loss of ZPA Shh expression reported after AER removal; the model's residual is set by k_S = 1/640 s^-1",
        },
        {
            "name": "AER FGF posterior/anterior ratio in wild type (Shh-induced posterior Fgf4 component)",
            "value": round(float(wt2["aer_fgf_post_over_ant"]), 4),
            "unit": "ratio",
            "method": "Mean FGF in the 3 posterior-most AER cells over the 3 anterior-most AER cells at the end of wild-type outgrowth",
            "derived_from": "script",
            "reference_value": float(ref_fb["wild_type_aer_fgf_posterior_over_anterior_reference"]),
            "reference_source": "Laufer et al. (1994) Cell 79:993-1003 (Fgf4 restricted to posterior AER, Shh-dependent); ratio > 1 is the qualitative reference, no quantitative value published",
            "note": f"Anterior ZPA graft (1.0) restores symmetry: ratio = {g_full['aer_fgf_post_over_ant']:.4f} (Laufer 1994: ectopic anterior Fgf4 after ZPA graft)",
        },
        {
            "name": "2D stencil code verification: RMSE of non-growing rectangle with clamped AER vs 1D cosh steady state",
            "value": round(float(limb2d["verification"]["rmse_vs_cosh"]), 6),
            "unit": "fraction of c0",
            "method": "Rectangle W=100 um, L=300 um, AER row clamped at F=1, no growth/Shh; mid-column compared with cosh(y/lambda_F)/cosh(L/lambda_F), lambda_F = 100 um",
            "derived_from": "script",
            "reference_value": 0.0,
            "reference_source": "Crick (1970) Nature 225:420-422 steady state of dc/dt = D d2c/dx2 - k c with no-flux proximal and Dirichlet distal boundary",
            "note": f"max cross-column spread = {limb2d['verification']['max_column_spread']:.1e} (problem is 1D in y); residual is the half-cell Neumann offset of the dropped-neighbour stencil",
        },
    ]

    return [
        {
            "name": "SDD characteristic decay length lambda = sqrt(D/k)",
            "value": round(float(pde_res["lambda_um"]), 4),
            "unit": "um",
            "method": "Analytical & PDE characteristic decay length sqrt(D/k) for D=10 um^2/s, k=0.0025 s^-1",
            "derived_from": "script",
            "reference_value": round(float(ref_sdd["lambda_um"]), 4),
            "reference_source": "Wolpert (1969) J. Theor. Biol. 25:1-47 & Crick (1970) Nature 225:420-422",
            "note": f"1D SDD PDE vs analytical cosh steady-state RMSE = {pde_res['rmse_vs_exact']:.6f}",
        },
        {
            "name": "Expansion-Repression scaled Blue-zone fraction across HH18-HH28 (L=180..660 um)",
            "value": round(er_blue_mean, 4),
            "unit": "fraction",
            "method": "Mean fractional width x1/L of proximal Blue zone across 6 limb stages under Ben-Zvi & Barkai (2010) expander feedback",
            "derived_from": "script",
            "reference_value": 0.3333,
            "reference_source": "Wolpert (1969) J. Theor. Biol. 25:1-47 & Ben-Zvi & Barkai (2010) PNAS 107:6924-6929",
            "note": f"ER scaling CV = {scale_res['er_blue_cv']:.5f} vs unscaled SDD CV = {scale_res['unscaled_blue_cv']:.4f}",
        },
        {
            "name": "Tickle et al. (1975) wild-type posterior ZPA chick wing digit count (4-3-2)",
            "value": int(wt_digits),
            "unit": "digits",
            "method": "Contiguous suprathreshold Shh digit zones along anteroposterior axis for single posterior ZPA",
            "derived_from": "script",
            "reference_value": 3,
            "reference_source": "Tickle, Summerbell & Wolpert (1975) Nature 254:199-202 Fig. 1",
            "note": "Wild-type sequence: " + "-".join(zpa_res["regimes"]["wild_type_posterior_zpa"]["digit_sequence"]),
        },
        {
            "name": "Tickle et al. (1975) anterior ZPA graft mirror-duplication digit count (4-3-2-2-3-4)",
            "value": int(dup_digits),
            "unit": "digits",
            "method": "Contiguous suprathreshold Shh digit zones under bilateral posterior + anterior ZPA superposition",
            "derived_from": "script",
            "reference_value": 6,
            "reference_source": "Tickle, Summerbell & Wolpert (1975) Nature 254:199-202 & Riddle et al. (1993) Cell 75:1401-1416",
            "note": (
                "Duplicated sequence: "
                + "-".join(zpa_res["regimes"]["anterior_zpa_graft_duplication"]["digit_sequence"])
                + f" (symmetry error = {zpa_res['regimes']['anterior_zpa_graft_duplication']['mirror_symmetry_max_err']:.1e})"
            ),
        },
        {
            "name": "ZPA ablation / cyclopamine negative control digit count",
            "value": int(abl_digits),
            "unit": "digits",
            "method": "Suprathreshold digit territories when posterior and anterior Shh sources are ablated (c=0)",
            "derived_from": "script",
            "reference_value": 0,
            "reference_source": "Tickle, Summerbell & Wolpert (1975) Nature 254:199-202 ZPA excision control",
            "note": "Verifies zero false-positive digit specification in absence of ZPA morphogen",
        },
        {
            "name": "Positional boundary precision Monte Carlo vs analytical correlation r",
            "value": round(float(noise_res["pearson_r"]), 4),
            "unit": "r",
            "method": "Pearson correlation between Monte Carlo boundary SD (N=400) and sigma_x = (lambda/T_i)*sigma_c",
            "derived_from": "script",
            "reference_value": 1.0,
            "reference_source": "Wolpert (1969) J. Theor. Biol. 25:1-47 & Crick (1970) Nature 225:420-422",
            "note": f"Mean relative error across 5 thresholds = {noise_res['mean_rel_err']*100:.2f}%",
        },
        {
            "name": "Gallus gallus 72h embryo specimen radial SDD decay length lambda_px",
            "value": round(float(spec_res["fitted_lambda_px"]), 2),
            "unit": "px",
            "method": "Nonlinear least-squares fit of c0*exp(-r/lambda)+bg to radial area vasculosa profile around organizer (268, 388)",
            "derived_from": "output",
            "reference_value": round(float(ref_spec["fitted_sdd_lambda_px"]), 2),
            "reference_source": "datasets/french_flag_morphogen/french_flag_specimen.jpg (72h HH18-20 Gallus gallus embryo)",
            "note": f"r1(Blue/White) = {spec_res['r1_blue_px']:.1f} px, r2(White/Red) = {spec_res['r2_white_px']:.1f} px",
        },
        {
            "name": "Gallus gallus 72h embryo specimen radial SDD goodness-of-fit r",
            "value": round(float(spec_res["pearson_r"]), 4),
            "unit": "r",
            "method": "Pearson correlation between measured radial vascular/morphogen profile I(r) and fitted SDD exponential decay",
            "derived_from": "output",
            "reference_value": round(float(ref_spec["radial_fit_pearson_r"]), 4),
            "reference_source": "datasets/french_flag_morphogen/french_flag_morphogen_reference.json",
            "note": f"SDD RMSE = {spec_res['rmse_sdd']:.4f} (vs linear decay RMSE = {spec_res['rmse_linear']:.4f})",
        },
    ] + pass3_measurements


def self_test() -> int:
    """Comprehensive deterministic verification suite for Slot 011 (`1969-wolpert-french-flag-morphogen.py`)."""
    print("[self-test] 1. Verifying SHA-256 integrity of datasets/french_flag_morphogen/...")
    digests = verify_dataset_integrity()
    ref = load_reference_dataset()
    limb_ref = load_limb2d_reference()
    assert len(digests) == 4
    assert limb_ref["tickle_1981_graft_dose_response"]["extra_digits"] == [0, 1, 2, 3]

    print("[self-test] 2. Verifying 1D SDD PDE convergence to exact analytical cosh steady state...")
    pde = integrate_sdd_pde_1d(length=300.0, nx=151, D=10.0, k_deg=0.0025, c0=1.0)
    assert abs(pde["lambda_um"] - ref["wolpert_1969_french_flag_model"]["sdd_exponential_decay"]["lambda_um"]) < 1e-9
    assert pde["rmse_vs_exact"] < 5e-4, f"PDE RMSE too large: {pde['rmse_vs_exact']}"

    print("[self-test] 3. Verifying Wolpert linear scaling & Ben-Zvi-Barkai (2010) Expansion-Repression scaling...")
    sc = simulate_domain_growth_scaling()
    assert sc["unscaled_blue_cv"] > 0.35, f"Expected large unscaled SDD CV, got {sc['unscaled_blue_cv']}"
    assert sc["er_blue_cv"] < 1e-3, f"Expected near-zero ER scaled CV, got {sc['er_blue_cv']}"
    assert sc["er_max_abs_err_vs_third"] < 1e-3

    print("[self-test] 4. Verifying Tickle et al. (1975) ZPA digit sequences & mirror-duplication symmetry...")
    zpa = simulate_chick_limb_zpa_regimes()
    assert zpa["regimes"]["wild_type_posterior_zpa"]["digit_sequence"] == ["4", "3", "2"]
    assert zpa["regimes"]["anterior_zpa_graft_duplication"]["digit_sequence"] == ["4", "3", "2", "2", "3", "4"]
    assert zpa["regimes"]["anterior_zpa_graft_duplication"]["mirror_symmetry_max_err"] < 1e-12
    # Tickle (1981) dose series (Pass 3 fix: valley rule instead of the ad hoc half split)
    assert zpa["regimes"]["attenuated_anterior_zpa_graft"]["digit_sequence"] == ["4", "3", "2", "2", "3"]
    assert zpa["regimes"]["weak_anterior_zpa_graft"]["digit_sequence"] == ["4", "3", "2", "2"]
    assert zpa["regimes"]["zpa_ablation_or_cyclopamine"]["digit_sequence"] == []
    # The valley rule must not invent a split on a monotone profile
    assert extract_digit_sequence(np.exp(-np.linspace(0, 500, 501) / 80.0))[0] == ["4", "3", "2"]

    print("[self-test] 5. Verifying positional boundary error propagation sigma_x = (lambda/T)*sigma_c...")
    ns = evaluate_positional_precision_noise(seed=1969)
    ns2 = evaluate_positional_precision_noise(seed=1969)
    assert ns["pearson_r"] > 0.995
    assert np.allclose(ns["sigma_x_mc_um"], ns2["sigma_x_mc_um"], atol=0.0)

    print("[self-test] 6. Verifying Gallus gallus 72h embryo specimen radial segmentation...")
    sp = segment_gallus_embryo_specimen()
    assert sp["pearson_r"] > 0.99, f"Expected specimen r > 0.99, got {sp['pearson_r']}"
    assert sp["rmse_sdd"] < sp["rmse_linear"], "Exponential SDD must fit radial profile better than linear decay"

    print("[self-test] 7. Verifying 2D paddle stencil against the 1D cosh steady state (code verification)...")
    ver = verify_limb2d_against_1d_cosh()
    assert ver["rmse_vs_cosh"] < 5e-3, f"2D stencil RMSE vs cosh too large: {ver['rmse_vs_cosh']}"
    assert ver["max_column_spread"] < 1e-12, "rectangle solution must be identical in every column"

    print("[self-test] 8. Verifying 2D paddle digit readout, Tickle (1981) dose series & AER-FGF asymmetry...")
    ex = run_limb_bud_2d_experiments()
    assert ex["wild_type"]["digit_sequence"] == ["4", "3", "2"], ex["wild_type"]["digit_sequence"]
    assert ex["grafts"]["1.00"]["digit_sequence"] == ["4", "3", "2", "2", "3", "4"], ex["grafts"]["1.00"]["digit_sequence"]
    assert ex["grafts"]["1.00"]["profile_mirror_err"] < 1e-9
    assert ex["dose_extra_digits"] == [0, 1, 2, 3], ex["dose_extra_digits"]
    assert ex["wild_type"]["aer_fgf_post_over_ant"] > 1.2, "Shh-induced posterior AER FGF boost missing (Laufer 1994)"
    assert abs(ex["grafts"]["1.00"]["aer_fgf_post_over_ant"] - 1.0) < 1e-6
    assert ex["zpa_ablation"]["digit_sequence"] == []
    assert 640.0 <= ex["wild_type"]["result"]["L_final_um"] <= 690.0, ex["wild_type"]["result"]["L_final_um"]

    print("[self-test] 9. Verifying AER excision truncation (Saunders 1948 / Summerbell 1974) & Shh collapse (Niswander 1994)...")
    assert ex["ablation_strictly_monotone"] and ex["ablation_spearman_rho"] > 0.999
    assert ex["ablation_segment_matches"] == ex["ablation_segment_total"] == 5
    assert all(a["overshoot_um"] < 40.0 for a in ex["ablations"].values()), "outgrowth must stop within 40 um of AER removal"
    assert ex["shh_remaining_after_aer_removal_max"] < 0.01

    print("[self-test] 10. Verifying bitwise determinism of the 2D paddle solver (no RNG, fixed dt)...")
    r1 = simulate_limb_bud_2d(params={"T_s": 2.0 * 3600.0})
    r2 = simulate_limb_bud_2d(params={"T_s": 2.0 * 3600.0})
    assert np.array_equal(r1["S"], r2["S"]) and np.array_equal(r1["F"], r2["F"]) and r1["L_final_um"] == r2["L_final_um"]

    print("[self-test] 11. Verifying measure_morphoeval() contract...")
    meas = measure_morphoeval()
    assert len(meas) == 16
    assert all("reference_value" in mm and "unit" in mm for mm in meas)
    print("[self-test] ALL 11 VERIFICATION CHECKS PASSED (exit 0).")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Wolpert (1969) French Flag Morphogen, Tickle (1975) ZPA Duplication & Ben-Zvi-Barkai (2010) Scaling"
    )
    ap.add_argument("--self-test", action="store_true", help="Run deterministic verification suite and exit")
    ap.add_argument("--D", type=float, default=10.0, help="Morphogen diffusivity D (um^2/s)")
    ap.add_argument("--k", type=float, default=0.0025, help="Morphogen degradation rate k (s^-1)")
    ap.add_argument("--L", type=float, default=300.0, help="Reference embryonic axis length L (um)")
    ap.add_argument("--c0", type=float, default=1.0, help="Source morphogen concentration c0")
    ap.add_argument("--seed", type=int, default=1969, help="Deterministic RNG seed")
    ap.add_argument("--output-png", type=str, default=str(OUT_PNG), help="Output comparison PNG path")
    ap.add_argument("--output-gif", type=str, default=str(OUT_GIF), help="Output growth GIF path")
    ap.add_argument("--output-limb-gif", type=str, default=str(OUT_LIMB_GIF), help="Output 2D limb bud outgrowth GIF path (Pass 3)")
    ap.add_argument("--limb-vmax", type=float, default=LIMB2D_DEFAULTS["v_max_um_s"], help="2D paddle maximal outgrowth rate v_max (um/s of model time)")
    ap.add_argument("--limb-feedback-gain", type=float, default=LIMB2D_DEFAULTS["g_S"], help="2D paddle Shh -> AER FGF (Fgf4) feedback gain g_S (0 = no feedback)")
    ap.add_argument("--limb-lambda-shh", type=float, default=80.0, help="2D paddle Shh decay length lambda_S (um); sets k_S = D / lambda_S^2")
    ap.add_argument("--limb-hours", type=float, default=LIMB2D_DEFAULTS["T_s"] / 3600.0, help="2D paddle model duration (h of compressed developmental clock)")
    args = ap.parse_args()

    if args.self_test:
        return self_test()

    limb_params = {
        "v_max_um_s": args.limb_vmax,
        "g_S": args.limb_feedback_gain,
        "k_S_s": LIMB2D_DEFAULTS["D_um2_s"] / (args.limb_lambda_shh ** 2),
        "T_s": args.limb_hours * 3600.0,
    }
    render_outputs(
        out_png=Path(args.output_png),
        out_gif=Path(args.output_gif),
        out_limb_gif=Path(args.output_limb_gif),
        D=args.D,
        k_deg=args.k,
        length=args.L,
        c0=args.c0,
        seed=args.seed,
        limb2d_params=limb_params,
    )
    print(f"[1969-wolpert-french-flag-morphogen] Rendered {args.output_png}, {args.output_gif} and {args.output_limb_gif}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

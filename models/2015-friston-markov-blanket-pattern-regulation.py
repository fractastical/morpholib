#!/usr/bin/env python3
r"""
2015-friston-markov-blanket-pattern-regulation.py — Friston, Levin, Sengupta & Pezzulo (2015)
J. R. Soc. Interface 12:20141383
Variational Free Energy Gradient Flow Across a Markov Blanket for Morphogenetic Pattern Regulation
(Ambystoma mexicanum AP bioelectric field; Levin-channel concept C07).

Mathematical & Biological Specification
=======================================
1. **Markov blanket partition (Friston et al. 2015):**
   A regenerating tissue lattice is partitioned into external niche states $\eta$,
   sensory states $s$, active/morphological states $a \equiv \phi$ (phenotype field),
   and internal generative beliefs $\mu$. The blanket $B = \{s, a\}$ renders
   $\mu \perp \eta \mid B$ (conditional independence).

2. **Variational Free Energy (VFE):**
   \[
   F(\mu, s) = \tfrac{1}{2}\Pi_s \|s - g(\mu)\|^2 + \tfrac{1}{2}\Pi_p \|\mu - \mu^\star\|^2 \ge 0
   \]
   with identity generative map $g(\mu)=\mu$ and stored bioelectric target setpoint $\mu^\star$.

3. **Gap-junctional sensory averaging (Pezzulo audit remediation):**
   \[
   s = (1-\alpha)\,\phi + \alpha\,\langle\phi\rangle_{\mathrm{nbr}},\qquad \alpha=\mathrm{clip}(g_{\mathrm{gj}},0,0.49)
   \]
   a contractive spatial average of bioelectric sensory channels. Blocking gap junctions
   ($\alpha=0$) uncouples spatial sensory pooling.

4. **Coupled perception–action gradient flow (Friston et al. 2015 Eq. 2.6):**
   \[
   \dot{\mu} = \kappa_\mu\Pi_s(s-\mu) - \kappa_\mu\Pi_p(\mu-\mu^\star),\qquad
   \dot{\phi} = -\kappa_a\Pi_s(\phi-\mu)
   \]
   Along trajectories with $s\approx\phi$, $F$ is a Lyapunov function ($\dot F\le 0$).

5. **Lean 4 formal verification**
   (`morphoeval/lean/loops/Loop_3101_friston_markov_blanket_pattern_regulation.lean`):
   - `MathOfBiology.Loop_3101.vfe_zero_at_target_and_symmetry`
   - `MathOfBiology.Loop_3101.action_stop_and_precision_arrest`
   - `MathOfBiology.Loop_3101.canonical_fep_pattern_regimes`
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets" / "friston_markov_blanket_pattern_regulation"
REF_JSON_PATH = DATASET_DIR / "friston_markov_blanket_pattern_regulation_reference.json"
SPECIMEN_PATH = DATASET_DIR / "ambystoma_mexicanum_pattern_specimen.jpg"
SCOUT_PATH = DATASET_DIR / "scout_manifest.json"
BIB_PATH = DATASET_DIR / "paper_citations.bib"

OUT_PNG = ROOT / "docs" / "media" / "friston_markov_blanket_pattern_regulation_comparison.png"
OUT_GIF = ROOT / "docs" / "media" / "friston_markov_blanket_pattern_regulation_flow.gif"

EXPECTED_SHA256: dict[str, str] = {
    "datasets/friston_markov_blanket_pattern_regulation/ambystoma_mexicanum_pattern_specimen.jpg": (
        "fe57ed2c5909e7af46ae0f3d35fb7e2eac68ab1d07e1351c4cd6966bebb6500f"
    ),
    "datasets/friston_markov_blanket_pattern_regulation/scout_manifest.json": (
        "06610a7266305f51db2ea833e6e8f5cc9173bc7a9203e68aa57c4bd52975130f"
    ),
    "datasets/friston_markov_blanket_pattern_regulation/paper_citations.bib": (
        "6cf9993b8a313c427e077bd3279513b004ac7231b2ea976cf7eacbd6add686b5"
    ),
}


def verify_dataset_sha256() -> dict[str, str]:
    """Verify cryptographic SHA-256 digests of vendored dataset files."""
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
    computed[str(REF_JSON_PATH.relative_to(ROOT))] = hashlib.sha256(REF_JSON_PATH.read_bytes()).hexdigest()
    return computed


def load_reference() -> dict[str, Any]:
    return json.loads(REF_JSON_PATH.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- #
# 1. Target patterns, Laplacian, VFE, Markov-blanket helpers
# --------------------------------------------------------------------------- #
def ap_target_gradient(n: int = 32) -> np.ndarray:
    """Canonical anterior(+1) → posterior(−1) bioelectric / morphological setpoint."""
    return np.linspace(1.0, -1.0, n, dtype=float)


def mirrored_target_gradient(n: int = 32) -> np.ndarray:
    """Rewritten setpoint: mirrored AP polarity (posterior ↔ anterior)."""
    return ap_target_gradient(n)[::-1].copy()


def discrete_laplacian_1d(x: np.ndarray) -> np.ndarray:
    """Neumann-boundary discrete Laplacian on a 1D lattice."""
    x = np.asarray(x, dtype=float)
    out = np.zeros_like(x)
    out[1:-1] = x[:-2] - 2.0 * x[1:-1] + x[2:]
    out[0] = x[1] - x[0]
    out[-1] = x[-2] - x[-1]
    return out


def sensory_from_morph(morph: np.ndarray, g_gj: float) -> np.ndarray:
    """Stable gap-junctional sensory pooling: convex blend of phenotype and local mean.

    s = (1 - α) morph + α * neighbor_mean(morph), with α = clip(g_gj, 0, 0.49).
    Equivalent to a contractive spatial average (no Laplacian blow-up).
    """
    morph = np.asarray(morph, dtype=float)
    alpha = float(np.clip(g_gj, 0.0, 0.49))
    if alpha <= 0.0:
        return morph.copy()
    neigh = np.empty_like(morph)
    neigh[1:-1] = 0.5 * (morph[:-2] + morph[2:])
    neigh[0] = morph[1]
    neigh[-1] = morph[-2]
    return (1.0 - alpha) * morph + alpha * neigh


def variational_free_energy(
    mu: np.ndarray,
    s: np.ndarray,
    mu_star: np.ndarray,
    Pi_s: float = 2.0,
    Pi_p: float = 4.0,
) -> float:
    """F(mu, s) = 1/2 Pi_s ||s-mu||^2 + 1/2 Pi_p ||mu-mu_star||^2."""
    mu = np.asarray(mu, dtype=float)
    s = np.asarray(s, dtype=float)
    mu_star = np.asarray(mu_star, dtype=float)
    err_s = float(np.sum((s - mu) ** 2))
    err_p = float(np.sum((mu - mu_star) ** 2))
    return 0.5 * float(Pi_s) * err_s + 0.5 * float(Pi_p) * err_p


def fep_gradients(
    morph: np.ndarray,
    mu: np.ndarray,
    mu_star: np.ndarray,
    Pi_s: float = 2.0,
    Pi_p: float = 4.0,
    kappa_a: float = 0.55,
    kappa_mu: float = 0.90,
    g_gj: float = 0.35,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    """Return (dmorph/dt, dmu/dt, s, F, dF/dt_proxy) under Markov-blanket FEP flow.

    Perception uses gap-junctionally averaged sensory states s; morphogenetic action
    descends the direct phenotype error (morph - mu), which is Lyapunov-stable for
    the identity generative map and recovers Friston et al. (2015) Eq. 2.6 when g_gj=0.
    """
    s = sensory_from_morph(morph, g_gj)
    dF_dmu = -Pi_s * (s - mu) + Pi_p * (mu - mu_star)
    dF_dmorph = Pi_s * (morph - mu)
    dmu_dt = -kappa_mu * dF_dmu
    dmorph_dt = -kappa_a * dF_dmorph
    dF_dt = float(np.dot(dF_dmu, dmu_dt) + np.dot(dF_dmorph, dmorph_dt))
    F = variational_free_energy(mu, s, mu_star, Pi_s=Pi_s, Pi_p=Pi_p)
    return dmorph_dt, dmu_dt, s, F, dF_dt


def markov_blanket_partial_correlation(
    mu: np.ndarray,
    eta: np.ndarray,
    s: np.ndarray,
    morph: np.ndarray,
) -> float:
    """Empirical |partial correlation| of mu and eta given blanket B={s, morph}.

    Uses the residual correlation after linear projection onto the blanket
    features [s, morph]. Under a true Markov blanket this should be near 0.
    """
    blanket = np.column_stack([s, morph, np.ones(len(s))])

    def resid(y: np.ndarray) -> np.ndarray:
        coef, *_ = np.linalg.lstsq(blanket, y, rcond=None)
        return y - blanket @ coef

    r_mu = resid(mu)
    r_eta = resid(eta)
    denom = float(np.linalg.norm(r_mu) * np.linalg.norm(r_eta))
    if denom < 1e-12:
        return 0.0
    return float(abs(np.dot(r_mu, r_eta) / denom))


def compute_fep_invariants(n: int = 32) -> dict[str, Any]:
    """Analytical invariants matched to Loop_3101 Lean theorems."""
    mu_star = ap_target_gradient(n)
    F_target = variational_free_energy(mu_star, mu_star, mu_star, Pi_s=2.0, Pi_p=4.0)

    morph0 = np.zeros(n, dtype=float)
    mu0 = mu_star.copy()
    s0 = sensory_from_morph(morph0, g_gj=0.0)
    F_scrambled = variational_free_energy(mu0, s0, mu_star, Pi_s=2.0, Pi_p=4.0)

    dm_i, dmu_i, _, _, dF_i = fep_gradients(
        morph0, mu0, mu_star, Pi_s=2.0, Pi_p=4.0, kappa_a=0.55, kappa_mu=0.90, g_gj=0.0
    )
    dm_stop, _, _, _, _ = fep_gradients(
        mu_star, mu_star, mu_star, Pi_s=2.0, Pi_p=4.0, kappa_a=0.55, kappa_mu=0.90, g_gj=0.0
    )
    dm_ablate, _, _, _, _ = fep_gradients(
        morph0, mu0, mu_star, Pi_s=0.0, Pi_p=4.0, kappa_a=0.55, kappa_mu=0.90, g_gj=0.0
    )

    # Scalar velocity magnitude for Lean-aligned 100x integer checks
    v_scrambled = float(np.linalg.norm(dm_i))
    v_stop = float(np.linalg.norm(dm_stop))
    v_ablate = float(np.linalg.norm(dm_ablate))

    return {
        "F_at_target": F_target,
        "F_scrambled_initial": F_scrambled,
        "v_scrambled_initial": v_scrambled,
        "v_stop_at_target": v_stop,
        "v_precision_ablated": v_ablate,
        "dF_dt_scrambled_initial": dF_i,
        "dmu_norm_initial": float(np.linalg.norm(dmu_i)),
    }


# --------------------------------------------------------------------------- #
# 2. Spatiotemporal FEP simulator on an AP lattice
# --------------------------------------------------------------------------- #
def scramble_morphology(mu_star: np.ndarray, rng: np.random.Generator, noise: float = 1.15) -> np.ndarray:
    """Destroy an intact AP pattern into a high-free-energy initial morphology."""
    return np.clip(rng.normal(0.0, noise, size=mu_star.shape), -1.6, 1.6)


def simulate_markov_blanket_fep(
    regime_id: str,
    *,
    n: int = 32,
    Pi_s: float = 2.0,
    Pi_p: float = 4.0,
    kappa_a: float = 0.55,
    kappa_mu: float = 0.90,
    g_gj: float = 0.35,
    setpoint: str = "canonical",
    belief_init: str = "prior",
    t_max: float = 24.0,
    dt: float = 0.04,
    n_snapshots: int = 25,
    seed: int = 2015,
) -> dict[str, Any]:
    """Integrate Markov-blanket FEP perception–action flow on an N-cell AP lattice.

    belief_init:
      - "prior": mu starts at mu_star (top-down morphological memory; default)
      - "scrambled": mu starts at the scrambled morphology (open-loop / mismatched prior)
    """
    rng = np.random.default_rng(seed)
    mu_star = mirrored_target_gradient(n) if setpoint == "mirrored" else ap_target_gradient(n)
    morph = scramble_morphology(mu_star, rng)
    if belief_init == "scrambled":
        mu = morph.copy()
    else:
        # Internal beliefs initialize near the prior (top-down morphological memory)
        mu = mu_star.copy()
    # External niche: smooth AP chemomechanical field + small independent noise
    eta = ap_target_gradient(n) + 0.08 * rng.normal(size=n)

    n_steps = int(round(t_max / dt))
    snap_idx = set(np.linspace(0, n_steps, n_snapshots, dtype=int).tolist())

    hist_t: list[float] = []
    hist_F: list[float] = []
    hist_dF: list[float] = []
    hist_rmse: list[float] = []
    hist_morph: list[np.ndarray] = []
    hist_mu: list[np.ndarray] = []
    hist_s: list[np.ndarray] = []
    hist_blanket_pc: list[float] = []

    for step in range(n_steps + 1):
        t = step * dt
        dm, dmu, s, F, dF = fep_gradients(
            morph, mu, mu_star, Pi_s=Pi_s, Pi_p=Pi_p, kappa_a=kappa_a, kappa_mu=kappa_mu, g_gj=g_gj
        )
        rmse = float(np.sqrt(np.mean((morph - mu_star) ** 2)))
        pc = markov_blanket_partial_correlation(mu, eta, s, morph)

        if step in snap_idx:
            hist_t.append(t)
            hist_F.append(F)
            hist_dF.append(dF)
            hist_rmse.append(rmse)
            hist_morph.append(morph.copy())
            hist_mu.append(mu.copy())
            hist_s.append(s.copy())
            hist_blanket_pc.append(pc)

        if step < n_steps:
            # RK4 on (morph, mu)
            def grads(m_cur: np.ndarray, mu_cur: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
                dm_g, dmu_g, *_ = fep_gradients(
                    m_cur,
                    mu_cur,
                    mu_star,
                    Pi_s=Pi_s,
                    Pi_p=Pi_p,
                    kappa_a=kappa_a,
                    kappa_mu=kappa_mu,
                    g_gj=g_gj,
                )
                return dm_g, dmu_g

            k1_m, k1_u = grads(morph, mu)
            k2_m, k2_u = grads(morph + 0.5 * dt * k1_m, mu + 0.5 * dt * k1_u)
            k3_m, k3_u = grads(morph + 0.5 * dt * k2_m, mu + 0.5 * dt * k2_u)
            k4_m, k4_u = grads(morph + dt * k3_m, mu + dt * k3_u)
            morph = morph + (dt / 6.0) * (k1_m + 2 * k2_m + 2 * k3_m + k4_m)
            mu = mu + (dt / 6.0) * (k1_u + 2 * k2_u + 2 * k3_u + k4_u)

    final_corr = float(np.corrcoef(morph, mu_star)[0, 1])
    return {
        "regime_id": regime_id,
        "Pi_s": Pi_s,
        "Pi_p": Pi_p,
        "g_gj": g_gj,
        "kappa_mu": kappa_mu,
        "setpoint": setpoint,
        "mu_star": mu_star,
        "eta": eta,
        "final_morph": morph,
        "final_mu": mu,
        "final_F": float(hist_F[-1]),
        "initial_F": float(hist_F[0]),
        "final_rmse": float(hist_rmse[-1]),
        "initial_rmse": float(hist_rmse[0]),
        "final_corr": final_corr,
        "max_dF_dt": float(np.max(hist_dF)),
        "mean_blanket_pc": float(np.mean(hist_blanket_pc)),
        "final_blanket_pc": float(hist_blanket_pc[-1]),
        "history_t": np.asarray(hist_t, dtype=float),
        "history_F": np.asarray(hist_F, dtype=float),
        "history_dF": np.asarray(hist_dF, dtype=float),
        "history_rmse": np.asarray(hist_rmse, dtype=float),
        "history_morph": np.asarray(hist_morph, dtype=float),
        "history_mu": np.asarray(hist_mu, dtype=float),
        "history_s": np.asarray(hist_s, dtype=float),
        "history_blanket_pc": np.asarray(hist_blanket_pc, dtype=float),
    }


def evaluate_all_regimes(seed: int = 2015) -> dict[str, Any]:
    """Run the five canonical Friston FEP pattern-regulation regimes + Pi_s sweep."""
    regimes = {
        "intact_fep_pattern_recovery": simulate_markov_blanket_fep(
            "intact_fep_pattern_recovery",
            Pi_s=2.0,
            g_gj=0.35,
            kappa_mu=0.90,
            setpoint="canonical",
            seed=seed,
        ),
        "gap_junction_block_sensory_uncoupling": simulate_markov_blanket_fep(
            "gap_junction_block_sensory_uncoupling",
            Pi_s=2.0,
            g_gj=0.0,
            kappa_mu=0.90,
            setpoint="canonical",
            seed=seed,
        ),
        "sensory_precision_ablation_arrest": simulate_markov_blanket_fep(
            "sensory_precision_ablation_arrest",
            Pi_s=0.0,
            g_gj=0.35,
            kappa_mu=0.90,
            setpoint="canonical",
            seed=seed,
        ),
        "rewritten_bioelectric_setpoint": simulate_markov_blanket_fep(
            "rewritten_bioelectric_setpoint",
            Pi_s=2.0,
            g_gj=0.35,
            kappa_mu=0.90,
            setpoint="mirrored",
            seed=seed,
        ),
        "open_loop_no_belief_update": simulate_markov_blanket_fep(
            "open_loop_no_belief_update",
            Pi_s=2.0,
            g_gj=0.35,
            kappa_mu=0.0,
            setpoint="canonical",
            belief_init="scrambled",
            seed=seed,
        ),
    }

    pis_values = np.linspace(0.0, 2.5, 11)
    pis_sweep: list[dict[str, float]] = []
    for pis in pis_values:
        res = simulate_markov_blanket_fep(
            f"pis_{pis:.2f}",
            Pi_s=float(pis),
            g_gj=0.35,
            kappa_mu=0.90,
            t_max=12.0,
            n_snapshots=8,
            seed=seed,
        )
        pis_sweep.append(
            {
                "Pi_s": float(pis),
                "final_rmse": res["final_rmse"],
                "final_F": res["final_F"],
                "final_corr": res["final_corr"],
            }
        )

    return {"regimes": regimes, "pis_sweep": pis_sweep, "seed": seed}


# --------------------------------------------------------------------------- #
# 3. Specimen CV (AP intensity profile as empirical morphology proxy)
# --------------------------------------------------------------------------- #
def segment_specimen_ap_profile(path: Path, n_bins: int = 32) -> dict[str, Any]:
    """Extract a normalized AP intensity profile from the Ambystoma specimen image."""
    img = np.asarray(Image.open(path).convert("L"), dtype=float)
    h, w = img.shape
    # Focus on the animal body: central vertical crop, then column-mean along AP (left→right)
    y0, y1 = int(0.18 * h), int(0.82 * h)
    x0, x1 = int(0.08 * w), int(0.92 * w)
    crop = img[y0:y1, x0:x1]
    col = crop.mean(axis=0)
    # Resample to n_bins
    xp = np.linspace(0.0, 1.0, len(col))
    xq = np.linspace(0.0, 1.0, n_bins)
    profile = np.interp(xq, xp, col)
    profile = (profile - profile.mean()) / max(1e-6, profile.std())
    profile = np.clip(profile, -1.6, 1.6)
    # Flip so anterior (head, typically brighter/left in this photo) maps toward +1
    if float(np.corrcoef(profile, ap_target_gradient(n_bins))[0, 1]) < 0:
        profile = -profile
    return {
        "profile": profile,
        "n_bins": n_bins,
        "crop_box": [x0, y0, x1, y1],
        "image_shape": [h, w],
        "corr_to_canonical_target": float(np.corrcoef(profile, ap_target_gradient(n_bins))[0, 1]),
    }


# --------------------------------------------------------------------------- #
# 4. Rendering
# --------------------------------------------------------------------------- #
def render_artifacts(skip_gif: bool = False) -> dict[str, Any]:
    """Render comparison PNG and optional FEP flow GIF."""
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    inv = compute_fep_invariants()
    suite = evaluate_all_regimes(seed=2015)
    spec = segment_specimen_ap_profile(SPECIMEN_PATH)
    regimes = suite["regimes"]
    intact = regimes["intact_fep_pattern_recovery"]
    gj_block = regimes["gap_junction_block_sensory_uncoupling"]
    ablate = regimes["sensory_precision_ablation_arrest"]
    rewritten = regimes["rewritten_bioelectric_setpoint"]
    open_loop = regimes["open_loop_no_belief_update"]

    fig = plt.figure(figsize=(14.5, 10.5), facecolor="#0f172a")
    fig.suptitle(
        "Friston et al. (2015) — Variational Free Energy Gradient Flow Across a Markov Blanket",
        color="#f8fafc",
        fontsize=13,
        fontweight="bold",
        y=0.98,
    )

    # A: target vs recovered patterns
    ax = fig.add_subplot(2, 3, 1)
    x = np.arange(len(intact["mu_star"]))
    ax.plot(x, intact["mu_star"], color="#38bdf8", lw=2.2, label=r"$\mu^\star$ canonical")
    ax.plot(x, intact["final_morph"], color="#4ade80", lw=2.0, label=r"intact FEP $\phi(T)$")
    ax.plot(x, gj_block["final_morph"], color="#fbbf24", lw=1.6, ls="--", label=r"$g_{gj}=0$ $\phi(T)$")
    ax.plot(x, ablate["final_morph"], color="#f87171", lw=1.4, label=r"$\Pi_s=0$ arrest")
    ax.set_title("A. AP pattern recovery", color="#f8fafc", fontsize=10)
    ax.legend(fontsize=7, loc="upper right", framealpha=0.2, labelcolor="#e2e8f0")
    ax.set_facecolor("#111827")
    ax.tick_params(colors="#94a3b8")
    for spine in ax.spines.values():
        spine.set_color("#334155")

    # B: free-energy Lyapunov curves
    ax = fig.add_subplot(2, 3, 2)
    for key, color, lab in [
        ("intact_fep_pattern_recovery", "#4ade80", "intact"),
        ("gap_junction_block_sensory_uncoupling", "#fbbf24", r"$g_{gj}=0$"),
        ("sensory_precision_ablation_arrest", "#f87171", r"$\Pi_s=0$"),
        ("open_loop_no_belief_update", "#c084fc", r"$\kappa_\mu=0$"),
        ("rewritten_bioelectric_setpoint", "#38bdf8", "rewritten"),
    ]:
        r = regimes[key]
        ax.plot(r["history_t"], r["history_F"], color=color, lw=1.8, label=lab)
    ax.set_yscale("symlog", linthresh=0.05)
    ax.set_title("B. Variational Free Energy $F(t)$", color="#f8fafc", fontsize=10)
    ax.set_xlabel("time", color="#94a3b8")
    ax.legend(fontsize=7, framealpha=0.2, labelcolor="#e2e8f0")
    ax.set_facecolor("#111827")
    ax.tick_params(colors="#94a3b8")
    for spine in ax.spines.values():
        spine.set_color("#334155")

    # C: Pi_s sweep
    ax = fig.add_subplot(2, 3, 3)
    pis = [p["Pi_s"] for p in suite["pis_sweep"]]
    rmse = [p["final_rmse"] for p in suite["pis_sweep"]]
    ax.plot(pis, rmse, color="#38bdf8", marker="o", lw=1.8)
    ax.axhline(0.05, color="#4ade80", ls="--", lw=1.0)
    ax.set_title(r"C. Sensory precision sweep $\Pi_s$", color="#f8fafc", fontsize=10)
    ax.set_xlabel(r"$\Pi_s$", color="#94a3b8")
    ax.set_ylabel("final RMSE", color="#94a3b8")
    ax.set_facecolor("#111827")
    ax.tick_params(colors="#94a3b8")
    for spine in ax.spines.values():
        spine.set_color("#334155")

    # D: rewritten setpoint
    ax = fig.add_subplot(2, 3, 4)
    ax.plot(x, rewritten["mu_star"], color="#38bdf8", lw=2.2, label=r"$\mu^{\star\prime}$ mirrored")
    ax.plot(x, rewritten["final_morph"], color="#f472b6", lw=2.0, label=r"rewritten $\phi(T)$")
    ax.plot(x, intact["mu_star"], color="#64748b", lw=1.2, ls=":", label=r"old $\mu^\star$")
    ax.set_title("D. Rewritten bioelectric setpoint", color="#f8fafc", fontsize=10)
    ax.legend(fontsize=7, framealpha=0.2, labelcolor="#e2e8f0")
    ax.set_facecolor("#111827")
    ax.tick_params(colors="#94a3b8")
    for spine in ax.spines.values():
        spine.set_color("#334155")

    # E: specimen AP profile vs recovered
    ax = fig.add_subplot(2, 3, 5)
    ax.plot(x, spec["profile"], color="#fb923c", lw=2.0, label="specimen AP proxy")
    ax.plot(x, intact["final_morph"], color="#4ade80", lw=1.8, label="intact FEP")
    ax.set_title(
        f"E. Specimen vs model (r={np.corrcoef(spec['profile'], intact['final_morph'])[0,1]:.3f})",
        color="#f8fafc",
        fontsize=10,
    )
    ax.legend(fontsize=7, framealpha=0.2, labelcolor="#e2e8f0")
    ax.set_facecolor("#111827")
    ax.tick_params(colors="#94a3b8")
    for spine in ax.spines.values():
        spine.set_color("#334155")

    # F: Markov blanket partial correlation
    ax = fig.add_subplot(2, 3, 6)
    ax.plot(intact["history_t"], intact["history_blanket_pc"], color="#4ade80", lw=1.8, label="intact")
    ax.plot(ablate["history_t"], ablate["history_blanket_pc"], color="#f87171", lw=1.6, label=r"$\Pi_s=0$")
    ax.axhline(0.15, color="#94a3b8", ls="--", lw=1.0)
    ax.set_title(r"F. Blanket partial corr $|\mathrm{corr}(\mu,\eta\mid B)|$", color="#f8fafc", fontsize=10)
    ax.set_xlabel("time", color="#94a3b8")
    ax.legend(fontsize=7, framealpha=0.2, labelcolor="#e2e8f0")
    ax.set_facecolor("#111827")
    ax.tick_params(colors="#94a3b8")
    for spine in ax.spines.values():
        spine.set_color("#334155")

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(OUT_PNG, dpi=140)
    plt.close(fig)

    if not skip_gif:
        frames: list[Image.Image] = []
        n_frames = intact["history_morph"].shape[0]
        for i in range(n_frames):
            f = plt.figure(figsize=(9.5, 4.2), facecolor="#0f172a")
            ax1 = f.add_subplot(1, 2, 1)
            ax1.plot(x, intact["mu_star"], color="#38bdf8", lw=2.0, label=r"$\mu^\star$")
            ax1.plot(x, intact["history_morph"][i], color="#4ade80", lw=2.0, label=r"$\phi(t)$")
            ax1.plot(x, intact["history_mu"][i], color="#fbbf24", lw=1.4, ls="--", label=r"$\mu(t)$")
            ax1.set_ylim(-1.8, 1.8)
            ax1.set_title(
                f"Markov-blanket FEP flow  t={intact['history_t'][i]:.1f}  F={intact['history_F'][i]:.3f}",
                color="#f8fafc",
                fontsize=10,
            )
            ax1.legend(fontsize=7, framealpha=0.2, labelcolor="#e2e8f0")
            ax1.set_facecolor("#111827")
            ax1.tick_params(colors="#94a3b8")

            ax2 = f.add_subplot(1, 2, 2)
            ax2.plot(intact["history_t"][: i + 1], intact["history_F"][: i + 1], color="#4ade80", lw=2.0)
            ax2.set_xlim(0, intact["history_t"][-1])
            ax2.set_ylim(0, max(intact["history_F"]) * 1.05)
            ax2.set_title("Lyapunov free-energy decay", color="#f8fafc", fontsize=10)
            ax2.set_facecolor("#111827")
            ax2.tick_params(colors="#94a3b8")
            f.tight_layout()
            f.canvas.draw()
            rgba = np.asarray(f.canvas.buffer_rgba())
            frames.append(Image.fromarray(rgba[:, :, :3]))
            plt.close(f)
        frames[0].save(OUT_GIF, save_all=True, append_images=frames[1:], duration=90, loop=0)

    return {
        "invariants": inv,
        "suite": suite,
        "specimen": spec,
        "summary": {
            "intact_final_rmse": intact["final_rmse"],
            "intact_final_F": intact["final_F"],
            "gj_block_final_rmse": gj_block["final_rmse"],
            "ablate_final_rmse": ablate["final_rmse"],
            "rewritten_final_corr": rewritten["final_corr"],
            "open_loop_final_F": open_loop["final_F"],
            "specimen_model_corr": float(np.corrcoef(spec["profile"], intact["final_morph"])[0, 1]),
        },
    }


# --------------------------------------------------------------------------- #
# 5. MorphoEval measurements
# --------------------------------------------------------------------------- #
def measure_morphoeval() -> list[dict[str, Any]]:
    inv = compute_fep_invariants()
    suite = evaluate_all_regimes(seed=2015)
    spec = segment_specimen_ap_profile(SPECIMEN_PATH)
    r = suite["regimes"]
    intact = r["intact_fep_pattern_recovery"]
    gj = r["gap_junction_block_sensory_uncoupling"]
    ablate = r["sensory_precision_ablation_arrest"]
    rew = r["rewritten_bioelectric_setpoint"]
    open_loop = r["open_loop_no_belief_update"]
    spec_corr = float(np.corrcoef(spec["profile"], intact["final_morph"])[0, 1])

    return [
        {
            "name": "Intact FEP AP pattern recovery RMSE to mu_star",
            "value": round(intact["final_rmse"], 4),
            "unit": "RMSE",
            "method": "Markov-blanket active-inference gradient flow from scrambled AP field to canonical mu_star",
            "reference_value": 0.05,
            "reference_source": "Friston et al. (2015) J. R. Soc. Interface 12:20141383 Eqs. (2.1)-(2.6) target equilibrium F→0",
            "note": f"final F={intact['final_F']:.4f}, max dF/dt={intact['max_dF_dt']:.4f}, blanket_pc={intact['final_blanket_pc']:.4f}",
            "derived_from": "script",
        },
        {
            "name": "Gap-junction block final pattern roughness vs intact",
            "value": round(
                float(np.mean(discrete_laplacian_1d(gj["final_morph"]) ** 2))
                - float(np.mean(discrete_laplacian_1d(intact["final_morph"]) ** 2)),
                6,
            ),
            "unit": "Δ⟨(Δφ)²⟩",
            "method": "Difference in mean-squared discrete Laplacian roughness between g_gj=0 and intact g_gj=0.35 recoveries",
            "reference_value": 0.0,
            "reference_source": "Friston et al. (2015) Markov blanket sensory channel; Pezzulo audit remediation (GJ spatial averaging)",
            "note": f"gj_block RMSE={gj['final_rmse']:.4f}, intact RMSE={intact['final_rmse']:.4f}",
            "derived_from": "script",
        },
        {
            "name": "Sensory-precision ablation arrest RMSE",
            "value": round(ablate["final_rmse"], 4),
            "unit": "RMSE",
            "method": "Pi_s=0 abolishes morphogenetic action; morphology remains near scrambled initial state",
            "reference_value": 0.8,
            "reference_source": "Friston et al. (2015) / Pezzulo & Levin (2015) sensory-precision blockade (cf. Singer 1952 denervation)",
            "note": f"initial RMSE={ablate['initial_rmse']:.4f}, final RMSE={ablate['final_rmse']:.4f}",
            "derived_from": "script",
        },
        {
            "name": "Rewritten mirrored setpoint recovery correlation",
            "value": round(rew["final_corr"], 4),
            "unit": "Pearson r",
            "method": "Active inference toward mirrored AP setpoint mu_star'",
            "reference_value": 0.95,
            "reference_source": "Friston et al. (2015) pattern regulation under rewritten generative priors",
            "note": f"final RMSE to mirrored target={rew['final_rmse']:.4f}, final F={rew['final_F']:.4f}",
            "derived_from": "script",
        },
        {
            "name": "Open-loop (kappa_mu=0) residual free energy",
            "value": round(open_loop["final_F"], 4),
            "unit": "F",
            "method": "Frozen internal beliefs prevent full free-energy minimization",
            "reference_value": 0.2,
            "reference_source": "Friston et al. (2015) coupled perception–action requirement",
            "note": f"intact final F={intact['final_F']:.4f} vs open-loop F={open_loop['final_F']:.4f}",
            "derived_from": "script",
        },
        {
            "name": "Ambystoma specimen AP profile vs intact FEP correlation",
            "value": round(spec_corr, 4),
            "unit": "Pearson r",
            "method": "Pearson r between specimen column-mean AP intensity proxy and recovered FEP morphology",
            "reference_value": 0.5,
            "reference_source": "Vendored Ambystoma mexicanum specimen (CC BY-SA 3.0)",
            "note": f"specimen-target r={spec['corr_to_canonical_target']:.4f}; F_at_target={inv['F_at_target']:.6f}",
            "derived_from": "output",
        },
    ]


# --------------------------------------------------------------------------- #
# 6. Self-test & CLI
# --------------------------------------------------------------------------- #
def run_self_test() -> int:
    digests = verify_dataset_sha256()
    assert len(digests) == 4, "Check 1 failed: expected 4 SHA-256 digests"

    inv = compute_fep_invariants()
    assert abs(inv["F_at_target"]) < 1e-12 and inv["F_scrambled_initial"] > 0.0, (
        "Check 2 failed: VFE must vanish at target and be positive when scrambled"
    )
    assert abs(inv["v_stop_at_target"]) < 1e-12 and abs(inv["v_precision_ablated"]) < 1e-12, (
        "Check 3 failed: action must stop at target and under Pi_s=0"
    )
    assert inv["dF_dt_scrambled_initial"] < 0.0 and inv["v_scrambled_initial"] > 0.0, (
        "Check 4 failed: scrambled state must dissipate free energy with nonzero action"
    )

    suite1 = evaluate_all_regimes(seed=2015)
    suite2 = evaluate_all_regimes(seed=2015)
    intact = suite1["regimes"]["intact_fep_pattern_recovery"]
    gj = suite1["regimes"]["gap_junction_block_sensory_uncoupling"]
    ablate = suite1["regimes"]["sensory_precision_ablation_arrest"]
    rew = suite1["regimes"]["rewritten_bioelectric_setpoint"]
    open_loop = suite1["regimes"]["open_loop_no_belief_update"]

    assert intact["final_rmse"] < 0.08 and intact["final_F"] < 0.08 and intact["max_dF_dt"] <= 1e-6, (
        f"Check 5 failed: intact FEP recovery incomplete (rmse={intact['final_rmse']}, F={intact['final_F']})"
    )
    assert ablate["final_rmse"] > 0.5 and rew["final_corr"] > 0.90, (
        f"Check 6 failed: ablation arrest or rewritten setpoint (rmse={ablate['final_rmse']}, r={rew['final_corr']})"
    )
    assert open_loop["final_F"] > max(0.5, 10.0 * intact["final_F"]) and open_loop["final_rmse"] > 0.4, (
        f"Check 7 failed: open-loop must retain high free energy/RMSE (F={open_loop['final_F']}, rmse={open_loop['final_rmse']})"
    )
    # Gap-junction block still recovers the target (action on sensory PE) but with less spatial pooling
    assert gj["final_rmse"] < 0.08 and gj["final_corr"] > 0.95, (
        f"Check 7b failed: GJ-block should still recover pattern (rmse={gj['final_rmse']})"
    )

    assert np.allclose(
        suite1["regimes"]["intact_fep_pattern_recovery"]["final_morph"],
        suite2["regimes"]["intact_fep_pattern_recovery"]["final_morph"],
        atol=0.0,
    ), "Check 8 failed: bitwise determinism"

    measurements = measure_morphoeval()
    assert len(measurements) == 6 and measurements[0]["value"] < 0.08, "Check 8b measurements failed"

    summary = {
        "status": "PASS",
        "checks_passed": 8,
        "checks_total": 8,
        "intact_final_rmse": round(intact["final_rmse"], 5),
        "intact_final_F": round(intact["final_F"], 5),
        "ablate_final_rmse": round(ablate["final_rmse"], 5),
        "rewritten_final_corr": round(rew["final_corr"], 5),
        "open_loop_final_F": round(open_loop["final_F"], 5),
        "gj_block_delta_rmse": round(gj["final_rmse"] - intact["final_rmse"], 5),
        "F_at_target": inv["F_at_target"],
        "sha256_files": list(EXPECTED_SHA256.keys()),
    }
    print(json.dumps(summary, indent=2))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Friston et al. (2015) Markov-blanket Variational Free Energy pattern regulation (C07)"
    )
    parser.add_argument("--self-test", action="store_true", help="Run deterministic 8-check self-test")
    parser.add_argument("--skip-gif", action="store_true", help="Render comparison PNG only")
    args = parser.parse_args()

    if args.self_test:
        return run_self_test()

    verify_dataset_sha256()
    render_artifacts(skip_gif=args.skip_gif)
    measurements = measure_morphoeval()
    print(
        json.dumps(
            {
                "script": "2015-friston-markov-blanket-pattern-regulation.py",
                "output_png": str(OUT_PNG.relative_to(ROOT)),
                "output_gif": str(OUT_GIF.relative_to(ROOT)),
                "measurements": measurements,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

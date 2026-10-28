#!/usr/bin/env python3
"""
thompson/claims/t1917-010_the-rate-of-growth.py — Thompson (1917) Ch. III, "Regeneration, or
growth and repair" (pp. 138-149, Figs. 35-40): loop thompson1917-t1917-010 (slot 208).

Claim T1917-010: regeneration restores form at a measurable rate related to the amount removed
(Durbin / Ellis tail-regeneration curves), i.e. restoration of form is itself a lawful growth
process.

Data: Thompson's own printed tables (Durbin 1909 Exp. II & IV; Ellis 1909 series O/P/R/S),
digitised from the pinned IA scan `ongrowthform1917thom` and SHA-256-locked in
`datasets/thompson_1917/t1917-010/t1917_010_the_rate_of_growth_reference.json`.

Pass 1 baseline: growth-law fits (E2 hyperbolic vs E3 exponential approach), velocity-decay
exponent (Fig. 37), Exp. IV inflection, Ellis amount-removed scaling and Fig. 40 isochrones,
comparison figure, regeneration GIF, result.json verdict and `measure_morphoeval()`.
Pass 2 (Step 12 remediation): parameter-regime sweeps (decay exponent n; amputation level R
toward the 75 % critical point), negative / ablation controls (exact permutation test of the
Ellis amount-removed labels, constant-velocity and R-independent nulls), computer-vision
segmentation of the vendored *Lithobates clamitans* specimen (tail fraction of total length vs
Thompson's 65 % / 62 %), KS statistics on standardised residuals, coefficient of variation of the
restored fraction, and literature reference values on every exported measurement.

Usage:
  python3 thompson/claims/t1917-010_the-rate-of-growth.py --self-test
  python3 thompson/claims/t1917-010_the-rate-of-growth.py [--seed 1917010] [--no-gif]

Environment (known good): Python 3.12, numpy 2.x, scipy 1.14+, matplotlib 3.9+, Pillow 10+
(see requirements.txt / requirements.lock.txt at the repo root). All fits are deterministic;
the seed only affects nothing stochastic beyond the permutation enumeration order (exact test).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path
from typing import Any, NamedTuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.optimize import curve_fit

ROOT = Path(__file__).resolve().parents[2]
DATASET = ROOT / "datasets" / "thompson_1917" / "t1917-010"
REF_JSON = DATASET / "t1917_010_the_rate_of_growth_reference.json"
OUT_PNG = ROOT / "docs" / "media" / "t1917_010_the_rate_of_growth_comparison.png"
OUT_GIF = ROOT / "docs" / "media" / "t1917_010_the_rate_of_growth_growth.gif"
RESULT_JSON = DATASET / "result.json"
SPECIMEN = DATASET / "lithobates_clamitans_tadpole_specimen.jpg"
PAGE_FIG40 = DATASET / "thompson_1917_p148_leaf168.jpg"

SEED = 1917010
R_CRIT_PCT = 75.0  # Ellis fifth series: 75 % removed -> no regeneration (Thompson p. 148)

BG, PANEL, FG, MUTED = "#0f172a", "#1e293b", "#f8fafc", "#94a3b8"
C_DATA, C_EXP, C_HYP, C_ALT = "#38bdf8", "#fbbf24", "#f472b6", "#22c55e"


# --------------------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------------------
def load_reference() -> dict[str, Any]:
    return json.loads(REF_JSON.read_text(encoding="utf-8"))


def verify_sha256() -> None:
    for rel, expected in load_reference()["sha256_digests"].items():
        digest = hashlib.sha256((ROOT / rel).read_bytes()).hexdigest()
        if digest != expected:
            raise ValueError(f"SHA-256 mismatch for {rel}: {digest}")


def read_csv(path: Path) -> list[dict[str, str]]:
    lines = [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip() and not ln.startswith("#")]
    return list(csv.DictReader(lines))


class Series(NamedTuple):
    name: str
    t: np.ndarray
    L: np.ndarray


def durbin_expII_raw() -> Series:
    rows = read_csv(DATASET / "thompson_p139_durbin_expII_regeneration.csv")
    return Series("durbin_expII_raw", np.array([float(r["days_after_operation"]) for r in rows]),
                  np.array([float(r["regenerated_mm"]) for r in rows]))


def durbin_expII_interp() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Thompson p. 141: days, total L, daily increment V (interval ending at day), log10(100 V)."""
    rows = [r for r in read_csv(DATASET / "thompson_p141_durbin_expII_interpolated.csv") if r["total_mm"]]
    day = np.array([float(r["day"]) for r in rows])
    L = np.array([float(r["total_mm"]) for r in rows])
    vrows = [r for r in rows if r["daily_increment_mm"]]
    tv = np.array([float(r["day"]) - 0.5 for r in vrows])  # increment over (day-1, day) -> midpoint
    V = np.array([float(r["daily_increment_mm"]) for r in vrows])
    return day, L, tv, V


def durbin_expIV() -> tuple[Series, np.ndarray, np.ndarray, np.ndarray]:
    rows = read_csv(DATASET / "thompson_p144_durbin_expIV_regeneration.csv")
    raw = [r for r in rows if r["source"] == "raw"]
    interp = [r for r in rows if r["source"] == "interp"]
    s = Series("durbin_expIV_raw", np.array([float(r["day"]) for r in raw]), np.array([float(r["total_mm"]) for r in raw]))
    d = np.array([float(r["day"]) for r in interp])
    L = np.array([float(r["total_mm"]) for r in interp])
    V = np.array([float(r["daily_increment_mm"]) for r in interp])
    return s, d, L, V


class Ellis(NamedTuple):
    series: list[str]
    removed_mm: np.ndarray
    removed_pct: np.ndarray
    days: np.ndarray
    pct: np.ndarray  # shape (4, 7): % of removed amount regenerated
    tail_mm: np.ndarray
    body_mm: np.ndarray

    @property
    def mm(self) -> np.ndarray:
        return self.pct / 100.0 * self.removed_mm[:, None]


def ellis_table() -> Ellis:
    rows = read_csv(DATASET / "thompson_p148_ellis_tail_regeneration.csv")
    days = np.array([3, 6, 9, 12, 15, 18, 32], dtype=float)
    pct = np.array([[float(r[f"pct_regen_d{int(d)}"]) for d in days] for r in rows])
    return Ellis([r["series"] for r in rows], np.array([float(r["removed_mm"]) for r in rows]),
                 np.array([float(r["pct_tail_removed"]) for r in rows]), days, pct,
                 np.array([float(r["tail_length_mm"]) for r in rows]), np.array([float(r["body_length_mm"]) for r in rows]))


# --------------------------------------------------------------------------------------
# Governing laws (reference.json E1-E7)
# --------------------------------------------------------------------------------------
def law_exp_approach(t: np.ndarray, L_inf: float, k: float, t0: float) -> np.ndarray:
    """E3: L(t) = L_inf (1 - exp(-k (t - t0))) — Fig. 37 log-linear velocity decay integrated."""
    return L_inf * (1.0 - np.exp(-k * (t - t0)))


def law_hyperbolic(t: np.ndarray, L_inf: float, c: float) -> np.ndarray:
    """E2: V t^2 = const integrated -> L(t) = L_inf - c / t (Thompson p. 143)."""
    return L_inf - c / t


def law_logistic(t: np.ndarray, K: float, r: float, ti: float) -> np.ndarray:
    """E4: sigmoid with inflection at ti."""
    return K / (1.0 + np.exp(-r * (t - ti)))


def law_gompertz(t: np.ndarray, K: float, b: float, c: float) -> np.ndarray:
    """E4 alternative: asymmetric sigmoid, inflection at ln(b)/c."""
    return K * np.exp(-b * np.exp(-c * t))


class Fit(NamedTuple):
    law: str
    params: tuple[float, ...]
    r: float
    rmse: float
    n: int


def gof(y: np.ndarray, y_hat: np.ndarray) -> tuple[float, float]:
    r = float(np.corrcoef(y, y_hat)[0, 1]) if np.std(y_hat) > 0 else 0.0
    return r, float(np.sqrt(np.mean((y - y_hat) ** 2)))


def fit_law(name: str, fn, t: np.ndarray, L: np.ndarray, p0: list[float], bounds=(-np.inf, np.inf)) -> Fit:
    params, _ = curve_fit(fn, t, L, p0=p0, bounds=bounds, maxfev=20000)
    r, rmse = gof(L, fn(t, *params))
    return Fit(name, tuple(float(p) for p in params), r, rmse, len(t))


def fit_growth_laws(s: Series) -> dict[str, Fit]:
    L_guess = float(s.L.max()) * 1.05
    return {
        "exp_approach": fit_law("exp_approach", law_exp_approach, s.t, s.L, [L_guess, 0.1, 1.0],
                                bounds=([0, 1e-4, -5], [50, 5, s.t.min()])),
        "hyperbolic": fit_law("hyperbolic", law_hyperbolic, s.t, s.L, [L_guess, 10.0]),
        "logistic": fit_law("logistic", law_logistic, s.t, s.L, [L_guess, 0.3, 5.0], bounds=([0, 1e-3, -10], [50, 5, 40])),
        "gompertz": fit_law("gompertz", law_gompertz, s.t, s.L, [L_guess, 3.0, 0.2], bounds=([0, 1e-3, 1e-3], [50, 100, 5])),
    }


def velocity_decay_exponents(tv: np.ndarray, V: np.ndarray) -> dict[str, float]:
    """Fig. 37 test: log V vs t (exponential) and log V vs log t (power, Thompson's V t^n)."""
    lv = np.log(V)
    k_slope, _ = np.polyfit(tv, lv, 1)
    r_exp = float(np.corrcoef(tv, lv)[0, 1])
    n_slope, _ = np.polyfit(np.log(tv), lv, 1)
    r_pow = float(np.corrcoef(np.log(tv), lv)[0, 1])
    return {"k_exponential_per_day": float(-k_slope), "r_log_v_vs_t": r_exp,
            "n_power": float(-n_slope), "r_log_v_vs_log_t": r_pow, "thompson_claimed_n": 2.0}


def inflection_day(d: np.ndarray, V: np.ndarray, fits: dict[str, Fit]) -> dict[str, float]:
    i = int(np.argmax(V))
    K, r, ti = fits["logistic"].params
    Kg, b, c = fits["gompertz"].params
    return {"argmax_daily_increment_day": float(d[i]), "max_daily_increment_mm": float(V[i]),
            "logistic_inflection_day": float(ti), "gompertz_inflection_day": float(np.log(b) / c),
            "thompson_claimed_day": 7.0}


def ellis_scaling(e: Ellis) -> dict[str, Any]:
    """E5: absolute regenerate vs amount removed at day 9 and at the day-32 plateau."""
    out: dict[str, Any] = {}
    for label, col in (("day9", 2), ("plateau_d32", 6)):
        y = e.mm[:, col]
        r = float(np.corrcoef(e.removed_mm, y)[0, 1])
        slope_origin = float(np.sum(e.removed_mm * y) / np.sum(e.removed_mm**2))
        a, b = np.polyfit(e.removed_mm, y, 1)
        out[label] = {"mm_regenerated": y.tolist(), "pearson_r": r, "slope_through_origin": slope_origin,
                      "ols_slope": float(a), "ols_intercept": float(b),
                      "fraction_of_removed": (y / e.removed_mm).tolist()}
    out["plateau_fraction_mean"] = float(np.mean(e.pct[:, 6]) / 100.0)
    out["plateau_fraction_cv"] = float(np.std(e.pct[:, 6]) / np.mean(e.pct[:, 6]))
    # early absolute rate (mm/day over days 0-9) vs amount removed
    rate9 = e.mm[:, 2] / 9.0
    out["early_rate_mm_per_day"] = rate9.tolist()
    out["early_rate_r"] = float(np.corrcoef(e.removed_mm, rate9)[0, 1])
    return out


def isochrone_time(days: np.ndarray, pct_row: np.ndarray, q: float) -> float:
    """Linear interpolation of the time at which the series first reaches q % of the removed amount."""
    pts_t = np.concatenate([[0.0], days])
    pts_p = np.concatenate([[0.0], pct_row])
    for i in range(1, len(pts_t)):
        if pts_p[i] >= q:
            p0, p1 = pts_p[i - 1], pts_p[i]
            if p1 == p0:
                return float(pts_t[i])
            return float(pts_t[i - 1] + (pts_t[i] - pts_t[i - 1]) * (q - p0) / (p1 - p0))
    return float("nan")


def ellis_isochrones(e: Ellis, qs=(10, 20, 30, 40)) -> dict[str, Any]:
    """E6: Fig. 40 isochrones t_q(R); log-log slope beta (cubic parabola <=> beta = 1/3)."""
    out: dict[str, Any] = {"q_pct": list(qs), "t_q_days": {}, "loglog_slope_beta": {}}
    betas = []
    for q in qs:
        tq = np.array([isochrone_time(e.days, e.pct[i], q) for i in range(len(e.series))])
        out["t_q_days"][str(q)] = tq.tolist()
        ok = np.isfinite(tq)
        if ok.sum() >= 3:
            beta, _ = np.polyfit(np.log(e.removed_pct[ok]), np.log(tq[ok]), 1)
            out["loglog_slope_beta"][str(q)] = float(beta)
            betas.append(float(beta))
    out["beta_mean"] = float(np.mean(betas))
    out["beta_cubic_parabola"] = 1.0 / 3.0
    out["R_crit_pct"] = R_CRIT_PCT
    return out


# --------------------------------------------------------------------------------------
# Pass 2: parameter-regime sweeps, controls, specimen CV, statistics
# --------------------------------------------------------------------------------------
def law_power_decay_integrated(t: np.ndarray, L_inf: float, c: float, n: float) -> np.ndarray:
    """Generalised E2: V = c' t^-n integrated for n > 1 -> L(t) = L_inf - c t^(1-n) (n = 2 is Thompson's gloss)."""
    return L_inf - c * np.power(t, 1.0 - n)


def sweep_decay_exponent(s: Series, ns: np.ndarray | None = None) -> dict[str, Any]:
    """Parameter regime 1: fix the decay exponent n in [1.1, 3.0], fit (L_inf, c), record RMSE.
    Shows how the integrated power law degrades away from its optimum and at Thompson's n = 2."""
    ns = np.linspace(1.1, 3.0, 20) if ns is None else ns
    rmse = []
    for n in ns:
        fn = lambda t, L_inf, c, _n=float(n): law_power_decay_integrated(t, L_inf, c, _n)
        p, _ = curve_fit(fn, s.t, s.L, p0=[float(s.L.max()) * 1.1, 10.0], maxfev=20000)
        rmse.append(gof(s.L, fn(s.t, *p))[1])
    rmse = np.array(rmse)
    free = fit_law("power_free_n", law_power_decay_integrated, s.t, s.L, [float(s.L.max()) * 1.1, 10.0, 1.5],
                   bounds=([0, 0, 1.01], [50, 500, 6]))
    return {"n_grid": ns.tolist(), "rmse_mm": rmse.tolist(), "n_best_grid": float(ns[int(np.argmin(rmse))]),
            "rmse_at_n2": float(rmse[int(np.argmin(np.abs(ns - 2.0)))]),
            "free_fit": {"L_inf": free.params[0], "c": free.params[1], "n_exponent": free.params[2], "r": free.r, "rmse": free.rmse}}


def law_isochrone(R: np.ndarray, a: float, gamma: float) -> np.ndarray:
    """E6 with the 75 % asymptote: t_q(R) = a R^(1/3) (R_crit / (R_crit - R))^gamma."""
    return a * np.power(R, 1.0 / 3.0) * np.power(R_CRIT_PCT / (R_CRIT_PCT - R), gamma)


def sweep_amputation_level(e: Ellis, iso: dict[str, Any], q: int = 20, gammas=(0.0, 0.25, 0.5, 1.0)) -> dict[str, Any]:
    """Parameter regime 2: amputation level R from 2 % to 74 % through the Fig. 40 isochrone law
    t_q = a R^(1/3) (75/(75-R))^gamma.  gamma = 0 is the pure cubic parabola; gamma > 0 encodes Thompson's
    'converge towards infinity' at the lethal 75 % level.  For each gamma the scale a is fitted to the four
    Ellis series; gamma is also fitted freely to show whether the measured range (R <= 57 %) demands the
    divergence or whether it is Thompson's extrapolation from the fifth (lethal) series."""
    tq = np.array(iso["t_q_days"][str(q)])
    ok = np.isfinite(tq)
    R_obs, t_obs = e.removed_pct[ok], tq[ok]
    R = np.linspace(2.0, 74.0, 73)
    family = []
    for g in gammas:
        fn = lambda RR, a, _g=g: law_isochrone(RR, a, _g)
        (a,), _ = curve_fit(fn, R_obs, t_obs, p0=[1.5])
        t_sw = fn(R, a)
        family.append({"gamma": float(g), "a": float(a), "rmse_days": gof(t_obs, fn(R_obs, a))[1],
                       "t_q_days": t_sw.tolist(), "monotone": bool(np.all(np.diff(t_sw) > 0)),
                       "t_q_at_70pct": float(fn(np.array([70.0]), a)[0])})
    p, _ = curve_fit(law_isochrone, R_obs, t_obs, p0=[1.5, 0.5], bounds=([0, 0], [20, 5]))
    t_free = law_isochrone(R, *p)
    return {"q_pct": q, "R_grid_pct": R.tolist(), "family": family,
            "a": float(p[0]), "gamma": float(p[1]), "t_q_days": t_free.tolist(),
            "monotone": bool(np.all(np.diff(t_free) > 0)), "t_q_at_70pct": float(law_isochrone(np.array([70.0]), *p)[0]),
            "rmse_days": gof(t_obs, law_isochrone(R_obs, *p))[1],
            "divergence_required_by_data": bool(p[1] > 0.25),
            "max_measured_R_pct": float(R_obs.max())}


def permutation_test_r(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    """Exact permutation test (all n! relabellings) of Pearson r between amount removed and regenerate."""
    r_obs = float(np.corrcoef(x, y)[0, 1])
    rs = np.array([np.corrcoef(x, np.array(perm))[0, 1] for perm in itertools.permutations(y.tolist())])
    p_one = float(np.mean(rs >= r_obs - 1e-12))
    return {"r_obs": r_obs, "n_perm": int(len(rs)), "p_one_sided": p_one, "r_null_mean": float(np.mean(rs)),
            "r_null_sd": float(np.std(rs))}


def ablation_controls(ctx_expII: Series, f2: dict[str, Fit], e: Ellis, es: dict[str, Any]) -> dict[str, Any]:
    """Negative / ablation controls.
    (a) constant-velocity null for Durbin Exp. II: L = v t (no decay) -> RMSE vs the decaying laws;
    (b) R-independent null for Ellis: every series regenerates the same mm (mean) -> RMSE vs L = p R;
    (c) exact permutation tests of the Ellis amount-removed labels at day 9 and at the plateau."""
    v = float(np.sum(ctx_expII.t * ctx_expII.L) / np.sum(ctx_expII.t**2))
    rmse_linear = gof(ctx_expII.L, v * ctx_expII.t)[1]
    out: dict[str, Any] = {"durbin_constant_velocity_null": {"v_mm_per_day": v, "rmse_mm": rmse_linear,
                                                             "rmse_ratio_vs_exp_approach": rmse_linear / f2["exp_approach"].rmse}}
    for label, col in (("day9", 2), ("plateau_d32", 6)):
        y = e.mm[:, col]
        rmse_null = float(np.sqrt(np.mean((y - y.mean()) ** 2)))
        rmse_prop = float(np.sqrt(np.mean((y - es[label]["slope_through_origin"] * e.removed_mm) ** 2)))
        out[f"ellis_{label}"] = {"rmse_R_independent_null_mm": rmse_null, "rmse_proportional_mm": rmse_prop,
                                 "rmse_ratio_null_over_model": rmse_null / max(rmse_prop, 1e-9),
                                 "permutation": permutation_test_r(e.removed_mm, y)}
    return out


def residual_statistics(s: Series, fit: Fit, fn) -> dict[str, float]:
    """KS test of standardised residuals against N(0,1) and residual coefficient of variation."""
    from scipy.stats import kstest

    resid = s.L - fn(s.t, *fit.params)
    z = (resid - resid.mean()) / max(resid.std(), 1e-9)
    stat, p = kstest(z, "norm")
    return {"ks_stat": float(stat), "ks_p": float(p), "resid_sd_mm": float(resid.std()),
            "resid_cv_vs_L_inf": float(resid.std() / fit.params[0])}


def segment_tadpole(img_path: Path, max_side: int = 640) -> dict[str, Any]:
    """Computer-vision morphometric on the vendored specimen (lateral tadpole on black background):
    threshold -> largest component -> principal axis -> width profile along the axis; the tail begins
    where the width falls below 45 % of the maximum (head-body) width.  Returns tail fraction of total
    length, compared against Thompson's tail/body figures (Ellis 26/40 = 65 %, Durbin 21.2/34.2 = 62 %)."""
    from scipy import ndimage as ndi

    im = Image.open(img_path).convert("L")
    w, h = im.size
    scale = min(1.0, max_side / max(w, h))
    if scale < 1.0:
        im = im.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)
    arr = np.asarray(im, dtype=float) / 255.0
    blur = ndi.gaussian_filter(arr, sigma=1.5)
    mask = blur > 0.18
    mask = ndi.binary_opening(mask, iterations=2)
    mask = ndi.binary_fill_holes(mask)
    labeled, n = ndi.label(mask)
    if n == 0:
        return {"ok": False}
    sizes = ndi.sum(mask, labeled, index=range(1, n + 1))
    body = labeled == (int(np.argmax(sizes)) + 1)
    ys, xs = np.nonzero(body)
    pts = np.column_stack([xs, ys]).astype(float)
    mu = pts.mean(axis=0)
    cov = np.cov((pts - mu).T)
    evals, evecs = np.linalg.eigh(cov)
    axis = evecs[:, int(np.argmax(evals))]
    perp = np.array([-axis[1], axis[0]])
    proj = (pts - mu) @ axis
    lat = (pts - mu) @ perp
    bins = np.linspace(proj.min(), proj.max(), 120)
    idx = np.clip(np.digitize(proj, bins) - 1, 0, len(bins) - 2)
    width = np.zeros(len(bins) - 1)
    for b in range(len(bins) - 1):
        sel = idx == b
        if sel.any():
            width[b] = lat[sel].max() - lat[sel].min()
    # orient so the head (widest part) is at the start
    if np.argmax(width) > len(width) / 2:
        width = width[::-1]
        proj = -proj
        bins = -bins[::-1]
        axis = -axis
    wmax = float(width.max())
    # Body/tail boundary.  The tail is the terminal run over which the (smoothed) width tapers
    # monotonically toward the tip (tolerance 8 % of the maximum width for mottling / fin-margin
    # noise); scanning anteriorly from the tip, that run ends at the posterior body bulge (vent
    # region).  The boundary is placed where the bulge has decayed halfway to the tail-base width.
    sm = np.convolve(width, np.ones(3) / 3.0, mode="same")
    tol = 0.08 * wmax
    i_peak = len(sm) - 1
    for i in range(len(sm) - 2, -1, -1):
        if sm[i] < sm[i + 1] - tol:
            i_peak = i + 1
            break
    tail_base = float(np.median(sm[min(i_peak + 5, len(sm) - 1):min(i_peak + 25, len(sm))]))
    half = 0.5 * (sm[i_peak] + tail_base)
    i_tail = next((i for i in range(i_peak, len(sm)) if sm[i] <= half), i_peak)
    total_len = float(proj.max() - proj.min())
    tail_start = float(bins[i_tail] - proj.min())
    tail_frac = 1.0 - tail_start / total_len
    return {"ok": True, "image_shape": list(arr.shape), "area_px": float(body.sum()), "total_length_px": total_len,
            "tail_length_px": total_len - tail_start, "tail_fraction": float(tail_frac),
            "max_width_px": wmax, "aspect_ratio": total_len / max(wmax, 1e-9),
            "reference_tail_fraction_ellis": 25.96 / 39.99, "reference_tail_fraction_durbin": 21.2 / 34.2,
            "mask": body, "axis": axis.tolist(), "centroid": mu.tolist(),
            "tail_boundary_proj": float(bins[i_tail]), "proj_min": float(proj.min()), "proj_max": float(proj.max())}


# --------------------------------------------------------------------------------------
# Figure + GIF
# --------------------------------------------------------------------------------------
def _style(ax, title: str) -> None:
    ax.set_facecolor(PANEL)
    ax.tick_params(colors="#e2e8f0", labelsize=8)
    ax.xaxis.label.set_color("#e2e8f0")
    ax.yaxis.label.set_color("#e2e8f0")
    ax.set_title(title, color=FG, fontsize=10)
    for sp in ax.spines.values():
        sp.set_color("#475569")


def render_comparison(ctx: dict[str, Any]) -> None:
    fig, axes = plt.subplots(3, 3, figsize=(15, 13.5), facecolor=BG)
    s2: Series = ctx["expII"]
    f2 = ctx["fits_expII"]
    tt = np.linspace(2.5, 31, 200)

    ax = axes[0, 0]
    _style(ax, "Durbin Exp. II (Thompson p. 139, Fig. 35)\nL(t): exponential approach vs V t² gloss")
    ax.scatter(s2.t, s2.L, c=C_DATA, s=45, zorder=4, label="Thompson table (mm)")
    ax.plot(tt, law_exp_approach(tt, *f2["exp_approach"].params), c=C_EXP, lw=2,
            label=f"E3 exp approach RMSE={f2['exp_approach'].rmse:.3f}")
    ax.plot(tt, law_hyperbolic(tt, *f2["hyperbolic"].params), c=C_HYP, lw=2, ls="--",
            label=f"E2 L∞−c/t (V t²=c) RMSE={f2['hyperbolic'].rmse:.3f}")
    ax.set_xlabel("days after amputation")
    ax.set_ylabel("regenerated length (mm)")
    ax.legend(fontsize=7, facecolor=PANEL, labelcolor="#e2e8f0")

    ax = axes[0, 1]
    vd = ctx["decay"]
    tv, V = ctx["tv"], ctx["V"]
    _style(ax, "Fig. 37 replica: log₁₀ V vs t\n" f"exp: r={vd['r_log_v_vs_t']:.3f}, k={vd['k_exponential_per_day']:.3f}/d; "
           f"power n={vd['n_power']:.2f} (claimed 2)")
    ax.scatter(tv, np.log10(100 * V), c=C_DATA, s=25, zorder=3)
    kk, b0 = np.polyfit(tv, np.log10(100 * V), 1)
    ax.plot(tv, kk * tv + b0, c=C_EXP, lw=2, label="log V linear in t (E3)")
    nn, c0 = np.polyfit(np.log(tv), np.log(V), 1)
    ax.plot(tv, np.log10(100 * np.exp(c0) * tv**nn), c=C_HYP, ls="--", lw=2, label=f"V ∝ t^−{-nn:.2f} (E2 free n)")
    ax.plot(tv, np.log10(100 * V[0] * (tv[0] / tv) ** 2), c=MUTED, ls=":", lw=1.5, label="V t² = const (Thompson)")
    ax.set_xlabel("days")
    ax.set_ylabel("log₁₀(100 × daily increment)")
    ax.legend(fontsize=7, facecolor=PANEL, labelcolor="#e2e8f0")

    ax = axes[0, 2]
    s4, d4, L4, V4 = ctx["expIV"]
    f4 = ctx["fits_expIV"]
    inf = ctx["inflection"]
    _style(ax, "Durbin Exp. IV (p. 144, Figs. 38-39)\n" f"inflection: increments peak day {inf['argmax_daily_increment_day']:.0f}, "
           f"Gompertz {inf['gompertz_inflection_day']:.1f} d (claimed ~7)")
    ax.scatter(s4.t, s4.L, c=C_DATA, s=45, zorder=4, label="raw (mm)")
    tt4 = np.linspace(0.5, 31, 200)
    ax.plot(tt4, law_gompertz(tt4, *f4["gompertz"].params), c=C_ALT, lw=2, label=f"Gompertz RMSE={f4['gompertz'].rmse:.3f}")
    ax.plot(tt4, law_exp_approach(tt4, *f4["exp_approach"].params), c=C_EXP, lw=1.5, ls="--",
            label=f"exp approach RMSE={f4['exp_approach'].rmse:.3f}")
    ax2 = ax.twinx()
    ax2.bar(d4, V4, color=C_HYP, alpha=0.35, width=0.8, label="daily increment (interp.)")
    ax2.tick_params(colors=MUTED, labelsize=7)
    ax2.set_ylabel("mm/day", color=MUTED)
    ax.set_ylim(0, float(s4.L.max()) * 1.12)
    ax.set_xlabel("days")
    ax.set_ylabel("regenerated length (mm)")
    ax.legend(fontsize=7, facecolor=PANEL, labelcolor="#e2e8f0", loc="lower right")

    ax = axes[1, 0]
    e: Ellis = ctx["ellis"]
    es = ctx["ellis_scaling"]
    _style(ax, "Ellis (p. 148): amount regenerated vs amount removed\n"
           f"day 9 r={es['day9']['pearson_r']:.3f}; plateau r={es['plateau_d32']['pearson_r']:.4f}")
    xr = np.linspace(0, 16, 50)
    ax.scatter(e.removed_mm, e.mm[:, 2], c=C_DATA, s=50, zorder=4, label="day 9")
    ax.plot(xr, es["day9"]["slope_through_origin"] * xr, c=C_DATA, lw=1.5, ls="--",
            label=f"∝ R (slope {es['day9']['slope_through_origin']:.2f})")
    ax.scatter(e.removed_mm, e.mm[:, 6], c=C_EXP, s=50, marker="s", zorder=4, label="plateau (day 32)")
    ax.plot(xr, es["plateau_d32"]["slope_through_origin"] * xr, c=C_EXP, lw=1.5, ls="--",
            label=f"∝ R (slope {es['plateau_d32']['slope_through_origin']:.2f})")
    ax.plot(xr, xr, c=MUTED, lw=1, ls=":", label="complete restoration")
    for i, nm in enumerate(e.series):
        ax.annotate(nm, (e.removed_mm[i], e.mm[i, 6]), color=FG, fontsize=8, xytext=(4, 4), textcoords="offset points")
    ax.set_xlabel("tail removed R (mm)")
    ax.set_ylabel("regenerated (mm)")
    ax.legend(fontsize=7, facecolor=PANEL, labelcolor="#e2e8f0")

    ax = axes[1, 1]
    iso = ctx["isochrones"]
    _style(ax, "Fig. 40 replica: isochrones t_q(R)\n" f"log-log slope β̄={iso['beta_mean']:.2f} (cubic parabola ⇒ 1/3)")
    cols = [C_DATA, C_EXP, C_ALT, C_HYP]
    for q, col in zip(iso["q_pct"], cols):
        tq = np.array(iso["t_q_days"][str(q)])
        ok = np.isfinite(tq)
        ax.scatter(tq[ok], e.removed_pct[ok], c=col, s=40, zorder=4)
        beta = iso["loglog_slope_beta"].get(str(q))
        if beta is not None:
            a = np.exp(np.mean(np.log(tq[ok]) - beta * np.log(e.removed_pct[ok])))
            RR = np.linspace(0.5, 74.0, 200)
            ax.plot(a * RR**beta, RR, c=col, lw=1.5, label=f"{q} % restored, β={beta:.2f}")
    ax.axhline(R_CRIT_PCT, c=MUTED, ls=":", lw=1)
    ax.text(0.3, R_CRIT_PCT + 1, "75 % removed: no regeneration (death)", color=MUTED, fontsize=7)
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 82)
    ax.set_xlabel("days")
    ax.set_ylabel("% of tail removed")
    ax.legend(fontsize=7, facecolor=PANEL, labelcolor="#e2e8f0", loc="lower right")

    ax = axes[1, 2]
    _style(ax, "Thompson (1917) p. 148 page scan (IA leaf 168)")
    page = Image.open(PAGE_FIG40).convert("L")
    w, h = page.size
    ax.imshow(page.crop((int(w * 0.08), int(h * 0.06), int(w * 0.95), int(h * 0.72))), cmap="gray")
    ax.axis("off")

    # ---- Pass 2 row: parameter sweeps, controls, specimen CV ----
    ax = axes[2, 0]
    sw = ctx["sweep_n"]
    ff = sw["free_fit"]
    _style(ax, "Pass 2 sweep: decay exponent n in V ∝ t⁻ⁿ (Exp. II)\n"
           f"RMSE min at n={sw['n_best_grid']:.2f} (free fit n={ff['n_exponent']:.2f}); n=2 ⇒ RMSE {sw['rmse_at_n2']:.3f}")
    ax.plot(sw["n_grid"], sw["rmse_mm"], "o-", c=C_ALT, ms=4)
    ax.axvline(2.0, c=C_HYP, ls="--", lw=1.5, label="Thompson: n = 2")
    ax.axhline(f2["exp_approach"].rmse, c=C_EXP, ls=":", lw=1.5, label=f"exp approach RMSE {f2['exp_approach'].rmse:.3f}")
    ax.set_xlabel("decay exponent n")
    ax.set_ylabel("RMSE of L(t) = L∞ − c t^(1−n) (mm)")
    ax.legend(fontsize=7, facecolor=PANEL, labelcolor="#e2e8f0")

    ax = axes[2, 1]
    swR = ctx["sweep_R"]
    ctl = ctx["controls"]
    pd9 = ctl["ellis_day9"]["permutation"]
    pdp = ctl["ellis_plateau_d32"]["permutation"]
    _style(ax, f"Pass 2 sweep: R → t₂₀ = a R^⅓ (75/(75−R))^γ, free γ={swR['gamma']:.2f}\n"
           f"controls: perm p={pd9['p_one_sided']:.3f} / {pdp['p_one_sided']:.3f}; "
           f"const-velocity null RMSE ×{ctl['durbin_constant_velocity_null']['rmse_ratio_vs_exp_approach']:.1f}")
    fam_cols = [MUTED, C_ALT, C_EXP, C_HYP]
    for fam, col in zip(swR["family"], fam_cols):
        ax.plot(swR["R_grid_pct"], fam["t_q_days"], c=col, lw=1.5, label=f"γ={fam['gamma']:.2f}, RMSE {fam['rmse_days']:.2f} d")
    tq20 = np.array(iso["t_q_days"]["20"])
    ax.scatter(e.removed_pct, tq20, c=C_DATA, s=45, zorder=4, label="Ellis t₂₀ (interpolated)")
    ax.axvline(R_CRIT_PCT, c=MUTED, ls=":", lw=1)
    ax.set_ylim(0, 25)
    ax.set_xlabel("% of tail removed R")
    ax.set_ylabel("days to restore 20 % of removed amount")
    ax.legend(fontsize=7, facecolor=PANEL, labelcolor="#e2e8f0", loc="upper left")
    ax.text(0.03, 0.5, f"R-indep. null RMSE / ∝R model RMSE:\n  day 9 ×{ctl['ellis_day9']['rmse_ratio_null_over_model']:.1f}, "
            f"plateau ×{ctl['ellis_plateau_d32']['rmse_ratio_null_over_model']:.1f}\nKS resid (Exp. II exp fit): "
            f"D={ctx['resid_expII']['ks_stat']:.2f}, p={ctx['resid_expII']['ks_p']:.2f}", color="#cbd5e1", fontsize=7.5,
            transform=ax.transAxes, va="center")

    ax = axes[2, 2]
    cv = ctx["cv"]
    _style(ax, "Pass 2 CV: L. clamitans tadpole (iNat 137092396, CC BY)\n"
           f"tail fraction {cv['tail_fraction']:.3f} vs Thompson 0.65 (Ellis) / 0.62 (Durbin)")
    im = Image.open(SPECIMEN).convert("RGB")
    mask = cv["mask"]
    im = im.resize((mask.shape[1], mask.shape[0]), Image.Resampling.LANCZOS)
    ax.imshow(np.asarray(im))
    overlay = np.zeros((*mask.shape, 4))
    overlay[mask] = (0.22, 0.74, 0.97, 0.35)
    ax.imshow(overlay)
    cx, cy = cv["centroid"]
    axv = np.array(cv["axis"])
    for pm, col in ((cv["proj_min"], MUTED), (cv["tail_boundary_proj"], "#ef4444"), (cv["proj_max"], MUTED)):
        px, py = cx + axv[0] * pm, cy + axv[1] * pm
        ax.plot([px - axv[1] * 60, px + axv[1] * 60], [py + axv[0] * 60, py - axv[0] * 60], c=col, lw=2)
    ax.text(0.02, 0.05, "red: body/tail boundary (end of terminal monotone taper)", color=FG, fontsize=7.5, transform=ax.transAxes)
    ax.axis("off")

    fig.suptitle("Thompson (1917) Ch. III — regeneration as a lawful growth process (T1917-010, Pass 2)", color=FG, fontsize=13)
    fig.tight_layout()
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=130, facecolor=fig.get_facecolor())
    plt.close(fig)


def _tadpole_patch(ax, x0: float, y0: float, body_len: float, tail_len: float, tail_full: float, color: str) -> None:
    """Schematic lateral tadpole: ellipse body + tapering tail of length tail_len (scaled mm)."""
    from matplotlib.patches import Ellipse, Polygon

    ax.add_patch(Ellipse((x0 + body_len / 2, y0), body_len, body_len * 0.6, color=color))
    xt = x0 + body_len * 0.9
    h0 = body_len * 0.22
    h1 = h0 * max(0.05, 1 - tail_len / max(tail_full, 1e-6)) + 0.02 * body_len
    tail = Polygon([[xt, y0 + h0], [xt + tail_len, y0 + h1], [xt + tail_len, y0 - h1], [xt, y0 - h0]], closed=True, color=color)
    ax.add_patch(tail)
    ax.plot([xt + tail_len, xt + tail_len], [y0 - h0 * 1.3, y0 + h0 * 1.3], c="#ef4444", lw=1)


def render_growth_gif(ctx: dict[str, Any], n_frames: int = 36) -> None:
    e: Ellis = ctx["ellis"]
    fits = ctx["ellis_fits"]
    days = np.linspace(0, 32, n_frames)
    frames: list[Image.Image] = []
    body = float(np.mean(e.body_mm - e.tail_mm))
    for d in days:
        fig, ax = plt.subplots(figsize=(7.2, 4.2), facecolor=BG)
        ax.set_facecolor(PANEL)
        ax.set_xlim(-2, 78)
        ax.set_ylim(-1, 4 * 9)
        ax.axis("off")
        for i, nm in enumerate(e.series):
            L_inf, k, t0 = fits[i]
            regen = float(max(0.0, law_exp_approach(np.array([d]), L_inf, k, t0)[0])) if d > t0 else 0.0
            tail_len = float(e.tail_mm[i] - e.removed_mm[i] + regen)
            y = 9 * (3 - i) + 4
            _tadpole_patch(ax, 0, y, body, tail_len, float(e.tail_mm[i]), [C_DATA, C_EXP, C_ALT, C_HYP][i])
            ax.text(body + e.tail_mm[i] + 1.5, y, f"{nm}: {e.removed_pct[i]:.0f}% removed  +{regen:.2f} mm", color=FG, fontsize=8, va="center")
        ax.text(0, 4 * 9 - 1.5, f"Ellis (1909) tadpole tails — day {d:4.1f}", color=FG, fontsize=11)
        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        buf = np.asarray(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)[..., :3]
        plt.close(fig)
        frames.append(Image.fromarray(buf))
    frames[0].save(OUT_GIF, save_all=True, append_images=frames[1:], duration=120, loop=0)


def fit_ellis_series(e: Ellis) -> list[tuple[float, float, float]]:
    out = []
    for i in range(len(e.series)):
        f = fit_law(f"ellis_{e.series[i]}", law_exp_approach, e.days, e.mm[i], [float(e.mm[i].max()), 0.2, 1.0],
                    bounds=([0, 1e-3, -5], [20, 5, 3.0]))
        out.append(f.params)
    return out


# --------------------------------------------------------------------------------------
# Analysis context, result, measurements
# --------------------------------------------------------------------------------------
def analyse() -> dict[str, Any]:
    verify_sha256()
    expII = durbin_expII_raw()
    day, Li, tv, V = durbin_expII_interp()
    s4, d4, L4, V4 = durbin_expIV()
    e = ellis_table()
    fits_expII = fit_growth_laws(expII)
    fits_expIV = fit_growth_laws(s4)
    es = ellis_scaling(e)
    iso = ellis_isochrones(e)
    cv = segment_tadpole(SPECIMEN)
    return {
        "expII": expII, "interp": (day, Li), "tv": tv, "V": V,
        "fits_expII": fits_expII, "decay": velocity_decay_exponents(tv, V),
        "expIV": (s4, d4, L4, V4), "fits_expIV": fits_expIV, "inflection": inflection_day(d4, V4, fits_expIV),
        "ellis": e, "ellis_scaling": es, "isochrones": iso,
        "ellis_fits": fit_ellis_series(e),
        # Pass 2
        "sweep_n": sweep_decay_exponent(expII),
        "sweep_R": sweep_amputation_level(e, iso),
        "controls": ablation_controls(expII, fits_expII, e, es),
        "resid_expII": residual_statistics(expII, fits_expII["exp_approach"], law_exp_approach),
        "resid_expIV": residual_statistics(s4, fits_expIV["gompertz"], law_gompertz),
        "cv": cv,
    }


def verdict(ctx: dict[str, Any]) -> tuple[str, list[str]]:
    es, vd, inf, iso = ctx["ellis_scaling"], ctx["decay"], ctx["inflection"], ctx["isochrones"]
    f2 = ctx["fits_expII"]
    caveats: list[str] = []
    core_ok = es["day9"]["pearson_r"] > 0.95 and es["plateau_d32"]["pearson_r"] > 0.95
    lawful_ok = f2["exp_approach"].r > 0.98 and ctx["fits_expIV"]["gompertz"].r > 0.98
    if abs(vd["n_power"] - 2.0) > 0.4:
        caveats.append(f"Thompson's 'V t^2 = const' gloss is not supported: fitted exponent n={vd['n_power']:.2f}; "
                       f"log V is better described as linear in t (r={vd['r_log_v_vs_t']:.3f}) than in log t (r={vd['r_log_v_vs_log_t']:.3f}).")
    if f2["hyperbolic"].rmse > 2 * f2["exp_approach"].rmse:
        caveats.append(f"Integrated V t^2 law (L_inf - c/t) RMSE {f2['hyperbolic'].rmse:.3f} mm vs exponential approach {f2['exp_approach'].rmse:.3f} mm on Exp. II.")
    if abs(inf["argmax_daily_increment_day"] - 7) > 1.5:
        caveats.append("Exp. IV inflection not near day 7.")
    if not (0.15 <= iso["beta_mean"] <= 0.5):
        caveats.append(f"Isochrone log-log slope {iso['beta_mean']:.2f} far from cubic-parabola 1/3.")
    ctl = ctx["controls"]
    if ctl["ellis_day9"]["permutation"]["p_one_sided"] > 0.05 or ctl["ellis_plateau_d32"]["permutation"]["p_one_sided"] > 0.05:
        caveats.append("Ellis amount-removed correlation not significant under exact permutation of the four series labels.")
    if ctl["durbin_constant_velocity_null"]["rmse_ratio_vs_exp_approach"] < 2.0:
        caveats.append("Constant-velocity null is not clearly worse than the decaying-velocity law on Exp. II.")
    swR = ctx["sweep_R"]
    if not swR["divergence_required_by_data"]:
        caveats.append(f"Fig. 40's divergence of the isochrones at 75 % removed is not required by the four measured series "
                       f"(R <= {swR['max_measured_R_pct']:.0f} %; free asymptote exponent gamma={swR['gamma']:.2f}); "
                       "it rests on Ellis's fifth (lethal) series, i.e. it is Thompson's extrapolation.")
    if core_ok and lawful_ok:
        return "supported", caveats
    if core_ok or lawful_ok:
        return "mixed", caveats
    return "refuted", caveats


def write_result_json(ctx: dict[str, Any]) -> dict[str, Any]:
    ref = load_reference()
    v, caveats = verdict(ctx)
    es, vd, inf, iso, f2, f4 = (ctx["ellis_scaling"], ctx["decay"], ctx["inflection"], ctx["isochrones"],
                                ctx["fits_expII"], ctx["fits_expIV"])
    payload = {
        "claim_id": "T1917-010",
        "date_made": "1917",
        "dataset": {
            "name": "Thompson (1917) pp. 139-148 tables (Durbin 1909 Exp. II/IV; Ellis 1909 series O/P/R/S) digitised from IA scan ongrowthform1917thom",
            "doi": None,
            "ia_identifier": ref["ia_scan"]["identifier"],
            "sha256": ref["sha256_digests"],
            "pre_scouted_dryad": {"doi": ref["pre_scouted_dataset"]["doi"], "verdict": ref["pre_scouted_dataset"]["match_verdict"],
                                  "sha256": ref["sha256_digests"]["datasets/thompson_1917/t1917-010/dryad_n6n6495_fitzpatrick_et_al_data.csv"]},
        },
        "verdict": v,
        "headline_numbers": {
            "ellis_day9_r_regenerated_vs_removed": round(es["day9"]["pearson_r"], 4),
            "ellis_plateau_r_regenerated_vs_removed": round(es["plateau_d32"]["pearson_r"], 4),
            "ellis_day9_fraction_of_removed_slope": round(es["day9"]["slope_through_origin"], 3),
            "ellis_plateau_fraction_mean": round(es["plateau_fraction_mean"], 3),
            "ellis_isochrone_loglog_beta_mean": round(iso["beta_mean"], 3),
            "durbin_expII_exp_approach_r": round(f2["exp_approach"].r, 4),
            "durbin_expII_exp_approach_rmse_mm": round(f2["exp_approach"].rmse, 4),
            "durbin_expII_hyperbolic_rmse_mm": round(f2["hyperbolic"].rmse, 4),
            "durbin_velocity_power_exponent_n": round(vd["n_power"], 3),
            "durbin_velocity_exp_k_per_day": round(vd["k_exponential_per_day"], 4),
            "durbin_expIV_gompertz_r": round(f4["gompertz"].r, 4),
            "durbin_expIV_inflection_day_increments": inf["argmax_daily_increment_day"],
            "durbin_expIV_inflection_day_gompertz": round(inf["gompertz_inflection_day"], 2),
            "sweep_n_best": round(ctx["sweep_n"]["n_best_grid"], 3),
            "sweep_n_free_fit_exponent": round(ctx["sweep_n"]["free_fit"]["n_exponent"], 3),
            "sweep_R_isochrone_gamma": round(ctx["sweep_R"]["gamma"], 3),
            "ellis_day9_permutation_p": round(ctx["controls"]["ellis_day9"]["permutation"]["p_one_sided"], 4),
            "ellis_plateau_permutation_p": round(ctx["controls"]["ellis_plateau_d32"]["permutation"]["p_one_sided"], 4),
            "durbin_constant_velocity_null_rmse_ratio": round(ctx["controls"]["durbin_constant_velocity_null"]["rmse_ratio_vs_exp_approach"], 2),
            "durbin_expII_resid_ks_p": round(ctx["resid_expII"]["ks_p"], 3),
            "specimen_tail_fraction_cv": round(ctx["cv"]["tail_fraction"], 4),
            "specimen_tail_fraction_thompson": [0.65, 0.62],
        },
        "controls": {k: v for k, v in ctx["controls"].items()},
        "specimen_cv": {k: v for k, v in ctx["cv"].items() if k not in {"mask"}},
        "caveats": caveats,
        "script": "thompson/claims/t1917-010_the-rate-of-growth.py",
        "figure": str(OUT_PNG.relative_to(ROOT)),
        "animation": str(OUT_GIF.relative_to(ROOT)),
        "notes": "Verdict computed on Thompson's own 1917 tables (exact Durbin/Ellis curves named by the claim). "
                 "The Dryad hit doi:10.5061/dryad.n6n6495 is a DATASET_MISMATCH analogue and does not enter the verdict.",
    }
    RESULT_JSON.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def measure_morphoeval() -> list[dict[str, Any]]:
    ctx = analyse()
    es, vd, inf, iso, f2, f4 = (ctx["ellis_scaling"], ctx["decay"], ctx["inflection"], ctx["isochrones"],
                                ctx["fits_expII"], ctx["fits_expIV"])
    ctl, cv, rs, sw, swR = ctx["controls"], ctx["cv"], ctx["resid_expII"], ctx["sweep_n"], ctx["sweep_R"]
    T = "Thompson (1917) Ch. III"
    m = [
        {"name": "ellis_day9_pearson_r", "value": es["day9"]["pearson_r"], "unit": "dimensionless",
         "method": "Pearson r, mm regenerated vs mm removed (n=4); exact permutation control p=%.3f" % ctl["ellis_day9"]["permutation"]["p_one_sided"],
         "reference_value": 1.0, "reference_source": f"{T} p. 148: amount regenerated increases with amount removed (perfect proportionality would give r = 1)"},
        {"name": "ellis_plateau_pearson_r", "value": es["plateau_d32"]["pearson_r"], "unit": "dimensionless",
         "method": "Pearson r at day-32 plateau; R-independent null RMSE / proportional-model RMSE = %.1f" % ctl["ellis_plateau_d32"]["rmse_ratio_null_over_model"],
         "reference_value": 1.0, "reference_source": f"{T} p. 148 (Ellis 1909)"},
        {"name": "ellis_plateau_fraction_mean", "value": es["plateau_fraction_mean"], "unit": "fraction",
         "method": "mean of 44,44,48,48 %% with CV=%.3f" % es["plateau_fraction_cv"],
         "reference_value": 0.46, "reference_source": f"{T} p. 146/148: restoration 'falls somewhat short' of complete (printed 44-48 %)"},
        {"name": "ellis_isochrone_beta_mean", "value": iso["beta_mean"], "unit": "dimensionless",
         "method": "log-log slope of t_q vs R, q=10..40 (cubic parabola <=> 1/3)",
         "reference_value": round(1.0 / 3.0, 4), "reference_source": f"{T} p. 149, Fig. 40: 'each curve approximates in form to a cubic parabola'"},
        {"name": "durbin_expII_exp_approach_rmse", "value": f2["exp_approach"].rmse, "unit": "mm",
         "method": "curve_fit L_inf(1-exp(-k(t-t0))); KS resid D=%.2f p=%.2f; constant-velocity null RMSE x%.1f" % (rs["ks_stat"], rs["ks_p"], ctl["durbin_constant_velocity_null"]["rmse_ratio_vs_exp_approach"])},
        {"name": "durbin_expII_hyperbolic_rmse", "value": f2["hyperbolic"].rmse, "unit": "mm",
         "method": "curve_fit L_inf - c/t (Thompson's V t^2 = const integrated) vs exponential alternative"},
        {"name": "durbin_velocity_power_exponent", "value": vd["n_power"], "unit": "dimensionless",
         "method": "OLS log V vs log t on Thompson's interpolated increments; sweep RMSE minimum at n=%.2f" % sw["n_best_grid"],
         "reference_value": 2.0, "reference_source": f"{T} p. 143: 'the velocity varies, or tends to vary, inversely as the square of the time'"},
        {"name": "durbin_velocity_exp_k", "value": vd["k_exponential_per_day"], "unit": "1/day",
         "method": "OLS log V vs t (Fig. 37 straight line), r=%.3f" % vd["r_log_v_vs_t"]},
        {"name": "durbin_expIV_inflection_day", "value": inf["gompertz_inflection_day"], "unit": "day",
         "method": "Gompertz ln(b)/c; argmax of printed daily increments = day %.0f" % inf["argmax_daily_increment_day"],
         "reference_value": 7.0, "reference_source": f"{T} p. 144: 'a visible point of inflection in the curve about the seventh day'"},
        {"name": "durbin_expIV_gompertz_r", "value": f4["gompertz"].r, "unit": "dimensionless", "method": "Pearson r fitted vs observed (10 points)"},
        {"name": "sweep_R_isochrone_gamma", "value": swR["gamma"], "unit": "dimensionless",
         "method": "asymptote exponent in t_q = a R^(1/3) (75/(75-R))^gamma, monotone=%s" % swR["monotone"]},
        {"name": "specimen_tail_fraction", "value": cv["tail_fraction"], "unit": "fraction",
         "method": "CV threshold + principal axis + width profile on vendored L. clamitans photo (tail = terminal monotone taper; boundary at half-decay of the posterior body bulge)",
         "reference_value": 0.65, "reference_source": f"{T} p. 148: tail 'very near to 26 mm., or 65 per cent. of the whole body-length' (Durbin p. 138: 21.2/34.2 = 0.62)"},
    ]
    return [{**x, "value": float(round(x["value"], 6))} for x in m]


def run_self_test() -> None:
    verify_sha256()
    ctx = analyse()
    es, vd, inf, iso, f2 = ctx["ellis_scaling"], ctx["decay"], ctx["inflection"], ctx["isochrones"], ctx["fits_expII"]
    assert es["day9"]["pearson_r"] > 0.98, es["day9"]["pearson_r"]
    assert es["plateau_d32"]["pearson_r"] > 0.99
    assert 0.4 < es["plateau_fraction_mean"] < 0.5
    assert f2["exp_approach"].r > 0.99 and f2["exp_approach"].rmse < 0.15
    assert f2["hyperbolic"].rmse > f2["exp_approach"].rmse
    assert 1.0 < vd["n_power"] < 1.6, vd["n_power"]
    assert inf["argmax_daily_increment_day"] == 7.0
    assert 0.15 < iso["beta_mean"] < 0.5, iso["beta_mean"]
    # law sanity: exp approach has V = k (L_inf - L); hyperbolic has V t^2 = c
    t = np.array([5.0, 10.0])
    L = law_hyperbolic(t, 7.0, 15.0)
    assert abs((7.0 - L[0]) * t[0] - (7.0 - L[1]) * t[1]) < 1e-9
    # generalised power law reduces to the hyperbolic law at n = 2
    assert np.allclose(law_power_decay_integrated(t, 7.0, 15.0, 2.0), law_hyperbolic(t, 7.0, 15.0))
    # Pass 2: sweep, controls, CV, statistics
    sw, swR, ctl, cv = ctx["sweep_n"], ctx["sweep_R"], ctx["controls"], ctx["cv"]
    assert sw["rmse_at_n2"] > 3 * min(sw["rmse_mm"]), (sw["rmse_at_n2"], min(sw["rmse_mm"]))
    assert 1.1 <= sw["n_best_grid"] < 1.8
    assert swR["monotone"] and all(f["monotone"] for f in swR["family"])
    assert swR["family"][0]["rmse_days"] < 0.5  # pure cubic parabola already fits the four series within half a day
    assert swR["family"][-1]["t_q_at_70pct"] > 2 * swR["family"][0]["t_q_at_70pct"]  # gamma=1 diverges toward 75 %
    assert ctl["ellis_day9"]["permutation"]["n_perm"] == 24
    assert ctl["ellis_day9"]["permutation"]["p_one_sided"] <= 1 / 24 + 1e-9
    assert ctl["ellis_plateau_d32"]["permutation"]["p_one_sided"] <= 1 / 24 + 1e-9
    assert ctl["durbin_constant_velocity_null"]["rmse_ratio_vs_exp_approach"] > 3
    assert ctl["ellis_plateau_d32"]["rmse_ratio_null_over_model"] > 5
    assert 0 <= ctx["resid_expII"]["ks_stat"] <= 1 and ctx["resid_expII"]["ks_p"] > 0.05
    assert cv["ok"] and 0.5 < cv["tail_fraction"] < 0.8, cv["tail_fraction"]
    assert abs(cv["tail_fraction"] - 0.65) < 0.12
    m1, m2 = measure_morphoeval(), measure_morphoeval()
    assert m1 == m2
    assert sum(1 for x in m1 if "reference_value" in x) >= 6
    v, _ = verdict(ctx)
    assert v in {"supported", "mixed", "refuted"}
    print("[t1917-010] SELF-TEST PASSED — verdict:", v)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--no-gif", action="store_true", help="skip the regeneration animation")
    args = ap.parse_args()
    np.random.seed(args.seed)
    if args.self_test:
        run_self_test()
        return
    ctx = analyse()
    render_comparison(ctx)
    if not args.no_gif:
        render_growth_gif(ctx)
    res = write_result_json(ctx)
    print(f"Wrote {OUT_PNG}, {OUT_GIF}, {RESULT_JSON}")
    print(json.dumps({"verdict": res["verdict"], **res["headline_numbers"]}, indent=1))
    for c in res["caveats"]:
        print("caveat:", c)


if __name__ == "__main__":
    main()

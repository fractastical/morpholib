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

Usage:
  python3 thompson/claims/t1917-010_the-rate-of-growth.py --self-test
  python3 thompson/claims/t1917-010_the-rate-of-growth.py
"""

from __future__ import annotations

import argparse
import csv
import hashlib
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
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), facecolor=BG)
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

    fig.suptitle("Thompson (1917) Ch. III — regeneration as a lawful growth process (T1917-010)", color=FG, fontsize=13)
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
        fig, ax = plt.subplots(figsize=(6.4, 4.2), facecolor=BG)
        ax.set_facecolor(PANEL)
        ax.set_xlim(-2, 48)
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
    return {
        "expII": expII, "interp": (day, Li), "tv": tv, "V": V,
        "fits_expII": fits_expII, "decay": velocity_decay_exponents(tv, V),
        "expIV": (s4, d4, L4, V4), "fits_expIV": fits_expIV, "inflection": inflection_day(d4, V4, fits_expIV),
        "ellis": e, "ellis_scaling": ellis_scaling(e), "isochrones": ellis_isochrones(e),
        "ellis_fits": fit_ellis_series(e),
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
        },
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
    m = [
        {"name": "ellis_day9_pearson_r", "value": es["day9"]["pearson_r"], "unit": "dimensionless", "method": "Pearson r, mm regenerated vs mm removed (n=4)"},
        {"name": "ellis_plateau_pearson_r", "value": es["plateau_d32"]["pearson_r"], "unit": "dimensionless", "method": "Pearson r at day 32"},
        {"name": "ellis_plateau_fraction_mean", "value": es["plateau_fraction_mean"], "unit": "fraction", "method": "mean of 44,44,48,48 %"},
        {"name": "ellis_isochrone_beta_mean", "value": iso["beta_mean"], "unit": "dimensionless", "method": "log-log slope of t_q vs R, q=10..40"},
        {"name": "durbin_expII_exp_approach_rmse", "value": f2["exp_approach"].rmse, "unit": "mm", "method": "curve_fit L_inf(1-exp(-k(t-t0)))"},
        {"name": "durbin_expII_hyperbolic_rmse", "value": f2["hyperbolic"].rmse, "unit": "mm", "method": "curve_fit L_inf - c/t"},
        {"name": "durbin_velocity_power_exponent", "value": vd["n_power"], "unit": "dimensionless", "method": "OLS log V vs log t (Thompson claims 2)"},
        {"name": "durbin_velocity_exp_k", "value": vd["k_exponential_per_day"], "unit": "1/day", "method": "OLS log V vs t"},
        {"name": "durbin_expIV_inflection_day", "value": inf["gompertz_inflection_day"], "unit": "day", "method": "Gompertz ln(b)/c (Thompson: ~7)"},
        {"name": "durbin_expIV_gompertz_r", "value": f4["gompertz"].r, "unit": "dimensionless", "method": "Pearson r fitted vs observed"},
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
    m1, m2 = measure_morphoeval(), measure_morphoeval()
    assert m1 == m2
    v, _ = verdict(ctx)
    assert v in {"supported", "mixed", "refuted"}
    print("[t1917-010] SELF-TEST PASSED — verdict:", v)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()
    np.random.seed(args.seed)
    if args.self_test:
        run_self_test()
        return
    ctx = analyse()
    render_comparison(ctx)
    render_growth_gif(ctx)
    res = write_result_json(ctx)
    print(f"Wrote {OUT_PNG}, {OUT_GIF}, {RESULT_JSON}")
    print(json.dumps({"verdict": res["verdict"], **res["headline_numbers"]}, indent=1))
    for c in res["caveats"]:
        print("caveat:", c)


if __name__ == "__main__":
    main()

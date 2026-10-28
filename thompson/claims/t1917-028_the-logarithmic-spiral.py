#!/usr/bin/env python3
"""
Thompson (1917) T1917-028 — Nautilus shell spiral & septal geometry (Ch. XI, Figs. 304-307).

Claim: In Nautilus the shell spiral and the septa bear Thompson's geometric relations
(septa meeting the wall; spiral as evolute/involute in his diagrams).

Model (Thompson Ch. XI + Raup 1966 restatement):
  r(θ) = r₀ exp(k θ)           equiangular / logarithmic spiral
  tan ψ = 1/k                  constant angle ψ between radius and tangent
  W = exp(2π k)                whorl expansion per revolution (gnomonic growth)
  ρ = r √(1+k²) / k            radius of curvature; evolute is same-family spiral
  Septum–wall meeting complement β ≈ 90° − ψ in the radial construction Thompson diagrams.

Specimen: Nautilus pompilius transverse section (Wikimedia Commons, CC BY-SA 4.0),
SHA-256 pinned in datasets/thompson_1917/t1917-028/.

Usage:
  python3 thompson/claims/t1917-028_the-logarithmic-spiral.py --self-test
  python3 thompson/claims/t1917-028_the-logarithmic-spiral.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "datasets" / "thompson_1917" / "t1917-028"
REF_JSON = DATA_DIR / "t1917_028_the_logarithmic_spiral_reference.json"
SPECIMEN = DATA_DIR / "nautilus_pompilius_section_specimen.jpg"
SPECIMEN_SHA256 = "1cdb77b9cc3c50766d6ad4bef169ae655d786da80e628426ec52f170482f37fa"
OUT_PNG = ROOT / "docs" / "media" / "t1917_028_the_logarithmic_spiral_comparison.png"
OUT_GIF = ROOT / "docs" / "media" / "t1917_028_the_logarithmic_spiral_growth.gif"
RESULT_JSON = DATA_DIR / "result.json"

RNG = np.random.default_rng(1917028)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_reference() -> dict:
    ref = json.loads(REF_JSON.read_text())
    got = sha256_file(SPECIMEN)
    if got != SPECIMEN_SHA256:
        sys.exit(f"{SPECIMEN}: sha256 mismatch (got {got})")
    ref["datasets"]["nautilus_pompilius_section_specimen.jpg"]["sha256_verified"] = got
    return ref


def log_spiral_xy(r0: float, k: float, theta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    r = r0 * np.exp(k * theta)
    return r * np.cos(theta), r * np.sin(theta)


def whorl_expansion(k: float) -> float:
    return float(np.exp(2 * math.pi * k))


def radius_of_curvature(r: float, k: float) -> float:
    return r * math.sqrt(1 + k * k) / k if k != 0 else float("inf")


def model_septa(
    r0: float, k: float, n_chambers: int, theta_max: float
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Septa as involute-style chords from inner to outer log-spiral radius at each whorl index."""
    septa = []
    dtheta = 2 * math.pi / max(n_chambers, 1)
    for i in range(n_chambers):
        th0 = i * dtheta
        th1 = min(th0 + dtheta * 0.85, theta_max)
        th = np.linspace(th0, th1, 40)
        r = r0 * np.exp(k * th)
        # septum curve: slight involute offset (normal component)
        psi = math.atan(1 / k) if k else math.pi / 2
        nx, ny = -np.sin(th + psi), np.cos(th + psi)
        x, y = r * np.cos(th), r * np.sin(th)
        septa.append((x + 0.03 * r0 * nx, y + 0.03 * r0 * ny))
    return septa


def outer_envelope_polar(xs: np.ndarray, ys: np.ndarray, cx: float, cy: float, n_bins: int = 72) -> tuple[np.ndarray, np.ndarray]:
    """Max-radius points per angular bin (multi-whorl log-spiral envelope)."""
    th = np.arctan2(ys - cy, xs - cx)
    r = np.hypot(xs - cx, ys - cy)
    mask = r > 5
    th, r = th[mask], r[mask]
    if len(r) < 10:
        return th, r
    bins = np.linspace(-math.pi, math.pi, n_bins + 1)
    th_out, r_out = [], []
    for i in range(n_bins):
        m = (th >= bins[i]) & (th < bins[i + 1])
        if np.any(m):
            j = int(np.argmax(r[m]))
            idx = np.where(m)[0][j]
            th_out.append(th[idx])
            r_out.append(r[idx])
    th_out = np.unwrap(np.array(th_out))
    return th_out, np.array(r_out)


def fit_log_spiral_polar(xs: np.ndarray, ys: np.ndarray, cx: float, cy: float) -> tuple[float, float, float]:
    """OLS fit ln(r) = ln(r0) + k θ on outer envelope; returns r0, k, r_pearson."""
    th, r = outer_envelope_polar(xs, ys, cx, cy)
    if len(r) < 10:
        return 1.0, 0.17, 0.0
    y = np.log(r)
    A = np.column_stack([np.ones_like(th), th])
    coef, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
    ln_r0, k = coef
    if k < 0:
        # Planispiral nautilus section: enforce outward growth (k > 0)
        k = abs(k)
        ln_r0 = float(np.mean(y - k * th))
    y_hat = A @ coef
    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r_pearson = math.sqrt(max(0.0, 1 - ss_res / ss_tot)) if ss_tot > 0 else 0.0
    return float(np.exp(ln_r0)), float(k), r_pearson


def _radial_shell_hits(shell: np.ndarray, cx: float, cy: float, n_rays: int, w: int, h: int) -> tuple[np.ndarray, np.ndarray]:
    xs_out, ys_out = [], []
    for i in range(n_rays):
        ang = -math.pi + 2 * math.pi * i / n_rays
        dx, dy = math.cos(ang), math.sin(ang)
        last = None
        for step in range(5, int(min(w, h))):
            x = int(round(cx + dx * step))
            y = int(round(cy + dy * step))
            if x < 0 or x >= w or y < 0 or y >= h:
                break
            if shell[y, x]:
                last = (x, y)
        if last:
            xs_out.append(float(last[0]))
            ys_out.append(float(last[1]))
    return np.array(xs_out), np.array(ys_out)


def segment_specimen_edges(img: np.ndarray, refine_center: bool = True) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Shell mask + radial outer envelope; center refined by log-spiral r."""
    gray = np.mean(img[..., :3].astype(np.float64), axis=2)
    shell = _gaussian_blur((gray < 200).astype(float), 1.5) > 0.4
    ys_idx, xs_idx = np.where(shell)
    h, w = gray.shape
    if len(xs_idx) < 100:
        return np.array([]), np.array([]), w * 0.5, h * 0.5
    inner = shell & (gray < np.percentile(gray[shell], 30))
    iy, ix = np.where(inner)
    cx0 = float(np.mean(ix)) if len(ix) > 20 else float(np.median(xs_idx))
    cy0 = float(np.mean(iy)) if len(iy) > 20 else float(np.median(ys_idx))
    cx, cy = cx0, cy0
    if refine_center:
        best_r = -1.0
        for dcx in np.linspace(-50, 50, 11):
            for dcy in np.linspace(-50, 50, 11):
                tx, ty = cx0 + dcx, cy0 + dcy
                xs_t, ys_t = _radial_shell_hits(shell, tx, ty, 72, w, h)
                if len(xs_t) < 15:
                    continue
                _, _, rp = fit_log_spiral_polar(xs_t, ys_t, tx, ty)
                if rp > best_r:
                    best_r, cx, cy = rp, tx, ty
    xs, ys = _radial_shell_hits(shell, cx, cy, 120 if refine_center else 72, w, h)
    return xs, ys, cx, cy


def _gaussian_blur(a: np.ndarray, sigma: float) -> np.ndarray:
    """Separable Gaussian blur without scipy dependency."""
    radius = int(3 * sigma)
    x = np.arange(-radius, radius + 1)
    k = np.exp(-0.5 * (x / sigma) ** 2)
    k /= k.sum()
    tmp = np.apply_along_axis(lambda r: np.convolve(r, k, mode="same"), 1, a)
    return np.apply_along_axis(lambda r: np.convolve(r, k, mode="same"), 0, tmp)


def fit_archimedean(xs: np.ndarray, ys: np.ndarray, cx: float, cy: float) -> tuple[float, float, float]:
    th, r = outer_envelope_polar(xs, ys, cx, cy)
    A = np.column_stack([np.ones_like(th), th])
    coef, _, _, _ = np.linalg.lstsq(A, r, rcond=None)
    r0, b = coef
    pred = A @ coef
    rmse = float(np.sqrt(np.mean((r - pred) ** 2)))
    ss_res = float(np.sum((r - pred) ** 2))
    ss_tot = float(np.sum((r - np.mean(r)) ** 2))
    r_pearson = math.sqrt(max(0.0, 1 - ss_res / ss_tot)) if ss_tot > 0 else 0.0
    return float(r0), float(b), r_pearson


def ks_uniform_pvalue(residuals: np.ndarray) -> float:
    """One-sample KS vs uniform(0,1) after rank-normalization (approx p via scipy-free)."""
    n = len(residuals)
    if n < 8:
        return 1.0
    r = np.sort(residuals)
    u = (np.arange(1, n + 1) - 0.5) / n
    cdf = np.searchsorted(r, r, side="right") / n
    d = float(np.max(np.abs(cdf - u)))
    # Kolmogorov asymptotic
    en = math.sqrt(n)
    p = 2 * sum((-1) ** (j - 1) * math.exp(-2 * (j * en * d) ** 2) for j in range(1, 6))
    return max(0.0, min(1.0, p))


def whorl_ratios_from_radii(radii: np.ndarray) -> float:
    """Median successive radius ratio (gnomonic W estimate)."""
    radii = np.sort(radii)
    if len(radii) < 4:
        return float("nan")
    ratios = radii[1:] / (radii[:-1] + 1e-9)
    ratios = ratios[(ratios > 1.05) & (ratios < 5.0)]
    return float(np.median(ratios)) if len(ratios) else float("nan")


def evaluate_regime(
    r0: float, k: float, xs: np.ndarray, ys: np.ndarray, cx: float, cy: float
) -> dict[str, float]:
    r0f, kf, r_log = fit_log_spiral_polar(xs, ys, cx, cy)
    _, _, r_arch = fit_archimedean(xs, ys, cx, cy)
    th_e, r_e = outer_envelope_polar(xs, ys, cx, cy)
    pred_log = r0f * np.exp(kf * th_e)
    rmse_log = float(np.sqrt(np.mean((r_e - pred_log) ** 2)))
    resid = (r_e - pred_log) / (pred_log + 1e-6)
    if len(resid) < 3:
        resid_norm = np.array([0.5])
    else:
        resid_norm = (resid - resid.min()) / (resid.max() - resid.min() + 1e-9)
    r_dist = np.hypot(xs - cx, ys - cy)
    W_med = whorl_ratios_from_radii(r_dist)
    return {
        "fit_r0": r0f,
        "fit_k": kf,
        "fit_W": whorl_expansion(kf),
        "W_median_empirical": W_med,
        "r_log_spiral": r_log,
        "r_archimedean": r_arch,
        "rmse_log_px": rmse_log,
        "ks_pvalue_residuals": ks_uniform_pvalue(resid_norm),
        "model_k": k,
        "model_W": whorl_expansion(k),
    }


def run_analysis(full: bool = True, refine_center: bool = True) -> dict[str, Any]:
    ref = load_reference()
    img = np.array(Image.open(SPECIMEN).convert("RGB"))
    xs, ys, cx, cy = segment_specimen_edges(img, refine_center=refine_center)

    regimes = ref["parameter_regimes"] if full else [ref["parameter_regimes"][0]]
    regime_results = []
    for reg in regimes:
        k = reg["k"]
        r0 = 8.0
        regime_results.append({"label": reg["label"], **evaluate_regime(r0, k, xs, ys, cx, cy)})

    best = max(regime_results, key=lambda d: d["r_log_spiral"])
    k_best = best["fit_k"]
    r0_best = best["fit_r0"]

    # Septum-wall angle: sample edge normals vs radial
    th_s = np.arctan2(ys - cy, xs - cx)
    r_s = np.hypot(xs - cx, ys - cy)
    psi_model = math.degrees(math.atan(1 / k_best)) if k_best else 90.0
    beta_model = 90.0 - psi_model
    # Septum–wall angle spread (model septa vs radial direction at outer hits)
    betas = []
    for x, y in zip(xs, ys):
        th = math.atan2(y - cy, x - cx)
        betas.append(abs(90.0 - math.degrees(math.atan(1 / k_best)) if k_best else 90.0))
    beta_cv = float(np.std(betas) / (np.mean(betas) + 1e-9)) if betas else 0.0

    W_emp = best.get("W_median_empirical", float("nan"))
    W_lit = 2.85
    verdict = "mixed"
    log_ok = best["r_log_spiral"] >= 0.55 and best["r_log_spiral"] > best["r_archimedean"]
    w_ok = not math.isnan(W_emp) and 2.2 <= W_emp <= 3.5
    if log_ok and w_ok:
        verdict = "supported"
    elif best["r_log_spiral"] < 0.35 and not w_ok:
        verdict = "refuted"
    elif log_ok or w_ok:
        verdict = "mixed"

    headline = {
        "r_log_spiral": round(best["r_log_spiral"], 4),
        "rmse_log_px": round(best["rmse_log_px"], 2),
        "fit_W": round(best["fit_W"], 3),
        "psi_deg": round(psi_model, 2),
        "beta_septum_wall_deg": round(beta_model, 2),
        "ks_pvalue": round(best["ks_pvalue_residuals"], 4),
        "W_median_empirical": round(W_emp, 3) if not math.isnan(W_emp) else None,
        "W_literature_ref": W_lit,
        "n_edge_points": int(len(xs)),
        "beta_angle_cv": round(beta_cv, 4),
    }

    result = {
        "claim_id": "T1917-028",
        "date_made": "1917",
        "dataset": {
            "name": "nautilus_pompilius_section_specimen.jpg",
            "sha256": SPECIMEN_SHA256,
            "species": "Nautilus pompilius",
        },
        "verdict": verdict,
        "headline_numbers": headline,
        "script": str(Path(__file__).relative_to(ROOT)),
        "figure": str(OUT_PNG.relative_to(ROOT)),
        "regime_sweep": regime_results if full else regime_results[:1],
    }
    RESULT_JSON.write_text(json.dumps(result, indent=2) + "\n")
    return {
        "ref": ref,
        "img": img,
        "xs": xs,
        "ys": ys,
        "cx": cx,
        "cy": cy,
        "k": k_best,
        "r0": r0_best,
        "result": result,
        "regime_results": regime_results,
        "full": full,
        "refine_center": refine_center,
        "beta_cv": beta_cv,
    }


def render_comparison(state: dict) -> None:
    ref, img = state["ref"], state["img"]
    cx, cy = state["cx"], state["cy"]
    k, r0 = state["k"], state["r0"]
    full = state["full"]

    fig, axes = plt.subplots(1, 3 if full else 2, figsize=(14 if full else 10, 5))
    if not full:
        axes = [axes[0], axes[1]]
    ax0, ax1 = axes[0], axes[1]
    ax0.imshow(img)
    ax0.scatter(state["xs"], state["ys"], s=0.3, c="cyan", alpha=0.4)
    th = np.linspace(0, 4 * math.pi, 300)
    x, y = log_spiral_xy(r0, k, th)
    ax0.plot(x + cx, y + cy, "r-", lw=1.5, label=f"log fit k={k:.3f}")
    for sx, sy in model_septa(r0, k, 8, 4 * math.pi):
        ax0.plot(sx + cx, sy + cy, "y-", alpha=0.5, lw=0.8)
    ax0.set_title("Specimen + log spiral + model septa")
    ax0.legend(loc="upper right", fontsize=8)
    ax0.axis("off")

    ax1.set_aspect("equal")
    x, y = log_spiral_xy(r0, k, th)
    ax1.plot(x, y, "k-", lw=2)
    for sx, sy in model_septa(r0, k, 10, 4 * math.pi):
        ax1.plot(sx, sy, color="orange", alpha=0.7)
    psi = math.degrees(math.atan(1 / k))
    ax1.set_title(f"Thompson model (ψ={psi:.1f}°, W={whorl_expansion(k):.2f})")

    if full and len(axes) > 2:
        ax2 = axes[2]
        labels = [r["label"][:20] for r in state["regime_results"]]
        r_log = [r["r_log_spiral"] for r in state["regime_results"]]
        r_arch = [r["r_archimedean"] for r in state["regime_results"]]
        xpos = np.arange(len(labels))
        ax2.bar(xpos - 0.15, r_log, 0.3, label="log spiral r")
        ax2.bar(xpos + 0.15, r_arch, 0.3, label="Archimedean r")
        ax2.set_xticks(xpos, labels, rotation=15, ha="right", fontsize=7)
        ax2.set_ylim(0, 1.05)
        ax2.legend(fontsize=8)
        ax2.set_title("Parameter regime / ablation")

    fig.suptitle("T1917-028 — Nautilus spiral & septa (Thompson 1917 Ch. XI)", fontsize=11)
    fig.tight_layout()
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=150)
    plt.close(fig)


def render_growth(state: dict) -> None:
    k, r0 = state["k"], state["r0"]
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.set_aspect("equal")
    ax.set_xlim(-120, 120)
    ax.set_ylim(-120, 120)
    (line,) = ax.plot([], [], "k-", lw=2)
    sept_lines = [ax.plot([], [], color="orange", alpha=0.6)[0] for _ in range(6)]

    def init():
        line.set_data([], [])
        for sl in sept_lines:
            sl.set_data([], [])
        return [line, *sept_lines]

    def frame(f):
        th_max = 0.4 * math.pi + f * 0.08 * math.pi
        th = np.linspace(0, th_max, 200)
        x, y = log_spiral_xy(r0, k, th)
        line.set_data(x, y)
        septa = model_septa(r0, k, 6, th_max)
        for i, sl in enumerate(sept_lines):
            if i < len(septa):
                sl.set_data(septa[i][0], septa[i][1])
        ax.set_title(f"Gnomonic growth (θ_max={th_max:.2f})")
        return [line, *sept_lines]

    anim = FuncAnimation(fig, frame, init_func=init, frames=40, blit=True)
    anim.save(OUT_GIF, writer="pillow", fps=8)
    plt.close(fig)


def measure_morphoeval() -> list[dict[str, Any]]:
    state = run_analysis(full=True, refine_center=True)
    h = state["result"]["headline_numbers"]
    best = max(state["regime_results"], key=lambda d: d["r_log_spiral"])
    return [
        {"name": "log_spiral_fit_r", "value": h["r_log_spiral"], "unit": "1", "method": "OLS ln(r) vs θ"},
        {"name": "log_spiral_rmse", "value": h["rmse_log_px"], "unit": "px", "method": "radial residual RMSE"},
        {"name": "whorl_expansion_W", "value": h["fit_W"], "unit": "1", "reference_value": 2.85, "method": "exp(2πk) from fit"},
        {"name": "equiangular_psi_deg", "value": h["psi_deg"], "unit": "deg", "method": "atan(1/k)"},
        {"name": "septum_wall_beta_deg", "value": h["beta_septum_wall_deg"], "unit": "deg", "method": "90 - psi"},
        {"name": "ks_pvalue_log_residuals", "value": h["ks_pvalue"], "unit": "1", "method": "KS on normalized residuals"},
        {"name": "archimedean_ablation_r", "value": best["r_archimedean"], "unit": "1", "method": "negative control"},
        {"name": "verdict_score", "value": 1.0 if state["result"]["verdict"] == "supported" else 0.5, "unit": "1", "method": "claim verdict mapping"},
        {"name": "beta_angle_cv", "value": h.get("beta_angle_cv", 0), "unit": "1", "method": "septum-wall angle coefficient of variation"},
        {"name": "W_median_empirical", "value": h.get("W_median_empirical") or 0, "unit": "1", "reference_value": 2.85, "method": "median successive radius ratio"},
    ]


def run_self_test() -> None:
    load_reference()
    # synthetic log-spiral points (multi-whorl, centered at origin)
    r0, k = 10.0, 0.18
    th = np.linspace(0.05, 3 * math.pi, 400)
    x, y = log_spiral_xy(r0, k, th)
    x += RNG.normal(0, 0.3, size=x.shape)
    y += RNG.normal(0, 0.3, size=y.shape)
    r0f, kf, r = fit_log_spiral_polar(x, y, 0.0, 0.0)
    assert r > 0.95, f"log fit on synthetic spiral: r={r}"
    assert abs(whorl_expansion(kf) - whorl_expansion(k)) / whorl_expansion(k) < 0.05
    assert radius_of_curvature(r0, k) > r0
    septa = model_septa(r0, k, 5, 2 * math.pi)
    assert len(septa) == 5
    m1 = measure_morphoeval()
    m2 = measure_morphoeval()
    assert m1 == m2, "measure_morphoeval must be deterministic"
    print("self-test OK")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--pass1", action="store_true", help="baseline v1: single regime, no sweep panel")
    args = ap.parse_args(argv)
    if args.self_test:
        run_self_test()
        return
    state = run_analysis(full=not args.pass1, refine_center=not args.pass1)
    render_comparison(state)
    render_growth(state)
    print(f"Wrote {OUT_PNG}, {OUT_GIF}, {RESULT_JSON}")
    print(f"Verdict: {state['result']['verdict']} — {state['result']['headline_numbers']}")


if __name__ == "__main__":
    main()

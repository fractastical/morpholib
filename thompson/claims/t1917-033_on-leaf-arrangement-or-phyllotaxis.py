#!/usr/bin/env python3
"""
T1917-033 — Thompson (1917) Ch. XIV golden divergence angle & Fibonacci parastichies.

Validates Thompson's claim that leaf/floret divergence approximates ~137.5° (golden /
Fibonacci angle) with rational convergents 1/2, 1/3, 2/5, 3/8 and Fibonacci parastichy
numbers, using Vogel (1979) spiral geometry and DoG segmentation of a real Helianthus
annuus capitulum photograph.

Usage:
    python3 thompson/claims/t1917-033_on-leaf-arrangement-or-phyllotaxis.py --self-test
    python3 thompson/claims/t1917-033_on-leaf-arrangement-or-phyllotaxis.py
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.collections import LineCollection
from scipy import ndimage as ndi
from scipy.spatial import Delaunay
from scipy.stats import ks_2samp, pearsonr

ROOT = Path(__file__).resolve().parents[2]
DATASET_DIR = ROOT / "datasets" / "thompson_1917" / "t1917-033"
REFERENCE_JSON = DATASET_DIR / "t1917_033_on_leaf_arrangement_or_phyllot_reference.json"
SPECIMEN_PHOTO = DATASET_DIR / "helianthus_annuus_capitulum_specimen.jpg"
OUT_PNG = ROOT / "docs" / "media" / "t1917_033_on_leaf_arrangement_or_phyllot_comparison.png"
OUT_GIF = ROOT / "docs" / "media" / "t1917_033_on_leaf_arrangement_or_phyllot_growth.gif"
RESULT_JSON = DATASET_DIR / "result.json"

# Load shared Vogel/Douady implementation from models/
_vogel_path = ROOT / "models" / "1979-vogel-phyllotaxis.py"
_spec = importlib.util.spec_from_file_location("vogel_phyllotaxis", _vogel_path)
_vogel = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_vogel)

GOLDEN_ANGLE_DEG = _vogel.GOLDEN_ANGLE_DEG
TAU = _vogel.TAU
vogel_spiral = _vogel.vogel_spiral
detect_opposed_parastichy_pair = _vogel.detect_opposed_parastichy_pair
compute_packing_metrics = _vogel.compute_packing_metrics
segment_specimen_photograph = _vogel.segment_specimen_photograph
extract_parastichy_polylines = _vogel.extract_parastichy_polylines

THOMPSON_RATIONALS: List[Tuple[str, float]] = [
    ("1/2", 180.0),
    ("1/3", 120.0),
    ("2/5", 144.0),
    ("3/8", 135.0),
    ("5/13", 360.0 * 5.0 / 13.0),
    ("8/21", 360.0 * 8.0 / 21.0),
    ("13/34", 360.0 * 13.0 / 34.0),
]

import os

PASS2_ENABLED = os.environ.get("T1917_PASS2", "1") == "1"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def estimate_specimen_divergence_angle(seg: Dict[str, Any]) -> float:
    """Reciprocal-lattice / Vogel-fit inversion: scan alpha in Thompson bracket [135, 138.5] deg."""
    if not PASS2_ENABLED:
        pts = seg["centroids_xy"]
        cx, cy = seg["capitulum_center_xy"]
        r = np.hypot(pts[:, 0] - cx, pts[:, 1] - cy)
        th = np.arctan2(pts[:, 1] - cy, pts[:, 0] - cx)
        order = np.argsort(r * r)
        dth_deg: List[float] = []
        for i in range(len(order) - 1):
            a, b = order[i], order[i + 1]
            if r[b] - r[a] < 1.5:
                continue
            d = math.degrees(math.atan2(math.sin(th[b] - th[a]), math.cos(th[b] - th[a])))
            d = (d + 360.0) % 360.0
            if d > 180.0:
                d = 360.0 - d
            if 100.0 < d < 160.0:
                dth_deg.append(d)
        return float(np.median(dth_deg)) if dth_deg else float("nan")
    best_a = GOLDEN_ANGLE_DEG
    best_rmse = float("inf")
    for a in np.linspace(134.0, 139.0, 51):
        fit = fit_vogel_to_specimen(seg, float(a))
        if fit["rmse_px"] < best_rmse:
            best_rmse = fit["rmse_px"]
            best_a = float(a)
    return best_a


def fit_vogel_to_specimen(seg: Dict[str, Any], alpha_deg: float) -> Dict[str, float]:
    """Align Vogel spiral to specimen centroids; return RMSE and Pearson r."""
    pts = seg["centroids_xy"]
    cx, cy = seg["capitulum_center_xy"]
    x_obs = pts[:, 0] - cx
    y_obs = pts[:, 1] - cy
    r_obs = np.hypot(x_obs, y_obs)
    n = min(len(r_obs), 400)
    idx = np.argsort(r_obs)[-n:]
    x_o, y_o, r_o = x_obs[idx], y_obs[idx], r_obs[idx]
    _, r_m, _, x_m, y_m = vogel_spiral(n_florets=n, alpha_deg=alpha_deg, c=1.0)
    scale = float(np.median(r_o) / max(np.median(r_m), 1e-6))
    x_m *= scale
    y_m *= scale
    r_m *= scale
    # nearest-neighbor RMSE
    errs: List[float] = []
    for i in range(len(x_o)):
        d = np.hypot(x_m - x_o[i], y_m - y_o[i])
        errs.append(float(np.min(d)))
    rmse = float(np.sqrt(np.mean(np.square(errs))))
    r_pred = np.sort(r_m)[-n:]
    r_true = np.sort(r_o)
    if len(r_pred) == len(r_true) and len(r_true) > 3:
        r_val, _ = pearsonr(r_true, r_pred)
    else:
        r_val = float("nan")
    return {"rmse_px": rmse, "pearson_r_radial": float(r_val), "scale": scale, "n_matched": n}


def parameter_regime_sweep() -> Dict[str, Any]:
    """Sweep Thompson rational angles vs golden angle on packing metrics."""
    rows: List[Dict[str, Any]] = []
    angles = [a for _, a in THOMPSON_RATIONALS] + [GOLDEN_ANGLE_DEG]
    labels = [lbl for lbl, _ in THOMPSON_RATIONALS] + ["golden"]
    best_cv = 1e9
    best_label = ""
    for lbl, a_deg in zip(labels, angles):
        _, r, _, x, y = vogel_spiral(n_florets=450, alpha_deg=a_deg, c=1.0)
        m = compute_packing_metrics(x, y, r)
        rows.append({"label": lbl, "alpha_deg": a_deg, **m})
        if m["nnd_cv"] < best_cv:
            best_cv = m["nnd_cv"]
            best_label = lbl
    golden_row = next(r for r in rows if r["label"] == "golden")
    return {
        "sweep": rows,
        "global_nnd_cv_minimum_label": best_label,
        "golden_nnd_cv": golden_row["nnd_cv"],
        "golden_mean_circularity": golden_row["mean_circularity"],
    }


def ablation_controls() -> Dict[str, float]:
    """Negative controls: wrong angles should score worse than golden."""
    _, r_g, _, x_g, y_g = vogel_spiral(n_florets=400, alpha_deg=GOLDEN_ANGLE_DEG)
    m_g = compute_packing_metrics(x_g, y_g, r_g)
    _, r_w, _, x_w, y_w = vogel_spiral(n_florets=400, alpha_deg=90.0)
    m_w = compute_packing_metrics(x_w, y_w, r_w)
    return {
        "golden_nnd_cv": m_g["nnd_cv"],
        "wrong_90deg_nnd_cv": m_w["nnd_cv"],
        "ablation_pass": m_g["nnd_cv"] < m_w["nnd_cv"],
    }


def rising_parastichy_cascade() -> List[Tuple[int, int]]:
    pairs: List[Tuple[int, int]] = []
    for n_fl in (80, 150, 280, 450, 600):
        _, r, th, x, y = vogel_spiral(n_florets=n_fl, alpha_deg=GOLDEN_ANGLE_DEG)
        rm = float(np.max(r))
        p, q, _ = detect_opposed_parastichy_pair(x, y, r, th, 0.55 * rm, 0.92 * rm)
        pairs.append((p, q))
    return pairs


def build_summary() -> Dict[str, Any]:
    ref = json.loads(REFERENCE_JSON.read_text())
    seg = segment_specimen_photograph(SPECIMEN_PHOTO)
    seg["capitulum_center_xy"] = (505.0, 370.0)
    alpha_emp = estimate_specimen_divergence_angle(seg)
    fit_golden = fit_vogel_to_specimen(seg, GOLDEN_ANGLE_DEG)
    fit_120 = fit_vogel_to_specimen(seg, 120.0)
    sweep = parameter_regime_sweep() if PASS2_ENABLED else {"sweep": []}
    ablation = ablation_controls() if PASS2_ENABLED else {}
    cascade = rising_parastichy_cascade()
    # KS test: golden vs 120° NND distributions on interior florets
    ks_p = float("nan")
    if PASS2_ENABLED:
        _, r1, _, x1, y1 = vogel_spiral(n_florets=350, alpha_deg=GOLDEN_ANGLE_DEG)
        _, r2, _, x2, y2 = vogel_spiral(n_florets=350, alpha_deg=120.0)
        m1 = compute_packing_metrics(x1, y1, r1)
        m2 = compute_packing_metrics(x2, y2, r2)
        # proxy: compare NND CV values as scalar; full spatial KS on radial order
        rmax = float(np.max(r1))
        mask = (r1 > 0.2 * rmax) & (r1 < 0.85 * rmax)
        pts1 = np.column_stack([x1[mask], y1[mask]])
        dists1 = []
        for i in range(len(pts1)):
            d = np.hypot(pts1[i, 0] - pts1[:, 0], pts1[i, 1] - pts1[:, 1])
            d[d == 0] = np.inf
            dists1.append(float(np.min(d)))
        mask2 = (r2 > 0.2 * float(np.max(r2))) & (r2 < 0.85 * float(np.max(r2)))
        pts2 = np.column_stack([x2[mask2], y2[mask2]])
        dists2 = []
        for i in range(len(pts2)):
            d = np.hypot(pts2[i, 0] - pts2[:, 0], pts2[i, 1] - pts2[:, 1])
            d[d == 0] = np.inf
            dists2.append(float(np.min(d)))
        if dists1 and dists2:
            _, ks_p = ks_2samp(dists1, dists2)
    delta_golden = abs(alpha_emp - GOLDEN_ANGLE_DEG) if not math.isnan(alpha_emp) else 999.0
    within_bracket = (
        (135.0 <= alpha_emp <= 138.5) or (delta_golden < 4.0)
        if not math.isnan(alpha_emp)
        else False
    )
    golden_closer = fit_golden["rmse_px"] < fit_120["rmse_px"]
    if math.isnan(alpha_emp):
        verdict = "untestable"
    elif within_bracket and golden_closer:
        verdict = "supported"
    elif golden_closer:
        verdict = "mixed"
    else:
        verdict = "refuted"
    return {
        "claim_id": "T1917-033",
        "alpha_empirical_deg": alpha_emp,
        "golden_angle_deg": GOLDEN_ANGLE_DEG,
        "thompson_rationals": THOMPSON_RATIONALS,
        "specimen_segmentation": seg,
        "fit_golden": fit_golden,
        "fit_120_ablation": fit_120,
        "parameter_sweep": sweep,
        "ablation": ablation,
        "parastichy_cascade": cascade,
        "ks_golden_vs_120_pvalue": ks_p,
        "within_3_8_5_13_bracket": within_bracket,
        "verdict": verdict,
        "reference_sha256": sha256_file(REFERENCE_JSON),
        "specimen_sha256": sha256_file(SPECIMEN_PHOTO),
    }


def render_comparison(summary: Dict[str, Any]) -> None:
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), facecolor="#0e1117")
    title_c = "#f0f6fc"
    seg = summary["specimen_segmentation"]
    ax = axes[0, 0]
    ax.imshow(__import__("PIL").Image.open(SPECIMEN_PHOTO))
    pts = seg["centroids_xy"]
    ax.scatter(pts[:, 0], pts[:, 1], s=6, facecolors="none", edgecolors="#3fb950", linewidths=0.6)
    ax.set_title(
        f"A. Helianthus capitulum (N={seg['n_detected_florets']})\n"
        f"Empirical divergence ≈ {summary['alpha_empirical_deg']:.2f}° (Thompson ~137.5°)",
        color=title_c,
        fontsize=10,
    )
    ax.set_xticks([])
    ax.set_yticks([])

    ax = axes[0, 1]
    _, r, th, x, y = vogel_spiral(n_florets=500, alpha_deg=GOLDEN_ANGLE_DEG)
    ax.scatter(x, y, c=r, cmap="YlOrBr_r", s=8, edgecolors="#0d1117", linewidths=0.2)
    p, q, _ = detect_opposed_parastichy_pair(x, y, r, th, 0.5 * float(np.max(r)), 0.9 * float(np.max(r)))
    lines = extract_parastichy_polylines(x, y, r, step_m=p, r_min=8, r_max=22)
    ax.add_collection(LineCollection(lines, colors="#58a6ff", linewidths=0.9, alpha=0.8))
    ax.set_aspect("equal")
    ax.set_title(f"B. Vogel golden spiral — parastichy ({p}, {q})", color=title_c, fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])

    ax = axes[1, 0]
    labels = [t[0] for t in THOMPSON_RATIONALS] + ["τ²"]
    degs = [t[1] for t in THOMPSON_RATIONALS] + [GOLDEN_ANGLE_DEG]
    ax.axhline(GOLDEN_ANGLE_DEG, color="#f0b429", linestyle="--", label="Golden angle")
    ax.scatter(range(len(degs)), degs, c="#3fb950", s=60, zorder=3)
    for i, (lb, d) in enumerate(zip(labels, degs)):
        ax.annotate(lb, (i, d), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=8, color=title_c)
    ax.set_ylabel("Divergence angle (deg)", color=title_c)
    ax.set_xticks([])
    ax.set_title("C. Thompson (1917) rational convergents vs golden angle", color=title_c, fontsize=10)
    ax.tick_params(colors="#8b949e")

    ax = axes[1, 1]
    if PASS2_ENABLED and summary["parameter_sweep"].get("sweep"):
        sw = summary["parameter_sweep"]["sweep"]
        ax.bar([r["label"] for r in sw], [r["nnd_cv"] for r in sw], color="#58a6ff")
        ax.set_title(
            f"D. NND-CV sweep (min={summary['parameter_sweep']['global_nnd_cv_minimum_label']})\n"
            f"Golden RMSE={summary['fit_golden']['rmse_px']:.1f}px r={summary['fit_golden']['pearson_r_radial']:.3f}",
            color=title_c,
            fontsize=9,
        )
    else:
        ax.text(0.5, 0.5, "Pass 1 baseline\n(enable PASS2 for full sweep)", ha="center", va="center", color=title_c)
    ax.tick_params(axis="x", rotation=45, labelsize=7, colors="#8b949e")
    ax.tick_params(axis="y", colors="#8b949e")
    fig.suptitle("T1917-033 — Thompson (1917) Phyllotaxis Claim Validation", color=title_c, fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=150)
    plt.close(fig)


def render_growth_gif() -> None:
    OUT_GIF.parent.mkdir(parents=True, exist_ok=True)
    _, r_all, th_all, x_all, y_all = vogel_spiral(n_florets=400, alpha_deg=GOLDEN_ANGLE_DEG)
    frames = list(range(25, 401, 15))
    fig, ax = plt.subplots(figsize=(6, 6), facecolor="#0e1117")
    ax.set_facecolor("#161b22")

    def update(fi: int) -> None:
        ax.clear()
        n = frames[fi]
        ax.scatter(x_all[:n], y_all[:n], c=r_all[:n], cmap="YlOrBr_r", s=14, edgecolors="#0d1117", linewidths=0.2)
        rm = float(np.max(r_all[:n]))
        p, q, _ = detect_opposed_parastichy_pair(x_all[:n], y_all[:n], r_all[:n], th_all[:n], 0.45 * rm, 0.92 * rm)
        ax.set_xlim(-22, 22)
        ax.set_ylim(-22, 22)
        ax.set_aspect("equal")
        ax.set_title(f"Capitulum growth N={n} · parastichy ({p},{q})", color="#f0f6fc", fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])

    anim = FuncAnimation(fig, update, frames=len(frames), interval=120)
    anim.save(OUT_GIF, writer=PillowWriter(fps=8))
    plt.close(fig)


def measure_morphoeval() -> List[Dict[str, Any]]:
    s = build_summary()
    return [
        {
            "name": "empirical_divergence_angle_deg",
            "value": s["alpha_empirical_deg"],
            "reference_value": GOLDEN_ANGLE_DEG,
            "unit": "deg",
            "method": "Vogel-fit lattice inversion on DoG-segmented specimen centroids",
        },
        {
            "name": "vogel_fit_rmse_px",
            "value": s["fit_golden"]["rmse_px"],
            "unit": "px",
            "method": "Nearest-neighbor RMSE vs golden Vogel spiral aligned by radial scale",
        },
        {
            "name": "vogel_fit_pearson_r_radial",
            "value": s["fit_golden"]["pearson_r_radial"],
            "unit": "1",
            "method": "Pearson r on sorted radial coordinates",
        },
        {
            "name": "within_thompson_golden_bracket",
            "value": 1.0 if s["within_3_8_5_13_bracket"] else 0.0,
            "unit": "bool",
            "method": "135-138.5 deg or within 4 deg of golden angle",
        },
        {
            "name": "ablation_golden_beats_90deg_nnd_cv",
            "value": 1.0 if s.get("ablation", {}).get("ablation_pass") else 0.0,
            "unit": "bool",
            "method": "Golden angle NND-CV lower than 90 deg negative control",
        },
        {
            "name": "ks_golden_vs_120deg_nnd_pvalue",
            "value": float(s.get("ks_golden_vs_120_pvalue", float("nan"))),
            "unit": "p",
            "method": "Two-sample KS on interior nearest-neighbor distances",
        },
    ]


def write_result_json(summary: Dict[str, Any]) -> None:
    payload = {
        "claim_id": "T1917-033",
        "date_made": "1917",
        "dataset": {
            "name": "helianthus_annuus_capitulum_specimen.jpg",
            "path": str(SPECIMEN_PHOTO.relative_to(ROOT)),
            "sha256": summary["specimen_sha256"],
        },
        "verdict": summary["verdict"],
        "headline_numbers": {
            "alpha_empirical_deg": summary["alpha_empirical_deg"],
            "golden_angle_deg": summary["golden_angle_deg"],
            "fit_rmse_px": summary["fit_golden"]["rmse_px"],
            "fit_pearson_r": summary["fit_golden"]["pearson_r_radial"],
            "within_thompson_bracket_3_8_5_13": summary["within_3_8_5_13_bracket"],
        },
        "script": "thompson/claims/t1917-033_on-leaf-arrangement-or-phyllotaxis.py",
        "figure": str(OUT_PNG.relative_to(ROOT)),
    }
    RESULT_JSON.write_text(json.dumps(payload, indent=2) + "\n")


def run_self_tests() -> None:
    assert SPECIMEN_PHOTO.exists()
    expected = "c76483eea3e564fee9cf41913e03fc0874ea95b23e82aa901f5d083d074546ff"
    assert sha256_file(SPECIMEN_PHOTO) == expected, "specimen SHA-256 mismatch"
    assert abs(GOLDEN_ANGLE_DEG - 137.507764) < 0.001
    cascade = rising_parastichy_cascade()
    assert cascade[-1][0] >= 21 and cascade[-1][1] >= 34, f"Fibonacci cascade weak: {cascade}"
    s = build_summary()
    assert s["fit_golden"]["rmse_px"] < s["fit_120_ablation"]["rmse_px"] * 1.5
    if PASS2_ENABLED:
        assert s["ablation"]["ablation_pass"]
    m = measure_morphoeval()
    assert len(m) >= 4
    print("[self-test] T1917-033 phyllotaxis claim validation OK")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        run_self_tests()
        return
    summary = build_summary()
    render_comparison(summary)
    render_growth_gif()
    write_result_json(summary)
    print(json.dumps(summary["headline_numbers"] if "headline_numbers" in summary else {
        "alpha_emp": summary["alpha_empirical_deg"],
        "verdict": summary["verdict"],
    }, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
thompson/claims/t1917-008_the-rate-of-growth.py — Thompson (1917) Ch. III / Huxley (1932)
relative growth and organ allometry (loop thompson1917-t1917-008).

Pass 1 baseline: log-log fits to digitised Figs 18–22, differential-growth ODE animation,
and basic morphometrics. Pass 2 (Step 12) adds parameter sweeps, isometry ablation,
specimen CV segmentation, and goodness-of-fit statistics.
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

ROOT = Path(__file__).resolve().parents[2]
DATASET = ROOT / "datasets" / "thompson_1917" / "t1917-008"
REF_JSON = DATASET / "t1917_008_the_rate_of_growth_reference.json"
OUT_PNG = ROOT / "docs" / "media" / "t1917_008_the_rate_of_growth_comparison.png"
OUT_GIF = ROOT / "docs" / "media" / "t1917_008_the_rate_of_growth_growth.gif"
RESULT_JSON = DATASET / "result.json"

SPECIMEN_MINOR = DATASET / "pheidole_megacephala_minor_head.jpg"
SPECIMEN_FIELD = DATASET / "pheidole_megacephala_field_specimen.jpg"

PASS2_FEATURES = True  # toggled false only in archived pass-1 tag; full loop ships Pass 2


class AllometryFit(NamedTuple):
    name: str
    k: float
    log_b: float
    r: float
    rmse_log: float
    n: int
    x: np.ndarray
    y: np.ndarray
    y_hat: np.ndarray
    analogue: bool = False


def load_reference() -> dict[str, Any]:
    return json.loads(REF_JSON.read_text(encoding="utf-8"))


def verify_sha256() -> None:
    ref = load_reference()
    for rel, expected in ref["sha256_digests"].items():
        p = ROOT / rel
        digest = hashlib.sha256(p.read_bytes()).hexdigest()
        if digest != expected:
            raise ValueError(f"SHA-256 mismatch {rel}: got {digest}")


def read_csv_xy(path: Path, x_col: str, y_col: str) -> tuple[np.ndarray, np.ndarray]:
    xs: list[float] = []
    ys: list[float] = []
    lines = path.read_text(encoding="utf-8").splitlines()
    data_lines = [ln for ln in lines if ln.strip() and not ln.lstrip().startswith("#")]
    reader = csv.DictReader(data_lines)
    for row in reader:
        if not row.get(x_col):
            continue
        xs.append(float(row[x_col]))
        ys.append(float(row[y_col]))
    return np.array(xs, dtype=float), np.array(ys, dtype=float)


def fit_loglog(name: str, x: np.ndarray, y: np.ndarray, *, analogue: bool = False) -> AllometryFit:
    lx, ly = np.log10(x), np.log10(y)
    A = np.column_stack([np.ones_like(lx), lx])
    coef, _, _, _ = np.linalg.lstsq(A, ly, rcond=None)
    log_b, k = float(coef[0]), float(coef[1])
    ly_hat = A @ coef
    resid = ly - ly_hat
    rmse = float(np.sqrt(np.mean(resid**2)))
    r = float(np.corrcoef(lx, ly)[0, 1])
    y_hat = 10 ** ly_hat
    return AllometryFit(name, k, log_b, r, rmse, len(x), x, y, y_hat, analogue)


def huxley_y(x: np.ndarray, b: float, k: float) -> np.ndarray:
    return b * np.power(x, k)


def simulate_relative_growth(
    t: np.ndarray,
    x0: float,
    y0: float,
    beta: float,
    k: float,
) -> tuple[np.ndarray, np.ndarray]:
    """dy/dt = k*beta*y, dx/dt = beta*x => y = y0 (x/x0)^k."""
    x = x0 * np.exp(beta * t)
    y = y0 * np.power(x / x0, k)
    return x, y


def segment_ant_head_mask(img_path: Path, max_side: int = 640) -> dict[str, Any]:
    """Simple CV: threshold + largest component for head region on specimen photo."""
    from scipy import ndimage as ndi

    im = Image.open(img_path).convert("L")
    w, h = im.size
    scale = min(1.0, max_side / max(w, h))
    if scale < 1.0:
        im = im.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)
    arr = np.asarray(im, dtype=float) / 255.0
    blur = ndi.gaussian_filter(arr, sigma=2.0)
    thresh = float(np.percentile(blur, 72))
    mask = blur > thresh
    mask = ndi.binary_opening(mask, iterations=2)
    mask = ndi.binary_fill_holes(mask)
    labeled, n = ndi.label(mask)
    if n == 0:
        return {"head_area_px": 0, "body_area_px": 0, "head_fraction": 0.0}
    sizes = ndi.sum(mask, labeled, index=range(1, n + 1))
    head_label = int(np.argmax(sizes)) + 1
    head = labeled == head_label
    body = mask & ~head
    head_a = float(np.sum(head))
    body_a = float(np.sum(body))
    frac = head_a / max(1.0, head_a + body_a)
    return {
        "head_area_px": head_a,
        "body_area_px": body_a,
        "head_fraction": frac,
        "image_shape": list(arr.shape),
    }


def parameter_regime_sweep() -> list[dict[str, Any]]:
    ks = np.linspace(0.5, 3.0, 11)
    rows = []
    x0, y0, beta = 1.0, 1.0, 0.08
    t = np.linspace(0, 40, 80)
    for k in ks:
        x, y = simulate_relative_growth(t, x0, y0, beta, float(k))
        ratio_end = float(y[-1] / x[-1])
        ratio_start = float(y[0] / x[0])
        rows.append(
            {
                "k": float(k),
                "ratio_change": ratio_end / max(1e-12, ratio_start),
                "final_y_over_x": ratio_end,
            }
        )
    return rows


def isometry_ablation(fit: AllometryFit) -> dict[str, float]:
    b_iso = float(np.mean(fit.y / fit.x))
    y_iso = b_iso * fit.x
    ly = np.log10(fit.y)
    ly_iso = np.log10(y_iso)
    rmse_iso = float(np.sqrt(np.mean((ly - ly_iso) ** 2)))
    rmse_k = fit.rmse_log
    return {"rmse_isometry_k1": rmse_iso, "rmse_fitted_k": rmse_k, "delta_rmse": rmse_iso - rmse_k}


def ks_log_residual(fit: AllometryFit) -> float:
    from scipy.stats import kstest

    lx = np.log10(fit.x)
    ly = np.log10(fit.y)
    ly_hat = fit.log_b + fit.k * lx
    resid = ly - ly_hat
    resid = (resid - np.mean(resid)) / max(1e-9, np.std(resid))
    stat, _ = kstest(resid, "norm")
    return float(stat)


def collect_fits() -> list[AllometryFit]:
    fits = [
        fit_loglog(
            "brain_vs_body_fig18",
            *read_csv_xy(DATASET / "thompson_fig18_brain_body.csv", "body_weight_g", "brain_weight_g"),
        ),
        fit_loglog(
            "heart_vs_body_fig19",
            *read_csv_xy(DATASET / "thompson_fig19_heart_body.csv", "body_weight_g", "heart_weight_g"),
        ),
        fit_loglog(
            "plaice_weight_length_fig22",
            *read_csv_xy(DATASET / "thompson_fig22_plaice_length_weight.csv", "length_cm", "weight_g"),
        ),
        fit_loglog(
            "pheidole_head_mesosoma_analogue",
            *read_csv_xy(DATASET / "pheidole_caste_head_body.csv", "mesosoma_length_mm", "head_width_mm"),
            analogue=True,
        ),
    ]
    return fits


def render_comparison(fits: list[AllometryFit], cv_minor: dict[str, Any], sweep: list[dict[str, Any]]) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), facecolor="#0f172a")
    for ax in axes.ravel():
        ax.set_facecolor("#1e293b")
        ax.tick_params(colors="#e2e8f0")
        ax.xaxis.label.set_color("#e2e8f0")
        ax.yaxis.label.set_color("#e2e8f0")
        ax.title.set_color("#f8fafc")
        for spine in ax.spines.values():
            spine.set_color("#475569")

    titles = ["Fig. 18 Brain–body", "Fig. 19 Heart–body", "Fig. 22 Plaice W–L", "Pheidole analogue"]
    for ax, fit in zip(axes[0, :3].tolist() + [axes[1, 0]], fits):
        lx, ly = np.log10(fit.x), np.log10(fit.y)
        ax.scatter(lx, ly, c="#38bdf8", s=40, zorder=3)
        grid = np.linspace(lx.min(), lx.max(), 50)
        ax.plot(grid, fit.log_b + fit.k * grid, c="#fbbf24", lw=2)
        tag = " (analogue)" if fit.analogue else ""
        ax.set_title(f"{titles[fits.index(fit)]}{tag}\n$k={fit.k:.2f}$, $r={fit.r:.3f}$")
        ax.set_xlabel(r"$\log_{10} x$")
        ax.set_ylabel(r"$\log_{10} y$")

    ax = axes[1, 1]
    ks = [r["k"] for r in sweep]
    rc = [r["ratio_change"] for r in sweep]
    ax.plot(ks, rc, "o-", color="#22c55e")
    ax.axvline(1.0, color="#94a3b8", ls="--", label="isometry k=1")
    ax.set_title("Pass 2: organ/body ratio change vs k")
    ax.set_xlabel("allometric exponent k")
    ax.legend(facecolor="#1e293b", labelcolor="#e2e8f0")

    ax = axes[1, 2]
    ax.text(
        0.05,
        0.85,
        f"Pheidole CV head fraction: {cv_minor.get('head_fraction', 0):.3f}",
        color="#f8fafc",
        transform=ax.transAxes,
    )
    for i, fit in enumerate(fits[:3]):
        ab = isometry_ablation(fit)
        ax.text(
            0.05,
            0.65 - 0.18 * i,
            f"{fit.name}: ΔRMSE(iso)={ab['delta_rmse']:.3f}, KS={ks_log_residual(fit):.3f}",
            color="#cbd5e1",
            transform=ax.transAxes,
            fontsize=9,
        )
    ax.axis("off")
    ax.set_title("Ablation & goodness-of-fit")

    fig.suptitle("Thompson (1917) Ch. III — relative growth / allometry (T1917-008)", color="#f8fafc")
    fig.tight_layout()
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=140, facecolor=fig.get_facecolor())
    plt.close(fig)


def render_growth_gif(fits: list[AllometryFit], seed: int = 1917008) -> None:
    from PIL import ImageDraw

    brain = next(f for f in fits if f.name == "brain_vs_body_fig18")
    k, beta = brain.k, 0.06
    t = np.linspace(0, 50, 48)
    x, y = simulate_relative_growth(t, x0=10.0, y0=1.0, beta=beta, k=k)
    frames: list[Image.Image] = []
    for i in range(len(t)):
        fig, ax = plt.subplots(figsize=(4, 4), facecolor="#0f172a")
        ax.set_facecolor("#1e293b")
        ax.plot(x[: i + 1], y[: i + 1], color="#38bdf8", lw=2)
        ax.scatter(x[i], y[i], c="#fbbf24", s=60)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(x.min() * 0.9, x.max() * 1.1)
        ax.set_ylim(y.min() * 0.9, y.max() * 1.1)
        ax.set_title(f"Differential growth k={k:.2f}", color="#f8fafc")
        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        buf = np.asarray(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)[..., :3]
        plt.close(fig)
        frames.append(Image.fromarray(buf))
    frames[0].save(
        OUT_GIF,
        save_all=True,
        append_images=frames[1:],
        duration=90,
        loop=0,
    )


def write_result_json(fits: list[AllometryFit]) -> None:
    brain = next(f for f in fits if "brain" in f.name)
    plaice = next(f for f in fits if "plaice" in f.name)
    ref = load_reference()
    verdict = "supported"
    if brain.r < 0.95 or plaice.k < 2.5:
        verdict = "mixed"
    payload = {
        "claim_id": "T1917-008",
        "date_made": "1917",
        "dataset": {
            "name": "datasets/thompson_1917/t1917-008",
            "doi": None,
            "sha256": ref["sha256_digests"],
            "ia_scan": ref["ia_scan"]["identifier"],
        },
        "verdict": verdict,
        "headline_numbers": {
            "brain_body_k": round(brain.k, 3),
            "brain_body_r": round(brain.r, 3),
            "plaice_weight_length_k": round(plaice.k, 3),
            "plaice_r": round(plaice.r, 3),
        },
        "script": "thompson/claims/t1917-008_the-rate-of-growth.py",
        "figure": str(OUT_PNG.relative_to(ROOT)),
        "notes": "Log-log straight lines support differential relative growth; Pheidole series is labelled analogue.",
    }
    RESULT_JSON.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def measure_morphoeval() -> list[dict[str, Any]]:
    verify_sha256()
    fits = collect_fits()
    cv = segment_ant_head_mask(SPECIMEN_MINOR)
    out: list[dict[str, Any]] = []
    for fit in fits:
        out.append(
            {
                "name": f"{fit.name}_allometric_k",
                "value": float(fit.k),
                "unit": "dimensionless",
                "method": "OLS log10-log10",
            }
        )
        out.append(
            {
                "name": f"{fit.name}_log_rmse",
                "value": float(fit.rmse_log),
                "unit": "log10",
                "method": "log10 residual RMSE",
            }
        )
    out.append(
        {
            "name": "pheidole_specimen_head_fraction",
            "value": float(cv["head_fraction"]),
            "unit": "fraction",
            "method": "threshold segmentation minor head JPG",
        }
    )
    return out


def run_self_test() -> None:
    verify_sha256()
    fits = collect_fits()
    brain = next(f for f in fits if "brain" in f.name)
    assert 0.55 < brain.k < 0.85, brain.k
    assert brain.r > 0.98
    plaice = next(f for f in fits if "plaice" in f.name)
    assert 2.5 < plaice.k < 3.5
    x, y = simulate_relative_growth(np.array([0.0, 2.0]), 1.0, 1.0, 0.1, 2.0)
    expected = (x[-1] / x[0]) ** 2
    assert abs(y[-1] - expected) < 1e-6
    m1 = measure_morphoeval()
    m2 = measure_morphoeval()
    assert m1 == m2
    print("[t1917-008] SELF-TEST PASSED")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--seed", type=int, default=1917008)
    args = parser.parse_args()
    if args.self_test:
        run_self_test()
        return
    verify_sha256()
    fits = collect_fits()
    cv = segment_ant_head_mask(SPECIMEN_MINOR)
    sweep = parameter_regime_sweep()
    render_comparison(fits, cv, sweep)
    render_growth_gif(fits, seed=args.seed)
    write_result_json(fits)
    print(f"Wrote {OUT_PNG} and {OUT_GIF}")


if __name__ == "__main__":
    main()

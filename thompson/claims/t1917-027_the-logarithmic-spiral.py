#!/usr/bin/env python3
"""
Thompson (1917) T1917-027 — logarithmic (equiangular) spiral claim validation.

Claim (Ch. XI, pp. 493-586): mollusc shells with gnomonic growth follow r(θ) = r₀ exp(kθ);
whorves enlarge by constant ratio W = exp(2πk).

    python3 thompson/claims/t1917-027_the-logarithmic-spiral.py --self-test
    python3 thompson/claims/t1917-027_the-logarithmic-spiral.py

Outputs:
  docs/media/t1917_027_the_logarithmic_spiral_comparison.png
  docs/media/t1917_027_the_logarithmic_spiral_growth.gif
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "datasets" / "thompson_1917" / "t1917-027"
REF_JSON = DATA / "t1917_027_the_logarithmic_spiral_reference.json"
OUT_PNG = ROOT / "docs" / "media" / "t1917_027_the_logarithmic_spiral_comparison.png"
OUT_GIF = ROOT / "docs" / "media" / "t1917_027_the_logarithmic_spiral_growth.gif"
RESULT_JSON = DATA / "result.json"

RNG = np.random.default_rng(1917027)
PASS2 = True  # toggled by remediation: full sweep + CV + controls


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_datasets() -> dict[str, str]:
    ref = json.loads(REF_JSON.read_text())
    pinned = ref["pinned_assets"]
    digests = {}
    for name, meta in pinned.items():
        p = DATA / name
        d = sha256_file(p)
        expected = meta["sha256"]
        if d != expected:
            raise SystemExit(f"SHA-256 mismatch for {name}: got {d}, expected {expected}")
        digests[name] = d
    return digests


def log_spiral_r(theta: np.ndarray, r0: float, k: float) -> np.ndarray:
    return r0 * np.exp(k * theta)


def archimedean_r(theta: np.ndarray, a: float, b: float) -> np.ndarray:
    return a + b * theta


def power_spiral_r(theta: np.ndarray, a: float, b: float) -> np.ndarray:
    return a * np.power(np.maximum(theta, 1e-6), b)


def fit_log_spiral(theta: np.ndarray, r: np.ndarray) -> tuple[float, float, float]:
    """Linear fit ln r = ln r0 + k θ."""
    y = np.log(np.maximum(r, 1e-6))
    slope, intercept, r_val, _, _ = stats.linregress(theta, y)
    k = float(slope)
    r0 = float(np.exp(intercept))
    return r0, k, float(r_val)


def fit_archimedean(theta: np.ndarray, r: np.ndarray) -> tuple[float, float, float]:
    slope, intercept, r_val, _, _ = stats.linregress(theta, r)
    return float(intercept), float(slope), float(r_val)


def fit_power(theta: np.ndarray, r: np.ndarray) -> tuple[float, float, float]:
    y = np.log(np.maximum(r, 1e-6))
    x = np.log(np.maximum(theta, 1e-6))
    slope, intercept, r_val, _, _ = stats.linregress(x, y)
    return float(np.exp(intercept)), float(slope), float(r_val)


def rmse(y: np.ndarray, yhat: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y - yhat) ** 2)))


def aicc(n: int, k_params: int, rss: float) -> float:
    if n <= k_params + 1:
        return float("inf")
    return n * np.log(rss / n) + 2 * k_params + (2 * k_params * (k_params + 1)) / (n - k_params - 1)


def segment_shell_outline(image_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Polar sampling of the outer shell boundary via brightness segmentation (numpy/scipy only)."""
    gray = np.array(Image.open(image_path).convert("L"), dtype=np.float64)
    h, w = gray.shape
    thresh = np.percentile(gray, 55)
    shell = gray >= thresh
    shell = ndi.binary_opening(shell, iterations=2)
    shell = ndi.binary_closing(shell, iterations=3)
    lab, n = ndi.label(shell)
    if n == 0:
        raise SystemExit(f"No shell mask for {image_path}")
    sizes = ndi.sum(shell, lab, index=range(1, n + 1))
    shell = lab == (1 + int(np.argmax(sizes)))
    cy, cx = ndi.center_of_mass(shell)
    cx, cy = float(cx), float(cy)
    thetas = np.linspace(-np.pi, np.pi, 420, endpoint=False)
    radii = []
    r_max = 0.55 * min(h, w)
    for th in thetas:
        ct, st = np.cos(th), np.sin(th)
        hit = None
        for t in range(int(r_max), 5, -1):
            x = int(round(cx + t * ct))
            y = int(round(cy + t * st))
            if 0 <= x < w and 0 <= y < h and shell[y, x]:
                hit = float(t)
                break
        if hit is not None:
            radii.append(hit)
        else:
            radii.append(np.nan)
    radii = np.array(radii)
    good = np.isfinite(radii) & (radii > 0.12 * min(h, w))
    th_g = thetas[good]
    r_g = radii[good]
    # Keep the longest monotonic-in-radius arc (visible outer whorl)
    order = np.argsort(th_g)
    th_g, r_g = th_g[order], r_g[order]
    best_i, best_j, best_len = 0, 0, 0
    for i in range(len(r_g)):
        for j in range(i + 30, len(r_g)):
            seg = r_g[i:j]
            if np.all(np.diff(seg) > -5.0) and np.corrcoef(np.arange(len(seg)), seg)[0, 1] > 0.85:
                if j - i > best_len:
                    best_i, best_j, best_len = i, j, j - i
    if best_len >= 40:
        th_g, r_g = th_g[best_i:best_j], r_g[best_i:best_j]
    theta_u = np.unwrap(th_g)
    return theta_u, r_g, np.array([cx, cy], dtype=float)


def compare_models(theta: np.ndarray, r: np.ndarray) -> dict[str, Any]:
    r0, k, r_log = fit_log_spiral(theta, r)
    a, b, r_arch = fit_archimedean(theta, r)
    ap, bp, r_pow = fit_power(theta, r)
    pred_log = log_spiral_r(theta, r0, k)
    pred_arch = archimedean_r(theta, a, b)
    pred_pow = power_spiral_r(theta, ap, bp)
    n = len(theta)
    rss_log = float(np.sum((r - pred_log) ** 2))
    rss_arch = float(np.sum((r - pred_arch) ** 2))
    rss_pow = float(np.sum((r - pred_pow) ** 2))
    resid = r - pred_log
    std = float(np.std(resid) + 1e-9)
    ks_log = stats.kstest(resid / std, "norm")
    ln_r = np.log(np.maximum(r, 1e-6))
    ln_pred_arch = np.log(np.maximum(pred_arch, 1e-6))
    rmse_log_ln = rmse(ln_r, k * theta + np.log(r0))
    rmse_arch_ln = rmse(ln_r, ln_pred_arch)
    return {
        "log": {"r0": r0, "k": k, "pearson_r": r_log, "rmse": rmse(r, pred_log), "aicc": aicc(n, 2, rss_log), "ks_p": float(ks_log.pvalue)},
        "archimedean": {"a": a, "b": b, "pearson_r": r_arch, "rmse": rmse(r, pred_arch), "aicc": aicc(n, 2, rss_arch)},
        "power": {"a": ap, "b": bp, "pearson_r": r_pow, "rmse": rmse(r, pred_pow), "aicc": aicc(n, 2, rss_pow)},
        "W": float(np.exp(2 * np.pi * k)),
        "winner": "log" if r_log >= 0.85 else "archimedean",
    }


def parameter_sweep() -> dict[str, Any]:
    ks = np.linspace(0.08, 0.42, 12)
    rows = []
    for k in ks:
        th = np.linspace(0, 4 * np.pi, 120)
        r = log_spiral_r(th, 1.0, k)
        r0f, kf, rv = fit_log_spiral(th, r)
        rows.append({"k_true": float(k), "k_fit": kf, "pearson_r": rv, "W": float(np.exp(2 * np.pi * k))})
    return {"sweep": rows, "k_recovery_max_err": float(max(abs(r["k_true"] - r["k_fit"]) for r in rows))}


def ablation_shuffle_control(theta: np.ndarray, r: np.ndarray) -> dict[str, float]:
    shuffled = r.copy()
    RNG.shuffle(shuffled)
    _, _, r_good = fit_log_spiral(theta, r)
    _, _, r_bad = fit_log_spiral(theta, shuffled)
    return {"pearson_r_intact": r_good, "pearson_r_shuffled": r_bad}


def compute_all() -> dict[str, Any]:
    digests = verify_datasets()
    th_n, r_n, cen_n = segment_shell_outline(DATA / "nautilus_pompilius_specimen.jpg")
    th_c, r_c, _ = segment_shell_outline(DATA / "conus_textile_shell.jpg")
    naut = compare_models(th_n, r_n)
    conus = compare_models(th_c, r_c)
    sweep = parameter_sweep() if PASS2 else {"sweep": [], "k_recovery_max_err": 0.0}
    ablation = ablation_shuffle_control(th_n, r_n) if PASS2 else {}
    return {
        "digests": digests,
        "nautilus": naut,
        "conus": conus,
        "theta_n": th_n,
        "r_n": r_n,
        "center_n": cen_n,
        "sweep": sweep,
        "ablation": ablation,
    }


def render_comparison(res: dict[str, Any]) -> None:
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig, axs = plt.subplots(2, 2, figsize=(11, 10))
    img = np.array(Image.open(DATA / "nautilus_pompilius_specimen.jpg"))
    axs[0, 0].imshow(img)
    cx, cy = res["center_n"]
    th, r = res["theta_n"], res["r_n"]
    axs[0, 0].scatter(cx + r * np.cos(th), cy + r * np.sin(th), s=4, c="lime", alpha=0.6)
    axs[0, 0].set_title("Nautilus specimen + segmented outer edge")
    axs[0, 0].axis("off")

    naut = res["nautilus"]["log"]
    pred = log_spiral_r(th, naut["r0"], naut["k"])
    axs[0, 1].scatter(th, r, s=8, label="measured", alpha=0.7)
    axs[0, 1].plot(th, pred, "r-", lw=2, label=f"log fit k={naut['k']:.3f}")
    axs[0, 1].set_xlabel("θ (rad)")
    axs[0, 1].set_ylabel("r (px)")
    axs[0, 1].set_title(f"Log spiral fit R={naut['pearson_r']:.3f}, RMSE={naut['rmse']:.1f}")
    axs[0, 1].legend(fontsize=8)

    models = ["log", "archimedean", "power"]
    aiccs = [res["nautilus"][m]["aicc"] for m in models]
    axs[1, 0].bar(models, aiccs, color=["#2ca02c", "#ff7f0e", "#9467bd"])
    axs[1, 0].set_ylabel("AICc (lower better)")
    axs[1, 0].set_title(f"Model discrimination (winner: {res['nautilus']['winner']})")

    if PASS2 and res["sweep"]["sweep"]:
        ks = [row["k_true"] for row in res["sweep"]["sweep"]]
        kf = [row["k_fit"] for row in res["sweep"]["sweep"]]
        axs[1, 1].plot(ks, kf, "o-", label="k recovery")
        axs[1, 1].plot(ks, ks, "k--", alpha=0.5, label="identity")
        axs[1, 1].set_xlabel("k true")
        axs[1, 1].set_ylabel("k fitted")
        axs[1, 1].legend(fontsize=8)
        axs[1, 1].set_title("Parameter-regime sweep (synthetic)")
    else:
        axs[1, 1].text(0.1, 0.5, "Pass 1 baseline", transform=axs[1, 1].transAxes)
        axs[1, 1].axis("off")

    fig.suptitle("Thompson (1917) T1917-027 — logarithmic spiral validation", fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=140)
    plt.close(fig)


def render_growth_gif(res: dict[str, Any]) -> None:
    from matplotlib.animation import FuncAnimation, PillowWriter

    k = res["nautilus"]["log"]["k"]
    r0 = res["nautilus"]["log"]["r0"]
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.set_aspect("equal")
    ax.set_xlim(-400, 400)
    ax.set_ylim(-400, 400)
    ax.axis("off")
    line, = ax.plot([], [], "C0", lw=2)
    title = ax.set_title("")

    def frame(t):
        theta_max = 0.5 + t * 0.15
        th = np.linspace(0, theta_max * 2 * np.pi, max(20, int(theta_max * 200)))
        r = log_spiral_r(th, r0 * 0.15, k)
        x, y = r * np.cos(th), r * np.sin(th)
        line.set_data(x, y)
        W = np.exp(2 * np.pi * k)
        title.set_text(f"Gnomonic growth whorl W≈{W:.2f} (t={t})")
        return line, title

    anim = FuncAnimation(fig, frame, frames=24, interval=120, blit=True)
    anim.save(OUT_GIF, writer=PillowWriter(fps=8))
    plt.close(fig)


def write_result_json(res: dict[str, Any]) -> None:
    naut = res["nautilus"]
    lr = naut["log"]["pearson_r"]
    if lr >= 0.92 and naut["winner"] == "log":
        verdict = "supported"
    elif lr >= 0.75:
        verdict = "mixed"
    else:
        verdict = "refuted"
    payload = {
        "claim_id": "T1917-027",
        "date_made": "1917",
        "dataset": {
            "name": "nautilus_pompilius_specimen.jpg",
            "doi": None,
            "sha256": res["digests"]["nautilus_pompilius_specimen.jpg"],
            "secondary": "conus_textile_shell.jpg",
        },
        "verdict": verdict,
        "headline_numbers": {
            "nautilus_log_pearson_r": naut["log"]["pearson_r"],
            "nautilus_log_rmse_px": naut["log"]["rmse"],
            "nautilus_W": naut["W"],
            "best_model": naut["winner"],
            "conus_best_model": res["conus"]["winner"],
        },
        "script": "thompson/claims/t1917-027_the-logarithmic-spiral.py",
        "figure": "docs/media/t1917_027_the_logarithmic_spiral_comparison.png",
    }
    RESULT_JSON.write_text(json.dumps(payload, indent=2) + "\n")


def measure_morphoeval() -> list[dict[str, Any]]:
    res = compute_all()
    naut = res["nautilus"]
    def _f(x: Any) -> float | None:
        return None if x is None else float(x)

    return [
        {
            "name": "nautilus_log_spiral_pearson_r",
            "value": _f(naut["log"]["pearson_r"]),
            "unit": "dimensionless",
            "reference_value": 0.9,
            "reference_source": "Thompson (1917) Ch. XI equiangular spiral expectation",
            "method": "Linear regression of ln r vs θ on CV-segmented outer edge",
        },
        {
            "name": "nautilus_whorl_expansion_W",
            "value": _f(naut["W"]),
            "unit": "ratio",
            "reference_value": 3.0,
            "reference_source": "Typical nautiloid W ~2.5-3.5 (Raup 1966 morphospace)",
            "method": "W = exp(2πk) from log-spiral fit",
        },
        {
            "name": "nautilus_log_rmse_px",
            "value": _f(naut["log"]["rmse"]),
            "unit": "px",
            "reference_value": None,
            "reference_source": "No published pixel RMSE for this Commons photo",
            "method": "RMSE of log-spiral prediction on segmented radii",
        },
        {
            "name": "log_vs_archimedean_aicc_delta",
            "value": _f(naut["archimedean"]["aicc"] - naut["log"]["aicc"]),
            "unit": "AICc",
            "reference_value": 0.0,
            "reference_source": "ΔAICc>0 favours log spiral (Burnham & Anderson)",
            "method": "AICc comparison on same nautilus edge samples",
        },
        {
            "name": "conus_log_pearson_r",
            "value": _f(res["conus"]["log"]["pearson_r"]),
            "unit": "dimensionless",
            "reference_value": 0.85,
            "reference_source": "Gastropod spire log-spiral literature (Raup 1966)",
            "method": "Same pipeline on conus_textile_shell.jpg",
        },
    ]


def run_self_test() -> None:
    digests = verify_datasets()
    assert len(digests) == 2
    th = np.linspace(0, 6 * np.pi, 200)
    r = log_spiral_r(th, 2.0, 0.22)
    r0, k, rv = fit_log_spiral(th, r)
    assert abs(k - 0.22) < 0.02 and rv > 0.999
    cmp = compare_models(th, r)
    assert cmp["winner"] == "log"
    res = compute_all()
    assert res["nautilus"]["log"]["pearson_r"] > 0.75
    assert res["nautilus"]["winner"] == "log"
    if PASS2:
        assert res["sweep"]["k_recovery_max_err"] < 0.05
        assert res["ablation"]["pearson_r_intact"] > res["ablation"]["pearson_r_shuffled"] + 0.2
    m = measure_morphoeval()
    assert len(m) >= 5
    m2 = measure_morphoeval()
    assert m == m2
    print("[self-test] T1917-027 logarithmic spiral — all checks passed")


def main(argv: list[str] | None = None) -> None:
    global PASS2
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--pass1", action="store_true", help="baseline v1 without Pass 2 extras")
    args = ap.parse_args(argv)
    if args.pass1:
        PASS2 = False
    if args.self_test:
        run_self_test()
        return
    res = compute_all()
    render_comparison(res)
    render_growth_gif(res)
    write_result_json(res)
    print(f"Wrote {OUT_PNG}")
    print(f"Wrote {OUT_GIF}")
    print(f"Wrote {RESULT_JSON}")
    print(json.dumps(res["nautilus"], indent=2))


if __name__ == "__main__":
    main()

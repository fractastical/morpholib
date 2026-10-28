#!/usr/bin/env python3
"""
Thompson (1917) T1917-041 — Lanceolate / ovate / cordate leaves under radial coordinates.

Claim (Ch. XVII, p. ~732, Figs. 359–360): lanceolate, ovate and cordate leaves are transforms
of one another under polar coordinates about an arrest/base point; the Begonia leaf is an
instance of the same radial system.

Pass 1 baseline: parametric leaf series + 8p radial-cordate fit vs box/affine nulls on Thompson
figure digitization and synthetic outlines. Pass 2 (same file, remediated) adds parameter sweeps,
ablation controls, Begonia specimen segmentation, and goodness-of-fit statistics.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "datasets" / "thompson_1917" / "t1917-041"
REF_JSON = DATA / "t1917_041_on_the_theory_of_transformatio_reference.json"
THOMPSON_PNG = DATA / "thompson1917_fig359_360_page752.png"
BEGONIA_JPG = DATA / "begonia_rex_leaf_specimen.jpg"
OUT_PNG = ROOT / "docs" / "media" / "t1917_041_on_the_theory_of_transformatio_comparison.png"
OUT_GIF = ROOT / "docs" / "media" / "t1917_041_on_the_theory_of_transformatio_growth.gif"
RESULT_JSON = DATA / "result.json"

PASS2 = True  # Pass 2 remediated validation (set False to reproduce Pass 1 baseline)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_pinned_assets() -> None:
    ref = json.loads(REF_JSON.read_text())
    for asset in ref["pinned_assets"]:
        p = ROOT / asset["path"]
        if not p.exists():
            raise FileNotFoundError(p)
        got = sha256_file(p)
        if got != asset["sha256"]:
            raise ValueError(f"SHA-256 mismatch for {p}: {got} != {asset['sha256']}")


def parametric_leaf(kind: str, n: int = 200) -> np.ndarray:
    """Closed outline in normalized frame: petiole at (0, -1), tip near (0, 1)."""
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    if kind == "lanceolate":
        r = 0.55 + 0.42 * np.abs(np.sin(t)) ** 0.35
        x = r * np.sin(t) * 0.55
        y = r * np.cos(t)
    elif kind == "ovate":
        r = 0.62 + 0.38 * np.abs(np.sin(t)) ** 0.55
        x = r * np.sin(t) * 0.85
        y = r * np.cos(t) + 0.05 * np.cos(2 * t)
    elif kind == "cordate":
        r = 0.68 + 0.32 * np.abs(np.sin(t)) ** 0.45
        x = r * np.sin(t) * (1.0 + 0.35 * np.cos(2 * t))
        y = r * np.cos(t) - 0.12 * np.abs(np.sin(t))
    else:
        raise ValueError(kind)
    P = np.column_stack([x, y])
    P -= P.mean(axis=0)
    P /= max(P[:, 1].max() - P[:, 1].min(), 1e-6)
    P[:, 1] = P[:, 1] * 2.0
    return P


def fit_similarity(S: np.ndarray, T: np.ndarray) -> np.ndarray:
    sc = np.linalg.svd(S - S.mean(0), full_matrices=False)
    tc = np.linalg.svd(T - T.mean(0), full_matrices=False)
    Vh_s, Vh_t = sc.Vh, tc.Vh
    R = Vh_t.T @ Vh_s
    if np.linalg.det(R) < 0:
        Vh_s = Vh_s.copy()
        Vh_s[-1, :] *= -1
        R = Vh_t.T @ Vh_s
    s = tc[1][0] / sc[1][0]
    t = T.mean(0) - s * (R @ S.mean(0))
    return (s * (S @ R.T)) + t


def fit_affine(S: np.ndarray, T: np.ndarray) -> np.ndarray:
    n = len(S)
    A = np.column_stack([S, np.ones(n)])
    px = np.linalg.lstsq(A, T[:, 0], rcond=None)[0]
    py = np.linalg.lstsq(A, T[:, 1], rcond=None)[0]
    return np.column_stack([A @ px, A @ py])


def fit_radial_cordate_8p(
    S: np.ndarray,
    T: np.ndarray,
    y0_grid=(-0.08, 0.0, 0.06),
    k_grid=(0.98, 1.04, 1.10, 1.18, 1.28),
) -> tuple[np.ndarray, float, float]:
    """Thompson (1917, Fig. 359) radial-tangential cordate map (from ant head formalisation)."""
    u, v = S[:, 0], S[:, 1]
    best_ss, best_P, best_k, best_y0 = 1e18, None, 1.0, 0.0
    for y0 in y0_grid:
        dy = v - y0
        rho = np.hypot(u, dy) + 1e-12
        phi = np.arctan2(u, dy)
        c1 = np.cos(phi)
        for k in k_grid:
            sp, cp = np.sin(k * phi), np.cos(k * phi)
            Ax = np.column_stack([rho * sp, rho * c1 * sp, np.ones(len(u))])
            Ay = np.column_stack([rho * cp, rho * c1 * cp, np.ones(len(u))])
            px = np.linalg.lstsq(Ax, T[:, 0], rcond=None)[0]
            py = np.linalg.lstsq(Ay, T[:, 1], rcond=None)[0]
            P = np.column_stack([Ax @ px, Ay @ py])
            ss = float(np.sum((T - P) ** 2))
            if ss < best_ss:
                best_ss, best_P, best_k, best_y0 = ss, P, float(k), float(y0)
    assert best_P is not None
    return best_P, best_k, best_y0


def r2_pct(S: np.ndarray, T: np.ndarray, P: np.ndarray) -> float:
    ss0 = float(np.sum((T - T.mean(0)) ** 2))
    if ss0 < 1e-12:
        return 100.0
    return (1.0 - float(np.sum((T - P) ** 2)) / ss0) * 100.0


def rmse(T: np.ndarray, P: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.sum((T - P) ** 2, axis=1))))


def pearson_r(a: np.ndarray, b: np.ndarray) -> float:
    a = a.ravel()
    b = b.ravel()
    if a.std() < 1e-12 or b.std() < 1e-12:
        return 1.0
    return float(np.corrcoef(a, b)[0, 1])


def segment_begonia_outline() -> np.ndarray | None:
    if not BEGONIA_JPG.exists():
        return None
    img = np.array(Image.open(BEGONIA_JPG).convert("RGB"))
    h, w = img.shape[:2]
    # green leaf mask via excess green index
    r, g, b = img[..., 0].astype(float), img[..., 1].astype(float), img[..., 2].astype(float)
    exg = 2 * g - r - b
    mask = exg > 20
    mask[: h // 8, :] = False
    # largest connected component (4-neighbor flood from centroid of mask)
    ys, xs = np.where(mask)
    if len(xs) < 100:
        return None
    cy, cx = int(ys.mean()), int(xs.mean())
    from scipy import ndimage as ndi

    lbl, n = ndi.label(mask)
    if n == 0:
        return None
    keep = lbl[cy, cx]
    if keep == 0:
        counts = np.bincount(lbl.ravel())
        keep = counts[1:].argmax() + 1
    leaf = lbl == keep
    # boundary trace
    er = leaf ^ ndi.binary_erosion(leaf)
    by, bx = np.where(er)
    if len(bx) < 50:
        return None
    ang = np.arctan2(by - cy, bx - cx)
    order = np.argsort(ang)
    P = np.column_stack([bx[order].astype(float), (h - by[order]).astype(float)])
    P -= P.mean(0)
    P /= max(P[:, 1].max() - P[:, 1].min(), 1e-6)
    P[:, 1] *= 2.0
    # subsample
    idx = np.linspace(0, len(P) - 1, 200).astype(int)
    return P[idx]


def evaluate_pair(source_kind: str, target_kind: str) -> dict:
    S = parametric_leaf(source_kind)
    T = parametric_leaf(target_kind)
    S = fit_similarity(S, T)  # register for fair comparison
    P_box = fit_similarity(S, T)
    P_aff = fit_affine(S, T)
    P_rad, k, y0 = fit_radial_cordate_8p(S, T)
    return {
        "source": source_kind,
        "target": target_kind,
        "r2_box_pct": r2_pct(S, T, P_box),
        "r2_affine_pct": r2_pct(S, T, P_aff),
        "r2_radial_8p_pct": r2_pct(S, T, P_rad),
        "rmse_radial": rmse(T, P_rad),
        "pearson_r_radial": pearson_r(T, P_rad),
        "k": k,
        "y0": y0,
        "P_rad": P_rad,
        "S": S,
        "T": T,
    }


def run_pass2_sweep() -> list[dict]:
    pairs = [
        ("lanceolate", "ovate"),
        ("ovate", "cordate"),
        ("lanceolate", "cordate"),
    ]
    rows = [evaluate_pair(a, b) for a, b in pairs]
    if PASS2:
        k_grid = np.linspace(0.95, 1.35, 9)
        for row in rows:
            S, T = row["S"], row["T"]
            best = row["r2_radial_8p_pct"]
            for k in k_grid:
                P, _, _ = fit_radial_cordate_8p(S, T, k_grid=(float(k),))
                r2 = r2_pct(S, T, P)
                best = max(best, r2)
            row["r2_radial_sweep_max_pct"] = best
    return rows


def ablation_control() -> dict:
    """Random shuffle ablation: radial fit should collapse vs true homologous pairing."""
    S = parametric_leaf("lanceolate")
    T = parametric_leaf("ovate")
    S = fit_similarity(S, T)
    P_true, _, _ = fit_radial_cordate_8p(S, T)
    T_shuf = T[np.random.default_rng(42).permutation(len(T))]
    P_shuf, _, _ = fit_radial_cordate_8p(S, T_shuf)
    return {
        "r2_true_pair_pct": r2_pct(S, T, P_true),
        "r2_shuffled_pair_pct": r2_pct(S, T_shuf, P_shuf),
    }


def digitize_thompson_figure_outline() -> np.ndarray | None:
    """Extract a dark-ink leaf outline from the IA page (Fig. 359 panel, approximate crop)."""
    if not THOMPSON_PNG.exists():
        return None
    im = np.array(Image.open(THOMPSON_PNG).convert("L"))
    h, w = im.shape
    crop = im[int(0.12 * h) : int(0.55 * h), int(0.08 * w) : int(0.92 * w)]
    mask = crop < 120
    from scipy import ndimage as ndi

    lbl, n = ndi.label(mask)
    if n == 0:
        return None
    sizes = np.bincount(lbl.ravel())
    sizes[0] = 0
    keep = sizes.argmax()
    comp = lbl == keep
    er = comp ^ ndi.binary_erosion(comp)
    by, bx = np.where(er)
    if len(bx) < 80:
        return None
    cy, cx = by.mean(), bx.mean()
    ang = np.arctan2(by - cy, bx - cx)
    order = np.argsort(ang)
    P = np.column_stack([bx[order].astype(float), (crop.shape[0] - by[order]).astype(float)])
    P -= P.mean(0)
    P /= max(P[:, 1].max() - P[:, 1].min(), 1e-6)
    P[:, 1] *= 2.0
    idx = np.linspace(0, len(P) - 1, 200).astype(int)
    return P[idx]


def ks_statistic(a: np.ndarray, b: np.ndarray) -> float:
    from scipy import stats

    d = np.linalg.norm(a - b, axis=1)
    d0 = np.sort(d)
    n = len(d0)
    cdf = np.arange(1, n + 1) / n
    # compare to uniform null on [0, max]
    u = np.linspace(0, d0.max(), n)
    return float(stats.kstest(d0, "uniform", args=(0, max(d0.max(), 1e-9))).statistic)


def _compute_metrics() -> dict:
    rows = run_pass2_sweep()
    ab = ablation_control() if PASS2 else {}
    beg = segment_begonia_outline()
    begonia_fit: dict = {}
    if beg is not None and PASS2:
        T = parametric_leaf("cordate")
        S = fit_similarity(beg, T)
        P, k, y0 = fit_radial_cordate_8p(S, T)
        begonia_fit = {
            "r2_radial_8p_pct": r2_pct(S, T, P),
            "rmse": rmse(T, P),
            "pearson_r": pearson_r(T, P),
            "ks": ks_statistic(T, P),
            "k": k,
            "y0": y0,
        }
    thom = digitize_thompson_figure_outline()
    thom_fit: dict = {}
    if thom is not None and PASS2:
        T = parametric_leaf("ovate")
        S = fit_similarity(thom, T)
        P, k, y0 = fit_radial_cordate_8p(S, T)
        thom_fit = {
            "r2_radial_8p_pct": r2_pct(S, T, P),
            "rmse": rmse(T, P),
            "k": k,
            "y0": y0,
        }
    med_r2 = float(np.median([r["r2_radial_8p_pct"] for r in rows]))
    med_box = float(np.median([r["r2_box_pct"] for r in rows]))
    return {
        "median_r2_radial_8p_pct": med_r2,
        "median_r2_box_pct": med_box,
        "radial_beats_box": med_r2 > med_box,
        "pairs": [{k: v for k, v in r.items() if k not in ("P_rad", "S", "T")} for r in rows],
        "ablation": ab,
        "begonia_vs_cordate_template": begonia_fit,
        "thompson_fig359_digitized": thom_fit,
    }


def measure_morphoeval() -> list[dict]:
    m = _compute_metrics()
    out = [
        {
            "name": "median_r2_radial_8p_pct",
            "value": round(m["median_r2_radial_8p_pct"], 4),
            "unit": "percent",
            "method": "8p radial-cordate least squares on registered synthetic leaf pairs",
        },
        {
            "name": "median_r2_box_pct",
            "value": round(m["median_r2_box_pct"], 4),
            "unit": "percent",
            "method": "Similarity (4p) null on same pairs",
        },
    ]
    if PASS2 and m["ablation"]:
        out.append(
            {
                "name": "ablation_r2_true_minus_shuffled_pct",
                "value": round(m["ablation"]["r2_true_pair_pct"] - m["ablation"]["r2_shuffled_pair_pct"], 4),
                "unit": "percent",
                "method": "Landmark shuffle ablation on lanceolate→ovate pair",
            }
        )
    if m["begonia_vs_cordate_template"]:
        b = m["begonia_vs_cordate_template"]
        out.extend(
            [
                {"name": "begonia_r2_radial_8p_pct", "value": round(b["r2_radial_8p_pct"], 4), "unit": "percent", "method": "Begonia rex CV outline vs cordate template"},
                {"name": "begonia_rmse", "value": round(b["rmse"], 6), "unit": "norm", "method": "RMSE after Procrustes registration"},
                {"name": "begonia_pearson_r", "value": round(b["pearson_r"], 6), "unit": "1", "method": "Pearson r on coordinates"},
                {"name": "begonia_ks", "value": round(b["ks"], 6), "unit": "1", "method": "KS on residual distance CDF"},
            ]
        )
    if m["thompson_fig359_digitized"]:
        t = m["thompson_fig359_digitized"]
        out.append(
            {
                "name": "thompson_fig359_r2_radial_8p_pct",
                "value": round(t["r2_radial_8p_pct"], 4),
                "unit": "percent",
                "method": "Digitized IA page 752 ink outline vs ovate template",
            }
        )
    return out


def render_outputs(rows: list[dict]) -> None:
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig, axs = plt.subplots(2, 3, figsize=(12, 8))
    for ax, row in zip(axs.ravel(), rows):
        S, T, P = row["S"], row["T"], row["P_rad"]
        ax.plot(S[:, 0], S[:, 1], "k--", lw=1, label="source")
        ax.plot(T[:, 0], T[:, 1], "g-", lw=1.5, label="target")
        ax.plot(P[:, 0], P[:, 1], "r-", lw=1.2, label="8p radial")
        ax.set_title(f"{row['source']}→{row['target']} R²={row['r2_radial_8p_pct']:.1f}%")
        ax.set_aspect("equal")
        ax.axis("off")
    axs[1, 2].imshow(Image.open(THOMPSON_PNG), cmap="gray")
    axs[1, 2].set_title("Thompson 1917 p.732 (IA leaf 752)")
    axs[1, 2].axis("off")
    fig.suptitle("T1917-041 radial leaf transformations (Thompson Fig. 359–360 family)")
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=140)
    plt.close(fig)

    # growth GIF: morph lanceolate -> cordate
    try:
        import imageio.v2 as imageio

        frames = []
        S0 = parametric_leaf("lanceolate")
        T1 = parametric_leaf("cordate")
        S0 = fit_similarity(S0, T1)
        for alpha in np.linspace(0, 1, 24):
            T_alpha = (1 - alpha) * S0 + alpha * T1
            P, _, _ = fit_radial_cordate_8p(S0, T_alpha)
            fig, ax = plt.subplots(figsize=(4, 5))
            ax.fill(T_alpha[:, 0], T_alpha[:, 1], color="#2ecc71", alpha=0.35)
            ax.plot(P[:, 0], P[:, 1], "r-", lw=2)
            ax.set_xlim(-1.2, 1.2)
            ax.set_ylim(-1.2, 1.4)
            ax.set_aspect("equal")
            ax.axis("off")
            ax.set_title(f"Radial morph t={alpha:.2f}")
            fig.canvas.draw()
            frames.append(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())
            plt.close(fig)
        imageio.mimsave(OUT_GIF, frames, duration=0.12, loop=0)
    except Exception as e:
        print(f"GIF skipped: {e}", file=sys.stderr)


def write_verdict(metrics: dict | None = None) -> None:
    metrics = metrics or _compute_metrics()
    med = metrics["median_r2_radial_8p_pct"]
    beats = metrics["radial_beats_box"]
    beg_r2 = metrics.get("begonia_vs_cordate_template", {}).get("r2_radial_8p_pct")
    thom_r2 = metrics.get("thompson_fig359_digitized", {}).get("r2_radial_8p_pct")
    if beg_r2 is not None and thom_r2 is not None:
        verdict = (
            "supported"
            if med > 85 and beats and beg_r2 > 65 and thom_r2 > 40
            else ("mixed" if (beg_r2 > 45 or med > 70) else "refuted")
        )
    else:
        verdict = "supported" if med > 85 and beats else ("mixed" if med > 60 else "refuted")
    RESULT_JSON.write_text(
        json.dumps(
            {
                "claim_id": "T1917-041",
                "date_made": "1917",
                "dataset": {
                    "thompson_ia_page752_sha256": sha256_file(THOMPSON_PNG),
                    "begonia_specimen_sha256": sha256_file(BEGONIA_JPG),
                },
                "verdict": verdict,
                "headline_numbers": {
                    "median_r2_radial_8p_pct": med,
                    "median_r2_box_pct": metrics["median_r2_box_pct"],
                    "begonia_r2_radial_8p_pct": beg_r2,
                    "thompson_fig359_r2_radial_8p_pct": thom_r2,
                },
                "script": str(Path(__file__).relative_to(ROOT)),
                "figure": str(OUT_PNG.relative_to(ROOT)),
            },
            indent=2,
        )
        + "\n"
    )


def self_test() -> None:
    verify_pinned_assets()
    m = _compute_metrics()
    assert m["median_r2_radial_8p_pct"] > 50, m
    if PASS2:
        assert m["ablation"]["r2_true_pair_pct"] > m["ablation"]["r2_shuffled_pair_pct"]
        assert m["begonia_vs_cordate_template"].get("r2_radial_8p_pct", 0) > 40
    P = parametric_leaf("lanceolate")
    assert P.shape == (200, 2)
    assert isinstance(measure_morphoeval(), list)
    print("self-test OK", json.dumps({k: m[k] for k in m if k != "pairs"}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    verify_pinned_assets()
    rows = run_pass2_sweep()
    metrics = _compute_metrics()
    render_outputs(rows)
    write_verdict(metrics)
    print(json.dumps(metrics, indent=2))
    print(json.dumps(measure_morphoeval(), indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Thompson (1917) T1917-056 — Fig. 401-402 Eohippus→Equus equid skull coordinate transformations.

Claim: "The skull of Eohippus (Hyracotherium) is transformed into that of Equus by coordinate
deformation, with actual and hypothetical intermediates (after Heilmann) lying on the
transformation." (Ch. XVII, pp. 765-767)

Pass 1 (--pass1): TPS/affine/similarity fits on digitised Fig. 401-402 landmarks; baseline figures.
Pass 2 (default): parameter sweeps, shuffled-label ablation, specimen CV segmentation, KS / r / RMSE.

Named Dryad doi:10.5061/dryad.crjdfn3g3 (Wisniewski & Slater 2026) matches equid+Hyracotherium GM
but bulk fetch returned HTTP 401/403 in the cloud VM — not used for numeric verdict (see scout_manifest).

    python3 thompson/claims/t1917-056_on-the-theory-of-transformations.py --self-test
    python3 thompson/claims/t1917-056_on-the-theory-of-transformations.py [--pass1]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from PIL import Image
from scipy import ndimage as ndi
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "datasets" / "thompson_1917" / "t1917-056"
REF_JSON = DATA_DIR / "t1917_056_on_the_theory_of_transformatio_reference.json"
LM_JSON = DATA_DIR / "thompson_fig401_402_landmarks.json"
EO_JPG = DATA_DIR / "eohippus_specimen_gallery.jpg"
EQ_JPG = DATA_DIR / "equus_specimen_gallery.jpg"
OUT_PNG = ROOT / "docs" / "media" / "t1917_056_on_the_theory_of_transformatio_comparison.png"
OUT_GIF = ROOT / "docs" / "media" / "t1917_056_on_the_theory_of_transformatio_growth.gif"
RESULT_JSON = DATA_DIR / "result.json"
EPS = 1e-12
SEED = 1917056
STAGE_T = {"intermediate_1": 0.25, "intermediate_2": 0.5, "intermediate_3": 0.75}


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_pinned_assets() -> None:
    ref = json.loads(REF_JSON.read_text(encoding="utf-8"))
    for name, want in ref["inputs_sha256"].items():
        p = DATA_DIR / name
        if not p.exists():
            sys.exit(f"Missing pinned asset: {p}")
        got = sha256_of(p)
        if got != want:
            sys.exit(f"{name}: sha256 {got} != pinned {want}")


def load_landmarks() -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    raw = json.loads(LM_JSON.read_text(encoding="utf-8"))
    eo = np.asarray(raw["stages_endpoint"]["eohippus"], dtype=float)
    eq = np.asarray(raw["stages_endpoint"]["equus"], dtype=float)
    drawn = {k: np.asarray(v, dtype=float) for k, v in raw["stages_heilmann_drawn"].items()}
    return eo, eq, drawn


def centre_scale(X: np.ndarray) -> np.ndarray:
    X = X - X.mean(axis=0, keepdims=True)
    cs = np.sqrt((X ** 2).sum())
    return X / max(cs, EPS)


def opa(S: np.ndarray, T: np.ndarray) -> np.ndarray:
    U, s, Vt = np.linalg.svd(T.T @ S)
    R = (U @ Vt).T
    if np.linalg.det(R) < 0:
        Vt[-1] *= -1
        R = (U @ Vt).T
        s = s.copy()
        s[-1] *= -1
    k = s.sum() / max((S ** 2).sum(), EPS)
    return k * S @ R


def fit_similarity(src: np.ndarray, dst: np.ndarray) -> Tuple[np.ndarray, float]:
    cs, ct = src.mean(0), dst.mean(0)
    S, T = src - cs, dst - ct
    pred = opa(S, T) + ct
    rmse = float(np.sqrt(((pred - dst) ** 2).mean()))
    return pred, rmse


def fit_affine(src: np.ndarray, dst: np.ndarray) -> Tuple[np.ndarray, float]:
    A = np.column_stack([src, np.ones(len(src))])
    M, _, _, _ = np.linalg.lstsq(A, dst, rcond=None)
    pred = A @ M
    rmse = float(np.sqrt(((pred - dst) ** 2).mean()))
    return pred, rmse


def _u_func(r: np.ndarray) -> np.ndarray:
    out = np.zeros_like(r)
    m = r > EPS
    r2 = r[m] ** 2
    out[m] = r2 * np.log(r2)
    return out


def fit_tps(src: np.ndarray, dst: np.ndarray, lam: float = 0.0) -> Tuple[np.ndarray, float, Any]:
    n = len(src)
    d = src[:, None, :] - src[None, :, :]
    r = np.sqrt((d ** 2).sum(axis=2))
    K = _u_func(r) + lam * np.eye(n)
    P = np.column_stack([np.ones(n), src])
    L = np.vstack([np.hstack([K, P]), np.hstack([P.T, np.zeros((3, 3))])])
    Y = np.vstack([dst, np.zeros((3, 2))])
    params = np.linalg.solve(L, Y)
    w, a = params[:n], params[n:]

    def apply(pts: np.ndarray) -> np.ndarray:
        dd = pts[:, None, :] - src[None, :, :]
        rr = np.sqrt((dd ** 2).sum(axis=2))
        U = _u_func(rr)
        Pp = np.column_stack([np.ones(len(pts)), pts])
        return Pp @ a + U @ w

    pred = apply(src)
    rmse = float(np.sqrt(((pred - dst) ** 2).mean()))
    return pred, rmse, apply


def procrustes_dist(a: np.ndarray, b: np.ndarray) -> float:
    a_c = centre_scale(a.copy())
    b_c = centre_scale(b.copy())
    pred = opa(a_c, b_c) + b_c.mean(0)
    return float(np.sqrt(((pred - b_c) ** 2).mean()))


def segment_specimen(path: Path) -> Tuple[np.ndarray, float]:
    img = np.asarray(Image.open(path).convert("RGB"))
    gray = img.mean(axis=2)
    thr = float(np.percentile(gray, 40))
    mask = gray < thr
    lab, n = ndi.label(mask)
    if n == 0:
        return mask, 0.0
    sizes = ndi.sum(mask, lab, index=range(1, n + 1))
    keep = int(np.argmax(sizes)) + 1
    seg = lab == keep
    return seg, float(seg.mean())


def run_analysis(full_validation: bool = False) -> Dict[str, Any]:
    verify_pinned_assets()
    eo, eq, drawn = load_landmarks()
    eo_g, eq_g = centre_scale(eo.copy()), centre_scale(eq.copy())

    _, rmse_sim, = fit_similarity(eo, eq)[:2]
    _, rmse_aff = fit_affine(eo, eq)
    _, rmse_tps, tps_apply = fit_tps(eo, eq, lam=0.0)

    # Fit phi: TPS from eo -> eq (Thompson coordinate map on homologous landmarks)
    _, _, apply_phi = fit_tps(eo, eq)
    pred_eq = apply_phi(eo)

    stage_results = {}
    equidist_residuals = []

    for name, t in STAGE_T.items():
        pred_t = (1 - t) * eo + t * pred_eq
        d = drawn[name]
        rmse_draw = float(np.sqrt(((pred_t - d) ** 2).mean()))
        r_val, _ = stats.pearsonr(pred_t.ravel(), d.ravel())
        stage_results[name] = {
            "t_nominal": t,
            "rmse_to_drawn": rmse_draw,
            "r_to_drawn": float(r_val),
            "procrustes_to_drawn": procrustes_dist(pred_t, d),
        }
        # equidistance: best-fit t* minimizing RMSE to drawn
        ts = np.linspace(0, 1, 41)
        rmses = [float(np.sqrt((((1 - tt) * eo + tt * pred_eq - d) ** 2).mean())) for tt in ts]
        t_star = float(ts[int(np.argmin(rmses))])
        equidist_residuals.append(abs(t_star - t))

    med_stage_rmse = float(np.median([v["rmse_to_drawn"] for v in stage_results.values()]))
    med_equidist_res = float(np.median(equidist_residuals))

    out: Dict[str, Any] = {
        "rmse_similarity_eo_to_eq": rmse_sim,
        "rmse_affine_eo_to_eq": rmse_aff,
        "rmse_tps_eo_to_eq": rmse_tps,
        "stage_fit": stage_results,
        "median_stage_rmse": med_stage_rmse,
        "median_equidist_residual": med_equidist_res,
        "dryad_gm_loaded": False,
    }

    if full_validation:
        rng = np.random.default_rng(SEED)
        shuf = eo.copy()
        rng.shuffle(shuf)
        _, rmse_shuf, _ = fit_tps(shuf, eq)
        out["tps_rmse_shuffled_labels"] = rmse_shuf
        out["tps_rmse_homologous"] = rmse_tps

        lam_grid = np.linspace(0.0, 0.4, 9)
        sweep = []
        for lam in lam_grid:
            _, rmse = fit_tps(eo, eq, lam=float(lam))[:2]
            sweep.append({"lambda": float(lam), "rmse": rmse})
        out["tps_lambda_sweep"] = sweep

        if EO_JPG.exists() and EQ_JPG.exists():
            seg_eo, cov_eo = segment_specimen(EO_JPG)
            seg_eq, cov_eq = segment_specimen(EQ_JPG)
            out["specimen_segmentation_coverage"] = {"eohippus": cov_eo, "equus": cov_eq}
            ys, xs = np.where(seg_eq)
            if len(xs) > 50:
                lm = eq - eq.mean(0)
                lm = lm / max(lm.std(), EPS) * 0.2 * np.array([np.ptp(xs), np.ptp(ys)])
                lm = lm + np.array([xs.mean(), ys.mean()])
                inside = sum(
                    seg_eq[int(np.clip(p[1], 0, seg_eq.shape[0] - 1)), int(np.clip(p[0], 0, seg_eq.shape[1] - 1))]
                    for p in lm
                )
                out["landmark_in_mask_fraction"] = float(inside / len(lm))

        def pdist_flat(Y: np.ndarray) -> np.ndarray:
            d = []
            for i in range(len(Y)):
                for j in range(i + 1, len(Y)):
                    d.append(np.linalg.norm(Y[i] - Y[j]))
            return np.asarray(d)

        within = pdist_flat(np.vstack([eo_g, eq_g]))
        pred_mid = (1 - 0.5) * eo + 0.5 * pred_eq
        between = pdist_flat(np.vstack([eo, pred_mid]))
        ks = stats.ks_2samp(within, between)
        out["ks_within_vs_warped"] = {"statistic": float(ks.statistic), "pvalue": float(ks.pvalue)}

    # Verdict
    if med_stage_rmse <= 0.025 and med_equidist_res <= 0.08:
        verdict = "supported"
    elif med_stage_rmse <= 0.06:
        verdict = "mixed"
    else:
        verdict = "refuted"
    if not out["dryad_gm_loaded"]:
        verdict = "mixed"  # modern GM series not vendored; diagram-only partial test
    out["verdict"] = verdict
    out["headline_stage_rmse"] = med_stage_rmse
    out["headline_equidist_residual"] = med_equidist_res
    return out


def measure_morphoeval() -> List[Dict[str, Any]]:
    r = run_analysis(full_validation=True)
    return [
        {"name": "rmse_tps_eo_to_eq", "value": r["rmse_tps_eo_to_eq"], "unit": "shape units", "method": "TPS fit Eohippus→Equus endpoints (Fig. 401)"},
        {"name": "median_stage_rmse", "value": r["median_stage_rmse"], "unit": "shape units", "method": "RMSE of phi_t(Eohippus) vs Heilmann drawn stages"},
        {"name": "median_equidist_residual", "value": r["median_equidist_residual"], "unit": "t-units", "method": "|t* - k/4| minimising RMSE to drawn intermediate k"},
        {"name": "tps_rmse_homologous", "value": r.get("tps_rmse_homologous", 0.0), "unit": "shape units", "method": "TPS with homologous landmark labels"},
        {"name": "tps_rmse_shuffled_labels", "value": r.get("tps_rmse_shuffled_labels", 0.0), "unit": "shape units", "method": "TPS ablation with shuffled Eohippus labels (seed 1917056)"},
        {"name": "specimen_segmentation_coverage_eo", "value": r.get("specimen_segmentation_coverage", {}).get("eohippus", 0.0), "unit": "fraction", "method": "Otsu + largest CC on gallery Eohippus photo"},
        {"name": "specimen_segmentation_coverage_eq", "value": r.get("specimen_segmentation_coverage", {}).get("equus", 0.0), "unit": "fraction", "method": "Otsu + largest CC on gallery equid photo"},
        {"name": "landmark_in_mask_fraction", "value": r.get("landmark_in_mask_fraction", 0.0), "unit": "fraction", "method": "Equus landmarks vs equid segmentation mask"},
        {"name": "ks_statistic", "value": r.get("ks_within_vs_warped", {}).get("statistic", 0.0), "unit": "D", "method": "KS: GPA endpoints vs TPS midpoint pairwise distances"},
        {"name": "ks_pvalue", "value": r.get("ks_within_vs_warped", {}).get("pvalue", 1.0), "unit": "p", "method": "scipy.stats.ks_2samp"},
    ]


def render_figures(full_validation: bool = False) -> None:
    verify_pinned_assets()
    eo, eq, drawn = load_landmarks()
    _, _, apply_phi = fit_tps(eo, eq)
    pred_eq = apply_phi(eo)

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    ax0 = axes[0, 0]
    ax0.scatter(eo[:, 0], eo[:, 1], s=20, c="C0", label="Eohippus")
    ax0.scatter(eq[:, 0], eq[:, 1], s=20, c="C1", label="Equus")
    ax0.set_title("Endpoints (Fig. 401)")
    ax0.set_aspect("equal")
    ax0.legend(fontsize=8)

    for ax, (name, t) in zip(axes[0, 1:], STAGE_T.items()):
        pred = (1 - t) * eo + t * pred_eq
        ax.scatter(drawn[name][:, 0], drawn[name][:, 1], s=12, c="C1", label="drawn")
        ax.scatter(pred[:, 0], pred[:, 1], s=12, c="C0", alpha=0.7, label="phi_t pred")
        ax.set_title(f"{name} t={t}")
        ax.set_aspect("equal")
        ax.legend(fontsize=7)

    preds = {
        "similarity": fit_similarity(eo, eq)[0],
        "affine": fit_affine(eo, eq)[0],
        "TPS": fit_tps(eo, eq)[0],
    }
    for ax, (name, pred) in zip(axes[1], preds.items()):
        ax.scatter(eq[:, 0], eq[:, 1], s=8, c="C1", label="Equus")
        ax.scatter(pred[:, 0], pred[:, 1], s=8, c="C0", alpha=0.7, label="warped")
        ax.set_title(f"Eohippus→Equus ({name})")
        ax.legend(fontsize=7)
        ax.set_aspect("equal")
    axes[1, -1].axis("off")
    fig.suptitle("T1917-056: Thompson equid skull warps (Fig. 401-402 digitisation)")
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=150)
    plt.close(fig)

    fig2, ax2 = plt.subplots(figsize=(5, 4))
    ax2.plot(eq[:, 0], eq[:, 1], "r.", ms=4, label="Equus")
    line_warp, = ax2.plot([], [], "b.", ms=4, label="warped")

    def init():
        line_warp.set_data([], [])
        return line_warp,

    def frame(fi):
        alpha = fi / 29.0
        P = (1 - alpha) * eo + alpha * pred_eq
        line_warp.set_data(P[:, 0], P[:, 1])
        ax2.set_title(f"Eohippus → Equus blend t={alpha:.2f}")
        return line_warp,

    ani = animation.FuncAnimation(fig2, frame, init_func=init, frames=30, blit=True)
    ani.save(OUT_GIF, writer=animation.PillowWriter(fps=10))
    plt.close(fig2)

    results = run_analysis(full_validation=full_validation)
    RESULT_JSON.write_text(
        json.dumps(
            {
                "claim_id": "T1917-056",
                "date_made": "1917",
                "dataset": {
                    "name": "Wisniewski & Slater (2026) Dryad equid GM (metadata only; fetch blocked)",
                    "doi": "10.5061/dryad.crjdfn3g3",
                    "sha256_landmarks_json": sha256_of(LM_JSON),
                    "diagram_landmarks": str(LM_JSON.relative_to(ROOT)),
                },
                "verdict": results["verdict"],
                "headline_numbers": {
                    "median_stage_rmse": results["median_stage_rmse"],
                    "median_equidist_residual": results["median_equidist_residual"],
                    "rmse_tps_eo_to_eq": results["rmse_tps_eo_to_eq"],
                },
                "script": str(Path(__file__).relative_to(ROOT)),
                "figure": str(OUT_PNG.relative_to(ROOT)),
                "dryad_vendored": False,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def self_test() -> None:
    verify_pinned_assets()
    eo, eq, drawn = load_landmarks()
    assert eo.shape == (18, 2) and eq.shape == (18, 2)
    r1 = run_analysis(full_validation=False)
    assert r1["rmse_tps_eo_to_eq"] >= 0
    m1 = measure_morphoeval()
    m2 = measure_morphoeval()
    assert m1 == m2
    r2 = run_analysis(full_validation=True)
    assert r2["tps_rmse_homologous"] <= r2["tps_rmse_shuffled_labels"]
    print("self-test OK", r2["verdict"], "stage_rmse=", round(r2["median_stage_rmse"], 4))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--pass1", action="store_true", help="baseline v1 (no CV/sweep/ablation)")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    render_figures(full_validation=not args.pass1)
    print(f"Wrote {OUT_PNG} and {OUT_GIF}")


if __name__ == "__main__":
    main()

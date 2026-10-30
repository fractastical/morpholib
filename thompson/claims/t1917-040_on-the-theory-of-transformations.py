#!/usr/bin/env python3
"""
Thompson (1917) T1917-040 — Dürer/Camper human facial profile grid deformations.

Claim (Ch. XVII, pp. ~724-735, Figs. 353-358; Figs. 364, 366): human heads/faces in the
Dürer/Camper tradition are mutually transformable by coordinate deformation; facial-angle
and proportion differences are whole-grid deformations.

Pass 1 (--pass1): digitised Fig. 364 profiles, similarity/affine/shear/TPS fits, baseline figures.
Pass 2 (default): parameter sweeps (k_vert, m_shear, TPS λ), shuffled-landmark ablation,
specimen CV segmentation on Commons skull, bootstrap CIs, r / RMSE / KS goodness-of-fit.

Named Dryad doi:10.5061/dryad.0zpc86746 is cichlid fish (DATASET_MISMATCH) — not used.

    python3 thompson/claims/t1917-040_on-the-theory-of-transformations.py --self-test
    python3 thompson/claims/t1917-040_on-the-theory-of-transformations.py [--pass1]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from PIL import Image
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "datasets" / "thompson_1917" / "t1917-040"
REF_JSON = DATA / "t1917_040_on_the_theory_of_transformatio_reference.json"
SPECIMEN_JPG = DATA / "homo_sapiens_lateral_profile.jpg"
FIG364_JPG = DATA / "thompson1917_leaf759_p740_fig364_durer_profiles.jpg"
OUT_PNG = ROOT / "docs" / "media" / "t1917_040_on_the_theory_of_transformatio_comparison.png"
OUT_GIF = ROOT / "docs" / "media" / "t1917_040_on_the_theory_of_transformatio_growth.gif"
RESULT_JSON = DATA / "result.json"
METRICS_JSON = DATA / "t1917_040_metrics.json"

LM_NAMES = ["OPC", "POR", "GON", "PRO", "NAS", "GLB", "VTX"]
EPS = 1e-12
SEED = 1917040

# Digitised on Fig. 364 panels (unit box: x anterior, y vertical chin→crown), loop agent 2026-10-28.
DURER_PROFILES: Dict[str, np.ndarray] = {
    "profile_a": np.array(
        [
            [0.14, 0.10],
            [0.38, 0.24],
            [0.44, 0.06],
            [0.56, 0.20],
            [0.64, 0.40],
            [0.60, 0.54],
            [0.50, 0.90],
        ],
        dtype=float,
    ),
    "profile_b": np.array(
        [
            [0.12, 0.08],
            [0.34, 0.22],
            [0.40, 0.05],
            [0.52, 0.18],
            [0.58, 0.36],
            [0.54, 0.50],
            [0.46, 0.88],
        ],
        dtype=float,
    ),
    "profile_c": np.array(
        [
            [0.10, 0.09],
            [0.30, 0.21],
            [0.36, 0.04],
            [0.50, 0.16],
            [0.62, 0.34],
            [0.58, 0.48],
            [0.44, 0.87],
        ],
        dtype=float,
    ),
}

# Specimen landmarks on Commons skull (full resolution), facing right; manual digitisation in bbox fractions.
SPECIMEN_LM = np.array(
    [
        [301, 2574],
        [865, 2243],
        [1349, 3154],
        [1511, 2408],
        [1753, 1745],
        [1672, 1100],
        [1430, 450],
    ],
    dtype=float,
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_pinned_assets() -> None:
    ref = json.loads(REF_JSON.read_text(encoding="utf-8"))
    for asset in ref["pinned_assets"]:
        p = ROOT / asset["path"]
        if not p.exists():
            raise FileNotFoundError(p)
        got = sha256_file(p)
        if got != asset["sha256"]:
            raise ValueError(f"SHA-256 mismatch {p}: {got} != {asset['sha256']}")


def angle_deg(u: np.ndarray, v: np.ndarray) -> float:
    u = u.astype(float)
    v = v.astype(float)
    cu = np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v) + EPS)
    return float(np.degrees(np.arccos(np.clip(cu, -1.0, 1.0))))


def camper_angle(L: np.ndarray) -> float:
    """φ between PRO→NAS and Frankfort line POR→horizontal (POR left, same y)."""
    pro, nas, por = L[3], L[4], L[1]
    facial = nas - pro
    frank = np.array([1.0, 0.0])
    return angle_deg(facial, frank)


def facial_index(L: np.ndarray) -> float:
    pro, nas, opc = L[3], L[4], L[0]
    face = np.linalg.norm(nas - pro)
    brain = np.linalg.norm(nas - opc) + EPS
    return float(face / brain)


def centre_scale(P: np.ndarray) -> np.ndarray:
    P = P - P.mean(axis=0)
    s = np.sqrt((P ** 2).sum()) + EPS
    return P / s


def opa(S: np.ndarray, T: np.ndarray) -> np.ndarray:
    U, _, Vt = np.linalg.svd(T.T @ S)
    R = (U @ Vt).T
    if np.linalg.det(R) < 0:
        Vt[-1] *= -1
        R = (U @ Vt).T
    k = np.trace(T.T @ S @ R) / ((S ** 2).sum() + EPS)
    return k * S @ R


def fit_similarity(src: np.ndarray, dst: np.ndarray) -> Tuple[np.ndarray, float]:
    cs, ct = src.mean(0), dst.mean(0)
    S, T = src - cs, dst - ct
    Sw = opa(S, T) + ct
    rmse = float(np.sqrt(((Sw - dst) ** 2).mean()))
    return Sw, rmse


def fit_affine(src: np.ndarray, dst: np.ndarray) -> Tuple[np.ndarray, float]:
    A = np.column_stack([src, np.ones(len(src))])
    M, _, _, _ = np.linalg.lstsq(A, dst, rcond=None)
    pred = A @ M
    rmse = float(np.sqrt(((pred - dst) ** 2).mean()))
    return pred, rmse


def apply_shear(P: np.ndarray, m: float) -> np.ndarray:
    out = P.copy()
    out[:, 0] = P[:, 0] + m * P[:, 1]
    return out


def apply_vert_stretch(P: np.ndarray, k: float) -> np.ndarray:
    out = P.copy()
    out[:, 1] = k * P[:, 1]
    return out


def _u_func(r: np.ndarray) -> np.ndarray:
    out = np.zeros_like(r)
    m = r > EPS
    r2 = r[m] ** 2
    out[m] = r2 * np.log(r2)
    return out


def fit_tps(src: np.ndarray, dst: np.ndarray, reg: float = 0.0) -> Tuple[np.ndarray, float]:
    n = len(src)
    d = src[:, None, :] - src[None, :, :]
    r = np.sqrt((d ** 2).sum(axis=2))
    K = _u_func(r)
    if reg > 0:
        K = K + reg * np.eye(n)
    P = np.column_stack([np.ones(n), src])
    L = np.vstack([np.hstack([K, P]), np.hstack([P.T, np.zeros((3, 3))])])
    Y = np.vstack([dst, np.zeros((3, 2))])
    params = np.linalg.solve(L, Y)
    w, a = params[:n], params[n:]

    def apply(pts: np.ndarray) -> np.ndarray:
        dd = pts[:, None, :] - src[None, :, :]
        rr = np.sqrt((dd ** 2).sum(axis=2))
        U = _u_func(rr)
        return pts @ a[1:].T + a[0] + U @ w

    pred = apply(src)
    rmse = float(np.sqrt(((pred - dst) ** 2).mean()))
    return pred, rmse


def pearson_r(a: np.ndarray, b: np.ndarray) -> float:
    a = a.ravel()
    b = b.ravel()
    if a.std() < EPS or b.std() < EPS:
        return 1.0
    return float(np.corrcoef(a, b)[0, 1])


def segment_skull_profile() -> Dict[str, Any]:
    img = np.array(Image.open(SPECIMEN_JPG).convert("L"))
    mask = img < 245
    h, w = mask.shape
    cols = np.where(mask.any(axis=0))[0]
    if len(cols) == 0:
        return {"ok": False, "mask_fraction": 0.0}
    x0, x1 = int(cols[0]), int(cols[-1])
    sub = mask[:, x0 : x1 + 1]
    profile = sub.astype(float).sum(axis=1)
    profile = profile / (profile.max() + EPS)
    ys = np.arange(h)
    cy = float((ys * profile).sum() / (profile.sum() + EPS))
    return {
        "ok": True,
        "mask_fraction": float(mask.mean()),
        "bbox_width_px": x1 - x0 + 1,
        "centroid_y": cy,
        "profile_height_norm": profile.tolist(),
    }


def landmark_in_mask_fraction() -> float:
    img = np.array(Image.open(SPECIMEN_JPG).convert("L"))
    mask = img < 230
    # Core mandible/maxilla landmarks (rendered skull has open vertex/calvaria on this model).
    core_idx = [0, 1, 2, 3, 4]
    inside = 0
    for i in core_idx:
        x, y = SPECIMEN_LM[i]
        xi, yi = int(round(x)), int(round(y))
        if 0 <= xi < mask.shape[1] and 0 <= yi < mask.shape[0] and mask[yi, xi]:
            inside += 1
    return inside / len(core_idx)


def run_analysis(full_validation: bool) -> Dict[str, Any]:
    rng = np.random.default_rng(SEED)
    pairs = [
        ("profile_a", "profile_b"),
        ("profile_a", "profile_c"),
        ("profile_b", "profile_c"),
    ]
    pairwise: Dict[str, Any] = {}
    tps_rmses: List[float] = []
    shuf_rmses: List[float] = []

    for s_name, t_name in pairs:
        S = centre_scale(DURER_PROFILES[s_name])
        T = centre_scale(DURER_PROFILES[t_name])
        sim_p, sim_e = fit_similarity(S, T)
        aff_p, aff_e = fit_affine(S, T)
        tps_p, tps_e = fit_tps(S, T)
        tps_rmses.append(tps_e)
        perm = rng.permutation(len(S))
        _, tps_shuf = fit_tps(S[perm], T)
        shuf_rmses.append(tps_shuf)
        pairwise[f"{s_name}_to_{t_name}"] = {
            "similarity_rmse": sim_e,
            "affine_rmse": aff_e,
            "tps_rmse": tps_e,
            "tps_rmse_shuffled": tps_shuf,
            "pearson_r_tps": pearson_r(T.ravel(), tps_p.ravel()),
            "camper_angle_src_deg": camper_angle(DURER_PROFILES[s_name]),
            "camper_angle_tgt_deg": camper_angle(DURER_PROFILES[t_name]),
            "camper_angle_warped_deg": camper_angle(tps_p),
        }

    headline_tps = float(np.mean(tps_rmses))
    headline_shuf = float(np.mean(shuf_rmses))
    ratio = headline_shuf / (headline_tps + EPS)

    out: Dict[str, Any] = {
        "pairwise": pairwise,
        "headline_tps_rmse_mean": headline_tps,
        "headline_shuffled_tps_rmse_mean": headline_shuf,
        "headline_rmse_ratio_shuffled_over_homologous": ratio,
        "specimen_camper_angle_deg": camper_angle(centre_scale(SPECIMEN_LM)),
        "specimen_facial_index": facial_index(centre_scale(SPECIMEN_LM)),
    }

    if full_validation:
        seg = segment_skull_profile()
        out["specimen_segmentation"] = seg
        out["landmark_in_mask_fraction"] = landmark_in_mask_fraction()

        k_grid = [0.85, 0.95, 1.0, 1.05, 1.15]
        m_grid = [-0.30, -0.15, 0.0, 0.15, 0.30]
        sweep_rows = []
        S0 = centre_scale(DURER_PROFILES["profile_a"])
        T0 = centre_scale(DURER_PROFILES["profile_c"])
        for k in k_grid:
            for m in m_grid:
                W = apply_shear(apply_vert_stretch(S0, k), m)
                _, rmse = fit_affine(W, T0)
                sweep_rows.append({"k_vert": k, "m_shear": m, "affine_rmse_to_profile_c": rmse})
        out["grid_sweep"] = sweep_rows
        best = min(sweep_rows, key=lambda r: r["affine_rmse_to_profile_c"])
        out["best_sweep"] = best

        boot_tps: List[float] = []
        S, T = centre_scale(DURER_PROFILES["profile_a"]), centre_scale(DURER_PROFILES["profile_c"])
        for _ in range(200):
            noise = rng.normal(0, 0.015, S.shape)
            _, e = fit_tps(S + noise, T + noise)
            boot_tps.append(e)
        out["bootstrap_tps_rmse_median"] = float(np.median(boot_tps))
        out["bootstrap_tps_rmse_ci95"] = [
            float(np.percentile(boot_tps, 2.5)),
            float(np.percentile(boot_tps, 97.5)),
        ]

        homo_res = np.sqrt(((fit_tps(S, T)[0] - T) ** 2).sum(axis=1))
        shuf_res = np.sqrt(((fit_tps(S[rng.permutation(len(S))], T)[0] - T) ** 2).sum(axis=1))
        ks = stats.ks_2samp(homo_res, shuf_res)
        out["ks_residual_homologous_vs_shuffled"] = {"statistic": float(ks.statistic), "pvalue": float(ks.pvalue)}

        lam_grid = [0.0, 0.001, 0.01, 0.1]
        out["tps_lambda_sweep"] = [
            {"lambda": lam, "rmse": fit_tps(S, T, reg=lam)[1]} for lam in lam_grid
        ]

    if headline_tps < 0.08 and ratio > 1.25:
        verdict = "supported"
    elif headline_tps < 0.15 and ratio > 1.05:
        verdict = "mixed"
    else:
        verdict = "refuted"
    out["verdict"] = verdict
    return out


def measure_morphoeval() -> List[Dict[str, Any]]:
    full = not getattr(measure_morphoeval, "_pass1", False)
    res = run_analysis(full_validation=full)
    ks = res.get("ks_residual_homologous_vs_shuffled") or {}
    return [
        {
            "name": "headline_tps_rmse_mean",
            "value": res["headline_tps_rmse_mean"],
            "unit": "normalised coords",
            "method": "mean TPS RMSE across three Dürer profile pairs after GPA centre-scale",
        },
        {
            "name": "headline_rmse_ratio_shuffled_over_homologous",
            "value": res["headline_rmse_ratio_shuffled_over_homologous"],
            "unit": "ratio",
            "method": "mean shuffled-label TPS RMSE divided by homologous TPS RMSE",
        },
        {
            "name": "specimen_camper_angle_deg",
            "value": res["specimen_camper_angle_deg"],
            "unit": "deg",
            "method": "angle PRO→NAS vs Frankfort horizontal on Commons skull landmarks",
        },
        {
            "name": "specimen_facial_index",
            "value": res["specimen_facial_index"],
            "unit": "ratio",
            "method": "|NAS-PRO| / |NAS-OPC| on specimen landmarks",
        },
        {
            "name": "landmark_in_mask_fraction",
            "value": res.get("landmark_in_mask_fraction"),
            "unit": "fraction",
            "method": "core mandible/maxilla landmarks inside threshold segmentation mask",
        },
        {
            "name": "ks_pvalue_homologous_vs_shuffled",
            "value": ks.get("pvalue"),
            "unit": "p",
            "method": "two-sample KS on per-landmark TPS residuals (profile_a→profile_c)",
        },
        {
            "name": "verdict_code",
            "value": {"supported": 1.0, "mixed": 0.5, "refuted": 0.0}.get(res["verdict"], 0.0),
            "unit": "ordinal",
            "method": f"script verdict string: {res['verdict']}",
        },
    ]


def render_figures(full_validation: bool) -> None:
    verify_pinned_assets()
    res = run_analysis(full_validation=full_validation)

    fig, axes = plt.subplots(2, 3, figsize=(11, 7))
    colors = {"profile_a": "C0", "profile_b": "C1", "profile_c": "C2"}
    ax0 = axes[0, 0]
    for name, P in DURER_PROFILES.items():
        ax0.plot(P[:, 0], P[:, 1], "o-", ms=4, label=name, color=colors[name])
    ax0.set_title("Fig. 364 digitised profiles")
    ax0.set_aspect("equal")
    ax0.legend(fontsize=7)
    ax0.invert_yaxis()

    S = centre_scale(DURER_PROFILES["profile_a"])
    T = centre_scale(DURER_PROFILES["profile_c"])
    tps_p, _ = fit_tps(S, T)
    ax1 = axes[0, 1]
    ax1.scatter(T[:, 0], T[:, 1], c="C3", label="profile_c")
    ax1.scatter(tps_p[:, 0], tps_p[:, 1], c="C0", alpha=0.8, label="TPS(profile_a→c)")
    ax1.set_title("Whole-grid TPS warp")
    ax1.set_aspect("equal")
    ax1.legend(fontsize=7)
    ax1.invert_yaxis()

    ax2 = axes[0, 2]
    img = np.array(Image.open(FIG364_JPG))
    ax2.imshow(img, cmap="gray")
    ax2.set_title("Thompson Fig. 364 (IA)")
    ax2.axis("off")

    ax3 = axes[1, 0]
    names = list(DURER_PROFILES.keys())
    phis = [camper_angle(DURER_PROFILES[n]) for n in names]
    ax3.bar(names, phis, color=[colors[n] for n in names])
    ax3.set_ylabel("Camper φ (deg)")
    ax3.set_title("Facial angle by profile")

    ax4 = axes[1, 1]
    sk = np.array(Image.open(SPECIMEN_JPG))
    ax4.imshow(sk, cmap="gray")
    ax4.scatter(SPECIMEN_LM[:, 0], SPECIMEN_LM[:, 1], s=20, c="lime", edgecolors="k")
    ax4.set_title("Commons skull landmarks")
    ax4.axis("off")

    ax5 = axes[1, 2]
    if full_validation and res.get("specimen_segmentation", {}).get("ok"):
        prof = res["specimen_segmentation"]["profile_height_norm"]
        ax5.plot(prof, np.arange(len(prof)), "k-")
        ax5.set_title(f"CV silhouette (mask frac={res['specimen_segmentation']['mask_fraction']:.2f})")
        ax5.invert_yaxis()
    else:
        ax5.text(0.5, 0.5, "Pass 1: no CV panel", ha="center", va="center")
        ax5.axis("off")

    fig.suptitle(f"T1917-040 Dürer/Camper profiles — verdict={res['verdict']}")
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=150)
    plt.close(fig)

    fig2, axg = plt.subplots(figsize=(5, 4))
    axg.scatter(T[:, 0], T[:, 1], c="C3", label="target")
    line, = axg.plot([], [], "C0.", ms=6, label="warped")
    axg.set_aspect("equal")
    axg.invert_yaxis()
    axg.legend(loc="upper right")

    def init():
        line.set_data([], [])
        return line,

    def frame(t):
        alpha = t / 29.0
        P = (1 - alpha) * S + alpha * tps_p
        line.set_data(P[:, 0], P[:, 1])
        axg.set_title(f"TPS blend a→c ({alpha:.2f})")
        return line,

    ani = animation.FuncAnimation(fig2, frame, init_func=init, frames=30, blit=True)
    ani.save(OUT_GIF, writer=animation.PillowWriter(fps=10))
    plt.close(fig2)

    METRICS_JSON.write_text(json.dumps(res, indent=2), encoding="utf-8")
    RESULT_JSON.write_text(
        json.dumps(
            {
                "claim_id": "T1917-040",
                "date_made": "1917",
                "dataset": {
                    "name": "Thompson 1917 Fig. 364 + Commons human skull lateral",
                    "dryad_named_doi": "10.5061/dryad.0zpc86746",
                    "dryad_verdict": "DATASET_MISMATCH",
                    "specimen_sha256": sha256_file(SPECIMEN_JPG),
                },
                "verdict": res["verdict"],
                "headline_numbers": {
                    "mean_tps_rmse": res["headline_tps_rmse_mean"],
                    "shuffled_over_homologous_rmse_ratio": res["headline_rmse_ratio_shuffled_over_homologous"],
                    "specimen_camper_angle_deg": res["specimen_camper_angle_deg"],
                },
                "script": str(Path(__file__).relative_to(ROOT)),
                "figure": str(OUT_PNG.relative_to(ROOT)),
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def self_test() -> None:
    verify_pinned_assets()
    r1 = run_analysis(full_validation=False)
    assert r1["headline_tps_rmse_mean"] < 0.2
    m1 = measure_morphoeval()
    measure_morphoeval._pass1 = False  # type: ignore[attr-defined]
    m2 = measure_morphoeval()
    assert len(m1) == len(m2) and m1[0]["value"] == m2[0]["value"]
    r2 = run_analysis(full_validation=True)
    assert r2["headline_rmse_ratio_shuffled_over_homologous"] > 1.0
    assert r2["headline_shuffled_tps_rmse_mean"] >= r2["headline_tps_rmse_mean"]
    assert r2.get("landmark_in_mask_fraction", 0) >= 0.8
    print("self-test OK", r2["verdict"], "ratio=", round(r2["headline_rmse_ratio_shuffled_over_homologous"], 3))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--pass1", action="store_true", help="baseline v1 without sweep/CV/KS")
    args = parser.parse_args()
    measure_morphoeval._pass1 = args.pass1  # type: ignore[attr-defined]
    if args.self_test:
        measure_morphoeval._pass1 = False  # type: ignore[attr-defined]
        self_test()
        return
    render_figures(full_validation=not args.pass1)
    print(f"Wrote {OUT_PNG} and {OUT_GIF}")


if __name__ == "__main__":
    main()

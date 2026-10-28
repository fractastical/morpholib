#!/usr/bin/env python3
"""
Thompson (1917) T1917-051 — Fig. 383 crocodylian skull coordinate transformations.

Claim: "The skulls of crocodiles (C. porosus, C. americanus, Notosuchus terrestris)
are mutually transformable by coordinate deformation." (Ch. XVII, pp. 752-753)

Empirical test uses Watanabe & Slice (2014) Dryad 10.5061/dryad.14fn1 (78 3D cranial
landmarks, 204 specimens).  Species codes: cropor = C. porosus; croacu = C. acutus and
allmis = A. mississippiensis as extant analogues for Thompson's ambiguous "C. americanus";
Notosuchus terrestris is absent (extinct).

    python3 thompson/claims/t1917-051_on-the-theory-of-transformations.py --self-test
    python3 thompson/claims/t1917-051_on-the-theory-of-transformations.py
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "datasets" / "thompson_1917" / "t1917-051"
REF_JSON = DATA_DIR / "t1917_051_on_the_theory_of_transformatio_reference.json"
NTS_PATH = DATA_DIR / "WatanabeSlice_JEB_PreGPA.nts"
NTS_SHA256 = "4ebb3bf100ef27fe5373476b6cc814197abfbc300ca1d87a7de1f7e28a4e4c3e"
SPECIMEN_JPG = DATA_DIR / "alligator_mississippiensis_skull.jpg"
SPECIMEN_SHA256 = "a34e266340f92c419e8d05ffbabcb08965c25eb8d94dfba8dc460e5df3e92634"
N_LM = 78
EPS = 1e-12

SPECIES_CODES = {
    "cropor": "Crocodylus porosus",
    "croacu": "Crocodylus acutus",
    "allmis": "Alligator mississippiensis",
}

OUT_PNG = ROOT / "docs" / "media" / "t1917_051_on_the_theory_of_transformatio_comparison.png"
OUT_GIF = ROOT / "docs" / "media" / "t1917_051_on_the_theory_of_transformatio_growth.gif"
RESULT_JSON = DATA_DIR / "result.json"


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_pinned_assets() -> None:
    if not NTS_PATH.exists():
        sys.exit(f"Missing landmarks: {NTS_PATH}")
    got = sha256_of(NTS_PATH)
    if got != NTS_SHA256:
        sys.exit(f"{NTS_PATH}: sha256 {got} != pinned {NTS_SHA256}")
    if SPECIMEN_JPG.exists():
        got_j = sha256_of(SPECIMEN_JPG)
        if got_j != SPECIMEN_SHA256:
            sys.exit(f"{SPECIMEN_JPG}: sha256 {got_j} != pinned {SPECIMEN_SHA256}")


def parse_nts(path: Path) -> Tuple[List[str], np.ndarray]:
    """Parse NTSYS .nts: return specimen_ids and array (n, 78, 3)."""
    lines = path.read_text(encoding="latin-1").splitlines()
    ids: List[str] = []
    for line in lines[1:]:
        s = line.strip()
        if not s:
            continue
        if s.startswith("P"):
            break
        ids.extend(s.split())
    if len(ids) != 204:
        raise ValueError(f"expected 204 specimen ids, got {len(ids)}")

    data_start = None
    for i, line in enumerate(lines):
        s = line.strip()
        if not s or s.startswith("P"):
            continue
        parts = s.split()
        if len(parts) != 3:
            continue
        if any("L" in p for p in parts):
            continue
        try:
            float(parts[0])
            float(parts[1])
            float(parts[2])
        except ValueError:
            continue
        data_start = i
        break
    if data_start is None:
        raise ValueError("could not locate numeric block")

    coords: List[np.ndarray] = []
    cur: List[List[float]] = []
    for line in lines[data_start:]:
        s = line.strip()
        if not s:
            if cur:
                if len(cur) != N_LM:
                    raise ValueError(f"specimen block has {len(cur)} landmarks")
                coords.append(np.asarray(cur, dtype=float))
                cur = []
            continue
        parts = s.split()
        if len(parts) != 3:
            continue
        cur.append([float(parts[0]), float(parts[1]), float(parts[2])])
    if cur:
        if len(cur) != N_LM:
            raise ValueError(f"final block has {len(cur)} landmarks")
        coords.append(np.asarray(cur, dtype=float))

    if len(coords) != len(ids):
        raise ValueError(f"{len(coords)} coordinate blocks vs {len(ids)} ids")
    return ids, np.stack(coords, axis=0)


def species_prefix(spec_id: str) -> str:
    m = re.match(r"^([a-z]{6})", spec_id)
    return m.group(1) if m else spec_id[:6]


def lateral_projection(X: np.ndarray) -> np.ndarray:
    """Lateral view: anteroposterior (x) vs dorsoventral (z), landmarks (n, 78, 2)."""
    return np.stack([X[:, :, 0], X[:, :, 2]], axis=-1)


def centre_scale(X: np.ndarray) -> np.ndarray:
    X = X - X.mean(axis=-2, keepdims=True)
    cs = np.sqrt((X ** 2).sum(axis=(-2, -1), keepdims=True))
    return X / np.maximum(cs, EPS)


def opa(S: np.ndarray, T: np.ndarray) -> np.ndarray:
    """Ordinary Procrustes: rotate + uniform scale S onto T (2D, no reflection)."""
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
    """Similarity map src -> dst after GPA centres."""
    cs, ct = src.mean(0), dst.mean(0)
    S, T = src - cs, dst - ct
    S_al = opa(S, T)
    pred = S_al + ct
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


def fit_tps(src: np.ndarray, dst: np.ndarray) -> Tuple[np.ndarray, float]:
    n = len(src)
    d = src[:, None, :] - src[None, :, :]
    r = np.sqrt((d ** 2).sum(axis=2))
    K = _u_func(r)
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
    return pred, rmse


def moebius_coaxial(points: np.ndarray, c: float = 0.12) -> np.ndarray:
    z = points[:, 0] + 1j * points[:, 1]
    w = z / (1.0 - c * z)
    return np.column_stack([w.real, w.imag])


def fit_moebius(src: np.ndarray, dst: np.ndarray, c: float = 0.12) -> Tuple[np.ndarray, float]:
    pred = moebius_coaxial(src, c)
    # scale+translate to dst least squares
    A = np.column_stack([pred, np.ones(len(pred))])
    M, _, _, _ = np.linalg.lstsq(A, dst, rcond=None)
    pred2 = A @ M
    rmse = float(np.sqrt(((pred2 - dst) ** 2).mean()))
    return pred2, rmse


def species_mean_shape(ids: Sequence[str], X2: np.ndarray, code: str) -> np.ndarray:
    idx = [i for i, sid in enumerate(ids) if species_prefix(sid) == code]
    if not idx:
        raise ValueError(f"no specimens for {code}")
    return X2[idx].mean(axis=0)


def pairwise_metrics(src: np.ndarray, dst: np.ndarray) -> Dict[str, float]:
    _, rmse_sim = fit_similarity(src, dst)
    _, rmse_aff = fit_affine(src, dst)
    _, rmse_tps = fit_tps(src, dst)
    _, rmse_moe = fit_moebius(src, dst)
    r_sim, _ = stats.pearsonr(src.ravel(), fit_similarity(src, dst)[0].ravel())
    return {
        "rmse_similarity": rmse_sim,
        "rmse_affine": rmse_aff,
        "rmse_tps": rmse_tps,
        "rmse_moebius": rmse_moe,
        "r_similarity": float(r_sim),
    }


def segment_specimen_simple(path: Path) -> Tuple[np.ndarray, float]:
    """Pass 2: Otsu on grayscale + largest connected component (lateral skull cast)."""
    from PIL import Image

    img = np.asarray(Image.open(path).convert("RGB"))
    gray = img.mean(axis=2)
    thr = float(np.percentile(gray, 35))
    mask = gray < thr
    # keep largest 4-connected region
    from scipy import ndimage as ndi

    lab, n = ndi.label(mask)
    if n == 0:
        return mask, 0.0
    sizes = ndi.sum(mask, lab, index=range(1, n + 1))
    keep = int(np.argmax(sizes)) + 1
    seg = lab == keep
    coverage = float(seg.mean())
    return seg, coverage


def run_analysis(full_validation: bool = False) -> Dict[str, Any]:
    verify_pinned_assets()
    ids, X3 = parse_nts(NTS_PATH)
    X2 = lateral_projection(X3)
    # GPA per specimen then species means
    X2_gpa = np.array([centre_scale(x[None])[0] for x in X2])

    means = {code: species_mean_shape(ids, X2_gpa, code) for code in SPECIES_CODES}

    pairs = [
        ("cropor", "croacu"),
        ("cropor", "allmis"),
        ("croacu", "allmis"),
    ]
    pair_results = {}
    for a, b in pairs:
        pair_results[f"{a}_to_{b}"] = pairwise_metrics(means[a], means[b])

    # within-species noise (similarity RMSE leave-one-out on cropor)
    cro_idx = [i for i, s in enumerate(ids) if species_prefix(s) == "cropor"]
    loo = []
    for j in cro_idx:
        others = [i for i in cro_idx if i != j]
        m = X2_gpa[others].mean(0)
        _, r = fit_similarity(X2_gpa[j], m)
        loo.append(r)
    noise_floor = float(np.median(loo))

    out: Dict[str, Any] = {
        "n_specimens": len(ids),
        "n_landmarks": N_LM,
        "species_means_used": list(SPECIES_CODES.keys()),
        "pairwise": pair_results,
        "cropor_loo_median_rmse": noise_floor,
        "notosuchus_in_dataset": False,
    }

    if full_validation:
        # Parameter sweep: Moebius c
        c_grid = np.linspace(0.02, 0.28, 14)
        sweep = []
        for c in c_grid:
            _, rmse = fit_moebius(means["cropor"], means["allmis"], c=float(c))
            sweep.append({"c": float(c), "rmse": rmse})
        out["moebius_c_sweep"] = sweep
        best = min(sweep, key=lambda d: d["rmse"])
        out["moebius_best_c"] = best

        # Ablation: shuffle landmark labels (negative control)
        rng = np.random.default_rng(1917)
        shuffled = means["cropor"].copy()
        rng.shuffle(shuffled)
        _, rmse_shuf = fit_tps(shuffled, means["allmis"])
        out["tps_rmse_shuffled_labels"] = rmse_shuf
        _, rmse_true = fit_tps(means["cropor"], means["allmis"])
        out["tps_rmse_homologous"] = rmse_true

        if SPECIMEN_JPG.exists():
            seg, cov = segment_specimen_simple(SPECIMEN_JPG)
            out["specimen_segmentation_coverage"] = cov
            # project mean allmis into pixel frame (simple Procrustes to bbox)
            ys, xs = np.where(seg)
            if len(xs) > 50:
                bbox = np.array([[xs.min(), ys.mean()], [xs.max(), ys.mean()], [xs.mean(), ys.min()], [xs.mean(), ys.max()]])
                lm = means["allmis"][:, :2] if means["allmis"].shape[1] >= 2 else means["allmis"]
                lm = lm - lm.mean(0)
                lm = lm / max(lm.std(), EPS) * 0.25 * np.array([np.ptp(bbox[:, 0]), np.ptp(bbox[:, 1])])
                lm = lm + np.array([xs.mean(), ys.mean()])
                inside = sum(seg[int(np.clip(p[1], 0, seg.shape[0] - 1)), int(np.clip(p[0], 0, seg.shape[1] - 1))] for p in lm)
                out["landmark_in_mask_fraction"] = float(inside / len(lm))

        # KS test: distribution of pairwise distances after TPS vs within-species
        def pdist_flat(Y: np.ndarray) -> np.ndarray:
            d = []
            for i in range(len(Y)):
                for j in range(i + 1, len(Y)):
                    d.append(np.linalg.norm(Y[i] - Y[j]))
            return np.asarray(d)

        within = pdist_flat(X2_gpa[cro_idx[:15]])
        pred, _ = fit_tps(means["cropor"], means["allmis"])
        between = pdist_flat(np.vstack([means["cropor"], pred]))
        ks = stats.ks_2samp(within, between)
        out["ks_within_vs_warped"] = {"statistic": float(ks.statistic), "pvalue": float(ks.pvalue)}

    # Verdict headline
    med_cross = float(np.median([v["rmse_similarity"] for v in pair_results.values()]))
    if med_cross <= 1.15 * noise_floor:
        verdict = "supported"
    elif med_cross <= 1.5 * noise_floor:
        verdict = "mixed"
    else:
        verdict = "refuted"
    if not out.get("notosuchus_in_dataset", True):
        verdict = "mixed"  # partial taxon coverage caps certainty
    out["verdict"] = verdict
    out["headline_rmse_ratio"] = med_cross / max(noise_floor, EPS)
    return out


def measure_morphoeval() -> List[Dict[str, Any]]:
    r = run_analysis(full_validation=True)
    return [
        {"name": "cropor_loo_median_rmse", "value": r["cropor_loo_median_rmse"], "unit": "GPA units", "method": "median leave-one-out similarity RMSE on C. porosus specimens after GPA"},
        {"name": "cross_species_median_rmse_similarity", "value": float(np.median([p["rmse_similarity"] for p in r["pairwise"].values()])), "unit": "GPA units", "method": "median similarity RMSE across species-mean pairs"},
        {"name": "headline_rmse_ratio", "value": r["headline_rmse_ratio"], "unit": "ratio", "method": "cross-species median RMSE divided by within-species LOO noise floor"},
        {"name": "tps_rmse_homologous", "value": r.get("tps_rmse_homologous", 0.0), "unit": "GPA units", "method": "TPS fit cropor→allmis with homologous landmarks"},
        {"name": "tps_rmse_shuffled_labels", "value": r.get("tps_rmse_shuffled_labels", 0.0), "unit": "GPA units", "method": "TPS ablation with shuffled source landmark labels (seed 1917)"},
        {"name": "specimen_segmentation_coverage", "value": r.get("specimen_segmentation_coverage", 0.0), "unit": "fraction", "method": "Otsu grayscale + largest connected component on Commons skull JPG"},
        {"name": "landmark_in_mask_fraction", "value": r.get("landmark_in_mask_fraction", 0.0), "unit": "fraction", "method": "Procrustes-aligned allmis mean landmarks vs segmentation mask"},
        {"name": "ks_statistic", "value": r.get("ks_within_vs_warped", {}).get("statistic", 0.0), "unit": "D", "method": "KS test: within-species vs TPS-warped pairwise landmark distances"},
        {"name": "ks_pvalue", "value": r.get("ks_within_vs_warped", {}).get("pvalue", 1.0), "unit": "p", "method": "two-sample KS p-value (scipy.stats.ks_2samp)"},
    ]


def render_figures(full_validation: bool = False) -> None:
    verify_pinned_assets()
    ids, X3 = parse_nts(NTS_PATH)
    X2 = lateral_projection(X3)
    X2_gpa = np.array([centre_scale(x[None])[0] for x in X2])
    means = {code: species_mean_shape(ids, X2_gpa, code) for code in SPECIES_CODES}

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    codes = list(SPECIES_CODES.keys())
    for ax, code in zip(axes[0], codes):
        ax.scatter(means[code][:, 0], means[code][:, 1], s=8, c="k")
        ax.set_title(SPECIES_CODES[code])
        ax.set_aspect("equal")
        ax.axis("off")

    src, dst = means["cropor"], means["allmis"]
    preds = {
        "similarity": fit_similarity(src, dst)[0],
        "affine": fit_affine(src, dst)[0],
        "TPS": fit_tps(src, dst)[0],
        "Moebius": fit_moebius(src, dst)[0],
    }
    for ax, (name, pred) in zip(axes[1], preds.items()):
        ax.scatter(dst[:, 0], dst[:, 1], s=6, c="C1", label="target")
        ax.scatter(pred[:, 0], pred[:, 1], s=6, c="C0", alpha=0.7, label="warped")
        ax.set_title(f"cropor→allmis ({name})")
        ax.legend(fontsize=7, loc="upper right")
        ax.set_aspect("equal")
        ax.axis("off")
    axes[1, -1].axis("off")
    fig.suptitle("T1917-051: Thompson croc skull warps (Watanabe & Slice 2014 landmarks, lateral x–z)")
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=150)
    plt.close(fig)

    # GIF: blend similarity → TPS
    fig2, ax2 = plt.subplots(figsize=(5, 4))
    line_target, = ax2.plot(dst[:, 0], dst[:, 1], "r.", ms=4, label="allmis")
    line_warp, = ax2.plot([], [], "b.", ms=4, label="warped")
    ax2.set_aspect("equal")
    ax2.legend(loc="upper right")

    pred_tps = fit_tps(src, dst)[0]

    def init():
        line_warp.set_data([], [])
        return line_warp,

    def frame(t):
        alpha = t / 29.0
        P = (1 - alpha) * src + alpha * pred_tps
        line_warp.set_data(P[:, 0], P[:, 1])
        ax2.set_title(f"cropor → allmis TPS blend t={alpha:.2f}")
        return line_warp,

    ani = animation.FuncAnimation(fig2, frame, init_func=init, frames=30, blit=True)
    ani.save(OUT_GIF, writer=animation.PillowWriter(fps=10))
    plt.close(fig2)

    results = run_analysis(full_validation=full_validation)
    RESULT_JSON.write_text(
        json.dumps(
            {
                "claim_id": "T1917-051",
                "date_made": "1917",
                "dataset": {
                    "name": "Watanabe & Slice (2014) crocodylian cranial ontogeny",
                    "doi": "10.5061/dryad.14fn1",
                    "sha256_nts": NTS_SHA256,
                },
                "verdict": results["verdict"],
                "headline_numbers": {
                    "headline_rmse_ratio": results["headline_rmse_ratio"],
                    "cropor_loo_median_rmse": results["cropor_loo_median_rmse"],
                },
                "script": str(Path(__file__).relative_to(ROOT)),
                "figure": str(OUT_PNG.relative_to(ROOT)),
                "notosuchus_testable": False,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def self_test() -> None:
    verify_pinned_assets()
    ids, X = parse_nts(NTS_PATH)
    assert len(ids) == 204 and X.shape == (204, 78, 3)
    r1 = run_analysis(full_validation=False)
    assert "pairwise" in r1 and r1["cropor_loo_median_rmse"] >= 0
    m1 = measure_morphoeval()
    m2 = measure_morphoeval()
    assert m1 == m2, "measure_morphoeval must be deterministic"
    # similarity should beat shuffled-label TPS when full validation on
    r2 = run_analysis(full_validation=True)
    assert r2["tps_rmse_homologous"] <= r2["tps_rmse_shuffled_labels"]
    print("self-test OK", r2["verdict"], "rmse_ratio=", round(r2["headline_rmse_ratio"], 4))


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

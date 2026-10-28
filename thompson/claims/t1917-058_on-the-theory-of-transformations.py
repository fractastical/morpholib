#!/usr/bin/env python3
"""
thompson/claims/t1917-058_on-the-theory-of-transformations.py
==============================================================

Validation of ONE dated claim (T1917-058) from
Thompson, D'Arcy W. (1917) *On Growth and Form*, Ch. XVII "On the Theory of
Transformations, or the Comparison of Related Forms", pp. 770-773, Figs. 404-408:

    "The human skull, inscribed in Cartesian coordinates and deformed, gives the
     skulls of chimpanzee, baboon and dog: ape/human skull difference is a
     whole-diagram transformation (notably of the facial/cranial proportions)."

Loop: thompson1917-t1917-058 (slot 247). Script version: v2 (Pass 2 remediation of audit
2026-10-28-t1917-058-on-the-theory-of-transformatio-audit, items R1-R5):
  R1 landmark-jitter bootstrap (sigma 1.5 % centroid size on every landmark, 3 px on every net
     node, 1000 replicates) -> CIs for E_C, orderings, facial indices; label-permutation null
     for E_C; goodness of fit r / RMSE / two-sample KS / paired Wilcoxon on per-landmark residuals.
  R2 parameter-regime sweep: TPS regularisation lambda (0 -> affine limit) and intensity k
     (0 .. 2.5, plus the mirrored k = -1 "anti-Thompson" net).
  R3 negative / ablation controls: affine-only, mirrored net, label-shuffled null, wrong-target
     specificity index.
  R4 computer-vision segmentation of the four specimen photographs (saturation/value or
     non-white mask -> largest component) -> silhouette height profile, anterior taper index
     (Lean law 4) and silhouette face fraction; replaces the tied cosine test for C5.
  R5 drawing-fidelity reported in the verdict; C4/C6 orderings re-run with Thompson's Fig. 404
     drawing as the human reference (robustness to the rendered human model).

What is tested and how (equation numbers refer to
datasets/thompson_1917/t1917-058/t1917_058_on_the_theory_of_transformatio_reference.json)
-----------------------------------------------------------------------------
E1  Thompson's deformed net is reconstructed as the thin-plate spline T through the 30
    digitised nodes of Fig. 405 (human 5x4 Cartesian net -> chimpanzee net), Bookstein 1989.
E2  T = affine part A (the "rectangular box stretch" null) + bending part.
E3  Bending energy J[T] = tr(W^T K W): 0 iff T is affine
    (Lean: MathOfBiology.Loop_247.affine_net_cells_congruent - an affine map keeps every
    cell congruent, so a non-flat cell-area field *requires* bending).
E4  Cell-area ratios lambda(i,j) of the 20 cells; Thompson's "approximately logarithmic
    order" (p. 771) is the regression log lambda = alpha + beta_i i + beta_j j
    (Lean: MathOfBiology.Loop_247.log_area_gradient gives the sign conditions
    beta_i > 0 behind->forwards, beta_j < 0 above->downwards).
E5  Full Procrustes distance d_P between 11-landmark configurations.
E6  THE TEST (sub-claim C1): apply Thompson's drawn T to the human skull landmarks of
    Fig. 404 and ask whether the result is closer to a REAL chimpanzee skull
    (Wikimedia Commons, Museum Wiesbaden MWNH 230) than the human configuration was:
        E_C = 1 - d_P(T(H), C) / d_P(H, C).
    Controls: the affine part alone (box stretch), the wrong targets (dog, human).
E7  Facial/cranial proportions on the real specimens (sub-claim C6): facial index
    F = |NAS-PRO| / |NAS-OPC| and the facial angle phi between PRO->NAS and the Frankfort
    line ORBI->POR; Thompson: phi "increases ... to nearly a right angle in man".
Intensity family T_k = Id + k (T - Id) (Lean: MathOfBiology.Loop_247.intensity_family):
    sub-claims C2/C4 - baboon = "same order, increased intensity"; chimp nearer than baboon;
    dog "on different lines" (C5; Lean: rectangular_vs_triangular).

Data (all SHA-256 pinned in the reference JSON; see scout_manifest.json for provenance)
-----------------------------------------------------------------------------
* Thompson's own Figs. 404-408, Internet Archive scan `ongrowthform1917thom`
  (public domain; printed page = leaf - 20).
* Real lateral skulls of the four named taxa from Wikimedia Commons (CC BY-SA 3.0):
  Homo sapiens (rendered anatomical model), Pan troglodytes, Papio ursinus, Canis
  familiaris (Museum Wiesbaden photographs).
* The Dryad dataset named in the pre-scout (doi:10.5061/dryad.7r85116) is New World
  monkeys only -> DATASET_MISMATCH, recorded, NOT used.

Claim boundary
--------------
reported_in_source      : Figs. 404-408 and the prose statements C1-C6 (pp. 770-773).
reconstructed_in_script : T from the digitised Fig. 405 net; landmarks digitised by the
                          loop agent on Thompson's Fig. 404/406 and on the Commons specimens
                          (+/- ~12 px); facial indices; Procrustes tests.
transfer_boundary       : one specimen per taxon; the baboon and dog nets (Figs. 407-408)
                          were NOT digitised (drawn over heavy skull outlines) - those taxa
                          are tested on real specimens only.
does_not_claim          : anything about growth, development or phylogeny; no claim that
                          Thompson's specific curves are the *best* smooth map.

Invocation
----------
    python3 thompson/claims/t1917-058_on-the-theory-of-transformations.py --self-test
    python3 thompson/claims/t1917-058_on-the-theory-of-transformations.py [--no-gif]

Outputs
-------
    docs/media/t1917_058_on_the_theory_of_transformatio_comparison.png
    docs/media/t1917_058_on_the_theory_of_transformatio_growth.gif
    datasets/thompson_1917/t1917-058/t1917_058_metrics.json
    datasets/thompson_1917/t1917-058/result.json

Dependencies: numpy, matplotlib (Agg), Pillow (scipy optional, for image warping speed-up).
Deterministic (seed 1917); no network access.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "datasets" / "thompson_1917" / "t1917-058"
REF_PATH = DATA / "t1917_058_on_the_theory_of_transformatio_reference.json"
MEDIA = REPO / "docs" / "media"
SLUG = "t1917_058_on_the_theory_of_transformatio"
SCRIPT_VERSION = "v2"
SEED = 1917

LANDMARKS = ["PRO", "NAS", "RHI", "ORBS", "ORBI", "VTX", "OPC", "POR", "GON", "GNA", "MPOST"]
SPECIMENS = {
    "human": "homo_sapiens_specimen",
    "chimpanzee": "pan_troglodytes_specimen",
    "baboon": "papio_ursinus_specimen",
    "dog": "canis_familiaris_specimen",
}
SPECIMEN_IMAGES = {
    "human": "homo_sapiens_lateral.jpg",
    "chimpanzee": "pan_troglodytes_lateral.jpg",
    "baboon": "papio_ursinus_lateral.jpg",
    "dog": "canis_familiaris_lateral.jpg",
}


# --------------------------------------------------------------------------- data
def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_reference() -> dict:
    with open(REF_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def verify_pinned_files(ref: dict) -> dict:
    """Re-hash every vendored file against the reference JSON (source traceability)."""
    out = {}
    for name, digest in ref["pinned_files_sha256"].items():
        p = DATA / name
        ok = p.exists() and sha256_of(p) == digest
        out[name] = bool(ok)
    return out


def human_grid_nodes() -> np.ndarray:
    """Fig. 404 net nodes in grid units, row-major (j = 0..4 bottom->top, i = 0..5)."""
    return np.array([[i, j] for j in range(5) for i in range(6)], dtype=float)


def chimp_net_nodes_px(ref: dict) -> np.ndarray:
    nodes = ref["digitisation"]["fig405_chimpanzee_net_nodes_px"]["nodes"]
    rows = [nodes[k] for k in ("0", "a", "b", "c", "d")]
    arr = np.array([p for row in rows for p in row], dtype=float)
    arr[:, 1] *= -1.0  # image y down -> y up
    return arr


def fig404_px_to_grid(ref: dict, px: np.ndarray) -> np.ndarray:
    g = ref["digitisation"]["fig404_grid_pixels"]
    xs = np.array(g["x_lines_px"], float)
    ys = np.array([g["y_lines_px_image_down"][k] for k in ("0", "a", "b", "c", "d")], float)
    sx = np.polyfit(np.arange(6), xs, 1)
    sy = np.polyfit(np.arange(5), ys, 1)
    gx = (px[:, 0] - sx[1]) / sx[0]
    gy = (px[:, 1] - sy[1]) / sy[0]
    return np.column_stack([gx, gy])


def landmarks_px(ref: dict, key: str) -> tuple[np.ndarray, str]:
    d = ref["digitisation"]["landmarks_px_image_down"][key]
    arr = np.array([d[n] for n in LANDMARKS], dtype=float)
    return arr, d.get("facing", "right")


def landmarks_xy(ref: dict, key: str) -> np.ndarray:
    """Landmarks as (x right, y up) with the skull facing right."""
    arr, facing = landmarks_px(ref, key)
    out = arr.copy()
    out[:, 1] *= -1.0
    if facing == "left":
        out[:, 0] *= -1.0
    return out


# --------------------------------------------------------------------------- geometry
def tps_kernel(r2: np.ndarray) -> np.ndarray:
    """U(r) = r^2 log r written in r^2 (0 at r = 0)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        u = 0.5 * r2 * np.log(r2)
    u[~np.isfinite(u)] = 0.0
    return u


def tps_fit(src: np.ndarray, dst: np.ndarray, lam: float = 0.0) -> dict:
    """Thin-plate spline src -> dst (E1).  Returns weights W (n x 2), affine A (3 x 2),
    bending energy (E3) and the affine-only image of src (E2)."""
    n = src.shape[0]
    d2 = ((src[:, None, :] - src[None, :, :]) ** 2).sum(-1)
    K = tps_kernel(d2)
    P = np.hstack([np.ones((n, 1)), src])
    L = np.zeros((n + 3, n + 3))
    L[:n, :n] = K + lam * np.eye(n)
    L[:n, n:] = P
    L[n:, :n] = P.T
    rhs = np.vstack([dst, np.zeros((3, 2))])
    sol = np.linalg.solve(L, rhs)
    W, A = sol[:n], sol[n:]
    bending = float(np.trace(W.T @ K @ W))
    return {"src": src, "W": W, "A": A, "K": K, "lam": lam, "bending_energy": bending,
            "affine_matrix": A[1:].T, "translation": A[0]}


def tps_eval(model: dict, pts: np.ndarray) -> np.ndarray:
    src, W, A = model["src"], model["W"], model["A"]
    d2 = ((pts[:, None, :] - src[None, :, :]) ** 2).sum(-1)
    U = tps_kernel(d2)
    P = np.hstack([np.ones((pts.shape[0], 1)), pts])
    return P @ A + U @ W


def affine_eval(model: dict, pts: np.ndarray) -> np.ndarray:
    P = np.hstack([np.ones((pts.shape[0], 1)), pts])
    return P @ model["A"]


def affine_fit(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Least-squares affine map (3 x 2) - the box-stretch/shear null (E2)."""
    P = np.hstack([np.ones((src.shape[0], 1)), src])
    A, *_ = np.linalg.lstsq(P, dst, rcond=None)
    return A


def centroid_size(X: np.ndarray) -> float:
    Xc = X - X.mean(0)
    return float(np.sqrt((Xc ** 2).sum()))


def procrustes_align(X: np.ndarray, Y: np.ndarray, scale: bool = True) -> tuple[np.ndarray, float]:
    """Similarity-align Y onto X (translation, rotation, optional scale; no reflection).
    Returns aligned Y (in X's frame) and the residual RMS per landmark."""
    Xc, Yc = X - X.mean(0), Y - Y.mean(0)
    U, S, Vt = np.linalg.svd(Yc.T @ Xc)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        S[-1] *= -1
        R = U @ Vt
    s = S.sum() / (Yc ** 2).sum() if scale else 1.0
    Ya = s * (Yc @ R) + X.mean(0)
    rms = float(np.sqrt(((Ya - X) ** 2).sum(1).mean()))
    return Ya, rms


def procrustes_distance(X: np.ndarray, Y: np.ndarray) -> float:
    """Full Procrustes distance between the pre-shapes of X and Y (E5), in [0, 1]."""
    Xp = (X - X.mean(0)) / centroid_size(X)
    Yp = (Y - Y.mean(0)) / centroid_size(Y)
    U, S, Vt = np.linalg.svd(Yp.T @ Xp)
    if np.linalg.det(U @ Vt) < 0:
        S[-1] *= -1
    rho = float(np.clip(S.sum(), -1.0, 1.0))
    return float(math.sqrt(max(0.0, 1.0 - rho ** 2)))


def shoelace(poly: np.ndarray) -> float:
    x, y = poly[:, 0], poly[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def cell_area_ratios(nodes: np.ndarray) -> np.ndarray:
    """Area of each of the 20 cells of a 6 x 5 node net (row-major), normalised to the
    net's mean cell area (E4); returns a (4, 5) array [j, i]."""
    N = nodes.reshape(5, 6, 2)
    A = np.zeros((4, 5))
    for j in range(4):
        for i in range(5):
            quad = np.array([N[j, i], N[j, i + 1], N[j + 1, i + 1], N[j + 1, i]])
            A[j, i] = abs(shoelace(quad))
    return A / A.mean()


def log_area_regression(ratios: np.ndarray) -> dict:
    """log lambda(i,j) = alpha + beta_i (i+0.5) + beta_j (j+0.5)  (E4 / Lean law 2)."""
    J, I = np.indices(ratios.shape)
    X = np.column_stack([np.ones(ratios.size), I.ravel() + 0.5, J.ravel() + 0.5])
    y = np.log(ratios.ravel())
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    pred = X @ coef
    ss_res = float(((y - pred) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    # monotonicity along the net directions: fraction of forward / downward steps that increase
    fwd = np.diff(ratios, axis=1)
    down = -np.diff(ratios, axis=0)
    return {
        "alpha": float(coef[0]), "beta_i": float(coef[1]), "beta_j": float(coef[2]),
        "r2": float(r2), "rmse_log": float(math.sqrt(ss_res / ratios.size)),
        "frac_forward_steps_increasing": float((fwd > 0).mean()),
        "frac_downward_steps_increasing": float((down > 0).mean()),
        "area_ratio_front_over_back": float(ratios[:, -1].mean() / ratios[:, 0].mean()),
        "area_ratio_bottom_over_top": float(ratios[0, :].mean() / ratios[-1, :].mean()),
    }


def angle_deg(v1: np.ndarray, v2: np.ndarray) -> float:
    c = float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
    return float(math.degrees(math.acos(max(-1.0, min(1.0, c)))))


def facial_proportions(L: np.ndarray) -> dict:
    """E7 on an 11-landmark configuration (x right/anterior, y up)."""
    idx = {n: k for k, n in enumerate(LANDMARKS)}
    PRO, NAS, OPC, ORBI, POR = (L[idx[n]] for n in ("PRO", "NAS", "OPC", "ORBI", "POR"))
    GNA, VTX = L[idx["GNA"]], L[idx["VTX"]]
    face = np.linalg.norm(NAS - PRO)
    brain = np.linalg.norm(NAS - OPC)
    frankfort = ORBI - POR            # Frankfort horizontal, pointing anteriorly
    facial_line = NAS - PRO           # Camper's facial line, from prosthion up to nasion
    # Camper/Thompson convention: angle on the anterior side between the facial line and the
    # horizontal -> ~90 deg for an orthognathous human, smaller the more prognathous the snout
    phi = angle_deg(facial_line, -frankfort)
    # prognathism: how far PRO projects anterior of NAS along the Frankfort direction, per braincase length
    fdir = frankfort / np.linalg.norm(frankfort)
    prog = float(np.dot(PRO - NAS, fdir) / brain)
    skull_len = float(np.dot(PRO - OPC, fdir))
    return {"facial_index_F": float(face / brain), "facial_angle_deg": float(phi),
            "prognathism_PRO_ahead_of_NAS_per_braincase": prog,
            "face_fraction_of_skull_length": float(np.dot(PRO - NAS, fdir) / skull_len) if skull_len else float("nan")}


# --------------------------------------------------------------------------- Pass 2: statistics, sweeps, controls, segmentation
LAM_SWEEP = [0.0, 1e-3, 1e-2, 1e-1, 1.0, 10.0]
K_SWEEP = [-1.0, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5]
N_BOOT = 1000
N_PERM = 2000
SIGMA_FRAC = 0.015      # landmark digitisation uncertainty, fraction of centroid size
SIGMA_NODE_PX = 3.0     # Fig. 405 node uncertainty in crop pixels


def explained(H: np.ndarray, P: np.ndarray, C: np.ndarray) -> float:
    d0 = procrustes_distance(H, C)
    return 1.0 - procrustes_distance(P, C) / d0 if d0 > 0 else float("nan")


def goodness_of_fit(H404: np.ndarray, P: np.ndarray, C: np.ndarray) -> dict:
    """r, RMSE, KS and Wilcoxon on per-landmark residuals of T(H) vs C, against H vs C."""
    from scipy import stats
    Pn = (P - P.mean(0)) / centroid_size(P)
    Hn = (H404 - H404.mean(0)) / centroid_size(H404)
    C_on_P, _ = procrustes_align(Pn, C)
    C_on_H, _ = procrustes_align(Hn, C)
    res_T = np.linalg.norm(C_on_P - Pn, axis=1)
    res_0 = np.linalg.norm(C_on_H - Hn, axis=1)
    r = float(np.corrcoef(Pn.ravel(), C_on_P.ravel())[0, 1])
    r0 = float(np.corrcoef(Hn.ravel(), C_on_H.ravel())[0, 1])
    ks = stats.ks_2samp(res_T, res_0)
    try:
        w = stats.wilcoxon(res_0 - res_T, alternative="greater")
        w_p = float(w.pvalue)
    except ValueError:
        w_p = float("nan")
    return {"pearson_r_T_vs_real": r, "pearson_r_human_vs_real": r0,
            "rmse_T_vs_real_centroid_units": float(np.sqrt((res_T ** 2).mean())),
            "rmse_human_vs_real_centroid_units": float(np.sqrt((res_0 ** 2).mean())),
            "per_landmark_residual_T": dict(zip(LANDMARKS, res_T.round(4).tolist())),
            "per_landmark_residual_human": dict(zip(LANDMARKS, res_0.round(4).tolist())),
            "n_landmarks_improved": int((res_T < res_0).sum()),
            "ks_statistic": float(ks.statistic), "ks_pvalue": float(ks.pvalue),
            "wilcoxon_pvalue_residuals_reduced": w_p}


def permutation_null(H404: np.ndarray, P: np.ndarray, C: np.ndarray, n_perm: int = N_PERM, seed: int = SEED) -> dict:
    """Label-shuffled null (R3): does T move H toward THIS chimpanzee configuration, or toward
    any configuration of these 11 points?"""
    rng = np.random.default_rng(seed + 1)
    e_obs = explained(H404, P, C)
    es = np.empty(n_perm)
    for b in range(n_perm):
        perm = rng.permutation(C.shape[0])
        es[b] = explained(H404, P, C[perm])
    p = float((1 + (es >= e_obs).sum()) / (n_perm + 1))
    return {"E_observed": e_obs, "E_null_mean": float(es.mean()), "E_null_sd": float(es.std()),
            "E_null_q95": float(np.quantile(es, 0.95)), "p_value": p, "n_perm": n_perm}


def bootstrap_jitter(ref: dict, n_boot: int = N_BOOT, sigma_frac: float = SIGMA_FRAC, seed: int = SEED) -> dict:
    """R1: propagate digitisation uncertainty of every landmark and every net node."""
    rng = np.random.default_rng(seed)
    H = human_grid_nodes()
    Cpx = chimp_net_nodes_px(ref)
    H404 = fig404_px_to_grid(ref, landmarks_px(ref, "fig404_human_drawing")[0])
    real = {sp: landmarks_xy(ref, key) for sp, key in SPECIMENS.items()}
    cs = {sp: centroid_size(real[sp]) for sp in real}
    cs404 = centroid_size(H404)
    keys = ["E_chimp", "E_chimp_affine", "E_baboon", "E_dog", "E_human_specimen", "dir_cos",
            "log_r2", "beta_i", "beta_j", "d_HC", "d_HB", "d_HD", "cos_cb", "cos_cd", "k_bc",
            "F_h", "F_c", "F_b", "F_d", "phi_h", "phi_c", "phi_b", "phi_d"]
    rec = {k: np.empty(n_boot) for k in keys}
    for b in range(n_boot):
        nodes_px = Cpx + rng.normal(0, SIGMA_NODE_PX, Cpx.shape)
        nodes_al, _ = procrustes_align(H, nodes_px, scale=True)
        T = tps_fit(H, nodes_al)
        h = H404 + rng.normal(0, sigma_frac * cs404, H404.shape)
        rj = {sp: real[sp] + rng.normal(0, sigma_frac * cs[sp], real[sp].shape) for sp in real}
        P = tps_eval(T, h)
        Pa = affine_eval(T, h)
        rec["E_chimp"][b] = explained(h, P, rj["chimpanzee"])
        rec["E_chimp_affine"][b] = explained(h, Pa, rj["chimpanzee"])
        rec["E_baboon"][b] = explained(h, P, rj["baboon"])
        rec["E_dog"][b] = explained(h, P, rj["dog"])
        rec["E_human_specimen"][b] = explained(h, P, rj["human"])
        C_on_h, _ = procrustes_align(h, rj["chimpanzee"])
        rec["dir_cos"][b] = displacement_cosine(P - h, C_on_h - h)
        lf = log_area_regression(cell_area_ratios(nodes_al))
        rec["log_r2"][b], rec["beta_i"][b], rec["beta_j"][b] = lf["r2"], lf["beta_i"], lf["beta_j"]
        Hs = rj["human"]
        rec["d_HC"][b] = procrustes_distance(Hs, rj["chimpanzee"])
        rec["d_HB"][b] = procrustes_distance(Hs, rj["baboon"])
        rec["d_HD"][b] = procrustes_distance(Hs, rj["dog"])
        fC = procrustes_align(Hs, rj["chimpanzee"])[0] - Hs
        fB = procrustes_align(Hs, rj["baboon"])[0] - Hs
        fD = procrustes_align(Hs, rj["dog"])[0] - Hs
        rec["cos_cb"][b] = displacement_cosine(fC, fB)
        rec["cos_cd"][b] = displacement_cosine(fC, fD)
        rec["k_bc"][b] = (fB * fC).sum() / (fC ** 2).sum()
        for sp, tag in (("human", "h"), ("chimpanzee", "c"), ("baboon", "b"), ("dog", "d")):
            pr = facial_proportions(rj[sp])
            rec[f"F_{tag}"][b], rec[f"phi_{tag}"][b] = pr["facial_index_F"], pr["facial_angle_deg"]

    def summ(a):
        return {"mean": float(a.mean()), "sd": float(a.std(ddof=1)), "cv": float(a.std(ddof=1) / abs(a.mean())) if a.mean() != 0 else float("nan"),
                "ci95": [float(np.quantile(a, 0.025)), float(np.quantile(a, 0.975))]}
    out = {k: summ(v) for k, v in rec.items()}
    out["probabilities"] = {
        "P_E_chimp_gt_0": float((rec["E_chimp"] > 0).mean()),
        "P_E_chimp_gt_E_affine": float((rec["E_chimp"] > rec["E_chimp_affine"]).mean()),
        "P_E_chimp_is_largest_target": float(((rec["E_chimp"] > rec["E_baboon"]) & (rec["E_chimp"] > rec["E_dog"]) & (rec["E_chimp"] > rec["E_human_specimen"])).mean()),
        "P_log_r2_ge_0.8_with_signs": float(((rec["log_r2"] >= 0.8) & (rec["beta_i"] > 0) & (rec["beta_j"] < 0)).mean()),
        "P_order_dHC_lt_dHB_lt_dHD": float(((rec["d_HC"] < rec["d_HB"]) & (rec["d_HB"] < rec["d_HD"])).mean()),
        "P_cos_cb_ge_0.8": float((rec["cos_cb"] >= 0.8).mean()),
        "P_k_bc_gt_1": float((rec["k_bc"] > 1).mean()),
        "P_F_h_lt_F_c_lt_F_b": float(((rec["F_h"] < rec["F_c"]) & (rec["F_c"] < rec["F_b"])).mean()),
        "P_phi_h_gt_phi_c_gt_phi_b": float(((rec["phi_h"] > rec["phi_c"]) & (rec["phi_c"] > rec["phi_b"])).mean()),
    }
    out["settings"] = {"n_boot": n_boot, "sigma_frac_centroid_size": sigma_frac, "sigma_node_px": SIGMA_NODE_PX, "seed": seed}
    return out


def parameter_sweep(ref: dict) -> dict:
    """R2: lambda (regularisation -> affine limit) and k (intensity, incl. the mirrored net)."""
    H = human_grid_nodes()
    H404 = fig404_px_to_grid(ref, landmarks_px(ref, "fig404_human_drawing")[0])
    real = {sp: landmarks_xy(ref, key) for sp, key in SPECIMENS.items()}
    lam_rows = []
    for lam in LAM_SWEEP:
        T = build_transformation(ref, lam=lam)
        P = tps_eval(T, H404)
        lam_rows.append({"lam": lam, "bending_energy": T["bending_energy"],
                         "node_rmse_grid_units": float(np.sqrt(((tps_eval(T, H) - T["chimp_nodes_aligned"]) ** 2).sum(1).mean())),
                         "E_chimp": explained(H404, P, real["chimpanzee"]),
                         "log_area_r2": log_area_regression(cell_area_ratios(tps_eval(T, H)))["r2"]})
    T = build_transformation(ref)
    k_rows = []
    for k in K_SWEEP:
        nodes_k = H + k * (T["chimp_nodes_aligned"] - H)
        Tk = tps_fit(H, nodes_k)
        P = tps_eval(Tk, H404)
        pr = facial_proportions(P)
        ratios = cell_area_ratios(nodes_k) if k != 0 else np.ones((4, 5))
        k_rows.append({"k": k, "E_chimp": explained(H404, P, real["chimpanzee"]),
                       "E_baboon": explained(H404, P, real["baboon"]), "E_dog": explained(H404, P, real["dog"]),
                       "E_human_specimen": explained(H404, P, real["human"]),
                       "predicted_facial_angle_deg": pr["facial_angle_deg"], "predicted_facial_index_F": pr["facial_index_F"],
                       "area_ratio_front_over_back": float(ratios[:, -1].mean() / ratios[:, 0].mean()),
                       "bending_energy": Tk["bending_energy"]})
    ks = [r["k"] for r in k_rows if 0 <= r["k"] <= 1.5]
    ang = [r["predicted_facial_angle_deg"] for r in k_rows if 0 <= r["k"] <= 1.5]
    # fine scan of the intensity along Thompson's own ray: where does each real target sit?
    kf = np.round(np.arange(0.0, 3.0001, 0.05), 3)
    Ef = {sp: [] for sp in ("chimpanzee", "baboon", "dog")}
    for k in kf:
        Tk = tps_fit(H, H + k * (T["chimp_nodes_aligned"] - H))
        P = tps_eval(Tk, H404)
        for sp in Ef:
            Ef[sp].append(explained(H404, P, real[sp]))
    k_opt = {sp: float(kf[int(np.argmax(Ef[sp]))]) for sp in Ef}
    E_opt = {sp: float(max(Ef[sp])) for sp in Ef}
    return {"lambda_sweep": lam_rows, "k_sweep": k_rows,
            "k_fine": kf.tolist(), "E_fine": {sp: np.round(Ef[sp], 4).tolist() for sp in Ef},
            "k_optimum_along_thompson_ray": k_opt, "E_at_k_optimum": E_opt,
            "baboon_optimum_more_intense_than_chimp": bool(k_opt["baboon"] > k_opt["chimpanzee"]),
            "facial_angle_monotone_decreasing_k_0_to_1.5": bool(all(np.diff(ang) < 0)),
            "bending_monotone_decreasing_in_lambda": bool(all(np.diff([r["bending_energy"] for r in lam_rows]) <= 1e-9)),
            "E_chimp_at_k": {str(r["k"]): r["E_chimp"] for r in k_rows}}


def segment_specimen(sp: str, ref: dict) -> dict:
    """R4: silhouette segmentation of the specimen photograph and height-profile morphometrics."""
    from PIL import Image
    from scipy import ndimage as ndi
    img = np.asarray(Image.open(DATA / SPECIMEN_IMAGES[sp]).convert("RGB")).astype(float) / 255.0
    if sp == "human":   # white model on a pure-white (RGB = 1.0) page: anything that is not page-white
        mask = img.min(axis=2) < 0.995
        mask = ndi.binary_closing(mask, structure=np.ones((15, 15)))
        method = "non-white mask (min RGB < 0.995; background is exactly 1.0) + closing 15 px on the rendered model"
    else:               # cream bone vs grey backdrop / white stand: HSV saturation + value
        mx, mn = img.max(axis=2), img.min(axis=2)
        sat = np.where(mx > 0, (mx - mn) / np.maximum(mx, 1e-6), 0)
        mask = (sat > 0.16) & (mx > 0.32)
        method = "HSV saturation > 0.16 and value > 0.32 (cream bone vs grey backdrop), opening r=3, hole fill, largest component"
    mask = ndi.binary_opening(mask, structure=np.ones((5, 5)))
    mask = ndi.binary_fill_holes(mask)
    lab, n = ndi.label(mask)
    if n == 0:
        return {"error": "no component"}
    sizes = ndi.sum(mask, lab, range(1, n + 1))
    mask = lab == (int(np.argmax(sizes)) + 1)
    ys, xs = np.nonzero(mask)
    pts = np.column_stack([xs, ys]).astype(float)
    lm, _ = landmarks_px(ref, SPECIMENS[sp])
    idx = {n_: k for k, n_ in enumerate(LANDMARKS)}
    axis = lm[idx["PRO"]] - lm[idx["OPC"]]
    L = np.linalg.norm(axis); ax = axis / L
    perp = np.array([-ax[1], ax[0]])
    u = (pts - lm[idx["OPC"]]) @ ax / L          # 0 at opisthocranion, 1 at prosthion
    v = (pts - lm[idx["OPC"]]) @ perp
    nb = 40
    edges = np.linspace(-0.1, 1.1, nb + 1)
    prof_u, prof_h = [], []
    for a, b_ in zip(edges[:-1], edges[1:]):
        sel = (u >= a) & (u < b_)
        if sel.sum() > 20:
            prof_u.append(0.5 * (a + b_)); prof_h.append(float(v[sel].max() - v[sel].min()))
    prof_u, prof_h = np.array(prof_u), np.array(prof_h)
    back = prof_h[(prof_u >= 0.15) & (prof_u <= 0.35)].mean()
    front = prof_h[(prof_u >= 0.65) & (prof_u <= 0.85)].mean()
    u_orb = float(((lm[idx["ORBI"]] - lm[idx["OPC"]]) @ ax) / L)
    area = float(mask.sum())
    hmax = float(prof_h.max())
    return {"method": method, "mask_area_px": area, "n_components": int(n),
            "taper_index_front_over_back": float(front / back),
            "triangularity_area_over_box": float(area / (L * hmax)) if hmax > 0 else float("nan"),
            "silhouette_face_fraction_anterior_of_orbit": float((u > u_orb).mean()),
            "u_orbitale": u_orb, "length_over_max_height": float(L / hmax),
            "profile_u": prof_u.round(3).tolist(), "profile_h_px": prof_h.round(1).tolist(),
            "_mask": mask}


# --------------------------------------------------------------------------- the analysis
def build_transformation(ref: dict, lam: float = 0.0) -> dict:
    """E1: Thompson's human->chimpanzee net as a TPS in human grid units."""
    H = human_grid_nodes()
    Cpx = chimp_net_nodes_px(ref)
    C_aligned, rms = procrustes_align(H, Cpx, scale=True)   # similarity only - keeps the warp
    model = tps_fit(H, C_aligned, lam=lam)
    model["chimp_nodes_aligned"] = C_aligned
    model["net_alignment_rms_grid_units"] = rms
    model["chimp_nodes_px"] = Cpx
    return model


def displacement_cosine(u: np.ndarray, v: np.ndarray) -> float:
    return float((u * v).sum() / (np.linalg.norm(u) * np.linalg.norm(v)))


def analyse(ref: dict, lam: float = 0.0, intensity_k: float = 1.0, full: bool = True) -> dict:
    T = build_transformation(ref, lam=lam)
    H = human_grid_nodes()
    nodes_k = H + intensity_k * (T["chimp_nodes_aligned"] - H)
    Tk = tps_fit(H, nodes_k, lam=lam)

    # --- E4 cell-area field of Thompson's net
    ratios = cell_area_ratios(T["chimp_nodes_aligned"])
    logfit = log_area_regression(ratios)
    affine_flat = cell_area_ratios(affine_eval(T, H))      # Lean law 1: must be flat (=1)

    # --- landmarks
    H404_px, _ = landmarks_px(ref, "fig404_human_drawing")
    H404 = fig404_px_to_grid(ref, np.column_stack([H404_px[:, 0], H404_px[:, 1]]))
    # fig404_px_to_grid handles the y flip through the fitted (negative) slope
    real = {sp: landmarks_xy(ref, key) for sp, key in SPECIMENS.items()}
    C406 = landmarks_xy(ref, "fig406_chimpanzee_drawing")

    # --- E6 the test: Thompson's drawn transformation applied to Thompson's drawn human
    P_full = tps_eval(Tk, H404)
    P_aff = affine_eval(Tk, H404)
    d = procrustes_distance
    targets = {"chimpanzee": real["chimpanzee"], "baboon": real["baboon"], "dog": real["dog"],
               "human_specimen": real["human"], "chimpanzee_drawing_fig406": C406}
    explained_targets = {}
    for name, tgt in targets.items():
        d0, d1, d1a = d(H404, tgt), d(P_full, tgt), d(P_aff, tgt)
        explained_targets[name] = {"d_human404_to_target": d0, "d_T_human404_to_target": d1,
                           "d_affineT_human404_to_target": d1a,
                           "explained_fraction_full": 1.0 - d1 / d0 if d0 > 0 else float("nan"),
                           "explained_fraction_affine_only": 1.0 - d1a / d0 if d0 > 0 else float("nan")}

    # --- direction agreement: Thompson's drawn displacement vs the real human->chimp displacement
    C_on_H404, _ = procrustes_align(H404, real["chimpanzee"])
    u_real = C_on_H404 - H404
    u_thompson = P_full - H404
    dir_cos = displacement_cosine(u_thompson, u_real)
    k_fit = float((u_thompson * u_real).sum() / (u_thompson ** 2).sum())

    # --- real-specimen comparisons (C2, C4, C5)
    Hs = real["human"]
    dist = {sp: d(Hs, real[sp]) for sp in ("chimpanzee", "baboon", "dog")}
    fields = {}
    for sp in ("chimpanzee", "baboon", "dog"):
        aligned, _ = procrustes_align(Hs, real[sp])
        fields[sp] = aligned - Hs
    cos_cb = displacement_cosine(fields["chimpanzee"], fields["baboon"])
    cos_cd = displacement_cosine(fields["chimpanzee"], fields["dog"])
    cos_bd = displacement_cosine(fields["baboon"], fields["dog"])
    k_baboon_over_chimp = float((fields["baboon"] * fields["chimpanzee"]).sum() / (fields["chimpanzee"] ** 2).sum())
    k_dog_over_chimp = float((fields["dog"] * fields["chimpanzee"]).sum() / (fields["chimpanzee"] ** 2).sum())

    # --- TPS decomposition of the real human->target changes (E2/E3)
    decomp = {}
    for sp in ("chimpanzee", "baboon", "dog"):
        aligned, _ = procrustes_align(Hs, real[sp])
        Hn = Hs / centroid_size(Hs)
        An = aligned / centroid_size(Hs)
        m = tps_fit(Hn, An)
        Aff = affine_fit(Hn, An)
        P = np.hstack([np.ones((Hn.shape[0], 1)), Hn])
        aff_disp = P @ Aff - Hn
        tot_disp = An - Hn
        decomp[sp] = {"bending_energy": m["bending_energy"],
                      "affine_fraction_of_displacement_ss": float((aff_disp ** 2).sum() / (tot_disp ** 2).sum()),
                      "affine_matrix": m["affine_matrix"].tolist(),
                      "det_affine": float(np.linalg.det(m["affine_matrix"]))}

    # --- E7 facial proportions
    props = {sp: facial_proportions(real[sp]) for sp in real}
    props["human_drawing_fig404"] = facial_proportions(H404)
    props["chimpanzee_drawing_fig406"] = facial_proportions(C406)
    props["T_human404_predicted_chimpanzee"] = facial_proportions(P_full)

    # drawing fidelity: how far Thompson's drawings sit from the real specimens
    fidelity = {"d_human_drawing404_vs_human_specimen": d(H404, real["human"]),
                "d_chimp_drawing406_vs_chimp_specimen": d(C406, real["chimpanzee"]),
                "d_human_specimen_vs_chimp_specimen": dist["chimpanzee"]}

    # ---------------- Pass 2 (R1-R5) ----------------
    pass2 = {}
    if full:
        pass2["goodness_of_fit"] = goodness_of_fit(H404, P_full, real["chimpanzee"])
        pass2["permutation_null"] = permutation_null(H404, P_full, real["chimpanzee"])
        pass2["bootstrap"] = bootstrap_jitter(ref)
        pass2["sweep"] = parameter_sweep(ref)
        # R3 controls: mirrored (anti-Thompson) net, wrong-target specificity
        nodes_m = H - 1.0 * (T["chimp_nodes_aligned"] - H)
        Pm = tps_eval(tps_fit(H, nodes_m), H404)
        pass2["controls"] = {
            "mirrored_net_E_chimp": explained(H404, Pm, real["chimpanzee"]),
            "affine_only_E_chimp": explained_targets["chimpanzee"]["explained_fraction_affine_only"],
            "specificity_index_E_chimp_minus_max_other": (explained_targets["chimpanzee"]["explained_fraction_full"]
                - max(explained_targets[k]["explained_fraction_full"] for k in ("baboon", "dog", "human_specimen"))),
            "E_by_target": {k: explained_targets[k]["explained_fraction_full"] for k in explained_targets},
        }
        # R4 segmentation
        seg = {sp: segment_specimen(sp, ref) for sp in SPECIMENS}
        pass2["segmentation"] = {sp: {k: v for k, v in seg[sp].items() if not k.startswith("_")} for sp in seg}
        pass2["_masks"] = {sp: seg[sp]["_mask"] for sp in seg}
        taper = {sp: seg[sp]["taper_index_front_over_back"] for sp in seg}
        pass2["segmentation"]["summary"] = {
            "taper_index": taper,
            "dog_most_triangular": taper["dog"] < min(taper[s_] for s_ in ("human", "chimpanzee", "baboon")),
            "triangularity": {sp: seg[sp]["triangularity_area_over_box"] for sp in seg},
            "silhouette_face_fraction": {sp: seg[sp]["silhouette_face_fraction_anterior_of_orbit"] for sp in seg},
            "face_fraction_order_H_lt_C_lt_B": (seg["human"]["silhouette_face_fraction_anterior_of_orbit"] <
                                                seg["chimpanzee"]["silhouette_face_fraction_anterior_of_orbit"] <
                                                seg["baboon"]["silhouette_face_fraction_anterior_of_orbit"]),
        }
        # R5 robustness: real-specimen orderings with Thompson's drawn human as the reference
        dist404 = {sp: d(H404, real[sp]) for sp in ("chimpanzee", "baboon", "dog")}
        pass2["robustness_human_reference"] = {
            "distances_from_fig404_human": dist404,
            "order_holds_with_fig404": dist404["chimpanzee"] < dist404["baboon"] < dist404["dog"],
            "facial_index_fig404": props["human_drawing_fig404"]["facial_index_F"],
            "facial_angle_fig404": props["human_drawing_fig404"]["facial_angle_deg"],
            "F_order_with_fig404": props["human_drawing_fig404"]["facial_index_F"] < props["chimpanzee"]["facial_index_F"] < props["baboon"]["facial_index_F"],
            "phi_order_with_fig404": props["human_drawing_fig404"]["facial_angle_deg"] > props["chimpanzee"]["facial_angle_deg"] > props["baboon"]["facial_angle_deg"],
        }

    return {
        "transformation": {
            "lam": lam, "intensity_k": intensity_k,
            "bending_energy_grid_units": T["bending_energy"],
            "affine_matrix": T["affine_matrix"].tolist(),
            "det_affine": float(np.linalg.det(T["affine_matrix"])),
            "net_alignment_rms_grid_units": T["net_alignment_rms_grid_units"],
            "cell_area_ratios_rows_bottom_to_top": ratios.tolist(),
            "affine_part_cell_area_ratios_max_abs_dev_from_flat": float(np.abs(affine_flat - 1).max()),
            "log_area_fit": logfit,
        },
        "landmarks": {"human_fig404_grid_units": H404.tolist(), "T_human_fig404": P_full.tolist(),
                      "affineT_human_fig404": P_aff.tolist(),
                      **{f"real_{sp}": real[sp].tolist() for sp in real},
                      "chimpanzee_fig406": C406.tolist()},
        "C1_whole_diagram_test": {"targets": explained_targets,
                                  "direction_cosine_thompson_vs_real_human_to_chimp": dir_cos,
                                  "intensity_k_fit_of_real_on_thompson": k_fit},
        "C4_distances_from_human_specimen": dist,
        "C2_C5_displacement_fields": {"cosine_chimp_baboon": cos_cb, "cosine_chimp_dog": cos_cd,
                                      "cosine_baboon_dog": cos_bd,
                                      "intensity_k_baboon_over_chimp": k_baboon_over_chimp,
                                      "intensity_k_dog_over_chimp": k_dog_over_chimp},
        "tps_decomposition_real": decomp,
        "C6_facial_proportions": props,
        "drawing_fidelity": fidelity,
        "pass2": {k: v for k, v in pass2.items() if not k.startswith("_")},
        "_masks": pass2.get("_masks", {}),
        "_model": T, "_Tk": Tk, "_H404": H404, "_P_full": P_full, "_real": real, "_C406": C406,
    }


def verdict_from(res: dict) -> dict:
    c1 = res["C1_whole_diagram_test"]["targets"]["chimpanzee"]
    lf = res["transformation"]["log_area_fit"]
    dist = res["C4_distances_from_human_specimen"]
    props = res["C6_facial_proportions"]
    fields = res["C2_C5_displacement_fields"]
    checks = {
        "C1_T_moves_human_toward_real_chimp": c1["explained_fraction_full"] > 0.0,
        "C1_T_beats_affine_box_stretch": c1["d_T_human404_to_target"] < c1["d_affineT_human404_to_target"],
        "C1_T_targets_chimp_best": (c1["d_T_human404_to_target"] <
                                    min(res["C1_whole_diagram_test"]["targets"][k]["d_T_human404_to_target"]
                                        for k in ("dog", "human_specimen"))),
        "C3_log_area_gradient_r2_ge_0.8": lf["r2"] >= 0.8,
        "C3_signs_forward_up_down_down": (lf["beta_i"] > 0) and (lf["beta_j"] < 0),
        "C4_chimp_nearer_than_baboon_than_dog": dist["chimpanzee"] < dist["baboon"] < dist["dog"],
        "C2_baboon_same_direction_as_chimp_cos_ge_0.8": fields["cosine_chimp_baboon"] >= 0.8,
        "C2_baboon_more_intense_k_gt_1": fields["intensity_k_baboon_over_chimp"] > 1.0,
        "C6_facial_index_H_lt_C_lt_B": props["human"]["facial_index_F"] < props["chimpanzee"]["facial_index_F"] < props["baboon"]["facial_index_F"],
        "C6_facial_angle_H_gt_C_gt_B": props["human"]["facial_angle_deg"] > props["chimpanzee"]["facial_angle_deg"] > props["baboon"]["facial_angle_deg"],
    }
    p2 = res.get("pass2") or {}
    if p2:
        bs, pn, seg = p2["bootstrap"], p2["permutation_null"], p2["segmentation"]["summary"]
        checks.update({
            "C1_bootstrap_CI95_of_E_chimp_excludes_0": bs["E_chimp"]["ci95"][0] > 0.0,
            "C1_permutation_null_p_lt_0.05": pn["p_value"] < 0.05,
            "C1_specificity_E_chimp_largest_of_targets": p2["controls"]["specificity_index_E_chimp_minus_max_other"] > 0.0,
            "C1_mirrored_net_moves_away": p2["controls"]["mirrored_net_E_chimp"] < 0.0,
            "C5_dog_most_triangular_silhouette": bool(seg["dog_most_triangular"]),
            "C4_order_robust_P_ge_0.9": bs["probabilities"]["P_order_dHC_lt_dHB_lt_dHD"] >= 0.9,
            "C6_face_orders_robust_P_ge_0.9": min(bs["probabilities"]["P_F_h_lt_F_c_lt_F_b"], bs["probabilities"]["P_phi_h_gt_phi_c_gt_phi_b"]) >= 0.9,
        })
    else:
        checks["C5_dog_less_aligned_with_chimp_than_baboon"] = fields["cosine_chimp_dog"] < fields["cosine_chimp_baboon"]
    n_pass = sum(checks.values())
    core = checks["C1_T_moves_human_toward_real_chimp"] and checks["C1_T_beats_affine_box_stretch"]
    if p2:
        core = core and checks["C1_bootstrap_CI95_of_E_chimp_excludes_0"] and checks["C1_permutation_null_p_lt_0.05"]
    frac = n_pass / len(checks)
    if core and frac >= 0.8:
        verdict = "supported"
    elif not checks["C1_T_moves_human_toward_real_chimp"] or (p2 and not checks["C1_bootstrap_CI95_of_E_chimp_excludes_0"] and not checks["C1_permutation_null_p_lt_0.05"]):
        verdict = "refuted"
    else:
        verdict = "mixed"
    failed = [k for k, v in checks.items() if not v]
    return {"verdict": verdict, "checks": checks, "n_checks_passed": int(n_pass), "n_checks": len(checks), "failed_checks": failed}


def _m(name, value, unit, method, reference=None, source=None, note=None, derived_from="script"):
    m = {"name": name, "value": None if value is None else float(round(float(value), 4)), "unit": unit,
         "method": method, "derived_from": derived_from}
    if reference is not None:
        m["reference_value"] = reference
    if source:
        m["reference_source"] = source
    if note:
        m["note"] = note
    return m


def measure_morphoeval(res: dict | None = None) -> list[dict]:
    """Reference-anchored morphometric measurements for morphoeval/measure_outputs.py."""
    if res is None:
        res = analyse(load_reference())
    c1 = res["C1_whole_diagram_test"]
    chimp = c1["targets"]["chimpanzee"]
    lf = res["transformation"]["log_area_fit"]
    props = res["C6_facial_proportions"]
    dist = res["C4_distances_from_human_specimen"]
    fld = res["C2_C5_displacement_fields"]
    dec = res["tps_decomposition_real"]
    out = [
        _m("Explained fraction of the human->chimpanzee Procrustes distance removed by Thompson's Fig. 405 net applied to the Fig. 404 human landmarks (E6)",
           chimp["explained_fraction_full"], "fraction (dimensionless)",
           "TPS through the 30 digitised Fig. 405 nodes applied to 11 human landmarks of Fig. 404; full Procrustes distance to the real Pan troglodytes specimen (MWNH 230) before and after",
           reference=0.0, source="null: no transformation (E = 0); Thompson (1917) p. 771 claims the deformed net 'constitutes a simple projection of our human skull' onto the chimpanzee",
           note=f"d(H,C)={chimp['d_human404_to_target']:.3f} -> d(T(H),C)={chimp['d_T_human404_to_target']:.3f}; affine box-stretch part alone: E={chimp['explained_fraction_affine_only']:+.3f}"),
        _m("Explained fraction by the affine (box-stretch) part of the net only (E2 null)",
           chimp["explained_fraction_affine_only"], "fraction (dimensionless)",
           "affine component A of the TPS applied to the Fig. 404 landmarks; Procrustes distance to the real chimpanzee",
           reference=0.0, source="repo box-vs-warping protocol (thompson/run_box_vs_warping_audit.py): a rectangular box stretch is the trivial alternative to Thompson's curvilinear warp"),
        _m("Direction cosine between Thompson's drawn displacement field and the real human->chimpanzee displacement field at the 11 landmarks",
           c1["direction_cosine_thompson_vs_real_human_to_chimp"], "cosine (dimensionless)",
           "both fields expressed in the Fig. 404 human frame after Procrustes superimposition of the real chimpanzee onto the drawn human",
           reference=1.0, source="perfect agreement of direction; 0 = unrelated"),
        _m("Log-linear fit R^2 of the chimpanzee-net cell areas, log lambda = alpha + beta_i i + beta_j j (E4)",
           lf["r2"], "R^2 (dimensionless)",
           "shoelace areas of the 20 digitised cells of Fig. 405 normalised to their mean, OLS on cell-centre indices",
           reference=1.0, source="Thompson (1917) p. 771: areas increase 'in a gradual and approximately logarithmic order'",
           note=f"beta_i={lf['beta_i']:+.3f} (behind->forwards), beta_j={lf['beta_j']:+.3f} (upwards); Lean law 2 requires beta_i>0, beta_j<0"),
        _m("Cell-area ratio front column / back column of the chimpanzee net", lf["area_ratio_front_over_back"], "ratio (dimensionless)",
           "mean normalised area of column i=4 over column i=0 (Fig. 405)", reference=1.0, source="affine null (Lean law 1: flat area field)"),
        _m("Cell-area ratio bottom row / top row of the chimpanzee net", lf["area_ratio_bottom_over_top"], "ratio (dimensionless)",
           "mean normalised area of row j=0 over row j=3 (Fig. 405)", reference=1.0, source="affine null (Lean law 1: flat area field)"),
        _m("Bending energy of the reconstructed Fig. 405 net (E3)", res["transformation"]["bending_energy_grid_units"], "grid-unit^2 (dimensionless)",
           "tr(W^T K W) of the TPS in human grid units after similarity alignment of the net", reference=0.0, source="affine null (zero bending)"),
        _m("Procrustes distance human specimen -> chimpanzee specimen", dist["chimpanzee"], "full Procrustes distance (dimensionless)",
           "11 landmarks on the Commons specimens", reference=None, source=None),
        _m("Procrustes distance human specimen -> baboon specimen", dist["baboon"], "full Procrustes distance (dimensionless)",
           "11 landmarks on the Commons specimens", note="Thompson p. 772: chimpanzee 'much nearer to the human type than is the baboon'"),
        _m("Procrustes distance human specimen -> dog specimen", dist["dog"], "full Procrustes distance (dimensionless)",
           "11 landmarks on the Commons specimens", note="Thompson p. 773: the dog bridges 'a wider gulf'"),
        _m("Cosine between the human->chimpanzee and human->baboon displacement fields (C2)", fld["cosine_chimp_baboon"], "cosine (dimensionless)",
           "Procrustes superimposition of each specimen on the human specimen", reference=1.0, source="Thompson p. 771: baboon transformation 'of precisely the same order'"),
        _m("Intensity ratio k_baboon / k_chimpanzee (projection of the baboon field on the chimpanzee field)", fld["intensity_k_baboon_over_chimp"], "ratio (dimensionless)",
           "<u_B,u_C>/<u_C,u_C>", reference=1.0, source="Thompson p. 771: baboon differs 'only in an increased intensity' (k > 1)"),
        _m("Cosine between the human->chimpanzee and human->dog displacement fields (C5)", fld["cosine_chimp_dog"], "cosine (dimensionless)",
           "Procrustes superimposition of each specimen on the human specimen", note="Thompson p. 773: dog deformation 'on different lines'"),
    ]
    camper = {"human": (80, 90), "chimpanzee": (50, 70), "baboon": (35, 60), "dog": (30, 50)}
    for sp in ("human", "chimpanzee", "baboon", "dog"):
        out.append(_m(f"Facial angle ({sp} specimen): PRO->NAS vs Frankfort ORBI->POR, anterior side (E7)",
                      props[sp]["facial_angle_deg"], "degrees",
                      "angle between Camper's facial line and the Frankfort horizontal on the digitised landmarks",
                      reference=f"{camper[sp][0]}-{camper[sp][1]}", source="textbook Camper facial-angle bands (Camper 1791; Thompson p. 770: 'nearly a right angle in man')"))
        out.append(_m(f"Facial index F = |NAS-PRO| / |NAS-OPC| ({sp} specimen)", props[sp]["facial_index_F"], "ratio (dimensionless)",
                      "distances between digitised landmarks", note="Thompson p. 770: relative 'diminution or enfeeblement' of the jaws in man"))
    for sp in ("chimpanzee", "baboon", "dog"):
        out.append(_m(f"Affine fraction of the human->{sp} landmark displacement (E2/E3 decomposition)",
                      dec[sp]["affine_fraction_of_displacement_ss"], "fraction of displacement sum of squares",
                      "least-squares affine fit of the Procrustes-superimposed specimen to the human specimen; remainder is non-affine (bending)",
                      reference=1.0, source="box-stretch null: all change affine"))
    p2 = res.get("pass2") or {}
    if p2:
        bs, pn, gof, sw, ct = p2["bootstrap"], p2["permutation_null"], p2["goodness_of_fit"], p2["sweep"], p2["controls"]
        seg = p2["segmentation"]
        out += [
            _m("Bootstrap mean of E_chimp under landmark-digitisation jitter (R1)", bs["E_chimp"]["mean"], "fraction (dimensionless)",
               f"{bs['settings']['n_boot']} replicates, sigma = {bs['settings']['sigma_frac_centroid_size']*100:.1f} % centroid size on all landmarks, {bs['settings']['sigma_node_px']:.0f} px on all 30 net nodes",
               reference=0.0, source="null: no transformation", note=f"95 % CI [{bs['E_chimp']['ci95'][0]:.3f}, {bs['E_chimp']['ci95'][1]:.3f}], CV {bs['E_chimp']['cv']:.2f}, P(E>0) = {bs['probabilities']['P_E_chimp_gt_0']:.3f}"),
            _m("Bootstrap 95 % CI lower bound of E_chimp (R1)", bs["E_chimp"]["ci95"][0], "fraction (dimensionless)", "2.5th percentile of the jitter bootstrap", reference=0.0, source="support requires the CI to exclude 0"),
            _m("Permutation-null p-value of E_chimp (label-shuffled chimpanzee landmarks, R3)", pn["p_value"], "probability",
               f"{pn['n_perm']} label permutations of the real chimpanzee configuration; null mean {pn['E_null_mean']:+.3f}, sd {pn['E_null_sd']:.3f}, q95 {pn['E_null_q95']:.3f}", reference=0.05, source="alpha"),
            _m("Pearson r between T(H) and the real chimpanzee coordinates after Procrustes superimposition (R1)", gof["pearson_r_T_vs_real"], "r (dimensionless)",
               "22 centred, scaled coordinates", reference=gof["pearson_r_human_vs_real"], source="r of the untransformed human drawing vs the same chimpanzee"),
            _m("RMSE of T(H) vs real chimpanzee per landmark (R1)", gof["rmse_T_vs_real_centroid_units"], "centroid-size units",
               "root-mean-square Procrustes residual over 11 landmarks", reference=gof["rmse_human_vs_real_centroid_units"], source="RMSE of the untransformed human drawing vs the same chimpanzee",
               note=f"{gof['n_landmarks_improved']}/11 landmarks improved; Wilcoxon p = {gof['wilcoxon_pvalue_residuals_reduced']:.3g}"),
            _m("Two-sample KS statistic, per-landmark residuals T(H)-vs-C against H-vs-C (R1)", gof["ks_statistic"], "D (dimensionless)",
               "scipy.stats.ks_2samp on 11 + 11 residual distances", note=f"p = {gof['ks_pvalue']:.3f} (n = 11 per sample: low power)"),
            _m("Mirrored (anti-Thompson, k = -1) net: E_chimp (R3 negative control)", ct["mirrored_net_E_chimp"], "fraction (dimensionless)",
               "T_{-1} = Id - (T - Id) applied to the Fig. 404 landmarks", reference=0.0, source="a direction-blind effect would give E ~ 0; a correct direction gives E < 0 here"),
            _m("Specificity index E_chimp - max(E_baboon, E_dog, E_human) (R3)", ct["specificity_index_E_chimp_minus_max_other"], "fraction (dimensionless)",
               "wrong-target control", reference=0.0, source="T drawn for the chimpanzee should help the chimpanzee most", note=f"P(chimp largest) = {bs['probabilities']['P_E_chimp_is_largest_target']:.3f}"),
            _m("Optimal intensity k* along Thompson's ray for the real chimpanzee (R2)", sw["k_optimum_along_thompson_ray"]["chimpanzee"], "k (dimensionless)",
               "argmax over k in [0, 3] step 0.05 of E(k) with T_k = Id + k (T - Id)", reference=1.0, source="Thompson drew the chimpanzee net at k = 1", note=f"E(k*) = {sw['E_at_k_optimum']['chimpanzee']:.3f}"),
            _m("Optimal intensity k* along Thompson's ray for the real baboon (R2)", sw["k_optimum_along_thompson_ray"]["baboon"], "k (dimensionless)",
               "argmax over k in [0, 3] of E_baboon(k)", reference=">k*_chimp", source="Thompson p. 771: baboon = same transformation at increased intensity", note=f"E(k*) = {sw['E_at_k_optimum']['baboon']:.3f}; dog k* = {sw['k_optimum_along_thompson_ray']['dog']:.2f}"),
            _m("Predicted facial angle of T_k(H) at k = 1 (Thompson's chimpanzee net) (R2)", [r for r in sw["k_sweep"] if r["k"] == 1.0][0]["predicted_facial_angle_deg"], "degrees",
               "E7 on the transformed Fig. 404 landmarks", reference=props["chimpanzee"]["facial_angle_deg"], source="facial angle measured on the real chimpanzee specimen",
               note="monotone decreasing in k on [0, 1.5]: " + str(sw["facial_angle_monotone_decreasing_k_0_to_1.5"])),
        ]
        for sp in ("human", "chimpanzee", "baboon", "dog"):
            out.append(_m(f"Silhouette taper index h(front)/h(back) from image segmentation ({sp}) (R4)", seg[sp]["taper_index_front_over_back"], "ratio (dimensionless)",
                          seg[sp]["method"] + "; height profile along OPC->PRO, front = 65-85 %, back = 15-35 % of length",
                          reference=1.0, source="rectangular conformation (Lean law 4); Thompson p. 773: the dog is triangular (< 1)"))
            out.append(_m(f"Silhouette face fraction anterior of orbitale ({sp}) (R4)", seg[sp]["silhouette_face_fraction_anterior_of_orbit"], "fraction of silhouette area",
                          "segmented mask pixels with projection on OPC->PRO beyond orbitale", note="Thompson p. 770: enlargement of the jaws relative to the braincase in apes"))
    return out


# --------------------------------------------------------------------------- figures
def _warp_image_through_net(img: np.ndarray, ref: dict, H: np.ndarray, nodes_k: np.ndarray,
                            out_shape=(360, 520), margin=0.4) -> np.ndarray:
    """Backward-warp the Fig. 404 crop so that its Cartesian net becomes the net `nodes_k`
    (Thompson's own demonstration).  Inverse map: TPS nodes_k -> human grid."""
    inv = tps_fit(nodes_k, H)
    xmin, ymin = nodes_k.min(0) - margin
    xmax, ymax = nodes_k.max(0) + margin
    h, w = out_shape
    gx, gy = np.meshgrid(np.linspace(xmin, xmax, w), np.linspace(ymax, ymin, h))
    pts = np.column_stack([gx.ravel(), gy.ravel()])
    src_grid = tps_eval(inv, pts)
    g = ref["digitisation"]["fig404_grid_pixels"]
    xs = np.array(g["x_lines_px"], float)
    ys = np.array([g["y_lines_px_image_down"][k] for k in ("0", "a", "b", "c", "d")], float)
    sx = np.polyfit(np.arange(6), xs, 1)
    sy = np.polyfit(np.arange(5), ys, 1)
    px = src_grid[:, 0] * sx[0] + sx[1]
    py = src_grid[:, 1] * sy[0] + sy[1]
    Hh, Ww = img.shape[:2]
    xi = np.clip(np.round(px).astype(int), 0, Ww - 1)
    yi = np.clip(np.round(py).astype(int), 0, Hh - 1)
    valid = (px >= 0) & (px < Ww) & (py >= 0) & (py < Hh)
    out = np.full((h * w,) + img.shape[2:], 235, dtype=img.dtype)
    out[valid] = img[yi[valid], xi[valid]]
    return out.reshape((h, w) + img.shape[2:])


def _grid_lines(model: dict, H: np.ndarray, n: int = 40):
    lines = []
    for i in range(6):
        t = np.linspace(0, 4, n)
        lines.append(tps_eval(model, np.column_stack([np.full(n, i), t])))
    for j in range(5):
        t = np.linspace(0, 5, n)
        lines.append(tps_eval(model, np.column_stack([t, np.full(n, j)])))
    return lines


def make_comparison_figure(ref: dict, res: dict, out_png: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    T, H404, P = res["_model"], res["_H404"], res["_P_full"]
    H = human_grid_nodes()
    p2 = res.get("pass2") or {}
    nrows = 4 if p2 else 3
    fig = plt.figure(figsize=(18, 15 if p2 else 11))
    gs = fig.add_gridspec(nrows, 4, height_ratios=[1.1, 1.0, 1.0, 1.0][:nrows])

    # (a) Fig. 404 with landmarks
    ax = fig.add_subplot(gs[0, 0])
    im404 = np.asarray(Image.open(DATA / "fig404_human_skull_cartesian.jpg").convert("RGB"))
    ax.imshow(im404)
    pxs, _ = landmarks_px(ref, "fig404_human_drawing")
    ax.scatter(pxs[:, 0], pxs[:, 1], s=28, c="red", zorder=3)
    for n, (x, y) in zip(LANDMARKS, pxs):
        ax.annotate(n, (x, y), fontsize=6, color="darkred", xytext=(3, 3), textcoords="offset points")
    ax.set_title("Thompson Fig. 404: human skull in Cartesian net\n(11 digitised landmarks)", fontsize=9)
    ax.axis("off")

    # (b) Fig. 405 digitised nodes + TPS reconstruction
    ax = fig.add_subplot(gs[0, 1])
    im405 = np.asarray(Image.open(DATA / "fig405_chimpanzee_coordinates.jpg").convert("RGB"))
    ax.imshow(im405)
    Cpx = T["chimp_nodes_px"].copy(); Cpx[:, 1] *= -1
    ax.scatter(Cpx[:, 0], Cpx[:, 1], s=18, c="red", zorder=3)
    ax.set_title("Fig. 405: chimpanzee net, 30 digitised nodes\n(skeleton junctions + 4 manual corners)", fontsize=9)
    ax.axis("off")

    # (c) TPS net + warped human skull drawing
    ax = fig.add_subplot(gs[0, 2])
    warped = _warp_image_through_net(im404, ref, H, T["chimp_nodes_aligned"])
    xmin, ymin = T["chimp_nodes_aligned"].min(0) - 0.4
    xmax, ymax = T["chimp_nodes_aligned"].max(0) + 0.4
    ax.imshow(warped, extent=[xmin, xmax, ymin, ymax])
    for ln in _grid_lines(T, H):
        ax.plot(ln[:, 0], ln[:, 1], color="steelblue", lw=0.7)
    ax.scatter(P[:, 0], P[:, 1], s=20, c="red", zorder=3)
    ax.set_title("E1: human drawing warped through the TPS of Fig. 405\n"
                 f"bending energy {T['bending_energy']:.2f}, det A = {np.linalg.det(T['affine_matrix']):.2f}", fontsize=9)
    ax.set_aspect("equal"); ax.axis("off")

    # (d) cell-area field
    ax = fig.add_subplot(gs[0, 3])
    ratios = np.array(res["transformation"]["cell_area_ratios_rows_bottom_to_top"])
    imsh = ax.imshow(np.log(ratios), origin="lower", cmap="viridis", extent=[0, 5, 0, 4])
    for j in range(4):
        for i in range(5):
            ax.text(i + 0.5, j + 0.5, f"{ratios[j, i]:.2f}", ha="center", va="center", fontsize=7, color="w")
    lf = res["transformation"]["log_area_fit"]
    ax.set_title(f"E4 cell-area ratio of the chimp net (log colour)\nlog-linear fit R²={lf['r2']:.2f}, "
                 f"β_i={lf['beta_i']:+.2f} (fwd), β_j={lf['beta_j']:+.2f} (up)", fontsize=9)
    ax.set_xlabel("abscissa i (back → front)"); ax.set_ylabel("ordinate j (bottom → top)")
    fig.colorbar(imsh, ax=ax, fraction=0.046)

    # (e) specimens with landmarks
    for col, sp in enumerate(("human", "chimpanzee", "baboon", "dog")):
        ax = fig.add_subplot(gs[1, col])
        img = np.asarray(Image.open(DATA / SPECIMEN_IMAGES[sp]).convert("RGB"))
        ax.imshow(img)
        pxs, _ = landmarks_px(ref, SPECIMENS[sp])
        ax.scatter(pxs[:, 0], pxs[:, 1], s=16, c="yellow", edgecolors="k", linewidths=0.4, zorder=3)
        pr = res["C6_facial_proportions"][sp]
        ax.set_title(f"{sp}: F={pr['facial_index_F']:.2f}, facial angle={pr['facial_angle_deg']:.0f}°", fontsize=9)
        ax.axis("off")

    # (f) the test: superimposed shapes
    ax = fig.add_subplot(gs[2, 0:2])
    real_c = res["_real"]["chimpanzee"]
    Hn = (H404 - H404.mean(0)) / centroid_size(H404)
    Ca, _ = procrustes_align(Hn, real_c)
    Pa, _ = procrustes_align(Hn, P)
    Aa, _ = procrustes_align(Hn, np.array(res["landmarks"]["affineT_human_fig404"]))
    for X, c, lab, mk in ((Hn, "0.5", "human (Fig. 404)", "o"), (Ca, "red", "REAL chimpanzee specimen", "s"),
                          (Pa, "royalblue", "T(human): Thompson's net applied", "^"),
                          (Aa, "orange", "affine part only (box stretch)", "x")):
        ax.scatter(X[:, 0], X[:, 1], c=c, label=lab, marker=mk, s=36, zorder=3)
    for k in range(len(LANDMARKS)):
        ax.plot([Hn[k, 0], Pa[k, 0]], [Hn[k, 1], Pa[k, 1]], color="royalblue", lw=0.6)
        ax.plot([Hn[k, 0], Ca[k, 0]], [Hn[k, 1], Ca[k, 1]], color="red", lw=0.6, ls=":")
        ax.annotate(LANDMARKS[k], Hn[k], fontsize=6, color="0.3", xytext=(2, 2), textcoords="offset points")
    c1 = res["C1_whole_diagram_test"]["targets"]["chimpanzee"]
    ax.set_title("E6 the test (Procrustes frame): does Thompson's deformed net move the human landmarks "
                 "toward a real chimpanzee?\n"
                 f"d(H,C)={c1['d_human404_to_target']:.3f} → d(T(H),C)={c1['d_T_human404_to_target']:.3f}: "
                 f"explained {100*c1['explained_fraction_full']:.0f}% (affine part only: "
                 f"{100*c1['explained_fraction_affine_only']:.0f}%); direction cosine "
                 f"{res['C1_whole_diagram_test']['direction_cosine_thompson_vs_real_human_to_chimp']:.2f}", fontsize=8.5)
    ax.legend(fontsize=7, loc="lower left"); ax.set_aspect("equal"); ax.grid(alpha=0.2)

    # (g) real-specimen summary bars
    ax = fig.add_subplot(gs[2, 2])
    dist = res["C4_distances_from_human_specimen"]
    sps = ["chimpanzee", "baboon", "dog"]
    ax.bar(sps, [dist[s] for s in sps], color=["royalblue", "darkorange", "firebrick"])
    f = res["C2_C5_displacement_fields"]
    ax.set_title(f"C4: Procrustes distance from the human specimen\n"
                 f"cos(chimp,baboon)={f['cosine_chimp_baboon']:.2f}, cos(chimp,dog)={f['cosine_chimp_dog']:.2f}, "
                 f"k_baboon/k_chimp={f['intensity_k_baboon_over_chimp']:.2f}", fontsize=8.5)
    ax.set_ylabel("d_P")

    ax = fig.add_subplot(gs[2, 3])
    props = res["C6_facial_proportions"]
    sps4 = ["human", "chimpanzee", "baboon", "dog"]
    ax.plot(sps4, [props[s]["facial_index_F"] for s in sps4], "o-", color="purple", label="facial index F")
    ax2 = ax.twinx()
    ax2.plot(sps4, [props[s]["facial_angle_deg"] for s in sps4], "s--", color="green", label="facial angle (°)")
    ax.set_ylabel("F = |NAS-PRO| / |NAS-OPC|", color="purple"); ax2.set_ylabel("facial angle (°)", color="green")
    ax.set_title("C6: facial/cranial proportions on the real specimens", fontsize=9)

    if p2:
        bs, pn, sw, seg = p2["bootstrap"], p2["permutation_null"], p2["sweep"], p2["segmentation"]
        # (h) bootstrap vs permutation null
        ax = fig.add_subplot(gs[3, 0])
        rng = np.random.default_rng(SEED)
        null = rng.normal(pn["E_null_mean"], pn["E_null_sd"], 4000)
        ax.hist(null, bins=40, color="0.7", alpha=0.8, density=True, label=f"label-shuffled null (p={pn['p_value']:.4f})")
        ci = bs["E_chimp"]["ci95"]
        ax.axvspan(ci[0], ci[1], color="royalblue", alpha=0.25, label=f"jitter bootstrap 95% CI [{ci[0]:.2f}, {ci[1]:.2f}]")
        ax.axvline(pn["E_observed"], color="royalblue", lw=2, label=f"observed E = {pn['E_observed']:.3f}")
        ax.axvline(p2["controls"]["mirrored_net_E_chimp"], color="firebrick", ls="--", label=f"mirrored net E = {p2['controls']['mirrored_net_E_chimp']:.2f}")
        ax.axvline(p2["controls"]["affine_only_E_chimp"], color="orange", ls=":", label=f"affine only E = {p2['controls']['affine_only_E_chimp']:.2f}")
        ax.set_xlabel("explained fraction E (human → real chimpanzee)"); ax.legend(fontsize=6.5, loc="upper left")
        ax.set_title("R1/R3: uncertainty and controls for the headline", fontsize=9)
        # (i) k sweep
        ax = fig.add_subplot(gs[3, 1])
        kf = np.array(sw["k_fine"])
        for sp, c in (("chimpanzee", "royalblue"), ("baboon", "darkorange"), ("dog", "firebrick")):
            ax.plot(kf, sw["E_fine"][sp], color=c, label=f"{sp}: k*={sw['k_optimum_along_thompson_ray'][sp]:.2f}")
            ax.axvline(sw["k_optimum_along_thompson_ray"][sp], color=c, lw=0.6, ls=":")
        ax.axhline(0, color="k", lw=0.5); ax.axvline(1.0, color="k", lw=0.8, ls="--")
        ax.set_xlabel("intensity k of T_k = Id + k (T − Id)   (k = 1: Thompson's net)"); ax.set_ylabel("explained fraction E(k)")
        ax2 = ax.twinx()
        ks_ = [r["k"] for r in sw["k_sweep"] if r["k"] >= 0]
        ax2.plot(ks_, [r["predicted_facial_angle_deg"] for r in sw["k_sweep"] if r["k"] >= 0], "s-", color="green", ms=3, label="predicted facial angle")
        ax2.set_ylabel("predicted facial angle (°)", color="green")
        ax.legend(fontsize=7, loc="lower left")
        ax.set_title("R2: intensity sweep along Thompson's own ray", fontsize=9)
        # (j) lambda sweep
        ax = fig.add_subplot(gs[3, 2])
        lams = [max(r["lam"], 1e-4) for r in sw["lambda_sweep"]]
        ax.semilogx(lams, [r["E_chimp"] for r in sw["lambda_sweep"]], "o-", color="royalblue", label="E_chimp")
        ax.set_ylabel("E_chimp", color="royalblue"); ax.set_xlabel("TPS regularisation λ (10⁻⁴ ≡ 0; → affine limit)")
        ax3 = ax.twinx()
        ax3.semilogx(lams, [r["bending_energy"] for r in sw["lambda_sweep"]], "^--", color="purple", label="bending energy")
        ax3.set_ylabel("bending energy", color="purple")
        ax.set_title("R2: regularisation sweep (smooth ↔ affine)", fontsize=9)
        # (k) segmentation taper
        ax = fig.add_subplot(gs[3, 3])
        for sp, c in (("human", "0.3"), ("chimpanzee", "royalblue"), ("baboon", "darkorange"), ("dog", "firebrick")):
            u = np.array(seg[sp]["profile_u"]); h = np.array(seg[sp]["profile_h_px"])
            ax.plot(u, h / h.max(), color=c, label=f"{sp}: taper {seg[sp]['taper_index_front_over_back']:.2f}, face {seg[sp]['silhouette_face_fraction_anterior_of_orbit']:.2f}")
        ax.axvspan(0.15, 0.35, color="0.9"); ax.axvspan(0.65, 0.85, color="0.9")
        ax.set_xlabel("position along OPC → PRO (0 = occiput, 1 = prosthion)"); ax.set_ylabel("silhouette height / max")
        ax.legend(fontsize=6.5, loc="lower left")
        ax.set_title("R4: segmented-silhouette height profiles (dog = most triangular?)", fontsize=9)
    v = verdict_from(res)
    fig.suptitle(f"T1917-058 — Thompson (1917) Ch. XVII Figs. 404-408: is the ape/human skull difference a whole-diagram "
                 f"transformation?   verdict: {v['verdict'].upper()} ({v['n_checks_passed']}/{v['n_checks']} checks)  [script {SCRIPT_VERSION}]",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=110)
    plt.close(fig)


def make_growth_gif(ref: dict, res: dict, out_gif: Path, n_frames: int = 16) -> None:
    """Morph the human Cartesian net (k = 0) into Thompson's chimpanzee net (k = 1) and on to
    his 'increased intensity' hypothesis (k = 2), warping the Fig. 404 drawing along."""
    from PIL import Image, ImageDraw
    T, H404 = res["_model"], res["_H404"]
    H = human_grid_nodes()
    im404 = np.asarray(Image.open(DATA / "fig404_human_skull_cartesian.jpg").convert("RGB"))
    frames = []
    ks = list(np.linspace(0, 1, n_frames // 2 + 1)) + list(np.linspace(1, 2, n_frames // 2 + 1))[1:]
    ks += [1.0, 0.0]  # hold frames: back to Thompson's net, then to the human net
    nodes_2 = H + 2.0 * (T["chimp_nodes_aligned"] - H)
    xmin, ymin = np.minimum(H.min(0), nodes_2.min(0)) - 0.5
    xmax, ymax = np.maximum(H.max(0), nodes_2.max(0)) + 0.5
    W, Hh = 480, 340
    sx, sy = W / (xmax - xmin), Hh / (ymax - ymin)
    for k in ks:
        nodes_k = H + k * (T["chimp_nodes_aligned"] - H)
        inv = tps_fit(nodes_k, H)
        gx, gy = np.meshgrid(np.linspace(xmin, xmax, W), np.linspace(ymax, ymin, Hh))
        pts = np.column_stack([gx.ravel(), gy.ravel()])
        src = tps_eval(inv, pts)
        g = ref["digitisation"]["fig404_grid_pixels"]
        xs = np.array(g["x_lines_px"], float)
        ys = np.array([g["y_lines_px_image_down"][kk] for kk in ("0", "a", "b", "c", "d")], float)
        fx = np.polyfit(np.arange(6), xs, 1); fy = np.polyfit(np.arange(5), ys, 1)
        px = src[:, 0] * fx[0] + fx[1]; py = src[:, 1] * fy[0] + fy[1]
        valid = (px >= 0) & (px < im404.shape[1]) & (py >= 0) & (py < im404.shape[0])
        xi = np.clip(np.round(px).astype(int), 0, im404.shape[1] - 1)
        yi = np.clip(np.round(py).astype(int), 0, im404.shape[0] - 1)
        out = np.full((Hh * W, 3), 240, np.uint8)
        out[valid] = im404[yi[valid], xi[valid]]
        frame = Image.fromarray(out.reshape(Hh, W, 3))
        dr = ImageDraw.Draw(frame)
        fwd = tps_fit(H, nodes_k)
        for ln in _grid_lines(fwd, H, n=30):
            pts2 = [((x - xmin) * sx, (ymax - y) * sy) for x, y in ln]
            dr.line(pts2, fill=(30, 90, 200), width=1)
        Pk = tps_eval(fwd, H404)
        for x, y in Pk:
            X, Y = (x - xmin) * sx, (ymax - y) * sy
            dr.ellipse([X - 3, Y - 3, X + 3, Y + 3], fill=(220, 30, 30))
        label = "human net (k=0)" if k < 0.02 else ("Thompson's chimpanzee net (k=1)" if abs(k - 1) < 0.02 else
                 ("'increased intensity' k=2" if k > 1.98 else f"intensity k = {k:.2f}"))
        dr.rectangle([0, 0, W, 18], fill=(255, 255, 255))
        dr.text((6, 3), f"T1917-058  T_k = Id + k (T - Id)   {label}", fill=(0, 0, 0))
        frames.append(frame.quantize(colors=48, method=Image.Quantize.MEDIANCUT))
    durations = [160] * (len(frames) - 2) + [900, 900]
    out_gif.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(out_gif, save_all=True, append_images=frames[1:], duration=durations, loop=0, optimize=True)


# --------------------------------------------------------------------------- self-test
def self_test() -> int:
    t0 = time.time()
    rng = np.random.default_rng(SEED)
    ref = load_reference()
    pins = verify_pinned_files(ref)
    assert all(pins.values()), f"pinned file digest mismatch: {[k for k, v in pins.items() if not v]}"
    # TPS interpolates its nodes exactly and is the identity for identical nodes
    src = rng.normal(size=(12, 2)); dst = src + 0.3 * rng.normal(size=(12, 2))
    m = tps_fit(src, dst)
    assert np.allclose(tps_eval(m, src), dst, atol=1e-8)
    mid = tps_fit(src, src)
    assert mid["bending_energy"] < 1e-9 and np.allclose(mid["affine_matrix"], np.eye(2), atol=1e-8)
    # Lean law 1 anchor: affine map -> flat cell-area field; a=3,d=2 -> ratio 6 relative to unit cells
    H = human_grid_nodes()
    aff = H * np.array([3.0, 2.0])
    A = cell_area_ratios(aff)
    assert np.allclose(A, 1.0), "affine box stretch must give a flat normalised area field"
    raw = np.array([abs(shoelace(np.array([[0, 0], [3, 0], [3, 2], [0, 2]], float)))])
    assert np.isclose(raw[0], 6.0)
    # Lean law 2 anchor: synthetic log-linear areas are recovered
    J, I = np.indices((4, 5))
    synth = np.exp(0.3 * (I + 0.5) - 0.2 * (J + 0.5))
    lf = log_area_regression(synth / synth.mean())
    assert abs(lf["beta_i"] - 0.3) < 1e-9 and abs(lf["beta_j"] + 0.2) < 1e-9 and lf["r2"] > 0.999
    # Lean law 3 anchor: intensity family displacement is linear in k
    T = build_transformation(ref)
    for k in (0.0, 0.5, 2.0):
        nodes_k = H + k * (T["chimp_nodes_aligned"] - H)
        assert np.allclose(nodes_k - H, k * (T["chimp_nodes_aligned"] - H))
    # Procrustes distance: 0 for similarity copies, symmetric, rotation invariant
    X = rng.normal(size=(11, 2)); th = 0.7
    R = np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]])
    Y = 2.5 * X @ R.T + np.array([3.0, -1.0])
    assert procrustes_distance(X, Y) < 1e-9
    Z = X + 0.2 * rng.normal(size=X.shape)
    assert abs(procrustes_distance(X, Z) - procrustes_distance(Z, X)) < 1e-9
    # data integrity
    assert chimp_net_nodes_px(ref).shape == (30, 2)
    for key in ("fig404_human_drawing", "fig406_chimpanzee_drawing", *SPECIMENS.values()):
        assert landmarks_px(ref, key)[0].shape == (11, 2), key
    # the reconstructed net must bend (Thompson's lines are curved) and dilate forward/down
    assert T["bending_energy"] > 0.01
    res = analyse(ref)
    lf = res["transformation"]["log_area_fit"]
    assert np.isfinite(lf["r2"]) and -1 <= res["C1_whole_diagram_test"]["targets"]["chimpanzee"]["explained_fraction_full"] <= 1
    mm = measure_morphoeval(res)
    assert len(mm) >= 20 and all({"name", "value", "unit", "method"} <= set(m) for m in mm)
    # Pass 2: permutation null is centred on 0 and reproducible; bootstrap reproducible; masks sane
    pn = res["pass2"]["permutation_null"]
    assert abs(pn["E_null_mean"]) < 0.05 and 0 < pn["p_value"] <= 1
    bs2 = bootstrap_jitter(ref, n_boot=50)
    bs3 = bootstrap_jitter(ref, n_boot=50)
    assert bs2["E_chimp"]["mean"] == bs3["E_chimp"]["mean"], "bootstrap must be seed-deterministic"
    for sp in SPECIMENS:
        sg = res["pass2"]["segmentation"][sp]
        assert 0.3 < sg["taper_index_front_over_back"] < 1.5 and sg["mask_area_px"] > 1e5, sp
    sw = res["pass2"]["sweep"]
    assert sw["lambda_sweep"][0]["node_rmse_grid_units"] < 1e-6 and sw["bending_monotone_decreasing_in_lambda"]
    assert [r for r in sw["k_sweep"] if r["k"] == 0.0][0]["E_chimp"] == 0.0
    print(f"[self-test] OK in {time.time() - t0:.1f}s  (pinned files {sum(pins.values())}/{len(pins)}, "
          f"net bending {T['bending_energy']:.3f}, E_chimp {mm[0]['value']:+.3f}, "
          f"log-area R² {lf['r2']:.2f}, {len(mm)} measurements)")
    return 0


# --------------------------------------------------------------------------- main
def pass2_headline(res: dict) -> dict:
    p2 = res.get("pass2") or {}
    if not p2:
        return {}
    bs, pn, gof, sw, seg = p2["bootstrap"], p2["permutation_null"], p2["goodness_of_fit"], p2["sweep"], p2["segmentation"]
    return {
        "E_chimp_bootstrap_mean_ci95": [round(bs["E_chimp"]["mean"], 3), [round(x, 3) for x in bs["E_chimp"]["ci95"]]],
        "E_chimp_permutation_p": pn["p_value"],
        "pearson_r_T_vs_real_chimp": round(gof["pearson_r_T_vs_real"], 3),
        "pearson_r_human_vs_real_chimp": round(gof["pearson_r_human_vs_real"], 3),
        "rmse_T_vs_real_chimp_centroid_units": round(gof["rmse_T_vs_real_centroid_units"], 4),
        "rmse_human_vs_real_chimp_centroid_units": round(gof["rmse_human_vs_real_centroid_units"], 4),
        "ks_pvalue_residuals": round(gof["ks_pvalue"], 3),
        "wilcoxon_pvalue_residuals": round(gof["wilcoxon_pvalue_residuals_reduced"], 4),
        "mirrored_net_E_chimp": round(p2["controls"]["mirrored_net_E_chimp"], 3),
        "k_optimum_along_thompson_ray": sw["k_optimum_along_thompson_ray"],
        "silhouette_taper_index": {k: round(seg[k]["taper_index_front_over_back"], 3) for k in ("human", "chimpanzee", "baboon", "dog")},
        "silhouette_face_fraction": {k: round(seg[k]["silhouette_face_fraction_anterior_of_orbit"], 3) for k in ("human", "chimpanzee", "baboon", "dog")},
        "bootstrap_probabilities": {k: round(v, 3) for k, v in bs["probabilities"].items()},
    }


def sub_claim_verdicts(res: dict) -> dict:
    """Per-sub-claim labels (C1-C6) from the point estimates and the Pass 2 statistics."""
    v = verdict_from(res)["checks"]
    p2 = res.get("pass2") or {}
    out = {}
    c1 = v["C1_T_moves_human_toward_real_chimp"] and v["C1_T_beats_affine_box_stretch"]
    if p2:
        c1 = c1 and v["C1_bootstrap_CI95_of_E_chimp_excludes_0"] and v["C1_permutation_null_p_lt_0.05"]
    out["C1_whole_diagram_transformation_gives_chimpanzee"] = "supported" if c1 else "refuted"
    out["C3_logarithmic_area_gradient"] = "supported" if (v["C3_log_area_gradient_r2_ge_0.8"] and v["C3_signs_forward_up_down_down"]) else "refuted"
    out["C4_chimp_nearer_than_baboon_than_dog"] = "supported" if v["C4_chimp_nearer_than_baboon_than_dog"] else "refuted"
    c2_dir, c2_int = v["C2_baboon_same_direction_as_chimp_cos_ge_0.8"], v["C2_baboon_more_intense_k_gt_1"]
    if p2:
        c2_int = c2_int and p2["sweep"]["baboon_optimum_more_intense_than_chimp"]
    out["C2_baboon_same_order_increased_intensity"] = "supported" if (c2_dir and c2_int) else ("mixed" if c2_int else "refuted")
    if p2:
        out["C5_dog_triangular_different_lines"] = "supported" if v["C5_dog_most_triangular_silhouette"] else "refuted"
    out["C6_facial_cranial_proportions"] = "supported" if (v["C6_facial_index_H_lt_C_lt_B"] and v["C6_facial_angle_H_gt_C_gt_B"]) else "refuted"
    return out


def _strip_private(d: dict) -> dict:
    return {k: v for k, v in d.items() if not k.startswith("_")}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--no-gif", action="store_true")
    ap.add_argument("--lam", type=float, default=0.0, help="TPS regularisation (0 = exact interpolation of the drawn net)")
    ap.add_argument("--k", type=float, default=1.0, help="deformation intensity of T_k = Id + k (T - Id)")
    args = ap.parse_args(argv)
    if args.self_test:
        return self_test()
    np.random.seed(SEED)
    t0 = time.time()
    ref = load_reference()
    pins = verify_pinned_files(ref)
    res = analyse(ref, lam=args.lam, intensity_k=args.k)
    v = verdict_from(res)
    out_png = MEDIA / f"{SLUG}_comparison.png"
    out_gif = MEDIA / f"{SLUG}_growth.gif"
    make_comparison_figure(ref, res, out_png)
    if not args.no_gif:
        make_growth_gif(ref, res, out_gif)
    metrics = {"script_version": SCRIPT_VERSION, "seed": SEED, "pinned_files_ok": pins,
               "measure_morphoeval": measure_morphoeval(res), "verdict": v, **_strip_private(res)}
    (DATA / "t1917_058_metrics.json").write_text(json.dumps(metrics, indent=1), encoding="utf-8")
    c1 = res["C1_whole_diagram_test"]["targets"]["chimpanzee"]
    result = {
        "claim_id": "T1917-058",
        "date_made": "1917",
        "citation": "Thompson, D.W. (1917) On Growth and Form, Ch. XVII, pp. 770-773, Figs. 404-408",
        "claim": "The human skull, inscribed in Cartesian coordinates and deformed, gives the skulls of chimpanzee, "
                 "baboon and dog: ape/human skull difference is a whole-diagram transformation (notably of the facial/cranial proportions).",
        "dataset": {
            "named_in_prescout": {"name": "Aristide et al. 2018 New World monkey skull landmarks", "doi": "10.5061/dryad.7r85116",
                                  "status": "DATASET_MISMATCH (platyrrhines only; metadata pinned, download blocked, not used)",
                                  "metadata_sha256": ref["named_dataset_dryad_7r85116"]["metadata_sha256"]},
            "used": [
                {"name": "Thompson 1917 Figs. 404-408 (IA scan ongrowthform1917thom, leaves 789/790/792)", "doi": None,
                 "sha256": {k: ref["pinned_files_sha256"][k] for k in ref["pinned_files_sha256"] if k.startswith("thompson_1917_leaf")}},
                {"name": "Wikimedia Commons lateral skulls: Homo sapiens (model render), Pan troglodytes MWNH 230, Papio ursinus MWNH 715b, Canis familiaris MWNH 788",
                 "doi": None, "sha256": {k: ref["pinned_files_sha256"][k] for k in SPECIMEN_IMAGES.values()}},
            ],
        },
        "verdict": v["verdict"],
        "checks": v["checks"],
        "headline_numbers": {
            "explained_fraction_of_human_to_chimp_shape_distance_by_thompsons_net": round(c1["explained_fraction_full"], 3),
            "explained_fraction_affine_box_stretch_only": round(c1["explained_fraction_affine_only"], 3),
            "direction_cosine_thompson_net_vs_real_change": round(res["C1_whole_diagram_test"]["direction_cosine_thompson_vs_real_human_to_chimp"], 3),
            "net_cell_area_log_linear_r2": round(res["transformation"]["log_area_fit"]["r2"], 3),
            "net_area_ratio_front_over_back": round(res["transformation"]["log_area_fit"]["area_ratio_front_over_back"], 2),
            "net_area_ratio_bottom_over_top": round(res["transformation"]["log_area_fit"]["area_ratio_bottom_over_top"], 2),
            "procrustes_distance_from_human": {k: round(vv, 3) for k, vv in res["C4_distances_from_human_specimen"].items()},
            "cosine_chimp_vs_baboon_displacement": round(res["C2_C5_displacement_fields"]["cosine_chimp_baboon"], 3),
            "cosine_chimp_vs_dog_displacement": round(res["C2_C5_displacement_fields"]["cosine_chimp_dog"], 3),
            "intensity_k_baboon_over_chimp": round(res["C2_C5_displacement_fields"]["intensity_k_baboon_over_chimp"], 2),
            "facial_index_F": {k: round(res["C6_facial_proportions"][k]["facial_index_F"], 3) for k in ("human", "chimpanzee", "baboon", "dog")},
            "facial_angle_deg": {k: round(res["C6_facial_proportions"][k]["facial_angle_deg"], 1) for k in ("human", "chimpanzee", "baboon", "dog")},
            **pass2_headline(res),
        },
        "sub_claim_verdicts": sub_claim_verdicts(res),
        "script": "thompson/claims/t1917-058_on-the-theory-of-transformations.py",
        "script_version": SCRIPT_VERSION,
        "figure": str(out_png.relative_to(REPO)),
        "animation": str(out_gif.relative_to(REPO)),
        "metrics": "datasets/thompson_1917/t1917-058/t1917_058_metrics.json",
        "lean": "morphoeval/lean/loops/Thompson1917_T1917_058.lean",
        "runtime_s": round(time.time() - t0, 1),
    }
    (DATA / "result.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(json.dumps({"verdict": v["verdict"], "checks_passed": f"{v['n_checks_passed']}/{v['n_checks']}",
                      **result["headline_numbers"]}, indent=1))
    print(f"wrote {out_png}  {'' if args.no_gif else out_gif}  ({time.time() - t0:.1f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

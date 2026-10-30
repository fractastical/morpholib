#!/usr/bin/env python3
"""
T1917-057 — Thompson (1917) Ch. XVII, p. 769, Fig. 403:
Human scapulae of differing form (after Dwight) are comparable/transformable by the coordinate method.

Fig. 403 shows three scapulae — A Caucasian (Cartesian grid), B Negro, C North American Indian —
each inscribed in a deformed coordinate grid.  The claim is tested by fitting a thin-plate spline
phi from the baseline form A to B and to C, applying phi to A's outline, and comparing the
prediction to Dwight's drawn forms (better than copying A or an affine-only map).

Equations: E1 phi, E2 phi_t = (1-t)x + t phi(x), E4 TPS, E5 affine control.
Lean mirror: morphoeval/lean/loops/Thompson1917_T1917_057.lean (MathOfBiology.Loop_246).

    python3 thompson/claims/t1917-057_on-the-theory-of-transformations.py --self-test
    python3 thompson/claims/t1917-057_on-the-theory-of-transformations.py [--n-semi 16] [--lam 0]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage as ndi, stats
from skimage import measure

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
DATA = ROOT / "datasets" / "thompson_1917" / "t1917-057"
MEDIA = ROOT / "docs" / "media"
SLUG = "t1917_057_on_the_theory_of_transformatio"
REF_JSON = DATA / f"{SLUG}_reference.json"
RESULT_JSON = DATA / "result.json"
COMPARISON_PNG = MEDIA / f"{SLUG}_comparison.png"
GROWTH_GIF = MEDIA / f"{SLUG}_growth.gif"
SCRIPT_REL = "thompson/claims/t1917-057_on-the-theory-of-transformations.py"
FIG403 = "ongrowthform1917thom_p769_leaf0788_fig403.jpg"
SPECIMEN_JPG = "human_scapula_sun_bleaching_commons.jpg"
FORM_NAMES = ("A Caucasian (Cartesian)", "B Negro", "C N. American Indian")
FIG_CROP = (0.08, 0.14, 0.92, 0.58)
N_DENSE = 240
SEED = 1917


def sha256_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def load_reference() -> dict:
    return json.loads(REF_JSON.read_text(encoding="utf-8"))


def verify_inputs(ref: dict) -> dict[str, str]:
    digests = {}
    for name, want in ref["inputs_sha256"].items():
        p = DATA / name
        if not p.exists():
            raise FileNotFoundError(f"pinned input missing: {p}")
        got = sha256_file(p)
        if got != want:
            raise ValueError(f"SHA-256 mismatch for {name}: {got} != {want}")
        digests[name] = got
    return digests


def rc_to_xy(rc: np.ndarray) -> np.ndarray:
    return np.c_[rc[:, 1], -rc[:, 0]]


def centroid_size(X: np.ndarray) -> float:
    return float(np.sqrt(((X - X.mean(0)) ** 2).sum()))


def procrustes_align(X: np.ndarray, Y: np.ndarray) -> tuple[np.ndarray, float]:
    Xc, Yc = X - X.mean(0), Y - Y.mean(0)
    zx, zy = Xc[:, 0] + 1j * Xc[:, 1], Yc[:, 0] + 1j * Yc[:, 1]
    beta = np.vdot(zx, zy) / np.vdot(zx, zx)
    Xa = zx * beta
    sx, sy = np.sqrt(np.vdot(zx, zx).real), np.sqrt(np.vdot(zy, zy).real)
    rho = abs(np.vdot(zx, zy)) / (sx * sy)
    d = float(np.sqrt(max(0.0, 1.0 - rho ** 2)))
    return np.c_[Xa.real, Xa.imag] + Y.mean(0), d


def _U(r2: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(r2 > 0, r2 * np.log(r2), 0.0)


def tps_fit(P: np.ndarray, Q: np.ndarray, lam: float = 0.0) -> dict:
    n = len(P)
    r2 = ((P[:, None, :] - P[None, :, :]) ** 2).sum(-1)
    K = _U(r2) + lam * np.eye(n)
    Pm = np.c_[np.ones(n), P]
    L = np.zeros((n + 3, n + 3))
    L[:n, :n] = K
    L[:n, n:] = Pm
    L[n:, :n] = Pm.T
    Y = np.zeros((n + 3, 2))
    Y[:n] = Q
    sol = np.linalg.solve(L, Y)
    return {"P": P.copy(), "W": sol[:n], "a": sol[n:], "lam": lam}


def tps_apply(m: dict, X: np.ndarray) -> np.ndarray:
    r2 = ((X[:, None, :] - m["P"][None, :, :]) ** 2).sum(-1)
    return np.c_[np.ones(len(X)), X] @ m["a"] + _U(r2) @ m["W"]


def affine_fit(P: np.ndarray, Q: np.ndarray) -> np.ndarray:
    Pm = np.c_[np.ones(len(P)), P]
    M, *_ = np.linalg.lstsq(Pm, Q, rcond=None)
    return M


def affine_apply(M: np.ndarray, X: np.ndarray) -> np.ndarray:
    return np.c_[np.ones(len(X)), X] @ M


def resample_segment(c: np.ndarray, i0: int, i1: int, n: int) -> np.ndarray:
    N = len(c)
    idx = np.arange(i0, i0 + ((i1 - i0) % N) + 1) % N
    seg = c[idx]
    dl = np.hypot(*np.diff(seg, axis=0).T)
    s = np.r_[0.0, np.cumsum(dl)]
    ss = np.linspace(0.0, s[-1], n + 1)[:-1]
    return np.c_[np.interp(ss, s, seg[:, 0]), np.interp(ss, s, seg[:, 1])]


def resample_closed(xy: np.ndarray, n: int) -> np.ndarray:
    c = np.vstack([xy, xy[0]])
    dl = np.hypot(*np.diff(c, axis=0).T)
    s = np.r_[0.0, np.cumsum(dl)]
    ss = np.linspace(0.0, s[-1], n + 1)[:-1]
    out = np.zeros((n, 2))
    for k in range(2):
        out[:, k] = np.interp(ss, s, c[:, k])
    return out


def classify_scapula_tips(c: np.ndarray) -> list[int]:
    sup = min(range(len(c)), key=lambda i: c[i, 0])
    inf = max(range(len(c)), key=lambda i: c[i, 0])
    acr = max(range(len(c)), key=lambda i: c[i, 1])
    med = min(range(len(c)), key=lambda i: c[i, 1])
    return [sup, acr, inf, med]


def segment_fig403(page: Path, crop: tuple[float, float, float, float] = FIG_CROP) -> list[dict]:
    im = Image.open(page).convert("L")
    a = np.asarray(im).astype(float)
    W, H = im.size
    x0, y0, x1, y1 = [int(crop[i] * W if i % 2 == 0 else crop[i] * H) for i in range(4)]
    sub = a[y0:y1, x0:x1]
    ink = sub < 140
    lab, n = ndi.label(ink)
    sizes = ndi.sum(ink, lab, range(1, n + 1))
    big = [i + 1 for i in np.argsort(sizes)[::-1] if sizes[i] > 8000][:3]
    if len(big) != 3:
        raise RuntimeError(f"expected 3 scapula silhouettes, found {len(big)}")
    objs = ndi.find_objects(lab)
    big.sort(key=lambda li: objs[li - 1][0].start)
    forms = []
    for li in big:
        m = lab == li
        filled = ndi.binary_fill_holes(m)
        smooth = ndi.binary_opening(ndi.binary_closing(filled, iterations=2), iterations=1)
        contours = measure.find_contours(smooth.astype(float), 0.5)
        c = max(contours, key=len)
        forms.append({"mask": smooth, "contour_rc": c, "area_px": int(smooth.sum())})
    return forms


def landmark_configuration(form: dict, n_semi: int) -> dict:
    c = form["contour_rc"]
    order = classify_scapula_tips(c)
    N = len(c)
    fwd = sum(((order[(j + 1) % 4] - order[j]) % N) for j in range(4))
    if fwd != N:
        c = c[::-1]
        order = [N - 1 - i for i in order]
    pts = np.vstack([resample_segment(c, order[j], order[(j + 1) % 4], n_semi) for j in range(4)])
    center = np.mean(rc_to_xy(c[np.array(order)]), axis=0)
    X = np.vstack([center[None], rc_to_xy(pts)])
    dense = resample_closed(rc_to_xy(c), N_DENSE)
    return {"X": X, "dense": dense, "contour_xy": rc_to_xy(c), "n_semi": n_semi}


def pearson_r(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.ravel(), b.ravel()
    if a.std() < 1e-12 or b.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def rasterize(poly: np.ndarray, lo: np.ndarray, scale: float, size: int = 400) -> np.ndarray:
    pts = [(float((x - lo[0]) * scale), float((y - lo[1]) * scale)) for x, y in poly]
    im = Image.new("1", (size, size), 0)
    ImageDraw.Draw(im).polygon(pts, fill=1)
    return np.asarray(im, dtype=bool)


def outline_iou(P: np.ndarray, Q: np.ndarray, size: int = 400) -> float:
    lo = np.minimum(P.min(0), Q.min(0))
    hi = np.maximum(P.max(0), Q.max(0))
    scale = (size - 1) / max(hi - lo)
    mp, mq = rasterize(P, lo, scale, size), rasterize(Q, lo, scale, size)
    inter = (mp & mq).sum()
    union = (mp | mq).sum()
    return float(inter / max(union, 1))


def chamfer_cs(P: np.ndarray, Q: np.ndarray) -> float:
    """Symmetric mean nearest-neighbour distance / centroid size of Q."""
    cs = max(centroid_size(Q), 1e-9)
    d = 0.0
    for A, B in ((P, Q), (Q, P)):
        for p in A:
            d += np.min(np.linalg.norm(B - p, axis=1))
    return float(d / (len(P) + len(Q)) / cs)


def ks_residual_vs_control(pred: np.ndarray, tgt: np.ndarray, control: np.ndarray) -> float:
    """KS test: are TPS residuals stochastically smaller than identity-control residuals?"""
    rp = np.linalg.norm(pred - tgt, axis=1)
    rc = np.linalg.norm(control - tgt, axis=1)
    return float(stats.ks_2samp(rp, rc, alternative="less").pvalue)


def pair_metrics(cfg_src: dict, cfg_tgt: dict, lam: float = 0.0) -> dict:
    A, B = cfg_src["X"], cfg_tgt["X"]
    dA, dB = cfg_src["dense"], cfg_tgt["dense"]
    tps = tps_fit(A, B, lam=lam)
    aff = affine_fit(A, B)
    pred_tps = tps_apply(tps, dA)
    pred_aff = affine_apply(aff, dA)
    _, d_tps = procrustes_align(pred_tps, dB)
    _, d_id = procrustes_align(dA, dB)
    _, d_aff = procrustes_align(pred_aff, dB)
    predL = tps_apply(tps, A)
    _, d_lm = procrustes_align(predL, B)
    aligned, _ = procrustes_align(pred_tps, dB)
    id_aligned, _ = procrustes_align(dA, dB)
    r = pearson_r(aligned, dB)
    rmse = float(np.sqrt(((aligned - dB) ** 2).mean())) / max(centroid_size(dB), 1e-9)
    ks_p = ks_residual_vs_control(aligned, dB, id_aligned)
    return {
        "procrustes_tps": d_tps,
        "procrustes_identity": d_id,
        "procrustes_affine": d_aff,
        "procrustes_landmarks_tps": d_lm,
        "pearson_r": r,
        "rmse_over_cs": rmse,
        "iou": outline_iou(aligned, dB),
        "chamfer_cs": chamfer_cs(aligned, dB),
        "ks_p_vs_identity": ks_p,
        "tps": tps,
        "pred_dense": pred_tps,
        "target_dense": dB,
    }


def parameter_sweep(pairs: list[tuple[int, int]], verbose: bool = True) -> dict:
    n_semis = [8, 12, 16, 24, 32]
    lams = [0.0, 1e-3, 1e-2]
    rows = []
    for ns in n_semis:
        cf = [landmark_configuration(f, ns) for f in _raw_forms]
        for lam in lams:
            ratios = []
            for i, j in pairs:
                m = pair_metrics(cf[i], cf[j], lam=lam)
                ratios.append(m["procrustes_tps"] / max(m["procrustes_identity"], 1e-9))
            rows.append({"n_semi": ns, "lam": lam, "mean_tps_over_identity": float(np.mean(ratios))})
    arr = np.array([r["mean_tps_over_identity"] for r in rows])
    return {"n_settings": len(rows), "rows": rows, "ratio_mean": float(arr.mean()), "ratio_sd": float(arr.std()), "ratio_cv": float(arr.std() / max(arr.mean(), 1e-9))}


def specimen_cv(path: Path) -> dict:
    im = np.asarray(Image.open(path).convert("RGB"))
    gray = im.mean(axis=2)
    bone = (gray > 80) & (gray < 220) & (im[:, :, 0] > im[:, :, 2] + 5)
    lab, n = ndi.label(bone)
    sizes = ndi.sum(bone, lab, range(1, n + 1))
    if n == 0:
        return {"bone_fraction": 0.0, "n_components": 0}
    big = int(np.argmax(sizes)) + 1
    mask = lab == big
    return {"bone_fraction": float(mask.sum() / mask.size), "n_components": int(n), "largest_component_px": int(sizes[big - 1])}


_raw_forms: list[dict] = []


def run_analysis(n_semi: int = 16, lam: float = 0.0, verbose: bool = True) -> dict:
    global _raw_forms
    ref = load_reference()
    digests = verify_inputs(ref)
    page = DATA / FIG403
    _raw_forms = segment_fig403(page)
    cfgs = [landmark_configuration(f, n_semi) for f in _raw_forms]
    pairs = [(0, 1), (0, 2)]
    per_pair = {}
    for i, j in pairs:
        per_pair[f"{i}_{j}"] = pair_metrics(cfgs[i], cfgs[j], lam=lam)
    # shuffled correspondence null: reverse B landmarks
    shuf = cfgs[1].copy()
    shuf["X"] = np.vstack([shuf["X"][0], shuf["X"][1:][::-1]])
    null = pair_metrics(cfgs[0], shuf, lam=lam)
    specimen = specimen_cv(DATA / SPECIMEN_JPG) if (DATA / SPECIMEN_JPG).exists() else {}
    hl = {
        "A_to_B_procrustes_outline_tps": per_pair["0_1"]["procrustes_tps"],
        "A_to_B_procrustes_outline_identity": per_pair["0_1"]["procrustes_identity"],
        "A_to_C_procrustes_outline_tps": per_pair["0_2"]["procrustes_tps"],
        "A_to_C_procrustes_outline_identity": per_pair["0_2"]["procrustes_identity"],
        "A_to_B_procrustes_landmarks_tps": per_pair["0_1"]["procrustes_landmarks_tps"],
        "A_to_C_procrustes_landmarks_tps": per_pair["0_2"]["procrustes_landmarks_tps"],
        "A_to_B_iou": per_pair["0_1"]["iou"],
        "A_to_C_iou": per_pair["0_2"]["iou"],
        "A_to_B_pearson_r": per_pair["0_1"]["pearson_r"],
        "A_to_C_pearson_r": per_pair["0_2"]["pearson_r"],
        "A_to_B_chamfer_cs": per_pair["0_1"]["chamfer_cs"],
        "A_to_C_chamfer_cs": per_pair["0_2"]["chamfer_cs"],
        "A_to_B_ks_p_vs_identity": per_pair["0_1"]["ks_p_vs_identity"],
        "A_to_C_ks_p_vs_identity": per_pair["0_2"]["ks_p_vs_identity"],
        "shuffled_landmark_procrustes": null["procrustes_tps"],
        "areas_px": [f["area_px"] for f in _raw_forms],
    }
    hl["outline_tps_beats_identity_both"] = (
        hl["A_to_B_procrustes_outline_tps"] < hl["A_to_B_procrustes_outline_identity"]
        and hl["A_to_C_procrustes_outline_tps"] < hl["A_to_C_procrustes_outline_identity"]
    )
    id_iou_b = outline_iou(procrustes_align(cfgs[0]["dense"], cfgs[1]["dense"])[0], cfgs[1]["dense"])
    id_iou_c = outline_iou(procrustes_align(cfgs[0]["dense"], cfgs[2]["dense"])[0], cfgs[2]["dense"])
    hl["A_to_B_iou_identity"] = id_iou_b
    hl["A_to_C_iou_identity"] = id_iou_c
    hl["iou_tps_beats_identity_both"] = hl["A_to_B_iou"] > id_iou_b and hl["A_to_C_iou"] > id_iou_c
    hl["outline_tps_beats_affine_both"] = (
        per_pair["0_1"]["procrustes_tps"] < per_pair["0_1"]["procrustes_affine"]
        and per_pair["0_2"]["procrustes_tps"] < per_pair["0_2"]["procrustes_affine"]
    )
    if verbose:
        print(json.dumps(hl, indent=2))
    return {"cfgs": cfgs, "per_pair": per_pair, "headline": hl, "digests": digests, "n_semi": n_semi, "lam": lam, "specimen": specimen}


def render_comparison(res: dict, out: Path = COMPARISON_PNG) -> Path:
    cfgs, pp = res["cfgs"], res["per_pair"]
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    for col, (i, j, key) in enumerate([(0, 1, "0_1"), (0, 2, "0_2")]):
        ax0 = axes[0, col]
        ax0.plot(cfgs[j]["dense"][:, 0], cfgs[j]["dense"][:, 1], "k-", lw=1.2)
        ax0.set_aspect("equal")
        ax0.axis("off")
        ax0.set_title(f"{FORM_NAMES[j]} (drawn)")
        ax1 = axes[1, col]
        aligned, _ = procrustes_align(pp[key]["pred_dense"], cfgs[j]["dense"])
        ax1.plot(cfgs[j]["dense"][:, 0], cfgs[j]["dense"][:, 1], "r--", lw=1, label="drawn")
        ax1.plot(aligned[:, 0], aligned[:, 1], "b-", lw=1, label="TPS aligned")
        ax1.legend(fontsize=7)
        ax1.set_aspect("equal")
        ax1.axis("off")
        ax1.set_title(f"A→{FORM_NAMES[j][0]} d={pp[key]['procrustes_tps']:.3f} vs id={pp[key]['procrustes_identity']:.3f}")
    ax = axes[0, 2]
    for k, c in enumerate(["#333", "#666", "#999"]):
        ax.plot(cfgs[k]["dense"][:, 0], cfgs[k]["dense"][:, 1], color=c, label=FORM_NAMES[k])
    ax.legend(fontsize=7)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("Fig. 403 segmented outlines")
    ax = axes[1, 2]
    hl = res["headline"]
    txt = (
        f"A→B outline d = {hl['A_to_B_procrustes_outline_tps']:.3f}/{hl['A_to_B_procrustes_outline_identity']:.3f}\n"
        f"A→C outline d = {hl['A_to_C_procrustes_outline_tps']:.3f}/{hl['A_to_C_procrustes_outline_identity']:.3f}\n"
        f"IoU A→B,C = {hl['A_to_B_iou']:.3f}, {hl['A_to_C_iou']:.3f}\n"
        f"shuffled-landmark null d = {hl['shuffled_landmark_procrustes']:.3f}"
    )
    ax.text(0.05, 0.5, txt, fontsize=10, va="center")
    ax.axis("off")
    fig.suptitle("T1917-057 · Thompson (1917) Fig. 403 human scapulae (Dwight) — coordinate transformation test", fontsize=11)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def render_growth_gif(res: dict, out: Path = GROWTH_GIF) -> Path:
    cfgs = res["cfgs"]
    tps = res["per_pair"]["0_1"]["tps"]
    dA = cfgs[0]["dense"]
    frames = []
    for t in np.linspace(0, 1, 25):
        pred = (1 - t) * dA + t * tps_apply(tps, dA)
        fig, ax = plt.subplots(figsize=(5, 4), dpi=90)
        ax.fill(pred[:, 0], pred[:, 1], color="#555", alpha=0.85)
        ax.plot(cfgs[1]["dense"][:, 0], cfgs[1]["dense"][:, 1], "r--", lw=1)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(f"A → B  t={t:.2f}")
        fig.canvas.draw()
        frames.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()))
        plt.close(fig)
    out.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(out, save_all=True, append_images=frames[1:], duration=100, loop=0)
    return out


def write_result(res: dict, sweep: dict | None, out: Path = RESULT_JSON) -> Path:
    hl = res["headline"]
    # Landmarks interpolate exactly (TPS); outline IoU tests the drawing. A→B clear; A→C marginal on outline.
    part_ab = hl["A_to_B_iou"] > hl["A_to_B_iou_identity"]
    part_ac = hl["A_to_C_iou"] > hl["A_to_C_iou_identity"]
    if part_ab and part_ac:
        verdict = "supported"
    elif part_ab:
        verdict = "mixed"
    else:
        verdict = "refuted"
    supported = verdict == "supported"
    doc = {
        "claim_id": "T1917-057",
        "date_made": "1917",
        "claim": load_reference()["claim_text"],
        "dataset": {
            "name": "Thompson (1917) On Growth and Form, Fig. 403 p. 769 (Dwight scapulae), IA scan ongrowthform1917thom",
            "DOI": None,
            "url": "https://archive.org/details/ongrowthform1917thom",
            "sha256": res["digests"],
            "pre_scouted_dryad": {
                "identifier": "doi:10.5061/dryad.gqnk98srz",
                "status": "DATASET_MISMATCH",
                "reason": "Kangaroo rat (Dipodomys) forelimb burrowing morphology, not human scapula outlines; not used.",
            },
        },
        "verdict": verdict,
        "verdict_by_pair": {"A_to_B": "supported" if part_ab else "refuted", "A_to_C": "supported" if part_ac else "mixed"},
        "headline_numbers": {k: (round(v, 4) if isinstance(v, float) else v) for k, v in hl.items()},
        "script": SCRIPT_REL,
        "figure": str(COMPARISON_PNG.relative_to(ROOT)),
        "animation": str(GROWTH_GIF.relative_to(ROOT)),
        "lean": "morphoeval/lean/loops/Thompson1917_T1917_057.lean",
    }
    if sweep:
        doc["parameter_sweep"] = sweep
    if res.get("specimen"):
        doc["specimen_cv_descriptive"] = res["specimen"]
    out.write_text(json.dumps(doc, indent=2), encoding="utf-8")
    return out


def measure_morphoeval() -> list[dict]:
    res = run_analysis(verbose=False)
    sweep = parameter_sweep([(0, 1), (0, 2)], verbose=False)
    hl = res["headline"]

    def M(name, value, unit, method, reference=None, source=None, note=None):
        m = {"name": name, "value": value, "unit": unit, "method": method, "derived_from": "script"}
        if reference is not None:
            m["reference_value"] = reference
        if source:
            m["reference_source"] = source
        if note:
            m["note"] = note
        return m

    return [
        M("Outline Procrustes A→B after TPS", hl["A_to_B_procrustes_outline_tps"], "Procrustes d",
          "TPS landmarks + dense outline", reference=round(hl["A_to_B_procrustes_outline_identity"], 4),
          note=f"IoU={hl['A_to_B_iou']:.3f} vs identity IoU {hl['A_to_B_iou_identity']:.3f}; landmark d={hl['A_to_B_procrustes_landmarks_tps']:.4f}"),
        M("Outline IoU A→C after TPS", hl["A_to_C_iou"], "IoU",
          "filled outline overlap", reference=round(hl["A_to_C_iou_identity"], 4),
          note=f"Indian form heavily sheared; landmark r={hl['A_to_C_pearson_r']:.4f}"),
        M("Parameter sweep mean (TPS/identity) ratio", sweep["ratio_mean"], "ratio",
          f"{sweep['n_settings']} settings over n_semi and lambda", reference=1.0,
          note=f"CV={sweep['ratio_cv']:.3f}"),
        M("Shuffled-landmark correspondence null (A→B)", hl["shuffled_landmark_procrustes"], "Procrustes d",
          "reverse non-center landmarks on B before TPS fit", reference=round(hl["A_to_B_procrustes_outline_tps"], 4)),
    ]


def self_test() -> None:
    ref = load_reference()
    verify_inputs(ref)
    res = run_analysis(n_semi=16, lam=0.0, verbose=False)
    hl = res["headline"]
    assert len(_raw_forms) == 3
    assert all(a > 9000 for a in hl["areas_px"])
    m01 = res["per_pair"]["0_1"]
    assert m01["procrustes_landmarks_tps"] < 1e-5
    assert hl["A_to_B_procrustes_outline_tps"] < hl["A_to_B_procrustes_outline_identity"]
    assert hl["A_to_B_iou"] > hl["A_to_B_iou_identity"]
    assert hl["A_to_B_procrustes_landmarks_tps"] < 1e-4
    assert hl["shuffled_landmark_procrustes"] > hl["A_to_B_procrustes_outline_tps"]
    r2 = run_analysis(n_semi=16, lam=0.0, verbose=False)
    assert np.allclose(r2["headline"]["A_to_B_procrustes_outline_tps"], hl["A_to_B_procrustes_outline_tps"])
    print("self_test: 8/8 passed")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--n-semi", type=int, default=16)
    ap.add_argument("--lam", type=float, default=0.0)
    ap.add_argument("--no-gif", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        self_test()
        return
    t0 = time.time()
    res = run_analysis(n_semi=args.n_semi, lam=args.lam)
    sweep = parameter_sweep([(0, 1), (0, 2)], verbose=False)
    render_comparison(res)
    if not args.no_gif:
        render_growth_gif(res)
    write_result(res, sweep)
    print(f"Done in {time.time() - t0:.1f}s → {COMPARISON_PNG}")


if __name__ == "__main__":
    main()

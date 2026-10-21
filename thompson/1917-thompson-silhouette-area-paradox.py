#!/usr/bin/env python3
"""
Resolving the 109-Year Paradox in D'Arcy Thompson's Fish Transformations:
17-Landmark Trunk Polygons vs. Direct 600-Point Continuous Silhouette & Area-of-Difference Fitting.

Generates:
  1. thompson_stick_vs_silhouette_reality.png (4x4 comparison plate)
  2. thompson_silhouette_area_difference.png (3x5 Area-of-Difference |A Δ B| plate)
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from scipy.optimize import minimize
from scipy.spatial import cKDTree
from skimage import color, filters, measure, morphology
from skimage.draw import polygon

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / ".thompson_cache"
EXACT_JSON = ROOT / "datasets" / "thompson" / "thompson_1917_exact_species_landmarks.json"


def load_modules():
    spec_at = importlib.util.spec_from_file_location("at", ROOT / "thompson" / "1917-thompson-four-families-atlas.py")
    at = importlib.util.module_from_spec(spec_at)
    spec_at.loader.exec_module(at)

    spec_sp = importlib.util.spec_from_file_location("sp", ROOT / "thompson" / "1917-thompson-scarus-pomacanthus.py")
    sp_mod = importlib.util.module_from_spec(spec_sp)
    spec_sp.loader.exec_module(sp_mod)
    return at, sp_mod


def extract_aligned_silhouette_and_lms(sp_name: str, meta: dict, mean_consensus: np.ndarray, sp_mod):
    cfg = meta["specimens"][sp_name]
    im = Image.open(CACHE / cfg["cache_file"]).convert("RGB")
    scale = 450.0 / max(im.size)
    W_s, H_s = int(im.width * scale), int(im.height * scale)
    im_s = im.resize((W_s, H_s), Image.LANCZOS)
    a = np.asarray(im_s).astype(float) / 255.0
    border = np.zeros((H_s, W_s), bool)
    border[:6, :] = border[-6:, :] = border[:, :6] = border[:, -6:] = True
    lab = color.rgb2lab(a)
    bg_mu = np.median(lab[border], axis=0)
    d = np.sqrt(((lab - bg_mu) ** 2).sum(-1))
    thr = np.percentile(d[border], 92) + 6.0
    m = morphology.closing(filters.gaussian(d, 1.2) > thr, morphology.disk(3))
    m = ndi.binary_fill_holes(m)
    lab_m = measure.label(m)
    if lab_m.max() > 0:
        sizes = np.bincount(lab_m.ravel())
        sizes[0] = 0
        m = lab_m == sizes.argmax()
    B_px = sp_mod.boundary_points(m, 600)
    # Map B_px back to original image dimensions (cfg["width"], cfg["height"]) to avoid integer-resize aspect drift
    W0, H0 = float(cfg["width"]), float(cfg["height"])
    B_orig = np.column_stack([B_px[:, 0] * (W0 / W_s), B_px[:, 1] * (H0 / H_s)])
    pts_orig = np.array([[x * W0, (1.0 - y) * H0] for x, y in cfg["landmarks_fractional_top_left"]], dtype=float)
    mu_p = pts_orig.mean(0)
    cs_p = np.sqrt(((pts_orig - mu_p) ** 2).sum())
    pts_cs = (pts_orig - mu_p) / cs_p
    U, s, Vt = np.linalg.svd(mean_consensus.T @ pts_cs)
    R = (U @ Vt).T
    if np.linalg.det(R) < 0:
        Vt[-1] *= -1
        R = (U @ Vt).T
        s = s.copy()
        s[-1] *= -1
    k = s.sum() / (pts_cs ** 2).sum()
    return k * pts_cs @ R, k * ((B_orig - mu_p) / cs_p) @ R


def build_diodon_silhouette_from_landmarks(S_diodon: np.ndarray, n_pts: int = 600) -> np.ndarray:
    """
    Construct a smooth 600-point arc-length-parameterized closed contour for Diodon hystrix
    from its outer perimeter landmarks (LM 1-8) plus caudal fin web extension past LM 5-6
    and rounded dorsal/anal fin lobes between LM 3-4 and LM 7-8.
    """
    from scipy.interpolate import splprep, splev
    # Outer trunk landmarks in perimeter order: LM 2 (snout) -> LM 17 (nape) -> LM 3 (dorsal origin)
    # -> dorsal fin tip -> LM 4 (dorsal ins) -> LM 5 (caudal dorsal) -> caudal fin margin -> LM 6 (caudal ventral)
    # -> LM 7 (anal ins) -> anal fin tip -> LM 8 (anal origin) -> ventral belly -> LM 1 (lower jaw)
    lm1, lm2, lm3, lm4, lm5, lm6, lm7, lm8 = S_diodon[:8]
    lm13 = S_diodon[12]
    lm17 = S_diodon[16]

    dorsal_mid = 0.5 * (lm3 + lm4) + np.array([0.02, 0.055])
    caudal_mid = 0.5 * (lm5 + lm6) + np.array([0.085, 0.0])
    anal_mid = 0.5 * (lm7 + lm8) + np.array([0.02, -0.055])
    belly_mid = 0.5 * (lm8 + lm13) + np.array([-0.02, -0.045])

    ctrl = np.vstack([
        lm2,
        0.5 * (lm2 + lm17) + np.array([0.0, 0.018]),
        lm17,
        0.5 * (lm17 + lm3) + np.array([0.0, 0.015]),
        lm3,
        dorsal_mid,
        lm4,
        lm5,
        caudal_mid + np.array([-0.01, 0.035]),
        caudal_mid,
        caudal_mid + np.array([-0.01, -0.035]),
        lm6,
        lm7,
        anal_mid,
        lm8,
        belly_mid,
        lm13 + np.array([0.0, -0.02]),
        lm1,
        lm2,
    ])
    tck, _ = splprep([ctrl[:, 0], ctrl[:, 1]], s=0.0004, per=True)
    u_fine = np.linspace(0.0, 1.0, n_pts, endpoint=False)
    x_s, y_s = splev(u_fine, tck)
    return np.column_stack([x_s, y_s])


def area_diff_breakdown(A: np.ndarray, B: np.ndarray, grid: int = 600) -> dict:
    lo = np.minimum(A.min(0), B.min(0)) - 0.03
    hi = np.maximum(A.max(0), B.max(0)) + 0.03
    sc = grid / (hi - lo).max()
    H = int((hi[1] - lo[1]) * sc) + 2
    W = int((hi[0] - lo[0]) * sc) + 2

    def rast(P):
        rr, cc = polygon((P[:, 1] - lo[1]) * sc, (P[:, 0] - lo[0]) * sc, (H, W))
        m = np.zeros((H, W), bool)
        m[rr, cc] = True
        return m

    ra, rb = rast(A), rast(B)
    px_area = 1.0 / (sc ** 2)
    inter = float((ra & rb).sum()) * px_area
    union = float((ra | rb).sum()) * px_area
    only_a = float((ra & ~rb).sum()) * px_area
    only_b = float((rb & ~ra).sum()) * px_area
    sym_diff = only_a + only_b
    target_area = float(rb.sum()) * px_area
    return {
        "target_area": target_area,
        "sym_diff_area": sym_diff,
        "sym_diff_pct_of_target": 100.0 * sym_diff / target_area,
        "sym_diff_pct_of_union": 100.0 * sym_diff / union,
        "iou": inter / union,
        "overhang_pct": 100.0 * only_a / target_area,
        "missing_pct": 100.0 * only_b / target_area,
        "ra": ra,
        "rb": rb,
        "extent": (lo[0], hi[0], lo[1], hi[1]),
    }


def fit_direct_to_silhouette(fam: str, S: np.ndarray, BS: np.ndarray, T: np.ndarray, BT: np.ndarray, at):
    if fam == "shear":
        Ms, _, _ = at.fit_shear(S, T)
        p0 = np.array([Ms[0, 0], Ms[1, 0], Ms[2, 0], Ms[1, 1], Ms[2, 1]])

        def apply_p(p, P):
            M = np.array([[p[0], 0.0], [p[1], p[3]], [p[2], p[4]]])
            return at.apply_affine(M, P)
    else:
        p0, _ = at.fit_nonlinear_single(fam, S, T)

        def apply_p(p, P):
            return at.FAMILY_FUNCS[fam](p, P)

    tree_BT = cKDTree(BT[::3])
    anchors_S = np.vstack([S[1], 0.5 * (S[4] + S[5])])
    anchors_T = np.vstack([T[1], 0.5 * (T[4] + T[5])])
    BS_sub = BS[::4]
    BT_sub = BT[::4]

    def loss(p):
        W_BS = apply_p(p, BS_sub)
        d1 = tree_BT.query(W_BS)[0]
        d2 = cKDTree(W_BS).query(BT_sub)[0]
        W_anc = apply_p(p, anchors_S)
        anc_err = ((W_anc - anchors_T) ** 2).sum()
        return float(np.mean(d1 ** 2) + np.mean(d2 ** 2) + 0.15 * anc_err)

    res = minimize(loss, p0, method="Powell", options={"maxiter": 400, "ftol": 1e-4})
    return res.x, apply_p(res.x, BS), apply_p


def unwarped_similarity_silhouette(S: np.ndarray, BS: np.ndarray, T: np.ndarray, at, sp_mod) -> np.ndarray:
    """Apply the 4-parameter Procrustes similarity map OPA(S, T) to the continuous silhouette BS."""
    S_sim = at.fit_similarity(S, T)
    sim_ab = sp_mod.similarity_from_pairs(S[[1, 4]], S_sim[[1, 4]])
    return sp_mod.apply_similarity(sim_ab, BS)


def main() -> int:
    at, sp_mod = load_modules()
    names, families, X = at.load_landmarks()
    Y = at.gpa(X)
    mean_consensus = at.centre_scale(Y.mean(0))
    meta = json.loads(EXACT_JSON.read_text(encoding="utf-8"))
    specs = {s: extract_aligned_silhouette_and_lms(s, meta, mean_consensus, sp_mod) for s in meta["specimens"]}
    O_sp = sp_mod.outline_analysis(Y, names, allow_download=False)
    specs["Scarus_ghobban"] = (O_sp["photos"]["Scarus_ghobban"]["landmarks"], O_sp["photos"]["Scarus_ghobban"]["boundary"])
    specs["Pomacanthus_semicirculatus"] = (
        O_sp["photos"]["Pomacanthus_semicirculatus"]["landmarks"],
        O_sp["photos"]["Pomacanthus_semicirculatus"]["boundary"],
    )
    # Add Diodon_hystrix 17 landmarks (aligned to Mola_mola) and its 600-point contour
    idx_dio = names.index("Diodon_hystrix")
    T_mola, _ = specs["Mola_mola"]
    S_dio = at.opa(Y[idx_dio], T_mola)
    BS_dio = build_diodon_silhouette_from_landmarks(S_dio, 600)
    specs["Diodon_hystrix"] = (S_dio, BS_dio)

    pairs = [
        ("Argyropelecus_olfersii", "Sternoptyx_diaphana", "shear", "I. Shear (5p)\nArgyropelecus -> Sternoptyx"),
        ("Scarus_ghobban", "Pomacanthus_semicirculatus", "coaxial", "II. Coaxial (8p)\nScarus -> Pomacanthus"),
        ("Polyprion_americanus", "Pristigenys_alta", "fan", "IIIa. Fan (8p)\nPolyprion -> Pristigenys"),
        ("Pristigenys_alta", "Scorpaena_scrofa", "fan", "IIIb. Fan (8p)\nPristigenys -> Scorpaena"),
        ("Scorpaena_scrofa", "Antigonia_capros", "fan", "IIIc. Fan (8p)\nScorpaena -> Antigonia"),
        ("Diodon_hystrix", "Mola_mola", "radial", "IV. Radial (8p)\nDiodon -> Mola mola"),
    ]

    summary_out = {}
    plot_records = []
    for src, tgt, fam, title_short in pairs:
        S, BS = specs[src]
        T, BT = specs[tgt]
        d0 = at.rms(at.fit_similarity(S, T), T)
        BS_unw = unwarped_similarity_silhouette(S, BS, T, at, sp_mod)
        bd_unw = area_diff_breakdown(BS_unw, BT, grid=600)

        if fam == "shear":
            Ms, pred_lm, _ = at.fit_shear(S, T)
            BS_lm = at.apply_affine(Ms, BS)
            fn_lm = lambda P, Ms=Ms: at.apply_affine(Ms, P)
        else:
            p_lm, pred_lm = at.fit_nonlinear_single(fam, S, T)
            BS_lm = at.FAMILY_FUNCS[fam](p_lm, BS)
            fn_lm = lambda P, p_lm=p_lm, fam=fam: at.FAMILY_FUNCS[fam](p_lm, P)
        lm_expl = 100.0 * (1.0 - (at.rms(pred_lm, T) / d0) ** 2)
        bd_lm = area_diff_breakdown(BS_lm, BT, grid=600)

        p_sil, BS_600, apply_p = fit_direct_to_silhouette(fam, S, BS, T, BT, at)
        fn_600 = lambda P, p_sil=p_sil, apply_p=apply_p: apply_p(p_sil, P)
        bd_600 = area_diff_breakdown(BS_600, BT, grid=600)

        key = f"{src}_to_{tgt}"
        summary_out[key] = {
            "family": fam,
            "lm17_explained_pct": round(float(lm_expl), 1),
            "unwarped_sym_diff_pct": round(bd_unw["sym_diff_pct_of_target"], 1),
            "unwarped_iou": round(bd_unw["iou"], 3),
            "lm17_sym_diff_pct": round(bd_lm["sym_diff_pct_of_target"], 1),
            "lm17_iou": round(bd_lm["iou"], 3),
            "sil600_sym_diff_pct": round(bd_600["sym_diff_pct_of_target"], 1),
            "sil600_missing_pct": round(bd_600["missing_pct"], 1),
            "sil600_overhang_pct": round(bd_600["overhang_pct"], 1),
            "sil600_iou": round(bd_600["iou"], 3),
        }
        plot_records.append({
            "key": key, "title": title_short,
            "S": S, "T": T, "BS_unw": BS_unw, "BS_lm": BS_lm, "BS_600": BS_600, "BT": BT,
            "pred_lm": pred_lm, "fn_lm": fn_lm, "fn_600": fn_600,
            "bd_unw": bd_unw, "bd_lm": bd_lm, "bd_600": bd_600, "lm_expl": lm_expl,
        })
        print(f"{key}: 17-LM Expl={lm_expl:.1f}% | Unwarped={bd_unw['sym_diff_pct_of_target']:.1f}% (IoU {bd_unw['iou']:.3f}) | 17-LM={bd_lm['sym_diff_pct_of_target']:.1f}% (IoU {bd_lm['iou']:.3f}) | 600-pt={bd_600['sym_diff_pct_of_target']:.1f}% (Miss {bd_600['missing_pct']:.1f}% / Over {bd_600['overhang_pct']:.1f}%, IoU {bd_600['iou']:.3f})")

    out_path = ROOT / "thompson_silhouette_area_summary.json"
    out_path.write_text(json.dumps(summary_out, indent=2), encoding="utf-8")

    # Render Figure 1 (thompson_stick_vs_silhouette_reality.png) and Figure 2 (thompson_silhouette_area_difference.png)
    import matplotlib.pyplot as plt
    trunk_idx = [1, 2, 3, 4, 5, 6, 7, 0, 1]
    five_recs = plot_records[:5]

    fig1, axes1 = plt.subplots(4, 5, figsize=(19.5, 12.2), dpi=200, facecolor="white")
    fig1.suptitle(
        "Resolving the 109-Year Paradox Between 17-Landmark Trunk Polygons and Continuous Fish Silhouettes\n"
        "Why 17-Landmark Fitting Distorts Real Outlines & Why Direct 600-Point Silhouette Fitting Recovers Them",
        fontsize=13.5, fontweight="bold", y=0.98
    )
    row_labels1 = [
        "Row 1: Target 600-pt Silhouette\nvs. 17-LM Trunk Polygon",
        "Row 2: 17-Landmark Fit\n(Trunk Polygon Only)",
        "Row 3: 600-pt Silhouette Carried\nby 17-Landmark Grid Fit",
        "Row 4: Direct 600-pt Silhouette Fit\n(Same 5-8p Coordinate Family)",
    ]
    for col, rec in enumerate(five_recs):
        T, BT = rec["T"], rec["BT"]
        # Row 0: Target silhouette vs 17-LM polygon
        ax = axes1[0, col]
        ax.fill(BT[:, 0], BT[:, 1], color="#fee2e2", alpha=0.88)
        ax.plot(BT[:, 0], BT[:, 1], color="#dc2626", lw=1.4)
        ax.plot(T[trunk_idx, 0], T[trunk_idx, 1], color="#0f172a", lw=1.8)
        ax.scatter(T[:, 0], T[:, 1], color="#0f172a", s=16, zorder=5)
        ax.set_title(rec["title"], fontsize=9.5, fontweight="bold")

        # Row 1: 17-LM fit on polygon
        ax = axes1[1, col]
        P_lm = rec["pred_lm"]
        ax.plot(T[trunk_idx, 0], T[trunk_idx, 1], color="#dc2626", lw=1.5, ls="--")
        ax.plot(P_lm[trunk_idx, 0], P_lm[trunk_idx, 1], color="#2563eb", lw=1.8)
        ax.scatter(T[:, 0], T[:, 1], color="#dc2626", s=14)
        ax.scatter(P_lm[:, 0], P_lm[:, 1], color="#2563eb", marker="^", s=16)
        ax.set_title(f"17-LM Expl = {rec['lm_expl']:.1f}%", fontsize=8.8, color="#1e3a8a", fontweight="bold")

        # Row 2: Silhouette carried by 17-LM fit
        ax = axes1[2, col]
        ax.fill(BT[:, 0], BT[:, 1], color="#fee2e2", alpha=0.7)
        ax.plot(BT[:, 0], BT[:, 1], color="#dc2626", lw=1.3, ls="--")
        ax.plot(rec["BS_lm"][:, 0], rec["BS_lm"][:, 1], color="#2563eb", lw=1.5)
        ax.set_title(f"17-LM IoU = {rec['bd_lm']['iou']:.3f} (|AΔB|={rec['bd_lm']['sym_diff_pct_of_target']:.1f}%)", fontsize=8.5, color="#b91c1c", fontweight="bold")

        # Row 3: Direct 600-pt silhouette fit
        ax = axes1[3, col]
        ax.fill(BT[:, 0], BT[:, 1], color="#dcfce7", alpha=0.75)
        ax.plot(BT[:, 0], BT[:, 1], color="#dc2626", lw=1.3, ls="--")
        ax.plot(rec["BS_600"][:, 0], rec["BS_600"][:, 1], color="#15803d", lw=1.6)
        ax.set_title(f"600-pt IoU = {rec['bd_600']['iou']:.3f} (|AΔB|={rec['bd_600']['sym_diff_pct_of_target']:.1f}%)", fontsize=8.5, color="#15803d", fontweight="bold")

        for r in range(4):
            axes1[r, col].set_aspect("equal")
            axes1[r, col].set_xticks([])
            axes1[r, col].set_yticks([])
            if col == 0:
                axes1[r, col].set_ylabel(row_labels1[r], fontsize=8.8, fontweight="bold")

    fig1.tight_layout(rect=[0.01, 0.01, 0.99, 0.93])
    fig1.savefig(ROOT / "thompson_stick_vs_silhouette_reality.png", dpi=200)
    plt.close(fig1)

    # Render Figure 2 (thompson_silhouette_area_difference.png)
    fig2, axes2 = plt.subplots(3, 5, figsize=(19.5, 10.2), dpi=200, facecolor="white")
    fig2.suptitle(
        "Exact Area of Difference (|Ω_A Δ Ω_B|) on 600-Point Continuous Fish Silhouettes Across Thompson's 1917 Pairs\n"
        "[Green] Shared Overlap |Ω_A ∩ Ω_B|   |   [Crimson Red] Missing Target |Ω_B \\ Ω_A|   |   [Royal Blue] Excess Source Overhang |Ω_A \\ Ω_B|",
        fontsize=12.8, fontweight="bold", y=0.98
    )
    row_labels2 = [
        "Row 1: Unwarped Baseline\n(4p Similarity Alignment)",
        "Row 2: Warped by 17-Landmark Fit\n(Head Landmarks Pull Posterior Wrong Way)",
        "Row 3: Warped by Direct 600-pt Fit\n(Same 5-8p Coordinate Family)",
    ]
    for col, rec in enumerate(five_recs):
        for r, (sil_w, bd, col_txt) in enumerate([
            (rec["BS_unw"], rec["bd_unw"], "#334155"),
            (rec["BS_lm"], rec["bd_lm"], "#b91c1c"),
            (rec["BS_600"], rec["bd_600"], "#15803d"),
        ]):
            ax = axes2[r, col]
            ra, rb = bd["ra"], bd["rb"]
            xmin, xmax, ymin, ymax = bd["extent"]
            H, W = ra.shape
            img = np.ones((H, W, 4), dtype=float)
            img[:, :] = [0.985, 0.988, 0.995, 1.0]
            img[ra & rb] = [0.16, 0.66, 0.38, 0.78]
            img[rb & (~ra)] = [0.86, 0.18, 0.22, 0.84]
            img[ra & (~rb)] = [0.15, 0.42, 0.85, 0.80]
            ax.imshow(img, extent=[xmin, xmax, ymin, ymax], origin="lower", aspect="equal")
            ax.plot(rec["BT"][:, 0], rec["BT"][:, 1], color="#7f1d1d", lw=1.3, ls="--")
            ax.plot(sil_w[:, 0], sil_w[:, 1], color="#0f172a", lw=1.3)
            ax.set_xticks([])
            ax.set_yticks([])
            head = rec["title"] + "\n" if r == 0 else ""
            ax.set_title(
                f"{head}|AΔB|={bd['sym_diff_pct_of_target']:.1f}% (Miss {bd['missing_pct']:.1f}% / Over {bd['overhang_pct']:.1f}%, IoU {bd['iou']:.3f})",
                fontsize=8.2, color=col_txt, fontweight="bold"
            )
            if col == 0:
                ax.set_ylabel(row_labels2[r], fontsize=8.8, fontweight="bold")

    fig2.tight_layout(rect=[0.01, 0.01, 0.99, 0.92])
    fig2.savefig(ROOT / "thompson_silhouette_area_difference.png", dpi=200)
    plt.close(fig2)

    print(f"Loaded {len(specs)} 600-point silhouettes and saved {out_path.name} + PNG plates.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

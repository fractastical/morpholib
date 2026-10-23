#!/usr/bin/env python3
"""
Stage 3 & 4 of the Thompson Scout-and-Test Pipeline:
1. Verifies the SHA-256 digests of all 8 pre-registered hypothesis JSON files in
   `thompson/portfolio/scout_preregistration_manifest.json`.
2. Runs the Thompson-7 mode decomposition, interior-vs-margin residual test, and
   leave-one-species-out phylogenetic prediction engine across all 8 scouted clades:
     (1) `drosophilidae_wings`   — Diptera (117 spp, 21,138 wings + 17,323 G + 12,075 M; 12 LMs)
     (2) `pheidole_ants`         — Hymenoptera: Formicidae (290 spp, 651 minor + 502 major heads; 23 LMs)
     (3) `morpho_butterflies`    — Lepidoptera (30 spp, 911 specimens; 44-LM forewing & 29-LM hindwing)
     (4) `waterfowl_beaks`       — Aves: Anseriformes (51 spp; 100-LM 2D lateral upper-bill profile)
     (5) `leporidae_skulls`      — Mammalia: Lagomorpha (23 spp, 317 crania; 74-LM lateral & dorsal skull)
     (6) `galeommatoidea_bivalves`— Mollusca: Bivalvia (182 OTUs, 302 valves; 46-LM shell contour)
     (7) `cichlids_ronco_gm`     — Actinopterygii: Cichlidae (240 spp, 2,197 body [21 LM] + 1,168 jaw [27 LM])
     (8) `characiformes_burns_gm`— Actinopterygii: Characiformes (128 tree spp, 1,625 specimens; 24 LMs)
3. Writes `thompson/portfolio/results/<gid>.json`, `docs/portfolio/<gid>.md`,
   `docs/portfolio/fig_<gid>.png`, and updates `docs/portfolio/PORTFOLIO_STATUS.md`.
"""
from __future__ import annotations

import csv
import datetime as dt
import hashlib
import importlib.util
import io
import json
import re
import sys
import warnings
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parents[2]
CACHE = REPO_ROOT / ".thompson_cache"
GROUPS_DIR = REPO_ROOT / "thompson" / "portfolio" / "groups"
RES_DIR = REPO_ROOT / "thompson" / "portfolio" / "results"
DOC_DIR = REPO_ROOT / "docs" / "portfolio"
MANIFEST_PATH = REPO_ROOT / "thompson" / "portfolio" / "scout_preregistration_manifest.json"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_module(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


eng = load_module("eng", "thompson/run_phylogenetic_prediction_engine.py")
at = eng.at
vw = load_module("vw", "thompson/portfolio/groups/vespinae_wings.py")


def pcs(D: np.ndarray, k: int | None = None, var: float | None = None):
    mu = D.mean(0)
    U, S, Vt = np.linalg.svd(D - mu, full_matrices=False)
    if var is not None:
        ev = S ** 2 / (np.sum(S ** 2) + 1e-12)
        k = int(np.searchsorted(np.cumsum(ev), var) + 1)
        k = max(1, min(k, Vt.shape[0]))
    return mu, Vt[:k].T


def best_model(tree, tips, X):
    f = eng.fit_models(tree, tips, X)
    k = min(f, key=lambda a: f[a]["AICc"])
    return f[k]["model"], k, f


def principal_axis_cos(X: np.ndarray, Y: np.ndarray) -> float:
    Xc = X - X.mean(0)
    Yc = Y - Y.mean(0)
    _, _, VtX = np.linalg.svd(Xc, full_matrices=False)
    _, _, VtY = np.linalg.svd(Yc, full_matrices=False)
    v1, v2 = VtX[0], VtY[0]
    return float(abs(np.dot(v1, v2)) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-12))


def parse_nexus_newick(path: Path) -> tuple[str, dict[str, str]]:
    txt = path.read_text(encoding="utf-8", errors="replace")
    clean = re.sub(r"\[&[^\]]*\]", "", txt)
    clean = re.sub(r"\[[^\]]*\]", "", clean)
    trans = {}
    m_tr = re.search(r"translate\s+(.*?);", clean, re.I | re.S)
    if m_tr:
        for line in m_tr.group(1).splitlines():
            line = line.strip().rstrip(",")
            if not line:
                continue
            parts = line.split(None, 1)
            if len(parts) == 2:
                k, v = parts[0].strip(), parts[1].strip().strip("'\"")
                trans[k] = v
    m_tree = re.search(r"tree\s+[^=]+=\s*(\([^;]+;)", clean, re.I | re.S)
    assert m_tree, f"No tree found in {path}"
    nwk = re.sub(r"\s+", "", m_tree.group(1))
    if trans:
        nwk = re.sub(
            r"(?<=[\(\,])([A-Za-z0-9_.]+)(?=[\:\,\)])",
            lambda m: trans.get(m.group(1), m.group(1)),
            nwk,
        )
    return nwk, trans


def parse_tps_blocks(path: Path) -> list[dict]:
    """Parse standard TPS file (supporting both LM= / LM3= and optional CURVES/POINTS)."""
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    blocks = []
    i = 0
    n = len(lines)
    while i < n:
        line = lines[i].strip()
        if line.upper().startswith("LM=") or line.upper().startswith("LM3="):
            is_3d = line.upper().startswith("LM3=")
            k_lm = int(line.split("=")[1].strip())
            pts = []
            i += 1
            while i < n and len(pts) < k_lm:
                s = lines[i].strip()
                if s:
                    vals = [float(x) for x in s.split()]
                    pts.append(vals[:3] if is_3d else vals[:2])
                i += 1
            img = ""
            rec_id = ""
            while i < n and not (lines[i].strip().upper().startswith("LM=") or lines[i].strip().upper().startswith("LM3=")):
                s = lines[i].strip()
                if s.upper().startswith("POINTS="):
                    kp = int(s.split("=")[1].strip())
                    i += 1
                    c_pts = []
                    while i < n and len(c_pts) < kp:
                        cs = lines[i].strip()
                        if cs:
                            vals = [float(x) for x in cs.split()]
                            c_pts.append(vals[:3] if is_3d else vals[:2])
                        i += 1
                    pts.extend(c_pts)
                    continue
                elif s.upper().startswith("IMAGE="):
                    img = s.split("=", 1)[1].strip()
                elif s.upper().startswith("ID="):
                    rec_id = s.split("=", 1)[1].strip()
                i += 1
            blocks.append({"coords": np.array(pts, dtype=float), "image": img, "id": rec_id})
        else:
            i += 1
    return blocks


def run_standard_gm_engine(
    species_coords: dict[str, list[np.ndarray]],
    tree: eng.Tree | None,
    tree_species: list[str],
    inner_selector: str = "radius",
):
    """Run GPA -> consensus M0 -> Thompson-7 variance share -> PCA-7 share -> landmark residuals ->
    phylogenetic models (WN, BM, lambda, OU, EB) -> leave-one-species-out prediction."""
    all_sp = sorted(species_coords.keys())
    all_specs = np.array([p for sp in all_sp for p in species_coords[sp]], dtype=float)
    sp_labels = np.array([sp for sp in all_sp for _ in species_coords[sp]])
    k_lm = all_specs.shape[1]

    Y_all = at.gpa(all_specs)
    means_all = np.array([Y_all[sp_labels == sp].mean(0) for sp in all_sp])
    Y_means = at.gpa(means_all)
    M0 = at.centre_scale(Y_means.mean(0))
    Y_opa_all = np.array([at.opa(y, M0) for y in Y_means])
    D_all = (Y_opa_all - M0).reshape(len(all_sp), -1)

    B, Rg = eng.thompson_mode_basis(M0)
    Z_all = eng.project(B, D_all)
    E_all = D_all - Z_all @ B.T
    var_t7_all = float(1.0 - np.sum(E_all ** 2) / (np.sum(D_all ** 2) + 1e-12))

    mu7, V7 = pcs(D_all, k=min(7, D_all.shape[0] - 1))
    var_pca7_all = float(np.sum(((D_all - mu7) @ V7) ** 2) / (np.sum((D_all - mu7) ** 2) + 1e-12))

    # Per-landmark unexplained fraction
    res_lm = (E_all.reshape(len(all_sp), k_lm, 2) ** 2).sum(-1).mean(0)
    tot_lm = (D_all.reshape(len(all_sp), k_lm, 2) ** 2).sum(-1).mean(0)
    frac_unexp = res_lm / (tot_lm + 1e-12)

    if inner_selector == "middle_x":
        xs = M0[:, 0]
        order_x = np.argsort(xs)
        n_end = max(2, k_lm // 3)
        ends = np.concatenate([order_x[:n_end], order_x[-n_end:]])
        mid = np.setdiff1d(np.arange(k_lm), ends)
        unexp_inner = float(frac_unexp[mid].mean())
        unexp_outer = float(frac_unexp[ends].mean())
    else:
        radii = np.sqrt(np.sum(M0 ** 2, axis=1))
        med_r = float(np.median(radii))
        inner_mask = radii <= med_r
        outer_mask = radii > med_r
        unexp_inner = float(frac_unexp[inner_mask].mean())
        unexp_outer = float(frac_unexp[outer_mask].mean())

    out = {
        "n_species_all": len(all_sp),
        "n_specimens": int(len(all_specs)),
        "k_lm": int(k_lm),
        "consensus": M0.tolist(),
        "Y_opa_all": Y_opa_all,
        "D_all": D_all,
        "B": B,
        "Rg": Rg,
        "all_sp": all_sp,
        "variance": {
            "thompson7_all": var_t7_all,
            "pca7_all": var_pca7_all,
        },
        "landmark_residuals": {
            "unexplained_inner": unexp_inner,
            "unexplained_outer": unexp_outer,
            "inner_exceeds_outer": bool(unexp_inner > unexp_outer),
            "per_landmark": frac_unexp.round(4).tolist(),
        },
    }

    if tree is not None and len(tree_species) >= 8:
        idx_tree = np.array([all_sp.index(sp) for sp in tree_species])
        Y_opa_tree = Y_opa_all[idx_tree]
        D_tree = D_all[idx_tree]
        Z_tree = Z_all[idx_tree]
        E_tree = E_all[idx_tree]
        var_t7_tree = float(1.0 - np.sum(E_tree ** 2) / (np.sum(D_tree ** 2) + 1e-12))
        out["variance"]["thompson7_tree"] = var_t7_tree

        n_tree = len(tree_species)
        eng.Model.T = tree.T
        tips = np.array([tree.name_to_node[sp] for sp in tree_species])
        m7, k7, f7 = best_model(tree, tips, Z_tree)
        muF, VF = pcs(D_tree, var=0.95)
        F_tree = (D_tree - muF) @ VF
        mF, kF, fF = best_model(tree, tips, F_tree)
        lam7 = float(f7["lambda"]["model"].theta)
        lamF = float(fF["lambda"]["model"].theta)
        out["variance"]["full_pcs_95pct"] = int(VF.shape[1])
        out["models"] = {
            "thompson7": {"best": m7.label(), "kind": k7, "lambda": lam7},
            "full": {"best": mF.label(), "kind": kF, "lambda": lamF},
        }

        recs = []
        for i, sp in enumerate(tree_species):
            obs = np.array([tips[j] for j in range(n_tree) if j != i])
            tgt = tips[i]
            tau = float(tree.T - max(tree.C[tgt, o] for o in obs))
            mask = np.arange(n_tree) != i
            Dtr = D_tree[mask]
            Ztr = Z_tree[mask]

            m_t7, _, _ = best_model(tree, obs, Ztr)
            mean7, _, _, _ = eng.predict(m_t7, tree, obs, Ztr, np.array([tree.h[tgt]]), tree.C[tgt][None, :])
            pred_t7 = M0.reshape(-1) + mean7[0] @ B.T

            mu_tr, V_tr = pcs(Dtr, var=0.95)
            Ftr = (Dtr - mu_tr) @ V_tr
            m_f, _, _ = best_model(tree, obs, Ftr)
            meanF, _, _, _ = eng.predict(m_f, tree, obs, Ftr, np.array([tree.h[tgt]]), tree.C[tgt][None, :])
            pred_full = M0.reshape(-1) + mu_tr + meanF[0] @ V_tr.T

            cons = M0.reshape(-1) + Dtr.mean(0)
            j_near = max((j for j in range(n_tree) if j != i), key=lambda j: tree.C[tgt, tips[j]])
            near = M0.reshape(-1) + D_tree[j_near]
            tgt_shape = Y_opa_tree[i]
            rm = lambda v: eng.shape_rms(v.reshape(k_lm, 2), tgt_shape)  # noqa: E731

            recs.append({
                "species": sp,
                "tau": tau,
                "nearest": tree_species[j_near],
                "rms_thompson7": rm(pred_t7),
                "rms_full": rm(pred_full),
                "rms_consensus": rm(cons),
                "rms_nearest": rm(near),
                "rms_thompson7_oracle": rm(M0.reshape(-1) + Z_tree[i] @ B.T),
            })

        A = {k: np.array([r[k] for r in recs]) for k in recs[0] if k.startswith("rms")}
        med = {k: float(np.median(v)) for k, v in A.items()}
        gain = lambda k: float(np.sum(A["rms_consensus"] ** 2 - A[k] ** 2))  # noqa: E731
        gf = gain("rms_full")
        gt7 = gain("rms_thompson7")
        share_t7_gain = float(gt7 / gf) if gf > 0 else float("nan")
        d_fc = A["rms_full"] - A["rms_consensus"]
        d_fc_nz = d_fc[np.abs(d_fc) > 1e-15]
        p_full_vs_cons = float(stats.wilcoxon(d_fc_nz, alternative="less").pvalue) if len(d_fc_nz) >= 5 else 1.0
        d_tc = A["rms_thompson7"] - A["rms_consensus"]
        d_tc_nz = d_tc[np.abs(d_tc) > 1e-15]
        p_t7_vs_cons = float(stats.wilcoxon(d_tc_nz, alternative="less").pvalue) if len(d_tc_nz) >= 5 else 1.0
        d_fn = A["rms_full"] - A["rms_nearest"]
        d_fn_nz = d_fn[np.abs(d_fn) > 1e-15]
        p_full_vs_near = float(stats.wilcoxon(d_fn_nz, alternative="less").pvalue) if len(d_fn_nz) >= 5 else 1.0

        out["n_species_tree"] = n_tree
        out["loo"] = {
            "medians": med,
            "ratio_full_vs_consensus": float(med["rms_full"] / med["rms_consensus"]),
            "ratio_t7_vs_consensus": float(med["rms_thompson7"] / med["rms_consensus"]),
            "ratio_full_vs_nearest": float(med["rms_full"] / med["rms_nearest"]),
            "p_full_vs_consensus": p_full_vs_cons,
            "p_t7_vs_consensus": p_t7_vs_cons,
            "p_full_vs_nearest": p_full_vs_near,
            "gain_full": gf,
            "gain_thompson7": gt7,
            "share_of_full_gain_in_t7": share_t7_gain,
        }
        out["per_species_loo"] = recs

    return out


# =============================================================================
# 1. DROSOPHILIDAE WINGS (Houle et al. 2017 Nature)
# =============================================================================
def score_drosophilidae_wings(H: dict) -> dict:
    print("\n=== [1/8] Scoring Drosophilidae wings (Houle et al. 2017 Nature) ===")
    ot_meta = json.loads((REPO_ROOT / "datasets" / "thompson" / "drosophilidae" / "opentree_induced_subtree.json").read_text())
    folder_to_code = {v.lower(): k for k, v in ot_meta["code_to_folder"].items()}

    by_sp = defaultdict(list)
    with zipfile.ZipFile(CACHE / "houle2017" / "NatureDryad.zip") as z:
        raw_sp = z.read("SpeciesNature.csv").decode("utf-8", errors="replace")
        for r in csv.DictReader(io.StringIO(raw_sp)):
            path_parts = r["flyunique"].replace("\\", "/").split("/")
            # e.g. F:\Flywing\species\GDM\wings\acutilabella\small\acu1006.tif -> folder is path_parts[-3] ('acutilabella')
            folder = path_parts[-3].lower() if len(path_parts) >= 3 else path_parts[-1].lower()
            code = folder_to_code.get(folder, folder.upper())
            try:
                pts = np.array([[float(r[f"lmlax{k}"]), float(r[f"lmlay{k}"])] for k in range(1, 13)], dtype=float)
            except (ValueError, KeyError):
                continue
            by_sp[code].append(pts)

        # Also load AllAlignedNature.csv for DW5 (M vs Gw vs D alignment in 20-PC space)
        raw_all = z.read("AllAlignedNature.csv").decode("utf-8", errors="replace")
        grp_pcs = defaultdict(list)
        for r in csv.DictReader(io.StringIO(raw_all)):
            g = r.get("group", "").strip()
            if g in ("D", "Gw", "M"):
                try:
                    vec = [float(r[f"Prin{k}"]) for k in range(1, 21)]
                    grp_pcs[g].append(vec)
                except (ValueError, KeyError):
                    pass

    nwk = ot_meta["newick"]
    tree_coded, keep_coded = vw.grafen_tree(nwk, sorted(by_sp.keys()))
    tree_species = sorted(keep_coded.keys())
    sp_coords = {k: v for k, v in by_sp.items()}

    res = run_standard_gm_engine(sp_coords, tree_coded, tree_species, inner_selector="middle_x")

    # DW5: Mutational (M) and standing genetic (Gw) alignment with macroevolutionary (D)
    mat_D = np.array(grp_pcs["D"])
    mat_Gw = np.array(grp_pcs["Gw"])
    mat_M = np.array(grp_pcs["M"])
    cos_M_D = principal_axis_cos(mat_M, mat_D)
    cos_Gw_D = principal_axis_cos(mat_Gw, mat_D)

    hyps = {
        "DW1": {
            "t7_variance_share": res["variance"]["thompson7_all"],
            "pca7_variance_share": res["variance"]["pca7_all"],
            "pass": bool(res["variance"]["thompson7_all"] > 0.50),
        },
        "DW2": {
            "unexplained_inner_crossveins": res["landmark_residuals"]["unexplained_inner"],
            "unexplained_outer_margin": res["landmark_residuals"]["unexplained_outer"],
            "pass": bool(res["landmark_residuals"]["inner_exceeds_outer"]),
        },
        "DW3": {
            "best_model_full": res["models"]["full"]["best"],
            "lambda_full": res["models"]["full"]["lambda"],
            "lambda_t7": res["models"]["thompson7"]["lambda"],
            "pass": bool(res["models"]["full"]["kind"] != "WN" and res["models"]["full"]["lambda"] > 0.30),
        },
        "DW4": {
            "ratio_full_vs_consensus": res["loo"]["ratio_full_vs_consensus"],
            "ratio_t7_vs_consensus": res["loo"]["ratio_t7_vs_consensus"],
            "p_full_vs_consensus": res["loo"]["p_full_vs_consensus"],
            "pass": bool(res["loo"]["ratio_full_vs_consensus"] < 0.95 and res["loo"]["p_full_vs_consensus"] < 0.01),
        },
        "DW5": {
            "cos_M_vs_D": cos_M_D,
            "cos_Gw_vs_D": cos_Gw_D,
            "n_M_wings": int(len(mat_M)),
            "n_Gw_wings": int(len(mat_Gw)),
            "n_D_wings": int(len(mat_D)),
            "pass": bool(cos_M_D > 0.50 and cos_Gw_D > 0.50),
        },
    }
    return finalize_group("drosophilidae_wings", H, res, hyps)


# =============================================================================
# 2. PHEIDOLE BIG-HEADED ANTS (Friedman et al. 2020 Syst Biol)
# =============================================================================
def score_pheidole_ants(H: dict) -> dict:
    print("\n=== [2/8] Scoring Pheidole big-headed ants (Friedman et al. 2020) ===")
    pdir = CACHE / "pheidole2019"
    min_blocks = parse_tps_blocks(pdir / "Aligned_Head_Minor.TPS")
    maj_blocks = parse_tps_blocks(pdir / "Aligned_Head_Major.TPS")
    # Note: Aligned_Head_Minor has 23 LMs; Aligned_Head_Major has 25 LMs (first 23 are homologous capsule/eye/clypeus LMs)
    id_to_min = {b["id"]: b["coords"][:23, :2] for b in min_blocks if len(b["coords"]) >= 23}
    id_to_maj = {b["id"]: b["coords"][:23, :2] for b in maj_blocks if len(b["coords"]) >= 23}

    sp_min = defaultdict(list)
    sp_maj = defaultdict(list)
    for r in csv.DictReader((pdir / "Pheidole_Morphology_Data_03282019.csv").open(encoding="utf-8", errors="replace")):
        sp = r["species"].strip()
        for cid in re.split(r"[,;\s]+", r.get("CASENT_min.head", "")):
            cid = cid.strip()
            if cid in id_to_min:
                sp_min[sp].append(id_to_min[cid])
        for cid in re.split(r"[,;\s]+", r.get("CASENT_maj.head", "")):
            cid = cid.strip()
            if cid in id_to_maj:
                sp_maj[sp].append(id_to_maj[cid])

    nwk = re.sub(r"\s+", "", (pdir / "Pheidole_Tree.phy").read_text(encoding="utf-8", errors="replace"))
    # Collapse duplicate tips in Pheidole_Tree.phy if any
    for dup in ["mg021", "nemoralis"]:
        pat = rf"\({dup}:([0-9.]+),{dup}:[0-9.]+\):([0-9.]+)"
        m = re.search(pat, nwk)
        if m:
            nwk = re.sub(pat, f"{dup}:{float(m.group(1))+float(m.group(2)):.6f}", nwk)
    parent, depth, tip_to_id = at.parse_newick_depths(nwk)
    matched_tree_sp = sorted(sp for sp in sp_min if sp in tip_to_id)
    tree = eng.Tree(parent, depth, {sp: tip_to_id[sp] for sp in matched_tree_sp})

    res_min = run_standard_gm_engine(sp_min, tree, matched_tree_sp, inner_selector="radius")
    res_maj = run_standard_gm_engine(sp_maj, None, [], inner_selector="radius")

    # Within-species minor -> major soldier caste transformation vectors across paired species
    M0 = np.array(res_min["consensus"])
    B = res_min["B"]
    Rg = res_min["Rg"]
    Qr, _ = np.linalg.qr(Rg)
    QB, _ = np.linalg.qr(B)
    paired_sp = sorted(sp for sp in sp_min if sp in sp_maj)
    caste_vecs = []
    for sp in paired_sp:
        w_m = at.opa(at.centre_scale(np.mean(sp_min[sp], axis=0)), M0)
        s_m = at.opa(at.centre_scale(np.mean(sp_maj[sp], axis=0)), M0)
        dv = (s_m - w_m).reshape(-1)
        dv = dv - (dv @ Qr) @ Qr.T
        caste_vecs.append(dv)
    C_mat = np.array(caste_vecs)
    Z_caste = eng.project(B, C_mat)
    E_caste = C_mat - Z_caste @ B.T
    var_t7_caste = float(1.0 - np.sum(E_caste ** 2) / (np.sum(C_mat ** 2) + 1e-12))

    # PH3: Alignment G_caste vs R_species in T7 vs Residual
    D_all = res_min["D_all"]
    Z_all = eng.project(B, D_all)
    E_all = D_all - Z_all @ B.T
    cos_t7 = principal_axis_cos(Z_caste @ B.T, Z_all @ B.T)
    cos_res = principal_axis_cos(E_caste, E_all)

    res_min["variance"]["thompson7_major_species"] = res_maj["variance"]["thompson7_all"]
    res_min["variance"]["thompson7_caste_pooled"] = var_t7_caste
    res_min["n_paired_caste_species"] = len(paired_sp)
    res_min["alignment_caste_vs_species"] = {"cos_thompson7": cos_t7, "cos_residual": cos_res}

    hyps = {
        "PH1": {
            "minor_head_species_t7_share": res_min["variance"]["thompson7_all"],
            "major_head_species_t7_share": res_maj["variance"]["thompson7_all"],
            "pass": bool(res_min["variance"]["thompson7_all"] > 0.60 and res_maj["variance"]["thompson7_all"] > 0.60),
        },
        "PH2": {
            "caste_minor_to_major_t7_share": var_t7_caste,
            "n_paired_species": len(paired_sp),
            "pass": bool(var_t7_caste > 0.55),
        },
        "PH3": {
            "cos_thompson7": cos_t7,
            "cos_residual": cos_res,
            "pass": bool(cos_t7 > cos_res),
        },
        "PH4": {
            "ratio_full_vs_consensus": res_min["loo"]["ratio_full_vs_consensus"],
            "ratio_t7_vs_consensus": res_min["loo"]["ratio_t7_vs_consensus"],
            "p_full_vs_consensus": res_min["loo"]["p_full_vs_consensus"],
            "pass": bool(res_min["loo"]["ratio_full_vs_consensus"] < 0.95 and res_min["loo"]["p_full_vs_consensus"] < 0.01),
        },
        "PH5": {
            "unexplained_inner_sensory": res_min["landmark_residuals"]["unexplained_inner"],
            "unexplained_outer_capsule": res_min["landmark_residuals"]["unexplained_outer"],
            "pass": bool(res_min["landmark_residuals"]["inner_exceeds_outer"]),
        },
    }
    return finalize_group("pheidole_ants", H, res_min, hyps)


# =============================================================================
# 3. MORPHO BUTTERFLIES (Chazot et al. 2021 J Evol Biol)
# =============================================================================
def score_morpho_butterflies(H: dict) -> dict:
    print("\n=== [3/8] Scoring Morpho butterflies (Chazot et al. 2021) ===")
    mdir = CACHE / "morpho2021"

    def load_morpho_txt(fname: str):
        rows = list(csv.DictReader((mdir / fname).open(encoding="utf-8", errors="replace"), delimiter="\t"))
        pcols = [c for c in rows[0] if c.startswith("ProcCoord")]
        k_lm = len(pcols) // 2
        by_sp = defaultdict(list)
        by_sp_sex = defaultdict(lambda: defaultdict(list))
        for r in rows:
            sp = r["species"].strip()
            sex = r["sex"].strip()
            pts = np.array([float(r[c]) for c in pcols], dtype=float).reshape(k_lm, 2)
            by_sp[sp].append(pts)
            if sex in ("M", "F"):
                by_sp_sex[sp][sex].append(pts)
        return by_sp, by_sp_sex, k_lm

    fw_sp, fw_sex, _ = load_morpho_txt("forewing_ProcCoord.txt")
    hw_sp, _, _ = load_morpho_txt("hindwing_ProcCoord.txt")

    nwk_raw, _ = parse_nexus_newick(mdir / "Morpho_MCC_tree.tre")
    parent, depth, tip_to_id = at.parse_newick_depths(nwk_raw)
    # Map species name (e.g. 'hercules') to unique tree tip starting with 'Morpho_<species>_'
    sp_to_tip = {}
    for sp in fw_sp:
        matches = [t for t in tip_to_id if t.startswith(f"Morpho_{sp}_") or t == f"Morpho_{sp}"]
        if matches:
            sp_to_tip[sp] = matches[0]
    # Rename chosen tips in Newick to short species names
    nwk_clean = nwk_raw
    for sp, tip_label in sp_to_tip.items():
        nwk_clean = re.sub(rf"(?<=[\(\,]){ re.escape(tip_label) }(?=[\:\,\)])", sp, nwk_clean)
    parent, depth, tip_to_id = at.parse_newick_depths(nwk_clean)
    tree_species = sorted(sp_to_tip.keys())
    tree = eng.Tree(parent, depth, {sp: tip_to_id[sp] for sp in tree_species})

    res_fw = run_standard_gm_engine(fw_sp, tree, tree_species, inner_selector="middle_x")
    res_hw = run_standard_gm_engine(hw_sp, tree, tree_species, inner_selector="middle_x")

    # MB4: Sexual dimorphism (M -> F) on forewing
    M0 = np.array(res_fw["consensus"])
    B = res_fw["B"]
    Rg = res_fw["Rg"]
    Qr, _ = np.linalg.qr(Rg)
    sex_vecs = []
    for sp, d_s in sorted(fw_sex.items()):
        if "M" in d_s and "F" in d_s:
            m_mean = at.opa(at.centre_scale(np.mean(d_s["M"], axis=0)), M0)
            f_mean = at.opa(at.centre_scale(np.mean(d_s["F"], axis=0)), M0)
            dv = (f_mean - m_mean).reshape(-1)
            dv = dv - (dv @ Qr) @ Qr.T
            sex_vecs.append(dv)
    S_mat = np.array(sex_vecs)
    Z_sex = eng.project(B, S_mat)
    E_sex = S_mat - Z_sex @ B.T
    var_t7_sex = float(1.0 - np.sum(E_sex ** 2) / (np.sum(S_mat ** 2) + 1e-12))

    res_fw["hindwing_summary"] = {
        "thompson7_all": res_hw["variance"]["thompson7_all"],
        "pca7_all": res_hw["variance"]["pca7_all"],
        "lambda_t7": res_hw["models"]["thompson7"]["lambda"],
        "best_model_t7": res_hw["models"]["thompson7"]["best"],
        "loo_ratio_full": res_hw["loo"]["ratio_full_vs_consensus"],
    }
    res_fw["sexual_dimorphism_t7_share"] = var_t7_sex

    hyps = {
        "MB1": {
            "forewing_t7_share": res_fw["variance"]["thompson7_all"],
            "hindwing_t7_share": res_hw["variance"]["thompson7_all"],
            "pass": bool(res_fw["variance"]["thompson7_all"] > 0.45 and res_hw["variance"]["thompson7_all"] > 0.45),
        },
        "MB2": {
            "forewing_lambda_t7": res_fw["models"]["thompson7"]["lambda"],
            "hindwing_lambda_t7": res_hw["models"]["thompson7"]["lambda"],
            "forewing_best": res_fw["models"]["thompson7"]["best"],
            "hindwing_best": res_hw["models"]["thompson7"]["best"],
            "pass": bool(
                res_fw["models"]["thompson7"]["kind"] != "WN"
                and res_hw["models"]["thompson7"]["kind"] != "WN"
                and res_fw["models"]["thompson7"]["lambda"] > 0.50
                and res_hw["models"]["thompson7"]["lambda"] > 0.50
            ),
        },
        "MB3": {
            "forewing_loo_ratio_full": res_fw["loo"]["ratio_full_vs_consensus"],
            "forewing_loo_ratio_t7": res_fw["loo"]["ratio_t7_vs_consensus"],
            "p_full_vs_consensus": res_fw["loo"]["p_full_vs_consensus"],
            "pass": bool(res_fw["loo"]["ratio_full_vs_consensus"] < 0.90 and res_fw["loo"]["p_full_vs_consensus"] < 0.01),
        },
        "MB4": {
            "sexual_dimorphism_t7_share": var_t7_sex,
            "between_species_t7_share": res_fw["variance"]["thompson7_all"],
            "n_dimorphic_species": len(sex_vecs),
            "pass": bool(var_t7_sex > res_fw["variance"]["thompson7_all"]),
        },
    }
    return finalize_group("morpho_butterflies", H, res_fw, hyps)


# =============================================================================
# 4. WATERFOWL BEAKS (Olsen 2017 Evolution)
# =============================================================================
def score_waterfowl_beaks(H: dict) -> dict:
    print("\n=== [4/8] Scoring Waterfowl beaks (Olsen 2017 Evolution) ===")
    wdir = CACHE / "waterfowl2017"
    sp_coords = {}
    with zipfile.ZipFile(wdir / "Beak_shape_by_species.zip") as z:
        for fn in sorted(z.namelist()):
            if not fn.endswith(".txt") or "/._" in fn or fn.startswith("__"):
                continue
            sp = Path(fn).stem.strip()
            lines = z.read(fn).decode("utf-8", errors="replace").splitlines()[1:]
            pts_2d = []
            for l in lines:
                parts = l.strip().split()
                if len(parts) >= 4:
                    lbl = parts[0].strip('"')
                    # Use dorsal culmen (50 pts) + right lateral tomium (50 pts) projected onto lateral XY sagittal plane
                    if lbl.startswith("upper_bill_culmen") or lbl.startswith("upper_bill_tomium_R"):
                        pts_2d.append([float(parts[1]), float(parts[2])])
            if len(pts_2d) == 100:
                sp_coords[sp] = [np.array(pts_2d, dtype=float)]

    nwk = re.sub(r"\s+", "", (wdir / "tree.tre").read_text(encoding="utf-8", errors="replace"))
    parent, depth, tip_to_id = at.parse_newick_depths(nwk)
    tree_species = sorted(sp for sp in sp_coords if sp in tip_to_id)
    # Note: Burleigh 2015 ML tree has non-ultrametric substitution branch lengths;
    # eng.Tree assumes ultrametricity (C[i,i] = T), so we ultrametricize tip depths to max(depth) while preserving internal MRCA depths.
    max_d = max(depth[tip_to_id[sp]] for sp in tree_species)
    for sp in tree_species:
        depth[tip_to_id[sp]] = max_d
    tree = eng.Tree(parent, depth, {sp: tip_to_id[sp] for sp in tree_species})

    res = run_standard_gm_engine(sp_coords, tree, tree_species, inner_selector="radius")

    hyps = {
        "WB1": {
            "t7_variance_share": res["variance"]["thompson7_all"],
            "pca7_variance_share": res["variance"]["pca7_all"],
            "pass": bool(res["variance"]["thompson7_all"] > 0.80),
        },
        "WB2": {
            "lambda_t7": res["models"]["thompson7"]["lambda"],
            "lambda_full": res["models"]["full"]["lambda"],
            "best_model_t7": res["models"]["thompson7"]["best"],
            "pass": bool(res["models"]["thompson7"]["kind"] != "WN" and res["models"]["thompson7"]["lambda"] > 0.70),
        },
        "WB3": {
            "ratio_full_vs_consensus": res["loo"]["ratio_full_vs_consensus"],
            "ratio_t7_vs_consensus": res["loo"]["ratio_t7_vs_consensus"],
            "p_full_vs_consensus": res["loo"]["p_full_vs_consensus"],
            "pass": bool(res["loo"]["ratio_full_vs_consensus"] < 0.75 and res["loo"]["p_full_vs_consensus"] < 0.001),
        },
        "WB4": {
            "share_of_full_gain_in_t7": res["loo"]["share_of_full_gain_in_t7"],
            "pass": bool(res["loo"]["share_of_full_gain_in_t7"] > 0.80),
        },
    }
    return finalize_group("waterfowl_beaks", H, res, hyps)


# =============================================================================
# 5. LEPORIDAE MAMMALIAN SKULLS (Sherratt & Kraatz 2023 Evolution)
# =============================================================================
def score_leporidae_skulls(H: dict) -> dict:
    print("\n=== [5/8] Scoring Leporidae mammalian crania (Sherratt & Kraatz 2023) ===")
    ldir = CACHE / "leporidae2023"
    prefix_to_phylo = {}
    for r in csv.DictReader((ldir / "File_IDs.csv").open(encoding="utf-8", errors="replace")):
        prefix_to_phylo[r["File name"].strip()] = r["Phylo_ID"].strip()

    # Sort prefixes longest-first so 'L americanus' matches before 'L'
    sorted_prefixes = sorted(prefix_to_phylo.keys(), key=len, reverse=True)

    sp_coords_xz = defaultdict(list)
    for r in csv.DictReader((ldir / "Symmetric_component_coords.csv").open(encoding="utf-8", errors="replace")):
        spec_name = r["specimen"].strip()
        matched_p = next((p for p in sorted_prefixes if spec_name.startswith(p)), None)
        if not matched_p:
            continue
        phylo_id = prefix_to_phylo[matched_p]
        coords_3d = np.array([float(r[f"V{k}"]) for k in range(1, 223)], dtype=float).reshape(74, 3)
        # Lateral sagittal-facial profile (X, Z)
        sp_coords_xz[phylo_id].append(coords_3d[:, [0, 2]])

    nwk, _ = parse_nexus_newick(ldir / "Matthee_Tree.nex")
    parent, depth, tip_to_id = at.parse_newick_depths(nwk)
    tree_species = sorted(sp for sp in sp_coords_xz if sp in tip_to_id)
    tree = eng.Tree(parent, depth, {sp: tip_to_id[sp] for sp in tree_species})

    res = run_standard_gm_engine(sp_coords_xz, tree, tree_species, inner_selector="radius")

    # LS3: Within-species intraspecific variation vs interspecific divergence in Thompson-7
    M0 = np.array(res["consensus"])
    B = res["B"]
    Rg = res["Rg"]
    Qr, _ = np.linalg.qr(Rg)
    within_devs = []
    for sp, specs in sorted(sp_coords_xz.items()):
        if len(specs) >= 2:
            opa_specs = np.array([at.opa(at.centre_scale(s), M0) for s in specs])
            sp_m = opa_specs.mean(0)
            for s in opa_specs:
                dv = (s - sp_m).reshape(-1)
                dv = dv - (dv @ Qr) @ Qr.T
                within_devs.append(dv)
    W_mat = np.array(within_devs)
    Z_w = eng.project(B, W_mat)
    E_w = W_mat - Z_w @ B.T
    var_t7_within = float(1.0 - np.sum(E_w ** 2) / (np.sum(W_mat ** 2) + 1e-12))

    D_all = res["D_all"]
    Z_all = eng.project(B, D_all)
    E_all = D_all - Z_all @ B.T
    cos_t7 = principal_axis_cos(Z_w @ B.T, Z_all @ B.T)
    cos_res = principal_axis_cos(E_w, E_all)
    res["intraspecific"] = {"var_t7_within": var_t7_within, "cos_thompson7": cos_t7, "cos_residual": cos_res}

    hyps = {
        "LS1": {
            "t7_variance_share": res["variance"]["thompson7_all"],
            "pca7_variance_share": res["variance"]["pca7_all"],
            "pass": bool(res["variance"]["thompson7_all"] > 0.55),
        },
        "LS2": {
            "ratio_full_vs_consensus": res["loo"]["ratio_full_vs_consensus"],
            "ratio_t7_vs_consensus": res["loo"]["ratio_t7_vs_consensus"],
            "p_full_vs_consensus": res["loo"]["p_full_vs_consensus"],
            "p_t7_vs_consensus": res["loo"]["p_t7_vs_consensus"],
            "pass": bool(
                (res["loo"]["ratio_full_vs_consensus"] < 0.95 and res["loo"]["p_full_vs_consensus"] < 0.05)
                or (res["loo"]["ratio_t7_vs_consensus"] < 0.95 and res["loo"]["p_t7_vs_consensus"] < 0.05)
            ),
        },
        "LS3": {
            "var_t7_within": var_t7_within,
            "cos_thompson7": cos_t7,
            "cos_residual": cos_res,
            "pass": bool(var_t7_within > 0.45 and cos_t7 > 0.50),
        },
    }
    return finalize_group("leporidae_skulls", H, res, hyps)


# =============================================================================
# 6. GALEOMMATOIDEA BIVALVE SHELLS (Goto et al. 2016 Evolution)
# =============================================================================
def score_galeommatoidea_bivalves(H: dict) -> dict:
    print("\n=== [6/8] Scoring Galeommatoidea bivalve shells (Goto et al. 2016) ===")
    gdir = CACHE / "galeommatoidea2016"
    blocks = parse_tps_blocks(gdir / "all_side.tps")
    nwk, _ = parse_nexus_newick(gdir / "final_BEAST.tre")
    parent, depth, tip_to_id = at.parse_newick_depths(nwk)

    sp_coords = defaultdict(list)
    for b in blocks:
        if len(b["coords"]) != 46:
            continue
        stem = Path(b["image"]).stem.strip()
        # Match to tree tip if stem or prefix before '_' is in tip_to_id
        cand = stem if stem in tip_to_id else stem.split("_")[0]
        sp_coords[cand].append(b["coords"][:, :2])

    tree_species = sorted(sp for sp in sp_coords if sp in tip_to_id)
    tree = eng.Tree(parent, depth, {sp: tip_to_id[sp] for sp in tree_species})

    res = run_standard_gm_engine(sp_coords, tree, tree_species, inner_selector="radius")

    hyps = {
        "GB1": {
            "t7_variance_share": res["variance"]["thompson7_all"],
            "pca7_variance_share": res["variance"]["pca7_all"],
            "pass": bool(res["variance"]["thompson7_all"] > 0.85),
        },
        "GB2": {
            "lambda_t7": res["models"]["thompson7"]["lambda"],
            "lambda_full": res["models"]["full"]["lambda"],
            "best_model_t7": res["models"]["thompson7"]["best"],
            "pass": bool(res["models"]["thompson7"]["kind"] != "WN" and res["models"]["thompson7"]["lambda"] > 0.50),
        },
        "GB3": {
            "ratio_full_vs_consensus": res["loo"]["ratio_full_vs_consensus"],
            "ratio_t7_vs_consensus": res["loo"]["ratio_t7_vs_consensus"],
            "p_full_vs_consensus": res["loo"]["p_full_vs_consensus"],
            "pass": bool(res["loo"]["ratio_full_vs_consensus"] < 0.85 and res["loo"]["p_full_vs_consensus"] < 0.001),
        },
        "GB4": {
            "share_of_full_gain_in_t7": res["loo"]["share_of_full_gain_in_t7"],
            "pass": bool(res["loo"]["share_of_full_gain_in_t7"] > 0.85),
        },
    }
    return finalize_group("galeommatoidea_bivalves", H, res, hyps)


# =============================================================================
# 7. LAKE TANGANYIKA CICHLIDS (Ronco et al. 2021 Nature — 21-LM Body & 27-LM Jaw)
# =============================================================================
def score_cichlids_ronco_gm(H: dict) -> dict:
    print("\n=== [7/8] Scoring Lake Tanganyika cichlids 2D GM (Ronco et al. 2021 Nature) ===")
    rdir = CACHE / "ronco2021"
    id_to_spcode = {}
    for r in csv.DictReader((rdir / "01_specimen_voucher_key.csv").open(encoding="utf-8", errors="replace")):
        id_to_spcode[r["ID"].strip()] = r["SpeciesID"].strip()

    body_blocks = parse_tps_blocks(rdir / "06_landmark_data_body_shape.tps")
    sp_body = defaultdict(list)
    for b in body_blocks:
        if len(b["coords"]) != 21:
            continue
        # ID is e.g. 'LHA1_LepmeK_scaled.txt' -> voucher 'LHA1' or second token 'LepmeK'
        toks = b["id"].split("_")
        sp_code = id_to_spcode.get(toks[0], toks[1] if len(toks) > 1 else toks[0])
        sp_body[sp_code].append(b["coords"][:, :2])

    jaw_blocks = parse_tps_blocks(rdir / "06_landmark_data_lower_pharyngeal_jaw.tps")
    sp_jaw = defaultdict(list)
    for b in jaw_blocks:
        if len(b["coords"]) != 27:
            continue
        toks = b["id"].split("_")
        sp_code = id_to_spcode.get(toks[0], toks[3] if len(toks) > 3 else toks[0])
        sp_jaw[sp_code].append(b["coords"][:, :2])

    nwk, _ = parse_nexus_newick(rdir / "05_BEAST_RAxML.tre")
    parent, depth, tip_to_id = at.parse_newick_depths(nwk)

    tree_sp_body = sorted(sp for sp in sp_body if sp in tip_to_id)
    tree_body = eng.Tree(parent, depth, {sp: tip_to_id[sp] for sp in tree_sp_body})
    res_body = run_standard_gm_engine(sp_body, tree_body, tree_sp_body, inner_selector="radius")

    tree_sp_jaw = sorted(sp for sp in sp_jaw if sp in tip_to_id)
    tree_jaw = eng.Tree(parent, depth, {sp: tip_to_id[sp] for sp in tree_sp_jaw})
    res_jaw = run_standard_gm_engine(sp_jaw, tree_jaw, tree_sp_jaw, inner_selector="radius")

    res_body["lower_pharyngeal_jaw"] = {
        "n_species_tree": len(tree_sp_jaw),
        "n_specimens": res_jaw["n_specimens"],
        "thompson7_all": res_jaw["variance"]["thompson7_all"],
        "pca7_all": res_jaw["variance"]["pca7_all"],
        "lambda_t7": res_jaw["models"]["thompson7"]["lambda"],
        "loo_ratio_full": res_jaw["loo"]["ratio_full_vs_consensus"],
        "loo_ratio_t7": res_jaw["loo"]["ratio_t7_vs_consensus"],
        "p_full_vs_consensus": res_jaw["loo"]["p_full_vs_consensus"],
    }

    hyps = {
        "CR1": {
            "body_t7_variance_share": res_body["variance"]["thompson7_all"],
            "body_pca7_variance_share": res_body["variance"]["pca7_all"],
            "pass": bool(res_body["variance"]["thompson7_all"] > 0.55),
        },
        "CR2": {
            "body_loo_ratio_full": res_body["loo"]["ratio_full_vs_consensus"],
            "body_p_full_vs_consensus": res_body["loo"]["p_full_vs_consensus"],
            "pass": bool(res_body["loo"]["ratio_full_vs_consensus"] < 0.65 and res_body["loo"]["p_full_vs_consensus"] < 1e-10),
        },
        "CR3": {
            "share_of_full_gain_in_t7": res_body["loo"]["share_of_full_gain_in_t7"],
            "body_loo_ratio_t7": res_body["loo"]["ratio_t7_vs_consensus"],
            "body_p_t7_vs_consensus": res_body["loo"]["p_t7_vs_consensus"],
            "pass": bool(res_body["loo"]["share_of_full_gain_in_t7"] > 0.60 and res_body["loo"]["p_t7_vs_consensus"] < 1e-10),
        },
        "CR4": {
            "jaw_t7_variance_share": res_jaw["variance"]["thompson7_all"],
            "jaw_loo_ratio_full": res_jaw["loo"]["ratio_full_vs_consensus"],
            "jaw_p_full_vs_consensus": res_jaw["loo"]["p_full_vs_consensus"],
            "pass": bool(
                res_jaw["variance"]["thompson7_all"] > 0.50
                and res_jaw["loo"]["ratio_full_vs_consensus"] < 0.75
                and res_jaw["loo"]["p_full_vs_consensus"] < 1e-10
            ),
        },
    }
    return finalize_group("cichlids_ronco_gm", H, res_body, hyps)


# =============================================================================
# 8. CHARACIFORMES 24-LM RADIATION (Burns & Sidlauskas 2019 Evolution)
# =============================================================================
def score_characiformes_burns_gm(H: dict) -> dict:
    print("\n=== [8/8] Scoring Characiformes 24-LM radiation (Burns & Sidlauskas 2019) ===")
    bdir = CACHE / "burns2019"
    blocks = parse_tps_blocks(bdir / "Disparity_Data.TPS")
    sp_coords = defaultdict(list)
    for b in blocks:
        if len(b["coords"]) == 24 and b["image"]:
            sp = b["image"].strip().replace(" ", "_")
            sp_coords[sp].append(b["coords"][:, :2])

    nwk_raw, _ = parse_nexus_newick(bdir / "trees_final.nexus")
    # Strip family number prefixes in tree tip labels (e.g. '10_Characidium_laterale' -> 'Characidium_laterale')
    nwk_clean = re.sub(r"(?<=[\(\,])[0-9]+_([A-Za-z0-9_]+)(?=[\:\,\)])", r"\1", nwk_raw)
    parent, depth, tip_to_id = at.parse_newick_depths(nwk_clean)
    tree_species = sorted(sp for sp in sp_coords if sp in tip_to_id)
    tree = eng.Tree(parent, depth, {sp: tip_to_id[sp] for sp in tree_species})

    res = run_standard_gm_engine(sp_coords, tree, tree_species, inner_selector="radius")

    hyps = {
        "CB1": {
            "t7_variance_share": res["variance"]["thompson7_all"],
            "t7_tree_variance_share": res["variance"]["thompson7_tree"],
            "pca7_variance_share": res["variance"]["pca7_all"],
            "pass": bool(res["variance"]["thompson7_tree"] > 0.55 or res["variance"]["thompson7_all"] > 0.55),
        },
        "CB2": {
            "lambda_t7": res["models"]["thompson7"]["lambda"],
            "lambda_full": res["models"]["full"]["lambda"],
            "best_model_t7": res["models"]["thompson7"]["best"],
            "pass": bool(res["models"]["thompson7"]["kind"] != "WN" and res["models"]["thompson7"]["lambda"] > 0.70),
        },
        "CB3": {
            "ratio_full_vs_consensus": res["loo"]["ratio_full_vs_consensus"],
            "ratio_t7_vs_consensus": res["loo"]["ratio_t7_vs_consensus"],
            "p_full_vs_consensus": res["loo"]["p_full_vs_consensus"],
            "pass": bool(res["loo"]["ratio_full_vs_consensus"] < 0.75 and res["loo"]["p_full_vs_consensus"] < 1e-8),
        },
        "CB4": {
            "unexplained_inner_head_eye": res["landmark_residuals"]["unexplained_inner"],
            "unexplained_outer_contour": res["landmark_residuals"]["unexplained_outer"],
            "pass": bool(res["landmark_residuals"]["inner_exceeds_outer"]),
        },
    }
    return finalize_group("characiformes_burns_gm", H, res, hyps)


# =============================================================================
# 9. VARANUS MONITOR LIZARDS (Brennan et al. 2021 Syst Biol)
# =============================================================================
def score_varanus_lizards(H: dict) -> dict:
    print("\n=== [9/11] Scoring Varanus monitor lizards (Brennan et al. 2021 Syst Biol) ===")
    vdir = CACHE / "varanus2021"

    def load_varanus_csv(fname: str, k_lm: int):
        by_sp = defaultdict(list)
        for r in csv.DictReader((vdir / fname).open(encoding="utf-8", errors="replace")):
            raw_t = r["Taxon"].strip()
            if raw_t.startswith("Varanus "):
                sp = raw_t[len("Varanus "):].strip().replace(" ", "_")
            else:
                sp = raw_t.replace(" ", "_")
            try:
                pts = np.array([[float(r[f"{k}x"]), float(r[f"{k}y"])] for k in range(1, k_lm + 1)], dtype=float)
            except (ValueError, KeyError):
                continue
            by_sp[sp].append(pts)
        return by_sp

    sp_dorsal = load_varanus_csv("Head_shape_dorsal.csv", 33)
    sp_lateral = load_varanus_csv("Head_shape_lateral.csv", 10)

    nwk = re.sub(r"\s+", "", (vdir / "Tree.tre").read_text(encoding="utf-8", errors="replace"))
    parent, depth, tip_to_id = at.parse_newick_depths(nwk)
    tree_sp_d = sorted(sp for sp in sp_dorsal if sp in tip_to_id)
    max_v = max(depth[tip_to_id[sp]] for sp in tree_sp_d)
    for sp in tip_to_id:
        depth[tip_to_id[sp]] = max_v
    tree_d = eng.Tree(parent, depth, {sp: tip_to_id[sp] for sp in tree_sp_d})
    res_d = run_standard_gm_engine(sp_dorsal, tree_d, tree_sp_d, inner_selector="radius")

    tree_sp_l = sorted(sp for sp in sp_lateral if sp in tip_to_id)
    tree_l = eng.Tree(parent, depth, {sp: tip_to_id[sp] for sp in tree_sp_l})
    res_l = run_standard_gm_engine(sp_lateral, tree_l, tree_sp_l, inner_selector="radius")

    # VL4: Within-species intraspecific variation vs between-species divergence on 33-LM dorsal head
    M0 = np.array(res_d["consensus"])
    B = res_d["B"]
    Rg = res_d["Rg"]
    Qr, _ = np.linalg.qr(Rg)
    within_devs = []
    for sp, specs in sorted(sp_dorsal.items()):
        if len(specs) >= 2:
            opa_specs = np.array([at.opa(at.centre_scale(s), M0) for s in specs])
            sp_m = opa_specs.mean(0)
            for s in opa_specs:
                dv = (s - sp_m).reshape(-1)
                dv = dv - (dv @ Qr) @ Qr.T
                within_devs.append(dv)
    W_mat = np.array(within_devs)
    Z_w = eng.project(B, W_mat)
    E_w = W_mat - Z_w @ B.T
    D_all = res_d["D_all"]
    Z_all = eng.project(B, D_all)
    E_all = D_all - Z_all @ B.T
    cos_t7 = principal_axis_cos(Z_w @ B.T, Z_all @ B.T)
    cos_res = principal_axis_cos(E_w, E_all)

    res_d["lateral_head_summary"] = {
        "n_species_all": res_l["n_species_all"],
        "n_specimens": res_l["n_specimens"],
        "thompson7_all": res_l["variance"]["thompson7_all"],
        "pca7_all": res_l["variance"]["pca7_all"],
        "lambda_t7": res_l["models"]["thompson7"]["lambda"],
        "best_model_t7": res_l["models"]["thompson7"]["best"],
        "loo_ratio_full": res_l["loo"]["ratio_full_vs_consensus"],
        "loo_ratio_t7": res_l["loo"]["ratio_t7_vs_consensus"],
    }
    res_d["intraspecific"] = {"cos_thompson7": cos_t7, "cos_residual": cos_res}

    hyps = {
        "VL1": {
            "dorsal_t7_share": res_d["variance"]["thompson7_all"],
            "lateral_t7_share": res_l["variance"]["thompson7_all"],
            "pass": bool(res_d["variance"]["thompson7_all"] > 0.55 and res_l["variance"]["thompson7_all"] > 0.55),
        },
        "VL2": {
            "dorsal_lambda_t7": res_d["models"]["thompson7"]["lambda"],
            "lateral_lambda_t7": res_l["models"]["thompson7"]["lambda"],
            "dorsal_best": res_d["models"]["thompson7"]["best"],
            "lateral_best": res_l["models"]["thompson7"]["best"],
            "pass": bool(
                res_d["models"]["thompson7"]["kind"] != "WN"
                and res_l["models"]["thompson7"]["kind"] != "WN"
                and res_d["models"]["thompson7"]["lambda"] > 0.70
                and res_l["models"]["thompson7"]["lambda"] > 0.70
            ),
        },
        "VL3": {
            "dorsal_loo_ratio_full": res_d["loo"]["ratio_full_vs_consensus"],
            "dorsal_loo_ratio_t7": res_d["loo"]["ratio_t7_vs_consensus"],
            "p_full_vs_consensus": res_d["loo"]["p_full_vs_consensus"],
            "p_t7_vs_consensus": res_d["loo"]["p_t7_vs_consensus"],
            "pass": bool(res_d["loo"]["ratio_full_vs_consensus"] < 0.85 and res_d["loo"]["p_full_vs_consensus"] < 1e-5),
        },
        "VL4": {
            "cos_thompson7": cos_t7,
            "cos_residual": cos_res,
            "pass": bool(cos_t7 > cos_res),
        },
    }
    return finalize_group("varanus_lizards", H, res_d, hyps)


# =============================================================================
# 10. AUSTRALIAN FROGS (Sherratt et al. 2017 Evol Biol)
# =============================================================================
def score_australian_frogs(H: dict) -> dict:
    print("\n=== [10/11] Scoring Australian Anura frogs (Sherratt et al. 2017 Evol Biol) ===")
    adir = CACHE / "anura2016"
    blocks = parse_tps_blocks(adir / "supp2.tps")
    sp_coords = {}
    for b in blocks:
        if len(b["coords"]) == 55 and b["id"]:
            sp = b["id"].strip()
            sp_coords[sp] = [b["coords"][:, :2]]

    nwk = re.sub(r"\s+", "", (adir / "supp4.tre").read_text(encoding="utf-8", errors="replace"))
    parent, depth, tip_to_id = at.parse_newick_depths(nwk)
    # Match either exact or with '.' replaced by '_'
    norm_to_tip = {t.replace("_", "."): t for t in tip_to_id}
    sp_coords_mapped = {}
    for sp, pts in sp_coords.items():
        key_dot = sp.replace("_", ".")
        if key_dot in norm_to_tip:
            sp_coords_mapped[norm_to_tip[key_dot]] = pts
        elif sp in tip_to_id:
            sp_coords_mapped[sp] = pts

    tree_species = sorted(sp for sp in sp_coords_mapped if sp in tip_to_id)
    max_d = max(depth[tip_to_id[sp]] for sp in tree_species)
    for sp in tree_species:
        depth[tip_to_id[sp]] = max_d
    tree = eng.Tree(parent, depth, {sp: tip_to_id[sp] for sp in tree_species})

    res = run_standard_gm_engine(sp_coords_mapped, tree, tree_species, inner_selector="radius")

    hyps = {
        "AF1": {
            "t7_variance_share": res["variance"]["thompson7_all"],
            "pca7_variance_share": res["variance"]["pca7_all"],
            "pass": bool(res["variance"]["thompson7_all"] > 0.70),
        },
        "AF2": {
            "lambda_t7": res["models"]["thompson7"]["lambda"],
            "lambda_full": res["models"]["full"]["lambda"],
            "best_model_t7": res["models"]["thompson7"]["best"],
            "pass": bool(res["models"]["thompson7"]["kind"] != "WN" and res["models"]["thompson7"]["lambda"] > 0.50),
        },
        "AF3": {
            "ratio_full_vs_consensus": res["loo"]["ratio_full_vs_consensus"],
            "ratio_t7_vs_consensus": res["loo"]["ratio_t7_vs_consensus"],
            "p_full_vs_consensus": res["loo"]["p_full_vs_consensus"],
            "pass": bool(res["loo"]["ratio_full_vs_consensus"] < 0.85 and res["loo"]["p_full_vs_consensus"] < 1e-6),
        },
        "AF4": {
            "share_of_full_gain_in_t7": res["loo"]["share_of_full_gain_in_t7"],
            "pass": bool(res["loo"]["share_of_full_gain_in_t7"] > 0.70),
        },
    }
    return finalize_group("australian_frogs", H, res, hyps)


# =============================================================================
# 11. ANTILLEAN GESNERIACEAE FLOWERS (Joly et al. 2018 Evolution)
# =============================================================================
def score_antillean_flowers(H: dict) -> dict:
    print("\n=== [11/11] Scoring Antillean Gesneriaceae angiosperm corollas (Joly et al. 2018 Evolution) ===")
    jdir = CACHE / "joly2018"
    with zipfile.ZipFile(jdir / "scriptsAndData.zip") as z:
        tre_txt = z.read("scriptsAndData/BestTree.tre").decode("utf-8", errors="replace")
        nts_lines = z.read("scriptsAndData/landmark_data.nts").decode("utf-8", errors="replace").splitlines()
        names_lines = z.read("scriptsAndData/names.txt").decode("utf-8", errors="replace").splitlines()[1:]

    # Map picture prefix -> CodeSpecies (e.g. 'GES_acaulis' -> tree tip 'Gacaulis', 'RHY_rupincola' -> 'Rrupincola', 'PHE_corymbosa' -> 'Pcorymbosa')
    pic_to_tree_sp = {}
    for l in names_lines:
        if not l.strip():
            continue
        fn, code_sp = l.strip().split()
        stem = fn.replace(".jpg", "")
        parts = code_sp.split("_", 1)
        tree_sp = parts[0][0].upper() + parts[1] if len(parts) == 2 else code_sp
        pic_to_tree_sp[stem] = tree_sp

    pic_names = nts_lines[1].split()
    coord_lines = [l.strip() for l in nts_lines[2:] if l.strip()]
    assert len(pic_names) == len(coord_lines), f"Mismatch {len(pic_names)} vs {len(coord_lines)}"

    sp_coords = defaultdict(list)
    for p_name, c_line in zip(pic_names, coord_lines):
        stem = p_name.replace(".jpg", "")
        base_stem = stem[:-2] if stem.endswith("_2") else stem
        sp = pic_to_tree_sp.get(base_stem)
        if not sp:
            # Fallback from filename prefix (e.g. GES_acaulis_...)
            toks = base_stem.split("_")
            if len(toks) >= 2:
                sp = toks[0][0].upper() + toks[1]
        vals = [float(x) for x in c_line.split()]
        if len(vals) == 64 and sp:
            pts = np.array(vals, dtype=float).reshape(32, 2)
            sp_coords[sp].append(pts)

    # Parse BestTree.tre
    tmp_tre = jdir / "BestTree_extracted.tre"
    tmp_tre.write_text(tre_txt, encoding="utf-8")
    nwk, _ = parse_nexus_newick(tmp_tre)
    parent, depth, tip_to_id = at.parse_newick_depths(nwk)
    tree_species = sorted(sp for sp in sp_coords if sp in tip_to_id)
    max_d = max(depth[tip_to_id[sp]] for sp in tree_species)
    for sp in tree_species:
        depth[tip_to_id[sp]] = max_d
    tree = eng.Tree(parent, depth, {sp: tip_to_id[sp] for sp in tree_species})

    res = run_standard_gm_engine(sp_coords, tree, tree_species, inner_selector="radius")

    # FP4: Within-species floral variation vs between-species divergence in Thompson-7
    M0 = np.array(res["consensus"])
    B = res["B"]
    Rg = res["Rg"]
    Qr, _ = np.linalg.qr(Rg)
    within_devs = []
    for sp, specs in sorted(sp_coords.items()):
        if len(specs) >= 2:
            opa_specs = np.array([at.opa(at.centre_scale(s), M0) for s in specs])
            sp_m = opa_specs.mean(0)
            for s in opa_specs:
                dv = (s - sp_m).reshape(-1)
                dv = dv - (dv @ Qr) @ Qr.T
                within_devs.append(dv)
    W_mat = np.array(within_devs)
    Z_w = eng.project(B, W_mat)
    E_w = W_mat - Z_w @ B.T
    D_all = res["D_all"]
    Z_all = eng.project(B, D_all)
    E_all = D_all - Z_all @ B.T
    cos_t7 = principal_axis_cos(Z_w @ B.T, Z_all @ B.T)
    cos_res = principal_axis_cos(E_w, E_all)
    res["intraspecific"] = {"cos_thompson7": cos_t7, "cos_residual": cos_res}

    hyps = {
        "FP1": {
            "t7_variance_share": res["variance"]["thompson7_all"],
            "pca7_variance_share": res["variance"]["pca7_all"],
            "pass": bool(res["variance"]["thompson7_all"] > 0.80),
        },
        "FP2": {
            "lambda_t7": res["models"]["thompson7"]["lambda"],
            "lambda_full": res["models"]["full"]["lambda"],
            "best_model_t7": res["models"]["thompson7"]["best"],
            "pass": bool(res["models"]["thompson7"]["kind"] != "WN" and res["models"]["thompson7"]["lambda"] > 0.50),
        },
        "FP3": {
            "ratio_full_vs_consensus": res["loo"]["ratio_full_vs_consensus"],
            "ratio_t7_vs_consensus": res["loo"]["ratio_t7_vs_consensus"],
            "p_full_vs_consensus": res["loo"]["p_full_vs_consensus"],
            "p_t7_vs_consensus": res["loo"]["p_t7_vs_consensus"],
            "pass": bool(
                (res["loo"]["ratio_full_vs_consensus"] < 0.90 and res["loo"]["p_full_vs_consensus"] < 0.01)
                or (res["loo"]["ratio_t7_vs_consensus"] < 0.90 and res["loo"]["p_t7_vs_consensus"] < 0.01)
            ),
        },
        "FP4": {
            "cos_thompson7": cos_t7,
            "cos_residual": cos_res,
            "pass": bool(cos_t7 > 0.70 and cos_t7 > cos_res),
        },
    }
    return finalize_group("antillean_flowers", H, res, hyps)


# =============================================================================
# REPORTING & PLOTTING
# =============================================================================
def finalize_group(gid: str, H: dict, res: dict, hyps: dict) -> dict:
    for hid, val in hyps.items():
        print(f"    {hid}: pass={val['pass']} -> {json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in val.items() if k != 'pass'})}")

    n_pass = sum(1 for v in hyps.values() if v["pass"])
    n_prim = sum(1 for h in H["hypotheses"] if hyps[h["id"]]["pass"] and h["primary"])
    brier = float(np.mean([(h["p"] - float(hyps[h["id"]]["pass"])) ** 2 for h in H["hypotheses"]]))

    out = {
        "id": gid,
        "group": H["group"],
        "portfolio_id": H.get("portfolio_id"),
        "scored_on": dt.date.today().isoformat(),
        "n_specimens": res["n_specimens"],
        "n_species_all": res["n_species_all"],
        "n_species_tree": res.get("n_species_tree", 0),
        "k_lm": res["k_lm"],
        "variance": res["variance"],
        "models": res.get("models", {}),
        "loo": res.get("loo", {}),
        "landmark_residuals": res["landmark_residuals"],
        "hypotheses": hyps,
        "n_pass": n_pass,
        "n_total": len(hyps),
        "n_primary_pass": n_prim,
        "brier": brier,
        "consensus": res["consensus"],
        "per_species_loo": res.get("per_species_loo", []),
    }
    for extra_k in ("hindwing_summary", "sexual_dimorphism_t7_share", "alignment_caste_vs_species", "n_paired_caste_species", "intraspecific", "lower_pharyngeal_jaw", "lateral_head_summary"):
        if extra_k in res:
            out[extra_k] = res[extra_k]

    RES_DIR.mkdir(parents=True, exist_ok=True)
    DOC_DIR.mkdir(parents=True, exist_ok=True)
    (RES_DIR / f"{gid}.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    write_group_md(gid, H, out)
    make_group_fig(gid, H, out)
    return out


def write_group_md(gid: str, H: dict, R: dict) -> None:
    v = R["variance"]
    loo = R.get("loo", {})
    m = R.get("models", {}).get("thompson7", {})
    L = [
        f"# {R['group']}",
        "",
        f"Scored {R['scored_on']} by `thompson/portfolio/run_all_scouted_clades.py`; hypotheses were pre-registered and SHA-256 locked in `thompson/portfolio/groups/{gid}_hypotheses.json` (`scout_preregistration_manifest.json`).",
        "",
        f"- **Sample:** {R['n_specimens']:,} specimens across {R['n_species_all']} species/OTUs ({R['n_species_tree']} on phylogeny), **{R['k_lm']} homologous 2D landmarks**.",
        f"- **Data source:** {H['data']['landmarks']}",
        f"- **Phylogeny:** {H['data']['phylogeny']}",
        "",
        f"![figure](./fig_{gid}.png)",
        "",
        "## What Thompson's method shows in this clade",
        "",
        H["what_thompson_specifically_adds"],
        "",
        "| Metric | Marine Reef Fishes (743 spp, 17 LMs) | Hornet Wings (30 spp, 19 LMs) | This Clade |",
        "|---|---|---|---|",
        f"| **Variance captured by 7 a-priori Thompson modes** | 57.9% | 38.8% | **{100*v['thompson7_all']:.1f}%** |",
        f"| **Variance captured by fitted 7-D PCA** | 86.2% | 95.5% | **{100*v['pca7_all']:.1f}%** |",
        f"| **Pagel's $\\lambda$ (Thompson-7 / Full)** | 0.959 / 0.924 | 0.533 (WN) | **{m.get('lambda', float('nan')):.3f}** ({m.get('best', '-')}) / {R.get('models', {}).get('full', {}).get('lambda', float('nan')):.3f} |",
        f"| **LOO Error Ratio vs. Consensus (Full / Thompson-7)** | 0.36 (<50 Myr) | 1.00 | **{loo.get('ratio_full_vs_consensus', float('nan')):.2f}** ($p={loo.get('p_full_vs_consensus', float('nan')):.2g}$) / **{loo.get('ratio_t7_vs_consensus', float('nan')):.2f}** |",
        f"| **Share of Full Prediction Gain in Thompson-7** | 55.5% | — | **{100*loo.get('share_of_full_gain_in_t7', float('nan')):.1f}%** |",
        "",
        "## Pre-registered hypotheses & verdicts",
        "",
        "| ID | Primary | Statement | Registered P | Realised Result | Verdict |",
        "|---|---|---|---|---|---|",
    ]
    for h in H["hypotheses"]:
        r = R["hypotheses"][h["id"]]
        summary = json.dumps({k: (round(val, 4) if isinstance(val, float) else val) for k, val in r.items() if k != "pass"})
        L.append(f"| {h['id']} | {'yes' if h['primary'] else 'no'} | {h['statement']} | {100*h['p']:.0f}% | `{summary}` | **{'✓' if r['pass'] else '✗'}** |")

    L += [
        "",
        f"**{R['n_pass']}/{R['n_total']} hypotheses supported ({R['n_primary_pass']} primary). Brier score: {R['brier']:.3f}.**",
        "",
    ]
    (DOC_DIR / f"{gid}.md").write_text("\n".join(L), encoding="utf-8")


def make_group_fig(gid: str, H: dict, R: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.8))

    # Panel A: Consensus shape + per-landmark unexplained fraction
    ax = axes[0]
    M0 = np.array(R["consensus"])
    fr = np.array(R["landmark_residuals"]["per_landmark"])
    sc = ax.scatter(M0[:, 0], M0[:, 1], c=fr, cmap="magma_r", s=65, vmin=0, vmax=1, edgecolors="k", linewidths=0.7)
    plt.colorbar(sc, ax=ax, label="Unexplained fraction after Thompson-7")
    ax.set_aspect("equal")
    ax.axis("off")
    u_in = R["landmark_residuals"]["unexplained_inner"]
    u_out = R["landmark_residuals"]["unexplained_outer"]
    ax.set_title(
        f"A. Consensus ({R['k_lm']} LMs) & T7 Residuals:\nInner {100*u_in:.1f}% vs. Outer {100*u_out:.1f}%",
        fontsize=9.5, fontweight="bold"
    )

    # Panel B: Variance Share & LOO Gain Comparison
    ax = axes[1]
    cats = ["Fish Bodies\n(17 LMs)", "Hornet Wings\n(19 LMs)", f"This Clade\n({R['k_lm']} LMs)"]
    t7_v = [57.9, 38.8, 100 * R["variance"]["thompson7_all"]]
    pca_v = [86.2, 95.5, 100 * R["variance"]["pca7_all"]]
    x = np.arange(3)
    w = 0.35
    ax.bar(x - w / 2, t7_v, width=w, color="#2563eb", label="7 A-Priori Thompson Modes")
    ax.bar(x + w / 2, pca_v, width=w, color="#9333ea", label="Fitted 7-D PCA")
    for i in range(3):
        ax.text(x[i] - w / 2, t7_v[i] + 1.5, f"{t7_v[i]:.1f}%", ha="center", va="bottom", fontsize=8, fontweight="bold", color="#2563eb")
        ax.text(x[i] + w / 2, pca_v[i] + 1.5, f"{pca_v[i]:.1f}%", ha="center", va="bottom", fontsize=8, fontweight="bold", color="#9333ea")
    ax.set_xticks(x)
    ax.set_xticklabels(cats, fontsize=8.5)
    ax.set_ylim(0, 112)
    ax.set_ylabel("% Between-Species Variance Captured")
    ax.legend(fontsize=8, loc="upper left")
    ax.set_title("B. How Thompsonian Is the Clade?", fontsize=9.5, fontweight="bold")

    # Panel C: Leave-One-Species-Out RMS sorted by tau
    ax = axes[2]
    recs = R.get("per_species_loo", [])
    if recs:
        order = np.argsort([r["tau"] for r in recs])
        for k_rms, col, lab in [
            ("rms_full", "#16a34a", "Full PCA-95%"),
            ("rms_thompson7", "#2563eb", "Thompson-7"),
            ("rms_nearest", "#f97316", "Copy nearest"),
            ("rms_consensus", "#64748b", "Consensus"),
        ]:
            ax.plot(range(len(recs)), [recs[i][k_rms] for i in order], "-", color=col, lw=1.2, alpha=0.85, label=lab)
        ax.set_xlabel(f"Held-out species sorted by phylogenetic horizon τ (n={len(recs)})")
        ax.set_ylabel("Procrustes RMS Error")
        ax.legend(fontsize=7.5, loc="upper left")
        ax.grid(alpha=0.25)
        loo = R["loo"]
        ax.set_title(
            f"C. LOO Phylogenetic Prediction (n={len(recs)}):\n"
            f"Full/Cons = {loo['ratio_full_vs_consensus']:.2f} (p={loo['p_full_vs_consensus']:.2g}), "
            f"T7/Cons = {loo['ratio_t7_vs_consensus']:.2f}",
            fontsize=9.5, fontweight="bold"
        )

    fig.suptitle(
        f"{R['group'].split('(')[0].strip()}: {R['n_pass']}/{R['n_total']} Hypotheses Supported (Brier {R['brier']:.3f})",
        fontsize=11, fontweight="bold", y=1.02
    )
    fig.savefig(DOC_DIR / f"fig_{gid}.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    print("[0] Verifying SHA-256 digests of all 8 pre-registered hypothesis files")
    man = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    for rel_path, expected_hash in man["hypothesis_files_sha256"].items():
        actual_hash = sha256(REPO_ROOT / rel_path)
        if actual_hash != expected_hash:
            print(f"    HASH MISMATCH for {rel_path}")
            return 2
        print(f"    {rel_path}: sha256 OK ({actual_hash[:12]}…)")

    scorers = [
        ("drosophilidae_wings", score_drosophilidae_wings),
        ("pheidole_ants", score_pheidole_ants),
        ("morpho_butterflies", score_morpho_butterflies),
        ("waterfowl_beaks", score_waterfowl_beaks),
        ("leporidae_skulls", score_leporidae_skulls),
        ("galeommatoidea_bivalves", score_galeommatoidea_bivalves),
        ("cichlids_ronco_gm", score_cichlids_ronco_gm),
        ("characiformes_burns_gm", score_characiformes_burns_gm),
        ("varanus_lizards", score_varanus_lizards),
        ("australian_frogs", score_australian_frogs),
        ("antillean_flowers", score_antillean_flowers),
    ]

    all_out = {}
    tot_pass, tot_hyps = 0, 0
    all_brier = []
    for gid, fn in scorers:
        H = json.loads((GROUPS_DIR / f"{gid}_hypotheses.json").read_text(encoding="utf-8"))
        r = fn(H)
        all_out[gid] = r
        tot_pass += r["n_pass"]
        tot_hyps += r["n_total"]
        for h in H["hypotheses"]:
            all_brier.append((h["p"] - float(r["hypotheses"][h["id"]]["pass"])) ** 2)

    print(f"\n=== ALL 8 SCOUTED CLADES SCORED: {tot_pass}/{tot_hyps} hypotheses supported ({100*tot_pass/tot_hyps:.1f}%), overall Brier = {np.mean(all_brier):.3f} ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())

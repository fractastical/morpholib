#!/usr/bin/env python3
"""
Bee forewings across 10 families (~110 Myr) + two Eocene fossil tribes: tests B1-B9 registered in
thompson/portfolio/groups/bee_wings_hypotheses.json (committed, with the dated tribe tree, before this ran).

Outputs: thompson/portfolio/results/bee_wings.json, docs/portfolio/bee_wings.md, docs/portfolio/fig_bee_wings.png
"""
from __future__ import annotations

import datetime as dt
import importlib.util
import json
import sys
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

warnings.filterwarnings("ignore")
REPO_ROOT = Path(__file__).resolve().parents[3]
XLSX = REPO_ROOT / ".thompson_cache" / "antscan" / "bee_fossil_s004.xlsx"
HYP = REPO_ROOT / "thompson" / "portfolio" / "groups" / "bee_wings_hypotheses.json"
TREE_JSON = REPO_ROOT / "datasets" / "thompson" / "bees" / "bee_tribe_tree_dated.json"
OUT_JSON = REPO_ROOT / "thompson" / "portfolio" / "results" / "bee_wings.json"
OUT_MD = REPO_ROOT / "docs" / "portfolio" / "bee_wings.md"
OUT_FIG = REPO_ROOT / "docs" / "portfolio" / "fig_bee_wings.png"
K_LM = 18
MIN_WINGS = 3
N_END = 5
CHI2_7_95 = 14.07


def load_module(name, rel):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / rel)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


eng = load_module("eng", "thompson/run_phylogenetic_prediction_engine.py")
at = eng.at


def pcs(D, k=None, var=None):
    mu = D.mean(0); U, S, Vt = np.linalg.svd(D - mu, full_matrices=False)
    if var is not None:
        k = int(np.searchsorted(np.cumsum(S ** 2 / np.sum(S ** 2)), var) + 1)
    return mu, Vt[:k].T


def best_model(tree, tips, X):
    f = eng.fit_models(tree, tips, X)
    k = min(f, key=lambda a: f[a]["AICc"])
    return f[k]["model"], k, f


def main() -> int:
    H = json.loads(HYP.read_text()); TJ = json.loads(TREE_JSON.read_text())
    print("[1] wings -> unit means")
    import openpyxl
    rows = [r for r in openpyxl.load_workbook(XLSX, read_only=True)["Scores"].iter_rows(values_only=True) if r and r[0]][1:]
    by_unit, fam_of = defaultdict(list), {}
    for r in rows:
        sf, fam, sub, tribe = r[:4]
        P = np.array(r[4:4 + 2 * K_LM], dtype=float).reshape(K_LM, 2)
        by_unit[tribe].append(P); fam_of[tribe] = (sf, fam, sub)
    fossils = {k: v for k, v in TJ["fossil_targets"].items()}
    units = sorted(u for u in by_unit if u not in fossils and len(by_unit[u]) >= MIN_WINGS and u in TJ["tips"])
    dropped = sorted(u for u in by_unit if u not in fossils and u not in units)
    print(f"    {sum(len(v) for v in by_unit.values())} wings, {len(by_unit)} units; {len(units)} extant units kept (dropped <{MIN_WINGS} wings: {dropped}); fossils {list(fossils)}")
    allW = np.array([w for u in units for w in by_unit[u]]); lab = np.array([u for u in units for _ in by_unit[u]])
    Yw = at.gpa(allW)
    means = np.array([Yw[lab == u].mean(0) for u in units])
    Y = at.gpa(means); M0 = at.centre_scale(Y.mean(0)); Y_opa = np.array([at.opa(y, M0) for y in Y])
    n = len(units); D = (Y_opa - M0).reshape(n, -1)
    # fossil unit means, superimposed on the extant consensus
    fossil_means = {}
    for fz in fossils:
        W = at.gpa(np.array(by_unit[fz])) if len(by_unit[fz]) > 1 else at.centre_scale(np.array(by_unit[fz]))
        fossil_means[fz] = at.opa(at.centre_scale(W.mean(0)), M0)
    within = {u: float(np.sqrt(((Yw[lab == u] - Yw[lab == u].mean(0)) ** 2).sum(-1).mean())) for u in units}

    print("[2] Thompson-7 basis, variance shares, residual location")
    B, _ = eng.thompson_mode_basis(M0)
    Z = eng.project(B, D); E = D - Z @ B.T
    var_t7 = 1 - float(np.sum(E ** 2) / np.sum(D ** 2))
    mu7, V7 = pcs(D, k=7); var_pca7 = float(np.sum(((D - mu7) @ V7) ** 2) / np.sum((D - mu7) ** 2))
    xs = M0[:, 0]; end_a, end_b = np.argsort(xs)[-N_END:], np.argsort(xs)[:N_END]
    mid = np.setdiff1d(np.arange(K_LM), np.concatenate([end_a, end_b]))
    fr = (E.reshape(n, K_LM, 2) ** 2).sum(-1).mean(0) / (D.reshape(n, K_LM, 2) ** 2).sum(-1).mean(0)
    b6 = {"unexplained_end_a": float(fr[end_a].mean()), "unexplained_end_b": float(fr[end_b].mean()), "unexplained_middle": float(fr[mid].mean()), "per_landmark": fr.round(3).tolist(),
          "end_a": [int(i) for i in end_a], "end_b": [int(i) for i in end_b]}
    b6["pass"] = bool(b6["unexplained_middle"] > max(b6["unexplained_end_a"], b6["unexplained_end_b"]))
    print(f"    T7 {var_t7:.3f} (fish 0.579, hornets 0.388); PCA-7 {var_pca7:.3f}; residual middle {b6['unexplained_middle']:.2f} vs ends {b6['unexplained_end_a']:.2f}/{b6['unexplained_end_b']:.2f}")

    print("[3] dated tribe tree, process models")
    parent, depth, tip_to_id = at.parse_newick_depths(TJ["newick"])
    tree = eng.Tree(parent, depth, {u: tip_to_id[u] for u in units}); eng.Model.T = tree.T
    tips = np.array([tree.name_to_node[u] for u in units])
    m7, k7, f7 = best_model(tree, tips, Z)
    muF, VF = pcs(D, var=0.95); F = (D - muF) @ VF
    mF, kF, fF = best_model(tree, tips, F)
    lam7, lamF = f7["lambda"]["model"].theta, fF["lambda"]["model"].theta
    print(f"    root {tree.T:.0f} Myr; T7 best {m7.label()} (lambda {lam7:.2f}); full ({VF.shape[1]} PCs) best {mF.label()} (lambda {lamF:.2f})")
    b3 = {"best_model_full": mF.label(), "best_kind_full": kF, "lambda_full": float(lamF), "best_model_t7": m7.label(), "lambda_t7": float(lam7), "dAICc_full": {k: round(v["dAICc"], 1) for k, v in fF.items()},
          "pass": bool(kF != "WN" and lamF > 0.5)}

    print("[4] leave-one-unit-out")
    recs = []
    for i, u in enumerate(units):
        obs = np.array([tips[j] for j in range(n) if j != i]); tgt = tips[i]
        tau = float(tree.T - max(tree.C[tgt, o] for o in obs))
        keep = np.arange(n) != i; Dtr = D[keep]
        m, _, _ = best_model(tree, obs, Z[keep])
        mean7, _, _, _ = eng.predict(m, tree, obs, Z[keep], np.array([tree.h[tgt]]), tree.C[tgt][None, :])
        pred_t7 = M0.reshape(-1) + mean7[0] @ B.T
        mu, V = pcs(Dtr, var=0.95); Ftr = (Dtr - mu) @ V
        m, kf, _ = best_model(tree, obs, Ftr)
        meanF, _, _, _ = eng.predict(m, tree, obs, Ftr, np.array([tree.h[tgt]]), tree.C[tgt][None, :])
        pred_full = M0.reshape(-1) + mu + meanF[0] @ V.T
        mu, V = pcs(Dtr, k=7); P7 = (Dtr - mu) @ V
        m, _, _ = best_model(tree, obs, P7)
        meanP, _, _, _ = eng.predict(m, tree, obs, P7, np.array([tree.h[tgt]]), tree.C[tgt][None, :])
        pred_p7 = M0.reshape(-1) + mu + meanP[0] @ V.T
        cons = M0.reshape(-1) + Dtr.mean(0)
        j_near = max((j for j in range(n) if j != i), key=lambda j: tree.C[tgt, tips[j]])
        rm = lambda v: eng.shape_rms(v.reshape(K_LM, 2), Y_opa[i])  # noqa: E731
        recs.append({"unit": u, "family": fam_of[u][1], "n_wings": len(by_unit[u]), "tau_myr": tau, "nearest": units[j_near], "model_full": kf, "rms_full": rm(pred_full), "rms_thompson7": rm(pred_t7), "rms_pca7": rm(pred_p7),
                     "rms_consensus": rm(cons), "rms_nearest": rm(M0.reshape(-1) + D[j_near]), "rms_t7_oracle": rm(M0.reshape(-1) + Z[i] @ B.T), "within_rms": within[u]})
    A = {k: np.array([r[k] for r in recs]) for k in recs[0] if k.startswith("rms")}
    med = {k: float(np.median(v)) for k, v in A.items()}
    p4 = float(stats.wilcoxon(A["rms_full"] - A["rms_consensus"], alternative="less").pvalue)
    p5 = float(stats.wilcoxon(A["rms_full"] - A["rms_nearest"], alternative="less").pvalue)
    gf = float(np.sum(A["rms_consensus"] ** 2 - A["rms_full"] ** 2)); share = float(np.sum(A["rms_consensus"] ** 2 - A["rms_thompson7"] ** 2) / gf) if gf > 0 else float("nan")
    taus = np.array([r["tau_myr"] for r in recs])
    near_, far_ = A["rms_full"][taus <= 30], A["rms_full"][taus > 60]
    b7 = {"n_tau_le_30": int(len(near_)), "n_tau_gt_60": int(len(far_)), "median_full_le_30": float(np.median(near_)) if len(near_) else None, "median_full_gt_60": float(np.median(far_)) if len(far_) else None}
    b7["ratio"] = (b7["median_full_gt_60"] / b7["median_full_le_30"]) if b7["median_full_le_30"] else None; b7["pass"] = bool(b7["ratio"] is not None and b7["ratio"] >= 1.5)

    print("[5] fossil time slice (45 Ma)")
    b8 = {}
    for fz, cfg in fossils.items():
        out = {}
        for key, tipname in [("primary", cfg["lineage_tip"]), ("alt", cfg["alt_lineage_tip"])]:
            row, hv = tree.virtual_node(tree.name_to_node[tipname], cfg["age_ma"])
            m, _, _ = best_model(tree, tips, Z)
            mean7, s2, a7, R7 = eng.predict(m, tree, tips, Z, np.array([hv]), row[None, :])
            pred_t7 = (M0.reshape(-1) + mean7[0] @ B.T).reshape(K_LM, 2)
            m, _, _ = best_model(tree, tips, F)
            meanF, _, _, _ = eng.predict(m, tree, tips, F, np.array([hv]), row[None, :])
            pred_full = (M0.reshape(-1) + muF + meanF[0] @ VF.T).reshape(K_LM, 2)
            z_f = eng.project(B, (fossil_means[fz] - M0).reshape(1, -1))[0]
            out[key] = {"lineage": tipname, "rms_full": eng.shape_rms(pred_full, fossil_means[fz]), "rms_thompson7": eng.shape_rms(pred_t7, fossil_means[fz]),
                        "rms_consensus": eng.shape_rms(M0 + D.mean(0).reshape(K_LM, 2), fossil_means[fz]), "rms_extant_lineage_tip": eng.shape_rms(Y_opa[units.index(tipname)], fossil_means[fz]),
                        "mahal2_t7": eng.mahal2(z_f, mean7[0], float(s2[0]) * R7), "chi2_7_95": CHI2_7_95, "pred_landmarks_full": pred_full.tolist()}
        out["n_wings"] = len(by_unit[fz]); out["observed_landmarks"] = fossil_means[fz].tolist()
        out["closer_than_consensus_full"] = bool(out["primary"]["rms_full"] < out["primary"]["rms_consensus"])
        out["inside_95_t7"] = bool(out["primary"]["mahal2_t7"] < CHI2_7_95)
        b8[fz] = out
        print(f"    {fz}: full {out['primary']['rms_full']:.4f} | T7 {out['primary']['rms_thompson7']:.4f} | consensus {out['primary']['rms_consensus']:.4f} | extant Apini {out['primary']['rms_extant_lineage_tip']:.4f} | Mahal2 T7 {out['primary']['mahal2_t7']:.1f}")
    b8["pass"] = bool(all(b8[fz]["closer_than_consensus_full"] for fz in fossils))

    R = {"B1": {"t7_variance_share": var_t7, "pass": bool(var_t7 < 0.58)},
         "B2": {"pca7_variance_share": var_pca7, "pass": bool(var_pca7 > 0.85)},
         "B3": b3,
         "B4": {"median_full": med["rms_full"], "median_consensus": med["rms_consensus"], "ratio": med["rms_full"] / med["rms_consensus"], "p": p4, "pass": bool(p4 < 0.05 and med["rms_full"] / med["rms_consensus"] < 0.8)},
         "B5": {"median_nearest": med["rms_nearest"], "ratio": med["rms_full"] / med["rms_nearest"], "p_full_less": p5, "pass": bool(med["rms_full"] / med["rms_nearest"] > 0.9)},
         "B6": b6, "B7": b7, "B8": b8,
         "B9": {"share": share, "defined": bool(gf > 0), "pass": bool(gf > 0 and share < 0.555)}}
    for k, v in R.items():
        print(f"    {k}: pass={v['pass']} " + json.dumps({a: (round(b, 4) if isinstance(b, float) else b) for a, b in v.items() if a in ("t7_variance_share", "pca7_variance_share", "lambda_full", "best_model_full", "ratio", "p", "share", "unexplained_middle", "median_full_gt_60", "median_full_le_30")}))
    n_pass = sum(v["pass"] for v in R.values()); n_prim = sum(R[h["id"]]["pass"] for h in H["hypotheses"] if h["primary"])
    brier = float(np.mean([(h["p"] - float(R[h["id"]]["pass"])) ** 2 for h in H["hypotheses"]]))
    out = {"group": H["group"], "scored_on": dt.date.today().isoformat(), "n_units": n, "n_wings_extant": int(len(allW)), "units": units, "dropped": dropped, "tree": {"file": str(TREE_JSON.relative_to(REPO_ROOT)), "root_ma": tree.T},
           "variance": {"thompson7": var_t7, "pca7": var_pca7, "full_pcs_95pct": int(VF.shape[1])}, "loo_medians": med, "hypotheses": R, "n_pass": n_pass, "n_primary_pass": n_prim, "brier": brier,
           "per_unit": recs, "consensus": M0.tolist(), "unit_means": Y_opa.tolist(), "mode_scores": Z.tolist()}
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True); OUT_JSON.write_text(json.dumps(out, indent=1), encoding="utf-8")
    write_md(out, H); make_fig(out, H, M0, B, Y_opa, units, recs, fossil_means)
    return 0


def write_md(R, H):
    res = R["hypotheses"]
    L = [f"# Bee forewings across 10 families: Thompson's grid, phylogenetic signal at 110 Myr, and an Eocene fossil test", "",
         f"Scored {R['scored_on']} by `thompson/portfolio/groups/bee_wings.py`; hypotheses B1-B9, the unit rule and the dated tribe tree (`{R['tree']['file']}`, root {R['tree']['root_ma']:.0f} Ma) were committed beforehand. "
         f"{R['n_wings_extant']:,} extant wings in {R['n_units']} tribe-level units (dropped for < 3 wings: {', '.join(R['dropped'])}); fossils: Electrapini, Melikertini (Baltic amber, ~45 Ma).", "",
         "![figure](./fig_bee_wings.png)", "", "## Why this group", "", H["why_this_group"], "",
         "| | Fish bodies (743 spp) | Hornet wings (30 spp, 30 Myr) | Bee wings (42 tribes, 110 Myr) |", "|---|---|---|---|",
         f"| Variance in the 7 a-priori Thompson modes | 57.9% | 38.8% | **{100 * R['variance']['thompson7']:.1f}%** |",
         f"| Variance in a fitted 7-D PCA | 86.2% | 95.5% | {100 * R['variance']['pca7']:.1f}% |",
         f"| Best process model, full space | Pagel lambda 0.92 | white noise | **{res['B3']['best_model_full']}** (lambda fit {res['B3']['lambda_full']:.2f}) |",
         f"| LOO error ratio full / consensus | 0.36 (nearest bin) | 1.00 | **{res['B4']['ratio']:.2f}** (p = {res['B4']['p']:.3g}) |",
         f"| LOO error ratio full / nearest-copy | 0.84 | 0.88 | {res['B5']['ratio']:.2f} |",
         f"| T7 share of prediction gain | 55.5% | undefined | {('%.0f%%' % (100 * res['B9']['share'])) if res['B9']['defined'] else 'undefined'} |",
         f"| T7 residual: middle vs ends | - | 0.71 vs 0.45/0.59 | {res['B6']['unexplained_middle']:.2f} vs {res['B6']['unexplained_end_a']:.2f}/{res['B6']['unexplained_end_b']:.2f} |",
         "", "## Registered hypotheses", "", "| ID | Primary | Statement | Registered P | Result | Verdict |", "|---|---|---|---|---|---|"]
    skip = {"pass", "per_landmark", "end_a", "end_b", "pred_landmarks_full", "observed_landmarks", "dAICc_full"}
    def summ(v):
        if "primary" in v and isinstance(v.get("primary"), dict):   # B8
            return "; ".join(f"{fz}: full {v[fz]['primary']['rms_full']:.4f} vs consensus {v[fz]['primary']['rms_consensus']:.4f} (T7 {v[fz]['primary']['rms_thompson7']:.4f}; Mahal² {v[fz]['primary']['mahal2_t7']:.1f} vs 14.1; closer={'yes' if v[fz]['closer_than_consensus_full'] else 'no'})" for fz in v if isinstance(v[fz], dict) and "primary" in v[fz])
        return json.dumps({a: (round(b, 3) if isinstance(b, float) else b) for a, b in v.items() if a not in skip and not isinstance(b, dict)})
    for h in H["hypotheses"]:
        r = res[h["id"]]
        L.append(f"| {h['id']} | {'yes' if h['primary'] else 'no'} | {h['statement']} | {100 * h['p']:.0f}% | `{summ(r) if h['id'] != 'B8' else ''}`{summ(r) if h['id'] == 'B8' else ''} | {'✓' if r['pass'] else '✗'} |")
    L += ["", f"**{R['n_pass']}/9 supported ({R['n_primary_pass']}/5 primary). Brier score of the registered probabilities: {R['brier']:.3f}.**", "",
          "## Fossil time slice (B8), 45 Ma", "", "| Fossil tribe | n | Lineage | full-space prediction | Thompson-7 prediction | extant consensus | extant Apini mean | Mahalanobis² in T7 (χ²₇ 95% = 14.1) |", "|---|---|---|---|---|---|---|---|"]
    for fz in ("Electrapini", "Melikertini"):
        v = res["B8"][fz]
        for key in ("primary", "alt"):
            p_ = v[key]
            L.append(f"| {fz if key == 'primary' else ''} | {v['n_wings'] if key == 'primary' else ''} | stem {p_['lineage']} @ 45 Ma | {p_['rms_full']:.4f} | {p_['rms_thompson7']:.4f} | {p_['rms_consensus']:.4f} | {p_['rms_extant_lineage_tip']:.4f} | {p_['mahal2_t7']:.1f} |")
    L += ["", "## Leave-one-unit-out (Procrustes RMS to the held-out unit mean)", "", "| full (PCs to 95%) | Thompson-7 | PCA-7 | copy nearest | consensus | T7 oracle |", "|---|---|---|---|---|---|",
          f"| **{R['loo_medians']['rms_full']:.4f}** | {R['loo_medians']['rms_thompson7']:.4f} | {R['loo_medians']['rms_pca7']:.4f} | {R['loo_medians']['rms_nearest']:.4f} | {R['loo_medians']['rms_consensus']:.4f} | {R['loo_medians']['rms_t7_oracle']:.4f} |", "",
          "| Unit | Family | n | tau (Myr) | nearest | full | T7 | nearest-copy | consensus | within-unit RMS |", "|---|---|---|---|---|---|---|---|---|---|"]
    for r in sorted(R["per_unit"], key=lambda r: r["tau_myr"]):
        L.append(f"| {r['unit']} | {r['family']} | {r['n_wings']} | {r['tau_myr']:.0f} | {r['nearest']} | {r['rms_full']:.4f} | {r['rms_thompson7']:.4f} | {r['rms_nearest']:.4f} | {r['rms_consensus']:.4f} | {r['within_rms']:.4f} |")
    A_ = np.array([[r["tau_myr"], r["rms_full"], r["rms_nearest"], r["rms_consensus"]] for r in R["per_unit"]])
    far = A_[A_[:, 0] > 60]; midb = A_[(A_[:, 0] > 30) & (A_[:, 0] <= 60)]
    e8, m8 = res["B8"]["Electrapini"]["primary"], res["B8"]["Melikertini"]["primary"]
    L += ["", "## Interpretation (written after the run; not pre-registered)", "",
          f"- **At 110 Myr, bee wings ARE about as Thompsonian as fish bodies.** The 7 global grid modes capture {100 * R['variance']['thompson7']:.0f}% of between-tribe variance (fish 58%, hornets 39%), and they carry {100 * res['B9']['share']:.0f}% of the phylogenetic prediction gain (fish 55%). B1 passed only by a hair and B9 failed in the direction opposite to the registration: across families, wing differences are dominated by global proportion changes (the kind Thompson drew), whereas between closely related hornet species they are local vein shifts. The Thompson description is a deep-time description.",
          f"- **The residual is at the wing's ends, not in the middle (B6 failed, reversing the hornet result).** Landmarks {', '.join(str(i + 1) for i in sorted(set(res['B6']['end_a']) | set(res['B6']['end_b'])))} (long-axis extremes: base and marginal-cell apex) hold the variance the global grid cannot express, consistent with the base and apex evolving semi-independently of the central cell system across families.",
          f"- **Phylogenetic signal is strong and Brownian.** Brownian motion beats white noise by ΔAICc = {res['B3']['dAICc_full']['WN']:.0f} and beats lambda/OU/EB by > 300 (those models are penalised by their extra parameter at n = 42). The consequence is visible in the horizon structure: the full-space prediction beats the consensus by {100 * (1 - res['B4']['ratio']):.0f}% overall (B4 failed only on the registered 0.8 threshold; p = {res['B4']['p']:.1e}), and, for the first time in this programme, **it beats copying the nearest relative by {100 * (1 - res['B5']['ratio']):.0f}%** (B5 failed in the favourable direction, p = {res['B5']['p_full_less']:.1e}). Beyond 60 Myr the full prediction is better than nearest-copy in {100 * np.mean(far[:, 1] < far[:, 2]):.0f}% of units (median {np.median(far[:, 1]):.4f} vs {np.median(far[:, 2]):.4f}): when the nearest relative is itself far away, averaging over the whole tree under BM is what works, and in fish that advantage was hidden because 93% of targets had a relative within 50 Myr.",
          f"- **The horizon rule survives but on thin evidence.** Units with a relative within 30 Myr (n = {res['B7']['n_tau_le_30']}, i.e. two) have half the error of units beyond 60 Myr (n = {res['B7']['n_tau_gt_60']}); B7 passes on the registered criterion but n = 2 makes the ratio unreliable. The 30-60 Myr bin (n = {len(midb)}) sits at {np.median(midb[:, 1]):.4f}, close to the deep bin, so the degradation here is gradual rather than a cliff.",
          f"- **Fossil test: one hit, one miss, both outside the intervals.** The 45 Ma stem-Apini prediction is nearer Electrapini than the consensus is ({e8['rms_full']:.4f} vs {e8['rms_consensus']:.4f}) but not nearer Melikertini ({m8['rms_full']:.4f} vs {m8['rms_consensus']:.4f}); B8 required both. In every case the prediction is far better than the extant Apini mean ({e8['rms_extant_lineage_tip']:.4f}, {m8['rms_extant_lineage_tip']:.4f}): honey-bee wings have moved a long way since the Eocene, and the ancestral-state machinery correctly backs them off toward the corbiculate stem. The Mahalanobis² values ({e8['mahal2_t7']:.0f}, {m8['mahal2_t7']:.0f} vs 14.1) say the Thompson-7 predictive ellipsoid is far too narrow for a 45 Myr extrapolation, exactly as the fish intervals were too narrow beyond 50 Myr; the fossil tribes also rest on 3-6 literature drawings. The alternative Bombini-lineage placement gives slightly better fits ({res['B8']['Electrapini']['alt']['rms_full']:.4f}, {res['B8']['Melikertini']['alt']['rms_full']:.4f}) but was not the registered one.",
          "- **Across the three landmark groups** the pattern is coherent: Thompson's global grid describes divergence between distantly related forms (fish families, bee families) and fails for closely related ones (hornet species), where change is local. Phylogenetic conditioning adds value over the nearest relative only when the nearest relative is far away. Both statements were visible in none of the groups alone.", "",
          "## Caveats (stated in the registration)", "",
          "- The tribe tree is an a-priori scaffold from published family- and subfamily-level phylogenies with ages rounded to 5 Myr; it is not a new estimate and several tribe positions are polytomies. Replacing it with a published dated tree (e.g. Bossert et al. 2019 for Apidae) is a one-file change.",
          "- Units are tribes, so 'species means' are tribe means over 3-21 wings from mixed genera; within-unit RMS is reported per unit.",
          "- 41 wings come from literature drawings rather than specimens; the two fossil tribes are entirely from Engel (2001) drawings.",
          f"- Per-unit values: `thompson/portfolio/results/bee_wings.json`.", ""]
    OUT_MD.write_text("\n".join(L), encoding="utf-8"); print(f"    wrote {OUT_MD}")


def make_fig(R, H, M0, B, Y_opa, units, recs, fossil_means):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    res = R["hypotheses"]
    fig = plt.figure(figsize=(18, 10.5)); gs = fig.add_gridspec(2, 4)
    ax = fig.add_subplot(gs[0, 0])
    ax.bar([0, 1, 2], [57.9, 38.8, 100 * R["variance"]["thompson7"]], width=0.38, color="#1f77b4", label="7 a-priori Thompson modes")
    ax.bar([0.4, 1.4, 2.4], [86.2, 95.5, 100 * R["variance"]["pca7"]], width=0.38, color="#9467bd", label="fitted 7-D PCA")
    ax.set_xticks([0.2, 1.2, 2.2]); ax.set_xticklabels(["fish bodies", "hornet wings", "bee wings"]); ax.set_ylabel("% of between-unit shape variance"); ax.set_ylim(0, 105); ax.legend(fontsize=7, loc="lower right")
    ax.set_title("A. How Thompsonian? (B1, B2)", fontsize=9)
    ax = fig.add_subplot(gs[0, 1])
    fr = np.array(res["B6"]["per_landmark"]); ends = set(res["B6"]["end_a"]) | set(res["B6"]["end_b"])
    sc = ax.scatter(M0[:, 0], M0[:, 1], c=fr, cmap="magma_r", s=130, vmin=0, vmax=1, edgecolors=["red" if i in ends else "none" for i in range(K_LM)], linewidths=1.5)
    for i, (x, y) in enumerate(M0):
        ax.text(x, y, str(i + 1), fontsize=6, ha="left", va="bottom")
    plt.colorbar(sc, ax=ax, label="fraction NOT captured by T7"); ax.set_aspect("equal"); ax.axis("off")
    ax.set_title(f"B. Where the global grid fails (B6): middle {res['B6']['unexplained_middle']:.2f} vs ends {res['B6']['unexplained_end_a']:.2f}/{res['B6']['unexplained_end_b']:.2f}", fontsize=9)
    ax = fig.add_subplot(gs[0, 2:])
    taus = np.array([r["tau_myr"] for r in recs]); order = np.argsort(taus)
    for k, c, lab in [("rms_full", "#2ca02c", "full space"), ("rms_thompson7", "#1f77b4", "Thompson-7"), ("rms_nearest", "#ff7f0e", "copy nearest"), ("rms_consensus", "#7f7f7f", "consensus")]:
        ax.plot(range(len(recs)), [recs[i][k] for i in order], "o-", color=c, ms=3, lw=1, label=lab)
    ax.set_xticks(range(len(recs))); ax.set_xticklabels([f"{recs[i]['unit']} ({recs[i]['tau_myr']:.0f})" for i in order], rotation=80, fontsize=6)
    ax.set_ylabel("Procrustes RMS (leave-one-unit-out)"); ax.legend(fontsize=7, ncol=4); ax.grid(alpha=0.3)
    ax.set_title(f"C. Predicting each tribe's wing from the other {len(recs) - 1}, sorted by tau (Myr to nearest unit): full/consensus {res['B4']['ratio']:.2f}, best model {res['B3']['best_model_full']}", fontsize=9)
    # D/E: fossil tribes
    for col, fz in zip([0, 1], ["Electrapini", "Melikertini"]):
        ax = fig.add_subplot(gs[1, col]); v = res["B8"][fz]
        obs = np.array(v["observed_landmarks"]); pred = np.array(v["primary"]["pred_landmarks_full"]); cons = M0 + np.array(R["unit_means"]).mean(0) - M0
        ax.plot(obs[:, 0], obs[:, 1], "ko", ms=5, label=f"{fz} mean (n={v['n_wings']}, 45 Ma)")
        ax.plot(pred[:, 0], pred[:, 1], "o", color="#2ca02c", ms=4, mfc="none", label="predicted stem-Apini @ 45 Ma (full)")
        ax.plot(cons[:, 0], cons[:, 1], "x", color="#7f7f7f", ms=4, label="extant consensus")
        for a_, b_ in zip(obs, pred):
            ax.plot([a_[0], b_[0]], [a_[1], b_[1]], "-", color="#2ca02c", lw=0.6, alpha=0.6)
        ax.set_aspect("equal"); ax.axis("off"); ax.legend(fontsize=6.5, loc="lower left")
        ax.set_title(f"{'D' if col == 0 else 'E'}. Fossil {fz}: RMS pred {v['primary']['rms_full']:.4f} vs consensus {v['primary']['rms_consensus']:.4f} ({'closer' if v['closer_than_consensus_full'] else 'NOT closer'}); Mahal² T7 {v['primary']['mahal2_t7']:.1f}", fontsize=8)
    # F: tau vs error scatter with horizon
    ax = fig.add_subplot(gs[1, 2])
    ax.scatter(taus, [r["rms_full"] for r in recs], c="#2ca02c", s=25, label="full space"); ax.scatter(taus, [r["rms_consensus"] for r in recs], c="#7f7f7f", s=12, marker="x", label="consensus")
    ax.axvline(30, color="k", ls=":", lw=0.8); ax.axvline(60, color="k", ls=":", lw=0.8)
    ax.set_xlabel("tau: Myr to nearest other unit"); ax.set_ylabel("LOO Procrustes RMS"); ax.legend(fontsize=7); ax.grid(alpha=0.3)
    b7 = res["B7"]; ax.set_title(f"F. Horizon (B7): median full error tau>60 / tau<=30 = {b7['ratio']:.2f} (n {b7['n_tau_gt_60']} / {b7['n_tau_le_30']})" if b7["ratio"] else "F. Horizon (B7): insufficient units", fontsize=9)
    # G: lambda / model table as text
    ax = fig.add_subplot(gs[1, 3]); ax.axis("off")
    txt = [f"Full space ({R['variance']['full_pcs_95pct']} PCs), ΔAICc:"] + [f"  {k:7s} {v:7.1f}" for k, v in res["B3"]["dAICc_full"].items()] + ["", f"Pagel λ (full) = {res['B3']['lambda_full']:.2f}", f"Pagel λ (T7)   = {res['B3']['lambda_t7']:.2f}", "", f"LOO medians: full {R['loo_medians']['rms_full']:.4f}", f"  T7 {R['loo_medians']['rms_thompson7']:.4f}  PCA-7 {R['loo_medians']['rms_pca7']:.4f}", f"  nearest {R['loo_medians']['rms_nearest']:.4f}  consensus {R['loo_medians']['rms_consensus']:.4f}"]
    ax.text(0.02, 0.98, "\n".join(txt), va="top", family="monospace", fontsize=8.5); ax.set_title("G. Process models and medians (B3)", fontsize=9)
    fig.suptitle(f"Bee forewings, 42 tribes / 10 families / 110 Myr: {R['n_pass']}/9 registered hypotheses supported (Brier {R['brier']:.2f})", fontsize=11, fontweight="bold")
    fig.tight_layout(); fig.savefig(OUT_FIG, dpi=140); print(f"    wrote {OUT_FIG}")


if __name__ == "__main__":
    sys.exit(main())

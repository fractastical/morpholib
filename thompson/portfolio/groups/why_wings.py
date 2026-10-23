#!/usr/bin/env python3
"""
Why do wings diverge where they diverge?  Causal decomposition of the Thompson-7 skew in hornet and bee
forewings: tests C1-C8 registered in thompson/portfolio/groups/why_wings_hypotheses.json (committed 1da6218f,
before this ran).

Accounts tested: A measurement artefact (C1); B channelled drift along the within-unit covariance P
(C2 alignment, C3/C4 Brownian simulation with rate matrix proportional to P, C5 T7 share of P); C allometry,
hornets only (C6-C8).

Outputs: thompson/portfolio/results/why_wings.json, docs/portfolio/why_wings.md, docs/portfolio/fig_why_wings.png
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
CACHE = REPO_ROOT / ".thompson_cache" / "antscan"
HYP = REPO_ROOT / "thompson" / "portfolio" / "groups" / "why_wings_hypotheses.json"
VESPA_TREE = REPO_ROOT / "datasets" / "thompson" / "vespinae" / "opentree_induced_subtree.json"
BEE_TREE = REPO_ROOT / "datasets" / "thompson" / "bees" / "bee_tribe_tree_dated.json"
OUT_JSON = REPO_ROOT / "thompson" / "portfolio" / "results" / "why_wings.json"
OUT_MD = REPO_ROOT / "docs" / "portfolio" / "why_wings.md"
OUT_FIG = REPO_ROOT / "docs" / "portfolio" / "fig_why_wings.png"
N_END = 5
N_SIM = 2000
N_PERM = 5000
SEED = 20261022


def load_module(name, rel):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / rel)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


eng = load_module("eng", "thompson/run_phylogenetic_prediction_engine.py")
at = eng.at
vw = load_module("vw", "thompson/portfolio/groups/vespinae_wings.py")


# ----------------------------------------------------------------------------- data
def load_hornets():
    spec = {r["Spec ID"]: r for r in vw.pangaea(CACHE / "vespinae_1025_specimens.tab")}
    rows = vw.pangaea(CACHE / "vespinae_1025_wings_landmarks.tab")
    xc = [c for c in rows[0] if "(x" in c]; yc = [c for c in rows[0] if "(y" in c]
    by, cs = defaultdict(list), defaultdict(list)
    for r in rows:
        s = spec.get(r["Spec ID"])
        if not s or s["Genus"] in ("Palaeovespa", "Vespula?") or not s["Species"] or s["Species"].startswith("sp"):
            continue
        try:
            P = np.array([[float(r[x]), float(r[y])] for x, y in zip(xc, yc)])
        except ValueError:
            continue
        k = f'{s["Genus"]}_{s["Species"]}'
        by[k].append(P); cs[k].append(float(np.sqrt(((P - P.mean(0)) ** 2).sum())))
    meta = json.loads(VESPA_TREE.read_text())
    units = sorted(s for s in by if s.replace("_", " ") in meta["ott"])
    tree, keep = vw.grafen_tree(meta["newick"], units)
    tips = np.array([tree.name_to_node[u] for u in units])
    size = np.array([np.log(np.median(cs[u])) for u in units])
    return {"name": "hornets", "label": "Hornet wings (Vespinae, 30 spp, ~30 Myr)", "units": units, "by": by, "tree": tree, "tips": tips, "size": size, "K": 19}


def load_bees():
    import openpyxl
    K = 18
    TJ = json.loads(BEE_TREE.read_text())
    rows = [r for r in openpyxl.load_workbook(CACHE / "bee_fossil_s004.xlsx", read_only=True)["Scores"].iter_rows(values_only=True) if r and r[0]][1:]
    by = defaultdict(list)
    for r in rows:
        by[r[3]].append(np.array(r[4:4 + 2 * K], dtype=float).reshape(K, 2))
    units = sorted(u for u in by if u not in TJ["fossil_targets"] and len(by[u]) >= 3 and u in TJ["tips"])
    parent, depth, tip_to_id = at.parse_newick_depths(TJ["newick"])
    tree = eng.Tree(parent, depth, {u: tip_to_id[u] for u in units})
    tips = np.array([tree.name_to_node[u] for u in units])
    return {"name": "bees", "label": "Bee wings (10 families, 42 tribes, ~110 Myr)", "units": units, "by": by, "tree": tree, "tips": tips, "size": None, "K": K}


# ----------------------------------------------------------------------------- core
def analyse(G, H, rng):
    units, by, K = G["units"], G["by"], G["K"]
    n = len(units)
    print(f"\n=== {G['label']} ===")
    # one common tangent frame: GPA all wings -> unit means -> GPA means -> consensus M0; every wing OPA'd onto M0
    allW = np.array([w for u in units for w in by[u]]); lab = np.array([u for u in units for _ in by[u]])
    Yw = at.gpa(allW)
    means = np.array([Yw[lab == u].mean(0) for u in units])
    M0 = at.centre_scale(at.gpa(means).mean(0))
    Wopa = np.array([at.opa(w, M0) for w in Yw]).reshape(len(allW), -1)
    Umean = np.array([Wopa[lab == u].mean(0) for u in units])
    D = Umean - M0.reshape(1, -1)                                        # among-unit deviations (n x 2K)
    Wd = np.vstack([Wopa[lab == u] - Umean[i] for i, u in enumerate(units)])   # within-unit deviations
    n_u = np.array([int(np.sum(lab == u)) for u in units])
    P = Wd.T @ Wd / (len(Wd) - n)                                        # pooled within-unit covariance
    B, Rg = eng.thompson_mode_basis(M0)
    Qr, _ = np.linalg.qr(Rg); QB, _ = np.linalg.qr(B)
    proj_out_rigid = lambda X: X - (X @ Qr) @ Qr.T  # noqa: E731
    D = proj_out_rigid(D); P = proj_out_rigid(proj_out_rigid(P).T).T
    m = 2 * K

    def t7_share(X):
        return float(np.sum((X @ QB) ** 2) / np.sum(X ** 2))

    def per_lm_unexplained(X):
        E = X - (X @ QB) @ QB.T
        return (E.reshape(len(X), K, 2) ** 2).sum(-1).mean(0) / (X.reshape(len(X), K, 2) ** 2).sum(-1).mean(0)

    xs = M0[:, 0]; ends = np.concatenate([np.argsort(xs)[-N_END:], np.argsort(xs)[:N_END]]); mid = np.setdiff1d(np.arange(K), ends)
    ends_mid = lambda fr: float(fr[ends].mean() / fr[mid].mean())  # noqa: E731

    share_D = t7_share(D); fr_obs = per_lm_unexplained(D); ratio_obs = ends_mid(fr_obs)
    print(f"[frame] {len(allW)} wings, {n} units; T7 share of among-unit variance {share_D:.3f}; ends/middle unexplained ratio {ratio_obs:.2f}")

    # C1 artefact: within per-landmark variance vs unexplained fraction
    w_lm = np.array([P[2 * j, 2 * j] + P[2 * j + 1, 2 * j + 1] for j in range(K)])
    a_lm = np.array([np.mean(D[:, 2 * j] ** 2 + D[:, 2 * j + 1] ** 2) for j in range(K)])
    rho1, p1 = stats.spearmanr(w_lm, fr_obs)
    rho_ev, p_ev = stats.spearmanr(w_lm, a_lm)      # descriptive: evolvability-predicts-divergence per landmark
    C1 = {"rho_within_vs_unexplained": float(rho1), "p": float(p1), "pass": bool(not (rho1 > 0.5 and p1 < 0.05)),
          "descriptive_rho_within_vs_among_per_landmark": float(rho_ev), "descriptive_p": float(p_ev),
          "within_var_per_lm": w_lm.round(8).tolist(), "unexplained_per_lm": fr_obs.round(3).tolist(), "among_var_per_lm": a_lm.round(8).tolist()}
    print(f"[C1] rho(within var, T7-unexplained) = {rho1:.2f} (p={p1:.3f}) -> {'not artefact-like' if C1['pass'] else 'ARTEFACT-LIKE'};  rho(within var, among var) = {rho_ev:.2f} (p={p_ev:.3f})")

    # C2 alignment with P
    evP, vP = np.linalg.eigh(P); vP = vP[:, ::-1]; evP = evP[::-1]
    Dc = D - D.mean(0); evD, vD = np.linalg.eigh(Dc.T @ Dc / (n - 1)); vD = vD[:, ::-1]; evD = evD[::-1]
    ang = float(np.degrees(np.arccos(min(1.0, abs(vD[:, 0] @ vP[:, 0])))))
    Rv = proj_out_rigid(rng.standard_normal((N_SIM, m))); Rv /= np.linalg.norm(Rv, axis=1, keepdims=True)
    null_ang = np.degrees(np.arccos(np.abs(Rv @ vP[:, 0])))
    share3 = float(np.sum((Dc @ vP[:, :3]) ** 2) / np.sum(Dc ** 2)); exp3 = 3 / (m - 4)
    # also: fraction of among-unit variance predicted by P in the Schluter sense (D on first k within PCs), k = 1..10
    share_k = [float(np.sum((Dc @ vP[:, :k]) ** 2) / np.sum(Dc ** 2)) for k in range(1, 11)]
    C2 = {"angle_dmax_pmax_deg": ang, "null_angle_p5": float(np.percentile(null_ang, 5)), "null_angle_median": float(np.median(null_ang)),
          "p_angle": float(np.mean(null_ang <= ang)), "share_among_on_first3_within_pcs": share3, "random_expectation": exp3, "share_on_first_k_within_pcs": share_k,
          "pmax_pct_of_P": float(evP[0] / evP.sum()), "dmax_pct_of_D": float(evD[0] / evD.sum()), "pmax_t7_share": t7_share(vP[:, :1].T), "dmax_t7_share": t7_share(vD[:, :1].T)}
    C2["pass"] = bool(ang < C2["null_angle_p5"] and share3 > 2 * exp3)
    print(f"[C2] angle(d_max, p_max) = {ang:.1f} deg (null 5th pct {C2['null_angle_p5']:.1f}, median {C2['null_angle_median']:.1f}); among var on first 3 within-PCs {share3:.2f} vs random {exp3:.3f} -> pass={C2['pass']}")

    # C3/C4 Brownian simulation with rate matrix ∝ P on the tree + tip sampling noise
    tree, tips = G["tree"], G["tips"]
    Ct = tree.C[np.ix_(tips, tips)]; wC, VC = np.linalg.eigh(Ct); LC = VC * np.sqrt(np.clip(wC, 0, None))
    LP = vP * np.sqrt(np.clip(evP, 0, None))
    # scale: choose sigma^2 so that simulated among-unit variance equals observed (shares are scale-free, but tip noise is not)
    exp_tr = float(np.trace(Ct - Ct.mean(0)[None, :] - Ct.mean(1)[:, None] + Ct.mean()) * np.trace(P)) / (n - 1)
    sig2 = max(float(np.sum(Dc ** 2) / (n - 1) - np.mean(np.trace(P) / n_u)), 1e-12) / exp_tr
    sim_share, sim_ratio = np.empty(N_SIM), np.empty(N_SIM)
    for s in range(N_SIM):
        X = np.sqrt(sig2) * LC @ rng.standard_normal((n, m)) @ LP.T + (rng.standard_normal((n, m)) @ LP.T) / np.sqrt(n_u)[:, None]
        X = proj_out_rigid(X - X.mean(0))
        sim_share[s] = t7_share(X); sim_ratio[s] = ends_mid(per_lm_unexplained(X))
    lo, hi = np.percentile(sim_share, [5, 95]); rlo, rhi = np.percentile(sim_ratio, [5, 95])
    C3 = {"observed_t7_share": share_D, "sim_median": float(np.median(sim_share)), "sim_p5": float(lo), "sim_p95": float(hi), "pass": bool(lo <= share_D <= hi),
          "direction": "above" if share_D > hi else ("below" if share_D < lo else "inside"), "p_sim_ge_obs": float(np.mean(sim_share >= share_D)), "sigma2_scale": sig2,
          "tip_noise_share_of_among_var": float(np.mean(np.trace(P) / n_u) / (np.sum(Dc ** 2) / (n - 1)))}
    C4 = {"observed_ends_over_middle": ratio_obs, "sim_median": float(np.median(sim_ratio)), "sim_p5": float(rlo), "sim_p95": float(rhi), "pass": bool(rlo <= ratio_obs <= rhi),
          "direction": "above" if ratio_obs > rhi else ("below" if ratio_obs < rlo else "inside"), "end_landmarks": [int(i) for i in ends]}
    print(f"[C3] T7 share: observed {share_D:.3f} vs P-drift simulation {C3['sim_median']:.3f} [{lo:.3f}, {hi:.3f}] -> {C3['direction']}  (tip noise = {100 * C3['tip_noise_share_of_among_var']:.1f}% of among-unit variance)")
    print(f"[C4] ends/middle: observed {ratio_obs:.2f} vs simulation {C4['sim_median']:.2f} [{rlo:.2f}, {rhi:.2f}] -> {C4['direction']}")

    # C5 is P itself Thompsonian?
    share_P = float(np.trace(QB.T @ P @ QB) / np.trace(P))
    C5 = {"t7_share_within_P": share_P, "t7_share_among_D": share_D, "gap_pts": 100 * (share_D - share_P), "pass": bool(share_D - share_P >= 0.10)}
    print(f"[C5] T7 share of within-unit P {share_P:.3f} vs among-unit D {share_D:.3f} (gap {C5['gap_pts']:+.1f} pts)")

    R = {"C1": C1, "C2": C2, "C3": C3, "C4": C4, "C5": C5}
    # C6-C8 allometry (size available)
    if G["size"] is not None:
        y = G["size"] - G["size"].mean()
        b = (y @ Dc) / (y @ y); fit = np.outer(y, b); r2 = float(np.sum(fit ** 2) / np.sum(Dc ** 2))
        perm = np.array([np.sum(np.outer(yp, (yp @ Dc) / (yp @ yp)) ** 2) / np.sum(Dc ** 2) for yp in (rng.permutation(y) for _ in range(N_PERM))])
        p6 = float(np.mean(perm >= r2))
        share_b = t7_share(b[None, :]); Dres = Dc - fit; share_res = t7_share(Dres)
        R["C6"] = {"r2_size": r2, "p_perm": p6, "pass": bool(r2 >= 0.15 and p6 < 0.05), "log_size": G["size"].tolist()}
        R["C7"] = {"t7_share_allometric_vector": share_b, "t7_share_among": share_D, "pass": bool(share_b - share_D >= 0.10)}
        R["C8"] = {"t7_share_size_corrected": share_res, "t7_share_uncorrected": share_D, "pass": bool(share_D - share_res >= 0.05)}
        # per-mode correlation with size (descriptive)
        Z = eng.project(B, Dc); R["C6"]["mode_size_r"] = [float(np.corrcoef(y, Z[:, j])[0, 1]) for j in range(7)]
        # phylogenetic GLS version (descriptive, not registered): is the allometry more than a Vespa-vs-Vespula contrast?
        V = Ct + 1e-6 * np.trace(Ct) / n * np.eye(n); Li = np.linalg.inv(np.linalg.cholesky(V))
        Xg = Li @ np.column_stack([np.ones(n), y]); Dg = Li @ D
        beta = np.linalg.lstsq(Xg, Dg, rcond=None)[0]; resg = Dg - Xg @ beta
        b0 = np.linalg.lstsq(Xg[:, :1], Dg, rcond=None)[0]; res0 = Dg - Xg[:, :1] @ b0
        r2_pgls = float(1 - np.sum(resg ** 2) / np.sum(res0 ** 2))
        perm_p = np.array([1 - np.sum((Dg - Xg_p @ np.linalg.lstsq(Xg_p, Dg, rcond=None)[0]) ** 2) / np.sum(res0 ** 2)
                           for Xg_p in (Li @ np.column_stack([np.ones(n), rng.permutation(y)]) for _ in range(N_PERM))])
        R["C6"]["pgls_r2"] = r2_pgls; R["C6"]["pgls_perm_p"] = float(np.mean(perm_p >= r2_pgls)); R["C6"]["pgls_t7_share_allometric_vector"] = t7_share(beta[1:2])
        print(f"     PGLS (BM, Grafen tree): R2 = {r2_pgls:.3f} (perm p={R['C6']['pgls_perm_p']:.4f}); T7 share of PGLS allometric vector {R['C6']['pgls_t7_share_allometric_vector']:.3f}")
        R["C6"]["size_scores_t7"] = (Dc @ (b / np.linalg.norm(b))).tolist()   # projection of each species on the allometric direction (figure only)
        print(f"[C6] size R2 = {r2:.3f} (perm p={p6:.4f}); [C7] T7 share of allometric vector {share_b:.3f} vs among {share_D:.3f}; [C8] size-corrected T7 share {share_res:.3f}")
        print(f"     per-mode r with log size: {np.round(R['C6']['mode_size_r'], 2).tolist()}")
    out = {"label": G["label"], "n_units": n, "n_wings": int(len(allW)), "K": K, "t7_share_among": share_D, "ends_over_middle": ratio_obs, "hypotheses": R,
           "sim_t7_share": sim_share.round(4).tolist(), "sim_ends_over_middle": sim_ratio.round(3).tolist(), "null_angles": null_ang.round(1).tolist(), "consensus": M0.tolist()}
    return out


def main() -> int:
    H = json.loads(HYP.read_text()); rng = np.random.default_rng(SEED)
    res = {"hornets": analyse(load_hornets(), H, rng), "bees": analyse(load_bees(), H, rng)}
    # scoring
    rows, sq = [], []
    for h in H["hypotheses"]:
        for g, key in (("hornets", "p_hornets"), ("bees", "p_bees")):
            if key in h and h["id"] in res[g]["hypotheses"]:
                ok = res[g]["hypotheses"][h["id"]]["pass"]; rows.append({"id": h["id"], "group": g, "p": h[key], "pass": bool(ok)}); sq.append((h[key] - float(ok)) ** 2)
    out = {"group": "why_wings", "scored_on": dt.date.today().isoformat(), "registration_commit": "1da6218f", "seed": SEED, "n_sim": N_SIM, "scores": rows,
           "n_pass": int(sum(r["pass"] for r in rows)), "n_claims": len(rows), "brier": float(np.mean(sq)), "groups": res}
    print(f"\n{out['n_pass']}/{out['n_claims']} registered claims supported; Brier {out['brier']:.3f}")
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    slim = json.loads(json.dumps(out))
    OUT_JSON.write_text(json.dumps(slim, indent=1), encoding="utf-8")
    write_md(out, H); make_fig(out, H)
    return 0


def write_md(O, H):
    g = O["groups"]; h_, b_ = g["hornets"]["hypotheses"], g["bees"]["hypotheses"]
    f = lambda x, d=3: f"{x:.{d}f}"  # noqa: E731
    L = [f"# Why do wings diverge where they diverge? A causal decomposition of the Thompson skew", "",
         f"Scored {O['scored_on']} by `thompson/portfolio/groups/why_wings.py`; hypotheses C1-C8 and the protocol were committed beforehand ({O['registration_commit']}). "
         f"Hornets: {g['hornets']['n_wings']:,} wings / {g['hornets']['n_units']} species; bees: {g['bees']['n_wings']:,} wings / {g['bees']['n_units']} tribes. {O['n_sim']} Brownian simulations per group, seed {O['seed']}.", "",
         "![figure](./fig_why_wings.png)", "",
         "## The question", "",
         "Thompson (1917, pp. 385-386) proposed 'a mathematical study of ... the deformation of the wing ... from one species to another' and read such deformations causally, as the record of forces during growth. "
         "The descriptive runs found that his 7 global grid modes carry 39% of between-species wing variance in hornets (30 Myr) but 56% of between-tribe variance in bees (110 Myr), with the residual central in hornets and at the base/apex in bees. "
         "This run asks what produces that skew, using the one causal lever both datasets contain: replicate wings per unit, hence a pooled within-unit covariance P (the direction individuals already vary in), plus a size proxy for hornets.", "",
         "## Results against the registered accounts", "",
         "| | Hornets (30 spp, 30 Myr) | Bees (42 tribes, 110 Myr) |", "|---|---|---|",
         f"| T7 share of among-unit variance (this frame) | {f(g['hornets']['t7_share_among'])} | {f(g['bees']['t7_share_among'])} |",
         f"| T7 share of within-unit P (C5) | {f(h_['C5']['t7_share_within_P'])} | {f(b_['C5']['t7_share_within_P'])} |",
         f"| rho(within-unit var, T7-unexplained) per landmark (C1) | {f(h_['C1']['rho_within_vs_unexplained'], 2)} (p {f(h_['C1']['p'], 3)}) | {f(b_['C1']['rho_within_vs_unexplained'], 2)} (p {f(b_['C1']['p'], 3)}) |",
         f"| rho(within-unit var, among-unit var) per landmark (descriptive) | {f(h_['C1']['descriptive_rho_within_vs_among_per_landmark'], 2)} (p {f(h_['C1']['descriptive_p'], 3)}) | {f(b_['C1']['descriptive_rho_within_vs_among_per_landmark'], 2)} (p {f(b_['C1']['descriptive_p'], 3)}) |",
         f"| angle d_max vs p_max (C2); random 5th pct / median | {f(h_['C2']['angle_dmax_pmax_deg'], 1)}° ; {f(h_['C2']['null_angle_p5'], 1)}° / {f(h_['C2']['null_angle_median'], 1)}° | {f(b_['C2']['angle_dmax_pmax_deg'], 1)}° ; {f(b_['C2']['null_angle_p5'], 1)}° / {f(b_['C2']['null_angle_median'], 1)}° |",
         f"| among-unit variance on first 3 within-PCs (C2); random | {f(h_['C2']['share_among_on_first3_within_pcs'], 2)} ; {f(h_['C2']['random_expectation'], 3)} | {f(b_['C2']['share_among_on_first3_within_pcs'], 2)} ; {f(b_['C2']['random_expectation'], 3)} |",
         f"| p_max share of P / d_max share of D | {f(h_['C2']['pmax_pct_of_P'], 2)} / {f(h_['C2']['dmax_pct_of_D'], 2)} | {f(b_['C2']['pmax_pct_of_P'], 2)} / {f(b_['C2']['dmax_pct_of_D'], 2)} |",
         f"| T7 share: P-drift simulation median [5%, 95%] vs observed (C3) | {f(h_['C3']['sim_median'])} [{f(h_['C3']['sim_p5'])}, {f(h_['C3']['sim_p95'])}] vs **{f(h_['C3']['observed_t7_share'])}** ({h_['C3']['direction']}) | {f(b_['C3']['sim_median'])} [{f(b_['C3']['sim_p5'])}, {f(b_['C3']['sim_p95'])}] vs **{f(b_['C3']['observed_t7_share'])}** ({b_['C3']['direction']}) |",
         f"| ends/middle unexplained ratio: simulation vs observed (C4) | {f(h_['C4']['sim_median'], 2)} [{f(h_['C4']['sim_p5'], 2)}, {f(h_['C4']['sim_p95'], 2)}] vs **{f(h_['C4']['observed_ends_over_middle'], 2)}** ({h_['C4']['direction']}) | {f(b_['C4']['sim_median'], 2)} [{f(b_['C4']['sim_p5'], 2)}, {f(b_['C4']['sim_p95'], 2)}] vs **{f(b_['C4']['observed_ends_over_middle'], 2)}** ({b_['C4']['direction']}) |",
         f"| tip sampling noise as share of among-unit variance | {100 * h_['C3']['tip_noise_share_of_among_var']:.1f}% | {100 * b_['C3']['tip_noise_share_of_among_var']:.1f}% |",
         f"| size R² (C6) | {f(h_['C6']['r2_size'])} (perm p {h_['C6']['p_perm']:.4f}) | no size |",
         f"| size R², phylogenetic GLS (descriptive) | {f(h_['C6']['pgls_r2'])} (perm p {h_['C6']['pgls_perm_p']:.4f}); PGLS allometric vector {100 * h_['C6']['pgls_t7_share_allometric_vector']:.0f}% in T7 | - |",
         f"| T7 share of allometric vector (C7) | {f(h_['C7']['t7_share_allometric_vector'])} | - |",
         f"| T7 share after size correction (C8) | {f(h_['C8']['t7_share_size_corrected'])} | - |",
         "", "## Registered claims", "", "| ID | Account | Group | Statement | Registered P | Verdict |", "|---|---|---|---|---|---|"]
    for h in H["hypotheses"]:
        for gname, key in (("hornets", "p_hornets"), ("bees", "p_bees")):
            if key in h and h["id"] in g[gname]["hypotheses"]:
                L.append(f"| {h['id']} | {h['account']} | {gname} | {h['statement']} | {100 * h[key]:.0f}% | {'✓' if g[gname]['hypotheses'][h['id']]['pass'] else '✗'} |")
    L += ["", f"**{O['n_pass']}/{O['n_claims']} supported. Brier {O['brier']:.3f}.**", "", "## Interpretation (written after the run; not pre-registered)", ""]
    L += interpretation(O)
    L += ["", "## Method notes", "",
          "- Frame: all wings GPA-aligned, unit means re-aligned to the consensus by OPA, every wing then OPA'd onto the consensus; D = unit means − consensus, W = wing − unit mean; the 4 similarity directions are projected out of both. T7 shares in this frame differ by a few points from the descriptive runs (which used a GPA of unit means).",
          "- P = pooled within-unit SSCP / (N_wings − n_units). For hornets it is within-species (growth/individual variation). For bees 'within-unit' is within-tribe and so mixes genera: bee P contains shallow divergence, which inflates the bee C2 alignment and makes the bee P-drift null partly circular. Treat the bee half as weaker.",
          "- Simulation (C3/C4): X = σ L_C G L_Pᵀ + tip noise with covariance P/n_u, where C is the MRCA-height matrix of the kept tips (Grafen for hornets, dated for bees) and σ² is set so the simulated among-unit variance matches the observed; T7 share and ends/middle ratio are scale-free.",
          "- Allometry: centroid size in pixels from single-lab imaging (Perrard et al. 2014); species ordering checked against known body sizes before registration (soror, mandarinia, ducalis largest; Vespula smallest; within-species CV 5-10%).",
          "- Per-group arrays (simulated shares, null angles) are in `thompson/portfolio/results/why_wings.json`.", ""]
    OUT_MD.write_text("\n".join(L), encoding="utf-8"); print(f"    wrote {OUT_MD}")


def interpretation(O):
    g = O["groups"]; h_, b_ = g["hornets"]["hypotheses"], g["bees"]["hypotheses"]
    L = []
    # artefact
    art = []
    for name, r in (("hornets", h_), ("bees", b_)):
        art.append(f"{name}: rho = {r['C1']['rho_within_vs_unexplained']:.2f} (p = {r['C1']['p']:.2f})")
    bad = [n for n, r in (("hornets", h_), ("bees", b_)) if not r["C1"]["pass"]]
    L.append(f"- **Measurement artefact (account A).** Per-landmark digitising noise {'does not predict' if not bad else 'PREDICTS'} where the global grid fails ({'; '.join(art)}). "
             + ("The central residual in hornets and the base/apex residual in bees are therefore not where landmarks are imprecise; they are biology." if not bad else f"For {', '.join(bad)} the residual location coincides with the noisiest landmarks and should not be read as biology until re-digitised."))
    # alignment
    for name, r in (("Hornets", h_), ("Bees", b_)):
        c2 = r["C2"]
        L.append(f"- **Alignment with the developmental channel, {name.lower()} (account B, C2).** The main axis of divergence sits {c2['angle_dmax_pmax_deg']:.0f}° from the main axis of within-unit variation (random pairs: median {c2['null_angle_median']:.0f}°, 5th percentile {c2['null_angle_p5']:.0f}°); the first three within-unit PCs carry {100 * c2['share_among_on_first3_within_pcs']:.0f}% of among-unit variance against a random expectation of {100 * c2['random_expectation']:.1f}%. "
                 + ("Divergence follows the lines of least resistance." if c2["pass"] else "Divergence is only weakly channelled by P."))
    # simulation
    for name, r in (("Hornets", h_), ("Bees", b_)):
        c3, c4, c5 = r["C3"], r["C4"], r["C5"]
        L.append(f"- **Does drift along P suffice, {name.lower()} (C3-C5)?** Brownian motion with rate matrix ∝ P gives a T7 share of {c3['sim_median']:.2f} [{c3['sim_p5']:.2f}, {c3['sim_p95']:.2f}]; observed {c3['observed_t7_share']:.2f} ({c3['direction']}). "
                 f"The ends/middle residual ratio under P-drift is {c4['sim_median']:.2f} [{c4['sim_p5']:.2f}, {c4['sim_p95']:.2f}]; observed {c4['observed_ends_over_middle']:.2f} ({c4['direction']}). P itself is {100 * c5['t7_share_within_P']:.0f}% Thompsonian against {100 * c5['t7_share_among_D']:.0f}% for divergence. "
                 + ("Both the amount of global-grid variance and its spatial residual are what neutral accumulation along P produces: no additional cause is required." if c3["pass"] and c4["pass"] else
                    (f"Divergence is MORE global than P-drift produces{' (marginally: observed sits just past the 95th percentile, P(sim ≥ obs) = %.3f)' % c3['p_sim_ge_obs'] if c3['p_sim_ge_obs'] > 0.01 else ''}: drift along the developmental channel accounts for most of the Thompson share, with a small excess that needs size or function (accounts C/D)." if c3["direction"] == "above" else
                     ("Divergence is LESS global than P-drift produces: change is more local than the developmental channel predicts, consistent with selection on particular veins/cells." if c3["direction"] == "below" else
                      "The amount of global-grid variance matches P-drift but its spatial pattern does not."))))
    # allometry
    c6, c7, c8 = h_["C6"], h_["C7"], h_["C8"]
    L.append(f"- **Allometry, hornets (account C).** Log size explains {100 * c6['r2_size']:.0f}% of among-species wing shape (permutation p = {c6['p_perm']:.3f}); under phylogenetic GLS, which discounts the Vespa-vs-Vespula/Dolichovespula size contrast visible in panel H, it is {100 * c6['pgls_r2']:.0f}% (p = {c6['pgls_perm_p']:.3f}; descriptive, not registered). The allometric direction is {100 * c7['t7_share_allometric_vector']:.0f}% inside Thompson's 7 modes ({100 * c6['pgls_t7_share_allometric_vector']:.0f}% for the PGLS vector) against {100 * c7['t7_share_among']:.0f}% for divergence overall; size-corrected divergence is {100 * c8['t7_share_size_corrected']:.0f}% Thompsonian. "
             + ("Size-correlated change IS the global-grid change: Thompson's reading of his grid as a growth record holds for hornet wings." if c6["pass"] and c7["pass"] else
                ("Size matters but size-related change is no more global than the rest." if c6["pass"] else
                 (f"C6 misses the registered 15% threshold by {100 * (0.15 - c6['r2_size']):.1f} points; C7 {'passes' if c7['pass'] else 'misses'} its 10-point margin ({100 * (c7['t7_share_allometric_vector'] - c7['t7_share_among']):+.1f}); more importantly the ordinary-regression signal {'disappears' if c6['pgls_perm_p'] > 0.05 else 'shrinks'} under PGLS, so what the raw regression picked up is the one deep size contrast in the tree (large Vespa vs small Vespula/Dolichovespula) rather than size tracking shape along branches. Size is not the cause of the hornet pattern." if c6["p_perm"] < 0.05 else
                  "Size is not a major driver of among-species wing shape in Vespinae at this resolution."))))
    # synthesis
    L.append("- **Synthesis.** " + synthesis(h_, b_))
    return L


def synthesis(h_, b_):
    hb = h_["C3"]["direction"]; bb = b_["C3"]["direction"]
    s = []
    if hb == "inside" and bb == "above":
        s.append("At 30 Myr hornet wings diverge as drift along their own developmental covariance would predict; at 110 Myr bee wings are more global than that. The deep-time Thompsonian pattern is produced, not inherited: over long times something (size, flight mechanics, or the integration of many small local changes into coordinated proportion shifts) concentrates divergence into the few global modes.")
    elif hb == "inside" and bb == "inside":
        s.append("In both groups, the amount of Thompsonian divergence is what neutral accumulation along the within-unit covariance produces. The 'Thompson describes deep divergence' finding reduces to: P is partly Thompsonian and time integrates it; no selective or mechanical explanation is required for the grid share itself.")
    elif hb == "above" and bb == "above":
        s.append("In both groups divergence is more global than drift along P would give. The global grid is not simply the developmental channel writ large; an external cause (size evolution, flight function) is implicated at both time scales.")
    elif hb == "below" and bb in ("above", "inside"):
        s.append(f"Three levels, three answers. Within a species, hornet wing variation is {100 * h_['C5']['t7_share_within_P']:.0f}% in Thompson's global modes: growth variation is Thompsonian, as he assumed. "
                 f"Between hornet species (30 Myr) divergence is {100 * h_['C3']['observed_t7_share']:.0f}%, BELOW what drift along that channel would give ({100 * h_['C3']['sim_p5']:.0f}-{100 * h_['C3']['sim_p95']:.0f}%) and not aligned with p_max ({h_['C2']['angle_dmax_pmax_deg']:.0f}°): what separates close species is local vein/cell geometry that lies off the developmental ridge, the signature of lineage-specific change (selection on particular cells, or local developmental shifts) rather than of accumulated growth variation. "
                 f"Across bee families (110 Myr) divergence is {100 * b_['C3']['observed_t7_share']:.0f}% global, aligned with the within-unit channel ({b_['C2']['angle_dmax_pmax_deg']:.0f}°, {100 * b_['C2']['share_among_on_first3_within_pcs']:.0f}% of variance on three within-PCs), and {'marginally above' if bb == 'above' else 'inside'} the drift band ({100 * b_['C3']['sim_p5']:.0f}-{100 * b_['C3']['sim_p95']:.0f}%). "
                 "The causal reading: local between-species differences do not compound over time (they are bounded, reversible or idiosyncratic), while global proportion change accumulates along the lines of least resistance; integrate for 100 Myr and the compounding part is all that is left. Thompson's grid describes deep divergence because deep divergence is the developmental channel integrated over time, not because it is a different kind of change.")
    else:
        s.append(f"Hornets sit {hb} and bees {bb} the P-drift band; see the per-group bullets.")
    s.append(f"Per landmark, the amount of among-unit change tracks the amount of within-unit variation in both groups (Spearman {h_['C1']['descriptive_rho_within_vs_among_per_landmark']:.2f} hornets, {b_['C1']['descriptive_rho_within_vs_among_per_landmark']:.2f} bees; descriptive, not registered): evolvability predicts divergence (Hansen & Houle 2008) at both time scales even where its direction does not.")
    s.append("Caveat on the asymmetry: hornet P is within-species; bee P is within-tribe and so contains shallow among-genus divergence, which makes the bee alignment partly circular and the bee test weaker than the hornet one. A within-species P for bees (per-species replicate wings) is the one dataset that would make the comparison clean.")
    s.append(f"Size in hornets: the raw regression gives {100 * h_['C6']['r2_size']:.0f}% (p = {h_['C6']['p_perm']:.3f}) but phylogenetic GLS gives {100 * h_['C6']['pgls_r2']:.0f}% (p = {h_['C6']['pgls_perm_p']:.2f}); the signal is the single Vespa-vs-Vespula size contrast, not size tracking shape along branches. "
             "Thompson's default explanation for a global grid change, growth and size, is therefore not what produces the hornet pattern, and the allometric direction being ~50% Thompsonian only says that the one deep contrast in the tree is global like everything deep. What remains for the local, off-channel differences between close hornet species is lineage-specific selection on particular cells (flight or mating function), which needs flight-ecology covariates to test. For bees, body size per tribe is still worth obtaining, but the hornet result lowers the prior that it will explain the deep-time global change.")
    return " ".join(s)


def make_fig(O, H):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    g = O["groups"]; cols = {"hornets": "#d62728", "bees": "#ff9900"}
    fig = plt.figure(figsize=(18, 10)); gs = fig.add_gridspec(2, 4)
    # A: C1 scatter both groups
    ax = fig.add_subplot(gs[0, 0])
    for name in ("hornets", "bees"):
        c1 = g[name]["hypotheses"]["C1"]; w = np.array(c1["within_var_per_lm"]); w = w / w.mean()
        ax.scatter(w, c1["unexplained_per_lm"], color=cols[name], s=30, label=f"{name}: rho {c1['rho_within_vs_unexplained']:.2f} (p {c1['p']:.2f})")
    ax.set_xlabel("within-unit variance per landmark (relative)"); ax.set_ylabel("fraction of among-unit variance NOT in T7"); ax.legend(fontsize=7); ax.grid(alpha=0.3)
    ax.set_title("A. Is the residual where the noise is? (C1, account A)", fontsize=9)
    # B: C2 null angles
    ax = fig.add_subplot(gs[0, 1])
    for name in ("hornets", "bees"):
        c2 = g[name]["hypotheses"]["C2"]
        ax.hist(g[name]["null_angles"], bins=40, alpha=0.35, color=cols[name], density=True)
        ax.axvline(c2["angle_dmax_pmax_deg"], color=cols[name], lw=2, label=f"{name}: {c2['angle_dmax_pmax_deg']:.0f}° (p {c2['p_angle']:.3f})")
    ax.set_xlabel("angle between d_max and p_max (deg)"); ax.legend(fontsize=7); ax.set_title("B. Divergence vs lines of least resistance (C2)", fontsize=9)
    # C: share on first k within PCs
    ax = fig.add_subplot(gs[0, 2])
    for name in ("hornets", "bees"):
        c2 = g[name]["hypotheses"]["C2"]; K = g[name]["K"]
        ax.plot(range(1, 11), c2["share_on_first_k_within_pcs"], "o-", color=cols[name], label=name)
        ax.plot(range(1, 11), [k / (2 * K - 4) for k in range(1, 11)], ":", color=cols[name], lw=1)
    ax.set_xlabel("first k within-unit PCs"); ax.set_ylabel("share of among-unit variance"); ax.legend(fontsize=7); ax.grid(alpha=0.3)
    ax.set_title("C. Among-unit variance captured by within-unit PCs (dotted = random)", fontsize=9)
    # D: C3 sims
    ax = fig.add_subplot(gs[0, 3])
    for name in ("hornets", "bees"):
        c3 = g[name]["hypotheses"]["C3"]
        ax.hist(g[name]["sim_t7_share"], bins=40, alpha=0.35, color=cols[name], density=True, label=f"{name}: P-drift")
        ax.axvline(c3["observed_t7_share"], color=cols[name], lw=2.5, label=f"{name} observed {c3['observed_t7_share']:.2f} ({c3['direction']})")
        ax.axvline(g[name]["hypotheses"]["C5"]["t7_share_within_P"], color=cols[name], lw=1, ls="--", label=f"{name} P itself {g[name]['hypotheses']['C5']['t7_share_within_P']:.2f}")
    ax.set_xlabel("T7 share of among-unit variance"); ax.legend(fontsize=6.5); ax.set_title("D. Does Brownian drift along P give the Thompson share? (C3, C5)", fontsize=9)
    # E: C4 sims
    ax = fig.add_subplot(gs[1, 0])
    for name in ("hornets", "bees"):
        c4 = g[name]["hypotheses"]["C4"]
        ax.hist(g[name]["sim_ends_over_middle"], bins=40, alpha=0.35, color=cols[name], density=True)
        ax.axvline(c4["observed_ends_over_middle"], color=cols[name], lw=2.5, label=f"{name} observed {c4['observed_ends_over_middle']:.2f} ({c4['direction']})")
    ax.axvline(1, color="k", lw=0.8, ls=":"); ax.set_xlabel("unexplained fraction: ends / middle"); ax.legend(fontsize=7)
    ax.set_title("E. Does P-drift put the residual where it is observed? (C4)", fontsize=9)
    # F: per-landmark maps of within var (size) vs unexplained (colour) for bees and hornets
    for col, name in ((1, "hornets"), (2, "bees")):
        ax = fig.add_subplot(gs[1, col]); c1 = g[name]["hypotheses"]["C1"]; M0 = np.array(g[name]["consensus"])
        w = np.array(c1["within_var_per_lm"]); s = 40 + 260 * (w - w.min()) / (w.max() - w.min() + 1e-12)
        sc = ax.scatter(M0[:, 0], M0[:, 1], s=s, c=c1["unexplained_per_lm"], cmap="magma_r", vmin=0, vmax=1, edgecolors=["red" if i in g[name]["hypotheses"]["C4"]["end_landmarks"] else "k"for i in range(len(M0))], linewidths=0.8)
        for i, (x, y) in enumerate(M0):
            ax.text(x, y, str(i + 1), fontsize=6, ha="left", va="bottom")
        plt.colorbar(sc, ax=ax, label="fraction NOT in T7"); ax.set_aspect("equal"); ax.axis("off")
        ax.set_title(f"{'F' if col == 1 else 'G'}. {name}: size = within-unit noise,\ncolour = where T7 fails (red ring = end landmarks)", fontsize=8)
    # H: hornet allometry
    ax = fig.add_subplot(gs[1, 3]); c6 = g["hornets"]["hypotheses"]["C6"]; c7 = g["hornets"]["hypotheses"]["C7"]; c8 = g["hornets"]["hypotheses"]["C8"]
    ax.scatter(c6["log_size"], c6["size_scores_t7"], color=cols["hornets"], s=25)
    ax.set_xlabel("log centroid size (px, species median)"); ax.set_ylabel("shape score along allometric vector"); ax.grid(alpha=0.3)
    ax.set_title(f"H. Hornet allometry (C6-C8): R² {c6['r2_size']:.2f} (p {c6['p_perm']:.3f}), PGLS {c6['pgls_r2']:.2f} (p {c6['pgls_perm_p']:.3f})\nallometric vector {100 * c7['t7_share_allometric_vector']:.0f}% in T7; size-corrected T7 share {c8['t7_share_size_corrected']:.2f}", fontsize=8)
    fig.suptitle(f"Why wings diverge where they diverge: {O['n_pass']}/{O['n_claims']} registered causal claims supported (Brier {O['brier']:.2f})", fontsize=11, fontweight="bold")
    fig.tight_layout(); fig.savefig(OUT_FIG, dpi=140); print(f"    wrote {OUT_FIG}")


if __name__ == "__main__":
    sys.exit(main())

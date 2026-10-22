#!/usr/bin/env python3
"""
Is hornet-wing evolution 'Thompsonian'?  Tests W1-W7 registered in
thompson/portfolio/groups/vespinae_wings_hypotheses.json (committed before this ran).

Data: Abels & Wedmann (2021) PANGAEA 937651/937635 (Perrard et al. 2014 wings; 17 landmarks).
Tree: Open Tree of Life induced subtree (topology), Grafen branch lengths (relative time).

Outputs: thompson/portfolio/results/vespinae_wings.json, docs/portfolio/vespinae_wings.md,
docs/portfolio/fig_vespinae_wings.png
"""
from __future__ import annotations

import datetime as dt
import importlib.util
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parents[3]
CACHE = REPO_ROOT / ".thompson_cache" / "antscan"
HYP = REPO_ROOT / "thompson" / "portfolio" / "groups" / "vespinae_wings_hypotheses.json"
TREE_JSON = REPO_ROOT / "datasets" / "thompson" / "vespinae" / "opentree_induced_subtree.json"
OUT_JSON = REPO_ROOT / "thompson" / "portfolio" / "results" / "vespinae_wings.json"
OUT_MD = REPO_ROOT / "docs" / "portfolio" / "vespinae_wings.md"
OUT_FIG = REPO_ROOT / "docs" / "portfolio" / "fig_vespinae_wings.png"
K_LM = 19
# apical / distal landmarks for W7: those with the largest consensus x after orienting base -> apex along +x
N_APICAL = 5


def load_module(name, rel):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / rel)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


eng = load_module("eng", "thompson/run_phylogenetic_prediction_engine.py")
at = eng.at


def pangaea(path: Path):
    L = path.read_text(encoding="utf-8", errors="replace").splitlines()
    i = next(k for k, l in enumerate(L) if l.startswith("*/"))
    hdr = L[i + 1].split("\t")
    return [dict(zip(hdr, l.split("\t"))) for l in L[i + 2:] if l.strip()]


def grafen_tree(newick: str, species: list[str]):
    parent, _, tip_to_id = at.parse_newick_depths(newick)
    n = len(parent)
    children = defaultdict(list)
    for v, p in enumerate(parent):
        if p >= 0:
            children[p].append(v)
    ntips = [0] * n
    order = []
    stack = [parent.index(-1)]
    while stack:
        u = stack.pop(); order.append(u); stack.extend(children[u])
    for u in reversed(order):
        ntips[u] = 1 if not children[u] else sum(ntips[c] for c in children[u])
    N = ntips[order[0]]
    depth = [(N - ntips[u]) / (N - 1) for u in range(n)]          # Grafen: root 0, tips 1
    keep = {s: tip_to_id[s] for s in species if s in tip_to_id}
    return eng.Tree(parent, depth, keep), keep


def pcs(D: np.ndarray, k: int | None = None, var: float | None = None):
    mu = D.mean(0); U, S, Vt = np.linalg.svd(D - mu, full_matrices=False)
    if var is not None:
        ev = S ** 2 / np.sum(S ** 2); k = int(np.searchsorted(np.cumsum(ev), var) + 1)
    return mu, Vt[:k].T


def best_model(tree, tips, X):
    f = eng.fit_models(tree, tips, X)
    k = min(f, key=lambda a: f[a]["AICc"])
    return f[k]["model"], k, f


def main() -> int:
    H = json.loads(HYP.read_text())
    print("[1] wings -> species means")
    spec = {r["Spec ID"]: r for r in pangaea(CACHE / "vespinae_1025_specimens.tab")}
    lm_rows = pangaea(CACHE / "vespinae_1025_wings_landmarks.tab")
    xcols = [c for c in lm_rows[0] if "(x" in c]; ycols = [c for c in lm_rows[0] if "(y" in c]
    assert len(xcols) == K_LM
    by_sp = defaultdict(list)
    for r in lm_rows:
        s = spec.get(r["Spec ID"])
        if not s or s["Genus"] in ("Palaeovespa", "Vespula?") or not s["Species"] or s["Species"].startswith("sp"):
            continue
        try:
            P = np.array([[float(r[x]), float(r[y])] for x, y in zip(xcols, ycols)])
        except ValueError:
            continue
        by_sp[f'{s["Genus"]}_{s["Species"]}'].append(P)
    tree_meta = json.loads(TREE_JSON.read_text())
    species = sorted(s for s in by_sp if s.replace("_", " ") in tree_meta["ott"])
    dropped = sorted(set(by_sp) - set(species))
    print(f"    {sum(len(v) for v in by_sp.values())} wings, {len(by_sp)} species; {len(species)} on the tree (dropped {dropped})")
    allW = np.array([w for s in species for w in by_sp[s]]); lab = np.array([s for s in species for _ in by_sp[s]])
    Yw = at.gpa(allW)
    means = np.array([Yw[lab == s].mean(0) for s in species])
    Y = at.gpa(means)
    M0 = at.centre_scale(Y.mean(0))
    # orient: base -> apex along +x (apex = landmark farthest from centroid on the long axis)
    Y_opa = np.array([at.opa(y, M0) for y in Y])
    n = len(species)
    D = (Y_opa - M0).reshape(n, -1)

    print("[2] Thompson-7 basis, variance shares")
    B, Rg = eng.thompson_mode_basis(M0)
    Z = eng.project(B, D); E = D - Z @ B.T
    tot = float(np.sum(D ** 2)); var_t7 = 1 - float(np.sum(E ** 2)) / tot
    mu7, V7 = pcs(D, k=7); var_pca7 = float(np.sum(((D - mu7) @ V7) ** 2) / np.sum((D - mu7) ** 2))
    # W7: where does the T7 residual live?
    xs = M0[:, 0]; end_pos = np.argsort(xs)[-N_APICAL:]; end_neg = np.argsort(xs)[:N_APICAL]
    res_lm = (E.reshape(n, K_LM, 2) ** 2).sum(-1).mean(0); tot_lm = (D.reshape(n, K_LM, 2) ** 2).sum(-1).mean(0)
    frac_unexplained_lm = res_lm / tot_lm
    middle = np.setdiff1d(np.arange(K_LM), np.concatenate([end_pos, end_neg]))
    w7 = {"end_a_landmarks_0based": [int(a) for a in end_pos], "end_b_landmarks_0based": [int(a) for a in end_neg],
          "unexplained_fraction_end_a": float(frac_unexplained_lm[end_pos].mean()), "unexplained_fraction_end_b": float(frac_unexplained_lm[end_neg].mean()), "unexplained_fraction_middle": float(frac_unexplained_lm[middle].mean()),
          "per_landmark_unexplained_fraction": frac_unexplained_lm.round(3).tolist(),
          "note": "the apex cannot be identified from coordinates alone (GPA orientation is arbitrary and the landmark key of Perrard et al. 2014 was not available), so both ends of the long axis are reported; W7 passes only if the end with the HIGHER unexplained fraction exceeds the middle, i.e. if the T7 residual concentrates at either extremity"}
    w7["pass"] = bool(max(w7["unexplained_fraction_end_a"], w7["unexplained_fraction_end_b"]) > w7["unexplained_fraction_middle"])
    # within- vs between-species noise (species with >= 5 wings)
    within = {}
    for s_ in species:
        W = Yw[lab == s_]
        if len(W) >= 5:
            m_ = W.mean(0); within[s_] = {"n": int(len(W)), "within_rms": float(np.sqrt(((W - m_) ** 2).sum(-1).mean(-1).mean())), "se_of_mean": float(np.sqrt(((W - m_) ** 2).sum(-1).mean(-1).mean() / len(W)))}
    between = float(np.median(np.sqrt((D.reshape(n, K_LM, 2) ** 2).sum(-1).mean(-1))))
    noise = {"median_between_species_rms_to_consensus": between, "median_within_species_rms": float(np.median([v["within_rms"] for v in within.values()])), "median_se_of_species_mean": float(np.median([v["se_of_mean"] for v in within.values()])),
             "n_species_with_single_wing": int(sum(len(by_sp[s_]) == 1 for s_ in species)), "species_detail": within}
    print(f"    T7 variance share {var_t7:.3f} (fish 0.579); PCA-7 {var_pca7:.3f} (fish 0.862); W7 ends {w7['unexplained_fraction_end_a']:.2f}/{w7['unexplained_fraction_end_b']:.2f} vs middle {w7['unexplained_fraction_middle']:.2f}")
    print(f"    noise: between-species RMS {noise['median_between_species_rms_to_consensus']:.4f}; within-species RMS {noise['median_within_species_rms']:.4f}; SE of species mean {noise['median_se_of_species_mean']:.4f}; single-wing species {noise['n_species_with_single_wing']}")

    print("[3] tree (Open Tree topology, Grafen lengths) and process models")
    tree, keep = grafen_tree(tree_meta["newick"], species)
    eng.Model.T = tree.T
    tips = np.array([tree.name_to_node[s] for s in species])
    m7, k7, f7 = best_model(tree, tips, Z)
    muF, VF = pcs(D, var=0.95); F = (D - muF) @ VF
    mF, kF, fF = best_model(tree, tips, F)
    lam7 = f7["lambda"]["model"].theta; lamF = fF["lambda"]["model"].theta
    print(f"    T7: best {m7.label()} (lambda fit {lam7:.3f}); full (PCs to 95%: {VF.shape[1]} dims): best {mF.label()} (lambda {lamF:.3f})")

    print("[4] leave-one-species-out prediction")
    recs = []
    for i, s in enumerate(species):
        obs = np.array([tips[j] for j in range(n) if j != i]); tgt = tips[i]
        tau = float(tree.T - max(tree.C[tgt, o] for o in obs))
        Dtr = D[np.arange(n) != i]
        # Thompson-7 (basis is a-priori: no fold dependence)
        m, _, _ = best_model(tree, obs, Z[np.arange(n) != i])
        mean7, _, _, _ = eng.predict(m, tree, obs, Z[np.arange(n) != i], np.array([tree.h[tgt]]), tree.C[tgt][None, :])
        pred_t7 = M0.reshape(-1) + mean7[0] @ B.T
        # full space (within-fold PCA to 95%)
        mu, V = pcs(Dtr, var=0.95); Ftr = (Dtr - mu) @ V
        m, _, _ = best_model(tree, obs, Ftr)
        meanF, _, _, _ = eng.predict(m, tree, obs, Ftr, np.array([tree.h[tgt]]), tree.C[tgt][None, :])
        pred_full = M0.reshape(-1) + mu + meanF[0] @ V.T
        # PCA-7 within fold
        mu, V = pcs(Dtr, k=7); P7 = (Dtr - mu) @ V
        m, _, _ = best_model(tree, obs, P7)
        meanP, _, _, _ = eng.predict(m, tree, obs, P7, np.array([tree.h[tgt]]), tree.C[tgt][None, :])
        pred_p7 = M0.reshape(-1) + mu + meanP[0] @ V.T
        cons = M0.reshape(-1) + Dtr.mean(0)
        j_near = max((j for j in range(n) if j != i), key=lambda j: tree.C[tgt, tips[j]])
        near = M0.reshape(-1) + D[j_near]
        tgt_shape = Y_opa[i]
        rm = lambda v: eng.shape_rms(v.reshape(K_LM, 2), tgt_shape)   # noqa: E731
        recs.append({"species": s, "tau_rel": tau, "nearest": species[j_near], "rms_thompson7": rm(pred_t7), "rms_full": rm(pred_full), "rms_pca7": rm(pred_p7), "rms_consensus": rm(cons), "rms_nearest": rm(near),
                     "rms_thompson7_oracle": rm(M0.reshape(-1) + Z[i] @ B.T)})
    A = {k: np.array([r[k] for r in recs]) for k in recs[0] if k.startswith("rms")}
    med = {k: float(np.median(v)) for k, v in A.items()}
    gain = lambda k: np.sum(A["rms_consensus"] ** 2 - A[k] ** 2)  # noqa: E731
    gf = float(gain("rms_full")); share = float(gain("rms_thompson7") / gf) if gf > 0 else float("nan")
    share_defined = bool(gf > 0 and med["rms_full"] < med["rms_consensus"])
    p3 = float(stats.wilcoxon(A["rms_full"] - A["rms_consensus"], alternative="less").pvalue)
    p4 = float(stats.wilcoxon(A["rms_full"] - A["rms_nearest"], alternative="less").pvalue)
    p6 = float(stats.wilcoxon(A["rms_pca7"] - A["rms_thompson7"], alternative="less").pvalue)
    results = {
        "W1": {"t7_variance_share": var_t7, "fish": 0.579, "pass": var_t7 > 0.58},
        "W2": {"share_of_full_prediction_gain_captured_by_t7": share, "fish": 0.555, "defined": share_defined, "pass": bool(share_defined and share > 0.555),
               "note": None if share_defined else "undefined: the full space has no prediction gain over the consensus (W3 failed), so there is no gain to apportion"},
        "W3": {"median_rms_full": med["rms_full"], "median_rms_consensus": med["rms_consensus"], "ratio": med["rms_full"] / med["rms_consensus"], "p": p3, "pass": bool(p3 < 0.05 and med["rms_full"] / med["rms_consensus"] < 0.8)},
        "W4": {"median_rms_nearest": med["rms_nearest"], "ratio": med["rms_full"] / med["rms_nearest"], "p_full_less": p4, "pass": bool(med["rms_full"] / med["rms_nearest"] > 0.9)},
        "W5": {"lambda_t7": float(lam7), "best_model_t7": m7.label(), "pass": bool(lam7 > 0.7)},
        "W6": {"pca7_variance_share": var_pca7, "median_rms_pca7": med["rms_pca7"], "median_rms_thompson7": med["rms_thompson7"], "p_pca7_less": p6, "pass": bool(var_pca7 > var_t7 and med["rms_pca7"] < med["rms_thompson7"])},
        "W7": w7,
    }
    for k, v in results.items():
        print(f"    {k}: pass={v['pass']} " + json.dumps({a: (round(b, 4) if isinstance(b, float) else b) for a, b in v.items() if a not in ("pass", "per_landmark_unexplained_fraction", "note", "species_detail")}))
    n_pass = sum(v["pass"] for v in results.values()); n_prim = sum(results[h["id"]]["pass"] for h in H["hypotheses"] if h["primary"])
    brier = float(np.mean([(h["p"] - float(results[h["id"]]["pass"])) ** 2 for h in H["hypotheses"]]))
    out = {"group": H["group"], "scored_on": dt.date.today().isoformat(), "n_species": n, "n_wings": int(len(allW)), "species": species, "dropped": dropped, "tree": {"source": "Open Tree of Life induced subtree", "branch_lengths": "Grafen (relative)", "n_tips": int(len(keep))},
           "variance": {"thompson7": var_t7, "pca7": var_pca7, "full_pcs_95pct": int(VF.shape[1])}, "models": {"thompson7": {"best": m7.label(), "lambda": float(lam7)}, "full": {"best": mF.label(), "lambda": float(lamF)}},
           "loo_medians": med, "noise": noise, "hypotheses": results, "n_pass": n_pass, "n_primary_pass": n_prim, "brier": brier, "per_species": recs, "consensus": M0.tolist(), "mode_scores": Z.tolist(), "species_means": Y_opa.tolist()}
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True); OUT_JSON.write_text(json.dumps(out, indent=1), encoding="utf-8")
    write_md(out, H); make_fig(out, H, M0, B, Z, Y_opa, species, recs)
    return 0


def write_md(R, H):
    res = R["hypotheses"]
    L = [f"# Vespinae wings: is hornet-wing evolution Thompsonian?", "",
         f"Scored {R['scored_on']} by `thompson/portfolio/groups/vespinae_wings.py`; hypotheses W1-W7 and the protocol were committed beforehand in `vespinae_wings_hypotheses.json`. "
         f"{R['n_wings']:,} wings, {R['n_species']} species on the tree (dropped: {', '.join(R['dropped']) or 'none'}). Tree: {R['tree']['source']}, {R['tree']['branch_lengths']} branch lengths, so horizons are fractions of crown depth, not Myr.", "",
         "![figure](./fig_vespinae_wings.png)", "", "## What Thompson's method specifically shows here", "", H["what_thompson_specifically_adds"], "",
         "| | Fish bodies (743 spp, 17 LMs) | Hornet wings (30 spp, 19 LMs) |", "|---|---|---|",
         f"| Variance captured by the 7 a-priori Thompson modes | 57.9% | **{100 * R['variance']['thompson7']:.1f}%** |",
         f"| Variance captured by a fitted 7-D PCA | 86.2% | {100 * R['variance']['pca7']:.1f}% |",
         f"| Share of phylogenetic prediction gain inside the 7 Thompson modes | 55.5% | **{100 * res['W2']['share_of_full_prediction_gain_captured_by_t7']:.1f}%** |",
         f"| Pagel lambda on Thompson-7 scores | 0.959 | {res['W5']['lambda_t7']:.3f} ({R['models']['thompson7']['best']}) |",
         f"| LOO error ratio, full space vs consensus | 0.36 (nearest bin) | {res['W3']['ratio']:.2f} |",
         f"| LOO error ratio, full space vs copying nearest relative | 0.84 (nearest bin) | {res['W4']['ratio']:.2f} |",
         "", "## Registered hypotheses", "", "| ID | Primary | Statement | Registered P | Result | Verdict |", "|---|---|---|---|---|---|"]
    for h in H["hypotheses"]:
        r = res[h["id"]]
        summary = json.dumps({a: (round(b, 3) if isinstance(b, float) else b) for a, b in r.items() if a not in ("pass", "per_landmark_unexplained_fraction", "end_a_landmarks_0based", "end_b_landmarks_0based", "fish", "note")})
        L.append(f"| {h['id']} | {'yes' if h['primary'] else 'no'} | {h['statement']} | {100 * h['p']:.0f}% | `{summary}` | {'✓' if r['pass'] else '✗'} |")
    L += ["", f"**{R['n_pass']}/7 supported ({R['n_primary_pass']}/3 primary). Brier score of the registered probabilities: {R['brier']:.3f}.**", "",
          "## Leave-one-species-out medians (Procrustes RMS to the held-out species mean)", "", "| full space (PCs to 95%) | Thompson-7 | PCA-7 | copy nearest relative | consensus | Thompson-7 oracle (representation floor) |", "|---|---|---|---|---|---|",
          f"| **{R['loo_medians']['rms_full']:.4f}** | {R['loo_medians']['rms_thompson7']:.4f} | {R['loo_medians']['rms_pca7']:.4f} | {R['loo_medians']['rms_nearest']:.4f} | {R['loo_medians']['rms_consensus']:.4f} | {R['loo_medians']['rms_thompson7_oracle']:.4f} |",
          "", "| Species | tau (rel.) | nearest | full | T7 | PCA-7 | nearest-copy | consensus |", "|---|---|---|---|---|---|---|---|"]
    for r in sorted(R["per_species"], key=lambda r: r["tau_rel"]):
        L.append(f"| {r['species'].replace('_', ' ')} | {r['tau_rel']:.2f} | {r['nearest'].replace('_', ' ')} | {r['rms_full']:.4f} | {r['rms_thompson7']:.4f} | {r['rms_pca7']:.4f} | {r['rms_nearest']:.4f} | {r['rms_consensus']:.4f} |")
    nz = R["noise"]; w3 = res["W3"]
    L += ["", "## Interpretation (written after the run; not pre-registered)", "",
          f"- **Hornet wings are not Thompsonian in the sense tested.** The 7 global grid modes capture {100 * R['variance']['thompson7']:.0f}% of between-species wing variance (fish: 58%), while a fitted 7-D PCA captures {100 * R['variance']['pca7']:.0f}%: species differences are extremely low-dimensional, but the dimension they occupy is not a smooth global deformation of the plane. W1 was the central Thompson-specific claim and it failed in the opposite direction from the one registered.",
          f"- **Between-species differences are small but real.** Median Procrustes RMS from a species mean to the consensus is {nz['median_between_species_rms_to_consensus']:.4f} (fish bodies: 0.030), about twice the within-species RMS ({nz['median_within_species_rms']:.4f}) and ten times the standard error of a well-sampled species mean ({nz['median_se_of_species_mean']:.4f}); only {nz['n_species_with_single_wing']} species rest on a single wing. Species means are therefore mostly well determined, and the failure of phylogenetic prediction below is not primarily a sampling problem.",
          f"- **No phylogenetic signal was detectable with this tree.** AICc picked white noise in both spaces (lambda point estimates {R['models']['thompson7']['lambda']:.2f} / {R['models']['full']['lambda']:.2f}), so the phylogenetic prediction collapsed onto the consensus (ratio {w3['ratio']:.2f}). Copying the nearest relative was *worse* than the consensus ({R['loo_medians']['rms_nearest']:.4f} vs {R['loo_medians']['rms_consensus']:.4f}), because on a tree where most of Vespa is one polytomy the 'nearest relative' is an arbitrary pick among ~15 equally close species, and copying it adds that species' idiosyncratic deviation. The Open Tree topology is a large polytomy within Vespa with Grafen lengths; a resolved, dated Vespa tree (Perrard et al. 2013) is the obvious next input and could change W3-W5.",
          "- **What Thompson's method did show:** the residual after the 7 global modes is not concentrated at either extremity of the wing but in the middle (W7 fails), i.e. the species-specific signal is a local repositioning of interior vein intersections. That is a positive, specific finding about wings versus fish bodies: fish differ by global proportions (where T7 does half the work), hornets differ by local venation geometry (where T7 does little).",
          "", "## Caveats (stated in the registration)", "",
          "- 30 species: the 'full' space is the within-fold PCA retaining 95% of variance (the registered 'all 30 dims' is singular with n = 30); every space is re-fitted inside each fold.",
          "- Branch lengths are Grafen heights from the Open Tree topology (which contains polytomies within Vespa). Phylogenetic signal and the horizon are therefore measured in relative time. Substituting a dated tree is a one-line change.",
          "- Species means pool castes, sexes and localities (Perrard et al. 2014 sampled mostly workers); intraspecific variance is not modelled.",
          f"- Per-species values: `thompson/portfolio/results/vespinae_wings.json`.", ""]
    OUT_MD.write_text("\n".join(L), encoding="utf-8"); print(f"    wrote {OUT_MD}")


def make_fig(R, H, M0, B, Z, Y_opa, species, recs):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    res = R["hypotheses"]
    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(2, 4, height_ratios=[1.0, 1.0])
    # A: wing consensus with Thompson grid + the 7 mode fields
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(M0[:, 0], M0[:, 1], "ko", ms=4)
    for i, (x, y) in enumerate(M0):
        ax.text(x, y, str(i + 1), fontsize=6, ha="left", va="bottom")
    ax.set_aspect("equal"); ax.set_title("A. Consensus hornet forewing (19 vein landmarks)", fontsize=9); ax.axis("off")
    # B: extreme species along the two dominant Thompson modes
    ax = fig.add_subplot(gs[0, 1])
    sd = Z.std(0); top2 = np.argsort(sd)[-2:][::-1]
    for j, c in zip(top2, ["#d62728", "#1f77b4"]):
        for sign in (-2, 2):
            S = (M0.reshape(-1) + sign * sd[j] * B[:, j]).reshape(-1, 2)
            ax.plot(S[:, 0], S[:, 1], "o", color=c, ms=2.5, alpha=0.5 if sign < 0 else 1)
    ax.plot(M0[:, 0], M0[:, 1], "ko", ms=3)
    ax.set_aspect("equal"); ax.axis("off")
    ax.set_title(f"B. ±2 SD along the two largest Thompson modes:\n{eng.MODE_NAMES[top2[0]]} (red), {eng.MODE_NAMES[top2[1]]} (blue)", fontsize=9)
    # C: variance shares fish vs wings
    ax = fig.add_subplot(gs[0, 2])
    ax.bar([0, 1], [57.9, 100 * R["variance"]["thompson7"]], color="#1f77b4", width=0.35, label="7 a-priori Thompson modes")
    ax.bar([0.35, 1.35], [86.2, 100 * R["variance"]["pca7"]], color="#9467bd", width=0.35, label="fitted 7-D PCA")
    w2 = res["W2"]
    ax.bar([0.7, 1.7], [55.5, 100 * w2["share_of_full_prediction_gain_captured_by_t7"] if w2["defined"] else 0], color="#2ca02c", width=0.35, label="share of phylogenetic prediction gain in T7")
    if not w2["defined"]:
        ax.text(1.7, 3, "undefined:\nno phylogenetic\ngain to share", ha="center", va="bottom", fontsize=7, color="#2ca02c")
    ax.set_xticks([0.35, 1.35]); ax.set_xticklabels(["fish bodies\n(743 spp)", "hornet wings\n(30 spp)"]); ax.set_ylabel("%"); ax.set_ylim(0, 105); ax.legend(fontsize=7, loc="lower right")
    ax.set_title("C. How Thompsonian is the group? (W1, W2, W6)", fontsize=9)
    # D: per-landmark unexplained fraction (W7)
    ax = fig.add_subplot(gs[0, 3])
    fr = np.array(res["W7"]["per_landmark_unexplained_fraction"]); ap = set(res["W7"]["end_a_landmarks_0based"]) | set(res["W7"]["end_b_landmarks_0based"])
    sc = ax.scatter(M0[:, 0], M0[:, 1], c=fr, cmap="magma_r", s=120, vmin=0, vmax=1, edgecolors=["red" if i in ap else "none" for i in range(len(fr))], linewidths=1.5)
    plt.colorbar(sc, ax=ax, label="fraction of landmark variance NOT captured by T7")
    ax.set_aspect("equal"); ax.axis("off"); ax.set_title(f"D. Where the global grid fails (W7): ends {res['W7']['unexplained_fraction_end_a']:.2f} / {res['W7']['unexplained_fraction_end_b']:.2f} vs middle {res['W7']['unexplained_fraction_middle']:.2f}", fontsize=9)
    # E: LOO error per species vs tau
    ax = fig.add_subplot(gs[1, :2])
    taus = np.array([r["tau_rel"] for r in recs]); order = np.argsort(taus)
    for k, c, lab in [("rms_full", "#2ca02c", "full space"), ("rms_thompson7", "#1f77b4", "Thompson-7"), ("rms_pca7", "#9467bd", "PCA-7"), ("rms_nearest", "#ff7f0e", "copy nearest"), ("rms_consensus", "#7f7f7f", "consensus")]:
        ax.plot(range(len(recs)), [recs[i][k] for i in order], "o-", color=c, ms=3, lw=1, label=lab)
    ax.set_xticks(range(len(recs))); ax.set_xticklabels([recs[i]["species"].replace("_", " ") for i in order], rotation=75, fontsize=6)
    ax.set_ylabel("Procrustes RMS (leave-one-species-out)"); ax.legend(fontsize=7, ncol=5); ax.grid(alpha=0.3)
    ax.set_title(f"E. Predicting each species' wing from the other {len(recs) - 1} (sorted by tau): full/consensus {res['W3']['ratio']:.2f}, full/nearest {res['W4']['ratio']:.2f}", fontsize=9)
    # F: example predicted vs observed wing (worst and best full-space)
    i_best = int(np.argmin([r["rms_full"] for r in recs])); i_worst = int(np.argmax([r["rms_full"] for r in recs]))
    for col, i in zip([2, 3], [i_best, i_worst]):
        ax = fig.add_subplot(gs[1, col]); r = recs[i]
        obs = np.array(Y_opa[i]); ax.plot(obs[:, 0], obs[:, 1], "ko", ms=4, label="observed species mean")
        ax.plot(M0[:, 0], M0[:, 1], "x", color="#7f7f7f", ms=4, label="consensus")
        ax.set_aspect("equal"); ax.axis("off"); ax.legend(fontsize=7, loc="lower left")
        ax.set_title(f"{'F. best' if col == 2 else 'G. worst'} full-space LOO: {r['species'].replace('_', ' ')}\nRMS full {r['rms_full']:.4f}, T7 {r['rms_thompson7']:.4f}, consensus {r['rms_consensus']:.4f}", fontsize=8)
    fig.suptitle(f"Vespinae hornet wings: {R['n_pass']}/7 registered hypotheses supported (Brier {R['brier']:.2f}). Thompson-7 captures {100 * R['variance']['thompson7']:.0f}% of wing-shape variance vs 58% in fish.", fontsize=11, fontweight="bold")
    fig.tight_layout(); fig.savefig(OUT_FIG, dpi=140); print(f"    wrote {OUT_FIG}")


if __name__ == "__main__":
    sys.exit(main())

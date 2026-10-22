#!/usr/bin/env python3
"""
Pre-register blind body-shape predictions for every species on the Rabosky et al. (2018)
time tree that is NOT in the Claverie & Wainwright (2014) landmark training set.

This is step 1 of a two-step blinded test:
  1. (this script)  predict -> write a locked prediction file + SHA-256 manifest + protocol;
                    commit, push and record the hashes in a nanopublication BEFORE any
                    comparison dataset is downloaded or inspected.
  2. (score_blind_predictions.py)  download an independent morphometric dataset that was
                    never used in this repository (FishShapes, Price et al. 2022, Ecology),
                    verify the hashes, intersect by species name, and score exactly the
                    hypotheses written below.

Models are identical to thompson/run_phylogenetic_prediction_engine.py: 7 linearised
Thompson modes and the full 30-D Procrustes shape space, each under the AICc-best
multivariate process (Pagel lambda) fitted by phylogenetic GLS on the 743 training species;
predictions are Gaussian conditional expectations given the training species.

Outputs (all under the repo root):
  thompson_blind_predictions_registered.json   full record (ratios, 95% MC intervals,
                                               predicted 17-landmark configurations, baselines)
  thompson_blind_predictions_registered.csv    compact per-species table of the scalar predictions
  thompson_blind_predictions_manifest.json     SHA-256 of both files + git commit + protocol
  docs/THOMPSON_BLIND_PREREGISTRATION.md       human-readable protocol and hypotheses
"""
from __future__ import annotations

import csv
import datetime as dt
import hashlib
import importlib.util
import json
import lzma
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_JSON = REPO_ROOT / "thompson_blind_predictions_registered.json"
OUT_CSV = REPO_ROOT / "thompson_blind_predictions_registered.csv"
OUT_MANIFEST = REPO_ROOT / "thompson_blind_predictions_manifest.json"
OUT_MD = REPO_ROOT / "docs" / "THOMPSON_BLIND_PREREGISTRATION.md"
TAXONOMY = REPO_ROOT / ".thompson_cache" / "PFC_taxonomy.csv.xz"
N_MC = 400
SEED = 1917


def load_module(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


eng = load_module("eng", "thompson/run_phylogenetic_prediction_engine.py")
at = eng.at

RATIO_DEFS = {
    "BDmax_over_SL": "maximum body depth excluding fin webs / standard length. Landmark proxy: vertical extent of the 17 landmarks / (snout to caudal-fin insertion midpoint). PRIMARY.",
    "CPD_over_SL": "caudal-peduncle depth / standard length. Landmark proxy: distance between dorsal (LM5) and ventral (LM6) caudal-fin insertions / SL. SECONDARY (the dataset's 'minimum' peduncle depth will be a little smaller than the insertion-to-insertion distance).",
    "HL_over_SL": "head length / standard length. Landmark proxy: snout (mean LM1-2) to opercle landmark (LM11) / SL. EXPLORATORY (only if the dataset reports head length).",
    "PD_over_SL": "predorsal length / standard length. Landmark proxy: snout to dorsal-fin origin (LM3), straight line / SL. EXPLORATORY.",
}

HYPOTHESES = [
    {"id": "H1", "statement": "For target species with a training relative within 50 Myr (tau < 50), the full 30-D phylogenetic prediction of BDmax/SL has smaller absolute error than the grand-consensus value. Test: one-sided paired Wilcoxon signed-rank on |error| across species; also median |error| ratio. Expected: full30 median |error| < 0.8 x consensus median |error|.", "primary": True},
    {"id": "H2", "statement": "For tau < 50, the full 30-D prediction beats copying the nearest training relative's observed landmark ratio (same test). Expected margin smaller than H1; this held at every horizon < 50 Myr in the internal hold-out.", "primary": True},
    {"id": "H3", "statement": "After the pre-specified measurement-convention calibration (Section 'Calibration'), the 95% predictive intervals of BDmax/SL for tau < 50 contain the observed species value in 85-97% of species. Raw (uncalibrated) coverage is also reported.", "primary": True},
    {"id": "H4", "statement": "For tau >= 50, the full 30-D prediction does NOT beat the consensus by more than 10% in median |error| (the ~50 Myr horizon found internally and in the Thompson-pair analysis).", "primary": True},
    {"id": "H5", "statement": "Median |error| of the full 30-D prediction increases monotonically across the tau bins [0,5), [5,10), [10,25), [25,50), [50,80), [80,120), [120,inf) (Spearman rho > 0 on bin medians; at least 5 of 6 consecutive steps non-decreasing).", "primary": False},
    {"id": "H6", "statement": "The a-priori 7-Thompson-mode prediction is worse than the full 30-D prediction (median |error| larger) but still beats the consensus for tau < 50.", "primary": False},
    {"id": "H7", "statement": "Secondary ratio CPD/SL: full 30-D beats consensus for tau < 50 (as H1). Exploratory HL/SL, PD/SL likewise if present in the dataset.", "primary": False},
]

PROTOCOL = {
    "comparison_dataset": "FishShapes v1 (Price, Larouche, Friedman, Corn, Wainwright & Martinez 2022, Ecology 103:e3829; Dryad). Linear measurements of museum specimens for ~6,000 teleost species. Never downloaded, opened, or used in this repository before the registration commit; the FishBase morphometric tables already in .thompson_cache/ are NOT the comparison dataset.",
    "fallback_dataset": "If FishShapes cannot be obtained, any independent published dataset of standard length and maximum body depth per species not previously used here, declared before opening it.",
    "species_matching": "Exact match of 'Genus_species' after normalising case, whitespace/underscore and stripping subspecies/author strings; no fuzzy matching. Synonyms are not resolved (a conservative choice that only loses power).",
    "per_species_value": "Median across specimens of (measurement / standard length) computed per specimen. Specimens with missing SL or measurement are dropped; species need >= 1 specimen.",
    "exclusions": "The 743 training species and the 7 digitised 1917 museum species are excluded from scoring (they may be used only for the convention calibration below). Species not on the Rabosky tree cannot be predicted and are excluded.",
    "calibration": "Landmark-based ratios differ systematically from calliper ratios (e.g. landmark BDmax is the vertical landmark extent). A single linear map obs = alpha + beta * landmark_ratio is fitted by least squares on the TRAINING species present in the comparison dataset, using their OBSERVED landmark ratios (not predictions). It is applied to predicted point estimates and interval bounds of every method (prediction, consensus, nearest-relative copy, family mean alike), so it cannot favour one method. Raw results are reported alongside.",
    "horizon_tau": "tau = age (Myr) of the MRCA between the target and its nearest training species on the Rabosky tree, computed from the tree alone, before any data are seen. Species outside the sampled crown clade get tau = crown age and the root-state prediction.",
    "error_metric": "absolute error |predicted ratio - observed species ratio|; summarised by median within tau bins and overall; paired one-sided Wilcoxon signed-rank tests, alpha = 0.01 for primary hypotheses, with n reported.",
    "no_tuning": "No model, basis, bin edge, exclusion or metric may be changed after the comparison data are opened. Anything learned afterwards is exploratory and will be labelled as such in the results document.",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


def ratios_vec(L: np.ndarray) -> dict:
    """L: (..., 17, 2). Returns dict of arrays (...)."""
    snout = 0.5 * (L[..., 0, :] + L[..., 1, :])
    caud = 0.5 * (L[..., 4, :] + L[..., 5, :])
    sl = np.linalg.norm(snout - caud, axis=-1)
    bdmax = L[..., :, 1].max(-1) - L[..., :, 1].min(-1)
    cpd = np.linalg.norm(L[..., 4, :] - L[..., 5, :], axis=-1)
    hl = np.linalg.norm(snout - L[..., 10, :], axis=-1)
    pd_ = np.linalg.norm(L[..., 2, :] - snout, axis=-1)
    return {"BDmax_over_SL": bdmax / sl, "CPD_over_SL": cpd / sl, "HL_over_SL": hl / sl, "PD_over_SL": pd_ / sl}


def main() -> int:
    print("[1] landmarks, GPA, consensus, bases")
    names, families, X = at.load_landmarks()
    Y = at.gpa(X)
    M0 = at.centre_scale(Y.mean(0))
    Y_opa = np.array([at.opa(y, M0) for y in Y])
    B, _ = eng.thompson_mode_basis(M0)
    m_dim = B.shape[1]
    D_all = (Y_opa - M0).reshape(len(names), -1)
    Z_all = eng.project(B, D_all)

    print("[2] full Rabosky tree; sampled crown; heights relative to the sampled crown")
    tree_path = at.fetch_phylogeny(allow_download=False)
    with lzma.open(tree_path, "rt") as fh:
        parent_all, depth_all, tip_to_id = at.parse_newick_depths(fh.read())
    exact_meta = json.loads(at.EXACT_SPECIES_JSON.read_text(encoding="utf-8"))["specimens"]
    exact_names = set(exact_meta)
    keep = {n: tip_to_id[n] for n in names if n in tip_to_id}
    tree = eng.Tree(parent_all, depth_all, keep)            # pruned tree for fitting (same as engine)
    eng.Model.T = tree.T
    train_names = [n for n in names if n in tree.name_to_node]
    tips = np.array([tree.name_to_node[n] for n in train_names])
    name_idx = {n: i for i, n in enumerate(names)}
    tr = np.array([name_idx[n] for n in train_names])
    Zt = Z_all[tr]
    Dt = D_all[tr]
    n = len(train_names)
    Dc = Dt.mean(0)
    _, S_, Vt_ = np.linalg.svd(Dt - Dc, full_matrices=False)
    P_full = Vt_[:30].T
    F_t = (Dt - Dc) @ P_full
    print(f"    {n} training tips; sampled crown age {tree.T:.1f} Ma")

    # crown node of the training tips in the FULL tree, and depth offset
    train_ids = [tip_to_id[nm] for nm in train_names]
    anc_count = defaultdict(int)
    for u in train_ids:
        c = u
        while c != -1:
            anc_count[c] += 1
            c = parent_all[c]
    crown_full = max((u for u, k in anc_count.items() if k == len(train_ids)), key=lambda u: depth_all[u])
    depth0 = depth_all[crown_full]
    depth_all = np.asarray(depth_all, dtype=float)
    parent_all = np.asarray(parent_all)
    T_full = float(depth_all.max() - depth0)
    assert abs(T_full - tree.T) < 1e-6, (T_full, tree.T)

    # observed tips under each node of the full tree (for MRCA heights target x observed)
    obs_pos = {tip_to_id[nm]: j for j, nm in enumerate(train_names)}
    under: dict[int, list] = defaultdict(list)
    for u, j in obs_pos.items():
        c = u
        while c != -1:
            under[c].append(j)
            c = parent_all[c]
    under = {k: np.array(v) for k, v in under.items()}

    def mrca_row(tip_id: int):
        row = np.full(n, 0.0)
        assigned = np.zeros(n, bool)
        c = tip_id
        while c != -1 and c != parent_all[crown_full]:
            idx = under.get(c)
            if idx is not None:
                new = idx[~assigned[idx]]
                row[new] = max(depth_all[c] - depth0, 0.0)
                assigned[new] = True
            c = parent_all[c]
        return row  # unassigned (target outside the crown) stay 0 = root

    print("[3] fit AICc-best process on Thompson-7 and full-30 spaces")
    fits_body = eng.fit_models(tree, tips, Zt)
    fits_full = eng.fit_models(tree, tips, F_t)
    model = fits_body[min(fits_body, key=lambda k: fits_body[k]["AICc"])]["model"]
    model_full = fits_full[min(fits_full, key=lambda k: fits_full[k]["AICc"])]["model"]
    print(f"    Thompson-7: {model.label()};  full-30: {model_full.label()}")

    # taxonomy for family-mean baseline
    fam_of_target = {}
    with lzma.open(TAXONOMY, "rt") as fh:
        for row in csv.DictReader(fh):
            fam_of_target[row["genus.species"].replace(" ", "_")] = row["family"]
    fam_of_train = dict(zip(names, families))
    fam_members = defaultdict(list)
    for nm in train_names:
        fam_members[fam_of_train[nm]].append(name_idx[nm])
    fam_mean_ratio = {}
    for f, idxs in fam_members.items():
        if len(idxs) >= 3:
            fam_mean_ratio[f] = {k: float(v) for k, v in ratios_vec(Y_opa[idxs].mean(0)).items()}

    print("[4] targets: every tip not in training / 1917 set")
    targets = [nm for nm in tip_to_id if nm not in keep and nm not in exact_names]
    targets.sort()
    print(f"    {len(targets)} target species")
    rows_C = np.array([mrca_row(tip_to_id[nm]) for nm in targets])
    rows_h = np.full(len(targets), tree.T)
    outside = rows_C.max(1) <= 1e-9
    tau = tree.T - rows_C.max(1)
    nearest = rows_C.argmax(1)

    print("[5] predict")
    rows_C_nodes = np.zeros((len(targets), tree.N))          # predict() indexes columns by pruned-tree node id
    rows_C_nodes[:, tips] = rows_C
    mean7, s2_7, a7, R7 = eng.predict(model, tree, tips, Zt, rows_h, rows_C_nodes)
    meanF, s2_F, aF, RF = eng.predict(model_full, tree, tips, F_t, rows_h, rows_C_nodes)
    del rows_C_nodes
    base = M0.reshape(-1)
    lm7 = (base + mean7 @ B.T).reshape(-1, 17, 2)
    lmF = (base + Dc + meanF @ P_full.T).reshape(-1, 17, 2)
    r7 = ratios_vec(lm7)
    rF = ratios_vec(lmF)
    cons = {k: float(v) for k, v in ratios_vec(M0).items()}
    train_ratios = ratios_vec(Y_opa[tr])  # per training species (observed landmarks)

    print("[6] Monte Carlo 95% intervals")
    rng = np.random.default_rng(SEED)
    L7 = np.linalg.cholesky(R7 + 1e-12 * np.eye(m_dim))
    LF = np.linalg.cholesky(RF + 1e-12 * np.eye(30))
    eps7 = rng.standard_normal((N_MC, m_dim)) @ L7.T
    epsF = rng.standard_normal((N_MC, 30)) @ LF.T
    ci7 = {k: np.zeros((len(targets), 2)) for k in RATIO_DEFS}
    ciF = {k: np.zeros((len(targets), 2)) for k in RATIO_DEFS}
    for i in range(len(targets)):
        z = mean7[i] + np.sqrt(s2_7[i]) * eps7
        f = meanF[i] + np.sqrt(s2_F[i]) * epsF
        rr7 = ratios_vec((base + z @ B.T).reshape(N_MC, 17, 2))
        rrF = ratios_vec((base + Dc + f @ P_full.T).reshape(N_MC, 17, 2))
        for k in RATIO_DEFS:
            ci7[k][i] = np.percentile(rr7[k], [2.5, 97.5])
            ciF[k][i] = np.percentile(rrF[k], [2.5, 97.5])
        if i % 2000 == 0:
            print(f"    {i}/{len(targets)}")

    print("[7] write")
    commit = git_commit()
    today = dt.date.today().isoformat()
    preds = []
    for i, nm in enumerate(targets):
        nn = train_names[nearest[i]]
        fam = fam_of_target.get(nm, "")
        rec = {
            "species": nm, "family_from_taxonomy": fam, "tau_myr": round(float(tau[i]), 3), "outside_sampled_crown": bool(outside[i]),
            "nearest_training_species": nn,
            "full30": {k: {"pred": round(float(rF[k][i]), 4), "lo95": round(float(ciF[k][i, 0]), 4), "hi95": round(float(ciF[k][i, 1]), 4)} for k in RATIO_DEFS},
            "thompson7": {k: {"pred": round(float(r7[k][i]), 4), "lo95": round(float(ci7[k][i, 0]), 4), "hi95": round(float(ci7[k][i, 1]), 4)} for k in RATIO_DEFS},
            "baseline_consensus": {k: round(cons[k], 4) for k in RATIO_DEFS},
            "baseline_nearest_training_species": {k: round(float(train_ratios[k][nearest[i]]), 4) for k in RATIO_DEFS},
            "baseline_training_family_mean": ({k: round(v, 4) for k, v in fam_mean_ratio[fam].items()} if fam in fam_mean_ratio else None),
            "pred_landmarks_full30": [[round(float(x), 5) for x in p] for p in at.centre_scale(lmF[i])],
        }
        preds.append(rec)

    out = {
        "registration": {"date": today, "git_commit_at_generation": commit, "script": "thompson/preregister_blind_predictions.py",
                         "n_targets": len(targets), "n_training_species": n, "n_outside_sampled_crown": int(outside.sum()),
                         "tree": "Rabosky et al. 2018 actinopt_12k_treePL", "sampled_crown_age_ma": tree.T,
                         "model_thompson7": model.label(), "model_full30": model_full.label(), "n_mc": N_MC, "seed": SEED},
        "ratio_definitions": RATIO_DEFS, "hypotheses": HYPOTHESES, "protocol": PROTOCOL,
        "landmark_scheme": "17 Claverie & Wainwright (2014) landmarks, Procrustes-aligned (unit centroid size, centred, consensus snout-caudal axis horizontal)",
        "tau_bins_myr": [[0, 5], [5, 10], [10, 25], [25, 50], [50, 80], [80, 120], [120, 1000]],
        "predictions": preds,
    }
    OUT_JSON.write_text(json.dumps(out, separators=(",", ":")), encoding="utf-8")
    with OUT_CSV.open("w", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        hdr = ["species", "family", "tau_myr", "outside_crown", "nearest_training_species"]
        for k in RATIO_DEFS:
            hdr += [f"full30_{k}", f"full30_{k}_lo95", f"full30_{k}_hi95", f"thompson7_{k}", f"thompson7_{k}_lo95", f"thompson7_{k}_hi95", f"consensus_{k}", f"nearest_{k}", f"fammean_{k}"]
        w.writerow(hdr)
        for p in preds:
            row = [p["species"], p["family_from_taxonomy"], p["tau_myr"], int(p["outside_sampled_crown"]), p["nearest_training_species"]]
            for k in RATIO_DEFS:
                fm = p["baseline_training_family_mean"]
                row += [p["full30"][k]["pred"], p["full30"][k]["lo95"], p["full30"][k]["hi95"], p["thompson7"][k]["pred"], p["thompson7"][k]["lo95"], p["thompson7"][k]["hi95"],
                        p["baseline_consensus"][k], p["baseline_nearest_training_species"][k], (fm[k] if fm else "")]
            w.writerow(row)
    manifest = {"registered_on": today, "git_commit_at_generation": commit,
                "files": {OUT_JSON.name: sha256(OUT_JSON), OUT_CSV.name: sha256(OUT_CSV)},
                "n_targets": len(targets), "hypotheses": [h["id"] for h in HYPOTHESES],
                "note": "Hashes are recorded in a nanopublication at registration time; score_blind_predictions.py refuses to run if they do not match."}
    OUT_MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    bins = out["tau_bins_myr"]
    counts = [int(np.sum((tau >= lo) & (tau < hi))) for lo, hi in bins]
    L = ["# Pre-registered blind predictions: Thompson 1917 phylogenetic prediction engine", "",
         f"Registered {today} at commit `{commit[:12]}` by [`thompson/preregister_blind_predictions.py`](../thompson/preregister_blind_predictions.py). "
         f"Predictions: [`thompson_blind_predictions_registered.json`](../thompson_blind_predictions_registered.json) (SHA-256 `{manifest['files'][OUT_JSON.name]}`), "
         f"[`thompson_blind_predictions_registered.csv`](../thompson_blind_predictions_registered.csv) (SHA-256 `{manifest['files'][OUT_CSV.name]}`).", "",
         "## What is predicted", "",
         f"Body-proportion ratios and the full 17-landmark configuration for **{len(targets):,} species** on the Rabosky et al. (2018) tree that are not among the {n} landmarked training species "
         f"(nor the 7 digitised 1917 species). {int(outside.sum())} targets lie outside the sampled crown clade and receive the root-state prediction. "
         f"Models: Thompson-7 = {model.label()}; full-30 = {model_full.label()}; {N_MC} Monte Carlo draws per species for 95% intervals (seed {SEED}).", "",
         "| tau bin (Myr) | " + " | ".join(f"{lo}-{hi}" for lo, hi in bins) + " |", "|---|" + "---|" * len(bins),
         "| n targets | " + " | ".join(str(c) for c in counts) + " |", "",
         "## Ratio definitions", ""] + [f"- **{k}**: {v}" for k, v in RATIO_DEFS.items()] + ["", "## Hypotheses (fixed before any comparison data were opened)", ""] + \
        [f"- **{h['id']}**{' (primary)' if h['primary'] else ''}: {h['statement']}" for h in HYPOTHESES] + ["", "## Protocol", ""] + \
        [f"- **{k.replace('_', ' ')}**: {v}" for k, v in PROTOCOL.items()] + ["",
         "## Status", "", "Step 1 (this document) is complete once the files above are committed and their hashes are recorded in a nanopublication. "
         "Step 2 is `thompson/score_blind_predictions.py`, which verifies the hashes before reading any comparison data and writes `docs/THOMPSON_BLIND_TEST_RESULTS.md`."]
    OUT_MD.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"    wrote {OUT_JSON.name} ({OUT_JSON.stat().st_size / 1e6:.1f} MB), {OUT_CSV.name}, {OUT_MANIFEST.name}, {OUT_MD.relative_to(REPO_ROOT)}")
    print(f"    tau bins {bins}: {counts}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

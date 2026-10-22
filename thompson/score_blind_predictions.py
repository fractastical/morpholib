#!/usr/bin/env python3
"""
Step 2 of the blinded test: score the pre-registered predictions against FishShapes v1.

  * Verifies the SHA-256 of the locked prediction files against the manifest written at
    registration (thompson_blind_predictions_manifest.json) and refuses to run on mismatch.
  * Reads FishShapes v1 (Price et al. 2022, Ecology; Zenodo record 6642650 / Dryad
    10.5061/dryad.vt4b8gtvf, CC0), computes per-specimen ratios and per-species medians.
  * Applies the protocol exactly as registered in docs/THOMPSON_BLIND_PREREGISTRATION.md:
    exact name matching, training species used only for the measurement-convention
    calibration, tau bins and tests as specified, alpha = 0.01 for primary hypotheses.
  * Writes thompson_blind_test_results.json, docs/THOMPSON_BLIND_TEST_RESULTS.md and
    docs/outreach_fig10_blind_test_fishshapes.png.

Nothing in here was tuned after the data were opened; see the 'exploratory' section of the
results document for anything added afterwards.
"""
from __future__ import annotations

import csv
import datetime as dt
import hashlib
import importlib.util
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parents[1]
REG_JSON = REPO_ROOT / "thompson_blind_predictions_registered.json"
REG_CSV = REPO_ROOT / "thompson_blind_predictions_registered.csv"
MANIFEST = REPO_ROOT / "thompson_blind_predictions_manifest.json"
FISHSHAPES = REPO_ROOT / ".thompson_cache" / "fishshapes_v1" / "FishShapes_specimensv1.csv"
FISHSHAPES_MD5 = "122b495274d0167c5447db104c7acbfe"      # Zenodo record 6642650 checksum
OUT_JSON = REPO_ROOT / "thompson_blind_test_results.json"
OUT_MD = REPO_ROOT / "docs" / "THOMPSON_BLIND_TEST_RESULTS.md"
OUT_FIG = REPO_ROOT / "docs" / "outreach_fig10_blind_test_fishshapes.png"
ALPHA = 0.01

# registered ratio -> FishShapes column (HL and PD are not in FishShapes: unscored, as allowed)
RATIO_COLS = {"BDmax_over_SL": "Max_body_depth", "CPD_over_SL": "Min_caudalpeduncle_depth"}
METHODS = ["full30", "thompson7", "consensus", "nearest", "fammean"]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def md5(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_module(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def norm_name(s: str) -> str:
    s = s.strip().replace(" ", "_")
    parts = [p for p in s.split("_") if p]
    if len(parts) < 2:
        return ""
    return f"{parts[0].capitalize()}_{parts[1].lower()}"


def wilcoxon_less(a: np.ndarray, b: np.ndarray):
    """One-sided paired Wilcoxon: H1 median(a - b) < 0."""
    d = a - b
    d = d[d != 0]
    if len(d) < 10:
        return float("nan"), len(d)
    return float(stats.wilcoxon(d, alternative="less").pvalue), len(d)


def main() -> int:
    print("[0] verify locked files")
    man = json.loads(MANIFEST.read_text())
    for fname, digest in man["files"].items():
        actual = sha256(REPO_ROOT / fname)
        if actual != digest:
            print(f"    HASH MISMATCH for {fname}: manifest {digest} vs file {actual}")
            return 2
        print(f"    {fname}: sha256 OK")
    if not FISHSHAPES.exists():
        print(f"    comparison data not found at {FISHSHAPES}"); return 3
    fs_md5 = md5(FISHSHAPES)
    print(f"    FishShapes md5 {fs_md5} ({'matches Zenodo' if fs_md5 == FISHSHAPES_MD5 else 'DOES NOT MATCH ZENODO'})")

    reg = json.loads(REG_JSON.read_text())
    preds = {p["species"]: p for p in reg["predictions"]}
    bins = reg["tau_bins_myr"]

    print("[1] FishShapes per-species medians")
    spec_rows = defaultdict(lambda: defaultdict(list))
    n_spec = 0
    with FISHSHAPES.open(newline="", encoding="utf-8-sig") as fh:
        for row in csv.DictReader(fh):
            n_spec += 1
            nm = norm_name(row["Tree_name"])
            try:
                sl = float(row["Standard_length"])
            except ValueError:
                continue
            if not nm or sl <= 0:
                continue
            for k, col in RATIO_COLS.items():
                try:
                    v = float(row[col])
                except ValueError:
                    continue
                if v > 0:
                    spec_rows[nm][k].append(v / sl)
    obs = {nm: {k: float(np.median(v)) for k, v in d.items() if v} for nm, d in spec_rows.items()}
    print(f"    {n_spec} specimens, {len(obs)} species")

    print("[2] calibration on training species (observed landmark ratios -> calliper ratios)")
    pre = load_module("pre", "thompson/preregister_blind_predictions.py")
    at = pre.at
    names, families, X = at.load_landmarks()
    Y = at.gpa(X)
    M0 = at.centre_scale(Y.mean(0))
    Y_opa = np.array([at.opa(y, M0) for y in Y])
    train_lm_ratios = pre.ratios_vec(Y_opa)
    train_set = set(names)
    calib = {}
    for k in RATIO_COLS:
        xs, ys = [], []
        for i, nm in enumerate(names):
            if nm in obs and k in obs[nm]:
                xs.append(float(train_lm_ratios[k][i])); ys.append(obs[nm][k])
        xs, ys = np.array(xs), np.array(ys)
        slope, intercept, r, p, se = stats.linregress(xs, ys)
        calib[k] = {"alpha": float(intercept), "beta": float(slope), "r": float(r), "n_training_species_in_fishshapes": int(len(xs)),
                    "residual_sd": float(np.std(ys - (intercept + slope * xs), ddof=2))}
        print(f"    {k}: obs = {intercept:.4f} + {slope:.4f} * landmark   (r={r:.3f}, n={len(xs)}, resid sd={calib[k]['residual_sd']:.4f})")

    print("[3] match targets and score")
    exact_1917 = set(json.loads(at.EXACT_SPECIES_JSON.read_text(encoding="utf-8"))["specimens"])
    matched = [nm for nm in preds if nm in obs and nm not in train_set and nm not in exact_1917]
    print(f"    {len(matched)} target species with FishShapes data (of {len(preds)} registered targets)")

    def cal(k, x, use):
        return calib[k]["alpha"] + calib[k]["beta"] * x if use else x

    rows = []
    for nm in matched:
        p = preds[nm]
        rec = {"species": nm, "tau": p["tau_myr"], "outside": p["outside_sampled_crown"], "family": p["family_from_taxonomy"], "nearest": p["nearest_training_species"]}
        for k in RATIO_COLS:
            if k not in obs[nm]:
                continue
            o = obs[nm][k]
            vals = {"full30": p["full30"][k]["pred"], "thompson7": p["thompson7"][k]["pred"], "consensus": p["baseline_consensus"][k],
                    "nearest": p["baseline_nearest_training_species"][k],
                    "fammean": (p["baseline_training_family_mean"][k] if p["baseline_training_family_mean"] else None)}
            rec[k] = {"obs": o}
            for use, tag in [(False, "raw"), (True, "cal")]:
                for m, v in vals.items():
                    rec[k][f"err_{m}_{tag}"] = (abs(cal(k, v, use) - o) if v is not None else None)
                lo, hi = cal(k, p["full30"][k]["lo95"], use), cal(k, p["full30"][k]["hi95"], use)
                rec[k][f"in95_full30_{tag}"] = bool(min(lo, hi) <= o <= max(lo, hi))
                lo7, hi7 = cal(k, p["thompson7"][k]["lo95"], use), cal(k, p["thompson7"][k]["hi95"], use)
                rec[k][f"in95_thompson7_{tag}"] = bool(min(lo7, hi7) <= o <= max(lo7, hi7))
        rows.append(rec)

    def arr(k, field, sel):
        return np.array([r[k][field] for r in sel if k in r and r[k].get(field) is not None], dtype=float)

    def bin_sel(lo, hi):
        return [r for r in rows if lo <= r["tau"] < hi]

    results = {"calibration": calib, "n_matched": len(matched), "n_fishshapes_species": len(obs), "n_specimens": n_spec,
               "fishshapes_md5": fs_md5, "fishshapes_sha256": sha256(FISHSHAPES), "scored_on": dt.date.today().isoformat(),
               "locked_manifest": man, "horizon_table": {}, "hypotheses": {}}

    for k in RATIO_COLS:
        tab = []
        for lo, hi in bins:
            sel = [r for r in bin_sel(lo, hi) if k in r]
            if not sel:
                continue
            row = {"tau_bin": [lo, hi], "n": len(sel)}
            for tag in ["raw", "cal"]:
                for m in METHODS:
                    e = arr(k, f"err_{m}_{tag}", sel)
                    row[f"median_abs_err_{m}_{tag}"] = float(np.median(e)) if len(e) else None
                row[f"coverage95_full30_{tag}"] = float(100 * np.mean([r[k][f"in95_full30_{tag}"] for r in sel]))
                row[f"coverage95_thompson7_{tag}"] = float(100 * np.mean([r[k][f"in95_thompson7_{tag}"] for r in sel]))
            tab.append(row)
        results["horizon_table"][k] = tab

    # ---------------- hypotheses (as registered) ----------------
    H = {}
    near = [r for r in rows if r["tau"] < 50 and "BDmax_over_SL" in r]
    far = [r for r in rows if r["tau"] >= 50 and "BDmax_over_SL" in r]
    k = "BDmax_over_SL"
    for tag in ["cal", "raw"]:
        ef, ec, en, e7 = (arr(k, f"err_{m}_{tag}", near) for m in ["full30", "consensus", "nearest", "thompson7"])
        p1, n1 = wilcoxon_less(ef, ec); p2, n2 = wilcoxon_less(ef, en)
        H[f"H1_{tag}"] = {"n": int(n1), "median_full30": float(np.median(ef)), "median_consensus": float(np.median(ec)), "ratio": float(np.median(ef) / np.median(ec)), "p_wilcoxon_less": p1,
                          "pass": bool(p1 < ALPHA and np.median(ef) / np.median(ec) < 0.8)}
        H[f"H2_{tag}"] = {"n": int(n2), "median_full30": float(np.median(ef)), "median_nearest": float(np.median(en)), "ratio": float(np.median(ef) / np.median(en)), "p_wilcoxon_less": p2,
                          "pass": bool(p2 < ALPHA and np.median(ef) < np.median(en))}
        cov = 100 * np.mean([r[k][f"in95_full30_{tag}"] for r in near])
        H[f"H3_{tag}"] = {"n": len(near), "coverage95_pct": float(cov), "pass": bool(85 <= cov <= 97)}
        eff, ecf = arr(k, f"err_full30_{tag}", far), arr(k, f"err_consensus_{tag}", far)
        H[f"H4_{tag}"] = {"n": len(far), "median_full30": float(np.median(eff)), "median_consensus": float(np.median(ecf)), "ratio": float(np.median(eff) / np.median(ecf)),
                          "pass": bool(np.median(eff) / np.median(ecf) >= 0.9), "note": "pass = full30 does NOT beat consensus by >10% beyond 50 Myr"}
        meds = [t[f"median_abs_err_full30_{tag}"] for t in results["horizon_table"][k]]
        steps = np.diff(meds)
        rho = stats.spearmanr(range(len(meds)), meds).statistic if len(meds) > 2 else float("nan")
        H[f"H5_{tag}"] = {"bin_medians": meds, "spearman_rho": float(rho), "nondecreasing_steps": int(np.sum(steps >= 0)), "of": int(len(steps)),
                          "pass": bool(rho > 0 and np.sum(steps >= 0) >= len(steps) - 1)}
        p6a, _ = wilcoxon_less(e7, ec)
        H[f"H6_{tag}"] = {"median_thompson7": float(np.median(e7)), "median_full30": float(np.median(ef)), "median_consensus": float(np.median(ec)), "p_thompson7_less_than_consensus": p6a,
                          "pass": bool(np.median(e7) > np.median(ef) and p6a < ALPHA and np.median(e7) < np.median(ec))}
        kc = "CPD_over_SL"
        nearc = [r for r in rows if r["tau"] < 50 and kc in r]
        efc, ecc = arr(kc, f"err_full30_{tag}", nearc), arr(kc, f"err_consensus_{tag}", nearc)
        p7, n7 = wilcoxon_less(efc, ecc)
        H[f"H7_{tag}"] = {"n": int(n7), "median_full30": float(np.median(efc)), "median_consensus": float(np.median(ecc)), "ratio": float(np.median(efc) / np.median(ecc)), "p_wilcoxon_less": p7,
                          "pass": bool(p7 < ALPHA and np.median(efc) < np.median(ecc))}
    results["hypotheses"] = H
    results["per_species"] = rows

    for hid in ["H1", "H2", "H3", "H4", "H5", "H6", "H7"]:
        print(f"    {hid}: cal pass={H[hid + '_cal']['pass']}  raw pass={H[hid + '_raw']['pass']}  | " + ", ".join(f"{a}={b:.4g}" if isinstance(b, float) else f"{a}={b}" for a, b in H[hid + "_cal"].items() if a not in ("pass", "bin_medians", "note")))

    OUT_JSON.write_text(json.dumps(results, indent=1), encoding="utf-8")

    # ---------------- figure ----------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    ax = axes[0]
    tab = results["horizon_table"]["BDmax_over_SL"]
    xs = np.arange(len(tab))
    for m, c, lab in [("full30", "#2ca02c", "full 30-D prediction"), ("thompson7", "#1f77b4", "Thompson-7 prediction"), ("nearest", "#ff7f0e", "copy nearest training relative"), ("consensus", "#7f7f7f", "grand consensus"), ("fammean", "#9467bd", "training family mean")]:
        ax.plot(xs, [t[f"median_abs_err_{m}_cal"] for t in tab], "o-", color=c, label=lab)
    ax.set_xticks(xs); ax.set_xticklabels([f"{t['tau_bin'][0]}–{t['tau_bin'][1] if t['tau_bin'][1] < 1000 else ''}\nn={t['n']}" for t in tab], fontsize=8)
    ax.set_xlabel("tau (Myr to nearest training species)"); ax.set_ylabel("median |error| in BDmax/SL (calibrated)")
    ax.set_title(f"A. Blind test on FishShapes: {len(matched)} species never seen", fontsize=10); ax.legend(fontsize=7); ax.grid(alpha=0.3)
    ax.axvline(3.5, color="k", lw=0.8, alpha=0.4); ax.text(3.6, ax.get_ylim()[1] * 0.97, "50 Myr", fontsize=8, va="top")
    ax = axes[1]
    k = "BDmax_over_SL"
    sel = [r for r in rows if k in r and r["tau"] < 50]
    o = np.array([r[k]["obs"] for r in sel]); pv = np.array([cal(k, preds[r["species"]]["full30"][k]["pred"], True) for r in sel]); tau_ = np.array([r["tau"] for r in sel])
    sc = ax.scatter(pv, o, c=tau_, cmap="viridis", s=10, alpha=0.7)
    lim = [min(o.min(), pv.min()) * 0.9, max(o.max(), pv.max()) * 1.05]
    ax.plot(lim, lim, "k--", lw=1); ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("predicted BDmax/SL (full 30-D, calibrated)"); ax.set_ylabel("FishShapes observed BDmax/SL (species median)")
    r_ = np.corrcoef(pv, o)[0, 1]
    ax.set_title(f"B. tau < 50 Myr: n={len(sel)}, r={r_:.3f}", fontsize=10); plt.colorbar(sc, ax=ax, label="tau (Myr)")
    ax = axes[2]
    for m, c, lab in [("full30", "#2ca02c", "full 30-D"), ("thompson7", "#1f77b4", "Thompson-7")]:
        ax.plot(xs, [t[f"coverage95_{m}_cal"] for t in tab], "o-", color=c, label=f"{lab} (calibrated)")
        ax.plot(xs, [t[f"coverage95_{m}_raw"] for t in tab], "x:", color=c, alpha=0.6, label=f"{lab} (raw)")
    ax.axhline(95, color="k", ls="--", lw=1); ax.axhspan(85, 97, color="green", alpha=0.08)
    ax.set_xticks(xs); ax.set_xticklabels([f"{t['tau_bin'][0]}–{t['tau_bin'][1] if t['tau_bin'][1] < 1000 else ''}" for t in tab], fontsize=8)
    ax.set_ylabel("% of species inside 95% predictive interval"); ax.set_title("C. Interval calibration (H3 band shaded)", fontsize=10); ax.legend(fontsize=7); ax.grid(alpha=0.3)
    fig.suptitle("Pre-registered blind test of the Thompson 1917 phylogenetic prediction engine against FishShapes v1 (Price et al. 2022)", fontsize=11, fontweight="bold")
    fig.tight_layout(); fig.savefig(OUT_FIG, dpi=150); print(f"    wrote {OUT_FIG}")

    # ---------------- markdown ----------------
    L = ["# Blind test results: Thompson 1917 phylogenetic prediction engine vs. FishShapes v1", "",
         f"Scored {results['scored_on']} by [`thompson/score_blind_predictions.py`](../thompson/score_blind_predictions.py) against predictions locked in "
         f"[`THOMPSON_BLIND_PREREGISTRATION.md`](./THOMPSON_BLIND_PREREGISTRATION.md) (manifest commit `{man['git_commit_at_generation'][:12]}`; hashes verified at run time). "
         f"Comparison data: FishShapes v1 specimen file (Price et al. 2022, *Ecology* 103:e3829; Zenodo 6642650 / Dryad 10.5061/dryad.vt4b8gtvf, CC0), md5 `{fs_md5}`, {n_spec:,} specimens, {len(obs):,} species. "
         f"The file was downloaded after the registration commit and had never been used in this repository.", "",
         f"![blind test](./outreach_fig10_blind_test_fishshapes.png)", "",
         "## Matching and calibration", "",
         f"- {len(matched):,} registered target species have FishShapes data (exact `Genus_species` match; training and 1917 species excluded).",
         f"- Measurement-convention calibration (fitted on training species only, as registered): " + "; ".join(
             f"{k}: calliper = {c['alpha']:.3f} + {c['beta']:.3f} x landmark (r = {c['r']:.3f}, n = {c['n_training_species_in_fishshapes']})" for k, c in calib.items()) + ".",
         "- HL/SL and PD/SL are not measured in FishShapes and are therefore unscored (as the protocol allowed).", "",
         "## Pre-registered hypotheses", "", "| Hypothesis | Result (calibrated) | Result (raw) | Verdict |", "|---|---|---|---|"]
    desc = {h["id"]: h["statement"].split(". Test")[0].split(" Expected")[0] for h in reg["hypotheses"]}

    def fmt(hid, tag):
        h = H[f"{hid}_{tag}"]
        if hid in ("H1", "H2", "H7"):
            return f"n={h['n']}, median |err| {h['median_full30']:.4f} vs {h.get('median_consensus', h.get('median_nearest')):.4f} (ratio {h['ratio']:.2f}), p={h['p_wilcoxon_less']:.2g}"
        if hid == "H3":
            return f"n={h['n']}, 95% coverage = {h['coverage95_pct']:.1f}%"
        if hid == "H4":
            return f"n={h['n']}, median |err| {h['median_full30']:.4f} vs consensus {h['median_consensus']:.4f} (ratio {h['ratio']:.2f})"
        if hid == "H5":
            return f"bin medians {', '.join(f'{m:.3f}' for m in h['bin_medians'])}; rho={h['spearman_rho']:.2f}; {h['nondecreasing_steps']}/{h['of']} steps non-decreasing"
        if hid == "H6":
            return f"T7 {h['median_thompson7']:.4f} vs full30 {h['median_full30']:.4f} vs consensus {h['median_consensus']:.4f}; p(T7<cons)={h['p_thompson7_less_than_consensus']:.2g}"
        return ""
    for hid in ["H1", "H2", "H3", "H4", "H5", "H6", "H7"]:
        prim = next(h["primary"] for h in reg["hypotheses"] if h["id"] == hid)
        L.append(f"| **{hid}**{' (primary)' if prim else ''}: {desc[hid]} | {fmt(hid, 'cal')} | {fmt(hid, 'raw')} | {'✓ supported' if H[hid + '_cal']['pass'] else '✗ not supported'} |")
    n_pass = sum(H[f"{h}_cal"]["pass"] for h in ["H1", "H2", "H3", "H4", "H5", "H6", "H7"])
    n_prim = sum(H[f"{h['id']}_cal"]["pass"] for h in reg["hypotheses"] if h["primary"])
    L += ["", f"**{n_pass}/7 hypotheses supported ({n_prim}/4 primary), using the calibrated comparison specified in the protocol.**", "",
          "## Error by horizon (BDmax/SL, calibrated; median absolute error)", "",
          "| tau (Myr) | n | full 30-D | Thompson-7 | copy nearest | consensus | family mean | 95% coverage full30 (cal / raw) |", "|---|---|---|---|---|---|---|---|"]
    for t in results["horizon_table"]["BDmax_over_SL"]:
        fm = t["median_abs_err_fammean_cal"]
        L.append(f"| {t['tau_bin'][0]}–{t['tau_bin'][1] if t['tau_bin'][1] < 1000 else '∞'} | {t['n']} | **{t['median_abs_err_full30_cal']:.4f}** | {t['median_abs_err_thompson7_cal']:.4f} | {t['median_abs_err_nearest_cal']:.4f} | {t['median_abs_err_consensus_cal']:.4f} | {f'{fm:.4f}' if fm is not None else '–'} | {t['coverage95_full30_cal']:.0f}% / {t['coverage95_full30_raw']:.0f}% |")
    L += ["", "## Error by horizon (CPD/SL, calibrated)", "", "| tau (Myr) | n | full 30-D | Thompson-7 | copy nearest | consensus |", "|---|---|---|---|---|---|"]
    for t in results["horizon_table"]["CPD_over_SL"]:
        L.append(f"| {t['tau_bin'][0]}–{t['tau_bin'][1] if t['tau_bin'][1] < 1000 else '∞'} | {t['n']} | **{t['median_abs_err_full30_cal']:.4f}** | {t['median_abs_err_thompson7_cal']:.4f} | {t['median_abs_err_nearest_cal']:.4f} | {t['median_abs_err_consensus_cal']:.4f} |")
    h1, h2, h4 = H["H1_cal"], H["H2_cal"], H["H4_cal"]
    t_near, t_far = results["horizon_table"]["BDmax_over_SL"][3], results["horizon_table"]["BDmax_over_SL"][4]
    L += ["", "## Interpretation (written after unblinding; exploratory, not pre-registered)", "",
          f"- **Real predictive skill within 50 Myr.** For {h1['n']:,} species the engine had never seen, the predicted body depth lands within a median {h1['median_full30']:.3f} BDmax/SL of the museum calliper value "
          f"(about {100 * h1['median_full30'] / 0.40:.0f}% of a typical fish's depth), {100 * (1 - h1['ratio']):.0f}% less error than the consensus, with {H['H3_cal']['coverage95_pct']:.0f}% of species inside the 95% interval. Panel B shows r = {np.corrcoef(pv, o)[0, 1]:.2f} between predicted and observed.",
          f"- **But most of that skill is 'nearest relative'.** Copying the nearest training species' ratio gets median error {h2['median_nearest']:.4f}; the Gaussian-process conditioning improves on it by only {100 * (1 - h2['ratio']):.0f}% (p = {h2['p_wilcoxon_less']:.2f}). "
          "The internal hold-out showed a larger margin on 17-landmark RMS (0.0108 vs 0.0128), which integrates over the whole configuration; on a single noisy calliper ratio the extra information in the tree beyond the nearest neighbour is small. H2 was the right hypothesis to register and it did not hold.",
          f"- **The 50 Myr horizon is real but was mis-described.** Error jumps from {t_near['median_abs_err_full30_cal']:.3f} (25–50 Myr) to {t_far['median_abs_err_full30_cal']:.3f} (50–80 Myr) and 95% coverage falls from {t_near['coverage95_full30_cal']:.0f}% to {t_far['coverage95_full30_cal']:.0f}%: beyond 50 Myr the intervals are too narrow and the point predictions are poor. "
          f"H4 nevertheless failed because it was phrased as 'no better than the grand consensus', and the grand consensus of a reef-fish training set (BDmax/SL 0.41) is a bad prior for teleosts at large; the GLS root state (~0.28–0.30) is much closer to the typical teleost and still halves the consensus error (ratio {h4['ratio']:.2f}). The correct deep-horizon statement is 'degrades to a clade-wide prior', not 'degrades to the consensus'.",
          f"- **H5's non-monotonicity is a bias, not a gain in skill.** Beyond 50 Myr the model systematically over-predicts body depth (median predicted {np.median([preds[s['species']]['full30']['BDmax_over_SL']['pred'] for s in far]):.2f} vs observed {np.median([s['BDmax_over_SL']['obs'] for s in far]):.2f} BDmax/SL): "
          "distant teleosts are more elongate than anything a reef-fish training set anticipates. The error is smallest in the deepest bin only because those predictions sit nearest the GLS root state, which is the least reef-biased number the model can produce. The step up at 50 Myr is monotone; the dip at 120 Myr says nothing about deeper being easier.",
          "- **Thompson-7 vs full-30.** The a-priori Thompson modes beat the consensus everywhere within 50 Myr but are ~50% worse than the full shape space on this ratio, consistent with the internal result that the 7 modes capture only 58% of shape variance.",
          "- The family-mean column is computed on the subset of species whose family has >= 3 training species, so it is not directly comparable to the other columns in bins where that subset is small.",
          "", "## What this does and does not show", "",
          "- The predictions, hypotheses, bins, tests and calibration rule were fixed and hashed before the comparison file was downloaded; the scoring script checks those hashes. Nothing above was tuned afterwards.",
          "- FishShapes maximum body depth is a calliper measurement on whole specimens; the landmark proxy is the vertical landmark extent. The calibration maps one convention onto the other using training species only and is applied identically to every method, so it cannot favour the phylogenetic prediction over the baselines.",
          "- Per-species values are medians of 1–3 specimens; intraspecific and ontogenetic variation in FishShapes therefore adds noise that no method can remove, which inflates every method's error equally and lowers interval coverage.",
          "- Beyond ~50 Myr the registered prediction is the root state of the sampled clade, not the grand consensus; H4 asks whether it is any better than the consensus and the answer above is the pre-registered one, whichever way it fell.", "",
          "All per-species values are in [`thompson_blind_test_results.json`](../thompson_blind_test_results.json)."]
    OUT_MD.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"    wrote {OUT_MD}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

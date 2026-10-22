#!/usr/bin/env python3
"""
Score the locked Thompson-engine predictions (thompson_blind_predictions_registered.json,
commit dac08493) against a comparison dataset from the registered portfolio
(docs/THOMPSON_DATASET_PORTFOLIO.md), using exactly the FishShapes protocol.

Designed to run unattended (GitHub Actions: .github/workflows/thompson-portfolio.yml).

    python3 thompson/portfolio/score_locked_predictions.py --source fishbase_morphmet
    python3 thompson/portfolio/score_locked_predictions.py --source fishshapes_v1     # re-scores the anchor

Each source is a loader returning {Genus_species: {trait: per-species median}} plus metadata.
Adding a dataset = adding a loader; the tests, bins, calibration and outputs are shared and fixed.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import gzip
import hashlib
import importlib.util
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parents[2]
REG_JSON = REPO_ROOT / "thompson_blind_predictions_registered.json"
MANIFEST = REPO_ROOT / "thompson_blind_predictions_manifest.json"
PORTFOLIO = REPO_ROOT / "thompson_dataset_portfolio.json"
CACHE = REPO_ROOT / ".thompson_cache"
OUT_DIR = REPO_ROOT / "thompson" / "portfolio" / "results"
DOC_DIR = REPO_ROOT / "docs" / "portfolio"
ALPHA = 0.01
HORIZON = 50.0
METHODS = ["full30", "thompson7", "consensus", "nearest", "fammean"]
ALL_TRAITS = ["BDmax_over_SL", "CPD_over_SL", "HL_over_SL", "PD_over_SL"]


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


def norm_name(s: str) -> str:
    parts = [p for p in s.strip().replace(" ", "_").split("_") if p]
    return f"{parts[0].capitalize()}_{parts[1].lower()}" if len(parts) >= 2 else ""


def wilcoxon_less(a, b):
    d = a - b
    d = d[d != 0]
    return (float(stats.wilcoxon(d, alternative="less").pvalue) if len(d) >= 10 else float("nan")), int(len(d))


# ---------------------------------------------------------------- sources
def load_fishshapes_v1():
    path = CACHE / "fishshapes_v1" / "FishShapes_specimensv1.csv"
    cols = {"BDmax_over_SL": "Max_body_depth", "CPD_over_SL": "Min_caudalpeduncle_depth"}
    rows = defaultdict(lambda: defaultdict(list)); n = 0
    with path.open(newline="", encoding="utf-8-sig") as fh:
        for r in csv.DictReader(fh):
            n += 1; nm = norm_name(r["Tree_name"])
            try:
                sl = float(r["Standard_length"])
            except ValueError:
                continue
            if not nm or sl <= 0:
                continue
            for k, c in cols.items():
                try:
                    v = float(r[c])
                except ValueError:
                    continue
                if v > 0:
                    rows[nm][k].append(v / sl)
    obs = {nm: {k: float(np.median(v)) for k, v in d.items() if v} for nm, d in rows.items()}
    return obs, {"name": "FishShapes v1 (Price et al. 2022)", "file": str(path.relative_to(REPO_ROOT)), "sha256": sha256(path), "n_records": n, "unit": "specimen",
                 "citation": "Price SA et al. 2022 Ecology 103:e3829; Zenodo 6642650 (CC0)", "traits": list(cols)}


def load_fishbase_morphmet():
    fb = CACHE / "fishbase_21_06"
    code = {}
    with gzip.open(fb / "species.tsv.gz", "rt", encoding="utf-8", errors="replace") as fh:
        for r in csv.DictReader(fh, delimiter="\t"):
            code[r["SpecCode"]] = f'{r["Genus"]}_{r["Species"]}'
    cols = {"BDmax_over_SL": "BD", "CPD_over_SL": "CH", "HL_over_SL": "HL", "PD_over_SL": "PDL"}
    rows = defaultdict(lambda: defaultdict(list)); n = 0
    with gzip.open(fb / "morphmet.tsv.gz", "rt", encoding="utf-8", errors="replace") as fh:
        for r in csv.DictReader(fh, delimiter="\t"):
            n += 1; nm = code.get(r["Speccode"])
            try:
                sl = float(r["SL"])
            except (ValueError, TypeError):
                continue
            if not nm or sl <= 0:
                continue
            for k, c in cols.items():
                try:
                    v = float(r[c])
                except (ValueError, TypeError):
                    continue
                if v > 0:
                    rows[nm][k].append(v / sl)
    obs = {nm: {k: float(np.median(v)) for k, v in d.items() if v} for nm, d in rows.items()}
    return obs, {"name": "FishBase MORPHMET picture-based morphometrics (fb-21.06)", "file": "fishbase_21_06/morphmet.tsv.gz", "sha256": sha256(fb / "morphmet.tsv.gz"), "n_records": n, "unit": "picture",
                 "citation": "Froese R & Pauly D (eds) 2021 FishBase 06/2021 (rfishbase release fb-21.06; CC BY-NC)", "traits": list(cols)}


SOURCES = {"fishshapes_v1": load_fishshapes_v1, "fishbase_morphmet": load_fishbase_morphmet}


# ---------------------------------------------------------------- scoring
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, choices=sorted(SOURCES))
    args = ap.parse_args()
    src = args.source

    print("[0] verify locked files")
    man = json.loads(MANIFEST.read_text())
    for fname, digest in man["files"].items():
        if sha256(REPO_ROOT / fname) != digest:
            print(f"    HASH MISMATCH for {fname}"); return 2
        print(f"    {fname}: sha256 OK")
    reg = json.loads(REG_JSON.read_text())
    preds = {p["species"]: p for p in reg["predictions"]}
    bins = [tuple(b) for b in reg["tau_bins_myr"]]
    portfolio = json.loads(PORTFOLIO.read_text()) if PORTFOLIO.exists() else {}
    registered = next((d for d in portfolio.get("fish_datasets", []) if d["id"] == src), None)

    print(f"[1] load comparison data: {src}")
    obs, meta = SOURCES[src]()
    traits = [t for t in ALL_TRAITS if t in meta["traits"]]
    print(f"    {meta['n_records']} {meta['unit']}s, {len(obs)} species, traits {traits}")

    print("[2] calibration on training species only")
    pre = load_module("pre", "thompson/preregister_blind_predictions.py")
    at = pre.at
    names, families, X = at.load_landmarks()
    Y = at.gpa(X); M0 = at.centre_scale(Y.mean(0)); Y_opa = np.array([at.opa(y, M0) for y in Y])
    train_lm = pre.ratios_vec(Y_opa); train_set = set(names)
    calib = {}
    for k in traits:
        xs, ys = zip(*[(float(train_lm[k][i]), obs[nm][k]) for i, nm in enumerate(names) if nm in obs and k in obs[nm]]) if any(nm in obs and k in obs[nm] for nm in names) else ([], [])
        xs, ys = np.array(xs), np.array(ys)
        if len(xs) < 10:
            calib[k] = {"alpha": 0.0, "beta": 1.0, "r": float("nan"), "n_training_species": int(len(xs)), "note": "fewer than 10 overlapping training species: raw scale used"}
        else:
            slope, intercept, r, _, _ = stats.linregress(xs, ys)
            calib[k] = {"alpha": float(intercept), "beta": float(slope), "r": float(r), "n_training_species": int(len(xs)), "residual_sd": float(np.std(ys - (intercept + slope * xs), ddof=2))}
        print(f"    {k}: {calib[k]}")

    def cal(k, x, use):
        return calib[k]["alpha"] + calib[k]["beta"] * x if use else x

    print("[3] score")
    exact_1917 = set(json.loads(at.EXACT_SPECIES_JSON.read_text(encoding="utf-8"))["specimens"])
    matched = [nm for nm in preds if nm in obs and nm not in train_set and nm not in exact_1917]
    rows = []
    for nm in matched:
        p = preds[nm]
        rec = {"species": nm, "tau": p["tau_myr"], "family": p["family_from_taxonomy"], "nearest": p["nearest_training_species"]}
        for k in traits:
            if k not in obs[nm]:
                continue
            o = obs[nm][k]
            vals = {"full30": p["full30"][k]["pred"], "thompson7": p["thompson7"][k]["pred"], "consensus": p["baseline_consensus"][k],
                    "nearest": p["baseline_nearest_training_species"][k], "fammean": (p["baseline_training_family_mean"][k] if p["baseline_training_family_mean"] else None)}
            rec[k] = {"obs": o}
            for use, tag in [(False, "raw"), (True, "cal")]:
                for m, v in vals.items():
                    rec[k][f"err_{m}_{tag}"] = abs(cal(k, v, use) - o) if v is not None else None
                for m in ("full30", "thompson7"):
                    lo, hi = cal(k, p[m][k]["lo95"], use), cal(k, p[m][k]["hi95"], use)
                    rec[k][f"in95_{m}_{tag}"] = bool(min(lo, hi) <= o <= max(lo, hi))
        rows.append(rec)
    print(f"    {len(matched)} target species scored")

    def arr(k, field, sel):
        return np.array([r[k][field] for r in sel if k in r and r[k].get(field) is not None], dtype=float)

    horizon, hyp = {}, {}
    for k in traits:
        tab = []
        for lo, hi in bins:
            sel = [r for r in rows if lo <= r["tau"] < hi and k in r]
            if not sel:
                continue
            row = {"tau_bin": [lo, hi], "n": len(sel)}
            for tag in ("raw", "cal"):
                for m in METHODS:
                    e = arr(k, f"err_{m}_{tag}", sel); row[f"median_abs_err_{m}_{tag}"] = float(np.median(e)) if len(e) else None
                for m in ("full30", "thompson7"):
                    row[f"coverage95_{m}_{tag}"] = float(100 * np.mean([r[k][f"in95_{m}_{tag}"] for r in sel]))
            tab.append(row)
        horizon[k] = tab
        near = [r for r in rows if r["tau"] < HORIZON and k in r]; far = [r for r in rows if r["tau"] >= HORIZON and k in r]
        for tag in ("cal", "raw"):
            h = {}
            if len(near) >= 10:
                ef, ec, en, e7 = (arr(k, f"err_{m}_{tag}", near) for m in ("full30", "consensus", "nearest", "thompson7"))
                p1, n1 = wilcoxon_less(ef, ec); p2, n2 = wilcoxon_less(ef, en); p6, _ = wilcoxon_less(e7, ec)
                cov = float(100 * np.mean([r[k][f"in95_full30_{tag}"] for r in near]))
                h["near"] = {"n": len(near), "median_full30": float(np.median(ef)), "median_consensus": float(np.median(ec)), "median_nearest": float(np.median(en)), "median_thompson7": float(np.median(e7)),
                             "ratio_consensus": float(np.median(ef) / np.median(ec)), "ratio_nearest": float(np.median(ef) / np.median(en)), "p_vs_consensus": p1, "p_vs_nearest": p2, "coverage95": cov,
                             "P1_beats_consensus": bool(p1 < ALPHA and np.median(ef) / np.median(ec) < 0.8), "H2_beats_nearest": bool(p2 < ALPHA and np.median(ef) < np.median(en)),
                             "H3_coverage_calibrated": bool(85 <= cov <= 97), "H6_thompson7_between": bool(np.median(e7) > np.median(ef) and p6 < ALPHA and np.median(e7) < np.median(ec)),
                             "r_pred_obs": float(np.corrcoef([cal(k, preds[r["species"]]["full30"][k]["pred"], tag == "cal") for r in near], [r[k]["obs"] for r in near])[0, 1])}
            if len(far) >= 10:
                eff, ecf = arr(k, f"err_full30_{tag}", far), arr(k, f"err_consensus_{tag}", far)
                covf = float(100 * np.mean([r[k][f"in95_full30_{tag}"] for r in far]))
                h["far"] = {"n": len(far), "median_full30": float(np.median(eff)), "median_consensus": float(np.median(ecf)), "ratio_consensus": float(np.median(eff) / np.median(ecf)), "coverage95": covf,
                            "median_pred": float(np.median([cal(k, preds[r["species"]]["full30"][k]["pred"], tag == "cal") for r in far])), "median_obs": float(np.median([r[k]["obs"] for r in far]))}
            hyp[f"{k}_{tag}"] = h

    # portfolio pass rules (BDmax, calibrated)
    verdict = {}
    hb = hyp.get("BDmax_over_SL_cal", {})
    if registered and hb.get("near") and "p_pass_registered" in registered:
        sim = registered.get("simulated", {})
        def inside(v, iv, tol=0.0):
            return bool(iv and iv["q05"] - tol <= v <= iv["q95"] + tol)
        verdict["P1_beats_consensus_within_50"] = {"registered_p": registered["p_pass_registered"]["BDmax_over_SL"]["P1_beats_consensus_within_50"], "realised": hb["near"]["P1_beats_consensus"], "ratio": hb["near"]["ratio_consensus"],
                                                   "simulated_interval": sim.get("ratio_consensus_near"), "inside_simulated_90pct_interval": inside(hb["near"]["ratio_consensus"], sim.get("ratio_consensus_near"))}
        verdict["P2_nearest_margin_under_10pct"] = {"registered_p_H2_passes": registered["p_pass_registered"]["BDmax_over_SL"]["P2_H2_passes"], "realised_H2": hb["near"]["H2_beats_nearest"], "ratio": hb["near"]["ratio_nearest"],
                                          "realised": bool(hb["near"]["ratio_nearest"] > 0.9), "H2_formally_passes": hb["near"]["H2_beats_nearest"], "simulated_interval": sim.get("ratio_nearest_near")}
        verdict["coverage_calibrated_within_50"] = {"registered_p": registered["p_pass_registered"]["BDmax_over_SL"]["coverage_calibrated_within_50"], "realised": hb["near"]["H3_coverage_calibrated"], "coverage": hb["near"]["coverage95"], "simulated_interval": sim.get("coverage_near")}
        if hb.get("far") and registered.get("P3_deep_prediction"):
            p3 = registered["P3_deep_prediction"]
            ri, ci = p3["ratio_consensus_interval"], p3["coverage_interval_pct"]
            verdict["P3_deep_horizon"] = {"registered_p": p3["p_pass_subjective"], "pass_interval_ratio": ri, "pass_interval_coverage": ci, "ratio": hb["far"]["ratio_consensus"], "coverage": hb["far"]["coverage95"],
                                          "realised": bool(ri[0] <= hb["far"]["ratio_consensus"] <= ri[1] and ci[0] <= hb["far"]["coverage95"] <= ci[1])}
    if "HL_over_SL" in traits and hyp.get("HL_over_SL_cal", {}).get("near") and hb.get("near"):
        rh, rp = hyp["HL_over_SL_cal"]["near"]["ratio_consensus"], hyp.get("PD_over_SL_cal", {}).get("near", {}).get("ratio_consensus")
        verdict["P4_head_predorsal_weaker"] = {"registered_p": 0.6, "ratio_BDmax": hb["near"]["ratio_consensus"], "ratio_HL": rh, "ratio_PD": rp,
                                               "realised": bool(rh >= hb["near"]["ratio_consensus"] + 0.1 and (rp is None or rp >= hb["near"]["ratio_consensus"] + 0.1))}

    results = {"source": src, "meta": meta, "scored_on": dt.date.today().isoformat(), "locked_manifest": man, "n_matched": len(matched), "n_species_in_source": len(obs), "calibration": calib,
               "horizon_table": horizon, "hypotheses": hyp, "portfolio_verdict": verdict, "per_species": rows}
    OUT_DIR.mkdir(parents=True, exist_ok=True); DOC_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / f"{src}.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
    for kk, v in verdict.items():
        print(f"    {kk}: registered P={v.get('registered_p', v.get('registered_p_H2_passes'))}  realised={v['realised']}")
    write_report(results, registered)
    make_figure(results, preds, cal)
    return 0


def write_report(R: dict, registered: dict | None) -> None:
    src = R["source"]; m = R["meta"]
    L = [f"# Portfolio sub-prediction: {m['name']}", "",
         f"Scored {R['scored_on']} by `thompson/portfolio/score_locked_predictions.py --source {src}` against the predictions locked at commit `{R['locked_manifest']['git_commit_at_generation'][:12]}` (hashes verified at run time). "
         f"Comparison data: {m['citation']}; `{m['file']}` sha256 `{m['sha256'][:16]}…`, {m['n_records']:,} {m['unit']}s, {R['n_species_in_source']:,} species; {R['n_matched']:,} registered target species matched (training and 1917 species excluded).", "",
         f"![figure](./fig_{src}.png)", "", "## Registered claims vs realised outcome (BDmax/SL, calibrated)", ""]
    if registered:
        L += [f"Registered in `docs/THOMPSON_DATASET_PORTFOLIO.md` as *{registered['dataset_class']}* with status *{registered['status']}*.", "", "| Claim | Registered probability | Simulated / pass interval | Realised | Verdict |", "|---|---|---|---|---|"]
        for kk, v in R["portfolio_verdict"].items():
            si = v.get("simulated_interval")
            iv = f"{si['median']:.2f} [{si['q05']:.2f}, {si['q95']:.2f}]" if si else (f"ratio {v['pass_interval_ratio']}, coverage {v['pass_interval_coverage']}" if "pass_interval_ratio" in v else "-")
            real = v.get("ratio", v.get("coverage")); real = f"{real:.3f}" if isinstance(real, float) else "-"
            ok = v["realised"]
            p = v.get("registered_p", v.get("registered_p_H2_passes"))
            note = f" (H2 formally {'passes' if v['H2_formally_passes'] else 'fails'}; registered P is for H2 passing)" if "H2_formally_passes" in v else ""
            L.append(f"| {kk} | {100 * p:.0f}% | {iv} | {real}{note} | {'✓' if ok else '✗'} |")
    else:
        L.append("(no portfolio entry for this source)")
    for k, tab in R["horizon_table"].items():
        L += ["", f"## Error by horizon: {k} (calibrated median |error|)", "", "| tau (Myr) | n | full 30-D | Thompson-7 | copy nearest | consensus | family mean | 95% cov. full30 (cal / raw) |", "|---|---|---|---|---|---|---|---|"]
        for t in tab:
            fm = t["median_abs_err_fammean_cal"]
            L.append(f"| {t['tau_bin'][0]}–{t['tau_bin'][1] if t['tau_bin'][1] < 1000 else '∞'} | {t['n']} | **{t['median_abs_err_full30_cal']:.4f}** | {t['median_abs_err_thompson7_cal']:.4f} | {t['median_abs_err_nearest_cal']:.4f} | {t['median_abs_err_consensus_cal']:.4f} | {f'{fm:.4f}' if fm is not None else '–'} | {t['coverage95_full30_cal']:.0f}% / {t['coverage95_full30_raw']:.0f}% |")
        h = R["hypotheses"].get(f"{k}_cal", {})
        c_ = R["calibration"][k]
        if not np.isnan(c_["r"]) and c_["r"] < 0.3:
            L += ["", f"**Trait-mapping failure, not a model test:** the training-species calibration between the landmark ratio and the source column has r = {c_['r']:.2f}, so the source column does not measure the registered trait "
                  f"(for FishBase `CH` the observed median is {np.median([r[k]['obs'] for r in R['per_species'] if k in r]):.2f} SL, a caudal-fin height, not a peduncle depth). Results for {k} on this source are reported for completeness only and do not count toward any portfolio claim."]
        if h.get("near"):
            n_ = h["near"]
            L += ["", f"Within 50 Myr (n={n_['n']}): ratio vs consensus {n_['ratio_consensus']:.2f} (p={n_['p_vs_consensus']:.2g}), vs nearest {n_['ratio_nearest']:.2f} (p={n_['p_vs_nearest']:.2g}), coverage {n_['coverage95']:.0f}%, r(pred, obs)={n_['r_pred_obs']:.2f}; "
                  f"T7 between consensus and full-30: {'yes' if n_['H6_thompson7_between'] else 'no'}."]
        if h.get("far"):
            f_ = h["far"]
            L.append(f"Beyond 50 Myr (n={f_['n']}): ratio vs consensus {f_['ratio_consensus']:.2f}, coverage {f_['coverage95']:.0f}%, median predicted {f_['median_pred']:.3f} vs observed {f_['median_obs']:.3f}.")
    L += ["", "## Calibration (training species only)", ""] + [f"- {k}: {json.dumps(c)}" for k, c in R["calibration"].items()]
    L += ["", "## Notes", "", "- Tests, bins, calibration rule and pass thresholds are those registered for the FishShapes blind test and the portfolio; nothing was tuned on this dataset.",
          "- FishBase MORPHMET values are measured on single pictures (often drawings), so measurement noise is larger than for FishShapes callipers and inflates every method's error equally.",
          f"- Per-species values are in `thompson/portfolio/results/{src}.json`.", ""]
    (DOC_DIR / f"{src}.md").write_text("\n".join(L), encoding="utf-8")
    print(f"    wrote {DOC_DIR / f'{src}.md'}")


def make_figure(R: dict, preds: dict, cal) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    traits = list(R["horizon_table"])
    fig, axes = plt.subplots(1, len(traits) + 1, figsize=(5 * (len(traits) + 1), 4.6))
    for ax, k in zip(axes, traits):
        tab = R["horizon_table"][k]; xs = np.arange(len(tab))
        for m, c, lab in [("full30", "#2ca02c", "full 30-D"), ("thompson7", "#1f77b4", "Thompson-7"), ("nearest", "#ff7f0e", "copy nearest"), ("consensus", "#7f7f7f", "consensus"), ("fammean", "#9467bd", "family mean")]:
            ys = [t[f"median_abs_err_{m}_cal"] for t in tab]
            ax.plot(xs, [y if y is not None else np.nan for y in ys], "o-", color=c, label=lab, ms=4)
        ax.set_xticks(xs); ax.set_xticklabels([f"{t['tau_bin'][0]}–{t['tau_bin'][1] if t['tau_bin'][1] < 1000 else ''}\nn={t['n']}" for t in tab], fontsize=7)
        ax.axvline(3.5, color="k", lw=0.8, alpha=0.4); ax.set_title(k.replace("_over_", "/"), fontsize=10); ax.grid(alpha=0.3); ax.set_xlabel("tau (Myr)")
        ax.set_ylabel("median |error| (calibrated)")
    axes[0].legend(fontsize=7)
    ax = axes[-1]; k = "BDmax_over_SL"
    sel = [r for r in R["per_species"] if k in r and r["tau"] < HORIZON]
    o = np.array([r[k]["obs"] for r in sel]); pv = np.array([cal(k, preds[r["species"]]["full30"][k]["pred"], True) for r in sel]); tau_ = np.array([r["tau"] for r in sel])
    sc = ax.scatter(pv, o, c=tau_, cmap="viridis", s=8, alpha=0.6); lim = [0.05, max(o.max(), pv.max()) * 1.05]
    ax.plot(lim, lim, "k--", lw=1); ax.set_xlim(lim); ax.set_ylim(lim); plt.colorbar(sc, ax=ax, label="tau (Myr)")
    ax.set_xlabel("predicted BDmax/SL (full 30-D, calibrated)"); ax.set_ylabel("observed BDmax/SL"); ax.set_title(f"tau < 50 Myr: n={len(sel)}, r={np.corrcoef(pv, o)[0, 1]:.2f}", fontsize=10)
    fig.suptitle(f"Locked Thompson-engine predictions vs {R['meta']['name']}: {R['n_matched']:,} never-seen species", fontsize=11, fontweight="bold")
    fig.tight_layout(); fig.savefig(DOC_DIR / f"fig_{R['source']}.png", dpi=140); print(f"    wrote {DOC_DIR / f'fig_{R['source']}.png'}")


if __name__ == "__main__":
    sys.exit(main())

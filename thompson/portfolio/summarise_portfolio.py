#!/usr/bin/env python3
"""
Collect every thompson/portfolio/results/*.json into docs/portfolio/PORTFOLIO_STATUS.md:
registered probability vs realised outcome for each portfolio item, plus a running Brier
score for the registered probabilities.  Run after the scorers (CI does this).
"""
from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
RES = REPO_ROOT / "thompson" / "portfolio" / "results"
PORTFOLIO = REPO_ROOT / "thompson_dataset_portfolio.json"
OUT = REPO_ROOT / "docs" / "portfolio" / "PORTFOLIO_STATUS.md"


def main() -> int:
    P = json.loads(PORTFOLIO.read_text())
    results = {p.stem: json.loads(p.read_text()) for p in sorted(RES.glob("*.json"))}
    L = ["# Thompson prediction engine: portfolio status", "",
         f"Updated {dt.date.today().isoformat()} by `thompson/portfolio/summarise_portfolio.py` (run unattended by `.github/workflows/thompson-portfolio.yml`). "
         "Registered probabilities come from `docs/THOMPSON_DATASET_PORTFOLIO.md` and are never edited; this page only fills in what was realised.", "",
         "## Fish datasets (locked predictions, commit dac08493)", "",
         "| Dataset | Status | n matched | BDmax ratio vs consensus (<50 Myr) | vs nearest | 95% coverage | Beyond 50 Myr: ratio / coverage | Claims: registered P -> realised |", "|---|---|---|---|---|---|---|---|"]
    brier, n_claims = [], 0
    for d in P["fish_datasets"]:
        r = results.get(d["id"])
        if not r:
            L.append(f"| {d['name']} | {d['status'].split(';')[0]} | - | - | - | - | - | pending |"); continue
        hb = r["hypotheses"].get("BDmax_over_SL_cal", {}); near, far = hb.get("near", {}), hb.get("far", {})
        claims = []
        for kk, v in r["portfolio_verdict"].items():
            p = v.get("registered_p", v.get("registered_p_H2_passes")); ok = v["realised"]
            if "H2_formally_passes" in v:          # registered P was for H2 passing; score the registered event
                brier.append((p - float(v["H2_formally_passes"])) ** 2)
            else:
                brier.append((p - float(ok)) ** 2)
            n_claims += 1
            claims.append(f"{kk.split('_')[0]} {100 * p:.0f}%→{'✓' if ok else '✗'}")
        L.append(f"| [{d['name']}](./{d['id']}.md) | scored {r['scored_on']} | {r['n_matched']:,} | {near.get('ratio_consensus', float('nan')):.2f} | {near.get('ratio_nearest', float('nan')):.2f} | {near.get('coverage95', float('nan')):.0f}% | "
                 f"{far.get('ratio_consensus', float('nan')):.2f} / {far.get('coverage95', float('nan')):.0f}% | {'; '.join(claims) if claims else 'anchor'} |")
    L += ["", "## Fish datasets without simulated anchor (subjective claims)", "", "| Dataset | Registered P | Status |", "|---|---|---|"]
    for d in P["fish_subjective"]:
        r = results.get(d["id"])
        L.append(f"| {d['name']} | {100 * d['p_pass']:.0f}% | {'scored ' + r['scored_on'] if r else 'pending'} |")
    L += ["", "## Other groups (re-trained engine; subjective priors)", "", "| Group | P(beats consensus) | P(coverage) | P(beats nearest) | Status |", "|---|---|---|---|---|"]
    for g in P["other_groups"]:
        r = results.get(g["id"])
        L.append(f"| {g['group']} | {100 * g['p_beats_consensus']:.0f}% | {100 * g['p_coverage_calibrated']:.0f}% | {100 * g['p_beats_nearest']:.0f}% | {'scored ' + r['scored_on'] if r else 'pending'} |")
    if brier:
        L += ["", f"## Calibration of the registered probabilities", "", f"Brier score over {n_claims} resolved claims: **{sum(brier) / len(brier):.3f}** (0 = perfect, 0.25 = uninformative coin). Each resolved claim scores (P − outcome)²."]
    L.append("")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(L), encoding="utf-8")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

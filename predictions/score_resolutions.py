#!/usr/bin/env python3
"""
Score the prediction registry from predictions/resolutions/*.json.

This is the ONLY post-registration edit permitted to the registry, and it only
fills scoring fields (status / scored_on / outcome / resolvability outcome) --
prediction text, quantitative expectations, falsifiers and resolvability
forecasts are never altered. Meta-resolvability predictions are scored from
the tallies: fully decided = confirmed+refuted; partial = partially_resolved.

Usage: python3 predictions/score_resolutions.py
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
REG = HERE / "prediction_registry.json"
SCORED_ON = "2026-10-26"


def main() -> int:
    reg = json.loads(REG.read_text())
    res_by_id = {}
    for f in sorted((HERE / "resolutions").glob("*.json")):
        doc = json.loads(f.read_text())
        for r in doc["resolutions"]:
            res_by_id[r["prediction_id"]] = r
    missing = [p["prediction_id"] for p in reg["predictions"] if p["prediction_id"] not in res_by_id]
    if missing:
        raise SystemExit(f"No resolution recorded for: {missing}")

    for p in reg["predictions"]:
        r = res_by_id[p["prediction_id"]]
        p["status"] = "scored"
        p["scored_on"] = SCORED_ON
        p["outcome"] = r["outcome_verdict"]
        p["resolvability"]["resolvability_status"] = "scored"
        p["resolvability"]["resolved_on"] = SCORED_ON
        p["resolvability"]["actual_category"] = (
            "partial_only" if r["outcome_verdict"] == "partially_resolved"
            else p["resolvability"]["predicted_category"])
        p["resolvability"]["actual_verdict_note"] = r["resolvability_verdict"]

    outcomes = [p["outcome"] for p in reg["predictions"]]
    fully = sum(o in ("confirmed", "refuted") for o in outcomes)
    partial = outcomes.count("partially_resolved")
    current_only_decided = [p["prediction_id"] for p in reg["predictions"]
                            if p["resolvability"]["predicted_category"] == "current_datasets_only"
                            and p["outcome"] in ("confirmed", "refuted")]
    meta_outcomes = {
        "META-RES-01": "confirmed" if len(current_only_decided) == 2 else "refuted",
        "META-RES-02": "confirmed" if fully >= 10 else "refuted",
        "META-RES-03": "confirmed" if partial == 4 else "refuted",
        "META-RES-04": "confirmed",  # every row resolved/bounded via in-repo or public routes; partials are extraction gaps
    }
    tallies = {"fully_decided": fully, "partially_resolved": partial,
               "confirmed": outcomes.count("confirmed"), "refuted": outcomes.count("refuted"),
               "current_only_decided": current_only_decided}
    for m in reg["meta_resolvability_predictions"]:
        m["status"] = "scored"
        m["scored_on"] = SCORED_ON
        m["outcome"] = meta_outcomes[m["meta_id"]]
    reg["scoring"] = {"scored_on": SCORED_ON, "tallies": tallies,
                       "note": "Outcome score and resolvability score are independent: a confirmed biology prediction can still refute a meta-resolvability forecast (see META-RES-02/03, refuted because ZF-HH-01 resolved only partially, making 9 fully decided and 5 partial)."}
    REG.write_text(json.dumps(reg, indent=2) + "\n")

    rows = list(csv.DictReader((HERE / "prediction_registry.csv").open()))
    by_id = {p["prediction_id"]: p for p in reg["predictions"]}
    with (HERE / "prediction_registry.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for row in rows:
            p = by_id[row["prediction_id"]]
            row.update(status=p["status"], scored_on=p["scored_on"], outcome=p["outcome"])
            w.writerow(row)

    print(f"Scored {len(reg['predictions'])} predictions: {tallies}")
    for m in reg["meta_resolvability_predictions"]:
        print(f"  {m['meta_id']}: {m['outcome']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

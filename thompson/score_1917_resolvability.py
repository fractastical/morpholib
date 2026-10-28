#!/usr/bin/env python3
"""
Log P(resolvable) for every Thompson (1917) claim -- and feed the graph.

For each of the 59 claims (all Thompson, 1917, 1st ed.) this computes the
probability that the claim can be RESOLVED (tested to a verdict) from an
available dataset by our loop, using a deterministic rubric over the
dataset-audit fields. It prices resolvability -- data in hand, automation
tier, effort, human blockers -- NOT the probability that Thompson was
right. (Same distinction as morphoeval/problem_scout_100.py's RUBRIC.)

Outputs / updates:
  thompson/thompson_1917_resolvability.json   the ledger (source of truth)
  thompson/thompson_1917_dataset_audit.json   rows gain p_resolvable_* fields
  thompson/thompson_1917_loop_queue.yaml      slots gain p_resolvable_percent

Consumers:
  docs/build_problem_graph_html.py  chapter edges carry prob = mean
      P(resolvable) of the chapter's claims (probStatus provisional_scored,
      probNote naming this rubric -- never presented as a Delphi posterior)
  thompson/run_overnight_1917.py    dispatches highest-P claims first

Usage: venv/bin/python thompson/score_1917_resolvability.py [--check]
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "thompson" / "thompson_1917_claims.json"
AUDIT = ROOT / "thompson" / "thompson_1917_dataset_audit.json"
QUEUE = ROOT / "thompson" / "thompson_1917_loop_queue.yaml"
LEDGER = ROOT / "thompson" / "thompson_1917_resolvability.json"

BASE = {  # P(resolvable) before adjustments, by dataset-existence class
    "IN_REPO": 85,            # dataset already in repo, SHA-pinned or fetchable
    "PUBLIC_SCRIPTABLE": 75,  # public dataset with DOI/API the loop can fetch
    "PUBLIC_CANDIDATE": 45,   # candidate exists; match to claim unverified
    "NO_DATA_NEEDED": 90,     # analytic claim; resolution is a derivation
    "NO_PUBLIC_FOUND": 15,    # probed; nothing suitable (digitising 1917 plates may still work)
    "UNPROBED": 30,           # probe failed -- wide uncertainty, NOT evidence of absence
    "NOT_TESTABLE": 5,        # cannot be resolved by a dataset as stated
}
METHOD = ("Deterministic rubric v1 over dataset-audit fields (existence base, "
          "Pass-2 status, automation tier, effort). Prices resolvability, not truth.")


def band(p: float) -> str:
    return "high" if p >= 70 else "medium" if p >= 40 else "low" if p >= 15 else "very_low"


def score_claim(existence: str, automation: str, pass2_status: str,
                effort_days) -> dict:
    """Pure function -- the whole rubric lives here so tests can pin it."""
    if existence not in BASE:
        raise ValueError(f"unknown existence class {existence!r}")
    adj: list[tuple[str, float]] = []
    base = float(BASE[existence])
    if pass2_status in ("DONE_REPO", "DONE_ELSEWHERE"):
        return {"p_percent": 100.0, "base": base,
                "adjustments": [["already resolved (" + pass2_status + ")", 100.0 - base]],
                "band": "resolved"}
    if pass2_status == "PARTIAL_REPO":
        adj.append(["partial repo progress already made", 10])
    if automation == "LOOP_WITH_SEED":
        adj.append(["needs a human-chosen seed before the loop", -5])
    hi = effort_days[1] if effort_days else 0
    if hi >= 15:
        adj.append([f"large effort ({effort_days[0]}-{hi} person-days)", -10])
    elif hi >= 8:
        adj.append([f"moderate effort ({effort_days[0]}-{hi} person-days)", -5])
    p = base + sum(a for _, a in adj)
    if automation == "HUMAN_REQUIRED":
        adj.append(["HUMAN_REQUIRED: capped at 25 until a human unblocks", 0])
        p = min(p, 25.0)
    p = max(2.0, min(98.0, p))
    return {"p_percent": round(p, 1), "base": base, "adjustments": adj, "band": band(p)}


def build_ledger(scored_on: str) -> tuple[dict, dict]:
    reg = {c["id"]: c for c in json.loads(REGISTRY.read_text())["claims"]}
    audit = json.loads(AUDIT.read_text())
    rows = []
    for r in audit["rows"]:
        # Status from the registry (source of truth, includes verdict
        # merge-backs); the audit row's pass2_status is the scout-time snapshot.
        status = reg[r["id"]]["pass2"]["status"] if r["id"] in reg else r["pass2_status"]
        s = score_claim(r["existence"], r["automation"], status, r.get("effort_days"))
        r["p_resolvable_percent"] = s["p_percent"]
        r["p_resolvable_band"] = s["band"]
        r["p_resolvable_components"] = {"base": s["base"], "adjustments": s["adjustments"]}
        r["p_resolvable_method"] = METHOD
        rows.append({"claim_id": r["id"], "date_made": "1917",
                     "citation": r["citation"], "p_percent": s["p_percent"], "band": s["band"],
                     "base": s["base"], "adjustments": s["adjustments"],
                     "existence": r["existence"], "automation": r["automation"],
                     "pass2_status": status})
    ps = [x["p_percent"] for x in rows]
    ledger = {
        "scored_on": scored_on,
        "applies_to": "Thompson (1917) On Growth and Form claims T1917-001..T1917-059 (all date_made=1917)",
        "definition": "P(this claim can be resolved to a verdict from an available dataset by the MorphoEval loop). NOT P(Thompson was right).",
        "method": METHOD,
        "rubric_base_by_existence": BASE,
        "summary": {
            "n": len(rows),
            "mean_percent": round(sum(ps) / len(ps), 1),
            "resolved_100": sum(1 for p in ps if p == 100.0),
            "high_ge_70": sum(1 for p in ps if 70 <= p < 100),
            "medium_40_69": sum(1 for p in ps if 40 <= p < 70),
            "low_15_39": sum(1 for p in ps if 15 <= p < 40),
            "very_low_lt_15": sum(1 for p in ps if p < 15),
        },
        "claims": rows,
    }
    return ledger, audit


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="validate only; write nothing")
    args = ap.parse_args()
    scored_on = dt.date.today().isoformat()
    ledger, audit = build_ledger(scored_on)
    ids = [x["claim_id"] for x in ledger["claims"]]
    assert ids == [f"T1917-{i:03d}" for i in range(1, 60)], "ledger must cover all 59 claims in order"
    print(f"P(resolvable) for 59 Thompson (1917) claims: {ledger['summary']}")
    top = sorted(ledger["claims"], key=lambda x: -x["p_percent"])[:8]
    print("highest:", ", ".join(f"{x['claim_id']}={x['p_percent']:g}%" for x in top))
    if args.check:
        print("PASS (check only, nothing written)")
        return 0
    LEDGER.write_text(json.dumps(ledger, indent=2) + "\n")
    AUDIT.write_text(json.dumps(audit, indent=2) + "\n")
    try:
        import yaml
        doc = yaml.safe_load(QUEUE.read_text())
        by_id = {x["claim_id"]: x for x in ledger["claims"]}
        for slot in doc["slots"]:
            slot["p_resolvable_percent"] = by_id[slot["claim_id"]]["p_percent"]
        QUEUE.write_text(yaml.safe_dump(doc, sort_keys=False, allow_unicode=True))
        print(f"wrote {LEDGER.relative_to(ROOT)}, enriched audit rows + {len(doc['slots'])} queue slots")
    except ImportError:
        print(f"wrote {LEDGER.relative_to(ROOT)} + audit (pyyaml missing: queue not enriched)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

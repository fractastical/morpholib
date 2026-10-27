#!/usr/bin/env python3
"""
morphoeval/audit_problem_framing.py — Is the promotion framing right for each problem?

`promote_findings.py` frames every promoted finding with one generic falsifier:
"Is <law> the right generative law for this structure — and what observation
kills it?" That is the right question for exactly one problem type (a
single-species simulation claim). The problem graph (docs/open-problems.json),
the registered prediction nanopubs (predictions/prediction_registry.json) and
the live arguments (predictions/live_arguments.json) contain problems of at
least SEVEN other types, each with its OWN native falsifier already written
down (`done`, `falsifier`, `settling_measurement`, `missing_evidence`).
Overriding those with the generic template would be a framing error — and for
pre-registered predictions it would violate blind discipline (the falsifier was
locked before scoring).

This script audits every node and emits, per node:
  problem_type          formal_proof | preregistered_blind_test | missing_dataset |
                        independent_rescore | failure_autopsy | structural_debt |
                        registered_prediction | live_argument
  native_falsifier      the node's own done/falsifier/settling criterion (verbatim)
  generic_framing       what promote_findings.py would currently emit
  framing_verdict       generic_fits | misframed_use_native | do_not_promote |
                        blind_locked_use_native_verbatim | adjudicate_camps
  correct_ask           the scientist-facing ask that matches the problem type
  audience              who can actually answer (not "a biologist" for a proof)
  promotion_track_hint  how promote_findings.py should treat it

Outputs: morphoeval/promotions/FRAMING_AUDIT.md and FRAMING_AUDIT.json.
`--check` exits 1 if any node would currently receive a misframed ask without
an override in FRAMING_RULES (used as a regression guard).

Usage:
  python3 morphoeval/audit_problem_framing.py            # write audit
  python3 morphoeval/audit_problem_framing.py --check    # CI-style guard
  python3 morphoeval/audit_problem_framing.py --self-test
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OPEN_PROBLEMS = ROOT / "docs" / "open-problems.json"
PREDICTIONS = ROOT / "predictions" / "prediction_registry.json"
LIVE_ARGUMENTS = ROOT / "predictions" / "live_arguments.json"
COUNTERCLAIMS = ROOT / "thompson" / "thompson_1917_counterclaims.json"
OUT_DIR = HERE / "promotions"

GENERIC_TEMPLATE = ("Is '<law>' the right generative law for this structure in <organism> — "
                    "and if not, what observation kills it?")

# Per problem type: the ask that fits, who can answer it, and how promotion should treat it.
FRAMING_RULES: dict[str, dict[str, Any]] = {
    "formal_proof": {
        "verdict": "misframed_use_native",
        "ask": "Is the stated gap exactly right, and is the proposed route (proof / counterexample / "
               "barrier result) known-dead in your field? Point to the obstruction or the paper.",
        "audience": "mathematicians / theoretical CS (symbolic dynamics, complexity, Lean formalisation)",
        "tiers": {"two_minute": "known-dead / open / wrong-gap-statement",
                  "fifteen_minute": "name the obstruction or the closest published result",
                  "one_sitting": "check the Lean loop file compiles with 0 sorry and the gap survives"},
        "track_hint": "promote_for_scrutiny only to the math audience; biologists are the wrong experts. "
                      "Posterior <15% → frame as 'barrier/negative result wanted', never as a result.",
    },
    "preregistered_blind_test": {
        "verdict": "misframed_use_native",
        "ask": "Before unblinding: does this pre-registration actually separate the hypotheses, and is "
               "the confound named in the gap really removed by the design? (Design review, not verdict.)",
        "audience": "experimentalists on the organism/assay + one methods/statistics reviewer",
        "tiers": {"two_minute": "design separates hypotheses: yes / no / not without X",
                  "fifteen_minute": "name a confound the registered scoring contract does not kill",
                  "one_sitting": "co-register: sign the hash-locked contract as external pre-registrant"},
        "track_hint": "Never promote the outcome before the registered test is scored; promote the "
                      "pre-registration for scrutiny. A negative result is promoted as a result.",
    },
    "missing_dataset": {
        "verdict": "misframed_use_native",
        "ask": "Do you have, or can you point to, exactly this series? The deliverable is data, "
               "not a verdict — the pipeline and hypotheses are already locked.",
        "audience": "labs that hold or can generate the named measurement; data curators",
        "tiers": {"two_minute": "have it / know who has it / does not exist",
                  "fifteen_minute": "point to the dataset or paper that supplies the series",
                  "one_sitting": "run the locked pipeline on your data without changing its hypotheses"},
        "track_hint": "Promote as a data request, with the locked pipeline attached. Closing = supplying "
                      "the series, not re-arguing the prediction.",
    },
    "independent_rescore": {
        "verdict": "misframed_use_native",
        "ask": "Blinded re-score of the named assessment tree with the current pipeline, published as a "
               "superseding assessment whichever way it comes out.",
        "audience": "independent assessor with no prior involvement (second-company / external auditor)",
        "tiers": {"two_minute": "willing to re-score blind: yes / no",
                  "fifteen_minute": "flag the one dimension most likely to move on re-score",
                  "one_sitting": "full blinded rubric re-score → superseding audit YAML"},
        "track_hint": "Promote as an audit request; result-track only after the superseding audit lands.",
    },
    "failure_autopsy": {
        "verdict": "misframed_use_native",
        "ask": "Why did the method fail in exactly these clades — landmark scheme, tree quality, or a real "
               "limit? Pick one and name the test that would confirm it.",
        "audience": "systematists / morphometricians who work on the failing clades",
        "tiers": {"two_minute": "landmark scheme / tree / real limit / other",
                  "fifteen_minute": "one sentence on why, with a citation or dataset",
                  "one_sitting": "convert one failure into a corrected, pre-registered prediction"},
        "track_hint": "Promote the failures themselves (negative results are results); the ask is a diagnosis.",
    },
    "structural_debt": {
        "verdict": "do_not_promote",
        "ask": None, "audience": "internal (builder loop)", "tiers": {},
        "track_hint": "Solved or internal engineering debt — not a scientist-facing finding.",
    },
    "registered_prediction": {
        "verdict": "blind_locked_use_native_verbatim",
        "ask": "Does the registered falsifier (quoted verbatim, locked before scoring) hold or fail on the "
               "evidence you know? If 'partially_resolved', can you supply the named missing series?",
        "audience": "experimentalists on the named organism/pathway",
        "tiers": {"two_minute": "falsifier holds / fails / cannot judge",
                  "fifteen_minute": "cite the measurement that decides it",
                  "one_sitting": "supply or locate the missing series named in the resolution record"},
        "track_hint": "The falsifier field is LOCKED (blind discipline) — promotion must quote it, never "
                      "paraphrase or replace it. 'confirmed' outcomes are result-track candidates once a "
                      "human expert agrees with the resolution.",
    },
    "live_argument": {
        "verdict": "adjudicate_camps",
        "ask": "Which camp does current evidence favour, and does the proposed settling measurement really "
               "discriminate between the camps? (Adjudication, not a kill-one-law question.)",
        "audience": "representatives of each camp (COI: each is a stakeholder — balance, do not exclude)",
        "tiers": {"two_minute": "camp A / B / C / settling measurement does not discriminate",
                  "fifteen_minute": "say what the settling measurement would show under YOUR camp",
                  "one_sitting": "co-register the settling measurement as an adversarial collaboration"},
        "track_hint": "Promote as adversarial-collaboration proposal; settled halves must not be re-litigated.",
    },
    "simulation_claim": {
        "verdict": "generic_fits",
        "ask": GENERIC_TEMPLATE, "audience": "developmental biologists on the organism",
        "tiers": {}, "track_hint": "Current promote_findings.py framing is correct for this type.",
    },
}


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# --------------------------------------------------------------------------- classification
def classify_open_problem(p: dict[str, Any]) -> str:
    """Type an OP node from its id/title/gap/done text (deterministic keyword rules)."""
    blob = " ".join(str(p.get(k, "")) for k in ("title", "gap", "done")).lower()
    if p.get("status") == "solved":
        return "structural_debt"
    if re.search(r"\bproof\b|lean-checked|discrepancy bound|barrier|relativisation|counterexample word", blob):
        return "formal_proof"
    if re.search(r"re-score|rescore|independent assess|superseding assessment", blob):
        return "independent_rescore"
    if re.search(r"autopsy|failures? (are|is) concentrated|failed", blob):
        return "failure_autopsy"
    if re.search(r"dataset with replicates|missing input|would make .* exact", blob):
        return "missing_dataset"
    if re.search(r"pre-register|preregist|blind test|unblinding|hash-locked", blob):
        return "preregistered_blind_test"
    if re.search(r"structural debt|self-test that fails", blob):
        return "structural_debt"
    return "simulation_claim"


def audit_open_problems(doc: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for p in doc.get("problems", []):
        ptype = classify_open_problem(p)
        rule = FRAMING_RULES[ptype]
        viab = (p.get("viability") or {}).get("pooled")
        hint = rule["track_hint"]
        if viab is not None and viab < 15 and ptype == "formal_proof":
            hint += f" (pooled Delphi posterior {viab}% — ask for the barrier/negative result, not the proof)"
        rows.append({
            "node_id": p["id"], "source": "docs/open-problems.json", "title": p.get("title"),
            "status": p.get("status"), "viability_pooled_percent": viab,
            "problem_type": ptype,
            "native_falsifier": p.get("done"),
            "generic_framing": GENERIC_TEMPLATE,
            "framing_verdict": rule["verdict"],
            "correct_ask": rule["ask"], "audience": rule["audience"], "ask_tiers": rule["tiers"],
            "promotion_track_hint": hint,
            "graph_url": p.get("graph_url"),
        })
    return rows


def audit_predictions(reg: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    rule = FRAMING_RULES["registered_prediction"]
    for p in reg.get("predictions", []):
        res = p.get("resolvability") or {}
        rows.append({
            "node_id": p["prediction_id"], "source": "predictions/prediction_registry.json",
            "title": p.get("prediction", "")[:140], "status": p.get("outcome") or p.get("status"),
            "organism": p.get("organism"), "mechanism": p.get("mechanism"), "datasets": p.get("datasets") or [],
            "problem_type": "registered_prediction",
            "native_falsifier": p.get("falsifier"),
            "native_falsifier_locked_on": p.get("registered_on"),
            "generic_framing": GENERIC_TEMPLATE,
            "framing_verdict": rule["verdict"] if p.get("falsifier") else "no_falsifier_needs_registration",
            "correct_ask": rule["ask"], "audience": rule["audience"], "ask_tiers": rule["tiers"],
            "promotion_track_hint": rule["track_hint"] + (
                " Outcome 'confirmed' → candidate for promote_as_result after one human expert concurs."
                if p.get("outcome") == "confirmed" else
                " Outcome 'partially_resolved' → promote as a DATA REQUEST for the named missing series."
                if p.get("outcome") == "partially_resolved" else ""),
            "resolution_route": res.get("resolution_route"),
        })
    return rows


def audit_live_arguments(la: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    rule = FRAMING_RULES["live_argument"]
    for a in la.get("arguments", []):
        rows.append({
            "node_id": a["id"], "source": "predictions/live_arguments.json",
            "title": a.get("question", "")[:160], "status": a.get("argument_status"),
            "camps": [c.get("side") if isinstance(c, dict) else c for c in a.get("camps") or []],
            "datasets": a.get("related_datasets") or [],
            "problem_type": "live_argument",
            "native_falsifier": a.get("settling_measurement"),
            "settled_part": a.get("settled_part"),
            "generic_framing": GENERIC_TEMPLATE,
            "framing_verdict": rule["verdict"],
            "correct_ask": rule["ask"], "audience": rule["audience"], "ask_tiers": rule["tiers"],
            "promotion_track_hint": rule["track_hint"] + (
                " Settled part exists — promote ONLY the remaining live part." if a.get("settled_part") else ""),
        })
    return rows


def audit_followups(doc: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    rule = FRAMING_RULES["missing_dataset"]
    for f in (doc.get("registered_prediction_followups") or {}).get("items", []):
        rows.append({
            "node_id": f["id"] + ":followup", "source": "docs/open-problems.json#registered_prediction_followups",
            "title": f"Missing series for {f['id']} ({f.get('organism')}, {f.get('mechanism')})",
            "status": "partially_resolved", "organism": f.get("organism"), "mechanism": f.get("mechanism"),
            "problem_type": "missing_dataset",
            "native_falsifier": f.get("missing_evidence"),
            "generic_framing": GENERIC_TEMPLATE, "framing_verdict": rule["verdict"],
            "correct_ask": rule["ask"], "audience": rule["audience"], "ask_tiers": rule["tiers"],
            "promotion_track_hint": rule["track_hint"],
        })
    return rows


def audit_counterclaims(cc: dict[str, Any]) -> dict[str, Any]:
    """Thompson 1917 claim/counterclaim nanopubs are HISTORICAL claims with dated
    literature verdicts — they are a record, not open problems, and must not be
    promoted as findings. Summarise only."""
    claims = cc.get("claims") or {}
    by_rel: dict[str, int] = {}
    for c in claims.values():
        by_rel[c.get("relation", "?")] = by_rel.get(c.get("relation", "?"), 0) + 1
    return {"source": "thompson/thompson_1917_counterclaims.json (→ nanopubs/thompson_predictions/*claim1917*)",
            "n_claims": len(claims), "by_relation": by_rel,
            "framing_verdict": "do_not_promote",
            "note": "Historical claim + literature verdict records (signed nanopubs). Not open problems; "
                    "promoting them as findings would misrepresent settled literature as new results. "
                    "Only 'repo_2026_status_separate' items that are OPEN re-enter via the problem graph."}


# --------------------------------------------------------------------------- lookup for promote_findings
def native_framing_for(organism: str | None, mechanism: str | None, datasets: list[str] | None,
                       rows: list[dict[str, Any]] | None = None) -> dict[str, Any] | None:
    """Find the problem-graph node a finding belongs to (by dataset dir, else
    organism+mechanism) and return its native falsifier + correct ask. Used by
    promote_findings.falsifier_for so generic framing is only the fallback."""
    rows = rows if rows is not None else run_audit(write=False)["rows"]
    ds = {d.rstrip("/") for d in (datasets or [])}
    mech = (mechanism or "").replace("-", "_").lower()
    org = (organism or "").lower()
    best = None
    for r in rows:
        if r["framing_verdict"] == "do_not_promote":
            continue
        r_ds = {d.rstrip("/") for d in (r.get("datasets") or [])}
        if ds and ds & r_ds:
            return r
        if org and r.get("organism") and org == str(r["organism"]).lower() and mech and \
                mech == str(r.get("mechanism") or "").replace("-", "_").lower():
            best = best or r
    return best


# --------------------------------------------------------------------------- run
def run_audit(write: bool = True) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    extras: dict[str, Any] = {}
    if OPEN_PROBLEMS.exists():
        op = json.loads(OPEN_PROBLEMS.read_text(encoding="utf-8"))
        rows += audit_open_problems(op) + audit_followups(op)
    if PREDICTIONS.exists():
        rows += audit_predictions(json.loads(PREDICTIONS.read_text(encoding="utf-8")))
    if LIVE_ARGUMENTS.exists():
        rows += audit_live_arguments(json.loads(LIVE_ARGUMENTS.read_text(encoding="utf-8")))
    if COUNTERCLAIMS.exists():
        extras["thompson_1917_counterclaims"] = audit_counterclaims(json.loads(COUNTERCLAIMS.read_text(encoding="utf-8")))

    by_verdict: dict[str, int] = {}
    by_type: dict[str, int] = {}
    for r in rows:
        by_verdict[r["framing_verdict"]] = by_verdict.get(r["framing_verdict"], 0) + 1
        by_type[r["problem_type"]] = by_type.get(r["problem_type"], 0) + 1
    generic_ok = by_verdict.get("generic_fits", 0)
    result = {"generated_at": utc_now(), "generic_template": GENERIC_TEMPLATE,
              "n_nodes": len(rows), "n_generic_framing_fits": generic_ok,
              "n_would_be_misframed_by_generic": len(rows) - generic_ok - by_verdict.get("do_not_promote", 0),
              "by_framing_verdict": by_verdict, "by_problem_type": by_type,
              "framing_rules": FRAMING_RULES, "rows": rows, **extras}
    if write:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        (OUT_DIR / "FRAMING_AUDIT.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        (OUT_DIR / "FRAMING_AUDIT.md").write_text(render_md(result), encoding="utf-8")
    return result


def render_md(res: dict[str, Any]) -> str:
    L = ["# Framing audit — is the promotion falsifier right for each problem?", "",
         f"Generated {res['generated_at']}. Nodes audited: **{res['n_nodes']}** "
         f"(problem graph OPs, prediction followups, registered predictions, live arguments).", "",
         f"Generic template currently used by `promote_findings.py`:", f"> {res['generic_template']}", "",
         f"**Fits as-is: {res['n_generic_framing_fits']}. Would be misframed by it: "
         f"{res['n_would_be_misframed_by_generic']}. Do-not-promote: {res['by_framing_verdict'].get('do_not_promote', 0)}.**", "",
         "## Verdict counts", "", "| Framing verdict | n |", "|---|---|"]
    L += [f"| {k} | {v} |" for k, v in sorted(res["by_framing_verdict"].items())]
    L += ["", "## Problem types found", "", "| Type | n | Correct ask | Audience |", "|---|---|---|---|"]
    for t, n in sorted(res["by_problem_type"].items()):
        r = FRAMING_RULES[t]
        L.append(f"| {t} | {n} | {(r['ask'] or '—')[:110]} | {r['audience']} |")
    L += ["", "## Per node", ""]
    for r in res["rows"]:
        L += [f"### {r['node_id']} — {r['problem_type']} · `{r['framing_verdict']}`",
              f"*{(r.get('title') or '')[:160]}*  ", f"status: `{r.get('status')}`"
              + (f" · pooled posterior {r['viability_pooled_percent']}%" if r.get("viability_pooled_percent") is not None else ""), "",
              f"- **Native falsifier (use this):** {(r.get('native_falsifier') or '— none registered')[:600]}",
              f"- **Correct ask:** {r.get('correct_ask') or '— do not promote'}",
              f"- **Audience:** {r.get('audience')}",
              f"- **Promotion:** {r.get('promotion_track_hint')}"]
        if r.get("camps"):
            L.append(f"- **Camps:** {', '.join(map(str, r['camps']))}")
        L.append("")
    cc = res.get("thompson_1917_counterclaims")
    if cc:
        L += ["## Thompson 1917 claim/counterclaim nanopubs", "",
              f"{cc['n_claims']} historical claims, relations {cc['by_relation']} — **{cc['framing_verdict']}**. {cc['note']}", ""]
    return "\n".join(L)


def self_test() -> int:
    assert classify_open_problem({"title": "Prize 1", "gap": "needs a global invariant", "done": "A proof (ideally Lean-checked) ... or a counterexample word", "status": "open"}) == "formal_proof"
    assert classify_open_problem({"title": "x", "gap": "y", "done": "A pre-registered deep-horizon test (registered before unblinding)", "status": "partial"}) == "preregistered_blind_test"
    assert classify_open_problem({"title": "bee P-matrix (the one missing input)", "gap": "", "done": "A within-species dataset with replicates", "status": "partial"}) == "missing_dataset"
    assert classify_open_problem({"title": "x", "gap": "", "done": "A blinded independent re-score", "status": "refuted"}) == "independent_rescore"
    assert classify_open_problem({"title": "x", "gap": "the failures are concentrated", "done": "A per-clade failure autopsy", "status": "partial"}) == "failure_autopsy"
    assert classify_open_problem({"title": "x", "gap": "", "done": "", "status": "solved"}) == "structural_debt"
    res = run_audit(write=False)
    assert res["n_nodes"] >= 30, res["n_nodes"]
    assert res["n_generic_framing_fits"] == 0, "no graph node should fit the generic simulation-claim template"
    preds = [r for r in res["rows"] if r["problem_type"] == "registered_prediction"]
    assert preds and all(r["framing_verdict"] == "blind_locked_use_native_verbatim" for r in preds)
    hit = native_framing_for("Danio rerio", "turing_wavelength", ["datasets/danio_rerio_turing_wavelength"], res["rows"])
    assert hit and hit["node_id"].startswith(("ZF-TUR", "LA-")), hit and hit["node_id"]
    assert native_framing_for("Neurospora crassa", "biomineral_accretion", [], res["rows"]) is None
    print(f"[audit_problem_framing] self-test PASSED ({res['n_nodes']} nodes; generic fits {res['n_generic_framing_fits']}; "
          f"misframed-by-generic {res['n_would_be_misframed_by_generic']})")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="exit 1 if any node lacks a framing rule or a native falsifier")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args(argv)
    if args.self_test:
        return self_test()
    res = run_audit(write=True)
    print(f"[audit_problem_framing] {res['n_nodes']} nodes → generic fits {res['n_generic_framing_fits']}, "
          f"misframed-by-generic {res['n_would_be_misframed_by_generic']}, do-not-promote {res['by_framing_verdict'].get('do_not_promote', 0)}")
    for k, v in sorted(res["by_problem_type"].items()):
        print(f"  {k:28s} {v:3d}  → {FRAMING_RULES[k]['verdict']}")
    print(f"  wrote {OUT_DIR / 'FRAMING_AUDIT.md'}")
    if args.check:
        bad = [r["node_id"] for r in res["rows"] if r["framing_verdict"] not in ("do_not_promote",) and not r.get("native_falsifier")]
        if bad:
            print(f"[audit_problem_framing] FAIL — nodes without a native falsifier: {bad}", file=sys.stderr)
            return 1
        print("[audit_problem_framing] check PASSED — every promotable node has a native falsifier")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

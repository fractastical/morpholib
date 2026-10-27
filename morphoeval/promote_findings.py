#!/usr/bin/env python3
"""
morphoeval/promote_findings.py — Scientist Promotion loop (Tier C).

Finds the findings scientists would actually care about and rewrites each one
into the version they would actually act on. See `promotion.yaml` for the
rubric/gates and the reasoning; the short version:

  * Promotion is fail-closed and two-track. Unvalidated findings are never
    promoted as results — they are promoted *for scrutiny*, as a specific,
    cheap, killable question. Only human-validated findings (consensus from
    `human_validation.py`) may be promoted as results.
  * The scientist-facing artifact is NOT the audit/validation file. It is a
    one-page action brief (`promotions/<finding>-scientist-brief.md`): one
    falsifier question, the comparison figure, three numbers, upfront caveats,
    and asks tiered by cost (2-minute triage -> 15-minute verdict -> one-sitting
    replication with the exact command).
  * Every send/response is recorded (never auto-sent), so we learn the funnel:
    care_rate, time_to_first_care_hours, scrutiny_conversion, and — joined with
    the validation loop — time_to_validate_or_not_hours.

Usage:
  python3 morphoeval/promote_findings.py rank [--limit 20]
  python3 morphoeval/promote_findings.py brief --finding <sim_id> [--expert <expert_id>] [--out F]
  python3 morphoeval/promote_findings.py build [--top 10]          # briefs + QUEUE.md/json
  python3 morphoeval/promote_findings.py track --finding <id> --event sent|care_response|accepted_review \
      [--expert <id>] [--channel direct_expert_invitation] [--care care|dont_care|kill_it|wrong_expert] [--at ISO]
  python3 morphoeval/promote_findings.py report [--md OUT.md]
  python3 morphoeval/promote_findings.py --self-test
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import human_validation as hv  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
CONFIG_PATH = HERE / "promotion.yaml"
PROMO_DIR = HERE / "promotions"


def now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def load_config() -> dict[str, Any]:
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8")) or {}


def slug_from_script(script: str | None) -> str:
    if not script:
        return ""
    stem = Path(script).stem
    parts = stem.split("-")
    # <year>-<author>-<slug...> | 2026-delphi10000-<slug...>
    return "-".join(parts[2:]) if len(parts) > 2 else stem


# --------------------------------------------------------------------------- collection
def collect_findings(limit_enrich: int = 60) -> list[dict[str, Any]]:
    """Cheap pass over audits/*.yaml (best audit per sim), then enrich the top
    candidates with runs / math-KG / reference / media / validation artifacts."""
    best: dict[str, dict[str, Any]] = {}
    for af in sorted((HERE / "audits").glob("*.yaml")):
        if af.name.endswith(".measurements.yaml"):
            continue
        try:
            doc = yaml.safe_load(af.read_text(encoding="utf-8")) or {}
        except Exception:
            continue
        for sim in doc.get("simulations") or []:
            sid = sim.get("id")
            if not sid:
                continue
            row = {"finding_id": sid, "organism": sim.get("organism"), "name": sim.get("name"),
                   "script": sim.get("script"), "overall_score": float(sim.get("overall_score") or 0),
                   "scores": {k: (v or {}).get("score") for k, v in (sim.get("scores") or {}).items()},
                   "phenomenon_size_m": sim.get("phenomenon_size_m"),
                   "audit_id": doc.get("audit_id"), "audit_file": str(af.relative_to(ROOT)),
                   "references": doc.get("references") or []}
            if sid not in best or row["overall_score"] > best[sid]["overall_score"]:
                best[sid] = row
    rows = sorted(best.values(), key=lambda r: -r["overall_score"])
    for row in rows[:limit_enrich]:
        enrich(row)
    return rows


def enrich(row: dict[str, Any]) -> dict[str, Any]:
    slug = slug_from_script(row.get("script"))
    uslug = slug.replace("-", "_")
    media = ROOT / "docs" / "media"
    rel = lambda p: str(p.relative_to(ROOT))  # briefs must be portable — never absolute local paths
    row["comparison_figure"] = next((rel(p) for p in
        [media / f"{uslug}_comparison.png", media / f"{uslug}_s0_comparison.png"] if p.exists()), None)
    row["animation"] = next((rel(p) for p in
        [media / f"{uslug}_growth.gif", media / f"{uslug}_s0_growth.gif"] if p.exists()), None)
    row["script_exists"] = bool(row.get("script")) and (ROOT / str(row["script"])).exists()

    mk = HERE / "math_kg_loops" / f"{row['finding_id']}.yaml"
    row["equations"], row["math_family"], row["lean_module"] = [], None, None
    if mk.exists():
        doc = yaml.safe_load(mk.read_text(encoding="utf-8")) or {}
        row["equations"] = doc.get("equations") or []
        row["math_family"] = doc.get("math_family")

    row["image_fitness"], row["pearson_r"], row["rmse"], row["parameters"] = None, None, None, {}
    if slug:
        runs_dir = HERE / "runs"
        if not hasattr(enrich, "_run_files"):
            enrich._run_files = [p.name for p in runs_dir.iterdir()] if runs_dir.exists() else []  # type: ignore[attr-defined]
        match = next((n for n in enrich._run_files if n.endswith(f"-{slug}-loop.yaml")), None)  # type: ignore[attr-defined]
        if match:
            doc = yaml.safe_load((runs_dir / match).read_text(encoding="utf-8")) or {}
            row["image_fitness"] = (doc.get("run") or {}).get("image_fitness")
        # direct path, no directory scans: datasets/<slug>/<slug>_reference.json
        for cand_dir in (uslug, f"{uslug}_s0"):
            ref = ROOT / "datasets" / cand_dir / f"{cand_dir}_reference.json"
            if ref.exists():
                doc = json.loads(ref.read_text(encoding="utf-8"))
                es = doc.get("empirical_series") or {}
                row["pearson_r"], row["rmse"] = es.get("pearson_r"), es.get("rmse")
                row["parameters"] = doc.get("parameters") or {}
                row["dataset_dir"] = str(ref.parent.relative_to(ROOT))
                break

    vf = hv.path_for(row["finding_id"])
    row["validation"] = None
    if vf.exists():
        req = hv.refresh(hv.load(vf))
        elig = [e for e in req["experts"] if e.get("eligibility", {}).get("eligible")]
        row["validation"] = {"file": str(vf.relative_to(ROOT)), "consensus": req["verdict_summary"]["consensus"],
                             "eligible_experts": len(elig),
                             "field_citations": sum(int(e.get("cited_by_count") or 0) for e in elig),
                             "experts": req["experts"],
                             "time_to_validate_or_not_hours": req["timing"]["time_to_validate_or_not_hours"]}
    row["falsifier"] = falsifier_for(row)
    return row


_FRAMING_ROWS: list[dict[str, Any]] | None = None


def _framing_rows() -> list[dict[str, Any]]:
    """Problem-graph nodes with their native falsifiers (audit_problem_framing.py), cached."""
    global _FRAMING_ROWS
    if _FRAMING_ROWS is None:
        try:
            import audit_problem_framing as apf  # noqa: WPS433
            _FRAMING_ROWS = apf.run_audit(write=False)["rows"]
        except Exception:
            _FRAMING_ROWS = []
    return _FRAMING_ROWS


def mechanism_from_slug(slug: str) -> str | None:
    """'drosophila-guttifera-flux-canalization' -> 'flux_canalization' (last 2 tokens, s0 stripped)."""
    toks = [t for t in slug.split("-") if t and not re.fullmatch(r"s\d+", t)]
    return "_".join(toks[-2:]) if len(toks) >= 2 else None


def falsifier_for(row: dict[str, Any]) -> str | None:
    """Native problem-graph falsifier first; generic simulation-claim template only as fallback.

    FRAMING_AUDIT (audit_problem_framing.py) showed 0/36 problem-graph nodes fit the generic
    "is <law> the right generative law?" question: proofs want a barrier/counterexample,
    pre-registrations want a design review, followups want a data series, live arguments
    want camp adjudication, and registered predictions carry a falsifier LOCKED before
    scoring that must be quoted verbatim.
    """
    slug = slug_from_script(row.get("script"))
    datasets = [row["dataset_dir"]] if row.get("dataset_dir") else []
    try:
        import audit_problem_framing as apf
        node = apf.native_framing_for(row.get("organism"), mechanism_from_slug(slug) if slug else None,
                                      datasets, _framing_rows())
    except Exception:
        node = None
    if node and node.get("native_falsifier"):
        row["problem_ref"] = {"node_id": node["node_id"], "problem_type": node["problem_type"],
                              "framing_verdict": node["framing_verdict"], "source": node["source"],
                              "correct_ask": node.get("correct_ask"), "audience": node.get("audience"),
                              "ask_tiers": node.get("ask_tiers") or {},
                              "locked_on": node.get("native_falsifier_locked_on")}
        return node["native_falsifier"]
    if not row.get("equations"):
        return None
    law = row["equations"][0].get("law_name") or row.get("math_family") or "the named model"
    row["problem_ref"] = {"node_id": None, "problem_type": "simulation_claim", "framing_verdict": "generic_fits"}
    return (f"Is '{law}' the right generative law for this structure in "
            f"{row.get('organism') or 'this organism'} — and if not, what observation kills it?")


# --------------------------------------------------------------------------- scoring + gates
def score_finding(row: dict[str, Any], cfg: dict | None = None) -> dict[str, Any]:
    cfg = cfg or load_config()
    comm = (cfg.get("care_score", {}).get("community_relevance", {}).get("organisms") or {})
    val = row.get("validation") or {}
    elig = int(val.get("eligible_experts") or 0)
    sub = {
        # a live audience we can actually name (validation-loop expert pool)
        "audience_demand": min(25.0, 6 * elig + min(8.0, (val.get("field_citations") or 0) / 500.0)
                               + (5.0 if elig else 0.0)),
        "evidence_strength": round(25 * (0.55 * (row.get("overall_score") or 0) / 100.0
                                         + 0.30 * ((row.get("image_fitness") or 0) / 100.0)
                                         + 0.15 * (1.0 if row.get("pearson_r") and row["pearson_r"] >= 0.98 else 0.0)), 2),
        "specificity": min(20.0, (8.0 if row.get("equations") else 0.0) + min(6.0, 2 * len(row.get("parameters") or {}))
                           + (4.0 if row.get("pearson_r") is not None else 0.0)
                           + (2.0 if row.get("phenomenon_size_m") else 0.0)),
        "actionability": min(20.0, (6.0 if row.get("script_exists") else 0.0)
                             + (6.0 if row.get("comparison_figure") else 0.0)
                             + (3.0 if row.get("animation") else 0.0) + (5.0 if row.get("dataset_dir") else 0.0)),
        "community_relevance": float(comm.get(row.get("organism") or "", 3)),
    }
    care = round(sum(sub.values()), 1)
    gates = cfg.get("gates", {})
    consensus = val.get("consensus")
    missing_result = [r for r, ok in (("human validation consensus", consensus in (gates.get("promote_as_result", {}).get("human_consensus_in") or [])),
                                      ("overall >= gate", (row.get("overall_score") or 0) >= gates.get("promote_as_result", {}).get("min_overall_score", 70)),
                                      ("comparison figure", bool(row.get("comparison_figure")))) if not ok]
    g = gates.get("promote_for_scrutiny", {})
    missing_scrutiny = [r for r, ok in (("care >= gate", care >= g.get("min_care_score", 45)),
                                        ("overall >= gate", (row.get("overall_score") or 0) >= g.get("min_overall_score", 50)),
                                        ("runnable script", bool(row.get("script_exists"))),
                                        ("comparison figure", bool(row.get("comparison_figure"))),
                                        ("named law/equation", bool(row.get("equations"))),
                                        ("falsifier question", bool(row.get("falsifier"))),
                                        ("eligible experts to ask", elig >= g.get("min_eligible_experts", 1))) if not ok]
    track = ("promote_as_result" if not missing_result else
             "promote_for_scrutiny" if not missing_scrutiny else "hold_do_not_promote")
    return {"care_score": care, "subscores": sub, "track": track,
            "missing_for_result": missing_result, "missing_for_scrutiny": missing_scrutiny,
            "eligible_experts": elig}


# --------------------------------------------------------------------------- the brief (what scientists act on)
def render_brief(row: dict[str, Any], scored: dict[str, Any], expert: dict[str, Any] | None = None,
                 cfg: dict | None = None) -> str:
    cfg = cfg or load_config()
    track = scored["track"]
    banner = {"promote_as_result": "✅ HUMAN-VALIDATED RESULT — promoted as a result",
              "promote_for_scrutiny": "🔎 UNVALIDATED — promoted for scrutiny, not as a result",
              "hold_do_not_promote": "⏸ HOLD — do not promote yet"}[track]
    org = row.get("organism") or "this organism"
    law = (row["equations"][0].get("law_name") if row.get("equations") else None) or row.get("math_family") or "the model"
    why_you = ""
    if expert:
        works = "; ".join(w.get("title", "") for w in (expert.get("example_works") or [])[:2])
        why_you = (f"\n**Why you, {expert.get('name')}:** you work on exactly this "
                   f"({', '.join(expert.get('institutions') or [])}). Your relevant work: {works}. "
                   f"If this is the wrong expertise, reply `wrong_expert` — that is a useful answer.\n")
    fig = f"\n![Simulation vs specimen]({row['comparison_figure']})\n" if row.get("comparison_figure") else "\n_(no comparison figure yet — one reason this may be on hold)_\n"
    script = row.get("script") or "<script>"
    pref = row.get("problem_ref") or {}
    ptype = pref.get("problem_type", "simulation_claim")
    tiers = pref.get("ask_tiers") or {}
    heading = {
        "registered_prediction": f"{org}: registered prediction `{pref.get('node_id')}` — does its locked falsifier hold?",
        "live_argument": f"{org}: live argument `{pref.get('node_id')}` — which camp, and does the settling measurement discriminate?",
        "missing_dataset": f"{org}: one missing data series would close `{pref.get('node_id')}`",
        "preregistered_blind_test": f"{org}: pre-registration `{pref.get('node_id')}` — design review before unblinding",
        "formal_proof": f"`{pref.get('node_id')}`: is this gap statement right, and is the route known-dead?",
        "independent_rescore": f"`{pref.get('node_id')}`: blinded independent re-score wanted",
        "failure_autopsy": f"`{pref.get('node_id')}`: why did the method fail in exactly these clades?",
    }.get(ptype, f"{org}: does {law} actually generate this form?")
    if ptype == "registered_prediction":
        falsifier_block = (f"**Registered falsifier (locked {pref.get('locked_on') or 'before scoring'} — quoted verbatim, not paraphrased):**\n"
                           f"> {row.get('falsifier')}\n\n**The ask:** {pref.get('correct_ask')}")
    elif ptype != "simulation_claim" and pref.get("node_id"):
        falsifier_block = (f"**What would close `{pref.get('node_id')}` (its own done-criterion):**\n> {row.get('falsifier')}\n\n"
                           f"**The ask:** {pref.get('correct_ask')}")
    else:
        falsifier_block = row.get("falsifier") or "No falsifier formulated yet — do not promote until there is one."
    triage_words = tiers.get("two_minute", "`care` / `dont_care` / `kill_it` / `wrong_expert`")
    problem_line = (f"\nProblem-graph node: `{pref.get('node_id')}` ({ptype}; framing `{pref.get('framing_verdict')}`; "
                    f"source `{pref.get('source')}`). Right audience: {pref.get('audience')}.\n") if pref.get("node_id") else ""
    return f"""# {heading}
{banner} · care score {scored['care_score']}/100 · finding `{row['finding_id']}`
{problem_line}{why_you}
## The one question (15 minutes to answer, 2 minutes to triage)
{falsifier_block}

**Triage (2 min):** reply one word — {triage_words}. Every answer is recorded and credited; `kill_it` closes this finding and saves everyone time.

## Why a scientist in this field might care
- A named, falsifiable law (`{law}`) is pinned to a real structure in *{org}* at {row.get('phenomenon_size_m')} m — it makes a quantitative claim that can simply be wrong.
- It ships with a runnable reproduction and a side-by-side specimen comparison (below), so you judge evidence, not prose.
- {('An active expert community exists for this system — ' + str(scored['eligible_experts']) + ' eligible, conflict-free reviewers already identified.') if scored['eligible_experts'] else 'No eligible expert pool identified yet — finding reviewers is step one, promotion step two.'}
{fig}
## Three numbers (the whole evidence case)
| Metric | Value | Source |
|---|---|---|
| AI-audit overall (MorphoEval, uncalibrated) | {row.get('overall_score')} / 100 | `{row.get('audit_file')}` |
| Simulation-vs-specimen image fitness | {row.get('image_fitness')} | run telemetry |
| Fit to empirical series (Pearson r / RMSE) | {row.get('pearson_r')} / {row.get('rmse')} | vendored reference data |

## What to do — pick your cost
1. **2-minute triage** — one word above.
2. **15-minute verdict** — {tiers.get('fifteen_minute', 'answer the one question: `validated` / `partially_validated` / `refuted` / `unable_to_assess`, plus one sentence')}. Recorded with your name (or anonymously) in the validation loop; time-to-verdict is published with the finding.
3. **One-sitting replication** — {tiers.get('one_sitting', 'run the reproduction yourself')}:
   ```bash
   python3 {script} --self-test && python3 {script}
   ```
   If the figure does not reproduce, that is a `refuted` verdict with evidence.

## What would kill it
The falsifier above. A single expert `refuted` + one confirming verdict closes this finding as refuted — and that closure, with its elapsed time, is part of the public record.

## Caveats, upfront
- This finding was produced and AI-audited by automated loops; **no human expert has validated it** (unless the banner above says otherwise). Scores are uncalibrated rubric ratings, not probabilities.
- Claim boundary: the code implements the cited law; whether that law is the right biology is exactly the open question.
- You are never auto-emailed again by this system; a human sent this, and only recorded replies count.

## Credit & record
Replies feed `morphoeval/validations/{row['finding_id']}-human-validation.json` (verdict + timing) and this promotion's funnel record. Reviewers are named on the finding unless they ask otherwise.
{f"Promotion status so far: track=`{track}`, missing for result-track: {scored['missing_for_result'] or '—'}" if track != 'promote_as_result' else ''}
"""


# --------------------------------------------------------------------------- funnel tracking
def promo_path(finding_id: str, directory: Path = PROMO_DIR) -> Path:
    slug = "".join(c if c.isalnum() or c in "-_" else "-" for c in finding_id)
    return directory / f"{slug}-promotion.json"


def track_event(finding_id: str, event: str, directory: Path = PROMO_DIR, **kw: Any) -> dict[str, Any]:
    p = promo_path(finding_id, directory)
    doc = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {
        "schema_version": "morphoeval-promotion-1.0", "finding_id": finding_id,
        "validation_file": str(hv.path_for(finding_id)), "events": []}
    doc["events"].append({"event": event, "at": kw.pop("at", None) or now_iso(), **{k: v for k, v in kw.items() if v is not None}})
    ev = sorted(doc["events"], key=lambda e: e["at"])
    first = lambda name: next((e["at"] for e in ev if e["event"] == name), None)  # noqa: E731
    cares = [e for e in ev if e["event"] == "care_response"]
    doc["funnel"] = {
        "brief_ready_at": first("brief_ready"), "first_sent_at": first("sent"),
        "first_care_response_at": first("care_response"),
        "time_to_first_care_hours": hv.hours_between(first("sent"), first("care_response")),
        "sends": sum(1 for e in ev if e["event"] == "sent"), "care_responses": len(cares),
        "care_rate": round(sum(1 for e in cares if e.get("care") == "care") / len(cares), 3) if cares else None,
        "kill_it_count": sum(1 for e in cares if e.get("care") == "kill_it"),
        "converted_to_review": any(e["event"] == "accepted_review" for e in ev),
        "verdict_recorded": any(e["event"] == "verdict" for e in ev)}
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    return doc


def report(directory: Path = PROMO_DIR) -> dict[str, Any]:
    docs = []
    for p in sorted(directory.glob("*-promotion.json")) if directory.exists() else []:
        try: docs.append(json.loads(p.read_text(encoding="utf-8")))
        except Exception: continue
    responded = [d for d in docs if d["funnel"]["care_responses"]]
    times = [d["funnel"]["time_to_first_care_hours"] for d in docs if d["funnel"]["time_to_first_care_hours"] is not None]
    return {"generated_at": now_iso(), "n_promotions": len(docs), "n_with_care_response": len(responded),
            "overall_care_rate": round(sum(d["funnel"]["care_rate"] or 0 for d in responded) / len(responded), 3) if responded else None,
            "median_time_to_first_care_hours": sorted(times)[len(times) // 2] if times else None,
            "n_converted_to_review": sum(1 for d in docs if d["funnel"]["converted_to_review"]),
            "n_verdicts": sum(1 for d in docs if d["funnel"]["verdict_recorded"]),
            "promotions": [{"finding_id": d["finding_id"], **d["funnel"]} for d in docs]}


# --------------------------------------------------------------------------- CLI
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true")
    sub = ap.add_subparsers(dest="cmd")
    p = sub.add_parser("rank"); p.add_argument("--limit", type=int, default=20)
    p = sub.add_parser("brief"); p.add_argument("--finding", required=True); p.add_argument("--expert"); p.add_argument("--out")
    p = sub.add_parser("build"); p.add_argument("--top", type=int, default=10); p.add_argument("--dir", default=str(PROMO_DIR))
    p = sub.add_parser("track"); p.add_argument("--finding", required=True)
    p.add_argument("--event", required=True, choices=["brief_ready", "sent", "care_response", "accepted_review", "verdict"])
    p.add_argument("--expert"); p.add_argument("--channel"); p.add_argument("--care", choices=["care", "dont_care", "kill_it", "wrong_expert"]); p.add_argument("--at")
    p = sub.add_parser("report"); p.add_argument("--dir", default=str(PROMO_DIR)); p.add_argument("--md")
    args = ap.parse_args(argv)
    if args.self_test:
        return self_test()
    cfg = load_config()

    if args.cmd == "rank":
        rows = collect_findings()
        print(f"{'care':>5} {'track':22s} {'overall':>7} {'experts':>7}  finding")
        for r in rows[: args.limit]:
            s = score_finding(r, cfg)
            print(f"{s['care_score']:5.1f} {s['track']:22s} {r['overall_score']:7.1f} {s['eligible_experts']:7d}  {r['finding_id']}")
        return 0

    if args.cmd == "brief":
        rows = collect_findings(limit_enrich=0)
        base = next((r for r in rows if r["finding_id"] == args.finding), None)
        if base is None:
            print(f"finding {args.finding} not found in audits/", file=sys.stderr); return 1
        row = enrich(base)
        scored = score_finding(row, cfg)
        expert = None
        if args.expert and row.get("validation"):
            expert = next((e for e in row["validation"]["experts"] if e["expert_id"] == args.expert), None)
        brief = render_brief(row, scored, expert, cfg)
        out = Path(args.out) if args.out else PROMO_DIR / f"{args.finding}-scientist-brief.md"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(brief, encoding="utf-8")
        track_event(args.finding, "brief_ready", PROMO_DIR)
        print(f"[promote] {scored['track']} care={scored['care_score']} -> {out}")
        return 0

    if args.cmd == "build":
        rows = collect_findings()
        scored_rows = sorted(((score_finding(r, cfg), r) for r in rows), key=lambda t: -t[0]["care_score"])
        d = Path(args.dir); d.mkdir(parents=True, exist_ok=True)
        queue = []
        for s, r in scored_rows[: args.top]:
            if "falsifier" not in r: enrich(r)
            out = d / f"{r['finding_id']}-scientist-brief.md"
            out.write_text(render_brief(r, s, cfg=cfg), encoding="utf-8")
            track_event(r["finding_id"], "brief_ready", d)
            queue.append({"finding_id": r["finding_id"], "organism": r["organism"], **s, "brief": str(out)})
        (d / "QUEUE.json").write_text(json.dumps({"generated_at": now_iso(), "queue": queue}, indent=2) + "\n", encoding="utf-8")
        lines = ["# Scientist Promotion Queue (ranked by care score; gates decide the track)\n",
                 "| Care | Track | Finding | Organism | Missing for scrutiny |", "|---|---|---|---|---|"]
        lines += [f"| {q['care_score']} | {q['track']} | {q['finding_id']} | {q['organism']} | {', '.join(q['missing_for_scrutiny']) or '—'} |" for q in queue]
        (d / "QUEUE.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"[promote] wrote {len(queue)} brief(s) + QUEUE.md/json to {d}")
        return 0

    if args.cmd == "track":
        if args.event == "care_response" and not args.care:
            print("care_response needs --care care|dont_care|kill_it|wrong_expert", file=sys.stderr); return 1
        doc = track_event(args.finding, args.event, PROMO_DIR, expert_id=args.expert,
                          channel=args.channel, care=args.care, at=args.at)
        print(f"[promote] {args.event} recorded for {args.finding}: funnel={doc['funnel']}")
        return 0

    if args.cmd == "report":
        rep = report(Path(args.dir))
        print(json.dumps(rep, indent=2))
        if args.md:
            lines = ["# Promotion funnel report\n", f"Generated {rep['generated_at']} — {rep['n_promotions']} promotion(s), "
                     f"care_rate={rep['overall_care_rate']}, median time-to-first-care={rep['median_time_to_first_care_hours']}h\n",
                     "| Finding | Sends | Care responses | Care rate | Time to first care (h) | Converted | Verdict |", "|---|---|---|---|---|---|---|"]
            lines += [f"| {p['finding_id']} | {p['sends']} | {p['care_responses']} | {p['care_rate']} | {p['time_to_first_care_hours']} | {p['converted_to_review']} | {p['verdict_recorded']} |" for p in rep["promotions"]]
            Path(args.md).write_text("\n".join(lines) + "\n", encoding="utf-8")
        return 0

    ap.print_help(); return 0


def self_test() -> int:
    cfg = load_config()
    row = {"finding_id": "test-finding", "organism": "Drosophila melanogaster", "script": "models/x.py",
           "overall_score": 62.0, "scores": {}, "phenomenon_size_m": 1e-4, "audit_file": "a.yaml",
           "image_fitness": 88.0, "pearson_r": 0.999, "rmse": 0.05, "parameters": {"a": 1, "b": 2},
           "equations": [{"law_name": "Test Law"}], "math_family": "Test", "script_exists": True,
           "comparison_figure": "docs/media/x_comparison.png", "animation": "x.gif", "dataset_dir": "datasets/x",
           "falsifier": "Is 'Test Law' right?", "validation": {"consensus": None, "eligible_experts": 3, "field_citations": 900}}
    s = score_finding(row, cfg)
    assert s["track"] == "promote_for_scrutiny", s          # unvalidated: never "result"
    assert "expert" not in s["missing_for_scrutiny"]
    row["validation"] = {"consensus": "validated", "eligible_experts": 3, "field_citations": 900}
    row["overall_score"] = 80.0
    assert score_finding(row, cfg)["track"] == "promote_as_result"
    row["validation"] = None; row["overall_score"] = 40.0
    assert score_finding(row, cfg)["track"] == "hold_do_not_promote"
    brief = render_brief(row, score_finding(row, cfg), cfg=cfg)
    for needle in ("2-minute triage", "kill_it", "Caveats, upfront", "What would kill it", "python3 models/x.py --self-test"):
        assert needle in brief, needle
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        track_event("test-finding", "sent", d, expert_id="oa-1", at="2026-10-01T00:00:00Z")
        doc = track_event("test-finding", "care_response", d, expert_id="oa-1", care="care", at="2026-10-01T06:00:00Z")
        assert doc["funnel"]["time_to_first_care_hours"] == 6.0 and doc["funnel"]["care_rate"] == 1.0
    print("[promote] self-test PASSED (two-track gates, brief contract, care funnel timing)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

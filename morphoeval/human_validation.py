#!/usr/bin/env python3
"""
morphoeval/human_validation.py — Tier C / Step 13: Human Expert Validation loop.

The 12-step MorphoEval loop (`track_loop.py`) is fully automated — Tier B's
"dual-agent audit" is two *AI* assessors. No human domain expert ever sees a
finding, so "validated" so far means "AI-audited". This module adds the
missing loop and, crucially, measures how long human validation takes —
including when it never happens:

    queued ──► expert_search ──► invited ──► accepted ──► in_review ──► verdict
                  │                 │           │
                  │                 ├─► declined ┐
                  │                 └─► no_response (after SLA) ┤
                  └─► no_expert_found ──────────────────────────┴─► closed_without_validation

Every transition is an append-only, timestamped event on one per-finding file
(`morphoeval/validations/<finding>-human-validation.json`, zero-conflict like
`audits/` and `runs/`). Derived timing (recomputed on every write, all hours):

    time_to_first_invite_hours     queued -> first invitation recorded
    time_to_first_response_hours   first invitation -> first accept/decline
    time_in_review_hours           first acceptance -> consensus verdict
    time_to_verdict_hours          queued -> consensus verdict
    time_to_non_validation_hours   queued -> closed_without_validation
    time_to_validate_or_not_hours  queued -> whichever terminal came first  ★

Ethics / scope: this tool FINDS and RANKS candidate experts (OpenAlex authors
of the wider literature on the finding's topic, source authors COI-excluded)
and RECORDS invitations a human sends. It never emails, messages, or otherwise
contacts anyone, and it never invents contact details — `contact_route` is a
human-filled field (institution page, ORCID, ...). See human_validation.yaml.

Usage:
  python3 morphoeval/human_validation.py init --run morphoeval/runs/<date>-<slug>-loop.yaml
  python3 morphoeval/human_validation.py init --audit morphoeval/audits/<...>-audit.yaml
  python3 morphoeval/human_validation.py init --finding-id <sim_id> --organism "..." --claim "..."
  python3 morphoeval/human_validation.py find-experts --validation <file> [--no-network] [--candidates-file pool.json]
  python3 morphoeval/human_validation.py assign --validation <file> --expert <expert_id>
  python3 morphoeval/human_validation.py record --validation <file> --event invited|accepted|declined|no_response|review_started|verdict|note --expert <id> [--verdict validated|partially_validated|refuted|unable_to_assess] [--channel email] [--notes "..."] [--at 2026-11-01T09:00:00Z]
  python3 morphoeval/human_validation.py close --validation <file> --outcome expired|declined_all|no_expert_found|withdrawn [--at ...]
  python3 morphoeval/human_validation.py status --validation <file>
  python3 morphoeval/human_validation.py report [--dir morphoeval/validations] [--md OUT.md] [--csv OUT.csv]
  python3 morphoeval/human_validation.py prompt --validation <file>   # cloud-agent expert-search prompt
  python3 morphoeval/human_validation.py --self-test
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import statistics
import sys
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
CONFIG_PATH = HERE / "human_validation.yaml"
VALIDATIONS_DIR = HERE / "validations"
SCHEMA = "morphoeval-human-validation-1.0"

VERDICTS = ("validated", "partially_validated", "refuted", "unable_to_assess")
NON_VALIDATION = ("expired", "declined_all", "no_expert_found", "withdrawn")
EVENTS = ("invited", "accepted", "declined", "no_response", "review_started",
          "verdict", "note", "coi_declared")


# --------------------------------------------------------------------------- time
def parse_ts(s: str | None) -> datetime | None:
    if not s:
        return None
    return datetime.fromisoformat(str(s).replace("Z", "+00:00")).astimezone(timezone.utc)


def now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def hours_between(a: str | None, b: str | None) -> float | None:
    ta, tb = parse_ts(a), parse_ts(b)
    if ta is None or tb is None:
        return None
    return round((tb - ta).total_seconds() / 3600.0, 3)


def load_config() -> dict[str, Any]:
    if CONFIG_PATH.exists():
        return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8")) or {}
    return {}


def route_expertise(cfg: dict, *texts: str) -> dict[str, str]:
    blob = " ".join(t for t in texts if t).lower()
    for r in cfg.get("expertise_routes", []):
        if any(str(m).lower() in blob for m in (r.get("match") or [])):
            return {"expertise": r["expertise"], "query": r["query"]}
    for r in cfg.get("expertise_routes", []):
        if r.get("id") == "default":
            return {"expertise": r["expertise"], "query": r["query"]}
    return {"expertise": "mathematical biology / morphogenesis (general)",
            "query": f"{blob} morphogenesis".strip()}


# --------------------------------------------------------------------------- finding intake
def finding_from_run(path: Path) -> dict[str, Any]:
    doc = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    run = doc.get("run", doc)
    return {
        "finding_id": run.get("simulation_id") or run.get("run_id") or path.stem,
        "run_id": run.get("run_id"), "organism": run.get("organism"),
        "script": run.get("script"), "source_run_file": str(path),
        "image_fitness": run.get("image_fitness"), "source_authors": [],
        "claim": f"Simulation {run.get('simulation_id')} faithfully reproduces the cited "
                 f"source model for {run.get('organism')} (image_fitness={run.get('image_fitness')}).",
    }


def finding_from_audit(path: Path) -> dict[str, Any]:
    doc = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    sim = (doc.get("simulations") or [{}])[0]
    return {
        "finding_id": sim.get("id") or doc.get("audit_id") or path.stem,
        "name": sim.get("name"),
        "audit_id": doc.get("audit_id"), "organism": sim.get("organism"),
        "script": sim.get("script"), "source_audit_file": str(path),
        "overall_score": sim.get("overall_score"), "source_authors": [],
        "claim": f"Simulation {sim.get('id')} faithfully reproduces the cited source model "
                 f"for {sim.get('organism')} (MorphoEval overall={sim.get('overall_score')}).",
    }


def new_request(finding: dict[str, Any], cfg: dict | None = None) -> dict[str, Any]:
    cfg = cfg or load_config()
    sla = cfg.get("sla", {}) or {}
    created = now_iso()
    route = route_expertise(cfg, str(finding.get("organism") or ""),
                            str(finding.get("finding_id") or ""), str(finding.get("claim") or ""))
    # Expert search must be finding-specific (organism + law/mechanism), not the
    # generic route query: a broad "Drosophila morphogenesis" search returns the
    # most-cited authors in the field, not the people who can judge THIS law.
    # The generic route stays as fallback expertise label only.
    organism = str(finding.get("organism") or "")
    law = str(finding.get("math_family") or "")
    if not law and finding.get("name") and organism and organism in str(finding["name"]):
        law = str(finding["name"]).split(organism, 1)[1].strip(" ()0123456789µm")
    specific = f"{organism} {law}".strip()
    if specific:
        route = {**route, "query": f"{specific} morphogenesis",
                 # OpenAlex search is near-AND: an over-specific query returns 0.
                 # Cascade from the exact law to the organism's field, merging pools.
                 "query_cascade": list(dict.fromkeys(
                     [f"{specific} morphogenesis", f"{organism} morphogenesis", route["query"]]))}
    fid = str(finding["finding_id"])
    req = {
        "schema_version": SCHEMA, "validation_id": f"{fid}-human-validation",
        "status": "queued", "finding": {**finding, **route},
        "quorum": int(cfg.get("quorum", 2)), "review_questions": cfg.get("review_questions", []),
        "sla": {**sla, "queued_at": created,
                "deadline_at": (parse_ts(created) + timedelta(days=float(sla.get("review_days", 21)))).strftime("%Y-%m-%dT%H:%M:%SZ")},
        "created_at": created, "experts": [], "events": [
            {"event": "queued", "at": created, "actor": "human_validation.py", "notes": "validation requested"}],
        "contact_policy": "record_only_never_auto_contact: a human sends invitations; this tool only logs them",
    }
    refresh(req)
    return req


# --------------------------------------------------------------------------- derived state + timing
def refresh(req: dict[str, Any], at: str | None = None) -> dict[str, Any]:
    """Recompute status, consensus and all durations from the append-only events."""
    ev = sorted(req.get("events", []), key=lambda e: e.get("at") or "")
    queued = req.get("sla", {}).get("queued_at") or req.get("created_at")
    # Deadline is always anchored to queued_at (never to file-creation time),
    # so backfilled/imported requests get a correct SLA.
    if queued and req.get("sla", {}).get("review_days") is not None:
        req["sla"]["deadline_at"] = (parse_ts(queued) + timedelta(
            days=float(req["sla"]["review_days"]))).strftime("%Y-%m-%dT%H:%M:%SZ")
    first_invite = next((e["at"] for e in ev if e.get("event") == "invited"), None)
    # Per-expert rollup from events
    for ex in req.get("experts", []):
        mine = [e for e in ev if e.get("expert_id") == ex.get("expert_id")]
        kinds = [e["event"] for e in mine]
        ex["invited_at"] = next((e["at"] for e in mine if e["event"] == "invited"), ex.get("invited_at"))
        ex["responded_at"] = next((e["at"] for e in mine if e["event"] in ("accepted", "declined", "no_response")), ex.get("responded_at"))
        ex["verdict_at"] = next((e["at"] for e in mine if e["event"] == "verdict"), ex.get("verdict_at"))
        vd = next((e for e in mine if e["event"] == "verdict"), None)
        if vd:
            ex["verdict"] = vd.get("verdict")
        ex["assignment_status"] = (
            "verdict_recorded" if "verdict" in kinds else "in_review" if "review_started" in kinds
            else "accepted" if "accepted" in kinds else "declined" if "declined" in kinds
            else "no_response" if "no_response" in kinds else "invited" if "invited" in kinds
            else ex.get("assignment_status", "candidate"))
    first_response = min([e["at"] for e in ev if e.get("event") in ("accepted", "declined")], default=None)
    first_accept = min([e["at"] for e in ev if e.get("event") == "accepted"], default=None)
    closed = next((e for e in ev if e.get("event") == "closed_without_validation"), None)

    verdicts = [e for e in ev if e.get("event") == "verdict" and e.get("verdict") in VERDICTS]
    eligible = {e.get("expert_id") for e in req.get("experts", []) if not (e.get("coi") or {}).get("ineligible")}
    counted = [e for e in verdicts if e.get("expert_id") in eligible]
    quorum = int(req.get("quorum", 2))
    consensus, consensus_at = "pending", None
    if len(counted) >= quorum:
        vals = [e["verdict"] for e in counted]
        if all(v == "validated" for v in vals): consensus = "validated"
        elif all(v == "refuted" for v in vals): consensus = "refuted"
        elif all(v == "unable_to_assess" for v in vals): consensus = "unable_to_assess"
        elif "refuted" in vals and ("validated" in vals or "partially_validated" in vals): consensus = "mixed_no_consensus"
        else: consensus = "partially_validated"
        if consensus != "mixed_no_consensus":
            consensus_at = sorted(e["at"] for e in counted)[quorum - 1]

    if consensus_at: req["status"] = f"verdict:{consensus}"
    elif closed: req["status"] = f"closed_without_validation:{closed.get('outcome')}"
    elif any(e.get("assignment_status") == "in_review" for e in req.get("experts", [])): req["status"] = "in_review"
    elif first_accept: req["status"] = "accepted"
    elif first_invite: req["status"] = "invited"
    elif req.get("experts"): req["status"] = "expert_search_done"
    else: req["status"] = "queued"

    moment = at or now_iso()
    t_verdict = hours_between(queued, consensus_at)
    t_nonval = hours_between(queued, closed.get("at") if closed else None)
    terminal = [t for t in (t_verdict, t_nonval) if t is not None]
    req["timing"] = {
        "queued_at": queued, "first_invited_at": first_invite, "first_response_at": first_response,
        "first_accepted_at": first_accept, "verdict_at": consensus_at,
        "closed_without_validation_at": closed.get("at") if closed else None,
        "time_to_first_invite_hours": hours_between(queued, first_invite),
        "time_to_first_response_hours": hours_between(first_invite, first_response),
        "time_in_review_hours": hours_between(first_accept, consensus_at),
        "time_to_verdict_hours": t_verdict, "time_to_non_validation_hours": t_nonval,
        "time_to_validate_or_not_hours": min(terminal) if terminal else None,
        "outcome": consensus if consensus_at else (closed.get("outcome") if closed else "pending"),
        "elapsed_open_hours": None if terminal else hours_between(queued, moment),
        "is_overdue": (not terminal) and bool(req.get("sla", {}).get("deadline_at")) and moment > req["sla"]["deadline_at"],
        "sla_deadline_at": req.get("sla", {}).get("deadline_at"),
    }
    req["verdict_summary"] = {"quorum": quorum, "verdicts_counted": len(counted),
                              "by_verdict": {v: [e["verdict"] for e in counted].count(v) for v in VERDICTS},
                              "consensus": consensus if consensus_at else None}
    return req


# --------------------------------------------------------------------------- expert discovery
def _http_get_json(url: str, timeout: float = 15.0) -> dict[str, Any]:
    req = urllib.request.Request(url, headers={
        "User-Agent": "ProjectDelphi-MorphoEval-HumanValidation/1.0", "Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8", errors="replace"))


def search_openalex_authors(query: str, limit: int = 25) -> list[dict[str, Any]]:
    """Candidate pool = authors of works whose TITLE/ABSTRACT matches `query`.

    General `search=` matches authors/institutions/venues too and returns the
    most-cited people in a broad field (wrong experts — the fastest way to burn
    scientist attention). `title_and_abstract.search:` keeps the pool on-topic.
    """
    url = ("https://api.openalex.org/works?filter="
           f"{urllib.parse.quote(f'title_and_abstract.search:{query}')}"
           f"&per-page={limit}&sort=cited_by_count:desc")
    data = _http_get_json(url)
    pool: dict[str, dict[str, Any]] = {}
    for work in data.get("results", []):
        for a in work.get("authorships") or []:
            au = a.get("author") or {}
            aid = au.get("id") or au.get("display_name")
            if not aid:
                continue
            e = pool.setdefault(aid, {"expert_id": "oa-" + str(aid).rstrip("/").split("/")[-1],
                                      "name": au.get("display_name"), "openalex_author_id": au.get("id"),
                                      "orcid": au.get("orcid"), "institutions": [], "topics": [],
                                      "works_in_area": 0, "cited_by_count": 0, "last_active_year": 0,
                                      "example_works": [], "contact_route": None})
            e["works_in_area"] += 1
            e["cited_by_count"] += int(work.get("cited_by_count") or 0)
            e["last_active_year"] = max(e["last_active_year"], int(work.get("publication_year") or 0))
            for inst in a.get("institutions") or []:
                if inst.get("display_name") and inst["display_name"] not in e["institutions"]:
                    e["institutions"].append(inst["display_name"])
            if work.get("display_name") and len(e["example_works"]) < 3:
                e["example_works"].append({"title": work["display_name"], "year": work.get("publication_year"),
                                           "doi": (work.get("doi") or "").replace("https://doi.org/", "")})
    return list(pool.values())


def rank_candidates(cands: list[dict[str, Any]], finding: dict[str, Any],
                    cfg: dict | None = None, current_year: int | None = None) -> list[dict[str, Any]]:
    """Score, COI-flag and sort candidates. Ranking only — humans shortlist."""
    cfg = cfg or load_config()
    coi_cfg, elig = cfg.get("conflict_of_interest", {}) or {}, cfg.get("eligibility", {}) or {}
    year = current_year or datetime.now(timezone.utc).year
    src_authors = {str(a).lower() for a in finding.get("source_authors") or []}
    src_insts = {str(i).lower() for i in finding.get("source_institutions") or []}
    out = []
    for c in cands:
        recency = max(0.0, 5.0 - 0.5 * max(0, year - int(c.get("last_active_year") or 0)))
        c = {**c, "relevance_score": round(3 * int(c.get("works_in_area") or 0)
                                           + math.log10(1 + int(c.get("cited_by_count") or 0)) + recency, 3)}
        is_author = bool(coi_cfg.get("exclude_source_authors")) and str(c.get("name") or "").lower() in src_authors
        same_inst = bool(coi_cfg.get("exclude_same_institution")) and bool(
            src_insts & {str(i).lower() for i in c.get("institutions") or []})
        too_few = int(c.get("works_in_area") or 0) < int(elig.get("min_works_in_area", 1))
        stale = bool(elig.get("active_within_years")) and (year - int(c.get("last_active_year") or 0)) > int(elig["active_within_years"])
        c["coi"] = {"is_source_author": is_author, "same_institution_as_source": same_inst,
                    "ineligible": bool(is_author or same_inst),
                    "notes": "COI-excluded" if (is_author or same_inst) else "no COI detected from OpenAlex metadata; declaration still required on acceptance"}
        c["eligibility"] = {"eligible": not (too_few or stale or c["coi"]["ineligible"]),
                            "reasons": [r for r, bad in (("below min_works_in_area", too_few), ("inactive", stale), ("coi", c["coi"]["ineligible"])) if bad]}
        c.setdefault("assignment_status", "candidate")
        out.append(c)
    return sorted(out, key=lambda e: (e["eligibility"]["eligible"], e["relevance_score"]), reverse=True)


# --------------------------------------------------------------------------- io
def path_for(finding_id: str, directory: Path = VALIDATIONS_DIR) -> Path:
    slug = "".join(ch if ch.isalnum() or ch in "-_" else "-" for ch in finding_id)
    return directory / f"{slug}-human-validation.json"


def save(req: dict[str, Any], path: Path) -> Path:
    refresh(req)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(req, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def record_event(req: dict, event: str, expert_id: str | None = None, at: str | None = None, **kw: Any) -> dict:
    if event not in EVENTS:
        raise ValueError(f"unknown event {event}; choose from {EVENTS}")
    if event == "verdict" and kw.get("verdict") not in VERDICTS:
        raise ValueError(f"--verdict required, one of {VERDICTS}")
    if expert_id and not any(e.get("expert_id") == expert_id for e in req.get("experts", [])):
        raise KeyError(f"expert {expert_id} not in candidate pool — run find-experts/assign first")
    req.setdefault("events", []).append({"event": event, "at": at or now_iso(),
                                         "expert_id": expert_id, **{k: v for k, v in kw.items() if v is not None}})
    return refresh(req, at=at)


# --------------------------------------------------------------------------- report
def aggregate(directory: Path = VALIDATIONS_DIR) -> dict[str, Any]:
    rows = []
    for p in sorted(directory.glob("*-human-validation.json")) if directory.exists() else []:
        try: req = refresh(load(p))
        except Exception: continue
        t = req["timing"]
        rows.append({"validation_id": req["validation_id"], "finding_id": req["finding"].get("finding_id"),
                     "organism": req["finding"].get("organism"), "expertise": req["finding"].get("expertise"),
                     "status": req["status"], "outcome": t["outcome"], "experts_found": len(req.get("experts", [])),
                     "experts_invited": sum(1 for e in req.get("experts", []) if e.get("invited_at")),
                     "verdicts": req["verdict_summary"]["verdicts_counted"],
                     "time_to_first_response_hours": t["time_to_first_response_hours"],
                     "time_to_verdict_hours": t["time_to_verdict_hours"],
                     "time_to_validate_or_not_hours": t["time_to_validate_or_not_hours"],
                     "is_overdue": t["is_overdue"], "file": str(p)})
    done = [r["time_to_validate_or_not_hours"] for r in rows if r["time_to_validate_or_not_hours"] is not None]
    verdict_times = [r["time_to_verdict_hours"] for r in rows if r["time_to_verdict_hours"] is not None]

    def stats(xs: list[float]) -> dict[str, Any]:
        return {"n": len(xs), "median_hours": round(statistics.median(xs), 2) if xs else None,
                "mean_hours": round(statistics.mean(xs), 2) if xs else None,
                "max_hours": max(xs) if xs else None}
    return {"generated_at": now_iso(), "n_requests": len(rows),
            "by_outcome": {o: sum(1 for r in rows if r["outcome"] == o) for o in (*VERDICTS, *NON_VALIDATION, "pending")},
            "n_overdue_open": sum(1 for r in rows if r["is_overdue"]),
            "time_to_validate_or_not": stats(done), "time_to_verdict": stats(verdict_times), "requests": rows}


def build_prompt(req: dict[str, Any]) -> str:
    f = req["finding"]
    return f"""You are running Step 13 (Tier C — Human Expert Validation) for one MorphoEval finding.

## FINDING     : {f.get('finding_id')} — {f.get('organism')}
## CLAIM       : {f.get('claim')}
## EXPERTISE   : {f.get('expertise')}   SEARCH: `{f.get('query')}`
## STATE FILE  : morphoeval/validations/{req['validation_id']}.json  (ONLY file you may create/edit)

1. Run: python3 morphoeval/human_validation.py find-experts --validation <state-file>
   (OpenAlex authors of the wider literature; source authors are COI-flagged ineligible automatically.)
2. Shortlist >= {req.get('quorum', 2)} eligible, COI-free experts. Fill each `contact_route`
   from a public institution/ORCID page. NEVER invent an email; NEVER contact anyone.
3. Write a one-page invitation draft to morphoeval/validations/{req['validation_id']}-invitation.md
   containing the claim, links to the script/figures, the review questions in the state file, the SLA, and a COI-declaration line.
4. STOP. A human sends the invitations and logs each step with `record --event invited|accepted|declined|verdict ...`.
   Your job ends at a ready-to-send shortlist — time-to-validate is measured from here by the state file.
Verify before committing: `git diff --name-only --diff-filter=MD origin/main` shows ZERO modified files."""


# --------------------------------------------------------------------------- CLI
def _req(args) -> tuple[dict, Path]:
    p = Path(args.validation)
    return load(p), p


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true")
    sub = ap.add_subparsers(dest="cmd")
    p = sub.add_parser("init"); p.add_argument("--run"); p.add_argument("--audit")
    p.add_argument("--finding-id"); p.add_argument("--organism"); p.add_argument("--claim")
    p.add_argument("--math-family"); p.add_argument("--source-author", action="append", default=[])
    p.add_argument("--out"); p.add_argument("--force", action="store_true")
    p = sub.add_parser("find-experts"); p.add_argument("--validation", required=True)
    p.add_argument("--limit", type=int, default=25); p.add_argument("--keep", type=int, default=50, help="store only the top N ranked candidates")
    p.add_argument("--no-network", action="store_true")
    p.add_argument("--candidates-file", help="offline candidate pool JSON (list of OpenAlex-like author dicts)")
    p = sub.add_parser("assign"); p.add_argument("--validation", required=True); p.add_argument("--expert", required=True)
    p = sub.add_parser("record"); p.add_argument("--validation", required=True)
    p.add_argument("--event", required=True, choices=EVENTS); p.add_argument("--expert")
    p.add_argument("--verdict", choices=VERDICTS); p.add_argument("--channel"); p.add_argument("--notes"); p.add_argument("--at")
    p = sub.add_parser("close"); p.add_argument("--validation", required=True)
    p.add_argument("--outcome", required=True, choices=NON_VALIDATION); p.add_argument("--notes"); p.add_argument("--at")
    p = sub.add_parser("status"); p.add_argument("--validation", required=True)
    p = sub.add_parser("prompt"); p.add_argument("--validation", required=True)
    p = sub.add_parser("report"); p.add_argument("--dir", default=str(VALIDATIONS_DIR))
    p.add_argument("--md"); p.add_argument("--csv")
    args = ap.parse_args(argv)

    if args.self_test:
        return self_test()
    cfg = load_config()

    if args.cmd == "init":
        if args.run: finding = finding_from_run(Path(args.run))
        elif args.audit: finding = finding_from_audit(Path(args.audit))
        elif args.finding_id: finding = {"finding_id": args.finding_id, "organism": args.organism,
                                         "claim": args.claim or "", "source_authors": args.source_author}
        else: ap.error("init needs --run, --audit, or --finding-id")
        if args.math_family: finding["math_family"] = args.math_family
        if args.source_author: finding["source_authors"] = args.source_author
        req = new_request(finding, cfg)
        out = Path(args.out) if args.out else path_for(req["finding"]["finding_id"])
        if out.exists() and not args.force:
            print(f"refusing to overwrite {out} (pass --force)", file=sys.stderr); return 1
        save(req, out)
        print(f"[human_validation] queued {req['validation_id']} -> {out} (deadline {req['sla']['deadline_at']})")
        return 0

    if args.cmd == "find-experts":
        req, path = _req(args)
        if args.candidates_file:
            cands = json.loads(Path(args.candidates_file).read_text(encoding="utf-8"))
        elif args.no_network:
            cands = []
        else:
            try:
                cands, seen = [], set()
                for q in req["finding"].get("query_cascade") or [req["finding"]["query"]]:
                    for c in search_openalex_authors(q, args.limit):
                        if c["expert_id"] not in seen:
                            seen.add(c["expert_id"]); cands.append(c)
                    if len(cands) >= 10:  # specific-enough pool found; stop broadening
                        break
            except Exception as exc:
                print(f"OpenAlex search failed ({exc}); state unchanged", file=sys.stderr); return 1
        ranked = rank_candidates(cands, req["finding"], cfg)
        known = {e["expert_id"] for e in req["experts"]}
        req["experts"] += [c for c in ranked if c["expert_id"] not in known][: max(args.keep - len(req["experts"]), 0)]
        if not ranked:
            req["events"].append({"event": "note", "at": now_iso(), "notes": "expert search returned no candidates"})
        save(req, path)
        elig = sum(1 for c in ranked if c["eligibility"]["eligible"])
        print(f"[human_validation] {len(ranked)} candidate(s), {elig} eligible/COI-free -> {path}")
        for c in ranked[:10]:
            print(f"  {c['relevance_score']:>6}  {'ELIGIBLE' if c['eligibility']['eligible'] else 'excluded:'+','.join(c['eligibility']['reasons']):28s} {c.get('name')} ({', '.join(c.get('institutions') or [])[:48]})")
        return 0

    if args.cmd == "assign":
        req, path = _req(args)
        ex = next((e for e in req["experts"] if e["expert_id"] == args.expert), None)
        if not ex: print(f"expert {args.expert} not in pool", file=sys.stderr); return 1
        if ex["coi"]["ineligible"]: print(f"expert {args.expert} is COI-ineligible", file=sys.stderr); return 1
        ex["assignment_status"] = "assigned"; save(req, path)
        print(f"[human_validation] assigned {args.expert} ({ex.get('name')})"); return 0

    if args.cmd == "record":
        req, path = _req(args)
        try: record_event(req, args.event, args.expert, at=args.at, verdict=args.verdict,
                           channel=args.channel, notes=args.notes)
        except (ValueError, KeyError) as exc: print(str(exc), file=sys.stderr); return 1
        save(req, path); print(f"[human_validation] {args.event} recorded; status={req['status']} timing={req['timing']}"); return 0

    if args.cmd == "close":
        req, path = _req(args)
        req["events"].append({"event": "closed_without_validation", "at": args.at or now_iso(),
                              "outcome": args.outcome, "notes": args.notes})
        save(req, path); print(f"[human_validation] closed without validation: {args.outcome}; time_to_validate_or_not_hours={req['timing']['time_to_validate_or_not_hours']}"); return 0

    if args.cmd == "status":
        req, _ = _req(args); refresh(req)
        print(json.dumps({"validation_id": req["validation_id"], "status": req["status"],
                          "verdict_summary": req["verdict_summary"], "timing": req["timing"],
                          "experts": [{k: e.get(k) for k in ("expert_id", "name", "assignment_status", "verdict")} for e in req["experts"]]}, indent=2))
        return 0

    if args.cmd == "prompt":
        req, _ = _req(args); print(build_prompt(req)); return 0

    if args.cmd == "report":
        rep = aggregate(Path(args.dir))
        print(f"=== Human Expert Validation — {rep['n_requests']} request(s), {rep['n_overdue_open']} overdue/open ===")
        print(f"  outcomes: {rep['by_outcome']}")
        print(f"  time_to_validate_or_not_hours: median={rep['time_to_validate_or_not']['median_hours']} mean={rep['time_to_validate_or_not']['mean_hours']} n={rep['time_to_validate_or_not']['n']}")
        for r in rep["requests"]:
            print(f"  {r['finding_id'][:52]:52s} {str(r['outcome']):18s} t_v_or_not={r['time_to_validate_or_not_hours']} overdue={r['is_overdue']}")
        if args.md:
            lines = ["# Human Expert Validation report", f"\nGenerated {rep['generated_at']} — {rep['n_requests']} request(s)\n",
                     "| Finding | Outcome | Experts invited | Verdicts | Time to validate-or-not (h) | Overdue |", "|---|---|---|---|---|---|"]
            lines += [f"| {r['finding_id']} | {r['outcome']} | {r['experts_invited']} | {r['verdicts']} | {r['time_to_validate_or_not_hours']} | {r['is_overdue']} |" for r in rep["requests"]]
            Path(args.md).write_text("\n".join(lines) + "\n", encoding="utf-8")
        if args.csv:
            buf = io.StringIO(); w = csv.DictWriter(buf, fieldnames=list(rep["requests"][0].keys()) if rep["requests"] else ["validation_id"])
            w.writeheader(); w.writerows(rep["requests"]); Path(args.csv).write_text(buf.getvalue(), encoding="utf-8")
        return 0

    ap.print_help(); return 0


def self_test() -> int:
    cfg = load_config()
    req = new_request({"finding_id": "selftest-finding", "organism": "Neurospora crassa",
                       "claim": "x", "source_authors": ["Ada Source"]}, cfg)
    req["sla"]["queued_at"] = req["created_at"] = "2026-10-01T00:00:00Z"; refresh(req)
    pool = [{"expert_id": "oa-1", "name": "Ada Source", "works_in_area": 9, "cited_by_count": 900, "last_active_year": 2026, "institutions": ["MIT"]},
            {"expert_id": "oa-2", "name": "Grace Expert", "works_in_area": 5, "cited_by_count": 200, "last_active_year": 2025, "institutions": ["Oxford"]},
            {"expert_id": "oa-3", "name": "Alan Expert", "works_in_area": 4, "cited_by_count": 100, "last_active_year": 2024, "institutions": ["Cambridge"]}]
    req["experts"] = rank_candidates(pool, req["finding"], cfg, current_year=2026)
    assert req["experts"][0]["name"] == "Grace Expert" and req["experts"][-1]["coi"]["ineligible"]
    record_event(req, "invited", "oa-2", at="2026-10-02T00:00:00Z")
    record_event(req, "invited", "oa-3", at="2026-10-02T00:00:00Z")
    record_event(req, "accepted", "oa-2", at="2026-10-04T00:00:00Z")
    record_event(req, "accepted", "oa-3", at="2026-10-04T00:00:00Z")
    record_event(req, "verdict", "oa-2", at="2026-10-11T00:00:00Z", verdict="validated")
    assert req["timing"]["time_to_validate_or_not_hours"] is None  # quorum=2 not yet met
    record_event(req, "verdict", "oa-3", at="2026-10-13T00:00:00Z", verdict="validated")
    assert req["status"] == "verdict:validated", req["status"]
    assert req["timing"]["time_to_verdict_hours"] == 12 * 24, req["timing"]
    assert req["timing"]["time_to_first_response_hours"] == 48.0
    # Non-validation path also yields a duration
    req2 = new_request({"finding_id": "selftest-2", "organism": "Danio rerio", "claim": "y"}, cfg)
    req2["events"].append({"event": "closed_without_validation", "at": "2026-11-01T00:00:00Z", "outcome": "expired"})
    refresh(req2, at="2026-11-02T00:00:00Z")
    assert req2["timing"]["time_to_validate_or_not_hours"] is not None and req2["timing"]["outcome"] == "expired"
    print("[human_validation] self-test PASSED (ranking, COI exclusion, quorum, validate & non-validation timing)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

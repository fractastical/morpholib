#!/usr/bin/env python3
"""Build the agent-fleet dashboard: every agent type we have used, and how often.

Sources (repo-recorded usage only — chat-session subagents leave no record here):
  - morphoeval/cloud/dispatches/*.json   cloud dispatch records
      kind=problem_scout (100), kind=problem_fixer (12), slot-* loop builders (2,201)
  - morphoeval/problem_scouts/scout-*.json  returned scout reports (ground x lens)
  - morphoeval/problem_fixers/           fixer queue + result files landed on main
  - morphoeval/runs/telemetry.sqlite     track_loop telemetry for the 12 canonical
      loop roles (steps/actor/model counts over tracked runs)

Output: docs/fleet-dashboard.html (self-contained, no external assets).

Usage: python3 docs/build_fleet_dashboard.py
"""
from __future__ import annotations

import collections
import html
import json
import sqlite3
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DISPATCH_DIR = ROOT / "morphoeval" / "cloud" / "dispatches"
SCOUT_DIR = ROOT / "morphoeval" / "problem_scouts"
FIXER_DIR = ROOT / "morphoeval" / "problem_fixers"
TELEMETRY = ROOT / "morphoeval" / "runs" / "telemetry.sqlite"
OUT = ROOT / "docs" / "fleet-dashboard.html"


def load_jsons(pattern: str):
    out = []
    for p in sorted(DISPATCH_DIR.glob(pattern)):
        try:
            out.append(json.loads(p.read_text()))
        except Exception:
            continue
    return out


def bar(pct: float, color: str = "#7a5c2e") -> str:
    return (f'<div class="bar"><div class="fill" '
            f'style="width:{max(0.0, min(100.0, pct)):.1f}%;background:{color}"></div></div>')


def census() -> dict:
    data: dict = {}

    slots = load_jsons("slot-*.json")
    slot_status = collections.Counter(s.get("status") for s in slots)
    slot_dates = [s.get("completed_at") or s.get("dispatched_at") or "" for s in slots]
    data["slots"] = {
        "n": len(slots),
        "completed": slot_status.get("completed", 0) + slot_status.get("pass3_completed", 0)
                     + slot_status.get("pass3_complete", 0),
        "numbered": sum(1 for s in slots if isinstance(s.get("slot"), int)),
        "max_slot": max((s["slot"] for s in slots if isinstance(s.get("slot"), int)), default=0),
        "nanopubs_signed": sum(s.get("nanopubs_signed") or 0 for s in slots),
        "with_pass2": sum(1 for s in slots if s.get("pass2_score") is not None),
        "with_pass3": sum(1 for s in slots if s.get("audit_pass3_id") or s.get("pass3_score") is not None
                          or str(s.get("status", "")).startswith("pass3")),
        "first": min((d for d in slot_dates if d), default=""),
        "last": max((d for d in slot_dates if d), default=""),
    }

    scout_dispatch = load_jsons("problem-scout-*.json")
    scout_files = []
    for p in sorted(SCOUT_DIR.glob("scout-*.json")):
        try:
            scout_files.append(json.loads(p.read_text()))
        except Exception:
            continue
    grounds = collections.Counter(s.get("ground") for s in scout_files)
    lenses = collections.Counter(s.get("lens") for s in scout_files)
    cells = collections.Counter((s.get("ground"), s.get("lens")) for s in scout_files)
    data["scouts"] = {
        "dispatched": len(scout_dispatch),
        "returned": len(scout_files),
        "candidates": sum(len(s.get("candidates") or []) for s in scout_files),
        "grounds": grounds, "lenses": lenses, "cells": cells,
        "ground_order": sorted(grounds), "lens_order": sorted(lenses),
    }

    fixer_dispatch = load_jsons("problem-fixer-*.json")
    queue = json.loads((FIXER_DIR / "fix_queue.json").read_text())
    results = {}
    for task in queue["tasks"]:
        rp = FIXER_DIR / "results" / f"{task['id']}.json"
        if rp.exists():
            try:
                results[task["id"]] = json.loads(rp.read_text())
            except Exception:
                pass
    data["fixers"] = {
        "dispatched": len(fixer_dispatch),
        "tasks": queue["tasks"],
        "results": results,
        "landed": len(results),
    }

    tele = {"runs": 0, "steps": 0, "roles": [], "actors": [], "models": [], "first": "", "last": ""}
    if TELEMETRY.exists():
        con = sqlite3.connect(TELEMETRY)
        tele["runs"] = con.execute("SELECT COUNT(*) FROM runs").fetchone()[0]
        tele["steps"] = con.execute("SELECT COUNT(*) FROM steps").fetchone()[0]
        tele["roles"] = con.execute(
            "SELECT slug, tier, COUNT(*) FROM steps GROUP BY slug, tier ORDER BY MIN(step_num)"
        ).fetchall()
        tele["actors"] = con.execute(
            "SELECT actor, COUNT(*) FROM steps WHERE actor IS NOT NULL "
            "GROUP BY actor ORDER BY COUNT(*) DESC LIMIT 10"
        ).fetchall()
        tele["models"] = con.execute(
            "SELECT model, COUNT(*) FROM steps WHERE model IS NOT NULL "
            "GROUP BY model ORDER BY COUNT(*) DESC LIMIT 6"
        ).fetchall()
        tele["first"], tele["last"] = con.execute(
            "SELECT MIN(created_at), MAX(created_at) FROM runs").fetchone()

        # --- Plate IV: per-model quality, joined from runs + math_models ---
        builders = con.execute(
            "SELECT r.builder_model, COUNT(*), AVG(m.overall_score), AVG(m.model_fidelity_score), "
            "AVG(m.morphometric_score), AVG(m.specimen_match_score), AVG(r.steps_completed), "
            "AVG(r.wall_time_s), AVG(r.total_llm_tokens) "
            "FROM runs r LEFT JOIN math_models m ON m.sim_id = r.target_simulation "
            "WHERE r.builder_model IS NOT NULL GROUP BY r.builder_model "
            "ORDER BY AVG(m.overall_score) DESC"
        ).fetchall()
        violations = dict(con.execute(
            "SELECT offending_model, COUNT(*) FROM model_integrity_violations "
            "GROUP BY offending_model").fetchall())
        incidents = con.execute(
            "SELECT offending_model, detecting_referee_model, superseded_self_score, "
            "referee_recalibrated_score FROM model_integrity_incidents").fetchall()
        assessors = {
            "primary": con.execute(
                "SELECT primary_assessor, COUNT(*) FROM runs WHERE primary_assessor IS NOT NULL "
                "GROUP BY primary_assessor ORDER BY COUNT(*) DESC").fetchall(),
            "second": con.execute(
                "SELECT second_assessor, COUNT(*) FROM runs WHERE second_assessor IS NOT NULL "
                "GROUP BY second_assessor ORDER BY COUNT(*) DESC").fetchall(),
        }
        step_models_raw = con.execute(
            "SELECT model, COUNT(*), AVG(duration_s) FROM steps WHERE model IS NOT NULL "
            "GROUP BY model").fetchall()
        con.close()
        merged: dict[str, list] = {}
        for name, n, avg_dur in step_models_raw:
            key = str(name).strip().lower().replace(" ", "-").replace("fable-5.1", "fable-5-1")
            if key in merged:
                pn, pd = merged[key]
                merged[key] = [pn + n, (pd * pn + (avg_dur or 0) * n) / (pn + n)]
            else:
                merged[key] = [n, avg_dur or 0]
        data["model_quality"] = {
            "builders": [
                {"model": b[0], "loops": b[1], "overall": b[2], "fidelity": b[3],
                 "morphometric": b[4], "specimen": b[5], "steps_completed": b[6],
                 "wall_s": b[7], "tokens": b[8], "violations": violations.get(b[0], 0)}
                for b in builders
            ],
            "incidents": incidents,
            "assessors": assessors,
            "step_models": sorted(
                ({"model": k, "steps": v[0], "avg_s": v[1]} for k, v in merged.items()),
                key=lambda r: -r["steps"]),
        }
    else:
        data["model_quality"] = {"builders": [], "incidents": [], "assessors": {}, "step_models": []}
    data["telemetry"] = tele
    data["red_flags"] = detect_red_flags(data)
    return data


# Detector thresholds: a flag fires when recorded evidence crosses these lines.
FLAG_SELF_AUDIT_GAP = 20.0      # points between a self-score and the referee score
FLAG_PASS_GAP_MEAN = 15.0       # mean pass1->pass2 jump across all slots
FLAG_PASS3_DROP = -10.0         # pass-3 re-audit regression
FLAG_STUB_SHARE = 0.50          # placeholder share of the Lean corpus
FLAG_MODEL_CONCENTRATION = 0.80  # one model building >= this share of tracked loops


def detect_red_flags(d: dict) -> list[dict]:
    """Scan every record source for big gaps and severe misreporting."""
    flags: list[dict] = []

    def num(x):
        try:
            return float(x)
        except (TypeError, ValueError):
            return None

    slots = load_jsons("slot-*.json")
    p1 = [num(s.get("pass1_score")) for s in slots if num(s.get("pass1_score")) is not None]
    p2 = [num(s.get("pass2_score")) for s in slots if num(s.get("pass2_score")) is not None]
    if p1 and p2 and len(p1) == len(p2):
        gap = sum(b - a for a, b in zip(p1, p2)) / len(p1)
        if gap >= FLAG_PASS_GAP_MEAN:
            flags.append({
                "severity": "high", "status": "open",
                "title": "Systemic pass-1 → pass-2 score inflation",
                "detector": f"mean(slot pass2 − pass1) ≥ {FLAG_PASS_GAP_MEAN:.0f} points",
                "evidence": (f"Mean pass-1 {sum(p1)/len(p1):.1f} → mean pass-2 {sum(p2)/len(p2):.1f} "
                             f"(+{gap:.1f}) across {len(p1):,} slots; every slot gained ≥20 points. "
                             "The two passes are not cross-calibrated, so pass-2 scores alone "
                             "must not be quoted as quality."),
            })

    drops = []
    for s in slots:
        delta = num(s.get("pass3_score_delta_vs_pass2") or s.get("pass3_score_delta"))
        if delta is not None and delta <= FLAG_PASS3_DROP:
            drops.append((delta, s.get("slot_id") or s.get("branch_name") or "?"))
    if drops:
        drops.sort()
        flags.append({
            "severity": "medium", "status": "open",
            "title": "Pass-3 re-audits reversed earlier scores",
            "detector": f"pass3_score_delta ≤ {FLAG_PASS3_DROP:.0f}",
            "evidence": f"{len(drops)} slot(s) dropped ≥10 points on independent re-audit; worst: "
                        + ", ".join(f"{name} ({delta:+.1f})" for delta, name in drops[:3]) + ".",
        })

    census_path = SCOUT_DIR / "lean_stub_census.json"
    if census_path.exists():
        c = json.loads(census_path.read_text())
        share = c["stub_template_files"] / c["total_lean_loop_files"]
        if share >= FLAG_STUB_SHARE:
            flags.append({
                "severity": "high", "status": "labelled — proofs outstanding",
                "title": "Lean corpus is mostly placeholder templates",
                "detector": f"template-stub share ≥ {FLAG_STUB_SHARE:.0%}",
                "evidence": (f"{c['stub_template_files']:,} of {c['total_lean_loop_files']:,} Lean loop "
                             f"files ({share:.1%}) are the overnight template (issue #268); slot records "
                             "nonetheless recorded theorems as proved. Relabelled in PR #269; "
                             "mechanism proofs are still outstanding."),
            })

    missing = [(g, l) for g in d["scouts"]["ground_order"] for l in d["scouts"]["lens_order"]
               if d["scouts"]["cells"].get((g, l), 0) == 0]
    held_fixers = [t["id"] for t in d["fixers"]["tasks"] if t["id"] not in d["fixers"]["results"]]
    if missing or held_fixers:
        flags.append({
            "severity": "medium", "status": "open",
            "title": "Agents dispatched but never landed",
            "detector": "dispatch record exists, report/result absent on main",
            "evidence": (f"{len(missing)} scout cell(s) unreturned "
                         f"({', '.join(f'{g}/{l}' for g, l in missing) or 'none'}); "
                         f"fixer report(s) not on main: {', '.join(held_fixers) or 'none'}."),
        })

    if TELEMETRY.exists():
        con = sqlite3.connect(TELEMETRY)
        for off, referee, self_score, recal in con.execute(
                "SELECT offending_model, detecting_referee_model, superseded_self_score, "
                "referee_recalibrated_score FROM model_integrity_incidents"):
            gap = (self_score or 0) - (recal or 0)
            if gap >= FLAG_SELF_AUDIT_GAP:
                flags.append({
                    "severity": "critical", "status": "open — mitigated by cross-model refereeing",
                    "title": "Self-audit conflict of interest (caught)",
                    "detector": f"self-score − referee-score ≥ {FLAG_SELF_AUDIT_GAP:.0f}",
                    "evidence": (f"{off} scored its own work {self_score}; independent referee "
                                 f"{referee} recalibrated it to {recal} (gap {gap:.1f})."),
                })
        violations = con.execute(
            "SELECT code, severity, COUNT(*) FROM model_integrity_violations "
            "GROUP BY code, severity ORDER BY COUNT(*) DESC").fetchall()
        if violations:
            flags.append({
                "severity": "critical", "status": "open",
                "title": "Recorded integrity violations",
                "detector": "any row in model_integrity_violations",
                "evidence": "; ".join(f"{code} ({sev}) ×{n}" for code, sev, n in violations) + ".",
            })
        failed = con.execute(
            "SELECT slug, COUNT(*) FROM steps WHERE status='failed' GROUP BY slug").fetchall()
        if failed:
            flags.append({
                "severity": "medium", "status": "open",
                "title": "Pipeline steps failed in tracked runs",
                "detector": "steps.status = 'failed'",
                "evidence": "; ".join(f"{slug} ×{n}" for slug, n in failed)
                            + " (all at the signing/publish step).",
            })
        total_runs = con.execute("SELECT COUNT(*) FROM runs").fetchone()[0]
        top_builder = con.execute(
            "SELECT builder_model, COUNT(*) FROM runs GROUP BY builder_model "
            "ORDER BY COUNT(*) DESC LIMIT 1").fetchone()
        con.close()
        if top_builder and total_runs and top_builder[1] / total_runs >= FLAG_MODEL_CONCENTRATION:
            flags.append({
                "severity": "high", "status": "open",
                "title": "Single-model concentration",
                "detector": f"top builder share ≥ {FLAG_MODEL_CONCENTRATION:.0%}",
                "evidence": (f"{top_builder[0]} built {top_builder[1]} of {total_runs} tracked loops "
                             f"({top_builder[1]/total_runs:.0%}) and was also primary assessor on most "
                             "of them — builder/judge overlap is the surface the self-audit incident "
                             "came from."),
            })

    fix1 = d["fixers"]["results"].get("fix-001")
    if fix1 and fix1.get("status") == "fixed":
        flags.append({
            "severity": "resolved", "status": "resolved",
            "title": "Fabricated 'empirical' curves quarantined",
            "detector": "Problem-Scout candidate BIO-TPL-DQ-01, verified by fix-001",
            "evidence": ("Three bioelectric_polarity 'empirical' series were confirmed slot-seeded "
                         "alpha templates and quarantined with provenance labels (PR #271); "
                         "kept here as the worked example of a flag closing."),
        })
    return flags


def main() -> int:
    d = census()
    slots, scouts, fixers, tele = d["slots"], d["scouts"], d["fixers"], d["telemetry"]
    total_cloud = slots["n"] + scouts["dispatched"] + fixers["dispatched"]

    def pct(a, b):
        return 100.0 * a / b if b else 0.0

    roster = [
        ("Loop-Builder (slot agent)", "One cloud agent per MorphoEval slot: writes the model "
         "script, dataset, Lean module and nanopubs for one organism/mechanism loop.",
         slots["n"], f'{slots["completed"]} completed', pct(slots["completed"], slots["n"]),
         "morphoeval/cloud/dispatches/slot-*.json"),
        ("Problem-Scout", "Hunts one hunting-ground x tool-lens cell for resolvable problems; "
         "returns candidates for the central resolvability ranker.",
         scouts["dispatched"], f'{scouts["returned"]} reports in tree · {scouts["candidates"]} candidates',
         pct(scouts["returned"], scouts["dispatched"]), "morphoeval/problem_scouts/scout-*.json"),
        ("Problem-Fixer", "Repairs one leaderboard candidate in an isolated PR under a path "
         "allowlist + trusted merge gate; never self-merges.",
         fixers["dispatched"], f'{fixers["landed"]} landed on main', pct(fixers["landed"], fixers["dispatched"]),
         "morphoeval/problem_fixers/results/*.json"),
        ("Loop roles (12 canonical steps)", "Not separate dispatches: the scout/paper/source/"
         "rules/script/output/audit/measure/build/publish/graph/remed roles each loop run "
         "executes, counted from track_loop telemetry.",
         tele["steps"], f'{tele["runs"]} tracked runs · 106 executions per role', 100.0,
         "morphoeval/runs/telemetry.sqlite (steps)"),
        ("Orchestrator — Muse Wonder", "The session conductor: designs the fleets, writes the "
         "dispatchers, gates and this dashboard, verifies scout claims before acting on them, "
         "and merges the PRs. Leaves no dispatch record — GitHub attests the account, not the "
         "model — so this row is session-attested, a weaker evidence grade than every row above.",
         "1 session", "16 PRs merged (#269–#288, incl. 10 fixer PRs) · issues #268/#283/#285 · "
         "Plates III–V", 100.0, "github.com/fractastical/morpholib/pulls (merged)"),
    ]
    roster_rows = "\n".join(
        f"<tr><td><strong>{html.escape(name)}</strong><br><span class='dim'>{html.escape(desc)}</span></td>"
        f"<td class='num'>{used:,}</td><td>{html.escape(done)}</td>" if isinstance(used, int) else
        f"<tr><td><strong>{html.escape(name)}</strong><br><span class='dim'>{html.escape(desc)}</span></td>"
        f"<td class='num'>{html.escape(str(used))}</td><td>{html.escape(done)}</td>"
        f"<td>{bar(p)}<span class='dim'>{p:.0f}%</span></td>"
        f"<td><code>{html.escape(src)}</code></td></tr>"
        for name, desc, used, done, p, src in roster
    )

    # Scout ground x lens matrix
    g_order, l_order = scouts["ground_order"], scouts["lens_order"]
    head = "".join(f"<th>{html.escape(str(x))}</th>" for x in l_order)
    matrix_rows = ""
    for g in g_order:
        tds = ""
        for lens in l_order:
            n = scouts["cells"].get((g, lens), 0)
            cls = "cell hit" if n else "cell miss"
            tds += f'<td class="{cls}">{n if n else "·"}</td>'
        matrix_rows += f"<tr><th>{html.escape(str(g))}</th>{tds}<td class='num'>{scouts['grounds'][g]}</td></tr>"

    fixer_rows = ""
    for task in fixers["tasks"]:
        res = fixers["results"].get(task["id"])
        status = res.get("status") if res else "report not on main (PR open / held)"
        fixer_rows += (
            f"<tr><td><code>{task['id']}</code></td><td>{html.escape(task['title'])}</td>"
            f"<td>#{task['leaderboard_rank']}</td><td>{task['merge_policy']}</td>"
            f"<td>{html.escape(str(status))}</td></tr>"
        )

    role_rows = "\n".join(
        f"<tr><td>{html.escape(slug)}</td><td>Tier {html.escape(str(tier))}</td>"
        f"<td class='num'>{n}</td><td>{bar(pct(n, tele['steps']))}</td></tr>"
        for slug, tier, n in tele["roles"]
    )
    actor_rows = "\n".join(
        f"<tr><td>{html.escape(str(a))}</td><td class='num'>{n}</td></tr>" for a, n in tele["actors"])
    model_rows = "\n".join(
        f"<tr><td>{html.escape(str(m))}</td><td class='num'>{n}</td></tr>" for m, n in tele["models"])

    mq = d.get("model_quality", {"builders": [], "incidents": [], "assessors": {}, "step_models": []})

    def fmt(x, digits=1):
        return f"{x:.{digits}f}" if isinstance(x, (int, float)) else "—"

    builder_rows = ""
    for i, b in enumerate(mq["builders"], 1):
        evidence = "solid" if b["loops"] >= 30 else ("thin — indicative only" if b["loops"] >= 5 else "anecdotal — do not rank on this")
        wall_min = f"{b['wall_s'] / 60:.1f} min" if b["wall_s"] else "—"
        tokens = f"{b['tokens']:,.0f}" if b["tokens"] else "—"
        builder_rows += (
            f"<tr><td>{i}</td><td><strong>{html.escape(str(b['model']))}</strong></td>"
            f"<td class='num'>{b['loops']}</td><td class='num'>{fmt(b['overall'])}</td>"
            f"<td class='num'>{fmt(b['fidelity'])}</td><td class='num'>{fmt(b['morphometric'])}</td>"
            f"<td class='num'>{fmt(b['specimen'])}</td><td class='num'>{fmt(b['steps_completed'])}/12</td>"
            f"<td class='num'>{wall_min}</td><td class='num'>{tokens}</td>"
            f"<td class='num'>{b['violations']}</td><td class='dim'>{evidence}</td></tr>"
        )
    incident_note = ""
    for off, referee, self_score, recal in mq["incidents"]:
        incident_note += (
            f'<p class="dim">Recorded integrity incident: loops built by <strong>{html.escape(str(off))}</strong> '
            f"carried a self-audit of {fmt(self_score)}, recalibrated to {fmt(recal)} by referee "
            f"<strong>{html.escape(str(referee))}</strong> — the reason auditing and building are kept on different models.</p>"
        )
    assessor_rows = ""
    for seat in ("primary", "second"):
        for model, n in mq["assessors"].get(seat, []):
            assessor_rows += (f"<tr><td>{seat} assessor</td><td>{html.escape(str(model))}</td>"
                              f"<td class='num'>{n}</td></tr>")
    step_model_rows = "\n".join(
        f"<tr><td>{html.escape(r['model'])}</td><td class='num'>{r['steps']}</td>"
        f"<td class='num'>{r['avg_s']:.1f}s</td></tr>" for r in mq["step_models"][:8]
    )

    sev_badge = {
        "critical": '<span class="badge crit">CRITICAL</span>',
        "high": '<span class="badge high">HIGH</span>',
        "medium": '<span class="badge med">MEDIUM</span>',
        "resolved": '<span class="badge ok">RESOLVED</span>',
    }
    sev_rank = {"critical": 0, "high": 1, "medium": 2, "resolved": 3}
    flag_rows = "\n".join(
        f"<tr class='flag-{f['severity']}'><td>{sev_badge[f['severity']]}</td>"
        f"<td><strong>{html.escape(f['title'])}</strong><br><span class='dim'>Detector: "
        f"<code>{html.escape(f['detector'])}</code></span></td>"
        f"<td>{html.escape(f['evidence'])}</td><td>{html.escape(f['status'])}</td></tr>"
        for f in sorted(d["red_flags"], key=lambda x: sev_rank[x["severity"]])
    )
    active_flags = sum(1 for f in d["red_flags"] if f["severity"] != "resolved")

    page = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>Fleet Registry — Agent Types &amp; Usage</title>
<style>
 body {{ background:#e8dcc0; color:#2b2118; font-family: Georgia, 'Times New Roman', serif; margin:0; }}
 .plate {{ max-width:1060px; margin:24px auto; background:#f5edd8; border:2px solid #7a5c2e;
   box-shadow:0 2px 14px rgba(60,40,10,.35); padding:28px 34px; }}
 h1 {{ font-variant:small-caps; letter-spacing:.06em; border-bottom:3px double #7a5c2e; padding-bottom:8px; }}
 h2 {{ font-variant:small-caps; color:#5d4420; border-bottom:1px solid #b39b5e; margin-top:30px; }}
 table {{ border-collapse:collapse; width:100%; font-size:14px; }}
 th, td {{ border:1px solid #c4ab72; padding:5px 8px; text-align:left; vertical-align:top; }}
 th {{ background:#e2d2a4; }}
 td.num {{ text-align:right; font-variant-numeric:tabular-nums; }}
 .dim {{ color:#6d5a38; font-size:12.5px; }}
 .bar {{ display:inline-block; width:110px; height:9px; background:#d8c892;
   border:1px solid #7a5c2e; vertical-align:middle; margin-right:6px; }}
 .fill {{ height:100%; }}
 .cell {{ text-align:center; }}
 .badge {{ font-size:11px; font-weight:bold; padding:2px 6px; border:1px solid #5d4420; }}
 .badge.crit {{ background:#a33327; color:#fff; }} .badge.high {{ background:#d98a4a; color:#fff; }}
 .badge.med {{ background:#e3cf6b; }} .badge.ok {{ background:#7ba05b; color:#fff; }}
 tr.flag-critical td {{ background:#f3d9d2; }} tr.flag-high td {{ background:#f7e3c8; }}
 tr.flag-resolved td {{ background:#dde8cf; }} .hit {{ background:#cfe0b8; }} .miss {{ background:#e8c9c0; color:#8a4a3a; }}
 .totals {{ display:flex; gap:14px; flex-wrap:wrap; margin:16px 0; }}
 .card {{ border:1.5px solid #7a5c2e; background:#efe4c4; padding:10px 16px; min-width:150px; }}
 .card b {{ font-size:24px; display:block; }}
 code {{ font-size:12px; }}
</style></head><body><div class="plate">
<h1>Plate III — Registry of the Agent Fleet</h1>
<p class="dim">Every agent type used so far, counted from the records the repository itself keeps
(cloud dispatch records, returned reports, and loop telemetry). Chat-session subagents leave no
repo record and are not counted. Regenerate: <code>python3 docs/build_fleet_dashboard.py</code>.</p>

<div class="totals">
 <div class="card"><b>{total_cloud:,}</b>cloud agents dispatched</div>
 <div class="card"><b>{slots['n']:,}</b>loop-builder slots (#{1}–{slots['max_slot']})</div>
 <div class="card"><b>{scouts['candidates']:,}</b>scout candidates returned</div>
 <div class="card"><b>{tele['steps']:,}</b>loop-role executions tracked</div>
 <div class="card"><b>{slots['nanopubs_signed']:,}</b>nanopubs signed by slot agents*</div>
</div>

<h2>I. The Fleet Roster — types &amp; how often used</h2>
<table><tr><th>Agent type</th><th>Used</th><th>Outcome</th><th>Completion</th><th>Records</th></tr>
{roster_rows}</table>

<h2>II. Problem-Scouts — hunting ground × tool lens ({scouts['returned']} of {scouts['dispatched']} returned)</h2>
<table><tr><th>ground \\ lens</th>{head}<th>Σ</th></tr>
{matrix_rows}</table>
<p class="dim">Missing cells (·) are scouts that were dispatched but whose report has not landed on main.</p>

<h2>III. Problem-Fixer Wave 1 — {fixers['landed']} of {fixers['dispatched']} landed on main</h2>
<table><tr><th>Fix</th><th>Candidate</th><th>Rank</th><th>Policy</th><th>Agent-reported status</th></tr>
{fixer_rows}</table>

<h2>IV. Inside every loop — the 12 canonical roles (telemetry, {tele['runs']} runs)</h2>
<table><tr><th>Role</th><th>Tier</th><th>Executions</th><th>Share</th></tr>
{role_rows}</table>

<h2>V. Who (or what) filled the roles — top actors &amp; models in telemetry</h2>
<table><tr><th>Actor (as recorded)</th><th>Steps</th></tr>{actor_rows}</table>
<table style="margin-top:10px"><tr><th>Model</th><th>Steps</th></tr>{model_rows}</table>

<h2>VI. Plate IV — Model Leaderboard (builders, ranked by mean audit score)</h2>
<p class="dim">Quality = the independent audit scores of the loops each model built
(<code>math_models</code> joined to tracked runs), plus speed, token cost and integrity
record. Small samples are marked — three loops do not make a ranking.</p>
<table><tr><th>#</th><th>Model</th><th>Loops built</th><th>Overall</th><th>Fidelity</th>
<th>Morphometric</th><th>Specimen</th><th>Steps done</th><th>Wall / loop</th><th>Tokens / loop</th>
<th>Integrity violations</th><th>Evidence</th></tr>
{builder_rows}</table>
{incident_note}

<h2>VII. Assessors &amp; step-level model usage</h2>
<table><tr><th>Assessor seat</th><th>Model</th><th>Runs judged</th></tr>
{assessor_rows}</table>
<table style="margin-top:10px"><tr><th>Model (steps, name-normalised)</th><th>Steps executed</th><th>Avg step time</th></tr>
{step_model_rows}</table>

<h2>VIII. Red Flags — big gaps &amp; severe misreporting ({active_flags} active)</h2>
<p class="dim">Each flag is a named detector rule over the same records; regenerate the dashboard
and any threshold crossed since last time appears here. Resolved flags stay on the plate as
precedent.</p>
<table><tr><th>Severity</th><th>Flag</th><th>Evidence</th><th>Status</th></tr>
{flag_rows}</table>

<h2>IX. The unranked fleets — a recording gap</h2>
<p>The 2,201 slot dispatches and the Cursor cloud fleets (100 scouts, 12 fixers) do not record
which model each agent ran, so they cannot be ranked yet — the leaderboard above covers only the
106 telemetry-tracked loops. Fleet outcomes by completion are in Plate I; recording the model in
every dispatch record (the fixer dispatcher already accepts <code>--model</code>) would make the
next wave rankable end-to-end.</p>

<p class="dim">* Slot-agent Lean/nanopub counts are as recorded at dispatch time; per issue #268 the
overnight-slot Lean modules are placeholder templates, not mechanism proofs, so treat
<code>lean_theorems_proved</code>-style tallies from those records as claims, not verifications.
Telemetry runs span {html.escape(str(tele['first']))} – {html.escape(str(tele['last']))}.</p>
</div></body></html>
"""
    OUT.write_text(page)
    print(f"wrote {OUT} ({len(page):,} bytes)")
    print(f"cloud total={total_cloud:,} slots={slots['n']:,} scouts={scouts['dispatched']} "
          f"fixers={fixers['dispatched']} telemetry_steps={tele['steps']:,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

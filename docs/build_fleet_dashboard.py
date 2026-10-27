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
        con.close()
    data["telemetry"] = tele
    return data


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
    ]
    roster_rows = "\n".join(
        f"<tr><td><strong>{html.escape(name)}</strong><br><span class='dim'>{html.escape(desc)}</span></td>"
        f"<td class='num'>{used:,}</td><td>{html.escape(done)}</td>"
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
 .cell {{ text-align:center; }} .hit {{ background:#cfe0b8; }} .miss {{ background:#e8c9c0; color:#8a4a3a; }}
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

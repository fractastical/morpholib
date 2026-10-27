#!/usr/bin/env python3
"""Problem-Scout 100 — one hundred agents hunting resolvable problems.

Each of 100 cloud agents owns one cell of a 10x10 grid: a *hunting ground*
(a territory of this repo where problems are known to live) crossed with a
*tool lens* (one of our own instruments). Its job is NOT to solve the
problem — it is to find problems that **could be resolved with our tools**,
state an exact definition-of-done, and score the evidence. A central,
deterministic ranker then re-scores every candidate on a published
resolvability rubric (agent-supplied totals are ignored) and emits the
leaderboard. Resolvability != viability: a problem can be almost certainly
solvable-in-principle yet unresolvable *here* (no data, no tool), and the
rubric prices exactly that.

Zero-conflict isolation (as cloud_orchestrator.py): each agent writes
exactly ONE new file, morphoeval/problem_scouts/scout-<NNN>.json, on its
own branch/PR (morphoeval/problem-scout-<NNN>). Nothing existing is edited;
the leaderboard is regenerated locally from the returned files.

Usage:
  python3 morphoeval/problem_scout_100.py plan            # write the 100-agent manifest
  python3 morphoeval/problem_scout_100.py prompt --agent 7
  python3 morphoeval/problem_scout_100.py rank            # (re)build leaderboard from candidates
  python3 morphoeval/problem_scout_100.py dispatch --count 100 --dry-run
  CURSOR_API_KEY=... python3 morphoeval/problem_scout_100.py dispatch --count 100
"""
from __future__ import annotations
import argparse, json, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
MANIFEST = HERE / "cloud" / "problem_scout_100.yaml"
SCOUT_DIR = HERE / "problem_scouts"

GROUNDS = [  # (id, territory, where problems live)
    ("rule30", "Rule 30 / Conus prize problems", "docs/RULE30_*.md, docs/open-problems.json (OP-01..03), Conus/Rule30 Lean & kymograph assets"),
    ("thompson", "Thompson transformations & portfolio residue", "docs/THOMPSON_*.md, docs/portfolio/, thompson/ (failed clades Leporidae/Vespinae, >50 Myr horizon)"),
    ("wavevector", "Ca2+ wave-vector imaging & blind tests", "wave-vector-analysis/, embryo-blind/, zebrafish-blind/ (orientation & mirror confounds OP-07/08)"),
    ("turing", "Turing / reaction-diffusion flagship debt", "models/ flagship scripts, datasets/*turing*, morphoeval audits (growing-domain solver, Meinhardt autonomy)"),
    ("bioelectric", "Bioelectric patterning (Levin latent space)", "datasets/*bioelectric*, docs/LEVIN_*.md, planarian & Xenopus bioelectric datasets"),
    ("lean", "Lean formalisation & Math-KG gaps", "morphoeval/lean/loops/, morphoeval/math_kg_loops/, unproved invariants named in audits"),
    ("predictions", "Registered-prediction follow-ups", "predictions/prediction_registry.json + resolutions/ (5 partially-resolved: missing quantitative series)"),
    ("library", "Library & dataset integrity", "datasets/ (3,122), scout_manifest/reference fitness, datasets/planform/damaged/, verify_image_fitness.py"),
    ("nanopub", "Nanopublication & knowledge-graph residue", "morphoeval/kg/ (327 published of thousands audited), sign_morphoeval.py, unpublished/superseded bundles"),
    ("scaling", "Scaling, allometry & the fossil record", "datasets/ paleo/scale catalogues, morphoeval/scale_and_paleo_catalog.py, phenomenon_size_m / geological_age_ma fields"),
]
LENSES = [  # (id, our tool, what it can resolve)
    ("literature", "scout_literature.py (OpenAlex/Crossref/Commons)", "problems whose missing piece is a findable paper, dataset or specimen image"),
    ("measure", "measure_outputs.py + audits", "problems decidable by re-measuring existing outputs against references"),
    ("lean", "Lean 4 loop files (0-sorry contract)", "claims that are, or could become, machine-checkable theorems"),
    ("simulation", "model scripts with --self-test + sweeps/ablations", "problems decidable by running, sweeping or ablating an existing model"),
    ("vision", "verify_image_fitness.py / specimen CV", "problems decidable by segmenting or measuring a held specimen photograph"),
    ("wavepipe", "wave-vector-analysis pipeline", "problems inside the spark-track -> cluster -> hypothesis-plot chain"),
    ("knowledge", "build_morphoeval_kg.py + sign/verify nanopubs", "problems that are really unpublished, unsigned or unranked claims"),
    ("registry", "prediction registry + score_resolutions.py", "problems statable as a blind, hash-locked, resolvable prediction"),
    ("remediation", "the 12-step MorphoEval loop (Pass1->Pass2)", "flagged audit weaknesses closable by one remediation pass"),
    ("delphi", "problem-graph + Bayesian Delphi scoring", "unscored or stale problems needing a viability prior before work is spent"),
]

# Resolvability rubric: factor -> (weight, what earns it). Deterministic;
# the ranker computes totals from factor evidence, never from agent totals.
RUBRIC = {
    "tool_fit": (25, "a named repo tool, run as-is, produces the deciding evidence"),
    "data_in_hand": (20, "deciding data already in repo, or one named public fetch away"),
    "falsifiable_done": (15, "an exact definition-of-done a scorer can check without taste"),
    "compute_in_repo": (15, "runs on the repo stack in hours, no wet lab / no new infrastructure"),
    "isolation": (10, "closable as new files only, zero-conflict with parallel agents"),
    "time_to_verdict": (10, "verdict reachable in one loop / one sitting"),
    "not_blocked": (5, "no recorded prior attempt already refuted this exact route"),
}

def agents() -> list[dict]:
    out = []
    for gi, (gid, gname, where) in enumerate(GROUNDS):
        for li, (lid, tool, resolves) in enumerate(LENSES):
            n = gi * len(LENSES) + li + 1
            out.append({"agent": n, "id": f"scout-{n:03d}", "ground": gid, "territory": gname,
                        "where": where, "lens": lid, "tool": tool, "resolves": resolves,
                        "branch": f"morphoeval/problem-scout-{n:03d}",
                        "deliverable": f"morphoeval/problem_scouts/scout-{n:03d}.json"})
    return out

def build_prompt(a: dict) -> str:
    rubric = "\n".join(f"  - {k} (0-{w}): {d}" for k, (w, d) in RUBRIC.items())
    return f"""You are Problem-Scout **{a['id']}** of the Problem-Scout 100 fleet (Project Delphi / MorphoEval).

## YOUR CELL OF THE HUNTING GRID
- **Territory:** {a['territory']} — hunt here: {a['where']}
- **Your one tool lens:** {a['tool']} — you may only propose problems it can resolve: {a['resolves']}

## MISSION — find problems, do not solve them
Search your territory (repo files first, then the literature your lens reaches) for problems that
**could be resolved with our tools**. For each candidate (aim 3-8, quality over quantity) record:
`title`, `gap` (exactly what is missing), `done` (a definition-of-done a scorer can check without taste),
`tool_route` (the named tool(s) and the concrete commands/inputs), `deciding_data` (in repo? named public source?),
`prior_attempts` (who tried, what happened — check docs/open-problems.json and the audits), and `factors`:
your evidence-based 0..weight score for EACH rubric factor below, each with a one-line `why`.
{rubric}
Rules: a problem your lens cannot decide scores tool_fit near 0 — report it anyway and say which lens could.
Negative knowledge counts: if a route is already refuted, record it as blocked, never re-propose it as new.
Pre-register before touching held-out data; hash-locked predictions stay blind.

## DELIVERABLE — exactly one new file, nothing else modified
Write `{a['deliverable']}`: {{"scout": "{a['id']}", "ground": "{a['ground']}", "lens": "{a['lens']}",
"candidates": [ ... ]}} on branch `{a['branch']}`. Validate it parses as JSON and that
`git diff --name-only --diff-filter=MD origin/main` is empty before committing."""

def score_candidate(c: dict) -> tuple[int, str]:
    """Deterministic resolvability 0-100 from factor evidence (agent totals ignored)."""
    f = c.get("factors") or {}
    total, parts = 0, []
    for k, (w, _d) in RUBRIC.items():
        v = f.get(k, 0)
        try: v = float(v.get("score", v) if isinstance(v, dict) else v)
        except Exception: v = 0
        v = max(0.0, min(float(w), v))
        total += v; parts.append(f"{k} {v:g}/{w}")
    band = "RESOLVE NOW" if total >= 70 else "RESOLVABLE WITH WORK" if total >= 45 else "HARD HERE" if total >= 25 else "NOT WITH OUR TOOLS (yet)"
    return int(round(total)), f"{band} — " + ", ".join(parts)

def seed_candidates() -> list[dict]:
    """The already-known board, honestly factored: docs/open-problems.json (OP-01..10)
    + the 5 partially-resolved registered predictions. Viability (Delphi %) is kept
    separate — resolvability prices our tools, data and done-criteria, not truth odds."""
    F = lambda **kw: {k: {"score": kw.get(k, 0), "why": kw.get(k + "_w", "")} for k in RUBRIC}
    return [
        {"id": "OP-11", "title": "Cow Turing follow-ups: blinded re-score + migration-wave colonisation variant", "source": "docs/open-problems.json OP-11", "factors": F(tool_fit=22, data_in_hand=18, falsifiable_done=14, compute_in_repo=13, isolation=10, time_to_verdict=8, not_blocked=5)},
        {"id": "OP-09", "title": "Zebrafish blind test: run the hash-locked registered scorer", "source": "OP-09 / zebrafish-blind preregistration", "factors": F(tool_fit=23, data_in_hand=19, falsifiable_done=15, compute_in_repo=14, isolation=10, time_to_verdict=9, not_blocked=5)},
        {"id": "ZF-VTX-01", "title": "Zebrafish vertex-packing: tabulate EVL cell shape-index across epiboly from published images", "source": "predictions/resolutions/zebrafish.json", "factors": F(tool_fit=20, data_in_hand=12, falsifiable_done=15, compute_in_repo=13, isolation=10, time_to_verdict=7, not_blocked=5)},
        {"id": "DM-MUR-NEG-01", "title": "Drosophila tracheal diameters: fit branching exponent vs Murray 2.8-3.2 band", "source": "predictions/resolutions/drosophila.json", "factors": F(tool_fit=19, data_in_hand=11, falsifiable_done=15, compute_in_repo=13, isolation=10, time_to_verdict=7, not_blocked=5)},
        {"id": "OP-06", "title": "Flagship structural debt: growing-domain Turing, autonomous Meinhardt, C. elegans voxel calibration", "source": "OP-06 / flagship15 Pass N+1", "factors": F(tool_fit=20, data_in_hand=16, falsifiable_done=12, compute_in_repo=10, isolation=9, time_to_verdict=5, not_blocked=4)},
        {"id": "ZF-CLK-01", "title": "Zebrafish clock-wavefront: paired her1-period + somite-length series at controlled wavefront", "source": "predictions/resolutions/zebrafish.json", "factors": F(tool_fit=17, data_in_hand=8, falsifiable_done=15, compute_in_repo=11, isolation=10, time_to_verdict=5, not_blocked=5)},
        {"id": "XT-CLK-01", "title": "Xenopus clock-wavefront: paired period + somite-length series (cross-species replication)", "source": "predictions/resolutions/xenopus.json", "factors": F(tool_fit=17, data_in_hand=7, falsifiable_done=15, compute_in_repo=11, isolation=10, time_to_verdict=5, not_blocked=5)},
        {"id": "ZF-HH-01", "title": "Zebrafish Shh lipidation: paired ptc1/ptc2 domain assay, lipidation-perturbed vs wild-type", "source": "predictions/resolutions/zebrafish.json", "factors": F(tool_fit=13, data_in_hand=6, falsifiable_done=14, compute_in_repo=8, isolation=9, time_to_verdict=4, not_blocked=5)},
        {"id": "OP-05", "title": "Within-species bee P-matrix wing dataset (the one missing input)", "source": "OP-05 / why-wings", "factors": F(tool_fit=16, data_in_hand=5, falsifiable_done=13, compute_in_repo=10, isolation=9, time_to_verdict=4, not_blocked=5)},
        {"id": "OP-07", "title": "Xenopus neighbour transfer v2: ground-truth-geometry-only preregistration", "source": "OP-07 / embryo-blind", "factors": F(tool_fit=17, data_in_hand=10, falsifiable_done=13, compute_in_repo=11, isolation=8, time_to_verdict=4, not_blocked=2)},
        {"id": "OP-08", "title": "Pixel v2: mirror vs headward neighbour test where the predictions separate", "source": "OP-08 (post-unblinding, exploratory)", "factors": F(tool_fit=16, data_in_hand=10, falsifiable_done=12, compute_in_repo=11, isolation=8, time_to_verdict=4, not_blocked=2)},
        {"id": "OP-04", "title": "Prediction beyond the ~50 Myr horizon: pre-registered deep-horizon test vs clade prior", "source": "OP-04 / KEY_DISCOVERIES 1.11", "factors": F(tool_fit=14, data_in_hand=12, falsifiable_done=12, compute_in_repo=10, isolation=9, time_to_verdict=3, not_blocked=3)},
        {"id": "OP-10", "title": "Portfolio residue autopsy: Leporidae 0/3, Vespinae 0/7 Thompson-7 failures", "source": "OP-10 / PORTFOLIO_STATUS", "factors": F(tool_fit=15, data_in_hand=14, falsifiable_done=10, compute_in_repo=11, isolation=9, time_to_verdict=4, not_blocked=4)},
        {"id": "OP-01", "title": "Rule30 Prize 1: defects on closed gates (global mixing invariant, Lean)", "source": "OP-01, Delphi pooled 11.5%", "factors": F(tool_fit=15, data_in_hand=16, falsifiable_done=15, compute_in_repo=9, isolation=10, time_to_verdict=2, not_blocked=1)},
        {"id": "OP-02", "title": "Rule30 Prize 2: diagonal-sample decorrelation discrepancy bound", "source": "OP-02, Delphi pooled 2.0%", "factors": F(tool_fit=10, data_in_hand=14, falsifiable_done=15, compute_in_repo=9, isolation=10, time_to_verdict=2, not_blocked=1)},
        {"id": "OP-03", "title": "Rule30 Prize 3: barrier result for fixed-seed specialisation (not another bound attempt)", "source": "OP-03, Delphi pooled 0.04%", "factors": F(tool_fit=8, data_in_hand=14, falsifiable_done=11, compute_in_repo=8, isolation=10, time_to_verdict=2, not_blocked=0)},
    ]

def cmd_plan(_a) -> int:
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# Problem-Scout 100 — generated by morphoeval/problem_scout_100.py; do not hand-edit.",
             "schema_version: problem-scout-100/1.0", "isolation: one-new-file-per-agent (morphoeval/problem_scouts/scout-<NNN>.json)",
             f"rubric: {json.dumps({k: w for k, (w, _) in RUBRIC.items()})}", "agents:"]
    for a in agents():
        lines.append(f"  - {{agent: {a['agent']}, id: {a['id']}, ground: {a['ground']}, lens: {a['lens']}, branch: {a['branch']}, deliverable: {a['deliverable']}}}")
    MANIFEST.write_text("\n".join(lines) + "\n")
    print(f"Wrote {MANIFEST} — 100 agents ({len(GROUNDS)} grounds x {len(LENSES)} lenses)")
    return 0

def cmd_prompt(a) -> int:
    ag = {x["agent"]: x for x in agents()}
    if a.agent not in ag: print(f"agent must be 1..100"); return 1
    print(build_prompt(ag[a.agent])); return 0

def cmd_rank(_a) -> int:
    SCOUT_DIR.mkdir(parents=True, exist_ok=True)
    cands = list(seed_candidates()); sources = ["seed: docs/open-problems.json + predictions/resolutions/"]
    for f in sorted(SCOUT_DIR.glob("scout-*.json")):
        try: doc = json.loads(f.read_text())
        except Exception: continue
        for c in doc.get("candidates", []):
            c.setdefault("id", f"{doc.get('scout','?')}-c"); c["source"] = f.name
            cands.append(c); sources.append(f.name)
    ranked = sorted(((score_candidate(c)[0], score_candidate(c)[1], c) for c in cands), key=lambda t: -t[0])
    (SCOUT_DIR / "leaderboard.json").write_text(json.dumps(
        [{"rank": i + 1, "resolvability": s, "band_detail": d, **c} for i, (s, d, c) in enumerate(ranked)], indent=2) + "\n")
    md = ["# Problem-Scout 100 — Resolvability Leaderboard", "",
          f"Ranked by the deterministic rubric {dict((k, w) for k, (w, _) in RUBRIC.items())} (max 100).",
          "Resolvability prices *our tools, data in hand, and a checkable definition-of-done* — not the probability the claim is true (that is the Delphi viability score, kept separate).",
          "", "| # | Score | Problem | Source |", "|---|---|---|---|"]
    md += [f"| {i+1} | **{s}** | {c.get('title','?')} | {c.get('source','?')} |" for i, (s, _d, c) in enumerate(ranked)]
    md += ["", "## Why each score (factor evidence)", ""]
    md += [f"{i+1}. **{c.get('id','?')} — {s}/100.** {d}" for i, (s, d, c) in enumerate(ranked)]
    md += ["", f"Candidates: {len(ranked)} (seed board + returned scout files: {sorted(set(sources))}).",
           "Returned agent files land as `morphoeval/problem_scouts/scout-<NNN>.json`; re-run `python3 morphoeval/problem_scout_100.py rank` to re-rank."]
    (SCOUT_DIR / "LEADERBOARD.md").write_text("\n".join(md) + "\n")
    print(f"Ranked {len(ranked)} candidates -> {SCOUT_DIR/'LEADERBOARD.md'}")
    for i, (s, _d, c) in enumerate(ranked[:10]): print(f"  {i+1:2d}. {s:3d}  {c.get('id','?'):14s} {c.get('title','')[:80]}")
    return 0

def cmd_dispatch(a) -> int:
    failures: list[tuple[str, str]] = []
    ag = agents()[: a.count] if not a.agents else [x for x in agents() if x["agent"] in
          {int(t) for part in a.agents.split(",") for t in (range(int(part.split("-")[0]), int(part.split("-")[1]) + 1) if "-" in part else [int(part)])}]
    for x in ag:
        if a.dry_run:
            print(f"  [DRY-RUN] {x['id']} ground={x['ground']} lens={x['lens']} -> {x['deliverable']} | prompt={len(build_prompt(x)):,} chars")
            continue
        log = HERE / "cloud" / "dispatches" / f"problem-{x['id']}.json"
        if log.exists() and not a.force:
            print(f"  [SKIP] {x['id']} already dispatched (record {log.name})"); continue
        try:
            from cloud_orchestrator import _dispatch_via_cursor_sdk  # same transport as the loop fleet
        except Exception as exc:
            print(f"dispatch unavailable: {exc}"); return 1
        rec, last_exc = None, None
        for attempt in range(1, 4):  # transient SDK Bridge timeouts: retry with backoff, never abort the wave
            try:
                rec = _dispatch_via_cursor_sdk(prompt=build_prompt(x), repo=a.repo, branch_name=x["branch"], model=a.model)
                break
            except Exception as exc:
                last_exc = exc
                print(f"  [RETRY {attempt}/3] {x['id']}: {str(exc)[:140]}")
                time.sleep(10 * attempt)
        if rec is None:
            failures.append((x["id"], str(last_exc)[:200])); continue
        log.write_text(json.dumps({"kind": "problem_scout", **x, **rec}, indent=2) + "\n")
        print(f"  [DISPATCHED] {x['id']} agent_id={rec.get('agent_id')}", flush=True)
        time.sleep(3)  # stagger: the run of 2026-10-27 died on agent 13 without it
    if a.dry_run: print(f"Dry-run only: {len(ag)} agents armed. Live dispatch needs CURSOR_API_KEY (see cloud_orchestrator.py doctor).")
    if failures:
        print(f"FAILED ({len(failures)}): " + "; ".join(f"{i}: {e}" for i, e in failures))
        return 1
    return 0

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd")
    sub.add_parser("plan", help="Write the 100-agent manifest")
    p = sub.add_parser("prompt", help="Print one agent's mission prompt"); p.add_argument("--agent", type=int, required=True)
    sub.add_parser("rank", help="Rank all candidates by resolvability")
    d = sub.add_parser("dispatch", help="Dispatch scout agents (cloud)")
    d.add_argument("--count", type=int, default=100); d.add_argument("--agents", default=None, help="e.g. 1-10,25")
    d.add_argument("--repo", default="fractastical/morpholib"); d.add_argument("--model", default=None)
    d.add_argument("--dry-run", action="store_true"); d.add_argument("--force", action="store_true", help="Re-dispatch even if a dispatch record exists (stragglers)")
    a = ap.parse_args()
    return {"plan": cmd_plan, "prompt": cmd_prompt, "rank": cmd_rank, "dispatch": cmd_dispatch}.get(a.cmd, lambda _a: (ap.print_help(), 0)[1])(a)

if __name__ == "__main__":
    raise SystemExit(main())

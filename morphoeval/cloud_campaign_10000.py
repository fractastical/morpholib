#!/usr/bin/env python3
"""
morphoeval/cloud_campaign_10000.py — Cloud dispatch for the 10,000-Animal Basic Look.

Slots #3101-#10000 (6,900 new loops; catalog: morphoeval/catalog_10000_animals.py).
Each cloud agent runs the basic 12-step loop for ONE animal at ONE scale in its
evolutionary history, then 2 remediation loops (Pass 2 always, Pass 3 only if a
gate fails), including the image-fitness test, on its own branch/PR with strict
zero-conflict isolation (only new, slot-unique files; no existing file edited).

Usage:
  python3 morphoeval/cloud_campaign_10000.py build-manifest
  python3 morphoeval/cloud_campaign_10000.py status
  python3 morphoeval/cloud_campaign_10000.py prompt --slot 3101
  python3 morphoeval/cloud_campaign_10000.py dispatch --count 10 --dry-run
  export CURSOR_API_KEY=...   # in ~/.zshrc; never commit it
  python3 morphoeval/cloud_campaign_10000.py dispatch --count 10
  python3 morphoeval/cloud_campaign_10000.py dispatch --slots 3101-3125
  python3 morphoeval/cloud_campaign_10000.py poll

Dispatch is capped per invocation (--max-batch, default 25) so a 6,900-agent
campaign is launched in reviewed waves, not one accidental burst.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from morphoeval.catalog_10000_animals import (  # noqa: E402
    END_SLOT,
    START_SLOT,
    build_all_specs,
    build_slot_spec,
    campaign_summary,
)

MANIFEST = HERE / "cloud" / "campaign_10000_slots.json"
SUMMARY = HERE / "runs" / "campaign_10000_summary.json"
DISPATCH_DIR = HERE / "cloud" / "dispatches10000"
DEFAULT_REPO = "fractastical/morpholib"
CURSOR_API_BASE = "https://api.cursor.com/v1"


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def cmd_build_manifest(_: argparse.Namespace) -> int:
    specs = build_all_specs()
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps({"campaign": "10000-animals-basic-look",
                                    "generated_at": utc_now(), "slots": specs}, indent=1) + "\n", encoding="utf-8")
    summary = campaign_summary(specs)
    summary["generated_at"] = utc_now()
    summary["manifest"] = str(MANIFEST.relative_to(ROOT))
    SUMMARY.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Wrote {MANIFEST.relative_to(ROOT)} ({len(specs)} slots)")
    return 0


def load_specs() -> list[dict[str, Any]]:
    if MANIFEST.exists():
        return json.loads(MANIFEST.read_text(encoding="utf-8"))["slots"]
    return build_all_specs()


def dispatched() -> dict[int, dict[str, Any]]:
    out: dict[int, dict[str, Any]] = {}
    if DISPATCH_DIR.exists():
        for f in DISPATCH_DIR.glob("slot-*.json"):
            try:
                r = json.loads(f.read_text(encoding="utf-8"))
                out[int(r["slot"])] = r
            except Exception:
                continue
    return out


def cmd_status(_: argparse.Namespace) -> int:
    specs = load_specs()
    done = dispatched()
    idx = HERE / "runs" / "aristotelian_categories_index.json"
    completed = {int(e["slot"]) for e in json.loads(idx.read_text()) if e.get("slot")} if idx.exists() else set()
    print(f"=== 10,000-Animal Basic Look (slots {START_SLOT}-{END_SLOT}: {len(specs)} new) ===")
    print(f"  Already in atlas index : {sum(1 for s in specs if s['slot'] in completed)}")
    print(f"  Dispatched (this campaign): {len(done)}")
    print(f"  Pending                : {len(specs) - len(done)}")
    print(json.dumps(campaign_summary(specs), indent=2))
    return 0


def build_prompt(spec: dict[str, Any]) -> str:
    slot, slug, snake = spec["slot"], spec["slug"], spec["snake"]
    d = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    run_id = f"{d}-{slug}-loop"
    branch = f"morphoeval/loop-{slot:04d}-{slug}"
    gate = spec["remediation"]["image_fitness_gate"]
    return f"""You are executing Autonomous Cloud Loop #{slot:04d} (`{branch}`) of the 10,000-Animal Basic Look (Project Delphi / MorphoEval).

## CRITICAL RULE #1 — STRICT ZERO-CONFLICT FILE ISOLATION
100+ loops run in parallel. Your final commit MUST NOT modify or delete ANY existing tracked file in `origin/main`. Create ONLY these new files (all unique to slot {slot}):
- `{spec['script']}`, `{spec['dataset_dir']}/` (scout_manifest.json, {snake}_reference.json, paper_citations.bib, README.md, specimen image), `{spec['lean_module']}`, `morphoeval/math_kg_loops/{spec['sim_id']}.yaml`, `docs/media/{snake}_comparison.png`, `docs/media/{snake}_growth.gif`, `morphoeval/audits/{d}-{slug}-audit*.yaml` (+ .measurements.yaml), `morphoeval/kg/{d}-{slug}-audit*/`, `morphoeval/runs/{run_id}.*`
- `verify_image_fitness.py` REWRITES `morphoeval/runs/aristotelian_categories_index.json` — if you run it, revert that file before committing (`git checkout -- morphoeval/runs/aristotelian_categories_index.json morphoeval/runs/image_fitness_and_provenance.json`). Write your per-loop fitness result ONLY to `morphoeval/runs/{run_id}-image-fitness.json`.
- Before committing verify `git diff --name-only --diff-filter=MD origin/main` is EMPTY. Never edit README/rubric/shared indexes/telemetry files (revert track_loop.py telemetry side-effects the same way).

## ASSIGNMENT — ONE ANIMAL, ONE SCALE, ONE POINT IN EVOLUTIONARY HISTORY
- **Animal:** *{spec['organism']}* ({spec['common_name']}) — {spec['megiston_genos']} -> {spec['subaltern_genos']} -> {spec['genus']}
- **Scale:** {spec['size_human']} ({spec['phenomenon_size_m']} m, log10={spec['log10_size_m']:+.3f}, tier {spec['scale_tier_num']}/8 `{spec['scale_tier']}`, {spec['scale_variant']} variant) — {spec['anatomical_structure']}
- **Evolutionary history:** {spec['geological_period']} ({spec['geological_age_ma']:g} Ma); lineage `{spec['fossil_lineage_group']}`; {spec['evolutionary_axis']}. Formalize what changes at THIS scale/age vs its predecessor/successor chronospecies — do not copy a sibling slot's parameters.
- **Cohort / law:** {spec['cohort_domain']} ({spec['efficient_cause']}); form {spec['poion_form']}; quantitative law {spec['poson_law']}; seed DOI {spec['seed_doi'] or 'search literature in Step 1'}

## THE BASIC LOOK — 12 STEPS, PASS 1
Init: `python3 morphoeval/track_loop.py init --run-id {run_id} --target {spec['sim_id']} --paper "{spec['organism']} {spec['cohort_domain']}"` (log each step with track_loop.py; revert shared telemetry files before commit).
1. scout: `python3 morphoeval/scout_literature.py --slot {slot} --out {spec['dataset_dir']}/scout_manifest.json` if present, else query OpenAlex/Crossref/Wikimedia Commons; confirm no existing script/dataset collides with `{slug}`.
2. paper: extract equations, state variables, boundary conditions, parameter table, empirical claims -> `{spec['dataset_dir']}/{snake}_reference.json`, `paper_citations.bib`, `README.md` (state scale {spec['size_human']} and age {spec['geological_age_ma']:g} Ma explicitly).
3. source: fetch/vendor a real CC-licensed specimen image for *{spec['organism']}* into `{spec['dataset_dir']}/` (fallback: `datasets/species_gallery/` copy {spec['gallery_path'] or 'n/a'}), SHA-256-pin it in the reference JSON.
4. rules: standalone Lean 4 in `{spec['lean_module']}` (namespace MathOfBiology.Loop_{slot:04d}, >=3 theorems, 0 `sorry`) + `morphoeval/math_kg_loops/{spec['sim_id']}.yaml`.
5. script (Pass 1 v1): `{spec['script']}` with the governing model, `--self-test`, `measure_morphoeval()`, renders `docs/media/{snake}_comparison.png` and an ANIMATED `docs/media/{snake}_growth.gif` (>=10 frames, >=480px — the image-fitness test rejects static/<340px output).
6. output (Pass 1): run `--self-test` and the script; record results.
7. audit (Pass 1): `morphoeval/audits/{d}-{slug}-audit.yaml` against all 7 `morphoeval/rubric.yaml` dimensions, line-pinned evidence, explicit remediation items.
8. measure (Pass 1): `python3 morphoeval/measure_outputs.py morphoeval/audits/{d}-{slug}-audit.yaml`.
9. build (Pass 1): `python3 morphoeval/build_morphoeval_kg.py morphoeval/audits/{d}-{slug}-audit.yaml --conform`.
10. publish (Pass 1): sign/verify nanopubs if `~/.nanopub/profile.yml` exists (`morphoeval/sign_morphoeval.py`, `node morphoeval/check_signed.mjs`); if signing is unavailable, record that honestly — do not fake it.
11. graph (Pass 1): verify `morphoeval/kg/{d}-{slug}-audit/REPORT.md`.
12. IMAGE-FITNESS TEST (Pass 1): score simulation vs the REAL specimen image per `morphoeval/verify_image_fitness.py`: image_fitness = 0.55*CV + 0.30*specimen-match + 0.15*morphometric-validation; CV = Sobel-orientation (12-bin) + radial-FFT (24-bin) + structure-tensor coherency similarity between specimen photo and the animated GIF's last frame. Write `morphoeval/runs/{run_id}-image-fitness.json` with image_fitness, cv_sim_similarity, n_frames, resolution, non-circular specimen metrics (value != reference — self-referential val==ref scores 0).

## REMEDIATION LOOP 1 — PASS 2 (ALWAYS)
Fix every Pass-1 remediation item: parameter-regime sweep, negative/ablation control, real specimen CV segmentation, goodness-of-fit (Pearson r, RMSE, KS/CV). Re-run steps 5-11 as audit-v2 (`supersedes_audit_id: {d}-{slug}-audit`) and RE-RUN the image-fitness test.
**Gates after Pass 2:** image_fitness >= {gate['image_fitness_min']}, cv_sim_similarity >= {gate['cv_sim_similarity_min']}, GIF frames >= {gate['animated_gif_min_frames']}, resolution >= {gate['min_resolution_px']}px, >=1 non-circular specimen metric, Pearson r > 0.98, audit overall >= 85.

## REMEDIATION LOOP 2 — PASS 3 (CONDITIONAL, MAX 3 PASSES TOTAL)
ONLY if any gate above still fails after Pass 2: diagnose the single largest failing term (CV orientation/FFT/coherency, occupancy, animation, or audit dimension), fix THAT in the model/render (not by inflating audit scores), re-run steps 5-11 as audit-v3 and the image-fitness test a third time. If a gate still fails after Pass 3, STOP, keep scores honest, and say exactly which gate failed and why in `morphoeval/runs/{run_id}.md` — never fabricate a pass.

Finish: `python3 morphoeval/track_loop.py summary --run-id {run_id} --md morphoeval/runs/{run_id}.md --csv morphoeval/runs/{run_id}.csv --trig morphoeval/runs/{run_id}.trig` (revert shared telemetry/index files after), verify zero modified tracked files, commit only your new files, open the PR.
"""


def cmd_prompt(args: argparse.Namespace) -> int:
    print(build_prompt(build_slot_spec(args.slot)))
    return 0


def _require_key() -> str:
    key = os.environ.get("CURSOR_API_KEY", "").strip()
    if not key:
        raise RuntimeError("CURSOR_API_KEY not set (it is exported from ~/.zshrc; use a login shell or export it). Use --dry-run to preview.")
    return key


def _dispatch(prompt: str, repo: str, branch: str, model: str | None) -> dict[str, Any]:
    key = _require_key()
    repo_url = repo if repo.startswith("http") else f"https://github.com/{repo}"
    payload: dict[str, Any] = {"prompt": {"text": prompt}, "repos": [{"url": repo_url, "startingRef": "main"}],
                               "autoCreatePR": True, "skipReviewerRequest": True}
    if model:
        payload["model"] = {"id": model}
    req = urllib.request.Request(f"{CURSOR_API_BASE}/agents", data=json.dumps(payload).encode(),
                                 headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json",
                                          "Accept": "application/json", "User-Agent": "Delphi-10000Animals/1.0"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            body = json.loads(r.read().decode("utf-8", "replace"))
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"POST /v1/agents -> HTTP {e.code}: {e.read().decode('utf-8','replace')[:1500]} "
                           "(Legacy Privacy Mode / missing GitHub App / bad key are the usual causes)") from e
    agent, run = body.get("agent") or {}, body.get("run") or {}
    return {"transport": "cursor_rest_v1", "agent_id": agent.get("id") or body.get("id"),
            "run_id": run.get("id"), "status": run.get("status") or agent.get("status", "CREATING"),
            "requested_branch_name": branch, "web_url": agent.get("url") or body.get("url")}


def parse_selector(sel: str) -> list[int]:
    out: set[int] = set()
    for p in sel.split(","):
        p = p.strip()
        if not p:
            continue
        if "-" in p:
            a, b = p.split("-", 1)
            out.update(range(int(a), int(b) + 1))
        else:
            out.add(int(p))
    return sorted(n for n in out if START_SLOT <= n <= END_SLOT)


def cmd_dispatch(args: argparse.Namespace) -> int:
    specs = {s["slot"]: s for s in load_specs()}
    done = dispatched()
    targets = parse_selector(args.slots) if args.slots else [s["slot"] for s in load_specs() if s["slot"] not in done][: args.count]
    if not args.force:
        targets = [t for t in targets if t not in done]
    targets = targets[: args.max_batch]
    if not targets:
        print("[10000] No pending slots matched.")
        return 0
    print(f"[10000] Dispatching {len(targets)} cloud agent(s): {targets[0]}..{targets[-1]} (wave cap {args.max_batch})")
    DISPATCH_DIR.mkdir(parents=True, exist_ok=True)
    for idx, slot in enumerate(targets, 1):
        spec = specs[slot]
        branch = f"morphoeval/loop-{slot:04d}-{spec['slug']}"
        if args.dry_run:
            print(f"  [DRY-RUN {idx}/{len(targets)}] #{slot} {spec['organism']} @ {spec['size_human']} "
                  f"({spec['geological_age_ma']:g} Ma) -> {branch} | prompt={len(build_prompt(spec)):,} chars")
            continue
        res = _dispatch(build_prompt(spec), args.repo, branch, args.model)
        rec = {"campaign": "10000-animals-basic-look", "slot": slot, "slot_id": spec["slot_id"],
               "organism": spec["organism"], "phenomenon_size_m": spec["phenomenon_size_m"],
               "geological_age_ma": spec["geological_age_ma"], "dispatched_at": utc_now(), **res}
        (DISPATCH_DIR / f"slot-{slot:04d}.json").write_text(json.dumps(rec, indent=2) + "\n", encoding="utf-8")
        print(f"  [DISPATCHED {idx}/{len(targets)}] #{slot} agent={res.get('agent_id')} branch={branch}")
        if idx < len(targets) and args.stagger_seconds > 0:
            time.sleep(args.stagger_seconds)
    if not args.dry_run:
        print("Agents run on Cursor cloud VMs and open PRs automatically. Re-run `poll`, merge PRs, then dispatch the next wave.")
    return 0


def cmd_poll(_: argparse.Namespace) -> int:
    done = dispatched()
    if not done:
        print("[10000] No dispatches yet in morphoeval/cloud/dispatches10000/.")
        return 0
    key = os.environ.get("CURSOR_API_KEY", "").strip()
    for slot, rec in sorted(done.items()):
        if key and rec.get("agent_id"):
            try:
                req = urllib.request.Request(f"{CURSOR_API_BASE}/agents/{rec['agent_id']}",
                                             headers={"Authorization": f"Bearer {key}", "Accept": "application/json"})
                with urllib.request.urlopen(req, timeout=30) as r:
                    live = json.loads(r.read().decode())
                rec["status"] = live.get("status", rec.get("status"))
                (DISPATCH_DIR / f"slot-{slot:04d}.json").write_text(json.dumps(rec, indent=2) + "\n", encoding="utf-8")
            except Exception as e:  # best-effort
                rec["poll_error"] = str(e)[:200]
        print(f"  #{slot} {rec.get('organism')}: agent={rec.get('agent_id')} status={rec.get('status')} branch={rec.get('requested_branch_name')}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd")
    sub.add_parser("build-manifest")
    sub.add_parser("status")
    p = sub.add_parser("prompt"); p.add_argument("--slot", type=int, required=True)
    d = sub.add_parser("dispatch")
    d.add_argument("--count", type=int, default=10); d.add_argument("--slots", default=None)
    d.add_argument("--max-batch", type=int, default=25); d.add_argument("--repo", default=DEFAULT_REPO)
    d.add_argument("--model", default=None); d.add_argument("--stagger-seconds", type=float, default=2.0)
    d.add_argument("--force", action="store_true"); d.add_argument("--dry-run", action="store_true")
    sub.add_parser("poll")
    args = ap.parse_args()
    fn = {"build-manifest": cmd_build_manifest, "status": cmd_status, "prompt": cmd_prompt,
          "dispatch": cmd_dispatch, "poll": cmd_poll}.get(args.cmd)
    if fn is None:
        ap.print_help()
        return 0
    return fn(args)


if __name__ == "__main__":
    raise SystemExit(main())

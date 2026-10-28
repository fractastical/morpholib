#!/usr/bin/env python3
"""
Overnight runner for the Thompson (1917) claim queue.

Dispatches Cursor Cloud Agents (they run on Cursor's cloud while this
machine sleeps) for the loop-queued Thompson 1917 claims, highest
P(resolvable) first, in waves; polls them; and keeps a resumable state +
morning summary. Nothing here edits the claim registry -- agents write
per-claim result.json files on their own PR branches, per the loop
isolation rule, and a human merges verdicts back.

MODEL POLICY (2026-10-27, per user): NO GEMINI. It has not been doing a
good job on this work. All agents are pinned to MODEL below and the model
is logged per dispatch in the state file.

State (all under thompson/overnight/, safe to re-run any time):
  state.json    per-claim dispatch/poll records (the checkpoint)
  events.jsonl  append-only event log
  SUMMARY.md    morning report, regenerated on every poll

Usage:
  venv/bin/python thompson/run_overnight_1917.py --plan
  venv/bin/python thompson/run_overnight_1917.py --dispatch-one T1917-051
  venv/bin/python thompson/run_overnight_1917.py --watch            # overnight
  venv/bin/python thompson/run_overnight_1917.py --poll             # morning
  venv/bin/python thompson/run_overnight_1917.py --summary
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STATE_DIR = ROOT / "thompson" / "overnight"
STATE = STATE_DIR / "state.json"
EVENTS = STATE_DIR / "events.jsonl"
SUMMARY = STATE_DIR / "SUMMARY.md"
QUEUE = ROOT / "thompson" / "thompson_1917_loop_queue.yaml"
REGISTRY = ROOT / "thompson" / "thompson_1917_claims.json"
RESULTS = ROOT / "datasets" / "thompson_1917"

MODEL = "composer-2.5"  # default; overridden by --model. NO GEMINI either way (see docstring).
# Cohort note: wave 1 (T1917-051/027/033/041/008/028, PRs #295-300) ran on
# composer-2.5 as the baseline; from 2026-10-27 22:28 ET the user switched
# the remaining waves to Fable. Every dispatch logs its own model, so the
# two cohorts can be graded against each other in the morning.
TERMINAL = {"FINISHED", "COMPLETED", "FAILED", "ERROR", "CANCELLED", "EXPIRED"}
RESOLVED = ("DONE_REPO", "DONE_ELSEWHERE")

sys.path.insert(0, str(ROOT / "morphoeval"))


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def log_event(kind: str, **kw) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    with EVENTS.open("a") as fh:
        fh.write(json.dumps({"t": now(), "kind": kind, **kw}) + "\n")


def load_state() -> dict:
    if STATE.exists():
        return json.loads(STATE.read_text())
    return {"created": now(), "model": MODEL, "model_policy": "no-gemini (pinned composer-2.5)",
            "claims": {}}


def save_state(state: dict) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    tmp = STATE.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2) + "\n")
    tmp.replace(STATE)


def plan(state: dict, min_p: float) -> tuple[list[dict], list[tuple[dict, str]]]:
    """Queue entries sorted by P(resolvable) desc, split into todo / skipped."""
    import yaml
    reg = {c["id"]: c for c in json.loads(REGISTRY.read_text())["claims"]}
    slots = yaml.safe_load(QUEUE.read_text())["slots"]
    todo, skipped = [], []
    for e in slots:
        cid = e["claim_id"]
        status = reg[cid]["pass2"]["status"]
        rec = state["claims"].get(cid)
        if status in RESOLVED:
            skipped.append((e, f"already resolved ({status})"))
        elif (RESULTS / cid.lower() / "result.json").exists():
            skipped.append((e, "result.json already present"))
        elif rec and rec.get("status") not in ("failed_dispatch",):
            skipped.append((e, f"already dispatched ({rec.get('status')})"))
        elif (e.get("p_resolvable_percent") or 0) < min_p:
            skipped.append((e, f"P(resolvable) {e.get('p_resolvable_percent')} < {min_p}"))
        else:
            todo.append(e)
    todo.sort(key=lambda e: -(e.get("p_resolvable_percent") or 0))
    return todo, skipped


def get_prompt(claim_id: str) -> str:
    out = subprocess.run([sys.executable, str(ROOT / "thompson" / "scout_1917_datasets.py"),
                          "--prompt", claim_id], capture_output=True, text=True, cwd=ROOT)
    if out.returncode != 0:
        raise RuntimeError(f"prompt generation failed for {claim_id}: {out.stderr.strip()}")
    return out.stdout


def dispatch_one(state: dict, entry: dict, model: str = MODEL) -> dict:
    import cloud_orchestrator as co
    cid = entry["claim_id"]
    prompt = get_prompt(cid)
    branch = f"morphoeval/loop-{entry['slot']:03d}-{entry['proposed_slug']}"
    res = co._dispatch_via_cursor_sdk(prompt=prompt, repo=co.DEFAULT_REPO,
                                      branch_name=branch, model=model)
    rec = {"claim_id": cid, "slot": entry["slot"], "p_resolvable_percent": entry.get("p_resolvable_percent"),
           "model": model, "dispatched_at": now(), "status": "dispatched",
           "requested_branch": branch, **res}
    state["claims"][cid] = rec
    save_state(state)
    log_event("dispatch", claim_id=cid, slot=entry["slot"], agent_id=res.get("agent_id"),
              p=entry.get("p_resolvable_percent"), model=model)
    print(f"  dispatched {cid} (slot {entry['slot']}, P={entry.get('p_resolvable_percent')}%) "
          f"agent={res.get('agent_id')} model={model}")
    return rec


def poll_all(state: dict) -> None:
    import cloud_orchestrator as co
    for cid, rec in state["claims"].items():
        if rec.get("status") in TERMINAL or not rec.get("agent_id"):
            continue
        try:
            live = co.poll_cloud_agent(rec["agent_id"])
            rec["status"] = (live.get("run_status") or live.get("status") or rec["status"]).upper()
            rec["last_polled"] = now()
            if live.get("pr_url"):
                rec["pr_url"] = live["pr_url"]
            if live.get("branch_name"):
                rec["branch_name"] = live["branch_name"]
            if live.get("web_url"):
                rec["web_url"] = live["web_url"]
        except Exception as exc:  # keep polling the rest
            rec["poll_error"] = str(exc)
    save_state(state)


def write_summary(state: dict) -> str:
    todo, skipped = plan(state, 0)
    claims = state["claims"]
    from collections import Counter
    counts = Counter(r.get("status", "?") for r in claims.values())
    lines = [f"# Thompson (1917) overnight run — summary",
             f"\nUpdated {now()} · models by claim below (policy: no Gemini) · "
             f"state `thompson/overnight/state.json`",
             f"\nDispatched: {len(claims)} claim(s); statuses: {dict(counts)}; "
             f"still queued (not yet dispatched): {len(todo)}; skipped: {len(skipped)}\n",
             "| Claim | P(resolve) | Status | Agent | PR |",
             "|---|---|---|---|---|"]
    for cid, r in sorted(claims.items(), key=lambda kv: -(kv[1].get("p_resolvable_percent") or 0)):
        pr = f"[PR]({r['pr_url']})" if r.get("pr_url") else "—"
        lines.append(f"| {cid} (Thompson 1917) | {r.get('p_resolvable_percent')}% | {r.get('status')} "
                     f"| `{r.get('agent_id')}` | {pr} |")
    if todo:
        lines += ["\nNext in queue (highest P first): "
                  + ", ".join(f"{e['claim_id']} ({e.get('p_resolvable_percent')}%)" for e in todo[:10])]
    lines += ["\nMorning checklist: `--poll`, review PRs, check `datasets/thompson_1917/<claim>/result.json` "
              "on each PR branch for DATASET_MISMATCH stops, then merge verdicts into the registry by hand."]
    SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY.write_text("\n".join(lines) + "\n")
    return str(SUMMARY)


def cmd_watch(args) -> int:
    state = load_state()
    deadline = time.time() + args.hours * 3600
    rotation = [m.strip() for m in (args.models or args.model).split(",") if m.strip()]
    assert all("gemini" not in m.lower() for m in rotation), "model policy: no Gemini"
    print(f"[overnight] models={rotation} (no Gemini) wave={args.wave} poll={args.poll_seconds}s "
          f"deadline={args.hours}h min_p={args.min_p}")
    while time.time() < deadline:
        todo, _ = plan(state, args.min_p)
        active = [r for r in state["claims"].values() if r.get("status") not in TERMINAL]
        if not todo and not active:
            print("[overnight] queue exhausted and all agents terminal — done.")
            break
        room = args.wave - len(active)
        for j, entry in enumerate(todo[:max(room, 0)]):
            model = rotation[(len(state["claims"]) + j) % len(rotation)]
            try:
                dispatch_one(state, entry, model=model)
            except Exception as exc:
                state["claims"][entry["claim_id"]] = {"claim_id": entry["claim_id"],
                                                      "status": "failed_dispatch", "error": str(exc),
                                                      "model": MODEL, "dispatched_at": now()}
                save_state(state)
                log_event("dispatch_error", claim_id=entry["claim_id"], error=str(exc))
                print(f"  dispatch failed {entry['claim_id']}: {exc}")
            time.sleep(args.stagger)
        poll_all(state)
        write_summary(state)
        active = [r for r in state["claims"].values() if r.get("status") not in TERMINAL]
        print(f"[overnight] {now()} active={len(active)} dispatched={len(state['claims'])}")
        if not active and not plan(state, args.min_p)[0]:
            break
        time.sleep(args.poll_seconds)
    write_summary(state)
    print(f"[overnight] stopped. Summary: {SUMMARY}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--plan", action="store_true", help="show dispatch order + skips, dispatch nothing")
    ap.add_argument("--dispatch-one", metavar="CLAIM_ID", help="smoke-test: dispatch exactly one claim")
    ap.add_argument("--watch", action="store_true", help="overnight loop: dispatch waves + poll until done/deadline")
    ap.add_argument("--poll", action="store_true", help="poll all dispatched agents once")
    ap.add_argument("--summary", action="store_true", help="regenerate SUMMARY.md from state")
    ap.add_argument("--wave", type=int, default=6, help="max concurrent agents")
    ap.add_argument("--poll-seconds", type=int, default=600)
    ap.add_argument("--stagger", type=int, default=20, help="seconds between dispatches in a wave")
    ap.add_argument("--hours", type=float, default=10.0, help="watch deadline")
    ap.add_argument("--min-p", type=float, default=15.0, help="minimum P(resolvable) %% to dispatch")
    ap.add_argument("--model", default=MODEL, help="cloud agent model id for --dispatch-one (never gemini; logged per claim)")
    ap.add_argument("--models", default=None,
                    help="comma-separated model rotation for --watch, e.g. claude-fable-5-1,composer-2.5 "
                         "(round-robin over P-sorted queue = stratified mix: every model gets high- and low-P claims)")
    args = ap.parse_args()

    state = load_state()
    if args.plan:
        todo, skipped = plan(state, args.min_p)
        print(f"model={args.model} (no Gemini) · TODO {len(todo)} in P order:")
        for e in todo:
            print(f"  {e['claim_id']}  P={e.get('p_resolvable_percent')}%  slot {e['slot']}  {e['proposed_slug']}")
        print(f"SKIPPED {len(skipped)}:")
        for e, why in skipped:
            print(f"  {e['claim_id']}: {why}")
        return 0
    if args.dispatch_one:
        import yaml
        slots = yaml.safe_load(QUEUE.read_text())["slots"]
        entry = next((s for s in slots if s["claim_id"] == args.dispatch_one), None)
        if entry is None:
            print(f"{args.dispatch_one} not in loop queue", file=sys.stderr)
            return 1
        dispatch_one(state, entry, model=args.model)
        write_summary(state)
        return 0
    if args.poll:
        poll_all(state)
        write_summary(state)
        for cid, r in sorted(state["claims"].items()):
            print(f"  {cid}: {r.get('status')} pr={r.get('pr_url') or 'pending'}")
        return 0
    if args.summary:
        print(write_summary(state))
        return 0
    if args.watch:
        return cmd_watch(args)
    ap.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())

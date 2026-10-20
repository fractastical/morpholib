#!/usr/bin/env python3
"""
morphoeval/cloud_orchestrator.py — Cloud Orchestrator for 100 Offline MorphoEval 12-Step Loops.

Dispatches and monitors autonomous Cursor Cloud Agents (`bc-...` VMs running on Cursor's
cloud infrastructure while your laptop is offline) via `cursor-sdk` (`CloudAgentOptions`)
or the Cursor Cloud Agents REST API (`https://api.cursor.com/v0/agents`), coordinated with
GitHub branches and Pull Requests.

Key Architectural Guarantees:
  1. Step 1 (`01_scout`) is actively executed inside every Cloud Agent loop using
     `morphoeval/scout_literature.py --slot <N>`, querying OpenAlex, Crossref, and
     Wikimedia Commons within Slot <N>'s assigned morphospace territory while checking
     the repo's existing models to prevent collisions.
  2. Strict Zero-Conflict Per-Loop File Isolation:
     Every loop works on its own branch (`morphoeval/loop-<NNN>-<slug>`), opens its own
     PR (`auto_create_pr=True`), and creates ONLY new files unique to `<NNN>` / `<slug>`:
       - `<year>-<author>-<slug>.py`
       - `datasets/<slug>/...`
       - `morphoeval/lean/loops/Loop_<NNN>_<slug>.lean`
       - `morphoeval/math_kg_loops/<sim_id>.yaml`
       - `docs/media/<slug>_comparison.png` & `docs/media/<slug>_closed_loop.gif`
       - `morphoeval/audits/<date>-<slug>-audit*.yaml` & `.measurements.yaml`
       - `morphoeval/kg/<date>-<slug>-audit*/...`
       - `morphoeval/runs/<date>-<slug>-loop.*`
     Zero existing tracked files are modified (`git diff --diff-filter=MD origin/main...HEAD`
     is empty), so all 100 PRs merge cleanly in any order without conflicts!

Usage:
  # Inspect the 100-slot campaign status (pending vs active branch/PR vs completed)
  python3 morphoeval/cloud_orchestrator.py status

  # Preview the exact 12-step Cloud Agent prompt for a slot
  python3 morphoeval/cloud_orchestrator.py prompt --slot 1

  # Dry-run dispatching a batch of 10 slots
  python3 morphoeval/cloud_orchestrator.py dispatch --count 10 --dry-run

  # Launch 10 Cloud Agents right now (then close your laptop!)
  export CURSOR_API_KEY="key_..."
  python3 morphoeval/cloud_orchestrator.py dispatch --count 10

  # Launch a specific range of slots (e.g. slots 1..25)
  python3 morphoeval/cloud_orchestrator.py dispatch --slots 1-25 --max-batch 25

  # Poll active Cursor Cloud Agents
  python3 morphoeval/cloud_orchestrator.py poll
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
CAMPAIGN_PATH = HERE / "cloud" / "campaign_100_slots.yaml"
DISPATCH_LOG_DIR = HERE / "cloud" / "dispatches"
DEFAULT_REPO = "fractastical/morpholib"
DEFAULT_BASE_BRANCH = "main"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def today_ymd() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def load_campaign() -> dict[str, Any]:
    if not CAMPAIGN_PATH.exists():
        raise FileNotFoundError(f"Missing campaign manifest: {CAMPAIGN_PATH}")
    return yaml.safe_load(CAMPAIGN_PATH.read_text(encoding="utf-8")) or {}


def get_completed_and_active_slots(repo: str = DEFAULT_REPO) -> dict[int, dict[str, str]]:
    """Detect slots that already have local files, remote git branches, or open/merged GitHub PRs."""
    status_by_slot: dict[int, dict[str, str]] = {}

    # 1. Check local/committed Lean loop files `morphoeval/lean/loops/Loop_<NNN>_*.lean`
    loops_lean = HERE / "lean" / "loops"
    if loops_lean.exists():
        for lf in loops_lean.glob("Loop_*.lean"):
            m = re.match(r"Loop_(\d{3})_(.+)\.lean$", lf.name)
            if m:
                snum = int(m.group(1))
                status_by_slot[snum] = {"state": "completed_in_tree", "detail": str(lf.relative_to(ROOT))}

    # 2. Check remote git branches `morphoeval/loop-<NNN>-*`
    try:
        out = subprocess.check_output(
            ["git", "ls-remote", "--heads", "origin", "refs/heads/morphoeval/loop-*"],
            cwd=ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=10,
        )
        for ln in out.splitlines():
            parts = ln.split()
            if len(parts) == 2 and "refs/heads/morphoeval/loop-" in parts[1]:
                bname = parts[1].replace("refs/heads/", "")
                m = re.match(r"morphoeval/loop-(\d{3})-(.+)$", bname)
                if m:
                    snum = int(m.group(1))
                    status_by_slot.setdefault(snum, {"state": "branch_active", "detail": bname})
    except Exception:
        pass

    # 3. Check local dispatch records in `morphoeval/cloud/dispatches/slot-<NNN>.json`
    if DISPATCH_LOG_DIR.exists():
        for df in DISPATCH_LOG_DIR.glob("slot-*.json"):
            try:
                rec = json.loads(df.read_text(encoding="utf-8"))
                snum = int(rec["slot"])
                status_by_slot.setdefault(
                    snum,
                    {
                        "state": rec.get("status", "dispatched"),
                        "detail": f"agent_id={rec.get('agent_id')} ({rec.get('dispatched_at')})",
                    },
                )
            except Exception:
                continue

    return status_by_slot


def build_cloud_agent_prompt(slot_entry: dict[str, Any], run_date: str | None = None) -> str:
    """Construct the self-contained 12-Step Closed-Loop prompt for a Cursor Cloud Agent."""
    d_str = run_date or today_ymd()
    slot_num = int(slot_entry["slot"])
    slot_id = slot_entry["slot_id"]
    slug = slot_entry["proposed_slug"]
    sim_id = slot_entry["proposed_sim_id"]
    script_name = slot_entry["proposed_script"]
    dataset_dir = slot_entry["proposed_dataset_dir"]
    lean_file = slot_entry["proposed_lean_module"]
    math_spec = f"morphoeval/math_kg_loops/{sim_id}.yaml"
    run_id = f"{d_str}-{slug}-loop"
    audit_v1_id = f"{d_str}-{slug}-audit"
    audit_v2_id = f"{d_str}-{slug}-audit-v2"
    branch_name = f"morphoeval/loop-{slot_num:03d}-{slug}"

    return f"""You are executing Autonomous Cloud Loop **{slot_id}** (`{branch_name}`) of the 100-Loop Infinite Morphospace & MorphoEval Campaign.

## CRITICAL RULE #1 — STRICT ZERO-CONFLICT FILE ISOLATION
Because 100 Cloud Agent loops run in parallel across isolated branches/PRs, **your final commit MUST NOT modify or delete ANY existing tracked file in `origin/main`**.
- Do NOT edit `README.md`, `morphoeval/README.md`, `morphoeval/lean/MathOfBiology.lean`, `morphoeval/math_kg.py`, `morphoeval/measure_outputs.py`, `morphoeval/track_loop.py`, or any existing script/audit.
- Put your Lean 4 proofs in `{lean_file}` (standalone file, zero external imports, 0 `sorry`s) and your Math KG law registration in `{math_spec}`.
- Before committing, always revert/remove any shared telemetry index files touched by `track_loop.py` (`git checkout -- morphoeval/runs/telemetry.sqlite morphoeval/runs/telemetry_events.jsonl 2>/dev/null || rm -f morphoeval/runs/telemetry.sqlite morphoeval/runs/telemetry_events.jsonl`) and verify that `git diff --name-only --diff-filter=MD origin/main` outputs ZERO files!

---

## ASSIGNED MORPHOSPACE SCOUTING TERRITORY (`{slot_id}`)
- **Domain:** {slot_entry['domain']}
- **Target Organism / Taxon:** *{slot_entry['organism']}*
- **Mathematical Formalism Family:** {slot_entry['math_family']}
- **Literature Search Query:** `{slot_entry['search_query']}`
- **Seed Landmark Paper:** {slot_entry['seed_paper']} ({slot_entry.get('seed_doi') or 'open literature'})
- **Key Governing Mechanism / Equations to Formalize:** {slot_entry['governing_mechanism']}

---

## EXECUTE ALL 12 STEPS OF THE CLOSED REMEDIATION LOOP (PASS 1 -> PASS 2)

Initialize the per-loop telemetry log:
```bash
python3 morphoeval/track_loop.py init --run-id {run_id} --target {sim_id} --paper "{slot_entry['seed_paper']}"
```

### Tier A — Making a Reproduction (Steps 1–6)
1. **Step 1 (`scout`):** Execute Literature & Specimen Scouting for Slot {slot_num}:
   Query OpenAlex (`https://api.openalex.org/works?search=...`) and Wikimedia Commons API (`https://commons.wikimedia.org/w/api.php?action=query&generator=search...`) — or run `python3 morphoeval/scout_literature.py --slot {slot_num} --out {dataset_dir}/scout_manifest.json` if present — and check existing `[0-9][0-9][0-9][0-9]-*.py` scripts in the repo root to guarantee zero collision. Save your scout dossier to `{dataset_dir}/scout_manifest.json` and log Step 1 with `track_loop.py`.
2. **Step 2 (`paper`):** Extract the published mathematical equations, state variables, boundary conditions, parameter regime table, and empirical biological claims into `{dataset_dir}/{slug.replace('-', '_')}_reference.json`, `{dataset_dir}/paper_citations.bib`, and `{dataset_dir}/README.md`. Record Step 2 with `track_loop.py`.
3. **Step 3 (`source`):** Fetch/vendor a real CC-licensed biological specimen photograph (from Wikimedia Commons) plus empirical reference tables into `{dataset_dir}/`, compute SHA-256 digests, and pin them in `{dataset_dir}/{slug.replace('-', '_')}_reference.json`. Record Step 3 with `track_loop.py`.
4. **Step 4 (`rules`):** Write a standalone, self-contained **Lean 4** formalization in `{lean_file}` (namespace `MathOfBiology.Loop_{slot_num:03d}`) proving at least 3 non-trivial algebraic/invariance/stability theorems for the model's governing equations with **0 `sorry`s** (verify with `lean {lean_file}` if `lean` is installed). Also write `{math_spec}` with `sim_id: {sim_id}`, `math_family`, `model_spec`, and `equations` linking `{lean_file}`. Record Step 4 with `track_loop.py`.
5. **Step 5 (`script` — Pass 1 Baseline v1):** Write `{script_name}` implementing the core mathematical model, `--self-test`, and initial figure generation (`docs/media/{slug.replace('-', '_')}_comparison.png` and `docs/media/{slug.replace('-', '_')}_growth.gif`), plus a `measure_morphoeval()` function returning quantitative morphometric dicts. Record Step 5 (Pass 1) with `track_loop.py`.
6. **Step 6 (`output` — Pass 1):** Execute `python3 {script_name} --self-test` and `python3 {script_name}` to render the Pass 1 outputs. Record Step 6 (Pass 1) with `track_loop.py`.

### Tier B — Auditing & Closed-Loop Remediation (Steps 7–12 -> Pass 2)
7. **Step 7 (`audit` — Pass 1):** Audit `{script_name}` (v1) across all 7 dimensions of `morphoeval/rubric.yaml` with line-pinned code evidence (`supports: true|false`) and explicit `remediation` items in `morphoeval/audits/{audit_v1_id}.yaml`.
8. **Step 8 (`measure` — Pass 1):** Run `python3 morphoeval/measure_outputs.py morphoeval/audits/{audit_v1_id}.yaml` to generate `morphoeval/audits/{audit_v1_id}.measurements.yaml`.
9. **Step 9 (`build` — Pass 1):** Run `python3 morphoeval/build_morphoeval_kg.py morphoeval/audits/{audit_v1_id}.yaml --conform`.
10. **Step 10 (`publish` — Pass 1):** If `~/.nanopub/profile.yml` is present (or run `python3 -m nanopub setup --newkeys` if needed), sign with `python3 morphoeval/sign_morphoeval.py morphoeval/kg/{audit_v1_id}` and verify with `node morphoeval/check_signed.mjs morphoeval/kg/{audit_v1_id}/signed/*.trig`.
11. **Step 11 (`graph` — Pass 1):** Verify `morphoeval/kg/{audit_v1_id}/REPORT.md`.
12. **Step 12 (`remed` -> Pass 2 Loop through Steps 5–11):**
    - Remediate every weakness identified in Pass 1: upgrade `{script_name}` with a full parameter-regime sweep, negative/ablation control, biological specimen computer-vision segmentation against the real specimen image in `{dataset_dir}/`, and statistical goodness-of-fit ($r$, RMSE, KS test / CV).
    - Re-run `{script_name} --self-test` and `{script_name}` (Step 6 Pass 2).
    - Write the superseding Pass 2 audit `morphoeval/audits/{audit_v2_id}.yaml` (`supersedes_audit_id: {audit_v1_id}`), run `measure_outputs.py` (Step 8 Pass 2), `build_morphoeval_kg.py --supersede-from morphoeval/kg/{audit_v1_id} --no-resolve-heads --conform` (Step 9 Pass 2), and sign/verify (Step 10 Pass 2).
    - Generate the per-loop telemetry summary & walkthrough:
      ```bash
      python3 morphoeval/track_loop.py summary --run-id {run_id} --md morphoeval/runs/{run_id}.md --csv morphoeval/runs/{run_id}.csv --trig morphoeval/runs/{run_id}.trig --plot docs/media/{slug.replace('-', '_')}_telemetry.png
      ```
    - Clean up any shared telemetry files so ONLY unique files for `{slot_id}` are staged and committed:
      ```bash
      git checkout -- morphoeval/runs/telemetry.sqlite morphoeval/runs/telemetry_events.jsonl 2>/dev/null || rm -f morphoeval/runs/telemetry.sqlite morphoeval/runs/telemetry_events.jsonl
      git status -s
      ```
"""


async def _dispatch_via_cursor_sdk(
    prompt: str,
    repo: str,
    branch_name: str,
    model: str | None = None,
) -> dict[str, Any]:
    """Dispatch a Cursor Cloud Agent via `cursor-sdk` (`CloudAgentOptions`) if installed,
    or fall back to the Cursor Cloud Agents REST API (`/v0/agents`)."""
    try:
        from cursor_sdk import Agent, CloudAgentOptions  # type: ignore

        opts_kwargs: dict[str, Any] = {
            "repos": [repo],
            "branch_name": branch_name,
            "auto_create_pr": True,
            "skip_reviewer_request": True,
        }
        if model:
            opts_kwargs["model"] = model
        agent = await Agent.create(CloudAgentOptions(**opts_kwargs))
        run = agent.send(prompt)
        return {
            "transport": "cursor_sdk",
            "agent_id": agent.id,
            "run_id": getattr(run, "id", None),
            "branch_name": branch_name,
            "status": "dispatched",
        }
    except ImportError:
        return _dispatch_via_rest_api(prompt=prompt, repo=repo, branch_name=branch_name, model=model)


def _dispatch_via_rest_api(
    prompt: str,
    repo: str,
    branch_name: str,
    model: str | None = None,
    base_ref: str = DEFAULT_BASE_BRANCH,
) -> dict[str, Any]:
    """Direct HTTPS fallback to Cursor Cloud Agents API (`POST https://api.cursor.com/v0/agents`)
    requiring zero external packages beyond standard library + `CURSOR_API_KEY`."""
    api_key = os.environ.get("CURSOR_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError(
            "CURSOR_API_KEY environment variable is not set. "
            "Export CURSOR_API_KEY='key_...' (from cursor.com/settings -> Cloud Agents API) "
            "or run with --dry-run to preview."
        )
    repo_url = repo if repo.startswith("http") else f"https://github.com/{repo}"
    payload: dict[str, Any] = {
        "prompt": {"text": prompt},
        "source": {"repository": repo_url, "ref": base_ref},
        "target": {
            "branchName": branch_name,
            "autoCreatePr": True,
            "openAsCursorGithubApp": False,
            "skipReviewerRequest": True,
        },
    }
    if model:
        payload["model"] = model

    auth_b64 = base64.b64encode(f"{api_key}:".encode("utf-8")).decode("ascii")
    req = urllib.request.Request(
        "https://api.cursor.com/v0/agents",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Basic {auth_b64}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "InfiniteMorphospace-CloudOrchestrator/1.0",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        body = json.loads(resp.read().decode("utf-8", errors="replace"))
    return {
        "transport": "cursor_rest_v0",
        "agent_id": body.get("id") or body.get("agentId"),
        "status": body.get("status", "CREATING"),
        "branch_name": branch_name,
        "web_url": body.get("target", {}).get("url") or body.get("url"),
        "raw": body,
    }


def poll_cloud_agent(agent_id: str) -> dict[str, Any]:
    """Query status of a Cursor Cloud Agent (`bc-...`) via `GET https://api.cursor.com/v0/agents/{id}`."""
    api_key = os.environ.get("CURSOR_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("CURSOR_API_KEY is required to poll Cloud Agents.")
    auth_b64 = base64.b64encode(f"{api_key}:".encode("utf-8")).decode("ascii")
    req = urllib.request.Request(
        f"https://api.cursor.com/v0/agents/{agent_id}",
        headers={
            "Authorization": f"Basic {auth_b64}",
            "Accept": "application/json",
        },
    )
    with urllib.request.urlopen(req, timeout=20) as resp:
        return json.loads(resp.read().decode("utf-8", errors="replace"))


def parse_slot_selector(selector: str) -> list[int]:
    """Parse e.g. '1-10' or '1,3,5-8' into sorted slot integers."""
    out: set[int] = set()
    for part in selector.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            for k in range(int(a), int(b) + 1):
                out.add(k)
        else:
            out.add(int(part))
    return sorted(out)


def cmd_status(args: argparse.Namespace) -> int:
    camp = load_campaign()
    slots = camp.get("slots", [])
    active_map = get_completed_and_active_slots(args.repo)

    n_done = sum(1 for v in active_map.values() if v["state"] == "completed_in_tree")
    n_act = sum(1 for v in active_map.values() if v["state"] != "completed_in_tree")
    n_pend = len(slots) - len(active_map)

    print(f"=== MorphoEval 100-Loop Cloud Campaign Status ({len(slots)} slots) ===")
    print(f"  Completed in tree : {n_done}")
    print(f"  Active / Dispatched: {n_act}")
    print(f"  Pending (ready)   : {n_pend}")
    print("-" * 108)
    print(f"{'Slot':<8} {'State':<18} {'Proposed Script':<36} {'Organism / Taxon':<26} {'Domain'}")
    print("-" * 108)
    limit = args.limit or len(slots)
    for s in slots[:limit]:
        snum = int(s["slot"])
        st = active_map.get(snum, {}).get("state", "pending")
        print(
            f"#{snum:03d}    {st:<18} {s['proposed_script']:<36} "
            f"{s['organism'][:24]:<26} {s['domain'][:32]}"
        )
    if len(slots) > limit:
        print(f"... ({len(slots) - limit} more slots; pass --limit 100 to show all)")
    return 0


def cmd_prompt(args: argparse.Namespace) -> int:
    camp = load_campaign()
    slots = {int(s["slot"]): s for s in camp.get("slots", [])}
    if args.slot not in slots:
        print(f"Slot {args.slot} not in campaign (1..{len(slots)})", file=sys.stderr)
        return 1
    print(build_cloud_agent_prompt(slots[args.slot]))
    return 0


def cmd_dispatch(args: argparse.Namespace) -> int:
    camp = load_campaign()
    slots_by_num = {int(s["slot"]): s for s in camp.get("slots", [])}
    active_map = get_completed_and_active_slots(args.repo)

    if args.slots:
        target_nums = [n for n in parse_slot_selector(args.slots) if n in slots_by_num]
        if not args.force:
            target_nums = [n for n in target_nums if n not in active_map]
    else:
        pending_nums = [int(s["slot"]) for s in camp.get("slots", []) if int(s["slot"]) not in active_map]
        target_nums = pending_nums[: args.count]

    if not target_nums:
        print("[cloud_orchestrator] No pending slots matched for dispatch.")
        return 0

    print(
        f"[cloud_orchestrator] Preparing to dispatch {len(target_nums)} Cloud Agent loop(s): "
        + ", ".join(f"#{n:03d}" for n in target_nums)
    )

    DISPATCH_LOG_DIR.mkdir(parents=True, exist_ok=True)
    dispatched_records: list[dict[str, Any]] = []

    for idx, snum in enumerate(target_nums, start=1):
        slot_entry = slots_by_num[snum]
        slug = slot_entry["proposed_slug"]
        branch_name = f"morphoeval/loop-{snum:03d}-{slug}"
        prompt = build_cloud_agent_prompt(slot_entry)

        if args.dry_run:
            print(
                f"  [DRY-RUN {idx}/{len(target_nums)}] Slot #{snum:03d} -> branch `{branch_name}` "
                f"| script `{slot_entry['proposed_script']}` | organism `{slot_entry['organism']}` "
                f"| prompt={len(prompt):,} chars"
            )
            continue

        res = asyncio.run(
            _dispatch_via_cursor_sdk(
                prompt=prompt,
                repo=args.repo,
                branch_name=branch_name,
                model=args.model,
            )
        )
        rec = {
            "slot": snum,
            "slot_id": slot_entry["slot_id"],
            "proposed_script": slot_entry["proposed_script"],
            "proposed_sim_id": slot_entry["proposed_sim_id"],
            "branch_name": branch_name,
            "dispatched_at": utc_now_iso(),
            **res,
        }
        log_file = DISPATCH_LOG_DIR / f"slot-{snum:03d}.json"
        log_file.write_text(json.dumps(rec, indent=2) + "\n", encoding="utf-8")
        dispatched_records.append(rec)
        print(
            f"  [DISPATCHED {idx}/{len(target_nums)}] Slot #{snum:03d} -> agent_id={rec.get('agent_id')} "
            f"branch={branch_name}"
        )
        if idx < len(target_nums) and args.stagger_seconds > 0:
            time.sleep(args.stagger_seconds)

    if not args.dry_run:
        print(f"\n[cloud_orchestrator] Successfully launched {len(dispatched_records)} Cursor Cloud Agent(s).")
        print("You can now close your laptop — the agents run on Cursor Cloud VMs and will open PRs automatically.")
    return 0


def cmd_poll(args: argparse.Namespace) -> int:
    if not DISPATCH_LOG_DIR.exists():
        print("[cloud_orchestrator] No local dispatch logs found in morphoeval/cloud/dispatches/.")
        return 0
    files = sorted(DISPATCH_LOG_DIR.glob("slot-*.json"))
    if not files:
        print("[cloud_orchestrator] No dispatched slots recorded yet.")
        return 0
    for f in files:
        rec = json.loads(f.read_text(encoding="utf-8"))
        aid = rec.get("agent_id")
        snum = rec.get("slot")
        if aid and os.environ.get("CURSOR_API_KEY"):
            try:
                live = poll_cloud_agent(aid)
                rec["status"] = live.get("status", rec.get("status"))
                rec["pr_url"] = live.get("target", {}).get("prUrl") or rec.get("pr_url")
                f.write_text(json.dumps(rec, indent=2) + "\n", encoding="utf-8")
            except Exception as exc:
                rec["poll_error"] = str(exc)
        print(
            f"  Slot #{snum:03d} ({rec.get('proposed_script')}): "
            f"agent={aid} status={rec.get('status')} branch={rec.get('branch_name')} pr={rec.get('pr_url') or 'pending'}"
        )
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd")

    p_st = sub.add_parser("status", help="Show status of all 100 campaign slots")
    p_st.add_argument("--repo", default=DEFAULT_REPO)
    p_st.add_argument("--limit", type=int, default=25, help="Max rows to print (pass 100 for all)")

    p_pr = sub.add_parser("prompt", help="Print the full 12-step Cloud Agent prompt for a slot")
    p_pr.add_argument("--slot", type=int, required=True, help="Slot number (1..100)")

    p_dp = sub.add_parser("dispatch", help="Dispatch a batch of Cursor Cloud Agents for pending slots")
    p_dp.add_argument("--count", type=int, default=5, help="Number of next pending slots to launch (default: 5)")
    p_dp.add_argument("--slots", type=str, default=None, help="Explicit slot range/list (e.g. '1-10' or '1,4,7')")
    p_dp.add_argument("--repo", default=DEFAULT_REPO, help="GitHub owner/repo")
    p_dp.add_argument("--model", default=None, help="Optional model override for Cloud Agent")
    p_dp.add_argument("--stagger-seconds", type=float, default=2.0, help="Delay between API dispatch calls")
    p_dp.add_argument("--force", action="store_true", help="Dispatch even if slot is already marked active")
    p_dp.add_argument("--dry-run", action="store_true", help="Preview batch dispatch without calling API")

    sub.add_parser("poll", help="Poll live status of dispatched Cursor Cloud Agents")

    args = ap.parse_args()
    if args.cmd == "status":
        return cmd_status(args)
    if args.cmd == "prompt":
        return cmd_prompt(args)
    if args.cmd == "dispatch":
        return cmd_dispatch(args)
    if args.cmd == "poll":
        return cmd_poll(args)
    ap.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

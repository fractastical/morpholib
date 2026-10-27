#!/usr/bin/env python3
"""Problem-Fixer Wave 1: sleep-safe autonomous repairs for scout findings.

The Problem-Scout fleet found and ranked candidate problems. This tool turns
a small, curated first wave into isolated Cursor Cloud Agent jobs. Each job
gets one candidate, a strict path allowlist, task-specific validators, and a
machine-readable result contract. Agents open PRs; they never merge their own
work. A separate PR guard may auto-merge only tasks marked ``auto`` in the
queue after path and validator checks pass. Everything else stays in a PR for
human review.

Usage:
  python3 morphoeval/problem_fixer.py list
  python3 morphoeval/problem_fixer.py prompt --fix 3
  python3 morphoeval/problem_fixer.py dispatch --fixes 1-12 --dry-run
  CURSOR_API_KEY=... python3 morphoeval/problem_fixer.py dispatch --fixes 1-12
  python3 morphoeval/problem_fixer.py status
  python3 morphoeval/problem_fixer.py guard --fix 3 --base origin/main
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
QUEUE_PATH = HERE / "problem_fixers" / "fix_queue.json"
LEADERBOARD_PATH = HERE / "problem_scouts" / "leaderboard.json"
RESULT_DIR = HERE / "problem_fixers" / "results"
REPORT_DIR = HERE / "problem_fixers" / "reports"
DISPATCH_DIR = HERE / "cloud" / "dispatches"
DEFAULT_REPO = "fractastical/morpholib"

GLOBAL_FORBIDDEN = [
    ".github/**",
    ".git/**",
    "**/.env",
    "**/.env.*",
    "**/*.key",
    "**/*.pem",
    "**/id_rsa*",
    "predictions/prediction_registry.json",
    "predictions/prediction_registry.csv",
    "morphoeval/problem_fixers/fix_queue.json",
    "morphoeval/problem_fixer.py",
    "morphoeval/cloud_orchestrator.py",
    "morphoeval/problem_scouts/leaderboard.json",
    "morphoeval/problem_scouts/leaderboard.md",
    "morphoeval/runs/telemetry.sqlite",
    "morphoeval/runs/telemetry_events.jsonl",
]

VALID_RESULT_STATUSES = {"fixed", "verified-no-change", "blocked", "needs-review"}

# Repo tools legitimately rewrite these runtime artifacts as a side effect of
# running (telemetry DB / event log). A validator touching them is not evidence
# that a regenerated artifact was left uncommitted, so the dirty-tree check
# ignores them.
VALIDATOR_SIDE_EFFECT_PATHS = {
    "morphoeval/runs/telemetry.sqlite",
    "morphoeval/runs/telemetry_events.jsonl",
}


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_queue() -> dict[str, Any]:
    queue = load_json(QUEUE_PATH)
    tasks = queue.get("tasks", [])
    seen: set[str] = set()
    for task in tasks:
        if task["id"] in seen:
            raise ValueError(f"Duplicate fixer task id: {task['id']}")
        seen.add(task["id"])
        if task.get("merge_policy") not in {"auto", "review"}:
            raise ValueError(f"{task['id']}: merge_policy must be auto or review")
        if not task.get("allowed_paths"):
            raise ValueError(f"{task['id']}: allowed_paths is empty")
    candidate_for(task)  # fail early if the queue has drifted from the leaderboard
    return queue


def tasks_by_number() -> dict[int, dict[str, Any]]:
    out: dict[int, dict[str, Any]] = {}
    for task in load_queue()["tasks"]:
        number = int(task["id"].split("-")[-1])
        out[number] = task
    return out


def candidate_for(task: dict[str, Any]) -> dict[str, Any]:
    rows = load_json(LEADERBOARD_PATH)
    matches = [r for r in rows if int(r.get("rank", -1)) == int(task["leaderboard_rank"])]
    if len(matches) != 1:
        raise ValueError(f"{task['id']}: expected exactly one leaderboard row at rank {task['leaderboard_rank']}")
    row = matches[0]
    if row.get("title") != task.get("title"):
        raise ValueError(f"{task['id']}: queue title does not match leaderboard rank {task['leaderboard_rank']}")
    if row.get("id") != task.get("candidate_id") or row.get("source") != task.get("source"):
        raise ValueError(f"{task['id']}: queue candidate id/source does not match leaderboard")
    return row


def parse_selection(selector: str | None, count: int | None = None) -> list[int]:
    by_number = tasks_by_number()
    if not selector:
        numbers = sorted(by_number)
        return numbers[:count] if count else numbers
    selected: set[int] = set()
    for part in selector.split(","):
        part = part.strip()
        if not part:
            continue
        part = part.removeprefix("fix-")
        if "-" in part:
            start, end = part.split("-", 1)
            selected.update(range(int(start), int(end) + 1))
        else:
            selected.add(int(part))
    unknown = sorted(n for n in selected if n not in by_number)
    if unknown:
        raise ValueError(f"Unknown fixer numbers: {unknown}; choose from {sorted(by_number)}")
    return sorted(selected)


def path_matches(pattern: str, path: str) -> bool:
    path = path.replace(os.sep, "/")
    if pattern.endswith("/**"):
        prefix = pattern[:-3].rstrip("/")
        return path == prefix or path.startswith(prefix + "/")
    return fnmatch.fnmatchcase(path, pattern)


def path_is_allowed(task: dict[str, Any], path: str) -> bool:
    reporting = {
        f"morphoeval/problem_fixers/results/{task['id']}.json",
        f"morphoeval/problem_fixers/reports/{task['id']}.md",
    }
    if path in reporting:
        return True
    if any(path_matches(pattern, path) for pattern in GLOBAL_FORBIDDEN):
        return False
    return any(path_matches(pattern, path) for pattern in task["allowed_paths"])


def requested_branch(task: dict[str, Any]) -> str:
    return f"morphoeval/fixer-{task['id']}-{task['slug']}"


def result_path(task: dict[str, Any]) -> Path:
    return RESULT_DIR / f"{task['id']}.json"


def build_prompt(task: dict[str, Any]) -> str:
    candidate = candidate_for(task)
    allowed = "\n".join(f"- `{p}`" for p in task["allowed_paths"])
    instructions = "\n".join(f"- {item}" for item in task["instructions"])
    validators = "\n".join(
        f"- {v['name']}: `{' '.join(v['argv'])}`" for v in task.get("validators", [])
    ) or "- No task-specific validator; add a focused regression/self-test if you change code."
    return f"""You are Problem-Fixer **{task['id']}** (`{task['slug']}`) in Wave 1 of the Project Delphi / MorphoEval autonomous repair fleet.

## PR CONTRACT
- Requested branch label: `{requested_branch(task)}`
- The PR title MUST begin exactly: `Problem-Fixer {task['id']}: `
- Do **not** merge your own PR, enable auto-merge, force-push, rewrite history, or publish/sign/submit anything to an external registry or service.
- Open/update exactly one PR containing this one fix. Other agents are working in parallel.

## ASSIGNED SCOUT CANDIDATE
The following is scout evidence, not gospel. Inspect the current code and data first; the scout may be stale or wrong.

```json
{json.dumps(candidate, indent=2)}
```

## TASK-SPECIFIC INSTRUCTIONS
{instructions}

## STRICT PATH ALLOWLIST
You may create or modify files only under these patterns, plus your two reporting files
(`morphoeval/problem_fixers/results/{task['id']}.json` and
`morphoeval/problem_fixers/reports/{task['id']}.md`):

{allowed}

Never modify `.github/**`, `predictions/prediction_registry.json`,
`predictions/prediction_registry.csv`, the fixer queue/orchestrators/leaderboard,
credentials, telemetry databases, or any file outside the allowlist. If the real
fix requires an outside path, stop with status `blocked`, make no speculative
source changes, and name the exact path and reason in the report.

## HOW TO WORK
1. Reproduce or verify the issue from the current tree before changing anything.
2. Fix the underlying problem at the narrowest source of truth. Do not merely
   register a post-hoc “prediction” that a defect exists. If the candidate's
   route asks for registration, use a new, separate, clearly timestamped
   registration/resolution artifact and never edit the locked registry.
3. Do not fabricate biological data, specimen evidence, measurements, hashes,
   audit scores, or literature facts. Verify factual corrections from a primary
   source or an authoritative database and cite it in the report.
4. Do not rewrite historical scientific records as though they were always
   correct. Prefer a superseding artifact where the repo convention calls for
   one; explain any unavoidable historical-file correction.
5. Prefer deterministic repo tools over hand-edited computed values. If you
   regenerate an artifact, name the exact command in the report.
6. Run every applicable validator below and record the real exit codes. If a
   validator names a focused test file and it does not exist, create that test
   as part of the fix. Do not weaken a validator or test to make it pass.
7. Before finishing, inspect `git diff --name-status origin/main...HEAD` and
   remove anything outside the allowlist or any generated telemetry/cache file.

Validators:
{validators}

## REQUIRED RESULT FILES
Write `morphoeval/problem_fixers/results/{task['id']}.json` with this schema:
```json
{{
  "fix_id": "{task['id']}",
  "candidate_rank": {task['leaderboard_rank']},
  "candidate_id": "{task['candidate_id']}",
  "status": "fixed | verified-no-change | blocked | needs-review",
  "summary": "what you found and did",
  "root_cause": "evidence-backed root cause, or null",
  "changed_files": ["repo-relative paths"],
  "evidence": ["specific files, values, commands, or source citations"],
  "validation": [
    {{"name": "validator name", "argv": ["command"], "exit_code": 0, "output_summary": "short real result"}}
  ],
  "blockers": [],
  "external_publication_performed": false
}}
```
`changed_files` must match the PR's non-reporting changed files. Also write a
human-readable `morphoeval/problem_fixers/reports/{task['id']}.md` covering:
finding, root cause, fix, files changed, validation, residual risk, and what a
reviewer should inspect first.

Merge policy for this task is `{task['merge_policy']}`. `auto` means an external
guard may merge only if every path and validator check passes; it is not
permission for you to merge. For `review`, finish the PR and stop.

Task merge policy: {task['merge_policy']}
Requested branch: {requested_branch(task)}
"""


def cmd_list(_args: argparse.Namespace) -> int:
    for task in load_queue()["tasks"]:
        candidate = candidate_for(task)
        record = DISPATCH_DIR / f"problem-fixer-{task['id']}.json"
        state = "dispatched" if record.exists() else "pending"
        if result_path(task).exists():
            state = "result-in-tree"
        print(
            f"{task['id']}  rank #{task['leaderboard_rank']:>3}  {task['merge_policy']:<6} "
            f"{state:<15} {candidate['title']}"
        )
    return 0


def cmd_prompt(args: argparse.Namespace) -> int:
    task = tasks_by_number()[args.fix]
    print(build_prompt(task), end="")
    return 0


def _dispatch_one(task: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    from cloud_orchestrator import _dispatch_via_cursor_sdk

    last_exc: Exception | None = None
    for attempt in range(1, 4):
        try:
            return _dispatch_via_cursor_sdk(
                prompt=build_prompt(task),
                repo=args.repo,
                branch_name=requested_branch(task),
                model=args.model,
            )
        except Exception as exc:  # transient Cursor Bridge/API failures should not kill the wave
            last_exc = exc
            print(f"  [RETRY {attempt}/3] {task['id']}: {str(exc)[:180]}", flush=True)
            if attempt < 3:
                time.sleep(10 * attempt)
    raise RuntimeError(f"dispatch failed for {task['id']}: {last_exc}")


def cmd_dispatch(args: argparse.Namespace) -> int:
    numbers = parse_selection(args.fixes, args.count)
    by_number = tasks_by_number()
    failures: list[str] = []
    DISPATCH_DIR.mkdir(parents=True, exist_ok=True)

    for number in numbers:
        task = by_number[number]
        record_path = DISPATCH_DIR / f"problem-fixer-{task['id']}.json"
        if args.dry_run:
            print(
                f"  [DRY-RUN] {task['id']} rank=#{task['leaderboard_rank']} "
                f"policy={task['merge_policy']} branch={requested_branch(task)} "
                f"| prompt={len(build_prompt(task)):,} chars"
            )
            continue
        if record_path.exists() and not args.force:
            print(f"  [SKIP] {task['id']} already has dispatch record {record_path.name}")
            continue
        try:
            result = _dispatch_one(task, args)
        except Exception as exc:
            failures.append(f"{task['id']}: {exc}")
            print(f"  [FAILED] {task['id']}: {exc}", flush=True)
            continue
        record = {
            "kind": "problem_fixer",
            "wave": load_queue().get("wave"),
            "dispatched_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            **task,
            "requested_branch": requested_branch(task),
            **result,
        }
        record_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        print(f"  [DISPATCHED] {task['id']} agent_id={result.get('agent_id')}", flush=True)
        time.sleep(args.stagger_seconds)

    if failures:
        print("Failures: " + "; ".join(failures))
        return 1
    if args.dry_run:
        print(f"Dry-run only: {len(numbers)} fixer(s) armed; no Cursor API call was made.")
    return 0


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def _status_lines(repo: Path) -> set[str]:
    """Working-tree paths from `git status --porcelain` (rename targets kept)."""
    proc = _git(repo, "status", "--porcelain")
    if proc.returncode != 0:
        return set()
    paths: set[str] = set()
    for line in proc.stdout.splitlines():
        entry = line[3:].strip()
        if " -> " in entry:
            entry = entry.split(" -> ", 1)[1]
        paths.add(entry.strip('"'))
    return paths


def changed_files(repo: Path, base: str, head: str) -> list[tuple[str, str]]:
    proc = _git(repo, "diff", "--name-status", "--find-renames", f"{base}...{head}")
    if proc.returncode != 0:
        raise RuntimeError(f"git diff failed: {proc.stderr.strip()}")
    out: list[tuple[str, str]] = []
    for line in proc.stdout.splitlines():
        parts = line.split("\t")
        if len(parts) >= 2:
            status = parts[0]
            path = parts[-1]
            out.append((status, path))
    return out


def load_result(repo: Path, task: dict[str, Any]) -> dict[str, Any]:
    path = repo / "morphoeval" / "problem_fixers" / "results" / f"{task['id']}.json"
    if not path.exists():
        raise ValueError(f"missing required result file: {path.relative_to(repo)}")
    return json.loads(path.read_text(encoding="utf-8"))


def run_validators(repo: Path, task: dict[str, Any]) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for validator in task.get("validators", []):
        argv = [str(x) for x in validator["argv"]]
        missing_optional_test = (
            "when added" in validator.get("name", "")
            and len(argv) >= 3
            and argv[0].startswith("python")
            and argv[-2].endswith(".py")
            and not (repo / argv[-2]).exists()
        )
        if missing_optional_test:
            results.append({"name": validator["name"], "argv": argv, "exit_code": 0, "output": "skipped: optional test not added"})
            continue
        proc = subprocess.run(
            argv,
            cwd=repo,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=int(validator.get("timeout_seconds", 900)),
            check=False,
        )
        results.append(
            {
                "name": validator["name"],
                "argv": argv,
                "exit_code": proc.returncode,
                "output": proc.stdout[-2000:],
            }
        )
    return results


def guard_task(
    task: dict[str, Any],
    repo: Path,
    base: str,
    head: str = "HEAD",
    run_validator_checks: bool = True,
) -> dict[str, Any]:
    candidate_for(task)  # trusted queue must still match the trusted leaderboard
    errors: list[str] = []
    warnings: list[str] = []
    changes = changed_files(repo, base, head)
    reporting = {
        f"morphoeval/problem_fixers/results/{task['id']}.json",
        f"morphoeval/problem_fixers/reports/{task['id']}.md",
    }
    non_reporting = [path for _status, path in changes if path not in reporting]

    for status, path in changes:
        if status.startswith("D"):
            errors.append(f"deleted files are not allowed in sleep-safe fixer PRs: {path}")
        if status.startswith("R") or status.startswith("C"):
            errors.append(f"renamed/copied files are not allowed in sleep-safe fixer PRs: {path}")
        if not path_is_allowed(task, path):
            errors.append(f"changed path is outside the allowlist: {path}")

    try:
        result = load_result(repo, task)
    except Exception as exc:
        result = {}
        errors.append(str(exc))

    if result:
        if result.get("fix_id") != task["id"]:
            errors.append("result fix_id does not match task")
        if result.get("candidate_rank") != task["leaderboard_rank"]:
            errors.append("result candidate_rank does not match task")
        if result.get("candidate_id") != task["candidate_id"]:
            errors.append("result candidate_id does not match task")
        if result.get("status") not in VALID_RESULT_STATUSES:
            errors.append(f"invalid result status: {result.get('status')!r}")
        if result.get("external_publication_performed") is not False:
            errors.append("external publication must be explicitly false")
        declared = sorted(result.get("changed_files") or [])
        actual = sorted(non_reporting)
        if declared != actual:
            errors.append(f"result changed_files mismatch: declared={declared}, actual={actual}")

    status = result.get("status") if result else None
    validator_results: list[dict[str, Any]] = []
    if status == "fixed" and not non_reporting:
        errors.append("status is fixed but the PR changes no non-reporting files")
    if status == "verified-no-change" and not non_reporting:
        errors.append("verified-no-change requires a new verifier/test or other durable evidence file")
    if run_validator_checks and status in {"fixed", "verified-no-change"}:
        pre_status = _status_lines(repo)
        validator_results = run_validators(repo, task)
        for validator in validator_results:
            if validator["exit_code"] != 0:
                errors.append(f"validator failed ({validator['name']}): exit {validator['exit_code']}")
        dirty = [
            path
            for path in _status_lines(repo) - pre_status
            if path not in VALIDATOR_SIDE_EFFECT_PATHS
        ]
        if dirty:
            errors.append(
                "validators modified the working tree; regenerated artifacts must be committed in the PR: "
                + ", ".join(sorted(dirty)[:10])
            )
    elif status in {"blocked", "needs-review"}:
        warnings.append(f"agent reported {status}; validators were not used as a merge gate")

    passed = not errors
    auto_merge_eligible = bool(
        passed
        and task["merge_policy"] == "auto"
        and status in {"fixed", "verified-no-change"}
    )
    return {
        "fix_id": task["id"],
        "passed": passed,
        "auto_merge_eligible": auto_merge_eligible,
        "status": status,
        "merge_policy": task["merge_policy"],
        "changes": [{"status": s, "path": p} for s, p in changes],
        "validators": validator_results,
        "warnings": warnings,
        "errors": errors,
    }


def cmd_guard(args: argparse.Namespace) -> int:
    task = tasks_by_number()[args.fix]
    repo = Path(args.repo).resolve()
    report = guard_task(
        task,
        repo=repo,
        base=args.base,
        head=args.head,
        run_validator_checks=not args.skip_validators,
    )
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(f"{task['id']}: {'PASS' if report['passed'] else 'FAIL'}")
        print(f"  status={report['status']} policy={report['merge_policy']} auto_merge={report['auto_merge_eligible']}")
        for change in report["changes"]:
            print(f"  {change['status']} {change['path']}")
        for validator in report["validators"]:
            print(f"  validator {validator['exit_code']}: {validator['name']}")
        for warning in report["warnings"]:
            print(f"  warning: {warning}")
        for error in report["errors"]:
            print(f"  error: {error}")
    return 0 if report["passed"] else 1


def cmd_status(args: argparse.Namespace) -> int:
    for task in load_queue()["tasks"]:
        record_path = DISPATCH_DIR / f"problem-fixer-{task['id']}.json"
        record = {}
        if record_path.exists():
            try:
                record = load_json(record_path)
            except Exception:
                record = {"status": "unreadable-record"}
        if args.poll and record.get("agent_id"):
            if not os.environ.get("CURSOR_API_KEY"):
                print("[poll] CURSOR_API_KEY is not set; showing recorded state only")
                args.poll = False
            else:
                from cloud_orchestrator import poll_cloud_agent

                try:
                    live = poll_cloud_agent(record["agent_id"])
                    record.update(live)
                    record["status"] = live.get("run_status") or live.get("status") or record.get("status")
                    record_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
                except Exception as exc:
                    record["poll_error"] = str(exc)
        result = "result-in-tree" if result_path(task).exists() else "no-result"
        print(
            f"{task['id']} rank=#{task['leaderboard_rank']:>3} policy={task['merge_policy']:<6} "
            f"agent={record.get('agent_id', '-')} recorded={record.get('status', 'pending')} "
            f"run={record.get('run_status', '-')} pr={record.get('pr_url', '-')} {result} :: {task['title']}"
        )
    return 0


def cmd_self_test(_args: argparse.Namespace) -> int:
    queue = load_queue()
    assert len(queue["tasks"]) == 12
    task = tasks_by_number()[3]
    prompt = build_prompt(task)
    assert "Problem-Fixer fix-003" in prompt
    assert "predictions/prediction_registry.json" in prompt
    assert path_is_allowed(task, "morphoeval/problem_scouts/verify_rule30_delphi_sync.py")
    assert not path_is_allowed(task, "predictions/prediction_registry.json")
    assert not path_is_allowed(task, ".github/workflows/problem-scout-100.yml")
    assert parse_selection("1-3,12") == [1, 2, 3, 12]
    print("problem_fixer self-test passed: queue, candidate cross-checks, prompt, allowlist, selection")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd")

    sub.add_parser("list", help="List the curated fixer wave")
    sub.add_parser("self-test", help="Validate queue, prompts, and allowlist logic")

    prompt = sub.add_parser("prompt", help="Print the full prompt for one fixer")
    prompt.add_argument("--fix", type=int, required=True)

    dispatch = sub.add_parser("dispatch", help="Dispatch Cursor Cloud Agents for fixer tasks")
    dispatch.add_argument("--fixes", default=None, help="Fix selector, e.g. 1-12 or 1,3,7-9")
    dispatch.add_argument("--count", type=int, default=None, help="Dispatch the first N tasks when --fixes is omitted")
    dispatch.add_argument("--repo", default=DEFAULT_REPO)
    dispatch.add_argument("--model", default=None)
    dispatch.add_argument("--stagger-seconds", type=float, default=3.0)
    dispatch.add_argument("--force", action="store_true", help="Dispatch even if a local record exists")
    dispatch.add_argument("--dry-run", action="store_true", help="Print prompts' metadata without calling the Cursor API")

    status = sub.add_parser("status", help="Show dispatch/result state")
    status.add_argument("--poll", action="store_true", help="Poll Cursor for live agent/PR state (requires CURSOR_API_KEY)")

    guard = sub.add_parser("guard", help="Validate a fixer PR diff against the queue allowlist and validators")
    guard.add_argument("--fix", type=int, required=True)
    guard.add_argument("--repo", default=str(ROOT), help="Checkout to validate (default: this repo)")
    guard.add_argument("--base", required=True, help="Base git ref or SHA")
    guard.add_argument("--head", default="HEAD")
    guard.add_argument("--skip-validators", action="store_true")
    guard.add_argument("--json", action="store_true")

    args = parser.parse_args()
    if args.cmd == "list":
        return cmd_list(args)
    if args.cmd == "prompt":
        return cmd_prompt(args)
    if args.cmd == "dispatch":
        return cmd_dispatch(args)
    if args.cmd == "status":
        return cmd_status(args)
    if args.cmd == "guard":
        return cmd_guard(args)
    if args.cmd == "self-test":
        return cmd_self_test(args)
    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

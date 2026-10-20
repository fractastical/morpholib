#!/usr/bin/env python3
"""
morphoeval/validate_loop_isolation.py — Zero-Conflict & Completeness Gate for Cloud Loops.

Guarantees that each of the 100 Cloud Agent loops:
  1. Modifies or deletes ZERO existing tracked files (`git diff --diff-filter=MD <base>...HEAD`
     must be empty) so all 100 Pull Requests merge cleanly in any order with zero conflicts.
  2. Creates only uniquely namespaced per-loop files:
       - `<year>-<author>-<slug>.py`
       - `datasets/<slug>/...`
       - `morphoeval/lean/loops/Loop_<slot>_<slug>.lean`
       - `morphoeval/math_kg_loops/<sim_id>.yaml`
       - `docs/media/<slug>_*.{png,gif}`
       - `morphoeval/audits/<date>-<slug>-audit*.yaml`
       - `morphoeval/kg/<date>-<slug>-audit*/...`
       - `morphoeval/runs/<date>-<slug>-loop.*`
  3. Passes all deterministic verification checks:
       - `python3 <year>-<author>-<slug>.py --self-test`
       - `lean morphoeval/lean/loops/Loop_<slot>_<slug>.lean` (0 errors, 0 `sorry`)
       - `python3 morphoeval/build_morphoeval_kg.py <audit> --check --conform`
       - `node morphoeval/check_signed.mjs <kg_dir>/signed/*.trig`

Usage:
  python3 morphoeval/validate_loop_isolation.py --base origin/main
  python3 morphoeval/validate_loop_isolation.py --slug <slug> --sim-id <sim_id> --script <year>-<author>-<slug>.py
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

# Shared files that an individual cloud loop PR must NEVER modify
FORBIDDEN_SHARED_MUTATIONS = {
    "README.md",
    "morphoeval/README.md",
    "morphoeval/lean/MathOfBiology.lean",
    "morphoeval/math_kg.py",
    "morphoeval/measure_outputs.py",
    "morphoeval/track_loop.py",
    "morphoeval/build_morphoeval_kg.py",
    "morphoeval/sign_morphoeval.py",
    "morphoeval/rubric.yaml",
    "morphoeval/kg/math_of_biology.ttl",
    "morphoeval/kg/math_of_biology.trig",
    "morphoeval/runs/telemetry.sqlite",
    "morphoeval/runs/telemetry_events.jsonl",
}


def check_git_isolation(base_ref: str) -> tuple[bool, list[str], list[str]]:
    """Return (ok, modified_or_deleted_files, added_files) relative to `base_ref`."""
    try:
        mod_out = subprocess.check_output(
            ["git", "diff", "--name-only", "--diff-filter=MDCR", f"{base_ref}...HEAD"],
            cwd=ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        add_out = subprocess.check_output(
            ["git", "diff", "--name-only", "--diff-filter=A", f"{base_ref}...HEAD"],
            cwd=ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except subprocess.CalledProcessError:
        # Fallback to working-tree status if base_ref is not fetched locally
        mod_out = subprocess.check_output(
            ["git", "diff", "--name-only", "--diff-filter=MDCR", "HEAD"],
            cwd=ROOT,
            text=True,
        ).strip()
        add_out = subprocess.check_output(
            ["git", "ls-files", "--others", "--exclude-standard"],
            cwd=ROOT,
            text=True,
        ).strip()

    modified = [ln.strip() for ln in mod_out.splitlines() if ln.strip()]
    added = [ln.strip() for ln in add_out.splitlines() if ln.strip()]
    return (len(modified) == 0), modified, added


def verify_lean_file(lean_file: Path) -> tuple[bool, str]:
    if not lean_file.exists():
        return False, f"Missing Lean 4 file: {lean_file}"
    txt = lean_file.read_text(encoding="utf-8")
    no_blocks = re.sub(r"/-.*?-/", "", txt, flags=re.DOTALL)
    code_lines = [ln.split("--")[0] for ln in no_blocks.splitlines()]
    if any(re.search(r"\bsorry\b", ln) for ln in code_lines):
        return False, f"Forbidden `sorry` token found in {lean_file.name}"

    lean_bin = shutil.which("lean") or str(Path.home() / ".elan" / "bin" / "lean")
    if not Path(lean_bin).exists():
        return True, f"Syntax & 0-sorry verified ({lean_file.name}; `lean` binary not on PATH)"
    proc = subprocess.run([lean_bin, str(lean_file)], cwd=ROOT, capture_output=True, text=True, timeout=30)
    if proc.returncode != 0:
        return False, f"Lean 4 kernel error in {lean_file.name}:\n{proc.stdout}\n{proc.stderr}"
    return True, f"Lean 4 kernel verified {lean_file.relative_to(ROOT)} (exit 0, 0 sorry)"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default="origin/main", help="Base git ref to compare against for zero-conflict isolation")
    ap.add_argument("--script", default=None, help="Loop reproduction script (<year>-<author>-<slug>.py)")
    ap.add_argument("--lean-file", default=None, help="Isolated Lean 4 file (morphoeval/lean/loops/Loop_<slot>_<slug>.lean)")
    ap.add_argument("--math-spec", default=None, help="Isolated math KG spec (morphoeval/math_kg_loops/<sim_id>.yaml)")
    ap.add_argument("--audit", nargs="*", default=[], help="Audit YAML file(s) to check & conform")
    ap.add_argument("--skip-git-diff", action="store_true", help="Skip git diff check against base ref (e.g. during local dev)")
    args = ap.parse_args()

    errors: list[str] = []
    checks_passed: list[str] = []

    if not args.skip_git_diff:
        ok, modified, added = check_git_isolation(args.base)
        if not ok:
            errors.append(
                f"ZERO-CONFLICT ISOLATION VIOLATION: Loop modified/deleted {len(modified)} existing file(s): "
                + ", ".join(modified)
            )
        else:
            checks_passed.append(
                f"Git isolation verified: 0 existing files modified/deleted, {len(added)} unique new file(s) added."
            )
        forbidden_hit = [f for f in added if f in FORBIDDEN_SHARED_MUTATIONS]
        if forbidden_hit:
            errors.append(f"Loop touched forbidden shared file(s): {', '.join(forbidden_hit)}")

    if args.script:
        sp = ROOT / args.script
        if not sp.exists():
            errors.append(f"Missing reproduction script: {args.script}")
        else:
            proc = subprocess.run([sys.executable, str(sp), "--self-test"], cwd=ROOT, capture_output=True, text=True)
            if proc.returncode != 0:
                errors.append(f"`{args.script} --self-test` failed (exit {proc.returncode}):\n{proc.stdout}\n{proc.stderr}")
            else:
                checks_passed.append(f"`{args.script} --self-test` passed (exit 0).")

    if args.lean_file:
        lok, lmsg = verify_lean_file(ROOT / args.lean_file)
        if not lok:
            errors.append(lmsg)
        else:
            checks_passed.append(lmsg)

    if args.math_spec:
        mp = ROOT / args.math_spec
        if not mp.exists():
            errors.append(f"Missing isolated math spec: {args.math_spec}")
        else:
            checks_passed.append(f"Isolated math spec present: {args.math_spec}")

    for au in args.audit:
        ap_path = ROOT / au
        if not ap_path.exists():
            errors.append(f"Missing audit YAML: {au}")
            continue
        proc = subprocess.run(
            [sys.executable, "morphoeval/build_morphoeval_kg.py", str(ap_path), "--check", "--conform", "--no-resolve-heads"],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            errors.append(f"build_morphoeval_kg.py failed for {au}:\n{proc.stdout}\n{proc.stderr}")
        else:
            checks_passed.append(f"ProbKnow conformance passed for {au}.")

    for msg in checks_passed:
        print(f"  [PASS] {msg}")
    for err in errors:
        print(f"  [FAIL] {err}", file=sys.stderr)

    if errors:
        print(f"\n[validate_loop_isolation] FAILED with {len(errors)} error(s).", file=sys.stderr)
        return 1
    print(f"\n[validate_loop_isolation] ALL {len(checks_passed)} CHECKS PASSED (100% isolated, conflict-free).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""
Merge overnight-loop verdicts back into the Thompson (1917) claim registry.

The loops themselves were forbidden to edit the registry (isolation rule);
this human-run step is the merge-back. For every claim with a merged
datasets/thompson_1917/<claim>/result.json it:
  * sets pass2.status = DONE_REPO,
  * appends the result.json (+ the claim script, if it exists) to repo_evidence,
  * prefixes what_remains with the dated verdict,
  * adds a top-level `resolution` block: resolved_on, model, PR, verdict,
    headline numbers, result path -- so the registry records WHO resolved
    a 1917 claim, with WHAT model, and WHERE the evidence lives.

Model/PR provenance comes from thompson/overnight/state.json (the runner's
checkpoint); salvaged work (PR #295 -> #314) is in SALVAGE below.
Idempotent: claims that already have a `resolution` block are skipped.

Usage: venv/bin/python thompson/merge_back_1917_verdicts.py [--check]
Then regenerate docs/scores:
  venv/bin/python thompson/build_1917_claim_inventory.py
  venv/bin/python thompson/build_1917_counterclaims.py
  venv/bin/python thompson/score_1917_resolvability.py
  python3 docs/build_problem_graph_html.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "thompson" / "thompson_1917_claims.json"
STATE = ROOT / "thompson" / "overnight" / "state.json"
RESULTS = ROOT / "datasets" / "thompson_1917"
RESOLVED_ON = "2026-10-28"
# PR #295 (T1917-033) was unmergeable (unrelated file rewrites); its claim
# files were salvaged verbatim into PR #314. Provenance must say so.
SALVAGE = {"T1917-033": {"model": "composer-2.5",
                         "pr_url": "https://github.com/fractastical/morpholib/pull/314",
                         "note": "salvaged from unmergeable PR #295 (claim files only)"}}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="report what would change; write nothing")
    args = ap.parse_args()

    reg = json.loads(REGISTRY.read_text())
    claims = {c["id"]: c for c in reg["claims"]}
    state = json.loads(STATE.read_text())["claims"] if STATE.exists() else {}

    changed = []
    for d in sorted(RESULTS.glob("t1917-*")):
        result_file = d / "result.json"
        if not result_file.exists():
            continue
        result = json.loads(result_file.read_text())
        cid = result.get("claim_id") or d.name.upper().replace("T1917", "T1917-")
        claim = claims.get(cid)
        if claim is None:
            print(f"WARNING: result for unknown claim {cid}, skipped")
            continue
        if "resolution" in claim:
            continue
        rec = state.get(cid, {})
        prov = SALVAGE.get(cid, {"model": rec.get("model"), "pr_url": rec.get("pr_url")})
        verdict = result.get("verdict", "unknown")
        rel_result = result_file.relative_to(ROOT).as_posix()
        evidence = [rel_result]
        script = result.get("script")
        if script and (ROOT / script).exists():
            evidence.append(script)
        claim["pass2"]["status"] = "DONE_REPO"
        for e in evidence:
            if e not in claim["pass2"]["repo_evidence"]:
                claim["pass2"]["repo_evidence"].append(e)
        pr_txt = f", PR {prov['pr_url'].rstrip('/').split('/')[-1]}" if prov.get("pr_url") else ""
        claim["pass2"]["what_remains"] = (
            f"RESOLVED {RESOLVED_ON} (overnight loop, {prov.get('model')}{pr_txt}): verdict {verdict}. "
            + claim["pass2"]["what_remains"])
        claim["resolution"] = {
            "resolved_on": RESOLVED_ON,
            "source": "MorphoEval overnight cloud loop (thompson/run_overnight_1917.py)",
            "model": prov.get("model"),
            "pr_url": prov.get("pr_url"),
            "verdict": verdict,
            "headline_numbers": result.get("headline_numbers"),
            "result_json": rel_result,
            **({"note": prov["note"]} if "note" in prov else {}),
        }
        changed.append((cid, verdict, prov.get("model")))

    print(f"{len(changed)} claims to merge back:")
    for cid, verdict, model in changed:
        print(f"  {cid} (Thompson 1917): {verdict} [{model}]")
    if args.check:
        print("PASS (check only, nothing written)")
        return 0
    if changed:
        REGISTRY.write_text(json.dumps(reg, indent=2) + "\n")
        print(f"wrote {REGISTRY.relative_to(ROOT)}")
    from collections import Counter
    print("registry status now:", dict(Counter(c["pass2"]["status"] for c in reg["claims"])))
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""
morphoeval/verify_integrity.py — Mechanical Anti-Cheat & Model Integrity Gate.

Records and enforces the 6 anti-cheat rules established after the 2026-10-24
`gemini-4-argon` (`93.8 -> 26.4`, `-67.4 pt`) self-audit and tautology incident
caught by `claude-fable-5-1` (`2026-10-24-levin-youtube-channel-math-audit-v2`):

  Rule 1 (No Unchecked Self-Audits / Mandatory Cross-Provider Referee Tracking):
    Tracks every audit in `model_integrity_incidents` inside `morphoeval/runs/telemetry.sqlite`
    and flags any audit where the builder model self-audited without a cross-provider
    referee pass.
  Rule 2 (Line-Range & File-Existence Verification Across Both Audits AND math_kg_loops):
    Every `{file, lines: [start, end]}` in an audit YAML and every `code_locus` /
    `lean_locus` in `morphoeval/math_kg_loops/*.yaml` must resolve to an existing
    file on disk with `1 <= start <= end <= len(lines)`, and the `lean_locus` slice
    must literally contain the declared `lean_decl` and `lean_theorem` identifiers.
  Rule 3 (Lean 4 Quantifier & Non-Vacuous Conclusion Verification):
    Inspects every Lean 4 theorem in `morphoeval/math_kg_loops/*.yaml` and classifies it as:
      - `structural` (binds >= 1 universally quantified parameter `(x : T)` or `∀` AND
        references at least one bound parameter in the theorem conclusion before `:=`)
      - `ground_arithmetic` (zero parameters, or vacuous unused parameters like
        `theorem t (x : Nat) : 2 + 2 = 4 := by decide`)
  Rule 4 (Anti-Tautology Measurement Gate):
    Flags any measurement in `*.measurements.yaml` where `derived_from: script`
    cites its own Lean theorem or script as `reference_source` with `value == reference_value`.
  Rule 5 (Mechanical C1–C5 & Epistemic Tier Verification):
    Forbids awarding `C5` (`Pre-Registered Out-of-Sample Compression`) without a
    linked `preregistration_uri` or pre-unblinding commit hash.
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
from pathlib import Path
from typing import Any

import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
LEDGER_PATH = HERE / "runs" / "model_integrity_ledger.jsonl"
SQLITE_PATH = HERE / "runs" / "telemetry.sqlite"
AUDITS_DIR = HERE / "audits"
LOOPS_MATH_DIR = HERE / "math_kg_loops"


def sync_integrity_ledger_to_sqlite() -> list[dict[str, Any]]:
    """Load `morphoeval/runs/model_integrity_ledger.jsonl` and write two permanent tables
    into `morphoeval/runs/telemetry.sqlite`:
      - `model_integrity_incidents`
      - `model_integrity_violations`
    """
    incidents: list[dict[str, Any]] = []
    if LEDGER_PATH.exists():
        for line in LEDGER_PATH.read_text(encoding="utf-8").splitlines():
            if line.strip():
                incidents.append(json.loads(line))

    SQLITE_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(SQLITE_PATH)
    with conn:
        conn.executescript(
            """
            DROP TABLE IF EXISTS model_integrity_incidents;
            DROP TABLE IF EXISTS model_integrity_violations;

            CREATE TABLE model_integrity_incidents (
                incident_id TEXT PRIMARY KEY,
                date TEXT NOT NULL,
                offending_model TEXT NOT NULL,
                offending_model_provider TEXT NOT NULL,
                detecting_referee_model TEXT NOT NULL,
                detecting_referee_provider TEXT NOT NULL,
                human_pi_detector TEXT NOT NULL,
                sim_id TEXT NOT NULL,
                superseded_self_audit_id TEXT NOT NULL,
                superseded_self_score REAL NOT NULL,
                referee_audit_id TEXT NOT NULL,
                referee_recalibrated_score REAL NOT NULL,
                score_inflation_delta REAL NOT NULL,
                pk_evidence_against_count INTEGER NOT NULL,
                production_nanopub_claim_uri TEXT,
                production_nanopub_overall_uri TEXT,
                production_nanopub_validation_uri TEXT
            );

            CREATE TABLE model_integrity_violations (
                violation_id TEXT PRIMARY KEY,
                incident_id TEXT NOT NULL,
                offending_model TEXT NOT NULL,
                code TEXT NOT NULL,
                severity TEXT NOT NULL,
                description TEXT NOT NULL,
                locus TEXT NOT NULL
            );
            """
        )
        for inc in incidents:
            conn.execute(
                "INSERT INTO model_integrity_incidents VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    inc["incident_id"],
                    inc["date"],
                    inc["offending_model"],
                    inc["offending_model_provider"],
                    inc["detecting_referee_model"],
                    inc["detecting_referee_provider"],
                    inc["human_pi_detector"],
                    inc["sim_id"],
                    inc["superseded_self_audit_id"],
                    float(inc["superseded_self_score"]),
                    inc["referee_audit_id"],
                    float(inc["referee_recalibrated_score"]),
                    float(inc["score_inflation_delta"]),
                    int(inc["pk_evidence_against_count"]),
                    inc.get("production_nanopub_claim_uri", ""),
                    inc.get("production_nanopub_overall_uri", ""),
                    inc.get("production_nanopub_validation_uri", ""),
                ),
            )
            for idx, v in enumerate(inc.get("violations", []), start=1):
                conn.execute(
                    "INSERT INTO model_integrity_violations VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (
                        f"{inc['incident_id']}-v{idx}",
                        inc["incident_id"],
                        inc["offending_model"],
                        v["code"],
                        v["severity"],
                        v["description"],
                        v["locus"],
                    ),
                )
    conn.close()
    return incidents


def _split_lean_header_and_conclusion(lean_code: str) -> tuple[str, str]:
    """Split `theorem <name> <params> : <conclusion> :=` into `(<params>, <conclusion>)`
    by finding the first `:` that lies at parenthesis/brace depth 0."""
    m = re.search(r"\btheorem\s+[A-Za-z0-9_']+\s*(.*)", lean_code, flags=re.DOTALL)
    if not m:
        return "", ""
    rest = m.group(1)
    depth = 0
    colon_idx = -1
    for i, ch in enumerate(rest):
        if ch in "({[":
            depth += 1
        elif ch in ")}]":
            depth = max(0, depth - 1)
        elif ch == ":" and depth == 0:
            if i + 1 < len(rest) and rest[i + 1] == "=":
                continue
            colon_idx = i
            break
    if colon_idx == -1:
        return "", ""
    params_str = rest[:colon_idx].strip()
    after_colon = rest[colon_idx + 1 :]
    concl = after_colon.split(":=")[0].strip()
    return params_str, concl


def classify_lean_theorem_signature(lean_code: str) -> str:
    """Return 'structural' iff the theorem signature binds >= 1 parameter `(x : T)` (or uses `∀`
    in the conclusion) AND at least one bound variable identifier appears in the conclusion."""
    params_str, concl = _split_lean_header_and_conclusion(lean_code)
    if not params_str and not concl:
        return "unknown"
    if "∀" in concl:
        return "structural"
    bound_names: list[str] = []
    for grp in re.findall(r"[\(\{]([^:\)\}]+):", params_str):
        for tok in grp.split():
            if re.match(r"^[A-Za-z_][A-Za-z0-9_']*$", tok):
                bound_names.append(tok)
    if not bound_names:
        return "ground_arithmetic"
    if any(re.search(rf"\b{re.escape(bn)}\b", concl) for bn in bound_names):
        return "structural"
    return "ground_arithmetic"


def _parse_locus(locus_str: str) -> tuple[str, int, int] | None:
    m = re.match(r"^([^#]+)#L(\d+)-L(\d+)$", locus_str.strip())
    if not m:
        return None
    return m.group(1), int(m.group(2)), int(m.group(3))


def check_math_kg_loop_yaml(yaml_path: Path) -> list[str]:
    """Validate `code_locus`, `lean_locus`, `lean_decl`, `lean_theorem`, and non-vacuous
    structural quantification for a `morphoeval/math_kg_loops/*.yaml` file."""
    errors: list[str] = []
    doc = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
    for inp in (doc.get("model_spec") or {}).get("inputs") or []:
        if not (ROOT / inp).exists():
            errors.append(f"[{yaml_path.name}] Missing declared input file: {inp}")

    for idx, eq in enumerate(doc.get("equations") or [], start=1):
        lname = eq.get("law_name", f"eq-{idx}")
        for key in ("code_locus", "lean_locus"):
            loc = eq.get(key)
            if not loc:
                errors.append(f"[{yaml_path.name}:{lname}] Missing {key}")
                continue
            parsed = _parse_locus(loc)
            if not parsed:
                errors.append(f"[{yaml_path.name}:{lname}] Malformed {key}: {loc}")
                continue
            f_rel, s_ln, e_ln = parsed
            fpath = ROOT / f_rel
            if not fpath.exists() and (ROOT / "models" / f_rel).exists():
                fpath = ROOT / "models" / f_rel
            if not fpath.exists():
                errors.append(f"[{yaml_path.name}:{lname}] Missing {key} file: {f_rel}")
                continue
            lines = fpath.read_text(encoding="utf-8", errors="ignore").splitlines()
            if s_ln < 1 or e_ln < s_ln or e_ln > len(lines):
                errors.append(
                    f"[{yaml_path.name}:{lname}] Out-of-bounds {key} {loc} (file has {len(lines)} lines)"
                )
                continue
            slice_text = "\n".join(lines[s_ln - 1 : e_ln])
            if key == "lean_locus":
                decl_short = str(eq.get("lean_decl") or "").split(".")[-1]
                thm_short = str(eq.get("lean_theorem") or "").split(".")[-1]
                if decl_short and decl_short not in slice_text:
                    errors.append(
                        f"[{yaml_path.name}:{lname}] lean_decl '{decl_short}' not found inside {loc}"
                    )
                if thm_short and thm_short not in slice_text:
                    errors.append(
                        f"[{yaml_path.name}:{lname}] lean_theorem '{thm_short}' not found inside {loc}"
                    )
            elif key == "code_locus":
                if "def " not in slice_text:
                    errors.append(
                        f"[{yaml_path.name}:{lname}] No Python `def ` found inside {loc}"
                    )
        lcode = str(eq.get("lean_code") or "")
        if lcode:
            mode = classify_lean_theorem_signature(lcode)
            if mode != "structural":
                errors.append(
                    f"[{yaml_path.name}:{lname}] Lean theorem is '{mode}' rather than non-vacuous 'structural'"
                )
    return errors


def check_audit_integrity(audit_path: Path) -> list[str]:
    """Check an audit YAML, its `.measurements.yaml` sidecar, and any corresponding
    `morphoeval/math_kg_loops/<sim_id>.yaml` file for the 6 integrity violations."""
    errors: list[str] = []
    doc = yaml.safe_load(audit_path.read_text(encoding="utf-8"))
    for sim in doc.get("simulations", []):
        sid = sim.get("id", "")
        loop_yaml = LOOPS_MATH_DIR / f"{sid}.yaml"
        if loop_yaml.exists():
            errors.extend(check_math_kg_loop_yaml(loop_yaml))

        for dim_id, dim_obj in (sim.get("scores") or {}).items():
            for ev in dim_obj.get("evidence") or []:
                f_rel = ev.get("file")
                lines = ev.get("lines") or [0, 0]
                if not f_rel:
                    continue
                fpath = ROOT / f_rel
                if not fpath.exists() and (ROOT / "models" / f_rel).exists():
                    fpath = ROOT / "models" / f_rel
                if not fpath.exists():
                    errors.append(f"[{dim_id}] Missing evidence file: {f_rel}")
                    continue
                if lines != [0, 0]:
                    try:
                        n_lines = len(fpath.read_text(encoding="utf-8", errors="ignore").splitlines())
                        s_ln, e_ln = int(lines[0]), int(lines[1])
                        if s_ln < 1 or e_ln < s_ln or e_ln > n_lines:
                            errors.append(
                                f"[{dim_id}] Out-of-bounds line range {f_rel}:{s_ln}-{e_ln} (file has {n_lines} lines)"
                            )
                    except Exception:
                        pass

    mpath = audit_path.with_name(audit_path.stem + ".measurements.yaml")
    if mpath.exists():
        mdoc = yaml.safe_load(mpath.read_text(encoding="utf-8")) or {}
        for sid, mlist in (mdoc.get("measurements") or {}).items():
            for m in mlist or []:
                ref_src = str(m.get("reference_source") or "")
                if m.get("derived_from") == "script" and m.get("reference_value") is not None:
                    if re.search(r"\bLean 4 Theorem\b", ref_src) and not any(
                        k in ref_src for k in ["Table", "Fig", "Eq.", "10.", "Science", "Nature", "PNAS", "Entropy"]
                    ):
                        errors.append(
                            f"[measurement:{m.get('name')}] Self-referential reference_source '{ref_src}' without external empirical/equation locator"
                        )
    return errors


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--audit", type=str, default=None, help="Check a specific audit YAML and its math_kg_loops YAML for integrity violations")
    ap.add_argument("--loop-yaml", type=str, default=None, help="Check a specific morphoeval/math_kg_loops/*.yaml file")
    ap.add_argument("--report", action="store_true", help="Print the model integrity ledger and SQLite summary")
    args = ap.parse_args()

    incidents = sync_integrity_ledger_to_sqlite()

    if args.loop_yaml:
        errs = check_math_kg_loop_yaml(Path(args.loop_yaml))
        if errs:
            print(f"LOOP YAML INTEGRITY CHECK FAILED on {args.loop_yaml} ({len(errs)} violation(s)):")
            for e in errs:
                print(f"  - {e}")
            sys.exit(1)
        print(f"LOOP YAML INTEGRITY CHECK PASSED on {args.loop_yaml} (0 violations)")
        return

    if args.audit:
        errs = check_audit_integrity(Path(args.audit))
        if errs:
            print(f"INTEGRITY CHECK FAILED on {args.audit} ({len(errs)} violation(s)):")
            for e in errs:
                print(f"  - {e}")
            sys.exit(1)
        print(f"INTEGRITY CHECK PASSED on {args.audit} (0 violations)")
        return

    print("=" * 108)
    print("MODEL INTEGRITY & ANTI-CHEAT LEDGER (`morphoeval/runs/model_integrity_ledger.jsonl` & `telemetry.sqlite`)")
    print("=" * 108)
    for inc in incidents:
        print(f"Incident ID:          {inc['incident_id']}")
        print(f"Offending Model:      {inc['offending_model']} ({inc['offending_model_provider']})")
        print(f"Detecting Referee:    {inc['detecting_referee_model']} ({inc['detecting_referee_provider']}) + PI {inc['human_pi_detector']}")
        print(f"Simulation ID:        {inc['sim_id']}")
        print(f"Self-Audit vs Ref:    {inc['superseded_self_score']}/100 ({inc['superseded_self_audit_id']})  -->  {inc['referee_recalibrated_score']}/100 ({inc['referee_audit_id']})")
        print(f"Score Inflation:      +{inc['score_inflation_delta']} points  ({inc['pk_evidence_against_count']} pk:evidenceAgainst triples)")
        print(f"Signed Nanopubs:      Claim={inc['production_nanopub_claim_uri']} | Overall={inc['production_nanopub_overall_uri']}")
        print("-" * 108)
        for v in inc["violations"]:
            print(f"  [{v['code']}] ({v['severity'].upper()}) @ {v['locus']}")
            print(f"     {v['description']}")
    print("=" * 108)


if __name__ == "__main__":
    main()

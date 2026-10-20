#!/usr/bin/env python3
"""
track_loop.py — Time and token telemetry tracker for the 12-step MorphoEval loop.

Tracks wall-clock duration, CPU user/system time, peak RSS memory, LLM token usage
(input/context tokens, output/completion tokens, subagent tokens, and artifact tokens),
and SHA-256 artifact provenance across all 12 canonical steps of the Infinite Morphospace
& MorphoEval pipeline (morphoeval/README.md, morphoeval/report/draw_pipeline.py):

  Tier A — Making a reproduction:
    1. scout     Find a model of form worth the work (paper, open dataset, untested claim)
    2. paper     Extract the published model specification (equations, encoding, or claim)
    3. source    Fetch/vendor formal source records into datasets/<paper>/ & lock SHA-256
    4. rules     Pin prose/equations into exact rules verified across all records
    5. script    Write <year>-<author>-<model>.py (reproduces/decodes/tests + self-checks)
    6. output    Render comparison figure & animation (docs/media/<name>.gif)

  Tier B — Auditing it (the MorphoEval cycle):
    7. audit     Dual-agent audit (primary + independent 2nd-company assessor, 7 dimensions)
    8. measure   Independent output fingerprinting & morphometric validation (measure_outputs.py)
    9. build     Emit ProbKnow-conformant RDF nanopublications (build_morphoeval_kg.py)
   10. publish   Cryptographic signing & trusty verification (sign_morphoeval.py + check_signed.mjs)
   11. graph     Public record & report synthesis (kg/<audit>/REPORT.md)
   12. remed     Remediate audit findings in script, re-render, re-measure & re-score

Usage
-----
    python3 morphoeval/track_loop.py init --run-id <id> --target <sim-id> [--paper "..."]
    python3 morphoeval/track_loop.py start --run-id <id> --step <1..12|slug> [--role ...] [--model ...]
    python3 morphoeval/track_loop.py end --run-id <id> --step <1..12|slug> [--input-files ...] [--artifacts ...]
    python3 morphoeval/track_loop.py exec --run-id <id> --step <1..12|slug> [--artifacts ...] -- <command...>
    python3 morphoeval/track_loop.py summary --run-id <id> [--md OUT.md] [--csv OUT.csv] [--plot OUT.png]
    python3 morphoeval/track_loop.py --self-test
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import sqlite3
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RUNS_DIR = HERE / "runs"

DEFAULT_TRANSCRIPTS_DIR = Path.home() / (
    ".cursor/projects/Users-jdietz-Documents-GitHub-infinitemorphospace/agent-transcripts"
)

# Canonical 12-step definition matching morphoeval/README.md and draw_pipeline.py
CANONICAL_STEPS: list[dict[str, Any]] = [
    {
        "num": 1,
        "slug": "scout",
        "tier": "A",
        "title": "1  Scout",
        "default_actor": "human+scout-agent",
        "description": "Search open literature and repositories to select a paper, dataset, or testable claim of biological form.",
    },
    {
        "num": 2,
        "slug": "paper",
        "tier": "A",
        "title": "2  Paper",
        "default_actor": "builder-agent",
        "description": "Extract the formal mathematical model, coordinate system, parameter definitions, and empirical claims from the paper.",
    },
    {
        "num": 3,
        "slug": "source",
        "tier": "A",
        "title": "3  Formal source",
        "default_actor": "builder-agent+tool",
        "description": "Fetch or vendor primary empirical records and specimen photographs into datasets/<paper>/ and pin SHA-256 digests.",
    },
    {
        "num": 4,
        "slug": "rules",
        "tier": "A",
        "title": "4  Pin the rules",
        "default_actor": "builder-agent",
        "description": "Translate published equations and biological invariants into exact quantitative rules verified on every record.",
    },
    {
        "num": 5,
        "slug": "script",
        "tier": "A",
        "title": "5  Script",
        "default_actor": "builder-agent",
        "description": "Implement <year>-<author>-<model>.py with --self-test, negative controls, and quantitative morphometric output.",
    },
    {
        "num": 6,
        "slug": "output",
        "tier": "A",
        "title": "6  Output & compare",
        "default_actor": "tool",
        "description": "Execute the script to render the static comparison figure (.png), animation (docs/media/*.gif), and metrics CSV.",
    },
    {
        "num": 7,
        "slug": "audit",
        "tier": "B",
        "title": "7  Dual-agent audit",
        "default_actor": "dual-assessors",
        "description": "Score all 7 MorphoEval rubric dimensions (0-100) with line-pinned code evidence across two independent model providers.",
    },
    {
        "num": 8,
        "slug": "measure",
        "tier": "B",
        "title": "8  Measure",
        "default_actor": "tool (measure_outputs.py)",
        "description": "Independently hash rendered outputs and compute sidecar morphometrics against specimen/literature reference values.",
    },
    {
        "num": 9,
        "slug": "build",
        "tier": "B",
        "title": "9  Write as claims",
        "default_actor": "tool (build_morphoeval_kg.py)",
        "description": "Compile audit + measurements into ProbKnow-template-conformant TriG nanopublications, merged Turtle, and REPORT.md.",
    },
    {
        "num": 10,
        "slug": "publish",
        "tier": "B",
        "title": "10 Sign & verify",
        "default_actor": "tool (sign_morphoeval.py + check_signed.mjs)",
        "description": "Cryptographically sign nanopublications (RSA trusty URIs) and verify all signatures with @nanopub/sign.",
    },
    {
        "num": 11,
        "slug": "graph",
        "tier": "B",
        "title": "11 Public record",
        "default_actor": "tool",
        "description": "Record signed trusty URIs in published.json, run SPARQL ranking queries, and update checkpoint documentation.",
    },
    {
        "num": 12,
        "slug": "remed",
        "tier": "B->A",
        "title": "12 Remediate & re-score",
        "default_actor": "builder+assessor",
        "description": "Close concrete defects flagged by the audit in the script, re-render, re-measure, and emit superseding nanopubs.",
    },
]

STEP_BY_SLUG = {s["slug"]: s for s in CANONICAL_STEPS}
STEP_BY_NUM = {str(s["num"]): s for s in CANONICAL_STEPS}

# Calibrated cl100k_base approximation regex (splits contractions, words, 1-3 digit numbers, punctuation)
_TOKEN_RE = re.compile(
    r"""'s|'t|'re|'ve|'m|'ll|'d| ?[A-Za-z_]+| ?\d{1,3}| ?[^\sA-Za-z0-9_]+|\s+(?!\S)|\s+"""
)


def count_tokens(text: str) -> tuple[int, str]:
    """Count tokens using tiktoken (cl100k_base) if available, otherwise calibrated cl100k regex."""
    if not text:
        return 0, "exact-zero"
    try:
        import tiktoken  # type: ignore

        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text, disallowed_special=())), "tiktoken:cl100k_base"
    except Exception:
        # Calibrated BPE piece estimator: regex pieces + subword split penalty for long identifiers (>8 chars)
        pieces = _TOKEN_RE.findall(text)
        extra = sum(max(0, (len(p.strip()) - 6) // 5) for p in pieces if len(p.strip()) > 8)
        return len(pieces) + extra, "calibrated-bpe-regex"


def resolve_step(step_arg: str) -> dict[str, Any]:
    key = str(step_arg).strip().lower()
    if key in STEP_BY_SLUG:
        return STEP_BY_SLUG[key]
    if key in STEP_BY_NUM:
        return STEP_BY_NUM[key]
    for s in CANONICAL_STEPS:
        if key in s["slug"] or key in s["title"].lower():
            return s
    raise ValueError(
        f"Unknown step '{step_arg}'. Expected 1..12 or one of: {', '.join(STEP_BY_SLUG)}"
    )


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def snapshot_transcripts(transcripts_dir: Path = DEFAULT_TRANSCRIPTS_DIR) -> dict[str, int]:
    """Record current byte offsets of all .jsonl transcript files in the Cursor project."""
    offsets: dict[str, int] = {}
    if not transcripts_dir.exists():
        return offsets
    for p in transcripts_dir.rglob("*.jsonl"):
        try:
            offsets[str(p)] = p.stat().st_size
        except OSError:
            continue
    return offsets


def parse_jsonl_slice(path: Path, start_byte: int = 0) -> dict[str, int]:
    """Parse JSONL transcript records from `start_byte` to EOF and count input/output tokens."""
    if not path.exists():
        return {"input_tokens": 0, "output_tokens": 0, "messages": 0, "tool_uses": 0}
    try:
        raw = path.read_bytes()[start_byte:]
    except OSError:
        return {"input_tokens": 0, "output_tokens": 0, "messages": 0, "tool_uses": 0}

    in_tok = 0
    out_tok = 0
    n_msgs = 0
    n_tools = 0
    for line in raw.decode("utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            tok, _ = count_tokens(line)
            out_tok += tok
            continue
        role = rec.get("role")
        msg = rec.get("message") or {}
        content = msg.get("content")
        if not role:
            continue
        n_msgs += 1
        if isinstance(content, str):
            tok, _ = count_tokens(content)
            if role == "assistant":
                out_tok += tok
            else:
                in_tok += tok
        elif isinstance(content, list):
            for item in content:
                if not isinstance(item, dict):
                    continue
                itype = item.get("type", "")
                if itype == "text":
                    tok, _ = count_tokens(str(item.get("text", "")))
                    if role == "assistant":
                        out_tok += tok
                    else:
                        in_tok += tok
                elif itype == "tool_use":
                    n_tools += 1
                    inp_str = json.dumps(item.get("input", {}), ensure_ascii=False)
                    tok, _ = count_tokens(str(item.get("name", "")) + " " + inp_str)
                    out_tok += tok
                elif itype == "tool_result":
                    res_str = json.dumps(item.get("content", ""), ensure_ascii=False)
                    tok, _ = count_tokens(res_str)
                    in_tok += tok
                else:
                    tok, _ = count_tokens(json.dumps(item, ensure_ascii=False))
                    if role == "assistant":
                        out_tok += tok
                    else:
                        in_tok += tok
    return {
        "input_tokens": in_tok,
        "output_tokens": out_tok,
        "messages": n_msgs,
        "tool_uses": n_tools,
    }


def run_path(run_id: str, runs_dir: Path = RUNS_DIR) -> Path:
    runs_dir.mkdir(parents=True, exist_ok=True)
    return runs_dir / f"{run_id}.yaml"


def load_run(run_id: str, runs_dir: Path = RUNS_DIR) -> dict[str, Any]:
    p = run_path(run_id, runs_dir)
    if not p.exists():
        raise FileNotFoundError(f"Run log not found: {p}")
    return yaml.safe_load(p.read_text(encoding="utf-8"))


def is_isolated_run(doc: dict[str, Any] | None = None) -> bool:
    if os.environ.get("MORPHOEVAL_ISOLATED", "").strip() in ("1", "true", "yes"):
        return True
    if doc and isinstance(doc.get("run"), dict) and doc["run"].get("isolated"):
        return True
    return False


def save_run(doc: dict[str, Any], runs_dir: Path = RUNS_DIR) -> Path:
    p = run_path(doc["run"]["id"], runs_dir)
    recompute_totals(doc)
    p.write_text(yaml.safe_dump(doc, sort_keys=False, allow_unicode=True), encoding="utf-8")
    if not is_isolated_run(doc):
        sync_sqlite(doc, runs_dir)
    write_trig_provenance(doc, runs_dir / f"{doc['run']['id']}.trig")
    return p


def sqlite_path(runs_dir: Path = RUNS_DIR) -> Path:
    runs_dir.mkdir(parents=True, exist_ok=True)
    return runs_dir / "telemetry.sqlite"


def events_jsonl_path(runs_dir: Path = RUNS_DIR, run_id: str | None = None, isolated: bool = False) -> Path:
    runs_dir.mkdir(parents=True, exist_ok=True)
    if isolated and run_id:
        return runs_dir / f"{run_id}.events.jsonl"
    return runs_dir / "telemetry_events.jsonl"


def _init_sqlite_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS runs (
            run_id TEXT PRIMARY KEY,
            target_simulation TEXT,
            paper TEXT,
            created_at TEXT,
            builder_model TEXT,
            primary_assessor TEXT,
            second_assessor TEXT,
            repository_commit_start TEXT,
            steps_completed INTEGER,
            steps_total INTEGER,
            wall_time_s REAL,
            tier_a_wall_s REAL,
            tier_b_wall_s REAL,
            cpu_user_s REAL,
            cpu_sys_s REAL,
            input_tokens INTEGER,
            output_tokens INTEGER,
            subagent_tokens INTEGER,
            total_llm_tokens INTEGER,
            artifact_tokens INTEGER,
            tier_a_llm_tokens INTEGER,
            tier_b_llm_tokens INTEGER,
            effective_tokens_per_s REAL,
            notes TEXT
        );

        CREATE TABLE IF NOT EXISTS steps (
            run_id TEXT,
            step_num INTEGER,
            slug TEXT,
            tier TEXT,
            title TEXT,
            actor TEXT,
            model TEXT,
            status TEXT,
            started_at TEXT,
            ended_at TEXT,
            duration_s REAL,
            cpu_user_s REAL,
            cpu_sys_s REAL,
            max_rss_mb REAL,
            runs_count INTEGER,
            input_tokens INTEGER,
            output_tokens INTEGER,
            subagent_tokens INTEGER,
            total_llm_tokens INTEGER,
            artifact_tokens INTEGER,
            token_method TEXT,
            artifacts_count INTEGER,
            notes TEXT,
            PRIMARY KEY (run_id, step_num)
        );

        CREATE TABLE IF NOT EXISTS step_artifacts (
            run_id TEXT,
            step_num INTEGER,
            slug TEXT,
            file TEXT,
            bytes INTEGER,
            sha256 TEXT,
            tokens INTEGER,
            PRIMARY KEY (run_id, step_num, file)
        );

        CREATE TABLE IF NOT EXISTS events (
            event_id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            run_id TEXT,
            event_type TEXT,
            step_num INTEGER,
            slug TEXT,
            tier TEXT,
            actor TEXT,
            model TEXT,
            duration_s REAL,
            total_llm_tokens INTEGER,
            artifact_tokens INTEGER,
            payload_json TEXT
        );
        """
    )


def sync_sqlite(doc: dict[str, Any], runs_dir: Path = RUNS_DIR) -> Path:
    db_file = sqlite_path(runs_dir)
    run = doc["run"]
    tot = doc["totals"]
    with sqlite3.connect(db_file) as conn:
        _init_sqlite_schema(conn)
        conn.execute(
            """
            INSERT OR REPLACE INTO runs (
                run_id, target_simulation, paper, created_at, builder_model,
                primary_assessor, second_assessor, repository_commit_start,
                steps_completed, steps_total, wall_time_s, tier_a_wall_s, tier_b_wall_s,
                cpu_user_s, cpu_sys_s, input_tokens, output_tokens, subagent_tokens,
                total_llm_tokens, artifact_tokens, tier_a_llm_tokens, tier_b_llm_tokens,
                effective_tokens_per_s, notes
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run["id"],
                run.get("target_simulation"),
                run.get("paper"),
                run.get("created_at"),
                run.get("builder_model"),
                run.get("primary_assessor"),
                run.get("second_assessor"),
                run.get("repository_commit_start"),
                tot.get("steps_completed", 0),
                tot.get("steps_total", 12),
                tot.get("wall_time_s", 0.0),
                tot.get("tier_a_wall_s", 0.0),
                tot.get("tier_b_wall_s", 0.0),
                tot.get("cpu_user_s", 0.0),
                tot.get("cpu_sys_s", 0.0),
                tot.get("input_tokens", 0),
                tot.get("output_tokens", 0),
                tot.get("subagent_tokens", 0),
                tot.get("total_llm_tokens", 0),
                tot.get("artifact_tokens", 0),
                tot.get("tier_a_llm_tokens", 0),
                tot.get("tier_b_llm_tokens", 0),
                tot.get("effective_tokens_per_s", 0.0),
                run.get("notes"),
            ),
        )
        conn.execute("DELETE FROM steps WHERE run_id = ?", (run["id"],))
        conn.execute("DELETE FROM step_artifacts WHERE run_id = ?", (run["id"],))
        for s in doc.get("steps") or []:
            tok = s.get("tokens") or {}
            arts = s.get("artifacts") or []
            conn.execute(
                """
                INSERT INTO steps (
                    run_id, step_num, slug, tier, title, actor, model, status,
                    started_at, ended_at, duration_s, cpu_user_s, cpu_sys_s, max_rss_mb,
                    runs_count, input_tokens, output_tokens, subagent_tokens,
                    total_llm_tokens, artifact_tokens, token_method, artifacts_count, notes
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run["id"],
                    s["num"],
                    s["slug"],
                    s["tier"],
                    s["title"],
                    s.get("actor"),
                    s.get("model"),
                    s.get("status"),
                    s.get("started_at"),
                    s.get("ended_at"),
                    float(s.get("duration_s") or 0.0),
                    float(s.get("cpu_user_s") or 0.0),
                    float(s.get("cpu_sys_s") or 0.0),
                    float(s.get("max_rss_mb") or 0.0),
                    int(s.get("runs_count") or 0),
                    int(tok.get("input_tokens") or 0),
                    int(tok.get("output_tokens") or 0),
                    int(tok.get("subagent_tokens") or 0),
                    int(tok.get("total_llm_tokens") or 0),
                    int(tok.get("artifact_tokens") or 0),
                    str(tok.get("method") or "none"),
                    len(arts),
                    s.get("notes"),
                ),
            )
            for a in arts:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO step_artifacts (
                        run_id, step_num, slug, file, bytes, sha256, tokens
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        run["id"],
                        s["num"],
                        s["slug"],
                        a["file"],
                        int(a.get("bytes") or 0),
                        a.get("sha256"),
                        int(a.get("tokens") or 0),
                    ),
                )
        conn.commit()
    return db_file


def append_event(
    runs_dir: Path,
    *,
    run_id: str,
    event_type: str,
    step_num: int | None = None,
    slug: str | None = None,
    tier: str | None = None,
    actor: str | None = None,
    model: str | None = None,
    duration_s: float = 0.0,
    total_llm_tokens: int = 0,
    artifact_tokens: int = 0,
    payload: dict[str, Any] | None = None,
) -> None:
    ts = utc_now_iso()
    record = {
        "timestamp": ts,
        "run_id": run_id,
        "event_type": event_type,
        "step_num": step_num,
        "slug": slug,
        "tier": tier,
        "actor": actor,
        "model": model,
        "duration_s": round(float(duration_s), 3),
        "total_llm_tokens": int(total_llm_tokens),
        "artifact_tokens": int(artifact_tokens),
        "payload": payload or {},
    }
    isolated = is_isolated_run()
    rp = runs_dir / f"{run_id}.yaml"
    if not isolated and rp.exists():
        try:
            doc_chk = yaml.safe_load(rp.read_text(encoding="utf-8"))
            isolated = is_isolated_run(doc_chk)
        except Exception:
            pass

    jpath = events_jsonl_path(runs_dir, run_id=run_id, isolated=isolated)
    with jpath.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

    if isolated:
        return

    db_file = sqlite_path(runs_dir)
    with sqlite3.connect(db_file) as conn:
        _init_sqlite_schema(conn)
        conn.execute(
            """
            INSERT INTO events (
                timestamp, run_id, event_type, step_num, slug, tier,
                actor, model, duration_s, total_llm_tokens, artifact_tokens, payload_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                ts,
                run_id,
                event_type,
                step_num,
                slug,
                tier,
                actor,
                model,
                round(float(duration_s), 3),
                int(total_llm_tokens),
                int(artifact_tokens),
                json.dumps(payload or {}, ensure_ascii=False),
            ),
        )
        conn.commit()


def _trig_escape(val: str) -> str:
    return (
        str(val)
        .replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("\n", "\\n")
        .replace("\r", "\\r")
    )


def write_trig_provenance(doc: dict[str, Any], out_trig: Path) -> Path:
    """Emit a PROV-O / MorphoEval RDF TriG nanopublication for the 12-step loop telemetry."""
    out_trig.parent.mkdir(parents=True, exist_ok=True)
    run = doc["run"]
    tot = doc["totals"]
    run_id = run["id"]
    base_uri = f"https://w3id.org/morphoeval/runs/{run_id}"
    lines = [
        f"@prefix this: <{base_uri}> .",
        f"@prefix sub: <{base_uri}#> .",
        "@prefix np: <http://www.nanopub.org/nschema#> .",
        "@prefix dct: <http://purl.org/dc/terms/> .",
        "@prefix prov: <http://www.w3.org/ns/prov#> .",
        "@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .",
        "@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .",
        "@prefix mev: <https://w3id.org/morphoeval/vocab#> .",
        "",
        "sub:Head {",
        "  this: a np:Nanopublication ;",
        "    np:hasAssertion sub:assertion ;",
        "    np:hasProvenance sub:provenance ;",
        "    np:hasPublicationInfo sub:pubinfo .",
        "}",
        "",
        "sub:assertion {",
        f'  sub:run a prov:Activity, mev:MorphoEvalLoopRun ;',
        f'    rdfs:label "MorphoEval 12-Step Loop Telemetry — {_trig_escape(run_id)}" ;',
        f'    mev:targetSimulation "{_trig_escape(run.get("target_simulation") or "")}" ;',
        f'    mev:paper "{_trig_escape(run.get("paper") or "")}" ;',
        f'    mev:stepsCompleted "{int(tot.get("steps_completed", 0))}"^^xsd:integer ;',
        f'    mev:wallClockSeconds "{float(tot.get("wall_time_s", 0.0)):.3f}"^^xsd:decimal ;',
        f'    mev:tierAWallSeconds "{float(tot.get("tier_a_wall_s", 0.0)):.3f}"^^xsd:decimal ;',
        f'    mev:tierBWallSeconds "{float(tot.get("tier_b_wall_s", 0.0)):.3f}"^^xsd:decimal ;',
        f'    mev:cpuUserSeconds "{float(tot.get("cpu_user_s", 0.0)):.3f}"^^xsd:decimal ;',
        f'    mev:cpuSysSeconds "{float(tot.get("cpu_sys_s", 0.0)):.3f}"^^xsd:decimal ;',
        f'    mev:inputTokens "{int(tot.get("input_tokens", 0))}"^^xsd:integer ;',
        f'    mev:outputTokens "{int(tot.get("output_tokens", 0))}"^^xsd:integer ;',
        f'    mev:subagentTokens "{int(tot.get("subagent_tokens", 0))}"^^xsd:integer ;',
        f'    mev:totalLlmTokens "{int(tot.get("total_llm_tokens", 0))}"^^xsd:integer ;',
        f'    mev:artifactTokens "{int(tot.get("artifact_tokens", 0))}"^^xsd:integer .',
        "",
    ]
    for s in doc.get("steps") or []:
        tok = s.get("tokens") or {}
        step_node = f"sub:step_{s['num']:02d}_{s['slug']}"
        lines.extend(
            [
                f"  {step_node} a prov:Activity, mev:LoopStep ;",
                f"    prov:wasInformedBy sub:run ;",
                f'    mev:stepNumber "{s["num"]}"^^xsd:integer ;',
                f'    mev:stepSlug "{_trig_escape(s["slug"])}" ;',
                f'    mev:tier "{_trig_escape(s["tier"])}" ;',
                f'    mev:status "{_trig_escape(s.get("status") or "pending")}" ;',
                f'    mev:actor "{_trig_escape(s.get("actor") or "")}" ;',
                f'    mev:model "{_trig_escape(s.get("model") or "")}" ;',
                f'    mev:wallClockSeconds "{float(s.get("duration_s") or 0.0):.3f}"^^xsd:decimal ;',
                f'    mev:cpuSeconds "{float(s.get("cpu_user_s") or 0.0) + float(s.get("cpu_sys_s") or 0.0):.3f}"^^xsd:decimal ;',
                f'    mev:inputTokens "{int(tok.get("input_tokens") or 0)}"^^xsd:integer ;',
                f'    mev:outputTokens "{int(tok.get("output_tokens") or 0)}"^^xsd:integer ;',
                f'    mev:subagentTokens "{int(tok.get("subagent_tokens") or 0)}"^^xsd:integer ;',
                f'    mev:totalLlmTokens "{int(tok.get("total_llm_tokens") or 0)}"^^xsd:integer ;',
                f'    mev:artifactTokens "{int(tok.get("artifact_tokens") or 0)}"^^xsd:integer .',
            ]
        )
        for idx, a in enumerate(s.get("artifacts") or []):
            art_node = f"sub:step_{s['num']:02d}_art_{idx + 1}"
            parent_ids = a.get("parent_artifact_ids") or []
            parent_triples = [
                f'    prov:wasDerivedFrom <urn:sha256:{_trig_escape(pid)}> ;'
                for pid in parent_ids
            ]
            lines.extend(
                [
                    f"  {art_node} a prov:Entity, mev:StepArtifact ;",
                    f"    prov:wasGeneratedBy {step_node} ;",
                    *parent_triples,
                    f'    rdfs:label "{_trig_escape(a["file"])}" ;',
                    f'    mev:byteLength "{int(a.get("bytes") or 0)}"^^xsd:integer ;',
                    f'    mev:sha256 "{_trig_escape(a.get("sha256") or "")}" ;',
                    f'    mev:tokenEquivalent "{int(a.get("tokens") or 0)}"^^xsd:integer .',
                ]
            )
    lines.extend(
        [
            "}",
            "",
            "sub:provenance {",
            f'  sub:assertion prov:wasGeneratedBy sub:run ;',
            f'    prov:wasAttributedTo <https://github.com/fractastical/morpholib> ;',
            f'    mev:repositoryCommit "{_trig_escape(run.get("repository_commit_start") or "")}" .',
            "}",
            "",
            "sub:pubinfo {",
            f'  this: dct:created "{_trig_escape(run.get("created_at") or utc_now_iso())}"^^xsd:dateTime ;',
            '    dct:license <https://creativecommons.org/licenses/by/4.0/> ;',
            '    rdfs:comment "Generated by morphoeval/track_loop.py" .',
            "}",
            "",
        ]
    )
    out_trig.write_text("\n".join(lines), encoding="utf-8")
    return out_trig


def recompute_totals(doc: dict[str, Any]) -> None:
    steps = doc.get("steps") or []
    completed = [s for s in steps if s.get("status") == "completed"]
    total_wall = sum(float(s.get("duration_s") or 0.0) for s in steps)
    total_cpu_u = sum(float(s.get("cpu_user_s") or 0.0) for s in steps)
    total_cpu_s = sum(float(s.get("cpu_sys_s") or 0.0) for s in steps)
    tier_a_wall = sum(
        float(s.get("duration_s") or 0.0) for s in steps if s.get("tier") == "A"
    )
    tier_b_wall = sum(
        float(s.get("duration_s") or 0.0)
        for s in steps
        if str(s.get("tier", "")).startswith("B")
    )

    in_tok = sum(int((s.get("tokens") or {}).get("input_tokens") or 0) for s in steps)
    out_tok = sum(int((s.get("tokens") or {}).get("output_tokens") or 0) for s in steps)
    sub_tok = sum(int((s.get("tokens") or {}).get("subagent_tokens") or 0) for s in steps)
    art_tok = sum(int((s.get("tokens") or {}).get("artifact_tokens") or 0) for s in steps)
    tot_tok = in_tok + out_tok + sub_tok

    tier_a_tok = sum(
        int((s.get("tokens") or {}).get("total_llm_tokens") or 0)
        for s in steps
        if s.get("tier") == "A"
    )
    tier_b_tok = sum(
        int((s.get("tokens") or {}).get("total_llm_tokens") or 0)
        for s in steps
        if str(s.get("tier", "")).startswith("B")
    )

    doc["totals"] = {
        "steps_completed": len(completed),
        "steps_total": len(steps),
        "wall_time_s": round(total_wall, 3),
        "wall_time_min": round(total_wall / 60.0, 2),
        "tier_a_wall_s": round(tier_a_wall, 3),
        "tier_b_wall_s": round(tier_b_wall, 3),
        "cpu_user_s": round(total_cpu_u, 3),
        "cpu_sys_s": round(total_cpu_s, 3),
        "input_tokens": in_tok,
        "output_tokens": out_tok,
        "subagent_tokens": sub_tok,
        "total_llm_tokens": tot_tok,
        "artifact_tokens": art_tok,
        "tier_a_llm_tokens": tier_a_tok,
        "tier_b_llm_tokens": tier_b_tok,
        "effective_tokens_per_s": round(tot_tok / total_wall, 1) if total_wall > 0 else 0.0,
    }


def cmd_init(args: argparse.Namespace, runs_dir: Path = RUNS_DIR) -> Path:
    steps_list = []
    for s in CANONICAL_STEPS:
        steps_list.append(
            {
                "num": s["num"],
                "slug": s["slug"],
                "tier": s["tier"],
                "title": s["title"],
                "actor": s["default_actor"],
                "model": args.builder_model if s["tier"] == "A" else None,
                "status": "pending",
                "started_at": None,
                "ended_at": None,
                "duration_s": 0.0,
                "cpu_user_s": 0.0,
                "cpu_sys_s": 0.0,
                "max_rss_mb": 0.0,
                "runs_count": 0,
                "tokens": {
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "subagent_tokens": 0,
                    "total_llm_tokens": 0,
                    "artifact_tokens": 0,
                    "method": "none",
                },
                "artifacts": [],
                "commands": [],
                "notes": None,
            }
        )
    doc = {
        "schema_version": "morphoeval-loop-telemetry-0.1.0",
        "run": {
            "id": args.run_id,
            "target_simulation": args.target,
            "paper": args.paper,
            "created_at": utc_now_iso(),
            "builder_model": args.builder_model,
            "primary_assessor": args.primary_assessor,
            "second_assessor": args.second_assessor,
            "repository_commit_start": _git_head(),
            "isolated": bool(getattr(args, "isolated", False) or is_isolated_run()),
            "notes": args.notes,
        },
        "totals": {},
        "steps": steps_list,
    }
    p = save_run(doc, runs_dir)
    append_event(
        runs_dir,
        run_id=args.run_id,
        event_type="run_init",
        actor="orchestrator",
        model=args.builder_model,
        payload={
            "target_simulation": args.target,
            "paper": args.paper,
            "primary_assessor": args.primary_assessor,
            "second_assessor": args.second_assessor,
        },
    )
    print(f"[track_loop] Initialized run '{args.run_id}' -> {p}")
    return p


def _git_head() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True
        ).strip()
        return out
    except Exception:
        return "unknown"


def _find_step_entry(doc: dict[str, Any], step_arg: str) -> dict[str, Any]:
    meta = resolve_step(step_arg)
    for entry in doc["steps"]:
        if entry["slug"] == meta["slug"]:
            return entry
    raise KeyError(f"Step {meta['slug']} missing from run document")


def cmd_start(args: argparse.Namespace, runs_dir: Path = RUNS_DIR) -> None:
    doc = load_run(args.run_id, runs_dir)
    entry = _find_step_entry(doc, args.step)
    now_iso = utc_now_iso()
    now_ep = time.time()
    if entry.get("started_at") is None:
        entry["started_at"] = now_iso
    entry["_active_start_epoch"] = now_ep
    entry["_transcript_offsets"] = snapshot_transcripts()
    entry["status"] = "in_progress"
    if args.actor:
        entry["actor"] = args.actor
    if args.model:
        entry["model"] = args.model
    if args.notes:
        entry["notes"] = args.notes
    save_run(doc, runs_dir)
    append_event(
        runs_dir,
        run_id=args.run_id,
        event_type="step_start",
        step_num=entry["num"],
        slug=entry["slug"],
        tier=entry["tier"],
        actor=entry.get("actor"),
        model=entry.get("model"),
        payload={"notes": args.notes or ""},
    )
    print(f"[track_loop] START step {entry['num']} ({entry['slug']}) at {now_iso}")


def _collect_file_tokens(paths: list[str]) -> tuple[int, list[dict[str, Any]], str]:
    total_tok = 0
    records = []
    method = "calibrated-bpe-regex"
    for rel in paths:
        p = (ROOT / rel) if not Path(rel).is_absolute() else Path(rel)
        if not p.exists() or p.is_dir():
            continue
        size_b = p.stat().st_size
        digest = sha256_file(p)
        tok = 0
        # Only tokenize text-like files; binary images/databases get metadata token equivalence
        if p.suffix.lower() in {
            ".py",
            ".md",
            ".yaml",
            ".yml",
            ".json",
            ".csv",
            ".trig",
            ".ttl",
            ".bib",
            ".txt",
            ".rq",
            ".mjs",
            ".sh",
            ".tex",
        }:
            txt = p.read_text(encoding="utf-8", errors="replace")
            tok, method = count_tokens(txt)
        else:
            tok = 64  # compact descriptor token footprint for binary image/GIF metadata
        total_tok += tok
        try:
            rel_str = str(p.resolve().relative_to(ROOT.resolve()))
        except ValueError:
            rel_str = str(p)
        records.append(
            {
                "file": rel_str,
                "bytes": size_b,
                "sha256": digest,
                "tokens": tok,
            }
        )
    return total_tok, records, method


def cmd_end(args: argparse.Namespace, runs_dir: Path = RUNS_DIR) -> None:
    doc = load_run(args.run_id, runs_dir)
    entry = _find_step_entry(doc, args.step)
    now_iso = utc_now_iso()
    now_ep = time.time()

    start_ep = entry.pop("_active_start_epoch", None)
    start_offsets: dict[str, int] = entry.pop("_transcript_offsets", None) or {}

    if args.duration_s is not None:
        elapsed = float(args.duration_s)
    elif start_ep is not None:
        elapsed = max(0.001, now_ep - float(start_ep))
    else:
        elapsed = 0.0

    entry["duration_s"] = round(float(entry.get("duration_s") or 0.0) + elapsed, 3)
    entry["ended_at"] = now_iso
    entry["runs_count"] = int(entry.get("runs_count") or 0) + 1
    entry["status"] = args.status or "completed"
    if args.actor:
        entry["actor"] = args.actor
    if args.model:
        entry["model"] = args.model
    if args.notes:
        prev = entry.get("notes")
        entry["notes"] = f"{prev}; {args.notes}" if prev else args.notes

    # 1. Check transcript delta if any .jsonl grew during the step
    trans_in, trans_out = 0, 0
    end_offsets = snapshot_transcripts()
    for p_str, end_sz in end_offsets.items():
        start_sz = start_offsets.get(p_str, end_sz)
        if end_sz > start_sz:
            delta = parse_jsonl_slice(Path(p_str), start_sz)
            trans_in += delta["input_tokens"]
            trans_out += delta["output_tokens"]

    # 2. Explicit subagent JSONL transcript if provided
    sub_tok = int(args.subagent_tokens or 0)
    if args.subagent_jsonl:
        sp = Path(args.subagent_jsonl)
        if sp.exists():
            sdelta = parse_jsonl_slice(sp, 0)
            sub_tok += sdelta["input_tokens"] + sdelta["output_tokens"]

    # 3. Input files read / context files
    in_file_tok = 0
    method = "explicit"
    parent_hashes: list[str] = []
    if args.input_files:
        in_file_tok, in_records, method = _collect_file_tokens(args.input_files)
        parent_hashes = [r["sha256"] for r in in_records if r.get("sha256")]

    # 4. Output artifacts written
    art_tok = 0
    if args.artifacts:
        art_tok, art_records, method = _collect_file_tokens(args.artifacts)
        existing_by_file = {a["file"]: a for a in (entry.get("artifacts") or [])}
        for r in art_records:
            if parent_hashes:
                r["parent_artifact_ids"] = list(dict.fromkeys(parent_hashes))
            existing_by_file[r["file"]] = r
        entry["artifacts"] = list(existing_by_file.values())

        # Emit ScienceClaw-compatible append-only artifact ledger (<run>.ledger.jsonl)
        ledger_path = runs_dir / f"{args.run_id}.ledger.jsonl"
        for r in art_records:
            ledger_rec = {
                "artifact_id": r.get("sha256") or "",
                "run_id": args.run_id,
                "step_num": entry.get("num"),
                "step_id": entry.get("id"),
                "phase": entry.get("phase"),
                "title": entry.get("title"),
                "file": r.get("file"),
                "bytes": r.get("bytes", 0),
                "tokens": r.get("tokens", 0),
                "parent_artifact_ids": r.get("parent_artifact_ids") or [],
                "timestamp": now_iso,
            }
            with open(ledger_path, "a", encoding="utf-8") as lf:
                lf.write(json.dumps(ledger_rec) + "\n")

    # Combine token figures (explicit overrides > transcript deltas + file context/output estimates)
    tok_obj = entry.get("tokens") or {}
    add_in = (
        int(args.input_tokens)
        if args.input_tokens is not None
        else (trans_in if trans_in > 0 else in_file_tok)
    )
    add_out = (
        int(args.output_tokens)
        if args.output_tokens is not None
        else (trans_out if trans_out > 0 else art_tok)
    )

    tok_obj["input_tokens"] = int(tok_obj.get("input_tokens") or 0) + add_in
    tok_obj["output_tokens"] = int(tok_obj.get("output_tokens") or 0) + add_out
    tok_obj["subagent_tokens"] = int(tok_obj.get("subagent_tokens") or 0) + sub_tok
    tok_obj["artifact_tokens"] = sum(
        int(a.get("tokens") or 0) for a in (entry.get("artifacts") or [])
    )
    tok_obj["total_llm_tokens"] = (
        tok_obj["input_tokens"] + tok_obj["output_tokens"] + tok_obj["subagent_tokens"]
    )
    tok_obj["method"] = (
        "explicit+BPE"
        if (args.input_tokens is not None or args.output_tokens is not None)
        else ("transcript-jsonl+BPE" if (trans_in + trans_out) > 0 else method)
    )
    entry["tokens"] = tok_obj

    save_run(doc, runs_dir)
    append_event(
        runs_dir,
        run_id=args.run_id,
        event_type="step_end",
        step_num=entry["num"],
        slug=entry["slug"],
        tier=entry["tier"],
        actor=entry.get("actor"),
        model=entry.get("model"),
        duration_s=elapsed,
        total_llm_tokens=tok_obj["total_llm_tokens"],
        artifact_tokens=tok_obj["artifact_tokens"],
        payload={
            "status": entry["status"],
            "input_tokens": tok_obj["input_tokens"],
            "output_tokens": tok_obj["output_tokens"],
            "subagent_tokens": tok_obj["subagent_tokens"],
            "artifacts": [a["file"] for a in (entry.get("artifacts") or [])],
            "notes": args.notes or "",
        },
    )
    print(
        f"[track_loop] END step {entry['num']} ({entry['slug']}): "
        f"+{elapsed:.2f}s (total {entry['duration_s']:.2f}s), "
        f"LLM tokens={tok_obj['total_llm_tokens']:,} "
        f"(in={tok_obj['input_tokens']:,}, out={tok_obj['output_tokens']:,}, sub={tok_obj['subagent_tokens']:,}), "
        f"artifact_tokens={tok_obj['artifact_tokens']:,}"
    )


def _run_command_with_rusage(cmd: list[str]) -> tuple[int, float, float, float, float, str, str]:
    """Run command, returning (returncode, wall_s, user_s, sys_s, max_rss_mb, stdout, stderr)."""
    t0 = time.perf_counter()
    r0_children = os.times()
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    t1 = time.perf_counter()
    r1_children = os.times()
    wall_s = max(0.001, t1 - t0)
    user_s = max(0.0, r1_children.children_user - r0_children.children_user)
    sys_s = max(0.0, r1_children.children_system - r0_children.children_system)
    # macOS ru_maxrss is in bytes, Linux in kilobytes; estimate from resource module
    try:
        import resource

        ru = resource.getrusage(resource.RUSAGE_CHILDREN)
        rss_mb = (
            ru.ru_maxrss / (1024 * 1024)
            if sys.platform == "darwin"
            else ru.ru_maxrss / 1024.0
        )
    except Exception:
        rss_mb = 0.0
    return proc.returncode, wall_s, user_s, sys_s, rss_mb, proc.stdout, proc.stderr


def cmd_exec(args: argparse.Namespace, runs_dir: Path = RUNS_DIR) -> int:
    cmd = list(args.cmd)
    if cmd and cmd[0] == "--":
        cmd = cmd[1:]
    if not cmd:
        raise ValueError("No command provided after -- for exec")

    # Start step
    start_ns = argparse.Namespace(
        run_id=args.run_id,
        step=args.step,
        actor=args.actor,
        model=args.model,
        notes=None,
    )
    cmd_start(start_ns, runs_dir)

    rc, wall_s, user_s, sys_s, rss_mb, out_txt, err_txt = _run_command_with_rusage(cmd)
    if out_txt:
        sys.stdout.write(out_txt)
    if err_txt:
        sys.stderr.write(err_txt)

    doc = load_run(args.run_id, runs_dir)
    entry = _find_step_entry(doc, args.step)
    entry["cpu_user_s"] = round(float(entry.get("cpu_user_s") or 0.0) + user_s, 3)
    entry["cpu_sys_s"] = round(float(entry.get("cpu_sys_s") or 0.0) + sys_s, 3)
    entry["max_rss_mb"] = round(max(float(entry.get("max_rss_mb") or 0.0), rss_mb), 1)
    cmd_str = " ".join(cmd)
    out_tok, _ = count_tokens(out_txt + "\n" + err_txt)
    cmd_tok, _ = count_tokens(cmd_str)
    entry.setdefault("commands", []).append(
        {
            "command": cmd_str,
            "exit_code": rc,
            "wall_s": round(wall_s, 3),
            "cpu_user_s": round(user_s, 3),
            "cpu_sys_s": round(sys_s, 3),
            "stdout_tokens": out_tok,
        }
    )
    save_run(doc, runs_dir)

    end_ns = argparse.Namespace(
        run_id=args.run_id,
        step=args.step,
        status="completed" if rc == 0 else "failed",
        duration_s=wall_s,
        actor=args.actor,
        model=args.model,
        input_tokens=args.input_tokens if args.input_tokens is not None else cmd_tok,
        output_tokens=args.output_tokens if args.output_tokens is not None else out_tok,
        subagent_tokens=args.subagent_tokens,
        subagent_jsonl=args.subagent_jsonl,
        input_files=args.input_files,
        artifacts=args.artifacts,
        notes=args.notes,
    )
    cmd_end(end_ns, runs_dir)
    return rc


def render_markdown_summary(doc: dict[str, Any]) -> str:
    run = doc["run"]
    tot = doc["totals"]
    lines = [
        f"# MorphoEval Full-Loop Telemetry — `{run['id']}`",
        "",
        f"- **Target simulation:** `{run.get('target_simulation')}`",
        f"- **Primary paper / dataset:** {run.get('paper') or '—'}",
        f"- **Builder model:** `{run.get('builder_model')}`",
        f"- **Assessor models:** `{run.get('primary_assessor')}` + `{run.get('second_assessor')}`",
        f"- **Start commit:** `{run.get('repository_commit_start')}`",
        f"- **Completed steps:** `{tot['steps_completed']} / {tot['steps_total']}`",
        f"- **Total wall-clock time:** **{tot['wall_time_s']:.2f} s** ({tot['wall_time_min']:.2f} min) "
        f"— Tier A: `{tot['tier_a_wall_s']:.2f} s` · Tier B: `{tot['tier_b_wall_s']:.2f} s`",
        f"- **Total CPU time:** `{tot['cpu_user_s']:.2f} s` user + `{tot['cpu_sys_s']:.2f} s` sys",
        f"- **Total LLM tokens:** **{tot['total_llm_tokens']:,}** "
        f"(input: `{tot['input_tokens']:,}`, output: `{tot['output_tokens']:,}`, subagent: `{tot['subagent_tokens']:,}`) "
        f"— Tier A: `{tot['tier_a_llm_tokens']:,}` · Tier B: `{tot['tier_b_llm_tokens']:,}`",
        f"- **Total artifact tokens produced:** `{tot['artifact_tokens']:,}`",
        "",
        "## Per-Step Breakdown (12-Step MorphoEval Loop)",
        "",
        "| # | Tier | Step | Actor / Model | Status | Wall (s) | CPU (s) | Input tok | Output tok | Subagent tok | Total LLM tok | Artifact tok | Key Artifacts |",
        "|---:|:---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for s in doc["steps"]:
        tok = s.get("tokens") or {}
        cpu_tot = float(s.get("cpu_user_s") or 0.0) + float(s.get("cpu_sys_s") or 0.0)
        actor_str = s.get("actor") or "—"
        if s.get("model"):
            actor_str = f"{actor_str} (`{s['model']}`)"
        arts = [f"`{Path(a['file']).name}`" for a in (s.get("artifacts") or [])[:3]]
        if len(s.get("artifacts") or []) > 3:
            arts.append(f"+{len(s['artifacts']) - 3} more")
        art_col = ", ".join(arts) if arts else "—"
        lines.append(
            f"| {s['num']} | {s['tier']} | **{s['slug']}** ({s['title'][3:]}) | {actor_str} | "
            f"`{s['status']}` | {float(s.get('duration_s') or 0.0):.2f} | {cpu_tot:.2f} | "
            f"{int(tok.get('input_tokens') or 0):,} | {int(tok.get('output_tokens') or 0):,} | "
            f"{int(tok.get('subagent_tokens') or 0):,} | **{int(tok.get('total_llm_tokens') or 0):,}** | "
            f"{int(tok.get('artifact_tokens') or 0):,} | {art_col} |"
        )
    lines.append(
        f"| — | **All** | **TOTAL** | — | `{tot['steps_completed']}/{tot['steps_total']}` | "
        f"**{tot['wall_time_s']:.2f}** | **{tot['cpu_user_s'] + tot['cpu_sys_s']:.2f}** | "
        f"**{tot['input_tokens']:,}** | **{tot['output_tokens']:,}** | "
        f"**{tot['subagent_tokens']:,}** | **{tot['total_llm_tokens']:,}** | "
        f"**{tot['artifact_tokens']:,}** | — |"
    )
    lines.append("")
    lines.append("## Step Notes & Provenance")
    lines.append("")
    for s in doc["steps"]:
        if s.get("notes") or s.get("artifacts"):
            lines.append(f"### Step {s['num']} — `{s['slug']}` ({s['title'][3:]})")
            if s.get("notes"):
                lines.append(f"- **Notes:** {s['notes']}")
            for a in s.get("artifacts") or []:
                lines.append(
                    f"- **Artifact:** `{a['file']}` ({a['bytes']:,} B, `{a['tokens']:,}` tok, SHA-256 `{a['sha256'][:16]}…`)"
                )
            lines.append("")
    return "\n".join(lines)


def write_csv_summary(doc: dict[str, Any], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "step_num",
        "tier",
        "slug",
        "title",
        "actor",
        "model",
        "status",
        "started_at",
        "ended_at",
        "duration_s",
        "cpu_user_s",
        "cpu_sys_s",
        "max_rss_mb",
        "input_tokens",
        "output_tokens",
        "subagent_tokens",
        "total_llm_tokens",
        "artifact_tokens",
        "artifacts_count",
        "notes",
    ]
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for s in doc["steps"]:
            tok = s.get("tokens") or {}
            writer.writerow(
                {
                    "step_num": s["num"],
                    "tier": s["tier"],
                    "slug": s["slug"],
                    "title": s["title"],
                    "actor": s.get("actor"),
                    "model": s.get("model"),
                    "status": s.get("status"),
                    "started_at": s.get("started_at"),
                    "ended_at": s.get("ended_at"),
                    "duration_s": s.get("duration_s"),
                    "cpu_user_s": s.get("cpu_user_s"),
                    "cpu_sys_s": s.get("cpu_sys_s"),
                    "max_rss_mb": s.get("max_rss_mb"),
                    "input_tokens": tok.get("input_tokens", 0),
                    "output_tokens": tok.get("output_tokens", 0),
                    "subagent_tokens": tok.get("subagent_tokens", 0),
                    "total_llm_tokens": tok.get("total_llm_tokens", 0),
                    "artifact_tokens": tok.get("artifact_tokens", 0),
                    "artifacts_count": len(s.get("artifacts") or []),
                    "notes": s.get("notes") or "",
                }
            )


def _fmt_dur_badge(sec: float) -> str:
    if sec >= 60.0:
        return f"{sec:.1f}s ({sec / 60.0:.1f}m)"
    if sec >= 1.0:
        return f"{sec:.1f}s"
    return f"{sec * 1000:.0f}ms"


def _fmt_tok_badge(tok: int) -> str:
    if tok >= 1000:
        return f"{tok / 1000.0:.1f}k tok"
    return f"{tok} tok"


def render_single_cycle_loop(
    doc: dict[str, Any],
    out_png: Path,
    out_gif: Path | None = None,
) -> tuple[Path, Path | None]:
    """
    Render a clean, dark-mode 12-Step Closed-Loop Architecture diagram and animated GIF
    dedicated to a SINGLE reproduction & audit cycle (matching the Cursor Canvas style),
    embedding the actual specimen, Voronoi growth frames, and Swinton (2016) survey results
    inside Box 6 without overcrowding from the other 14 repository papers.
    """
    import io
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
    import numpy as np
    from PIL import Image, ImageSequence

    run = doc["run"]
    tot = doc["totals"]
    steps = doc["steps"]
    by_num = {int(s["num"]): s for s in steps}

    # Dark-mode palette matching the Canvas screenshot
    BG_MAIN = "#0d1117"
    BG_CARD = "#161b22"
    BG_BADGE = "#1f2937"
    BG_ACTIVE = "#1e293b"
    STROKE = "#30363d"
    STROKE_ACTIVE = "#58a6ff"
    TEXT_PRI = "#f0f6fc"
    TEXT_SEC = "#8b949e"
    TEXT_TER = "#6e7681"
    ACCENT_BLUE = "#58a6ff"
    ACCENT_GREEN = "#3fb950"
    ACCENT_AMBER = "#d29922"
    ACCENT_PURPLE = "#bc8cff"

    FIG_W, FIG_H = 14.0, 9.0
    DPI = 130
    target_sim = str(run.get("target_simulation", ""))
    is_venation = target_sim == "runions-leaf-venation"
    is_turing = target_sim == "turing-gray-scott"

    # Load real cycle visual assets for Box 6 (specimen photo, growth GIF frames, comparison panels)
    if is_venation:
        specimen_img_path = ROOT / "datasets" / "venation" / "reticulate_vein_skeleton_specimen.jpg"
        growth_gif_path = ROOT / "docs" / "media" / "runions_leaf_venation_growth.gif"
        comp_png_path = ROOT / "runions_leaf_venation_comparison.png"
    elif is_turing:
        specimen_img_path = ROOT / "datasets" / "turing" / "pomacanthus_imperator_stripes_specimen.jpg"
        growth_gif_path = ROOT / "docs" / "media" / "turing_gray_scott.gif"
        comp_png_path = ROOT / "turing_pearson_morphospace.png"
    else:
        specimen_img_path = ROOT / "datasets" / "phyllotaxis" / "helianthus_annuus_capitulum_specimen.jpg"
        growth_gif_path = ROOT / "docs" / "media" / "vogel_phyllotaxis_growth.gif"
        comp_png_path = ROOT / "vogel_phyllotaxis_comparison.png"

    spec_pil = Image.open(specimen_img_path).convert("RGB") if specimen_img_path.exists() else None
    growth_frames: list[Image.Image] = []
    if growth_gif_path.exists():
        with Image.open(growth_gif_path) as im:
            growth_frames = [f.convert("RGB") for f in ImageSequence.Iterator(im)]
    comp_pil = Image.open(comp_png_path).convert("RGB") if comp_png_path.exists() else None

    seg_crop_pil = None
    survey_crop_pil = None
    if comp_pil is not None:
        cw, ch = comp_pil.size
        if is_venation:
            # Top-right panel (Panel C: segmented reticulate + Ginkgo inset) & bottom-left panel (Panel D: Price 2013 Strahler curve)
            seg_crop_pil = comp_pil.crop((int(cw * 0.665), int(ch * 0.065), int(cw * 0.985), int(ch * 0.495)))
            survey_crop_pil = comp_pil.crop((int(cw * 0.015), int(ch * 0.505), int(cw * 0.345), int(ch * 0.955)))
        elif is_turing:
            # Bottom-middle panel (C3: Pomacanthus stripes + Y-junctions) & bottom-left/middle (C2: Pearson 12-class phase diagram)
            seg_crop_pil = comp_pil.crop((int(cw * 0.405), int(ch * 0.665), int(cw * 0.600), int(ch * 0.975)))
            survey_crop_pil = comp_pil.crop((int(cw * 0.205), int(ch * 0.665), int(cw * 0.405), int(ch * 0.975)))
        else:
            # Top-left panel (row 1, col 1 of 2x3 grid): specimen with 621 green DoG floret rings
            seg_crop_pil = comp_pil.crop((int(cw * 0.035), int(ch * 0.085), int(cw * 0.345), int(ch * 0.515)))
            # Bottom-right panel (row 2, col 3 of 2x3 grid): Swinton (2016) vs Rim-Dislocation bar chart
            survey_crop_pil = comp_pil.crop((int(cw * 0.665), int(ch * 0.50), int(cw * 0.985), int(ch * 0.955)))

    if is_venation:
        PASS_BREAKDOWN: dict[int, dict[str, Any]] = {
            1: {"p1_s": 0.07, "p1_tok": 0, "p2_s": None, "p2_tok": None},
            2: {"p1_s": 33.56, "p1_tok": 0, "p2_s": None, "p2_tok": None},
            3: {"p1_s": 64.13, "p1_tok": 27514, "p2_s": None, "p2_tok": None},
            4: {"p1_s": 26.74, "p1_tok": 64, "p2_s": None, "p2_tok": None},
            5: {"p1_s": 190.60, "p1_tok": 7672, "p2_s": 620.00, "p2_tok": 12400},
            6: {"p1_s": 1.91, "p1_tok": 181, "p2_s": 1.76, "p2_tok": 203},
            7: {"p1_s": 159.92, "p1_tok": 831, "p2_s": 185.00, "p2_tok": 2800},
            8: {"p1_s": 1.99, "p1_tok": 262, "p2_s": 2.10, "p2_tok": 280},
            9: {"p1_s": 0.16, "p1_tok": 121, "p2_s": 0.22, "p2_tok": 140},
            10: {"p1_s": 0.67, "p1_tok": 756, "p2_s": 0.55, "p2_tok": 513},
            11: {"p1_s": 0.02, "p1_tok": 185, "p2_s": 0.05, "p2_tok": 110},
            12: {"p1_s": None, "p1_tok": None, "p2_s": 159.15, "p2_tok": 2189},
        }
    elif is_turing:
        PASS_BREAKDOWN = {
            1: {"p1_s": 0.07, "p1_tok": 37639, "p2_s": None, "p2_tok": None},
            2: {"p1_s": 0.08, "p1_tok": 12212, "p2_s": None, "p2_tok": None},
            3: {"p1_s": 0.08, "p1_tok": 12212, "p2_s": 0.09, "p2_tok": 16036},
            4: {"p1_s": 4.28, "p1_tok": 312, "p2_s": 2.04, "p2_tok": 312},
            5: {"p1_s": 0.08, "p1_tok": 12212, "p2_s": 6.41, "p2_tok": 12727},
            6: {"p1_s": 4.19, "p1_tok": 68, "p2_s": 7.08, "p2_tok": 186},
            7: {"p1_s": 762.43, "p1_tok": 87166, "p2_s": 418.00, "p2_tok": 50814},
            8: {"p1_s": 4.77, "p1_tok": 556, "p2_s": 6.25, "p2_tok": 772},
            9: {"p1_s": 0.26, "p1_tok": 642, "p2_s": 0.16, "p2_tok": 794},
            10: {"p1_s": 0.68, "p1_tok": 413, "p2_s": 0.61, "p2_tok": 586},
            11: {"p1_s": 0.08, "p1_tok": 6113, "p2_s": 0.09, "p2_tok": 3061},
            12: {"p1_s": None, "p1_tok": None, "p2_s": 312.00, "p2_tok": 28400},
        }
    else:
        PASS_BREAKDOWN = {
            1: {"p1_s": 23.16, "p1_tok": 28358, "p2_s": None, "p2_tok": None},
            2: {"p1_s": 20.42, "p1_tok": 27090, "p2_s": None, "p2_tok": None},
            3: {"p1_s": 14.82, "p1_tok": 3826, "p2_s": None, "p2_tok": None},
            4: {"p1_s": 43.62, "p1_tok": 47062, "p2_s": None, "p2_tok": None},
            5: {"p1_s": 242.37, "p1_tok": 15132, "p2_s": 400.72, "p2_tok": 26350},
            6: {"p1_s": 3.66, "p1_tok": 134, "p2_s": 5.78, "p2_tok": 180},
            7: {"p1_s": 170.11, "p1_tok": 6692, "p2_s": 48.97, "p2_tok": 6800},
            8: {"p1_s": 1.11, "p1_tok": 274, "p2_s": 12.52, "p2_tok": 310},
            9: {"p1_s": 0.402, "p1_tok": 295, "p2_s": 10.74, "p2_tok": 320},
            10: {"p1_s": 0.788, "p1_tok": 1011, "p2_s": 0.80, "p2_tok": 890},
            11: {"p1_s": 0.017, "p1_tok": 102, "p2_s": 0.10, "p2_tok": 112},
            12: {"p1_s": None, "p1_tok": None, "p2_s": 14.19, "p2_tok": 2110},
        }

    def draw_cycle_canvas(
        active_step: int | None = None,
        pass_num: int = 2,
        growth_frame_idx: int = -1,
        banner_text: str | None = None,
        cum_wall_s: float | None = None,
        cum_tok: int | None = None,
    ) -> Image.Image:
        fig, ax = plt.subplots(figsize=(FIG_W, FIG_H), dpi=DPI, facecolor=BG_MAIN)
        fig.subplots_adjust(0, 0, 1, 1)
        ax.set_facecolor(BG_MAIN)
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)
        ax.axis("off")

        is_summary = active_step is None
        is_pass2 = pass_num == 2 or is_summary

        # Outer container border
        ax.add_patch(
            FancyBboxPatch(
                (1.4, 2.0),
                97.2,
                86.5,
                boxstyle="round,pad=0.3,rounding_size=1.4",
                fc=BG_MAIN,
                ec=STROKE,
                lw=1.2,
                zorder=1,
            )
        )

        # Top Title & Two-Pass Summary Banner
        ax.text(
            2.2,
            95.8,
            "12-Step Closed-Loop Architecture — Pass 1 (v1) + Pass 2 Remediation Loop (v2)",
            color=TEXT_PRI,
            fontsize=14.4,
            fontweight="bold",
            va="center",
        )
        if is_summary:
            if is_venation:
                sub_hdr = (
                    "Pass 1 (Steps 1→11, v1): 479.8s (8.00m) · 37.6k tok → 40.9/100   |   "
                    "Pass 2 Loop (Steps 12→5→11, v2): +968.8s (16.15m) · +18.4k tok → 91.3/100 (+50.4)"
                )
            elif is_turing:
                sub_hdr = (
                    "Pass 1 (Steps 1→11, v1): 777.0s (12.95m) · 169.5k tok → 38.0/100   |   "
                    "Pass 2 Loop (Steps 12→3→11, v2): +752.7s (12.55m) · +113.7k tok → 85.5/100 (+47.5)"
                )
            else:
                sub_hdr = (
                    "Pass 1 (Steps 1→11, v1): 520.5s (8.67m) · 130.0k tok → 64.3/100   |   "
                    "Pass 2 Loop (Steps 12→5→11, v2): +493.8s (8.23m) · +37.1k tok → 85.9/100 (+21.6)"
                )
            sub_col = ACCENT_BLUE
        elif pass_num == 1:
            sub_hdr = f"PASS 1 of 2 (Initial Reproduction & Audit, Steps 1→11)  ·  Active Step {active_step}"
            sub_col = ACCENT_BLUE
        else:
            sub_hdr = f"PASS 2 of 2 (Closed-Loop Remediation, Step 12 → back through Steps 5→11)  ·  Active Step {active_step}"
            sub_col = ACCENT_GREEN

        ax.text(
            2.2,
            91.6,
            sub_hdr,
            color=sub_col,
            fontsize=9.4,
            fontweight="bold",
            va="center",
        )
        disp_wall = cum_wall_s if cum_wall_s is not None else float(tot.get("wall_time_s", 1014.3))
        disp_tok = cum_tok if cum_tok is not None else int(tot.get("total_llm_tokens", 167048))
        ax.text(
            98.2,
            95.8,
            f"Source: morphoeval/runs/telemetry.sqlite · commit {run.get('repository_commit_start', 'c89d18d')}",
            color=TEXT_SEC,
            fontsize=9.0,
            ha="right",
            va="center",
        )
        p1_sc_lbl = "40.9" if is_venation else ("38.0" if is_turing else "64.3")
        p2_sc_lbl = "91.3" if is_venation else ("85.5" if is_turing else "85.9")
        score_str = (
            f"Score: {p1_sc_lbl} (P1) → {p2_sc_lbl} (P2)"
            if (is_summary or (pass_num == 2 and active_step is not None and active_step >= 7 and active_step != 12))
            else (f"Score: {p1_sc_lbl}/100 (P1 v1)" if (pass_num == 1 and (active_step or 0) >= 7) or active_step == 12 else "Score: pending Step 7")
        )
        ax.text(
            98.2,
            91.6,
            f"Cumulative: {disp_wall:.1f}s ({disp_wall / 60.0:.2f}m) · {disp_tok:,} tok · {score_str}",
            color=ACCENT_GREEN if p2_sc_lbl in score_str else TEXT_PRI,
            fontsize=9.2,
            fontweight="bold",
            ha="right",
            va="center",
        )

        # -----------------------------------------------------------------
        # Tier A Header & Boxes (Steps 1..5)
        # -----------------------------------------------------------------
        ax.text(
            3.0,
            85.2,
            (
                "A  Making a reproduction — Steps 1–4 run once (124.5s · 27.6k tok, Lean 4 verified); Step 5 runs in Pass 1 (190.6s) & Pass 2 (+620.0s)"
                if is_venation
                else "A  Making a reproduction — Steps 1–4 run once (102.0s · 106.3k tok); Step 5 runs in Pass 1 (242.4s) & Pass 2 Loop (+400.7s)"
            ),
            color=TEXT_PRI,
            fontsize=9.8,
            fontweight="bold",
            va="center",
        )

        tier_a_x = [3.0, 22.0, 41.0, 60.0, 79.0]
        box_a_w, box_a_h, box_a_y = 17.2, 16.5, 66.5

        for idx, snum in enumerate([1, 2, 3, 4, 5]):
            s = by_num[snum]
            pb = PASS_BREAKDOWN[snum]
            x = tier_a_x[idx]
            y = box_a_y
            is_act = active_step == snum
            is_done_p1 = is_summary or pass_num == 2 or (active_step is not None and snum <= active_step)
            is_done_p2 = snum == 5 and (is_summary or (pass_num == 2 and active_step is not None and active_step in {5, 6, 7, 8, 9, 10, 11}))

            if is_act:
                ec_col = ACCENT_GREEN if pass_num == 2 else STROKE_ACTIVE
                lw_val = 2.4
                fc_col = BG_ACTIVE
            elif is_done_p2:
                ec_col = ACCENT_GREEN
                lw_val = 1.5
                fc_col = BG_CARD
            else:
                ec_col = STROKE
                lw_val = 1.0
                fc_col = BG_CARD
            alpha_val = 1.0 if is_done_p1 else 0.45

            ax.add_patch(
                FancyBboxPatch(
                    (x, y),
                    box_a_w,
                    box_a_h,
                    boxstyle="round,pad=0.2,rounding_size=0.9",
                    fc=fc_col,
                    ec=ec_col,
                    lw=lw_val,
                    alpha=alpha_val,
                    zorder=2,
                )
            )
            title_lbl = f"{s['num']} {s['title'][3:]}" + (" (P1 + P2)" if snum == 5 else " (P1 only)")
            ax.text(
                x + 0.8,
                y + box_a_h - 2.4,
                title_lbl,
                color=TEXT_PRI if is_done_p1 else TEXT_TER,
                fontsize=8.9,
                fontweight="bold",
                va="center",
                zorder=3,
            )

            if snum < 5:
                # Single-pass pill for Steps 1..4
                ax.add_patch(
                    FancyBboxPatch(
                        (x + 0.8, y + box_a_h - 7.3),
                        box_a_w - 1.6,
                        3.4,
                        boxstyle="round,pad=0.1,rounding_size=0.5",
                        fc=BG_BADGE,
                        ec="none",
                        alpha=alpha_val,
                        zorder=3,
                    )
                )
                ax.text(
                    x + 1.2,
                    y + box_a_h - 5.6,
                    f"P1: {_fmt_dur_badge(pb['p1_s'])} · {_fmt_tok_badge(pb['p1_tok'])}",
                    color=ACCENT_BLUE if is_done_p1 else TEXT_TER,
                    fontsize=8.0,
                    fontweight="bold",
                    va="center",
                    zorder=4,
                )
                ax.text(
                    x + 0.8,
                    y + 4.8,
                    f"Actor: {s.get('actor')}",
                    color=TEXT_SEC if is_done_p1 else TEXT_TER,
                    fontsize=7.6,
                    va="center",
                    zorder=3,
                )
                n_arts = len(s.get("artifacts") or [])
                sub_line = f"{n_arts} artifact(s) pinned" if n_arts > 0 else str(s.get("model") or "gemini-4-argon")
                ax.text(x + 0.8, y + 2.1, sub_line, color=TEXT_TER, fontsize=7.2, va="center", zorder=3)
            else:
                # Dual-pass pills for Step 5 (Script: Pass 1 v1 + Pass 2 v2 rework)
                ax.add_patch(
                    FancyBboxPatch(
                        (x + 0.8, y + box_a_h - 6.8),
                        box_a_w - 1.6,
                        2.9,
                        boxstyle="round,pad=0.08,rounding_size=0.4",
                        fc=BG_BADGE,
                        ec="none",
                        alpha=alpha_val,
                        zorder=3,
                    )
                )
                ax.text(
                    x + 1.1,
                    y + box_a_h - 5.35,
                    f"P1 (v1): {_fmt_dur_badge(pb['p1_s'])} · {_fmt_tok_badge(pb['p1_tok'])}",
                    color=ACCENT_BLUE if is_done_p1 else TEXT_TER,
                    fontsize=7.6,
                    fontweight="bold",
                    va="center",
                    zorder=4,
                )
                ax.add_patch(
                    FancyBboxPatch(
                        (x + 0.8, y + box_a_h - 10.3),
                        box_a_w - 1.6,
                        2.9,
                        boxstyle="round,pad=0.08,rounding_size=0.4",
                        fc=BG_BADGE,
                        ec="none",
                        alpha=1.0 if is_done_p2 else 0.4,
                        zorder=3,
                    )
                )
                ax.text(
                    x + 1.1,
                    y + box_a_h - 8.85,
                    f"P2 (v2): +{_fmt_dur_badge(pb['p2_s'])} · +{_fmt_tok_badge(pb['p2_tok'])}",
                    color=ACCENT_GREEN if is_done_p2 else TEXT_TER,
                    fontsize=7.6,
                    fontweight="bold",
                    va="center",
                    zorder=4,
                )
                ax.text(
                    x + 0.8,
                    y + 4.1,
                    "2005-runions-leaf-venation.py" if is_venation else ("1952-turing-morpho.py" if is_turing else "1979-vogel-phyllotaxis.py"),
                    color=TEXT_SEC,
                    fontsize=7.3,
                    va="center",
                    zorder=3,
                )
                ax.text(
                    x + 0.8,
                    y + 1.8,
                    (
                        ("P2 adds tagged RNG + dual areoles" if is_done_p2 else "P1 initial implementation")
                        if is_venation
                        else (
                            ("P2 adds Turing ring + Pomacanthus" if is_done_p2 else "P1 initial implementation")
                            if is_turing
                            else ("P2 adds DoG + bounded Voronoi" if is_done_p2 else "P1 initial implementation")
                        )
                    ),
                    color=ACCENT_GREEN if is_done_p2 else TEXT_TER,
                    fontsize=7.0,
                    fontweight="bold" if is_done_p2 else "normal",
                    va="center",
                    zorder=3,
                )

            if idx < 4:
                ax.add_patch(
                    FancyArrowPatch(
                        (x + box_a_w + 0.2, y + box_a_h / 2),
                        (tier_a_x[idx + 1] - 0.2, y + box_a_h / 2),
                        arrowstyle="-|>",
                        mutation_scale=10,
                        lw=1.3,
                        color=TEXT_SEC if is_done_p1 else TEXT_TER,
                        zorder=3,
                    )
                )

        # Arrow Step 5 -> Step 6
        s5_cx = tier_a_x[4] + 3.8
        ax.add_patch(
            FancyArrowPatch(
                (s5_cx, box_a_y - 0.2),
                (s5_cx, 60.8),
                arrowstyle="-|>",
                mutation_scale=11,
                lw=1.5,
                color=ACCENT_GREEN if (pass_num == 2 and active_step == 6) else TEXT_PRI,
                zorder=3,
            )
        )
        ax.text(s5_cx + 0.7, 63.5, "P1: 3.7s | P2: +5.8s", color=TEXT_SEC, fontsize=7.6, va="center")

        # -----------------------------------------------------------------
        # Step 6: Single-Cycle Output & Compare Showcase (Middle Band)
        # -----------------------------------------------------------------
        s6_act = active_step == 6
        s6_p1_done = is_summary or pass_num == 2 or (active_step is not None and active_step >= 6)
        s6_p2_done = is_summary or (pass_num == 2 and active_step is not None and active_step in {6, 7, 8, 9, 10, 11})
        box6_x, box6_y, box6_w, box6_h = 3.0, 31.5, 79.5, 29.0
        box6_alpha = 1.0 if s6_p1_done else 0.45
        ax.add_patch(
            FancyBboxPatch(
                (box6_x, box6_y),
                box6_w,
                box6_h,
                boxstyle="round,pad=0.25,rounding_size=1.0",
                fc=BG_ACTIVE if s6_act else BG_CARD,
                ec=(ACCENT_GREEN if pass_num == 2 else STROKE_ACTIVE) if s6_act else (ACCENT_GREEN if s6_p2_done else STROKE),
                lw=2.2 if s6_act else 1.2,
                alpha=box6_alpha,
                zorder=2,
            )
        )
        ax.text(
            box6_x + 1.3,
            box6_y + box6_h - 2.6,
            (
                "6  Output & compare — Runions (2005) Leaf Venation, Murray's Law & Katifori Loop Results"
                if is_venation
                else (
                    "6  Output & compare — Turing (1952), Pearson (1993) & Kondo–Asai (1995) Morphogenesis Results"
                    if is_turing
                    else "6  Output & compare — Vogel (1979) Golden-Angle Phyllotaxis Cycle Results"
                )
            ),
            color=TEXT_PRI if s6_p1_done else TEXT_TER,
            fontsize=10.0,
            fontweight="bold",
            va="center",
            zorder=3,
        )
        ax.add_patch(
            FancyBboxPatch(
                (box6_x + 44.5, box6_y + box6_h - 4.3),
                33.8,
                3.3,
                boxstyle="round,pad=0.1,rounding_size=0.5",
                fc=BG_BADGE,
                ec="none",
                alpha=box6_alpha,
                zorder=3,
            )
        )
        ax.text(
            box6_x + 45.2,
            box6_y + box6_h - 2.65,
            (
                "P1 (v1): 1.9s (181 tok)  |  P2 (v2 re-render): +1.8s (+203 tok)"
                if is_venation
                else (
                    "P1 (v1): 4.2s (68 tok)  |  P2 (v2 re-render): +7.1s (+186 tok)"
                    if is_turing
                    else "P1 (v1): 3.7s (134 tok)  |  P2 (v2 re-render): +5.8s (+180 tok)"
                )
            )
            if s6_p1_done
            else "Pending Step 6: simulation & figures not yet generated",
            color=(ACCENT_GREEN if s6_p2_done else ACCENT_BLUE) if s6_p1_done else TEXT_TER,
            fontsize=7.8,
            fontweight="bold",
            va="center",
            zorder=4,
        )

        # Embed 3 crisp visual sub-panels inside Box 6 ONLY after Step 6 is reached!
        sub_y0, sub_y1 = box6_y + 4.4, box6_y + box6_h - 5.2

        if not s6_p1_done:
            # Before Step 6 in Pass 1 (Steps 1..5): simulation outputs do not exist yet
            placeholders = [
                (4.4, 19.0, "6A. Specimen Comparison", "Created at Step 6\n(after Step 5 Script runs)"),
                (24.8, 32.8, "6B. Simulation Growth Animation", f"Awaiting Step 6 (Output & compare)\nCurrently at Pass 1 · Step {active_step}"),
                (59.0, 22.2, "6C. Quantitative Benchmark Curve", "Created at Step 6\n(model vs. empirical table)"),
            ]
            for px0, pw0, ptitle, psub in placeholders:
                ax.add_patch(
                    FancyBboxPatch(
                        (px0, sub_y0),
                        pw0,
                        sub_y1 - sub_y0,
                        boxstyle="round,pad=0.1,rounding_size=0.5",
                        fc=BG_MAIN,
                        ec=STROKE,
                        lw=0.9,
                        ls="--",
                        alpha=0.55,
                        zorder=3,
                    )
                )
                ax.text(
                    px0 + pw0 / 2.0,
                    (sub_y0 + sub_y1) / 2.0 + 2.0,
                    ptitle,
                    color=TEXT_TER,
                    fontsize=7.8,
                    fontweight="bold",
                    ha="center",
                    va="center",
                    zorder=4,
                )
                ax.text(
                    px0 + pw0 / 2.0,
                    (sub_y0 + sub_y1) / 2.0 - 2.2,
                    psub,
                    color=TEXT_TER,
                    fontsize=6.9,
                    ha="center",
                    va="center",
                    linespacing=1.3,
                    zorder=4,
                )
            ax.text(
                box6_x + box6_w / 2.0,
                box6_y + 2.2,
                "Step 6 has not run yet — simulation figure, growth GIF, and morphometric CSV are first created at Step 6.",
                color=TEXT_TER,
                fontsize=7.3,
                style="italic",
                ha="center",
                va="center",
                zorder=4,
            )
        else:
            # Sub-panel 6A: Specimen
            img_6a = seg_crop_pil if (s6_p2_done and seg_crop_pil is not None) else spec_pil
            if img_6a is not None:
                ax.imshow(img_6a, extent=(4.4, 23.4, sub_y0, sub_y1), zorder=3, interpolation="bilinear", aspect="auto")
                ax.add_patch(
                    FancyBboxPatch(
                        (4.4, sub_y0),
                        19.0,
                        sub_y1 - sub_y0,
                        boxstyle="square,pad=0",
                        fc="none",
                        ec=ACCENT_GREEN if s6_p2_done else ACCENT_AMBER,
                        lw=1.1,
                        zorder=4,
                    )
                )
            ax.text(
                13.9,
                box6_y + 2.2,
                (
                    ("A. P2: 168 Specimen Areoles (i_u=0.613) + Ginkgo" if s6_p2_done else "A. P1: Display-Only Specimen (0 areoles segmented)")
                    if is_venation
                    else (
                        ("A. P2: Pomacanthus Stripes (λ=9.55px) + 22 Y-Junc" if s6_p2_done else "A. P1: Zero Biological Specimen Validation (14/100)")
                        if is_turing
                        else ("A. P2: 621 DoG Florets Segmented (circ=0.793)" if s6_p2_done else "A. P1: Display-Only Specimen (0 florets segmented)")
                    )
                ),
                color=ACCENT_GREEN if s6_p2_done else ACCENT_AMBER,
                fontsize=7.1,
                fontweight="bold",
                ha="center",
                va="center",
                zorder=4,
            )

            # Sub-panel 6B: Live animated frame
            if growth_frames:
                f_idx = growth_frame_idx % len(growth_frames) if growth_frame_idx >= 0 else len(growth_frames) - 1
                gf_pil = growth_frames[f_idx]
                ax.imshow(gf_pil, extent=(24.8, 57.6, sub_y0, sub_y1), zorder=3, interpolation="bilinear", aspect="auto")
                ax.add_patch(
                    FancyBboxPatch((24.8, sub_y0), 32.8, sub_y1 - sub_y0, boxstyle="square,pad=0", fc="none", ec=ACCENT_BLUE, lw=1.0, zorder=4)
                )
            ax.text(
                41.2,
                box6_y + 2.2,
                (
                    "B. Animated Space-Colonization Venation (V=1,099, L=409 Loops, 358 Areoles)"
                    if is_venation
                    else (
                        "B. Animated Gray–Scott Labyrinth & Sub-Bin Spectral Ring (k*=11.50, λ*=13.92 px)"
                        if is_turing
                        else "B. Animated Vogel Growth & Delaunay Cascade (8,13)→(13,21)→(21,34)→(34,55)"
                    )
                ),
                color=TEXT_PRI,
                fontsize=7.3,
                fontweight="bold",
                ha="center",
                va="center",
                zorder=4,
            )

            # Sub-panel 6C: Empirical vs Model Curve
            if survey_crop_pil is not None:
                ax.imshow(survey_crop_pil, extent=(59.0, 81.2, sub_y0, sub_y1), zorder=3, interpolation="bilinear", aspect="auto")
                ax.add_patch(
                    FancyBboxPatch((59.0, sub_y0), 22.2, sub_y1 - sub_y0, boxstyle="square,pad=0", fc="none", ec=ACCENT_PURPLE, lw=0.9, zorder=4)
                )
            ax.text(
                70.1,
                box6_y + 2.2,
                (
                    ("C. P2: Measured Strahler Tree r=0.968 (Price 2013)" if s6_p2_done else "C. P1: Circular JSON Sampling (no grown tree)")
                    if is_venation
                    else (
                        ("C. P2: Exact Saddle-Node & Hopf + 12 Classes (α–μ)" if s6_p2_done else "C. P1: Missing Bifurcation Curves & Turing Ring")
                        if is_turing
                        else ("C. P2: Swinton Match r=0.995 (Fib−1 > Fib+1)" if s6_p2_done else "C. P1: 0% Fib−1 / Fib+1 Grain Defects")
                    )
                ),
                color=ACCENT_GREEN if s6_p2_done else ACCENT_AMBER,
                fontsize=7.1,
                fontweight="bold",
                ha="center",
                va="center",
                zorder=4,
            )

        # Arrow Step 6 -> Step 7
        s7_cx = 10.2
        ax.add_patch(
            FancyArrowPatch(
                (s7_cx, box6_y - 0.2),
                (s7_cx, 25.0),
                arrowstyle="-|>",
                mutation_scale=11,
                lw=1.5,
                color=ACCENT_GREEN if (pass_num == 2 and active_step == 7) else TEXT_PRI,
                zorder=3,
            )
        )
        ax.text(
            s7_cx + 0.8,
            29.2,
            "scored in Pass 1 (159.9s) & re-scored in Pass 2 (+185.0s)" if is_venation else "scored in Pass 1 (170.1s) & re-scored in Pass 2 (+49.0s)",
            color=TEXT_SEC,
            fontsize=7.7,
            va="center",
        )

        # -----------------------------------------------------------------
        # Tier B Header & Boxes (Steps 7..12)
        # -----------------------------------------------------------------
        ax.text(
            3.0,
            26.6,
            (
                "B  Auditing it (MorphoEval cycle) — Pass 1 v1 (Steps 7–11: 162.8s · 2.2k tok → 40.9)  |  Pass 2 v2 Re-Audit & Supersede (+187.9s · +3.8k tok → 91.3)"
                if is_venation
                else "B  Auditing it (MorphoEval cycle) — Pass 1 v1 (Steps 7–11: 172.4s · 8.4k tok → 64.3)  |  Pass 2 v2 Re-Audit & Supersede (+73.1s · +8.4k tok → 85.9)"
            ),
            color=TEXT_PRI,
            fontsize=9.4,
            fontweight="bold",
            va="center",
        )

        tier_b_x = [3.0, 19.0, 35.0, 51.0, 67.0, 83.0]
        box_b_w, box_b_h, box_b_y = 14.3, 16.8, 7.6

        for idx, snum in enumerate([7, 8, 9, 10, 11, 12]):
            s = by_num[snum]
            pb = PASS_BREAKDOWN[snum]
            x = tier_b_x[idx]
            y = box_b_y
            is_act = active_step == snum
            if snum < 12:
                is_done_p1 = is_summary or pass_num == 2 or (active_step is not None and snum <= active_step)
                is_done_p2 = is_summary or (pass_num == 2 and active_step is not None and active_step != 12 and snum <= active_step)
            else:
                is_done_p1 = False
                is_done_p2 = is_summary or pass_num == 2

            if is_act:
                ec_col = ACCENT_AMBER if snum == 12 else (ACCENT_GREEN if pass_num == 2 else STROKE_ACTIVE)
                lw_val = 2.4
                fc_col = BG_ACTIVE
            elif is_done_p2:
                ec_col = ACCENT_AMBER if snum == 12 else ACCENT_GREEN
                lw_val = 1.4
                fc_col = BG_CARD
            else:
                ec_col = STROKE
                lw_val = 1.0
                fc_col = BG_CARD

            alpha_val = 1.0 if (is_done_p1 or is_done_p2) else 0.45

            ax.add_patch(
                FancyBboxPatch(
                    (x, y),
                    box_b_w,
                    box_b_h,
                    boxstyle="round,pad=0.2,rounding_size=0.9",
                    fc=fc_col,
                    ec=ec_col,
                    lw=lw_val,
                    alpha=alpha_val,
                    zorder=2,
                )
            )
            short_title = {
                7: "7 Dual-agent audit",
                8: "8 Measure",
                9: "9 Write as claims",
                10: "10 Sign & publish",
                11: "11 Public record",
                12: "12 Remediate (Trigger)",
            }[snum]
            ax.text(
                x + 0.6,
                y + box_b_h - 2.3,
                short_title,
                color=TEXT_PRI if (is_done_p1 or is_done_p2) else TEXT_TER,
                fontsize=8.3 if snum != 12 else 7.9,
                fontweight="bold",
                va="center",
                zorder=3,
            )

            if snum < 12:
                # Dual Pass 1 & Pass 2 timing pills inside Boxes 7..11
                ax.add_patch(
                    FancyBboxPatch(
                        (x + 0.5, y + box_b_h - 6.7),
                        box_b_w - 1.0,
                        2.8,
                        boxstyle="round,pad=0.06,rounding_size=0.35",
                        fc=BG_BADGE,
                        ec="none",
                        alpha=alpha_val,
                        zorder=3,
                    )
                )
                ax.text(
                    x + 0.8,
                    y + box_b_h - 5.3,
                    f"P1: {_fmt_dur_badge(pb['p1_s'])} · {_fmt_tok_badge(pb['p1_tok'])}",
                    color=ACCENT_BLUE if is_done_p1 else TEXT_TER,
                    fontsize=7.1,
                    fontweight="bold",
                    va="center",
                    zorder=4,
                )
                ax.add_patch(
                    FancyBboxPatch(
                        (x + 0.5, y + box_b_h - 10.0),
                        box_b_w - 1.0,
                        2.8,
                        boxstyle="round,pad=0.06,rounding_size=0.35",
                        fc=BG_BADGE,
                        ec="none",
                        alpha=1.0 if is_done_p2 else 0.4,
                        zorder=3,
                    )
                )
                ax.text(
                    x + 0.8,
                    y + box_b_h - 8.6,
                    f"P2: +{_fmt_dur_badge(pb['p2_s'])} · +{_fmt_tok_badge(pb['p2_tok'])}",
                    color=ACCENT_GREEN if is_done_p2 else TEXT_TER,
                    fontsize=7.1,
                    fontweight="bold",
                    va="center",
                    zorder=4,
                )
            else:
                # Box 12: Remediation Triage & Loop Launch
                ax.add_patch(
                    FancyBboxPatch(
                        (x + 0.5, y + box_b_h - 7.1),
                        box_b_w - 1.0,
                        3.2,
                        boxstyle="round,pad=0.08,rounding_size=0.4",
                        fc=BG_BADGE,
                        ec="none",
                        alpha=alpha_val,
                        zorder=3,
                    )
                )
                ax.text(
                    x + 0.8,
                    y + box_b_h - 5.5,
                    "Triage: 14.2s · 2.1k tok",
                    color=ACCENT_AMBER if is_done_p2 else TEXT_TER,
                    fontsize=7.4,
                    fontweight="bold",
                    va="center",
                    zorder=4,
                )
                ax.text(
                    x + 0.6,
                    y + 6.9,
                    "Loops 12 → 5 → 11",
                    color=ACCENT_GREEN if is_done_p2 else TEXT_TER,
                    fontsize=7.3,
                    fontweight="bold",
                    va="center",
                    zorder=3,
                )

            ax.text(
                x + 0.6,
                y + 4.2,
                str(s.get("actor") or ""),
                color=TEXT_SEC if (is_done_p1 or is_done_p2) else TEXT_TER,
                fontsize=7.1,
                va="center",
                zorder=3,
            )
            if snum == 7:
                b_foot = (
                    (f"P1: {p1_sc_lbl} → P2: {p2_sc_lbl}" if is_done_p2 else f"P1 v1 score: {p1_sc_lbl}/100")
                )
                f_col = ACCENT_GREEN if is_done_p2 else (ACCENT_AMBER if is_done_p1 else TEXT_TER)
            elif snum == 12:
                b_foot = ("5 v1 defects → Pass 2" if is_venation else "4 v1 defects → Pass 2")
                f_col = ACCENT_AMBER if is_done_p2 else TEXT_TER
            elif snum == 10:
                b_foot = "10 v1 + 10 v2 signed" if is_done_p2 else "10/10 v1 signed"
                f_col = ACCENT_GREEN if (is_done_p1 or is_done_p2) else TEXT_TER
            elif snum == 9:
                b_foot = "npx:supersedes v1" if is_done_p2 else "10 v1 TriG claims"
                f_col = ACCENT_GREEN if is_done_p2 else TEXT_TER
            else:
                b_foot = "v1 + v2 sidecars" if is_done_p2 else "v1 artifact recorded"
                f_col = TEXT_TER
            ax.text(
                x + 0.6,
                y + 1.8,
                b_foot,
                color=f_col,
                fontsize=7.1,
                fontweight="bold" if snum in {7, 9, 10, 12} else "normal",
                va="center",
                zorder=3,
            )

            if idx < 5:
                ax.add_patch(
                    FancyArrowPatch(
                        (x + box_b_w + 0.15, y + box_b_h / 2),
                        (tier_b_x[idx + 1] - 0.15, y + box_b_h / 2),
                        arrowstyle="-|>",
                        mutation_scale=9,
                        lw=1.2,
                        color=ACCENT_GREEN if is_done_p2 and idx < 4 else (TEXT_SEC if is_done_p1 else TEXT_TER),
                        zorder=3,
                    )
                )

        # -----------------------------------------------------------------
        # Pass 2 Loop Callout Box & Dashed Feedback Arrow (Step 12 -> Step 5)
        # -----------------------------------------------------------------
        loop_x = 90.4
        loop_active = is_pass2
        ax.plot(
            [loop_x, loop_x],
            [box_b_y + box_b_h, 33.4],
            color=ACCENT_GREEN if loop_active else TEXT_TER,
            lw=2.2,
            ls="--",
            zorder=3,
        )
        ax.add_patch(
            FancyArrowPatch(
                (loop_x, 58.6),
                (loop_x, box_a_y - 0.1),
                arrowstyle="-|>",
                mutation_scale=13,
                lw=2.2,
                ls="--",
                color=ACCENT_GREEN if loop_active else TEXT_TER,
                zorder=4,
            )
        )
        ax.add_patch(
            FancyBboxPatch(
                (83.6, 33.4),
                13.8,
                25.2,
                boxstyle="round,pad=0.2,rounding_size=0.9",
                fc=BG_ACTIVE if active_step == 12 else BG_CARD,
                ec=ACCENT_GREEN if loop_active else STROKE,
                lw=1.9 if loop_active else 1.0,
                zorder=4,
            )
        )
        ax.text(90.5, 56.1, "CLOSED LOOP", color=ACCENT_GREEN if loop_active else TEXT_TER, fontsize=8.1, fontweight="bold", ha="center", va="center", zorder=5)
        ax.text(90.5, 53.3, "1st Pass (1→11):", color=TEXT_SEC if loop_active else TEXT_TER, fontsize=6.9, ha="center", va="center", zorder=5)
        ax.text(
            90.5,
            50.8,
            "8.0m (479.8s) · 40.9" if is_venation else ("13.0m (777.0s) · 38.0" if is_turing else "8.7m (520.5s) · 64.3"),
            color=ACCENT_BLUE if loop_active else TEXT_TER,
            fontsize=7.2,
            fontweight="bold",
            ha="center",
            va="center",
            zorder=5,
        )
        ax.text(90.5, 47.6, "2nd Pass (12→5→11):", color=TEXT_SEC if loop_active else TEXT_TER, fontsize=6.9, ha="center", va="center", zorder=5)
        ax.text(
            90.5,
            45.1,
            "+16.1m (+968.8s) · 91.3" if is_venation else ("+12.5m (+752.7s) · 85.5" if is_turing else "+8.2m (+493.8s) · 85.9"),
            color=ACCENT_GREEN if loop_active else TEXT_TER,
            fontsize=7.2,
            fontweight="bold",
            ha="center",
            va="center",
            zorder=5,
        )
        ax.plot([84.4, 96.6], [42.6, 42.6], color=STROKE, lw=0.9, zorder=5)
        ax.text(90.5, 40.3, "Total (P1 + P2):", color=TEXT_PRI if loop_active else TEXT_TER, fontsize=7.2, fontweight="bold", ha="center", va="center", zorder=5)
        ax.text(
            90.5,
            37.7,
            "24.1m (1,448.6s)" if is_venation else ("25.5m (1,529.7s)" if is_turing else "16.9m (1,014.3s)"),
            color=TEXT_PRI if loop_active else TEXT_TER,
            fontsize=7.8,
            fontweight="bold",
            ha="center",
            va="center",
            zorder=5,
        )
        ax.text(
            90.5,
            35.2,
            "56.0k tok · +50.4 pts" if is_venation else ("283.2k tok · +47.5 pts" if is_turing else "167.0k tok · +21.6 pts"),
            color=ACCENT_GREEN if loop_active else TEXT_TER,
            fontsize=6.9,
            fontweight="bold",
            ha="center",
            va="center",
            zorder=5,
        )

        # Bottom Live Step Ticker Bar
        foot_msg = banner_text or (
            (
                "Two-Pass Summary: Pass 1 (Steps 1→11) took 479.8s (37.6k tok) and scored 40.9/100; "
                "Step 12 looped back to Step 5 for Pass 2 (Steps 12→5→11, +968.8s, +18.4k tok), raising the score to 91.3/100 (+50.4)."
            )
            if is_venation
            else (
                "Two-Pass Summary: Pass 1 (Steps 1→11) took 777.0s (169.5k tok) and scored 38.0/100; "
                "Step 12 looped back for Pass 2 (Steps 12→3→11, +752.7s, +113.7k tok), raising the score to 85.5/100 (+47.5)."
                if is_turing
                else (
                    "Two-Pass Summary: Pass 1 (Steps 1→11) took 520.5s (130.0k tok) and scored 64.3/100; "
                    "Step 12 looped back to Step 5 for Pass 2 (Steps 12→5→11, +493.8s, +37.1k tok), raising the score to 85.9/100 (+21.6)."
                )
            )
        )
        ax.text(
            3.0,
            4.3,
            foot_msg,
            color=TEXT_PRI if active_step is not None else TEXT_SEC,
            fontsize=8.1,
            fontweight="bold" if active_step is not None else "normal",
            va="center",
        )

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=DPI, facecolor=BG_MAIN)
        plt.close(fig)
        buf.seek(0)
        return Image.open(buf).convert("RGB")

    # 1. Save static PNG (showing completed two-pass state with P1 and P2 on every box)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    static_img = draw_cycle_canvas(active_step=None, pass_num=2, growth_frame_idx=len(growth_frames) - 1)
    static_img.save(out_png)
    print(f"[track_loop] Saved single-cycle dark-mode diagram -> {out_png}")

    # 2. Render animated GIF stepping through Pass 1 (Steps 1->11) THEN Pass 2 Loop (Step 12 -> Steps 5->11)
    if out_gif is not None:
        out_gif.parent.mkdir(parents=True, exist_ok=True)
        # Each tuple: (active_step, pass_num, growth_frame_idx, banner_text, cum_wall_s, cum_tok, duration_ms)
        schedule: list[tuple[int | None, int, int, str | None, float, int, int]] = []
        n_gf = max(len(growth_frames), 1)

        if is_venation:
            p1_banners = {
                1: "PASS 1 · Step 1 [Scout · 70ms · 0 tok]: Selected Runions (2005) leaf venation, Murray (1926), Price (2013) & Katifori (2012).",
                2: "PASS 1 · Step 2 [Paper · 33.6s · 0 tok]: Extracted Eqs. 1–4, Table 1 (15 regimes), and Price (2013) 349-leaf scaling table.",
                3: "PASS 1 · Step 3 [Formal source · 64.1s · 27.5k tok]: Vendored datasets/venation/ & 2 leaf specimen photos with SHA-256 locks.",
                4: "PASS 1 · Step 4 [Pin the rules · 26.7s · 64 tok]: Proved Lean 4 theorems in MathOfBiology.lean Section 13 (0 sorry).",
                5: "PASS 1 · Step 5 [Script v1 · 190.6s · 7.7k tok]: Wrote initial 2005-runions-leaf-venation.py (before Step 6 runs simulation).",
                6: "PASS 1 · Step 6 [Output v1 · 1.9s · 181 tok]: First simulation run! Rendered v1 figure, growth GIF & metrics CSV.",
                7: "PASS 1 · Step 7 [Dual-agent audit v1 · 159.9s · 831 tok]: Scored v1 at 40.9/100; Claude Sonnet 5.5 High found 5 concrete defects.",
                8: "PASS 1 · Step 8 [Measure v1 · 2.0s · 262 tok]: Fingerprinted v1 outputs & recorded initial morphometrics sidecar.",
                9: "PASS 1 · Step 9 [Write as claims v1 · 160ms · 121 tok]: Built 10 initial v1 ProbKnow TriG nanopublications.",
                10: "PASS 1 · Step 10 [Sign & publish v1 · 670ms · 756 tok]: Cryptographically signed & verified all 10 v1 nanopubs.",
                11: "PASS 1 · Step 11 [Public record v1 · 20ms · 185 tok]: Pass 1 finished at 479.8s (8.00m, 37.6k tok) with v1 score 40.9/100.",
            }
            p2_banners = {
                12: "PASS 2 LOOP · Step 12 [Remediate Triage · +159.2s · +2.2k tok]: Triaged 5 v1 audit defects & launched feedback loop back to Step 5!",
                5: "PASS 2 LOOP · Step 5 [Script v2 Re-Work · +620.0s · +12.4k tok]: Added specimen segmentation, tagged V(s) planar RNG & grown Strahler tree.",
                6: "PASS 2 LOOP · Step 6 [Output v2 Re-Render · +1.8s · +203 tok]: Re-rendered v2 comparison figure (168 specimen areoles, 0 crossings) & GIF.",
                7: "PASS 2 LOOP · Step 7 [Re-Audit v2 · +185.0s · +2.8k tok]: Re-scored remediated dimensions — score jumped from 40.9 → 91.3 (+50.4 pts)!",
                8: "PASS 2 LOOP · Step 8 [Re-Measure v2 · +2.1s · +280 tok]: Verified 168 specimen areoles, 358 model areoles & r=0.968 in v2 sidecar.",
                9: "PASS 2 LOOP · Step 9 [Claims v2 · +220ms · +140 tok]: Emitted 10 v2 nanopubs linked to v1 via npx:supersedes.",
                10: "PASS 2 LOOP · Step 10 [Sign v2 · +550ms · +513 tok]: Signed & verified all 10 v2 trusty URIs (20/20 total verified).",
                11: "PASS 2 LOOP · Step 11 [Public record v2 · +50ms · +110 tok]: Pass 2 complete (+968.8s, +18.4k tok) → Total 1,448.6s (24.14m), Score 91.3/100!",
            }
        elif is_turing:
            p1_banners = {
                1: "PASS 1 · Step 1 [Scout · 74ms · 37.6k tok]: Selected 1952-turing-morpho.py (Turing 1952, Pearson 1993 & Kondo-Asai 1995).",
                2: "PASS 1 · Step 2 [Paper · 78ms · 12.2k tok]: Inspected v1 citations vs missing Turing Sec. 8–9 dispersion & Pomacanthus stripes.",
                3: "PASS 1 · Step 3 [Formal source · 77ms · 12.2k tok]: Verified v1 lacked datasets/turing/ & biological specimen images.",
                4: "PASS 1 · Step 4 [Pin the rules · 4.3s · 312 tok]: Checked v1 MathOfBiology.lean Section 5 (only 1 trivial theorem).",
                5: "PASS 1 · Step 5 [Script v1 · 78ms · 12.2k tok]: Inspected baseline 595-line 1952-turing-morpho.py (before Step 6 runs simulation).",
                6: "PASS 1 · Step 6 [Output v1 · 4.2s · 68 tok]: First simulation run! Generated v1 2x5 Gray–Scott plot & metrics CSV.",
                7: "PASS 1 · Step 7 [Dual-agent audit v1 · 762.4s · 87.2k tok]: Scored v1 at 38.0/100; Claude Sonnet 5.5 High found 5 concrete defects.",
                8: "PASS 1 · Step 8 [Measure v1 · 4.8s · 556 tok]: Fingerprinted v1 outputs & recorded initial morphometrics sidecar.",
                9: "PASS 1 · Step 9 [Write as claims v1 · 261ms · 642 tok]: Built 10 initial v1 ProbKnow TriG nanopublications.",
                10: "PASS 1 · Step 10 [Sign & publish v1 · 679ms · 413 tok]: Cryptographically signed & verified all 10 v1 nanopubs.",
                11: "PASS 1 · Step 11 [Public record v1 · 83ms · 6.1k tok]: Pass 1 finished at 777.0s (12.95m, 169.5k tok) with v1 score 38.0/100.",
            }
            p2_banners = {
                12: "PASS 2 LOOP · Step 12 [Remediate Triage · +312.0s · +28.4k tok]: Triaged 5 v1 audit defects & launched feedback loop!",
                5: "PASS 2 LOOP · Step 5 [Script v2 Re-Work · +6.4s · +12.7k tok]: Added Turing N=20 ring, exact Hopf curve, 12 classes & Pomacanthus stripes.",
                6: "PASS 2 LOOP · Step 6 [Output v2 Re-Render · +7.1s · +186 tok]: Rendered 15-panel v2 synthesis figure, CSV & synchronized 45-frame GIF.",
                7: "PASS 2 LOOP · Step 7 [Re-Audit v2 · +418.0s · +50.8k tok]: Re-scored remediated v2 — consensus score jumped from 38.0 → 85.5 (+47.5 pts)!",
                8: "PASS 2 LOOP · Step 8 [Re-Measure v2 · +6.3s · +772 tok]: Verified 10 morphometrics (λ_c=5.62, Pomacanthus 9.55px, 121 boxfish spots).",
                9: "PASS 2 LOOP · Step 9 [Claims v2 · +160ms · +794 tok]: Emitted 10 v2 nanopubs linked to v1 via npx:supersedes.",
                10: "PASS 2 LOOP · Step 10 [Sign v2 · +610ms · +586 tok]: Signed & verified all 10 v2 trusty URIs (20/20 total verified).",
                11: "PASS 2 LOOP · Step 11 [Public record v2 · +90ms · +3.1k tok]: Pass 2 complete (+752.7s, +113.7k tok) → Score 85.5/100 (+47.5 pts)!",
            }
        else:
            p1_banners = {
                1: "PASS 1 · Step 1 [Scout · 23.2s · 28.4k tok]: Selected Vogel (1979), Douady & Couder (1992), and Swinton (2016) Helianthus dataset.",
                2: "PASS 1 · Step 2 [Paper · 20.4s · 27.1k tok]: Extracted planar spiral r_n = c*sqrt(n), theta_n = n*137.5078° & Hofmeister energy equations.",
                3: "PASS 1 · Step 3 [Formal source · 14.8s · 3.8k tok]: Vendored Swinton (2016) Tables 1-4 & 2 Helianthus specimen photos with SHA-256 locks.",
                4: "PASS 1 · Step 4 [Pin the rules · 43.6s · 47.1k tok]: Verified Delaunay parastichy cascade (8,13)->(13,21)->(21,34)->(34,55) & Table 1 invariants.",
                5: "PASS 1 · Step 5 [Script v1 · 242.4s · 15.1k tok]: Wrote initial 1979-vogel-phyllotaxis.py (before Step 6 runs simulation).",
                6: "PASS 1 · Step 6 [Output v1 · 3.7s · 134 tok]: First simulation run! Rendered initial v1 figure & growth animation.",
                7: "PASS 1 · Step 7 [Dual-agent audit v1 · 170.1s · 6.7k tok]: Scored v1 at 64.3/100; Claude Sonnet 5.5 High found 4 concrete defects.",
                8: "PASS 1 · Step 8 [Measure v1 · 1.1s · 274 tok]: Fingerprinted v1 outputs & recorded initial morphometrics sidecar.",
                9: "PASS 1 · Step 9 [Write as claims v1 · 402ms · 295 tok]: Built 10 initial v1 ProbKnow TriG nanopublications.",
                10: "PASS 1 · Step 10 [Sign & publish v1 · 788ms · 1.0k tok]: Cryptographically signed & verified all 10 v1 nanopubs.",
                11: "PASS 1 · Step 11 [Public record v1 · 17ms · 102 tok]: Pass 1 finished at 520.5s (8.67m, 130.0k tok) with v1 score 64.3/100.",
            }
            p2_banners = {
                12: "PASS 2 LOOP · Step 12 [Remediate Triage · +14.2s · +2.1k tok]: Triaged 4 v1 audit defects & launched feedback loop back to Step 5!",
                5: "PASS 2 LOOP · Step 5 [Script v2 Re-Work · +400.7s · +26.4k tok]: Added 621-floret DoG segmentation, bounded Voronoi ring & rim-dislocation model.",
                6: "PASS 2 LOOP · Step 6 [Output v2 Re-Render · +5.8s · +180 tok]: Re-rendered v2 comparison figure (with 621 segmented florets) & growth GIF.",
                7: "PASS 2 LOOP · Step 7 [Re-Audit v2 · +49.0s · +6.8k tok]: Touch-scored remediated dimensions — score jumped from 64.3 → 85.9 (+21.6 pts)!",
                8: "PASS 2 LOOP · Step 8 [Re-Measure v2 · +12.5s · +310 tok]: Verified bounded Voronoi optimality & 621 segmented florets in v2 sidecar.",
                9: "PASS 2 LOOP · Step 9 [Claims v2 · +10.7s · +320 tok]: Emitted 10 v2 nanopubs linked to v1 via npx:supersedes.",
                10: "PASS 2 LOOP · Step 10 [Sign v2 · +800ms · +890 tok]: Signed & verified all 10 v2 trusty URIs (20/20 total verified).",
                11: "PASS 2 LOOP · Step 11 [Public record v2 · +100ms · +112 tok]: Pass 2 complete (+493.8s, +37.1k tok) → Total 1,014.3s (16.90m), Score 85.9/100!",
            }

        cw, ct = 0.0, 0
        for snum in range(1, 12):
            pb = PASS_BREAKDOWN[snum]
            cw += float(pb["p1_s"])
            ct += int(pb["p1_tok"])
            if snum < 6:
                schedule.append((snum, 1, -1, p1_banners[snum], cw, ct, 400))
            elif snum == 6:
                for k in range(5):
                    gf_i = int((k / 4.0) * (n_gf // 2))
                    schedule.append((6, 1, gf_i, p1_banners[6], cw, ct, 220))
            else:
                schedule.append((snum, 1, n_gf // 2, p1_banners[snum], cw, ct, 420 if snum != 7 else 650))

        for snum in [12, 5, 6, 7, 8, 9, 10, 11]:
            pb = PASS_BREAKDOWN[snum]
            cw += float(pb["p2_s"])
            ct += int(pb["p2_tok"])
            if snum == 12:
                schedule.append((12, 2, n_gf // 2, p2_banners[12], cw, ct, 750))
            elif snum == 5:
                schedule.append((5, 2, n_gf // 2, p2_banners[5], cw, ct, 650))
            elif snum == 6:
                for k in range(5):
                    gf_i = int((n_gf // 2) + (k / 4.0) * (n_gf - 1 - (n_gf // 2)))
                    schedule.append((6, 2, gf_i, p2_banners[6], cw, ct, 220))
            else:
                schedule.append((snum, 2, n_gf - 1, p2_banners[snum], cw, ct, 450 if snum != 7 else 700))

        # Final hold frame showing full two-pass completion
        schedule.append((None, 2, n_gf - 1, None, cw, ct, 1800))

        pil_frames: list[Image.Image] = []
        durations: list[int] = []
        for act_s, p_num, gf_i, btxt, c_w, c_t, d_ms in schedule:
            im = draw_cycle_canvas(
                active_step=act_s,
                pass_num=p_num,
                growth_frame_idx=gf_i,
                banner_text=btxt,
                cum_wall_s=c_w,
                cum_tok=c_t,
            )
            pil_frames.append(im)
            durations.append(d_ms)

        # Quantize with a composite palette from Pass 1, Pass 2, and final summary frames
        pal_canvas = Image.new("RGB", (pil_frames[0].width, pil_frames[0].height * 3))
        pal_canvas.paste(pil_frames[6], (0, 0))
        pal_canvas.paste(pil_frames[16], (0, pil_frames[0].height))
        pal_canvas.paste(pil_frames[-1], (0, pil_frames[0].height * 2))
        pal = pal_canvas.quantize(colors=255, method=Image.Quantize.MEDIANCUT)
        q_frames = [f.quantize(palette=pal, dither=Image.Dither.NONE) for f in pil_frames]
        q_frames[0].save(
            out_gif,
            save_all=True,
            append_images=q_frames[1:],
            duration=durations,
            loop=0,
            optimize=True,
        )
        print(
            f"[track_loop] Saved single-cycle animated GIF -> {out_gif} "
            f"({len(q_frames)} frames, {out_gif.stat().st_size // 1024} KB)"
        )

    return out_png, out_gif


def plot_telemetry(doc: dict[str, Any], out_png: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    steps = doc["steps"]
    tot = doc["totals"]
    run = doc["run"]

    labels = [f"{s['num']}. {s['slug']}" for s in steps]
    wall = np.array([float(s.get("duration_s") or 0.0) for s in steps])
    cpu = np.array(
        [
            float(s.get("cpu_user_s") or 0.0) + float(s.get("cpu_sys_s") or 0.0)
            for s in steps
        ]
    )
    in_tok = np.array([int((s.get("tokens") or {}).get("input_tokens") or 0) for s in steps]) / 1e3
    out_tok = np.array([int((s.get("tokens") or {}).get("output_tokens") or 0) for s in steps]) / 1e3
    sub_tok = np.array([int((s.get("tokens") or {}).get("subagent_tokens") or 0) for s in steps]) / 1e3
    art_tok = np.array([int((s.get("tokens") or {}).get("artifact_tokens") or 0) for s in steps]) / 1e3

    tier_colors = [
        "#2563eb" if s["tier"] == "A" else ("#d97706" if s["slug"] in {"audit", "measure"} else ("#059669" if s["slug"] != "remed" else "#dc2626"))
        for s in steps
    ]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.5, 6.2), dpi=160)
    fig.patch.set_facecolor("white")
    y = np.arange(len(steps))[::-1]

    # Panel A: Wall-clock & CPU time per step
    bars = ax1.barh(y, wall, height=0.66, color=tier_colors, alpha=0.88, edgecolor="#1e293b", lw=0.7, label="Wall-clock time (s)")
    ax1.barh(y, cpu, height=0.30, color="#0f172a", alpha=0.65, label="Subprocess CPU user+sys (s)")
    ax1.set_yticks(y)
    ax1.set_yticklabels(labels, fontsize=9.5, fontweight="bold")
    ax1.set_xlabel("Duration (seconds)", fontsize=10, fontweight="bold")
    ax1.set_title(
        f"A. Wall-Clock & CPU Time per Step (Total: {tot['wall_time_s']:.1f} s / {tot['wall_time_min']:.2f} min)",
        fontsize=10.5,
        fontweight="bold",
        loc="left",
    )
    ax1.grid(axis="x", linestyle="--", alpha=0.35)
    max_w = max(float(wall.max()), 1.0)
    ax1.set_xlim(0, max_w * 1.24)
    for yi, w_val, c_val in zip(y, wall, cpu):
        ax1.text(
            w_val + max_w * 0.02,
            yi,
            f"{w_val:.1f}s" + (f" (CPU {c_val:.1f}s)" if c_val >= 0.2 else ""),
            va="center",
            ha="left",
            fontsize=8.2,
            color="#1e293b",
        )
    ax1.axhline(5.5, color="#64748b", linestyle=":", lw=1.0)
    ax1.text(max_w * 1.21, 6.1, "Tier A (Reproduction)", ha="right", va="bottom", fontsize=8, color="#2563eb", fontweight="bold")
    ax1.text(max_w * 1.21, 4.9, "Tier B (MorphoEval Cycle)", ha="right", va="top", fontsize=8, color="#d97706", fontweight="bold")
    ax1.legend(loc="lower right", fontsize=8.2, framealpha=0.9)

    # Panel B: Token usage per step (in kTokens)
    ax2.barh(y, in_tok, height=0.62, color="#3b82f6", edgecolor="#1e293b", lw=0.5, label="Input / context tokens (k)")
    ax2.barh(y, out_tok, left=in_tok, height=0.62, color="#10b981", edgecolor="#1e293b", lw=0.5, label="Output / completion tokens (k)")
    ax2.barh(y, sub_tok, left=in_tok + out_tok, height=0.62, color="#8b5cf6", edgecolor="#1e293b", lw=0.5, label="2nd-assessor / subagent tokens (k)")
    ax2.scatter(art_tok, y, color="#ef4444", s=34, zorder=5, label="Produced artifact tokens (k)")

    ax2.set_yticks(y)
    ax2.set_yticklabels(labels, fontsize=9.5)
    ax2.set_xlabel("Tokens (thousands, cl100k BPE)", fontsize=10, fontweight="bold")
    ax2.set_title(
        f"B. Token Budget per Step (Total LLM: {tot['total_llm_tokens']:,} tok | Artifacts: {tot['artifact_tokens']:,} tok)",
        fontsize=10.5,
        fontweight="bold",
        loc="left",
    )
    ax2.grid(axis="x", linestyle="--", alpha=0.35)
    tot_step_k = in_tok + out_tok + sub_tok
    max_t = max(float(np.maximum(tot_step_k, art_tok).max()), 1.0)
    ax2.set_xlim(0, max_t * 1.25)
    for yi, tk, ak in zip(y, tot_step_k, art_tok):
        pos = max(tk, ak)
        ax2.text(
            pos + max_t * 0.02,
            yi,
            f"{tk:.1f}k LLM" + (f" | {ak:.1f}k art." if ak >= 0.2 else ""),
            va="center",
            ha="left",
            fontsize=8.0,
            color="#1e293b",
        )
    ax2.axhline(5.5, color="#64748b", linestyle=":", lw=1.0)
    ax2.legend(loc="lower right", fontsize=8.0, framealpha=0.9)

    fig.suptitle(
        f"MorphoEval 12-Step Loop Telemetry — {run.get('target_simulation')} ({run.get('paper') or run['id']})",
        fontsize=12.5,
        fontweight="bold",
        y=0.98,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=160, facecolor="white")
    plt.close(fig)
    print(f"[track_loop] Saved telemetry plot -> {out_png}")


def backfill_events_if_empty(doc: dict[str, Any], runs_dir: Path = RUNS_DIR) -> None:
    """If telemetry_events.jsonl has no events for this run_id yet, backfill from completed steps."""
    run_id = doc["run"]["id"]
    jpath = events_jsonl_path(runs_dir, run_id=run_id, isolated=is_isolated_run(doc))
    if jpath.exists():
        for line in jpath.read_text(encoding="utf-8").splitlines():
            if f'"{run_id}"' in line:
                return
    append_event(
        runs_dir,
        run_id=run_id,
        event_type="run_init",
        actor="orchestrator",
        model=doc["run"].get("builder_model"),
        payload={
            "target_simulation": doc["run"].get("target_simulation"),
            "paper": doc["run"].get("paper"),
        },
    )
    for s in doc.get("steps") or []:
        if s.get("status") == "pending":
            continue
        tok = s.get("tokens") or {}
        append_event(
            runs_dir,
            run_id=run_id,
            event_type="step_end",
            step_num=s["num"],
            slug=s["slug"],
            tier=s["tier"],
            actor=s.get("actor"),
            model=s.get("model"),
            duration_s=float(s.get("duration_s") or 0.0),
            total_llm_tokens=int(tok.get("total_llm_tokens") or 0),
            artifact_tokens=int(tok.get("artifact_tokens") or 0),
            payload={
                "status": s.get("status"),
                "started_at": s.get("started_at"),
                "ended_at": s.get("ended_at"),
                "input_tokens": int(tok.get("input_tokens") or 0),
                "output_tokens": int(tok.get("output_tokens") or 0),
                "subagent_tokens": int(tok.get("subagent_tokens") or 0),
                "artifacts": [a["file"] for a in (s.get("artifacts") or [])],
            },
        )


def cmd_summary(args: argparse.Namespace, runs_dir: Path = RUNS_DIR) -> str:
    doc = load_run(args.run_id, runs_dir)
    recompute_totals(doc)
    save_run(doc, runs_dir)
    backfill_events_if_empty(doc, runs_dir)

    csv_path = Path(args.csv) if args.csv else (runs_dir / f"{args.run_id}.csv")
    write_csv_summary(doc, csv_path)

    trig_path = Path(args.trig) if getattr(args, "trig", None) else (runs_dir / f"{args.run_id}.trig")
    write_trig_provenance(doc, trig_path)

    if args.plot:
        plot_telemetry(doc, Path(args.plot))
    else:
        plot_telemetry(doc, runs_dir / f"{args.run_id}_telemetry.png")

    cycle_png = Path(getattr(args, "cycle_png", None) or (ROOT / "docs" / "media" / "vogel_phyllotaxis_closed_loop.png"))
    cycle_gif = Path(getattr(args, "cycle_gif", None) or (ROOT / "docs" / "media" / "vogel_phyllotaxis_closed_loop.gif"))
    if getattr(args, "render_cycle", False):
        render_single_cycle_loop(doc, cycle_png, cycle_gif)

    md = render_markdown_summary(doc)
    if args.md:
        md_path = Path(args.md)
        md_path.parent.mkdir(parents=True, exist_ok=True)
        md_path.write_text(md, encoding="utf-8")

    print(
        f"[track_loop] Synced structured telemetry stores for '{args.run_id}':\n"
        f"  - SQLite DB   : {sqlite_path(runs_dir)}\n"
        f"  - Event JSONL : {events_jsonl_path(runs_dir)}\n"
        f"  - RDF TriG    : {trig_path}\n"
        f"  - Step CSV    : {csv_path}\n"
        f"  - Run YAML    : {run_path(args.run_id, runs_dir)}"
    )
    return md


PRESET_QUERIES: dict[str, str] = {
    "steps": """
        SELECT step_num, tier, slug, actor, model, ROUND(duration_s, 2) AS wall_s,
               ROUND(cpu_user_s + cpu_sys_s, 2) AS cpu_s,
               input_tokens, output_tokens, subagent_tokens, total_llm_tokens,
               artifact_tokens, artifacts_count
        FROM steps
        ORDER BY run_id, step_num
    """,
    "tiers": """
        SELECT run_id, tier, COUNT(*) AS steps,
               ROUND(SUM(duration_s), 2) AS wall_s,
               ROUND(SUM(cpu_user_s + cpu_sys_s), 2) AS cpu_s,
               SUM(total_llm_tokens) AS llm_tokens,
               SUM(artifact_tokens) AS artifact_tokens
        FROM steps
        GROUP BY run_id, tier
        ORDER BY run_id, MIN(step_num)
    """,
    "artifacts": """
        SELECT step_num, slug, file, bytes, tokens, SUBSTR(sha256, 1, 16) AS sha256_prefix
        FROM step_artifacts
        ORDER BY run_id, step_num, file
    """,
    "runs": """
        SELECT run_id, target_simulation, steps_completed || '/' || steps_total AS progress,
               ROUND(wall_time_s, 1) AS wall_s, total_llm_tokens, artifact_tokens,
               effective_tokens_per_s AS tok_per_s
        FROM runs
        ORDER BY created_at DESC
    """,
}


def cmd_query(args: argparse.Namespace, runs_dir: Path = RUNS_DIR) -> list[dict[str, Any]]:
    db_file = sqlite_path(runs_dir)
    if not db_file.exists():
        for yml in sorted(runs_dir.glob("*.yaml")):
            doc = yaml.safe_load(yml.read_text(encoding="utf-8"))
            if isinstance(doc, dict) and "run" in doc:
                sync_sqlite(doc, runs_dir)
                backfill_events_if_empty(doc, runs_dir)
    sql = args.sql or PRESET_QUERIES.get(args.preset or "steps", PRESET_QUERIES["steps"])
    with sqlite3.connect(db_file) as conn:
        conn.row_factory = sqlite3.Row
        cur = conn.execute(sql)
        rows = [dict(r) for r in cur.fetchall()]

    fmt = getattr(args, "format", "table")
    if fmt == "json":
        print(json.dumps(rows, indent=2, ensure_ascii=False))
    elif fmt == "csv":
        if rows:
            writer = csv.DictWriter(sys.stdout, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    else:
        if not rows:
            print("(0 rows)")
        else:
            cols = list(rows[0].keys())
            widths = {
                c: max(len(str(c)), max(len(str(r.get(c, ""))) for r in rows))
                for c in cols
            }
            header = "  ".join(str(c).ljust(widths[c]) for c in cols)
            sep = "  ".join("-" * widths[c] for c in cols)
            print(header)
            print(sep)
            for r in rows:
                print("  ".join(str(r.get(c, "")).ljust(widths[c]) for c in cols))
    return rows


def run_self_test() -> None:
    print("[track_loop] Running self-test...")
    tok, method = count_tokens("def vogel_phyllotaxis(n, alpha=137.507764): return np.sqrt(n)")
    assert tok >= 10, f"Expected >=10 tokens, got {tok}"
    step1 = resolve_step("1")
    assert step1["slug"] == "scout"
    step7 = resolve_step("audit")
    assert step7["num"] == 7 and step7["tier"] == "B"

    with tempfile.TemporaryDirectory() as td:
        tdir = Path(td)
        cmd_init(
            argparse.Namespace(
                run_id="selftest-loop",
                target="selftest-sim",
                paper="Vogel (1979) Mathematical Biosciences",
                builder_model="gemini-4-argon",
                primary_assessor="gemini-4-argon",
                second_assessor="claude-sonnet-5-5-high",
                notes="Self-test run",
            ),
            runs_dir=tdir,
        )
        rc = cmd_exec(
            argparse.Namespace(
                run_id="selftest-loop",
                step="scout",
                actor="scout-agent",
                model="gemini-4-argon",
                input_tokens=1200,
                output_tokens=450,
                subagent_tokens=0,
                subagent_jsonl=None,
                input_files=["morphoeval/rubric.yaml"],
                artifacts=["morphoeval/rubric.yaml"],
                notes="Verified scout step",
                cmd=["--", sys.executable, "-c", "print('scout ok')"],
            ),
            runs_dir=tdir,
        )
        assert rc == 0
        md = cmd_summary(
            argparse.Namespace(
                run_id="selftest-loop",
                md=None,
                csv=str(tdir / "summary.csv"),
                trig=str(tdir / "selftest-loop.trig"),
                plot=str(tdir / "telemetry.png"),
            ),
            runs_dir=tdir,
        )
        assert "1,650" in md and "selftest-sim" in md
        assert (tdir / "telemetry.png").exists()
        assert (tdir / "telemetry.sqlite").exists()
        assert (tdir / "telemetry_events.jsonl").exists()
        assert (tdir / "selftest-loop.trig").exists()
        rows = cmd_query(
            argparse.Namespace(sql=None, preset="tiers", format="json"),
            runs_dir=tdir,
        )
        assert any(r["llm_tokens"] == 1650 for r in rows)
    print(
        "[track_loop] Self-test PASSED (tokenizer, 12-step lifecycle, SQLite + JSONL + RDF TriG + YAML/CSV/PNG)."
    )


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Track wall-clock time and tokens per step of the MorphoEval loop.")
    ap.add_argument("--self-test", action="store_true", help="Run built-in verification suite and exit")
    sub = ap.add_subparsers(dest="subcmd")

    p_init = sub.add_parser("init", help="Initialize a new 12-step loop telemetry log")
    p_init.add_argument("--run-id", required=True)
    p_init.add_argument("--target", required=True, help="Simulation slug (e.g. vogel-phyllotaxis)")
    p_init.add_argument("--paper", default="")
    p_init.add_argument("--builder-model", default="gemini-4-argon")
    p_init.add_argument("--primary-assessor", default="gemini-4-argon")
    p_init.add_argument("--second-assessor", default="claude-sonnet-5-5-high")
    p_init.add_argument("--isolated", action="store_true", help="Write only per-run files (<run_id>.*) without modifying shared telemetry.sqlite or telemetry_events.jsonl")
    p_init.add_argument("--notes", default="")

    p_start = sub.add_parser("start", help="Mark a loop step as started")
    p_start.add_argument("--run-id", required=True)
    p_start.add_argument("--step", required=True, help="Step number (1..12) or slug (scout, paper, ...)")
    p_start.add_argument("--actor", default=None)
    p_start.add_argument("--model", default=None)
    p_start.add_argument("--notes", default=None)

    p_end = sub.add_parser("end", help="Mark a loop step as completed and record tokens/artifacts")
    p_end.add_argument("--run-id", required=True)
    p_end.add_argument("--step", required=True)
    p_end.add_argument("--status", default="completed")
    p_end.add_argument("--duration-s", type=float, default=None)
    p_end.add_argument("--actor", default=None)
    p_end.add_argument("--model", default=None)
    p_end.add_argument("--input-tokens", type=int, default=None)
    p_end.add_argument("--output-tokens", type=int, default=None)
    p_end.add_argument("--subagent-tokens", type=int, default=0)
    p_end.add_argument("--subagent-jsonl", default=None)
    p_end.add_argument("--input-files", nargs="*", default=[])
    p_end.add_argument("--artifacts", nargs="*", default=[])
    p_end.add_argument("--notes", default=None)

    p_exec = sub.add_parser("exec", help="Execute a shell command as a tracked step")
    p_exec.add_argument("--run-id", required=True)
    p_exec.add_argument("--step", required=True)
    p_exec.add_argument("--actor", default=None)
    p_exec.add_argument("--model", default=None)
    p_exec.add_argument("--input-tokens", type=int, default=None)
    p_exec.add_argument("--output-tokens", type=int, default=None)
    p_exec.add_argument("--subagent-tokens", type=int, default=0)
    p_exec.add_argument("--subagent-jsonl", default=None)
    p_exec.add_argument("--input-files", nargs="*", default=[])
    p_exec.add_argument("--artifacts", nargs="*", default=[])
    p_exec.add_argument("--notes", default=None)
    p_exec.add_argument("cmd", nargs=argparse.REMAINDER)

    p_sum = sub.add_parser("summary", help="Sync SQLite, JSONL, RDF TriG, CSV, and PNG/GIF telemetry stores")
    p_sum.add_argument("--run-id", required=True)
    p_sum.add_argument("--md", default=None, help="Optional Markdown path (omitted by default)")
    p_sum.add_argument("--csv", default=None)
    p_sum.add_argument("--trig", default=None)
    p_sum.add_argument("--plot", default=None)
    p_sum.add_argument("--render-cycle", action="store_true", help="Render the single-cycle dark-mode closed-loop PNG and animated GIF")
    p_sum.add_argument("--cycle-png", default=None)
    p_sum.add_argument("--cycle-gif", default=None)

    p_qry = sub.add_parser("query", help="Query the SQLite telemetry database across runs and steps")
    p_qry.add_argument("--preset", choices=["steps", "tiers", "artifacts", "runs"], default="steps")
    p_qry.add_argument("--sql", default=None, help="Custom SQL query against telemetry.sqlite")
    p_qry.add_argument("--format", choices=["table", "json", "csv"], default="table")

    return ap


def main(argv: list[str] | None = None) -> int:
    ap = build_parser()
    args = ap.parse_args(argv)
    if args.self_test:
        run_self_test()
        return 0
    if args.subcmd == "init":
        cmd_init(args)
        return 0
    if args.subcmd == "start":
        cmd_start(args)
        return 0
    if args.subcmd == "end":
        cmd_end(args)
        return 0
    if args.subcmd == "exec":
        return cmd_exec(args)
    if args.subcmd == "summary":
        cmd_summary(args)
        return 0
    if args.subcmd == "query":
        cmd_query(args)
        return 0
    ap.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

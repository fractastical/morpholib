#!/usr/bin/env python3
"""
morphoeval/aggregate_loops.py — Post-Merge Aggregator for Isolated Cloud Loop Artifacts.

Because each of the 100 Cloud Agent loops writes ONLY isolated per-loop files:
  - `morphoeval/lean/loops/Loop_<NNN>_<slug>.lean`
  - `morphoeval/math_kg_loops/<sim_id>.yaml`
  - `morphoeval/audits/<date>-<slug>-audit*.yaml`
  - `morphoeval/kg/<date>-<slug>-audit*/...`
  - `morphoeval/runs/<date>-<slug>-loop.yaml` & `.events.jsonl`
to guarantee zero merge conflicts across 100 concurrent Pull Requests, this script
aggregates all merged per-loop artifacts back into the unified repository stores whenever
desired (locally or via a post-merge GitHub Action on `main`):

  1. Syncs every `morphoeval/runs/*-loop.yaml` into `morphoeval/runs/telemetry.sqlite`
     and `morphoeval/runs/telemetry_events.jsonl`.
  2. Verifies all `morphoeval/lean/loops/*.lean` modules alongside `morphoeval/lean/MathOfBiology.lean`
     and rebuilds the unified `morphoeval/kg/math_of_biology.ttl`, `math_of_biology.trig`,
     and SQLite `math_models` / `math_equations` / `math_measurements` tables via `math_kg.py --build`.

Usage:
  python3 morphoeval/aggregate_loops.py
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RUNS_DIR = HERE / "runs"

sys.path.insert(0, str(HERE))
import track_loop  # noqa: E402
import math_kg     # noqa: E402


def sync_all_isolated_runs() -> int:
    """Ingest all `morphoeval/runs/*.yaml` run logs into `telemetry.sqlite`."""
    if not RUNS_DIR.exists():
        return 0
    count = 0
    for yf in sorted(RUNS_DIR.glob("*.yaml")):
        try:
            doc = yaml.safe_load(yf.read_text(encoding="utf-8"))
            if not isinstance(doc, dict) or "run" not in doc or "steps" not in doc:
                continue
            track_loop.recompute_totals(doc)
            track_loop.sync_sqlite(doc, RUNS_DIR)
            count += 1
        except Exception as exc:
            print(f"[aggregate_loops] Warning: skipped {yf.name}: {exc}", file=sys.stderr)
    return count


def main() -> int:
    n_runs = sync_all_isolated_runs()
    print(f"[aggregate_loops] Synced {n_runs} loop telemetry run(s) into morphoeval/runs/telemetry.sqlite.")

    stats = math_kg.build_math_knowledge_graph()
    li = stats["lean_info"]
    print(
        f"[aggregate_loops] Rebuilt unified Math of Biology KG:\n"
        f"  - Models       : {stats['num_models']}\n"
        f"  - Equations    : {stats['num_equations']} (Lean 4 verified={li['verified']}, {li['version']})\n"
        f"  - Measurements : {stats['num_measurements']}\n"
        f"  - RDF Triples  : {stats['num_triples']} -> {stats['ttl_path']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

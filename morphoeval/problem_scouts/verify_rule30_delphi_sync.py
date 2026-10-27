#!/usr/bin/env python3
"""
Cross-artifact viability sync for Rule 30 grand-prize Delphi bridge posteriors.

Compares OP-01..OP-03 pooled bridge posteriors (percent, 0–100) across:
  - docs/open-problems.json (MorphoSwarm feed)
  - docs/problem-graph.html embedded DATA.DELPHI (problem graph)
  - morphoeval/audits/2026-10-20-rule30-grand-prizes-bayesian-delphi.yaml
    panel_consensus bridge_pooled.posterior_logodds_pooled (source of truth)

Exit 0 when all checks pass; exit 1 and print offending ids/values otherwise.

Tolerances (scout fix-003 / rule30-delphi-artifact-sync):
  - |open-problems pooled − graph DELPHI pooled posterior| ≤ 0.6 (percentage points)
  - YAML bridge pooled vs open-problems and vs graph: ≤ 0.01% relative error
    (with a 0.01 percentage-point floor for display rounding)
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent

OPEN_PROBLEMS_PATH = ROOT / "docs" / "open-problems.json"
DEFAULT_GRAPH_HTML = ROOT / "docs" / "problem-graph.html"
AUDIT_YAML_PATH = (
    ROOT / "morphoeval" / "audits" / "2026-10-20-rule30-grand-prizes-bayesian-delphi.yaml"
)

GRAPH_ABS_TOLERANCE_PP = 0.6
YAML_REL_TOLERANCE = 0.0001  # 0.01%
YAML_ABS_FLOOR_PP = 0.01  # display rounding slack (2–3 decimal places in graph)

PRIZES: list[tuple[str, str, str]] = [
    ("OP-01", "p1", "problem-1-center-aperiodicity"),
    ("OP-02", "p2", "problem-2-equidistribution"),
    ("OP-03", "p3", "problem-3-computational-irreducibility"),
]


def load_open_problems_pooled(path: Path) -> dict[str, float]:
    feed = json.loads(path.read_text(encoding="utf-8"))
    out: dict[str, float] = {}
    for op_id, _pk, _ykey in PRIZES:
        row = next(p for p in feed["problems"] if p["id"] == op_id)
        vi = row.get("viability") or {}
        if vi.get("kind") != "delphi_bridge_posterior_percent":
            raise ValueError(f"{op_id}: expected delphi_bridge_posterior_percent viability")
        pooled = vi.get("pooled")
        if pooled is None:
            raise ValueError(f"{op_id}: missing viability.pooled")
        out[op_id] = float(pooled)
    return out


def extract_delphi_from_graph_html(path: Path) -> dict[str, float]:
    html = path.read_text(encoding="utf-8")
    m = re.search(r"const DATA = (\{.*?\});\s*\nconst \{AGENTS", html, re.DOTALL)
    if not m:
        raise ValueError(f"{path}: could not find embedded DATA JSON")
    data: dict[str, Any] = json.loads(m.group(1))
    delphi = data.get("DELPHI")
    if not isinstance(delphi, dict):
        raise ValueError(f"{path}: DATA.DELPHI missing or not an object")
    out: dict[str, float] = {}
    for op_id, pk, _ykey in PRIZES:
        if pk not in delphi:
            raise ValueError(f"{path}: DELPHI missing prize key {pk!r} ({op_id})")
        pooled = delphi[pk].get("pooled", {}).get("posterior")
        if pooled is None:
            raise ValueError(f"{path}: DELPHI[{pk!r}].pooled.posterior missing")
        out[op_id] = float(pooled)
    return out


def load_yaml_bridge_pooled_percent(path: Path) -> dict[str, float]:
    doc = yaml.safe_load(path.read_text(encoding="utf-8"))
    audit = doc.get("audit") if isinstance(doc.get("audit"), dict) else doc
    panel = audit.get("panel_consensus") or {}
    problems = panel.get("problems") or {}
    out: dict[str, float] = {}
    for op_id, _pk, ykey in PRIZES:
        prob = problems.get(ykey)
        if not prob:
            raise ValueError(f"{path}: panel_consensus.problems missing {ykey!r}")
        bridge = prob.get("bridge_pooled") or {}
        p = bridge.get("posterior_logodds_pooled")
        if p is None:
            raise ValueError(f"{path}: {ykey} bridge_pooled.posterior_logodds_pooled missing")
        out[op_id] = float(p) * 100.0
    return out


def yaml_within_tolerance(yaml_pct: float, other_pct: float) -> bool:
    diff = abs(yaml_pct - other_pct)
    if diff <= YAML_ABS_FLOOR_PP:
        return True
    if yaml_pct == 0.0:
        return diff <= YAML_ABS_FLOOR_PP
    return diff / abs(yaml_pct) <= YAML_REL_TOLERANCE


def verify(
    open_probs: dict[str, float],
    graph: dict[str, float],
    yaml_pct: dict[str, float],
) -> list[str]:
    errs: list[str] = []
    for op_id, _pk, ykey in PRIZES:
        op_v = open_probs[op_id]
        g_v = graph[op_id]
        y_v = yaml_pct[op_id]

        if abs(op_v - g_v) > GRAPH_ABS_TOLERANCE_PP:
            errs.append(
                f"{op_id}: open-problems pooled {op_v}% vs graph DELPHI {g_v}% "
                f"(|Δ|={abs(op_v - g_v):.4f} > {GRAPH_ABS_TOLERANCE_PP} pp)"
            )

        if not yaml_within_tolerance(y_v, op_v):
            rel = abs(y_v - op_v) / abs(y_v) if y_v else float("inf")
            errs.append(
                f"{op_id}: audit YAML bridge pooled {y_v:.6f}% vs open-problems {op_v}% "
                f"(|Δ|={abs(y_v - op_v):.6f}, rel={rel:.6%}; need ≤{YAML_REL_TOLERANCE:.4%} rel "
                f"or ≤{YAML_ABS_FLOOR_PP} pp) [{ykey}]"
            )

        if not yaml_within_tolerance(y_v, g_v):
            rel = abs(y_v - g_v) / abs(y_v) if y_v else float("inf")
            errs.append(
                f"{op_id}: audit YAML bridge pooled {y_v:.6f}% vs graph DELPHI {g_v}% "
                f"(|Δ|={abs(y_v - g_v):.6f}, rel={rel:.6%}; need ≤{YAML_REL_TOLERANCE:.4%} rel "
                f"or ≤{YAML_ABS_FLOOR_PP} pp) [{ykey}]"
            )

    return errs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--open-problems", type=Path, default=OPEN_PROBLEMS_PATH)
    ap.add_argument("--graph-html", type=Path, default=DEFAULT_GRAPH_HTML)
    ap.add_argument("--audit-yaml", type=Path, default=AUDIT_YAML_PATH)
    ap.add_argument("--quiet", action="store_true", help="Only print errors")
    args = ap.parse_args()

    open_probs = load_open_problems_pooled(args.open_problems)
    graph = extract_delphi_from_graph_html(args.graph_html)
    yaml_pct = load_yaml_bridge_pooled_percent(args.audit_yaml)

    if not args.quiet:
        print("Rule 30 Delphi bridge pooled posteriors (% — bridge H^bridge panel log-odds pool):")
        for op_id, pk, ykey in PRIZES:
            print(
                f"  {op_id} ({pk}): yaml={yaml_pct[op_id]:.6f}  "
                f"open-problems={open_probs[op_id]}  graph={graph[op_id]}"
            )

    errs = verify(open_probs, graph, yaml_pct)
    if errs:
        print("DELPHI SYNC FAILED:", file=sys.stderr)
        for e in errs:
            print(f"  - {e}", file=sys.stderr)
        return 1

    if not args.quiet:
        print("OK: OP-01..OP-03 pooled posteriors agree across open-problems.json, "
              "problem-graph.html DELPHI, and audit YAML within tolerance.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

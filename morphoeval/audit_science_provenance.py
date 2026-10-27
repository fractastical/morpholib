#!/usr/bin/env python3
"""
morphoeval/audit_science_provenance.py — Which loops actually contain the science?

A loop only counts if the paper and the actual science are in it. This audit
classifies every loop in the atlas index + the 10,000-animal local summaries
by what is verifiably inside its dataset directory:

  Tier A  science_included — paper extracted AND literature scouted:
          reference JSON carries paper-specific structure (governing_equations /
          canonical_parameters / regime tables — not a generic alpha*x^1.25
          series) AND scout_manifest has real OpenAlex candidates + Wikimedia
          specimen candidates. These are the loops that count.
  Tier B  paper_named_only — a real seed paper/DOI is named (catalog taxa
          papers), but the data series and model are template-generated.
          Promotable: extract the paper, replace the series, re-derive.
  Tier C  scaffold_only — fabricated/slot-encoded DOI, synthetic RNG series,
          template model. Does not count as science in any tally.

Outputs `morphoeval/runs/science_provenance_audit.json` with tier counts and
the Tier-A (counted) loop list. Read-only; never edits datasets.

Usage: venv/bin/python morphoeval/audit_science_provenance.py
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

PAPER_STRUCTURE_KEYS = {
    "governing_equations", "canonical_parameters", "parameter_regime",
    "parameter_table", "empirical_claims", "hydra_regimes_and_experiments",
    "model_equations", "boundary_conditions", "regimes_and_experiments",
}
GENERIC_PARAM_KEYS = {"alpha_scale", "beta_offset", "int_a", "int_b", "int_c"}


def classify_dataset(ds: Path) -> dict[str, Any]:
    out = {"governing_equations": False, "openalex": False, "wikimedia_specimens": False,
           "synthetic_series": False, "slot_encoded_doi": False, "doi": ""}
    refs = sorted(ds.glob("*_reference.json"))
    if refs:
        try:
            ref = json.loads(refs[0].read_text(encoding="utf-8", errors="replace"))
        except Exception:
            ref = {}
        keys = set(ref.keys())
        params = ref.get("parameters", {}) if isinstance(ref.get("parameters"), dict) else {}
        out["governing_equations"] = bool(keys & PAPER_STRUCTURE_KEYS) or (
            bool(params) and not set(params.keys()) <= (GENERIC_PARAM_KEYS | {"characteristic_size_m"}))
        out["synthetic_series"] = "empirical_series" in keys and (
            str(ref.get("schema_version", "")).startswith(("morphoeval-reference-2.", "morphoeval-reference-10000")))
    scout = ds / "scout_manifest.json"
    if scout.exists():
        try:
            sm = json.loads(scout.read_text(encoding="utf-8", errors="replace"))
        except Exception:
            sm = {}
        out["openalex"] = bool(sm.get("candidate_papers_openalex"))
        out["wikimedia_specimens"] = bool(sm.get("candidate_specimens_wikimedia"))
        doi = (sm.get("territory", {}) or {}).get("seed_doi", "") or ""
        out["doi"] = doi
        # Overnight generators fabricate DOIs by appending the slot: 10.1242/dev.1998.0101
        out["slot_encoded_doi"] = bool(re.search(r"\.\d{4}$", doi)) or doi.startswith("10.0000/")
    if out["governing_equations"] and out["openalex"]:
        tier = "A_science_included"
    elif out["doi"] and not out["slot_encoded_doi"]:
        tier = "B_paper_named_only"
    elif out["openalex"] or out["governing_equations"]:
        tier = "B_paper_named_only"
    else:
        tier = "C_scaffold_only"
    out["tier"] = tier
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(HERE / "runs" / "science_provenance_audit.json"))
    args = ap.parse_args()

    entries: list[dict[str, Any]] = []
    idx = HERE / "runs" / "aristotelian_categories_index.json"
    if idx.exists():
        entries += json.loads(idx.read_text(encoding="utf-8"))
    for summ in (HERE / "runs").glob("campaign_10000_local_summary_*.json"):
        entries += json.loads(summ.read_text(encoding="utf-8"))

    seen: set[str] = set()
    counts = {"A_science_included": 0, "B_paper_named_only": 0, "C_scaffold_only": 0}
    counted: list[dict[str, Any]] = []
    promotable: list[dict[str, Any]] = []
    cache: dict[str, dict[str, Any]] = {}
    for e in entries:
        ds_rel = e.get("dataset_dir", "")
        if not ds_rel or ds_rel in seen:
            continue
        seen.add(ds_rel)
        if ds_rel not in cache:
            cache[ds_rel] = classify_dataset(ROOT / ds_rel)
        c = cache[ds_rel]
        counts[c["tier"]] += 1
        rec = {"slot": e.get("slot"), "sim_id": e.get("sim_id") or e.get("slug"),
               "organism": e.get("organism") or e.get("species"), "dataset_dir": ds_rel, "doi": c["doi"]}
        if c["tier"] == "A_science_included":
            counted.append(rec)
        elif c["tier"] == "B_paper_named_only":
            promotable.append(rec)

    report = {
        "rule": "A loop counts only if the paper is extracted (governing equations/parameters in the reference) AND the literature was actually scouted (OpenAlex + Wikimedia candidates). Template loops with synthetic RNG series do not count.",
        "total_loops_classified": sum(counts.values()),
        "tier_counts": counts,
        "counted_science_loops": len(counted),
        "promotable_paper_named_loops": len(promotable),
        "counted": counted,
        "promotable_first_100": promotable[:100],
        "implication": "Stop mass-generating scaffolds. Freeze slot expansion; promote Tier B -> A one paper at a time (extract equations/params, replace synthetic series with paper data, substantive Lean), and only tallies/KG claims should cite Tier A.",
    }
    out = Path(args.out)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("total_loops_classified", "tier_counts", "counted_science_loops", "promotable_paper_named_loops")}, indent=2))
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

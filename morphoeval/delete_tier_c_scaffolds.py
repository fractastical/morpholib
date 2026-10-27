#!/usr/bin/env python3
"""
morphoeval/delete_tier_c_scaffolds.py — Remove Tier-C scaffold-only loops.

Tier C (per `audit_science_provenance.py`): fabricated/slot-encoded DOI,
synthetic RNG series, template model — no paper extracted, no literature
scouted. Tier A (science included) and Tier B (real animal/paper named,
promotable) are kept untouched, including the shared species gallery.

For each Tier-C loop this deletes ONLY its slot-unique artifacts:
  datasets/<dir>/, the model script, Lean loop file, math_kg_loops spec,
  docs/media comparison+GIF, audit+measurements YAMLs, KG audit dir,
  run records, dispatch records. Then it filters Tier-C entries out of
  the aristotelian index and campaign summary JSONs so tallies stop
  counting scaffolds.

Default is --dry-run (writes the manifest, deletes nothing).
Usage:
  venv/bin/python morphoeval/delete_tier_c_scaffolds.py            # dry run
  venv/bin/python morphoeval/delete_tier_c_scaffolds.py --apply    # delete
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from morphoeval.audit_science_provenance import classify_dataset  # noqa: E402

MANIFEST = HERE / "runs" / "tier_c_deletion_manifest.json"

# Slots whose Lean/KG layer was substantively re-formalized (verified_mechanism_proof)
# despite a scaffold dataset — promotion in progress, do not delete.
EXEMPT_SLOTS = {101, 201, 301, 401, 501, 601, 701, 801, 901, 1001}


def load_entries() -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    idx = HERE / "runs" / "aristotelian_categories_index.json"
    if idx.exists():
        entries += json.loads(idx.read_text(encoding="utf-8"))
    for summ in sorted((HERE / "runs").glob("campaign_*summary*.json")):
        try:
            data = json.loads(summ.read_text(encoding="utf-8"))
            if isinstance(data, list):
                entries += data
        except Exception:
            continue
    return entries


def tier_c_entries(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: dict[str, dict[str, Any]] = {}
    cache: dict[str, str] = {}
    for e in entries:
        ds = e.get("dataset_dir", "")
        if not ds:
            continue
        if ds not in cache:
            cache[ds] = classify_dataset(ROOT / ds)["tier"]
        if cache[ds] == "C_scaffold_only" and e.get("slot") not in EXEMPT_SLOTS:
            seen[ds] = e
    return list(seen.values())


def _dir_index(d: Path) -> list[str]:
    try:
        return [p.name for p in d.iterdir()]
    except Exception:
        return []


# Built once (single pass over each big directory), instead of globbing per slug.
AUDIT_NAMES = _dir_index(HERE / "audits")
KG_NAMES = _dir_index(HERE / "kg")
RUN_NAMES = _dir_index(HERE / "runs")


def _matching(names: list[str], parent: Path, slug: str, suffixes: tuple[str, ...]) -> list[Path]:
    return [parent / n for n in names
            if f"-{slug}-" in n and any(n.endswith(sfx) for sfx in suffixes)]


def artifact_paths(e: dict[str, Any]) -> list[Path]:
    paths: list[Path] = []
    ds = e.get("dataset_dir")
    if ds:
        paths.append(ROOT / ds)
    script = e.get("script")
    if script:
        paths.append(ROOT / script if script.startswith("models/") else ROOT / script)
    lean = e.get("lean_module") or e.get("lean_file")
    if lean:
        paths.append(ROOT / lean)
    sim_id = e.get("sim_id")
    if sim_id:
        paths.append(HERE / "math_kg_loops" / f"{sim_id}.yaml")
    slug = e.get("slug", "")
    snake = slug.replace("-", "_")
    if snake:
        paths += [ROOT / "docs" / "media" / f"{snake}_comparison.png",
                  ROOT / "docs" / "media" / f"{snake}_growth.gif"]
        paths += _matching(AUDIT_NAMES, HERE / "audits", slug, (".yaml",))
        paths += [HERE / "kg" / n for n in KG_NAMES if f"-{slug}-audit" in n]
        paths += _matching(RUN_NAMES, HERE / "runs", slug, (".md", ".yaml", ".csv", ".trig", ".json"))
    slot = e.get("slot")
    if slot:
        paths.append(HERE / "cloud" / "dispatches10000" / f"slot-{int(slot):04d}.json")
        paths.append(HERE / "cloud" / "dispatches" / f"slot-{int(slot):04d}.json")
    # Only slot-unique paths inside the repo, never shared dirs
    out, seen_p = [], set()
    for p in paths:
        rp = str(p)
        if rp not in seen_p and str(ROOT) in rp and p.name not in ("datasets", "models", "media", "audits", "kg", "runs"):
            seen_p.add(rp)
            out.append(p)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--apply", action="store_true", help="Actually delete (default: dry-run manifest only)")
    args = ap.parse_args()

    entries = load_entries()
    tier_c = tier_c_entries(entries)
    all_paths: list[str] = []
    n_files = n_dirs = total_bytes = 0
    for e in tier_c:
        for p in artifact_paths(e):
            if p.exists() or p.is_symlink():
                all_paths.append(str(p.relative_to(ROOT)))
                if p.is_dir():
                    n_dirs += 1
                    total_bytes += sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
                else:
                    n_files += 1
                    total_bytes += p.stat().st_size
    manifest = {"tier": "C_scaffold_only", "loops": len(tier_c),
                "slots": sorted(int(e["slot"]) for e in tier_c if e.get("slot")),
                "files": n_files, "dirs": n_dirs, "bytes": total_bytes,
                "paths": sorted(set(all_paths)), "applied": bool(args.apply)}
    MANIFEST.write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
    print(f"Tier C loops: {len(tier_c)} | files {n_files} + dirs {n_dirs} | {total_bytes/1e9:.2f} GB | manifest {MANIFEST.relative_to(ROOT)}")
    if not args.apply:
        print("DRY-RUN — nothing deleted. Re-run with --apply.")
        return 0

    deleted = 0
    for rel in manifest["paths"]:
        p = ROOT / rel
        try:
            if p.is_dir():
                shutil.rmtree(p)
            elif p.exists() or p.is_symlink():
                p.unlink()
            deleted += 1
        except Exception as exc:
            print(f"  skip {rel}: {exc}")
    # Filter Tier-C entries out of the index + campaign summaries
    c_ds = {e.get("dataset_dir") for e in tier_c}
    idx_path = HERE / "runs" / "aristotelian_categories_index.json"
    targets = [idx_path] + sorted((HERE / "runs").glob("campaign_*summary*.json"))
    for t in targets:
        try:
            data = json.loads(t.read_text(encoding="utf-8"))
        except Exception:
            continue
        if isinstance(data, list):
            kept = [x for x in data if x.get("dataset_dir") not in c_ds]
            if len(kept) != len(data):
                t.write_text(json.dumps(kept, indent=1) + "\n", encoding="utf-8")
                print(f"  filtered {t.name}: {len(data)} -> {len(kept)} entries")
    print(f"Deleted {deleted} paths. Tier A + Tier B (real animals/papers) untouched.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

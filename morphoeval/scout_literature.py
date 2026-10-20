#!/usr/bin/env python3
"""
morphoeval/scout_literature.py — Step 1 (`01_scout`) Literature & Specimen Scout CLI.

Used by autonomous Cloud Agent loops (and local runs) during Step 1 of the 12-Step
MorphoEval Closed Loop to:
  1. Inspect a campaign slot (`--slot 1..100`) or custom search query (`--query "..."`).
  2. Scan the repository (`*.py`, `datasets/*`, `morphoeval/audits/*.yaml`,
     `morphoeval/math_kg_loops/*.yaml`) to build the exclusion set of already-modeled
     papers, DOIs, and organisms so no cloud loop ever duplicates existing work.
  3. Query OpenAlex, Crossref, and Wikimedia Commons / iNaturalist public APIs (no API
     key required) to discover open-access mathematical morphogenesis papers with
     explicit governing equations, quantitative measurements, and CC-licensed biological
     specimen photographs.
  4. Emit a structured scout dossier (`datasets/<slug>/scout_manifest.json`) ready for
     Steps 2–12 without modifying any existing tracked files.

Usage:
  python3 morphoeval/scout_literature.py --slot 1 --dry-run
  python3 morphoeval/scout_literature.py --slot 1 --out datasets/<slug>/scout_manifest.json
  python3 morphoeval/scout_literature.py --query "diatom frustule biosilica pattern formation" --taxon "Bacillariophyceae"
  python3 morphoeval/scout_literature.py --list-existing
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
CAMPAIGN_PATH = HERE / "cloud" / "campaign_100_slots.yaml"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def collect_existing_models() -> dict[str, Any]:
    """Scan the repository to find all existing scripts, simulation IDs, DOIs, and datasets."""
    scripts = sorted(p.name for p in ROOT.glob("[0-9][0-9][0-9][0-9]-*.py"))
    datasets = sorted(p.name for p in (ROOT / "datasets").iterdir() if p.is_dir()) if (ROOT / "datasets").exists() else []
    sim_ids: set[str] = set()
    dois: set[str] = set()

    audits_dir = HERE / "audits"
    if audits_dir.exists():
        for af in sorted(audits_dir.glob("*.yaml")):
            if af.name.endswith(".measurements.yaml"):
                continue
            try:
                doc = yaml.safe_load(af.read_text(encoding="utf-8")) or {}
                for r in doc.get("references", []) or []:
                    if r.get("doi"):
                        dois.add(str(r["doi"]).lower().strip())
                for sim in doc.get("simulations", []) or []:
                    if sim.get("id"):
                        sim_ids.add(str(sim["id"]))
            except Exception:
                continue

    loops_dir = HERE / "math_kg_loops"
    if loops_dir.exists():
        for lf in sorted(loops_dir.glob("*.yaml")):
            sim_ids.add(lf.stem)

    return {
        "scripts": scripts,
        "datasets": datasets,
        "simulation_ids": sorted(sim_ids),
        "dois": sorted(dois),
    }


def _http_get_json(url: str, timeout: float = 12.0) -> dict[str, Any]:
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "InfiniteMorphospace-MorphoEval-Scout/1.0 (https://github.com/fractastical/morpholib; mailto:morphoeval@example.org)",
            "Accept": "application/json",
        },
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8", errors="replace"))


def search_openalex(query: str, limit: int = 8, excluded_dois: set[str] | None = None) -> list[dict[str, Any]]:
    """Query OpenAlex works API for high-impact mathematical biology / morphogenesis papers."""
    excluded = {d.lower() for d in (excluded_dois or set())}
    q_enc = urllib.parse.quote(query)
    url = f"https://api.openalex.org/works?search={q_enc}&per-page={limit}&sort=cited_by_count:desc"
    results: list[dict[str, Any]] = []
    try:
        data = _http_get_json(url)
        for item in data.get("results", []):
            raw_doi = item.get("doi") or ""
            doi = raw_doi.replace("https://doi.org/", "").strip()
            if doi and doi.lower() in excluded:
                continue
            auths = [
                a.get("author", {}).get("display_name", "")
                for a in (item.get("authorships") or [])[:4]
                if a.get("author", {}).get("display_name")
            ]
            first_surname = auths[0].split()[-1].lower() if auths else "author"
            first_surname = re.sub(r"[^a-z0-9]", "", first_surname) or "model"
            year = item.get("publication_year") or 2000
            oa = item.get("open_access") or {}
            results.append(
                {
                    "title": item.get("display_name") or item.get("title") or "",
                    "year": int(year),
                    "authors": auths,
                    "first_author_slug": first_surname,
                    "doi": doi,
                    "cited_by_count": int(item.get("cited_by_count") or 0),
                    "is_oa": bool(oa.get("is_oa")),
                    "oa_url": oa.get("oa_url"),
                    "openalex_id": item.get("id"),
                }
            )
    except Exception as exc:
        results.append({"error": f"OpenAlex query failed: {exc}", "query": query})
    return results


def search_wikimedia_specimens(taxon: str, limit: int = 5) -> list[dict[str, Any]]:
    """Search Wikimedia Commons for CC-licensed / public-domain biological specimen images."""
    q_enc = urllib.parse.quote(f"{taxon} filetype:bitmap")
    url = (
        f"https://commons.wikimedia.org/w/api.php?action=query&generator=search"
        f"&gsrsearch={q_enc}&gsrnamespace=6&gsrlimit={limit}"
        f"&prop=imageinfo&iiprop=url|size|mime|extmetadata&format=json"
    )
    specimens: list[dict[str, Any]] = []
    try:
        data = _http_get_json(url)
        pages = (data.get("query") or {}).get("pages") or {}
        for _, pg in sorted(pages.items(), key=lambda kv: kv[0]):
            ii = (pg.get("imageinfo") or [{}])[0]
            ext = ii.get("extmetadata") or {}
            lic = (ext.get("LicenseShortName") or {}).get("value", "CC/Public Domain")
            img_url = ii.get("url")
            if not img_url:
                continue
            specimens.append(
                {
                    "title": pg.get("title", ""),
                    "url": img_url,
                    "width": ii.get("width"),
                    "height": ii.get("height"),
                    "mime": ii.get("mime"),
                    "license": lic,
                }
            )
    except Exception as exc:
        specimens.append({"error": f"Wikimedia Commons query failed: {exc}", "taxon": taxon})
    return specimens


def load_slot(slot_num: int) -> dict[str, Any]:
    if not CAMPAIGN_PATH.exists():
        raise FileNotFoundError(f"Campaign manifest not found: {CAMPAIGN_PATH}")
    doc = yaml.safe_load(CAMPAIGN_PATH.read_text(encoding="utf-8")) or {}
    for s in doc.get("slots", []):
        if int(s["slot"]) == int(slot_num):
            return s
    raise KeyError(f"Slot {slot_num} not found in {CAMPAIGN_PATH}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--slot", type=int, default=None, help="Slot number (1..100) from morphoeval/cloud/campaign_100_slots.yaml")
    ap.add_argument("--query", type=str, default=None, help="Custom literature search query")
    ap.add_argument("--taxon", type=str, default=None, help="Biological taxon or organism for specimen image scouting")
    ap.add_argument("--list-existing", action="store_true", help="Print existing scripts, datasets, simulation IDs, and DOIs in the repo")
    ap.add_argument("--no-network", action="store_true", help="Use slot seed metadata without live OpenAlex/Wikimedia HTTP calls")
    ap.add_argument("--out", type=str, default=None, help="Write scout manifest JSON to this path (e.g. datasets/<slug>/scout_manifest.json)")
    ap.add_argument("--dry-run", action="store_true", help="Print scout dossier to stdout without writing files")
    args = ap.parse_args()

    existing = collect_existing_models()
    if args.list_existing:
        print(json.dumps(existing, indent=2))
        return 0

    slot_meta: dict[str, Any] = {}
    if args.slot is not None:
        slot_meta = load_slot(args.slot)

    query = args.query or slot_meta.get("search_query") or "mathematical morphogenesis biological pattern formation"
    taxon = args.taxon or slot_meta.get("organism") or "Morphogenesis"

    papers: list[dict[str, Any]] = []
    specimens: list[dict[str, Any]] = []
    if not args.no_network:
        papers = search_openalex(query, limit=6, excluded_dois=set(existing["dois"]))
        specimens = search_wikimedia_specimens(taxon, limit=4)

    dossier = {
        "schema_version": "morphoeval-scout-dossier-1.0",
        "scouted_at": utc_now_iso(),
        "slot": slot_meta.get("slot"),
        "slot_id": slot_meta.get("slot_id"),
        "territory": {
            "domain": slot_meta.get("domain", "Biological Morphogenesis"),
            "organism": taxon,
            "math_family": slot_meta.get("math_family", "Partial Differential Equations / Geometric Morphometrics"),
            "search_query": query,
            "seed_landmark_paper": slot_meta.get("seed_paper"),
            "seed_doi": slot_meta.get("seed_doi"),
            "proposed_script": slot_meta.get("proposed_script"),
            "proposed_sim_id": slot_meta.get("proposed_sim_id"),
            "proposed_dataset_dir": slot_meta.get("proposed_dataset_dir"),
            "proposed_lean_module": slot_meta.get("proposed_lean_module"),
        },
        "uniqueness_check": {
            "existing_scripts_count": len(existing["scripts"]),
            "existing_sim_ids_count": len(existing["simulation_ids"]),
            "collision_detected": (
                slot_meta.get("proposed_script") in existing["scripts"]
                or slot_meta.get("proposed_sim_id") in existing["simulation_ids"]
                or (slot_meta.get("seed_doi") or "").lower() in {d.lower() for d in existing["dois"]}
            ),
        },
        "candidate_papers_openalex": papers,
        "candidate_specimens_wikimedia": specimens,
    }

    text = json.dumps(dossier, indent=2, ensure_ascii=False)
    if args.out and not args.dry_run:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text + "\n", encoding="utf-8")
        digest = hashlib.sha256(out_path.read_bytes()).hexdigest()
        print(f"[scout_literature] Wrote scout manifest -> {out_path} (SHA-256: {digest})")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

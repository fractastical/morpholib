#!/usr/bin/env python3
"""
Package a versioned science-team deliverable bundle for Box.

Creates an immutable release folder with reports, tables, figures, and a
MANIFEST.json that records git commit, Python/package versions, input data
paths, and per-file SHA256 hashes.

Default Box destination:
  ~/Library/CloudStorage/Box-Box/Calcium analysis deliverables/releases/

Run:
  python package_science_deliverables.py
  python package_science_deliverables.py --release-id 2026-06-06_claims-v1
  python package_science_deliverables.py --include-gifs --include-detection-qa
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from provenance import RUN_INDEX, collect  # noqa: E402

REPO = HERE.parent
RESULTS = HERE / "analysis_results"
DEFAULT_BOX_ROOT = Path.home() / "Library/CloudStorage/Box-Box"
DEFAULT_RELEASES = DEFAULT_BOX_ROOT / "Calcium analysis deliverables" / "releases"

# (source relative to RESULTS or absolute, dest relative to release root)
REPORT_FILES = [
    "claims_all_in_one.pdf",
    "claims_tested_summary.pdf",
    "claim_video_mapping.pdf",
]
TABLE_FILES = [
    "xy_ground_truth.csv",
    "landmark_responses.csv",
    "wave_catalog/wave_laterality.csv",
    "wave_catalog/wave_catalog.csv",
]
FIGURE_FILES = [
    "landmark_responses.png",
    "wave_catalog/wave_laterality.png",
    "wave_catalog/wave_laterality_conditions.png",
    "wave_catalog/wave_catalog_comparison.png",
]
OVERLAY_PNGS = [
    "wave_catalog/overlays/claim_bidirectional.png",
    "wave_catalog/overlays/claim_neighbor_wave.png",
    "wave_catalog/overlays/claim_neighbor_local.png",
    "wave_catalog/overlays/claim_layer3.png",
    "wave_catalog/overlays/claim_no_contact.png",
]
OVERLAY_GIFS = [
    "wave_catalog/overlays/claim_bidirectional.gif",
    "wave_catalog/overlays/claim_neighbor_wave.gif",
    "wave_catalog/overlays/claim_neighbor_local.gif",
    "wave_catalog/overlays/claim_layer3.gif",
    "wave_catalog/overlays/claim_no_contact.gif",
]
DETECTION_QA = [
    "detection_summary/detection_summary.md",
    "detection_summary/detection_visualizations.pdf",
    "detection_summary/detection_warnings.log",
]
KG_FILES = [
    "calcium_claims.pkg.trig",
    "calcium_claims_graph.json",
]

PIPELINE_STEPS = [
    ("parse_xy_coordinates.py", "xy_ground_truth.csv"),
    ("batch_wave_catalog.py", "wave_catalog/"),
    ("wave_laterality_analysis.py", "wave_catalog/wave_laterality.csv"),
    ("score_landmark_responses.py", "landmark_responses.csv"),
    ("generate_claim_wave_overlays.py", "wave_catalog/overlays/"),
    ("generate_tested_claims_summary.py", "claims_tested_summary.pdf"),
    ("generate_claims_inventory_pdf.py", "claims_inventory.pdf"),
    ("merge_claims_all_in_one.py", "claims_all_in_one.pdf"),
    ("export_calcium_claims_to_pkg.py", "calcium_claims.pkg.trig"),
]

INPUT_PATHS = {
    "xy_coordinates_xlsx": DEFAULT_BOX_ROOT / "Calcium videos" / "XY coordinates.xlsx",
    "calcium_videos_root": DEFAULT_BOX_ROOT / "Calcium videos",
}


def _file_meta(path: Path) -> dict:
    st = path.stat()
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return {
        "bytes": st.st_size,
        "sha256": h.hexdigest(),
        "source_mtime_local": dt.datetime.fromtimestamp(st.st_mtime).isoformat(
            timespec="seconds"),
    }


def _input_meta(path: Path) -> dict | None:
    if not path.exists():
        return {"path": str(path), "exists": False}
    st = path.stat()
    return {
        "path": str(path),
        "exists": True,
        "bytes": st.st_size,
        "modified_local": dt.datetime.fromtimestamp(st.st_mtime).isoformat(
            timespec="seconds"),
    }


def _claim_status() -> dict | None:
    inv = RESULTS / "claims_inventory.pdf"
    if not inv.exists():
        return None
    # Best-effort: read from inventory generator if importable
    try:
        sys.path.insert(0, str(HERE))
        from claims_doc_parser import parse_claims_doc  # noqa: E402
        from generate_claims_inventory_pdf import (  # noqa: E402
            classify, load_landmarks, load_laterality)
        doc = DEFAULT_BOX_ROOT / "Calcium videos" / "Calcium claims.docx"
        if not doc.exists():
            doc = REPO / "Calcium claims.docx"
        recs = parse_claims_doc(doc) if doc.exists() else []
        lat = load_laterality()
        lm = load_landmarks()
        counts: dict[str, int] = {}
        for r in recs:
            status, _, _ = classify(r, lat, lm)
            counts[status] = counts.get(status, 0) + 1
        return {"total": len(recs), **counts}
    except Exception:
        return None


def _copy(src: Path, dest: Path) -> dict:
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)
    meta = _file_meta(dest)
    meta["source"] = str(src)
    meta["bundled_as"] = str(dest.relative_to(dest.parents[len(dest.parents) - 1]))
    return meta


def build_release(
    release_id: str,
    out_root: Path,
    include_gifs: bool,
    include_detection_qa: bool,
    dry_run: bool,
) -> Path:
    release_dir = out_root / release_id
    if release_dir.exists() and not dry_run:
        raise SystemExit(f"Release already exists: {release_dir}\n"
                         "Bump --release-id or remove the folder first.")

    plan: list[tuple[Path, Path]] = []

    for name in REPORT_FILES:
        src = RESULTS / name
        if src.exists():
            plan.append((src, release_dir / "01_reports" / name))

    for rel in TABLE_FILES:
        src = RESULTS / rel
        if src.exists():
            plan.append((src, release_dir / "02_tables" / Path(rel).name))

    for rel in FIGURE_FILES + OVERLAY_PNGS:
        src = RESULTS / rel
        if src.exists():
            dest_name = Path(rel).name
            sub = "overlays" if "overlays" in rel else ""
            plan.append((src, release_dir / "03_figures" / sub / dest_name))

    if include_gifs:
        for rel in OVERLAY_GIFS:
            src = RESULTS / rel
            if src.exists():
                plan.append((src, release_dir / "03_figures" / "overlays"
                             / Path(rel).name))

    if include_detection_qa:
        for rel in DETECTION_QA:
            src = RESULTS / rel
            if src.exists():
                plan.append((src, release_dir / "05_detection_qa" / Path(rel).name))

    for rel in KG_FILES:
        src = RESULTS / rel
        if src.exists():
            plan.append((src, release_dir / "06_knowledge_graph" / Path(rel).name))

    missing = []
    for src, _ in plan:
        if not src.exists():
            missing.append(str(src))
    if missing:
        print("Warning: some planned files are missing and will be skipped:")
        for m in missing:
            print(f"  - {m}")
    plan = [(s, d) for s, d in plan if s.exists()]

    if not plan:
        raise SystemExit("No deliverable files found to package.")

    pack_record = collect(
        "package_science_deliverables.py",
        [release_dir / "MANIFEST.json"],
        extra={"release_id": release_id, "dry_run": dry_run},
    )
    git = pack_record["repository"]
    manifest_files: list[dict] = []

    if not dry_run:
        release_dir.mkdir(parents=True, exist_ok=True)
        for src, dest in plan:
            meta = _copy(src, dest)
            meta["bundled_as"] = str(dest.relative_to(release_dir))
            manifest_files.append(meta)

    else:
        for src, dest in plan:
            manifest_files.append({
                "bundled_as": str(dest.relative_to(release_dir)),
                "source": str(src),
                **_file_meta(src),
            })

    manifest = {
        "release_id": release_id,
        "generated_at_utc": pack_record["generated_at_utc"],
        "generated_at_local": pack_record["generated_at_local"],
        "repository": git,
        "python": pack_record["python"],
        "packages": pack_record["packages"],
        "input_data": {k: _input_meta(p) for k, p in INPUT_PATHS.items()},
        "pipeline_steps": [
            {"script": s, "primary_output": o} for s, o in PIPELINE_STEPS
        ],
        "claim_status": _claim_status(),
        "options": {
            "include_gifs": include_gifs,
            "include_detection_qa": include_detection_qa,
        },
        "files": manifest_files,
    }

    readme = _readme(release_id, manifest, plan)

    if not dry_run:
        prov = release_dir / "04_provenance"
        prov.mkdir(parents=True, exist_ok=True)
        (prov / "MANIFEST.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        (release_dir / "README.txt").write_text(readme, encoding="utf-8")
        shutil.copy2(prov / "MANIFEST.json", release_dir / "MANIFEST.json")
        # Pipeline sidecars + run log
        for sidecar in RESULTS.rglob("*.provenance.json"):
            rel = sidecar.relative_to(RESULTS)
            dest = prov / "pipeline_sidecars" / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(sidecar, dest)
        if RUN_INDEX.exists():
            shutil.copy2(RUN_INDEX, prov / "RUN_INDEX.jsonl")

    return release_dir


def _readme(release_id: str, manifest: dict, plan: list[tuple[Path, Path]]) -> str:
    git = manifest["repository"]
    cs = manifest.get("claim_status") or {}
    lines = [
        f"Calcium Claims — Science Deliverable Release",
        f"===========================================",
        f"",
        f"Release ID:     {release_id}",
        f"Generated UTC:  {manifest['generated_at_utc']}",
        f"Git commit:     {git.get('commit_short', '?')} ({git.get('commit_message', '')})",
        f"Repository:     {git.get('remote', '')}",
        f"Branch:         {git.get('branch', '')}",
        f"Working tree:   {'DIRTY (uncommitted local changes)' if git.get('dirty') else 'clean'}",
        f"",
        f"START HERE",
        f"----------",
        f"1. Open 01_reports/claims_all_in_one.pdf — combined report with summary slide,",
        f"   full claims inventory (16 TESTED / 3 NOT YET), and component analyses.",
        f"2. Skim 02_tables/ for the ground-truth CSVs and per-video scores.",
        f"3. See 03_figures/ for wave-vector overlays and organ-landmark corroboration.",
        f"",
        f"Claim status (from inventory classifier):",
    ]
    if cs:
        lines.append(f"  Total verbatim claims: {cs.get('total', '?')}")
        for k in sorted(cs):
            if k != "total":
                lines.append(f"  {k}: {cs[k]}")
    else:
        lines.append("  (not computed — claims_inventory inputs unavailable)")

    lines += [
        f"",
        f"Ground-truth inputs",
        f"-------------------",
    ]
    for key, meta in manifest.get("input_data", {}).items():
        if not meta:
            continue
        exists = meta.get("exists", False)
        lines.append(f"  {key}: {meta.get('path', '?')}"
                     + ("" if exists else " [NOT FOUND ON THIS MACHINE]"))
        if exists:
            lines.append(f"    modified: {meta.get('modified_local', '?')}")

    lines += [
        f"",
        f"Files in this bundle ({len(plan)}):",
    ]
    for _, dest in plan:
        lines.append(f"  - {dest.name}")

    lines += [
        f"",
        f"Provenance: see MANIFEST.json (SHA256 per file, package versions, pipeline steps).",
        f"",
        f"To reproduce: checkout git commit above, run the pipeline_steps scripts",
        f"documented in MANIFEST.json against the same Box Calcium videos + XY spreadsheet.",
    ]
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--release-id", default=None,
                    help="Folder name (default: YYYY-MM-DD_claims-v1)")
    ap.add_argument("--out-root", default=str(DEFAULT_RELEASES),
                    help="Parent directory for release folders")
    ap.add_argument("--include-gifs", action="store_true",
                    help="Include animated overlay GIFs (~18 MB)")
    ap.add_argument("--include-detection-qa", action="store_true",
                    help="Include canonical detection_summary QA PDF/log")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print plan + manifest only; do not copy files")
    args = ap.parse_args()

    today = dt.date.today().isoformat()
    release_id = args.release_id or f"{today}_claims-v1"
    out_root = Path(args.out_root)

    if not args.dry_run and not out_root.parent.exists():
        print(f"Creating Box deliverables root: {out_root.parent}")
        out_root.parent.mkdir(parents=True, exist_ok=True)

    release_dir = build_release(
        release_id=release_id,
        out_root=out_root,
        include_gifs=args.include_gifs,
        include_detection_qa=args.include_detection_qa,
        dry_run=args.dry_run,
    )

    if args.dry_run:
        print(f"Dry run — would create: {release_dir}")
    else:
        nfiles = len(list(release_dir.rglob("*")))
        size_mb = sum(f.stat().st_size for f in release_dir.rglob("*") if f.is_file()) / 1e6
        print(f"Wrote release: {release_dir}")
        print(f"  {nfiles} paths, {size_mb:.1f} MB total")
        print(f"  README.txt + MANIFEST.json at release root")
        print(f"\nUpload/sync this folder in Box:")
        print(f"  {release_dir}")


if __name__ == "__main__":
    main()

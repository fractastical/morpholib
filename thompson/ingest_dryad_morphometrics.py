#!/usr/bin/env python3
"""
Ingest external morphometric datasets (Dryad-style) for Thompson workflows.

Primary use:
  - Input a Dryad dataset archive (.zip) or extracted folder
  - Parse TPS landmark files into a standardized long-format CSV
  - Copy tabular files (csv/tsv/txt) for traceability
  - Emit a manifest JSON with file inventory and parse stats

This script is intentionally dependency-light (stdlib + pandas) and does not
assume a single dataset schema.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple


TPS_EXTENSIONS = {".tps"}
TABULAR_EXTENSIONS = {".csv", ".tsv", ".txt"}
IGNORE_DIRS = {".git", "__pycache__"}


@dataclass
class TpsRecord:
    specimen_id: str
    block_index: int
    landmark_index: int
    x: float
    y: float
    z: Optional[float]
    source_file: str
    dimensions: int


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _normalize_text_lines(path: Path) -> List[str]:
    # Use latin-1 fallback to avoid hard failures on legacy text files.
    for enc in ("utf-8", "latin-1"):
        try:
            return path.read_text(encoding=enc).splitlines()
        except UnicodeDecodeError:
            continue
    return path.read_text(errors="replace").splitlines()


def parse_tps_file(path: Path) -> Tuple[List[TpsRecord], Dict]:
    """
    Parse a TPS file to long-format landmark records.

    Supported block markers:
      - LM=<n>  (2D)
      - LM3=<n> (3D)
      - optional ID=<value>
    """
    lines = _normalize_text_lines(path)
    records: List[TpsRecord] = []
    parse_meta = {
        "file": str(path),
        "blocks": 0,
        "records": 0,
        "warnings": [],
    }

    i = 0
    block_idx = -1
    current_id = None
    while i < len(lines):
        raw = lines[i].strip()
        if not raw:
            i += 1
            continue

        upper = raw.upper()
        is_lm2 = upper.startswith("LM=")
        is_lm3 = upper.startswith("LM3=")
        if is_lm2 or is_lm3:
            dims = 3 if is_lm3 else 2
            block_idx += 1
            parse_meta["blocks"] += 1
            try:
                n_landmarks = int(raw.split("=", 1)[1].strip())
            except Exception:
                parse_meta["warnings"].append(
                    f"Could not parse landmark count at line {i + 1}: {raw}"
                )
                i += 1
                continue

            # Reset ID for this block; can be set later by ID= line.
            current_id = f"{path.stem}_block_{block_idx:04d}"

            # Read landmark rows directly following LM / LM3
            i += 1
            lm_rows = 0
            while i < len(lines) and lm_rows < n_landmarks:
                row = lines[i].strip()
                if not row:
                    i += 1
                    continue
                if "=" in row and row.split("=", 1)[0].isalpha():
                    # Metadata line appears before expected landmarks.
                    parse_meta["warnings"].append(
                        f"Early metadata line before all landmarks in block {block_idx}: {row}"
                    )
                    break
                parts = row.replace(",", " ").split()
                if len(parts) < dims:
                    parse_meta["warnings"].append(
                        f"Short landmark row at line {i + 1}: {row}"
                    )
                    i += 1
                    continue
                try:
                    x = float(parts[0])
                    y = float(parts[1])
                    z = float(parts[2]) if dims == 3 else None
                except ValueError:
                    parse_meta["warnings"].append(
                        f"Non-numeric landmark row at line {i + 1}: {row}"
                    )
                    i += 1
                    continue
                records.append(
                    TpsRecord(
                        specimen_id=current_id,
                        block_index=block_idx,
                        landmark_index=lm_rows,
                        x=x,
                        y=y,
                        z=z,
                        source_file=str(path),
                        dimensions=dims,
                    )
                )
                lm_rows += 1
                i += 1

            if lm_rows != n_landmarks:
                parse_meta["warnings"].append(
                    f"Block {block_idx}: expected {n_landmarks} landmarks, parsed {lm_rows}"
                )

            # Continue scanning for optional ID= lines for this block.
            # If an ID appears immediately after block, update recently added rows.
            j = i
            while j < len(lines):
                nxt = lines[j].strip()
                if not nxt:
                    j += 1
                    continue
                nxt_upper = nxt.upper()
                if nxt_upper.startswith("ID="):
                    block_id = nxt.split("=", 1)[1].strip() or current_id
                    # Update specimen IDs for this block.
                    start = len(records) - lm_rows
                    for k in range(start, len(records)):
                        records[k].specimen_id = block_id
                    i = j + 1
                    break
                if nxt_upper.startswith("LM=") or nxt_upper.startswith("LM3="):
                    # Next block starts, stop scanning metadata.
                    i = j
                    break
                # Other metadata lines (SCALE, IMAGE, COMMENTS, etc.)
                j += 1
            else:
                i = j
            continue

        i += 1

    parse_meta["records"] = len(records)
    return records, parse_meta


def discover_files(root: Path) -> Dict[str, List[Path]]:
    tps_files: List[Path] = []
    tabular_files: List[Path] = []
    other_files: List[Path] = []
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        if any(part in IGNORE_DIRS for part in p.parts):
            continue
        ext = p.suffix.lower()
        if ext in TPS_EXTENSIONS:
            tps_files.append(p)
        elif ext in TABULAR_EXTENSIONS:
            tabular_files.append(p)
        else:
            other_files.append(p)
    return {
        "tps": sorted(tps_files),
        "tabular": sorted(tabular_files),
        "other": sorted(other_files),
    }


def ingest(input_path: Path, output_dir: Path, dataset_id: str) -> Dict:
    _ensure_dir(output_dir)
    raw_dir = output_dir / "raw_files"
    _ensure_dir(raw_dir)

    manifest: Dict = {
        "dataset_id": dataset_id,
        "input_path": str(input_path.resolve()),
        "output_dir": str(output_dir.resolve()),
        "tps_files": [],
        "tabular_files": [],
        "other_files_count": 0,
        "tps_parse_stats": {
            "files": 0,
            "blocks": 0,
            "records": 0,
            "warnings": 0,
        },
    }

    files = discover_files(input_path)
    manifest["other_files_count"] = len(files["other"])

    # Parse all TPS into one normalized long CSV.
    all_records: List[TpsRecord] = []
    parse_details: List[Dict] = []
    for tps in files["tps"]:
        rel = tps.relative_to(input_path)
        dst = raw_dir / rel
        _ensure_dir(dst.parent)
        shutil.copy2(tps, dst)
        manifest["tps_files"].append(str(rel))

        recs, meta = parse_tps_file(tps)
        all_records.extend(recs)
        parse_details.append(meta)
        manifest["tps_parse_stats"]["files"] += 1
        manifest["tps_parse_stats"]["blocks"] += meta["blocks"]
        manifest["tps_parse_stats"]["records"] += meta["records"]
        manifest["tps_parse_stats"]["warnings"] += len(meta["warnings"])

    landmarks_csv = output_dir / "landmarks_long.csv"
    with landmarks_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "dataset_id",
                "specimen_id",
                "block_index",
                "landmark_index",
                "x",
                "y",
                "z",
                "dimensions",
                "source_file",
            ]
        )
        for r in all_records:
            writer.writerow(
                [
                    dataset_id,
                    r.specimen_id,
                    r.block_index,
                    r.landmark_index,
                    r.x,
                    r.y,
                    "" if r.z is None else r.z,
                    r.dimensions,
                    r.source_file,
                ]
            )

    # Copy tabular files unchanged for provenance/reference.
    tabular_index_rows = []
    for tab in files["tabular"]:
        rel = tab.relative_to(input_path)
        dst = raw_dir / rel
        _ensure_dir(dst.parent)
        shutil.copy2(tab, dst)
        manifest["tabular_files"].append(str(rel))
        tabular_index_rows.append((str(rel), tab.suffix.lower()))

    tabular_index_csv = output_dir / "tabular_index.csv"
    with tabular_index_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["relative_path", "extension"])
        writer.writerows(tabular_index_rows)

    manifest["outputs"] = {
        "landmarks_long_csv": str(landmarks_csv.resolve()),
        "tabular_index_csv": str(tabular_index_csv.resolve()),
    }
    manifest["tps_parse_details"] = parse_details

    manifest_path = output_dir / "ingestion_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ingest Dryad-style morphometric dataset archives/folders."
    )
    parser.add_argument(
        "input",
        help="Path to dataset zip file or extracted dataset directory",
    )
    parser.add_argument(
        "--dataset-id",
        default="dryad_unknown",
        help="Identifier for output rows (e.g., dryad_14fn1)",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory (default: thompson/datasets/<dataset-id>)",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input does not exist: {input_path}")

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(__file__).resolve().parent / "datasets" / args.dataset_id

    # If zip provided, extract to temporary directory first.
    if input_path.is_file() and input_path.suffix.lower() == ".zip":
        with tempfile.TemporaryDirectory(prefix="dryad_ingest_") as tmp:
            tmp_dir = Path(tmp)
            with zipfile.ZipFile(input_path, "r") as zf:
                zf.extractall(tmp_dir)
            # Handle archives containing one top-level directory.
            children = [p for p in tmp_dir.iterdir()]
            root = children[0] if len(children) == 1 and children[0].is_dir() else tmp_dir
            manifest = ingest(root, output_dir, args.dataset_id)
    elif input_path.is_dir():
        manifest = ingest(input_path, output_dir, args.dataset_id)
    else:
        raise ValueError("Input must be a .zip archive or directory")

    print("Ingestion complete.")
    print(f"  Dataset: {manifest['dataset_id']}")
    print(f"  TPS files: {manifest['tps_parse_stats']['files']}")
    print(f"  TPS blocks: {manifest['tps_parse_stats']['blocks']}")
    print(f"  Landmarks: {manifest['tps_parse_stats']['records']}")
    print(f"  Warnings: {manifest['tps_parse_stats']['warnings']}")
    print(f"  Manifest: {output_dir / 'ingestion_manifest.json'}")


if __name__ == "__main__":
    main()


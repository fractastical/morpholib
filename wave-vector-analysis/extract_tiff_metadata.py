#!/usr/bin/env python3
"""
Extract zoom and related metadata from TIFF files.

Reads TIFF tags (especially ImageDescription, tag 270) that ImageJ's
"Image > Show Info" displays. Parses for zoom-related keys and reports
XResolution, YResolution, ResolutionUnit as fallback.

Usage:
  python extract_tiff_metadata.py path/to/file.tif [file2.tif ...]
  python extract_tiff_metadata.py path/to/dir --batch
  python extract_tiff_metadata.py file.tif --verbose
  python extract_tiff_metadata.py path/to/dir --batch --output metadata.json
"""

import argparse
import json
import re
import sys
from pathlib import Path

import tifffile as tiff

# TIFF tag names for resolution (Baseline TIFF)
TAG_IMAGE_DESCRIPTION = 270
TAG_X_RESOLUTION = 282
TAG_Y_RESOLUTION = 283
TAG_RESOLUTION_UNIT = 296

# Max chars of ImageDescription to print by default (full in returned dict)
IMAGE_DESCRIPTION_PRINT_LIMIT = 500


def _tag_value_to_str(value):
    """Convert a tag value to string; decode bytes as UTF-8."""
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, (list, tuple)):
        # Rationals or multi-value; show first or join
        if not value:
            return ""
        v = value[0]
        if hasattr(v, "num") and hasattr(v, "den"):  # Rational
            return str(float(v.num) / v.den) if v.den else str(v.num)
        return " ".join(str(x) for x in value)
    return str(value)


def _parse_zoom_from_description(description_str):
    """
    Parse ImageDescription text for zoom-related values.
    ImageJ "Show Info" uses Key = Value pairs; zoom appears as "Zoom = 5.60x".
    Returns (zoom_value, raw_matched_line) or (None, None).
    """
    if not description_str or not isinstance(description_str, str):
        return None, None
    desc = description_str.strip()
    if not desc:
        return None, None

    def _to_float(s):
        """Strip trailing 'x' (magnification) and convert to float."""
        s = (s or "").strip().rstrip("xX")
        if not s:
            return None
        try:
            return float(s)
        except ValueError:
            return None

    # Split into lines and look for zoom (case-insensitive)
    for line in desc.splitlines():
        line = line.strip()
        if not line:
            continue
        lower = line.lower()
        if "zoom" not in lower:
            continue

        # ImageJ format: "Zoom = 5.60x" (key = value)
        if "=" in line:
            key, _, rest = line.partition("=")
            if "zoom" in key.lower():
                val_str = rest.strip().split()[0] if rest else ""
                zoom_val = _to_float(val_str)
                if zoom_val is not None:
                    return zoom_val, line
                return val_str or None, line

        # Fallback: "Zoom: 5.60x" or "Zoom 5.60x"
        match = re.search(r"zoom\s*[:\s]+\s*([0-9.]+x?)", line, re.IGNORECASE)
        if match:
            zoom_val = _to_float(match.group(1))
            if zoom_val is not None:
                return zoom_val, line

        # Line contains "zoom" but no clear number; return line for debugging
        return None, line

    return None, None


def extract_tiff_metadata(tiff_path, include_all_tags=False):
    """
    Read TIFF tags from the first page and extract zoom/metadata.

    Args:
        tiff_path: Path to a TIFF file (str or Path).
        include_all_tags: If True, add 'all_tags' to the result (dict of tag name -> value string).

    Returns:
        Dict with:
          - path: str path
          - zoom: float or str or None
          - image_description: str (full) or None
          - x_resolution: float or None
          - y_resolution: float or None
          - resolution_unit: str or None (e.g. "2" for inches)
          - n_pages: int
        If include_all_tags=True, also 'all_tags': dict.
    """
    path = Path(tiff_path).resolve()
    result = {
        "path": str(path),
        "zoom": None,
        "image_description": None,
        "x_resolution": None,
        "y_resolution": None,
        "resolution_unit": None,
        "n_pages": 0,
    }
    if include_all_tags:
        result["all_tags"] = {}

    try:
        with tiff.TiffFile(path) as tif:
            result["n_pages"] = len(tif.pages)
            if not tif.pages:
                return result

            page = tif.pages[0]
            tags = page.tags

            for tag in tags.values():
                name = tag.name
                raw = tag.value
                value_str = _tag_value_to_str(raw)

                if include_all_tags:
                    result["all_tags"][name] = value_str

                if name == "ImageDescription" or tag.code == TAG_IMAGE_DESCRIPTION:
                    result["image_description"] = value_str or None
                    zoom_val, _ = _parse_zoom_from_description(value_str)
                    result["zoom"] = zoom_val

                elif name == "XResolution" or tag.code == TAG_X_RESOLUTION:
                    if isinstance(raw, (list, tuple)) and len(raw) >= 1:
                        r = raw[0]
                        if hasattr(r, "num") and hasattr(r, "den") and r.den:
                            result["x_resolution"] = r.num / r.den
                        else:
                            try:
                                result["x_resolution"] = float(r)
                            except (TypeError, ValueError):
                                result["x_resolution"] = None
                    elif value_str:
                        try:
                            result["x_resolution"] = float(value_str.split()[0])
                        except (ValueError, IndexError):
                            pass

                elif name == "YResolution" or tag.code == TAG_Y_RESOLUTION:
                    if isinstance(raw, (list, tuple)) and len(raw) >= 1:
                        r = raw[0]
                        if hasattr(r, "num") and hasattr(r, "den") and r.den:
                            result["y_resolution"] = r.num / r.den
                        else:
                            try:
                                result["y_resolution"] = float(r)
                            except (TypeError, ValueError):
                                result["y_resolution"] = None
                    elif value_str:
                        try:
                            result["y_resolution"] = float(value_str.split()[0])
                        except (ValueError, IndexError):
                            pass

                elif name == "ResolutionUnit" or tag.code == TAG_RESOLUTION_UNIT:
                    result["resolution_unit"] = value_str or None

            # If we have description but zoom not yet set, try parsing again (in case tag order differed)
            if result["zoom"] is None and result["image_description"]:
                zoom_val, _ = _parse_zoom_from_description(result["image_description"])
                result["zoom"] = zoom_val

    except Exception as e:
        result["error"] = str(e)

    return result


def metadata_to_export_dict(meta):
    """
    Return a JSON-serializable dict with all metadata for export.
    Uses 'tags' (all tag name -> string value) and standard parsed fields.
    """
    out = {
        "path": meta.get("path"),
        "filename": Path(meta["path"]).name if meta.get("path") else None,
        "zoom": meta.get("zoom"),
        "image_description": meta.get("image_description"),
        "x_resolution": meta.get("x_resolution"),
        "y_resolution": meta.get("y_resolution"),
        "resolution_unit": meta.get("resolution_unit"),
        "n_pages": meta.get("n_pages", 0),
    }
    if "error" in meta:
        out["error"] = meta["error"]
    if "all_tags" in meta:
        out["tags"] = dict(meta["all_tags"])
    return out


def _print_result(meta, verbose=False, description_limit=IMAGE_DESCRIPTION_PRINT_LIMIT):
    """Print one extraction result to stdout."""
    path = meta.get("path", "")
    name = Path(path).name
    print(f"\n{'='*60}")
    print(f"File: {name}")
    print(f"Path: {path}")
    print(f"{'='*60}")

    if "error" in meta:
        print(f"  Error: {meta['error']}")
        return

    print(f"  Pages: {meta.get('n_pages', 0)}")

    desc = meta.get("image_description")
    if desc:
        if len(desc) > description_limit:
            print(f"  ImageDescription (first {description_limit} chars):")
            print(f"    {desc[:description_limit]!r}...")
        else:
            print(f"  ImageDescription: {desc!r}")
    else:
        print("  ImageDescription: (not present)")

    zoom = meta.get("zoom")
    if zoom is not None:
        print(f"  Zoom: {zoom}")
    else:
        print("  Zoom: (not found in ImageDescription)")

    xr = meta.get("x_resolution")
    yr = meta.get("y_resolution")
    ru = meta.get("resolution_unit")
    if xr is not None or yr is not None or ru:
        print(f"  XResolution: {xr}")
        print(f"  YResolution: {yr}")
        print(f"  ResolutionUnit: {ru}")

    if verbose and "all_tags" in meta:
        print("\n  All tags:")
        for k, v in sorted(meta["all_tags"].items()):
            v_show = v if len(str(v)) <= 80 else str(v)[:77] + "..."
            print(f"    {k}: {v_show}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract zoom and metadata from TIFF files (ImageDescription and resolution tags)."
    )
    parser.add_argument(
        "input",
        nargs="+",
        help="TIFF file(s) or a single directory (use with --batch)",
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help="If a single input is a directory, process all .tif/.tiff files inside",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print all TIFF tag names and values",
    )
    parser.add_argument(
        "--output",
        "-o",
        metavar="FILE",
        help="Write all metadata to a single JSON file (array of records, one per TIFF)",
    )
    parser.add_argument(
        "--output-dir",
        metavar="DIR",
        help="Write one JSON file per TIFF into DIR (filename: <stem>.json)",
    )
    args = parser.parse_args()

    inputs = list(args.input)
    if len(inputs) == 1 and args.batch:
        base = Path(inputs[0])
        if not base.is_dir():
            print(f"Error: with --batch, input must be a directory: {base}", file=sys.stderr)
            sys.exit(1)
        paths = sorted(base.rglob("*.tif")) + sorted(base.rglob("*.tiff"))
        # Deduplicate (e.g. .tif and .tiff same stem)
        seen = set()
        unique = []
        for p in paths:
            if p not in seen:
                seen.add(p)
                unique.append(p)
        paths = unique
        if not paths:
            print(f"No TIFF files found under {base}", file=sys.stderr)
            sys.exit(1)
    else:
        paths = []
        for p in inputs:
            path = Path(p)
            if not path.exists():
                print(f"Error: not found: {path}", file=sys.stderr)
                continue
            if path.is_file():
                if path.suffix.lower() in (".tif", ".tiff"):
                    paths.append(path)
                else:
                    print(f"Warning: skipping non-TIFF file: {path}", file=sys.stderr)
            else:
                print(f"Warning: skipping directory (use --batch): {path}", file=sys.stderr)

        if not paths:
            sys.exit(1)

    export_to_file = bool(args.output)
    export_to_dir = bool(args.output_dir)
    include_all_tags = args.verbose or export_to_file or export_to_dir

    all_export = []  # for --output single file
    written_per_file = 0

    for tiff_path in paths:
        meta = extract_tiff_metadata(tiff_path, include_all_tags=include_all_tags)
        if not export_to_file and not export_to_dir:
            _print_result(meta, verbose=args.verbose)
        else:
            record = metadata_to_export_dict(meta)
            all_export.append(record)
            if export_to_dir:
                out_dir = Path(args.output_dir)
                out_dir.mkdir(parents=True, exist_ok=True)
                out_path = out_dir / f"{tiff_path.stem}.json"
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(record, f, indent=2, ensure_ascii=False)
                print(f"  Wrote {out_path}")
                written_per_file += 1

    if export_to_dir and written_per_file:
        print(f"Wrote {written_per_file} JSON file(s) to {args.output_dir}")

    if export_to_file and all_export:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(
                {"source": "extract_tiff_metadata.py", "files": all_export},
                f,
                indent=2,
                ensure_ascii=False,
            )
        print(f"Wrote metadata for {len(all_export)} file(s) to {out_path}")

    if not export_to_file and not export_to_dir:
        print()


if __name__ == "__main__":
    main()

"""
Literature speed benchmarks and comparison against our pipeline results.

Stores benchmark propagation speeds and distances from reference figures (μm/s, μm).
Comparisons are done per run with explicit zoom/scale so zoom differentials
between TIFFs are not mixed. Output is written to a dedicated comparison file
(e.g. analysis_results/literature_comparison.json or .csv).

Usage:
  from literature_benchmarks import LITERATURE_BENCHMARKS, compare_run_to_literature
  compare_run_to_literature(
      spark_tracks_path="path/to/spark_tracks.csv",
      output_path="analysis_results/literature_comparison.json",
      embryo_length_um=1000.0,   # or None to skip μm/s conversion
      tiff_metadata_path=None,   # optional: JSON from extract_tiff_metadata --output
      run_id="folder_1/B - Substack",
  )
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# -----------------------------------------------------------------------------
# Literature benchmarks (from reference figures): speed (μm/s) and distance (μm)
# Each entry: scenario id, speed_min, speed_max (μm/s), distance_um (optional)
# -----------------------------------------------------------------------------

LITERATURE_BENCHMARKS = [
    {"id": "head_head_neighbor", "scenario": "Head–head (neighbor)", "speed_um_s_min": 97.64, "speed_um_s_max": 130.19, "distance_um": 1952.824, "notes": "Local response neighbor"},
    {"id": "tail_head", "scenario": "Tail–head", "speed_um_s_min": 10.1, "speed_um_s_max": 10.4, "distance_um": 1656.845, "notes": "Latency 0–5 s + 159 s"},
    {"id": "head_head_other", "scenario": "Head–head (other)", "speed_um_s_min": 22.5, "speed_um_s_max": 23.7, "distance_um": 2203.296, "notes": "Latency 0–5 s + 93 s"},
    {"id": "single_embryo_short", "scenario": "Single embryo (short)", "speed_um_s_min": 194.4, "speed_um_s_max": 972.2, "distance_um": 972.16, "notes": "Length 972.16 µm, time 0–5 s"},
    {"id": "single_embryo_long", "scenario": "Single embryo (long)", "speed_um_s_min": 364.3, "speed_um_s_max": 1821.4, "distance_um": 1821.44, "notes": "Length 1821.44 µm, time 0–5 s"},
    {"id": "stimulus_tail_neighbor_1", "scenario": "Stimulus → tail in neighbor", "speed_um_s_min": 634.9, "speed_um_s_max": 3174.5, "distance_um": None, "notes": ""},
    {"id": "stimulus_anterior_neighbor", "scenario": "Stimulus → anterior in neighbor", "speed_um_s_min": 126.6, "speed_um_s_max": 443.0, "distance_um": None, "notes": ""},
    {"id": "stimulus_tail_neighbor_2", "scenario": "Stimulus → tail in neighbor (2)", "speed_um_s_min": 576.6, "speed_um_s_max": 2883.0, "distance_um": None, "notes": ""},
    {"id": "within_embryo_tail", "scenario": "Within embryo: stimulus → tail", "speed_um_s_min": 331.1, "speed_um_s_max": 1655.5, "distance_um": None, "notes": ""},
    {"id": "stimulus_response_neighbor", "scenario": "Stimulus and response neighbor", "speed_um_s_min": 305.0, "speed_um_s_max": 813.1, "distance_um": None, "notes": ""},
]


def um_per_px_from_embryo_length(embryo_length_px: float, embryo_length_um: float) -> float:
    """Compute μm per pixel from reference embryo length in px and in μm (avoids mixing zoom)."""
    if embryo_length_px is None or embryo_length_px <= 0 or embryo_length_um is None or embryo_length_um <= 0:
        raise ValueError("embryo_length_px and embryo_length_um must be positive")
    return float(embryo_length_um) / float(embryo_length_px)


def compare_speed_to_benchmarks(
    our_speed_um_s: float,
    benchmarks: Optional[List[Dict]] = None,
) -> List[Dict]:
    """
    Compare a single speed (μm/s) to all literature benchmarks.
    Returns list of dicts with benchmark id, scenario, whether our value falls in range, and distance to range.
    """
    if benchmarks is None:
        benchmarks = LITERATURE_BENCHMARKS
    results = []
    for b in benchmarks:
        lo, hi = b["speed_um_s_min"], b["speed_um_s_max"]
        within = lo <= our_speed_um_s <= hi
        if our_speed_um_s < lo:
            dist_to_range = lo - our_speed_um_s
        elif our_speed_um_s > hi:
            dist_to_range = our_speed_um_s - hi
        else:
            dist_to_range = 0.0
        results.append({
            "benchmark_id": b["id"],
            "scenario": b["scenario"],
            "benchmark_speed_um_s_min": lo,
            "benchmark_speed_um_s_max": hi,
            "our_speed_um_s": our_speed_um_s,
            "within_range": within,
            "dist_to_range_um_s": dist_to_range,
        })
    return results


def run_comparison(
    spark_tracks_path: str,
    output_path: str,
    run_id: Optional[str] = None,
    embryo_length_um: Optional[float] = None,
    um_per_px: Optional[float] = None,
    tiff_metadata_path: Optional[str] = None,
    zoom: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Load spark_tracks.csv, compute our speed stats (px/s and, if scale given, μm/s),
    compare to literature benchmarks, and write results to output_path.
    Zoom/scale are stored per run so differentials are explicit.

    One of (embryo_length_um, um_per_px) must be provided to get μm/s. If tiff_metadata_path
    points to a JSON from extract_tiff_metadata --output, zoom can be read from the first file.
    """
    import pandas as pd
    from embryo_units import speed_px_s_to_um_s

    path = Path(spark_tracks_path)
    if not path.exists():
        raise FileNotFoundError(f"spark_tracks not found: {spark_tracks_path}")

    df = pd.read_csv(path)
    if run_id is None:
        run_id = path.stem or "spark_tracks"

    # Per-run reference length (pixels)
    embryo_length_px = None
    if "embryo_length_px" in df.columns:
        vals = df["embryo_length_px"].dropna()
        if len(vals) > 0:
            embryo_length_px = float(vals.iloc[0])

    # Scale: μm per pixel (from explicit um_per_px or from embryo length in μm)
    scale_um_per_px = um_per_px
    if scale_um_per_px is None and embryo_length_um is not None and embryo_length_px is not None and embryo_length_px > 0:
        scale_um_per_px = um_per_px_from_embryo_length(embryo_length_px, embryo_length_um)

    # Optional: read zoom from metadata file (one TIFF per run; zoom differentials in file)
    if zoom is None and tiff_metadata_path and Path(tiff_metadata_path).exists():
        with open(tiff_metadata_path, encoding="utf-8") as f:
            meta = json.load(f)
        files = meta.get("files", [])
        if files:
            zoom = files[0].get("zoom")

    # Our speed stats (px/s)
    speed_col = df["speed"] if "speed" in df.columns else None
    if speed_col is not None:
        valid = speed_col.dropna()
        valid = valid[valid > 0]
        our_mean_speed_px_s = float(valid.mean()) if len(valid) > 0 else None
        our_median_speed_px_s = float(valid.median()) if len(valid) > 0 else None
        our_max_speed_px_s = float(valid.max()) if len(valid) > 0 else None
    else:
        our_mean_speed_px_s = our_median_speed_px_s = our_max_speed_px_s = None

    # Convert to μm/s if we have scale
    our_mean_speed_um_s = our_median_speed_um_s = our_max_speed_um_s = None
    if scale_um_per_px is not None and our_mean_speed_px_s is not None:
        our_mean_speed_um_s = speed_px_s_to_um_s(our_mean_speed_px_s, scale_um_per_px)
        our_median_speed_um_s = speed_px_s_to_um_s(our_median_speed_px_s, scale_um_per_px) if our_median_speed_px_s is not None else None
        our_max_speed_um_s = speed_px_s_to_um_s(our_max_speed_px_s, scale_um_per_px) if our_max_speed_px_s is not None else None

    # Compare mean speed to benchmarks (if we have μm/s)
    benchmark_comparisons = None
    if our_mean_speed_um_s is not None:
        benchmark_comparisons = compare_speed_to_benchmarks(our_mean_speed_um_s)

    out = {
        "run_id": run_id,
        "spark_tracks_path": str(path.resolve()),
        "zoom": zoom,
        "embryo_length_px": embryo_length_px,
        "embryo_length_um": embryo_length_um,
        "um_per_px": scale_um_per_px,
        "our_mean_speed_px_s": our_mean_speed_px_s,
        "our_median_speed_px_s": our_median_speed_px_s,
        "our_max_speed_px_s": our_max_speed_px_s,
        "our_mean_speed_um_s": our_mean_speed_um_s,
        "our_median_speed_um_s": our_median_speed_um_s,
        "our_max_speed_um_s": our_max_speed_um_s,
        "literature_benchmarks": LITERATURE_BENCHMARKS,
        "benchmark_comparisons": benchmark_comparisons,
    }

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    return out


def load_comparisons(comparison_path: str) -> Dict:
    """Load a single comparison JSON (one run)."""
    with open(comparison_path, encoding="utf-8") as f:
        return json.load(f)


def load_all_comparisons(comparison_dir: str, pattern: str = "literature_comparison*.json") -> List[Dict]:
    """Load all comparison JSONs in a directory (e.g. one per run with zoom differentials)."""
    from glob import glob
    path = Path(comparison_dir)
    if not path.is_dir():
        return []
    files = sorted(path.glob(pattern))
    return [load_comparisons(str(f)) for f in files]


def write_comparisons_table(comparison_paths: List[str], output_csv: str) -> None:
    """
    Write a single CSV table summarizing multiple comparison files (one row per run).
    Columns include run_id, zoom, um_per_px, our_mean_speed_um_s, and one column per
    benchmark (within_range_<id> or similar) so zoom differentials are visible.
    """
    import pandas as pd
    rows = []
    for p in comparison_paths:
        c = load_comparisons(p)
        row = {
            "run_id": c.get("run_id"),
            "zoom": c.get("zoom"),
            "embryo_length_px": c.get("embryo_length_px"),
            "um_per_px": c.get("um_per_px"),
            "our_mean_speed_px_s": c.get("our_mean_speed_px_s"),
            "our_mean_speed_um_s": c.get("our_mean_speed_um_s"),
            "our_median_speed_um_s": c.get("our_median_speed_um_s"),
            "our_max_speed_um_s": c.get("our_max_speed_um_s"),
        }
        for bc in (c.get("benchmark_comparisons") or []):
            bid = bc.get("benchmark_id", "")
            row[f"within_{bid}"] = bc.get("within_range")
            row[f"dist_to_range_{bid}"] = bc.get("dist_to_range_um_s")
        rows.append(row)
    pd.DataFrame(rows).to_csv(output_csv, index=False)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Compare pipeline speeds to literature benchmarks; output to a separate file (zoom/scale per run).")
    parser.add_argument("spark_tracks", help="Path to spark_tracks.csv")
    parser.add_argument("--output", "-o", default="analysis_results/literature_comparison.json", help="Output comparison file (JSON)")
    parser.add_argument("--run-id", help="Run identifier (default: stem of spark_tracks path)")
    parser.add_argument("--embryo-length-um", type=float, help="Reference embryo length in μm (with embryo_length_px gives um_per_px)")
    parser.add_argument("--um-per-px", type=float, help="Explicit μm per pixel (overrides embryo-length-um if set)")
    parser.add_argument("--tiff-metadata", help="Optional JSON from extract_tiff_metadata --output (to attach zoom)")
    parser.add_argument("--zoom", type=float, help="Optional zoom value for this run (e.g. from ImageJ)")
    args = parser.parse_args()
    if not args.embryo_length_um and not args.um_per_px:
        parser.error("One of --embryo-length-um or --um-per-px is required for μm/s comparison")
    result = run_comparison(
        args.spark_tracks,
        args.output,
        run_id=args.run_id,
        embryo_length_um=args.embryo_length_um,
        um_per_px=args.um_per_px,
        tiff_metadata_path=args.tiff_metadata,
        zoom=args.zoom,
    )
    print(f"Wrote comparison to {args.output}")
    print(f"  run_id: {result['run_id']}, zoom: {result.get('zoom')}, um_per_px: {result.get('um_per_px')}")
    print(f"  our_mean_speed_um_s: {result.get('our_mean_speed_um_s')}")

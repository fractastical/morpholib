#!/usr/bin/env python3
"""
Batch wave catalog: run dense pixel vectors + wave roll-up on every calcium
video and aggregate into a single catalog.

For each `*_C-*.tif` under --root:
  1. generate_pixel_brightness_vectors.py  -> pixel_frame_deltas.csv
  2. rollup_pixel_vectors_to_waves.py       -> waves/wave_events.csv
Then merge every wave_events.csv into one `wave_catalog.csv` (tagged with the
video name, stimulus, contact, orientation parsed from the filename/path) and
render comparison plots.

NOTE: fps is set to 1.0 for every video (true frame rates are not recorded in
the stacks), so speeds are in px/frame and are comparable *across* videos.
"""

import argparse
import csv
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tifffile as tiff

HERE = Path(__file__).resolve().parent


def parse_meta(path: Path):
    """Pull stimulus / contact / orientation hints from name + folder path."""
    name = path.name
    prefix = name.split("_")[0]
    stim = {"W": "wound", "P": "press"}.get(prefix[:1], "?")
    # second prefix letter: G vs S series
    series = prefix[1:2]
    m = re.search(r"_C-([A-Za-z]+)", name)
    orient = m.group(1) if m else ""
    lower = str(path).lower()
    if "no physical contact" in lower or "no-contact" in lower or "no contact" in lower:
        contact = "no-contact"
    elif "physical contact" in lower:
        contact = "contact"
    else:
        contact = "?"
    return {
        "video": name,
        "prefix": prefix,
        "stimulus": stim,
        "series": series,
        "orientation": orient,
        "contact": contact,
    }


def run(cmd):
    print("  $", " ".join(str(c) for c in cmd[-4:]))
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print(res.stdout[-500:])
        print(res.stderr[-800:])
    return res.returncode == 0


def process_video(video: Path, out_root: Path, threshold, max_points, max_link):
    meta = parse_meta(video)
    vid_dir = out_root / video.stem
    deltas = vid_dir / "pixel_frame_deltas.csv"
    waves_csv = vid_dir / "waves" / "wave_events.csv"

    try:
        with tiff.TiffFile(video) as tf:
            shape = tf.pages[0].shape
            n_pages = len(tf.pages)
    except Exception as e:  # noqa: BLE001
        print(f"  ! cannot read {video.name}: {e}")
        return meta, []
    h, w = shape[:2]

    if not deltas.exists():
        ok = run([
            sys.executable, str(HERE / "generate_pixel_brightness_vectors.py"),
            str(video), "0", "--fps", "1.0",
            "--output-dir", str(vid_dir),
            "--linear-threshold", str(threshold),
            "--max-points", str(max_points),
            "--max-link-px", str(max_link),
            "--png-every", "0",
        ])
        if not ok or not deltas.exists():
            print(f"  ! vector step failed for {video.name}")
            return meta, []

    if not waves_csv.exists():
        ok = run([
            sys.executable, str(HERE / "rollup_pixel_vectors_to_waves.py"),
            str(deltas), "--img-height", str(h), "--img-width", str(w),
        ])
        if not ok or not waves_csv.exists():
            print(f"  ! rollup step failed for {video.name}")
            return meta, []

    waves = []
    with open(waves_csv, newline="") as f:
        for row in csv.DictReader(f):
            row.update(meta)
            row["n_pages"] = n_pages
            waves.append(row)
    return meta, waves


def render_comparison(catalog, out_png):
    if not catalog:
        return
    groups = defaultdict(lambda: {"prop": [], "front": [], "dur": []})
    for w in catalog:
        key = w["prefix"][:2]
        try:
            prop = float(w["propagation_speed_px_per_s"])
            front = float(w["mean_front_speed_px_per_s"])
            dur = float(w["duration_s"])
        except (ValueError, KeyError):
            continue
        if np.isfinite(prop):
            groups[key]["prop"].append(prop)
        if np.isfinite(front):
            groups[key]["front"].append(front)
        groups[key]["dur"].append(dur)

    keys = sorted(groups)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    def box(ax, field, title, ylabel):
        data = [groups[k][field] for k in keys]
        bp = ax.boxplot(data, tick_labels=keys, showfliers=False)
        for i, k in enumerate(keys, start=1):
            ys = groups[k][field]
            xs = np.random.normal(i, 0.05, size=len(ys))
            ax.scatter(xs, ys, s=14, alpha=0.5, color="#1f77b4")
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xlabel("video series")

    box(axes[0], "prop", "Propagation speed (centroid)", "px/frame")
    box(axes[1], "front", "Front speed (per-pixel)", "px/frame")
    box(axes[2], "dur", "Wave duration", "frames")

    counts = {k: len(groups[k]["dur"]) for k in keys}
    fig.suptitle("Wave catalog by series  (" +
                 ", ".join(f"{k}: {counts[k]} waves" for k in keys) + ")",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="/Users/jdietz/Library/CloudStorage/"
                    "Box-Box/Calcium videos/Calcium videos")
    ap.add_argument("--out", default=str(HERE / "analysis_results" / "wave_catalog"))
    ap.add_argument("--threshold", type=float, default=0.4)
    ap.add_argument("--max-points", type=int, default=1500)
    ap.add_argument("--max-link", type=float, default=12.0)
    ap.add_argument("--limit", type=int, default=None,
                    help="process only first N videos (debug)")
    args = ap.parse_args()

    root = Path(args.root)
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    videos = sorted(p for p in root.rglob("*")
                    if p.is_file() and p.suffix.lower() in (".tif", ".tiff")
                    and "_C-" in p.name)
    if args.limit:
        videos = videos[:args.limit]
    print(f"Found {len(videos)} videos under {root}")

    catalog = []
    summary = []
    for i, v in enumerate(videos, 1):
        print(f"[{i}/{len(videos)}] {v.name}")
        meta, waves = process_video(v, out_root, args.threshold,
                                    args.max_points, args.max_link)
        catalog.extend(waves)
        summary.append({**meta, "n_waves": len(waves)})

    if catalog:
        fields = list(catalog[0].keys())
        with open(out_root / "wave_catalog.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(catalog)
        render_comparison(catalog, out_root / "wave_catalog_comparison.png")

    with open(out_root / "video_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        w.writeheader()
        w.writerows(summary)

    print(f"\nDone. {len(catalog)} waves across {len(videos)} videos.")
    print(f"  {out_root/'wave_catalog.csv'}")
    print(f"  {out_root/'video_summary.csv'}")
    print(f"  {out_root/'wave_catalog_comparison.png'}")


if __name__ == "__main__":
    main()

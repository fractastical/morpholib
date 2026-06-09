#!/usr/bin/env python3
"""
Score the manually-annotated organ-response landmarks against the calcium data.

For every organ landmark in `xy_ground_truth.csv` (cement gland, eye, tail
response, local signal), this measures the fluorescence time-course in a small
disk at the annotated pixel location, straight from the C-channel TIFF, and
reports whether the calcium signal actually rises there (ΔF/F₀) — an automated
corroboration of the human annotation.

It also classifies each landmark as on the STIMULATED embryo (same side as the
poke/pressure) or the NEIGHBOR embryo, using the ground-truth poke side
(no PCA estimate needed).

Outputs:
  landmark_responses.csv   - one row per organ landmark with ΔF/F₀ + verdict
  landmark_responses.png   - corroboration rate + ΔF/F₀ by landmark class

Method
------
  F0   = 20th-percentile of the disk mean over sampled frames (quiet baseline)
  peak = max disk mean
  ΔF/F₀ = (peak − F0) / F0
  localized = disk is brighter than the whole-frame mean at the peak frame
  corroborated = ΔF/F₀ ≥ --dff-threshold (default 0.30)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tifffile as tiff

HERE = Path(__file__).resolve().parent
from provenance import csv_comment_header, record_run  # noqa: E402

DEFAULT_ROOT = "/Users/jdietz/Library/CloudStorage/Box-Box/Calcium videos"
ORGAN_CLASSES = ("cement_gland", "eye", "tail_response", "local")


def build_prefix_index(root: Path) -> dict:
    idx = {}
    for p in root.rglob("*_C-*.tif*"):
        idx.setdefault(p.stem.split("_")[0], p)
    return idx


def disk_stat(frame: np.ndarray, x: float, y: float, r: int,
              pct: float = 90.0) -> float:
    """Robust-peak brightness in a disk (high percentile, not mean).

    Organ landmarks are small bright structures; a disk mean is diluted by
    surrounding tissue/background, so we use a high percentile to capture the
    localized response while staying robust to single hot pixels.
    """
    h, w = frame.shape
    x0, x1 = max(0, int(x - r)), min(w, int(x + r) + 1)
    y0, y1 = max(0, int(y - r)), min(h, int(y + r) + 1)
    if x1 <= x0 or y1 <= y0:
        return float("nan")
    patch = frame[y0:y1, x0:x1]
    yy, xx = np.ogrid[y0:y1, x0:x1]
    mask = (xx - x) ** 2 + (yy - y) ** 2 <= r * r
    vals = patch[mask]
    return float(np.percentile(vals, pct)) if vals.size else float("nan")


def sample_indices(n: int, max_frames: int) -> list[int]:
    if n <= max_frames:
        return list(range(n))
    step = int(np.ceil(n / max_frames))
    return list(range(0, n, step))


def analyse_video(tif_path: Path, landmarks: pd.DataFrame, radius: int,
                  max_frames: int, disk_pct: float = 90.0) -> list[dict]:
    try:
        tf = tiff.TiffFile(tif_path)
    except Exception as e:  # noqa: BLE001
        print(f"  ! cannot open {tif_path.name}: {e}")
        return []
    with tf:
        n = len(tf.pages)
        idxs = sample_indices(n, max_frames)
        # accumulate per-landmark disk means + global means across sampled frames
        lm_list = landmarks.to_dict("records")
        courses = {i: [] for i in range(len(lm_list))}
        glob = []
        frame_times = []
        for fi in idxs:
            img = tf.pages[fi].asarray()
            img = np.asarray(img, dtype=np.float32)
            if img.ndim == 3:
                img = img.mean(axis=2)
            glob.append(float(np.percentile(img, disk_pct)))
            frame_times.append(fi)
            for k, lm in enumerate(lm_list):
                courses[k].append(disk_stat(img, lm["x"], lm["y"], radius,
                                            disk_pct))

    glob = np.asarray(glob)
    out = []
    for k, lm in enumerate(lm_list):
        c = np.asarray(courses[k], dtype=float)
        if np.all(~np.isfinite(c)):
            continue
        f0 = np.nanpercentile(c, 20)
        peak = np.nanmax(c)
        peak_i = int(np.nanargmax(c))
        dff = (peak - f0) / f0 if f0 > 0 else float("nan")
        localized = bool(c[peak_i] > glob[peak_i])
        out.append({
            "prefix": lm["prefix"],
            "video_stem": lm["video_stem"],
            "landmark_class": lm["landmark_class"],
            "landmark_raw": lm["landmark_raw"],
            "side": lm["side"],
            "embryo_role": lm["embryo_role"],
            "x": lm["x"], "y": lm["y"],
            "annot_frame": lm["frame"],
            "peak_frame": frame_times[peak_i],
            "F0": round(f0, 1),
            "peak": round(peak, 1),
            "dff": round(dff, 3) if np.isfinite(dff) else "",
            "localized": int(localized),
        })
    return out


def assign_roles(gt: pd.DataFrame) -> pd.DataFrame:
    """Tag each row with embryo_role = stimulated / neighbor / single."""
    gt = gt.copy()
    gt["embryo_role"] = "single"
    for prefix, sub in gt.groupby("prefix"):
        stim = sub[sub.landmark_class.isin(["poke", "pressure"])]
        if stim.empty:
            continue
        stim_side = stim.iloc[0]["side"]
        if stim_side == "single":
            continue
        for i, row in sub.iterrows():
            if row["side"] == "single":
                gt.at[i, "embryo_role"] = "single"
            elif row["side"] == stim_side:
                gt.at[i, "embryo_role"] = "stimulated"
            else:
                gt.at[i, "embryo_role"] = "neighbor"
    return gt


def render_summary(df: pd.DataFrame, out_png: Path, thr: float) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    classes = [c for c in ORGAN_CLASSES if c in df.landmark_class.unique()]

    ax = axes[0]
    rates, ns = [], []
    for c in classes:
        sub = df[df.landmark_class == c]
        rates.append(100.0 * (pd.to_numeric(sub.dff, errors="coerce") >= thr).mean())
        ns.append(len(sub))
    bars = ax.bar(classes, rates, color="#2a7a2a")
    for b, n, r in zip(bars, ns, rates):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1,
                f"{r:.0f}%\n(n={n})", ha="center", fontsize=9)
    ax.set_ylim(0, 115)
    ax.set_ylabel(f"% landmarks corroborated (ΔF/F₀ ≥ {thr})")
    ax.set_title("Organ-response landmarks corroborated by calcium signal")
    ax.tick_params(axis="x", rotation=20)

    ax = axes[1]
    data = [pd.to_numeric(df[df.landmark_class == c].dff, errors="coerce").dropna()
            for c in classes]
    ax.boxplot(data, tick_labels=classes, showfliers=False)
    for i, c in enumerate(classes, 1):
        ys = pd.to_numeric(df[df.landmark_class == c].dff, errors="coerce").dropna()
        ax.scatter(np.random.normal(i, 0.05, len(ys)), ys, s=18, alpha=0.6,
                   color="#1f77b4")
    ax.axhline(thr, color="#d62728", ls="--", lw=1)
    ax.set_ylabel("ΔF/F₀ at landmark")
    ax.set_title("Signal rise at each annotated landmark")
    ax.tick_params(axis="x", rotation=20)

    fig.suptitle("Manual organ-response annotations vs measured calcium signal",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_png, dpi=140)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ground-truth",
                    default=str(HERE / "analysis_results" / "xy_ground_truth.csv"))
    ap.add_argument("--root", default=DEFAULT_ROOT)
    ap.add_argument("--out-csv",
                    default=str(HERE / "analysis_results" / "landmark_responses.csv"))
    ap.add_argument("--out-png",
                    default=str(HERE / "analysis_results" / "landmark_responses.png"))
    ap.add_argument("--radius", type=int, default=45,
                    help="disk radius (px) around each landmark")
    ap.add_argument("--max-frames", type=int, default=60,
                    help="frames to sample per video")
    ap.add_argument("--dff-threshold", type=float, default=0.30)
    ap.add_argument("--disk-pct", type=float, default=90.0,
                    help="percentile within the disk used as the local "
                         "brightness (robust peak; avoids background dilution)")
    args = ap.parse_args()

    gt = pd.read_csv(args.ground_truth)
    gt = assign_roles(gt)
    organ = gt[gt.landmark_class.isin(ORGAN_CLASSES)].copy()
    if organ.empty:
        raise SystemExit("No organ landmarks found in ground truth.")

    idx = build_prefix_index(Path(args.root))
    rows = []
    for prefix, sub in organ.groupby("prefix"):
        tif = idx.get(prefix)
        if tif is None:
            print(f"  ! no TIFF for {prefix}; skipping {len(sub)} landmarks")
            continue
        print(f"[{prefix}] {len(sub)} organ landmark(s) — {tif.name}")
        rows.extend(analyse_video(tif, sub, args.radius, args.max_frames,
                                  args.disk_pct))

    df = pd.DataFrame(rows)
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    rec = record_run(
        "score_landmark_responses.py", out_csv,
        outputs=[out_csv, args.out_png],
        inputs={"ground_truth": args.ground_truth},
        extra={
            "n_landmarks": len(df),
            "dff_threshold": args.dff_threshold,
            "disk_pct": args.disk_pct,
            "radius_px": args.radius,
        },
    )
    with open(out_csv, "w", encoding="utf-8") as f:
        f.write(csv_comment_header(rec))
        df.to_csv(f, index=False)
    render_summary(df, Path(args.out_png), args.dff_threshold)

    print(f"\nWrote {out_csv} ({len(df)} landmarks)")
    dffn = pd.to_numeric(df.dff, errors="coerce")
    for c in ORGAN_CLASSES:
        s = df[df.landmark_class == c]
        if len(s):
            cor = (pd.to_numeric(s.dff, errors="coerce") >= args.dff_threshold).sum()
            print(f"  {c:14} {cor}/{len(s)} corroborated "
                  f"(median ΔF/F₀ {pd.to_numeric(s.dff, errors='coerce').median():.2f})")
    print(f"  Wrote {args.out_png}")


if __name__ == "__main__":
    main()

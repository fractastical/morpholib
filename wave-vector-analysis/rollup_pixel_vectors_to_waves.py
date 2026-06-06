#!/usr/bin/env python3
"""
Roll up dense per-pixel frame deltas into coherent WAVE events.

Input  : pixel_frame_deltas.csv  (from generate_pixel_brightness_vectors.py)
Output : wave_events.csv        (one row per wave)
         wave_front_frames.csv  (per wave, per frame: the moving front)
         wave_tracks.png        (centroid paths, colored by wave)
         wave_summary.png       (speed / duration / size overview)

A "wave" is defined operationally as a spatiotemporally coherent group of
bright-pixel motion:
  1. Within each frame, the matched bright pixels are spatially clustered
     (connected components within `--eps` px) into "fronts".
  2. Fronts are linked frame-to-frame (nearest centroid within `--link-px`)
     into wave tracks that persist over time.
  3. Each wave track is summarized: origin, path, duration, front speed
     (per-pixel motion) and propagation speed (centroid displacement / time).
"""

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree


def connected_components(points, eps):
    """Label points by connected components within `eps` (union-find)."""
    n = len(points)
    if n == 0:
        return np.zeros(0, dtype=int)
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    tree = cKDTree(points)
    for i, j in tree.query_pairs(eps):
        union(i, j)

    roots = np.array([find(i) for i in range(n)])
    _, labels = np.unique(roots, return_inverse=True)
    return labels


def load_matched_rows(csv_path):
    rows = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            if str(r.get("matched_from_last", "")).strip() not in ("True", "true", "1"):
                continue
            try:
                rows.append({
                    "frame_idx": int(float(r["frame_idx"])),
                    "time_s": float(r["time_s"]),
                    "x_curr": float(r["x_curr"]),
                    "y_curr": float(r["y_curr"]),
                    "dx": float(r["delta_x_px"]),
                    "dy": float(r["delta_y_px"]),
                    "speed": float(r["speed_px_per_s"]) if r["speed_px_per_s"] != "" else float("nan"),
                    "bright_raw": float(r["brightness_curr_raw"]),
                    "bright_lin": float(r["brightness_curr_linear"]),
                })
            except (ValueError, KeyError):
                continue
    return rows


def build_frame_fronts(rows, eps, min_pixels):
    """Cluster matched pixels within each frame into fronts."""
    by_frame = defaultdict(list)
    for r in rows:
        by_frame[r["frame_idx"]].append(r)

    fronts = []  # each: dict with frame_idx, time_s, centroid, stats
    for fi in sorted(by_frame):
        pts_rows = by_frame[fi]
        coords = np.array([[p["x_curr"], p["y_curr"]] for p in pts_rows])
        labels = connected_components(coords, eps)
        for lab in np.unique(labels):
            members = [pts_rows[k] for k in range(len(pts_rows)) if labels[k] == lab]
            if len(members) < min_pixels:
                continue
            xs = np.array([m["x_curr"] for m in members])
            ys = np.array([m["y_curr"] for m in members])
            dxs = np.array([m["dx"] for m in members])
            dys = np.array([m["dy"] for m in members])
            spd = np.array([m["speed"] for m in members])
            braw = np.array([m["bright_raw"] for m in members])
            wbraw = braw / braw.sum() if braw.sum() > 0 else np.ones_like(braw) / len(braw)
            fronts.append({
                "frame_idx": fi,
                "time_s": members[0]["time_s"],
                "x": float(np.average(xs, weights=wbraw)),
                "y": float(np.average(ys, weights=wbraw)),
                "n_pixels": len(members),
                "total_brightness": float(braw.sum()),
                "mean_brightness": float(braw.mean()),
                "mean_dx": float(np.mean(dxs)),
                "mean_dy": float(np.mean(dys)),
                "mean_speed": float(np.nanmean(spd)),
                "dir_deg": float(math.degrees(math.atan2(-np.mean(dys), np.mean(dxs)))),
            })
    return fronts


def link_fronts(fronts, link_px, max_gap):
    """Greedy link of fronts across frames into wave tracks."""
    by_frame = defaultdict(list)
    for fr in fronts:
        by_frame[fr["frame_idx"]].append(fr)

    next_wave_id = 0
    active = []  # list of dict: wave_id, last_frame, x, y
    for fr in fronts:
        fr["wave_id"] = None

    for fi in sorted(by_frame):
        frame_fronts = by_frame[fi]
        # candidate tracks still alive (within max_gap frames)
        cands = [a for a in active if fi - a["last_frame"] <= max_gap and a["last_frame"] < fi]
        used = set()
        # sort fronts by brightness so dominant fronts grab links first
        for fr in sorted(frame_fronts, key=lambda f: -f["total_brightness"]):
            best = None
            best_d = link_px
            for a in cands:
                if id(a) in used:
                    continue
                d = math.hypot(fr["x"] - a["x"], fr["y"] - a["y"])
                if d <= best_d:
                    best_d = d
                    best = a
            if best is None:
                wid = next_wave_id
                next_wave_id += 1
                active.append({"wave_id": wid, "last_frame": fi, "x": fr["x"], "y": fr["y"]})
            else:
                wid = best["wave_id"]
                best["last_frame"] = fi
                best["x"] = fr["x"]
                best["y"] = fr["y"]
                used.add(id(best))
            fr["wave_id"] = wid
    return next_wave_id


def summarize_waves(fronts):
    by_wave = defaultdict(list)
    for fr in fronts:
        by_wave[fr["wave_id"]].append(fr)

    waves = []
    for wid, frs in by_wave.items():
        frs = sorted(frs, key=lambda f: f["frame_idx"])
        t0, t1 = frs[0]["time_s"], frs[-1]["time_s"]
        duration = t1 - t0
        xs = [f["x"] for f in frs]
        ys = [f["y"] for f in frs]
        path_len = sum(
            math.hypot(xs[i] - xs[i - 1], ys[i] - ys[i - 1])
            for i in range(1, len(xs))
        )
        net_disp = math.hypot(xs[-1] - xs[0], ys[-1] - ys[0])
        prop_speed = net_disp / duration if duration > 0 else float("nan")
        path_speed = path_len / duration if duration > 0 else float("nan")
        waves.append({
            "wave_id": wid,
            "frame_start": frs[0]["frame_idx"],
            "frame_end": frs[-1]["frame_idx"],
            "t_start_s": round(t0, 3),
            "t_end_s": round(t1, 3),
            "duration_s": round(duration, 3),
            "n_frames": len(frs),
            "x_origin": round(xs[0], 1),
            "y_origin": round(ys[0], 1),
            "x_end": round(xs[-1], 1),
            "y_end": round(ys[-1], 1),
            "path_length_px": round(path_len, 1),
            "net_displacement_px": round(net_disp, 1),
            "net_direction_deg": round(
                math.degrees(math.atan2(-(ys[-1] - ys[0]), xs[-1] - xs[0])), 1),
            "propagation_speed_px_per_s": round(prop_speed, 3),
            "path_speed_px_per_s": round(path_speed, 3),
            "mean_front_speed_px_per_s": round(
                float(np.nanmean([f["mean_speed"] for f in frs])), 3),
            "peak_n_pixels": max(f["n_pixels"] for f in frs),
            "peak_total_brightness": round(max(f["total_brightness"] for f in frs), 1),
            "total_pixel_steps": sum(f["n_pixels"] for f in frs),
        })
    waves.sort(key=lambda w: -w["peak_total_brightness"])
    return waves


def write_csv(path, rows, fields):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})


def plot_wave_tracks(fronts, waves, out_png, top_n, img_shape=None):
    fig, ax = plt.subplots(figsize=(13, 6))
    by_wave = defaultdict(list)
    for fr in fronts:
        by_wave[fr["wave_id"]].append(fr)

    top_ids = [w["wave_id"] for w in waves[:top_n]]
    cmap = plt.get_cmap("turbo")
    for k, wid in enumerate(top_ids):
        frs = sorted(by_wave[wid], key=lambda f: f["frame_idx"])
        xs = [f["x"] for f in frs]
        ys = [f["y"] for f in frs]
        color = cmap(k / max(1, len(top_ids) - 1))
        sizes = [max(8, f["n_pixels"]) for f in frs]
        ax.plot(xs, ys, "-", color=color, lw=1.4, alpha=0.9)
        ax.scatter(xs, ys, s=sizes, color=color, alpha=0.5, edgecolors="none")
        ax.scatter([xs[0]], [ys[0]], marker="o", s=70, facecolors="none",
                   edgecolors=color, lw=2)
        ax.annotate(f"W{wid}", (xs[0], ys[0]), color=color, fontsize=8,
                    xytext=(4, 4), textcoords="offset points")
        if len(xs) > 1:
            ax.annotate("", xy=(xs[-1], ys[-1]), xytext=(xs[-2], ys[-2]),
                        arrowprops=dict(arrowstyle="->", color=color, lw=1.6))

    ax.set_facecolor("#0b0b0b")
    if img_shape is not None:
        ax.set_xlim(0, img_shape[1])
        ax.set_ylim(img_shape[0], 0)
    else:
        ax.invert_yaxis()
    ax.set_aspect("equal")
    ax.set_title(f"Wave tracks (top {len(top_ids)} by peak brightness) — "
                 f"open circle = origin, arrow = direction")
    ax.set_xlabel("x (px)")
    ax.set_ylabel("y (px)")
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def plot_wave_summary(waves, out_png):
    if not waves:
        return
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    dur = [w["duration_s"] for w in waves]
    prop = [w["propagation_speed_px_per_s"] for w in waves
            if np.isfinite(w["propagation_speed_px_per_s"])]
    front = [w["mean_front_speed_px_per_s"] for w in waves
             if np.isfinite(w["mean_front_speed_px_per_s"])]

    axes[0].hist(dur, bins=20, color="#2c7fb8")
    axes[0].set_title("Wave duration")
    axes[0].set_xlabel("seconds")
    axes[0].set_ylabel("# waves")

    axes[1].hist(prop, bins=20, color="#d95f0e")
    axes[1].set_title("Propagation speed (centroid)")
    axes[1].set_xlabel("px/s")

    axes[2].scatter(front, prop if len(prop) == len(front) else front,
                    alpha=0.6, color="#31a354")
    axes[2].set_title("Front speed vs propagation speed")
    axes[2].set_xlabel("mean front speed (px/s)")
    axes[2].set_ylabel("propagation speed (px/s)")

    fig.suptitle(f"Wave roll-up summary  (n={len(waves)} waves)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("deltas_csv", help="pixel_frame_deltas.csv")
    ap.add_argument("--output-dir", default=None,
                    help="default: <deltas_csv parent>/waves")
    ap.add_argument("--eps", type=float, default=25.0,
                    help="px radius to group pixels into a single front")
    ap.add_argument("--min-pixels", type=int, default=5,
                    help="min pixels for a front to count")
    ap.add_argument("--link-px", type=float, default=40.0,
                    help="max centroid jump to link a front across frames")
    ap.add_argument("--max-gap", type=int, default=2,
                    help="max frame gap a wave can bridge")
    ap.add_argument("--min-wave-frames", type=int, default=2,
                    help="drop waves shorter than this many frames")
    ap.add_argument("--top-n", type=int, default=20,
                    help="how many waves to draw in the track plot")
    ap.add_argument("--img-height", type=int, default=None)
    ap.add_argument("--img-width", type=int, default=None)
    args = ap.parse_args()

    csv_path = Path(args.deltas_csv)
    out_dir = Path(args.output_dir) if args.output_dir else csv_path.parent / "waves"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading matched vectors from {csv_path} …")
    rows = load_matched_rows(csv_path)
    print(f"  {len(rows)} matched pixel-steps")

    fronts = build_frame_fronts(rows, eps=args.eps, min_pixels=args.min_pixels)
    print(f"  {len(fronts)} per-frame fronts (eps={args.eps}, "
          f"min_pixels={args.min_pixels})")

    n_waves = link_fronts(fronts, link_px=args.link_px, max_gap=args.max_gap)
    waves = summarize_waves(fronts)
    waves = [w for w in waves if w["n_frames"] >= args.min_wave_frames]
    keep_ids = {w["wave_id"] for w in waves}
    fronts = [fr for fr in fronts if fr["wave_id"] in keep_ids]
    print(f"  {len(waves)} waves (>= {args.min_wave_frames} frames) "
          f"out of {n_waves} raw tracks")

    wave_fields = [
        "wave_id", "frame_start", "frame_end", "t_start_s", "t_end_s",
        "duration_s", "n_frames", "x_origin", "y_origin", "x_end", "y_end",
        "path_length_px", "net_displacement_px", "net_direction_deg",
        "propagation_speed_px_per_s", "path_speed_px_per_s",
        "mean_front_speed_px_per_s", "peak_n_pixels", "peak_total_brightness",
        "total_pixel_steps",
    ]
    front_fields = [
        "wave_id", "frame_idx", "time_s", "x", "y", "n_pixels",
        "total_brightness", "mean_brightness", "mean_dx", "mean_dy",
        "mean_speed", "dir_deg",
    ]
    write_csv(out_dir / "wave_events.csv", waves, wave_fields)
    write_csv(out_dir / "wave_front_frames.csv",
              sorted(fronts, key=lambda f: (f["wave_id"], f["frame_idx"])),
              front_fields)

    img_shape = None
    if args.img_height and args.img_width:
        img_shape = (args.img_height, args.img_width)
    plot_wave_tracks(fronts, waves, out_dir / "wave_tracks.png",
                     top_n=args.top_n, img_shape=img_shape)
    plot_wave_summary(waves, out_dir / "wave_summary.png")

    print(f"\nWrote:")
    print(f"  {out_dir/'wave_events.csv'}")
    print(f"  {out_dir/'wave_front_frames.csv'}")
    print(f"  {out_dir/'wave_tracks.png'}")
    print(f"  {out_dir/'wave_summary.png'}")
    if waves:
        top = waves[0]
        print(f"\nBrightest wave: W{top['wave_id']} | "
              f"t {top['t_start_s']}–{top['t_end_s']}s "
              f"({top['duration_s']}s, {top['n_frames']} frames) | "
              f"prop {top['propagation_speed_px_per_s']} px/s | "
              f"front {top['mean_front_speed_px_per_s']} px/s")


if __name__ == "__main__":
    main()

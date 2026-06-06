#!/usr/bin/env python3
"""
Frame-by-frame pixel brightness motion (strict t vs t-1 deltas).

Brightness is reduced to a single LINEAR per-pixel scale: every pixel's raw
intensity is mapped to [0, 1] using one global reference range (stack-wide
min/max by default, or the full dtype range). This makes brightness directly
comparable across pixels and across frames -- a dim frame stays dim, a bright
frame stays bright, rather than each frame being independently stretched.

For every frame after the first, each bright pixel in the *current* frame is
matched to the nearest bright pixel in the *immediately previous* frame. Each
row is exactly one 1-frame step: position_last -> position_curr.

Outputs CSV, PNG/PDF/MP4 with arrows drawn from last-frame position to the
current pixel (tail = where it was, head = where it is now), colored by the
linear brightness value.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tifffile as tiff
from matplotlib.colors import Normalize
from scipy.ndimage import maximum_filter
from scipy.spatial import cKDTree


def natural_key(s):
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split(r"(\d+)", s)]


def build_frame_mapping(folder_path):
    """Return sorted list of (path, page_idx, use_tifffile).

    Accepts either a directory of TIFFs or a single multi-page TIFF file.
    """
    p = Path(folder_path)
    paths = []
    if p.is_file() and p.suffix.lower() in (".tif", ".tiff"):
        paths = [str(p)]
    else:
        for root, _dirs, files in os.walk(folder_path):
            for f in files:
                if f.lower().endswith((".tif", ".tiff")):
                    paths.append(os.path.join(root, f))
    if not paths:
        raise RuntimeError(f"No TIFF files in {folder_path}")
    paths.sort(key=lambda p: natural_key(os.path.basename(p)))

    frame_mapping = []
    for path in paths:
        try:
            with tiff.TiffFile(path) as tf:
                n_pages = len(tf.pages)
            use_tifffile = True
        except Exception:
            n_pages = 1
            use_tifffile = False
        for page_idx in range(n_pages):
            frame_mapping.append((path, page_idx, use_tifffile))
    return frame_mapping


def read_tiff_page(path, page_idx=0, use_tifffile=True):
    if use_tifffile:
        with tiff.TiffFile(path) as tf:
            img = tf.asarray(key=page_idx)
            if not isinstance(img, np.ndarray):
                img = np.array(img)
            return img.copy(), False
    if page_idx > 0:
        raise ValueError("OpenCV only supports first page")
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None or img.size == 0:
        raise RuntimeError(f"Could not read {path}")
    return img, True


def to_grayscale_float(raw, is_bgr=False):
    """Return 2D float32 intensity in native scale (16-bit preserved)."""
    if raw.ndim == 3:
        if is_bgr:
            gray = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
        else:
            gray = cv2.cvtColor(raw, cv2.COLOR_RGB2GRAY)
    elif raw.ndim == 2:
        gray = raw
    else:
        raise ValueError(f"Unsupported shape {raw.shape}")
    return gray.astype(np.float32)


def dtype_max_for(raw):
    if raw.dtype == np.uint16:
        return 65535.0
    if raw.dtype == np.uint8:
        return 255.0
    return None


def compute_global_range(frame_mapping, n_use, use_dtype_range=False):
    """
    First pass: find one linear brightness reference for the whole stack.

    Returns (global_min, global_max). With use_dtype_range, returns the full
    container range (0 .. dtype max) so the scale is fixed regardless of content.
    """
    gmin = np.inf
    gmax = -np.inf
    dtype_hi = None

    for frame_idx in range(n_use):
        path, page_idx, use_tifffile = frame_mapping[frame_idx]
        raw, is_bgr = read_tiff_page(path, page_idx, use_tifffile)
        if dtype_hi is None:
            dtype_hi = dtype_max_for(raw)
        gray = to_grayscale_float(raw, is_bgr=is_bgr)
        del raw
        fmin = float(gray.min())
        fmax = float(gray.max())
        if fmin < gmin:
            gmin = fmin
        if fmax > gmax:
            gmax = fmax

    if use_dtype_range and dtype_hi is not None:
        return 0.0, float(dtype_hi)

    if not np.isfinite(gmin) or not np.isfinite(gmax) or gmax <= gmin:
        # Degenerate stack; fall back to dtype range or 0..1
        if dtype_hi is not None:
            return 0.0, float(dtype_hi)
        return 0.0, 1.0
    return gmin, gmax


def linearize(values, gmin, gmax):
    """Map raw intensities to a linear [0, 1] scale using the global range."""
    return np.clip((values - gmin) / (gmax - gmin + 1e-9), 0.0, 1.0)


def find_bright_points(
    gray_f32,
    gmin,
    gmax,
    linear_threshold=0.5,
    min_separation_px=3,
    max_points=4000,
):
    """
    Local maxima whose LINEAR brightness exceeds linear_threshold (0..1 on the
    global scale). Returns (N, 4) array: x, y, intensity_raw, intensity_linear.
    """
    blurred = cv2.GaussianBlur(gray_f32, (3, 3), 0)
    lin = linearize(blurred, gmin, gmax)

    bright = lin >= linear_threshold
    if not np.any(bright):
        return np.zeros((0, 4), dtype=np.float32)

    size = max(3, 2 * int(min_separation_px) + 1)
    local_max = maximum_filter(lin, size=size)
    peaks = bright & (lin >= local_max - 1e-6)

    ys, xs = np.where(peaks)
    if len(xs) == 0:
        return np.zeros((0, 4), dtype=np.float32)

    raw_vals = blurred[ys, xs]
    lin_vals = lin[ys, xs]

    if len(xs) > max_points:
        order = np.argsort(lin_vals)[::-1][:max_points]
        xs, ys, raw_vals, lin_vals = (
            xs[order], ys[order], raw_vals[order], lin_vals[order]
        )

    return np.column_stack([
        xs.astype(np.float32),
        ys.astype(np.float32),
        raw_vals.astype(np.float32),
        lin_vals.astype(np.float32),
    ])


def match_current_to_last_frame(last_pts, curr_pts, max_link_px):
    """
    For each bright pixel in the *current* frame, find the nearest pixel in the
    *last* frame (t-1) within max_link_px, using a KD-tree for speed. Brightest
    current pixels are matched first; each last-frame pixel is used at most once.
    Brightness fields are LINEAR (0..1).
    """
    rows = []
    if len(curr_pts) == 0:
        return rows

    order = np.argsort(curr_pts[:, 3])[::-1]  # brightest (linear) first

    tree = None
    if len(last_pts) > 0:
        tree = cKDTree(last_pts[:, :2])
    used_last = set()
    # Query enough neighbors that we can skip already-used last pixels.
    k = int(min(len(last_pts), 6)) if len(last_pts) > 0 else 0

    for ci in order:
        x_c, y_c, raw_c, lin_c = curr_pts[ci]
        row = {
            "x_curr": float(x_c),
            "y_curr": float(y_c),
            "brightness_curr_raw": float(raw_c),
            "brightness_curr_linear": float(lin_c),
            "matched_from_last": False,
            "x_last": "",
            "y_last": "",
            "brightness_last_raw": "",
            "brightness_last_linear": "",
            "delta_x_px": "",
            "delta_y_px": "",
            "displacement_px": "",
            "brightness_delta_linear": "",
        }

        if tree is None:
            rows.append(row)
            continue

        dists, idxs = tree.query([x_c, y_c], k=k,
                                 distance_upper_bound=max_link_px)
        dists = np.atleast_1d(dists)
        idxs = np.atleast_1d(idxs)
        best_li = None
        best_d = None
        for d, li in zip(dists, idxs):
            if not np.isfinite(d) or li >= len(last_pts):
                continue
            if li in used_last:
                continue
            best_li = int(li)
            best_d = float(d)
            break

        if best_li is None:
            rows.append(row)
            continue
        used_last.add(best_li)
        x_l, y_l, raw_l, lin_l = last_pts[best_li]
        dx = float(x_c - x_l)
        dy = float(y_c - y_l)
        row.update({
            "matched_from_last": True,
            "x_last": float(x_l),
            "y_last": float(y_l),
            "brightness_last_raw": float(raw_l),
            "brightness_last_linear": float(lin_l),
            "delta_x_px": dx,
            "delta_y_px": dy,
            "displacement_px": float(np.hypot(dx, dy)),
            "brightness_delta_linear": float(lin_c - lin_l),
        })
        rows.append(row)

    return rows


def process_folder(
    folder_path,
    fps,
    poke_frame_idx=0,
    linear_threshold=0.5,
    min_separation_px=3,
    max_points_per_frame=4000,
    max_link_px=8.0,
    max_frames=None,
    include_unmatched=True,
    use_dtype_range=False,
):
    """
    Two passes:
      1. Global linear brightness range across the stack.
      2. Per-frame bright-pixel detection + strict t vs t-1 matching.
    """
    frame_mapping = build_frame_mapping(folder_path)
    if poke_frame_idx < 0 or poke_frame_idx >= len(frame_mapping):
        raise ValueError(f"poke_frame_idx {poke_frame_idx} out of range")

    n_use = len(frame_mapping)
    if max_frames is not None:
        n_use = min(n_use, max_frames)

    print("Pass 1/2: computing global linear brightness range …")
    gmin, gmax = compute_global_range(
        frame_mapping, n_use, use_dtype_range=use_dtype_range
    )
    print(f"  global brightness range: [{gmin:.1f}, {gmax:.1f}] "
          f"({'dtype' if use_dtype_range else 'stack'} scale)")

    print("Pass 2/2: detecting bright pixels and frame-by-frame deltas …")
    all_rows = []
    display_frames = {}
    last_points = None

    for frame_idx in range(n_use):
        path, page_idx, use_tifffile = frame_mapping[frame_idx]
        raw, is_bgr = read_tiff_page(path, page_idx, use_tifffile)
        gray = to_grayscale_float(raw, is_bgr=is_bgr)
        del raw

        curr_points = find_bright_points(
            gray, gmin, gmax,
            linear_threshold=linear_threshold,
            min_separation_px=min_separation_px,
            max_points=max_points_per_frame,
        )

        # Display: GLOBAL linear scale (consistent across all frames)
        display_frames[frame_idx] = (
            linearize(gray, gmin, gmax) * 255
        ).astype(np.uint8)

        time_s = (frame_idx - poke_frame_idx) / float(fps)
        if Path(folder_path).is_file():
            rel_path = f"{os.path.basename(path)}#page{page_idx}"
        else:
            rel_path = os.path.relpath(path, folder_path)
            folder_name = os.path.basename(os.path.normpath(folder_path))
            if rel_path and not rel_path.startswith(folder_name):
                rel_path = f"{folder_name}/{rel_path}"

        if frame_idx > 0 and last_points is not None:
            dt_s = 1.0 / float(fps)
            frame_idx_last = frame_idx - 1
            time_s_last = (frame_idx_last - poke_frame_idx) / float(fps)
            matches = match_current_to_last_frame(
                last_points, curr_points, max_link_px
            )
            for m in matches:
                if not include_unmatched and not m["matched_from_last"]:
                    continue
                rec = {
                    "frame_idx": frame_idx,
                    "frame_idx_last": frame_idx_last,
                    "frame_delta": 1,
                    "time_s": time_s,
                    "time_s_last": time_s_last,
                    "dt_s": dt_s,
                    "filename": rel_path,
                    **m,
                }
                if m["matched_from_last"]:
                    rec["speed_px_per_s"] = m["displacement_px"] / dt_s
                    rec["angle_deg"] = float(
                        np.degrees(np.arctan2(-m["delta_y_px"], m["delta_x_px"]))
                    )
                else:
                    rec["speed_px_per_s"] = ""
                    rec["angle_deg"] = ""
                all_rows.append(rec)

        last_points = curr_points

        if frame_idx % 50 == 0:
            print(f"  frame {frame_idx + 1}/{n_use}: "
                  f"{len(curr_points)} bright pixels (linear ≥ {linear_threshold})")

    return all_rows, display_frames, (gmin, gmax)


def write_vectors_csv(rows, csv_path):
    if not rows:
        print("No vectors to write.")
        return
    fieldnames = list(rows[0].keys())
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    matched = sum(1 for r in rows if r.get("matched_from_last"))
    print(f"Wrote {len(rows)} rows ({matched} with 1-frame match) → {csv_path}")


def _matched_vectors(vectors_for_frame):
    return [v for v in vectors_for_frame if v.get("matched_from_last")]


def render_frame_overlay(
    display_gray,
    vectors_for_frame,
    output_path,
    title=None,
    arrow_scale=1.0,
    max_arrows=6000,
):
    """Current frame; arrows from (x_last, y_last) → (x_curr, y_curr)."""
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(display_gray, cmap="gray", origin="upper", vmin=0, vmax=255)

    sub = _matched_vectors(vectors_for_frame)
    if not sub:
        ax.set_title(title or "No 1-frame matches")
        ax.axis("off")
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
        return

    if len(sub) > max_arrows:
        scores = np.array([
            v["displacement_px"] * 0.5 + v["brightness_curr_linear"] for v in sub
        ])
        order = np.argsort(scores)[::-1][:max_arrows]
        sub = [sub[i] for i in order]

    x_last = np.array([v["x_last"] for v in sub])
    y_last = np.array([v["y_last"] for v in sub])
    u = np.array([v["delta_x_px"] for v in sub]) * arrow_scale
    vv = np.array([v["delta_y_px"] for v in sub]) * arrow_scale
    lin = np.array([v["brightness_curr_linear"] for v in sub])
    norm = Normalize(vmin=0.0, vmax=1.0)  # fixed linear scale

    ax.quiver(
        x_last, y_last, u, vv, lin,
        cmap="plasma", norm=norm,
        angles="xy", scale_units="xy", scale=1,
        width=0.0012, headwidth=2.2, headlength=2.8, headaxislength=2.2,
        alpha=0.85,
    )
    sm = plt.cm.ScalarMappable(cmap="plasma", norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Linear brightness (0–1, global scale)")

    ax.set_title(title or f"1-frame delta ({len(sub)} vectors, last → current)")
    ax.set_aspect("equal")
    ax.axis("off")
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def render_summary_pdf(rows, display_frames, pdf_path, poke_frame_idx, fps,
                       sample_every=5, arrow_scale=1.0):
    from matplotlib.backends.backend_pdf import PdfPages

    by_frame = {}
    for r in rows:
        if not r.get("matched_from_last"):
            continue
        by_frame.setdefault(r["frame_idx"], []).append(r)

    frame_indices = sorted(by_frame.keys())
    if not frame_indices:
        print("No matched 1-frame vectors for PDF.")
        return
    if sample_every > 1:
        frame_indices = frame_indices[::sample_every]

    with PdfPages(pdf_path) as pdf:
        for fi in frame_indices:
            if fi not in display_frames:
                continue
            fig, ax = plt.subplots(figsize=(11, 8.5))
            ax.imshow(display_frames[fi], cmap="gray", origin="upper", vmin=0, vmax=255)
            vecs = by_frame[fi]
            x_last = np.array([v["x_last"] for v in vecs])
            y_last = np.array([v["y_last"] for v in vecs])
            u = np.array([v["delta_x_px"] for v in vecs]) * arrow_scale
            vv = np.array([v["delta_y_px"] for v in vecs]) * arrow_scale
            lin = np.array([v["brightness_curr_linear"] for v in vecs])
            norm = Normalize(vmin=0.0, vmax=1.0)
            ax.quiver(
                x_last, y_last, u, vv, lin,
                cmap="plasma", norm=norm,
                angles="xy", scale_units="xy", scale=1,
                width=0.0010, headwidth=2.0, headlength=2.5, alpha=0.8,
            )
            t_s = (fi - poke_frame_idx) / fps
            ax.set_title(
                f"t = {t_s:.2f} s  |  {len(vecs)} vectors  "
                f"(Δ from frame {fi - 1} → {fi})"
            )
            ax.axis("off")
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
    print(f"Wrote PDF → {pdf_path}")


def render_mp4(rows, display_frames, mp4_path, fps_out, poke_frame_idx, fps):
    by_frame = {}
    for r in rows:
        if r.get("matched_from_last"):
            by_frame.setdefault(r["frame_idx"], []).append(r)

    indices = sorted(display_frames.keys())
    if not indices:
        return

    h, w = display_frames[indices[0]].shape
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(mp4_path), fourcc, fps_out, (w, h))

    for fi in indices:
        bgr = cv2.cvtColor(display_frames[fi], cv2.COLOR_GRAY2BGR)
        vecs = by_frame.get(fi, [])
        for v in vecs:
            p0 = (int(round(v["x_last"])), int(round(v["y_last"])))
            p1 = (int(round(v["x_curr"])), int(round(v["y_curr"])))
            t = float(np.clip(v["brightness_curr_linear"], 0, 1))
            color = (int(255 * (1 - t)), int(128 * t), int(255 * t))
            cv2.arrowedLine(bgr, p0, p1, color, 1, tipLength=0.35)
        t_s = (fi - poke_frame_idx) / fps
        cv2.putText(
            bgr, f"t={t_s:.1f}s Δt=1f n={len(vecs)}", (8, 22),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA,
        )
        writer.write(bgr)
    writer.release()
    print(f"Wrote MP4 → {mp4_path}")


def render_speed_summary(rows, output_path, poke_frame_idx, fps):
    """
    Wave-motion speed overview: distribution of per-pixel speeds and mean speed
    over time (speed = 1-frame displacement / dt).
    """
    matched = [r for r in rows if r.get("matched_from_last")]
    if not matched:
        print("No matched vectors; skipping speed summary.")
        return

    speeds = np.array([r["speed_px_per_s"] for r in matched
                       if isinstance(r["speed_px_per_s"], (int, float))], dtype=float)
    times = np.array([r["time_s"] for r in matched], dtype=float)
    spd_all = np.array([r["speed_px_per_s"] for r in matched], dtype=float)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].hist(speeds[np.isfinite(speeds)], bins=40, color="#1f77b4", alpha=0.85)
    axes[0].set_xlabel("Speed (px/s)")
    axes[0].set_ylabel("Count of pixel-steps")
    axes[0].set_title("Distribution of wave/pixel speeds")
    if speeds.size:
        axes[0].axvline(np.median(speeds), color="red", ls="--",
                        label=f"median={np.median(speeds):.2f} px/s")
        axes[0].legend()

    # Mean speed over time
    order = np.argsort(times)
    t_sorted = times[order]
    s_sorted = spd_all[order]
    uniq_t = np.unique(t_sorted)
    mean_speed = np.array([
        np.nanmean(s_sorted[t_sorted == tt]) for tt in uniq_t
    ])
    axes[1].plot(uniq_t, mean_speed, color="#d62728", lw=1.2)
    axes[1].axvline(0, color="gray", ls="--", alpha=0.6, label="poke (t=0)")
    axes[1].set_xlabel("Time (s, relative to poke)")
    axes[1].set_ylabel("Mean pixel speed (px/s)")
    axes[1].set_title("Mean wave speed over time")
    axes[1].legend()

    fig.suptitle(
        f"Pixel wave speed summary  (n={len(matched)} matched 1-frame vectors)",
        fontsize=13,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Wrote speed summary → {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Per-frame pixel brightness deltas on a global LINEAR scale: each "
            "bright pixel in frame t is matched to the nearest pixel in t-1."
        ),
    )
    parser.add_argument("folder", help="Folder of TIFF frames (recursive)")
    parser.add_argument("poke_frame", type=int, help="0-based poke frame (t=0)")
    parser.add_argument("--fps", type=float, default=1.0)
    parser.add_argument("--output-dir", default="pixel_brightness_vectors")
    parser.add_argument(
        "--linear-threshold",
        type=float,
        default=0.5,
        help="Bright cutoff on the 0..1 global linear scale (default 0.5)",
    )
    parser.add_argument(
        "--dtype-range",
        action="store_true",
        help="Use full container range (0..65535/255) instead of stack min/max",
    )
    parser.add_argument("--max-points", type=int, default=4000)
    parser.add_argument("--max-link-px", type=float, default=8.0)
    parser.add_argument("--min-separation", type=int, default=3)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument(
        "--omit-unmatched",
        action="store_true",
        help="Do not write rows for current pixels with no match in t-1",
    )
    parser.add_argument("--png-every", type=int, default=10)
    parser.add_argument(
        "--arrow-scale", type=float, default=1.0,
        help="Magnify the tiny 1-frame vectors for visibility (e.g. 8 or 12)",
    )
    parser.add_argument("--pdf", action="store_true")
    parser.add_argument("--mp4", action="store_true")
    parser.add_argument("--mp4-fps", type=float, default=5.0)

    args = parser.parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Processing {args.folder} (strict 1-frame deltas, linear scale) …")
    rows, display_frames, (gmin, gmax) = process_folder(
        args.folder,
        fps=args.fps,
        poke_frame_idx=args.poke_frame,
        linear_threshold=args.linear_threshold,
        min_separation_px=args.min_separation,
        max_points_per_frame=args.max_points,
        max_link_px=args.max_link_px,
        max_frames=args.max_frames,
        include_unmatched=not args.omit_unmatched,
        use_dtype_range=args.dtype_range,
    )

    csv_path = out_dir / "pixel_frame_deltas.csv"
    write_vectors_csv(rows, csv_path)

    # Record the linear scale used, for provenance
    with open(out_dir / "linear_scale.txt", "w") as f:
        f.write(f"global_min={gmin}\nglobal_max={gmax}\n")
        f.write(f"scale={'dtype' if args.dtype_range else 'stack'}\n")
        f.write(f"linear_threshold={args.linear_threshold}\n")

    by_frame = {}
    for r in rows:
        by_frame.setdefault(r["frame_idx"], []).append(r)

    if args.png_every > 0:
        png_dir = out_dir / "frames"
        png_dir.mkdir(exist_ok=True)
        for fi in sorted(display_frames.keys()):
            if fi == 0 or fi % args.png_every != 0:
                continue
            vecs = by_frame.get(fi, [])
            t_s = (fi - args.poke_frame) / args.fps
            n_m = len(_matched_vectors(vecs))
            render_frame_overlay(
                display_frames[fi],
                vecs,
                png_dir / f"delta_frame_{fi:05d}_t{t_s:+.2f}s.png",
                title=f"t = {t_s:.2f} s | {n_m} matched 1-frame deltas",
                arrow_scale=args.arrow_scale,
            )
        print(f"PNG overlays → {png_dir}/")

    if args.pdf:
        render_summary_pdf(
            rows, display_frames,
            out_dir / "pixel_frame_deltas.pdf",
            args.poke_frame, args.fps,
            sample_every=max(1, args.png_every),
            arrow_scale=args.arrow_scale,
        )

    if args.mp4:
        render_mp4(
            rows, display_frames,
            out_dir / "pixel_frame_deltas.mp4",
            args.mp4_fps, args.poke_frame, args.fps,
        )

    # Speed summary (wave motion) — always written when there are matches
    render_speed_summary(
        rows, out_dir / "pixel_speed_summary.png",
        args.poke_frame, args.fps,
    )

    matched = sum(1 for r in rows if r.get("matched_from_last"))
    print(
        f"Done. {matched} matched 1-frame deltas, "
        f"{len(rows) - matched} unmatched current pixels. "
        f"Linear scale [{gmin:.1f}, {gmax:.1f}]."
    )


if __name__ == "__main__":
    main()

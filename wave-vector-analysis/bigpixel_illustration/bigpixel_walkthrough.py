#!/usr/bin/env python3
"""
Big-pixel walkthrough of the Ca2+ wave-vector approach.

This is a deliberately TINY, fully transparent version of the real pipeline
(`generate_pixel_brightness_vectors.py` + `rollup_pixel_vectors_to_waves.py`).
Instead of running on thousands of native camera pixels, it coarsens each frame
into a handful of large ~50 px "big pixels" so that every number in every step
is small enough to read off the figure by eye.

Every computation here mirrors the production pipeline exactly:

  1. Linear brightness   : raw intensity -> [0, 1] using one global stack range.
  2. F0 baseline         : per big pixel, F0 = low percentile of its time course.
  3. dF/F0 activation     : (F - F0) / F0, the normalized activation map.
  4. Motion vectors      : each bright big pixel in frame t matched to the
                           nearest bright big pixel in t-1 (dx, dy, speed, angle).
  5. Wave front + speed  : bright big pixels per frame -> a front; track the
                           front centroid over time -> propagation speed/direction.

By default it builds a synthetic two-embryo scene with a poke and an
injury-style Ca2+ wave so the document is self-contained and reproducible.
Point it at a real TIFF stack with --tiff to coarsen real data instead.

Outputs (into --output-dir):
  fig1_raw_to_bigpixels.png   raw frame -> big-pixel grid with mean values
  fig2_f0_baseline.png        per-big-pixel F0 map + one big pixel's time course
  fig3_dff_maps.png           dF/F0 activation maps over time
  fig4a_vectors_perframe.png  big-pixel motion vectors (t-1 -> t)
  fig4b_propagation_field.png wave propagation-direction field
  fig5a_grouping.png          bright big pixels grouped into fronts
  fig5b_centroid_path.png     wave-front centroid track over time
  fig5c_prop_speed.png        wave propagation speed over time
  bigpixel_walkthrough.pdf    all panels in one document (one panel per slide)
  bigpixel_numbers.txt        the exact numbers used in the worked example
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle


# --------------------------------------------------------------------------- #
# 0. Input frames: synthetic scene (default) or a real TIFF stack
# --------------------------------------------------------------------------- #
def make_synthetic_stack(
    height=200,
    width=400,
    n_frames=16,
    fps=2.0,
    poke_frame=2,
):
    """
    Two embryo bodies side by side. A poke at the far-left edge triggers one
    coherent Ca2+ wave that sweeps left -> right across both bodies -- an
    idealized injury wave propagating from the poked embryo into its neighbor
    (cf. Tung 2024). A single, moderately slow, thick front means every big
    pixel lights up smoothly for a few frames as the wave passes, which keeps
    the teaching figures clean.

    Returns float32 stack of shape (n_frames, height, width) in "camera" units.
    """
    # Coordinate grids. NOTE: NumPy 2.2.x can mis-fire its in-place
    # "temporary elision" optimization on large data-owning arrays in function
    # scope, so an expression like `xx - front` silently overwrites `xx`.
    # Taking a non-owning view (`[:, :]`) disables elision for these arrays and
    # makes all the arithmetic below behave (verified for this NumPy version).
    yy, xx = np.mgrid[0:height, 0:width]
    xx = xx.astype(np.float64)[:, :]
    yy = yy.astype(np.float64)[:, :]

    # Embryo bodies (gentle baseline glow so F0 is non-zero everywhere on tissue)
    cxA, cyA, rA = width * 0.30, height * 0.5, height * 0.46
    cxB, cyB, rB = width * 0.70, height * 0.5, height * 0.46
    bodyA = np.exp(-(((xx - cxA) ** 2 + (yy - cyA) ** 2) / (2 * (rA * 0.8) ** 2)))
    bodyB = np.exp(-(((xx - cxB) ** 2 + (yy - cyB) ** 2) / (2 * (rB * 0.8) ** 2)))
    tissue = np.clip(bodyA + bodyB, 0, 1)

    baseline = 400.0          # dark resting fluorescence
    body_gain = 200.0         # resting tissue glow
    wave_gain = 3500.0        # bright Ca2+ band amplitude
    band_sigma = 30.0         # thickness of the propagating front (px)
    wave_speed = 50.0         # px / frame ~ one big pixel per frame (clear vectors)

    # poke site at the far-left edge; the wavefront is the set of pixels whose
    # x-distance from the poke equals wave_speed * (t - poke_frame).
    poke_x, poke_y = width * 0.06, height * 0.5

    def add_band(frame, gain, band):
        """frame += gain * band * tissue via a fresh buffer (avoids a numpy
        temporary-elision bug that otherwise corrupts the named arrays)."""
        term = np.empty_like(frame)
        np.multiply(band, tissue, out=term)
        term *= gain
        frame += term

    stack = np.zeros((n_frames, height, width), dtype=np.float32)
    front_x = []

    for t in range(n_frames):
        frame = (baseline + body_gain * tissue).astype(np.float64)
        dt = t - poke_frame
        if dt >= 0:
            front = poke_x + wave_speed * dt           # current front x-position
            band = np.exp(-((xx - front) ** 2) / (2 * band_sigma ** 2))
            add_band(frame, wave_gain, band)
            front_x.append((t, front))
        frame += np.random.default_rng(t).normal(0, 22.0, size=frame.shape)
        stack[t] = np.clip(frame, 0, None)

    meta = dict(fps=fps, poke_frame=poke_frame, poke_xy=(poke_x, poke_y),
                front_x=front_x, wave_speed=wave_speed)
    return stack, meta


def _list_pages(tiff_path):
    """Return [(file_path, page_idx), ...] across a file or a folder of TIFFs."""
    import tifffile as tiff

    p = Path(tiff_path)
    files = [p] if p.is_file() else (sorted(p.glob("*.tif")) + sorted(p.glob("*.tiff")))
    pages = []
    for fp in files:
        with tiff.TiffFile(fp) as tf:
            for k in range(len(tf.pages)):
                pages.append((fp, k))
    if not pages:
        raise RuntimeError(f"No TIFF pages found in {tiff_path}")
    return pages


def load_tiff_stack(tiff_path, frame_start=0, frame_stride=1, max_frames=None,
                    autocrop=True, crop_pad=15, crop_pct=80.0):
    """
    Load selected frames of a (possibly huge) multi-page TIFF into a float32
    stack -- only the requested pages are read into memory.

    frame_start / frame_stride / max_frames pick a sparse temporal window so
    the wave actually moves between sampled frames; autocrop trims the dark
    border to the tissue bounding box so the big pixels land on the embryos.
    """
    import tifffile as tiff

    pages = _list_pages(tiff_path)
    sel = list(range(frame_start, len(pages), max(1, frame_stride)))
    if max_frames:
        sel = sel[:max_frames]
    if not sel:
        raise RuntimeError("No frames selected (check --frame-start/--stride).")

    frames = []
    for idx in sel:
        fp, k = pages[idx]
        with tiff.TiffFile(fp) as tf:
            img = np.asarray(tf.pages[k].asarray(), dtype=np.float32)
        if img.ndim == 3:
            img = img.mean(axis=2)
        frames.append(img)

    h = min(f.shape[0] for f in frames)
    w = min(f.shape[1] for f in frames)
    stack = np.stack([f[:h, :w] for f in frames]).astype(np.float32)

    if autocrop:
        # Use a high percentile of the max-projection so the threshold sits on
        # tissue, not background noise -- otherwise the bbox spans the whole
        # (mostly dark) frame and the embryos end up only a few big pixels tall.
        proj = stack.max(axis=0)
        thr = np.percentile(proj, crop_pct)
        ys, xs = np.where(proj > thr)
        if len(xs) > 0:
            y0 = max(0, ys.min() - crop_pad); y1 = min(h, ys.max() + crop_pad)
            x0 = max(0, xs.min() - crop_pad); x1 = min(w, xs.max() + crop_pad)
            stack = stack[:, y0:y1, x0:x1]
    return stack


# --------------------------------------------------------------------------- #
# 1. Big pixels: coarsen each frame into ~big_px x big_px cells
# --------------------------------------------------------------------------- #
def coarsen_to_bigpixels(stack, big_px):
    """
    Block-average the native frames into big pixels.

    Returns:
      big      (T, gh, gw) mean intensity per big pixel per frame
      centers  (gh, gw, 2) the (x, y) native-pixel center of each big pixel
    """
    T, H, W = stack.shape
    gh, gw = H // big_px, W // big_px
    Hc, Wc = gh * big_px, gw * big_px  # crop to a whole number of big pixels
    cropped = stack[:, :Hc, :Wc]
    big = cropped.reshape(T, gh, big_px, gw, big_px).mean(axis=(2, 4))

    ys = (np.arange(gh) + 0.5) * big_px
    xs = (np.arange(gw) + 0.5) * big_px
    cx, cy = np.meshgrid(xs, ys)
    centers = np.stack([cx, cy], axis=-1)
    return big.astype(np.float32), centers


# --------------------------------------------------------------------------- #
# 2-3. Linear brightness, F0 baseline, and dF/F0  (same defs as the pipeline)
# --------------------------------------------------------------------------- #
def linearize(values, gmin, gmax):
    """Map raw intensity to a global linear [0, 1] scale (pipeline eq.)."""
    return np.clip((values - gmin) / (gmax - gmin + 1e-9), 0.0, 1.0)


def compute_f0(big, percentile=20.0):
    """Per big pixel, F0 = low percentile of its own intensity time course."""
    return np.percentile(big, percentile, axis=0)


def compute_dff(big, f0):
    """dF/F0 = (F - F0) / F0, the normalized activation, per big pixel per frame."""
    return (big - f0[None]) / np.where(f0 > 1e-9, f0, np.nan)[None]


# --------------------------------------------------------------------------- #
# 4. Motion vectors: bright big pixel in t matched to nearest in t-1
# --------------------------------------------------------------------------- #
def bright_bigpixels(dff_frame, centers, thresh):
    """Return centers (x, y) and dff of big pixels with dF/F0 >= thresh."""
    mask = dff_frame >= thresh
    pts = centers[mask]                # (n, 2)
    vals = dff_frame[mask]             # (n,)
    return pts, vals


def match_to_previous(curr_pts, curr_vals, prev_pts, prev_vals, max_link_px):
    """
    For each bright big pixel in the current frame, link it to the nearest
    bright big pixel in the previous frame within max_link_px. Brightest first,
    each previous big pixel used at most once -- identical policy to the
    full-resolution pipeline, just with far fewer points.
    """
    vectors = []
    if len(curr_pts) == 0 or len(prev_pts) == 0:
        return vectors
    order = np.argsort(curr_vals)[::-1]  # brightest current first
    used = set()
    for ci in order:
        x_c, y_c = curr_pts[ci]
        d = np.hypot(prev_pts[:, 0] - x_c, prev_pts[:, 1] - y_c)
        cand = np.argsort(d)
        chosen = None
        for pi in cand:
            if d[pi] > max_link_px:
                break
            if pi in used:
                continue
            chosen = pi
            break
        if chosen is None:
            continue
        used.add(chosen)
        x_p, y_p = prev_pts[chosen]
        dx, dy = x_c - x_p, y_c - y_p
        vectors.append(dict(
            x_last=x_p, y_last=y_p, x_curr=x_c, y_curr=y_c,
            dx=dx, dy=dy,
            displacement_px=float(np.hypot(dx, dy)),
            brightness=float(curr_vals[ci]),
            angle_deg=float(np.degrees(np.arctan2(-dy, dx))),
        ))
    return vectors


def activation_time_field(dff, fps, poke_frame, thresh):
    """
    For every big pixel, the time it first reaches activation (dF/F0 >= thresh)
    -- i.e. when the wave *arrives* at that cell. The spatial gradient of this
    arrival-time map gives the local wave propagation direction (early -> late)
    and speed (1 / |grad t|). This is the clean, big-pixel view of "which way
    the wave travelled", complementing the per-frame t-1 -> t matching.
    """
    T, gh, gw = dff.shape
    d = np.nan_to_num(dff)
    t_arrival = np.full((gh, gw), np.nan)
    for j in range(gh):
        for i in range(gw):
            hits = np.where(d[:, j, i] >= thresh)[0]
            if len(hits):
                t_arrival[j, i] = (hits[0] - poke_frame) / fps
    return t_arrival


# --------------------------------------------------------------------------- #
# 5. Wave front: brightness-weighted centroid of bright big pixels per frame
# --------------------------------------------------------------------------- #
def wave_front_centroids(dff, centers, thresh, fps, poke_frame, min_pixels=3):
    """
    Per frame, take all bright big pixels (dF/F0 >= thresh) as one front and
    compute its brightness-weighted centroid. Tracking that centroid over time
    gives the wave's propagation path, speed, and direction. Frames with fewer
    than `min_pixels` bright cells are skipped (too sparse to be a real front).
    """
    T = dff.shape[0]
    fronts = []
    for t in range(T):
        pts, vals = bright_bigpixels(dff[t], centers, thresh)
        if len(pts) < min_pixels:
            continue
        w = vals / vals.sum()
        cx = float((pts[:, 0] * w).sum())
        cy = float((pts[:, 1] * w).sum())
        fronts.append(dict(
            frame=t,
            time_s=(t - poke_frame) / fps,
            x=cx, y=cy,
            n_pixels=int(len(pts)),
            total_activation=float(vals.sum()),
        ))
    # per-step propagation speed/direction
    for i in range(1, len(fronts)):
        a, b = fronts[i - 1], fronts[i]
        dt = b["time_s"] - a["time_s"]
        dx, dy = b["x"] - a["x"], b["y"] - a["y"]
        b["step_disp_px"] = float(np.hypot(dx, dy))
        b["prop_speed_px_per_s"] = float(np.hypot(dx, dy) / dt) if dt > 0 else float("nan")
        b["dir_deg"] = float(np.degrees(np.arctan2(-dy, dx)))
    return fronts


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def _panel_figsize(gh, gw, base_w=13.0, extra_h=1.9, min_h=3.2, max_h=9.0):
    """Figure size for a single full-slide image panel, sized to the big-pixel
    grid's aspect ratio so the data band fills the slide instead of floating in
    whitespace (the embryo strips are very wide and short)."""
    h = base_w * (gh / gw) + extra_h
    return (base_w, float(np.clip(h, min_h, max_h)))


def _annotate_grid(ax, big_frame, centers, big_px, fmt="{:.0f}", color="w",
                   max_labels=140):
    """Draw the big-pixel grid; annotate cell values only when the grid is
    small enough to stay legible (font auto-shrinks with grid size)."""
    gh, gw = big_frame.shape
    n = gh * gw
    if n <= max_labels:
        fs = 7 if n <= 48 else (5.5 if n <= 90 else 4.5)
        for j in range(gh):
            for i in range(gw):
                x, y = centers[j, i]
                ax.text(x, y, fmt.format(big_frame[j, i]), ha="center",
                        va="center", fontsize=fs, color=color)
    for j in range(gh + 1):
        ax.axhline(j * big_px, color="cyan", lw=0.4, alpha=0.5)
    for i in range(gw + 1):
        ax.axvline(i * big_px, color="cyan", lw=0.4, alpha=0.5)


def fig_raw_to_bigpixels(stack, big, centers, big_px, frame_idx, gmin, gmax, out):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    H, W = stack.shape[1:]

    axes[0].imshow(stack[frame_idx], cmap="magma", vmin=gmin, vmax=gmax)
    axes[0].set_title(f"(a) Raw frame  t-index={frame_idx}\n{W}x{H} native pixels")
    axes[0].set_xlabel("x (px)"); axes[0].set_ylabel("y (px)")

    gh, gw = big[frame_idx].shape
    axes[1].imshow(big[frame_idx], cmap="magma", vmin=gmin, vmax=gmax,
                   extent=[0, gw * big_px, gh * big_px, 0], interpolation="nearest")
    _annotate_grid(axes[1], big[frame_idx], centers, big_px)
    axes[1].set_title(f"(b) Big pixels: {big_px}x{big_px}-px blocks -> {gw}x{gh} grid\n"
                      f"each value = mean raw intensity of its block")
    axes[1].set_xlabel("x (px)"); axes[1].set_ylabel("y (px)")
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def fig_f0(big, f0, centers, big_px, fps, poke_frame, probe_ji, out):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    gh, gw = f0.shape

    im = axes[0].imshow(f0, cmap="viridis",
                        extent=[0, gw * big_px, gh * big_px, 0], interpolation="nearest")
    _annotate_grid(axes[0], f0, centers, big_px, color="w")
    pj, pi = probe_ji
    px, py = centers[pj, pi]
    axes[0].add_patch(Rectangle((pi * big_px, pj * big_px), big_px, big_px,
                                fill=False, edgecolor="red", lw=2.5))
    axes[0].set_title("(a) F0 baseline per big pixel\n"
                      "F0 = 20th percentile of each big pixel's time course")
    axes[0].set_xlabel("x (px)"); axes[0].set_ylabel("y (px)")
    fig.colorbar(im, ax=axes[0], fraction=0.035, pad=0.02, label="F0 (raw units)")

    T = big.shape[0]
    t_axis = (np.arange(T) - poke_frame) / fps
    course = big[:, pj, pi]
    f0v = f0[pj, pi]
    axes[1].plot(t_axis, course, "-o", color="#1f77b4", ms=3, label="F(t) big pixel")
    axes[1].axhline(f0v, color="red", ls="--", label=f"F0 = {f0v:.0f}")
    axes[1].axvline(0, color="gray", ls=":", alpha=0.7, label="poke (t=0)")
    axes[1].set_title(f"(b) Time course of the red big pixel (row {pj}, col {pi})")
    axes[1].set_xlabel("time (s, rel. poke)"); axes[1].set_ylabel("intensity (raw)")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def fig_dff(dff, centers, big_px, fps, poke_frame, frame_list, thresh, out):
    n = len(frame_list)
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 4.4))
    if n == 1:
        axes = [axes]
    gh, gw = dff.shape[1:]
    vmax = float(np.nanpercentile(dff, 99))
    norm = Normalize(vmin=0, vmax=max(vmax, thresh * 2))
    for ax, t in zip(axes, frame_list):
        im = ax.imshow(np.nan_to_num(dff[t]), cmap="inferno", norm=norm,
                       extent=[0, gw * big_px, gh * big_px, 0], interpolation="nearest")
        _annotate_grid(ax, np.nan_to_num(dff[t]), centers, big_px, fmt="{:.1f}")
        # outline bright (above threshold) big pixels
        for j in range(gh):
            for i in range(gw):
                if np.nan_to_num(dff[t])[j, i] >= thresh:
                    ax.add_patch(Rectangle((i * big_px, j * big_px), big_px, big_px,
                                           fill=False, edgecolor="lime", lw=2))
        ax.set_title(f"t = {(t - poke_frame) / fps:+.1f}s\n(green = dF/F0 >= {thresh})")
        ax.set_xticks([]); ax.set_yticks([])
    fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02, label="dF/F0")
    fig.suptitle("dF/F0 activation maps  (normalized brightness rise above baseline)",
                 fontsize=13)
    fig.savefig(out, dpi=140)
    plt.close(fig)


def fig_vectors_perframe(dff, centers, big_px, fps, poke_frame, frame_pairs, thresh,
                         max_link_px, out, arrow_scale=1.0):
    """Step 5(a), its own slide: the faithful per-frame t-1 -> t nearest-match
    vectors (the production motion-vector rule)."""
    gh, gw = dff.shape[1:]
    fig, ax = plt.subplots(figsize=_panel_figsize(gh, gw))

    # pick the frame with the most genuinely-moving matches to display
    def n_moving_at(tt):
        pp, pv = bright_bigpixels(np.nan_to_num(dff[tt - 1]), centers, thresh)
        cp, cv = bright_bigpixels(np.nan_to_num(dff[tt]), centers, thresh)
        return sum(1 for v in match_to_previous(cp, cv, pp, pv, max_link_px)
                   if v["displacement_px"] > 1e-6)
    cand = frame_pairs or list(range(1, dff.shape[0]))
    t = max(cand, key=n_moving_at)
    ax.imshow(np.nan_to_num(dff[t]), cmap="gray",
              extent=[0, gw * big_px, gh * big_px, 0], interpolation="nearest",
              alpha=0.8)
    prev_pts, prev_vals = bright_bigpixels(np.nan_to_num(dff[t - 1]), centers, thresh)
    curr_pts, curr_vals = bright_bigpixels(np.nan_to_num(dff[t]), centers, thresh)
    vecs = match_to_previous(curr_pts, curr_vals, prev_pts, prev_vals, max_link_px)
    if len(prev_pts):
        ax.scatter(prev_pts[:, 0], prev_pts[:, 1], s=44, facecolors="none",
                   edgecolors="deepskyblue", lw=1.4, label="bright t-1")
    if len(curr_pts):
        ax.scatter(curr_pts[:, 0], curr_pts[:, 1], s=20, c="orange", label="bright t")
    n_moving = 0
    for v in vecs:
        if v["displacement_px"] < 1e-6:
            continue
        n_moving += 1
        ax.annotate("", xy=(v["x_last"] + arrow_scale * v["dx"],
                            v["y_last"] + arrow_scale * v["dy"]),
                    xytext=(v["x_last"], v["y_last"]),
                    arrowprops=dict(arrowstyle="-|>", color="red", lw=2.2))
    ax.set_title(f"Step 5(a) - per-frame motion vectors  t-1 -> t  "
                 f"({(t - poke_frame) / fps:+.1f}s)\n"
                 f"nearest-match rule; {n_moving} moving (arrows x{arrow_scale:g})",
                 fontsize=14)
    ax.set_xlim(0, gw * big_px); ax.set_ylim(gh * big_px, 0)
    ax.set_xticks([]); ax.set_yticks([])
    ax.legend(fontsize=11, loc="upper right")
    fig.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)


def fig_vectors_field(dff, centers, big_px, fps, poke_frame, thresh, out):
    """Step 5(b), its own slide: the activation-time propagation field (which
    way the wave travelled), built up from the per-frame vectors."""
    gh, gw = dff.shape[1:]
    fig, ax = plt.subplots(figsize=_panel_figsize(gh, gw))

    t_arr = activation_time_field(dff, fps, poke_frame, thresh)
    im = ax.imshow(t_arr, cmap="turbo",
                   extent=[0, gw * big_px, gh * big_px, 0], interpolation="nearest")
    # propagation direction = +gradient of arrival time (early -> late).
    # Smooth the arrival map first so the arrows reflect the coherent wave
    # direction rather than single-cell noise.
    from scipy.ndimage import gaussian_filter
    filled = np.where(np.isfinite(t_arr), t_arr, np.nanmax(t_arr) + 1)
    sm = gaussian_filter(filled, sigma=0.8)
    gy, gx = np.gradient(sm)
    mag = np.hypot(gx, gy)
    thr_mag = np.nanpercentile(mag[np.isfinite(t_arr)], 35) if np.isfinite(t_arr).any() else 0
    ux = gx / (mag + 1e-9)
    uy = gy / (mag + 1e-9)
    X = centers[..., 0]; Y = centers[..., 1]
    valid = np.isfinite(t_arr) & (mag > max(thr_mag, 1e-6))
    ax.quiver(X[valid], Y[valid], ux[valid], -uy[valid], color="white",
              angles="xy", scale_units="xy", scale=1.0 / (0.5 * big_px),
              width=0.005, headwidth=3.5, headlength=4.5, alpha=0.95)
    ax.set_title("Step 5(b) - propagation-direction field\narrows point the way the "
                 "wave travelled (early -> late arrival)", fontsize=14)
    ax.set_xlim(0, gw * big_px); ax.set_ylim(gh * big_px, 0)
    ax.set_xticks([]); ax.set_yticks([])
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02, label="wave arrival time (s)")
    fig.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)


def label_fronts(dff_frame, thresh, min_cells=2):
    """Group the bright big pixels of one frame into connected components --
    each connected blob is one wave 'front' (4-connectivity, as in the
    production roll-up). Components smaller than `min_cells` are dropped as
    noise. Returns a relabeled grid (0 = background, 1..k = fronts)."""
    from scipy.ndimage import label
    mask = np.nan_to_num(dff_frame) >= thresh
    lab, n = label(mask)
    out = np.zeros_like(lab)
    k = 0
    for g in range(1, n + 1):
        if (lab == g).sum() >= min_cells:
            k += 1
            out[lab == g] = k
    return out


def fig_waves_grouping(dff, centers, big_px, fps, poke_frame, thresh, out):
    """Step 6(a), its own slide: bright big pixels grouped into connected
    wave fronts, with each front's centroid marked."""
    gh, gw = dff.shape[1:]
    ext = [0, gw * big_px, gh * big_px, 0]
    fig, ax = plt.subplots(figsize=_panel_figsize(gh, gw))

    # representative frame = the one that best shows the grouping idea, i.e.
    # the most distinct fronts; tie-broken by total activation.
    def _grouping_score(t):
        lab_t = label_fronts(dff[t], thresh)
        total = float(np.nansum(np.where(np.nan_to_num(dff[t]) >= thresh,
                                         np.nan_to_num(dff[t]), 0.0)))
        return (int(lab_t.max()), total)
    rep = max(range(dff.shape[0]), key=_grouping_score)
    lab = label_fronts(dff[rep], thresh)
    n_groups = int(lab.max())
    disp = np.where(lab > 0, lab, np.nan)
    ax.imshow(np.nan_to_num(dff[rep]), cmap="gray", extent=ext,
              interpolation="nearest", alpha=0.55)
    ax.imshow(disp, cmap="tab10", extent=ext, interpolation="nearest",
              vmin=1, vmax=10, alpha=0.85)
    for g in range(1, n_groups + 1):
        cells = lab == g
        if cells.sum() < 1:
            continue
        gx = float(centers[..., 0][cells].mean())
        gy = float(centers[..., 1][cells].mean())
        ax.plot(gx, gy, "*", color="white", ms=22, mec="k", zorder=4)
    ax.set_title(f"Step 6(a) - Grouping: bright big pixels -> wave fronts\n"
                 f"frame t={(rep - poke_frame) / fps:+.1f}s, "
                 f"{n_groups} connected front(s); * = front centroid", fontsize=14)
    ax.set_xlim(0, gw * big_px); ax.set_ylim(gh * big_px, 0)
    ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)


def fig_waves_centroid(fronts, dff, big_px, out):
    """Step 6(b), its own slide: the brightness-weighted front centroid path
    over time, on the peak-dF/F0 backdrop."""
    gh, gw = dff.shape[1:]
    ext = [0, gw * big_px, gh * big_px, 0]
    fig, ax = plt.subplots(figsize=_panel_figsize(gh, gw))

    peak = np.nanmax(np.nan_to_num(dff), axis=0)
    ax.imshow(peak, cmap="gray", extent=ext, interpolation="nearest")
    xs = [f["x"] for f in fronts]
    ys = [f["y"] for f in fronts]
    ts = [f["time_s"] for f in fronts]
    sc = ax.scatter(xs, ys, c=ts, cmap="turbo", s=90, edgecolors="k", zorder=3)
    ax.plot(xs, ys, "-", color="white", lw=1.6, alpha=0.7, zorder=2)
    for f in fronts:
        ax.annotate(f"{f['time_s']:+.1f}s", (f["x"], f["y"]),
                    fontsize=9, color="white",
                    xytext=(4, 4), textcoords="offset points")
    ax.set_title("Step 6(b) - Wave-front centroid path over time\n"
                 "(brightness-weighted centroid each frame)", fontsize=14)
    ax.set_xlim(0, gw * big_px); ax.set_ylim(gh * big_px, 0)
    ax.set_xticks([]); ax.set_yticks([])
    fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.02, label="time (s)")
    fig.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)


def fig_waves_speed(fronts, out):
    """Step 6(c), its own slide: wave propagation speed over time."""
    fig, ax = plt.subplots(figsize=(11, 7.0))
    speeds = [(f["time_s"], f.get("prop_speed_px_per_s")) for f in fronts
              if f.get("prop_speed_px_per_s") is not None
              and np.isfinite(f.get("prop_speed_px_per_s", float("nan")))]
    if speeds:
        st, sv = zip(*speeds)
        ax.plot(st, sv, "-o", color="#d62728", ms=6)
    ax.axvline(0, color="gray", ls=":", alpha=0.7, label="poke (t=0)")
    ax.set_title("Step 6(c) - Wave propagation speed over time\n"
                 "(centroid displacement / dt)", fontsize=14)
    ax.set_xlabel("time (s, rel. poke)")
    ax.set_ylabel("propagation speed (px/s)")
    ax.legend(fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)


def _find_line(path, needle):
    """Return (lineno, stripped_text) of the first line in `path` containing
    `needle`, or (None, None). Read live so the cited line numbers are always
    current, even after the source files are edited."""
    try:
        with open(path, "r") as fh:
            for i, line in enumerate(fh, start=1):
                if needle in line:
                    return i, line.strip()
    except OSError:
        pass
    return None, None


# Per-figure code cross-reference into the PRODUCTION pipeline. Each ref is
# (production_file, anchor_substring, what_this_line_does); the line is located
# live by its anchor so the printed line numbers stay accurate after edits.
_PIPE = Path(__file__).resolve().parent.parent  # wave-vector-analysis/
_GPBV = _PIPE / "generate_pixel_brightness_vectors.py"
_SCORE = _PIPE / "score_landmark_responses.py"
_ROLL = _PIPE / "rollup_pixel_vectors_to_waves.py"

CODE_NOTES = {
    "fig1_raw_to_bigpixels.png": {
        "title": "Step 1 - Big pixels (block mean)",
        "refs": [],
        "note": "No production line: production runs on full native-resolution\n"
                "pixels. The big-pixel coarsening here is pedagogical only.",
    },
    "fig2_f0_baseline.png": {
        "title": "Step 3 - Baseline F0",
        "refs": [
            (_SCORE, "f0 = np.nanpercentile(c, 20)",
             "F0 = 20th percentile of the pixel's time course"),
        ],
    },
    "fig3_dff_maps.png": {
        "title": "Step 4 - Activation dF/F0",
        "refs": [
            (_SCORE, "dff = (peak - f0) / f0",
             "dF/F0 = (F - F0) / F0"),
        ],
    },
    "fig4a_vectors_perframe.png": {
        "title": "Step 5(a) - How a motion vector is created (production: "
                 "generate_pixel_brightness_vectors.py)",
        "refs": [
            (_GPBV, "lin = linearize(blurred, gmin, gmax)",
             "1. put the frame on the global linear [0,1] brightness scale"),
            (_GPBV, "local_max = maximum_filter(lin, size=size)",
             "2. find local maxima of brightness ..."),
            (_GPBV, "peaks = bright & (lin >= local_max - 1e-6)",
             "   ... that are also above threshold -> these are the bright points"),
            (_GPBV, "order = np.argsort(curr_pts[:, 3])[::-1]",
             "3. process current-frame points brightest-first"),
            (_GPBV, "dists, idxs = tree.query([x_c, y_c], k=k,",
             "4. nearest bright point in frame t-1, within max_link_px (KD-tree)"),
            (_GPBV, "used_last.add(best_li)",
             "5. each t-1 point can be claimed only once"),
            (_GPBV, "dx = float(x_c - x_l)",
             "6. the vector = current point minus its matched t-1 point (dx, dy)"),
            (_GPBV, "\"displacement_px\": float(np.hypot(dx, dy)),",
             "   length = sqrt(dx^2 + dy^2); speed = displacement / dt"),
        ],
    },
    "fig4b_propagation_field.png": {
        "title": "Step 5(b) - Propagation-direction field",
        "refs": [],
        "note": "No direct production line: this arrival-time gradient field is a\n"
                "pedagogical summary view of where the per-frame vectors point\n"
                "overall (early -> late activation = the way the wave travelled).",
    },
    "fig5a_grouping.png": {
        "title": "Step 6(a) - Group bright big pixels into connected fronts",
        "refs": [
            (_ROLL, "labels = connected_components(coords, eps)",
             "group the bright points of a frame into connected fronts"),
        ],
    },
    "fig5b_centroid_path.png": {
        "title": "Step 6(b) - Track each front's centroid over time",
        "refs": [
            (_ROLL, "\"x\": float(np.average(xs, weights=wbraw)),",
             "front position = brightness-weighted centroid"),
        ],
    },
    "fig5c_prop_speed.png": {
        "title": "Step 6(c) - Wave propagation speed",
        "refs": [
            (_ROLL, "prop_speed = net_disp / duration if duration > 0",
             "propagation speed = net front displacement / duration"),
        ],
    },
}


def _code_note_text(name):
    spec = CODE_NOTES.get(name)
    if not spec:
        return None
    lines = [spec["title"], ""]
    for path, needle, label in spec["refs"]:
        ln, txt = _find_line(path, needle)
        if ln is None:
            continue
        if len(txt) > 92:
            txt = txt[:89] + "..."
        lines.append(label)
        lines.append(f"    {Path(path).name}:{ln}   {txt}")
    if spec.get("note"):
        lines += ["", spec["note"]]
    return "\n".join(lines)


def write_pdf(out_dir, png_names, pdf_path):
    from matplotlib.backends.backend_pdf import PdfPages
    import matplotlib.image as mpimg
    with PdfPages(pdf_path) as pdf:
        for name in png_names:
            p = out_dir / name
            if not p.exists():
                continue
            img = mpimg.imread(p)
            h, w = img.shape[:2]
            img_w_in = 11.0
            img_h_in = img_w_in * h / w
            note = _code_note_text(name)
            n_lines = 0 if note is None else note.count("\n") + 1
            txt_h_in = 0.0 if note is None else ((n_lines + 2.5) * 0.205 + 0.25)
            total_h = img_h_in + txt_h_in
            fig = plt.figure(figsize=(img_w_in, total_h))
            ax_img = fig.add_axes([0, txt_h_in / total_h, 1, img_h_in / total_h])
            ax_img.imshow(img)
            ax_img.axis("off")
            if note is not None:
                ax_txt = fig.add_axes([0, 0, 1, txt_h_in / total_h])
                ax_txt.axis("off")
                ax_txt.add_patch(plt.Rectangle((0.012, 0.06), 0.976, 0.9,
                                 transform=ax_txt.transAxes, facecolor="#f4f4f4",
                                 edgecolor="#bbbbbb", lw=0.8, zorder=0))
                ax_txt.text(0.025, 0.93, "Production code cross-reference",
                            transform=ax_txt.transAxes,
                            va="top", ha="left", family="monospace", fontsize=8.5,
                            fontweight="bold")
                ax_txt.text(0.025, 0.78, note, transform=ax_txt.transAxes,
                            va="top", ha="left", family="monospace", fontsize=8.0,
                            linespacing=1.35)
            pdf.savefig(fig, dpi=150)
            plt.close(fig)


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tiff", default=None,
                    help="Real TIFF file or folder (default: synthetic scene)")
    ap.add_argument("--big-px", type=int, default=50,
                    help="Big-pixel block size in native pixels (default 50)")
    ap.add_argument("--fps", type=float, default=2.0)
    ap.add_argument("--poke-frame", type=int, default=2)
    ap.add_argument("--max-frames", type=int, default=16)
    ap.add_argument("--frame-start", type=int, default=0,
                    help="(TIFF) first page to read")
    ap.add_argument("--frame-stride", type=int, default=1,
                    help="(TIFF) read every Nth page so the wave moves between frames")
    ap.add_argument("--no-autocrop", action="store_true",
                    help="(TIFF) keep the full frame instead of cropping to tissue")
    ap.add_argument("--crop-pct", type=float, default=80.0,
                    help="(TIFF) max-projection percentile used as the tissue "
                         "crop threshold (higher = tighter crop)")
    ap.add_argument("--f0-percentile", type=float, default=20.0)
    ap.add_argument("--dff-thresh", type=float, default=0.5,
                    help="dF/F0 cutoff for a big pixel to count as 'bright'")
    ap.add_argument("--max-link-px", type=float, default=None,
                    help="Max vector link distance (default: 2.5 * big_px)")
    ap.add_argument("--arrow-scale", type=float, default=1.0,
                    help="Magnify the per-frame vectors for visibility (e.g. 3-5)")
    ap.add_argument("--output-dir", default=str(Path(__file__).parent / "output"))
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    max_link_px = args.max_link_px or 2.5 * args.big_px

    # 0. frames
    if args.tiff:
        print(f"Loading real TIFF stack: {args.tiff}")
        stack = load_tiff_stack(args.tiff, frame_start=args.frame_start,
                                frame_stride=args.frame_stride,
                                max_frames=args.max_frames,
                                autocrop=not args.no_autocrop,
                                crop_pct=args.crop_pct)
        meta = dict(fps=args.fps, poke_frame=args.poke_frame)
    else:
        print("Building synthetic two-embryo scene ...")
        stack, meta = make_synthetic_stack(
            n_frames=args.max_frames, fps=args.fps, poke_frame=args.poke_frame)
    fps = meta["fps"]
    poke_frame = meta["poke_frame"]
    T, H, W = stack.shape
    print(f"  stack: {T} frames, {W}x{H} px, fps={fps}, poke_frame={poke_frame}")

    # 1. big pixels
    big, centers = coarsen_to_bigpixels(stack, args.big_px)
    gh, gw = big.shape[1:]
    print(f"  big pixels: {gw}x{gh} grid ({args.big_px}px blocks)")

    # 2. global linear range (for display, matches pipeline)
    gmin, gmax = float(stack.min()), float(stack.max())

    # 3. F0 + dF/F0
    f0 = compute_f0(big, percentile=args.f0_percentile)
    dff = compute_dff(big, f0)

    # pick a probe big pixel for the worked example: prefer a cell whose
    # activation peaks in the *interior* of the window (a clear rise-and-fall,
    # i.e. the wave arriving and passing) rather than one already-bright at the
    # very first frame. Among those, take the strongest.
    dnum = np.nan_to_num(dff)
    peak_dff = dnum.max(axis=0)
    peak_t = dnum.argmax(axis=0)
    starts_low = dnum[0] < args.dff_thresh           # at rest at the start
    interior = (peak_t >= 2) & (peak_t <= T - 2)      # rises then falls
    cand = starts_low & interior & (peak_dff > args.dff_thresh)
    # Among cells with a clean rise-and-fall, take the strongest activation --
    # the clearest example of the wave arriving, peaking, and passing.
    score = np.where(cand, peak_dff, -np.inf)
    if np.isfinite(score).any() and score.max() > -np.inf:
        pj, pi = np.unravel_index(np.argmax(score), score.shape)
    else:
        pj, pi = np.unravel_index(np.argmax(peak_dff), peak_dff.shape)

    # 5. wave front
    fronts = wave_front_centroids(dff, centers, args.dff_thresh, fps, poke_frame)

    # ---- figures ----
    busy_frame = int(np.argmax(np.nan_to_num(dff).reshape(T, -1).max(axis=1)))
    fig_raw_to_bigpixels(stack, big, centers, args.big_px, busy_frame, gmin, gmax,
                         out_dir / "fig1_raw_to_bigpixels.png")
    fig_f0(big, f0, centers, args.big_px, fps, poke_frame, (pj, pi),
           out_dir / "fig2_f0_baseline.png")

    # choose 3 dff snapshots spread across the active period
    active = [f["frame"] for f in fronts] or list(range(T))
    snap = sorted(set([active[len(active) // 4], active[len(active) // 2],
                       active[3 * len(active) // 4]]))
    while len(snap) < 3:
        snap.append(min(T - 1, snap[-1] + 1))
    fig_dff(dff, centers, args.big_px, fps, poke_frame, snap[:3], args.dff_thresh,
            out_dir / "fig3_dff_maps.png")

    pair_frames = [f for f in snap[:3] if f >= 1] or [max(1, busy_frame)]
    # Step 5 panels, each on its own slide
    fig_vectors_perframe(dff, centers, args.big_px, fps, poke_frame, pair_frames,
                         args.dff_thresh, max_link_px,
                         out_dir / "fig4a_vectors_perframe.png",
                         arrow_scale=args.arrow_scale)
    fig_vectors_field(dff, centers, args.big_px, fps, poke_frame, args.dff_thresh,
                      out_dir / "fig4b_propagation_field.png")
    # Step 6 panels, each on its own slide
    fig_waves_grouping(dff, centers, args.big_px, fps, poke_frame, args.dff_thresh,
                       out_dir / "fig5a_grouping.png")
    fig_waves_centroid(fronts, dff, args.big_px, out_dir / "fig5b_centroid_path.png")
    fig_waves_speed(fronts, out_dir / "fig5c_prop_speed.png")

    png_names = ["fig1_raw_to_bigpixels.png", "fig2_f0_baseline.png",
                 "fig3_dff_maps.png",
                 "fig4a_vectors_perframe.png", "fig4b_propagation_field.png",
                 "fig5a_grouping.png", "fig5b_centroid_path.png",
                 "fig5c_prop_speed.png"]
    write_pdf(out_dir, png_names, out_dir / "bigpixel_walkthrough.pdf")

    # ---- worked-example numbers ----
    t_axis = (np.arange(T) - poke_frame) / fps
    course = big[:, pj, pi]
    f0v = float(f0[pj, pi])
    peak_t = int(np.nanargmax(np.nan_to_num(dff[:, pj, pi])))
    lines = []
    lines.append("BIG-PIXEL WORKED EXAMPLE\n" + "=" * 40)
    lines.append(f"big-pixel size            : {args.big_px} x {args.big_px} native px")
    lines.append(f"grid                      : {gw} cols x {gh} rows")
    lines.append(f"fps / poke_frame          : {fps} / {poke_frame}")
    lines.append(f"global linear range       : [{gmin:.1f}, {gmax:.1f}]")
    _cx, _cy = (float(v) for v in centers[pj, pi])
    lines.append(f"probe big pixel (row,col) : ({pj}, {pi})  center px ({_cx:.1f}, {_cy:.1f})")
    lines.append("")
    lines.append("F(t) raw intensity time course of probe big pixel:")
    for t in range(T):
        mark = "  <- peak dF/F0" if t == peak_t else ""
        lines.append(f"  t={t_axis[t]:+5.1f}s  frame {t:2d}  F={course[t]:8.1f}"
                     f"  dF/F0={np.nan_to_num(dff[t, pj, pi]):6.2f}{mark}")
    lines.append("")
    lines.append(f"F0 (P{args.f0_percentile:.0f} of F(t)) = {f0v:.1f}")
    lines.append(f"peak F                    = {course.max():.1f}")
    lines.append(f"peak dF/F0                = (peakF - F0)/F0 "
                 f"= ({course.max():.1f} - {f0v:.1f})/{f0v:.1f} "
                 f"= {(course.max() - f0v) / f0v:.2f}")
    lines.append("")
    lines.append("WAVE-FRONT CENTROID TRACK (bright big pixels each frame):")
    lines.append(f"  {'t(s)':>6} {'x':>7} {'y':>7} {'n_px':>5} {'prop_speed(px/s)':>17} {'dir(deg)':>9}")
    for f in fronts:
        sp = f.get("prop_speed_px_per_s")
        di = f.get("dir_deg")
        sp_s = f"{sp:17.1f}" if sp is not None and np.isfinite(sp) else f"{'-':>17}"
        di_s = f"{di:9.0f}" if di is not None else f"{'-':>9}"
        lines.append(f"  {f['time_s']:6.1f} {f['x']:7.1f} {f['y']:7.1f} "
                     f"{f['n_pixels']:5d}{sp_s}{di_s}")
    valid_sp = [f["prop_speed_px_per_s"] for f in fronts
                if f.get("prop_speed_px_per_s") and np.isfinite(f["prop_speed_px_per_s"])]
    if valid_sp:
        lines.append("")
        lines.append(f"mean propagation speed    = {np.mean(valid_sp):.1f} px/s "
                     f"(= {np.mean(valid_sp) * fps / args.big_px:.2f} big-pixels/frame)")
    text = "\n".join(lines)
    (out_dir / "bigpixel_numbers.txt").write_text(text + "\n")
    print("\n" + text)
    print(f"\nWrote figures + PDF + numbers to {out_dir}/")


if __name__ == "__main__":
    main()

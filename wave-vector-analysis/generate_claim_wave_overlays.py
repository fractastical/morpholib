#!/usr/bin/env python3
"""
Per-claim wave-vector overlays on the real embryo frame.

For each TESTED claim group, this renders the wave-front motion vectors that are
RELEVANT TO THAT CLAIM (stimulated-side for the bidirectional claims, neighbor-
side for the neighbor claims) on top of the brightest microscopy frame of the
representative videos — a montage of "all relevant embryos" for the claim — and
also writes an animated GIF of the best representative video.

Outputs (under analysis_results/wave_catalog/overlays/):
  claim_bidirectional.png / .gif   - claims 1-2 (two opposing fronts, stim side)
  claim_neighbor_wave.png / .gif   - claims 9-11 (wave reaching the neighbor)
  claim_neighbor_local.png / .gif  - claims 14-16 (local front speed in neighbor)
  claim_layer3.png / .gif          - claims 7-8 (wound vs press neighbor)
  claim_no_contact.png / .gif      - claim 12 (neighbor wave without contact)

The vectors come from each video's `waves/wave_front_frames.csv`; the side of
each wave is recovered with the same principal-axis split used by
`wave_laterality_analysis.py`.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tifffile as tiff

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from wave_laterality_analysis import (  # noqa: E402
    principal_axis, split_two, load_geometry, _seg_dist)

DEFAULT_ROOT = "/Users/jdietz/Library/CloudStorage/Box-Box/Calcium videos"
STIM_WARM = "#ff5e3a"    # bidirectional lobe A
STIM_COOL = "#33aaff"    # lobe B
NEIGH_C = "#ff35d0"      # neighbor vectors
STIM_FADE = "#5a8fb0"    # stimulated side (context) when highlighting neighbor


# --------------------------------------------------------------------------- #
#  Geometry: label each wave stim / neighbor (mirrors wave_laterality_analysis)
# --------------------------------------------------------------------------- #
def label_sides(we: pd.DataFrame, prefix: str | None = None,
                geo: dict | None = None):
    """Return (labels dict {wave_id: 'stim'|'neighbor'}, mean, axis).

    Uses the manual poke + head/tail geometry when available (same rule as
    wave_laterality_analysis); otherwise falls back to the PCA split.
    """
    xy = we[["x_origin", "y_origin"]].to_numpy(dtype=float)
    bright = we["peak_total_brightness"].to_numpy(dtype=float)
    bright = np.where(np.isfinite(bright) & (bright > 0), bright, 1.0)

    vgeo = (geo or {}).get(prefix) if prefix else None
    if vgeo is not None:
        sides = ["Left", "Right"]
        d_left = np.array([_seg_dist(p, *vgeo["embryos"]["Left"]) for p in xy])
        d_right = np.array([_seg_dist(p, *vgeo["embryos"]["Right"]) for p in xy])
        lab = (d_right < d_left).astype(int)  # 0=Left, 1=Right
        stim_label = sides.index(vgeo["stim_side"])
        h, t = vgeo["embryos"][vgeo["stim_side"]]
        axis = (t - h) / (np.linalg.norm(t - h) + 1e-9)
        out = {}
        for wid, lb in zip(we["wave_id"], lab):
            out[int(wid)] = "stim" if lb == stim_label else "neighbor"
        return out, xy.mean(axis=0), axis

    mean, axis = principal_axis(xy, bright)
    proj = (xy - mean) @ axis
    lab, _sep = split_two(proj)

    def stats(mask):
        sub = we[mask]
        return (float(sub["t_start_s"].min()),
                float(sub["peak_total_brightness"].max()), int(mask.sum()))

    m0, m1 = lab == 0, lab == 1
    s0, s1 = stats(m0), stats(m1)
    if s0[2] == 0 or s1[2] == 0:
        # single embryo of activity -> everything is "stim"
        out = {int(w): "stim" for w in we["wave_id"]}
        return out, mean, axis
    # earliest onset is stimulated; tie -> brighter
    stim_is_0 = (s0[0], -s0[1]) <= (s1[0], -s1[1])
    out = {}
    for wid, lb in zip(we["wave_id"], lab):
        is_stim = (lb == 0) == stim_is_0
        out[int(wid)] = "stim" if is_stim else "neighbor"
    return out, mean, axis


# --------------------------------------------------------------------------- #
#  Source TIFF lookup + brightest frame
# --------------------------------------------------------------------------- #
def build_tiff_index(root: Path) -> dict:
    idx = {}
    for p in root.rglob("*_C-*.tif*"):
        idx[p.stem] = p
    return idx


def brightest_frame_idx(ff: pd.DataFrame) -> int:
    tot = ff.groupby("frame_idx")["total_brightness"].sum()
    return int(tot.idxmax()) if len(tot) else 0


def load_frame(tif_path: Path, frame_idx: int) -> np.ndarray | None:
    try:
        with tiff.TiffFile(tif_path) as tf:
            n = len(tf.pages)
            img = tf.pages[min(frame_idx, n - 1)].asarray()
    except Exception:
        return None
    img = np.asarray(img, dtype=float)
    if img.ndim == 3:
        img = img.mean(axis=2)
    lo, hi = np.percentile(img, [2, 99.5])
    return np.clip((img - lo) / (hi - lo + 1e-9), 0, 1)


# --------------------------------------------------------------------------- #
#  Rendering
# --------------------------------------------------------------------------- #
def draw_overlay(ax, disp, ff, sides, axis, mode, arrow_scale=18):
    """Draw the relevant-side wave vectors on a frame.

    mode: 'bidirectional' (stim side, colored by axis lobe) or
          'neighbor' (neighbor side bright, stim side faded for context).
    """
    if disp is not None:
        ax.imshow(disp, cmap="gray")
    ax.axis("off")

    ff = ff.copy()
    ff["side"] = ff["wave_id"].map(sides)

    if mode == "bidirectional":
        stim = ff[ff["side"] == "stim"]
        # along-axis sign of each front's motion -> two opposing lobes
        along = (stim["mean_dx"].to_numpy() * axis[0]
                 - stim["mean_dy"].to_numpy() * axis[1])
        fwd = stim[along >= 0]
        bwd = stim[along < 0]
        for sub, col, lbl in ((fwd, STIM_WARM, "front → +axis"),
                              (bwd, STIM_COOL, "front → −axis")):
            if len(sub):
                ax.quiver(sub["x"], sub["y"], sub["mean_dx"], -sub["mean_dy"],
                          color=col, scale=arrow_scale, width=0.003,
                          headwidth=4, alpha=0.9, label=lbl)
    else:
        stim = ff[ff["side"] == "stim"]
        neigh = ff[ff["side"] == "neighbor"]
        if len(stim):
            ax.quiver(stim["x"], stim["y"], stim["mean_dx"], -stim["mean_dy"],
                      color=STIM_FADE, scale=arrow_scale, width=0.0022,
                      alpha=0.45, label="stimulated (context)")
        if len(neigh):
            ax.quiver(neigh["x"], neigh["y"], neigh["mean_dx"], -neigh["mean_dy"],
                      color=NEIGH_C, scale=arrow_scale, width=0.0032,
                      headwidth=4, alpha=0.95, label="neighbor wave")


def montage(videos_info, mode, title, out_png):
    """videos_info: list of (label, disp, ff, sides, axis). Up to 4."""
    n = len(videos_info)
    if n == 0:
        return False
    rows = 2 if n > 2 else 1
    cols = 2 if n > 1 else 1
    fig, axes = plt.subplots(rows, cols, figsize=(7.6 * cols, 3.0 * rows))
    axes = np.atleast_1d(axes).ravel()
    for ax in axes[n:]:
        ax.axis("off")
    for ax, (label, disp, ff, sides, axis) in zip(axes, videos_info):
        draw_overlay(ax, disp, ff, sides, axis, mode)
        ax.set_title(label, fontsize=9)
        if ax is axes[0]:
            ax.legend(loc="lower right", fontsize=6.5, framealpha=0.6,
                      facecolor="black", labelcolor="white")
    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=130)
    plt.close(fig)
    return True


def make_gif(label, disp, ff, sides, axis, mode, out_gif, fps=6):
    """Animate the relevant-side fronts appearing frame by frame."""
    ff = ff.copy()
    ff["side"] = ff["wave_id"].map(sides)
    if mode == "bidirectional":
        ff = ff[ff["side"] == "stim"]
    frames = sorted(ff["frame_idx"].unique())
    if len(frames) < 2 or disp is None:
        return False
    fig, ax = plt.subplots(figsize=(11, 4.2))

    def render(fi):
        ax.clear()
        ax.imshow(disp, cmap="gray")
        ax.axis("off")
        upto = ff[ff["frame_idx"] <= fi]
        draw_overlay(ax, None, upto, sides, axis, mode)
        ax.imshow(disp, cmap="gray", zorder=-1)
        ax.set_title(f"{label} — frame {fi}", fontsize=10)

    anim = animation.FuncAnimation(fig, render, frames=frames, interval=1000 / fps)
    out_gif.parent.mkdir(parents=True, exist_ok=True)
    try:
        anim.save(str(out_gif), writer=animation.PillowWriter(fps=fps), dpi=80)
    except Exception as e:  # noqa: BLE001
        print(f"  ! GIF failed for {label}: {e}")
        plt.close(fig)
        return False
    plt.close(fig)
    return True


# --------------------------------------------------------------------------- #
#  Video selection per claim group
# --------------------------------------------------------------------------- #
def load_video(catalog_dir: Path, stem: str, tiff_index: dict,
               geo: dict | None = None):
    vdir = catalog_dir / stem
    we_path = vdir / "waves" / "wave_events.csv"
    ff_path = vdir / "waves" / "wave_front_frames.csv"
    if not (we_path.exists() and ff_path.exists()):
        return None
    we = pd.read_csv(we_path)
    ff = pd.read_csv(ff_path)
    if len(we) < 2 or ff.empty:
        return None
    sides, mean, axis = label_sides(we, stem.split("_")[0], geo)
    bf = brightest_frame_idx(ff)
    tif = tiff_index.get(stem)
    disp = load_frame(tif, bf) if tif else None
    return {"stem": stem, "we": we, "ff": ff, "sides": sides,
            "axis": axis, "bf": bf, "disp": disp}


def pick(lat: pd.DataFrame, mask, key, n=4):
    sub = lat[mask].sort_values(key, ascending=False)
    return sub["video"].tolist()[:n]


GROUPS = [
    # (out_name, title, mode, relevant_key, filter_fn)
    ("claim_bidirectional", "Claims 1–2: bidirectional wave in the stimulated "
     "embryo (two opposing fronts from the poke)", "bidirectional",
     "n_waves_stim",
     lambda lat: (lat["bidirectional_stim"] == 1)),
    ("claim_neighbor_wave", "Claims 9–11: calcium wave reaching the NEIGHBOR "
     "embryo (wounding)", "neighbor", "n_waves_neighbor",
     lambda lat: (lat["neighbor_has_wave"] == 1) & (lat["stimulus"] == "wound")),
    ("claim_neighbor_local", "Claims 14–16: local (front) response in the "
     "NEIGHBOR embryo", "neighbor", "neighbor_mean_front_speed",
     lambda lat: (lat["neighbor_has_wave"] == 1)),
    ("claim_layer3", "Claims 7–8: neighbor response, wounding vs pressure",
     "neighbor", "neighbor_peak_brightness",
     lambda lat: (lat["neighbor_has_wave"] == 1)),
    ("claim_no_contact", "Claim 12: neighbor wave WITHOUT physical contact",
     "neighbor", "n_waves_neighbor",
     lambda lat: (lat["neighbor_has_wave"] == 1) & (lat["contact"] == "no-contact")),
]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--catalog-dir",
                    default=str(HERE / "analysis_results" / "wave_catalog"))
    ap.add_argument("--laterality-csv",
                    default=str(HERE / "analysis_results" / "wave_catalog"
                               / "wave_laterality.csv"))
    ap.add_argument("--root", default=DEFAULT_ROOT,
                    help="Box root to find source C-channel TIFFs")
    ap.add_argument("--out-dir",
                    default=str(HERE / "analysis_results" / "wave_catalog"
                               / "overlays"))
    ap.add_argument("--no-gif", action="store_true")
    ap.add_argument("--montage-n", type=int, default=4)
    ap.add_argument("--ground-truth",
                    default=str(HERE / "analysis_results" / "xy_ground_truth.csv"))
    args = ap.parse_args()

    catalog_dir = Path(args.catalog_dir)
    lat = pd.read_csv(args.laterality_csv)
    out_dir = Path(args.out_dir)
    geo = load_geometry(Path(args.ground_truth))
    print(f"Loaded real geometry for {len(geo)} videos")
    print("Indexing source TIFFs …")
    tiff_index = build_tiff_index(Path(args.root))
    print(f"  {len(tiff_index)} C-channel TIFFs found")

    cache: dict[str, dict] = {}

    def get(stem):
        if stem not in cache:
            cache[stem] = load_video(catalog_dir, stem, tiff_index, geo)
        return cache[stem]

    for out_name, title, mode, key, filt in GROUPS:
        mask = filt(lat)
        if mode == "neighbor" and out_name == "claim_layer3":
            # one wound + best press, to contrast the two conditions
            stems = (pick(lat, mask & (lat["stimulus"] == "wound"), key, 2)
                     + pick(lat, mask & (lat["stimulus"] == "press"), key, 2))
        else:
            stems = pick(lat, mask, key, args.montage_n)
        infos = []
        for stem in stems:
            v = get(stem)
            if v is None or v["disp"] is None:
                continue
            n_rel = sum(1 for s in v["sides"].values()
                        if (s == "stim") == (mode == "bidirectional"))
            lbl = f"{stem.split('_')[0]}  ({n_rel} {'stim' if mode=='bidirectional' else 'neighbor'} waves)"
            infos.append((lbl, v["disp"], v["ff"], v["sides"], v["axis"]))
        ok = montage(infos, mode, title, out_dir / f"{out_name}.png")
        print(f"[{out_name}] {'wrote montage' if ok else 'NO DATA'} "
              f"({len(infos)} videos)")
        if ok and not args.no_gif and infos:
            lbl, disp, ff, sides, axis = infos[0]
            g = make_gif(lbl, disp, ff, sides, axis, mode,
                         out_dir / f"{out_name}.gif")
            if g:
                print(f"  + GIF {out_name}.gif")

    print(f"\nDone -> {out_dir}")


if __name__ == "__main__":
    main()

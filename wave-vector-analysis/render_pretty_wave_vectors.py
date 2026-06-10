#!/usr/bin/env python3
"""
Prettier embryo wave visualizations: calcium glow background, propagation trails,
and speed-colored wave-front vectors.

Reads each catalog video's waves/wave_events.csv + wave_front_frames.csv.
No-contact mode uses the real microscope frame at wave onset (not peak),
shows poke + both embryos, and pink direction arrows on the neighbor wave.
GIFs animate actual frames so brightness rises over time.

Outputs (default: wave-vector-analysis/pretty_waves/):
  no_contact/<stem>_neighbor.png / .gif
  no_contact/snapshots/<id>_t10s.png …  — several time points after poke
  no_contact/all_six_no_contact.png     — static 2×3 grid (all WGD81–WGD86)
  no_contact/all_six_no_contact.gif     — animated 2×3 grid (time synced)
  no_contact/timeline_montage.png       — grid of all times × videos

Example:
  python render_pretty_wave_vectors.py --no-contact
  python render_pretty_wave_vectors.py --no-contact --snapshots 5,15,25,35
  python render_pretty_wave_vectors.py --video WGD82_C-oHTHT-pT-Zoom5.60 --no-contact
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
from matplotlib.colors import LinearSegmentedColormap, Normalize
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from generate_claim_wave_overlays import (  # noqa: E402
    STIM_COOL,
    STIM_FADE,
    STIM_WARM,
    NEIGH_C,
    brightest_frame_idx,
    build_tiff_index,
    label_sides,
    load_frame,
)
from wave_laterality_analysis import load_geometry  # noqa: E402

DEFAULT_ROOT = "/Users/jdietz/Library/CloudStorage/Box-Box/Calcium videos"
DEFAULT_CATALOG = HERE / "analysis_results" / "wave_catalog"
# Outside analysis_results/ so PNG/GIF aren't hidden by .gitignore in the IDE.
DEFAULT_OUT = HERE / "pretty_waves"

BG = "#070b12"
CALCIUM_CMAP = LinearSegmentedColormap.from_list(
    "calcium_glow",
    ["#070b12", "#1a1030", "#5c1a4a", "#c43a20", "#ffd080", "#fff8e8"],
    N=256,
)
SPEED_CMAP = plt.cm.coolwarm

SHOWCASE = [
    ("WGD84_C-oHTHT-pT-Zoom5.60", "neighbor", "Neighbor wave (no contact)"),
    ("WGD82_C-oHTHT-pT-Zoom5.60", "neighbor", "Neighbor wave (wound)"),
    ("WG30_C-oHTHT-pT-Zoom6.30", "bidirectional", "Bidirectional stim"),
    ("WG39_C-oHTHT-pT-Zoom5.60", "neighbor", "Neighbor + stim context"),
]

# Claim 12: calcium reaches the neighbor embryo with no physical contact (WGD series).
NO_CONTACT_NEIGHBOR = [
    ("WGD82_C-oHTHT-pT-Zoom5.60", "WGD82"),
    ("WGD84_C-oHTHT-pT-Zoom5.60", "WGD84"),
]

ALL_NO_CONTACT = [
    ("WGD81_C-oHTHT-pT-Zoom5.60", "WGD81"),
    ("WGD82_C-oHTHT-pT-Zoom5.60", "WGD82"),
    ("WGD83_C-oHTHT-pT-Zoom5.60", "WGD83"),
    ("WGD84_C-oHTHT-pT-Zoom5.60", "WGD84"),
    ("WGD85_C-oHTHT-pT-Zoom5.60", "WGD85"),
    ("WGD86_C-oHTHT-pT-Zoom5.60", "WGD86"),
]

EXPLAIN_NEIGHBOR = (
    "Pink arrows = wave front direction on the neighbor embryo. "
    "Magenta X = poke site. Embryos are not touching."
)
POKE_COLOR = "#ff44cc"
ARROW_SCALE = 22.0
DEFAULT_SNAPSHOT_SECONDS = (10, 20, 30, 40)


def calcium_rgb(gray: np.ndarray, gamma: float = 0.85) -> np.ndarray:
    """Map normalized grayscale [0,1] to an RGB calcium-glow image."""
    g = np.clip(gray, 0, 1) ** gamma
    rgb = CALCIUM_CMAP(g)[:, :, :3]
    dim = (g < 0.08).astype(float)[..., None]
    rgb = rgb * (1 - 0.65 * dim) + np.array([0.03, 0.04, 0.07]) * (0.65 * dim)
    return np.clip(rgb, 0, 1)


def side_color(side: str, along_sign: float, mode: str) -> str:
    if side == "neighbor":
        return NEIGH_C
    if mode == "bidirectional":
        return STIM_WARM if along_sign >= 0 else STIM_COOL
    return STIM_FADE


def poke_frame_idx(prefix: str, geo: dict | None, gt_csv: Path) -> int:
    """Frame index of the manual poke annotation (default 0)."""
    if gt_csv.exists():
        gt = pd.read_csv(gt_csv, comment="#")
        sub = gt[(gt["prefix"] == prefix)
                 & (gt["landmark_class"].isin(["poke", "pressure"]))]
        if len(sub):
            return max(0, int(sub.iloc[0].frame) - 1)
    return 0


def pick_snapshot_frame(ff: pd.DataFrame, sides: dict,
                        side_filter: set[str], poke_f: int) -> int:
    """Frame when the neighbor wave is spreading (not the global peak snapshot)."""
    keep = {wid for wid, s in sides.items() if s in side_filter}
    sub = ff[ff["wave_id"].isin(keep)]
    if sub.empty:
        return brightest_frame_idx(ff)
    by_f = sub.groupby("frame_idx")["total_brightness"].sum().sort_index()
    peak = float(by_f.max())
    if peak <= 0:
        return int(by_f.index[0])
    # First frame reaching ~50% of peak neighbor brightness.
    hit = by_f[by_f >= 0.5 * peak]
    fi = int(hit.index[0]) if len(hit) else int(by_f.idxmax())
    return max(fi, poke_f + 3)


def animation_frames(ff: pd.DataFrame, sides: dict, side_filter: set[str],
                     poke_f: int, tif_pages: int, step: int = 2) -> list[int]:
    keep = {wid for wid, s in sides.items() if s in side_filter}
    sub = ff[ff["wave_id"].isin(keep)]
    if sub.empty:
        end = min(tif_pages - 1, poke_f + 40)
    else:
        end = int(sub["frame_idx"].max()) + 8
    start = max(0, poke_f - 1)
    end = min(tif_pages - 1, end)
    return list(range(start, end + 1, max(1, step)))


def norm_range_for_frames(tif: Path, frame_indices: list[int]) -> tuple[float, float]:
    """Shared lo/hi percentiles so brightness visibly rises across a GIF."""
    samples = []
    try:
        with tiff.TiffFile(tif) as tf:
            n = len(tf.pages)
            for fi in frame_indices:
                img = tf.pages[min(fi, n - 1)].asarray()
                if img.ndim == 3:
                    img = img.mean(axis=2)
                samples.append(np.asarray(img, dtype=float).ravel())
    except Exception:
        return 0.0, 1.0
    if not samples:
        return 0.0, 1.0
    stack = np.concatenate(samples)
    lo, hi = np.percentile(stack, [1, 99.5])
    return float(lo), float(hi)


def load_frame_scaled(tif: Path, frame_idx: int,
                      lo: float | None = None, hi: float | None = None
                      ) -> np.ndarray | None:
    try:
        with tiff.TiffFile(tif) as tf:
            n = len(tf.pages)
            img = tf.pages[min(frame_idx, n - 1)].asarray()
    except Exception:
        return None
    img = np.asarray(img, dtype=float)
    if img.ndim == 3:
        img = img.mean(axis=2)
    if lo is None or hi is None:
        lo, hi = np.percentile(img, [2, 99.5])
    return np.clip((img - lo) / (hi - lo + 1e-9), 0, 1)


def tiff_page_count(tif: Path | None) -> int:
    if tif is None:
        return 0
    try:
        with tiff.TiffFile(tif) as tf:
            return len(tf.pages)
    except Exception:
        return 0


def load_video_data(catalog_dir: Path, stem: str, tiff_index: dict,
                    geo: dict | None = None, gt_csv: Path | None = None):
    vdir = catalog_dir / stem
    we_path = vdir / "waves" / "wave_events.csv"
    ff_path = vdir / "waves" / "wave_front_frames.csv"
    if not (we_path.exists() and ff_path.exists()):
        return None
    we = pd.read_csv(we_path)
    ff = pd.read_csv(ff_path)
    if ff.empty:
        return None
    prefix = stem.split("_")[0]
    sides, mean, axis = label_sides(we, prefix, geo)
    tif = tiff_index.get(stem)
    poke_f = poke_frame_idx(prefix, geo, gt_csv or Path())
    snap_f = pick_snapshot_frame(ff, sides, {"neighbor"}, poke_f)
    n_pages = tiff_page_count(tif)
    disp = load_frame_scaled(tif, snap_f) if tif else None
    return {
        "stem": stem,
        "prefix": prefix,
        "we": we,
        "ff": ff,
        "sides": sides,
        "axis": axis,
        "poke_f": poke_f,
        "snap_f": snap_f,
        "disp": disp,
        "tif": tif,
        "n_pages": n_pages,
    }


def _along_sign(row, axis: np.ndarray) -> float:
    return float(row["mean_dx"] * axis[0] - row["mean_dy"] * axis[1])


def _wave_palette(we: pd.DataFrame, top_n: int,
                  sides: dict | None = None,
                  side_filter: set[str] | None = None) -> list[int]:
    sub = we.copy()
    if sides and side_filter:
        keep = {wid for wid, s in sides.items() if s in side_filter}
        sub = sub[sub["wave_id"].isin(keep)]
    sub = sub.sort_values("peak_total_brightness", ascending=False)
    return sub["wave_id"].head(top_n).tolist()


def load_no_contact_neighbor_videos(laterality_csv: Path) -> list[str]:
    """Videos where contact=no-contact and a neighbor-side wave was detected."""
    if not laterality_csv.exists():
        return [s for s, _ in NO_CONTACT_NEIGHBOR]
    lat = pd.read_csv(laterality_csv)
    mask = (lat["contact"] == "no-contact") & (lat["neighbor_has_wave"] == 1)
    return lat.loc[mask, "video"].tolist()


def load_all_no_contact_videos(laterality_csv: Path) -> list[str]:
    """All six WGD no-contact calcium videos (WGD81–WGD86)."""
    if not laterality_csv.exists():
        return [s for s, _ in ALL_NO_CONTACT]
    lat = pd.read_csv(laterality_csv)
    stems = lat.loc[lat["contact"] == "no-contact", "video"].tolist()
    order = {s: i for i, (s, _) in enumerate(ALL_NO_CONTACT)}
    return sorted(stems, key=lambda s: order.get(s, 99))


def neighbor_wave_flags(laterality_csv: Path) -> dict[str, bool]:
    if not laterality_csv.exists():
        pos = {s for s, _ in NO_CONTACT_NEIGHBOR}
        return {s: s in pos for s, _ in ALL_NO_CONTACT}
    lat = pd.read_csv(laterality_csv)
    nc = lat[lat["contact"] == "no-contact"]
    return dict(zip(nc["video"], nc["neighbor_has_wave"].astype(bool)))


def poke_xy(prefix: str, geo: dict | None) -> np.ndarray | None:
    vgeo = (geo or {}).get(prefix)
    if vgeo is None:
        return None
    poke = vgeo.get("poke")
    return np.asarray(poke, dtype=float) if poke is not None else None


def draw_poke_marker(ax, poke: np.ndarray, zorder: int = 20, size: float = 13):
    """Poke site — magenta X (matches detection-summary style)."""
    px, py = float(poke[0]), float(poke[1])
    ax.plot(
        px, py, linestyle="None", marker="X",
        color=POKE_COLOR, markersize=size, markeredgewidth=2.8,
        markeredgecolor="white", zorder=zorder,
    )


def draw_embryo_context(ax, prefix: str, geo: dict | None, alpha: float = 0.7,
                        show_poke: bool = True, compact: bool = False):
    """Head–tail axes, embryo labels, and poke marker."""
    vgeo = (geo or {}).get(prefix)
    if vgeo is None:
        return
    stim_side = vgeo.get("stim_side", "Left")
    poke_size = 9 if compact else 13
    for side, (h, t) in vgeo.get("embryos", {}).items():
        mid = (h + t) / 2
        is_poked = side == stim_side
        color = "#ffcc66" if is_poked else "#c8a0ff"
        ax.plot([h[0], t[0]], [h[1], t[1]], "-", color=color,
                lw=1.2 if compact else 1.6, alpha=alpha, zorder=12)
        if not compact:
            label = "Poked embryo" if is_poked else "Neighbor embryo"
            ax.text(
                mid[0], mid[1] - 42, label,
                color=color, fontsize=9, fontweight="bold", ha="center", va="bottom",
                zorder=13,
                bbox=dict(boxstyle="round,pad=0.25", facecolor="#101820",
                          edgecolor=color, alpha=0.8),
            )
    if show_poke:
        pk = poke_xy(prefix, geo)
        if pk is not None:
            draw_poke_marker(ax, pk, size=poke_size)


def draw_frame_vectors(ax, ff: pd.DataFrame, wave_ids: list[int],
                       frame_idx: int, color: str = NEIGH_C,
                       arrow_scale: float = ARROW_SCALE):
    """Directional arrows for wave fronts at a single time point."""
    sub = ff[(ff["frame_idx"] == frame_idx) & (ff["wave_id"].isin(wave_ids))]
    if sub.empty:
        return
    ax.quiver(
        sub["x"], sub["y"],
        sub["mean_dx"], -sub["mean_dy"],
        color=color, scale=arrow_scale, width=0.0035,
        headwidth=4.5, headlength=5, alpha=0.92, zorder=11,
    )


def draw_wave_trails(ax, ff: pd.DataFrame, we: pd.DataFrame, sides: dict,
                     axis: np.ndarray, mode: str, wave_ids: list[int] | None,
                     upto_frame: int | None = None, show_vectors: bool = True,
                     vectors_at_end: bool = False,
                     side_filter: set[str] | None = None):
    """Cumulative propagation trails + optional front vectors."""
    ff = ff.copy()
    if upto_frame is not None:
        ff = ff[ff["frame_idx"] <= upto_frame]
    if ff.empty:
        return

    if wave_ids is None:
        wave_ids = sorted(ff["wave_id"].unique())

    speed_vals = ff["mean_speed"].replace([np.inf, -np.inf], np.nan).dropna()
    speed_norm = Normalize(
        vmin=float(speed_vals.quantile(0.05)) if len(speed_vals) else 0.0,
        vmax=float(speed_vals.quantile(0.95)) if len(speed_vals) else 1.0,
    )

    for wid in wave_ids:
        grp = ff[ff["wave_id"] == wid].sort_values("frame_idx")
        if grp.empty:
            continue
        side = sides.get(int(wid), "stim")
        if side_filter and side not in side_filter:
            continue
        along = _along_sign(grp.iloc[-1], axis)
        color = NEIGH_C if side_filter == {"neighbor"} else side_color(side, along, mode)

        xs = grp["x"].to_numpy()
        ys = grp["y"].to_numpy()
        n = len(xs)
        alphas = np.linspace(0.25, 0.95, n)
        widths = np.linspace(0.8, 2.4, n)

        for i in range(1, n):
            ax.plot(
                xs[i - 1:i + 1], ys[i - 1:i + 1], "-",
                color=color, lw=widths[i] + 2.5, alpha=alphas[i] * 0.18,
                zorder=6, solid_capstyle="round",
            )
            ax.plot(
                xs[i - 1:i + 1], ys[i - 1:i + 1], "-",
                color=color, lw=widths[i], alpha=alphas[i], zorder=7,
                solid_capstyle="round",
            )

        if show_vectors and not (side_filter == {"neighbor"}):
            rows = [grp.iloc[-1]] if vectors_at_end else [grp.iloc[i] for i in range(n)]
            for row in rows:
                x, y = float(row["x"]), float(row["y"])
                dx, dy = float(row["mean_dx"]), float(row["mean_dy"])
                spd = float(row["mean_speed"])
                if not np.isfinite(spd) or spd <= 0:
                    continue
                sc = SPEED_CMAP(speed_norm(spd))
                scale = 10 if vectors_at_end else 6
                ax.annotate(
                    "", xy=(x + dx * scale, y - dy * scale), xytext=(x, y),
                    arrowprops=dict(
                        arrowstyle="-|>", color=sc, lw=1.4 if vectors_at_end else 1.1,
                        mutation_scale=9 if vectors_at_end else 7, alpha=0.85,
                    ),
                    zorder=9,
                )


def frame_at_seconds_after_poke(ff: pd.DataFrame, poke_f: int,
                                target_s: float, n_pages: int) -> int:
    """Pick the catalog frame nearest `target_s` seconds after the poke."""
    frames = sorted(ff["frame_idx"].unique())
    if not frames:
        return min(poke_f, max(0, n_pages - 1))
    best = min(
        frames,
        key=lambda fi: abs(_seconds_after_poke(int(fi), poke_f, ff) - target_s),
    )
    return int(min(max(0, best), n_pages - 1))


def _seconds_after_poke(frame_idx: int, poke_f: int, ff: pd.DataFrame) -> float:
    row = ff.loc[ff["frame_idx"] == frame_idx, "time_s"]
    if len(row):
        t0 = ff.loc[ff["frame_idx"] == poke_f, "time_s"]
        base = float(t0.iloc[0]) if len(t0) else float(poke_f)
        return float(row.iloc[0]) - base
    return float(frame_idx - poke_f)


def render_static(vdata: dict, out_png: Path, mode: str = "auto",
                  top_n: int = 12, geo: dict | None = None,
                  title: str | None = None, neighbor_only: bool = False,
                  short_label: str | None = None, frame_idx: int | None = None):
    if vdata["tif"] is None:
        return False

    ff, we = vdata["ff"], vdata["we"]
    sides, axis = vdata["sides"], vdata["axis"]
    side_filter = {"neighbor"} if neighbor_only else None
    if neighbor_only:
        mode = "neighbor"
    elif mode == "auto":
        n_neigh = sum(1 for s in sides.values() if s == "neighbor")
        n_stim = sum(1 for s in sides.values() if s == "stim")
        mode = "neighbor" if n_neigh > n_stim else "bidirectional"

    wave_ids = _wave_palette(we, top_n, sides, side_filter)
    if not wave_ids:
        return False

    if frame_idx is None:
        fi = vdata["snap_f"] if neighbor_only else brightest_frame_idx(ff)
    else:
        fi = int(frame_idx)
    disp = load_frame_scaled(vdata["tif"], fi)
    if disp is None:
        return False
    rgb = calcium_rgb(disp)

    fig, ax = plt.subplots(figsize=(12, 5.2), facecolor=BG)
    ax.set_facecolor(BG)
    ax.imshow(rgb, origin="upper", interpolation="bilinear")
    ax.axis("off")

    if neighbor_only:
        draw_frame_vectors(ax, ff, wave_ids, fi)
        draw_embryo_context(ax, vdata["prefix"], geo)
    else:
        draw_embryo_context(ax, vdata["prefix"], geo, alpha=0.45)
        draw_wave_trails(ax, ff, we, sides, axis, mode, wave_ids,
                         vectors_at_end=True, side_filter=side_filter)

    from matplotlib.lines import Line2D
    if neighbor_only:
        label = short_label or vdata["prefix"]
        dt = _seconds_after_poke(fi, vdata["poke_f"], ff)
        ax.set_title(
            title or (
                f"{label} — no contact  |  "
                f"{dt:.0f} s after poke (wave spreading on neighbor)"
            ),
            color="white", fontsize=11, fontweight="bold", pad=10,
        )
        ax.text(
            0.5, 0.02, EXPLAIN_NEIGHBOR,
            transform=ax.transAxes, ha="center", va="bottom",
            color="#d8dce8", fontsize=8.5, style="italic",
            bbox=dict(boxstyle="round,pad=0.35", facecolor="#101820",
                      edgecolor="#334455", alpha=0.75),
        )
        n_arrows = len(ff[(ff["frame_idx"] == fi) & (ff["wave_id"].isin(wave_ids))])
        legend_els = [
            Line2D([0], [0], marker="X", linestyle="None", markersize=11,
                   markerfacecolor=POKE_COLOR, markeredgecolor="white",
                   markeredgewidth=1.5, label="poke site"),
            Line2D([0], [0], color=NEIGH_C, lw=2.5,
                   label=f"wave front direction ({n_arrows} arrows)"),
        ]
    else:
        stim_n = sum(1 for w in wave_ids if sides.get(w) == "stim")
        neigh_n = sum(1 for w in wave_ids if sides.get(w) == "neighbor")
        subtitle = f"{len(wave_ids)} waves  |  stim {stim_n}  neighbor {neigh_n}"
        ax.set_title(
            title or f"{vdata['stem']}\n{subtitle}",
            color="white", fontsize=11, pad=8,
        )
        legend_els = [
            Line2D([0], [0], color=NEIGH_C, lw=2, label="neighbor wave"),
            Line2D([0], [0], color=STIM_WARM, lw=2, label="stim → +axis"),
            Line2D([0], [0], color=STIM_COOL, lw=2, label="stim → −axis"),
        ]
    leg = ax.legend(handles=legend_els, loc="upper right", fontsize=7.5,
                    framealpha=0.35, facecolor="#101820", labelcolor="white")
    leg.get_frame().set_edgecolor("#334455")

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=160, facecolor=BG, bbox_inches="tight")
    plt.close(fig)
    return True


def render_gif(vdata: dict, out_gif: Path, mode: str = "auto",
               top_n: int = 10, geo: dict | None = None, fps: float = 8,
               neighbor_only: bool = False, short_label: str | None = None):
    if vdata["tif"] is None:
        return False

    ff, we = vdata["ff"], vdata["we"]
    sides, axis = vdata["sides"], vdata["axis"]
    side_filter = {"neighbor"} if neighbor_only else None
    if neighbor_only:
        mode = "neighbor"
    elif mode == "auto":
        n_neigh = sum(1 for s in sides.values() if s == "neighbor")
        n_stim = sum(1 for s in sides.values() if s == "stim")
        mode = "neighbor" if n_neigh > n_stim else "bidirectional"

    wave_ids = _wave_palette(we, top_n, sides, side_filter)
    if not wave_ids:
        return False
    ff_sub = ff[ff["wave_id"].isin(wave_ids)]

    if neighbor_only:
        frames = animation_frames(
            ff, sides, {"neighbor"}, vdata["poke_f"], vdata["n_pages"], step=2,
        )
    else:
        frames = sorted(ff_sub["frame_idx"].unique())
    if len(frames) < 2:
        return False

    lo, hi = norm_range_for_frames(vdata["tif"], frames)
    fig, ax = plt.subplots(figsize=(11, 4.8), facecolor=BG)

    def render_frame(fi):
        ax.clear()
        ax.set_facecolor(BG)
        disp = load_frame_scaled(vdata["tif"], fi, lo=lo, hi=hi)
        if disp is None:
            return
        ax.imshow(calcium_rgb(disp), origin="upper", interpolation="bilinear")
        ax.axis("off")
        if neighbor_only:
            draw_frame_vectors(ax, ff_sub, wave_ids, fi)
            draw_embryo_context(ax, vdata["prefix"], geo, alpha=0.75)
        else:
            draw_embryo_context(ax, vdata["prefix"], geo, alpha=0.75)
            draw_wave_trails(
                ax, ff_sub, we, sides, axis, mode, wave_ids,
                upto_frame=fi, show_vectors=True, side_filter=side_filter,
            )
        dt = _seconds_after_poke(fi, vdata["poke_f"], ff)
        label = short_label or vdata["prefix"]
        if neighbor_only:
            ax.set_title(
                f"{label} — calcium rising  |  {dt:.0f} s after poke",
                color="white", fontsize=10,
            )
        else:
            ax.set_title(
                f"{vdata['stem']}  |  t = {dt:.1f} s",
                color="white", fontsize=10,
            )

    anim = animation.FuncAnimation(fig, render_frame, frames=frames,
                                   interval=1000 / fps)
    out_gif.parent.mkdir(parents=True, exist_ok=True)
    try:
        anim.save(str(out_gif), writer=animation.PillowWriter(fps=fps), dpi=90)
    except Exception as exc:  # noqa: BLE001
        print(f"  ! GIF failed: {exc}")
        plt.close(fig)
        return False
    plt.close(fig)
    return True


def _panel_frame(vdata: dict, has_neighbor_wave: bool, target_s: float = 20.0) -> int:
    if has_neighbor_wave:
        return vdata["snap_f"]
    return frame_at_seconds_after_poke(
        vdata["ff"], vdata["poke_f"], target_s, vdata["n_pages"],
    )


def _draw_neighbor_panel(ax, vdata: dict, fi: int, geo: dict | None,
                         wave_ids: list[int], title: str,
                         show_neighbor_arrows: bool = True,
                         lo: float | None = None, hi: float | None = None,
                         compact: bool = False):
    disp = load_frame_scaled(vdata["tif"], fi, lo=lo, hi=hi)
    if disp is None:
        return False
    ax.imshow(calcium_rgb(disp), origin="upper", interpolation="bilinear")
    ax.axis("off")
    if show_neighbor_arrows and wave_ids:
        draw_frame_vectors(ax, vdata["ff"], wave_ids, fi)
    draw_embryo_context(ax, vdata["prefix"], geo, alpha=0.65, compact=compact)
    ax.set_title(title, color="white", fontsize=9, fontweight="bold")
    return True


def render_all_six_montage(catalog_dir: Path, tiff_index: dict, geo: dict | None,
                           out_png: Path, laterality_csv: Path, top_n: int = 8):
    """2×3 grid of all no-contact WGD videos at a representative time point."""
    gt_csv = HERE / "analysis_results" / "xy_ground_truth.csv"
    flags = neighbor_wave_flags(laterality_csv)
    entries = ALL_NO_CONTACT
    nrows, ncols = 2, 3
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(5.4 * ncols, 3.5 * nrows), facecolor=BG,
    )
    axes = np.atleast_2d(axes)

    for idx, (stem, short) in enumerate(entries):
        ri, ci = divmod(idx, ncols)
        ax = axes[ri, ci]
        ax.set_facecolor(BG)
        vdata = load_video_data(catalog_dir, stem, tiff_index, geo, gt_csv)
        if vdata is None or vdata["tif"] is None:
            ax.text(0.5, 0.5, f"missing\n{short}", ha="center", va="center",
                    color="white", transform=ax.transAxes)
            ax.axis("off")
            continue
        has_neigh = flags.get(stem, False)
        fi = _panel_frame(vdata, has_neigh)
        dt = _seconds_after_poke(fi, vdata["poke_f"], vdata["ff"])
        wave_ids = (
            _wave_palette(vdata["we"], top_n, vdata["sides"], {"neighbor"})
            if has_neigh else []
        )
        suffix = "neighbor wave" if has_neigh else "stim only"
        title = f"{short}  |  {dt:.0f} s  |  {suffix}"
        if not _draw_neighbor_panel(
            ax, vdata, fi, geo, wave_ids, title, show_neighbor_arrows=has_neigh,
        ):
            ax.text(0.5, 0.5, "no frame", ha="center", va="center",
                    color="white", transform=ax.transAxes)
            ax.axis("off")

    fig.suptitle(
        "All no-contact pokes (WGD81–WGD86) — tail wound, embryos separated",
        color="white", fontsize=13, fontweight="bold", y=1.01,
    )
    fig.text(
        0.5, 0.01, EXPLAIN_NEIGHBOR,
        ha="center", va="bottom", color="#c8ccd8", fontsize=9, style="italic",
    )
    fig.tight_layout(rect=[0, 0.04, 1, 0.96])
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=155, facecolor=BG, bbox_inches="tight")
    plt.close(fig)
    return True


def render_all_six_gif(catalog_dir: Path, tiff_index: dict, geo: dict | None,
                       out_gif: Path, laterality_csv: Path, top_n: int = 8,
                       fps: float = 6, step_s: int = 2, max_s: int = 44):
    """Animated 2×3 grid — all six videos advance in sync (seconds after poke)."""
    gt_csv = HERE / "analysis_results" / "xy_ground_truth.csv"
    flags = neighbor_wave_flags(laterality_csv)
    timeline = list(range(0, max_s + 1, max(1, step_s)))

    panels = []
    for stem, short in ALL_NO_CONTACT:
        vdata = load_video_data(catalog_dir, stem, tiff_index, geo, gt_csv)
        if vdata is None or vdata["tif"] is None:
            continue
        has_neigh = flags.get(stem, False)
        wave_ids = (
            _wave_palette(vdata["we"], top_n, vdata["sides"], {"neighbor"})
            if has_neigh else []
        )
        fis = [
            frame_at_seconds_after_poke(
                vdata["ff"], vdata["poke_f"], float(t), vdata["n_pages"],
            )
            for t in timeline
        ]
        lo, hi = norm_range_for_frames(vdata["tif"], fis)
        panels.append({
            "stem": stem, "short": short, "vdata": vdata,
            "has_neigh": has_neigh, "wave_ids": wave_ids,
            "lo": lo, "hi": hi,
        })

    if not panels:
        return False

    nrows, ncols = 2, 3
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(5.4 * ncols, 3.5 * nrows), facecolor=BG,
    )
    axes_flat = np.atleast_2d(axes).ravel()

    def render_step(ti: int):
        t_s = timeline[ti]
        for ax in axes_flat:
            ax.clear()
            ax.set_facecolor(BG)
        fig.suptitle(
            f"All no-contact pokes (WGD81–WGD86)  |  {t_s} s after poke",
            color="white", fontsize=13, fontweight="bold", y=1.01,
        )
        for ax, panel in zip(axes_flat, panels):
            vdata = panel["vdata"]
            fi = frame_at_seconds_after_poke(
                vdata["ff"], vdata["poke_f"], float(t_s), vdata["n_pages"],
            )
            suffix = "neighbor wave" if panel["has_neigh"] else "stim only"
            title = f"{panel['short']}  |  {suffix}"
            _draw_neighbor_panel(
                ax, vdata, fi, geo, panel["wave_ids"], title,
                show_neighbor_arrows=panel["has_neigh"],
                lo=panel["lo"], hi=panel["hi"], compact=True,
            )

    anim = animation.FuncAnimation(
        fig, render_step, frames=len(timeline), interval=1000 / fps,
    )
    out_gif.parent.mkdir(parents=True, exist_ok=True)
    try:
        anim.save(str(out_gif), writer=animation.PillowWriter(fps=fps), dpi=90)
    except Exception as exc:  # noqa: BLE001
        print(f"  ! all-six GIF failed: {exc}")
        plt.close(fig)
        return False
    plt.close(fig)
    return True


def render_snapshot_series(vdata: dict, out_dir: Path, geo: dict | None,
                           short_label: str | None, top_n: int,
                           seconds: tuple[float, ...]) -> list[Path]:
    """Write one PNG per time point (seconds after poke)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    label = short_label or vdata["prefix"]
    for target_s in seconds:
        fi = frame_at_seconds_after_poke(
            vdata["ff"], vdata["poke_f"], target_s, vdata["n_pages"],
        )
        dt = _seconds_after_poke(fi, vdata["poke_f"], vdata["ff"])
        out_png = out_dir / f"{label}_t{int(round(dt))}s.png"
        if render_static(
            vdata, out_png, mode="neighbor", top_n=top_n, geo=geo,
            neighbor_only=True, short_label=label, frame_idx=fi,
            title=f"{label} — no contact  |  {dt:.0f} s after poke",
        ):
            written.append(out_png)
            print(f"  snapshot → {out_png}")
    return written


def render_timeline_montage(catalog_dir: Path, tiff_index: dict, geo: dict | None,
                            out_png: Path, seconds: tuple[float, ...],
                            top_n: int = 8):
    """Grid: rows = videos, columns = time points after poke."""
    entries = NO_CONTACT_NEIGHBOR
    gt_csv = HERE / "analysis_results" / "xy_ground_truth.csv"
    nrows, ncols = len(entries), len(seconds)
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(4.6 * ncols, 3.6 * nrows), facecolor=BG,
    )
    if nrows == 1 and ncols == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = axes[np.newaxis, :]
    elif ncols == 1:
        axes = axes[:, np.newaxis]

    for ri, (stem, short) in enumerate(entries):
        vdata = load_video_data(catalog_dir, stem, tiff_index, geo, gt_csv)
        for ci, target_s in enumerate(seconds):
            ax = axes[ri, ci]
            ax.set_facecolor(BG)
            if vdata is None:
                ax.axis("off")
                continue
            fi = frame_at_seconds_after_poke(
                vdata["ff"], vdata["poke_f"], target_s, vdata["n_pages"],
            )
            dt = _seconds_after_poke(fi, vdata["poke_f"], vdata["ff"])
            wave_ids = _wave_palette(
                vdata["we"], top_n, vdata["sides"], {"neighbor"},
            )
            col_title = f"{short}  |  {dt:.0f} s"
            if not _draw_neighbor_panel(ax, vdata, fi, geo, wave_ids, col_title):
                ax.text(0.5, 0.5, "no frame", ha="center", va="center",
                        color="white", transform=ax.transAxes)
                ax.axis("off")

    fig.suptitle(
        "No-contact embryos — calcium wave over time (X = poke)",
        color="white", fontsize=13, fontweight="bold", y=1.01,
    )
    fig.text(
        0.5, 0.01, EXPLAIN_NEIGHBOR,
        ha="center", va="bottom", color="#c8ccd8", fontsize=9, style="italic",
    )
    fig.tight_layout(rect=[0, 0.035, 1, 0.97])
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150, facecolor=BG, bbox_inches="tight")
    plt.close(fig)
    return True


def render_no_contact_montage(catalog_dir: Path, tiff_index: dict,
                              geo: dict | None, out_png: Path,
                              top_n: int = 8):
    """Side-by-side montage of the two no-contact neighbor-wave examples."""
    entries = NO_CONTACT_NEIGHBOR
    n = len(entries)
    fig, axes = plt.subplots(1, n, figsize=(7.0 * n, 4.2), facecolor=BG)
    axes = np.atleast_1d(axes).ravel()

    gt_csv = HERE / "analysis_results" / "xy_ground_truth.csv"
    for ax, (stem, short) in zip(axes, entries):
        vdata = load_video_data(catalog_dir, stem, tiff_index, geo, gt_csv)
        ax.set_facecolor(BG)
        if vdata is None or vdata["disp"] is None:
            ax.text(0.5, 0.5, f"missing: {stem}", ha="center", va="center",
                    color="white", transform=ax.transAxes)
            ax.axis("off")
            continue
        fi = vdata["snap_f"]
        rgb = calcium_rgb(vdata["disp"])
        ax.imshow(rgb, origin="upper", interpolation="bilinear")
        ax.axis("off")
        wave_ids = _wave_palette(
            vdata["we"], top_n, vdata["sides"], {"neighbor"},
        )
        draw_frame_vectors(ax, vdata["ff"], wave_ids, fi)
        draw_embryo_context(ax, vdata["prefix"], geo, alpha=0.65)
        dt = _seconds_after_poke(fi, vdata["poke_f"], vdata["ff"])
        ax.set_title(f"{short} — {dt:.0f} s after poke", color="white",
                     fontsize=11, fontweight="bold")

    fig.suptitle(
        "No-contact embryos: calcium wave on the unpoked neighbor",
        color="white", fontsize=13, fontweight="bold", y=1.02,
    )
    fig.text(
        0.5, 0.01, EXPLAIN_NEIGHBOR,
        ha="center", va="bottom", color="#c8ccd8", fontsize=9,
        style="italic",
    )
    fig.tight_layout(rect=[0, 0.04, 1, 0.96])
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150, facecolor=BG, bbox_inches="tight")
    plt.close(fig)
    return True


def render_methods_figure(catalog_dir: Path, tiff_index: dict, geo: dict | None,
                          out_png: Path, stem: str, frame_idx: int | None = None,
                          mag: float = 9.0):
    """Four-stage methods figure (raw -> detect -> vectors -> linked waves).

    Built entirely from existing artifacts:
      * the calcium TIFF frame
      * <stem>/pixel_frame_deltas.csv  (per-pixel detections + 1-frame deltas)
      * <stem>/waves/wave_front_frames.csv  (linked wave fronts)
    """
    gt_csv = HERE / "analysis_results" / "xy_ground_truth.csv"
    vdir = catalog_dir / stem
    deltas_csv = vdir / "pixel_frame_deltas.csv"
    if not deltas_csv.exists():
        print(f"  ! no pixel_frame_deltas.csv for {stem}")
        return False

    vdata = load_video_data(catalog_dir, stem, tiff_index, geo, gt_csv)
    if vdata is None or vdata["tif"] is None:
        print(f"  ! cannot load video {stem}")
        return False

    deltas = pd.read_csv(deltas_csv)
    matched_all = deltas[deltas["matched_from_last"] == True]  # noqa: E712

    if frame_idx is None:
        # busiest matched frame = clearest illustration
        frame_idx = int(matched_all.groupby("frame_idx").size().idxmax())

    det = deltas[deltas["frame_idx"] == frame_idx]
    mat = det[det["matched_from_last"] == True]  # noqa: E712
    if det.empty:
        print(f"  ! no detections at frame {frame_idx}")
        return False

    disp = load_frame_scaled(vdata["tif"], frame_idx)
    rgb = calcium_rgb(disp)

    # Shared crop to the active region (legible pixels + arrows in every panel)
    pad = 80
    x0 = max(0, det["x_curr"].min() - pad)
    x1 = min(rgb.shape[1], det["x_curr"].max() + pad)
    y0 = max(0, det["y_curr"].min() - pad)
    y1 = min(rgb.shape[0], det["y_curr"].max() + pad)

    fig, axes = plt.subplots(4, 1, figsize=(13, 12), facecolor=BG)
    dt = _seconds_after_poke(frame_idx, vdata["poke_f"], vdata["ff"])

    def base(ax, dim=1.0):
        ax.imshow(rgb * dim, origin="upper", interpolation="bilinear")
        ax.set_xlim(x0, x1)
        ax.set_ylim(y1, y0)  # inverted y (image coords)
        ax.axis("off")

    # Panel 1: raw calcium frame (global-linear scale)
    base(axes[0])
    axes[0].set_title(
        "1. Calcium frame — intensities mapped to one global linear 0–1 scale",
        color="white", fontsize=11, fontweight="bold", loc="left",
    )

    # Panel 2: bright-pixel detection (local maxima above threshold)
    base(axes[1], dim=0.55)
    axes[1].scatter(
        det["x_curr"], det["y_curr"],
        c=det["brightness_curr_linear"], cmap="plasma", vmin=0, vmax=1,
        s=10, edgecolors="white", linewidths=0.2, zorder=5,
    )
    axes[1].set_title(
        f"2. Bright-pixel detection — local maxima ≥ threshold "
        f"({len(det)} sites)",
        color="white", fontsize=11, fontweight="bold", loc="left",
    )

    # Panel 3: per-pixel frame-to-frame motion vectors (t-1 -> t), x mag
    base(axes[2], dim=0.55)
    if len(mat):
        spd = mat["speed_px_per_s"].to_numpy(dtype=float)
        norm = Normalize(
            vmin=float(np.nanpercentile(spd, 5)) if len(spd) else 0.0,
            vmax=float(np.nanpercentile(spd, 95)) if len(spd) else 1.0,
        )
        axes[2].quiver(
            mat["x_last"], mat["y_last"],
            mat["delta_x_px"] * mag, -mat["delta_y_px"] * mag,
            spd, cmap=SPEED_CMAP, norm=norm,
            angles="xy", scale_units="xy", scale=1,
            width=0.0016, headwidth=4, headlength=5, alpha=0.9, zorder=5,
        )
    axes[2].set_title(
        f"3. Frame-to-frame motion vectors (t−1 → t, ×{mag:g}) — "
        f"{len(mat)} matched, colour = speed",
        color="white", fontsize=11, fontweight="bold", loc="left",
    )

    # Panel 4: linked waves (fronts grouped + tracked across frames)
    base(axes[3], dim=0.55)
    ff = vdata["ff"]
    ff_upto = ff[ff["frame_idx"] <= frame_idx]
    wave_ids = (
        ff_upto.groupby("wave_id")["total_brightness"].sum()
        .sort_values(ascending=False).head(10).index.tolist()
    )
    cmap = plt.get_cmap("turbo")
    for k, wid in enumerate(wave_ids):
        trk = ff_upto[ff_upto["wave_id"] == wid].sort_values("frame_idx")
        if trk.empty:
            continue
        col = cmap(k / max(1, len(wave_ids) - 1))
        axes[3].plot(trk["x"], trk["y"], "-", color=col, lw=2.0,
                     alpha=0.9, zorder=6, solid_capstyle="round")
        axes[3].scatter([trk["x"].iloc[0]], [trk["y"].iloc[0]], s=45,
                        facecolors="none", edgecolors=col, lw=1.6, zorder=7)
        if len(trk) > 1:
            axes[3].annotate(
                "", xy=(trk["x"].iloc[-1], trk["y"].iloc[-1]),
                xytext=(trk["x"].iloc[-2], trk["y"].iloc[-2]),
                arrowprops=dict(arrowstyle="-|>", color=col, lw=2.0),
                zorder=7,
            )
    axes[3].set_title(
        f"4. Linked Ca²⁺ waves — pixels grouped into fronts, tracked over time "
        f"({len(wave_ids)} shown)",
        color="white", fontsize=11, fontweight="bold", loc="left",
    )

    fig.suptitle(
        f"Wave-vector measurement pipeline — {vdata['prefix']}  "
        f"(frame {frame_idx}, {dt:.0f} s after poke)",
        color="white", fontsize=13, fontweight="bold", y=0.995,
    )
    fig.text(
        0.5, 0.005,
        "Each bright pixel at t is matched to the nearest bright pixel at t−1 "
        "(≤ max link px); displacement/Δt = speed, atan2(−dy,dx) = direction. "
        "Fronts = brightness-weighted pixel clusters; waves = fronts linked across frames.",
        ha="center", va="bottom", color="#c8ccd8", fontsize=8.5, style="italic",
    )
    fig.tight_layout(rect=[0, 0.02, 1, 0.975])
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=160, facecolor=BG, bbox_inches="tight")
    plt.close(fig)
    return True


def render_montage(entries: list[tuple], catalog_dir: Path, tiff_index: dict,
                   geo: dict | None, out_png: Path):
    """entries: [(stem, mode, label), ...]"""
    n = len(entries)
    cols = min(2, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(7.2 * cols, 3.4 * rows),
                             facecolor=BG)
    axes = np.atleast_1d(axes).ravel()
    for ax in axes[n:]:
        ax.axis("off")

    for ax, (stem, mode, label) in zip(axes, entries):
        vdata = load_video_data(
            catalog_dir, stem, tiff_index, geo,
            HERE / "analysis_results" / "xy_ground_truth.csv",
        )
        ax.set_facecolor(BG)
        if vdata is None or vdata["disp"] is None:
            ax.text(0.5, 0.5, f"missing: {stem}", ha="center", va="center",
                    color="white", transform=ax.transAxes)
            ax.axis("off")
            continue
        rgb = calcium_rgb(vdata["disp"])
        ax.imshow(rgb, origin="upper", interpolation="bilinear")
        ax.axis("off")
        draw_embryo_context(ax, vdata["prefix"], geo, alpha=0.35)
        wave_ids = _wave_palette(vdata["we"], 8)
        draw_wave_trails(
            ax, vdata["ff"], vdata["we"], vdata["sides"],
            vdata["axis"], mode, wave_ids, show_vectors=False,
        )
        ax.set_title(label, color="white", fontsize=9)

    fig.suptitle("Calcium wave propagation — representative embryos",
                 color="white", fontsize=13, fontweight="bold", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150, facecolor=BG, bbox_inches="tight")
    plt.close(fig)
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--catalog-dir", type=Path, default=DEFAULT_CATALOG)
    ap.add_argument("--box-root", type=Path, default=Path(DEFAULT_ROOT))
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--video", action="append", default=[],
                    help="Catalog stem(s), e.g. WGD82_C-oHTHT-pT-Zoom5.60")
    ap.add_argument("--batch", action="store_true",
                    help="Render showcase montage + default representative videos")
    ap.add_argument("--no-contact", action="store_true",
                    help="Only no-contact videos with neighbor waves; "
                         "pink neighbor paths only (simpler)")
    ap.add_argument("--mode", choices=["auto", "neighbor", "bidirectional"],
                    default="auto")
    ap.add_argument("--top-n", type=int, default=12,
                    help="Brightest waves to draw")
    ap.add_argument("--no-gif", action="store_true")
    ap.add_argument("--fps", type=float, default=8)
    ap.add_argument(
        "--snapshots", default=",".join(str(s) for s in DEFAULT_SNAPSHOT_SECONDS),
        help="Comma-separated seconds-after-poke for extra PNGs (no-contact)",
    )
    ap.add_argument("--no-snapshots", action="store_true",
                    help="Skip per-time snapshot PNGs and timeline montage")
    ap.add_argument("--methods", action="store_true",
                    help="Render the 4-stage methods figure and exit")
    ap.add_argument("--methods-frame", type=int, default=None,
                    help="Frame index for the methods figure (default: busiest)")
    args = ap.parse_args()

    tiff_index = build_tiff_index(args.box_root)
    gt_csv = HERE / "analysis_results" / "xy_ground_truth.csv"
    geo = load_geometry(gt_csv)

    if args.methods:
        stem = args.video[0] if args.video else "WGD82_C-oHTHT-pT-Zoom5.60"
        out_png = args.output_dir / "methods" / f"{stem.split('_')[0]}_methods.png"
        if render_methods_figure(
            args.catalog_dir, tiff_index, geo, out_png, stem,
            frame_idx=args.methods_frame,
        ):
            print(f"Methods figure → {out_png}")
        else:
            raise SystemExit("Methods figure failed — check pixel_frame_deltas.csv.")
        return

    neighbor_only = args.no_contact
    out_dir = args.output_dir
    if neighbor_only:
        out_dir = out_dir / "no_contact"

    laterality_csv = args.catalog_dir / "wave_laterality.csv"
    short_names = {s: lbl for s, lbl in NO_CONTACT_NEIGHBOR}

    stems = list(args.video)
    if args.no_contact:
        stems.extend(load_no_contact_neighbor_videos(laterality_csv))
    elif args.batch:
        stems.extend(s for s, _, _ in SHOWCASE)
    stems = list(dict.fromkeys(stems))

    if not stems and not args.batch and not args.no_contact:
        ap.error("Pass --video <stem>, --batch, and/or --no-contact")

    ok = 0
    for stem in stems:
        vdata = load_video_data(args.catalog_dir, stem, tiff_index, geo, gt_csv)
        if vdata is None:
            print(f"skip (no waves): {stem}")
            continue
        suffix = "_neighbor" if neighbor_only else "_waves"
        out_png = out_dir / f"{stem}{suffix}.png"
        out_gif = out_dir / f"{stem}{suffix}.gif"
        mode = "neighbor" if neighbor_only else args.mode
        short = short_names.get(stem)
        if render_static(
            vdata, out_png, mode=mode, top_n=args.top_n, geo=geo,
            neighbor_only=neighbor_only, short_label=short,
        ):
            print(f"PNG → {out_png}")
            ok += 1
        if not args.no_gif and render_gif(
            vdata, out_gif, mode=mode, top_n=min(args.top_n, 10),
            geo=geo, fps=args.fps,
            neighbor_only=neighbor_only, short_label=short,
        ):
            print(f"GIF → {out_gif}")

        if neighbor_only and not args.no_snapshots:
            try:
                snap_secs = tuple(
                    float(s.strip()) for s in args.snapshots.split(",") if s.strip()
                )
            except ValueError:
                snap_secs = DEFAULT_SNAPSHOT_SECONDS
            snap_dir = out_dir / "snapshots"
            render_snapshot_series(
                vdata, snap_dir, geo, short, args.top_n, snap_secs,
            )

    if args.batch and not args.no_contact:
        montage_path = out_dir / "showcase_montage.png"
        render_montage(SHOWCASE, args.catalog_dir, tiff_index, geo, montage_path)
        print(f"Montage → {montage_path}")

    if args.no_contact:
        montage_path = out_dir / "no_contact_neighbor_montage.png"
        render_no_contact_montage(
            args.catalog_dir, tiff_index, geo, montage_path,
            top_n=min(args.top_n, 8),
        )
        print(f"Montage → {montage_path}")
        all_six_path = out_dir / "all_six_no_contact.png"
        render_all_six_montage(
            args.catalog_dir, tiff_index, geo, all_six_path,
            laterality_csv, top_n=min(args.top_n, 8),
        )
        print(f"All six → {all_six_path}")
        if not args.no_gif:
            all_six_gif = out_dir / "all_six_no_contact.gif"
            if render_all_six_gif(
                args.catalog_dir, tiff_index, geo, all_six_gif,
                laterality_csv, top_n=min(args.top_n, 8), fps=args.fps,
            ):
                print(f"All six GIF → {all_six_gif}")
        if not args.no_snapshots:
            try:
                snap_secs = tuple(
                    float(s.strip()) for s in args.snapshots.split(",") if s.strip()
                )
            except ValueError:
                snap_secs = DEFAULT_SNAPSHOT_SECONDS
            timeline_path = out_dir / "timeline_montage.png"
            render_timeline_montage(
                args.catalog_dir, tiff_index, geo, timeline_path,
                snap_secs, top_n=min(args.top_n, 8),
            )
            print(f"Timeline → {timeline_path}")

    if ok == 0 and not args.batch and not args.no_contact:
        raise SystemExit("No renders produced — check catalog paths and TIFF access.")
    print("Done.")


if __name__ == "__main__":
    main()

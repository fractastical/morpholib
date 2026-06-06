#!/usr/bin/env python3
"""
Wave laterality + directionality analysis.

Closes the "PARTIAL" claims that only needed a scoring step on top of the
existing wave roll-up (no new microscopy required):

  * Layer 2, claims 1-2  -> bidirectional wave WITHIN the stimulated embryo
  * Layer 4, claims 9-11 -> a calcium WAVE appears in the NEIGHBOR embryo
  * Layer 4, claims 14-16-> a LOCAL response (front speed) in the NEIGHBOR

Why this is possible with current data
--------------------------------------
Every Box video in the wave catalog has:
  * `waves/wave_events.csv`  - one row per detected wave (origin x/y, net
    direction, onset time, brightness, front + propagation speed)
  * a filename that encodes the two-embryo orientation (e.g. `oHTHT`) and the
    poke location (`pT` tail / `pM` mid / `pH` head).

Two embryos lie along one arrangement axis. We:
  1. Fit the principal axis of the wave ORIGINS (brightness-weighted, robust).
  2. Project origins onto that axis and split them into two clusters (the two
     embryos) with 1-D 2-means; keep a separation-quality score.
  3. Call the cluster whose waves start EARLIEST and BRIGHTEST the *stimulated*
     embryo (the poke site lights up first); the other is the *neighbor*.
  4. Score, per video:
       - neighbor_has_wave / n_waves_neighbor / neighbor_mean_*_speed
       - onset_lag_s   (neighbor first wave - stimulated first wave)
       - bidirectional_stim: do stimulated-side waves propagate BOTH ways along
         the embryo axis (two opposing lobes), not just one direction?

Outputs
-------
  wave_laterality.csv     - one row per video with all metrics
  wave_laterality.png     - summary panels used by the claims report

The verdict for each claim is whatever the aggregate shows; the point of this
script is that the claims are now *tested*, not merely testable.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent

# Minimum peak brightness (relative to the brightest wave in a video) for a
# wave to count as a real event rather than detector noise.
NEIGHBOR_REL_BRIGHT = 0.10
# A side counts as "bidirectional" if at least this fraction of its waves go
# each way along the axis.
BIDIR_MIN_FRAC = 0.25


def principal_axis(xy: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Brightness-weighted principal axis of a set of 2-D points."""
    w = weights / weights.sum()
    mean = (xy * w[:, None]).sum(axis=0)
    centered = xy - mean
    cov = (centered.T * w) @ centered
    vals, vecs = np.linalg.eigh(cov)
    axis = vecs[:, int(np.argmax(vals))]
    return mean, axis


def split_two(proj: np.ndarray) -> tuple[np.ndarray, float]:
    """1-D 2-means split. Returns (labels in {0,1}, separation score 0..1).

    Separation score = gap between cluster means / spread of the data; higher
    means the two embryos are cleanly separated along the axis.
    """
    p = proj.astype(float)
    if len(p) < 2 or np.ptp(p) == 0:
        return np.zeros(len(p), dtype=int), 0.0
    # init at the extremes
    c0, c1 = p.min(), p.max()
    labels = np.zeros(len(p), dtype=int)
    for _ in range(50):
        labels = (np.abs(p - c1) < np.abs(p - c0)).astype(int)
        if labels.all() or not labels.any():
            break
        nc0 = p[labels == 0].mean()
        nc1 = p[labels == 1].mean()
        if math.isclose(nc0, c0) and math.isclose(nc1, c1):
            break
        c0, c1 = nc0, nc1
    spread = p.std() if p.std() > 0 else 1.0
    sep = abs(c1 - c0) / (spread + 1e-9)
    # normalize to ~0..1 (sep of ~2 std is a clean split)
    sep = float(min(1.0, sep / 2.0))
    return labels, sep


def _seg_dist(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    """Distance from point p to the head->tail segment a-b."""
    ab = b - a
    denom = float(ab @ ab)
    if denom <= 1e-9:
        return float(np.linalg.norm(p - a))
    t = float(np.clip((p - a) @ ab / denom, 0.0, 1.0))
    return float(np.linalg.norm(p - (a + t * ab)))


def load_geometry(gt_csv: Path) -> dict:
    """Per-video real geometry from the manual XY ground truth.

    Returns {prefix: {"embryos": {"Left": (head_xy, tail_xy), "Right": ...},
                      "poke": xy, "stim_side": "Left"|"Right"}}
    only for two-embryo videos that have a poke and both head/tail pairs.
    """
    if not gt_csv.exists():
        return {}
    try:
        df = pd.read_csv(gt_csv)
    except Exception:
        return {}
    geo = {}
    for pre, sub in df.groupby("prefix"):
        embryos = {}
        for side in ("Left", "Right"):
            s = sub[sub.side == side]
            h = s[s.landmark_class == "head"]
            t = s[s.landmark_class == "tail"]
            if len(h) and len(t):
                embryos[side] = (
                    np.array([float(h.iloc[0].x), float(h.iloc[0].y)]),
                    np.array([float(t.iloc[0].x), float(t.iloc[0].y)]),
                )
        poke = sub[sub.landmark_class.isin(["poke", "pressure"])]
        if len(embryos) == 2 and len(poke):
            pk = np.array([float(poke.iloc[0].x), float(poke.iloc[0].y)])
            # Prefer the poke's own annotated side; fall back to the embryo
            # whose head->tail segment is nearest the poke.
            annot_side = str(poke.iloc[0].side).capitalize()
            stim_side = (annot_side if annot_side in embryos
                         else min(embryos,
                                  key=lambda sd: _seg_dist(pk, *embryos[sd])))
            geo[pre] = {"embryos": embryos, "poke": pk, "stim_side": stim_side}
    return geo


def parse_poke_orientation(video: str) -> tuple[str, str]:
    """Return (orientation, poke) from a name like WG30_C-oHTHT-pT-Zoom6.30."""
    orient, poke = "", ""
    for tok in video.replace(".tif", "").split("-"):
        if tok.startswith("o") and len(tok) > 1 and tok[1:].isalpha():
            orient = tok[1:]
        elif tok.startswith("p") and len(tok) == 2 and tok[1] in "HMT":
            poke = {"H": "head", "M": "mid", "T": "tail"}[tok[1]]
    return orient, poke


def analyse_video(video_dir: Path, geo: dict | None = None) -> dict | None:
    we_path = video_dir / "waves" / "wave_events.csv"
    if not we_path.exists():
        return None
    try:
        we = pd.read_csv(we_path)
    except Exception:
        return None
    if len(we) < 2:
        return None

    xy = we[["x_origin", "y_origin"]].to_numpy(dtype=float)
    bright = we["peak_total_brightness"].to_numpy(dtype=float)
    bright = np.where(np.isfinite(bright) & (bright > 0), bright, 1.0)

    prefix = video_dir.name.split("_")[0]
    vgeo = (geo or {}).get(prefix)
    geom_source = "pca"
    forced_stim_label = None

    if vgeo is not None:
        # --- real geometry: assign each origin to the nearest embryo segment,
        #     stimulated embryo = the one nearest the poke ---
        sides = ["Left", "Right"]
        d_left = np.array([_seg_dist(p, *vgeo["embryos"]["Left"]) for p in xy])
        d_right = np.array([_seg_dist(p, *vgeo["embryos"]["Right"]) for p in xy])
        labels = (d_right < d_left).astype(int)  # 0=Left, 1=Right
        stim_idx = sides.index(vgeo["stim_side"])
        forced_stim_label = stim_idx
        # axis = stimulated embryo's head->tail direction (true body axis)
        h, t = vgeo["embryos"][vgeo["stim_side"]]
        axis = t - h
        axis = axis / (np.linalg.norm(axis) + 1e-9)
        mean = xy.mean(axis=0)
        # separation = how distinctly origins prefer one embryo over the other
        gap = np.abs(d_left - d_right)
        scale = np.linalg.norm(vgeo["embryos"]["Left"][1]
                               - vgeo["embryos"]["Right"][0]) + 1e-9
        sep = float(min(1.0, np.median(gap) / (0.25 * scale)))
        geom_source = "ground_truth"
    else:
        mean, axis = principal_axis(xy, bright)
        proj = (xy - mean) @ axis
        labels, sep = split_two(proj)

    proj = (xy - mean) @ axis

    # Per-cluster onset (earliest) and brightness; stimulated = earliest onset,
    # tie-broken by brighter peak.
    def cluster_stats(mask):
        sub = we[mask]
        if len(sub) == 0:
            return {"n": 0, "t0": float("nan"), "bright": 0.0,
                    "front": float("nan"), "prop": float("nan"),
                    "proj": np.array([])}
        return {
            "n": int(mask.sum()),
            "t0": float(sub["t_start_s"].min()),
            "bright": float(sub["peak_total_brightness"].max()),
            "front": float(np.nanmean(sub["mean_front_speed_px_per_s"])),
            "prop": float(np.nanmean(sub["propagation_speed_px_per_s"])),
            "proj": proj[mask.to_numpy() if hasattr(mask, "to_numpy") else mask],
        }

    m0 = labels == 0
    m1 = labels == 1
    s0, s1 = cluster_stats(m0), cluster_stats(m1)

    # If one cluster is empty (all same label) -> single embryo activity only.
    single = (s0["n"] == 0) or (s1["n"] == 0)

    empty = {"n": 0, "t0": float("nan"), "bright": 0.0,
             "front": float("nan"), "prop": float("nan"), "proj": np.array([])}

    if forced_stim_label is not None:
        # Real geometry fixes which embryo is stimulated (poke side). We keep
        # this assignment even if the poke-side embryo shows no waves: that is
        # a real (and informative) outcome, not a reason to relabel sides.
        if forced_stim_label == 0:
            stim, neigh, stim_mask = s0, s1, m0
        else:
            stim, neigh, stim_mask = s1, s0, m1
        single = (neigh["n"] == 0)
    elif single:
        stim, neigh = (s0 if s0["n"] else s1), empty
        stim_mask = m0 if s0["n"] else m1
    else:
        # earliest onset wins; tie -> brighter
        if (s0["t0"], -s0["bright"]) <= (s1["t0"], -s1["bright"]):
            stim, neigh, stim_mask = s0, s1, m0
        else:
            stim, neigh, stim_mask = s1, s0, m1

    # Bidirectional test on the stimulated side: project stim-side wave NET
    # DIRECTION onto the axis; do waves go both + and - along the axis?
    stim_we = we[stim_mask.to_numpy() if hasattr(stim_mask, "to_numpy") else stim_mask]
    ang = np.deg2rad(stim_we["net_direction_deg"].to_numpy(dtype=float))
    # net_direction_deg uses image convention (y inverted); axis is in x,y px.
    dir_vec = np.column_stack([np.cos(ang), -np.sin(ang)])
    along = dir_vec @ axis
    disp = stim_we["net_displacement_px"].to_numpy(dtype=float)
    moving = disp >= 3.0  # ignore near-stationary waves
    along_m = along[moving]
    n_fwd = int((along_m > 0).sum())
    n_bwd = int((along_m < 0).sum())
    n_dir = n_fwd + n_bwd
    if n_dir >= 2:
        frac_min = min(n_fwd, n_bwd) / n_dir
        bidirectional = frac_min >= BIDIR_MIN_FRAC
    else:
        frac_min = 0.0
        bidirectional = False

    # Neighbor wave must clear a relative-brightness floor to count.
    max_bright = float(we["peak_total_brightness"].max())
    neighbor_has_wave = (not single) and (
        neigh["bright"] >= NEIGHBOR_REL_BRIGHT * max_bright)

    onset_lag = (neigh["t0"] - stim["t0"]) if (not single) else float("nan")

    return {
        "video": video_dir.name,
        "n_waves": int(len(we)),
        "geom_source": geom_source,
        "axis_sep_score": round(sep, 3),
        "single_embryo_activity": int(single),
        "n_waves_stim": stim["n"],
        "n_waves_neighbor": neigh["n"],
        "neighbor_has_wave": int(neighbor_has_wave),
        "stim_peak_brightness": round(stim["bright"], 1),
        "neighbor_peak_brightness": round(neigh["bright"], 1),
        "neighbor_rel_brightness": round(
            neigh["bright"] / max_bright if max_bright else 0.0, 3),
        "stim_mean_front_speed": round(stim["front"], 3),
        "neighbor_mean_front_speed": round(neigh["front"], 3),
        "stim_mean_prop_speed": round(stim["prop"], 3),
        "neighbor_mean_prop_speed": round(neigh["prop"], 3),
        "onset_lag_s": round(onset_lag, 2) if np.isfinite(onset_lag) else "",
        "stim_n_fwd": n_fwd,
        "stim_n_bwd": n_bwd,
        "stim_bidir_frac": round(frac_min, 3),
        "bidirectional_stim": int(bidirectional),
    }


def load_meta_map(catalog_csv: Path) -> dict:
    """Authoritative {video_stem: {stimulus, contact, series}} from the catalog.

    The catalog's contact label is derived from the folder PATH ("no physical
    contact" folders), which the per-video directory name alone cannot recover.
    """
    if not catalog_csv.exists():
        return {}
    try:
        wc = pd.read_csv(catalog_csv)
    except Exception:
        return {}
    out = {}
    for _, r in wc.iterrows():
        stem = str(r["video"]).rsplit(".tif", 1)[0]
        out[stem] = {
            "stimulus": r.get("stimulus", "?"),
            "contact": r.get("contact", "?"),
            "series": r.get("series", ""),
        }
    return out


def build_catalog(catalog_dir: Path, meta_map: dict | None = None,
                  geo: dict | None = None) -> pd.DataFrame:
    meta_map = meta_map or {}
    rows = []
    for vid_dir in sorted(p for p in catalog_dir.iterdir() if p.is_dir()):
        rec = analyse_video(vid_dir, geo=geo)
        if rec is None:
            continue
        orient, poke = parse_poke_orientation(rec["video"])
        prefix = rec["video"].split("_")[0]
        rec["orientation"] = orient
        rec["poke"] = poke
        meta = meta_map.get(rec["video"], {})
        # Prefer the catalog's path-derived stimulus/contact; fall back to name.
        rec["stimulus"] = meta.get("stimulus") or {
            "W": "wound", "P": "press"}.get(prefix[:1], "?")
        rec["series"] = meta.get("series") or prefix[1:2]
        contact = meta.get("contact")
        if contact in (None, "", "?"):
            # WGD* = wound no-contact ("distance") series; else assume contact.
            contact = "no-contact" if prefix[:3] == "WGD" else "contact"
        rec["contact"] = contact
        rows.append(rec)
    return pd.DataFrame(rows)


def render_summary(df: pd.DataFrame, out_png: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    # 1) neighbor-wave prevalence by stimulus
    ax = axes[0, 0]
    grp = df.groupby("stimulus")["neighbor_has_wave"].mean() * 100
    cnts = df.groupby("stimulus")["neighbor_has_wave"].count()
    bars = ax.bar(grp.index, grp.values, color=["#d95f0e", "#2c7fb8", "#999999"][:len(grp)])
    for b, k in zip(bars, grp.index):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1,
                f"{grp[k]:.0f}%\n(n={cnts[k]})", ha="center", fontsize=9)
    ax.set_title("Claims 9-11: wave detected in NEIGHBOR embryo")
    ax.set_ylabel("% of videos with a neighbor-side wave")
    ax.set_ylim(0, 110)

    # 2) neighbor vs stim front speed (local response) - claims 14-16
    ax = axes[0, 1]
    sub = df[df["neighbor_has_wave"] == 1]
    ax.scatter(sub["stim_mean_front_speed"], sub["neighbor_mean_front_speed"],
               c=sub["stimulus"].map({"wound": "#d95f0e", "press": "#2c7fb8"}).fillna("#999"),
               s=30, alpha=0.7)
    lim = np.nanmax([df["stim_mean_front_speed"].max(),
                     df["neighbor_mean_front_speed"].max(), 1])
    ax.plot([0, lim], [0, lim], "--", color="#888", lw=1)
    ax.set_title("Claims 14-16: NEIGHBOR local (front) speed")
    ax.set_xlabel("stimulated-side front speed (px/frame)")
    ax.set_ylabel("neighbor-side front speed (px/frame)")

    # 3) bidirectional prevalence by stimulus - claims 1,2
    ax = axes[1, 0]
    bd = df.groupby("stimulus")["bidirectional_stim"].mean() * 100
    bc = df.groupby("stimulus")["bidirectional_stim"].count()
    bars = ax.bar(bd.index, bd.values, color=["#756bb1", "#31a354", "#999"][:len(bd)])
    for b, k in zip(bars, bd.index):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1,
                f"{bd[k]:.0f}%\n(n={bc[k]})", ha="center", fontsize=9)
    ax.set_title("Claims 1-2: BIDIRECTIONAL wave in stimulated embryo")
    ax.set_ylabel("% of videos with bidirectional spread")
    ax.set_ylim(0, 110)

    # 4) onset lag distribution (neighbor lights up after stim)
    ax = axes[1, 1]
    lag = pd.to_numeric(df["onset_lag_s"], errors="coerce").dropna()
    if len(lag):
        ax.hist(lag, bins=20, color="#2c7fb8")
        ax.axvline(0, color="#d62728", lw=1.5, ls="--")
        ax.set_title(f"Neighbor onset lag (median {lag.median():.1f} frames)")
    else:
        ax.set_title("Neighbor onset lag (no paired data)")
    ax.set_xlabel("neighbor first wave - stimulated first wave (frames)")
    ax.set_ylabel("# videos")

    fig.suptitle(f"Wave laterality + directionality  (n={len(df)} videos)",
                 fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def render_conditions(df: pd.DataFrame, out_png: Path) -> None:
    """Wound-vs-press and contact-vs-no-contact neighbor response.

    Backs Layer-3 claims 7-8 (wounding increases neighbor response, pressure
    does not) and Layer-4 claim 12 (signaling without physical contact).
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))

    # 1) neighbor peak brightness: wound vs press
    ax = axes[0]
    order = [s for s in ("wound", "press") if s in df["stimulus"].unique()]
    data = [df[df["stimulus"] == s]["neighbor_peak_brightness"].dropna().values
            for s in order]
    ax.boxplot(data, tick_labels=order, showfliers=False)
    for i, s in enumerate(order, 1):
        ys = df[df["stimulus"] == s]["neighbor_peak_brightness"].dropna().values
        ax.scatter(np.random.normal(i, 0.05, len(ys)), ys, s=18, alpha=0.6,
                   color="#d95f0e" if s == "wound" else "#2c7fb8")
    ax.set_title("Claims 7-8: neighbor response\n(wound vs press)")
    ax.set_ylabel("neighbor peak brightness")

    # 2) neighbor-wave prevalence: wound vs press
    ax = axes[1]
    prev = df.groupby("stimulus")["neighbor_has_wave"].mean() * 100
    cnt = df.groupby("stimulus")["neighbor_has_wave"].count()
    order2 = [s for s in ("wound", "press") if s in prev.index]
    bars = ax.bar(order2, [prev[s] for s in order2],
                  color=["#d95f0e", "#2c7fb8"][:len(order2)])
    for b, s in zip(bars, order2):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1,
                f"{prev[s]:.0f}%\n(n={cnt[s]})", ha="center", fontsize=9)
    ax.set_title("Neighbor wave prevalence\n(wound vs press)")
    ax.set_ylabel("% videos with neighbor wave")
    ax.set_ylim(0, 110)

    # 3) contact vs no-contact (claim 12)
    ax = axes[2]
    corder = [c for c in ("contact", "no-contact") if c in df["contact"].unique()]
    cprev = df.groupby("contact")["neighbor_has_wave"].mean() * 100
    ccnt = df.groupby("contact")["neighbor_has_wave"].count()
    bars = ax.bar(corder, [cprev[c] for c in corder],
                  color=["#31a354", "#756bb1"][:len(corder)])
    for b, c in zip(bars, corder):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1,
                f"{cprev[c]:.0f}%\n(n={ccnt[c]})", ha="center", fontsize=9)
    ax.set_title("Claim 12: neighbor wave\n(contact vs no-contact)")
    ax.set_ylabel("% videos with neighbor wave")
    ax.set_ylim(0, 110)

    fig.suptitle("Neighbor response by condition", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--catalog-dir",
                    default=str(HERE / "analysis_results" / "wave_catalog"),
                    help="dir containing per-video subfolders with waves/")
    ap.add_argument("--catalog-csv",
                    default=str(HERE / "analysis_results" / "wave_catalog"
                               / "wave_catalog.csv"),
                    help="authoritative per-video meta (stimulus/contact)")
    ap.add_argument("--out-csv",
                    default=str(HERE / "analysis_results" / "wave_catalog"
                               / "wave_laterality.csv"))
    ap.add_argument("--out-png",
                    default=str(HERE / "analysis_results" / "wave_catalog"
                               / "wave_laterality.png"))
    ap.add_argument("--out-conditions-png",
                    default=str(HERE / "analysis_results" / "wave_catalog"
                               / "wave_laterality_conditions.png"))
    ap.add_argument("--ground-truth",
                    default=str(HERE / "analysis_results" / "xy_ground_truth.csv"),
                    help="manual XY ground truth; enables real poke + "
                         "head/tail geometry side-split (else PCA fallback)")
    args = ap.parse_args()

    cat_dir = Path(args.catalog_dir)
    meta_map = load_meta_map(Path(args.catalog_csv))
    geo = load_geometry(Path(args.ground_truth))
    df = build_catalog(cat_dir, meta_map, geo=geo)
    if df.empty:
        raise SystemExit(f"No per-video wave data found under {cat_dir}")

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    render_summary(df, Path(args.out_png))
    render_conditions(df, Path(args.out_conditions_png))

    # Console verdict summary
    def pct(col):
        return 100.0 * df[col].mean()
    print(f"Analysed {len(df)} videos -> {out_csv}")
    if "geom_source" in df:
        ng = int((df["geom_source"] == "ground_truth").sum())
        print(f"  Side-split source     : {ng} via real poke+head/tail "
              f"geometry, {len(df) - ng} via PCA fallback")
    print(f"  Neighbor wave present : {pct('neighbor_has_wave'):.0f}% of videos")
    print(f"  Bidirectional (stim)  : {pct('bidirectional_stim'):.0f}% of videos")
    for stim, g in df.groupby("stimulus"):
        print(f"   [{stim:5}] neighbor-wave {100*g['neighbor_has_wave'].mean():.0f}% | "
              f"neighbor-bright(med) {g['neighbor_peak_brightness'].median():.0f} | "
              f"bidir {100*g['bidirectional_stim'].mean():.0f}% | n={len(g)}")
    for con, g in df.groupby("contact"):
        print(f"   [{con:10}] neighbor-wave {100*g['neighbor_has_wave'].mean():.0f}% | "
              f"n={len(g)}")
    print(f"  Wrote {args.out_png}, {args.out_conditions_png}")


if __name__ == "__main__":
    main()

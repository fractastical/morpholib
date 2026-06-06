#!/usr/bin/env python3
"""
Generate no-mask claims PDF from Box calcium videos + XY workbook.

Key rule from user:
- Do not create embryo masks.
- Use all pixels from the relevant embryo side region (left/right split),
  or all frame pixels for single-embryo recordings.
"""

from __future__ import annotations

import argparse
import math
import re
from dataclasses import dataclass
from pathlib import Path
from textwrap import wrap
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import tifffile as tiff
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import Normalize
from scipy import stats


@dataclass
class SheetEntry:
    sheet_name: str
    background_video: str
    calcium_video: str
    coords_b: dict
    coords_c: dict


def normalize_name(s: str) -> str:
    s = str(s).strip()
    s = re.sub(r"\s+", " ", s)
    return s


def parse_side(s: str) -> Optional[str]:
    low = str(s).lower()
    if "left" in low:
        return "Left"
    if "right" in low:
        return "Right"
    return None


def parse_coords_block(df: pd.DataFrame, start_row: int) -> dict:
    out = {}
    i = start_row + 1
    while i < len(df):
        row = df.iloc[i].tolist()
        low_line = " ".join("" if pd.isna(x) else str(x) for x in row).lower()
        if not low_line.strip() or "results" in low_line or "video" in low_line:
            break

        if len(row) >= 5:
            side_raw = "" if pd.isna(row[1]) else str(row[1])
            loc_raw = "" if pd.isna(row[2]) else str(row[2])
            x_raw, y_raw = row[3], row[4]
        else:
            side_raw = ""
            loc_raw = "" if pd.isna(row[1]) else str(row[1])
            x_raw, y_raw = row[2], row[3]

        try:
            x = float(x_raw)
            y = float(y_raw)
        except Exception:
            i += 1
            continue

        side = parse_side(side_raw) or "Single"
        blob = f"{side_raw} {loc_raw}".lower()
        out.setdefault(side, {})
        if "poke" in blob or "wound" in blob:
            out[side]["Poke"] = (x, y)
        elif "head" in blob:
            out[side]["Head"] = (x, y)
        elif "tail" in blob:
            out[side]["Tail"] = (x, y)
        i += 1
    return out


def parse_workbook(xy_path: Path) -> List[SheetEntry]:
    xl = pd.ExcelFile(xy_path)
    entries: List[SheetEntry] = []
    for sheet in xl.sheet_names:
        df = pd.read_excel(xy_path, sheet_name=sheet, header=None)
        col0 = df.iloc[:, 0].astype(str).str.strip()
        b_rows = col0[col0.str.lower() == "background video"].index.tolist()
        c_rows = col0[col0.str.lower() == "calcium video"].index.tolist()
        if not b_rows or not c_rows:
            continue
        bi, ci = int(b_rows[0]), int(c_rows[0])

        b_name, c_name = "", ""
        for r in range(bi + 1, min(len(df), bi + 6)):
            v = df.iloc[r, 0]
            if pd.notna(v) and str(v).strip():
                b_name = normalize_name(v)
                break
        for r in range(ci + 1, min(len(df), ci + 6)):
            v = df.iloc[r, 0]
            if pd.notna(v) and str(v).strip():
                c_name = normalize_name(v)
                break
        if not b_name or not c_name:
            continue

        frame_rows = df.index[df.iloc[:, 0].astype(str).str.strip().str.lower() == "frame"].tolist()
        b_frame = next((r for r in frame_rows if bi < r < ci), None)
        c_frame = next((r for r in frame_rows if r > ci), None)
        if b_frame is None or c_frame is None:
            continue

        coords_b = parse_coords_block(df, int(b_frame))
        coords_c = parse_coords_block(df, int(c_frame))
        entries.append(SheetEntry(sheet, b_name, c_name, coords_b, coords_c))
    return entries


def build_tiff_index(data_root: Path) -> Dict[str, Path]:
    idx: Dict[str, Path] = {}
    for p in data_root.rglob("*"):
        if p.is_file() and p.suffix.lower() in (".tif", ".tiff"):
            idx.setdefault(p.name.lower(), p)
            idx.setdefault(p.stem.lower(), p)
    return idx


def find_tiff(name: str, idx: Dict[str, Path]) -> Optional[Path]:
    raw = normalize_name(name)
    cands = [raw.lower(), Path(raw).name.lower(), Path(raw).stem.lower()]
    for c in cands:
        if c in idx:
            return idx[c]
    return None


def parse_condition(video_name: str) -> str:
    m = re.match(r"([A-Za-z]{2})\d+_", Path(video_name).stem)
    return m.group(1).upper() if m else "UNK"


def parse_orientation(video_name: str) -> str:
    m = re.search(r"-o([A-Za-z]+)", Path(video_name).stem, flags=re.IGNORECASE)
    return m.group(1) if m else "UNK"


def side_masks(shape: Tuple[int, int], coords: dict) -> Dict[str, np.ndarray]:
    h, w = shape
    yy, xx = np.indices((h, w))
    if "Left" in coords and "Right" in coords and "Head" in coords["Left"] and "Head" in coords["Right"]:
        xl = 0.5 * (coords["Left"]["Head"][0] + coords["Left"].get("Tail", coords["Left"]["Head"])[0])
        xr = 0.5 * (coords["Right"]["Head"][0] + coords["Right"].get("Tail", coords["Right"]["Head"])[0])
        mid = 0.5 * (xl + xr)
        return {"Left": xx < mid, "Right": xx >= mid}
    return {"Single": np.ones((h, w), dtype=bool)}


def load_trace_and_mean(video_path: Path, masks: Dict[str, np.ndarray]) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    traces = {k: [] for k in masks}
    sum_img = None
    count = 0
    with tiff.TiffFile(video_path) as tf:
        for p in tf.pages:
            arr = p.asarray()
            if arr.ndim == 3:
                arr = arr[..., 0]
            frame = np.array(arr).astype(float)
            if sum_img is None:
                sum_img = np.zeros_like(frame, dtype=float)
            sum_img += frame
            count += 1
            for k, m in masks.items():
                vals = frame[m]
                traces[k].append(float(np.mean(vals)) if vals.size else np.nan)
    mean_img = sum_img / max(count, 1)
    return {k: np.array(v, dtype=float) for k, v in traces.items()}, mean_img


def project_ap(point: Tuple[float, float], head: Tuple[float, float], tail: Tuple[float, float]) -> float:
    p = np.array(point, dtype=float)
    h = np.array(head, dtype=float)
    t = np.array(tail, dtype=float)
    axis = t - h
    den = float(np.dot(axis, axis))
    if den < 1e-9:
        return float("nan")
    return float(np.dot(p - h, axis) / den)


def render_claim5_speed_page(pdf, wave_catalog_path: Path) -> None:
    """Claim 5: distinguish wave (propagation) speed from local (front) speed,
    using the rolled-up wave catalog. Also compares contact vs no-contact."""
    if not wave_catalog_path.exists():
        fig = plt.figure(figsize=(11, 8.5))
        fig.text(0.04, 0.95, "Claim 5 \u2014 Wave vs local speed", fontsize=16, va="top")
        fig.text(
            0.04, 0.86,
            "Wave catalog not found:\n"
            f"  {wave_catalog_path}\n\n"
            "Build it with:\n"
            "  python wave-vector-analysis/batch_wave_catalog.py\n\n"
            "It runs dense pixel vectors + the wave roll-up on every calcium "
            "video and writes wave_catalog.csv with two speeds per wave:\n"
            "  - propagation_speed_px_per_s  (centroid motion = the wave)\n"
            "  - mean_front_speed_px_per_s   (per-pixel motion = local flicker)",
            fontsize=11,
        )
        pdf.savefig(fig)
        plt.close(fig)
        return

    df = pd.read_csv(wave_catalog_path)
    for col in ("propagation_speed_px_per_s", "mean_front_speed_px_per_s",
                "duration_s"):
        df[col] = pd.to_numeric(df.get(col), errors="coerce")

    prop = df["propagation_speed_px_per_s"].dropna().to_numpy()
    front = df["mean_front_speed_px_per_s"].dropna().to_numpy()

    fig = plt.figure(figsize=(11, 8.5))
    fig.suptitle("Claim 5 \u2014 Wave (propagation) vs local (front) speed",
                 fontsize=15, fontweight="bold")
    gs = fig.add_gridspec(2, 2, hspace=0.42, wspace=0.28,
                          left=0.08, right=0.96, top=0.88, bottom=0.30)

    # Panel 1: propagation vs front speed (the core distinction)
    ax = fig.add_subplot(gs[0, 0])
    bp = ax.boxplot([prop, front],
                    tick_labels=["propagation\n(centroid)", "front\n(per-pixel)"],
                    showfliers=False)
    for i, ys in enumerate([prop, front], start=1):
        xs = np.random.RandomState(0).normal(i, 0.05, size=len(ys))
        ax.scatter(xs, ys, s=12, alpha=0.45, color="#1f77b4")
    ax.set_ylabel("speed (px/frame)")
    ax.set_title("Two speeds per wave")

    # Panel 2: per-pixel scatter prop vs front
    ax = fig.add_subplot(gs[0, 1])
    paired = df.dropna(subset=["propagation_speed_px_per_s",
                               "mean_front_speed_px_per_s"])
    ax.scatter(paired["mean_front_speed_px_per_s"],
               paired["propagation_speed_px_per_s"],
               s=16, alpha=0.5, color="#31a354")
    lim = max(1.0, paired[["propagation_speed_px_per_s",
                           "mean_front_speed_px_per_s"]].to_numpy().max())
    ax.plot([0, lim], [0, lim], "--", color="gray", lw=1)
    ax.set_xlabel("front speed (px/frame)")
    ax.set_ylabel("propagation speed (px/frame)")
    ax.set_title("Wave vs local, per event")

    # Panel 3: contact vs no-contact propagation speed
    ax = fig.add_subplot(gs[1, 0])
    groups, labels = [], []
    if "contact" in df.columns:
        for key in ("contact", "no-contact"):
            vals = df.loc[df["contact"] == key,
                          "propagation_speed_px_per_s"].dropna().to_numpy()
            if len(vals):
                groups.append(vals)
                labels.append(f"{key}\n(n={len(vals)})")
    if groups:
        ax.boxplot(groups, tick_labels=labels, showfliers=False)
        for i, ys in enumerate(groups, start=1):
            xs = np.random.RandomState(1).normal(i, 0.05, size=len(ys))
            ax.scatter(xs, ys, s=12, alpha=0.45, color="#d95f0e")
    ax.set_ylabel("propagation speed (px/frame)")
    ax.set_title("Contact vs no-contact")

    # Panel 4: propagation speed by series (WG/WS/PG/PS)
    ax = fig.add_subplot(gs[1, 1])
    if "prefix" in df.columns:
        series2 = df["prefix"].astype(str).str[:2]
        keys = sorted(series2.dropna().unique())
        data = [(k, df.loc[series2 == k,
                           "propagation_speed_px_per_s"].dropna().to_numpy())
                for k in keys]
        data = [(k, d) for k, d in data if len(d)]
        if data:
            ax.boxplot([d for _, d in data],
                       tick_labels=[f"{k}\n(n={len(d)})" for k, d in data],
                       showfliers=False)
    ax.set_ylabel("propagation speed (px/frame)")
    ax.set_title("By video series")

    n_waves = len(df)
    n_vids = df["video"].nunique() if "video" in df.columns else "?"
    med_prop = float(np.median(prop)) if len(prop) else float("nan")
    med_front = float(np.median(front)) if len(front) else float("nan")
    summary = (
        f"n = {n_waves} waves across {n_vids} videos.  "
        f"Median propagation (wave) speed = {med_prop:.2f} px/frame;  "
        f"median front (local) speed = {med_front:.2f} px/frame.\n"
        "Wave = motion of the bright front's CENTROID across frames (bulk "
        "propagation). Local/front = mean per-pixel motion inside the front "
        "(flicker/expansion). These are measured, distinct quantities; "
        "speeds are px/frame because true frame rates are not stored in the "
        "stacks (use the same fps when comparing across videos)."
    )
    fig.text(0.08, 0.04, "\n".join(wrap(summary, 105)), fontsize=9,
             va="bottom", color="#333333")
    pdf.savefig(fig)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="No-mask claims PDF generator.")
    ap.add_argument("--xy-path", default="/Users/jdietz/Library/CloudStorage/Box-Box/Calcium videos/XY coordinates.xlsx")
    ap.add_argument("--data-root", default="/Users/jdietz/Library/CloudStorage/Box-Box/Calcium videos/Calcium videos")
    ap.add_argument("--output-pdf", default="wave-vector-analysis/analysis_results/claims_nomask_report.pdf")
    ap.add_argument("--alpha", type=float, default=0.05, help="Significance threshold for basic claim verdicts.")
    ap.add_argument("--wave-catalog",
                    default="wave-vector-analysis/analysis_results/wave_catalog/wave_catalog.csv",
                    help="Rolled-up wave events used for the Claim 5 speed page.")
    args = ap.parse_args()

    xy_path = Path(args.xy_path)
    data_root = Path(args.data_root)
    output_pdf = Path(args.output_pdf)
    wave_catalog_path = Path(args.wave_catalog)
    alpha = float(args.alpha)
    output_pdf.parent.mkdir(parents=True, exist_ok=True)

    entries = parse_workbook(xy_path)
    idx = build_tiff_index(data_root)

    claim1_rows = []  # (id, cond, stim_peak, stim_mean_delta)
    claim2_rows = []  # (id, cond, neighbor_peak)
    claim3_rows = []  # (id, orient, alignment_error)
    claim4_rows = []  # (id, contact_group, neighbor_peak)
    brightness_rows = []  # (id, stim_class, total_brightness, mean_brightness)

    per_pair = []

    for e in entries:
        b_path = find_tiff(e.background_video, idx)
        c_path = find_tiff(e.calcium_video, idx)
        if not b_path or not c_path:
            continue

        # Use C coords first (poke usually in C), fallback to B.
        coords = e.coords_c if e.coords_c else e.coords_b
        try:
            with tiff.TiffFile(c_path) as tf:
                f0 = tf.pages[0].asarray()
                if f0.ndim == 3:
                    f0 = f0[..., 0]
                masks = side_masks(f0.shape[:2], coords)
        except Exception:
            continue

        try:
            b_tr, b_mean = load_trace_and_mean(b_path, masks)
            c_tr, c_mean = load_trace_and_mean(c_path, masks)
        except Exception:
            continue

        dff_b, dff_c = {}, {}
        for side in masks:
            f0_side = float(np.nanmean(b_tr[side]))
            if not np.isfinite(f0_side) or abs(f0_side) < 1e-9:
                continue
            dff_b[side] = (b_tr[side] - f0_side) / f0_side
            dff_c[side] = (c_tr[side] - f0_side) / f0_side
        if not dff_c:
            continue

        cond = parse_condition(e.calcium_video)
        orient = parse_orientation(e.calcium_video)

        poke_side = None
        for side_name, meta in (coords or {}).items():
            if "Poke" in meta:
                poke_side = side_name
                break
        stim_side = poke_side if poke_side in dff_c else ("Single" if "Single" in dff_c else None)
        if stim_side is None and dff_c:
            stim_side = sorted(dff_c.keys())[0]
        neighbor_side = None
        if stim_side == "Left" and "Right" in dff_c:
            neighbor_side = "Right"
        elif stim_side == "Right" and "Left" in dff_c:
            neighbor_side = "Left"

        pair_id = Path(e.calcium_video).stem

        # Total RAW brightness in the stimulated region (not normalized), grouped
        # by stimulus type: wounded/poked (W*) vs pressed (P*).
        if stim_side in c_tr:
            stim_class = None
            if cond.startswith("W"):
                stim_class = "Wounded (poke)"
            elif cond.startswith("P"):
                stim_class = "Pressed"
            if stim_class is not None:
                npix = int(masks[stim_side].sum())
                region_trace = c_tr[stim_side]
                mean_bright = float(np.nanmean(region_trace))
                total_bright = float(np.nansum(region_trace) * npix)
                if np.isfinite(mean_bright):
                    brightness_rows.append((pair_id, stim_class, total_bright, mean_bright))

        # Claim 1: stimulated embryo response for PS/WS, single-embryo setup (oHT)
        if cond in {"PS", "WS"} and orient == "HT" and stim_side in dff_c:
            stim_peak = float(np.nanmax(dff_c[stim_side]))
            stim_mean_delta = float(np.nanmean(dff_c[stim_side]) - np.nanmean(dff_b.get(stim_side, np.array([np.nan]))))
            claim1_rows.append((pair_id, cond, stim_peak, stim_mean_delta))

        # Claim 2: neighbor peak for PG/WG and HTHT orientation
        if cond in {"PG", "WG"} and orient in {"HTHT", "HTTH", "THHT"} and neighbor_side in dff_c:
            neighbor_peak = float(np.nanmax(dff_c[neighbor_side]))
            claim2_rows.append((pair_id, cond, orient, neighbor_peak))

        # Claim 4: contact vs no-contact across ALL pairs that have a neighbor
        # response and a contact label (any condition/orientation). The contact
        # status comes from the folder path, independent of the filename's
        # condition/orientation tokens (so WGD* "no-contact" videos are included).
        pstr = str(c_path).lower()
        contact_group = (
            "no-contact" if "no physical contact" in pstr
            else ("contact" if "physical contact" in pstr else "unknown")
        )
        if contact_group in {"contact", "no-contact"} and neighbor_side in dff_c:
            neighbor_peak_any = float(np.nanmax(dff_c[neighbor_side]))
            if np.isfinite(neighbor_peak_any):
                claim4_rows.append((pair_id, contact_group, neighbor_peak_any))

        # Claim 3: positional alignment (orientation THHT/HTTH/HTHT)
        if orient in {"THHT", "HTTH", "HTHT"} and stim_side in {"Left", "Right"} and neighbor_side in {"Left", "Right"}:
            stim_meta = coords.get(stim_side, {})
            neigh_meta = coords.get(neighbor_side, {})
            if "Poke" in stim_meta and "Head" in stim_meta and "Tail" in stim_meta and "Head" in neigh_meta and "Tail" in neigh_meta:
                # Position of max C-B in neighbor side.
                delta = c_mean - b_mean
                nmask = masks[neighbor_side]
                if np.any(nmask):
                    vals = np.where(nmask, delta, -np.inf)
                    yi, xi = np.unravel_index(np.argmax(vals), vals.shape)
                    wound_ap = project_ap(stim_meta["Poke"], stim_meta["Head"], stim_meta["Tail"])
                    neigh_ap = project_ap((float(xi), float(yi)), neigh_meta["Head"], neigh_meta["Tail"])
                    if np.isfinite(wound_ap) and np.isfinite(neigh_ap):
                        claim3_rows.append((pair_id, orient, abs(neigh_ap - wound_ap)))

        per_pair.append((pair_id, cond, orient, b_path, c_path, dff_b, dff_c, stim_side, neighbor_side))

    def get_vals(rows, idx_cond, cond, idx_val):
        return [r[idx_val] for r in rows if r[idx_cond] == cond and np.isfinite(r[idx_val])]

    def get_pairs(rows, idx_cond, cond, idx_val, idx_id=0):
        """Return [(value, pair_id), ...] for one condition (finite values only)."""
        return [
            (r[idx_val], r[idx_id])
            for r in rows
            if r[idx_cond] == cond and np.isfinite(r[idx_val])
        ]

    def overlay_strip(ax, datasets, positions, labels=None, seed=0,
                      point_size=22, caption=True):
        """
        Overlay jittered per-video points on a boxplot so individual videos are
        visible (not just the aggregate box). `datasets` is a list of value-lists
        aligned with `positions`; `labels` (optional) is a list of id-lists.
        """
        rng = np.random.default_rng(seed)
        for di, (data, pos) in enumerate(zip(datasets, positions)):
            if not data:
                continue
            xs = pos + (rng.random(len(data)) - 0.5) * 0.16
            ax.scatter(
                xs, data, s=point_size, color="#1f77b4", alpha=0.75,
                edgecolors="white", linewidths=0.4, zorder=3,
            )
            if labels is not None and di < len(labels) and labels[di] is not None:
                for x, y, lab in zip(xs, data, labels[di]):
                    ax.annotate(
                        str(lab), (x, y), fontsize=4.5, alpha=0.6,
                        xytext=(2, 2), textcoords="offset points", zorder=4,
                    )
        if caption:
            ax.text(
                0.99, 0.01,
                "Dots = individual video pairs (jittered); box = aggregate distribution.",
                transform=ax.transAxes, ha="right", va="bottom",
                fontsize=7, color="#555555",
            )

    # ---- Precompute claim data + verdict categories (shared by dashboard) ----
    d1_ps = get_pairs(claim1_rows, 1, "PS", 2)
    d1_ws = get_pairs(claim1_rows, 1, "WS", 2)
    ps_v = [v for v, _ in d1_ps]
    ws_v = [v for v, _ in d1_ws]
    d2_pg = get_pairs(claim2_rows, 1, "PG", 3)
    d2_wg = get_pairs(claim2_rows, 1, "WG", 3)
    pg_v = [v for v, _ in d2_pg]
    wg_v = [v for v, _ in d2_wg]
    d3_by = {}
    for _pid, _orient, _err in claim3_rows:
        if np.isfinite(_err):
            d3_by.setdefault(_orient, []).append(_err)
    d3_labels = sorted(d3_by)
    d3_vals = [d3_by[k] for k in d3_labels]
    d4_contact = [r[2] for r in claim4_rows if r[1] == "contact" and np.isfinite(r[2])]
    d4_nocontact = [r[2] for r in claim4_rows if r[1] == "no-contact" and np.isfinite(r[2])]

    def _cat1():
        if ps_v and ws_v:
            p_ps = stats.ttest_1samp(ps_v, 0.0, nan_policy="omit").pvalue
            p_ws = stats.ttest_1samp(ws_v, 0.0, nan_policy="omit").pvalue
            p_bt = stats.ttest_ind(ws_v, ps_v, equal_var=False, nan_policy="omit").pvalue
            return "SUPPORTS" if (p_ps < alpha and p_ws < alpha and p_bt >= alpha) else "INCONCLUSIVE"
        return "INSUFFICIENT"

    def _cat2():
        if pg_v and wg_v:
            p_pg = stats.ttest_1samp(pg_v, 0.0, nan_policy="omit").pvalue
            p_wg = stats.ttest_1samp(wg_v, 0.0, nan_policy="omit").pvalue
            p_bt = stats.ttest_ind(wg_v, pg_v, equal_var=False, nan_policy="omit").pvalue
            return "SUPPORTS" if (p_wg < alpha and p_pg >= alpha and p_bt < alpha) else "INCONCLUSIVE"
        return "INSUFFICIENT"

    def _cat4():
        if d4_contact and d4_nocontact:
            p = stats.ttest_ind(d4_nocontact, d4_contact, equal_var=False, nan_policy="omit").pvalue
            return "SUPPORTS" if p >= alpha else "INCONCLUSIVE"
        return "INSUFFICIENT"

    def _cat5():
        """Wave (propagation) vs local (front) speed are measured and distinct."""
        if not wave_catalog_path.exists():
            return "PENDING", None
        try:
            wdf = pd.read_csv(wave_catalog_path)
            prop = pd.to_numeric(wdf["propagation_speed_px_per_s"], errors="coerce")
            front = pd.to_numeric(wdf["mean_front_speed_px_per_s"], errors="coerce")
            paired = pd.DataFrame({"p": prop, "f": front}).dropna()
            n = len(paired)
            if n < 3:
                return "INSUFFICIENT", n
            p = stats.wilcoxon(paired["p"], paired["f"]).pvalue
            return ("SUPPORTS" if p < alpha else "DESCRIPTIVE"), n
        except Exception:  # noqa: BLE001
            return "INSUFFICIENT", None

    _c5_cat, _c5_n = _cat5()

    statuses = [
        ("Claim 1", "Pressure & wounding both respond within stimulated embryo",
         _cat1(), len(ps_v) + len(ws_v)),
        ("Claim 2", "Wounding (not pressure) responds in neighbor",
         _cat2(), len(pg_v) + len(wg_v)),
        ("Claim 3", "Inter-embryo communication is positionally informative",
         "DESCRIPTIVE" if d3_labels else "INSUFFICIENT",
         sum(len(v) for v in d3_vals)),
        ("Claim 4", "Signal does not require physical contact (all non-touching pairs)",
         _cat4(), len(d4_contact) + len(d4_nocontact)),
        ("Claim 5", "Wave (propagation) vs local (front) speed are distinct",
         _c5_cat, _c5_n),
    ]
    status_colors = {
        "SUPPORTS": "#2a7a2a",
        "INCONCLUSIVE": "#c87f0a",
        "INSUFFICIENT": "#aa2a2a",
        "DESCRIPTIVE": "#1f6fb2",
        "PENDING": "#777777",
    }

    with PdfPages(output_pdf) as pdf:
        # ---- Summary dashboard (first slide) ----
        fig = plt.figure(figsize=(11, 8.5))
        gs = fig.add_gridspec(3, 3, hspace=0.5, wspace=0.3,
                              left=0.06, right=0.97, top=0.88, bottom=0.07)
        fig.suptitle("Calcium Claims — Summary Dashboard", fontsize=16, fontweight="bold")
        fig.text(0.06, 0.915,
                 f"Pairs analyzed: {len(per_pair)}   |   "
                 f"Method: all-pixel side-region $\\Delta F/F_0$ ($F_0$ from B)   |   "
                 f"alpha={alpha:.2f}",
                 fontsize=9, color="#444444")

        # Claim 1 mini
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.boxplot([ps_v, ws_v], tick_labels=["PS", "WS"])
        overlay_strip(ax1, [ps_v, ws_v], [1, 2], point_size=10, caption=False)
        ax1.set_title("Claim 1: stim peak ΔF/F₀", fontsize=9)
        ax1.tick_params(labelsize=8)

        # Claim 2 mini
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.boxplot([pg_v, wg_v], tick_labels=["PG", "WG"])
        overlay_strip(ax2, [pg_v, wg_v], [1, 2], point_size=10, caption=False)
        ax2.set_title("Claim 2: neighbor peak ΔF/F₀", fontsize=9)
        ax2.tick_params(labelsize=8)

        # Claim 3 mini
        ax3 = fig.add_subplot(gs[1, 0])
        if d3_labels:
            ax3.boxplot(d3_vals, tick_labels=d3_labels)
            overlay_strip(ax3, d3_vals, list(range(1, len(d3_labels) + 1)),
                          point_size=10, caption=False)
        else:
            ax3.text(0.5, 0.5, "no data", ha="center", va="center")
        ax3.set_title("Claim 3: AP mismatch", fontsize=9)
        ax3.tick_params(labelsize=7)

        # Claim 4 mini
        ax4 = fig.add_subplot(gs[1, 1])
        ax4.boxplot([d4_contact, d4_nocontact], tick_labels=["contact", "no-contact"])
        overlay_strip(ax4, [d4_contact, d4_nocontact], [1, 2], point_size=10, caption=False)
        ax4.set_title("Claim 4: contact vs no-contact", fontsize=9)
        ax4.tick_params(labelsize=8)

        # Claim 5 mini (wave propagation vs local front speed, from catalog)
        ax5 = fig.add_subplot(gs[2, 0:2])
        c5_prop, c5_front = [], []
        if wave_catalog_path.exists():
            try:
                _wdf = pd.read_csv(wave_catalog_path)
                c5_prop = pd.to_numeric(_wdf["propagation_speed_px_per_s"],
                                        errors="coerce").dropna().tolist()
                c5_front = pd.to_numeric(_wdf["mean_front_speed_px_per_s"],
                                         errors="coerce").dropna().tolist()
            except Exception:  # noqa: BLE001
                pass
        if c5_prop and c5_front:
            ax5.boxplot([c5_prop, c5_front],
                        tick_labels=["propagation\n(wave)", "front\n(local)"],
                        vert=True, showfliers=False)
            overlay_strip(ax5, [c5_prop, c5_front], [1, 2], point_size=6,
                          caption=False)
            ax5.set_ylabel("px/frame", fontsize=8)
        else:
            ax5.text(0.5, 0.5, "wave catalog not found\n(run batch_wave_catalog.py)",
                     ha="center", va="center", fontsize=8)
        ax5.set_title("Claim 5: wave (propagation) vs local (front) speed", fontsize=9)
        ax5.tick_params(labelsize=8)

        # Status panel (spans right column)
        ax_s = fig.add_subplot(gs[:, 2])
        ax_s.axis("off")
        ax_s.text(0.0, 1.0, "Claim status", fontsize=11, fontweight="bold",
                  transform=ax_s.transAxes, va="top")
        y = 0.93
        for name, desc, cat, n in statuses:
            color = status_colors.get(cat, "#777777")
            ax_s.text(0.0, y, name, fontsize=9.5, fontweight="bold",
                      transform=ax_s.transAxes, va="top")
            ax_s.text(1.0, y, cat, fontsize=9, fontweight="bold", color=color,
                      transform=ax_s.transAxes, va="top", ha="right")
            y -= 0.035
            n_txt = f"n={n}" if n is not None else "n=—"
            for ln in wrap(f"{desc} ({n_txt})", 34):
                ax_s.text(0.0, y, ln, fontsize=7.5, color="#555555",
                          transform=ax_s.transAxes, va="top")
                y -= 0.028
            y -= 0.02
        ax_s.text(0.0, 0.02,
                  "Dots on each plot = individual video pairs.",
                  fontsize=7, color="#777777", transform=ax_s.transAxes, va="bottom")

        pdf.savefig(fig)
        # Also export the dashboard as a standalone 1-page PDF for the combined report
        dashboard_pdf = output_pdf.parent / "claims_summary_dashboard.pdf"
        fig.savefig(dashboard_pdf)
        print(f"Generated dashboard: {dashboard_pdf}")
        plt.close(fig)

        # Title page
        fig = plt.figure(figsize=(11, 8.5))
        fig.text(0.04, 0.95, "Claims Report (No Embryo Masks)", fontsize=16, va="top")
        fig.text(0.04, 0.90, "Method: all pixels in side regions from XY geometry; $\\Delta F/F_0$ with $F_0$ from B.", fontsize=10)
        fig.text(0.04, 0.86, f"XY path: {xy_path}", fontsize=9)
        fig.text(0.04, 0.83, f"Data root: {data_root}", fontsize=9)
        fig.text(0.04, 0.79, f"Pairs analyzed: {len(per_pair)}", fontsize=10)
        fig.text(0.04, 0.75, f"Claim1 rows: {len(claim1_rows)}", fontsize=10)
        fig.text(0.04, 0.72, f"Claim2 rows: {len(claim2_rows)}", fontsize=10)
        fig.text(0.04, 0.69, f"Claim3 rows: {len(claim3_rows)}", fontsize=10)
        fig.text(0.04, 0.66, f"Claim4 rows: {len(claim4_rows)}", fontsize=10)
        pdf.savefig(fig)
        plt.close(fig)

        # Claim 1 page (Claim -> Result)
        fig, ax = plt.subplots(figsize=(11, 8.5))
        ps_pairs = get_pairs(claim1_rows, 1, "PS", 2)
        ws_pairs = get_pairs(claim1_rows, 1, "WS", 2)
        ps = [v for v, _ in ps_pairs]
        ws = [v for v, _ in ws_pairs]
        ax.boxplot([ps, ws], tick_labels=["PS stimulated peak $\\Delta F/F_0$", "WS stimulated peak $\\Delta F/F_0$"])
        overlay_strip(ax, [ps, ws], [1, 2],
                      labels=[[p for _, p in ps_pairs], [p for _, p in ws_pairs]])
        ax.set_title("Claim 1")
        ax.set_ylabel("Peak $\\Delta F/F_0$")
        txt_y = 0.99
        ax.text(
            0.02,
            txt_y,
            "Claim: Both pressure and wounding produce a similar calcium response within the stimulated embryo.",
            transform=ax.transAxes,
            va="top",
            fontsize=10,
        )
        txt_y -= 0.05
        result_lines = []
        if ps:
            t = stats.ttest_1samp(ps, 0.0, nan_policy="omit")
            result_lines.append(f"PS: n={len(ps)} mean={np.mean(ps):.4f} p(one-sample)={t.pvalue:.4g}")
        if ws:
            t = stats.ttest_1samp(ws, 0.0, nan_policy="omit")
            result_lines.append(f"WS: n={len(ws)} mean={np.mean(ws):.4f} p(one-sample)={t.pvalue:.4g}")
        if ps and ws:
            t = stats.ttest_ind(ws, ps, equal_var=False, nan_policy="omit")
            result_lines.append(f"WS vs PS (Welch): p={t.pvalue:.4g}")
        if ps and ws:
            verdict = "supports claim (both show response; no significant difference between WS and PS)" if (stats.ttest_1samp(ps, 0.0, nan_policy="omit").pvalue < alpha and stats.ttest_1samp(ws, 0.0, nan_policy="omit").pvalue < alpha and stats.ttest_ind(ws, ps, equal_var=False, nan_policy="omit").pvalue >= alpha) else "inconclusive or does not support claim at current threshold"
        else:
            verdict = "insufficient data for full claim test"
        ax.text(0.02, txt_y, "Result: " + (" | ".join(result_lines) if result_lines else "Insufficient qualifying data."), transform=ax.transAxes, va="top", fontsize=10)
        ax.text(0.02, txt_y - 0.05, f"Basic verdict (alpha={alpha:.2f}): {verdict}", transform=ax.transAxes, va="top", fontsize=10, fontweight="bold")
        pdf.savefig(fig)
        plt.close(fig)

        # Claim 2 page (Claim -> Result)
        fig, ax = plt.subplots(figsize=(11, 8.5))
        pg_pairs = get_pairs(claim2_rows, 1, "PG", 3)
        wg_pairs = get_pairs(claim2_rows, 1, "WG", 3)
        pg = [v for v, _ in pg_pairs]
        wg = [v for v, _ in wg_pairs]
        ax.boxplot([pg, wg], tick_labels=["PG neighbor peak $\\Delta F/F_0$", "WG neighbor peak $\\Delta F/F_0$"])
        overlay_strip(ax, [pg, wg], [1, 2],
                      labels=[[p for _, p in pg_pairs], [p for _, p in wg_pairs]])
        ax.set_title("Claim 2")
        ax.set_ylabel("Peak $\\Delta F/F_0$")
        txt_y = 0.99
        ax.text(
            0.02,
            txt_y,
            "Claim: Wounding, but not pressure, produces a calcium response in a neighboring embryo.",
            transform=ax.transAxes,
            va="top",
            fontsize=10,
        )
        txt_y -= 0.05
        result_lines = []
        if pg:
            t = stats.ttest_1samp(pg, 0.0, nan_policy="omit")
            result_lines.append(f"PG: n={len(pg)} mean={np.mean(pg):.4f} p(one-sample)={t.pvalue:.4g}")
        if wg:
            t = stats.ttest_1samp(wg, 0.0, nan_policy="omit")
            result_lines.append(f"WG: n={len(wg)} mean={np.mean(wg):.4f} p(one-sample)={t.pvalue:.4g}")
        if pg and wg:
            t = stats.ttest_ind(wg, pg, equal_var=False, nan_policy="omit")
            result_lines.append(f"WG vs PG (Welch): p={t.pvalue:.4g}")
        if pg and wg:
            p_pg = stats.ttest_1samp(pg, 0.0, nan_policy="omit").pvalue
            p_wg = stats.ttest_1samp(wg, 0.0, nan_policy="omit").pvalue
            p_between = stats.ttest_ind(wg, pg, equal_var=False, nan_policy="omit").pvalue
            verdict = "supports claim (WG significant, PG not significant, and WG vs PG differs)" if (p_wg < alpha and p_pg >= alpha and p_between < alpha) else "inconclusive or does not support claim at current threshold"
        else:
            verdict = "insufficient data for full claim test"
        ax.text(0.02, txt_y, "Result: " + (" | ".join(result_lines) if result_lines else "Insufficient qualifying data."), transform=ax.transAxes, va="top", fontsize=10)
        ax.text(0.02, txt_y - 0.05, f"Basic verdict (alpha={alpha:.2f}): {verdict}", transform=ax.transAxes, va="top", fontsize=10, fontweight="bold")
        pdf.savefig(fig)
        plt.close(fig)

        # Claim 3 page (Claim -> Result)
        fig, ax = plt.subplots(figsize=(11, 8.5))
        by_orient = {}
        by_orient_ids = {}
        for pid, orient, err in claim3_rows:
            if not np.isfinite(err):
                continue
            by_orient.setdefault(orient, []).append(err)
            by_orient_ids.setdefault(orient, []).append(pid)
        labels = sorted(by_orient)
        vals = [by_orient[k] for k in labels] if labels else [[]]
        ax.boxplot(vals, tick_labels=labels if labels else ["none"])
        if labels:
            overlay_strip(
                ax, vals, list(range(1, len(labels) + 1)),
                labels=[by_orient_ids[k] for k in labels],
            )
        ax.set_title("Claim 3")
        ax.set_ylabel("Absolute AP mismatch")
        ax.text(
            0.02,
            0.99,
            "Claim: Inter-embryo wound communication is informationally rich (local position in A maps to local position in B).",
            transform=ax.transAxes,
            va="top",
            fontsize=10,
        )
        if claim3_rows:
            errs = [r[2] for r in claim3_rows if np.isfinite(r[2])]
            res = f"n={len(errs)} mean |AP mismatch|={np.mean(errs):.4f} (lower is better positional mapping)"
            verdict = "descriptive support if mismatch is low; no explicit null threshold configured"
        else:
            res = "Insufficient qualifying data."
            verdict = "insufficient data"
        ax.text(0.02, 0.94, f"Result: {res}", transform=ax.transAxes, va="top", fontsize=10)
        ax.text(0.02, 0.89, f"Basic verdict: {verdict}", transform=ax.transAxes, va="top", fontsize=10, fontweight="bold")
        pdf.savefig(fig)
        plt.close(fig)

        # Claim 4 page (Claim -> Result)
        fig, ax = plt.subplots(figsize=(11, 8.5))
        contact_pairs = [(r[2], r[0]) for r in claim4_rows if r[1] == "contact" and np.isfinite(r[2])]
        nocontact_pairs = [(r[2], r[0]) for r in claim4_rows if r[1] == "no-contact" and np.isfinite(r[2])]
        contact = [v for v, _ in contact_pairs]
        nocontact = [v for v, _ in nocontact_pairs]
        ax.boxplot([contact, nocontact], tick_labels=["contact", "no-contact"])
        overlay_strip(ax, [contact, nocontact], [1, 2],
                      labels=[[p for _, p in contact_pairs], [p for _, p in nocontact_pairs]])
        ax.set_title("Claim 4")
        ax.set_ylabel("Neighbor peak $\\Delta F/F_0$")
        ax.text(
            0.02,
            0.99,
            "Claim: Inter-embryo signal does not require physical contact "
            "(all pairs with a neighbor response and a contact label, any orientation/condition).",
            transform=ax.transAxes,
            va="top",
            fontsize=10,
        )
        if contact and nocontact:
            t = stats.ttest_ind(nocontact, contact, equal_var=False, nan_policy="omit")
            result = (
                f"contact n={len(contact)} mean={np.mean(contact):.4f}; "
                f"no-contact n={len(nocontact)} mean={np.mean(nocontact):.4f}; "
                f"Welch p={t.pvalue:.4g}"
            )
            verdict = "supports claim (no-contact response persists and is not reduced vs contact)" if t.pvalue >= alpha else "inconclusive or does not support claim at current threshold"
        else:
            result = f"Insufficient qualifying data for two-group test (contact n={len(contact)}, no-contact n={len(nocontact)})."
            verdict = "insufficient data"
        ax.text(0.02, 0.94, f"Result: {result}", transform=ax.transAxes, va="top", fontsize=10)
        ax.text(0.02, 0.89, f"Basic verdict (alpha={alpha:.2f}): {verdict}", transform=ax.transAxes, va="top", fontsize=10, fontweight="bold")
        if len(nocontact) == 0:
            caveat = (
                "DATA GAP: no-contact videos (the 'WGD*' / 'no physical contact' series) "
                "have NO entries in XY coordinates.xlsx, so the geometry-based side-region "
                "method cannot locate their embryos and they are not analyzed here. "
                "n=0 means 'not measurable yet', NOT 'no signal'. To test this claim, either "
                "add XY coordinate sheets for the no-contact videos, or add a coordinate-free "
                "segmentation path (the embryos are physically separated in those videos)."
            )
            y_c = 0.83
            for ln in wrap(caveat, 95):
                ax.text(0.02, y_c, ln, transform=ax.transAxes, va="top",
                        fontsize=8.5, color="#aa2a2a")
                y_c -= 0.028
        pdf.savefig(fig)
        plt.close(fig)

        # Total brightness page: wounded (poke) vs pressed (raw intensity)
        fig, axes = plt.subplots(1, 2, figsize=(11, 8.5))
        w_tot = [(r[2], r[0]) for r in brightness_rows if r[1] == "Wounded (poke)"]
        p_tot = [(r[2], r[0]) for r in brightness_rows if r[1] == "Pressed"]
        w_mean = [(r[3], r[0]) for r in brightness_rows if r[1] == "Wounded (poke)"]
        p_mean = [(r[3], r[0]) for r in brightness_rows if r[1] == "Pressed"]

        wt = [v for v, _ in w_tot]
        pt = [v for v, _ in p_tot]
        axes[0].boxplot([wt, pt], tick_labels=["Wounded (poke)", "Pressed"])
        overlay_strip(axes[0], [wt, pt], [1, 2],
                      labels=[[i for _, i in w_tot], [i for _, i in p_tot]])
        axes[0].set_title("Total brightness in stimulated region", fontsize=11)
        axes[0].set_ylabel("Σ raw pixel intensity (region × all frames)")
        axes[0].tick_params(labelsize=8)

        wm = [v for v, _ in w_mean]
        pm = [v for v, _ in p_mean]
        axes[1].boxplot([wm, pm], tick_labels=["Wounded (poke)", "Pressed"])
        overlay_strip(axes[1], [wm, pm], [1, 2],
                      labels=[[i for _, i in w_mean], [i for _, i in p_mean]])
        axes[1].set_title("Mean brightness (per pixel, per frame)", fontsize=11)
        axes[1].set_ylabel("Mean raw pixel intensity")
        axes[1].tick_params(labelsize=8)

        fig.suptitle(
            "Total brightness: wounded (poke) vs pressed — RAW intensity in stimulated region",
            fontsize=13,
        )
        note = (
            "Raw camera intensity (NOT normalized like $\\Delta F/F_0$). "
            "Total = sum of all pixel intensities in the stimulated region over every frame; "
            "Mean = average pixel intensity. "
            "Totals depend on frame count and region size, so use Mean for fair comparison."
        )
        if len(wt) >= 2 and len(pt) >= 2:
            pval = stats.ttest_ind(wt, pt, equal_var=False, nan_policy="omit").pvalue
            stat_line = (
                f"Total: wounded n={len(wt)} mean={np.mean(wt):.3g}; "
                f"pressed n={len(pt)} mean={np.mean(pt):.3g}; Welch p={pval:.4g}"
            )
        else:
            stat_line = (
                f"Insufficient n for test (wounded n={len(wt)}, pressed n={len(pt)})."
            )
        fig.text(0.5, 0.06, stat_line, ha="center", fontsize=9)
        for j, ln in enumerate(wrap(note, 110)):
            fig.text(0.5, 0.035 - j * 0.018, ln, ha="center", fontsize=8, color="#555555")
        fig.subplots_adjust(bottom=0.16, top=0.9, wspace=0.3)
        pdf.savefig(fig)
        plt.close(fig)

        # Brightness map page: each video as a swatch colored by brightness
        # (dark = dim, bright yellow = bright) using an intuitive heat colormap.
        w_mean_pairs = [(r[3], r[0]) for r in brightness_rows
                        if r[1] == "Wounded (poke)" and np.isfinite(r[3])]
        p_mean_pairs = [(r[3], r[0]) for r in brightness_rows
                        if r[1] == "Pressed" and np.isfinite(r[3])]
        all_means = [v for v, _ in w_mean_pairs] + [v for v, _ in p_mean_pairs]

        fig, axes = plt.subplots(1, 2, figsize=(11, 8.5))
        cmap = plt.cm.inferno
        if all_means:
            norm = Normalize(vmin=min(all_means), vmax=max(all_means))
        else:
            norm = Normalize(vmin=0, vmax=1)

        def draw_brightness_col(ax, pairs, title):
            pairs_sorted = sorted(pairs, key=lambda t: t[0], reverse=True)
            if pairs_sorted:
                vals = np.array([[v] for v, _ in pairs_sorted])
            else:
                vals = np.zeros((1, 1))
            im = ax.imshow(vals, cmap=cmap, norm=norm, aspect="auto")
            ax.set_xticks([])
            ax.set_yticks(range(len(pairs_sorted)))
            ax.set_yticklabels(
                [f"{vid}  ({v:.0f})" for v, vid in pairs_sorted], fontsize=6
            )
            ax.set_title(f"{title} (n={len(pairs_sorted)})", fontsize=11)
            return im

        im = draw_brightness_col(axes[0], w_mean_pairs, "Wounded (poke)")
        draw_brightness_col(axes[1], p_mean_pairs, "Pressed")
        cbar = fig.colorbar(im, ax=axes, fraction=0.046, pad=0.06)
        cbar.set_label("Mean brightness (raw intensity)")
        fig.suptitle(
            "Brightness map — brighter color = brighter region (mean raw intensity)",
            fontsize=13,
        )
        fig.text(
            0.5, 0.04,
            "Each row is one video; color encodes its mean region brightness on a "
            "shared scale (dark = dim, yellow/white = bright). Sorted brightest-first.",
            ha="center", fontsize=8, color="#555555",
        )
        pdf.savefig(fig)
        plt.close(fig)

        # Claim 5: wave (propagation) vs local (front) speed, from the rolled-up
        # wave catalog (generate_pixel_brightness_vectors.py -> rollup -> batch).
        render_claim5_speed_page(pdf, wave_catalog_path)

    print(f"Generated PDF: {output_pdf}")
    print(f"Pairs analyzed: {len(per_pair)}")
    print(f"Claim1 rows={len(claim1_rows)} Claim2 rows={len(claim2_rows)} Claim3 rows={len(claim3_rows)} Claim4 rows={len(claim4_rows)}")


if __name__ == "__main__":
    main()


#!/usr/bin/env python3
"""
Generate a current-dataset claim justification PDF with delta charts.

This uses current available data only (no future-sample speculation).
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import tifffile as tiff
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy import stats


@dataclass
class SheetEntry:
    sheet_name: str
    background_video: str
    calcium_video: str
    coords_b: dict
    coords_c: dict


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", str(s).strip())


def parse_side(s: str) -> Optional[str]:
    low = str(s).lower()
    if "left" in low:
        return "Left"
    if "right" in low:
        return "Right"
    return None


def parse_condition(video_name: str) -> str:
    m = re.match(r"([A-Za-z]{2})\d+_", Path(video_name).stem)
    return m.group(1).upper() if m else "UNK"


def parse_orientation(video_name: str) -> str:
    m = re.search(r"-o([A-Za-z]+)", Path(video_name).stem, flags=re.IGNORECASE)
    return m.group(1).upper() if m else "UNK"


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
            x, y = float(x_raw), float(y_raw)
        except Exception:
            i += 1
            continue
        side = parse_side(side_raw) or "Single"
        out.setdefault(side, {})
        blob = f"{side_raw} {loc_raw}".lower()
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
        b_name = c_name = ""
        for r in range(bi + 1, min(len(df), bi + 6)):
            v = df.iloc[r, 0]
            if pd.notna(v) and str(v).strip():
                b_name = norm(v)
                break
        for r in range(ci + 1, min(len(df), ci + 6)):
            v = df.iloc[r, 0]
            if pd.notna(v) and str(v).strip():
                c_name = norm(v)
                break
        if not b_name or not c_name:
            continue
        frame_rows = df.index[df.iloc[:, 0].astype(str).str.strip().str.lower() == "frame"].tolist()
        b_frame = next((r for r in frame_rows if bi < r < ci), None)
        c_frame = next((r for r in frame_rows if r > ci), None)
        if b_frame is None or c_frame is None:
            continue
        entries.append(
            SheetEntry(
                sheet_name=sheet,
                background_video=b_name,
                calcium_video=c_name,
                coords_b=parse_coords_block(df, int(b_frame)),
                coords_c=parse_coords_block(df, int(c_frame)),
            )
        )
    return entries


def build_tiff_index(root: Path) -> Dict[str, Path]:
    idx = {}
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in (".tif", ".tiff"):
            idx.setdefault(p.name.lower(), p)
            idx.setdefault(p.stem.lower(), p)
    return idx


def find_tiff(name: str, idx: Dict[str, Path]) -> Optional[Path]:
    raw = norm(name)
    cands = [raw.lower(), Path(raw).name.lower(), Path(raw).stem.lower()]
    for c in cands:
        if c in idx:
            return idx[c]
    return None


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
    acc = None
    n = 0
    with tiff.TiffFile(video_path) as tf:
        for p in tf.pages:
            arr = p.asarray()
            if arr.ndim == 3:
                arr = arr[..., 0]
            frame = np.array(arr).astype(float)
            if acc is None:
                acc = np.zeros_like(frame, dtype=float)
            acc += frame
            n += 1
            for k, m in masks.items():
                vals = frame[m]
                traces[k].append(float(np.mean(vals)) if vals.size else np.nan)
    return {k: np.array(v, dtype=float) for k, v in traces.items()}, (acc / max(n, 1))


def project_ap(point: Tuple[float, float], head: Tuple[float, float], tail: Tuple[float, float]) -> float:
    p = np.array(point, dtype=float)
    h = np.array(head, dtype=float)
    t = np.array(tail, dtype=float)
    axis = t - h
    den = float(np.dot(axis, axis))
    if den < 1e-9:
        return float("nan")
    return float(np.dot(p - h, axis) / den)


def verdict_label(text: str) -> str:
    return text.upper()


def fmt_p(x: float) -> str:
    return f"{x:.4g}" if np.isfinite(x) else "NA"


def overlay_strip(ax, datasets, positions, labels=None, seed=0,
                  point_size=22, caption=True):
    """Overlay jittered per-video points on a boxplot (one dot = one pair)."""
    rng = np.random.default_rng(seed)
    for di, (data, pos) in enumerate(zip(datasets, positions)):
        if not data:
            continue
        xs = pos + (rng.random(len(data)) - 0.5) * 0.16
        ax.scatter(xs, data, s=point_size, color="#1f77b4", alpha=0.75,
                   edgecolors="white", linewidths=0.4, zorder=3)
        if labels is not None and di < len(labels) and labels[di] is not None:
            for x, y, lab in zip(xs, data, labels[di]):
                ax.annotate(str(lab), (x, y), fontsize=4.5, alpha=0.6,
                            xytext=(2, 2), textcoords="offset points", zorder=4)
    if caption:
        ax.text(0.99, 0.01,
                "Dots = individual video pairs (jittered); box = aggregate.",
                transform=ax.transAxes, ha="right", va="bottom",
                fontsize=7, color="#555555")


def finite_pairs(rows, cond_idx, cond, val_idx, id_idx=0):
    """Return ([values], [ids]) for finite values matching a condition."""
    vals, ids = [], []
    for r in rows:
        if r[cond_idx] == cond and np.isfinite(r[val_idx]):
            vals.append(r[val_idx])
            ids.append(r[id_idx])
    return vals, ids


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate claim justification PDF with delta charts.")
    ap.add_argument("--xy-path", default="/Users/jdietz/Library/CloudStorage/Box-Box/Calcium videos/XY coordinates.xlsx")
    ap.add_argument("--data-root", default="/Users/jdietz/Library/CloudStorage/Box-Box/Calcium videos/Calcium videos")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--output-pdf", default="wave-vector-analysis/analysis_results/claim_justification_with_deltas.pdf")
    args = ap.parse_args()

    xy_path = Path(args.xy_path)
    data_root = Path(args.data_root)
    out_pdf = Path(args.output_pdf)
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    alpha = float(args.alpha)

    entries = parse_workbook(xy_path)
    idx = build_tiff_index(data_root)

    c1 = []  # (pair, cond, stim_delta_mean, stim_peak)
    c2 = []  # (pair, cond, neighbor_delta_mean, neighbor_peak)
    c3 = []  # (pair, orient, ap_mismatch)
    c4 = []  # (pair, contact_group, neighbor_delta_mean)

    for e in entries:
        b_path = find_tiff(e.background_video, idx)
        c_path = find_tiff(e.calcium_video, idx)
        if not b_path or not c_path:
            continue
        cond = parse_condition(e.calcium_video)
        orient = parse_orientation(e.calcium_video)
        coords = e.coords_c if e.coords_c else e.coords_b

        try:
            with tiff.TiffFile(c_path) as tf:
                f0 = tf.pages[0].asarray()
                if f0.ndim == 3:
                    f0 = f0[..., 0]
                masks = side_masks(f0.shape[:2], coords)
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

        poke_side = None
        for s, meta in (coords or {}).items():
            if "Poke" in meta:
                poke_side = s
                break
        stim_side = poke_side if poke_side in dff_c else ("Single" if "Single" in dff_c else None)
        if stim_side is None and dff_c:
            stim_side = sorted(dff_c.keys())[0]
        neighbor_side = None
        if stim_side == "Left" and "Right" in dff_c:
            neighbor_side = "Right"
        elif stim_side == "Right" and "Left" in dff_c:
            neighbor_side = "Left"

        pair = Path(e.calcium_video).stem

        # Claim 1: PS/WS oHT
        if cond in {"PS", "WS"} and orient == "HT" and stim_side in dff_c:
            stim_delta_mean = float(np.nanmean(dff_c[stim_side]) - np.nanmean(dff_b.get(stim_side, np.array([np.nan]))))
            stim_peak = float(np.nanmax(dff_c[stim_side]))
            c1.append((pair, cond, stim_delta_mean, stim_peak))

        # Claim 2: PG/WG oHTHT neighbor
        if cond in {"PG", "WG"} and orient == "HTHT" and neighbor_side in dff_c:
            n_delta_mean = float(np.nanmean(dff_c[neighbor_side]) - np.nanmean(dff_b.get(neighbor_side, np.array([np.nan]))))
            n_peak = float(np.nanmax(dff_c[neighbor_side]))
            c2.append((pair, cond, n_delta_mean, n_peak))

            if cond == "WG":
                pstr = str(c_path).lower()
                contact_group = "no-contact" if "no physical contact" in pstr else ("contact" if "physical contact" in pstr else "unknown")
                c4.append((pair, contact_group, n_delta_mean))

        # Claim 3: positional mapping WG THHT/HTTH/HTHT
        if cond == "WG" and orient in {"THHT", "HTTH", "HTHT"} and stim_side in {"Left", "Right"} and neighbor_side in {"Left", "Right"}:
            stim_meta = coords.get(stim_side, {})
            neigh_meta = coords.get(neighbor_side, {})
            if "Poke" in stim_meta and "Head" in stim_meta and "Tail" in stim_meta and "Head" in neigh_meta and "Tail" in neigh_meta:
                delta_img = c_mean - b_mean
                nmask = masks[neighbor_side]
                if np.any(nmask):
                    vals = np.where(nmask, delta_img, -np.inf)
                    yi, xi = np.unravel_index(np.argmax(vals), vals.shape)
                    wound_ap = project_ap(stim_meta["Poke"], stim_meta["Head"], stim_meta["Tail"])
                    neigh_ap = project_ap((float(xi), float(yi)), neigh_meta["Head"], neigh_meta["Tail"])
                    if np.isfinite(wound_ap) and np.isfinite(neigh_ap):
                        c3.append((pair, orient, abs(neigh_ap - wound_ap)))

    # Stats
    ps_delta = [r[2] for r in c1 if r[1] == "PS"]
    ws_delta = [r[2] for r in c1 if r[1] == "WS"]
    pg_delta = [r[2] for r in c2 if r[1] == "PG"]
    wg_delta = [r[2] for r in c2 if r[1] == "WG"]
    c3_vals = [r[2] for r in c3]
    c4_contact = [r[2] for r in c4 if r[1] == "contact"]
    c4_nocontact = [r[2] for r in c4 if r[1] == "no-contact"]

    # Verdicts based on current data only
    def t1(vals):
        return stats.ttest_1samp(vals, 0.0, nan_policy="omit").pvalue if len(vals) >= 2 else np.nan

    def t2(a, b):
        return stats.ttest_ind(a, b, equal_var=False, nan_policy="omit").pvalue if len(a) >= 2 and len(b) >= 2 else np.nan

    p_ps = t1(ps_delta)
    p_ws = t1(ws_delta)
    p_c1_between = t2(ws_delta, ps_delta)

    p_pg = t1(pg_delta)
    p_wg = t1(wg_delta)
    p_c2_between = t2(wg_delta, pg_delta)

    p_c4_between = t2(c4_nocontact, c4_contact)

    v1 = "provisionally justified" if (np.isfinite(p_ps) and np.isfinite(p_ws) and p_ps < alpha and p_ws < alpha) else "partially justified"
    v2 = "provisionally justified" if (np.isfinite(p_wg) and p_wg < alpha and (not np.isfinite(p_pg) or p_pg >= alpha)) else "partially justified"
    v3 = "suggestive / preliminary" if len(c3_vals) >= 5 else "insufficient data"
    if len(c4_contact) >= 2 and len(c4_nocontact) >= 2:
        v4 = "provisionally justified" if (np.isfinite(p_c4_between) and p_c4_between >= alpha) else "not yet justified"
    else:
        v4 = "insufficient contact split data"
    v5 = "speed data exists; classification step pending"

    with PdfPages(out_pdf) as pdf:
        # Summary status page
        fig = plt.figure(figsize=(11, 8.5))
        fig.text(0.04, 0.95, "Current Claim Justification Status (with deltas)", fontsize=16, va="top")
        fig.text(0.04, 0.91, "Dataset-only assessment (no future-data assumptions)", fontsize=10)
        y = 0.85
        lines = [
            f"Claim 1: {verdict_label(v1)} | n={len(c1)} (PS={len(ps_delta)}, WS={len(ws_delta)}) | pPS={fmt_p(p_ps)}",
            f"Claim 2: {verdict_label(v2)} | n={len(c2)} (PG={len(pg_delta)}, WG={len(wg_delta)}) | pWG={fmt_p(p_wg)}",
            f"Claim 3: {verdict_label(v3)} | n={len(c3_vals)} positional AP-mismatch values",
            f"Claim 4: {verdict_label(v4)} | WG-HTHT contact={len(c4_contact)}, no-contact={len(c4_nocontact)}",
            f"Claim 5: {verdict_label(v5)} | spark_tracks has vx/vy/speed and vector_clusters has mean/net/peak speed",
        ]
        for ln in lines:
            fig.text(0.05, y, ln, fontsize=11)
            y -= 0.055
        fig.text(0.05, y - 0.02, f"Alpha: {alpha}", fontsize=10)
        pdf.savefig(fig)
        plt.close(fig)

        # Claim 1 delta chart
        fig, ax = plt.subplots(figsize=(11, 8.5))
        ps_v, ps_i = finite_pairs(c1, 1, "PS", 2)
        ws_v, ws_i = finite_pairs(c1, 1, "WS", 2)
        ax.boxplot([ps_delta, ws_delta], tick_labels=["PS-oHT $\\Delta$mean $\\Delta F/F_0$", "WS-oHT $\\Delta$mean $\\Delta F/F_0$"])
        overlay_strip(ax, [ps_v, ws_v], [1, 2], labels=[ps_i, ws_i])
        ax.axhline(0, color="gray", lw=1, ls="--")
        ax.set_title("Claim 1 delta chart (stimulated embryo)")
        ax.set_ylabel("$\\Delta$ = mean(C $\\Delta F/F_0$) - mean(B $\\Delta F/F_0$)")
        txt = f"n: PS={len(ps_delta)}, WS={len(ws_delta)} | p(WS vs PS)={p_c1_between:.4g}" if np.isfinite(p_c1_between) else "Insufficient n for between-group test"
        ax.text(0.02, 0.95, txt, transform=ax.transAxes, va="top")
        pdf.savefig(fig)
        plt.close(fig)

        # Claim 2 delta chart
        fig, ax = plt.subplots(figsize=(11, 8.5))
        pg_v, pg_i = finite_pairs(c2, 1, "PG", 2)
        wg_v, wg_i = finite_pairs(c2, 1, "WG", 2)
        ax.boxplot([pg_delta, wg_delta], tick_labels=["PG-oHTHT neighbor $\\Delta$mean", "WG-oHTHT neighbor $\\Delta$mean"])
        overlay_strip(ax, [pg_v, wg_v], [1, 2], labels=[pg_i, wg_i])
        ax.axhline(0, color="gray", lw=1, ls="--")
        ax.set_title("Claim 2 delta chart (neighbor embryo)")
        ax.set_ylabel("$\\Delta$ = mean(C $\\Delta F/F_0$) - mean(B $\\Delta F/F_0$)")
        txt = f"n: PG={len(pg_delta)}, WG={len(wg_delta)} | p(WG vs PG)={p_c2_between:.4g}" if np.isfinite(p_c2_between) else "Insufficient n for between-group test"
        ax.text(0.02, 0.95, txt, transform=ax.transAxes, va="top")
        pdf.savefig(fig)
        plt.close(fig)

        # Claim 3 chart
        fig, ax = plt.subplots(figsize=(11, 8.5))
        if c3_vals:
            ax.hist(c3_vals, bins=min(12, max(4, int(np.sqrt(len(c3_vals))))), alpha=0.8)
            ax.set_ylabel("Count")
            # Rug of individual video pairs along the x-axis
            c3_ids = [r[0] for r in c3 if np.isfinite(r[2])]
            ymax = ax.get_ylim()[1]
            rng = np.random.default_rng(0)
            yr = ymax * 0.03 + ymax * 0.02 * rng.random(len(c3_vals))
            ax.scatter(c3_vals, yr, s=22, color="#d62728", alpha=0.8,
                       edgecolors="white", linewidths=0.4, zorder=5)
            for x, y, lab in zip(c3_vals, yr, c3_ids):
                ax.annotate(str(lab), (x, y), fontsize=4.5, alpha=0.6,
                            xytext=(2, 2), textcoords="offset points", zorder=6)
            ax.text(0.99, 0.01, "Red dots = individual video pairs.",
                    transform=ax.transAxes, ha="right", va="bottom",
                    fontsize=7, color="#555555")
        ax.set_title("Claim 3 positional metric")
        ax.set_xlabel("|AP(neighbor max-response) - AP(wound)|  (lower is better)")
        if c3_vals:
            ax.text(0.02, 0.95, f"n={len(c3_vals)} mean={np.mean(c3_vals):.4f} median={np.median(c3_vals):.4f}", transform=ax.transAxes, va="top")
        else:
            ax.text(0.02, 0.95, "No qualifying values.", transform=ax.transAxes, va="top")
        pdf.savefig(fig)
        plt.close(fig)

        # Claim 4 chart
        fig, ax = plt.subplots(figsize=(11, 8.5))
        c4c_v, c4c_i = finite_pairs(c4, 1, "contact", 2)
        c4n_v, c4n_i = finite_pairs(c4, 1, "no-contact", 2)
        ax.boxplot([c4_contact, c4_nocontact], tick_labels=["contact", "no-contact"])
        overlay_strip(ax, [c4c_v, c4n_v], [1, 2], labels=[c4c_i, c4n_i])
        ax.axhline(0, color="gray", lw=1, ls="--")
        ax.set_title("Claim 4 delta chart (WG-oHTHT neighbor)")
        ax.set_ylabel("$\\Delta$ = mean(C $\\Delta F/F_0$) - mean(B $\\Delta F/F_0$)")
        txt = f"n: contact={len(c4_contact)}, no-contact={len(c4_nocontact)}"
        if np.isfinite(p_c4_between):
            txt += f" | p(no-contact vs contact)={p_c4_between:.4g}"
        ax.text(0.02, 0.95, txt, transform=ax.transAxes, va="top")
        pdf.savefig(fig)
        plt.close(fig)

    print(f"Generated PDF: {out_pdf}")
    print(f"Claim1 n={len(c1)} Claim2 n={len(c2)} Claim3 n={len(c3_vals)} Claim4 n={len(c4)}")


if __name__ == "__main__":
    main()


#!/usr/bin/env python3
"""
Generate a B-vs-C regional dF/F report from XY coordinates and TIFF stacks.

Method:
1) Load each frame as grayscale float.
2) Build an embryo mask from the reference frame.
3) Use XY head/tail(/poke) coordinates from workbook.
4) Split embryo mask into head/tail regions along head-tail axis.
5) Compute F(t) as mean intensity in each region.
6) Compute dF/F0 where F0 is mean F(t) across background video (B), per region.
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
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


def normalize_video_name(name: str) -> str:
    s = str(name).strip()
    s = re.sub(r"\s+", " ", s)
    return s


def parse_side(text: str) -> Optional[str]:
    t = str(text).lower()
    if "left" in t:
        return "Left"
    if "right" in t:
        return "Right"
    return None


def parse_coords_block(df: pd.DataFrame, start_row: int) -> dict:
    coords = {}
    n = len(df)
    i = start_row + 1
    while i < n:
        row = df.iloc[i].tolist()
        as_text = " ".join("" if pd.isna(x) else str(x) for x in row).strip().lower()
        if not as_text or "results" in as_text or "video" in as_text:
            break
        nums = []
        for x in row:
            try:
                nums.append(float(x))
            except Exception:
                pass
        if len(nums) >= 2:
            if len(row) >= 5:
                orient_raw = "" if pd.isna(row[1]) else str(row[1])
                loc_raw = "" if pd.isna(row[2]) else str(row[2])
                x_val = row[3]
                y_val = row[4]
            else:
                orient_raw = ""
                loc_raw = "" if pd.isna(row[1]) else str(row[1])
                x_val = row[2]
                y_val = row[3]
            try:
                x = float(x_val)
                y = float(y_val)
            except Exception:
                i += 1
                continue
            side = parse_side(orient_raw) or "Single"
            blob = f"{orient_raw} {loc_raw}".lower()
            if "poke" in blob:
                coords.setdefault(side, {})["Poke"] = (x, y)
            elif "head" in blob:
                coords.setdefault(side, {})["Head"] = (x, y)
            elif "tail" in blob:
                coords.setdefault(side, {})["Tail"] = (x, y)
        i += 1
    return coords


def parse_workbook(xy_path: Path) -> List[SheetEntry]:
    xl = pd.ExcelFile(xy_path)
    out: List[SheetEntry] = []
    for sheet in xl.sheet_names:
        df = pd.read_excel(xy_path, sheet_name=sheet, header=None)
        col0 = df.iloc[:, 0].astype(str).str.strip()
        b_idx = col0[col0.str.lower() == "background video"].index
        c_idx = col0[col0.str.lower() == "calcium video"].index
        if len(b_idx) == 0 or len(c_idx) == 0:
            continue
        bi, ci = int(b_idx[0]), int(c_idx[0])
        b_name = ""
        c_name = ""
        for j in range(bi + 1, min(len(df), bi + 6)):
            if pd.notna(df.iloc[j, 0]) and str(df.iloc[j, 0]).strip():
                b_name = normalize_video_name(df.iloc[j, 0])
                break
        for j in range(ci + 1, min(len(df), ci + 6)):
            if pd.notna(df.iloc[j, 0]) and str(df.iloc[j, 0]).strip():
                c_name = normalize_video_name(df.iloc[j, 0])
                break
        if not b_name or not c_name:
            continue
        frame_rows = df.index[df.iloc[:, 0].astype(str).str.strip().str.lower() == "frame"].tolist()
        b_frame = next((r for r in frame_rows if r > bi and r < ci), None)
        c_frame = next((r for r in frame_rows if r > ci), None)
        if b_frame is None or c_frame is None:
            continue
        coords_b = parse_coords_block(df, int(b_frame))
        coords_c = parse_coords_block(df, int(c_frame))
        out.append(SheetEntry(sheet, b_name, c_name, coords_b, coords_c))
    return out


def build_tiff_index(data_root: Path) -> Dict[str, Path]:
    idx: Dict[str, Path] = {}
    for p in data_root.rglob("*"):
        if p.is_file() and p.suffix.lower() in (".tif", ".tiff"):
            key = p.name.lower()
            idx.setdefault(key, p)
            idx.setdefault(p.stem.lower(), p)
    return idx


def find_tiff_path(name: str, idx: Dict[str, Path]) -> Optional[Path]:
    raw = normalize_video_name(name)
    candidates = [raw, raw.lower(), Path(raw).name, Path(raw).name.lower()]
    stem = Path(raw).stem.lower()
    candidates.append(stem)
    for c in candidates:
        if c in idx:
            return idx[c]
    return None


def make_embryo_mask(frame: np.ndarray) -> np.ndarray:
    arr = frame.astype(np.float32)
    lo, hi = np.percentile(arr, [2, 98])
    if hi <= lo:
        hi = lo + 1.0
    scaled = np.clip((arr - lo) / (hi - lo), 0, 1)
    gray8 = (scaled * 255).astype(np.uint8)
    blur = cv2.GaussianBlur(gray8, (0, 0), 2.0)
    _, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    n, labels, stats_arr, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n <= 1:
        return mask > 0
    keep = np.argsort(stats_arr[1:, cv2.CC_STAT_AREA])[::-1][:2] + 1
    out = np.isin(labels, keep)
    return out


def split_regions(mask: np.ndarray, coords: dict) -> Dict[str, np.ndarray]:
    h, w = mask.shape
    yy, xx = np.indices((h, w))
    regions: Dict[str, np.ndarray] = {}
    sides = [s for s in coords.keys() if "Head" in coords[s] and "Tail" in coords[s]]
    if not sides:
        return regions
    if "Left" in sides and "Right" in sides:
        x_left = 0.5 * (coords["Left"]["Head"][0] + coords["Left"]["Tail"][0])
        x_right = 0.5 * (coords["Right"]["Head"][0] + coords["Right"]["Tail"][0])
        mid = 0.5 * (x_left + x_right)
        side_masks = {"Left": mask & (xx < mid), "Right": mask & (xx >= mid)}
    else:
        s = sides[0]
        side_masks = {s: mask}
    for side, emask in side_masks.items():
        head = np.array(coords[side]["Head"], dtype=float)
        tail = np.array(coords[side]["Tail"], dtype=float)
        axis = tail - head
        den = float(axis.dot(axis)) if float(axis.dot(axis)) > 1e-6 else 1.0
        proj = ((xx - head[0]) * axis[0] + (yy - head[1]) * axis[1]) / den
        head_mask = emask & (proj <= 0.5)
        tail_mask = emask & (proj > 0.5)
        regions[f"{side}_Head"] = head_mask
        regions[f"{side}_Tail"] = tail_mask
    return regions


def video_region_traces(video_path: Path, coords: dict) -> Dict[str, np.ndarray]:
    traces: Dict[str, List[float]] = {}
    with tiff.TiffFile(video_path) as tf:
        first = tf.pages[0].asarray()
        if first.ndim == 3:
            first = first[..., 0]
        mask = make_embryo_mask(first.astype(np.float32))
        regions = split_regions(mask, coords)
        for k in regions:
            traces[k] = []
        for page in tf.pages:
            img = page.asarray()
            if img.ndim == 3:
                img = img[..., 0]
            arr = np.array(img).astype(float)
            for k, rmask in regions.items():
                vals = arr[rmask]
                traces[k].append(float(np.mean(vals)) if vals.size else np.nan)
    return {k: np.array(v, dtype=float) for k, v in traces.items()}


def classify_condition(video_name: str) -> str:
    m = re.match(r"([A-Za-z]{2})\d+_", Path(video_name).stem)
    return m.group(1).upper() if m else "UNK"


def is_neighbor_setup(video_name: str) -> bool:
    return "oHTHT" in video_name or "oHTTH" in video_name


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate XY-based regional dF/F PDF.")
    ap.add_argument("--xy-path", default="/Users/jdietz/Library/CloudStorage/Box-Box/Calcium videos/XY coordinates.xlsx")
    ap.add_argument("--data-root", default="/Users/jdietz/Library/CloudStorage/Box-Box/Calcium videos/Calcium videos")
    ap.add_argument("--output-pdf", default="wave-vector-analysis/analysis_results/xy_dff_claim_report.pdf")
    args = ap.parse_args()

    xy_path = Path(args.xy_path)
    data_root = Path(args.data_root)
    out_pdf = Path(args.output_pdf)
    out_pdf.parent.mkdir(parents=True, exist_ok=True)

    entries = parse_workbook(xy_path)
    idx = build_tiff_index(data_root)
    results = []

    for e in entries:
        b_path = find_tiff_path(e.background_video, idx)
        c_path = find_tiff_path(e.calcium_video, idx)
        if not b_path or not c_path:
            continue
        try:
            b_tr = video_region_traces(b_path, e.coords_b if e.coords_b else e.coords_c)
            c_tr = video_region_traces(c_path, e.coords_c if e.coords_c else e.coords_b)
        except Exception:
            continue
        if not b_tr or not c_tr:
            continue
        f0 = {k: float(np.nanmean(v)) for k, v in b_tr.items() if np.isfinite(np.nanmean(v))}
        dff_b = {}
        dff_c = {}
        for k, f0v in f0.items():
            if abs(f0v) < 1e-9:
                continue
            if k in b_tr and k in c_tr:
                dff_b[k] = (b_tr[k] - f0v) / f0v
                dff_c[k] = (c_tr[k] - f0v) / f0v
        if not dff_b:
            continue
        results.append((e, b_path, c_path, dff_b, dff_c))

    claim1_rows = []  # PS/WS, stimulated embryo
    claim2_rows = []  # PG/WG, neighbor embryo

    with PdfPages(out_pdf) as pdf:
        fig = plt.figure(figsize=(11, 8.5))
        fig.text(
            0.05,
            0.95,
            "XY-based Regional $\\Delta F/F_0$ Report\n$F_0$ is mean $F(t)$ over B video, per region",
            va="top",
            fontsize=14,
        )
        fig.text(0.05, 0.84, f"XY file: {xy_path}", fontsize=9)
        fig.text(0.05, 0.81, f"TIFF root: {data_root}", fontsize=9)
        fig.text(0.05, 0.78, f"Pairs analyzed: {len(results)}", fontsize=10)
        pdf.savefig(fig)
        plt.close(fig)

        for e, b_path, c_path, dff_b, dff_c in results:
            keys = sorted(dff_b.keys())
            n = len(keys)
            ncols = 2
            nrows = int(np.ceil(n / ncols))
            fig, axes = plt.subplots(nrows, ncols, figsize=(11, 8.5), squeeze=False)
            fig.suptitle(f"{e.sheet_name}\nB: {Path(e.background_video).name} | C: {Path(e.calcium_video).name}", fontsize=11)
            for i, k in enumerate(keys):
                ax = axes[i // ncols][i % ncols]
                ax.plot(dff_b[k], label="B $\\Delta F/F_0$", alpha=0.9)
                ax.plot(dff_c[k], label="C $\\Delta F/F_0$", alpha=0.9)
                ax.set_title(k)
                ax.set_xlabel("Frame")
                ax.set_ylabel("$\\Delta F/F_0$")
                ax.legend(fontsize=7, loc="best")
            for j in range(n, nrows * ncols):
                axes[j // ncols][j % ncols].axis("off")
            plt.tight_layout(rect=(0, 0, 1, 0.92))
            pdf.savefig(fig)
            plt.close(fig)

            cond = classify_condition(e.calcium_video)
            mean_b = {k: float(np.nanmean(v)) for k, v in dff_b.items()}
            mean_c = {k: float(np.nanmean(v)) for k, v in dff_c.items()}
            if not is_neighbor_setup(e.calcium_video) and cond in {"PS", "WS"}:
                keys1 = [k for k in mean_b if "Head" in k or "Tail" in k]
                if keys1:
                    delta = float(np.nanmean([mean_c[k] - mean_b[k] for k in keys1]))
                    claim1_rows.append((Path(e.calcium_video).stem, cond, delta))
            if is_neighbor_setup(e.calcium_video) and cond in {"PG", "WG"}:
                poke_side = None
                for side, meta in (e.coords_c or {}).items():
                    if "Poke" in meta:
                        poke_side = side
                        break
                if poke_side in {"Left", "Right"}:
                    neighbor = "Right" if poke_side == "Left" else "Left"
                    keys2 = [f"{neighbor}_Head", f"{neighbor}_Tail"]
                    vals = [mean_c[k] - mean_b[k] for k in keys2 if k in mean_b and k in mean_c]
                    if vals:
                        claim2_rows.append((Path(e.calcium_video).stem, cond, float(np.nanmean(vals))))

        fig = plt.figure(figsize=(11, 8.5))
        fig.suptitle("Claim Summary ($\\Delta$ activity = mean(C $\\Delta F/F_0$) - mean(B $\\Delta F/F_0$))", fontsize=13)
        y = 0.9
        fig.text(0.05, y, "Claim 1: Stimulated embryo (PS-oHT, WS-oHT)", fontsize=11)
        y -= 0.04
        if claim1_rows:
            for cond in ["PS", "WS"]:
                vals = [r[2] for r in claim1_rows if r[1] == cond]
                if vals:
                    t = stats.ttest_1samp(vals, 0.0, nan_policy="omit")
                    fig.text(0.07, y, f"{cond}: n={len(vals)}, mean_delta={np.mean(vals):.4f}, p(one-sample)={t.pvalue:.4g}", fontsize=10)
                    y -= 0.03
        else:
            fig.text(0.07, y, "No qualifying pairs found.", fontsize=10)
            y -= 0.03

        y -= 0.03
        fig.text(0.05, y, "Claim 2: Neighbor embryo (PG-oHTHT, WG-oHTHT)", fontsize=11)
        y -= 0.04
        if claim2_rows:
            pg = [r[2] for r in claim2_rows if r[1] == "PG"]
            wg = [r[2] for r in claim2_rows if r[1] == "WG"]
            if pg:
                t = stats.ttest_1samp(pg, 0.0, nan_policy="omit")
                fig.text(0.07, y, f"PG: n={len(pg)}, mean_delta={np.mean(pg):.4f}, p(one-sample)={t.pvalue:.4g}", fontsize=10)
                y -= 0.03
            if wg:
                t = stats.ttest_1samp(wg, 0.0, nan_policy="omit")
                fig.text(0.07, y, f"WG: n={len(wg)}, mean_delta={np.mean(wg):.4f}, p(one-sample)={t.pvalue:.4g}", fontsize=10)
                y -= 0.03
            if pg and wg:
                t = stats.ttest_ind(wg, pg, equal_var=False, nan_policy="omit")
                fig.text(0.07, y, f"WG vs PG: p(Welch)={t.pvalue:.4g}", fontsize=10)
                y -= 0.03
        else:
            fig.text(0.07, y, "No qualifying pairs found.", fontsize=10)
            y -= 0.03
        pdf.savefig(fig)
        plt.close(fig)

    print(f"Generated PDF: {out_pdf}")
    print(f"Pairs analyzed: {len(results)}")
    print(f"Claim1 rows: {len(claim1_rows)} | Claim2 rows: {len(claim2_rows)}")


if __name__ == "__main__":
    main()


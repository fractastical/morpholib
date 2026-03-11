#!/usr/bin/env python3
"""
Generate Thompson-style transformation grid panels from landmark data.

Input:
  - landmarks_long.csv produced by ingest_dryad_morphometrics.py

Outputs:
  - Per-pair panel image(s): source grid, affine warp, TPS warp
  - Optional JSON manifest for batch runs

Usage examples:
  python thompson/generate_tps_transformation_panels.py \
      thompson/datasets/dryad_14fn1/landmarks_long.csv \
      --specimen-a A --specimen-b B \
      --output thompson/output/A_to_B_tps_panel.png

  python thompson/generate_tps_transformation_panels.py \
      thompson/datasets/dryad_14fn1/landmarks_long.csv \
      --pair-table thompson/pairs.csv \
      --output-dir thompson/output/panels
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


EPS = 1e-12


@dataclass
class PairSpec:
    specimen_a: str
    specimen_b: str
    label: str


def _u_func(r: np.ndarray) -> np.ndarray:
    """Thin-plate spline radial basis U(r) = r^2 log(r^2), with U(0)=0."""
    out = np.zeros_like(r, dtype=float)
    mask = r > EPS
    rr2 = r[mask] ** 2
    out[mask] = rr2 * np.log(rr2)
    return out


def _fit_affine(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """
    Fit affine map from src to dst.
    Returns matrix M (3x2) where [x, y, 1] @ M -> [X, Y].
    """
    a = np.column_stack([src, np.ones(src.shape[0])])
    m, _, _, _ = np.linalg.lstsq(a, dst, rcond=None)
    return m  # shape (3,2)


def _apply_affine(points: np.ndarray, m: np.ndarray) -> np.ndarray:
    p = np.column_stack([points, np.ones(points.shape[0])])
    return p @ m


def _fit_tps(src: np.ndarray, dst: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fit 2D TPS mapping src -> dst.
    Returns:
      w: (n,2) non-affine weights
      a: (3,2) affine terms
    """
    n = src.shape[0]
    d = src[:, None, :] - src[None, :, :]
    r = np.sqrt((d**2).sum(axis=2))
    k = _u_func(r)
    p = np.column_stack([np.ones(n), src])  # (n,3)

    top = np.hstack([k, p])
    bottom = np.hstack([p.T, np.zeros((3, 3))])
    l = np.vstack([top, bottom])  # (n+3, n+3)
    y = np.vstack([dst, np.zeros((3, 2))])  # (n+3,2)

    params = np.linalg.solve(l, y)
    w = params[:n, :]
    a = params[n:, :]
    return w, a


def _apply_tps(points: np.ndarray, src: np.ndarray, w: np.ndarray, a: np.ndarray) -> np.ndarray:
    d = points[:, None, :] - src[None, :, :]
    r = np.sqrt((d**2).sum(axis=2))
    u = _u_func(r)  # (m,n)
    p = np.column_stack([np.ones(points.shape[0]), points])  # (m,3)
    return p @ a + u @ w


def _extract_specimen_points(df: pd.DataFrame, specimen_id: str) -> pd.DataFrame:
    sub = df[df["specimen_id"] == specimen_id].copy()
    if sub.empty:
        raise ValueError(f"Specimen not found: {specimen_id}")
    # Keep 2D points only for TPS panel generation.
    sub = sub[sub["dimensions"] == 2].copy()
    if sub.empty:
        raise ValueError(f"Specimen has no 2D landmarks: {specimen_id}")
    # Ensure deterministic order by landmark index.
    sub = sub.sort_values(["landmark_index"])
    return sub


def _align_pair(df: pd.DataFrame, a_id: str, b_id: str) -> Tuple[np.ndarray, np.ndarray]:
    a = _extract_specimen_points(df, a_id)
    b = _extract_specimen_points(df, b_id)
    merged = a.merge(
        b,
        on="landmark_index",
        suffixes=("_a", "_b"),
        how="inner",
    ).sort_values("landmark_index")
    if merged.empty:
        raise ValueError(f"No shared landmark indices between {a_id} and {b_id}")
    src = merged[["x_a", "y_a"]].to_numpy(float)
    dst = merged[["x_b", "y_b"]].to_numpy(float)
    if src.shape[0] < 3:
        raise ValueError("Need at least 3 shared landmarks for affine/TPS.")
    return src, dst


def _grid_in_bbox(points: np.ndarray, n_lines: int = 12, pad: float = 0.1) -> Tuple[List[np.ndarray], np.ndarray]:
    x_min, y_min = points.min(axis=0)
    x_max, y_max = points.max(axis=0)
    dx = x_max - x_min
    dy = y_max - y_min
    x_min -= dx * pad
    x_max += dx * pad
    y_min -= dy * pad
    y_max += dy * pad

    xs = np.linspace(x_min, x_max, n_lines)
    ys = np.linspace(y_min, y_max, n_lines)
    t = np.linspace(0.0, 1.0, 200)

    lines: List[np.ndarray] = []
    # vertical lines
    for x in xs:
        y = y_min + t * (y_max - y_min)
        line = np.column_stack([np.full_like(y, x), y])
        lines.append(line)
    # horizontal lines
    for y in ys:
        x = x_min + t * (x_max - x_min)
        line = np.column_stack([x, np.full_like(x, y)])
        lines.append(line)

    bbox = np.array([[x_min, y_min], [x_max, y_max]], dtype=float)
    return lines, bbox


def _plot_panel(
    src: np.ndarray,
    dst: np.ndarray,
    affine_m: np.ndarray,
    tps_w: np.ndarray,
    tps_a: np.ndarray,
    label: str,
    output_path: Path,
    n_grid: int = 12,
    pad: float = 0.1,
) -> None:
    lines, _ = _grid_in_bbox(src, n_lines=n_grid, pad=pad)

    fig, axs = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)

    # Panel 1: source landmarks + source grid
    ax = axs[0]
    for line in lines:
        ax.plot(line[:, 0], line[:, 1], color="#4c78a8", alpha=0.65, linewidth=0.8)
    ax.scatter(src[:, 0], src[:, 1], c="#1f77b4", s=16, label="Source landmarks")
    ax.set_title("Source Form + Grid")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.2)

    # Panel 2: affine warp
    ax = axs[1]
    for line in lines:
        warped = _apply_affine(line, affine_m)
        ax.plot(warped[:, 0], warped[:, 1], color="#f58518", alpha=0.65, linewidth=0.8)
    ax.scatter(dst[:, 0], dst[:, 1], c="#d62728", s=16, label="Target landmarks")
    ax.set_title("Affine Warp Grid")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.2)

    # Panel 3: TPS warp
    ax = axs[2]
    for line in lines:
        warped = _apply_tps(line, src, tps_w, tps_a)
        ax.plot(warped[:, 0], warped[:, 1], color="#54a24b", alpha=0.65, linewidth=0.8)
    ax.scatter(dst[:, 0], dst[:, 1], c="#d62728", s=16, label="Target landmarks")
    ax.set_title("TPS Warp Grid")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.2)

    fig.suptitle(f"Thompson-Style Transformation: {label}", fontsize=13)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _read_pair_table(path: Path) -> List[PairSpec]:
    df = pd.read_csv(path)
    required = {"specimen_a", "specimen_b"}
    if not required.issubset(df.columns):
        raise ValueError("pair-table must include columns: specimen_a, specimen_b (optional: label)")
    pairs: List[PairSpec] = []
    for idx, row in df.iterrows():
        a = str(row["specimen_a"])
        b = str(row["specimen_b"])
        label = str(row["label"]) if "label" in df.columns and pd.notna(row["label"]) else f"{a}_to_{b}"
        pairs.append(PairSpec(specimen_a=a, specimen_b=b, label=label))
    return pairs


def generate_for_pair(
    landmarks_df: pd.DataFrame,
    pair: PairSpec,
    output_path: Path,
    n_grid: int,
    pad: float,
) -> Dict:
    src, dst = _align_pair(landmarks_df, pair.specimen_a, pair.specimen_b)
    affine_m = _fit_affine(src, dst)
    tps_w, tps_a = _fit_tps(src, dst)
    _plot_panel(
        src=src,
        dst=dst,
        affine_m=affine_m,
        tps_w=tps_w,
        tps_a=tps_a,
        label=pair.label,
        output_path=output_path,
        n_grid=n_grid,
        pad=pad,
    )
    return {
        "specimen_a": pair.specimen_a,
        "specimen_b": pair.specimen_b,
        "label": pair.label,
        "n_landmarks_shared": int(src.shape[0]),
        "output_path": str(output_path.resolve()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate Thompson-style affine + TPS deformation grid panels."
    )
    parser.add_argument("landmarks_csv", help="Path to landmarks_long.csv")
    parser.add_argument("--specimen-a", help="Source specimen_id")
    parser.add_argument("--specimen-b", help="Target specimen_id")
    parser.add_argument("--pair-table", help="CSV with specimen_a,specimen_b[,label]")
    parser.add_argument("--output", help="Output image path (single pair mode)")
    parser.add_argument("--output-dir", help="Output directory (pair-table mode)")
    parser.add_argument("--manifest", help="Optional manifest JSON path for batch mode")
    parser.add_argument("--grid-lines", type=int, default=12, help="Number of horizontal/vertical grid lines")
    parser.add_argument("--pad", type=float, default=0.1, help="Grid bbox padding fraction")
    args = parser.parse_args()

    landmarks_path = Path(args.landmarks_csv)
    if not landmarks_path.exists():
        raise FileNotFoundError(f"landmarks_csv not found: {landmarks_path}")
    df = pd.read_csv(landmarks_path)
    needed_cols = {"specimen_id", "landmark_index", "x", "y", "dimensions"}
    if not needed_cols.issubset(df.columns):
        raise ValueError(f"landmarks_csv missing columns: {sorted(needed_cols - set(df.columns))}")

    if args.pair_table:
        pair_specs = _read_pair_table(Path(args.pair_table))
        if not pair_specs:
            raise ValueError("pair-table is empty")
        out_dir = Path(args.output_dir) if args.output_dir else Path("thompson/output/tps_panels")
        out_dir.mkdir(parents=True, exist_ok=True)
        results: List[Dict] = []
        for pair in pair_specs:
            safe_label = "".join(ch if ch.isalnum() or ch in ("_", "-", ".") else "_" for ch in pair.label)
            out_path = out_dir / f"{safe_label}.png"
            result = generate_for_pair(df, pair, out_path, args.grid_lines, args.pad)
            results.append(result)
            print(f"Wrote {out_path}")

        if args.manifest:
            manifest_path = Path(args.manifest)
        else:
            manifest_path = out_dir / "tps_panels_manifest.json"
        manifest_path.write_text(json.dumps({"panels": results}, indent=2), encoding="utf-8")
        print(f"Wrote manifest {manifest_path}")
        return

    # Single-pair mode
    if not args.specimen_a or not args.specimen_b:
        # Fallback: first two specimen IDs from file for quick start.
        specimen_ids = list(dict.fromkeys(df["specimen_id"].astype(str).tolist()))
        if len(specimen_ids) < 2:
            raise ValueError("Need at least two specimen IDs in landmarks_csv")
        specimen_a = specimen_ids[0]
        specimen_b = specimen_ids[1]
    else:
        specimen_a = args.specimen_a
        specimen_b = args.specimen_b

    label = f"{specimen_a}_to_{specimen_b}"
    pair = PairSpec(specimen_a=specimen_a, specimen_b=specimen_b, label=label)
    output_path = Path(args.output) if args.output else Path("thompson/output") / f"{label}_tps_panel.png"
    result = generate_for_pair(df, pair, output_path, args.grid_lines, args.pad)
    print(f"Wrote {output_path}")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()


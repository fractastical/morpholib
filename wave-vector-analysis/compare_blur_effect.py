#!/usr/bin/env python3
"""
Is the Gaussian pre-smoothing blur necessary?

Regenerates the pixel-vector + wave roll-up pipeline twice for each video -- once
WITH the 3x3 Gaussian blur (the production default) and once WITHOUT it
(raw-frame detection) -- then quantifies how the blur changes the result.

The blur lives in `find_bright_points` in generate_pixel_brightness_vectors.py,
just before local-maxima detection. The question this script answers: does that
smoothing actually buy us anything, or could we drop it?

Metrics per video / variant
----------------------------
  n_detections       total bright pixels written (current-frame detections)
  matched_frac       fraction of detections linked to the previous frame
  med_displacement   median 1-frame displacement of matched pixels (px)
  coherence_R        mean resultant length of matched motion directions
                     (per-frame, brightness-weighted, averaged over frames).
                     0 = directions random (noise), 1 = perfectly aligned.
  n_waves            coherent waves recovered by the roll-up step

Interpretation
---------------
If dropping the blur mostly inflates `n_detections` while *lowering*
`coherence_R` (and shuffling `n_waves`), the extra detections are noise and the
blur is doing useful denoising. If the two variants are near-identical, the blur
is cosmetic and could be removed.

Usage
-----
  python compare_blur_effect.py                 # 6 no-contact WGD videos
  python compare_blur_effect.py --all           # every *_C-*.tif under --root
  python compare_blur_effect.py --video WGD82_C-oHTHT-pT-Zoom5.60
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tifffile as tiff

HERE = Path(__file__).resolve().parent
DEFAULT_ROOT = Path(
    "/Users/jdietz/Library/CloudStorage/Box-Box/Calcium videos/Calcium videos"
)
DEFAULT_OUT = HERE / "analysis_results" / "blur_compare"
DEFAULT_CATALOG = HERE / "analysis_results" / "wave_catalog"

# Same detection params the wave catalog was built with, so the reused blurred
# deltas are apples-to-apples with the freshly generated no-blur deltas.
THRESHOLD = 0.4
MAX_POINTS = 1500
MAX_LINK = 12.0

NO_CONTACT = [
    "WGD81_C-oHTHT-pT-Zoom5.60",
    "WGD82_C-oHTHT-pT-Zoom5.60",
    "WGD83_C-oHTHT-pT-Zoom5.60",
    "WGD84_C-oHTHT-pT-Zoom5.60",
    "WGD85_C-oHTHT-pT-Zoom5.60",
    "WGD86_C-oHTHT-pT-Zoom5.60",
]


def run(cmd: list) -> bool:
    print("  $", " ".join(str(c) for c in cmd))
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print(res.stdout[-600:])
        print(res.stderr[-900:])
    return res.returncode == 0


def build_tiff_index(root: Path) -> dict:
    idx = {}
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in (".tif", ".tiff"):
            idx.setdefault(p.stem.lower(), p)
            idx.setdefault(p.name.lower(), p)
    return idx


def gen_deltas(video: Path, out_dir: Path, blur: bool) -> Path:
    deltas = out_dir / "pixel_frame_deltas.csv"
    if deltas.exists():
        return deltas
    cmd = [
        sys.executable, str(HERE / "generate_pixel_brightness_vectors.py"),
        str(video), "0", "--fps", "1.0",
        "--output-dir", str(out_dir),
        "--linear-threshold", str(THRESHOLD),
        "--max-points", str(MAX_POINTS),
        "--max-link-px", str(MAX_LINK),
        "--png-every", "0",
    ]
    if not blur:
        cmd.append("--no-blur")
    if not run(cmd) or not deltas.exists():
        raise RuntimeError(f"vector step failed for {video.name} (blur={blur})")
    return deltas


def gen_waves(deltas: Path, h: int, w: int) -> Path:
    waves = deltas.parent / "waves" / "wave_events.csv"
    if waves.exists():
        return waves
    cmd = [
        sys.executable, str(HERE / "rollup_pixel_vectors_to_waves.py"),
        str(deltas), "--img-height", str(h), "--img-width", str(w),
    ]
    if not run(cmd) or not waves.exists():
        raise RuntimeError(f"rollup failed for {deltas}")
    return waves


def per_frame_coherence(df: pd.DataFrame) -> float:
    """Brightness-weighted mean resultant length, averaged over frames.

    For each frame: R = |sum w_i * u_i| / sum w_i, where u_i are unit motion
    vectors of matched pixels and w_i their current linear brightness.
    """
    m = df[df["matched_from_last"] == True].copy()  # noqa: E712
    if m.empty:
        return float("nan")
    disp = np.hypot(m["delta_x_px"].to_numpy(float), m["delta_y_px"].to_numpy(float))
    ok = disp > 1e-6
    m = m[ok]
    disp = disp[ok]
    if m.empty:
        return float("nan")
    ux = m["delta_x_px"].to_numpy(float) / disp
    uy = m["delta_y_px"].to_numpy(float) / disp
    w = m["brightness_curr_linear"].to_numpy(float)
    frames = m["frame_idx"].to_numpy()
    rs = []
    for fi in np.unique(frames):
        sel = frames == fi
        if sel.sum() < 3:
            continue
        wf = w[sel]
        sw = wf.sum()
        if sw <= 0:
            continue
        rx = (wf * ux[sel]).sum() / sw
        ry = (wf * uy[sel]).sum() / sw
        rs.append(float(np.hypot(rx, ry)))
    return float(np.mean(rs)) if rs else float("nan")


def metrics(deltas: Path, waves: Path) -> dict:
    df = pd.read_csv(deltas)
    matched = df["matched_from_last"] == True  # noqa: E712
    n_det = len(df)
    n_match = int(matched.sum())
    disp = df.loc[matched, "displacement_px"]
    disp = pd.to_numeric(disp, errors="coerce").dropna()
    try:
        n_waves = len(pd.read_csv(waves))
    except Exception:
        n_waves = 0
    return {
        "n_detections": n_det,
        "n_matched": n_match,
        "matched_frac": n_match / n_det if n_det else float("nan"),
        "med_displacement": float(disp.median()) if len(disp) else float("nan"),
        "coherence_R": per_frame_coherence(df),
        "n_waves": n_waves,
    }


def process(stem: str, video: Path, out_root: Path, catalog: Path) -> list[dict]:
    with tiff.TiffFile(video) as tf:
        h, w = tf.pages[0].shape[:2]

    rows = []
    # WITH blur: reuse catalog deltas if present (same params, old 3x3 blur),
    # otherwise regenerate.
    cat_deltas = catalog / stem / "pixel_frame_deltas.csv"
    cat_waves = catalog / stem / "waves" / "wave_events.csv"
    if cat_deltas.exists():
        blur_deltas = cat_deltas
        blur_waves = cat_waves if cat_waves.exists() else gen_waves(cat_deltas, h, w)
        src = "catalog (reused)"
    else:
        bdir = out_root / "blur" / stem
        bdir.mkdir(parents=True, exist_ok=True)
        blur_deltas = gen_deltas(video, bdir, blur=True)
        blur_waves = gen_waves(blur_deltas, h, w)
        src = "fresh"
    rows.append({"video": stem, "variant": "blur", "source": src,
                 **metrics(blur_deltas, blur_waves)})

    # WITHOUT blur: always fresh.
    ndir = out_root / "noblur" / stem
    ndir.mkdir(parents=True, exist_ok=True)
    nb_deltas = gen_deltas(video, ndir, blur=False)
    nb_waves = gen_waves(nb_deltas, h, w)
    rows.append({"video": stem, "variant": "noblur", "source": "fresh",
                 **metrics(nb_deltas, nb_waves)})
    return rows


def render_figure(df: pd.DataFrame, out_png: Path):
    vids = sorted(df["video"].unique())
    fields = [
        ("n_detections", "Detections / video", "count"),
        ("matched_frac", "Matched fraction", "fraction"),
        ("coherence_R", "Direction coherence R", "0 (noise) → 1 (aligned)"),
        ("n_waves", "Waves recovered", "count"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    x = np.arange(len(vids))
    width = 0.38
    for ax, (field, title, ylab) in zip(axes.ravel(), fields):
        b = [df[(df.video == v) & (df.variant == "blur")][field].values[0] for v in vids]
        n = [df[(df.video == v) & (df.variant == "noblur")][field].values[0] for v in vids]
        ax.bar(x - width / 2, b, width, label="with blur (3×3)", color="#2c7fb8")
        ax.bar(x + width / 2, n, width, label="no blur (raw)", color="#d95f0e")
        ax.set_title(title, fontsize=12)
        ax.set_ylabel(ylab)
        ax.set_xticks(x)
        ax.set_xticklabels([v.split("_")[0] for v in vids], rotation=0)
        ax.legend(fontsize=9)
    fig.suptitle("Effect of the Gaussian pre-smoothing blur on detection & waves",
                 fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"Wrote figure → {out_png}")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    ap.add_argument("--video", action="append", help="specific stem(s)")
    ap.add_argument("--all", action="store_true",
                    help="all *_C-*.tif under --root")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    idx = build_tiff_index(args.root)

    if args.all:
        stems = sorted({p.stem for p in args.root.rglob("*")
                        if p.is_file() and p.suffix.lower() in (".tif", ".tiff")
                        and "_C-" in p.name})
    elif args.video:
        stems = args.video
    else:
        stems = NO_CONTACT

    all_rows = []
    for i, stem in enumerate(stems, 1):
        video = idx.get(stem.lower())
        if video is None:
            print(f"[{i}/{len(stems)}] {stem}: TIFF not found, skipping")
            continue
        print(f"[{i}/{len(stems)}] {stem}")
        try:
            all_rows.extend(process(stem, video, args.out, args.catalog))
        except Exception as e:  # noqa: BLE001
            print(f"  ! failed: {e}")

    if not all_rows:
        print("No results.")
        return

    df = pd.DataFrame(all_rows)
    csv_path = args.out / "blur_comparison.csv"
    df.to_csv(csv_path, index=False)
    render_figure(df, args.out / "blur_comparison.png")

    # Console summary: aggregate ratios noblur / blur.
    piv = df.pivot_table(index="video", columns="variant",
                         values=["n_detections", "coherence_R", "n_waves",
                                 "matched_frac"])
    print("\n=== Per-video (blur vs noblur) ===")
    with pd.option_context("display.width", 160, "display.max_columns", 20):
        print(piv.round(3))

    det_ratio = (df[df.variant == "noblur"]["n_detections"].mean()
                 / df[df.variant == "blur"]["n_detections"].mean())
    coh_blur = df[df.variant == "blur"]["coherence_R"].mean()
    coh_nb = df[df.variant == "noblur"]["coherence_R"].mean()
    print("\n=== Aggregate ===")
    print(f"  detections  noblur/blur : {det_ratio:.2f}x")
    print(f"  coherence R   blur      : {coh_blur:.3f}")
    print(f"  coherence R   noblur    : {coh_nb:.3f}")
    print(f"\nWrote {csv_path}")


if __name__ == "__main__":
    main()

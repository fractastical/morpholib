#!/usr/bin/env python3
"""Odonata wing cells through deep time: fossil vs extant venation geometry.

Dragonflies and damselflies (Odonata) are the one insect lineage for which we
have *the same measurement* on fossil and living wings: the size of the cells
(vein-bounded domains) that our developmental model
(``2018-hoffmann-wing-venation.py``) generates from a single inhibition length
scale. This script puts the two records side by side.

Data (see ``datasets/odonata/README.md`` for provenance; nothing is hand-entered)
------------------------------------------------------------------------------
* Eshghi S, Buesch A, Gorb S (2026). Critical Crack Length as an Evolutionary
  Constant in Odonata Wing Venation: Lessons from Fossils — Supplementary
  Material. Zenodo. https://doi.org/10.5281/zenodo.19899648 (CC BY 4.0).
  69 fossil wings (22 species, Late Jurassic–Eocene) + 394 extant wings
  (204 species): wing length, cell count, cell-length statistics, and
  venation drawings of every fossil wing.
* Paleobiology Database (https://paleobiodb.org): ages/formations of the 22
  fossil species and all Odonatoptera occurrences (Carboniferous→Recent) for
  the genus-richness backdrop.
* Hoffmann J, Donoughe S, Li K, Salcedo MK, Rycroft CH (2018). PNAS
  115(40):9905–9910. https://doi.org/10.1073/pnas.1721248115 — segmented
  North American dragonfly wings (GitHub), shown as the extant counterpart of
  the Eocene Aeshnidae / Gomphidae fossils, and used to cross-check cell
  counts for the 42 species both datasets segmented independently.

What is tested
--------------
If the venation model's inhibition length were a conserved developmental
constant, cell size would be independent of wing size and of geological age,
and cell count would scale isometrically with wing area (N ∝ L²). The script
fits these scalings on fossil and extant wings separately and reports them;
the figure and animation show what the data say rather than the hypothesis.

Usage
-----
    python3 2026-eshghi-odonata-wing-cells.py                 # figure + animation (downloads ~100 MB once)
    python3 2026-eshghi-odonata-wing-cells.py --no-anim        # statistics + static figure only (no downloads)
    python3 2026-eshghi-odonata-wing-cells.py --zenodo-zip /path/to/archive.zip --hoffmann-tar /path/to/Dragonfly_NA_WingSegmentation.tar.gz

Outputs (git-ignored): odonata_wing_cells_scaling.png, odonata_wing_cells_deep_time.gif
Downloads are cached in .wing_cache/odonata/ (git-ignored).
"""

from __future__ import annotations

import argparse
import io
import math
import os
import re
import sys
import tarfile
import urllib.request
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib

if not os.environ.get("DISPLAY") and sys.platform != "darwin":
    matplotlib.use("Agg")

import matplotlib.animation as animation  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from PIL import Image  # noqa: E402
from scipy import ndimage  # noqa: E402

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "datasets" / "odonata"
CACHE_DIR = HERE / ".wing_cache" / "odonata"

ZENODO_ZIP_URL = ("https://zenodo.org/api/records/19899648/files/"
                  "Eshghi%20et%20al.%2C%202026%2C%20Supplementary%20Material%2C%20FossilWing%20CellSize.zip/content")
ZENODO_ZIP_BYTES = 89_961_457
HOFFMANN_TAR_URL = ("https://github.com/hoffmannjordan/insect-wing-venation-patterns/raw/master/"
                    "Dragonfly_NA_WingSegmentation.tar.gz")

GROUP_LABEL = {"fossil": "fossil (Jurassic–Eocene)", "extant_dragonfly": "extant dragonflies",
               "extant_damselfly": "extant damselflies"}
GROUP_COLOR = {"fossil": "#b5541c", "extant_dragonfly": "#1f5fa8", "extant_damselfly": "#3a9d5d"}
CELL_BAND = (0.5, 1.5)  # mm, the "dominant cell-size range" of Eshghi et al.

PERIODS = [  # ICS 2023 boundaries (Ma); only what the axis needs
    ("Carboniferous", 358.9, 298.9), ("Permian", 298.9, 251.9), ("Triassic", 251.9, 201.4),
    ("Jurassic", 201.4, 145.0), ("Cretaceous", 145.0, 66.0), ("Paleogene", 66.0, 23.0),
    ("Neogene", 23.0, 2.6), ("Q.", 2.6, 0.0),
]


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------
def load_wings(data_dir: Path) -> pd.DataFrame:
    df = pd.read_csv(data_dir / "eshghi2026_odonata_wing_cells.csv")
    df["family"] = df["family"].fillna("")
    return df


def load_pbdb(data_dir: Path) -> pd.DataFrame:
    return pd.read_csv(data_dir / "pbdb_odonatoptera_occurrences.csv")


def genus_richness(pbdb: pd.DataFrame, bin_myr: float = 10.0, oldest: float = 330.0) -> Tuple[np.ndarray, np.ndarray]:
    """Range-through genus counts per time bin (a genus counts in every bin it spans)."""
    g = pbdb.dropna(subset=["genus"]).groupby("genus").agg(max_ma=("max_ma", "max"), min_ma=("min_ma", "min"))
    edges = np.arange(0.0, oldest + bin_myr, bin_myr)
    mids = (edges[:-1] + edges[1:]) / 2
    counts = np.array([((g.max_ma >= lo) & (g.min_ma <= hi)).sum() for lo, hi in zip(edges[:-1], edges[1:])])
    return mids, counts


def ols_loglog(x: pd.Series, y: pd.Series) -> Tuple[float, float, float]:
    lx, ly = np.log10(x.to_numpy(float)), np.log10(y.to_numpy(float))
    slope, intercept = np.polyfit(lx, ly, 1)
    r = float(np.corrcoef(lx, ly)[0, 1])
    return float(slope), float(intercept), r


def report(df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    print("\nScaling of cell geometry with wing length (OLS on log10 values)")
    print(f"{'group':26s} {'n':>4s} {'L range (mm)':>14s} {'N~L slope':>10s} {'r':>5s} {'cell~L slope':>13s} "
          f"{'median cell (mm)':>17s} {'IQR':>12s}")
    for grp, d in df.groupby("group"):
        s_n, _, r_n = ols_loglog(d.wing_length_mm, d.cell_count)
        s_c, _, _ = ols_loglog(d.wing_length_mm, d.cell_len_median_mm)
        q1, q2, q3 = d.cell_len_median_mm.quantile([0.25, 0.5, 0.75])
        out[grp] = {"n": len(d), "slope_count": s_n, "r_count": r_n, "slope_cell": s_c, "median_cell": q2}
        print(f"{GROUP_LABEL[grp]:26s} {len(d):4d} {d.wing_length_mm.min():6.1f}–{d.wing_length_mm.max():5.1f}"
              f" {s_n:10.2f} {r_n:5.2f} {s_c:13.2f} {q2:17.2f} {q1:5.2f}–{q3:4.2f}")
    print("  isometric (constant cell size) prediction: N~L slope 2.00, cell~L slope 0.00")
    fo = df[df.group == "fossil"]
    print("\nFossil wings by age (PBDB midpoint):")
    for (age, form), d in fo.groupby(["age_ma", "formation"]):
        print(f"  {age:6.1f} Ma  {str(form):18s} n={len(d):2d} species={d.species.nunique():2d} "
              f"median cell {d.cell_len_median_mm.median():.2f} mm, L {d.wing_length_mm.min():.0f}–{d.wing_length_mm.max():.0f} mm")
    print("\nSame family, fossil vs extant (median cell length, mm):")
    for fam in ["Aeshnidae", "Gomphidae"]:
        f_ = fo[fo.family == fam]
        e_ = df[(df.group != "fossil") & (df.family == fam)]
        print(f"  {fam:10s} fossil n={len(f_)} ({', '.join(sorted(f_.species.unique()))}): "
              f"{', '.join(f'{v:.2f}' for v in f_.cell_len_median_mm)} | extant n={len(e_)} "
              f"({e_.species.nunique()} spp): median {e_.cell_len_median_mm.median():.2f}, "
              f"IQR {e_.cell_len_median_mm.quantile(.25):.2f}–{e_.cell_len_median_mm.quantile(.75):.2f}")
    return out


# --------------------------------------------------------------------------
# Downloads (cached, resumable)
# --------------------------------------------------------------------------
def fetch(url: str, dest: Path, expected: Optional[int] = None, allow_download: bool = True) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and (expected is None or dest.stat().st_size == expected):
        return dest
    if not allow_download:
        sys.exit(f"{dest.name} not cached and downloads disabled (--no-download). Pass the file explicitly.")
    print(f"  downloading {url}\n    -> {dest}")
    while True:
        have = dest.stat().st_size if dest.exists() else 0
        req = urllib.request.Request(url, headers={"User-Agent": "morpholib/odonata", "Range": f"bytes={have}-"})
        try:
            with urllib.request.urlopen(req, timeout=120) as resp, open(dest, "ab") as fh:
                if resp.status == 200 and have:  # server ignored Range: start over
                    fh.seek(0)
                    fh.truncate()
                while True:
                    chunk = resp.read(1 << 20)
                    if not chunk:
                        break
                    fh.write(chunk)
        except Exception as exc:  # resume on transient failures
            size = dest.stat().st_size if dest.exists() else 0
            if expected and size < expected:
                print(f"    interrupted at {size / 1e6:.1f} MB ({exc}); resuming")
                continue
            if size == 0:
                raise
        size = dest.stat().st_size
        if expected is None or size >= expected:
            return dest
        print(f"    {size / 1e6:.1f} / {expected / 1e6:.1f} MB; resuming")


# --------------------------------------------------------------------------
# Images
# --------------------------------------------------------------------------
def fossil_drawing(zip_path: Path, image_name: str) -> Optional[np.ndarray]:
    """Return the fossil venation drawing as a float mask-like array (1 = ink), scale bar removed."""
    with zipfile.ZipFile(zip_path) as zf:
        cands = [n for n in zf.namelist() if "Fossil Wing Images/" in n and Path(n).stem == image_name]
        if not cands:
            return None
        with zf.open(cands[0]) as fh:
            im = Image.open(io.BytesIO(fh.read())).convert("LA")
    arr = np.asarray(im, dtype=float)
    ink = (arr[..., 0] < 128) & (arr[..., 1] > 0)  # dark and not transparent
    # keep only the largest connected component (the wing); drops the scale bar + text
    lab, n = ndimage.label(ndimage.binary_dilation(ink, iterations=2))
    if n == 0:
        return None
    sizes = ndimage.sum(np.ones_like(lab), lab, index=np.arange(1, n + 1))
    wing = lab == (int(np.argmax(sizes)) + 1)
    ink &= wing
    rows = np.where(ink.any(axis=1))[0]
    cols = np.where(ink.any(axis=0))[0]
    return ink[rows.min():rows.max() + 1, cols.min():cols.max() + 1].astype(float)


def hoffmann_domains(tar_path: Path, species: str) -> Optional[np.ndarray]:
    """Label matrix of a Hoffmann et al. segmented wing (right forewing preferred), cropped."""
    key = species.lower().replace(" ", "-")
    with tarfile.open(tar_path) as tf:
        members = [m for m in tf.getmembers() if key in m.name and m.name.endswith("_FMM_seg_with_MASK.csv")]
        if not members:
            return None
        members.sort(key=lambda m: ("-RF_" not in m.name, m.name))
        data = tf.extractfile(members[0]).read().decode()
    lab = np.loadtxt(io.StringIO(data), delimiter=",")
    labels, counts = np.unique(lab, return_counts=True)
    bg = labels[np.argmax(counts)]
    wing = lab != bg
    rows = np.where(wing.any(axis=1))[0]
    cols = np.where(wing.any(axis=0))[0]
    lab = lab[rows.min():rows.max() + 1, cols.min():cols.max() + 1]
    lab = np.where(lab == bg, -1, lab)
    return lab


def hoffmann_domain_count(tar_path: Path, species: str) -> Optional[int]:
    lab = hoffmann_domains(tar_path, species)
    if lab is None:
        return None
    return int(len(np.unique(lab[lab >= 0])))


def domain_boundaries(lab: np.ndarray) -> np.ndarray:
    """Ink mask of domain boundaries (veins) from a label matrix, drawn like a venation tracing."""
    edge = np.zeros(lab.shape, dtype=bool)
    edge[:, 1:] |= lab[:, 1:] != lab[:, :-1]
    edge[1:, :] |= lab[1:, :] != lab[:-1, :]
    edge &= ndimage.binary_dilation(lab >= 0, iterations=1)
    return ndimage.binary_dilation(edge, iterations=1).astype(float)


def show_wing_mm(ax, ink: np.ndarray, wing_length_mm: float, y0: float = 0.0) -> None:
    """Draw an ink mask so that its horizontal extent equals ``wing_length_mm`` (axes in mm)."""
    h, w = ink.shape[:2]
    mm_per_px = wing_length_mm / w
    rgba = np.zeros(ink.shape + (4,))
    rgba[..., 3] = ink  # transparent background so wings can be stacked
    ax.imshow(rgba, extent=(0, w * mm_per_px, y0, y0 + h * mm_per_px), interpolation="bilinear")
    ax.set_xlabel("mm", fontsize=8)
    ax.set_yticks([])
    ax.tick_params(labelsize=8)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)


# --------------------------------------------------------------------------
# Static figure
# --------------------------------------------------------------------------
def scaling_figure(df: pd.DataFrame, out: Path, hoffmann_counts: Optional[Dict[str, int]] = None) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    (ax_n, ax_c), (ax_t, ax_f) = axes

    # (a) cell count vs wing length
    for grp, d in df.groupby("group"):
        s, b, r = ols_loglog(d.wing_length_mm, d.cell_count)
        ax_n.scatter(d.wing_length_mm, d.cell_count, s=14, alpha=0.6, color=GROUP_COLOR[grp],
                     label=f"{GROUP_LABEL[grp]} (n={len(d)}): N ∝ L^{s:.2f}, r={r:.2f}")
        xs = np.linspace(d.wing_length_mm.min(), d.wing_length_mm.max(), 50)
        ax_n.plot(xs, 10 ** b * xs ** s, color=GROUP_COLOR[grp], lw=1.5)
    ext = df[df.group != "fossil"]
    k = (ext.cell_count / ext.wing_length_mm ** 2).median()
    xs = np.linspace(8, 70, 50)
    ax_n.plot(xs, k * xs ** 2, "k--", lw=1, label="isometric N ∝ L² (constant cell size)")
    ax_n.set_xscale("log")
    ax_n.set_yscale("log")
    ax_n.set_xlabel("wing length L (mm)")
    ax_n.set_ylabel("segmented cells per wing N")
    ax_n.legend(fontsize=7.5, loc="upper left")
    ax_n.set_title("(a) cell count scales sub-isometrically with wing size", fontsize=10, loc="left")

    # (b) median cell length vs wing length
    ax_c.axhspan(*CELL_BAND, color="#f2e8c9", zorder=0, label="0.5–1.5 mm band (Eshghi et al.)")
    for grp, d in df.groupby("group"):
        s, b, _ = ols_loglog(d.wing_length_mm, d.cell_len_median_mm)
        ax_c.scatter(d.wing_length_mm, d.cell_len_median_mm, s=14, alpha=0.6, color=GROUP_COLOR[grp],
                     label=f"{GROUP_LABEL[grp]}: median cell ∝ L^{s:.2f}")
        xs = np.linspace(d.wing_length_mm.min(), d.wing_length_mm.max(), 50)
        ax_c.plot(xs, 10 ** b * xs ** s, color=GROUP_COLOR[grp], lw=1.5)
    ax_c.scatter(df.wing_length_mm, df.cell_len_max_mm, s=6, alpha=0.25, color="#777777", label="largest cell per wing")
    ax_c.set_xscale("log")
    ax_c.set_yscale("log")
    ax_c.set_xlabel("wing length L (mm)")
    ax_c.set_ylabel("cell length (mm)")
    ax_c.legend(fontsize=7.5, loc="upper left")
    ax_c.set_title("(b) typical cell size is nearly independent of wing size; the largest cells are not",
                   fontsize=10, loc="left")

    # (c) median cell length through time
    ax_t.axhspan(*CELL_BAND, color="#f2e8c9", zorder=0)
    fo = df[df.group == "fossil"]
    rng = np.random.default_rng(1)
    ax_t.scatter(fo.age_ma + rng.normal(0, 1.2, len(fo)), fo.cell_len_median_mm, s=18, color=GROUP_COLOR["fossil"],
                 alpha=0.75, label="fossil wings (age = PBDB midpoint, jittered)")
    for grp, x0 in (("extant_dragonfly", 4.0), ("extant_damselfly", -4.0)):
        d = df[df.group == grp]
        ax_t.boxplot([d.cell_len_median_mm], positions=[x0], widths=6, vert=True, showfliers=False, manage_ticks=False,
                     medianprops=dict(color=GROUP_COLOR[grp]), boxprops=dict(color=GROUP_COLOR[grp]),
                     whiskerprops=dict(color=GROUP_COLOR[grp]), capprops=dict(color=GROUP_COLOR[grp]))
        ax_t.text(x0, d.cell_len_median_mm.quantile(0.75) + 0.05, GROUP_LABEL[grp].split()[-1],
                  ha="center", fontsize=7, color=GROUP_COLOR[grp])
    for name, lo, hi in PERIODS:
        if hi < 170:
            ax_t.axvline(hi, color="#cccccc", lw=0.6, zorder=0)
            ax_t.text((min(lo, 170) + hi) / 2, 1.9, name, ha="center", fontsize=7, color="#555555")
    for k, (age, d) in enumerate(fo.groupby("age_ma")):
        forms = sorted({str(f) for f in d.formation if isinstance(f, str) and f != "?"}) or [str(d.interval.iloc[0])]
        ax_t.text(age, 1.62 if k % 2 else 0.3, f"{'/'.join(forms)}\nn={len(d)}", ha="center", fontsize=6.5,
                  color=GROUP_COLOR["fossil"])
    ax_t.set_xlim(170, -10)
    ax_t.set_ylim(0.25, 2.0)
    ax_t.set_xticks([150, 125, 100, 75, 50, 25, 0])
    ax_t.set_xlabel("age (Ma)")
    ax_t.set_ylabel("median cell length per wing (mm)")
    ax_t.legend(fontsize=7.5, loc="lower center")
    ax_t.set_title("(c) 150 Myr of Odonata: the typical cell stays in the same half-millimetre band",
                   fontsize=10, loc="left")

    # (d) same families, fossil vs extant; plus independent-segmentation cross-check
    pos = 0
    ticks, labels = [], []
    for fam in ["Aeshnidae", "Gomphidae"]:
        e_ = df[(df.group != "fossil") & (df.family == fam)]
        f_ = fo[fo.family == fam]
        ax_f.boxplot([e_.cell_len_median_mm], positions=[pos], widths=0.5, showfliers=False, manage_ticks=False,
                     medianprops=dict(color=GROUP_COLOR["extant_dragonfly"]))
        ax_f.scatter(np.full(len(e_), pos) + rng.normal(0, 0.06, len(e_)), e_.cell_len_median_mm, s=10,
                     color=GROUP_COLOR["extant_dragonfly"], alpha=0.5)
        ax_f.scatter(np.full(len(f_), pos + 0.9) + rng.normal(0, 0.04, len(f_)), f_.cell_len_median_mm, s=36,
                     color=GROUP_COLOR["fossil"], zorder=3)
        for _, r in f_.iterrows():
            ax_f.annotate(f"{r.species.split()[0]}\n{r.age_ma:.0f} Ma", (pos + 0.9, r.cell_len_median_mm),
                          xytext=(8, 0), textcoords="offset points", fontsize=6.5, va="center")
        ticks += [pos, pos + 0.9]
        labels += [f"{fam}\nextant (n={len(e_)})", f"{fam}\nEocene (n={len(f_)})"]
        pos += 2.6
    ax_f.axhspan(*CELL_BAND, color="#f2e8c9", zorder=0)
    ax_f.set_xticks(ticks)
    ax_f.set_xticklabels(labels, fontsize=8)
    ax_f.set_xlim(-0.6, 4.6)
    ax_f.set_ylim(0.5, 2.1)
    ax_f.set_ylabel("median cell length per wing (mm)")
    ax_f.set_title("(d) same families 50 Myr apart", fontsize=10, loc="left")
    if hoffmann_counts:
        ins = ax_f.inset_axes([0.36, 0.6, 0.34, 0.38])
        sp = [s for s in hoffmann_counts if s in set(df.species)]
        xs_ = [df[(df.species == s) & (df.group != "fossil")].cell_count.mean() for s in sp]
        ys_ = [hoffmann_counts[s] for s in sp]
        ins.scatter(xs_, ys_, s=10, color="#444444")
        lim = (min(xs_ + ys_) * 0.9, max(xs_ + ys_) * 1.1)
        ins.plot(lim, lim, "k--", lw=0.7)
        ins.set_xlim(lim)
        ins.set_ylim(lim)
        ins.set_xlabel("cells, Eshghi 2026", fontsize=6.5)
        ins.set_ylabel("domains, Hoffmann 2018", fontsize=6.5)
        ins.tick_params(labelsize=6)
        ins.set_title(f"two segmentations, {len(sp)} shared species", fontsize=7)

    fig.suptitle("Odonata wing cells, fossil and extant — data: Eshghi, Buesch & Gorb 2026 (Zenodo 10.5281/zenodo.19899648), "
                 "PBDB ages, Hoffmann et al. 2018", fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  saved {out}")


# --------------------------------------------------------------------------
# Animation
# --------------------------------------------------------------------------
def deep_time_animation(df: pd.DataFrame, pbdb: pd.DataFrame, zip_path: Path, tar_path: Path, out: Path,
                        fps: float = 1.0) -> None:
    fo = df[df.group == "fossil"].copy()
    ext = df[df.group != "fossil"]
    # one representative wing per species: the one with the most cells (best-preserved tracing)
    reps = fo.sort_values("cell_count", ascending=False).groupby("species").head(1)
    reps = reps.sort_values(["age_ma", "species"], ascending=[False, True])

    extant_pairs = [  # (fossil family, extant species present in both datasets)
        ("Aeshnidae", "Aeshna juncea"), ("Gomphidae", "Arigomphus villosipes"),
    ]

    frames: List[Tuple[str, object]] = [("intro", None)]
    frames += [("fossil", r) for _, r in reps.iterrows()]
    frames += [("extant", p) for p in extant_pairs]
    frames += [("final", None)] * 3

    mids, rich = genus_richness(pbdb)
    fig = plt.figure(figsize=(14, 8.2))
    gs = fig.add_gridspec(3, 2, height_ratios=[0.8, 1.7, 1.6], width_ratios=[1.0, 1.0], hspace=0.5, wspace=0.25,
                          left=0.06, right=0.97, top=0.9, bottom=0.08)
    ax_time = fig.add_subplot(gs[0, :])
    ax_wing = fig.add_subplot(gs[1, :])
    ax_cell = fig.add_subplot(gs[2, 0])
    ax_info = fig.add_subplot(gs[2, 1])

    img_cache: Dict[str, Optional[np.ndarray]] = {}
    wing_mm_max = float(fo.wing_length_mm.max()) * 1.04
    wing_mm_height = 22.0  # tallest fossil drawing (Cratostenophlebia, 64 mm) is ~20 mm deep

    def draw_time(cursor: Optional[float]) -> None:
        ax_time.clear()
        ax_time.fill_between(mids, rich, color="#d9d9d9", step="mid")
        ax_time.set_xlim(330, -5)
        ax_time.set_ylim(0, rich.max() * 1.3)
        for name, lo, hi in PERIODS:
            ax_time.axvline(hi, color="white", lw=1)
            ax_time.text((min(lo, 330) + hi) / 2, rich.max() * 1.12, name, ha="center", fontsize=7.5, color="#444444")
        ax_time.scatter(fo.age_ma, np.full(len(fo), rich.max() * 0.08), marker="|", s=140, color=GROUP_COLOR["fossil"])
        ax_time.text(160, rich.max() * 0.3, "| fossil wings with cell measurements", fontsize=7.5,
                     color=GROUP_COLOR["fossil"], ha="left")
        if cursor is not None:
            ax_time.axvline(cursor, color="#d6453d", lw=2)
        ax_time.set_ylabel("Odonatoptera genera\nper 10 Myr (PBDB)", fontsize=8)
        ax_time.set_xlabel("Ma", fontsize=8)
        ax_time.tick_params(labelsize=8)
        ax_time.spines[["top", "right"]].set_visible(False)

    def draw_cells(upto_age: Optional[float], highlight: Optional[pd.Series] = None, extant: bool = False) -> None:
        ax_cell.clear()
        ax_cell.axhspan(*CELL_BAND, color="#f2e8c9", zorder=0)
        shown = fo if upto_age is None else fo[fo.age_ma >= upto_age - 0.01]
        rng = np.random.default_rng(1)
        ax_cell.scatter(shown.age_ma + rng.normal(0, 1.2, len(shown)), shown.cell_len_median_mm, s=16,
                        color=GROUP_COLOR["fossil"], alpha=0.7)
        if highlight is not None:
            ax_cell.scatter([highlight.age_ma], [highlight.cell_len_median_mm], s=110, facecolor="none",
                            edgecolor="#d6453d", lw=2, zorder=5)
        if extant:
            for grp, x0 in (("extant_dragonfly", 4.0), ("extant_damselfly", -4.0)):
                d = df[df.group == grp]
                ax_cell.boxplot([d.cell_len_median_mm], positions=[x0], widths=6, showfliers=False, manage_ticks=False,
                                medianprops=dict(color=GROUP_COLOR[grp]), boxprops=dict(color=GROUP_COLOR[grp]),
                                whiskerprops=dict(color=GROUP_COLOR[grp]), capprops=dict(color=GROUP_COLOR[grp]))
            ax_cell.text(0, 1.58, "extant\n(n=%d)" % len(ext), ha="center", fontsize=7, color="#333333")
        ax_cell.set_xlim(165, -12)
        ax_cell.set_ylim(0.25, 1.85)
        ax_cell.set_xticks([150, 100, 50, 0])
        ax_cell.set_xlabel("age (Ma)", fontsize=8)
        ax_cell.set_ylabel("median cell length per wing (mm)", fontsize=8)
        ax_cell.tick_params(labelsize=8)
        ax_cell.set_title("typical cell size through time (band: 0.5–1.5 mm)", fontsize=9, loc="left")

    def info(text: str) -> None:
        ax_info.clear()
        ax_info.axis("off")
        ax_info.text(0.0, 1.0, text, va="top", ha="left", fontsize=9.5, transform=ax_info.transAxes, wrap=True)

    def update(i: int):
        kind, payload = frames[i]
        ax_wing.clear()
        if kind == "intro":
            draw_time(None)
            draw_cells(None)
            ax_wing.axis("off")
            info("Odonata wing cells through deep time\n\n"
                 f"{len(fo)} fossil wings · {fo.species.nunique()} extinct species · "
                 f"{fo.age_ma.max():.0f}–{fo.age_ma.min():.0f} Ma\n"
                 f"{len(ext)} extant wings · {ext.species.nunique()} living species\n\n"
                 "Each frame shows one fossil species: its venation drawing at true scale (mm),\n"
                 "and where its typical cell size falls in the whole record.\n\n"
                 "Griffinflies (Meganisoptera) dominate the Carboniferous–Permian part of the\n"
                 "PBDB record but have no cell-level measurements in this dataset.")
            fig.suptitle("Odonata: fossil and living wing venation — Eshghi, Buesch & Gorb 2026 · PBDB · Hoffmann et al. 2018",
                         fontsize=12)
            return
        if kind == "fossil":
            r = payload
            draw_time(r.age_ma)
            draw_cells(r.age_ma, highlight=r)
            if r.image_name not in img_cache:
                img_cache[r.image_name] = fossil_drawing(zip_path, r.image_name)
            img = img_cache[r.image_name]
            if img is not None:
                show_wing_mm(ax_wing, img, r.wing_length_mm)
            else:
                ax_wing.text(0.5, 0.5, "drawing not found in archive", ha="center", transform=ax_wing.transAxes)
            ax_wing.set_xlim(0, wing_mm_max)
            ax_wing.set_ylim(0, wing_mm_height)
            fam = r.family if isinstance(r.family, str) else "?"
            form = f"{r.formation} Formation" if isinstance(r.formation, str) and r.formation != "?" else "formation not recorded in PBDB"
            n_sp = (fo.species == r.species).sum()
            ax_wing.set_title(f"{r.species}  ({fam}) — venation drawing from Eshghi et al. 2026, shown at true scale",
                              fontsize=9, loc="left")
            info(f"{r.species}\n{fam}\n\n"
                 f"{form}\n{r.interval}: {r.max_ma:.1f}–{r.min_ma:.1f} Ma (PBDB)\n\n"
                 f"wing length {r.wing_length_mm:.1f} mm\n"
                 f"{int(r.cell_count)} cells segmented\n"
                 f"median cell {r.cell_len_median_mm:.2f} mm (IQR {r.cell_len_iqr_mm:.2f}), largest {r.cell_len_max_mm:.1f} mm\n"
                 f"{r.pct_cells_0p5_1p5mm:.0f}% of cells between 0.5 and 1.5 mm"
                 + (f"\n\n{n_sp} wings of this species measured; the best-preserved is shown." if n_sp > 1 else ""))
            fig.suptitle(f"{r.age_ma:.0f} Ma — {r.species}", fontsize=12)
            return
        if kind == "extant":
            fam, species = payload
            draw_time(0.0)
            draw_cells(0.0, extant=True)
            lab = hoffmann_domains(tar_path, species) if tar_path.exists() else None
            e_ = ext[ext.species == species]
            L = float(e_.wing_length_mm.mean())
            n_dom = int(len(np.unique(lab[lab >= 0]))) if lab is not None else None
            if lab is not None:
                show_wing_mm(ax_wing, domain_boundaries(lab), L)
            ax_wing.set_xlim(0, wing_mm_max)
            ax_wing.set_ylim(0, wing_mm_height)
            fossils = fo[fo.family == fam]
            ax_wing.set_title(f"{species}  ({fam}, extant) — vein domains segmented by Hoffmann et al. 2018, drawn at true scale",
                              fontsize=9, loc="left")
            info(f"{species}\n{fam}, living\n\n"
                 f"Hoffmann et al. 2018 segmentation: {n_dom if n_dom else '?'} domains\n"
                 f"Eshghi et al. 2026: wing length {L:.1f} mm, {int(e_.cell_count.mean())} cells,\n"
                 f"median cell {e_.cell_len_median_mm.mean():.2f} mm\n\n"
                 f"Eocene {fam} in the fossil set:\n" +
                 "\n".join(f"  {s}: median cell {v:.2f} mm, L {l:.0f} mm"
                           for s, v, l in zip(fossils.species, fossils.cell_len_median_mm, fossils.wing_length_mm)) +
                 f"\n\nExtant {fam} ({ext[ext.family == fam].species.nunique()} spp): median cell "
                 f"{ext[ext.family == fam].cell_len_median_mm.median():.2f} mm")
            fig.suptitle(f"Today — {fam}: the same family 50 Myr after its Eocene fossils", fontsize=12)
            return
        # final: Eocene and living Aeshnidae at the same scale
        draw_time(None)
        draw_cells(None, extant=True)
        yp = fo[fo.species == "Ypshna latipennata"].sort_values("cell_count", ascending=False).iloc[0]
        if yp.image_name not in img_cache:
            img_cache[yp.image_name] = fossil_drawing(zip_path, yp.image_name)
        img = img_cache[yp.image_name]
        lab = hoffmann_domains(tar_path, "Aeshna juncea") if tar_path.exists() else None
        if img is not None:
            show_wing_mm(ax_wing, img, yp.wing_length_mm, y0=0.0)
            ax_wing.text(yp.wing_length_mm + 1, 4, f"{yp.species}, {yp.age_ma:.0f} Ma", fontsize=8, va="center")
        if lab is not None:
            L = float(ext[ext.species == "Aeshna juncea"].wing_length_mm.mean())
            show_wing_mm(ax_wing, domain_boundaries(lab), L, y0=14.0)
            ax_wing.text(L + 1, 19, "Aeshna juncea, living", fontsize=8, va="center")
        ax_wing.set_xlim(0, wing_mm_max)
        ax_wing.set_ylim(0, 28)
        ax_wing.set_title("Aeshnidae, 52 Myr apart, at the same scale", fontsize=9, loc="left")
        s_f = ols_loglog(fo.wing_length_mm, fo.cell_count)[0]
        s_e = ols_loglog(ext.wing_length_mm, ext.cell_count)[0]
        info("What 150 Myr of wings say about the venation model\n\n"
             f"• median cell length: fossils {fo.cell_len_median_mm.median():.2f} mm "
             f"(IQR {fo.cell_len_median_mm.quantile(.25):.2f}–{fo.cell_len_median_mm.quantile(.75):.2f}); "
             f"extant {ext.cell_len_median_mm.median():.2f} mm "
             f"(IQR {ext.cell_len_median_mm.quantile(.25):.2f}–{ext.cell_len_median_mm.quantile(.75):.2f})\n\n"
             f"• cell count vs wing length: fossils N ∝ L^{s_f:.2f}, extant N ∝ L^{s_e:.2f}\n"
             "  (a strictly constant cell size would give N ∝ L²)\n\n"
             "• the typical cell is a conserved length scale across 150 Myr and a\n"
             "  6× range of wing length; the largest cells grow with the wing\n\n"
             "For 2018-hoffmann-wing-venation.py: the inhibition length is approximately\n"
             "conserved across species and time, but not strictly — large extant wings\n"
             "carry fewer, larger cells than a fixed scale would produce.")
        fig.suptitle("Odonata: fossil and living wing venation — summary", fontsize=12)

    anim = animation.FuncAnimation(fig, update, frames=len(frames), interval=1000 / fps, blit=False)
    anim.save(out, writer="pillow", fps=fps)
    plt.close(fig)
    print(f"  saved {out}  ({len(frames)} frames)")


# --------------------------------------------------------------------------
def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", default=str(DATA_DIR))
    p.add_argument("--cache-dir", default=str(CACHE_DIR))
    p.add_argument("--out-dir", default=".")
    p.add_argument("--zenodo-zip", help="already-downloaded Eshghi et al. 2026 Zenodo archive")
    p.add_argument("--hoffmann-tar", help="already-downloaded Dragonfly_NA_WingSegmentation.tar.gz")
    p.add_argument("--no-download", action="store_true", help="never download; fail if a needed file is not cached")
    p.add_argument("--no-anim", action="store_true", help="skip the animation (no image downloads needed)")
    p.add_argument("--no-crosscheck", action="store_true", help="skip the Hoffmann cell-count cross-check inset")
    p.add_argument("--fps", type=float, default=0.8)
    args = p.parse_args(argv)

    data_dir, cache, out_dir = Path(args.data_dir), Path(args.cache_dir), Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = load_wings(data_dir)
    pbdb = load_pbdb(data_dir)
    print(f"{len(df)} wings: " + ", ".join(f"{GROUP_LABEL[g]} {n}" for g, n in df.group.value_counts().items()))
    report(df)

    tar_path = Path(args.hoffmann_tar) if args.hoffmann_tar else cache / "Dragonfly_NA_WingSegmentation.tar.gz"
    hoff_counts = None
    if not args.no_crosscheck or not args.no_anim:
        tar_path = fetch(HOFFMANN_TAR_URL, tar_path, allow_download=not args.no_download)
    if not args.no_crosscheck:
        shared = sorted(set(df[df.group != "fossil"].species))
        hoff_counts = {}
        for s in shared:
            n = hoffmann_domain_count(tar_path, s)
            if n is not None:
                hoff_counts[s] = n
        both = [s for s in hoff_counts]
        esh = np.array([df[(df.species == s) & (df.group != "fossil")].cell_count.mean() for s in both])
        hof = np.array([hoff_counts[s] for s in both])
        print(f"\nIndependent segmentations agree: {len(both)} shared species, median |ΔN|/N = "
              f"{np.median(np.abs(hof - esh) / esh):.1%}, r = {np.corrcoef(esh, hof)[0, 1]:.3f}")

    scaling_figure(df, out_dir / "odonata_wing_cells_scaling.png", hoff_counts)

    if not args.no_anim:
        zip_path = Path(args.zenodo_zip) if args.zenodo_zip else fetch(
            ZENODO_ZIP_URL, cache / "eshghi2026_supplementary.zip", ZENODO_ZIP_BYTES, allow_download=not args.no_download)
        deep_time_animation(df, pbdb, zip_path, tar_path, out_dir / "odonata_wing_cells_deep_time.gif", fps=args.fps)


if __name__ == "__main__":
    main()

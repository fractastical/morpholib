#!/usr/bin/env python3
"""C. elegans embryo from the WormGUIDES 4D nuclear atlas (Santella et al. 2015):
lineage, geometry over time and the OpenWorm connectome, side by side.

The nematode member of the "four views" family (planarian: Planform, limbs:
Limbform).  For C. elegans the four layers exist but in different projects:

  photograph / EM      WormAtlas (https://www.wormatlas.org; not redistributable, cited only)
  schematic            the Sulston lineage chart - here RE-DERIVED from the atlas names
  formal geometry      WormGUIDES nuclear-position atlas: every nucleus, every minute,
                       named by lineage (Santella et al. 2015; Bao lab, zhirongbaolab/WormGUIDES)
  formal connectivity  the hermaphrodite connectome as shipped by OpenWorm c302
                       (Cook et al. 2019 edge list) - connectivity, NOT geometry
  software model       OpenWorm / Sibernetic (Sarma et al. 2018); not run here

This script downloads the atlas files at a pinned commit, verifies them against a
SHA-256 manifest, decodes the nuclei files (StarryNite/AceTree format, columns as
read by WormGUIDES' AceTreeTableLineageDataLoader: valid flag, x, y, z, diameter,
lineage name), checks that the names form the Sulston tree, measures the embryo
over time, and draws (a) the lineage chart, (b) the nuclei coloured by founder
lineage at six times, (c) the atlas's cell-shape meshes at one time point and
(d) the connectome adjacency matrix.  A GIF animates the 360 frames.

References
----------
  Santella A, Catena R, Kovacevic I, Shah P, Yu Z, Marquina-Solis J, Kumar A, Wu Y,
  Schaff J, Colón-Ramos D, Shroff H, Mohler WA, Bao Z (2015). WormGUIDES: an
  interactive single cell developmental atlas and tool for collaborative multidimensional
  data exploration. BMC Bioinformatics 16:189. https://doi.org/10.1186/s12859-015-0627-8
  Data/app: https://github.com/zhirongbaolab/WormGUIDES (GPL-3.0), commit 17be611f;
  nuclear positions from image series DISPIM_061416, strain JIM113, 60 s per frame,
  "movie start time 20 min" (ProductionInfo.csv), i.e. frame t = t + 19 movie-minutes.
  Sarma GP et al. (2018). OpenWorm: overview and recent advances in integrative
  biological simulation of Caenorhabditis elegans. Phil Trans R Soc B 373:20170382.
  https://doi.org/10.1098/rstb.2017.0382  - connectome edge list from
  https://github.com/openworm/c302 (MIT), c302/data/herm_full_edgelist.csv, commit f54d92d7
  (Cook SJ et al. 2019, Nature 571:63, https://doi.org/10.1038/s41586-019-1352-7).
  Sulston JE, Schierenberg E, White JG, Thomson JN (1983). The embryonic cell lineage of
  the nematode Caenorhabditis elegans. Dev Biol 100:64-119.
  https://doi.org/10.1016/0012-1606(83)90201-4  (671 nuclei generated, 113 deaths, 558 at
  hatching; quoted via WormAtlas, Embryo introduction / Introduction §3.1)
  Gönczy P et al. (1999). J Cell Biol 144:927 (https://doi.org/10.1083/jcb.144.5.927):
  the one-cell embryo is a "50-µm-long cell" (citing Nigon 1960; Sulston 1983) - used to
  CALIBRATE the atlas's unit-less coordinates (ProductionInfo X/Y/Z_SCALE = 1) by setting the
  atlas's own eggshell outline mesh (obj_files/embryo_t330.obj, "Embryo Outline" in
  CellShapesConfig.csv) to that length; see calibrate() for why frame 330.
  WormAtlas, Embryo introduction (https://www.wormatlas.org/embryo/introduction/EIntroframeset.html):
  gastrulation starts at the ~30-cell stage; Ea and Ep enter the embryo at 100 min after
  first cleavage; ~550 cells by the end of proliferation (330-350 min); 671 / 113 / 558.

Usage
-----
    python3 2015-santella-wormguides-celegans.py --download        # fetch + verify (~430 small files)
    python3 2015-santella-wormguides-celegans.py --no-anim
    python3 2015-santella-wormguides-celegans.py --csv celegans_embryo_timeseries.csv

Outputs (git-ignored): celegans_four_views.png, celegans_lineage_chart.png,
celegans_embryo_timeline.gif, optional CSV of per-frame measurements.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
import os
import re
import sys
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib

if not os.environ.get("DISPLAY") and sys.platform != "darwin":
    matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.collections import EllipseCollection, LineCollection, PolyCollection  # noqa: E402
from PIL import Image  # noqa: E402

HERE = Path(__file__).resolve().parent
CACHE = HERE / ".celegans_cache"
MANIFEST = HERE / "datasets" / "celegans" / "pinned_inputs.sha256"

WG_REPO = "zhirongbaolab/WormGUIDES"
WG_COMMIT = "17be611f10cdc8ea438e67dc9ca3de859487a05e"          # retrieved 2026-10-15
WG_NUCLEI = "src/atlas_model/data/nuclei_files/t{t:03d}-nuclei"
WG_FILES = {
    "partslist": "src/application_src/application_model/data/CElegansData/PartsList/partslist.txt",
    "deaths": "src/application_src/application_model/data/CElegansData/CellDeaths/CellDeaths.csv",
    "production": "src/atlas_model/parameters/production_info/ProductionInfo.csv",
    "shapes": "src/atlas_model/configurations/shapes_config/CellShapesConfig.csv",
}
WG_OBJ = "src/atlas_model/data/obj_files/{cell}_t{t}.obj"
WG_SHELL = "src/atlas_model/data/obj_files/embryo_t{t}.obj"     # the atlas's eggshell outline ("Embryo Outline" in CellShapesConfig)
N_FRAMES = 360
MESH_FRAME = 330
PICK_FRAMES = (1, 30, 60, 100, 200, 360)
SHELL_FRAMES = (1, 30, 60, 100, 200, 250, 280, 300, 320, 330, 340, 350, 359)   # shell files run embryo_t0 .. embryo_t359
CALIB_FRAME = MESH_FRAME     # calibrate on the re-segmented shell of the frame that also carries the cell meshes

C302_REPO = "openworm/c302"
C302_COMMIT = "f54d92d7e9d2410ff6c47d9ae866649f1a419ee8"        # retrieved 2026-10-15
C302_EDGES = "c302/data/herm_full_edgelist.csv"

EMBRYO_LENGTH_UM = 50.0      # Gönczy et al. 1999 ("50-µm-long cell"); applied to the eggshell outline mesh at CALIB_FRAME
LIT = {                      # literature values used for the checks (see module docstring)
    "deaths": 113, "generated": 671, "hatching": 558,
    "gastrulation_min": 100, "gastrulation_cells": 30, "proliferation_end_min": (330, 350), "proliferation_end_cells": 550,
}

FOUNDERS = ["AB", "MS", "E", "C", "D", "P4"]
FOUNDER_COLOR = {"AB": "#4f86c6", "MS": "#e07b39", "E": "#3fa34d", "C": "#b86fc0", "D": "#d9b23f", "P4": "#c0392b",
                 "other": "#8a8a8a"}
FOUNDER_PARENT = {"AB": "P0", "P1": "P0", "EMS": "P1", "P2": "P1", "MS": "EMS", "E": "EMS", "C": "P2", "P3": "P2",
                  "D": "P3", "P4": "P3", "Z2": "P4", "Z3": "P4"}
NAME_RE = re.compile(r"^(AB|MS|E|C|D|EMS|P[0-4]|Z[23])[aplrdv]*$")


# --------------------------------------------------------------------------
# Inputs: pinned download + manifest verification
# --------------------------------------------------------------------------
def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def raw_url(repo: str, commit: str, path: str) -> str:
    return f"https://raw.githubusercontent.com/{repo}/{commit}/{path}"


def input_list(mesh_cells: Sequence[str] = ()) -> List[Tuple[str, str, str]]:
    """(cache-relative path, repo, repo path) for every pinned input."""
    items = [(f"wormguides/{p}", WG_REPO, p) for p in WG_FILES.values()]
    items += [(f"wormguides/{WG_NUCLEI.format(t=t)}", WG_REPO, WG_NUCLEI.format(t=t)) for t in range(1, N_FRAMES + 1)]
    items += [(f"wormguides/{WG_OBJ.format(cell=c, t=MESH_FRAME)}", WG_REPO, WG_OBJ.format(cell=c, t=MESH_FRAME)) for c in mesh_cells]
    items += [(f"wormguides/{WG_SHELL.format(t=t)}", WG_REPO, WG_SHELL.format(t=t)) for t in SHELL_FRAMES]
    items.append((f"c302/{C302_EDGES}", C302_REPO, C302_EDGES))
    return items


def load_manifest() -> Dict[str, str]:
    if not MANIFEST.exists():
        return {}
    out = {}
    for line in MANIFEST.read_text().splitlines():
        if line.strip() and not line.startswith("#"):
            h, p = line.split(None, 1)
            out[p.strip()] = h
    return out


def fetch_inputs(download: bool, write_manifest: bool) -> Dict[str, Path]:
    """Return {cache-relative path: local file}; download what is missing; verify every file
    against datasets/celegans/pinned_inputs.sha256 (or write that manifest with --pin)."""
    manifest = load_manifest()
    # mesh cells for MESH_FRAME come from the shapes config, which must be fetched first
    first = [it for it in input_list() if it[0].endswith("CellShapesConfig.csv")]
    _fetch(first, download)
    mesh_cells = mesh_cells_at(CACHE / first[0][0], MESH_FRAME)
    items = input_list(mesh_cells)
    if manifest:        # the manifest records which mesh files exist upstream at MESH_FRAME (the config ranges are sparse)
        items = [it for it in items if not it[0].endswith(".obj") or it[0] in manifest or "/embryo_t" in it[0]]
    _fetch(items, download)
    local = {rel: CACHE / rel for rel, _, _ in items if (CACHE / rel).exists() or not rel.endswith(".obj") or "/embryo_t" in rel}
    missing = [rel for rel, p in local.items() if not p.exists()]
    if missing:
        sys.exit(f"{len(missing)} input files missing (e.g. {missing[0]}); run with --download")
    bad, unpinned, lines = [], [], []
    for rel, p in local.items():
        h = sha256_of(p)
        lines.append(f"{h}  {rel}")
        if rel in manifest and manifest[rel] != h:
            bad.append(rel)
        elif rel not in manifest:
            unpinned.append(rel)
    if write_manifest or not manifest:
        MANIFEST.parent.mkdir(parents=True, exist_ok=True)
        MANIFEST.write_text(f"# SHA-256 of every input used by 2015-santella-wormguides-celegans.py\n"
                            f"# WormGUIDES {WG_REPO}@{WG_COMMIT}; c302 {C302_REPO}@{C302_COMMIT}; retrieved 2026-10-15\n"
                            + "\n".join(sorted(lines)) + "\n")
        print(f"  pinned {len(lines)} inputs -> {MANIFEST.relative_to(HERE)}")
    else:
        if bad:
            sys.exit(f"SHA-256 mismatch against {MANIFEST.name} for {len(bad)} file(s), e.g. {bad[0]}")
        print(f"  {len(local) - len(unpinned)} inputs verified against {MANIFEST.relative_to(HERE)}"
              + (f"; {len(unpinned)} not in the manifest (run --pin to add them)" if unpinned else ""))
    return local


def _fetch(items, download: bool) -> None:
    todo = [(rel, repo, path) for rel, repo, path in items if not (CACHE / rel).exists()]
    if not todo:
        return
    if not download:
        sys.exit(f"{len(todo)} input files are not cached (e.g. {todo[0][0]}); run with --download")
    print(f"  downloading {len(todo)} files into {CACHE.relative_to(HERE)}/ ...")

    def one(it):
        rel, repo, path = it
        commit = WG_COMMIT if repo == WG_REPO else C302_COMMIT
        dst = CACHE / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        req = urllib.request.Request(raw_url(repo, commit, path), headers={"User-Agent": "morpholib/celegans"})
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                data = resp.read()
        except urllib.error.HTTPError as e:
            if e.code == 404 and path.endswith(".obj") and "/embryo_t" not in path:   # cell-mesh ranges are sparse
                return 0
            raise
        dst.write_bytes(data)
        return len(data)
    with ThreadPoolExecutor(max_workers=8) as ex:
        sizes = list(ex.map(one, todo))
    print(f"  fetched {sum(sizes) / 1e6:.1f} MB ({sum(s == 0 for s in sizes)} mesh files absent at this frame)")


def mesh_cells_at(shapes_csv: Path, frame: int) -> List[str]:
    """Cells whose single-cell OBJ series (CellShapesConfig.csv) covers the frame."""
    cells = []
    with open(shapes_csv, newline="") as fh:
        for row in csv.DictReader(fh):
            if row.get("Type") == "Single Cell" and row.get("Resource Location", "").endswith(row["Cells"]):
                try:
                    if int(row["Start Time"]) <= frame <= int(row["End Time"]):
                        cells.append(row["Cells"].strip())
                except ValueError:
                    pass
    return sorted(set(cells))


# --------------------------------------------------------------------------
# Atlas decoding
# --------------------------------------------------------------------------
@dataclass
class Frame:
    t: int
    names: List[str]
    xyz: np.ndarray       # (n, 3) atlas units
    diam: np.ndarray      # (n,)


def read_nuclei(path: Path, t: int) -> Frame:
    """StarryNite/AceTree nuclei file; WormGUIDES reads columns 1 (valid), 5-7 (x, y, z), 8 (diameter),
    9 (lineage name) of each comma-separated line (AceTreeTableLineageDataLoader.java)."""
    names, xyz, diam = [], [], []
    for line in path.read_text().splitlines():
        tok = [s.strip() for s in line.split(",")]
        if len(tok) < 10 or tok[1] != "1" or not tok[9]:
            continue
        names.append(tok[9]); xyz.append((float(tok[5]), float(tok[6]), float(tok[7]))); diam.append(float(tok[8]))
    return Frame(t, names, np.array(xyz, float).reshape(-1, 3), np.array(diam, float))


def movie_offset(production_csv: Path) -> int:
    """WormGUIDES: movie time = frame + (start time - 1); ProductionInfo row 'all-nuclear positions',
    column 'Movie start timeProperty (min)'."""
    rows = list(csv.reader(open(production_csv, newline="")))
    header = rows[1]
    for r in rows[2:]:
        if r and r[0].strip() == "all-nuclear positions":
            return int(r[header.index("Movie start timeProperty (min)")]) - 1
    return 0


def founder(name: str) -> str:
    for f in ("AB", "MS", "EMS", "E", "C", "D", "P4", "Z2", "Z3"):
        if name.startswith(f) and (f != "E" or not name.startswith("EMS")):
            return {"EMS": "other", "Z2": "P4", "Z3": "P4"}.get(f, f)
    return "other"


def parent_of(name: str) -> Optional[str]:
    if name in FOUNDER_PARENT:
        return FOUNDER_PARENT[name]
    return name[:-1] if len(name) > 1 and name[-1] in "aplrdv" and NAME_RE.match(name[:-1]) else None


def check_encoding(frames: List[Frame]) -> Dict[str, float]:
    """Names parse as Sulston names; each non-founder cell's parent was present before it; the
    daughters of a division carry complementary suffixes."""
    birth: Dict[str, int] = {}
    for f in frames:
        for n in f.names:
            birth.setdefault(n, f.t)
    names = list(birth)
    parse_ok = sum(bool(NAME_RE.match(n)) for n in names)
    parent_ok = total = 0
    no_parent = []
    for n in names:
        p = parent_of(n)
        if p is None or n in ("P0",):
            continue
        if p not in birth:                      # parent divided before the movie starts (AB, P1 at a 4-cell start)
            no_parent.append(n)
            continue
        total += 1
        if birth[p] < birth[n]:
            parent_ok += 1
    print(f"  encoding check: {parse_ok}/{len(names)} lineage names parse as Sulston names; "
          f"{parent_ok}/{total} cells appear after their parent in the atlas; "
          f"{len(no_parent)} cells whose parent predates the movie: {', '.join(sorted(no_parent))}")
    return {"n_names": len(names), "parse_ok": parse_ok, "parent_ok": parent_ok, "parent_total": total,
            "parent_predates_movie": len(no_parent)}


# --------------------------------------------------------------------------
# Measurements
# --------------------------------------------------------------------------
def shell_outline(path: Path) -> np.ndarray:
    """Eggshell mesh -> (vertices, xyz extent)."""
    v, _ = read_obj(path)
    return v


def shell_xy_hull(v: np.ndarray) -> np.ndarray:
    """2-D outline of the eggshell: convex hull of the mesh vertices projected on x, y."""
    from scipy.spatial import ConvexHull
    pts = v[:, :2]
    h = ConvexHull(pts)
    return pts[h.vertices]


def calibrate(shell: np.ndarray) -> float:
    """µm per atlas unit so that the eggshell outline mesh at CALIB_FRAME is EMBRYO_LENGTH_UM long
    along x (the anterior-posterior axis of the image).  The 50 µm is a literature value
    (Gönczy et al. 1999); the length it is applied to is the atlas's own segmentation of the
    shell, so the only assumption left is that this egg has the typical length.
    Why frame 330 and not frame 1: the atlas reuses one static shell mesh (285.8 units long) for
    frames 1-~250, and the nuclei overrun it at the anterior pole by up to 19 units from frame 30
    on, so it is not the shell of this egg; from ~frame 280 the shell is re-segmented per frame
    (286.8-303.1 units) and encloses the nuclei.  The spread of the per-frame meshes is reported
    as the scale uncertainty."""
    return EMBRYO_LENGTH_UM / float(np.ptp(shell[:, 0]))


def shell_report(shells: Dict[int, np.ndarray], frames: List[Frame], um: float) -> Dict[str, float]:
    """Extents of the sampled shell meshes and of the nuclei relative to them (atlas units)."""
    lengths = {t: float(np.ptp(v[:, 0])) for t, v in shells.items()}
    by_t = {f.t: f for f in frames}
    overrun = {}
    for t, v in shells.items():
        f = by_t[t]
        lo, hi = (f.xyz[:, 0] - f.diam / 2).min(), (f.xyz[:, 0] + f.diam / 2).max()
        overrun[t] = float(max(v[:, 0].min() - lo, hi - v[:, 0].max(), 0.0))
    return {"length_min": min(lengths.values()), "length_max": max(lengths.values()),
            "length_calib": lengths[CALIB_FRAME], "overrun_max": max(overrun.values()),
            "overrun_calib": overrun[CALIB_FRAME], "um_min": EMBRYO_LENGTH_UM / max(lengths.values()),
            "um_max": EMBRYO_LENGTH_UM / min(lengths.values())}


def nuclear_cloud_scale(frames: List[Frame]) -> float:
    """The earlier estimate (longest nuclear cloud + one mean diameter = 50 µm), kept as a
    cross-check: nuclei do not reach the shell, so it overestimates µm per unit."""
    best = max(frames, key=lambda f: np.ptp(f.xyz[:, 0]) if len(f.xyz) else 0)
    return EMBRYO_LENGTH_UM / (np.ptp(best.xyz[:, 0]) + best.diam.mean())


def measure(frames: List[Frame], offset: int, um: float) -> List[Dict]:
    rows = []
    for f in frames:
        n = len(f.names)
        c = Counter(founder(x) for x in f.names)
        r = {"frame": f.t, "movie_min": f.t + offset, "n_cells": n}
        for k in FOUNDERS:
            r[f"n_{k}"] = c.get(k, 0)
        if n:
            ext = np.ptp(f.xyz, axis=0) + f.diam.mean()
            r.update(extent_x_um=round(ext[0] * um, 2), extent_y_um=round(ext[1] * um, 2), extent_z_um=round(ext[2] * um, 2),
                     mean_diam_um=round(f.diam.mean() * um, 3))
            centre = f.xyz.mean(axis=0)
            rad = np.linalg.norm((f.xyz - centre) / np.maximum(np.ptp(f.xyz, axis=0) / 2, 1e-6), axis=1)
            e_mask = np.array([founder(x) == "E" for x in f.names])
            r["E_radial_rel"] = round(float(rad[e_mask].mean() / np.median(rad)), 3) if e_mask.any() and np.median(rad) > 0 else ""
        rows.append(r)
    return rows


def gastrulation_onset(rows: List[Dict], thresh: float = 0.8, hold: int = 5) -> Optional[Dict]:
    """First frame from which the E cells sit at < `thresh` of the median normalised radius for
    `hold` consecutive frames (E ingression = start of gastrulation)."""
    vals = [(r["frame"], r["E_radial_rel"]) for r in rows if r["E_radial_rel"] != ""]
    for i in range(len(vals) - hold):
        if all(v < thresh for _, v in vals[i:i + hold]):
            fr = vals[i][0]
            return next(r for r in rows if r["frame"] == fr)
    return None


def literature_checks(rows: List[Dict], deaths: List[str], parts: Dict[str, str], frames: List[Frame]) -> List[str]:
    out = []
    g = gastrulation_onset(rows)
    if g:
        out.append(f"gastrulation onset (E ingression): frame {g['frame']}, {g['n_cells']} cells, movie-minute {g['movie_min']}; "
                   f"WormAtlas: ~{LIT['gastrulation_cells']}-cell stage, Ea/Ep enter at {LIT['gastrulation_min']} min after first cleavage")
    lo, hi = LIT["proliferation_end_min"]
    at = [r for r in rows if lo <= r["movie_min"] <= hi]
    if at:
        out.append(f"cells at movie-minute {lo}-{hi}: {at[0]['n_cells']}-{at[-1]['n_cells']}; WormAtlas: ~{LIT['proliferation_end_cells']} "
                   f"at the end of proliferation ({lo}-{hi} min)")
    out.append(f"cell-death list in the atlas: {len(deaths)} names; Sulston et al. 1983 (via WormAtlas): {LIT['deaths']} embryonic deaths")
    seen = {n for f in frames for n in f.names}
    out.append(f"deaths observed in the atlas before it ends: {sum(d in seen for d in deaths)}/{len(deaths)}")
    last = frames[-1]
    named = sum(n in parts for n in last.names)
    out.append(f"last frame ({last.t}, movie-minute {rows[-1]['movie_min']}): {len(last.names)} nuclei, {named} with a parts-list identity; "
               f"Sulston: {LIT['generated']} generated, {LIT['hatching']} at hatching")
    return out


# --------------------------------------------------------------------------
# Lineage chart (the Sulston diagram, re-derived from the atlas)
# --------------------------------------------------------------------------
def lineage_intervals(frames: List[Frame]) -> Dict[str, Tuple[int, int]]:
    birth, last = {}, {}
    for f in frames:
        for n in f.names:
            birth.setdefault(n, f.t); last[n] = f.t
    return {n: (birth[n], last[n]) for n in birth}


def lineage_layout(iv: Dict[str, Tuple[int, int]]) -> Dict[str, float]:
    """x position per cell: leaves in depth-first order of the tree, parents centred on children."""
    children: Dict[str, List[str]] = defaultdict(list)
    roots = []
    for n in iv:
        p = parent_of(n)
        if p in iv:
            children[p].append(n)
        else:
            roots.append(n)
    order = {"a": 0, "p": 1, "l": 0, "r": 1, "d": 0, "v": 1}
    x: Dict[str, float] = {}
    counter = [0]

    def place(n):
        kids = sorted(children.get(n, []), key=lambda k: (order.get(k[-1], 0), k))
        if not kids:
            x[n] = counter[0]; counter[0] += 1
        else:
            for k in kids:
                place(k)
            x[n] = sum(x[k] for k in kids) / len(kids)
    for r in sorted(roots, key=lambda r: FOUNDERS.index(founder(r)) if founder(r) in FOUNDERS else 99):
        place(r)
    return x


def draw_lineage(ax, frames: List[Frame], offset: int, lw: float = 0.5) -> None:
    iv = lineage_intervals(frames)
    x = lineage_layout(iv)
    segs, cols = [], []
    for n, (b, e) in iv.items():
        col = FOUNDER_COLOR.get(founder(n), FOUNDER_COLOR["other"])
        segs.append([(x[n], b + offset), (x[n], e + offset + 1)]); cols.append(col)
        p = parent_of(n)
        if p in iv:
            segs.append([(x[p], b + offset), (x[n], b + offset)]); cols.append(col)
    ax.add_collection(LineCollection(segs, colors=cols, linewidths=lw))
    ax.set_xlim(-2, max(x.values()) + 2); ax.set_ylim(N_FRAMES + offset + 2, offset - 2)
    ax.set_ylabel("movie time (min)"); ax.set_xticks([])
    for f in FOUNDERS:
        xs = [x[n] for n in iv if founder(n) == f]
        if xs:
            ax.text(np.mean(xs), offset - 4, f, ha="center", va="bottom", fontsize=8, color=FOUNDER_COLOR[f], fontweight="bold")


# --------------------------------------------------------------------------
# Drawing
# --------------------------------------------------------------------------
def draw_shell(ax, shell: Optional[np.ndarray], um: float) -> None:
    if shell is None:
        return
    hull = shell_xy_hull(shell) * um
    ax.add_patch(plt.Polygon(hull, closed=True, fill=False, edgecolor="#555", lw=0.8, ls="-", zorder=0))


def draw_embryo(ax, f: Frame, um: float, offset: int, title: bool = True, s_scale: float = 1.0,
                shell: Optional[np.ndarray] = None) -> None:
    """Dorsal-ish projection (x = anterior-posterior axis of the image, y = image y); nuclei as discs
    with their recorded diameter, coloured by founder lineage, back-to-front by z; the atlas's
    eggshell outline (convex hull of its mesh in x, y) when given."""
    if not len(f.xyz):
        ax.axis("off"); return
    draw_shell(ax, shell, um)
    order = np.argsort(f.xyz[:, 2])
    cols = [FOUNDER_COLOR.get(founder(f.names[i]), FOUNDER_COLOR["other"]) for i in order]
    xy = f.xyz[order][:, :2] * um
    d = f.diam[order] * um * s_scale
    # discs in data units (µm), so the recorded nuclear diameters are drawn to scale
    ax.add_collection(EllipseCollection(d, d, np.zeros_like(d), units="xy", offsets=xy, offset_transform=ax.transData,
                                        facecolors=cols, edgecolors="black", linewidths=0.3, alpha=0.9))
    ax.set_aspect("equal"); ax.set_xlim(5, 65); ax.set_ylim(40, 0)
    ax.set_xticks([10, 35, 60]); ax.set_yticks([0, 30]); ax.tick_params(labelsize=7)
    if title:
        ax.set_title(f"frame {f.t} · {f.t + offset} min · {len(f.names)} cells", fontsize=9)


def read_obj(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    v, faces = [], []
    for line in path.read_text().splitlines():
        if line.startswith("v "):
            v.append([float(s) for s in line.split()[1:4]])
        elif line.startswith("f "):
            faces.append([int(s.split("/")[0]) - 1 for s in line.split()[1:]])
    return np.array(v), np.array([f[:3] for f in faces if len(f) >= 3])


def draw_meshes(ax, frame: Frame, local: Dict[str, Path], cells: Sequence[str], um: float, shells: Dict[int, np.ndarray]) -> int:
    """The atlas's segmented cell shapes (OBJ, same coordinate frame as the nuclei) over the nuclei."""
    draw_shell(ax, shells.get(MESH_FRAME), um)
    ax.scatter(frame.xyz[:, 0] * um, frame.xyz[:, 1] * um, s=4, c="#9a9a9a", linewidths=0, zorder=1)
    ax.set_aspect("equal"); ax.set_xlim(5, 65); ax.set_ylim(40, 0)
    ax.set_xticks([10, 35, 60]); ax.set_yticks([0, 30]); ax.tick_params(labelsize=7)
    n = 0
    for c in cells:
        p = local.get(f"wormguides/{WG_OBJ.format(cell=c, t=MESH_FRAME)}")
        if not p or not p.exists():
            continue
        v, faces = read_obj(p)
        if not len(v) or not len(faces):
            continue
        tri = v[faces][:, :, :2] * um
        depth = v[faces][:, :, 2].mean(axis=1)
        order = np.argsort(depth)
        col = FOUNDER_COLOR.get(founder(c), FOUNDER_COLOR["other"])
        ax.add_collection(PolyCollection(tri[order], facecolors=col, edgecolors=col, linewidths=0.15, alpha=0.55, zorder=3))
        n += 1
    return n


def read_edges(path: Path) -> List[Tuple[str, str, str, float]]:
    rows = []
    with open(path, newline="") as fh:
        rd = csv.reader(fh)
        header = next(rd)
        cols = {h.strip().lower(): i for i, h in enumerate(header)}
        src = cols.get("source", cols.get("pre", 0)); tgt = cols.get("target", cols.get("post", 1))
        typ = cols.get("type", 2); wt = cols.get("weight", cols.get("synapses", 3))
        for r in rd:
            if len(r) <= max(src, tgt, typ, wt):
                continue
            try:
                rows.append((r[src].strip(), r[tgt].strip(), r[typ].strip(), float(r[wt] or 0)))
            except ValueError:
                pass
    return rows


NEURON_RE = re.compile(r"^[A-Z][A-Z0-9]*$")   # Cook et al. name neurons in capitals; muscles / end organs are lowercase-prefixed


def canonical_cell(name: str) -> str:
    """`DA01` (edge list) -> `DA1` (parts list / WormAtlas)."""
    return re.sub(r"^([A-Z]+)0(\d)$", r"\1\2", name.strip())


def draw_connectome(ax, edges, parts: Dict[str, str]) -> Dict[str, int]:
    """Neuron-to-neuron adjacency matrix (chemical + electrical), neurons = uppercase node names of
    the Cook et al. edge list (muscles and end organs are lowercase-prefixed), ordered by name; log
    colour scale.  Connectivity layer, not geometry."""
    edges = [(canonical_cell(a), canonical_cell(b), t, w) for a, b, t, w in edges]
    nodes = sorted({a for a, b, _, _ in edges} | {b for a, b, _, _ in edges})
    neurons = [n for n in nodes if NEURON_RE.match(n)]
    idx = {n: i for i, n in enumerate(neurons)}
    M = np.zeros((len(neurons), len(neurons)))
    nchem = nelec = 0
    for a, b, t, w in edges:
        if a in idx and b in idx:
            M[idx[a], idx[b]] += w
            if "electrical" in t.lower() or "gap" in t.lower():
                nelec += 1
            else:
                nchem += 1
    shown = np.ma.masked_where(M <= 0, np.log1p(M))
    cmap = plt.get_cmap("viridis").copy(); cmap.set_bad("white")
    ax.imshow(shown, cmap=cmap, interpolation="nearest", aspect="equal")
    ax.set_xlabel("postsynaptic neuron (alphabetical)", fontsize=8); ax.set_ylabel("presynaptic neuron", fontsize=8)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_linewidth(0.5)
    return {"neurons": len(neurons), "neurons_in_partslist": sum(n in parts for n in neurons), "nodes": len(nodes),
            "chemical_edges": nchem, "electrical_edges": nelec, "nonzero": int((M > 0).sum())}


def shell_for(shells: Dict[int, np.ndarray], t: int) -> Optional[np.ndarray]:
    """Shell at the nearest available frame (shell files end at embryo_t359)."""
    if not shells:
        return None
    return shells[min(shells, key=lambda k: abs(k - t))]


def four_views(frames, local, um, offset, parts, mesh_cells, shells, out: Path, pick=PICK_FRAMES) -> Dict:
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 6, height_ratios=[1.0, 0.5, 1.25], hspace=0.55, wspace=0.3, top=0.92, bottom=0.05, left=0.05, right=0.98)
    ax = fig.add_subplot(gs[0, :])
    draw_lineage(ax, frames, offset)
    ax.set_title("A  Lineage chart re-derived from the atlas names: every nucleus of 360 one-minute frames, "
                 "x = depth-first order of the Sulston tree (the schematic layer)", fontsize=10, loc="left", pad=14)
    by_t = {f.t: f for f in frames}
    for i, t in enumerate(pick):
        a = fig.add_subplot(gs[1, i]); draw_embryo(a, by_t[t], um, offset, shell=shell_for(shells, t))
        if i == 0:
            a.set_ylabel("µm (eggshell = 50 µm)", fontsize=7)
    row_b = fig.add_subplot(gs[1, :], frameon=False); row_b.set_xticks([]); row_b.set_yticks([])
    row_b.set_title("B  Formal geometry: nuclear positions and recorded diameters (to scale) inside the atlas's eggshell outline, "
                    "coloured by founder lineage " + "  ".join(f"■{k}" for k in FOUNDERS) + "  ■other  — WormGUIDES DISPIM_061416 / JIM113",
                    fontsize=10, loc="left", pad=28)
    legend_handles = [plt.Line2D([], [], marker="o", ls="", color=FOUNDER_COLOR[k], label=k) for k in FOUNDERS]
    legend_handles.append(plt.Line2D([], [], marker="o", ls="", color=FOUNDER_COLOR["other"], label="other"))
    a = fig.add_subplot(gs[2, :3])
    n_mesh = draw_meshes(a, by_t[MESH_FRAME], local, mesh_cells, um, shells)
    a.set_title(f"C  Cell shapes: {n_mesh} segmented single-cell meshes of the atlas at frame {MESH_FRAME} "
                f"({MESH_FRAME + offset} min),\nsame coordinate frame; grey dots = all {len(by_t[MESH_FRAME].names)} nuclei of that frame",
                fontsize=10, loc="left")
    a.set_xlabel("µm", fontsize=8); a.set_ylabel("µm", fontsize=8)
    a.legend(handles=legend_handles, loc="upper right", fontsize=7, ncol=3, frameon=False, title="founder", title_fontsize=7)
    a = fig.add_subplot(gs[2, 3:])
    stats = draw_connectome(a, read_edges(local[f"c302/{C302_EDGES}"]), parts)
    a.set_title(f"D  Formal connectivity (not geometry): c302 edge list, Cook et al. 2019\n"
                f"{stats['neurons']} neurons, {stats['nonzero']} connected ordered pairs; colour = log(1 + weight)",
                fontsize=10, loc="left")
    fig.suptitle("C. elegans embryo: the WormGUIDES 4D nuclear atlas decoded (Santella et al. 2015) with the OpenWorm connectome\n"
                 "photograph/EM layer: WormAtlas (cited, not reproduced)", fontsize=12)
    fig.savefig(out, dpi=140); plt.close(fig)
    print(f"  saved {out}")
    return stats


def lineage_figure(frames, offset, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(22, 9))
    draw_lineage(ax, frames, offset, lw=0.6)
    ax.set_title("C. elegans embryonic lineage as recorded in the WormGUIDES atlas (1 min frames; "
                 "colours = founder lineage; a branch ends where the atlas ends or the cell dies/is lost)", fontsize=11)
    fig.tight_layout(); fig.savefig(out, dpi=130); plt.close(fig)
    print(f"  saved {out}")


def timeline_gif(frames, rows, um, offset, shells, out: Path, step: int = 3, fps: int = 10) -> None:
    imgs = []
    counts = np.array([r["n_cells"] for r in rows]); tmin = np.array([r["movie_min"] for r in rows])
    for f in frames[::step]:
        fig = plt.figure(figsize=(10, 5.2))
        gs = fig.add_gridspec(1, 2, width_ratios=[1.35, 1])
        a = fig.add_subplot(gs[0]); draw_embryo(a, f, um, offset, shell=shell_for(shells, f.t))
        a.set_xlabel("µm (eggshell outline = 50 µm long)"); a.set_ylabel("µm")
        b = fig.add_subplot(gs[1])
        b.plot(tmin, counts, color="#444"); b.axvline(f.t + offset, color="#c0392b", lw=1)
        stack = np.array([[r[f"n_{k}"] for k in FOUNDERS] for r in rows]).T
        b.stackplot(tmin, stack, colors=[FOUNDER_COLOR[k] for k in FOUNDERS], alpha=0.6, labels=FOUNDERS)
        b.set_xlabel("movie time (min)"); b.set_ylabel("nuclei"); b.legend(loc="upper left", fontsize=7, ncol=2)
        b.set_title("cell count by founder lineage", fontsize=9)
        fig.suptitle("WormGUIDES atlas: C. elegans embryo, nuclei by lineage", fontsize=11)
        fig.tight_layout()
        fig.canvas.draw()
        imgs.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()))
        plt.close(fig)
    imgs[0].save(out, save_all=True, append_images=imgs[1:], duration=int(1000 / fps), loop=0)
    print(f"  saved {out}  ({len(imgs)} frames)")


# --------------------------------------------------------------------------
def main(argv: Optional[Sequence[str]] = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--download", action="store_true", help="fetch the pinned inputs into .celegans_cache/")
    p.add_argument("--pin", action="store_true", help="(re)write datasets/celegans/pinned_inputs.sha256 from the cache")
    p.add_argument("--out-dir", default=".")
    p.add_argument("--csv", help="write per-frame measurements to this CSV")
    p.add_argument("--no-anim", action="store_true")
    p.add_argument("--step", type=int, default=3, help="frame step of the GIF")
    args = p.parse_args(argv)

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    local = fetch_inputs(args.download, args.pin)
    offset = movie_offset(local[f"wormguides/{WG_FILES['production']}"])
    frames = [read_nuclei(local[f"wormguides/{WG_NUCLEI.format(t=t)}"], t) for t in range(1, N_FRAMES + 1)]
    parts = {}
    for line in local[f"wormguides/{WG_FILES['partslist']}"].read_text().splitlines():
        tok = line.split("\t")
        if len(tok) >= 2:
            parts[tok[1].strip()] = tok[2].strip() if len(tok) > 2 else ""
            parts[tok[0].strip()] = tok[2].strip() if len(tok) > 2 else ""
    deaths = [l.strip() for l in local[f"wormguides/{WG_FILES['deaths']}"].read_text().splitlines() if l.strip()]
    mesh_cells = mesh_cells_at(local[f"wormguides/{WG_FILES['shapes']}"], MESH_FRAME)
    print(f"WormGUIDES atlas @ {WG_COMMIT[:8]}: {len(frames)} frames, movie offset +{offset} min, "
          f"{len(frames[0].names)} -> {len(frames[-1].names)} nuclei, {len(mesh_cells)} cell meshes at frame {MESH_FRAME}")
    enc = check_encoding(frames)
    shells = {t: shell_outline(local[f"wormguides/{WG_SHELL.format(t=t)}"]) for t in SHELL_FRAMES}
    um = calibrate(shells[CALIB_FRAME])
    ext = np.ptp(shells[CALIB_FRAME], axis=0) * um
    sr = shell_report(shells, frames, um)
    print(f"  calibration: {um:.4f} µm per atlas unit (eggshell outline mesh at frame {CALIB_FRAME} = {EMBRYO_LENGTH_UM:.0f} µm long); "
          f"shell {ext[0]:.1f} x {ext[1]:.1f} x {ext[2]:.1f} µm; sampled shell meshes span {sr['length_min']:.1f}-{sr['length_max']:.1f} units "
          f"({sr['um_min']:.4f}-{sr['um_max']:.4f} µm/unit); nuclei overrun the shell in x by up to {sr['overrun_max']:.1f} units "
          f"(static early mesh), {sr['overrun_calib']:.1f} at frame {CALIB_FRAME}; nuclear-cloud estimate: {nuclear_cloud_scale(frames):.4f}")
    rows = measure(frames, offset, um)
    for line in literature_checks(rows, deaths, parts, frames):
        print("  check:", line)

    stats = four_views(frames, local, um, offset, parts, mesh_cells, shells, out_dir / "celegans_four_views.png")
    lineage_figure(frames, offset, out_dir / "celegans_lineage_chart.png")
    if args.csv:
        with open(args.csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[-1].keys())); w.writeheader(); w.writerows(rows)
        print(f"  saved {args.csv}  ({len(rows)} rows)")
    if not args.no_anim:
        timeline_gif(frames, rows, um, offset, shells, out_dir / "celegans_embryo_timeline.gif", step=args.step)
    print(f"  connectome: {stats}")


if __name__ == "__main__":
    main()

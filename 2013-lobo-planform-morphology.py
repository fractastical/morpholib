#!/usr/bin/env python3
"""Planform morphology formalism: decoder, renderer and morphospace timeline.

A fresh take on the PlanformDB morphology animation. The earlier script
(``1900-planformDB_parser.py``) approximated every morphology as the convex
hull of its organ landmarks and discarded the region graph, so all phenotypes
collapsed onto the same blob. This script decodes the graph formalism exactly
as specified in

  Lobo D, Malone TJ, Levin M (2013). Towards a bioinformatics of patterning:
  a computational approach to understanding regulative morphogenesis.
  Biology Open 2(2):156-169. https://doi.org/10.1242/bio.20123400

  Lobo D, Malone TJ, Levin M (2013). Planform: an application and database
  of graph-encoded planarian regenerative experiments.
  Bioinformatics 29(8):1098-1100. https://doi.org/10.1093/bioinformatics/btt088

Encoding (Biology Open paper, "Formalism for phenotype morphologies"):

* Regions (head / trunk / tail) are graph vertices. Their shape is a list of
  radii: the distance from the region centre to its border in a fixed set of
  directions.
    - 0 links : 4 radii -> right, anterior, left, posterior
    - 1 link  : 3 radii -> +90, +180, +270 degrees from the edge direction
    - k>=2    : k radii -> one per bisector of consecutive edges (edges
                sorted by angle)
  This rule holds for all 1181 regions in planformDB 2.5.0 (verified).
* Links (RegionsLink) carry Dist and Ang (centre -> centre vector of
  ToRegion relative to FromRegion) and Ratio: the border between the two
  regions sits at Ratio * Dist from the FromRegion centre.
* Organs are vectors from the region centre; spot organs (eye, brain,
  pharynx) carry a rotation, line organs (ventral nerve cord) carry two end
  points relative to the organ position.

Coordinate convention in the database: +y is anterior (the WT head->trunk
link has Ang = 3*pi/2, i.e. the trunk lies posterior of the head).

Usage
-----
    export PLANFORM_DB_PATH=/path/to/planformDB_2.5.0.edb
    python3 2013-lobo-planform-morphology.py                 # gallery + timeline GIF
    python3 2013-lobo-planform-morphology.py --ids 1 2 3     # render specific morphologies
    python3 2013-lobo-planform-morphology.py --csv morphospace.csv --no-anim

Outputs (git-ignored): planform_morphology_gallery.png,
planform_morphospace_timeline.gif, and optionally a CSV of per-morphology
morphometrics derived from the rendered silhouettes.
"""

from __future__ import annotations

import argparse
import math
import os
import sqlite3
import sys
import textwrap
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib

if not os.environ.get("DISPLAY") and sys.platform != "darwin":
    matplotlib.use("Agg")

import matplotlib.animation as animation  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Circle, Ellipse, Polygon  # noqa: E402
from matplotlib.path import Path as MplPath  # noqa: E402
from scipy.interpolate import PchipInterpolator  # noqa: E402
from scipy.ndimage import gaussian_filter  # noqa: E402

TWO_PI = 2.0 * math.pi

REGION_COLORS = {  # Planform cartoon convention: head red, trunk grey, tail blue
    "Head": "#d6453d",
    "Trunk": "#a8a8a8",
    "Tail": "#3f6fb5",
}
WT_LENGTH_UNITS = 560.0  # anterior tip to posterior tip of morphology 1 (Wild type)


# --------------------------------------------------------------------------
# Data model
# --------------------------------------------------------------------------
@dataclass
class Region:
    id: int
    type: str
    params: List[float]


@dataclass
class Link:
    from_region: int
    to_region: int
    dist: float
    ratio: float
    ang: float


@dataclass
class Organ:
    region: int
    vx: float
    vy: float
    kind: str  # Eye | Brain | Phar. | VNC
    rot: Optional[float] = None
    ends: Optional[Tuple[float, float, float, float]] = None


@dataclass
class Morphology:
    id: int
    name: str
    regions: Dict[int, Region]
    links: List[Link]
    organs: List[Organ]
    positions: Dict[int, Tuple[float, float]] = field(default_factory=dict)
    outlines: Dict[int, np.ndarray] = field(default_factory=dict)

    def count(self, kind: str) -> int:
        if kind in REGION_COLORS:
            return sum(1 for r in self.regions.values() if r.type == kind)
        return sum(1 for o in self.organs if o.kind == kind)


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------
def resolve_db_path(cli_path: Optional[str]) -> str:
    path = cli_path or os.getenv("PLANFORM_DB_PATH")
    if not path or not Path(path).is_file():
        sys.exit(
            "PlanformDB not found. Pass the .edb path as the first argument or set "
            "PLANFORM_DB_PATH (planformDB_2.5.0.edb is distributed with Planform)."
        )
    return path


def load_morphologies(db_path: str) -> Dict[int, Morphology]:
    conn = sqlite3.connect(db_path)
    try:
        cur = conn.cursor()
        morphs: Dict[int, Morphology] = {}
        for mid, name in cur.execute("SELECT Id, Name FROM Morphology ORDER BY Id"):
            morphs[mid] = Morphology(mid, name.strip(), {}, [], [])

        region_owner: Dict[int, int] = {}
        for rid, mid, rtype in cur.execute(
            "SELECT r.Id, r.Morphology, t.Name FROM Region r JOIN RegionType t ON t.Id = r.Type"
        ):
            morphs[mid].regions[rid] = Region(rid, rtype, [])
            region_owner[rid] = mid

        for rid, ind, val in cur.execute(
            "SELECT Region, ParamInd, Value FROM RegionParam ORDER BY Region, ParamInd"
        ):
            reg = morphs[region_owner[rid]].regions[rid]
            while len(reg.params) <= ind:
                reg.params.append(float("nan"))
            reg.params[ind] = float(val)

        for a, b, dist, ratio, ang in cur.execute(
            "SELECT FromRegion, ToRegion, Dist, Ratio, Ang FROM RegionsLink"
        ):
            morphs[region_owner[a]].links.append(Link(a, b, float(dist), float(ratio), float(ang)))

        for row in cur.execute(
            """
            SELECT o.Region, o.VecX, o.VecY, st.Name, s.Rot,
                   lt.Name, lo.VecEnd1X, lo.VecEnd1Y, lo.VecEnd2X, lo.VecEnd2Y
            FROM Organ o
            LEFT JOIN SpotOrgan s ON s.Id = o.SpotOrgan
            LEFT JOIN SpotOrganType st ON st.Id = s.Type
            LEFT JOIN LineOrgan lo ON lo.Id = o.LineOrgan
            LEFT JOIN LineOrganType lt ON lt.Id = lo.Type
            """
        ):
            rid, vx, vy, spot, rot, line, e1x, e1y, e2x, e2y = row
            if spot is not None:
                organ = Organ(rid, vx, vy, spot, rot=float(rot))
            else:
                organ = Organ(rid, vx, vy, line, ends=(e1x, e1y, e2x, e2y))
            morphs[region_owner[rid]].organs.append(organ)
    finally:
        conn.close()

    # Three entries have no geometry at all ("Lethal", "Midbody Buldge",
    # "Double head, side tail" — ids 38, 333, 359 in 2.5.0); they are outcomes
    # without an encoded shape and cannot be drawn.
    empty = [mid for mid, m in morphs.items() if not m.regions]
    for mid in empty:
        print(f"  skipping morphology {mid} '{morphs[mid].name}': no regions encoded")
        del morphs[mid]

    for m in morphs.values():
        layout(m)
        for rid in m.regions:
            m.outlines[rid] = region_outline(m, rid)
    return morphs


def load_first_report_year(db_path: str) -> Dict[int, Tuple[int, str, List[str]]]:
    """Morphology id -> (first year reported, species, publication titles)."""
    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute(
            """
            SELECT rm.Morphology, p.Year, sp.Name, p.Title
            FROM ResultantMorphology rm
            JOIN ResultSet rs ON rs.Id = rm.ResultSet
            JOIN Experiment e ON e.Id = rs.Experiment
            JOIN Publication p ON p.Id = e.Publication
            JOIN Species sp ON sp.Id = e.Species
            WHERE p.Year IS NOT NULL
            """
        ).fetchall()
        pubs_per_year = dict(
            conn.execute(
                "SELECT Year, COUNT(*) FROM Publication WHERE Year IS NOT NULL GROUP BY Year"
            ).fetchall()
        )
    finally:
        conn.close()

    first: Dict[int, Tuple[int, str, List[str]]] = {}
    for mid, year, species, title in rows:
        cur = first.get(mid)
        if cur is None or year < cur[0]:
            first[mid] = (year, species, [title])
        elif year == cur[0] and title not in cur[2]:
            cur[2].append(title)
    first["_pubs_per_year"] = pubs_per_year  # type: ignore[index]
    return first


# --------------------------------------------------------------------------
# Geometry: layout and region outlines
# --------------------------------------------------------------------------
def layout(m: Morphology) -> None:
    """Place region centres by walking the link graph (BFS from the hub)."""
    adjacency: Dict[int, List[Tuple[int, float, float]]] = defaultdict(list)
    for lk in m.links:
        adjacency[lk.from_region].append((lk.to_region, lk.dist, lk.ang))
        adjacency[lk.to_region].append((lk.from_region, lk.dist, lk.ang + math.pi))

    pos: Dict[int, Tuple[float, float]] = {}
    offset_x = 0.0
    for root in sorted(m.regions, key=lambda r: (-len(adjacency[r]), r)):
        if root in pos:
            continue
        pos[root] = (offset_x, 0.0)
        queue = deque([root])
        while queue:
            a = queue.popleft()
            ax, ay = pos[a]
            for b, dist, ang in adjacency[a]:
                if b in pos:
                    continue
                pos[b] = (ax + dist * math.cos(ang), ay + dist * math.sin(ang))
                queue.append(b)
        # disconnected components (6 morphologies) are laid out side by side
        offset_x += 400.0
    m.positions = pos


def _edges_of(m: Morphology, rid: int) -> List[Tuple[float, float]]:
    """(direction from region to neighbour, radius to the shared border)."""
    edges = []
    for lk in m.links:
        if lk.from_region == rid:
            edges.append((lk.ang % TWO_PI, lk.ratio * lk.dist))
        elif lk.to_region == rid:
            edges.append(((lk.ang + math.pi) % TWO_PI, (1.0 - lk.ratio) * lk.dist))
    return edges


def control_points(m: Morphology, rid: int) -> List[Tuple[float, float]]:
    """(angle, radius) controls for a region per the Lobo et al. 2013 rule."""
    reg = m.regions[rid]
    edges = sorted(_edges_of(m, rid))
    params = [p for p in reg.params if not math.isnan(p)]
    pts: List[Tuple[float, float]] = list(edges)

    if not edges:
        dirs = [0.0, math.pi / 2, math.pi, 3 * math.pi / 2]  # right, anterior, left, posterior
    elif len(edges) == 1:
        d = edges[0][0]
        dirs = [d + math.pi / 2, d + math.pi, d + 3 * math.pi / 2]
    else:
        angs = [e[0] for e in edges]
        dirs = []
        for i in range(len(angs)):
            a0 = angs[i]
            a1 = angs[(i + 1) % len(angs)]
            if i == len(angs) - 1:
                a1 += TWO_PI
            dirs.append((a0 + a1) / 2.0)

    if len(params) < len(dirs):  # never happens in planformDB 2.5.0; keep shape sane
        fill = float(np.nanmean(params)) if params else 60.0
        params = params + [fill] * (len(dirs) - len(params))
    for d, r in zip(dirs, params):
        pts.append((d % TWO_PI, max(r, 1.0)))
    pts.sort()
    return pts


def region_outline(m: Morphology, rid: int, n: int = 240) -> np.ndarray:
    """Smooth closed outline through the polar control points (shape-preserving)."""
    pts = control_points(m, rid)
    angs = np.array([p[0] for p in pts])
    rads = np.array([p[1] for p in pts])
    # tile three periods so the periodic interpolant has no seam
    a3 = np.concatenate([angs - TWO_PI, angs, angs + TWO_PI])
    r3 = np.concatenate([rads, rads, rads])
    interp = PchipInterpolator(a3, r3)
    theta = np.linspace(0, TWO_PI, n, endpoint=False)
    r = np.clip(interp(theta), 1.0, None)
    cx, cy = m.positions[rid]
    return np.column_stack([cx + r * np.cos(theta), cy + r * np.sin(theta)])


# --------------------------------------------------------------------------
# Silhouette (union of regions) and morphometrics
# --------------------------------------------------------------------------
def extent(m: Morphology, pad: float = 20.0) -> Tuple[float, float, float, float]:
    pts = np.vstack(list(m.outlines.values()))
    return (pts[:, 0].min() - pad, pts[:, 0].max() + pad, pts[:, 1].min() - pad, pts[:, 1].max() + pad)


def silhouette(m: Morphology, res: int = 256, smooth: float = 0.0
               ) -> Tuple[np.ndarray, Tuple[float, float, float, float]]:
    """Boolean union of the region outlines on a ``res``x``res`` grid.

    ``smooth`` (in model units) applies a Gaussian blur and re-thresholds, which
    rounds the small concave notches where neighbouring region blobs meet
    without changing the overall extent appreciably. 0 disables smoothing.
    """
    x0, x1, y0, y1 = extent(m)
    xs = np.linspace(x0, x1, res)
    ys = np.linspace(y0, y1, res)
    gx, gy = np.meshgrid(xs, ys)
    grid = np.column_stack([gx.ravel(), gy.ravel()])
    mask = np.zeros(grid.shape[0], dtype=bool)
    for poly in m.outlines.values():
        mask |= MplPath(poly).contains_points(grid)
    mask = mask.reshape(res, res)
    if smooth > 0:
        sigma = smooth / ((x1 - x0) / (res - 1))
        mask = gaussian_filter(mask.astype(float), sigma) >= 0.5
    return mask, (x0, x1, y0, y1)


def morphometrics(m: Morphology) -> Dict[str, float]:
    mask, (x0, x1, y0, y1) = silhouette(m)
    res = mask.shape[0]
    dx = (x1 - x0) / (res - 1)
    dy = (y1 - y0) / (res - 1)
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    length = (rows.max() - rows.min()) * dy if rows.size else 0.0
    width = (cols.max() - cols.min()) * dx if cols.size else 0.0
    return {
        "id": m.id,
        "name": m.name,
        "n_regions": len(m.regions),
        "n_heads": m.count("Head"),
        "n_trunks": m.count("Trunk"),
        "n_tails": m.count("Tail"),
        "n_eyes": m.count("Eye"),
        "n_brains": m.count("Brain"),
        "n_pharynges": m.count("Phar."),
        "n_vnc": m.count("VNC"),
        "ap_length": round(length, 1),
        "ml_width": round(width, 1),
        "aspect": round(length / width, 3) if width else float("nan"),
        "area": round(float(mask.sum()) * dx * dy, 0),
        "length_rel_wt": round(length / WT_LENGTH_UNITS, 3),
    }


# --------------------------------------------------------------------------
# Drawing
# --------------------------------------------------------------------------
def _stadium(cx: float, cy: float, length: float, width: float, ang: float, n: int = 24) -> np.ndarray:
    """Capsule polygon centred at (cx, cy) with its axis along ``ang``."""
    half = length / 2.0 - width / 2.0
    r = width / 2.0
    t = np.linspace(-math.pi / 2, math.pi / 2, n)
    right = np.column_stack([half + r * np.cos(t), r * np.sin(t)])
    left = np.column_stack([-half - r * np.cos(t), -r * np.sin(t)])
    pts = np.vstack([right, left])
    c, s = math.cos(ang), math.sin(ang)
    rot = pts @ np.array([[c, s], [-s, c]])
    return rot + np.array([cx, cy])


def draw_morphology(ax, m: Morphology, show_organs: bool = True, outline_lw: float = 1.6,
                    smooth: float = 8.0) -> None:
    mask, (x0, x1, y0, y1) = silhouette(m, smooth=smooth)

    # body outline = (smoothed) union boundary of all regions
    cs = ax.contour(
        np.linspace(x0, x1, mask.shape[1]), np.linspace(y0, y1, mask.shape[0]),
        mask.astype(float), levels=[0.5], colors="#222222", linewidths=outline_lw, zorder=3,
    )
    clip = cs.get_paths()[0] if cs.get_paths() else None

    # region fills clipped to the body outline; heads/tails sit above the trunk
    order = {"Trunk": 0, "Tail": 1, "Head": 2}
    for rid, reg in sorted(m.regions.items(), key=lambda kv: order[kv[1].type]):
        patch = Polygon(m.outlines[rid], closed=True, facecolor=REGION_COLORS[reg.type],
                        edgecolor="none", alpha=0.85, zorder=1 + order[reg.type] * 0.1)
        ax.add_patch(patch)
        if clip is not None and len(clip.vertices):
            patch.set_clip_path(clip, transform=ax.transData)

    if not show_organs:
        return
    for o in m.organs:
        cx, cy = m.positions[o.region]
        px, py = cx + o.vx, cy + o.vy
        if o.kind == "VNC" and o.ends is not None:
            e1x, e1y, e2x, e2y = o.ends
            ax.plot([px + e1x, px + e2x], [py + e1y, py + e2y], color="#3a3a3a",
                    lw=1.1, alpha=0.8, zorder=4, solid_capstyle="round")
        elif o.kind == "Phar.":
            rot = o.rot or 0.0
            ax.add_patch(Polygon(_stadium(px, py, 95, 24, rot), closed=True, facecolor="#f4e7c3",
                                 edgecolor="#5a4a2a", lw=1.0, zorder=5))
            # the mouth opens at the end opposite to Rot (posterior in WT, Rot = pi/2)
            mx, my = px - 47 * math.cos(rot), py - 47 * math.sin(rot)
            ax.add_patch(Circle((mx, my), 6, facecolor="#5a4a2a", edgecolor="none", zorder=6))
        elif o.kind == "Brain":
            ax.add_patch(Ellipse((px, py), 34, 16, angle=math.degrees(o.rot or 0.0),
                                 facecolor="#7b5aa6", edgecolor="#3d2a5a", lw=0.8, alpha=0.9, zorder=5))
        elif o.kind == "Eye":
            rot = o.rot or 0.0
            ax.add_patch(Circle((px, py), 8, facecolor="#ffffff", edgecolor="#111111", lw=0.9, zorder=6))
            # pigment cup drawn on the side given by the organ rotation
            ax.add_patch(Circle((px + 3 * math.cos(rot), py + 3 * math.sin(rot)), 4.2,
                                facecolor="#111111", edgecolor="none", zorder=7))


def fit_axes(ax, bounds: Tuple[float, float, float, float], centre: Tuple[float, float] = None) -> None:
    """Square-aspect axes framing ``bounds`` (x0, x1, y0, y1)."""
    x0, x1, y0, y1 = bounds
    if centre is None:
        centre = ((x0 + x1) / 2.0, (y0 + y1) / 2.0)
    half = max(x1 - x0, y1 - y0) / 2.0
    ax.set_xlim(centre[0] - half, centre[0] + half)
    ax.set_ylim(centre[1] - half, centre[1] + half)
    ax.set_aspect("equal")
    ax.axis("off")


def frame_bounds(morphs: Sequence[Morphology]) -> float:
    """Common half-size so every morphology in a frame is drawn to the same scale."""
    half = 0.0
    for m in morphs:
        x0, x1, y0, y1 = extent(m)
        half = max(half, (x1 - x0) / 2.0, (y1 - y0) / 2.0)
    return half


def centre_of(m: Morphology) -> Tuple[float, float]:
    x0, x1, y0, y1 = extent(m)
    return ((x0 + x1) / 2.0, (y0 + y1) / 2.0)


def wrap_title(name: str, width: int = 26) -> str:
    return "\n".join(textwrap.wrap(name, width=width)[:2])


# --------------------------------------------------------------------------
# Outputs
# --------------------------------------------------------------------------
def render_gallery(morphs: Dict[int, Morphology], ids: Sequence[int], out: Path, cols: int = 4) -> None:
    sel = [morphs[i] for i in ids if i in morphs]
    rows = math.ceil(len(sel) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 3.6 * rows))
    axes = np.atleast_1d(axes).ravel()
    half = frame_bounds(sel)
    for ax, m in zip(axes, sel):
        draw_morphology(ax, m)
        cx, cy = centre_of(m)
        fit_axes(ax, (cx - half, cx + half, cy - half, cy + half))
        ax.set_title(f"#{m.id}  {wrap_title(m.name)}", fontsize=9)
    for ax in axes[len(sel):]:
        ax.axis("off")
    fig.suptitle("PlanformDB morphologies decoded from the Lobo–Malone–Levin (2013) graph formalism\n"
                 "head = red, trunk = grey, tail = blue; eyes, brain lobes, pharynx and ventral nerve cords from the organ table",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  saved {out}")


def _paginate(items: List[int], per_page: int) -> List[List[int]]:
    return [items[i:i + per_page] for i in range(0, len(items), per_page)] or [[]]


def render_timeline(morphs: Dict[int, Morphology], first: Dict, out: Path, fps: int = 2,
                    per_page: int = 8, hold_final: int = 6) -> None:
    pubs_per_year = first["_pubs_per_year"]
    by_year: Dict[int, List[int]] = defaultdict(list)
    for mid, (year, _sp, _t) in ((k, v) for k, v in first.items() if isinstance(k, int)):
        if mid in morphs:
            by_year[year].append(mid)
    years = sorted(by_year)
    cumulative = np.cumsum([len(by_year[y]) for y in years])

    frames: List[Tuple[str, int, List[int], int, int]] = []  # (kind, year, ids, page, npages)
    frames.append(("intro", years[0], [1], 1, 1))
    for y in years:
        pages = _paginate(sorted(by_year[y]), per_page)
        for i, page in enumerate(pages, 1):
            frames.append(("year", y, page, i, len(pages)))
    for _ in range(hold_final):
        frames.append(("final", years[-1], [], 1, 1))

    ncols = 4
    nrows = math.ceil(per_page / ncols)
    fig = plt.figure(figsize=(15, 7.8))
    gs = fig.add_gridspec(nrows, ncols + 2, width_ratios=[1.6, 0.15] + [1] * ncols, wspace=0.08, hspace=0.35)
    ax_curve = fig.add_subplot(gs[:, 0])
    cell_axes = [fig.add_subplot(gs[r, c + 2]) for r in range(nrows) for c in range(ncols)]
    wide_axes = [fig.add_subplot(gs[:, c + 2]) for c in range(ncols)]  # pages with <= ncols items
    ax_big = fig.add_subplot(gs[:, 2:])  # used by the intro and final frames only
    all_dated = [mid for y in years for mid in sorted(by_year[y])]

    def draw_curve(year: int) -> None:
        ax_curve.clear()
        ax_curve.step(years, cumulative, where="post", color="#333333", lw=1.8)
        idx = years.index(year)
        ax_curve.scatter([year], [cumulative[idx]], s=70, color="#d6453d", zorder=5)
        ax_curve.bar(years, [pubs_per_year.get(y, 0) for y in years], width=1.0,
                     color="#3f6fb5", alpha=0.35, label="publications / year")
        ax_curve.set_xlim(years[0] - 3, years[-1] + 3)
        ax_curve.set_ylim(0, cumulative[-1] * 1.08)
        ax_curve.set_xlabel("year of first report (Publication.Year)")
        ax_curve.set_ylabel("distinct regenerative morphologies catalogued (cumulative)")
        ax_curve.legend(loc="upper left", fontsize=8, frameon=False)
        ax_curve.spines[["top", "right"]].set_visible(False)

    def update(fi: int):
        kind, year, ids, page, npages = frames[fi]
        for ax in cell_axes + wide_axes + [ax_big]:
            ax.clear()
            ax.axis("off")
        draw_curve(year)

        if kind == "intro":
            wt = morphs[1]
            draw_morphology(ax_big, wt)
            fit_axes(ax_big, extent(wt, pad=60))
            ax_big.set_title(
                "Wild type (#1) as encoded in PlanformDB: 3 regions, 2 links, 7 organs\n"
                "region radii: head (60, 80, 60), trunk (60, 60), tail (50, 80, 50); border at Ratio·Dist = 100",
                fontsize=10)
            fig.suptitle("Planarian morphospace, decoded from the Planform graph formalism (Lobo, Malone & Levin 2013)",
                         fontsize=13)
            return

        if kind == "final":
            thumbs = ax_big
            n = len(all_dated)
            cell_h = 2.2  # planarians are ~4.6x longer than wide; use tall cells
            tc = math.ceil(math.sqrt(n * 1.5 * cell_h))
            tr = math.ceil(n / tc)
            for k, mid in enumerate(all_dated):
                r, c = divmod(k, tc)
                m = morphs[mid]
                x0, x1, y0, y1 = extent(m, pad=10)
                cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
                sc = min(0.92 / (x1 - x0), 0.92 * cell_h / (y1 - y0))  # fit each to its cell
                origin = np.array([c + 0.5, (tr - r - 0.5) * cell_h])
                for rid, reg in m.regions.items():
                    poly = (m.outlines[rid] - np.array([cx, cy])) * sc + origin
                    thumbs.add_patch(Polygon(poly, closed=True, facecolor=REGION_COLORS[reg.type],
                                             edgecolor="#222222", lw=0.3, alpha=0.9))
            thumbs.set_xlim(0, tc)
            thumbs.set_ylim(0, tr * cell_h)
            thumbs.set_aspect("equal")
            thumbs.set_title(f"All {n} dated morphologies (each fitted to its cell), ordered by year of first report",
                             fontsize=10)
            fig.suptitle("Planarian morphospace, decoded from the Planform graph formalism (Lobo, Malone & Levin 2013)",
                         fontsize=13)
            return

        sel = [morphs[i] for i in ids]
        half = frame_bounds(sel)
        targets = wide_axes if len(sel) <= ncols else cell_axes[:per_page]
        for ax, m in zip(targets, sel):
            draw_morphology(ax, m)
            cx, cy = centre_of(m)
            fit_axes(ax, (cx - half, cx + half, cy - half, cy + half))
            ax.set_title(f"#{m.id} {wrap_title(m.name, 24)}", fontsize=7.5)
            ax.text(0.5, -0.02, first[m.id][1], transform=ax.transAxes, ha="center", va="top",
                    fontsize=7, fontstyle="italic", color="#444444")
        titles = first[ids[0]][2] if ids else []
        src = textwrap.shorten("; ".join(titles), 110, placeholder=" …") if titles else ""
        page_txt = f"  (page {page}/{npages})" if npages > 1 else ""
        fig.suptitle(
            f"{year}: {len(by_year[year])} morphologies first reported{page_txt}   "
            f"— {pubs_per_year.get(year, 0)} publication(s)\n{src}",
            fontsize=12)

    anim = animation.FuncAnimation(fig, update, frames=len(frames), interval=1000 / fps, blit=False)
    anim.save(out, writer="pillow", fps=fps)
    plt.close(fig)
    print(f"  saved {out}  ({len(frames)} frames, {len(years)} years, {len(all_dated)} dated morphologies)")


def write_csv(morphs: Dict[int, Morphology], first: Dict, out: Path) -> None:
    import csv

    rows = []
    for m in morphs.values():
        row = morphometrics(m)
        y = first.get(m.id)
        row["first_year"] = y[0] if y else ""
        row["species"] = y[1] if y else ""
        rows.append(row)
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"  saved {out}  ({len(rows)} rows)")


def check_encoding(morphs: Dict[int, Morphology]) -> None:
    """Report whether every region obeys the parameter-count rule of the formalism."""
    bad = 0
    for m in morphs.values():
        for rid, reg in m.regions.items():
            k = len(_edges_of(m, rid))
            expected = 4 if k == 0 else (3 if k == 1 else k)
            if len([p for p in reg.params if not math.isnan(p)]) != expected:
                bad += 1
                print(f"  morphology {m.id} region {rid} ({reg.type}): {k} links but {len(reg.params)} params")
    n = sum(len(m.regions) for m in morphs.values())
    print(f"  encoding check: {n - bad}/{n} regions obey the 0→4 / 1→3 / k→k radii rule")
    wt = morphometrics(morphs[1])
    print(f"  wild type: AP length {wt['ap_length']} units, ML width {wt['ml_width']}, aspect {wt['aspect']}")


# --------------------------------------------------------------------------
def main(argv: Optional[Sequence[str]] = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("db", nargs="?", help="path to planformDB_*.edb (or set PLANFORM_DB_PATH)")
    p.add_argument("--out-dir", default=".", help="where to write outputs (default: cwd)")
    p.add_argument("--ids", type=int, nargs="*", help="morphology ids for the gallery (default: a curated set)")
    p.add_argument("--csv", help="write per-morphology morphometrics CSV to this path")
    p.add_argument("--no-anim", action="store_true", help="skip the timeline GIF")
    p.add_argument("--no-gallery", action="store_true", help="skip the gallery PNG")
    p.add_argument("--fps", type=int, default=2)
    p.add_argument("--per-page", type=int, default=8, help="morphologies per timeline frame")
    args = p.parse_args(argv)

    db = resolve_db_path(args.db)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {db}")
    morphs = load_morphologies(db)
    first = load_first_report_year(db)
    print(f"  {len(morphs)} morphologies, {sum(len(m.regions) for m in morphs.values())} regions, "
          f"{sum(len(m.links) for m in morphs.values())} links, {sum(len(m.organs) for m in morphs.values())} organs")
    check_encoding(morphs)

    if not args.no_gallery:
        # the most frequently reported outcomes plus the classic topological variants
        ids = args.ids or [1, 24, 20, 2, 21, 71, 133, 30, 14, 53, 3, 7, 12, 104, 364, 125]
        render_gallery(morphs, ids, out_dir / "planform_morphology_gallery.png")
    if args.csv:
        write_csv(morphs, first, Path(args.csv))
    if not args.no_anim:
        render_timeline(morphs, first, out_dir / "planform_morphospace_timeline.gif",
                        fps=args.fps, per_page=args.per_page)


if __name__ == "__main__":
    main()

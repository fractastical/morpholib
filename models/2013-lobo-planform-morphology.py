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
    python3 2013-lobo-planform-morphology.py --strict-db                # refuse any other .edb

The database is identified by SHA-256 (PLANFORM_DB_SHA256 below, also in
datasets/planform/planformDB.sha256); a different file is reported as a warning,
or refused with --strict-db.

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
from matplotlib.colors import LinearSegmentedColormap, to_rgb  # noqa: E402
from scipy.ndimage import gaussian_filter  # noqa: E402

TWO_PI = 2.0 * math.pi

REGION_COLORS = {  # Planform cartoon convention (Fig. 1D): head red, trunk grey, tail blue
    "Head": "#e04a3f",
    "Trunk": "#b9b9bd",
    "Tail": "#3d5fc4",
}
# Planform fades head/tail colour towards the trunk border; these are the faded ends
REGION_FADE = {"Head": "#f2b9b4", "Trunk": "#b9b9bd", "Tail": "#b6bfe6"}
ORGAN_GREEN = "#6cbf4a"
ORGAN_GREEN_EDGE = "#2f6a1f"
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
# The database the decoding rule was verified on and the audits were run against
# (planformDB_2.5.0.edb as distributed with Planform; 372 morphologies, 1181 regions).
# It is not vendored (no redistribution licence); its identity is pinned here and in
# datasets/planform/planformDB.sha256 so a run on another copy is flagged.
PLANFORM_DB_SHA256 = "9effd13ddb87664b9ef7a9b6c9c1959b502fcdd6c1b06ee016501b2d0be83b89"
PLANFORM_DB_BYTES = 744448


def db_fingerprint(path: str) -> tuple[str, int]:
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest(), Path(path).stat().st_size


def resolve_db_path(cli_path: Optional[str], strict: bool = False) -> str:
    path = cli_path or os.getenv("PLANFORM_DB_PATH")
    if not path or not Path(path).is_file():
        sys.exit(
            "PlanformDB not found. Pass the .edb path as the first argument or set "
            "PLANFORM_DB_PATH (planformDB_2.5.0.edb is distributed with Planform)."
        )
    digest, size = db_fingerprint(path)
    if digest == PLANFORM_DB_SHA256:
        print(f"PlanformDB sha256 ok ({size} bytes; planformDB_2.5.0.edb as audited)")
    else:
        msg = (f"PlanformDB {path}: sha256 {digest} ({size} bytes) is not the audited "
               f"planformDB_2.5.0.edb ({PLANFORM_DB_SHA256}, {PLANFORM_DB_BYTES} bytes); "
               "the decoding rule and the morphometrics were verified on that file only")
        if strict:
            sys.exit(msg)
        print("WARNING: " + msg)
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


N_FREE = 2.2    # superellipse exponent between two free (shape) radii: rounded cap
N_BORDER = 7.0  # exponent next to a border radius: nearly square, i.e. a flat border and straight side


def region_outline(m: Morphology, rid: int, per_arc: int = 60) -> np.ndarray:
    """Closed outline through the polar control points, drawn the way Planform's cartoons do.

    Consecutive control radii (θ1, r1), (θ2, r2) are joined by a superellipse arc
    p(t) = r1·û1·cos(t)^(2/n) + r2·û2·sin(t)^(2/n), t ∈ [0, π/2]. With n = 2 this is
    an ellipse quadrant; large n approaches the rectangle corner. Arcs that touch a
    border direction (the flat adjacency with a neighbouring region) use a large n,
    so the body gets straight sides and flat region borders; arcs between two
    shape radii use n ≈ 2, so free ends (head tip, tail tip) are rounded caps.
    Reproduces Fig. 1D of Lobo, Malone & Levin 2013 for the wild type.
    """
    pts = control_points(m, rid)
    border_angles = {round(e[0], 6) for e in _edges_of(m, rid)}
    cx, cy = m.positions[rid]
    out = []
    k = len(pts)
    for i in range(k):
        a1, r1 = pts[i]
        a2, r2 = pts[(i + 1) % k]
        if i == k - 1:
            a2 += TWO_PI
        is_border = round(a1 % TWO_PI, 6) in border_angles or round(a2 % TWO_PI, 6) in border_angles
        n = N_BORDER if is_border else N_FREE
        t = np.linspace(0, math.pi / 2, per_arc, endpoint=False)
        u1 = np.array([math.cos(a1), math.sin(a1)])
        u2 = np.array([math.cos(a2), math.sin(a2)])
        c = np.cos(t) ** (2 / n)
        s = np.sin(t) ** (2 / n)
        arc = np.outer(c, r1 * u1) + np.outer(s, r2 * u2)
        out.append(arc)
    poly = np.vstack(out) + np.array([cx, cy])
    return poly


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


def _free_direction(m: Morphology, rid: int) -> Optional[float]:
    """Direction of a one-link region's free end (opposite its border), else None."""
    edges = _edges_of(m, rid)
    if len(edges) != 1:
        return None
    return (edges[0][0] + math.pi) % TWO_PI


def _gradient_fill(ax, poly: np.ndarray, c_tip: str, c_base: str, direction: Optional[float], zorder: float) -> None:
    """Fill ``poly`` with a linear gradient from c_tip (free end) to c_base (border side)."""
    x0, y0 = poly.min(axis=0)
    x1, y1 = poly.max(axis=0)
    patch = Polygon(poly, closed=True, facecolor="none", edgecolor="none")
    ax.add_patch(patch)
    if direction is None:
        ax.add_patch(Polygon(poly, closed=True, facecolor=c_base, edgecolor="none", zorder=zorder))
        return
    res = 64
    xs = np.linspace(x0, x1, res)
    ys = np.linspace(y0, y1, res)
    gx, gy = np.meshgrid(xs, ys)
    proj = (gx - (x0 + x1) / 2) * math.cos(direction) + (gy - (y0 + y1) / 2) * math.sin(direction)
    span = max(proj.max() - proj.min(), 1e-6)
    t = (proj - proj.min()) / span  # 0 at border side, 1 at free tip
    cmap = LinearSegmentedColormap.from_list("fade", [to_rgb(c_base), to_rgb(c_tip)])
    im = ax.imshow(cmap(t ** 0.7), extent=(x0, x1, y0, y1), origin="lower", interpolation="bilinear",
                   zorder=zorder, aspect="auto")
    im.set_clip_path(patch)


def draw_morphology(ax, m: Morphology, show_organs: bool = True, outline_lw: float = 1.6,
                    smooth: float = 6.0) -> None:
    """Planform-style cartoon: one smooth body outline, region fills with head/tail fades, organs."""
    mask, (x0, x1, y0, y1) = silhouette(m, smooth=smooth)

    # body outline = (smoothed) union boundary of all regions
    cs = ax.contour(
        np.linspace(x0, x1, mask.shape[1]), np.linspace(y0, y1, mask.shape[0]),
        mask.astype(float), levels=[0.5], colors="#333333", linewidths=outline_lw, zorder=3,
    )
    clip = cs.get_paths()[0] if cs.get_paths() else None

    # region fills (trunk first, then tail, then head) clipped to the body outline
    order = {"Trunk": 0, "Tail": 1, "Head": 2}
    for rid, reg in sorted(m.regions.items(), key=lambda kv: order[kv[1].type]):
        before = len(ax.images), len(ax.patches)
        _gradient_fill(ax, m.outlines[rid], REGION_COLORS[reg.type], REGION_FADE[reg.type],
                       _free_direction(m, rid), zorder=1 + order[reg.type] * 0.1)
        if clip is not None and len(clip.vertices):
            for art in list(ax.images[before[0]:]) + list(ax.patches[before[1]:]):
                art.set_clip_path(clip, transform=ax.transData)

    if not show_organs:
        return
    # ventral nerve cords first (under the other organs), then pharynx, brain, eyes
    for o in sorted(m.organs, key=lambda o: {"VNC": 0, "Phar.": 1, "Brain": 2, "Eye": 3}[o.kind]):
        cx, cy = m.positions[o.region]
        px, py = cx + o.vx, cy + o.vy
        if o.kind == "VNC" and o.ends is not None:
            e1x, e1y, e2x, e2y = o.ends
            ax.plot([px + e1x, px + e2x], [py + e1y, py + e2y], color=ORGAN_GREEN_EDGE, lw=4.2, alpha=0.9,
                    zorder=4, solid_capstyle="round")
            ax.plot([px + e1x, px + e2x], [py + e1y, py + e2y], color=ORGAN_GREEN, lw=2.6, zorder=4.1,
                    solid_capstyle="round")
        elif o.kind == "Phar.":
            rot = o.rot or 0.0
            ax.add_patch(Polygon(_stadium(px, py, 100, 26, rot), closed=True, facecolor="#d7d7dc",
                                 edgecolor="#55555c", lw=1.1, zorder=5))
            ax.add_patch(Polygon(_stadium(px, py, 84, 12, rot), closed=True, facecolor="#f0f0f3",
                                 edgecolor="#8a8a92", lw=0.7, zorder=5.1))
        elif o.kind == "Brain":
            ax.add_patch(Ellipse((px, py), 40, 22, angle=math.degrees(o.rot or 0.0),
                                 facecolor=ORGAN_GREEN, edgecolor=ORGAN_GREEN_EDGE, lw=1.0, zorder=5.2))
        elif o.kind == "Eye":
            rot = o.rot or 0.0
            # white oval with a black pupil displaced along the eye's rotation (Fig. 1D style)
            ax.add_patch(Ellipse((px, py), 16, 22, angle=math.degrees(rot) + 90,
                                 facecolor="#ffffff", edgecolor="#222222", lw=0.9, zorder=6))
            ax.add_patch(Circle((px + 3.5 * math.cos(rot), py + 3.5 * math.sin(rot) + 2.5), 4.5,
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


CANONICAL_SILHOUETTE_IDS: Tuple[int, ...] = (1, 2, 3, 20, 24)

CANONICAL_SILHOUETTE_ROLES: Dict[int, Dict[str, str]] = {
    1: {
        "canonical_key": "wild_type",
        "rusty_matter_outcome_label": "wild_type",
        "source_target_anchor": "source:planformdb_250::target:planformdb_curated_subset",
        "biological_role": "Normal wild-type anterior-posterior polarity (1 head, 1 trunk with 1 pharynx, 1 tail).",
    },
    2: {
        "canonical_key": "double_head_two_pharynxes",
        "rusty_matter_outcome_label": "double_head_two_pharynges",
        "source_target_anchor": "source:oviedo_2010_dev_biol::target:gap_block_conductance_and_crop_position",
        "biological_role": "Bipolar two-headed planarian with duplicated pharynx induced by octanol gap-junction blockade or triple innexin RNAi (Oviedo et al. 2010).",
    },
    3: {
        "canonical_key": "side_head",
        "rusty_matter_outcome_label": "ectopic_lateral_head",
        "source_target_anchor": "source:planformdb_250::target:planformdb_curated_subset",
        "biological_role": "Ectopic lateral head branching off the trunk with secondary pharynx and VNC.",
    },
    20: {
        "canonical_key": "headless",
        "rusty_matter_outcome_label": "failed_ap_regeneration",
        "source_target_anchor": "source:emmons_bell_2015_ijms::target:species_like_head_labels::future_outcome_taxonomy",
        "biological_role": "Acephalic / headless regenerant (trunk + tail only, no brain or eyes).",
    },
    24: {
        "canonical_key": "double_head_one_pharynx",
        "rusty_matter_outcome_label": "double_head_like",
        "source_target_anchor": "source:durant_2019_biophys_j::target:persistent_bioelectric_memory_vs_transient_modulation",
        "biological_role": "Bipolar two-headed planarian with a single central pharynx.",
    },
}


def silhouette_iou(m_a: Morphology, m_b: Morphology, res: int = 256) -> float:
    """Intersection-over-union of two decoded Planform silhouettes on a shared grid."""
    pts = np.vstack(list(m_a.outlines.values()) + list(m_b.outlines.values()))
    x0, x1 = float(pts[:, 0].min() - 10.0), float(pts[:, 0].max() + 10.0)
    y0, y1 = float(pts[:, 1].min() - 10.0), float(pts[:, 1].max() + 10.0)
    gx, gy = np.meshgrid(np.linspace(x0, x1, res), np.linspace(y0, y1, res))
    grid = np.column_stack([gx.ravel(), gy.ravel()])
    ma = np.zeros(len(grid), dtype=bool)
    mb = np.zeros(len(grid), dtype=bool)
    for poly in m_a.outlines.values():
        ma |= MplPath(poly).contains_points(grid)
    for poly in m_b.outlines.values():
        mb |= MplPath(poly).contains_points(grid)
    return float((ma & mb).sum() / max((ma | mb).sum(), 1))


def export_silhouettes_json(
    morphs: Dict[int, Morphology],
    out: Path,
    ids: Sequence[int] = CANONICAL_SILHOUETTE_IDS,
    outline_points_per_arc: int = 16,
) -> Dict:
    """Export a rights-safe JSON silhouette & morphometrics fixture for canonical Planform phenotypes.

    Designed for downstream consumption by MesmerPrism/rusty-matter
    (docs/PLANARIAN_METRIC_TARGETS.md) and 2021-Cervera-Levin-Mafe.py.
    Contains decoded region outlines, organ landmarks, morphometrics, and
    pairwise silhouette IoU against #1 Wild type, preserving the Lobo Lab
    PlanformDB 2.5.0 notice.
    """
    import json

    wt = morphs[1]
    records = []
    for mid in ids:
        if mid not in morphs:
            continue
        m = morphs[mid]
        mm = morphometrics(m)
        role = CANONICAL_SILHOUETTE_ROLES.get(
            mid,
            {
                "canonical_key": m.name.lower().replace(" ", "_").replace(",", ""),
                "rusty_matter_outcome_label": "unclassified_teaching_abstraction",
                "source_target_anchor": "source:planformdb_250::target:planformdb_curated_subset",
                "biological_role": m.name,
            },
        )
        regions_out = []
        for rid, reg in sorted(m.regions.items()):
            cx, cy = m.positions[rid]
            poly_compact = region_outline(m, rid, per_arc=outline_points_per_arc)
            regions_out.append(
                {
                    "region_id": int(rid),
                    "type": reg.type,
                    "center": [round(float(cx), 4), round(float(cy), 4)],
                    "radii_params": [round(float(p), 4) for p in reg.params if not math.isnan(p)],
                    "control_points_polar": [
                        {"angle_rad": round(float(ang), 6), "radius": round(float(rad), 4)}
                        for ang, rad in control_points(m, rid)
                    ],
                    "outline_polygon": [
                        [round(float(pt[0]), 3), round(float(pt[1]), 3)] for pt in poly_compact
                    ],
                }
            )
        links_out = [
            {
                "from_region": int(lk.from_region),
                "to_region": int(lk.to_region),
                "dist": round(float(lk.dist), 4),
                "ratio": round(float(lk.ratio), 6),
                "ang_rad": round(float(lk.ang), 6),
            }
            for lk in m.links
        ]
        organs_out = []
        for o in m.organs:
            rcx, rcy = m.positions[o.region]
            entry = {
                "region_id": int(o.region),
                "kind": o.kind,
                "offset_from_region": [round(float(o.vx), 4), round(float(o.vy), 4)],
                "world_position": [round(float(rcx + o.vx), 4), round(float(rcy + o.vy), 4)],
            }
            if o.rot is not None:
                entry["rotation_rad"] = round(float(o.rot), 6)
            if o.ends is not None:
                entry["line_ends_relative"] = [round(float(v), 4) for v in o.ends]
            organs_out.append(entry)

        x0, x1, y0, y1 = extent(m, pad=0.0)
        iou_vs_wt = 1.0 if mid == 1 else round(silhouette_iou(wt, m), 4)
        records.append(
            {
                "morphology_id": int(m.id),
                "name": m.name,
                "canonical_key": role["canonical_key"],
                "rusty_matter_outcome_label": role["rusty_matter_outcome_label"],
                "source_target_anchor": role["source_target_anchor"],
                "biological_role": role["biological_role"],
                "morphometrics": {
                    "n_regions": int(mm["n_regions"]),
                    "n_heads": int(mm["n_heads"]),
                    "n_trunks": int(mm["n_trunks"]),
                    "n_tails": int(mm["n_tails"]),
                    "n_eyes": int(mm["n_eyes"]),
                    "n_brains": int(mm["n_brains"]),
                    "n_pharynges": int(mm["n_pharynges"]),
                    "n_vnc": int(mm["n_vnc"]),
                    "ap_length": float(mm["ap_length"]),
                    "ml_width": float(mm["ml_width"]),
                    "aspect": float(mm["aspect"]),
                    "area": float(mm["area"]),
                    "length_rel_wt": float(mm["length_rel_wt"]),
                    "iou_vs_wild_type": float(iou_vs_wt),
                    "head_region_ap_fraction": round(
                        sum(
                            float(np.ptp(region_outline(m, r.id)[:, 1]))
                            for r in m.regions.values()
                            if r.type == "Head"
                        )
                        / max(float(mm["ap_length"]), 1e-6),
                        4,
                    ),
                },
                "bounding_box_unpadded": {
                    "x_min": round(float(x0), 3),
                    "x_max": round(float(x1), 3),
                    "y_min": round(float(y0), 3),
                    "y_max": round(float(y1), 3),
                },
                "regions": regions_out,
                "links": links_out,
                "organs": organs_out,
            }
        )

    payload = {
        "schema_id": "morpholib.planform.canonical_silhouettes.v1",
        "fixture_id": "fixture.planform.canonical_silhouettes.v1",
        "evidence_type": "decoded_graph_formalism_silhouettes",
        "image_policy": "generated mathematical polygons decoded from PlanformDB graph parameters; no paper figure reuse",
        "coordinate_convention": "+y is anterior, +x is right lateral; origin (0, 0) at root trunk centre; WT AP length = 560.0 units",
        "database_provenance": {
            "source_id": "planformdb_250",
            "file_name": "planformDB_2.5.0.edb",
            "sha256": PLANFORM_DB_SHA256,
            "bytes": PLANFORM_DB_BYTES,
            "notice": (
                "All contents of Planform, Copyright 2012-2016 Daniel Lobo. Use for research and "
                "educational purposes is free of charge, provided proper citation is given to the "
                "database and the functional work that gave rise to the datasets used."
            ),
            "primary_citations": [
                {
                    "authors": "Lobo D, Malone TJ, Levin M",
                    "year": 2013,
                    "title": "Towards a bioinformatics of patterning: a computational approach to understanding regulative morphogenesis",
                    "venue": "Biology Open 2(2):156-169",
                    "doi": "10.1242/bio.20123400",
                },
                {
                    "authors": "Lobo D, Malone TJ, Levin M",
                    "year": 2013,
                    "title": "Planform: an application and database of graph-encoded planarian regenerative experiments",
                    "venue": "Bioinformatics 29(8):1098-1100",
                    "doi": "10.1093/bioinformatics/btt088",
                },
            ],
        },
        "downstream_targets": [
            "MesmerPrism/rusty-matter::docs/PLANARIAN_METRIC_TARGETS.md",
            "source:planformdb_250::target:planformdb_curated_subset",
            "source:beane_2013_dev::target:head_size_scaling::future_metric",
            "source:emmons_bell_2015_ijms::target:species_like_head_labels::future_outcome_taxonomy",
        ],
        "morphologies": records,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
        fh.write("\n")
    print(f"  saved {out}  ({len(records)} canonical morphologies)")
    return payload


def load_canonical_fixture_morphologies(fixture_path: Optional[Path] = None) -> Dict[int, Morphology]:
    """Load the 6 canonical Planform morphologies from the vendored JSON fixture when
    `planformDB_2.5.0.edb` is not present locally."""
    import json
    fp = fixture_path or (Path(__file__).resolve().parent.parent / "datasets" / "planform" / "planform_canonical_silhouettes_v1.json")
    data = json.loads(fp.read_text(encoding="utf-8"))
    morphs: Dict[int, Morphology] = {}
    for rec in data.get("morphologies", []):
        mid = int(rec["morphology_id"])
        regs = {
            int(r["region_id"]): Region(
                id=int(r["region_id"]),
                type=str(r["type"]),
                params=[float(p) for p in r["radii_params"]],
            )
            for r in rec.get("regions", [])
        }
        lks = [
            Link(
                from_region=int(lk["from_region"]),
                to_region=int(lk["to_region"]),
                dist=float(lk["dist"]),
                ratio=float(lk["ratio"]),
                ang=float(lk["ang_rad"]),
            )
            for lk in rec.get("links", [])
        ]
        orgs = [
            Organ(
                kind=str(o["kind"]),
                region=int(o["region_id"]),
                vx=float(o["offset_from_region"][0]),
                vy=float(o["offset_from_region"][1]),
                rot=float(o["rotation_rad"]) if "rotation_rad" in o else None,
                ends=tuple(float(v) for v in o["line_ends_relative"]) if "line_ends_relative" in o else None,
            )
            for o in rec.get("organs", [])
        ]
        m = Morphology(id=mid, name=str(rec["name"]), regions=regs, links=lks, organs=orgs)
        layout(m)
        m.outlines = {rid: region_outline(m, rid) for rid in m.regions}
        morphs[mid] = m
    return morphs


def export_implicit_slice_2d(
    morphs: Dict[int, Morphology],
    out: Path,
    ids: Sequence[int] = CANONICAL_SILHOUETTE_IDS,
    sdf_res: int = 64,
) -> Dict:
    """Export decoded Planform morphologies as a renderer-neutral `ImplicitSlice2D` JSON
    bundle (compatible with MesmerPrism/rusty-optics & rusty-lattice), including region
    polygons, organ landmarks, and signed distance fields (SDF < 0 inside, > 0 outside)."""
    import json
    from scipy.ndimage import distance_transform_edt

    slices = []
    for mid in ids:
        if mid not in morphs:
            continue
        m = morphs[mid]
        mask, (x0, x1, y0, y1) = silhouette(m, res=sdf_res, smooth=0.0)
        dx = (x1 - x0) / max(sdf_res - 1, 1)
        dy = (y1 - y0) / max(sdf_res - 1, 1)
        d_out = distance_transform_edt(~mask, sampling=(dy, dx))
        d_in = distance_transform_edt(mask, sampling=(dy, dx))
        sdf = d_out - d_in
        mm = morphometrics(m)
        regions_list = [
            {
                "region_id": int(rid),
                "type": reg.type,
                "centre_xy": [round(float(m.positions[rid][0]), 3), round(float(m.positions[rid][1]), 3)],
                "polygon_xy": [[round(float(px), 3), round(float(py), 3)] for px, py in m.outlines[rid][::2]],
            }
            for rid, reg in sorted(m.regions.items())
        ]
        organs_list = [
            {
                "organ_index": int(idx),
                "kind": o.kind,
                "region_id": int(o.region),
                "centre_xy": [
                    round(float(m.positions[o.region][0] + o.vx), 3),
                    round(float(m.positions[o.region][1] + o.vy), 3),
                ],
            }
            for idx, o in enumerate(m.organs)
        ]
        slices.append(
            {
                "schema_version": "rusty-optics-implicit-slice-2d.v1",
                "morphology_id": int(mid),
                "name": m.name,
                "grid_resolution": [int(sdf_res), int(sdf_res)],
                "bounds_xy": [round(float(x0), 3), round(float(x1), 3), round(float(y0), 3), round(float(y1), 3)],
                "morphometrics": mm,
                "regions": regions_list,
                "organs": organs_list,
                "sdf_min": round(float(sdf.min()), 4),
                "sdf_max": round(float(sdf.max()), 4),
                "zero_contour_area_units2": float(mm["area"]),
                "sdf_grid": [[round(float(val), 3) for val in row] for row in sdf],
            }
        )
    payload = {
        "schema_version": "rusty-optics-implicit-slice-2d-bundle.v1",
        "upstream_attribution": "MesmerPrism/rusty-optics & rusty-lattice (MIT OR Apache-2.0)",
        "model": "Lobo et al. (2013) PlanformDB 2.5.0 graph-decoded 2D planarian morphologies",
        "slices": slices,
    }
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"  exported ImplicitSlice2D JSON bundle ({len(slices)} slices) to {out}")
    return payload


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


def self_test() -> int:
    """Offline deterministic checks (Pass N+1 remediation): canonical fixture
    decode, wild-type morphometrics, and — when PLANFORM_DB_PATH is set — the
    pinned PlanformDB fingerprint. Returns 0 on pass, 1 on fail."""
    fails = []
    morphs = load_canonical_fixture_morphologies()
    for mid in CANONICAL_SILHOUETTE_IDS:
        if mid not in morphs:
            fails.append(f"canonical morphology #{mid} missing from fixture")
    if not fails:
        wt = morphometrics(morphs[1])
        if not (450.0 <= wt["ap_length"] <= 650.0 and wt["aspect"] > 3.0):
            fails.append(f"wild-type morphometrics out of range: {wt}")
        dh = morphometrics(morphs[2])
        if dh["n_heads"] < 2:
            fails.append(f"double-head #2 has n_heads={dh['n_heads']}, expected >= 2")
        print(f"[{'PASS' if not fails else 'FAIL'}] canonical fixture: WT AP "
              f"{wt['ap_length']} x ML {wt['ml_width']} (aspect {wt['aspect']}), "
              f"#2 heads={dh['n_heads']}")
    db_env = os.getenv("PLANFORM_DB_PATH")
    if db_env and Path(db_env).is_file():
        digest, size = db_fingerprint(db_env)
        ok = digest == PLANFORM_DB_SHA256 and size == PLANFORM_DB_BYTES
        print(f"[{'PASS' if ok else 'FAIL'}] PlanformDB fingerprint {digest[:12]}… ({size} bytes)")
        if not ok:
            fails.append("PLANFORM_DB_PATH file is not the audited planformDB_2.5.0.edb")
    for f in fails:
        print(f"  - {f}")
    print(f"[{'PASS' if not fails else 'FAIL'}] 2013-lobo-planform-morphology self-test")
    return 1 if fails else 0


# --------------------------------------------------------------------------
def main(argv: Optional[Sequence[str]] = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("db", nargs="?", help="path to planformDB_*.edb (or set PLANFORM_DB_PATH)")
    p.add_argument("--self-test", action="store_true",
                   help="run offline deterministic checks (canonical fixture decode + WT morphometrics) and exit")
    p.add_argument("--strict-db", action="store_true",
                   help="exit instead of warning if the database is not the audited planformDB_2.5.0.edb")
    p.add_argument("--out-dir", default=".", help="where to write outputs (default: cwd)")
    p.add_argument("--ids", type=int, nargs="*", help="morphology ids for the gallery (default: a curated set)")
    p.add_argument("--csv", help="write per-morphology morphometrics CSV to this path")
    p.add_argument("--export-silhouettes", help="write canonical Planform silhouettes JSON fixture to this path")
    p.add_argument("--export-frame", help="write renderer-neutral ImplicitSlice2D JSON bundle (polygons + SDF) to this path")
    p.add_argument("--no-anim", action="store_true", help="skip the timeline GIF")
    p.add_argument("--no-gallery", action="store_true", help="skip the gallery PNG")
    p.add_argument("--fps", type=int, default=2)
    p.add_argument("--per-page", type=int, default=8, help="morphologies per timeline frame")
    args = p.parse_args(argv)

    if args.self_test:
        sys.exit(self_test())

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cand_db = args.db or os.getenv("PLANFORM_DB_PATH")
    if (not cand_db or not Path(cand_db).is_file()) and args.export_frame and args.no_gallery and args.no_anim and not args.strict_db:
        print("PlanformDB .edb not provided; loading canonical morphologies from datasets/planform/planform_canonical_silhouettes_v1.json")
        morphs = load_canonical_fixture_morphologies()
        export_ids = args.ids if args.ids else CANONICAL_SILHOUETTE_IDS
        export_implicit_slice_2d(morphs, Path(args.export_frame), ids=export_ids)
        return

    db = resolve_db_path(args.db, strict=args.strict_db)

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
    if args.export_silhouettes:
        export_ids = args.ids if args.ids else CANONICAL_SILHOUETTE_IDS
        export_silhouettes_json(morphs, Path(args.export_silhouettes), ids=export_ids)
    if args.export_frame:
        export_ids = args.ids if args.ids else CANONICAL_SILHOUETTE_IDS
        export_implicit_slice_2d(morphs, Path(args.export_frame), ids=export_ids)
    if not args.no_anim:
        render_timeline(morphs, first, out_dir / "planform_morphospace_timeline.gif",
                        fps=args.fps, per_page=args.per_page)


if __name__ == "__main__":
    main()

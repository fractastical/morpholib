#!/usr/bin/env python3
"""Limbform (Lobo, Feldman, Shah, Malone & Levin 2014): decode and draw every
limb morphology in LimbformDB.

The sibling of ``2013-lobo-planform-morphology.py``.  Limbform is the Levin/Lobo
lab's formal database of limb-regeneration experiments: 565 limb morphologies of
salamanders, frogs, crustaceans, insects and arachnids (axolotl, newts,
*Xenopus*, fiddler crab, crayfish, *Cancer* crabs, cockroaches, cricket,
huntsman spider), each encoded as a labelled graph.  Figure 1 of the
*Regeneration* paper shows, for one species per category, the photograph, the
graph and the cartoon the software draws from it; this script reproduces the
cartoon column for all 565 from the database alone.

References
----------
  Lobo D, Feldman EB, Shah M, Malone TJ, Levin M (2014). Limbform: a functional
  ontology-based database of limb regeneration experiments.
  Bioinformatics 30(24):3598-3600. https://doi.org/10.1093/bioinformatics/btu582

  Lobo D, Feldman EB, Shah M, Malone TJ, Levin M (2014). A bioinformatics expert
  system linking functional data to anatomical outcomes in limb regeneration.
  Regeneration 1(2):37-56. https://doi.org/10.1002/reg2.13

  Database: LimbformDB 2.1.0 (31 Jul 2014), SQLite, from
  https://lobolab.umbc.edu/limbform/download/  (--download fetches and verifies it)

The formalism (Regeneration 1:37, "Formalism for limb morphologies")
--------------------------------------------------------------------
* Limb SEGMENTS are graph edges (``Link``: Type, FromJoint, ToJoint, Dist, Ang,
  Side, Reversed, Inverted, NumBones).  ``Ang`` is the direction of the segment
  from its proximal to its distal joint, ``Dist`` its length.
* JOINTS are graph vertices (``Joint`` + ``JointParam``).  The parameters are
  radii from the joint centre to the limb border: one per bisector of each pair
  of consecutive segments at the joint, or, for a joint with a single segment,
  three radii at +90, +180 and +270 degrees from the segment direction.  This
  is the same rule as Planform's regions; here it holds for 3137 of the 3146
  joints in LimbformDB 2.1.0 (checked at load time) once a sub-joint counts its
  host segment in both directions.
* SUB-JOINTS (``LinkSubJoint``) sit on a segment at (Dist, Ang) from its
  proximal joint; grafted or ectopic limbs attach there.
* DIGITS (``Digit``) hang off a hand/foot segment: direction ``Ang``, length
  ``Dist`` from the segment's distal joint, ``NumBones`` phalanges+metacarpal.
* ``NumBones`` on a segment is the number of long bones it contains (1 humerus,
  2 radius+ulna, 8 carpals, ...); drawn for salamanders and frogs only, as in
  the Limbform cartoons.  ``Side``/``Reversed``/``Inverted`` are read and
  reported; no morphology in 2.1.0 is reversed or inverted.

Usage
-----
    python3 2014-lobo-limbform-morphology.py --download          # fetch LimbformDB into .limbform_cache/ (SHA-256 checked)
    python3 2014-lobo-limbform-morphology.py /path/to/limbformDB_2.1.0.ldb
    export LIMBFORM_DB_PATH=/path/to/limbformDB_2.1.0.ldb
    python3 2014-lobo-limbform-morphology.py --ids 1 280 314 490 491 --no-anim
    python3 2014-lobo-limbform-morphology.py --csv limb_morphospace.csv

Outputs (git-ignored): limbform_wildtype_gallery.png (the five wild types of the
paper's Fig. 1 plus the other wild-type limbs), limbform_morphospace_timeline.gif
(year-by-year first reports of each morphology, ending with one morphospace grid
per category), and optionally a CSV of per-morphology measurements.
"""

from __future__ import annotations

import argparse
import hashlib
import math
import os
import sqlite3
import sys
import urllib.request
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib

if not os.environ.get("DISPLAY") and sys.platform != "darwin":
    matplotlib.use("Agg")

import matplotlib.colors  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Ellipse, PathPatch, Polygon  # noqa: E402
from matplotlib.path import Path as MplPath  # noqa: E402
from PIL import Image  # noqa: E402
from scipy.ndimage import gaussian_filter  # noqa: E402

HERE = Path(__file__).resolve().parent
CACHE_DIR = HERE / ".limbform_cache"
DB_URL = "https://lobolab.umbc.edu/limbform/downloads/limbformDB_2.1.0.ldb"
DB_SHA256 = "9c26bf3930bcc225f900f414cbd3a2ad3ac2f4d99e694d389b4fc36de15099f0"   # retrieved 2026-10-14
DB_BYTES = 897_024

TWO_PI = 2 * math.pi
N_FREE = 2.2    # superellipse exponent for the rounded cap of a terminal joint
FIG1_IDS = [1, 280, 314, 490, 491]   # axolotl forelimb, Xenopus hindlimb, fiddler crab, cricket T3, huntsman spider leg I
# Fig. 1 row + other wild-type entries: salamander hindlimb & stage-41 larva, frog forelimb & stage-51 hindlimb,
# Cancer claw, crayfish 4th walking leg, fiddler-crab claw, Blaberus foreleg, Blattella leg, spider pedipalp
GALLERY_IDS = FIG1_IDS + [443, 42, 278, 588, 375, 336, 461, 446, 487, 546]

# Proximal -> distal order of segment types per category, used for colour.
SEGMENT_ORDER = {
    "Salamander": ["Trunk", "Upper arm", "Thigh", "Fore arm", "Shank", "Hand", "Foot"],
    "Frog": ["Trunk", "Upper arm", "Thigh", "Fore arm", "Shank", "Hand", "Foot"],
    "Crustacean": ["Cephalothorax", "Abdomen", "Coxa", "Basis", "Ischium", "Merus", "Carpus", "Propus", "Dactyl",
                   "Fixed finger"],
    "Insect": ["Head", "Thorax", "Abdomen", "Coxa", "Trochanter", "Femur", "Tibia", "Tarsomere", "Pretarsus"],
    "Arachnid": ["Cephalothorax", "Abdomen", "Coxa", "Trochanter", "Femur", "Patella", "Tibia", "Metatarsus",
                 "Tarsus", "Pretarsus"],
}
BODY_TYPES = {"Trunk", "Cephalothorax", "Abdomen", "Head", "Thorax"}
PALETTE = ["#c9c9cf", "#e8a33d", "#6fb36a", "#4f86c6", "#b86fc0", "#d95f5f", "#3fb8b0", "#a0744a", "#7f8c3a",
           "#e07ba4", "#5f6fd9"]
BODY_COLOR = "#dcdcdc"
# Fig. 1 colour scheme of the Limbform cartoons
GRADIENT_CMAP = "RdBu"          # red proximal -> white -> blue distal
COXA_GREEN = "#5aa64a"
SECOND_YELLOW = "#f0c330"
FLAT_KIND = {**{t: 1 for t in BODY_TYPES}, "Coxa": 2, "Basis": 3, "Trochanter": 3}
DIGIT_BLUE = "#2a4d9b"
BONE_FACE = "#161616"
BONE_EDGE = "#161616"
DIGIT_BONE = "#0b1a4a"
DRAWS_BONES = {"Salamander", "Frog"}


# --------------------------------------------------------------------------
# Data model
# --------------------------------------------------------------------------
@dataclass
class Joint:
    id: int
    params: List[float]
    host_link: Optional[int] = None          # set for sub-joints
    host_dist: float = 0.0
    host_ang: float = 0.0


@dataclass
class Digit:
    dist: float
    ang: float
    n_bones: int
    reversed: bool


@dataclass
class Link:
    id: int
    type: str
    from_joint: int
    to_joint: int
    dist: float
    ang: float
    side: bool
    reversed: bool
    inverted: bool
    n_bones: int
    digits: List[Digit] = field(default_factory=list)
    subjoints: List[int] = field(default_factory=list)


@dataclass
class Morphology:
    id: int
    category: str
    name: str
    joints: Dict[int, Joint]
    links: Dict[int, Link]
    positions: Dict[int, Tuple[float, float]] = field(default_factory=dict)
    control: Dict[int, List[Tuple[float, float]]] = field(default_factory=dict)   # joint -> [(angle, radius)]
    pieces: List[Tuple[Link, int, int]] = field(default_factory=list)              # (link, joint a, joint b); links split at sub-joints
    segments: List[Tuple[Link, np.ndarray]] = field(default_factory=list)          # (link, polygon) per piece (fallback)
    outline: Optional[np.ndarray] = None                                           # closed smooth border (tree limbs)
    tour_points: List[Tuple[float, float]] = field(default_factory=list)           # raw control points along the tour
    pd_of_joint: Dict[int, float] = field(default_factory=dict)                    # proximodistal path distance from the root
    digit_polys: List[Tuple[Digit, np.ndarray, Tuple[float, float], float]] = field(default_factory=list)

    @property
    def n_digits(self) -> int:
        return sum(len(l.digits) for l in self.links.values())


def _b(v) -> bool:
    return str(v).lower() in ("1", "true")


# --------------------------------------------------------------------------
# Database
# --------------------------------------------------------------------------
def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def resolve_db_path(cli_path: Optional[str], download: bool) -> str:
    path = cli_path or os.getenv("LIMBFORM_DB_PATH")
    cached = CACHE_DIR / "limbformDB_2.1.0.ldb"
    if not path and cached.exists():
        path = str(cached)
    if not path and download:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        print(f"  downloading {DB_URL} -> {cached}")
        req = urllib.request.Request(DB_URL, headers={"User-Agent": "morpholib/limbform"})
        with urllib.request.urlopen(req, timeout=120) as resp, open(cached, "wb") as fh:
            fh.write(resp.read())
        path = str(cached)
    if not path or not Path(path).is_file():
        sys.exit("LimbformDB not found: pass the .ldb path, set LIMBFORM_DB_PATH, or use --download "
                 "(https://lobolab.umbc.edu/limbform/download/).")
    got = sha256_of(Path(path))
    if got != DB_SHA256:
        print(f"  warning: {Path(path).name} SHA-256 {got[:12]}… differs from LimbformDB 2.1.0 ({DB_SHA256[:12]}…); "
              "results may differ from the documented ones")
    return path


def load_morphologies(db_path: str) -> Dict[int, Morphology]:
    conn = sqlite3.connect(db_path)
    try:
        cur = conn.cursor()
        cats = dict(cur.execute("SELECT Id, Name FROM Category"))
        ltypes = dict(cur.execute("SELECT Id, Name FROM LinkType"))
        morphs: Dict[int, Morphology] = {}
        for mid, cat, name in cur.execute("SELECT Id, Category, Name FROM Morphology ORDER BY Id"):
            morphs[mid] = Morphology(mid, cats[cat], " ".join(name.split()), {}, {})
        owner: Dict[int, int] = {}
        for jid, mid in cur.execute("SELECT Id, Morphology FROM Joint"):
            morphs[mid].joints[jid] = Joint(jid, [])
            owner[jid] = mid
        for jid, ind, val in cur.execute("SELECT Joint, ParamInd, Value FROM JointParam ORDER BY Joint, ParamInd"):
            morphs[owner[jid]].joints[jid].params.append(float(val))
        for row in cur.execute("SELECT Id, Type, FromJoint, ToJoint, Dist, Ang, Side, Reversed, Inverted, NumBones FROM Link"):
            lid, t, fj, tj, dist, ang, side, rev, inv, nb = row
            morphs[owner[fj]].links[lid] = Link(lid, ltypes[t], fj, tj, float(dist), float(ang), _b(side), _b(rev),
                                                _b(inv), int(nb or 0))
        for lid, dist, ang, nb, rev in cur.execute("SELECT Link, Dist, Ang, NumBones, Reversed FROM Digit ORDER BY Id"):
            for m in morphs.values():
                if lid in m.links:
                    m.links[lid].digits.append(Digit(float(dist), float(ang), int(nb), _b(rev)))
                    break
        for lid, jid, dist, ang in cur.execute("SELECT Link, Joint, Dist, Ang FROM LinkSubJoint"):
            m = morphs[owner[jid]]
            if lid in m.links:
                m.links[lid].subjoints.append(jid)
                m.joints[jid].host_link, m.joints[jid].host_dist, m.joints[jid].host_ang = lid, float(dist), float(ang)
    finally:
        conn.close()
    empty = [mid for mid, m in morphs.items() if not m.links]
    for mid in empty:
        print(f"  skipping morphology {mid} '{morphs[mid].name}': no segments encoded")
        del morphs[mid]
    for m in morphs.values():
        layout(m)
        build_geometry(m)
    return morphs


def load_first_report_year(db_path: str) -> Dict[int, Tuple[int, str, List[str]]]:
    """Morphology id -> (first year reported as a result, species, publication titles)."""
    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute(
            "SELECT rm.Morphology, p.Year, s.Name, p.Title FROM ResultantMorphology rm "
            "JOIN ResultSet rs ON rs.Id = rm.ResultSet JOIN Experiment e ON e.Id = rs.Experiment "
            "JOIN Publication p ON p.Id = e.Publication JOIN Species s ON s.Id = e.Species "
            "WHERE p.Year IS NOT NULL ORDER BY p.Year").fetchall()
    finally:
        conn.close()
    out: Dict[int, Tuple[int, str, List[str]]] = {}
    for mid, year, species, title in rows:
        if mid not in out:
            out[mid] = (int(year), species, [title])
        elif out[mid][0] == year and title not in out[mid][2]:
            out[mid][2].append(title)
    return out


# --------------------------------------------------------------------------
# Geometry: layout, joint control points, segment polygons
# --------------------------------------------------------------------------
def layout(m: Morphology) -> None:
    """Joint positions from the segment Dist/Ang labels (BFS from the first joint)."""
    adj: Dict[int, List[Tuple[int, float, float]]] = defaultdict(list)
    for l in m.links.values():
        dx, dy = l.dist * math.cos(l.ang), l.dist * math.sin(l.ang)
        adj[l.from_joint].append((l.to_joint, dx, dy))
        adj[l.to_joint].append((l.from_joint, -dx, -dy))
    for j in m.joints.values():
        if j.host_link is not None and j.host_link in m.links:
            host = m.links[j.host_link]
            dx, dy = j.host_dist * math.cos(j.host_ang), j.host_dist * math.sin(j.host_ang)
            adj[host.from_joint].append((j.id, dx, dy))
            adj[j.id].append((host.from_joint, -dx, -dy))
    pos: Dict[int, Tuple[float, float]] = {}
    for start in m.joints:
        if start in pos:
            continue
        pos[start] = (0.0, 0.0)
        q = deque([start])
        while q:
            a = q.popleft()
            for b, dx, dy in adj[a]:
                if b not in pos:
                    pos[b] = (pos[a][0] + dx, pos[a][1] + dy)
                    q.append(b)
    m.positions = pos


def _edges_of(m: Morphology, jid: int) -> List[float]:
    """Directions (radians) of every segment leaving the joint; a sub-joint's host
    segment counts in both directions."""
    dirs: List[float] = []
    for l in m.links.values():
        if l.from_joint == jid:
            dirs.append(l.ang % TWO_PI)
        if l.to_joint == jid:
            dirs.append((l.ang + math.pi) % TWO_PI)
    j = m.joints[jid]
    if j.host_link is not None and j.host_link in m.links:
        host = m.links[j.host_link]
        dirs += [host.ang % TWO_PI, (host.ang + math.pi) % TWO_PI]
    return sorted(dirs)


def control_points(m: Morphology, jid: int) -> List[Tuple[float, float]]:
    """[(direction, radius)] of the joint's border points, in edge order."""
    j = m.joints[jid]
    edges = _edges_of(m, jid)
    if not edges:
        return []
    if len(edges) == 1:
        dirs = [(edges[0] + k * math.pi / 2) % TWO_PI for k in (1, 2, 3)]
    else:
        dirs = []
        for i, a0 in enumerate(edges):
            a1 = edges[(i + 1) % len(edges)]
            if i == len(edges) - 1:
                a1 += TWO_PI
            dirs.append(((a0 + a1) / 2) % TWO_PI)
    params = [p for p in j.params if not math.isnan(p)]
    if len(params) < len(dirs):                       # 9 joints in 2.1.0 are short of parameters
        fill = float(np.mean(params)) if params else 20.0
        params = params + [fill] * (len(dirs) - len(params))
    # kept in edge order: point i lies on the bisector between edges i and i+1 (or +90/+180/+270 for a terminal)
    return list(zip(dirs, [max(p, 1.0) for p in params[:len(dirs)]]))


def _neighbours(pts: List[Tuple[float, float]], direction: float) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """The control points just before and just after an edge direction (angularly)."""
    d = direction % TWO_PI
    before = max(pts, key=lambda p: ((p[0] - d) % TWO_PI) if (p[0] - d) % TWO_PI > 1e-9 else -1)  # largest below d
    after = min(pts, key=lambda p: ((p[0] - d) % TWO_PI) if (p[0] - d) % TWO_PI > 1e-9 else TWO_PI + 1)
    return before, after


def _xy(centre: Tuple[float, float], a: float, r: float) -> np.ndarray:
    return np.array([centre[0] + r * math.cos(a), centre[1] + r * math.sin(a)])


def _cap(centre: Tuple[float, float], p_left: Tuple[float, float], p_tip: Tuple[float, float],
         p_right: Tuple[float, float], n: int = 24) -> np.ndarray:
    """Rounded terminal cap: two superellipse quarter arcs left -> tip -> right."""
    out = []
    for (a1, r1), (a2, r2) in [(p_left, p_tip), (p_tip, p_right)]:
        t = np.linspace(0, math.pi / 2, n, endpoint=False)
        u1 = np.array([math.cos(a1), math.sin(a1)])
        u2 = np.array([math.cos(a2), math.sin(a2)])
        out.append(np.outer(np.cos(t) ** (2 / N_FREE), r1 * u1) + np.outer(np.sin(t) ** (2 / N_FREE), r2 * u2))
    return np.vstack(out) + np.array(centre)


def build_geometry(m: Morphology) -> None:
    """Control points at every joint, the closed outline (Euler tour of the limb tree, smoothed),
    per-piece polygons as a fallback for non-tree graphs, proximodistal distances and digits."""
    for jid in m.joints:
        m.control[jid] = control_points(m, jid)
    m.pieces = []
    for l in m.links.values():
        subs = sorted(l.subjoints, key=lambda s: m.joints[s].host_dist)   # from -> sub-joints -> to
        chain = [l.from_joint] + subs + [l.to_joint]
        m.pieces += [(l, a, b) for a, b in zip(chain[:-1], chain[1:])]
    m.segments = []
    for l, a, b in m.pieces:
        poly = segment_polygon(m, l, a, b)
        if poly is not None:
            m.segments.append((l, poly))
    m.tour_points = boundary_tour(m)
    m.outline = catmull_rom_closed(np.array(m.tour_points)) if len(m.tour_points) >= 3 else None
    m.pd_of_joint = proximodistal_distances(m)
    m.digit_polys = []
    for l in m.links.values():
        base = m.positions[l.to_joint]
        for d in l.digits:
            m.digit_polys.append((d, digit_polygon(base, d), base, l.ang))


def _piece_dir(m: Morphology, l: Link, a: int, b: int) -> float:
    ca, cb = m.positions[a], m.positions[b]
    return math.atan2(cb[1] - ca[1], cb[0] - ca[0]) if ca != cb else l.ang


def _rotation_system(m: Morphology) -> Dict[int, List[Tuple[float, int]]]:
    """joint -> [(direction, neighbour joint)] sorted by direction (the planar embedding)."""
    adj: Dict[int, List[Tuple[float, int]]] = defaultdict(list)
    for l, a, b in m.pieces:
        th = _piece_dir(m, l, a, b)
        adj[a].append((th % TWO_PI, b))
        adj[b].append(((th + math.pi) % TWO_PI, a))
    for j in adj:
        adj[j].sort()
    return adj


def boundary_tour(m: Morphology) -> List[Tuple[float, float]]:
    """Walk once around the limb tree and collect the border control points in order.

    Arriving at a joint along one segment, the walk leaves along the next segment
    counter-clockwise and emits the border point on the bisector between the two
    (that joint parameter); at a terminal joint it emits the three points at
    +90/+180/+270 degrees.  Returns [] if the graph is not a tree."""
    adj = _rotation_system(m)
    n_edges = len(m.pieces)
    joints = set(adj)
    if not joints or n_edges != len(joints) - 1:
        return []
    terminals = sorted(j for j in joints if len(adj[j]) == 1)
    if not terminals:
        return []
    start = terminals[0]
    pts: List[Tuple[float, float]] = []
    prev, cur = adj[start][0][1], start           # pretend we arrived at `start` from its only neighbour
    for _ in range(2 * n_edges):                  # every directed edge exactly once
        edges = adj[cur]
        nbs = [nb for _, nb in edges]
        i = nbs.index(prev)
        d_in = edges[i][0]
        c, ctrl = m.positions[cur], m.control[cur]
        if len(edges) == 1:                       # terminal: +90, +180, +270 from the incoming direction
            order = sorted(ctrl, key=lambda p: (p[0] - d_in) % TWO_PI)
            pts += [(c[0] + r * math.cos(a), c[1] + r * math.sin(a)) for a, r in order]
            nxt = nbs[0]
        else:                                     # bisector between edges[i] and edges[i+1]
            a, r = ctrl[i]
            pts.append((c[0] + r * math.cos(a), c[1] + r * math.sin(a)))
            nxt = nbs[(i + 1) % len(edges)]
        prev, cur = cur, nxt
    return pts


def catmull_rom_closed(P: np.ndarray, samples: int = 10, alpha: float = 0.5) -> np.ndarray:
    """Closed centripetal Catmull-Rom spline through the points (consecutive duplicates removed)."""
    keep = [0]
    for i in range(1, len(P)):
        if np.linalg.norm(P[i] - P[keep[-1]]) > 1e-6:
            keep.append(i)
    if np.linalg.norm(P[keep[-1]] - P[keep[0]]) <= 1e-6 and len(keep) > 1:
        keep.pop()
    P = P[keep]
    n = len(P)
    if n < 3:
        return P
    out = []
    for i in range(n):
        p0, p1, p2, p3 = P[(i - 1) % n], P[i], P[(i + 1) % n], P[(i + 2) % n]
        t0 = 0.0
        t1 = t0 + max(np.linalg.norm(p1 - p0), 1e-9) ** alpha
        t2 = t1 + max(np.linalg.norm(p2 - p1), 1e-9) ** alpha
        t3 = t2 + max(np.linalg.norm(p3 - p2), 1e-9) ** alpha
        t = np.linspace(t1, t2, samples, endpoint=False)[:, None]
        A1 = (t1 - t) / (t1 - t0) * p0 + (t - t0) / (t1 - t0) * p1
        A2 = (t2 - t) / (t2 - t1) * p1 + (t - t1) / (t2 - t1) * p2
        A3 = (t3 - t) / (t3 - t2) * p2 + (t - t2) / (t3 - t2) * p3
        B1 = (t2 - t) / (t2 - t0) * A1 + (t - t0) / (t2 - t0) * A2
        B2 = (t3 - t) / (t3 - t1) * A2 + (t - t1) / (t3 - t1) * A3
        out.append((t2 - t) / (t2 - t1) * B1 + (t - t1) / (t2 - t1) * B2)
    return np.vstack(out)


def root_joint(m: Morphology) -> int:
    """Proximal root: the proximal joint of the first body segment, else of the first segment."""
    for l in sorted(m.links.values(), key=lambda l: l.id):
        if l.type in BODY_TYPES:
            return l.from_joint
    return min(m.links.values(), key=lambda l: l.id).from_joint


def proximodistal_distances(m: Morphology) -> Dict[int, float]:
    adj: Dict[int, List[Tuple[int, float]]] = defaultdict(list)
    for l, a, b in m.pieces:
        d = math.dist(m.positions[a], m.positions[b])
        adj[a].append((b, d)); adj[b].append((a, d))
    root = root_joint(m)
    dist = {root: 0.0}
    q = deque([root])
    while q:
        a = q.popleft()
        for b, d in adj[a]:
            if b not in dist:
                dist[b] = dist[a] + d
                q.append(b)
    for j in m.joints:
        dist.setdefault(j, 0.0)
    return dist


def segment_polygon(m: Morphology, l: Link, a: int, b: int) -> Optional[np.ndarray]:
    ca, cb = m.positions[a], m.positions[b]
    theta = math.atan2(cb[1] - ca[1], cb[0] - ca[0]) if (ca != cb) else l.ang
    pa, pb = m.control[a], m.control[b]
    if not pa or not pb:
        return None
    a_right, a_left = _neighbours(pa, theta)                 # before = clockwise side = right of travel
    b_left, b_right = _neighbours(pb, theta + math.pi)       # at the far end the roles swap
    pts = [_xy(ca, *a_left), _xy(cb, *b_left)]
    if len(pb) == 3 and len(_edges_of(m, b)) == 1:           # terminal joint: rounded cap through the tip
        tip = [p for p in pb if p not in (b_left, b_right)][0]
        pts.append(_cap(cb, b_left, tip, b_right)[1:])
    pts.append(_xy(cb, *b_right))
    pts.append(_xy(ca, *a_right))
    if len(pa) == 3 and len(_edges_of(m, a)) == 1:
        tip = [p for p in pa if p not in (a_left, a_right)][0]
        pts.append(_cap(ca, a_right, tip, a_left)[1:])
    return np.vstack([np.atleast_2d(p) for p in pts])


def digit_polygon(base: Tuple[float, float], d: Digit, n: int = 16) -> np.ndarray:
    """A tapered digit of length Dist along Ang: straight sides, rounded tip."""
    L = max(d.dist, 1.0)
    w0, w1 = min(max(0.16 * L, 7.0), 22.0), min(max(0.10 * L, 4.0), 14.0)
    u = np.array([math.cos(d.ang), math.sin(d.ang)]); v = np.array([-u[1], u[0]])
    b = np.array(base)
    t = np.linspace(math.pi / 2, -math.pi / 2, n)          # from the +v side over the tip to the -v side
    tip = b + u * (L - w1) + np.outer(np.cos(t), u * w1) + np.outer(np.sin(t), v * w1)
    return np.vstack([b + v * w0, tip, b - v * w0])


# --------------------------------------------------------------------------
# Silhouette and morphometrics
# --------------------------------------------------------------------------
def all_polys(m: Morphology) -> List[np.ndarray]:
    body = [m.outline] if m.outline is not None else [p for _, p in m.segments]
    return body + [p for _, p, _, _ in m.digit_polys]


def extent(m: Morphology, pad: float = 15.0) -> Tuple[float, float, float, float]:
    pts = np.vstack(all_polys(m))
    return pts[:, 0].min() - pad, pts[:, 0].max() + pad, pts[:, 1].min() - pad, pts[:, 1].max() + pad


def silhouette(m: Morphology, res: int = 256, smooth: float = 0.0) -> Tuple[np.ndarray, Tuple[float, float, float, float]]:
    x0, x1, y0, y1 = extent(m)
    gx, gy = np.meshgrid(np.linspace(x0, x1, res), np.linspace(y0, y1, res))
    grid = np.column_stack([gx.ravel(), gy.ravel()])
    mask = np.zeros(len(grid), bool)
    for poly in all_polys(m):
        mask |= MplPath(poly).contains_points(grid)
    mask = mask.reshape(res, res)
    if smooth > 0:
        sigma = smooth / ((x1 - x0) / res)
        mask = gaussian_filter(mask.astype(float), sigma) >= 0.5
    return mask, (x0, x1, y0, y1)


def longest_path_length(m: Morphology) -> float:
    """Proximal-distal length: the longest simple path through the segment graph (sum of Dist)."""
    adj: Dict[int, List[Tuple[int, float]]] = defaultdict(list)
    for l in m.links.values():
        adj[l.from_joint].append((l.to_joint, l.dist)); adj[l.to_joint].append((l.from_joint, l.dist))
    best = 0.0

    def dfs(node, seen, acc):
        nonlocal best
        best = max(best, acc)
        for nb, d in adj[node]:
            if nb not in seen:
                dfs(nb, seen | {nb}, acc + d)
    for j in m.joints:
        dfs(j, {j}, 0.0)
    return best


def morphometrics(m: Morphology) -> Dict[str, float]:
    mask, (x0, x1, y0, y1) = silhouette(m)
    cell = ((x1 - x0) / mask.shape[1]) * ((y1 - y0) / mask.shape[0])
    ys, xs = np.where(mask)
    w = (xs.max() - xs.min() + 1) * (x1 - x0) / mask.shape[1] if len(xs) else 0.0
    h = (ys.max() - ys.min() + 1) * (y1 - y0) / mask.shape[0] if len(ys) else 0.0
    types = [l.type for l in m.links.values()]
    return {
        "id": m.id, "category": m.category, "name": m.name,
        "n_joints": len(m.joints), "n_segments": len(m.links),
        "n_segment_types": len(set(types)), "segment_types": "|".join(dict.fromkeys(types)),
        "n_limb_segments": sum(t not in BODY_TYPES for t in types),
        "n_digits": m.n_digits, "digit_bones": "|".join(str(d.n_bones) for l in m.links.values() for d in l.digits),
        "n_subjoints": sum(j.host_link is not None for j in m.joints.values()),
        "pd_length": round(longest_path_length(m), 1), "bbox_w": round(w, 1), "bbox_h": round(h, 1),
        "area": round(float(mask.sum() * cell), 0),
    }


def check_encoding(morphs: Dict[int, Morphology]) -> Tuple[int, int]:
    bad = n = 0
    for m in morphs.values():
        for jid, j in m.joints.items():
            k = len(_edges_of(m, jid))
            expected = 3 if k == 1 else k
            n += 1
            if len(j.params) != expected:
                bad += 1
    print(f"  encoding check: {n - bad}/{n} joints obey the 1→3 / k→k radii rule "
          f"(sub-joints count their host segment twice)")
    return n - bad, n


# --------------------------------------------------------------------------
# Drawing
# --------------------------------------------------------------------------
def segment_color(category: str, seg_type: str) -> str:
    """Flat colour per segment type (used for the non-tree fallback and legends)."""
    if seg_type in BODY_TYPES:
        return BODY_COLOR
    order = [t for t in SEGMENT_ORDER.get(category, []) if t not in BODY_TYPES]
    idx = order.index(seg_type) if seg_type in order else len(order)
    return PALETTE[idx % len(PALETTE)]


def colour_field(m: Morphology, res: int = 220) -> Tuple[np.ndarray, Tuple[float, float, float, float]]:
    """RGBA image of the Fig. 1 colour scheme over the limb's bounding box: the arthropod coxa green
    and the next segment (basis/trochanter) yellow, body segments grey, everything else a
    red -> white -> blue proximodistal gradient by path distance along the limb from the root.
    Clipped to the outline when drawn."""
    x0, x1, y0, y1 = extent(m)
    gx, gy = np.meshgrid(np.linspace(x0, x1, res), np.linspace(y0, y1, res))
    best_d = np.full(gx.shape, np.inf)
    pd = np.zeros(gx.shape)
    kind = np.zeros(gx.shape, dtype=np.int8)          # 0 limb gradient, 1 body, 2 coxa
    for l, a, b in m.pieces:
        ca, cb = np.array(m.positions[a]), np.array(m.positions[b])
        seg = cb - ca
        L2 = float(seg @ seg)
        if L2 < 1e-9:
            t = np.zeros(gx.shape)
            d = np.hypot(gx - ca[0], gy - ca[1])
        else:
            t = np.clip(((gx - ca[0]) * seg[0] + (gy - ca[1]) * seg[1]) / L2, 0, 1)
            d = np.hypot(gx - (ca[0] + t * seg[0]), gy - (ca[1] + t * seg[1]))
        better = d < best_d
        best_d = np.where(better, d, best_d)
        pd = np.where(better, m.pd_of_joint[a] + t * math.sqrt(L2), pd)
        kind = np.where(better, FLAT_KIND.get(l.type, 0), kind)
    limb = [(m.pd_of_joint[a], m.pd_of_joint[b]) for l, a, b in m.pieces if l.type not in FLAT_KIND]
    lo = min((min(p) for p in limb), default=0.0)
    hi = max((max(p) for p in limb), default=1.0)
    u = np.clip((pd - lo) / max(hi - lo, 1e-9), 0, 1)
    rgba = plt.get_cmap(GRADIENT_CMAP)(u)
    rgba[kind == 1] = matplotlib.colors.to_rgba(BODY_COLOR)
    rgba[kind == 2] = matplotlib.colors.to_rgba(COXA_GREEN)
    rgba[kind == 3] = matplotlib.colors.to_rgba(SECOND_YELLOW)
    return rgba, (x0, x1, y0, y1)


def _stadium(c0: np.ndarray, c1: np.ndarray, width: float, n: int = 12) -> np.ndarray:
    u = c1 - c0; L = np.linalg.norm(u)
    if L < 1e-6:
        u = np.array([1.0, 0.0]); L = 1.0
    u = u / L; v = np.array([-u[1], u[0]]); r = width / 2
    t = np.linspace(-math.pi / 2, math.pi / 2, n)
    end1 = c1 + np.outer(np.cos(t), u * r) + np.outer(np.sin(t), v * r)
    end0 = c0 + np.outer(np.cos(t + math.pi), u * r) + np.outer(np.sin(t + math.pi), v * r)
    return np.vstack([end1, end0])


def draw_bones(ax, m: Morphology, l: Link, zorder: float) -> None:
    """Long bones along the segment axis (1 or 2), or a cluster of small bones (hand/foot)."""
    if l.n_bones <= 0:
        return
    ca, cb = np.array(m.positions[l.from_joint]), np.array(m.positions[l.to_joint])
    u = cb - ca; L = np.linalg.norm(u)
    if L < 1e-6:
        return
    u /= L; v = np.array([-u[1], u[0]])
    half_w = max(min(np.mean([r for _, r in m.control[l.from_joint]]), np.mean([r for _, r in m.control[l.to_joint]])), 4.0)
    kw = dict(facecolor=BONE_FACE, edgecolor=BONE_EDGE, lw=0.8, zorder=zorder)
    if l.n_bones <= 2:
        inset = 0.16 * L
        width = min(0.55 * half_w, 0.22 * L)
        offsets = [0.0] if l.n_bones == 1 else [0.5 * half_w, -0.5 * half_w]
        for off in offsets:
            ax.add_patch(Polygon(_stadium(ca + u * inset + v * off, cb - u * inset + v * off, width), closed=True, **kw))
    else:
        n = l.n_bones
        cols = min(4, n); rows = math.ceil(n / cols)
        r = max(min(0.55 * half_w / rows, 0.28 * L / cols), 2.0)
        for i in range(n):
            ri, ci = divmod(i, cols)
            cx = ca + u * (L * (0.25 + 0.5 * (ri + 0.5) / rows)) + v * ((ci - (cols - 1) / 2) * 2.3 * r)
            ax.add_patch(Ellipse(cx, 2 * r, 2 * r * 0.85, **kw))


def draw_digit_bones(ax, d: Digit, base: Tuple[float, float], zorder: float) -> None:
    if d.n_bones <= 0:
        return
    u = np.array([math.cos(d.ang), math.sin(d.ang)]); v = np.array([-u[1], u[0]])
    L = max(d.dist, 1.0); n = d.n_bones
    gap = 0.08 * L / n
    seg = (0.84 * L - gap * (n - 1)) / n
    width = min(max(0.09 * L, 3.5), 11.0)
    start = np.array(base) + u * 0.08 * L
    for i in range(n):
        c0 = start + u * (i * (seg + gap)); c1 = c0 + u * seg
        ax.add_patch(Polygon(_stadium(c0, c1, width), closed=True, facecolor=DIGIT_BONE, edgecolor="none",
                             zorder=zorder))


def draw_morphology(ax, m: Morphology, show_bones: bool = True, outline_lw: float = 1.2,
                    gradient: bool = True) -> None:
    edge = "#1f1f1f"
    if m.outline is not None and gradient:
        patch = Polygon(m.outline, closed=True, facecolor="none", edgecolor=edge, lw=outline_lw, zorder=4)
        ax.add_patch(patch)
        rgba, (x0, x1, y0, y1) = colour_field(m)
        im = ax.imshow(rgba, extent=(x0, x1, y0, y1), origin="lower", interpolation="bilinear", zorder=2)
        im.set_clip_path(patch)
    elif m.outline is not None:
        ax.add_patch(Polygon(m.outline, closed=True, facecolor="#cfd6e6", edgecolor=edge, lw=outline_lw, zorder=2))
    else:                                                        # non-tree graph: flat per-segment polygons
        for l, poly in m.segments:
            ax.add_patch(Polygon(poly, closed=True, facecolor=segment_color(m.category, l.type), edgecolor=edge,
                                 lw=0.5, zorder=2))
    for d, poly, base, _ in m.digit_polys:
        ax.add_patch(Polygon(poly, closed=True, facecolor=DIGIT_BLUE, edgecolor=edge, lw=outline_lw * 0.6, zorder=3))
    if show_bones and m.category in DRAWS_BONES:
        for l in m.links.values():
            if l.type not in BODY_TYPES:
                draw_bones(ax, m, l, zorder=3.5)
        for d, poly, base, _ in m.digit_polys:
            draw_digit_bones(ax, d, base, zorder=3.6)
    ax.set_aspect("equal"); ax.axis("off")


def fit_axes(ax, m: Morphology, half: Optional[float] = None) -> None:
    x0, x1, y0, y1 = extent(m)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    h = half or max(x1 - x0, y1 - y0) / 2
    ax.set_xlim(cx - h, cx + h); ax.set_ylim(cy - h, cy + h)


def wrap_title(name: str, width: int = 28) -> str:
    words, lines, cur = name.split(), [], ""
    for w in words:
        if len(cur) + len(w) + 1 > width and cur:
            lines.append(cur); cur = w
        else:
            cur = (cur + " " + w).strip()
    lines.append(cur)
    return "\n".join(lines[:3])


def render_gallery(morphs: Dict[int, Morphology], ids: Sequence[int], out: Path, cols: int = 5) -> None:
    ids = [i for i in ids if i in morphs]
    rows = math.ceil(len(ids) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 3.9 * rows + 0.8))
    axes = np.atleast_1d(axes).ravel()
    for ax, mid in zip(axes, ids):
        m = morphs[mid]
        draw_morphology(ax, m); fit_axes(ax, m)
        ax.set_title(f"#{m.id} {m.category}\n{wrap_title(m.name)}", fontsize=8.5)
    for ax in axes[len(ids):]:
        ax.axis("off")
    fig.suptitle("LimbformDB wild-type limbs decoded from the graph formalism (Lobo et al. 2014; top row = Fig. 1)\n"
                 "colour proximal red → distal blue (coxa green, basis/trochanter yellow); joint radii give the border; "
                 "bones and digits as encoded", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95], h_pad=2.0)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  saved {out}")


def _paginate(items: List[int], per_page: int) -> List[List[int]]:
    return [items[i:i + per_page] for i in range(0, len(items), per_page)] or [[]]


def render_timeline(morphs: Dict[int, Morphology], first: Dict, out: Path, fps: int = 2, per_page: int = 12) -> None:
    years = sorted({y for y, _, _ in first.values()})
    by_year: Dict[int, List[int]] = defaultdict(list)
    for mid, (y, _, _) in first.items():
        if mid in morphs:
            by_year[y].append(mid)
    frames: List[Image.Image] = []

    def grab(fig):
        fig.canvas.draw()
        frames.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()))
        plt.close(fig)

    FIG = (16, 9)
    # intro: the five wild types of the paper's Fig. 1
    fig, axes = plt.subplots(1, 5, figsize=FIG)
    for ax, mid in zip(axes, FIG1_IDS):
        m = morphs[mid]; draw_morphology(ax, m); fit_axes(ax, m)
        ax.set_title(f"{m.category}: {wrap_title(m.name, 30)}", fontsize=9)
    fig.suptitle("LimbformDB: wild-type limbs of the five categories, decoded from the graph formalism\n"
                 "(the cartoon column of Lobo et al. 2014, Regeneration 1:37, Fig. 1)", fontsize=13)
    fig.text(0.5, 0.06, "Segments: Dist/Ang edges.  Joints: border radii on the bisectors (three at a terminal joint).  "
             "Colour: proximodistal red → blue; coxa green, basis/trochanter yellow.  Bones and digits as encoded.",
             ha="center", fontsize=9.5, color="#444444")
    fig.tight_layout(rect=[0, 0.08, 1, 0.9]); grab(fig)

    cumulative = 0
    for y in years:
        ids = sorted(by_year[y])
        cumulative += len(ids)
        for page_i, page in enumerate(_paginate(ids, per_page)):
            cols = 6; rows = 2
            fig, axes = plt.subplots(rows, cols, figsize=FIG)
            axes = np.atleast_1d(axes).ravel()
            for ax, mid in zip(axes, page):
                m = morphs[mid]; draw_morphology(ax, m, outline_lw=1.0); fit_axes(ax, m)
                ax.set_title(f"#{m.id} {m.category[:4]}. {wrap_title(m.name, 26)}", fontsize=8)
            for ax in axes[len(page):]:
                ax.axis("off")
            more = f"  (page {page_i + 1}/{math.ceil(len(ids) / per_page)})" if len(ids) > per_page else ""
            fig.suptitle(f"{y}: {len(ids)} limb morpholog{'y' if len(ids) == 1 else 'ies'} first reported"
                         f"{more}   —   {cumulative} of {len(first)} dated morphologies so far", fontsize=13)
            fig.tight_layout(rect=[0, 0, 1, 0.93]); grab(fig)

    # final: morphospace grids per category (18 x 9 cells per frame)
    cols, rows = 18, 9
    for cat in ["Salamander", "Frog", "Crustacean", "Insect", "Arachnid"]:
        ids = [mid for mid, m in morphs.items() if m.category == cat]
        pages = _paginate(ids, cols * rows)
        for page_i, page in enumerate(pages):
            fig, axes = plt.subplots(rows, cols, figsize=FIG)
            axes = np.atleast_1d(axes).ravel()
            for ax, mid in zip(axes, page):
                m = morphs[mid]; draw_morphology(ax, m, show_bones=False, outline_lw=0.5); fit_axes(ax, m)
            for ax in axes[len(page):]:
                ax.axis("off")
            more = f" ({page_i + 1}/{len(pages)})" if len(pages) > 1 else ""
            fig.suptitle(f"Morphospace: all {len(ids)} {cat.lower()} limb morphologies in LimbformDB 2.1.0{more}  "
                         f"(each fitted to its cell)", fontsize=13)
            fig.tight_layout(rect=[0, 0, 1, 0.95]); grab(fig)
    n_grid = sum(len(_paginate([0] * sum(m.category == c for m in morphs.values()), cols * rows))
                 for c in ["Salamander", "Frog", "Crustacean", "Insect", "Arachnid"])

    sizes = {f.size for f in frames}
    if len(sizes) > 1:
        W = max(s[0] for s in sizes); H = max(s[1] for s in sizes)
        frames = [_pad(f, W, H) for f in frames]
    durations = [int(1000 / fps)] * len(frames)
    durations[0] = 4000
    for i in range(len(frames) - n_grid, len(frames)):
        durations[i] = 4000
    frames[0].save(out, save_all=True, append_images=frames[1:], duration=durations, loop=0, optimize=False)
    print(f"  saved {out}  ({len(frames)} frames, {len(years)} years, {len(first)} dated morphologies)")


def _pad(im: Image.Image, W: int, H: int) -> Image.Image:
    bg = Image.new("RGB", (W, H), "white")
    bg.paste(im, ((W - im.width) // 2, (H - im.height) // 2))
    return bg


def write_csv(morphs: Dict[int, Morphology], first: Dict, out: Path) -> None:
    import csv
    rows = []
    for m in morphs.values():
        r = morphometrics(m)
        y = first.get(m.id)
        r["first_year"] = y[0] if y else ""
        r["species"] = y[1] if y else ""
        rows.append(r)
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"  saved {out}  ({len(rows)} rows)")


# --------------------------------------------------------------------------
def main(argv: Optional[Sequence[str]] = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("db", nargs="?", help="path to limbformDB_2.1.0.ldb (or set LIMBFORM_DB_PATH)")
    p.add_argument("--download", action="store_true", help="download LimbformDB 2.1.0 into .limbform_cache/")
    p.add_argument("--out-dir", default=".")
    p.add_argument("--ids", type=int, nargs="*", help="morphology ids for the gallery")
    p.add_argument("--csv", help="write per-morphology measurements to this CSV")
    p.add_argument("--no-anim", action="store_true")
    p.add_argument("--no-gallery", action="store_true")
    p.add_argument("--fps", type=int, default=2)
    p.add_argument("--per-page", type=int, default=12)
    args = p.parse_args(argv)

    db = resolve_db_path(args.db, args.download)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    print(f"LimbformDB: {db}")
    morphs = load_morphologies(db)
    first = load_first_report_year(db)
    cats = defaultdict(int)
    for m in morphs.values():
        cats[m.category] += 1
    print(f"  {len(morphs)} morphologies: " + ", ".join(f"{c} {n}" for c, n in cats.items()))
    check_encoding(morphs)
    for mid in FIG1_IDS:
        mm = morphometrics(morphs[mid])
        print(f"  #{mid} {mm['category']:10s} {mm['name'][:40]:40s} segments {mm['n_segments']}, digits {mm['n_digits']}, "
              f"PD length {mm['pd_length']}, bbox {mm['bbox_w']}×{mm['bbox_h']}")

    if not args.no_gallery:
        ids = args.ids or GALLERY_IDS
        render_gallery(morphs, ids, out_dir / "limbform_wildtype_gallery.png")
    if args.csv:
        write_csv(morphs, first, Path(args.csv))
    if not args.no_anim:
        render_timeline(morphs, first, out_dir / "limbform_morphospace_timeline.gif", fps=args.fps,
                        per_page=args.per_page)


if __name__ == "__main__":
    main()

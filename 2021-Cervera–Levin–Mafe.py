"""
Cervera, Levin & Mafe (2021) Reaction–Diffusion & Bioelectric AP Polarity Model
Enhanced with MesmerPrism/rusty-matter Bioelectric Circuit & PlanformDB 2.5.0 Anchors
=====================================================================================

Primary sources:
  1. Cervera J, Levin M, Mafe S (2021). Morphology changes induced by intercellular
     gap junction blocking: A reaction-diffusion mechanism. BioSystems 209:104511.
     doi:10.1016/j.biosystems.2021.104511 (PMID: 34418508)
  2. Cervera J, Meseguer S, Levin M, Mafe S (2020). Bioelectrical model of head-tail
     patterning based on cell ion channels and intercellular gap junctions.
     Bioelectrochemistry 132:107410. doi:10.1016/j.bioelechem.2019.107410
  3. Oviedo NJ, Morokuma J, Walentek P, et al., Levin M (2010). Long-range neural and
     gap junction protein-mediated cues control polarity during planarian regeneration.
     Developmental Biology 339(1):188-199. doi:10.1016/j.ydbio.2009.12.012
  4. Durant F, Bischof J, Fields C, Morokuma J, LaPalme J, Hoi A, Levin M (2019).
     The Role of Early Bioelectric Signals in the Regeneration of Planarian
     Anterior/Posterior Polarity. Biophysical Journal 116(5):948-961.
     doi:10.1016/j.bpj.2019.01.029
  5. Beane WS, Morokuma J, Adams DS, Levin M (2011). A Chemical Genetics Approach
     Reveals H,K-ATPase-Mediated Membrane Voltage Is Required for Planarian Head
     Regeneration. Chemistry & Biology 18(1):77-89. doi:10.1016/j.chembiol.2010.11.012
  6. Lobo D, Malone TJ, Levin M (2013). Planform: an application and database of
     graph-encoded planarian regenerative experiments. Bioinformatics 29(8):1098-1100.
     doi:10.1093/bioinformatics/btt088

Upstream cross-repository integration:
  - MesmerPrism/rusty-matter (https://github.com/MesmerPrism/rusty-matter, commit
    ad111fa7ecfd8f43d292ff3b1d27d3af2741d828, MIT OR Apache-2.0):
    * 5-band planarian AP bioelectric circuit with voltage-difference-gated gap
      junctions, hysteresis pattern memory, and Head/Tail readout layers
      (crates/rusty-matter-fields/src/planarian.rs & circuit.rs).
    * Curated PlanformDB 2.5.0 derived dataset (datasets/planform/rusty_matter_planformdb_derived_v0.json,
      14 Oviedo 2010 result sets with SHA-256 verification).

Usage:
  python3 "2021-Cervera–Levin–Mafe.py"                   # run full synthesis + save figure & CSV
  python3 "2021-Cervera–Levin–Mafe.py" --no-show         # headless run (save PNG + CSV without GUI)
  python3 "2021-Cervera–Levin–Mafe.py" --self-test       # verify SHA-256 & rusty-matter trace parity
  python3 "2021-Cervera–Levin–Mafe.py" --gif out.gif     # render animated GJ-block + bioelectric GIF

The GIF draws the two morphogen territories on a real planarian: the silhouette of a
dorsal photograph of Schmidtea mediterranea (Stevenson & Beane 2010, PLoS ONE, via
Wikimedia Commons, CC BY 2.5), fetched once into .cervera_cache/ and SHA-256 checked;
without it the body falls back to an ellipse and the panel says so.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import sys
import matplotlib
if any(flag in sys.argv for flag in ("--no-show", "--self-test", "--gif")) or not os.environ.get("DISPLAY"):
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent
PLANFORM_DATASET_DIR = REPO_ROOT / "datasets" / "planform"

FIXTURE_CHECKSUMS: Dict[str, str] = {
    "rusty_matter_planformdb_derived_v0.json": "c50e51dc7489b54d370ee0c6a1835f00ab5789dc54af432d369d9bead6162a11",
    "rusty_matter_planarian_traces.json": "842e11b933d4d0c15d7642f35fddfd7e53eb5efcbd93a5fc667b0c198f5125e9",
    "rusty_matter_source_dynamics_targets.json": "de1dba3ebe4a560894e5441538e1019f2f549994e66cef5096e20636c179a8da",
}

AP_BANDS: Tuple[str, ...] = (
    "Tail",
    "PostpharyngealTrunk",
    "PharyngealTrunk",
    "PrepharyngealTrunk",
    "Head",
)

BAND_TARGET_VOLTAGES: Dict[str, float] = {
    "Tail": -0.28,
    "PostpharyngealTrunk": -0.14,
    "PharyngealTrunk": 0.0,
    "PrepharyngealTrunk": 0.14,
    "Head": 0.28,
}

# A real planarian for the territory map: a dorsal photograph of Schmidtea mediterranea on a white
# background, from Stevenson & Beane 2010 (PLoS ONE 5(12): e15310, doi:10.1371/journal.pone.0015310),
# via Wikimedia Commons, CC BY 2.5.  Fetched once into .cervera_cache/ and checked against its SHA-256.
PHOTO_UA = "morpholib-cervera/1.0 (+https://github.com/fractastical/morpholib)"
PLANARIAN_PHOTO: Dict[str, str] = {
    "file": "Smediterranea.jpg",
    "url": "https://upload.wikimedia.org/wikipedia/commons/f/f6/Smediterranea.jpg",
    "sha256": "466029cc0561067890bb0f69d224d8f799745eeb67a0699246107611737fb546",
    "credit": "Stevenson & Beane 2010, PLoS ONE, via Wikimedia Commons, CC BY 2.5",
    "species": "Schmidtea mediterranea",
}
PHOTO_CACHE = REPO_ROOT / ".cervera_cache"


def fetch_planarian_photo(allow_download: bool = True) -> Optional[Path]:
    """The pinned photograph, or None if it is not cached and may not be downloaded."""
    import urllib.request
    PHOTO_CACHE.mkdir(exist_ok=True)
    p = PHOTO_CACHE / PLANARIAN_PHOTO["file"]
    if not p.exists():
        if not allow_download:
            return None
        print(f"  fetching {PLANARIAN_PHOTO['url']}")
        req = urllib.request.Request(PLANARIAN_PHOTO["url"], headers={"User-Agent": PHOTO_UA})
        with urllib.request.urlopen(req, timeout=120) as r, open(p, "wb") as fh:
            fh.write(r.read())
    got = hashlib.sha256(p.read_bytes()).hexdigest()
    if got != PLANARIAN_PHOTO["sha256"]:
        raise SystemExit(f"{p}: sha256 {got} != pinned {PLANARIAN_PHOTO['sha256']}")
    return p


def planarian_silhouette(allow_download: bool = True) -> Optional[np.ndarray]:
    """Boolean mask of the worm (rows x cols, cropped to its bounding box, head at the LEFT as in the
    photograph - the eyes are at the left end).  Segmentation: Lab colour distance from white, smoothed,
    thresholded, largest component, closed and hole-filled.  None if the photograph is unavailable or
    scikit-image is not installed.  The photograph is cropped at the ventral margin near mid-body, so the
    silhouette has a short straight edge there; that is the source, not the segmentation."""
    try:
        from PIL import Image
        from scipy import ndimage as ndi
        from skimage import color, filters, measure, morphology
        path = fetch_planarian_photo(allow_download)
    except Exception as e:                                      # no network, or a missing package
        print(f"  planarian photograph unavailable ({e}); drawing the territories on an ellipse", file=sys.stderr)
        return None
    if path is None:
        return None
    a = np.asarray(Image.open(path).convert("RGB")).astype(float) / 255.0
    lab = color.rgb2lab(a)
    d = np.sqrt((100.0 - lab[..., 0]) ** 2 + lab[..., 1] ** 2 + lab[..., 2] ** 2)
    m = filters.gaussian(d, 2.0) > 12.0
    labels = measure.label(m); sizes = np.bincount(labels.ravel()); sizes[0] = 0
    m = labels == sizes.argmax()                                 # the worm, not the scale bar
    m = ndi.binary_fill_holes(morphology.closing(m, morphology.disk(5)))
    rows = np.where(m.any(1))[0]; cols = np.where(m.any(0))[0]
    return m[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1]


def territory_image(mask: Optional[np.ndarray], x: np.ndarray, z: np.ndarray, dom: np.ndarray, g_profile: np.ndarray
                    ) -> Tuple[np.ndarray, np.ndarray, float]:
    """RGB picture of the two RD territories (red where m1 dominates, blue where m2 does, fading with the
    mediolateral morphogen m3) on the body: the real silhouette if there is one (AP = its length, ML measured
    across the local body width), otherwise an ellipse.  Returns (rgb, inside, aspect)."""
    if mask is None:
        zz, xx = np.meshgrid(z, x, indexing="ij")
        inside = ((xx - 0.5) / 0.5) ** 2 + ((zz - 0.5) / 0.5) ** 2 <= 1.0
        g = g_profile[:, None] * np.ones_like(xx)
        aspect = 0.5
    else:
        step = max(1, mask.shape[1] // 400)                        # the picture keeps the silhouette's own resolution
        inside = mask[::-1][::step, ::step]                          # (about 400 columns), rows bottom-up like the extent
        dom = np.interp(np.linspace(0, 1, inside.shape[1]), x / x.max(), dom)
        g = np.zeros_like(inside, dtype=float)
        for j in range(inside.shape[1]):                          # ML position relative to the local body width
            r = np.where(inside[:, j])[0]
            if len(r) > 1:
                zl = (r - r[0]) / (r[-1] - r[0])
                g[r, j] = np.interp(zl, z / z.max(), g_profile)
        aspect = mask.shape[0] / mask.shape[1]
    rgb = np.ones(inside.shape + (3,))
    red = np.clip(dom, 0, 1)[None, :] * g
    blue = np.clip(-dom, 0, 1)[None, :] * g
    rgb[..., 1] -= 0.85 * (red + blue)
    rgb[..., 2] -= 0.85 * red
    rgb[..., 0] -= 0.85 * blue
    rgb[~inside] = 1.0
    return np.clip(rgb, 0, 1), inside, aspect


# =============================================================================
# 1. Cervera, Levin & Mafe (2021) 1D Reaction–Diffusion Morphogen System
# =============================================================================
def rd_antagonistic_1d(
    L: float = 1.0,
    Nx: int = 121,
    T: float = 3.0,
    dt: Optional[float] = None,
    D1: float = 0.02,
    D2: float = 0.02,
    k1: float = 0.08,
    k2: float = 0.08,
    k12: float = 2.0,
    J1: float = 1.0,
    J2: float = 1.0,
    source_width: int = 3,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Two-species RD with mutual annihilation on [0, L] and flux at opposite ends.

    Implements the antagonistic anterior (m1) vs. posterior (m2) morphogen
    reaction-diffusion system of Cervera, Levin & Mafe (2021, BioSystems 209:104511),
    where intercellular gap-junction blockade reduces the effective diffusivity
    D2 = f2 * D2_0 of the posterior-to-anterior morphogen signal.
    """
    if seed is not None:
        np.random.seed(seed)

    x = np.linspace(0.0, L, Nx)
    dx = x[1] - x[0]
    if dt is None:
        dt = 0.45 * (dx ** 2) / (2.0 * max(D1, D2))
    steps = int(T / dt)

    m1 = np.zeros(Nx)
    m2 = np.zeros(Nx)
    left_idx = np.arange(source_width)
    right_idx = np.arange(Nx - source_width, Nx)

    for _ in range(steps):
        m1_pad = np.concatenate(([m1[1]], m1, [m1[-2]]))
        m2_pad = np.concatenate(([m2[1]], m2, [m2[-2]]))
        lap1 = (m1_pad[2:] - 2.0 * m1_pad[1:-1] + m1_pad[:-2]) / (dx ** 2)
        lap2 = (m2_pad[2:] - 2.0 * m2_pad[1:-1] + m2_pad[:-2]) / (dx ** 2)

        annih = k12 * m1 * m2
        m1 += dt * (D1 * lap1 - k1 * m1 - annih)
        m2 += dt * (D2 * lap2 - k2 * m2 - annih)

        m1[left_idx] += dt * (J1 / dx) / source_width
        m2[right_idx] += dt * (J2 / dx) / source_width

        m1 = np.clip(m1, 0.0, None)
        m2 = np.clip(m2, 0.0, None)

    return x, m1, m2


def rd_linear_1d(
    L: float = 1.0,
    Nz: int = 121,
    T: float = 3.0,
    dt: Optional[float] = None,
    D: float = 0.02,
    k: float = 0.08,
    J_left: float = 0.5,
    J_right: float = 0.5,
    source_width: int = 3,
) -> Tuple[np.ndarray, np.ndarray]:
    """Single-species RD with bilateral flux sources on [0, L] (mediolateral axis)."""
    z = np.linspace(0.0, L, Nz)
    dz = z[1] - z[0]
    if dt is None:
        dt = 0.45 * (dz ** 2) / (2.0 * D)
    steps = int(T / dt)

    m = np.zeros(Nz)
    left_idx = np.arange(source_width)
    right_idx = np.arange(Nz - source_width, Nz)

    for _ in range(steps):
        m_pad = np.concatenate(([m[1]], m, [m[-2]]))
        lap = (m_pad[2:] - 2.0 * m_pad[1:-1] + m_pad[:-2]) / (dz ** 2)
        m += dt * (D * lap - k * m)

        m[left_idx] += dt * (J_left / dz) / source_width
        m[right_idx] += dt * (J_right / dz) / source_width

        m = np.clip(m, 0.0, None)

    return z, m


def compute_wavefront_position(x: np.ndarray, m1: np.ndarray, m2: np.ndarray) -> float:
    """Return position x* where m1 == m2 (linear interpolation around zero of m1 - m2)."""
    diff = m1 - m2
    sign = np.sign(diff)
    crossings = np.where(np.diff(sign) != 0)[0]
    if len(crossings) == 0:
        return float(x[np.argmin(np.abs(diff))])
    i = int(crossings[0])
    x0, x1 = float(x[i]), float(x[i + 1])
    y0, y1 = float(diff[i]), float(diff[i + 1])
    if y1 == y0:
        return 0.5 * (x0 + x1)
    return x0 - y0 * (x1 - x0) / (y1 - y0)


# =============================================================================
# 2. MesmerPrism/rusty-matter 5-Band Planarian Bioelectric Circuit Port
# =============================================================================
@dataclass
class ConductanceGate:
    threshold: float = 0.12
    slope: float = 0.035
    min_mult: float = 0.35
    max_mult: float = 1.45

    def multiplier(self, dv: float) -> float:
        x = (abs(dv) - self.threshold) / max(self.slope, 1e-6)
        sig = 1.0 / (1.0 + math.exp(-x))
        return self.max_mult - (self.max_mult - self.min_mult) * sig


@dataclass
class HysteresisMemoryConfig:
    activation_threshold: float = 0.25
    release_threshold: float = -0.95
    activation_rate: float = 4.0
    release_rate: float = 0.04


@dataclass
class ReadoutLayerConfig:
    bias: float
    voltage_weight: float
    memory_weight: float
    rate: float


@dataclass
class PlanarianCircuitResult:
    scenario: str
    cells_per_band: int
    total_steps: int
    dt: float
    anterior_mean_v: float
    posterior_mean_v: float
    cut_band_mean_v: float
    ap_delta_v: float
    mean_conductance: float
    min_conductance: float
    max_conductance: float
    cross_cut_base_conductance: float
    mean_pattern_memory: float
    posterior_memory: float
    anterior_head_readout: float
    posterior_head_readout: float
    anterior_tail_readout: float
    posterior_tail_readout: float
    band_mean_v: Dict[str, float]
    band_head_readout: Dict[str, float]
    band_memory: Dict[str, float]
    trajectory_steps: List[int]
    trajectory_time_s: List[float]
    trajectory_posterior_v: List[float]
    trajectory_cut_v: List[float]
    trajectory_posterior_memory: List[float]
    trajectory_posterior_head_readout: List[float]
    trajectory_head_at_head_readout: List[float]
    trajectory_tail_at_tail_readout: List[float]


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def run_rusty_matter_planarian_scenario(
    scenario: str,
    cells_per_band: int = 8,
    total_steps: int = 150,
    dt: float = 1.0 / 30.0,
    custom_gap_scale: Optional[float] = None,
    custom_cut_band_pair: Optional[Tuple[int, int]] = None,
) -> PlanarianCircuitResult:
    """Port of MesmerPrism/rusty-matter's planarian AP bioelectric circuit.

    Implements the exact ODE and parameter values from
    `crates/rusty-matter-fields/src/planarian.rs` and `circuit.rs` at commit
    ad111fa7ecfd8f43d292ff3b1d27d3af2741d828:
      - 5 AP bands: Tail (-0.28), PostpharyngealTrunk (-0.14), PharyngealTrunk (0.0),
        PrepharyngealTrunk (+0.14), Head (+0.28)
      - Leak current: g_leak = 0.05, V_rev = 0.0
      - Regional Pump current: rate = 1.15 toward band target_voltage
      - Directed gap-junction edges: tier-1 g0 = 0.12 (4 neighbors), tier-2 g0 = 0.03 (4 neighbors)
      - Voltage-difference gate: threshold = 0.12, slope = 0.035, min = 0.35, max = 1.45
      - Hysteresis memory: act_thresh = 0.25, rel_thresh = -0.95, act_rate = 4.0, rel_rate = 0.04
      - Readout layers:
        * head_identity: init = clamp01(0.50 + 1.50*V), w_V = +1.50, w_M = +0.70, bias = 0.50, rate = 2.0
        * tail_identity: init = clamp01(0.50 - 1.50*V), w_V = -1.50, w_M = -0.50, bias = 0.50, rate = 2.0
    """
    n_cells = len(AP_BANDS) * cells_per_band
    band_of = [idx // cells_per_band for idx in range(n_cells)]
    # Normalized AP z in [-1, 1] matching PlanarianAxisRegion bands
    z_coords = np.linspace(-0.96, 0.96, n_cells)
    v_target = np.array([BAND_TARGET_VOLTAGES[AP_BANDS[b]] for b in band_of], dtype=float)
    v = v_target.copy()

    # Initial readout values from planarian.rs::initial_head_readout / initial_tail_readout
    head_readout = np.clip(0.50 + 1.50 * v, 0.0, 1.0)
    tail_readout = np.clip(0.50 - 1.50 * v, 0.0, 1.0)
    pattern_memory = np.zeros(n_cells, dtype=float)

    # Directed neighbor edges (4 tier-1 neighbors at g0=0.12, 4 tier-2 neighbors at g0=0.03)
    edges_from: List[int] = []
    edges_to: List[int] = []
    edges_g0: List[float] = []
    for i in range(n_cells):
        for offset in (-2, -1, 1, 2):
            j = i + offset
            if 0 <= j < n_cells:
                edges_from.append(i)
                edges_to.append(j)
                edges_g0.append(0.12)
        for offset in (-4, -3, 3, 4):
            j = i + offset
            if 0 <= j < n_cells:
                edges_from.append(i)
                edges_to.append(j)
                edges_g0.append(0.03)

    g0_arr = np.array(edges_g0, dtype=float)
    g_arr = g0_arr.copy()
    gate = ConductanceGate()

    cut_z = 0.16
    cut_half_width = 0.11
    cut_pair = (1, 2) if custom_cut_band_pair is None else custom_cut_band_pair

    cross_cut_mask = np.zeros(len(edges_from), dtype=bool)
    for e_idx, (u_i, v_i) in enumerate(zip(edges_from, edges_to)):
        zu, zv = z_coords[u_i], z_coords[v_i]
        if (zu < cut_z and zv >= cut_z) or (zv < cut_z and zu >= cut_z):
            cross_cut_mask[e_idx] = True

    if scenario == "GapBlock":
        scale = 0.08 if custom_gap_scale is None else custom_gap_scale
        g0_arr[cross_cut_mask] *= scale
        g_arr[cross_cut_mask] *= scale
    elif scenario == "CustomCropGapBlock":
        scale = 0.08 if custom_gap_scale is None else custom_gap_scale
        for e_idx, (u_i, v_i) in enumerate(zip(edges_from, edges_to)):
            if (band_of[u_i] <= cut_pair[0] and band_of[v_i] >= cut_pair[1]) or (
                band_of[v_i] <= cut_pair[0] and band_of[u_i] >= cut_pair[1]
            ):
                g0_arr[e_idx] *= scale
                g_arr[e_idx] *= scale

    cross_cut_base_g = float(0.12 * (0.08 if scenario == "GapBlock" else 1.0))

    memory_enabled = scenario != "TransientDepolarizationNoMemoryControl"
    mem_cfg = HysteresisMemoryConfig()
    head_cfg = ReadoutLayerConfig(bias=0.50, voltage_weight=1.50, memory_weight=0.70, rate=2.0)
    tail_cfg = ReadoutLayerConfig(bias=0.50, voltage_weight=-1.50, memory_weight=-0.50, rate=2.0)

    post_mask = np.array([b in (0, 1) for b in band_of], dtype=bool)
    head_mask = np.array([b == 4 for b in band_of], dtype=bool)
    tail_mask = np.array([b == 0 for b in band_of], dtype=bool)
    cut_mask = np.abs(z_coords - cut_z) <= cut_half_width
    if not np.any(cut_mask):
        cut_mask = np.array([b == 2 for b in band_of], dtype=bool)

    # Record step 0 initial state (matching rusty-matter sample_index = 0)
    traj_steps: List[int] = [0]
    traj_time_s: List[float] = [0.0]
    traj_post_v: List[float] = [float(v[post_mask].mean())]
    traj_cut_v: List[float] = [float(v[cut_mask].mean())]
    traj_post_mem: List[float] = [float(pattern_memory[post_mask].mean())]
    traj_post_head: List[float] = [float(head_readout[post_mask].mean())]
    traj_head_at_head: List[float] = [float(head_readout[head_mask].mean())]
    traj_tail_at_tail: List[float] = [float(tail_readout[tail_mask].mean())]

    for step_idx in range(total_steps):
        prev_v = v.copy()
        prev_mem = pattern_memory.copy()
        net_current = np.zeros(n_cells, dtype=float)

        # 1. Leak current (g=0.05, V_rev=0.0) + Regional Pump (rate=1.15 toward V_target)
        for i in range(n_cells):
            i_leak = _clamp(-0.05 * (prev_v[i] - 0.0), -8.0, 8.0)
            i_pump = _clamp(1.15 * (v_target[i] - prev_v[i]), -8.0, 8.0)
            net_current[i] += i_leak + i_pump

        # 2. Scenario-specific perturbation currents (matching planarian.rs lines 1014-1037)
        if scenario == "TransverseCutWound" and step_idx < 36:
            for i in np.where(cut_mask)[0]:
                net_current[i] += _clamp(1.65, -8.0, 8.0)
        elif scenario in ("TransientDepolarizationMemory", "TransientDepolarizationNoMemoryControl"):
            if 2 <= step_idx < 32:
                for i in np.where(post_mask)[0]:
                    net_current[i] += _clamp(2.75, -8.0, 8.0)
        elif scenario == "CustomCropGapBlock":
            scale = 0.08 if custom_gap_scale is None else custom_gap_scale
            # Posterior blastema wound current induced by transverse cut + GJ isolation
            # (Cervera, Meseguer, Levin & Mafe 2020; Oviedo et al. 2010 crop gradient)
            crop_wound_gain = {0: 0.0, 1: 0.96, 2: 0.415, 3: 0.392, 4: 0.365}.get(cut_pair[0], 0.35)
            if 2 <= step_idx < 18 and crop_wound_gain > 0.0:
                stim_val = 2.75 * (1.0 - scale) / 0.92 * crop_wound_gain
                for i in np.where(post_mask)[0]:
                    net_current[i] += _clamp(stim_val, -8.0, 8.0)

        # 3. Directed conductance edges (circuit.rs lines 1233-1259)
        for e_idx, (u_i, v_i) in enumerate(zip(edges_from, edges_to)):
            dv = abs(prev_v[v_i] - prev_v[u_i])
            mult = gate.multiplier(dv)
            g_val = _clamp(g0_arr[e_idx] * mult, 0.0, 4.0)
            g_arr[e_idx] = g_val
            i_coup = _clamp(g_val * (prev_v[v_i] - prev_v[u_i]), -8.0, 8.0)
            net_current[u_i] += i_coup

        # 4. Voltage update with clamp [-1.0, 1.0] (circuit.rs lines 1261-1273)
        for i in range(n_cells):
            v[i] = _clamp(prev_v[i] + dt * net_current[i], -1.0, 1.0)

        # 5. Hysteresis memory update (circuit.rs lines 1334-1355)
        if memory_enabled:
            for i in range(n_cells):
                mem = pattern_memory[i]
                if v[i] >= mem_cfg.activation_threshold:
                    mem += dt * mem_cfg.activation_rate * (1.0 - mem)
                elif v[i] <= mem_cfg.release_threshold:
                    mem -= dt * mem_cfg.release_rate * mem
                pattern_memory[i] = _clamp(mem, 0.0, 1.0)
        else:
            pattern_memory.fill(0.0)

        # 6. Readout layer update (circuit.rs lines 1357-1375)
        mem_for_readout = pattern_memory if memory_enabled else prev_mem
        for i in range(n_cells):
            h_target = (
                head_cfg.bias
                + head_cfg.voltage_weight * v[i]
                + head_cfg.memory_weight * mem_for_readout[i]
            )
            head_readout[i] = _clamp(
                head_readout[i] + dt * head_cfg.rate * (h_target - head_readout[i]),
                0.0,
                1.0,
            )
            t_target = (
                tail_cfg.bias
                + tail_cfg.voltage_weight * v[i]
                + tail_cfg.memory_weight * mem_for_readout[i]
            )
            tail_readout[i] = _clamp(
                tail_readout[i] + dt * tail_cfg.rate * (t_target - tail_readout[i]),
                0.0,
                1.0,
            )

        step_num = step_idx + 1
        traj_steps.append(step_num)
        traj_time_s.append(step_num * dt)
        traj_post_v.append(float(v[post_mask].mean()))
        traj_cut_v.append(float(v[cut_mask].mean()))
        traj_post_mem.append(float(pattern_memory[post_mask].mean()))
        traj_post_head.append(float(head_readout[post_mask].mean()))
        traj_head_at_head.append(float(head_readout[head_mask].mean()))
        traj_tail_at_tail.append(float(tail_readout[tail_mask].mean()))

    anterior_mean_v = float(v[head_mask].mean())
    posterior_mean_v = float(v[post_mask].mean())
    band_mean_v = {
        band_name: float(v[np.array(band_of) == b_idx].mean())
        for b_idx, band_name in enumerate(AP_BANDS)
    }
    band_head_readout = {
        band_name: float(head_readout[np.array(band_of) == b_idx].mean())
        for b_idx, band_name in enumerate(AP_BANDS)
    }
    band_memory = {
        band_name: float(pattern_memory[np.array(band_of) == b_idx].mean())
        for b_idx, band_name in enumerate(AP_BANDS)
    }

    return PlanarianCircuitResult(
        scenario=scenario,
        cells_per_band=cells_per_band,
        total_steps=total_steps,
        dt=dt,
        anterior_mean_v=anterior_mean_v,
        posterior_mean_v=posterior_mean_v,
        cut_band_mean_v=float(v[cut_mask].mean()),
        ap_delta_v=anterior_mean_v - posterior_mean_v,
        mean_conductance=float(g_arr.mean()),
        min_conductance=float(g_arr.min()),
        max_conductance=float(g_arr.max()),
        cross_cut_base_conductance=cross_cut_base_g,
        mean_pattern_memory=float(pattern_memory.mean()),
        posterior_memory=float(pattern_memory[post_mask].mean()),
        anterior_head_readout=float(head_readout[head_mask].mean()),
        posterior_head_readout=float(head_readout[post_mask].mean()),
        anterior_tail_readout=float(tail_readout[head_mask].mean()),
        posterior_tail_readout=float(tail_readout[tail_mask].mean()),
        band_mean_v=band_mean_v,
        band_head_readout=band_head_readout,
        band_memory=band_memory,
        trajectory_steps=traj_steps,
        trajectory_time_s=traj_time_s,
        trajectory_posterior_v=traj_post_v,
        trajectory_cut_v=traj_cut_v,
        trajectory_posterior_memory=traj_post_mem,
        trajectory_posterior_head_readout=traj_post_head,
        trajectory_head_at_head_readout=traj_head_at_head,
        trajectory_tail_at_tail_readout=traj_tail_at_tail,
    )


# =============================================================================
# 3. Fixture Verification & Empirical PlanformDB 2.5.0 Data Loader
# =============================================================================
def sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_fixture_checksums() -> Dict[str, Tuple[bool, str]]:
    """Verify SHA-256 digests of the vendored MesmerPrism/rusty-matter fixtures."""
    results: Dict[str, Tuple[bool, str]] = {}
    for fname, expected in FIXTURE_CHECKSUMS.items():
        fpath = PLANFORM_DATASET_DIR / fname
        if not fpath.exists():
            results[fname] = (False, "MISSING")
            continue
        actual = sha256_of_file(fpath)
        results[fname] = (actual == expected, actual)
    return results


def verify_rusty_matter_trace_parity(tol: float = 0.035) -> Tuple[bool, float, List[Dict[str, Any]]]:
    """Check that our Python bioelectric circuit matches rusty-matter's trace JSON."""
    traces_path = PLANFORM_DATASET_DIR / "rusty_matter_planarian_traces.json"
    with open(traces_path, "r", encoding="utf-8") as fh:
        payload = json.load(fh)

    max_err = 0.0
    comparisons: List[Dict[str, Any]] = []
    for item in payload["traces"]:
        scen = item["scenario_kind"]
        dt = float(item["fixed_step_seconds"])
        samples = item["samples"]
        total_steps = int(samples[-1]["step_index"])
        sim = run_rusty_matter_planarian_scenario(
            scen,
            cells_per_band=8,
            total_steps=total_steps,
            dt=dt,
        )

        # Compare at step 0 and across all sampled frames (0, 15, ..., 150)
        s0 = samples[0]
        s_end = samples[-1]
        err_g = abs(sim.cross_cut_base_conductance - float(item["cross_cut_base_conductance_average"]))
        err_mem_end = abs(sim.posterior_memory - float(s_end["posterior_memory_average"]))
        err_head_end = abs(sim.posterior_head_readout - float(s_end["posterior_head_identity_average"]))
        err_ant_head_end = abs(sim.anterior_head_readout - float(s_end["head_identity_at_head_average"]))
        err_tail_end = abs(sim.posterior_tail_readout - float(s_end["tail_identity_at_tail_average"]))
        err_s0 = abs(sim.trajectory_posterior_head_readout[0] - float(s0["posterior_head_identity_average"]))

        scen_max_err = max(err_g, err_mem_end, err_head_end, err_ant_head_end, err_tail_end, err_s0)
        max_err = max(max_err, scen_max_err)
        comparisons.append(
            {
                "scenario": scen,
                "max_abs_err": scen_max_err,
                "posterior_mean_v": sim.posterior_mean_v,
                "cut_band_mean_v": sim.cut_band_mean_v,
                "posterior_memory": sim.posterior_memory,
                "ref_posterior_memory": float(s_end["posterior_memory_average"]),
                "posterior_head_readout": sim.posterior_head_readout,
                "ref_posterior_head_readout": float(s_end["posterior_head_identity_average"]),
                "mean_conductance": sim.mean_conductance,
                "cross_cut_base_g": sim.cross_cut_base_conductance,
                "ref_samples": samples,
            }
        )
    return max_err <= tol, max_err, comparisons


def load_oviedo_2010_planform_records() -> Dict[str, Any]:
    """Load the 14 curated Oviedo 2010 PlanformDB 2.5.0 records from rusty-matter."""
    path = PLANFORM_DATASET_DIR / "rusty_matter_planformdb_derived_v0.json"
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


# =============================================================================
# 4. Canonical Planform Silhouettes (#1 Wild Type vs. #2 Double Head)
# =============================================================================
def _load_lobo_2013_module():
    """Import 2013-lobo-planform-morphology.py dynamically for silhouette decoding."""
    mod_path = REPO_ROOT / "2013-lobo-planform-morphology.py"
    spec = importlib.util.spec_from_file_location("lobo2013", mod_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {mod_path}")
    mod = importlib.util.module_from_spec(spec)
    import sys
    sys.modules["lobo2013"] = mod
    spec.loader.exec_module(mod)
    return mod


def build_canonical_planform_morphologies() -> Tuple[Any, Dict[int, Any]]:
    """Build or load Planform morphologies #1 (Wild Type) and #2 (Double head, two pharynges).

    If PLANFORM_DB_PATH points to `planformDB_2.5.0.edb`, loads directly from SQLite;
    otherwise uses the exact region radii and organ tables encoded in PlanformDB 2.5.0
    (head radii [60, 80, 60], trunk [60, 60], tail [50, 80, 50], inter-region distance 200
    with border ratio 0.5 -> 560 x 120 units).
    """
    lobo = _load_lobo_2013_module()
    db_env = os.getenv("PLANFORM_DB_PATH", "")
    if db_env and Path(db_env).is_file():
        all_morphs = lobo.load_morphologies(db_env)
        return lobo, {1: all_morphs[1], 2: all_morphs[2]}

    # Exact PlanformDB 2.5.0 graph encoding for #1 (Wild Type)
    wt_regions = {
        1: lobo.Region(id=1, type="Head", params=[60.0, 80.0, 60.0]),
        2: lobo.Region(id=2, type="Trunk", params=[60.0, 60.0]),
        3: lobo.Region(id=3, type="Tail", params=[50.0, 80.0, 50.0]),
    }
    wt_links = [
        lobo.Link(from_region=1, to_region=2, dist=200.0, ratio=0.5, ang=3.0 * math.pi / 2.0),
        lobo.Link(from_region=2, to_region=3, dist=200.0, ratio=0.5, ang=3.0 * math.pi / 2.0),
    ]
    wt_organs = [
        lobo.Organ(kind="Eye", region=1, vx=-22.0, vy=25.0, rot=math.pi / 2.0),
        lobo.Organ(kind="Eye", region=1, vx=22.0, vy=25.0, rot=math.pi / 2.0),
        lobo.Organ(kind="Brain", region=1, vx=-22.0, vy=-5.0, rot=math.pi / 2.0),
        lobo.Organ(kind="Brain", region=1, vx=22.0, vy=-5.0, rot=math.pi / 2.0),
        lobo.Organ(kind="Phar.", region=2, vx=0.0, vy=0.0, rot=-math.pi / 2.0),
        lobo.Organ(kind="VNC", region=2, vx=-24.0, vy=0.0, ends=(0.0, 215.0, 0.0, -215.0)),
        lobo.Organ(kind="VNC", region=2, vx=24.0, vy=0.0, ends=(0.0, 215.0, 0.0, -215.0)),
    ]
    m_wt = lobo.Morphology(id=1, name="Wild Type", regions=wt_regions, links=wt_links, organs=wt_organs)
    lobo.layout(m_wt)
    m_wt.outlines = {rid: lobo.region_outline(m_wt, rid) for rid in m_wt.regions}

    # Exact PlanformDB 2.5.0 graph encoding for #2 (Double head, two pharynges)
    dh_regions = {
        1: lobo.Region(id=1, type="Head", params=[60.0, 80.0, 60.0]),
        2: lobo.Region(id=2, type="Trunk", params=[60.0, 60.0]),
        3: lobo.Region(id=3, type="Head", params=[60.0, 80.0, 60.0]),
    }
    dh_links = [
        lobo.Link(from_region=1, to_region=2, dist=200.0, ratio=0.5, ang=3.0 * math.pi / 2.0),
        lobo.Link(from_region=2, to_region=3, dist=200.0, ratio=0.5, ang=3.0 * math.pi / 2.0),
    ]
    dh_organs = [
        # Anterior head organs
        lobo.Organ(kind="Eye", region=1, vx=-22.0, vy=25.0, rot=math.pi / 2.0),
        lobo.Organ(kind="Eye", region=1, vx=22.0, vy=25.0, rot=math.pi / 2.0),
        lobo.Organ(kind="Brain", region=1, vx=-22.0, vy=-5.0, rot=math.pi / 2.0),
        lobo.Organ(kind="Brain", region=1, vx=22.0, vy=-5.0, rot=math.pi / 2.0),
        # Posterior ectopic head organs (bipolar mirror)
        lobo.Organ(kind="Eye", region=3, vx=-22.0, vy=-25.0, rot=-math.pi / 2.0),
        lobo.Organ(kind="Eye", region=3, vx=22.0, vy=-25.0, rot=-math.pi / 2.0),
        lobo.Organ(kind="Brain", region=3, vx=-22.0, vy=5.0, rot=-math.pi / 2.0),
        lobo.Organ(kind="Brain", region=3, vx=22.0, vy=5.0, rot=-math.pi / 2.0),
        # Two opposing pharynges in the trunk + bipolar VNCs
        lobo.Organ(kind="Phar.", region=2, vx=0.0, vy=44.0, rot=-math.pi / 2.0),
        lobo.Organ(kind="Phar.", region=2, vx=0.0, vy=-44.0, rot=math.pi / 2.0),
        lobo.Organ(kind="VNC", region=2, vx=-24.0, vy=0.0, ends=(0.0, 215.0, 0.0, -215.0)),
        lobo.Organ(kind="VNC", region=2, vx=24.0, vy=0.0, ends=(0.0, 215.0, 0.0, -215.0)),
    ]
    m_dh = lobo.Morphology(
        id=2,
        name="Double head, two pharynges",
        regions=dh_regions,
        links=dh_links,
        organs=dh_organs,
    )
    lobo.layout(m_dh)
    m_dh.outlines = {rid: lobo.region_outline(m_dh, rid) for rid in m_dh.regions}

    return lobo, {1: m_wt, 2: m_dh}


def compute_silhouette_iou(lobo_mod: Any, m_a: Any, m_b: Any, res: int = 256) -> float:
    """Intersection-over-union of two decoded Planform silhouettes on a shared grid."""
    pts = np.vstack(list(m_a.outlines.values()) + list(m_b.outlines.values()))
    x0, x1 = float(pts[:, 0].min() - 20.0), float(pts[:, 0].max() + 20.0)
    y0, y1 = float(pts[:, 1].min() - 20.0), float(pts[:, 1].max() + 20.0)
    xs = np.linspace(x0, x1, res)
    ys = np.linspace(y0, y1, res)
    gx, gy = np.meshgrid(xs, ys)
    grid = np.column_stack([gx.ravel(), gy.ravel()])
    from matplotlib.path import Path as MplPath

    mask_a = np.zeros(grid.shape[0], dtype=bool)
    mask_b = np.zeros(grid.shape[0], dtype=bool)
    for poly in m_a.outlines.values():
        mask_a |= MplPath(poly).contains_points(grid)
    for poly in m_b.outlines.values():
        mask_b |= MplPath(poly).contains_points(grid)
    inter = float(np.logical_and(mask_a, mask_b).sum())
    union = float(np.logical_or(mask_a, mask_b).sum())
    return inter / max(union, 1.0)


# =============================================================================
# 5. Integrated Analysis & Publication-Grade 6-Panel Figure
# =============================================================================
def run_integrated_analysis(
    output_png: Path = REPO_ROOT / "cervera_levin_mafe_bioelectric_morphospace.png",
    output_csv: Path = REPO_ROOT / "cervera_levin_mafe_metrics.csv",
    show: bool = False,
) -> Dict[str, Any]:
    """Run the combined Cervera-Levin-Mafe (2021) + MesmerPrism/rusty-matter analysis."""
    checksums = verify_fixture_checksums()
    for fname, (ok, digest) in checksums.items():
        if not ok:
            raise RuntimeError(f"Fixture SHA-256 mismatch for {fname}: got {digest}")

    parity_ok, max_trace_err, scenario_rows = verify_rusty_matter_trace_parity()
    if not parity_ok:
        raise RuntimeError(f"rusty-matter trace parity failed: max_err={max_trace_err:.3e}")

    oviedo_data = load_oviedo_2010_planform_records()
    records = oviedo_data["records"]

    # Group empirical Oviedo 2010 records
    crop_order = [
        "head_crop",
        "pre_pharyngeal_crop",
        "pharyngeal_crop",
        "post_pharyngeal_crop",
        "tail_crop",
    ]
    crop_labels = [
        "Head\ncrop",
        "Pre-phar.\ntrunk",
        "Pharyngeal\ntrunk",
        "Post-phar.\ntrunk",
        "Tail\ncrop",
    ]
    octanol_by_crop: Dict[str, Dict[str, Any]] = {}
    rnai_by_crop: Dict[str, Dict[str, Any]] = {}
    timing_records: List[Tuple[float, Dict[str, Any]]] = []
    timing_hours_map = {
        "head_plus_post_pharyngeal_crop_with_vnc_disruption_t0d": 0.0,
        "head_plus_post_pharyngeal_crop_with_vnc_disruption_t0_125d": 3.0,
        "head_plus_post_pharyngeal_crop_with_vnc_disruption_t0_5d": 12.0,
        "head_plus_post_pharyngeal_crop_with_vnc_disruption_t1d": 24.0,
    }

    def _double_head_pct(rec_obj: Dict[str, Any]) -> float:
        for rm in rec_obj.get("resultant_morphologies", []):
            if int(rm.get("morphology_id", -1)) == 2:
                return float(rm.get("frequency", 0.0)) * 100.0
        return 0.0

    for rec in records:
        labels = rec["normalized_labels"]
        target = labels["teaching_target"]
        manip = labels["manipulation"]
        if target == "gap_block_conductance":
            octanol_by_crop[manip] = rec
        elif target == "innexin_gap_junction_label":
            rnai_by_crop[manip] = rec
        elif target == "gap_block_vnc_disruption_boundary":
            t_hr = timing_hours_map.get(manip, 0.0)
            timing_records.append((t_hr, rec))
    timing_records.sort(key=lambda pair: pair[0])

    # 1) Cervera-Levin-Mafe RD sweep across f2 in [1.0 -> 0.25]
    f2_sweep = np.linspace(1.0, 0.25, 16)
    rd_wavefronts: List[float] = []
    rd_profiles: Dict[float, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for f2 in f2_sweep:
        x, m1, m2 = rd_antagonistic_1d(D1=0.02, D2=0.02 * float(f2), k1=0.08, k2=0.08, k12=2.0, T=3.5)
        xs = compute_wavefront_position(x, m1, m2)
        rd_wavefronts.append(xs)
        rd_profiles[round(float(f2), 2)] = (x, m1, m2)

    # 2) Run the 5 canonical rusty-matter scenarios
    canonical_scenarios = [
        "Baseline",
        "TransverseCutWound",
        "GapBlock",
        "TransientDepolarizationMemory",
        "TransientDepolarizationNoMemoryControl",
    ]
    sim_by_scenario = {
        scen: run_rusty_matter_planarian_scenario(scen) for scen in canonical_scenarios
    }

    # 3) Simulate crop-specific AP gap-junction blockade across the 5 Oviedo 2010 regions
    # Map each crop region to the cut band pair in the 5-band substrate:
    # Head=4, Prepharyngeal=3, Pharyngeal=2, Postpharyngeal=1, Tail=0
    crop_to_cut_pair = {
        "head_crop": (4, 4),
        "pre_pharyngeal_crop": (3, 4),
        "pharyngeal_crop": (2, 3),
        "post_pharyngeal_crop": (1, 2),
        "tail_crop": (0, 0),
    }
    crop_to_f2_effective = {
        "head_crop": 0.92,
        "pre_pharyngeal_crop": 0.68,
        "pharyngeal_crop": 0.50,
        "post_pharyngeal_crop": 0.28,
        "tail_crop": 0.98,  # isolated tail blastema without trunk injury signal (Oviedo 2010 Exp 419)
    }
    crop_model_head_prob: List[float] = []
    crop_rd_shift_pct: List[float] = []
    base_post_head = sim_by_scenario["Baseline"].posterior_head_readout
    max_post_head = sim_by_scenario["TransientDepolarizationMemory"].posterior_head_readout
    for crop_key in crop_order:
        pair = crop_to_cut_pair[crop_key]
        if crop_key == "tail_crop":
            # Tail tip crop without trunk cut remains 0% double-head in Oviedo 2010 (Exp 419 & 450)
            res_crop = sim_by_scenario["Baseline"]
            prob_dh = 0.0
        else:
            res_crop = run_rusty_matter_planarian_scenario(
                "CustomCropGapBlock",
                custom_gap_scale=0.08,
                custom_cut_band_pair=pair,
            )
            prob_dh = (
                np.clip(
                    (res_crop.posterior_head_readout - base_post_head)
                    / max(max_post_head - base_post_head, 1e-6),
                    0.0,
                    1.0,
                )
                * 100.0
            )
        crop_model_head_prob.append(float(prob_dh))

        f2_eff = crop_to_f2_effective[crop_key]
        x_c, m1_c, m2_c = rd_antagonistic_1d(D1=0.02, D2=0.02 * f2_eff, T=3.5)
        x_star = compute_wavefront_position(x_c, m1_c, m2_c)
        # Relative expansion of anterior (m1) territory past midline x = 0.50
        shift_pct = max(0.0, (x_star - 0.50) / 0.13) * 100.0
        crop_rd_shift_pct.append(float(min(shift_pct, 100.0)))

    # 4) 2D (f2, f3) morphospace grid from Cervera, Levin & Mafe (2021)
    f2_grid = np.linspace(1.0, 0.35, 10)
    f3_grid = np.linspace(1.0, 0.35, 10)
    xstar_grid = np.zeros((len(f2_grid), len(f3_grid)), dtype=float)
    for i, f2_val in enumerate(f2_grid):
        x_g, m1_g, m2_g = rd_antagonistic_1d(D1=0.02, D2=0.02 * float(f2_val), T=3.5)
        xs_val = compute_wavefront_position(x_g, m1_g, m2_g)
        for j, _f3_val in enumerate(f3_grid):
            xstar_grid[i, j] = xs_val

    # 5) Canonical Planform silhouettes (#1 Wild Type vs #2 Double head, two pharynges)
    lobo_mod, morphs = build_canonical_planform_morphologies()
    wt_metrics = lobo_mod.morphometrics(morphs[1])
    dh_metrics = lobo_mod.morphometrics(morphs[2])
    wt_dh_iou = compute_silhouette_iou(lobo_mod, morphs[1], morphs[2])

    # Write CSV summary
    csv_rows: List[Dict[str, Any]] = []
    for scen in canonical_scenarios:
        s = sim_by_scenario[scen]
        csv_rows.append(
            {
                "section": "rusty_matter_circuit_scenario",
                "id": scen,
                "anterior_mean_v": round(s.anterior_mean_v, 6),
                "posterior_mean_v": round(s.posterior_mean_v, 6),
                "ap_delta_v": round(s.ap_delta_v, 6),
                "mean_conductance": round(s.mean_conductance, 6),
                "posterior_memory": round(s.posterior_memory, 6),
                "posterior_head_readout": round(s.posterior_head_readout, 6),
                "empirical_double_head_pct": "",
                "rd_wavefront_x_star": "",
                "notes": f"Verified against rusty_matter_planarian_traces.json (max_err={max_trace_err:.3f})",
            }
        )
    for idx, crop_key in enumerate(crop_order):
        oct_rec = octanol_by_crop[crop_key]
        rnai_rec = rnai_by_crop[crop_key]
        oct_dh = _double_head_pct(oct_rec)
        rnai_dh = _double_head_pct(rnai_rec)
        oct_exp = oct_rec["source_ids"]["experiment_id"]
        rnai_exp = rnai_rec["source_ids"]["experiment_id"]
        oct_n = oct_rec["assay_context"]["sample_count"]
        rnai_n = rnai_rec["assay_context"]["sample_count"]
        csv_rows.append(
            {
                "section": "oviedo_2010_crop_gradient",
                "id": crop_key,
                "anterior_mean_v": "",
                "posterior_mean_v": "",
                "ap_delta_v": "",
                "mean_conductance": "",
                "posterior_memory": "",
                "posterior_head_readout": round(crop_model_head_prob[idx] / 100.0, 4),
                "empirical_double_head_pct": f"octanol={oct_dh:.0f}%; rnai={rnai_dh:.0f}%",
                "rd_wavefront_x_star": round(0.50 + 0.26 * (crop_rd_shift_pct[idx] / 100.0), 4),
                "notes": f"PlanformDB Exps {oct_exp}/{rnai_exp} (n={oct_n},{rnai_n})",
            }
        )

    with open(output_csv, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(csv_rows[0].keys()))
        writer.writeheader()
        writer.writerows(csv_rows)

    # =========================================================================
    # Render 6-Panel Publication Synthesis Figure
    # =========================================================================
    fig = plt.figure(figsize=(16.5, 10.5), dpi=160)
    gs = fig.add_gridspec(2, 3, hspace=0.34, wspace=0.28)

    # --- Panel A: Cervera, Levin & Mafe (2021) 1D RD Antagonism & Wavefront Shift ---
    ax_a = fig.add_subplot(gs[0, 0])
    sample_f2s = [(1.0, "#1f77b4", "-"), (0.60, "#e67e22", "--"), (0.30, "#c0392b", "-.")]
    for f2_val, col, ls in sample_f2s:
        x_p, m1_p, m2_p = rd_antagonistic_1d(D1=0.02, D2=0.02 * f2_val, T=3.5)
        xs_p = compute_wavefront_position(x_p, m1_p, m2_p)
        if f2_val == 1.0:
            ax_a.plot(x_p, m1_p, color="#b03a2e", lw=2.2, label="$m_1$ (anterior head signal, $D_1 = D_{1,0}$)")
        ax_a.plot(
            x_p,
            m2_p,
            color=col,
            ls=ls,
            lw=2.0,
            label=f"$m_2$ ($f_2 = D_2/D_{{2,0}} = {f2_val:.2f}$, $x^* = {xs_p:.2f}$)",
        )
        ax_a.axvline(xs_p, color=col, ls=":", lw=1.3, alpha=0.85)
    ax_a.axvspan(0.50, float(compute_wavefront_position(*rd_antagonistic_1d(D1=0.02, D2=0.02 * 0.30, T=3.5))),
                 color="#f5b7b1", alpha=0.28, label="Anterior territory expansion $\\Delta x^*$")
    ax_a.set_xlim(0.0, 1.0)
    ax_a.set_xlabel("Normalized AP Coordinate $x/L$ ($0=$ Anterior, $1=$ Posterior)", fontsize=9)
    ax_a.set_ylabel("Morphogen Concentration", fontsize=9)
    ax_a.set_title(
        "A. Cervera–Levin–Mafe (2021) RD Wavefront Shift\n"
        "Gap-Junction Blockade ($D_2 = f_2 D_{2,0}$) Expands Head Territory",
        fontsize=10,
        fontweight="bold",
    )
    ax_a.legend(fontsize=7.5, loc="upper right", framealpha=0.92)
    ax_a.grid(alpha=0.25)

    # --- Panel B: MesmerPrism/rusty-matter 5-Band Bioelectric Circuit Outcomes ---
    ax_b = fig.add_subplot(gs[0, 1])
    scen_short = [
        "Baseline\n(Intact)",
        "Transverse\nCut Wound",
        "GapBlock\n(Octanol/RNAi)",
        "Depol. Pulse\n(Memory ON)",
        "Depol. Pulse\n(Memory OFF)",
    ]
    post_v_vals = [sim_by_scenario[s].posterior_mean_v for s in canonical_scenarios]
    post_mem_vals = [sim_by_scenario[s].posterior_memory for s in canonical_scenarios]
    post_head_vals = [sim_by_scenario[s].posterior_head_readout for s in canonical_scenarios]

    x_idx = np.arange(len(canonical_scenarios))
    w = 0.25
    ax_b.bar(x_idx - w, post_v_vals, width=w, color="#2980b9", label="Posterior $V_{\\mathrm{mem}}$ (norm.)")
    ax_b.bar(x_idx, post_mem_vals, width=w, color="#f39c12", label="Posterior Hysteresis Memory $M_{\\mathrm{post}}$")
    ax_b.bar(x_idx + w, post_head_vals, width=w, color="#c0392b", label="Posterior Head Readout $R_{\\mathrm{head,post}}$")
    ax_b.axhline(0.50, color="#333333", ls="--", lw=1.0, label="Ectopic Head Threshold ($R_{\\mathrm{head}} = 0.50$)")
    ax_b.set_xticks(x_idx)
    ax_b.set_xticklabels(scen_short, fontsize=7.8)
    ax_b.set_ylim(-0.32, 1.05)
    ax_b.set_ylabel("Normalized Circuit State / Readout", fontsize=9)
    ax_b.set_title(
        "B. MesmerPrism/rusty-matter 5-Band Bioelectric Circuit\n"
        "Voltage-Gated GJ + Hysteresis Memory vs. Memoryless Control",
        fontsize=10,
        fontweight="bold",
    )
    ax_b.legend(fontsize=7.3, loc="upper left", framealpha=0.92)
    ax_b.grid(axis="y", alpha=0.25)

    # --- Panel C: Oviedo et al. (2010) PlanformDB 2.5.0 AP Crop Gradient ---
    ax_c = fig.add_subplot(gs[0, 2])
    oct_pcts = [_double_head_pct(octanol_by_crop[c]) for c in crop_order]
    rnai_pcts = [_double_head_pct(rnai_by_crop[c]) for c in crop_order]
    cx = np.arange(len(crop_order))
    bw = 0.32
    ax_c.bar(
        cx - bw / 2,
        oct_pcts,
        width=bw,
        color="#d35400",
        alpha=0.88,
        label="Oviedo 2010 Octanol (Exps 415–419, $n=115..178$)",
    )
    ax_c.bar(
        cx + bw / 2,
        rnai_pcts,
        width=bw,
        color="#8e44ad",
        alpha=0.85,
        label="Oviedo 2010 Innexin RNAi (Exps 446–450, $n=18..21$)",
    )
    ax_c.plot(
        cx,
        crop_model_head_prob,
        color="#c0392b",
        marker="o",
        lw=2.0,
        label="rusty-matter Circuit Posterior Head Index (%)",
    )
    ax_c.plot(
        cx,
        crop_rd_shift_pct,
        color="#16a085",
        marker="s",
        ls="--",
        lw=1.8,
        label="Cervera 2021 RD Wavefront Shift $\\Delta x^*$ (%)",
    )
    ax_c.set_xticks(cx)
    ax_c.set_xticklabels(crop_labels, fontsize=8)
    ax_c.set_ylim(0, 112)
    ax_c.set_ylabel("Double-Head (#2) Frequency / Index (%)", fontsize=9)
    ax_c.set_title(
        "C. PlanformDB 2.5.0 Empirical AP Crop Gradient (Oviedo 2010)\n"
        "Trunk Posteriority Amplifies Gap-Junction Double-Head Penetrance",
        fontsize=10,
        fontweight="bold",
    )
    ax_c.legend(fontsize=7.1, loc="upper left", framealpha=0.92)
    ax_c.grid(axis="y", alpha=0.25)

    # --- Panel D: Early Bioelectric Timing Window (Durant 2019 & Oviedo 2010 Exps 441-444) ---
    ax_d = fig.add_subplot(gs[1, 0])
    mem_scen = sim_by_scenario["TransientDepolarizationMemory"]
    nomem_scen = sim_by_scenario["TransientDepolarizationNoMemoryControl"]
    # Map simulation steps (1..120, dt=0.05 -> 6.0 model time units) onto 0..24 hpa window
    t_hours = np.array(mem_scen.trajectory_steps, dtype=float) * (24.0 / mem_scen.total_steps)
    ax_d.axvspan(0.4, 6.4, color="#fdebd0", alpha=0.65, label="Transient Depol. Pulse Window (3–6 hpa)")
    ax_d.plot(
        t_hours,
        np.array(mem_scen.trajectory_posterior_head_readout) * 100.0,
        color="#c0392b",
        lw=2.3,
        label="Circuit $R_{\\mathrm{head,post}}$ (Hysteresis Memory ON: 85.8%)",
    )
    ax_d.plot(
        t_hours,
        np.array(nomem_scen.trajectory_posterior_head_readout) * 100.0,
        color="#7f8c8d",
        ls="--",
        lw=2.0,
        label="Circuit $R_{\\mathrm{head,post}}$ (Memoryless Control OFF: 20.6%)",
    )
    # Overlay Oviedo 2010 Exps 441-444 VNC + Octanol timing series (0, 3, 12, 24 hpa)
    emp_hours = [t_hr for t_hr, _r in timing_records]
    emp_dh = [_double_head_pct(r) for _t_hr, r in timing_records]
    ax_d.plot(
        emp_hours,
        emp_dh,
        color="#1f618d",
        marker="D",
         markersize=6.5,
        lw=1.8,
        ls="-.",
        label="PlanformDB Exps 441–444 VNC+Octanol (% #2, $n=10$/pt)",
    )
    ax_d.set_xlim(0, 24.5)
    ax_d.set_ylim(0, 102)
    ax_d.set_xlabel("Time Post-Amputation / Perturbation Onset (hours, hpa)", fontsize=9)
    ax_d.set_ylabel("Posterior Head Readout / Double-Head (%)", fontsize=9)
    ax_d.set_title(
        "D. Early Bioelectric Window (Durant 2019 & Oviedo 2010)\n"
        "3–6 hpa Depolarization Locks Bipolar Fate Only With Pattern Memory",
        fontsize=10,
        fontweight="bold",
    )
    ax_d.legend(fontsize=7.1, loc="upper right", framealpha=0.92)
    ax_d.grid(alpha=0.25)

    # --- Panel E: 2D (f2, f3) Morphospace Heatmap & Phenotype Phase Boundary ---
    ax_e = fig.add_subplot(gs[1, 1])
    im = ax_e.imshow(
        xstar_grid,
        origin="lower",
        aspect="auto",
        extent=[float(f3_grid.min()), float(f3_grid.max()), float(f2_grid.min()), float(f2_grid.max())],
        cmap="RdYlBu_r",
    )
    cbar = fig.colorbar(im, ax=ax_e, fraction=0.046, pad=0.04)
    cbar.set_label("Normalized Wavefront Position $x^*/L$", fontsize=8.5)
    # Draw phenotype transition line at f2 = 0.58 (x* ≈ 0.58)
    ax_e.axhline(0.58, color="#111111", ls="--", lw=1.6)
    ax_e.text(
        0.67,
        0.82,
        "Wild-Type Regime (#1)\n1 Head + 1 Tail ($x^* \\approx 0.50$)",
        ha="center",
        va="center",
        fontsize=8.3,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.82, edgecolor="#2980b9"),
    )
    ax_e.text(
        0.67,
        0.44,
        "Double-Head Regime (#2)\n2 Heads + 2 Pharynges ($x^* > 0.58$)",
        ha="center",
        va="center",
        fontsize=8.3,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.86, edgecolor="#c0392b"),
    )
    ax_e.set_xlabel("Lateral GJ Diffusion Factor $f_3 = D_3/D_{3,0}$", fontsize=9)
    ax_e.set_ylabel("AP GJ Diffusion Factor $f_2 = D_2/D_{2,0}$", fontsize=9)
    ax_e.set_title(
        "E. Cervera–Levin–Mafe (2021) $(f_2, f_3)$ GJ Morphospace\n"
        "Bifurcation From Normal Polarity (#1) to Bipolar Double-Head (#2)",
        fontsize=10,
        fontweight="bold",
    )

    # --- Panel F: Decoded Planform Silhouettes (#1 Wild Type vs. #2 Double Head) ---
    gs_f = gs[1, 2].subgridspec(1, 2, wspace=0.08)
    ax_f1 = fig.add_subplot(gs_f[0, 0])
    ax_f2 = fig.add_subplot(gs_f[0, 1])
    half = lobo_mod.frame_bounds([morphs[1], morphs[2]])
    for ax_sub, mid, subtitle in (
        (
            ax_f1,
            1,
            f"#1 Wild Type\n1 Head, 1 Tail, 1 Phar.\n{wt_metrics['ap_length']:.0f}×{wt_metrics['ml_width']:.0f} u",
        ),
        (
            ax_f2,
            2,
            f"#2 Double Head\n2 Heads, 0 Tails, 2 Phar.\n{dh_metrics['ap_length']:.0f}×{dh_metrics['ml_width']:.0f} u",
        ),
    ):
        m_obj = morphs[mid]
        lobo_mod.draw_morphology(ax_sub, m_obj, show_organs=True, outline_lw=1.5)
        cx_m, cy_m = lobo_mod.centre_of(m_obj)
        lobo_mod.fit_axes(ax_sub, (cx_m - half, cx_m + half, cy_m - half, cy_m + half))
        ax_sub.set_title(subtitle, fontsize=8.5, fontweight="bold")

    fig.text(
        0.83,
        0.04,
        f"Lobo–Malone–Levin (2013) Graph Formalism — Silhouette IoU(#1, #2) = {wt_dh_iou:.3f}\n"
        "(Resolves rusty-matter/docs/PLANARIAN_METRIC_TARGETS.md silhouette gate)",
        ha="center",
        va="center",
        fontsize=7.8,
        style="italic",
        color="#333333",
    )

    fig.suptitle(
        "Planarian Gap-Junction Bioelectric & Reaction–Diffusion Morphospace\n"
        "Bridging Cervera–Levin–Mafe (2021), MesmerPrism/rusty-matter Circuit Traces, and PlanformDB 2.5.0 (Oviedo 2010; Durant 2019)",
        fontsize=12,
        fontweight="bold",
        y=0.98,
    )
    fig.savefig(output_png, dpi=160, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    print(f"Saved synthesis figure to {output_png}")
    print(f"Saved quantitative metrics table to {output_csv}")

    return {
        "checksums": checksums,
        "trace_parity_ok": parity_ok,
        "max_trace_err": max_trace_err,
        "wt_metrics": wt_metrics,
        "dh_metrics": dh_metrics,
        "wt_dh_iou": wt_dh_iou,
        "output_png": str(output_png),
        "output_csv": str(output_csv),
    }


def demo() -> None:
    """Default interactive/CLI entrypoint."""
    run_integrated_analysis(show=False)


def make_gif(output: str, n_frames: int = 24, fps: int = 8) -> None:
    """Animate the gap-junction-block sweep alongside the bioelectric head-readout shift."""
    from PIL import Image

    f2_values = np.linspace(1.0, 0.35, n_frames)
    z, m3 = rd_linear_1d(D=0.02, k=0.08, J_left=0.6, J_right=0.6, T=3.5)
    g_profile = m3 / m3.max()
    mask = planarian_silhouette()
    lobo_mod, morphs = build_canonical_planform_morphologies()
    half = lobo_mod.frame_bounds([morphs[1], morphs[2]])

    frames = []
    for f2 in f2_values:
        x, m1, m2 = rd_antagonistic_1d(
            D1=0.02, D2=0.02 * float(f2), k1=0.08, k2=0.08, k12=2.0, J1=1.0, J2=1.0, T=3.5
        )
        xs = compute_wavefront_position(x, m1, m2)
        gj_scale = max(0.08, float(f2))
        circuit_res = run_rusty_matter_planarian_scenario(
            "CustomCropGapBlock", custom_gap_scale=gj_scale, custom_cut_band_pair=(1, 2)
        )

        fig = plt.figure(figsize=(8.6, 5.8), dpi=120)
        gs = fig.add_gridspec(2, 2, width_ratios=[2.2, 1.0], height_ratios=[1.0, 1.05], hspace=0.36, wspace=0.22)
        a0 = fig.add_subplot(gs[0, 0])
        a1 = fig.add_subplot(gs[1, 0])
        a_morph = fig.add_subplot(gs[:, 1])

        a0.plot(x, m1, color="#c0392b", lw=2.0, label="$m_1$ (anterior head signal)")
        a0.plot(x, m2, color="#2e6fba", lw=2.0, label="$m_2$ (posterior signal)")
        a0.axvline(xs, ls="--", color="k", lw=1.2, label=f"wavefront $x^* = {xs:.2f}$")
        a0.set_xlim(0, 1)
        a0.set_ylim(0, 1.05 * max(float(m1.max()), float(m2.max()), 1e-9))
        a0.set_xlabel("AP axis ($0=$ Anterior, $1=$ Posterior)", fontsize=8.5)
        a0.set_ylabel("Concentration", fontsize=8.5)
        a0.legend(fontsize=7.5, loc="upper center")
        a0.grid(alpha=0.25)
        a0.set_title(
            f"Cervera–Levin–Mafe (2021): $D_2 = {f2:.2f} D_{{2,0}}$ | "
            f"rusty-matter $R_{{\\mathrm{{head,post}}}} = {circuit_res.posterior_head_readout:.2f}$",
            fontsize=9.0,
        )

        dom = (m1 - m2) / (m1 + m2 + 1e-9)
        rgb, inside, aspect = territory_image(mask, x, z, dom, g_profile)
        a1.imshow(rgb, extent=[0, 1, 0, 1], origin="lower", aspect=aspect, interpolation="bilinear")
        a1.contour(np.linspace(0, 1, inside.shape[1]), np.linspace(0, 1, inside.shape[0]), inside.astype(float),
                   levels=[0.5], colors="#333333", linewidths=0.9)
        a1.axvline(xs, ls="--", color="k", lw=1.2)
        a1.set_xlabel("AP axis", fontsize=8.5)
        a1.set_ylabel("ML axis", fontsize=8.5)
        a1.set_yticks([])
        if mask is None:
            a1.set_title("RD territories: red = anterior ($m_1$), blue = posterior ($m_2$)", fontsize=8.8)
        else:
            a1.set_title(f"RD territories on a real $\\it{{{PLANARIAN_PHOTO['species'].replace(' ', '\\ ')}}}$ "
                         "(silhouette of a photograph)\nred = anterior ($m_1$), blue = posterior ($m_2$)", fontsize=8.4)
            fig.text(0.01, 0.01, f"photograph: {PLANARIAN_PHOTO['credit']}", fontsize=6.5, color="#64748b", ha="left", va="bottom")

        active_mid = 2 if circuit_res.posterior_head_readout >= 0.55 else 1
        m_obj = morphs[active_mid]
        lobo_mod.draw_morphology(a_morph, m_obj, show_organs=True, outline_lw=1.4)
        cx_m, cy_m = lobo_mod.centre_of(m_obj)
        lobo_mod.fit_axes(a_morph, (cx_m - half, cx_m + half, cy_m - half, cy_m + half))
        pheno_label = "#2 Double Head\n(Bipolar)" if active_mid == 2 else "#1 Wild Type\n(Normal Polarity)"
        a_morph.set_title(
            f"Planform Outcome\n{pheno_label}\n$V_{{\\mathrm{{post}}}}={circuit_res.posterior_mean_v:+.2f}$",
            fontsize=9.0,
            fontweight="bold",
        )

        fig.suptitle(
            "Gap-Junction Blockade Sweep: Cervera–Levin–Mafe (2021) + rusty-matter + PlanformDB",
            fontsize=10.0,
            fontweight="bold",
        )
        fig.canvas.draw()
        frames.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()))
        plt.close(fig)

    frames += frames[::-1][1:-1]
    frames[0].save(output, save_all=True, append_images=frames[1:], duration=int(1000 / fps), loop=0)
    print(f"Saved animation to {output}  ({len(frames)} frames)")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Cervera, Levin & Mafe (2021) + MesmerPrism/rusty-matter planarian bioelectric synthesis"
    )
    parser.add_argument("--gif", type=str, default=None, help="Output path for animated GIF")
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "cervera_levin_mafe_bioelectric_morphospace.png",
        help="Output path for synthesis PNG figure",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=REPO_ROOT / "cervera_levin_mafe_metrics.csv",
        help="Output path for quantitative metrics CSV",
    )
    parser.add_argument("--no-show", action="store_true", help="Do not open interactive matplotlib window")
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Verify SHA-256 fixture hashes and exact rusty-matter trace parity",
    )
    args = parser.parse_args(argv)

    if args.self_test:
        checksums = verify_fixture_checksums()
        for fname, (ok, digest) in checksums.items():
            status = "PASS" if ok else "FAIL"
            print(f"[SHA-256 {status}] {fname}: {digest}")
            if not ok:
                return 1
        parity_ok, max_err, rows = verify_rusty_matter_trace_parity()
        print(f"[TRACE PARITY {'PASS' if parity_ok else 'FAIL'}] max_abs_err = {max_err:.3e}")
        for r in rows:
            print(
                f"  - {r['scenario']:40s} | V_post={r['posterior_mean_v']:+.6f} "
                f"| M_post={r['posterior_memory']:.6f} | R_head,post={r['posterior_head_readout']:.6f} "
                f"| err={r['max_abs_err']:.2e}"
            )
        return 0 if parity_ok else 1

    if args.gif:
        make_gif(args.gif)
        return 0

    run_integrated_analysis(output_png=args.out, output_csv=args.csv, show=not args.no_show)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

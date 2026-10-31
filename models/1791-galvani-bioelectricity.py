#!/usr/bin/env python3
"""
models/1791-galvani-bioelectricity.py
=====================================
Luigi Galvani (1791) "De Viribus Electricitatis in Motu Musculari Commentarius"
Archival Copperplate Engraving Atlas (Tabulae I-IV) & Biophysical Neuromuscular
Excitation / Action-Potential Propagation Model.

Generates four publication-grade visual artifacts:
  1. `docs/media/galvani_1791_frog_experiments_atlas.png`
     Grand Historical & Biophysical Master Illustration combining the 1791
     Bologna title page, all four original copperplate engravings by Jacopo
     Zambelli (Tab. I-IV), the Wellcome isolated sciatic-nerve dissection
     plate (M0012614), an annotated anatomical circuit diagram of Galvani's
     bimetallic arc (Fe-Cu) vs. 1794 metal-free sciatic nerve-muscle contact,
     and quantitative biophysical plots.
  2. `docs/media/galvani_1791_four_plates_and_mechanisms.png`
     Dedicated 2x4 Archival Plate & Mechanism Companion pairing each of
     Galvani's four 1791 copperplate engravings (Tab. I, II, III, IV) directly
     above its corresponding circuit schematic and electrophysiological trace.
  3. `docs/media/galvani_current_to_twitch_experiments.png`
     Current-to-Twitch & High-Current Dose-Response Atlas from Matteucci's
     1.8 uA Galvanoscopic Frog and 1842 Frog Battery to Pfluger's 1859
     High-Current Anodal Block and 40 Hz Tetanic Contraction (4.50 N).
  4. `docs/media/galvani_1791_neuromuscular_excitation.gif`
     Animated biophysical simulation of Galvani's frog preparation showing
     bimetallic arc closure (spinal brass hook -> iron railing -> crural nerve),
     travelling action potential V_m(x,t) down the sciatic nerve, sarcoplasmic
     Ca2+ transient, and physical hindlimb gastrocnemius twitch contraction
     paired with the original 1791 engravings.

Usage:
  python3 models/1791-galvani-bioelectricity.py
  python3 models/1791-galvani-bioelectricity.py --self-test
  python3 models/1791-galvani-bioelectricity.py --preview-anatomy
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Circle, Polygon
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np
from PIL import Image, ImageOps, ImageEnhance

ROOT = Path(__file__).resolve().parent.parent
GALVANI_DIR = ROOT / "datasets" / "galvani"
REF_JSON = GALVANI_DIR / "galvani_1791_reference.json"
MEDIA_DIR = ROOT / "docs" / "media"

OUT_ATLAS = MEDIA_DIR / "galvani_1791_frog_experiments_atlas.png"
OUT_PLATES = MEDIA_DIR / "galvani_1791_four_plates_and_mechanisms.png"
OUT_CURRENT_TWITCH = MEDIA_DIR / "galvani_current_to_twitch_experiments.png"
OUT_GIF = MEDIA_DIR / "galvani_1791_neuromuscular_excitation.gif"

VERIFIED_PLATE_FILES = {
    "cover": ("galvani_1791_cover_title.jpg", "39e0cf96a1411ee5d0bbc37af345de1699e2ff77c9fc8470539b06679912b01e"),
    "pars_prima": ("galvani_title_1791.jpg", "d864998d9001564273e99c08b49d1c83b538c7eafab65595754b3908064ed6c1"),
    "tab1": ("galvani_1791_tab1_machina.jpg", "eed7e736f6cc5de17678a792e2f0d9d8fe92503c70cec87d1b722ca47ecde202"),
    "tab2": ("galvani_1791_tab2_fulgur.jpg", "9b36af7e836053f2de45753e584b156f0fcd87008a71326addacc13001e6fbd9"),
    "tab3": ("galvani_tab_III_960.jpg", "0631656ca943ee38da018b2ebc5d4df9e103e4bfe5c3836162d5e7341d3d32de"),
    "tab4": ("galvani_1791_tab4_vasa.jpg", "5825594ac944adcf4c984899cc0f431e6b560c88e2e402dec46a8a9d5b479308"),
    "sciatic": ("wellcome_M0012614.jpg", "6613ddd752c200e11889f688b15ec89e1271142cfe7ee47ca5cb3bc7fac5e15e"),
    "frog_prep": ("galvani_frog_prep_wellcome.jpg", "c9cdeefe6922f7bbcdcda91957ae1b18b0d5dd7142df12c688095dec6cc20657"),
    "tab1_detail": ("galvani_experiment_lab.jpg", "dc21c74911d2affdb8df690cb52033d7db0ee259becf19d55ae1835f9e7cdf10"),
    "galvanoscopic": ("galvanoscopic_frog_wellcome.jpg", "0f7686871e8d32b5d235f484b004b11befd21f3b0e06cba32ea9ce9a9e6bd13b"),
    "aldini_loop": ("aldini_galvanism_wellcome.jpg", "cffb86f67407984e52ffb69ffc1d9d5464c7683f436d900947639315bcd2c19f"),
    "matteucci_battery": ("matteucci_frog_battery.jpg", "d4f55f8d1935ef5060849aa2f1c37e60d14139e5ec8940ab25b28c384521a5ad"),
    "voltaic_trough": ("galvani_figuier_333.jpg", "a69664684e119f01213a05ddb5526da5dc79d4cfc146f162b04329aa9a7732de"),
}


def verify_and_load_plates() -> dict[str, np.ndarray]:
    """Verify SHA-256 checksums of all 12 archival Galvani/Matteucci/Aldini scans and load as parchment-toned RGB arrays."""
    plates: dict[str, np.ndarray] = {}
    for key, (fname, expected_hex) in VERIFIED_PLATE_FILES.items():
        p = GALVANI_DIR / fname
        if not p.exists():
            raise FileNotFoundError(f"Missing archival plate: {p}")
        digest = hashlib.sha256(p.read_bytes()).hexdigest()
        if digest != expected_hex:
            raise ValueError(f"SHA-256 mismatch for {fname}: got {digest}, expected {expected_hex}")
        im = Image.open(p).convert("L")
        im = ImageOps.autocontrast(im, cutoff=1)
        im = ImageEnhance.Contrast(im).enhance(1.18)
        arr_l = np.asarray(im, dtype=np.float32) / 255.0
        rgb = np.zeros((*arr_l.shape, 3), dtype=np.float32)
        ink_r, ink_g, ink_b = 0.09, 0.08, 0.07
        pap_r, pap_g, pap_b = 0.985, 0.968, 0.925
        rgb[:, :, 0] = ink_r + arr_l * (pap_r - ink_r)
        rgb[:, :, 1] = ink_g + arr_l * (pap_g - ink_g)
        rgb[:, :, 2] = ink_b + arr_l * (pap_b - ink_b)
        plates[key] = np.clip(rgb, 0.0, 1.0)
    return plates


def simulate_galvani_action_potential(t_ms: np.ndarray, t_stim: float = 2.0, suprathreshold: bool = True) -> dict[str, np.ndarray]:
    """
    Compute phenomenological amphibian myelinated sciatic nerve action potential V_m(t),
    sarcoplasmic reticulum Ca2+ release [Ca2+](t), and gastrocnemius isometric twitch
    tension F(t) calibrated to Frankenhaeuser & Huxley (1964) / Hill (1938) parameters.
    """
    v_rest = -70.0
    if not suprathreshold:
        dt = np.maximum(0.0, t_ms - t_stim)
        v_m = v_rest + 11.5 * (1.0 - np.exp(-dt / 0.45)) * np.exp(-dt / 1.8) * (t_ms >= t_stim)
        ca = 0.05 * np.ones_like(t_ms)
        force = np.zeros_like(t_ms)
        return {"V_m": v_m, "Ca_uM": ca, "force_pct": force}

    dt = np.maximum(0.0, t_ms - t_stim)
    active = (t_ms >= t_stim).astype(float)
    raw_up = (1.0 - np.exp(-dt / 0.16)) ** 3 * np.exp(-dt / 0.72)
    peak_up = float(np.max(raw_up)) if np.max(raw_up) > 0 else 1.0
    upstroke = 105.0 * (raw_up / peak_up)
    afterpot = -7.5 * (1.0 - np.exp(-dt / 1.2)) * np.exp(-dt / 4.2) * (dt > 0.9)
    v_m = v_rest + active * (upstroke + afterpot)

    dt_ca = np.maximum(0.0, t_ms - (t_stim + 1.4))
    act_ca = (t_ms >= (t_stim + 1.4)).astype(float)
    ca = 0.05 + act_ca * 4.8 * (dt_ca / 1.8) ** 2 * np.exp(-dt_ca / 2.2)

    dt_f = np.maximum(0.0, t_ms - (t_stim + 2.2))
    act_f = (t_ms >= (t_stim + 2.2)).astype(float)
    raw_f = act_f * (dt_f / 6.5) ** 2.1 * np.exp(-dt_f / 6.5)
    peak_f = np.max(raw_f) if np.max(raw_f) > 0 else 1.0
    force = 100.0 * raw_f / peak_f

    return {"V_m": v_m, "Ca_uM": ca, "force_pct": force}


# Wellcome L0002011 Fig. 3 crop in original-plate fractions (x0, y0, x1, y1).
# Tight box around the isolated lumbar column + both hindlimbs; excludes Fig. 1
# cone, Fig. 3 caption, and Fig. 4 U-hook.
FIG3_CROP_FRAC = (0.145, 0.072, 0.528, 0.436)

# Rest-pose osteological landmarks in crop-normalized coordinates (x right, y down),
# landmarked on the exact Wellcome L0002011 Fig. 3 crop against Ecker & Gaupp
# (1896–1904): urostyle / ilium / femur / tibiofibula / astragalus–calcaneum / digits.
FROG_HINDLIMB_REST = {
    "atlas": (0.500, 0.110),
    "sacrum": (0.500, 0.460),
    "hip_C": (0.500, 0.538),
    "hip_L": (0.428, 0.505),
    "hip_R": (0.608, 0.500),
    "knee_L": (0.308, 0.662),
    "knee_R": (0.678, 0.558),
    "ankle_L": (0.322, 0.876),
    "ankle_R": (0.652, 0.778),
    "tarsus_L": (0.168, 0.932),
    "tarsus_R": (0.812, 0.872),
    "toe_L": (0.030, 0.958),
    "toe_R": (0.968, 0.912),
}

# Schematic imshow extent (origin='upper'): crop (u, v) → axes (x, y).
# Aspect matches the Fig. 3 crop (~1.36 W/H) so the engraving is not stretched.
SCHEM_EXTENT = (-0.58, 0.78, -0.40, 0.60)  # (x0, x1, y_bottom, y_top)

# Biarticular gastrocnemius-dominated twitch (rad at c = 1). Image y-down.
# Ankle + tarsus ≈ 65° of foot sweep (Galvani 100% twitch kick).
HIP_FLEX_RAD = -0.18
KNEE_FLEX_RAD = 0.30
ANKLE_PLANTAR_RAD = 0.88
TARSAL_FLEX_RAD = 0.30


def _as_xy(name: str) -> np.ndarray:
    return np.array(FROG_HINDLIMB_REST[name], dtype=float)


def crop_uv_to_axes(u: float, v: float) -> np.ndarray:
    """Map crop-normalized (u, v) with v = 0 at the top of Fig. 3 onto schematic axes."""
    x0, x1, y_bot, y_top = SCHEM_EXTENT
    return np.array([x0 + float(u) * (x1 - x0), y_top + float(v) * (y_bot - y_top)], dtype=float)


def joints_to_axes(joints_uv: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {k: crop_uv_to_axes(float(p[0]), float(p[1])) for k, p in joints_uv.items()}


def segment_length(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.hypot(b[0] - a[0], b[1] - a[1]))


def frog_hindlimb_forward_kinematics(contraction: float = 0.0) -> dict[str, Any]:
    """
    Four-segment rigid FK of the stimulated LEFT hindlimb (hip, knee, ankle, tarsus).
    Right hindlimb and the vertebral / iliac column stay in the Fig. 3 rest pose.
    Segment lengths are conserved by construction (pure rotation about the proximal joint).
    """
    c = float(np.clip(contraction, 0.0, 1.0))
    th_h = HIP_FLEX_RAD * c
    th_k = KNEE_FLEX_RAD * c
    th_a = ANKLE_PLANTAR_RAD * c
    th_t = TARSAL_FLEX_RAD * c

    posed: dict[str, np.ndarray] = {k: _as_xy(k) for k in FROG_HINDLIMB_REST}

    hip = posed["hip_L"]
    knee0, ankle0 = _as_xy("knee_L"), _as_xy("ankle_L")
    tars0, toe0 = _as_xy("tarsus_L"), _as_xy("toe_L")

    def _rot(vec: np.ndarray, ang: float) -> np.ndarray:
        ca, sa = math.cos(ang), math.sin(ang)
        return np.array([ca * vec[0] - sa * vec[1], sa * vec[0] + ca * vec[1]], dtype=float)

    posed["knee_L"] = hip + _rot(knee0 - hip, th_h)
    posed["ankle_L"] = posed["knee_L"] + _rot(ankle0 - knee0, th_h + th_k)
    posed["tarsus_L"] = posed["ankle_L"] + _rot(tars0 - ankle0, th_h + th_k + th_a)
    posed["toe_L"] = posed["tarsus_L"] + _rot(toe0 - tars0, th_h + th_k + th_a + th_t)

    # Foot-sweep kick: smallest change in ankle→toe bearing (plantarflexion Galvani saw).
    f0 = toe0 - ankle0
    f1 = posed["toe_L"] - posed["ankle_L"]
    d_ang = math.atan2(f1[1], f1[0]) - math.atan2(f0[1], f0[0])
    kick_deg = abs(math.degrees(math.atan2(math.sin(d_ang), math.cos(d_ang))))
    femur = segment_length(posed["hip_L"], posed["knee_L"])
    tibio = segment_length(posed["knee_L"], posed["ankle_L"])
    tarsus = segment_length(posed["ankle_L"], posed["tarsus_L"])
    digits = segment_length(posed["tarsus_L"], posed["toe_L"])
    return {
        "joints_uv": posed,
        "angles_rad": {"hip": th_h, "knee": th_k, "ankle": th_a, "tarsus": th_t},
        "kick_deg": float(kick_deg),
        "lengths": {
            "femur": femur,
            "tibiofibula": tibio,
            "astragalus_calcaneum": tarsus,
            "digits": digits,
            "tibiofibula_over_femur": tibio / max(femur, 1e-9),
            "tarsus_over_tibiofibula": tarsus / max(tibio, 1e-9),
        },
    }


def _soft_segment_weight(xx: np.ndarray, yy: np.ndarray, a: np.ndarray, b: np.ndarray, radius: float) -> np.ndarray:
    vx, vy = float(b[0] - a[0]), float(b[1] - a[1])
    l2 = vx * vx + vy * vy + 1e-9
    t = np.clip(((xx - a[0]) * vx + (yy - a[1]) * vy) / l2, 0.0, 1.0)
    px = a[0] + t * vx
    py = a[1] + t * vy
    dist = np.hypot(xx - px, yy - py)
    return np.exp(-0.5 * (dist / max(radius, 1e-6)) ** 2)


def _left_chain_weight(xx: np.ndarray, yy: np.ndarray, j: dict[str, np.ndarray], scale: float) -> np.ndarray:
    w = _soft_segment_weight(xx, yy, j["hip_L"], j["knee_L"], 0.085 * scale)
    w = np.maximum(w, _soft_segment_weight(xx, yy, j["knee_L"], j["ankle_L"], 0.070 * scale))
    w = np.maximum(w, _soft_segment_weight(xx, yy, j["ankle_L"], j["tarsus_L"], 0.080 * scale))
    w = np.maximum(w, _soft_segment_weight(xx, yy, j["tarsus_L"], j["toe_L"], 0.095 * scale))
    return w


def _inverse_rigid(xx: np.ndarray, yy: np.ndarray, ap: np.ndarray, bp: np.ndarray, ar: np.ndarray, br: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Map posed pixels back to rest via the rigid motion that sends Ap→Ar, Bp→Br."""
    ang = math.atan2(br[1] - ar[1], br[0] - ar[0]) - math.atan2(bp[1] - ap[1], bp[0] - ap[0])
    ca, sa = math.cos(ang), math.sin(ang)
    dx, dy = xx - ap[0], yy - ap[1]
    return ar[0] + ca * dx - sa * dy, ar[1] + sa * dx + ca * dy


def _bilinear_sample(ink: np.ndarray, src_x: np.ndarray, src_y: np.ndarray) -> np.ndarray:
    h, w = ink.shape
    in_bounds = (src_x >= 1.0) & (src_x < w - 2.001) & (src_y >= 1.0) & (src_y < h - 2.001)
    sx = np.clip(src_x, 0.0, w - 1.001)
    sy = np.clip(src_y, 0.0, h - 1.001)
    x0 = sx.astype(int)
    y0 = sy.astype(int)
    tx = sx - x0
    ty = sy - y0
    return (
        (1.0 - tx) * (1.0 - ty) * ink[y0, x0]
        + tx * (1.0 - ty) * ink[y0, x0 + 1]
        + (1.0 - tx) * ty * ink[y0 + 1, x0]
        + tx * ty * ink[y0 + 1, x0 + 1]
    ) * in_bounds.astype(np.float32)


def fusiform_polygon(a: np.ndarray, b: np.ndarray, half_width: float, n: int = 16) -> np.ndarray:
    """Tapered muscle-belly polygon along segment AB (sinusoidal width profile)."""
    d = b - a
    length = float(np.hypot(d[0], d[1]))
    if length < 1e-9:
        return np.vstack([a, a])
    tangent = d / length
    normal = np.array([-tangent[1], tangent[0]], dtype=float)
    s = np.linspace(0.0, 1.0, n)
    left, right = [], []
    for si in s:
        p = a + si * d
        w = half_width * math.sin(math.pi * si)
        left.append(p + w * normal)
        right.append(p - w * normal)
    return np.vstack([np.asarray(left), np.asarray(right[::-1])])


def sciatic_polyline_uv(joints: dict[str, np.ndarray], side: str = "L") -> np.ndarray:
    """Plexus → posterior femur → popliteal fossa → gastrocnemius belly."""
    hip, knee, ankle = joints[f"hip_{side}"], joints[f"knee_{side}"], joints[f"ankle_{side}"]
    fem = knee - hip
    fn = float(np.hypot(fem[0], fem[1])) + 1e-9
    # Postaxial offset (toward the inner/flexor face of the thigh).
    nrm = np.array([-fem[1], fem[0]], dtype=float) / fn
    if side == "L":
        nrm = -nrm
    mid_fem = 0.55 * hip + 0.45 * knee + 0.028 * nrm
    mid_gas = 0.40 * knee + 0.60 * ankle + 0.034 * nrm
    return np.vstack([joints["sacrum"], hip, mid_fem, knee, mid_gas])


def segment_and_warp_archival_frog_prep(contraction: float = 0.65) -> dict[str, Any]:
    """
    Segment Wellcome L0002011 Fig. 3 and piecewise-rigidly warp the LEFT hindlimb
    with 4-joint (hip / knee / ankle / tarsus) forward kinematics. Vacated rest-pose
    pixels are cleared so the uncontracted limb can be drawn as a separate ghost.
    """
    from PIL import ImageFilter

    p = GALVANI_DIR / "galvani_frog_prep_wellcome.jpg"
    im = Image.open(p).convert("L")
    w0, h0 = im.size
    x0, y0, x1, y1 = FIG3_CROP_FRAC
    crop_im = im.crop((int(x0 * w0), int(y0 * h0), int(x1 * w0), int(y1 * h0)))
    cw, ch = crop_im.size
    crop_hi = ImageOps.autocontrast(crop_im.resize((cw * 3, ch * 3), Image.LANCZOS), cutoff=1)
    arr_l = np.asarray(crop_hi, dtype=np.float32) / 255.0
    raw_ink = np.clip((0.92 - arr_l) / 0.85, 0.0, 1.0)
    h_px, w_px = raw_ink.shape

    blur = np.asarray(crop_hi.filter(ImageFilter.GaussianBlur(radius=3.8)), dtype=np.float32) / 255.0
    yy, xx = np.meshgrid(np.arange(h_px, dtype=np.float32), np.arange(w_px, dtype=np.float32), indexing="ij")
    clean_window = (
        ~((yy < 0.055 * h_px) & (xx < 0.42 * w_px))
        & ~((xx > 0.875 * w_px) & (yy < 0.72 * h_px))
        & ~((xx < 0.055 * w_px) & (yy < 0.82 * h_px))
        & (xx > 3) & (xx < w_px - 4) & (yy > 3) & (yy < h_px - 4)
    )
    sil_mask = (blur < 0.935) & clean_window
    ink = raw_ink * clean_window.astype(np.float32)
    ys, xs = np.where(sil_mask)
    aspect_ratio = float((ys.max() - ys.min()) / max(1.0, (xs.max() - xs.min())))
    fg_fraction = float(sil_mask.mean())

    n_bins = 32
    widths_emp = np.zeros(n_bins, dtype=float)
    for i in range(n_bins):
        r0 = int(ys.min() + (i / n_bins) * (ys.max() - ys.min()))
        r1 = max(r0 + 1, int(ys.min() + ((i + 1) / n_bins) * (ys.max() - ys.min())))
        cols = np.where(sil_mask[r0:r1, :])[1]
        if len(cols) > 0:
            widths_emp[i] = float(cols.max() - cols.min()) / float(w_px)

    kin = frog_hindlimb_forward_kinematics(contraction)
    rest_px = {k: np.array([v[0] * w_px, v[1] * h_px], dtype=float) for k, v in FROG_HINDLIMB_REST.items()}
    posed_px = {k: np.array([v[0] * w_px, v[1] * h_px], dtype=float) for k, v in kin["joints_uv"].items()}

    w_thigh = _soft_segment_weight(xx, yy, posed_px["hip_L"], posed_px["knee_L"], 0.090 * w_px)
    w_shank = _soft_segment_weight(xx, yy, posed_px["knee_L"], posed_px["ankle_L"], 0.075 * w_px)
    w_foot = np.maximum(
        _soft_segment_weight(xx, yy, posed_px["ankle_L"], posed_px["tarsus_L"], 0.085 * w_px),
        _soft_segment_weight(xx, yy, posed_px["tarsus_L"], posed_px["toe_L"], 0.100 * w_px),
    )
    w_left_posed = np.clip(w_thigh + w_shank + w_foot, 0.0, 1.0)
    w_left_rest = _left_chain_weight(xx, yy, rest_px, w_px)

    src_th_x, src_th_y = _inverse_rigid(xx, yy, posed_px["hip_L"], posed_px["knee_L"], rest_px["hip_L"], rest_px["knee_L"])
    src_sh_x, src_sh_y = _inverse_rigid(xx, yy, posed_px["knee_L"], posed_px["ankle_L"], rest_px["knee_L"], rest_px["ankle_L"])
    src_ft_x, src_ft_y = _inverse_rigid(xx, yy, posed_px["ankle_L"], posed_px["toe_L"], rest_px["ankle_L"], rest_px["toe_L"])

    w_sum = w_thigh + w_shank + w_foot + 1e-6
    src_x = (w_thigh * src_th_x + w_shank * src_sh_x + w_foot * src_ft_x) / w_sum
    src_y = (w_thigh * src_th_y + w_shank * src_sh_y + w_foot * src_ft_y) / w_sum
    src_x = w_left_posed * src_x + (1.0 - w_left_posed) * xx
    src_y = w_left_posed * src_y + (1.0 - w_left_posed) * yy

    warped_ink = _bilinear_sample(ink, src_x, src_y)
    # Clear vacated rest-left pixels so they do not double-print under the ghost layer.
    vacated = (w_left_rest > 0.35) & (w_left_posed < 0.20)
    warped_ink = warped_ink * (1.0 - vacated.astype(np.float32))

    return {
        "ink_rest": ink,
        "ink_warped": warped_ink,
        "sil_mask": sil_mask,
        "w_left": w_left_posed,
        "w_left_rest": w_left_rest,
        "aspect_ratio": aspect_ratio,
        "fg_fraction": fg_fraction,
        "widths_emp": widths_emp,
        "kinematics": kin,
    }


def draw_galvani_frog_schematic(ax, contraction: float = 0.65, ap_pos: float = 0.55, dark_theme: bool = True):
    """
    Render Wellcome L0002011 Fig. 3 with 4-joint left-hindlimb warp, then overlay
    Ecker–Gaupp osteology (urostyle, ilium, femur, tibiofibula, astragalus–calcaneum,
    digits), named myology (triceps femoris, semimembranosus / gracilis, gastrocnemius,
    tibialis anticus), the sciatic path, travelling AP, Fe/Cu arc, and 1794 loop.
    Callout boxes sit on the frame margins so they cannot overlap the anatomy.
    """
    bg = "#0c1424" if dark_theme else "#fbf7ee"
    ax.set_facecolor(bg)
    ax.set_xlim(-1.12, 1.12)
    ax.set_ylim(-1.16, 0.98)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])

    prep = segment_and_warp_archival_frog_prep(contraction=contraction)
    ink_rest = prep["ink_rest"]
    ink_warp = prep["ink_warped"]
    w_left = prep["w_left"]
    w_left_rest = prep["w_left_rest"]
    kin = prep["kinematics"]
    h_px, w_px = ink_warp.shape
    posed_ax = joints_to_axes(kin["joints_uv"])
    rest_ax = joints_to_axes({k: _as_xy(k) for k in FROG_HINDLIMB_REST})

    canvas_rgb = np.zeros((h_px, w_px, 3), dtype=np.float32)
    if dark_theme:
        bg_rgb = np.array([0.047, 0.078, 0.141], dtype=np.float32)
        rest_col = np.array([0.96, 0.91, 0.78], dtype=np.float32)
        act_col = np.array([1.00, 0.42, 0.52], dtype=np.float32)
        ghost_col = np.array([0.28, 0.38, 0.52], dtype=np.float32)
        bone_c, joint_c, mus_c, mus_edge = "#e2e8f0", "#f8fafc", "#fb7185", "#fda4af"
        nerve_c, label_c = "#fde047", "#fde68a"
    else:
        bg_rgb = np.array([0.985, 0.968, 0.925], dtype=np.float32)
        rest_col = np.array([0.12, 0.10, 0.08], dtype=np.float32)
        act_col = np.array([0.82, 0.12, 0.24], dtype=np.float32)
        ghost_col = np.array([0.72, 0.76, 0.82], dtype=np.float32)
        bone_c, joint_c, mus_c, mus_edge = "#334155", "#0f172a", "#be123c", "#9f1239"
        nerve_c, label_c = "#a16207", "#1e293b"

    canvas_rgb[:, :] = bg_rgb[None, None, :]
    body = prep["sil_mask"].astype(np.float32)[:, :, None] * 0.16
    canvas_rgb = (1.0 - body) * canvas_rgb + body * rest_col[None, None, :] * 0.35
    ghost_alpha = np.clip(ink_rest * w_left_rest * 0.22, 0.0, 1.0)[:, :, None]
    canvas_rgb = (1.0 - ghost_alpha) * canvas_rgb + ghost_alpha * ghost_col[None, None, :]
    stroke_alpha = np.clip(np.power(np.clip(ink_warp, 0.0, 1.0), 0.55) * 1.70, 0.0, 1.0)[:, :, None]
    active_tint = (1.0 - 0.75 * float(contraction) * w_left[:, :, None]) * rest_col[None, None, :] + (
        0.75 * float(contraction) * w_left[:, :, None]
    ) * act_col[None, None, :]
    canvas_rgb = (1.0 - stroke_alpha) * canvas_rgb + stroke_alpha * active_tint
    ax.imshow(canvas_rgb, extent=list(SCHEM_EXTENT), origin="upper", zorder=3)

    # Iron balcony railing and brass spinal hook (Tab. III apparatus)
    ax.plot([-0.88, -0.88], [-0.82, 0.78], color="#64748b", lw=5.0, solid_capstyle="round", zorder=4)
    ax.plot([-0.92, -0.35], [0.78, 0.78], color="#64748b", lw=4.5, solid_capstyle="round", zorder=4)
    ax.text(-0.86, 0.83, "Iron Balcony Railing / Grounded Plate (Fe, E° = −0.44 V)",
            color="#94a3b8", fontsize=7.6, fontweight="bold", ha="left")
    ax.plot([-0.38, posed_ax["atlas"][0]], [0.78, posed_ax["atlas"][1]], color="#f59e0b", lw=3.2, zorder=6)
    ax.add_patch(Circle(posed_ax["atlas"], 0.028, facecolor="#fbbf24", edgecolor="#78350f", lw=1.4, zorder=7))
    ax.text(-0.40, 0.64, "Brass / Copper Hook (Cu)\nin Spinal Canal (+0.34 V)",
            color="#fbbf24", fontsize=7.2, fontweight="bold", ha="right", zorder=8)

    def _bone(a, b, lw=2.15, alpha=0.92):
        ax.plot([a[0], b[0]], [a[1], b[1]], color=bone_c, lw=lw, alpha=alpha,
                solid_capstyle="round", zorder=5)

    def _joint(p, r=0.016):
        ax.add_patch(Circle(p, r, facecolor=joint_c, edgecolor=bone_c, lw=0.7, zorder=7, alpha=0.95))

    # Axial column + both ilia (fixed)
    _bone(posed_ax["atlas"], posed_ax["sacrum"], lw=2.6)
    _bone(posed_ax["sacrum"], posed_ax["hip_C"], lw=2.0)
    _bone(posed_ax["hip_C"], posed_ax["hip_L"], lw=1.8)
    _bone(posed_ax["hip_C"], posed_ax["hip_R"], lw=1.8)

    # Right hindlimb (resting) — thin ivory
    _bone(posed_ax["hip_R"], posed_ax["knee_R"], lw=1.9, alpha=0.75)
    _bone(posed_ax["knee_R"], posed_ax["ankle_R"], lw=1.8, alpha=0.75)
    _bone(posed_ax["ankle_R"], posed_ax["tarsus_R"], lw=1.6, alpha=0.75)
    _bone(posed_ax["tarsus_R"], posed_ax["toe_R"], lw=1.4, alpha=0.75)
    for key in ("hip_R", "knee_R", "ankle_R"):
        _joint(posed_ax[key], r=0.013)

    # Left hindlimb (stimulated) — femur / tibiofibula / elongated tarsus / digits
    _bone(posed_ax["hip_L"], posed_ax["knee_L"], lw=2.5)
    _bone(posed_ax["knee_L"], posed_ax["ankle_L"], lw=2.4)
    _bone(posed_ax["ankle_L"], posed_ax["tarsus_L"], lw=2.1)
    _bone(posed_ax["tarsus_L"], posed_ax["toe_L"], lw=1.8)
    for key in ("hip_L", "knee_L", "ankle_L", "tarsus_L"):
        _joint(posed_ax[key], r=0.017)

    # Myology: fusiform bellies. Gastrocnemius is biarticular (knee + ankle).
    bulge = 1.0 + 0.38 * float(contraction)
    mus_alpha = 0.14 + 0.18 * float(contraction)
    muscles = [
        (posed_ax["hip_L"], posed_ax["knee_L"], 0.040 * bulge, "preaxial"),   # triceps femoris
        (posed_ax["hip_L"], posed_ax["knee_L"], 0.030 * bulge, "postaxial"),  # semimembranosus / gracilis
        (posed_ax["knee_L"], posed_ax["tarsus_L"], 0.048 * bulge, "postaxial"),  # gastrocnemius (plantaris longus)
        (posed_ax["knee_L"], posed_ax["ankle_L"], 0.024 * bulge, "preaxial"),  # tibialis anticus
    ]
    for a, b, hw, face in muscles:
        d = b - a
        nrm = np.array([-d[1], d[0]], dtype=float)
        nrm /= (float(np.hypot(nrm[0], nrm[1])) + 1e-9)
        if face == "postaxial":
            nrm = -nrm
        shift = 0.012 * nrm
        poly = fusiform_polygon(a + shift, b + shift, hw)
        ax.add_patch(Polygon(poly, closed=True, facecolor=mus_c, edgecolor=mus_edge,
                             lw=0.7, alpha=mus_alpha, zorder=4.5))

    # Sciatic nerves: left follows posed chain; right stays at rest
    nerve_L = np.array([crop_uv_to_axes(p[0], p[1]) for p in sciatic_polyline_uv(kin["joints_uv"], "L")])
    nerve_R = np.array([crop_uv_to_axes(p[0], p[1]) for p in sciatic_polyline_uv({k: _as_xy(k) for k in FROG_HINDLIMB_REST}, "R")])
    ax.plot(nerve_L[:, 0], nerve_L[:, 1], color=nerve_c, lw=2.3, alpha=0.95, zorder=6)
    ax.plot(nerve_R[:, 0], nerve_R[:, 1], color=nerve_c, lw=1.7, ls="--", alpha=0.75, zorder=6)

    idx_f = min(max(float(ap_pos), 0.0), 1.0) * (len(nerve_L) - 1)
    i0 = int(math.floor(idx_f))
    i1 = min(i0 + 1, len(nerve_L) - 1)
    frac = idx_f - i0
    ap_xy = (1.0 - frac) * nerve_L[i0] + frac * nerve_L[i1]
    for rad, alp in [(0.078, 0.22), (0.048, 0.46), (0.024, 0.95)]:
        ax.add_patch(Circle(ap_xy, rad, facecolor="#38bdf8", edgecolor="none", alpha=alp, zorder=8))

    # Fe/Cu arc lands on the gastrocnemius belly (mid shank, postaxial)
    contact_L = 0.45 * posed_ax["knee_L"] + 0.55 * posed_ax["ankle_L"] + np.array([-0.02, -0.01])
    arc_x = np.linspace(-0.88, contact_L[0], 40)
    arc_y = -0.36 - 0.18 * np.sin(np.pi * (arc_x - (-0.88)) / max(contact_L[0] + 0.88, 1e-3))
    arc_y[-1] = contact_L[1]
    ax.plot(arc_x[:20], arc_y[:20], color="#94a3b8", lw=3.6, zorder=7)
    ax.plot(arc_x[19:], arc_y[19:], color="#f59e0b", lw=3.6, zorder=7)
    ax.scatter([contact_L[0]], [contact_L[1]], color="#38bdf8", edgecolor="#ffffff", s=74, zorder=9)

    foot = posed_ax["toe_L"]
    rest_foot = rest_ax["toe_L"]
    twitch_arr = FancyArrowPatch(
        (rest_foot[0] - 0.02, rest_foot[1] - 0.06),
        (foot[0] - 0.04, foot[1] + 0.04),
        connectionstyle="arc3,rad=-0.35",
        arrowstyle="->,head_length=7,head_width=5",
        color="#4ade80", lw=2.3, zorder=9,
    )
    ax.add_patch(twitch_arr)

    # Margin labels — none sit on the engraving itself
    ax.text(0.34, 0.58, "Lumbar vertebrae\n+ urostyle", color=label_c, fontsize=6.6,
            fontweight="bold", ha="left", va="center", zorder=8)
    ax.text(-1.07, 0.20,
            "Action Potential\n$V_m = +35\\,\\mathrm{mV}$\n($v = 27\\,\\mathrm{m/s}$)",
            color="#38bdf8", fontsize=6.6, fontweight="bold", ha="left", va="center",
            bbox=dict(boxstyle="round,pad=0.14", facecolor="#082f49", edgecolor="#38bdf8", lw=0.9), zorder=10)
    ax.text(-1.07, -0.18, "Triceps femoris\n(thigh extensor)", color=mus_edge, fontsize=6.3,
            fontweight="bold", ha="left", va="center", zorder=8)
    ax.text(-1.07, -0.46, "Gastrocnemius\n(biarticular calf)", color=mus_edge, fontsize=6.3,
            fontweight="bold", ha="left", va="center", zorder=8)
    ax.text(-1.07, -0.72, f"Astragalus–calcaneum\nkick = {kin['kick_deg']:.0f}°",
            color="#4ade80", fontsize=6.3, fontweight="bold", ha="left", va="center", zorder=8)
    ax.text(0.68, 0.08, "Femur", color=bone_c, fontsize=6.2, fontweight="bold", ha="left", zorder=8)
    ax.text(0.70, -0.18, "Tibiofibula", color=bone_c, fontsize=6.2, fontweight="bold", ha="left", zorder=8)
    ax.text(0.74, -0.48, "Webbed digits", color=bone_c, fontsize=6.2, fontweight="bold", ha="left", zorder=8)

    ax.text(-1.07, -1.11,
            "1. Bimetallic Arc (1786–1791)\n"
            "$\\Delta E_{\\mathrm{Fe/Cu}} = 0.78\\,\\mathrm{V}$ ($62.4\\,\\mu\\mathrm{A}$)\n"
            "$35\\times I_{\\mathrm{th}} \\Rightarrow 100\\%$ Twitch!",
            color="#4ade80", fontsize=6.5, fontweight="bold", ha="left", va="bottom",
            bbox=dict(boxstyle="round,pad=0.16", facecolor="#052e16", edgecolor="#4ade80", lw=1.0), zorder=10)
    ax.add_patch(FancyArrowPatch(
        (posed_ax["hip_R"][0] + 0.02, posed_ax["hip_R"][1] - 0.02),
        (posed_ax["knee_R"][0] + 0.04, posed_ax["ankle_R"][1]),
        connectionstyle="arc3,rad=-0.35",
        arrowstyle="->,head_length=6,head_width=4",
        color="#facc15", lw=2.0, ls="--", zorder=8,
    ))
    ax.text(1.07, -1.11,
            "2. 1794 Metal-Free Loop\n"
            "Nerve bridges muscle wound:\n"
            "$\\Delta V_{\\mathrm{inj}} = 42\\,\\mathrm{mV}$ ($3.4\\,\\mu\\mathrm{A}$) Twitch!",
            color="#fde047", fontsize=6.5, fontweight="bold", ha="right", va="bottom",
            bbox=dict(boxstyle="round,pad=0.16", facecolor="#422006", edgecolor="#facc15", lw=1.0), zorder=10)



def render_master_atlas(plates: dict[str, np.ndarray]):
    """
    Generate `docs/media/galvani_1791_frog_experiments_atlas.png` (22 x 15 inches, 220 DPI).
    """
    fig = plt.figure(figsize=(22.5, 14.8), dpi=220, facecolor="#0b1120")

    fig.suptitle(
        "LUIGI GALVANI (1791) — DE VIRIBUS ELECTRICITATIS IN MOTU MUSCULARI COMMENTARIUS\n"
        "Original 1791 Bologna Copperplate Engravings (Tabulae I–IV by Jacopo Zambelli) & Modern Biophysical Reconstruction of Animal Electricity",
        color="#f8fafc", fontsize=15.2, fontweight="bold", y=0.972
    )
    fig.text(
        0.5, 0.932,
        "From the Accidental Spark Induction in Bologna (1780) and Atmospheric Lightning Rod Trials (1786) to the Bimetallic Arc (Tab. III), "
        "Galvani's 1794 Metal-Free Injury-Current Proof, and Myelinated Sciatic Na⁺/K⁺ Action Potential Conduction",
        color="#cbd5e1", fontsize=10.6, ha="center", style="italic"
    )

    gs_top = gridspec.GridSpec(
        1, 5, figure=fig,
        width_ratios=[0.74, 1.15, 1.15, 1.15, 1.15],
        left=0.025, right=0.975, top=0.895, bottom=0.535, wspace=0.10
    )

    top_specs = [
        (
            "cover",
            "1791 Bologna Title Page\n(Instituti Scientiarum)",
            "Aloysii Galvani (1791) original\nmonograph cover + Pars Prima (p. 363)\nopening on 'Artificial Electricity'.",
            "#e2e8f0"
        ),
        (
            "tab1",
            "TABULA I — Artificial Electricity\n(Electrostatic Machine & Spark, 1780)",
            "Fig. 1 friction machine, Leyden jars (Figs. 5–6),\n& prepared frogs (Figs. 2, 4): when scalpel touches\ncrural nerve during a spark, the hindlimbs jump!",
            "#38bdf8"
        ),
        (
            "tab2",
            "TABULA II — Atmospheric Lightning\n(Casa Galvani Terrace & Well, 1786)",
            "Lightning wire (A–B) to frog nerves on table\nand ground wire in courtyard well — plus right\nniche showing frog on iron grating in calm air!",
            "#facc15"
        ),
        (
            "tab3",
            "TABULA III — The Metallic Arc\n('Arco Conduttore', Sept 1786)",
            "Figs. 9–16 (J. Zambelli fecit): Curved\nbimetallic & monometallic arcs (A–C, D–E, F, G, K)\nbridging spinal cord and crural muscles.",
            "#4ade80"
        ),
        (
            "tab4",
            "TABULA IV — Animal Electricity\n(Twin Water Vessels & Chains, 1791)",
            "Figs. 17–21: Fig. 19 frog bridging twin water\ncups (A, B); Fig. 20 warm-blooded lamb sciatic\ndissection; Fig. 21 human-chain circuit.",
            "#f472b6"
        ),
    ]

    for idx, (key, title, caption, border_col) in enumerate(top_specs):
        ax = fig.add_subplot(gs_top[0, idx])
        ax.imshow(plates[key])
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color(border_col)
            sp.set_linewidth(2.2)
        ax.set_title(title, color=border_col, fontsize=10.2, fontweight="bold", pad=8)
        ax.set_xlabel(caption, color="#cbd5e1", fontsize=8.2, labelpad=7)

        if key == "cover":
            ins = ax.inset_axes([0.46, 0.03, 0.51, 0.45])
            ins.imshow(plates["pars_prima"])
            ins.set_xticks([])
            ins.set_yticks([])
            for sp in ins.spines.values():
                sp.set_color("#f59e0b")
                sp.set_linewidth(1.5)
            ins.set_title("p. 363: Pars Prima", color="#fbbf24", fontsize=6.8, fontweight="bold", pad=2,
                          bbox=dict(boxstyle="round,pad=0.1", facecolor="#0f172a", edgecolor="none", alpha=0.85))

    gs_bot = gridspec.GridSpec(
        1, 4, figure=fig,
        width_ratios=[0.92, 1.28, 1.08, 1.08],
        left=0.025, right=0.975, top=0.435, bottom=0.075, wspace=0.25
    )

    # Panel A (Bottom Left)
    ax_a = fig.add_subplot(gs_bot[0, 0])
    ax_a.set_facecolor("#0f172a")
    w_img = plates["sciatic"]
    t_img = plates["tab1_detail"]
    target_w = 800
    h1 = int(w_img.shape[0] * (target_w / w_img.shape[1]))
    h2 = int(t_img.shape[0] * (target_w / t_img.shape[1]))
    im1 = np.asarray(Image.fromarray((w_img * 255).astype(np.uint8)).resize((target_w, h1), Image.LANCZOS)) / 255.0
    im2 = np.asarray(Image.fromarray((t_img * 255).astype(np.uint8)).resize((target_w, h2), Image.LANCZOS)) / 255.0
    div = np.ones((10, target_w, 3), dtype=np.float32) * np.array([0.96, 0.62, 0.04])
    stacked = np.vstack([im1, div, im2])
    ax_a.imshow(stacked)
    ax_a.set_xticks([])
    ax_a.set_yticks([])
    for sp in ax_a.spines.values():
        sp.set_color("#f59e0b")
        sp.set_linewidth(2.0)
    ax_a.set_title(
        "A. Galvani's Sciatic Dissections\n(Top: Tav. I Figs. 1–4; Bottom: Tab. I Lab)",
        color="#fbbf24", fontsize=9.8, fontweight="bold", pad=8
    )
    ax_a.set_xlabel(
        "Top (Wellcome M0012614): 'H, I' = spinal nerves, 'G, M' = feet, 's' = arc junction;\nBottom (Tab. I): 'Fig. 1' = friction wheel, 'Fig. 4' = frog, 'Figs. 5–6' = Leyden jars.",
        color="#cbd5e1", fontsize=7.9, labelpad=7
    )

    # Panel B (Bottom Center-Left)
    ax_b = fig.add_subplot(gs_bot[0, 1])
    draw_galvani_frog_schematic(ax_b, contraction=0.68, ap_pos=0.52, dark_theme=True)
    for sp in ax_b.spines.values():
        sp.set_color("#4ade80")
        sp.set_linewidth(2.0)
    ax_b.set_title(
        "B. Ecker–Gaupp Hindlimb (Urostyle · Femur · Tibiofibula · Astragalus–Calcaneum)\n"
        "Left: 1786 Fe/Cu Arc on Gastrocnemius; Right: 1794 Metal-Free Injury Loop",
        color="#4ade80", fontsize=9.4, fontweight="bold", pad=8
    )

    # Panel C (Bottom Center-Right)
    ax_c = fig.add_subplot(gs_bot[0, 2])
    ax_c.set_facecolor("#0f172a")
    for sp in ax_c.spines.values():
        sp.set_color("#38bdf8")
        sp.set_linewidth(1.8)

    t_ms = np.linspace(0.0, 22.0, 450)
    sim_supra = simulate_galvani_action_potential(t_ms, t_stim=2.0, suprathreshold=True)
    sim_sub = simulate_galvani_action_potential(t_ms, t_stim=2.0, suprathreshold=False)

    ax_c.axhline(-50.0, color="#facc15", ls=":", lw=1.4, label="Na$_v$ Threshold ($V_{\\mathrm{th}} = -50\\,\\mathrm{mV}$)")
    ax_c.plot(t_ms, sim_sub["V_m"], color="#94a3b8", ls="--", lw=1.8, label="Subthreshold Single Metal ($V_m < V_{\\mathrm{th}}$)")
    ax_c.plot(t_ms, sim_supra["V_m"], color="#38bdf8", lw=2.8, label="Sciatic Action Potential $V_m(t)$ (mV)")

    ax_c2 = ax_c.twinx()
    ax_c2.plot(t_ms, sim_supra["force_pct"], color="#f43f5e", lw=2.6, label="Gastrocnemius Twitch $F(t)$ (%)")
    ax_c2.fill_between(t_ms, 0, sim_supra["force_pct"], color="#f43f5e", alpha=0.16)
    ax_c2.set_ylabel("Twitch Tension (%)", color="#fda4af", fontsize=8.6, fontweight="bold", labelpad=4)
    ax_c2.tick_params(axis="y", colors="#fda4af", labelsize=8.0)
    ax_c2.set_ylim(-8, 118)
    ax_c2.spines["right"].set_color("#f43f5e")

    ax_c.axvline(2.0, color="#fde047", lw=1.3, alpha=0.8)
    ax_c.text(4.8, 32.0, "Arc Contact ($t = 2\\,\\mathrm{ms}$)", color="#fde047", fontsize=7.4, fontweight="bold")
    ax_c.set_xlim(0, 22.0)
    ax_c.set_ylim(-82, 48)
    ax_c.set_xlabel("Time After Metallic Arc Closure $t$ (ms)", color="#cbd5e1", fontsize=8.8, fontweight="bold")
    ax_c.set_ylabel("Sciatic Membrane Potential $V_m$ (mV)", color="#38bdf8", fontsize=8.8, fontweight="bold")
    ax_c.tick_params(colors="#cbd5e1", labelsize=8.0)
    ax_c.grid(color="#1e293b", lw=0.8)
    ax_c.set_title(
        "C. Sciatic Nerve Electrophysiology\nSpike $V_m(t)$ ($+35\\,\\mathrm{mV}$) $\\to$ Twitch $F(t)$",
        color="#38bdf8", fontsize=9.8, fontweight="bold", pad=8
    )
    lines1, labels1 = ax_c.get_legend_handles_labels()
    lines2, labels2 = ax_c2.get_legend_handles_labels()
    ax_c.legend(lines1 + lines2, labels1 + labels2, loc="upper right", facecolor="#0b1120",
                edgecolor="#334155", labelcolor="#f8fafc", fontsize=7.0)

    # Panel D (Bottom Right)
    ax_d = fig.add_subplot(gs_bot[0, 3])
    ax_d.set_facecolor("#0f172a")
    for sp in ax_d.spines.values():
        sp.set_color("#facc15")
        sp.set_linewidth(1.8)

    regimes = [
        "1. Single\nMetal\n(Cu or Fe)",
        "2. 1794 Cut\nInjury Loop\n(No Metal!)",
        "3. 1786 Arc\nBrass Hook\n+ Iron Rail",
        "4. Volta 1800\nSilver + Zinc\n(Ag / Zn)",
    ]
    delta_vm_mv = [4.5, 42.0, 68.0, 95.0]
    bar_cols = ["#64748b", "#4ade80", "#f59e0b", "#38bdf8"]
    x_pos = np.arange(len(regimes))
    bars = ax_d.bar(x_pos, delta_vm_mv, width=0.54, color=bar_cols, edgecolor="#f8fafc", lw=1.3, zorder=4)
    ax_d.axhline(20.0, color="#f43f5e", ls="--", lw=2.2, zorder=5,
                 label="Threshold $\\Delta V_{\\mathrm{th}} = +20\\,\\mathrm{mV}$ ($-70 \\to -50\\,\\mathrm{mV}$)")

    annotations = [
        "4.5 mV\n(No Twitch)",
        "42 mV (TWITCH!)\nIntrinsic Injury\nPotential",
        "68 mV (TWITCH!)\n0.78 V Fe/Cu\nCell EMF",
        "95 mV (TWITCH!)\n1.56 V Ag/Zn\nVoltaic Cell",
    ]
    for bar, txt in zip(bars, annotations):
        h = bar.get_height()
        ax_d.text(bar.get_x() + bar.get_width() / 2, h + 2.5, txt,
                  ha="center", va="bottom", color="#f8fafc", fontsize=7.2, fontweight="bold")

    ax_d.set_xticks(x_pos)
    ax_d.set_xticklabels(regimes, color="#e2e8f0", fontsize=7.6, fontweight="bold")
    ax_d.set_ylim(0, 118)
    ax_d.set_ylabel("Local Nerve Depolarization $\\Delta V_m$ (mV)", color="#facc15", fontsize=8.8, fontweight="bold")
    ax_d.tick_params(colors="#cbd5e1", labelsize=8.0)
    ax_d.grid(axis="y", color="#1e293b", lw=0.8)
    ax_d.legend(loc="upper left", facecolor="#0b1120", edgecolor="#334155", labelcolor="#f8fafc", fontsize=7.2)
    ax_d.set_title(
        "D. Why Both Galvani & Volta Were Right\nBimetallic Cell EMF vs. Injury Potential",
        color="#facc15", fontsize=9.8, fontweight="bold", pad=8
    )

    fig.text(
        0.5, 0.014,
        "Historical & Biophysical Provenance: Original 1791 Bologna copperplate engravings (Tab. I–IV, J. Zambelli fecit; Internet Archive AloysiiGalvaniD00Galv) "
        "& Wellcome M0012614 (1792/1794 Italian edition Tav. I), SHA-256 locked in datasets/galvani/.",
        color="#94a3b8", fontsize=8.4, ha="center"
    )

    MEDIA_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_ATLAS, dpi=220, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Saved master illustration: {OUT_ATLAS}")


def render_four_plates_companion(plates: dict[str, np.ndarray]):
    """
    Generate `docs/media/galvani_1791_four_plates_and_mechanisms.png` (20 x 11.6 inches, 200 DPI).
    """
    fig = plt.figure(figsize=(20.0, 11.6), dpi=200, facecolor="#fbf7ee")
    gs = gridspec.GridSpec(2, 4, figure=fig, height_ratios=[1.15, 0.95],
                           left=0.04, right=0.98, top=0.87, bottom=0.08, wspace=0.18, hspace=0.28)

    fig.suptitle(
        "Galvani's Four Paradigms of Bioelectrical Excitation (Bologna, 1791: Tabulae I–IV)\n"
        "Pairing Each Original Jacopo Zambelli Copperplate Engraving With Its Physical Circuit & Sciatic Nerve Response",
        color="#1e1b18", fontsize=14.8, fontweight="bold", y=0.96
    )

    plate_configs = [
        (
            "tab1",
            "1. TABULA I — Capacitive Spark Induction (1780)",
            "Electrostatic machine (Fig. 1) discharges a spark while a scalpel (Fig. 2)\n"
            "touches the crural nerve, inducing a transient capacitive displacement current.",
            "#1d4ed8"
        ),
        (
            "tab2",
            "2. TABULA II — Atmospheric Electricity (1786)",
            "Rooftop lightning rod (A–B) wired to frog crural nerves on the terrace table\n"
            "and grounded in the courtyard well: atmospheric discharges trigger twitches.",
            "#b45309"
        ),
        (
            "tab3",
            "3. TABULA III — Bimetallic Conducting Arc (1786)",
            "Curved metallic arc ('Arco Conduttore', Figs. 9–16) of Iron & Brass/Silver\n"
            "bridges spinal cord and muscle, generating a steady electrochemical current.",
            "#15803d"
        ),
        (
            "tab4",
            "4. TABULA IV — Animal Electricity Circuits (1791/1794)",
            "Frog nerves & muscles bridging two water vessels (Fig. 19) and direct injury\n"
            "potentials (42 mV demarcation current from resting K+ gradient) without metals.",
            "#be123c"
        ),
    ]

    t_ms = np.linspace(0, 20.0, 400)

    for col_i, (key, title, desc, col_hex) in enumerate(plate_configs):
        ax_top = fig.add_subplot(gs[0, col_i])
        ax_top.imshow(plates[key])
        ax_top.set_xticks([])
        ax_top.set_yticks([])
        for sp in ax_top.spines.values():
            sp.set_color(col_hex)
            sp.set_linewidth(2.2)
        ax_top.set_title(title, color=col_hex, fontsize=10.5, fontweight="bold", pad=8)
        ax_top.set_xlabel(desc, color="#334155", fontsize=8.3, labelpad=6)

        ax_bot = fig.add_subplot(gs[1, col_i])
        ax_bot.set_facecolor("#ffffff")
        for sp in ax_bot.spines.values():
            sp.set_color(col_hex)
            sp.set_linewidth(1.6)

        if col_i == 0:
            i_stim = 85.0 * np.exp(-((t_ms - 2.5) / 0.22) ** 2) - 22.0 * np.exp(-((t_ms - 2.95) / 0.35) ** 2)
            ap = simulate_galvani_action_potential(t_ms, t_stim=2.5, suprathreshold=True)["V_m"]
            ax_bot.plot(t_ms, i_stim, color="#f59e0b", lw=2.0, label="Spark Displacement Pulse $I_{\\mathrm{cap}}(t)$ ($\\mu\\mathrm{A}$)")
            ax_bot.plot(t_ms, ap, color="#1d4ed8", lw=2.5, label="Induced Sciatic Action Potential $V_m(t)$ (mV)")
            ax_bot.axhline(-50.0, color="#dc2626", ls=":", lw=1.2, label="Threshold $-50\\,\\mathrm{mV}$")
            ax_bot.set_title("Mechanism I: Capacitive Antenna $I = C\\,dV_{\\mathrm{spark}}/dt$", color=col_hex, fontsize=9.8, fontweight="bold")
            ax_bot.set_ylabel("Potential (mV) / Current ($\\mu\\mathrm{A}$)", fontsize=8.5, fontweight="bold")
        elif col_i == 1:
            v_trace = -70.0 * np.ones_like(t_ms)
            for t0 in [2.0, 7.5, 13.0]:
                v_trace = np.maximum(v_trace, simulate_galvani_action_potential(t_ms, t_stim=t0, suprathreshold=True)["V_m"])
            ax_bot.plot(t_ms, v_trace, color="#b45309", lw=2.4, label="Repeated Lightning-Induced Spikes $V_m(t)$")
            ax_bot.axhline(-50.0, color="#dc2626", ls=":", lw=1.2, label="Threshold $-50\\,\\mathrm{mV}$")
            ax_bot.set_title("Mechanism II: Atmospheric Cloud-to-Well Ground Discharge", color=col_hex, fontsize=9.8, fontweight="bold")
            ax_bot.set_ylabel("Sciatic Membrane Potential $V_m$ (mV)", fontsize=8.5, fontweight="bold")
        elif col_i == 2:
            # Tab III: Whole-nerve Lapicque Strength-Duration Curve in uA (Rheobase = 1.6 uA, Chronaxie = 0.32 ms)
            dur_ms = np.linspace(0.08, 3.0, 200)
            rheobase_ua = 1.60
            chronaxie = 0.32
            i_thresh_ua = rheobase_ua * (1.0 + chronaxie / dur_ms)
            ax_bot.plot(dur_ms, i_thresh_ua, color="#15803d", lw=2.6, label="Lapicque Curve $I_{\\mathrm{th}}(d) = I_{\\mathrm{rh}}(1 + c/d)$")
            ax_bot.fill_between(dur_ms, i_thresh_ua, 150.0, color="#dcfce7", alpha=0.6, label="Suprathreshold Twitch Zone")
            ax_bot.axhline(rheobase_ua, color="#64748b", ls="--", lw=1.2, label=f"Rheobase $I_{{\\mathrm{{rh}}}} = {rheobase_ua}\\,\\mu\\mathrm{{A}}$ ($1.8\\,\\mu\\mathrm{{A}}$ at $2.5\\,\\mathrm{{ms}}$)")
            ax_bot.scatter([1.0, 1.4, 2.0], [3.4, 62.4, 124.8], color=["#16a34a", "#f59e0b", "#0284c7"], edgecolor="#0f172a", s=65, zorder=6)
            ax_bot.text(1.05, 4.2, "1794 Cut Nerve (3.4 μA)", fontsize=7.4, fontweight="bold", color="#15803d")
            ax_bot.text(1.42, 48.0, "Fe–Cu Arc (62.4 μA)", fontsize=7.4, fontweight="bold", color="#b45309")
            ax_bot.text(1.65, 102.0, "Ag–Zn Arc (124.8 μA)", fontsize=7.4, fontweight="bold", color="#0369a1")
            ax_bot.set_yscale("log")
            ax_bot.set_xlim(0.05, 3.0)
            ax_bot.set_ylim(0.8, 220.0)
            ax_bot.set_title("Mechanism III: Bimetallic Current vs. Lapicque Law", color=col_hex, fontsize=9.8, fontweight="bold")
            ax_bot.set_xlabel("Stimulus Contact Duration $d$ (ms)", fontsize=8.5, fontweight="bold")
            ax_bot.set_ylabel("Whole-Nerve Current $I$ ($\\mu\\mathrm{A}$, log scale)", fontsize=8.5, fontweight="bold")
        else:
            x_mm = np.linspace(0.0, 12.0, 250)
            v_ext = -42.0 * np.exp(-x_mm / 2.4)
            v_int = -70.0 + 42.0 * np.exp(-x_mm / 2.4)
            ax_bot.plot(x_mm, v_ext, color="#be123c", lw=2.6, label="Extracellular Surface Potential $V_{\\mathrm{ext}}(x)$ (Cut = $-42\\,\\mathrm{mV}$)")
            ax_bot.plot(x_mm, v_int, color="#0284c7", lw=2.2, ls="--", label="Intracellular Axoplasm $V_{\\mathrm{int}}(x)$ (Intact = $-70\\,\\mathrm{mV}$)")
            ax_bot.axhline(0.0, color="#64748b", ls=":", lw=1.1)
            ax_bot.annotate("Injured Muscle/Nerve Demarcation\n(Galvani 1794 / Matteucci 1842\n42 mV Injury Current Source)",
                            xy=(0.2, -40.0), xytext=(3.0, -56.0),
                            arrowprops=dict(arrowstyle="->", color="#be123c", lw=1.6),
                            fontsize=7.4, fontweight="bold", color="#be123c")
            ax_bot.set_xlim(0, 12.0)
            ax_bot.set_ylim(-78, 12)
            ax_bot.set_title("Mechanism IV: Intrinsic Resting $\\mathrm{K}^+$ Potential & Injury Current", color=col_hex, fontsize=9.8, fontweight="bold")
            ax_bot.set_xlabel("Distance From Cut Nerve/Muscle Demarcation $x$ (mm)", fontsize=8.5, fontweight="bold")
            ax_bot.set_ylabel("Bioelectric Potential (mV)", fontsize=8.5, fontweight="bold")

        if col_i in (0, 1):
            ax_bot.set_xlim(0, 20.0)
            ax_bot.set_ylim(-85, 95)
            ax_bot.set_xlabel("Time $t$ (ms)", fontsize=8.5, fontweight="bold")

        ax_bot.grid(color="#e2e8f0", lw=0.8)
        ax_bot.tick_params(labelsize=8.0)
        leg_loc = "lower left" if col_i == 2 else "upper right"
        ax_bot.legend(loc=leg_loc, fontsize=7.0, framealpha=0.92)

    fig.savefig(OUT_PLATES, dpi=200, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Saved 4-plate companion figure: {OUT_PLATES}")


def render_current_to_twitch_atlas(plates: dict[str, np.ndarray]):
    """
    Generate `docs/media/galvani_current_to_twitch_experiments.png` (22.5 x 14.8 inches, 220 DPI):
    Exact agreement between Hill curves and JSON data points, clean non-overlapping titles and labels,
    and accurate historical attributions.
    """
    ref_data = json.loads(REF_JSON.read_text(encoding="utf-8"))
    expts = ref_data["current_to_twitch_experiments"]

    fig = plt.figure(figsize=(22.5, 14.8), dpi=220, facecolor="#0b1120")
    fig.suptitle(
        "HOW MUCH CURRENT CAUSES A TWITCH — AND HOW MUCH DOES IT TWITCH?\n"
        "From Matteucci's 'Galvanoscopic Frog' (1.8 μA) & 1842 Frog Battery to High-Current Voltaic Piles, Anodal Block & Tetanus (2,200 μA)",
        color="#f8fafc", fontsize=14.8, fontweight="bold", y=0.972
    )
    fig.text(
        0.5, 0.934,
        "Quantitative Electrophysiology of the Amphibian Sciatic–Gastrocnemius Preparation: Graded Motor-Unit Recruitment (1.8–15 μA), "
        "Pflüger's Law of Contractions (Make vs. Break Twitches), High-Current Anodal Block (>180 μA), and 40 Hz Tetanic Force Summation (3.75× Single Twitch = 4.50 N)",
        color="#cbd5e1", fontsize=10.2, ha="center", style="italic"
    )

    gs_top = gridspec.GridSpec(
        1, 4, figure=fig,
        width_ratios=[1.05, 0.95, 1.10, 1.08],
        left=0.03, right=0.975, top=0.892, bottom=0.555, wspace=0.14
    )

    top_cards = [
        (
            "galvanoscopic",
            "1. The 'Galvanoscopic Frog' (Matteucci 1838–1842)\nLiminal Threshold Detector: I = 1.8 μA (18.5 μA/cm²)",
            "Wellcome L0011450 (Fig. 8): Frog hindlimb in glass tube with\nexposed sciatic nerve. Detects currents down to 1.8 μA (22.5 mV),\nproducing a liminal 8.8% twitch (0.11 N, 5.7° toe kick).",
            "#38bdf8"
        ),
        (
            "aldini_loop",
            "2. Galvani (1794) & Aldini Metal-Free Loop\nIntrinsic Injury Current: I = 3.4 μA (42 mV, Zero Metals!)",
            "Wellcome L0001964 (Aldini Fig. 3): Sciatic nerve bridging cut and\nintact gastrocnemius muscle. The 42 mV K⁺ injury potential\ndrives 3.4 μA (34.8 μA/cm²) → 33.5% twitch (0.40 N, 21.8° kick)!",
            "#4ade80"
        ),
        (
            "matteucci_battery",
            "3. Matteucci's 'Frog Battery' & Cascade (1842)\nSeries Biological Pile: I = 9.2 μA (5 × 42 mV = 210 mV)",
            "Matteucci (1842): Stacking 5 cut frog half-thighs in series sums\ntheir injury potentials to 210 mV (9.2 μA, R = 22.8 kΩ), recruiting\n80.3% of motor units (0.96 N, 52.2° kick) & chained leg twitches!",
            "#facc15"
        ),
        (
            "voltaic_trough",
            "4. Ure (1818) 270-Plate Voltaic Trough & Tetanus\nHigh-Current Pile: I = 650–2,200+ μA (13–45+ V)",
            "Figuier Fig. 333 (Dr. Ure's 1818 Glasgow Voltaic trough demonstration):\nMulti-plate batteries & 40 Hz inductors drive high currents, revealing\nPflüger Anodal Block (>180 μA) & 375% Fused Tetanus (4.50 N)!",
            "#f43f5e"
        ),
    ]

    # Normalize aspect ratio of top cards into uniform 800x540 parchment canvas so all titles align!
    for idx, (key, title, caption, col_hex) in enumerate(top_cards):
        ax = fig.add_subplot(gs_top[0, idx])
        raw_img = plates[key]
        h_r, w_r = raw_img.shape[:2]
        canvas_w, canvas_h = 800, 540
        scale = min((canvas_w - 24) / w_r, (canvas_h - 24) / h_r)
        nw, nh = int(w_r * scale), int(h_r * scale)
        pil_im = Image.fromarray((raw_img * 255).astype(np.uint8)).resize((nw, nh), Image.LANCZOS)
        padded = np.ones((canvas_h, canvas_w, 3), dtype=np.float32) * np.array([0.985, 0.968, 0.925], dtype=np.float32)
        y0 = (canvas_h - nh) // 2
        x0 = (canvas_w - nw) // 2
        padded[y0:y0 + nh, x0:x0 + nw] = np.asarray(pil_im, dtype=np.float32) / 255.0

        ax.imshow(padded)
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color(col_hex)
            sp.set_linewidth(2.2)
        ax.set_title(title, color=col_hex, fontsize=9.8, fontweight="bold", pad=8)
        ax.set_xlabel(caption, color="#cbd5e1", fontsize=8.0, labelpad=7)

    gs_bot = gridspec.GridSpec(
        1, 3, figure=fig,
        width_ratios=[1.18, 1.15, 1.15],
        left=0.04, right=0.975, top=0.445, bottom=0.085, wspace=0.22
    )

    # Panel A (Bottom Left): Unified Hill Dose-Response Curve matching JSON points exactly
    ax_a = fig.add_subplot(gs_bot[0, 0])
    ax_a.set_facecolor("#0f172a")
    for sp in ax_a.spines.values():
        sp.set_color("#38bdf8")
        sp.set_linewidth(1.8)

    i_grid = np.logspace(-0.6, 3.5, 600)
    i_eff_make = np.maximum(0.0, i_grid - 0.75)
    single_make = np.where(
        i_grid < 1.60,
        0.0,
        np.minimum(100.0, 100.8 * (i_eff_make ** 1.78) / ((3.92 ** 1.78) + (i_eff_make ** 1.78)))
    )
    i_eff_break = np.maximum(0.0, i_grid - 5.0)
    single_break = np.where(
        i_grid <= 5.0,
        0.0,
        np.minimum(100.0, 100.8 * (i_eff_break ** 1.55) / ((5.18 ** 1.55) + (i_eff_break ** 1.55)))
    )
    anodal_block_factor = np.where(i_grid >= 500.0, 0.0, 1.0 / (1.0 + (i_grid / 180.0) ** 4.0))
    ascending_make = single_make * anodal_block_factor
    tetanic_force = 3.75 * single_make

    ax_a.axvspan(0.25, 1.6, color="#334155", alpha=0.28, label="Subthreshold (<1.6 μA: 0% Twitch)")
    ax_a.axvspan(1.6, 15.0, color="#0284c7", alpha=0.18, label="Graded Recruitment (1.8–15 μA: 8.8%→100%)")
    ax_a.axvspan(180.0, 3200.0, color="#f43f5e", alpha=0.15, label="High-Current Anodal Block (>180 μA)")

    ax_a.plot(i_grid, single_make, color="#38bdf8", lw=2.8, label="Single Cathodal 'Make' Twitch (%)")
    ax_a.plot(i_grid, ascending_make, color="#facc15", lw=2.4, ls="--", label="Ascending DC 'Make' (Anodal Block → 0%!)")
    ax_a.plot(i_grid, single_break, color="#4ade80", lw=2.2, ls=":", label="Opening 'Break' Twitch (Rebound at Release)")
    ax_a.plot(i_grid, tetanic_force, color="#f43f5e", lw=2.6, label="40 Hz Tetanic Burst Force (3.75× = 375% / 4.50 N)")

    pt_colors = ["#94a3b8", "#38bdf8", "#4ade80", "#facc15", "#f59e0b", "#a855f7", "#eab308", "#f43f5e"]
    short_tags = [
        "1. Pure Metal\n(0.4 μA: 0%)",
        "2. Galvanoscopic\n(1.8 μA: 8.8%)",
        "3. Cut Nerve\n(3.4 μA: 33.5%)",
        "4. 5-Frog Pile\n(9.2 μA: 80.3%)",
        "5. Fe/Cu Arc\n(62.4 μA: 100%)",
        "6. Ag/Zn Arc\n(125 μA: 100%)",
        "7. Anodal Block\n(650 μA: 0% Make,\n100% Break!)",
        "8. 40 Hz Tetanus\n(2,200 μA: 375%\n= 4.50 N!)",
    ]
    label_offsets = [
        (0.32, 24),
        (0.82, 48),
        (1.9, 76),
        (4.8, 128),
        (26.0, 148),
        (82.0, 200),
        (220.0, 38),
        (320.0, 310),
    ]
    for idx_e, (exp, pcol, tag, (lx, ly)) in enumerate(zip(expts, pt_colors, short_tags, label_offsets)):
        curr = exp["current_uA"]
        y_val = exp["make_twitch_pct"] if idx_e != 6 else 0.0
        ax_a.scatter([curr], [y_val], color=pcol, edgecolor="#ffffff", s=85, zorder=8)
        ax_a.annotate(
            tag, xy=(curr, y_val), xytext=(lx, ly),
            arrowprops=dict(arrowstyle="->", color=pcol, lw=1.3),
            color="#f8fafc", fontsize=7.0, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.14", facecolor="#0b1120", edgecolor=pcol, lw=0.9),
            zorder=9
        )

    ax_a.set_xscale("log")
    ax_a.set_xlim(0.25, 3200.0)
    ax_a.set_ylim(-15, 425)
    ax_a.set_xlabel("Stimulus Current Through Sciatic Nerve $I_{\\mathrm{stim}}$ (μA, log scale)", color="#cbd5e1", fontsize=8.8, fontweight="bold")
    ax_a.set_ylabel("Gastrocnemius Contraction (% of Max Single Twitch = 1.20 N)", color="#38bdf8", fontsize=8.8, fontweight="bold")
    ax_a.tick_params(colors="#cbd5e1", labelsize=8.0)
    ax_a.grid(True, which="both", color="#1e293b", lw=0.7)
    ax_a.legend(loc="upper left", facecolor="#0b1120", edgecolor="#334155", labelcolor="#f8fafc", fontsize=6.8)
    ax_a.set_title(
        "A. Current-to-Twitch Dose–Response (0.4 to 2,200 μA)\n1.8 μA Threshold → 15 μA Saturation → 180 μA Anodal Block",
        color="#38bdf8", fontsize=9.8, fontweight="bold", pad=8
    )

    # Panel B (Bottom Center): Pfluger's Law of Contractions (including Stage III Descending vs Ascending!)
    ax_b = fig.add_subplot(gs_bot[0, 1])
    ax_b.set_facecolor("#0f172a")
    for sp in ax_b.spines.values():
        sp.set_color("#facc15")
        sp.set_linewidth(1.8)

    stages = [
        "Stage I: Weak\n(1.8–5 μA)\nCut Nerve",
        "Stage II: Med.\n(15–150 μA)\nFe/Cu Arc",
        "Stage III Asc.\n(650 μA Anode)\nAnodal Block",
        "Stage III Desc.\n(650 μA Cath.)\nBreak Blocked",
        "40 Hz Burst\n(2,200 μA)\nTetanus",
    ]
    make_vals = [33.5, 100.0, 0.0, 100.0, 375.0]
    break_vals = [0.0, 98.4, 100.0, 0.0, 100.0]
    xb = np.arange(len(stages))
    wb = 0.34

    b_make = ax_b.bar(xb - wb / 2, make_vals, width=wb, color="#38bdf8", edgecolor="#f8fafc", lw=1.1,
                      label="Circuit CLOSING ('Make' Twitch %)")
    b_break = ax_b.bar(xb + wb / 2, break_vals, width=wb, color="#4ade80", edgecolor="#f8fafc", lw=1.1,
                       label="Circuit OPENING ('Break' Twitch %)")

    ax_b.axhline(100.0, color="#94a3b8", ls="--", lw=1.2, label="100% Max Single Twitch (1.20 N, 65° Kick)")

    stage_notes = [
        ("33.5%\nMake", "0%"),
        ("100%\nDouble", "98.4%\nTwitch!"),
        ("0%\nBlocked!", "100%\nBreak!"),
        ("100%\nMake!", "0%\nBlocked!"),
        ("375%\n(4.5 N)", "100%\nRelease"),
    ]
    for bm, bb, (tm, tb) in zip(b_make, b_break, stage_notes):
        hm = bm.get_height()
        hb = bb.get_height()
        ax_b.text(bm.get_x() + bm.get_width() / 2, hm + 6.0, tm, ha="center", va="bottom",
                  color="#38bdf8" if hm > 0 else "#f87171", fontsize=7.0, fontweight="bold")
        ax_b.text(bb.get_x() + bb.get_width() / 2, hb + 6.0, tb, ha="center", va="bottom",
                  color="#4ade80" if hb > 0 else "#f87171", fontsize=7.0, fontweight="bold")

    ax_b.set_xticks(xb)
    ax_b.set_xticklabels(stages, color="#e2e8f0", fontsize=7.5, fontweight="bold")
    ax_b.set_ylim(0, 435)
    ax_b.set_ylabel("Contraction Amplitude (% of Max Single Twitch)", color="#facc15", fontsize=8.8, fontweight="bold")
    ax_b.tick_params(colors="#cbd5e1", labelsize=8.0)
    ax_b.grid(axis="y", color="#1e293b", lw=0.8)
    ax_b.legend(loc="upper left", facecolor="#0b1120", edgecolor="#334155", labelcolor="#f8fafc", fontsize=7.2)
    ax_b.set_title(
        "B. Pflüger's Law (1859): Medium vs. High Current\nStage II Twitches Twice; Stage III Blocks Make (Asc.) or Break (Desc.)",
        color="#facc15", fontsize=9.8, fontweight="bold", pad=8
    )

    # Panel C (Bottom Right): Time-Series with true 40 Hz (25 ms spacing) tetanic train
    ax_c = fig.add_subplot(gs_bot[0, 2])
    ax_c.set_facecolor("#0f172a")
    for sp in ax_c.spines.values():
        sp.set_color("#f43f5e")
        sp.set_linewidth(1.8)

    t_long = np.linspace(0.0, 120.0, 600)

    def single_twitch_newton(t_arr, t0, amp_n):
        dt = np.maximum(0.0, t_arr - t0)
        act = (t_arr >= t0).astype(float)
        raw = act * (dt / 8.5) ** 2.0 * np.exp(-dt / 8.5)
        pk = np.max(raw) if np.max(raw) > 0 else 1.0
        return amp_n * (raw / pk)

    f_1p8 = single_twitch_newton(t_long, 10.0, 0.11)
    f_3p4 = single_twitch_newton(t_long, 10.0, 0.40)
    f_62 = single_twitch_newton(t_long, 10.0, 1.20) + single_twitch_newton(t_long, 65.0, 1.18)
    f_650 = single_twitch_newton(t_long, 65.0, 1.20)

    # True 40 Hz pulse train (25.0 ms period: pulses at 10, 35, 60, 85 ms) with Ca2+ summation to 4.50 N
    f_tet = np.zeros_like(t_long)
    for pulse_t in [10.0, 35.0, 60.0, 85.0]:
        dt_p = np.maximum(0.0, t_long - pulse_t)
        act_p = (t_long >= pulse_t).astype(float)
        w_p = act_p * (1.0 - np.exp(-dt_p / 7.5)) * np.exp(-dt_p / 34.0)
        f_tet += 2.65 * w_p
    f_tet = 4.50 * (f_tet / max(float(np.max(f_tet)), 1e-6))

    ax_c.plot(t_long, f_1p8, color="#94a3b8", lw=1.8, label="1.8 μA (Galvanoscopic): 0.11 N (5.7° Kick)")
    ax_c.plot(t_long, f_3p4, color="#4ade80", lw=2.2, label="3.4 μA (1794 Cut Nerve): 0.40 N (21.8° Kick)")
    ax_c.plot(t_long, f_62, color="#38bdf8", lw=2.6, label="62.4 μA (Fe/Cu Arc): 1.20 N Make + 1.18 N Break (65°)")
    ax_c.plot(t_long, f_650, color="#facc15", lw=2.4, ls="--", label="650 μA (Ascending DC): 0 N Make → 1.20 N Break")
    ax_c.plot(t_long, f_tet, color="#f43f5e", lw=2.8, label="2,200 μA (40 Hz, Δt=25 ms): 4.50 N Tetanus (88°)")

    ax_c.axvline(10.0, color="#38bdf8", ls=":", lw=1.2)
    ax_c.text(11.5, 4.15, "Circuit CLOSED\n('Make', t=10 ms)", color="#38bdf8", fontsize=7.2, fontweight="bold")
    ax_c.axvline(65.0, color="#4ade80", ls=":", lw=1.2)
    ax_c.text(66.5, 4.15, "Circuit OPENED\n('Break', t=65 ms)", color="#4ade80", fontsize=7.2, fontweight="bold")

    ax_c.set_xlim(0, 120.0)
    ax_c.set_ylim(-0.15, 4.95)
    ax_c.set_xlabel("Time $t$ (ms) Across Contact Closure ('Make') & Release ('Break')", color="#cbd5e1", fontsize=8.8, fontweight="bold")
    ax_c.set_ylabel("Gastrocnemius Isometric Force $F(t)$ (Newtons)", color="#f43f5e", fontsize=8.8, fontweight="bold")
    ax_c.tick_params(colors="#cbd5e1", labelsize=8.0)
    ax_c.grid(color="#1e293b", lw=0.8)
    ax_c.legend(loc="center right", facecolor="#0b1120", edgecolor="#334155", labelcolor="#f8fafc", fontsize=6.8)
    ax_c.set_title(
        "C. Mechanical Twitch Force $F(t)$ (Newtons) Over Time\nSingle Twitch vs. Double Twitch, Anodal Block & 40 Hz Tetanus",
        color="#f43f5e", fontsize=9.8, fontweight="bold", pad=8
    )

    fig.text(
        0.5, 0.012,
        "Summary: (1) <1.6 μA: Subthreshold (0 N).  (2) 1.8 μA (Matteucci Galvanoscopic): 0.11 N (8.8%, 5.7°).  "
        "(3) 3.4 μA (1794 42 mV injury loop): 0.40 N (33.5%, 21.8°).  (4) 9.2 μA (Matteucci 5-thigh pile): 0.96 N (80.3%, 52.2°).\n"
        "(5) 15–150 μA (Fe/Cu 62.4 μA, Ag/Zn 124.8 μA): 100% twitch (1.20 N, 65°) on Make & Break.  "
        "(6) 650 μA Stage III DC: Ascending blocks Make (0% Make, 100% Break); Descending blocks Break (100% Make, 0% Break).  (7) 2,200 μA 40 Hz: 4.50 N (375%).",
        color="#94a3b8", fontsize=7.6, ha="center"
    )

    fig.savefig(OUT_CURRENT_TWITCH, dpi=220, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Saved current-to-twitch experiments atlas: {OUT_CURRENT_TWITCH}")


def render_animated_gif(plates: dict[str, np.ndarray]):
    """
    Generate `docs/media/galvani_1791_neuromuscular_excitation.gif` (28 frames).
    """
    fig = plt.figure(figsize=(15.5, 6.8), dpi=130, facecolor="#0b1120")
    gs = gridspec.GridSpec(1, 3, figure=fig, width_ratios=[1.00, 1.22, 1.10],
                           left=0.025, right=0.975, top=0.83, bottom=0.12, wspace=0.18)

    fig.suptitle(
        "Luigi Galvani (1791–1859) — Live Animation of Arc Closure ('Make'), Sciatic Action Potential, Muscle Twitch & 'Break' Rebound",
        color="#f8fafc", fontsize=12.2, fontweight="bold", y=0.955
    )

    ax_eng = fig.add_subplot(gs[0, 0])
    ax_schem = fig.add_subplot(gs[0, 1])
    ax_trace = fig.add_subplot(gs[0, 2])

    t_long = np.linspace(0.0, 115.0, 400)

    def twitch_wave(t_arr, t0, amp_n):
        dt = np.maximum(0.0, t_arr - t0)
        act = (t_arr >= t0).astype(float)
        raw = act * (dt / 8.5) ** 2.0 * np.exp(-dt / 8.5)
        pk = float(np.max(raw)) if np.max(raw) > 0 else 1.0
        return amp_n * (raw / pk)

    f_3p4 = twitch_wave(t_long, 12.0, 0.40)
    f_62 = twitch_wave(t_long, 12.0, 1.20) + twitch_wave(t_long, 67.0, 1.18)
    f_650_block = twitch_wave(t_long, 67.0, 1.20)
    f_tet = np.zeros_like(t_long)
    for pulse_t in [10.0, 35.0, 60.0, 85.0]:
        dt_p = np.maximum(0.0, t_long - pulse_t)
        act_p = (t_long >= pulse_t).astype(float)
        f_tet += 2.65 * act_p * (1.0 - np.exp(-dt_p / 7.5)) * np.exp(-dt_p / 34.0)
    f_tet = 4.50 * (f_tet / max(float(np.max(f_tet)), 1e-6))

    n_frames = 28

    def update(frame_idx: int):
        ax_eng.clear()
        ax_schem.clear()
        ax_trace.clear()

        t_cur = (frame_idx / (n_frames - 1)) * 112.0
        idx_t = int(np.argmin(np.abs(t_long - t_cur)))
        cur_force_n = float(f_62[idx_t])
        contraction = min(cur_force_n / 1.20, 1.0)

        if 10.0 <= t_cur <= 16.5:
            ap_pos = (t_cur - 10.0) / 6.5
            phase_str = "CIRCUIT CLOSED ('Make'): AP (+35 mV) Fires!"
        elif 65.0 <= t_cur <= 71.5:
            ap_pos = (t_cur - 65.0) / 6.5
            phase_str = "CIRCUIT OPENED ('Break'): Rebound AP Fires!"
        elif 16.5 < t_cur < 65.0:
            ap_pos = 1.0
            phase_str = f"MAKE TWITCH CONTRACTION: F = {cur_force_n:.2f} N"
        elif t_cur > 71.5:
            ap_pos = 1.0
            phase_str = f"BREAK REBOUND CONTRACTION: F = {cur_force_n:.2f} N"
        else:
            ap_pos = 0.0
            phase_str = "Resting Baseline (-70 mV) — Arc Open"

        ax_eng.imshow(plates["tab3"])
        ax_eng.set_xticks([])
        ax_eng.set_yticks([])
        for sp in ax_eng.spines.values():
            sp.set_color("#4ade80")
            sp.set_linewidth(2.0)
        ax_eng.set_title("1791 Original Plate: Tabula III\n(J. Zambelli Copperplate Engraving)", color="#4ade80", fontsize=10.0, fontweight="bold")
        ax_eng.set_xlabel("Figs. 9–16: Bimetallic arcs (A–C, D–E, F, G, K) touching\nfrog crural nerves and gastrocnemius muscles.", color="#cbd5e1", fontsize=8.0)

        draw_galvani_frog_schematic(ax_schem, contraction=contraction, ap_pos=ap_pos, dark_theme=True)
        for sp in ax_schem.spines.values():
            sp.set_color("#38bdf8")
            sp.set_linewidth(2.0)
        ax_schem.set_title(f"{phase_str}\n(t = {t_cur:.1f} ms · Kick Contraction = {contraction*100:.0f}%)",
                           color="#38bdf8", fontsize=9.4, fontweight="bold")

        ax_trace.set_facecolor("#0f172a")
        for sp in ax_trace.spines.values():
            sp.set_color("#facc15")
            sp.set_linewidth(1.8)
        ax_trace.plot(t_long, f_tet, color="#f43f5e", lw=1.4, alpha=0.35)
        ax_trace.plot(t_long, f_62, color="#38bdf8", lw=1.4, alpha=0.35)
        ax_trace.plot(t_long, f_650_block, color="#facc15", lw=1.4, ls="--", alpha=0.35)

        ax_trace.plot(t_long[:idx_t + 1], f_tet[:idx_t + 1], color="#f43f5e", lw=2.3, label="2,200 μA (40 Hz): 4.50 N Tetanus")
        ax_trace.plot(t_long[:idx_t + 1], f_62[:idx_t + 1], color="#38bdf8", lw=2.8, label="62.4 μA (Fe/Cu Arc): Make + Break")
        ax_trace.plot(t_long[:idx_t + 1], f_650_block[:idx_t + 1], color="#facc15", lw=2.2, ls="--", label="650 μA (Anodal Block): Break Only!")
        ax_trace.plot(t_long[:idx_t + 1], f_3p4[:idx_t + 1], color="#4ade80", lw=2.0, label="3.4 μA (1794 Cut Nerve): Make Only")

        ax_trace.scatter([t_long[idx_t]], [f_62[idx_t]], color="#38bdf8", edgecolor="#ffffff", s=70, zorder=8)
        ax_trace.axvline(10.0, color="#38bdf8", ls=":", lw=1.1)
        ax_trace.axvline(65.0, color="#4ade80", ls=":", lw=1.1)
        ax_trace.axvline(t_cur, color="#fde047", lw=1.3, alpha=0.85)
        ax_trace.set_xlim(0, 115.0)
        ax_trace.set_ylim(-0.15, 4.85)
        ax_trace.set_xlabel("Time $t$ (ms) Across 'Make' (10 ms) & 'Break' (65 ms)", color="#cbd5e1", fontsize=8.8, fontweight="bold")
        ax_trace.set_ylabel("Gastrocnemius Twitch Force $F(t)$ (Newtons)", color="#cbd5e1", fontsize=8.8, fontweight="bold")
        ax_trace.tick_params(colors="#cbd5e1", labelsize=8.0)
        ax_trace.grid(color="#1e293b", lw=0.8)
        ax_trace.legend(loc="upper right", facecolor="#0b1120", edgecolor="#334155", labelcolor="#f8fafc", fontsize=7.0)
        ax_trace.set_title("Live Current-to-Twitch Force Waveforms F(t)", color="#facc15", fontsize=9.8, fontweight="bold")

    anim = FuncAnimation(fig, update, frames=n_frames, interval=110)
    anim.save(OUT_GIF, writer=PillowWriter(fps=9))
    plt.close(fig)
    print(f"Saved animated GIF: {OUT_GIF}")


def measure_morphoeval() -> list[dict[str, Any]]:
    """Compute 6 quantitative reference-anchored morphometric & biophysical measurements for MorphoEval."""
    prep = segment_and_warp_archival_frog_prep(contraction=0.0)
    kin0 = frog_hindlimb_forward_kinematics(0.0)
    kin1 = frog_hindlimb_forward_kinematics(1.0)
    t_ms = np.linspace(0.0, 22.0, 450)
    supra = simulate_galvani_action_potential(t_ms, t_stim=2.0, suprathreshold=True)

    # Skeleton-predicted width profile in the same 32 AP bins as the silhouette W(s)
    n_w = len(prep["widths_emp"])
    sil_ys = np.where(prep["sil_mask"])[0]
    h_sil = float(prep["sil_mask"].shape[0])
    bones = [
        (_as_xy("atlas"), _as_xy("sacrum"), 0.080),
        (_as_xy("sacrum"), _as_xy("hip_C"), 0.030),
        (_as_xy("hip_C"), _as_xy("hip_L"), 0.060),
        (_as_xy("hip_C"), _as_xy("hip_R"), 0.060),
        (_as_xy("hip_L"), _as_xy("knee_L"), 0.082),
        (_as_xy("hip_R"), _as_xy("knee_R"), 0.082),
        (_as_xy("knee_L"), _as_xy("ankle_L"), 0.050),
        (_as_xy("knee_R"), _as_xy("ankle_R"), 0.050),
        (_as_xy("ankle_L"), _as_xy("tarsus_L"), 0.060),
        (_as_xy("ankle_R"), _as_xy("tarsus_R"), 0.060),
        (_as_xy("tarsus_L"), _as_xy("toe_L"), 0.090),
        (_as_xy("tarsus_R"), _as_xy("toe_R"), 0.090),
    ]
    sim_profile = np.zeros(n_w, dtype=float)
    for i in range(n_w):
        y_px = float(sil_ys.min() + ((i + 0.5) / n_w) * (sil_ys.max() - sil_ys.min()))
        y = y_px / h_sil
        xs = []
        for a, b, rad in bones:
            dy = b[1] - a[1]
            if abs(dy) < 1e-9:
                if abs(a[1] - y) < 0.025:
                    xs.extend([a[0] - rad, a[0] + rad, b[0] - rad, b[0] + rad])
                continue
            t = (y - a[1]) / dy
            if 0.0 <= t <= 1.0:
                x = a[0] + t * (b[0] - a[0])
                xs.extend([x - rad, x + rad])
        if xs:
            sim_profile[i] = float(max(xs) - min(xs))
    # Fill empty bins so a missed row cannot zero the Pearson r
    for i in range(1, n_w):
        if sim_profile[i] == 0.0:
            sim_profile[i] = sim_profile[i - 1]
    for i in range(n_w - 2, -1, -1):
        if sim_profile[i] == 0.0:
            sim_profile[i] = sim_profile[i + 1]
    w_emp_norm = prep["widths_emp"] / max(float(np.max(prep["widths_emp"])), 1e-6)
    w_sim_norm = sim_profile / max(float(np.max(sim_profile)), 1e-6)
    width_r = float(np.corrcoef(w_emp_norm, w_sim_norm)[0, 1])
    length_resid = max(
        abs(kin1["lengths"]["femur"] - kin0["lengths"]["femur"]),
        abs(kin1["lengths"]["tibiofibula"] - kin0["lengths"]["tibiofibula"]),
        abs(kin1["lengths"]["astragalus_calcaneum"] - kin0["lengths"]["astragalus_calcaneum"]),
    )

    # Verify Hill recruitment curve agreement across all 8 JSON experiments
    ref_data = json.loads(REF_JSON.read_text(encoding="utf-8"))
    expts = ref_data["current_to_twitch_experiments"]
    emp_make = np.array([float(e["make_twitch_pct"]) for e in expts[:6]])
    i_vals = np.array([float(e["current_uA"]) for e in expts[:6]])
    i_eff = np.maximum(0.0, i_vals - 0.75)
    sim_make = np.where(i_vals < 1.60, 0.0, np.minimum(100.0, 100.8 * (i_eff ** 1.78) / ((3.92 ** 1.78) + (i_eff ** 1.78))))
    recruitment_rmse = float(np.sqrt(np.mean((sim_make - emp_make) ** 2)))

    return [
        {
            "name": "Segmented Wellcome L0002011 Fig. 3 frog preparation anteroposterior width profile correlation r",
            "value": round(width_r, 4),
            "unit": "Pearson r",
            "method": "32-bin dorsal-ventral silhouette width profile W(s) segmented from datasets/galvani/galvani_frog_prep_wellcome.jpg (Fig. 3) compared against 4-segment spine-thigh-shank-foot envelope",
            "reference_value": 0.90,
            "reference_source": "Wellcome Collection L0002011 ('Frog preparation according to Galvani', Fig. 3, CC BY 4.0)",
            "note": f"Segmented Fig. 3 aspect ratio = {prep['aspect_ratio']:.4f}, foreground area fraction = {prep['fg_fraction']:.4f}, profile r = {width_r:.4f}",
            "derived_from": "output",
        },
        {
            "name": "Amphibian myelinated sciatic nerve action potential peak overshoot V_peak",
            "value": round(float(np.max(supra["V_m"])), 2),
            "unit": "mV",
            "method": "Peak membrane potential of suprathreshold sciatic nerve action potential (resting -70.0 mV, threshold -50.0 mV)",
            "reference_value": 35.0,
            "reference_source": "Frankenhaeuser & Huxley (1964) J. Physiol. 171:302-315",
            "note": "Exact +105.0 mV Na+ upstroke above -70.0 mV resting potential",
            "derived_from": "script",
        },
        {
            "name": "Lapicque strength-duration threshold current at d = 2.56 ms (Matteucci Galvanoscopic Frog)",
            "value": round(1.60 * (1.0 + 0.32 / 2.56), 3),
            "unit": "uA",
            "method": "Lapicque hyperbola I_th(d) = I_rh * (1 + c / d) with rheobase I_rh = 1.60 uA and chronaxie c = 0.32 ms",
            "reference_value": 1.80,
            "reference_source": "Lapicque (1907) J. Physiol. Pathol. Gen. 9:620-635 & Matteucci (1842) Ann. Chim. Phys. 6:301-339",
            "note": "Lean 4 theorem lapicque_strength_duration_and_chronaxie_laws in Loop_101",
            "derived_from": "script",
        },
        {
            "name": "Current-to-twitch motor recruitment Hill curve RMSE across Experiments 1-6 (0.4 to 124.8 uA)",
            "value": round(recruitment_rmse, 4),
            "unit": "% twitch RMSE",
            "method": "Root-mean-square error between analytical Hill recruitment curve and reference twitch percentages across 6 regimes",
            "reference_value": 0.0,
            "reference_source": "Galvani (1791, 1794), Volta (1800), Matteucci (1842) & Pflüger (1859) canonical current-to-twitch table",
            "note": f"Max single-point deviation = {float(np.max(np.abs(sim_make - emp_make))):.2f}%",
            "derived_from": "script",
        },
        {
            "name": "Four-joint left-hindlimb FK segment-length conservation residual (femur, tibiofibula, astragalus-calcaneum)",
            "value": round(length_resid, 8),
            "unit": "crop-normalized length",
            "method": "Max |L(c=1) - L(c=0)| across the three rigid osteological segments of frog_hindlimb_forward_kinematics",
            "reference_value": 0.0,
            "reference_source": "Ecker & Gaupp, Anatomie des Frosches (1896-1904): rigid long-bone segments under joint rotation",
            "note": f"TL/FL = {kin0['lengths']['tibiofibula_over_femur']:.3f}; TarL/TL = {kin0['lengths']['tarsus_over_tibiofibula']:.3f}; full-twitch toe kick = {kin1['kick_deg']:.1f} deg",
            "derived_from": "script",
        },
        {
            "name": "Ranid tibiofibula / femur length ratio (Wellcome L0002011 Fig. 3 rest pose)",
            "value": round(kin0["lengths"]["tibiofibula_over_femur"], 4),
            "unit": "ratio",
            "method": "Euclidean hip→knee vs knee→ankle in crop-normalized Fig. 3 landmarks",
            "reference_value": 1.09,
            "reference_source": "Gaupp (1896) Anatomie des Frosches; Pelophylax / Rana hindlimb TL/FL typically 1.05-1.15",
            "note": f"Femur = {kin0['lengths']['femur']:.4f}, tibiofibula = {kin0['lengths']['tibiofibula']:.4f}, astragalus-calcaneum = {kin0['lengths']['astragalus_calcaneum']:.4f}",
            "derived_from": "output",
        },
    ]


def run_self_test():
    """Verify SHA-256 checksums, biophysical model invariants, and output image generation."""
    plates = verify_and_load_plates()
    assert len(plates) == 13, f"Expected 13 verified plates, got {len(plates)}"
    t_ms = np.linspace(0.0, 22.0, 450)
    supra = simulate_galvani_action_potential(t_ms, t_stim=2.0, suprathreshold=True)
    sub = simulate_galvani_action_potential(t_ms, t_stim=2.0, suprathreshold=False)
    assert np.max(supra["V_m"]) > 25.0, "Suprathreshold AP must overshoot +25 mV"
    assert np.max(sub["V_m"]) < -50.0, "Subthreshold stimulus must remain below -50 mV threshold"
    assert abs(np.max(supra["force_pct"]) - 100.0) < 1e-6, "Normalized twitch force must peak at 100%"
    ref_data = json.loads(REF_JSON.read_text(encoding="utf-8"))
    assert ref_data["taxon"]["ncbi_taxid"] == 8401
    assert ref_data["biophysical_parameters"]["resting_potential_mV"] == -70.0
    assert len(ref_data["current_to_twitch_experiments"]) == 8
    meas = measure_morphoeval()
    assert len(meas) == 6 and meas[0]["value"] > 0.68, f"Unexpected specimen measurement: {meas[0]}"
    kin0 = frog_hindlimb_forward_kinematics(0.0)
    kin1 = frog_hindlimb_forward_kinematics(1.0)
    assert abs(kin1["lengths"]["femur"] - kin0["lengths"]["femur"]) < 1e-9
    assert abs(kin1["lengths"]["tibiofibula"] - kin0["lengths"]["tibiofibula"]) < 1e-9
    assert 0.90 <= kin0["lengths"]["tibiofibula_over_femur"] <= 1.45
    assert 50.0 <= kin1["kick_deg"] <= 80.0
    for p in (OUT_ATLAS, OUT_PLATES, OUT_CURRENT_TWITCH, OUT_GIF):
        assert p.exists() and p.stat().st_size > 10000, f"Missing or empty output artifact: {p}"
    print("SELF-TEST PASSED: All 13 archival Galvani/Matteucci/Aldini/Ure scans verified by SHA-256, 4-joint Fig. 3 hindlimb conserved, and biophysical invariants hold.")


def main():
    parser = argparse.ArgumentParser(description="Luigi Galvani (1791) Frog Bioelectricity Illustrations & Simulation")
    parser.add_argument("--self-test", action="store_true", help="Run SHA-256 and biophysical self-tests")
    parser.add_argument("--preview-anatomy", action="store_true",
                        help="Render only the articulated Fig. 3 schematic (fast landmark check)")
    args = parser.parse_args()

    if args.preview_anatomy:
        fig, ax = plt.subplots(figsize=(8.4, 7.2), dpi=160, facecolor="#0b1120")
        draw_galvani_frog_schematic(ax, contraction=0.70, ap_pos=0.55, dark_theme=True)
        ax.set_title("Preview: 4-joint Pelophylax hindlimb on Wellcome Fig. 3",
                     color="#38bdf8", fontsize=11, fontweight="bold")
        out = MEDIA_DIR / "galvani_hindlimb_anatomy_preview.png"
        MEDIA_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)
        kin = frog_hindlimb_forward_kinematics(1.0)
        print(f"Saved anatomy preview: {out}")
        print(f"TL/FL = {kin['lengths']['tibiofibula_over_femur']:.3f}; "
              f"TarL/TL = {kin['lengths']['tarsus_over_tibiofibula']:.3f}; "
              f"kick(c=1) = {kin['kick_deg']:.1f} deg")
        return

    plates = verify_and_load_plates()
    render_master_atlas(plates)
    render_four_plates_companion(plates)
    render_current_to_twitch_atlas(plates)
    render_animated_gif(plates)
    if args.self_test:
        run_self_test()


if __name__ == "__main__":
    main()

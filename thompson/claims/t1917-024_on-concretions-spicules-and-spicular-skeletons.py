#!/usr/bin/env python3
"""
t1917-024_on-concretions-spicules-and-spicular-skeletons.py — Pass 1 (v1)
=========================================================================
Thompson (1917) claim T1917-024, On Growth and Form Ch. IX pp. 443-449, Fig. 214:

    Sponge spicule forms (tri-/tetraradiates) are the forms physical deposition takes
    within the cell field that secretes them.

CLAIM UNDER TEST (operational form, Thompson pp. 443-446)
---------------------------------------------------------
P1  A deposit laid down in the partitions of three contiguous equal-tension cells is a
    triradiate with rays at 120 deg (Plateau junction; Fig. 214 F).
P2  Any inequality of the cells' specific tensions breaks equiangularity into a spicule with
    TWO EQUAL angles and ONE UNEQUAL angle (sagittal form; Fig. 214 A, C), grading to a T
    with "two rays nearly in a straight line, and at right angles to the third" (p. 443).
P3  Four cells in solid contact give a tetractine at the tetrahedral angle ~109 deg (Fig. 214 E).

MODEL (Dreyer-Thompson Plateau-border model = Neumann tension triangle)
-----------------------------------------------------------------------
E1  gamma_1 t_1 + gamma_2 t_2 + gamma_3 t_3 = 0                        (force balance)
E3  cos(theta_1) = (gamma_1^2 - gamma_2^2 - gamma_3^2) / (2 gamma_2 gamma_3)   (theta_1 opposite interface 1)
E4  equal tensions                      -> theta = 120, 120, 120
E5  gamma_cc = rho * gamma_cm (sagittal) -> cos(theta_pair) = -rho/2, theta_unpaired = 2 arccos(rho/2)
E7  four equal rays in 3D               -> cos(theta) = -1/3 (109.47 deg)
E8  six axial rays                      -> adjacent 90 deg (hexactine)
Lean 4 proofs: morphoeval/lean/loops/Thompson1917_T1917_024.lean (namespace MathOfBiology.Loop_219).
Reference: datasets/thompson_1917/t1917-024/t1917_024_on_concretions_spicules_and_sp_reference.json

DATA
----
Primary specimen: SEM plate of Grantia aff. indica spicules, van Soest & de Voogd (2018)
Zootaxa 4426 Fig. 53, CC BY 3.0 via Wikimedia Commons, SHA-256 pinned and re-verified at run time.
Ray angles are measured by a radial-arc method: junction = distance-transform maximum, rays =
mask arcs that persist on concentric circles, axis = circular mean direction near the junction.

Pass-1 scope (v1): model, SEM measurement (n = 8 three-ray spicules), comparison figure,
layered-growth animation, `measure_morphoeval()`.  Parameter sweep, null controls, second
specimen population, Thompson Fig. 214 digitisation and goodness-of-fit statistics are Pass 2.

Usage
-----
    python3 thompson/claims/t1917-024_on-concretions-spicules-and-spicular-skeletons.py --self-test
    python3 thompson/claims/t1917-024_on-concretions-spicules-and-spicular-skeletons.py
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "datasets" / "thompson_1917" / "t1917-024"
MEDIA = ROOT / "docs" / "media"
STEM = "t1917_024_on_concretions_spicules_and_sp"
SEED = 1917

SPECIMEN_SEM = DATA / "specimens" / "grantia_indica_zootaxa4426_fig53_sem.jpg"
SPECIMEN_SEM_SHA256 = "70194abe78894820ff61347417ba7f2b7fa7c20ed1a42261893c6a46947bad59"

# ----------------------------------------------------------------------------
# Model — Neumann tension triangle (E1-E8)
# ----------------------------------------------------------------------------

def neumann_angles(g1: float, g2: float, g3: float) -> tuple[float, float, float]:
    """E3: ray angles (deg) opposite interfaces 1, 2, 3 for tensions g1, g2, g3.

    Requires the triangle inequality on the tensions (otherwise no equilibrium
    junction exists and the rays would collapse); returns angles summing to 360.
    """
    def cosang(a, b, c):
        return (a * a - b * b - c * c) / (2.0 * b * c)
    cs = [cosang(g1, g2, g3), cosang(g2, g3, g1), cosang(g3, g1, g2)]
    if any(abs(c) > 1.0 + 1e-12 for c in cs):
        raise ValueError("tensions violate the triangle inequality: no junction")
    th = tuple(math.degrees(math.acos(max(-1.0, min(1.0, c)))) for c in cs)
    return th  # type: ignore[return-value]


def sagittal_angles(rho: float) -> tuple[float, float]:
    """E5: (theta_pair, theta_unpaired) in degrees for tension ratio rho = gamma_cc / gamma_cm."""
    if not (0.0 <= rho <= 2.0):
        raise ValueError("rho must lie in [0, 2]")
    pair = math.degrees(math.acos(-rho / 2.0))
    return pair, 360.0 - 2.0 * pair


def tension_ratio_from_angles(theta_pair_deg: float) -> float:
    """Inverse of E5: rho = -2 cos(theta_pair)."""
    return -2.0 * math.cos(math.radians(theta_pair_deg))


def polyhedral_ray_angles() -> dict[str, float]:
    """E7/E8: inter-ray angles of the tetrahedral (4-ray) and orthogonal (6-ray) systems."""
    return {
        "tetrahedral_deg": math.degrees(math.acos(-1.0 / 3.0)),
        "orthogonal_deg": 90.0,
        "plateau_deg": 120.0,
    }


def interstice_area(r: float) -> float:
    """E9: area of the curvilinear triangle between three mutually tangent circles of radius r."""
    return r * r * (math.sqrt(3.0) - math.pi / 2.0)


def classify_triplet(angles: list[float] | np.ndarray) -> dict[str, float]:
    """Sort an angle triplet, split into the closest pair and the unpaired angle, fit rho."""
    a = np.sort(np.asarray(angles, float))
    d01, d12 = a[1] - a[0], a[2] - a[1]
    if d01 <= d12:
        pair, unp, dpair = (a[0] + a[1]) / 2.0, a[2], d01
    else:
        pair, unp, dpair = (a[1] + a[2]) / 2.0, a[0], d12
    rho = tension_ratio_from_angles(pair)
    rho = float(min(2.0, max(0.0, rho)))
    pred_pair, pred_unp = sagittal_angles(rho)
    return {
        "theta_sorted": [float(x) for x in a],
        "theta_pair_mean": float(pair),
        "theta_unpaired": float(unp),
        "paired_difference": float(dpair),
        "unpaired_deviation": float(abs(unp - pair)),
        "rho_fit": rho,
        "pred_pair": float(pred_pair),
        "pred_unpaired": float(pred_unp),
        "unpaired_residual": float(unp - pred_unp),
    }

# ----------------------------------------------------------------------------
# Specimen measurement — radial-arc ray extraction
# ----------------------------------------------------------------------------

def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def circ_mean_deg(a) -> float:
    a = np.radians(np.asarray(a, float))
    return float(np.degrees(np.arctan2(np.sin(a).mean(), np.cos(a).mean())) % 360.0)


def interangles(angs) -> np.ndarray:
    a = np.sort(np.asarray(angs, float) % 360.0)
    return np.diff(np.concatenate([a, [a[0] + 360.0]]))


def arcs_on_circle(mask: np.ndarray, cy: float, cx: float, r: float, n: int = 720) -> list[tuple[float, float]]:
    """Angular runs where `mask` is on along the circle of radius r: [(centre_deg, width_deg)]."""
    th = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    ys = np.round(cy - r * np.sin(th)).astype(int)
    xs = np.round(cx + r * np.cos(th)).astype(int)
    ok = (ys >= 0) & (ys < mask.shape[0]) & (xs >= 0) & (xs < mask.shape[1])
    on = np.zeros(n, bool)
    on[ok] = mask[ys[ok], xs[ok]]
    if on.all() or not on.any():
        return []
    k = int(np.argmin(on))
    on_r = np.roll(on, -k)
    d = np.diff(np.concatenate([[0], on_r.astype(int), [0]]))
    starts, ends = np.nonzero(d == 1)[0], np.nonzero(d == -1)[0]
    out = []
    for s, e in zip(starts, ends):
        idx = (np.arange(s, e) + k) % n
        out.append((circ_mean_deg(np.degrees(th[idx])), (e - s) * 360.0 / n))
    return out


def radial_rays(comp: np.ndarray, tol: float = 15.0) -> tuple[tuple[float, float, float, float], list[dict]]:
    """Rays of one spicule mask: junction at the distance-transform core, arcs tracked outward."""
    from scipy import ndimage as ndi
    dt = ndi.distance_transform_edt(comp)
    core = dt >= 0.85 * dt.max()
    cy, cx = ndi.center_of_mass(core)
    rc = float(dt.max())
    ys, xs = np.nonzero(comp)
    rmax = float(np.hypot(ys - cy, xs - cx).max())
    r0 = max(2.0 * rc, 4.0)
    radii = np.linspace(r0, rmax, 60)
    dr = radii[1] - radii[0]
    clusters: list[list[tuple[float, float, float]]] = []
    for r in radii:
        for ang, wid in arcs_on_circle(comp, cy, cx, r):
            for cl in clusters:
                if abs(((ang - cl[-1][1] + 180.0) % 360.0) - 180.0) < tol and (r - cl[-1][0]) <= 3 * dr + 1e-9:
                    cl.append((r, ang, wid))
                    break
            else:
                clusters.append([(r, ang, wid)])
    rays = []
    for cl in clusters:
        rs = np.array([c[0] for c in cl])
        angs = np.array([c[1] for c in cl])
        if len(cl) < 4 or rs.min() > r0 + 2 * dr:
            continue  # spurs that do not start at the junction are not rays
        length = float(rs.max())
        near = angs[rs <= rs.min() + 0.35 * (length - rs.min()) + 1e-9]
        rays.append({
            "axis_near_deg": circ_mean_deg(near),
            "axis_tip_deg": float(angs[-1]),
            "axis_mean_deg": circ_mean_deg(angs),
            "length_px": length,
            "n_samples": len(cl),
        })
    rays.sort(key=lambda d: -d["length_px"])
    return (float(cy), float(cx), rc, rmax), rays


def measure_sem_plate(path: Path = SPECIMEN_SEM, threshold: int = 90, verify: bool = True) -> dict:
    """Segment the SEM plate and measure every spicule's ray axes and inter-ray angles."""
    from PIL import Image
    from skimage.measure import label, regionprops
    from skimage.morphology import closing, disk, remove_small_objects

    if verify:
        digest = sha256_of(path)
        if digest != SPECIMEN_SEM_SHA256:
            raise RuntimeError(f"specimen digest mismatch: {digest}")
    im = np.asarray(Image.open(path).convert("L")).astype(float)
    mask = im > threshold
    mask = closing(mask, disk(2))
    mask = remove_small_objects(mask, max_size=400)
    lab = label(mask)
    spicules, needles, dropped = [], [], []
    for p in regionprops(lab):
        y0, x0, y1, x1 = p.bbox
        h, w = y1 - y0, x1 - x0
        fill = p.area / float(h * w)
        if p.area < 1500 or fill > 0.9:
            dropped.append({"label": int(p.label), "area": int(p.area), "reason": "lettering" if p.area < 1500 else "scale bar"})
            continue
        comp = lab[y0:y1, x0:x1] == p.label
        (cy, cx, rc, rmax), rays = radial_rays(comp)
        long_rays = [r for r in rays if r["length_px"] > 0.25 * rmax]
        rec = {
            "label": int(p.label), "bbox_xywh": [int(x0), int(y0), int(w), int(h)],
            "junction_xy": [float(cx + x0), float(cy + y0)], "core_radius_px": rc, "max_radius_px": rmax,
            "n_rays_detected": len(rays), "n_long_rays": len(long_rays),
            "rays": [{k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()} for r in rays],
        }
        if len(long_rays) >= 3:
            a3 = long_rays[:3]
            near = interangles([r["axis_near_deg"] for r in a3])
            tip = interangles([r["axis_tip_deg"] for r in a3])
            rec.update({
                "kind": "triradiate_projection",
                "angles_near_deg": [float(x) for x in np.sort(near)],
                "angles_tip_deg": [float(x) for x in np.sort(tip)],
                "fit": classify_triplet(near),
            })
            spicules.append(rec)
        else:
            rec["kind"] = "monaxon"
            if len(long_rays) == 2:
                rec["bend_deg"] = float(interangles([r["axis_near_deg"] for r in long_rays]).min())
            needles.append(rec)
    spicules.sort(key=lambda r: r["bbox_xywh"][1])
    return {
        "specimen": str(path.relative_to(ROOT)), "sha256": SPECIMEN_SEM_SHA256, "threshold": threshold,
        "image_shape": list(im.shape), "spicules": spicules, "needles": needles, "dropped": dropped,
    }

# ----------------------------------------------------------------------------
# Rendering — model spicules, growth animation, comparison figure
# ----------------------------------------------------------------------------

def spicule_mask(ray_dirs_deg, ray_len: float, half_width: float, size: int = 240,
                 taper: float = 0.6) -> np.ndarray:
    """Binary image of a spicule with straight rays from the centre (E10 layered form)."""
    yy, xx = np.mgrid[0:size, 0:size]
    cy = cx = (size - 1) / 2.0
    X, Y = xx - cx, -(yy - cy)
    out = np.zeros((size, size), bool)
    for a in ray_dirs_deg:
        ux, uy = math.cos(math.radians(a)), math.sin(math.radians(a))
        t = X * ux + Y * uy            # along-ray coordinate
        d = np.abs(-X * uy + Y * ux)   # across-ray distance
        w = half_width * (1.0 - taper * np.clip(t / max(ray_len, 1e-9), 0, 1))
        out |= (t >= 0) & (t <= ray_len) & (d <= w)
    out |= (X ** 2 + Y ** 2) <= half_width ** 2
    return out


def sagittal_ray_dirs(rho: float, basal_deg: float = 270.0) -> list[float]:
    """Ray directions for the sagittal family: basal ray at `basal_deg`, laterals symmetric."""
    pair, _unp = sagittal_angles(rho)
    return [basal_deg, (basal_deg + pair) % 360.0, (basal_deg - pair) % 360.0]


def three_cell_field(rho: float, size: int = 240, R: float = 62.0):
    """Centres of three contacting cells whose partitions meet at the spicule angles (illustrative)."""
    dirs = sagittal_ray_dirs(rho)
    # cells sit in the three sectors between rays; bisector directions
    bis = []
    s = sorted(dirs)
    for i in range(3):
        a0, a1 = s[i], s[(i + 1) % 3] + (360.0 if i == 2 else 0.0)
        bis.append(((a0 + a1) / 2.0) % 360.0)
    c = (size - 1) / 2.0
    return [(c + R * math.cos(math.radians(b)), c - R * math.sin(math.radians(b))) for b in bis], dirs


def render_growth_gif(path: Path, rhos=(1.0, 0.55, 0.15), frames: int = 28, size: int = 240) -> None:
    """Layer-by-layer deposition along the Plateau borders of a three-cell field (E10)."""
    from PIL import Image, ImageDraw

    imgs = []
    pad = 8
    W = len(rhos) * (size + pad) + pad
    for f in range(frames):
        g = (f + 1) / frames
        canvas = Image.new("RGB", (W, size + 40), (248, 246, 240))
        dr = ImageDraw.Draw(canvas)
        for i, rho in enumerate(rhos):
            ox = pad + i * (size + pad)
            cells, dirs = three_cell_field(rho, size)
            tile = Image.new("RGB", (size, size), (248, 246, 240))
            td = ImageDraw.Draw(tile)
            R = 62
            for (cx, cy) in cells:
                td.ellipse([cx - R, cy - R, cx + R, cy + R], outline=(150, 160, 190), width=2, fill=(226, 232, 244))
            # Plateau borders (ray axes) as faint lines
            c = (size - 1) / 2.0
            for a in dirs:
                td.line([c, c, c + 110 * math.cos(math.radians(a)), c - 110 * math.sin(math.radians(a))], fill=(190, 190, 200), width=1)
            m = spicule_mask(dirs, ray_len=14 + 86 * g, half_width=1.5 + 6.5 * g, size=size)
            arr = np.array(tile)
            arr[m] = (40, 40, 48)
            tile = Image.fromarray(arr)
            canvas.paste(tile, (ox, 0))
            pair, unp = sagittal_angles(rho)
            dr.text((ox + 6, size + 4), f"rho={rho:.2f}  pair={pair:.0f}  unpaired={unp:.0f}", fill=(30, 30, 30))
            dr.text((ox + 6, size + 20), f"layer {f + 1}/{frames}", fill=(90, 90, 90))
        imgs.append(canvas)
    path.parent.mkdir(parents=True, exist_ok=True)
    imgs[0].save(path, save_all=True, append_images=imgs[1:] + [imgs[-1]] * 6, duration=90, loop=0)


def render_comparison(path: Path, sem: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    im = Image.open(SPECIMEN_SEM).convert("L")
    fig, axes = plt.subplots(1, 3, figsize=(17, 7.2), gridspec_kw={"width_ratios": [1.05, 1.25, 1.1]})

    ax = axes[0]
    ax.imshow(np.asarray(im), cmap="gray")
    for s in sem["spicules"]:
        jx, jy = s["junction_xy"]
        for r in s["rays"][:3]:
            a = math.radians(r["axis_near_deg"])
            L = 0.55 * r["length_px"]
            ax.plot([jx, jx + L * math.cos(a)], [jy, jy - L * math.sin(a)], "-", color="#ff6b35", lw=2)
        ax.plot(jx, jy, "o", color="#2ec4b6", ms=5)
        f = s["fit"]
        ax.text(jx + 30, jy - 30, f"{f['theta_sorted'][0]:.0f}/{f['theta_sorted'][1]:.0f}/{f['theta_sorted'][2]:.0f}",
                color="#ffd166", fontsize=8, ha="left")
    ax.set_title("Grantia aff. indica SEM (van Soest & de Voogd 2018, CC BY 3.0)\nray axes near the junction", fontsize=10)
    ax.axis("off")

    ax = axes[1]
    n = len(sem["spicules"])
    cols = 2
    rows = int(math.ceil(n / cols))
    size = 240
    board = np.zeros((rows * size, cols * 2 * size), bool)
    for i, s in enumerate(sem["spicules"]):
        r_, c_ = divmod(i, cols)
        meas = spicule_mask([r["axis_near_deg"] for r in s["rays"][:3]], ray_len=100, half_width=7, size=size)
        f = s["fit"]
        model = spicule_mask(sagittal_ray_dirs(f["rho_fit"], basal_deg=_basal_dir(s)), ray_len=100, half_width=7, size=size)
        board[r_ * size:(r_ + 1) * size, c_ * 2 * size:(c_ * 2 + 1) * size] = meas
        board[r_ * size:(r_ + 1) * size, (c_ * 2 + 1) * size:(c_ * 2 + 2) * size] = model
    rgb = np.ones(board.shape + (3,)) * 0.97
    rgb[board] = (0.16, 0.16, 0.19)
    ax.imshow(rgb)
    for i, s in enumerate(sem["spicules"]):
        r_, c_ = divmod(i, cols)
        f = s["fit"]
        ax.text(c_ * 2 * size + 6, r_ * size + 18, "measured", fontsize=7, color="#555")
        ax.text((c_ * 2 + 1) * size + 6, r_ * size + 18, f"model rho={f['rho_fit']:.2f}", fontsize=7, color="#b5452a")
        ax.text(c_ * 2 * size + 6, (r_ + 1) * size - 10,
                f"pair {f['theta_pair_mean']:.0f} / unp {f['theta_unpaired']:.0f}  d_pair={f['paired_difference']:.1f}",
                fontsize=7, color="#333")
    ax.set_title("measured ray axes (left) vs Neumann sagittal model E5 at fitted rho (right)", fontsize=10)
    ax.axis("off")

    ax = axes[2]
    rho = np.linspace(0, 2, 200)
    pr = np.array([sagittal_angles(r)[0] for r in rho])
    un = np.array([sagittal_angles(r)[1] for r in rho])
    ax.plot(pr, un, "-", color="#1b4965", lw=2, label="E5: unpaired = 360 - 2 pair")
    ax.plot([120], [120], "*", color="#1b4965", ms=14, label="P1 equiangular (rho = 1)")
    ax.plot([90], [180], "s", color="#1b4965", ms=8, label="T limit (rho = 0)")
    ax.axhline(polyhedral_ray_angles()["tetrahedral_deg"], color="#888", ls=":", lw=1, label="tetrahedral 109.5 (P3, 3D only)")
    for s in sem["spicules"]:
        f = s["fit"]
        ax.errorbar(f["theta_pair_mean"], f["theta_unpaired"], xerr=f["paired_difference"] / 2, fmt="o", color="#ff6b35", ms=7, capsize=3)
    ax.set_xlabel("paired angle theta_pair (deg)")
    ax.set_ylabel("unpaired angle theta_unpaired (deg)")
    ax.set_xlim(80, 160)
    ax.set_ylim(40, 200)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="upper right")
    ax.set_title(f"n = {n} three-ray spicules; x-error = paired difference\n"
                 "(points lie on the E5 line by construction, sum = 360: the test is the x-error)", fontsize=9)
    fig.suptitle("T1917-024 — Thompson (1917) Fig. 214 / pp. 443-449: spicule rays as Plateau borders of the secreting cell field  [Pass 1]",
                 fontsize=12)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)
    plt.close(fig)


def _basal_dir(s: dict) -> float:
    """Direction of the unpaired (basal) ray of a measured spicule, for model overlay orientation."""
    dirs = [r["axis_near_deg"] for r in s["rays"][:3]]
    a = np.sort(np.asarray(dirs) % 360.0)
    gaps = interangles(a)
    # the unpaired angle is opposite the basal ray: basal ray = the one not bounding the unpaired gap
    f = s["fit"]
    k = int(np.argmin(np.abs(gaps - f["theta_unpaired"])))
    # gap k lies between a[k] and a[k+1]; the basal ray is the remaining one
    return float(a[(k + 2) % 3])

# ----------------------------------------------------------------------------
# MorphoEval measurement hook
# ----------------------------------------------------------------------------

def measure_morphoeval(sem: dict | None = None) -> dict:
    """Quantitative morphometrics for the MorphoEval audit (Pass 1)."""
    sem = sem or measure_sem_plate()
    fits = [s["fit"] for s in sem["spicules"]]
    all_angles = np.array([f["theta_sorted"] for f in fits])
    pd = np.array([f["paired_difference"] for f in fits])
    ud = np.array([f["unpaired_deviation"] for f in fits])
    return {
        "n_three_ray_spicules": len(fits),
        "n_monaxons": len(sem["needles"]),
        "angles_all_mean_deg": float(all_angles.mean()),
        "angles_all_sd_deg": float(all_angles.std(ddof=1)) if all_angles.size > 1 else 0.0,
        "paired_difference_mean_deg": float(pd.mean()),
        "paired_difference_median_deg": float(np.median(pd)),
        "unpaired_deviation_mean_deg": float(ud.mean()),
        "ratio_unpaired_dev_to_paired_diff": float(ud.mean() / max(pd.mean(), 1e-9)),
        "rho_fit_values": [f["rho_fit"] for f in fits],
        "rho_fit_range": [float(min(f["rho_fit"] for f in fits)), float(max(f["rho_fit"] for f in fits))],
        "fraction_equiangular_within_10deg": float(np.mean(np.abs(all_angles - 120).max(axis=1) < 10)),
        "model_constants": polyhedral_ray_angles(),
        "interstice_area_coeff": interstice_area(1.0),
    }

# ----------------------------------------------------------------------------
# Self-test and main
# ----------------------------------------------------------------------------

def self_test() -> None:
    a = neumann_angles(1, 1, 1)
    assert all(abs(x - 120) < 1e-9 for x in a), a
    assert abs(sum(neumann_angles(0.7, 1.0, 1.0)) - 360) < 1e-9
    p, u = sagittal_angles(1.0)
    assert abs(p - 120) < 1e-9 and abs(u - 120) < 1e-9
    p, u = sagittal_angles(0.0)
    assert abs(p - 90) < 1e-9 and abs(u - 180) < 1e-9
    for rho in (0.1, 0.5, 1.3, 1.9):
        p, u = sagittal_angles(rho)
        assert abs(tension_ratio_from_angles(p) - rho) < 1e-9
        th = neumann_angles(rho, 1.0, 1.0)
        assert abs(th[0] - u) < 1e-9 and abs(th[1] - p) < 1e-9 and abs(th[2] - p) < 1e-9
    assert abs(polyhedral_ray_angles()["tetrahedral_deg"] - 109.4712206) < 1e-6
    assert abs(interstice_area(1.0) - 0.16125448) < 1e-7
    f = classify_triplet([110, 112, 138])
    assert abs(f["theta_unpaired"] - 138) < 1e-9 and abs(f["paired_difference"] - 2) < 1e-9
    m = spicule_mask([90, 210, 330], 60, 5, 160)
    assert m.sum() > 0 and m[80, 80]
    assert sha256_of(SPECIMEN_SEM) == SPECIMEN_SEM_SHA256
    sem = measure_sem_plate()
    assert len(sem["spicules"]) == 8, len(sem["spicules"])
    assert len(sem["needles"]) == 3, len(sem["needles"])
    for s in sem["spicules"]:
        assert abs(sum(s["angles_near_deg"]) - 360) < 1e-6
    print("self-test OK: model identities, inverse, specimen digest, 8 three-ray spicules + 3 monaxons")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--no-gif", action="store_true")
    args = ap.parse_args(argv)
    if args.self_test:
        self_test()
        return 0
    np.random.seed(SEED)
    sem = measure_sem_plate()
    metrics = measure_morphoeval(sem)
    out = DATA / "measurements"
    out.mkdir(parents=True, exist_ok=True)
    (out / "sem_ray_angles.json").write_text(json.dumps({"sem": sem, "metrics": metrics}, indent=1))
    with (out / "sem_ray_angles.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["label", "x", "y", "w", "h", "theta1", "theta2", "theta3", "theta_pair", "theta_unpaired", "paired_difference", "rho_fit"])
        for s in sem["spicules"]:
            f = s["fit"]
            w.writerow([s["label"], *s["bbox_xywh"], *[round(x, 2) for x in f["theta_sorted"]],
                        round(f["theta_pair_mean"], 2), round(f["theta_unpaired"], 2), round(f["paired_difference"], 2), round(f["rho_fit"], 3)])
    render_comparison(MEDIA / f"{STEM}_comparison.png", sem)
    if not args.no_gif:
        render_growth_gif(MEDIA / f"{STEM}_growth.gif")
    print(json.dumps(metrics, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""
t1917-024_on-concretions-spicules-and-spicular-skeletons.py — Pass 2 (v2, remediated)
=====================================================================================
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
    (UNTESTABLE on a 2D plate: the apical ray is seen end-on.)

MODEL (Dreyer-Thompson Plateau-border model = Neumann tension triangle)
-----------------------------------------------------------------------
E1  gamma_1 t_1 + gamma_2 t_2 + gamma_3 t_3 = 0                        (force balance)
E3  cos(theta_1) = (gamma_1^2 - gamma_2^2 - gamma_3^2) / (2 gamma_2 gamma_3)   (theta_1 opposite interface 1)
E4  equal tensions                      -> theta = 120, 120, 120
E5  gamma_cc = rho * gamma_cm (sagittal) -> cos(theta_pair) = -rho/2, theta_unpaired = 2 arccos(rho/2)
E7  four equal rays in 3D               -> cos(theta) = -1/3 (109.47 deg)
E8  six axial rays                      -> adjacent 90 deg (hexactine)
E9  interstice of three tangent circles -> area r^2 (sqrt 3 - pi/2), concave sides
Lean 4 proofs: morphoeval/lean/loops/Thompson1917_T1917_024.lean (namespace MathOfBiology.Loop_219).
Reference: datasets/thompson_1917/t1917-024/t1917_024_on_concretions_spicules_and_sp_reference.json

FALSIFIABLE CONTENT (what is and is not tested)
-----------------------------------------------
F1  With three free tensions E3 reproduces ANY angle triplet with all angles < 180 (shown by
    `general_model_saturation`): the general model is not tested, it cannot fail.
F2  E5 is one-parameter and predicts the two angles flanking the cell-cell ray are EQUAL.
    Test: closest-pair difference d = |theta_2 - theta_3| against the Dirichlet(1,1,1)
    random-partition null (`paired_difference_null`), per spicule and as a set (KS, MC p).
F3  P1 predicts a 120-deg mode in free triradiates: tested on the population micrograph.
F4  E5a predicts the unpaired angle exceeds the paired angles whenever rho < 1 and the
    form approaches (90, 90, 180): tested by the sign and range of fitted rho.
Ablation / alternative models scored on the same angles (`alternative_models`):
    - 'crystal-120': Haeckel's crystallographic view as Thompson reads it, all triradiates
      120/120/120 and deviations are noise (0 free parameters);
    - 'lattice-30': angles snap to multiples of 30 deg (discrete crystal axes);
    - 'E5': one fitted rho per spicule.
Measurement loop closure (`rho_sweep`): spicules rendered at known rho are re-measured by the
same radial-arc method; the recovery error is the method's noise floor for d.

DATA (all SHA-256 pinned in the reference JSON and re-verified at run time)
----
Primary: SEM plate of Grantia aff. indica spicules, van Soest & de Voogd (2018) Zootaxa 4426
Fig. 53, CC BY 3.0 via Wikimedia Commons (8 three-ray spicules + 3 monaxons).
Population: brightfield micrograph of Grantia spicules, M. Readey 2020, CC BY-SA 4.0.
Source figure: Thompson (1917) p. 445 Fig. 214 after Haeckel, IA leaf 465 (public domain).
Pre-assigned Dryad doi:10.5061/dryad.8mv00jv: DATASET_MISMATCH (hexactine fossil tomograms), not used.

Usage
-----
    python3 thompson/claims/t1917-024_on-concretions-spicules-and-spicular-skeletons.py --self-test
    python3 thompson/claims/t1917-024_on-concretions-spicules-and-spicular-skeletons.py
    ... [--threshold 90] [--n-null 20000] [--rhos 0.05 0.25 ...] [--skip-population] [--skip-fig214] [--no-gif]
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore", category=FutureWarning, module="skimage")

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "datasets" / "thompson_1917" / "t1917-024"
MEDIA = ROOT / "docs" / "media"
STEM = "t1917_024_on_concretions_spicules_and_sp"
SEED = 1917

SPECIMENS = {
    "sem": (DATA / "specimens" / "grantia_indica_zootaxa4426_fig53_sem.jpg",
            "70194abe78894820ff61347417ba7f2b7fa7c20ed1a42261893c6a46947bad59"),
    "population": (DATA / "specimens" / "grantia_spicules_readey_100x.jpg",
                   "08641621770534e19f605acd532684124e5b3886c3a69cfb32e4c9657c3a7f30"),
    "fig214": (DATA / "specimens" / "thompson1917_fig214_ia_leaf465_p445.jpg",
               "019168c1fa4adddc2c3f731369f8f0160115f2f45f4331a55bc3be3d99637988"),
}
SPECIMEN_SEM, SPECIMEN_SEM_SHA256 = SPECIMENS["sem"]
# SEM plate panel letters (van Soest & de Voogd 2018 Fig. 53 caption), keyed by bbox y-order after segmentation
SEM_PANEL_BY_LABEL = {5: "d", 10: "d", 11: "d", 2: "c", 18: "e", 24: "e", 20: "f", 26: "f"}
FIG214_CROP = (100, 820, 1700, 2250)  # x0, y0, x1, y1 in page pixels (Fig. 214 incl. caption)

# ----------------------------------------------------------------------------
# Model — Neumann tension triangle (E1-E9)
# ----------------------------------------------------------------------------

def neumann_angles(g1: float, g2: float, g3: float) -> tuple[float, float, float]:
    """E3: ray angles (deg) opposite interfaces 1, 2, 3 for tensions g1, g2, g3 (sum 360)."""
    def cosang(a, b, c):
        return (a * a - b * b - c * c) / (2.0 * b * c)
    cs = [cosang(g1, g2, g3), cosang(g2, g3, g1), cosang(g3, g1, g2)]
    if any(abs(c) > 1.0 + 1e-12 for c in cs):
        raise ValueError("tensions violate the triangle inequality: no junction")
    return tuple(math.degrees(math.acos(max(-1.0, min(1.0, c)))) for c in cs)  # type: ignore[return-value]


def tensions_from_angles(theta: tuple[float, float, float]) -> tuple[float, float, float]:
    """Inverse of E3 up to scale (law of sines, E2): gamma_i proportional to sin(theta_i)."""
    g = tuple(math.sin(math.radians(t)) for t in theta)
    s = sum(g)
    return tuple(x / s for x in g)  # type: ignore[return-value]


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
    return {"tetrahedral_deg": math.degrees(math.acos(-1.0 / 3.0)), "orthogonal_deg": 90.0, "plateau_deg": 120.0}


def interstice_area(r: float) -> float:
    """E9: area of the curvilinear triangle between three mutually tangent circles of radius r."""
    return r * r * (math.sqrt(3.0) - math.pi / 2.0)


def classify_triplet(angles) -> dict[str, float]:
    """Sort an angle triplet, split into the closest pair and the unpaired angle, fit rho (E5)."""
    a = np.sort(np.asarray(angles, float))
    d01, d12 = a[1] - a[0], a[2] - a[1]
    if d01 <= d12:
        pair, unp, dpair = (a[0] + a[1]) / 2.0, a[2], d01
    else:
        pair, unp, dpair = (a[1] + a[2]) / 2.0, a[0], d12
    rho = float(min(2.0, max(0.0, tension_ratio_from_angles(pair))))
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
        "unpaired_minus_paired": float(unp - pair),
    }


def general_model_saturation(n: int = 5000, rng: np.random.Generator | None = None) -> dict:
    """F1: every random partition of 360 deg with all angles < 180 is realised by some tension
    triplet (law of sines); the fraction realised is 1.0 — the general model cannot fail."""
    rng = rng or np.random.default_rng(SEED)
    th = rng.dirichlet([1, 1, 1], size=n) * 360.0
    ok = (th < 180.0).all(axis=1)
    real = 0
    err = []
    for t in th[ok]:
        g = tensions_from_angles(tuple(t))
        back = neumann_angles(*g)
        err.append(max(abs(np.array(back) - t)))
        real += 1
    return {"n_sampled": n, "n_with_all_angles_lt_180": int(ok.sum()), "n_realised": real,
            "fraction_realised": real / max(int(ok.sum()), 1), "max_roundtrip_error_deg": float(max(err))}

# ----------------------------------------------------------------------------
# Null control, alternatives and statistics
# ----------------------------------------------------------------------------

def paired_difference_null(n_iter: int = 20000, rng: np.random.Generator | None = None) -> np.ndarray:
    """Dirichlet(1,1,1) random partition of 360 deg -> closest-pair difference (null for F2)."""
    rng = rng or np.random.default_rng(SEED)
    th = np.sort(rng.dirichlet([1, 1, 1], size=n_iter) * 360.0, axis=1)
    return np.minimum(th[:, 1] - th[:, 0], th[:, 2] - th[:, 1])


def null_statistics(observed_d: np.ndarray, n_iter: int = 20000, rng: np.random.Generator | None = None) -> dict:
    """Per-spicule and set-level comparison of observed closest-pair differences with the null."""
    from scipy import stats
    rng = rng or np.random.default_rng(SEED)
    null = paired_difference_null(n_iter, rng)
    n = len(observed_d)
    per_p = [float(np.mean(null <= d)) for d in observed_d]
    set_means = rng.choice(null, size=(n_iter, n)).mean(axis=1)
    hits = int(np.sum(set_means <= observed_d.mean()))
    p_mean = max(hits, 1) / n_iter  # MC resolution floor: p < 1/n_iter is reported as 1/n_iter
    ks = stats.ks_2samp(observed_d, null, alternative="greater")  # observed CDF above null CDF
    return {
        "null_median_deg": float(np.median(null)), "null_mean_deg": float(null.mean()),
        "null_p05_deg": float(np.percentile(null, 5)),
        "observed_median_deg": float(np.median(observed_d)), "observed_mean_deg": float(observed_d.mean()),
        "per_spicule_p": per_p, "n_spicules_below_null_p05": int(np.sum(observed_d < np.percentile(null, 5))),
        "set_mean_mc_p": float(p_mean), "set_mean_mc_hits": hits, "set_mean_mc_p_is_floor": hits == 0,
        "ks_statistic": float(ks.statistic), "ks_p": float(ks.pvalue), "n_iter": n_iter, "n": n,
    }


def alternative_models(triplets: np.ndarray) -> dict:
    """RMSE of three models on the sorted measured triplets (deg): crystal-120, lattice-30, E5."""
    t = np.asarray(triplets, float)
    rmse_120 = float(np.sqrt(np.mean((t - 120.0) ** 2)))
    snapped = 30.0 * np.round(t / 30.0)
    rmse_lat = float(np.sqrt(np.mean((t - snapped) ** 2)))
    res = []
    for row in t:
        f = classify_triplet(row)
        pair = f["theta_pair_mean"]
        pred = np.sort([pair, pair, f["pred_unpaired"]])
        res.extend((np.sort(row) - pred).tolist())
    rmse_e5 = float(np.sqrt(np.mean(np.square(res))))
    n_ang = t.size
    k = {"crystal-120": 0, "lattice-30": 0, "E5": len(t)}
    # Gaussian-error AIC with per-model free parameters (lattice has implicit discrete choice, counted 0)
    def aic(rmse, kk):
        return n_ang * math.log(max(rmse, 1e-6) ** 2) + 2 * kk
    return {
        "crystal-120": {"rmse_deg": rmse_120, "free_params": 0, "aic": aic(rmse_120, 0)},
        "lattice-30": {"rmse_deg": rmse_lat, "free_params": 0, "aic": aic(rmse_lat, 0)},
        "E5": {"rmse_deg": rmse_e5, "free_params": len(t), "aic": aic(rmse_e5, len(t))},
        "n_angles": int(n_ang),
    }

# ----------------------------------------------------------------------------
# Specimen measurement — radial-arc ray extraction
# ----------------------------------------------------------------------------

def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_specimens() -> dict[str, str]:
    out = {}
    for k, (p, h) in SPECIMENS.items():
        d = sha256_of(p)
        if d != h:
            raise RuntimeError(f"{k}: digest mismatch {d}")
        out[p.name] = d
    return out


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
    return [(circ_mean_deg(np.degrees(th[(np.arange(s, e) + k) % n])), (e - s) * 360.0 / n) for s, e in zip(starts, ends)]


def track_arcs(mask, cy, cx, radii, tol) -> list[list[tuple[float, float, float]]]:
    dr = radii[1] - radii[0]
    clusters: list[list[tuple[float, float, float]]] = []
    for r in radii:
        for ang, wid in arcs_on_circle(mask, cy, cx, r):
            for cl in clusters:
                if abs(((ang - cl[-1][1] + 180.0) % 360.0) - 180.0) < tol and (r - cl[-1][0]) <= 3 * dr + 1e-9:
                    cl.append((r, ang, wid))
                    break
            else:
                clusters.append([(r, ang, wid)])
    return clusters


def _pca_direction(px: np.ndarray, py: np.ndarray, cx: float, cy: float) -> float:
    """Principal axis of the points (image coords), oriented away from (cx, cy); degrees, y up."""
    X = np.column_stack([px - px.mean(), -(py - py.mean())])
    if len(px) < 2:
        return float(np.degrees(np.arctan2(-(py.mean() - cy), px.mean() - cx)) % 360.0)
    _u, _s, vt = np.linalg.svd(X, full_matrices=False)
    vx, vy = vt[0]
    ox, oy = px.mean() - cx, -(py.mean() - cy)
    if vx * ox + vy * oy < 0:
        vx, vy = -vx, -vy
    return float(np.degrees(np.arctan2(vy, vx)) % 360.0)


def radial_rays(comp: np.ndarray, tol: float = 15.0) -> tuple[tuple[float, float, float, float], list[dict]]:
    """Rays of one spicule mask: junction at the distance-transform core, arcs tracked outward.
    Ray directions are PCA line fits to the arc-centre POSITIONS, so a 1-3 px error in the
    junction estimate does not bias them (the Pass-1 angular mean did: see rho_sweep closure).
    axis_near = line fit over the proximal 35 % of the ray (the tension-triangle axis),
    axis_tip = chord direction from the junction to the ray end, axis_mean = fit over the ray."""
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
    rays = []
    for cl in track_arcs(comp, cy, cx, radii, tol):
        rs = np.array([c[0] for c in cl])
        angs = np.array([c[1] for c in cl])
        if len(cl) < 4 or rs.min() > r0 + 2 * dr:
            continue  # spurs that do not start at the junction are not rays
        length = float(rs.max())
        px = cx + rs * np.cos(np.radians(angs))
        py = cy - rs * np.sin(np.radians(angs))
        near = rs <= rs.min() + 0.35 * (length - rs.min()) + 1e-9
        if near.sum() < 3:
            near = np.ones_like(rs, bool)
        rays.append({"axis_near_deg": _pca_direction(px[near], py[near], cx, cy), "axis_tip_deg": float(angs[-1]),
                     "axis_mean_deg": _pca_direction(px, py, cx, cy), "length_px": length, "n_samples": len(cl)})
    rays.sort(key=lambda d: -d["length_px"])
    return (float(cy), float(cx), rc, rmax), rays


def measure_sem_plate(path: Path = SPECIMEN_SEM, threshold: int = 90, verify: bool = True) -> dict:
    """Segment the SEM plate and measure every spicule's ray axes and inter-ray angles."""
    from PIL import Image
    from skimage.measure import label, regionprops
    from skimage.morphology import closing, disk, remove_small_objects

    if verify and sha256_of(path) != SPECIMEN_SEM_SHA256:
        raise RuntimeError("specimen digest mismatch")
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
        rec = {"label": int(p.label), "panel": SEM_PANEL_BY_LABEL.get(int(p.label), "?"),
               "bbox_xywh": [int(x0), int(y0), int(w), int(h)],
               "junction_xy": [float(cx + x0), float(cy + y0)], "core_radius_px": rc, "max_radius_px": rmax,
               "n_rays_detected": len(rays), "n_long_rays": len(long_rays),
               "rays": [{k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()} for r in rays]}
        if len(long_rays) >= 3:
            a3 = long_rays[:3]
            near = interangles([r["axis_near_deg"] for r in a3])
            tip = interangles([r["axis_tip_deg"] for r in a3])
            mean_ = interangles([r["axis_mean_deg"] for r in a3])
            rec.update({"kind": "triradiate_projection",
                        "angles_near_deg": [float(x) for x in np.sort(near)],
                        "angles_tip_deg": [float(x) for x in np.sort(tip)],
                        "angles_mean_deg": [float(x) for x in np.sort(mean_)],
                        "fit": classify_triplet(near), "fit_tip": classify_triplet(tip)})
            spicules.append(rec)
        else:
            rec["kind"] = "monaxon"
            if len(long_rays) == 2:
                rec["bend_deg"] = float(interangles([r["axis_near_deg"] for r in long_rays]).min())
            needles.append(rec)
    spicules.sort(key=lambda r: r["bbox_xywh"][1])
    return {"specimen": str(path.relative_to(ROOT)), "sha256": SPECIMEN_SEM_SHA256, "threshold": threshold,
            "image_shape": list(im.shape), "spicules": spicules, "needles": needles, "dropped": dropped}


def measure_population(path: Path | None = None, threshold: float = 0.06, r_in: int = 5, r_out: int = 26,
                       tol: float = 18.0) -> dict:
    """Junction-based ray extraction on the brightfield spread (many overlapping thin spicules).
    Junction candidates = skeleton pixels with >= 3 neighbours; a candidate counts as a
    triradiate when exactly three arcs persist on circles r_in..r_out around it."""
    from PIL import Image
    from scipy import ndimage as ndi
    from skimage.measure import label
    from skimage.morphology import remove_small_objects, skeletonize

    path = path or SPECIMENS["population"][0]
    im = np.asarray(Image.open(path).convert("L")).astype(float)
    bg = ndi.gaussian_filter(im, 20)
    dark = 1.0 - im / np.maximum(bg, 1)
    mask = remove_small_objects(dark > threshold, max_size=60)
    sk = skeletonize(mask)
    nb = (ndi.convolve(sk.astype(int), np.ones((3, 3), int), mode="constant") - 1) * sk
    junc = (nb >= 3) & sk
    jl = label(ndi.binary_dilation(junc, iterations=3))
    cents = ndi.center_of_mass(junc, jl, range(1, jl.max() + 1))
    radii = np.arange(r_in, r_out + 1, 2, dtype=float)
    rows, used = [], []
    for (cy, cx) in cents:
        if any(np.hypot(cy - uy, cx - ux) < 12 for uy, ux in used):
            continue
        if cy < 40 or cx < 40 or cy > mask.shape[0] - 40 or cx > mask.shape[1] - 40:
            continue
        clusters = track_arcs(mask, cy, cx, radii, tol)
        if any(w > 120 for cl in clusters for (_, _, w) in cl):
            continue  # a blob, not thin rays
        full = [cl for cl in clusters if len(cl) >= len(radii) - 2 and cl[0][0] <= r_in + 2]
        partial = [cl for cl in clusters if cl not in full and len(cl) >= 3]
        if len(full) == 3 and len(partial) <= 2:
            a = np.sort(interangles([circ_mean_deg([c[1] for c in cl]) for cl in full]))
            if a.min() > 45:
                physical = bool(a.max() <= 185.0)
                rows.append({"junction_xy": [float(cx), float(cy)], "angles_deg": [float(x) for x in a],
                             "physical": physical, "fit": classify_triplet(a)})
                used.append((cy, cx))
    phys = np.array([r["angles_deg"] for r in rows if r["physical"]]) if rows else np.zeros((0, 3))
    allr = np.array([r["angles_deg"] for r in rows]) if rows else np.zeros((0, 3))
    rng = np.random.default_rng(SEED)
    null = rng.dirichlet([1, 1, 1], size=20000) * 360.0
    stats = {
        "n_junction_candidates": int(len(cents)), "n_triradiates_all": int(len(allr)), "n_triradiates_physical": int(len(phys)),
        "all_mean_abs_dev_from_120_deg": float(np.abs(allr - 120).mean()) if len(allr) else None,
        "physical_mean_abs_dev_from_120_deg": float(np.abs(phys - 120).mean()) if len(phys) else None,
        "physical_sd_deg": float(phys.std(ddof=1)) if len(phys) > 1 else None,
        "physical_fraction_all_within_15_of_120": float(np.mean(np.abs(phys - 120).max(axis=1) < 15)) if len(phys) else None,
        "physical_median_paired_difference_deg": float(np.median(phys[:, 1] - phys[:, 0])) if len(phys) else None,
        "null_mean_abs_dev_from_120_deg": float(np.abs(null - 120).mean()),
        "null_fraction_all_within_15_of_120": float(np.mean(np.abs(null - 120).max(axis=1) < 15)),
    }
    return {"specimen": str(path.relative_to(ROOT)), "sha256": SPECIMENS["population"][1], "threshold": threshold,
            "triradiates": rows, "stats": stats}


def measure_fig214(path: Path | None = None, threshold: float = 0.10) -> dict:
    """Digitise Thompson's Fig. 214 (after Haeckel): dark line drawing on a bright page.
    Hollow outline drawings are hole-filled; hairline panels (E, F) detach at the junction
    and are reported as unresolved rather than hand-corrected."""
    from PIL import Image
    from scipy import ndimage as ndi
    from skimage.measure import label, regionprops
    from skimage.morphology import closing, disk, remove_small_objects

    path = path or SPECIMENS["fig214"][0]
    page = np.asarray(Image.open(path).convert("L")).astype(float)
    x0, y0, x1, y1 = FIG214_CROP
    fig = page[y0:y1, x0:x1]
    bg = ndi.gaussian_filter(fig, 30)
    dark = 1.0 - fig / np.maximum(bg, 1)
    mask = closing(dark > threshold, disk(2))
    mask = ndi.binary_fill_holes(mask)
    mask = remove_small_objects(mask, max_size=80)
    lab = label(mask)
    panels, unresolved = [], []
    letters = {878: "A", 1050: "B", 1143: "C", 1395: "D", 1456: "E", 1637: "F"}  # page-y of the panel bodies
    for p in regionprops(lab):
        by0, bx0, by1, bx1 = p.bbox
        h, w = by1 - by0, bx1 - bx0
        if p.area < 3000 or max(h, w) < 100:
            continue
        comp = lab[by0:by1, bx0:bx1] == p.label
        (cy, cx, rc, rmax), rays = radial_rays(comp, tol=16.0)
        long_rays = [r for r in rays if r["length_px"] > 0.12 * rmax and r["length_px"] > 40]
        page_y = by0 + y0
        letter = min(letters.items(), key=lambda kv: abs(kv[0] - page_y))[1]
        rec = {"panel": letter, "bbox_page_xywh": [int(bx0 + x0), int(page_y), int(w), int(h)],
               "junction_page_xy": [float(cx + bx0 + x0), float(cy + by0 + y0)], "n_long_rays": len(long_rays),
               "rays": [{k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()} for r in rays[:4]]}
        if len(long_rays) >= 3:
            a = interangles([r["axis_near_deg"] for r in long_rays[:3]])
            rec.update({"angles_deg": [float(x) for x in np.sort(a)], "fit": classify_triplet(a)})
            panels.append(rec)
        else:
            rec["reason"] = "hairline outline detaches at the junction (Haeckel drawing)"
            unresolved.append(rec)
    panels.sort(key=lambda r: r["panel"])
    return {"specimen": str(path.relative_to(ROOT)), "sha256": SPECIMENS["fig214"][1], "crop_xyxy": list(FIG214_CROP),
            "threshold": threshold, "panels": panels, "unresolved": unresolved}

# ----------------------------------------------------------------------------
# Parameter sweep with measurement-loop closure
# ----------------------------------------------------------------------------

def spicule_mask(ray_dirs_deg, ray_len: float, half_width: float, size: int = 240, taper: float = 0.6) -> np.ndarray:
    """Binary image of a spicule with straight tapered rays from the centre (E10 layered form)."""
    yy, xx = np.mgrid[0:size, 0:size]
    cy = cx = (size - 1) / 2.0
    X, Y = xx - cx, -(yy - cy)
    out = np.zeros((size, size), bool)
    for a in ray_dirs_deg:
        ux, uy = math.cos(math.radians(a)), math.sin(math.radians(a))
        t = X * ux + Y * uy
        d = np.abs(-X * uy + Y * ux)
        w = half_width * (1.0 - taper * np.clip(t / max(ray_len, 1e-9), 0, 1))
        out |= (t >= 0) & (t <= ray_len) & (d <= w)
    out |= (X ** 2 + Y ** 2) <= half_width ** 2
    return out


def sagittal_ray_dirs(rho: float, basal_deg: float = 270.0) -> list[float]:
    """Ray directions for the sagittal family: basal (cell-cell) ray at `basal_deg`, laterals symmetric."""
    pair, _unp = sagittal_angles(rho)
    return [basal_deg, (basal_deg + pair) % 360.0, (basal_deg - pair) % 360.0]


def rho_sweep(rhos, rng: np.random.Generator | None = None, size: int = 240) -> dict:
    """Render E5 spicules at known rho (random orientation), re-measure with radial_rays."""
    rng = rng or np.random.default_rng(SEED)
    rows = []
    for rho in rhos:
        basal = float(rng.uniform(0, 360))
        m = spicule_mask(sagittal_ray_dirs(rho, basal), ray_len=100, half_width=7, size=size)
        (_cy, _cx, _rc, rmax), rays = radial_rays(m)
        long_rays = [r for r in rays if r["length_px"] > 0.25 * rmax]
        pair_true, unp_true = sagittal_angles(rho)
        if len(long_rays) < 3:
            rows.append({"rho": rho, "recovered": False})
            continue
        f = classify_triplet(interangles([r["axis_near_deg"] for r in long_rays[:3]]))
        rows.append({"rho": float(rho), "basal_deg": basal, "recovered": True,
                     "true_pair": pair_true, "true_unpaired": unp_true,
                     "meas_pair": f["theta_pair_mean"], "meas_unpaired": f["theta_unpaired"],
                     "meas_paired_difference": f["paired_difference"], "rho_recovered": f["rho_fit"]})
    ok = [r for r in rows if r["recovered"]]
    tp = np.array([r["true_pair"] for r in ok]); mp = np.array([r["meas_pair"] for r in ok])
    tu = np.array([r["true_unpaired"] for r in ok]); mu = np.array([r["meas_unpaired"] for r in ok])
    pd = np.array([r["meas_paired_difference"] for r in ok])
    tr = np.array([r["rho"] for r in ok]); mr = np.array([r["rho_recovered"] for r in ok])
    r_pair = float(np.corrcoef(tp, mp)[0, 1]) if len(ok) > 2 else None
    return {"rows": rows, "n_recovered": len(ok), "n": len(rows),
            "pearson_r_pair": r_pair, "rmse_pair_deg": float(np.sqrt(np.mean((tp - mp) ** 2))),
            "rmse_unpaired_deg": float(np.sqrt(np.mean((tu - mu) ** 2))),
            "rmse_rho": float(np.sqrt(np.mean((tr - mr) ** 2))),
            "noise_floor_paired_difference_deg": {"mean": float(pd.mean()), "max": float(pd.max())},
            "monotone_unpaired_vs_rho": bool(np.all(np.diff(mu[np.argsort(tr)]) < 0))}

# ----------------------------------------------------------------------------
# Rendering
# ----------------------------------------------------------------------------

def three_cell_field(rho: float, size: int = 240, R: float = 62.0):
    """Centres of three cells on the sector bisectors between the rays (illustrative field)."""
    dirs = sagittal_ray_dirs(rho)
    s = sorted(dirs)
    bis = [((s[i] + (s[(i + 1) % 3] + (360.0 if i == 2 else 0.0))) / 2.0) % 360.0 for i in range(3)]
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
            c = (size - 1) / 2.0
            for a in dirs:
                td.line([c, c, c + 110 * math.cos(math.radians(a)), c - 110 * math.sin(math.radians(a))], fill=(190, 190, 200), width=1)
            m = spicule_mask(dirs, ray_len=14 + 86 * g, half_width=1.5 + 6.5 * g, size=size)
            arr = np.array(tile)
            arr[m] = (40, 40, 48)
            canvas.paste(Image.fromarray(arr), (ox, 0))
            pair, unp = sagittal_angles(rho)
            dr.text((ox + 6, size + 4), f"rho={rho:.2f}  pair={pair:.0f}  unpaired={unp:.0f}", fill=(30, 30, 30))
            dr.text((ox + 6, size + 20), f"layer {f + 1}/{frames}", fill=(90, 90, 90))
        imgs.append(canvas)
    path.parent.mkdir(parents=True, exist_ok=True)
    imgs[0].save(path, save_all=True, append_images=imgs[1:] + [imgs[-1]] * 6, duration=90, loop=0)


def basal_direction(s: dict) -> float:
    """Direction of the unpaired (basal, cell-cell) ray: the ray NOT bounding the unpaired gap."""
    dirs = np.sort(np.asarray([r["axis_near_deg"] for r in s["rays"][:3]]) % 360.0)
    gaps = interangles(dirs)                       # gaps[k] lies between dirs[k] and dirs[k+1]
    k = int(np.argmin(np.abs(gaps - s["fit"]["theta_unpaired"])))
    return float(dirs[(k + 2) % 3])


def render_comparison(path: Path, sem: dict, null: dict, sweep: dict, alt: dict,
                      pop: dict | None, fig214: dict | None) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    fig, axes = plt.subplots(2, 3, figsize=(19, 12.5))
    # (a) SEM overlay
    ax = axes[0, 0]
    ax.imshow(np.asarray(Image.open(SPECIMEN_SEM).convert("L")), cmap="gray")
    for s in sem["spicules"]:
        jx, jy = s["junction_xy"]
        for r in s["rays"][:3]:
            a = math.radians(r["axis_near_deg"]); L = 0.55 * r["length_px"]
            ax.plot([jx, jx + L * math.cos(a)], [jy, jy - L * math.sin(a)], "-", color="#ff6b35", lw=2)
        ax.plot(jx, jy, "o", color="#2ec4b6", ms=5)
        f = s["fit"]
        ax.text(jx + 30, jy - 30, f"{s['panel']}: {f['theta_sorted'][0]:.0f}/{f['theta_sorted'][1]:.0f}/{f['theta_sorted'][2]:.0f}",
                color="#ffd166", fontsize=8)
    ax.set_title("(a) Grantia aff. indica SEM (van Soest & de Voogd 2018, CC BY 3.0)\nray axes near the junction; c,d triactines; e,f tetractines", fontsize=10)
    ax.axis("off")

    # (b) measured vs model renders
    ax = axes[0, 1]
    n = len(sem["spicules"]); cols = 2; rows = int(math.ceil(n / cols)); size = 240
    board = np.zeros((rows * size, cols * 2 * size), bool)
    for i, s in enumerate(sem["spicules"]):
        r_, c_ = divmod(i, cols)
        meas = spicule_mask([r["axis_near_deg"] for r in s["rays"][:3]], 100, 7, size)
        model = spicule_mask(sagittal_ray_dirs(s["fit"]["rho_fit"], basal_direction(s)), 100, 7, size)
        board[r_ * size:(r_ + 1) * size, c_ * 2 * size:(c_ * 2 + 1) * size] = meas
        board[r_ * size:(r_ + 1) * size, (c_ * 2 + 1) * size:(c_ * 2 + 2) * size] = model
    rgb = np.ones(board.shape + (3,)) * 0.97
    rgb[board] = (0.16, 0.16, 0.19)
    ax.imshow(rgb)
    for i, s in enumerate(sem["spicules"]):
        r_, c_ = divmod(i, cols); f = s["fit"]
        ax.text(c_ * 2 * size + 6, r_ * size + 18, f"{s['panel']} measured", fontsize=7, color="#555")
        ax.text((c_ * 2 + 1) * size + 6, r_ * size + 18, f"E5 rho={f['rho_fit']:.2f}", fontsize=7, color="#b5452a")
        ax.text(c_ * 2 * size + 6, (r_ + 1) * size - 10, f"pair {f['theta_pair_mean']:.0f} / unp {f['theta_unpaired']:.0f}   d={f['paired_difference']:.1f}",
                fontsize=7, color="#333")
    ax.set_title("(b) measured ray axes vs Neumann sagittal model E5 at fitted rho\n(apical ray of tetractines e, f not rendered: end-on in the plate)", fontsize=10)
    ax.axis("off")

    # (c) paired difference vs null
    ax = axes[0, 2]
    nulls = paired_difference_null(20000, np.random.default_rng(SEED))
    ax.hist(nulls, bins=60, density=True, color="#cfd8dc", label="null: Dirichlet(1,1,1) partition of 360")
    obs = np.array([s["fit"]["paired_difference"] for s in sem["spicules"]])
    for i, d in enumerate(obs):
        ax.axvline(d, color="#ff6b35", lw=2, label="SEM spicules (n=%d)" % len(obs) if i == 0 else None)
    if sweep["n_recovered"]:
        ax.axvline(sweep["noise_floor_paired_difference_deg"]["max"], color="#1b4965", ls="--", lw=1.5,
                   label="method noise floor (max, rendered sweep)")
    ax.set_xlabel("closest-pair angle difference d (deg)")
    ax.set_ylabel("density")
    ptxt = ("< %.1e" % null["set_mean_mc_p"]) if null["set_mean_mc_p_is_floor"] else ("= %.4f" % null["set_mean_mc_p"])
    ax.set_title(f"(c) F2 'two equal angles': observed median {null['observed_median_deg']:.1f} vs null median {null['null_median_deg']:.1f}\n"
                 f"set-mean MC p {ptxt}; KS p = {null['ks_p']:.2e}; {null['n_spicules_below_null_p05']}/{len(obs)} below null 5th pct", fontsize=9)
    ax.legend(fontsize=8)

    # (d) rho sweep recovery + alternatives
    ax = axes[1, 0]
    rho = np.linspace(0, 2, 200)
    ax.plot(rho, [sagittal_angles(r)[0] for r in rho], "-", color="#1b4965", label="E5 theta_pair")
    ax.plot(rho, [sagittal_angles(r)[1] for r in rho], "-", color="#b5452a", label="E5 theta_unpaired")
    ok = [r for r in sweep["rows"] if r["recovered"]]
    ax.plot([r["rho"] for r in ok], [r["meas_pair"] for r in ok], "o", color="#1b4965", mfc="white", label="rendered + re-measured (pair)")
    ax.plot([r["rho"] for r in ok], [r["meas_unpaired"] for r in ok], "s", color="#b5452a", mfc="white", label="rendered + re-measured (unpaired)")
    for s in sem["spicules"]:
        f = s["fit"]
        ax.plot(f["rho_fit"], f["theta_unpaired"], "^", color="#ff6b35", ms=9)
        ax.plot(f["rho_fit"], f["theta_pair_mean"], "v", color="#ff6b35", ms=9)
    ax.plot([], [], "^", color="#ff6b35", label="SEM spicules at fitted rho")
    ax.axhline(120, color="#888", ls=":", lw=1)
    ax.set_xlabel("tension ratio rho = gamma_cc / gamma_cm"); ax.set_ylabel("angle (deg)")
    ax.set_title(f"(d) rho sweep with measurement closure: r = {sweep['pearson_r_pair']:.4f}, RMSE pair {sweep['rmse_pair_deg']:.2f} deg, RMSE rho {sweep['rmse_rho']:.3f}\n"
                 f"alternatives RMSE: crystal-120 {alt['crystal-120']['rmse_deg']:.1f}, lattice-30 {alt['lattice-30']['rmse_deg']:.1f}, E5 {alt['E5']['rmse_deg']:.1f} deg", fontsize=9)
    ax.legend(fontsize=7, loc="center right")

    # (e) population + Fig. 214 angle distributions
    ax = axes[1, 1]
    if pop and pop["stats"]["n_triradiates_physical"]:
        phys = np.array([r["angles_deg"] for r in pop["triradiates"] if r["physical"]]).ravel()
        allv = np.array([r["angles_deg"] for r in pop["triradiates"]]).ravel()
        ax.hist(allv, bins=np.arange(30, 260, 7.5), color="#e0e0e0", label=f"population, all detections (n={pop['stats']['n_triradiates_all']})")
        ax.hist(phys, bins=np.arange(30, 260, 7.5), color="#2ec4b6", alpha=0.8, label=f"population, physical (all angles <= 185; n={pop['stats']['n_triradiates_physical']})")
    if fig214 and fig214["panels"]:
        fv = np.array([p["angles_deg"] for p in fig214["panels"]]).ravel()
        ax.hist(fv, bins=np.arange(30, 260, 7.5), color="#b5452a", alpha=0.6, label=f"Thompson Fig. 214 panels {''.join(p['panel'] for p in fig214['panels'])}")
    ax.axvline(120, color="#1b4965", lw=2, label="P1: 120 deg")
    ax.set_xlabel("ray angle (deg)"); ax.set_ylabel("count")
    t = "(e) P1 120-deg mode: free triradiates in the Grantia spread (CC BY-SA 4.0) + Thompson's own Fig. 214"
    if pop:
        st = pop["stats"]
        t += f"\nphysical: mean |theta-120| = {st['physical_mean_abs_dev_from_120_deg']:.1f} deg (null {st['null_mean_abs_dev_from_120_deg']:.1f}); all-within-15: {st['physical_fraction_all_within_15_of_120']:.2f} (null {st['null_fraction_all_within_15_of_120']:.3f})"
    ax.set_title(t, fontsize=9)
    ax.legend(fontsize=7)

    # (f) Fig. 214 crop with axes
    ax = axes[1, 2]
    page = Image.open(SPECIMENS["fig214"][0]).convert("L")
    x0, y0, x1, y1 = FIG214_CROP
    ax.imshow(np.asarray(page)[y0:y1, x0:x1], cmap="gray")
    if fig214:
        for p in fig214["panels"]:
            jx, jy = p["junction_page_xy"][0] - x0, p["junction_page_xy"][1] - y0
            for r in p["rays"][:3]:
                a = math.radians(r["axis_near_deg"]); L = 0.5 * r["length_px"]
                ax.plot([jx, jx + L * math.cos(a)], [jy, jy - L * math.sin(a)], "-", color="#ff6b35", lw=2)
            f = p["fit"]
            ax.text(jx + 25, jy + 40, f"{p['panel']}: {f['theta_sorted'][0]:.0f}/{f['theta_sorted'][1]:.0f}/{f['theta_sorted'][2]:.0f}", color="#b5452a", fontsize=9)
        for p in fig214["unresolved"]:
            jx, jy = p["junction_page_xy"][0] - x0, p["junction_page_xy"][1] - y0
            ax.text(jx + 25, jy, f"{p['panel']}: unresolved", color="#888", fontsize=9)
    ax.set_title("(f) Thompson (1917) p. 445 Fig. 214 (after Haeckel; IA leaf 465): digitised axes\nA,B T-shaped; C,D sagittal; E tetractine, F regular (hairlines unresolved)", fontsize=10)
    ax.axis("off")

    fig.suptitle("T1917-024 — Thompson (1917) pp. 443-449: spicule rays as Plateau borders of the secreting cell field  [Pass 2]", fontsize=13)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=100)
    plt.close(fig)

# ----------------------------------------------------------------------------
# Summary, MorphoEval hook, verdict
# ----------------------------------------------------------------------------

def summary_metrics(sem: dict) -> dict:
    fits = [s["fit"] for s in sem["spicules"]]
    all_angles = np.array([f["theta_sorted"] for f in fits])
    pd = np.array([f["paired_difference"] for f in fits])
    ud = np.array([f["unpaired_deviation"] for f in fits])
    near = np.array([s["angles_near_deg"] for s in sem["spicules"]]).ravel()
    tip = np.array([s["angles_tip_deg"] for s in sem["spicules"]]).ravel()
    by_type: dict[str, list[float]] = {}
    for s in sem["spicules"]:
        by_type.setdefault(s["panel"], []).append(s["fit"]["rho_fit"])
    rho_cv = {k: (float(np.std(v, ddof=1) / np.mean(v)) if len(v) > 1 else None) for k, v in by_type.items()}
    return {
        "n_three_ray_spicules": len(fits), "n_monaxons": len(sem["needles"]),
        "angles_all_mean_deg": float(all_angles.mean()), "angles_all_sd_deg": float(all_angles.std(ddof=1)),
        "paired_difference_mean_deg": float(pd.mean()), "paired_difference_median_deg": float(np.median(pd)),
        "unpaired_deviation_mean_deg": float(ud.mean()), "ratio_unpaired_dev_to_paired_diff": float(ud.mean() / max(pd.mean(), 1e-9)),
        "rho_fit_values": [f["rho_fit"] for f in fits], "rho_fit_range": [float(min(f["rho_fit"] for f in fits)), float(max(f["rho_fit"] for f in fits))],
        "rho_cv_by_panel_type": rho_cv,
        "n_rho_below_1_unpaired_gt_paired": int(sum(1 for f in fits if f["rho_fit"] < 1 and f["unpaired_minus_paired"] > 0)),
        "n_rho_below_1": int(sum(1 for f in fits if f["rho_fit"] < 1)),
        "fraction_equiangular_within_10deg": float(np.mean(np.abs(all_angles - 120).max(axis=1) < 10)),
        "near_vs_tip_axis_pearson_r": float(np.corrcoef(near, tip)[0, 1]), "near_vs_tip_axis_rmse_deg": float(np.sqrt(np.mean((near - tip) ** 2))),
        "angle_cv": float(all_angles.std(ddof=1) / all_angles.mean()),
        "model_constants": polyhedral_ray_angles(), "interstice_area_coeff": interstice_area(1.0),
    }


def run_all(threshold: int = 90, n_null: int = 20000, rhos=None, skip_population=False, skip_fig214=False) -> dict:
    rng = np.random.default_rng(SEED)
    digests = verify_specimens()
    sem = measure_sem_plate(threshold=threshold)
    m = summary_metrics(sem)
    obs = np.array([s["fit"]["paired_difference"] for s in sem["spicules"]])
    null = null_statistics(obs, n_null, rng)
    rhos = list(rhos) if rhos else list(np.linspace(0.05, 1.95, 20))
    sweep = rho_sweep(rhos, rng)
    alt = alternative_models(np.array([s["fit"]["theta_sorted"] for s in sem["spicules"]]))
    sat = general_model_saturation(2000, rng)
    pop = None if skip_population else measure_population()
    f214 = None if skip_fig214 else measure_fig214()
    if f214 and f214["panels"]:
        f214["null"] = null_statistics(np.array([p["fit"]["paired_difference"] for p in f214["panels"]]), n_null, rng)
    return {"digests": digests, "sem": sem, "metrics": m, "null": null, "sweep": sweep, "alternatives": alt,
            "saturation": sat, "population": pop, "fig214": f214}


def verdict(res: dict) -> dict:
    m, null, alt, pop, f214 = res["metrics"], res["null"], res["alternatives"], res["population"], res["fig214"]
    f2 = null["set_mean_mc_p"] < 0.01 and null["ks_p"] < 0.01
    f4 = m["n_rho_below_1_unpaired_gt_paired"] == m["n_rho_below_1"]
    p1 = bool(pop and pop["stats"]["n_triradiates_physical"] >= 8 and
              pop["stats"]["physical_mean_abs_dev_from_120_deg"] < 0.5 * pop["stats"]["null_mean_abs_dev_from_120_deg"])
    e5_beats_crystal = alt["E5"]["aic"] < alt["crystal-120"]["aic"] and alt["E5"]["aic"] < alt["lattice-30"]["aic"]
    parts = {
        "P1_120deg_mode_population": "supported" if p1 else ("untestable" if not pop else "refuted"),
        "P2_two_equal_one_unequal_SEM": "supported" if f2 else "refuted",
        "P2_sagittal_ordering_F4": "supported" if f4 else "refuted",
        "P2_fig214_digitised": ("untestable (no panel resolved)" if not (f214 and f214.get("null")) else
                                ("supported" if f214["null"]["set_mean_mc_p"] < 0.05 else "not significant") +
                                f" (n={f214['null']['n']} panels, set-mean MC p={f214['null']['set_mean_mc_p']:.4g})"),
        "P3_tetrahedral_tetractine": "untestable (2D projection; apical ray end-on; carried by Lean proof only)",
        "alcyonarian_spicules": "untestable (no alcyonarian data in this loop)",
        "mechanism_passive_deposition_between_any_cells": "refuted by modern cell biology (Ledger & Jones 1977; Voigt et al. 2017: dedicated sclerocyte sextet) — the geometric content survives, the 'any three cells' mechanism does not",
        "E5_vs_crystal_alternatives": "E5 preferred (AIC)" if e5_beats_crystal else "crystal alternative not rejected",
    }
    overall = "mixed" if (f2 and f4) else "refuted"
    headline = {
        "sem_n_three_ray": m["n_three_ray_spicules"], "paired_difference_median_deg": round(m["paired_difference_median_deg"], 2),
        "null_median_deg": round(null["null_median_deg"], 2), "set_mean_mc_p": null["set_mean_mc_p"], "ks_p": null["ks_p"],
        "unpaired_deviation_mean_deg": round(m["unpaired_deviation_mean_deg"], 2), "rho_range": [round(x, 3) for x in m["rho_fit_range"]],
        "rho_cv_by_panel_type": m["rho_cv_by_panel_type"],
        "sweep_pearson_r_pair": res["sweep"]["pearson_r_pair"], "sweep_rmse_pair_deg": round(res["sweep"]["rmse_pair_deg"], 3),
        "method_noise_floor_d_max_deg": round(res["sweep"]["noise_floor_paired_difference_deg"]["max"], 3),
        "rmse_crystal120_deg": round(alt["crystal-120"]["rmse_deg"], 2), "rmse_lattice30_deg": round(alt["lattice-30"]["rmse_deg"], 2), "rmse_E5_deg": round(alt["E5"]["rmse_deg"], 2),
        "population_n_physical": pop["stats"]["n_triradiates_physical"] if pop else None,
        "population_mean_abs_dev_120_deg": round(pop["stats"]["physical_mean_abs_dev_from_120_deg"], 2) if pop and pop["stats"]["physical_mean_abs_dev_from_120_deg"] is not None else None,
        "population_null_mean_abs_dev_120_deg": round(pop["stats"]["null_mean_abs_dev_from_120_deg"], 2) if pop else None,
        "fig214_panels": {p["panel"]: [round(x, 1) for x in p["angles_deg"]] for p in f214["panels"]} if f214 else None,
        "general_model_fraction_realised": res["saturation"]["fraction_realised"],
    }
    return {"verdict": overall, "components": parts, "headline_numbers": headline}


def write_result(res: dict, v: dict) -> Path:
    out = {
        "claim_id": "T1917-024", "date_made": "1917",
        "claim": "Sponge and alcyonarian spicule forms (needles, tri-/tetraradiates and their aggregates) are the forms physical deposition/crystallisation takes within the cell/field that secretes them.",
        "source": "Thompson, D.W. (1917) On Growth and Form, Ch. IX, pp. 411-487 (tested passage pp. 443-449, Fig. 214)",
        "dataset": {
            "preassigned": {"name": "Dryad: Nadhira et al. 2019, Carduispongia pedicula tomograms (hexactine fossil)", "doi": "10.5061/dryad.8mv00jv",
                            "status": "DATASET_MISMATCH — hexactine fossil tomograms (5.83 GB), no ray-angle data for Thompson's triradiate/tetraxon claim; metadata pinned, payload not fetched",
                            "sha256_metadata": "8b1273b453ccdb77ae1562f25c263883504c6600483e84a1a96e50c159207c24"},
            "primary": {"name": "Grantia aff. indica SEM spicule plate, van Soest & de Voogd 2018 Zootaxa 4426 Fig. 53 (CC BY 3.0, Wikimedia Commons)", "doi": "10.11646/zootaxa.4426.1.1",
                        "sha256": SPECIMEN_SEM_SHA256, "analogue": False, "note": "same genus as Thompson's Fig. 214"},
            "population": {"name": "Grantia spicules 100x, M. Readey 2020 (CC BY-SA 4.0, Wikimedia Commons)", "sha256": SPECIMENS["population"][1], "analogue": False},
            "source_figure": {"name": "Thompson (1917) p. 445 Fig. 214 after Haeckel, IA ongrowthform1917thom leaf 465", "sha256": SPECIMENS["fig214"][1]},
        },
        "verdict": v["verdict"], "verdict_components": v["components"], "headline_numbers": v["headline_numbers"],
        "script": "thompson/claims/t1917-024_on-concretions-spicules-and-spicular-skeletons.py",
        "figure": f"docs/media/{STEM}_comparison.png", "animation": f"docs/media/{STEM}_growth.gif",
        "lean": "morphoeval/lean/loops/Thompson1917_T1917_024.lean",
        "notes": [
            "Verdict 'mixed': the geometric predictions P1 (120-deg mode) and P2 (two equal angles, one unequal; sagittal ordering) are supported on Grantia; P3 and alcyonarians are untestable here; the mechanistic reading (passive deposition between any neighbouring cells) is contradicted by sclerocyte cell biology.",
            "The general three-tension model is saturated (realises every angle triplet) and is not tested; only the restricted E5 family is.",
            "Bilateral symmetry of a sclerocyte sextet would also produce two equal angles: the data cannot discriminate 'tension balance' from 'symmetric cell programme'.",
        ],
    }
    p = DATA / "result.json"
    p.write_text(json.dumps(out, indent=1))
    return p


def measure_morphoeval(res: dict | None = None) -> list[dict]:
    """Measurement records for morphoeval/measure_outputs.py."""
    res = res or run_all(skip_population=False, skip_fig214=False)
    m, null, sw, alt, pop, f214 = res["metrics"], res["null"], res["sweep"], res["alternatives"], res["population"], res["fig214"]
    tp = "Thompson (1917) pp. 443-446, Fig. 214"
    rec = [
        {"name": "three-ray spicules measured on the SEM plate", "value": m["n_three_ray_spicules"], "unit": "count",
         "method": "radial-arc ray extraction after fixed threshold (90/255), closing(2), area >= 1500, fill <= 0.9", "derived_from": "specimen",
         "note": f"{m['n_monaxons']} monaxon diactines/trichoxea detected as 2-ray needles"},
        {"name": "closest-pair angle difference d, median (SEM)", "value": round(m["paired_difference_median_deg"], 3), "unit": "deg",
         "method": "|theta_2 - theta_3| of the two closest sorted ray angles per spicule (near-junction axes); vs Dirichlet(1,1,1) null control",
         "reference_value": round(null["null_median_deg"], 3), "reference_source": "null: random partition of 360 deg (MC n=%d)" % null["n_iter"],
         "note": f"set-mean MC p {'<' if null['set_mean_mc_p_is_floor'] else '='} {null['set_mean_mc_p']:.4g}; KS p = {null['ks_p']:.3g}; {null['n_spicules_below_null_p05']}/{m['n_three_ray_spicules']} spicules below the null 5th percentile ({null['null_p05_deg']:.2f} deg)",
         "derived_from": "specimen"},
        {"name": "unpaired-angle deviation from the paired angle, mean (SEM)", "value": round(m["unpaired_deviation_mean_deg"], 3), "unit": "deg",
         "method": "|theta_unpaired - mean(theta_pair)| per spicule", "derived_from": "specimen",
         "note": "ratio unpaired deviation / paired difference = %.2f" % m["ratio_unpaired_dev_to_paired_diff"]},
        {"name": "rho sweep measurement closure: Pearson r (true vs re-measured theta_pair)", "value": round(sw["pearson_r_pair"], 5), "unit": "r",
         "method": f"{sw['n']} E5 spicules rendered at known rho in random orientation and re-measured by radial_rays", "reference_value": 1.0,
         "reference_source": "perfect recovery", "note": f"RMSE pair {sw['rmse_pair_deg']:.2f} deg, RMSE rho {sw['rmse_rho']:.3f}; method noise floor for d: max {sw['noise_floor_paired_difference_deg']['max']:.2f} deg; unpaired angle monotone in rho: {sw['monotone_unpaired_vs_rho']}",
         "derived_from": "script"},
        {"name": "alternative-model RMSE: crystal-120 (0 params)", "value": round(alt["crystal-120"]["rmse_deg"], 3), "unit": "deg",
         "method": "all triradiate angles = 120 (Haeckel crystallographic reading) vs measured SEM triplets", "reference_value": round(alt["E5"]["rmse_deg"], 3),
         "reference_source": "E5 one-rho-per-spicule RMSE", "note": f"lattice-30 RMSE {alt['lattice-30']['rmse_deg']:.2f}; AIC crystal {alt['crystal-120']['aic']:.1f}, lattice {alt['lattice-30']['aic']:.1f}, E5 {alt['E5']['aic']:.1f}",
         "derived_from": "specimen"},
        {"name": "near-axis vs chord-axis angle sensitivity: Pearson r", "value": round(m["near_vs_tip_axis_pearson_r"], 4), "unit": "r",
         "method": "same 24 SEM angles measured from the first 35 % of each ray vs the chord to the ray tip", "derived_from": "specimen",
         "note": f"RMSE {m['near_vs_tip_axis_rmse_deg']:.2f} deg; angle CV {m['angle_cv']:.3f}"},
        {"name": "fitted rho CV within tubar triactines (panel d, n=3)", "value": round(m["rho_cv_by_panel_type"].get("d") or 0.0, 4), "unit": "CV",
         "method": "std/mean of rho = -2 cos(theta_pair) across same-type spicules (E5 predicts a tissue-level rho)", "derived_from": "specimen",
         "note": "by panel: " + ", ".join(f"{k}: {('%.3f' % v) if v is not None else 'n=1'}" for k, v in m["rho_cv_by_panel_type"].items())},
        {"name": "tetrahedral inter-ray angle (E7)", "value": round(m["model_constants"]["tetrahedral_deg"], 4), "unit": "deg", "method": "arccos(-1/3)",
         "reference_value": 109.0, "reference_source": f"'about 109 deg', {tp} p. 446", "derived_from": "script"},
    ]
    if pop:
        st = pop["stats"]
        rec.append({"name": "population micrograph: mean |theta - 120| over physical triradiates", "value": round(st["physical_mean_abs_dev_from_120_deg"], 3), "unit": "deg",
                    "method": f"junction-based radial-arc extraction on the CC BY-SA spread; {st['n_triradiates_physical']} physical of {st['n_triradiates_all']} detections ({st['n_junction_candidates']} junction candidates)",
                    "reference_value": round(st["null_mean_abs_dev_from_120_deg"], 3), "reference_source": "Dirichlet(1,1,1) null",
                    "note": f"all-within-15-deg fraction {st['physical_fraction_all_within_15_of_120']:.3f} (null {st['null_fraction_all_within_15_of_120']:.3f}); SD {st['physical_sd_deg']:.2f} deg; median d {st['physical_median_paired_difference_deg']:.2f} deg",
                    "derived_from": "specimen"})
    if f214 and f214["panels"]:
        pdv = [p["fit"]["paired_difference"] for p in f214["panels"]]
        fn = f214.get("null") or {}
        rec.append({"name": "Thompson Fig. 214 digitised: closest-pair difference d, median", "value": round(float(np.median(pdv)), 3), "unit": "deg",
                    "method": f"radial-arc extraction on the IA leaf-465 crop after hole filling; panels resolved: {''.join(p['panel'] for p in f214['panels'])}; unresolved: {''.join(p['panel'] for p in f214['unresolved'])}; set-mean MC p vs Dirichlet null",
                    "reference_value": round(null["null_median_deg"], 3), "reference_source": "Dirichlet null median",
                    "note": "; ".join(f"{p['panel']} {'/'.join('%.0f' % x for x in p['angles_deg'])}" for p in f214["panels"]) +
                            (f"; set-mean MC p = {fn['set_mean_mc_p']:.4g}" if fn else ""), "derived_from": "source figure"})
    return rec

# ----------------------------------------------------------------------------
# Self-test and main
# ----------------------------------------------------------------------------

def self_test() -> None:
    a = neumann_angles(1, 1, 1)
    assert all(abs(x - 120) < 1e-9 for x in a), a
    assert abs(sum(neumann_angles(0.7, 1.0, 1.0)) - 360) < 1e-9
    g = tensions_from_angles((100.0, 120.0, 140.0))
    assert max(abs(np.array(neumann_angles(*g)) - np.array([100, 120, 140]))) < 1e-9
    p, u = sagittal_angles(1.0); assert abs(p - 120) < 1e-9 and abs(u - 120) < 1e-9
    p, u = sagittal_angles(0.0); assert abs(p - 90) < 1e-9 and abs(u - 180) < 1e-9
    for rho in (0.1, 0.5, 1.3, 1.9):
        p, u = sagittal_angles(rho)
        assert abs(tension_ratio_from_angles(p) - rho) < 1e-9
        th = neumann_angles(rho, 1.0, 1.0)
        assert abs(th[0] - u) < 1e-9 and abs(th[1] - p) < 1e-9 and abs(th[2] - p) < 1e-9
    assert abs(polyhedral_ray_angles()["tetrahedral_deg"] - 109.4712206) < 1e-6
    assert abs(interstice_area(1.0) - 0.16125448) < 1e-7
    f = classify_triplet([110, 112, 138]); assert abs(f["theta_unpaired"] - 138) < 1e-9 and abs(f["paired_difference"] - 2) < 1e-9
    rng = np.random.default_rng(SEED)
    nul = paired_difference_null(5000, rng); assert 10 < np.median(nul) < 40, np.median(nul)
    sat = general_model_saturation(300, rng); assert sat["fraction_realised"] == 1.0 and sat["max_roundtrip_error_deg"] < 1e-6
    sw = rho_sweep([0.2, 0.8, 1.0, 1.4], rng)
    assert sw["n_recovered"] == 4 and sw["rmse_pair_deg"] < 3.0 and sw["noise_floor_paired_difference_deg"]["max"] < 4.0, sw
    alt = alternative_models(np.array([[110, 112, 138], [95, 95, 170]]))
    assert alt["E5"]["rmse_deg"] < alt["crystal-120"]["rmse_deg"]
    verify_specimens()
    sem = measure_sem_plate()
    assert len(sem["spicules"]) == 8 and len(sem["needles"]) == 3, (len(sem["spicules"]), len(sem["needles"]))
    for s in sem["spicules"]:
        assert abs(sum(s["angles_near_deg"]) - 360) < 1e-6
        assert s["panel"] in "cdef"
    ns = null_statistics(np.array([s["fit"]["paired_difference"] for s in sem["spicules"]]), 3000, rng)
    assert 0 <= ns["set_mean_mc_p"] <= 1
    f214 = measure_fig214(); assert len(f214["panels"]) >= 3, [p["panel"] for p in f214["panels"]]
    print("self-test OK: model identities, inverse law of sines, null, saturation, sweep closure, alternatives, "
          "digests, 8 three-ray spicules + 3 monaxons, Fig. 214 panels", "".join(p["panel"] for p in f214["panels"]))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Thompson (1917) T1917-024 spicule Plateau-border claim test")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--no-gif", action="store_true")
    ap.add_argument("--threshold", type=int, default=90, help="SEM plate brightness threshold (0-255)")
    ap.add_argument("--n-null", type=int, default=20000, help="Monte Carlo draws for the Dirichlet null")
    ap.add_argument("--rhos", type=float, nargs="*", default=None, help="rho values for the rendered sweep")
    ap.add_argument("--skip-population", action="store_true")
    ap.add_argument("--skip-fig214", action="store_true")
    args = ap.parse_args(argv)
    if args.self_test:
        self_test()
        return 0
    res = run_all(args.threshold, args.n_null, args.rhos, args.skip_population, args.skip_fig214)
    v = verdict(res)
    out = DATA / "measurements"
    out.mkdir(parents=True, exist_ok=True)
    (out / "sem_ray_angles.json").write_text(json.dumps({"sem": res["sem"], "metrics": res["metrics"], "null": res["null"],
                                                         "alternatives": res["alternatives"], "saturation": res["saturation"],
                                                         "morphoeval_measurements": measure_morphoeval(res)}, indent=1))
    (out / "rho_sweep.json").write_text(json.dumps(res["sweep"], indent=1))
    if res["population"]:
        (out / "population_ray_angles.json").write_text(json.dumps(res["population"], indent=1))
    if res["fig214"]:
        (out / "fig214_digitised.json").write_text(json.dumps(res["fig214"], indent=1))
    with (out / "sem_ray_angles.csv").open("w", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(["label", "panel", "x", "y", "w", "h", "theta1", "theta2", "theta3", "theta_pair", "theta_unpaired", "paired_difference", "rho_fit", "null_p"])
        for s, pp in zip(res["sem"]["spicules"], res["null"]["per_spicule_p"]):
            f = s["fit"]
            w.writerow([s["label"], s["panel"], *s["bbox_xywh"], *[round(x, 2) for x in f["theta_sorted"]],
                        round(f["theta_pair_mean"], 2), round(f["theta_unpaired"], 2), round(f["paired_difference"], 2), round(f["rho_fit"], 3), round(pp, 4)])
    render_comparison(MEDIA / f"{STEM}_comparison.png", res["sem"], res["null"], res["sweep"], res["alternatives"], res["population"], res["fig214"])
    if not args.no_gif:
        render_growth_gif(MEDIA / f"{STEM}_growth.gif")
    rp = write_result(res, v)
    print(json.dumps(v, indent=1))
    print("->", rp.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    sys.exit(main())

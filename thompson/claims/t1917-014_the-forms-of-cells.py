#!/usr/bin/env python3
"""
t1917-014_the-forms-of-cells.py — Thompson (1917) claim T1917-014, Pass 1 baseline (v1).

Claim (On Growth and Form, Ch. V "The Forms of Cells", pp. 201-276, Figs. 61-67, 72-96):
    "The forms of single cells and cell-like organisms (Plateau's oil globules; Lagena and
    other forams/protists) are surfaces of revolution from the minimal-surface catalogue:
    sphere, cylinder, unduloid, nodoid, catenoid."
Thompson's own wording for the test genus (p. 256, Fig. 84 "Various species of Lagena, after
Brady"): "flask-shaped shells, whose surface is simply that of an unduloid, or more frequently,
like that of a flask itself, an unduloid combined with a portion of a sphere."

Mathematics implemented (ids follow datasets/thompson_1917/t1917-014/*_reference.json):
    E3  Delaunay first integral  r / sqrt(1 + r'^2) - H r^2 = C ; turning points r_min, r_max
        are the roots of H r^2 - r + C = 0  ->  r_min + r_max = 1/H, r_min r_max = C/H.
    E4  Kenmotsu (1980) arc-length parametrisation of every CMC surface of revolution with
        H != 0:  r(s) = sqrt(1 + B^2 + 2 B sin 2Hs) / (2H),  dz/ds = (1 + B sin 2Hs) / sqrt(...).
        B = 0 cylinder, 0 < B < 1 unduloid, B = 1 string of spheres, B > 1 nodoid;
        the catenoid is the H = 0 member r = c cosh(z / c).
    E5  neck ratio rho = r_min / r_max = (1 - B) / (1 + B).
    E6  unduloid period Lambda = 4 a E(e), e = B; bead aspect ratio Lambda / (2 r_max) <= pi.
    E7  "Delaunay flask": sphere (radius R) C1-joined to an unduloid whose neck is C1-joined to
        a cylinder — Thompson's "unduloid combined with a portion of a sphere" made explicit.
    E8  "Delaunay bead": a single unduloid bulge-to-neck window continued by its neck cylinder.

Data: Thompson's Fig. 84 (13 Lagena silhouettes, Internet Archive leaf 0276 of
ongrowthform1917thom, public domain), SHA-256 pinned in the reference JSON.  The pre-scouted
Dryad dataset doi:10.5061/dryad.28816 was verified as DATASET_MISMATCH (see scout_manifest.json)
and is not used.

Usage:
    python3 thompson/claims/t1917-014_the-forms-of-cells.py --self-test
    python3 thompson/claims/t1917-014_the-forms-of-cells.py          # writes figure, GIF, metrics

Outputs:
    docs/media/t1917_014_the_forms_of_cells_comparison.png
    docs/media/t1917_014_the_forms_of_cells_growth.gif
    datasets/thompson_1917/t1917-014/t1917_014_metrics.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from scipy.optimize import least_squares
from scipy.special import ellipe

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "datasets" / "thompson_1917" / "t1917-014"
MEDIA_DIR = ROOT / "docs" / "media"
REFERENCE_JSON = DATA_DIR / "t1917_014_the_forms_of_cells_reference.json"
FIG84_PNG = DATA_DIR / "thompson_1917_fig84_lagena_after_brady.png"
FIG84_SHA256 = "c02d29af384791df4ad18705c466f233d111be828e991d64047b5b8fea00dec9"
COMPARISON_PNG = MEDIA_DIR / "t1917_014_the_forms_of_cells_comparison.png"
GROWTH_GIF = MEDIA_DIR / "t1917_014_the_forms_of_cells_growth.gif"
METRICS_JSON = DATA_DIR / "t1917_014_metrics.json"

# Segmentation of the 1917 plate (grey paper, show-through text, shaded line drawings).
SEG_BG_PERCENTILE = 90          # local paper brightness estimate
SEG_BG_SIZE = 61                # footprint of the percentile filter on the 4x-downsampled plate
SEG_THRESHOLD = 0.045           # darker-than-paper threshold (fraction of full scale)
SEG_MIN_AREA = 3000             # px; drops show-through text fragments
SEG_ROW_SPLIT = 700             # Fig. 84 has an upper row (8) and a lower row (5) of specimens

# Delaunay-compatibility decision (Pass 1): normalised RMSE of the best catalogue fit.
RMSE_SUPPORT = 0.06             # 6 % of r_max (edge noise of the drawings is ~2-4 %)


# --------------------------------------------------------------------------- provenance
def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_inputs() -> dict:
    ref = json.loads(REFERENCE_JSON.read_text(encoding="utf-8"))
    pins = ref["input_dataset_sha256"]
    got = sha256_of(FIG84_PNG)
    if got != pins[FIG84_PNG.name] or got != FIG84_SHA256:
        raise SystemExit(f"SHA-256 mismatch for {FIG84_PNG.name}: {got}")
    return ref


# --------------------------------------------------------------------------- E4-E6: Delaunay catalogue
def classify_delaunay(B: float, H: float = 1.0) -> str:
    """Plateau's catalogue member for Kenmotsu parameter B (H = 0 is the catenoid)."""
    if H == 0.0:
        return "catenoid"
    if B == 0.0:
        return "cylinder"
    if 0.0 < B < 1.0:
        return "unduloid"
    if B == 1.0:
        return "sphere"
    return "nodoid"


def neck_ratio(B: float) -> float:
    """E5: r_min / r_max = (1 - B) / (1 + B)."""
    return (1.0 - B) / (1.0 + B)


def unduloid_period(B: float, H: float) -> float:
    """E6: axial distance between successive necks, Lambda = 4 a E(e) with a = 1/(2H), e = B."""
    a = 1.0 / (2.0 * H)
    return 4.0 * a * ellipe(B * B)       # scipy's ellipe takes the parameter m = e^2


def bead_aspect_ratio(B: float) -> float:
    """E6: length / width of one unduloid bead = 2 E(e) / (1 + e); equals pi at B = 0, 1 at B = 1."""
    return 2.0 * ellipe(B * B) / (1.0 + B)


def kenmotsu_profile(H: float, B: float, n: int = 400) -> tuple[np.ndarray, np.ndarray]:
    """E4: descending half-period of the unduloid (bulge s = pi/4H -> neck s = 3pi/4H).

    Returns (dz, r): axial offset from the bulge (>= 0, trapezoid-integrated dz/ds) and radius.
    """
    s = np.linspace(np.pi / (4 * H), 3 * np.pi / (4 * H), n)
    sig = np.sin(2 * H * s)
    root = np.sqrt(1 + B * B + 2 * B * sig)
    r = root / (2 * H)
    dzds = (1 + B * sig) / root
    dz = np.concatenate([[0.0], np.cumsum(0.5 * (dzds[1:] + dzds[:-1]) * np.diff(s))])
    return dz, r


def kenmotsu_full(H: float, B: float, s: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """E4 over an arbitrary arc-length array (used for the catalogue gallery and the GIF)."""
    sig = np.sin(2 * H * s)
    root = np.sqrt(np.clip(1 + B * B + 2 * B * sig, 1e-12, None))
    r = root / (2 * H)
    dzds = (1 + B * sig) / root
    z = np.concatenate([[0.0], np.cumsum(0.5 * (dzds[1:] + dzds[:-1]) * np.diff(s))])
    return z, r


def catenoid_profile(c: float, z: np.ndarray) -> np.ndarray:
    """H = 0 member: r = c cosh(z / c)."""
    return c * np.cosh(z / c)


# --------------------------------------------------------------------------- E7-E8: composite models
def flask_profile(z: np.ndarray, R: float, B: float, t: float, Lneck: float, z0: float) -> np.ndarray:
    """E7 Delaunay flask.  Sphere of radius R with its aboral pole at z0, C1-joined to the
    descending branch of the unduloid (H = kappa / R, kappa = (1-B)/2 + t B), whose neck is
    C1-joined to a cylinder of length Lneck.  The join radius is the closed form
    r_j^2 = R (1 - B^2) / (4 H (1 - kappa)); t = 1 is the equator join (r_j = R)."""
    kappa = (1.0 - B) / 2.0 + t * B
    H = kappa / R
    rj = math.sqrt(max(R * (1 - B * B) / (4 * H * (1 - kappa)), 1e-12))
    rj = min(rj, R)
    zc = z0 + R
    zj = zc + math.sqrt(max(R * R - rj * rj, 0.0))
    dz_u, r_u = kenmotsu_profile(H, B)
    dz_join = float(np.interp(rj, r_u[::-1], dz_u[::-1]))
    z_u = zj + (dz_u - dz_join)
    z_neck, r_neck = z_u[-1], r_u[-1]
    r = np.zeros_like(z, dtype=float)
    sph = (z >= z0) & (z <= zj)
    r[sph] = np.sqrt(np.clip(R * R - (z[sph] - zc) ** 2, 0, None))
    und = (z > zj) & (z <= z_neck)
    r[und] = np.interp(z[und], z_u, r_u)
    cyl = (z > z_neck) & (z <= z_neck + Lneck)
    r[cyl] = r_neck
    return r


def bead_profile(z: np.ndarray, H: float, B: float, zc: float) -> np.ndarray:
    """E8 Delaunay bead: unduloid (H, B) with its bulge at zc, continued beyond the necks by the
    neck cylinder r = r_min."""
    dz_u, r_u = kenmotsu_profile(H, B)
    return np.interp(np.abs(z - zc), dz_u, r_u, right=r_u[-1])


# --------------------------------------------------------------------------- segmentation
def segment_fig84(path: Path) -> tuple[np.ndarray, list[tuple[tuple[slice, slice], np.ndarray]]]:
    """Background-normalised threshold -> closing -> hole fill -> opening -> area filter.
    Returns the grey plate and the 13 (bbox, mask) components ordered row-wise, left to right."""
    g = np.asarray(Image.open(path).convert("L"), dtype=float) / 255.0
    gs = ndi.gaussian_filter(g, 1.0)
    small = g[::4, ::4]
    bg_s = ndi.percentile_filter(small, SEG_BG_PERCENTILE, size=SEG_BG_SIZE)
    bg = ndi.zoom(bg_s, 4, order=1)
    pad = ((0, max(0, g.shape[0] - bg.shape[0])), (0, max(0, g.shape[1] - bg.shape[1])))
    bg = np.pad(bg, pad, mode="edge")[: g.shape[0], : g.shape[1]]
    m = (bg - gs) > SEG_THRESHOLD
    m = ndi.binary_closing(m, iterations=5)
    m = ndi.binary_fill_holes(m)
    m = ndi.binary_opening(m, iterations=3)
    lab, n = ndi.label(m)
    sizes = ndi.sum(m, lab, range(1, n + 1))
    comps = []
    for i, sl in enumerate(ndi.find_objects(lab)):
        if sizes[i] > SEG_MIN_AREA:
            comps.append((sl, lab[sl] == (i + 1)))
    comps.sort(key=lambda c: (c[0][0].start > SEG_ROW_SPLIT, c[0][1].start))
    return g, comps


def radius_profile(mask: np.ndarray) -> dict:
    """Principal-axis rotation, per-row left/right half-widths about a least-squares axis.
    Returns z (0 at the base, increasing toward the neck), mean radius r, the two half-widths,
    the asymmetry index mean|r_L - r_R| / r_max (surface-of-revolution premise) and the rotated mask."""
    ys, xs = np.nonzero(mask)
    cy, cx = ys.mean(), xs.mean()
    w, v = np.linalg.eigh(np.cov(np.vstack([ys - cy, xs - cx])))
    major = v[:, np.argmax(w)]
    ang = math.degrees(math.atan2(major[1], major[0]))
    rot = ndi.rotate(mask.astype(np.uint8), ang, order=0, reshape=True) > 0
    rows = np.nonzero(rot.any(axis=1))[0]
    left = np.array([np.nonzero(rot[i])[0][0] for i in rows], float)
    right = np.array([np.nonzero(rot[i])[0][-1] for i in rows], float)
    mid, wts = 0.5 * (left + right), right - left + 1.0
    A = np.vstack([rows, np.ones_like(rows)]).T.astype(float)
    coef = np.linalg.lstsq(A * wts[:, None], mid * wts, rcond=None)[0]
    axis = A @ coef
    rL, rR = axis - left, right - axis
    r = 0.5 * (rL + rR)
    z = rows.astype(float)
    k = max(3, len(z) // 7)
    if r[:k].mean() <= r[-k:].mean():      # put the narrower (apertural) end at high z
        z, r, rL, rR = z[::-1], r[::-1], rL[::-1], rR[::-1]
    z = np.abs(z - z[0])
    return {"z": z, "r": r, "rL": rL, "rR": rR,
            "asymmetry": float(np.mean(np.abs(rL - rR)) / r.max()),
            "rotated_mask": rot, "axis_angle_deg": ang}


# --------------------------------------------------------------------------- fitting
def _fit(fun, z, r, x0s, bounds):
    best = None
    for x0 in x0s:
        res = least_squares(lambda p: fun(z, *p) - r, x0, bounds=bounds, max_nfev=400)
        if best is None or res.cost < best.cost:
            best = res
    return best


def nrmse(model: np.ndarray, r: np.ndarray) -> float:
    return float(np.sqrt(np.mean((model - r) ** 2)) / r.max())


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    if a.std() == 0 or b.std() == 0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def fit_delaunay(z: np.ndarray, r: np.ndarray) -> dict:
    """Fit E7 (flask) and E8 (bead) to a radius profile; return parameters and goodness of fit."""
    rmax, L = float(r.max()), float(z.max())
    x0f = [[rmax, B, t, 0.2 * L, 0.0] for B in (0.3, 0.6, 0.85) for t in (0.5, 0.95)]
    f = _fit(flask_profile, z, r, x0f,
             ([0.3 * rmax, 0.01, 0.01, 0.0, -0.2 * L], [1.5 * rmax, 0.999, 1.0, L, 0.3 * L]))
    x0b = [[(1 + B) / (2 * rmax), B, zc] for B in (0.3, 0.6, 0.9) for zc in (0.3 * L, 0.5 * L, 0.7 * L)]
    b = _fit(bead_profile, z, r, x0b, ([0.2 / rmax, 0.01, -0.5 * L], [5.0 / rmax, 0.999, 1.5 * L]))
    rf, rb = flask_profile(z, *f.x), bead_profile(z, *b.x)
    out = {
        "flask": {"R": float(f.x[0]), "B": float(f.x[1]), "t": float(f.x[2]),
                  "L_neck_over_R": float(f.x[3] / f.x[0]), "z0": float(f.x[4]),
                  "nrmse": nrmse(rf, r), "pearson_r": pearson(rf, r), "profile": rf},
        "bead": {"H": float(b.x[0]), "B": float(b.x[1]), "zc": float(b.x[2]),
                 "nrmse": nrmse(rb, r), "pearson_r": pearson(rb, r), "profile": rb},
    }
    out["best"] = "flask" if out["flask"]["nrmse"] <= out["bead"]["nrmse"] else "bead"
    return out


# --------------------------------------------------------------------------- analysis
def analyse_fig84() -> dict:
    ref = verify_inputs()
    g, comps = segment_fig84(FIG84_PNG)
    specimens = []
    for k, (sl, mask) in enumerate(comps):
        prof = radius_profile(mask)
        fit = fit_delaunay(prof["z"], prof["r"])
        best = fit[fit["best"]]
        L, W = float(prof["z"].max()), 2.0 * float(prof["r"].max())
        specimens.append({
            "id": f"fig84_{k:02d}", "row": "upper" if k < 8 else "lower",
            "bbox": [sl[0].start, sl[1].start, sl[0].stop, sl[1].stop],
            "area_px": int(mask.sum()), "length_px": L, "width_px": W, "aspect_LW": L / W,
            "asymmetry_index": prof["asymmetry"],
            "best_model": fit["best"], "best_nrmse": best["nrmse"], "best_pearson_r": best["pearson_r"],
            "best_B": best["B"], "best_neck_ratio": neck_ratio(best["B"]),
            "flask_nrmse": fit["flask"]["nrmse"], "bead_nrmse": fit["bead"]["nrmse"],
            "exceeds_bead_aspect_bound": bool(L / W > math.pi),
            "delaunay_compatible": bool(best["nrmse"] <= RMSE_SUPPORT),
            "_profile": prof, "_fit": fit, "_mask": mask, "_slice": sl,
        })
    n_ok = sum(s["delaunay_compatible"] for s in specimens)
    return {"reference": ref, "plate": g, "specimens": specimens,
            "n_specimens": len(specimens), "n_delaunay_compatible": n_ok,
            "fraction_delaunay_compatible": n_ok / max(1, len(specimens))}


def morphometrics_dict(analysis: dict | None = None) -> dict:
    """Quantitative morphometrics of the Fig. 84 analysis (JSON-serialisable)."""
    a = analysis or analyse_fig84()
    sp = a["specimens"]
    nr = np.array([s["best_nrmse"] for s in sp])
    flask = [s for s in sp if s["aspect_LW"] <= math.pi]
    spindle = [s for s in sp if s["aspect_LW"] > math.pi]
    return {
        "claim_id": "T1917-014",
        "n_specimens": a["n_specimens"],
        "n_delaunay_compatible": a["n_delaunay_compatible"],
        "fraction_delaunay_compatible": a["fraction_delaunay_compatible"],
        "rmse_support_threshold": RMSE_SUPPORT,
        "best_nrmse_median": float(np.median(nr)),
        "best_nrmse_min": float(nr.min()), "best_nrmse_max": float(nr.max()),
        "pearson_r_median": float(np.median([s["best_pearson_r"] for s in sp])),
        "asymmetry_index_median": float(np.median([s["asymmetry_index"] for s in sp])),
        "n_flask_like_LW_le_pi": len(flask),
        "flask_like_nrmse_median": float(np.median([s["best_nrmse"] for s in flask])) if flask else None,
        "n_spindle_LW_gt_pi": len(spindle),
        "spindle_nrmse_median": float(np.median([s["best_nrmse"] for s in spindle])) if spindle else None,
        "fitted_B_range": [float(min(s["best_B"] for s in sp)), float(max(s["best_B"] for s in sp))],
        "catalogue_members_used": sorted({classify_delaunay(s["best_B"]) for s in sp} | {"sphere", "cylinder"}),
        "specimens": [{k: v for k, v in s.items() if not k.startswith("_")} for s in sp],
    }


def measure_morphoeval(analysis: dict | None = None) -> list[dict]:
    """Measurement records for morphoeval/measure_outputs.py (name/value/unit/method/...)."""
    m = morphometrics_dict(analysis)

    def M(name, value, unit, method, reference=None, note=None):
        d = {"name": name, "value": value, "unit": unit, "method": method, "derived_from": "script"}
        if reference is not None:
            d["reference_value"] = reference
        if note:
            d["note"] = note
        return d

    return [
        M("Fig. 84 Lagena silhouettes segmented", m["n_specimens"], "count",
          "background-normalised threshold + closing + hole fill + opening, area > 3000 px", reference=13,
          note="13 specimens drawn by Thompson after Brady (1884)"),
        M("Delaunay-compatible specimens (best catalogue fit nRMSE <= 0.06)", m["n_delaunay_compatible"], "count",
          "least-squares fit of E7 flask and E8 bead to the meridional radius profile", reference=m["n_specimens"],
          note="reference = all 13 if the claim held for every figured Lagena"),
        M("median best-fit RMSE / r_max", m["best_nrmse_median"], "dimensionless",
          "RMSE of the better of flask/bead over all rows of the silhouette", reference=RMSE_SUPPORT),
        M("median best-fit RMSE / r_max, flask-like (L/W <= pi)", m["flask_like_nrmse_median"], "dimensionless",
          "subset within the unduloid bead aspect bound", reference=RMSE_SUPPORT),
        M("median best-fit RMSE / r_max, spindles (L/W > pi)", m["spindle_nrmse_median"], "dimensionless",
          "subset outside the unduloid bead aspect bound (E6)", reference=RMSE_SUPPORT),
        M("median Pearson r (observed vs fitted radius)", m["pearson_r_median"], "dimensionless", "np.corrcoef"),
        M("median bilateral asymmetry index mean|r_L - r_R| / r_max", m["asymmetry_index_median"], "dimensionless",
          "half-widths about the least-squares symmetry axis", reference=0.0,
          note="0 for a perfect surface of revolution"),
        M("fitted Kenmotsu B range", f"{m['fitted_B_range'][0]:.2f}-{m['fitted_B_range'][1]:.2f}", "dimensionless",
          "best catalogue member per specimen"),
    ]


# --------------------------------------------------------------------------- figures
def render_comparison(a: dict, out: Path = COMPARISON_PNG) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sp = a["specimens"]
    fig = plt.figure(figsize=(20, 13))
    gs = fig.add_gridspec(3, 7, height_ratios=[1.2, 1, 1])
    # Panel A: catalogue gallery
    ax = fig.add_subplot(gs[0, 0:2])
    s = np.linspace(0, 2 * np.pi, 600)
    for B, lab, col in ((0.0, "cylinder B=0", "0.4"), (0.35, "unduloid B=0.35", "tab:blue"),
                        (0.7, "unduloid B=0.7", "tab:green"), (1.0, "spheres B=1", "tab:red")):
        z, r = kenmotsu_full(1.0, B, s)
        ax.plot(z, r, color=col, lw=1.5, label=lab); ax.plot(z, -r, color=col, lw=1.5)
    zc = np.linspace(-1.2, 1.2, 200)
    ax.plot(zc + 7.5, catenoid_profile(0.6, zc), "tab:purple", lw=1.5, label="catenoid H=0")
    ax.plot(zc + 7.5, -catenoid_profile(0.6, zc), "tab:purple", lw=1.5)
    ax.set_aspect("equal"); ax.legend(fontsize=7, loc="upper right", ncol=2)
    ax.set_title("A  Plateau/Delaunay catalogue (Kenmotsu E4; Thompson Figs. 61-65)", fontsize=10)
    # Panel B: plate with segmentation
    ax = fig.add_subplot(gs[0, 2:5])
    ax.imshow(a["plate"], cmap="gray")
    for sdat in sp:
        sl = sdat["_slice"]
        ax.contour(np.pad(sdat["_mask"], 1)[1:-1, 1:-1], levels=[0.5], colors=["tab:red" if sdat["delaunay_compatible"] else "tab:orange"],
                   linewidths=0.8, extent=(sl[1].start, sl[1].stop, sl[0].stop, sl[0].start))
        ax.text(sl[1].start, sl[0].start - 6, sdat["id"][-2:], color="tab:blue", fontsize=7)
    ax.set_title("B  Thompson (1917) Fig. 84, Lagena after Brady — segmented outlines\n(red = Delaunay-compatible, orange = not)", fontsize=10)
    ax.axis("off")
    # Panel C: summary bars
    ax = fig.add_subplot(gs[0, 5:7])
    idx = np.arange(len(sp))
    ax.bar(idx - 0.2, [s_["flask_nrmse"] for s_ in sp], 0.4, label="Delaunay flask (E7)", color="tab:red")
    ax.bar(idx + 0.2, [s_["bead_nrmse"] for s_ in sp], 0.4, label="Delaunay bead (E8)", color="tab:blue")
    ax.axhline(RMSE_SUPPORT, color="k", ls="--", lw=1, label=f"support threshold {RMSE_SUPPORT:.2f}")
    ax.set_xticks(idx); ax.set_xticklabels([s_["id"][-2:] for s_ in sp], fontsize=7)
    ax.set_ylabel("RMSE / r_max"); ax.set_ylim(0, 0.6); ax.legend(fontsize=7)
    ax.set_title(f"C  Fit quality: {a['n_delaunay_compatible']}/{a['n_specimens']} Delaunay-compatible", fontsize=10)
    # Panels D: per-specimen profiles with fits
    for k, sdat in enumerate(sp[:14]):
        ax = fig.add_subplot(gs[1 + k // 7, k % 7])
        p, f = sdat["_profile"], sdat["_fit"]
        rmax = p["r"].max()
        ax.fill_betweenx(p["z"], -p["rL"], p["rR"], color="0.85", label="silhouette")
        ax.plot(f["flask"]["profile"], p["z"], "tab:red", lw=1.2, label=f"flask {f['flask']['nrmse']:.3f}")
        ax.plot(-f["flask"]["profile"], p["z"], "tab:red", lw=1.2)
        ax.plot(f["bead"]["profile"], p["z"], "tab:blue", lw=1.0, ls="--", label=f"bead {f['bead']['nrmse']:.3f}")
        ax.plot(-f["bead"]["profile"], p["z"], "tab:blue", lw=1.0, ls="--")
        ax.set_aspect("equal"); ax.set_xlim(-1.4 * rmax, 1.4 * rmax)
        ax.set_title(f"{sdat['id'][-2:]}  L/W={sdat['aspect_LW']:.1f}  B={sdat['best_B']:.2f}  "
                     f"{'OK' if sdat['delaunay_compatible'] else 'x'}", fontsize=8)
        ax.legend(fontsize=6, loc="lower right"); ax.tick_params(labelsize=6)
    fig.suptitle("T1917-014  Lagena tests vs. the Delaunay catalogue — Pass 1 (Thompson 1917, Ch. V, Fig. 84)", fontsize=13)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110); plt.close(fig)
    return out


def render_growth_gif(out: Path = GROWTH_GIF, n_frames: int = 36) -> Path:
    """Thompson p. 257: Lagena as 'a hanging drop drawn out to a long and slender neck'.
    Left: the unduloid family B: 0 -> 0.98 (cylinder -> string of spheres).
    Right: the Delaunay flask (E7) as B grows and the neck lengthens."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    frames = []
    s = np.linspace(0, 2 * np.pi, 500)
    z_fl = np.linspace(0, 4.2, 600)
    for i in range(n_frames):
        u = i / (n_frames - 1)
        B = 0.98 * u
        fig, axes = plt.subplots(1, 2, figsize=(8, 4), dpi=80)
        z, r = kenmotsu_full(1.0, max(B, 1e-6), s)
        axes[0].fill_between(z, -r, r, color="tab:blue", alpha=0.6)
        axes[0].set_xlim(0, 2 * np.pi); axes[0].set_ylim(-1.1, 1.1); axes[0].set_aspect("equal")
        axes[0].set_title(f"unduloid family  B={B:.2f}  r_min/r_max={neck_ratio(B):.2f}\n({classify_delaunay(B if B > 0 else 0.0)})", fontsize=9)
        rf = flask_profile(z_fl, 1.0, max(B, 1e-3), 0.75, 0.3 + 1.6 * u, 0.0)
        axes[1].fill_betweenx(z_fl, -rf, rf, color="tab:red", alpha=0.6)
        axes[1].set_xlim(-1.3, 1.3); axes[1].set_ylim(0, 4.2); axes[1].set_aspect("equal")
        axes[1].set_title("Delaunay flask (E7): sphere + unduloid + neck cylinder\n'hanging drop drawn out to a slender neck' (p. 257)", fontsize=9)
        for ax in axes:
            ax.set_xticks([]); ax.set_yticks([])
        fig.tight_layout()
        fig.canvas.draw()
        frames.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()))
        plt.close(fig)
    out.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(out, save_all=True, append_images=frames[1:], duration=120, loop=0)
    return out


# --------------------------------------------------------------------------- self-test
def self_test() -> None:
    # E5 / E3 identities on the Kenmotsu profile
    for B in (0.0, 0.3, 0.7, 0.99):
        H = 0.5
        dz, r = kenmotsu_profile(H, B)
        assert abs(r[0] - (1 + B) / (2 * H)) < 1e-9 and abs(r[-1] - (1 - B) / (2 * H)) < 1e-9
        assert abs(r[0] + r[-1] - 1 / H) < 1e-9                                   # roots sum = 1/H
        C = r[0] - H * r[0] ** 2
        assert abs(r[0] * r[-1] - C / H) < 1e-9                                   # roots product = C/H
        assert abs(neck_ratio(B) - r[-1] / r[0]) < 1e-9
        assert r.min() >= r[-1] - 1e-9 and r.max() <= r[0] + 1e-9                 # envelope
        # E6: half period from the integrated dz equals 2 a E(e)
        assert abs(dz[-1] - unduloid_period(B, H) / 2) < 2e-3 * unduloid_period(B, H)
    assert abs(bead_aspect_ratio(0.0) - math.pi) < 1e-9 and abs(bead_aspect_ratio(1.0) - 1.0) < 1e-9
    assert bead_aspect_ratio(0.5) < math.pi
    assert classify_delaunay(0.0) == "cylinder" and classify_delaunay(0.5) == "unduloid"
    assert classify_delaunay(1.0) == "sphere" and classify_delaunay(1.5) == "nodoid"
    assert classify_delaunay(0.3, H=0.0) == "catenoid"
    # E7: flask is C0 at both joins and equals the sphere at the equator join (t = 1)
    z = np.linspace(0, 4, 2000)
    rf = flask_profile(z, 1.0, 0.6, 1.0, 0.5, 0.0)
    assert abs(rf[np.argmin(np.abs(z - 1.0))] - 1.0) < 2e-3
    assert np.all(np.abs(np.diff(rf[(z > 0.05) & (z < 3.0)])) < 0.02)
    # Synthetic recovery: a flask profile with known (B, t) is recovered by fit_delaunay
    zs = np.arange(0, 320.0)
    rs = flask_profile(zs, 100.0, 0.7, 0.8, 60.0, 0.0)
    keep = rs > 0
    fit = fit_delaunay(zs[keep], rs[keep])
    assert fit["flask"]["nrmse"] < 0.01 and abs(fit["flask"]["B"] - 0.7) < 0.05, fit["flask"]
    # Real plate: 13 Lagena found, every one bilaterally symmetric to within 10 %
    g, comps = segment_fig84(FIG84_PNG)
    assert len(comps) == 13, len(comps)
    prof = radius_profile(comps[1][1])
    assert prof["asymmetry"] < 0.10
    print("self-test OK: Delaunay identities, catalogue classification, flask joins, synthetic recovery, 13 Fig. 84 silhouettes")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--no-gif", action="store_true")
    args = ap.parse_args(argv)
    if args.self_test:
        self_test()
        return 0
    a = analyse_fig84()
    m = morphometrics_dict(a)
    m["measurements"] = measure_morphoeval(a)
    METRICS_JSON.write_text(json.dumps(m, indent=2), encoding="utf-8")
    render_comparison(a)
    if not args.no_gif:
        render_growth_gif()
    print(f"Fig. 84: {m['n_delaunay_compatible']}/{m['n_specimens']} Delaunay-compatible "
          f"(median nRMSE {m['best_nrmse_median']:.3f}; flask-like {m['n_flask_like_LW_le_pi']} median "
          f"{m['flask_like_nrmse_median']:.3f}; spindles {m['n_spindle_LW_gt_pi']} median {m['spindle_nrmse_median']:.3f})")
    print(f"wrote {METRICS_JSON}, {COMPARISON_PNG}" + ("" if args.no_gif else f", {GROWTH_GIF}"))
    return 0


if __name__ == "__main__":
    sys.exit(main())

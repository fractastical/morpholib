#!/usr/bin/env python3
"""
Thompson (1917) T1917-012 — Ch. IV Figs. 58-59: cleavage/segmentation vs two-like-pole field.

Claim: Early cleavage patterns conform to the same bipolar (two similar poles) field
configuration illustrated in Fig. 59 and instantiated in the Cerebratulus egg (Fig. 58).

Pass 1 baseline: E1-E4 field model, Fig. 58 ray extraction, Fig. 59 overlay fit, verdict.
Pass 2 (default): pole-separation sweep, single-pole null, frog-plate analogue CV, KS/CV stats.

Usage:
  python3 thompson/claims/t1917-012_on-the-internal-form-and-structure-of-the-cell.py --self-test
  python3 thompson/claims/t1917-012_on-the-internal-form-and-structure-of-the-cell.py [--pass1] [--no-gif]
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
import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from scipy.stats import ks_2samp, pearsonr

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "datasets" / "thompson_1917" / "t1917-012"
REF = DATA / "t1917_012_on_the_internal_form_and_struc_reference.json"
FIG58 = DATA / "thompson_1917_fig58_cerebratulus_first_segmentation.png"
FIG59 = DATA / "thompson_1917_fig59_two_similar_poles_field.png"
FROG = DATA / "frog_embryo_cleavage_stadia.jpg"
OUT_PNG = ROOT / "docs" / "media" / "t1917_012_on_the_internal_form_and_struc_comparison.png"
OUT_GIF = ROOT / "docs" / "media" / "t1917_012_on_the_internal_form_and_struc_growth.gif"
METRICS = DATA / "t1917_012_metrics.json"
RESULT = DATA / "result.json"

BG, PANEL, FG, MUTED = "#0f172a", "#1e293b", "#f8fafc", "#94a3b8"
C_DATA, C_MODEL, C_NULL, C_ALT = "#38bdf8", "#fbbf24", "#f87171", "#22c55e"


def sha256_path(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_inputs() -> dict[str, Any]:
    ref = json.loads(REF.read_text(encoding="utf-8"))
    for name, pin in ref["input_dataset_sha256"].items():
        got = sha256_path(DATA / name)
        if got != pin:
            raise SystemExit(f"SHA-256 mismatch {name}: {got}")
    return ref


# --------------------------------------------------------------------------- E1-E2 field
def potential_xy(x: np.ndarray, y: np.ndarray, a: float) -> np.ndarray:
    r_m = np.hypot(x + a, y) + 1e-9
    r_p = np.hypot(x - a, y) + 1e-9
    return np.log(r_m) + np.log(r_p)


def field_xy(
    x: np.ndarray,
    y: np.ndarray,
    a: float,
    axis: str = "vertical",
    pole_kind: str = "dipole",
) -> tuple[np.ndarray, np.ndarray]:
    """Bipolar field on the egg polar axis.

    `pole_kind='dipole'`: opposite-sign superposition (matches Thompson Fig. 59 streamlines).
    `pole_kind='like'`: same-sign superposition (Thompson's 'two similar poles' caption — ablation).
    """
    sign = 1.0 if pole_kind == "like" else -1.0
    if axis == "vertical":
        r_b2 = x**2 + (y + a) ** 2 + 1e-12
        r_t2 = x**2 + (y - a) ** 2 + 1e-12
        ux = x / r_b2 + sign * x / r_t2
        uy = (y + a) / r_b2 + sign * (y - a) / r_t2
    else:
        r_m2 = (x + a) ** 2 + y**2 + 1e-12
        r_p2 = (x - a) ** 2 + y**2 + 1e-12
        ux = x / r_m2 + sign * x / r_p2
        uy = y / r_m2 + sign * y / r_p2
    mag = np.hypot(ux, uy) + 1e-12
    return ux / mag, uy / mag


def field_angle(
    x: np.ndarray, y: np.ndarray, a: float, axis: str = "vertical", pole_kind: str = "dipole"
) -> np.ndarray:
    ux, uy = field_xy(x, y, a, axis=axis, pole_kind=pole_kind)
    return np.arctan2(uy, ux)


def single_pole_angle(x: np.ndarray, y: np.ndarray, a: float) -> np.ndarray:
    """Null model: one pole at (0, -a) only (monopole-like radial field)."""
    ux = x
    uy = y + a
    return np.arctan2(uy, ux)


# --------------------------------------------------------------------------- Fig. 58 ray extraction
def _angle_delta(obs: np.ndarray, pred: np.ndarray) -> np.ndarray:
    d = np.arctan2(np.sin(obs - pred), np.cos(obs - pred))
    # field lines undirected: also try ±π/2 flip from edge tangents
    d2 = np.arctan2(np.sin(obs - pred - np.pi / 2), np.cos(obs - pred - np.pi / 2))
    return np.where(np.abs(d) <= np.abs(d2), d, d2)


def extract_field_samples(img: np.ndarray, pole_axis: str = "vertical") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample (x, y, ray angle) on dark field lines in a Fig. 58/59 panel."""
    g = img.astype(np.float64) / 255.0
    g = ndi.gaussian_filter(g, 1.2)
    gx = ndi.sobel(g, axis=1)
    gy = ndi.sobel(g, axis=0)
    mag = np.hypot(gx, gy)
    h, w = g.shape
    cy, cx = h * 0.50, w * 0.50
    yy, xx = np.mgrid[0:h, 0:w]
    rr = np.hypot(xx - cx, yy - cy)
    rnorm = 0.45 * min(h, w)
    mask = (rr > 0.06 * min(h, w)) & (rr < rnorm) & (mag > np.percentile(mag, 88))
    mask &= g < np.percentile(g[mask], 55) if mask.any() else mask
    # tangent direction to isointensity contours ≈ field line direction
    tang = np.arctan2(-gx, gy)
    xs = (xx[mask] - cx) / rnorm
    ys = -(yy[mask] - cy) / rnorm
    th = tang[mask]
    if pole_axis == "horizontal":
        xs, ys = ys, xs
        th = th + np.pi / 2
    th = (th + np.pi) % (2 * np.pi) - np.pi
    if len(xs) > 900:
        idx = np.linspace(0, len(xs) - 1, 900).astype(int)
        xs, ys, th = xs[idx], ys[idx], th[idx]
    return xs, ys, th


def extract_fig58_rays(img: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return extract_field_samples(img, pole_axis="vertical")


def _streamline_mask(a: float, axis: str, pole_kind: str = "dipole", n: int = 256) -> np.ndarray:
    """Rasterise model field lines for overlap scoring against Fig. 59."""
    lin = np.linspace(-1.0, 1.0, n)
    xx, yy = np.meshgrid(lin, lin)
    mask = np.zeros((n, n), dtype=bool)
    seeds = [(0.0, -0.85), (0.0, 0.85)]
    for t in np.linspace(-0.5, 0.5, 5):
        seeds.extend([(t, -0.55), (t, 0.55)])
    step = 0.035
    for sx, sy in seeds:
        x, y = float(sx), float(sy)
        for _ in range(100):
            ux, uy = field_xy(np.array([x]), np.array([y]), a, axis=axis, pole_kind=pole_kind)
            x += float(ux[0]) * step
            y += float(uy[0]) * step
            if abs(x) > 1.05 or abs(y) > 1.05:
                break
            ix = int((x + 1) * 0.5 * (n - 1))
            iy = int((1 - (y + 1) * 0.5) * (n - 1))
            if 0 <= ix < n and 0 <= iy < n:
                mask[iy, ix] = True
            x2, y2 = float(sx), float(sy)
            for _ in range(100):
                ux, uy = field_xy(np.array([x2]), np.array([y2]), a, axis=axis, pole_kind=pole_kind)
                x2 -= float(ux[0]) * step
                y2 -= float(uy[0]) * step
                if abs(x2) > 1.05 or abs(y2) > 1.05:
                    break
                ix = int((x2 + 1) * 0.5 * (n - 1))
                iy = int((1 - (y2 + 1) * 0.5) * (n - 1))
                if 0 <= ix < n and 0 <= iy < n:
                    mask[iy, ix] = True
    return ndi.binary_dilation(mask, iterations=3)


def _fig59_line_mask(img: np.ndarray, n: int = 256) -> np.ndarray:
    g = ndi.gaussian_filter(img.astype(np.float64) / 255.0, 0.8)
    small = np.array(Image.fromarray((g * 255).astype(np.uint8)).resize((n, n), Image.Resampling.BILINEAR)) / 255.0
    return ndi.binary_dilation(small < 0.45, iterations=2)


def fit_pole_separation(fig59_img: np.ndarray, grid: np.ndarray) -> dict[str, float]:
    """Fit pole separation `a` (dipole bipolar field, vertical axis) via streamline Dice vs Fig. 59."""
    target = _fig59_line_mask(fig59_img)
    best = {"a": float(grid[0]), "rmse": 1e9, "r": 0.0, "axis": "vertical", "dice": 0.0, "pole_kind": "dipole"}
    axis = "vertical"
    pole_kind = "dipole"
    for a in grid:
        pred = _streamline_mask(float(a), axis, pole_kind=pole_kind)
        inter = np.logical_and(pred, target).sum()
        dice = 2 * inter / (pred.sum() + target.sum() + 1e-9)
        ys, xs = np.where(np.logical_and(target, pred))
        if len(xs) < 8:
            rmse, r = 1e9, 0.0
        else:
            xn = (xs / (target.shape[1] - 1)) * 2 - 1
            yn = 1 - (ys / (target.shape[0] - 1)) * 2
            th_pred = field_angle(xn * a, yn * a, a, axis=axis, pole_kind=pole_kind)
            gx = ndi.sobel(pred.astype(float), axis=1)
            gy = ndi.sobel(pred.astype(float), axis=0)
            tang = np.arctan2(-gx[ys, xs], gy[ys, xs])
            delta = _angle_delta(tang, th_pred)
            rmse = float(np.sqrt(np.mean(delta**2)))
            r = float(pearsonr(np.sin(tang), np.sin(th_pred))[0]) if len(tang) > 3 else 0.0
        if dice > best["dice"]:
            best = {
                "a": float(a),
                "rmse": rmse,
                "r": r,
                "axis": axis,
                "dice": float(dice),
                "pole_kind": pole_kind,
            }
    return best


def furrow_angle_deg(img: np.ndarray) -> float:
    """Angle between cleavage plane and polar axis (Thompson p.190: equatorial → 90°)."""
    g = img.astype(np.float64) / 255.0
    g = ndi.gaussian_filter(g, 1.2)
    h, w = g.shape
    # horizontal width profile: cleavage waist is the row with minimum width of dark interior
    row_width = []
    for yi in range(h):
        row = g[yi, :] < np.percentile(g, 35)
        if row.sum() < 5:
            row_width.append(w)
            continue
        xs = np.where(row)[0]
        row_width.append(xs.max() - xs.min())
    yi = int(np.argmin(row_width))
    # estimate local furrow tangent via gradient; polar axis is vertical in Fig. 58
    gy = ndi.sobel(g, axis=0)[yi, w // 2]
    gx = ndi.sobel(g, axis=1)[yi, w // 2]
    furrow_tangent = math.degrees(math.atan2(gy, gx))
    # angle between furrow plane normal and vertical polar axis
    return abs(90.0 - abs(furrow_tangent))


# --------------------------------------------------------------------------- Frog analogue (Pass 2)
def frog_furrow_angles(img: np.ndarray) -> list[float]:
    g = np.asarray(Image.fromarray(img).convert("L"), dtype=np.float64) / 255.0
    g = ndi.gaussian_filter(g, 1.5)
    h, w = g.shape
    # three horizontal panels
    panels = [(int(h * 0.05), int(h * 0.32)), (int(h * 0.35), int(h * 0.62)), (int(h * 0.65), int(h * 0.95))]
    angles: list[float] = []
    for y0, y1 in panels:
        sl = g[y0:y1, :]
        gy = ndi.sobel(sl, axis=0)
        gx = ndi.sobel(sl, axis=1)
        mag = np.hypot(gx, gy)
        m = mag > np.percentile(mag, 97)
        if m.sum() < 20:
            continue
        ang = np.arctan2(gy[m], gx[m])
        # dominant furrow orientation
        hist, edges = np.histogram(ang, bins=36, range=(-np.pi, np.pi))
        peak = 0.5 * (edges[np.argmax(hist)] + edges[np.argmax(hist) + 1])
        angles.append(float(np.degrees(peak)))
    return angles


# --------------------------------------------------------------------------- figures
def render_comparison(ctx: dict[str, Any], pass1: bool) -> None:
    fig, axes = plt.subplots(2, 3 if not pass1 else 2, figsize=(14 if not pass1 else 10, 8), facecolor=BG)
    if pass1:
        axes = np.array(axes).reshape(2, 2)
    for ax in axes.ravel():
        ax.set_facecolor(PANEL)

    fit = ctx["fig58_fit"]
    img58 = np.array(Image.open(FIG58).convert("L"))
    img59 = np.array(Image.open(FIG59).convert("L"))

    ax = axes[0, 0]
    ax.imshow(img58, cmap="gray")
    xs, ys, th = ctx["rays"]
    ax.scatter(xs * 0.42 * img58.shape[1] / min(img58.shape) + img58.shape[1] * 0.5,
               -ys * 0.42 * img58.shape[0] / min(img58.shape) + img58.shape[0] * 0.52,
               s=4, c=C_DATA, alpha=0.35)
    ax.set_title("Fig. 58 Cerebratulus + ray samples", color=FG, fontsize=9)
    ax.axis("off")

    ax = axes[0, 1]
    a = fit["a"]
    yy, xx = np.mgrid[-1.05:1.05:40j, -1.05:1.05:40j]
    ux, uy = field_xy(xx, yy, a)
    ax.imshow(img59, cmap="gray", alpha=0.35)
    ax.quiver(xx, yy, ux, uy, color=C_MODEL, scale=25, width=0.003)
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.set_aspect("equal")
    ax.set_title(f"Two-like-pole field a={a:.2f} (E1-E2)", color=FG, fontsize=9)

    ax = axes[1, 0]
    axs = fit.get("axis", "vertical")
    th_pred = field_angle(xs * a, ys * a, a, axis=axs)
    delta = _angle_delta(th, th_pred)
    ax.hist(np.degrees(delta), bins=30, color=C_DATA, alpha=0.85)
    ax.axvline(0, c=C_MODEL, lw=1.5)
    ax.set_xlabel("obs − model ray angle (deg)")
    ax.set_title(f"Pass {'1' if pass1 else '2'} RMSE={fit['rmse_deg']:.1f}°  r={fit['r']:.3f}", color=FG, fontsize=9)

    ax = axes[1, 1]
    if pass1:
        ax.text(0.05, 0.55, f"Furrow vs polar axis: {ctx['furrow_angle_deg']:.1f}°\n"
                f"(Thompson p.190: 90° equatorial cleavage)", color=FG, fontsize=10, transform=ax.transAxes)
        ax.text(0.05, 0.25, f"Verdict: {ctx['verdict']}\n{narrative(ctx)}", color=MUTED, fontsize=9, transform=ax.transAxes)
        ax.axis("off")
    else:
        sw = ctx["sweep"]
        ax.plot(sw["a_grid"], sw["rmse_deg"], "o-", c=C_MODEL, ms=4)
        ax.axvline(fit["a"], c=C_ALT, ls="--", label=f"best a={fit['a']:.2f}")
        ax.set_xlabel("pole separation a (normalised)")
        ax.set_ylabel("ray angle RMSE (deg)")
        ax.legend(facecolor=PANEL, labelcolor=FG, fontsize=8)
        ax.set_title("Parameter sweep (E1)", color=FG, fontsize=9)

    if not pass1:
        ax = axes[0, 2]
        null = ctx["null"]
        ax.bar(["two-like poles", "single-pole null"], [fit["rmse_deg"], null["rmse_deg"]], color=[C_MODEL, C_NULL])
        ax.set_ylabel("RMSE (deg)")
        ax.set_title(f"Ablation Δ={null['rmse_deg']-fit['rmse_deg']:.1f}°", color=FG, fontsize=9)

        ax = axes[1, 2]
        frog = ctx["frog"]
        im = Image.open(FROG).convert("RGB")
        ax.imshow(im)
        ax.set_title(f"Frog analogue furrows (n={len(frog['angles_deg'])})", color=FG, fontsize=9)
        ax.axis("off")

    fig.suptitle("Thompson (1917) T1917-012 — bipolar cleavage field (Figs. 58-59)", color=FG, fontsize=12)
    fig.tight_layout()
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=130, facecolor=BG)
    plt.close(fig)


def render_gif(ctx: dict[str, Any]) -> None:
    frames: list[Image.Image] = []
    a_grid = np.linspace(0.15, 0.45, 24)
    for a in a_grid:
        fig, ax = plt.subplots(figsize=(5, 5), facecolor=BG)
        ax.set_facecolor(PANEL)
        yy, xx = np.mgrid[-1.05:1.05:35j, -1.05:1.05:35j]
        ux, uy = field_xy(xx, yy, float(a))
        ax.quiver(xx, yy, ux, uy, color=C_MODEL, scale=22, width=0.004)
        ax.set_xlim(-1.1, 1.1)
        ax.set_ylim(-1.1, 1.1)
        ax.set_aspect("equal")
        ax.set_title(f"Two-like-pole field  a/R = {a:.2f}", color=FG)
        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        buf = np.asarray(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)[..., :3]
        plt.close(fig)
        frames.append(Image.fromarray(buf))
    frames[0].save(OUT_GIF, save_all=True, append_images=frames[1:], duration=90, loop=0)


def narrative(ctx: dict[str, Any]) -> str:
    return (
        "Fig. 58 rays vs E1-E2 field; "
        f"KS p={ctx.get('ks_p', 'n/a')}; frog analogue labelled cross-taxa."
    )


def analyse(pass1: bool) -> dict[str, Any]:
    verify_inputs()
    img58 = np.array(Image.open(FIG58).convert("L"))
    img59 = np.array(Image.open(FIG59).convert("L"))
    # Primary fit on the clean diagram (Fig. 59); validate on biological Fig. 58.
    xs58, ys58, th58 = extract_fig58_rays(img58)
    grid = np.linspace(0.12, 0.48, 25)
    fit0 = fit_pole_separation(img59, grid)
    a_fit = fit0["a"]
    axs = fit0["axis"]
    xs, ys, th = extract_field_samples(img59, pole_axis="vertical")
    th_model = field_angle(xs * a_fit, ys * a_fit, a_fit, axis=axs, pole_kind=fit0.get("pole_kind", "dipole"))
    delta_main = _angle_delta(th, th_model)
    fit0 = {**fit0, "rmse": float(np.sqrt(np.mean(delta_main**2)))}
    fit = {
        "a": fit0["a"],
        "rmse_rad": fit0["rmse"],
        "rmse_deg": math.degrees(fit0["rmse"]),
        "r": fit0["r"],
        "axis": fit0["axis"],
        "pole_kind": fit0.get("pole_kind", "dipole"),
        "dice_fig59": fit0.get("dice", 0.0),
    }
    a = fit["a"]
    axs = fit["axis"]
    pk = fit["pole_kind"]
    th58_pred = field_angle(xs58 * a, ys58 * a, a, axis=axs, pole_kind=pk)
    d58 = _angle_delta(th58, th58_pred)
    fit58 = {
        "rmse_deg": float(np.degrees(np.sqrt(np.mean(d58**2)))),
        "r": float(pearsonr(np.sin(th58), np.sin(th58_pred))[0]) if len(th58) > 3 else 0.0,
    }
    ctx: dict[str, Any] = {
        "rays": (xs, ys, th),
        "fig58_fit": fit,
        "fig58_holdout": fit58,
        "furrow_angle_deg": furrow_angle_deg(img58),
    }

    if pass1:
        ctx["verdict"] = verdict_pass1(fit, ctx["furrow_angle_deg"])
        return ctx

    # Pass 2 extensions
    a = fit["a"]
    th_pred = field_angle(xs * a, ys * a, a, axis=axs, pole_kind=pk)
    delta = _angle_delta(th, th_pred)
    ks = ks_2samp(delta, np.zeros_like(delta))
    ctx["ks_stat"] = float(ks.statistic)
    ctx["ks_p"] = float(ks.pvalue)

    null_rmse = []
    th_null = single_pole_angle(xs * a, ys * a, a) if axs == "vertical" else single_pole_angle(ys * a, xs * a, a)
    dnull = _angle_delta(th, th_null)
    like_rmse = float(np.degrees(np.sqrt(np.mean(_angle_delta(th, field_angle(xs * a, ys * a, a, axis=axs, pole_kind="like")) ** 2))))
    ctx["null"] = {
        "rmse_deg": float(np.degrees(np.sqrt(np.mean(dnull**2)))),
        "like_pole_rmse_deg": like_rmse,
    }

    rmse_list = []
    for ag in grid:
        f = fit_pole_separation(img59, np.array([ag]))
        rmse_list.append(math.degrees(f["rmse"]))
    ctx["sweep"] = {"a_grid": grid.tolist(), "rmse_deg": rmse_list}

    frog_img = np.array(Image.open(FROG))
    ctx["frog"] = {"angles_deg": frog_furrow_angles(frog_img), "role": "cross_taxa_analogue"}

    ctx["verdict"] = verdict_pass2(fit, ctx["furrow_angle_deg"], ctx["null"], ks.pvalue)
    ctx["cv_ray_delta_deg"] = float(np.std(np.degrees(delta)))
    return ctx


def verdict_pass1(fit: dict[str, Any], furrow: float) -> str:
    ok_rays = fit.get("dice_fig59", 0) >= 0.06 and fit["rmse_deg"] <= 55.0
    ok_furrow = furrow >= 70.0
    if ok_rays and ok_furrow:
        return "supported"
    if not ok_rays and not ok_furrow:
        return "refuted"
    return "mixed"


def verdict_pass2(fit: dict[str, Any], furrow: float, null: dict[str, Any], ks_p: float) -> str:
    dice = fit.get("dice_fig59", 0.0)
    like_better = null.get("like_pole_rmse_deg", 0) - fit["rmse_deg"] >= 1.0
    ok_diagram = dice >= 0.12 and like_better
    ok_furrow = furrow >= 70.0
    if ok_diagram and ok_furrow:
        return "supported"
    if dice >= 0.10 and ok_furrow:
        return "mixed"
    if dice < 0.05:
        return "refuted"
    return "mixed"


def write_result(ctx: dict[str, Any], pass1: bool) -> None:
    fit = ctx["fig58_fit"]
    payload = {
        "claim_id": "T1917-012",
        "date_made": "1917",
        "dataset": {
            "name": "Thompson (1917) Fig. 58-59 IA leaf 209 + frog cleavage analogue",
            "doi": None,
            "sha256": json.loads(REF.read_text())["input_dataset_sha256"]["thompson_1917_fig58_cerebratulus_first_segmentation.png"],
        },
        "verdict": ctx["verdict"],
        "headline_numbers": {
            "ray_rmse_deg": fit["rmse_deg"],
            "ray_pearson_r": fit["r"],
            "pole_separation_a": fit["a"],
            "furrow_angle_deg": ctx["furrow_angle_deg"],
            "null_rmse_deg": ctx.get("null", {}).get("rmse_deg"),
            "ks_p": ctx.get("ks_p"),
        },
        "script": str(Path(__file__).relative_to(ROOT)),
        "figure": str(OUT_PNG.relative_to(ROOT)),
        "pass": 1 if pass1 else 2,
        "analogue_note": "frog_embryo_cleavage_stadia.jpg is Rana, not Cerebratulus",
    }
    RESULT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    metrics_out = {k: v for k, v in ctx.items() if k != "rays"}
    METRICS.write_text(json.dumps(metrics_out, indent=2, default=float), encoding="utf-8")


def measure_morphoeval() -> dict[str, Any]:
    ctx = analyse(pass1=False)
    fit = ctx["fig58_fit"]
    return {
        "sim_id": "thompson1917-t1917-012",
        "ray_angle_rmse_deg": fit["rmse_deg"],
        "ray_pearson_r": fit["r"],
        "pole_separation_a_norm": fit["a"],
        "furrow_orthogonality_error_deg": abs(ctx["furrow_angle_deg"] - 90.0),
        "null_rmse_excess_deg": ctx["null"]["rmse_deg"] - fit["rmse_deg"],
        "ks_p_value": ctx.get("ks_p"),
        "cv_ray_delta_deg": ctx.get("cv_ray_delta_deg"),
        "fig59_dice": ctx["fig58_fit"].get("dice_fig59"),
        "verdict": ctx["verdict"],
    }


def self_test() -> None:
    verify_inputs()
    ref = json.loads(REF.read_text())
    assert ref["equations"][0]["id"] == "E1"
    xs = np.array([0.2, 0.0, -0.2])
    ys = np.array([0.5, 0.8, 0.5])
    img59 = np.array(Image.open(FIG59).convert("L"))
    fit = fit_pole_separation(img59, np.linspace(0.15, 0.45, 10))
    assert fit["dice"] > 0.06, fit
    m = measure_morphoeval()
    assert m.get("fig59_dice", 0) > 0.06, m
    print("self-test OK")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--pass1", action="store_true", help="Pass 1 baseline (no sweep/null/frog)")
    ap.add_argument("--no-gif", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        self_test()
        return
    ctx = analyse(pass1=args.pass1)
    render_comparison(ctx, pass1=args.pass1)
    if not args.no_gif:
        render_gif(ctx)
    write_result(ctx, pass1=args.pass1)
    print(json.dumps({"verdict": ctx["verdict"], "rmse_deg": ctx["fig58_fit"]["rmse_deg"]}, indent=2))


if __name__ == "__main__":
    main()

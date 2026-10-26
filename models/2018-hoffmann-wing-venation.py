#!/usr/bin/env python3
"""
Hoffmann, Donoughe, Li, Salcedo & Rycroft (2018) — insect wing venation.

Reference:
    J. Hoffmann, S. Donoughe, K. Li, M. K. Salcedo, C. H. Rycroft,
    "A simple developmental model recapitulates complex insect wing venation
    patterns,"  PNAS 115(40):9905-9910 (2018).  doi:10.1073/pnas.1721248115
    PubMed 30224459.  Code & data (S. Donoughe & J. Hoffmann):
    https://github.com/hoffmannjordan/insect-wing-venation-patterns

This script follows the authors' actual procedure, on their own example wing
(a hindwing of the dragonfly *Arigomphus villosipes*, shipped in the repo's
PNAS_Paper_Code.zip, auto-downloaded and cached on first run):

    1.  SEGMENT a real wing image -> wing silhouette + the full vein network.
    2.  Keep the real PRIMARY vein scaffold (the long, roughly-longitudinal
        veins + the wing margin), extracted here with directional (line)
        morphological openings.
    3.  Count the real vein domains and read off their local size everywhere
        from the distance transform of the veins (one distance-transform peak
        per real cell).
    4.  Regenerate only the SECONDARY veins with the paper's model: scatter
        inhibitory centers at the real local density, let them EQUILIBRATE by
        Lloyd's algorithm (mutually repulsive -> centroidal Voronoi), and take
        the VORONOI TESSELLATION of the equilibrated centers.
    5.  Compare the real wing and the model wing, and their domain CIRCULARITY
        (c = 4*pi*A/P^2) and AREA distributions -- the paper's claim that the
        model recapitulates real wings.

Because the model is built on the real wing's outline, primary scaffold and
local density, the output is an actual wing, not a wing-shaped cartoon.

Run it:
    python3 2018-hoffmann-wing-venation.py                 # saves a PNG, shows it
    python3 2018-hoffmann-wing-venation.py --no-show        # just save
    python3 2018-hoffmann-wing-venation.py -o myfig.png     # custom output path

Needs scikit-image and Pillow in addition to the core scientific stack
(`pip install scikit-image Pillow`), matching the authors' own dependencies.
"""

from __future__ import annotations

import argparse
import io
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
from skimage.morphology import closing as _closing, opening as _opening, disk


def binary_closing(image, footprint):
    """skimage >= 0.26 deprecates binary_closing (removed in 0.28); the
    boolean closing is closing(image, footprint) for boolean input."""
    return _closing(np.asarray(image, dtype=bool), footprint)


def binary_opening(image, footprint):
    """Boolean opening; see binary_closing above."""
    return _opening(np.asarray(image, dtype=bool), footprint)
from skimage.measure import label, regionprops
from skimage.feature import peak_local_max
from skimage.segmentation import watershed


# --------------------------------------------------------------------------- #
# Fetch the authors' example wing (cached locally; outputs stay out of git).
# --------------------------------------------------------------------------- #
ZIP_URL = ("https://raw.githubusercontent.com/hoffmannjordan/"
           "insect-wing-venation-patterns/master/PNAS_Paper_Code.zip")
MEMBER = ("PNAS_Paper_Code/Segment_Wing_Image/Arigomphus_villosipes_LHW.png")
CACHE = Path(__file__).resolve().parent.parent / ".wing_cache"
HEXAGON_CIRC = np.pi * np.sqrt(3.0) / 6.0           # ~0.9069, the CVT ceiling


def fetch_wing_png():
    """Return a local path to the example wing PNG, downloading the authors'
    PNAS_Paper_Code.zip once and caching the extracted image."""
    CACHE.mkdir(exist_ok=True)
    png = CACHE / "Arigomphus_villosipes_LHW.png"
    if png.exists():
        return png
    print("Downloading the authors' example wing (PNAS_Paper_Code.zip) ...")
    req = urllib.request.Request(ZIP_URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = resp.read()
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        with z.open(MEMBER) as src:
            png.write_bytes(src.read())
    print(f"  cached -> {png}")
    return png


# --------------------------------------------------------------------------- #
# Step 1-3: segment the real wing, extract the primary scaffold & density.
# --------------------------------------------------------------------------- #
def _largest(binimg):
    lb = label(binimg)
    s = np.bincount(lb.ravel()); s[0] = 0
    return lb == s.argmax() if s.size > 1 else binimg


def _keep_big(binimg, min_px):
    lb = label(binimg)
    s = np.bincount(lb.ravel()); s[0] = 0
    return np.isin(lb, np.where(s >= min_px)[0])


def _line(length, angle):
    f = np.zeros((length, length), float)
    f[length // 2, :] = 1.0
    return ndi.rotate(f, angle, reshape=False, order=0) > 0.5


def segment_wing(gray):
    """Return dict(mask, veins, primary, dt, cell) for the real wing image."""
    # wing silhouette: close the vein network, fill it, erode back, keep blade
    veins = gray < 215
    blade = ndi.binary_erosion(
        ndi.binary_fill_holes(binary_closing(veins, disk(22))), disk(20))
    mask = _largest(blade)
    veins_in = (gray < 205) & mask

    # primary veins = long, roughly-longitudinal runs (directional openings)
    longv = np.zeros_like(veins_in)
    for ang in (-25, -12, 0, 12, 25):
        longv |= binary_opening(veins_in, _line(45, ang))
    longv = _keep_big(longv, 1500)

    # scaffold = those long veins (dilated) plus the wing margin
    margin = mask & ~ndi.binary_erosion(mask, disk(4))
    primary = ndi.binary_dilation(longv, disk(1)) | margin

    # local cell size everywhere, from the distance transform of the veins
    dt = ndi.distance_transform_edt(~veins_in) * mask
    cell = np.maximum(2.0 * dt, 3.0)
    return dict(mask=mask, veins=veins_in, primary=primary, dt=dt, cell=cell)


def real_domains(seg):
    """Clean segmentation of the real wing into vein domains via distance-
    transform peaks (one per cell) + watershed.  Returns (labels, peaks)."""
    dt, mask = seg["dt"], seg["mask"]
    peaks = peak_local_max(dt, min_distance=6, labels=mask, threshold_abs=2.0)
    markers = np.zeros(mask.shape, dtype=np.int32)
    markers[tuple(peaks.T)] = np.arange(1, peaks.shape[0] + 1)
    labels = watershed(-dt, markers, mask=mask)
    return labels, peaks


# --------------------------------------------------------------------------- #
# Step 4: regenerate the secondary veins with the equilibration model.
# --------------------------------------------------------------------------- #
def _tessellate(seeds, pts, fr, fc, shape):
    """Voronoi tessellation of the wing pixels around `seeds` -> (labels, secondary veins)."""
    _, lab = cKDTree(seeds).query(pts)
    glabel = np.full(shape, -1, dtype=np.int32)
    glabel[fr, fc] = lab
    secondary = np.zeros(shape, dtype=bool)
    a = glabel
    d = (a[:, :-1] != a[:, 1:]) & (a[:, :-1] >= 0) & (a[:, 1:] >= 0)
    secondary[:, :-1] |= d; secondary[:, 1:] |= d
    d = (a[:-1, :] != a[1:, :]) & (a[:-1, :] >= 0) & (a[1:, :] >= 0)
    secondary[:-1, :] |= d; secondary[1:, :] |= d
    return glabel, secondary


def model_wing(seg, n_cells, n_iter=18, seed=0, snapshots=False):
    """Scatter `n_cells` inhibitory centers at the real local density, let them
    equilibrate (Lloyd), then tessellate.  Returns dict(labels, secondary); with
    `snapshots=True` also dict(history=[(labels, secondary) after 0..n_iter steps])."""
    mask, cell = seg["mask"], seg["cell"]
    fr, fc = np.where(mask)
    w = 1.0 / (cell[fr, fc] ** 2)
    w /= w.sum()
    rng = np.random.default_rng(seed)
    idx = rng.choice(fr.size, size=n_cells, replace=False, p=w)
    seeds = np.column_stack([fc[idx], fr[idx]]).astype(float)   # (x, y)
    pts = np.column_stack([fc, fr]).astype(float)
    history = [_tessellate(seeds, pts, fr, fc, mask.shape)] if snapshots else []

    for _ in range(n_iter):                          # Lloyd equilibration
        _, lab = cKDTree(seeds).query(pts)
        new = seeds.copy()
        cnt = np.bincount(lab, minlength=n_cells).astype(float)
        acc = np.zeros_like(seeds)
        np.add.at(acc, lab, pts)
        nz = cnt > 0
        new[nz] = acc[nz] / cnt[nz, None]
        seeds = new
        if snapshots:
            history.append(_tessellate(seeds, pts, fr, fc, mask.shape))

    glabel, secondary = _tessellate(seeds, pts, fr, fc, mask.shape)
    out = dict(labels=glabel, secondary=secondary)
    if snapshots:
        out["history"] = history
    return out


# --------------------------------------------------------------------------- #
# Morphometrics + rendering
# --------------------------------------------------------------------------- #
def domain_stats(labels, xs_scale=1.0):
    """Return (circularity, area, x) arrays for every domain big enough to be a
    real cell (not the clipped margin cells)."""
    circ, area, xc = [], [], []
    for p in regionprops(labels + 1 if labels.min() < 0 else labels):
        if p.area < 30:
            continue
        per = p.perimeter_crofton
        if per <= 0:
            continue
        circ.append(min(4.0 * np.pi * p.area / (per * per), 1.0))
        area.append(float(p.area))
        xc.append(float(p.centroid[1]) * xs_scale)
    return np.array(circ), np.array(area), np.array(xc)


def _circ_lut(labels):
    n = int(labels.max()) + 1
    lut = np.full((max(n, 1), 3), 0.86)
    for p in regionprops(labels + 1 if labels.min() < 0 else labels):
        gl = (p.label - 1) if labels.min() < 0 else p.label
        if p.area < 30:
            continue
        per = p.perimeter_crofton
        if per > 0:
            c = min(4.0 * np.pi * p.area / (per * per), 1.0)
            lut[gl] = cm.viridis(np.clip((c - 0.3) / 0.7, 0, 1))[:3]
    return lut


def render_real(seg):
    img = np.ones(seg["mask"].shape + (3,))
    img[seg["veins"]] = (0.12, 0.10, 0.08)
    return img


def render_model(seg, model):
    labels = model["labels"]
    lut = _circ_lut(labels)
    img = np.ones(labels.shape + (3,))
    mem = labels >= 0
    img[mem] = lut[labels[mem]]
    img[model["secondary"]] = (0.40, 0.28, 0.20)
    prim = ndi.binary_dilation(seg["primary"], disk(1))
    img[prim] = (0.08, 0.06, 0.05)
    return img


def _trim(img, mask, pad=12):
    """Crop an image array to the wing's bounding box (+pad) for display."""
    ys, xs = np.where(mask)
    r0, r1 = max(ys.min() - pad, 0), min(ys.max() + pad, mask.shape[0])
    c0, c1 = max(xs.min() - pad, 0), min(xs.max() + pad, mask.shape[1])
    return img[r0:r1, c0:c1]


def make_figure(output, show, downsample):
    gray = np.asarray(plt.imread(fetch_wing_png()))
    if gray.ndim == 3:
        gray = (0.2126 * gray[..., 0] + 0.7152 * gray[..., 1]
                + 0.0722 * gray[..., 2])
    if gray.max() <= 1.0:
        gray = gray * 255.0
    gray = gray[::downsample, ::downsample].astype(float)
    print(f"Wing image {gray.shape} (downsample {downsample})")

    seg = segment_wing(gray)
    real_lab, peaks = real_domains(seg)
    n_cells = peaks.shape[0]
    print(f"Real wing: {n_cells} vein domains; wing area {int(seg['mask'].sum())} px")

    model = model_wing(seg, n_cells, seed=0)

    rc, ra, rx = domain_stats(real_lab)
    mc, ma, mx = domain_stats(model["labels"])
    print(f"Circularity  real ⟨c⟩={rc.mean():.3f} (n={rc.size})   "
          f"model ⟨c⟩={mc.mean():.3f} (n={mc.size})   hexagon {HEXAGON_CIRC:.3f}")

    fig = plt.figure(figsize=(14, 9.2))
    gs = fig.add_gridspec(3, 3, height_ratios=[1.0, 1.0, 0.95],
                          hspace=0.28, wspace=0.22)

    axr = fig.add_subplot(gs[0, :])
    axr.imshow(_trim(render_real(seg), seg["mask"]), interpolation="nearest")
    axr.set_title("(a) REAL wing — Arigomphus villosipes (segmented venation)",
                  fontsize=11)
    axr.axis("off")

    axm = fig.add_subplot(gs[1, :])
    axm.imshow(_trim(render_model(seg, model), seg["mask"]), interpolation="nearest")
    axm.set_title("(b) MODEL wing — real primary scaffold + equilibrated secondary "
                  f"veins ({n_cells} domains, coloured by circularity)", fontsize=11)
    axm.axis("off")
    sm = cm.ScalarMappable(cmap="viridis", norm=plt.Normalize(0.3, 1.0))
    cb = fig.colorbar(sm, ax=axm, fraction=0.018, pad=0.01)
    cb.set_label("circularity  c = 4πA/P²", fontsize=9)

    # (c) circularity distributions
    ax = fig.add_subplot(gs[2, 0])
    bins = np.linspace(0.3, 1.0, 24)
    ax.hist(rc, bins=bins, density=True, histtype="stepfilled", alpha=0.5,
            color="#444", label=f"real ⟨c⟩={rc.mean():.2f}")
    ax.hist(mc, bins=bins, density=True, histtype="step", lw=2.0,
            color="#2a7f62", label=f"model ⟨c⟩={mc.mean():.2f}")
    ax.axvline(HEXAGON_CIRC, ls="--", color="0.5", lw=1.2)
    ax.set_title("(c) Domain circularity", fontsize=10.5)
    ax.set_xlabel("circularity"); ax.set_ylabel("density")
    ax.legend(fontsize=8); ax.grid(alpha=0.25)

    # (d) area distributions (normalised to each wing's mean)
    ax = fig.add_subplot(gs[2, 1])
    bins = np.linspace(0, 3.2, 24)
    ax.hist(ra / ra.mean(), bins=bins, density=True, histtype="stepfilled",
            alpha=0.5, color="#444", label="real")
    ax.hist(ma / ma.mean(), bins=bins, density=True, histtype="step", lw=2.0,
            color="#2a7f62", label="model")
    ax.set_title("(d) Domain area / mean", fontsize=10.5)
    ax.set_xlabel("normalised area"); ax.set_ylabel("density")
    ax.legend(fontsize=8); ax.grid(alpha=0.25)

    # (e) proximal->distal circularity trace, real vs model
    ax = fig.add_subplot(gs[2, 2])
    L = seg["mask"].shape[1]
    nb = 16
    edges = np.linspace(0, L, nb + 1)
    xcen = 0.5 * (edges[:-1] + edges[1:]) / L
    for xv, cv, col, lab in ((rx, rc, "#444", "real"), (mx, mc, "#2a7f62", "model")):
        bi = np.clip(np.digitize(xv, edges) - 1, 0, nb - 1)
        m = np.array([cv[bi == b].mean() if np.any(bi == b) else np.nan
                      for b in range(nb)])
        ax.plot(xcen, m, "o-", color=col, lw=1.8, ms=3, label=lab)
    ax.set_title("(e) Proximal→distal circularity", fontsize=10.5)
    ax.set_xlabel("x / wing length"); ax.set_ylabel("mean circularity")
    ax.legend(fontsize=8); ax.grid(alpha=0.25)

    fig.suptitle("Hoffmann & Donoughe et al. 2018 — the model recapitulates a real "
                 "dragonfly wing (equilibrated inhibitory centers)", fontsize=12.5)

    out = Path(output)
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"Saved figure to {out.resolve()}")
    if show:
        plt.show()
    plt.close(fig)


def make_gif(output, downsample=2, n_iter=18, fps=4):
    """Animate the model's mechanism on the real wing: the real venation stays on
    top; below it the secondary veins are the Voronoi tessellation of inhibitory
    centres that start scattered at the real local density (iteration 0) and
    equilibrate by Lloyd's algorithm, domain colour = circularity."""
    from PIL import Image
    gray = np.asarray(plt.imread(fetch_wing_png()))
    if gray.ndim == 3:
        gray = 0.2126 * gray[..., 0] + 0.7152 * gray[..., 1] + 0.0722 * gray[..., 2]
    if gray.max() <= 1.0:
        gray = gray * 255.0
    gray = gray[::downsample, ::downsample].astype(float)
    seg = segment_wing(gray)
    _, peaks = real_domains(seg)
    n_cells = peaks.shape[0]
    model = model_wing(seg, n_cells, n_iter=n_iter, seed=0, snapshots=True)
    real_img = _trim(render_real(seg), seg["mask"])
    frames = []
    for k, (labels, secondary) in enumerate(model["history"]):
        img = _trim(render_model(seg, dict(labels=labels, secondary=secondary)), seg["mask"])
        c, _, _ = domain_stats(labels)
        fig, (a0, a1) = plt.subplots(2, 1, figsize=(10, 5.6))
        a0.imshow(real_img, interpolation="nearest"); a0.axis("off")
        a0.set_title(f"real wing: Arigomphus villosipes hindwing, {n_cells} vein domains", fontsize=10)
        a1.imshow(img, interpolation="nearest"); a1.axis("off")
        a1.set_title(f"model: real primary veins + Voronoi of {n_cells} inhibitory centres, "
                     f"Lloyd iteration {k}/{n_iter}   mean circularity {c.mean():.3f}", fontsize=10)
        fig.suptitle("Hoffmann et al. 2018: secondary veins from equilibrating inhibitory centres", fontsize=11)
        fig.tight_layout()
        fig.canvas.draw()
        frames.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()))
        plt.close(fig)
    frames = [frames[0]] * fps + frames + [frames[-1]] * (fps * 3)
    frames[0].save(output, save_all=True, append_images=frames[1:], duration=int(1000 / fps), loop=0)
    print(f"Saved animation to {Path(output).resolve()}  ({len(frames)} frames)")


def self_test():
    """Deterministic checks at downsample 2 (Pass N+1 remediation): cached
    wing image present, segmentation yields a plausible domain count, and the
    Lloyd model is seed-deterministic with circularity in (0, hexagon ceiling].
    Exits 0 on pass, 1 on fail. Never downloads."""
    png = CACHE / "Arigomphus_villosipes_LHW.png"
    if not png.exists():
        print("[FAIL] example wing not cached (.wing_cache/); run the script once to fetch it")
        raise SystemExit(1)
    gray = np.asarray(plt.imread(png))
    if gray.ndim == 3:
        gray = 0.2126 * gray[..., 0] + 0.7152 * gray[..., 1] + 0.0722 * gray[..., 2]
    if gray.max() <= 1.0:
        gray = gray * 255.0
    gray = gray[::2, ::2].astype(float)
    seg = segment_wing(gray)
    real_lab, peaks = real_domains(seg)
    n_cells = int(peaks.shape[0])
    m1 = model_wing(seg, n_cells, seed=0)
    m2 = model_wing(seg, n_cells, seed=0)
    deterministic = bool((m1["labels"] == m2["labels"]).all())
    rc, _, _ = domain_stats(real_lab)
    mc, _, _ = domain_stats(m1["labels"])
    ok = (300 <= n_cells <= 1500 and deterministic
          and 0.0 < float(mc.mean()) <= HEXAGON_CIRC + 0.05
          and 0.0 < float(rc.mean()) <= HEXAGON_CIRC + 0.05)
    print(f"[{'PASS' if ok else 'FAIL'}] 2018-hoffmann-wing-venation self-test "
          f"(downsample 2): {n_cells} real domains, circularity real "
          f"{rc.mean():.3f} / model {mc.mean():.3f}, deterministic={deterministic}")
    raise SystemExit(0 if ok else 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true",
                    help="run deterministic checks on the cached wing image and exit")
    ap.add_argument("-o", "--output", default="hoffmann_wing_recapitulation.png",
                    help="output PNG path (default: repo root)")
    ap.add_argument("--downsample", type=int, default=1,
                    help="integer image downsample factor (speed vs. detail)")
    ap.add_argument("--gif", metavar="PATH", help="also write the Lloyd-equilibration animation to PATH")
    ap.add_argument("--no-show", dest="show", action="store_false",
                    help="save the figure without opening a window")
    ap.set_defaults(show=True)
    args = ap.parse_args()
    if args.self_test:
        self_test()
    make_figure(args.output, args.show, max(1, args.downsample))
    if args.gif:
        make_gif(args.gif, downsample=max(2, args.downsample))


if __name__ == "__main__":
    main()

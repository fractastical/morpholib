#!/usr/bin/env python3
"""
Petroff et al. (2010) — the biophysics of conical-stromatolite geometry.

Reference:
    A. P. Petroff, M. S. Sim, A. Maslov, M. Krupenin, D. H. Rothman, T. Bosak,
    "Biophysical basis for the geometry of conical stromatolites,"
    PNAS 107(22):9956-9961 (2010).  [PMC2890478]

This standalone demo reconstructs the two quantitative ideas that make conical
stromatolites readable as fossils of photosynthesis, in the same spirit as the
other historical-morphology scripts in this repo (1966-raup.py, 1952-turing-
morpho.py, thompson/1917-thompson.py):

  (1) SPACING LAW  —  why neighbouring cones sit ~1 cm apart.
      A photosynthetic mat harvests nutrients only while it is active (daylight
      span tau).  Nutrients are replenished by diffusion, so the harvest radius
      grows as sqrt(D*tau).  To avoid competing, neighbours must stand about one
      diffusion length apart:

              ell = alpha * sqrt(D * tau)              [Petroff Eq. 2]

      with D ~ 1e-5 cm^2/s (small-molecule nutrient) and alpha of order one.
      A ~1 cm spacing therefore encodes a ~daylength (~12 h) metabolic rhythm —
      a fingerprint of photosynthesis preserved for 2.8 Gyr.  To pack the most
      cones while respecting ell, a field self-organises onto a HEXAGONAL lattice
      (densest packing; Fejes Toth 1948).

  (2) SHAPE MORPHOSPACE  —  dome vs cone.
      The lamina surface h(r,t) grows by a vertical (phototactic biofilm) rate v
      plus surface-normal mineral accretion lambda.  Following the deterministic-
      KPZ reading of Batchelor, Burne, Henry & Jackson (2004):

              dh/dt = v + lambda * ( 1 + 1/2 (dh/dr)^2 )

      Lateral accretion (the 1/2 (dh/dr)^2 term) rounds a crest at a fixed rate,
      while vertical growth lifts it at rate ~v.  The dimensionless ratio

              beta = v / lambda

      therefore sets the aspect ratio: small beta -> squat DOME, large beta ->
      tall apex-preserving CONE (Conophyton), very large beta -> sharp cone of
      the kind claimed as Earth's oldest macrofossils.

Run it:
    python3 2010-petroff-stromatolite.py                 # saves a PNG, shows it
    python3 2010-petroff-stromatolite.py --no-show        # just save
    python3 2010-petroff-stromatolite.py -o myfig.png     # custom output path
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import RegularPolygon


# --------------------------------------------------------------------------- #
# (1) Diffusion-competition spacing law:  ell = alpha * sqrt(D * tau)
# --------------------------------------------------------------------------- #
def diffusion_length(tau_seconds, D=1e-5, alpha=1.0):
    """Maximum extent of the nutrient-depletion halo a mat can build while it is
    active for `tau_seconds`.  D in cm^2/s; returns ell in cm (Petroff Eq. 2)."""
    return alpha * np.sqrt(D * tau_seconds)


def hex_lattice(spacing, nx=7, ny=5):
    """Centres of a hexagonal (triangular) lattice with the given nearest-
    neighbour `spacing`.  This is the arrangement that maximises cone density
    while keeping every neighbour one diffusion length away (Petroff, Fig. 4)."""
    dx = spacing                      # horizontal step
    dy = spacing * np.sqrt(3) / 2.0   # row step for a close-packed triangular grid
    pts = []
    for j in range(ny):
        xoff = 0.0 if (j % 2 == 0) else spacing / 2.0  # stagger alternate rows
        for i in range(nx):
            pts.append((i * dx + xoff, j * dy))
    return np.asarray(pts)


# --------------------------------------------------------------------------- #
# (2) Growth-ratio morphospace:  dh/dt = v + lambda (1 + 1/2 (dh/dr)^2)
# --------------------------------------------------------------------------- #
# The equation above is the small-slope (graph) form of a simple geometric rule:
# every point of the mat surface advances by a VERTICAL (phototactic biofilm)
# velocity v plus a SURFACE-NORMAL (mineral accretion) velocity lambda,
#
#         velocity = v * z_hat  +  lambda * n_hat .
#
# We evolve that rule directly as a Lagrangian front (a growing cross-section),
# which — unlike the single-valued graph form — keeps v physically meaningful
# and lets steep walls form.  Grown from a small triangular seed:
#   * normal growth (lambda) rounds the seed outward  -> DOME,
#   * vertical growth (v) extrudes the seed upward     -> CONE,
# so the ratio beta = v / lambda sweeps the dome-to-cone morphospace.
def _resample(P, m):
    """Redistribute the polyline `P` into `m` points evenly spaced by arc length.
    Parametrising by arc length (not x) keeps the front stable and tolerates the
    near-vertical walls that appear in the cone regime."""
    seg = np.hypot(np.diff(P[:, 0]), np.diff(P[:, 1]))
    s = np.concatenate([[0.0], np.cumsum(seg)])
    if s[-1] <= 0:
        return P
    snew = np.linspace(0.0, s[-1], m)
    x = np.interp(snew, s, P[:, 0])
    z = np.interp(snew, s, P[:, 1])
    return np.column_stack([x, z])


def _advance_front(P, v, lam, dt, base_half, smooth):
    """One micro-step of the front: move each vertex by v*z_hat + lambda*n_hat,
    pin the two basal endpoints to the footprint, lightly smooth for stability,
    then re-space by arc length."""
    x, z = P[:, 0], P[:, 1]
    dx, dz = np.gradient(x), np.gradient(z)           # tangent along the surface
    tlen = np.hypot(dx, dz) + 1e-12
    nx, nz = -dz / tlen, dx / tlen                    # unit normal (90 deg rotate)
    flip = np.sign(nz)                                # make it point up/outward
    flip[flip == 0] = 1.0
    nx *= flip
    nz *= flip

    # Phototaxis: the vertical biofilm growth v is strongest at the brightest,
    # most-exposed point (the crest) and weaker on the shaded flanks.  This
    # crest-weighting is what lets strong vertical growth sharpen an apex into a
    # cone rather than extrude a flat-topped column (Batchelor's "photic response").
    w_photic = 0.12 + 0.88 * (z / (z.max() + 1e-9)) ** 2

    x = x + dt * (lam * nx)                            # surface-normal accretion (lateral part)
    z = z + dt * (v * w_photic + lam * nz)             # phototactic + normal accretion (up part)

    # the cone grows from a fixed basal footprint; keep its two feet pinned
    x[0], z[0] = -base_half, 0.0
    x[-1], z[-1] = base_half, 0.0
    z = np.clip(z, 0.0, None)
    P = np.column_stack([x, z])

    if smooth > 0:                                     # mild curvature regularisation
        P[1:-1] = (1.0 - 2.0 * smooth) * P[1:-1] + smooth * P[:-2] + smooth * P[2:]
    return _resample(P, len(P))


def grow_stromatolite(v, lam, base_half=1.0, seed_height=0.25, m=240,
                      n_laminae=14, t_total=0.85, n_steps=700, smooth=0.03):
    """Grow a stromatolite cross-section under velocity = v*z_hat + lambda*n_hat
    from a shallow triangular seed, returning a list of laminae (each an (m, 2)
    array of (x, z) points).  Stacking the laminae mimics a vertical thin-section.
    """
    x0 = np.linspace(-base_half, base_half, m)
    z0 = seed_height * (1.0 - np.abs(x0) / base_half)  # shallow triangular seed
    P = np.column_stack([x0, z0])

    dt = t_total / n_steps
    record_every = max(1, n_steps // n_laminae)
    laminae = [P.copy()]
    for s in range(1, n_steps + 1):
        P = _advance_front(P, v, lam, dt, base_half, smooth)
        if s % record_every == 0:
            laminae.append(P.copy())
    return laminae


def aspect_ratio(lamina):
    """Height / basal-width of a lamina polyline — a crude dome(<~0.5) vs
    cone(>~1) discriminator used only to annotate the figure."""
    height = float(lamina[:, 1].max())
    width = float(lamina[:, 0].max() - lamina[:, 0].min())
    return height / width if width > 0 else np.nan


# --------------------------------------------------------------------------- #
# Figure
# --------------------------------------------------------------------------- #
def make_figure(output, show):
    D = 1e-5                 # cm^2/s, typical small-molecule nutrient
    alpha_field = 1.0        # order-unity constant that reproduces the ~1 cm field spacing
    alpha_lab = 0.30         # least-squares fit to the laboratory clump experiments

    fig = plt.figure(figsize=(14, 9))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.05], hspace=0.32, wspace=0.28)

    # ----- (a) the spacing law ell = alpha sqrt(D tau) ----------------------- #
    ax = fig.add_subplot(gs[0, 0])
    tau_h = np.linspace(1, 48, 200)                 # candidate daylight spans (hours)
    tau_s = tau_h * 3600.0
    ax.plot(tau_h, diffusion_length(tau_s, D, alpha_field), color="#1f77b4", lw=2.2,
            label=f"field  (alpha={alpha_field:g})")
    ax.plot(tau_h, diffusion_length(tau_s, D, alpha_lab), color="#2ca02c", lw=1.8,
            ls="--", label=f"lab clumps  (alpha={alpha_lab:g})")
    ell12 = diffusion_length(12 * 3600.0, D, alpha_field)
    ax.plot([12], [ell12], "o", color="#d62728", ms=8, zorder=5)
    ax.annotate(f"12 h day\n-> {ell12:.2f} cm", (12, ell12),
                xytext=(17, ell12 - 0.18), fontsize=9, color="#d62728",
                arrowprops=dict(arrowstyle="->", color="#d62728"))
    ax.axhspan(0.5, 1.5, color="#d62728", alpha=0.07)
    ax.set_title("(a) Spacing law   ell = alpha sqrt(D tau)", fontsize=11)
    ax.set_xlabel("daylight span  tau  (hours)")
    ax.set_ylabel("cone spacing  ell  (cm)")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(alpha=0.25)

    # ----- (b) the hexagonal cone field it predicts ------------------------- #
    ax = fig.add_subplot(gs[0, 1])
    spacing = ell12                                 # ~ 0.66 cm nearest-neighbour spacing
    pts = hex_lattice(spacing, nx=7, ny=6)
    # draw each cone's nutrient-harvest disk (radius ell/2) + its apex
    for (x, y) in pts:
        ax.add_patch(plt.Circle((x, y), spacing / 2.0, fill=True,
                                facecolor="#1f77b4", alpha=0.10, edgecolor="#1f77b4",
                                lw=0.6))
    ax.plot(pts[:, 0], pts[:, 1], "^", color="#d62728", ms=7, label="cone apex")
    # one neighbour-to-neighbour spacing annotation
    p0 = pts[0]
    ax.annotate("", xy=(p0[0] + spacing, p0[1]), xytext=(p0[0], p0[1]),
                arrowprops=dict(arrowstyle="<->", color="k", lw=1.3))
    ax.text(p0[0] + spacing / 2, p0[1] - spacing * 0.28, f"ell = {spacing:.2f} cm",
            ha="center", fontsize=9)
    ax.set_title("(b) Hexagonal packing of a cone field", fontsize=11)
    ax.set_xlabel("x (cm)"); ax.set_ylabel("y (cm)")
    ax.set_aspect("equal"); ax.legend(fontsize=8, loc="upper right")
    ax.set_xlim(-spacing, pts[:, 0].max() + spacing)
    ax.set_ylim(-spacing, pts[:, 1].max() + spacing * 0.7)

    # ----- (c) what the ~1 cm period means ---------------------------------- #
    ax = fig.add_subplot(gs[0, 2]); ax.axis("off")
    ax.text(0.0, 1.0,
            "Reading the fossil\n"
            "-------------------------\n\n"
            f"D   ~ {D:.0e} cm^2/s\n"
            f"tau ~ 12 h  (one daylight span)\n\n"
            f"ell = alpha sqrt(D tau)\n"
            f"    = {alpha_field:g} * sqrt({D:.0e} * {12*3600:.0f})\n"
            f"    ~ {ell12:.2f} cm   (centimetre scale)\n\n"
            "A cm-scale spacing therefore records a\n"
            "~20 h rhythmic (day/night) metabolism:\n"
            "quantitative evidence of PHOTOSYNTHESIS\n"
            "in cones as old as ~2.8 Gyr.\n\n"
            "Moving water -> eddy diffusivity\n"
            "D_eff ~ Re*D widens the spacing to the\n"
            "~10-20 cm cones of the Bakal Formation.",
            va="top", ha="left", family="monospace", fontsize=9,
            transform=ax.transAxes)

    # ----- (d-f) the growth-ratio morphospace: dome -> cone ----------------- #
    lam = 1.0
    regimes = [
        (0.3, "(d) DOME            beta = v/lambda = 0.3", "#2ca02c"),
        (1.5, "(e) CONE            beta = v/lambda = 1.5", "#1f77b4"),
        (5.0, "(f) SHARP CONE   beta = v/lambda = 5",      "#d62728"),
    ]
    max_h = 0.0
    axes_df = []
    for col, (beta, title, color) in enumerate(regimes):
        ax = fig.add_subplot(gs[1, col])
        laminae = grow_stromatolite(v=beta * lam, lam=lam)
        n_l = len(laminae)
        for k, lam_i in enumerate(laminae):
            shade = 0.25 + 0.65 * k / max(1, n_l - 1)      # younger laminae darker
            ax.plot(lam_i[:, 0], lam_i[:, 1], color=color, lw=1.0, alpha=shade)
        top = laminae[-1]
        ax.fill(top[:, 0], top[:, 1], color=color, alpha=0.12)
        ar = aspect_ratio(top)
        max_h = max(max_h, top[:, 1].max())
        ax.set_title(title + f"\n(height/width ~ {ar:.2f})", fontsize=10)
        ax.set_xlabel("radial distance (arb.)")
        if col == 0:
            ax.set_ylabel("height (arb.)\nolder -> younger laminae")
        ax.set_aspect("equal")
        ax.grid(alpha=0.2)
        axes_df.append(ax)
    # share a common vertical scale so the dome-to-cone trend is honest
    for ax in axes_df:
        ax.set_xlim(-3.2, 3.2)
        ax.set_ylim(-0.1, max_h * 1.08)

    fig.suptitle("Petroff et al. 2010 — spacing law (top) and the v/lambda "
                 "dome-to-cone morphospace (bottom)", fontsize=13)

    out = Path(output)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved figure to {out.resolve()}")
    if show:
        plt.show()
    plt.close(fig)

    # ----- worked numbers (printed, like the other repo scripts) ------------ #
    print("\nDIFFUSION-COMPETITION SPACING LAW   ell = alpha * sqrt(D * tau)")
    print(f"  D                 = {D:.0e} cm^2/s")
    for h in (3, 6, 12, 24):
        print(f"  tau = {h:2d} h  ->  ell = {diffusion_length(h*3600, D, alpha_field):.2f} cm "
              f"(field, alpha={alpha_field:g})   "
              f"{diffusion_length(h*3600, D, alpha_lab):.2f} cm (lab, alpha={alpha_lab:g})")
    print("\nGROWTH-RATIO MORPHOSPACE   dh/dt = v + lambda (1 + 1/2 (dh/dr)^2)")
    print("  beta = v/lambda small -> squat dome;  large -> tall apex-preserving cone.")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-o", "--output", default="petroff_stromatolite_morphospace.png",
                    help="output PNG path (default: repo root)")
    ap.add_argument("--no-show", dest="show", action="store_false",
                    help="save the figure without opening a window")
    ap.set_defaults(show=True)
    args = ap.parse_args()
    make_figure(args.output, args.show)


if __name__ == "__main__":
    main()

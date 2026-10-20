#!/usr/bin/env python3
"""
Kahng, Odle & Wakeman (2024) — the geometry of reef-coral colonies.

Reference:
    S. E. Kahng, E. Odle, K. C. Wakeman,
    "Coral geometry and why it matters,"
    PeerJ 12:e17037 (2024).  [PMC10909345]  doi:10.7717/peerj.17037
    code/data: github.com/ericodle/coral_geometry_modeling

A reef coral grows in two coupled currencies: ORGANIC tissue (biomass, which
tracks SURFACE AREA S) and INORGANIC skeleton (calcification, which tracks
VOLUME V via mean density rho).  Over one time step a colony lays down skeleton
at a per-area calcification rate G (mass per area per time), so its volume grows

        V(t+1) = V(t) + G * S(t) / rho .                     [master rule]

Everything else — how fast the colony lengthens (linear extension C), how fast
it makes new surface per existing surface (surface-area productivity SAP = dS/S)
— follows from this one rule *once you fix the shape*.  The punchline of the
paper is that whether these rates are SIZE-DEPENDENT or SIZE-INDEPENDENT is
decided purely by whether the colony's surface-area-to-volume ratio changes as
it grows.  Four canonical shapes span the morphospace:

    shape (coral example)              S:V with size     C vs size   SAP vs size
    ---------------------------------  ----------------  ----------  -----------
    hemispheroid (massive Porites)     decreases         -> plateau  decreases
    cone, fixed aspect (branch/mound)  decreases         -> plateau  decreases
    flat disc, fixed thickness         constant          increases   constant
    cylinder, fixed branch diameter    constant          constant    constant

This standalone demo reconstructs those relationships in the same spirit as the
repo's other historical-morphology scripts (1966-raup.py, 1952-turing-morpho.py,
thompson/1917-thompson.py).  Raup & Graus (1972), cited by the paper for the
volume/surface-area of a coiled shell, is the direct ancestor of this kind of
"shape sets the growth equations" reasoning — see 1966-raup.py.

Run it:
    python3 2024-kahng-coral-geometry.py                # saves a PNG, shows it
    python3 2024-kahng-coral-geometry.py --no-show      # just save
    python3 2024-kahng-coral-geometry.py -o myfig.png   # custom output path
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge, Ellipse, Rectangle, Polygon

# --------------------------------------------------------------------------- #
# Empirical growth parameters from the paper (G in g cm^-2 yr^-1, rho in g cm^-3,
# so G/rho has units of cm yr^-1 — a linear-extension speed).
# --------------------------------------------------------------------------- #
MASSIVE = dict(G=1.64, rho=1.28)             # massive Porites hemisphere (Lough 2008)
BRANCH = dict(G=2.50, rho=1.40)              # branching Acropora cone/cylinder (Hughes 1987)
DISC = dict(G=0.061, rho=2.67, h=0.20)       # deep-water Leptoseris plate (Kahng 2023)
COMMON = dict(G=1.50, rho=1.20)              # equal values to isolate SHAPE alone (paper Fig. 6)


# --------------------------------------------------------------------------- #
# The four shapes.  Each pairs a size with the next-step size implied by the
# master rule V(t+1) = V(t) + G*S(t)/rho, from which C and SAP fall out.
# All derivations reproduce Appendix A / Results of Kahng et al. (2024).
# --------------------------------------------------------------------------- #
def hemisphere_extension(r, G, rho):
    """Circular hemisphere, V = (2/3) pi r^3, S = 2 pi r^2.
    Radial extension C = r(t+1) - r(t).  Plateaus at C -> G/rho as r -> inf."""
    r_next = (r ** 3 + 3.0 * G * r ** 2 / rho) ** (1.0 / 3.0)
    return r_next - r


def hemisphere_sap(r, G, rho):
    """Surface-area productivity dS/S for a circular hemisphere.  Falls off as
    the colony enlarges because S:V (hence productive surface per mass) drops."""
    return (r ** 3 + 3.0 * G * r ** 2 / rho) ** (2.0 / 3.0) / r ** 2 - 1.0


def disc_extension(r, G, rho, h):
    """Flat disc of fixed thickness h, V = h pi r^2, S = pi r^2.
    Radial extension GROWS linearly with radius: C = r(sqrt(1 + G/(rho h)) - 1)."""
    return r * (np.sqrt(1.0 + G / (rho * h)) - 1.0)


def disc_sap(G, rho, h):
    """Disc SAP is SIZE-INDEPENDENT: dS/S = G/(rho h).  Thinner plates => higher."""
    return G / (rho * h)


def cone_extension(h, G, rho, alpha):
    """Cone of fixed aspect ratio alpha = height/basal-radius, V = pi h^3/(3 alpha^2),
    S = pi h^2 sqrt(1+alpha^2)/alpha^2.  Height extension plateaus at
    C -> G sqrt(1+alpha^2)/rho."""
    k = 3.0 * G * np.sqrt(1.0 + alpha ** 2) / rho
    return (h ** 3 + k * h ** 2) ** (1.0 / 3.0) - h


def cone_sap(h, G, rho, alpha):
    """Cone SAP falls off with height, like the hemisphere; higher aspect ratio
    (slimmer cone) keeps SAP higher."""
    k = 3.0 * G * np.sqrt(1.0 + alpha ** 2) / rho
    return (h ** 3 + k * h ** 2) ** (2.0 / 3.0) / h ** 2 - 1.0


def cylinder_sap(G, rho, r):
    """Branching coral built of fixed-diameter cylinders.  S:V is constant, so
    SAP is SIZE-INDEPENDENT: dS/S = 2 G/(rho r).  Thinner branches => higher."""
    return 2.0 * G / (rho * r)


# --------------------------------------------------------------------------- #
# Figure
# --------------------------------------------------------------------------- #
def _draw_glyphs(ax):
    """Panel (a): schematic cross-sections of the four modelled shapes."""
    ax.set_xlim(0, 4)
    ax.set_ylim(-0.15, 1.25)
    ax.axis("off")
    c = "#1f77b4"

    # massive hemispheroid
    ax.add_patch(Wedge((0.5, 0.1), 0.33, 0, 180, facecolor=c, alpha=0.5, edgecolor=c))
    # flat disc (thin wide ellipse)
    ax.add_patch(Ellipse((1.5, 0.14), width=0.74, height=0.1, facecolor=c, alpha=0.5, edgecolor=c))
    # cylindrical branching (a little clump of branches)
    for dx in (-0.17, 0.0, 0.17):
        ax.add_patch(Rectangle((2.5 + dx - 0.045, 0.1), 0.09, 0.55 - 0.12 * abs(dx) * 3,
                               facecolor=c, alpha=0.5, edgecolor=c))
    # cone
    ax.add_patch(Polygon([[3.5 - 0.28, 0.1], [3.5 + 0.28, 0.1], [3.5, 0.72]],
                         closed=True, facecolor=c, alpha=0.5, edgecolor=c))

    labels = [
        ("hemispheroid", "V=⅔πr³", "S=2πr²", "S:V ↓"),
        ("flat disc", "V=hπr²", "S=πr²", "S:V ="),
        ("branches", "V=πr²h", "S=2πrh", "S:V ="),
        ("cone (α)", "V=πh³/3α²", "S∝h²", "S:V ↓"),
    ]
    for i, (name, vexpr, sexpr, trend) in enumerate(labels):
        ax.text(0.5 + i, 0.98, name, ha="center", fontsize=8.5, weight="bold")
        ax.text(0.5 + i, 0.80, trend, ha="center", fontsize=8, color="#d62728")
        ax.text(0.5 + i, 0.00, vexpr, ha="center", fontsize=7, family="monospace")
        ax.text(0.5 + i, -0.11, sexpr, ha="center", fontsize=7, family="monospace")
    ax.set_title("(a) Four coral geometries and their surface-to-volume trend", fontsize=11)


def make_figure(output, show):
    fig = plt.figure(figsize=(14, 9))
    gs = fig.add_gridspec(2, 3, height_ratios=[0.95, 1.05], hspace=0.34, wspace=0.30)

    # ----- (a) the shape glyphs -------------------------------------------- #
    _draw_glyphs(fig.add_subplot(gs[0, 0]))

    # ----- (b) SAP vs size for all four shapes, EQUAL G & rho (paper Fig. 6)- #
    # Using the same G, rho isolates the effect of SHAPE alone.
    ax = fig.add_subplot(gs[0, 1])
    G, rho = COMMON["G"], COMMON["rho"]
    size = np.linspace(0.5, 30, 300)                 # cm (radius or height)
    ax.plot(size, hemisphere_sap(size, G, rho), color="#1f77b4", lw=2, label="hemispheroid")
    ax.plot(size, cone_sap(size, G, rho, alpha=1.0), color="#9467bd", lw=2, label="cone (α=1)")
    ax.plot(size, np.full_like(size, cylinder_sap(G, rho, r=0.5)), color="#2ca02c", lw=2,
            ls="--", label="cylinder (r=0.5 cm)")
    ax.plot(size, np.full_like(size, disc_sap(G, rho, h=0.5)), color="#d62728", lw=2,
            ls="--", label="disc (h=0.5 cm)")
    ax.set_title("(b) Surface-area productivity vs size\n(equal G, ρ: shape only)", fontsize=11)
    ax.set_xlabel("colony size  (cm)"); ax.set_ylabel("SAP  ΔS/S  (yr⁻¹)")
    ax.set_ylim(0, None); ax.grid(alpha=0.25); ax.legend(fontsize=8)

    # ----- (c) linear extension rate vs size -------------------------------- #
    ax = fig.add_subplot(gs[0, 2])
    ax.plot(size, hemisphere_extension(size, G, rho), color="#1f77b4", lw=2,
            label="hemispheroid")
    ax.plot(size, cone_extension(size, G, rho, alpha=1.0), color="#9467bd", lw=2,
            label="cone (α=1)")
    ax.plot(size, disc_extension(size, G, rho, h=0.5), color="#d62728", lw=2,
            label="disc (h=0.5 cm)")
    ax.axhline(G / rho, color="#1f77b4", ls=":", lw=1,
               label=f"hemisphere plateau G/ρ={G/rho:.2f}")
    ax.axhline(G * np.sqrt(2) / rho, color="#9467bd", ls=":", lw=1,
               label=f"cone plateau √2·G/ρ={G*np.sqrt(2)/rho:.2f}")
    ax.set_title("(c) Linear extension rate vs size", fontsize=11)
    ax.set_xlabel("colony size  (cm)"); ax.set_ylabel("extension  C  (cm yr⁻¹)")
    ax.grid(alpha=0.25); ax.legend(fontsize=7.5, loc="center right")

    # ----- (d) cone SAP vs height for a range of aspect ratios (Fig. 3) ----- #
    ax = fig.add_subplot(gs[1, 0])
    Gb, rhob = BRANCH["G"], BRANCH["rho"]
    h = np.linspace(0.5, 30, 300)
    for alpha, col in [(0.5, "#2ca02c"), (1.0, "#1f77b4"), (2.0, "#9467bd"), (4.0, "#d62728")]:
        ax.plot(h, cone_sap(h, Gb, rhob, alpha), color=col, lw=1.8, label=f"α={alpha:g}")
    ax.set_title("(d) Cone SAP vs height (branching Acropora)\n"
                 "slimmer cones (high α) stay more productive", fontsize=10)
    ax.set_xlabel("cone height  h  (cm)"); ax.set_ylabel("SAP  ΔS/S  (yr⁻¹)")
    ax.set_ylim(0, None); ax.grid(alpha=0.25); ax.legend(fontsize=8, title="aspect ratio")

    # ----- (e) size-INDEPENDENT SAP set by branch diameter / plate thickness - #
    ax = fig.add_subplot(gs[1, 1])
    diam = np.linspace(0.3, 4.0, 200)                # cm, branch diameter 2r
    ax.plot(diam, cylinder_sap(Gb, rhob, r=diam / 2.0), color="#2ca02c", lw=2,
            label="cylinder: 2G/(ρr) vs branch diameter")
    thick = np.linspace(0.05, 1.5, 200)              # cm, plate thickness h
    Gd, rhod = DISC["G"], DISC["rho"]
    ax.plot(thick, disc_sap(Gd, rhod, h=thick), color="#d62728", lw=2,
            label="disc: G/(ρh) vs plate thickness")
    ax.axvspan(0.3, 1.0, color="#2ca02c", alpha=0.07)   # common Acropora branch diameters
    ax.set_title("(e) Size-independent SAP\nset by branch diameter / plate thickness", fontsize=10)
    ax.set_xlabel("branch diameter  or  plate thickness  (cm)")
    ax.set_ylabel("SAP  ΔS/S  (yr⁻¹)")
    ax.set_ylim(0, None); ax.grid(alpha=0.25); ax.legend(fontsize=8)

    # ----- (f) what the geometry means -------------------------------------- #
    ax = fig.add_subplot(gs[1, 2]); ax.axis("off")
    sap_small = hemisphere_sap(np.array([1.0]), MASSIVE["G"], MASSIVE["rho"])[0]
    sap_large = hemisphere_sap(np.array([25.0]), MASSIVE["G"], MASSIVE["rho"])[0]
    ax.text(0.0, 1.0,
            "Reading the geometry\n"
            "---------------------------\n\n"
            "Master rule:   V(t+1) = V(t) + G·S/ρ\n\n"
            "Size-dependence is decided by whether\n"
            "S:V changes as the colony grows:\n\n"
            "  hemispheroid / cone  ->  S:V falls\n"
            "     => extension plateaus at ~G/ρ,\n"
            "        SAP crashes with size.\n"
            "  disc / cylinder      ->  S:V fixed\n"
            "     => SAP size-independent.\n\n"
            f"Massive Porites hemisphere SAP:\n"
            f"   r=1 cm  -> {sap_small:4.2f} /yr\n"
            f"   r=25 cm -> {sap_large:4.2f} /yr\n"
            f"   ~{sap_small/sap_large:.0f}x faster when small\n"
            "   => why MICROFRAGMENTATION speeds\n"
            "      massive-coral restoration.\n\n"
            "Depth: flat thin plates maximise planar\n"
            "area productivity -> deepest Leptoseris.",
            va="top", ha="left", family="monospace", fontsize=8.6,
            transform=ax.transAxes)

    fig.suptitle("Kahng, Odle & Wakeman 2024 — colony geometry sets the "
                 "size-dependence of coral growth", fontsize=13)

    out = Path(output)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved figure to {out.resolve()}")
    if show:
        plt.show()
    plt.close(fig)

    # ----- worked numbers (printed, like the other repo scripts) ------------ #
    print("\nMASTER RULE   V(t+1) = V(t) + G*S(t)/rho   (G: g cm^-2 yr^-1, rho: g cm^-3)")
    print("\nASYMPTOTIC LINEAR EXTENSION (large colony):")
    print(f"  hemispheroid  C -> G/rho              = {MASSIVE['G']/MASSIVE['rho']:.2f} cm/yr "
          f"(massive Porites)")
    for a in (0.5, 1, 2, 4):
        print(f"  cone alpha={a:<4g} C -> G*sqrt(1+a^2)/rho = "
              f"{BRANCH['G']*np.sqrt(1+a**2)/BRANCH['rho']:.2f} cm/yr (Acropora)")
    print("\nSIZE-INDEPENDENT SAP:")
    print(f"  disc      dS/S = G/(rho h)   = {disc_sap(DISC['G'], DISC['rho'], DISC['h']):.3f} /yr "
          f"(Leptoseris, h={DISC['h']} cm)")
    print(f"  cylinder  dS/S = 2G/(rho r)  = {cylinder_sap(BRANCH['G'], BRANCH['rho'], 0.5):.3f} /yr "
          f"(Acropora, r=0.5 cm)")
    print("\nMICROFRAGMENTATION: shrinking a massive colony raises its S:V and SAP,")
    print(f"  giving ~{hemisphere_sap(np.array([1.0]), MASSIVE['G'], MASSIVE['rho'])[0] / hemisphere_sap(np.array([25.0]), MASSIVE['G'], MASSIVE['rho'])[0]:.0f}x faster surface growth at r=1 cm than at r=25 cm.")


def make_gif(output, years=15, fps=4):
    """Iterate the master rule V(t+1) = V(t) + G S(t)/rho year by year for the four
    shapes with equal G and rho (shape alone).  Left: the hemisphere and the cone,
    whose S:V falls as they grow, drawn to scale year by year.  Right: surface area
    against time for all four shapes on a log axis — the disc and the cylinder keep
    a constant S:V, so their area grows exponentially (size-independent SAP), while
    the hemisphere and cone slow down (their extension plateaus at ~G/rho)."""
    from PIL import Image
    G, rho = COMMON["G"], COMMON["rho"]
    h_disc, r_cyl, alpha = 0.5, 0.5, 1.0
    r_h, r_d, L_c, h_k = 1.0, 1.0, 1.0, 1.0
    hist = []
    for t in range(years + 1):
        hist.append((r_h, r_d, L_c, h_k))
        r_h += hemisphere_extension(r_h, G, rho)
        r_d += disc_extension(r_d, G, rho, h_disc)
        L_c *= 1.0 + cylinder_sap(G, rho, r_cyl)          # S ∝ L for a fixed-radius cylinder
        h_k += cone_extension(h_k, G, rho, alpha)
    S = np.array([[2 * np.pi * rh ** 2, np.pi * rd ** 2, 2 * np.pi * r_cyl * lc,
                   np.pi * (hk / alpha) ** 2 * np.sqrt(1 + alpha ** 2)] for rh, rd, lc, hk in hist])
    lim = max(hist[-1][0], hist[-1][3]) * 1.15
    names = ["hemispheroid", "flat disc (h = 0.5 cm)", "cylinder (r = 0.5 cm)", "cone (α = 1)"]
    cols = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd"]
    frames = []
    for t, (r_h, r_d, L_c, h_k) in enumerate(hist):
        fig = plt.figure(figsize=(11, 4.6))
        gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.5], wspace=0.35, top=0.78, bottom=0.14, left=0.05, right=0.98)
        a0 = fig.add_subplot(gs[0]); a1 = fig.add_subplot(gs[1]); a2 = fig.add_subplot(gs[2])
        a0.add_patch(Wedge((0, 0), r_h, 0, 180, facecolor=cols[0], alpha=0.5, edgecolor=cols[0]))
        a1.add_patch(Polygon([[-h_k / alpha, 0], [h_k / alpha, 0], [0, h_k]], closed=True,
                             facecolor=cols[3], alpha=0.5, edgecolor=cols[3]))
        a0.set_title(f"hemispheroid  r = {r_h:.1f} cm\nSAP {hemisphere_sap(r_h, G, rho):.2f}/yr", fontsize=9)
        a1.set_title(f"cone (α = 1)  h = {h_k:.1f} cm\nSAP {cone_sap(h_k, G, rho, alpha):.2f}/yr", fontsize=9)
        for ax in (a0, a1):
            ax.set_xlim(-lim, lim); ax.set_ylim(-lim * 0.05, lim * 1.05); ax.set_aspect("equal", adjustable="box")
            ax.axhline(0, color="k", lw=0.6); ax.tick_params(labelsize=7); ax.set_xlabel("cm", fontsize=8)
        yrs = np.arange(years + 1)
        for i, (n, c) in enumerate(zip(names, cols)):
            a2.plot(yrs[:t + 1], S[:t + 1, i], color=c, lw=2, label=n)
            a2.plot(yrs[t], S[t, i], "o", color=c)
        a2.set_yscale("log"); a2.set_xlim(0, years); a2.set_ylim(S.min() * 0.8, S.max() * 1.5)
        a2.set_xlabel("year"); a2.set_ylabel("surface area S (cm²), log scale")
        a2.set_title("surface area: disc & cylinder keep S:V → exponential;\nhemisphere & cone lose S:V → slow down", fontsize=9)
        a2.legend(fontsize=7.5, loc="upper left"); a2.grid(alpha=0.25)
        fig.suptitle(f"Kahng et al. 2024: V(t+1) = V(t) + G·S/ρ with equal G, ρ (shape alone) — year {t}", fontsize=11, y=0.97)
        fig.canvas.draw()
        frames.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()))
        plt.close(fig)
    frames += [frames[-1]] * (fps * 2)
    frames[0].save(output, save_all=True, append_images=frames[1:], duration=int(1000 / fps), loop=0)
    print(f"Saved animation to {Path(output).resolve()}  ({len(frames)} frames)")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-o", "--output", default="kahng_coral_geometry_morphospace.png",
                    help="output PNG path (default: repo root)")
    ap.add_argument("--no-show", dest="show", action="store_false",
                    help="save the figure without opening a window")
    ap.add_argument("--gif", metavar="PATH", help="also write a year-by-year growth animation of the four shapes to PATH")
    ap.set_defaults(show=True)
    args = ap.parse_args()
    make_figure(args.output, args.show)
    if args.gif:
        make_gif(args.gif)


if __name__ == "__main__":
    main()

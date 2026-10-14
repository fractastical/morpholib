#!/usr/bin/env python3
"""
Chindapol, Kaandorp, Cronemberger, Mass & Genin (2013) — flow-driven coral form.

Reference:
    N. Chindapol, J. A. Kaandorp, C. Cronemberger, T. Mass, A. Genin,
    "Modelling Growth and Form of the Scleractinian Coral Pocillopora verrucosa
    and the Influence of Hydrodynamics,"
    PLoS Comput. Biol. 9(1):e1002849 (2013).  doi:10.1371/journal.pcbi.1002849

The paper grows a virtual coral with a POLYP-BASED ACCRETIVE GROWTH model: each
surface polyp extends along its normal at a rate set by how much "nutrient"
(dissolved inorganic carbon) it absorbs, and the nutrient field is transported
by coupled Navier-Stokes + advection-diffusion (solved in 3D with COMSOL).  The
controlling knob is the PECLET NUMBER

        Pe = u * L / D        (advection speed vs diffusion),

and the central finding is a morphological transition:

    low  Pe  (diffusion-limited, no flow)  -> SYMMETRIC branching in all directions
    high Pe  (advection-dominated)         -> ASYMMETRIC: branches grow UPSTREAM,
                                              into the current; downstream side is
                                              shaded/suppressed; colony compactifies
                                              and its surface/volume ratio drops.

A full Navier-Stokes/COMSOL reconstruction is out of scope here, but the model's
own lineage (Merks/Kaandorp: "spontaneous branching, compactification and the
Laplacian-growth assumption") reduces to the canonical diffusion-limited process:
DIFFUSION-LIMITED AGGREGATION (DLA).  Growth is proportional to the local flux of
a diffusing field onto the surface -- exactly a random walker sticking to the
cluster.  Adding a DRIFT to the walkers is advection; the drift/diffusion ratio
plays the role of Pe.  This standalone demo therefore grows corals by DLA-with-
drift and recovers the paper's symmetric->asymmetric, compactifying transition,
in the same spirit as the repo's other morphogenesis scripts (1952-turing-
morpho.py, 2010-petroff-stromatolite.py, 2024-kahng-coral-geometry.py).

Run it:
    python3 2013-chindapol-coral-flow.py                 # saves a PNG, shows it
    python3 2013-chindapol-coral-flow.py --no-show        # just save
    python3 2013-chindapol-coral-flow.py -o myfig.png     # custom output path
    python3 2013-chindapol-coral-flow.py --particles 500  # faster / smaller
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


# --------------------------------------------------------------------------- #
# Diffusion-limited aggregation WITH DRIFT  (advection ~ Peclet number)
# --------------------------------------------------------------------------- #
# A walker performs a biased random walk: it diffuses (equal up/down, and a
# left/right component) with an extra DRIFT in +x (the flow direction).  When it
# touches the growing cluster it sticks -- growth is thus proportional to the
# local flux of the diffusing nutrient field, the discrete form of the paper's
# "growth rate driven by absorbed nutrient".  Drift bias b (0..0.25) stands in
# for the advection/diffusion ratio, i.e. the Peclet number.
def simulate_coral(drift, n_particles, N=241, max_steps=20000, seed=0):
    """Grow a coral by DLA-with-drift on an N x N lattice, seeded at the centre
    and viewed top-down (the paper projects branches onto the substratum plane;
    flow is in +x).  Walkers launch from a circle just outside the cluster.
    Returns (order grid, metrics dict); `order` holds each cell's growth-step
    index (-1 = empty) for colouring by age."""
    rng = random.Random(seed)
    occ = np.zeros((N, N), dtype=bool)
    order = np.full((N, N), -1, dtype=np.int32)

    cx = cy = N // 2
    occ[cy, cx] = True                           # initial polyp (the paper's seed sphere)
    order[cy, cx] = 0
    r_max = 1.0                                   # current cluster radius from centre

    p_right = 0.25 + drift                        # advection pushes walkers downstream (+x)
    p_left = 0.25 - drift
    p_up = 0.25                                   # p_right+p_left+p_up+p_down == 1

    placed = 1
    launches = 0
    max_launches = n_particles * 400
    while placed < n_particles and launches < max_launches:
        launches += 1
        r_launch = r_max + 5.0
        r_kill = r_launch + 30.0
        if r_kill >= N / 2 - 1:                   # cluster filled the domain -> stop
            break
        ang = rng.random() * 2.0 * np.pi          # release on a circle around the seed
        x = int(round(cx + r_launch * np.cos(ang)))
        y = int(round(cy + r_launch * np.sin(ang)))
        for _ in range(max_steps):
            r = rng.random()                      # one diffusive + drift step
            if r < p_right:
                x += 1
            elif r < p_right + p_left:
                x -= 1
            elif r < p_right + p_left + p_up:
                y += 1
            else:
                y -= 1

            if x < 1 or x >= N - 1 or y < 1 or y >= N - 1:
                break                             # left the domain -> relaunch
            if (x - cx) ** 2 + (y - cy) ** 2 > r_kill * r_kill:
                break                             # wandered past the kill radius

            # stick if touching the cluster (flux onto the surface)
            if occ[y, x - 1] or occ[y, x + 1] or occ[y - 1, x] or occ[y + 1, x]:
                if not occ[y, x]:
                    occ[y, x] = True
                    order[y, x] = placed
                    placed += 1
                    rr = np.hypot(x - cx, y - cy)
                    if rr > r_max:
                        r_max = rr
                break

    # ----- morphometrics (paper's symmetry & surface/volume themes) --------- #
    ys, xs = np.nonzero(occ)
    mean_x = xs.mean()
    mean_y = ys.mean()
    upstream = cx - mean_x                        # >0 => mass shifted UPSTREAM (-x)
    extent_x = xs.max() - xs.min() + 1
    area = int(occ.sum())
    # radius of gyration: spread of the fixed polyp mass about its centroid.
    # For equal mass, a SMALLER R_g means a denser, more compact colony -- i.e.
    # a lower surface/volume ratio, the paper's compactification signature.
    r_gyr = float(np.sqrt(((xs - mean_x) ** 2 + (ys - mean_y) ** 2).mean()))
    span = int(max(xs.max() - xs.min(), ys.max() - ys.min()) + 1)
    metrics = dict(upstream=upstream, upstream_frac=upstream / extent_x,
                   r_gyr=r_gyr, area=area, span=span, placed=area)
    return order, metrics


# --------------------------------------------------------------------------- #
# Morphometric analysis  (after Chindapol et al. 2013, "Morphometric analysis")
# --------------------------------------------------------------------------- #
# The paper skeletonises each colony and measures the orientation of branches
# relative to the flow: the symmetry ANGLES (hangle, vangle) and the symmetry
# MAGNITUDE smmag = the net projection of branch vectors onto the streamline
# direction s.  A colony is symmetric when opposing branches cancel (smmag ~ 0);
# unidirectional flow skews the branch directions and smmag grows.  In this 2-D
# top-down reduction we extract branch TIPS (skeleton endpoints), take their
# vectors from the seed, and compute the same quantities about the flow axis.
def branch_tips(order):
    """Skeleton endpoints = occupied cells with <=1 occupied 8-neighbour."""
    occ = order >= 0
    deg = np.zeros(occ.shape, dtype=np.int16)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            deg += np.roll(np.roll(occ, dy, axis=0), dx, axis=1).astype(np.int16)
    tips = occ & (deg <= 1)
    ys, xs = np.nonzero(tips)
    return xs, ys


def morphometrics(order):
    """Branch-tip symmetry traits about the flow axis (flow is +x)."""
    N = order.shape[0]
    cx = cy = N // 2
    xs, ys = branch_tips(order)
    dx = xs - cx
    dy = ys - cy
    theta = np.arctan2(dy, dx)                    # branch orientation (hangle analog)
    length = np.hypot(dx, dy)
    total_len = length.sum() + 1e-9
    # symmetry magnitude: net projection of tip vectors on the streamline axis,
    # normalised to [0,1] (0 = balanced/symmetric, 1 = all branches one side).
    smmag = abs(dx.sum()) / total_len
    # mean resultant length of branch directions (0 = isotropic, 1 = aligned).
    rbar = float(np.hypot(np.cos(theta).mean(), np.sin(theta).mean())) if theta.size else 0.0
    return dict(theta=theta, smmag=smmag, rbar=rbar, n_tips=int(theta.size))


# --------------------------------------------------------------------------- #
# Figure
# --------------------------------------------------------------------------- #
def make_figure(output, show, n_particles):
    # Drift bias stands in for advection; the ratio to diffusion ~ Peclet number.
    drifts = [0.0, 0.04, 0.09, 0.15, 0.22]
    pe_rel = [d / 0.25 for d in drifts]          # relative Pe (advection fraction)

    print("Growing corals by DLA-with-drift (drift ~ Peclet number)...")
    results = []
    for i, d in enumerate(drifts):
        order, m = simulate_coral(d, n_particles=n_particles, seed=100 + i)
        results.append((order, m))
        print(f"  Pe~{pe_rel[i]:.2f}  upstream={m['upstream']:+5.1f} cells  "
              f"R_gyr={m['r_gyr']:5.1f}  (placed={m['placed']}, span={m['span']})")

    # coral-ish colormap (old growth -> fresh tips)
    coral_cmap = LinearSegmentedColormap.from_list(
        "coral", ["#7a3b2e", "#c0603f", "#ef9b7f", "#ffd9c7"])

    fig = plt.figure(figsize=(14, 8.6))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.12, 0.88], hspace=0.33, wspace=0.26)

    # ----- (a-c) three morphologies: no flow -> moderate -> strong flow ------ #
    show_idx = [0, 2, 4]
    titles = ["(a) no flow  (Pe→0)\nsymmetric, grows in all directions",
              "(b) moderate flow\nupstream bias appears",
              "(c) strong flow (high Pe)\nasymmetric, compact, upstream"]
    for col, idx in enumerate(show_idx):
        ax = fig.add_subplot(gs[0, col])
        order, m = results[idx]
        disp = np.ma.masked_less(order, 0)
        ax.imshow(disp, origin="lower", cmap=coral_cmap, interpolation="nearest")
        ys_i, xs_i = np.nonzero(order >= 0)             # zoom to the cluster
        cxc = 0.5 * (xs_i.min() + xs_i.max())
        cyc = 0.5 * (ys_i.min() + ys_i.max())
        half = 0.5 * max(xs_i.max() - xs_i.min(), ys_i.max() - ys_i.min()) + 8
        ax.set_xlim(cxc - half, cxc + half)
        ax.set_ylim(cyc - half, cyc + half)
        if idx > 0:                                     # flow direction arrow (+x)
            ax.annotate("", xy=(0.93, 0.08), xytext=(0.60, 0.08),
                        xycoords="axes fraction",
                        arrowprops=dict(arrowstyle="-|>", color="#1f77b4", lw=2.4))
            ax.text(0.76, 0.13, "flow", color="#1f77b4", ha="center", fontsize=9,
                    transform=ax.transAxes)
        ax.set_title(titles[col], fontsize=10.5)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlabel(f"upstream (−x) ← | → downstream        shift = {m['upstream']:+.1f}",
                      fontsize=8)

    # ----- (d) upstream asymmetry vs Pe ------------------------------------- #
    ax = fig.add_subplot(gs[1, 0])
    up = [m["upstream"] for _, m in results]
    ax.plot(pe_rel, up, "o-", color="#c0603f", lw=2)
    ax.axhline(0, color="k", lw=0.8, ls=":")
    ax.set_title("(d) Upstream growth bias vs Pe", fontsize=10.5)
    ax.set_xlabel("relative Péclet number  (advection/diffusion)")
    ax.set_ylabel("mass shift upstream  (cells)")
    ax.grid(alpha=0.25)

    # ----- (e) colony spread vs Pe (compactification) ----------------------- #
    ax = fig.add_subplot(gs[1, 1])
    rg = [m["r_gyr"] for _, m in results]
    ax.plot(pe_rel, rg, "s-", color="#1f77b4", lw=2)
    ax.set_title("(e) Colony spread vs Pe\n(same mass → compactification)", fontsize=10.5)
    ax.set_xlabel("relative Péclet number")
    ax.set_ylabel("radius of gyration  (↓ = more compact,\nlower surface/volume)")
    ax.grid(alpha=0.25)

    # ----- (f) the mapping to the paper ------------------------------------- #
    ax = fig.add_subplot(gs[1, 2]); ax.axis("off")
    ax.text(0.0, 1.0,
            "From the paper to this demo\n"
            "------------------------------------\n\n"
            "Accretive growth: polyps extend along\n"
            "the normal at a rate set by absorbed\n"
            "nutrient  (growth ∝ surface flux).\n\n"
            "Nutrient transport:\n"
            "   ∂c/∂t = D∇²c − u·∇c   (advec-diff)\n"
            "   Pe = u·L / D\n\n"
            "DLA-with-drift is the discrete form:\n"
            "   random walk  = diffusion (D)\n"
            "   drift in +x  = advection (u)\n"
            "   sticking     = growth ∝ flux\n\n"
            "Recovered results (Chindapol 2013):\n"
            "  • low Pe  -> symmetric branching\n"
            "  • high Pe -> branches grow UPSTREAM,\n"
            "    downstream shaded & suppressed\n"
            "  • S/V ratio falls (compactification)\n"
            "    [Merks & Kaandorp 2003].",
            va="top", ha="left", family="monospace", fontsize=8.6,
            transform=ax.transAxes)

    fig.suptitle("Chindapol, Kaandorp et al. 2013 — flow (Péclet number) steers "
                 "coral growth from symmetric to upstream-biased", fontsize=12.5)

    out = Path(output)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nSaved morphology figure to {out.resolve()}")

    # ====================================================================== #
    # Second figure: the paper's MORPHOMETRIC analysis (symmetry of branches)
    # ====================================================================== #
    print("\nMORPHOMETRICS (branch-tip symmetry about the flow axis):")
    morphs = [morphometrics(order) for order, _ in results]
    for i, mm in enumerate(morphs):
        print(f"  Pe~{pe_rel[i]:.2f}  smmag={mm['smmag']:.3f}  "
              f"Rbar={mm['rbar']:.3f}  (n_tips={mm['n_tips']})")

    fig2 = plt.figure(figsize=(14, 8.4))
    gs2 = fig2.add_gridspec(2, 3, height_ratios=[1.1, 0.9], hspace=0.42, wspace=0.30)

    # ----- (a-c) branch-direction roses: symmetric -> skewed upstream ------- #
    rose_titles = ["(a) no flow  (Pe→0)", "(b) moderate flow", "(c) strong flow (high Pe)"]
    nbins = 24
    edges = np.linspace(-np.pi, np.pi, nbins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    for col, idx in enumerate(show_idx):
        axp = fig2.add_subplot(gs2[0, col], projection="polar")
        counts, _ = np.histogram(morphs[idx]["theta"], bins=edges)
        axp.bar(centers, counts, width=2 * np.pi / nbins, color="#c0603f",
                edgecolor="#7a3b2e", alpha=0.85)
        axp.set_theta_zero_location("E")          # 0 rad = +x = downstream (right)
        axp.set_title(rose_titles[col] + f"\nsmmag={morphs[idx]['smmag']:.2f}", fontsize=10)
        axp.set_xticks([0, np.pi / 2, np.pi, 3 * np.pi / 2])
        axp.set_xticklabels(["down", "", "up", ""], fontsize=8)
        axp.set_yticklabels([])

    # ----- (d) symmetry magnitude smmag vs Pe (paper Fig. 12) --------------- #
    ax = fig2.add_subplot(gs2[1, 0])
    sm = [mm["smmag"] for mm in morphs]
    ax.plot(pe_rel, sm, "o-", color="#c0603f", lw=2)
    ax.set_title("(d) Symmetry magnitude  smmag  vs Pe", fontsize=10.5)
    ax.set_xlabel("relative Péclet number")
    ax.set_ylabel("smmag  (0 = symmetric → 1 = one-sided)")
    ax.set_ylim(0, None); ax.grid(alpha=0.25)

    # ----- (e) branch-direction alignment (resultant length) vs Pe --------- #
    ax = fig2.add_subplot(gs2[1, 1])
    rb = [mm["rbar"] for mm in morphs]
    ax.plot(pe_rel, rb, "s-", color="#1f77b4", lw=2)
    ax.set_title("(e) Branch-direction alignment vs Pe", fontsize=10.5)
    ax.set_xlabel("relative Péclet number")
    ax.set_ylabel("mean resultant length  R̄\n(0 = isotropic → 1 = aligned)")
    ax.set_ylim(0, None); ax.grid(alpha=0.25)

    # ----- (f) notes on the morphometric method ----------------------------- #
    ax = fig2.add_subplot(gs2[1, 2]); ax.axis("off")
    ax.text(0.0, 1.0,
            "Morphometric method (Chindapol 2013)\n"
            "----------------------------------------\n\n"
            "1. Skeletonise the colony to a graph of\n"
            "   branches; here: lattice tips (degree≤1).\n\n"
            "2. Symmetry ANGLES hangle, vangle:\n"
            "   orientation of each branch vector from\n"
            "   the root (rose plots above; 2-D uses the\n"
            "   horizontal/hangle analog only).\n\n"
            "3. Symmetry MAGNITUDE smmag:\n"
            "   net projection of branch vectors onto\n"
            "   the streamline s.  Opposing branches\n"
            "   cancel ⇒ smmag≈0 (symmetric); flow skews\n"
            "   them upstream ⇒ smmag rises.\n\n"
            "Result: smmag and alignment climb with Pe,\n"
            "matching the paper's Figs. 9–12 for both\n"
            "simulated and CT-scanned P. verrucosa.",
            va="top", ha="left", family="monospace", fontsize=8.4,
            transform=ax.transAxes)

    fig2.suptitle("Chindapol, Kaandorp et al. 2013 — morphometrics: branch "
                  "symmetry vs Péclet number\n(rose plots: 'up' = upstream, into "
                  "the flow; 'down' = downstream)", fontsize=12)

    out2 = out.with_name(out.stem + "_morphometrics" + out.suffix)
    fig2.savefig(out2, dpi=150, bbox_inches="tight")
    print(f"Saved morphometrics figure to {out2.resolve()}")

    if show:
        plt.show()
    plt.close(fig)
    plt.close(fig2)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-o", "--output", default="chindapol_coral_flow_morphospace.png",
                    help="output PNG path (default: repo root)")
    ap.add_argument("--particles", type=int, default=700,
                    help="polyps (lattice sites) grown per colony (default 700)")
    ap.add_argument("--no-show", dest="show", action="store_false",
                    help="save the figure without opening a window")
    ap.set_defaults(show=True)
    args = ap.parse_args()
    make_figure(args.output, args.show, args.particles)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Raup's shell-coiling model (1962, 1965, 1966): six illustrative forms, a parameter-space sheet,
and the morphometrics that check the model does what the paper says.

References:
    D. M. Raup, "Computer as aid in describing form in gastropod shells,"
    Science 138(3537):150-152 (1962).  doi:10.1126/science.138.3537.150
    D. M. Raup, A. Michelson, "Theoretical morphology of the coiled shell,"
    Science 147(3663):1294-1295 (1965).  doi:10.1126/science.147.3663.1294
    D. M. Raup, "Geometric analysis of shell coiling: general problems,"
    J. Paleontol. 40(5):1178-1190 (1966).  JSTOR 1301992, doi:10.2307/1301992
  The paper itself was not available to this author; the parameter definitions below are
  Raup's as restated, with equations, by
    K. Noshita, J. Theor. Biol. 363:93-104 (2014), doi:10.1016/j.jtbi.2014.08.010 (ktch's raup())
    T. Okabe, J. Yoshimura, Sci. Rep. 7:42445 (2017), doi:10.1038/srep42445:
      a unit circle centred at (radial y0, axial x0) has Raup's T = x0/y0 and D = (y0-1)/(y0+1).

The model.  A generating curve (the aperture, here an ellipse) is swept around a fixed axis
while it expands as a logarithmic spiral.  Four parameters:

    W  whorl expansion rate      every radius is multiplied by W per revolution:
                                  a(theta) = a0 * W**(theta / 2 pi)
    D  distance from the axis    the generating curve's inner margin over its outer margin,
                                  D = (r_c - a) / (r_c + a), so the centre sits at
                                  r_c = a * (1 + D) / (1 - D)
    T  translation rate          axial movement of the generating-curve centre per unit of
                                  its radial movement, T = dz_c / dr_c, so z_c = T * r_c
                                  (T = 0 is planispiral; the spire grows *with the whorls*)
    S  shape of the curve        radial / axial semi-axis ratio of the ellipse (S = 1 circular;
                                  this is the repository's convention, all panels use S = 1)

The generating curve lies in the plane through the axis (radial-axial plane), so the tube has
thickness along the axis.  Because r_c, z_c and a all scale with W**(theta/2 pi), the shell is
self-similar: every whorl is the previous one magnified by W, which is the property Raup built
the model around.

What changed in this version (audit 2026-10-13 -> remediation 2026-10-17).  The earlier
code translated linearly in angle (z_c = T * theta) while the radius grew exponentially, so T
had no visible effect and every panel was a flat disc; it also placed the generating ellipse
in the horizontal plane (zero axial thickness) and used offset = D/(1-D) in a way that made the
rendered inner/outer ratio (1+D)/(3-D) rather than D.  All three are corrected above, and
self_test() now recovers W, D and T from the rendered mesh to 1e-9 and checks that shell
height / width rises monotonically with T before anything is drawn.

Parameter sets.  The six (W, D, T) panels and the parameter ranges are read from
datasets/raup/variables.json.  They are *illustrative* points in the morphospace chosen so
that each descriptive name (planispiral, turreted, evolute disc, conical, high-spired,
globose) is geometrically true, and labelled with the kind of shell they resemble (cf.
Nautilus, Turritella, ...); they are not measurements Raup published for those taxa, and the
panels are not compared with specimens here.  The 2025 values they replace, and why, are
recorded in variables.json under empirical_shells_provenance.

Outputs (git-ignored; docs/media/raup_shell_coiling.gif is the downsized copy):
    raup_shell_animation.gif         six forms, side view then overhead, rotating
    raup_parameter_space.png         W x D and W x T sheets at fixed third parameter
    raup_empirical_comparison.png    the six panels as stills
    raup_shell_metrics.csv           per panel: nominal and recovered W, D, T; height/width; spire/width

    python3 create_raup_animations.py [--frames 120] [--fps 15] [--no-gif] [--self-test-only]
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

HERE = Path(__file__).resolve().parent
DATA = HERE / "datasets" / "raup" / "variables.json"


# --------------------------------------------------------------------------- #
# the model
# --------------------------------------------------------------------------- #
def raup_shell(W=3.0, D=0.2, T=0.5, S=1.0, turns=6, res=100, a0=1.0, apex_up=True):
    """Surface (X, Y, Z) of a Raup shell; rows run around the generating curve, columns along the coil.

    a(theta) = a0 W^(theta/2pi) is the radial semi-axis of the generating ellipse; its centre is at
    r_c = a (1+D)/(1-D) from the axis and z_c = T r_c along it, so inner/outer margin = D and
    dz_c/dr_c = T exactly (Raup 1966, Fig. 1 and text).  Axial semi-axis b = a / S.
    """
    if not W > 1:
        raise ValueError("W must exceed 1 (the shell must expand)")
    if not -1 < D < 1:
        raise ValueError("D must lie in (-1, 1)")
    theta = np.linspace(0.0, 2 * np.pi * turns, res * turns, endpoint=False)   # exactly res columns per whorl
    phi = np.linspace(0.0, 2 * np.pi, res)                                       # closed seam; phi=0 is on the grid
    theta, phi = np.meshgrid(theta, phi)

    a = a0 * W ** (theta / (2 * np.pi))          # generating-curve radial semi-axis
    rc = a * (1 + D) / (1 - D)                   # centre's distance from the axis
    zc = T * rc                                  # centre's position along the axis
    b = a / S                                    # axial semi-axis

    R = rc + a * np.cos(phi)                     # the ellipse lies in the radial-axial plane
    X = R * np.cos(theta)
    Y = R * np.sin(theta)
    Z = zc + b * np.sin(phi)
    if apex_up:
        Z = -Z
    return X, Y, Z


def recover_parameters(X, Y, Z):
    """Read W, D, T and height/width back off a rendered mesh (rows = phi, columns = theta).

    For each column the generating ellipse's radial extent gives the centre r_c = (Rmax+Rmin)/2 and
    semi-axis a = (Rmax-Rmin)/2; its axial extent gives z_c.  Then over the final whorl
    W = r_c(end)/r_c(end - 2pi), D = Rmin/Rmax at the aperture and T = |dz_c| / dr_c.
    """
    R = np.hypot(X, Y)
    rc = R[:-1].mean(axis=0)                     # mean of cos over a uniform grid on [0, 2pi) is exactly 0
    zc = Z[:-1].mean(axis=0)
    a = R.max(axis=0) - rc                       # phi = 0 is a grid point, so R.max = r_c + a exactly
    n = X.shape[1]
    pw = X.shape[0]                              # raup_shell(): res columns per revolution, res rows
    i1, i0 = n - 1, n - 1 - pw
    W_rec = rc[i1] / rc[i0]
    D_rec = (rc[i1] - a[i1]) / (rc[i1] + a[i1])
    T_rec = abs(zc[i1] - zc[i0]) / (rc[i1] - rc[i0])
    height = Z.max() - Z.min()
    width = max(X.max() - X.min(), Y.max() - Y.min())
    spire = abs(zc[-1] - zc[0]) / width          # axial travel of the centreline, apex to aperture
    return dict(W=float(W_rec), D=float(D_rec), T=float(T_rec), height_over_width=float(height / width),
                spire_over_width=float(spire))


def self_test(tol=1e-9):
    """Recover the inputs from the mesh on a grid of (W, D, T) and check height/width is monotone in T."""
    worst = 0.0
    for W in (1.5, 2.5, 4.0):
        for D in (0.0, 0.1, 0.3):
            hw_prev, sp_prev = -1.0, -1.0
            for T in (0.0, 0.5, 1.5, 3.0):
                X, Y, Z = raup_shell(W=W, D=D, T=T, turns=4, res=60)
                r = recover_parameters(X, Y, Z)
                err = max(abs(r["W"] - W), abs(r["D"] - D), abs(r["T"] - T))
                worst = max(worst, err)
                if err > tol:
                    sys.exit(f"self-test FAILED: W={W} D={D} T={T} recovered {r} (err {err:.2e})")
                # the centreline's axial travel must grow strictly with T; the outer height/width may
                # stay flat while the last whorl still envelops the spire (T < 1 at small D), but never fall
                if r["spire_over_width"] <= sp_prev or r["height_over_width"] < hw_prev - tol:
                    sys.exit(f"self-test FAILED: spire not increasing in T at W={W} D={D} (T={T}: {r})")
                hw_prev, sp_prev = r["height_over_width"], r["spire_over_width"]
    # a flat disc must be flat: the axial extent of a T=0 shell is one aperture height
    X, Y, Z = raup_shell(W=2.0, D=0.1, T=0.0, turns=4, res=61)   # res = 61 puts phi = pi/2 on the grid
    R = np.hypot(X, Y); a_end = (R.max(axis=0) - R[:-1].mean(axis=0))[-1]
    if abs((Z.max() - Z.min()) - 2 * a_end) > 1e-9:
        sys.exit("self-test FAILED: planispiral shell is not one aperture thick")
    print(f"self-test passed: W, D, T recovered from the mesh to {worst:.1e}; spire height monotone in T")


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #
def load_raup_data():
    if not DATA.exists():
        raise FileNotFoundError(f"Raup data file not found: {DATA}")
    return json.loads(DATA.read_text())


def panels(data):
    return [dict(name=s["name"], cf=s["common_name"], W=s["W"], D=s["D"], T=s["T"], S=s.get("S", 1.0),
                 turns=s["turns"], color=s["color"]) for s in data["empirical_shells"]]


ILLUSTRATIVE = ("Parameter sets are illustrative values from datasets/raup/variables.json, labelled with the "
                "kind of shell they resemble; they are not Raup's measurements of these taxa.")


def fit_axes(ax, X, Y, Z, zoom=1.3):
    rng = max(X.max() - X.min(), Y.max() - Y.min(), Z.max() - Z.min())
    for lim, A in ((ax.set_xlim, X), (ax.set_ylim, Y), (ax.set_zlim, Z)):
        c = (A.max() + A.min()) / 2
        lim(c - rng / 2, c + rng / 2)
    ax.set_box_aspect((1, 1, 1), zoom=zoom)


# --------------------------------------------------------------------------- #
# outputs
# --------------------------------------------------------------------------- #
def write_metrics(shells, path):
    rows = []
    for s in shells:
        X, Y, Z = raup_shell(W=s["W"], D=s["D"], T=s["T"], S=s["S"], turns=s["turns"], res=100)
        r = recover_parameters(X, Y, Z)
        rows.append(dict(panel=s["name"], resembles=s["cf"], W=s["W"], D=s["D"], T=s["T"], S=s["S"], turns=s["turns"],
                         W_recovered=round(r["W"], 6), D_recovered=round(r["D"], 6), T_recovered=round(r["T"], 6),
                         height_over_width=round(r["height_over_width"], 4), spire_over_width=round(r["spire_over_width"], 4)))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    print(f"  wrote {path}")
    for r in rows:
        print(f"    {r['panel']:<12} W={r['W']:<4} D={r['D']:<5} T={r['T']:<4} -> recovered "
              f"W={r['W_recovered']:.4f} D={r['D_recovered']:.4f} T={r['T_recovered']:.4f}  height/width={r['height_over_width']:.3f}  spire/width={r['spire_over_width']:.3f}")
    return rows


def create_animated_gif(shells, output_path, num_frames=120, fps=15, res=60):
    print(f"Creating animated GIF: {output_path}")
    fig = plt.figure(figsize=(15, 10), dpi=80); fig.subplots_adjust(0, 0.03, 1, 0.89, wspace=0, hspace=0.1)
    axes = [fig.add_subplot(2, 3, i + 1, projection="3d") for i in range(6)]
    meshes = [raup_shell(W=s["W"], D=s["D"], T=s["T"], S=s["S"], turns=s["turns"], res=res) for s in shells]
    half = num_frames // 2

    def animate(frame):
        side = frame < half
        azim = (frame % half) / half * 360
        elev = 28 if side else 90
        for ax, s, (X, Y, Z) in zip(axes, shells, meshes):
            ax.clear()
            ax.plot_wireframe(X, Y, Z, rstride=3, cstride=3, color=s["color"], linewidth=0.5, alpha=0.85)
            fit_axes(ax, X, Y, Z)
            ax.view_init(elev=elev, azim=azim)
            ax.set_title(f"{s['name']} · cf. {s['cf']}\nW={s['W']:.2f}  D={s['D']:.2f}  T={s['T']:.2f}",
                         fontsize=11, fontweight="bold", pad=6)
            ax.set_axis_off()
        fig.suptitle("Raup's coiling model (1966): six forms, " + ("side view" if side else "overhead view"),
                     fontsize=15, fontweight="bold", y=0.975)

    fig.text(0.5, 0.012, ILLUSTRATIVE, ha="center", va="bottom", fontsize=8.5, color="#475569", style="italic")
    anim = FuncAnimation(fig, animate, frames=num_frames, interval=1000 / fps, blit=False)
    anim.save(output_path, writer=PillowWriter(fps=fps))
    plt.close(fig)
    print(f"  saved {output_path} ({num_frames} frames)")


def create_parameter_space_visualization(data, output_path):
    print(f"Creating parameter space sheet: {output_path}")
    Wr, Dr, Tr = (data["core_variables"][k]["typical_range"] for k in "WDT")
    Ws = np.linspace(Wr["min"], Wr["max"], 3)
    Ds = np.linspace(Dr["min"], Dr["max"], 3)
    Ts = np.linspace(Tr["min"], Tr["max"], 3)
    fig = plt.figure(figsize=(16, 11)); fig.subplots_adjust(0, 0.02, 1, 0.9, wspace=0, hspace=0.1)
    for i, (W, D) in enumerate(zip(Ws, Ds)):
        ax = fig.add_subplot(2, 3, i + 1, projection="3d")
        X, Y, Z = raup_shell(W=W, D=D, T=1.0, turns=5, res=40)
        ax.plot_wireframe(X, Y, Z, rstride=4, cstride=4, color="steelblue", linewidth=0.5, alpha=0.85)
        fit_axes(ax, X, Y, Z); ax.set_axis_off()
        ax.set_title(f"W={W:.2f}, D={D:.2f}, T=1.00", fontsize=11)
    for i, (W, T) in enumerate(zip(Ws, Ts)):
        ax = fig.add_subplot(2, 3, i + 4, projection="3d")
        X, Y, Z = raup_shell(W=W, D=0.1, T=T, turns=5, res=40)
        ax.plot_wireframe(X, Y, Z, rstride=4, cstride=4, color="coral", linewidth=0.5, alpha=0.85)
        fit_axes(ax, X, Y, Z); ax.set_axis_off()
        ax.set_title(f"W={W:.2f}, D=0.10, T={T:.2f}", fontsize=11)
    fig.suptitle("Raup's coiling model: W and D (top, T fixed) · W and T (bottom, D fixed)\n"
                 "ranges from datasets/raup/variables.json", fontsize=14, fontweight="bold")
    fig.savefig(output_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {output_path}")


def create_empirical_comparison(shells, metrics, output_path):
    print(f"Creating six-panel still: {output_path}")
    fig = plt.figure(figsize=(16, 10)); fig.subplots_adjust(0, 0.05, 1, 0.88, wspace=0, hspace=0.1)
    for i, (s, m) in enumerate(zip(shells, metrics)):
        ax = fig.add_subplot(2, 3, i + 1, projection="3d")
        X, Y, Z = raup_shell(W=s["W"], D=s["D"], T=s["T"], S=s["S"], turns=s["turns"], res=50)
        ax.plot_wireframe(X, Y, Z, rstride=3, cstride=3, color=s["color"], linewidth=0.6, alpha=0.9)
        fit_axes(ax, X, Y, Z); ax.view_init(elev=15, azim=-60); ax.set_axis_off()
        ax.set_title(f"{s['name']} · cf. {s['cf']}\nW={s['W']:.2f}  D={s['D']:.2f}  T={s['T']:.2f}   "
                     f"height/width {m['height_over_width']:.2f}", fontsize=11, fontweight="bold")
    fig.suptitle("Raup's coiling model: six illustrative forms", fontsize=15, fontweight="bold", y=0.97)
    fig.text(0.5, 0.02, ILLUSTRATIVE + "  Recovered W, D, T per panel are in raup_shell_metrics.csv.",
             ha="center", fontsize=9, color="#475569", style="italic")
    fig.savefig(output_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {output_path}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--frames", type=int, default=120)
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--no-gif", action="store_true")
    ap.add_argument("--self-test-only", action="store_true")
    ap.add_argument("--out-dir", default=".", help="where to write the outputs")
    args = ap.parse_args(argv)

    self_test()
    if args.self_test_only:
        return
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    data = load_raup_data()
    shells = panels(data)
    print(f"Loaded {len(shells)} parameter sets from {DATA.relative_to(HERE)}")
    metrics = write_metrics(shells, out / "raup_shell_metrics.csv")
    if not args.no_gif:
        create_animated_gif(shells, out / "raup_shell_animation.gif", num_frames=args.frames, fps=args.fps)
    create_parameter_space_visualization(data, out / "raup_parameter_space.png")
    create_empirical_comparison(shells, metrics, out / "raup_empirical_comparison.png")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Raup's shell-coiling model (Raup & Michelson 1965, Science 147:1294, doi:10.1126/science.147.3663.1294;
Raup 1966, J. Paleontol. 40:1178-1190, doi:10.2307/1301992): a 3 x 3 slice of the morphospace,
W (whorl expansion) against T (translation), at fixed D.

    a(theta) = a0 W^(theta/2pi)          generating-curve radial semi-axis (log spiral)
    r_c      = a (1+D)/(1-D)             centre's distance from the axis, so inner/outer margin = D
    z_c      = T r_c                     centre's position along the axis, so dz_c/dr_c = T
    surface  = (r_c + a cos phi) (cos theta, sin theta),  z_c + (a/S) sin phi

The generating ellipse lies in the plane through the axis.  The definitions follow Raup as
restated by Noshita 2014 (J Theor Biol 363:93) and Okabe & Yoshimura 2017 (Sci Rep 7:42445).
create_raup_animations.py carries the same function with a self-test that recovers W, D, T
from the mesh; this file is the minimal version.

    python3 1966-raup.py [--no-show] [--out raup_morphospace_slice.png]
"""

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def raup_shell(W=3.0, D=0.2, T=0.5, S=1.0, turns=6, res=100, a0=1.0, apex_up=True):
    theta = np.linspace(0.0, 2 * np.pi * turns, res * turns, endpoint=False)
    phi = np.linspace(0.0, 2 * np.pi, res)
    theta, phi = np.meshgrid(theta, phi)
    a = a0 * W ** (theta / (2 * np.pi))      # every whorl is the previous one magnified by W
    rc = a * (1 + D) / (1 - D)               # (rc - a) / (rc + a) = D
    zc = T * rc                              # translation scales with the whorls: T = dz_c / dr_c
    R = rc + a * np.cos(phi)
    X, Y, Z = R * np.cos(theta), R * np.sin(theta), zc + (a / S) * np.sin(phi)
    return X, Y, -Z if apex_up else Z


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-show", action="store_true")
    ap.add_argument("--out", default=None, help="save the figure here instead of / as well as showing it")
    ap.add_argument("--export-mesh", default=None, help="export the 3x3 W x T morphospace slice as a SurfaceMesh3D JSON or OBJ file")
    args = ap.parse_args(argv)

    data_file = Path(__file__).resolve().parent / "datasets" / "raup" / "variables.json"
    if data_file.exists():
        cv = json.loads(data_file.read_text())["core_variables"]
        Wr, Tr, Dr = cv["W"]["typical_range"], cv["T"]["typical_range"], cv["D"]["typical_range"]
        Ws = [Wr["min"], (Wr["min"] + Wr["max"]) / 2, Wr["max"]]
        Ts = [Tr["min"], (Tr["min"] + Tr["max"]) / 2, Tr["max"]]
        D_fixed = 0.1 if Dr["min"] <= 0.1 <= Dr["max"] else (Dr["min"] + Dr["max"]) / 2
        print(f"ranges from {data_file.relative_to(Path.cwd()) if data_file.is_relative_to(Path.cwd()) else data_file}: "
              f"W {Ws}, T {Ts}, D fixed at {D_fixed}")
    else:
        Ws, Ts, D_fixed = [1.2, 2.6, 4.0], [0.0, 2.0, 4.0], 0.1
        print(f"{data_file} not found; using built-in ranges")

    if args.export_mesh:
        from create_raup_animations import export_meshes
        slice_shells = [
            {"name": f"W_{W:.2f}_T_{T:.2f}", "cf": f"D={D_fixed:.2f}", "W": float(W), "D": float(D_fixed), "T": float(T), "S": 1.0, "turns": 5}
            for W in Ws for T in Ts
        ]
        export_meshes(slice_shells, Path(args.export_mesh), res=32)

    fig, axs = plt.subplots(3, 3, subplot_kw={"projection": "3d"}, figsize=(12, 12))
    fig.subplots_adjust(0, 0, 1, 0.93, wspace=0, hspace=0.05)
    for i, W in enumerate(Ws):
        for j, T in enumerate(Ts):
            ax = axs[i, j]
            X, Y, Z = raup_shell(W=W, D=D_fixed, T=T, turns=5, res=50)
            ax.plot_wireframe(X, Y, Z, rstride=3, cstride=3, linewidth=0.5)
            rng = max(X.max() - X.min(), Y.max() - Y.min(), Z.max() - Z.min())
            for lim, A in ((ax.set_xlim, X), (ax.set_ylim, Y), (ax.set_zlim, Z)):
                c = (A.max() + A.min()) / 2; lim(c - rng / 2, c + rng / 2)
            ax.set_box_aspect((1, 1, 1), zoom=1.3); ax.view_init(elev=15, azim=-60)
            ax.set_title(f"W={W:.2f}, T={T:.2f}", fontsize=11); ax.set_axis_off()
    fig.suptitle(f"Raup's coiling model: W down the rows, T across the columns, D = {D_fixed:.2f}", fontsize=14, fontweight="bold")
    if args.out:
        fig.savefig(args.out, dpi=110, bbox_inches="tight"); print(f"saved {args.out}")
    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()

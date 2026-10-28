#!/usr/bin/env python3
"""
Generate the Referee-Audited Animated Arthropod & Crab Carapace GIF
(`docs/media/thompson_1917_arthropods_and_ant_castes_summary.gif`):
  - Panel A: Ant Soldier Head Caste Allometry (Minor -> Major, 82 LMs, Camponotus)
             Watch the 8p Thompson Bilobed-Fan / Fig. 359 Radial-Cordate grid flare
             the posterior occipital lobes and indent the vertex notch (LOO-CV: 8.2% Box
             -> 61.6% Bilobed-Fan, nested F = 54.2, 42/42 diffeomorphic).
  - Panel B: 3D Spider Crab Carapace (Geryon -> Chorinus / Leptomithrax, Fig. 369f)
             Watch the anteriorly converging f1 < 0 Thompson grid and 14-LM setup
             capture the pyriform spider-crab carapace (IoU: 0.810 TPS-8 -> 0.883
             Thompson-14 / 0.912 TPS-14; Displacement R²: 55.1% Box -> 90.5% Thompson).
  - Panel C: Convergently Carcinized Anomuran King Crab (Geryon -> Paralomis / Lithodes, Fig. 369d)
             Watch the exact same anteriorly converging f1 < 0 fan family transform
             Geryon into Lithodes maja (Displacement R²: 13.8% Box -> 73.9% Thompson;
             IoU: 0.735 Box -> 0.862 Thompson / 0.889 TPS-14).
  - Panel D: Copepods (Oithona nana -> Sapphirina, Figs. 367–368, 42 Independent Grid Nodes)
             Watch Thompson's p. 743 logarithmic-quadratic grid deform the slender
             cyclopoid into the foliaceous Sapphirina (LOO-CV: 58.6% Box -> 92.5% Thompson).
  - Panel E: Live Warp Progression Curve R²(alpha) & IoU(alpha) as alpha: 0 -> 1
             comparing 4p Box Stretch vs. 8p Thompson Coordinate Grids across all clades.
"""

import importlib.util
import io
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_GIF = REPO_ROOT / "docs" / "media" / "thompson_1917_arthropods_and_ant_castes_summary.gif"


def load_module(name: str, rel_path: str):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / rel_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def close_c(pts: np.ndarray) -> np.ndarray:
    return np.vstack([pts, pts[0]])


def blend(P0: np.ndarray, P1: np.ndarray, alpha: float) -> np.ndarray:
    return (1.0 - alpha) * P0 + alpha * P1


def main():
    ant_mod = load_module("ant_mod", "thompson/run_antscan_caste_allometry_tests.py")
    crab_mod = load_module("crab_mod", "thompson/run_crab_carapace_paradox_audit.py")
    arth_mod = load_module("arth_mod", "thompson/run_all_1917_arthropods_audit.py")

    print("Loading referee-audited Ant Caste, 3D Crab, and Copepod datasets...")
    camp_res, ex_head = ant_mod.run_camponotus_634_analyses()
    S_ant, T_ant = ex_head["S"], ex_head["T"]
    P_box_ant, P_bf_ant = ex_head["P_box"], ex_head["P_bf"]
    mx_ant, my_ant = ex_head["mx"], ex_head["my"]

    arr2 = np.array(Image.open(crab_mod.CRAB_DIR / "scholtz_2020_fig2_real_3d_crabs.png").convert("RGB"))
    real_crabs = crab_mod.extract_crab_dataset(arr2, int(arr2.shape[0] * 0.24), int(arr2.shape[0] * 0.47), use_real_lms=True)
    S_ger = real_crabs["a_Geryon"]
    fit_chor = crab_mod.fit_crab_models(S_ger, real_crabs["f_Chorinus_Leptomithrax"])
    fit_lith = crab_mod.fit_crab_models(S_ger, real_crabs["d_Paralomis_Lithodes"])

    cop = arth_mod.test_copepods_fig367_368()

    # Build smooth alpha sequence (0 -> 1 -> hold -> 0)
    t_up = np.linspace(0.0, 1.0, 14)
    alphas = np.concatenate([
        [0.0, 0.0],
        0.5 - 0.5 * np.cos(np.pi * t_up),
        [1.0, 1.0, 1.0, 1.0],
        0.5 - 0.5 * np.cos(np.pi * t_up[::-1]),
    ])

    frames = []
    print(f"Rendering {len(alphas)} animation frames...")
    for idx_f, alpha in enumerate(alphas):
        fig = plt.figure(figsize=(15.5, 9.4), dpi=110, facecolor="#070b14")
        gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.34, wspace=0.24)

        def style_ax(ax, title_str, sub_str, sub_col="#38bdf8"):
            ax.set_facecolor("#0f172a")
            for sp in ax.spines.values():
                sp.set_edgecolor("#263859")
                sp.set_linewidth(1.2)
            ax.tick_params(colors="#64748b", labelsize=7.5)
            ax.set_title(title_str, fontsize=9.8, fontweight="bold", color="#f8fafc", pad=16)
            ax.text(
                0.5, 1.02, sub_str, transform=ax.transAxes, ha="center", va="bottom",
                fontsize=8.0, fontweight="bold", color=sub_col,
                bbox=dict(boxstyle="round,pad=0.18", facecolor="#131e36", edgecolor=sub_col, lw=0.9),
            )

        # Panel A: Ant Soldier Head (Camponotus, 82 LMs)
        ax_a = fig.add_subplot(gs[0, 0])
        u_min, u_max = S_ant[:, 0].min() * 1.12, S_ant[:, 0].max() * 1.12
        v_min, v_max = S_ant[:, 1].min() * 1.12, S_ant[:, 1].max() * 1.12
        for u_val in np.linspace(u_min, u_max, 7):
            vv = np.linspace(v_min, v_max, 40)
            uu = np.full_like(vv, u_val)
            gx1 = mx_ant[0] * uu + mx_ant[1] * vv + mx_ant[2] * (uu * vv) + mx_ant[3]
            gy1 = my_ant[0] * uu + my_ant[1] * vv + my_ant[2] * (uu ** 2) + my_ant[3]
            ax_a.plot(blend(uu, gx1, alpha), blend(vv, gy1, alpha), color="#38bdf8", lw=0.75, alpha=0.45)
        for v_val in np.linspace(v_min, v_max, 7):
            uu = np.linspace(u_min, u_max, 40)
            vv = np.full_like(uu, v_val)
            gx1 = mx_ant[0] * uu + mx_ant[1] * vv + mx_ant[2] * (uu * vv) + mx_ant[3]
            gy1 = my_ant[0] * uu + my_ant[1] * vv + my_ant[2] * (uu ** 2) + my_ant[3]
            ax_a.plot(blend(uu, gx1, alpha), blend(vv, gy1, alpha), color="#38bdf8", lw=0.75, alpha=0.45)

        cur_box_a = blend(S_ant, P_box_ant, alpha)
        cur_th_a = blend(S_ant, P_bf_ant, alpha)
        ax_a.plot(close_c(T_ant)[:, 0], close_c(T_ant)[:, 1], color="#f43f5e", ls="--", lw=2.0, label="Major Soldier Target")
        ax_a.plot(close_c(cur_box_a)[:, 0], close_c(cur_box_a)[:, 1], color="#f97316", ls=":", lw=1.7, label="4p Box (LOO 8.2%)")
        ax_a.plot(close_c(cur_th_a)[:, 0], close_c(cur_th_a)[:, 1], color="#22c55e", lw=2.3, label="8p Bilobed-Fan (LOO 61.6%)")
        ax_a.set_aspect("equal")
        ax_a.set_xlim(-0.18, 0.18)
        ax_a.set_ylim(-0.18, 0.16)
        ax_a.legend(loc="lower center", fontsize=6.8, facecolor="#0f172a", edgecolor="#263859", labelcolor="#f8fafc")
        style_ax(
            ax_a,
            "1. Ant Soldier Caste (Minor → Major Head, 82 LMs)",
            f"Warp α = {alpha*100:3.0f}% · Nested F = 54.2 (42/42 Diffeomorphic)",
            "#22c55e",
        )

        # Panel B: 3D Spider Crab (Geryon -> Chorinus / Leptomithrax, Fig. 369f)
        ax_b = fig.add_subplot(gs[0, 1])
        BS_ger = S_ger["contour"]
        mx_c, my_c = fit_chor["mx14"], fit_chor["my14"]
        S14_c, T14_c = fit_chor["S14"], fit_chor["T14"]
        for u_val in np.linspace(S14_c[:, 0].min(), S14_c[:, 0].max(), 7):
            vv = np.linspace(S14_c[:, 1].min(), S14_c[:, 1].max(), 40)
            uu = np.full_like(vv, u_val)
            gx1 = mx_c[0] * uu + mx_c[1] * uu * vv + mx_c[2] * uu * (vv ** 2) + mx_c[3]
            gy1 = my_c[0] * vv + my_c[1] * (vv ** 2) + my_c[2] * (uu ** 2) + my_c[3]
            ax_b.plot(blend(uu, gx1, alpha), blend(vv, gy1, alpha), color="#38bdf8", lw=0.75, alpha=0.45)
        cur_tps8_b = blend(BS_ger, fit_chor["B_8lm"], alpha)
        cur_th_b = blend(BS_ger, fit_chor["B_th"], alpha)
        ax_b.plot(close_c(fit_chor["BT"])[:, 0], close_c(fit_chor["BT"])[:, 1], color="#f43f5e", ls="--", lw=2.0, label="3D Scanned Leptomithrax")
        ax_b.plot(close_c(cur_tps8_b)[:, 0], close_c(cur_tps8_b)[:, 1], color="#f97316", ls=":", lw=1.7, label="8-LM TPS (0 Lat. LMs: IoU 0.810)")
        ax_b.plot(close_c(cur_th_b)[:, 0], close_c(cur_th_b)[:, 1], color="#22c55e", lw=2.3, label="14-LM Thompson (IoU 0.883, R² 90.5%)")
        ax_b.scatter(T14_c[:8, 0], T14_c[:8, 1], color="#eab308", s=24, zorder=5)
        ax_b.scatter(T14_c[8:, 0], T14_c[8:, 1], color="#22c55e", marker="^", s=32, zorder=6)
        ax_b.set_aspect("equal")
        ax_b.set_xlim(-0.48, 0.48)
        ax_b.set_ylim(-0.48, 0.48)
        ax_b.legend(loc="lower center", fontsize=6.6, facecolor="#0f172a", edgecolor="#263859", labelcolor="#f8fafc")
        style_ax(
            ax_b,
            "2. 3D Spider Crab: Geryon → Leptomithrax (Fig. 369f)",
            f"Scholtz Confirmed: f₁ = {fit_chor['fan_parameter_f1']:+.2f} < 0 · TPS-14 IoU = 0.912",
            "#38bdf8",
        )

        # Panel C: Convergent King Crab (Geryon -> Paralomis / Lithodes maja, Fig. 369d)
        ax_c = fig.add_subplot(gs[0, 2])
        mx_l, my_l = fit_lith["mx14"], fit_lith["my14"]
        S14_l, T14_l = fit_lith["S14"], fit_lith["T14"]
        for u_val in np.linspace(S14_l[:, 0].min(), S14_l[:, 0].max(), 7):
            vv = np.linspace(S14_l[:, 1].min(), S14_l[:, 1].max(), 40)
            uu = np.full_like(vv, u_val)
            gx1 = mx_l[0] * uu + mx_l[1] * uu * vv + mx_l[2] * uu * (vv ** 2) + mx_l[3]
            gy1 = my_l[0] * vv + my_l[1] * (vv ** 2) + my_l[2] * (uu ** 2) + my_l[3]
            ax_c.plot(blend(uu, gx1, alpha), blend(vv, gy1, alpha), color="#38bdf8", lw=0.75, alpha=0.45)
        cur_box_c = blend(BS_ger, fit_lith["B_box"], alpha)
        cur_th_c = blend(BS_ger, fit_lith["B_th"], alpha)
        ax_c.plot(close_c(fit_lith["BT"])[:, 0], close_c(fit_lith["BT"])[:, 1], color="#f43f5e", ls="--", lw=2.0, label="3D Scanned Lithodes maja")
        ax_c.plot(close_c(cur_box_c)[:, 0], close_c(cur_box_c)[:, 1], color="#f97316", ls=":", lw=1.7, label="4p Box (Disp. R² 13.8%)")
        ax_c.plot(close_c(cur_th_c)[:, 0], close_c(cur_th_c)[:, 1], color="#22c55e", lw=2.3, label="14-LM Thompson (Disp. R² 73.9%)")
        ax_c.scatter(T14_l[:8, 0], T14_l[:8, 1], color="#eab308", s=24, zorder=5)
        ax_c.scatter(T14_l[8:, 0], T14_l[8:, 1], color="#22c55e", marker="^", s=32, zorder=6)
        ax_c.set_aspect("equal")
        ax_c.set_xlim(-0.48, 0.48)
        ax_c.set_ylim(-0.48, 0.48)
        ax_c.legend(loc="lower center", fontsize=6.6, facecolor="#0f172a", edgecolor="#263859", labelcolor="#f8fafc")
        style_ax(
            ax_c,
            "3. Carcinized King Crab: Geryon → Lithodes (Fig. 369d)",
            f"Same Anteriorly Converging Family f₁ = {fit_lith['fan_parameter_f1']:+.2f} < 0!",
            "#f59e0b",
        )

        # Panel D: Copepods (Oithona nana -> Sapphirina, Figs. 367–368, 42 Grid Nodes)
        ax_d = fig.add_subplot(gs[1, 0])
        cur_box_d = blend(cop["C_oith"], cop["C_box"], alpha)
        cur_th_d = blend(cop["C_oith"], cop["C_th"], alpha)
        cur_pts_d = blend(cop["S0"], cop["P_th"], alpha)
        ax_d.plot(close_c(cop["C_sapp"])[:, 0], close_c(cop["C_sapp"])[:, 1], color="#f43f5e", ls="--", lw=2.0, label="Sapphirina Target (p. 743 Table)")
        ax_d.plot(close_c(cur_box_d)[:, 0], close_c(cur_box_d)[:, 1], color="#f97316", ls=":", lw=1.7, label="4p Box (LOO 58.6%)")
        ax_d.plot(close_c(cur_th_d)[:, 0], close_c(cur_th_d)[:, 1], color="#22c55e", lw=2.3, label="6p Log-Quad (LOO 92.5%)")
        ax_d.scatter(cur_pts_d[:, 0], cur_pts_d[:, 1], color="#38bdf8", s=16, alpha=0.85, label="42 Independent Grid Nodes")
        ax_d.set_aspect("equal")
        ax_d.set_xlim(-0.34, 0.34)
        ax_d.set_ylim(-0.36, 0.36)
        ax_d.legend(loc="lower center", fontsize=6.6, facecolor="#0f172a", edgecolor="#263859", labelcolor="#f8fafc")
        style_ax(
            ax_d,
            "4. Copepods: Oithona → Sapphirina (Figs. 367–368)",
            "42 Right-Half Table Nodes: LOO-CV 58.6% → 92.5% (+33.9 pts)",
            "#22c55e",
        )

        # Panel E: Live Warp Trajectory Curve R²(alpha) across Clades
        ax_e = fig.add_subplot(gs[1, 1:])
        a_grid = np.linspace(0.0, 1.0, 50)
        # Displacement R²(alpha) = 1 - ||T - ((1-alpha)*S + alpha*P)||^2 / ||T - S||^2
        def r2_curve(S_pts, T_pts, P_pts):
            ss0 = float(np.sum((T_pts - S_pts) ** 2))
            vals = []
            for a_v in a_grid:
                cur = (1.0 - a_v) * S_pts + a_v * P_pts
                vals.append((1.0 - float(np.sum((T_pts - cur) ** 2)) / ss0) * 100.0)
            return np.array(vals)

        c_ant_th = r2_curve(S_ant, T_ant, P_bf_ant)
        c_ant_box = r2_curve(S_ant, T_ant, P_box_ant)
        c_cop_th = r2_curve(cop["S0"], cop["T0"], cop["P_th"])
        c_crab_th = r2_curve(fit_chor["S14"], fit_chor["T14"], crab_mod.fit_8p_thompson_carapace(fit_chor["S14"], fit_chor["T14"])[0](fit_chor["S14"]))

        ax_e.plot(a_grid * 100, c_cop_th, color="#38bdf8", lw=2.4, label="Copepods 6p Log-Quad Grid (Figs. 367–368: → 93.5% In-Sample / 92.5% LOO)")
        ax_e.plot(a_grid * 100, c_crab_th, color="#a855f7", lw=2.4, label="Spider Crab Leptomithrax 14-LM Thompson Fan (Fig. 369f: → 90.5%, f₁ = −1.14 < 0)")
        ax_e.plot(a_grid * 100, c_ant_th, color="#22c55e", lw=2.6, label="Ant Soldier Head 8p Bilobed-Fan (C. punctulatus: → 79.6%, 42-spp. LOO 61.6%)")
        ax_e.plot(a_grid * 100, c_ant_box, color="#f97316", ls="--", lw=2.0, label="Ant Soldier Head 4p Rectangular Box Stretch (→ 3.0%, 42-spp. LOO 8.2%)")

        ax_e.axvline(alpha * 100, color="#f8fafc", ls=":", lw=1.8)
        ax_e.scatter(
            [alpha * 100] * 4,
            [
                np.interp(alpha, a_grid, c_cop_th),
                np.interp(alpha, a_grid, c_crab_th),
                np.interp(alpha, a_grid, c_ant_th),
                np.interp(alpha, a_grid, c_ant_box),
            ],
            color=["#38bdf8", "#a855f7", "#22c55e", "#f97316"],
            edgecolors="#fff",
            s=55,
            zorder=6,
        )
        ax_e.set_xlim(0, 100)
        ax_e.set_ylim(-5, 105)
        ax_e.set_xlabel("Coordinate Warp Progression α (%)", fontsize=9.5, fontweight="bold", color="#f8fafc")
        ax_e.set_ylabel("Displacement Explained R²(α) (%)", fontsize=9.5, fontweight="bold", color="#f8fafc")
        ax_e.legend(loc="upper left", fontsize=8.0, facecolor="#0f172a", edgecolor="#263859", labelcolor="#f8fafc")
        ax_e.grid(True, linestyle="--", alpha=0.25, color="#64748b")
        style_ax(
            ax_e,
            "5. Live Coordinate Warp Progression α ∈ [0%, 100%] Across Arthropods & Ant Castes",
            f"Current Warp α = {alpha*100:3.0f}% · Blinded Fable 5.1 Referee-Audited Benchmarks",
            "#a855f7",
        )

        fig.suptitle(
            "D'Arcy Thompson's Arthropod & Ant Caste Transformations (Figs. 359, 367–370 · Scholtz et al. 2020 3D Crabs · Antscan 2026)",
            fontsize=12.6,
            fontweight="bold",
            color="#f8fafc",
            y=0.98,
        )

        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)
        buf.seek(0)
        frames.append(Image.open(buf).convert("P", palette=Image.ADAPTIVE, colors=128))

    OUT_GIF.parent.mkdir(parents=True, exist_ok=True)
    durations = [110] * len(frames)
    durations[0] = 450
    durations[1] = 450
    durations[16] = 600
    durations[17] = 600
    frames[0].save(
        OUT_GIF,
        save_all=True,
        append_images=frames[1:],
        duration=durations,
        loop=0,
        optimize=True,
    )
    print(f"Saved animated GIF ({OUT_GIF.stat().st_size / 1024:.1f} KB, {len(frames)} frames) to {OUT_GIF}")


if __name__ == "__main__":
    main()

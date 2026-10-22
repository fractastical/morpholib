#!/usr/bin/env python3
"""
Generate an animated summary GIF (`docs/media/thompson_1917_silhouette_paradox_summary.gif`)
that visually tells the complete story of the paper as warp parameter alpha in [0, 1]
deforms source fish silhouettes into target fish silhouettes:
  - Panel A: 17-Landmark Fit on Scarus -> Pomacanthus (8p Coaxial)
             Watch the 9/17 head landmarks steer the posterior grid the wrong way,
             WORSENING |A Δ B| from 40.6% -> 56.2% (47.2% blue overhang).
  - Panel B: Direct 600-Point Silhouette Fit on Scarus -> Pomacanthus (Same 8p Coaxial)
             Watch the same 8p coaxial family fitted to the true 600-pt contour
             slash |A Δ B| in half to 28.5% (IoU = 0.749).
  - Panel C: Direct 600-Point Fan Fit on Pristigenys -> Scorpaena & Scorpaena -> Antigonia
             Slashing |A Δ B| from 120.1% -> 35.5% (3.4x reduction).
  - Panel D: Live Area-of-Difference |A Δ B|(alpha) curve as alpha: 0 -> 1, showing the
             diverging trajectories of 17-Landmark fitting (Red, going UP/worse) vs.
             Direct 600-pt Silhouette fitting (Green, going DOWN/better).
  - Panel E: The ~45-50 Myr Phylogenetic Cliff across 4,334 Dated Reef-Fish Pairs
             (791 species, Rabosky et al. 2018) & 1.48x Fin-vs.-Head Decoupling.
"""

import importlib.util
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image
import io

REPO_ROOT = Path(__file__).resolve().parents[1]
EXACT_JSON = REPO_ROOT / "datasets" / "thompson" / "thompson_1917_exact_species_landmarks.json"
OUT_GIF = REPO_ROOT / "docs" / "media" / "thompson_1917_silhouette_paradox_summary.gif"


def load_module(name, rel_path):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / rel_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


sil_mod = load_module("sil_mod", "thompson/1917-thompson-silhouette-area-paradox.py")


def blend_pts(P0, P1, alpha):
    return (1.0 - alpha) * P0 + alpha * P1


def render_shaded_panel(ax, src_t, tgt_sil, src_lm_t, tgt_lm, grid_lines_t, title, badge_text, badge_color, extent):
    xmin, xmax, ymin, ymax = extent

    # Recompute raster on fixed extent so camera frame never jumps during animation
    xs = np.linspace(xmin, xmax, 260)
    ys = np.linspace(ymin, ymax, 260)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.column_stack([XX.ravel(), YY.ravel()])
    from matplotlib.path import Path as MplPath
    ra = MplPath(src_t).contains_points(pts).reshape(260, 260)
    rb = MplPath(tgt_sil).contains_points(pts).reshape(260, 260)

    img = np.ones((260, 260, 4), dtype=float)
    img[:, :] = [0.985, 0.988, 0.995, 1.0]
    img[ra & rb] = [0.16, 0.66, 0.38, 0.80]       # Emerald Green: shared overlap
    img[rb & (~ra)] = [0.86, 0.18, 0.22, 0.84]    # Crimson Red: missing target
    img[ra & (~rb)] = [0.15, 0.42, 0.85, 0.82]    # Royal Blue: excess overhang

    ax.imshow(img, extent=[xmin, xmax, ymin, ymax], origin="lower", aspect="equal", zorder=1)

    for wline in grid_lines_t:
        ax.plot(wline[:, 0], wline[:, 1], color="#334155", lw=0.65, alpha=0.45, zorder=2)

    ax.plot(tgt_sil[:, 0], tgt_sil[:, 1], color="#7f1d1d", lw=1.6, ls="--", zorder=4)
    ax.plot(src_t[:, 0], src_t[:, 1], color="#0f172a", lw=1.6, zorder=5)

    trunk_idx = [1, 2, 3, 4, 5, 6, 7, 0, 1]
    if tgt_lm is not None:
        ax.plot(tgt_lm[trunk_idx, 0], tgt_lm[trunk_idx, 1], color="#7f1d1d", lw=1.0, ls=":", alpha=0.7, zorder=6)
    if src_lm_t is not None:
        ax.scatter(src_lm_t[:, 0], src_lm_t[:, 1], color="#0f172a", s=16, zorder=7)
        # Highlight the 9 head landmarks in gold/orange to show lever-arm concentration
        head_idx = [0, 1, 10, 11, 12, 13, 14, 15, 16]
        ax.scatter(src_lm_t[head_idx, 0], src_lm_t[head_idx, 1], color="#f59e0b", edgecolor="#0f172a", lw=0.6, s=26, zorder=8)

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=9.8, fontweight="bold", color="#0f172a", pad=18)
    ax.text(
        0.5, 1.018, badge_text,
        transform=ax.transAxes, ha="center", va="bottom",
        fontsize=8.3, color=badge_color, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.20", facecolor="#f8fafc", edgecolor=badge_color, lw=0.9)
    )


def main():
    at, sp_mod = sil_mod.load_modules()
    names, families, X = at.load_landmarks()
    Y = at.gpa(X)
    mean_consensus = at.centre_scale(Y.mean(0))
    meta = json.loads(EXACT_JSON.read_text(encoding="utf-8"))
    specs = {s: sil_mod.extract_aligned_silhouette_and_lms(s, meta, mean_consensus, sp_mod) for s in meta["specimens"]}
    O_sp = sp_mod.outline_analysis(Y, names, allow_download=False)
    specs["Scarus_ghobban"] = (O_sp["photos"]["Scarus_ghobban"]["landmarks"], O_sp["photos"]["Scarus_ghobban"]["boundary"])
    specs["Pomacanthus_semicirculatus"] = (
        O_sp["photos"]["Pomacanthus_semicirculatus"]["landmarks"],
        O_sp["photos"]["Pomacanthus_semicirculatus"]["boundary"],
    )

    # Prepare Scarus -> Pomacanthus (8p Coaxial)
    S_sc, BS_sc = specs["Scarus_ghobban"]
    T_po, BT_po = specs["Pomacanthus_semicirculatus"]
    BS_sc_unw = sil_mod.unwarped_similarity_silhouette(S_sc, BS_sc, T_po, at, sp_mod)
    S_sc_unw = at.fit_similarity(S_sc, T_po)

    p_sc_lm, S_sc_lm1 = at.fit_nonlinear_single("coaxial", S_sc, T_po)
    BS_sc_lm1 = at.FAMILY_FUNCS["coaxial"](p_sc_lm, BS_sc)
    fn_sc_lm = lambda P: at.FAMILY_FUNCS["coaxial"](p_sc_lm, P)

    p_sc_sil, BS_sc_sil1, apply_sc = sil_mod.fit_direct_to_silhouette("coaxial", S_sc, BS_sc, T_po, BT_po, at)
    S_sc_sil1 = apply_sc(p_sc_sil, S_sc)
    fn_sc_sil = lambda P: apply_sc(p_sc_sil, P)

    # Prepare Pristigenys -> Scorpaena (8p Fan)
    S_pr, BS_pr = specs["Pristigenys_alta"]
    T_sr, BT_sr = specs["Scorpaena_scrofa"]
    BS_pr_unw = sil_mod.unwarped_similarity_silhouette(S_pr, BS_pr, T_sr, at, sp_mod)
    S_pr_unw = at.fit_similarity(S_pr, T_sr)
    p_pr_sil, BS_pr_sil1, apply_pr = sil_mod.fit_direct_to_silhouette("fan", S_pr, BS_pr, T_sr, BT_sr, at)
    S_pr_sil1 = apply_pr(p_pr_sil, S_pr)
    fn_pr_sil = lambda P: apply_pr(p_pr_sil, P)

    # Precompute fixed extents and base grid lines
    all_sc = np.vstack([BS_sc_unw, BS_sc_lm1, BS_sc_sil1, BT_po])
    ext_sc = (all_sc[:, 0].min() - 0.04, all_sc[:, 0].max() + 0.04, all_sc[:, 1].min() - 0.04, all_sc[:, 1].max() + 0.04)

    all_pr = np.vstack([BS_pr_unw, BS_pr_sil1, BT_sr])
    ext_pr = (all_pr[:, 0].min() - 0.04, all_pr[:, 0].max() + 0.04, all_pr[:, 1].min() - 0.04, all_pr[:, 1].max() + 0.04)

    def make_base_grid(ext):
        xmin, xmax, ymin, ymax = ext
        lines = []
        for gx in np.linspace(xmin + 0.03, xmax - 0.03, 9):
            lines.append(np.column_stack([np.full(70, gx), np.linspace(ymin + 0.03, ymax - 0.03, 70)]))
        for gy in np.linspace(ymin + 0.03, ymax - 0.03, 7):
            lines.append(np.column_stack([np.linspace(xmin + 0.03, xmax - 0.03, 70), np.full(70, gy)]))
        return lines

    base_grid_sc = make_base_grid(ext_sc)
    base_grid_pr = make_base_grid(ext_pr)

    # Precompute metrics along alpha in [0, 1]
    n_steps = 25
    alphas_forward = np.linspace(0.0, 1.0, n_steps)
    curve_lm_sc = []
    curve_sil_sc = []
    curve_sil_pr = []
    iou_lm_sc = []
    iou_sil_sc = []
    iou_sil_pr = []

    for a in alphas_forward:
        b_lm = sil_mod.area_diff_breakdown(blend_pts(BS_sc_unw, BS_sc_lm1, a), BT_po, grid=600)
        b_sil = sil_mod.area_diff_breakdown(blend_pts(BS_sc_unw, BS_sc_sil1, a), BT_po, grid=600)
        b_pr = sil_mod.area_diff_breakdown(blend_pts(BS_pr_unw, BS_pr_sil1, a), BT_sr, grid=600)
        curve_lm_sc.append(b_lm["sym_diff_pct_of_target"])
        curve_sil_sc.append(b_sil["sym_diff_pct_of_target"])
        curve_sil_pr.append(b_pr["sym_diff_pct_of_target"])
        iou_lm_sc.append(b_lm["iou"])
        iou_sil_sc.append(b_sil["iou"])
        iou_sil_pr.append(b_pr["iou"])

    # Smooth loop schedule: hold at 0 (4 frames), forward 0->1 (25 frames), hold at 1 (10 frames), backward 1->0 (12 frames)
    schedule_idx = [0] * 4 + list(range(n_steps)) + [n_steps - 1] * 10 + list(range(n_steps - 1, -1, -2))
    frames = []

    bins_labels = ["0-10\nMyr", "10-25\nMyr", "25-50\nMyr", "50-80\nMyr", "80-110\nMyr", "110-160\nMyr"]
    xb = np.arange(len(bins_labels))
    sim_rms = [0.0139, 0.0151, 0.0162, 0.0285, 0.0499, 0.0383]
    aff_rms = [0.0123, 0.0127, 0.0132, 0.0218, 0.0427, 0.0304]
    thomp_rms = [0.0106, 0.0104, 0.0106, 0.0154, 0.0347, 0.0247]
    thomp_floor = [53.0, 61.3, 49.7, 10.6, 1.0, 0.0]
    floor_med = 0.0110

    for step_idx in schedule_idx:
        a = alphas_forward[step_idx]
        fig = plt.figure(figsize=(15.5, 8.8), dpi=125, facecolor="white")
        gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.36, wspace=0.23, top=0.86, bottom=0.08, left=0.05, right=0.97)

        fig.suptitle(
            f"Why 17-Landmark Morphometrics Fails D'Arcy Thompson's Fish Grids — & How 600-Point Silhouettes Recover Them\n"
            f"Warp Progression: α = {a*100:5.1f}%   |   [Green] Shared Overlap   [Red] Missing Target   [Blue] Excess Source Overhang   [Gold Dots] 9/17 Head Landmarks",
            fontsize=11.8, fontweight="bold", color="#0f172a", y=0.965
        )

        # Panel A: 17-LM Fit on Scarus -> Pomacanthus
        ax_a = fig.add_subplot(gs[0, 0])
        src_lm_a = blend_pts(BS_sc_unw, BS_sc_lm1, a)
        pts_lm_a = blend_pts(S_sc_unw, S_sc_lm1, a)
        grid_a = [blend_pts(g, fn_sc_lm(g), a) for g in base_grid_sc]
        render_shaded_panel(
            ax_a, src_lm_a, BT_po, pts_lm_a, T_po, grid_a,
            "A. 17-Landmark Fit (Scarus → Pomacanthus)\n9/17 Head LMs Pull Posterior Grid Wrong Way!",
            f"|A Δ B| = {curve_lm_sc[step_idx]:.1f}%  (IoU = {iou_lm_sc[step_idx]:.3f})  [WORSENS!]",
            "#b91c1c", ext_sc
        )

        # Panel B: Direct 600-pt Fit on Scarus -> Pomacanthus
        ax_b = fig.add_subplot(gs[0, 1])
        src_sil_b = blend_pts(BS_sc_unw, BS_sc_sil1, a)
        grid_b = [blend_pts(g, fn_sc_sil(g), a) for g in base_grid_sc]
        render_shaded_panel(
            ax_b, src_sil_b, BT_po, None, T_po, grid_b,
            "B. Direct 600-Point Silhouette Fit (8p Coaxial)\nSame Coordinate Law Fitted to True Contour",
            f"|A Δ B| = {curve_sil_sc[step_idx]:.1f}%  (IoU = {iou_sil_sc[step_idx]:.3f})  [RECOVERED!]",
            "#15803d", ext_sc
        )

        # Panel C: Direct 600-pt Fit on Pristigenys -> Scorpaena
        ax_c = fig.add_subplot(gs[0, 2])
        src_sil_c = blend_pts(BS_pr_unw, BS_pr_sil1, a)
        grid_c = [blend_pts(g, fn_pr_sil(g), a) for g in base_grid_pr]
        render_shaded_panel(
            ax_c, src_sil_c, BT_sr, None, T_sr, grid_c,
            "C. Pristigenys → Scorpaena (8p Fan Silhouette Fit)\nSlashes Area of Difference 3.4× (120.1% → 35.5%)",
            f"|A Δ B| = {curve_sil_pr[step_idx]:.1f}%  (IoU = {iou_sil_pr[step_idx]:.3f})  [-84.6 pts]",
            "#15803d", ext_pr
        )

        # Panel D: Live Trajectory of Area of Difference |A Δ B|(alpha)
        ax_d = fig.add_subplot(gs[1, 0:2])
        pct_x = alphas_forward * 100.0
        ax_d.plot(pct_x, curve_lm_sc, "--", color="#fca5a5", lw=1.8)
        ax_d.plot(pct_x, curve_sil_sc, "--", color="#86efac", lw=1.8)
        ax_d.plot(pct_x, curve_sil_pr, "--", color="#93c5fd", lw=1.8)

        ax_d.plot(pct_x[:step_idx + 1], curve_lm_sc[:step_idx + 1], "-", color="#dc2626", lw=3.0,
                  label=f"Scarus → Pomacanthus: 17-Landmark Fit (40.6% → {curve_lm_sc[step_idx]:.1f}% [Worse!])")
        ax_d.plot(pct_x[:step_idx + 1], curve_sil_sc[:step_idx + 1], "-", color="#16a34a", lw=3.0,
                  label=f"Scarus → Pomacanthus: Direct 600-pt Silhouette Fit (40.6% → {curve_sil_sc[step_idx]:.1f}% [Cut in Half!])")
        ax_d.plot(pct_x[:step_idx + 1], curve_sil_pr[:step_idx + 1], "-", color="#2563eb", lw=2.8,
                  label=f"Pristigenys → Scorpaena: Direct 600-pt Silhouette Fit (120.1% → {curve_sil_pr[step_idx]:.1f}% [3.4× Drop!])")

        ax_d.scatter([pct_x[step_idx]], [curve_lm_sc[step_idx]], color="#dc2626", s=65, zorder=6)
        ax_d.scatter([pct_x[step_idx]], [curve_sil_sc[step_idx]], color="#16a34a", s=65, zorder=6)
        ax_d.scatter([pct_x[step_idx]], [curve_sil_pr[step_idx]], color="#2563eb", s=65, zorder=6)
        ax_d.axvline(pct_x[step_idx], color="#475569", ls=":", lw=1.4)

        ax_d.set_xlim(0, 103)
        ax_d.set_ylim(15, 138)
        ax_d.set_xlabel("Coordinate Transformation Progression α (%)", fontsize=9.2, fontweight="bold")
        ax_d.set_ylabel("Area of Difference |A Δ B| (% Target)", fontsize=9.2, fontweight="bold")
        ax_d.set_title("D. Live Silhouette Error Trajectory: Why 17-LM Fitting Diverges While 600-pt Contour Fitting Converges",
                       fontsize=10.0, fontweight="bold", pad=8)
        ax_d.legend(loc="upper right", fontsize=8.0, framealpha=0.94)
        ax_d.grid(True, ls="--", alpha=0.35)

        # Panel E: Continuous Procrustes RMS Residual vs. Within-Genus Disparity Band (sigma_genus)
        ax_e = fig.add_subplot(gs[1, 2])
        ax_e.axvspan(-0.4, 2.45, color="#dcfce7", alpha=0.45)
        ax_e.axvspan(2.45, 5.4, color="#fee2e2", alpha=0.38)
        ax_e.axhspan(0.0, floor_med, color="#bbf7d0", alpha=0.55,
                     label=f"Within-Genus Disparity (σ_gen ≤ {floor_med:.3f})")
        ax_e.axhline(floor_med, color="#15803d", ls=":", lw=1.4)
        ax_e.axvline(2.45, color="#b91c1c", ls="--", lw=2.0)

        ax_e.plot(xb, sim_rms, "d--", color="#64748b", lw=1.7, markersize=5, label="Unwarped 4p (d₀)")
        ax_e.plot(xb, aff_rms, "^-", color="#2563eb", lw=2.0, markersize=5.5, label="6p Linear Affine")
        ax_e.plot(xb, thomp_rms, "o-", color="#15803d", lw=2.8, markersize=7, label="8p Thompson Grid")

        # Animated phylogenetic scanner bar synced with alpha
        scan_x = a * 5.0
        ax_e.axvline(scan_x, color="#0f172a", ls="-.", lw=1.5, alpha=0.65)

        for i, val in enumerate(thomp_rms):
            off_y = -0.0040 if i < 3 else 0.0022
            ax_e.text(xb[i], val + off_y, f"{val:.3f}\n({thomp_floor[i]:.0f}%)",
                      ha="center", va="bottom" if off_y > 0 else "top",
                      fontsize=7.0, fontweight="bold", color="#14532d")

        ax_e.text(2.52, 0.045, "← ~50 Myr Transition\n(<50 Myr: inside σ_gen;\n >50 Myr: Fin/Head=1.48×)",
                  color="#991b1b", fontsize=7.2, fontweight="bold",
                  bbox=dict(boxstyle="round,pad=0.18", facecolor="#fff1f2", edgecolor="#fecdd3", alpha=0.92))

        ax_e.set_ylim(0.003, 0.058)
        ax_e.set_xticks(xb)
        ax_e.set_xticklabels(bins_labels, fontsize=7.6, fontweight="bold")
        ax_e.set_ylabel("Post-Warp Procrustes RMS", fontsize=8.6, fontweight="bold")
        ax_e.set_title("E. Procrustes RMS vs. Intrageneric Disparity (4,334 Pairs)\n<50 Myr Within σ_genus (50-61%); >50 Myr Fin/Head Decouples",
                       fontsize=9.0, fontweight="bold", pad=8)
        ax_e.legend(loc="upper left", fontsize=6.8, framealpha=0.94)
        ax_e.grid(True, ls="--", alpha=0.35)

        fig.text(
            0.5, 0.012,
            "TL;DR: 9/17 landmarks in the head + 0% on fin edges cause 17-LM fits to distort posterior silhouettes; direct 600-pt contour fitting recovers Thompson's 1917 grids (100% bijective, det J > 0).",
            ha="center", va="bottom", fontsize=8.8, fontweight="bold", color="#1e293b",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="#f1f5f9", edgecolor="#cbd5e1", lw=0.8)
        )

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=125)
        plt.close(fig)
        buf.seek(0)
        frames.append(Image.open(buf).convert("P", palette=Image.Palette.ADAPTIVE, colors=256))

    OUT_GIF.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        OUT_GIF,
        save_all=True,
        append_images=frames[1:],
        duration=110,
        loop=0,
        optimize=True,
    )
    print(f"Saved animated summary GIF ({len(frames)} frames) to {OUT_GIF}")


if __name__ == "__main__":
    main()

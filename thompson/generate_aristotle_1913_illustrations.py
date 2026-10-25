#!/usr/bin/env python3
"""
Generate bespoke high-resolution illustration plates for D'Arcy Wentworth Thompson's
1913 Herbert Spencer Lecture at Oxford: 'On Aristotle as a Biologist, With a Prooemion
on Herbert Spencer', annotated with critical commentary and connections to Morpholib
& Morphogenesis Research.

Outputs (saved to docs/media/):
  1. thompson_1913_fig1_spencer_vs_aristotle.png
     - The Engineer (Spencer: stress, strain, scaling L^2 vs L^3) vs. The Naturalist
       (Aristotle: developmental differentiation, Goethe's law of subordination of parts,
       and Thompson's 1917 bridge).
  2. thompson_1913_fig2_lesbos_pyrrha_lagoon.png
     - Map of the Aegean & the Lagoon of Pyrrha (Kalloni) on Lesbos paired with a
       6-panel anatomical & ecological plate of the exact creatures Aristotle observed
       there (Starfish pentaradial diagram, Scallop Pecten shell, Sea-Urchin test,
       Purple Murex shell, Chameleon & Lagoon Fish migration diagram, and Sepia
       cuttlebone & anchor-arm anatomy).
  3. thompson_1913_fig3_cephalopod_embryology_soul.png
     - Crisp anatomical & embryological diagrams of Aristotle's four master case studies:
       (A) Cephalopod comparative anatomy (Sepia vs Octopus funnel, gut & hectocotylus arm)
       (B) Selachian placenta vitellina (Mustelus laevis yolk-sac interdigitating with uterus)
       (C) Chick Day-3 punctum saliens (beating heart, vitelline & allantoic vessels)
       (D) Plant embryo cor/cerebrum & experimental regeneration
       + (E) The Tripartite Soul (Threptike -> Aisthetike -> Nous) vs. Bergson's Diverging Tree
             and Driesch's Entelechy (Editorial Commentaries 7–13).
  4. thompson_1913_fig4_morpholib_four_connections.png
     - Visual synthesis of the 4 Morpholib Connections in the Editorial Synthesis:
       (C1) 17-LM wireframe (IoU 0.618) vs. 600-pt silhouette (IoU 0.749) on Scarus -> Pomacanthus
       (C2) Photograph-derived contour (Mola mola) vs. landmark-reconstructed B-spline contour (Diodon)
       (C3) 90x90 Finite-Difference Jacobian Determinant field (det DPhi > 0, 100% sampled positivity)
       (C4) MorphoEval 3-pillar uncalibrated assessment separation (Impl 30 / Bio 40 / Repro 30)
            and Pai et al. (2012) Xenopus bioelectric Vmem intervention vs. observation.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.path import Path as MplPath
import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
MEDIA_DIR = REPO_ROOT / "docs" / "media"
MEDIA_DIR.mkdir(parents=True, exist_ok=True)


def load_module(name: str, rel_path: str):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / rel_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_specimen_img(rel_path: str, size: tuple[int, int] = (320, 240)) -> np.ndarray:
    p = REPO_ROOT / rel_path
    if not p.exists():
        return np.full((size[1], size[0], 3), 240, dtype=np.uint8)
    im = Image.open(p).convert("RGB")
    w, h = im.size
    target_ratio = size[0] / size[1]
    cur_ratio = w / h
    if cur_ratio > target_ratio:
        new_w = int(h * target_ratio)
        left = (w - new_w) // 2
        im = im.crop((left, 0, left + new_w, h))
    else:
        new_h = int(w / target_ratio)
        top = (h - new_h) // 2
        im = im.crop((0, top, w, top + new_h))
    im = im.resize(size, Image.LANCZOS)
    return np.asarray(im)


def build_fig1_spencer_vs_aristotle():
    """
    Figure 1: Prooemion on Herbert Spencer vs. Aristotle & Goethe
    (Editorial Commentaries 1, 2, and 3).
    """
    fig = plt.figure(figsize=(15.5, 5.6), dpi=200, facecolor="#ffffff")
    gs = gridspec.GridSpec(
        1, 3, figure=fig, width_ratios=[1.05, 1.05, 1.1],
        wspace=0.22, left=0.04, right=0.97, top=0.80, bottom=0.14
    )

    fig.suptitle(
        "Plate I — The Engineer and the Naturalist: Spencer’s Mechanical Scaling vs. Goethe & Aristotle’s Subordination of Parts\n"
        "(Illustrating the Prooemion on Herbert Spencer & Editorial Commentaries 1–3, Original pp. 4–11)",
        fontsize=12.5, fontweight="bold", color="#0f172a", y=0.95
    )

    # Panel A: Spencer's Bridge-Builder & Galileo-Spencer Scaling Law (pp. 7-8, Note 9)
    ax1 = fig.add_subplot(gs[0, 0])
    L = np.linspace(0.5, 4.0, 200)
    strength_cross_section = L ** 2
    weight_volume = L ** 3
    aquatic_speed = 1.8 * (L ** 0.5)

    ax1.plot(L, strength_cross_section / strength_cross_section[-1] * 100, color="#0284c7", lw=2.6,
             label=r"Structural Cross-Section / Bone Stress Limit ($\propto L^2$)")
    ax1.plot(L, weight_volume / weight_volume[-1] * 100, color="#dc2626", lw=2.6, ls="--",
             label=r"Gravitational Mass / Load on Terrestrial Frame ($\propto L^3$)")
    ax1.plot(L, aquatic_speed / aquatic_speed[-1] * 65, color="#0d9488", lw=2.3, ls="-.",
             label=r"Aquatic Locomotive Speed (Buoyant Medium, Note 9)")

    ax1.axvline(2.4, color="#64748b", ls=":", lw=1.3)
    ax1.annotate(
        "Terrestrial / Avian\nSize Ceiling\n(Princ. Biol. i. 156)",
        xy=(2.4, 52), xytext=(1.05, 68),
        fontsize=8.8, fontweight="bold", color="#991b1b",
        bbox=dict(boxstyle="round,pad=0.3", fc="#fef2f2", ec="#f87171", lw=1.0),
        arrowprops=dict(arrowstyle="->", color="#dc2626", lw=1.2)
    )
    ax1.set_title("A. Spencer the Engineer (pp. 7–8, Note 9)\nStress, Strain & Size Limits in Organisms", fontsize=10.8, fontweight="bold", color="#0f172a", pad=8)
    ax1.set_xlabel("Characteristic Linear Dimension $L$ (Relative Scale)", fontsize=9.2, color="#334155")
    ax1.set_ylabel("Normalized Mechanical Magnitude (%)", fontsize=9.2, color="#334155")
    ax1.set_xlim(0.5, 4.0)
    ax1.set_ylim(0, 105)
    ax1.grid(True, ls=":", alpha=0.45, color="#cbd5e1")
    ax1.legend(loc="upper left", fontsize=7.8, frameon=True, facecolor="#f8fafc", edgecolor="#cbd5e1")

    # Panel B: Goethe (1795/1807), Von Baer & Spencer's Law of Differentiation (pp. 6-7)
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 10)
    ax2.axis("off")
    ax2.set_title("B. Goethe, Von Baer & Spencer (pp. 6–7, Notes 3–6)\nHomogeneous Repetition → Heterogeneous Subordination", fontsize=10.8, fontweight="bold", color="#0f172a", pad=8)

    ax2.text(5.0, 9.25, "1. Homogeneous / Unsubordinated State (Goethe, Zur Morphologie)",
             ha="center", va="center", fontsize=8.8, fontweight="bold", color="#334155")
    for i, x_c in enumerate(np.linspace(1.5, 8.5, 7)):
        circ = mpatches.Circle((x_c, 7.6), 0.52, facecolor="#bae6fd", edgecolor="#0284c7", lw=1.6)
        ax2.add_patch(circ)
        if i < 6:
            ax2.plot([x_c + 0.52, x_c + 0.64], [7.6, 7.6], color="#0284c7", lw=1.8)
    ax2.text(5.0, 6.55, "Identical metameric units ($U_1 \\simeq U_2 \\simeq \\dots \\simeq U_n$): low functional subordination",
             ha="center", va="center", fontsize=8.0, style="italic", color="#475569")

    ax2.annotate("", xy=(5.0, 5.15), xytext=(5.0, 6.15),
                 arrowprops=dict(arrowstyle="->,head_width=0.35,head_length=0.35", color="#0f172a", lw=2.0))
    ax2.text(5.3, 5.65, "Ontogenetic Differentiation\n(Wolff → Goethe → Von Baer)", fontsize=8.0, fontweight="bold", color="#0f172a", va="center")

    ax2.text(5.0, 4.55, "2. Heterogeneous / Subordinated Organization",
             ha="center", va="center", fontsize=8.8, fontweight="bold", color="#15803d")
    nodes = [
        (5.0, 3.4, "Central Coordinating Axis\n(Heart / Neural Hub)", "#dcfce7", "#15803d", 2.7, 0.9),
        (1.9, 1.3, "Sensory / Cranial\nModule", "#fef3c7", "#b45309", 2.2, 0.85),
        (5.0, 1.3, "Visceral / Nutrient\nModule", "#fce7f3", "#be185d", 2.2, 0.85),
        (8.1, 1.3, "Locomotor / Distal\nAppendages", "#e0e7ff", "#4338ca", 2.2, 0.85),
    ]
    for x0, y0, lbl, fc, ec, w, h in nodes:
        box = mpatches.FancyBboxPatch(
            (x0 - w / 2, y0 - h / 2), w, h,
            boxstyle="round,pad=0.12,rounding_size=0.2", facecolor=fc, edgecolor=ec, lw=1.5
        )
        ax2.add_patch(box)
        ax2.text(x0, y0, lbl, ha="center", va="center", fontsize=7.7, fontweight="bold", color="#0f172a")
    for x_t in (1.9, 5.0, 8.1):
        ax2.annotate("", xy=(x_t, 1.82), xytext=(5.0, 2.88),
                     arrowprops=dict(arrowstyle="<->", color="#475569", lw=1.3))

    # Panel C: Editorial Commentary 1-3 Table — Four Distinct Meanings of "Evolution" & Method
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.set_xlim(0, 10)
    ax3.set_ylim(0, 10)
    ax3.axis("off")
    ax3.set_title("C. Editorial Commentaries 1–3: Disentangling Scales & Habits\nWhy the Engineer and Naturalist See Different Problems", fontsize=10.8, fontweight="bold", color="#0f172a", pad=8)

    cards = [
        (8.1, "1. Cosmic Equilibration (Spencer)", "Redistribution of matter & motion; alternation of\naggregation and dissolution ('Persistence of Force').", "#f1f5f9", "#475569"),
        (5.8, "2. Individual Ontogeny (Aristotle / Von Baer)", "Embryonic differentiation from homogeneous germ\nto subordinated organs within a single lifetime.", "#e0f2fe", "#0284c7"),
        (3.5, "3. Phylogenetic Descent (Darwin)", "Historical divergence of lineages across geological time\n(distinct from a conceptual scala naturae).", "#dcfce7", "#15803d"),
        (1.2, "4. Physical Morphogenesis (Roux / Thompson 1917)", "Entwicklungsmechanik: how stress, strain, tension & bioelectric\nfields physically shape tissue—bridging Spencer & Aristotle.", "#fef3c7", "#b45309"),
    ]
    for y_c, title_c, body_c, fc, ec in cards:
        rect = mpatches.FancyBboxPatch(
            (0.3, y_c - 0.85), 9.4, 1.85,
            boxstyle="round,pad=0.1,rounding_size=0.2", facecolor=fc, edgecolor=ec, lw=1.5
        )
        ax3.add_patch(rect)
        ax3.text(0.65, y_c + 0.42, title_c, ha="left", va="center", fontsize=8.7, fontweight="bold", color=ec)
        ax3.text(0.65, y_c - 0.25, body_c, ha="left", va="center", fontsize=7.9, color="#1e293b")

    out_path = MEDIA_DIR / "thompson_1913_fig1_spencer_vs_aristotle.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Saved {out_path}")


def build_fig2_lesbos_pyrrha_lagoon():
    """
    Figure 2: Aristotle's Field Station on Lesbos (The Lagoon of Pyrrha / Kalloni)
    and the Fauna of the Asiatic Coast (pp. 12–14, Commentaries 4–5).
    """
    fig = plt.figure(figsize=(16.0, 6.7), dpi=200, facecolor="#ffffff")
    gs = gridspec.GridSpec(
        1, 2, figure=fig, width_ratios=[1.12, 1.38],
        wspace=0.16, left=0.04, right=0.97, top=0.83, bottom=0.09
    )

    fig.suptitle(
        "Plate II — Insula Nobilis et Amoena: Aristotle’s Marine Field Station at the Lagoon of Pyrrha (Lesbos, 345–342 BC)\n"
        "(Illustrating Original pp. 12–14, Notes 12–13 & Editorial Commentaries 4–5: From Local Place-Names to Biological Science)",
        fontsize=12.4, fontweight="bold", color="#0f172a", y=0.95
    )

    # Left panel: Schematic Biogeographical Map of the Northern/Eastern Aegean & Lesbos / Pyrrha Lagoon
    ax_map = fig.add_subplot(gs[0, 0])
    ax_map.set_facecolor("#e0f2fe")
    ax_map.set_xlim(22.5, 27.5)
    ax_map.set_ylim(37.8, 41.2)

    macedon = [(22.5, 40.2), (23.3, 40.0), (23.4, 39.9), (23.9, 40.0), (24.4, 40.3), (25.5, 40.8), (26.5, 40.6), (26.5, 41.2), (22.5, 41.2)]
    greece_main = [(22.5, 37.8), (23.8, 37.9), (24.1, 38.3), (23.3, 38.9), (22.9, 39.4), (22.5, 40.2)]
    euboea = [(23.1, 38.8), (23.9, 38.6), (24.6, 38.0), (24.2, 38.4), (23.4, 39.0)]
    asia_minor = [(26.2, 40.0), (26.08, 39.48), (26.6, 39.3), (26.9, 38.9), (26.3, 38.3), (27.2, 37.8), (27.5, 37.8), (27.5, 40.5), (26.6, 40.3)]

    for poly_pts in (macedon, greece_main, euboea, asia_minor):
        ax_map.add_patch(mpatches.Polygon(poly_pts, closed=True, facecolor="#f1f5f9", edgecolor="#64748b", lw=1.2))

    lesbos_outer = [
        (25.83, 39.18), (26.05, 39.35), (26.38, 39.38), (26.60, 39.05),
        (26.42, 38.97), (26.25, 39.03),
        (26.12, 39.09), (26.17, 39.21), (26.26, 39.22), (26.23, 39.13), (26.08, 39.07),
        (25.92, 39.10)
    ]
    ax_map.add_patch(mpatches.Polygon(lesbos_outer, closed=True, facecolor="#fde68a", edgecolor="#b45309", lw=2.0, zorder=4))

    ax_map.scatter([26.20], [39.18], s=140, color="#dc2626", edgecolor="white", lw=1.5, zorder=6)
    ax_map.annotate(
        "LAGOON OF PYRRHA (Lesbos)\n• Starfish pest & scallop drought\n• Winter-breeding sea-urchins\n• Breeding migration of lagoon fishes",
        xy=(26.20, 39.18), xytext=(22.65, 39.32),
        fontsize=8.1, fontweight="bold", color="#7f1d1d",
        bbox=dict(boxstyle="round,pad=0.35", fc="#fef2f2", ec="#dc2626", lw=1.3),
        arrowprops=dict(arrowstyle="->", color="#dc2626", lw=1.6),
        zorder=7
    )

    sites = [
        (23.78, 40.52, "Stagira\n(Birthplace, 384 BC)", "#1e293b", (-25, -26)),
        (26.34, 39.49, "Assos / Atarneus\n(Court of Hermias, 347–345 BC)", "#0369a1", (-55, 16)),
        (26.08, 39.48, "Cape Lectum\n(Giant purple Murex)", "#7c3aed", (-85, -8)),
        (26.55, 39.10, "Mitylene & Cape Malia\n(Landward vs. seaward sponges)", "#b45309", (-55, -28)),
        (23.72, 37.98, "Athens (Lyceum)\n(335–323 BC)", "#15803d", (-10, 14)),
        (23.60, 38.46, "Chalcis (Euboea)\n(Retirement, 322 BC)", "#475569", (-65, 12)),
        (26.85, 38.45, "Asiatic / Ionian Coast\n(Chameleon dissections)", "#0f766e", (-75, -22)),
    ]
    for lon, lat, lbl, col, offset in sites:
        ax_map.scatter([lon], [lat], s=50, color=col, edgecolor="white", lw=1.0, zorder=5)
        ax_map.annotate(
            lbl, xy=(lon, lat), xytext=offset, textcoords="offset points",
            fontsize=7.5, fontweight="bold", color=col,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=col, alpha=0.9, lw=0.8),
            arrowprops=dict(arrowstyle="-", color=col, lw=0.8),
            zorder=6
        )

    ax_map.set_title("A. Biographical & Zoological Geography of the Historia Animalium\nWhy Lesbos & the Asiatic Coast Preceded the Lyceum (pp. 12–14)", fontsize=10.5, fontweight="bold", color="#0f172a", pad=8)
    ax_map.set_xlabel("Longitude (°E)", fontsize=8.8, color="#334155")
    ax_map.set_ylabel("Latitude (°N)", fontsize=8.8, color="#334155")
    ax_map.grid(True, ls=":", alpha=0.4, color="#94a3b8")

    # Right panel: 2x3 Gallery combining real museum specimen photos & crisp anatomical diagrams
    gs_spec = gridspec.GridSpecFromSubplotSpec(2, 3, subplot_spec=gs[0, 1], hspace=0.38, wspace=0.15)

    # Tile 1: Crisp Pentaradial Starfish Diagram (Asterias in Pyrrha Lagoon)
    ax_s1 = fig.add_subplot(gs_spec[0, 0])
    ax_s1.set_facecolor("#0f172a")
    theta = np.linspace(0, 2 * np.pi, 500)
    r_star = 0.52 + 0.36 * np.cos(5 * theta - np.pi / 2)
    r_star = np.maximum(r_star, 0.18)
    ax_s1.fill(r_star * np.cos(theta), r_star * np.sin(theta), color="#fb923c", edgecolor="#fed7aa", lw=2.0, alpha=0.9)
    for k_arm in range(5):
        ang = k_arm * 2 * np.pi / 5 + np.pi / 2
        rs = np.linspace(0.12, 0.78, 8)
        ax_s1.plot(rs * np.cos(ang), rs * np.sin(ang), "o-", color="#fef08a", markersize=3.2, lw=1.0)
    ax_s1.add_patch(mpatches.Circle((0, 0), 0.12, facecolor="#f97316", edgecolor="#fef08a", lw=1.5))
    ax_s1.set_xlim(-1.05, 1.05)
    ax_s1.set_ylim(-1.0, 1.0)
    ax_s1.set_aspect("equal")
    ax_s1.set_xticks([]); ax_s1.set_yticks([])
    ax_s1.set_title("1. Starfish (Asterias)", fontsize=9.2, fontweight="bold", color="#0f172a", pad=4)
    ax_s1.set_xlabel("Abundant in Pyrrha Lagoon;\n'a pest to the fishermen' (p. 13)", fontsize=7.8, color="#334155", labelpad=4)

    # Tile 2: Real Sea-Urchin Test Photo (Strongylocentrotus / Echinus)
    ax_s2 = fig.add_subplot(gs_spec[0, 1])
    urchin_img = load_specimen_img("datasets/species_gallery/II_Anaima_Invertebrata/Strongylocentrotus_purpuratus.jpg", size=(320, 220))
    ax_s2.imshow(urchin_img)
    ax_s2.set_xticks([]); ax_s2.set_yticks([])
    ax_s2.set_title("2. Sea-Urchin (Echinus)", fontsize=9.2, fontweight="bold", color="#0f172a", pad=4)
    ax_s2.set_xlabel("Comes into season in winter\nin the Pyrrha lagoon (p. 13)", fontsize=7.8, color="#334155", labelpad=4)

    # Tile 3: Scallop (Pecten) Radial Ribbed Valve Diagram
    ax_s3 = fig.add_subplot(gs_spec[0, 2])
    ax_s3.set_facecolor("#0f172a")
    th_p = np.linspace(-0.62 * np.pi, 0.62 * np.pi, 300)
    r_p = 0.85 * (np.cos(th_p * 0.72) ** 0.6) * (1.0 + 0.04 * np.cos(16 * th_p))
    x_p = r_p * np.sin(th_p)
    y_p = r_p * np.cos(th_p) - 0.35
    ax_s3.fill(np.concatenate([[-0.32, 0.32], x_p]), np.concatenate([[-0.35, -0.35], y_p]),
               color="#fda4af", edgecolor="#ffe4e6", lw=1.8, alpha=0.9)
    # Hinge ears (auricles)
    ax_s3.fill([-0.38, 0.38, 0.25, -0.25], [-0.35, -0.35, -0.48, -0.48], color="#fb7185", edgecolor="#ffe4e6", lw=1.4)
    for rib_a in np.linspace(-0.52 * np.pi, 0.52 * np.pi, 13):
        r_end = 0.82 * (np.cos(rib_a * 0.72) ** 0.6)
        ax_s3.plot([0, r_end * np.sin(rib_a)], [-0.35, r_end * np.cos(rib_a) - 0.35], color="#881337", lw=1.0, alpha=0.7)
    ax_s3.set_xlim(-0.95, 0.95)
    ax_s3.set_ylim(-0.62, 0.65)
    ax_s3.set_aspect("equal")
    ax_s3.set_xticks([]); ax_s3.set_yticks([])
    ax_s3.set_title("3. Scallop (Pecten)", fontsize=9.2, fontweight="bold", color="#0f172a", pad=4)
    ax_s3.set_xlabel("Exterminated in Pyrrha by\ndrought & fishermen's dredge (p. 13)", fontsize=7.8, color="#334155", labelpad=4)

    # Tile 4: Real Murex Shell Photograph (Cape Lectum)
    ax_s4 = fig.add_subplot(gs_spec[1, 0])
    murex_img = load_specimen_img("datasets/species_gallery/II_Anaima_Invertebrata/Murex_pecten.jpg", size=(320, 220))
    ax_s4.imshow(murex_img)
    ax_s4.set_xticks([]); ax_s4.set_yticks([])
    ax_s4.set_title("4. Purple Murex Shell", fontsize=9.2, fontweight="bold", color="#0f172a", pad=4)
    ax_s4.set_xlabel("Big purple shells recorded\nat Cape Lectum (p. 12)", fontsize=7.8, color="#334155", labelpad=4)

    # Tile 5: Real Coiled Cephalopod Shell Photo (Nautilus / Argonauta comparison)
    ax_s5 = fig.add_subplot(gs_spec[1, 1])
    naut_img = load_specimen_img("datasets/species_gallery/II_Anaima_Invertebrata/Nautilus_pompilius.jpg", size=(320, 220))
    ax_s5.imshow(naut_img)
    ax_s5.set_xticks([]); ax_s5.set_yticks([])
    ax_s5.set_title("5. Shelled Cephalopods", fontsize=9.2, fontweight="bold", color="#0f172a", pad=4)
    ax_s5.set_xlabel("Argonauta & chambered shells\nin Mediterranean waters (p. 19)", fontsize=7.8, color="#334155", labelpad=4)

    # Tile 6: Lagoon of Pyrrha Ecological & Seasonal Table (p. 13)
    ax_s6 = fig.add_subplot(gs_spec[1, 2])
    ax_s6.set_facecolor("#f8fafc")
    ax_s6.set_xlim(0, 10)
    ax_s6.set_ylim(0, 10)
    ax_s6.set_xticks([]); ax_s6.set_yticks([])
    ax_s6.text(5.0, 8.8, "Pyrrha Lagoon Field Notes", ha="center", va="center", fontsize=8.4, fontweight="bold", color="#0f172a")
    notes_txt = (
        "• Present:\n"
        "  Starfish, winter sea-urchins,\n"
        "  resident gudgeon (κωβιός)\n"
        "• Absent inside lagoon:\n"
        "  Octopus, parrot-wrasse,\n"
        "  spiny dogfish, crawfish\n"
        "• Asiatic coast:\n"
        "  Chameleon dissections"
    )
    ax_s6.text(0.7, 4.4, notes_txt, ha="left", va="center", fontsize=7.4, color="#1e293b", linespacing=1.28)
    ax_s6.set_title("6. Presence / Absence Ecology", fontsize=9.2, fontweight="bold", color="#0f172a", pad=4)
    ax_s6.set_xlabel("Systematic habitat census vs.\nfolklore (Comm. 5 & 7, p. 13)", fontsize=7.8, color="#334155", labelpad=4)

    for ax_tile in (ax_s1, ax_s2, ax_s3, ax_s4, ax_s5, ax_s6):
        for sp in ax_tile.spines.values():
            sp.set_color("#b45309")
            sp.set_linewidth(1.4)

    out_path = MEDIA_DIR / "thompson_1913_fig2_lesbos_pyrrha_lagoon.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Saved {out_path}")


def build_fig3_cephalopod_embryology_soul():
    """
    Figure 3: Aristotle's Masterpieces of Comparative Anatomy & Embryology
    (Cuttlefish, Selachian Yolk-Sac Placenta, Chick Punctum Saliens) and the
    Biological Roots of De Anima (pp. 17–29, Commentaries 7–13).
    """
    fig = plt.figure(figsize=(16.0, 6.6), dpi=200, facecolor="#ffffff")
    gs = gridspec.GridSpec(
        1, 2, figure=fig, width_ratios=[1.25, 1.15],
        wspace=0.18, left=0.04, right=0.97, top=0.82, bottom=0.10
    )

    fig.suptitle(
        "Plate III — Anatomia Animata: From Cephalopod, Shark & Chick Embryology to the Biological Architecture of De Anima\n"
        "(Illustrating Original pp. 17–29, Notes 18–29 & Editorial Commentaries 7–13: Connected Observations vs. Theoretical Inference)",
        fontsize=12.3, fontweight="bold", color="#0f172a", y=0.95
    )

    # Left: 2x2 Crisp Anatomical & Embryological Diagrams (pp. 17-23)
    gs_left = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=gs[0, 0], hspace=0.40, wspace=0.16)

    # Panel A: Cuttlefish (Sepia) Anatomy, Yolk-Sac Embryo & Hectocotylus Arm (pp. 17-19)
    ax_a = fig.add_subplot(gs_left[0, 0])
    ax_a.set_facecolor("#f8fafc")
    ax_a.set_xlim(0, 10)
    ax_a.set_ylim(0, 10)
    ax_a.set_xticks([]); ax_a.set_yticks([])

    # Mantle + Cuttlebone + Ink-bag + Funnel + 8 short arms + 2 long anchor tentacles + Hectocotylus
    mantle = mpatches.Ellipse((3.2, 5.4), 4.2, 2.6, facecolor="#bae6fd", edgecolor="#0369a1", lw=1.8)
    cuttlebone = mpatches.Ellipse((3.1, 6.1), 3.4, 0.7, facecolor="#ffffff", edgecolor="#0284c7", lw=1.2, ls="--")
    inkbag = mpatches.Ellipse((2.6, 4.6), 1.3, 0.55, facecolor="#1e293b", edgecolor="#0f172a", lw=1.0)
    head_c = mpatches.Circle((5.6, 5.4), 0.95, facecolor="#7dd3fc", edgecolor="#0369a1", lw=1.6)
    yolk_c = mpatches.Circle((7.5, 7.6), 0.95, facecolor="#fde047", edgecolor="#ca8a04", lw=1.5)
    for p_patch in (mantle, cuttlebone, inkbag, head_c, yolk_c):
        ax_a.add_patch(p_patch)
    ax_a.plot([5.6, 5.6], [5.7, 5.7], "o", color="#0f172a", markersize=5)  # Eye
    # Two long anchor arms (Sepia)
    t_x = np.linspace(6.4, 9.5, 40)
    ax_a.plot(t_x, 5.1 + 0.35 * np.sin((t_x - 6.4) * 2.2), color="#0369a1", lw=2.2)
    ax_a.plot(t_x, 4.3 - 0.35 * np.sin((t_x - 6.4) * 2.2), color="#dc2626", lw=2.2, ls="-.")
    ax_a.plot([6.2, 7.0], [6.0, 7.0], color="#ca8a04", lw=1.8)  # Cephalic yolk stalk

    ax_a.text(3.1, 6.1, "Cuttlebone", ha="center", va="center", fontsize=6.6, fontweight="bold", color="#0369a1")
    ax_a.text(2.6, 4.6, "Ink-bag", ha="center", va="center", fontsize=6.3, color="white", fontweight="bold")
    ax_a.text(7.5, 7.6, "Cephalic\nYolk-Sac", ha="center", va="center", fontsize=6.5, fontweight="bold", color="#854d0e")
    ax_a.text(8.1, 5.8, "Anchor Arms", ha="center", fontsize=6.8, fontweight="bold", color="#0369a1")
    ax_a.text(7.8, 3.1, "Male Hectocotylus Arm", ha="center", fontsize=6.8, fontweight="bold", color="#dc2626")
    ax_a.set_title("A. Cephalopod Treatise (pp. 17–19)", fontsize=9.3, fontweight="bold", color="#0f172a", pad=4)
    ax_a.set_xlabel("Sepia cuttlebone, largest ink-bag, anchor arms,\ncephalic yolk-sac & hectocotylus arm", fontsize=7.6, color="#334155", labelpad=4)

    # Panel B: Selachian Yolk-Sac Placenta (Mustelus laevis / γαλεὸς λεῖος, pp. 20-21)
    ax_b = fig.add_subplot(gs_left[0, 1])
    ax_b.set_facecolor("#f8fafc")
    ax_b.set_xlim(0, 10)
    ax_b.set_ylim(0, 10)
    ax_b.set_xticks([]); ax_b.set_yticks([])

    # Maternal uterine wall at bottom with interdigitating folds
    x_u = np.linspace(0.6, 9.4, 300)
    y_u = 2.4 + 0.55 * np.sin((x_u - 0.6) * 3.6)
    ax_b.fill_between(x_u, 0.4, y_u, color="#fecdd3", edgecolor="#be123c", lw=1.6)
    ax_b.text(5.0, 1.05, "Maternal Uterine Wall & Vascular Folds", ha="center", va="center", fontsize=7.2, fontweight="bold", color="#9f1239")

    # Interweaving Yolk-Sac Placenta
    y_yolk = 2.65 + 0.55 * np.sin((x_u - 0.6) * 3.6)
    ax_b.fill_between(x_u[60:240], y_yolk[60:240], 4.8, color="#fef08a", edgecolor="#ca8a04", lw=1.5)
    ax_b.text(5.0, 3.85, "Interdigitating Yolk-Sac (Placenta Vitellina)", ha="center", va="center", fontsize=6.9, fontweight="bold", color="#854d0e")

    # Embryo shark above connected by umbilical yolk stalk
    shark_body = mpatches.Ellipse((5.2, 7.6), 5.2, 1.4, facecolor="#cbd5e1", edgecolor="#334155", lw=1.6)
    ax_b.add_patch(shark_body)
    ax_b.plot([5.0, 5.0], [4.8, 6.9], color="#dc2626", lw=2.4)
    ax_b.text(5.2, 7.6, "Embryo γαλεὸς λεῖος (Mustelus laevis)", ha="center", va="center", fontsize=7.2, fontweight="bold", color="#0f172a")
    ax_b.text(6.8, 5.85, "Vitelline Stalk", ha="center", fontsize=6.8, fontweight="bold", color="#dc2626")
    ax_b.set_title("B. Shark Placenta Vitellina (pp. 20–21)", fontsize=9.3, fontweight="bold", color="#0f172a", pad=4)
    ax_b.set_xlabel("Rediscovered by Steno (1673) & J. Müller (1842);\nyolk-sac distinguished from mammalian allantois", fontsize=7.6, color="#334155", labelpad=4)

    # Panel C: Chick Embryo Day 3 — Punctum Saliens (pp. 21-23, Commentary 8)
    ax_c = fig.add_subplot(gs_left[1, 0])
    ax_c.set_facecolor("#f8fafc")
    ax_c.set_xlim(0, 10)
    ax_c.set_ylim(0, 10)
    ax_c.set_xticks([]); ax_c.set_yticks([])

    egg_shell = mpatches.Ellipse((5.0, 5.0), 8.6, 6.8, facecolor="#fffbeb", edgecolor="#d97706", lw=1.8)
    yolk_sac = mpatches.Ellipse((3.6, 4.4), 3.6, 2.8, facecolor="#fde047", edgecolor="#ca8a04", lw=1.5)
    allantois = mpatches.Ellipse((6.8, 4.2), 2.5, 2.0, facecolor="#bae6fd", edgecolor="#0284c7", lw=1.5)
    embryo_c = mpatches.Ellipse((5.1, 6.8), 2.4, 1.2, facecolor="#fce7f3", edgecolor="#be185d", lw=1.5)
    for p_patch in (egg_shell, yolk_sac, allantois, embryo_c):
        ax_c.add_patch(p_patch)
    # Punctum saliens (beating heart speck) + 2 umbilical vessels
    ax_c.scatter([5.1], [6.6], s=95, color="#dc2626", edgecolor="white", lw=1.2, zorder=6)
    ax_c.plot([5.1, 3.6], [6.6, 4.6], color="#dc2626", lw=2.0)
    ax_c.plot([5.1, 6.8], [6.6, 4.3], color="#2563eb", lw=2.0)
    ax_c.text(5.1, 7.75, "Day-3 Punctum Saliens (Heart)", ha="center", fontsize=7.0, fontweight="bold", color="#dc2626")
    ax_c.text(3.6, 4.3, "Yolk-Sac\nVessel", ha="center", va="center", fontsize=6.7, fontweight="bold", color="#854d0e")
    ax_c.text(6.8, 4.2, "Allantois\nVessel", ha="center", va="center", fontsize=6.7, fontweight="bold", color="#0369a1")
    ax_c.set_title("C. Chick Embryo Day 3–10 (pp. 21–23)", fontsize=9.3, fontweight="bold", color="#0f172a", pad=4)
    ax_c.set_xlabel("Observation (heart appears first + 2 vessels)\nvs. Inference ('central seat of the soul', Comm. 8)", fontsize=7.6, color="#334155", labelpad=4)

    # Panel D: Plant Embryo Center (Cor/Cerebrum) & Polity-Organism Analogy (pp. 23-25, Comm. 9)
    ax_d = fig.add_subplot(gs_left[1, 1])
    ax_d.set_facecolor("#f8fafc")
    ax_d.set_xlim(0, 10)
    ax_d.set_ylim(0, 10)
    ax_d.set_xticks([]); ax_d.set_yticks([])

    cot_l = mpatches.Ellipse((3.4, 5.2), 2.6, 4.2, angle=15, facecolor="#bbf7d0", edgecolor="#15803d", lw=1.6)
    cot_r = mpatches.Ellipse((6.6, 5.2), 2.6, 4.2, angle=-15, facecolor="#bbf7d0", edgecolor="#15803d", lw=1.6)
    ax_d.add_patch(cot_l); ax_d.add_patch(cot_r)
    # Plumule (stem up) and Radicle (root down) from central node
    ax_d.annotate("", xy=(5.0, 8.7), xytext=(5.0, 5.2), arrowprops=dict(arrowstyle="->", color="#15803d", lw=2.4))
    ax_d.annotate("", xy=(5.0, 1.3), xytext=(5.0, 5.2), arrowprops=dict(arrowstyle="->", color="#b45309", lw=2.4))
    ax_d.scatter([5.0], [5.2], s=110, color="#dc2626", edgecolor="white", lw=1.4, zorder=6)
    ax_d.text(5.0, 9.1, "Ascending Shoot (Stem)", ha="center", fontsize=6.9, fontweight="bold", color="#15803d")
    ax_d.text(5.0, 0.75, "Descending Radicle (Root)", ha="center", fontsize=6.9, fontweight="bold", color="#b45309")
    ax_d.text(5.0, 5.2, "  Cor / Node", ha="left", va="center", fontsize=7.0, fontweight="bold", color="#dc2626")
    ax_d.text(2.5, 5.2, "Seed-\nLeaf", ha="center", va="center", fontsize=6.8, fontweight="bold", color="#166534")
    ax_d.text(7.5, 5.2, "Seed-\nLeaf", ha="center", va="center", fontsize=6.8, fontweight="bold", color="#166534")
    ax_d.set_title("D. Plant Embryo & Inductive Method (pp. 23–25)", fontsize=9.3, fontweight="bold", color="#0f172a", pad=4)
    ax_d.set_xlabel("Central point between cotyledons (p. 23);\n200+ Constitutions compiled like species (Comm. 9)", fontsize=7.6, color="#334155", labelpad=4)

    for ax_tile in (ax_a, ax_b, ax_c, ax_d):
        for sp in ax_tile.spines.values():
            sp.set_color("#0284c7")
            sp.set_linewidth(1.4)

    # Right: Nested Geometry of the Soul ("like a triangle within a square", p. 27)
    # vs. Bergson's Diverging Tree (p. 28) & Driesch's Entelechy (p. 29)
    ax_r = fig.add_subplot(gs[0, 1])
    ax_r.set_xlim(0, 10)
    ax_r.set_ylim(0, 10)
    ax_r.axis("off")
    ax_r.set_title(
        "E. Aristotle’s Nested Psychē (De Anima II.1–3) vs. Bergson’s Diverging Tree\n"
        "‘As a triangle within a square, each lower grade is implicit in the higher’ (pp. 26–29)",
        fontsize=10.4, fontweight="bold", color="#0f172a", pad=8
    )

    nested_boxes = [
        (0.3, 3.6, 5.4, 5.8, "#ede9fe", "#6d28d9",
         "3. Διανοητικὴ Ψυχή / Νοῦς (Rational Soul — Man)",
         "• Knowledge, deliberation & wisdom\n• 'Cometh from afar' (θύραθεν, De Gen. An. 736b27)"),
        (0.65, 3.85, 4.7, 3.9, "#e0f2fe", "#0284c7",
         "2. Αἰσθητικὴ & Κινητικὴ Ψυχή (Sentient — Animals)",
         "• Sensation, locomotion, fear, courage & sagacity\n• Comparative psychology across animal kinds"),
        (1.0, 4.1, 4.0, 2.0, "#dcfce7", "#15803d",
         "1. Πρώτη / Θρεπτικὴ Ψυχή (Nutrient Soul — Plants & All Life)",
         "• Nutrition, growth & reproduction\n• Form/actuality of an organized body (ἐντελέχεια ἡ πρώτη)"),
    ]
    for x0, y0, w, h, fc, ec, head_txt, body_txt in nested_boxes:
        rect = mpatches.FancyBboxPatch(
            (x0, y0), w, h,
            boxstyle="round,pad=0.08,rounding_size=0.18", facecolor=fc, edgecolor=ec, lw=1.6
        )
        ax_r.add_patch(rect)
        ax_r.text(x0 + 0.2, y0 + h - 0.36, head_txt, ha="left", va="center", fontsize=8.0, fontweight="bold", color=ec)
        ax_r.text(x0 + 0.2, y0 + h - 0.98, body_txt, ha="left", va="center", fontsize=7.4, color="#1e293b")

    ax_r.text(7.9, 9.1, "Bergson’s Recasting (p. 28)\nDiverging Outgrowths vs. Ladder",
              ha="center", va="center", fontsize=8.4, fontweight="bold", color="#b45309")
    ax_r.plot([7.9, 7.9], [4.2, 5.4], color="#78350f", lw=3.0)
    ax_r.plot([7.9, 6.6], [5.4, 7.6], color="#15803d", lw=2.2)
    ax_r.plot([7.9, 7.9], [5.4, 7.8], color="#d97706", lw=2.2)
    ax_r.plot([7.9, 9.2], [5.4, 7.6], color="#2563eb", lw=2.2)
    ax_r.scatter([7.9, 6.6, 7.9, 9.2], [4.2, 7.6, 7.8, 7.6], s=55, color=["#78350f", "#15803d", "#d97706", "#2563eb"], zorder=5)
    ax_r.text(7.9, 3.85, "Original Activity", ha="center", va="top", fontsize=7.4, fontweight="bold", color="#78350f")
    ax_r.text(6.6, 7.85, "Torpor\n(Plants)", ha="center", va="bottom", fontsize=7.3, fontweight="bold", color="#15803d")
    ax_r.text(7.9, 8.0, "Instinct\n(Insects)", ha="center", va="bottom", fontsize=7.3, fontweight="bold", color="#b45309")
    ax_r.text(9.2, 7.85, "Intelligence\n(Vertebrates)", ha="center", va="bottom", fontsize=7.3, fontweight="bold", color="#2563eb")

    comm_box = mpatches.FancyBboxPatch(
        (0.3, 0.35), 9.4, 2.8,
        boxstyle="round,pad=0.1,rounding_size=0.2", facecolor="#fffbeb", edgecolor="#d97706", lw=1.5
    )
    ax_r.add_patch(comm_box)
    ax_r.text(
        0.55, 2.75,
        "Editorial Commentaries 10–13: Separating Conceptual Order, Vitalism & Mechanism",
        ha="left", va="center", fontsize=8.6, fontweight="bold", color="#92400e"
    )
    ax_r.text(
        0.55, 1.48,
        "• Comm. 10 (Psychē): Organization & capacities of a living body—not a claim that every plant has subjective consciousness.\n"
        "• Comm. 11 (Order vs. Genealogy): A conceptual grading of capacities ≠ a phylogenetic lineage or developmental trajectory.\n"
        "• Comm. 12 (Entelechy vs. Driesch): Aristotle’s actuality of an organized body ≠ Driesch’s non-physical vitalist exception.\n"
        "• Comm. 13 (Final Causes → Homeostasis): Natural purposiveness (Physics II.8) translates in modern morphogenesis into\n"
        "  testable feedback questions: specify the maintained variable, the tolerated perturbations, and the recovery mechanism.",
        ha="left", va="center", fontsize=7.55, color="#1e293b"
    )

    out_path = MEDIA_DIR / "thompson_1913_fig3_cephalopod_embryology_soul.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Saved {out_path}")


def build_fig4_morpholib_four_connections():
    """
    Figure 4: The Editorial Synthesis for Morphogenesis Research —
    Visualizing [E6] Pai et al. (2012) Bioelectric Patterning and
    Morpholib Connections 1, 2, 3, and 4 ([E7]–[E10]).
    """
    sil_mod = load_module("sil_mod", "thompson/1917-thompson-silhouette-area-paradox.py")
    at, sp_mod = sil_mod.load_modules()
    names, families, X = at.load_landmarks()
    Y = at.gpa(X)
    O_sp = sp_mod.outline_analysis(Y, names, allow_download=False)
    meta = sil_mod.json.loads(sil_mod.EXACT_JSON.read_text(encoding="utf-8"))
    mean_c = Y.mean(0)

    S_sc = O_sp["photos"]["Scarus_ghobban"]["landmarks"]
    BS_sc = O_sp["photos"]["Scarus_ghobban"]["boundary"]
    T_po = O_sp["photos"]["Pomacanthus_semicirculatus"]["landmarks"]
    BT_po = O_sp["photos"]["Pomacanthus_semicirculatus"]["boundary"]

    p_lm, _ = at.fit_nonlinear_single("coaxial", S_sc, T_po)
    BS_sc_lm = at.FAMILY_FUNCS["coaxial"](p_lm, BS_sc)
    p_sil, BS_sc_sil, apply_sil = sil_mod.fit_direct_to_silhouette("coaxial", S_sc, BS_sc, T_po, BT_po, at)

    idx_dio = names.index("Diodon_hystrix")
    S_dio_raw = Y[idx_dio]
    BS_dio_raw = sil_mod.build_diodon_silhouette_from_landmarks(S_dio_raw, 600)
    T_mola, BT_mola = sil_mod.extract_aligned_silhouette_and_lms("Mola_mola", meta, mean_c, sp_mod)

    fig = plt.figure(figsize=(16.2, 9.6), dpi=200, facecolor="#ffffff")
    gs = gridspec.GridSpec(
        2, 2, figure=fig, hspace=0.32, wspace=0.18,
        left=0.05, right=0.97, top=0.88, bottom=0.07
    )

    fig.suptitle(
        "Plate IV — Editorial Synthesis for Morphogenesis Research: Representation, Provenance, Numerical Checks & Bioelectric Intervention\n"
        "(Illustrating Citations [E6]–[E10] & Morpholib Connections 1–4 from the Annotated 1913 Lecture)",
        fontsize=12.4, fontweight="bold", color="#0f172a", y=0.96
    )

    # Panel A: Morpholib Connection 1 [E7] — Choosing What Counts as Form
    ax1 = fig.add_subplot(gs[0, 0])
    head_idx = [0, 1, 10, 11, 12, 13, 14, 15, 16]
    fin_idx = [2, 3, 4, 5, 6, 7, 8, 9]

    ax1.fill(BT_po[:, 0], BT_po[:, 1], color="#e2e8f0", alpha=0.65, label="Target Outline: Pomacanthus (600 pts)")
    ax1.plot(BT_po[:, 0], BT_po[:, 1], color="#475569", lw=1.8, ls="--")
    ax1.plot(BS_sc_lm[:, 0], BS_sc_lm[:, 1], color="#dc2626", lw=2.0, ls=":",
             label="17-Landmark Fit (9/17 in skull → IoU drops 0.662 → 0.618)")
    ax1.plot(BS_sc_sil[:, 0], BS_sc_sil[:, 1], color="#15803d", lw=2.5,
             label="Direct 600-pt Silhouette Fit (IoU improves to 0.749)")
    ax1.scatter(T_po[head_idx, 0], T_po[head_idx, 1], c="#f59e0b", edgecolors="#0f172a", s=45, zorder=5,
                label="9 Craniofacial Landmarks (53% weight)")
    ax1.scatter(T_po[fin_idx, 0], T_po[fin_idx, 1], c="#ec4899", edgecolors="#0f172a", marker="s", s=40, zorder=5,
                label="8 Fin-Base Insertions (0% on distal fins)")

    ax1.set_aspect("equal")
    ax1.set_xlim(-0.46, 0.52)
    ax1.set_ylim(-0.38, 0.38)
    ax1.set_xticks([]); ax1.set_yticks([])
    ax1.set_title(
        "A. Morpholib Connection 1 [E7]: Choosing What Counts as Form\n"
        "17 Skeletal Landmarks vs. 600-Point Whole-Body Silhouette (Scarus → Pomacanthus)",
        fontsize=10.2, fontweight="bold", color="#0f172a", pad=6
    )
    ax1.legend(loc="lower left", fontsize=7.5, frameon=True, facecolor="#f8fafc", edgecolor="#cbd5e1", ncol=1)

    # Panel B: Morpholib Connection 2 [E8] — The Provenance of an Outline
    ax2 = fig.add_subplot(gs[0, 1])
    dio_shift = np.array([-0.30, 0.0])
    mola_shift = np.array([0.28, 0.0])

    BS_dio_s = BS_dio_raw * 0.75 + dio_shift
    S_dio_s = S_dio_raw * 0.75 + dio_shift
    BT_mola_s = BT_mola * 0.72 + mola_shift
    T_mola_s = T_mola * 0.72 + mola_shift

    ax2.fill(BS_dio_s[:, 0], BS_dio_s[:, 1], color="#fef3c7", alpha=0.85)
    ax2.plot(BS_dio_s[:, 0], BS_dio_s[:, 1], color="#d97706", lw=2.3,
             label="Reconstructed Contour: build_diodon_silhouette_from_landmarks()\n(17 LMs + synthetic fin/caudal extensions + periodic B-spline)")
    ax2.scatter(S_dio_s[:, 0], S_dio_s[:, 1], c="#b45309", edgecolors="white", s=38, zorder=5)

    ax2.fill(BT_mola_s[:, 0], BT_mola_s[:, 1], color="#ccfbf1", alpha=0.85)
    ax2.plot(BT_mola_s[:, 0], BT_mola_s[:, 1], color="#0d9488", lw=2.3,
             label="Photograph-Derived Contour: extract_aligned_silhouette_and_lms()\n(LAB background threshold + morphological closing on 1887 plate)")
    ax2.scatter(T_mola_s[:, 0], T_mola_s[:, 1], c="#0f766e", edgecolors="white", s=38, zorder=5)

    ax2.axvline(0.0, color="#94a3b8", ls="--", lw=1.2)
    ax2.text(-0.30, 0.29, "Provenance Class A:\nLandmark-Interpolated Spline", ha="center", fontsize=8.4, fontweight="bold", color="#b45309")
    ax2.text(0.28, 0.29, "Provenance Class B:\nImage-Segmented Silhouette", ha="center", fontsize=8.4, fontweight="bold", color="#0f766e")

    ax2.set_aspect("equal")
    ax2.set_xlim(-0.62, 0.62)
    ax2.set_ylim(-0.36, 0.36)
    ax2.set_xticks([]); ax2.set_yticks([])
    ax2.set_title(
        "B. Morpholib Connection 2 [E8]: Distinct Provenance Labels for Contours\n"
        "Why 600 Sampled Points ≠ 600 Independent Biological Observations",
        fontsize=10.2, fontweight="bold", color="#0f172a", pad=6
    )
    ax2.legend(loc="lower center", fontsize=7.5, frameon=True, facecolor="#f8fafc", edgecolor="#cbd5e1")

    # Panel C: Morpholib Connection 3 [E9] — What a Mathematical Check Establishes
    ax3 = fig.add_subplot(gs[1, 0])
    lo = BS_sc.min(0)
    hi = BS_sc.max(0)
    xs = np.linspace(lo[0], hi[0], 90)
    ys = np.linspace(lo[1], hi[1], 90)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.column_stack([XX.ravel(), YY.ravel()])
    inside = MplPath(BS_sc).contains_points(pts)

    eps = 1e-4
    dx = np.array([eps, 0.0])
    dy = np.array([0.0, eps])
    grid_fn = lambda P: apply_sil(p_sil, P)
    eval_pts = np.vstack([pts[inside], BS_sc])
    dP_dx = (grid_fn(eval_pts + dx) - grid_fn(eval_pts - dx)) / (2.0 * eps)
    dP_dy = (grid_fn(eval_pts + dy) - grid_fn(eval_pts - dy)) / (2.0 * eps)
    det_J = dP_dx[:, 0] * dP_dy[:, 1] - dP_dy[:, 0] * dP_dx[:, 1]

    warped_inside = grid_fn(pts[inside])
    det_inside = det_J[: len(warped_inside)]
    sc_plot = ax3.scatter(warped_inside[:, 0], warped_inside[:, 1], c=det_inside, cmap="viridis", s=8, alpha=0.9)
    ax3.plot(BS_sc_sil[:, 0], BS_sc_sil[:, 1], color="#0f172a", lw=2.0, label="Warped 600-pt Contour Boundary")
    cbar = fig.colorbar(sc_plot, ax=ax3, fraction=0.042, pad=0.02)
    cbar.set_label(r"Sampled $\det D\Phi(x,y)$ (90×90 Grid)", fontsize=8.2)

    min_det = float(np.min(det_J))
    pct_pos = float(100.0 * np.mean(det_J > 0))
    ax3.annotate(
        f"Sampled Check (compute_jacobian_stats):\n"
        f"• Grid: 90×90 interior + 600 contour pts\n"
        f"• Positive det J: {pct_pos:.1f}% (min = {min_det:.3f})\n"
        f"• Scope note [E9]: Tests sampled local folding;\n"
        f"  does not alone prove global continuous injectivity",
        xy=(0.0, -0.25), xytext=(-0.44, -0.34),
        fontsize=7.6, color="#0f172a",
        bbox=dict(boxstyle="round,pad=0.3", fc="#f8fafc", ec="#0284c7", lw=1.1)
    )

    ax3.set_aspect("equal")
    ax3.set_xlim(-0.46, 0.52)
    ax3.set_ylim(-0.38, 0.38)
    ax3.set_xticks([]); ax3.set_yticks([])
    ax3.set_title(
        "C. Morpholib Connection 3 [E9]: What a Mathematical Check Establishes\n"
        "Finite-Difference Jacobian Determinant on a 90×90 Grid vs. Global Continuum Claims",
        fontsize=10.2, fontweight="bold", color="#0f172a", pad=6
    )

    # Panel D: Morpholib Connection 4 [E10] & Bioelectric Intervention [E6]
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.set_xlim(0, 10)
    ax4.set_ylim(0, 10)
    ax4.axis("off")
    ax4.set_title(
        "D. Morpholib Connection 4 [E10] & Bioelectric Intervention [E6]\n"
        "Separating Implementation, Empirical Validation & Perturbation Tests",
        fontsize=10.2, fontweight="bold", color="#0f172a", pad=6
    )

    ax4.text(5.0, 9.25, "MorphoEval Rubric Separation [E10] (Uncalibrated Ratings, Not Probabilities)",
             ha="center", va="center", fontsize=8.6, fontweight="bold", color="#0f172a")
    pillars = [
        (0.4, 6.2, 2.9, 2.5, "#e0f2fe", "#0284c7", "1. Implementation (30)",
         "• model-fidelity (20)\n• param-expressivity (10)\nDoes code match paper?"),
        (3.55, 6.2, 2.9, 2.5, "#dcfce7", "#15803d", "2. Bio Validation (40)",
         "• specimen-match (25)\n• morphometrics (15)\nDoes model fit organism?"),
        (6.7, 6.2, 2.9, 2.5, "#f3e8ff", "#7c3aed", "3. Reproducibility (30)",
         "• source-trace (10)\n• code-trace (8)\n• repro-package (12)"),
    ]
    for x0, y0, w, h, fc, ec, t_p, b_p in pillars:
        ax4.add_patch(mpatches.FancyBboxPatch(
            (x0, y0), w, h, boxstyle="round,pad=0.08,rounding_size=0.18", facecolor=fc, edgecolor=ec, lw=1.5
        ))
        ax4.text(x0 + w / 2, y0 + h - 0.38, t_p, ha="center", va="center", fontsize=8.0, fontweight="bold", color=ec)
        ax4.text(x0 + 0.16, y0 + 0.95, b_p, ha="left", va="center", fontsize=7.3, color="#1e293b")

    xen_img = load_specimen_img("datasets/species_gallery/I_Enaima_Vertebrata/Xenopus_laevis.jpg", size=(240, 180))
    ax_inset = ax4.inset_axes([0.04, 0.06, 0.30, 0.46])
    ax_inset.imshow(xen_img)
    ax_inset.set_xticks([]); ax_inset.set_yticks([])
    for sp in ax_inset.spines.values():
        sp.set_color("#15803d")
        sp.set_linewidth(1.4)
    ax_inset.set_title("Xenopus laevis [E6]", fontsize=7.8, fontweight="bold", color="#0f172a", pad=3)

    bio_box = mpatches.FancyBboxPatch(
        (3.6, 0.55), 6.0, 4.7,
        boxstyle="round,pad=0.1,rounding_size=0.18", facecolor="#f0fdf4", edgecolor="#15803d", lw=1.5
    )
    ax4.add_patch(bio_box)
    ax4.text(3.85, 4.75, "From Observation to Intervention (Pai et al. 2012, Dev. 139:313) [E6]",
             ha="left", va="center", fontsize=8.1, fontweight="bold", color="#166534")
    ax4.text(
        3.85, 2.55,
        "1. Record Pattern (Aristotle’s Step):\n"
        "   Map endogenous hyperpolarized $V_{\\mathrm{mem}}$ eye-field domain in early embryo.\n"
        "2. Propose Physical/Causal Law (Thompson’s Step):\n"
        "   Transmembrane voltage window controls Pax6 / Rx1 eye morphogenesis.\n"
        "3. Discriminating Perturbation Test (Modern Experimental Step):\n"
        "   Depolarizing native field disrupts normal eye; imposing hyperpolarized\n"
        "   $V_{\\mathrm{mem}}$ mRNA elsewhere induces ectopic eyes outside the head.",
        ha="left", va="center", fontsize=7.35, color="#1e293b"
    )

    out_path = MEDIA_DIR / "thompson_1913_fig4_morpholib_four_connections.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Saved {out_path}")


def main():
    build_fig1_spencer_vs_aristotle()
    build_fig2_lesbos_pyrrha_lagoon()
    build_fig3_cephalopod_embryology_soul()
    build_fig4_morpholib_four_connections()


if __name__ == "__main__":
    main()

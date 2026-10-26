#!/usr/bin/env python3
"""
Generate authentic museum-grade archival & copperplate plates for D'Arcy Wentworth
Thompson's 1913 Herbert Spencer Lecture at Oxford: 'On Aristotle as a Biologist,
With a Prooemion on Herbert Spencer', integrating the portraits, 17th–19th century
monograph engravings, and archival plates from both the 1913 lecture and the
Readable .docx edition.

Outputs (saved to docs/media/):
  1. thompson_1913_fig1_spencer_vs_aristotle.png
     - Museum Folio Plate I (Prooemion & Commentaries 1–3):
       (A) 1913 Clarendon Press Title Page (Oxford)
       (B) Moffat Cabinet Portrait of D'Arcy Wentworth Thompson (c. 1906, St Andrews)
       (C) Herbert Spencer Portrait (1820–1903)
       (D) Joseph Karl Stieler's Portrait of Goethe (1828)
       (E) P. J. F. Turpin's 'Végétal Type, Idéal, Appendiculé' (1837 Urpflanze engraving)
       (F) Culmann & H. Meyer's Crane-Head vs. Human Femur Trabeculae (On Growth and Form 1917, Fig. 335)
       (G) Thompson's Cantilever Bridge & Titanotherium Backbone Stress-Diagram (1917, Figs. 346 & 349)

  2. thompson_1913_fig2_lesbos_pyrrha_lagoon.png
     - Museum Folio Plate II (Aristotle on Lesbos & the Lagoon of Pyrrha, pp. 12–14):
       (A) Royal Navy Admiralty Chart No. 1665 of Mityleni Island (Lesbos) & Port Kalloni
           (ancient Lagoon of Pyrrha), surveyed by Capt. R. Copeland R.N. (1832/1848)
       (B) Palazzo Altemps Classical Marble Bust of Aristotle (after Lysippos)
       (C) Haeckel's Tafel 40 (Asteridea / Starfishes of the Lagoon, 1904 lithograph)
       (D) Nicolas Robert's 17th-century vellum painting of Pecten jacobaeus (Scallop)
       (E) Wenceslas Hollar's 17th-century copperplate etching of Murex brandaris (Cape Lectum)
       (F) Duméril & Bibron's 19th-century lithograph of Chamaeleo chamaeleon skeleton (Asiatic coast)

  3. thompson_1913_fig3_cephalopod_embryology_soul.png
     - Museum Folio Plate III (Anatomia Animata & De Anima, pp. 17–29):
       (A) J. A. Herklots (1859) lithograph of Sepia officinalis (showing two long anchor tentacles)
       (B) J.-B. Vérany (1851) chromolithograph of Argonauta argo (Plate 17, Mollusques méditerranéens)
       (C) Cuvier (1829) engraving of the detached male Hectocotylus arm
       (D) Johannes Müller (1842) Tafel III from 'Über den glatten Hai des Aristoteles' showing
           Mustelus laevis (γαλεὸς λεῖος) embryo attached to the maternal placenta vitellina
       (E) Marcello Malpighi (1673/1687) 'De formatione pulli in ovo' copperplate of the Day-3
           chick embryo punctum saliens, area vasculosa, and looping heart tubes
       (F) Didacus Valades (1579) Renaissance copperplate of the Scala Naturae (Great Chain of Being)
       (G) 13th/14th-century illuminated manuscript leaf of Aristotle's De Anima (Vaticanus Pal. lat. 1033)
       (H) Classical Geometric Engraving of Aristotle's Nested Psychē ('as a triangle within a square',
           De Anima II.3) and First vs. Second Actuality (ἐντελέχεια ἡ πρώτη)

  4. thompson_1913_fig4_morpholib_four_connections.png
     - Museum & Computational Synthesis Plate IV (Editorial Synthesis & Morpholib Connections 1–4):
       Pairs D'Arcy Thompson's original 1917 woodcuts (Figs. 375–376 Scarus -> Pomacanthus and
       Figs. 381–382 Diodon -> Orthagoriscus/Mola) and British Museum 1887 engraving with the
       4 Morpholib quantitative verification panels ([E6]–[E10]).
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import shutil

import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.path import Path as MplPath
import numpy as np
from PIL import Image, ImageOps

REPO_ROOT = Path(__file__).resolve().parents[1]
MEDIA_DIR = REPO_ROOT / "docs" / "media"
CACHE_1913 = REPO_ROOT / ".thompson_cache" / "aristotle_1913"
CACHE_DOCX = REPO_ROOT / ".thompson_cache" / "aristotle_docx"
CACHE_MAIN = REPO_ROOT / ".thompson_cache"
MEDIA_DIR.mkdir(parents=True, exist_ok=True)

PARCHMENT_BG = "#f7f3e8"
CARD_BG = "#fcfaf2"
BORDER_GOLD = "#8c6d3f"
INK_DARK = "#1c1917"
INK_MUTED = "#44403c"


def load_module(name: str, rel_path: str):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / rel_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def open_fitted(path: Path, box_size: tuple[int, int] = (500, 500), pad_color=(252, 250, 242), crop_frac: tuple[float, float, float, float] | None = None) -> np.ndarray:
    if not path.exists():
        return np.full((box_size[1], box_size[0], 3), 245, dtype=np.uint8)
    im = Image.open(path).convert("RGB")
    if crop_frac is not None:
        w, h = im.size
        x0, y0, x1, y1 = crop_frac
        im = im.crop((int(x0 * w), int(y0 * h), int(x1 * w), int(y1 * h)))
    im = ImageOps.contain(im, box_size, Image.LANCZOS)
    canvas = Image.new("RGB", box_size, pad_color)
    ox = (box_size[0] - im.width) // 2
    oy = (box_size[1] - im.height) // 2
    canvas.paste(im, (ox, oy))
    return np.asarray(canvas)


def style_museum_ax(ax, title: str, subtitle: str = "", border_color: str = BORDER_GOLD):
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_color(border_color)
        sp.set_linewidth(1.5)
    ax.set_title(title, fontsize=9.8, fontweight="bold", fontfamily="serif", color=INK_DARK, pad=5)
    if subtitle:
        ax.set_xlabel(subtitle, fontsize=8.0, fontfamily="serif", color=INK_MUTED, labelpad=4)


def copy_standalone_museum_assets():
    """Also copy key standalone museum portraits/plates into docs/media/ for inline embedding."""
    pairs = [
        (CACHE_DOCX / "image1.jpg", MEDIA_DIR / "thompson_1906_moffat_portrait.jpg"),
        (CACHE_DOCX / "image2.jpg", MEDIA_DIR / "goethe_1828_stieler_portrait.jpg"),
        (CACHE_DOCX / "turpin_1837_urpflanze.jpg", MEDIA_DIR / "turpin_1837_urpflanze_plate.jpg"),
        (CACHE_1913 / "muller1842_n84.jpg", MEDIA_DIR / "muller_1842_mustelus_placenta_taf3.jpg"),
        (CACHE_1913 / "malpighi_chick_1687.jpg", MEDIA_DIR / "malpighi_1687_pulli_in_ovo_plate.jpg"),
        (CACHE_1913 / "argonauta_verany_1851.jpg", MEDIA_DIR / "verany_1851_argonauta_plate17.jpg"),
        (CACHE_1913 / "lesbos_admiralty_1860.jpg", MEDIA_DIR / "lesbos_admiralty_chart_1860.jpg"),
    ]
    for src, dst in pairs:
        if src.exists():
            shutil.copy2(src, dst)


def build_fig1_spencer_vs_aristotle():
    """
    Plate I: Prooemion on Herbert Spencer, Goethe's Morphology, and Physical Form (1913 -> 1917).
    """
    fig = plt.figure(figsize=(16.5, 9.4), dpi=200, facecolor=PARCHMENT_BG)
    gs = gridspec.GridSpec(
        2, 4, figure=fig, height_ratios=[1.0, 1.08], width_ratios=[0.92, 0.92, 0.92, 1.24],
        hspace=0.34, wspace=0.18, left=0.03, right=0.97, top=0.88, bottom=0.07
    )

    fig.suptitle(
        "PLATE I — PROOEMION ON HERBERT SPENCER, GOETHE’S MORPHOLOGY & PHYSICAL STRUCTURAL MECHANICS\n"
        "Archival Portraits, the 1913 Oxford Title Page, Turpin’s 1837 Urpflanze & Culmann’s Graphic Statics in On Growth and Form (1917)",
        fontsize=12.6, fontweight="bold", fontfamily="serif", color=INK_DARK, y=0.96
    )

    # Row 1, Col 0: 1913 Clarendon Press Title Page
    ax00 = fig.add_subplot(gs[0, 0])
    img_title = open_fitted(CACHE_1913 / "thompson1913_title_n4.jpg", (420, 560), crop_frac=(0.06, 0.04, 0.94, 0.95))
    ax00.imshow(img_title)
    style_museum_ax(
        ax00,
        "A. Oxford Clarendon Press (1913)",
        "Original title page of the Herbert Spencer\nLecture delivered Feb. 14, 1913"
    )

    # Row 1, Col 1: D'Arcy Wentworth Thompson (c. 1906 Moffat Cabinet Card)
    ax01 = fig.add_subplot(gs[0, 1])
    img_dwt = open_fitted(CACHE_DOCX / "image1.jpg", (420, 560))
    ax01.imshow(img_dwt)
    style_museum_ax(
        ax01,
        "B. D’Arcy W. Thompson (c. 1906)",
        "Cabinet portrait by Moffat, Edinburgh;\nSt Andrews ms48534-ph-2-5 [E1, E12]"
    )

    # Row 1, Col 2: Herbert Spencer (1820–1903)
    ax02 = fig.add_subplot(gs[0, 2])
    img_spencer = open_fitted(CACHE_1913 / "spencer_portrait.jpg", (420, 560))
    ax02.imshow(img_spencer)
    style_museum_ax(
        ax02,
        "C. Herbert Spencer (1820–1903)",
        "Railway bridge-builder on the Birmingham–\nCrewe line & author of Synthetic Philosophy"
    )

    # Row 2, Col 0: Goethe (1828 Stieler Portrait)
    ax10 = fig.add_subplot(gs[1, 0])
    img_goethe = open_fitted(CACHE_DOCX / "image2.jpg", (420, 520))
    ax10.imshow(img_goethe)
    style_museum_ax(
        ax10,
        "D. J. W. von Goethe (Stieler, 1828)",
        "Author of Zur Morphologie (1795/1807):\n‘subordination of parts’ (pp. 6–7) [E3, E13]"
    )

    # Row 2, Col 1: Turpin's 1837 Ideal Plant (Urpflanze)
    ax11 = fig.add_subplot(gs[1, 1])
    img_turpin = open_fitted(CACHE_DOCX / "turpin_1837_urpflanze.jpg", (420, 520), crop_frac=(0.04, 0.02, 0.96, 0.96))
    ax11.imshow(img_turpin)
    style_museum_ax(
        ax11,
        "E. Turpin’s Ideal Plant (1837)",
        "Végétal Type, Idéal, Appendiculé illustrating\nGoethe’s organ metamorphosis [E4, E14]"
    )

    # Row 2, Col 2: Culmann's Crane-Head vs. Human Femur Trabeculae (On Growth and Form 1917, Fig. 335)
    ax12 = fig.add_subplot(gs[1, 2])
    img_femur = open_fitted(CACHE_1913 / "gutenberg_i682.png", (420, 520), pad_color=(255, 255, 255))
    ax12.imshow(img_femur)
    style_museum_ax(
        ax12,
        "F. Crane-Head & Femur (1917, Fig. 335)",
        "Culmann & H. Meyer’s stress trajectories\nin bone trabeculae (pp. 7–8, Comm. 2) [E1]"
    )

    # Col 3 (spanning both rows): Bridge-Builder Cantilevers (1917 Figs. 341 & 349) + Scaling Law (Note 9)
    gs_right = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[:, 3], height_ratios=[0.95, 1.05], hspace=0.32)

    ax_bridge = fig.add_subplot(gs_right[0, 0])
    im_b1 = Image.open(CACHE_1913 / "gutenberg_i696b.png").convert("RGB") if (CACHE_1913 / "gutenberg_i696b.png").exists() else Image.open(CACHE_1913 / "gutenberg_i700b.png").convert("RGB")
    im_b2 = Image.open(CACHE_1913 / "gutenberg_i704.png").convert("RGB")
    w_comp = 560
    h_comp = 360
    comp_canvas = Image.new("RGB", (w_comp, h_comp), (255, 255, 255))
    b1_fit = ImageOps.contain(im_b1, (w_comp - 20, 150), Image.LANCZOS)
    b2_fit = ImageOps.contain(im_b2, (w_comp - 20, 175), Image.LANCZOS)
    comp_canvas.paste(b1_fit, ((w_comp - b1_fit.width) // 2, 12))
    comp_canvas.paste(b2_fit, ((w_comp - b2_fit.width) // 2, 175))
    ax_bridge.imshow(np.asarray(comp_canvas))
    style_museum_ax(
        ax_bridge,
        "G. Bridge-Builder Statics in Anatomy (1917, Figs. 346 & 349)",
        "Cantilever truss & stress-diagram of the quadrupedal spine (p. 8, Note 8)"
    )

    ax_scale = fig.add_subplot(gs_right[1, 0])
    ax_scale.set_facecolor(CARD_BG)
    L = np.linspace(0.5, 4.0, 200)
    ax_scale.plot(L, (L ** 2) / 16.0 * 100, color="#1d4ed8", lw=2.3, label=r"Supporting Cross-Section Area ($\propto L^2$)")
    ax_scale.plot(L, (L ** 3) / 64.0 * 100, color="#b91c1c", lw=2.3, ls="--", label=r"Gravitational Bulk / Weight ($\propto L^3$)")
    ax_scale.plot(L, (L ** 0.5) / 2.0 * 68, color="#0f766e", lw=2.1, ls="-.", label=r"Aquatic Speed in Buoyant Water (Note 9)")
    ax_scale.axvline(2.4, color=BORDER_GOLD, ls=":", lw=1.4)
    ax_scale.annotate(
        "Terrestrial & Avian\nSize Limit\n(Princ. Biol. i. 156)",
        xy=(2.4, 48), xytext=(0.75, 62),
        fontsize=8.0, fontfamily="serif", fontweight="bold", color="#7f1d1d",
        bbox=dict(boxstyle="round,pad=0.28", fc="#fef2f2", ec="#b91c1c", lw=0.9),
        arrowprops=dict(arrowstyle="->", color="#b91c1c", lw=1.1)
    )
    ax_scale.set_xlim(0.5, 4.0)
    ax_scale.set_ylim(0, 105)
    ax_scale.set_xlabel("Relative Linear Scale $L$ (Commentary 2 / Reader Note 6)", fontsize=8.2, fontfamily="serif", color=INK_DARK)
    ax_scale.set_ylabel("Normalized Magnitude (%)", fontsize=8.2, fontfamily="serif", color=INK_DARK)
    ax_scale.set_title("H. Spencer’s Scaling Corollary (p. 8, Note 9)", fontsize=9.8, fontweight="bold", fontfamily="serif", color=INK_DARK, pad=5)
    ax_scale.grid(True, ls=":", alpha=0.45, color="#a8a29e")
    ax_scale.legend(loc="upper left", fontsize=7.3, frameon=True, facecolor=PARCHMENT_BG, edgecolor=BORDER_GOLD)
    for sp in ax_scale.spines.values():
        sp.set_color(BORDER_GOLD)
        sp.set_linewidth(1.5)

    out_path = MEDIA_DIR / "thompson_1913_fig1_spencer_vs_aristotle.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Saved {out_path}")


def build_fig2_lesbos_pyrrha_lagoon():
    """
    Plate II: Insula Nobilis et Amoena — The 1848/1860 Admiralty Chart of Mityleni (Lesbos)
    & Port Kalloni (Lagoon of Pyrrha) paired with 17th–19th century museum engravings.
    """
    fig = plt.figure(figsize=(16.5, 9.2), dpi=200, facecolor=PARCHMENT_BG)
    gs = gridspec.GridSpec(
        2, 4, figure=fig, width_ratios=[1.28, 1.28, 1.0, 1.0],
        hspace=0.32, wspace=0.16, left=0.03, right=0.97, top=0.88, bottom=0.07
    )

    fig.suptitle(
        "PLATE II — INSULA NOBILIS ET AMOENA: ADMIRALTY CHART OF LESBOS (PYRRHA LAGOON) & MUSEUM SPECIMEN PLATES\n"
        "The 1848/1860 Hydrographic Survey of Mityleni & Port Kalloni alongside 17th–19th Century Engravings of Aristotle’s Lagoon Fauna (pp. 12–14)",
        fontsize=12.4, fontweight="bold", fontfamily="serif", color=INK_DARK, y=0.96
    )

    # Left 2x2 span: 1848/1860 Admiralty Chart No. 1665 of Mityleni Island (Lesbos) with Port Kalloni (Pyrrha) annotated
    ax_chart = fig.add_subplot(gs[:, :2])
    chart_im = Image.open(CACHE_1913 / "lesbos_admiralty_1860.jpg").convert("RGB")
    cw, ch = chart_im.size
    ax_chart.imshow(np.asarray(chart_im))

    # Annotate Port Kalloni (Lagoon of Pyrrha), Cape Malia, and Entrance inset on the Admiralty Chart
    # In Admiralty Chart 1665: Port Kalloni runs into the center of the island around (0.43*cw, 0.46*ch)
    ax_chart.scatter([0.43 * cw], [0.45 * ch], s=180, facecolors="none", edgecolors="#b91c1c", lw=2.6, zorder=6)
    ax_chart.annotate(
        "PORT KALLONI — ANCIENT LAGOON OF PYRRHA\n"
        "‘Far into its bosom, by the little town of Pyrrha,\n"
        "runs a broad and sheltered lagoon’ (pp. 12–13):\n"
        "• Starfish pest & scallop drought/dredging\n"
        "• Winter-breeding sea-urchins & breeding fish migration",
        xy=(0.43 * cw, 0.45 * ch), xytext=(0.30 * cw, 0.12 * ch),
        fontsize=8.3, fontfamily="serif", fontweight="bold", color="#7f1d1d",
        bbox=dict(boxstyle="round,pad=0.38", fc="#fef2f2", ec="#b91c1c", lw=1.5),
        arrowprops=dict(arrowstyle="->,head_width=0.35", color="#b91c1c", lw=1.8),
        zorder=7
    )
    # Inset chart on the left of the Admiralty sheet: "ENTRANCE TO PORT KALLONI"
    ax_chart.annotate(
        "Admiralty Inset:\nEntrance to Port Kalloni\n(Lagoon Channel)",
        xy=(0.12 * cw, 0.50 * ch), xytext=(0.04 * cw, 0.78 * ch),
        fontsize=7.8, fontfamily="serif", fontweight="bold", color="#1e3a8a",
        bbox=dict(boxstyle="round,pad=0.3", fc="#eff6ff", ec="#1d4ed8", lw=1.2),
        arrowprops=dict(arrowstyle="->", color="#1d4ed8", lw=1.5),
        zorder=7
    )
    # Cape Malia (SE promontory near Mitylene)
    ax_chart.annotate(
        "Mitylene & Cape Malia\n(Landward vs. seaward sponges, p. 12)",
        xy=(0.68 * cw, 0.67 * ch), xytext=(0.54 * cw, 0.85 * ch),
        fontsize=7.8, fontfamily="serif", fontweight="bold", color="#78350f",
        bbox=dict(boxstyle="round,pad=0.3", fc="#fffbeb", ec="#b45309", lw=1.2),
        arrowprops=dict(arrowstyle="->", color="#b45309", lw=1.5),
        zorder=7
    )
    style_museum_ax(
        ax_chart,
        "A. British Admiralty Chart No. 1665: Mityleni Island (Lesbos) & Gulfs of Adramyti and Sandarli (Surveyed 1832–1849)",
        "Showing the sheltered Lagoon of Pyrrha (Port Kalloni), the narrow seaward channel, Mitylene, Cape Malia, and the Mysian/Trojan coast (pp. 12–14)"
    )

    # Right 2x2 grid: 4 Museum Plates
    # R0, C2: Palazzo Altemps Bust of Aristotle + Haeckel Asteridea
    ax_ar = fig.add_subplot(gs[0, 2])
    img_ar = open_fitted(CACHE_1913 / "aristotle_bust.jpg", (420, 440), crop_frac=(0.05, 0.02, 0.95, 0.95))
    ax_ar.imshow(img_ar)
    style_museum_ax(
        ax_ar,
        "B. Bust of Aristotle (Palazzo Altemps)",
        "Pentelic marble copy after Lysippos;\n‘first and foremost a biologist’ (p. 11)"
    )

    # R0, C3: Haeckel Tafel 40 — Asteridea (Starfishes)
    ax_st = fig.add_subplot(gs[0, 3])
    img_st = open_fitted(CACHE_1913 / "haeckel_asteridea_40.jpg", (420, 440), crop_frac=(0.04, 0.04, 0.96, 0.96))
    ax_st.imshow(img_st)
    style_museum_ax(
        ax_st,
        "C. Starfishes — Asteridea (Tafel 40)",
        "Ernst Haeckel (1904) lithograph; abundant\nin Pyrrha as ‘a pest to fishermen’ (p. 13)"
    )

    # R1, C2: Nicolas Robert (17th c.) Scallop (Pecten jacobaeus) + Wenceslas Hollar (17th c.) Murex
    ax_sh = fig.add_subplot(gs[1, 2])
    im_pec = Image.open(CACHE_1913 / "robert_pecten.jpg").convert("RGB")
    im_mur = Image.open(CACHE_1913 / "hollar_murex.jpg").convert("RGB")
    sh_canvas = Image.new("RGB", (420, 440), (252, 250, 242))
    pec_fit = ImageOps.contain(im_pec, (400, 210), Image.LANCZOS)
    mur_fit = ImageOps.contain(im_mur, (400, 200), Image.LANCZOS)
    sh_canvas.paste(pec_fit, ((420 - pec_fit.width) // 2, 8))
    sh_canvas.paste(mur_fit, ((420 - mur_fit.width) // 2, 228))
    ax_sh.imshow(np.asarray(sh_canvas))
    style_museum_ax(
        ax_sh,
        "D. Pecten (Robert) & Murex (Hollar, 17th c.)",
        "Top: Pecten jacobaeus (dredged in Pyrrha);\nBottom: Murex etching (Cape Lectum, p. 12)"
    )

    # R1, C3: Chameleon Skeleton Engraving (Duméril & Bibron, Erpétologie générale)
    ax_ch = fig.add_subplot(gs[1, 3])
    img_ch = open_fitted(CACHE_1913 / "chameleon_historical.jpg", (420, 440), pad_color=(255, 255, 255))
    ax_ch.imshow(img_ch)
    style_museum_ax(
        ax_ch,
        "E. Chameleon Skeleton (Duméril, 1836)",
        "Erpétologie générale copperplate; native to\nthe Asiatic coast, dissected by Aristotle (p. 13)"
    )

    out_path = MEDIA_DIR / "thompson_1913_fig2_lesbos_pyrrha_lagoon.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Saved {out_path}")


def build_fig3_cephalopod_embryology_soul():
    """
    Plate III: Anatomia Animata — Original 16th–19th Century Copperplates of Aristotle's
    Zoological & Embryological Discoveries and the Architecture of De Anima.
    """
    fig = plt.figure(figsize=(16.5, 9.6), dpi=200, facecolor=PARCHMENT_BG)
    gs = gridspec.GridSpec(
        2, 4, figure=fig, height_ratios=[1.04, 1.0], width_ratios=[1.0, 1.0, 1.05, 1.05],
        hspace=0.34, wspace=0.16, left=0.03, right=0.97, top=0.88, bottom=0.07
    )

    fig.suptitle(
        "PLATE III — ANATOMIA ANIMATA: HISTORICAL COPPERPLATES OF ARISTOTLE’S EMBRYOLOGY & THE ARCHITECTURE OF DE ANIMA\n"
        "Herklots (1859), Vérany (1851), Cuvier (1829), Johannes Müller’s Mustelus Placenta (1842), Malpighi’s Chick (1673/1687) & De Anima (pp. 17–29)",
        fontsize=12.2, fontweight="bold", fontfamily="serif", color=INK_DARK, y=0.96
    )

    # R0, C0: Herklots (1859) Sepia officinalis + Cuvier (1829) Hectocotylus arm
    ax00 = fig.add_subplot(gs[0, 0])
    im_sep = Image.open(CACHE_1913 / "sepia_herklots_1859.jpg").convert("RGB")
    im_hec = Image.open(CACHE_1913 / "hectocotylus_cuvier.jpg").convert("RGB")
    c00 = Image.new("RGB", (440, 460), (255, 255, 255))
    s_fit = ImageOps.contain(im_sep, (240, 440), Image.LANCZOS)
    h_fit = ImageOps.contain(im_hec, (180, 440), Image.LANCZOS)
    c00.paste(s_fit, (8, (460 - s_fit.height) // 2))
    c00.paste(h_fit, (252, (460 - h_fit.height) // 2))
    ax00.imshow(np.asarray(c00))
    style_museum_ax(
        ax00,
        "A. Sepia (1859) & Hectocotylus (1829)",
        "Left: Sepia officinalis with 2 long anchor arms;\nRight: Detached male hectocotylus arm (pp. 17–19)"
    )

    # R0, C1: Vérany (1851) Argonauta argo chromolithograph (Plate 17)
    ax01 = fig.add_subplot(gs[0, 1])
    img_arg = open_fitted(CACHE_1913 / "argonauta_verany_1851.jpg", (440, 460), crop_frac=(0.03, 0.03, 0.97, 0.97))
    ax01.imshow(img_arg)
    style_museum_ax(
        ax01,
        "B. Argonauta argo (Vérany, 1851, Pl. 17)",
        "Mollusques méditerranéens chromolithograph\nshowing the two webbed dorsal arms (p. 19)"
    )

    # R0, C2: Johannes Müller (1842) Tafel III — Mustelus laevis (γαλεὸς λεῖος) Placenta Vitellina
    ax02 = fig.add_subplot(gs[0, 2])
    img_mul = open_fitted(CACHE_1913 / "muller1842_n84.jpg", (460, 460), crop_frac=(0.04, 0.04, 0.96, 0.95))
    ax02.imshow(img_mul)
    style_museum_ax(
        ax02,
        "C. Johannes Müller’s Tafel III (1842)",
        "Über den glatten Hai des Aristoteles: Mustelus\nlaevis embryo & yolk-sac placenta (pp. 20–21)"
    )

    # R0, C3: Marcello Malpighi (1673/1687) De formatione pulli in ovo (Chick Embryo Punctum Saliens)
    ax03 = fig.add_subplot(gs[0, 3])
    img_mal = open_fitted(CACHE_1913 / "malpighi_chick_1687.jpg", (460, 460), crop_frac=(0.03, 0.03, 0.97, 0.97))
    ax03.imshow(img_mal)
    style_museum_ax(
        ax03,
        "D. Malpighi’s Pulli in Ovo (1673/1687)",
        "Royal Society copperplate of the chick embryo,\nvitelline vessels & looping heart (pp. 21–23)"
    )

    # R1, C0: Müller & Henle (1841) Mustelus vulgaris vs. Mustelus laevis
    ax10 = fig.add_subplot(gs[1, 0])
    img_mh = open_fitted(CACHE_1913 / "mustelus_muller_henle_1841.jpg", (440, 430), crop_frac=(0.02, 0.05, 0.98, 0.95))
    ax10.imshow(img_mh)
    style_museum_ax(
        ax10,
        "E. Mustelus vulgaris vs. M. laevis (1841)",
        "Müller & Henle Plate 27: the two almost indis-\ntinguishable sharks (only M. laevis is placental)"
    )

    # R1, C1: Illuminated Manuscript of Aristotle's De Anima (Vaticanus Palatinus latinus 1033)
    ax11 = fig.add_subplot(gs[1, 1])
    img_ms = open_fitted(CACHE_1913 / "aristotle_de_anima_manuscript.jpg", (440, 430), crop_frac=(0.05, 0.02, 0.95, 0.95))
    ax11.imshow(img_ms)
    style_museum_ax(
        ax11,
        "F. Aristotle’s De Anima (Vat. Pal. lat. 1033)",
        "Medieval illuminated parchment leaf of On the\nSoul: biology as the foundation of psychology"
    )

    # R1, C2: Valades (1579) Scala Naturae (Great Chain of Being)
    ax12 = fig.add_subplot(gs[1, 2])
    img_sc = open_fitted(CACHE_1913 / "scala_naturae_bonnet_or_bovillus.jpg", (460, 430), crop_frac=(0.06, 0.03, 0.94, 0.97))
    ax12.imshow(img_sc)
    style_museum_ax(
        ax12,
        "G. Scala Naturae Engraving (Valades, 1579)",
        "Rhetorica Christiana copperplate of the ladder of\nnature (plants → animals → man, pp. 26–28)"
    )

    # R1, C3: Classical Euclidean Woodcut Diagram of De Anima II.3 ("Triangle within a Square") & First Actuality
    ax13 = fig.add_subplot(gs[1, 3])
    ax13.set_facecolor(CARD_BG)
    ax13.set_xlim(0, 10)
    ax13.set_ylim(0, 10)

    # Classical woodcut-style double border inside panel H
    ax13.add_patch(mpatches.Rectangle((0.35, 0.35), 9.3, 9.3, fill=False, edgecolor=INK_DARK, lw=1.2))
    ax13.add_patch(mpatches.Rectangle((0.50, 0.50), 9.0, 9.0, fill=False, edgecolor=BORDER_GOLD, lw=0.8))

    ax13.text(5.0, 8.95, "DE ANIMA II.3 (414b20–32) — GEOMETRICAL INCLUSION",
              ha="center", va="center", fontsize=7.6, fontfamily="serif", fontweight="bold", color=INK_DARK)

    # Concentric classical woodcut frames: Pentagon/Square containing Triangle
    # Outer frame: Rational (Dianoetike / Nous)
    ax13.add_patch(mpatches.Rectangle((0.9, 4.1), 8.2, 4.3, facecolor="#f5f0e1", edgecolor=INK_DARK, lw=1.6))
    ax13.text(5.0, 7.95, "III. ΔΙΑΝΟΗΤΙΚΗ ΨΥΧΗ / ΝΟΥΣ (Rational — Humans)",
              ha="center", va="center", fontsize=7.5, fontfamily="serif", fontweight="bold", color=INK_DARK)

    # Middle frame: Sensitive (Aisthetike)
    ax13.add_patch(mpatches.Rectangle((1.45, 4.35), 7.1, 3.1, facecolor="#ebe3ce", edgecolor=INK_DARK, lw=1.4, ls="--"))
    ax13.text(5.0, 7.05, "II. ΑΙΣΘΗΤΙΚΗ ΨΥΧΗ (Sensitive — Animals & Humans)",
              ha="center", va="center", fontsize=7.3, fontfamily="serif", fontweight="bold", color=INK_DARK)

    # Inner frame: Nutritive (Threptike / Prote Psyche) with inscribed Triangle-in-Square emblem
    ax13.add_patch(mpatches.Rectangle((2.0, 4.6), 6.0, 2.0, facecolor="#dfd4b8", edgecolor=INK_DARK, lw=1.3))
    ax13.text(5.0, 6.12, "I. ΠΡΩΤΗ / ΘΡΕΠΤΙΚΗ ΨΥΧΗ (Nutritive — All Life)",
              ha="center", va="center", fontsize=7.2, fontfamily="serif", fontweight="bold", color=INK_DARK)
    ax13.text(5.0, 5.20, "Nourishment · Growth · Reproduction\n‘As the triangle is implied in the quadrangle’ (p. 27)",
              ha="center", va="center", fontsize=6.9, fontfamily="serif", style="italic", color=INK_MUTED)

    # Bottom section of H: Potentiality -> First Actuality (Entelecheia) -> Second Actuality (Figures 5 & 6 of .docx)
    ax13.plot([0.8, 9.2], [3.65, 3.65], color=BORDER_GOLD, lw=1.0)
    ax13.text(5.0, 3.25, "ΕΝΤΕΛΕΧΕΙΑ Η ΠΡΩΤΗ (De Anima II.1, 412a27 — Reader Note 23)",
              ha="center", va="center", fontsize=7.3, fontfamily="serif", fontweight="bold", color=INK_DARK)

    stages = [
        (2.0, 1.85, "1. Δύναμις\n(Potentiality)\nAble to learn", "#f5f0e1"),
        (5.0, 1.85, "2. Ἐντελέχεια Α΄\n(First Actuality)\nPossesses capacity", "#dfd4b8"),
        (8.0, 1.85, "3. Ἐνέργεια / Β΄\n(Second Actuality)\nActively exercising", "#efe8d4"),
    ]
    for sx, sy, stxt, sfc in stages:
        ax13.add_patch(mpatches.FancyBboxPatch(
            (sx - 1.18, sy - 0.85), 2.36, 1.7,
            boxstyle="round,pad=0.06,rounding_size=0.12", facecolor=sfc, edgecolor=INK_DARK, lw=1.2
        ))
        ax13.text(sx, sy, stxt, ha="center", va="center", fontsize=6.6, fontfamily="serif", color=INK_DARK)
    ax13.annotate("", xy=(3.78, 1.85), xytext=(3.22, 1.85), arrowprops=dict(arrowstyle="->", color=INK_DARK, lw=1.3))
    ax13.annotate("", xy=(6.78, 1.85), xytext=(6.22, 1.85), arrowprops=dict(arrowstyle="->", color=INK_DARK, lw=1.3))

    style_museum_ax(
        ax13,
        "H. Nested Psychē & First Actuality (De Anima II.1–3)",
        "Classical restatement of Figures 5 & 6 (Reader Notes 18–23):\ninclusive capacities & first vs. second actuality [E4, E9]"
    )

    out_path = MEDIA_DIR / "thompson_1913_fig3_cephalopod_embryology_soul.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Saved {out_path}")


def build_fig4_morpholib_four_connections():
    """
    Plate IV: Editorial Synthesis for Morphogenesis Research — Pairing Thompson's
    original 1917 woodcuts & museum lithographs with the 4 Morpholib Connections ([E6]–[E10]).
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

    fig = plt.figure(figsize=(16.5, 9.6), dpi=200, facecolor=PARCHMENT_BG)
    gs = gridspec.GridSpec(
        2, 2, figure=fig, hspace=0.34, wspace=0.18,
        left=0.04, right=0.97, top=0.88, bottom=0.07
    )

    fig.suptitle(
        "PLATE IV — FROM 1917 WOODCUTS TO COMPUTATIONAL MORPHOSPACE: THE FOUR MORPHOLIB CONNECTIONS\n"
        "Thompson’s Original 1917 Engravings (Figs. 375–376 & 381–382) Paired with Representation, Provenance, Jacobian & Bioelectric Audits [E6–E10]",
        fontsize=12.2, fontweight="bold", fontfamily="serif", color=INK_DARK, y=0.96
    )

    # Panel A: Morpholib Connection 1 [E7] — 1917 Figs. 375–376 Woodcut + 17-LM vs 600-pt Silhouette
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_facecolor(CARD_BG)
    head_idx = [0, 1, 10, 11, 12, 13, 14, 15, 16]
    fin_idx = [2, 3, 4, 5, 6, 7, 8, 9]

    # Inset of Thompson's original 1917 Fig. 375-376 woodcut on the left of Panel A
    ax1_in = ax1.inset_axes([0.02, 0.14, 0.38, 0.78])
    img_375 = open_fitted(CACHE_1913 / "gutenberg_i749.png", (360, 320), pad_color=(255, 255, 255))
    ax1_in.imshow(img_375)
    ax1_in.set_xticks([]); ax1_in.set_yticks([])
    for sp in ax1_in.spines.values():
        sp.set_color(BORDER_GOLD)
        sp.set_linewidth(1.2)
    ax1_in.set_title("1917 Woodcut (Figs. 375–376)\nScarus → Pomacanthus", fontsize=7.6, fontfamily="serif", fontweight="bold", color=INK_DARK, pad=3)

    # Right side of Panel A: Quantitative overlay
    offset_x = 0.24
    ax1.fill(BT_po[:, 0] + offset_x, BT_po[:, 1], color="#e7e5e4", alpha=0.8, label="Target: Pomacanthus (600-pt outline)")
    ax1.plot(BT_po[:, 0] + offset_x, BT_po[:, 1], color="#44403c", lw=1.7, ls="--")
    ax1.plot(BS_sc_lm[:, 0] + offset_x, BS_sc_lm[:, 1], color="#b91c1c", lw=2.0, ls=":",
             label="17-Landmark Fit (9/17 in skull → IoU 0.618)")
    ax1.plot(BS_sc_sil[:, 0] + offset_x, BS_sc_sil[:, 1], color="#15803d", lw=2.4,
             label="Direct 600-pt Outline Fit (IoU 0.749)")
    ax1.scatter(T_po[head_idx, 0] + offset_x, T_po[head_idx, 1], c="#d97706", edgecolors=INK_DARK, s=40, zorder=5,
                label="9 Craniofacial LMs (53% weight)")
    ax1.scatter(T_po[fin_idx, 0] + offset_x, T_po[fin_idx, 1], c="#be185d", edgecolors=INK_DARK, marker="s", s=36, zorder=5,
                label="8 Fin-Base LMs (0% on distal fins)")

    ax1.set_aspect("equal")
    ax1.set_xlim(-0.56, 0.72)
    ax1.set_ylim(-0.38, 0.38)
    style_museum_ax(
        ax1,
        "A. Morpholib Connection 1 [E7]: Choosing What Counts as Form (Figure 4 of .docx)",
        "Why fitting 17 internal skeletal dots degrades outline IoU (0.662 → 0.618) while 600-pt outline fitting achieves 0.749"
    )
    ax1.legend(loc="lower right", fontsize=7.2, frameon=True, facecolor=PARCHMENT_BG, edgecolor=BORDER_GOLD)

    # Panel B: Morpholib Connection 2 [E8] — 1917 Figs. 381–382 Woodcut + BMNH 1887 Mola Engraving + Provenance Classes
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_facecolor(CARD_BG)

    ax2_in = ax2.inset_axes([0.02, 0.14, 0.36, 0.78])
    img_381 = open_fitted(CACHE_1913 / "gutenberg_i751.png", (320, 340), pad_color=(255, 255, 255))
    ax2_in.imshow(img_381)
    ax2_in.set_xticks([]); ax2_in.set_yticks([])
    for sp in ax2_in.spines.values():
        sp.set_color(BORDER_GOLD)
        sp.set_linewidth(1.2)
    ax2_in.set_title("1917 Woodcut (Figs. 381–382)\nDiodon → Orthagoriscus", fontsize=7.6, fontfamily="serif", fontweight="bold", color=INK_DARK, pad=3)

    dio_shift = np.array([-0.02, 0.0])
    mola_shift = np.array([0.44, 0.0])
    BS_dio_s = BS_dio_raw * 0.68 + dio_shift
    S_dio_s = S_dio_raw * 0.68 + dio_shift
    BT_mola_s = BT_mola * 0.65 + mola_shift
    T_mola_s = T_mola * 0.65 + mola_shift

    ax2.fill(BS_dio_s[:, 0], BS_dio_s[:, 1], color="#fef3c7", alpha=0.88)
    ax2.plot(BS_dio_s[:, 0], BS_dio_s[:, 1], color="#b45309", lw=2.2,
             label="Class A (Diodon): Reconstructed from 17 LMs + fin extensions (build_diodon_silhouette_from_landmarks)")
    ax2.scatter(S_dio_s[:, 0], S_dio_s[:, 1], c="#92400e", edgecolors="white", s=34, zorder=5)

    ax2.fill(BT_mola_s[:, 0], BT_mola_s[:, 1], color="#dcfce7", alpha=0.88)
    ax2.plot(BT_mola_s[:, 0], BT_mola_s[:, 1], color="#15803d", lw=2.2,
             label="Class B (Mola mola): Segmented from 1887 British Museum engraving (extract_aligned_silhouette_and_lms)")
    ax2.scatter(T_mola_s[:, 0], T_mola_s[:, 1], c="#166534", edgecolors="white", s=34, zorder=5)

    ax2.text(-0.02, 0.27, "Provenance Class A:\nLandmark B-Spline", ha="center", fontsize=7.8, fontfamily="serif", fontweight="bold", color="#92400e")
    ax2.text(0.44, 0.27, "Provenance Class B:\n1887 Plate Silhouette", ha="center", fontsize=7.8, fontfamily="serif", fontweight="bold", color="#166534")

    ax2.set_aspect("equal")
    ax2.set_xlim(-0.58, 0.74)
    ax2.set_ylim(-0.38, 0.38)
    style_museum_ax(
        ax2,
        "B. Morpholib Connection 2 [E8]: The Provenance of an Outline",
        "Why a landmark-interpolated contour (Diodon) and an image-segmented contour (Mola) require distinct provenance labels"
    )
    ax2.legend(loc="lower right", fontsize=6.9, frameon=True, facecolor=PARCHMENT_BG, edgecolor=BORDER_GOLD)

    # Panel C: Morpholib Connection 3 [E9] — 90x90 Finite-Difference Jacobian Check
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.set_facecolor(CARD_BG)
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
    sc_plot = ax3.scatter(warped_inside[:, 0], warped_inside[:, 1], c=det_inside, cmap="YlOrBr", s=8, alpha=0.9)
    ax3.plot(BS_sc_sil[:, 0], BS_sc_sil[:, 1], color=INK_DARK, lw=2.0)
    cbar = fig.colorbar(sc_plot, ax=ax3, fraction=0.042, pad=0.02)
    cbar.set_label(r"Sampled $\det D\Phi(x,y)$ (90×90 Grid)", fontsize=8.0, fontfamily="serif")

    min_det = float(np.min(det_J))
    pct_pos = float(100.0 * np.mean(det_J > 0))
    ax3.annotate(
        f"compute_jacobian_stats() [E9]:\n"
        f"• Sampled on 90×90 bounding grid + 600 contour pts\n"
        f"• Sampled positive det J: {pct_pos:.1f}% (min = {min_det:.3f})\n"
        f"• Epistemic scope: rules out sampled local folding;\n"
        f"  does not by itself prove continuum global injectivity",
        xy=(0.0, -0.25), xytext=(-0.45, -0.35),
        fontsize=7.5, fontfamily="serif", color=INK_DARK,
        bbox=dict(boxstyle="round,pad=0.3", fc=PARCHMENT_BG, ec=BORDER_GOLD, lw=1.1)
    )
    ax3.set_aspect("equal")
    ax3.set_xlim(-0.46, 0.52)
    ax3.set_ylim(-0.38, 0.38)
    style_museum_ax(
        ax3,
        "C. Morpholib Connection 3 [E9]: What a Numerical Check Establishes",
        "Separating a sampled finite-difference Jacobian test (90×90 grid) from an analytical proof over a continuous domain"
    )

    # Panel D: Morpholib Connection 4 [E10], Aristotle's Four Causes (Reader Note 24), & Bioelectric Patterning [E6]
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.set_facecolor(CARD_BG)
    ax4.set_xlim(0, 10)
    ax4.set_ylim(0, 10)

    # Top: Aristotle's Four Causes (Reader Note 24 Table) -> MorphoEval Three Pillars [E10]
    ax4.text(5.0, 9.28, "ARISTOTLE’S FOUR CAUSES [E5, E11] & MORPHOEVAL’S THREE PILLARS [E10]",
             ha="center", va="center", fontsize=8.1, fontfamily="serif", fontweight="bold", color=INK_DARK)

    causes = [
        (0.35, 5.95, 4.45, 2.85, "#f5f0e1",
         "Aristotle’s Four Causes (Reader Note 24)",
         "• Material (Ὕλη): Bronze / tissue & ion channels\n"
         "• Formal (Εἶδος): Defining structure & geometry\n"
         "• Efficient (Κινοῦν): Sculptor / biophysical process\n"
         "• Final (Τέλος): Functional end / homeostatic setpoint"),
        (5.15, 5.95, 4.50, 2.85, "#ebe3ce",
         "MorphoEval Audit Separation [E10]",
         "• Implementation (30): Code-to-paper fidelity\n"
         "• Biological Validation (40): Specimen & metric fit\n"
         "• Reproducibility (30): SHA-256 & line traceability\n"
         "  (Uncalibrated ratings, linked to checkpoint evidence)"),
    ]
    for x0, y0, w, h, fc, head_c, body_c in causes:
        ax4.add_patch(mpatches.FancyBboxPatch(
            (x0, y0), w, h, boxstyle="round,pad=0.08,rounding_size=0.14", facecolor=fc, edgecolor=BORDER_GOLD, lw=1.3
        ))
        ax4.text(x0 + 0.18, y0 + h - 0.38, head_c, ha="left", va="center", fontsize=7.6, fontfamily="serif", fontweight="bold", color=INK_DARK)
        ax4.text(x0 + 0.18, y0 + 1.12, body_c, ha="left", va="center", fontsize=7.0, fontfamily="serif", color=INK_MUTED, linespacing=1.28)

    # Bottom: Pai et al. (2012) Xenopus laevis Bioelectric Intervention [E6]
    xen_img = open_fitted(REPO_ROOT / "datasets/species_gallery/I_Enaima_Vertebrata/Xenopus_laevis.jpg", (240, 180))
    ax_in4 = ax4.inset_axes([0.04, 0.07, 0.29, 0.45])
    ax_in4.imshow(xen_img)
    ax_in4.set_xticks([]); ax_in4.set_yticks([])
    for sp in ax_in4.spines.values():
        sp.set_color(BORDER_GOLD)
        sp.set_linewidth(1.3)
    ax_in4.set_title("Xenopus laevis [E6]", fontsize=7.5, fontfamily="serif", fontweight="bold", color=INK_DARK, pad=3)

    ax4.add_patch(mpatches.FancyBboxPatch(
        (3.55, 0.65), 6.10, 4.65,
        boxstyle="round,pad=0.08,rounding_size=0.14", facecolor="#f5f0e1", edgecolor=INK_DARK, lw=1.3
    ))
    ax4.text(3.78, 4.82, "Observation vs. Perturbation: Pai et al. (2012) Development 139:313 [E6]",
             ha="left", va="center", fontsize=7.7, fontfamily="serif", fontweight="bold", color=INK_DARK)
    ax4.text(
        3.78, 2.55,
        "1. Connected Natural History (Aristotle): Record endogenous hyperpolarized\n"
        "   transmembrane voltage ($V_{\\mathrm{mem}}$) domain in the anterior neural field.\n"
        "2. Physical & Geometric Question (Thompson): Specify how voltage gradients\n"
        "   couple to morphogenetic gene expression (Pax6 / Rx1) and tissue boundaries.\n"
        "3. Discriminating Experimental Intervention (Modern): Depolarizing the native\n"
        "   domain disrupts normal eye formation; imposing the hyperpolarized $V_{\\mathrm{mem}}$\n"
        "   state ectopically induces complete eyes outside the head.",
        ha="left", va="center", fontsize=6.95, fontfamily="serif", color=INK_MUTED, linespacing=1.26
    )

    style_museum_ax(
        ax4,
        "D. Morpholib Connection 4 [E10], the Four Causes & Bioelectric Intervention [E6]",
        "Preserving model, measured organism, and causal perturbation as connected but distinct scientific records"
    )

    out_path = MEDIA_DIR / "thompson_1913_fig4_morpholib_four_connections.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Saved {out_path}")


def main():
    copy_standalone_museum_assets()
    build_fig1_spencer_vs_aristotle()
    build_fig2_lesbos_pyrrha_lagoon()
    build_fig3_cephalopod_embryology_soul()
    build_fig4_morpholib_four_connections()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Generate authentic museum-grade historical plates & woodcut-style diagrams
for the Readable Edition of D'Arcy Wentworth Thompson's 1913 Herbert Spencer
Lecture ('On Aristotle as a Biologist, With a Prooemion on Herbert Spencer').

Also renders museum-grade woodcut/engraving replacements for the modern green
flat diagrams in the .docx (Figure 5: Nested Capacities of the Soul, and
Figure 6: Potentiality vs. First & Second Actuality).
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
    """Copy standalone museum portraits and engravings into docs/media/."""
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


def build_museum_fig5_nested_soul():
    """
    Replace the modern flat green Figure 5 from the .docx with a museum-grade
    classical Renaissance woodcut/manuscript plate pairing the Vatican De Anima
    manuscript leaf and Valades's 1579 Scala Naturae with the nested capacities diagram.
    """
    fig = plt.figure(figsize=(14.5, 5.8), dpi=200, facecolor=PARCHMENT_BG)
    gs = gridspec.GridSpec(1, 3, figure=fig, width_ratios=[0.85, 1.35, 0.85], wspace=0.16, left=0.03, right=0.97, top=0.84, bottom=0.10)

    fig.suptitle(
        "Figure 5 — Nested Capacities of the Soul in Aristotle’s De Anima II.2–3 (‘As a Triangle Within a Square’)\n"
        "Framed by the Medieval Illuminated Manuscript of De Anima (Vat. Pal. lat. 1033) & Valades’s 1579 Scala Naturae Copperplate",
        fontsize=11.6, fontweight="bold", fontfamily="serif", color=INK_DARK, y=0.95
    )

    # Left: Vatican De Anima Manuscript
    ax_l = fig.add_subplot(gs[0, 0])
    img_ms = open_fitted(CACHE_1913 / "aristotle_de_anima_manuscript.jpg", (400, 480), crop_frac=(0.05, 0.02, 0.95, 0.95))
    ax_l.imshow(img_ms)
    style_museum_ax(
        ax_l,
        "A. Aristotle’s De Anima Manuscript",
        "Vaticanus Palatinus latinus 1033;\nOn the Soul (De Anima) [E9, E20]"
    )

    # Center: Museum Parchment Engraving of the Nested Capacities (exact text of Figure 5 from .docx)
    ax_c = fig.add_subplot(gs[0, 1])
    ax_c.set_facecolor(CARD_BG)
    ax_c.set_xlim(0, 10)
    ax_c.set_ylim(0, 10)

    # Double classical border
    ax_c.add_patch(mpatches.Rectangle((0.25, 0.25), 9.5, 9.5, fill=False, edgecolor=INK_DARK, lw=1.4))
    ax_c.add_patch(mpatches.Rectangle((0.40, 0.40), 9.2, 9.2, fill=False, edgecolor=BORDER_GOLD, lw=0.9))

    # Outer Box: NUTRITIVE
    ax_c.add_patch(mpatches.FancyBboxPatch(
        (0.75, 1.45), 8.5, 7.75,
        boxstyle="round,pad=0.08,rounding_size=0.18", facecolor="#f3ede0", edgecolor=INK_DARK, lw=1.8
    ))
    ax_c.text(5.0, 8.65, "NUTRITIVE  (ἡ πρώτη / θρεπτικὴ ψυχή)", ha="center", va="center",
              fontsize=10.5, fontfamily="serif", fontweight="bold", color=INK_DARK)
    ax_c.text(5.0, 8.05, "Plants, animals, humans", ha="center", va="center",
              fontsize=9.2, fontfamily="serif", style="italic", color=INK_MUTED)
    ax_c.text(5.0, 7.50, "Nourishment  ·  growth  ·  reproduction", ha="center", va="center",
              fontsize=9.0, fontfamily="serif", color=INK_DARK)

    # Middle Box: SENSITIVE
    ax_c.add_patch(mpatches.FancyBboxPatch(
        (1.45, 1.85), 7.1, 5.05,
        boxstyle="round,pad=0.08,rounding_size=0.18", facecolor="#e6dec8", edgecolor=INK_DARK, lw=1.6
    ))
    ax_c.text(5.0, 6.35, "SENSITIVE  (ἡ αἰσθητικὴ ψυχή)", ha="center", va="center",
              fontsize=10.2, fontfamily="serif", fontweight="bold", color=INK_DARK)
    ax_c.text(5.0, 5.75, "Animals and humans", ha="center", va="center",
              fontsize=9.0, fontfamily="serif", style="italic", color=INK_MUTED)
    ax_c.text(5.0, 5.20, "Perception and associated capacities", ha="center", va="center",
              fontsize=8.8, fontfamily="serif", color=INK_DARK)

    # Inner Box: RATIONAL
    ax_c.add_patch(mpatches.FancyBboxPatch(
        (2.25, 2.25), 5.5, 2.35,
        boxstyle="round,pad=0.08,rounding_size=0.18", facecolor="#d6cbb1", edgecolor=INK_DARK, lw=1.6
    ))
    ax_c.text(5.0, 4.02, "RATIONAL  (ἡ διανοητικὴ ψυχή / νοῦς)", ha="center", va="center",
              fontsize=10.0, fontfamily="serif", fontweight="bold", color=INK_DARK)
    ax_c.text(5.0, 3.42, "Humans", ha="center", va="center",
              fontsize=8.9, fontfamily="serif", style="italic", color=INK_MUTED)
    ax_c.text(5.0, 2.82, "Understanding and reasoning", ha="center", va="center",
              fontsize=8.8, fontfamily="serif", color=INK_DARK)

    ax_c.text(5.0, 0.88, "Locomotion belongs to some animals; it is not required of every animal. (De Anima II.2–3)",
              ha="center", va="center", fontsize=8.3, fontfamily="serif", style="italic", color=INK_MUTED)

    style_museum_ax(
        ax_c,
        "B. Inclusive Hierarchy of Capacities (Reader Notes 18 & 20)",
        "‘Just as we may discern a triangle within a square, so is each lower grade of ψυχή implicit in the higher’ (p. 27)"
    )

    # Right: Valades 1579 Scala Naturae
    ax_r = fig.add_subplot(gs[0, 2])
    img_sc = open_fitted(CACHE_1913 / "scala_naturae_bonnet_or_bovillus.jpg", (400, 480), crop_frac=(0.06, 0.03, 0.94, 0.97))
    ax_r.imshow(img_sc)
    style_museum_ax(
        ax_r,
        "C. Scala Naturae (Valades, 1579)",
        "Renaissance copperplate of the graded\nscale of nature (Reader Note 22)"
    )

    out_path = MEDIA_DIR / "thompson_1913_fig5_nested_soul_museum.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Saved {out_path}")


def build_museum_fig6_first_actuality():
    """
    Replace the modern flat green Figure 6 from the .docx with a classical
    parchment/engraving plate illustrating Potentiality -> First Actuality (Entelecheia)
    -> Second Actuality (De Anima II.1 & II.5, Reader Note 23).
    """
    fig = plt.figure(figsize=(13.8, 4.4), dpi=200, facecolor=PARCHMENT_BG)
    ax = fig.add_axes([0.03, 0.12, 0.94, 0.74])
    ax.set_facecolor(CARD_BG)
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5.2)

    fig.suptitle(
        "Figure 6 — Potentiality, First Actuality (Ἐντελέχεια ἡ πρώτη) & Second Actuality in De Anima II.1 & II.5\n"
        "(Illustrating Reader Note 23 & Editorial Commentary 12: Why Having an Organized Capacity Differs from Actively Exercising It)",
        fontsize=11.4, fontweight="bold", fontfamily="serif", color=INK_DARK, y=0.95
    )

    ax.add_patch(mpatches.Rectangle((0.18, 0.18), 11.64, 4.84, fill=False, edgecolor=INK_DARK, lw=1.4))
    ax.add_patch(mpatches.Rectangle((0.28, 0.28), 11.44, 4.64, fill=False, edgecolor=BORDER_GOLD, lw=0.9))

    boxes = [
        (0.85, 1.35, 2.9, 2.95, "#f3ede0",
         "ABLE TO LEARN",
         "Could learn to read\n(Uninstructed capacity)",
         "Potentiality (Δύναμις)"),
        (4.55, 1.35, 2.9, 2.95, "#e6dec8",
         "HAS THE ABILITY",
         "Has learned to read\n(Organized state / Soul)",
         "First actuality (Ἐντελέχεια ἡ πρώτη)"),
        (8.25, 1.35, 2.9, 2.95, "#d6cbb1",
         "USING THE ABILITY",
         "Is reading now\n(Waking exercise of power)",
         "Second actuality (Ἐνέργεια)"),
    ]
    for x0, y0, w, h, fc, top_t, mid_t, bot_t in boxes:
        ax.add_patch(mpatches.FancyBboxPatch(
            (x0, y0), w, h, boxstyle="round,pad=0.08,rounding_size=0.16", facecolor=fc, edgecolor=INK_DARK, lw=1.6
        ))
        ax.text(x0 + w / 2, y0 + h - 0.52, top_t, ha="center", va="center",
                fontsize=9.8, fontfamily="serif", fontweight="bold", color=INK_DARK)
        ax.plot([x0 + 0.3, x0 + w - 0.3], [y0 + h - 0.92, y0 + h - 0.92], color=BORDER_GOLD, lw=1.0)
        ax.text(x0 + w / 2, y0 + 1.48, mid_t, ha="center", va="center",
                fontsize=8.8, fontfamily="serif", style="italic", color=INK_MUTED)
        ax.text(x0 + w / 2, y0 + 0.52, bot_t, ha="center", va="center",
                fontsize=8.8, fontfamily="serif", fontweight="bold", color="#78350f")

    # Forward arrows
    ax.annotate("", xy=(4.48, 2.85), xytext=(3.82, 2.85),
                arrowprops=dict(arrowstyle="->,head_width=0.35,head_length=0.35", color=INK_DARK, lw=1.8))
    ax.text(4.15, 3.22, "Learn", ha="center", va="bottom", fontsize=9.0, fontfamily="serif", fontweight="bold", color=INK_DARK)

    ax.annotate("", xy=(8.18, 2.85), xytext=(7.52, 2.85),
                arrowprops=dict(arrowstyle="->,head_width=0.35,head_length=0.35", color=INK_DARK, lw=1.8))
    ax.text(7.85, 3.22, "Exercise", ha="center", va="bottom", fontsize=9.0, fontfamily="serif", fontweight="bold", color=INK_DARK)

    # Return curved arrow from Box 3 to Box 2
    ax.annotate(
        "", xy=(6.0, 1.28), xytext=(9.7, 1.28),
        arrowprops=dict(arrowstyle="->,head_width=0.32,head_length=0.32", connectionstyle="arc3,rad=-0.22", color="#78350f", lw=1.6)
    )
    ax.text(7.85, 0.52, "Stop reading; retain the ability  (e.g., a sleeping organism retains its soul as first actuality)",
            ha="center", va="center", fontsize=8.4, fontfamily="serif", style="italic", color=INK_DARK)

    style_museum_ax(
        ax,
        "",
        "ψυχὴ ἐστὶν ἐντελέχεια ἡ πρώτη σώματος φυσικοῦ δυνάμει ζωὴν ἔχοντος — ‘Soul is the first actuality of a natural body potentially having life’ (Note 26)"
    )

    out_path = MEDIA_DIR / "thompson_1913_fig6_first_actuality_museum.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Saved {out_path}")


def main():
    # Run the main 4 folio plates from the existing module logic plus the 2 new museum figures
    copy_standalone_museum_assets()
    build_museum_fig5_nested_soul()
    build_museum_fig6_first_actuality()


if __name__ == "__main__":
    main()

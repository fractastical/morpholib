#!/usr/bin/env python3
"""Draw how a reproduction is made and how it is audited, as one figure.

    python3 morphoeval/report/draw_pipeline.py            # -> docs/media/morphoeval_pipeline.png
    python3 morphoeval/report/draw_pipeline.py --out /tmp/p.png --svg

Two tiers.  The upper tier is the reproduction itself: paper -> the paper's formal source -> the rules
pinned down (inferred from prose, verified on every record, disambiguated with the authors' code where
it exists) -> the script -> comparison with the authors' figures or specimens.  The lower tier is the
MorphoEval cycle: audit -> measure -> build graph -> sign & publish -> knowledge graph -> remediate,
whose fixes feed back into the script.  The same diagram is kept as Mermaid in morphoeval/README.md;
this one is for the PDF report and anything that cannot render Mermaid.  No external drawing library.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[2]

BUILD = "#dbeafe"; EVAL = "#fde68a"; PUB = "#bbf7d0"; LOOP = "#fecaca"; EDGE = "#334155"; RED = "#b91c1c"
# (x, y, w, h, colour, title, body) on a 100 x 100 canvas
YA, YB, H, HB = 56, 13, 17, 20                             # tier B boxes are taller: plain words take more room
WA, XA = 18.0, [1.5, 21.25, 41.0, 60.75, 80.5]              # five boxes, upper tier (left -> right)
WB, XB = 15.0, [84.0, 67.4, 50.8, 34.2, 17.6, 1.0]          # six boxes, lower tier (right -> left)
BOXES = {
    "paper":   (XA[0], YA, WA, H, BUILD, "1  Paper",
                "the model as published:\nequations (Turing, Raup, Hoffmann),\nan encoding (Planform, Limbform,\nWormGUIDES) or a claim (Thompson)"),
    "source":  (XA[1], YA, WA, H, BUILD, "2  Formal source",
                "the paper's own data:\nparameter values; database / atlas\nrecords; measurement tables —\nfetched or vendored, SHA-256 pinned"),
    "rules":   (XA[2], YA, WA, H, BUILD, "3  Pin the rules",
                "prose → precise rule, checked on\nevery record (1181/1181 regions,\n3137/3146 joints); authors' code read\nto settle semantics, not copied"),
    "script":  (XA[3], YA, WA, H, BUILD, "4  Script",
                "<year>-<author>-<model>.py\nreproduce · decode · test\n→ figure + GIF (git-ignored),\nmorphometrics CSV, self-checks"),
    "compare": (XA[4], YA, WA, H, BUILD, "5  Compare",
                "output beside the authors' figures\nor real specimens; distinct forms\nmust come out distinct"),
    "audit":   (XB[0], YB, WB, HB, EVAL,  "6  Audit",
                "a reviewer scores the script\non 7 questions, 0–100 each,\nwith a written reason and\nthe lines of code it rests on,\nat one named commit"),
    "measure": (XB[1], YB, WB, HB, EVAL,  "7  Measure",
                "the outputs are hashed and\nmeasured independently —\nsizes, counts, shapes — against\nthe specimens or the paper:\nnumbers, not opinions"),
    "build":   (XB[2], YB, WB, HB, PUB,   "8  Write as claims",
                "each score becomes one small\nmachine-readable statement:\n\"this script reproduces X\" —\nhow likely, says who, on what\nevidence, under what licence"),
    "publish": (XB[3], YB, WB, HB, PUB,   "9  Sign & publish",
                "each statement is signed and\nput on a public network of\nservers under a permanent\naddress; checked on a test\nnetwork first, then for real"),
    "graph":   (XB[4], YB, WB, HB, PUB,   "10  Public record",
                "anyone can look a score up,\ncite it, check its signature\nor query across all audits;\nthe repository keeps the\naddresses and a report"),
    "remed":   (XB[5], YB, WB, HB, LOOP,  "11  Remediate",
                "what the audit found wrong\nbecomes a to-do list\n(GitHub issue #1); a fix is\na new commit, then the touched\nscores are judged again"),
}
# the file or term behind each tier-B box, in small type under the plain-language body
TERMS = {
    "audit": "audits/<date>-<name>.yaml", "measure": "measure_outputs.py",
    "build": "build_morphoeval_kg.py (RDF)", "publish": "sign_morphoeval.py",
    "graph": "kg/<audit>/REPORT.md", "remed": "GitHub issue #1",
}

# the papers, in year order, with a frame of each reproduction's own animation (docs/media/*.gif)
MEDIA = ROOT / "docs" / "media"
PAPERS = [  # (gif, caption lines, kind)
    ("thompson_transformations.gif",      "Thompson 1917\nOn Growth and Form, ch. XVII\nmethod of transformations", "reproduce"),
    ("thompson_scarus_pomacanthus.gif",   "Thompson 1917, Figs. 375–376\ntested on Claverie & Wainwright\n2014 landmarks (Dryad)", "test"),
    ("turing_gray_scott.gif",             "Turing 1952\nPhil Trans R Soc B\nvia Gray & Scott 1984, Pearson 1993", "reproduce"),
    ("raup_shell_coiling.gif",            "Raup 1966\nJ Paleontol\nshell-coiling morphospace", "reproduce"),
    ("petroff_stromatolite_dome_to_cone.gif", "Petroff et al. 2010\nPNAS\nconical stromatolites", "reproduce"),
    ("chindapol_coral_flow_growth.gif",   "Chindapol et al. 2013\nPLoS Comput Biol\ncoral growth in flow", "reproduce"),
    ("planform_morphospace_timeline.gif", "Lobo, Malone & Levin 2013\nBioinformatics\nPlanform, PlanformDB 2.5.0", "decode"),
    ("limbform_morphospace_timeline.gif", "Lobo et al. 2014\nBioinformatics\nLimbform, LimbformDB", "decode"),
    ("celegans_embryo_timeline.gif",      "Santella et al. 2015\nBMC Bioinformatics\nWormGUIDES atlas, OpenWorm c302", "decode"),
    ("hoffmann_wing_lloyd.gif",           "Hoffmann et al. 2018\nPNAS\ninsect wing venation", "reproduce"),
    ("cervera_levin_mafe_gj_block.gif",   "Cervera, Levin & Mafe 2021\nbioelectric morphogens\n(demo, no DOI pinned)", "reproduce"),
    ("kahng_coral_geometry_growth.gif",   "Kahng, Odle & Wakeman 2024\nPeerJ\ncoral geometry", "reproduce"),
    ("odonata_wing_cells_deep_time.gif",  "Eshghi, Buesch & Gorb 2026\nZenodo, Tables S1–S3\nOdonata wing cells, fossil vs extant", "test"),
]
KIND_COL = {"reproduce": "#2563eb", "decode": "#059669", "test": "#d97706"}
YS = 83                                                      # bottom of the thumbnail strip
FIG_W, FIG_H, YLIM = 13.0, 12.3, 123.0


THUMBS = []                                                  # (gif name, data-coordinate extent) filled by strip()


def gif_frames(name: str):
    from PIL import Image, ImageSequence
    return [f.convert("RGB") for f in ImageSequence.Iterator(Image.open(MEDIA / name))]


def gif_frame(name: str, frac: float = 0.55):
    frames = gif_frames(name)
    return frames[int(frac * (len(frames) - 1))]


def strip(ax):
    """Two rows of thumbnails (7 + 6): each paper with a frame of its reproduction's animation."""
    cols, gap = 7, 1.0; tw = (97.0 - (cols - 1) * gap) / cols; row_h = 14.2
    ax.text(1.5, YS + 2 * row_h + 2.0, "The papers, and the forms their reproductions produce", fontsize=10.5, fontweight="bold", color=EDGE, va="center")
    for i, (gif, cap, kind) in enumerate(PAPERS):
        r, c = divmod(i, cols)
        x = 1.5 + c * (tw + gap); y = YS + (1 - r) * row_h
        img = gif_frame(gif)
        th = tw * (img.height / img.width) * (FIG_W / 100.0) / (FIG_H / YLIM)
        th = min(th, row_h - 6.2)
        ax.imshow(img, extent=(x, x + tw, y + 5.6, y + 5.6 + th), zorder=3, interpolation="lanczos")
        THUMBS.append((gif, (x, x + tw, y + 5.6, y + 5.6 + th)))
        ax.add_patch(FancyBboxPatch((x, y + 5.6), tw, th, boxstyle="square,pad=0", fc="none", ec="#94a3b8", lw=0.5, zorder=4))
        ax.add_patch(FancyBboxPatch((x, y + 4.9), tw, 0.5, boxstyle="square,pad=0", fc=KIND_COL[kind], ec="none", zorder=3))
        ax.text(x + tw / 2, y + 2.3, cap, ha="center", va="center", fontsize=5.7, color="#1e293b", linespacing=1.15)
    x = 1.5 + 6 * (tw + gap); y = YS                                  # legend in the free slot of row 2
    for j, (k, c) in enumerate(KIND_COL.items()):
        ax.add_patch(FancyBboxPatch((x, y + 10.4 - j * 2.8), 1.6, 0.5, boxstyle="square,pad=0", fc=c, ec="none"))
        ax.text(x + 2.2, y + 10.65 - j * 2.8, {"reproduce": "reproduce the\npaper's model", "decode": "decode the\nauthors' encoding",
                                              "test": "test a\npublished claim"}[k], fontsize=6.0, va="center", color=EDGE, linespacing=1.1)


def box(ax, key):
    x, y, w, h, col, title, body = BOXES[key]
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2", fc=col, ec=EDGE, lw=1.0))
    ax.text(x + w / 2, y + h - 2.3, title, ha="center", va="center", fontsize=9.2, fontweight="bold", color=EDGE)
    ax.text(x + w / 2, y + h / 2 - 1.2 + (0.6 if key in TERMS else -0.8), body, ha="center", va="center", fontsize=6.6, color="#1e293b", linespacing=1.25)
    if key in TERMS:
        ax.text(x + w / 2, y + 1.8, TERMS[key], ha="center", va="center", fontsize=5.2, color="#64748b", family="monospace")


def note(ax):
    """What a nanopublication is, for a reader who has never met one."""
    x, y, w, h = 1.0, 1.0, 62.0, 10.6
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3,rounding_size=0.8", fc="#f8fafc", ec="#94a3b8", lw=0.7))
    ax.text(x + 1.2, y + h - 1.6, "What a nanopublication is, and why the scores are published as them", fontsize=7.4, fontweight="bold", color=EDGE, va="center")
    import textwrap
    body = ("A nanopublication is the smallest publishable unit of science: one statement — here \"this script's reproduction of Raup 1966 is faithful, "
            "probability 0.64\" — packaged with who asserted it, when, on what evidence and under what licence, cryptographically signed, and stored on a "
            "public network of servers under a permanent address that anyone can cite or fetch. Once published it cannot be edited; a re-score publishes a "
            "new statement that points at the one it replaces, so the history stays visible. Because each statement is addressed and open, other assessors — "
            "other models, or people — can publish their own assessment of the same hypothesis next to ours, and a reader can query all of them together.")
    ax.text(x + 1.2, y + h - 3.1, textwrap.fill(body, 134), fontsize=6.0, color="#1e293b", va="top", linespacing=1.3)


def pt(key, side):
    x, y, w, h = BOXES[key][:4]
    return {"l": (x, y + h / 2), "r": (x + w, y + h / 2), "t": (x + w / 2, y + h), "b": (x + w / 2, y)}[side]


def arrow(ax, a, sa, b, sb, label=None, rad=0.0, color=EDGE, ls="-", lx=0.0, ly=0.0, fs=6.6):
    p, q = pt(a, sa), pt(b, sb)
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=11, lw=1.1, color=color, ls=ls,
                                 connectionstyle=f"arc3,rad={rad}", shrinkA=2, shrinkB=2))
    if label:
        ax.text((p[0] + q[0]) / 2 + lx, (p[1] + q[1]) / 2 + ly, label, ha="center", va="center", fontsize=fs,
                color=color, bbox=dict(fc="white", ec="none", pad=1.0))


def animate(fig, out: Path, n_frames: int = 40, duration_ms: int = 130, dpi: int = 80):
    """The same figure as a GIF: everything static except the thumbnails, which each play their
    own animation once per loop.  Only the thumbnail rectangles are re-quantised per frame, so
    the rest of the image is byte-identical between frames and the encoder stores just the deltas."""
    import io
    import numpy as np
    from PIL import Image

    fig.set_dpi(dpi); fig.canvas.draw()
    buf = io.BytesIO(); fig.savefig(buf, dpi=dpi, facecolor="white"); buf.seek(0)
    base = Image.open(buf).convert("RGB"); W, Hpx = base.size
    ax = fig.axes[0]

    def px(extent):                                           # data coords -> pixel box, via the axes transform
        x0, x1, y0, y1 = extent
        (X0, Y0), (X1, Y1) = ax.transData.transform([(x0, y0), (x1, y1)])
        return (round(X0), round(Hpx - Y1), round(X1), round(Hpx - Y0))

    clips = []
    for gif, extent in THUMBS:
        b = px(extent); size = (b[2] - b[0], b[3] - b[1])
        frames = gif_frames(gif)
        clips.append((b, [f.resize(size, Image.LANCZOS) for f in frames]))

    sample = base.copy()                                      # palette from the base plus a spread of clip frames
    for k in (0, n_frames // 3, 2 * n_frames // 3):
        for b, frames in clips:
            sample.paste(frames[(k * len(frames)) // n_frames], b[:2])
    pal = sample.quantize(colors=255, method=Image.Quantize.MEDIANCUT)
    base_p = base.quantize(palette=pal, dither=Image.Dither.FLOYDSTEINBERG)

    out_frames = []
    for k in range(n_frames):
        f = base_p.copy()
        for b, frames in clips:
            f.paste(frames[(k * len(frames)) // n_frames].quantize(palette=pal, dither=Image.Dither.FLOYDSTEINBERG), b[:2])
        out_frames.append(f)
    out_frames[0].save(out, save_all=True, append_images=out_frames[1:], duration=duration_ms, loop=0, optimize=True)
    print(f"saved {out}  ({W}x{Hpx}, {n_frames} frames, {out.stat().st_size // 1024} KB)")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "docs" / "media" / "morphoeval_pipeline.png"))
    ap.add_argument("--svg", action="store_true")
    ap.add_argument("--gif", action="store_true", help="also write the animated version next to --out")
    args = ap.parse_args(argv)

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H)); fig.subplots_adjust(0, 0, 1, 1)
    ax.set_xlim(0, 100); ax.set_ylim(0, YLIM); ax.set_aspect("auto"); ax.axis("off")

    # tier labels
    strip(ax)
    ax.add_patch(FancyArrowPatch((10.5, YS + 0.2), (10.5, YA + H + 0.4), arrowstyle="-|>", mutation_scale=12, color=EDGE, lw=1.1, zorder=5))
    ax.text(12.0, YA + H + 4.6, "A  Making a reproduction", fontsize=10.5, fontweight="bold", color=EDGE, va="center")
    ax.text(12.0, YA + H + 2.0, "each paper goes through A once, then round B until it holds", fontsize=7.5, style="italic", color="#475569", va="center")
    ax.text(1.5, YB + HB + 3.0, "B  Auditing it — the MorphoEval cycle", fontsize=10.5, fontweight="bold", color=EDGE, va="center")

    for k in BOXES:
        box(ax, k)

    # tier A
    arrow(ax, "paper", "r", "source", "l")
    arrow(ax, "source", "r", "rules", "l")
    arrow(ax, "rules", "r", "script", "l")
    arrow(ax, "script", "r", "compare", "l")
    ax.text(12, YA - 5.0, "authors' code, where it exists (WormGUIDES loader, OpenWorm c302, Hoffmann's segmentation), is read to settle semantics, not copied",
            ha="left", va="center", fontsize=6.6, color="#64748b", style="italic")

    # A -> B
    arrow(ax, "compare", "b", "audit", "t", "scored, with evidence", lx=4.5, ly=0.0)

    # tier B, right -> left
    arrow(ax, "audit", "l", "measure", "r")
    arrow(ax, "measure", "l", "build", "r", "measurements.yaml", ly=4.6, fs=6.0)
    arrow(ax, "build", "l", "publish", "r")
    arrow(ax, "publish", "l", "graph", "r", "signed, published\nstatements", ly=4.6, fs=6.0)
    arrow(ax, "graph", "l", "remed", "r", "remediation\nitems", ly=4.6, fs=6.0)

    # the loop back into tier A: up from Remediate, across the gap, up into Script (an elbow, so it crosses no text)
    x0, y0 = pt("remed", "t"); x1, y1 = pt("script", "b"); ym = YB + HB + 8
    ax.plot([x0, x0, x1], [y0 + 0.5, ym, ym], color=RED, lw=1.2, solid_capstyle="round")
    ax.add_patch(FancyArrowPatch((x1, ym), (x1, y1), arrowstyle="-|>", mutation_scale=11, lw=1.2, color=RED, shrinkA=0, shrinkB=2))
    ax.text((x0 + x1) / 2, ym + 2.6, "fix → new commit → the touched scores are judged again and published anew; the old statements stay on record, "
            "marked as replaced (nothing published is ever edited)",
            ha="center", va="center", fontsize=6.6, color=RED, bbox=dict(fc="white", ec="none", pad=1.0))
    note(ax)

    ax.text(50, YLIM - 2.6, "How a form gets into this repository, and how it is scored", ha="center", va="center", fontsize=12.5, fontweight="bold", color=EDGE)
    ax.text(50, YLIM - 5.3, "each pass pins a commit; scores are 0–100 per dimension, weighted mean overall; the audit scores the reproduction, not the biology; "
            "every score is published as a nanopublication (what that is: bottom left)",
            ha="center", va="center", fontsize=7.6, color="#475569")
    for i, (c, l) in enumerate([(BUILD, "reproduce"), (EVAL, "evaluate"), (PUB, "publish"), (LOOP, "remediate")]):
        ax.add_patch(FancyBboxPatch((66 + i * 8.5, 2.0), 2.2, 2.2, boxstyle="round,pad=0.1", fc=c, ec=EDGE, lw=0.6))
        ax.text(69 + i * 8.5, 3.1, l, fontsize=7, va="center", color=EDGE)

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160, bbox_inches="tight", facecolor="white")
    if args.svg:
        fig.savefig(out.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    print(f"saved {out}")
    if args.gif:
        animate(fig, out.with_suffix(".gif"))


if __name__ == "__main__":
    main()

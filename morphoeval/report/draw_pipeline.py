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
YA, YB, H = 60, 12, 17
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
                "<year>-<author>-<model>.py\nsimulate · decode · test\n→ figure + GIF (git-ignored),\nmorphometrics CSV, self-checks"),
    "compare": (XA[4], YA, WA, H, BUILD, "5  Compare",
                "output beside the authors' figures\nor real specimens; distinct forms\nmust come out distinct"),
    "audit":   (XB[0], YB, WB, H, EVAL,  "6  Audit",
                "audits/<date>-<name>.yaml\n7 dimensions × 0–100,\nrationale, line-pinned\nevidence @ commit"),
    "measure": (XB[1], YB, WB, H, EVAL,  "7  Measure",
                "measure_outputs.py\nfingerprint outputs,\nmorphometrics vs\nspecimens / literature"),
    "build":   (XB[2], YB, WB, H, PUB,   "8  Build graph",
                "build_morphoeval_kg.py\npk:Claim + 8 assessments\nper script; ProbKnow\ntemplates; conformance"),
    "publish": (XB[3], YB, WB, H, PUB,   "9  Sign & publish",
                "test registry →\ncheck_signed.mjs →\nfetch back → production;\nnpx:supersedes on re-score"),
    "graph":   (XB[4], YB, WB, H, PUB,   "10  Graph",
                "nanopub network,\nNanopub Query;\nkg/<audit>/REPORT.md,\npublished.json"),
    "remed":   (XB[5], YB, WB, H, LOOP,  "11  Remediate",
                "overall's pk:evidenceAgainst\n→ GitHub issue #1;\nfix → new commit →\nre-audit touched dims"),
}


def box(ax, key):
    x, y, w, h, col, title, body = BOXES[key]
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2", fc=col, ec=EDGE, lw=1.0))
    ax.text(x + w / 2, y + h - 2.3, title, ha="center", va="center", fontsize=9.2, fontweight="bold", color=EDGE)
    ax.text(x + w / 2, y + h / 2 - 2.0, body, ha="center", va="center", fontsize=6.6, color="#1e293b", linespacing=1.25)


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


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "docs" / "media" / "morphoeval_pipeline.png"))
    ap.add_argument("--svg", action="store_true")
    args = ap.parse_args(argv)

    fig, ax = plt.subplots(figsize=(13, 8.6))
    ax.set_xlim(0, 100); ax.set_ylim(0, 92); ax.axis("off")

    # tier labels
    ax.text(1.5, YA + H + 3.2, "A  Making a reproduction", fontsize=10.5, fontweight="bold", color=EDGE, va="center")
    ax.text(12, YA - 6.2, "three kinds of script:   simulate the paper's maths (Turing, Raup, Hoffmann, Chindapol)\n"
            "decode the authors' formal encoding into form (Planform, Limbform, WormGUIDES)\n"
            "test a published claim against measurements (Odonata, Thompson)",
            fontsize=6.9, color="#475569", va="center", ha="left", linespacing=1.3)
    ax.text(1.5, YB + H + 3.0, "B  Auditing it — the MorphoEval cycle", fontsize=10.5, fontweight="bold", color=EDGE, va="center")

    for k in BOXES:
        box(ax, k)

    # tier A
    arrow(ax, "paper", "r", "source", "l")
    arrow(ax, "source", "r", "rules", "l")
    arrow(ax, "rules", "r", "script", "l")
    arrow(ax, "script", "r", "compare", "l")
    ax.text(12, YA - 11.5, "authors' code, where it exists (WormGUIDES loader, OpenWorm c302, Hoffmann's segmentation), is read to settle semantics, not copied",
            ha="left", va="center", fontsize=6.6, color="#64748b", style="italic")

    # A -> B
    arrow(ax, "compare", "b", "audit", "t", "scored, with evidence", lx=4.5, ly=0.0)

    # tier B, right -> left
    arrow(ax, "audit", "l", "measure", "r")
    arrow(ax, "measure", "l", "build", "r", "measurements.yaml", ly=2.6, fs=6.2)
    arrow(ax, "build", "l", "publish", "r")
    arrow(ax, "publish", "l", "graph", "r", "signed nanopubs", ly=2.6, fs=6.2)
    arrow(ax, "graph", "l", "remed", "r", "remediation\nitems", ly=3.4, fs=6.2)

    # the loop back into tier A: up from Remediate, across the gap, up into Script (an elbow, so it crosses no text)
    x0, y0 = pt("remed", "t"); x1, y1 = pt("script", "b"); ym = YB + H + 8
    ax.plot([x0, x0, x1], [y0 + 0.5, ym, ym], color=RED, lw=1.2, solid_capstyle="round")
    ax.add_patch(FancyArrowPatch((x1, ym), (x1, y1), arrowstyle="-|>", mutation_scale=11, lw=1.2, color=RED, shrinkA=0, shrinkB=2))
    ax.text((x0 + x1) / 2, ym + 2.6, "fix → new commit → re-score the touched dimensions; new nanopubs npx:supersedes the old (published ones are immutable)",
            ha="center", va="center", fontsize=6.6, color=RED, bbox=dict(fc="white", ec="none", pad=1.0))
    # a second checkpoint also re-measures
    ax.text(XB[4] + WB / 2, YB - 4.0, "other assessors (models, humans) may publish\ntheir own assessments of the same hypotheses",
            ha="center", va="center", fontsize=6.4, color="#475569", style="italic")

    ax.text(50, 89.5, "How a form gets into this repository, and how it is scored", ha="center", va="center", fontsize=12.5, fontweight="bold", color=EDGE)
    ax.text(50, 86.8, "each pass pins a commit; scores are 0–100 per dimension, weighted mean overall; the audit scores the reproduction, not the biology",
            ha="center", va="center", fontsize=7.6, color="#475569")
    for i, (c, l) in enumerate([(BUILD, "reproduce"), (EVAL, "evaluate"), (PUB, "publish"), (LOOP, "remediate")]):
        ax.add_patch(FancyBboxPatch((66 + i * 8.5, 2.0), 2.2, 2.2, boxstyle="round,pad=0.1", fc=c, ec=EDGE, lw=0.6))
        ax.text(69 + i * 8.5, 3.1, l, fontsize=7, va="center", color=EDGE)

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160, bbox_inches="tight", facecolor="white")
    if args.svg:
        fig.savefig(out.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    print(f"saved {out}")


if __name__ == "__main__":
    main()

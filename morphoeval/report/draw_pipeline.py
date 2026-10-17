#!/usr/bin/env python3
"""Draw the MorphoEval build / evaluate / publish cycle as a figure.

    python3 morphoeval/report/draw_pipeline.py            # -> docs/media/morphoeval_pipeline.png
    python3 morphoeval/report/draw_pipeline.py --out /tmp/p.png --svg

The same diagram is kept as Mermaid in README.md / morphoeval/README.md; this one is for the
PDF report and for places that cannot render Mermaid.  No external drawing library: boxes and
arrows are matplotlib patches, so the figure is deterministic and reproducible from the repo.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[2]

# (x, y, w, h) in a 100 x 62 canvas; colour by stage
BUILD = "#dbeafe"; EVAL = "#fde68a"; PUB = "#bbf7d0"; LOOP = "#fecaca"; EDGE = "#334155"
BOXES = {
    "source":  (2,  44, 20, 14, BUILD, "1  Source",
                "paper (DOI), authors' figures,\ndata with SHA-256 / licence\n(fetched or vendored)"),
    "script":  (27, 44, 20, 14, BUILD, "2  Reproduce",
                "<year>-<author>-<model>.py\n→ figure + GIF (git-ignored),\nmorphometrics CSV, self-checks"),
    "audit":   (52, 44, 20, 14, EVAL,  "3  Audit",
                "audits/<date>-<name>.yaml:\n7 dimensions × 0–100, rationale,\nline-pinned evidence @ commit"),
    "measure": (77, 44, 21, 14, EVAL,  "4  Measure",
                "measure_outputs.py:\nfingerprint outputs (SHA-256),\nmorphometrics vs specimens"),
    "build":   (77, 22, 21, 14, PUB,   "5  Build graph",
                "build_morphoeval_kg.py:\npk:Claim + 8 pk:LLMClaimAssessment\nper script; ProbKnow templates;\noffline conformance check"),
    "publish": (52, 22, 20, 14, PUB,   "6  Sign & publish",
                "sign → test registry →\ncheck_signed.mjs (nanopub-js) →\nfetch back → production;\nnpx:supersedes on re-score"),
    "track":   (27, 22, 20, 14, LOOP,  "7  Remediate",
                "remediation items → GitHub issue;\nfix the script → new commit\n→ re-audit touched dimensions"),
    "graph":   (2,  22, 20, 14, PUB,   "Knowledge graph",
                "nanopub network / Nanopub Query;\nREPORT.md + published.json\nper audit in morphoeval/kg/"),
}


def box(ax, key):
    x, y, w, h, col, title, body = BOXES[key]
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2",
                                fc=col, ec=EDGE, lw=1.0))
    ax.text(x + w / 2, y + h - 2.2, title, ha="center", va="center", fontsize=9.5, fontweight="bold", color=EDGE)
    ax.text(x + w / 2, y + h / 2 - 2.2, body, ha="center", va="center", fontsize=7.2, color="#1e293b", linespacing=1.25)


def centre(key, side):
    x, y, w, h = BOXES[key][:4]
    return {"l": (x, y + h / 2), "r": (x + w, y + h / 2), "t": (x + w / 2, y + h), "b": (x + w / 2, y)}[side]


def arrow(ax, a, sa, b, sb, label=None, rad=0.0, color=EDGE, ls="-", lx=0, ly=0):
    p, q = centre(a, sa), centre(b, sb)
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=12, lw=1.1, color=color, ls=ls,
                                 connectionstyle=f"arc3,rad={rad}", shrinkA=2, shrinkB=2))
    if label:
        ax.text((p[0] + q[0]) / 2 + lx, (p[1] + q[1]) / 2 + ly, label, ha="center", va="center", fontsize=6.8,
                color=color, bbox=dict(fc="white", ec="none", pad=1.0))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "docs" / "media" / "morphoeval_pipeline.png"))
    ap.add_argument("--svg", action="store_true", help="also write an SVG beside the PNG")
    args = ap.parse_args(argv)

    fig, ax = plt.subplots(figsize=(13, 7.6))
    ax.set_xlim(0, 100); ax.set_ylim(0, 74); ax.axis("off")
    for k in BOXES:
        box(ax, k)

    arrow(ax, "source", "r", "script", "l")
    arrow(ax, "script", "r", "audit", "l", "read code, run,\ninspect output", ly=3.2)
    arrow(ax, "audit", "r", "measure", "l")
    arrow(ax, "measure", "b", "build", "t", "measurements.yaml", lx=0, ly=0)
    arrow(ax, "build", "l", "publish", "r")
    arrow(ax, "publish", "l", "track", "r", "remediation items\n= pk:evidenceAgainst", ly=-8.6)
    arrow(ax, "publish", "b", "graph", "b", None, rad=-0.35)
    ax.text(30, 12.0, "signed nanopubs, URIs in published.json", ha="center", va="center", fontsize=6.8, color=EDGE,
            bbox=dict(fc="white", ec="none", pad=1.0))
    # the loop: remediation -> new commit -> script -> re-audit (supersedes)
    arrow(ax, "track", "t", "script", "b", "fix → new commit", color="#b91c1c", lx=-8, ly=0)
    arrow(ax, "track", "t", "audit", "b", "re-score touched\ndimensions;\nnpx:supersedes", color="#b91c1c", rad=-0.25, lx=-2, ly=-2.5)
    # source hashes feed the audit's traceability
    arrow(ax, "source", "t", "audit", "t", "DOIs, hashes →\nsource-traceability", rad=-0.18, color="#64748b", ls="--", ly=1.5)
    # graph feeds back: other assessors can add assessments of the same hypotheses
    ax.text(14, 8.5, "other assessors (models, humans) may publish\ntheir own pk:*Assessment of the same hypotheses",
            ha="center", va="center", fontsize=6.6, color="#475569", style="italic")

    ax.text(50, 71.5, "MorphoEval: build → evaluate → publish → remediate", ha="center", va="center", fontsize=12, fontweight="bold", color=EDGE)
    ax.text(50, 68.6, "each pass pins a commit; scores are 0–100 per dimension, weighted mean overall; published nanopubs are immutable, re-scores supersede",
            ha="center", va="center", fontsize=7.6, color="#475569")
    for i, (c, l) in enumerate([(BUILD, "build"), (EVAL, "evaluate"), (PUB, "publish"), (LOOP, "remediate")]):
        ax.add_patch(FancyBboxPatch((70 + i * 7.5, 2.2), 2.2, 2.2, boxstyle="round,pad=0.1", fc=c, ec=EDGE, lw=0.6))
        ax.text(73 + i * 7.5, 3.3, l, fontsize=7, va="center", color=EDGE)

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160, bbox_inches="tight", facecolor="white")
    if args.svg:
        fig.savefig(out.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    print(f"saved {out}")


if __name__ == "__main__":
    main()

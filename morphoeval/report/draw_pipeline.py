#!/usr/bin/env python3
"""Draw how a reproduction is made and how it is audited, as one figure.

    python3 morphoeval/report/draw_pipeline.py            # -> docs/media/morphoeval_pipeline.png
    python3 morphoeval/report/draw_pipeline.py --out /tmp/p.png --svg

Two tiers with the outputs between them.  Tier A is the reproduction: a person scouts -> paper -> the
paper's formal source -> the rules pinned down (inferred from prose, verified on every record,
disambiguated with the authors' code where it exists) -> the script.  Box 6, spanning the width, is
what the scripts produce: a frame of each reproduction's own animation, labelled with its paper -
the thing that is compared and scored.  Tier B is the MorphoEval cycle: audit -> measure -> write as
claims -> sign & publish -> public record -> remediate, and the fix loop runs straight back up from
Remediate, between two columns of outputs, into Script.  The same diagram is kept as Mermaid in morphoeval/README.md;
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

BUILD = "#dbeafe"; EVAL = "#fde68a"; PUB = "#bbf7d0"; LOOP = "#fecaca"; EDGE = "#334155"; RED = "#b91c1c"; HUMAN = "#7c3aed"

# A robot's colour is the company whose model it is; its role (builder or assessor) is said in words beside it.
# Which models have assessed is read from the audits (audit.assessor.homepage); the builder is not recorded in
# the audits - it is the coding agent of the session that wrote the scripts - and is set here.
PROVIDERS = {"anthropic.com": ("Anthropic", "#d97757"), "openai.com": ("OpenAI", "#10a37f"),
             "google.com": ("Google", "#4285f4"), "deepmind.google": ("Google", "#4285f4"),
             "x.ai": ("xAI", "#111827"), "mistral.ai": ("Mistral", "#ff7000"), "meta.com": ("Meta", "#0866ff")}
OTHER_PROVIDER = ("another company", "#6b7280")
BUILDER_HOMEPAGE = "https://www.anthropic.com"               # the Cursor agent (Claude) that wrote the scripts so far


def provider(homepage):
    host = (homepage or "").split("//")[-1].split("/")[0].lower()
    for dom, v in PROVIDERS.items():
        if host == dom or host.endswith("." + dom):
            return v
    return OTHER_PROVIDER


def assessors_from_audits():
    """Distinct (company, colour, model label) of the assessors named in the audits, in order of first use."""
    import yaml
    seen, counts, passes = {}, {}, {}
    for f in sorted(p for p in (ROOT / "morphoeval" / "audits").glob("*.yaml") if ".measurements." not in p.name):
        doc = yaml.safe_load(open(f))
        a = (doc.get("audit") or {}).get("assessor") or {}
        name, col = provider(a.get("homepage"))
        seen.setdefault(name, (name, col, a.get("label") or a.get("id") or "?"))
        counts[name] = counts.get(name, 0) + 1
        n_eval = sum(1 for s in (doc.get("simulations") or []) if s.get("status") == "evaluated")
        passes[name] = passes.get(name, 0) + n_eval
    global AUDIT_COUNTS, PASS_COUNTS
    AUDIT_COUNTS = counts
    PASS_COUNTS = passes
    return list(seen.values())


AUDIT_COUNTS = {}
PASS_COUNTS = {}


BUILDER = provider(BUILDER_HOMEPAGE)[1]
ASSESSORS = assessors_from_audits() or [OTHER_PROVIDER + ("?",)]
ASSESSOR = ASSESSORS[0][1]
# (x, y, w, h, colour, title, body) on a 100 x 100 canvas.  Top to bottom: tier A (five boxes, left -> right),
# the outputs band (box 6), tier B (six boxes, left -> right), the note.
YA, YB, H = 78, 13, 20
WA, XA = 18.0, [1.5, 21.25, 41.0, 60.75, 80.5]
W, XB = 15.0, [1.0, 17.6, 34.2, 50.8, 67.4, 84.0]
BOXES = {
    "scout":   (XA[0], YA, WA, H, BUILD, "1  Scout",
                "a person finds a model of form worth\nthe work: a classic paper, an open\ndataset, a claim nobody has tested,\na score that is too low — and\npoints the agent at it"),
    "paper":   (XA[1], YA, WA, H, BUILD, "2  Paper",
                "the model as the authors published it:\na set of equations, a database of\ndrawings, or a sentence that\nmakes a claim"),
    "source":  (XA[2], YA, WA, H, BUILD, "3  Formal source",
                "the paper's own numbers and records —\nparameter values, database entries,\nmeasurement tables — fetched or copied\nin and locked to a fingerprint, so a\nchanged file is refused, not used"),
    "rules":   (XA[3], YA, WA, H, BUILD, "4  Pin the rules",
                "what the paper says in prose is turned\ninto an exact rule and tested on every\nrecord in the data; where the authors\npublished code, it is read to settle\nwhat they meant, never copied"),
    "script":  (XA[4], YA, WA, H, BUILD, "5  Script",
                "one program per paper, which\nreproduces the model, decodes the\nauthors' data or tests the claim —\nthen draws the result, measures it\nand checks itself"),
    "audit":   (XB[0], YB, W, H, EVAL,  "7  Dual-agent audit",
                "two assessor agents from different\ncompanies — not just the builder\nscoring itself — independently score\n7 questions (0–100) with line-pinned\nevidence at one commit"),
    "measure": (XB[1], YB, W, H, EVAL,  "8  Measure",
                "the outputs are fingerprinted\nand measured independently —\nsizes, counts, shapes — against\nthe specimens or the paper:\nnumbers, not opinions"),
    "build":   (XB[2], YB, W, H, PUB,   "9  Write as claims",
                "each score becomes one small\nmachine-readable statement:\n\"this script reproduces X\" —\nwhat rubric score, says who,\non what evidence & licence"),
    "publish": (XB[3], YB, W, H, PUB,   "10  Sign & publish",
                "each statement is signed and\nput on a public network of\nservers under a permanent\naddress; checked on a test\nnetwork first, then for real"),
    "graph":   (XB[4], YB, W, H, PUB,   "11  Public record",
                "anyone can look a score up,\ncite it, check its signature\nor query across all audits;\nthe repository keeps the\naddresses and a report"),
    "remed":   (XB[5], YB, W, H, LOOP,  "12  Remediate",
                "what the audit found wrong\nbecomes a public to-do list;\na person picks, the builder agent\nre-works the script, and the\ntouched scores are judged again"),
}
# the file or term behind each tier-B box, in small type under the plain-language body
TERMS = {
    "scout": "so far: 15 papers, chosen by hand", "paper": "Raup 1966 · PlanformDB · Thompson 1917",
    "source": "SHA-256 pinned · datasets/<paper>/", "rules": "e.g. 1181/1181 regions, 3137/3146 joints",
    "script": "<year>-<author>-<model>.py", "audit": "audits/<date>-<name>.yaml", "measure": "measure_outputs.py",
    "build": "build_morphoeval_kg.py (RDF)", "publish": "sign_morphoeval.py",
    "graph": "kg/<audit>/REPORT.md", "remed": "GitHub issue #1",
}
HUMANS = {"scout": "a person\nchooses", "remed": "a person picks\nwhat to fix"}
AGENTS = {"script": (BUILDER, "the builder agent\nwrites it"), "audit": (ASSESSOR, "two independent\nassessors score it")}

# the papers, in year order, with a frame of each reproduction's own animation (docs/media/*.gif)
MEDIA = ROOT / "docs" / "media"
PAPERS = [  # (gif, caption lines, kind)
    ("thompson_transformations.gif",      "Thompson 1917\nOn Growth and Form, ch. XVII\na real Scarus on the deformed grid", "reproduce"),
    ("turing_gray_scott.gif",             "Turing 1952\nPhil Trans R Soc B\nvia Gray & Scott 1984, Pearson 1993", "reproduce"),
    ("raup_shell_coiling.gif",            "Raup 1966\nJ Paleontol\nshell-coiling morphospace", "reproduce"),
    ("conus_textile_rule30.gif",          "Wolfram 1984 · Meinhardt 1987\nNature · J Theor Biol\nConus textile & Rule 30", "reproduce"),
    ("bressloff_v1_form_constants.gif",   "Bressloff et al. 2001\nPhil Trans R Soc B\nV1 form constants & retinal map", "reproduce"),
    ("petroff_stromatolite_dome_to_cone.gif", "Petroff et al. 2010\nPNAS\nconical stromatolites", "reproduce"),
    ("chindapol_coral_flow_growth.gif",   "Chindapol et al. 2013\nPLoS Comput Biol\ncoral growth in flow", "reproduce"),
    ("planform_morphospace_timeline.gif", "Lobo, Malone & Levin 2013\nBioinformatics\nPlanform, PlanformDB 2.5.0", "decode"),
    ("limbform_morphospace_timeline.gif", "Lobo et al. 2014\nBioinformatics\nLimbform, LimbformDB", "decode"),
    ("celegans_embryo_timeline.gif",      "Santella et al. 2015\nBMC Bioinformatics\nWormGUIDES atlas + OpenWorm", "decode"),
    ("hoffmann_wing_lloyd.gif",           "Hoffmann et al. 2018\nPNAS\ninsect wing venation", "reproduce"),
    ("cervera_levin_mafe_gj_block.gif",   "Cervera, Levin & Mafe 2021\nbioelectric morphogens\non a real S. mediterranea", "reproduce"),
    ("kahng_coral_geometry_growth.gif",   "Kahng, Odle & Wakeman 2024\nPeerJ\ncoral geometry", "reproduce"),
    ("zhang_sorting_morphogenesis.gif",   "Zhang, Goldstein & Levin 2025\nAdaptive Behavior\nself-sorting cell arrays & chimeras", "reproduce"),
    ("odonata_wing_cells_deep_time.gif",  "Eshghi, Buesch & Gorb 2026\nZenodo, Tables S1–S3\nOdonata wing cells, fossil vs extant", "test"),
]
KIND_COL = {"reproduce": "#2563eb", "decode": "#059669", "test": "#d97706"}
YS = 40.5                                                    # bottom of the outputs band (box 6)
FIG_W, FIG_H, YLIM = 13.0, 11.3, 113.0


THUMBS = []                                                  # (gif name, data-coordinate extent) filled by strip()


def gif_frames(name: str):
    from PIL import Image, ImageSequence
    return [f.convert("RGB") for f in ImageSequence.Iterator(Image.open(MEDIA / name))]


def gif_frame(name: str, frac: float = 0.55):
    frames = gif_frames(name)
    return frames[int(frac * (len(frames) - 1))]


def strip(ax):
    """Box 6: the outputs.  Two rows of thumbnails (8 + 6), each a frame of a script's own animation, labelled
    with the paper it comes from.  Framed like a tier-A box because it is one: the thing Compare and Audit look at."""
    cols, gap, wide = 8, 0.9, 2.5; tw = (97.0 - (cols - 2) * gap - wide) / cols; row_h = 14.2   # the last gap is wider: the fix loop runs through it
    top = YS + 2 * row_h + 5.8
    col_x = [1.5 + c * (tw + gap) + (wide - gap if c == cols - 1 else 0) for c in range(cols)]
    ax.add_patch(FancyBboxPatch((1.0, YS + 0.4), 98.0, top - YS - 0.4, boxstyle="round,pad=0.4,rounding_size=1.2", fc="#eff6ff", ec=EDGE, lw=1.0, zorder=1))
    ax.text(50, top - 2.0, "6  Output & compare", ha="center", va="center", fontsize=9.2, fontweight="bold", color=EDGE)
    ax.text(50, top - 4.3, "what each script produces — its own animation, here a frame of it, labelled with the paper it reproduces — "
            "set beside the authors' figures or real specimens; distinct forms must come out distinct.  This is what gets scored.",
            ha="center", va="center", fontsize=6.6, color="#1e293b")
    for i, (gif, cap, kind) in enumerate(PAPERS):
        r, c = divmod(i, cols)
        x = col_x[c]; y = YS + (1 - r) * row_h
        img = gif_frame(gif)
        th = tw * (img.height / img.width) * (FIG_W / 100.0) / (FIG_H / YLIM)
        th = min(th, row_h - 6.2)
        ax.imshow(img, extent=(x, x + tw, y + 5.6, y + 5.6 + th), zorder=3, interpolation="lanczos")
        THUMBS.append((gif, (x, x + tw, y + 5.6, y + 5.6 + th)))
        ax.add_patch(FancyBboxPatch((x, y + 5.6), tw, th, boxstyle="square,pad=0", fc="none", ec="#94a3b8", lw=0.5, zorder=4))
        ax.add_patch(FancyBboxPatch((x, y + 4.9), tw, 0.5, boxstyle="square,pad=0", fc=KIND_COL[kind], ec="none", zorder=3))
        ax.text(x + tw / 2, y + 2.3, cap, ha="center", va="center", fontsize=5.3, color="#1e293b", linespacing=1.15, zorder=3)
    # Inline legend in top-right of Box 6 header (leaving col_x[7] of row 2 open for the remediation loop)
    lx0 = 66.5; ly0 = top - 2.1
    for j, (k, c) in enumerate(KIND_COL.items()):
        xj = lx0 + j * 10.6
        ax.add_patch(FancyBboxPatch((xj, ly0 - 0.25), 1.3, 0.5, boxstyle="square,pad=0", fc=c, ec="none", zorder=3))
        ax.text(xj + 1.6, ly0, {"reproduce": "reproduce model", "decode": "decode encoding",
                                "test": "test claim"}[k], fontsize=5.3, va="center", color=EDGE, zorder=3)
    return top, col_x[7] - wide / 2                                    # band top; x of the (wide) gap between columns 7 and 8


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
    body = ("A nanopublication is the smallest publishable unit of science: one statement — here \"this script's reproduction of Raup 1966 scores 64.1/100 "
            "on the MorphoEval rubric (mostly faithful)\" — packaged with who asserted it, when, on what evidence and under what licence, cryptographically "
            "signed, and stored on a public network of servers under a permanent address that anyone can cite or fetch. Once published it cannot be edited; "
            "a re-score publishes a new statement that points at the one it replaces, so the history stays visible. Because each statement is addressed and "
            "open, other assessors — other models, or people — can publish their own assessment of the same hypothesis next to ours, and a reader can query all of them together.")
    ax.text(x + 1.2, y + h - 3.1, textwrap.fill(body, 134), fontsize=6.0, color="#1e293b", va="top", linespacing=1.3)


def person(ax, x, y, s=1.0, color=HUMAN):
    """A small person: head and shoulders, in data coordinates (x units are ~1.3x wider than y units)."""
    from matplotlib.patches import Ellipse
    ax.add_patch(Ellipse((x, y + 2.3 * s), 1.5 * s, 1.9 * s, fc=color, ec="none", zorder=6))
    ax.add_patch(FancyBboxPatch((x - 1.4 * s, y), 2.8 * s, 1.6 * s, boxstyle="round,pad=0,rounding_size=0.9", fc=color, ec="none", zorder=6))


def agent(ax, x, y, s=1.0, color=BUILDER):
    """A small robot: square head with two eyes and an antenna, and shoulders."""
    from matplotlib.patches import Ellipse
    ax.add_patch(FancyBboxPatch((x - 0.8 * s, y + 1.5 * s), 1.6 * s, 1.5 * s, boxstyle="round,pad=0,rounding_size=0.3", fc=color, ec="none", zorder=6))
    ax.plot([x, x], [y + 3.0 * s, y + 3.6 * s], color=color, lw=1.0, zorder=6); ax.add_patch(Ellipse((x, y + 3.7 * s), 0.5 * s, 0.4 * s, fc=color, ec="none", zorder=6))
    for dx in (-0.35, 0.35):
        ax.add_patch(Ellipse((x + dx * s, y + 2.3 * s), 0.4 * s, 0.4 * s, fc="white", ec="none", zorder=7))
    ax.add_patch(FancyBboxPatch((x - 1.4 * s, y), 2.8 * s, 1.3 * s, boxstyle="round,pad=0,rounding_size=0.5", fc=color, ec="none", zorder=6))


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

    # box 6, the outputs, between the tiers
    band_top, x_gap = strip(ax)

    # tier labels
    ax.text(XA[1], YA + H + 4.6, "A  Making a reproduction", fontsize=10.5, fontweight="bold", color=EDGE, va="center")
    ax.text(XA[1], YA + H + 2.0, "each paper goes through A once; then round B, and back into A, until it holds", fontsize=7.5, style="italic", color="#475569", va="center")
    ax.text(XB[1] + 2.0, YB + H + 2.0, "B  Auditing it — the MorphoEval cycle", fontsize=10.5, fontweight="bold", color=EDGE, va="center")

    for k in BOXES:
        box(ax, k)
    # the people: who decides what the agent is pointed at, and what to fix next
    x, y, w, h = BOXES["scout"][:4]
    person(ax, x + 1.0, y + h + 1.2, 1.1); ax.text(x + 3.0, y + h + 2.9, HUMANS["scout"], fontsize=6.4, color=HUMAN, va="center", fontweight="bold", linespacing=1.1)
    x, y, w, h = BOXES["remed"][:4]
    person(ax, x + 3.6, y + h + 1.2, 1.1); ax.text(x + 5.6, y + h + 2.9, HUMANS["remed"], fontsize=6.2, color=HUMAN, va="center", fontweight="bold", linespacing=1.1)
    # the agents: who writes the code, who scores it (they should not be the same model)
    for key, (col, label) in AGENTS.items():
        x, y, w, h = BOXES[key][:4]
        cols = [a[1] for a in ASSESSORS] if key == "audit" else [col]       # one robot per company that has assessed
        step = 2.4 if len(cols) > 2 else 3.0
        for j, c in enumerate(cols):
            agent(ax, x + 0.9 + step * j, y + h + 1.0, 0.92 if len(cols) > 2 else 1.0, c)
        ax.text(x + 0.6 + step * len(cols), y + h + 2.9, label, fontsize=6.1, color=cols[-1], va="center", fontweight="bold", linespacing=1.1)

    # tier A
    arrow(ax, "scout", "r", "paper", "l")
    arrow(ax, "paper", "r", "source", "l")
    arrow(ax, "source", "r", "rules", "l")
    arrow(ax, "rules", "r", "script", "l")
    ax.text(XA[0], YA - 2.4, "five of the fourteen papers come with the authors' own or reference code; it is read to settle what they meant, never copied",
            ha="left", va="center", fontsize=6.6, color="#64748b", style="italic")

    # Script -> outputs (down into box 6), outputs -> Audit (down out of box 6)
    xs = XA[4] + WA / 2 + 4.0
    ax.add_patch(FancyArrowPatch((xs, YA), (xs, band_top), arrowstyle="-|>", mutation_scale=11, lw=1.1, color=EDGE, shrinkA=2, shrinkB=1))
    ax.text(xs + 1.2, (YA + band_top) / 2, "writes", fontsize=6.6, color=EDGE, va="center", ha="left")
    xa = XB[0] + W - 2.0
    ax.add_patch(FancyArrowPatch((xa, YS + 0.4), (xa, YB + H), arrowstyle="-|>", mutation_scale=11, lw=1.1, color=EDGE, shrinkA=1, shrinkB=2))
    ax.text(xa + 1.2, YS - 2.2, "scored, with evidence", fontsize=6.6, color=EDGE, va="center", ha="left")

    # tier B, left -> right
    arrow(ax, "audit", "r", "measure", "l")
    arrow(ax, "measure", "r", "build", "l", "the measurements", ly=4.6, fs=6.0)
    arrow(ax, "build", "r", "publish", "l")
    arrow(ax, "publish", "r", "graph", "l", "signed, published\nstatements", ly=4.6, fs=6.0)
    arrow(ax, "graph", "r", "remed", "l", "remediation\nitems", ly=4.6, fs=6.0)

    # the loop: straight up from Remediate, between two columns of outputs, into Script
    y0 = YB + H; y1 = YA
    ax.add_patch(FancyArrowPatch((x_gap, y0 + 0.5), (x_gap, y1), arrowstyle="-|>", mutation_scale=11, lw=1.2, color=RED, shrinkA=0, shrinkB=2, zorder=5))
    ax.text(x_gap - 1.5, YS - 2.6, "the builder agent re-works the script — its fidelity to the paper, its sources, its checks —\n"
            "the outputs are re-made and the assessor judges the touched scores again; the old statements\n"
            "stay on record, marked as replaced (nothing published is ever edited)",
            ha="right", va="center", fontsize=6.4, color=RED, linespacing=1.2)
    agent(ax, x_gap - 46.0, YS - 4.6, 1.0)                               # the builder, at the start of the loop's label
    note(ax)

    ax.text(50, YLIM - 2.6, "How a body gets its form, and how a form gets into this repository", ha="center", va="center", fontsize=12.5, fontweight="bold", color=EDGE)
    ax.text(50, YLIM - 5.2, "morphogenesis is how a body gets its shape and regeneration how it gets it back; the newer papers here hold that the pattern is stored bioelectrically,\n"
            "as a map of cell voltages — change the voltage and a planarian regrows two heads.  Each paper becomes a running program; MorphoEval scores how faithfully, 0–100 on seven questions.",
            ha="center", va="top", fontsize=7.4, color="#1e293b", linespacing=1.35)
    for i, (c, l) in enumerate([(BUILD, "reproduce"), (EVAL, "evaluate"), (PUB, "publish"), (LOOP, "remediate")]):
        ax.add_patch(FancyBboxPatch((65 + i * 8.0, 1.4), 2.2, 2.2, boxstyle="round,pad=0.1", fc=c, ec=EDGE, lw=0.6))
        ax.text(68 + i * 8.0, 2.5, l, fontsize=7, va="center", color=EDGE)
    person(ax, 66.1, 4.0, 0.6); ax.text(68, 5.0, "a person decides what to go after and what to fix", fontsize=6.3, va="center", color=HUMAN)
    agent(ax, 66.1, 6.0, 0.55); ax.text(68, 7.0, "a builder agent writes and re-works the code; an assessor agent scores it", fontsize=6.3, va="center", color=EDGE)
    used = [(n, c) for n, c, _ in ASSESSORS]
    if provider(BUILDER_HOMEPAGE)[0] not in [n for n, _ in used]:
        used.insert(0, provider(BUILDER_HOMEPAGE)[:2])
    xx = 66.1                                                             # the companies whose models have taken part, one robot each
    extra = [("other", OTHER_PROVIDER[1])] if len(used) >= 3 else [(OTHER_PROVIDER[0] + ", once its model takes part", OTHER_PROVIDER[1])]
    for n, c in used + extra:
        agent(ax, xx, 8.1, 0.55, c); ax.text(xx + 1.9, 9.1, n, fontsize=6.3, va="center", color=c, fontweight="bold" if c != OTHER_PROVIDER[1] else "normal")
        xx += 3.6 + 0.5 * len(n)
    builder_co = provider(BUILDER_HOMEPAGE)[0]
    n_passes = sum(PASS_COUNTS.values()); n_own_p = PASS_COUNTS.get(builder_co, 0)
    others_p = ", ".join(f"{n} by {co}'s" for co, n in PASS_COUNTS.items() if co != builder_co)
    ax.text(68, 11.8, "a robot's colour is the company whose model it is.  Every evaluated script is dual-scored by two", fontsize=6.0, va="center", color=EDGE)
    ax.text(68, 10.7, (f"companies' models (no agent only grades itself): {n_own_p}/{n_passes} passes by {builder_co}'s, {others_p}."),
            fontsize=6.0, va="center", color=EDGE)

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160, bbox_inches="tight", facecolor="white")
    if args.svg:
        fig.savefig(out.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    print(f"saved {out}")
    if args.gif:
        animate(fig, out.with_suffix(".gif"))


if __name__ == "__main__":
    main()

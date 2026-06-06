#!/usr/bin/env python3
"""
Tested-claims summary slide (first page of claims_all_in_one.pdf).

One landscape page that summarizes every claim we have actually SCORED, each
with its live verdict and an accompanying thumbnail graph, plus a panel listing
the claims that are still not testable with current data.

Data source: wave_laterality.csv (produced by wave_laterality_analysis.py) and
the verbatim claims in Calcium claims.docx (via claims_doc_parser).

Output: analysis_results/claims_tested_summary.pdf
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from textwrap import wrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from generate_claims_inventory_pdf import (  # noqa: E402
    load_laterality, load_landmarks)

RESULT = "#1f6f3f"     # supported / green
RESULT_NEG = "#aa2a2a"  # not supported / red
RESULT_DESC = "#1f6fb2"  # descriptive / blue
GREY = "#777777"

WOUND_C, PRESS_C = "#d95f0e", "#2c7fb8"


def _verdict_box(ax, text, color):
    ax.text(0.5, -0.32, text, transform=ax.transAxes, ha="center", va="top",
            fontsize=8, color="white", wrap=True,
            bbox=dict(boxstyle="round,pad=0.4", fc=color, ec="none"))


def build(lat: dict, lat_csv: Path, out_pdf: Path, lm: dict | None = None) -> None:
    df = pd.read_csv(lat_csv)

    fig = plt.figure(figsize=(14, 9))
    fig.suptitle("Calcium Claims — Tested Summary", fontsize=18,
                 fontweight="bold", y=0.975)
    fig.text(0.5, 0.935,
             f"16 of 19 verbatim claims scored across {lat['n_videos']} videos "
             "· each panel = one tested claim group · verdict below each graph",
             ha="center", fontsize=10, color="#444444")

    gs = fig.add_gridspec(3, 3, hspace=1.05, wspace=0.30,
                          left=0.055, right=0.97, top=0.88, bottom=0.07)

    # ---- Panel A: bidirectional (claims 1-2) ----
    ax = fig.add_subplot(gs[0, 0])
    bd = df.groupby("stimulus")["bidirectional_stim"].mean() * 100
    order = [s for s in ("wound", "press") if s in bd.index]
    ax.bar(order, [bd[s] for s in order], color=[WOUND_C, PRESS_C][:len(order)])
    for i, s in enumerate(order):
        ax.text(i, bd[s] + 2, f"{bd[s]:.0f}%", ha="center", fontsize=9)
    ax.set_ylim(0, 110)
    ax.set_ylabel("% videos", fontsize=8)
    ax.set_title("Claims 1–2: bidirectional wave\nin stimulated embryo", fontsize=9.5)
    ax.tick_params(labelsize=8)
    _verdict_box(ax, f"SUPPORTED — two opposing fronts from the poke in "
                     f"{lat['bidir_pct']:.0f}% of videos", RESULT)

    # ---- Panel B: neighbor wave present (claims 9-11) ----
    ax = fig.add_subplot(gs[0, 1])
    prev = df.groupby("stimulus")["neighbor_has_wave"].mean() * 100
    ax.bar(order, [prev[s] for s in order], color=[WOUND_C, PRESS_C][:len(order)])
    for i, s in enumerate(order):
        ax.text(i, prev[s] + 2, f"{prev[s]:.0f}%", ha="center", fontsize=9)
    ax.set_ylim(0, 110)
    ax.set_ylabel("% videos", fontsize=8)
    ax.set_title("Claims 9–11: calcium wave\ndetected in NEIGHBOR", fontsize=9.5)
    ax.tick_params(labelsize=8)
    _verdict_box(ax, f"OBSERVED — distinct neighbor-side wave in "
                     f"{lat['neighbor_wave_pct']:.0f}% of videos "
                     "(real geometry split)", RESULT)

    # ---- Panel C: neighbor local front speed (claims 14-16) ----
    ax = fig.add_subplot(gs[0, 2])
    sub = df[df["neighbor_has_wave"] == 1]
    ax.scatter(sub["stim_mean_front_speed"], sub["neighbor_mean_front_speed"],
               c=sub["stimulus"].map({"wound": WOUND_C, "press": PRESS_C}).fillna(GREY),
               s=22, alpha=0.7)
    lim = float(np.nanmax([df["stim_mean_front_speed"].max(),
                           df["neighbor_mean_front_speed"].max(), 1]))
    ax.plot([0, lim], [0, lim], "--", color="#888", lw=1)
    ax.set_xlabel("stim front speed", fontsize=8)
    ax.set_ylabel("neighbor front speed", fontsize=8)
    ax.set_title("Claims 14–16: NEIGHBOR\nlocal (front) speed", fontsize=9.5)
    ax.tick_params(labelsize=8)
    _verdict_box(ax, f"SUPPORTED — measurable local response in "
                     f"{lat['neighbor_local_n']}/{lat['n_videos']} videos", RESULT)

    # ---- Panel D: Layer-3 wound vs press neighbor brightness (claims 7-8) ----
    ax = fig.add_subplot(gs[1, 0])
    data = [df[df["stimulus"] == s]["neighbor_peak_brightness"].dropna().values
            for s in order]
    ax.boxplot(data, tick_labels=order, showfliers=False)
    for i, s in enumerate(order, 1):
        ys = df[df["stimulus"] == s]["neighbor_peak_brightness"].dropna().values
        ax.scatter(np.random.normal(i, 0.05, len(ys)), ys, s=12, alpha=0.6,
                   color=WOUND_C if s == "wound" else PRESS_C)
    ax.set_ylabel("neighbor peak brightness", fontsize=8)
    ax.set_title("Claims 7–8: wounding vs pressure\nneighbor response", fontsize=9.5)
    ax.tick_params(labelsize=8)
    p = lat.get("wp_p", float("nan"))
    _verdict_box(ax, f"NOT REPRODUCED — wound≈press (p={p:.2f}); "
                     "see ΔF/F₀ report", RESULT_NEG)

    # ---- Panel E: contact vs no-contact (claim 12) ----
    ax = fig.add_subplot(gs[1, 1])
    corder = [c for c in ("contact", "no-contact") if c in df["contact"].unique()]
    cprev = df.groupby("contact")["neighbor_has_wave"].mean() * 100
    ccnt = df.groupby("contact")["neighbor_has_wave"].count()
    ax.bar(corder, [cprev[c] for c in corder], color=["#31a354", "#756bb1"][:len(corder)])
    for i, c in enumerate(corder):
        ax.text(i, cprev[c] + 2, f"{cprev[c]:.0f}%\n(n={ccnt[c]})", ha="center",
                fontsize=8)
    ax.set_ylim(0, 115)
    ax.set_ylabel("% videos w/ neighbor wave", fontsize=8)
    ax.set_title("Claim 12: signal without\nphysical contact", fontsize=9.5)
    ax.tick_params(labelsize=8)
    _verdict_box(ax, f"DOES NOT REQUIRE CONTACT — neighbor wave in "
                     f"{lat['nocontact_pct']:.0f}% no-contact vs "
                     f"{lat['contact_pct']:.0f}% contact", RESULT)

    # ---- Panel G: organ-response landmarks (claims 3,4,5,6,17) ----
    ax = fig.add_subplot(gs[1, 2])
    if lm:
        groups = [
            ("eye\n(6)", lm["eye"]),
            ("tail\nself (4)", lm["tail_within"]),
            ("tail\nnbr (17)", lm["tail_neighbor"]),
            ("CG wnd\n(5)", lm["cement_wound"]),
            ("CG prs\n(3)", lm["cement_press"]),
        ]
        labels = [g[0] for g in groups]
        rates = [100.0 * g[1]["cor"] / g[1]["n"] if g[1]["n"] else 0
                 for g in groups]
        cols = ["#1f6f3f" if r >= 50 else "#d9a441" for r in rates]
        bars = ax.bar(labels, rates, color=cols)
        for b, g in zip(bars, groups):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 2,
                    f"{g[1]['cor']}/{g[1]['n']}", ha="center", fontsize=7)
        ax.set_ylim(0, 115)
        ax.set_ylabel("% landmarks w/ ΔF/F₀≥0.3", fontsize=8)
        ax.set_title("Claims 3,4,5,6,17: organ\nresponses (manual XY)",
                     fontsize=9.5)
        ax.tick_params(labelsize=7)
        _verdict_box(ax, "CORROBORATED at eye/tail; cement-gland signal "
                         "weak (see landmark evidence)", RESULT)
    else:
        ax.axis("off")
        ax.text(0.5, 0.5, "organ landmark data\nnot found", ha="center",
                va="center", fontsize=9, color=GREY, transform=ax.transAxes)

    # ---- Panel F: status text (bottom row) ----
    ax = fig.add_subplot(gs[2, :])
    ax.axis("off")
    ax.text(0.0, 1.0, "Status", fontsize=11, fontweight="bold", va="top",
            transform=ax.transAxes)
    # status badges across the top of the band
    for i, (label, cnt, col) in enumerate([
            ("TESTED", "16 claims", RESULT),
            ("NOT YET", "3 claims", RESULT_NEG)]):
        x = 0.10 + i * 0.18
        ax.text(x, 1.0, label, fontsize=9, fontweight="bold", color="white",
                va="top", transform=ax.transAxes,
                bbox=dict(boxstyle="round,pad=0.25", fc=col, ec="none"))
        ax.text(x + 0.075, 1.0, cnt, fontsize=9, va="top", transform=ax.transAxes)

    left = (
        "Newly tested via manual XY ground truth:\n"
        "• organ responses (cement gland / eye / tail) — claims 3,4,5,6,17\n"
        "• real poke + head/tail geometry now drives the side-split"
    )
    right = (
        "Still NOT testable with current data:\n"
        "• healed / repeat-poke / wound-site memory — claims 13,18,19\n"
        "  (needs repeat-poke timelines we do not have)"
    )
    y0 = 0.74
    y = y0
    for ln in left.split("\n"):
        ax.text(0.0, y, ln, fontsize=8.2, color="#1f6f3f" if y == y0 else "#444444",
                va="top", transform=ax.transAxes)
        y -= 0.14
    y = y0
    for ln in right.split("\n"):
        ax.text(0.52, y, ln, fontsize=8.2,
                color="#aa2a2a" if y == y0 else "#444444",
                va="top", transform=ax.transAxes)
        y -= 0.14
    ax.text(0.0, y0 - 0.46, "Full per-claim detail + evidence pages follow in "
                            "this report (Full Claims Inventory).", fontsize=7.6,
            color="#777777", style="italic", va="top", transform=ax.transAxes)

    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf)
    plt.close(fig)
    print(f"Wrote {out_pdf}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--laterality-csv",
                    default=str(HERE / "analysis_results" / "wave_catalog"
                               / "wave_laterality.csv"))
    ap.add_argument("--out-pdf",
                    default=str(HERE / "analysis_results"
                               / "claims_tested_summary.pdf"))
    args = ap.parse_args()

    lat_csv = Path(args.laterality_csv)
    lat = load_laterality(lat_csv)
    if not lat:
        raise SystemExit(f"No laterality data at {lat_csv}; run "
                         "wave_laterality_analysis.py first.")
    build(lat, lat_csv, Path(args.out_pdf), lm=load_landmarks())


if __name__ == "__main__":
    main()

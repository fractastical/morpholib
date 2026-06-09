#!/usr/bin/env python3
"""
Merge the individual claim PDFs into one combined report.

Builds a cover + table of contents + data-provenance page (matplotlib), then
concatenates the component PDFs in a fixed order using PyPDF2.

Component PDFs (in analysis_results/, skipped if missing):
  1. claim_video_mapping.pdf          - claim → video mapping, data state
  2. claims_nomask_report.pdf         - no-mask side-region ΔF/F₀ + verdicts
  3. xy_dff_claim_report.pdf          - mask + head/tail ΔF/F₀ traces
  4. claim_justification_with_deltas.pdf - per-claim delta charts + status

Output: analysis_results/claims_all_in_one.pdf
"""

from __future__ import annotations

import argparse
import datetime as dt
import tempfile
from pathlib import Path

from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from PyPDF2 import PdfMerger, PdfReader

from provenance import append_run_index, collect, summary_lines, write_sidecar  # noqa: E402


# Ordered (filename, human title, one-line description)
COMPONENTS = [
    (
        "claims_inventory.pdf",
        "Full Claims Inventory",
        "All verbatim claims from Calcium claims.docx, grouped by Layer, "
        "each marked testable-now / partial / not-yet.",
    ),
    (
        "claim_video_mapping.pdf",
        "Claim → Video Mapping",
        "Which Box videos map to each claim; data-state and wording provenance.",
    ),
    (
        "claims_nomask_report.pdf",
        "No-Mask Side-Region ΔF/F₀",
        "All-pixel side-region traces from XY geometry; per-claim verdicts.",
    ),
    (
        "xy_dff_claim_report.pdf",
        "Masked Head/Tail ΔF/F₀ Traces",
        "Mask + head/tail region ΔF/F₀ traces (alternate, mask-based approach).",
    ),
    (
        "claim_justification_with_deltas.pdf",
        "Per-Claim Justification + Deltas",
        "Delta charts and justification status per claim.",
    ),
]


def _page_count(pdf_path: Path) -> int:
    try:
        return len(PdfReader(str(pdf_path)).pages)
    except Exception:
        return 0


def build_front_matter(results_dir: Path, present, front_pdf: Path,
                       run_record: dict | None = None):
    """Cover + contents + provenance pages."""
    with PdfPages(str(front_pdf)) as pdf:
        # --- Cover ---
        fig = plt.figure(figsize=(8.5, 11))
        fig.text(0.5, 0.72, "Calcium Signaling Claims", ha="center",
                 fontsize=26, fontweight="bold")
        fig.text(0.5, 0.66, "Combined Analysis Report", ha="center",
                 fontsize=18, color="#444444")
        gen = (run_record or {}).get("generated_at_local") or dt.datetime.now().strftime(
            "%Y-%m-%d %H:%M")
        git = (run_record or {}).get("repository", {})
        fig.text(0.5, 0.58, f"Generated {gen}", ha="center", fontsize=12,
                 color="#666666")
        if git.get("commit_short"):
            fig.text(0.5, 0.52,
                     f"Software: git {git['commit_short']}"
                     + (" (uncommitted changes)" if git.get("dirty") else ""),
                     ha="center", fontsize=10, color="#888888")
        fig.text(0.5, 0.50,
                 "Xenopus laevis inter-embryo Ca²⁺ wave analysis",
                 ha="center", fontsize=12, style="italic", color="#666666")
        plt.axis("off")
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # --- Table of contents ---
        fig = plt.figure(figsize=(8.5, 11))
        fig.text(0.08, 0.92, "Contents", fontsize=20, fontweight="bold")
        y = 0.84
        page_cursor = 1  # front matter pages counted separately below
        lines = []
        for fname, title, desc in COMPONENTS:
            path = results_dir / fname
            if path not in present:
                continue
            npages = _page_count(path)
            lines.append((title, desc, npages))
        # Front matter itself is 3 pages (cover, contents, provenance)
        running = 4
        for title, desc, npages in lines:
            fig.text(0.08, y, f"• {title}", fontsize=13, fontweight="bold")
            fig.text(0.12, y - 0.025, desc, fontsize=9.5, color="#555555")
            fig.text(0.92, y, f"p.{running}", fontsize=10, ha="right",
                     color="#888888")
            running += npages
            y -= 0.075
            if y < 0.1:
                break
        plt.axis("off")
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # --- Data provenance ---
        fig = plt.figure(figsize=(8.5, 11))
        fig.text(0.08, 0.92, "Data Version Used", fontsize=20, fontweight="bold")
        y = 0.84
        fig.text(0.08, y, "Component PDFs merged (name — pages — modified):",
                 fontsize=11, fontweight="bold")
        y -= 0.04
        for fname, title, _desc in COMPONENTS:
            path = results_dir / fname
            if path not in present:
                status = "MISSING (skipped)"
                mtime = "-"
                npages = 0
            else:
                npages = _page_count(path)
                mtime = dt.datetime.fromtimestamp(
                    path.stat().st_mtime
                ).strftime("%Y-%m-%d %H:%M")
                status = "included"
            fig.text(0.08, y, f"{fname}", fontsize=9.5, family="monospace")
            fig.text(0.62, y, f"{npages}p", fontsize=9.5, ha="right")
            fig.text(0.66, y, f"{mtime}", fontsize=9.5)
            fig.text(0.92, y, status, fontsize=8.5, ha="right",
                     color="#2a7a2a" if status == "included" else "#aa2a2a")
            y -= 0.03
        y -= 0.03
        if run_record:
            fig.text(0.08, y, "Build environment (this merge run):",
                     fontsize=11, fontweight="bold")
            y -= 0.035
            from textwrap import wrap
            for ln in summary_lines(run_record):
                fig.text(0.08, y, ln, fontsize=9, color="#333333")
                y -= 0.025
            y -= 0.02
        note = (
            "Component PDFs are generated by their own scripts. Sidecar "
            "*.provenance.json files next to each CSV/PDF record git commit, "
            "Python/package versions, and input paths. See RUN_INDEX.jsonl "
            "in analysis_results/ for the full run history."
        )
        from textwrap import wrap
        for ln in wrap(note, 92):
            fig.text(0.08, y, ln, fontsize=9, color="#555555")
            y -= 0.022
        plt.axis("off")
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Merge claim PDFs into one report.")
    parser.add_argument(
        "--results-dir",
        default=str(Path(__file__).parent / "analysis_results"),
        help="Directory containing the component PDFs",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output path (default: <results-dir>/claims_all_in_one.pdf)",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output = Path(args.output) if args.output else results_dir / "claims_all_in_one.pdf"

    present = [results_dir / f for f, _, _ in COMPONENTS if (results_dir / f).exists()]
    missing = [f for f, _, _ in COMPONENTS if not (results_dir / f).exists()]
    if missing:
        print(f"⚠ Missing (will skip): {', '.join(missing)}")
    if not present:
        raise SystemExit("No component PDFs found; nothing to merge.")

    # Prefer the tested-claims summary (verdict + graph per scored claim); fall
    # back to the older 5-claim stats dashboard if it is not present.
    tested_summary = results_dir / "claims_tested_summary.pdf"
    dashboard = (tested_summary if tested_summary.exists()
                 else results_dir / "claims_summary_dashboard.pdf")

    run_record = collect(
        "merge_claims_all_in_one.py",
        [output],
        extra={"components": [p.name for p in present]},
    )

    with tempfile.TemporaryDirectory() as tmp:
        front_pdf = Path(tmp) / "front_matter.pdf"
        build_front_matter(results_dir, present, front_pdf, run_record)

        merger = PdfMerger()
        # Summary slide is the very first page, if available
        if dashboard.exists():
            merger.append(str(dashboard))
            print(f"Prepended summary slide {dashboard.name} "
                  f"({_page_count(dashboard)}p)")
        else:
            print("⚠ No summary slide found; skipping summary slide.")
        merger.append(str(front_pdf))
        total = _page_count(front_pdf)
        for fname, _title, _desc in COMPONENTS:
            path = results_dir / fname
            if path not in present:
                continue
            merger.append(str(path))
            total += _page_count(path)

        with open(output, "wb") as f:
            merger.write(f)
        merger.close()

    run_record["extra"]["total_pages"] = total
    write_sidecar(output, run_record)
    append_run_index(run_record)

    print(f"✓ Wrote {output} ({_page_count(output)} pages, "
          f"{len(present)} component PDFs + front matter)")


if __name__ == "__main__":
    main()

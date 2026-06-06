#!/usr/bin/env python3
"""
Build a PDF that maps each claim to applicable videos.

This is a planning/QC artifact: it does not run signal analysis.
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional
from collections import Counter
from zipfile import ZipFile
from textwrap import wrap

import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


@dataclass
class PairRecord:
    sheet: str
    b_video: str
    c_video: str
    condition: str
    orientation: str
    b_path: Optional[Path]
    c_path: Optional[Path]
    contact_group: str

    @property
    def pair_id(self) -> str:
        stem = Path(self.c_video).stem if self.c_video else Path(self.b_video).stem
        return stem

    @property
    def matched(self) -> bool:
        return self.b_path is not None and self.c_path is not None


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", str(s).strip())


def parse_condition(video_name: str) -> str:
    m = re.match(r"([A-Za-z]{2})\d+_", Path(video_name).stem)
    return m.group(1).upper() if m else "UNK"


def parse_orientation(video_name: str) -> str:
    m = re.search(r"-o([A-Za-z]+)", Path(video_name).stem, flags=re.IGNORECASE)
    return m.group(1).upper() if m else "UNK"


def read_pairs_from_xy(xy_path: Path) -> List[PairRecord]:
    xl = pd.ExcelFile(xy_path)
    out: List[PairRecord] = []
    for sheet in xl.sheet_names:
        df = pd.read_excel(xy_path, sheet_name=sheet, header=None)
        col0 = df.iloc[:, 0].astype(str).str.strip()
        b_idx = col0[col0.str.lower() == "background video"].index.tolist()
        c_idx = col0[col0.str.lower() == "calcium video"].index.tolist()
        if not b_idx or not c_idx:
            continue
        bi, ci = int(b_idx[0]), int(c_idx[0])

        b_name = ""
        c_name = ""
        for r in range(bi + 1, min(len(df), bi + 6)):
            v = df.iloc[r, 0]
            if pd.notna(v) and str(v).strip():
                b_name = norm(v)
                break
        for r in range(ci + 1, min(len(df), ci + 6)):
            v = df.iloc[r, 0]
            if pd.notna(v) and str(v).strip():
                c_name = norm(v)
                break
        if not b_name or not c_name:
            continue

        cond = parse_condition(c_name)
        orient = parse_orientation(c_name)
        out.append(
            PairRecord(
                sheet=sheet,
                b_video=b_name,
                c_video=c_name,
                condition=cond,
                orientation=orient,
                b_path=None,
                c_path=None,
                contact_group="unknown",
            )
        )
    return out


def build_tiff_index(root: Path) -> Dict[str, Path]:
    idx: Dict[str, Path] = {}
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in (".tif", ".tiff"):
            idx.setdefault(p.name.lower(), p)
            idx.setdefault(p.stem.lower(), p)
    return idx


def find_video_path(name: str, idx: Dict[str, Path]) -> Optional[Path]:
    raw = norm(name)
    cands = [raw.lower(), Path(raw).name.lower(), Path(raw).stem.lower()]
    for c in cands:
        if c in idx:
            return idx[c]
    return None


def classify_contact(path: Optional[Path]) -> str:
    if path is None:
        return "unknown"
    p = str(path).lower()
    if "no physical contact" in p:
        return "no-contact"
    if "physical contact" in p:
        return "contact"
    return "unknown"


def claim_filters(records: List[PairRecord]) -> Dict[str, List[PairRecord]]:
    return {
        "Claim 1": [
            r for r in records
            if r.condition in {"PS", "WS"} and r.orientation == "HT"
        ],
        "Claim 2": [
            r for r in records
            if r.condition in {"PG", "WG"} and r.orientation == "HTHT"
        ],
        "Claim 3": [
            r for r in records
            if r.condition == "WG" and r.orientation in {"THHT", "HTTH", "HTHT"}
        ],
        "Claim 4": [
            r for r in records
            if r.condition == "WG" and r.orientation == "HTHT"
        ],
        "Claim 5 (all orientations)": [
            r for r in records
            if r.orientation in {"HT", "HTHT", "HTTH", "THHT"}
        ],
        "Claim 5 (HTHT contact split)": [
            r for r in records
            if r.orientation == "HTHT"
        ],
    }


def add_text_pages(pdf: PdfPages, title: str, subtitle: str, lines: List[str], per_page: int = 34) -> None:
    if not lines:
        lines = ["(none)"]

    # Wrap long lines to avoid clipping past right page margin.
    wrapped: List[str] = []
    for ln in lines:
        s = str(ln) if ln is not None else ""
        pieces = wrap(
            s,
            width=120,
            break_long_words=False,
            break_on_hyphens=False,
        )
        if pieces:
            wrapped.extend(pieces)
        else:
            wrapped.append("")

    pages = (len(wrapped) + per_page - 1) // per_page
    for i in range(pages):
        chunk = wrapped[i * per_page:(i + 1) * per_page]
        fig = plt.figure(figsize=(11, 8.5))
        fig.text(0.04, 0.96, title, fontsize=14, va="top")
        fig.text(0.04, 0.92, subtitle, fontsize=10, va="top")
        fig.text(0.96, 0.96, f"Page {i+1}/{pages}", fontsize=9, ha="right", va="top")
        y = 0.88
        for ln in chunk:
            fig.text(0.04, y, ln, fontsize=8.8, family="monospace", va="top")
            y -= 0.024
        pdf.savefig(fig)
        plt.close(fig)


def extract_docx_paragraphs(docx_path: Path) -> List[str]:
    if not docx_path.exists():
        return []
    try:
        with ZipFile(docx_path) as zf:
            xml = zf.read("word/document.xml").decode("utf-8", errors="ignore")
    except Exception:
        return []
    paras: List[str] = []
    for para in re.findall(r"<w:p[\s\S]*?</w:p>", xml):
        ts = re.findall(r"<w:t[^>]*>(.*?)</w:t>", para)
        if not ts:
            continue
        s = "".join(ts).strip()
        s = s.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
        if s:
            paras.append(s)
    return paras


def find_first(paras: List[str], includes_all: List[str]) -> str:
    for p in paras:
        pl = p.lower()
        if all(tok.lower() in pl for tok in includes_all):
            return p
    return ""


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate claim-to-video mapping PDF.")
    ap.add_argument("--xy-path", default="/Users/jdietz/Library/CloudStorage/Box-Box/Calcium videos/XY coordinates.xlsx")
    ap.add_argument("--data-root", default="/Users/jdietz/Library/CloudStorage/Box-Box/Calcium videos/Calcium videos")
    ap.add_argument("--claims-docx", default="/Users/jdietz/Library/CloudStorage/Box-Box/Calcium videos/Calcium claims.docx")
    ap.add_argument("--output-pdf", default="wave-vector-analysis/analysis_results/claim_video_mapping.pdf")
    args = ap.parse_args()

    xy_path = Path(args.xy_path)
    data_root = Path(args.data_root)
    claims_docx = Path(args.claims_docx)
    out_pdf = Path(args.output_pdf)
    out_pdf.parent.mkdir(parents=True, exist_ok=True)

    records = read_pairs_from_xy(xy_path)
    idx = build_tiff_index(data_root)
    for r in records:
        r.b_path = find_video_path(r.b_video, idx)
        r.c_path = find_video_path(r.c_video, idx)
        r.contact_group = classify_contact(r.c_path)

    claims = claim_filters(records)
    matched_records = [r for r in records if r.matched]
    claim_doc_paras = extract_docx_paragraphs(claims_docx)

    # Data-state summaries for "what do we currently have?"
    cond_counts_all = Counter(r.condition for r in records)
    cond_counts_matched = Counter(r.condition for r in matched_records)
    orient_counts_all = Counter(r.orientation for r in records)
    orient_counts_matched = Counter(r.orientation for r in matched_records)
    contact_counts_ht = Counter(r.contact_group for r in records if r.orientation == "HTHT")
    contact_counts_ht_matched = Counter(r.contact_group for r in matched_records if r.orientation == "HTHT")

    with PdfPages(out_pdf) as pdf:
        # overview
        fig = plt.figure(figsize=(11, 8.5))
        fig.text(0.04, 0.95, "Claim -> Video Mapping", fontsize=16, va="top")
        fig.text(0.04, 0.90, f"XY source: {xy_path}", fontsize=9)
        fig.text(0.04, 0.87, f"TIFF root: {data_root}", fontsize=9)
        fig.text(0.04, 0.84, f"Claims source: {claims_docx}", fontsize=9)
        fig.text(0.04, 0.80, f"Total B/C pairs parsed from XY: {len(records)}", fontsize=11)
        matched = sum(1 for r in records if r.matched)
        fig.text(0.04, 0.77, f"Pairs with both B and C files found: {matched}", fontsize=11)
        y = 0.72
        for cname, rows in claims.items():
            m = sum(1 for r in rows if r.matched)
            fig.text(0.06, y, f"{cname}: {len(rows)} candidate pairs ({m} matched)", fontsize=10)
            y -= 0.035
        pdf.savefig(fig)
        plt.close(fig)

        # Data state page
        fig = plt.figure(figsize=(11, 8.5))
        fig.text(0.04, 0.95, "State of Data at Present", fontsize=16, va="top")
        fig.text(0.04, 0.90, "This section summarizes current usable coverage before statistical testing.", fontsize=10)
        y = 0.85
        fig.text(0.04, y, f"Total XY B/C pairs: {len(records)}", fontsize=11); y -= 0.035
        fig.text(0.04, y, f"Matched B/C TIFF pairs: {len(matched_records)}", fontsize=11); y -= 0.05

        fig.text(0.04, y, "By condition (all / matched):", fontsize=11); y -= 0.03
        for k in sorted(set(cond_counts_all) | set(cond_counts_matched)):
            fig.text(0.06, y, f"{k}: {cond_counts_all.get(k,0)} / {cond_counts_matched.get(k,0)}", fontsize=10)
            y -= 0.027
        y -= 0.02

        fig.text(0.04, y, "By orientation (all / matched):", fontsize=11); y -= 0.03
        for k in sorted(set(orient_counts_all) | set(orient_counts_matched)):
            fig.text(0.06, y, f"{k}: {orient_counts_all.get(k,0)} / {orient_counts_matched.get(k,0)}", fontsize=10)
            y -= 0.027
        y -= 0.02

        fig.text(0.04, y, "HTHT contact labels (all / matched):", fontsize=11); y -= 0.03
        for k in ["contact", "no-contact", "unknown"]:
            fig.text(0.06, y, f"{k}: {contact_counts_ht.get(k,0)} / {contact_counts_ht_matched.get(k,0)}", fontsize=10)
            y -= 0.027
        pdf.savefig(fig)
        plt.close(fig)

        # Claim-adjustment guidance page
        claim_notes = []
        c1_n = len(claims["Claim 1"])
        c2_n = len(claims["Claim 2"])
        c3_n = len(claims["Claim 3"])
        c4_n = len(claims["Claim 4"])
        c5_n = len(claims["Claim 5 (all orientations)"])
        c5_ht_n = len(claims["Claim 5 (HTHT contact split)"])
        ht_contact = contact_counts_ht_matched.get("contact", 0)
        ht_nocontact = contact_counts_ht_matched.get("no-contact", 0)

        claim_notes.append(f"Claim 1 (PS/WS, oHT): {c1_n} candidate pairs. Current data likely sufficient for initial effect test.")
        claim_notes.append(f"Claim 2 (PG/WG, HTHT): {c2_n} candidate pairs. Suitable for initial neighbor-response comparison.")
        claim_notes.append(f"Claim 3 (WG, THHT/HTTH/HTHT): {c3_n} candidate pairs. Good coverage; define positional fidelity metric before final claim language.")
        if ht_contact > 0 and ht_nocontact > 0:
            claim_notes.append(
                f"Claim 4 (WG HTHT contact vs no-contact): contact={ht_contact}, no-contact={ht_nocontact}. "
                "Two-group comparison is feasible."
            )
        else:
            claim_notes.append(
                f"Claim 4 (WG HTHT contact vs no-contact): contact={ht_contact}, no-contact={ht_nocontact}. "
                "Adjust claim wording to exploratory unless both groups have enough samples."
            )
        claim_notes.append(
            f"Claim 5 (speed analyses): {c5_n} all-orientation pairs, {c5_ht_n} HTHT pairs. "
            "Speed columns already exist in spark outputs (spark_tracks: vx/vy/speed; vector_clusters: mean/net/peak speed); "
            "remaining work is to classify tracks/clusters as wave vs local and then run stratified tests."
        )
        claim_notes.append(
            "Recommended immediate adjustment workflow: lock video roster first, then tune claim wording to "
            "match current sample counts and available measurement type (intensity vs speed)."
        )

        add_text_pages(
            pdf,
            "Data Used to Adjust Claims",
            "Guidance based on currently matched data coverage.",
            [f"- {ln}" for ln in claim_notes],
            per_page=26,
        )

        # Original vs adjusted claim wording (what user asked to see "both")
        # Pull "original" wording from Box claims doc when possible.
        c1_orig_ws = find_first(claim_doc_paras, ["wounding", "within an embryo"])
        c1_orig_ps = find_first(claim_doc_paras, ["pressure", "within an embryo"])
        c2_orig_wg = find_first(claim_doc_paras, ["wounding", "neighboring embryo"])
        c2_orig_pg = find_first(claim_doc_paras, ["pressure", "neighboring embryo"])
        c3_orig = find_first(claim_doc_paras, ["local response", "neighbor"])
        c4_orig = find_first(claim_doc_paras, ["does not require physical contact"])
        c5_orig = find_first(claim_doc_paras, ["calcium wave", "neighbor"])

        original_vs_adjusted = []
        original_vs_adjusted.append(
            "Claim 1 (original from Calcium claims.docx): "
            + (f"{c1_orig_ps} / {c1_orig_ws}" if (c1_orig_ps or c1_orig_ws) else "[not found in document by keyword]")
        )
        original_vs_adjusted.append(
            "Claim 1 (adjusted): In the current matched PS/WS oHT dataset (n="
            f"{c1_n}), both conditions are testable for stimulated-embryo response; report effect sizes and uncertainty, "
            "and treat similarity as provisional until formal equivalence/non-inferiority criteria are met."
        )
        original_vs_adjusted.append(
            "Upgrade path: add equivalence margin + pre-registered similarity test."
        )
        original_vs_adjusted.append("")

        original_vs_adjusted.append(
            "Claim 2 (original from Calcium claims.docx): "
            + (f"{c2_orig_wg} / {c2_orig_pg}" if (c2_orig_wg or c2_orig_pg) else "[not found in document by keyword]")
        )
        original_vs_adjusted.append(
            "Claim 2 (adjusted): In the current matched PG/WG HTHT dataset (n="
            f"{c2_n}), neighbor response can be compared directly; phrase as "
            "\"evidence for differential neighbor response\" until directional significance criteria are satisfied."
        )
        original_vs_adjusted.append(
            "Upgrade path: ensure balanced PG vs WG sample sizes and lock peak-window definition."
        )
        original_vs_adjusted.append("")

        original_vs_adjusted.append(
            "Claim 3 (original from Calcium claims.docx): "
            + (c3_orig if c3_orig else "[not found in document by keyword]")
        )
        original_vs_adjusted.append(
            "Claim 3 (adjusted): In WG THHT/HTTH/HTHT pairs (n="
            f"{c3_n}), positional correspondence is assessable; phrase as "
            "\"preliminary evidence of position-dependent mapping\" pending a fixed null model/threshold."
        )
        original_vs_adjusted.append(
            "Upgrade path: define null permutation baseline and a preregistered AP-mismatch cutoff."
        )
        original_vs_adjusted.append("")

        original_vs_adjusted.append(
            "Claim 4 (original from Calcium claims.docx): "
            + (c4_orig if c4_orig else "[not found in document by keyword]")
        )
        original_vs_adjusted.append(
            "Claim 4 (adjusted): In matched WG HTHT pairs (n="
            f"{c4_n}, contact={ht_contact}, no-contact={ht_nocontact}), "
            "contact-dependence is testable; keep wording conservative unless both groups are sufficiently represented."
        )
        original_vs_adjusted.append(
            "Upgrade path: increase no-contact and contact counts if imbalance remains."
        )
        original_vs_adjusted.append("")

        original_vs_adjusted.append(
            "Claim 5 (original from Calcium claims.docx): "
            + (c5_orig if c5_orig else "[speed-specific wording not found verbatim in document]")
        )
        original_vs_adjusted.append(
            "Claim 5 (adjusted): Current roster (all orientations n="
            f"{c5_n}, HTHT n={c5_ht_n}) is ready for speed analysis planning, "
            "and speed metrics are already present in spark outputs; "
            "claim finalization now depends on explicit wave-vs-local classification and contact-stratified testing."
        )
        original_vs_adjusted.append(
            "Upgrade path: compute wave/local speed metrics first, then run orientation/contact stratified tests."
        )

        add_text_pages(
            pdf,
            "Original vs Adjusted Claim Wording",
            "Both versions shown side-by-side in text form, with a concrete upgrade path.",
            [f"- {ln}" for ln in original_vs_adjusted],
            per_page=24,
        )

        # per claim lists
        for cname, rows in claims.items():
            rows_sorted = sorted(rows, key=lambda r: (r.condition, r.orientation, r.sheet, r.pair_id))
            lines = []
            for r in rows_sorted:
                b_ok = "Y" if r.b_path else "N"
                c_ok = "Y" if r.c_path else "N"
                lines.append(
                    f"{r.condition:<3} {r.orientation:<5} {r.contact_group:<10} "
                    f"B:{b_ok} C:{c_ok} | {r.sheet[:24]:<24} | {r.pair_id}"
                )
            subtitle = "Columns: cond orient contact Bfound Cfound | sheet | pair_id"
            add_text_pages(pdf, cname, subtitle, lines)

        # unmatched pairs
        missing = [r for r in records if not r.matched]
        miss_lines = []
        for r in sorted(missing, key=lambda x: (x.sheet, x.pair_id)):
            miss_lines.append(
                f"{r.sheet[:28]:<28} | {r.pair_id:<35} | B_found={'Y' if r.b_path else 'N'} C_found={'Y' if r.c_path else 'N'}"
            )
        add_text_pages(
            pdf,
            "Unmatched/Incomplete Pairs",
            "Pairs from XY where one or both TIFF files were not found by filename matching.",
            miss_lines,
        )

    print(f"Generated PDF: {out_pdf}")
    print(f"Total pairs: {len(records)}")
    print(f"Matched pairs: {sum(1 for r in records if r.matched)}")
    for cname, rows in claims.items():
        print(f"{cname}: {len(rows)} candidates, {sum(1 for r in rows if r.matched)} matched")


if __name__ == "__main__":
    main()


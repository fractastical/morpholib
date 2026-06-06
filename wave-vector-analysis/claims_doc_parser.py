#!/usr/bin/env python3
"""
Parse the official `Calcium claims.docx` table into structured claim records.

The document is a Word table with columns:
  Layer | Pattern | Claim | Orientation | Poke location | N= | Observed | Not observed

This module exposes `parse_claims_doc(path) -> list[dict]` and, when run
directly, dumps the parsed records so they can be verified against the doc.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from zipfile import ZipFile

DEFAULT_DOCX = Path(
    "/Users/jdietz/Library/CloudStorage/Box-Box/Calcium videos/Calcium claims.docx"
)

COLUMNS = ["layer", "pattern", "claim", "orientation", "poke", "n",
           "observed", "not_observed"]


def _cell_text(cell_xml: str) -> str:
    """Join all <w:t> runs in a table cell, preserving paragraph breaks."""
    parts = []
    for para in re.split(r"</w:p>", cell_xml):
        # Match the <w:t> run-text tag EXACTLY (not <w:tcPr>, <w:tbl>, ...)
        texts = re.findall(r"<w:t(?:\s[^>]*)?>(.*?)</w:t>", para, flags=re.S)
        line = "".join(texts)
        # unescape common XML entities
        line = (line.replace("&amp;", "&").replace("&lt;", "<")
                .replace("&gt;", ">").replace("&quot;", '"')
                .replace("&apos;", "'"))
        if line.strip():
            parts.append(line.strip())
    return " ".join(parts).strip()


def _split_cells(row_xml: str) -> list[str]:
    """Return the text of each <w:tc> in a row."""
    cells = re.findall(r"<w:tc\b.*?</w:tc>", row_xml, flags=re.S)
    return [_cell_text(c) for c in cells]


def parse_claims_doc(path: Path = DEFAULT_DOCX) -> list[dict]:
    path = Path(path)
    xml = ZipFile(path).read("word/document.xml").decode("utf-8", "ignore")

    rows = re.findall(r"<w:tr\b.*?</w:tr>", xml, flags=re.S)
    records = []
    last_layer = ""
    last_pattern = ""
    for row_xml in rows:
        cells = _split_cells(row_xml)
        if len(cells) < 3:
            continue
        # Skip the header row
        joined = " ".join(cells).lower()
        if "claim" in cells[2].lower() and "orientation" in joined:
            continue
        if cells[2].strip().lower() == "claim":
            continue

        # pad to 8 columns
        cells = (cells + [""] * 8)[:8]
        layer, pattern, claim, orientation, poke, n, observed, not_obs = cells

        # carry forward merged Layer / Pattern cells
        layer = layer.strip() or last_layer
        pattern = pattern.strip() or last_pattern
        if cells[0].strip():
            last_layer = cells[0].strip()
        if cells[1].strip():
            last_pattern = cells[1].strip()

        claim = claim.strip()
        # A real claim row must contain a claim sentence
        if not re.search(r"(produces|increase|response|wave|signal|contact|"
                         r"re-occur|pattern)", claim, re.I):
            continue
        if len(claim.split()) < 4:
            continue

        records.append({
            "layer": layer,
            "pattern": pattern,
            "claim": claim,
            "orientation": orientation.strip(),
            "poke": poke.strip(),
            "n": n.strip(),
            "observed": observed.strip(),
            "not_observed": not_obs.strip(),
        })
    return records


if __name__ == "__main__":
    docx = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_DOCX
    recs = parse_claims_doc(docx)
    print(f"# Parsed {len(recs)} claim rows from {docx.name}\n")
    for i, r in enumerate(recs, 1):
        print(f"[{i:02d}] L{r['layer']} | {r['pattern']!r}")
        print(f"     CLAIM: {r['claim']}")
        print(f"     orient={r['orientation']!r} poke={r['poke']!r} "
              f"N={r['n']!r}")
        print(f"     observed={r['observed']!r}")
        print(f"     not_observed={r['not_observed']!r}")
    # also emit JSON for programmatic checking
    Path("/tmp/claims_parsed.json").write_text(json.dumps(recs, indent=2))
    print("\nWrote /tmp/claims_parsed.json")

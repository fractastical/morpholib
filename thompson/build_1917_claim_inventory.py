#!/usr/bin/env python3
"""
Build / check the Thompson (1917) claim inventory.

Pass 1 (citation):  thompson/thompson_1917_claims.json -- every distinct testable
claim in D'Arcy Thompson, *On Growth and Form* (Cambridge University Press,
**1917**, 1st edition), cited by chapter, printed pages and figure numbers.
Pass 2 (validation): the `pass2` block of each row -- status against this repo's
artifacts, what remains, method, data needed, and effort in person-days.

This script (a) validates the registry (schema, status/effort codes, that every
repo_evidence path exists, that every claim is dated), (b) optionally verifies
chapter/figure anchors against the public-domain 1917 full text from the
Internet Archive (SHA-256 pinned in the registry; cached, git-ignored, under
.thompson_cache/claim_inventory/), and (c) regenerates the human-readable
companion docs/THOMPSON_1917_CLAIM_INVENTORY.md so the two never drift.

Usage:
    python3 thompson/build_1917_claim_inventory.py            # validate + write MD
    python3 thompson/build_1917_claim_inventory.py --check    # validate only (CI)
    python3 thompson/build_1917_claim_inventory.py --no-text  # skip IA text check
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = Path(__file__).with_name("thompson_1917_claims.json")
OUT_MD = ROOT / "docs" / "THOMPSON_1917_CLAIM_INVENTORY.md"
CACHE_TXT = ROOT / ".thompson_cache" / "claim_inventory" / "ongrowthform1917thom_djvu.txt"

STATUSES = {"DONE_REPO", "PARTIAL_REPO", "DONE_ELSEWHERE", "UNTESTED", "NOT_TESTABLE_AS_STATED"}
CLASSES = {"S", "M", "L", "XL"}


def validate(reg: dict, check_text: bool) -> list[str]:
    errs: list[str] = []
    src = reg["source"]
    if src.get("date_made") != "1917":
        errs.append("source.date_made must be '1917' -- this registry is the 1917 edition only")
    seen = set()
    for c in reg["claims"]:
        cid = c["id"]
        if cid in seen:
            errs.append(f"{cid}: duplicate id")
        seen.add(cid)
        if c.get("date_made") != "1917":
            errs.append(f"{cid}: date_made != 1917 (state the year on every claim)")
        for k in ("chapter", "pages", "figures", "claim"):
            if not c.get(k):
                errs.append(f"{cid}: missing citation field '{k}'")
        p2 = c["pass2"]
        if p2["status"] not in STATUSES:
            errs.append(f"{cid}: bad status {p2['status']}")
        if p2["effort_class"] not in CLASSES:
            errs.append(f"{cid}: bad effort_class {p2['effort_class']}")
        lo, hi = p2["effort_days"]
        if not (0 <= lo <= hi):
            errs.append(f"{cid}: bad effort_days {p2['effort_days']}")
        for ev in p2.get("repo_evidence", []):
            # evidence strings are 'path -- detail' prose; check the first file-like token, if any
            m = re.search(r"([A-Za-z0-9_./*\-]+\.(?:py|json|md|csv|txt))", ev)
            if not m:
                continue
            path = m.group(1)
            if "*" in path:
                if not list(ROOT.glob(path)):
                    errs.append(f"{cid}: repo_evidence glob matches nothing: {path}")
            elif not (ROOT / path).exists():
                errs.append(f"{cid}: repo_evidence path not found: {path}")
        if p2["status"] == "DONE_REPO" and not p2.get("repo_evidence"):
            errs.append(f"{cid}: DONE_REPO requires repo_evidence")
    if check_text:
        errs += verify_against_1917_text(reg)
    return errs


def verify_against_1917_text(reg: dict) -> list[str]:
    """Chapter headings and figure numbers must occur in the pinned 1917 text."""
    errs: list[str] = []
    if not CACHE_TXT.exists():
        CACHE_TXT.parent.mkdir(parents=True, exist_ok=True)
        url = reg["source"]["primary_text"]["full_text_url"]
        req = urllib.request.Request(url, headers={"User-Agent": "projectdelphi/1.0 (Thompson 1917 claim inventory)"})
        CACHE_TXT.write_bytes(urllib.request.urlopen(req).read())  # noqa: S310 - pinned URL + hash checked below
    data = CACHE_TXT.read_bytes()
    want = reg["source"]["primary_text"]["full_text_sha256"]
    got = hashlib.sha256(data).hexdigest()
    if got != want:
        return [f"1917 full-text SHA-256 mismatch: got {got}, registry pins {want} (refusing to verify against changed text)"]
    text = data.decode("utf-8", errors="ignore")
    chapters = {c["chapter"] for c in reg["claims"]}
    for ch in chapters:
        # Ch. XVII rows carry suffixes like 'XVII (fish I)'; check the first roman token.
        roman = ch.split()[0].split("-")[0]
        if roman == "Epilogue":
            if "EPILOGUE" not in text.upper():
                errs.append("EPILOGUE heading not found in 1917 text")
        elif not re.search(rf"CHAPTER\s+{roman}\b", text):
            errs.append(f"chapter heading 'CHAPTER {roman}' not found in 1917 text")
    # OCR drops some 'Fig. N' strings; the List of Illustrations (before Ch. I) is the fallback anchor.
    loi = text[: text.find("CHAPTER I")]
    for c in reg["claims"]:
        for fig in re.findall(r"\d+", c["figures"]):
            if not (re.search(rf"Fig\.\s*{fig}\b", text) or re.search(rf"(?m)^\s*{fig}[.,]", loi)):
                errs.append(f"{c['id']}: Fig. {fig} not found in 1917 text or List of Illustrations")
    return errs


def render_md(reg: dict) -> str:
    claims = reg["claims"]
    from collections import Counter

    st = Counter(c["pass2"]["status"] for c in claims)
    remaining = [c for c in claims if c["pass2"]["status"] in ("UNTESTED", "PARTIAL_REPO")]
    lo = sum(c["pass2"]["effort_days"][0] for c in remaining)
    hi = sum(c["pass2"]["effort_days"][1] for c in remaining)
    lines = [
        "# Thompson (1917) — Every Claim, Cited, Then Assessed for Validation",
        "",
        "**Source for every row below (state the date whenever you cite one):**",
        "Thompson, D'Arcy Wentworth (**1917**). *On Growth and Form*. 1st edition, Cambridge University Press.",
        "Internet Archive item [`ongrowthform1917thom`](https://archive.org/details/ongrowthform1917thom) (public domain);",
        f"full-text SHA-256 `{reg['source']['primary_text']['full_text_sha256']}`; printed page = IA leaf − 20.",
        "Not the 1942 2nd edition and not the 1961 abridgement. Machine-readable registry (the source of truth for this file):",
        "[`thompson/thompson_1917_claims.json`](../thompson/thompson_1917_claims.json), regenerated/checked by",
        "[`thompson/build_1917_claim_inventory.py`](../thompson/build_1917_claim_inventory.py).",
        "",
        "## How to read this (the two passes)",
        "",
        "- **Pass 1 — cite it.** One row per distinct testable assertion or figure-demonstration Thompson made in **1917**,",
        "  cited by chapter, printed pages and figure numbers. Where Thompson repeats one assertion across several named",
        "  example pairs (Ch. XVII), each pair is its own row because each can pass or fail separately. Verbatim quotes are",
        "  included only where already pinned in repo code; completing verbatim-quote extraction for every row is Pass-1",
        "  extension item E1 below (OCR-checked, ~3–5 days). Rhetorical/methodological passages are rows T1917-001/002/059",
        "  and must not be scored as if they were measurements.",
        "- **Pass 2 — can we validate it, and what would it cost?** Each row carries: status against this repo's actual",
        "  artifacts, the verdict where one exists, what remains, the method and data that would settle it, and effort in",
        "  person-days (S ≤2d, M 3–10d, L 2–6 wk, XL >6 wk; includes the audit write-up, reusing repo tooling).",
        "",
        f"**Count:** {len(claims)} claims — DONE_REPO {st['DONE_REPO']}, PARTIAL_REPO {st['PARTIAL_REPO']}, "
        f"DONE_ELSEWHERE {st['DONE_ELSEWHERE']}, UNTESTED {st['UNTESTED']}, NOT_TESTABLE_AS_STATED {st['NOT_TESTABLE_AS_STATED']}.",
        f"**Remaining work if we did everything left:** {lo}–{hi} person-days summed row-by-row; materially less in practice",
        "because rows bundle (one epithelium script covers T1917-018+019; one spiral fitter covers T1917-027+029+030+031+032;",
        "one Ch. XVII GM protocol covers T1917-051+055+056+058). Realistic bundled total: **~120–240 person-days**, of which",
        "the P0 set below is **~30–55 days**. DONE_ELSEWHERE rows need no repo work unless we want a reproduction.",
        "",
        "## Headline verdicts so far (all claims dated 1917)",
        "",
        "1. **The fish claim he is famous for is, literally read, refuted — and bounded, not buried.** *Scarus* → *Pomacanthus*",
        "   by (conformal) coaxial circles (1917, p. 749, Figs. 375–376): Möbius explains ~1% — a conformal map cannot deepen a",
        "   fish (depth/length 0.377 → 0.587); the generous oblique reading explains 32%, fails leave-one-landmark-out, and on",
        "   real outlines is worse than doing nothing (IoU 0.618 vs 0.662). See T1917-046.",
        "2. **The general transformation thesis survives only inside a horizon:** low-parameter warps reach the biological noise",
        "   floor within genera and within ~45–50 Myr, and fail between families (0.3% reach the floor). T1917-039/050.",
        "3. **Arthropod figure pairs (copepods, crabs, amphipods, 1917, pp. 743–746) fare better than the fish pair** on his own",
        "   drawings (e.g. copepods 84.2% vs 20.4% for a box stretch). T1917-042/043/044.",
        "4. **Most of the rest of the book has never been tested by us at all** — 21 UNTESTED rows, concentrated in exactly the",
        "   chapters people cite most after Ch. XVII: logarithmic spirals/forams (XI–XII), horns (XIII), eggs (XV), bone",
        "   trajectories (XVI), and the non-fish Ch. XVII skull series (crocodiles, horse, primates).",
        "",
        "## P0 — do these first (cheap, central, or both)",
        "",
        "| ID | Claim (1917) | Status | Effort | Why first |",
        "|---|---|---|---|---|",
    ]
    for c in claims:
        if c["pass2"]["priority"] == "P0":
            p2 = c["pass2"]
            lines.append(f"| {c['id']} | Ch. {c['chapter']}, pp. {c['pages']}, Figs. {c['figures']} — {c['claim'][:110]}… | {p2['status']} | {p2['effort_days'][0]}–{p2['effort_days'][1]}d ({p2['effort_class']}) | {p2['what_remains'][:130]}… |")
    lines += ["", "## Full inventory (Pass 1 citation + Pass 2 assessment)", ""]
    cur = None
    for c in claims:
        base = c["chapter"].split()[0].split("-")[0]  # XVII (fish I) -> XVII; IX-X -> IX
        if base != cur:
            cur = base
            heading = "Epilogue (1917)" if base == "Epilogue" else f"Chapter {cur} — {c['chapter_title']} (1917)"
            lines += [f"### {heading}", ""]
        p2 = c["pass2"]
        lines += [
            f"#### {c['id']} — Thompson (1917), Ch. {c['chapter']}, pp. {c['pages']}, Figs. {c['figures']}",
            "",
            f"**Claim (made 1917):** {c['claim']}",
            "",
        ]
        if c.get("quote"):
            lines += [f"> {c['quote']}", ""]
        lines += [
            f"**Pass 2 — status:** `{p2['status']}` · **effort to settle:** {p2['effort_days'][0]}–{p2['effort_days'][1]} person-days ({p2['effort_class']}) · **priority:** {p2['priority']}",
            "",
            f"**What remains / verdict:** {p2['what_remains']}",
            "",
            f"**Method:** {p2['method']} · **Data needed:** {p2['data_needed']} · **Blockers:** {p2['blockers']}",
            "",
        ]
        if p2.get("repo_evidence"):
            lines += ["**Repo evidence:** " + "; ".join(f"`{e}`" for e in p2["repo_evidence"]), ""]
    lines += [
        "## Pass-1 extensions (completing the citation pass itself)",
        "",
        "- **E1. Verbatim quote for every row** (OCR-located, hand-checked against the pinned scan leaf): 3–5 days. Until then,",
        "  rows without a `quote` are cited by chapter/page/figure, which is a complete bibliographic citation but not a quotation.",
        "- **E2. Sentence-level census:** splitting compound rows (esp. Chs. III, V, VIII, IX) into single-sentence claims would",
        "  take the registry from 59 to an estimated 200+ rows: 5–10 days, low marginal value unless a funder/journal requires it.",
        "- **E3. Figure-by-figure census of Ch. XVII** (Figs. 353–408 individually, incl. every Heilmann/Osborn intermediate): 2–4 days;",
        "  several are already rows T1917-051..058.",
        "",
        "## Rules for citing from this registry",
        "",
        "1. Always give the year: **Thompson (1917)**, 1st edition — never cite these rows as 'Thompson (1942)'.",
        "2. T1917-050 and all portfolio/blind-test results are **2026 extensions**, not 1917 claims; cite them as such.",
        "3. A `DONE_ELSEWHERE` verdict (e.g. honeycomb, square–cube) must name the modern source when used, not just 'Thompson was right'.",
        "4. `NOT_TESTABLE_AS_STATED` rows (001/002/017/059) may be quoted as programme, never scored as predictions.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="validate only, do not write the MD")
    ap.add_argument("--no-text", action="store_true", help="skip the Internet Archive full-text verification")
    args = ap.parse_args()
    reg = json.loads(REGISTRY.read_text())
    errs = validate(reg, check_text=not args.no_text)
    if errs:
        print("FAIL claim inventory:")
        for e in errs:
            print(f"  - {e}")
        return 1
    from collections import Counter

    st = Counter(c["pass2"]["status"] for c in reg["claims"])
    print(f"PASS {len(reg['claims'])} claims, all dated 1917: {dict(st)}")
    if not args.check:
        OUT_MD.write_text(render_md(reg))
        print(f"wrote {OUT_MD.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

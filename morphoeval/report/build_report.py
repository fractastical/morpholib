#!/usr/bin/env python3
"""Build the MorphoEval Reproducibility Report (PDF) from report.md and the audit files.

    python3 morphoeval/report/build_report.py                 # -> build/MorphoEval-Reproducibility-Report.pdf
    python3 morphoeval/report/build_report.py --out-dir /tmp/r --no-pdf   # markdown only

The prose lives in report.md; this script fills its placeholders from the audit YAMLs so the
tables cannot drift from the published scores:

    {{DATE}} {{COMMIT}}          today / `git rev-parse --short HEAD`
    {{RUBRIC_TABLE}}             from morphoeval/rubric.yaml
    {{SCORE_TABLE}}              current head of each script's chain, per dimension + overall
    {{HISTORY_TABLE}}            every audit of every script, in date order
    {{N_EVALUATED}} {{N_NANOPUBS}} {{N_HEADS}}   counts from kg/*/published.json

The pipeline figure is drawn by draw_pipeline.py into the build dir.  PDF via pandoc + xelatex.
Everything under build/ is git-ignored; the PDF is a deposit artefact (Zenodo), not a repo file.
"""
from __future__ import annotations

import argparse
import datetime as dt
import glob
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RUBRIC = ROOT / "morphoeval" / "rubric.yaml"


def sh(*cmd, **kw) -> str:
    return subprocess.run(cmd, check=True, capture_output=True, text=True, cwd=ROOT, **kw).stdout.strip()


def load_audits() -> list[dict]:
    out = []
    for f in sorted(glob.glob(str(ROOT / "morphoeval" / "audits" / "*-audit.yaml"))):
        a = yaml.safe_load(open(f))
        a["_file"] = Path(f).name
        out.append(a)
    return out


def rubric_table(rubric: dict) -> str:
    rows = ["| group | dimension | weight | question |", "|---|---|---:|---|"]
    for d in rubric["dimensions"]:
        q = " ".join(d["question"].split())
        rows.append(f"| {d['group']} | **{d['name']}** (`{d['id']}`) | {d['weight']} | {q} |")
    return "\n".join(rows)


def overall(scores: dict, weights: dict) -> float:
    return sum(scores[d] * w for d, w in weights.items()) / sum(weights.values())


def group_mean(scores: dict, rubric: dict, group: str) -> float:
    ds = [d for d in rubric["dimensions"] if d["group"] == group]
    return sum(scores[d["id"]] * d["weight"] for d in ds) / sum(d["weight"] for d in ds)


def collect(audits: list[dict], rubric: dict):
    """(latest-per-script, full history) with per-dimension scores and overall recomputed."""
    weights = {d["id"]: d["weight"] for d in rubric["dimensions"]}
    hist = []
    for a in audits:
        for s in a["simulations"]:
            if s.get("status") != "evaluated":
                continue
            sc = {d: v["value"] for d, v in s["scores"].items()}
            na = [d for d, v in s["scores"].items() if v.get("not_applicable")]
            hist.append(dict(audit=a["audit"]["id"], date=str(a["audit"]["date"]), commit=a["audit"]["checkpoint"]["commit"],
                             sim=s["id"], name=s["name"], script=s["script"], scores=sc, na=na,
                             overall=overall(sc, weights), faith=group_mean(sc, rubric, "faithfulness"),
                             repro=group_mean(sc, rubric, "reproducibility")))
    hist.sort(key=lambda r: (r["date"], r["audit"]))
    latest = {}
    for r in hist:
        latest[r["script"]] = r          # later audits overwrite; keyed by script so a rewrite is a new row
    return sorted(latest.values(), key=lambda r: -r["overall"]), hist


def score_table(latest, rubric) -> str:
    dims = [d["id"] for d in rubric["dimensions"]]
    short = {"model-fidelity": "fidelity", "specimen-match": "specimen", "parameter-expressivity": "params",
             "morphometric-validation": "morphom.", "source-traceability": "source", "code-traceability": "code",
             "repro-package": "repro"}
    head = "| script | audit (commit) | " + " | ".join(short[d] for d in dims) + " | faith. | repro. | **overall** |"
    # relative column widths in pandoc pipe tables follow the dash counts of the separator row
    rows = [head, "|" + "-" * 34 + "|" + "-" * 24 + "|" + "------:|" * len(dims) + "------:|" * 2 + "-------:|"]
    for r in latest:
        cells = [f"{r['scores'][d]}{'*' if d in r['na'] else ''}" for d in dims]
        aud = r["audit"].replace("-audit", "")
        rows.append(f"| `{r['script']}` | {aud} ({r['commit']}) | " + " | ".join(cells) +
                    f" | {r['faith']:.1f} | {r['repro']:.1f} | **{r['overall']:.1f}** |")
    rows.append("")
    rows.append("Table: Current scores (0–100). Weights: " + ", ".join(f"{short[d['id']]} {d['weight']}" for d in rubric["dimensions"]) +
                ". An asterisk marks a dimension scored neutral (50) as not applicable. "
                "`1900-planformDB_parser.py` is listed because it was audited; it is superseded by `2013-lobo-planform-morphology.py`.")
    return "\n".join(rows)


def history_table(hist) -> str:
    rows = ["| date | audit | commit | script | overall | note |",
            "|" + "-" * 12 + "|" + "-" * 28 + "|" + "-" * 9 + "|" + "-" * 40 + "|" + "-" * 8 + ":|" + "-" * 30 + "|"]
    prev = {}
    for r in hist:
        note = ""
        if r["script"] in prev:
            note = f"re-score; was {prev[r['script']]:.1f}"
        elif r["sim"] == "planform-morphology-animation" and r["script"].startswith("2013"):
            note = "rewrite of `1900-planformDB_parser.py` (14.7)"
        rows.append(f"| {r['date']} | {r['audit']} | {r['commit']} | `{r['script']}` | {r['overall']:.1f} | {note} |")
        prev[r["script"]] = r["overall"]
    return "\n".join(rows)


def nanopub_counts() -> tuple[int, int]:
    """(published to production, current heads).  Production URIs come from every kg/*/published.json
    (current `nanopub` plus its `history`); a URI is a head unless some signed nanopub in the repo
    `npx:supersedes` it (that is the only record of cross-audit supersession)."""
    import re
    prod = set()
    for f in glob.glob(str(ROOT / "morphoeval" / "kg" / "*" / "published.json")):
        for e in json.load(open(f)).values():
            for v in [e] + e.get("history", []):
                if v.get("published") == "prod" and v.get("nanopub"):
                    prod.add(v["nanopub"])
    superseded = set()
    pat = re.compile(r"npx:supersedes\s+<([^>]+)>")
    for f in glob.glob(str(ROOT / "morphoeval" / "kg" / "*" / "signed" / "*.trig")):
        superseded.update(pat.findall(Path(f).read_text()))
    heads_offline = prod - superseded
    # Authoritative: ask Nanopub Query for the head of every chain (the local record loses a link when a
    # superseded signed file is overwritten by a later re-issue).  Fall back to the offline count.
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location("bmk", ROOT / "morphoeval" / "build_morphoeval_kg.py")
        bmk = importlib.util.module_from_spec(spec); spec.loader.exec_module(bmk)
        heads = set()
        for u in sorted(prod):
            h = bmk.latest_version(u, timeout=20)
            if h is None:
                raise RuntimeError(f"no head for {u}")
            heads.add(h)
        heads &= prod
        if heads != heads_offline:
            print(f"  heads: Nanopub Query {len(heads)}, offline record {len(heads_offline)} (using the query)")
        return len(prod), len(heads)
    except Exception as ex:                                     # noqa: BLE001
        print(f"  head resolution unavailable ({ex}); using the offline record")
        return len(prod), len(heads_offline)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(HERE / "build"))
    ap.add_argument("--no-pdf", action="store_true")
    ap.add_argument("--counts", help="override 'N_NANOPUBS,N_HEADS' (the manifest-derived count is reported for comparison)")
    args = ap.parse_args(argv)
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    rubric = yaml.safe_load(open(RUBRIC))
    audits = load_audits()
    latest, hist = collect(audits, rubric)
    if args.counts:
        n_pub, n_heads = (int(x) for x in args.counts.split(","))
    else:
        n_pub, n_heads = nanopub_counts()
    commit = sh("git", "rev-parse", "--short", "HEAD")

    subprocess.run([sys.executable, str(HERE / "draw_pipeline.py"), "--out", str(out / "pipeline.png")], check=True,
                   env={**os.environ, "MPLBACKEND": "Agg"})

    text = (HERE / "report.md").read_text()
    fill = {
        "{{DATE}}": dt.date.today().isoformat(), "{{COMMIT}}": commit,
        "{{RUBRIC_TABLE}}": rubric_table(rubric), "{{SCORE_TABLE}}": score_table(latest, rubric),
        "{{HISTORY_TABLE}}": history_table(hist),
        "{{N_EVALUATED}}": str(len(hist)), "{{N_NANOPUBS}}": str(n_pub), "{{N_HEADS}}": str(n_heads),
    }
    for k, v in fill.items():
        text = text.replace(k, v)
    left = [l for l in text.splitlines() if "{{" in l]
    if left:
        sys.exit(f"unfilled placeholders: {left}")
    md = out / "MorphoEval-Reproducibility-Report.md"
    md.write_text(text)
    print(f"wrote {md}  ({len(hist)} audit rows, {len(latest)} current; {n_pub} nanopubs, {n_heads} heads)")

    if args.no_pdf:
        return
    if not shutil.which("pandoc"):
        sys.exit("pandoc not found (brew install pandoc; also needs xelatex)")
    pdf = out / "MorphoEval-Reproducibility-Report.pdf"
    cmd = ["pandoc", str(md), "-o", str(pdf), "--pdf-engine=xelatex", "--resource-path", str(out),
           "-V", "geometry:margin=2.2cm", "-V", "fontsize=10pt", "-V", "colorlinks=true", "-V", "linkcolor=NavyBlue",
           "-V", "urlcolor=NavyBlue", "-V", "mainfont=Arial Unicode MS", "-V", "monofont=Menlo", "-V", "monofontoptions=Scale=0.82",
           "--number-sections", "--toc", "--toc-depth=2",
           "-H", str(HERE / "header.tex")]
    subprocess.run(cmd, check=True, cwd=out)
    print(f"wrote {pdf}  ({pdf.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()

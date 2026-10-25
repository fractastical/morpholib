#!/usr/bin/env python3
"""
docs/zenodo_deposit_wolfram_planform.py —
Package and deposit "Planarian and Limb Regeneration as a Multiway Double-Pushout (DPO)
Hypergraph Rewrite System" on Zenodo (and generate an arXiv / bioRxiv submission bundle).

Usage:
  # 1. Build the Zenodo + arXiv/bioRxiv bundle locally (no token needed):
  ./venv/bin/python3 docs/zenodo_deposit_wolfram_planform.py --build-only

  # 2. Create a DRAFT on Zenodo Sandbox (needs ZENODO_SANDBOX_TOKEN):
  ZENODO_SANDBOX_TOKEN=... ./venv/bin/python3 docs/zenodo_deposit_wolfram_planform.py --sandbox

  # 3. Create a DRAFT on production Zenodo (needs ZENODO_TOKEN):
  ZENODO_TOKEN=... ./venv/bin/python3 docs/zenodo_deposit_wolfram_planform.py

  # 4. Publish the draft on production Zenodo (mints a permanent DOI):
  ZENODO_TOKEN=... ./venv/bin/python3 docs/zenodo_deposit_wolfram_planform.py --publish --deposition <id>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
BUILD_DIR = HERE / "build_wolfram_planform_zenodo"

PDF_SRC = HERE / "wolfram_planform_multiway_manuscript.pdf"
TEX_SRC = HERE / "wolfram_planform_multiway_manuscript.tex"
FIG1_SRC = HERE / "media" / "wolfram_planform_multiway_comparison.png"
FIG2_SRC = HERE / "media" / "wolfram_planform_isomers_atlas.png"
MODEL_SRC = ROOT / "models" / "2026-wolfram-planform-multiway-rewriting.py"
LEAN_SRC = ROOT / "morphoeval" / "lean" / "loops" / "Loop_wolfram_planform_multiway.lean"
REF_JSON_SRC = ROOT / "datasets" / "planform_multiway" / "planform_multiway_reference.json"
BIB_SRC = ROOT / "datasets" / "planform_multiway" / "paper_citations.bib"
README_SRC = ROOT / "datasets" / "planform_multiway" / "README.md"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True, cwd=ROOT
    ).stdout.strip()


def build_bundle() -> tuple[Path, Path, Path, dict]:
    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    commit_full = git_head()
    commit_short = commit_full[:8]

    pdf_out = BUILD_DIR / "Dietz-2026-Planarian-Limb-Multiway-DPO-Hypergraph-Rewriting.pdf"
    shutil.copy2(PDF_SRC, pdf_out)

    # 1. Reproducibility code + Lean 4 + SHA-256 fixture archive for Zenodo
    repro_zip = BUILD_DIR / f"wolfram-planform-multiway-repro-{commit_short}.zip"
    bundle_files = [
        (PDF_SRC, "docs/wolfram_planform_multiway_manuscript.pdf"),
        (TEX_SRC, "docs/wolfram_planform_multiway_manuscript.tex"),
        (FIG1_SRC, "docs/media/wolfram_planform_multiway_comparison.png"),
        (FIG2_SRC, "docs/media/wolfram_planform_isomers_atlas.png"),
        (MODEL_SRC, "models/2026-wolfram-planform-multiway-rewriting.py"),
        (LEAN_SRC, "morphoeval/lean/loops/Loop_wolfram_planform_multiway.lean"),
        (REF_JSON_SRC, "datasets/planform_multiway/planform_multiway_reference.json"),
        (BIB_SRC, "datasets/planform_multiway/paper_citations.bib"),
        (README_SRC, "datasets/planform_multiway/README.md"),
    ]
    with zipfile.ZipFile(repro_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for src, arcname in bundle_files:
            zf.write(src, arcname=f"wolfram-planform-multiway-{commit_short}/{arcname}")

    # 2. Flat TeX + figures archive ready for direct arXiv / bioRxiv upload
    arxiv_zip = BUILD_DIR / "arxiv_submission_bundle.zip"
    with zipfile.ZipFile(arxiv_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(TEX_SRC, arcname="wolfram_planform_multiway_manuscript.tex")
        zf.write(FIG1_SRC, arcname="media/wolfram_planform_multiway_comparison.png")
        zf.write(FIG2_SRC, arcname="media/wolfram_planform_isomers_atlas.png")

    metadata = {
        "title": (
            "Planarian and Limb Regeneration as a Multiway Double-Pushout (DPO) Hypergraph Rewrite System: "
            "A Three-Rule Generative Basis, Seven Hidden Topological Isomers, and 44,534x McKay IR Quotient "
            "Compression Across PlanformDB 2.5.0 and LimbformDB 2.1.0"
        ),
        "upload_type": "publication",
        "publication_type": "preprint",
        "description": (
            "<p>When a planarian flatworm or an amphibian/arthropod limb is surgically cut or bioelectrically "
            "perturbed, it can regenerate a wide variety of abnormal body plans: two heads, three heads, lateral "
            "side-heads sprouting from the belly, or a second head growing directly out of the tip of an existing "
            "head. In two foundational papers, Lobo, Levin, and colleagues introduced <strong>PlanformDB 2.5.0</strong> "
            "(Lobo, Malone &amp; Levin 2013, encoding 369 planarian morphologies across 1,181 regions) and "
            "<strong>LimbformDB 2.1.0</strong> (Lobo et al. 2014, encoding 555 salamander, frog, crustacean, insect, "
            "and arachnid limbs across 3,146 joints) to store these experimental outcomes as formal graphs. However, "
            "both databases remained static descriptive catalogs, leaving three foundational problems open: "
            "(i) <em>The Graph Equivalence Problem</em>&mdash;how many distinct body-plan wiring diagrams actually "
            "exist in the databases up to exact graph isomorphism, and when does simple part-counting fail? "
            "(ii) <em>The Generative Grammar Problem</em>&mdash;what is the minimal set of formal graph-rewrite rules "
            "needed to grow 100% of the observed morphologies from cut tissue fragments? and "
            "(iii) <em>The Combinatorial Search Problem</em>&mdash;how can an automated theory-discovery engine search "
            "millions of multi-step growth trajectories without drowning in isomorphic node-numbering duplicates?</p>"
            "<p>Here we solve all three problems by formulating planarian and limb regeneration as a "
            "<strong>Multiway Double-Pushout (DPO) Hypergraph Rewrite System</strong>, coupling the standalone C++20 "
            "<code>hg_evolve</code> engine from the Wolfram Physics Project "
            "(<code>WolframInstitute/HypergraphRewritingEngine</code>, <code>v1.0.0-rc2</code>; Assar 2026) directly to "
            "<code>PlanformDB 2.5.0</code> and <code>LimbformDB 2.1.0</code> over Wolfram Exchange Format (WXF), and "
            "verifying the governing algebraic laws in the <strong>Lean 4</strong> theorem prover (<code>0 sorry</code>):</p>"
            "<ol>"
            "<li><strong>Why Counting Body Parts Fails (38 Planform Classes &amp; 7 Hidden Topological Isomers):</strong> "
            "Exact McKay individualization&ndash;refinement (IR) canonicalization compresses the 369 PlanformDB rows into "
            "<strong>38 exact region-link isomorphism classes</strong> (9.71x redundancy; 126 classes with organs) and the "
            "555 LimbformDB rows into <strong>51 segment-tree classes</strong> (10.88x redundancy; 92 with digits), "
            "uncovering <strong>7 hidden topological isomers</strong> (31 part-count buckets &rarr; 38 IR classes) that "
            "share identical (Head, Trunk, Tail, Links) counts but differ in star vs. linear chain wiring.</li>"
            "<li><strong>All 369 Planarians Are Generated by Just Three Primitive Rules (B_3 = {L_1, D_1, D_2}):</strong> "
            "All 8 domain-specific regeneration rules (R1&ndash;R6 terminal sprouting + R7&ndash;R8 intercalary bridging) "
            "reduce algebraically to <strong>one universal budding rule</strong> <code>L_1</code> (shared identically with "
            "LimbformDB) and <strong>two in-place bioelectric fate switches</strong> <code>D_1</code> (Blastema &rarr; Head) "
            "and <code>D_2</code> (Blastema &rarr; Tail). Stripping region colors projects all 369 worms onto just "
            "<strong>12 unlabelled skeleton topologies</strong>, and intercalary heads/tails emerge automatically whenever "
            "<code>L_1</code> branches twice before terminal differentiation lock (<code>D_1, D_2</code>).</li>"
            "<li><strong>44,534x Quotient Exploration Speedup on LimbformDB (OEIS A000055 in 85 ms):</strong> "
            "Using <code>ExploreFromCanonicalStatesOnly</code>, <code>hg_evolve</code> expands only the <strong>987 unique "
            "unlabelled trees</strong> (OEIS A000055) up to V = 12 joints while analytically reconstructing all 43,954,714 "
            "raw factorial trajectories in <strong>85 ms</strong> (44,533.65x compression), matching <strong>35/35 (100%)</strong> "
            "of LimbformDB classes with &le;12 joints (532/555 = 95.86% of the database).</li>"
            "<li><strong>Bioelectric RuleWeights Gating &amp; Lean 4 Verification:</strong> Coupling posterior "
            "<code>RuleWeights</code> to electrotonic gap-junction decay reproduces Oviedo et al. (2010) 1-octanol Series S1 "
            "(r = 0.9823) and held-out innexin-RNAi Series S2 (r = 0.9088), with all 4 structural theorems kernel-checked "
            "in Lean 4 (<code>0 sorry</code>).</li>"
            "</ol>"
            f"<p><strong>Reproducibility Archive:</strong> <code>{repro_zip.name}</code> (commit <code>{commit_short}</code> "
            f"of <code>https://github.com/fractastical/morpholib</code>, SHA-256 <code>{sha256_file(repro_zip)}</code>).</p>"
        ),
        "creators": [
            {
                "name": "Dietz, Joel",
                "orcid": "0009-0007-4948-9192",
            }
        ],
        "access_right": "open",
        "license": "MIT",
        "keywords": [
            "planarian regeneration",
            "PlanformDB",
            "LimbformDB",
            "hypergraph rewriting",
            "double-pushout graph grammar",
            "McKay canonicalization",
            "graph isomorphism",
            "topological isomers",
            "bioelectricity",
            "Wolfram Physics Project",
            "Lean 4",
            "morphospace",
        ],
        "related_identifiers": [
            {
                "identifier": f"https://github.com/fractastical/morpholib/tree/{commit_full}",
                "relation": "isSupplementTo",
                "resource_type": "software",
                "scheme": "url",
            },
            {
                "identifier": "https://github.com/WolframInstitute/HypergraphRewritingEngine",
                "relation": "requires",
                "resource_type": "software",
                "scheme": "url",
            },
            {
                "identifier": "10.1093/bioinformatics/btt088",
                "relation": "isDerivedFrom",
                "resource_type": "publication-article",
                "scheme": "doi",
            },
            {
                "identifier": "10.1093/bioinformatics/btu582",
                "relation": "isDerivedFrom",
                "resource_type": "publication-article",
                "scheme": "doi",
            },
            {
                "identifier": "10.5281/zenodo.20567719",
                "relation": "continues",
                "resource_type": "software",
                "scheme": "doi",
            },
            {
                "identifier": "10.5281/zenodo.20622878",
                "relation": "continues",
                "resource_type": "publication-preprint",
                "scheme": "doi",
            },
        ],
        "version": "1.0.0",
        "language": "eng",
    }

    meta_path = BUILD_DIR / "zenodo-metadata.json"
    meta_path.write_text(json.dumps({"metadata": metadata}, indent=2) + "\n")
    print(f"Built Zenodo & preprint artefacts in {BUILD_DIR}:")
    print(f"  1. {pdf_out.name} ({pdf_out.stat().st_size // 1024} KB, sha256={sha256_file(pdf_out)[:16]}...)")
    print(f"  2. {repro_zip.name} ({repro_zip.stat().st_size // 1024} KB, sha256={sha256_file(repro_zip)[:16]}...)")
    print(f"  3. {arxiv_zip.name} ({arxiv_zip.stat().st_size // 1024} KB, ready for arXiv/bioRxiv TeX upload)")
    print(f"  4. {meta_path.name}")
    return pdf_out, repro_zip, arxiv_zip, metadata


def api(base: str, token: str):
    import requests
    s = requests.Session()
    s.params = {"access_token": token}

    def call(method: str, path: str, **kw):
        url = path if path.startswith("http") else f"{base}/api{path}"
        r = s.request(method, url, **kw)
        if r.status_code >= 400:
            sys.exit(f"Zenodo {method} {path}: {r.status_code} {r.text[:800]}")
        return r.json() if r.text else {}

    return call


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--build-only", action="store_true", help="Build PDF, reproducibility zip, arXiv zip, and metadata JSON locally")
    ap.add_argument("--sandbox", action="store_true", help="Deposit to sandbox.zenodo.org using ZENODO_SANDBOX_TOKEN")
    ap.add_argument("--deposition", type=int, help="Existing draft deposition ID to update/publish")
    ap.add_argument("--new-version-of", type=int, metavar="RECORD_ID", help="Create a new version of an existing Zenodo record")
    ap.add_argument("--publish", action="store_true", help="Publish the Zenodo deposition (mints permanent DOI)")
    args = ap.parse_args(argv)

    pdf_out, repro_zip, _arxiv_zip, meta = build_bundle()
    if args.build_only:
        return

    base = "https://sandbox.zenodo.org" if args.sandbox else "https://zenodo.org"
    env_key = "ZENODO_SANDBOX_TOKEN" if args.sandbox else "ZENODO_TOKEN"
    token = os.environ.get(env_key)
    if not token:
        sys.exit(f"Missing {env_key} in environment. Set {env_key}=<your_personal_access_token> or run with --build-only.")

    call = api(base, token)
    if args.deposition:
        dep = call("GET", f"/deposit/depositions/{args.deposition}")
    elif args.new_version_of:
        dep = call("POST", f"/deposit/depositions/{args.new_version_of}/actions/newversion")
        dep = call("GET", dep["links"]["latest_draft"])
        for f in dep.get("files", []):
            call("DELETE", f"/deposit/depositions/{dep['id']}/files/{f['id']}")
    else:
        dep = call("POST", "/deposit/depositions", json={})

    dep_id = dep["id"]
    print(f"Draft deposition {dep_id}: {base}/deposit/{dep_id}")

    bucket = dep["links"]["bucket"]
    have = {f["filename"] for f in dep.get("files", [])}
    for p in (pdf_out, repro_zip):
        if p.name in have:
            print(f"  {p.name} already uploaded")
            continue
        with open(p, "rb") as fh:
            call("PUT", f"{bucket}/{p.name}", data=fh)
        print(f"  uploaded {p.name}")

    call("PUT", f"/deposit/depositions/{dep_id}", json={"metadata": meta})
    print("  metadata set")

    if not args.publish:
        print(f"Draft ready — review at {base}/deposit/{dep_id}, then run with --publish --deposition {dep_id}")
        return

    rec = call("POST", f"/deposit/depositions/{dep_id}/actions/publish")
    print(f"PUBLISHED! DOI: {rec.get('doi')} | Concept DOI: {rec.get('conceptdoi')} | URL: {rec.get('links', {}).get('record_html')}")


if __name__ == "__main__":
    main()

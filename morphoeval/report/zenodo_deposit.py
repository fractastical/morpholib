#!/usr/bin/env python3
"""Deposit the MorphoEval report + a source archive on Zenodo, the way BioEval and NeuroEval were.

    # 1. build the artefacts only (no network, no token): PDF + source zip + the metadata that would be sent
    python3 morphoeval/report/zenodo_deposit.py --build-only

    # 2. create a DRAFT on the Zenodo sandbox and upload (needs ZENODO_SANDBOX_TOKEN); nothing is published
    ZENODO_SANDBOX_TOKEN=... python3 morphoeval/report/zenodo_deposit.py --sandbox

    # 3. the same on production Zenodo (needs ZENODO_TOKEN); still a draft, inspect it in the browser
    ZENODO_TOKEN=... python3 morphoeval/report/zenodo_deposit.py

    # 4. publish the draft (mints the DOI; irreversible) — asks for a typed 'yes'
    ZENODO_TOKEN=... python3 morphoeval/report/zenodo_deposit.py --publish --deposition <id>

    # new version of an existing record (keeps the concept DOI)
    ZENODO_TOKEN=... python3 morphoeval/report/zenodo_deposit.py --new-version-of <record id>

Artefacts (written to morphoeval/report/build/, git-ignored):
    MorphoEval-Reproducibility-Report.pdf       from build_report.py (pandoc + xelatex)
    morpholib-source-<commit>.zip               `git archive HEAD` — tracked files only, so no caches,
                                                no rendered outputs, no local databases
    zenodo-metadata.json                        .zenodo.json with version/description filled for this revision
Metadata comes from .zenodo.json at the repository root (also read by the GitHub–Zenodo integration).
The working tree must be clean for the tracked files that go into the archive (checked).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BUILD = HERE / "build"
PDF = BUILD / "MorphoEval-Reproducibility-Report.pdf"
ARCHIVE_EXCLUDE = ["wave-vector-analysis", "presentation-assets", "archive", ".mplconfig", ".DS_Store",
                   "parser_output.png", "batch_parse.log", "COORDINATE_SYSTEM_WARNING.md",
                   "XY_COORDINATES_DOCUMENTATION.md"]   # plus every root-level *.gif (full-size historical outputs);
                                                        # the README's downsized animations in docs/media/ stay in


def sh(*cmd) -> str:
    return subprocess.run(cmd, check=True, capture_output=True, text=True, cwd=ROOT).stdout.strip()


def build_artefacts(counts: str | None) -> tuple[Path, Path, dict]:
    commit = sh("git", "rev-parse", "--short", "HEAD")
    dirty = sh("git", "status", "--porcelain", "--untracked-files=no")
    # unrelated local modifications (wave-vector outputs, .DS_Store) are tolerated; anything under the
    # audited paths must be committed so the archive matches the report's commit
    guarded = ("morphoeval/", "thompson/", "datasets/", "README.md", "requirements", "LICENSE", ".zenodo.json", "CITATION.cff")
    bad = [l for l in dirty.splitlines() if l[3:].startswith(guarded) or l[3:].endswith(".py") and "/" not in l[3:]]
    if bad:
        sys.exit("commit these before depositing (the archive is taken from HEAD):\n" + "\n".join(bad))
    BUILD.mkdir(exist_ok=True)
    cmd = [sys.executable, str(HERE / "build_report.py")] + (["--counts", counts] if counts else [])
    subprocess.run(cmd, check=True, cwd=ROOT, env={**os.environ, "MPLBACKEND": "Agg"})
    zip_path = BUILD / f"morpholib-source-{commit}.zip"
    # the reproductions, MorphoEval and the datasets; not the Ca2+ wave-imaging pipeline and its results,
    # presentation assets, historical full-size GIFs or stray logs that happen to be tracked
    root_gifs = [f for f in sh("git", "ls-tree", "--name-only", "HEAD").splitlines() if f.endswith(".gif")]
    subprocess.run(["git", "archive", "--format=zip", f"--prefix=morpholib-{commit}/", "-o", str(zip_path), "HEAD", "--", "."]
                   + [f":(exclude){p}" for p in ARCHIVE_EXCLUDE + root_gifs], check=True, cwd=ROOT)
    meta = json.load(open(ROOT / ".zenodo.json"))
    meta["description"] += (f"<p>Source archive: <code>{zip_path.name}</code>, repository revision <code>{commit}</code> "
                            f"of https://github.com/fractastical/morpholib. Rendered outputs are not included; every script "
                            f"regenerates them, and external inputs are fetched and SHA-256-verified by the scripts.</p>")
    meta.setdefault("related_identifiers", []).append(
        {"identifier": f"https://github.com/fractastical/morpholib/tree/{sh('git', 'rev-parse', 'HEAD')}",
         "relation": "isIdenticalTo", "resource_type": "software", "scheme": "url"})
    (BUILD / "zenodo-metadata.json").write_text(json.dumps({"metadata": meta}, indent=2))
    print(f"artefacts: {PDF.name} ({PDF.stat().st_size // 1024} KB), {zip_path.name} ({zip_path.stat().st_size // 1024} KB), zenodo-metadata.json")
    return PDF, zip_path, meta


def api(base: str, token: str):
    import requests
    s = requests.Session(); s.params = {"access_token": token}
    def call(method, path, **kw):
        r = s.request(method, path if path.startswith("http") else f"{base}/api{path}", **kw)
        if r.status_code >= 400:
            sys.exit(f"Zenodo {method} {path}: {r.status_code} {r.text[:800]}")
        return r.json() if r.text else {}
    return call


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--build-only", action="store_true")
    ap.add_argument("--sandbox", action="store_true", help="use sandbox.zenodo.org and ZENODO_SANDBOX_TOKEN")
    ap.add_argument("--deposition", type=int, help="existing draft id to upload into / publish")
    ap.add_argument("--new-version-of", type=int, metavar="RECORD_ID", help="create a new version of a published record")
    ap.add_argument("--publish", action="store_true", help="publish the draft (mints the DOI; irreversible)")
    ap.add_argument("--counts", help="pass through to build_report.py (skips the Nanopub Query head resolution)")
    args = ap.parse_args(argv)

    pdf, zip_path, meta = build_artefacts(args.counts)
    if args.build_only:
        return

    base = "https://sandbox.zenodo.org" if args.sandbox else "https://zenodo.org"
    token = os.environ.get("ZENODO_SANDBOX_TOKEN" if args.sandbox else "ZENODO_TOKEN")
    if not token:
        sys.exit(f"set {'ZENODO_SANDBOX_TOKEN' if args.sandbox else 'ZENODO_TOKEN'} (a personal access token with deposit:write"
                 f"{' and deposit:actions' if args.publish else ''}) or use --build-only")
    call = api(base, token)

    if args.deposition:
        dep = call("GET", f"/deposit/depositions/{args.deposition}")
    elif args.new_version_of:
        dep = call("POST", f"/deposit/depositions/{args.new_version_of}/actions/newversion")
        dep = call("GET", dep["links"]["latest_draft"])
        for f in dep.get("files", []):                      # drop the previous version's files
            call("DELETE", f"/deposit/depositions/{dep['id']}/files/{f['id']}")
    else:
        dep = call("POST", "/deposit/depositions", json={})
    dep_id = dep["id"]
    print(f"draft deposition {dep_id}: {base}/deposit/{dep_id}")

    bucket = dep["links"]["bucket"]
    have = {f["filename"] for f in dep.get("files", [])}
    for p in (pdf, zip_path):
        if p.name in have:
            print(f"  {p.name} already uploaded"); continue
        with open(p, "rb") as fh:
            call("PUT", f"{bucket}/{p.name}", data=fh)
        print(f"  uploaded {p.name}")
    call("PUT", f"/deposit/depositions/{dep_id}", json={"metadata": meta})
    print("  metadata set")

    if not args.publish:
        print(f"draft ready — review it at {base}/deposit/{dep_id}, then run again with --publish --deposition {dep_id}")
        return
    if input(f"Publish deposition {dep_id} on {base} (mints a DOI; cannot be undone)? type 'yes': ").strip() != "yes":
        sys.exit("not published")
    rec = call("POST", f"/deposit/depositions/{dep_id}/actions/publish")
    print(f"published: DOI {rec.get('doi')}  concept DOI {rec.get('conceptdoi')}  {rec['links'].get('record_html')}")


if __name__ == "__main__":
    main()

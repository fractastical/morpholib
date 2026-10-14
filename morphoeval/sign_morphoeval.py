#!/usr/bin/env python3
"""
sign_morphoeval.py — sign (and optionally publish) the MorphoEval nanopubs.

    python3 morphoeval/sign_morphoeval.py morphoeval/kg/<audit-id>              # sign only
    python3 morphoeval/sign_morphoeval.py morphoeval/kg/<audit-id> --publish test
    python3 morphoeval/sign_morphoeval.py morphoeval/kg/<audit-id> --publish prod   # after review!

Signs every unsigned *.trig in the directory with the RSA key from ~/.nanopub/profile.yml,
writes the signed nanopubs to <dir>/signed/ and records their trusty URIs in
<dir>/published.json ({name: {nanopub, published: null|"test"|"prod", date}}).

Signing is deterministic for a given input (dct:created is fixed in the pubinfo), so
sign-only, test and prod runs all yield the same https://w3id.org/np/RA... URIs; what is
checked on the test registry is byte-for-byte what goes to production.

Why not `nanopub` CLI / `np_utils.py`: nanopub-py 2.1.0 cannot read ~/.nanopub/profile.yml
on Python 3.14 (yatiml Union error), so the Profile is built here directly; and its
`has_valid_trusty` verifier is broken on this stack (it rejects nanopub-java-signed,
published nanopubs too), so verification is done independently with the Rust/WASM
`@nanopub/sign` checker: `node morphoeval/check_signed.mjs <dir>/signed/*.trig`.
Published nanopubs are immutable: a re-score is a new nanopub, not an edit.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import warnings
from datetime import date
from pathlib import Path

warnings.filterwarnings("ignore")
from rdflib import Dataset, Namespace             # noqa: E402
from nanopub import Nanopub, NanopubConf, Profile  # noqa: E402

NPX = Namespace("http://purl.org/nanopub/x/")


def load_profile() -> Profile:
    prof = Path.home() / ".nanopub" / "profile.yml"
    if not prof.exists():
        sys.exit("~/.nanopub/profile.yml not found; run `python3 -m nanopub setup --newkeys`")
    kv = dict(re.match(r"(\w+):\s*(.*)", l).groups()
              for l in prof.read_text().splitlines() if re.match(r"\w+:", l))
    return Profile(orcid_id=kv["orcid_id"], name=kv["name"],
                   private_key=Path(kv["private_key"]), public_key=Path(kv["public_key"]))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("kg_dir", help="morphoeval/kg/<audit-id> (unsigned *.trig from build_morphoeval_kg.py)")
    ap.add_argument("--publish", choices=["test", "prod"], default=None)
    ap.add_argument("--only", nargs="*", default=None, help="restrict to these file names")
    a = ap.parse_args()

    kg = Path(a.kg_dir)
    out = kg / "signed"; out.mkdir(exist_ok=True)
    manifest_path = kg / "published.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}

    if a.publish == "prod":
        not_tested = [n for n, m in manifest.items() if m.get("published") != "test"]
        if not_tested or not manifest:
            sys.exit("refusing --publish prod: publish to the test registry first "
                     f"({len(not_tested) or 'all'} nanopubs not yet test-published)")
        if input("Publish to the PRODUCTION nanopub network (immutable)? type 'yes': ").strip() != "yes":
            sys.exit("aborted")

    conf = NanopubConf(profile=load_profile(), use_test_server=(a.publish == "test"))
    files = sorted(f for f in kg.glob("*.trig") if not a.only or f.name in a.only)
    for f in files:
        ds = Dataset(); ds.parse(f, format="trig")
        np_ = Nanopub(rdf=ds, conf=conf)
        np_.sign()
        uri = np_.source_uri
        assert uri and uri.startswith("https://w3id.org/np/RA"), uri
        entry = manifest.get(f.name, {})
        prev = entry.get("nanopub")
        if prev and prev != uri:
            supersedes = {str(o) for o in np_.pubinfo.objects(None, NPX.supersedes)}
            if prev not in supersedes:
                sys.exit(f"{f.name}: content changed since it was recorded as {prev} (now {uri}) but does not "
                         "npx:supersedes it; published nanopubs are immutable — rebuild with --supersede")
            history = entry.get("history", []) + [{"nanopub": prev, "published": entry.get("published"), "date": entry.get("date")}]
            entry = {"history": history}            # new version starts unpublished
        if a.publish:
            np_.publish()
        (out / f.name).write_text(np_.rdf.serialize(format="trig"))
        status = entry.get("published") if entry.get("nanopub") == uri else None
        rank = {None: 0, "test": 1, "prod": 2}
        if rank.get(a.publish, 0) >= rank.get(status, 0):
            status = a.publish
        new = {"nanopub": uri, "published": status, "date": date.today().isoformat()}
        if entry.get("history"):
            new["history"] = entry["history"]
        manifest[f.name] = new
        print(f"{('published[' + a.publish + ']') if a.publish else 'signed'}  {f.name} -> {uri}")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"{len(files)} nanopubs -> {out}  (manifest: {manifest_path})")
    print("verify independently: node morphoeval/check_signed.mjs", str(out / "*.trig"))


if __name__ == "__main__":
    main()

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
Its publish() is not used either: it re-serialises through rdflib, which turns the
plain-decimal xsd:double literals the hash was computed over into exponent form; the
signed text is fixed up here and posted verbatim instead.
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
import requests                                    # noqa: E402
from rdflib import Dataset, Namespace             # noqa: E402
from nanopub import Nanopub, NanopubConf, Profile  # noqa: E402
from nanopub.definitions import NANOPUB_REGISTRY_URLS, TEST_NANOPUB_REGISTRY_URL  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_morphoeval_kg import plain_doubles      # noqa: E402

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

    files = sorted(f for f in kg.glob("*.trig") if not a.only or f.name in a.only)
    if a.publish == "prod":
        # every file about to be posted must already be on the test registry (or on production from
        # an earlier run); manifest entries whose file is gone (superseded elsewhere) do not count
        not_tested = [f.name for f in files if manifest.get(f.name, {}).get("published") not in ("test", "prod")]
        if not_tested or not manifest:
            sys.exit("refusing --publish prod: publish to the test registry first "
                     f"({len(not_tested) or 'all'} nanopubs not yet test-published)")
        if input("Publish to the PRODUCTION nanopub network (immutable)? type 'yes': ").strip() != "yes":
            sys.exit("aborted")

    # Files built with `--ontology-version 2.0 --allow-staged-templates` link placeholder template
    # IRIs (https://w3id.org/np/STAGED…) for an offline shape check only (morpholib #23).
    if staged := [f.name for f in files if "https://w3id.org/np/STAGED" in f.read_text()]:
        sys.exit(f"refusing to sign {len(staged)} file(s) that link a STAGED (unpublished) Project Delphi 2.0 "
                 f"template placeholder: {', '.join(staged[:3])}{'…' if len(staged) > 3 else ''}; rebuild with "
                 f"the published template IRIs (--template-iri / vendor/templates/delphi-2.0-template-iris.json)")

    conf = NanopubConf(profile=load_profile(), use_test_server=(a.publish == "test"))
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
            if prev not in supersedes and entry.get("published") in ("test", "prod"):
                sys.exit(f"{f.name}: content changed since it was recorded as {prev} (now {uri}) but does not "
                         "npx:supersedes it; published nanopubs are immutable — rebuild with --supersede")
            # a signed-but-never-published version is not on the network: it is simply replaced
            # (kept in history) and must not be superseded, which would leave a dangling link
            history = entry.get("history", []) + [{"nanopub": prev, "published": entry.get("published"), "date": entry.get("date")}]
            entry = {"history": history}            # new version starts unpublished
        # The lexical form of every literal is part of the trusty hash, and the hash was computed
        # over the forms parsed from `f` ("0.76"^^xsd:double).  rdflib's TriG serializer rewrites
        # doubles in exponent form (7.6e-01), which is a different lexical form: a file or POST
        # body written that way fails independent verification.  So the signed text is fixed up
        # and that exact text is both stored and posted (same endpoint/headers as nanopub-py).
        text = plain_doubles(np_.rdf.serialize(format="trig"))
        if a.publish:
            # The registry URL must be chosen here explicitly: nanopub-py switches to the test
            # registry only on its own copy of the conf inside Nanopub.__init__, so reading
            # `conf.use_server` from the outer object silently posts to production.
            server = TEST_NANOPUB_REGISTRY_URL if a.publish == "test" else NANOPUB_REGISTRY_URLS[0]
            r = requests.post(server, headers={"Content-Type": "application/trig"},
                              data=text.encode("utf-8"), timeout=120)
            r.raise_for_status()
            check = requests.get(server + uri.rsplit("/", 1)[1], headers={"Accept": "application/trig"}, timeout=60)
            if check.status_code != 200 or uri.rsplit("/", 1)[1] not in check.text:
                sys.exit(f"{f.name}: posted to {server} but it does not serve {uri} back (HTTP {check.status_code})")
        (out / f.name).write_text(text)
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

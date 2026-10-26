#!/usr/bin/env python3
"""
Publish EACH Thompson (1917) claim + its literature verdict as a nanopub.

One nanopublication per claim (59 total), stating side by side:
  * the claim as made by Thompson in 1917 (1st edition, chapter / printed
    pages / figures, verbatim quote where pinned), and
  * what the literature did with it afterwards: relation
    (REFUTED / PARTIALLY_REFUTED / SUPPORTED / MIXED / SUPERSEDED /
    QUALIFIED / NO_DIRECT_TEST / NOT_TESTABLE), who and WHEN, the
    counterclaim itself, and the quantitative margin where the literature
    gives one (e.g. honeycomb: Fejes Toth 1964, better base saves <0.35%;
    Kelvin foam: Weaire & Phelan 1994, ~0.3% less area).

Discipline enforced in every assertion:
  * Thompson's side is always dated 1917; the literature side always
    carries its own who/when -- the two are never merged.
  * NO_DIRECT_TEST is published as an explicit finding ("no direct test
    found in this 2026-10-26 literature review"), not left blank.
  * The repo's own 2026 validation status is recorded in a separate
    predicate (repo2026StatusSeparate) and must not be cited as
    literature. For T1917-046 the quantitative refutation is repo 2026;
    the literature, as of this review, contains no quantitative test.

Sources (SHA-256 pinned in each assertion's provenance artifacts):
  thompson/thompson_1917_claims.json        (claim registry, 59 x 1917)
  thompson/thompson_1917_counterclaims.json (literature layer)
  docs/THOMPSON_1917_COUNTERCLAIMS.md

Usage (repo venv):
  venv/bin/python nanopubs/thompson_1917_counterclaims_to_nanopubs.py
  venv/bin/python nanopubs/thompson_1917_counterclaims_to_nanopubs.py --sign
  venv/bin/python nanopubs/thompson_1917_counterclaims_to_nanopubs.py --publish test
  venv/bin/python nanopubs/thompson_1917_counterclaims_to_nanopubs.py --publish prod
"""
from __future__ import annotations

import argparse
import datetime as dt
import importlib.util
import json
import sys
from pathlib import Path

from rdflib import Literal, URIRef
from rdflib.namespace import DCTERMS, PROV, RDF, RDFS, XSD

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "nanopubs" / "thompson_predictions"
PROD_REGISTRY = "https://registry.knowledgepixels.com/np/"
GH = "https://github.com/fractastical/morpholib/blob/main/"
REVIEW_DATE = "2026-10-26"

_spec = importlib.util.spec_from_file_location(
    "thompson_process_np", Path(__file__).with_name("thompson_1917_process_to_nanopub.py"))
_base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_base)

MLT, NP = _base.MLT, _base.NP
THOMPSON_1917 = _base.THOMPSON_1917
new_dataset, read_profile, git_commit, sha256 = (
    _base.new_dataset, _base.read_profile, _base.git_commit, _base.sha256)

RELATIONS = {"REFUTED", "PARTIALLY_REFUTED", "SUPPORTED", "MIXED",
             "SUPERSEDED", "QUALIFIED", "NO_DIRECT_TEST", "NOT_TESTABLE"}
CLAIMS_REL = REPO_ROOT / "thompson" / "thompson_1917_claims.json"
COUNTER_REL = REPO_ROOT / "thompson" / "thompson_1917_counterclaims.json"


def artifact(g, rel: str, label: str):
    node = URIRef(GH + rel)
    p = REPO_ROOT / rel
    g.add((node, RDF.type, PROV.Entity))
    g.add((node, RDFS.label, Literal(label)))
    if p.exists():
        g.add((node, MLT.sha256, Literal(sha256(p))))
        g.add((node, MLT.bytes, Literal(len(p.read_bytes()), datatype=XSD.integer)))
    return node


def claim_nanopub(row: dict, lit: dict, commit: str, today: str, orcid, name):
    cid = row["id"]
    np_id = f"claim1917-{cid.lower()}"
    ds, base, A, P, I = new_dataset(np_id)
    subj = URIRef(MLT + f"claim/{cid.lower()}")
    verdict = URIRef(MLT + f"literature-verdict/{cid.lower()}")

    A.add((subj, RDF.type, MLT.HistoricalClaim))
    A.add((subj, RDFS.label, Literal(
        f"Thompson (1917) claim {cid} -- Ch. {row['chapter']}, pp. {row['pages']}, Figs. {row['figures']}")))
    A.add((subj, MLT.claimId, Literal(cid)))
    A.add((subj, MLT.dateMade, Literal("1917", datatype=XSD.gYear)))
    A.add((subj, MLT.sourceEdition, Literal(
        "Thompson, D.W. (1917) On Growth and Form, 1st edition, Cambridge University Press "
        "(not the 1942 2nd edition or 1961 abridgement)")))
    A.add((subj, MLT.citation, Literal(
        f"Thompson, D.W. (1917) On Growth and Form. Cambridge University Press. "
        f"Ch. {row['chapter']}, pp. {row['pages']}, Figs. {row['figures']}.")))
    A.add((subj, MLT.chapter, Literal(row["chapter"])))
    A.add((subj, MLT.pages, Literal(str(row["pages"]))))
    A.add((subj, MLT.figures, Literal(str(row["figures"]))))
    A.add((subj, MLT.claimType, Literal(row.get("type", ""))))
    A.add((subj, MLT.claim, Literal(lit["claim"])))
    if row.get("quote"):
        A.add((subj, MLT.verbatimQuote, Literal(row["quote"])))
    A.add((subj, MLT.hasLiteratureVerdict, verdict))

    A.add((verdict, RDF.type, MLT.LiteratureVerdict))
    A.add((verdict, RDFS.label, Literal(
        f"Literature verdict on {cid} (Thompson 1917): {lit['relation']} -- {lit['who_and_when']}")))
    A.add((verdict, MLT.verdictOn, subj))
    A.add((verdict, MLT.literatureRelation, Literal(lit["relation"])))
    A.add((verdict, MLT.literatureWhoAndWhen, Literal(lit["who_and_when"])))
    A.add((verdict, MLT.literatureCitation, Literal(lit["literature_citation"])))
    A.add((verdict, MLT.counterclaim, Literal(lit["counterclaim"])))
    if lit.get("quantitative_margin"):
        A.add((verdict, MLT.quantitativeMargin, Literal(lit["quantitative_margin"])))
    if lit["relation"] == "NO_DIRECT_TEST":
        A.add((verdict, MLT.noDirectTestFoundAsOf, Literal(REVIEW_DATE, datatype=XSD.date)))
        A.add((verdict, MLT.statementOfAbsence, Literal(
            "As of this literature review (2026-10-26), no published work directly tests this "
            "1917 claim as Thompson stated it. Neighbouring results are not a test of this claim.")))
    A.add((verdict, MLT.reviewedOn, Literal(REVIEW_DATE, datatype=XSD.date)))

    # Repo 2026 state: deliberately a different predicate family. Not literature.
    A.add((subj, MLT.repo2026StatusSeparate, Literal(row["pass2"]["status"])))
    A.add((subj, MLT.repo2026AssessmentNotLiterature, Literal(row["pass2"].get("what_remains") or "")))
    A.add((subj, MLT.usesArtifact, artifact(A, CLAIMS_REL.relative_to(REPO_ROOT).as_posix(),
                                            "Thompson 1917 claim registry (59 claims, all dated 1917)")))
    A.add((subj, MLT.usesArtifact, artifact(A, COUNTER_REL.relative_to(REPO_ROOT).as_posix(),
                                            "Thompson 1917 literature counterclaim layer")))
    A.add((subj, DCTERMS.source, THOMPSON_1917))

    activity = URIRef(f"https://github.com/fractastical/morpholib/commit/{commit}")
    script = URIRef(GH + "nanopubs/thompson_1917_counterclaims_to_nanopubs.py")
    P.add((A.identifier, PROV.wasGeneratedBy, activity))
    P.add((activity, RDF.type, PROV.Activity))
    P.add((activity, RDFS.label, Literal(
        f"Thompson (1917) claim+counterclaim nanopub generation at morpholib commit {commit[:12]} "
        "(working-tree source artifacts pinned by SHA-256 in the assertion)")))
    P.add((activity, PROV.used, script))
    P.add((activity, PROV.used, THOMPSON_1917))
    P.add((activity, PROV.endedAtTime, Literal(today, datatype=XSD.date)))
    P.add((A.identifier, PROV.wasAttributedTo, script))
    P.add((script, RDF.type, PROV.SoftwareAgent))
    P.add((script, RDFS.label, Literal(
        "Generator joining thompson/thompson_1917_claims.json (1917 claims) with "
        "thompson/thompson_1917_counterclaims.json (dated literature verdicts)")))

    I.add((base, DCTERMS.created, Literal(today, datatype=XSD.date)))
    I.add((base, DCTERMS.license, URIRef("https://creativecommons.org/publicdomain/zero/1.0/")))
    I.add((base, DCTERMS.isPartOf, URIRef(GH + "docs/THOMPSON_1917_COUNTERCLAIMS.md")))
    if orcid:
        I.add((base, DCTERMS.creator, URIRef(orcid)))
        if name:
            I.add((URIRef(orcid), RDFS.label, Literal(name)))
    return np_id, ds


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sign", action="store_true")
    ap.add_argument("--publish", choices=["test", "prod"])
    ap.add_argument("--out", type=Path, default=OUT_DIR)
    args = ap.parse_args()

    reg = json.loads(CLAIMS_REL.read_text())
    counter = json.loads(COUNTER_REL.read_text())["claims"]
    claims = reg["claims"]
    assert len(claims) == 59, len(claims)
    ids = {c["id"] for c in claims}
    assert ids == set(counter), f"coverage mismatch: {ids ^ set(counter)}"
    for c in claims:
        assert c["date_made"] == "1917", c["id"]
        assert counter[c["id"]]["relation"] in RELATIONS, c["id"]

    commit, today = git_commit(), dt.date.today().isoformat()
    prof = read_profile()
    orcid, name = prof.get("orcid_id"), prof.get("name")
    args.out.mkdir(parents=True, exist_ok=True)

    written = [claim_nanopub(c, counter[c["id"]], commit, today, orcid, name) for c in claims]
    for np_id, ds in written:
        path = args.out / f"np_thompson_{np_id}.trig"
        ds.serialize(destination=str(path), format="trig")
    from collections import Counter
    print(f"wrote {len(written)} claim nanopubs; relations: {dict(Counter(counter[c['id']]['relation'] for c in claims))}")

    if args.sign or args.publish:
        from nanopub import Nanopub, NanopubConf, Profile
        if not prof:
            print("no ~/.nanopub/profile.yml; cannot sign")
            return 1
        profile = Profile(orcid_id=prof["orcid_id"], name=prof["name"],
                          private_key=Path(prof["private_key"]), public_key=Path(prof["public_key"]),
                          introduction_nanopub_uri=prof.get("introduction_nanopub_uri"))
        conf = NanopubConf(profile=profile, use_test_server=(args.publish == "test"),
                           add_prov_generated_time=False, add_pubinfo_generated_time=False,
                           attribute_publication_to_profile=False)
        if args.publish == "prod":
            conf.use_server = PROD_REGISTRY
        ok = pub = 0
        for np_id, _ in written:
            src = args.out / f"np_thompson_{np_id}.trig"
            try:
                npub = Nanopub(rdf=src, conf=conf)
                npub.sign()
                if not npub.has_valid_signature:
                    raise RuntimeError("signature did not verify")
                signed = args.out / f"signed.np_thompson_{np_id}.trig"
                npub.rdf.serialize(destination=str(signed), format="trig")
                ok += 1
                print(f"signed {signed.name} (valid signature)")
                if args.publish:
                    npub.publish()
                    pub += 1
                    manifest = args.out / f"published_manifest_{args.publish}.txt"
                    with manifest.open("a") as fh:
                        fh.write(f"{src.name}|{npub.source_uri}|{dt.datetime.now().isoformat()}|{args.publish}\n")
                    print(f"  published -> {npub.source_uri}")
            except Exception as e:  # noqa: BLE001
                print(f"  failed for {src.name}: {e}")
        print(f"signed {ok}" + (f", published {pub} ({args.publish})" if args.publish else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())

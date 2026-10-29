#!/usr/bin/env python3
"""
Publish the LINEAR-TYPES-01 pre-registered predictions as Project Delphi nanopublications.

Source of truth: predictions/registrations/LINEAR-TYPES-01.json (one delphi:TestableHypothesis per
prediction, with delphi:hasPriorProbability / delphi:hasResolvability / delphi:proposedTest), plus one
portfolio nanopub that lists them all and hash-locks the registration file and the spec.

The priors are the registrant's credence that the stated formal-verification outcome will be achieved
by the deadline using the linear-resource DSL (morphoeval/lean/LinearResource.lean) and the proposed
Project Delphi ontology terms. They are NOT probabilities that any biological claim is true.

Usage:
    python3 nanopubs/linear_types_predictions_to_nanopubs.py                 # write TriG only
    python3 nanopubs/linear_types_predictions_to_nanopubs.py --publish test  # sign + publish to test server
    python3 nanopubs/linear_types_predictions_to_nanopubs.py --publish prod  # sign + publish to production
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

from rdflib import Dataset, Graph, Literal, Namespace, URIRef
from rdflib.namespace import DCTERMS, PROV, RDF, RDFS, SKOS, XSD


def num(x) -> Literal:
    """xsd:decimal with the registration's lexical form. NOTE: Delphi declares xsd:double for the probability
    properties, but rdflib serialises doubles as 9.7e-01 and the nanopub registry's trusty-URI check then
    rejects the artefact (422 'artifact code does not verify'); xsd:decimal round-trips (same workaround as
    thompson_predictions_to_nanopubs.py). Flagged in the ontology review issue."""
    return Literal(str(x), datatype=XSD.decimal)

REPO_ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("tp", REPO_ROOT / "nanopubs" / "thompson_predictions_to_nanopubs.py")
tp = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(tp)  # read_profile, nanopub_conf, git_commit

REGISTRATION = REPO_ROOT / "predictions" / "registrations" / "LINEAR-TYPES-01.json"
SPEC_MD = REPO_ROOT / "docs" / "LINEAR_TYPE_EXTENSION_SPEC.md"
LEAN_PROTO = REPO_ROOT / "morphoeval" / "lean" / "LinearResource.lean"
OUT_DIR = REPO_ROOT / "nanopubs" / "linear_types_predictions"
GITHUB = "https://github.com/fractastical/morpholib"
ONTOLOGY_REPO = "https://github.com/fractastical/project-delphi-ontology"

BASE = "https://w3id.org/morpholib/linear-types/np/"
MLT = Namespace("https://w3id.org/morpholib/linear-types/")
DELPHI = Namespace("https://w3id.org/delphi/ontology/")
DELPHIR = "https://w3id.org/delphi/resource/"
NP = Namespace("http://www.nanopub.org/nschema#")
SCHEMA = Namespace("https://schema.org/")
DOMAIN = URIRef(DELPHIR + "domain/morphoeval")
LEAN_HOME = URIRef("https://lean-lang.org/")


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def new_dataset(np_id: str):
    base = URIRef(BASE + np_id + "/")
    ds = Dataset()
    for pfx, ns in [("np", NP), ("mlt", MLT), ("delphi", DELPHI), ("dcterms", DCTERMS), ("prov", PROV),
                    ("xsd", XSD), ("rdfs", RDFS), ("skos", SKOS), ("schema", SCHEMA)]:
        ds.bind(pfx, ns)
    head = ds.graph(URIRef(base + "Head"))
    A = ds.graph(URIRef(base + "assertion"))
    P = ds.graph(URIRef(base + "provenance"))
    I = ds.graph(URIRef(base + "pubinfo"))
    head.add((base, RDF.type, NP.Nanopublication))
    head.add((base, NP.hasAssertion, A.identifier))
    head.add((base, NP.hasProvenance, P.identifier))
    head.add((base, NP.hasPublicationInfo, I.identifier))
    return ds, base, A, P, I


def add_provenance(P: Graph, assertion: URIRef, commit: str, today: str, reg_digest: str):
    activity = URIRef(f"{GITHUB}/commit/{commit}")
    agent = URIRef(f"{GITHUB}/blob/{commit}/nanopubs/linear_types_predictions_to_nanopubs.py")
    reg_uri = URIRef(f"{GITHUB}/blob/{commit}/predictions/registrations/LINEAR-TYPES-01.json")
    P.add((assertion, PROV.wasGeneratedBy, activity))
    P.add((activity, RDF.type, PROV.Activity))
    P.add((activity, RDFS.label, Literal(f"Pre-registration of LINEAR-TYPES-01 at morpholib commit {commit[:12]}")))
    P.add((activity, PROV.used, reg_uri))
    P.add((activity, PROV.endedAtTime, Literal(today, datatype=XSD.date)))
    P.add((assertion, PROV.wasDerivedFrom, reg_uri))
    P.add((reg_uri, MLT.sha256, Literal(reg_digest)))
    P.add((assertion, PROV.wasAttributedTo, agent))
    P.add((agent, RDF.type, PROV.SoftwareAgent))
    P.add((agent, RDFS.label, Literal("morpholib LINEAR-TYPES-01 registration exporter (Project Delphi vocabulary, ontology 2.0)")))


def add_pubinfo(I: Graph, base: URIRef, today: str, orcid: str | None, name: str | None):
    I.add((base, DCTERMS.created, Literal(today, datatype=XSD.date)))
    I.add((base, DCTERMS.license, URIRef("https://creativecommons.org/licenses/by/4.0/")))
    I.add((base, DCTERMS.conformsTo, URIRef("https://w3id.org/delphi/ontology/2.0")))
    I.add((base, DELPHI.domain, DOMAIN))
    I.add((DOMAIN, RDF.type, DELPHI.Domain))
    I.add((DOMAIN, RDFS.label, Literal("morphoeval")))
    I.add((base, DCTERMS.isPartOf, URIRef(f"{GITHUB}/blob/main/docs/LINEAR_TYPE_EXTENSION_SPEC.md")))
    if orcid:
        I.add((base, DCTERMS.creator, URIRef(orcid)))
        if name:
            I.add((URIRef(orcid), RDFS.label, Literal(name)))


def hypothesis_uri(pid: str) -> URIRef:
    return URIRef(f"{DELPHIR}hypothesis/morpholib-linear-types-{pid.lower()}")


def prediction_nanopub(pred: dict, reg: dict, commit: str, today: str, orcid, name, reg_digest: str):
    pid = pred["prediction_id"]
    np_id = f"prediction-{pid.lower()}"
    ds, base, A, P, I = new_dataset(np_id)
    h = hypothesis_uri(pid)
    A.add((h, RDF.type, DELPHI.TestableHypothesis))
    A.add((h, RDFS.label, Literal(f"{pid}: {pred['title']}", lang="en")))
    A.add((h, DCTERMS.description, Literal(pred["statement"], lang="en")))
    A.add((h, DELPHI.proposedTest, Literal(pred["proposed_test"])))
    A.add((h, DELPHI.hasPriorProbability, num(pred["prior_probability"])))
    A.add((h, DELPHI.hasResolvability, num(pred["resolvability"])))
    A.add((h, DELPHI.hypothesisType, Literal("formal-verification capability prediction (linear-resource DSL in Lean 4)")))
    A.add((h, DELPHI.context, Literal(
        "Pre-registered capability prediction, registration LINEAR-TYPES-01. The prior is the registrant's credence that the "
        "formal-verification outcome is achieved by the deadline, not a probability about biology. "
        f"Loops in scope: {', '.join(pred['loops']) if pred['loops'] else 'none (ontology-level)'}.")))
    A.add((h, MLT.falsifier, Literal(pred["falsifier"])))
    A.add((h, MLT.basis, Literal(pred["basis"])))
    A.add((h, MLT.deadline, Literal(pred["deadline"], datatype=XSD.date)))
    A.add((h, MLT.registrationId, Literal(reg["registration_id"])))
    A.add((h, MLT.predictionId, Literal(pid)))
    A.add((h, SKOS.broader, URIRef(f"{DELPHIR}hypothesis/morpholib-linear-types-portfolio")))
    for loop in pred["loops"]:
        A.add((h, MLT.inScopeLoop, Literal(loop)))
    A.add((h, SCHEMA.programmingLanguage, Literal("Lean 4")))
    A.add((h, RDFS.seeAlso, URIRef(f"{GITHUB}/blob/{commit}/morphoeval/lean/LinearResource.lean")))
    A.add((h, RDFS.seeAlso, URIRef(f"{GITHUB}/blob/{commit}/docs/LINEAR_TYPE_EXTENSION_SPEC.md")))
    add_provenance(P, A.identifier, commit, today, reg_digest)
    add_pubinfo(I, base, today, orcid, name)
    return np_id, ds


def portfolio_nanopub(reg: dict, commit: str, today: str, orcid, name, reg_digest: str):
    np_id = "portfolio-linear-types-01"
    ds, base, A, P, I = new_dataset(np_id)
    s = URIRef(f"{DELPHIR}hypothesis/morpholib-linear-types-portfolio")
    A.add((s, RDF.type, DELPHI.TestableHypothesis))
    A.add((s, RDF.type, MLT.PredictionPortfolio))
    A.add((s, RDFS.label, Literal("LINEAR-TYPES-01: ten pre-registered predictions on what a linear-resource DSL in Lean 4 + Project Delphi proof-kind terms can formalise in the MorphoEval corpus", lang="en")))
    A.add((s, DCTERMS.description, Literal(reg["discipline"], lang="en")))
    ps = reg["portfolio_summary"]
    A.add((s, MLT.nPredictions, Literal(int(ps["n_predictions"]), datatype=XSD.integer)))
    A.add((s, MLT.meanPrior, num(ps["mean_prior"])))
    A.add((s, MLT.expectedConfirmations, num(ps["expected_confirmations"])))
    A.add((s, MLT.scoringRule, Literal(ps["scoring"])))
    A.add((s, DELPHI.hasPriorProbability, num(ps["mean_prior"])))
    A.add((s, DELPHI.hasResolvability, num("0.9")))
    A.add((s, DELPHI.proposedTest, Literal("Resolve each member prediction by its deadline with a resolution nanopub; compute the Brier score at 2027-03-31.")))
    A.add((s, DELPHI.hypothesisType, Literal("formal-verification capability prediction portfolio")))
    A.add((s, DELPHI.context, Literal(
        f"Baseline: {reg['baseline_snapshot']['lean_loop_files']} mechanism-specific Lean loop files, "
        f"{reg['baseline_snapshot']['lean_loop_template_stubs']} template stubs. Prototype {reg['prototype']['module']} "
        f"({reg['prototype']['soundness_theorem']}) kernel-checked with {reg['prototype']['toolchain']}.")))
    for pred in reg["predictions"]:
        h = hypothesis_uri(pred["prediction_id"])
        A.add((s, SKOS.narrower, h))
        A.add((h, RDFS.label, Literal(f"{pred['prediction_id']}: {pred['title']}", lang="en")))
        A.add((h, DELPHI.hasPriorProbability, num(pred["prior_probability"])))
    for label, path in [("registration", REGISTRATION), ("spec", SPEC_MD), ("lean-prototype", LEAN_PROTO)]:
        node = URIRef(f"{s}-locked-{label}")
        A.add((s, MLT.lockedFile, node))
        A.add((node, RDF.type, MLT.LockedFile))
        A.add((node, RDFS.label, Literal(str(path.relative_to(REPO_ROOT)))))
        A.add((node, MLT.sha256, Literal(sha256(path))))
        A.add((node, MLT.url, URIRef(f"{GITHUB}/blob/{commit}/{path.relative_to(REPO_ROOT)}")))
    A.add((s, RDFS.seeAlso, URIRef(ONTOLOGY_REPO)))
    add_provenance(P, A.identifier, commit, today, reg_digest)
    add_pubinfo(I, base, today, orcid, name)
    return np_id, ds


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--publish", choices=["test", "prod"])
    ap.add_argument("--commit", help="git commit that contains the locked files (default: HEAD)")
    args = ap.parse_args()

    reg = json.loads(REGISTRATION.read_text())
    reg_digest = sha256(REGISTRATION)
    commit = args.commit or tp.git_commit()
    today = dt.date.today().isoformat()
    prof = tp.read_profile()
    orcid, name = prof.get("orcid_id"), prof.get("name")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    items = [prediction_nanopub(p, reg, commit, today, orcid, name, reg_digest) for p in reg["predictions"]]
    items.append(portfolio_nanopub(reg, commit, today, orcid, name, reg_digest))

    written = []
    for np_id, ds in items:
        path = OUT_DIR / f"np_linear_types_{np_id}.trig"
        ds.serialize(destination=str(path), format="trig")
        written.append((np_id, path, ds))
        print(f"wrote {path.relative_to(REPO_ROOT)}")

    if args.publish:
        from nanopub import Nanopub
        conf = tp.nanopub_conf(prof, use_test_server=(args.publish == "test"))
        manifest = OUT_DIR / f"published_manifest_{args.publish}.txt"
        for np_id, path, ds in written:
            npub = Nanopub(rdf=ds, conf=conf)
            npub.sign()
            assert npub.has_valid_signature
            npub.rdf.serialize(destination=str(OUT_DIR / f"signed.np_linear_types_{np_id}.trig"), format="trig")
            npub.publish()
            with manifest.open("a") as fh:
                fh.write(f"{path.name}|{npub.source_uri}|{dt.datetime.now().isoformat()}|{args.publish}\n")
            print(f"published ({args.publish}) {np_id} -> {npub.source_uri}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

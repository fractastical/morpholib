#!/usr/bin/env python3
"""
Record the pre-registration of the blind Thompson-engine predictions as a nanopublication:
SHA-256 hashes of the locked prediction files, the git commit, the hypotheses and the scoring
protocol. Publishing this BEFORE the comparison dataset is opened is what makes the test blind.

Usage:
    python3 nanopubs/blind_preregistration_nanopub.py            # write TriG
    python3 nanopubs/blind_preregistration_nanopub.py --publish test|prod
"""
from __future__ import annotations

import argparse
import datetime as dt
import importlib.util
import json
import sys
from pathlib import Path

from rdflib import Literal, URIRef
from rdflib.namespace import DCTERMS, RDF, RDFS, XSD

REPO_ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("tp", REPO_ROOT / "nanopubs" / "thompson_predictions_to_nanopubs.py")
tp = importlib.util.module_from_spec(spec); spec.loader.exec_module(tp)

MANIFEST = REPO_ROOT / "thompson_blind_predictions_manifest.json"
REGISTERED = REPO_ROOT / "thompson_blind_predictions_registered.json"
OUT_DIR = REPO_ROOT / "nanopubs" / "thompson_predictions"
GITHUB = "https://github.com/fractastical/morpholib"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--publish", choices=["test", "prod"])
    ap.add_argument("--commit", help="git commit that contains the locked files (default: HEAD)")
    args = ap.parse_args()

    man = json.loads(MANIFEST.read_text())
    reg = json.loads(REGISTERED.read_text())
    commit = args.commit or tp.git_commit()
    today = dt.date.today().isoformat()
    prof = tp.read_profile()
    np_id = f"blind-preregistration-{today}"
    ds, base, A, P, I = tp.new_dataset(np_id)
    s = URIRef(tp.MLT + f"preregistration/{np_id}")
    A.add((s, RDF.type, tp.MLT.BlindPredictionPreregistration))
    A.add((s, RDFS.label, Literal(f"Pre-registration of {man['n_targets']} blind body-shape predictions (Thompson 1917 phylogenetic prediction engine), {today}")))
    A.add((s, tp.MLT.nTargetSpecies, Literal(int(man["n_targets"]), datatype=XSD.integer)))
    A.add((s, tp.MLT.gitCommit, URIRef(f"{GITHUB}/commit/{commit}")))
    for fname, digest in man["files"].items():
        node = URIRef(f"{s}-file-{fname.replace('.', '-')}")
        A.add((s, tp.MLT.lockedFile, node))
        A.add((node, RDF.type, tp.MLT.LockedFile))
        A.add((node, RDFS.label, Literal(fname)))
        A.add((node, tp.MLT.sha256, Literal(digest)))
        A.add((node, tp.MLT.url, URIRef(f"{GITHUB}/blob/{commit}/{fname}")))
    for h in reg["hypotheses"]:
        node = URIRef(f"{s}-{h['id']}")
        A.add((s, tp.MLT.hasHypothesis, node))
        A.add((node, RDF.type, tp.MLT.PreregisteredHypothesis))
        A.add((node, RDFS.label, Literal(h["id"])))
        A.add((node, tp.MLT.isPrimary, Literal(bool(h["primary"]), datatype=XSD.boolean)))
        A.add((node, DCTERMS.description, Literal(h["statement"])))
    for k, v in reg["protocol"].items():
        A.add((s, tp.MLT[f"protocol_{k}"], Literal(v)))
    for k, v in reg["ratio_definitions"].items():
        A.add((s, tp.MLT[f"ratioDefinition_{k}"], Literal(v)))
    r = reg["registration"]
    A.add((s, tp.MLT.modelThompson7, Literal(r["model_thompson7"])))
    A.add((s, tp.MLT.modelFull30, Literal(r["model_full30"])))
    A.add((s, tp.MLT.nTrainingSpecies, Literal(int(r["n_training_species"]), datatype=XSD.integer)))
    A.add((s, DCTERMS.description, Literal(
        "Predictions of body-proportion ratios and 17-landmark configurations for every species on the Rabosky et al. (2018) tree not in the Claverie & Wainwright (2014) training set, "
        "frozen before the comparison dataset (FishShapes, Price et al. 2022) was downloaded or inspected. The hashes above identify the frozen files; the hypotheses and protocol above are the complete analysis plan.")))

    tp.add_common_provenance(P, A.identifier, commit, today, [tp.LANDMARK_SOURCE, tp.TREE_SOURCE])
    tp.add_common_pubinfo(I, base, today, prof.get("orcid_id"), prof.get("name"))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"np_thompson_{np_id}.trig"
    ds.serialize(destination=str(path), format="trig")
    print(f"wrote {path}")

    if args.publish:
        from nanopub import Nanopub
        conf = tp.nanopub_conf(prof, use_test_server=(args.publish == "test"))
        npub = Nanopub(rdf=path, conf=conf)
        npub.sign()
        assert npub.has_valid_signature
        npub.rdf.serialize(destination=str(OUT_DIR / f"signed.np_thompson_{np_id}.trig"), format="trig")
        npub.publish()
        with (OUT_DIR / f"published_manifest_{args.publish}.txt").open("a") as fh:
            fh.write(f"{path.name}|{npub.source_uri}|{dt.datetime.now().isoformat()}|{args.publish}\n")
        print(f"published ({args.publish}) -> {npub.source_uri}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

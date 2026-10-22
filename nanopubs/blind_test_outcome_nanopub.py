#!/usr/bin/env python3
"""
Record the outcome of the pre-registered blind test (score_blind_predictions.py) as a
nanopublication that cites the preregistration nanopub, the locked-file hashes, the
comparison-data checksum, and the pass/fail of every registered hypothesis.

Usage:
    python3 nanopubs/blind_test_outcome_nanopub.py [--publish test|prod]
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
spec = importlib.util.spec_from_file_location("tp", REPO_ROOT / "nanopubs" / "thompson_predictions_to_nanopubs.py")
tp = importlib.util.module_from_spec(spec); spec.loader.exec_module(tp)

RESULTS = REPO_ROOT / "thompson_blind_test_results.json"
REGISTERED = REPO_ROOT / "thompson_blind_predictions_registered.json"
OUT_DIR = REPO_ROOT / "nanopubs" / "thompson_predictions"
MANIFEST_TXT = OUT_DIR / "published_manifest_test.txt"
GITHUB = "https://github.com/fractastical/morpholib"
FISHSHAPES_DOI = URIRef("https://doi.org/10.5061/dryad.vt4b8gtvf")
FISHSHAPES_ZENODO = URIRef("https://zenodo.org/records/6642650")
FISHSHAPES_PAPER = URIRef("https://doi.org/10.1002/ecy.3829")


def prereg_uri() -> str | None:
    if not MANIFEST_TXT.exists():
        return None
    for line in MANIFEST_TXT.read_text().splitlines():
        if "blind-preregistration" in line:
            return line.split("|")[1]
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--publish", choices=["test", "prod"])
    args = ap.parse_args()
    res = json.loads(RESULTS.read_text())
    reg = json.loads(REGISTERED.read_text())
    commit = tp.git_commit()
    today = dt.date.today().isoformat()
    prof = tp.read_profile()
    np_id = f"blind-test-outcome-{today}"
    ds, base, A, P, I = tp.new_dataset(np_id)
    s = URIRef(tp.MLT + f"blind-test/{np_id}")
    H = res["hypotheses"]
    n_pass = sum(H[f"{h['id']}_cal"]["pass"] for h in reg["hypotheses"])
    n_prim = sum(H[f"{h['id']}_cal"]["pass"] for h in reg["hypotheses"] if h["primary"])
    A.add((s, RDF.type, tp.MLT.BlindPredictionTestOutcome))
    A.add((s, RDFS.label, Literal(f"Outcome of the pre-registered blind test of the Thompson 1917 prediction engine against FishShapes v1: {n_pass}/7 hypotheses supported ({n_prim}/4 primary), {res['n_matched']} species")))
    pre = prereg_uri()
    if pre:
        A.add((s, tp.MLT.testsPreregistration, URIRef(pre)))
    for fname, digest in res["locked_manifest"]["files"].items():
        A.add((s, tp.MLT.verifiedLockedFileSha256, Literal(f"{fname}: {digest}")))
    A.add((s, tp.MLT.comparisonDataset, FISHSHAPES_DOI))
    A.add((s, tp.MLT.comparisonDatasetMirror, FISHSHAPES_ZENODO))
    A.add((s, tp.MLT.comparisonDatasetMd5, Literal(res["fishshapes_md5"])))
    A.add((s, tp.MLT.comparisonDatasetSha256, Literal(res["fishshapes_sha256"])))
    A.add((s, tp.MLT.nSpeciesScored, Literal(int(res["n_matched"]), datatype=XSD.integer)))
    A.add((s, tp.MLT.nSpecimens, Literal(int(res["n_specimens"]), datatype=XSD.integer)))
    for k, c in res["calibration"].items():
        A.add((s, tp.MLT[f"calibration_{k}"], Literal(f"calliper = {c['alpha']:.4f} + {c['beta']:.4f} x landmark; r = {c['r']:.3f}; n_training = {c['n_training_species_in_fishshapes']}")))
    for h in reg["hypotheses"]:
        hid = h["id"]; r_ = H[f"{hid}_cal"]
        node = URIRef(f"{s}-{hid}")
        A.add((s, tp.MLT.hasHypothesisOutcome, node))
        A.add((node, RDF.type, tp.MLT.HypothesisOutcome))
        A.add((node, RDFS.label, Literal(hid)))
        A.add((node, tp.MLT.isPrimary, Literal(bool(h["primary"]), datatype=XSD.boolean)))
        A.add((node, tp.MLT.supported, Literal(bool(r_["pass"]), datatype=XSD.boolean)))
        A.add((node, DCTERMS.description, Literal(h["statement"])))
        summary = {k_: v_ for k_, v_ in r_.items() if k_ not in ("pass", "bin_medians", "note")}
        A.add((node, tp.MLT.resultSummary, Literal(json.dumps(summary))))
    tab = res["horizon_table"]["BDmax_over_SL"]
    for t in tab:
        node = URIRef(f"{s}-tau-{int(t['tau_bin'][0])}-{int(t['tau_bin'][1])}")
        A.add((s, tp.MLT.hasHorizonBin, node))
        A.add((node, RDF.type, tp.MLT.HorizonBin))
        A.add((node, tp.MLT.tauLowerMyr, Literal(t["tau_bin"][0], datatype=XSD.decimal)))
        A.add((node, tp.MLT.tauUpperMyr, Literal(t["tau_bin"][1], datatype=XSD.decimal)))
        A.add((node, tp.MLT.n, Literal(int(t["n"]), datatype=XSD.integer)))
        A.add((node, tp.MLT.medianAbsErrFull30Calibrated, Literal(round(t["median_abs_err_full30_cal"], 4), datatype=XSD.decimal)))
        A.add((node, tp.MLT.medianAbsErrNearestCalibrated, Literal(round(t["median_abs_err_nearest_cal"], 4), datatype=XSD.decimal)))
        A.add((node, tp.MLT.medianAbsErrConsensusCalibrated, Literal(round(t["median_abs_err_consensus_cal"], 4), datatype=XSD.decimal)))
        A.add((node, tp.MLT.coverage95Full30Calibrated, Literal(round(t["coverage95_full30_cal"], 1), datatype=XSD.decimal)))
    A.add((s, DCTERMS.description, Literal(
        "Within 50 Myr of a training species the full 30-D phylogenetic prediction of maximum body depth / standard length has 59% less error than the consensus (H1) and 93% interval coverage (H3), "
        "but improves on copying the nearest training relative by only 4% (H2 not supported). Beyond 50 Myr error triples and coverage falls to ~65%, yet the root-state prediction still halves the consensus error, "
        "so H4 ('no better than consensus') was not supported: the deep-horizon statement should be 'degrades to a clade-wide prior', not 'to the consensus'. Thompson-7 modes beat the consensus but are ~50% worse than the full shape space (H6). "
        "Caudal-peduncle depth replicates the body-depth result (H7).")))
    A.add((s, tp.MLT.resultsDocument, URIRef(f"{GITHUB}/blob/main/docs/THOMPSON_BLIND_TEST_RESULTS.md")))

    tp.add_common_provenance(P, A.identifier, commit, today, [tp.LANDMARK_SOURCE, tp.TREE_SOURCE, FISHSHAPES_DOI, FISHSHAPES_PAPER])
    P.add((A.identifier, PROV.wasGeneratedBy, URIRef(f"{GITHUB}/blob/main/thompson/score_blind_predictions.py")))
    tp.add_common_pubinfo(I, base, today, prof.get("orcid_id"), prof.get("name"))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"np_thompson_{np_id}.trig"
    ds.serialize(destination=str(path), format="trig")
    print(f"wrote {path}")
    if args.publish:
        from nanopub import Nanopub
        conf = tp.nanopub_conf(prof, use_test_server=(args.publish == "test"))
        npub = Nanopub(rdf=path, conf=conf)
        npub.sign(); assert npub.has_valid_signature
        npub.rdf.serialize(destination=str(OUT_DIR / f"signed.np_thompson_{np_id}.trig"), format="trig")
        npub.publish()
        with (OUT_DIR / f"published_manifest_{args.publish}.txt").open("a") as fh:
            fh.write(f"{path.name}|{npub.source_uri}|{dt.datetime.now().isoformat()}|{args.publish}\n")
        print(f"published ({args.publish}) -> {npub.source_uri}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

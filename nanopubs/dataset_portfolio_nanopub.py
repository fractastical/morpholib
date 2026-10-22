#!/usr/bin/env python3
"""
Record the forward dataset portfolio (thompson_dataset_portfolio.json) as a nanopublication:
one node per candidate dataset / group with its registered confidence, plus the
portfolio-level hypotheses and the sub-prediction protocol.  Cites the FishShapes
preregistration and outcome nanopubs.

Usage: python3 nanopubs/dataset_portfolio_nanopub.py [--publish test|prod] [--commit SHA]
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

from rdflib import Literal, URIRef
from rdflib.namespace import DCTERMS, PROV, RDF, RDFS, XSD

REPO_ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("tp", REPO_ROOT / "nanopubs" / "thompson_predictions_to_nanopubs.py")
tp = importlib.util.module_from_spec(spec); spec.loader.exec_module(tp)
PORTFOLIO = REPO_ROOT / "thompson_dataset_portfolio.json"
OUT_DIR = REPO_ROOT / "nanopubs" / "thompson_predictions"
GITHUB = "https://github.com/fractastical/morpholib"
NP = "https://w3id.org/morpholib/thompson/np/"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--publish", choices=["test", "prod"])
    ap.add_argument("--commit")
    args = ap.parse_args()
    P = json.loads(PORTFOLIO.read_text())
    commit = args.commit or tp.git_commit()
    today = dt.date.today().isoformat()
    prof = tp.read_profile()
    np_id = f"dataset-portfolio-{today}"
    ds, base, A, Pv, I = tp.new_dataset(np_id)
    s = URIRef(tp.MLT + f"portfolio/{np_id}")
    A.add((s, RDF.type, tp.MLT.PredictionPortfolio))
    A.add((s, RDFS.label, Literal(f"Forward dataset portfolio for the Thompson 1917 phylogenetic prediction engine: {len(P['fish_datasets']) + len(P['fish_subjective'])} fish datasets and {len(P['other_groups'])} other groups with registered confidence, {today}")))
    A.add((s, tp.MLT.portfolioFileSha256, Literal(hashlib.sha256(PORTFOLIO.read_bytes()).hexdigest())))
    A.add((s, tp.MLT.anchoredOnPreregistration, URIRef(NP + "blind-preregistration-2026-10-22/" + P["anchor"]["preregistration_nanopub"])))
    A.add((s, tp.MLT.anchoredOnOutcome, URIRef(NP + "blind-test-outcome-2026-10-22/" + P["anchor"]["outcome_nanopub_prod"])))
    A.add((s, tp.MLT.horizonRuleFraction, Literal(round(P["horizon_rule"]["fraction"], 3), datatype=XSD.decimal)))
    A.add((s, DCTERMS.description, Literal(
        "States, before the comparisons are made, which evolutionary-biology datasets the engine (ray-finned fishes, predictions locked at commit dac08493) or the same machinery "
        "re-trained on another clade should have predictive power on, with a probability for each registered test. Simulated probabilities resample the FishShapes per-species errors in "
        "matching horizon bins for the species actually in scope; subjective probabilities are priors written down to be Brier-scored later. Species-level sub-predictions are registered per dataset as the data are obtained.")))
    for d in P["fish_datasets"]:
        node = URIRef(f"{s}-{d['id']}")
        A.add((s, tp.MLT.hasCandidateDataset, node))
        A.add((node, RDF.type, tp.MLT.CandidateDataset))
        A.add((node, RDFS.label, Literal(d["name"])))
        A.add((node, DCTERMS.source, Literal(d["source"])))
        A.add((node, tp.MLT.datasetClass, Literal(d["dataset_class"])))
        A.add((node, tp.MLT.status, Literal(d["status"])))
        A.add((node, tp.MLT.nSpeciesInScope, Literal(int(d["scope_n"]), datatype=XSD.integer)))
        A.add((node, tp.MLT.confidenceType, Literal(d["confidence_type"])))
        if d.get("tau_summary", {}).get("frac_within_50") is not None:
            A.add((node, tp.MLT.fractionWithin50Myr, Literal(round(d["tau_summary"]["frac_within_50"], 3), datatype=XSD.decimal)))
        if d.get("simulated"):
            for k in ("ratio_consensus_near", "ratio_nearest_near", "coverage_near", "ratio_consensus_far", "coverage_far"):
                v = d["simulated"].get(k)
                if v:
                    A.add((node, tp.MLT[f"predicted_{k}"], Literal(f"{v['median']:.3f} [{v['q05']:.3f}, {v['q95']:.3f}]")))
        for t, pp in d.get("p_pass_registered", {}).items():
            A.add((node, tp.MLT.traitProbability, Literal(json.dumps({"trait": t, **{k_: (round(v, 3) if isinstance(v, float) else v) for k_, v in pp.items()}}))))
        if d.get("P3_deep_prediction"):
            A.add((node, tp.MLT.deepHorizonPassInterval, Literal(json.dumps(d["P3_deep_prediction"]))))
        if d.get("realised"):
            A.add((node, tp.MLT.realised, Literal(json.dumps(d["realised"]))))
    for d in P["fish_subjective"]:
        node = URIRef(f"{s}-{d['id']}")
        A.add((s, tp.MLT.hasCandidateDataset, node))
        A.add((node, RDF.type, tp.MLT.CandidateDataset))
        A.add((node, RDFS.label, Literal(d["name"])))
        A.add((node, DCTERMS.source, Literal(d["source"])))
        A.add((node, tp.MLT.datasetClass, Literal(d["dataset_class"])))
        A.add((node, tp.MLT.prediction, Literal(d["prediction"])))
        A.add((node, tp.MLT.probabilitySubjective, Literal(d["p_pass"], datatype=XSD.decimal)))
        A.add((node, tp.MLT.reasoning, Literal(d["reasoning"])))
    for g in P["other_groups"]:
        node = URIRef(f"{s}-{g['id']}")
        A.add((s, tp.MLT.hasCandidateGroup, node))
        A.add((node, RDF.type, tp.MLT.CandidateGroup))
        A.add((node, RDFS.label, Literal(g["group"])))
        A.add((node, tp.MLT.phylogeny, Literal(g["phylogeny"])))
        A.add((node, tp.MLT.traitData, Literal(g["traits"])))
        A.add((node, tp.MLT.traitType, Literal(g["trait_type"])))
        A.add((node, tp.MLT.crownAgeMaApprox, Literal(g["crown_age_ma"], datatype=XSD.decimal)))
        A.add((node, tp.MLT.predictedHorizonMyr, Literal(g["predicted_horizon_myr"], datatype=XSD.decimal)))
        A.add((node, tp.MLT.blindDesign, Literal(g["blind_design"])))
        A.add((node, tp.MLT.pBeatsConsensusSubjective, Literal(g["p_beats_consensus"], datatype=XSD.decimal)))
        A.add((node, tp.MLT.pCoverageCalibratedSubjective, Literal(g["p_coverage_calibrated"], datatype=XSD.decimal)))
        A.add((node, tp.MLT.pBeatsNearestSubjective, Literal(g["p_beats_nearest"], datatype=XSD.decimal)))
        A.add((node, tp.MLT.reasoning, Literal(g["reasoning"])))
    for h in P["portfolio_hypotheses"]:
        node = URIRef(f"{s}-{h['id']}")
        A.add((s, tp.MLT.hasHypothesis, node))
        A.add((node, RDF.type, tp.MLT.PortfolioHypothesis))
        A.add((node, RDFS.label, Literal(h["id"])))
        A.add((node, tp.MLT.isPrimary, Literal(bool(h["primary"]), datatype=XSD.boolean)))
        A.add((node, DCTERMS.description, Literal(h["statement"])))
        A.add((node, tp.MLT.confidence, Literal(h["confidence"])))
    for k, v in P["subprediction_protocol"].items():
        A.add((s, tp.MLT[f"protocol_{k}"], Literal(v)))
    A.add((s, tp.MLT.portfolioDocument, URIRef(f"{GITHUB}/blob/{commit}/docs/THOMPSON_DATASET_PORTFOLIO.md")))

    tp.add_common_provenance(Pv, A.identifier, commit, today, [tp.LANDMARK_SOURCE, tp.TREE_SOURCE])
    Pv.add((A.identifier, PROV.wasGeneratedBy, URIRef(f"{GITHUB}/blob/{commit}/thompson/register_dataset_portfolio.py")))
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

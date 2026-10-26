#!/usr/bin/env python3
"""
Record the outcome of the pre-registered Zebrahub zebrafish 24 hpf blind test
(embryo-blind/zebrafish-blind/, scored by score_zebrafish_blind.py --unblind on
2026-10-26) as a nanopublication: locked-file hashes, sealed-target checksum,
and the pass/fail of every registered hypothesis (Z1-Z4; 0/4 supported).

Usage:
    python3 nanopubs/zebrafish_blind_outcome_nanopub.py [--publish test|prod]
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

ZB = REPO_ROOT / "embryo-blind" / "zebrafish-blind"
RESULTS = ZB / "zebrafish_blind_results.json"
REGISTERED = ZB / "zebrafish_predictions_registered.json"
MANIFEST = ZB / "zebrafish_manifest.json"
OUT_DIR = REPO_ROOT / "nanopubs" / "zebrafish_blind"
GITHUB = "https://github.com/fractastical/morpholib"
ZEBRAHUB_FIGSHARE = URIRef("https://doi.org/10.6084/m9.figshare.20510367.v1")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--publish", choices=["test", "prod"])
    args = ap.parse_args()
    res = json.loads(RESULTS.read_text())
    reg = json.loads(REGISTERED.read_text())
    man = json.loads(MANIFEST.read_text())
    commit = tp.git_commit()
    today = dt.date.today().isoformat()
    prof = tp.read_profile()
    np_id = f"zebrafish-blind-test-outcome-{today}"
    ds, base, A, P, I = tp.new_dataset(np_id)
    s = URIRef(tp.MLT + f"blind-test/{np_id}")
    H = res["hypotheses"]
    n_pass = sum(H[h["id"]]["pass"] for h in reg["hypotheses"])
    n_prim = sum(H[h["id"]]["pass"] for h in reg["hypotheses"] if h["primary"])
    A.add((s, RDF.type, tp.MLT.BlindPredictionTestOutcome))
    A.add((s, RDFS.label, Literal(f"Outcome of the pre-registered blind test of 24 hpf zebrafish cell-type composition (Zebrahub): {n_pass}/4 hypotheses supported ({n_prim}/2 primary), {res['n_classes_union']} classes in union")))
    for fname, digest in man["files"].items():
        A.add((s, tp.MLT.verifiedLockedFileSha256, Literal(f"{fname}: {digest}")))
    A.add((s, tp.MLT.comparisonDataset, ZEBRAHUB_FIGSHARE))
    A.add((s, tp.MLT.comparisonDatasetSha256, Literal(man["sealed_target"]["sha256"])))
    A.add((s, tp.MLT.comparisonDatasetMd5, Literal(man["sealed_target"]["md5_figshare"])))
    for h in reg["hypotheses"]:
        hid = h["id"]; r_ = H[hid]
        node = URIRef(f"{s}-{hid}")
        A.add((s, tp.MLT.hasHypothesisOutcome, node))
        A.add((node, RDF.type, tp.MLT.HypothesisOutcome))
        A.add((node, RDFS.label, Literal(hid)))
        A.add((node, tp.MLT.isPrimary, Literal(bool(h["primary"]), datatype=XSD.boolean)))
        A.add((node, tp.MLT.supported, Literal(bool(r_["pass"]), datatype=XSD.boolean)))
        A.add((node, DCTERMS.description, Literal(h["statement"])))
        summary = {k_: v_ for k_, v_ in r_.items() if k_ != "pass"}
        A.add((node, tp.MLT.resultSummary, Literal(json.dumps(summary))))
    A.add((s, DCTERMS.description, Literal(
        "All four registered hypotheses failed. The logit-linear extrapolation from 10-19 hpf is worse than simply copying the 19 hpf composition "
        f"(Z1: TVD {H['Z1']['tvd_primary']:.3f} vs last-stage {H['Z1']['tvd_last19']:.3f}); only {H['Z2']['frac_within_factor2']:.0%} of classes >=2% land within a factor of 2 (Z2); "
        f"emergent classes carry {H['Z3']['emergent_mass_at_24hpf']:.2f} of 24 hpf cells (Z3 bound 0.25); interval coverage is {H['Z4']['coverage_ge_0.01']:.0%} at a narrow median width (Z4). "
        "Per-class inspection shows developmental label succession, not smooth proportion drift: optic vesicle (predicted 0.136) is replaced by optic cup (true 0.072, predicted 0), "
        "and hindbrain/neurons/fin/muscle pioneers emerge after 19 hpf. Cell-type composition at this stage transition turns over by maturation and renaming, "
        "which a per-class proportion extrapolator cannot represent. Negative result, reported as registered.")))
    A.add((s, tp.MLT.resultsDocument, URIRef(f"{GITHUB}/blob/main/embryo-blind/zebrafish-blind/ZEBRAFISH_BLIND_RESULTS.md")))
    A.add((s, tp.MLT.preregistrationDocument, URIRef(f"{GITHUB}/blob/main/embryo-blind/zebrafish-blind/ZEBRAFISH_BLIND_PREREGISTRATION.md")))

    tp.add_common_provenance(P, A.identifier, commit, today, [ZEBRAHUB_FIGSHARE])
    P.add((A.identifier, PROV.wasGeneratedBy, URIRef(f"{GITHUB}/blob/main/embryo-blind/zebrafish-blind/score_zebrafish_blind.py")))
    tp.add_common_pubinfo(I, base, today, prof.get("orcid_id"), prof.get("name"))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"np_{np_id}.trig"
    ds.serialize(destination=str(path), format="trig")
    print(f"wrote {path}")
    if args.publish:
        from nanopub import Nanopub
        conf = tp.nanopub_conf(prof, use_test_server=(args.publish == "test"))
        npub = Nanopub(rdf=path, conf=conf)
        npub.sign(); assert npub.has_valid_signature
        npub.rdf.serialize(destination=str(OUT_DIR / f"signed.np_{np_id}.trig"), format="trig")
        npub.publish()
        with (OUT_DIR / f"published_manifest_{args.publish}.txt").open("a") as fh:
            fh.write(f"{path.name}|{npub.source_uri}|{dt.datetime.now().isoformat()}|{args.publish}\n")
        print(f"published ({args.publish}) -> {npub.source_uri}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

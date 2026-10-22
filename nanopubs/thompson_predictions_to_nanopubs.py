#!/usr/bin/env python3
"""
Record the registered phylogenetic predictions of the Thompson 1917 prediction engine
as nanopublications (one TriG file per prediction), plus one nanopub per Monte Bolca
fossil-test outcome and one summary nanopub for the engine's validation.

Input : thompson_phylogenetic_prediction_engine.json (written by
        thompson/run_phylogenetic_prediction_engine.py)
Output: nanopubs/thompson_predictions/np_thompson_pred_<id>.trig

Each prediction nanopub asserts, for a named lineage at a named age:
  * the predicted body-proportion ratios (BDmax/SL, HL/SL, PD/SL) with 95% intervals,
    from the full 30-D shape-space model and from the 7 Thompson modes;
  * the predicted 17-landmark Procrustes configuration (JSON literal);
  * the expected landmark error at that horizon (from the held-out calibration);
  * an explicit falsification criterion.
The provenance graph ties the assertion to the script, the git commit, and the two
source datasets (Claverie & Wainwright 2014 landmarks; Rabosky et al. 2018 tree).

Usage:
    python3 nanopubs/thompson_predictions_to_nanopubs.py            # write TriG files
    python3 nanopubs/thompson_predictions_to_nanopubs.py --sign     # also sign with ~/.nanopub profile
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import subprocess
import sys
from pathlib import Path

from rdflib import Dataset, Graph, Literal, Namespace, URIRef
from rdflib.namespace import DCTERMS, PROV, RDF, RDFS, XSD

REPO_ROOT = Path(__file__).resolve().parents[1]
IN_JSON = REPO_ROOT / "thompson_phylogenetic_prediction_engine.json"
OUT_DIR = REPO_ROOT / "nanopubs" / "thompson_predictions"
PROFILE = Path.home() / ".nanopub" / "profile.yml"

BASE = "https://w3id.org/morpholib/thompson/np/"
MLT = Namespace("https://w3id.org/morpholib/thompson/")
NP = Namespace("http://www.nanopub.org/nschema#")
CITO = Namespace("http://purl.org/spar/cito/")

SCRIPT_URI = URIRef("https://github.com/fractastical/morpholib/blob/main/thompson/run_phylogenetic_prediction_engine.py")
PREDICTIONS_DOC = URIRef("https://github.com/fractastical/morpholib/blob/main/docs/THOMPSON_PHYLOGENETIC_PREDICTIONS.md")
LANDMARK_SOURCE = URIRef("https://doi.org/10.1371/journal.pone.0112732")   # Claverie & Wainwright 2014, PLoS ONE
TREE_SOURCE = URIRef("https://doi.org/10.1038/s41586-018-0273-1")           # Rabosky et al. 2018, Nature
THOMPSON_1917 = URIRef("https://www.biodiversitylibrary.org/bibliography/11332")  # On Growth and Form (1917)

RATIO_LABELS = {
    "BDmax_over_SL": "maximum body depth (excluding fin webs) / standard length",
    "BD_over_SL": "dorsal-fin origin to anal-fin origin / standard length",
    "HL_over_SL": "head length / standard length",
    "PD_over_SL": "predorsal length / standard length",
}
KIND_LABELS = {
    "crown": "crown-node ancestral body plan of a sampled family",
    "mrca": "most recent common ancestor of one of D'Arcy Thompson's 1917 transformation pairs",
    "bolca50": "state of the family lineage at the age of the Monte Bolca Lagerstaette (50 Ma)",
}


def git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


def read_profile() -> dict:
    """Parse ~/.nanopub/profile.yml ourselves (nanopub.load_profile() trips over an empty
    introduction_nanopub_uri under yatiml on recent Pythons)."""
    if not PROFILE.exists():
        return {}
    out = {}
    for line in PROFILE.read_text().splitlines():
        if ":" in line and not line.startswith("#"):
            k, v = line.split(":", 1)
            out[k.strip()] = v.strip() or None
    return out


def nanopub_conf(prof: dict, use_test_server: bool):
    from nanopub import NanopubConf, Profile
    profile = Profile(orcid_id=prof["orcid_id"], name=prof["name"], private_key=Path(prof["private_key"]), public_key=Path(prof["public_key"]),
                      introduction_nanopub_uri=prof.get("introduction_nanopub_uri"))
    conf = NanopubConf(profile=profile, use_test_server=use_test_server, add_prov_generated_time=False,
                       add_pubinfo_generated_time=False, attribute_publication_to_profile=False)
    if not use_test_server:
        # nanopub-py's default production server (registry.petapico.org) refuses connections;
        # use the KnowledgePixels production registry, as nanopubs/publish_all_nanopubs.py does.
        conf.use_server = PROD_REGISTRY
    return conf


PROD_REGISTRY = "https://registry.knowledgepixels.com/np/"


def slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "-", s).strip("-")


def new_dataset(np_id: str):
    base = URIRef(BASE + np_id + "/")
    ds = Dataset()
    for pfx, ns in [("np", NP), ("mlt", MLT), ("dcterms", DCTERMS), ("prov", PROV), ("xsd", XSD), ("rdfs", RDFS), ("cito", CITO)]:
        ds.bind(pfx, ns)
    head = ds.graph(URIRef(base + "Head"))
    assertion = ds.graph(URIRef(base + "assertion"))
    provenance = ds.graph(URIRef(base + "provenance"))
    pubinfo = ds.graph(URIRef(base + "pubinfo"))
    head.add((base, RDF.type, NP.Nanopublication))
    head.add((base, NP.hasAssertion, assertion.identifier))
    head.add((base, NP.hasProvenance, provenance.identifier))
    head.add((base, NP.hasPublicationInfo, pubinfo.identifier))
    return ds, base, assertion, provenance, pubinfo


def add_common_provenance(provenance: Graph, assertion_uri: URIRef, commit: str, generated_on: str, derived_from: list[URIRef]):
    activity = URIRef(f"https://github.com/fractastical/morpholib/commit/{commit}")
    provenance.add((assertion_uri, PROV.wasGeneratedBy, activity))
    provenance.add((activity, RDF.type, PROV.Activity))
    provenance.add((activity, RDFS.label, Literal(f"Run of thompson/run_phylogenetic_prediction_engine.py at morpholib commit {commit[:12]}")))
    provenance.add((activity, PROV.used, SCRIPT_URI))
    provenance.add((activity, PROV.used, LANDMARK_SOURCE))
    provenance.add((activity, PROV.used, TREE_SOURCE))
    provenance.add((activity, PROV.endedAtTime, Literal(generated_on, datatype=XSD.date)))
    for d in derived_from:
        provenance.add((assertion_uri, PROV.wasDerivedFrom, d))
    provenance.add((assertion_uri, PROV.wasAttributedTo, SCRIPT_URI))
    provenance.add((SCRIPT_URI, RDF.type, PROV.SoftwareAgent))
    provenance.add((SCRIPT_URI, RDFS.label, Literal("Thompson 1917 phylogenetic prediction engine (7 linearised Thompson modes + Pagel-lambda PGLS on the Rabosky 2018 tree)")))


def add_common_pubinfo(pubinfo: Graph, base: URIRef, today: str, orcid: str | None, name: str | None):
    pubinfo.add((base, DCTERMS.created, Literal(today, datatype=XSD.date)))
    pubinfo.add((base, DCTERMS.license, URIRef("https://creativecommons.org/publicdomain/zero/1.0/")))
    pubinfo.add((base, DCTERMS.isPartOf, PREDICTIONS_DOC))
    if orcid:
        pubinfo.add((base, DCTERMS.creator, URIRef(orcid)))
        if name:
            pubinfo.add((URIRef(orcid), RDFS.label, Literal(name)))


def add_ratio_block(g: Graph, subj: URIRef, prefix: str, ratios: dict, ratios_95: dict, model_label: str):
    for key, val in ratios.items():
        if key not in RATIO_LABELS:
            continue
        node = URIRef(f"{subj}-{prefix}-{key}")
        g.add((subj, MLT.hasPredictedRatio, node))
        g.add((node, RDF.type, MLT.PredictedRatio))
        g.add((node, MLT.ratioName, Literal(key)))
        g.add((node, RDFS.label, Literal(f"{RATIO_LABELS[key]} ({model_label})")))
        g.add((node, MLT.model, Literal(model_label)))
        g.add((node, MLT.pointEstimate, Literal(round(val, 4), datatype=XSD.decimal)))
        lo, hi = ratios_95[key]
        g.add((node, MLT.lower95, Literal(round(lo, 4), datatype=XSD.decimal)))
        g.add((node, MLT.upper95, Literal(round(hi, 4), datatype=XSD.decimal)))


def prediction_nanopub(rec: dict, commit: str, today: str, orcid, name) -> tuple[str, Dataset]:
    kind, target, age = rec["kind"], rec["target"], rec["age_ma"]
    np_id = f"pred-{kind}-{slug(target)}"
    ds, base, A, P, I = new_dataset(np_id)
    pred = URIRef(MLT + f"prediction/{np_id}")
    fu, th = rec["full30"], rec["thompson7"]
    bd, bd95 = fu["ratios"]["BDmax_over_SL"], fu["ratios_95"]["BDmax_over_SL"]

    A.add((pred, RDF.type, MLT.PhylogeneticMorphologyPrediction))
    A.add((pred, MLT.predictionKind, Literal(kind)))
    A.add((pred, RDFS.label, Literal(f"Predicted body plan: {target.replace('_', ' ')} at {age:.1f} Ma ({KIND_LABELS[kind]})")))
    A.add((pred, MLT.targetLineage, Literal(target.replace("_", " "))))
    A.add((pred, MLT.ageMa, Literal(round(age, 2), datatype=XSD.decimal)))
    A.add((pred, MLT.nSampledLivingSpecies, Literal(int(rec["n_sampled_tips"]), datatype=XSD.integer)))
    if kind == "bolca50":
        A.add((pred, MLT.sampledCrownAgeMa, Literal(round(rec["family_crown_age_ma"], 2), datatype=XSD.decimal)))
        A.add((pred, MLT.lineageRepresentedByTip, Literal(rec["lineage_tip"].replace("_", " "))))
        A.add((pred, MLT.knownFossilTaxaAtThisAge, Literal(rec["bolca_taxa"])))
        for k_, v_ in rec["living_family_mean_ratios"].items():
            if k_ in RATIO_LABELS:
                A.add((pred, MLT[f"livingFamilyMean_{k_}"], Literal(round(v_, 4), datatype=XSD.decimal)))
                A.add((pred, MLT[f"livingFamilySD_{k_}"], Literal(round(rec["living_family_ratio_sd"][k_], 4), datatype=XSD.decimal)))
    add_ratio_block(A, pred, "full30", fu["ratios"], fu["ratios_95"], "full 30-D shape-space Pagel-lambda model (primary)")
    add_ratio_block(A, pred, "thompson7", th["ratios"], th["ratios_95"], "7 linearised Thompson modes")
    A.add((pred, MLT.predictedLandmarksFull30, Literal(json.dumps([[round(x, 5) for x in p] for p in fu["pred_landmarks"]]))))
    A.add((pred, MLT.predictedLandmarksThompson7, Literal(json.dumps([[round(x, 5) for x in p] for p in th["pred_landmarks"]]))))
    A.add((pred, MLT.landmarkScheme, Literal("17 Claverie & Wainwright (2014) landmarks, Procrustes-aligned (centroid size 1, centred), listed in the dataset's landmark order")))
    if rec.get("expected_landmark_rms_at_this_horizon_full30") is not None:
        A.add((pred, MLT.expectedProcrustesRMSFull30, Literal(round(rec["expected_landmark_rms_at_this_horizon_full30"], 4), datatype=XSD.decimal)))
    if rec.get("expected_landmark_rms_at_this_horizon_thompson7") is not None:
        A.add((pred, MLT.expectedProcrustesRMSThompson7, Literal(round(rec["expected_landmark_rms_at_this_horizon_thompson7"], 4), datatype=XSD.decimal)))
    A.add((pred, MLT.evolutionaryProcess, Literal("multivariate Pagel lambda (lambda = 0.959 for the Thompson modes, 0.924 for full shape), selected by AICc over white noise, Brownian motion, Ornstein-Uhlenbeck and Early Burst")))
    A.add((pred, MLT.falsificationCriterion, Literal(
        f"A well-preserved articulated fossil of {target.replace('_', ' ')} dated to {age:.0f} +/- {max(5.0, 0.2 * age):.0f} Ma, lying on or near this lineage, "
        f"with maximum body depth / standard length outside [{bd95[0]:.3f}, {bd95[1]:.3f}] falsifies this prediction (point estimate {bd:.3f}); "
        f"a 17-landmark Procrustes RMS to the predicted configuration much larger than the expected error at this horizon does likewise.")))
    A.add((pred, DCTERMS.description, Literal(
        f"Conditional expectation of the 17-landmark body planform of the {target.replace('_', ' ')} lineage at {age:.1f} Ma given {rec['n_sampled_tips']} sampled living species of the clade "
        f"and 743 landmarked reef-fish species on the Rabosky et al. (2018) time tree. {KIND_LABELS[kind].capitalize()}. Registered before any comparison with fossils of this age.")))

    add_common_provenance(P, A.identifier, commit, today, [LANDMARK_SOURCE, TREE_SOURCE, THOMPSON_1917])
    add_common_pubinfo(I, base, today, orcid, name)
    return np_id, ds


def fossil_test_nanopub(rec: dict, pred_by_fam: dict, commit: str, today: str, orcid, name) -> tuple[str, Dataset]:
    taxon, fam = rec["taxon"], rec["family"]
    np_id = f"fossiltest-{slug(taxon)}"
    ds, base, A, P, I = new_dataset(np_id)
    test = URIRef(MLT + f"fossil-test/{np_id}")
    pred_uri = URIRef(MLT + f"prediction/pred-bolca50-{slug(fam)}")
    A.add((test, RDF.type, MLT.FossilPredictionTest))
    A.add((test, RDFS.label, Literal(f"Monte Bolca test: {taxon.replace('_', ' ')} ({fam}) vs. predicted {fam} lineage at {rec['age_ma']:.0f} Ma")))
    A.add((test, MLT.testsPrediction, pred_uri))
    A.add((test, MLT.fossilTaxon, Literal(taxon.replace("_", " "))))
    A.add((test, MLT.family, Literal(fam)))
    A.add((test, MLT.ageMa, Literal(round(float(rec["age_ma"]), 2), datatype=XSD.decimal)))
    A.add((test, MLT.phylogeneticPlacement, Literal(rec["placement"])))
    A.add((test, DCTERMS.bibliographicCitation, Literal(rec["source"])))
    if rec.get("note"):
        A.add((test, RDFS.comment, Literal(rec["note"])))
    for key, m in rec["measurements"].items():
        node = URIRef(f"{test}-{key}")
        A.add((test, MLT.hasMeasurementComparison, node))
        A.add((node, RDF.type, MLT.MeasurementComparison))
        A.add((node, MLT.ratioName, Literal(key)))
        A.add((node, RDFS.label, Literal(RATIO_LABELS.get(key, key))))
        lo, hi = m["fossil"]
        A.add((node, MLT.fossilValueLower, Literal(round(lo, 4), datatype=XSD.decimal)))
        A.add((node, MLT.fossilValueUpper, Literal(round(hi, 4), datatype=XSD.decimal)))
        A.add((node, MLT.predictedFull30, Literal(round(m["pred_full30"], 4), datatype=XSD.decimal)))
        A.add((node, MLT.predictedFull30Lower95, Literal(round(m["ci95_full30"][0], 4), datatype=XSD.decimal)))
        A.add((node, MLT.predictedFull30Upper95, Literal(round(m["ci95_full30"][1], 4), datatype=XSD.decimal)))
        A.add((node, MLT.predictedThompson7, Literal(round(m["pred_thompson7"], 4), datatype=XSD.decimal)))
        A.add((node, MLT.livingFamilyMean, Literal(round(m["living_family_mean"], 4), datatype=XSD.decimal)))
        A.add((node, MLT.fossilInsidePredictiveInterval95, Literal(bool(m["fossil_inside_ci95_full30"]), datatype=XSD.boolean)))
        A.add((node, MLT.predictionCloserThanStasisNull, Literal(bool(m["prediction_closer_than_living_mean_full30"]), datatype=XSD.boolean)))
        if m["direction_of_change_correct_full30"] is not None:
            A.add((node, MLT.directionOfChangeCorrect, Literal(bool(m["direction_of_change_correct_full30"]), datatype=XSD.boolean)))
    bd = rec["measurements"].get("BDmax_over_SL")
    if bd is not None:
        verdict = "consistent with the prediction" if bd["fossil_inside_ci95_full30"] else "outside the 95% predictive interval (prediction falsified as stated, subject to the fossil's placement on the lineage)"
        A.add((test, MLT.verdictBodyDepth, Literal(verdict)))
    A.add((test, MLT.nullHypothesis, Literal("phylogenetic stasis: fossil proportions equal the living family mean")))

    add_common_provenance(P, A.identifier, commit, today, [LANDMARK_SOURCE, TREE_SOURCE, URIRef("https://github.com/fractastical/morpholib/blob/main/datasets/thompson/monte_bolca_fossil_ratios.json")])
    add_common_pubinfo(I, base, today, orcid, name)
    return np_id, ds


def summary_nanopub(d: dict, n_pred: int, n_tests: int, commit: str, today: str, orcid, name) -> tuple[str, Dataset]:
    np_id = "engine-validation-summary"
    ds, base, A, P, I = new_dataset(np_id)
    s = URIRef(MLT + "engine/validation-summary")
    ts, ms, hv = d["trait_space"], d["model_selection"], d["holdout_validation"]
    A.add((s, RDF.type, MLT.PredictionEngineValidation))
    A.add((s, RDFS.label, Literal("Validation of the Thompson 1917 phylogenetic prediction engine (held-out horizon calibration)")))
    A.add((s, MLT.nTrainingSpecies, Literal(int(ts["n_training_species"]), datatype=XSD.integer)))
    A.add((s, MLT.shapeVarianceCapturedThompson7, Literal(round(ts["shape_variance_captured_thompson7"], 4), datatype=XSD.decimal)))
    A.add((s, MLT.shapeVarianceCapturedPCA7, Literal(round(ts["shape_variance_captured_pca7"], 4), datatype=XSD.decimal)))
    A.add((s, MLT.sigmaGenusProcrustesRMS, Literal(round(ts["sigma_genus_procrustes_rms"], 4), datatype=XSD.decimal)))
    for space in ["thompson_modes", "jaw_residual", "full_shape_30d"]:
        best = min(ms[space], key=lambda k: ms[space][k]["dAICc"])
        A.add((s, MLT[f"bestProcess_{space}"], Literal(ms[space][best]["label"])))
    A.add((s, MLT.nHeldOutPredictions, Literal(int(hv["n_predictions"]), datatype=XSD.integer)))
    A.add((s, MLT.coverage95OverallPct, Literal(round(hv["coverage95_overall_pct"], 1), datatype=XSD.decimal)))
    A.add((s, MLT.shareOfFullSpaceGainCapturedByThompson7, Literal(round(hv["share_of_full_space_prediction_gain_captured_by_thompson7"], 3), datatype=XSD.decimal)))
    for r in hv["horizon_table"]:
        node = URIRef(f"{s}-horizon-{int(r['tau_bin_myr'][0])}-{int(r['tau_bin_myr'][1])}")
        A.add((s, MLT.hasHorizonBin, node))
        A.add((node, RDF.type, MLT.HorizonBin))
        A.add((node, MLT.tauLowerMyr, Literal(r["tau_bin_myr"][0], datatype=XSD.decimal)))
        A.add((node, MLT.tauUpperMyr, Literal(r["tau_bin_myr"][1], datatype=XSD.decimal)))
        A.add((node, MLT.n, Literal(int(r["n"]), datatype=XSD.integer)))
        A.add((node, MLT.medianRMSFull30, Literal(round(r["median_rms_full30"], 4), datatype=XSD.decimal)))
        A.add((node, MLT.medianRMSThompson7, Literal(round(r["median_rms_thompson"], 4), datatype=XSD.decimal)))
        A.add((node, MLT.medianRMSPCA7, Literal(round(r["median_rms_pca7"], 4), datatype=XSD.decimal)))
        A.add((node, MLT.medianRMSNearestRelative, Literal(round(r["median_rms_nearest_tip"], 4), datatype=XSD.decimal)))
        A.add((node, MLT.medianRMSConsensus, Literal(round(r["median_rms_consensus"], 4), datatype=XSD.decimal)))
        A.add((node, MLT.coverage95Pct, Literal(round(r["coverage95_pct"], 1), datatype=XSD.decimal)))
    A.add((s, MLT.nRegisteredPredictions, Literal(n_pred, datatype=XSD.integer)))
    A.add((s, MLT.nFossilTests, Literal(n_tests, datatype=XSD.integer)))
    A.add((s, DCTERMS.description, Literal(
        "Full 30-D shape-space phylogenetic prediction beats copying the nearest sampled relative at every horizon to ~50 Myr and collapses to the grand-consensus level beyond it; "
        "the a-priori 7 Thompson modes are outperformed by an equal-dimension PCA basis and are bounded by their representation floor. Predictive ellipsoids are approximately calibrated within 50 Myr. "
        "The companion prediction nanopubs in this series are registered against this error model.")))
    add_common_provenance(P, A.identifier, commit, today, [LANDMARK_SOURCE, TREE_SOURCE])
    add_common_pubinfo(I, base, today, orcid, name)
    return np_id, ds


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sign", action="store_true", help="also write signed.<name>.trig using the ~/.nanopub profile")
    ap.add_argument("--publish", choices=["test", "prod"], help="sign and publish each nanopub to the test or production registry; appends to a manifest")
    ap.add_argument("--only", nargs="*", help="restrict signing/publishing to nanopub ids containing any of these substrings")
    ap.add_argument("--commit", help="git commit to cite as the generating commit (default: HEAD); pass the original commit to reproduce identical trusty URIs")
    ap.add_argument("--out", type=Path, default=OUT_DIR)
    args = ap.parse_args()

    d = json.loads(IN_JSON.read_text())
    commit = args.commit or git_commit()
    today = dt.date.today().isoformat()
    prof = read_profile()
    orcid, name = prof.get("orcid_id"), prof.get("name")
    args.out.mkdir(parents=True, exist_ok=True)

    written = []
    pred_by_fam = {r["target"]: r for r in d["registered_predictions"] if r["kind"] == "bolca50"}
    for rec in d["registered_predictions"]:
        written.append(prediction_nanopub(rec, commit, today, orcid, name))
    for rec in d.get("monte_bolca_published_ratio_test", []):
        written.append(fossil_test_nanopub(rec, pred_by_fam, commit, today, orcid, name))
    n_pred = len(d["registered_predictions"]); n_tests = len(d.get("monte_bolca_published_ratio_test", []))
    written.append(summary_nanopub(d, n_pred, n_tests, commit, today, orcid, name))

    index_lines = ["# Thompson 1917 prediction nanopubs", "", f"Generated {today} from `thompson_phylogenetic_prediction_engine.json` at commit `{commit[:12]}` by `nanopubs/thompson_predictions_to_nanopubs.py`.", "",
                   f"{n_pred} registered predictions, {n_tests} Monte Bolca fossil-test outcomes, 1 validation summary.", "", "| File | Label |", "|---|---|"]
    for np_id, ds in written:
        path = args.out / f"np_thompson_{np_id}.trig"
        ds.serialize(destination=str(path), format="trig")
        assertion_graph = ds.graph(URIRef(BASE + np_id + "/assertion"))
        labels = [str(o) for s_, _, o in assertion_graph.triples((None, RDFS.label, None)) if "/prediction/" in str(s_) or "/fossil-test/" in str(s_) or "/engine/" in str(s_)]
        label = labels[0] if labels else np_id
        index_lines.append(f"| `{path.name}` | {label} |")
    (args.out / "README.md").write_text("\n".join(index_lines) + "\n", encoding="utf-8")
    print(f"wrote {len(written)} nanopubs to {args.out}")

    if args.sign or args.publish:
        import time
        from nanopub import Nanopub
        if not prof:
            print("no ~/.nanopub/profile.yml; cannot sign"); return 1
        conf = nanopub_conf(prof, use_test_server=(args.publish == "test"))
        manifest = args.out / f"published_manifest_{args.publish}.txt" if args.publish else None
        ok = pub_ok = 0
        for np_id, _ in written:
            if args.only and not any(s in np_id for s in args.only):
                continue
            src = args.out / f"np_thompson_{np_id}.trig"
            try:
                npub = Nanopub(rdf=src, conf=conf)
                npub.sign()
                if not npub.has_valid_signature:
                    raise RuntimeError("signature did not verify")
                npub.rdf.serialize(destination=str(args.out / f"signed.np_thompson_{np_id}.trig"), format="trig")
                ok += 1
                if args.publish:
                    npub.publish()
                    pub_ok += 1
                    with manifest.open("a") as fh:
                        fh.write(f"{src.name}|{npub.source_uri}|{dt.datetime.now().isoformat()}|{args.publish}\n")
                    print(f"  published {src.name} -> {npub.source_uri}")
                    time.sleep(0.5)
            except Exception as e:  # noqa: BLE001
                print(f"  failed for {src.name}: {e}")
        print(f"signed {ok}" + (f", published {pub_ok} ({args.publish})" if args.publish else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())

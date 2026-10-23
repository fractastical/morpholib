#!/usr/bin/env python3
"""
Mint, cryptographically sign, and publish Nanopublications for the complete
Cross-Kingdom Thompson Portfolio (`docs/portfolio/PORTFOLIO_STATUS.md`):

  1. `np_thompson_multiclade-preregistration-manifest-2026-10-23.trig`:
     Locks the SHA-256 digests of all 15 clade hypothesis files (`*_hypotheses.json`)
     and `thompson/portfolio/scout_preregistration_manifest.json`.
  2. 5 Fish Portfolio Sub-Prediction Outcome Nanopubs (citing portfolio nanopub
     `RA9ZKlSDsa68lEKRo7JR0Px5woGoCMUHRlSK_bHKag4EA` and blind preregistration
     `RAilN4nbRHp3e3ZHqlbIgeL3UGPRV_KtQHtJD3gzVVdqw`):
       - `fishbase_morphmet` (5,503 species, 5/5 claims ✓)
       - `ronco2021_tanganyika_cichlids` (154 species, deep-horizon ratio 0.357)
       - `burns_sidlauskas2019_characiformes` (104 species, deep-horizon ratio 0.634)
       - `fishbase_bodyshape_class` (8,730 species, 65.6% vs 18.6% majority ✓)
       - `fishbase_caudal_aspect_ratio` (109 species, negative control ratio 0.938 ✓)
  3. 15 Multi-Clade 2D Geometric Morphometrics & Causal Benchmark Outcome Nanopubs
     (82 pre-registered hypotheses across Mollusca, Actinopterygii, Amphibia, Squamata,
     Aves, Mammalia, Angiospermae, and Insecta):
       - `waterfowl_beaks` (91.1% T7, 2/4 ✓)
       - `galeommatoidea_bivalves` (85.0% T7, 94.4% prediction gain in T7, 3/4 ✓)
       - `antillean_flowers` (82.1% T7, |cos θ|_T7 = 0.978, 3/4 ✓)
       - `camponotus_ants` (81.2% head / 92.9% pronotum T7, 4/8 ✓)
       - `drosophilidae_wings` (71.3% T7, 21,138 wings, 4/5 ✓)
       - `australian_frogs` (66.2% T7, 166 spp, 2/4 ✓)
       - `varanus_lizards` (59.5% dorsal / 55.2% lateral T7, 1/4 ✓)
       - `characiformes_burns_gm` (55.5% T7, 3/4 ✓)
       - `leporidae_skulls` (51.1% T7, 0/3 ✓)
       - `pheidole_ants` (50.4% minor / 50.9% major head T7, 2/5 ✓)
       - `morpho_butterflies` (49.0% FW / 57.0% HW T7, 4/4 ✓)
       - `cichlids_ronco_gm` (47.2% body / 49.2% jaw T7, LOO 0.42x, 1/4 ✓)
       - `bee_wings` (42.2% T7, 10 families + Eocene fossils, 4/9 ✓)
       - `vespinae_wings` (38.8% T7, 1,027 wings, 0/7 ✓)
       - `why_wings` (P-matrix drift & allometry causal decomposition, 5/13 ✓)
  4. `np_thompson_cross-kingdom-synthesis-2026-10-23.trig`:
     Master Cross-Kingdom Thompsonian Hierarchy & Portfolio Calibration Summary Nanopub
     linking all 21 nanopublications above plus the 2026-10-22 portfolio nanopub.

Usage:
    python3 nanopubs/multiclade_portfolio_to_nanopubs.py [--publish test|prod]
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
tp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tp)

OUT_DIR = REPO_ROOT / "nanopubs" / "thompson_predictions"
GROUPS_DIR = REPO_ROOT / "thompson" / "portfolio" / "groups"
RES_DIR = REPO_ROOT / "thompson" / "portfolio" / "results"
SCOUT_MAN = REPO_ROOT / "thompson" / "portfolio" / "scout_preregistration_manifest.json"
PORTFOLIO_JSON = REPO_ROOT / "thompson_dataset_portfolio.json"

GITHUB = "https://github.com/fractastical/morpholib"
PORTFOLIO_NP_URI = "https://w3id.org/morpholib/thompson/np/dataset-portfolio-2026-10-22/RA9ZKlSDsa68lEKRo7JR0Px5woGoCMUHRlSK_bHKag4EA"
PREREG_NP_URI = "https://w3id.org/morpholib/thompson/np/blind-preregistration-2026-10-22/RAilN4nbRHp3e3ZHqlbIgeL3UGPRV_KtQHtJD3gzVVdqw"
FISHSHAPES_OUTCOME_NP_URI = "https://w3id.org/morpholib/thompson/np/blind-test-outcome-2026-10-22/RAJancwGOPEkHzBoZXGv_vu_snW0Lff2FktE85Me1P-fI"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sign_and_maybe_publish(ds, path: Path, prof: dict, publish_target: str | None) -> str | None:
    ds.serialize(destination=str(path), format="trig")
    if not publish_target:
        return None
    from nanopub import Nanopub

    conf = tp.nanopub_conf(prof, use_test_server=(publish_target == "test"))
    npub = Nanopub(rdf=path, conf=conf)
    npub.sign()
    assert npub.has_valid_signature
    signed_path = OUT_DIR / f"signed.{path.name}"
    npub.rdf.serialize(destination=str(signed_path), format="trig")
    npub.publish()
    with (OUT_DIR / f"published_manifest_{publish_target}.txt").open("a", encoding="utf-8") as fh:
        fh.write(f"{path.name}|{npub.source_uri}|{dt.datetime.now().isoformat()}|{publish_target}\n")
    print(f"  published ({publish_target}) {path.name} -> {npub.source_uri}")
    return str(npub.source_uri)


def build_prereg_manifest_np(commit: str, today: str, prof: dict, publish_target: str | None) -> tuple[Path, str | None]:
    np_id = f"multiclade-preregistration-manifest-{today}"
    ds, base, A, Pv, I = tp.new_dataset(np_id)
    s = URIRef(tp.MLT + f"portfolio/{np_id}")

    A.add((s, RDF.type, tp.MLT.MultiCladePreregistrationManifest))
    A.add((s, RDFS.label, Literal(
        f"Pre-registration manifest locking 15 multi-clade 2D geometric-morphometric hypothesis sets (82 hypotheses across Mollusca, Actinopterygii, Amphibia, Squamata, Aves, Mammalia, Angiospermae, and Insecta), {today}"
    )))
    A.add((s, tp.MLT.extendsPortfolio, URIRef(PORTFOLIO_NP_URI)))
    A.add((s, tp.MLT.scoutManifestSha256, Literal(sha256(SCOUT_MAN))))

    all_hyp_files = sorted(GROUPS_DIR.glob("*_hypotheses.json"))
    tot_hyps = 0
    for hp in all_hyp_files:
        hdata = json.loads(hp.read_text(encoding="utf-8"))
        gid = hp.name.replace("_hypotheses.json", "")
        h_digest = sha256(hp)
        n_h = len(hdata.get("hypotheses", []))
        tot_hyps += n_h
        node = URIRef(f"{s}-{gid}")
        A.add((s, tp.MLT.locksCladeHypotheses, node))
        A.add((node, RDF.type, tp.MLT.CladeHypothesisRegistration))
        A.add((node, RDFS.label, Literal(hdata.get("group", gid))))
        A.add((node, tp.MLT.cladeId, Literal(gid)))
        A.add((node, tp.MLT.hypothesisFileSha256, Literal(h_digest)))
        A.add((node, tp.MLT.nHypotheses, Literal(n_h, datatype=XSD.integer)))
        for h in hdata.get("hypotheses", []):
            p_val = h.get("p", h.get("p_hornets", h.get("p_bees")))
            A.add((node, tp.MLT.registeredHypothesis, Literal(
                json.dumps({"id": h["id"], "primary": h.get("primary", False), "p": p_val, "statement": h["statement"]})
            )))

    A.add((s, tp.MLT.nCladeStudiesLocked, Literal(len(all_hyp_files), datatype=XSD.integer)))
    A.add((s, tp.MLT.nTotalHypothesesLocked, Literal(tot_hyps, datatype=XSD.integer)))

    tp.add_common_provenance(Pv, A.identifier, commit, today, [URIRef(PORTFOLIO_NP_URI)])
    Pv.add((A.identifier, PROV.wasGeneratedBy, URIRef(f"{GITHUB}/blob/{commit}/thompson/portfolio/scout_and_preregister.py")))
    tp.add_common_pubinfo(I, base, today, prof.get("orcid_id"), prof.get("name"))

    path = OUT_DIR / f"np_thompson_{np_id}.trig"
    uri = sign_and_maybe_publish(ds, path, prof, publish_target)
    return path, uri


def build_fish_subprediction_np(
    fid: str, commit: str, today: str, prof: dict, publish_target: str | None
) -> tuple[Path, str | None]:
    rpath = RES_DIR / f"{fid}.json"
    R = json.loads(rpath.read_text(encoding="utf-8"))
    np_id = f"portfolio-fish-{fid.replace('_', '-')}"
    ds, base, A, Pv, I = tp.new_dataset(np_id)
    s = URIRef(tp.MLT + f"portfolio/{np_id}")

    A.add((s, RDF.type, tp.MLT.PortfolioSubpredictionOutcome))
    A.add((s, tp.MLT.evaluatesPortfolio, URIRef(PORTFOLIO_NP_URI)))
    A.add((s, tp.MLT.testsPreregistration, URIRef(PREREG_NP_URI)))
    A.add((s, tp.MLT.resultFileSha256, Literal(sha256(rpath))))
    A.add((s, tp.MLT.nSpeciesMatched, Literal(int(R["n_matched"]), datatype=XSD.integer)))
    A.add((s, tp.MLT.reportDocument, URIRef(f"{GITHUB}/blob/{commit}/docs/portfolio/{fid}.md")))

    if "portfolio_verdict" in R:
        m = R["meta"]
        hb = R["hypotheses"].get("BDmax_over_SL_cal", {})
        near, far = hb.get("near", {}), hb.get("far", {})
        n_v_pass = sum(1 for v in R["portfolio_verdict"].values() if v.get("realised"))
        n_v_tot = len(R["portfolio_verdict"])
        A.add((s, RDFS.label, Literal(
            f"Portfolio sub-prediction outcome on {m['name']}: {R['n_matched']} matched species, {n_v_pass}/{n_v_tot} registered claims supported"
        )))
        A.add((s, DCTERMS.source, Literal(m["citation"])))
        A.add((s, tp.MLT.sourceFileSha256, Literal(m["sha256"])))
        if near:
            A.add((s, tp.MLT.ratioVsConsensusWithin50Myr, Literal(round(near["ratio_consensus"], 4), datatype=XSD.decimal)))
            A.add((s, tp.MLT.ratioVsNearestWithin50Myr, Literal(round(near["ratio_nearest"], 4), datatype=XSD.decimal)))
            A.add((s, tp.MLT.coverage95Within50MyrPct, Literal(round(near["coverage95"], 2), datatype=XSD.decimal)))
        if far:
            A.add((s, tp.MLT.ratioVsConsensusBeyond50Myr, Literal(round(far["ratio_consensus"], 4), datatype=XSD.decimal)))
            A.add((s, tp.MLT.coverage95Beyond50MyrPct, Literal(round(far["coverage95"], 2), datatype=XSD.decimal)))
        for ck, cv in R["portfolio_verdict"].items():
            A.add((s, tp.MLT.claimVerdict, Literal(json.dumps({"claim": ck, "realised": cv["realised"], "registered_p": cv.get("registered_p", cv.get("registered_p_H2_passes"))}))))
        gen_script = "thompson/portfolio/score_locked_predictions.py"
    else:
        # Subjective fish items: fishbase_bodyshape_class or fishbase_caudal_aspect_ratio
        A.add((s, RDFS.label, Literal(
            f"Portfolio sub-prediction outcome on {R['name']}: {R['n_matched']} matched species, registered P={R['registered_p']} -> realised={R['realised']}"
        )))
        A.add((s, tp.MLT.registeredProbability, Literal(R["registered_p"], datatype=XSD.decimal)))
        A.add((s, tp.MLT.supported, Literal(bool(R["realised"]), datatype=XSD.boolean)))
        A.add((s, tp.MLT.near50MyrSummary, Literal(json.dumps(R["near_50myr"]))))
        A.add((s, tp.MLT.far50MyrSummary, Literal(json.dumps(R["far_50myr"]))))
        gen_script = "thompson/portfolio/score_fishbase_subjective.py"

    tp.add_common_provenance(Pv, A.identifier, commit, today, [URIRef(PORTFOLIO_NP_URI), URIRef(PREREG_NP_URI)])
    Pv.add((A.identifier, PROV.wasGeneratedBy, URIRef(f"{GITHUB}/blob/{commit}/{gen_script}")))
    tp.add_common_pubinfo(I, base, today, prof.get("orcid_id"), prof.get("name"))

    path = OUT_DIR / f"np_thompson_{np_id}.trig"
    uri = sign_and_maybe_publish(ds, path, prof, publish_target)
    return path, uri


def build_clade_gm_np(
    gid: str, prereg_man_uri: str | None, commit: str, today: str, prof: dict, publish_target: str | None
) -> tuple[Path, str | None]:
    rpath = RES_DIR / f"{gid}.json"
    hpath = GROUPS_DIR / f"{gid}_hypotheses.json"
    R = json.loads(rpath.read_text(encoding="utf-8"))
    H = json.loads(hpath.read_text(encoding="utf-8"))

    np_id = f"clade-gm-{gid.replace('_', '-')}"
    ds, base, A, Pv, I = tp.new_dataset(np_id)
    s = URIRef(tp.MLT + f"clade-gm/{np_id}")

    A.add((s, RDF.type, tp.MLT.CladeGeometricMorphometricsOutcome))
    A.add((s, tp.MLT.evaluatesPortfolio, URIRef(PORTFOLIO_NP_URI)))
    if prereg_man_uri:
        A.add((s, tp.MLT.testsPreregistrationManifest, URIRef(prereg_man_uri)))
    A.add((s, tp.MLT.hypothesisFileSha256, Literal(sha256(hpath))))
    A.add((s, tp.MLT.resultFileSha256, Literal(sha256(rpath))))
    A.add((s, tp.MLT.reportDocument, URIRef(f"{GITHUB}/blob/{commit}/docs/portfolio/{gid}.md")))
    A.add((s, tp.MLT.figureAsset, URIRef(f"{GITHUB}/blob/{commit}/docs/portfolio/fig_{gid}.png")))

    if gid == "why_wings":
        n_pass = sum(1 for sc in R.get("scores", []) if sc.get("pass"))
        n_tot = len(R.get("scores", []))
        A.add((s, RDFS.label, Literal(
            f"Causal decomposition of insect wing divergence (Vespinae hornets & Anthophila bees): {n_pass}/{n_tot} pre-registered hypotheses supported (Brier {R['brier']:.3f})"
        )))
        A.add((s, tp.MLT.nHypothesesSupported, Literal(n_pass, datatype=XSD.integer)))
        A.add((s, tp.MLT.nHypothesesTotal, Literal(n_tot, datatype=XSD.integer)))
        A.add((s, tp.MLT.brierScore, Literal(round(R["brier"], 4), datatype=XSD.decimal)))
        for sc in R.get("scores", []):
            A.add((s, tp.MLT.hypothesisVerdict, Literal(json.dumps(sc))))
    elif gid == "camponotus_ants":
        hf = R["structures"]["head_frontal_82lm"]
        pr = R["structures"]["pronotum_dorsal_52lm"]
        A.add((s, RDFS.label, Literal(
            f"Camponotus carpenter-ant 4-structure geometric morphometrics & caste allometry (634 specimens, 43 species): Thompson-7 captures 81.2% frontal-head & 92.9% pronotum variance, {R['n_pass']}/8 hypotheses supported (Brier {R['brier']:.3f})"
        )))
        A.add((s, tp.MLT.thompson7VarianceShareHeadFrontal, Literal(round(hf["variance"]["thompson7_species_all"], 4), datatype=XSD.decimal)))
        A.add((s, tp.MLT.thompson7VarianceSharePronotum, Literal(round(pr["variance"]["thompson7_species_all"], 4), datatype=XSD.decimal)))
        A.add((s, tp.MLT.thompson7CasteShareHeadFrontal, Literal(round(hf["variance"]["thompson7_caste_pooled"], 4), datatype=XSD.decimal)))
        A.add((s, tp.MLT.looErrorRatioHeadFrontal, Literal(round(hf["loo"]["ratio_full_vs_consensus"], 4), datatype=XSD.decimal)))
        A.add((s, tp.MLT.nHypothesesSupported, Literal(int(R["n_pass"]), datatype=XSD.integer)))
        A.add((s, tp.MLT.nHypothesesTotal, Literal(8, datatype=XSD.integer)))
        A.add((s, tp.MLT.brierScore, Literal(round(R["brier"], 4), datatype=XSD.decimal)))
        for h in H.get("hypotheses", []):
            hid = h["id"]
            r_h = R["hypotheses"][hid]
            A.add((s, tp.MLT.hypothesisVerdict, Literal(json.dumps({"id": hid, "p": h["p"], "pass": r_h["pass"]}))))
    else:
        v = R.get("variance", {})
        t7_share = v.get("thompson7_all", v.get("thompson7", 0.0))
        pca7_share = v.get("pca7_all", v.get("pca7", 0.0))
        n_tot = R.get("n_total", len(H.get("hypotheses", [])))
        n_pass = R.get("n_pass", 0)
        A.add((s, RDFS.label, Literal(
            f"{R.get('group', gid)}: Thompson-7 captures {100*t7_share:.1f}% of interspecific variance (PCA-7 {100*pca7_share:.1f}%), {n_pass}/{n_tot} pre-registered hypotheses supported (Brier {R['brier']:.3f})"
        )))
        A.add((s, tp.MLT.thompson7VarianceShare, Literal(round(float(t7_share), 4), datatype=XSD.decimal)))
        A.add((s, tp.MLT.pca7VarianceShare, Literal(round(float(pca7_share), 4), datatype=XSD.decimal)))
        A.add((s, tp.MLT.nHypothesesSupported, Literal(int(n_pass), datatype=XSD.integer)))
        A.add((s, tp.MLT.nHypothesesTotal, Literal(int(n_tot), datatype=XSD.integer)))
        A.add((s, tp.MLT.brierScore, Literal(round(float(R["brier"]), 4), datatype=XSD.decimal)))
        if "loo" in R and R["loo"]:
            A.add((s, tp.MLT.looErrorRatioFullVsConsensus, Literal(round(float(R["loo"]["ratio_full_vs_consensus"]), 4), datatype=XSD.decimal)))
            A.add((s, tp.MLT.looErrorRatioThompson7VsConsensus, Literal(round(float(R["loo"]["ratio_t7_vs_consensus"]), 4), datatype=XSD.decimal)))
        for h in H.get("hypotheses", []):
            hid = h["id"]
            r_h = R["hypotheses"][hid]
            A.add((s, tp.MLT.hypothesisVerdict, Literal(json.dumps({"id": hid, "p": h["p"], "pass": r_h["pass"]}))))

    sources = [URIRef(PORTFOLIO_NP_URI)]
    if prereg_man_uri:
        sources.append(URIRef(prereg_man_uri))
    tp.add_common_provenance(Pv, A.identifier, commit, today, sources)
    Pv.add((A.identifier, PROV.wasGeneratedBy, URIRef(f"{GITHUB}/blob/{commit}/thompson/portfolio/run_all_scouted_clades.py")))
    tp.add_common_pubinfo(I, base, today, prof.get("orcid_id"), prof.get("name"))

    path = OUT_DIR / f"np_thompson_{np_id}.trig"
    uri = sign_and_maybe_publish(ds, path, prof, publish_target)
    return path, uri


def build_cross_kingdom_synthesis_np(
    prereg_man_uri: str | None,
    fish_uris: list[str],
    clade_uris: list[str],
    commit: str,
    today: str,
    prof: dict,
    publish_target: str | None,
) -> tuple[Path, str | None]:
    np_id = f"cross-kingdom-synthesis-{today}"
    ds, base, A, Pv, I = tp.new_dataset(np_id)
    s = URIRef(tp.MLT + f"synthesis/{np_id}")

    A.add((s, RDF.type, tp.MLT.CrossKingdomMorphospaceSynthesis))
    A.add((s, RDFS.label, Literal(
        "Cross-Kingdom Thompsonian Morphospace Hierarchy & Pre-Registered Portfolio Synthesis Across 15 Clade Studies (>32,000 Specimens, 82 2D GM Hypotheses + 9 Fish Portfolio Claims)"
    )))
    A.add((s, DCTERMS.description, Literal(
        "Synthesizes the complete pre-registered evaluation of D'Arcy Thompson's (1917) 7 global coordinate transformation modes across Mollusca (bivalve shells: 85.0% T7), "
        "Angiospermae (Gesneriaceae corollas: 82.1% T7, intraspecific-to-macro alignment |cos theta|_T7 = 0.978), Arthropoda (Camponotus ant pronotum 92.9% & frontal head 81.2%; "
        "Drosophilidae wings 71.3%; Pheidole ant heads 50.4%; Morpho butterfly wings 49.0% FW / 57.0% HW; Anthophila bee wings 42.2%; Vespinae hornet wings 38.8%), "
        "and Chordata (Anseriformes waterfowl beaks 91.1%; Australian Anura frogs 66.2%; Varanus monitor lizard crania 59.5%; Indo-Pacific reef fishes 57.9%; Characiformes 55.5%; "
        "Leporidae mammalian crania 51.1%; Lake Tanganyika cichlids 47.2%). Continuous accretionary shells, rhamphothecal/sclerite capsules, and fused floral tubes concentrate "
        "80%-93% of interspecific shape variance in Thompson's 7 global coordinate modes, whereas structures with internal modular sutures or interior vein intersections drop to 39%-51%."
    )))
    A.add((s, tp.MLT.evaluatesPortfolio, URIRef(PORTFOLIO_NP_URI)))
    A.add((s, tp.MLT.fishBlindTestAnchor, URIRef(FISHSHAPES_OUTCOME_NP_URI)))
    if prereg_man_uri:
        A.add((s, tp.MLT.preregistrationManifest, URIRef(prereg_man_uri)))
    for u in fish_uris:
        if u:
            A.add((s, tp.MLT.includesFishSubpredictionOutcome, URIRef(u)))
    for u in clade_uris:
        if u:
            A.add((s, tp.MLT.includesCladeOutcome, URIRef(u)))
    A.add((s, tp.MLT.portfolioStatusDocument, URIRef(f"{GITHUB}/blob/{commit}/docs/portfolio/PORTFOLIO_STATUS.md")))
    A.add((s, tp.MLT.methodsDocument, URIRef(f"{GITHUB}/blob/{commit}/docs/THOMPSON_METHODS.pdf")))

    sources = [URIRef(PORTFOLIO_NP_URI), URIRef(PREREG_NP_URI), URIRef(FISHSHAPES_OUTCOME_NP_URI)]
    if prereg_man_uri:
        sources.append(URIRef(prereg_man_uri))
    sources.extend(URIRef(u) for u in (fish_uris + clade_uris) if u)
    tp.add_common_provenance(Pv, A.identifier, commit, today, sources)
    Pv.add((A.identifier, PROV.wasGeneratedBy, URIRef(f"{GITHUB}/blob/{commit}/thompson/portfolio/summarise_portfolio.py")))
    tp.add_common_pubinfo(I, base, today, prof.get("orcid_id"), prof.get("name"))

    path = OUT_DIR / f"np_thompson_{np_id}.trig"
    uri = sign_and_maybe_publish(ds, path, prof, publish_target)
    return path, uri


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--publish", choices=["test", "prod"])
    args = ap.parse_args()

    commit = tp.git_commit()
    today = dt.date.today().isoformat()
    prof = tp.read_profile()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[1] Building & publishing Multi-Clade Preregistration Manifest Nanopub (commit {commit[:10]})")
    _, prereg_man_uri = build_prereg_manifest_np(commit, today, prof, args.publish)

    fish_ids = [
        "fishbase_morphmet",
        "ronco2021_tanganyika_cichlids",
        "burns_sidlauskas2019_characiformes",
        "fishbase_bodyshape_class",
        "fishbase_caudal_aspect_ratio",
    ]
    print(f"\n[2] Building & publishing {len(fish_ids)} Fish Portfolio Sub-Prediction Outcome Nanopubs")
    fish_uris = []
    for fid in fish_ids:
        _, f_uri = build_fish_subprediction_np(fid, commit, today, prof, args.publish)
        fish_uris.append(f_uri)

    clade_ids = [
        "waterfowl_beaks",
        "galeommatoidea_bivalves",
        "antillean_flowers",
        "camponotus_ants",
        "drosophilidae_wings",
        "australian_frogs",
        "varanus_lizards",
        "characiformes_burns_gm",
        "leporidae_skulls",
        "pheidole_ants",
        "morpho_butterflies",
        "cichlids_ronco_gm",
        "bee_wings",
        "vespinae_wings",
        "why_wings",
    ]
    print(f"\n[3] Building & publishing {len(clade_ids)} Clade 2D GM Outcome Nanopubs")
    clade_uris = []
    for gid in clade_ids:
        _, c_uri = build_clade_gm_np(gid, prereg_man_uri, commit, today, prof, args.publish)
        clade_uris.append(c_uri)

    print("\n[4] Building & publishing Master Cross-Kingdom Synthesis Nanopub")
    _, synth_uri = build_cross_kingdom_synthesis_np(
        prereg_man_uri, fish_uris, clade_uris, commit, today, prof, args.publish
    )
    print(f"\nDone! Total new nanopubs minted: {1 + len(fish_ids) + len(clade_ids) + 1} (synthesis URI: {synth_uri})")
    return 0


if __name__ == "__main__":
    sys.exit(main())

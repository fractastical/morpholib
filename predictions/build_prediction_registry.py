#!/usr/bin/env python3
"""
Record CLEAR predictions for the 2-3 best-supported organisms in this repo.

Organisms were chosen by evidence depth, not enthusiasm:
  1. Drosophila melanogaster -- 610 dataset references, 600 with phenomenon
     sizes, catalog proteins dpp/wg/hh with UniProt states, FlyBase alleles.
  2. Danio rerio -- 110 references, catalog bmp2b/chd/shha/fgf8a/her1 plus the
     protein-state pilot sidecars, ZFIN mutant phenotypes.
  3. Xenopus tropicalis -- only 10 references, but catalog bmp4/nog/shh and
     the classic BMP/Noggin embryology test literature (Xenbase).

Discipline (same as thompson_blind_predictions_registered.csv):
  * Every prediction is REGISTERED BEFORE SCORING: status starts as
    "registered_untested", with an explicit falsifier and test source.
  * TWO predictions per row, registered separately: (1) the OUTCOME prediction
    (what will happen) and (2) the RESOLVABILITY prediction (whether current
    datasets -- or current + named public sources -- can decide it at all),
    with a confidence, an expected verdict, and a resolution route. Meta-level
    resolvability predictions (META-RES-*) predict the overall solve rate.
  * Predictions are based on poson_law invariants and UniProt protein states.
  * They are NOT based on alpha_scale / empirical_series slopes: in the 3,000
    core datasets the slope is ~1.57 * alpha_scale for every law (corr ~0.99),
    i.e. a shared template, so alpha-derived "predictions" would be circular.
  * Negative predictions (a dataset's assigned law should FAIL) are first-class:
    they falsify mechanism assignments, which is where these datasets are weakest.

Outputs: predictions/prediction_registry.json and .csv (identical content).

Usage:
  python3 predictions/build_prediction_registry.py
  python3 predictions/build_prediction_registry.py --self-test
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
REGISTERED_ON = "2026-10-26"  # registration date; never rewritten on regeneration

ORGANISMS = {
    "Danio rerio": {"key": "danio_rerio", "refs": 110, "test_atlas": "ZFIN (https://zfin.org)"},
    "Drosophila melanogaster": {"key": "drosophila_melanogaster", "refs": 610, "test_atlas": "FlyBase (https://flybase.org)"},
    "Xenopus tropicalis": {"key": "xenopus_tropicalis", "refs": 10, "test_atlas": "Xenbase (https://www.xenbase.org)"},
}

# Each prediction: statement, quantitative expectation (a number/range where the
# basis supports one, otherwise an explicit direction), basis in existing data,
# falsifier, and where it can be scored. Status is set by the builder.
PREDICTIONS = [
    # ---------------------------------------------------------------- Danio
    {
        "prediction_id": "ZF-TUR-01", "organism": "Danio rerio", "mechanism": "turing_wavelength",
        "datasets": ["datasets/danio_rerio_turing_wavelength"],
        "prediction": "BMP patterning in the zebrafish gastrula is NOT free differential diffusion of BMP vs Chordin: Chordin's effective mobility in the embryo will exceed the free-diffusion size prior, because transport is by Chordin shuttling + tolloid release.",
        "quantitative_expectation": "Free-diffusion prior from protein states: D_chordin/D_mature-BMP2b = 0.51 (921 aa vs 122 aa, D~MW^-1/3), which violates the Turing requirement D_inhibitor/D_activator > 1. Prediction: measured effective BMP-gradient formation will behave as if the ratio is >= 1 (shuttling), never as 0.51 free diffusion.",
        "basis": ["datasets/danio_rerio_turing_wavelength/protein_states.json: activator_inhibitor_size_check", "UniProt O57472 (chordin, tolloid-cleavage PTM comment) / O93369 (bmp2b)"],
        "falsifier": "Direct mobility measurements (e.g. FRAP of tagged BMP/Chordin in gastrula) showing both species freely diffusing with D ratio ~0.5 AND normal dorsoventral patterning.",
        "test_source": "ZFIN chordin (dino) / tolloid (minifin) phenotypes; published pSmad1/5 gradient measurements in zebrafish gastrula",
        "class": "protein_state_biological",
    },
    {
        "prediction_id": "ZF-TUR-02", "organism": "Danio rerio", "mechanism": "turing_wavelength",
        "datasets": ["datasets/danio_rerio_turing_wavelength"],
        "prediction": "Blocking tolloid cleavage of Chordin (more intact Chordin, less BMP release) contracts the ventral pSmad1/5 signalling domain and dorsalizes the embryo.",
        "quantitative_expectation": "Direction: pSmad domain contracts, dorsal markers expand (dorsalized phenotype, as in tolloid/minifin loss). No magnitude claimed -- cleavage kinetics are literature parameters, not in UniProt.",
        "basis": ["UniProt O57472 PTM comment: 'Cleaved by tolloid proteases; cleavage participates in dorsoventral patterning'", "datasets/danio_rerio_turing_wavelength/protein_states.json"],
        "falsifier": "Tolloid loss/cleavage-resistant Chordin producing ventralization or an expanded pSmad domain.",
        "test_source": "ZFIN tolloid/minifin mutant phenotype; chordin cleavage literature",
        "class": "protein_state_biological",
    },
    {
        "prediction_id": "ZF-HH-01", "organism": "Danio rerio", "mechanism": "morphogen_condensation",
        "datasets": ["datasets/danio_rerio_turing_wavelength"],
        "prediction": "ShhA signalling is short-range and membrane-associated despite a small signalling product, because ShhNp is dually lipidated; disrupting lipidation flattens the gradient (lower peak near source, longer distal tail).",
        "quantitative_expectation": "Size prior alone gives D_rel = 0.89 for the 174 aa N-product (near BMP/FGF mobility). Prediction: measured effective range will be markedly SHORTER than that prior in wild type, and lipidation loss will expand the ptc-expression domain distally while reducing its peak level near the source.",
        "basis": ["UniProt Q92008: chains 24-197 ShhN product; lipidation N-palmitoyl Cys24 + cholesteryl Gly197", "datasets/danio_rerio_turing_wavelength/protein_states.json: shha_tethering"],
        "falsifier": "Lipidation-deficient ShhA producing an unchanged or steeper/higher-peak gradient in zebrafish neural tube/otic assays.",
        "test_source": "ZFIN shha (syu) and HHAT-inhibition literature; ptc1/ptc2 expression-domain measurements",
        "class": "protein_state_biological",
    },
    {
        "prediction_id": "ZF-CLK-01", "organism": "Danio rerio", "mechanism": "clock_wavefront",
        "datasets": ["datasets/danio_rerio_clock_wavefront"],
        "prediction": "Somite length scales in direct proportion to segmentation-clock period at fixed wavefront speed (S = u * T_clock), including for period changes caused by altered Her1 degradation.",
        "quantitative_expectation": "S multiplier = T multiplier: period -20%/-10%/+10%/+20% -> somite length x0.80/x0.90/x1.10/x1.20 (tolerance: within 5 percentage points after correcting for embryo size/growth). Her1's UniProt state (Q90463, 328 aa nuclear bHLH) gives no half-life, so the period change itself must be measured, not derived.",
        "basis": ["datasets/danio_rerio_clock_wavefront poson_law: Somite Length S = u*T_clock", "datasets/danio_rerio_clock_wavefront/protein_states.json: worked_sensitivity"],
        "falsifier": "A verified clock-period change (live her1 reporter) with wavefront speed controlled, producing somite-length change of opposite sign or <50% of the proportional expectation.",
        "test_source": "Zebrafish somitogenesis literature: her1/her7 perturbations and temperature/period manipulations with somite-length readouts",
        "class": "law_invariant_biological",
    },
    {
        "prediction_id": "ZF-VTX-01", "organism": "Danio rerio", "mechanism": "vertex_packing",
        "datasets": ["datasets/danio_rerio_vertex_packing"],
        "prediction": "Zebrafish epithelial tissue assigned the vertex-packing law undergoes its jamming/unjamming transition at cell shape index p0* = 3.813.",
        "quantitative_expectation": "Transition in the band 3.76-3.87 (centred 3.813); fluid-like behaviour above, solid-like below.",
        "basis": ["datasets/danio_rerio_vertex_packing poson_law: Shape Index p_0* = 3.813"],
        "falsifier": "Measured transition in a zebrafish epithelium (e.g. enveloping layer during epiboly) outside 3.7-3.9, or no transition in shape index at all.",
        "test_source": "Published zebrafish epiboly/EVL cell-shape measurements; vertex-model literature value 3.813",
        "class": "law_invariant_biological",
    },
    # ------------------------------------------------------------ Drosophila
    {
        "prediction_id": "DM-WG-01", "organism": "Drosophila melanogaster", "mechanism": "morphogen_condensation",
        "datasets": ["datasets/drosophila_melanogaster_morphogen_condensation"],
        "prediction": "Wingless spread requires Porcupine O-palmitoleoylation at Ser239: blocking it (porcupine loss or Wg Ser239 substitution) collapses Wg protein distribution and its short-range target domain to the producing cells; Notum depalmitoleoylation does the same.",
        "quantitative_expectation": "Direction: extracellular Wg gradient and Senseless-positive range contract sharply (toward ~1 cell row from source) versus wild type; no absolute D claimed.",
        "basis": ["UniProt P09615 (wg): lipidation O-palmitoleoyl Ser239 by PORCN; PTM comment on Notum depalmitoleoylation", "sequences/sequence_links.json drosophila_melanogaster wg protein_state"],
        "falsifier": "Porcupine/Wg-lipidation mutants with a wild-type-width Wg gradient and target domain in wing disc.",
        "test_source": "FlyBase porcupine / wingless alleles; wing-disc Wg gradient literature",
        "class": "protein_state_biological",
    },
    {
        "prediction_id": "DM-HH-01", "organism": "Drosophila melanogaster", "mechanism": "morphogen_condensation",
        "datasets": ["datasets/drosophila_melanogaster_morphogen_condensation"],
        "prediction": "The Hh signalling species is the 173 aa dually-lipidated N-product, not the 471 aa precursor: cholesterol-deficient HhN spreads farther from the A/P boundary but induces a lower peak response (ptc) near the source.",
        "quantitative_expectation": "Direction: distal expansion + reduced near-source peak. Size prior for the N-product vs precursor: using the precursor would understate mobility by factor (173/471)^(1/3) = 0.72 in D, before tethering is even considered.",
        "basis": ["UniProt Q02936 (hh): chains 85-257 N-product; lipidation N-palmitoyl Cys85 + cholesteryl Gly257; autocatalytic-cleavage PTM comment"],
        "falsifier": "Cholesterol-deficient HhN showing a shorter range or higher near-source peak than wild type in wing disc.",
        "test_source": "FlyBase hedgehog alleles; wing-disc Hh gradient literature (cholesterol-modified vs unmodified HhN)",
        "class": "protein_state_biological",
    },
    {
        "prediction_id": "DM-DPP-01", "organism": "Drosophila melanogaster", "mechanism": "turing_wavelength",
        "datasets": ["datasets/drosophila_melanogaster_turing_wavelength"],
        "prediction": "The diffusible Dpp species in wing disc is the annotated 132 aa mature chain (residues 457-588), so gradient models using the 588 aa precursor misestimate Dpp mobility.",
        "quantitative_expectation": "D_precursor/D_mature = (132/588)^(1/3) = 0.61 by size prior: precursor-based models understate D by ~39% and any sqrt(D/k) length scale by ~22% (factor 0.78). Directional prior only.",
        "basis": ["UniProt P07713 (dpp): chain 457-588, 132 aa; signal peptide 1-23", "sequences/sequence_links.json drosophila_melanogaster dpp protein_state"],
        "falsifier": "Evidence that uncleaved 588 aa Dpp precursor is the extracellular signalling species in wing disc.",
        "test_source": "FlyBase decapentaplegic; Dpp processing/gradient literature",
        "class": "protein_state_model",
    },
    {
        "prediction_id": "DM-CLK-NEG-01", "organism": "Drosophila melanogaster", "mechanism": "clock_wavefront",
        "datasets": ["datasets/drosophila_melanogaster_clock_wavefront"],
        "prediction": "NEGATIVE: the clock-wavefront law assigned to Drosophila will FAIL for blastoderm segmentation -- Drosophila is a long-germ embryo with no travelling Hes/her oscillation wavefront; pair-rule stripes are specified near-simultaneously, not laid down sequentially by S = u*T_clock.",
        "quantitative_expectation": "No periodic her/hes wavefront sweeping the blastoderm will be found; stripe formation timing will not fit a constant-u wavefront with a single clock period.",
        "basis": ["datasets/drosophila_melanogaster_clock_wavefront assigns 'Somite Length S = u*T_clock' / 'Phase/Delay Oscillators' to Drosophila", "Established long-germ development of Drosophila (test literature)"],
        "falsifier": "Demonstration of a segmentation-clock oscillation wavefront that sequentially specifies Drosophila blastoderm stripes.",
        "test_source": "FlyBase / Drosophila segmentation literature (pair-rule gene dynamics, live imaging)",
        "class": "negative_mechanism_assignment",
    },
    {
        "prediction_id": "DM-MUR-NEG-01", "organism": "Drosophila melanogaster", "mechanism": "branching_tree",
        "datasets": ["datasets/drosophila_melanogaster_branching_tree"],
        "prediction": "NEGATIVE / discriminating: Drosophila tracheal branching, assigned Murray's law, will not obey the fluid-optimal exponent: parent/daughter diameters will deviate systematically from r0^3 = r1^3 + r2^3 because tracheal branching is FGF(Branchless)-guided cell migration, not flow-optimised plumbing.",
        "quantitative_expectation": "Fitted Murray exponent differs from 3.0 (outside 2.8-3.2) or shows no consistent exponent across branch orders.",
        "basis": ["datasets/drosophila_melanogaster_branching_tree poson_law: Murray Law r_0^3 = r_1^3 + r_2^3"],
        "falsifier": "Tracheal diameter measurements across branch orders fitting exponent 3.0 within 2.8-3.2 with small residuals.",
        "test_source": "Published Drosophila tracheal morphometry; Branchless/FGF guidance literature",
        "class": "negative_mechanism_assignment",
    },
    # --------------------------------------------------------------- Xenopus
    {
        "prediction_id": "XT-BMP-01", "organism": "Xenopus tropicalis", "mechanism": "turing_wavelength",
        "datasets": ["datasets/xenopus_tropicalis_turing_wavelength"],
        "prediction": "Noggin (196 aa secreted chain) patterns Xenopus dorsoventral axis by binding BMP4, not by diffusing as a Turing inhibitor: ventral Noggin misexpression dorsalizes (induces secondary dorsal structures/axis), and Noggin loss ventralizes.",
        "quantitative_expectation": "Direction only: gain -> dorsalization/secondary axis; loss -> ventralization. No D-ratio Turing claim is made (same lesson as zebrafish Chordin: D_noggin by size, (196 aa), would be fast, but binding/sequestration, not differential diffusion, is the mechanism).",
        "basis": ["UniProt F6RNY4 (nog): signal 1-26, chain 27-222 Noggin, secreted", "sequences/sequence_links.json xenopus_tropicalis nog protein_state"],
        "falsifier": "Noggin misexpression with no dorsalizing effect, or Noggin loss with dorsalization, in Xenopus embryos.",
        "test_source": "Xenbase noggin; classic Spemann-organizer / Noggin injection literature",
        "class": "protein_state_biological",
    },
    {
        "prediction_id": "XT-SHH-01", "organism": "Xenopus tropicalis", "mechanism": "morphogen_condensation",
        "datasets": ["datasets/xenopus_tropicalis_morphogen_condensation"],
        "prediction": "Xenopus Shh is dually lipidated like its zebrafish and fly orthologs, even though the current UniProt entry (A0A803JKD3, unreviewed) annotates no lipidation: the N-product will prove to carry N-palmitate + C-cholesterol, with the same short-range tethered-gradient behaviour.",
        "quantitative_expectation": "Conservation prediction: lipidation sites equivalent to zebrafish Cys24/Gly197 (shha Q92008) and fly Cys85/Gly257 (hh Q02936) will be confirmed in a reviewed Xenopus entry or experimentally; range behaviour matches ZF-HH-01 direction.",
        "basis": ["sequences/sequence_links.json xenopus_tropicalis shh protein_state (416 aa, no lipidation annotated)", "Ortholog states: danio_rerio shha / drosophila_melanogaster hh protein_state"],
        "falsifier": "Reviewed Xenopus tropicalis Shh annotation or experiment showing an unlipidated, freely diffusing N-product.",
        "test_source": "Xenbase shh; UniProt reviewed-entry updates; hedgehog lipidation literature",
        "class": "protein_state_biological",
    },
    {
        "prediction_id": "XT-DQ-01", "organism": "Xenopus tropicalis", "mechanism": "turing_wavelength",
        "datasets": ["datasets/xenopus_tropicalis_turing_wavelength"],
        "prediction": "DATA-QUALITY: the UniProt accession currently linked for Xenopus bmp4 (Q28FI5, 69 aa) is a fragment, not the BMP4 precursor: it will not match the 469 aa Ensembl protein, and any BMP4 protein-state conclusion drawn from it is invalid until re-resolved against a full-length reviewed entry.",
        "quantitative_expectation": "Re-resolution yields a ~400+ aa BMP4 precursor with a C-terminal TGF-beta mature domain of ~110-130 aa (ortholog pattern: zebrafish bmp2b 411 aa/122 aa domain, fly dpp 588 aa/132 aa chain).",
        "basis": ["sequences/sequence_links.json xenopus_tropicalis bmp4: ensembl protein_length_aa=469 vs protein_state.precursor_length_aa=69, chain 24-69"],
        "falsifier": "Q28FI5 shown to be the complete, reviewed Xenopus tropicalis BMP4 precursor.",
        "test_source": "UniProt Q28FI5 entry inspection; Ensembl ENSXETG00000008620",
        "class": "data_quality",
    },
    {
        "prediction_id": "XT-CLK-01", "organism": "Xenopus tropicalis", "mechanism": "clock_wavefront",
        "datasets": ["datasets/xenopus_tropicalis_clock_wavefront"],
        "prediction": "Xenopus tropicalis somitogenesis obeys the same proportional clock-wavefront relation as zebrafish: somite length changes in proportion to segmentation-clock period at fixed wavefront speed.",
        "quantitative_expectation": "S multiplier = T multiplier (x0.80/x0.90/x1.10/x1.20 for -20%/-10%/+10%/+20% period), within 5 percentage points after size correction -- cross-species replication of ZF-CLK-01, not an independent free fit.",
        "basis": ["datasets/xenopus_tropicalis_clock_wavefront poson_law: Somite Length S = u*T_clock", "Cross-reference ZF-CLK-01"],
        "falsifier": "Xenopus period manipulations (hes-gene perturbation/temperature) with controlled wavefront speed producing non-proportional or opposite somite-length change.",
        "test_source": "Xenbase somitogenesis literature; Xenopus segmentation-clock (hes) studies",
        "class": "law_invariant_biological",
    },
]


# ---------------------------------------------------------------------------
# RESOLVABILITY predictions: registered BEFORE any resolution attempt.
# For every outcome prediction we separately predict whether it can be decided
# with current data. Categories:
#   current_datasets_only        -- repo datasets + sequence_links + sidecars suffice
#   current_plus_public_sources  -- also needs the named public atlas/literature
#   partial_only                 -- current + public data can bound it, not decide it
# expected_verdict is the predicted OUTCOME of scoring (confirm/refute the
# outcome prediction), kept distinct from resolvability itself.
RESOLVABILITY = {
    "ZF-TUR-01": ("current_plus_public_sources", 0.60, "confirm",
                  "Datasets give the protein states and the 0.51 size prior; deciding needs published mobility/pSmad data, which exist but must be extracted.",
                  "Extract FRAP / pSmad-gradient evidence from zebrafish gastrula literature; compare effective D ratio to the 0.51 free-diffusion prior."),
    "ZF-TUR-02": ("current_plus_public_sources", 0.85, "confirm",
                  "Protein state (tolloid cleavage of Chordin) is in UniProt via the sidecar; the phenotype test is a ZFIN lookup, no new data needed.",
                  "Score against ZFIN tolloid/minifin phenotype and chordin-cleavage literature."),
    "ZF-HH-01": ("current_plus_public_sources", 0.70, "confirm",
                  "Lipidation state captured in the sidecar; gradient-shape test needs published ptc-domain measurements.",
                  "Compare wild-type vs lipidation-perturbed Shh ptc-expression domains in the cited literature."),
    "ZF-CLK-01": ("partial_only", 0.55, "confirm",
                  "Law and sensitivity table are in the dataset; deciding needs paired period+somite measurements with wavefront controlled, which the repo does not contain and literature reports only piecemeal.",
                  "Mine somitogenesis literature for paired T and S values; a clean decision likely needs data extraction across papers, possibly new measurements."),
    "ZF-VTX-01": ("partial_only", 0.40, "confirm",
                  "Only the invariant (3.813) is in the dataset; no zebrafish cell-shape-index series exists in the repo or as a ready public table.",
                  "Measure shape index from published zebrafish EVL/epiboly images; otherwise unresolved with current data."),
    "DM-WG-01": ("current_plus_public_sources", 0.85, "confirm",
                  "Lipidation site and Notum mechanism are in the UniProt state; test is FlyBase allele + wing-disc literature, both public.",
                  "Score against porcupine/wingless lipidation-mutant gradient data in FlyBase-linked literature."),
    "DM-HH-01": ("current_plus_public_sources", 0.80, "confirm",
                  "N-product chain and dual lipidation are in the UniProt state; test data are published wing-disc experiments.",
                  "Score against cholesterol-modified vs unmodified HhN gradient literature."),
    "DM-DPP-01": ("current_datasets_only", 0.95, "confirm",
                  "Fully decidable in-repo: UniProt chain annotation (457-588) is already in sequence_links.json; the claim is about which species models must use.",
                  "Inspect P07713 chain annotation in sequence_links.json; no external source required."),
    "DM-CLK-NEG-01": ("current_plus_public_sources", 0.90, "confirm",
                   "The dataset assignment is in-repo; its failure is decidable from established Drosophila embryology in FlyBase-linked literature.",
                   "Check pair-rule/long-germ literature for any wavefront claim; absence plus simultaneous stripe specification confirms the negative."),
    "DM-MUR-NEG-01": ("partial_only", 0.45, "confirm",
                      "Law is in the dataset, but the repo holds no tracheal diameter series and public morphometry is scattered.",
                      "Extract tracheal parent/daughter diameters from published morphometry and fit the Murray exponent; may remain bounded-not-decided."),
    "XT-BMP-01": ("current_plus_public_sources", 0.85, "confirm",
                  "Noggin chain/state in sequence_links; test is the classic Xenopus injection literature via Xenbase.",
                  "Score against Noggin gain/loss dorsoventral phenotypes in Xenbase-linked literature."),
    "XT-SHH-01": ("current_plus_public_sources", 0.75, "confirm",
                  "Current link is an unreviewed entry with no lipidation; decidable by checking a reviewed UniProt/Xenbase record, a public-source lookup.",
                  "Query reviewed Xenopus tropicalis Shh annotation; confirm palmitate/cholesterol sites."),
    "XT-DQ-01": ("current_datasets_only", 0.99, "confirm",
                 "Fully decidable in-repo: sequence_links.json already records Ensembl 469 aa vs UniProt precursor 69 aa for Q28FI5.",
                 "Compare the two lengths already stored in sequence_links.json; confirm fragment status on the UniProt entry page."),
    "XT-CLK-01": ("partial_only", 0.50, "confirm",
                  "Same gap as ZF-CLK-01: law in dataset, paired period+somite data not in repo and only piecemeal in Xenopus literature.",
                  "Mine Xenopus hes/somitogenesis literature for paired T and S; likely bounded-not-decided without new measurements."),
}

META_RESOLVABILITY_PREDICTIONS = [
    {"meta_id": "META-RES-01", "prediction": "Exactly 2 of 14 outcome predictions are resolvable with current repo data alone (DM-DPP-01, XT-DQ-01).",
     "falsifier": "A third prediction decided without leaving the repo, or one of the two requiring external data."},
    {"meta_id": "META-RES-02", "prediction": "At least 10 of 14 are fully resolvable without new experiments (2 current-only + 8 with the named public atlases/literature).",
     "falsifier": "Fewer than 10 decided after a good-faith pass through the cited ZFIN/FlyBase/Xenbase/UniProt sources."},
    {"meta_id": "META-RES-03", "prediction": "Exactly 4 of 14 (ZF-CLK-01, ZF-VTX-01, DM-MUR-NEG-01, XT-CLK-01) will be only partially resolvable with current + public data, because the decisive quantitative series (paired clock period/somite length, cell-shape index, tracheal diameters) are not in the repo and not tabulated publicly.",
     "falsifier": "Any of the four cleanly decided from existing public tables, or a fifth prediction joining the partial set."},
    {"meta_id": "META-RES-04", "prediction": "0 of 14 require new wet-lab experiments merely to be DECIDABLE in principle -- every row has a named public or in-repo test source; the partial four are data-extraction problems, not experiment problems.",
     "falsifier": "A prediction whose only resolution route is generating new experimental data."},
]


def evidence_snapshot() -> dict:
    """Snapshot of the existing-data evidence each prediction stands on."""
    links = json.loads((ROOT / "sequences/sequence_links.json").read_text())
    snap = {"organisms": {}, "proteins": {}}
    for org, meta in ORGANISMS.items():
        canonical = []
        mech_dir = ROOT / "datasets"
        for d in sorted(mech_dir.glob(f"{meta['key']}_*")):
            if d.is_dir() and (d / f"{d.name}_reference.json").exists() and d.name.count("_") <= len(meta["key"].split("_")) + 2:
                canonical.append(d.name)
        snap["organisms"][org] = {**meta, "canonical_datasets_found": canonical[:10]}
    for r in links["resolved"]:
        if r["species"] in {m["key"] for m in ORGANISMS.values()}:
            st = r.get("protein_state") or {}
            snap["proteins"][f"{r['species']}:{r['gene_symbol']}"] = {
                "uniprot": r.get("uniprot"), "ensembl_protein_aa": r.get("protein_length_aa"),
                "uniprot_precursor_aa": st.get("precursor_length_aa"),
                "chains": st.get("chains"), "lipidation": st.get("lipidation"),
                "subcellular_location": st.get("subcellular_location"),
            }
    return snap


def build() -> dict:
    snap = evidence_snapshot()
    rows = []
    for p in PREDICTIONS:
        row = dict(p)
        category, confidence, verdict, why, route = RESOLVABILITY[p["prediction_id"]]
        row["resolvability"] = {
            "predicted_category": category,
            "confidence": confidence,
            "expected_verdict_on_outcome": verdict,
            "why": why,
            "resolution_route": route,
            "resolvability_status": "registered_untested",
            "resolved_on": None,
            "actual_category": None,
        }
        row["registered_on"] = REGISTERED_ON
        row["status"] = "registered_untested"
        row["scored_on"] = None
        row["outcome"] = None
        rows.append(row)
    registry = {
        "schema_version": "prediction-registry-1.0",
        "registered_on": REGISTERED_ON,
        "discipline": (
            "Registered before scoring. Based on poson_law invariants and UniProt protein states only. "
            "alpha_scale/empirical_series are excluded as a basis: across the 3,000 core datasets the "
            "empirical slope is ~1.57*alpha_scale for every law (corr ~0.99), a shared template, so "
            "alpha-derived predictions would be circular. Negative predictions falsify mechanism "
            "assignments. No row may be edited after registration except to fill status/scored_on/outcome."),
        "organism_selection": {
            "criterion": "dataset depth x protein-catalog coverage x availability of an external mutant/literature test atlas",
            "selected": {o: {"dataset_references": m["refs"], "test_atlas": m["test_atlas"]} for o, m in ORGANISMS.items()},
            "runner_up": "Mus musculus (10 refs, Bmp4/Shh in catalog, MGI) -- held for the next registration round.",
        },
        "evidence_snapshot": snap,
        "meta_resolvability_predictions": [
            {**m, "registered_on": REGISTERED_ON, "status": "registered_untested", "scored_on": None, "outcome": None}
            for m in META_RESOLVABILITY_PREDICTIONS],
        "predictions": rows,
    }
    (HERE / "prediction_registry.json").write_text(json.dumps(registry, indent=2) + "\n")
    with (HERE / "prediction_registry.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["prediction_id", "organism", "mechanism", "class", "prediction",
                                           "quantitative_expectation", "falsifier", "test_source",
                                           "resolvability_category", "resolvability_confidence",
                                           "expected_verdict_on_outcome", "resolution_route",
                                           "registered_on", "status", "scored_on", "outcome"])
        w.writeheader()
        for r in rows:
            flat = {k: r.get(k) for k in w.fieldnames}
            flat.update(resolvability_category=r["resolvability"]["predicted_category"],
                        resolvability_confidence=r["resolvability"]["confidence"],
                        expected_verdict_on_outcome=r["resolvability"]["expected_verdict_on_outcome"],
                        resolution_route=r["resolvability"]["resolution_route"])
            w.writerow(flat)
    return registry


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--self-test", action="store_true", help="offline validation of the prediction set, no writes")
    args = ap.parse_args(argv)
    if args.self_test:
        ids = [p["prediction_id"] for p in PREDICTIONS]
        assert len(ids) == len(set(ids)), "duplicate prediction ids"
        for p in PREDICTIONS:
            assert p["organism"] in ORGANISMS and p["falsifier"] and p["test_source"] and p["quantitative_expectation"]
            assert p["datasets"] and all(not d.startswith("http") for d in p["datasets"])
        assert {p["organism"] for p in PREDICTIONS} == set(ORGANISMS), "every selected organism needs predictions"
        assert set(RESOLVABILITY) == set(ids), "every outcome prediction needs a resolvability prediction"
        cats = [v[0] for v in RESOLVABILITY.values()]
        assert all(0 < v[1] <= 1 and v[2] in ("confirm", "refute") and v[3] and v[4] for v in RESOLVABILITY.values())
        assert cats.count("current_datasets_only") == 2 and cats.count("partial_only") == 4, "meta-resolvability counts must match the per-row categories"
        assert len(META_RESOLVABILITY_PREDICTIONS) == 4 and all(m["falsifier"] for m in META_RESOLVABILITY_PREDICTIONS)
        print(f"[PASS] self-test: {len(PREDICTIONS)} outcome predictions each with a resolvability prediction + expected verdict, 4 meta-resolvability predictions, category counts consistent")
        return 0
    reg = build()
    for r in reg["predictions"]:
        res = r["resolvability"]
        print(f"[{r['status']}] {r['prediction_id']:14s} {r['organism']:26s} {r['mechanism']:28s} resolvable={res['predicted_category']} (conf {res['confidence']}, expect {res['expected_verdict_on_outcome']})")
    from collections import Counter
    print("\nResolvability forecast:", dict(Counter(r["resolvability"]["predicted_category"] for r in reg["predictions"])))
    print(f"Meta-resolvability predictions: {len(reg['meta_resolvability_predictions'])} (META-RES-01..04)")
    print(f"Registered {len(reg['predictions'])} outcome + {len(reg['predictions'])} resolvability predictions on {REGISTERED_ON} (all untested)")
    print(f"Wrote {HERE / 'prediction_registry.json'}")
    print(f"Wrote {HERE / 'prediction_registry.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

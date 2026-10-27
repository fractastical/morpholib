#!/usr/bin/env python3
"""Turing open problems — forward from 1952: resolvability + provisional viability.

Problem sourcing in this repo has so far run *inward* (Problem-Scout 100 hunts
repo debt). This file runs *outward* from one historical model, Turing (1952),
to the problems the field itself still calls open (Krause–Gaffney–Maini–Klika
2021 theme issue; Landge et al. 2020; Brückner 2026), and prices each one on the
SAME two separate axes the repo already uses:

  1. Resolvability 0-100 — the deterministic Problem-Scout rubric
     (morphoeval/problem_scout_100.py RUBRIC): can OUR tools, data in hand and a
     taste-free definition-of-done reach a verdict here? Totals are recomputed
     by `score_candidate`; the factor evidence is the input, never the total.
  2. Viability % — a single-assessor PROVISIONAL Delphi posterior (prior 50%,
     signed deciban ledger, same discipline as predictions/round2_live_arguments.json).
     NOT a multi-agent panel. A Fable/GPT/Gemini panel pass may supersede it;
     it must not edit the recorded numbers.

Outputs (all regenerable from this file, which is the single source of truth):
  morphoeval/audits/2026-10-27-turing-open-problems-assessment.yaml   full ledger
  docs/turing-open-problems.json                                      graph feed
  docs/TURING_OPEN_PROBLEMS.md                                        human report
  morphoeval/problem_scouts/scout-101.json                            leaderboard input
  morphoeval/kg/2026-10-27-turing-open-problems-assessment/*.trig     10 unsigned nanopubs

Then:  python3 morphoeval/sign_morphoeval.py morphoeval/kg/2026-10-27-turing-open-problems-assessment [--publish test]
       node morphoeval/check_signed.mjs morphoeval/kg/2026-10-27-turing-open-problems-assessment/signed/*.trig
       python3 docs/build_problem_graph_html.py        # injects the band + OP-12..OP-16 and attaches the trusty URIs
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import yaml
from rdflib import Dataset, Literal, Namespace, URIRef
from rdflib.namespace import DCTERMS, PROV, RDF, RDFS, XSD

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
from problem_scout_100 import RUBRIC, score_candidate  # noqa: E402
from build_morphoeval_kg import plain_doubles          # noqa: E402

AUDIT_ID = "2026-10-27-turing-open-problems-assessment"
DATE = "2026-10-27"
CREATED = "2026-10-27T15:30:00Z"       # fixed: signing must be deterministic
ORCID = URIRef("https://orcid.org/0009-0007-4948-9192")
ASSESSOR_ID = "claude-fable-5-1"
ASSESSOR_LABEL = "Claude Fable 5.1 Thinking High (single assessor, provisional)"
PRIOR_PERCENT = 50.0
REPO = "https://github.com/fractastical/morpholib"
REPORT_URL = f"{REPO}/blob/main/docs/TURING_OPEN_PROBLEMS.md"

PK = Namespace("https://w3id.org/delphi/ontology/")
PKR = Namespace("https://w3id.org/delphi/resource/")
NP = Namespace("http://www.nanopub.org/nschema#")
NPX = Namespace("http://purl.org/nanopub/x/")
NT = Namespace("https://w3id.org/np/o/ntemplate/")
SUB = Namespace("http://purl.org/nanopub/temp/np001/")

PAPERS = {  # doi -> (year, title)
    "10.1098/rstb.1952.0012": (1952, "The Chemical Basis of Morphogenesis"),
    "10.1007/BF00289234": (1972, "A theory of biological pattern formation"),
    "10.1016/0022-5193(81)90334-9": (1981, "A Pre-pattern Formation Mechanism for Animal Coat Markings"),
    "10.1038/376765a0": (1995, "A reaction-diffusion wave on the skin of the marine angelfish Pomacanthus"),
    "10.1006/bulm.1999.0131": (1999, "Reaction and diffusion on growing domains: scenarios for robust pattern formation"),
    "10.1126/science.1130088": (2006, "WNT and DKK determine hair follicle spacing through a reaction-diffusion mechanism"),
    "10.1073/pnas.0808622106": (2009, "Interactions between zebrafish pigment cells responsible for the generation of Turing patterns"),
    "10.1073/pnas.0810311106": (2009, "The neural origins of shell structure and pattern in aquatic mollusks"),
    "10.1103/PhysRevE.84.011112": (2011, "Fluctuation-driven Turing patterns"),
    "10.1038/ng.1090": (2012, "Periodic stripe formation by a Turing mechanism operating at growth zones in the mammalian palate"),
    "10.1126/science.1221920": (2012, "Differential diffusivity of Nodal and Lefty underlies a reaction-diffusion patterning system"),
    "10.1126/science.1226804": (2012, "Hox genes regulate digit patterning by controlling the wavelength of a Turing-type mechanism"),
    "10.1126/science.1252960": (2014, "Digit patterning is controlled by a Bmp-Sox9-Wnt Turing network modulated by morphogen gradients"),
    "10.1103/PhysRevLett.114.138101": (2015, "Scaling and Regeneration of Self-Organized Patterns"),
    "10.1016/j.cels.2015.12.001": (2015, "Orientation of Turing-like Patterns by Morphogen Gradients and Tissue Anisotropies"),
    "10.7554/eLife.14022": (2016, "High-throughput mathematical analysis identifies Turing networks for patterning with equally diffusing signals"),
    "10.1038/ncomms10288": (2016, "Reconciling diverse mammalian pigmentation patterns with a fundamental mathematical model"),
    "10.1073/pnas.1522130113": (2016, "Discovery of fairy circles in Australia supports self-organization theory"),
    "10.1038/nature20801": (2017, "A theoretical foundation for multi-scale regular vegetation patterns"),
    "10.1038/nature22031": (2017, "A living mesoscopic cellular automaton made of skin scales"),
    "10.1126/science.aai7868": (2017, "Emergent cellular self-organization and mechanosensation initiate follicle pattern in the avian skin"),
    "10.1038/s41467-018-07847-x": (2018, "Synthetic mammalian pattern formation driven by differential diffusivity of Nodal and Lefty"),
    "10.1016/j.cels.2019.07.007": (2019, "A Comprehensive Network Atlas Reveals That Turing Patterns Are Common but Not Robust"),
    "10.1016/j.ydbio.2019.10.031": (2020, "Pattern formation mechanisms of self-organizing reaction-diffusion systems"),
    "10.1098/rsta.2020.0280": (2021, "Introduction to 'Recent progress and open frontiers in Turing's theory of morphogenesis'"),
    "10.1016/j.cell.2023.01.015": (2023, "The developmental basis of fingerprint pattern formation and variation"),
    "10.1073/pnas.2304470121": (2024, "Self-organized BMP signaling dynamics underlie the development and evolution of digit segmentation patterns in birds and mammals"),
    "10.1016/j.cels.2024.11.002": (2024, "A three-node Turing gene circuit forms periodic spatial patterns in bacteria"),
    "10.1101/2025.05.27.656324": (2025, "Fundamental limits on patterning by Turing-like reaction-diffusion mechanisms"),
    "10.1038/s41576-025-00925-z": (2026, "From models to molecules: self-organized and instructed modes of developmental patterning"),
}


def F(**kw) -> dict:
    """factor -> {score, why}; keys are rubric factors, <factor>_w carries the evidence line."""
    return {k: {"score": kw[k], "why": kw[k + "_w"]} for k in RUBRIC}


PROBLEMS = [
    {
        "id": "TOP-01", "slug": "mechanism-discriminator", "anchor": "Turing 1952 -> Murray 1981 -> OP-11 (2026)",
        "title": "Mechanism identification: a pre-registered statistic that separates reaction-diffusion from colonisation, phase separation and buckling on static patterns",
        "gap": "Pattern phenomenology does not identify mechanism: zebrafish stripes are cell-contact mediated (Nakamasu 2009), feather buds mechanical first (Shyer 2017), Holstein spotting indistinguishable between growing RD and Eden colonisation at p=0.91 (OP-11). No registered discriminator exists that, given only an adult pattern, says which class of generator made it with a stated error rate.",
        "done": "A hash-locked discriminator (candidate features: radial power-spectrum shape, Minkowski functionals, defect/junction statistics, two-point anisotropy) trained and scored on synthetic ground truth from four generators (Gray-Scott/Schnakenberg RD, Eden colonisation, Cahn-Hilliard phase separation, elastic buckling) with a pre-registered held-out accuracy threshold and published confusion matrix; then applied blind to the 10 coat-masked Holstein specimens, reported either way.",
        "tool_route": "models/1981-murray-bos-taurus-turing-wavelength.py (RD + Eden generators, coat masks, paired harness); models/1952-turing-morpho.py (Pearson classes); morphoeval/measure_outputs.py; predictions/ registry for the hash lock. Cahn-Hilliard and buckling generators must be written fresh (the catalog *_morphogen_condensation / *_elastic_buckling scripts are alpha-template scaffolds, scout-048).",
        "deciding_data": "In repo: 10 Holstein masks, RD/Eden outputs. Synthetic ground truth generated in-run. No wet data needed.",
        "prior_attempts": "OP-11 (static Turing rejected; growing RD vs Eden p=0.91) is the direct precursor; scout-040 OP-11 Delphi rescore pending.",
        "papers": ["10.1098/rstb.1952.0012", "10.1016/0022-5193(81)90334-9", "10.1073/pnas.0808622106", "10.1126/science.aai7868", "10.1038/ncomms10288", "10.1038/s41576-025-00925-z"],
        "factors": F(tool_fit=17, tool_fit_w="RD + Eden generators and the paired scale-matched harness exist (OP-11); phase-separation and buckling generators do not",
                     data_in_hand=15, data_in_hand_w="10 coat-masked specimens and both generators in repo; synthetic truth is free",
                     falsifiable_done=13, falsifiable_done_w="held-out accuracy vs registered threshold + confusion matrix is taste-free; feature choice is not",
                     compute_in_repo=13, compute_in_repo_w="2D RD/Eden/CH runs at 256^2 are minutes each on the repo stack",
                     isolation=9, isolation_w="new generator scripts + one registry entry; touches no tracked file",
                     time_to_verdict=6, time_to_verdict_w="two loops: build+validate discriminator, then blind application",
                     not_blocked=4, not_blocked_w="OP-11 null is a warning, not a refutation of a multi-feature discriminator"),
        "hypothesis": "A pre-registered multi-feature discriminator reaches its registered held-out accuracy on synthetic four-generator ground truth AND returns a definite class for the Holstein set.",
        "evidence_for": [[3, "RD and Eden generators plus the paired, scale-matched measurement harness already exist (OP-11 pipeline)"],
                         [2, "Minkowski functionals / spectral shape are standard, cheap and already separate RD labyrinths from spinodal fields in the physics literature"],
                         [2, "Synthetic ground truth makes the discriminator's own error rate decidable regardless of what the cows turn out to be"]],
        "evidence_against": [[-3, "OP-11: adult patch statistics did not separate growing RD from Eden at n=9 (p=0.91) — the hard case is already known to be hard"],
                             [-2, "Phase-separation and buckling generators in the catalog are template scaffolds (scout-048); both must be written and validated first"],
                             [-1, "A discriminator fitted to our own four generators may not transfer to the true generator (misspecification)"]],
        "confidence": 0.60, "verdict": "ACHIEVABLE_IN_TWO_LOOPS_HARD_CASE_KNOWN",
    },
    {
        "id": "TOP-02", "slug": "diffusion-ratio-census", "anchor": "Gierer-Meinhardt 1972 -> Marcon 2016 -> 2025 'fundamental limits'",
        "title": "The diffusion-ratio problem: which molecularly identified Turing systems can be two-species at all?",
        "gap": "Two-species activator-inhibitor models need D_inh/D_act above the Lean-checked threshold (d_c = 3.93 for the repo's Gray-Scott Jacobian; ~4-10 generally), yet measured extracellular diffusivities rarely show it; three-node networks with an immobile node remove the requirement (Marcon 2016) and extracellular wiring changes behaviour qualitatively (2025 bioRxiv). Nobody has tabulated, system by system, which of the molecularly supported cases (WNT/DKK, FGF/Shh palate, Bmp-Sox9-Wnt digits, GDF5/NOG joints, EDAR/WNT/BMP ridges, Nodal/Lefty) actually satisfies the two-species condition with measured D.",
        "done": "A registered table of >=6 molecular Turing systems giving measured or bounded D_act, D_inh (with DOI), run through the repo's dispersion relation, classifying each as two-node-feasible / needs immobile third node / D unmeasured; the count in each class pre-registered before the literature values are looked up.",
        "tool_route": "python3 morphoeval/scout_literature.py --query '<morphogen> diffusion coefficient FRAP FCS' per system; models/1952-turing-morpho.py dispersion (Lean turing_differential_diffusion_instability, d_c = 3.9271); predictions/ registry row for the pre-registered counts.",
        "deciding_data": "Public literature only. Known in hand: Nodal/Lefty D measured (Mueller 2012); GDF5/NOG model needs no differential diffusion (Grall 2024). Others likely unmeasured — that is itself the finding.",
        "prior_attempts": "None in repo. scout-063 ZF-TUR-01 kernel-checked a size-prior D ratio (D_inh/D_act < 1 for chd/bmp2b) — a related negative.",
        "papers": ["10.1007/BF00289234", "10.1126/science.1221920", "10.7554/eLife.14022", "10.1126/science.1130088", "10.1038/ng.1090", "10.1126/science.1252960", "10.1073/pnas.2304470121", "10.1016/j.cell.2023.01.015", "10.1101/2025.05.27.656324"],
        "factors": F(tool_fit=19, tool_fit_w="scout_literature.py + the existing dispersion code and Lean d_c decide each row",
                     data_in_hand=10, data_in_hand_w="all values are one public fetch away, but most D's are probably unmeasured",
                     falsifiable_done=13, falsifiable_done_w="pre-registered class counts; the 'unmeasured' class is checkable by absence of a DOI",
                     compute_in_repo=14, compute_in_repo_w="linear stability per row is instantaneous",
                     isolation=10, isolation_w="one new table + registry row",
                     time_to_verdict=7, time_to_verdict_w="one sitting of literature extraction",
                     not_blocked=5, not_blocked_w="no prior attempt"),
        "hypothesis": "The census reaches a classification for >=6 systems, with the pre-registered class counts scored, and at least one system falls in each of the three classes.",
        "evidence_for": [[3, "d_c is Lean-checked and the dispersion code exists; each row is a one-line computation once D's are in hand"],
                         [2, "Nodal/Lefty D's are measured (Mueller 2012) and GDF5/NOG is published as not needing differential diffusion (Grall 2024): two rows are already decidable"],
                         [1, "Literature route is the repo's own scout_literature.py"]],
        "evidence_against": [[-3, "For WNT/DKK, FGF/Shh, EDAR/BMP the extracellular D's are not measured: the majority verdict may be 'unmeasured', informative but thin"],
                             [-2, "Effective D depends on receptor binding and sequestration (2025 'fundamental limits'), so a raw D ratio is not the deciding quantity for the biology, only for the two-species model class"]],
        "confidence": 0.65, "verdict": "DECIDABLE_TABLE_MOSTLY_UNMEASURED_ROWS",
    },
    {
        "id": "TOP-03", "slug": "robustness-fraction", "anchor": "Murray 1982 -> Scholes 2019 -> Tica 2024",
        "title": "The fine-tuning objection: replicate the two-node vs three-node Turing parameter-volume ratio and the 1%-uncertainty retention claim",
        "gap": "Turing instability is topologically common but occupies tiny parameter volumes: 0.00035% of sampled space for the classical two-node circuit vs 0.022% for the best three-node topology (Tica et al. 2024, Latin-hypercube linear stability), with 33% retention under 1% parameter uncertainty around a Turing I point. These numbers decide whether the fine-tuning objection to biological Turing systems stands; they have not been independently reproduced.",
        "done": "A pre-registered Latin-hypercube linear-stability sweep on the repo's two-node kinetics and a three-node immobile-node topology within published parameter bounds, reporting both Turing fractions and the 1%/5% local-retention fractions; verdict = within one order of magnitude of the published ratio (>=10x three-node over two-node) or not.",
        "tool_route": "models/1952-turing-morpho.py (Jacobian / dispersion code, Pearson Saddle-Node & Hopf curves, Lean d_c); new sweep script under morphoeval/ or models/; registry row for the ratio before running.",
        "deciding_data": "Published fractions (Tica 2024 Fig. 2A, Table S5 bounds; Scholes 2019 atlas) — no biological data needed.",
        "prior_attempts": "scout-034 proposed a 12-class Pearson (F,k) sweep, not a robustness fraction; nothing recorded on parameter volume.",
        "papers": ["10.1098/rstb.1952.0012", "10.1016/j.cels.2019.07.007", "10.7554/eLife.14022", "10.1016/j.cels.2024.11.002", "10.1103/PhysRevE.84.011112"],
        "factors": F(tool_fit=20, tool_fit_w="linear stability + Lean d_c already in the flagship script; a sweep driver is all that is missing",
                     data_in_hand=18, data_in_hand_w="target numbers and parameter bounds are published; nothing to fetch but a supplementary table",
                     falsifiable_done=14, falsifiable_done_w="order-of-magnitude ratio and retention fractions are numbers a scorer reads off",
                     compute_in_repo=13, compute_in_repo_w="10^6-10^7 LHS Jacobian evaluations in numpy, under an hour",
                     isolation=10, isolation_w="one new script + registry row",
                     time_to_verdict=8, time_to_verdict_w="one loop",
                     not_blocked=5, not_blocked_w="no prior attempt"),
        "hypothesis": "The repo's sweep reproduces the published two-node vs three-node Turing-fraction ratio within one order of magnitude and the >=10x ordering, with the pre-registered numbers scored.",
        "evidence_for": [[4, "Pure deterministic computation; linear stability of small Jacobians is exactly what the flagship already does and Lean already checks"],
                         [3, "Published target numbers and bounds exist to score against"],
                         [2, "Pearson/Turing LSA code and the d_c theorem are already in the repo"]],
        "evidence_against": [[-2, "Fractions are sensitive to the exact parameter bounds and sampling conventions (Table S5); mismatch shifts them by orders of magnitude"],
                             [-1, "LHS undersamples high-dimensional parameter space (the authors say so); 'within one order of magnitude' may still fail on sampling noise"]],
        "confidence": 0.70, "verdict": "LIKELY_REPRODUCIBLE_ONE_LOOP",
    },
    {
        "id": "TOP-04", "slug": "stripe-insertion-blind", "anchor": "Kondo-Asai 1995 -> Crampin-Gaffney-Maini 1999 -> Krause 2021",
        "title": "Growing domains: an independent ontogenetic stripe-count series to test the insertion law blind",
        "gap": "Classical Turing conditions are neither necessary nor sufficient on growing domains and wavelength selection under growth has no closed theory (Krause et al. 2021). Empirically, Kondo-Asai's Pomacanthus series remains nearly the only ontogenetic stripe-count dataset, and the repo's Kondo stripe table is calibrated on it (circularity recorded in scout-034/038; OP-06 growing-domain solver debt still open).",
        "done": "A hash-locked prediction N_stripes(L) = round(L/lambda_c) with lambda_c fixed from one stage, scored on an independently sourced Pomacanthus (or other stripe-inserting fish) ontogenetic series of >=6 stages with body length metadata, each image SHA-256 pinned before counting; result reported either way.",
        "tool_route": "python3 morphoeval/scout_literature.py --taxon 'Pomacanthus semicirculatus' (Commons + OpenAlex); verify_image_fitness.py / the Y-junction skeleton detector for counting; predictions/ registry for the lock.",
        "deciding_data": "Not in repo. Needs public juvenile photographs with size metadata or a published series other than Kondo-Asai 1995 Fig. 1.",
        "prior_attempts": "scout-031 (Kondo ontogenetic series beyond one adult photo), scout-034/038 (circular calibration), scout-032 (Y-junction over-count) — all record the gap, none closed it.",
        "papers": ["10.1038/376765a0", "10.1006/bulm.1999.0131", "10.1098/rsta.2020.0280"],
        "factors": F(tool_fit=16, tool_fit_w="literature scout + CV counter exist; the counter over-counts adult junctions (scout-032)",
                     data_in_hand=8, data_in_hand_w="no independent series in repo; juvenile images with sizes are scarce on Commons",
                     falsifiable_done=13, falsifiable_done_w="integer stripe counts vs a locked formula are taste-free once images are pinned",
                     compute_in_repo=12, compute_in_repo_w="counting is cheap; 2D growing-domain solver would be the expensive add-on",
                     isolation=9, isolation_w="new dataset folder + registry row",
                     time_to_verdict=5, time_to_verdict_w="data hunt dominates; verdict in one loop only if a series is found",
                     not_blocked=3, not_blocked_w="circularity of the current table is recorded; no refutation of an independent test"),
        "hypothesis": "An independent >=6-stage series with size metadata is found and the locked insertion-law prediction is scored decisively (all stages within +-1 stripe, or not).",
        "evidence_for": [[2, "The insertion law is sharp: stripe count doubles when body length doubles, so +-1 tolerance is a real test"],
                         [2, "A stripe/junction detector already exists and only needs re-tuning for juveniles"]],
        "evidence_against": [[-4, "No independent ontogenetic series is known; the only published one calibrated the repo's own table"],
                             [-2, "Commons juvenile photographs lack body-length metadata"],
                             [-1, "The existing detector over-counts on adult ROIs (scout-032/034)"]],
        "confidence": 0.60, "verdict": "BLOCKED_ON_DATA_NOT_ON_TOOLS",
    },
    {
        "id": "TOP-05", "slug": "wavelength-scaling", "anchor": "Turing 1952 (intrinsic wavelength) -> Werner 2015 (scaling)",
        "title": "Scaling: does biological pattern wavelength scale with body size across taxa, against Turing's intrinsic wavelength?",
        "gap": "Turing wavelength is set by kinetics and diffusion, not domain size, yet many biological patterns scale with the organism (planarian regeneration, Werner 2015; expansion-repression schemes). No cross-taxon regression of measured wavelength against body size exists. The repo's 976 *_turing_wavelength dataset folders look like such an asset but their empirical_series are alpha-template curves (x_control 0.5-5.0, r=0.999 fits), not measurements (scout-048 / scout-062 data-quality findings), so they cannot decide this.",
        "done": "A pre-registered log-log regression of measured pattern wavelength on body size for >=10 taxa with real, DOI-cited measurements (hair-follicle spacing, fish stripe period, digit spacing, palatal ruga spacing), slope and CI locked before extraction; verdict = slope CI excludes 0 (scaling) or excludes 1 (intrinsic) or neither.",
        "tool_route": "scout_literature.py for measurements; a new extraction table under datasets/; registry row. The catalog templates must NOT be used as evidence.",
        "deciding_data": "Not usable in repo (template artefacts). Public literature extraction required.",
        "prior_attempts": "scout-048 BIO-TPL-DQ-01 and scout-062 established the catalog empirical curves are template artefacts; no scaling test attempted.",
        "papers": ["10.1098/rstb.1952.0012", "10.1103/PhysRevLett.114.138101", "10.1038/376765a0", "10.1126/science.1130088"],
        "factors": F(tool_fit=12, tool_fit_w="regression is trivial; the tool gap is a trustworthy measurement table",
                     data_in_hand=4, data_in_hand_w="the 976 catalog series are template curves, not data; everything must be extracted fresh",
                     falsifiable_done=12, falsifiable_done_w="locked slope CI test is taste-free; wavelength definitions across taxa are not",
                     compute_in_repo=12, compute_in_repo_w="negligible compute",
                     isolation=9, isolation_w="new table + registry row",
                     time_to_verdict=4, time_to_verdict_w="literature extraction across >=10 taxa is days, not a loop",
                     not_blocked=1, not_blocked_w="the obvious in-repo route is recorded as an artefact (scout-048/062)"),
        "hypothesis": "A >=10-taxon measured wavelength-vs-size table is assembled and the locked slope test returns a decisive class.",
        "evidence_for": [[2, "The question is sharp (slope 0 vs 1) and a few real measured systems exist (fish stripes across sizes, planarian scaling, follicle spacing)"]],
        "evidence_against": [[-5, "The repo's apparent asset (976 series) is void: template curves, recorded as such"],
                             [-3, "Heterogeneous wavelength definitions across taxa make a single regression contestable"],
                             [-1, "Growth-time and developmental-stage confounds"]],
        "confidence": 0.65, "verdict": "DATA_ASSET_IS_AN_ARTEFACT",
    },
    {
        "id": "TOP-06", "slug": "reproducible-topology", "anchor": "Sheth 2012 -> Raspopovic 2014 -> Green-Sharpe 2015",
        "title": "Reproducible topology: why five digits every time when RD from noise gives defective labyrinths",
        "gap": "RD from random initial conditions yields labyrinths with defects; embryos give five digits almost every time. The Turing + positional-information hybrid (Hox dosage sets wavelength, FGF/Wnt gradients orient) explains cases, but there is no general, tested account of how prepattern and boundaries lock topology.",
        "done": "A Bmp-Sox9-Wnt three-node RD on a growing fan-shaped limb-bud domain with published parameters and gradient modulation, pre-registered to produce exactly five rays in >=95% of 200 random-IC runs (and the Hox-dose polydactyly series of Sheth 2012 on dose reduction); scored as registered.",
        "tool_route": "New script (no BSW model exists in models/); 2D RD solver base from models/1952-turing-morpho.py; registry row for the 95% threshold.",
        "deciding_data": "Published parameters (Raspopovic 2014 SOM). No wet data.",
        "prior_attempts": "None in repo.",
        "papers": ["10.1126/science.1226804", "10.1126/science.1252960", "10.1016/j.cels.2015.12.001"],
        "factors": F(tool_fit=9, tool_fit_w="only the generic 2D RD base exists; the three-node growing-fan model must be built",
                     data_in_hand=9, data_in_hand_w="parameters are published; validation target (five rays, polydactyly series) is qualitative",
                     falsifiable_done=12, falsifiable_done_w="ray count over 200 runs is a number; domain shape and growth schedule are modelling choices",
                     compute_in_repo=11, compute_in_repo_w="200 2D runs on a growing domain: hours",
                     isolation=10, isolation_w="one new script",
                     time_to_verdict=4, time_to_verdict_w="model build + validation is more than one loop",
                     not_blocked=5, not_blocked_w="no prior attempt"),
        "hypothesis": "A faithful BSW growing-domain model reproduces five rays in >=95% of random-IC runs and the Hox-dose polydactyly ordering.",
        "evidence_for": [[2, "Equations and parameters are published and the original work reproduced digit patterns in silico"],
                         [1, "The repo has a validated 2D RD base and the Lean d_c machinery"]],
        "evidence_against": [[-3, "No BSW script exists; building and validating it precedes any test"],
                             [-3, "The 95% topological reproducibility criterion is stronger than what the source papers claimed; growth schedule and boundary choices can decide it either way"],
                             [-1, "More than one loop before a verdict"]],
        "confidence": 0.55, "verdict": "MODEL_MUST_BE_BUILT_FIRST",
    },
    {
        "id": "TOP-07", "slug": "rd-to-ca-reduction", "anchor": "Meinhardt 1995 -> Boettiger-Ermentrout-Oster 2009 -> Manukyan 2017",
        "title": "When does continuous reaction-diffusion collapse to a discrete automaton? Meinhardt mantle PDE vs the Lean-proved Rule 30 gate",
        "gap": "Ocellated lizard scales behave as a cellular automaton emerging from RD on a thick-skin lattice (Manukyan 2017), and the repo has Lean-proved that the Ermentrout-Oster neural mantle gate is bit-for-bit Rule 30 — but whether the repo's Meinhardt-Klingler Conus PDE, discretised at the mantle, induces a Rule 30-class local rule has never been measured. It is the bridge between the Rule 30 programme and the RD programme, and the RD-vs-neural shell-pattern debate is still molecularly unresolved.",
        "done": "From a Meinhardt PDE kymograph at the mantle edge, extract the empirical 3-neighbourhood -> next-state transducer table (binary threshold pre-registered), report the fraction of transitions agreeing with Rule 30 vs Rule 90 vs best-fit rule, plus the 2D ANF-carry density and wall chirality already used to separate R30 from R90 (node r0); verdict thresholds locked before running.",
        "tool_route": "models/1984-wolfram-conus-textile-rule30.py (Meinhardt-Klingler PDE + Rule 30 + kymograph); morphoeval/lean/ Ermentrout-Oster gate theorem; measure_outputs.py for ANF carry / chirality; registry row.",
        "deciding_data": "All in repo (PDE, CA, Lean gate, 2D observables).",
        "prior_attempts": "scout-003 (Rule30 <-> Ermentrout-Oster gate chain traceability); scout-034/038/039 (stochastic activator injection debt, OP-06b). None extracted the transducer table.",
        "papers": ["10.1073/pnas.0810311106", "10.1038/nature22031", "10.1098/rstb.1952.0012"],
        "factors": F(tool_fit=18, tool_fit_w="PDE, CA, Lean gate and 2D observables all exist; only the transducer extraction is new",
                     data_in_hand=17, data_in_hand_w="everything is generated in repo",
                     falsifiable_done=12, falsifiable_done_w="rule-agreement fractions are numbers; the binarisation threshold is the taste risk, so it is pre-registered",
                     compute_in_repo=13, compute_in_repo_w="1D PDE kymographs are seconds",
                     isolation=9, isolation_w="new analysis script + registry row",
                     time_to_verdict=6, time_to_verdict_w="one loop if the PDE is used as-is",
                     not_blocked=3, not_blocked_w="OP-06b: the PDE only patterns with an unsourced stochastic injection, so the object reduced is not autonomous"),
        "hypothesis": "The extracted transducer table from the Meinhardt mantle PDE agrees with Rule 30 above the locked threshold on both 1D transitions and 2D ANF/chirality observables.",
        "evidence_for": [[3, "Both objects are in the repo and the gate = Rule 30 identity is Lean-proved"],
                         [2, "Transducer-table extraction from a kymograph is mechanical"],
                         [1, "Manukyan 2017 is a published template for RD -> CA reduction"]],
        "evidence_against": [[-3, "The Conus PDE patterns only with the unsourced stochastic injection (OP-06b): any reduction is of a non-autonomous object until that is fixed"],
                             [-2, "Continuous -> binary thresholding can manufacture or destroy the rule match"],
                             [-1, "1D chords cannot separate R30 from R90 (KS 0.2623 vs 0.2627, node r0); the 2D observables must carry the verdict"]],
        "confidence": 0.60, "verdict": "DECIDABLE_BUT_OBJECT_NOT_YET_AUTONOMOUS",
    },
    {
        "id": "TOP-08", "slug": "synthetic-eukaryotic-turing", "anchor": "Sekine 2018 -> Tica 2024",
        "title": "Synthetic realisation: a robust, deterministic periodic Turing pattern in eukaryotic cells",
        "gap": "Bacterial three-node circuits now give reproducible concentric stripes but remain condition-sensitive (Tica 2024); the mammalian Nodal-Lefty circuit most likely produced solitary, not periodic, patterns (Sekine 2018). A robust eukaryotic periodic Turing pattern has not been built.",
        "done": "Wet-lab: a eukaryotic circuit producing periodic patterns with pre-registered wavelength reproducibility across >=3 biological replicates. The repo can only contribute the Turing-vs-solitary regime map of the published circuit model.",
        "tool_route": "None adequate: no wet lab. Modelling satellite only (regime map of the Sekine competitive-inhibition model).",
        "deciding_data": "Not obtainable here.",
        "prior_attempts": "None in repo.",
        "papers": ["10.1038/s41467-018-07847-x", "10.1016/j.cels.2024.11.002"],
        "factors": F(tool_fit=3, tool_fit_w="no instrument in this repo can run a wet-lab experiment",
                     data_in_hand=4, data_in_hand_w="only the published model equations",
                     falsifiable_done=12, falsifiable_done_w="replicate wavelength reproducibility is checkable — elsewhere",
                     compute_in_repo=2, compute_in_repo_w="the deciding evidence is not computational",
                     isolation=10, isolation_w="nothing to conflict with",
                     time_to_verdict=1, time_to_verdict_w="months to years, in a lab",
                     not_blocked=5, not_blocked_w="no prior attempt"),
        "hypothesis": "This repo brings the synthetic eukaryotic Turing problem to a decisive verdict.",
        "evidence_for": [[1, "A regime map of the published Sekine model (Turing vs solitary) is a tractable modelling satellite"]],
        "evidence_against": [[-10, "The deciding evidence is wet-lab; none of the repo's instruments reach it"]],
        "confidence": 0.80, "verdict": "NOT_WITH_OUR_TOOLS",
    },
    {
        "id": "TOP-09", "slug": "turing-hopf-wave-signature", "anchor": "Turing 1952 Sec. 10 (oscillatory case) -> Ca2+ wave-vector pipeline",
        "title": "Turing's neglected case: is there a wave (Turing-Hopf) dispersion signature in the embryonic Ca2+ wave field?",
        "gap": "Turing 1952 also derived oscillatory, travelling-wave instabilities; essentially no developmental system has been shown to use one. The repo's Ca2+ spark-track data (x, y, t per event) is the only place here that can ask whether an observed wave field carries a dispersion ridge omega(k) distinct from a single driven front (omega = v k).",
        "done": "A pre-registered space-time spectral test on existing spark_tracks CSVs: estimate the (k, omega) power surface per embryo, fit the driven-front line, and test for a secondary ridge against phase-randomised surrogates at a locked false-positive rate; report per-embryo outcomes and the pooled verdict either way.",
        "tool_route": "wave-vector-analysis/ (spark_tracks.csv schema, embryo_units.py for em/s conversion); new spectral script; predictions/ registry for the surrogate threshold.",
        "deciding_data": "In repo: wave-vector-analysis/test_data/spark_tracks_qc_folder31.csv, _folder33.csv and analysis_results/*.csv; orientation and coordinate caveats recorded (COORDINATE_SYSTEM_WARNING.md, OP-07).",
        "prior_attempts": "scout-036 proposed binned AP spark spectra vs Turing k_c and a travelling-front vs static-labyrinth gate; neither run. Blind tests OP-07/08 found orientation, not wave features, carried signal.",
        "papers": ["10.1098/rstb.1952.0012", "10.1016/j.ydbio.2019.10.031"],
        "factors": F(tool_fit=15, tool_fit_w="pipeline and unit conversion exist; the spectral estimator is new but small",
                     data_in_hand=14, data_in_hand_w="several spark_tracks CSVs are in the repo",
                     falsifiable_done=10, falsifiable_done_w="surrogate-tested secondary ridge at a locked rate is checkable; ridge definition needs care",
                     compute_in_repo=13, compute_in_repo_w="FFTs of event fields: seconds",
                     isolation=9, isolation_w="new script + registry row",
                     time_to_verdict=6, time_to_verdict_w="one loop",
                     not_blocked=4, not_blocked_w="OP-07/08 negatives concern neighbour prediction, not dispersion"),
        "hypothesis": "The registered surrogate test returns a decisive per-embryo verdict (secondary ridge present/absent at the locked rate) on the in-repo spark-track data.",
        "evidence_for": [[2, "Event (x, y, t) data are in repo and space-time spectra are cheap"],
                         [2, "The null (single driven front, omega = v k) is sharp and the surrogate test is standard"]],
        "evidence_against": [[-3, "Poke-triggered waves are driven, not spontaneous; a wave instability would have to appear as a weak secondary ridge, and prior blind tests found wave features carry little signal"],
                             [-2, "Short, sparse event fields give poor (k, omega) resolution; the ridge criterion can slide into taste unless locked tightly"],
                             [-1, "Orientation/coordinate confounds documented in OP-07 and COORDINATE_SYSTEM_WARNING.md"]],
        "confidence": 0.55, "verdict": "TESTABLE_NOW_LIKELY_NULL",
    },
    {
        "id": "TOP-10", "slug": "ecological-turing-fairy-circles", "anchor": "Klausmeier 1999 -> Getzin 2016 -> Tarnita 2017",
        "title": "Ecology-scale Turing: fairy circles as scale-dependent-feedback RD vs termite engineering",
        "gap": "Namibian and Australian fairy circles are still contested between vegetation reaction-diffusion (scale-dependent feedback) and termite engineering; Tarnita 2017 shows the two are not exclusive. Pattern statistics alone (hexagonal order, pair correlation) have not settled it — the same lesson as TOP-01.",
        "done": "A pre-registered comparison of a vegetation RD model and a termite-territory model against public aerial-imagery pair-correlation and Voronoi statistics with locked discrimination thresholds; reported either way.",
        "tool_route": "No ecological model or imagery in repo; new scripts and a public data fetch would be required.",
        "deciding_data": "Public aerial imagery / published point patterns (Getzin et al.). Nothing in repo.",
        "prior_attempts": "None in repo.",
        "papers": ["10.1073/pnas.1522130113", "10.1038/nature20801"],
        "factors": F(tool_fit=8, tool_fit_w="generic RD base only; no vegetation or termite model, no imagery tooling",
                     data_in_hand=5, data_in_hand_w="all data external",
                     falsifiable_done=10, falsifiable_done_w="locked thresholds are checkable but the mechanisms are not exclusive",
                     compute_in_repo=10, compute_in_repo_w="feasible compute once models exist",
                     isolation=10, isolation_w="new files only",
                     time_to_verdict=3, time_to_verdict_w="several loops",
                     not_blocked=5, not_blocked_w="no prior attempt"),
        "hypothesis": "Pattern statistics from public imagery decisively favour one mechanism at the locked thresholds.",
        "evidence_for": [[2, "Public imagery and standard point-pattern statistics exist"]],
        "evidence_against": [[-4, "No repo data or model; everything must be built"],
                             [-2, "Mechanisms are not mutually exclusive (Tarnita 2017), so pattern statistics are unlikely to be decisive — the TOP-01 lesson"]],
        "confidence": 0.50, "verdict": "OUTSIDE_THE_REPO_AND_LIKELY_INDECISIVE",
    },
]


def posterior(prior_pct: float, net_db: float) -> float:
    odds = (prior_pct / 100) / (1 - prior_pct / 100) * 10 ** (net_db / 10)
    return 100 * odds / (1 + odds)


def assess(p: dict) -> dict:
    """Attach the two ratings: deterministic resolvability + provisional viability."""
    score, band_detail = score_candidate(p)
    for_db = sum(db for db, _ in p["evidence_for"])
    against_db = sum(db for db, _ in p["evidence_against"])
    net = for_db + against_db
    post = posterior(PRIOR_PERCENT, net)
    band = band_detail.split(" — ")[0]
    return {**p, "resolvability": score, "band": band, "band_detail": band_detail,
            "viability": {"assessor": ASSESSOR_LABEL, "prior_percent": PRIOR_PERCENT, "for_db": for_db,
                          "against_db": against_db, "net_db": net, "bayes_factor": round(10 ** (net / 10), 4),
                          "posterior_percent": round(post, 1), "confidence": p["confidence"], "verdict": p["verdict"],
                          "classification": "possible_pathway" if post >= 15 else "below_15_percent"}}


def kg_dir() -> Path:
    return HERE / "kg" / AUDIT_ID


def trig_name(p: dict) -> str:
    return f"{p['id'].lower()}-{p['slug']}.trig"


def build_nanopub(p: dict) -> str:
    ds = Dataset()
    for pfx, ns in (("np", NP), ("npx", NPX), ("nt", NT), ("pk", PK), ("pkr", PKR), ("prov", PROV), ("dct", DCTERMS),
                    ("rdfs", RDFS), ("xsd", XSD), ("sub", SUB)):
        ds.bind(pfx, ns)
    head, asrt, prov_g, pubinfo = (ds.graph(SUB[g]) for g in ("Head", "assertion", "provenance", "pubinfo"))
    this = URIRef(str(SUB))
    head.add((this, RDF.type, NP.Nanopublication))
    head.add((this, NP.hasAssertion, SUB.assertion)); head.add((this, NP.hasProvenance, SUB.provenance))
    head.add((this, NP.hasPublicationInfo, SUB.pubinfo))

    cid = f"turing-open-problem-{p['id'].lower()}-{p['slug']}"
    claim = PKR[f"claim/{cid}"]; a_res = PKR[f"assessment/{cid}-resolvability"]
    a_via = PKR[f"assessment/{cid}-viability"]; hyp = PKR[f"hypothesis/{cid}-decisive-verdict"]
    model = PKR[f"model/{ASSESSOR_ID}"]; report = URIRef(REPORT_URL)
    v = p["viability"]
    L = lambda s: Literal(s)
    asrt.add((claim, RDF.type, PK.Claim))
    asrt.add((claim, RDFS.label, L(f"{p['id']} — {p['title']}")))
    asrt.add((claim, DCTERMS.description, L(
        f"OPEN PROBLEM in Turing-pattern biology, sourced forward from {p['anchor']}. THE GAP: {p['gap']} "
        f"DEFINITION OF DONE: {p['done']} Rated on two separate axes: resolvability {p['resolvability']}/100 "
        f"({p['band']}) by the deterministic Problem-Scout rubric (prices our tools, data in hand and a checkable "
        f"definition-of-done, not truth odds), and a PROVISIONAL single-assessor viability posterior of "
        f"{v['posterior_percent']}% (prior {PRIOR_PERCENT:g}%, net {v['net_db']:+g} dB) that the named route reaches a "
        f"decisive verdict — not a multi-agent panel score. A belief under assessment, not a fact.")))
    asrt.add((claim, DCTERMS.subject, report))
    asrt.add((claim, PK.context, L(f"tool_route: {p['tool_route']} | deciding_data: {p['deciding_data']} | prior_attempts: {p['prior_attempts']}")))
    for doi in p["papers"]:
        asrt.add((claim, PK.derivedFromPaper, URIRef(f"https://doi.org/{doi}")))
    asrt.add((claim, PK.hasAssessment, a_res)); asrt.add((claim, PK.hasAssessment, a_via))
    asrt.add((claim, PK.weightOfEvidence, Literal(round(v["posterior_percent"] / 100, 4), datatype=XSD.double)))

    asrt.add((a_res, RDF.type, PK.LLMClaimAssessment))
    asrt.add((a_res, RDFS.label, L(f"{p['id']} resolvability {p['resolvability']}/100 — {p['band']}")))
    asrt.add((a_res, DCTERMS.date, Literal(DATE, datatype=XSD.date)))
    asrt.add((a_res, DCTERMS.description, L("Deterministic Problem-Scout rubric (morphoeval/problem_scout_100.py RUBRIC): "
                                             + "; ".join(f"{k} (0-{w}): {d}" for k, (w, d) in RUBRIC.items())
                                             + ". Score = sum of factor scores / 100; an uncalibrated rubric score, not a probability.")))
    asrt.add((a_res, PK.assessedBy, model))
    asrt.add((a_res, PK.assessmentType, L("problem-scout/1.0/resolvability")))
    asrt.add((a_res, PK.hasScore, Literal(round(p["resolvability"] / 100, 4), datatype=XSD.double)))
    asrt.add((a_res, PK.hasVerdict, L(p["band"])))
    for k, (w, _d) in RUBRIC.items():
        f = p["factors"][k]
        asrt.add((a_res, PK.evidenceFor, L(f"{k} {f['score']}/{w}: {f['why']}")))

    asrt.add((hyp, RDF.type, PK.TestableHypothesis))
    asrt.add((hyp, RDFS.label, L(p["hypothesis"])))
    asrt.add((hyp, PK.hasAssessment, a_via))
    asrt.add((hyp, PK.hasPriorProbability, Literal(PRIOR_PERCENT / 100, datatype=XSD.double)))
    asrt.add((hyp, PK.hasResolvability, Literal(round(p["resolvability"] / 100, 4), datatype=XSD.double)))
    asrt.add((a_via, RDF.type, PK.BayesianAssessment))
    asrt.add((a_via, RDFS.label, L(f"{p['id']} viability {v['posterior_percent']}% — provisional single-assessor Delphi")))
    asrt.add((a_via, PROV.wasAttributedTo, model))
    asrt.add((a_via, PK.calibrationMethod, L(f"Single-assessor provisional signed deciban ledger, prior {PRIOR_PERCENT:g}% (predictions/round2_live_arguments.json discipline); NOT a multi-agent panel; a panel pass may supersede but not edit")))
    asrt.add((a_via, PK.evaluatesHypothesis, hyp))
    for db, text in p["evidence_for"]:
        asrt.add((a_via, PK.evidenceFor, L(f"[{db:+g} dB] {text}")))
    for db, text in p["evidence_against"]:
        asrt.add((a_via, PK.evidenceAgainst, L(f"[{db:+g} dB] {text}")))
    asrt.add((a_via, PK.weightOfEvidenceDeciban, Literal(float(v["net_db"]), datatype=XSD.double)))
    asrt.add((a_via, PK.bayesFactor, Literal(v["bayes_factor"], datatype=XSD.double)))
    asrt.add((a_via, PK.hasProbability, Literal(round(v["posterior_percent"] / 100, 4), datatype=XSD.double)))
    asrt.add((a_via, PK.hasConfidence, Literal(v["confidence"], datatype=XSD.double)))
    asrt.add((a_via, PK.hasVerdict, L(v["verdict"])))
    asrt.add((model, RDF.type, PK.Model)); asrt.add((model, RDFS.label, L(ASSESSOR_LABEL)))
    asrt.add((model, DCTERMS.identifier, L(ASSESSOR_ID))); asrt.add((model, PROV.actedOnBehalfOf, ORCID))

    prov_g.add((SUB.assertion, PROV.wasAttributedTo, ORCID)); prov_g.add((SUB.assertion, PROV.wasAttributedTo, model))
    for doi in p["papers"]:
        u = URIRef(f"https://doi.org/{doi}"); yr, title = PAPERS[doi]
        prov_g.add((SUB.assertion, PROV.wasDerivedFrom, u))
        prov_g.add((u, DCTERMS.title, L(title))); prov_g.add((u, DCTERMS.date, Literal(str(yr), datatype=XSD.gYear)))

    pubinfo.add((this, RDFS.label, L(f"Open problem: {p['id']} {p['title'][:90]}")))
    pubinfo.add((this, DCTERMS.created, Literal(CREATED, datatype=XSD.dateTime)))
    pubinfo.add((this, DCTERMS.creator, ORCID))
    pubinfo.add((this, DCTERMS.license, URIRef("https://creativecommons.org/licenses/by/4.0/")))
    pubinfo.add((this, NPX.hasNanopubType, PK.Claim)); pubinfo.add((this, NPX.introduces, claim))
    pubinfo.add((this, RDFS.seeAlso, report))
    dom = PKR["domain/morphogenesis"]
    pubinfo.add((this, PK.domain, dom)); pubinfo.add((dom, RDFS.label, L("morphogenesis")))
    pubinfo.add((this, NT.wasCreatedFromProvenanceTemplate, URIRef("https://w3id.org/np/RASh_Tn3WNfWht2FgL8XQGseMKF3vVdWi-PRROMGuBu9Y")))
    for t in ("RACJ58Gvyn91LqCKIO9zu1eijDQIeEff28iyDrJgjSJF8", "RAGIjRyQ2tgVqY29I6MKx5hzoIwr1f1kGtQw8EP1tTCPY", "RAukAcWHRDlkqxk7H2XNSegc1WnHI569INvNr-xdptDGI"):
        pubinfo.add((this, NT.wasCreatedFromPubinfoTemplate, URIRef(f"https://w3id.org/np/{t}")))
    pubinfo.add((this, NT.wasCreatedFromTemplate, URIRef("https://w3id.org/np/RA3z187EzQXy4zopVZAmFg2sa_QymD1MKfzMxGzNqdNZM/template")))
    return plain_doubles(ds.serialize(format="trig"))


def write_report(problems: list[dict], path: Path) -> None:
    ranked = sorted(problems, key=lambda p: -p["resolvability"])
    L = ["# Turing open problems — forward from 1952", "",
         f"Snapshot {DATE}. Source of truth: `morphoeval/build_turing_open_problems.py` (regenerate everything with it). "
         "Problem sourcing here runs *outward* from one historical model to what the field still calls open "
         "(Krause, Gaffney, Maini & Klika 2021 theme issue; Landge et al. 2020; Brückner 2026), then prices each problem on the two axes this repo keeps separate:", "",
         "- **Resolvability (0-100)** — the deterministic Problem-Scout rubric "
         f"`{json.dumps({k: w for k, (w, _) in RUBRIC.items()})}`: can *our* tools, data in hand and a taste-free definition-of-done reach a verdict? Not truth odds.",
         f"- **Viability (%)** — a **provisional single-assessor** Delphi posterior ({ASSESSOR_LABEL}; prior {PRIOR_PERCENT:g}%, signed deciban ledger, same discipline as `predictions/round2_live_arguments.json`). "
         "NOT a multi-agent panel; a Fable/GPT/Gemini panel pass may supersede it and may not edit it. Colour rule as the graph: green only >70%.", "",
         "Problems with resolvability >= 70 (RESOLVE NOW band) enter the MorphoSwarm queue as OP-12..OP-16; all ten enter the problem graph as a band under the Thompson claims.", "",
         "| Rank | ID | Problem | Anchor | Resolvability | Band | Viability | Net dB | Verdict |", "|---|---|---|---|---|---|---|---|---|"]
    for i, p in enumerate(ranked, 1):
        v = p["viability"]
        L.append(f"| {i} | {p['id']} | {p['title']} | {p['anchor']} | **{p['resolvability']}** | {p['band']} | {v['posterior_percent']}% | {v['net_db']:+g} | {v['verdict']} |")
    L += ["", "## Each problem", ""]
    for p in ranked:
        v = p["viability"]
        L += [f"### {p['id']} — {p['title']}", "", f"*Anchor:* {p['anchor']}", "", f"**Gap.** {p['gap']}", "",
              f"**Definition of done.** {p['done']}", "", f"**Tool route.** {p['tool_route']}", "",
              f"**Deciding data.** {p['deciding_data']}", "", f"**Prior attempts.** {p['prior_attempts']}", "",
              f"**Resolvability {p['resolvability']}/100 — {p['band']}.** " + "; ".join(f"{k} {p['factors'][k]['score']}/{w}: {p['factors'][k]['why']}" for k, (w, _) in RUBRIC.items()), "",
              f"**Viability {v['posterior_percent']}% (prior {PRIOR_PERCENT:g}%, {v['for_db']:+g} dB for, {v['against_db']:+g} dB against, net {v['net_db']:+g} dB, BF {v['bayes_factor']}, confidence {v['confidence']}) — {v['verdict']}.**", "",
              f"*Hypothesis scored:* {p['hypothesis']}", ""]
        L += [f"- [{db:+g} dB] {t}" for db, t in p["evidence_for"]] + [f"- [{db:+g} dB] {t}" for db, t in p["evidence_against"]]
        L += ["", "*Sources:* " + "; ".join(f"[{PAPERS[d][1]} ({PAPERS[d][0]})](https://doi.org/{d})" for d in p["papers"]), ""]
    L += ["## Artefacts", "", f"- Ledger: `morphoeval/audits/{AUDIT_ID}.yaml`", "- Graph feed: `docs/turing-open-problems.json` (injected by `docs/build_problem_graph_html.py`)",
          "- Leaderboard input: `morphoeval/problem_scouts/scout-101.json` (ground `turing-forward`, lens `literature`)",
          f"- Nanopublications: `morphoeval/kg/{AUDIT_ID}/` (one `pk:Claim` per problem carrying a `pk:LLMClaimAssessment` for resolvability and a `pk:BayesianAssessment` for viability); trusty URIs in `published.json` after signing.", ""]
    path.write_text("\n".join(L))


def main() -> int:
    problems = [assess(p) for p in PROBLEMS]
    audits = HERE / "audits"; audits.mkdir(exist_ok=True)
    ledger = {"audit": {"id": AUDIT_ID, "date": DATE, "assessor": ASSESSOR_LABEL, "prior_percent": PRIOR_PERCENT,
                        "rubric": {k: {"weight": w, "earns": d} for k, (w, d) in RUBRIC.items()},
                        "discipline": "Resolvability totals are recomputed by problem_scout_100.score_candidate from factor evidence. Viability is a single-assessor provisional deciban ledger (prior 50%); numbers are recorded before any attempt and are never edited afterwards — a panel pass or an attempt outcome is a new record.",
                        "problems": problems}}
    (audits / f"{AUDIT_ID}.yaml").write_text(yaml.safe_dump(ledger, sort_keys=False, allow_unicode=True, width=120))
    feed = {"name": "Turing open problems — forward from 1952", "snapshot": DATE, "source": f"morphoeval/build_turing_open_problems.py · morphoeval/audits/{AUDIT_ID}.yaml",
            "kg_dir": f"morphoeval/kg/{AUDIT_ID}", "assessor": ASSESSOR_LABEL, "prior_percent": PRIOR_PERCENT,
            "op_threshold_resolvability": 70,
            "problems": [{k: p[k] for k in ("id", "slug", "anchor", "title", "gap", "done", "tool_route", "deciding_data", "prior_attempts", "papers", "resolvability", "band", "band_detail", "viability", "hypothesis", "factors")} | {"trig": trig_name(p)} for p in problems]}
    (ROOT / "docs" / "turing-open-problems.json").write_text(json.dumps(feed, indent=2, ensure_ascii=False) + "\n")
    scout = {"scout": "scout-101", "ground": "turing-forward", "lens": "literature",
             "note": "Outward-facing scout: field open problems sourced forward from Turing 1952 (not repo debt). Factors as in the 100-agent grid; the ranker recomputes totals.",
             "candidates": [{"id": p["id"], "title": p["title"], "gap": p["gap"], "done": p["done"], "tool_route": p["tool_route"],
                             "deciding_data": p["deciding_data"], "prior_attempts": p["prior_attempts"], "factors": p["factors"],
                             "viability_provisional_percent": p["viability"]["posterior_percent"]} for p in problems]}
    (HERE / "problem_scouts" / "scout-101.json").write_text(json.dumps(scout, indent=2, ensure_ascii=False) + "\n")
    write_report(problems, ROOT / "docs" / "TURING_OPEN_PROBLEMS.md")
    kd = kg_dir(); kd.mkdir(parents=True, exist_ok=True)
    for p in problems:
        (kd / trig_name(p)).write_text(build_nanopub(p))
    for p in sorted(problems, key=lambda p: -p["resolvability"]):
        print(f"{p['id']}  resolvability {p['resolvability']:3d}  {p['band']:<26s} viability {p['viability']['posterior_percent']:5.1f}%  ({p['viability']['net_db']:+g} dB)  {p['title'][:70]}")
    print(f"wrote {audits / (AUDIT_ID + '.yaml')}, docs/turing-open-problems.json, docs/TURING_OPEN_PROBLEMS.md, "
          f"problem_scouts/scout-101.json, {len(problems)} trig -> {kd}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""
Dataset-level hypothesis portfolio for the Thompson 1917 phylogenetic prediction engine.

The FishShapes blind test (docs/THOMPSON_BLIND_TEST_RESULTS.md) gave one empirical
error-vs-horizon profile.  This script turns that profile into *forward* claims about
which evolutionary-biology datasets the engine (or the same machinery re-trained on another
clade) should have predictive power on, and with what confidence, BEFORE those datasets are
compared with the predictions.  Each item carries a protocol for the species-level
sub-predictions to be registered once the data are in hand.

Two kinds of confidence are reported and are kept strictly apart:

  * simulated   - fish datasets whose target species sit on the Rabosky tree.  The tau
                  (time to nearest training species) of every species in scope is known
                  from the locked registration; per-species errors are resampled from the
                  FishShapes outcome within matching tau bins, and the registered tests are
                  re-run on each resample.  P(pass) is the fraction of resamples that pass.
  * subjective  - everything else (other traits, other clades).  A single number with the
                  reasoning written next to it.  No pretence that it is calibrated.

Outputs: thompson_dataset_portfolio.json, docs/THOMPSON_DATASET_PORTFOLIO.md,
docs/outreach_fig11_dataset_portfolio.png.
"""
from __future__ import annotations

import csv
import datetime as dt
import gzip
import json
import lzma
import subprocess
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parents[1]
REG_JSON = REPO_ROOT / "thompson_blind_predictions_registered.json"
RESULTS_JSON = REPO_ROOT / "thompson_blind_test_results.json"
TAXONOMY = REPO_ROOT / ".thompson_cache" / "PFC_taxonomy.csv.xz"
FISHBASE = REPO_ROOT / ".thompson_cache" / "fishbase_21_06"
OUT_JSON = REPO_ROOT / "thompson_dataset_portfolio.json"
OUT_MD = REPO_ROOT / "docs" / "THOMPSON_DATASET_PORTFOLIO.md"
OUT_FIG = REPO_ROOT / "docs" / "outreach_fig11_dataset_portfolio.png"
ALPHA = 0.01
B = 2000
SEED = 1917
FISH_HORIZON_MYR = 50.0
FISH_SAMPLED_CROWN_MA = 177.5
HORIZON_FRACTION = FISH_HORIZON_MYR / FISH_SAMPLED_CROWN_MA   # 0.28: the one-point rule extrapolated to other clades
P3_RATIO_TOL, P3_COV_TOL, P3_P = 0.10, 10.0, 0.6               # tolerance around the simulated deep-horizon intervals, and subjective P(pass)

# ----------------------------------------------------------------------------------------------
# Fish datasets: confidence is simulated from the FishShapes outcome, tau-bin matched
# ----------------------------------------------------------------------------------------------
FISH_DATASETS = [
    {"id": "fishshapes_v1", "short": "FishShapes v1", "name": "FishShapes v1 (Price et al. 2022) - scored", "group": "ray-finned fishes",
     "source": "Price SA et al. 2022 Ecology 103:e3829; Dryad 10.5061/dryad.vt4b8gtvf (Zenodo 6642650)", "dataset_class": "A linear calliper traits, same clade",
     "status": "scored 2026-10-22 (anchor for all simulated confidences)", "scope": "fishshapes", "traits": {"BDmax_over_SL": "Max_body_depth/SL", "CPD_over_SL": "Min_caudalpeduncle_depth/SL"},
     "trait_factor": 1.0, "n_expected": None},
    {"id": "fishbase_morphmet", "short": "FishBase MORPHMET", "name": "FishBase picture-based morphometrics (MORPHMET table)", "group": "ray-finned fishes",
     "source": "Froese R & Pauly D (eds) 2021 FishBase version 06/2021, www.fishbase.org; table MORPHMET (SL, BD, CH, HL, PDL per picture)", "dataset_class": "A linear traits, same clade, different measurement protocol",
     "status": "in hand: cached in .thompson_cache/fishbase_21_06 for unrelated analyses (run_three_dataset_validation.py); never compared with the engine; predictions locked at dac08493 before any comparison -> semi-blind",
     "scope": "fishbase_morphmet", "traits": {"BDmax_over_SL": "BD/SL", "CPD_over_SL": "CH/SL", "HL_over_SL": "HL/SL", "PD_over_SL": "PDL/SL"},
     "trait_factor": 0.9, "trait_factor_reason": "single-picture measurements are noisier than FishShapes callipers; HL and PD have no empirical anchor (factor 0.8 applied to those two)", "n_expected": None},
    {"id": "ronco2021_tanganyika_cichlids", "short": "Tanganyika cichlids (Ronco 2021)", "name": "Lake Tanganyika cichlid body-shape landmarks (Ronco et al. 2021)", "group": "ray-finned fishes",
     "source": "Ronco F et al. 2021 Nature 589:76-81 (data on Dryad); 240 species, body landmarks + oral/pharyngeal jaw data", "dataset_class": "B landmark geometric morphometrics from another lab, same clade, deep horizon",
     "status": "not downloaded", "scope": {"family": "Cichlidae"}, "traits": {"BDmax_over_SL": "from landmark coordinates", "HL_over_SL": "from landmark coordinates"},
     "trait_factor": 0.8, "trait_factor_reason": "ratios must be re-derived from a different landmark scheme", "n_expected": 150, "n_expected_reason": "240 Tanganyika species, of which roughly 150 are expected to be on the Rabosky tree under the same name"},
    {"id": "burns_sidlauskas2019_characiformes", "short": "Characiformes (Burns & Sidlauskas 2019)", "name": "Characiformes body-shape landmarks (Burns & Sidlauskas 2019)", "group": "ray-finned fishes",
     "source": "Burns MD & Sidlauskas BL 2019 Evolution 73:569-587 (data on Dryad); landmark-based body shape for on the order of 1,000 characiform species", "dataset_class": "C landmark data, same engine, beyond the horizon (Otophysi; tau > 120 Myr for essentially every species)",
     "status": "not downloaded", "scope": {"order": "Characiformes"}, "traits": {"BDmax_over_SL": "from landmark coordinates", "PD_over_SL": "from landmark coordinates"},
     "trait_factor": 0.8, "n_expected": 500, "n_expected_reason": "about half of ~1,000 digitised species expected to match Rabosky tip names"},
]

# Datasets that cannot be bootstrapped from the FishShapes profile (categorical, no landmarks for the trait)
FISH_SUBJECTIVE = [
    {"id": "fishbase_bodyshape_class", "name": "FishBase BodyShapeI categorical body-shape class", "group": "ray-finned fishes",
     "source": "FishBase SPECIES table, field BodyShapeI (elongated / fusiform-normal / short and-or deep / eel-like)", "dataset_class": "F categorical trait derived from shape",
     "status": "in hand (cached); never compared with the engine", "n_expected": 8700,
     "prediction": "ordinal class predicted from locked BDmax/SL with thresholds fitted on training species only; accuracy above the majority-class baseline (43% 'elongated') within 50 Myr",
     "p_pass": 0.80, "reasoning": "coarse 3-4 level class; the BDmax error of 0.038 within 50 Myr is far smaller than the class width (~0.1 BDmax/SL); FishBase classes are inconsistently assigned, which caps accuracy"},
    {"id": "fishbase_caudal_aspect_ratio", "name": "FishBase caudal-fin aspect ratio (SWIMMING table)", "group": "ray-finned fishes",
     "source": "FishBase SWIMMING table, field AspectRatio", "dataset_class": "G trait outside the landmark set (negative control)",
     "status": "in hand (cached); 109 target species, 21 within 50 Myr", "n_expected": 109,
     "prediction": "NO skill: the 17-landmark set has no caudal-fin landmarks, so the engine should not beat the consensus on aspect ratio (ratio >= 0.9)",
     "p_pass": 0.70, "reasoning": "peduncle depth (LM5-LM6) correlates weakly with fin aspect ratio across teleosts, so some incidental skill is possible; n is tiny"},
]

# ----------------------------------------------------------------------------------------------
# Other groups: the same machinery re-trained on that clade's tree + trait data.
# Confidence is subjective.  'blind_design' says how a genuinely blind split is possible.
# ----------------------------------------------------------------------------------------------
OTHER_GROUPS = [
    {"id": "ants_globalants", "group": "insects: ants (Formicidae)",
     "phylogeny": "Economo EP et al. 2018 Nat Commun 9:1778 (species-level tree, ~14,000 tips, grafted)", "crown_age_ma": 150,
     "traits": "GlobalAnts (Parr CL et al. 2017 Insect Conserv Divers 10:5-20): worker head width, head length, Weber's length, eye length, scape and femur length",
     "trait_type": "linear morphometrics (worker caste)", "n_species_order": "10^3",
     "blind_design": "train on species with records in the first GlobalAnts release; predict species added in later releases or in AntWeb specimen measurements",
     "p_beats_consensus": 0.85, "p_coverage_calibrated": 0.55, "p_beats_nearest": 0.35,
     "reasoning": "worker size and Weber's length are strongly phylogenetically conserved in ants; caste polymorphism (majors/minors) and inconsistent measurement protocols add noise; in-hand Camponotus GM data (Zenodo 10.5281/zenodo.18807285) and Pheidole images in .thompson_cache/antscan offer a small first test"},
    {"id": "butterflies_leptraits", "group": "insects: butterflies (Papilionoidea)",
     "phylogeny": "Kawahara AY et al. 2023 Nat Ecol Evol (2,244 species, time-calibrated)", "crown_age_ma": 100,
     "traits": "LepTraits 1.0 (Shirey V et al. 2022 Sci Data 9:382): wingspan / forewing length for 12,448 species", "trait_type": "linear size trait", "n_species_order": "10^3",
     "blind_design": "the 2,244 tree species are a small fraction of LepTraits; train on half the tree species, predict the other half, then predict species added to the tree by later studies",
     "p_beats_consensus": 0.85, "p_coverage_calibrated": 0.55, "p_beats_nearest": 0.35,
     "reasoning": "wing size has high phylogenetic signal; LepTraits values are literature-compiled ranges (noisy); essentially a one-trait test of the machinery"},
    {"id": "drosophilidae_wings", "group": "insects: Drosophilidae wings",
     "phylogeny": "Houle D, Bolstad GH, van der Linde K, Hansen TF 2017 Nature 548:447-450 (111 species, ~40 Myr)", "crown_age_ma": 40,
     "traits": "wing landmarks / semilandmarks for the same 111 species (Houle et al. 2017 supplementary data)", "trait_type": "landmark geometric morphometrics", "n_species_order": "10^2",
     "blind_design": "leave-clade-out within the 111 species; a truly blind test needs new species digitised to the same scheme",
     "p_beats_consensus": 0.75, "p_coverage_calibrated": 0.45, "p_beats_nearest": 0.30,
     "reasoning": "closest analogue to the fish engine (2-D landmarks on a time tree) and the dataset where a Thompson-style low-dimensional mode set is most likely to work; small n limits power, hence the lower probability"},
    {"id": "birds_avonet", "group": "birds",
     "phylogeny": "Jetz W et al. 2012 Nature 491:444 (9,993 species; birdtree.org)", "crown_age_ma": 110,
     "traits": "AVONET (Tobias JA et al. 2022 Ecol Lett 25:581-597): 11 linear traits (beak length/width/depth, tarsus, wing, Kipp's distance, tail, mass) for 11,009 species; 3-D beak shape PCs for 2,028 species (Cooney CR et al. 2017 Nature 542:344)",
     "trait_type": "linear morphometrics + 3-D landmark PCs", "n_species_order": "10^4",
     "blind_design": "train the engine on the 2,028 Cooney beak-shape species, predict AVONET beak length/width/depth for the remaining ~9,000 species (independent measurements, independent team)",
     "p_beats_consensus": 0.95, "p_coverage_calibrated": 0.60, "p_beats_nearest": 0.40,
     "reasoning": "best-case group: complete tree, complete trait coverage, museum callipers, lambda ~ 0.9 for most AVONET traits; the beak-shape -> beak-calliper design mirrors the fish landmark -> calliper design exactly"},
    {"id": "mammals_pantheria_combine", "group": "mammals",
     "phylogeny": "Upham NS et al. 2019 PLoS Biol 17:e3000494 (5,911 species)", "crown_age_ma": 180,
     "traits": "PanTHERIA (Jones KE et al. 2009 Ecology 90:2648) -> COMBINE (Soria CD et al. 2021 Ecology 102:e03344): body mass, head-body length, and life history; cranial 3-D landmarks for 322 species (Goswami A et al. 2022 Science 378:377)",
     "trait_type": "linear size traits + 3-D cranial landmarks", "n_species_order": "10^3",
     "blind_design": "train on PanTHERIA (2009) species, predict species whose first measurement appears only in COMBINE (2021): genuinely blind on species not yet measured when the training compilation was made",
     "p_beats_consensus": 0.90, "p_coverage_calibrated": 0.55, "p_beats_nearest": 0.35,
     "reasoning": "body mass is among the most conserved traits known (lambda > 0.95); the two-compilation design gives a clean blind split; COMBINE imputes some values and those must be excluded"},
    {"id": "squamates_meiri", "group": "squamate reptiles",
     "phylogeny": "Tonini JFR et al. 2016 Biol Conserv 204:23 (9,755 species)", "crown_age_ma": 200,
     "traits": "Meiri S 2018 Glob Ecol Biogeogr 27:1168 'Traits of lizards of the world': maximum SVL, limb/leg state, clutch size, habitat", "trait_type": "linear size + categorical", "n_species_order": "10^3",
     "blind_design": "train on lizards, predict snakes' SVL? No - different body plan. Train on 2018 release, predict species added in the 2024 update of the same database",
     "p_beats_consensus": 0.85, "p_coverage_calibrated": 0.50, "p_beats_nearest": 0.35,
     "reasoning": "SVL is conserved but repeated limb loss and body elongation across squamates make shape far less predictable than size; a good place to test the horizon rule because the clade is old"},
    {"id": "amphibians_amphibio", "group": "amphibians",
     "phylogeny": "Jetz W & Pyron RA 2018 Nat Ecol Evol 2:850 (7,238 species)", "crown_age_ma": 300,
     "traits": "AmphiBIO (Oliveira BF et al. 2017 Sci Data 4:170123): body size, reproductive and ecological traits", "trait_type": "linear size + categorical", "n_species_order": "10^3",
     "blind_design": "train on AmphiBIO v1, predict species added in later releases",
     "p_beats_consensus": 0.80, "p_coverage_calibrated": 0.50, "p_beats_nearest": 0.30,
     "reasoning": "size conserved at the family level, but miniaturisation is frequent and convergent; the 300 Myr crown implies, under the 0.28 rule, a horizon of ~85 Myr, which is a strong and testable claim"},
    {"id": "plants_diaz_try", "group": "seed plants",
     "phylogeny": "Smith SA & Brown JW 2018 Am J Bot 105:302 (seed-plant megatree, ~80,000 tips)", "crown_age_ma": 330,
     "traits": "Diaz S et al. 2016 Nature 529:167 (six traits: height, seed mass, LMA, leaf area, leaf N, stem density; ~46,000 species) and TRY (Kattge J et al. 2020 Glob Change Biol 26:119)",
     "trait_type": "linear functional traits", "n_species_order": "10^4",
     "blind_design": "train on Diaz et al. 2016 species, predict TRY-only species added after 2016",
     "p_beats_consensus": 0.70, "p_coverage_calibrated": 0.45, "p_beats_nearest": 0.30,
     "reasoning": "seed mass and height have high phylogenetic signal, leaf economics traits much less (lambda ~ 0.5-0.7); plasticity and intraspecific variance are large; the lowest-confidence group on the list"},
]

# Portfolio-level hypotheses: cross-dataset claims registered now
PORTFOLIO_HYPOTHESES = [
    {"id": "P1", "primary": True, "statement": "For any dataset of linear body-proportion traits on ray-finned fishes (class A), the locked full-30 predictions beat the consensus (ratio < 0.8, paired Wilcoxon p < 0.01) for species within 50 Myr of a training species.",
     "confidence": "simulated per dataset below"},
    {"id": "P2", "primary": True, "statement": "The nearest-relative margin does NOT hold up: on class A fish datasets the full-30 prediction improves on copying the nearest training species by less than 10% (H2-type test fails again).",
     "confidence": "simulated per dataset below (reported as P(H2 passes); P2 is supported if it does not)"},
    {"id": "P3", "primary": True, "statement": "Beyond 50 Myr the behaviour is bin-specific, not uniform: the realised BDmax error ratio vs consensus and the realised 95% coverage of each deep-horizon dataset fall inside the simulated intervals given below (ratio +/- 0.10, coverage +/- 10 points). Concretely: cichlids (80-120 Myr) ~0.77 with ~68% coverage; characiforms (>120 Myr, root-state predictions) ~0.43 with ~82% coverage.",
     "confidence": f"subjective {P3_P}: intervals are simulated, but the tolerance and the transfer across landmark schemes are judgement calls"},
    {"id": "P4", "primary": False, "statement": "Head-length and predorsal ratios (HL/SL, PD/SL) show weaker skill than BDmax/SL on the same species (error ratio vs consensus at least 0.1 higher).",
     "confidence": "subjective 0.6: the jaw/head module had a worse process fit than the body module in the engine; FishBase picture measurements of HL are noisy"},
    {"id": "P5", "primary": False, "statement": "Traits outside the landmark set (caudal-fin aspect ratio) show no skill (ratio vs consensus >= 0.9).", "confidence": "subjective 0.7"},
    {"id": "P6", "primary": True, "statement": "The same machinery (AICc-selected process model on a time tree, Gaussian conditional prediction, horizon-binned calibration) re-trained on another clade beats that clade's consensus within the horizon for every group listed.",
     "confidence": "subjective, per group (p_beats_consensus); product over groups is the probability all succeed"},
    {"id": "P7", "primary": False, "statement": f"The horizon beyond which 95% coverage collapses below 75% falls at {HORIZON_FRACTION:.2f} +/- 50% of the clade's sampled crown age (fish: 50 of 177.5 Myr) in every group tested.",
     "confidence": "subjective 0.4: one-point extrapolation; it is registered because it is cheap to falsify"},
    {"id": "P8", "primary": False, "statement": "Where a Thompson-style low-dimensional a-priori mode set exists (Drosophila wings; bird beaks), it beats the consensus but is worse than the full shape space, as in fish (H6).",
     "confidence": "subjective 0.65"},
]

SUBPREDICTION_PROTOCOL = {
    "step_1": "Before downloading or opening a dataset: if its traits are already in the locked registration (BDmax, CPD, HL, PD over SL for all 10,888 Rabosky targets, commit dac08493) nothing new is generated; otherwise generate species-level predictions, SHA-256 them, commit, and publish a preregistration nanopub citing this portfolio nanopub.",
    "step_2": "Download; record checksum; match species by exact name; per-species median over specimens/pictures.",
    "step_3": "Calibrate the measurement convention with a linear map fitted on TRAINING species only (as in the FishShapes protocol); apply to all methods.",
    "step_4": "Score exactly the registered tests (paired one-sided Wilcoxon, alpha = 0.01; tau bins 0-5, 5-10, 10-25, 25-50, 50-80, 80-120, >120 Myr as registered; 95% coverage).",
    "step_5": "Publish an outcome nanopub citing the preregistration and this portfolio; update the realised columns of docs/THOMPSON_DATASET_PORTFOLIO.md without changing the registered probabilities.",
    "for_other_clades": "Fit the engine on the training split of that clade's tree, lock predictions for every other tip, then obtain the comparison values. Horizon bins are scaled to the clade's crown age.",
}


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def load_scopes(preds: dict) -> dict:
    """Return {scope_key: [species]} for each fish dataset."""
    order_of = {}
    with lzma.open(TAXONOMY, "rt") as fh:
        for row in csv.DictReader(fh):
            order_of[row["genus.species"].replace(" ", "_")] = row["order"]
    scopes = {}
    # FishBase morphmet: names only (which species have a usable record) - values are not read here
    code_to_name = {}
    with gzip.open(FISHBASE / "species.tsv.gz", "rt", encoding="utf-8", errors="replace") as fh:
        for r in csv.DictReader(fh, delimiter="\t"):
            code_to_name[r["SpecCode"]] = f'{r["Genus"]}_{r["Species"]}'
    have = set()
    with gzip.open(FISHBASE / "morphmet.tsv.gz", "rt", encoding="utf-8", errors="replace") as fh:
        for r in csv.DictReader(fh, delimiter="\t"):
            nm = code_to_name.get(r["Speccode"])
            try:
                if nm and float(r["SL"]) > 0 and float(r["BD"]) > 0:
                    have.add(nm)
            except ValueError:
                pass
    scopes["fishbase_morphmet"] = sorted(nm for nm in have if nm in preds)
    for d in FISH_DATASETS:
        sc = d["scope"]
        if isinstance(sc, dict):
            if "family" in sc:
                scopes[d["id"]] = sorted(nm for nm, p in preds.items() if p["family_from_taxonomy"] == sc["family"])
            else:
                scopes[d["id"]] = sorted(nm for nm in preds if order_of.get(nm) == sc["order"])
    return scopes


def simulate(taus: np.ndarray, pools: dict, bins: list, n: int, rng: np.random.Generator) -> dict:
    """Resample n species by tau bin from the FishShapes per-species error pools; run the registered tests."""
    bin_of = np.array([next(i for i, (lo, hi) in enumerate(bins) if lo <= t < hi) for t in taus])
    out = defaultdict(list)
    for _ in range(B):
        idx = rng.integers(0, len(taus), n)
        bsel = bin_of[idx]
        recs = []
        for b in np.unique(bsel):
            pool = pools[b]
            k = int(np.sum(bsel == b))
            recs.append(pool[rng.integers(0, len(pool), k)])
        R = np.concatenate(recs)                      # columns: tau, err_full30, err_consensus, err_nearest, in95
        near = R[R[:, 0] < FISH_HORIZON_MYR]
        far = R[R[:, 0] >= FISH_HORIZON_MYR]
        if len(near) >= 10:
            ef, ec, en = near[:, 1], near[:, 2], near[:, 3]
            r1 = np.median(ef) / np.median(ec); r2 = np.median(ef) / np.median(en)
            d1 = ef - ec; d1 = d1[d1 != 0]; d2 = ef - en; d2 = d2[d2 != 0]
            p1 = stats.wilcoxon(d1, alternative="less").pvalue if len(d1) >= 10 else 1.0
            p2 = stats.wilcoxon(d2, alternative="less").pvalue if len(d2) >= 10 else 1.0
            out["ratio_consensus_near"].append(r1); out["ratio_nearest_near"].append(r2)
            out["pass_H1"].append(p1 < ALPHA and r1 < 0.8); out["pass_H2"].append(p2 < ALPHA and r2 < 1.0)
            cov = 100 * near[:, 4].mean(); out["coverage_near"].append(cov); out["pass_H3"].append(85 <= cov <= 97)
        if len(far) >= 10:
            rf = np.median(far[:, 1]) / np.median(far[:, 2])
            out["ratio_consensus_far"].append(rf); out["coverage_far"].append(100 * far[:, 4].mean())
    res = {}
    for k, v in out.items():
        v = np.array(v, dtype=float)
        if k.startswith("pass_"):
            res[k.replace("pass_", "p_")] = float(v.mean()) if len(v) else None
        else:
            res[k] = {"median": float(np.median(v)), "q05": float(np.quantile(v, 0.05)), "q95": float(np.quantile(v, 0.95))} if len(v) else None
    return res


def main() -> int:
    rng = np.random.default_rng(SEED)
    reg = json.loads(REG_JSON.read_text())
    preds = {p["species"]: p for p in reg["predictions"]}
    bins = [tuple(b) for b in reg["tau_bins_myr"]]
    res = json.loads(RESULTS_JSON.read_text())
    k = "BDmax_over_SL"
    rows = [r for r in res["per_species"] if k in r]
    pools = defaultdict(list)
    for r in rows:
        b = next(i for i, (lo, hi) in enumerate(bins) if lo <= r["tau"] < hi)
        pools[b].append([r["tau"], r[k]["err_full30_cal"], r[k]["err_consensus_cal"], r[k]["err_nearest_cal"], float(r[k]["in95_full30_cal"])])
    pools = {b: np.array(v) for b, v in pools.items()}
    print(f"FishShapes pools per tau bin: { {bins[b]: len(v) for b, v in pools.items()} }")

    scopes = load_scopes(preds)
    items = []
    for d in FISH_DATASETS:
        d = dict(d)
        if d["scope"] == "fishshapes":
            H = res["hypotheses"]
            d["scope_n"] = res["n_matched"]; d["n_used"] = res["n_matched"]
            d["tau_summary"] = {"frac_within_50": float(np.mean([r["tau"] < 50 for r in rows])), "median_tau": float(np.median([r["tau"] for r in rows]))}
            d["realised"] = {"ratio_consensus_near": H["H1_cal"]["ratio"], "ratio_nearest_near": H["H2_cal"]["ratio"], "coverage_near": H["H3_cal"]["coverage95_pct"],
                             "ratio_consensus_far": H["H4_cal"]["ratio"], "H1": H["H1_cal"]["pass"], "H2": H["H2_cal"]["pass"], "H3": H["H3_cal"]["pass"]}
            d["confidence_type"] = "realised"
            items.append(d); continue
        sp = scopes[d["id"]]
        taus = np.array([preds[nm]["tau_myr"] for nm in sp])
        n = d["n_expected"] or len(sp)
        d["scope_n"] = len(sp); d["n_used"] = int(n)
        d["tau_summary"] = {"frac_within_50": float(np.mean(taus < 50)) if len(taus) else None, "median_tau": float(np.median(taus)) if len(taus) else None,
                            "n_within_50_expected": int(round(n * float(np.mean(taus < 50)))) if len(taus) else 0}
        sim = simulate(taus, pools, bins, int(n), rng)
        d["simulated"] = sim
        tf = d["trait_factor"]
        d["p_pass_registered"] = {}
        for trait in d["traits"]:
            f = tf * (0.8 / 0.9 if trait in ("HL_over_SL", "PD_over_SL") else 1.0)
            d["p_pass_registered"][trait] = {"P1_beats_consensus_within_50": (sim.get("p_H1") or 0.0) * f if sim.get("p_H1") is not None else None,
                                             "P2_H2_passes": (sim.get("p_H2") or 0.0) * f if sim.get("p_H2") is not None else None,
                                             "coverage_calibrated_within_50": (sim.get("p_H3") or 0.0) * f if sim.get("p_H3") is not None else None}
        if sim.get("ratio_consensus_far"):
            rf, cf = sim["ratio_consensus_far"], sim["coverage_far"]
            d["P3_deep_prediction"] = {"ratio_consensus_interval": [round(rf["q05"] - P3_RATIO_TOL, 2), round(rf["q95"] + P3_RATIO_TOL, 2)],
                                       "coverage_interval_pct": [round(cf["q05"] - P3_COV_TOL, 0), round(cf["q95"] + P3_COV_TOL, 0)],
                                       "pass_rule": "realised BDmax ratio vs consensus AND realised 95% coverage for species >= 50 Myr both fall inside these intervals", "p_pass_subjective": P3_P}
        d["confidence_type"] = "simulated (tau-bin resampling of the FishShapes BDmax outcome) x trait factor"
        print(f"{d['id']}: scope {len(sp)}, n_used {n}, within-50 {d['tau_summary']['frac_within_50']:.2f}, sim {json.dumps({k_: (round(v, 3) if isinstance(v, float) else v) for k_, v in sim.items() if k_.startswith('p_')})}")
        items.append(d)

    for d in OTHER_GROUPS:
        d = dict(d); d["predicted_horizon_myr"] = round(HORIZON_FRACTION * d["crown_age_ma"], 0); d["confidence_type"] = "subjective"

    portfolio = {"registered_on": dt.date.today().isoformat(), "git_commit_at_generation": git_commit(), "script": "thompson/register_dataset_portfolio.py",
                 "anchor": {"dataset": "FishShapes v1 blind test", "results_file": "thompson_blind_test_results.json", "preregistration_nanopub": "RAilN4nbRHp3e3ZHqlbIgeL3UGPRV_KtQHtJD3gzVVdqw", "outcome_nanopub_prod": "RAJancwGOPEkHzBoZXGv_vu_snW0Lff2FktE85Me1P-fI"},
                 "horizon_rule": {"fish_horizon_myr": FISH_HORIZON_MYR, "fish_sampled_crown_ma": FISH_SAMPLED_CROWN_MA, "fraction": HORIZON_FRACTION},
                 "simulation": {"B": B, "seed": SEED, "alpha": ALPHA, "pools": "FishShapes per-species calibrated BDmax errors, tau-bin matched"},
                 "fish_datasets": items, "fish_subjective": FISH_SUBJECTIVE,
                 "other_groups": [dict(d, predicted_horizon_myr=round(HORIZON_FRACTION * d["crown_age_ma"]), confidence_type="subjective") for d in OTHER_GROUPS],
                 "portfolio_hypotheses": PORTFOLIO_HYPOTHESES, "subprediction_protocol": SUBPREDICTION_PROTOCOL}
    OUT_JSON.write_text(json.dumps(portfolio, indent=1), encoding="utf-8")
    print(f"wrote {OUT_JSON}")
    write_markdown(portfolio)
    make_figure(portfolio, preds, scopes)
    return 0


def pct(x):
    return "-" if x is None else f"{100 * x:.0f}%"


def write_markdown(P: dict) -> None:
    L = ["# Thompson 1917 prediction engine: dataset portfolio and forward confidence", "",
         f"Registered {P['registered_on']} at commit `{P['git_commit_at_generation'][:12]}` by `{P['script']}`. Anchor: the FishShapes v1 blind test "
         f"(preregistration nanopub `{P['anchor']['preregistration_nanopub']}`, outcome `{P['anchor']['outcome_nanopub_prod']}`).", "",
         "This document states, before the comparisons are made, which datasets the engine should and should not have predictive power on, and how confident we are. "
         "**Simulated** probabilities come from resampling the FishShapes per-species errors in matching horizon bins (tau = time to nearest training species) for the species actually in scope on the Rabosky tree, then re-running the registered tests "
         f"({P['simulation']['B']} resamples, seed {P['simulation']['seed']}). **Subjective** probabilities are exactly that; the reasoning is given beside each. "
         "Realised columns will be filled in by the scoring scripts; the registered probabilities are never edited.", "",
         "## 1. Ray-finned fishes (the trained engine; predictions already locked at `dac08493`)", "",
         "| Dataset | Class | Status | Species in scope / used | Within 50 Myr | Pred. error ratio vs consensus (<50 Myr) | vs nearest | 95% coverage | P(beats consensus) | P(beats nearest) | P(coverage 85-97%) | Realised |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for d in P["fish_datasets"]:
        ts = d["tau_summary"]
        if d.get("realised"):
            r = d["realised"]
            L.append(f"| {d['name']} | {d['dataset_class']} | {d['status']} | {d['scope_n']} | {100 * ts['frac_within_50']:.0f}% | {r['ratio_consensus_near']:.2f} | {r['ratio_nearest_near']:.2f} | {r['coverage_near']:.0f}% | - | - | - | "
                     f"H1 {'pass' if r['H1'] else 'fail'}, H2 {'pass' if r['H2'] else 'fail'}, H3 {'pass' if r['H3'] else 'fail'}; far ratio {r['ratio_consensus_far']:.2f} |")
            continue
        s = d["simulated"]
        def iv(key, fmt="{:.2f}", suffix=""):
            v = s.get(key)
            return "-" if not v else f"{fmt.format(v['median'])}{suffix} [{fmt.format(v['q05'])}, {fmt.format(v['q95'])}]"
        pp = d["p_pass_registered"]["BDmax_over_SL"]
        L.append(f"| {d['name']} | {d['dataset_class']} | {d['status']} | {d['scope_n']} / {d['n_used']} | {100 * ts['frac_within_50']:.0f}% | {iv('ratio_consensus_near')} | {iv('ratio_nearest_near')} | "
                 f"{iv('coverage_near', '{:.0f}', '%')} | {pct(pp['P1_beats_consensus_within_50'])} | {pct(pp['P2_H2_passes'])} | {pct(pp['coverage_calibrated_within_50'])} | pending |")
    L += ["", "Deep-horizon profile (species >= 50 Myr from any training species), simulated. The P3 pass rule is that the realised ratio and coverage both land inside the widened intervals:", "",
          "| Dataset | n >= 50 Myr (of used) | Pred. error ratio vs consensus | Pred. 95% coverage | P3 pass interval (ratio; coverage) | P(pass), subjective |", "|---|---|---|---|---|---|"]
    for d in P["fish_datasets"]:
        s = d.get("simulated")
        if not s or not s.get("ratio_consensus_far"):
            continue
        f = s["ratio_consensus_far"]; c = s["coverage_far"]; p3 = d["P3_deep_prediction"]
        L.append(f"| {d['name']} | {d['n_used'] - d['tau_summary']['n_within_50_expected']} | {f['median']:.2f} [{f['q05']:.2f}, {f['q95']:.2f}] | {c['median']:.0f}% [{c['q05']:.0f}, {c['q95']:.0f}] | "
                 f"{p3['ratio_consensus_interval'][0]:.2f}-{p3['ratio_consensus_interval'][1]:.2f}; {p3['coverage_interval_pct'][0]:.0f}-{p3['coverage_interval_pct'][1]:.0f}% | {pct(p3['p_pass_subjective'])} |")
    L += ["", "Per-trait probabilities (trait factor applied; HL and PD carry an extra 0.8/0.9 because they have no empirical anchor):", "", "| Dataset | Trait | P(beats consensus < 50 Myr) | P(H2 passes) | P(coverage calibrated) |", "|---|---|---|---|---|"]
    for d in P["fish_datasets"]:
        for t, pp in d.get("p_pass_registered", {}).items():
            L.append(f"| {d['id']} | {t} ({d['traits'][t]}) | {pct(pp['P1_beats_consensus_within_50'])} | {pct(pp['P2_H2_passes'])} | {pct(pp['coverage_calibrated_within_50'])} |")
    L += ["", "Fish datasets without a simulated anchor (subjective):", "", "| Dataset | Class | Status | n | Prediction | P | Reasoning |", "|---|---|---|---|---|---|---|"]
    for d in P["fish_subjective"]:
        L.append(f"| {d['name']} | {d['dataset_class']} | {d['status']} | {d['n_expected']} | {d['prediction']} | {pct(d['p_pass'])} | {d['reasoning']} |")
    L += ["", "## 2. Other groups (same machinery, re-trained; all probabilities subjective)", "",
          f"The horizon rule being tested (P7): coverage collapses at about {P['horizon_rule']['fraction']:.2f} x the clade's sampled crown age (fish: {P['horizon_rule']['fish_horizon_myr']:.0f} of {P['horizon_rule']['fish_sampled_crown_ma']:.1f} Myr).", "",
          "| Group | Phylogeny | Trait data | Trait type | n (order) | Crown (Ma) | Pred. horizon (Myr) | Blind design | P(beats consensus) | P(coverage 85-97%) | P(beats nearest) |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for d in P["other_groups"]:
        L.append(f"| {d['group']} | {d['phylogeny']} | {d['traits']} | {d['trait_type']} | {d['n_species_order']} | ~{d['crown_age_ma']} | ~{d['predicted_horizon_myr']:.0f} | {d['blind_design']} | {pct(d['p_beats_consensus'])} | {pct(d['p_coverage_calibrated'])} | {pct(d['p_beats_nearest'])} |")
    L += ["", "Reasoning per group:", ""]
    for d in P["other_groups"]:
        L.append(f"- **{d['group']}** - {d['reasoning']}")
    prod = float(np.prod([d["p_beats_consensus"] for d in P["other_groups"]]))
    L += ["", f"Joint probability that *every* listed group beats its consensus within the horizon (P6, product of the subjective numbers): {100 * prod:.0f}%. "
          f"Expected number of groups succeeding: {sum(d['p_beats_consensus'] for d in P['other_groups']):.1f} of {len(P['other_groups'])}.", "",
          "## 3. Portfolio-level hypotheses (registered now)", "", "| ID | Primary | Statement | Confidence |", "|---|---|---|---|"]
    for h in P["portfolio_hypotheses"]:
        L.append(f"| {h['id']} | {'yes' if h['primary'] else 'no'} | {h['statement']} | {h['confidence']} |")
    L += ["", "## 4. Sub-prediction protocol (applies to every item)", ""]
    for k_, v in P["subprediction_protocol"].items():
        L.append(f"- **{k_}**: {v}")
    L += ["", "## 5. What this is and is not", "",
          "- The simulated probabilities assume the FishShapes error profile transfers across measurement protocols for the same trait on the same tree. That is the assumption being tested; if FishBase MORPHMET comes out far from the simulated interval, the simulation itself is falsified.",
          "- The subjective probabilities for other groups are priors, written down so they can be scored (Brier) once the groups are tested. They are deliberately not dressed up as outputs of a model.",
          "- Nothing in this document is updated after a dataset is scored except the 'Realised' column.", ""]
    OUT_MD.write_text("\n".join(L), encoding="utf-8")
    print(f"wrote {OUT_MD}")


def make_figure(P: dict, preds: dict, scopes: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(19, 5.6), gridspec_kw={"width_ratios": [1.1, 1.0, 1.2]})
    ax = axes[0]
    edges = np.array([0, 5, 10, 25, 50, 80, 120, 180])
    for d, c in zip([x for x in P["fish_datasets"] if x["id"] != "fishshapes_v1"], ["#2ca02c", "#9467bd", "#d62728"]):
        taus = np.array([preds[nm]["tau_myr"] for nm in scopes[d["id"]]])
        ax.hist(np.clip(taus, 0, 179), bins=edges, histtype="step", lw=2, color=c, label=f"{d['short']} (n={len(taus)})", density=True)
    fs = np.array([r["tau"] for r in json.loads(RESULTS_JSON.read_text())["per_species"]])
    ax.hist(np.clip(fs, 0, 179), bins=edges, color="0.75", alpha=0.6, density=True, label=f"FishShapes v1, scored (n={len(fs)})")
    ax.axvline(50, color="k", ls="--", lw=1); ax.text(52, ax.get_ylim()[1] * 0.92, "50 Myr horizon", fontsize=9)
    ax.set_xlabel("tau: time to nearest training species (Myr)"); ax.set_ylabel("density"); ax.set_title("A. Where the target species sit relative to the horizon", fontsize=11, loc="left")
    ax.legend(fontsize=8, loc="upper right")

    ax = axes[1]
    labels, med, lo, hi, cols = [], [], [], [], []
    for d in P["fish_datasets"]:
        if d.get("realised"):
            labels.append("FishShapes v1 (realised)"); med.append(d["realised"]["ratio_consensus_near"]); lo.append(np.nan); hi.append(np.nan); cols.append("0.3")
        elif d["simulated"].get("ratio_consensus_near"):
            s = d["simulated"]["ratio_consensus_near"]; labels.append(d["short"] + " (<50 Myr)"); med.append(s["median"]); lo.append(s["q05"]); hi.append(s["q95"]); cols.append("#2ca02c")
        if d.get("simulated") and d["simulated"].get("ratio_consensus_far"):
            s = d["simulated"]["ratio_consensus_far"]; labels.append(d["short"] + " (>=50 Myr)"); med.append(s["median"]); lo.append(s["q05"]); hi.append(s["q95"]); cols.append("#d62728")
    y = np.arange(len(labels))
    for i in range(len(labels)):
        if not np.isnan(lo[i]):
            ax.plot([lo[i], hi[i]], [y[i], y[i]], color=cols[i], lw=3, alpha=0.5)
        ax.plot(med[i], y[i], "o", color=cols[i], ms=8)
    ax.axvline(1.0, color="k", lw=1); ax.axvline(0.8, color="k", ls=":", lw=1); ax.text(0.79, -0.6, "pass threshold 0.8", fontsize=8, ha="right")
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=8); ax.invert_yaxis(); ax.set_xlim(0.2, 1.1)
    ax.set_xlabel("median |error| ratio: full-30 prediction / consensus"); ax.set_title("B. Predicted error ratio (median, 90% interval)", fontsize=11, loc="left")

    ax = axes[2]
    groups = P["other_groups"]
    names = [g["group"] for g in groups]
    y = np.arange(len(names))
    ax.barh(y - 0.22, [g["p_beats_consensus"] for g in groups], height=0.2, color="#2ca02c", label="beats consensus within horizon")
    ax.barh(y, [g["p_coverage_calibrated"] for g in groups], height=0.2, color="#1f77b4", label="95% interval coverage 85-97%")
    ax.barh(y + 0.22, [g["p_beats_nearest"] for g in groups], height=0.2, color="#ff7f0e", label="beats nearest relative")
    for i, g in enumerate(groups):
        ax.text(1.02, i, f"horizon ~{g['predicted_horizon_myr']:.0f} Myr", va="center", fontsize=8, color="0.3")
    ax.set_yticks(y); ax.set_yticklabels(names, fontsize=8); ax.invert_yaxis(); ax.set_xlim(0, 1.0)
    ax.set_xlabel("subjective probability (registered before testing)"); ax.set_title("C. Other groups: re-trained engine, prior confidence", fontsize=11, loc="left")
    ax.legend(fontsize=8, loc="lower right")
    fig.suptitle("Forward portfolio: which datasets the Thompson prediction engine should work on, registered before the comparisons", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(OUT_FIG, dpi=150)
    print(f"wrote {OUT_FIG}")


if __name__ == "__main__":
    raise SystemExit(main())

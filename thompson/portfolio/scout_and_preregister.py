#!/usr/bin/env python3
"""
Stage 1 & 2 of the Overnight Thompson Scout-and-Test Pipeline:
1. Registers the scouted public datasets (Zenodo / Dryad / OpenTree) and their file SHA-256 digests.
2. Locks pre-registered hypotheses and calibrated subjective probabilities (Brier-scored)
   in `thompson/portfolio/groups/<gid>_hypotheses.json` BEFORE running the Thompson-7 mode
   decomposition or leave-one-species-out phylogenetic prediction engine.
3. Writes `thompson/portfolio/scout_preregistration_manifest.json` pinning the SHA-256 of every
   hypothesis file prior to scoring.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
GROUPS_DIR = REPO_ROOT / "thompson" / "portfolio" / "groups"
MANIFEST_PATH = REPO_ROOT / "thompson" / "portfolio" / "scout_preregistration_manifest.json"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
    except Exception:
        return "unknown"


SCOUTED_HYPOTHESES = {
    "drosophilidae_wings": {
        "group": "insects: Drosophilidae wings (Houle et al. 2017 Nature: 117 species, 21,138 species wings, 17,323 G-matrix wings, 12,075 M-matrix wings; 12 landmarks)",
        "registered_on": dt.date.today().isoformat(),
        "portfolio_id": "drosophilidae_wings",
        "data": {
            "landmarks": "Houle D, Bolstad GH, van der Linde K, Hansen TF 2017 Nature 548:447-450; Zenodo 4999258 (`NatureDryad.zip`: `SpeciesNature.csv` 12 2D wing vein landmarks across 21,138 wings from 117 Drosophilidae species, plus `AllAlignedNature.csv` with 17,323 additive genetic Gw wings and 12,075 mutational M wings).",
            "phylogeny": "Open Tree of Life induced subtree (`datasets/thompson/drosophilidae/opentree_induced_subtree.json`, 88 matched species with Grafen branch lengths).",
        },
        "what_thompson_specifically_adds": "Houle et al. (2017) famously showed that spontaneous mutation (M) and standing genetic variance (G) predict 40 Myr of drosophilid wing evolution (D). Applying Thompson's 7-mode basis tests whether (1) drosophilid wing venation has a higher Thompson-7 variance share than Hymenoptera wings (~40%) because the 12 landmarks span a simpler dipteran vein grid, and (2) mutational (M) and genetic (G) covariance matrices channel macroevolutionary divergence (D) preferentially along the 7 global Thompson coordinate modes.",
        "hypotheses": [
            {
                "id": "DW1",
                "primary": True,
                "statement": "Across the 117 drosophilid species means (12 2D landmarks), the 7 a-priori Thompson coordinate modes capture > 50% of between-species Procrustes variance (exceeding the 38.8% hornet and 42.2% bee forewing benchmarks).",
                "p": 0.80,
                "reasoning": "With 12 landmarks (20 Procrustes DOF vs 32-34 in Hymenoptera), 7 smooth coordinate modes span a larger fraction of shape space and capture the primary proximal-distal radio-medial vein shifts.",
            },
            {
                "id": "DW2",
                "primary": True,
                "statement": "Interior crossvein landmarks (the 4 middle landmarks along the proximal-distal axis) have a higher unexplained residual fraction after Thompson-7 than the 8 proximal/apical margin landmarks.",
                "p": 0.75,
                "reasoning": "In both hornets and bees, interior crossvein intersections shifted locally relative to the global wing outline; dipteran anterior/posterior crossveins should show the same local residual concentration.",
            },
            {
                "id": "DW3",
                "primary": True,
                "statement": "Phylogenetic signal on the 88-species OpenTree topology: best multivariate evolutionary model in full PCA-95% space is NOT White Noise (Pagel's lambda > 0.3).",
                "p": 0.75,
                "reasoning": "With 88 species across Drosophilidae, subgeneric and species-group clustering on the OpenTree topology is much better resolved than the 30-species Vespa polytomy.",
            },
            {
                "id": "DW4",
                "primary": True,
                "statement": "Leave-one-species-out phylogenetic prediction in full shape space beats the grand consensus (median Procrustes RMS ratio < 0.95, paired one-sided Wilcoxon p < 0.01).",
                "p": 0.75,
                "reasoning": "Matches the 0.75 subjective prior registered for `drosophilidae_wings` in `thompson_dataset_portfolio.json`.",
            },
            {
                "id": "DW5",
                "primary": False,
                "statement": "Mutational-macroevolutionary alignment (Houle M vs D): in the 20-PC space of `AllAlignedNature.csv`, the leading axis of spontaneous mutational variance (M, n=12,075) and standing genetic variance (Gw, n=17,323) both align strongly with between-species macroevolutionary divergence (D, n=21,138), with |cos theta| > 0.50.",
                "p": 0.85,
                "reasoning": "Direct check of Houle et al.'s (2017) core finding that mutational and genetic lines of least resistance align with macroevolutionary divergence.",
            },
        ],
    },
    "pheidole_ants": {
        "group": "insects: Pheidole big-headed ants (Friedman et al. 2019: 651 minor heads, 502 major heads [23 LMs], minor & major mesosomas [10 LMs] across ~200 species on Pheidole_Tree.phy)",
        "registered_on": dt.date.today().isoformat(),
        "portfolio_id": "ants_globalants",
        "data": {
            "landmarks": "Friedman NR, Bennett BL, Fischer G, Sarnat EM, Huang JP, Knowles LL, Economo EP 2020 Syst Biol / Dryad (Zenodo 5096592): 23-LM head shape (`Aligned_Head_Minor.TPS`, `Aligned_Head_Major.TPS`) and 10-LM mesosoma profile (`Pheidole_minors_Mesosoma.tps`, `Pheidole_majors_Mesosoma.tps`), mapped via `Pheidole_Morphology_Data_03282019.csv`.",
            "phylogeny": "Time-calibrated phylogeny `Pheidole_Tree.phy` (447 tips; ~200 matched species with landmark data).",
        },
        "what_thompson_specifically_adds": "Provides a direct independent replication of our Camponotus carpenter-ant caste discovery in the most famously dimorphic ant genus on Earth (*Pheidole*, which Thompson himself cited on p. 732 for the ovate-to-cordate soldier head transformation!). Tests whether within-species minor -> major soldier polyphenism and between-species radiation follow the 7 Thompson modes across ~200 species.",
        "hypotheses": [
            {
                "id": "PH1",
                "primary": True,
                "statement": "Between-species 23-LM head shape in both minor workers and major soldiers is strongly Thompsonian: Thompson-7 captures > 60% of interspecific Procrustes variance across the ~200 Pheidole species.",
                "p": 0.85,
                "reasoning": "In Camponotus (82 LMs), Thompson-7 captured 81.2% of interspecific frontal-head variance; Pheidole's 23 head landmarks trace the same continuous bilobed occipital/lateral capsule plus eye/mandible anchors.",
            },
            {
                "id": "PH2",
                "primary": True,
                "statement": "Within-species minor -> major soldier caste transformation vectors on the 23-LM head are well-captured by Thompson's 7 global coordinate modes (> 55% of pooled caste-shift variance).",
                "p": 0.85,
                "reasoning": "Soldier head hypertrophy in Pheidole is the textbook juvenile-hormone growth gradient (ovate minor -> cordate/bilobed major occipital expansion).",
            },
            {
                "id": "PH3",
                "primary": True,
                "statement": "Developmental channelling (G_caste vs R_species): the principal axis of within-species minor->major caste divergence aligns more strongly with the principal axis of between-species macroevolutionary divergence inside the 7-D Thompson subspace than in the orthogonal non-Thompson residual subspace (|cos theta|_T7 > |cos theta|_res) on the 23-LM head.",
                "p": 0.80,
                "reasoning": "Direct replication of Hypothesis A7 from Camponotus (where |cos theta|_T7 = 0.903 vs |cos theta|_res = 0.301 on the frontal head).",
            },
            {
                "id": "PH4",
                "primary": True,
                "statement": "Leave-one-species-out phylogenetic prediction on `Pheidole_Tree.phy` beats the grand consensus (median Procrustes RMS ratio < 0.95, paired one-sided Wilcoxon p < 0.01) for minor head shape.",
                "p": 0.80,
                "reasoning": "Resolved molecular phylogeny across ~200 species captures Old World vs New World and clade-level head-shape syndromes.",
            },
            {
                "id": "PH5",
                "primary": False,
                "statement": "Interior sensory/oral landmarks (eyes, antennal insertions, clypeus: the 8 landmarks closest to the anterior/interior centroid) have a higher unexplained residual fraction after Thompson-7 than the 15 outer head-capsule contour landmarks.",
                "p": 0.80,
                "reasoning": "Replicates Hypothesis A8 from Camponotus: the outer head capsule deforms as a smooth continuous shell under Thompson-7, while eye and antennal-socket positions shift locally.",
            },
        ],
    },
    "morpho_butterflies": {
        "group": "insects: Morpho butterflies (Chazot et al. 2021 J Evol Biol: 911 specimens, 30 species on Morpho_MCC_tree.tre; forewing 22 LMs & hindwing 18 LMs)",
        "registered_on": dt.date.today().isoformat(),
        "portfolio_id": "butterflies_leptraits",
        "data": {
            "landmarks": "Chazot N, Blandin P, Debat V, Elias M, Condamine FL 2021 J Evol Biol 34:1076-1090; Zenodo 5391696 (`Chazotetal_2021_forewing_ProcCoord.txt` 22 2D landmarks, `Chazotetal_2021_hindwing_ProcCoord.txt` 18 2D landmarks across 911 specimens and 30 Morpho species).",
            "phylogeny": "Time-calibrated MCC phylogeny `Chazotetal_2021_Morpho_MCC_tree.tre` (30 matched Morpho species, crown age ~28 Myr).",
        },
        "what_thompson_specifically_adds": "Tests the portfolio's Papilionoidea / butterfly entry on true 2D wing landmarks across both forewings and hindwings on a time-calibrated tree, allowing a direct comparison of Lepidoptera wing venation/outline against Hymenoptera (hornets, bees) and Diptera (Drosophila).",
        "hypotheses": [
            {
                "id": "MB1",
                "primary": True,
                "statement": "Across the 30 Morpho species means, Thompson-7 captures > 45% of between-species Procrustes variance on both forewings (22 LMs) and hindwings (18 LMs).",
                "p": 0.80,
                "reasoning": "Butterfly wing landmarks include margin/apex/tornus contour points alongside discal cell veins, giving a slightly higher global coordinate share than Hymenoptera interior venation (38.8-42.2%).",
            },
            {
                "id": "MB2",
                "primary": True,
                "statement": "Phylogenetic signal on `Morpho_MCC_tree.tre`: Pagel's lambda > 0.5 (best model != White Noise) in both forewing and hindwing shape spaces.",
                "p": 0.85,
                "reasoning": "Canopy vs understory flight-strata syndromes in Morpho follow well-known subgeneric clades on the calibrated MCC tree.",
            },
            {
                "id": "MB3",
                "primary": True,
                "statement": "Leave-one-species-out phylogenetic prediction on `Morpho_MCC_tree.tre` beats the grand consensus (median Procrustes RMS ratio < 0.90, paired one-sided Wilcoxon p < 0.01) on forewing shape.",
                "p": 0.85,
                "reasoning": "Matches the 0.85 subjective prior registered for butterflies in `thompson_dataset_portfolio.json`.",
            },
            {
                "id": "MB4",
                "primary": False,
                "statement": "Sexual dimorphism vs interspecific divergence: within-species male -> female wing shape vectors have a higher Thompson-7 variance share than between-species divergence on the forewing.",
                "p": 0.75,
                "reasoning": "Like ant caste polyphenism, sexual wing-shape dimorphism in Morpho (broader female wings for egg load vs narrower male patrol wings) is an ontogenetic scaling of the same wing imaginal disc.",
            },
        ],
    },
    "waterfowl_beaks": {
        "group": "birds: Anseriformes waterfowl beaks (Olsen 2017 Evolution: 51 species on Burleigh_2015.tre; 2D dorsal/lateral upper-bill culmen & tomium contour landmarks)",
        "registered_on": dt.date.today().isoformat(),
        "portfolio_id": "birds_avonet",
        "data": {
            "landmarks": "Olsen AM 2017 Evolution 71:1143-1153; Dryad/Zenodo 5001646 (`Beak shape by species.zip`: 51 waterfowl species, 3D/2D upper-bill culmen and tomium curves).",
            "phylogeny": "Pruned ML phylogeny `Burleigh, Kimball, Braun, 2015 ML Tree, pruned.tre` (51 matched species).",
        },
        "what_thompson_specifically_adds": "Tests the portfolio's bird beak clade on a continuous 2D beak profile (dorsal culmen + ventral tomium outline across ducks, geese, swans, mergansers, screamers). Because the rhamphotheca/premaxilla develops from a single BMP4/Calmodulin growth zone (Abzhanov et al. 2004, Campàs et al. 2010), bird beak outlines should be among the most Thompsonian structures in vertebrates.",
        "hypotheses": [
            {
                "id": "WB1",
                "primary": True,
                "statement": "Bird beak profiles are exceptionally Thompsonian: across the 51 waterfowl species, the 7 a-priori Thompson modes capture > 80% of between-species Procrustes variance on the 2D lateral beak outline (exceeding fish bodies at 57.9%).",
                "p": 0.90,
                "reasoning": "Campàs et al. (2010) showed that avian beak shapes collapse onto simple scaling/shearing transformations; a smooth culmen-tomium contour has no decoupled interior modules.",
            },
            {
                "id": "WB2",
                "primary": True,
                "statement": "Phylogenetic signal on the 51-species waterfowl tree is strong: Pagel's lambda > 0.7 (best model != White Noise) in both Thompson-7 and full PCA-95% spaces.",
                "p": 0.90,
                "reasoning": "Grazing geese, dabbling ducks, diving pochards, and fish-catching mergansers form strongly conserved phylogenetic clades.",
            },
            {
                "id": "WB3",
                "primary": True,
                "statement": "Leave-one-species-out phylogenetic prediction beats the grand consensus by a wide margin (median Procrustes RMS ratio < 0.75, paired one-sided Wilcoxon p < 0.001).",
                "p": 0.95,
                "reasoning": "Matches the 0.95 subjective prior registered for birds in `thompson_dataset_portfolio.json`.",
            },
            {
                "id": "WB4",
                "primary": False,
                "statement": "Thompson-7 captures > 80% of the full-space phylogenetic prediction gain relative to consensus.",
                "p": 0.85,
                "reasoning": "Because >80% of beak shape variance lies in Thompson-7, the 7-mode basis should retain nearly all of the phylogenetic predictive power.",
            },
        ],
    },
    "leporidae_skulls": {
        "group": "mammals: Lagomorpha / Leporidae crania (Sherratt & Kraatz 2023 Evolution: 317 specimens, 23 species on Matthee_Tree.nex; 74 cranial landmarks)",
        "registered_on": dt.date.today().isoformat(),
        "portfolio_id": "mammals_pantheria_combine",
        "data": {
            "landmarks": "Sherratt E & Kraatz B 2023 Evolution 77:1349-1362; Zenodo 7615230 (`Symmetric_component_coords.csv` 74 3D cranial landmarks across 317 specimens, mapped to 23 lagomorph/leporid species via `File_IDs.csv`).",
            "phylogeny": "Calibrated phylogeny `Matthee_Tree.nex` (23 matched extant species).",
        },
        "what_thompson_specifically_adds": "Tests Thompson's Chapter XVII mammalian skull transformations (pp. 754-775) on a 74-landmark mammalian cranium dataset (projected onto lateral sagittal-facial XZ and dorsal XY views) across hares, rabbits, and cottontails.",
        "hypotheses": [
            {
                "id": "LS1",
                "primary": True,
                "statement": "Across the 23 lagomorph species means on the 74-LM lateral cranial profile, Thompson-7 captures > 55% of between-species Procrustes variance.",
                "p": 0.80,
                "reasoning": "Facial tilt (klinorhynchy/airorhynchy) and rostral elongation dominate leporid cranial disparity (Kraatz et al. 2015) and map directly onto Thompson's shear and quadratic bending modes.",
            },
            {
                "id": "LS2",
                "primary": True,
                "statement": "Leave-one-species-out phylogenetic prediction on `Matthee_Tree.nex` beats the grand consensus (median Procrustes RMS ratio < 0.95, paired one-sided Wilcoxon p < 0.05) on the cranial profile.",
                "p": 0.75,
                "reasoning": "Genus-level conservatism within *Lepus* and *Sylvilagus* provides tree signal, though cursorial/saltatorial convergence across genera moderates the gain on n=23 species.",
            },
            {
                "id": "LS3",
                "primary": False,
                "statement": "Within-species intraspecific cranial variation (pooled across the 317 specimens) has a higher or comparable Thompson-7 variance share (> 45%) and aligns significantly with interspecific divergence inside the Thompson-7 subspace (|cos theta|_T7 > 0.5).",
                "p": 0.75,
                "reasoning": "Ontogenetic/static cranial allometry (rostral lengthening with size) channels interspecific divergence along the primary Thompsonian facial-elongation axis.",
            },
        ],
    },
    "galeommatoidea_bivalves": {
        "group": "molluscs: Galeommatoidea marine bivalve shells (Goto et al. 2016 Evolution: 302 specimens, ~100 species on final_BEAST.tre; 46 shell-outline landmarks)",
        "registered_on": dt.date.today().isoformat(),
        "portfolio_id": "galeommatoidea_bivalves",
        "data": {
            "landmarks": "Goto R, Ishikawa H, Hamamura Y, Sato S, Kato M 2016 Evolution 70:1308-1322; Dryad/Zenodo 4938020 (`all_side.tps`: 1 umbo landmark + 45 shell-margin semilandmarks = 46 2D landmarks across 302 specimens).",
            "phylogeny": "Time-calibrated BEAST phylogeny `final_BEAST.tre` (225 tips; ~100 matched species/OTUs with shell landmarks).",
        },
        "what_thompson_specifically_adds": "Mollusc shells are the foundational archetype of *On Growth and Form* (Chapters XI-XV). Because a bivalve valve grows by continuous marginal accretion around the umbo without internal skeletal sutures, its 2D lateral contour should be the purest test of Thompson's 7 smooth coordinate modes anywhere in Metazoa.",
        "hypotheses": [
            {
                "id": "GB1",
                "primary": True,
                "statement": "Bivalve shells are ultra-Thompsonian: across the ~100 Galeommatoidea species means (46 2D landmarks), the 7 a-priori Thompson modes capture > 85% of between-species Procrustes variance (exceeding fish bodies, insect wings, and vertebrate skulls).",
                "p": 0.90,
                "reasoning": "Accretionary marginal growth around the umbo generates smooth aspect-ratio, ventral-arch, and anterior-posterior asymmetry gradients with zero internal modular landmarks.",
            },
            {
                "id": "GB2",
                "primary": True,
                "statement": "Phylogenetic signal on `final_BEAST.tre` is non-zero: Pagel's lambda > 0.5 (best model != White Noise) in both Thompson-7 and full PCA-95% spaces.",
                "p": 0.85,
                "reasoning": "Despite repeated transitions between free-living and host-commensal habits, subclades share conserved valve elongation and umbo placement.",
            },
            {
                "id": "GB3",
                "primary": True,
                "statement": "Leave-one-species-out phylogenetic prediction on `final_BEAST.tre` beats the grand consensus (median Procrustes RMS ratio < 0.85, paired one-sided Wilcoxon p < 0.001).",
                "p": 0.85,
                "reasoning": "Dense taxon sampling (~100 tips on a calibrated BEAST tree) provides close relatives within most commensal and free-living lineages.",
            },
            {
                "id": "GB4",
                "primary": False,
                "statement": "Thompson-7 captures > 85% of the full-space phylogenetic prediction gain relative to consensus.",
                "p": 0.85,
                "reasoning": "Because the 46-point shell contour lies almost entirely in the 7-D Thompson subspace, full PCA-95% adds almost no extra phylogenetically predictable signal.",
            },
        ],
    },
    "cichlids_ronco_gm": {
        "group": "ray-finned fishes: Lake Tanganyika cichlid radiation (Ronco et al. 2021 Nature: 2,197 specimens, ~235 species on 05_BEAST_RAxML.tre; 21-LM body shape vs 27-LM lower pharyngeal jaw)",
        "registered_on": dt.date.today().isoformat(),
        "portfolio_id": "ronco2021_tanganyika_cichlids",
        "data": {
            "landmarks": "Ronco F et al. 2021 Nature 589:76-81; Zenodo 3989859 (`06_landmark_data_body_shape.tps` 21 2D body landmarks across 2,197 specimens; `06_landmark_data_lower_pharyngeal_jaw.tps` 27 micro-CT lower pharyngeal jaw landmarks across 1,196 specimens).",
            "phylogeny": "Time-calibrated species tree `05_BEAST_RAxML.tre` (265 tips, ~9.7 Myr crown radiation).",
        },
        "what_thompson_specifically_adds": "While the Rabosky test (`ronco2021_tanganyika_cichlids`) tested deep-horizon (>100 Myr) scalar ratio transfer from marine reef fish, re-training the 2D GM engine directly on Ronco's 235-species BEAST tree tests: (1) Thompson-7 variance share on an explosive freshwater adaptive radiation, (2) external hydrodynamic body shape (21 LMs) vs internal trophic Lower Pharyngeal Jaw (27 LMs) on the exact same species, and (3) within-radiation phylogenetic prediction.",
        "hypotheses": [
            {
                "id": "CR1",
                "primary": True,
                "statement": "Across the ~235 Tanganyika cichlid species on `05_BEAST_RAxML.tre`, Thompson-7 captures > 55% of between-species Procrustes variance on the 21-LM body shape (matching the 57.9% marine reef-fish benchmark).",
                "p": 0.85,
                "reasoning": "Cichlid body-shape radiation (deep-bodied benthic grazers vs elongate pelagic piscivores) follows the same dorsoventral and caudal-peduncle modes as marine percomorphs.",
            },
            {
                "id": "CR2",
                "primary": True,
                "statement": "Leave-one-species-out phylogenetic prediction on `05_BEAST_RAxML.tre` strongly beats the grand consensus (median Procrustes RMS ratio < 0.65, paired one-sided Wilcoxon p < 1e-10) on 21-LM body shape.",
                "p": 0.95,
                "reasoning": "Dense sampling (~235 species across 12 Tanganyika tribes within a 10 Myr radiation) places most species within 1-3 Myr of a congeneric relative.",
            },
            {
                "id": "CR3",
                "primary": True,
                "statement": "Thompson-7 captures > 60% of the full-space phylogenetic prediction gain on 21-LM body shape AND beats the consensus on its own (Wilcoxon p < 1e-10).",
                "p": 0.90,
                "reasoning": "Tribe-level body-plan differences in Tanganyika cichlids are dominated by global elongation and depth modes.",
            },
            {
                "id": "CR4",
                "primary": False,
                "statement": "Internal trophic jaw vs external body: the 27-LM Lower Pharyngeal Jaw (2D dorsal XY projection) is also strongly predicted by phylogeny (LOO ratio vs consensus < 0.75, Wilcoxon p < 1e-10) and has > 50% Thompson-7 variance share.",
                "p": 0.80,
                "reasoning": "The cichlid lower pharyngeal jaw is a triangular/heart-shaped plate whose width-to-length ratio and posterior horn flare deform smoothly under Thompson's modes.",
            },
        ],
    },
    "characiformes_burns_gm": {
        "group": "ray-finned fishes: Characiformes radiation (Burns & Sidlauskas 2019 Evolution: 1,625 specimens, 128 species on trees_final.nexus; 24-LM body shape)",
        "registered_on": dt.date.today().isoformat(),
        "portfolio_id": "burns_sidlauskas2019_characiformes",
        "data": {
            "landmarks": "Burns MD & Sidlauskas BL 2019 Evolution 73:569-587; Dryad/Zenodo 4988906 (`Chapter 1_Disparity Data.TPS`: 24 2D landmarks across 1,625 specimens and 263 species).",
            "phylogeny": "Time-calibrated phylogeny `Chapter 1_trees_final.nexus` (129 tips, 128 matched species).",
        },
        "what_thompson_specifically_adds": "Tests whether an ancient (~110 Myr) continental freshwater radiation of Otophysi (tetras, piranhas, headstanders, hatchetfishes, wolf-fishes) exhibits the same ~58% Thompson-7 variance share and within-clade phylogenetic predictability as marine acanthomorphs.",
        "hypotheses": [
            {
                "id": "CB1",
                "primary": True,
                "statement": "Across the 128 Characiformes tree species (24 2D landmarks), Thompson-7 captures > 55% of between-species Procrustes variance.",
                "p": 0.85,
                "reasoning": "Characiform body plans span fusiform darters (`Characidium`) to deep-bodied rhomboid serrasalmids and anostomids along the classic Thompson dorsoventral/shear axes.",
            },
            {
                "id": "CB2",
                "primary": True,
                "statement": "Phylogenetic signal on `trees_final.nexus` is strong: Pagel's lambda > 0.7 (best model != White Noise) in both Thompson-7 and full PCA-95% spaces.",
                "p": 0.90,
                "reasoning": "Characiform families (Crenuchidae, Curimatidae, Anostomidae, Characidae) occupy distinct, conserved regions of morphospace.",
            },
            {
                "id": "CB3",
                "primary": True,
                "statement": "Leave-one-species-out phylogenetic prediction on `trees_final.nexus` beats the grand consensus (median Procrustes RMS ratio < 0.75, paired one-sided Wilcoxon p < 1e-8).",
                "p": 0.90,
                "reasoning": "Within-clade conditional prediction across 128 characiform species leverages family-level and genus-level body-plan conservatism.",
            },
            {
                "id": "CB4",
                "primary": False,
                "statement": "Head/eye/opercle interior landmarks have a higher unexplained residual fraction after Thompson-7 than the outer dorsal/ventral/caudal body contour landmarks.",
                "p": 0.75,
                "reasoning": "As in Claverie reef fishes, the outer body contour deforms smoothly under Thompson-7 while internal head/eye/jaw landmarks shift locally with trophic niche.",
            },
        ],
    },
    "varanus_lizards": {
        "group": "squamate reptiles: Indo-Australasian Varanus monitor lizards (Brennan et al. 2021 Syst Biol / Zenodo 5655818: 1,091 dorsal + 1,098 lateral crania across 80 taxa on calibrated Tree.tre; 33 dorsal LMs & 10 lateral LMs)",
        "registered_on": dt.date.today().isoformat(),
        "portfolio_id": "squamates_meiri",
        "data": {
            "landmarks": "Brennan IG, Lemmon AR, Lemmon EM, Portik DM, Weijola V, Welton L, Donnellan SC, Keogh JS 2021 Syst Biol / Zenodo 5655818 (`Head_shape_dorsal.csv` 33 2D dorsal head landmarks across 1,091 specimens; `Head_shape_lateral.csv` 10 2D lateral head landmarks across 1,098 specimens; 80 Varanus taxa).",
            "phylogeny": "Time-calibrated phylogeny `Tree.tre` (76 tips across Varanus + Shinisaurus outgroup; ~74 matched Varanus species).",
        },
        "what_thompson_specifically_adds": "Tests the portfolio's squamate reptile entry (`squamates_meiri`) on a 1,091-specimen, 74-species time-calibrated radiation of monitor lizards (*Varanus*, from pygmy *V. sparnus* to giant *V. komodoensis*) digitized in both dorsal (33 LMs) and lateral (10 LMs) cranial views.",
        "hypotheses": [
            {
                "id": "VL1",
                "primary": True,
                "statement": "Across the ~80 Varanus taxa, Thompson-7 captures > 55% of between-species Procrustes variance on both dorsal (33 LMs) and lateral (10 LMs) head views.",
                "p": 0.85,
                "reasoning": "Monitor lizard cranial evolution is dominated by snout elongation/broadening and cranial vault depth across dwarf rock-crevice vs giant terrestrial predators.",
            },
            {
                "id": "VL2",
                "primary": True,
                "statement": "Phylogenetic signal on `Tree.tre` is strong: Pagel's lambda > 0.7 (best model != White Noise) in Thompson-7 space for both dorsal and lateral head views.",
                "p": 0.85,
                "reasoning": "Subgenera (*Odatria*, *Varanus*, *Euprepiosaurus*, *Soterosaurus*) show strong clade-level cranial conservatism.",
            },
            {
                "id": "VL3",
                "primary": True,
                "statement": "Leave-one-species-out phylogenetic prediction on `Tree.tre` beats the grand consensus (median Procrustes RMS ratio < 0.85, paired one-sided Wilcoxon p < 1e-5) on dorsal head shape.",
                "p": 0.85,
                "reasoning": "Matches the 0.85 subjective prior registered for squamate reptiles in `thompson_dataset_portfolio.json`.",
            },
            {
                "id": "VL4",
                "primary": False,
                "statement": "Within-species intraspecific/ontogenetic cranial variation aligns more strongly with between-species divergence inside the 7-D Thompson subspace than in the orthogonal modular residual subspace (|cos theta|_T7 > |cos theta|_res) on the 33-LM dorsal head.",
                "p": 0.75,
                "reasoning": "Allometric snout lengthening within species channels interspecific cranial radiation along the primary Thompsonian elongation mode.",
            },
        ],
    },
    "australian_frogs": {
        "group": "amphibians: Continental radiation of Australian Anura (Sherratt et al. 2017 Evol Biol / Zenodo 5008593: 166 frog species on time-calibrated phylogeny; 55 2D larval tadpole body/tail landmarks + 21 adult postcranial/cranial traits)",
        "registered_on": dt.date.today().isoformat(),
        "portfolio_id": "amphibians_amphibio",
        "data": {
            "landmarks": "Sherratt E, Vidal-García M, Anstis M, Keogh JS 2017 Evol Biol 44:273-284; Dryad/Zenodo 5008593 (`Supplementary data file 2.tps`: 55 2D lateral body-and-tail landmarks across 166 Australian frog species spanning Hylidae, Myobatrachidae, Limnodynastidae, Microhylidae, Ranidae; `Supplementary data file 1.csv`: 21 adult morphometric traits).",
            "phylogeny": "Time-calibrated phylogeny `Supplementary data file 4.tre` (166 matched Australian anuran species).",
        },
        "what_thompson_specifically_adds": "Tests the portfolio's amphibian entry (`amphibians_amphibio`) across a continental-scale radiation of 166 anuran species on a time-calibrated tree, comparing 55-landmark whole-body/tail geometry against Thompson's 7 global coordinate modes.",
        "hypotheses": [
            {
                "id": "AF1",
                "primary": True,
                "statement": "Across the 166 Australian frog species (55 2D landmarks), the 7 a-priori Thompson modes capture > 70% of between-species Procrustes variance (exceeding marine reef fishes at 57.9%).",
                "p": 0.85,
                "reasoning": "Unlike teleost fish with decoupled spiny dorsal/anal fin insertions and protractile jaws, the 55 anuran tadpole landmarks trace a continuous, fin-fringed axial body-and-tail contour governed by smooth dorsoventral arch and tail-taper gradients.",
            },
            {
                "id": "AF2",
                "primary": True,
                "statement": "Phylogenetic signal on the 166-species anuran tree is non-zero: Pagel's lambda > 0.5 (best model != White Noise) in both Thompson-7 and full PCA-95% spaces.",
                "p": 0.80,
                "reasoning": "Major Australian frog families (Hylidae vs Myobatrachidae/Limnodynastidae) and torrent/burrowing/arboreal guilds cluster on the phylogeny.",
            },
            {
                "id": "AF3",
                "primary": True,
                "statement": "Leave-one-species-out phylogenetic prediction beats the grand consensus (median Procrustes RMS ratio < 0.85, paired one-sided Wilcoxon p < 1e-6) across the 166 species.",
                "p": 0.80,
                "reasoning": "Matches the 0.80 subjective prior registered for amphibians in `thompson_dataset_portfolio.json`.",
            },
            {
                "id": "AF4",
                "primary": False,
                "statement": "Thompson-7 captures > 70% of the full-space phylogenetic prediction gain relative to consensus.",
                "p": 0.80,
                "reasoning": "Because the continuous 55-LM body/tail outline is dominated by global Thompson modes, Thompson-7 should carry the majority of tree-predictable signal.",
            },
        ],
    },
    "antillean_flowers": {
        "group": "seed plants: Antillean Gesneriaceae angiosperm corollas (Joly et al. 2018 Evolution / Zenodo 5011744: 296 corollas across 48 species on *BEAST tree; 32 2D floral landmarks across hummingbird, bat, and generalist pollination syndromes)",
        "registered_on": dt.date.today().isoformat(),
        "portfolio_id": "plants_diaz_try",
        "data": {
            "landmarks": "Joly S, Lambert F, Alexandre H, Clavel J, Léveillé-Bourret É, Clark JL 2018 Evolution 72:244-260; Dryad/Zenodo 5011744 (`scriptsAndData.zip`: `landmark_data.nts` 32 2D lateral corolla landmarks across 296 floral profiles and 48 Gesneriaceae species in *Gesneria*, *Rhytidophyllum*, and *Pheidonocarpa*).",
            "phylogeny": "Calibrated *BEAST maximum clade credibility species tree `BestTree.tre` (48 matched species).",
        },
        "what_thompson_specifically_adds": "Tests the portfolio's seed-plant entry (`plants_diaz_try`) on Thompson's explicit Chapter XVI botanical archetype: sympetalous floral corolla tubes (tubular hummingbird flowers vs campanulate bat/generalist bells), testing whether angiosperm floral radiation across pollination syndromes is governed by Thompson's 7 global coordinate modes.",
        "hypotheses": [
            {
                "id": "FP1",
                "primary": True,
                "statement": "Angiosperm corolla tubes are ultra-Thompsonian: across the 48 Gesneriaceae species (32 2D landmarks), the 7 a-priori Thompson modes capture > 80% of between-species Procrustes variance.",
                "p": 0.85,
                "reasoning": "A sympetalous corolla tube grows as a continuous fused petal sheet whose variation (tubular vs campanulate flare, dorsal-ventral zygomorphic curvature, throat constriction) maps directly onto Thompson's aspect, fan, and quadratic bending modes.",
            },
            {
                "id": "FP2",
                "primary": True,
                "statement": "Phylogenetic signal on `BestTree.tre` is non-zero: Pagel's lambda > 0.5 (best model != White Noise) in Thompson-7 space.",
                "p": 0.75,
                "reasoning": "Despite repeated pollinator shifts between hummingbirds and bats, subclades retain strong floral tube conservatism.",
            },
            {
                "id": "FP3",
                "primary": True,
                "statement": "Leave-one-species-out phylogenetic prediction on `BestTree.tre` beats the grand consensus (median Procrustes RMS ratio < 0.90, paired one-sided Wilcoxon p < 0.01).",
                "p": 0.70,
                "reasoning": "Matches the 0.70 subjective prior registered for seed plants in `thompson_dataset_portfolio.json`.",
            },
            {
                "id": "FP4",
                "primary": False,
                "statement": "Developmental/intraspecific channelling: within-species floral variation (pooled across the 296 corollas) aligns strongly with between-species floral divergence inside the Thompson-7 subspace (|cos theta|_T7 > 0.70 and |cos theta|_T7 > |cos theta|_res).",
                "p": 0.80,
                "reasoning": "Joly et al. (2018) showed that intraspecific floral variation channels evolutionary transitions between pollination syndromes; this alignment should concentrate in the 7 global Thompson growth modes.",
            },
        ],
    },
}


def main() -> int:
    GROUPS_DIR.mkdir(parents=True, exist_ok=True)
    manifest_files = {}
    total_hyps = 0
    for gid, payload in SCOUTED_HYPOTHESES.items():
        path = GROUPS_DIR / f"{gid}_hypotheses.json"
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        manifest_files[str(path.relative_to(REPO_ROOT))] = sha256(path)
        n_h = len(payload["hypotheses"])
        total_hyps += n_h
        print(f"  locked {path.relative_to(REPO_ROOT)} ({n_h} hypotheses, sha256={manifest_files[str(path.relative_to(REPO_ROOT))][:12]}…)")

    manifest = {
        "generated_on": dt.datetime.now(dt.timezone.utc).isoformat(),
        "git_commit_at_registration": git_commit(),
        "n_clades_registered": len(SCOUTED_HYPOTHESES),
        "n_hypotheses_registered": total_hyps,
        "hypothesis_files_sha256": manifest_files,
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {MANIFEST_PATH.relative_to(REPO_ROOT)} ({len(SCOUTED_HYPOTHESES)} clades, {total_hyps} pre-registered hypotheses)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

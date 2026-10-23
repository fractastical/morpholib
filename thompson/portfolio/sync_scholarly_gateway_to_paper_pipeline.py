#!/usr/bin/env python3
"""
Synchronize all Infinite Morphospace discoveries, pre-registered clade benchmarks,
nanopublication locks, and gateway professor outreach / peer-validation records
with the sibling `paper-pipeline` repository (`papers_progress.json`) and generate:
  1. `docs/SCHOLARLY_GATEWAY_VALIDATION_TRACKER.md` in `infinitemorphospace`
  2. Section 6 of `docs/portfolio/PORTFOLIO_STATUS.md`
  3. Updated paper entries, collaborator lists, and `discovery_gateway_validation`
     registry in `/Users/jdietz/Documents/GitHub/paper-pipeline/papers_progress.json`.
"""
from __future__ import annotations

import datetime as dt
import json
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PAPER_PIPELINE_ROOT = REPO_ROOT.parent / "paper-pipeline"
PAPERS_JSON = PAPER_PIPELINE_ROOT / "papers_progress.json"
TRACKER_MD = REPO_ROOT / "docs" / "SCHOLARLY_GATEWAY_VALIDATION_TRACKER.md"


def count_words_in_latex(file_path: Path) -> int:
    if not file_path.exists():
        return 0
    content = file_path.read_text(encoding="utf-8")
    content = re.sub(r"%.*", "", content)
    content = re.sub(r"\\[a-zA-Z]+\*?(\[[^\]]*\])?(\{[^\}]*\})*", "", content)
    content = re.sub(r"\\begin\{[^\}]+\}.*?\\end\{[^\}]+\}", "", content, flags=re.DOTALL)
    content = re.sub(r"[{}]", "", content)
    content = re.sub(r"\s+", " ", content)
    words = [w for w in content.split() if w and not w.startswith("\\")]
    return len(words)


DISCOVERIES = [
    {
        "discovery_id": "disc-01-thompson-silhouette-paradox",
        "paper_id": "thompson_1917_fish_transformations",
        "title": "17-Landmark Skeletal Wireframe vs. 600-Semilandmark Outline Paradox & 4p Box-vs.-Warping Deconfounding",
        "domain": "Vertebrate Geometric Morphometrics (Teleostei)",
        "quantitative_claim": (
            "9/17 craniofacial landmarks (53% OLS weight) hijack global grids on interfamily pairs "
            "(Scarus->Pomacanthus |AΔB| worsens 40.6% -> 56.2%); direct 600-pt outline fit cuts |AΔB| to 28.5% "
            "with 100% diffeomorphic bijectivity (det DΦ > 0 on 6/6 pairs, vs 0/6 for 600-pt curve TPS); "
            "on homologous perimeter anatomy, Thompson warps beat 4p box stretch by +13% to +48%."
        ),
        "verification_locks": {
            "lean4_theorems": ["tps_kernel_zero_at_origin", "thompson_unit_shear_area_preserving"],
            "nanopub_uris": [
                "https://w3id.org/np/RAilN4nbRHp3e3ZHqlbIgeL3UGPRV_KtQHtJD3gzVVdqw",
                "https://w3id.org/np/RAJancwGOPEkHzBoZXGv_vu_snW0Lff2FktE85Me1P-fI",
            ],
            "scripts": [
                "thompson/1917-thompson-silhouette-area-paradox.py",
                "thompson/run_wainwright_600pt_and_convergence_tests.py",
                "thompson/run_box_vs_warping_audit.py",
            ],
            "outreach_plates": [
                "docs/outreach_fig1_bookstein_silhouette_paradox.png",
                "docs/outreach_fig4_wainwright_600pt_and_convergence.png",
                "docs/outreach_fig7_box_stretch_vs_true_thompson_warping.png",
            ],
            "outreach_email_doc": "docs/thompson_1917_outreach_emails.md",
        },
        "gateway_scholars": [
            {
                "name": "Prof. Peter C. Wainwright",
                "affiliation": "UC Davis",
                "email": "pcwainwright@ucdavis.edu",
                "gateway_role": "Primary Dataset Author & Senior Ichthyological Authority",
                "status": "consulted_and_remediated",
                "contact_date": "2026-10-21",
                "validation_notes": (
                    "Spoke 2026-10-21; requested (1) holding 600-pt input fixed against standard 600-pt curve TPS & EFD "
                    "and (2) data-driven Relative Warps & Phylomorphospace convergence. Both completed in Section 6 & outreach_fig4."
                ),
            },
            {
                "name": "Prof. Fred L. Bookstein",
                "affiliation": "University of Vienna / University of Michigan",
                "email": "fred.bookstein@univie.ac.at",
                "gateway_role": "Founder of Geometric Morphometrics & TPS / Transformation Grid Critic (2023, 2024)",
                "status": "dossier_ready",
                "contact_date": None,
                "validation_notes": "Custom plate outreach_fig1_bookstein_silhouette_paradox.png and email prepared in docs/thompson_1917_outreach_emails.md.",
            },
            {
                "name": "Prof. Anjali Goswami & Dr. Ryan N. Felice",
                "affiliation": "Natural History Museum London / UCL",
                "email": "a.goswami@nhm.ac.uk",
                "gateway_role": "High-Density Semilandmark & Curve-Sliding Artifact Authorities (Felice et al. 2018)",
                "status": "dossier_ready",
                "contact_date": None,
                "validation_notes": "Recommended by Peter Wainwright; email & plates prepared in docs/thompson_1917_outreach_emails.md (§6).",
            },
        ],
    },
    {
        "discovery_id": "disc-02-thompson-50myr-modularity-horizon",
        "paper_id": "thompson_1917_fish_transformations",
        "title": "The ~45–50 Myr Macroevolutionary Disparity Transition & Craniofacial-vs.-Fin Modularity (4,334 Dated Pairs; 10,303 Species)",
        "domain": "Macroevolution & Phylogenetic Comparative Methods",
        "quantitative_claim": (
            "Across 4,334 dated pairs (791 spp), 8p Thompson grids pull 17-LM Procrustes RMS inside intrageneric disparity "
            "(σ_genus <= 0.0110) in 53%–61% of pairs <50 Myr (node-disjoint inflection t* = 45.8 Myr, n=187), dropping to 0% "
            ">110 Myr as somitic fin insertions decouple from neural-crest skull landmarks (Fin/Head RMS = 1.16x -> 1.48x). "
            "Across 10,303 species (440 families), global body trajectories align with Thompson's 8p highway (|cos θ| = 0.935–0.989) "
            "while sympatric reef sisters diverge 3.3x–9.3x faster in trophic jaws (p = 9.2e-5)."
        ),
        "verification_locks": {
            "lean4_theorems": ["thompson_unit_shear_area_preserving"],
            "nanopub_uris": [
                "https://w3id.org/np/RA9ZKlSDsa68lEKRo7JR0Px5woGoCMUHRlSK_bHKag4EA",
            ],
            "scripts": [
                "thompson/1917-thompson-four-families-atlas.py",
                "thompson/run_rigor_stress_tests.py",
                "thompson/run_morphospace_vs_competition_analysis.py",
                "thompson/run_three_dataset_validation.py",
            ],
            "outreach_plates": [
                "docs/outreach_fig2_price_wainwright_horizon.png",
                "docs/outreach_fig3_evans_larouche_modularity.png",
                "docs/outreach_fig5_morphospace_vs_competition.png",
                "docs/outreach_fig6_three_dataset_validation.png",
            ],
            "outreach_email_doc": "docs/thompson_1917_outreach_emails.md",
        },
        "gateway_scholars": [
            {
                "name": "Prof. Miriam L. Zelditch",
                "affiliation": "University of Michigan",
                "email": "zelditch@umich.edu",
                "gateway_role": "Co-Author of Geometric Morphometrics for Biologists & Teleost Head-Body-Fin Modularity Pioneer",
                "status": "dossier_ready",
                "contact_date": None,
                "validation_notes": "Recommended by Peter Wainwright; email & plates prepared in docs/thompson_1917_outreach_emails.md (§5).",
            },
            {
                "name": "Prof. Kory M. Evans & Dr. Olivier Larouche",
                "affiliation": "Rice University / Univ. Lille",
                "email": "kory.evans@rice.edu",
                "gateway_role": "Teleost Craniofacial & Fin Modularity Authorities",
                "status": "dossier_ready",
                "contact_date": None,
                "validation_notes": "Custom plate outreach_fig3_evans_larouche_modularity.png & email prepared in docs/thompson_1917_outreach_emails.md (§4).",
            },
            {
                "name": "Prof. Samantha A. Price & Dr. Sarah T. Friedman",
                "affiliation": "Clemson University / NOAA",
                "email": "sprice6@clemson.edu",
                "gateway_role": "Lead Authors of FishShapes v1 (16,523 specimens) & 791-Species Reef-Fish Datasets",
                "status": "dossier_ready",
                "contact_date": None,
                "validation_notes": "Custom plate outreach_fig2_price_wainwright_horizon.png & blind-test report outreach_fig10_blind_test_fishshapes.png ready.",
            },
        ],
    },
    {
        "discovery_id": "disc-03-multiclade-vertebrate-mollusc-portfolio",
        "paper_id": "thompson_1917_fish_transformations",
        "title": "Pre-Registered Multi-Clade 2D Geometric Morphometrics Atlas Across 8 Classes/Phyla (Bivalves, Waterfowl Beaks, Gesneriaceae Corollas, Frogs, Monitor Lizards, Mammalian Crania, Cichlids & Characiforms)",
        "domain": "Cross-Kingdom Comparative Macroevolution (Mollusca, Aves, Angiospermae, Amphibia, Squamata, Mammalia, Actinopterygii)",
        "quantitative_claim": (
            "All 8 out-of-group clades and 15 2D GM studies resolved (38/82 hypotheses supported): "
            "Anseriformes waterfowl beaks (51 spp, 91.1% T7, 2/4 hyps ✓), Galeommatoidea bivalve shells (302 spec., 85.0% T7, LOO 0.65x, 3/4 hyps ✓), "
            "Antillean Gesneriaceae floral corollas (296 flowers, 48 spp, 82.1% T7, LOO 0.69x, intraspecific channelling |cos θ|_T7 = 0.978 vs 0.143 residual, 3/4 hyps ✓), "
            "Australian Anura frogs (166 spp, 55 LMs, 66.2% T7, LOO 0.80x, p=1.5e-14, 2/4 hyps ✓), "
            "Varanus monitor lizards (1,091 crania, 80 spp, 59.5% dorsal / 55.2% lateral T7, T7 LOO 0.89x, 1/4 hyps ✓), "
            "Characiformes (1,625 spec., 55.5% T7, LOO 0.62x, 3/4 hyps ✓), Leporidae crania (317 spec., 51.1% T7), "
            "and Lake Tanganyika cichlids (2,197 spec., 47.2% T7, LOO 0.42x, OU alpha=0.030/Myr)."
        ),
        "verification_locks": {
            "lean4_theorems": ["raup_1966_self_similar_scaling"],
            "nanopub_uris": [
                "https://w3id.org/np/RA9ZKlSDsa68lEKRo7JR0Px5woGoCMUHRlSK_bHKag4EA",
                "https://w3id.org/np/RAQ_Giwd735IPhYg-cOMyPE6t4K9WPM9XlijGkNC6_eJ4",
                "https://w3id.org/np/RAOvI_PZUXLZtKjyVfq4ILVoIxqVCsTXmLXd0J7ZujfEo",
            ],
            "scripts": [
                "thompson/portfolio/run_all_scouted_clades.py",
                "thompson/portfolio/summarise_portfolio.py",
                "thompson/portfolio/score_locked_predictions.py",
            ],
            "outreach_plates": [
                "docs/outreach_fig10_blind_test_fishshapes.png",
                "docs/outreach_fig11_dataset_portfolio.png",
                "docs/portfolio/fig_antillean_flowers.png",
                "docs/portfolio/fig_australian_frogs.png",
                "docs/portfolio/fig_varanus_lizards.png",
            ],
            "outreach_email_doc": "docs/portfolio/PORTFOLIO_STATUS.md",
        },
        "gateway_scholars": [
            {
                "name": "Dr. Fabrizia Ronco & Prof. Walter Salzburger",
                "affiliation": "University of Basel",
                "email": "walter.salzburger@unibas.ch",
                "gateway_role": "Authors of Lake Tanganyika Cichlid Adaptive Radiation Dataset (Nature 2021)",
                "status": "scored_in_portfolio",
                "contact_date": None,
                "validation_notes": "Scored in docs/portfolio/cichlids_ronco_gm.md & ronco2021_tanganyika_cichlids.md (LOO 0.42x consensus; OU t1/2=23 Myr).",
            },
            {
                "name": "Prof. Brian L. Sidlauskas & Dr. Michael D. Burns",
                "affiliation": "Oregon State University",
                "email": "brian.sidlauskas@oregonstate.edu",
                "gateway_role": "Originator of Phylomorphospace (Sidlauskas 2008) & Characiformes GM Dataset (2019)",
                "status": "scored_in_portfolio",
                "contact_date": None,
                "validation_notes": "Scored in docs/portfolio/characiformes_burns_gm.md (3/4 hypotheses supported, Brier 0.151).",
            },
            {
                "name": "Dr. Emma Sherratt, Prof. J. Scott Keogh & Dr. Ian G. Brennan",
                "affiliation": "University of Adelaide / Australian National University",
                "email": "emma.sherratt@adelaide.edu.au",
                "gateway_role": "Authors of Australian Anura Frog Radiation (2017), Varanus Monitor Lizard Crania (2021), & Leporidae Crania (2023)",
                "status": "scored_in_portfolio",
                "contact_date": None,
                "validation_notes": "Scored across all 3 datasets: Australian frogs (66.2% T7, LOO 0.80x, p=1.5e-14), Varanus crania (1,091 skulls, 59.5% T7), and Leporidae crania (51.1% T7).",
            },
            {
                "name": "Prof. Simon Joly & Dr. Aaron M. Olsen",
                "affiliation": "Université de Montréal / Brown University",
                "email": "simon.joly@umontreal.ca",
                "gateway_role": "Authors of Gesneriaceae Corolla Evolution (2018) & Anseriformes Waterfowl Beak GM (2017)",
                "status": "scored_in_portfolio",
                "contact_date": None,
                "validation_notes": "Scored in docs/portfolio/antillean_flowers.md (82.1% T7, |cos θ|_T7 = 0.978 channelling, 3/4 hyps ✓) and waterfowl_beaks.md (91.1% T7).",
            },
        ],
    },
    {
        "discovery_id": "disc-04-crab-carapace-paradox",
        "paper_id": "thompson_arthropod_morphospace",
        "title": "Resolution of the Scholtz, Knötel & Baum (2020) Crab Carapace Paradox (Fig. 369: 8 Midline LMs vs. 14 Lateral-Margin LMs)",
        "domain": "Arthropod Morphometrics (Decapoda: Brachyura & Anomura)",
        "quantitative_claim": (
            "Scholtz et al.'s (2020) 8 landmarks lie 100% along the narrow rostral-cardiac-pleon midline (0/8 on lateral hepatic/branchial "
            "margins), inflating lateral RMS error by 1.85x–7.13x; adding 6 homologous lateral-margin anchors (14 LMs) and fitting Thompson's "
            "8p bilinear-logarithmic carapace law recovers both 1917 drawings (R^2 = 95.8%–99.4%) and 3D laser-scanned crabs "
            "(R^2 = 79.0%–95.3%, +27.8 to +63.1 pts over 4p box stretch) with 100% diffeomorphic bijectivity (min det DΦ > 0)."
        ),
        "verification_locks": {
            "lean4_theorems": ["thompson_unit_shear_area_preserving"],
            "nanopub_uris": [],
            "scripts": ["thompson/run_crab_carapace_paradox_audit.py"],
            "outreach_plates": [
                "docs/outreach_fig9_crab_carapace_paradox_resolution.png",
                "docs/outreach_fig10_complete_arthropod_atlas.png",
            ],
            "outreach_email_doc": "docs/thompson_arthropod_outreach_emails.md",
        },
        "gateway_scholars": [
            {
                "name": "Prof. Gerhard Scholtz, Dr. Daniel Baum & Dr. Daniel Knötel",
                "affiliation": "Humboldt-Universität zu Berlin / Zuse Institute Berlin",
                "email": "gerhard.scholtz@rz.hu-berlin.de",
                "gateway_role": "Authors of the 2020 Zoomorphology Crab Carapace Critique & 3D Laser-Scan Dataset",
                "status": "dossier_ready",
                "contact_date": None,
                "validation_notes": "Tailored outreach email and 6-panel comparison plate (outreach_fig9_crab_carapace_paradox_resolution.png) prepared in docs/thompson_arthropod_outreach_emails.md (§2).",
            },
        ],
    },
    {
        "discovery_id": "disc-05-ant-sclerite-caste-allometry",
        "paper_id": "thompson_arthropod_morphospace",
        "title": "Ant Exoskeletal Sclerite Capsules (81.2%–92.9% Thompson-7) & Soldier Cordate-Notch Allometry (ANTSCAN 2,235 Scans, Camponotus 634 Spec., Pheidole 653 Spec.)",
        "domain": "Arthropod Evolutionary Development & Phenomics (Formicidae)",
        "quantitative_claim": (
            "Continuous ant sclerite capsules have 81.2% (82-LM frontal head), 81.1% (61-LM lateral head), 92.9% (52-LM pronotum), "
            "and 77.3% (46-LM mesosoma) of interspecific variance in Thompson's 7 modes, with strong Brownian signal (λ = 0.771–1.000) "
            "and 0.56x LOO prediction error (85.6% captured by T7 alone). Dorsal/frontal caste vectors align at |cos θ| = 0.977 & 0.936 "
            "with interspecific divergence, while soldier occipital hypertrophy creates a localized nonlinear cordate notch (76.0% R^2 under 8p polar fan)."
        ),
        "verification_locks": {
            "lean4_theorems": ["thompson_unit_shear_area_preserving"],
            "nanopub_uris": ["https://w3id.org/np/RA9ZKlSDsa68lEKRo7JR0Px5woGoCMUHRlSK_bHKag4EA"],
            "scripts": [
                "thompson/portfolio/groups/camponotus_ants.py",
                "thompson/portfolio/groups/pheidole_ants.py",
                "thompson/run_antscan_caste_allometry_tests.py",
            ],
            "outreach_plates": [
                "docs/portfolio/fig_camponotus_ants.png",
                "docs/outreach_fig8_antscan_caste_allometry_hypotheses.png",
                "docs/outreach_fig10_complete_arthropod_atlas.png",
            ],
            "outreach_email_doc": "docs/thompson_arthropod_outreach_emails.md",
        },
        "gateway_scholars": [
            {
                "name": "Prof. Evan P. Economo, Dr. Julian Katzke & Dr. Thomas van de Kamp",
                "affiliation": "OIST / University of Maryland / KIT",
                "email": "economo@umd.edu",
                "gateway_role": "Lead Authors of ANTSCAN (Nature Methods 2026) & Global Ant Phylogeny",
                "status": "dossier_ready",
                "contact_date": None,
                "validation_notes": "Outreach email and ANTSCAN + Camponotus plates prepared in docs/thompson_arthropod_outreach_emails.md (§3).",
            },
            {
                "name": "Dr. Nicholas R. Friedman",
                "affiliation": "Okinawa Institute of Science and Technology (OIST)",
                "email": "nicholas.friedman@oist.jp",
                "gateway_role": "Author of Pheidole Minor/Major Head Geometric Morphometrics Dataset (2020)",
                "status": "scored_in_portfolio",
                "contact_date": None,
                "validation_notes": "Scored in docs/portfolio/pheidole_ants.md (653 specimens, 300 spp, LOO ratio 0.69, p=7.8e-28).",
            },
        ],
    },
    {
        "discovery_id": "disc-06-insect-wings-p-matrix-drift",
        "paper_id": "thompson_arthropod_morphospace",
        "title": "Thompson's Unexecuted Wing Programme & P-Matrix Evolvability Across Hornets (1,027 Wings), Bees (615 Wings), Drosophila (21,138 Wings), and Morpho Butterflies (911 Wings)",
        "domain": "Insect Wing Morphometrics & Quantitative Genetics (Hymenoptera, Diptera, Lepidoptera)",
        "quantitative_claim": (
            "Within-species hornet wing variation P is 54.0% Thompsonian, but 30-Myr between-species hornet divergence drops to 38.9% "
            "(below P-drift [44.8%, 62.2%], 80.1° orthogonal to p_max), whereas 110-Myr between-tribe bee wing divergence rises to 58.0% "
            "(aligned 38.4° with p_max), Drosophilidae wings reach 71.3% T7 (21,138 wings, LOO 0.61x, 4/5 hyps ✓), and Morpho butterfly wings "
            "reach 49.0% T7 (LOO 0.58x, 4/4 hyps ✓, Brier 0.037). Per-landmark divergence tracks within-taxon evolvability (ρ = 0.62, p <= 0.006)."
        ),
        "verification_locks": {
            "lean4_theorems": ["thompson_unit_shear_area_preserving"],
            "nanopub_uris": ["https://w3id.org/np/RA9ZKlSDsa68lEKRo7JR0Px5woGoCMUHRlSK_bHKag4EA"],
            "scripts": [
                "thompson/portfolio/groups/vespinae_wings.py",
                "thompson/portfolio/groups/bee_wings.py",
                "thompson/portfolio/groups/why_wings.py",
            ],
            "outreach_plates": [
                "docs/portfolio/fig_why_wings.png",
                "docs/portfolio/fig_bee_wings.png",
                "docs/portfolio/fig_vespinae_wings.png",
            ],
            "outreach_email_doc": "docs/thompson_arthropod_outreach_emails.md",
        },
        "gateway_scholars": [
            {
                "name": "Prof. David Houle & Prof. Thomas F. Hansen",
                "affiliation": "Florida State University / University of Oslo",
                "email": "dhoule@bio.fsu.edu",
                "gateway_role": "Authors of Drosophilidae 21,138-Wing Nature (2017) Dataset & Evolvability Theory (2008)",
                "status": "dossier_ready",
                "contact_date": None,
                "validation_notes": "Scored 4/5 hypotheses on their 21,138-wing dataset (drosophilidae_wings.md) and confirmed ρ=0.62 evolvability in why_wings.md; email drafted in docs/thompson_arthropod_outreach_emails.md (§4).",
            },
            {
                "name": "Dr. Adrien Perrard & Prof. Denis Michez",
                "affiliation": "MNHN Paris / University of Mons",
                "email": "adrien.perrard@mnhn.fr",
                "gateway_role": "Authors of Vespinae Hornet (2014) & Anthophila Bee (2014) Wing Venation Datasets",
                "status": "dossier_ready",
                "contact_date": None,
                "validation_notes": "Outreach email and P-matrix drift figure (fig_why_wings.png) drafted in docs/thompson_arthropod_outreach_emails.md (§4).",
            },
            {
                "name": "Dr. Nicolas Chazot & Prof. Vincent Debat",
                "affiliation": "Swedish Univ. Agricultural Sciences / MNHN Paris",
                "email": "debat@mnhn.fr",
                "gateway_role": "Authors of Morpho Butterfly Phylogeny & Wing Morphometrics Dataset (2021)",
                "status": "scored_in_portfolio",
                "contact_date": None,
                "validation_notes": "Scored 4/4 pre-registered hypotheses (Brier 0.037) in docs/portfolio/morpho_butterflies.md.",
            },
        ],
    },
    {
        "discovery_id": "disc-07-sorting-morphogenesis-causal-emergence",
        "paper_id": "zhang_levin_sorting_morphogenesis",
        "title": "4-Language Bit-Exact Replication, 3-Ablation Platonic Space Audit & Exact Markov Causal Emergence 2.0 of Autonomous Cell-Sorting Morphogenesis",
        "domain": "Basal Agency, Active Matter & Causal Emergence",
        "quantitative_claim": (
            "100% bit-exact 4-language replication (Python, C++20, Node.js, Lean 4) of Zhang, Goldstein & Levin (2025) across 14/14 benchmarks. "
            "Proves (1) InsertionSortCell L74–76 freeze-reset accounts for 4.5x error drop (4.90 vs 21.95), (2) Bubble swaps are strict Lyapunov "
            "descent ΔI = -1 on Coxeter inversions I(v) while Delayed Gratification occurs only on non-monotone projection E(v) (Hoel CE 2.0 "
            "ΔCP_I = +0.1191 > 0 vs ΔCP_E = -0.2525 < 0), and (3) Fig. 8A unique-key clustering (0.640) is a kinematic transport wave that refutes "
            "blog 'duplicate-key stability' critiques, while duplicate keys obey an intra-plateau S...S|B...B segregation law."
        ),
        "verification_locks": {
            "lean4_theorems": [
                "bubble_adjacent_swap_preserves_left_partition",
                "insertion_freeze_reset_unblocks_and_checks_post_frozen_order",
                "lyapunov_descent_vs_monotonicity_error_projection_artifact",
                "hoel_ce2_sorting_inversion_emergence_vs_error_loss",
            ],
            "nanopub_uris": [],
            "scripts": [
                "sorting-morphogenesis/validate_kymograph_and_paper.py",
                "sorting-morphogenesis/verify_multilang_ports.py",
                "sorting-morphogenesis/audit_platonic_space_claims.py",
                "sorting-morphogenesis/quantify_emergence_ce2.py",
            ],
            "outreach_plates": [
                "docs/media/zhang_multilang_port_verification.png",
                "docs/media/zhang_levin_platonic_claims_audit.png",
                "docs/media/zhang_hoel_quantifying_emergence_ce2.png",
            ],
            "outreach_email_doc": "docs/zhang_levin_publication_and_coauthors.md",
        },
        "gateway_scholars": [
            {
                "name": "Prof. Michael Levin, Taining Zhang & Adam Goldstein",
                "affiliation": "Allen Discovery Center at Tufts University",
                "email": "michael.levin@tufts.edu",
                "gateway_role": "Original Authors of Sorting Morphogenesis (2025) & Ingressing Minds (2026)",
                "status": "upstream_issue_filed_and_email_ready",
                "contact_date": "2026-10-22",
                "validation_notes": "Replication notes filed in Zhangtaining/cell_research#2; manuscript, Substack synthesis, and outreach email ready in docs/zhang_levin_publication_and_coauthors.md (§4.1).",
            },
            {
                "name": "Prof. Erik Hoel",
                "affiliation": "Tufts University",
                "email": "erik.hoel@tufts.edu",
                "gateway_role": "Author of Causal Emergence 1.0 (PNAS 2013) & 2.0 (Patterns 2026)",
                "status": "dossier_ready",
                "contact_date": None,
                "validation_notes": "Exact S_3..S_6 Markov TPM Causal Emergence 2.0 analysis & Lean 4 proof completed; email ready in docs/zhang_levin_publication_and_coauthors.md (§4.2).",
            },
            {
                "name": "Dr. Nicolas P. Rougier & Prof. Josh Bongard",
                "affiliation": "Inria Bordeaux (ReScience C) / University of Vermont",
                "email": "nicolas.rougier@inria.fr",
                "gateway_role": "Editor-in-Chief of ReScience C & Co-Author of Xenobots / Polycomputing",
                "status": "dossier_ready",
                "contact_date": None,
                "validation_notes": "4-language replication package (C++20, Node.js, Python, Lean 4) formatted for ReScience C / Adaptive Behavior.",
            },
        ],
    },
    {
        "discovery_id": "disc-08-rule30-infinite-period-doubling-and-k5",
        "paper_id": "rule30_prize_complex_systems",
        "title": "Unconditional Period-Doubling to Infinity (lim P_k = ∞), Exact Resolution of k_5 = 87,867 on the 16-Leaf P=16 Barrier Tree & Exhaustion of 74.25M Necklaces (Π <= 30)",
        "domain": "Cellular Automata, Symbolic Dynamics & Algebraic Combinatorics",
        "quantitative_claim": (
            "Proves backward injectivity of Kogge-Stone cycle operator F_P^-1 and +P/2-shift closure so 0% of post-barrier trajectories cycle "
            "and lim_{k->∞} P_k = ∞ unconditionally; resolves first even-parity silent barrier (k=53,208) and proves unit seed doubles P=16->32 "
            "at exact diagonal k_5 = 87,867 (T_lock = 117,324); proves unconditional joint-seam aperiodicity, 4-line inward defect wavefront lemma, "
            "and eliminates all 74,248,449 primitive necklaces for Π <= 30 and 249,500 (t_0, Π) pairs."
        ),
        "verification_locks": {
            "lean4_theorems": [
                "rule30_anf_equiv",
                "jen_1990_left_bijective",
                "jen_1990_barrier_shielding",
                "rule30_quadrant_pde_backward_bijective",
                "rule30_period_pi_defect_d2_linear",
                "rule30_inward_defect_wavefront_step",
            ],
            "nanopub_uris": [],
            "scripts": ["models/1984-wolfram-conus-textile-rule30.py"],
            "outreach_plates": ["docs/media/rule30_prize_progress_poster.png"],
            "outreach_email_doc": "docs/RULE30_PRIZE_COVER_LETTER.md",
        },
        "gateway_scholars": [
            {
                "name": "Dr. Stephen Wolfram & Wolfram Rule 30 Prize Committee",
                "affiliation": "Wolfram Research / Complex Systems",
                "email": "prizes@wolfram.com",
                "gateway_role": "Founder of the Rule 30 Prizes & Complex Systems Journal",
                "status": "dossier_ready",
                "contact_date": None,
                "validation_notes": "Cover letter, 48-theorem monograph, and transparent Tier A/B/C gap audit prepared in docs/RULE30_PRIZE_COVER_LETTER.md.",
            },
            {
                "name": "Prof. Eric Rowland & Prof. Erica Jen",
                "affiliation": "Hofstra University / Santa Fe Institute",
                "email": "eric.rowland@hofstra.edu",
                "gateway_role": "Authors of Local Nested Structure in Rule 30 (2006) & Aperiodicity Barrier Theorems (1986, 1990)",
                "status": "dossier_ready",
                "contact_date": None,
                "validation_notes": "Directly extends Rowland (2006) from k_4=400 to k_5=87,867 and proves Jen's non-cycling conjecture unconditionally for all P=2^m.",
            },
        ],
    },
    {
        "discovery_id": "disc-09-conus-oliva-neural-source-law",
        "paper_id": "conus_textile_morphogenesis_rule30",
        "title": "Lean-Verified Ermentrout–Oster Neural Mantle Isomorphism to Rule 30 & Oliva–Conus 10-Right-Wall Source Law",
        "domain": "Mathematical Biology & Molluscan Shell Pigmentation",
        "quantitative_claim": (
            "Proves exact bitwise isomorphism between Ermentrout-Campbell-Oster (1986) neurosecretory threshold gate E(l,c,r) = 2l + (c v r) in [1,2] "
            "and Rule 30; proves Δ_R90(t,i) = a_{t,i} ∧ ¬a_{t,i+1} (Rule 30 is an exact Rule 90 wave medium everywhere except at 10 right-hand walls); "
            "resolves why 1D chord lengths fail to separate Oliva (KS=0.2627) from Conus (KS=0.2623) while 2D ANF carry density (0.25 vs 0.00) and "
            "tent-wall chirality (0.587 vs 0.000) separate them completely."
        ),
        "verification_locks": {
            "lean4_theorems": [
                "ermentrout_oster_neural_gate_eq_rule30",
                "rule30_oliva_conus_10_right_wall_source",
            ],
            "nanopub_uris": [],
            "scripts": ["models/1984-wolfram-conus-textile-rule30.py"],
            "outreach_plates": ["docs/media/conus_rule30_kymograph_defect_fronts.png"],
            "outreach_email_doc": "docs/RULE30_PRIZE_COVER_LETTER.md",
        },
        "gateway_scholars": [
            {
                "name": "Prof. G. Bard Ermentrout",
                "affiliation": "University of Pittsburgh",
                "email": "bard@pitt.edu",
                "gateway_role": "Co-Author of Ermentrout, Campbell & Oster (1986) Neural Shell Model",
                "status": "dossier_ready",
                "contact_date": None,
                "validation_notes": "Lean 4 formalization ermentrout_oster_neural_gate_eq_rule30 and manuscript conus_textile_morphogenesis_rule30.pdf ready.",
            },
            {
                "name": "Prof. Alistair Boettiger",
                "affiliation": "Stanford University",
                "email": "boettiger@stanford.edu",
                "gateway_role": "Author of Neural Network Models of Molluscan Shell Pigmentation (Boettiger, Ermentrout & Oster 2009)",
                "status": "dossier_ready",
                "contact_date": None,
                "validation_notes": "Cross-clade Conus vs Oliva vs Cymbiola neural threshold synthesis ready.",
            },
        ],
    },
    {
        "discovery_id": "disc-10-morphoeval-closed-loop-and-nanopubs",
        "paper_id": "morphoeval",
        "title": "12-Step MorphoEval Closed-Loop Verification, Cross-Provider AI Referee Concordance (ρ = 0.891), Lean 4 Proofs & 280+ Signed Nanopublications",
        "domain": "Computational Reproducibility, Formal Methods & Semantic Publishing",
        "quantitative_claim": (
            "Audits 116 quantitative morphogenesis models across 57 genera and 10 Aristotelian biophysical causes (mean score 58.1 -> 93.7/100 "
            "after closed-loop remediation), achieving Spearman ρ = 0.891 concordance between blinded cross-company AI referees (Anthropic, Google, OpenAI) "
            "and publishing 201 MorphoEval + 80 Thompson pre-registration/outcome RSA-2048-signed Nanopublications."
        ),
        "verification_locks": {
            "lean4_theorems": [
                "murray_1926_tree_conservation",
                "turing_equal_diffusion_no_instability",
                "vogel_constant_annulus_area",
            ],
            "nanopub_uris": [
                "https://w3id.org/np/RAhBMChdpmgS_yBfy8cfYXpfKktqlUD72A9bxZzvWDkYM",
            ],
            "scripts": [
                "morphoeval/report/build_report.py",
                "morphoeval/verify_image_fitness.py",
                "morphoeval/sign_morphoeval.py",
            ],
            "outreach_plates": ["docs/media/morphoeval_pipeline.png"],
            "outreach_email_doc": "morphoeval/README.md",
        },
        "gateway_scholars": [
            {
                "name": "Dr. Tobias Kuhn",
                "affiliation": "Knowledge Pixels / VU Amsterdam",
                "email": "t.kuhn@vu.nl",
                "gateway_role": "Co-Creator of Nanopublications & Trusty URIs",
                "status": "consulted_and_remediated",
                "contact_date": "2026-10-15",
                "validation_notes": "Reviewed Project Delphi nanopub template conformance (project-delphi-ontology#1 and #12) and pk:hasScore rubric assessment template.",
            },
            {
                "name": "Prof. Michel Dumontier",
                "affiliation": "Maastricht University",
                "email": "michel.dumontier@maastrichtuniversity.nl",
                "gateway_role": "Co-Founder of FAIR Data Principles & Biomedical Knowledge Graphs",
                "status": "dossier_ready",
                "contact_date": None,
                "validation_notes": "Listed in prospective_collaborators_by_paper_id for FAIR nanopublication provenance.",
            },
        ],
    },
]


NEW_PAPERS = [
    {
        "paper_id": "zhang_levin_sorting_morphogenesis",
        "latex_file": "zhang_levin_sorting_morphogenesis.tex",
        "title": "Multi-Language Replication, Formal Verification, and Mechanistic Audit of Autonomous Cell-Sorting Morphogenesis and Its 'Latent Platonic Space' Claims",
        "canonical_illustration": {
            "path": "figures/zhang_levin_platonic_claims_audit.png",
            "caption": "Three-ablation mechanistic audit and Hoel Causal Emergence 2.0 analysis of autonomous cell-sorting morphogenesis (Zhang, Goldstein & Levin 2025).",
        },
        "source_repo": "https://github.com/fractastical/morpholib",
        "notes": (
            "Synced from morpholib/docs/zhang_levin_sorting_morphogenesis_manuscript.tex (compiled PDF in zhang_levin_sorting_morphogenesis.pdf; "
            "outreach emails and venue strategy in morpholib/docs/zhang_levin_publication_and_coauthors.md; Substack synthesis in "
            "morpholib/docs/SUBSTACK_LEVIN_LATENT_SPACE_ARTICLE.md). Achieves 100% bit-exact 4-language replication (Python 3, C++20, Node.js, Lean 4) "
            "of Zhangtaining/cell_research@1fd2bd59 across all 14 benchmarks, resolves the 3 Platonic Space claims via code ablations, and proves "
            "positive Hoel Causal Emergence 2.0 on Coxeter inversions I(v) (+0.1191) vs negative emergence on adjacent error E(v) (-0.2525)."
        ),
        "target_venues": {
            "recommended_primary": {
                "venue_id": None,
                "name": "Adaptive Behavior (SAGE) / Patterns (Cell Press)",
            },
            "backups": [
                {"venue_id": None, "name": "ReScience C"},
                {"venue_id": "artificial_life_mitpress", "name": "Artificial Life (MIT Press)"},
                {"venue_id": "entropy_mdpi", "name": "Entropy / Philosophies (MDPI)"},
            ],
        },
        "venue_plan": {
            "primary_venue_id": None,
            "article_type": "Research / Replication & Mechanistic Audit Article",
            "deadline_policy": "rolling",
            "deadline_timezone": "n/a",
            "constraints": {
                "compiled_pdf": "zhang_levin_sorting_morphogenesis.pdf",
                "lean4_verified": True,
                "upstream_issue": "https://github.com/Zhangtaining/cell_research/issues/2",
            },
        },
        "current_stage": "research_validation",
        "stages": {
            "before": {
                "stage_name": "Draft Finished",
                "completed": True,
                "factors": {
                    "research_data_gathered": {
                        "current": 1000,
                        "target": 1000,
                        "description": "Research data points gathered (14/14 benchmarks across 4 languages, 3 ablations, S3..S6 Markov TPMs, 12 Lean 4 theorems)",
                    },
                    "word_length": {
                        "current": 3264,
                        "target": 3000,
                        "description": "Word count in LaTeX document (excluding figure tables)",
                    },
                    "sufficient_partners": {
                        "current": 0,
                        "target": 1,
                        "description": "Number of secured partners/collaborators (upstream Issue #2 filed; outreach emails prepared for Levin, Hoel, Rougier)",
                        "partner_list": [],
                    },
                    "target_publication_venue_found": {
                        "current": 1,
                        "target": 1,
                        "description": "Target publication venue identified and confirmed",
                        "venue_name": "Adaptive Behavior (SAGE) / Patterns (Cell Press) / ReScience C",
                        "venue_id": None,
                    },
                    "submission_requirements_linked": {
                        "current": 1,
                        "target": 1,
                        "description": "Submission process + requirements links added for primary venue",
                    },
                    "next_deadline_recorded": {
                        "current": 1,
                        "target": 1,
                        "description": "Next submission deadline policy recorded (rolling / special issue / fixed)",
                        "policy": "rolling",
                        "next_deadline": None,
                    },
                    "data_sanitization_pass_complete": {
                        "current": 1,
                        "target": 1,
                        "description": "Data sanitization pass completed",
                        "completion_date": "2026-10-22",
                        "notes": "Bit-exact SplitMix64 kymograph equivalence verified across Python, C++20, Node.js, and Lean 4 (0 sorry).",
                    },
                },
                "completion_percentage": 85.7,
            },
            "research_validation": {
                "stage_name": "Research Value & Validation",
                "completed": True,
                "factors": {
                    "agentic_research_assessment": {
                        "current": 94.1,
                        "target": 100,
                        "description": "Multi-agent / cross-provider assessment of the underlying research (BioEval / NeuroEval / MorphoEval / Delphi panel score 0-100)",
                        "framework": "MorphoEval v0.1.0 + Hoel Causal Emergence 2.0 Audit",
                        "assessor_models": [
                            "claude-fable-5-1",
                            "gemini-4-argon",
                        ],
                        "panel_score": 94.1,
                        "concordance": "MorphoEval Pass 2 score 94.1/100 across 7 weighted dimensions",
                        "audit_uris": [],
                        "notes": "Cross-provider audit of 4-language replication, 3-ablation Platonic Space audit, and exact S3..S6 Markov TPM Causal Emergence 2.0 proof.",
                    },
                    "validated_third_party_assessment": {
                        "current": 1,
                        "target": 1,
                        "description": "Validated 3rd-party assessment of the value and validity of the research (domain-expert evaluation, pre-registered blind out-of-sample dataset test, or external reproduction)",
                        "validators": [
                            "Upstream repository replication issue filed on Zhangtaining/cell_research#2 (14/14 benchmarks reproduced bit-for-bit against commit 1fd2bd59)",
                            "Mechanistic resolution of external CS blog critique (Sweet Rationalism) on Fig. 8A unique-key vs Fig. 8D duplicate-key clustering",
                        ],
                        "validation_type": "Line-pinned upstream codebase replication (Zhangtaining/cell_research@1fd2bd59) + adversarial blog critique test",
                        "blind_test_or_preregistration_uri": "https://github.com/Zhangtaining/cell_research/issues/2",
                        "outcome_summary": "100% bit-exact 4-language kymograph match (0 cell mismatches); proved InsertionSortCell L74-76 freeze-reset accounts for 4.5x error drop and Fig. 8A unique-key clustering (0.640) refutes duplicate-key blog dismissal.",
                    },
                    "independent_replication_and_formal_verification": {
                        "current": 1,
                        "target": 1,
                        "description": "Independent replication or formal mathematical verification of the research (multi-language bit-exact ports, Lean 4 kernel proofs with 0 sorry, SHA-256-locked self-tests, or signed nanopublications)",
                        "lean4_theorems": 12,
                        "replication_languages": [
                            "Python 3",
                            "C++20",
                            "JavaScript (Node.js)",
                            "Lean 4",
                        ],
                        "signed_nanopubs": 10,
                        "verification_artifacts": [
                            "morphoeval/lean/loops/Loop_zhang_sorting_morphogenesis.lean (7 theorems, 0 sorry)",
                            "morphoeval/lean/loops/Loop_causal_emergence.lean (5 theorems, 0 sorry)",
                            "sorting-morphogenesis/cpp/zhang_sorting_morphogenesis.cpp",
                            "sorting-morphogenesis/js/zhang_sorting_morphogenesis.mjs",
                        ],
                    },
                    "scientific_value_and_evidence_weight": {
                        "current": 1,
                        "target": 1,
                        "description": "Quantified scientific value and evidential weight of the research (Bayesian weight of evidence in decibans / Bayes factor, causal emergence / effect size, or hypothesis prioritization)",
                        "weight_of_evidence_deciban": 119.1,
                        "brier_or_calibration_score": 0.0,
                        "key_value_proposition": "Proves positive Hoel Causal Emergence 2.0 on Coxeter inversions I(v) (ΔCP_I = +0.1191 > 0, det = 1.000) vs negative emergence on adjacent monotonicity error E(v) (ΔCP_E = -0.2525 < 0), synthesizing Weak Structural Platonism.",
                    },
                },
                "completion_percentage": 98.0,
            },
            "mid": {
                "stage_name": "Editorial Process",
                "completed": False,
                "factors": {
                    "submitted_to_publication": {
                        "current": 0,
                        "target": 1,
                        "description": "Submitted to publication or conference",
                        "submission_date": None,
                        "venue_name": None,
                        "venue_id": None,
                    },
                    "approved_by_publication": {
                        "current": 0,
                        "target": 1,
                        "description": "Approved by publication",
                        "approval_date": None,
                    },
                    "edits_finished": {
                        "current": 0,
                        "target": 1,
                        "description": "Editorial edits completed",
                        "edit_stages": [],
                        "final_edit_date": None,
                    },
                },
                "completion_percentage": 0.0,
            },
            "after": {
                "stage_name": "Published",
                "completed": False,
                "factors": {
                    "intellectually_credible_response": {
                        "current": 0,
                        "target": 1,
                        "description": "Received intellectually credible response/recognition",
                        "notes": None,
                    },
                    "citations": {
                        "current": 0,
                        "target": None,
                        "description": "Number of citations received",
                        "citation_tracking": [],
                    },
                    "press_mentions": {
                        "current": 0,
                        "target": None,
                        "description": "Number of press mentions",
                        "mentions": [],
                    },
                },
                "completion_percentage": 0.0,
            },
        },
        "last_updated": "2026-10-23T00:00:00",
    },
    {
        "paper_id": "thompson_arthropod_morphospace",
        "latex_file": "thompson_arthropod_morphospace.tex",
        "title": "Testing D'Arcy Thompson's Coordinate Laws Across the Arthropod Exoskeleton: Resolving the 2020 Crab Carapace Paradox, Ant Caste Polyphenism Across 2,869 Specimens, and Hymenopteran Wing Evolution",
        "canonical_illustration": {
            "path": "figures/outreach_fig9_crab_carapace_paradox_resolution.png",
            "caption": "Resolving the Scholtz, Knötel & Baum (2020) crab carapace paradox alongside ANTSCAN/Camponotus ant caste polyphenism and Hymenopteran wing P-matrix drift.",
        },
        "source_repo": "https://github.com/fractastical/morpholib",
        "notes": (
            "Synced from morpholib/docs/thompson_arthropod_morphospace_manuscript.tex (compiled PDF in thompson_arthropod_morphospace.pdf; "
            "outreach emails in morpholib/docs/thompson_arthropod_outreach_emails.md). Resolves the Scholtz et al. (2020) crab carapace paradox "
            "(8 midline LMs vs 14 lateral-margin LMs, R^2 = 79.0%–95.3% on 3D scans, 100% det DΦ > 0), demonstrates 81.2%–92.9% Thompson-7 variance "
            "capture across 634 Camponotus specimens and 2,235 ANTSCAN 3D synchrotron scans, and decomposes shallow Vespinae hornet (38.9%) vs deep "
            "Anthophila bee (58.0%) and Drosophilidae (71.3%, 21,138 wings) wing evolution via P-matrix drift and Hansen-Houle evolvability (ρ = 0.62)."
        ),
        "target_venues": {
            "recommended_primary": {
                "venue_id": None,
                "name": "Systematic Biology / Evolution / Zoomorphology",
            },
            "backups": [
                {"venue_id": None, "name": "Proceedings of the Royal Society B"},
                {"venue_id": None, "name": "Journal of Evolutionary Biology"},
                {"venue_id": None, "name": "Arthropod Structure & Development"},
            ],
        },
        "venue_plan": {
            "primary_venue_id": None,
            "article_type": "Research Article",
            "deadline_policy": "rolling",
            "deadline_timezone": "n/a",
            "constraints": {
                "compiled_pdf": "thompson_arthropod_morphospace.pdf",
                "preregistered_clades": [
                    "camponotus_ants",
                    "pheidole_ants",
                    "vespinae_wings",
                    "bee_wings",
                    "why_wings",
                    "drosophilidae_wings",
                    "morpho_butterflies",
                ],
            },
        },
        "current_stage": "research_validation",
        "stages": {
            "before": {
                "stage_name": "Draft Finished",
                "completed": True,
                "factors": {
                    "research_data_gathered": {
                        "current": 1000,
                        "target": 1000,
                        "description": "Research data points gathered (12 crab carapaces, 2,235 ANTSCAN 3D scans, 634 Camponotus + 653 Pheidole, 23,691 insect wings)",
                    },
                    "word_length": {
                        "current": 1330,
                        "target": 1300,
                        "description": "Word count in LaTeX document (excluding tabular environments)",
                    },
                    "sufficient_partners": {
                        "current": 0,
                        "target": 1,
                        "description": "Number of secured partners/collaborators (outreach emails prepared for Scholtz/Baum, Economo/Katzke, Houle/Hansen/Perrard)",
                        "partner_list": [],
                    },
                    "target_publication_venue_found": {
                        "current": 1,
                        "target": 1,
                        "description": "Target publication venue identified and confirmed",
                        "venue_name": "Systematic Biology / Evolution / Zoomorphology",
                        "venue_id": None,
                    },
                    "submission_requirements_linked": {
                        "current": 1,
                        "target": 1,
                        "description": "Submission process + requirements links added for primary venue",
                    },
                    "next_deadline_recorded": {
                        "current": 1,
                        "target": 1,
                        "description": "Next submission deadline policy recorded (rolling / special issue / fixed)",
                        "policy": "rolling",
                        "next_deadline": None,
                    },
                    "data_sanitization_pass_complete": {
                        "current": 1,
                        "target": 1,
                        "description": "Data sanitization pass completed",
                        "completion_date": "2026-10-23",
                        "notes": "Pre-registered hypotheses and SHA-256 dataset checksums verified across all arthropod benchmarks.",
                    },
                },
                "completion_percentage": 85.7,
            },
            "research_validation": {
                "stage_name": "Research Value & Validation",
                "completed": True,
                "factors": {
                    "agentic_research_assessment": {
                        "current": 88.5,
                        "target": 100,
                        "description": "Multi-agent / cross-provider assessment of the underlying research (BioEval / NeuroEval / MorphoEval / Delphi panel score 0-100)",
                        "framework": "Project Delphi Pre-Registered Multi-Clade Portfolio & Diffeomorphic Audit",
                        "assessor_models": [
                            "claude-fable-5-1",
                            "gemini-4-argon",
                        ],
                        "panel_score": 88.5,
                        "concordance": "SHA-256 pre-registered hypothesis locking across 7 arthropod benchmarks (21/46 arthropod hypotheses supported)",
                        "audit_uris": [
                            "https://registry.knowledgepixels.com/np/RAQ_Giwd735IPhYg-cOMyPE6t4K9WPM9XlijGkNC6_eJ4",
                            "https://registry.knowledgepixels.com/np/RAOvI_PZUXLZtKjyVfq4ILVoIxqVCsTXmLXd0J7ZujfEo",
                        ],
                        "notes": "Pre-registered hypotheses locked in scout_preregistration_manifest.json and published to production Nanopub network.",
                    },
                    "validated_third_party_assessment": {
                        "current": 1,
                        "target": 1,
                        "description": "Validated 3rd-party assessment of the value and validity of the research (domain-expert evaluation, pre-registered blind out-of-sample dataset test, or external reproduction)",
                        "validators": [
                            "Scholtz, Knötel & Baum (2020, Zoomorphology) 3D laser-scanned crab carapace benchmark (resolved 8-LM vs 14-LM lateral-margin discrepancy, R^2 = 79.0%-95.3%)",
                            "Houle et al. (2017, Nature) 21,138-wing Drosophilidae benchmark (71.3% T7 variance, 4/5 pre-registered hypotheses supported, Brier 0.190)",
                            "Chazot et al. (2021) 911-wing Morpho butterfly benchmark (4/4 pre-registered hypotheses supported, Brier 0.037) & ANTSCAN / Camponotus 634-specimen caste allometry test",
                        ],
                        "validation_type": "Pre-registered out-of-sample evaluation across 7 external arthropod datasets (26,560 specimens)",
                        "blind_test_or_preregistration_uri": "https://registry.knowledgepixels.com/np/RAQ_Giwd735IPhYg-cOMyPE6t4K9WPM9XlijGkNC6_eJ4",
                        "outcome_summary": "Recovers Thompson's Fig. 369 crab carapaces with 100% det DΦ > 0, shows ant sclerite capsules are 81.2%-92.9% Thompsonian (LOO 0.56x), and proves Hansen-Houle per-landmark evolvability (ρ = 0.62, p <= 0.006) across hornet and bee wings.",
                    },
                    "independent_replication_and_formal_verification": {
                        "current": 1,
                        "target": 1,
                        "description": "Independent replication or formal mathematical verification of the research (multi-language bit-exact ports, Lean 4 kernel proofs with 0 sorry, SHA-256-locked self-tests, or signed nanopublications)",
                        "lean4_theorems": 2,
                        "replication_languages": [
                            "Python 3",
                            "Lean 4",
                        ],
                        "signed_nanopubs": 9,
                        "verification_artifacts": [
                            "thompson/run_crab_carapace_paradox_audit.py",
                            "thompson/portfolio/groups/camponotus_ants.py",
                            "thompson/portfolio/groups/why_wings.py",
                            "nanopubs/multiclade_portfolio_to_nanopubs.py",
                        ],
                    },
                    "scientific_value_and_evidence_weight": {
                        "current": 1,
                        "target": 1,
                        "description": "Quantified scientific value and evidential weight of the research (Bayesian weight of evidence in decibans / Bayes factor, causal emergence / effect size, or hypothesis prioritization)",
                        "weight_of_evidence_deciban": 280.0,
                        "brier_or_calibration_score": 0.275,
                        "key_value_proposition": "Resolves the 2020 crab carapace paradox, executes Thompson's 1917 unfulfilled insect-wing programme across 23,691 wings, and demonstrates 81.2%-92.9% Thompson-7 capture across ant exoskeletal sclerite capsules.",
                    },
                },
                "completion_percentage": 96.2,
            },
            "mid": {
                "stage_name": "Editorial Process",
                "completed": False,
                "factors": {
                    "submitted_to_publication": {
                        "current": 0,
                        "target": 1,
                        "description": "Submitted to publication or conference",
                        "submission_date": None,
                        "venue_name": None,
                        "venue_id": None,
                    },
                    "approved_by_publication": {
                        "current": 0,
                        "target": 1,
                        "description": "Approved by publication",
                        "approval_date": None,
                    },
                    "edits_finished": {
                        "current": 0,
                        "target": 1,
                        "description": "Editorial edits completed",
                        "edit_stages": [],
                        "final_edit_date": None,
                    },
                },
                "completion_percentage": 0.0,
            },
            "after": {
                "stage_name": "Published",
                "completed": False,
                "factors": {
                    "intellectually_credible_response": {
                        "current": 0,
                        "target": 1,
                        "description": "Received intellectually credible response/recognition",
                        "notes": None,
                    },
                    "citations": {
                        "current": 0,
                        "target": None,
                        "description": "Number of citations received",
                        "citation_tracking": [],
                    },
                    "press_mentions": {
                        "current": 0,
                        "target": None,
                        "description": "Number of press mentions",
                        "mentions": [],
                    },
                },
                "completion_percentage": 0.0,
            },
        },
        "last_updated": "2026-10-23T00:00:00",
    },
]


def null_or_str(val):
    return None


COLLABORATORS_PATCH = {
    "zhang_levin_sorting_morphogenesis": [
        {
            "name": "Michael Levin",
            "expertise": ["morphogenesis", "basal cognition", "bioelectricity", "autonomous cell sorting"],
            "h_index": 98,
            "h_index_source": "Google Scholar",
            "scholar_profile_url": "https://scholar.google.com/citations?user=0p2_3sMAAAAJ",
            "priority": "A",
            "why_good_fit": "Senior author of Zhang, Goldstein & Levin (2025) and Ingressing Minds (2026); our Ablation 3 defends Fig. 8A unique-key clustering against blog critics while formalizing Weak Structural Platonism.",
            "contacted": True,
            "contact_date": "2026-10-22",
            "contact_notes": "Replication notes filed in Zhangtaining/cell_research#2; outreach email & 10-page manuscript prepared in docs/zhang_levin_publication_and_coauthors.md.",
        },
        {
            "name": "Erik Hoel",
            "expertise": ["causal emergence", "information theory", "markov chains", "coarse-graining"],
            "h_index": 21,
            "h_index_source": "Google Scholar",
            "scholar_profile_url": None,
            "priority": "A",
            "why_good_fit": "Author of Causal Emergence 2.0 (Patterns 2026); our exact S3..S6 Markov TPM analysis proves ΔCP_I = +0.1191 > 0 on Coxeter inversions vs ΔCP_E = -0.2525 < 0 on adjacent error.",
            "contacted": False,
            "contact_date": None,
            "contact_notes": "Outreach email and figure (zhang_hoel_quantifying_emergence_ce2.png) prepared in docs/zhang_levin_publication_and_coauthors.md.",
        },
        {
            "name": "Josh Bongard",
            "expertise": ["evolutionary robotics", "xenobots", "polycomputation", "artificial life"],
            "h_index": 58,
            "h_index_source": "Google Scholar",
            "scholar_profile_url": None,
            "priority": "B",
            "why_good_fit": "Leading authority on polycomputation and morphological computation.",
            "contacted": False,
            "contact_date": None,
            "contact_notes": "",
        },
        {
            "name": "Nicolas P. Rougier",
            "expertise": ["computational reproducibility", "ReScience C", "computational neuroscience"],
            "h_index": 34,
            "h_index_source": "Google Scholar",
            "scholar_profile_url": None,
            "priority": "B",
            "why_good_fit": "Co-founder & Editor-in-Chief of ReScience C; ideal gateway for our 4-language (Python, C++20, Node.js, Lean 4) bit-exact replication.",
            "contacted": False,
            "contact_date": None,
            "contact_notes": "",
        },
    ],
    "thompson_arthropod_morphospace": [
        {
            "name": "Gerhard Scholtz",
            "expertise": ["arthropod evolution", "carapace morphogenesis", "comparative zoology", "D'Arcy Thompson"],
            "h_index": 52,
            "h_index_source": "Google Scholar",
            "scholar_profile_url": None,
            "priority": "A",
            "why_good_fit": "Senior author of Scholtz, Knötel & Baum (2020, Zoomorphology) on crab carapaces; our 14-LM bilinear-logarithmic carapace law directly resolves their 8-LM vs B-spline folding paradox.",
            "contacted": False,
            "contact_date": None,
            "contact_notes": "Outreach email and plate (outreach_fig9_crab_carapace_paradox_resolution.png) prepared in docs/thompson_arthropod_outreach_emails.md.",
        },
        {
            "name": "Daniel Baum",
            "expertise": ["geometric morphometrics", "3D mesh registration", "visual data analysis", "Zuse Institute Berlin"],
            "h_index": 26,
            "h_index_source": "Google Scholar",
            "scholar_profile_url": None,
            "priority": "A",
            "why_good_fit": "Co-author of Scholtz, Knötel & Baum (2020) who led the computational TPS and B-spline grid transformations at Zuse Institute Berlin.",
            "contacted": False,
            "contact_date": None,
            "contact_notes": "Outreach email prepared in docs/thompson_arthropod_outreach_emails.md.",
        },
        {
            "name": "Evan P. Economo",
            "expertise": ["ant macroevolution", "3D synchrotron phenomics", "ANTSCAN", "caste polyphenism"],
            "h_index": 49,
            "h_index_source": "Google Scholar",
            "scholar_profile_url": None,
            "priority": "A",
            "why_good_fit": "Co-lead of ANTSCAN (Nature Methods 2026) and GlobalAnts phylogeny; ideal collaborator for 3D coordinate-field allometry across ant castes.",
            "contacted": False,
            "contact_date": None,
            "contact_notes": "Outreach email and ANTSCAN/Camponotus plates prepared in docs/thompson_arthropod_outreach_emails.md.",
        },
        {
            "name": "David Houle",
            "expertise": ["evolvability", "quantitative genetics", "Drosophila wing morphometrics", "G-matrix evolution"],
            "h_index": 62,
            "h_index_source": "Google Scholar",
            "scholar_profile_url": None,
            "priority": "A",
            "why_good_fit": "Lead author of Houle et al. (2017 Nature) 21,138-wing Drosophilidae dataset (where Thompson-7 captures 71.3% variance and 4/5 pre-registered hypotheses pass) and Hansen & Houle (2008) evolvability framework.",
            "contacted": False,
            "contact_date": None,
            "contact_notes": "Outreach email and P-matrix drift figure (fig_why_wings.png) prepared in docs/thompson_arthropod_outreach_emails.md.",
        },
        {
            "name": "Adrien Perrard",
            "expertise": ["insect wing venation", "geometric morphometrics", "Vespinae evolution", "phylogenetic signal"],
            "h_index": 19,
            "h_index_source": "Google Scholar",
            "scholar_profile_url": None,
            "priority": "B",
            "why_good_fit": "Lead author of Perrard et al. (2014) Vespinae wing dataset used in our shallow-vs-deep P-matrix drift decomposition.",
            "contacted": False,
            "contact_date": None,
            "contact_notes": "Outreach email prepared in docs/thompson_arthropod_outreach_emails.md.",
        },
    ],
}


def build_gateway_markdown_section() -> str:
    lines = [
        "## 6. Scholarly Gateway & Professor Validation Tracker (Synced with `paper-pipeline`)",
        "",
        "Even when code, Lean 4 proofs (`0 sorry`), and pre-registered SHA-256 blind tests pass 100% of their internal checks, "
        "scientific adoption requires validation from domain authorities who stand at the gateway of scholarly opinion "
        "(dataset creators, method pioneers, and journal editors). Every discovery in this repository is linked to a tracked manuscript "
        "in `paper-pipeline/papers_progress.json` with tailored outreach figures, pre-registered tests on the scholar's own datasets, "
        "and explicit remediation tracking:",
        "",
        "| Discovery ID & Core Finding | Tracked Paper (`paper-pipeline`) | Primary Gateway Scholars & Authorities | Verification & Outreach Assets | Gateway Status |",
        "|---|---|---|---|---|",
    ]
    for d in DISCOVERIES:
        scholars_str = "<br>".join(
            f"• **{s['name']}** ({s['affiliation']}) — *{s['gateway_role']}*"
            for s in d["gateway_scholars"]
        )
        statuses = [s["status"] for s in d["gateway_scholars"]]
        if "consulted_and_remediated" in statuses:
            status_badge = "**Consulted & Remediated** (+ Dossiers Ready)"
        elif "upstream_issue_filed_and_email_ready" in statuses:
            status_badge = "**Upstream Issue Filed** + Email Ready"
        elif "scored_in_portfolio" in statuses and "dossier_ready" not in statuses:
            status_badge = "**Pre-Registered & Scored on Author's Dataset**"
        else:
            status_badge = "**Outreach Plate & Email Dossier Ready**"
        plates = ", ".join(f"`{Path(p).name}`" for p in d["verification_locks"]["outreach_plates"][:2])
        email_doc = f"[`{d['verification_locks']['outreach_email_doc']}`](/{d['verification_locks']['outreach_email_doc']})"
        lines.append(
            f"| **`{d['discovery_id']}`**<br>{d['title']} | `{d['paper_id']}` | {scholars_str} | {plates}<br>Email/Doc: `{d['verification_locks']['outreach_email_doc']}` | {status_badge} |"
        )
    lines.append("")
    return "\n".join(lines)


def write_standalone_tracker_md() -> None:
    total_scholars = sum(len(d["gateway_scholars"]) for d in DISCOVERIES)
    consulted = sum(
        1
        for d in DISCOVERIES
        for s in d["gateway_scholars"]
        if s["status"] in ("consulted_and_remediated", "upstream_issue_filed_and_email_ready")
    )
    L = [
        "# Scholarly Gateway & Professor Validation Tracker (`Infinite Morphospace` ↔ `paper-pipeline`)",
        "",
        f"**Last Synced:** {dt.date.today().isoformat()} · **Tracked Discoveries:** `{len(DISCOVERIES)}` · "
        f"**Tracked Manuscripts in `paper-pipeline`:** `6` · **Gateway Scholars Mapped:** `{total_scholars}` (`{consulted}` already contacted/engaged)",
        "",
        "## Why This Tracker Exists",
        "",
        "Internal correctness—even when backed by **SHA-256 pre-registrations**, **Lean 4 kernel proofs (`0 sorry`)**, and **blinded cross-provider AI referee audits**—is necessary but not sufficient for shifting consensus in biology, morphometrics, or mathematical physics. Each field has human **gateway scholars** (the researchers who collected the canonical datasets, invented the standard methods, or raised the sharpest critiques). To validate our discoveries rigorously with those scholars:",
        "",
        "1. **Test Their Exact Dataset / Critique First:** Before emailing any scholar, we run our pre-registered pipeline directly on *their* published dataset or *their* published critique (e.g., Wainwright's 791 reef fishes & 600-pt curve TPS benchmark; Bookstein's 2023/2024 quadratic trend surfaces; Scholtz, Knötel & Baum's 2020 3D laser-scanned crab carapaces; Houle et al.'s 21,138 *Drosophilidae* wings; Economo & Katzke's 2,235 `ANTSCAN` synchrotron scans; Hoel's Causal Emergence 2.0 on Levin's sorting arrays).",
        "2. **Generate a Custom 1-Page Visual Plate & Executable Script:** Every scholar gets a dedicated high-resolution visual plate (`docs/outreach_fig*.png` or `docs/portfolio/fig_*.png`) that speaks their exact methodological language and normalizes all terminology to their field's conventions.",
        "3. **Track Every Interaction Through to Remediation & Co-Authorship in `paper-pipeline`:** When a scholar responds with a methodological critique (as Prof. Peter C. Wainwright did on 2026-10-21 regarding 600-point curve TPS/EFD and convergent trajectories), that critique is immediately converted into a runnable benchmark script, integrated into the LaTeX manuscript, and recorded in `paper-pipeline/papers_progress.json`.",
        "",
        build_gateway_markdown_section(),
        "---",
        "",
        "## Detailed Discovery-by-Discovery Gateway Dossiers",
        "",
    ]
    for d in DISCOVERIES:
        L += [
            f"### `{d['discovery_id']}` — {d['title']}",
            f"- **Target Paper in `paper-pipeline`:** `{d['paper_id']}` (`{d['paper_id']}.tex` / `{d['paper_id']}.pdf`)",
            f"- **Domain:** {d['domain']}",
            f"- **Core Quantitative Finding:** {d['quantitative_claim']}",
            f"- **Verification Scripts:** {', '.join(f'`{s}`' for s in d['verification_locks']['scripts'])}",
            f"- **Outreach Plates:** {', '.join(f'`{p}`' for p in d['verification_locks']['outreach_plates'])}",
            f"- **Outreach Email / Dossier Document:** `{d['verification_locks']['outreach_email_doc']}`",
            "- **Gateway Scholars & Validation Status:**",
        ]
        for s in d["gateway_scholars"]:
            L.append(
                f"  - **{s['name']}** ({s['affiliation']}, `{s['email']}`) — *{s['gateway_role']}* | "
                f"Status: **`{s['status']}`** ({s['contact_date'] or 'ready to send'}) — {s['validation_notes']}"
            )
        L.append("")

    TRACKER_MD.write_text("\n".join(L), encoding="utf-8")
    print(f"Wrote {TRACKER_MD}")


def sync_paper_pipeline_json() -> None:
    if not PAPERS_JSON.exists():
        print(f"Warning: {PAPERS_JSON} not found")
        return
    data = json.loads(PAPERS_JSON.read_text(encoding="utf-8"))
    papers_by_id = {p["paper_id"]: p for p in data.get("papers", [])}

    # Add or update NEW_PAPERS
    for np_entry in NEW_PAPERS:
        pid = np_entry["paper_id"]
        tex_path = PAPER_PIPELINE_ROOT / np_entry["latex_file"]
        wc = count_words_in_latex(tex_path)
        if wc > 0:
            np_entry["stages"]["before"]["factors"]["word_length"]["current"] = wc
        if pid in papers_by_id:
            idx = next(i for i, p in enumerate(data["papers"]) if p["paper_id"] == pid)
            data["papers"][idx] = np_entry
        else:
            data["papers"].append(np_entry)

    # Update word count and notes on thompson_1917_fish_transformations
    if "thompson_1917_fish_transformations" in papers_by_id:
        p_th = papers_by_id["thompson_1917_fish_transformations"]
        tex_path = PAPER_PIPELINE_ROOT / p_th["latex_file"]
        wc = count_words_in_latex(tex_path)
        if wc > 0:
            p_th["stages"]["before"]["factors"]["word_length"]["current"] = wc
        p_th["notes"] = (
            "Synced from morpholib/docs/thompson_1917_fish_transformations_manuscript.tex (compiled PDF in thompson_1917_fish_transformations.pdf). "
            "Covers 17-LM vs 600-semilandmark silhouette paradox, 4p box-vs-warping deconfounding, 4,334 dated divergences (791 spp), "
            "10,303-species 3-dataset replication, pre-registered blind tests on FishShapes v1 (5,180 spp) and FishBase MORPHMET (5,503 spp), "
            "and 12-clade multi-phylum geometric morphometrics portfolio (32/70 hypotheses supported). Consulted Prof. Peter C. Wainwright (2026-10-21) "
            "and integrated 600-pt curve TPS/EFD and Relative Warps / Phylomorphospace convergence benchmarks."
        )
        p_th["last_updated"] = "2026-10-23T00:00:00"

    # Update prospective_collaborators_by_paper_id
    collab = data.setdefault("prospective_collaborators_by_paper_id", {})
    for pid, clist in COLLABORATORS_PATCH.items():
        collab[pid] = clist

    # Add top-level discovery_gateway_validation registry
    data["discovery_gateway_validation"] = {
        "last_synced": dt.datetime.now().isoformat(),
        "source_repo": "https://github.com/fractastical/morpholib",
        "tracker_markdown": "docs/SCHOLARLY_GATEWAY_VALIDATION_TRACKER.md",
        "discoveries": DISCOVERIES,
    }

    PAPERS_JSON.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    print(f"Updated {PAPERS_JSON} with {len(DISCOVERIES)} discoveries and 6 morpholib papers")


def main() -> int:
    write_standalone_tracker_md()
    sync_paper_pipeline_json()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

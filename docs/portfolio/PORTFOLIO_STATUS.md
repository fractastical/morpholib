# Thompson prediction engine: portfolio status

Updated 2026-10-24 by `thompson/portfolio/summarise_portfolio.py` (run unattended by `.github/workflows/thompson-portfolio.yml`). Registered probabilities come from `docs/THOMPSON_DATASET_PORTFOLIO.md` and `thompson/portfolio/scout_preregistration_manifest.json` and are never edited; this page only fills in what was realised.

## 1. Fish datasets (locked predictions, commit `dac08493`)

| Dataset | Status | n matched | BDmax ratio vs consensus (<50 Myr) | vs nearest | 95% coverage | Beyond 50 Myr: ratio / coverage | Claims: registered P -> realised |
|---|---|---|---|---|---|---|---|
| [FishShapes v1 (Price et al. 2022) - scored](./fishshapes_v1.md) | scored 2026-10-22 | 5,180 | 0.41 | 0.96 | 93% | 0.52 / 75% | anchor |
| [FishBase picture-based morphometrics (MORPHMET table)](./fishbase_morphmet.md) | scored 2026-10-22 | 5,503 | 0.41 | 0.97 | 91% | 0.48 / 83% | P1 90%→✓; P2 39%→✓; coverage 90%→✓; P3 60%→✓; P4 60%→✓ |
| [Lake Tanganyika cichlid body-shape landmarks (Ronco et al. 2021)](./ronco2021_tanganyika_cichlids.md) | scored 2026-10-23 | 154 | — | — | — | 0.36 / 100% | P3 60%→✗ |
| [Characiformes body-shape landmarks (Burns & Sidlauskas 2019)](./burns_sidlauskas2019_characiformes.md) | scored 2026-10-23 | 104 | — | — | — | 0.63 / 94% | P3 60%→✗ |

## 2. Fish datasets without simulated anchor (subjective claims)

| Dataset | Registered P | n matched | Status & Realised Outcome |
|---|---|---|---|
| [FishBase BodyShapeI categorical body-shape class](./fishbase_bodyshape_class.md) | 80% | 8,730 | scored 2026-10-23: **✓** |
| [FishBase caudal-fin aspect ratio (SWIMMING table)](./fishbase_caudal_aspect_ratio.md) | 70% | 109 | scored 2026-10-23: **✓** |

## 3. Other groups (re-trained 2D GM engine; subjective priors)

| Group | P(beats consensus) | P(coverage) | P(beats nearest) | Status & Realised 2D GM Benchmark |
|---|---|---|---|---|
| [insects: ants (Formicidae) — *Pheidole (300 spp, 23 LMs) & Camponotus (43 spp, 4 structures)*](./pheidole_ants.md) | 85% | 55% | 35% | scored 2026-10-23: T7 share = **50.4%**, LOO Full/Cons = **0.69** (p=7.8e-28), **2/5** hyps ✓ |
| [insects: butterflies (Papilionoidea) — *Morpho (30 spp, 911 wings, 44/29 LMs)*](./morpho_butterflies.md) | 85% | 55% | 35% | scored 2026-10-23: T7 share = **49.0%**, LOO Full/Cons = **0.58** (p=4.7e-08), **4/4** hyps ✓ |
| [insects: Drosophilidae wings — *Houle et al. 2017 Nature (115 spp, 21,138 wings, 12 LMs)*](./drosophilidae_wings.md) | 75% | 45% | 30% | scored 2026-10-23: T7 share = **71.3%**, LOO Full/Cons = **0.61** (p=2.8e-12), **4/5** hyps ✓ |
| [birds — *Olsen 2017 Anseriformes beaks (51 spp, 100 LMs)*](./waterfowl_beaks.md) | 95% | 60% | 40% | scored 2026-10-23: T7 share = **91.1%**, LOO Full/Cons = **1.01** (p=0.28), **2/4** hyps ✓ |
| [mammals — *Sherratt & Kraatz 2023 Leporidae crania (22 spp, 317 skulls, 74 LMs)*](./leporidae_skulls.md) | 90% | 55% | 35% | scored 2026-10-23: T7 share = **51.1%**, LOO Full/Cons = **0.89** (p=0.28), **0/3** hyps ✓ |
| [squamate reptiles — *Brennan et al. 2021 Varanus monitor lizards (80 spp, 1,091 skulls, 33/10 LMs)*](./varanus_lizards.md) | 85% | 50% | 35% | scored 2026-10-23: T7 share = **59.5%**, LOO Full/Cons = **1.00** (p=1), **1/4** hyps ✓ |
| [amphibians — *Sherratt et al. 2017 Australian Anura frogs (166 spp, 55 LMs)*](./australian_frogs.md) | 80% | 50% | 30% | scored 2026-10-23: T7 share = **66.2%**, LOO Full/Cons = **0.80** (p=1.5e-14), **2/4** hyps ✓ |
| [seed plants — *Joly et al. 2018 Gesneriaceae corollas (48 spp, 296 flowers, 32 LMs)*](./antillean_flowers.md) | 70% | 45% | 30% | scored 2026-10-23: T7 share = **82.1%**, LOO Full/Cons = **0.69** (p=0.0055), **3/4** hyps ✓ |

## 4. Complete Multi-Clade 2D Geometric Morphometrics Atlas (`thompson/portfolio/groups/`)

| Clade / Study | Phylum / Class | Taxonomic & Landmark Scope | Thompson-7 Variance Share | Fitted 7-D PCA Share | Pagel's $\lambda$ (T7) | LOO Error Ratio vs. Consensus (Full / T7) | Pre-Registered Hypotheses Supported |
|---|---|---|---|---|---|---|---|
| **Indo-Pacific Reef Fishes (Baseline)** | Chordata: Actinopterygii | 743 tree spp (791 total), 17 LMs | **57.9%** | 86.2% | 0.959 | **0.36** (<50 Myr) / **0.51** | Baseline engine |
| [Anseriformes waterfowl beaks (Olsen 2017)](./waterfowl_beaks.md) | Chordata: Aves | 51 spec., 51 spp (51 tree), 100 LMs | **91.1%** | 98.2% | 1.000 (Brownian motion) | **1.01** / **0.87** | **2/4** (Brier 0.411) |
| [Galeommatoidea marine bivalve shells (Goto et al. 2016)](./galeommatoidea_bivalves.md) | Mollusca: Bivalvia | 302 spec., 202 spp (164 tree), 46 LMs | **85.0%** | 98.8% | 0.746 (Pagel lambda=0.746) | **0.65** / **0.74** | **3/4** (Brier 0.219) |
| [Antillean Gesneriaceae floral corollas (Joly et al. 2018)](./antillean_flowers.md) | Angiospermae: Gesneriaceae | 296 spec., 51 spp (42 tree), 32 LMs | **82.1%** | 96.8% | 0.643 (White noise (star)) | **0.69** / **1.00** | **3/4** (Brier 0.179) |
| [Camponotus carpenter ants (4 structures, 634 specimens)](./camponotus_ants.md) | Arthropoda: Insecta | 634 specimens, 43 spp, 82/61/52/46 LMs | **81.2%** head / **92.9%** pronotum | 99.7% | 0.890 (BM) | **0.56** / **0.65** (head) | 4/8 (Brier 0.369) |
| [Drosophilidae wings (Houle et al. 2017 Nature, 21,138 wings)](./drosophilidae_wings.md) | Arthropoda: Insecta | 21,138 spec., 115 spp (83 tree), 12 LMs | **71.3%** | 96.5% | 0.602 (Pagel lambda=0.602) | **0.61** / **0.76** | **4/5** (Brier 0.190) |
| [Australian Anura frogs (Sherratt et al. 2017, 166 spp, 55 LMs)](./australian_frogs.md) | Chordata: Amphibia | 166 spec., 166 spp (166 tree), 55 LMs | **66.2%** | 88.9% | 0.682 (Pagel lambda=0.682) | **0.80** / **0.90** | **2/4** (Brier 0.361) |
| [Varanus monitor lizards (Brennan et al. 2021, 1,091 crania, 33/10 LMs)](./varanus_lizards.md) | Chordata: Squamata | 1,090 spec., 80 spp (73 tree), 33 LMs | **59.5%** | 85.6% | 0.469 (Pagel lambda=0.469) | **1.00** / **0.89** | **1/4** (Brier 0.507) |
| [Characiformes radiation (Burns & Sidlauskas 2019, 24 LMs)](./characiformes_burns_gm.md) | Chordata: Actinopterygii | 1,625 spec., 263 spp (72 tree), 24 LMs | **55.5%** | 92.3% | 0.915 (Pagel lambda=0.915) | **0.62** / **0.80** | **3/4** (Brier 0.151) |
| [Leporidae mammalian crania (Sherratt & Kraatz 2023, 74 LMs)](./leporidae_skulls.md) | Chordata: Mammalia | 317 spec., 22 spp (18 tree), 74 LMs | **51.1%** | 90.9% | 1.000 (Brownian motion) | **0.89** / **0.96** | **0/3** (Brier 0.588) |
| [Pheidole big-headed ants (Friedman et al. 2020, minor & major heads)](./pheidole_ants.md) | Arthropoda: Insecta | 653 spec., 300 spp (246 tree), 23 LMs | **50.4%** | 94.6% | 0.924 (Pagel lambda=0.924) | **0.69** / **0.86** | **2/5** (Brier 0.433) |
| [Morpho butterflies (Chazot et al. 2021, forewing & hindwing)](./morpho_butterflies.md) | Arthropoda: Insecta | 911 spec., 30 spp (29 tree), 44 LMs | **49.0%** | 95.7% | 0.877 (Brownian motion) | **0.58** / **0.84** | **4/4** (Brier 0.037) |
| [Lake Tanganyika cichlids (Ronco et al. 2021 Nature, body & jaw)](./cichlids_ronco_gm.md) | Chordata: Actinopterygii | 2,197 spec., 242 spp (235 tree), 21 LMs | **47.2%** | 88.3% | 0.990 (OU (alpha=0.0296/Myr, t1/2=23 Myr)) | **0.42** / **0.77** | **1/4** (Brier 0.544) |
| [Anthophila bee forewings (10 families + Eocene fossils)](./bee_wings.md) | Arthropoda: Insecta | 615 wings, 42 tribes, 18 LMs | **42.2%** | 93.7% | 0.659 (OU) | **0.85** / 0.94 | 4/9 (Brier 0.242) |
| [Vespinae hornet forewings (Perrard et al. 2014)](./vespinae_wings.md) | Arthropoda: Insecta | 1,027 wings, 30 spp, 19 LMs | **38.8%** | 95.5% | 0.533 (WN) | 1.00 / 0.99 | 0/7 (Brier 0.448) |
| [Causal decomposition of wing divergence (P-drift vs. allometry)](./why_wings.md) | Arthropoda: Insecta | Hornet + Bee forewings (P-matrix drift & allometry) | — | — | — | — | 5/13 (Brier 0.254) |
| [Vespinae venation-truss mechanics & resolved Perrard (2013) tree (D1-D8)](./vespinae_venation_mechanics.md) | Arthropoda: Insecta | 1,027 wings, 30 spp + 373 bee wings (resolved tree + truss/dipole) | **65.8%** perim / **64.2%** dipole-7 | 95.5% | **0.946** (BM) | **0.63** / **0.79** | **7/8** (Brier 0.131) |

## 5. Calibration of the registered probabilities & Published Nanopublications

- **Fish portfolio claims resolved:** 9 claims (7/9 passed; Brier score = **0.174**, where 0 = perfect and 0.25 = coin flip).
- **Multi-clade 2D Geometric Morphometrics hypotheses resolved:** **45/90** pre-registered hypotheses supported across 15 clade studies spanning **Mollusca, Actinopterygii, Amphibia, Squamata, Aves, Mammalia, Angiospermae, and Insecta (Diptera, Lepidoptera, Hymenoptera)**.
- **Signed Production Nanopublications (`nanopubs/multiclade_portfolio_to_nanopubs.py`):**
  - **Multi-Clade Preregistration Manifest:** [`RAQ_Giwd735IPhYg-cOMyPE6t4K9WPM9XlijGkNC6_eJ4`](https://registry.knowledgepixels.com/np/RAQ_Giwd735IPhYg-cOMyPE6t4K9WPM9XlijGkNC6_eJ4)
  - **Master Cross-Kingdom Synthesis Nanopub (linking all 5 fish + 15 clade outcome nanopubs):** [`RAOvI_PZUXLZtKjyVfq4ILVoIxqVCsTXmLXd0J7ZujfEo`](https://registry.knowledgepixels.com/np/RAOvI_PZUXLZtKjyVfq4ILVoIxqVCsTXmLXd0J7ZujfEo)

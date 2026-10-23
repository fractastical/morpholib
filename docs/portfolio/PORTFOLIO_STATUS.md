# Thompson prediction engine: portfolio status

Updated 2026-10-23 by `thompson/portfolio/summarise_portfolio.py` (run unattended by `.github/workflows/thompson-portfolio.yml`). Registered probabilities come from `docs/THOMPSON_DATASET_PORTFOLIO.md` and are never edited; this page only fills in what was realised.

## Fish datasets (locked predictions, commit dac08493)

| Dataset | Status | n matched | BDmax ratio vs consensus (<50 Myr) | vs nearest | 95% coverage | Beyond 50 Myr: ratio / coverage | Claims: registered P -> realised |
|---|---|---|---|---|---|---|---|
| [FishShapes v1 (Price et al. 2022) - scored](./fishshapes_v1.md) | scored 2026-10-23 | 5,180 | 0.41 | 0.96 | 93% | 0.52 / 75% | anchor |
| [FishBase picture-based morphometrics (MORPHMET table)](./fishbase_morphmet.md) | scored 2026-10-23 | 5,503 | 0.41 | 0.97 | 91% | 0.48 / 83% | P1 90%→✓; P2 39%→✓; coverage 90%→✓; P3 60%→✓; P4 60%→✓ |
| Lake Tanganyika cichlid body-shape landmarks (Ronco et al. 2021) | not downloaded | - | - | - | - | - | pending |
| Characiformes body-shape landmarks (Burns & Sidlauskas 2019) | not downloaded | - | - | - | - | - | pending |

## Fish datasets without simulated anchor (subjective claims)

| Dataset | Registered P | Status |
|---|---|---|
| FishBase BodyShapeI categorical body-shape class | 80% | pending |
| FishBase caudal-fin aspect ratio (SWIMMING table) | 70% | pending |

## Other groups (re-trained engine; subjective priors)

| Group | P(beats consensus) | P(coverage) | P(beats nearest) | Status |
|---|---|---|---|---|
| insects: ants (Formicidae) | 85% | 55% | 35% | pending |
| insects: butterflies (Papilionoidea) | 85% | 55% | 35% | pending |
| insects: Drosophilidae wings | 75% | 45% | 30% | pending |
| birds | 95% | 60% | 40% | pending |
| mammals | 90% | 55% | 35% | pending |
| squamate reptiles | 85% | 50% | 35% | pending |
| amphibians | 80% | 50% | 30% | pending |
| seed plants | 70% | 45% | 30% | pending |

## Calibration of the registered probabilities

Brier score over 5 resolved claims: **0.143** (0 = perfect, 0.25 = uninformative coin). Each resolved claim scores (P − outcome)².

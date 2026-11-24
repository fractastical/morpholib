# T1917-025 — Spicules / Geodetics (Thompson 1917, Ch. IX–X)

**Claim:** Radiolarian and similar siliceous skeletons approximate geodesic/minimal frameworks on a sphere (latticed shells as systems of shortest lines and near-equal partitions).

**Loop:** `thompson1917-t1917-025` (MorphoEval slot 220)

## Dataset verification (Step 1)

The pre-scouted path `datasets/acanthometra_pellucida_flux_canalization/acanthometra_pellucida_specimen.jpg` is **not present** in this repository snapshot and targets a different measurement programme (flux canalization). Step 1 therefore vetted Commons radiolarian lattice plates that match Thompson’s Haeckel/Challenger discussion:

| File | Source | License |
|------|--------|---------|
| `radiolaria_challenger_plate111.jpg` | [Commons: Radiolaria (Challenger) Plate 111](https://commons.wikimedia.org/wiki/File:Radiolaria_(Challenger)_Plate_111.jpg) | Public domain |
| `haeckel_radiolaria_plate_1888.jpg` | [Commons: Haeckel Radiolaria monograph plate](https://commons.wikimedia.org/wiki/File:Die_Radiolarien_(Rhizopoda_radiaria)_-_eine_Monographie_(1888)_(20932349692).jpg) | No restrictions |

SHA-256 digests are pinned in `t1917_025_spicules_geodetics_reference.json`.

## Model (script)

`thompson/claims/t1917-025_spicules-geodetics.py` — icosahedral geodesic reference frameworks on the unit sphere, edge-length partition statistics, skeleton extraction from the Challenger plate crop, goodness-of-fit vs geodesic and ablation controls (Euclidean grid, random chords).

## Formal proofs

`morphoeval/lean/loops/Thompson1917_T1917_025.lean` — integer-scaled geodesic length and equal-partition lemmas (`MathOfBiology.Loop_220`).

## Verdict artefact

`result.json` — written by the claim script after Pass 2 remediation.

# *Echinarachnius parma* — 60 µm Turing wavelength (slot 3214)

Holocene (extant), **0 Ma**. Detail scale **60 µm** (6×10⁻⁵ m, log10 ≈ −4.222), tier `4_cellular_10um`.

The chronoseries in this slot has one member. There is no older *Echinarachnius* chronospecies to difference against. What changes at this scale, relative to its scale neighbours at the same age, is whether a Turing mode fits in the patch:

| Patch | Age | Wavelength count `floor(L / 60 µm)` | Pattern |
| --- | --- | --- | --- |
| 10 µm predecessor tier | 0 Ma | 0 | quenched; the unstable wavenumber does not fit |
| **480 µm field of view (this slot)** | **0 Ma** | **8** | **60 µm pigment-ocellus lattice** |
| 600 µm successor tier | 0 Ma | 10 | same local wavelength, more repeats |

The centimetre-scale petaloid on the test is a different structure (Seilacher 1979, constructional morphology). It is not this wavelength. Sibling slot 160’s dimensionless cartoon (`α = 1.71`) is not reused.

## Law

Schnakenberg (1979) kinetics with the Turing (1952) dispersion relation. `λ = 2π / k_max`, where `k_max` maximises the larger eigenvalue of `J − k²D`. Diffusivities are calibrated so the equilibrated lattice sits on 60 µm. They are not measured tissue diffusivities.

## Specimen

Primary plate: **USNM E39365**, Smithsonian National Museum of Natural History, CC0, via Wikimedia Commons.

`echinarachnius_parma_usnm_e39365_specimen.jpeg`

SHA-256 `1af65dc69de55d4b0ff00c1f4134565eed1f5e151e1539fbbdc52730ed880787`

Auxiliary plates (not the fitness target):

- `echinarachnius_parma_ccby.jpg` — MNHN, Wikimedia license string CC BY 4.0, denuded test with a 50 mm bar
- `echinarachnius_parma_0zz.jpg` — Wikimedia CC BY-SA 4.0, weathered aboral test

The project gallery file `datasets/species_gallery/II_Anaima_Invertebrata/Echinarachnius_parma.jpg` is a synthetic swirl with a 100 µm bar. It is not the specimen used here.

## Reproduce

```bash
python3 models/2026-delphi10000-echinarachnius-parma-turing-wavelength-s0.py --self-test
python3 models/2026-delphi10000-echinarachnius-parma-turing-wavelength-s0.py
```

Seed 3214. Outputs:

- `docs/media/echinarachnius_parma_turing_wavelength_s0_comparison.png`
- `docs/media/echinarachnius_parma_turing_wavelength_s0_growth.gif`

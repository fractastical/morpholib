# *Drosophila buzzatii* — Turing wavelength, detail scale (slot 3179)

Holocene (extant), **0 Ma**. Phenomenon scale **2.9 µm** (**2.908 × 10⁻⁶ m**, log10 = −5.536), tier `3_subcellular_um`.

This slot is the detail-scale variant (`s0`) of Turing wavenumber selection for the prickly-pear cactophilic fly. It is not the 102 µm primordium slot (`norry-drosophila-buzzatii-turing-wavelength`, slot 2991) and it does not reuse that slot's power-law coefficients.

## What this scale selects

Schnakenberg kinetics (Schnakenberg 1979) on a periodic cuticle patch, with the Turing critical wavenumber (Turing 1952)

`k_c² = γ (D_v f_u + D_u g_v) / (2 D_u D_v)`, `λ_c = 2π / k_c`.

Parameters are fixed so `λ_c = 2.908 µm`. Activator diffusivity is vesicle-limited (`D_u = 0.040 µm²/time`) and the inhibitor is faster by exactly `D_v/D_u = 29`. The pattern that emerges is a field of micro-ocelli, not organ-scale wing patches.

## What changes relative to the chronoseries neighbours

Both neighbours are Holocene (0 Ma). The change at this tier is the selected wavelength, not the geological age.

| Chronospecies | Role | λ | Modes on a 1.70 mm wing chord |
|---|---|---|---|
| *D. bipectinata* | Predecessor (slot 2851) | 85.92 µm | 19 |
| *D. buzzatii* | This slot (3179) | 2.908 µm | 584 |
| *D. cyrtoloma* | Successor (slot 3061) | 207.6 µm | 8 |

*D. buzzatii* therefore packs about 30× more pigment modes across the wing chord than *D. bipectinata* and about 70× more than the picture-wing successor *D. cyrtoloma*. Norry, Vilardi & Hasson (1997) showed that wing length and thorax length are correlated size traits whose correlations are rearranged by inversions. At 2.9 µm the reaction–diffusion wavenumber does not track that allometry: organ length changes the **mode count** `N = L/λ_c`, not `λ_c` itself.

## Specimen image

Wikimedia Commons, iNaturalist, and GBIF returned no CC-licensed bitmap of a *D. buzzatii* adult. The pinned image is the repository species-card fallback

`datasets/species_gallery/II_Anaima_Invertebrata/Drosophila_buzzatii.jpg`

copied to `drosophila_buzzatii_turing_wavelength_s0_specimen.jpg`. It is a radial texture card with a 100 µm scale bar, not a wing micrograph. The simulation paints the predicted micro-ocellus density onto that card; the pure Schnakenberg field is reported separately and is not claimed to be a photograph.

## Run

```bash
python3 models/2026-delphi10000-drosophila-buzzatii-turing-wavelength-s0.py --self-test
python3 models/2026-delphi10000-drosophila-buzzatii-turing-wavelength-s0.py
```

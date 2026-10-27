# Ammonia beccarii — 31 µm Turing wavelength (Holocene, 0 Ma)

Slot 3108 of the 10,000-Animal Basic Look. One animal, one scale, one point in evolutionary history.

- **Animal:** *Ammonia beccarii* (Linnaeus, 1758), benthic rotaliid foraminifer.
- **Scale:** 31 µm detail variant of tier `4_cellular_10um` (phenomenon size `3.137e-05` m, log10 = −4.503). This is the critical Turing wavelength λ_c = 2π / k_c, not the whole-test diameter and not the pore diameter.
- **Age:** Holocene (Extant), **0 Ma**. Ammonia chronoseries **1/1**. There is no predecessor chronospecies and no successor chronospecies in this series.

The cohort law is Turing wavenumber selection (chemical reaction–diffusion). On this calcite test the “spots & ocelli” class is read as a pustule / pore-cluster spacing, not as a vertebrate pigment stripe. *Ammonia* does not have ocelli.

## What changes at this scale

| Mode | Length | Wavenumber | In the unstable band? |
| --- | --- | --- | --- |
| Pore diameter (Subba & Ghosh 2023, *A.* cf. *beccarii*) | 0.564 µm | ~11.1 µm⁻¹ | no — far too fine |
| **This slot, λ_c** | **31.37 µm (labeled 31 µm)** | **0.200 µm⁻¹** | **yes — the selected mode** |
| Hillewaert test diameter | ~600 µm | ~0.010 µm⁻¹ | no — one test-scale patch is not selected |

Moving from the pore mode to this detail mode lengthens the pattern about 56-fold and drops the wavenumber into the narrow Schnakenberg band (about 27–39 µm at the parameters in the reference JSON). The whole-test mode stays below the band. Because the chronoseries is a single Holocene terminal, that wavenumber shift is a within-test scale change at 0 Ma, not a transition from another chronospecies.

## Specimen

`Ammonia_beccarii_Hillewaert_2012.jpg` — Hans Hillewaert, CC BY-SA 4.0, Wikimedia Commons `File:Ammonia_beccarii.jpg`. Six tests, 500 µm scale bar, caption diameter ~600 µm. The species-gallery file `datasets/species_gallery/IV_Protista_Monera_Mycota/Ammonia_beccarii.jpg` shows a *Caryophyllia* coral and is not used.

North Sea specimens labeled *A. beccarii* are often not the Adriatic type species (Hayward et al. 2004). Schönfeld et al. 2021 measured living Adriatic *Ammonia* means of 131–199 µm; true *A. beccarii* in that study was a larger, heavily ornamented dead-assemblage form. The 600 µm caption size sits with the large beccarii morphogroup, not with those living means. The model does not treat the photograph as a sequenced type specimen.

## Run

```bash
python3 models/2026-delphi10000-ammonia-beccarii-turing-wavelength-s0.py --self-test
python3 models/2026-delphi10000-ammonia-beccarii-turing-wavelength-s0.py
```

Outputs:

- `docs/media/ammonia_beccarii_turing_wavelength_s0_comparison.png`
- `docs/media/ammonia_beccarii_turing_wavelength_s0_growth.gif`
- `datasets/ammonia_beccarii_turing_wavelength_s0/ammonia_beccarii_turing_wavelength_s0_metrics.json`

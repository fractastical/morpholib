# *Drosophila grimshawi* — Turing wavelength at 2.8 µm (slot 3186)

Detail-scale reproduction of **Turing reaction-diffusion wavelength selection** for the Hawaiian picture-winged fly *Drosophila grimshawi*. This slot is the subcellular detail variant, not the 199 µm primordium slot (`datasets/drosophila_grimshawi_turing_wavelength/`).

- **Scale:** 2.8 µm (`2.765e-06 m`, `log10 = -5.558`, tier `3_subcellular_um`, detail variant). The calibrated fastest-growing Schnakenberg mode is held at **2.765 µm**.
- **Age:** Holocene (Extant), **0 Ma**.
- **Lineage:** Hawaiian Picture-Winged Drosophila Adaptive Radiation. Drosophila chronoseries 15/41. Predecessor *Drosophila ficusphila*. Successor *Drosophila guttifera*.
- **Form:** stripes, spots, and ocelli (picture-wing lacework / retinal pigment wave).
- **Law:** Turing wavenumber `k_c` from the Schnakenberg dispersion relation `σ(k)`.

## What changes at this scale

Organ-scale wing spots of the predecessor (*D. ficusphila*, sex-comb and wing-spot sophophoran) and the successor (*D. guttifera*, Wingless-induced polka dots; Werner et al. 2010) are ~100 µm patterns and are not re-fit here. At the **2.8 µm** detail tier the same Schnakenberg kinetics are kept and only the inhibitor/activator diffusivity ratio `δ = D_v / D_u` changes:

| chronospecies | role | δ | fastest-mode λ |
|---|---|---:|---:|
| *D. ficusphila* | predecessor | 10.40 | 2.512 µm |
| *D. guttifera* | successor | 12.85 | 2.629 µm |
| *D. grimshawi* | this slot | 16.3186 | 2.766 µm |

Raising `δ` lengthens the fastest-growing wavelength. *D. grimshawi* therefore selects the longest detail-scale pigment wavelength of the three: the picture-wing lacework sits at the assigned 2.8 µm class, longer than the predecessor’s tighter wave and longer than the successor’s partial return toward a shorter subcellular spacing. All three species are extant (**0 Ma**); the difference is morphospace position, not geological age.

## Specimen

`drosophila_grimshawi_turing_wavelength_s0_specimen.jpg` is the project gallery asset for this species (SHA-256 `4ea3740942a29423fc5f6c08fbbd634d3c65eb3440bf568b3bd693b081340a1e`), copied from `datasets/species_gallery/II_Anaima_Invertebrata/Drosophila_grimshawi.jpg`. Wikimedia Commons searches on 2026-10-27 returned Carlquist envelope labels (CC BY 4.0) and no anatomical wing or retina micrograph, so those files were not used as the morphometric target.

The gallery frame is a brown spiral pigment field with a caption bar. It is the canonical image the fitness test scores against. It is not a calibrated 2.8 µm electron micrograph; the physical wavelength is the dispersion calibration, and the image comparison is texture and radial-amplitude agreement with this asset.

## Run

```bash
python3 models/2026-delphi10000-drosophila-grimshawi-turing-wavelength-s0.py --self-test
python3 models/2026-delphi10000-drosophila-grimshawi-turing-wavelength-s0.py
```

Outputs:

- `docs/media/drosophila_grimshawi_turing_wavelength_s0_comparison.png`
- `docs/media/drosophila_grimshawi_turing_wavelength_s0_growth.gif`

## Sources

Edwards, Doescher, Kaneshiro & Yamamoto (2007) PLoS ONE 2:e487, doi:10.1371/journal.pone.0000487. Clark et al. (2007) Nature 450:203–218, doi:10.1038/nature06341. Turing (1952) Phil. Trans. R. Soc. B 237:37–72. Schnakenberg (1979) J. Theor. Biol. 81:389–400. Werner, Koshikawa, Williams & Carroll (2010) Nature 464:1143–1148, doi:10.1038/nature08896. Koshikawa (2020) Dev. Growth Differ. 62:269–278. Tanaka et al. (2011) PLoS Biol. 9:e1001131.

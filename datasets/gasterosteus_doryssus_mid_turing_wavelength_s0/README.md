# *Gasterosteus doryssus_mid* — 42 µm Turing wavelength (slot 3230)

Detail-scale reproduction of **Turing reaction–diffusion wavelength selection** on the flank of the Miocene stickleback chronospecies *Gasterosteus doryssus_mid*.

- **Scale:** 42 µm (4.194e-05 m, log10 = −4.377), tier `4_cellular_10um`, detail variant. This is the chromatophore / melanophore spacing, not the 1.7 mm organ-scale stripe of slot 1208.
- **Age:** Late Miocene, Truckee Formation, **10.01 Ma**. Chronoseries position 2/6 of the Gasterosteus pelvic-reduction lineage: vestigial pelvic spine at **T = +9 kyr**. Predecessor *G. doryssus_early* (full girdle, T = 0, 10.02 Ma). Successor *G. doryssus_late* (pelvic loss, T = +22 kyr, 9.99 Ma).
- **Law:** critical Turing wavenumber \(k_c = \arg\max_k \sigma(k)\) of a Schnakenberg activator–inhibitor pair (Turing 1952; pigment-wave phenomenology of Kondo & Asai 1995).
- **Form:** stripes, spots, and ocelli.

## What changes at this scale and age

Armor and pelvic completeness change along the Truckee chronoseries (Bell & Haglund 1982; Hunt, Bell & Travis 2008). Extant pelvic reduction is a *Pitx1* enhancer deletion (Shapiro et al. 2004; Chan et al. 2010) and co-occurs with shifts in nuptial melanism. This slot **transfers** that association to the 42 µm pigment wavelength only:

| Stage | Age | Pelvic index | \(D_v/D_u\) | \(\gamma\) | Predicted \(\lambda_c\) |
| --- | --- | --- | --- | --- | --- |
| *doryssus_early* | 10.02 Ma, T = 0 | 0.96 (full girdle) | 14.2 | 1.55 | 32.14 µm |
| *doryssus_mid* (this slot) | **10.01 Ma, T = +9 kyr** | 0.42 (vestigial spine) | **18.0** | **1.00** | **41.94 µm** |
| *doryssus_late* | 9.99 Ma, T = +22 kyr | 0.06 (pelvic loss) | 24.8 | 0.62 | 56.90 µm |

\(\lambda\) lengthens as pelvic score falls because the reaction rate \(\gamma\) drops faster than the diffusivity ratio rises. The mid-stage diffusivity \(D_u\) is bisected so the exact fastest-growing mode sits on the assigned 42 µm. Early and late wavelengths are predictions of that same law, not copies of the organ-scale sibling.

The Truckee fossils preserve bone, not chromatophores. The 42 µm figure is a scale assignment plus a transferred melanophore-spacing hypothesis. It is not a direct histological measurement.

## Specimen

`gasterosteus_doryssus_mid_turing_wavelength_s0_specimen.jpg` is Wikimedia Commons `File:Gasterosteus_doryssus_morphs.jpg` (CC BY 4.0), the Voje, Bell & Stuart (2022) plate of weakly armored lineage I and strongly armored lineage II. The centimetre scale bar on that plate cannot resolve 42 µm; it is the real anatomical specimen against which the rendered field is compared.

## Run

```bash
python3 models/2026-delphi10000-gasterosteus-doryssus-mid-turing-wavelength-s0.py --self-test
python3 models/2026-delphi10000-gasterosteus-doryssus-mid-turing-wavelength-s0.py
```

Outputs:

- `docs/media/gasterosteus_doryssus_mid_turing_wavelength_s0_comparison.png`
- `docs/media/gasterosteus_doryssus_mid_turing_wavelength_s0_growth.gif`

Formalization: `morphoeval/lean/loops/Loop_3230_gasterosteus_doryssus_mid_turing_wavelength_s0.lean` (`MathOfBiology.Loop_3230`).

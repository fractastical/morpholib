# *Clytia hemisphaerica* — Turing wavelength, detail scale (slot 3150)

Holocene extant, **0 Ma**. Clytia chronoseries **1/1**: this is the only chronospecies in the series, so there is no earlier age to difference against. The change formalised here is the **spatial** one.

| Quantity | Value |
| --- | --- |
| Assigned scale | 8.6 µm (8.64×10⁻⁶ m, log10 = −5.063) |
| Tier | 3/8 `3_subcellular_um`, detail variant `s0` |
| Age | 0 Ma (Holocene) |
| Law | Turing wavenumber k\* = argmax Re σ(k) |
| Kinetics | Schnakenberg (1979), not the slot-0168 dimensionless template |

Slot 0168 (`clytia_hemisphaerica_turing_wavelength`) stores a unitless triple (α = 1.63, β = 0.25, integers 9, 4, 36) and a placeholder DOI. This slot does not copy those numbers. It pins

- a = 0.10, b = 0.90
- Dᵤ = 1 µm² s⁻¹, Dᵥ/Dᵤ = 20
- γ ≈ 2.227 s⁻¹ so that λ\* = 2π/k\* = 8.64 µm

On a reconstructed 250 µm day-2 planula that wavelength repeats about 28 times. The same law on a 2 mm medusa bell would repeat about 231 times. The detail look is the short domain, not a new species and not a new age.

## Specimen

`clytia_hemisphaerica_specimen.jpg` is Tsuyoshi Momose’s DIC micrograph of a 2-day *Clytia hemisphaerica* planula (Wikimedia Commons, file `Clytia hemisphaerica day2 planula.jpg`, CC BY-SA 4.0, 22 April 2021). The frame has no scale bar, so lengths in micrometres other than the assigned λ\* are reconstructions.

A polyp-colony frame by the same author is recorded in `scout_manifest.json` and was not vendored (6 MB). The project atlas icon `datasets/species_gallery/II_Anaima_Invertebrata/Clytia_hemisphaerica.jpg` is a synthetic spiral plate with a 100 µm bar; it is the wrong image for this segmentation.

## What is and is not claimed

The script realises the Turing dispersion relation at the assigned 8.64 µm detail length and paints the resulting grain inside the segmented planula. It does **not** claim that a published measurement found an 8.64 µm ocellus pigment stripe, and it does **not** claim a wavelength shift relative to a fossil *Clytia*. Ocellus ultrastructure in hydrozoans is cited from Singla (1974) only as anatomical context for why a subcellular pigment wavelength is a meaningful detail scale on this animal.

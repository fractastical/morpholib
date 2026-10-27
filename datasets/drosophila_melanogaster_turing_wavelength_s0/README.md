# Drosophila melanogaster — Turing wavelength, detail scale s0

Slot 3192 of the 10,000-animal basic look. One animal, one scale, one age.

* **Animal.** *Drosophila melanogaster* (fruit fly). Aristotelian place: II. Anaima (Invertebrata) → Entoma (Arthropoda) → *Drosophila*.
* **Scale.** 2.7 µm, exactly **2.736e-06 m** (log10 = **−5.563**), tier `3_subcellular_um`, detail variant `s0`. This is the critical Turing wavelength of a subcellular melanin / ocellar-pigment reticulum, not the adult body (~2.5 mm), not an ommatidium (~15–20 µm), and not a published electron-micrograph spacing.
* **Age.** Holocene (Extant), **0 Ma**. Drosophila chronoseries 21/41. Predecessor *Drosophila mauritiana*. Successor *Drosophila miranda*. All three are extant, so the difference at this slot is phylogenetic parameter change at a shared age, not a stratigraphic one.

## Law

Schnakenberg (1979) kinetics as a concrete Turing (1952) system:

∂u/∂t = γ (a − u + u²v) + D_u ∇²u

∂v/∂t = γ (b − u²v) + D_v ∇²v

λ_c = 2π / k_c with k_c = argmax σ(k). γ is solved once, on the melanogaster parameters, so that λ_c = 2.736 µm. Mauritiana and miranda keep that γ and use their own (a, b, D_u, D_v).

| species | role | a | b | D_u | D_v | λ_c |
|---|---|---:|---:|---:|---:|---:|
| *D. miranda* | successor | 0.07 | 1.25 | 0.020 | 2.20 | 2.242 µm |
| *D. melanogaster* | focal, 0 Ma | 0.11 | 0.97 | 0.035 | 1.62 | 2.737 µm |
| *D. mauritiana* | predecessor | 0.16 | 0.72 | 0.055 | 1.15 | 3.243 µm |

The longer mauritiana wave is the reconstruction of reduced abdominal melanin relative to melanogaster males (True et al. 1997; Yellow/Ebony in Wittkopp, True & Carroll 2002). The shorter miranda wave is the reconstruction of a denser obscura-group melanic reticulum. Richards et al. 2005 sequenced *D. pseudoobscura*, the clade that includes *D. miranda*; that paper does not measure this wavelength.

## Specimen

`Drosophila_melanogaster_specimen.jpg` — adult male, André Karwath, Wikimedia Commons `File:Drosophila melanogaster - side (aka).jpg`, CC BY-SA 2.5.

SHA-256: `2a66a100d8405cc95d2c17474a8a55ade755e5580a5d47f6863dc22342cdf05b`

## Run

```bash
python3 models/2026-delphi10000-drosophila-melanogaster-turing-wavelength-s0.py --self-test
python3 models/2026-delphi10000-drosophila-melanogaster-turing-wavelength-s0.py
```

Seed 3192. Outputs: `docs/media/drosophila_melanogaster_turing_wavelength_s0_comparison.png` and `docs/media/drosophila_melanogaster_turing_wavelength_s0_growth.gif`.

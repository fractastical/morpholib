# Drosophila silvestris — Turing wavelength at 3.0 µm (0 Ma)

Slot 3204 of the 10,000-Animal Basic Look. One animal, one scale, one point in evolutionary history.

- **Animal.** *Drosophila silvestris* (Perkins), Hawaiian Big-Island picture-wing fly. Aristotelian place: II. Anaima (Invertebrata) → Entoma (Arthropoda) → *Drosophila*.
- **Scale.** 3.0 µm detail tier (`3_subcellular_um`), stored as **2.995×10⁻⁶ m** (log10 = −5.524). This is the critical Turing wavelength λ_c of a Schnakenberg pigment-ocellus field, not the body length (~5 mm) and not the macroscopic wing-spot diameter visible in the voucher photograph.
- **Age.** Holocene (extant), **0 Ma**. Lineage: Hawaiian Picture-Winged *Drosophila* adaptive radiation. Drosophila chronoseries position 33/41. Predecessor in that series: *Drosophila serrata* (cellular stripe slot, 92.26 µm). Successor: *Drosophila simulans* (cellular/clasper slot, 82.56 µm).

## What changes at this scale

All three chronospecies are extant, so the contrast is phylogenetic and spatial, not a geological time step.

| Node | λ | k_c relative to this slot | Pattern regime used here |
|---|---:|---:|---|
| *D. serrata* (predecessor) | 92.26 µm | k_this / k_pred ≈ 30.81 | longer-wave stripe reference, diffusion ratio 8 |
| *D. silvestris* (this slot) | 2.995 µm | 1 | ocellus/spot regime, diffusion ratio 20 |
| *D. simulans* (successor) | 82.56 µm | k_this / k_succ ≈ 27.57 | longer-wave stripe reference, diffusion ratio 10 |

The detail-tier wavenumber is about thirty times the predecessor’s cellular wavenumber. The successor does not keep that fine scale: its assigned cellular wavelength is again tens of micrometres. On a 23.96 µm patch (exactly eight critical wavelengths of 2.995 µm) the predecessor and successor wavelengths do not fit even once.

The diffusion ratios 8 and 10 are **contrasts defined in this slot** so the ocellus regime can be compared with a stripe regime. They are not parameters copied from the serrata or simulans scripts.

## Biology the model is allowed to use

Carson & Bryant (1979, PNAS 76:1929–1932) showed that males from the north and east of Hawaiʻi carry a mean of 20–30 extra tibial cilia between two major rows, and that those extra cilia are absent in *D. heteroneura* and in three related species on adjacent islands. The gallery common name “100+ tibial courtship cilia rows” is a label, not the count in that paper. Spiess & Carson (1981) tested whether females discriminate among males that differ in the tibial trait. Carson (1982) places the species on the young volcanoes of Hawaiʻi.

Those papers do **not** measure a 3.0 µm Turing wavelength. The 2.995 µm figure is the detail-tier wavelength this loop was assigned, reconstructed as the critical wavenumber of Schnakenberg kinetics (Schnakenberg 1979) under Turing’s dispersion relation (Turing 1952). Macroscopic picture-wing spots in *Drosophila* have a documented cis-regulatory account (Gompel et al. 2005). This slot does not claim those wing spots are a 3 µm reaction-diffusion pattern. The Kilohana photograph is the species voucher for computer-vision specimen match; the reaction-diffusion field is the subcellular wavelength model.

## Specimen

`drosophila_silvestris_turing_wavelength_s0_specimen.jpg`

- Source: [File:Drosophila silvestris Kilohana 5161a.jpg](https://commons.wikimedia.org/wiki/File:Drosophila_silvestris_Kilohana_5161a.jpg)
- Wikimedia Commons user KarlM, 29 September 2019. *Drosophila silvestris* on banana bait, Kohala, Hawaiʻi.
- License: [CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0). Attribution required. Share-alike applies to adaptations of the photograph.

## Run

```bash
python3 models/2026-delphi10000-drosophila-silvestris-turing-wavelength-s0.py --self-test
python3 models/2026-delphi10000-drosophila-silvestris-turing-wavelength-s0.py
```

Outputs:

- `docs/media/drosophila_silvestris_turing_wavelength_s0_comparison.png`
- `docs/media/drosophila_silvestris_turing_wavelength_s0_growth.gif`

Governing law: Turing wavenumber k_c from the dispersion relation of the Schnakenberg system, with λ_c = 2π / k_c = 2.995 µm at the reference reaction scale γ = 1 and inhibitor/activator diffusivity ratio 20.

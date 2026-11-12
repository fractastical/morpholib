# *Botryllus schlosseri* — Turing wavelength, detail scale (slot 3136)

Holocene (extant), **0 Ma**. Characteristic scale **39 µm** (**3.872×10⁻⁵ m**, log10 = −4.412), tier `4_cellular_10um`, detail variant. This is chronoseries step **1/1** for *Botryllus*: there is no fossil predecessor or successor chronospecies in the slot. The comparison formalized by the model is across scale inside the living species, not across a species replacement.

The governing law is Turing’s critical wavenumber

\[
k_c^2 = \frac{D_v f_u + D_u g_v}{2 D_u D_v}, \qquad \lambda_c = \frac{2\pi}{k_c}.
\]

Kinetics are Schnakenberg (1979) on a periodic window of seven wavelengths (271.04 µm). Parameters are specific to slot 3136 (`a = 0.097`, `b = 1.246`, `γ = 1.37`, seed 3136) and are not copied from the sibling slot 0167 script.

## What changes at this scale

- Coarser than this slot, star systems of zooids are spaced on a millimetre scale. That spacing is not \(k_c\) here.
- At 39 µm the model selects one unstable band. The detail window holds seven repeats.
- Finer than this slot, pigment granules a few micrometres across sit at \(k \gg k_c\), where the growth rate is negative, so they are not the selected mode.
- At 0 Ma both the larval ocellus (sensory vesicle; Sorrentino, Manni, Lane & Burighel 2000) and the adult colony pigment pattern are extant. Equal diffusivities are the negative control: the band disappears.

The 39 µm figure is the assigned MorphoEval characteristic length. It is not a caliper reading taken off the colony photograph. The photograph is the organism-scale specimen used for visual comparison.

## Specimen

`Botryllus_schlosseri_Parent1991_Banyuls.jpg` — Géry Parent, *Botryllus schlosseri* (Pallas, 1766), Banyuls-sur-Mer, July 1991. Wikimedia Commons file “Botryllus schlosseri (Pallas, 1766).jpg”, public domain. Vendored JPEG is a quality-90 recompression of that file (3200×2080). SHA-256 is recorded in `botryllus_schlosseri_turing_wavelength_s0_reference.json`.

## Run

```bash
python3 models/2026-delphi10000-botryllus-schlosseri-turing-wavelength-s0.py --self-test
python3 models/2026-delphi10000-botryllus-schlosseri-turing-wavelength-s0.py
```

Outputs:

- `docs/media/botryllus_schlosseri_turing_wavelength_s0_comparison.png`
- `docs/media/botryllus_schlosseri_turing_wavelength_s0_growth.gif`
- `botryllus_schlosseri_turing_wavelength_s0_metrics.json`

## Uniqueness

Repository search found sibling *Botryllus* Turing material under slot 0167 (`manni-botryllus-schlosseri-turing-wavelength`) and none under the `-s0` slug, dataset directory, Lean module `Loop_3136`, or this script name.

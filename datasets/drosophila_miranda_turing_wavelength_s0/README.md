# Drosophila miranda — detail-scale Turing wavelength (slot 3193)

Holocene (extant) **0 Ma** model of the **1.9 µm** (1.864×10⁻⁶ m, log10 = −5.730) Turing wavelength on the ocellar pigment field of *Drosophila miranda* (neo-X / neo-Y sex-chromosome fly). Scale tier `3_subcellular_um`, detail variant. This is not the 102 µm abdominal-band slot (`drosophila_miranda_turing_wavelength`, slot 2911).

## What is at this scale

1.9 µm is the subcellular spacing of an ommochrome pigment-granule / ocellar stripe, not the adult tergite band and not the whole ocellus. Shoup (1966) measured *D. melanogaster* eye pigment granules as sub-micron to about one-micron organelles. The reconstruction places the miranda detail wavelength between its chronoseries neighbours, all at geological age **0 Ma**:

| Chronospecies | Role | Detail wavelength | Wavenumber | Modes in a 22.368 µm field |
|---|---|---|---|---|
| *Drosophila melanogaster* | predecessor | 1.412 µm | higher k | 16 |
| *Drosophila miranda* | this slot (22/41) | 1.864 µm (labelled 1.9 µm) | k_max = 3.371 rad/µm | 12 |
| *Drosophila mojavensis* | successor | 2.286 µm | lower k | 10 |

Nothing in geological age changes along this triple (all Holocene). What changes is the selected wavenumber. Activator diffusivity scales as λ², so the melanogaster lattice is finer and the larger repleta-group mojavensis lattice is coarser. The miranda value is the neo-Y dosage-shifted intermediate. Those predecessor and successor wavelengths are reconstructions under that scaling, not new electron-microscope measurements.

## Model

Schnakenberg kinetics (Schnakenberg 1979), dispersion relation from Turing (1952):

- ∂u/∂t = D_u ∇²u + a − u + u²v
- ∂v/∂t = D_v ∇²v + b − u²v
- Steady state (u₀, v₀) = (a+b, b/(a+b)²) = (1, 0.92) with a = 0.08, b = 0.92
- Trace tr J = −0.16 < 0 and det J = 1 > 0, so the uniform state is stable without diffusion
- D_u = 0.028630 µm²/time, D_v / D_u = 9.601, just above the Turing onset (onset ratio ≈ 7.870)
- Selected wavelength λ_max = 1.864 µm on a periodic 22.368 µm field

An equal-diffusion control (D_u = D_v) keeps the diffusion-weighted trace negative and the pattern collapses. A diffusivity sweep moves λ monotonically as √D_u.

## Specimen plate

`Drosophila_miranda_specimen.jpg` is the repository species-gallery plate (SHA-256 pinned in the reference JSON), copied from `datasets/species_gallery/II_Anaima_Invertebrata/Drosophila_miranda.jpg`. Wikimedia Commons did not return a CC photograph of the species *D. miranda* (hits were the Miranda protein in neuroblasts). The plate is the canonical image this slot is scored against. It is not a transmission electron micrograph of miranda pigment granules.

## Run

```bash
python3 models/2026-delphi10000-drosophila-miranda-turing-wavelength-s0.py --self-test
python3 models/2026-delphi10000-drosophila-miranda-turing-wavelength-s0.py
```

Outputs:

- `docs/media/drosophila_miranda_turing_wavelength_s0_comparison.png`
- `docs/media/drosophila_miranda_turing_wavelength_s0_growth.gif`

Lean proofs: `morphoeval/lean/loops/Loop_3193_drosophila_miranda_turing_wavelength_s0.lean` (`MathOfBiology.Loop_3193`, six theorems, no `sorry`).

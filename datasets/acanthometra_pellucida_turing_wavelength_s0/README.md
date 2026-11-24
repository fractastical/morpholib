# *Acanthometra pellucida* — Turing wavelength, 206 nm detail scale

Slot 3101 of the 10,000-animal basic look. One organism, one length, one point in time.

- **Organism:** *Acanthometra pellucida* Müller (Radiolaria, Acantharia). Extant. Aristotelian path: IV. Protista & Monera → Mineralized Protists (Silica/Calcite) → *Acanthometra*.
- **Age:** Holocene (extant), **0 Ma**. Chronoseries 1/1: there is no earlier chronospecies in this slot. The comparison formalized here is against the neighbouring *scales*, not against a fossil predecessor.
- **Scale:** **206 nm** (2.064×10⁻⁷ m, log10 = −6.685), tier `2_macromolecular_100nm`, detail variant. This is the critical Turing wavelength λ_c of a Schnakenberg activator–inhibitor field on a cortical patch, not the cell-scale spine length.
- **Law:** Turing wavenumber k_c (cohort 1, chemical reaction–diffusion). Form class: stripes, spots, and ocelli — here the carrier that would tile the capsule cortex. The plate drawing resolves the spine plan, not this carrier.

## What changes at this scale and age

*A. pellucida* mineralizes strontium-sulfate (celestite) spicules. The crystallographic cell of celestite is sub-nanometre, so a 206 nm period sits about two orders of magnitude above the lattice: it is a reaction–diffusion wavelength, not a unit-cell repeat.

On a 1339 nm cortical patch (6.5 assigned wavelengths):

| Wavelength | Waves across the patch | What selects it |
|---|---|---|
| ~1 nm (lattice predecessor) | hundreds | crystal structure of SrSO₄ |
| **206 nm (this slot)** | **6** | Schnakenberg k_c at (a, b, γ, d) = (0.14, 0.76, 10, 18) |
| 412 nm (one octave coarser) | 3 | same kinetics, longer scale |
| ~100 μm (organism / plate) | the carrier is aliased (~485 waves per 100 μm) | Müller’s 20-spine ground plan |

At 0 Ma the extant species is the only member of this chronoseries. Nothing in the slot records a Pleistocene-to-Holocene change of λ_c. The Holocene terminal keeps the radial celestite skeleton; the quantity fixed at *this* tier is λ_c = 2π/k_c = 206 nm.

Sibling slot 193 (`thompson-acanthometra-pellucida-turing-wavelength`) is a different simulation, with different parameters. This file does not reuse them.

## Model

Schnakenberg kinetics (Gierer–Meinhardt / Murray form of an activator–inhibitor Turing system; Turing 1952):

∂u/∂t = D_u ∇²u + γ (a − u + u²v)

∂v/∂t = D_v ∇²v + γ (b − u²v)

with a = 0.14, b = 0.76, D_u = 1, D_v/D_u = 18, γ_ref = 10. The steady state is (u, v) = (a+b, b/(a+b)²). The critical wavenumber is

k_c² = (D_v f_u + D_u g_v) / (2 D_u D_v),

λ_c = 2π/k_c. The nondimensional length unit is chosen so that this analytical λ_c equals **206 nm** at γ_ref. The test is not that identity (it is the unit choice). The test is that a nonlinear grid simulation’s dominant wavelength tracks λ_c(γ) across a γ sweep (Pearson r, RMSE), that equal diffusion abolishes the pattern, and that a drawing built from the plate’s measured spine axes plus this carrier resembles the plate.

Boundary conditions on the cortical patch: periodic. The whole-cell drawing masks the field to the capsule disk.

## Specimen

`acanthometra_pellucida_specimen.jpg` is a crop of figure 2 on Ernst Haeckel’s *Kunstformen der Natur*, Tafel 21 (public domain; Wikimedia Commons file `Acanthometra. - Stachelstrahlinge LCCN2015648908.jpg`). The plate is captioned *Acanthometra*, not a photograph of the type of *A. pellucida*. It is the genus-level specimen used for spine-axis measurements. It does not resolve 206 nm.

SHA-256 of the vendored crop is recorded in `acanthometra_pellucida_turing_wavelength_s0_reference.json`.

## Run

```bash
python3 models/2026-delphi10000-acanthometra-pellucida-turing-wavelength-s0.py --self-test
python3 models/2026-delphi10000-acanthometra-pellucida-turing-wavelength-s0.py
```

Outputs:

- `docs/media/acanthometra_pellucida_turing_wavelength_s0_comparison.png`
- `docs/media/acanthometra_pellucida_turing_wavelength_s0_growth.gif`

## Sources

Turing (1952) 10.1098/rstb.1952.0012; Gierer & Meinhardt (1972) 10.1007/BF00289234; Murray (1981) 10.1016/0022-5193(81)90334-9; Kondo & Miura (2010) 10.1126/science.1179047; Decelle et al. (2012) 10.1073/pnas.1212303109 (living acantharians); Haeckel (1904) Tafel 21.

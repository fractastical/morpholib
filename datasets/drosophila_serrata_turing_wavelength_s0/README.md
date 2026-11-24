# Drosophila serrata — Turing wavelength at 2.1 µm (Holocene, 0 Ma)

Slot 3203 of the 10,000-Animal Basic Look. One animal, one scale, one point in the Drosophila chronoseries.

- **Animal:** *Drosophila serrata* Malloch, 1927 — Australasian montium-subgroup fly, known for cuticular-hydrocarbon and tergite clines.
- **Scale:** 2.1 µm detail tier (`3_subcellular_um`). Assigned phenomenon size **2.123 × 10⁻⁶ m** (log10 = −5.673). This is the critical Turing wavelength λ_c = 2π / k_c of an ocellar / pigment-granule reaction–diffusion mode, not the ~92 µm cellular-tier sibling slot.
- **Age:** Holocene (extant), **0 Ma**. Chronoseries position 32/41. Lineage: Sophophora ananassae–montium clade. Predecessor *Drosophila sechellia*. Successor *Drosophila silvestris*.

## What changes at this scale and age

Activator diffusivity is held fixed at the value that places the serrata band-center wavenumber on 2.123 µm. The inhibitor:activator diffusivity ratio δ moves along the chronoseries:

| Chronospecies | Role | δ / δ_c | λ |
|---|---|---|---|
| *D. sechellia* | predecessor | 1.025 | 2134 nm |
| *D. serrata* | this slot, 0 Ma | 1.085 | **2123 nm** |
| *D. silvestris* | successor | 1.220 | 2102 nm |

δ_c is the Schnakenberg onset. Serrata sits between a longer-wavelength island predecessor and a shorter-wavelength picture-wing successor. The Holocene east-coast CHC cline (Frentiu & Chenoweth 2010) is modelled as a further ±1.2% shift of δ about the serrata value; that cline is a reconstruction, not a published nanometre measurement.

## Model

Schnakenberg (1979) kinetics on the Turing (1952) dispersion relation:

∂u/∂t = D_u ∇²u + a − u + u²v

∂v/∂t = D_v ∇²v + b − u²v

with a = 0.037, b = 1.146. The critical wavenumber is

k_c² = (D_v f_u + D_u g_v) / (2 D_u D_v), λ_c = 2π / k_c.

An equal-diffusion control (D_v = D_u) is linearly stable. A radial critical mode J_0(k_c r), weakly anisotropic along the tergite anterior–posterior axis, is the ocellus shown in the growth animation.

## Specimen

Wikimedia Commons, iNaturalist, and GBIF did not return a CC-licensed single-animal photograph (Commons hit: a multi-species plate, `Australian fruitflies.png`, CC BY 2.5; GBIF media: a distribution map). The pinned plate is the project gallery fallback:

`drosophila_serrata_turing_wavelength_s0_specimen.jpg`

copied from `datasets/species_gallery/II_Anaima_Invertebrata/Drosophila_serrata.jpg`.

## Run

```bash
python3 models/2026-delphi10000-drosophila-serrata-turing-wavelength-s0.py --self-test
python3 models/2026-delphi10000-drosophila-serrata-turing-wavelength-s0.py
```

Outputs:

- `docs/media/drosophila_serrata_turing_wavelength_s0_comparison.png`
- `docs/media/drosophila_serrata_turing_wavelength_s0_growth.gif`

Lean proofs: `morphoeval/lean/loops/Loop_3203_drosophila_serrata_turing_wavelength_s0.lean` (`MathOfBiology.Loop_3203`).

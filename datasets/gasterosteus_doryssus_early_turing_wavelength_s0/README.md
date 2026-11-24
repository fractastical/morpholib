# *Gasterosteus doryssus_early* — cellular Turing wavelength (slot 3229)

Miocene stickleback chronospecies I, full pelvic girdle, at the **33 µm** detail rung of the cellular scale (`3.286e-05` m, log10 = −4.483, tier `4_cellular_10um`). Geological age **10.02 Ma**, Late Miocene, Truckee Formation. This slot is chronoseries position 1/6 in the *Gasterosteus* pelvic-reduction lineage. The successor is *Gasterosteus doryssus_mid* (vestigial pelvic spine, T = +9 kyr, 10.01 Ma). There is no earlier chronospecies in this series.

The sibling reproduction `models/2006-bell-gasterosteus-doryssus-early-turing-wavelength.py` treats the same nominal chronospecies at an organ-scale flank wavelength of 2.3 mm. The parameters here are solved for a dermal patch of 280 µm whose fastest-growing Schnakenberg mode sits at λ_c = 32.86 µm. They are not a rescaling of that organ-scale script.

## What changes at this scale and age

| Horizon | Age | Pelvis | Cellular λ | Role in this slot |
|---|---:|---|---:|---|
| Marine complete-morph outgroup | — | full plates | 28.40 µm | shorter ancestral lattice; not a chronoseries predecessor |
| *G. doryssus_early* | 10.02 Ma | full girdle, T = 0 | **32.86 µm** | this model |
| *G. doryssus_mid* (successor) | 10.01 Ma | vestigial spine, T = +9 kyr | 41.50 µm | wavelength lengthens; stated analytically in pass 1 |
| *G. doryssus_late* | 9.99 Ma | pelvic loss, T = +22 kyr | 55 µm | outside this slot |

Hunt, Bell & Travis (2008) showed the Truckee *G. doryssus* pelvis moving toward a reduced optimum across roughly 22 kyr. Bell, Travis & Blouw (2006) inferred selection on that fossil series, and Stuart, Travis & Bell (2020) recovered a Pitx1 contribution. None of those papers report a pigment spacing of 33 µm. The length assigned here is the Delphi cellular-detail rung. The mechanism transferred onto it is the Schnakenberg form of Turing’s dispersion relation, the same class Nakamasu et al. (2009) and Kondo & Miura (2010) used for teleost chromatophores. Extant pelvic reduction (Chan et al. 2010, Pitx1) and armor-plate evolution (Colosimo et al. 2005, Eda) are cited only as the living parallels, not as measurements on this fossil.

At 32.86 µm the early chronospecies still fits about eight critical wavelengths into a 280 µm dermal patch, and the full pelvic girdle is read as an intact circumferential boundary that can hold a two-armed ocellus (angular charge m = 2). The successor’s longer wavelength (41.50 µm) drops that count and is the quantity a later slot should test.

## Governing law

Schnakenberg kinetics on the patch, periodic, with diffusivities locked so the maximum linear growth rate lies at k_c = 2π / 32.86 µm⁻¹:

```
∂u/∂t = D_u ∇²u + a − u + u²v
∂v/∂t = D_v ∇²v + b − u²v
k_c²  = (D_v f_u + D_u g_v) / (2 D_u D_v)
λ_c   = 2π / k_c
```

Locked constants (signalling-time units, lengths in µm): a = 0.062, b = 0.948, D_v/D_u = 12.276923076923076, D_u = 8.204081632653061 µm². Equal diffusion is the negative control: the Jacobian trace is negative and det J is positive, so every wavenumber decays when D_u = D_v.

## Specimen

`gasterosteus_doryssus_early_specimen.jpg` is the project species-gallery image for this chronospecies (SHA-256 `746a8e6539215ffdae2f266963a9c619c30ce4bc965ab94dd6c04d0bc70c3034`), copied because no CC-licensed Truckee Formation thin section at 33 µm was found. It carries a 100 µm scale bar (111 px) and a spiral ocellus. A high-pass radial spectrum of that photograph peaks near 83 µm, which is the visible band, not λ_c. The script reports both numbers and does not set them equal.

## Run

```bash
python3 models/2026-delphi10000-gasterosteus-doryssus-early-turing-wavelength-s0.py --self-test
```

Outputs:

- `docs/media/gasterosteus_doryssus_early_turing_wavelength_s0_comparison.png`
- `docs/media/gasterosteus_doryssus_early_turing_wavelength_s0_growth.gif`

# *Drosophila affinis* — Turing wavelength, detail tier (slot 3174)

Holocene (extant), **0 Ma**. Characteristic scale of this loop: **2.4 µm** (2.444 × 10⁻⁶ m, log10 = −5.612, tier `3_subcellular_um`).

This is the detail-scale carrier wave for ocellar and screening-pigment spacing in the Nearctic woodland obscura fly *Drosophila affinis* Sturtevant, 1916. It is not the 96 µm cellular-tergite slot already in this repository (`datasets/drosophila_affinis_turing_wavelength`, slot 2921), and it does not reuse that slot’s power-law parameters.

## What changes along the chronoseries

Delphi orders this fly as Drosophila chronoseries position 3/41 in the Sophophora obscura & willistoni clade. The neighbors are a project ordering, not a claim that one species is the other’s direct ancestor.

| Position | Species | Age | Assigned wavelength | Tier |
| --- | --- | --- | --- | --- |
| Predecessor | *Drosophila berryi* (Miocene Dominican amber) | 19 Ma | 98.43 µm | cellular |
| **This slot** | ***Drosophila affinis*** | **0 Ma** | **2.444 µm** | **subcellular detail** |
| Successor | *Drosophila americana* (virilis group, extant) | 0 Ma | 138.7 µm | primordium |

At fixed Schnakenberg kinetics the classical critical wavelength scales with the activator diffusivity. Moving from berryi’s 98.43 µm cellular assignment down to this 2.444 µm detail assignment is a factor of 40.3 in length and about 1.6 × 10³ in Du if the same kinetics were reused. This slot does not reuse that diffusivity. It sets Du so the determinant-minimizing wavenumber satisfies λ_c = 2π/k_c = 2.444 µm on a 20 × 15 wavelength patch (48.88 µm × 36.66 µm), with inhibitor/activator diffusivity ratio 10, just above the onset ratio ≈ 7.63. Versus americana’s 138.7 µm puparial-scale assignment the same law is evaluated 56.8 times finer. The morphology selected here is a field of pigment-granule / ocellar spots inside one tergite or ocellar patch, not a whole-abdomen stripe and not a puparial color field.

The diagnostic sex comb on the distal tarsomere (Miller 1955) identifies the species. Tooth spacing is coarser than 2.444 µm and is not the quantity this law selects.

## Model

Schnakenberg kinetics (Schnakenberg 1979) on the Turing dispersion relation (Turing 1952):

- ∂u/∂t = a − u + u²v + Du ∇²u
- ∂v/∂t = b − u²v + Dv ∇²v
- k_c² = (Dv f_u + Du g_v) / (2 Du Dv)
- λ_c = 2π / k_c = 2.444 µm

Equal diffusivities are the negative control: the homogeneous steady state stays stable and no wavelength is selected. A ratio sweep from 1 to 24 is the parameter-expressivity check.

## Specimen

`drosophila_affinis_specimen.jpg` is a lateral habitus of a female, converted from Wikimedia Commons `File:Daff-fem.tif` (MA Han / user Crawdaunt, CC BY-SA 4.0). The print carries a 250 µm scale bar, so the 2.444 µm wavelength is below the sampling of the photograph. The habitus is the empirical adult; the 2.444 µm field is the subcellular reconstruction shown in the growth strip and in the dispersion panel.

SHA-256 of the JPEG is recorded in `drosophila_affinis_turing_wavelength_s0_reference.json`.

## Run

```bash
python3 models/2026-delphi10000-drosophila-affinis-turing-wavelength-s0.py --self-test
python3 models/2026-delphi10000-drosophila-affinis-turing-wavelength-s0.py
```

Outputs:

- `docs/media/drosophila_affinis_turing_wavelength_s0_comparison.png`
- `docs/media/drosophila_affinis_turing_wavelength_s0_growth.gif`

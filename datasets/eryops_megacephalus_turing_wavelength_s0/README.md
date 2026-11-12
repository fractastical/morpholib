# Eryops megacephalus — 127 µm Turing wavelength (slot 3222)

Early Permian (Cisuralian) temnospondyl, **295 Ma**. This slot is the detail-scale
primordium tier (`5_primordium_100um`): the selected wavelength is **127 µm**
(1.267×10⁻⁴ m, log10 = −3.897). It is not the 8.184 mm organ-scale dermal-pit
slot (Loop 1217).

Eryops is a one-point chronoseries here. The predecessor comparison is the
Famennian stem-tetrapod horizon at 365 Ma. In micrometres,
8184 = 64 × 127 + 56, so that coarser pit holds 64 of these primordium
wavelengths and a 56 µm remainder. There is no later Eryops chronospecies in
the series; amniote epidermal scales are a different regime and are not given
a copied wavelength.

## Model

Schnakenberg (1979) kinetics with the Turing (1952) critical wavenumber

```
k_c² = γ (D_v f_u + D_u g_v) / (2 D_u D_v),    λ_c = 2π / k_c.
```

Parameters unique to this slot: `a = 0.085`, `b = 1.22`, `γ = 1`,
reference `(D_u, D_v) = (0.16, 8)`, then both diffusivities are scaled so
`λ_c` is 12 pixels and 12 pixels are 127 µm. The patch is 8 wavelengths
wide (1.016 mm), periodic, seed 3222.

## Specimen

`eryops_megacephalus_amnh_specimen.jpg` is the CC BY 4.0 photograph by
Vicpeters of the *Eryops megacephalus* mount at the American Museum of
Natural History (Wikimedia Commons, 14 May 2025). SHA-256
`11fc4e4eec64d5f1cd5cc143b6ac1832f8ea67ac18f4d0a5ee25c2af919f0b66`.

The photograph identifies the animal. It does not resolve 127 µm. The
ocellus field is drawn only inside an eroded Otsu bone mask.

## Run

```bash
python3 models/2026-delphi10000-eryops-megacephalus-turing-wavelength-s0.py --self-test
```

Outputs:

- `docs/media/eryops_megacephalus_turing_wavelength_s0_comparison.png`
- `docs/media/eryops_megacephalus_turing_wavelength_s0_growth.gif`

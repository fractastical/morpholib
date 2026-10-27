# Astyanax mexicanus — Turing wavelength at 666 nm (slot 3125)

Holocene cave morph, geological age **0 Ma**. Phenomenon scale **666 nm**
(SI value on the loop card: **6.656e-7 m**, log10 = -6.177, tier
`2_macromolecular_100nm`, detail variant `s0`).

The law is Turing wavenumber selection, k_c, for stripes, spots and ocelli
(cohort: chemical Turing / reaction-diffusion). The carrier kinetics are
Schnakenberg (1979). This file is the 666 nm detail slot. It is not the
24 µm cellular sibling (`datasets/astyanax_mexicanus_turing_wavelength`,
λ = 2.418e-5 m).

## What changes at this scale

The chronoseries for this animal in the campaign is a single extant point
(Holocene cave morph, 0 Ma). There is no older fossil predecessor and no
successor chronospecies. The comparison that this scale forces is against

- the sympatric surface morph at the same age (pigment amplitude, not k_c), and
- the cellular-scale sibling wavelength, 24 µm.

| Quantity | 666 nm (this slot) | 24 µm sibling |
|---|---|---|
| λ | 6.656e-7 m | 2.418e-5 m |
| k ratio | 1 | ≈ 36.3× smaller k |
| (k ratio)^2 | 1 | ≈ 1320× |
| Biological reading | melanosome-scale packing | chromatophore spacing |

A flank a few millimetres long holds millions of 666 nm wavelengths and only
about a hundred 24 µm wavelengths. Holding the reaction Jacobian fixed, the
diffusion-time ratio D/γ that selects 666 nm is ~1300 times smaller than the
ratio that selects 24 µm. That is cytosolic or intra-melanophore transport,
not an extracellular morphogen diffusing across the flank. The cave morph
adds a second change that is not a change in k_c: Oca2 down-regulation and
Mc1r reduce melanin, so the mode is optically nearly silent. Eye loss
(lens apoptosis under Hedgehog; Yamamoto, Stock & Jeffery 2004) and neuromast
expansion (Yoshizawa et al. 2010) are not predictions of this wavelength.

## Model

∂u/∂t = Du ∇²u + γ (a − u + u²v)

∂v/∂t = Dv ∇²v + γ (b − u²v)

with a = 0.1, b = 0.9, Du = 2, Dv = 20, γ = 1 on a periodic grid whose
metres-per-pixel are set so the fastest linear mode has wavelength 6.656e-7 m.
The script reports the forward-Euler dispersion fit, one nonlinear patch,
a Dv/Du sweep on the unstable branch (wavelength lengthens as the ratio
grows), a Du = Dv ablation (growth rate stays negative), and a
Kolmogorov-Smirnov comparison of flank orientations on the voucher.

## Specimen

`astyanax_mexicanus_specimen.jpg` — captive cave tetra (*Astyanax mexicanus*),
side profile, photographed by Seb Morič, 2025-09-20. Wikimedia Commons file
`Cave_tetra_(Astyanax_mexicanus).jpg`, licence CC BY-SA 4.0.
https://commons.wikimedia.org/wiki/File:Cave_tetra_(Astyanax_mexicanus).jpg

The photograph does not resolve 666 nm. The animation composites the simulated
patch onto the body mask as a visualization. It is not a claim that the
voucher shows melanin stripes at that wavelength.

## Run

```bash
python3 models/2026-delphi10000-astyanax-mexicanus-turing-wavelength-s0.py --self-test
```

Outputs:

- `docs/media/astyanax_mexicanus_turing_wavelength_s0_comparison.png`
- `docs/media/astyanax_mexicanus_turing_wavelength_s0_growth.gif`
- `datasets/astyanax_mexicanus_turing_wavelength_s0/pass1_metrics.json`

Formal statements: `morphoeval/lean/loops/Loop_3125_astyanax_mexicanus_turing_wavelength_s0.lean`
(`MathOfBiology.Loop_3125`).

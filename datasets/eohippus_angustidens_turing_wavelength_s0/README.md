# Eohippus angustidens — Turing wavelength at 78 µm (slot 3219)

Early Eocene (Wasatchian, **52.0 Ma**) four-toed dawn horse. This dataset is the
cellular-detail variant (`4_cellular_10um`, detail `s0`) of Turing pigment
stripe / ocellus wavelength selection. The characteristic length is **78 µm**
(**7.752e-05 m**, log10 = **−4.111**).

The chronoseries inside *Eohippus angustidens* has a single stage (1/1). There is
no earlier *E. angustidens* chronospecies to difference against. What changes at
this scale, relative to the same animal’s organ-scale wavelength (3.3 mm) and to
the succeeding Bridgerian equid *Orohippus* (~48 Ma), is stated in
`eohippus_angustidens_turing_wavelength_s0_reference.json`.

## Governing model

Schnakenberg (1979) kinetics with the Turing (1952) dispersion relation and the
Murray (1981) critical-wavenumber formula

```
k_c^2 = (D_v f_u + D_u g_v) / (2 D_u D_v)
lambda_c = 2 pi / k_c = 7.752e-05 m
```

Diffusion ratio `D_v/D_u = 8.22` is fixed for this slot so the formula returns
that wavelength. It is not the parameter set of the 3.3 mm organ-scale sibling.

## Specimen

`eohippus_angustidens_specimen.jpg` is the canonical gallery plate
(`datasets/species_gallery/I_Enaima_Vertebrata/Eohippus_angustidens.jpg`),
SHA-256 `4f492b7fff0d285dc090002b29f9aca4a757bad6f4c08a8f5dd93b8a1bcfc0bf`.
The plate carries a 100 µm scale bar (111 px). Wikimedia Commons returned no
file for this taxon during the scout, so the gallery plate is the vendored
specimen. It is a pattern plate at the cellular tier, not a histological
section of fossil integument. Eocene horse coat pigment is not preserved;
the 78 µm target is the assigned phenomenon scale, not a measured fossil
spacing.

## Reproduce

```bash
python3 models/2026-delphi10000-eohippus-angustidens-turing-wavelength-s0.py --self-test
python3 models/2026-delphi10000-eohippus-angustidens-turing-wavelength-s0.py
```

Outputs:

- `docs/media/eohippus_angustidens_turing_wavelength_s0_comparison.png`
- `docs/media/eohippus_angustidens_turing_wavelength_s0_growth.gif`
- `eohippus_angustidens_turing_wavelength_s0_metrics.json` (written beside this file)

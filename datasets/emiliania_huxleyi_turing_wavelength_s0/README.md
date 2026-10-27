# Emiliania huxleyi — Turing wavelength at 5.8 nm (Holocene, 0 Ma)

Slot 3217. One organism, one scale, one age.

*Emiliania huxleyi* is an extant coccolithophore. This loop is the Holocene node of its chronoseries (index 1 of 1, **0 Ma**). The modelled scale is **5.8 nm** (5.808 × 10⁻⁹ m, log10 = −8.236), tier `1_molecular_nm`.

## Law

Schnakenberg substrate-depletion kinetics (Schnakenberg 1979) with the Turing marginal wavenumber (Turing 1952):

```
k_c² = (D_v f_u + D_u g_v) / (2 D_u D_v)
λ_c  = 2π / k_c = 5.808 nm
```

Parameters `a = 0.10`, `b = 0.90`, `D_v/D_u = 18` put that peak on a 46.464 nm patch (eight wavelengths). An equal-diffusion ablation moves the determinant’s vertex off the positive-k axis, so no wavelength is selected.

## What changes at this age

There is no successor chronospecies. The reconstructed predecessor places six wavelengths on the same patch, which forces `λ = 7.744 nm`. Crossing into the Holocene node shortens the repeat by 1.936 nm and adds two modes. That predecessor length is a reconstruction for this slot, not a fossil measurement and not a parameter copied from another coccolithophore loop.

## Specimen

`emiliania_huxleyi_coccosphere_coccolith.jpg` is the Wikimedia Commons scan of an SEM of a coccosphere and a coccolith (CC BY-SA 4.0; Neukermans, Fournier, and Young; SHA-256 `94d4b431c4912727a89fa0bea09c87c8add8c286e7f3c259558b85341950b65d`). The micrograph is micrometre-scale. It is the phenotypic target the molecular field is mapped onto. It is not a 5.8 nm image.

The tangential-Sobel annulus on the right-hand coccolith peaks at angular harmonic 30 (spoke score 5.20). That count sets the shield spokes in the animation.

## Run

```bash
python3 models/2026-delphi10000-emiliania-huxleyi-turing-wavelength-s0.py --self-test
python3 models/2026-delphi10000-emiliania-huxleyi-turing-wavelength-s0.py
```

Outputs:

- `docs/media/emiliania_huxleyi_turing_wavelength_s0_comparison.png`
- `docs/media/emiliania_huxleyi_turing_wavelength_s0_growth.gif`
- `emiliania_huxleyi_turing_wavelength_s0_metrics.json`

On the selection band the integrator matches analytic growth rates at Pearson r = 0.994 (RMSE 0.018). The strongly damped wing is worse (RMSE 0.063) and is reported separately.

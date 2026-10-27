# Asterias rubens — Turing wavelength at 115 µm (slot 3123)

Holocene (extant), **0 Ma**. Characteristic scale **115 µm** (catalog length **0.0001152 m**, log10 ≈ −3.9386), tier `5_primordium_100um`, detail variant. The pattern class is stripes, spots and ocelli. The quantitative law is the Turing wavenumber \(k_c\).

This chronoseries contains one point. There is no earlier *Asterias* chronospecies in the slot. What changes at this scale, relative to a ~10 µm cellular tier and a ~1 mm arm-tip organ tier, is that a diffusion-driven wavelength first fits in an epithelial patch: the Neumann domain is **576.0 µm = 5 × 115.2 µm**, one critical wavelength per pentaradial sector.

## Model

Schnakenberg (1979) kinetics, linearised with the Turing (1952) conditions as arranged by Murray (1981):

- \(\partial_t u = D_u \nabla^2 u + a - u + u^2 v\)
- \(\partial_t v = D_v \nabla^2 v + b - u^2 v\)
- \(k_c^2 = (D_v f_u + D_u g_v) / (2 D_u D_v)\), \(\lambda_c = 2\pi / k_c = 115.2\,\mu\mathrm{m}\)

Slot parameters (not the slot-154 template): \(a = 0.072\), \(b = 0.918\), \(D_v/D_u = 7.62\), seed 3123. The ratio sits just above onset, so on this patch the five-wavelength mode is unstable and the four-wavelength mode is not. Boundary condition: Neumann.

## Specimen

`asterias_rubens_specimen.jpg` — Hans Hillewaert, CC BY-SA 4.0, Wikimedia Commons `File:Asterias rubens.jpg`, Belgian continental shelf, studio photograph. SHA-256 `7d7d25e638cab5b218ae8c7514434eb9e477da03d1f2f87f23da08d9778c4545`.

The plate is an organism-scale animal and has **no scale bar**. It is not a 115.2 µm micrograph. Garm & Nilsson (2014, doi:10.1098/rspb.2013.3011) document a compound eye at the starfish arm tip; their experiments are on *Linckia laevigata*, so that paper is not treated as an *Asterias* wavelength measurement.

## Run

```bash
python3 models/2026-delphi10000-asterias-rubens-turing-wavelength-s0.py --self-test
python3 models/2026-delphi10000-asterias-rubens-turing-wavelength-s0.py
```

Outputs: `docs/media/asterias_rubens_turing_wavelength_s0_comparison.png` and `docs/media/asterias_rubens_turing_wavelength_s0_growth.gif`.

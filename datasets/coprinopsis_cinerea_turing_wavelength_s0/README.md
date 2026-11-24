# Coprinopsis cinerea — Turing wavelength at 77 µm (slot 3154)

Holocene extant gray shag (*Coprinopsis cinerea*), age **0 Ma**, chronoseries **1/1**.
Characteristic scale **77 µm** (exact **7.744e-05 m**, 77.44 µm, log10 = **-4.111**), tier `4_cellular_10um`, detail variant.

This chronoseries has no predecessor and no successor. The scale contrast is the tier floor at 10 µm. The detail variant is 7.744 times longer. Under the Turing wavenumber

```
k_c² = γ √(det J / (D_u D_v)),    λ_c = 2π / k_c
```

with Schnakenberg kinetics held fixed, moving from 77.44 µm down to the 10 µm floor means raising the reaction scale γ by about 60, or shrinking the diffusion product D_u D_v by about 3600. At 0 Ma the extant pileus is the terminus, so that rescaling is a scale contrast, not a second chronospecies.

## Model

Schnakenberg (1979) kinetics on Turing's (1952) wavelength-selection law:

- f = a − u + u²v, g = b − u²v, with a = 0.10, b = 0.90
- steady state (u₀, v₀) = (1, 0.9), det J = 1, trace J = −0.2
- D_u = 1, D_v = 16 at the operating point (equal diffusion is the ablation)
- 96 × 96 periodic patch, 12 pixels per critical wave, explicit Euler

γ is solved so λ_c equals 7.744e-05 m. It is not taken from another animal's script.

## Specimen

`coprinopsis_cinerea_specimen.jpg` is a 960 px Wikimedia thumbnail of Mushroom Observer image 593470 (Igor / Igor_Yevdokimov, 2015-05-30), license CC BY-SA 3.0.

SHA-256: `f5905e9c02019fc55d651c22742ce29a53541673e37355631a29a5076046b06b`

The photograph has no scale bar. Gill spacing on the mature cap is not the 77 µm cellular wavelength. The simulation is an 8-wavelength cuticle patch (619.52 µm on a side), coloured with the photograph's mean RGB only.

## Sources

- Turing 1952, doi:10.1098/rstb.1952.0012
- Schnakenberg 1979, doi:10.1016/0022-5193(79)90042-0
- Gierer & Meinhardt 1972, doi:10.1007/BF00289234
- Kües 2000, doi:10.1128/MMBR.64.2.316-353.2000
- Stajich et al. 2010, doi:10.1073/pnas.1003391107

Citations are in `paper_citations.bib`. The scout record is `scout_manifest.json`.

## Run

```bash
python3 models/2026-delphi10000-coprinopsis-cinerea-turing-wavelength-s0.py --self-test
python3 models/2026-delphi10000-coprinopsis-cinerea-turing-wavelength-s0.py
```

Outputs:

- `docs/media/coprinopsis_cinerea_turing_wavelength_s0_comparison.png`
- `docs/media/coprinopsis_cinerea_turing_wavelength_s0_growth.gif`
- `datasets/coprinopsis_cinerea_turing_wavelength_s0/coprinopsis_cinerea_turing_wavelength_s0_metrics.json`

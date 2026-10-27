# Diplomystus dentatus — 184.3 µm Turing wavelength (slot 3170)

Detail-tier reaction–diffusion wavelength for *Diplomystus dentatus* Cope, 1877, the Green River armored herring (ellimmichthyid). This slot is one point in the Actinopterygian Axial Fossil Series: **Early Eocene (Ypresian), 51.5 Ma**, chronoseries index **1/1**.

| Quantity | Value |
| --- | --- |
| Phenomenon | Turing pigment stripe / ocellus wavelength |
| Scale | **184.3 µm** (0.0001843 m, log10 = −3.734) |
| Tier | `5_primordium_100um`, detail variant |
| Age | **51.5 Ma**, Early Eocene (Ypresian) |
| Cohort | 1. Chemical Turing / RD — Stripes, Spots & Ocelli — wavenumber \(k_c\) |
| Specimen | Didier Descouens, Fossil Lake, Wyoming, CC BY-SA 4.0, caption size 48 cm |

The sibling organ-tier model (`models/1984-grande-diplomystus-dentatus-turing-wavelength.py`) is a different slot: **7.56 mm**, not this file. There is no older or younger *Diplomystus* chronospecies in this series. What changes at 184.3 µm is the scale, relative to a sub-wavelength chromatophore predecessor (~20 µm, shorter than \(\lambda_c/2\)) and to that 7.56 mm successor (wavenumber ratio \(7560/184.3 > 41\)).

## Law

Schnakenberg kinetics on a no-flux flank patch. With Jacobian entries \(f_u, f_v, g_u, g_v\) and diffusivities \(D_u, D_v\),

\[
k_c^2 = \frac{D_v f_u + D_u g_v}{2 D_u D_v}, \qquad \lambda_c = \frac{2\pi}{k_c}.
\]

\(D_u\) is solved so \(\lambda_c = 184.3\,\mu\mathrm{m}\) at the slot ratio \(D_v/D_u = 9.17\), just above the Turing threshold for these kinetics. Parameters are not taken from the 7.56 mm sibling.

## Run

```bash
python3 models/2026-delphi10000-diplomystus-dentatus-turing-wavelength-s0.py --self-test
python3 models/2026-delphi10000-diplomystus-dentatus-turing-wavelength-s0.py
```

Outputs:

- `docs/media/diplomystus_dentatus_turing_wavelength_s0_comparison.png`
- `docs/media/diplomystus_dentatus_turing_wavelength_s0_growth.gif`

## Claim boundary

Grande (1982) records dorsal scutes (usually 22–36) and a maximum length near 65 cm. Those are organ-scale serial elements. The 184.3 µm figure is the catalogued detail-tier wavelength selected by the dispersion relation. It is not a spacing counted on the fossil photograph. Pigment kinetics are the Schnakenberg system used for teleost Turing patterns (Kondo & Asai 1995; Nakamasu et al. 2009), transferred to this flank. The 51.5 Ma age is the series datum; Smith, Carroll & Singer (2008) supply the Green River geochronologic framework and are not re-dated here.

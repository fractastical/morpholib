# Encrinus liliiformis — 12 nm Turing wavelength (Loop 3218)

Middle Triassic sea-lily *Encrinus liliiformis* (Muschelkalk, **240 Ma**), scored at the molecular detail scale **12 nm** (`1.238e-8 m`, log10 = −7.907, tier `1_molecular_nm`).

This is not the 612 nm pore-scale sibling (`models/2014-gorzelak-encrinus-liliiformis-turing-wavelength.py`). At 12 nm the field of view is the interior of one calcite nanograin. Stolarski, Gorzelak et al. (2009, doi:10.4202/app.2009.0108) measured crinoid stereom nanograins of about 100 nm separated by gaps of a few nanometres. The 12 nm wavelength sits in that gap-to-grain window. It is an assigned Delphi detail scale, not a TEM measurement on a Muschelkalk ossicle.

## Law

Schnakenberg kinetics (doi:10.1016/0022-5193(79)90042-0) on the Turing dispersion relation (doi:10.1098/rstb.1952.0012):

```
k_c^2 = (D_v f_u + D_u g_v) / (2 D_u D_v)
lambda_c = 2 pi / k_c = 1.238e-8 m
```

Encrinus parameters, used only at this scale: `a = 0.047`, `b = 1.153`, `D_v/D_u = 22.4`.

## What changes at 240 Ma

| Chronospecies | Age | λ | D_v/D_u |
| --- | --- | --- | --- |
| *Holocrinus dubius* (predecessor) | 248 Ma | 18.72 nm | 18.0 |
| *Encrinus liliiformis* (this slot) | 240 Ma | 12.38 nm | 22.4 |
| *Chelocrinus cassianus* (successor) | 232 Ma | 15.64 nm | 15.6 |

The Muschelkalk point is the short-wavelength, high-δ end-member. Equal diffusion (`D_v = D_u`) is the negative control and does not pattern.

## Specimen

`encrinus_liliiformis_specimen.jpg` is a CC BY-SA 4.0 photograph by Wikimedia user Ghedoghedo, taken at the Muschelkalkmuseum Ingelfingen on 2017-07-05 (File:Encrinus liliiformis 5645.jpg). SHA-256 `9bb90bc584ae7e8508fad1e7e4243f3c44d949bfcfccc947ebddfe9e596d42f3`.

`encrinus_liliiformis_stereom_gyroid_canonical.jpg` is the project gallery rendering used only as the radial microstructure target. Its scale bar reads 100 µm; it is not a 12 nm micrograph.

## Run

```bash
python3 models/2026-delphi10000-encrinus-liliiformis-turing-wavelength-s0.py --self-test
```

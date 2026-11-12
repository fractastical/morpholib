# Ceratopteris richardii — Turing wavelength at 256 µm (slot 3143)

Holocene extant *Ceratopteris richardii* (water fern, C-fern), age **0 Ma**.
This slot is chronoseries node **1/1** (no fossil predecessor and no successor in the series).

Scale: **256 µm** (0.000256 m, log10 = −3.592), tier 5/8 `5_primordium_100um`, detail variant.
The quantitative law is Turing wavenumber selection, \(k_c\), for stripes, spots and ocelli.

The observation window is one critical wavelength. A young notch-meristem / archegonium
field of side \(3\lambda_c = 768\) µm holds three wavelengths. That is the primordium-detail
scale, not the centimetre frond and not the whole cordate gametophyte.

## What is specific to this scale and age

Diffusivities are set in micrometre units for this window only: activator
\(D_u = 8\) µm² s⁻¹, inhibitor \(D_v = 96\) µm² s⁻¹ (ratio 12). Schnakenberg
rates \(a = 0.08\), \(b = 0.92\), and the reaction scale \(\gamma\) is the unique
value that places \(k_c = 2\pi / 256\,\mu\mathrm{m}\). A counterfactual predecessor
with a doubled apical-cell length, diffusivities scaling as length squared and
\(\gamma\) held fixed, would select \(\lambda = 512\) µm. Nothing in the series
measures that fossil. The extant Holocene node is the only one scored.

The canonical gallery plate carries a drawn 100 µm scale bar (106 px). Plate width
is therefore about 755 µm, or 2.95 critical wavelengths. The CC BY-SA voucher is a
separate transmitted-light gametophyte, not the fitness-crop plate.

## Sources

Governing kinetics: Turing (1952), Gierer & Meinhardt (1972), Schnakenberg (1979).
Meristem context at this scale: Bartz & Gola (2018), Geng, Yan & Zhou (2022).
Organism: Hickok, Warne & Fribourg (1995), Banks (1999), Plackett et al. (2014, 2015),
Marchant et al. (2022). Citations are in `paper_citations.bib`. Those papers do not
report a measured 256 µm pigment wavelength; the length is the slot scale.

## Images

- `Ceratopteris_richardii_primordium_plate.jpg` — canonical gallery plate (SHA-256 pinned in the reference JSON). Fitness comparison uses this plate.
- `Ceratopteris_richardii_cfern_3weeks_cc-by-sa.jpg` — Wikimedia Commons `File:C-Fern Development - 3 weeks.jpg`, CC BY-SA 4.0, anatomical voucher of a real gametophyte.

## Run

```bash
python3 models/2026-delphi10000-ceratopteris-richardii-turing-wavelength-s0.py --self-test
python3 models/2026-delphi10000-ceratopteris-richardii-turing-wavelength-s0.py
```

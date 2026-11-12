# *Chlamydomonas reinhardtii* — 10.32 nm eyespot pigment stripe (slot 3146)

Holocene extant biflagellate microalga (0 Ma). Chronoseries 1/1: there is no fossil
*Chlamydomonas* predecessor in the catalog, so the comparison is across **length scales**
of the eyespot (ocellus), not across chronospecies.

| Scale | Length | What it is |
| --- | --- | --- |
| Predecessor (molecular monomer) | ~3.1 nm | Long axis of one β-carotene molecule. Below a collective wavenumber. |
| **This slot (detail, tier 1 `1_molecular_nm`)** | **10.32 nm (1.032×10⁻⁸ m, log10 = −7.986)** | Turing pigment stripe selected inside one carotenoid globule. Age **0 Ma**. |
| Successor (quarter-wave plate) | ~93 nm | Globule-layer spacing d = λ_light / (4 n) for 540 nm light at n = 1.45 (Foster & Smyth 1980). |

Sibling slot 195 models cell-scale eyespot *placement*. This slot does not reuse its
parameters (α = 1.63, β = 0.43, integers 14 / 10 / 140). The kinetics here are
Schnakenberg (1979) with a = 0.07, b = 0.95, D_u = 1, D_v = 36.

The spatial unit is chosen so the **fastest linearly growing wavelength**
λ(k_max) equals 10.32 nm. The onset wavenumber k_c (Turing 1952) is reported
beside it: above onset, k_max < k_c, so the stripe that actually emerges is the
10.32 nm mode, not the shorter marginal wavelength 2π/k_c.

A periodic patch of six stripes is 61.92 nm wide, which fits inside one carotenoid
globule (~100 nm) and is far below the ~10 µm cell body on the specimen SEM.

## Specimen

Public-domain scanning electron micrograph, Dartmouth Electron Microscope Facility:

`chlamydomonas_reinhardtii_specimen.jpg`

SHA-256 `7f2a2df501eb79778d39295925e1de1102c6b2aa6ab8a3dac4443ee5a5fc2a02`

Source: https://commons.wikimedia.org/wiki/File:Chlamydomonas6-1.jpg

The 10 nm pigment lattice is below SEM resolution. The animation therefore keeps
the real cell silhouettes and flagella, and modulates only the segmented cell
bodies by the simulated activator field.

## Run

```bash
python3 models/2026-delphi10000-chlamydomonas-reinhardtii-turing-wavelength-s0.py --self-test
python3 models/2026-delphi10000-chlamydomonas-reinhardtii-turing-wavelength-s0.py
```

This continuum reaction-diffusion system is a reduced description of pigment
aggregation. It does not replace the EYE2/EYE3 assembly pathway (Boyd et al. 2011).

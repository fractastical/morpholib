# *Drosophila hydei* — Turing wavelength, detail scale (slot 3189)

Holocene (extant), geological age **0 Ma**. Anatomical scale **2.3 µm** (canonical **2.344×10⁻⁶ m**, log10 = **−5.630**, tier 3/8 `3_subcellular_um`).

This slot is the detail-scale critical wavelength of a pigment stripe / ocellar-cuticle mode in *Drosophila hydei* (repleta-group giant-sperm fly; campaign epithet “Giant-Sperm Repleta Fly”). It is not the primordium-scale sibling `datasets/drosophila_hydei_turing_wavelength/` (slot 3001, 117.2 µm).

## What changes at this scale

All three chronoseries neighbours are alive in the Holocene (0 Ma). The difference is the selected wavenumber, not the geological age.

| Chronospecies | Role | λ | k / k_hydei,detail |
| --- | --- | --- | --- |
| *D. heteroneura* | predecessor (picture-wing band) | 175.4 µm | 1/74.83 |
| *D. hydei* primordium slot 3001 | same species, coarser tier | 117.2 µm | 1/50 exactly |
| *D. kikkawai* | successor (tergite pigment polymorphism) | 93.41 µm | 1/39.85 |
| **D. hydei s0 (this slot)** | detail cuticle stripe | **2.344 µm** | **1** |

Lineage label: Subgenus *Drosophila* virilis–repleta–guttifera radiation. Drosophila chronoseries position 18/41. The predecessor is a Hawaiian picture-wing (*Idiomyia*), the successor is a *Sophophora* montium fly. At 0 Ma the repleta detail scale selects a cuticle-immobilized activator wavenumber about 50× the same species’ primordium slot and about 75× the predecessor’s wing-band slot. The successor does not inherit that micrometre wavenumber; its tergite comparison sits at 93 µm.

## Law

Schnakenberg (1979) kinetics with the Turing (1952) dispersion relation. Steady state `(u*, v*) = (a+b, b/(a+b)²) = (1, 0.9)` at `a = 0.10`, `b = 0.90`. With `γ = 1`, `D_u = 1`, `D_v = 20`:

- `f_u = 0.8`, `f_v = 1`, `g_u = −1.8`, `g_v = −1`
- trace = −0.2 < 0, det = 1 > 0
- `k_c² = (D_v f_u + D_u g_v) / (2 D_u D_v)`, `λ_c = 2π / k_c`
- length unit chosen so `λ_c(γ = 1) = 2.344 µm`
- quadrupling `γ` halves `λ` at fixed diffusion (`λ ∝ γ^(−1/2)`)
- equal diffusion (`D_u = D_v`) makes `D f_u + D g_v = D · trace < 0`, so the Turing condition fails (negative control)

Boundary condition: periodic square patch, side `4 λ_c = 9.376 µm` (four critical wavelengths). This is a thoracic cuticle patch, not the whole animal.

## Specimen

`drosophila_hydei_turing_wavelength_s0_specimen.jpg` is a real photograph, not the synthetic gallery JPEG.

- File: [Drosophila hydei 09.JPG](https://commons.wikimedia.org/wiki/File:Drosophila_hydei_09.JPG) on Wikimedia Commons
- Photographer: Pogrebnoj-Alexandroff (User:Cherus), Sony DSC-H9, 2012-10-31
- Licence: CC BY 3.0
- Direct URL: https://upload.wikimedia.org/wikipedia/commons/9/9d/Drosophila_hydei_09.JPG
- SHA-256: `483d25650491bb683516b121a95022e8dde199389da0aa3a098a5ea0083e80c3`

The photograph is a whole fly (body length on the order of 3 mm). It does **not** resolve a 2.3 µm stripe. Computer-vision scores compare body-plan texture of this photograph with the animated render. The 2.344 µm figure is the model’s critical wavelength on the cuticle patch, checked against the linear dispersion law and the nonlinear FFT of the field.

## Claim boundary

Reported in source: Turing’s critical-wavenumber instability; Schnakenberg kinetics; Pitnick, Spicer & Markow (1995) on giant *Drosophila* sperm; Gompel et al. (2005) that *Drosophila* pigment patterns have a demonstrated cis-regulatory (yellow) origin in other species.

Reconstructed here: a Schnakenberg field whose linear `λ_c` is the slot scale 2.344 µm, with the chronoseries ratios above.

Not claimed: that a 2.3 µm Turing instability has been measured in *D. hydei* cuticle; that the campaign epithet “58 mm sperm” was re-measured in this loop (Pitnick, Spicer & Markow 1995, *Nature* 375:109, is the giant-sperm citation; *D. bifurca* is the species that paper is known for near 6 cm); that Gompel-style cis-regulatory control is absent.

## Run

```bash
python3 models/2026-delphi10000-drosophila-hydei-turing-wavelength-s0.py --self-test
python3 models/2026-delphi10000-drosophila-hydei-turing-wavelength-s0.py
```

Outputs:

- `docs/media/drosophila_hydei_turing_wavelength_s0_comparison.png`
- `docs/media/drosophila_hydei_turing_wavelength_s0_growth.gif`

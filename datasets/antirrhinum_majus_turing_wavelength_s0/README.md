# Antirrhinum majus — Turing ocellus wavelength at 259 µm (slot 3114)

Holocene (extant), **0 Ma**. One chronospecies. The assigned scale is **259 µm** (0.0002592 m, log10 = −3.586), tier `5_primordium_100um`, detail variant `s0`.

This is not slot 176. That slot’s placeholder coefficients are not used.

## What changes at this scale

There is no fossil predecessor in the Antirrhinum chronoseries (1/1). The comparison is across scales of the living plant:

- **Predecessor (sub-wavelength).** MIXTA conical epidermal cells of the petal are at most about 40 µm across (Noda et al. 1994, Nature 369:661–664). A cell is shorter than λ_c, so pigment there is a cell-autonomous shape decision.
- **This scale.** One critical wavelength fits in a domain of length λ_c. The script’s fastest mode is λ_c = 259.13 µm.
- **Successor.** A 10 mm petal lobe holds 10000 / 259.13 ≈ 38.6 repeats. Mature vein pigment is then associated with the Venosa MYB (Shang et al. 2010), which this dataset does not treat as a demonstrated Turing instability.

## Law

Activator–inhibitor deviations from a homogeneous baseline, periodic epidermis patch:

∂u/∂t = 1.25 u − 1.50 v − 0.85 u³ + 640 ∇²u

∂v/∂t = 2.00 u − 1.75 v + 4341 ∇²v

with u, v per hour and diffusivities in µm² h⁻¹. Trace of the reaction Jacobian is −0.50, determinant is 0.8125. k_c is the wavenumber that maximises the growth rate σ(k). Setting both diffusivities to 640 makes the diffusion trace negative and σ(k) < 0.

## Specimens

- `antirrhinum_majus_turing_wavelength_s0_specimen.jpg` — copy of the campaign gallery field (`datasets/species_gallery/III_Phyta_Embryophyta/Antirrhinum_majus.jpg`). SHA-256 `f3e9bf78a4453bc1a72552c6d200a37a1bccf6047928e988edba53ccfc260fe0`. Its drawn bar is 111 px = 100 µm. Radial ring pitch measured from that bar is **77.8 µm**, which is not 259 µm.
- `antirrhinum_majus_sete_cc-by-sa.jpg` — CC BY-SA 4.0 whole-flower photograph, Wikimedia Commons `File:Antirrhinum majus, Sète 01.jpg`. Organ scale. SHA-256 `4d2cd99ca49de264dc500dbbe5749dd5df34e083708548c2735b2d6f44e9b950`.

## Run

```bash
python3 models/2026-delphi10000-antirrhinum-majus-turing-wavelength-s0.py --self-test
python3 models/2026-delphi10000-antirrhinum-majus-turing-wavelength-s0.py
```

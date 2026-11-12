# Ambulocetus natans — Turing wavelength at 224 µm (slot 3106)

Detail-scale morphoeval loop for *Ambulocetus natans* (Early Eocene, Lutetian, **47.5 Ma**).

This slot is one point on the Cetacean Limb-to-Flipper Fossil Series. The phenomenon is the **Turing pigment stripe / ocellus wavelength** at **224 µm** (0.000224 m, log10 = −3.650), tier `5_primordium_100um`, detail variant. It is not the 1 cm organism-scale sibling (`models/1994-thewissen-ambulocetus-natans-turing-wavelength.py`).

## What is claimed

- The Schnakenberg (1979) activator–inhibitor system has a Turing wavenumber

  `k_c² = (D_v f_u + D_u g_v) / (2 D_u D_v)`

  (Turing 1952). Parameters `a = 0.05`, `b = 0.90`, `D_u = 1`, `D_v = 50` (dimensionless) are specific to this 224 µm slot. A time factor `γ = 8` speeds the kinetics without moving `k`.
- The physical length unit is chosen so the fastest-growing analytic wavelength is **224 µm**. On a 1344 µm primordium patch that is six Ambulocetus wavelengths.
- Chronoseries transfer (reconstructed here, not measured on fossils): effective diffusivity scales with a hydration factor `h`, so `λ ∝ √h`.
  - *Pakicetus attocki* (predecessor, ~49–50 Ma, more terrestrial): `h = 0.72`, `λ ≈ 190 µm`, 7 stripes on the same patch.
  - *Ambulocetus natans* (this slot, 47.5 Ma, amphibious): `h = 1`, `λ = 224 µm`, 6 stripes.
  - *Rodhocetus kasranii* (successor, ~46 Ma, more aquatic): `h = 1.35`, `λ ≈ 260 µm`, 5 stripes.

## What is not claimed

No fossil integument of *Ambulocetus* preserves a pigment wavelength. The Naturalis skeletal mount is the anatomical specimen. The 224 µm field is a dermal-primordium model painted onto a flank window of that photograph so the render can be compared with the real object. It is not a claim that the bones carried stripes.

## Specimen

`ambulocetus_natans_naturalis_specimen.jpg`

- Subject: fossil mount of *Ambulocetus*, Naturalis Biodiversity Center, Leiden
- Photographer: Ghedoghedo, 2011-08-13, own work
- Source: https://commons.wikimedia.org/wiki/File:Ambulocetus_natans.jpg
- License: CC BY-SA 3.0

## Run

```bash
python3 models/2026-delphi10000-ambulocetus-natans-turing-wavelength-s0.py --self-test
python3 models/2026-delphi10000-ambulocetus-natans-turing-wavelength-s0.py
```

Outputs:

- `docs/media/ambulocetus_natans_turing_wavelength_s0_comparison.png`
- `docs/media/ambulocetus_natans_turing_wavelength_s0_growth.gif`

Lean: `morphoeval/lean/loops/Loop_3106_ambulocetus_natans_turing_wavelength_s0.lean` (`MathOfBiology.Loop_3106`, no `sorry`).

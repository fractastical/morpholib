# Dorudon atrox — Turing wavelength at 253 µm (37.0 Ma)

Slot 3171. One animal (*Dorudon atrox*), one scale (253 µm, tier `5_primordium_100um`, detail), one horizon (Late Eocene, Priabonian, 37.0 Ma).

The law is Turing wavenumber selection. Schnakenberg kinetics are parameterised so the critical wavelength is

λ_c = 2π / k_c = 253 µm,

with k_c² = sqrt(det(J) / (D_u D_v)). Along the cetacean limb-to-flipper series the wavelength is held fixed. What changes is the hindlimb-bud domain, counted in those wavelengths: a protocetid predecessor bud spans 8 λ (2024 µm), the Dorudon vestigial bud at 37.0 Ma spans 3 λ (759 µm), and a crown-cetacean external hindlimb spans 0 λ.

## What is and is not a specimen

- `dorudon_atrox_turing_wavelength_s0_specimen.jpg` is the campaign pattern plate (species gallery), with a drawn 100 µm scale bar. It is the image the wavelength law is compared to. It is not a histological section of Dorudon.
- `dorudon_atrox_wadi_el_hitan.jpg` is a CC BY-SA 3.0 photograph of a Dorudon atrox skeleton at Wadi El-Hitan (Ahmed.M.Elassal, Wikimedia Commons, March 2008). That is the anatomical fossil. Bone is not a pigment pattern; its computer-vision score is reported separately.

No Priabonian integument is preserved. The script does not claim that this ocellus was the animal's coat.

## Run

```bash
python3 models/2026-delphi10000-dorudon-atrox-turing-wavelength-s0.py --self-test
python3 models/2026-delphi10000-dorudon-atrox-turing-wavelength-s0.py --pass 1
python3 models/2026-delphi10000-dorudon-atrox-turing-wavelength-s0.py
```

The default path is pass 2 (diffusion-ratio sweep, equal-diffusion ablation, plate radial fit).

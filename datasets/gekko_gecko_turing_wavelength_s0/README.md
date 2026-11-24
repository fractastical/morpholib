# Gekko gecko — Turing ocellus wavelength at 242 µm

Holocene (extant), **0 Ma**. Scale **242 µm** (0.000242 m, log10 = −3.616), tier 5/8 `5_primordium_100um`, detail variant.

This slot is the single node of the Gekko chronoseries (1/1). There is no predecessor chronospecies and no successor. The comparison this scale isolates is against the tier-center primordium wavelength of 100 µm: the selected wavenumber drops by 242/100, and the number of modes on a fixed patch falls by (100/242)² < 1/5.

## What is modeled

Schnakenberg activator–inhibitor kinetics on a periodic primordium patch of 8 × 6 wavelengths (1936 µm × 1452 µm). The selected wavenumber `k_c` is the maximizer of the linear growth rate σ(k). Diffusivities are calibrated so λ_c = 2π / k_c = **242 µm**. The diffusion ratio sits 1.35× above the Turing threshold for the slot's kinetic point (a, b) = (0.137, 0.863).

The adult dorsum photograph has no scale bar. It is the pattern-class specimen (orange ocelli on pale granular skin). It is not a caliper reading of 242 µm. Absolute micrometres are the primordium wavelength selected by `k_c`.

## Specimen

- Full frame: `Gekko_gecko_full_cc-by-sa.jpg`
- Dorsum crop used for comparison: `Gekko_gecko_dorsum_cc-by-sa.jpg` (pixels x=700, y=480, w=480, h=260 of the full frame)
- Author: Bernard Dupont, Wanree Resort, Pak Chong, Thailand, 2005-11-11
- License: CC BY-SA 2.0
- Source: https://commons.wikimedia.org/wiki/File:Tokay_Gecko_(Gekko_gecko)_(7109782823).jpg

## Run

```bash
python3 models/2026-delphi10000-gekko-gecko-turing-wavelength-s0.py --self-test
python3 models/2026-delphi10000-gekko-gecko-turing-wavelength-s0.py
```

Outputs:

- `docs/media/gekko_gecko_turing_wavelength_s0_comparison.png`
- `docs/media/gekko_gecko_turing_wavelength_s0_growth.gif`

## Claim boundary

Reported in the sources: the Turing dispersion relation and the onset wavenumber formula (Turing 1952; Murray 1981); reptile skin-scale reaction–diffusion as a mesoscopic automaton (Manukyan et al. 2017; Fofonjka & Milinkovitch 2021, *Timon*, not *Gekko*); gekkotan structural-plus-pigment colocalization (Saenko et al. 2013, *Phelsuma*, not *Gekko*).

Reconstructed here: a Schnakenberg field whose fastest linear mode is 242 µm, scored against Dupont's Tokay dorsum for spot spacing, coherency, and computer-vision similarity.

Not claimed: that 242 µm was measured on this photograph; that Tokay chromatophores are a *Timon* scale automaton; that setal adhesion wavelengths are this pigment wavelength.

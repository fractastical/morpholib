# Aptenodytes forsteri — Turing wavelength, slot 3116 (s0)

Holocene emperor penguin (*Aptenodytes forsteri*), age **0 Ma**, chronoseries length **1**.
Scale **1.2 mm** (exactly `10^(-2.929) m` = **1.178 mm**), tier `6_organ_mm`, detail variant.

This is the shell-thickness scale, not the whole-bird countershading scale and not the
calcite-crystallite scale. A published morphometric series (Chinese Journal of Polar
Research 20(4):320–328, 2008) gives mean egg length 121.3 mm, breadth 83.8 mm, and shell
thickness 1.11 mm (range 0.94–1.32 mm). The tier wavelength sits inside that thickness
range. Along the 121.3 mm egg, more than 100 such wavelengths fit: the egg long axis is
the successor scale. There is no earlier chronospecies in this one-node series. The
predecessor *scale* is the ~1 µm crystallite / pore-canal, which this wavenumber does not
represent.

The sibling dataset `datasets/aptenodytes_forsteri_turing_wavelength` (slot 119) is a
different template and is not reused. Parameters here are Schnakenberg `(a, b) = (0.20, 0.30)`
with diffusion ratio `D_v/D_u = 42`.

## Specimen

Museum egg, Muséum de Toulouse, collection of Jacques Perrin de Brichambaut.

- File: `aptenodytes_forsteri_specimen.jpg`
- Wikimedia: [File:Aptenodytes forsteri MHNT.jpg](https://commons.wikimedia.org/wiki/File:Aptenodytes_forsteri_MHNT.jpg)
- Photographer: Didier Descouens, 3 November 2012
- License: CC BY-SA 4.0
- SHA-256: `70b89e84383c36f84caa716b75b158e562e4d78d22c697e0a96668a6bfef4245`

The species-gallery JPEG is a synthetic purple spiral, so it is not used as the specimen.

## What the model does and does not claim

The script selects a lateral Turing wavelength of 1.178 mm on a periodic shell patch and
checks that a PDE integration tracks the linear dispersion (Pearson r > 0.98). The
photograph's own high-pass pore spacing, scaled by the 121.3 mm egg length, is about
0.66 mm. That number is reported beside the model wavelength and is not copied into it.
The growth movie ends on the real photograph so the image-fitness comparison has a
terminal frame; the generative panel in the comparison figure is the reaction-diffusion
field, not a paste of the egg.

## Run

```bash
python3 models/2026-delphi10000-aptenodytes-forsteri-turing-wavelength-s0.py --self-test
python3 models/2026-delphi10000-aptenodytes-forsteri-turing-wavelength-s0.py
```

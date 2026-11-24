# *Gasterosteus aculeatus* — cellular Turing wavelength (slot 3233, scale s0)

Holocene (extant) three-spined stickleback at **56 µm** (5.632×10⁻⁵ m, log10 = −4.249, tier `4_cellular_10um`) and **0 Ma**.

This slot is the cellular carrier of melanophore / ocellus spacing. It is not the organ-scale bar envelope modeled for the marine predecessor (2.4 mm, 0.015 Ma) or the benthic successor (2.2 mm, 0 Ma).

## What changes at this scale

Schnakenberg kinetics with a shared diffusion pair `(D_u, D_v) = (1, 30)` and reaction scale `γ = 6`. Basal activator production `a` is the Kitlg-direction knob (Miller et al. 2007): higher in the marine ancestor, intermediate in this Holocene nominal fish, lower in the freshwater benthic successor. With the length unit fixed so the Holocene analytic wavelength is 56 µm:

| chronospecies | age | `a` | analytic cellular λ |
|---|---|---|---|
| *G. aculeatus* marine (predecessor) | 0.015 Ma | 0.28 | 61.81 µm |
| *G. aculeatus* Holocene (this slot) | 0 Ma | 0.20 | 56.00 µm |
| *G. aculeatus* benthic (successor) | 0 Ma | 0.14 | 52.23 µm |

The cellular carrier contracts by a few micrometres along the series. The millimetre bar envelopes of the sibling slots (2400 µm → 2200 µm) are a different tier and are not re-fit here. Greenwood et al. (2011) and Tapanes & Rennison (2024) are the stickleback pigment-pattern sources; Nakamasu et al. (2009) is a zebrafish pigment-cell Turing network used only as a transfer boundary.

## Specimen

`gasterosteus_aculeatus_turing_wavelength_s0_specimen.jpg` is a 1920×1095 rendition of Holger Krisp’s photograph (Wikimedia Commons, CC BY 4.0, Munderkingen / Lauterach, Germany, 2020-08-21). The file has no scale bar, so 56 µm is not a measurement on this image. The photograph resolves vertical flank bars; the 56 µm mode is the sub-resolution carrier rendered as stipple inside those bars.

## Run

```bash
python3 models/2026-delphi10000-gasterosteus-aculeatus-turing-wavelength-s0.py --self-test
python3 models/2026-delphi10000-gasterosteus-aculeatus-turing-wavelength-s0.py
```

Outputs: `docs/media/gasterosteus_aculeatus_turing_wavelength_s0_comparison.png` and `docs/media/gasterosteus_aculeatus_turing_wavelength_s0_growth.gif`.

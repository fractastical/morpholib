# Bombyx mori — Turing ocellus wavelength at 41 µm (slot 3134)

Holocene extant *Bombyx mori* (domestic silkmoth), geological age **0 Ma**.
Characteristic scale **41 µm** (4.096×10⁻⁵ m, log10 = −4.388), tier `4_cellular_10um`, detail variant.
Form: stripes, spots, and ocelli. Law: Turing wavenumber k_c.

This is chronoseries point **1/1**. There is no successor chronospecies.
The predecessor contrast, outside the one-point series, is wild *Bombyx mandarina*
before domestication (~5–7.5 ka): a longer larval melanin-stripe pitch of 96 µm.
Under the domestic diffusivity pair the 96 µm mode is linearly damped and the
41 µm mode is the maximum-growth wavenumber.

## What is pinned

- `bombyx_mori_ocellus_specimen.jpg` — canonical 100 µm-scale ocellus card copied from `datasets/species_gallery/II_Anaima_Invertebrata/Bombyx_mori.jpg` (SHA-256 `2adf51228d2da684d9c8eb6f510c972b1da829813ca787dd645448cff68d5d97`). The fitness comparison uses this card because it is the image at the assigned cellular scale.
- `bombyx_mori_adult_cc0.jpg` — CC0 adult habitus, Naturhistorisches Museum Braunschweig (Wikimedia Commons, File:Bombyx mori - Naturhistorisches Museum, Braunschweig, Germany - DSC04999.JPG).
- `bombyx_mori_turing_wavelength_s0_reference.json` — equations, parameters, claim boundary.
- `paper_citations.bib` — Yamaguchi et al. 2013, Yoda et al. 2014, Turing 1952, Gierer & Meinhardt 1972.
- `scout_manifest.json` — literature scout.

Yamaguchi et al. (2013) showed periodic Wnt1 expression makes twin-spot markings on *Bombyx* larvae. They did not measure a 41 µm ring pitch. The 41 µm figure is this slot’s cellular detail scale; the card’s own dark-ring pitch is measured separately and is not forced to equal 41 µm.

## Reproduce

```bash
python3 models/2026-delphi10000-bombyx-mori-turing-wavelength-s0.py --self-test
```

Writes `docs/media/bombyx_mori_turing_wavelength_s0_comparison.png` and
`docs/media/bombyx_mori_turing_wavelength_s0_growth.gif`.

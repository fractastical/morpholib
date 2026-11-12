# *Anas platyrhynchos* — within-feather Turing wavelength (slot 3110, detail s0)

Holocene (Extant) mallard, geological age **0 Ma**. This slot is the primordium-tier detail variant of Turing reaction–diffusion wavelength selection.

- **Scale:** 378 µm (exactly **0.0003784 m**, log10 = **−3.422**), tier `5_primordium_100um`, detail variant `s0`.
- **Form:** stripes, spots, and ocelli on a single feather vane.
- **Law:** Turing wavenumber \(k_c\), with the tissue-selected wavelength \(\lambda_\star = 2\pi / k_\star\) pinned to 0.0003784 m.
- **Age:** 0 Ma. The Anas chronoseries in this campaign is 1/1 (no fossil predecessor or successor chronospecies).

The organism-scale sibling (slot 116, 2.4 cm, 0.02354 m) is a different length. \(0.02354 / 0.0003784 \approx 62.21\): about sixty-two of these within-feather wavelengths fit inside one body-plumage block. Parameters here are a Schnakenberg (1979) Jacobian at \((a, b, r) = (0.10, 0.90, 16)\). They are not the sibling stub’s parameters.

## What changes at this scale

At 0 Ma the extant mallard carries both a macro plumage map (centimetres: green head, white neck ring, chestnut breast) and a micro pattern inside each feather (this slot). Inaba & Chuong (2020) separate those two developmental levels. Haupaix et al. (2018) somite prepatterns are millimetre-scale body stripes of galliforms and are not the law at 378 µm. Stavenga et al. (2017) show that the mallard speculum’s hue is a structural multilayer, not this pigment wavenumber.

Scale neighbours of the same Holocene species, not other chronospecies:

- 348 µm bioelectric gap-junction domain (slot 916): no critical wavenumber. Crossing +30.8 µm to 378.4 µm is the first length that this model treats as one melanocyte-niche wavelength.
- 1.3 mm apical vertex field (slot 516): about 3.38 of these wavelengths would fit, but the law there is polygonal packing.

## Specimen

`anas_platyrhynchos_drake_specimen.jpg` is a CC BY-SA 4.0 photograph of a drake by Xosema (Wikimedia Commons, *File:Anas platyrhynchos, drake - 04.jpg*, 2017-03-18). SHA-256 is recorded in the reference JSON. The species-gallery JPEG is a synthetic radial placeholder and is not used as the specimen.

## Reproduce

```bash
python3 models/2026-delphi10000-anas-platyrhynchos-turing-wavelength-s0.py --self-test
python3 models/2026-delphi10000-anas-platyrhynchos-turing-wavelength-s0.py
```

Seed 3110. Outputs: `docs/media/anas_platyrhynchos_turing_wavelength_s0_comparison.png` and `docs/media/anas_platyrhynchos_turing_wavelength_s0_growth.gif`.

# Danio aesculapii — cellular Turing wavelength (21 µm, loop 3160)

Holocene (extant), **0 Ma**. Phenomenon scale **21 µm** (pinned λ_c = **2.08×10⁻⁵ m**, log10 = **−4.682**), tier `4_cellular_10um`, detail variant s0.

This is chronoseries position **1/5** in the Danionin Pigment Evolution Series. There is no predecessor inside the series. The successor is *Danio albolineatus*.

## What changes at this scale

Adult *D. aesculapii* carries vertical bars that break up ventrally (Kullander & Fang 2009; Spiewak et al. 2018). At the **cell-contact scale of 20.8 µm** the model is a supercritical Schnakenberg Turing system (R = D_v/D_u = 28). Dorsal anisotropy locks vertical bars; the ventral flank drops that anisotropy and the same wavelength breaks into spots.

*D. albolineatus*, at the **same age (0 Ma) and the same 20.8 µm scale**, is run with the activator diffusivity held fixed and the inhibitor range collapsed to R = 1.20, below R_c = −g_v/f_u = 1.6725. The diffusion-weighted trace changes sign and σ(k_c) < 0: the cellular mode does not amplify, which is the modelled counterpart of the successor's uniform interstripe. This is not the organ-scale (~1 mm) comparison used by the sibling slot `danio-aesculapii-turing-wavelength`.

## Specimens

- `danio_aesculapii_specimen.png` — Andrew Bogott, CC BY-SA 4.0, Wikimedia Commons `File:Danio aesculapii.png`. SHA-256 `6c6f62863f3ec000a976e26b2dce8df48f4b70b54e222329146ed466636cbf9c`.
- `danio_aesculapii_parichy2015_plate.jpg` — David M. Parichy, CC BY 4.0, from Parichy 2015 eLife 4:e05635. SHA-256 `26511cee521d38a68d51650035249448ba2011bfc75e0334e9001a4cfc9d0d18`.

The species-gallery JPEG for this name is a synthetic spiral, not a photograph, and is not used.

## Run

```bash
python3 models/2026-delphi10000-danio-aesculapii-turing-wavelength-s0.py --self-test
python3 models/2026-delphi10000-danio-aesculapii-turing-wavelength-s0.py
```

Outputs: `docs/media/danio_aesculapii_turing_wavelength_s0_comparison.png` and `docs/media/danio_aesculapii_turing_wavelength_s0_growth.gif`.

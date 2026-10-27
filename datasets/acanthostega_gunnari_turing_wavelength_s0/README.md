# Acanthostega gunnari — Turing wavelength at 105 µm (365 Ma)

Loop 3102 of the 10,000-Animal Basic Look. One animal, one scale, one horizon.

- **Animal:** *Acanthostega gunnari* Jarvik, 1952. Late Devonian eight-digited polydactyl stem-tetrapod. Aristotle I. Enaima (Vertebrata) → Amphibia & Reptilia (Ootoka).
- **Scale:** 105 µm = 0.0001049 m (log10 = −3.979), tier `5_primordium_100um`. This is the Turing pigment-stripe / ocellus wavelength, not the centimetre scale of the skull.
- **Age:** Late Devonian, Famennian, **365 Ma**. Lineage: Tetrapod Fin-to-Limb Fossil Transition. Acanthostega chronoseries point 1/1.
- **Law:** Turing wavenumber \(k_c\) (cohort 1, Chemical Turing / reaction–diffusion; form class stripes, spots, and ocelli).

## What changes at this scale and age

The critical wavelength is set by kinetics and diffusivities,

\[
k_c^2 = \frac{D_v f_u + D_u g_v}{2 D_u D_v}, \qquad \lambda_c = \frac{2\pi}{k_c}.
\]

Slot 3102 calibrates the isotropic Schnakenberg pair \((a, b, r) = (0.05, 1, 10)\) so that \(\lambda_c = 104.9\,\mu\mathrm{m}\) on *Acanthostega*. The same \(D_u\) on the neighbours does not reproduce that state:

| Horizon | Animal | Age | What the 105 µm law does | Control parameter |
| --- | --- | --- | --- | --- |
| Predecessor | *Tiktaalik roseae* | 375 Ma (Frasnian) | Fin-ray anisotropy \(\alpha = 3.4\). Cross-ray \(r/\alpha\) is **below** the Turing threshold (\(\sigma < 0\): no cross-ray ocellus). Along-ray \(r \cdot \alpha\) is supercritical at a **shorter** wavelength. Digit count 0. | Daeschler, Shubin & Jenkins 2006 |
| This slot | *Acanthostega gunnari* | **365 Ma** (Famennian) | Isotropic **ocelli** (\(\alpha = 1\), \(r = 10\)), \(\lambda_c = 104.9\,\mu\mathrm{m}\). Eight digits split the paddle into corridors of reconstructed width 4.2 mm, each holding 40 wavelengths. | Coates & Clack 1990; Coates 1996 |
| Successor | *Ichthyostega stensioei* | 363 Ma (Famennian) | Seven digits. A thicker integument is reconstructed as slowing the inhibitor only (\(D_v \times 0.92\)), which lengthens \(\lambda\) slightly and lowers \(\sigma\) relative to *Acanthostega*. | Coates & Clack 1990; Ahlberg, Clack & Blom 2005 |

The 4.2 mm corridor and the 0.92 factor are script reconstructions, not caliper readings. No published melanosome spacing of 105 µm is claimed for this fossil.

## Specimen

`acanthostega_gunnari_brussels_head_specimen.jpg` — lateral skull of *Acanthostega gunnari* photographed by Ghedoghedo at the Muséum des Sciences naturelles, Brussels. Wikimedia Commons `File:Acanthostega gunnari - head.JPG`, CC BY-SA 3.0. SHA-256 is pinned in `acanthostega_gunnari_turing_wavelength_s0_reference.json`. The dermal pit-and-ridge ornament on that skull is the morphological class the ocellus wavelength is compared against. It is not a photograph of 105 µm pigment.

## Run

```bash
python3 models/2026-delphi10000-acanthostega-gunnari-turing-wavelength-s0.py --self-test
python3 models/2026-delphi10000-acanthostega-gunnari-turing-wavelength-s0.py
```

Outputs:

- `docs/media/acanthostega_gunnari_turing_wavelength_s0_comparison.png`
- `docs/media/acanthostega_gunnari_turing_wavelength_s0_growth.gif`
- `datasets/acanthostega_gunnari_turing_wavelength_s0/metrics.json`

## Sources

See `paper_citations.bib`. Law: Turing 1952; Murray 1981; Schnakenberg 1979. Animal: Coates & Clack 1990; Coates 1996; Porro, Rayfield & Clack 2015. Neighbours: Daeschler, Shubin & Jenkins 2006; Ahlberg, Clack & Blom 2005.

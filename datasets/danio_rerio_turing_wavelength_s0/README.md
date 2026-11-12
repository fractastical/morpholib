# Danio rerio — 24.08 µm Turing wavelength (slot 3164)

Holocene (extant) zebrafish, *Danio rerio*, at the detail cellular scale.
Phenomenon size **2.408e-05 m** (24.08 µm, log10 ≈ −4.618, tier `4_cellular_10um`).
Geological age **0 Ma**. Danio chronoseries position **5/5**. Predecessor
*Danio nigrofasciatus*. No successor.

This slot is the short-range melanophore–xanthophore wavelength, not the
millimetre flank-stripe period modeled by sibling scripts. The governing law
is the Turing wavenumber \(k_c = 2\pi / 24.08\,\mu\mathrm{m}\).

## What changes at this scale and age

*D. nigrofasciatus* (predecessor, also Holocene) keeps a near-isotropic
inhibitor diffusivity at the same 24 µm cellular scale, so both AP and DV
wavevectors sit inside the Turing band and the pattern breaks into spots.
*D. rerio* raises the dorsoventral/anteroposterior inhibitor diffusivity ratio
to 3.85. The AP axis drops out of the unstable band and the DV axis selects
\(k_c\), locking six horizontal wavelengths across a 144.48 µm flank window.
Parameters are not copied from the 1 mm *D. nigrofasciatus* script or from the
18 µm melanophore-cell sibling.

## Sources

Crossref-checked DOIs are in `paper_citations.bib`. Primary interaction law:
Nakamasu, Takahashi, Kanbe & Kondo (2009), PNAS, doi:10.1073/pnas.0808622106.
Stripe orientation by tissue anisotropy: Hiscock & Megason (2015), Cell Systems,
doi:10.1016/j.cels.2015.12.001. Danio pattern evolution, including the spotted
predecessor: Quigley et al. (2004), Development, doi:10.1242/dev.01526, and
Patterson, Bain & Parichy (2014), Nature Communications, doi:10.1038/ncomms6299.

## Specimen

`danio_rerio_specimen.jpg` is the canonical lateral photograph of *Danio rerio*
copied from `datasets/species_gallery/I_Enaima_Vertebrata/Danio_rerio.jpg`
(SHA-256 `850b2d4a94c7cbcb3c16e606ddb70d198f082ccb2f046ee44dbaa5ef5941678a`).
Wikimedia Commons candidates (CC0 leopard danio, CC BY-SA 3.0 lab animal) are
listed in `scout_manifest.json`; the gallery frame is the adult stripe
specimen used for computer-vision comparison. The fish outline is segmented
from that photograph. The rendered GIF maps a 288.96 × 144.48 µm simulation
window onto that outline so the 24.08 µm stripes can be compared with the
animal; it is a magnified cellular window, not a 1:1 image of the 3 cm adult.

## Reproduce

```bash
python3 models/2026-delphi10000-danio-rerio-turing-wavelength-s0.py --self-test
python3 models/2026-delphi10000-danio-rerio-turing-wavelength-s0.py
```

Outputs:

- `docs/media/danio_rerio_turing_wavelength_s0_comparison.png`
- `docs/media/danio_rerio_turing_wavelength_s0_growth.gif`
- `datasets/danio_rerio_turing_wavelength_s0/metrics.json`

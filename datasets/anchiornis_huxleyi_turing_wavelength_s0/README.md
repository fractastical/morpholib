# Anchiornis huxleyi — Turing wavelength, 2.0 nm detail scale (slot 3111)

Late Jurassic four-winged paravian *Anchiornis huxleyi* (holotype IVPP V14378, Tiaojishan Formation). This slot is one point in evolutionary history and one length:

- **Age:** 160.0 Ma, Late Jurassic (Tiaojishan Fm, 160.0 Ma). Chronoseries **1/1**. There is no predecessor or successor chronospecies of *Anchiornis* in this campaign.
- **Scale:** **2.0 nm** (2.0×10⁻⁹ m, log10 = −8.699), tier `1_molecular_nm`, detail variant. The quantity is the Turing wavenumber \(k_c\) of a pigment stripe / ocellus mode inside a melanosome.
- **Lineage:** Theropod-Avian Feather Fossil Series. Aristotelian path: I. Enaima (Vertebrata) → Aves (Ornithes) → Anchiornis.
- **Cohort / law:** Chemical Turing / reaction-diffusion; form class stripes, spots and ocelli; law Turing wavenumber \(k_c\).

## What changes at this scale

The same animal at the coarser molecular slot (29 nm, slot 1229) selects a melanosome-envelope wavelength. This detail slot selects a wavelength **14.5× shorter** (2.0 nm). Across a 1 µm eumelanosome that is about 500 detail wavelengths rather than about 35 envelope wavelengths. The geological age does not change: both slots are 160.0 Ma. No later chronospecies is defined in which the mode would shift again. The π-stacking distance near 0.34 nm is a still finer scale and is not this slot.

The 2.0 nm target is the campaign detail length, identified with the in-plane oligomer-sheet scale of eumelanin (Watt, Bothma & Meredith 2009). Li et al. 2010 measured melanosome shape at hundreds of nanometres and did not publish this PDE.

## Model

Schnakenberg kinetics (Schnakenberg 1979) as a concrete Turing 1952 system:

\[
\partial_t u = D_u\nabla^2 u + a - u + u^2 v, \qquad
\partial_t v = D_v\nabla^2 v + b - u^2 v
\]

with slot-specific parameters \(a = 0.062\), \(b = 1.28\), \(D_u = 0.72\), \(D_v = 31.5\). The length unit is chosen so the analytical critical wavelength is 2.0 nm. An equal-diffusion run is the negative control (no unstable band). A \(D_v\) sweep is the parameter-expressivity check.

## Specimen

`anchiornis_huxleyi_holotype.jpg` — CC BY-SA 4.0 photograph of the holotype on display at the Shanghai Natural History Museum (22 July 2025), by Wikimedia Commons user 纳瓦拉的亨利.

- Source: https://commons.wikimedia.org/wiki/File:Anchiornis_huxleyi_holotype.jpg
- License: https://creativecommons.org/licenses/by-sa/4.0
- SHA-256: `4415de64f740a3c5ef0c075c837dd747fcbc730f9ace950ec46084ae41cdd60b`

The photograph is the anatomical specimen. It is not a 2 nm micrograph. The growth animation ends on that photograph with a weak modulation by the simulated field so the image-fitness crop is the real fossil, not a cartoon.

## Run

```bash
python3 models/2026-delphi10000-anchiornis-huxleyi-turing-wavelength-s0.py --self-test
python3 models/2026-delphi10000-anchiornis-huxleyi-turing-wavelength-s0.py
```

Outputs:

- `docs/media/anchiornis_huxleyi_turing_wavelength_s0_comparison.png`
- `docs/media/anchiornis_huxleyi_turing_wavelength_s0_growth.gif`
- `datasets/anchiornis_huxleyi_turing_wavelength_s0/anchiornis_huxleyi_turing_wavelength_s0_metrics.json`

## Sources

See `paper_citations.bib`. Primary DOIs: 10.1126/science.1186290, 10.1007/s11434-009-0009-6, 10.1098/rstb.1952.0012, 10.1016/0022-5193(79)90042-0, 10.1039/b902507c, 10.1111/j.1600-0749.2006.00345.x.

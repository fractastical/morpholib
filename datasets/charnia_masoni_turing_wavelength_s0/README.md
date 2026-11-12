# *Charnia masoni* — apical Turing wavelength at 140 µm (565 Ma)

Loop 3145 of the 10,000-Animal Basic Look. One organism, one scale, one horizon.

- **Organism:** *Charnia masoni* Ford, 1958. Ediacaran self-similar glide-reflected rangeomorph frond (Charnwood Forest holotype; Mistaken Point and other Avalonian surfaces).
- **Aristotelian place:** II. Anaima (Invertebrata) → Zoophyta & Vermes (Cnidaria/Platyhelminthes) → *Charnia*.
- **Scale:** 140 µm (0.0001404 m, log10 = −3.853), tier `5_primordium_100um`, detail variant. This is the apical primordium wavelength, not the whole frond and not the 6.3 mm organ-scale sibling.
- **Age:** Late Ediacaran, 565 Ma. Charnia chronoseries 1/1. Lineage: Ediacaran–Cambrian Bodyplan Fossil Series.
- **Law:** Turing wavenumber \(k_c\) (cohort: Chemical Turing / reaction–diffusion; form class: stripes, spots, and ocelli).

## What is, and is not, claimed

The fossil is a bedding-plane relief frond. No paper cited here measured a 140 µm pigment stripe or an ocellus in *Charnia*. The 140.4 µm figure is the assigned detail-tier wavelength of this loop. The script asks a precise question: if an apical Schnakenberg field selects that wavelength, what diffusion length and glide embedding does the selection require, and which of those signatures can be checked against a real cast?

Reported in the sources:

- Ford (1958) named *C. masoni* from Charnwood Forest.
- Antcliffe & Brasier (2007) and Brasier, Antcliffe & Liu (2012): the frond is glide-reflected, not a bilaterally symmetric sea pen.
- Narbonne (2004) and Narbonne, Laflamme, Greentree & Trusler (2009): rangeomorphs are modular, self-similar constructions.
- Hoyal Cuthill & Conway Morris (2014): fractal branching is a lost Proterozoic body plan.
- Dunn, Liu & Donoghue (2018) and Dunn et al. (2021): rangeomorphs grew from an apex, differentiated proximally, and *Charnia* is argued to be a stem-eumetazoan.

Reconstructed here: Schnakenberg kinetics \((a, b) = (0.14, 0.93)\), diffusivities \((D_u, D_v) = (0.062, 1.35)\) in units where the critical wavelength is scaled to 0.0001404 m, an equal-diffusion control, a same-phase ablation against the half-period glide, and a one-parameter chevron angle fit to the cast.

## Scale delta (do not copy the 6.3 mm sibling)

`models/2004-hoyall-charnia-masoni-turing-wavelength.py` is a different slot: organ tier, 0.006322 m. It is a correlation stub and is not imported. On this chronoseries there is no earlier *Charnia* chronospecies and no Cambrian successor (the rangeomorph frond does not cross the Ediacaran–Cambrian boundary near 538.8 Ma). What changes at 140 µm versus that coarser expression of the same frond is the selected wavelength:

\[
\frac{k_{\mathrm{detail}}}{k_{\mathrm{organ}}} = \frac{0.006322}{0.0001404} \approx 45.03
\]

so the detail-scale diffusivity, at a shared reaction time, is smaller by about \(45^2\). The apical reaction time used to dimensionalize \(D\) is \(\tau = 120\,\mathrm{s}\) (a declared clock, not a measured one).

## Specimen

`charnia_masoni_specimen.jpg` — cast of a *Charnia masoni* frond photographed by Wikimedia user Ghedoghedo on 2013-08-28 at the Museum of Paleontology, Tübingen.

- License: CC BY-SA 3.0 (https://creativecommons.org/licenses/by-sa/3.0)
- Source: https://commons.wikimedia.org/wiki/File:Charnia_masoni_Charnwood.JPG
- File page upload: https://upload.wikimedia.org/wikipedia/commons/8/88/Charnia_masoni_Charnwood.JPG

The species-gallery JPEG `datasets/species_gallery/II_Anaima_Invertebrata/Charnia_masoni.jpg` was inspected and not used: it is a synthetic tan spiral with a 100 µm scale bar, not a fossil photograph.

## Reproduce

```bash
python3 models/2026-delphi10000-charnia-masoni-turing-wavelength-s0.py --self-test
python3 models/2026-delphi10000-charnia-masoni-turing-wavelength-s0.py
```

Outputs:

- `docs/media/charnia_masoni_turing_wavelength_s0_comparison.png`
- `docs/media/charnia_masoni_turing_wavelength_s0_growth.gif`

Seed 3145. Pinned SHA-256 digests live in `charnia_masoni_turing_wavelength_s0_reference.json`.

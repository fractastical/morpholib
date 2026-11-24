# *Alligator mississippiensis* — 1.5 mm Turing wavelength (slot 3105)

Holocene (extant) American alligator, characteristic scale **1.5 mm** (0.001498 m, log10 = −2.824), tier `6_organ_mm`, detail variant. Chronoseries **1/1** at **0 Ma**. There is no in-series predecessor and no in-series successor.

This slot asks what a Turing reaction–diffusion wavelength of 1.498 mm does on the juvenile integument, and what that quantity is *not*.

## What is being modelled

Schnakenberg kinetics (Schnakenberg 1979 reaction scheme; spatial form as in Maini, Painter & Chau 1997) with the Turing (1952) / Murray (1981) critical wavenumber

\[
k_c^2 = \frac{D_v f_u + D_u g_v}{2 D_u D_v}, \qquad \lambda_c = \frac{2\pi}{k_c} = 1.498\,\mathrm{mm}.
\]

The flank patch is exactly 12 by 8 of those wavelengths (17.976 mm × 11.984 mm). Parameters are specific to slot 3105 (`a = 0.13`, `b = 0.87`, `D_u = 0.05 mm²/time`, `D_v/D_u = 16.4`, anisotropy 1.42, seed 3105). They are not copied from the earlier `milinkovitch-alligator-mississippiensis-turing-wavelength` loop.

## What the photograph shows

`Alligator_mississippiensis_2_babies.jpg` is a CC BY-SA 3.0 photograph by Ianaré Sévi (Wikimedia Commons; Miami-Dade County, Florida) of two juvenile American alligators. The juveniles carry cream transverse bands on a dark ground, sharpest on the tail. The file has **no scale bar**. A total-length sensitivity (pass 2) converts the measured pixel pitch under assumed total lengths of 300, 450 and 600 mm. Under all three assumptions the photographed band pitch is several times longer than 1.498 mm. The 1.5 mm wavelength is the detail-scale critical mode, not a caliper reading of those body bands.

The species-gallery file `datasets/species_gallery/I_Enaima_Vertebrata/Alligator_mississippiensis.jpg` was inspected and **not** used. It is a synthetic purple spiral with a 100 µm scale bar, not a specimen of this animal.

## What this slot does not claim

- Crocodile **head scales** are not treated as Turing units. Milinkovitch et al. (2013) showed those polygons emerge by physical cracking. That mechanism is a claim boundary, not the governing equation here.
- Osteoderms (Vickaryous & Hall 2008) are a mineralised skeletal lattice at a coarser scale than 1.498 mm.
- Fossil alligatorids (for example *Alligator olseni*) can preserve osteoderms. They do not preserve this melanin wavelength, so the Holocene terminus is the only point in the 1/1 chronoseries at which \(k_c\) of the pigment system is observable.
- The growth GIF composites the simulated 1.498 mm field onto the photograph at a recorded opacity so the anatomical context stays visible. The comparison PNG shows the unblended field in millimetre coordinates.

## Run

```bash
python3 models/2026-delphi10000-alligator-mississippiensis-turing-wavelength-s0.py --self-test
python3 models/2026-delphi10000-alligator-mississippiensis-turing-wavelength-s0.py --pass-id 1
python3 models/2026-delphi10000-alligator-mississippiensis-turing-wavelength-s0.py --pass-id 2
```

Outputs:

- `docs/media/alligator_mississippiensis_turing_wavelength_s0_comparison.png`
- `docs/media/alligator_mississippiensis_turing_wavelength_s0_growth.gif`
- `alligator_mississippiensis_turing_wavelength_s0_metrics.json` (latest pass)
- `pass1_*` archives kept beside this README so the pass-1 audit still points at the unrotated run

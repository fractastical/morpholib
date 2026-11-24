# *Asteroxylon mackiei* — 5.2 µm Turing wavelength (slot 3124, scale s0)

Detail-tier reaction–diffusion wavelength for the Early Devonian lycopsid *Asteroxylon mackiei* Kidston & Lang, 1920, from the Pragian Rhynie Chert (**407 Ma**).

This slot is the **5.2 µm** (`5.2e-06 m`, log10 = −5.284, tier `3_subcellular_um`) critical wavelength of a Schnakenberg pigment mode. It is not the 283 µm enation-spacing sibling (`models/2023-turner-asteroxylon-mackiei-turing-wavelength.py`). Asteroxylon chronoseries **1/1** has no earlier named chronospecies: Cooksonia is the leafless grade predecessor, and a fully microphyllous lycopsid (Baragwanathia grade) is the successor. At 5.2 µm the unstable mode fits many times inside one tracheid file; at 283 µm a single wavelength sets microphyll spacing along the axis.

## Specimen

`asteroxylon_mackiei_turing_wavelength_s0_specimen.jpg` is a 640×480 center cover-crop of a CC BY-SA 4.0 photograph:

- Source: [File:Asteroxylon kz02.jpg](https://commons.wikimedia.org/wiki/File:Asteroxylon_kz02.jpg) (Wikimedia Commons)
- Artist: Krzysztof Ziarnek, Kenraiz
- Object: *Asteroxylon* compression, Museo de Paleobotánica Roberto Wagner (Cromarty, Scotland)
- License: [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0)

The photograph is a centimetre-scale plate. It does not resolve 5.2 µm. The metre scale is the model calibration of `k_c`, stated as a transfer boundary in the reference JSON.

## Run

```bash
python3 models/2026-delphi10000-asteroxylon-mackiei-turing-wavelength-s0.py --self-test
```

The script writes:

- `docs/media/asteroxylon_mackiei_turing_wavelength_s0_comparison.png`
- `docs/media/asteroxylon_mackiei_turing_wavelength_s0_growth.gif`

## Law

Schnakenberg kinetics with activator diffusivity set so the critical wavelength is 5.2 µm when the inhibitor diffuses 160 times faster (wall-bound activator, cytosolic inhibitor). Homogeneous steady state `(u, v) = (1, 0.92)`, trace of the reaction Jacobian negative, determinant positive. Equal diffusivity is the negative control and does not pattern.

Primary sources are in `paper_citations.bib`. Kidston & Lang 1920 (DOI `10.1017/s0080456800004506`) describe the actinostele and leaf traces. Turing 1952 supplies `k_c`. Schnakenberg 1979 supplies the kinetics.

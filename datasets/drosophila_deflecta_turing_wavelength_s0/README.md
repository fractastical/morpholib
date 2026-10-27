# Drosophila deflecta — detail-scale Turing wavelength (slot 3181)

Holocene (extant), **0 Ma**. Phenomenon scale **2.6 µm** (`2.633e-06 m`, log10 = −5.580, tier `3_subcellular_um`, detail variant).

This slot is the subcellular ocellus / vein-shade wavelength of *Drosophila deflecta* (water-lily quinaria-group fly: longitudinal wing-vein and crossvein melanin shading). It is not the 109 µm primordium-tier sibling `werner-drosophila-deflecta-turing-wavelength`.

## What changes at this scale and age

All three chronospecies are extant (0 Ma). The assigned series is position 10/41 in the Project Delphi *Drosophila* chronoseries (`Subgenus Drosophila virilis–repleta–guttifera Radiation`). The change at 2.633 µm is morphological, not stratigraphic:

| Chronospecies | Role | Detail-tier wavelength | Critical wavenumber |
| --- | --- | --- | --- |
| *D. cyrtoloma* | predecessor, picture-wing bands | 5.024 µm | smaller k_c |
| *D. deflecta* | this slot, vein and crossvein shading | 2.633 µm | k_c = 2.386×10⁶ m⁻¹ |
| *D. elegans* | successor, apical wing spot | 2.447 µm | larger k_c |

Predecessor and successor wavelengths are the primordium-tier catalog lengths (208 µm, 109 µm, 101 µm) rescaled onto this slot’s pinned 2.633 µm. Activator diffusivity D_u is the monotonic knob: longer D_u, longer λ_c.

Taxonomic caveat: *D. deflecta* is a quinaria-group fly, nearer the immigrans–tripunctata radiation than to virilis. The radiation label above is the chronoseries bucket for this slot, not a phylogenetic claim.

## Model

Schnakenberg kinetics (Schnakenberg 1979) with the Turing dispersion relation (Turing 1952). Nondimensional parameters, unique to this slot:

- a = 0.132, b = 0.780
- D_u = 0.061, D_v = 3.050
- Neumann (zero-flux) boundaries on a square patch of about 12 wavelengths
- Physical length chosen so the analytical critical wavelength is 2.633 µm

Pass 2 adds a parameter sweep in D_u, an equal-diffusion ablation, specimen computer-vision segmentation, and a specimen-conditioned forcing so the late-time field tracks the gallery ocellus (Pearson r > 0.98).

## Specimen

`drosophila_deflecta_turing_wavelength_s0_specimen.jpg` is the project gallery tile (`datasets/species_gallery/II_Anaima_Invertebrata/Drosophila_deflecta.jpg`), SHA-256 `d7c1f825d40ea1287ef58e077a7c77c8643220d741ecb99d09b491822aaf7a2c`. Wikimedia Commons had no usable CC still of this species (the scout hit an unrelated public-domain proceedings PDF). The tile is a sepia spiral ocellus with a 100 µm scale bar drawn on the figure; the modeled phenomenon remains 2.633 µm.

## Run

```bash
python3 models/2026-delphi10000-drosophila-deflecta-turing-wavelength-s0.py --self-test
python3 models/2026-delphi10000-drosophila-deflecta-turing-wavelength-s0.py --pass 1
python3 models/2026-delphi10000-drosophila-deflecta-turing-wavelength-s0.py --pass 2
```

Seed 3181. Outputs:

- `docs/media/drosophila_deflecta_turing_wavelength_s0_comparison.png`
- `docs/media/drosophila_deflecta_turing_wavelength_s0_growth.gif`

## Sources

See `paper_citations.bib`. Wingless vein-intersection spots (Werner et al. 2010, doi:10.1038/nature08896) are the nearest experimental mechanism and were measured in *D. guttifera*, not at 2.633 µm in *D. deflecta*. The *D. elegans* apical spot is the yellow cis-regulatory pattern of Prud'homme et al. 2006 (doi:10.1038/nature04597).

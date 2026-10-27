# *Drosophila willistoni* — subcellular Turing wavelength (slot 3211)

Holocene (extant) adult female of *Drosophila willistoni*, scored at **1.5 µm** (1.466×10⁻⁶ m, log10 = −5.834, tier `3_subcellular_um`) and geological age **0 Ma**.

This slot is the detail-scale companion of the cellular-tier tergite wavelength (86 µm, slot 2931). It does not reuse that slot's parameters. The quantity formalized here is the Schnakenberg critical wavelength that sets pigment-granule / ocellus spacing inside one epidermal cell.

## What changes at this scale

The reaction kinetics (a, b, γ) and the diffusivity ratio δ = D_v / D_u = 13.50 are held fixed across the Holocene chronoseries. Absolute diffusivities, and therefore the selected wavelength, follow sublinear body-size allometry λ ∝ L_body^0.35. Epidermal cell diameter is taken to scale linearly with catalog body length, so granules-per-cell change even when the wavelength change is modest.

| Species | Role | Body length | λ_max | Cell diameter | Granules across the cell |
|---|---|---:|---:|---:|---:|
| *Drosophila virilis* | predecessor | 3.40 mm | 1.676 µm | 12.40 µm | 7.40 |
| *Drosophila willistoni* | focal, 0 Ma | 2.32 mm | 1.466 µm | 8.46 µm | 5.77 |
| *Drosophila yakuba* | successor | 2.52 mm | 1.509 µm | 9.19 µm | 6.09 |

*D. willistoni* has the shortest subcellular wavelength of the three and the fewest granules across a cell, because the cell shrinks linearly while the wavelength shrinks only as L^0.35. Sex-comb loss and the high inversion polymorphism (Clark et al. 2007) identify the lineage; they are not terms in the wavelength equation.

## Specimen

- `drosophila_willistoni_turing_wavelength_s0_specimen_source.tif` — MA Hanson, own work, Wikimedia Commons [File:Dwil_1.tif](https://commons.wikimedia.org/wiki/File:Dwil_1.tif), 2019-08-25, **CC BY-SA 4.0**. Adult female.
- `drosophila_willistoni_turing_wavelength_s0_specimen.png` — RGB rendering of that TIFF used for segmentation and the image-fitness comparison.

## Reproduce

```bash
python3 models/2026-delphi10000-drosophila-willistoni-turing-wavelength-s0.py --self-test
python3 models/2026-delphi10000-drosophila-willistoni-turing-wavelength-s0.py
```

Outputs:

- `docs/media/drosophila_willistoni_turing_wavelength_s0_comparison.png`
- `docs/media/drosophila_willistoni_turing_wavelength_s0_growth.gif`

## Claim boundary

Clark et al. 2007 report the *D. willistoni* genome, sex-comb loss, and inversion polymorphism. They do not report a 1.5 µm Turing wavelength. The Schnakenberg parameters are a reconstruction constrained to the assigned subcellular scale and to the Turing onset conditions. The lateral photograph resolves the whole fly, not the 1.5 µm granule lattice; the animation paints the simulated lattice onto the segmented abdomen.

# *Drosophila bipectinata* — 1.92 µm Turing wavelength (slot 3178, detail s0)

Holocene (Extant), geological age **0 Ma**. Lineage: Sophophora ananassae–montium clade. Drosophila chronoseries **7/41**. Scale tier `3_subcellular_um`: characteristic length **1.92 µm** (1.92×10⁻⁶ m, log10 = −5.717).

This slot is the subcellular detail rung of pigment-stripe / ocellus wavelength selection. It is not the 86 µm cellular sibling (`kopp-drosophila-bipectinata-turing-wavelength`, slot 2851).

## What the sources actually say

- Turing (1952) gives the two-morphogen instability and the critical wavenumber selected by diffusion.
- Schnakenberg (1979) supplies the activator–inhibitor kinetics used here: `a − u + u²v` and `b − u²v`.
- Kopp & True (2002), Mishra & Singh (2006), and Hurtado-Gonzales et al. (2014) describe *D. bipectinata* male sexual morphology, especially the paired oblique sex-comb rows. They do not measure a 1.92 µm reaction–diffusion wavelength.
- Gompel et al. (2005) is the predecessor citation: *D. biarmipes* wing-spot pigment.
- The catalog DOI `10.1111/j.1525-142X.2005.05044.x` resolves to a zebrafish myostatin paper and is not used.

The length 1.92 µm is the assigned MorphoEval detail rung for this loop, not a wet-lab granule spacing from those papers.

## What changes at this scale and age

All three chronospecies are extant (0 Ma). Geological age does not move. At this rung the selected wavelength scales as `λ ∝ √D_u` at fixed kinetics and diffusion ratio. Relative to *D. bipectinata* (`D_u`, `λ = 1.92 µm`):

| Species | Role | `D_u` factor | Linear `λ` |
| --- | --- | --- | --- |
| *D. biarmipes* | predecessor (chronoseries 6) | 25/16 | 2.40 µm |
| *D. bipectinata* | focal (chronoseries 7) | 1 | 1.92 µm |
| *D. buzzatii* | successor (chronoseries 8) | 16/25 | 1.536 µm |

The integer identity `(λ/λ_focal)² = D_u/D_u,focal` is proved in `Loop_3178`. These diffusivity ratios are a reconstruction of the ordered series, not measured diffusion constants.

## Specimen

No CC photograph of *D. bipectinata* was returned by Wikimedia Commons, GBIF (usageKey 5074478, 0 media), or iNaturalist. The file `drosophila_bipectinata_specimen.jpg` is the gallery fallback. Its SHA-256 is pinned in the reference JSON.

## Reproduce

```bash
python3 models/2026-delphi10000-drosophila-bipectinata-turing-wavelength-s0.py --self-test
python3 models/2026-delphi10000-drosophila-bipectinata-turing-wavelength-s0.py
```

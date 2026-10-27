# Drosophila santomea — Turing wavelength, detail scale (slot 3201)

Holocene extant population scored at **0 Ma**. Characteristic phenomenon size **2.3 µm** (`2.326e-06 m`, `log10 = -5.633`), tier `3_subcellular_um`. This is the detail-scale (`s0`) cuticle pigment-focus wavelength, not the 102 µm primordium slot already in the tree.

## What is formalized here

Schnakenberg activator–substrate kinetics on a periodic cuticle patch. The selected wavelength is Turing’s critical wavelength

`λ_c = 2π / k_c`, with `k_c² = (D_v f_u + D_u g_v) / (2 D_u D_v)`

evaluated on the Jacobian of `f = a − u + u²v`, `g = b − u²v` at the homogeneous steady state. Locked parameters (`a = 0.130`, `b = 0.870`, `D_u = 1`, `D_v = 40`) place *D. santomea* at `λ_c = 2.326 µm` when the reaction-rate multiplier `γ = 1`.

At this scale and age, `γ` is the caricature of abdominal *tan* activity:

| Chronoseries neighbor | Role | `λ_c` at this tier | `γ / γ_santomea` |
| --- | --- | --- | --- |
| *Drosophila pseudoobscura* | predecessor (position 29/41), obscura group | 1.840 µm | 1.598 |
| *Drosophila santomea* | this slot (30/41), Holocene, 0 Ma | 2.326 µm | 1 |
| *Drosophila sechellia* | successor (31/41), melanogaster subgroup | 2.050 µm | 1.287 |

*D. santomea* is longer-wavelength than both neighbors because loss of the abdominal *tan* enhancer (Jeong et al. 2008, Cell) lowers pigment-conversion rate and therefore `k_c`. The biological sister species is *D. yakuba*, not the chronoseries predecessor. *D. pseudoobscura* is the preceding entry in the project chronoseries, not a direct ancestor.

## Specimen image

`drosophila_santomea_specimen.jpg` is a byte copy of `datasets/species_gallery/II_Anaima_Invertebrata/Drosophila_santomea.jpg` (SHA-256 `9ea05ad5963c10d32cbb31e1b123209339557f48d01b60361533a9069859be37`).

Wikimedia Commons had no still photograph (only PLOS ONE supplementary courtship videos). iNaturalist returned no taxon photo. The gallery fallback is a sepia spiral-wave field with a drawn 100 µm scale bar and a caption bar. It is **not** a micrograph that resolves 2.3 µm pigment foci. The script measures that scale bar and the field’s arm spacing and does not treat them as a published 2.326 µm observation.

## Run

```bash
python3 models/2026-delphi10000-drosophila-santomea-turing-wavelength-s0.py --self-test
python3 models/2026-delphi10000-drosophila-santomea-turing-wavelength-s0.py --pass 2
```

`--pass 1` writes the core dispersion fit and the growth GIF only. `--pass 2` (default) adds the `γ` sweep, the equal-diffusion ablation, specimen scale-bar segmentation, and the KS comparison.

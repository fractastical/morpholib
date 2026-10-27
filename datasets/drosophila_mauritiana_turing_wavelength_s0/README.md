# *Drosophila mauritiana* — Turing wavelength, detail scale (slot 3191)

Holocene (extant) Mauritius endemic fruit fly, compact spinose genital-arch lobe.
This slot is the **1.6 µm** detail variant of Turing pigment-stripe / ocellus wavelength selection
(`1.637e-06 m`, `log10 = -5.786`, tier `3_subcellular_um`). Geological age is **0 Ma**.

It is not the cellular-tier sibling at 89 µm (`datasets/drosophila_mauritiana_turing_wavelength`, slot 2721).
The dispersion law is the same family (Turing wavenumber `k_c`); the selected wavelength, diffusion
contrast, and mode count on the compact lobe are specific to this scale.

## Chronoseries

Drosophila chronoseries stage 20/41, lineage Sophophora melanogaster subgroup radiation (0–12 Ma).

| Role | Taxon | What changes at this scale |
| --- | --- | --- |
| Predecessor | *Drosophila kikkawai* | Cellular pigment-stripe wavelength 93.41 µm. One such wavelength is longer than the compact lobe, so the cellular mode does not tile the spinose arch. |
| This slot | *Drosophila mauritiana* | Detail wavelength **1.637 µm** (0 Ma). About 57× shorter than the kikkawai cellular wavelength, so a 38.4 µm compact-lobe window holds ~23 micro-ridges. |
| Successor | *Drosophila melanogaster* | Bare triangular posterior lobe (Glassford et al. 2015). Modelled here as a 1.22× longer inhibitor range, which shifts `k_c` upward in wavelength. Not copied from a melanogaster script. |

## Model

Schnakenberg kinetics with diffusion-driven instability (Turing 1952; Murray 1981). The rate `γ` is
calibrated so the fastest-growing wavenumber satisfies `λ = 2π / k_max = 1.637 µm`. A free (`κ = 0`)
branch measures that wavelength. An equal-diffusion ablation is the negative control. A nudged branch
assimilates the canonical ocellus illustration for goodness-of-fit (Pearson r, RMSE, KS).

## Specimen

Wikimedia Commons had no species-specific CC photograph of *D. mauritiana* on 2026-10-27.
The image `Drosophila_mauritiana.jpg` is the repository gallery fallback
(`datasets/species_gallery/II_Anaima_Invertebrata/Drosophila_mauritiana.jpg`), SHA-256-pinned in the
reference JSON. It is an illustrated sepia ocellus/spiral used as the canonical specimen for this taxon,
not a new genital-arch micrograph.

## Run

```bash
python3 models/2026-delphi10000-drosophila-mauritiana-turing-wavelength-s0.py --self-test
python3 models/2026-delphi10000-drosophila-mauritiana-turing-wavelength-s0.py
```

Scale stated for this package: **1.6 µm** (`1.637e-06 m`). Age stated for this package: **0 Ma**.

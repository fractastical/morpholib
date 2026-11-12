# Drosophila teissieri — Turing wavelength, detail scale (slot 3209)

Extant Holocene specimen of *Drosophila teissieri* Tsacas, 1971 (Afrotropical highland forest fly; multi-spined clasper surstylus). Aristotelian place: II. Anaima (Invertebrata) → Entoma (Arthropoda) → *Drosophila*. Chronoseries position 38/41 in the Project Delphi Drosophila series, lineage Sophophora melanogaster subgroup radiation (0–12 Ma). Geological age of the modeled specimen: **0 Ma**.

This slot is the **detail variant (s0)** of the pigment-stripe / ocellus Turing wavelength. The assigned observation scale is **1.7 µm** (`1.72e-06 m`, `log10 = -5.764`, tier `3_subcellular_um`). It is not the 107 µm primordium-tier sibling (`models/2017-turissini-drosophila-teissieri-turing-wavelength.py`, slot 2761) and it does not reuse that script’s power-law coefficients.

## What the model is

Schnakenberg (1979) kinetics on a periodic cuticle patch, with diffusivities chosen so Turing’s critical wavenumber (1952) satisfies

```
λ_c = 2π / k_c = 1.72 µm
k_c² = (D_v f_u + D_u g_v) / (2 D_u D_v)
```

The patch side is `12 × 1.72 µm = 20.64 µm`. Predecessor contrast: *D. takahashii* cell-tier atlas stripe is 97.6 µm (slot 2811), so this detail wavenumber is about 57× finer. Successor contrast: *D. virilis* atlas floor is 2.72 µm, longer than this detail wavelength by a factor of about 1.58. All three terminals are extant (0 Ma); the contrast is a chronoseries of scales, not a claim that virilis descends from teissieri.

## Specimen

`drosophila_teissieri_turing_wavelength_s0_specimen.jpg` is a byte copy of `datasets/species_gallery/II_Anaima_Invertebrata/Drosophila_teissieri.jpg` (SHA-256 `530859c7d91e90a6c46d095c76a7a5e85abd4230e94bb9ec512369bc0d0d6066`). Wikimedia Commons returned no CC-licensed bitmap for this species (one CC BY 2.5 video of *D. yakuba* / *D. santomea* only). The micrograph is labeled with a 100 µm scale bar (measured here as 111 px). A 1.72 µm wavelength is about 1.9 px on that frame, so the animation shows it as a magnified 20.64 µm inset rather than as a resolvable stripe on the full field.

## Run

```bash
python3 models/2026-delphi10000-drosophila-teissieri-turing-wavelength-s0.py --self-test
python3 models/2026-delphi10000-drosophila-teissieri-turing-wavelength-s0.py
```

Outputs: `docs/media/drosophila_teissieri_turing_wavelength_s0_comparison.png` and `docs/media/drosophila_teissieri_turing_wavelength_s0_growth.gif`.

## Sources

Crossref-checked: Turing 1952 (`10.1098/rstb.1952.0012`), Schnakenberg 1979 (`10.1016/0022-5193(79)90042-0`), Kondo & Miura 2010 (`10.1126/science.1179047`), Crampin & Maini 2001 (`10.4310/maa.2001.v8.n3.a3`), Tamura et al. (`10.1093/molbev/msg236`). Taxonomy: GBIF 5074619. DOI `10.1093/molbev/msx210` was **not** used: Crossref resolves it to a yeast de-novo-gene paper.

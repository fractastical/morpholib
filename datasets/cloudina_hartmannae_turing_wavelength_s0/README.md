# Cloudina hartmannae — 2.4 µm Turing wavelength (slot 3149)

Detail-scale reproduction of **Turing wavenumber selection** on the organic lamellae of *Cloudina hartmannae* (terminal Ediacaran, Nama Group, **549 Ma**).

This slot is the subcellular rung, not the coarser funnel rung.

| | This slot (3149, s0) | Predecessor rung (slot 1252) |
|---|---|---|
| Scale | **2.4 µm** (2.397 × 10⁻⁶ m, log10 = −5.620, tier `3_subcellular_um`) | 104.2 µm (tier `5_primordium_100um`) |
| What the wavelength is | Wall-normal organic-lamella spacing | Cone-in-cone funnel / primordium repeat |
| Age | 549 Ma, chronoseries 1/1 | Same terminal-Ediacaran body; no second age in this series |
| Parameters | Schnakenberg (a, b, Du, Dv) locked so λ_c = 2.397 µm | Stub script with no diffusivity; not reused |

## Why 2.397 µm is this rung

- Yang et al. (2020, Scientific Reports, doi:10.1038/s41598-019-56317-x) measured organic lamellae **0.5–5 µm** thick in Namibian *Cloudina*, up to **eight** lamellae in one wall layer.
- Grant (1990, Am. J. Sci. 290-A:261–294) measured shell layers **8–50 µm** thick.
- Eight lamellae × 2.397 µm = **19.176 µm**, which sits inside Grant’s layer range. The 2.397 µm mode is the lamella spacing; the 104.2 µm mode is ~43.5× longer.
- Because λ ∝ √D at fixed reaction rates, the lamella-scale diffusivities are smaller than a funnel-scale pair by (2.397/104.2)² ≈ 5.29 × 10⁻⁴.
- Finer than this slot, and not modeled here: Yang’s sub-micrometre *Zuunia* lamellae and 80–200 nm fibrils.

The assignment age is **549 Ma** (terminal Ediacaran, Nama Group). Penny et al. (2014) date an ash above the Omkyk Member at 547.32 ± 0.31 Ma; that radiometric age is a constraint on the same succession, not a second chronospecies.

## Model

Schnakenberg (1979) kinetics on a periodic wall-normal interval, with the Turing (1952) critical wavenumber

k_c² = (D_v f_u + D_u g_v) / (2 D_u D_v), λ_c = 2π / k_c.

Parameters (unique to this rung; not the textbook pair a = 0.1, b = 0.9 and not slot 1252):

- a = 0.0642, b = 0.9915, D_v/D_u = 8.65
- D_u and D_v set so λ_c = 2.397 µm exactly
- The discriminant is only slightly positive, so the fastest-growing mode stays on k_c

## Images

- `cloudina_hartmannae_specimen.jpg` — project species card copied from `datasets/species_gallery/II_Anaima_Invertebrata/Cloudina_hartmannae.jpg`. The card’s scale bar reads 100 µm, so it is the tube, not a 2.4 µm micrograph. The growth GIF is scored against this card.
- `cloudina_hartmannae_wikimedia_cc_by_sa.jpg` — Wikimedia `File:Cloudina_NT.jpg`, CC BY-SA 4.0. Life restoration, not a fossil photograph.
- `cloudina_corumba_cc_by_2.jpg` — Wikimedia photograph of *Cloudina* tubes from the Neoproterozoic of Corumbá, Brazil (CC BY 2.0). Genus-level fossil slab (*C. carinata* assemblage, not Nama *C. hartmannae*). Used as an independent foreground segmentation, not as the fitness target.

## Run

```bash
python3 models/2026-delphi10000-cloudina-hartmannae-turing-wavelength-s0.py --self-test
python3 models/2026-delphi10000-cloudina-hartmannae-turing-wavelength-s0.py
```

Outputs: `docs/media/cloudina_hartmannae_turing_wavelength_s0_comparison.png` and `docs/media/cloudina_hartmannae_turing_wavelength_s0_growth.gif`.

## What this does not claim

Cloudina is not known to have had pigment ocelli. Grant argued for a cnidarian-grade filter feeder; Yang et al. argue for an originally organic, annelid-grade tube. The script reconstructs a lamella-scale Turing wavelength inside the published thickness band. It does not decide affinity, and it does not claim that 2.397 µm was measured as a pigment stripe on the species card.

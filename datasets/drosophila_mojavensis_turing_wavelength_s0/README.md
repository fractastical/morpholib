# Dataset: `drosophila_mojavensis_turing_wavelength_s0` (Slot #3194)

*Drosophila mojavensis* — Sonoran Desert organ-pipe cactus fly. This slot is the **detail-scale** Turing wavelength, not the 96 µm cellular slot (`datasets/drosophila_mojavensis_turing_wavelength`, slot 2981).

- **Scale:** 2.4 µm (`2.38e-06 m`, catalog `log10 = -5.623`, tier `3_subcellular_um`). Exact `log10(2.38e-06) = -5.623423`.
- **Age:** Holocene (Extant), **0 Ma**. Drosophila chronoseries 23/41. Lineage: Subgenus *Drosophila* virilis–repleta–guttifera radiation.
- **Chronospecies at this scale:** predecessor *Drosophila miranda* (longer epicuticular range, 3.237 µm); focal *D. mojavensis* (2.38 µm); successor *Drosophila novamexicana* (shorter range, 1.948 µm). The ordering is the campaign chronoseries, not a claim that *D. miranda* is the cladistic parent of *D. mojavensis*.
- **Law:** Schnakenberg reaction–diffusion. The reported wavelength is the most-unstable mode `λ★ = 2π / k★` with `k★ = argmax σ(k)`, calibrated so `λ★ = 2.38 µm`. Absolute diffusivity is that calibration; the diffusion ratio `R = D_v/D_u = 80` and the kinetics `(a, b, γ) = (0.37, 0.71, 4)` are the mechanistic parameters.
- **What changes vs predecessor/successor:** both diffusivities scale by `s = (λ/λ★)²` with the Jacobian and `R` held fixed, so `λ ∝ √D`. Organ-pipe cactophily (Matzkin et al. 2006, 2012; Allan & Matzkin 2019) is the biological reason to expect a shifted epicuticular lipid range; the numerical values of `s` are a transfer hypothesis at 2.4 µm, not a published measurement of those two species at this wavelength.

## Files

| File | Role |
| --- | --- |
| `drosophila_mojavensis_turing_wavelength_s0_specimen.jpg` | Canonical species-gallery plate (byte-identical to `datasets/species_gallery/II_Anaima_Invertebrata/Drosophila_mojavensis.jpg`). Image-fitness target. |
| `drosophila_mojavensis_wikimedia_female.png` | Public-domain female habitus, Wikimedia Commons, 225×99. |
| `drosophila_mojavensis_turing_wavelength_s0_reference.json` | Parameters, claim boundary, SHA-256 pins. |
| `paper_citations.bib` | Crossref-checked citations. |
| `scout_manifest.json` | Step-1 dossier. |
| `metrics.json` | Written by the model script (Pearson r, RMSE, KS, sweep). |

## Reproduce

```bash
python3 models/2026-delphi10000-drosophila-mojavensis-turing-wavelength-s0.py --self-test
python3 models/2026-delphi10000-drosophila-mojavensis-turing-wavelength-s0.py
```

Outputs: `docs/media/drosophila_mojavensis_turing_wavelength_s0_comparison.png` and `docs/media/drosophila_mojavensis_turing_wavelength_s0_growth.gif`.

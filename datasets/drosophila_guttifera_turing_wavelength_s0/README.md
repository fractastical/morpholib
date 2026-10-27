# Drosophila guttifera — Turing wavelength, detail tier (slot 3187)

Holocene (extant) male *Drosophila guttifera* at the **2.12 µm** pigment micro-ocellus scale (2.12×10⁻⁶ m, log10 = −5.674, tier `3_subcellular_um`, detail variant `s0`). Geological age **0 Ma**. Lineage: Subgenus *Drosophila* virilis–repleta–guttifera radiation. Campaign chronoseries 16/41. Predecessor chronospecies: *Drosophila grimshawi*. Successor chronospecies: *Drosophila heteroneura*.

This slot is not the whole-wing polka-dot pattern (sibling slot 3011, ~102 µm). It is the subcellular Turing wavelength inside the yellow-melanin field that Wingless nucleates at a campaniform sensillum.

## What is taken from the literature

- Werner, Koshikawa, Williams & Carroll (2010) *Nature* 464:1143–1148, [10.1038/nature08896](https://doi.org/10.1038/nature08896). Wingless at vein intersections and campaniform sensilla induces the yellow wing spots.
- Koshikawa et al. (2015) *PNAS* 112:7524–7529, [10.1073/pnas.1509022112](https://doi.org/10.1073/pnas.1509022112). New *wingless* cis-regulatory domains.
- Koseki, Tanaka & Koshikawa (2021) *Dev Genes Evol* 231:85–93, [10.1007/s00427-021-00674-z](https://doi.org/10.1007/s00427-021-00674-z). *wingless* in specific campaniform-sensillum cell types.
- Fukutomi et al. (2017) *Dev Genes Evol* 227:171–180, [10.1007/s00427-017-0578-3](https://doi.org/10.1007/s00427-017-0578-3). Pupal pigmentation schedule.
- Turing (1952) [10.1098/rstb.1952.0012](https://doi.org/10.1098/rstb.1952.0012) and Schnakenberg (1979) [10.1016/0022-5193(79)90042-0](https://doi.org/10.1016/0022-5193(79)90042-0) supply the dispersion law. Werner 2010 does not publish a reaction–diffusion PDE and does not measure 2.12 µm.

## What changes at this scale and age

All three chronospecies are extant (0 Ma). The contrast is the pigment prepattern at 2.12 µm, not a stratigraphic time step and not the organism-scale body length.

| Chronospecies | Prepattern at 2.12 µm | Diffusion | Pattern class |
|---|---|---|---|
| *D. grimshawi* (predecessor) | Vein-locked lacework, no quinaria sensillum-spot gain | Anisotropic, D∥/D⊥ = 3.15 | Stripes |
| *D. guttifera* (this slot) | One campaniform Wingless source | Isotropic Du = 0.016, Dv = 1.25 | Spots / micro-ocelli, λ(k_max) = 2.12 µm |
| *D. heteroneura* (successor) | Uniform costal-band elevation of basal activator (a = 0.24) | Isotropic, no sensillum Gaussian | Shifted wavelength, no punctate nucleator |

The anisotropy ratio and the heteroneura activator elevation are reconstructed contrasts. They are not diffusion coefficients reported by Edwards et al. (2007) or Spieth (1981).

## Specimen

`drosophila_guttifera_turing_wavelength_s0_specimen.jpg` is a 960 px JPEG rendition of Wikimedia Commons `File:Dguttifera.tif` (MA Hanson, own work, 2021-06-19, CC BY-SA 4.0). SHA-256 is pinned in the reference JSON.

## Run

```bash
python3 models/2026-delphi10000-drosophila-guttifera-turing-wavelength-s0.py --self-test
python3 models/2026-delphi10000-drosophila-guttifera-turing-wavelength-s0.py
```

Outputs:

- `docs/media/drosophila_guttifera_turing_wavelength_s0_comparison.png`
- `docs/media/drosophila_guttifera_turing_wavelength_s0_growth.gif`
- `datasets/drosophila_guttifera_turing_wavelength_s0/run_metrics.json`

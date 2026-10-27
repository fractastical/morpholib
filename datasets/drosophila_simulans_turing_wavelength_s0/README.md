# Drosophila simulans — detail-scale Turing wavelength (slot 3205)

Holocene extant *Drosophila simulans* at **1.9 µm** (**1.92 × 10⁻⁶ m**, log10 = −5.717, tier `3_subcellular_um`) and geological age **0 Ma**.

This is the detail-scale carrier of the Turing wavenumber on the male epandrial posterior lobe (clasper arc), not the 83 µm tergite/ocellus band of slot 2701. The catalog chronoseries position is 34/41 in the Sophophora melanogaster subgroup radiation (0–12 Ma). The predecessor entry is *Drosophila silvestris* and the successor entry is *Drosophila subobscura*. Those neighbours are not phylogenetic ancestors; they are the adjacent bins of this chronoseries, and the posterior lobe itself is a melanogaster-subgroup novelty (Glassford et al. 2015).

## What changes at 1.92 µm and 0 Ma

Schnakenberg kinetics are held fixed at `(a, b) = (0.10, 0.90)` with diffusivities `Du = 0.05 µm²` and `Dv = 1.00 µm²` per unit time. Only the kinetic scale `γ` changes, and `γ` is set so the linear dispersion peak sits on each bin's wavelength:

| chronospecies | age role | posterior lobe | λ_c | γ / γ_simulans |
| --- | --- | --- | --- | --- |
| *D. silvestris* | predecessor | absent (picture-wing melanin bands) | 2.64 µm | (1.92/2.64)² |
| *D. simulans* | this slot, 0 Ma | present | 1.92 µm | 1 |
| *D. subobscura* | successor | absent (obscura-group anal plate; latitudinal size cline) | 1.60 µm | (1.92/1.60)² |

On a 96 µm arc (exactly 50 simulans wavelengths) the integer mode counts are 36, 50 and 60. Equal diffusivities make the diffusion drive `h = Du · trace(J)` negative, so no wavelength is selected. That ablation is the negative control.

The pinned micrograph is the repository species plate (100 µm scale bar, annular pigment field). Its measured ring spacing is an image-scale morphometric, about 67 µm, and is not the 1.92 µm detail law. The detail law is how many pigment-granule wavelengths pack into one measured annulus and into the 96 µm arc. A CC BY-SA 3.0 adult female (Andrew Weeks, Wikimedia) is vendored beside the plate; it does not resolve 1.92 µm.

## Laws

- Turing 1952 dispersion: a critical wavenumber `k_c` from `det(J − k² D) = 0`.
- Schnakenberg 1979 kinetics: `u* = a + b`, `v* = b / (a + b)²`.
- `k_c² = γ (Dv fu + Du gv) / (2 Du Dv)`, `λ_c = 2π / k_c`.
- Mode count `N = round(L / λ_c)` on the 96 µm arc.

## Reproduce

```bash
python3 models/2026-delphi10000-drosophila-simulans-turing-wavelength-s0.py --self-test
python3 models/2026-delphi10000-drosophila-simulans-turing-wavelength-s0.py
```

Outputs: `docs/media/drosophila_simulans_turing_wavelength_s0_comparison.png` and `docs/media/drosophila_simulans_turing_wavelength_s0_growth.gif`.

## Sources

See `paper_citations.bib`. Seed law: Turing 1952, doi:10.1098/rstb.1952.0012. Seed anatomy: Glassford et al. 2015, doi:10.1016/j.devcel.2015.08.005. Posterior-lobe diversification inside the last 0.24 Ma: Hagen et al. 2020. Picture-wing pigment: Edwards et al. 2007. Subobscura size cline: Huey et al. 2000.

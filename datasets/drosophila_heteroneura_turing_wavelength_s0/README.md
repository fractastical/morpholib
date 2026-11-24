# Drosophila heteroneura — detail-scale Turing wavelength (slot 3188)

Holocene (extant), **0 Ma**. Phenomenon scale **5.2 µm** (5.222 × 10⁻⁶ m, log10 = −5.282), tier `3_subcellular_um`.

This dossier is the detail-scale companion to the primordium-scale *D. heteroneura* slot (175 µm). It does not reuse that slot’s power-law parameters. The question at this scale is which critical wavenumber k_c a Schnakenberg pigment system selects on the hypercephalic head capsule, and how that wavenumber differs from the Delphi chronoseries neighbors *Drosophila guttifera* (predecessor) and *Drosophila hydei* (successor). Those neighbors are extant comparisons at 0 Ma, not a stratigraphic ancestor–descendant series.

## What the law selects

Schnakenberg kinetics (Schnakenberg 1979; dispersion in the sense of Turing 1952 and Murray 1981):

∂u/∂t = D_u ∇²u + γ (a − u + u²v)

∂v/∂t = D_v ∇²v + γ (b − u²v)

with a = 0.064, b = 1.137, D_v/D_u = 36.5, and γ_het = 18.6. The critical wavenumber is

k_c² = (D_v f_u + D_u g_v) / (2 D_u D_v),    λ_c = 2π / k_c.

Spatial calibration sets λ_c(*D. heteroneura*) = **5.222 µm**. Because λ ∝ γ^(−1/2) at fixed diffusivities, the same kinetics at the neighbors’ detail-tier rates give a longer wave: γ_gut = 7.188 → 8.400 µm (*D. guttifera*, Wingless-induced yellow melanin; Werner et al. 2010) and γ_hyd = 12.579 → 6.350 µm (*D. hydei*, repleta thoracic pigment; Pitnick et al. 1995). On a reconstructed 1.72 mm male head width (Spieth 1981 documents the hammerhead head relative to *D. silvestris*; the millimetre value is reconstructed), the mode counts are 329 (*heteroneura*), 270 (*hydei*), and 204 (*guttifera*).

Equal diffusivities (D_v/D_u = 1) keep the same reactions but the dispersion minimum stays positive: that is the negative control.

## Specimen

`drosophila_heteroneura_specimen.jpg` is the project gallery micrograph (SHA-256 pinned in the reference JSON). Wikimedia Commons returned no bitmap for this taxon during scouting, so the gallery file is the vendored image. A white scale bar on the frame measures 111 px = 100 µm. The resolved dark rings are spaced near 68 µm. That envelope is what the photograph shows. The 5.222 µm critical mode is a finer carrier (about 5.8 px at this sampling) and is not claimed to be the ring spacing of the micrograph.

## Reproduce

```bash
python3 models/2026-delphi10000-drosophila-heteroneura-turing-wavelength-s0.py --self-test
python3 models/2026-delphi10000-drosophila-heteroneura-turing-wavelength-s0.py
```

Outputs:

- `docs/media/drosophila_heteroneura_turing_wavelength_s0_comparison.png`
- `docs/media/drosophila_heteroneura_turing_wavelength_s0_growth.gif`
- `datasets/drosophila_heteroneura_turing_wavelength_s0/fit_metrics.json`

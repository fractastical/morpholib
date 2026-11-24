# Drosophila pseudoobscura — Turing wavelength, detail rung (slot 3200)

Holocene (extant) adult male of *Drosophila pseudoobscura* (Sophophora, obscura group), scored here at one point in the Drosophila chronoseries (position 29/41) and at one spatial rung.

- **Scale:** 2.0 µm (1.961×10⁻⁶ m, log10 = −5.708), tier `3_subcellular_um`, detail variant.
- **Age:** 0 Ma, Holocene (Extant). Geological age does not change relative to the predecessor or the successor; both are also living species. What changes is the wavelength rung selected by the same Turing wavenumber law.
- **Form:** stripes on the critical ring |k| = k_*, the subcellular carrier under the obscura-group cuticle. This file does not treat the two-segment sex comb or the third-chromosome inversions as a 2 µm pigment stripe.
- **Law:** Turing wavenumber. The determinant of the reaction–diffusion symbol is minimised at

  k_c² = (D_v f_u + D_u g_v) / (2 D_u D_v),

  and the realised pattern is the fastest-growing mode k_* of the dispersion relation σ(k). Parameters are held just above the Turing threshold so that λ_* = 2π/k_* = 1.961 µm.

## Chronoseries neighbours (same law, different rung)

| Node | Species | Age | Wavelength rung |
| --- | --- | --- | --- |
| Predecessor | *Drosophila planitibia* | 0 Ma | 197.2 µm primordium (picture-wing melanin band) |
| This slot | *Drosophila pseudoobscura* | 0 Ma | 1.961 µm detail (this model) |
| Successor | *Drosophila santomea* | 0 Ma | 102 µm primordium (abdominal pigment) |

Diffusion coefficients that place k_* at 1.961 µm are about (197.2/1.961)² ≈ 1.01×10⁴ times smaller than the diffusivities that would place the same Jacobian on the planitibia primordium rung, and about (102/1.961)² ≈ 2.71×10³ times smaller than the santomea rung. Age Δ = 0 Ma.

## Specimen

`drosophila_pseudoobscura_turing_wavelength_s0_specimen.jpg` is a JPEG flattening of Wikimedia Commons `File:Dpse_1.tif` (MA Hanson, 2019-08-25, own work), a lateral photograph of a *Drosophila pseudoobscura* male.

- License: CC BY-SA 4.0 (https://creativecommons.org/licenses/by-sa/4.0)
- Source: https://commons.wikimedia.org/wiki/File:Dpse_1.tif
- SHA-256: `8bb7f34ac600cac7feff3a3e2810b1f89199880dad8180913ec86151f88476aa`

The species-gallery JPEG is a smaller export of the same habitus and was not used, because the Commons file carries an explicit CC BY-SA 4.0 grant.

## Reproduce

```bash
python3 models/2026-delphi10000-drosophila-pseudoobscura-turing-wavelength-s0.py --self-test
python3 models/2026-delphi10000-drosophila-pseudoobscura-turing-wavelength-s0.py
```

Outputs:

- `docs/media/drosophila_pseudoobscura_turing_wavelength_s0_comparison.png`
- `docs/media/drosophila_pseudoobscura_turing_wavelength_s0_growth.gif`

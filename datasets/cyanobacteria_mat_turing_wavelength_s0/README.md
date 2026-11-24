# Cyanobacteria mat — Turing wavelength at 64 µm (loop 3158)

Holocene extant stromatolite-forming mat, geological age **0 Ma**. Chronoseries **1/1**: there is no predecessor chronospecies and no successor. Scale tier `4_cellular_10um`, detail variant. Characteristic length **6.419e-5 m** (64.19 µm, log10 = −4.193).

The assigned law is Turing wavenumber selection (Schnakenberg activator–inhibitor). The pattern class is spots / ocelli. Parameters `(a, b, D_v/D_u) = (0.11, 0.83, 16)` are specific to this slot.

## What changes at this scale

This is the only node in the chronoseries, so there is no temporal delta. The scale comparison is against other wavelengths of the same living mat:

- Petroff et al. (2010) conical stromatolites have a diffusion length of order 1 cm, about 150 times this wavelength. That cone model is not integrated here.
- A cyanobacterial cell is a few micrometres, so 64 µm is a many-cell pigment patch, not a single cell and not the millimetre-to-centimetre reticulate mats of Shepard & Sumner (2010).

## Specimen

`cyanobacteria_mat_holocene_culture_plate.jpg` is the repository species-gallery photograph of a cyanobacterial culture plate (SHA-256 `858930a5ad520ee46fa0270c232d2062e6aa15ba9307b6f74d3a6b63439fff8b`). It is a macroscopic plate. The simulation domain is 8 × 64.19 µm. Comparisons are scale-free (computer-vision spectra, median-normalised colony versus spot diameters). Wikimedia hits from the scout are Paleoarchean Dresser Formation stromatolites (3.48 Ga) and were not used.

## Run

```bash
python3 models/2026-delphi10000-cyanobacteria-mat-turing-wavelength-s0.py --self-test
python3 models/2026-delphi10000-cyanobacteria-mat-turing-wavelength-s0.py
```

Outputs:

- `docs/media/cyanobacteria_mat_turing_wavelength_s0_comparison.png`
- `docs/media/cyanobacteria_mat_turing_wavelength_s0_growth.gif` (12 frames, 640×520)
- `cyanobacteria_mat_turing_wavelength_s0_metrics.json`

Seed 3158. Diffusion is spectral; the reaction step is explicit Euler with `dt = 0.08` on the nonlinear field and `dt = 0.01` on the eigenvector probes.

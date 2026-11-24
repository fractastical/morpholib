# Ceratites nodosus — cellular Turing wavelength (slot 3142)

Middle Triassic (Ladinian, **240 Ma**) ammonoid *Ceratites nodosus* at the detail scale of **26 µm** (2.56×10⁻⁵ m, log10 = −4.592, tier `4_cellular_10um`).

This slot asks what a Turing reaction–diffusion system selects when the wavelength is an epithelial-cell scale rather than a rib scale. The quantitative law is the Turing wavenumber \(k_c\). The form class is stripes, spots, and ocelli on the mantle that secretes the shell.

## What is different at this scale and age

The campaign chronoseries inside *Ceratites* is a single horizon (1/1): the Ladinian nodosus zone at 240 Ma. There is no second *Ceratites* chronospecies in this slot. The predecessor and successor are the neighbouring points of the Cephalopod Fractal Suture Fossil Series, with shell size and suture descriptor taken from `datasets/species_gallery/species_manifest.json`:

| Horizon | Age | Shell size | Campaign suture descriptor | Cellular wavelengths across the shell at 26 µm |
|---|---:|---:|---|---:|
| *Goniatites crenistria* (predecessor) | 333 Ma | 18 mm | angular zigzag, D_f = 1.18 | 703 |
| *Ceratites nodosus* (this slot) | 240 Ma | 32 mm | serrated lobes, D_f = 1.34 | 1250 |
| *Dactylioceras commune* (successor) | 181 Ma | 42 mm | bifurcating ribs, D_f = 1.55 | 1641 |

The cellular wavelength is held fixed at 26 µm (a mantle-epithelial scale). What changes along the series is how many of those wavelengths fit across the shell, and the suture descriptor (D_f 1.18 → 1.34 → 1.55). Boyajian & Lutz (1992) is the method citation for ammonoid suture complexity. The D_f numbers above are the campaign descriptors in the species manifest, not new measurements on this photograph.

The same species at organ scale (sibling script `models/2004-klug-ceratites-nodosus-turing-wavelength.py`, λ = 1.088 mm) is a different slot. One organ-scale wavelength contains 42 cellular wavelengths (1.088 mm / 25.6 µm). Diffusivities here are solved so the selected wavenumber sits at 26 µm. They are not rescaled copies of the 1.1 mm parameter table.

## Model

Schnakenberg (1979) kinetics, the two-species system whose linearisation has a closed-form Turing wavenumber:

- ∂u/∂t = D_u ∇²u + a − u + u²v
- ∂v/∂t = D_v ∇²v + b − u²v

State variables: u activator (pigment precursor), v inhibitor. Boundary condition on the modelled mantle patch: periodic, matching the Fourier mode assumed by the k_c derivation. Operating point: a = 0.062, b = 1.140, diffusion ratio 1.15 times the onset ratio, domain 8 × 26 µm on a 96² grid, seed 3142.

Meinhardt & Klingler (1987) is the molluscan shell-pigment interpretation (a mantle edge lays down spots and stripes). This script implements the wavenumber-selection law, not their full travelling-wave shell record.

## Specimen

`ceratites_nodosus_specimen.jpg` is the campaign gallery photograph (`datasets/species_gallery/II_Anaima_Invertebrata/Ceratites_nodosus.jpg`), SHA-256 `3c37b42ed1bd3fdedd3bbdb0eeef8244f5d9c6fffd9f8fa40804afed5dbc58f0`. Wikimedia Commons also holds CC-licensed *Ceratites nodosus* shells (see `scout_manifest.json`). The fossil photograph resolves coiling, ribs, and nodes. It does not resolve a 26 µm pigment. The growth GIF keeps the shell luminance and uses colour for the simulated activator.

## Run

```bash
python3 models/2026-delphi10000-ceratites-nodosus-turing-wavelength-s0.py --self-test
```

Outputs:

- `docs/media/ceratites_nodosus_turing_wavelength_s0_comparison.png`
- `docs/media/ceratites_nodosus_turing_wavelength_s0_growth.gif`
- `datasets/ceratites_nodosus_turing_wavelength_s0/simulation_metrics.json`

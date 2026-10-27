# *Drosophila virilis* — Turing wavelength at 3.1 µm (slot 3210)

Detail-scale reaction-diffusion model for Loop 3210 of the 10,000-Animal Basic Look.
This slot is the **3.1 µm** (3.101 × 10⁻⁶ m, log10 = −5.508, tier `3_subcellular_um`) cut of the Holocene (extant, **0 Ma**) *Drosophila virilis* chronoseries step. It is not the 136 µm primordium-scale sibling (`models/2007-clark-drosophila-virilis-turing-wavelength.py`).

## What is being asked at this scale

The assigned anatomical target is the Turing pigment stripe / ocellus wavelength. At 3.1 µm that wavelength sits inside one epidermal cell field (ocellar pigment epithelium and abdominal tergite melanocytes), not on the spacing of whole tergite bands. The sibling 136 µm slot is about 44 of these detail wavelengths long (136000 / 3101 nm ≈ 43.9).

Geological age is **0 Ma** for *D. virilis* and for both chronoseries neighbors. Nothing in this slot is a fossil time series. The predecessor is *Drosophila teissieri* and the successor is *Drosophila willistoni*, both extant. What changes along that step, at this scale, is the basal activator production that decides whether a wavenumber is selected, not a change in molecular diffusivity and not a change in age.

## Model

Schnakenberg kinetics (Schnakenberg 1979), the two-species reduction used to illustrate Turing’s 1952 dispersion relation:

- ∂u/∂t = Du ∇²u + a − u + u²v
- ∂v/∂t = Dv ∇²v + b − u²v

Homogeneous steady state (u₀, v₀) = (a + b, b / (a + b)²). Critical wavenumber

k_c² = (Dv f_u + Du g_v) / (2 Du Dv), λ_c = 2π / k_c.

Shared across the three Holocene species (molecular diffusivities, detail-scale invariant):

- b = 0.90
- Dv / Du = 10 (near the Turing threshold, so the fastest mode stays close to onset k_c)
- Du solved so that *D. virilis* at a = 0.10 selects **λ_c = 3.101 µm**

Species-specific basal activator production `a`:

| Species | Role | Adult length used for mode counts | a | Selected λ |
| --- | --- | --- | --- | --- |
| *D. teissieri* | predecessor | 2.5 mm | 0.058 | 2.924 µm (Turing on; posterior-gated in the reading) |
| *D. virilis* | this slot | 3.4 mm | 0.100 | 3.101 µm (Turing on across the tergite field) |
| *D. willistoni* | successor | 2.32 mm | 0.360 | none (past the Turing boundary; uniform dark steady state) |

Adult lengths are the project species-atlas `characteristic_size_m` values, used only to convert a wavelength into a mode count N ≈ (L / 7) / λ on one tergite. They are not a new measurement of these specimens.

## Specimen

`drosophila_virilis_habitus.jpg` — lateral habitus of *Drosophila virilis*, photographed by M. A. Hanson (Wikimedia Commons `File:Dvir 1-2.jpg`, 25 Aug 2019, CC BY-SA 4.0). SHA-256 is recorded in `drosophila_virilis_turing_wavelength_s0_reference.json`. The photograph is the English Wikipedia lead image for the species. Its pixel scale does not resolve 3.1 µm; the script reports that Nyquist limit instead of pretending the habitus FFT is the Turing wavelength.

The project vernacular name “6-Chorion-Filament Egg” is retained as the slot label. This model does not use filament number as a parameter, and this dataset does not treat that count as a primary measurement.

## Reproduce

```bash
python3 models/2026-delphi10000-drosophila-virilis-turing-wavelength-s0.py --self-test
python3 models/2026-delphi10000-drosophila-virilis-turing-wavelength-s0.py
```

Outputs:

- `docs/media/drosophila_virilis_turing_wavelength_s0_comparison.png`
- `docs/media/drosophila_virilis_turing_wavelength_s0_growth.gif`

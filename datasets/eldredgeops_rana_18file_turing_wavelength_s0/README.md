# Eldredgeops rana_18file — 492 nm Turing wavelength (slot 3215)

Detail-scale reproduction for the 10,000-Animal Basic Look. This slot is chronospecies 1/2 of the Eldredgeops eye-file series at **387 Ma** (Middle Devonian, Cazenovian). The assigned phenomenon scale is **492 nm** (`4.915e-07 m`, log10 = -6.308, tier `2_macromolecular_100nm`, variant `s0`).

The sibling cellular-tier script `models/1972-eldredge-eldredgeops-rana-18file-turing-wavelength.py` uses an 18 µm wavelength (`1.786e-05 m`). Nothing here reuses that diffusivity, grid, or steady state. At 492 nm the selected quantity is the linear Turing wavenumber `k_c = 2π / λ_c` of a Schnakenberg activator–inhibitor pair inside one schizochroal lens, not the dorsoventral file spacing of the whole eye.

## What changes at this scale and age

Eldredge (1972, Bull. AMNH 147(2); AMNH handle 2246/1095) tracked *Phacops rana* (now *Eldredgeops rana*) by the number of dorsoventral files of calcite lenses. This slot is the **18-file** morph at 387 Ma. The successor in the campaign series is *Eldredgeops rana_17file* at 384 Ma (Tioughniogan), one file fewer.

The 492 nm wavelength is the campaign’s detail-scale pigment / ocellus wavelength. It is **not** a length Eldredge measured. File loss does not, in this reconstruction, retune `k_c`: the critical wavelength is fixed by the Jacobian and the two diffusivities inside a lens. If the dorsoventral eye arc is conserved while one file is deleted, each remaining file subtends `18/17` times as many 492 nm wavelengths. That ratio is the chronospecies change **at this scale**. The integer file count is the change at organism scale.

## Model

Schnakenberg (1979) kinetics with Turing’s (1952) diffusion-driven instability:

```
∂u/∂t = Du ∇²u + γ (a − u + u² v)
∂v/∂t = Dv ∇²v + γ (b − u² v)
k_c²  = (Dv f_u + Du g_v) / (2 Du Dv)
λ_c   = 2π / k_c = 492 nm
```

Steady state `(a, b, γ) = (0.146, 1.038, 2.40)`. Pre-scale diffusivities `Du = 0.018`, `Dv = 0.492` (the inhibitor value carries the 492 nm label). Space is then rescaled so the analytic critical wavelength is exactly one model unit, and that unit is defined to be `4.915e-07 m`.

A negative control sets `Du = Dv`. The homogeneous state stays stable: the largest real growth rate on the sampled band is negative. Pass 2 adds the simulated equal-diffusion field, a γ and `Dv/Du` sweep, multi-seed wavelength dispersion, and an Otsu / radial-spectrum comparison to the specimen.

## Specimen

`eldredgeops_rana_18file_turing_wavelength_s0_specimen.jpg`

- Wikimedia Commons `File:Eldredgeops eye.jpg`
- Artist: RealGatba (own work, 2022-07-13)
- License: CC BY-SA 4.0 (https://creativecommons.org/licenses/by-sa/4.0)
- Page: https://commons.wikimedia.org/wiki/File:Eldredgeops_eye.jpg
- SHA-256: `3a83d6e639450366d894856ec070022103defb1e7f56bb6faec9648b6d601986`

The photograph shows the schizochroal eye of *Eldredgeops rana*. It is not a proven Cazenovian 18-file stratigraphic voucher; the file count used here is the chronospecies label. The synthetic gallery plate `datasets/species_gallery/II_Anaima_Invertebrata/Eldredgeops_rana_18file.jpg` was not used.

## Run

```bash
python3 models/2026-delphi10000-eldredgeops-rana-18file-turing-wavelength-s0.py --self-test
python3 models/2026-delphi10000-eldredgeops-rana-18file-turing-wavelength-s0.py
```

Outputs:

- `docs/media/eldredgeops_rana_18file_turing_wavelength_s0_comparison.png`
- `docs/media/eldredgeops_rana_18file_turing_wavelength_s0_growth.gif`

The GIF is a 12-frame, 720×540 animation of the reaction–diffusion field. Its last frame is the Commons photograph so the image-fitness score compares the simulation canvas to the real eye. The photograph is not a solution of the PDE; the comparison figure carries the dispersion curve.

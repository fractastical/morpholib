# Eldredgeops rana_17file — Turing wavelength at 342 nm (slot 3216)

Middle Devonian (Tioughniogan) chronospecies *Eldredgeops rana* with a **17-file** schizochroal eye, at **384 Ma**. This slot models one scale only: the Turing reaction–diffusion wavelength of the lens pigment / ocellus pattern, **342 nm** (3.422×10⁻⁷ m, log10 = −6.466, tier `2_macromolecular_100nm`, detail variant).

The predecessor in the Eldredgeops chronoseries is *Eldredgeops rana_18file* (18 dorsoventral lens files). There is no successor in this two-step series. Eldredge (1972) showed the file count steps from 18 to 17 between the Cazenovian and Tioughniogan horizons rather than drifting continuously. At the 342 nm pigment scale the adopted law is **chemistry-conserved wavenumber selection**: the critical wavelength stays 342 nm, and the anatomical integer that changes is the file count (18 → 17). A self-similar alternative, in which wavelength scales with file pitch, would put the predecessor at 342 × 17/18 = 323 nm. That alternative is reported as a contrast and is not this slot’s parameter.

## Governing model

Schnakenberg kinetics (Schnakenberg 1979) on the Turing (1952) dispersion relation:

- ∂u/∂t = ρ_a − u + u²v + D_u ∇²u
- ∂v/∂t = ρ_b − u²v + D_v ∇²v
- k_c² = (D_v f_u + D_u g_v) / (2 D_u D_v)
- λ_c = 2π / k_c, with the length unit chosen so λ_c = 342 nm

Slot parameters (not taken from the 18 µm sibling script): ρ_a = 0.082, ρ_b = 0.918, D_u = 0.060, D_v = 1.200, periodic cell side 1.15 λ_c, seed 3216.

## Run

```bash
python3 models/2026-delphi10000-eldredgeops-rana-17file-turing-wavelength-s0.py --self-test
python3 models/2026-delphi10000-eldredgeops-rana-17file-turing-wavelength-s0.py
```

Outputs:

- `docs/media/eldredgeops_rana_17file_turing_wavelength_s0_comparison.png`
- `docs/media/eldredgeops_rana_17file_turing_wavelength_s0_growth.gif`

The specimen plate is the project gallery image for this chronospecies (a radial ocellus field with a labeled 100 µm scale bar). It is a scale-free shape target for the pigment-ring profile. It is not a measured 342 nm micrograph of a Tioughniogan lens.

## Sources

See `paper_citations.bib`. Primary mathematical source: Turing (1952), doi:10.1098/rstb.1952.0012. Kinetics: Schnakenberg (1979). File-count punctuation: Eldredge (1972), https://digitallibrary.amnh.org/handle/2246/1096, and Eldredge & Gould (1972).

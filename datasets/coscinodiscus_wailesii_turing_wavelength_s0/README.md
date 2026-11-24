# *Coscinodiscus wailesii* — Turing wavelength at 4.3 mm (slot 3156)

Holocene (extant), age **0 Ma**. Coscinodiscus chronoseries **1/1**. Characteristic scale **4.3 mm** (0.004316 m, log10 = -2.365), tier 6/8 `6_organ_mm`, detail variant.

This slot asks for the Turing reaction-diffusion wavelength of a coarse-grained areola / ocellus field on the centric diatom *Coscinodiscus wailesii*. The series has no predecessor and no successor. What changes at this scale and age is the bifurcation: the homogeneous Schnakenberg steady state is stable below the Turing threshold and selects a finite wavenumber above it. The nominal wavelength from `k_c = argmax sigma_+(k)` is calibrated to 4.3 mm.

The vendored micrograph is the CC BY-SA 4.0 Wikimedia SEM “Inner and external plates of a single valve of the centric diatom Coscinodiscus wailesii”, byte-identical to `datasets/species_gallery/IV_Protista_Monera_Mycota/Coscinodiscus_wailesii.jpg` (SHA-256 `cf556dab9e116068cea62ccde9e852647efad520e12fa3a7ee1f547bdadcfd12`).

## Sources

- Turing, A. M. (1952) *Phil. Trans. R. Soc. B* 237:37–72. doi:10.1098/rstb.1952.0012
- Schnakenberg, J. (1979) *J. Theor. Biol.* 81:389–400. doi:10.1016/0022-5193(79)90042-0
- Schmid, A.-M. M. & Volcani, B. E. (1983) *J. Phycol.* 19:387–402. doi:10.1111/j.0022-3646.1983.00387.x
- Edwards, M., John, A. W. G., Johns, D. G. & Reid, P. C. (2001) *J. Mar. Biol. Ass. U.K.* 81:207–211. doi:10.1017/S0025315401003654
- Kondo, S. & Miura, T. (2010) *Science* 329:1616–1620. doi:10.1126/science.1179047

Schmid & Volcani describe centrifugal, proximal-to-distal valve formation, a rib system that outlines areolae, and distal cribra. Edwards et al. place the species in the north-east Atlantic from 1977 through 1995. Neither abstract states a valve diameter. The length 4.3 mm is the Delphi wavelength for this slot, not a measured frustule diameter.

## Law

Schnakenberg kinetics on a plane:

    du/dt = D_u ∇²u + a − u + u²v
    dv/dt = D_v ∇²v + b − u²v

with `a = 0.10`, `b = 0.90`, `D_u = 1.60`, nominal `D_v = 32` (seed 3156). The reported wavenumber is the maximizer of the positive dispersion branch `sigma_+(k)`, not the determinant minimizer `q_det`, which shortens the wavelength as `D_v` grows on this branch.

## Run

```bash
python3 models/2026-delphi10000-coscinodiscus-wailesii-turing-wavelength-s0.py --self-test
python3 models/2026-delphi10000-coscinodiscus-wailesii-turing-wavelength-s0.py
```

Outputs:

- `docs/media/coscinodiscus_wailesii_turing_wavelength_s0_comparison.png`
- `docs/media/coscinodiscus_wailesii_turing_wavelength_s0_growth.gif`

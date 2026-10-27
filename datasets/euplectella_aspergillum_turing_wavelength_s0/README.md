# Euplectella aspergillum — Turing wavelength, detail scale s0

Slot 3224 of the 10,000-animal basic look (Project Delphi / MorphoEval).

- **Animal:** *Euplectella aspergillum* (Venus flower basket), a hexactinellid glass sponge.
- **Scale:** 34 µm detail variant, pinned as λ_c = 3.43×10⁻⁵ m (34.3 µm, log10 = −4.465), tier `4_cellular_10um`.
- **Age:** Holocene (extant), 0 Ma. Chronoseries length 1/1: there is no predecessor or successor chronospecies in this slot. The scale comparison is against the coarser 51 µm cellular-tier wavelength (λ = 5.10×10⁻⁵ m), which is a different slot and is not copied here.
- **Law:** Turing wavenumber k_c = 2π / λ_c on the critical circle, with the Euplectella planform taken as four wavevectors of equal |k| (orthogonal struts plus both diagonal ridges; Aizenberg et al. 2005).

The cohort name “pigment stripe / ocellus” is a transfer. This animal has no ocelli and no pigment stripes. The wavelength is applied to the skeletal-lattice pitch selected by a reaction–diffusion dispersion relation, not to a coat pattern.

## Reproduce

```bash
python3 models/2026-delphi10000-euplectella-aspergillum-turing-wavelength-s0.py --self-test
python3 models/2026-delphi10000-euplectella-aspergillum-turing-wavelength-s0.py
```

Outputs:

- `docs/media/euplectella_aspergillum_turing_wavelength_s0_comparison.png`
- `docs/media/euplectella_aspergillum_turing_wavelength_s0_growth.gif`

## Specimen

`euplectella_aspergillum_turing_wavelength_s0_specimen.jpg` is the NOAA Okeanos Explorer photograph *File:Euplectella aspergillum Okeanos.jpg* (Wikimedia Commons, public domain). SHA-256 is recorded in the reference JSON.

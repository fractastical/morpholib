# *Gasterosteus doryssus_late* — 27 µm Turing wavelength (slot 3231)

Detail-scale reaction-diffusion wavelength for Miocene stickleback chronospecies III
(pelvic loss, T = +22 kyr) in the Truckee Formation.

- **Scale:** 27 µm (2.707 × 10⁻⁵ m, log10 = −4.568), tier 4/8 `4_cellular_10um`, detail variant.
  This is the melanophore / ocellus wavelength, not the 1.7 mm organ-scale stripe spacing
  used by the sibling slot `bell-gasterosteus-doryssus-late-turing-wavelength`.
- **Age:** 9.99 Ma, Late Miocene (Truckee Fm). Lineage: Gasterosteus Pelvic Reduction Chronospecies.
  Predecessor *G. doryssus_mid* (10.01 Ma, vestigial spine). Successor *G. aculeatus_marine*
  (Late Pleistocene, complete armor).
- **Law:** Turing wavenumber. The quantity reported is the fastest-growing mode λ_m = 2π / k_m
  of a Schnakenberg activator-inhibitor system (Turing 1952 dispersion; pigment RD framework
  of Kondo & Miura 2010). Pelvic evolution is taken from Hunt, Bell & Travis (2008).

The species-gallery plate is an illustrated ocellus card (100 µm scale bar), not a fossil
thin section. Fossils in this chronoseries record pelvic armor, not a measured 27 µm
melanophore spacing. The 27 µm figure is a scale-tier reconstruction: what the selected
pigment wavelength does at this cellular detail scale as the pelvic girdle is lost.

Run:

```bash
python3 models/2026-delphi10000-gasterosteus-doryssus-late-turing-wavelength-s0.py --self-test
python3 models/2026-delphi10000-gasterosteus-doryssus-late-turing-wavelength-s0.py
```

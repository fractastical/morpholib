# Bos taurus — Turing wavelength, detail scale (slot 3135)

Holocene extant *Bos taurus* (0 Ma). Bos chronoseries point 1 of 1: there is no assigned predecessor chronospecies. The quantity that changes relative to the neighbouring **scales**, not a sibling chronospecies, is whether a domain can admit the critical Turing mode.

| | |
| --- | --- |
| Scale | **3.1 µm** (3.137 × 10⁻⁶ m, log10 = −5.503) |
| Tier | 3/8 `3_subcellular_um` (detail variant) |
| Age | **0 Ma**, Holocene (Extant) |
| Form | Stripes, spots and ocelli |
| Law | Turing wavenumber k_c (selected mode k_star) |
| Kinetics | Gierer & Meinhardt (1972), not the Schnakenberg flank model |

The assigned wavelength sits between a sub-micrometre melanosome (tier 2 predecessor, taken as 500 nm following Wasmeier et al. 2008) and a melanocyte-soma domain (tier 4 successor, 20 µm). At 500 nm the fundamental wavenumber lies above the unstable band, so the organelle stays spatially uniform. At 3.137 µm the critical mode fits. At 20 µm the same wavenumber tiles into several stripes or ocelli (20 µm / 3.137 µm ≥ 6). Along the chronoseries the age offset is zero: domestication changes which coat-colour alleles are common, and this loop does not claim that it rescales the subcellular wavelength.

The voucher photograph is an unmarked Hereford calf (Jjron / Fir0002, CC BY-SA 3.0). A portrait does not resolve 3.137 µm. Morphometrics are taken on the simulated field; the portrait is the organism-level specimen the rendered animation is scored against.

This slot does not reuse `models/1981-murray-bos-taurus-turing-wavelength.py` (centimetre-scale Holstein patches, Schnakenberg kinetics, D_v/D_u = 20).

```bash
python3 models/2026-delphi10000-bos-taurus-turing-wavelength-s0.py --self-test
python3 models/2026-delphi10000-bos-taurus-turing-wavelength-s0.py
```

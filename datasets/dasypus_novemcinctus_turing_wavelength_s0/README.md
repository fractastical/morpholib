# Dasypus novemcinctus — Turing wavelength, detail scale (slot 3166)

Holocene (Extant) nine-banded armadillo, *Dasypus novemcinctus*, at the assigned subcellular detail scale of **4.1 µm** (**4.108 × 10⁻⁶ m**, log10 = −5.386, tier `3_subcellular_um`). Geological age **0 Ma**. Chronoseries length 1/1: there is no predecessor or successor chronospecies in this Dasypus series.

The quantitative law is the Turing critical wavenumber

`k_c² = (D_v f_u + D_u g_v) / (2 D_u D_v)`, `λ_c = 2π / k_c`,

realized with Schnakenberg kinetics on a periodic dermal patch. Parameters are reconstructed so that the linear wavelength equals 4.108 µm. They are not copied from the 273 µm primordium sibling `chen-dasypus-novemcinctus-turing-wavelength`.

What changes at this scale: one 273 µm primordium wavelength holds **66** detail-scale wavelengths (integer division of 273000 nm by 4108 nm), with a 1872 nm remainder. The Holocene age delta versus a missing chronospecies neighbour is 0 Ma.

Specimen photograph: Rhododendrites, “Nine-banded armadillo (13578)”, La Fortuna, Costa Rica, 2024-12-11, CC BY-SA 4.0, Wikimedia Commons. Vickaryous & Hall (2006) describe the carapace rosettes; Chen et al. (2011) describe osteoderm microstructure. Neither paper reports a 4.1 µm Turing wavelength; that target is the loop’s scale rung.

Run:

```bash
python3 models/2026-delphi10000-dasypus-novemcinctus-turing-wavelength-s0.py
python3 models/2026-delphi10000-dasypus-novemcinctus-turing-wavelength-s0.py --self-test
```

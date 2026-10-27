# Ciona intestinalis — 51 µm Turing wavelength (Loop 3148)

Holocene extant vase tunicate, age **0 Ma**, chronoseries **1/1**. There is no earlier or later chronospecies in this slot. The length that changes relative to neighbouring scales is the selected wavelength, not a species name.

**Scale.** Tier label 51 µm, SI pin **λ_c = 5.056×10⁻⁵ m** (log10 ≈ −4.296), tier `4_cellular_10um`, detail variant. The script integrates Schnakenberg kinetics on a periodic patch four wavelengths wide (about 202 µm). That patch is the detail-tier field. It is shorter than the adult body on the specimen photograph and longer than a single melanosome.

**Law.** Turing wavenumber

k_c² = γ (D_v f_u + D_u g_v) / (2 D_u D_v), λ_c = 2π / k_c

with Schnakenberg (1979) kinetics and the dispersion relation of Turing (1952). Activator u and inhibitor v are not assigned to a named Ciona gene. Pigment-cell context is Abitua et al. (2012), Jeffery et al. (2004), Hudson (2016), Hotta et al. (2007), and Kourakis et al. (2021).

**Specimen.** Yale Peabody Museum IZ 082346, photographed by Eric A. Lazo-Wasem, CC0. The print on the photograph is a 10 mm bar. Adult orange speckle spacing measured from that bar is hundreds of micrometres, not 51 µm. The adult photograph identifies the animal and supplies the silhouette for the computer-vision comparison. It does not resolve the detail-tier wavelength.

**Run.**

```bash
python3 models/2026-delphi10000-ciona-intestinalis-turing-wavelength-s0.py --self-test
python3 models/2026-delphi10000-ciona-intestinalis-turing-wavelength-s0.py
```

Outputs: `docs/media/ciona_intestinalis_turing_wavelength_s0_comparison.png` and `docs/media/ciona_intestinalis_turing_wavelength_s0_growth.gif`.

Sibling slot `ciona-intestinalis-turing-wavelength` (script id munro, 2002) is a different simulation id and is not this scale pin.

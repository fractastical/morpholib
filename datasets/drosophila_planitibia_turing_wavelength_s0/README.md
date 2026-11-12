# Drosophila planitibia — Turing wavelength, detail tier (slot 3199)

Holocene extant picture-winged fly *Drosophila planitibia* (Maui; band-winged lek display and foretarsal ciliation), Aristotelian path II. Anaima (Invertebrata) → Entoma (Arthropoda) → Drosophila. Lineage: Hawaiian Picture-Winged Drosophila Adaptive Radiation. Drosophila chronoseries position 28/41.

- **Scale:** 5.1 µm detail tier (`5.13e-06` m, assigned log10 = −5.290, tier `3_subcellular_um`). This is the critical Turing wavelength λ_c of a Schnakenberg activator–inhibitor system on the wing-membrane pigment field, not the whole-wing spot primordium.
- **Age:** 0 Ma, Holocene (Extant). Predecessor *Drosophila persimilis* and successor *Drosophila pseudoobscura* are the same geological age in this chronoseries. What changes at this step is the selected wavenumber, not the horizon.

The predecessor primordium-tier wavelength used by the chronoseries is 117 µm and the successor’s is 114 µm. Holding Schnakenberg kinetics and the diffusion ratio fixed, activator diffusivity scales as λ², so the planitibia detail mode is a ~520-fold collapse of D_u relative to *D. persimilis* and a ~494-fold collapse relative to *D. pseudoobscura*.

The vendored micrograph is the project gallery image `datasets/species_gallery/II_Anaima_Invertebrata/Drosophila_planitibia.jpg` (SHA-256 pinned in the reference JSON). Its printed 100 µm scale bar measures a coarse spiral carrier near 68 µm. That carrier is reported separately from λ_c = 5.13 µm. Edwards et al. (2007) and Kambysellis et al. (1995) do not publish a 5.13 µm measurement; the length is the assigned detail-tier phenomenon scale for this loop.

Governing law: Turing (1952) dispersion relation on Schnakenberg (1979) kinetics,

```
k_c^2 = ρ (D_v f_u + D_u g_v) / (2 D_u D_v),    λ_c = 2π / k_*
```

with k_* the fastest-growing wavenumber of σ(k). Equal diffusion (D_v / D_u = 1) is the negative control and lies below the Turing threshold.

Run from the repository root:

```bash
python3 models/2026-delphi10000-drosophila-planitibia-turing-wavelength-s0.py --self-test
```

# Drosophila suzukii — Turing wavelength at 2.0 µm (slot 3207)

Detail-scale carrier for **Turing Reaction-Diffusion Wavelength Selection** in *Drosophila suzukii* (spotted-wing drosophila: male apical wing spot and female serrated ovipositor).

- **Scale:** 2.0 µm (onset wavelength λ_c = 2.046 × 10⁻⁶ m, log10 = −5.689, tier `3_subcellular_um`).
- **Age:** Holocene (extant), **0 Ma**.
- **Lineage:** Sophophora suzukii–biarmipes wing-spot clade.
- **Chronoseries:** Drosophila 36/41. Predecessor *Drosophila subobscura*. Successor *Drosophila takahashii*. All three are extant; the series is a comparative order, not a stratigraphic succession.
- **Law:** Turing wavenumber k_c from the linear dispersion relation of a Schnakenberg activator–inhibitor pair.

The macroscopic wing spot is a *yellow* cis-regulatory domain (Gompel et al. 2005; Prud'homme et al. 2006; Koshikawa 2020). This slot does **not** claim that the spot outline is itself a 2.046 µm Turing pattern. It asks what the onset wavelength does *inside* melanized cuticle at this detail scale, and how that carrier changes from the predecessor (autocatalysis off) to the successor (longer wavelength).

Specimen photograph: `drosophila_suzukii_specimen.jpg`, copied from `datasets/species_gallery/II_Anaima_Invertebrata/Drosophila_suzukii.jpg` (Shane F. McEvey 2017 male and female plate; CC-licensed museum photograph; SHA-256 pinned in the reference JSON).

Run:

```bash
python3 models/2026-delphi10000-drosophila-suzukii-turing-wavelength-s0.py --self-test
python3 models/2026-delphi10000-drosophila-suzukii-turing-wavelength-s0.py
```

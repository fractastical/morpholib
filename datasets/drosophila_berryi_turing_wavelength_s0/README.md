# Drosophila berryi — detail-scale Turing wavelength (loop 3173)

**Animal.** *Drosophila berryi*, Miocene Dominican amber fruit fly (Hymenaea inclusion), II. Anaima (Invertebrata) → Entoma (Arthropoda) → *Drosophila*.

**Age.** Early Miocene, Burdigalian Dominican amber, **19 Ma**. Chronoseries position 2/41 on the lineage *Cenozoic Amber Drosophilid Chronospecies (44 Ma Eocene → 0 Ma)*. Predecessor *Drosophila succini* (44 Ma, Middle Eocene Baltic amber). Successor *Drosophila affinis* (0 Ma, Holocene).

**Scale.** Detail variant `s0` of the Turing pigment / ocellus wavelength, tier `3_subcellular_um`. The assigned phenomenon size is **1.8 µm** exactly **1.754×10⁻⁶ m** (log10 = −5.756). This is not the tissue-scale stripe of the same species (98 µm, slot 3081). It is the subcellular wavelength selected by the Turing wavenumber \(k_c\).

**Law.** Turing reaction–diffusion wavelength selection (cohort 1, chemical Turing / RD). Form: stripes, spots and ocelli. Quantitative law: critical wavenumber

\[
k_c^2 = \frac{D_v f_u + D_u g_v}{2 D_u D_v}, \qquad \lambda_c = \frac{2\pi}{k_c}.
\]

Kinetics are the Schnakenberg (1979) pair, with a Jacobian evaluated at the homogeneous steady state as in Turing (1952). The spatial unit is calibrated so the classical \(\lambda_c\) is 1.754 µm on the berryi metric. The fastest-growing mode \(\lambda_\sigma\) sits slightly longward of \(\lambda_c\) because the trace penalty \(-k^2(D_u+D_v)\) shifts the peak of \(\mathrm{Re}\,\sigma(k)\).

**What changes at this scale and age.** The subcellular mode count per tissue stripe is held fixed at \(N = 98/1.754 \approx 55.87\). Absolute wavelength tracks the catalogued tissue stripe (114 → 98 → 96 µm):

| chronospecies | age | tissue \(\lambda\) | detail \(\lambda_c\) | \(k_c\) |
|---|---:|---:|---:|---:|
| *D. succini* (predecessor) | 44 Ma | 114 µm | 2.040 µm | 3.080×10⁶ m⁻¹ |
| *D. berryi* (this slot) | 19 Ma | 98 µm | **1.754 µm** | **3.582×10⁶ m⁻¹** |
| *D. affinis* (successor) | 0 Ma | 96 µm | 1.718 µm | 3.657×10⁶ m⁻¹ |

From succini to berryi the detail wavelength shortens by 0.286 µm (−14.0%). From berryi to affinis it shortens by 0.036 µm (−2.1%). Physical diffusivities scale with the square of that metric, so the berryi inhibitor diffusivity is \((1.754/2.040)^2 \approx 0.74\) of the succini value. The nondimensional Schnakenberg parameters \((a,b,D_u,D_v)\) stay inside one Turing band; the chronospecies change is the epithelial metric, not a new reaction scheme.

**Specimen.** `Drosophila_berryi_specimen.jpg` is the project gallery asset (SHA-256 pinned in the reference JSON). It is an amber-toned spiral plate with a 100 µm scale bar, not a published scanning micrograph of 1.8 µm pigment granules. Grimaldi (1987, *American Museum Novitates* 2880) is the amber drosophilid monograph; it does not report this wavelength.

**Run.**

```bash
python3 models/2026-delphi10000-drosophila-berryi-turing-wavelength-s0.py --self-test
python3 models/2026-delphi10000-drosophila-berryi-turing-wavelength-s0.py
```

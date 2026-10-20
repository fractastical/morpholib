# `datasets/limb_bud_elongation_pushing/` — Boehm et al. (2010) & Gros et al. (2010) Vertebrate Limb Bud Elongation Reference Package

Formal SHA-256-locked dataset and biological specimen image for `2015-boehm-limb-bud-elongation-pushing.py` (Slot 017, `boehm-limb-bud-elongation-pushing`).

## Contents

1. **`limb_bud_elongation_pushing_reference.json`**
   - **Boehm et al. (2010) *PLoS Biology* 8(7):e1000420 (`doi:10.1371/journal.pbio.1000420`)**:
     - Refutation of the 40-year "Proliferation Gradient Hypothesis" (growth-based morphogenesis):
       $$\nabla \cdot \mathbf{v} = s(\mathbf{x}) = \frac{\ln 2}{T_c(\mathbf{x})} - s_d, \qquad -\nabla p + \frac{1}{\text{Re}} \nabla^2 \mathbf{v} = \mathbf{0}$$
     - Dual-pulse IddU/BrdU cell-cycle time atlas across 30 3D regions ($T_i = 2.5\text{ h}$): $T_c(\mathbf{x})$ ranges from $11.0\text{ h}$ distally under the AER ($s_p = 0.063\text{ h}^{-1}$) to $25.0\text{ h}$ proximally ($s_p = 0.028\text{ h}^{-1}$) with density compaction $s_d = 0.017\text{ h}^{-1}$.
     - Across 5 orders of magnitude in Reynolds number ($\text{Re} \in [10^{-6}, 10^{-1}]$), pure isotropic proliferation produces only $3.4\%\text{--}11.8\%$ PD elongation over $6\text{ h}$ (aspect ratio $L_{\text{PD}}/W_{\text{AP}} \approx 1.07$, bulbous expansion) versus empirical $+42.2\%$ ($1020\,\mu\text{m} \to 1450\,\mu\text{m}$ from E11.0 $S_{t0}$ to E11.25 $S_{t1}$).
     - Inverse 525-vertex Hooke–Jeeves parameter optimization proves that matching $S_{t1}$ via isotropic source rates alone requires unphysical distal proliferation $s > 0.50\text{ h}^{-1}$ ($T_c < 1.5\text{ h}$) coupled with negative tissue shrinkage ($s < 0$) across $22.5\%$ of the limb bud volume.
   - **Gros et al. (2010) *Current Biology* 20(22):1993–2004 (`doi:10.1016/j.cub.2010.09.063`) & Kicheva & Briscoe (2010) *PLoS Biology* 8(7):e1000421 (`doi:10.1371/journal.pbio.1000421`)**:
     - Anisotropic active stress $\boldsymbol{\sigma}^{\text{active}} = \zeta (\mathbf{n}\otimes\mathbf{n} - \frac{1}{3}\mathbf{I})$ from Wnt5a/JNK-polarized oriented cell division and convergent-extension intercalation plus AER-FGF8/MAPK distal velocity gradient drives proximodistal limb bud elongation ($L_{\text{PD}}/W_{\text{AP}} > 2.1$ from E9.5 to E11.5), whereas $Wnt5a^{-/-}$ mutants form broad, short bulbous limb buds ($L_{\text{PD}}/W_{\text{AP}} \approx 1.32$).

2. **`mouse_limb_bud_specimen.jpg`**
   - Optical projection tomography (OPT) 3D iso-surfaces, brightfield micrographs, and 3D tetrahedral FEM mesh of *Mus musculus* right hindlimb buds at E11.0 ($S_{t0}$) and E11.25 ($S_{t1}$, $6\text{ h}$ later) from Boehm et al. (2010) *PLoS Biology* 8(7):e1000420 Figure 2 (CC BY 4.0).
   - SHA-256: `50fa6fde96e75c636ddee764951cbca3cff015903f3e0b7cc45a0146e2809f2e`.

3. **`scout_manifest.json`**
   - Step 1 literature & specimen scout dossier for Slot 017.
   - SHA-256: `8e7dc46a51c711bedf370d63c132f004f188810ef1fae6cdaa555d970f2083b3`.

4. **`paper_citations.bib`**
   - BibTeX entries for Boehm et al. (2010), Kicheva & Briscoe (2010), Gros et al. (2010), Wyngaarden et al. (2010), Morishita & Iwasa (2008), and Dillon & Othmer (1999).
   - SHA-256: `a3191cedf6d442125515ef89b54c11bb72ebdbb395c10b4f06befaeb90174a94`.

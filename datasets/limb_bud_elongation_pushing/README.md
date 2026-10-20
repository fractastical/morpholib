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

## Pass 3 (2026-10-19) — specimen-derived 3D visual hull and tetrahedral Stokes FEM

The Pass 2 audit's single open item asked for the 2D mid-plane pressure-Poisson solver to be extended to a 3D tetrahedral finite-element mesh imported from OPT iso-surfaces. The raw OPT voxel volumes of Boehm et al. (2010) are **not** vendored (only Fig. 2 is, as `mouse_limb_bud_specimen.jpg`), so Pass 3 imports the geometry the honest way that *is* possible from this package: the $S_{t0}$ and $S_{t1}$ iso-surface renderings of Fig. 2F are segmented in both orthogonal views (dorsal AP–PD and posterior DV–PD) and the 3D surface is reconstructed as their **visual hull** (elliptical cross-sections with AP semi-axis $a(x)$ from the dorsal silhouette and DV semi-axis $b(x)$ from the posterior silhouette). Pixel scale is fixed per view by the OPT $S_{t0}$ length $L_{PD}=1020\,\mu$m. All numbers below are *measured from the specimen at run time* by `reconstruct_opt_visual_hull()`; nothing here is copied from the paper.

| Quantity (specimen Fig. 2F, visual hull) | $S_{t0}$ (E11.0) | $S_{t1}$ (E11.25) | ratio | OPT reference table |
|---|---:|---:|---:|---|
| $L_{PD}$ (dorsal view) | 1020 µm (scale anchor) | 1391 µm | ×1.364 | 1020 → 1450 µm (×1.422) |
| $L_{PD}$ (posterior view, independent) | 1020 µm | 1386 µm | ×1.359 | — |
| $W_{AP}$ = 2 max $a(x)$ | 1071 µm | 1308 µm | ×1.222 | 960 → 1075 µm (×1.120) |
| $H_{DV}$ = 2 max $b(x)$ | 926 µm | 936 µm | ×1.010 | 580 → 586 µm (×1.010) |
| dorsal px → µm | 5.455 | (same) | | |
| posterior px → µm | 5.543 | (same) | | |

Note the two discrepancies this exposes and that the v3 audit carries as Pass 4 items: the specimen's $S_{t1}/S_{t0}$ PD ratio (1.36) is below the table's 1.42, and the specimen DV thickness (926 µm) is far above the table's 580 µm (the table value may be a mid-bud rather than maximal thickness; to be checked against the paper's text).

### 3D FEM protocol (`2015-boehm-limb-bud-elongation-pushing.py`, Section 5)

- Mesh: structured prism-split P1 tetrahedral mesh of the $S_{t0}$ hull, $(n_{PD}, n_{rings}, n_\theta) = (14, 3, 10)$ → 466 nodes / 2150 tets (coarse 188/744, fine 1198/5978 for convergence; Model I prediction changes by 0.018 pp between default and fine).
- Physics: $-\nabla p + \mu\nabla^2\mathbf v + \nabla\cdot\boldsymbol\sigma_{active} + \mathbf f_{FGF8} = 0$, $\nabla\cdot\mathbf v = s(\mathbf x)$ with $s=\ln 2/T_c(\mathbf x) - s_d$ from the IddU/BrdU atlas; extensile Wnt5a/JNK stress $\boldsymbol\sigma_{active} = -2\mu\,k_\zeta\zeta\,w(x)\,(\mathbf e_{PD}\otimes\mathbf e_{PD} - \mathbf I/3)$; AER-FGF8 body force $\mathbf f = 2\mu\,k_\chi\chi\,\partial_x c_{FGF8}\,\mathbf e_{PD}$; traction-free ectoderm; $v_x = v_{in}$ on the flank plane ($v_{in}=0$ pins it; $v_{in}>0$ is an ALE inflow plane). P1–P1 Brezzi–Pitkäranta stabilisation; Lagrangian RK2, $\Delta t = 0.5$ h.
- Verification: two closed-form Stokes solutions (uniform dilation; uniaxial extensile response) reproduced to $8\times10^{-14}$ relative error (Lean Theorem 4 `stokes_growth_oracles_are_exact`); volume balance $\Delta V/\iint s\,dV\,dt = 1.005$ (Lean Theorem 5 `flank_influx_volume_budget`).
- Calibration: exactly two gains fitted — $k_\chi = 0.1114\ \mathrm{h^{-1}}$ to the FGF-only regime (29.8 %) and $k_\zeta = 0.0352\ \mathrm{h^{-1}}$ to WT (42.2 %); Wnt5a⁻/⁻, Model I and the uniform control are out-of-sample.

### Headline FEM results (6 h, E11.0 → E11.25)

| Scenario | PD | AP | DV | $V_1/V_0$ | dorsal IoU vs specimen $S_{t1}$ |
|---|---:|---:|---:|---:|---:|
| Model I — measured $s(x)$ only, flank pinned (parameter-free) | +5.2 % | +4.4 % | +2.5 % | 1.123 | 0.683 |
| Uniform control $T_c = 16.5$ h | +5.1 % | +5.1 % | +5.1 % | 1.162 | — |
| Wnt5a⁻/⁻ ($\zeta=0.12,\chi=0.25$; out-of-sample, ref 18.5 %) | +12.0 % | +1.0 % | −1.6 % | 1.124 | — |
| FGF-only (calibrated to 29.8 %) | +29.7 % | −6.1 % | −12.3 % | 1.126 | — |
| Model II WT active stress (calibrated to 42.2 %) | +42.2 % | **−10.9 %** | **−14.5 %** | 1.125 | 0.770 |
| Model I + flank influx $v_{in}^* = 61.4$ µm/h (closes OPT volume ×1.659) | +42.4 % | +4.9 % | +2.8 % | 1.659 | **0.847** |
| OPT reference table | +42.2 % | +12.0 % | +1.0 % | 1.659 | 1 |

Reading: the paper's Model I conclusion (isotropic measured proliferation gives a bulbous bud with only a few % PD elongation) is reproduced parameter-free and inside the paper's 3.4–11.8 % band. But the measured source supplies only 12 % of volume against the 66 % the OPT table reports, so an incompressible active-stress "pushing" model that hits +42 % PD must contract AP and DV — the opposite of the specimen. Closing the volume budget with body-wall inflow through the flank reproduces PD, the signs of AP/DV and the best silhouette match with **no** anisotropic stress. Pass 4 must (i) verify the OPT volume table against the paper, (ii) combine influx with a reduced active stress to fit AP/DV, and (iii) replace the visual hull with true OPT voxel data if it can be obtained.

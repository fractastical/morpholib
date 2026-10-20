# Compressible Viscous Stokes–Brinkman Global Morphogenetic Flow & *Drosophila* Germband Extension (`datasets/global_morphogenetic_flow/`)

Reference parameters, formal 2D compressible Stokes–Brinkman active nematic stress tensor equations, Streichan et al. (2018) wild-type and mutant regimes, and whole-embryo confocal microscopy for **Slot 048 (`slot-048-global-morphogenetic-flow`, `streichan-global-morphogenetic-flow`)** of the Infinite Morphospace & MorphoEval Campaign, reproducing:

- **Streichan, S. J., Lefebvre, M. F., Noppe, N., Hufnagel, L. & Shraiman, B. I. (2018).** *"Global morphogenetic flow is accurately predicted by the spatial distribution of myosin."* *eLife* 7:e27454. DOI: `10.7554/eLife.27454`
- **Mayer, M., Depken, M., Bois, J. S., Jülicher, F. & Grill, S. W. (2010).** *"Anisotropies in cortical tension reveal the physical basis of polarizing cortical flows."* *Nature* 467(7315):617–621. DOI: `10.1038/nature09376`
- **Lefebvre, M. F., Claussen, N. H., Mitchell, N. P., Gustafson, H. J. & Streichan, S. J. (2023).** *"Geometric control of myosin II orientation during axis elongation."* *eLife* 12:e78787. DOI: `10.7554/eLife.78787`
- **Brauns, F., Claussen, N. H., Lefebvre, M. F., Wieschaus, E. F. & Shraiman, B. I. (2024).** *"The geometric basis of epithelial convergent extension."* *eLife* 13:RP95521. DOI: `10.7554/eLife.95521`

## Governing Mathematical Laws

1. **Positive-Definite Compressible Stokes–Brinkman Fourier Operator Invertibility (`stokes_brinkman_fourier_operator_pos_def`):**
   \[
   \eta \nabla^2 \mathbf{v} + \eta_b \nabla(\nabla \cdot \mathbf{v}) - \xi \mathbf{v} = -\nabla \cdot \boldsymbol{\sigma}^a, \qquad \ell_H = \sqrt{\frac{\eta}{\xi}} \approx 180\,\mu\text{m}.
   \]
   In Fourier space, the longitudinal mode eigenvalue \(\lambda_L(k) = (\eta + \eta_b) k^2 + \xi\) and transverse shear mode eigenvalue \(\lambda_T(k) = \eta k^2 + \xi\) are strictly positive for all wavenumbers \(k \in \mathbb{Z}^2\) whenever \(\eta > 0, \eta_b \ge 0, \xi > 0\), guaranteeing unique global velocity inversion across the embryonic surface.
2. **Trace-Free Active Nematic Myosin Stress Tensor & Quadrupolar Convergent Extension (`nematic_dv_cable_trace_free_and_ap_extension`):**
   \[
   \sigma^a_{ij}(\mathbf{x}) = \alpha_{\text{iso}}\, m_{\text{iso}}(\mathbf{x})\,\delta_{ij} + \alpha_{\text{aniso}}\, m_{\text{aniso}}(\mathbf{x})\, Q_{ij}(\mathbf{x}), \qquad \mathrm{Tr}(\mathbf{Q}) = Q_{xx} + Q_{yy} = 0.
   \]
   Dorsoventral (DV) polarized Myosin-II cables (\(\hat{\mathbf{n}} = \hat{\mathbf{e}}_{\text{DV}}\)) have \(Q_{yy} > 0\) and \(Q_{xx} = -Q_{yy} < 0\), simultaneously driving DV convergence (\(\partial_y v_y < 0\)) and AP germband extension (\(\partial_x v_x > 0\)).
3. **Streichan et al. (2018) 6-Regime Global Flow & Maternal/Pair-Rule Mutant Hierarchy (`streichan_2018_global_flow_and_mutant_collapse`):**
   Wild-type anisotropic DV cables + posterior midgut (PMG) invagination sink achieves \(>92\%\) spatial velocity correlation (\(v_{\text{AP}} = 6.40\,\mu\text{m/min}\), GBE extension ratio \(1.92\times\)), whereas maternal AP-null triple mutants (`bcd nos tsl`) have \(Q_{ij} = 0\) and zero posterior germband extension flow.

## Vendored Files & Cryptographic Digests

- `scout_manifest.json` — OpenAlex + Wikimedia Commons literature & specimen scout dossier (`990f3ea83019f3d1cb7cbbdb0fc28fd80fddd21ee604989871060517be33b473`)
- `global_morphogenetic_flow_reference.json` — Formal Stokes–Brinkman parameters, active nematic tensor laws, and 6-regime Streichan et al. (2018) dataset
- `drosophila_germband_extension_specimen.jpg` — Whole-embryo confocal micrograph of *Drosophila melanogaster* during early gastrulation and germband extension (`2d7fad8f57edcadf3ccc80636e45dee9187f09d0f108595b12f50df040c63e39`, CC BY 4.0, IDR0151 / Clark et al. 2022 *eLife* 11:e78902, Wikimedia Commons)
- `paper_citations.bib` — BibTeX citations for Streichan et al. (2018), Mayer et al. (2010), Lefebvre et al. (2023), and Brauns et al. (2024)

## Pass 3 additions (audit v3, `intercalation_reference.json`)

- `intercalation_reference.json` — prolate-spheroid blastoderm geometry (\(a = 250\,\mu\text{m}\), \(b = 105\,\mu\text{m}\), metric \(g_{ss} = a^2\sin^2 s + b^2\cos^2 s\), \(g_{\phi\phi} = b^2 \sin^2 s\), Gaussian curvature \(K = a^2/g_{ss}^2\)), the curved-surface Stokes–Brinkman dissipation functional and its sphere-limit eigenvalue \((\eta+\eta_b)\,l(l+1)/R^2 - 2\eta/R^2 + \xi\), the Lefebvre et al. (2023)-type strain-rate positive-feedback recruitment parameters (\(\beta = 0.35\), \(e_{\text{ref}} = 0.06\,\text{min}^{-1}\)), and the T1 / rosette intercalation ledger anchors (rosette order \(r = k + 3\) cells for \(k\) collapsed DV junctions, 5–11 cells per Blankenship et al. 2006; 5–10 min junction shrinkage per Bertet et al. 2004; intercalation share ≈ 0.5–0.8 of fast-phase extension per Brauns et al. 2024). SHA-256 `7d0cd54341a21cc07b8a127ab5a4562d5a02008d807c05e3bf17ffe616490cac`, pinned in the script.
  **Caveat:** the entries flagged `verified_offline: false` were entered from recollection of the cited papers during an offline loop and the ledger rate constants were *calibrated* to those windows; they are anchors, not digitised data.
- Additional citations in `paper_citations.bib`: Bertet, Sulak & Lecuit (2004) *Nature* 429:667–671 (`10.1038/nature02590`); Blankenship et al. (2006) *Dev. Cell* 11:459–470 (`10.1016/j.devcel.2006.09.007`).
- New Lean 4 theorems (Loop_048): `surface_stokes_brinkman_gradient_mode_pos`, `myosin_feedback_fixed_point`, `rosette_resolution_junction_balance`.

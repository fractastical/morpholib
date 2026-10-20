# `datasets/branching_random_walk_annihilation/` — Hannezo et al. (2017) *Cell* & Scheele et al. (2017) *Nature* Unifying Theory of Branching Morphogenesis (BARW) Dataset

Formal SHA-256-locked reference dataset and *Mus musculus* branched epithelial ductal network whole-mount fluorescence micrograph for `2017-hannezo-branching-random-walk-annihilation.py` (Slot 056 of the Infinite Morphospace & MorphoEval Campaign).

## Governing Mathematical Mechanism

1. **Hannezo et al. (2017) *Cell* 171(1):242–255.e27 (`doi:10.1016/j.cell.2017.08.026`) & Scheele et al. (2017) *Nature* 542:313–317 (`doi:10.1038/nature21060`)**:
   - **Stochastic Branching and Annihilating Random Walk (BARW)** rules governing active proliferative distal tips ($A$) and differentiated inactive ducts ($I$):
     $$A \xrightarrow{k_b} 2A \quad (\text{stochastic tip bifurcation}), \qquad A \xrightarrow{v_0} A + I \quad (\text{persistent ductal trail deposition}), \qquad A + I \xrightarrow{R_a} I \quad (\text{tip-duct steric/TGF-}\beta\text{ annihilation})$$
   - **Mean-Field Travelling-Wave Reaction-Diffusion PDEs**:
     $$\frac{\partial a}{\partial t} = D \nabla^2 a - \mathbf{v}_0 \cdot \nabla a + k_b a - k_a a (a + i), \qquad \frac{\partial i}{\partial t} = v_0 a$$
     yielding an invading Fisher–KPP front wave of active tips $a(x, t)$ at speed $c^* \ge 2\sqrt{D k_b}$ that leaves behind a self-organized, space-filling homeostatic duct density:
     $$a_\infty = 0, \qquad i_\infty = \frac{c_0 k_b}{k_a} \propto \frac{k_b}{v_0 R_a}$$
   - **Topological Binary-Tree Balance & Critical Galton–Watson Universality**:
     In any finite self-terminating binary ductal tree initiated from a single primary stalk, the number of terminal tips $N_{\text{term}}$ and internal branch nodes $N_{\text{branch}}$ obey $N_{\text{term}} = N_{\text{branch}} + 1$, pinning the ensemble branching probability to the critical point:
     $$q_{\text{branch}} = \frac{N_{\text{branch}}}{N_{\text{branch}} + N_{\text{term}}} = \frac{N_{\text{branch}}}{2 N_{\text{branch}} + 1} \xrightarrow{N_{\text{branch}} \to \infty} \frac{1}{2} = q_{\text{term}}$$
     and generating parameter-free critical Galton–Watson subtree size distributions $P(S) \sim S^{-3/2}$ across mouse mammary gland, metanephric kidney, and embryonic pancreas.

## Files

- `branching_random_walk_annihilation_reference.json` — Governing BARW parameters, mean-field PDE constants, three-organ universality benchmarks (mammary gland, kidney, pancreas), generational branching probability profiles, and TGF-$\beta$ / $R_a$ perturbation benchmarks.
- `mouse_branched_ductal_network_specimen.jpg` — *Mus musculus* embryonic branched glandular epithelial network whole-mount fluorescence micrograph showing cyan/green active distal buds ($A$, $N_{\text{term}} = 34$ tips localized at normalized radius $r / R_{\max} = 0.783$) and blue/magenta differentiated ductal trails ($I$, $N_{\text{branch}} = 33$ internal bifurcations, $q_{\text{branch}} = 33 / 67 = 0.4925$), Wikimedia Commons CC BY 4.0 (`sha256: d8946f47dcc4d313d06963cbbb37786bc792fe27523a0c851adf64692a047d60`).
- `paper_citations.bib` — BibTeX references with DOIs (Pass 3 adds Moran 1958 and Klein & Simons 2011; those two DOIs were not verified offline).
- `scout_manifest.json` — Step 1 (`01_scout`) literature and specimen manifest.
- `barw_3d_ecm_clonal_reference.json` — **Pass 3** reference table (SHA-256 `c716e3a9cce08d6f37db10917d5c6ded869e596982a4c863c0581c4d5718e2ae`, pinned in the script): 3D cylindrical fat-pad and spherical (optionally inflating) mesenchymal geometry parameters, the ECM fiber-alignment sweep $\lambda \in \{0, 0.05, 0.1, 0.2, 0.4\}$ with the nematic-order reference values ($S_{\text{iso}} = 0$, $S_{\parallel} = 1$), the inflation-rate sweep (Bordeu et al. 2023), and the clonal neutral-drift parameters and reference values ($N_{\text{TEB}} = 64$, $f_0 = 0.125$ martingale, clone-size tail exponent $3/2$). The unsourced free parameters ($N_{\text{TEB}}$, Moran rate) are listed explicitly in the file.

## Pass 3 mechanisms (added 2026-10-19)

1. **3D BARW with anisotropic ECM (Bordeu et al. 2023, `doi:10.1038/s41467-023-39124-x`)** — tips move as a persistent random walk on the unit sphere with a nematic ECM torque
   $$\dot{\mathbf{u}} = \sigma_\theta\,\boldsymbol{\xi}_\perp + \lambda\,(\mathbf{f}\cdot\mathbf{u})\,\big(\mathbf{f} - (\mathbf{f}\cdot\mathbf{u})\mathbf{u}\big),$$
   inside a cylindrical fat pad ($R_c = 14$, $L_c = 110$) or a spherical capsule $R_s(t) = R_s(0)(1 + \gamma t)$. Alignment is quantified by $S = \langle (3\cos^2\theta - 1)/2 \rangle$. $N_{\text{term}} = N_{\text{branch}} + 1$ holds in every geometry. Over-alignment ($\lambda \ge 0.4$) collapses the tree because realigned sibling tips fall within $R_a$ of each other and annihilate.
2. **Clonal neutral drift in TEBs (Scheele et al. 2017, `doi:10.1038/nature21060`)** — each TEB carries $k$ labelled of $N$ equipotent stem cells; Moran steps $k \to k \pm 1$ with probability $k(N-k)/N^2$ along each duct, hypergeometric half-split and doubling at every $A \to 2A$, differentiation at $A + I \to I$. The duct-length-weighted labelled fraction is an unbiased estimator of $f_0$ (Lean Theorems 5–6), label loss and clonal variance grow with generation, and surviving single-cell clones span critical Galton–Watson subtrees (tail exponent compared with $3/2$).

Supplementary figures: `docs/media/branching_random_walk_annihilation_3d_ecm_fatpad.png`, `docs/media/branching_random_walk_annihilation_clonal_drift.png`.

# `datasets/kidney_ureteric_bud_gdnf_ret/` — Menshykau & Iber (2013), Zubkov et al. (2015) & Watanabe & Costantini (2004) Kidney Ureteric Bud GDNF-RET-Wnt11 Turing Branching Dataset

Formal SHA-256-locked reference dataset and mouse metanephric ureteric bud whole-mount fluorescence specimen for `2012-menshykau-fgf10-shh-lung-turing.py` (Slot 054 of the Infinite Morphospace & MorphoEval Campaign).

## Governing Mathematical Mechanism

1. **Menshykau & Iber (2013) *Physical Biology* 10(4):046003 (`doi:10.1088/1478-3975/10/4/046003`)**:
   - Schnakenberg-type receptor-ligand reaction-diffusion system coupling epithelial RET receptor ($R$, diffusivity $D_R \ll D_G$) and mesenchymal GDNF ligand ($G$, diffusivity $D_G$):
     $$\frac{\partial R}{\partial t} = D_R \nabla^2 R + \gamma \left(a - R + R^2 G\right), \qquad \frac{\partial G}{\partial t} = D_G \nabla^2 G + \gamma \left(b - R^2 G\right)$$
   - Positive cooperative receptor-ligand signaling ($R^2 G$) represents homodimeric GDNF binding to two RET/GFR$\alpha$1 receptor complexes with Wnt11-mediated mesenchymal feedback and Sprouty1 (*Spry1*) intracellular RTK/ERK antagonism.
   - Unique homogeneous steady state $(R^*, G^*) = (a + b,\; b / (a + b)^2)$ with exact reaction-rate sum cancellation $f(R, G) + g(R, G) = a + b - R$ and Jacobian determinant $\det(J_0) = (a + b)^2 > 0$.

2. **Zubkov et al. (2015) *Bull. Math. Biol.* 77:565–595 (`doi:10.1007/s11538-015-0073-9`) & Watanabe & Costantini (2004) *Dev. Biol.* 271:98–106 (`doi:10.1016/j.ydbio.2004.03.025`)**:
   - Dichotomous ($m = 2$) vs trichotomous ($m = 3$) ureteric bud tip splitting as a function of the normalized ampulla arc-to-wavelength ratio $\rho_{\text{tip}} = \pi R_{\text{tip}} / \lambda_c$.
   - Self-avoiding ureteric tree arborization driven by local mesenchymal GDNF consumption sinks ($-\gamma R^2 G$), reproducing Horton-Strahler bifurcation ratio $R_b \approx 2.04$ across E11.5–E13.5 mouse metanephros development.
   - Genetic and bead perturbation benchmarks: Wild-Type control, $Gdnf^{+/-}$ / $PRR^{UB-/-}$ hypoplasia, $Spry1^{-/-}$ hyper-branching (Basson et al. 2005 *Dev. Cell* 8:229–239), and exogenous GDNF-soaked bead supernumerary budding.

## Files

- `kidney_ureteric_bud_gdnf_ret_reference.json` — Governing PDE parameters, tip-splitting mode thresholds, E11.0–E13.5 branching generation table, and perturbation benchmarks.
- `kidney_ureteric_tree_specimen.jpg` — E12.5 mouse metanephros GFP ureteric bud branching (Panel A wild-type 12 tips vs Panel B $PRR^{UB-/-}$ hypoplastic mutant 6 tips) and pErk1/2 immunofluorescence (Panels D–I), Wikimedia Commons CC BY 4.0 (`sha256: 3264f509ba3a668690c5900ddd7dc47ae028cc26c7c3f3c6f308b7de67ec76bc`).
- `paper_citations.bib` — BibTeX references with DOIs.
- `scout_manifest.json` — Step 1 (`01_scout`) literature and specimen manifest.

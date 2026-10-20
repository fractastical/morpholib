# Anisotropic Differential Adhesion & Polarized T1 Intercalation in Convergent Extension (`datasets/convergent_extension_intercalation/`)

Reference parameters, formal vertex/adhesion equations, experimental Wnt/PCP and Myosin II perturbation tables, and biological specimen photography for **Slot 093 (`slot-093-convergent-extension-intercalation`)** of the Infinite Morphospace & MorphoEval Campaign, reproducing:

- **Zajac, M., Jones, G. L. & Glazier, J. A. (2003).** *"Simulating convergent extension by way of anisotropic differential adhesion."* *Journal of Theoretical Biology* 222(2):247–259. DOI: `10.1016/S0022-5193(03)00033-X` (`10.1006/jtbi.2003.3177`)
- **Zajac, M., Jones, G. L. & Glazier, J. A. (2000).** *"Model of convergent extension in animal morphogenesis."* *Physical Review Letters* 85(9):2022–2025. DOI: `10.1103/PhysRevLett.85.2022`
- **Bertet, C., Sulak, L. & Lecuit, T. (2004).** *"Myosin-dependent junction remodelling controls planar cell intercalation and axis elongation."* *Nature* 429(6992):667–671. DOI: `10.1038/nature02590`
- **Wallingford, J. B., Rowning, B. A., Vogeli, K. M., Rothbächer, U., Fraser, S. E. & Harland, R. M. (2000).** *"Dishevelled controls cell polarity during Xenopus gastrulation."* *Nature* 405(6782):81–85. DOI: `10.1038/35011077`

## Governing Mathematical Laws

1. **Zajac, Jones & Glazier (2000, 2003) Anisotropic Differential Adhesion & Bertet, Sulak & Lecuit (2004) Polarized T1 Junction Tension Law:**
   \[
   J(\theta) = J_0 - \Delta J \cos(2\theta), \qquad
   \Lambda(\theta) = \Lambda_0 - \Delta\Lambda \cos(2\theta),
   \]
   so vertical/AP-facing junctions (\(\theta = \pi/2\), separating mediolateral neighbors) carry elevated tension \(\Lambda_{\text{vert}} = \Lambda_0 + \Delta\Lambda\) compared to horizontal/ML junctions (\(\theta = 0\), \(\Lambda_{\text{horiz}} = \Lambda_0 - \Delta\Lambda\)), yielding a strictly negative free-energy drop across an irreversible T1 quartet transition:
   \[
   \Delta E_{\text{T1}}(l_0) = (\Lambda_{\text{horiz}} - \Lambda_{\text{vert}}) l_0 = -2\,\Delta\Lambda\,l_0 < 0 \quad (\Delta\Lambda > 0,\, l_0 > 0).
   \]
2. **Incompressible 2D Pure-Shear Tissue Area Conservation & Aspect Ratio Exponential Scaling Law:**
   \[
   \frac{dL_{\text{AP}}}{dt} = +\dot{\varepsilon}_{\text{T1}} L_{\text{AP}}, \qquad
   \frac{dL_{\text{ML}}}{dt} = -\dot{\varepsilon}_{\text{T1}} L_{\text{ML}}, \qquad
   A_{\text{tissue}}(t) = L_{\text{AP}}(t) L_{\text{ML}}(t) = A_0, \qquad
   \frac{L_{\text{AP}}(t)}{L_{\text{ML}}(t)} = \frac{L_{\text{AP},0}}{L_{\text{ML},0}} \exp\!\left(2\dot{\varepsilon}_{\text{T1}} t\right).
   \]
3. **Zajac et al. (2003) & Bertet et al. (2004) Five-Regime Perturbation Hierarchy:**
   Wild-type polarized T1 intercalation (\(L_{\text{AP}}/L_{\text{ML}} = 3.86\), \(N_{\text{T1}}/N_{\text{cell}} = 1.45\)) > Moderate anisotropy hypomorph (\(L_{\text{AP}}/L_{\text{ML}} = 2.38\)) > Wnt/PCP *Dishevelled* (*Xdd1* / *stbm*) polarity-scrambled mutant (\(L_{\text{AP}}/L_{\text{ML}} = 1.56\)) > Myosin II (*zip* / *sqh* / Y-27632) junction-remodelling defective mutant (\(L_{\text{AP}}/L_{\text{ML}} = 1.20\)) > Isotropic adhesion / *bcd;nos;tsl* negative control (\(\Delta\Lambda = 0\), \(L_{\text{AP}}/L_{\text{ML}} = 1.00\), \(N_{\text{T1}} = 0\)).

## Vendored Files & Cryptographic Digests

- `scout_manifest.json` — OpenAlex + Wikimedia Commons literature & specimen scout dossier
- `convergent_extension_intercalation_reference.json` — Formal anisotropic adhesion / polarized T1 vertex parameters and 5-regime perturbation dataset
- `convergent_extension_axis_specimen.jpg` — Confocal fluorescence micrographs and directional rose diagrams of *Xenopus laevis* gastrula explants comparing isotropic animal cap cells (Panel a, a': \(L/W = 0.962\), nematic \(S_2 = 0.094\)) against mediolaterally polarized chordamesoderm cells undergoing convergent extension (Panel b, b': \(L/W = 4.945\), nematic \(S_2 = 0.613\)) (`0c2620bce19ad4cf8602ea8cead0e85ae4b816e7b9e3e8b0441122a4c906d61d`, CC BY 4.0, Wikimedia Commons)
- `paper_citations.bib` — BibTeX citations for Zajac et al. (2000, 2003), Bertet et al. (2004), Keller et al. (1992), and Wallingford et al. (2000)

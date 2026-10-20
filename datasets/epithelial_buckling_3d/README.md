# Dataset: 3D Epithelial Sheet Morphology — Squamous-to-Cuboidal-to-Columnar Scaling & Apico-Basal Buckling (`epithelial_buckling_3d`)

## Primary Literature & DOIs

1. **Hannezo E, Prost J, Joanny J-F (2014)**.
   *Theory of epithelial sheet morphology in three dimensions.* **Proceedings of the National Academy of Sciences** 111(1):27–32.
   DOI: [`10.1073/pnas.1321766111`](https://doi.org/10.1073/pnas.1321766111)
2. **Krajnc M, Ziherl P (2015)**.
   *Theory of epithelial elasticity.* **Physical Review E** 92(5):052713.
   DOI: [`10.1103/PhysRevE.92.052713`](https://doi.org/10.1103/PhysRevE.92.052713)
3. **Štorgel N, Krajnc M, Mrak P, Štrus J, Ziherl P (2016)**.
   *Quantitative Morphology of Epithelial Folds.* **Biophysical Journal** 110(1):269–277.
   DOI: [`10.1016/j.bpj.2015.11.024`](https://doi.org/10.1016/j.bpj.2015.11.024)
4. **Sui L, Alt S, Weigert M, Dye N, Eaton S, Jülicher F, Dahmann C (2018)**.
   *Differential lateral and basal tension drive folding of Drosophila wing discs through two distinct mechanisms.* **Nature Communications** 9:4620.
   DOI: [`10.1038/s41467-018-06497-3`](https://doi.org/10.1038/s41467-018-06497-3)
5. **Gómez-Gálvez P, Vicente-Munuera P, Tagua A, et al., Escudero LM (2018)** *(added in Pass 3)*.
   *Scutoids are a geometrical solution to three-dimensional packing of epithelia.* **Nature Communications** 9:2960.
   DOI: [`10.1038/s41467-018-05376-1`](https://doi.org/10.1038/s41467-018-05376-1) — DOI not re-resolved online in the offline Pass 3 round; it matches the entry vendored by Slot 029.

## Governing Mathematical & Biomechanical Framework

In Hannezo, Prost & Joanny (2014) *PNAS* 111(1):27–32 and Krajnc & Ziherl (2015) *Phys. Rev. E* 92:052713, an epithelial monolayer is modeled as a 3D incompressible cellular sheet of conserved single-cell volume $V_0 = A h$ (where $A = d^2$ is the apical/basal cross-sectional area and $h$ is the apico-basal cell height) governed by apical surface tension $\Gamma_a$, basal surface tension $\Gamma_b$, lateral cell-cell interfacial tension $\Gamma_l$, and elastic substrate stiffness $K_{\text{sub}}$:

$$E_{\text{cell}}(A) = (\Gamma_a + \Gamma_b) A + \frac{c \Gamma_l V_0}{2 \sqrt{A}}, \qquad c = 2\sqrt{2\sqrt{3}} \approx 3.7224.$$

1. **Universal Equilibrium Cell Height & Aspect Ratio Law:**
   Minimizing $E_{\text{cell}}(A)$ with respect to $A$ ($\partial E_{\text{cell}} / \partial A = 0$) yields the equilibrium cell height $h_0$, width $d_0 = \sqrt{A_0}$, and aspect ratio $h_0 / d_0$:
   $$h_0 = \left(\frac{c \Gamma_l \sqrt{V_0}}{4(\Gamma_a + \Gamma_b)}\right)^{2/3}, \qquad \frac{h_0}{d_0} = \frac{c \Gamma_l}{4(\Gamma_a + \Gamma_b)}, \qquad E''_{\text{cell}}(A_0) = \frac{3 c \Gamma_l V_0}{8 A_0^{5/2}} > 0.$$
   This single dimensionless tension ratio continuously spans **squamous** ($h_0/d_0 \ll 1$), **cuboidal** ($h_0/d_0 \approx 1$), and **tall columnar** ($h_0/d_0 \gg 1$) epithelia.
2. **Apico-Basal Differential Tension & Spontaneous Curvature Sign Reversal:**
   Asymmetry between apical and basal actomyosin tensions $\Delta\Gamma = \Gamma_a - \Gamma_b$ generates a spontaneous bending curvature:
   $$C_0 \propto \frac{\Gamma_a - \Gamma_b}{\Gamma_l h_0},$$
   which reverses sign between apical constriction ($\Gamma_a > \Gamma_b \implies C_0 > 0$) and basal constriction ($\Gamma_b > \Gamma_a \implies C_0 < 0$).
3. **3D Epithelial Buckling Wavelength & Phase Diagram:**
   Coupling the monolayer effective bending rigidity $\kappa_{\text{eff}} \propto \Gamma_l h_0^3 / d_0$ to an elastic stromal foundation $K_{\text{sub}}$ under compressive proliferation pressure selects the buckling wavelength:
   $$\lambda_c = 2\pi \left(\frac{\kappa_{\text{eff}}}{K_{\text{sub}}}\right)^{1/4},$$
   partitioning the 3D epithelial phase space $(\Gamma_l/\Gamma_a, K_{\text{sub}})$ into flat squamous, flat cuboidal, flat columnar, buckled tubular/mucosal folds, and closed spherical acini.
4. **3D Polyhedral Scutoid Packing on Curved Tubes (Pass 3, Gómez-Gálvez et al. 2018):**
   Cells of height $h_0$ on a tube of apical radius $R_a$ have a basal surface ratio $s = R_b/R_a = (R_a + h_0)/R_a$ (apical lumen) or $(R_a - h_0)/R_a$ (basal lumen, villus). The apical Voronoi tessellation of the cell seeds and the Voronoi tessellation of the same seeds radially projected to the basal surface (circumferential coordinate stretched by $s$) differ whenever $s \neq 1$; a cell whose apical and basal neighbour sets differ is a **scutoid** (an apico-basal neighbour intercalation on its lateral face). At a buckled fold crest the local surface ratio is $s_{\text{crest}} = 1 + h_0\,\kappa_{\max}$. Identities used as references: $f_s(1) = 0$, $f_s$ non-decreasing in $s$, mean neighbour number $= 6$ on both surfaces (trivalent torus, $2E = 6F$), and the lateral face area of the scutoid tessellation $\le$ that of the intercalation-free frustum tessellation.

## Vendored Files

- `scout_manifest.json` — Literature & specimen discovery manifest for Slot 047.
- `paper_citations.bib` — BibTeX citations with DOIs for Hannezo et al. (2014), Krajnc & Ziherl (2015), Štorgel et al. (2016), Sui et al. (2018) and (Pass 3) Gómez-Gálvez et al. (2018). SHA-256 `77a77d7dc42170a5f8c36dea8aa050a024210469ef1631ab3d501adf9537ae81` (re-pinned in Pass 3 after the Gómez-Gálvez entry was appended).
- `epithelial_buckling_3d_reference.json` — Quantitative reference dataset (12-tissue *Mus musculus* & *Drosophila melanogaster* aspect-ratio & wavelength table, 5 perturbation controls, and specimen reference metrics).
- `scutoid_tube_reference.json` — *(Pass 3)* Scutoid tube-mesh reference table: geometric identities used as reference values ($f_s(1)=0$, Spearman monotonicity 1, Euler mean neighbours 6, lateral-area bound 1), the surface-ratio sweep, tube defaults, and the wild-type fold-crest surface ratio $s_{\text{crest}} = 2.303$ derived from the already-pinned WT amplitude (42.5 µm), wavelength (136.4 µm) and $h_0$ (28.9 µm). All values are identities or derivations — none is digitised from the Gómez-Gálvez figures (stated in `parameter_status`). SHA-256 `85d9db6cbb5b95ff4a9a8673b163e98c8adb852088b981d458566d1db129c2d0`.
- `epithelial_sheet_buckling_specimen.jpg` — Wikimedia Commons CC0 trichrome histological section of folded simple columnar epithelium on connective tissue substrate (`File:Epithelial Tissues Simple Columnar Epithelium (40824137585).jpg`, 960×541 px, SHA-256 `cd05cc85a7f49dd2bac2644f4b530701871f16355928a75c92559615b1648841`).

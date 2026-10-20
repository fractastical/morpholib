# `datasets/cortical_sulcal_axon_tension/` — Van Essen (1997), Hilgetag & Barbas (2006) & Mota & Herculano-Houzel (2015) Cortical Sulcal Axon Tension Reference Package

Formal SHA-256-locked dataset and biological specimen image for `2020-karbowski-cortical-sulcal-axon-tension.py` (Slot 090, `karbowski-cortical-sulcal-axon-tension`).

## Contents

1. **`cortical_sulcal_axon_tension_reference.json`**
   - **Van Essen DC (1997) *Nature* 385(6614):313–318 (`doi:10.1038/385313a0`)**:
     - Viscoelastic cortico-cortical white-matter axonal tension $T_{ij} = k_a C_{ij} (\ell_{ij} - L_{0,ij})_+$ pulling strongly interconnected cortical regions ($C_{ij} \gg 1$, e.g. V1–V2, MT–MST, BA46–BA9) together across outward gyral crowns, replacing geodesic surface arc distance $\ell_{\text{arc}}$ with compact 3D Euclidean chord wiring $\ell_{\text{chord}} < \ell_{\text{arc}}$ to minimize total wiring volume $V_{\text{wire}} = \sum_{i<j} C_{ij} \ell_{ij}$, while weakly connected boundaries fold inward as sulcal fundi.
   - **Hilgetag CC & Barbas H (2006) *PLoS Computational Biology* 2(3):e22 (`doi:10.1371/journal.pcbi.0020022`)**:
     - Isometric cortical column volume conservation $V_{\text{col}} = A_{\text{surf}} \cdot h = A_0 h_0$ across primate cortical convolutions, predicting radial thickening in gyral crowns ($h_{\text{gyrus}} = 3.14\text{ mm}$, $\lambda_t = 0.78$), baseline thickness along sulcal walls ($h_{\text{wall}} = 2.45\text{ mm}$, $\lambda_t = 1.00$), and tangential stretching / radial thinning in sulcal fundi ($h_{\text{fundus}} = 1.94\text{ mm}$, $\lambda_t = 1.26$, $h_{\text{gyrus}} / h_{\text{fundus}} = 1.6186$).
   - **Mota B & Herculano-Houzel S (2015) *Science* 349(6243):74–77 (`doi:10.1126/science.aaa9101`)**:
     - 12-species mammalian universal cortical folding power law $A_T \sqrt{h} = k A_E^{5/4}$ ($\log_{10}(A_T \sqrt{h}) = 1.25 \log_{10}(A_E) + \log_{10} k$) and gyrification index $\text{GI}_{3\text{D}} = A_T / A_E = k h^{-1/2} A_E^{1/4}$ across Rodentia, Lagomorpha, Carnivora, Artiodactyla, Cetacea, and Primates.
   - **Toro & Burnod (2005) *Cerebral Cortex* 15:1900–1913 (`doi:10.1093/cercor/bhi068`) & Karbowski (2001) *Phys. Rev. Lett.* 86:3674 (`doi:10.1103/PhysRevLett.86.3674`)**:
     - Towed-axon viscoelastic rest-length adaptation $dL_{0,ij}/dt = \gamma_a (T_{ij} - \tau_0)_+$ and axonal tension ablation / callosotomy controls ($k_a = 0$).
   - SHA-256: `51d4ed1c896638d969157aa5addd3b5225fbca517b25f0362201e90420aa0288`.

2. **`coronal_cortex_axon_tracts_specimen.jpg`**
   - Coronal cross-section of a real human cerebral hemisphere showing outward gyral crowns, inward sulcal fundi, thicker gyral vs thinner sulcal gray-matter ribbon, and radiating white-matter axonal tracts including the corpus callosum and subcortical white-matter core ([Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Human_brain_frontal_(coronal)_section.JPG), CC BY 2.5).
   - SHA-256: `ff9a25e993fd5aadb99b44e0e7b8063e96a6a4c348228ce6ee69c6a8c01638da`.

3. **`scout_manifest.json`**
   - Step 1 literature & specimen scout dossier for Slot 090.
   - SHA-256: `3ef66e67c6994708117f3a9ac636c736f7ce3c33c83e8837c73ed76489939cf7`.

4. **`paper_citations.bib`**
   - BibTeX entries for Van Essen (1997), Hilgetag & Barbas (2006), Mota & Herculano-Houzel (2015), Toro & Burnod (2005), and Karbowski (2001).
   - SHA-256: `bcd7b0529c1c497d8c932e201f75a66888866f58236861b9bbf556672561849d`.

# Epithelial Somite Rosette Vertex Mechanics & Mesenchymal-to-Epithelial Transition (MET) Dataset (`datasets/somitoid_chirality_buckling/`)

Reference parameters, formal apico-basal differential interfacial tension vertex equations, EphA4/ephrinB2 somite cleft sorting laws, experimental perturbation regimes (Matrigel embedding, Y-27632 ROCK inhibition, Eph/ephrin boundary knockdown), and biological specimen photography for **Slot 029 (`slot-029-somitoid-chirality-buckling`)** of the Infinite Morphospace & MorphoEval Campaign, reproducing:

- **Yamanaka, Y. et al. (2023).** *"Reconstituting human somitogenesis in vitro."* *Nature* 614(7948):509–520. DOI: `10.1038/s41586-022-05649-2`
- **Miao, Y. et al. (2023).** *"Reconstruction and deconstruction of human somitogenesis in vitro."* *Nature* 614(7948):500–508. DOI: `10.1038/s41586-022-05655-4`
- **Adhyapok, P. et al. (2021).** *"A mechanical model of early somite segmentation."* *Developmental Cell* 56(9):1312–1324. DOI: `10.1016/j.devcel.2021.03.026`
- **Sanaki-Matsumiya, M. et al. (2022).** *"Periodic formation of epithelial somites from human pluripotent stem cells."* *Nature Communications* 13(1):2325. DOI: `10.1038/s41467-022-29967-1`

## Governing Mathematical Laws

1. **Apico-Basal Differential Interfacial Tension & Wedge-Cell Rosette Equilibrium Law (`yaman-somitoid-chirality-buckling-1`):**
   \[
   E(R_{\text{in}}, N) = 2\pi \gamma_{\text{ap}} R_{\text{in}} + 2\pi \gamma_{\text{bas}} R_{\text{out}} + N \gamma_{\text{lat}} (R_{\text{out}} - R_{\text{in}}), \qquad
   R_{\text{out}} = \sqrt{R_{\text{in}}^2 + \frac{N A_0}{\pi}}
   \]
   Minimizing \(E(R_{\text{in}}, N)\) with respect to the central somitecoel lumen radius \(R_{\text{in}}\) yields the exact analytical apical-to-basal radius ratio:
   \[
   \frac{R_{\text{in}}^*}{R_{\text{out}}^*} = \frac{L_{\text{ap}}^*}{L_{\text{bas}}^*} = \frac{N \gamma_{\text{lat}} - 2\pi \gamma_{\text{ap}}}{N \gamma_{\text{lat}} + 2\pi \gamma_{\text{bas}}}
   \]
   with wedge angle \(\theta_{\text{wedge}} = 2\pi / N > 0\) and strict apical wedge tapering \(L_{\text{ap}} < L_{\text{bas}}\) whenever \(\gamma_{\text{ap}} > 0\) and \(\gamma_{\text{bas}} > 0\).

2. **EphA4 / ephrinB2 Inter-Somitic Cleft Boundary Sorting Energy Barrier Law (`yaman-somitoid-chirality-buckling-2`):**
   \[
   \Delta E_{\text{cleft}} = 2 L_{\text{cleft}} \left(\gamma_{\text{het}} - \frac{\gamma_{\text{ant}} + \gamma_{\text{post}}}{2}\right) > 0 \quad \Longleftrightarrow \quad 2\gamma_{\text{het}} > \gamma_{\text{ant}} + \gamma_{\text{post}}
   \]
   Following MESP2/RIPPLY2 wavefront arrest, heterotypic EphA4/ephrinB2 contact repulsion elevates \(\gamma_{\text{het}}\) above homotypic N-cadherin adhesion, driving mechanical cleft separation between adjacent somite rosettes.

3. **Y-27632 ROCK Inhibition & Minus-Matrigel Apico-Basal Tension Collapse Law (`yaman-somitoid-chirality-buckling-3`):**
   \[
   \gamma_{\text{ap}} = \gamma_{\text{bas}} \implies \Delta \gamma_{\text{ap-bas}} = \gamma_{\text{ap}} - \gamma_{\text{bas}} = 0
   \]
   Inhibiting Rho-kinase (Y-27632) or omitting Matrigel extracellular matrix abolishes the apico-basal differential tension (\(\gamma_{\text{ap}} \to \gamma_{\text{bas}}\)), collapsing spontaneous wedge curvature and preventing polarized epithelial somite rosette formation.

## Vendored Files & Cryptographic Digests

- `scout_manifest.json` — OpenAlex + Wikimedia Commons literature & specimen scout dossier (`8326704c93aa8a75c90977f41fe3e00040bc848a2cceabe362af10fed750b806`)
- `somitoid_chirality_buckling_reference.json` — Formal vertex energy equations, equilibrium lumen ratio laws, and 6-regime experimental perturbation table
- `somitoid_rosette_specimen.jpg` — Dorsal vertebrate embryo scanning electron micrograph showing bilateral epithelialized somite rosettes flanking the neural tube (`8b80971796cb29e290c10886c0026092bc30f3f624d2a75fa72f9aaf05187d5f`, CC BY 4.0, Wikimedia Commons)
- `paper_citations.bib` — BibTeX citations for Yamanaka et al. (2023), Miao et al. (2023), Adhyapok et al. (2021), and Sanaki-Matsumiya et al. (2022)

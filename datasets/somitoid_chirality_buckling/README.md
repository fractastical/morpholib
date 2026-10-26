# Epithelial Somite Rosette Vertex Mechanics & Mesenchymal-to-Epithelial Transition (MET) Dataset (`datasets/somitoid_chirality_buckling/`)

Reference parameters, formal apico-basal differential interfacial tension vertex equations, EphA4/ephrinB2 somite cleft sorting laws, experimental perturbation regimes (Matrigel embedding, Y-27632 ROCK inhibition, Eph/ephrin boundary knockdown), and biological specimen photography for **Slot 029 (`slot-029-somitoid-chirality-buckling`)** of the Project Delphi & MorphoEval Campaign, reproducing:

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

4. **3D Polyhedral Shell Somitecoel Lumen Pressurization Law (`yaman-somitoid-chirality-buckling-4`, Pass 3):**
   \[
   E_{3D} = \sum_i \Big[\tfrac{K_V}{2}(V_i - V_0)^2 + \gamma_{\text{ap}} A_{\text{ap},i} + \gamma_{\text{bas}} A_{\text{bas},i}\Big] + \gamma_{\text{lat}} \sum_{\text{faces } ij} A_{\text{lat},ij} - P\,V_{\text{lumen}}
   \]
   for a closed shell of \(N_{3D} = 96\) radial frustum cells whose apical/basal faces are the spherical-Voronoi polygons of Fibonacci-sphere seeds (48 hexagons, 30 pentagons, 18 heptagons, 282 lateral faces). With \(\Lambda = \gamma_{\text{lat}} \sum_{ij} l_{ij}/2\) the radial balance gives
   \[
   \rho_{3D} = \frac{R_{\text{in}}}{R_{\text{out}}} = \frac{\Lambda - 4\pi\gamma_{\text{ap}} + 2\pi P R_{\text{in}}}{\Lambda + 4\pi\gamma_{\text{bas}}}, \qquad \rho_{3D} < 1 \iff P < \frac{2(\gamma_{\text{ap}} + \gamma_{\text{bas}})}{R_{\text{in}}}.
   \]
   Scutoid statement (Gómez-Gálvez et al. 2018): radial frusta share one apical/basal neighbour graph, so the shell has 0 scutoids by construction, which is also what the Voronoi argument predicts for isotropic (spherical) curvature; scutoid formation under tubular curvature is **not** modelled.

5. **HES7 Clock–Wavefront Arrest & Tension Maturation Gradient Law (`yaman-somitoid-chirality-buckling-5`, Pass 3):**
   \[
   \frac{dm}{dt} = \frac{k}{1 + (p(t - T_m)/p_0)^2} - c\,m, \qquad \frac{dp}{dt} = a\,m(t - T_p) - b\,p \quad \text{(Lewis 2003, time-scaled to } T_{\text{HES7}} \approx 5\text{ h)}
   \]
   \[
   x_{\text{wf}}(t) = v t, \quad t_{a,j} = x_j / v, \quad S = v\,T_{\text{HES7}}, \quad \gamma_{\text{ap},j}(t) = \gamma_{\text{bas}} + (\gamma_{\text{ap}}^{WT} - \gamma_{\text{bas}})\big(1 - e^{-(t - t_{a,j})/\tau_{\text{ROCK}}}\big)
   \]
   The arrested HES7 phase gates MESP2 (anterior half-segment, EphA4⁺), RIPPLY2 switches MESP2 off and fixes the boundary at \(t_{\text{det},k}\), and \(\gamma_{\text{het},k}(t)\) ramps toward the EphA4/ephrinB2 value after \(t_{\text{det},k}\). The resulting tensions are fed live into Laws 1, 2 and 4.

## Vendored Files & Cryptographic Digests

- `scout_manifest.json` — OpenAlex + Wikimedia Commons literature & specimen scout dossier (`8326704c93aa8a75c90977f41fe3e00040bc848a2cceabe362af10fed750b806`)
- `somitoid_chirality_buckling_reference.json` — Formal vertex energy equations, equilibrium lumen ratio laws, 6-regime experimental perturbation table, and (Pass 3) the `shell_3d_governing_equations` and `hes7_clock_wavefront` blocks; now SHA-256 pinned in the script (`EXPECTED_REFERENCE_JSON_SHA256`)
- `somitoid_rosette_specimen.jpg` — Dorsal vertebrate embryo scanning electron micrograph showing bilateral epithelialized somite rosettes flanking the neural tube (`8b80971796cb29e290c10886c0026092bc30f3f624d2a75fa72f9aaf05187d5f`, CC BY 4.0, Wikimedia Commons)
- `paper_citations.bib` — BibTeX citations for Yamanaka et al. (2023), Miao et al. (2023), Adhyapok et al. (2021), Sanaki-Matsumiya et al. (2022), and (Pass 3) Lewis (2003), Diaz-Cuadros et al. (2020), Matsuda et al. (2020), Cooke & Zeeman (1976), Gómez-Gálvez et al. (2018)

## Known Discrepancies (carried to Pass 4)

- The scout-stage `rosette_metrics` (lower row 8 somites, bilateral spacing 47.14 px, CV 0.2010) disagree with the script's DoG segmentation (lower row 10 peaks, 45.375 px, CV 0.2391). Neither value has been overwritten; the JSON carries a `provenance_note_pass3`.
- The clock-coupling parameters `v`, `tau_ROCK`, `tau_Eph`, `tau_RIPPLY2` are model choices (segment ≈ 100 µm, MET over ≈ 1 clock period), not values digitised from Yamanaka et al. (2023).
- No measured somitocoel pressure exists in the source papers; `P_crit` is referenced only to the model-internal Laplace bound.

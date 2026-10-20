# Dataset: Epithelial Buckling into Ridges, Zigzags, and Intestinal Villi & Crypts (`epithelial_buckling_crypt_villus`)

## Primary Literature & DOIs

1. **Shyer AE, Tallinen T, Nerurkar NL, Wei Z, Gil ES, Kaplan DL, Tabin CJ, Mahadevan L (2013)**.
   *Villification: how the gut gets its villi.* **Science** 342(6155):212–218.
   DOI: [`10.1126/science.1238842`](https://doi.org/10.1126/science.1238842)
2. **Hannezo E, Prost J, Joanny J-F (2011)**.
   *Instabilities of monolayered epithelia: coupled bending and polarity-driven forces.* **Physical Review Letters** 107(7):078104.
   DOI: [`10.1103/PhysRevLett.107.078104`](https://doi.org/10.1103/PhysRevLett.107.078104)
3. **Tallinen T, Biggins JS, Mahadevan L (2013)**.
   *Surface sulci in squeezed slabs.* **Physical Review Letters** 110(2):024302.
   DOI: [`10.1103/PhysRevLett.110.024302`](https://doi.org/10.1103/PhysRevLett.110.024302)
4. **Walton KD et al. (2012)**.
   *Hedgehog-responsive mesenchymal clusters direct patterned epithelial folding in the developing mouse intestine.* **PNAS** 109(39):15817–15822.
   DOI: [`10.1073/pnas.1205669109`](https://doi.org/10.1073/pnas.1205669109)

## Governing Mathematical & Biomechanical Framework

The developing embryonic gut mucosa (endodermal epithelium of thickness $h_e$ and bending rigidity $B_e = E_e h_e^3 / [12(1 - \nu^2)]$ coupled to a compliant subepithelial mesenchymal stroma of effective Winkler foundation stiffness $K_s$) undergoes sequential mechanical buckling driven by the differentiation of three concentric smooth muscle layers:

$$\mathcal{F}[w] = \iint_{\Omega} \left[ \frac{B_e}{2} (\nabla^2 w)^2 - \frac{N_x}{2} (\partial_x w)^2 - \frac{N_y}{2} (\partial_y w)^2 + \frac{K_s}{2} w^2 - \frac{\alpha_3}{6} w^3 + \frac{\alpha_4}{24} w^4 \right] dx\,dy$$

where $N_x = E_e h_e \varepsilon_x$ and $N_y = E_e h_e \varepsilon_y$ are the circumferential and longitudinal compressive forces per unit length, and $\alpha_3 > 0$ encodes the apicobasal/lumen-stroma asymmetry breaking $w \to -w$ inversion symmetry.

1. **Wavelength & Critical Strain Law:**
   $$q_* = \left(\frac{K_s}{B_e}\right)^{1/4}, \qquad \lambda_* = \frac{2\pi}{q_*} = 2\pi \left(\frac{B_e}{K_s}\right)^{1/4}, \qquad N_c = 2\sqrt{B_e K_s}, \qquad \varepsilon_c = \frac{2\sqrt{B_e K_s}}{E_e h_e}.$$
2. **Sequential Smooth-Muscle Buckling Cascade (Chick E6 $\to$ E18):**
   - **E6 (Pre-buckling Smooth Tube):** No smooth muscle ($\varepsilon_x < \varepsilon_c$, $\varepsilon_y < \varepsilon_c$) $\implies w(x,y) = 0$.
   - **E8–E10 (1D Parallel Longitudinal Ridges):** Circular smooth muscle differentiates ($\varepsilon_x > \varepsilon_c > \varepsilon_y$) $\implies$ 1D circumferential buckling $w(x,y) = A_r \cos(q_* x)$ with $N_{\text{ridges}} \in [8, 16]$ ($N_{\text{ridges}} = 12$ at E10).
   - **E13–E14 (2D Herringbone Zigzags):** Outer longitudinal smooth muscle differentiates ($\varepsilon_x > \varepsilon_c$, $\varepsilon_y > \varepsilon_{c,2}$) $\implies$ secondary oblique buckling with chevron half-angle $\theta_{\text{zz}} = \arctan(q_y / q_x) \approx 38.5^\circ$.
   - **E16–E18 (Hexagonal Array of Villi & Crypt Invaginations):** Inner longitudinal smooth muscle (muscularis mucosae) differentiates under isotropic biaxial compression ($\varepsilon_x \approx \varepsilon_y > \varepsilon_c$) with cubic asymmetry $\alpha_3 > 0$, stabilizing the resonant hexagonal wavevector triad $\mathbf{q}_1 + \mathbf{q}_2 + \mathbf{q}_3 = \mathbf{0}$ ($|\mathbf{q}_i| = q_*$).
3. **FK506 & Surgical Ablation Controls:**
   - **FK506 at E6 (No Circular Muscle):** Arrests lumen in smooth unbuckled state ($N_{\text{ridges}} = 0$).
   - **FK506 at E11 (No Outer Longitudinal Muscle):** Arrests morphogenesis at parallel longitudinal ridges ($\theta_{\text{zz}} = 0^\circ$).
   - **FK506 at E14 (No Inner Longitudinal Muscle):** Arrests morphogenesis at herringbone zigzags ($\theta_{\text{zz}} \approx 38.2^\circ$, no discrete villus pinch-off).

## Vendored Files

- `scout_manifest.json` — Literature & specimen discovery manifest for Slot 063.
- `paper_citations.bib` — BibTeX citations with DOIs for Shyer et al. (2013), Hannezo et al. (2011), Tallinen et al. (2013), and Walton et al. (2012).
- `epithelial_buckling_crypt_villus_reference.json` — Quantitative reference tables (chick E6–E18 ontogenetic trajectory, 12-point cross-stage/species scaling collapse, FK506/surgical ablation controls, and SHA-256 digests).
- `intestinal_villi_crypts_specimen.jpg` — Wikimedia Commons CC0 transverse H&E histological micrograph of small intestinal villi (`File:Cross-section histology of small intestinal villi of the terminal ileum.jpg`, 960×718 px).

# `datasets/hox_turing_polydactyly/` — Sheth et al. (2012) *Science* Hox-Modulated Turing Polydactyly Dataset

Formal SHA-256-locked reference dataset and vertebrate autopod radiograph specimen for `2012-sheth-hox-turing-polydactyly.py` (Slot 015 of the 100-Loop Infinite Morphospace Campaign).

## Contents

1. **`hox_turing_polydactyly_reference.json`**
   - **Sheth et al. (2012) *Science* 338(6113):1476–1480 (`doi:10.1126/science.1226804`)**:
     - Cubic-saturated 2-component activator-inhibitor Turing reaction-diffusion system on the embryonic fan-shaped autopod:
       $$\frac{\partial u}{\partial t} = f_u(H)\,u + f_v\,v - u^3 + D_u \nabla^2 u, \qquad \frac{\partial v}{\partial t} = g_u(H)\,u + g_v\,v + D_v \nabla^2 v$$
     - Distal 5' *Hoxa13* / *Hoxd11–13* dose $H \in [0, 1]$ in a *Gli3*-null background modulates the Turing critical wavenumber $k_c(H) = 14 - 9H$ ($k_c^2(H) = (D_v f_u(H) + D_u g_v)/(2 D_u D_v) = f_u(H) - 2$), shrinking the digit stripe wavelength $\lambda(H) = 2\pi / k_c(H)$ from $500\,\mu\text{m}$ (WT, 5 digits) to $178.6\,\mu\text{m}$ (*Hoxa13*${}^{-/-}$; *Hoxd11–13*${}^{-/-}$; *Gli3*${}^{-/-}$, 14 thin closely-spaced digits) at constant autopod arc width $L_{\text{phys}} = 2500\,\mu\text{m}$.
   - **6-Genotype Allelic Series Table**:
     - Covers `WT` ($H=1.0$, 5 digits), `Gli3-/-` ($H=7/9$, 7 digits), `Hoxd11-13+/-;Gli3-/-` ($H=6/9$, 8 digits), `Hoxd11-13-/-;Gli3-/-` ($H=5/9$, 9 digits), `Hoxa13+/-;Hoxd11-13-/-;Gli3-/-` ($H=3/9$, 11 digits), and `Hoxa13-/-;Hoxd11-13-/-;Gli3-/-` ($H=0.0$, 14 digits).

2. **`vertebrate_autopod_digits_specimen.jpg`**
   - Postaxial hexadactylous vertebrate autopod anteroposterior (AP) radiograph exhibiting 6 fan-arranged metacarpals and phalangeal digit rays.
   - Source: [Wikimedia Commons `File:Polydactyly_01_Rhand_AP.jpg`](https://commons.wikimedia.org/wiki/File:Polydactyly_01_Rhand_AP.jpg) (CC BY-SA 3.0).
   - SHA-256: `995e43a2a0cf781547dc8f573685a533013cea3502fb2ebbb9c1fc8a6cb14e90`.

3. **`scout_manifest.json`**
   - Step 1 (`01_scout`) OpenAlex + Wikimedia Commons discovery dossier for Slot 015.
   - SHA-256: `c0608869f6d213a4b1b557968bddb544e9c58f0dbec460061979254edd930f80`.

4. **`paper_citations.bib`**
   - BibTeX citations for Sheth et al. (2012), Raspopovic et al. (2014), Newman & Frisch (1979), and Miura et al. (2006).

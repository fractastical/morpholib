# `datasets/hox_turing_polydactyly/` — Sheth et al. (2012) *Science* Hox-Modulated Turing Polydactyly Dataset

Formal SHA-256-locked reference dataset and vertebrate autopod radiograph specimen for `2012-sheth-hox-turing-polydactyly.py` (Slot 015 of the 100-Loop Project Delphi Campaign).

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
   - Pass 3 additions: Crampin, Gaffney & Maini (1999) `doi:10.1006/bulm.1999.0131`, Boehm et al. (2010) `doi:10.1371/journal.pbio.1000420`, Summerbell (1974) `doi:10.1242/dev.32.3.651`, Kondo & Miura (2010) `doi:10.1126/science.1179047`.
   - SHA-256 (re-pinned in the script after the Pass 3 append): `c2cf41b4ebfc4449594e1275c3ce7ee87a9bad914e6e2d32dee628c0c514e72e`.

5. **`pass3_aer_growth_reference.json`** (Pass 3)
   - Parameter table and source anchors for the AER-FGF8 mitogenic outgrowth coupling added in Pass 3: the fan sector $r \in [r_{\text{prox}}, R(t)]$ grows through a prescribed kinematic area dilation $g(r,t) = g_{\max} A(t) e^{-(R(t)-r)/\ell_F}$ localized at the AER; mass conservation gives the radial tissue velocity $v(r,t) = \frac{1}{r}\int_{r_{\text{prox}}}^{r} g\, r'\,dr'$ and the rim outgrowth $\dot R = v(R,t)$. The Sheth et al. (2012) reaction-diffusion system is solved in ALE form with the Crampin–Gaffney–Maini (1999) dilution term $-(\nabla\cdot v)\,u$ on the physical (deformed) polar metric.
   - $g_{\max}$ is calibrated (Brent root-find on the outgrowth ODE) so that the WT distal arc reaches $2500\,\mu\text{m}$ at $T = 5$; AER regression starts at $t = 2.4$; the AER-ablation series uses $t_{\text{abl}} \in \{0.6, 1.2, 1.8\}$ (Summerbell 1974 truncation ordering); the $g_{\max}$ sweep tests the digit-number/width law $N = L(T)/\lambda(H)$ (Lean 4 Theorem 4) and the growth-dilution threshold $\sigma_{\max} > g_{\text{rim}}$ (Lean 4 Theorem 5).
   - Honesty note: growth is kinematic (no elastic constitutive law or stress balance) and digit-ray orientation is imposed by PD/AP diffusion anisotropy ($D_r = 0.05\,D$); both are carried as open Pass 4 items.
   - SHA-256: `311a1e81311eabab1c73b1b74df6c32be857d2bdfe1cb52232d7503da8199966`.

## Pass 3 outputs

- `docs/media/hox_turing_polydactyly_comparison.png` — 12-panel synthesis (Panels I–L: AER-grown WT and triple-null fans with the deforming Lagrangian mesh, outgrowth $L(t)$ / sequential digit emergence $N(t)$, and the digit-number-vs-width law with growth lag).
- `docs/media/hox_turing_polydactyly_growth_mesh.png` — 2×8 time-lapse of the deforming tissue mesh (WT vs triple null).
- `docs/media/hox_turing_polydactyly_growth.gif` — 6 allelic-series frames followed by 8 AER-growth time-lapse frames.

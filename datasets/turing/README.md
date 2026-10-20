# `datasets/turing/` — Turing (1952), Pearson (1993) & Kondo–Asai (1995) Reference Package

Formal SHA-256-locked dataset and biological specimen photographs for `1952-turing-morpho.py`.

## Contents

1. **`turing_reference.json`**
   - **Turing (1952) *Phil. Trans. R. Soc. B* 237(641):37–72 (`doi:10.1098/rstb.1952.0012`)**:
     - Two-morphogen linearized reaction-diffusion Jacobian $J = \begin{pmatrix} a & b \\ c & d \end{pmatrix}$, dispersion relation $\sigma(k^2)$, critical diffusivity ratio $d_c$, critical wavenumber $k_c$, analytical wavelength $\lambda_c = 2\pi / k_c$, and the Section 9 ring of $N = 20$ cells discrete Fourier mode spectrum.
   - **Pearson (1993) *Science* 261(5118):189–192 (`doi:10.1126/science.261.5118.189`)**:
     - Physical domain scaling ($L = 2.5$, $N = 256$, $D_u = 2\times 10^{-5}$, $D_v = 10^{-5}$), Saddle-Node bifurcation curve $4(F+k)^2 = F \iff k_{\text{SN}}(F) = \frac{1}{2}\sqrt{F} - F$, Hopf bifurcation curve $k_{\text{Hopf}}(F) = \frac{\sqrt{F} - 2F - \sqrt{F(1 - 4\sqrt{F})}}{2}$, and all 12 pattern classes ($\alpha, \beta, \gamma, \delta, \epsilon, \zeta, \eta, \theta, \iota, \kappa, \lambda, \mu$).
   - **Kondo & Asai (1995) *Nature* 376(6543):765–768 (`doi:10.1038/376765a0`)**:
     - Marine angelfish (*Pomacanthus imperator* / *P. semicirculatus*) ontogenetic body growth ($L_{\text{body}} = 2.0 \to 12.0\text{ cm}$), stripe count scaling ($3 \to 19$ stripes at constant intrinsic wavelength $\lambda_{\text{stripe}} \approx 3.8\text{ mm}$), and Y-branch dislocation insertion.

2. **`pomacanthus_imperator_stripes_specimen.jpg`**
   - Adult Emperor angelfish (*Pomacanthus imperator*), Red Sea, Egypt.
   - Photograph by Diego Delso ([Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Pez_%C3%A1ngel_emperador_(Pomacanthus_imperator),_mar_Rojo,_Egipto,_2023-04-14,_DD_30.jpg), CC BY-SA 4.0).
   - SHA-256: `aa2b4ee7effb9f4e093277d08914dfbe960382ed1a10fe000a9dee766d520e7d`.

3. **`ostracion_meleagris_spots_specimen.jpg`**
   - Adult Whitespotted boxfish (*Ostracion meleagris*), Zanzibar, Tanzania.
   - Photograph by Diego Delso ([Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Pez_cofre_moteado_(Ostracion_meleagris),_Zanz%C3%ADbar,_Tanzania,_2024-05-30,_DD_83.jpg), CC BY-SA 4.0).
   - SHA-256: `90890e6d8c4bb9120f5f1bdb3fd823aa7819d557a3b9468f2b0b2f5967f36acd`.

4. **`paper_citations.bib`** — BibTeX entries for Turing (1952), Pearson (1993), Lee et al. (1994), and Kondo & Asai (1995).

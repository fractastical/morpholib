# `datasets/limb_chondrogenesis_turing/` — Newman & Frisch (1979) *Science* Chick Limb Chondrogenesis Eigenmodes Dataset

Formal SHA-256-locked reference dataset and *Gallus gallus* forelimb skeleton specimen for `1979-newman-limb-chondrogenesis-turing.py` (Slot 012 of the 100-Loop Project Delphi Campaign).

## Contents

1. **`limb_chondrogenesis_turing_reference.json`**
   - **Newman & Frisch (1979) *Science* 205(4407):662–668 (`doi:10.1126/science.462174`)**:
     - 3D Helmholtz/Turing standing-wave eigenmodes in the uncondensed apical mesenchyme box $[0, L_{\text{AP}}] \times [0, L_{\text{DV}}] \times [0, L_{\text{PD}}(t)]$ beneath the Apical Ectodermal Ridge (AER):
       $$\Phi_{m, n}(x, y, z) = \cos\!\left(\frac{m \pi x}{L_{\text{AP}}}\right) \cos\!\left(\frac{n \pi y}{L_{\text{DV}}}\right) \sin\!\left(\frac{\pi z}{L_{\text{PD}}(t)}\right)$$
     - Nonlinear 2-component fibronectin/TGF-$\beta$ activator-inhibitor Turing reaction-diffusion PDE coupled to proximodistal progress-zone depth reduction $L_{\text{PD}}(t)$ and hyaluronidase-mediated effective diffusivity decrease $D_{\text{eff}}(t)$:
       $$\frac{\partial u}{\partial t} = f_u(z)\,u + f_v\,v - u^3 + D_u \nabla^2 u, \qquad \frac{\partial v}{\partial t} = g_u(z)\,u + g_v\,v + D_v \nabla^2 v$$
     - Sequential proximodistal skeletal mode transitions across Hamburger-Hamilton (HH) stages in *Gallus gallus*:
       - **HH20–21 Stylopod ($L_{\text{PD}} = 500\,\mu\text{m}$):** $m = 1$ ($1$ element: humerus in wing / femur in leg).
       - **HH24 Zeugopod ($L_{\text{PD}} = 250\,\mu\text{m}$):** $m = 2$ ($2$ parallel elements: radius + ulna in wing / tibia + fibula in leg).
       - **HH27–28 Autopod ($L_{\text{PD}} = 140\,\mu\text{m}$):** $m = 3$ in chick wing ($3$ digits: II, III, IV) and $m = 4$ in chick hindlimb ($4$ pedal digits: I–IV).
     - **AER Excision Truncation Series (Saunders 1948; Summerbell et al. 1973; Newman & Frisch 1979)**:
       - Excision at HH20 truncates limb to stylopod only ($[1, 0, 0]$ elements); excision at HH24 yields stylopod + zeugopod ($[1, 2, 0]$ elements); intact AER yields full $[1, 2, 3]$ chick wing skeleton.

2. **`chick_wing_skeleton_specimen.jpg`**
   - *Gallus gallus* forelimb osteological specimen showing the proximal-to-distal transition from humerus (stylopod, $m=1$) $\to$ radius and ulna (zeugopod, $m=2$) $\to$ carpus and digits II, III, IV (autopod, $m=3$).
   - Source: [Wikimedia Commons `File:Gallus_wingbones_01.JPG`](https://commons.wikimedia.org/wiki/File:Gallus_wingbones_01.JPG) (Author: Conty, License: CC BY 3.0).
   - SHA-256: `771e26f9fe8c813de108d87086e896597b450c09e8e9a706bb0f1edddeb26e41`.

3. **`scout_manifest.json`**
   - Step 1 (`01_scout`) OpenAlex + Wikimedia Commons discovery dossier for Slot 012.
   - Pass 3 correction (fix-004): `territory.seed_doi` was a transposed-digit typo pointing at an unrelated *Science* article; corrected to Newman & Frisch (1979) `10.1126/science.462174` (see `pass3_correction` in the JSON).
   - SHA-256: `71ab73feedf8dbfd5009fe9cf32fa10e2007c32a76f54c26993c1e7ca3598453`.

4. **`paper_citations.bib`**
   - BibTeX citations for Newman & Frisch (1979), Saunders (1948), Summerbell et al. (1973), Hentschel et al. (2004), and Stewart et al. (2017).

## Pass 3 additions (FGF8-graded moving-boundary 3D wing bud)

5. **`fgf8_moving_boundary_reference.json`** (SHA-256-pinned in the script's `EXPECTED_SHA256`)
   - Provenance for every default of `FGF8_GROWTH_DEFAULTS` in Section 4b of `1979-newman-limb-chondrogenesis-turing.py`: the one-parameter hyaluronidase clock $\lambda_F(t) = \lambda_0 e^{-t/\tau_H}$ fitted to the three Newman & Frisch Table 1 wing depths ($\tau_H = 37.0$ h, max $|$rel err$| = 4.6\,\%$), the FGF8 thresholds $F_{\text{ap}} = 0.5$ / $F_{\text{thr}} = 0.1$ (three-zone structure after Hentschel et al. 2004 and Zhu et al. 2010), the Hill-2 mitogenic response $S_{\max} = \ln 2 / 12\,\text{h}$ (Sun et al. 2002; Boehm et al. 2010), the AER-excision rule (Summerbell 1974, 1977; Rowe & Fallon 1982) and its reference series ($t_{\text{exc}} = 0 \to 1$ element, $24\,\text{h} \to 3$, intact $\to 6$).
   - Entries marked `model assumption` are not measured values; they are listed so that the audit can distinguish sourced constants from calibrations.

- **`paper_citations.bib`** now also contains Crampin, Gaffney & Maini (1999), Zhu et al. (2010), Boehm et al. (2010), Sun, Mariani & Martin (2002), Toole (1972), Solursh (1984), and Summerbell (1974, 1976, 1977).

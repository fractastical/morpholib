# `datasets/zebrafish_three_cell_stripes/` — Nakamasu et al. (2009) Zebrafish Three-Cell Stripe Morphogenesis Reference Package

Formal SHA-256-locked dataset and biological specimen photograph for `2009-nakamasu-zebrafish-three-cell-stripes.py` (Slot 007 of the 100-Loop Infinite Morphospace Campaign).

## Contents

1. **`zebrafish_three_cell_stripes_reference.json`**
   - **Nakamasu, Takahashi, Kanbe & Kondo (2009) *PNAS* 106(21):8429–8434 (`doi:10.1073/pnas.0808622106`)**:
     - Three-variable pigment-cell reaction-diffusion system coupling melanophores ($u$, black stripes), xanthophores ($v$, yellow interstripes), and a fast-diffusing long-range survival factor ($w$):
       $$\frac{\partial u}{\partial t} = F(u, v, w) - c_u u + D_u \nabla^2 u, \quad \frac{\partial v}{\partial t} = G(u, v, w) - c_v v + D_v \nabla^2 v, \quad \frac{\partial w}{\partial t} = H(u, v, w) - c_w w + D_w \nabla^2 w$$
     - Two-scale interaction circuit (Fig. 4D): short-range mutual repulsion ($b_u = -1.50 < 0, a_v = -1.50 < 0 \implies \det(J_{uv}) = c_u c_v - b_u a_v = -1.61 < 0$) creates local anti-phase pigment segregation ($\lambda_{\text{anti}} = -c_u + |b_u| = +0.70 > 0$), while long-range xanthophore $\to$ factor $w \to$ melanophore survival promotion ($b_w = +1.20 > 0, d_u = +1.20 > 0, D_w / D_u = 25$) stabilizes the homogeneous equilibrium at $k = 0$ ($p_1 = 2.50, p_2 = 1.63, p_3 = 2.961, p_1 p_2 - p_3 = 1.114 > 0$) and generates a bounded Turing instability band $k \in [2.1134, 5.3669]$ ($k_c = 3.4106, \lambda_c = 1.8423$).
   - **Laser-Ablation Polarity & *leopard* ($cx41.8$) Mutant Morphospace**:
     - Quantitative benchmarks for xanthophore interstripe ablation (Fig. 2 melanophore collapse), directional stripe regeneration after rectangular laser ablation (Fig. 3), and *leopard* (*gja5b* / $cx41.8$) spot-to-labyrinth-to-stripe transition (Fig. 5; Watanabe et al. 2006 *EMBO Rep.* 7:893–897).

2. **`danio_rerio_specimen.jpg`**
   - Adult wild-type zebrafish (*Danio rerio*) right lateral profile showing alternating dark melanophore stripes and yellow/iridescent xanthophore–iridophore interstripes across the trunk flank and caudal fin.
   - Source: [Wikimedia Commons (`File:Danio rerio lab right.JPG`)](https://commons.wikimedia.org/wiki/File:Danio_rerio_lab_right.JPG), CC BY-SA 3.0.
   - SHA-256: `850b2d4a94c7cbcb3c16e606ddb70d198f082ccb2f046ee44dbaa5ef5941678a`.

3. **`scout_manifest.json`**
   - Step 1 (`01_scout`) literature and Wikimedia Commons discovery dossier for Slot 007.
   - SHA-256: `d19df92d7c80de8a6e00fc0876165a1de8c4639b15b364de2d6600d02f895706`.

4. **`paper_citations.bib`**
   - BibTeX citations for Nakamasu et al. (2009), Yamaguchi et al. (2007), Watanabe et al. (2006), and Inaba et al. (2012).
   - SHA-256: `33a934e879adbd20447c2c30f3f934fc1dd5924a3b8002ef353e1a0c9cbb883f`.

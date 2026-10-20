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

   - **v2 (Pass 3)**: `dataset_id` bumped to `nakamasu-2009-zebrafish-three-cell-stripes-v2`; 5 new sources (Volkening & Sandstede 2015/2018, Singh 2014, Frohnhöfer 2013, Parichy 2009) and the pin of the new companion file below. SHA-256: `716558fb68296749d2a2aa8b24ea86a47a8af7733589e942514895bd78841296`.

2. **`danio_rerio_specimen.jpg`**
   - Adult wild-type zebrafish (*Danio rerio*) right lateral profile showing alternating dark melanophore stripes and yellow/iridescent xanthophore–iridophore interstripes across the trunk flank and caudal fin.
   - Source: [Wikimedia Commons (`File:Danio rerio lab right.JPG`)](https://commons.wikimedia.org/wiki/File:Danio_rerio_lab_right.JPG), CC BY-SA 3.0.
   - SHA-256: `850b2d4a94c7cbcb3c16e606ddb70d198f082ccb2f046ee44dbaa5ef5941678a`.

3. **`scout_manifest.json`**
   - Step 1 (`01_scout`) literature and Wikimedia Commons discovery dossier for Slot 007.
   - Pass 3 correction: the OpenAlex candidate titles for Watanabe et al. (2006) and Yamaguchi et al. (2007) were garbled in the original dossier and were replaced by the published titles (DOIs unchanged); a `pass3_correction` field records this.
   - SHA-256: `4a8716bc59f35f30f245d1a7c5483714a78cca13d7df838c7577eea1d5159735` (was `d19df92d…895706`).

4. **`paper_citations.bib`**
   - BibTeX citations for Nakamasu et al. (2009), Yamaguchi et al. (2007), Watanabe et al. (2006), Inaba et al. (2012), and (Pass 3) Volkening & Sandstede (2015, 2018), Singh et al. (2014), Frohnhöfer et al. (2013), Patterson & Parichy (2013), Parichy et al. (2009), Crampin, Gaffney & Maini (1999).
   - SHA-256: `4b64e2dc8820bfad13b9b77dd45892f44403d46ab71d29b941f8b5298d122c98` (was `33a934e8…bb883f`).

5. **`iridophore_ontogeny_reference.json`** *(added in Pass 3)*
   - Juvenile-to-adult stripe-addition normal table (Parichy et al. 2009 stage anchors PB 7.2 mm → A 18 mm with approximate stripe counts read from Parichy 2009 / Frohnhöfer 2013 / Singh 2014 — treat as ±1 stripe, see `confidence_note`), the reduced Volkening & Sandstede (2018) iridophore dense (S) ↔ loose (L) switching rules and feedback terms implemented in `simulate_hybrid_iridophore_growth()`, the isometric growth ↔ standard-length mapping (`g0 = 0.40 ↔ SL 7.2 mm`), and the Turing-band Neumann mode table checked by Lean 4 Theorem 4.
   - SHA-256: `c25943ac48e02adf11bb66e59c9774c265d603cc0ed2fc7c17441e5f6b518010`.

## What the script computes from this package (Pass 3)

- Sections 2–6 of the script (theory, template-cued PDE, laser ablation, *leopard* sweep, negative controls, specimen segmentation) are unchanged from Pass 2 and keep the pinned `zebrafish_three_cell_stripes_reference.json` values.
- Section 5b (`simulate_hybrid_iridophore_growth`, `run_hybrid_growth_experiments`) couples the continuum (u, v, w) PDE to discrete iridophore agents on a uniformly growing flank and compares the stripe-count ontogeny with the normal table in `iridophore_ontogeny_reference.json`, the S/L occupancy expectations, and an iridophore-less (*shady*-like) control.
- Declared residuals (see `morphoeval/audits/2026-10-19-zebrafish-three-cell-stripes-audit-v3.yaml`): dorsoventrally symmetric insertion, insertion by stretched-stripe splitting rather than marginal addition, isometric SL mapping, and the calibrated x-coherent iridophore kernel.

# `datasets/tgfb_fibronectin_limb/` — Hentschel, Glimm, Glazier & Newman (2004) *Proc. R. Soc. B* / Kiskowski et al. (2004) *Dev. Biol.* Multiscale TGF-$\beta$, FGF & Fibronectin Haptotaxis Limb Dataset

Formal SHA-256-locked reference dataset and *Gallus gallus* forelimb skeletal specimen for `2004-hentschel-tgfb-fibronectin-limb.py` (Slot 014 of the 100-Loop Project Delphi Campaign).

## Contents

1. **`tgfb_fibronectin_limb_reference.json`**
   - **Hentschel, Glimm, Glazier & Newman (2004) *Proc. R. Soc. Lond. B* 271(1549):1713–1722 (`doi:10.1098/rspb.2004.2772`)** & **Kiskowski, Alber, Glimm, Hentschel, Glazier & Newman (2004) *Dev. Biol.* 271(2):372–387 (`doi:10.1016/j.ydbio.2004.03.038`)**:
     - Multiscale reactor-diffusion-haptotaxis PDE system coupling diffusible activator TGF-$\beta$ ($c_a$), FGF-dependent lateral inhibitor ($c_i$), immobile extracellular matrix fibronectin ($F$), and haptotactic mesenchymal cell density ($R = R_1 + R_2$):
       $$\frac{\partial c_a}{\partial t} = f_a(c_{\text{FGF}})\,c_a + f_i\,c_i - c_a^3 + D_a \nabla^2 c_a, \qquad \frac{\partial c_i}{\partial t} = g_a(c_{\text{FGF}})\,c_a + g_i\,c_i + D_i \nabla^2 c_i$$
       $$\frac{\partial F}{\partial t} = k_F\,c_a^+\,R_2(c_a, R) - \gamma_F\,F, \qquad \frac{\partial R}{\partial t} = D_R \nabla^2 R - \chi_F \nabla \cdot (R \nabla F) + r_R R (1 - R)$$
     - **Proximodistal AER-FGF Zonation ($1 \to 2 \to 3 \to 4$ Skeletal Mode Transition)**:
       - **HH20–21 Stylopod ($f_a = 3.0, c_{\text{FGF}} = 0.25$):** $m = 1$ ($1$ element: humerus).
       - **HH24 Zeugopod ($f_a = 6.0, c_{\text{FGF}} = 0.55$):** $m = 2$ ($2$ parallel elements: radius + ulna).
       - **HH27–28 Wing Autopod ($f_a = 11.0, c_{\text{FGF}} = 0.85$):** $m = 3$ ($3$ chick wing digits: II, III, IV).
       - **HH28 Leg Autopod / High TGF-$\beta$ ($f_a = 18.0, c_{\text{FGF}} = 1.00$):** $m = 4$ ($4$ pedal digits: I–IV).
     - **Micromass Nodular vs. Labyrinthine Stripe Morphospace (Miura & Shiota 2000; Kiskowski et al. 2004)**:
       - Standard high-density micromass forms discrete nodular islands ($\approx 25\text{--}30\%$ area fraction), exogenous TGF-$\beta_2$ fuses islands into bar-like labyrinthine stripes ($\approx 45\text{--}55\%$ area fraction), and anti-fibronectin / RGD haptotaxis blockade ($\chi_F = 0$) abolishes mesenchymal condensation.
   - SHA-256: `33d554e44b712e2b7d277044801369362417591008adf2812e9c304b7fcde085`.

2. **`chick_limb_chondrogenesis_specimen.jpg`**
   - *Gallus gallus* forelimb osteological specimen showing the proximal-to-distal transition from humerus (stylopod, $m=1$) $\to$ radius and ulna (zeugopod, $m=2$) $\to$ carpometacarpus and digits II, III, IV (autopod, $m=3$).
   - Source: [Wikimedia Commons `File:Gallus_wingbones_01.JPG`](https://commons.wikimedia.org/wiki/File:Gallus_wingbones_01.JPG) (Author: Conty, License: CC BY 3.0).
   - SHA-256: `771e26f9fe8c813de108d87086e896597b450c09e8e9a706bb0f1edddeb26e41`.

3. **`scout_manifest.json`**
   - Step 1 (`01_scout`) OpenAlex + Wikimedia Commons discovery dossier for Slot 014.
   - SHA-256: `c810bd77e66557c8372e0f5ba74dcb630eb93aecdb7315cc242e4347b6c0d15f`.

4. **`paper_citations.bib`**
   - BibTeX citations for Hentschel et al. (2004), Kiskowski et al. (2004), Miura & Shiota (2000), Chaturvedi et al. (2005), and Christley et al. (2007); Pass 3 adds Graner & Glazier (1992), Glazier & Graner (1993), Savill & Hogeweg (1997), Hillen & Painter (2001), and Tapia & D'Souza (2011).
   - SHA-256 (Pass 3): `c2be675a1125adf799bc3ddb069c921d55b31b5597154a7efb4e71850d4e42fc`.

5. **`kiskowski_ggh_cpm_reference.json`** (Pass 3)
   - Parameter and metric table for the Glazier-Graner-Hogeweg cellular Potts cell lattice coupled to the continuum PDE (`simulate_ggh_cellular_potts_condensation`): Graner & Glazier (1992) contact energies $J_{R1R1}=14,\ J_{R2R2}=2,\ J_{R1R2}=11,\ J_{cM}=16$, $T=10$, $\lambda_V=6$, $V_T=25$ px, haptotaxis strength $\mu_F=10$, Monte Carlo step budgets, the Hill-switch cell-state coupling, the discrete fibronectin deposition law, and the discrete-continuum parity metrics (condensation index, heterotypic boundary fraction, $r$ of discrete vs continuum $R_2$, wing-bud compartment counts, `clip_fraction_R`).
   - Also records the volume-filling haptotaxis form (Hillen & Painter 2001) adopted for the 2D micromass continuum solver and why (Pass 2 solver collapse).
   - SHA-256: `a7d13368e41d762a9cbf86337e575173a5fde9328cb90865cfb7877791e88a9f`.

## Pass 3 findings recorded here

- The Pass 2 micromass "nodular spots (19.8%) vs labyrinthine stripes (44.8%)" contrast was an artefact of thresholding a chemotactically collapsed field; with volume-filling haptotaxis both haptotactic regimes are labyrinthine bands (islands differ in count and mean size, not in spot/stripe topology).
- The 1D stage and 2D wing-bud continuum solvers still pin part of $R(x)$ at the clip bounds (reported as `clip_fraction_R` in `measure_morphoeval()`); their condensation *counts* are preserved but their *amplitudes* are not resolved. Carried to Pass 4.
- The discrete GGH model reproduces the zeugopod (2) and autopod (3) condensations but not the stylopod (1): the stylopod activator peak $c_a^{\max}=0.35$ never crosses the half-maximal Hill switch $K_a=0.60$.

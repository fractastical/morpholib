# Outreach Email Package & Manuscript Summary: D'Arcy Thompson's 1917 Fish Transformations

## 1. Compiled Manuscript & High-Resolution Visual Assets

- **Compiled LaTeX Manuscript PDF (16 pages):** `docs/thompson_1917_fish_transformations_manuscript.pdf`
- **LaTeX Source:** `docs/thompson_1917_fish_transformations_manuscript.tex`
- **Full-Page Manuscript Figures:**
  1. `thompson_stick_vs_silhouette_reality.png` — 4x5 plate contrasting the 8-vertex 17-landmark skeletal wireframe against the 600-semilandmark whole-body outline across Thompson's 1917 pairs.
  2. `thompson_silhouette_area_difference.png` — 3x5 plate quantifying the exact non-overlapping area ($|\Omega_A \Delta \Omega_B| = |\Omega_B \setminus \Omega_A|\text{ [Red Missing]} + |\Omega_A \setminus \Omega_B|\text{ [Blue Overhang]}$) across Unwarped 4p Procrustes similarity (Row 1), 17-Landmark Fit (Row 2), and Direct 600-Semilandmark Outline Fit (Row 3).
  3. `thompson_four_families_atlas.png` — 3x4 macroevolutionary atlas across 791 species, 4,505 pairs, and 4,334 time-calibrated divergences on the Rabosky et al. (2018) megaphylogeny.
  4. `thompson_rigor_ablation_and_jacobian.png` — 2x2 methodological stress-test plate: (A) landmark partition ablation (9-Craniofacial-LM vs. All 17-LM vs. 10-Outer-Trunk-LM vs. Direct 600-Semilandmark Outline across all 6 pairs), (B) Jacobian diffeomorphic bijectivity ($\det D\Phi(x,y) > 0$), (C) bidirectional invertibility ($S \leftrightarrow T$), and (D) unbinned logistic disparity transition & 187 phylogenetically independent node-disjoint pairs.
  5. `docs/outreach_fig4_wainwright_600pt_and_convergence.png` — 2x3 benchmark plate: (A–C) head-to-head comparison of 5 competing 600-semilandmark outline methods (Elliptic Fourier Descriptors, 600-semilandmark curve TPS, 6p uniform affine, 12p quadratic trend surface, and 8p Thompson grid) on diffeomorphic bijectivity and out-of-sample interior landmark prediction; (D–F) empirical non-affine Relative Warps (Rohlf 1993), Phylomorphospace (`PC1` vs. `PC2`; Sidlauskas 2008), and Phenotypic Trajectory Analysis ($r_{\text{vec}} = \cos \theta$; Adams & Collyer 2009) across convergent deep-bodied and elongate/fusiform clades.
- **Tailored Email Summary Plates (1-page visual attachments):**
  1. `docs/outreach_fig1_bookstein_silhouette_paradox.png` (for Prof. Fred L. Bookstein)
  2. `docs/outreach_fig2_price_wainwright_horizon.png` (for Prof. Samantha A. Price, Dr. Sarah T. Friedman, and Prof. Peter C. Wainwright)
  3. `docs/outreach_fig3_evans_larouche_modularity.png` (for Prof. Kory M. Evans and Dr. Olivier Larouche)
  4. `docs/outreach_fig4_wainwright_600pt_and_convergence.png` (Follow-up for Prof. Peter C. Wainwright addressing 600-semilandmark methods & phylomorphospace convergence)

---

## 2. Follow-Up Email Draft — Prof. Peter C. Wainwright (UC Davis)

**To:** `pcwainwright@ucdavis.edu`  
**Subject:** Re: Testing Thompson's 1917 grids — (1) 8p Thompson vs. standard 600-semilandmark TPS/EFD & (2) Relative Warps, Phylomorphospace, and Phenotypic Trajectory Analysis  
**Attachments:**
1. `docs/outreach_fig4_wainwright_600pt_and_convergence.png` (inline + attached)
2. `docs/outreach_fig2_price_wainwright_horizon.png` (updated with continuous Procrustes RMS vs. intrageneric disparity $\sigma_{\text{genus}}$)
3. `docs/thompson_1917_fish_transformations_manuscript.pdf` (updated 16-page manuscript with Section 6)

---

Hi Peter,

Thank you so much for talking with us today — your two critiques hit the exact methodological heart of the paper:
1. *When we move from 17 landmarks to 600 outline points, how do we show D'Arcy Thompson's 8-parameter coordinate law is actually better than any standard 600-semilandmark outline method?*
2. *How do we take a genuinely data-driven approach to the morphospace and test phyletic vs. convergent evolution across the 791 species (and normalize all our terminology onto standard geometric morphometric & ichthyological conventions)?*

We just ran both sets of analyses across all six 1917 pairs and all 791 species (48 families, 62 multi-species genera), updated the manuscript (Section 6, attached), and summarized the results in the attached 6-panel plate (**`outreach_fig4_wainwright_600pt_and_convergence.png`**):

### 1. Holding the 600-Semilandmark Input Fixed: Why 8p Thompson Outperforms Standard 600-Semilandmark Curve TPS, EFD, and Polynomial Surfaces (Panels A–C)

Previously, comparing a 17-landmark fit against a 600-point Thompson fit changed the input data ($17 \to 600$ points). To make a strictly fair comparison, we **held the 600-semilandmark outline input fixed** across all six of Thompson's 1917 pairs and benchmarked **five competing 600-point outline methods** head-to-head on four criteria: (i) non-overlapping silhouette area $|\Omega_A \Delta \Omega_B|$, (ii) Chamfer contour residual, (iii) **global 2D diffeomorphic bijectivity ($\det D\Phi(x,y) > 0$ everywhere, no tissue grid folding)**, and (iv) **out-of-sample prediction of the 14 held-out interior skeletal landmarks** (eye, jaw, opercle, pectoral & median fin insertions) when the coordinate map is fitted *solely* to the 600-point outer outline:

* **Elliptic Fourier Descriptors (EFD, 8p = 2 harmonics; Kuhl & Giardina 1982):** Parameterizes only a 1D closed boundary curve $(x(s), y(s))$ and defines **no 2D ambient coordinate transformation $\Phi: \mathbb{R}^2 \to \mathbb{R}^2$** (`0%` ability to predict where internal craniofacial or axial skeletal landmarks map inside the body).
* **Standard 600-Semilandmark Curve Thin-Plate Spline (Bookstein 1997; Panel A):** Because curve-semilandmark TPS pairs boundary points by perimeter arc-length ($s \in [0, 1]$), disproportionate hypertrophy of the dorsal and anal fin lobes in *Pomacanthus*, *Sternoptyx*, *Scorpaena*, *Antigonia*, and *Mola* slides semilandmarks non-homologously along the contour, **folding and inverting the 2D interior coordinate grid on `6 of 6` pairs (`0%` globally diffeomorphic; $\min \det D\Phi = -6.76$, with up to $25.1\%$ of the body domain inverted, shown in red in Panel A)** and degrading **out-of-sample prediction of held-out interior skeletal landmarks on `6 of 6` pairs** (up to **$2.4\times$ higher Procrustes RMS**: `0.072` vs. **`0.032`** on *Polyprion* $\to$ *Pristigenys*; `0.107` vs. **`0.045`** on *Pristigenys* $\to$ *Scorpaena*; `0.134` vs. **`0.091`** on *Scorpaena* $\to$ *Antigonia*).
* **12-Parameter Quadratic Polynomial Trend Surface (Bookstein 1991) on the 600 Semilandmarks:** Folds the 2D coordinate grid ($\det D\Phi \le 0$) on **`4 of 6` pairs** (`33%` globally diffeomorphic) and degrades held-out interior landmark prediction by up to **$2.1\times$** (`0.092` vs. **`0.045`** on *Pristigenys* $\to$ *Scorpaena*).
* **6-Parameter Uniform (Affine) Component vs. 8-Parameter Thompson Grid on the 600 Semilandmarks (Panels B–C):** Adding Thompson's 2 non-affine bending parameters to the 6p uniform component cuts Chamfer contour loss by **$16.2\%\text{–}32.2\%$** on the exact same 600 points (reducing $|\Omega_A \Delta \Omega_B|$ from $27.4\% \to \mathbf{22.6\%}$, $\mathrm{IoU} = 0.794$ on *Scarus* $\to$ *Pomacanthus* under multi-start optimization). **Thompson's 8-parameter coordinate grid is the only 600-point method that simultaneously achieves low non-affine contour error, `100%` global diffeomorphic bijectivity (`6/6` pairs, $\det D\Phi > 0$), and accurate out-of-sample prediction of held-out interior skeletal landmarks from the outer outline alone.**

### 2. Data-Driven Relative Warps (Rohlf 1993), Phylomorphospace (Sidlauskas 2008), and Phenotypic Trajectory Analysis of Convergent Evolution (Adams & Collyer 2009) (Panels D–F)

To test the morphospace and convergent evolution directly from the 791-species data without assuming Thompson's formulas *a priori*, we ran three standard geometric morphometric analyses:

* **Leave-One-Family-Out Cross-Validation Against Empirical 8p Relative Warps (Rohlf 1993; Panel D):**
  Decomposing total Procrustes variation from the consensus configuration $M_0$ into the **6-parameter uniform (affine) component** plus the **non-affine Relative Warps** ($\mathrm{RW1}, \mathrm{RW2}$ learned strictly on the 47 training families in each fold across all 48 families), the 6p uniform component explains **`25.1%` median** (`29.3%` mean) of shape variance across the 791 held-out species, and the optimal data-driven **8p Empirical Relative-Warp ceiling (`6p Uniform + RW1 + RW2`) explains `55.3%` median** (`53.1%` mean). Without a single data-trained eigenvector, **D'Arcy Thompson's 1917 analytical 8p coordinate laws explain `50.9%` median (`50.8%` mean) — capturing `92.0%` (median) / `95.6%` (mean) of the theoretical data-driven 8-parameter Relative-Warp variance ceiling!**
* **Capturing the Primary Phylomorphospace Plane (`PC1 + PC2`; Sidlauskas 2008; Panel D–E):**
  Across the 62 multi-species reef-fish genera in phylomorphospace, the 6p uniform component captures only **`33.1%` of Phylomorphospace `PC1`** ($34.6\%$ of clade variance) and **`48.9%` of the `(PC1, PC2)` plane**, because `PC1` couples body deepening with non-affine anterior-to-posterior fanning of the dorsal and anal fin insertions (`LM 3–4, 7–8`). By contrast, **8-parameter Thompson grids capture `83.4%` of `PC1`, `95.8%` of `PC2`, and `88.6%` of the primary `(PC1, PC2)` phylomorphospace plane**.
* **Phenotypic Trajectory Analysis (Adams & Collyer 2009) & Many-to-One Mapping (Wainwright et al. 2005; Panels E–F):**
  Applying Phenotypic Trajectory Analysis from the consensus ancestral shape $M_0$ to each genus mean ($\Delta Y_g = \Delta Y_g^{\text{Thompson}} + \Delta Y_g^{\text{Modular}}$) across **5 distantly related deep-bodied reef-fish families (`> 95 Myr` apart: Chaetodontidae, Pomacanthidae, Acanthuridae, Pomacentridae, Siganidae; 38 inter-family pairs)** and **6 elongate/fusiform families (14 inter-family pairs)** shows:
  1. Along convergent trajectories from $M_0$, 8p Thompson coordinate grids add **`+16.3%` to `+43.6%` explained Procrustes variance** over the 6p uniform component (*Siganus* `4.2% → 41.6%`, *Acanthurus* `19.0% → 62.6%`, *Ctenochaetus* `37.2% → 68.7%`, *Chaetodontoplus* `23.7% → 54.4%`, *Pomacanthus* `37.0% → 53.3%`).
  2. Distantly related families converging on the deep-bodied or elongate/fusiform morphotype exhibit **high phenotypic trajectory correlation** along their global 8p Thompson vectors (**median $r_{\text{vec}} = \cos \theta = +0.693$** in deep-bodied clades and **$r_{\text{vec}} = +0.714$** in elongate/fusiform clades, vs. **$-0.699$** between divergent clades, $p < 10^{-8}$).
  3. At the same time, their **local craniofacial and fin-insertion modular residuals are nearly twice as divergent ($r_{\text{vec}} = +0.416$)** — giving a direct coordinate-field demonstration of **many-to-one mapping of form to function (Wainwright et al. 2005)**: butterflyfishes, angelfishes, surgeonfishes, damselfishes, and rabbitfishes follow the *same* global 8p Thompson trajectory into deep-bodied phylomorphospace, while modifying their internal jaws and fin-base insertions along clade-specific paths.

We also normalized all terminology and axis labels across the manuscript and figures (including plotting continuous post-warp Procrustes RMS residuals directly against the intrageneric disparity band $\sigma_{\text{genus}} \le 0.0110$ in `outreach_fig2_price_wainwright_horizon.png`). Thank you again for the extraordinarily helpful guidance!

Best regards,

Joel Dietz (Project Delphi · [github.com/fractastical/morpholib](https://github.com/fractastical/morpholib))

---

## 3. Email Draft 1 — Prof. Fred L. Bookstein (University of Vienna / University of Michigan)

**To:** `fred.bookstein@univie.ac.at` (or `fred@cns.nyu.edu` / `fbookst@umich.edu`)  
**Subject:** Empirical test of your 2023/2024 *Evolutionary Biology* critiques across Thompson's 1917 fish pairs & 791 species (17-landmark wireframes vs. 600-semilandmark outlines)  
**Attachments:**
1. `docs/outreach_fig1_bookstein_silhouette_paradox.png` (inline + attached)
2. `docs/thompson_1917_fish_transformations_manuscript.pdf`

---

Dear Professor Bookstein,

Your recent papers in *Evolutionary Biology* (*"Reworking geometric morphometrics into a methodology of transformation grids,"* 2023; *"Quadratic trends: A morphometric tool both old and new,"* 2024) made a point we took as our starting hypothesis: exact Thin-Plate Spline interpolants fit any pair of landmark configurations with zero residual and therefore cannot falsify D'Arcy Thompson's hypothesis of low-dimensional coordinate transformations (*On Growth and Form*, 1917, Ch. XVII, pp. 748–751, Figs. 373–382).

Inspired by your critique, we formalized all four of Thompson's 1917 fish coordinate families as explicit 5- to 8-parameter maps — **I. Oblique shear** (5p, Figs. 373–374), **II. Complex Möbius coaxial circles** (8p, Figs. 375–376), **III. Bilinear fan of coordinates** (8p, Figs. 377–380), and **IV. Hyperbolic/radial flare** (8p, Figs. 381–382) — and benchmarked them against 4p Procrustes similarity, the 6p uniform (affine) component, 12p quadratic polynomial trend surfaces, and 34p exact TPS across all six of Thompson's 1917 pairs and 4,505 pairs from the 791-species Indo-Pacific reef-fish dataset (Claverie & Wainwright 2014; Friedman et al. 2019).

When we inspected the resulting transformation grids visually, we found a methodological discrepancy that bears on your 2023–2024 papers (summarized in the attached plate **`outreach_fig1_bookstein_silhouette_paradox.png`**):

1. **Why Fitting to the Standard 17 Ichthyological Landmarks Degrades Whole-Body Fish Silhouettes (Panels A–B):**
   In the standard 17-landmark reef-fish protocol, **9 of the 17 landmarks (53% of the least-squares weight) sit inside the craniofacial partition**, while the remaining 8 only mark where the fin bases insert into the body wall — omitting 100% of the distal dorsal fin, anal fin, caudal fin, and ventral keel. Consequently, when we fit Thompson's 8-parameter coaxial grid (or even an exact 34-parameter TPS!) to the 17 landmarks of *Scarus ghobban* $\to$ *Pomacanthus semicirculatus* (Figs. 375–376) and carry the real 600-semilandmark fish outline along with the grid, the 9 craniofacial landmarks steer the grid the wrong way for the posterior body: the symmetric non-overlapping area ($|A \Delta B| = |B \setminus A| + |A \setminus B|$) **worsens from 40.6% (unwarped) to 56.2% of target fish area** (creating a **47.2% blue posterior overhang**, Panel B), and silhouette $\text{IoU}$ drops from $0.662 \to 0.618$ ($0.629$ under exact 34p TPS). The same degradation occurs on *Scorpaena* $\to$ *Antigonia* ($|A \Delta B|: 102.2\% \to 143.7\%$) and *Argyropelecus* $\to$ *Sternoptyx* ($58.4\% \to 63.0\%$).

2. **Direct 600-Semilandmark Outline Fitting (Anchored on Your Two-Point Baseline) Recovers Thompson's 1917 Transformations (Panels C–E):**
   Thompson never fitted 17 craniofacial-heavy points — he deformed continuous outlines of the entire fish including its median fins. When we fit the **exact same 5- to 8-parameter Thompson coordinate families directly to the 600-semilandmark outlines** (anchored only at your two axial baseline endpoints, snout tip $\text{LM } 2$ and caudal peduncle midpoint $\tfrac{1}{2}(\text{LM } 5 + \text{LM } 6)$, to prevent tangential perimeter sliding, with $100\%$ positive Jacobian determinant $\det D\Phi > 0$ everywhere), Thompson's transformations are recovered across every 1917 pair (whereas a standard 600-semilandmark curve TPS folds the interior coordinate grid with $\det D\Phi < 0$ on all 6 pairs):
   - *Scarus* $\to$ *Pomacanthus* (8p Coaxial, Panel C): Non-overlapping area $|A \Delta B|$ is **cut in half from 56.2% to 28.5%** ($\text{IoU}: 0.618 \to \mathbf{0.749}$; reverse *Pomacanthus* $\to$ *Scarus* $= \mathbf{26.0\%}$, $\text{IoU} = \mathbf{0.765}$).
   - *Scorpaena* $\to$ *Antigonia* (8p Fan, Panel D): $|A \Delta B|$ drops **$2.5\times$ from 143.7% to 57.6%** ($\text{IoU}: 0.389 \to \mathbf{0.599}$).
   - *Pristigenys* $\to$ *Scorpaena* (8p Fan, Panel E): $|A \Delta B|$ drops **$3.4\times$ from 120.1% to 35.5%** ($\text{IoU} = \mathbf{0.694}$).
   - *Polyprion* $\to$ *Pristigenys* (8p Fan): $|A \Delta B|$ drops to **17.2%** ($\text{IoU} = \mathbf{0.839}$).

3. **8-Parameter Thompson Coordinate Laws vs. 12-Parameter Quadratic Trends Across 4,505 Pairs (Panel F):**
   On the 17-landmark configuration ($n = 34$ scalar coordinates), your 2024 quadratic trend surface (12p) and Thompson's 8-parameter families can be compared directly via small-sample $\text{AIC}_c$ and Leave-One-Landmark-Out cross-validation (LOO-CV). Across 4,505 pairs, 8-parameter Thompson grids sit at the parsimony optimum: they beat the 6p uniform affine component in **82.2%** of intrafamily pairs and beat 12p quadratic trends by $\text{AIC}_c$ in **84.9%** of intrafamily and **92.1%** of interfamily pairs, while capturing **92.0%** of the optimal 8-parameter empirical Relative-Warp variance (Rohlf 1993) across all 791 species.

The one thing we would ask you to judge: does fitting an exact TPS to 17 landmarks and carrying the outline along constitute a fair test of your 2023 point, or have we misread it? The manuscript is attached; the scripts are public. We will record your answer, either way, in the acknowledgements and our validation log.

With warm regards,

Joel Dietz (Project Delphi · [github.com/fractastical/morpholib](https://github.com/fractastical/morpholib))

---

## 4. Email Draft 3 — Prof. Kory M. Evans (Rice University) & Dr. Olivier Larouche

**To:** `kory.evans@rice.edu`, `olivier.larouche@univ-lille.fr` (or current institutional address)  
**Subject:** How teleost craniofacial-trunk-fin modularity explains the 17-landmark vs. outline discrepancy in D'Arcy Thompson's 1917 fish grids  
**Attachments:**
1. `docs/outreach_fig3_evans_larouche_modularity.png` (inline + attached)
2. `docs/thompson_1917_fish_transformations_manuscript.pdf`

---

Dear Professor Evans and Dr. Larouche,

Your work on craniofacial-trunk-fin modularity and evolutionary allometry across teleost fishes inspired us to look closely at what happens to individual anatomical partitions when D'Arcy Thompson's famous 1917 coordinate transformations (*On Growth and Form*, Figs. 373–382) are tested across 791 Indo-Pacific reef fishes (4,505 pairs) and all six of Thompson's original 1917 species pairs.

In doing so, we found a geometric consequence of teleost modularity that explains why standard 17-landmark morphometrics and continuous whole-body outlines give opposite answers on Thompson's classic pairs (shown in the attached plate **`outreach_fig3_evans_larouche_modularity.png`**):

1. **Why 9/17 Craniofacial Landmarks Bias Global Coordinate Grids (Panels A–C):**
   In the standard 17-landmark teleost scheme, 9 of the 17 points ($53\%$ of the ordinary least-squares weight) lie inside the craniofacial partition ($\text{LM } 1\text{–}2, 11\text{–}17$), while the other 8 mark only the basal insertion points of the fins ($\text{LM } 3\text{–}10$). When craniofacial proportions and median-fin contours decouple across families, fitting Thompson's grid to the 17 landmarks forces the grid to satisfy the 9 craniofacial points at the expense of the posterior trunk and fins:
   - On *Scorpaena scrofa* $\to$ *Antigonia capros* (Figs. 379–380, 8p Fan), the 17-landmark fit inflates the symmetric non-overlapping area $|A \Delta B|$ on the real 600-semilandmark fish outline from **102.2% (unwarped) to 143.7%**, creating a **135.4% blue posterior overhang** (Panel B).
   - When the exact same 8-parameter fan is fitted **directly to the 600-semilandmark whole-body outline** (Panel C), $|A \Delta B|$ is cut **$2.5\times$ to 57.6%** ($\text{IoU}$ jumps from $0.389 \to \mathbf{0.599}$).
   - Across all six of Thompson's 1917 pairs (Panel D), 17-landmark fitting worsens the whole-body silhouette non-overlapping area in three pairs (*Argyropelecus* $\to$ *Sternoptyx*, *Scarus* $\to$ *Pomacanthus*, and *Scorpaena* $\to$ *Antigonia*), whereas direct 600-semilandmark outline fitting substantially reduces $|A \Delta B|$ in every single pair with $100\%$ diffeomorphic bijectivity.

2. **The *Diodon hystrix* $\to$ *Mola mola* Trunk-vs.-Fin Allometric Partition (Panel E):**
   On Thompson's culmination pair (*Diodon* $\to$ *Mola mola*, Figs. 381–382, $79.3\text{ Myr}$ divergence), an 8-parameter exponential radial flare ($x' = u + r_1 v^2,\; y' = v e^{r_2 u}$) explains **79.7% of the 17-landmark trunk variance** compared to just **25.2% for the 6p uniform affine component** ($\Delta\text{AIC}_c = -41.9$, LOO-CV $= 0.0461 < d_0 = 0.0814$). At the same time, *Mola mola*'s full 600-semilandmark silhouette (including its hypertrophied dorsal and anal fins) is **$2.26\times$ taller** than its 17-landmark trunk wireframe ($0.786$ vs. $0.349$), providing a textbook separation between axial truncation (captured by the 8p radial grid) and median-fin hypertrophy.

The one thing we would ask you to judge: is the craniofacial-vs-fin decoupling we measure the modularity your 2017–2018 papers describe, or a different partition? The manuscript is attached; the scripts are public. We will record your answer, either way, in the acknowledgements and our validation log.

Warm regards,

Joel Dietz (Project Delphi · [github.com/fractastical/morpholib](https://github.com/fractastical/morpholib))

---

## 5. Email Draft 4 — Prof. Miriam L. Zelditch (University of Michigan)

*Redrafted 2026-11-04 to ask for one specific judgement rather than co-authorship. Status: not sent. The earlier, longer draft is in git history (commit cd62496).*

**To:** `zelditch@umich.edu`  
**Subject:** Peter Wainwright suggested you: is this a fair test of an 8-parameter Thompson grid against thin-plate splines?  
**Attachments:** `docs/outreach_fig4_wainwright_600pt_and_convergence.png` (one page). Manuscript on request.

---

Dear Professor Zelditch,

Peter Wainwright suggested we ask you one question about a test of D'Arcy Thompson's 1917 fish grids. A yes, a no, or "wrong criterion" is all we need in reply.

**What we did.** We wrote Thompson's four Chapter XVII fish families as explicit 5–8-parameter maps and fitted them, and a standard 600-semilandmark curve thin-plate spline, to the *same* 600-point outlines of his six 1917 pairs. We scored both on (a) whether the interior grid stays orientation-preserving (det DΦ > 0 everywhere) and (b) how well a map fitted to the outline alone predicts the 14 interior skeletal landmarks it never saw. On the 791-species Claverie–Wainwright set we also compared Thompson's fixed 8 parameters against the best data-driven 8-parameter basis (uniform + RW1 + RW2) under leave-one-family-out.

**What came out.** Curve TPS folds on 6/6 pairs and predicts the held-out interior landmarks worse on 6/6 (e.g. Procrustes RMS 0.072 vs 0.032, *Polyprion* → *Pristigenys*); the 8p grid folds on 0/6. Thompson's fixed parameters explain 50.9% of shape variance (median) against the 55.3% data-driven ceiling.

**What we are asking you to judge.** Is criterion (b) fair? A spline is an interpolant, not a predictor of unseen interior points, so a critic could say we scored TPS on a job it was never meant to do and that the comparison is rigged toward a low-parameter global family. If the criterion is wrong, we would like to know what the right one is; we will run it and report the result either way. A second question, only if you have time: is the relative-warp ceiling the right yardstick for "how much of fish shape is Thompsonian", or does your head–body–fin modularity work imply the comparison should be made per module?

The figure, the script (`thompson/run_wainwright_600pt_and_convergence_tests.py`) and the numbers file are public at github.com/fractastical/morpholib. With your permission we would record your answer, whichever it is, in the paper's acknowledgements and in our validation log.

With thanks,

Joel Dietz (Project Delphi · [github.com/fractastical/morpholib](https://github.com/fractastical/morpholib))

---

## 6. Email Draft 5 — Prof. Anjali Goswami (Natural History Museum, London / UCL) & Dr. Ryan N. Felice (University College London)

*Redrafted 2026-11-04 to ask for one specific judgement rather than co-authorship. Status: not sent. The earlier, longer draft is in git history (commit cd62496).*

**To:** `a.goswami@nhm.ac.uk`, `r.felice@ucl.ac.uk`  
**Subject:** Peter Wainwright suggested you: does our 600-semilandmark curve TPS fold because we did not slide the semilandmarks?  
**Attachments:** `docs/outreach_fig4_wainwright_600pt_and_convergence.png` (one page). Manuscript on request.

---

Dear Professor Goswami and Dr. Felice,

Peter Wainwright suggested you as the people to ask one technical question about semilandmark methods. A short answer is all we need.

**The observation.** Fitting a 600-semilandmark curve thin-plate spline between each of D'Arcy Thompson's six 1917 fish pairs (*Scarus* → *Pomacanthus*, *Diodon* → *Mola*, and so on), with semilandmarks paired by perimeter arc-length, the interior grid folds (det DΦ < 0 somewhere, up to 25% of the body on *Scorpaena* → *Antigonia*) on all six pairs. An 8-parameter Thompson coordinate grid fitted to the same 600 points never folds and predicts the 14 held-out interior skeletal landmarks better on all six pairs.

**The question.** We did not slide the semilandmarks (by bending energy or Procrustes distance, as in your 2018 *Evolution* and *Systematic Biology* work). Is the folding an artefact of that omission, so that a properly slid TPS would be diffeomorphic and the comparison collapses? Or does the hypertrophy of the median fin lobes between these pairs defeat sliding too? We would rather hear "your TPS baseline is wrong, here is the right one" from you now than from a referee later, and we will rerun the benchmark with whatever sliding protocol you name and report the outcome either way.

A second question, only if you have time: as a criterion for comparing an outline method with a global coordinate family, is out-of-sample prediction of interior landmarks from the outline alone legitimate, or does it favour the low-parameter model by construction?

The script and numbers file (`thompson/run_wainwright_600pt_and_convergence_tests.py`, `thompson_wainwright_600pt_and_convergence.json`) are public at github.com/fractastical/morpholib. With your permission we would record your answer in the acknowledgements and in our validation log.

With thanks,

Joel Dietz (Project Delphi · [github.com/fractastical/morpholib](https://github.com/fractastical/morpholib))

---

## 7. Email Draft 6 — Dr. Emma Sherratt (University of Adelaide), Prof. J. Scott Keogh & Dr. Ian G. Brennan (Australian National University)

**To:** `emma.sherratt@adelaide.edu.au`, `scott.keogh@anu.edu.au`, `ian.brennan@anu.edu.au`  
**Subject:** Pre-registered tests of D'Arcy Thompson's 7 global coordinate modes on your Australian Anura (166 spp), *Varanus* crania (1,091 skulls), and Leporidae crania (317 skulls) datasets  
**Attachments:**
1. `docs/portfolio/fig_australian_frogs.png` (inline + attached)
2. `docs/portfolio/fig_varanus_lizards.png` (inline + attached)
3. `docs/portfolio/fig_leporidae_skulls.png` (inline + attached)

---

Dear Dr. Sherratt, Professor Keogh, and Dr. Brennan,

To test how D'Arcy Thompson's 1917 global coordinate transformations (*On Growth and Form*, Ch. XVII) generalize beyond teleost fishes across tetrapod radiations, we built a pre-registered phylogenetic morphospace and prediction pipeline ([`fractastical/morpholib`](https://github.com/fractastical/morpholib), with hypotheses SHA-256-locked and published as cryptographic Nanopublications [`RAQ_Giwd735...`](https://registry.knowledgepixels.com/np/RAQ_Giwd735IPhYg-cOMyPE6t4K9WPM9XlijGkNC6_eJ4) before running) and evaluated it across three of your landmark datasets:

1. **Continental Radiation of Australian Anura (`166` Species, `55` Lateral Body/Tail Landmarks; Sherratt, Vidal-García, Anstis & Keogh 2017, *Evol. Biol.*):**
   - Unlike reef fishes (`57.9%` of interspecific variance in Thompson's 7 global modes due to decoupled spiny dorsal/anal fin insertions), your 55-landmark anuran contours have **`66.2%` of between-species Procrustes variance** inside the 7 a-priori Thompson modes (`88.9%` in 7-D PCA), with strong phylogenetic signal ($\hat{\lambda} = 0.682$ in Thompson-7, $\hat{\lambda} = 0.737$ in full space; **Hypothesis AF2 ✓**).
   - Leave-one-species-out phylogenetic prediction across the 166 species significantly beats the grand consensus (**median Procrustes RMS ratio $= 0.805$, $p = 1.5 \times 10^{-14}$; Hypothesis AF3 ✓**), with the 7 Thompson modes alone capturing **`60.2%`** of the full-space phylogenetic prediction gain (`fig_australian_frogs.png`).
2. **Indo-Australasian *Varanus* Monitor Lizard Crania (`1,091` Dorsal [33 LM] & `1,098` Lateral [10 LM] Skulls Across `80` Taxa; Brennan et al. 2021, *Syst. Biol.*):**
   - Across the 80 *Varanus* taxa (from dwarf *V. sparnus* to giant *V. komodoensis*), Thompson's 7 global modes capture **`59.5%` of interspecific dorsal cranial variance** and **`55.2%` of lateral cranial variance** (**Hypothesis VL1 ✓**; `fig_varanus_lizards.png`).
   - While full-space leave-one-species-out prediction overfits local cranial landmarks (`LOO ratio = 1.00`), restricting phylogenetic prediction strictly to **Thompson's 7 global coordinate modes beats the consensus (`LOO ratio = 0.893`, $p < 10^{-6}$)**, while intraspecific cranial variation aligns strongly with interspecific divergence in the local modular residual subspace ($|\cos\theta|_{\text{res}} = 0.937$ vs. $0.106$ in T7).
3. **Leporidae Mammalian Crania (`317` Skulls, `22` Species, `74` Landmarks; Sherratt & Kraatz 2023):**
   - Captures **`51.1%`** of interspecific variance in Thompson's 7 modes with pure Brownian phylogenetic signal ($\hat{\lambda} = 1.000$) and `0.89×` LOO prediction error (`fig_leporidae_skulls.png`).

All code, pre-registered JSON manifests, and per-species predictions are openly archived in our repository ([`docs/portfolio/PORTFOLIO_STATUS.md`](https://github.com/fractastical/morpholib/blob/main/docs/portfolio/PORTFOLIO_STATUS.md)). The one thing we would ask you to judge: are the landmark subsets we used on each of your three datasets the ones you would use for a whole-body shape question? We will record your answer, either way, in our validation log.

Warm regards,

Joel Dietz (Project Delphi · [github.com/fractastical/morpholib](https://github.com/fractastical/morpholib))

---

## 8. Email Draft 7 — Prof. Simon Joly (Université de Montréal / Montreal Botanical Garden)

**To:** `simon.joly@umontreal.ca`  
**Subject:** Pre-registered test of D'Arcy Thompson's global coordinate modes & developmental channelling on your Antillean Gesneriaceae floral corolla dataset (296 flowers, 48 species)  
**Attachments:**
1. `docs/portfolio/fig_antillean_flowers.png` (inline + attached)
2. `docs/outreach_fig11_dataset_portfolio.png` (inline + attached)

---

Dear Professor Joly,

In Chapter XVI of *On Growth and Form* (1917), D'Arcy Thompson pointed to sympetalous floral corollas (such as tubular vs. campanulate bells) as the botanical archetype of smooth coordinate transformations, yet never tested them quantitatively across a phylogeny. When we built our pre-registered cross-kingdom morphospace pipeline ([`fractastical/morpholib`](https://github.com/fractastical/morpholib)), we selected your Antillean Gesneriaceae dataset (Joly et al. 2018, *Evolution* `72:244–260`; `296` corollas across `48` species and `32` 2D landmarks on `BestTree.tre`) as our primary seed-plant test (`thompson/portfolio/groups/antillean_flowers_hypotheses.json`, SHA-256-locked in Nanopub [`RAQ_Giwd735...`](https://registry.knowledgepixels.com/np/RAQ_Giwd735IPhYg-cOMyPE6t4K9WPM9XlijGkNC6_eJ4)).

Scoring the pre-registered hypotheses (`docs/portfolio/antillean_flowers.md`, attached in **`fig_antillean_flowers.png`**) agreed with your 2018 findings (**`3/4` hypotheses supported, Brier score `0.179`**):

1. **Angiosperm Corolla Tubes Are Ultra-Thompsonian (`82.1%` Variance in 7 Modes; Hypothesis FP1 ✓):**
   Across all 48 Gesneriaceae species spanning hummingbird, bat, and generalist pollination syndromes, D'Arcy Thompson's 7 a-priori global coordinate modes capture **`82.1%` of between-species Procrustes variance** (`96.8%` in fitted 7-D PCA), far exceeding teleost fish bodies (`57.9%`) and insect wings (`38.8%–49.0%`).
2. **Leave-One-Species-Out Phylogenetic Prediction Beats Consensus (`0.69×` Error; Hypothesis FP3 ✓):**
   On `BestTree.tre`, leave-one-species-out phylogenetic prediction reduces median Procrustes RMS error to **`0.689×` the grand consensus** ($p = 0.0055$; $\hat{\lambda}_{\text{full}} = 0.852$).
3. **Near-Perfect Developmental Channelling Inside the 7-D Thompson Subspace (`|cos θ| = 0.978`; Hypothesis FP4 ✓):**
   In your 2018 *Evolution* paper, you showed that intraspecific floral variation channels evolutionary transitions between pollination syndromes. When we decompose that alignment into Thompson's 7 global growth modes vs. the 53-dimensional non-Thompson residual subspace, **intraspecific floral variation aligns with interspecific macroevolutionary divergence at $|\cos\theta|_{\text{T7}} = 0.9779$ inside the 7-D Thompson subspace, vs. $|\cos\theta|_{\text{res}} = 0.1430$ in the orthogonal residual subspace.** That is, the developmental line of least resistance that steers floral transitions between hummingbird tubes and bat bells is almost purely a 7-parameter D'Arcy Thompson coordinate transformation.

The one thing we would ask you to judge: is the 7-mode Thompson subspace a reasonable stand-in for the developmental line of least resistance you described, or does it leave out the corolla features that matter? We will record your answer, either way, in our validation log.

With warm regards,

Joel Dietz (Project Delphi · [github.com/fractastical/morpholib](https://github.com/fractastical/morpholib))



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

Your recent papers in *Evolutionary Biology* (*"Reworking geometric morphometrics into a methodology of transformation grids,"* 2023; *"Quadratic trends: A morphometric tool both old and new,"* 2024) made a point that struck us deeply: exact Thin-Plate Spline interpolants fit any pair of landmark configurations with zero residual and therefore cannot falsify D'Arcy Thompson's hypothesis of low-dimensional coordinate transformations (*On Growth and Form*, 1917, Ch. XVII, pp. 748–751, Figs. 373–382).

Inspired by your critique, we formalized all four of Thompson's 1917 fish coordinate families as explicit 5- to 8-parameter maps — **I. Oblique shear** (5p, Figs. 373–374), **II. Complex Möbius coaxial circles** (8p, Figs. 375–376), **III. Bilinear fan of coordinates** (8p, Figs. 377–380), and **IV. Hyperbolic/radial flare** (8p, Figs. 381–382) — and benchmarked them against 4p Procrustes similarity, the 6p uniform (affine) component, 12p quadratic polynomial trend surfaces, and 34p exact TPS across all six of Thompson's 1917 pairs and 4,505 pairs from the 791-species Indo-Pacific reef-fish dataset (Claverie & Wainwright 2014; Friedman et al. 2019).

When we inspected the resulting transformation grids visually, we uncovered a striking methodological discrepancy that directly extends your 2023–2024 papers (summarized in the attached plate **`outreach_fig1_bookstein_silhouette_paradox.png`**):

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

We have drafted a manuscript (attached as `thompson_1917_fish_transformations_manuscript.pdf`, with all Python code openly reproducible) and would be honored to hear your thoughts — or, if this direction interests you, to invite you to join us as a co-author in shaping the paper for *Evolutionary Biology* or *Systematic Biology*.

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

In doing so, we uncovered a striking geometric consequence of teleost modularity that explains why standard 17-landmark morphometrics and continuous whole-body outlines give opposite answers on Thompson's classic pairs (shown in the attached plate **`outreach_fig3_evans_larouche_modularity.png`**):

1. **Why 9/17 Craniofacial Landmarks Bias Global Coordinate Grids (Panels A–C):**
   In the standard 17-landmark teleost scheme, 9 of the 17 points ($53\%$ of the ordinary least-squares weight) lie inside the craniofacial partition ($\text{LM } 1\text{–}2, 11\text{–}17$), while the other 8 mark only the basal insertion points of the fins ($\text{LM } 3\text{–}10$). When craniofacial proportions and median-fin contours decouple across families, fitting Thompson's grid to the 17 landmarks forces the grid to satisfy the 9 craniofacial points at the expense of the posterior trunk and fins:
   - On *Scorpaena scrofa* $\to$ *Antigonia capros* (Figs. 379–380, 8p Fan), the 17-landmark fit inflates the symmetric non-overlapping area $|A \Delta B|$ on the real 600-semilandmark fish outline from **102.2% (unwarped) to 143.7%**, creating a **135.4% blue posterior overhang** (Panel B).
   - When the exact same 8-parameter fan is fitted **directly to the 600-semilandmark whole-body outline** (Panel C), $|A \Delta B|$ is cut **$2.5\times$ to 57.6%** ($\text{IoU}$ jumps from $0.389 \to \mathbf{0.599}$).
   - Across all six of Thompson's 1917 pairs (Panel D), 17-landmark fitting worsens the whole-body silhouette non-overlapping area in three pairs (*Argyropelecus* $\to$ *Sternoptyx*, *Scarus* $\to$ *Pomacanthus*, and *Scorpaena* $\to$ *Antigonia*), whereas direct 600-semilandmark outline fitting substantially reduces $|A \Delta B|$ in every single pair with $100\%$ diffeomorphic bijectivity.

2. **The *Diodon hystrix* $\to$ *Mola mola* Trunk-vs.-Fin Allometric Partition (Panel E):**
   On Thompson's culmination pair (*Diodon* $\to$ *Mola mola*, Figs. 381–382, $79.3\text{ Myr}$ divergence), an 8-parameter exponential radial flare ($x' = u + r_1 v^2,\; y' = v e^{r_2 u}$) explains **79.7% of the 17-landmark trunk variance** compared to just **25.2% for the 6p uniform affine component** ($\Delta\text{AIC}_c = -41.9$, LOO-CV $= 0.0461 < d_0 = 0.0814$). At the same time, *Mola mola*'s full 600-semilandmark silhouette (including its hypertrophied dorsal and anal fins) is **$2.26\times$ taller** than its 17-landmark trunk wireframe ($0.786$ vs. $0.349$), providing a textbook separation between axial truncation (captured by the 8p radial grid) and median-fin hypertrophy.

We have attached the full manuscript (`thompson_1917_fish_transformations_manuscript.pdf`) and would love to hear your thoughts — and if you are interested in collaborating to extend the modular analysis (e.g., comparing separate sub-grids for the craniofacial, axial trunk, and median-fin modules), we would be delighted to welcome you as co-authors.

Warm regards,

Joel Dietz (Project Delphi · [github.com/fractastical/morpholib](https://github.com/fractastical/morpholib))

---

## 5. Email Draft 4 — Prof. Miriam L. Zelditch (University of Michigan)

**To:** `zelditch@umich.edu`  
**Subject:** Peter Wainwright suggested we reach out — Testing D'Arcy Thompson's 1917 fish grids, Relative Warps, and head-body-fin modularity across 791 species  
**Attachments:**
1. `docs/outreach_fig3_evans_larouche_modularity.png` (inline + attached)
2. `docs/outreach_fig4_wainwright_600pt_and_convergence.png` (inline + attached)
3. `docs/thompson_1917_fish_transformations_manuscript.pdf`

---

Dear Professor Zelditch,

We were speaking today with Peter Wainwright about a new quantitative study we have been conducting on D'Arcy Thompson's 1917 fish coordinate transformations (*On Growth and Form*, Ch. XVII, pp. 748–751, Figs. 373–382), and he immediately suggested we share our findings with you.

In Chapter 1 of *Geometric Morphometrics for Biologists* — and in your foundational papers with Olivier Larouche and Richard Cloutier on head, body, and fin modularity in fishes (2017, *BMC Biology*; 2018, *Evolutionary Biology*) — you highlighted both how Thompson's global transformation grids inspired modern geometric morphometrics and how teleost fishes partition variation across semi-autonomous craniofacial, trunk, and fin modules.

We formalized all four of Thompson's 1917 fish coordinate families as explicit 5- to 8-parameter maps (5p oblique shear, 8p coaxial circles, 8p bilinear fan, and 8p exponential radial flare) composed with the 6-parameter uniform affine component, and tested them across all six of Thompson's original 1917 species pairs (both on the standard 17 homologous skeletal landmarks and on 600-semilandmark whole-body outlines) as well as 791 Indo-Pacific reef-fish species (4,505 pairs; 4,334 time-calibrated divergences on the Rabosky et al. 2018 phylogeny).

Three findings emerged that connect directly to your work (summarized in the two attached plates, **`outreach_fig3_evans_larouche_modularity.png`** and **`outreach_fig4_wainwright_600pt_and_convergence.png`**):

1. **How Head–Body–Fin Modularity Creates a 17-Landmark vs. Whole-Body Outline Discrepancy (`outreach_fig3`, Panels A–D):**
   In the standard 17-landmark teleost protocol, **9 of the 17 landmarks (53% of the ordinary least-squares weight) lie in the craniofacial partition** ($\text{LM } 1\text{–}2, 11\text{–}17$), while the remaining 8 mark only the basal insertion points of the fins along the body wall ($\text{LM } 3\text{–}10$) — omitting 100% of the distal dorsal, anal, and caudal fin webs and ventral keel. Because median-fin insertions decouple from the cranium across families (post-warp fin-insertion Procrustes RMS residuals are **$1.16\times \to 1.28\times \to 1.48\times$** larger than craniofacial residuals), fitting a global coordinate grid (or even an exact 34p Thin-Plate Spline) to the 17 landmarks on interfamily pairs such as *Scarus* $\to$ *Pomacanthus* or *Scorpaena* $\to$ *Antigonia* **actively degrades the real 600-semilandmark whole-body outline compared to no non-rigid warp at all** (non-overlapping area $|A \Delta B|$ worsens from $40.6\% \to 56.2\%$ on *Scarus* $\to$ *Pomacanthus* and $102.2\% \to 143.7\%$ on *Scorpaena* $\to$ *Antigonia*). Conversely, fitting the **exact same 5- to 8-parameter Thompson coordinate families directly to the 600-semilandmark outlines** (anchored only at the snout tip and caudal peduncle midpoint) recovers every 1917 transformation ($|A \Delta B| = \mathbf{28.5\%}$ on *Scarus* $\to$ *Pomacanthus*, $\mathbf{17.2\%}$ on *Polyprion* $\to$ *Pristigenys*, $\mathbf{35.5\%}$ on *Pristigenys* $\to$ *Scorpaena*, and $\mathbf{57.6\%}$ on *Scorpaena* $\to$ *Antigonia*), with $100\%$ diffeomorphic bijectivity ($\det D\Phi > 0$ everywhere).

2. **Why 8p Thompson Outperforms Standard 600-Semilandmark Curve TPS on Out-of-Sample Interior Anatomy (`outreach_fig4`, Panels A–C):**
   When we hold the 600-semilandmark outline input fixed across all six 1917 pairs, a standard 600-semilandmark curve TPS (Bookstein 1997) slides semilandmarks along hypertrophied dorsal and anal fin lobes, **folding and inverting the 2D interior coordinate grid on 6 of 6 pairs** ($\min \det D\Phi = -6.76$) and degrading out-of-sample prediction of the 14 held-out interior skeletal landmarks by up to $2.4\times$. In contrast, Thompson's 8-parameter coordinate grid remains $100\%$ diffeomorphic (`6/6` pairs) and accurately predicts held-out internal craniofacial and fin-base landmarks from the outer contour alone.

3. **92% Capture of Empirical Non-Affine Relative Warps & Convergent Phylomorphospace Trajectories (`outreach_fig4`, Panels D–F):**
   Using Leave-One-Family-Out cross-validation across all 48 families (791 species), Thompson's 1917 analytical 8-parameter laws capture **92.0% (median) / 95.6% (mean)** of the variance explained by the optimal data-driven 8-parameter Relative-Warp basis (`6p Uniform + RW1 + RW2`, Rohlf 1993) (`50.9%` vs. `55.3%`, compared to `25.1%` for the 6p uniform component) and **88.6%** of the primary `(PC1, PC2)` phylomorphospace plane. Furthermore, Phenotypic Trajectory Analysis (Adams & Collyer 2009) across 5 convergent deep-bodied families and 6 elongate/fusiform families (`> 95 Myr` apart) shows that convergent clades follow strongly parallel global 8p Thompson trajectories ($r_{\text{vec}} = +0.693$ and $+0.714$, vs. $-0.699$ between divergent clades) while diverging in their local craniofacial and fin-insertion modules ($r_{\text{vec}} = +0.416$).

We have attached our manuscript (`thompson_1917_fish_transformations_manuscript.pdf`, with all code and data openly reproducible). We would be deeply honored to hear your thoughts — and if this interests you, to explore collaborating as a co-author on a submission to *Systematic Biology*, *Evolution*, or *Evolutionary Biology*.

With warm regards,

Joel Dietz (Project Delphi · [github.com/fractastical/morpholib](https://github.com/fractastical/morpholib))

---

## 6. Email Draft 5 — Prof. Anjali Goswami (Natural History Museum, London / UCL) & Dr. Ryan N. Felice (University College London)

**To:** `a.goswami@nhm.ac.uk`, `r.felice@ucl.ac.uk`  
**Subject:** Peter Wainwright suggested we reach out — High-density 600-semilandmark outlines vs. 17 sparse landmarks & developmental modularity across D'Arcy Thompson's 1917 fish transformations  
**Attachments:**
1. `docs/outreach_fig4_wainwright_600pt_and_convergence.png` (inline + attached)
2. `docs/outreach_fig2_price_wainwright_horizon.png` (inline + attached)
3. `docs/thompson_1917_fish_transformations_manuscript.pdf`

---

Dear Professor Goswami and Dr. Felice,

We were speaking today with Peter Wainwright about a study testing D'Arcy Thompson's 1917 fish transformations (*On Growth and Form*, Figs. 373–382) across 791 Indo-Pacific reef-fish species and 600-semilandmark outlines, and he suggested we reach out to both of you given how closely our findings intersect your work on high-density semilandmark morphometrics, curve-sliding artifacts, and the developmental origins of macroevolutionary modularity (Goswami et al. 2014, 2022; Felice & Goswami 2018, *PNAS*; Felice et al. 2018, *Evolution* / *Systematic Biology*).

We formalized all four of Thompson's 1917 fish coordinate families as explicit 5- to 8-parameter maps (5p oblique shear, 8p coaxial circles, 8p bilinear fan, and 8p exponential radial flare) and evaluated them across all six of Thompson's 1917 species pairs (both on standard 17 homologous skeletal landmarks and on 600-semilandmark whole-body outlines) as well as 4,334 time-calibrated divergences on the Rabosky et al. (2018) megaphylogeny.

Three results emerged that directly parallel and extend your findings (summarized in the attached plates **`outreach_fig4_wainwright_600pt_and_convergence.png`** and **`outreach_fig2_price_wainwright_horizon.png`**):

1. **Sparse Landmark Density Bias vs. 600-Semilandmark Outlines & Curve-TPS Folding Artifacts (`outreach_fig4`, Panels A–C):**
   In the standard 17-landmark teleost protocol, **9 of the 17 landmarks (53% of ordinary least-squares weight) sit in the craniofacial partition** while $0\%$ sample the distal dorsal, anal, and caudal fin contours. Consequently, fitting a global coordinate grid to the 17 sparse landmarks on interfamily pairs such as *Scarus* $\to$ *Pomacanthus* or *Scorpaena* $\to$ *Antigonia* **actively degrades the real 600-semilandmark whole-body silhouette compared to no non-rigid deformation at all** (non-overlapping area $|A \Delta B|$ worsens from $40.6\% \to 56.2\%$ and $102.2\% \to 143.7\%$).
   Yet when we hold the **600-semilandmark outline input fixed** and compare five competing 600-point outline methods head-to-head (Panel A–C), standard 600-semilandmark curve TPS (Bookstein 1997) suffers from non-homologous perimeter sliding across hypertrophied median-fin lobes (echoing your critique in Felice et al. 2018), **folding and inverting the 2D interior coordinate grid on 6 of 6 pairs** ($\min \det D\Phi = -6.76$, with up to $25.1\%$ of the body domain inverted in red in Panel A) and degrading out-of-sample prediction of the 14 held-out interior skeletal landmarks by up to **$2.4\times$**. By contrast, fitting Thompson's 8-parameter coordinate grid directly to the 600 semilandmarks (anchored only at the snout tip and caudal peduncle midpoint) cuts $|A \Delta B|$ to **$28.5\%$** (*Scarus* $\to$ *Pomacanthus*), **$17.2\%$** (*Polyprion* $\to$ *Pristigenys*), and **$35.5\%$** (*Pristigenys* $\to$ *Scorpaena*), maintains **$100\%$ diffeomorphic bijectivity ($\det D\Phi > 0$ on `6/6` pairs)**, and achieves the lowest out-of-sample error on held-out internal anatomy.

2. **A $\sim 45\text{–}50\text{ Myr}$ Macroevolutionary Disparity Transition Driven by Neural-Crest Craniofacial vs. Somitic Fin-Insertion Modularity (`outreach_fig2`, Panels A–B):**
   Across 4,334 time-calibrated pairs (`791` species), 8-parameter Thompson grids pull median 17-landmark Procrustes RMS residuals right down inside the intrageneric Procrustes disparity band ($\sigma_{\text{genus}} \le 0.0110$) for divergences under $50\text{ Myr}$ (`0.0106`, `0.0104`, `0.0106` RMS; $53\%\text{–}61\%$ of pairs within $\sigma_{\text{genus}}$). Past **$\sim 45\text{–}50\text{ Myr}$** ($t^* = 45.8\text{ Myr}$ on $n = 187$ phylogenetically independent node-disjoint pairs), post-warp Procrustes RMS jumps $2.5\times\text{–}3.3\times$ (`0.0154 → 0.0347`) because the somitic median/paired fin insertions ($\text{LM } 3\text{–}10$) decouple from the neural-crest craniofacial module ($\text{LM } 1\text{–}2, 11\text{–}17$; $\text{Fin/Head RMS} = 1.16\times \to 1.28\times \to 1.48\times$), providing a fish counterpart to your developmental modularity findings in tetrapods (Felice & Goswami 2018).

3. **92% Capture of Empirical Relative Warps & Convergent Phylomorphospace Trajectories (`outreach_fig4`, Panels D–F):**
   Under Leave-One-Family-Out cross-validation across all 48 families, Thompson's 1917 analytical 8-parameter equations capture **92.0% (median) / 95.6% (mean)** of the theoretical 8-parameter empirical Relative-Warp ceiling (`6p Uniform + RW1 + RW2`) and **88.6%** of the primary `(PC1, PC2)` phylomorphospace plane. Under Phenotypic Trajectory Analysis (Adams & Collyer 2009), 5 distantly related deep-bodied families and 6 elongate/fusiform families (`> 95 Myr` apart) evolve along strongly parallel global 8p Thompson trajectories ($r_{\text{vec}} = +0.693$ and $+0.714$, vs. $-0.699$ between divergent clades) while diverging in their local craniofacial and fin-insertion modules ($r_{\text{vec}} = +0.416$).

We have attached our manuscript (`thompson_1917_fish_transformations_manuscript.pdf`) and would love to hear your thoughts — and if this direction excites you, we would be thrilled to explore collaborating together as co-authors.

With best regards,

Joel Dietz (Project Delphi · [github.com/fractastical/morpholib](https://github.com/fractastical/morpholib))


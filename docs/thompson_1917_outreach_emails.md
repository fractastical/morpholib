# Outreach Email Package & Manuscript Summary: D'Arcy Thompson's 1917 Fish Transformations

## 1. Compiled Manuscript & High-Resolution Visual Assets

- **Compiled LaTeX Manuscript PDF (12 pages):** `docs/thompson_1917_fish_transformations_manuscript.pdf`
- **LaTeX Source:** `docs/thompson_1917_fish_transformations_manuscript.tex`
- **Full-Page Manuscript Figures:**
  1. `thompson_stick_vs_silhouette_reality.png` — 4x5 plate contrasting the 8-vertex 17-landmark trunk polygon against the 600-point continuous fish silhouette across Thompson's 1917 pairs.
  2. `thompson_silhouette_area_difference.png` — 3x5 plate quantifying the exact Area of Difference ($|\Omega_A \Delta \Omega_B| = |\Omega_B \setminus \Omega_A|\text{ [Red Missing]} + |\Omega_A \setminus \Omega_B|\text{ [Blue Overhang]}$) across Unwarped (Row 1), 17-Landmark Fit (Row 2), and Direct 600-Point Silhouette Fit (Row 3).
  3. `thompson_four_families_atlas.png` — 3x4 macroevolutionary atlas across 791 species, 4,505 pairs, and 4,334 time-calibrated divergences on the Rabosky et al. (2018) megaphylogeny.
  4. `thompson_rigor_ablation_and_jacobian.png` — 2x2 methodological stress-test plate: (A) causal landmark ablation (9-Head-LM vs. 17-LM vs. 10-Outer-Trunk-LM vs. Direct 600-pt Silhouette across all 6 pairs), (B) Jacobian diffeomorphic bijectivity ($\det D\Phi(x,y) > 0$), (C) bidirectional invertibility ($S \leftrightarrow T$), and (D) unbinned logistic horizon & 187 node-disjoint independent pairs.
- **Tailored Email Summary Plates (1-page visual attachments):**
  1. `docs/outreach_fig1_bookstein_silhouette_paradox.png` (for Prof. Fred L. Bookstein)
  2. `docs/outreach_fig2_price_wainwright_horizon.png` (for Prof. Samantha A. Price, Dr. Sarah T. Friedman, and Prof. Peter C. Wainwright)
  3. `docs/outreach_fig3_evans_larouche_modularity.png` (for Prof. Kory M. Evans and Dr. Olivier Larouche)

---

## 2. Email Draft 1 — Prof. Fred L. Bookstein (University of Vienna / University of Michigan)

**To:** `fred.bookstein@univie.ac.at` (or `fred@cns.nyu.edu` / `fbookst@umich.edu`)  
**Subject:** Empirical test of your 2023/2024 *Evolutionary Biology* critiques across Thompson's 1917 fish pairs & 791 species (with a surprising 17-landmark vs. silhouette paradox)  
**Attachments:**
1. `docs/outreach_fig1_bookstein_silhouette_paradox.png` (inline + attached)
2. `docs/thompson_1917_fish_transformations_manuscript.pdf`

---

Dear Professor Bookstein,

Your recent papers in *Evolutionary Biology* (*"Reworking geometric morphometrics into a methodology of transformation grids,"* 2023; *"Quadratic trends: A morphometric tool both old and new,"* 2024) made a point that struck us deeply: exact Thin-Plate Spline interpolants fit any pair of landmark configurations with zero residual and therefore cannot falsify D'Arcy Thompson's hypothesis of low-dimensional coordinate transformations (*On Growth and Form*, 1917, Ch. XVII, pp. 748–751, Figs. 373–382).

Inspired by your critique, we formalized all four of Thompson's 1917 fish coordinate families as explicit 5- to 8-parameter maps — **I. Oblique shear** (5p, Figs. 373–374), **II. Complex Möbius coaxial circles** (8p, Figs. 375–376), **III. Bilinear fan of coordinates** (8p, Figs. 377–380), and **IV. Hyperbolic/radial flare** (8p, Figs. 381–382) — and benchmarked them against 4p similarity, 6p affine, 12p quadratic polynomial trend surfaces, and 34p exact TPS across all six of Thompson's 1917 pairs and 4,505 pairs from the 791-species Indo-Pacific reef-fish dataset (Claverie & Wainwright 2014; Friedman et al. 2019).

When we inspected the resulting transformation grids Visually, we uncovered a striking methodological paradox that directly extends your 2023–2024 papers (summarized in the attached plate **`outreach_fig1_bookstein_silhouette_paradox.png`**):

1. **Why Fitting to the Standard 17 Ichthyological Landmarks Actually Worsens Real Fish Silhouettes (Panels A–B):**
   In the standard 17-landmark reef-fish protocol, **9 of the 17 landmarks (53% of the least-squares weight) sit inside the head and opercle**, while the remaining 8 only mark where the fin bases insert into the body wall — amputating 100% of the dorsal fin, anal fin, caudal tail, and ventral keel. Consequently, when we fit Thompson's 8-parameter coaxial grid (or even an exact 34-parameter TPS!) to the 17 landmarks of *Scarus ghobban* $\to$ *Pomacanthus semicirculatus* (Figs. 375–376) and carry the real 600-point fish silhouette along with the grid, the 9 head landmarks pull the grid the wrong way for the posterior body: the symmetric Area of Difference ($|A \Delta B| = |B \setminus A| + |A \setminus B|$) **worsens from 40.6% (unwarped) to 56.2% of target fish area** (creating a **47.2% blue posterior overhang**, Panel B), and silhouette $\text{IoU}$ drops from $0.662 \to 0.618$ ($0.629$ under exact 34p TPS). The same degradation occurs on *Scorpaena* $\to$ *Antigonia* ($|A \Delta B|: 102.1\% \to 143.7\%$) and *Argyropelecus* $\to$ *Sternoptyx* ($58.4\% \to 63.0\%$).

2. **Direct 600-Point Continuous Silhouette Fitting (Anchored on Your Two-Point Baseline) Recovers Thompson's 1917 Transformations (Panels C–E):**
   Thompson never fitted 17 head-heavy points — he drew continuous outlines of the entire fish including its fins. When we fit the **exact same 5- to 8-parameter Thompson coordinate families directly to the 600-point continuous fish silhouettes** (anchored only at your two axial baseline endpoints, snout tip $\text{LM } 2$ and caudal peduncle base $\tfrac{1}{2}(\text{LM } 5 + \text{LM } 6)$, to prevent tangential perimeter sliding, with $100\%$ positive Jacobian determinant $\det D\Phi > 0$ everywhere), Thompson's transformations are dramatically recovered across every 1917 pair:
   - *Scarus* $\to$ *Pomacanthus* (8p Coaxial, Panel C): Area of Difference $|A \Delta B|$ is **cut in half from 56.2% to 28.5%** ($\text{IoU}: 0.618 \to \mathbf{0.749}$; reverse *Pomacanthus* $\to$ *Scarus* $= \mathbf{26.0\%}$, $\text{IoU} = \mathbf{0.765}$).
   - *Scorpaena* $\to$ *Antigonia* (8p Fan, Panel D): $|A \Delta B|$ drops **$2.5\times$ from 143.7% to 57.6%** ($\text{IoU}: 0.389 \to \mathbf{0.599}$).
   - *Pristigenys* $\to$ *Scorpaena* (8p Fan, Panel E): $|A \Delta B|$ drops **$3.4\times$ from 120.1% to 35.5%** ($\text{IoU} = \mathbf{0.694}$).
   - *Polyprion* $\to$ *Pristigenys* (8p Fan): $|A \Delta B|$ drops to **17.2%** ($\text{IoU} = \mathbf{0.839}$).

3. **8-Parameter Thompson Coordinate Laws vs. 12-Parameter Quadratic Trends Across 4,505 Pairs (Panel F):**
   On the 17-landmark skeleton ($n = 34$ scalar coordinates), your 2024 quadratic trend surface (12p) and Thompson's 8-parameter families can be compared directly via small-sample $\text{AIC}_c$ and Leave-One-Landmark-Out cross-validation (LOO-CV). Across 4,505 pairs, 8-parameter Thompson grids sit at the exact parsimony sweet spot: they beat 6p affine in **82.2%** of intrafamily pairs and beat 12p quadratic trends by $\text{AIC}_c$ in **84.9%** of intrafamily and **92.1%** of interfamily pairs. And on *Diodon hystrix* $\to$ *Mola mola* (Figs. 381–382), Thompson's 8-parameter exponential radial flare ($y' = v e^{r_2 u}$) jumps from **25.2% (affine) to 79.7% variance explained** ($\Delta\text{AIC}_c = -41.9$, passing LOO-CV with $0.0461 < d_0 = 0.0814$).

We have drafted a short manuscript (attached as `thompson_1917_fish_transformations_manuscript.pdf`, with all Python code openly reproducible) and would be honored to hear your thoughts — or, if this direction interests you, to invite you to join us as a co-author in shaping the paper for *Evolutionary Biology* or *Systematic Biology*.

With warm regards,

[Your Name]

---

## 3. Email Draft 2 — Prof. Samantha A. Price, Dr. Sarah T. Friedman & Prof. Peter C. Wainwright (Clemson / UC Davis)

**To:** `sprice6@clemson.edu`, `pcwainwright@ucdavis.edu`, `sarah.friedman@noaa.gov` (or current address)  
**Subject:** A ~45–50 Myr phylogenetic horizon for D'Arcy Thompson's coordinate grids across your 791 reef-fish dataset + Satterfield et al. (2024) fin-body decoupling  
**Attachments:**
1. `docs/outreach_fig2_price_wainwright_horizon.png` (inline + attached)
2. `docs/thompson_1917_fish_transformations_manuscript.pdf`

---

Dear Professors Price and Wainwright and Dr. Friedman,

Your 791-species Indo-Pacific shore-fish landmark dataset (Claverie & Wainwright 2014; Friedman et al. 2019) and your recent work on fin-body modularity (Satterfield, Claverie & Wainwright 2024, *Evolution*) made it possible to answer a century-old question in evolutionary morphology: **Over what phylogenetic timescale do D'Arcy Thompson's 1917 global coordinate transformations (*On Growth and Form*, Figs. 373–382) actually govern teleost body shape evolution, and why do they break down across deep divergences?**

We formalized all four of Thompson's 1917 coordinate families (5p oblique shear, 8p coaxial circles, 8p bilinear fan, and 8p hyperbolic/radial expansion) and fitted them across **4,505 taxonomic pairs** in your 791-species dataset, **4,334 time-calibrated divergences** on the Rabosky et al. (2018) tree, and **all six exact species pairs Thompson drew in 1917** (both on the 17 landmarks and on 600-point continuous fish silhouettes).

The attached visual summary (**`outreach_fig2_price_wainwright_horizon.png`**) highlights three findings that directly connect your 2014, 2019, and 2024 papers:

1. **A Sharp $\sim 45\text{–}50\text{ Myr}$ Phylogenetic Horizon Across 4,334 Dated Pairs (Panel A):**
   When we measure how often an 8-parameter Thompson coordinate grid reduces the Procrustes RMS residual all the way to the target genus's own interspecific biological noise floor, Thompson's grids succeed in **53.0%** of pairs diverged $0\text{–}10\text{ Myr}$ ($n=149$), **61.3%** at $10\text{–}25\text{ Myr}$ ($n=359$), and **49.7%** at $25\text{–}50\text{ Myr}$ ($n=346$ — nearly double the 6p affine rate of $27.5\%$). Then, right at **$\sim 45\text{–}50\text{ Myr}$**, global coordinate transformations hit a sharp phylogenetic cliff, dropping to **10.6%** at $50\text{–}80\text{ Myr}$ ($n=170$), **1.0%** at $80\text{–}110\text{ Myr}$ ($n=618$), and **0.0%** above $110\text{ Myr}$ ($n=2,692$).

2. **Direct Confirmation of Satterfield, Claverie & Wainwright (2024) Fin–Body Decoupling (Panel B):**
   Partitioning the post-transformation residuals by anatomical module reveals *why* global grids break above $50\text{ Myr}$: median and paired fin insertion landmarks ($\text{LM } 3\text{–}10$) exhibit **$1.16\times$ (intrageneric) $\to 1.28\times$ (intrafamily) $\to 1.48\times$ to $2.03\times$ (interfamily)** larger residuals than craniofacial landmarks ($\text{LM } 1\text{–}2, 11\text{–}17$). Above $\sim 45\text{–}50\text{ Myr}$, dorsal, anal, and pectoral fin insertions migrate across the body wall independently of the cranium.

3. **Resolving the 17-Landmark vs. Whole-Fish Silhouette Paradox (Panels C–E):**
   Because 9 of the 17 landmarks ($53\%$) sit in the head and opercle while $0\%$ sit on the outer fin margins, fitting a coordinate grid to the 17 landmarks on interfamily pairs such as *Scarus ghobban* $\to$ *Pomacanthus semicirculatus* (Figs. 375–376, $112.8\text{ Myr}$) actually **worsens the real 600-point fish silhouette** compared to not warping at all (the Area of Difference $|A \Delta B|$ increases from **40.6% to 56.2%** due to a **47.2% blue posterior overhang**, Panel C). Yet when we fit the exact same 8-parameter coaxial grid **directly to the 600-point continuous fish silhouettes** (Panel D), the Area of Difference is slashed in half to **28.5%** ($\text{IoU} = \mathbf{0.749}$), and on *Polyprion* $\to$ *Pristigenys* (Panel E) it drops to **17.2%** ($\text{IoU} = \mathbf{0.839}$).

We have attached our draft manuscript (`thompson_1917_fish_transformations_manuscript.pdf`). Because this work builds directly on your reef-fish morphospace and fin-body modularity discoveries, we would love to hear your feedback and would be thrilled to explore collaborating as co-authors on a joint submission to *Evolution*, *Proceedings of the Royal Society B*, or *Systematic Biology*.

Best regards,

[Your Name]

---

## 4. Email Draft 3 — Prof. Kory M. Evans (Rice University) & Dr. Olivier Larouche

**To:** `kory.evans@rice.edu`, `olivier.larouche@univ-lille.fr` (or current institutional address)  
**Subject:** How teleost head-trunk-fin modularity explains the 17-landmark vs. silhouette paradox in D'Arcy Thompson's 1917 fish grids  
**Attachments:**
1. `docs/outreach_fig3_evans_larouche_modularity.png` (inline + attached)
2. `docs/thompson_1917_fish_transformations_manuscript.pdf`

---

Dear Professor Evans and Dr. Larouche,

Your work on head-trunk-fin modularity and evolutionary allometry across teleost fishes inspired us to look closely at what happens to individual anatomical modules when D'Arcy Thompson's famous 1917 coordinate transformations (*On Growth and Form*, Figs. 373–382) are tested across 791 Indo-Pacific reef fishes (4,505 pairs) and all six of Thompson's original 1917 species pairs.

In doing so, we discovered a striking geometric consequence of teleost modularity that explains why standard 17-landmark morphometrics and continuous whole-fish silhouettes give opposite answers on Thompson's classic pairs (shown in the attached plate **`outreach_fig3_evans_larouche_modularity.png`**):

1. **Why 9/17 Craniofacial Landmarks Hijack Global Coordinate Grids (Panels A–C):**
   In the standard 17-landmark teleost scheme, 9 of the 17 points ($53\%$ of the ordinary least-squares weight) lie inside the head and opercle ($\text{LM } 1\text{–}2, 11\text{–}17$), while the other 8 mark only the basal insertion points of the fins ($\text{LM } 3\text{–}10$). When head proportions and median fin contours decouple across families, fitting Thompson's grid to the 17 landmarks forces the grid to satisfy the 9 head points at the expense of the posterior trunk and fins:
   - On *Scorpaena scrofa* $\to$ *Antigonia capros* (Figs. 379–380, 8p Fan), the 17-landmark fit blows up the symmetric Area of Difference $|A \Delta B|$ on the real 600-point fish silhouette from **102.1% (unwarped) to 143.7%**, creating a massive **135.4% blue posterior overhang** (Panel B).
   - When the exact same 8-parameter fan is fitted **directly to the 600-point whole-body silhouette** (Panel C), $|A \Delta B|$ is cut **$2.5\times$ to 57.6%** ($\text{IoU}$ jumps from $0.389 \to \mathbf{0.599}$).
   - Across all six of Thompson's 1917 pairs (Panel D), 17-landmark fitting worsens the whole-body silhouette Area of Difference in three pairs (*Argyropelecus* $\to$ *Sternoptyx*, *Scarus* $\to$ *Pomacanthus*, and *Scorpaena* $\to$ *Antigonia*), whereas direct 600-point silhouette fitting substantially reduces $|A \Delta B|$ in every single pair.

2. **The *Diodon hystrix* $\to$ *Mola mola* Trunk-vs.-Fin Allometric Triumph (Panel E):**
   On Thompson's culmination pair (*Diodon* $\to$ *Mola mola*, Figs. 381–382, $79.3\text{ Myr}$ divergence), an 8-parameter exponential radial flare ($x' = u + r_1 v^2,\; y' = v e^{r_2 u}$) explains **79.7% of the 17-landmark trunk variance** compared to just **25.2% for affine** ($\Delta\text{AIC}_c = -41.9$, LOO-CV $= 0.0461 < d_0 = 0.0814$). At the same time, *Mola mola*'s full 600-point silhouette (including its towering dorsal and anal fins) is **$2.26\times$ taller** than its 17-landmark trunk polygon ($0.786$ vs. $0.349$), providing a textbook separation between axial truncation (captured by the 8p radial grid) and appendicular fin hypertrophy.

We have attached the full manuscript (`thompson_1917_fish_transformations_manuscript.pdf`) and would love to hear your thoughts — and if you are interested in collaborating to extend the modular analysis (e.g., comparing separate sub-grids for the craniofacial, trunk, and fin modules), we would be delighted to welcome you as co-authors.

Warm regards,

[Your Name]

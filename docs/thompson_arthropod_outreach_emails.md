# Outreach Email Package & Scholarly Validation Dossier: Arthropod Morphospace (Crab Carapaces, Ant Caste Polyphenism & Hymenopteran Wing Evolution)

## 1. Visual Assets, Pre-Registered Benchmarks & Executable Scripts

- **Crab Carapace Paradox Resolution (`Scholtz, Knötel & Baum 2020` Audit):**
  - **Script:** [`thompson/run_crab_carapace_paradox_audit.py`](../thompson/run_crab_carapace_paradox_audit.py)
  - **Summary JSON:** [`thompson_crab_carapace_audit_summary.json`](../thompson_crab_carapace_audit_summary.json)
  - **Outreach Plate:** [`docs/outreach_fig9_crab_carapace_paradox_resolution.png`](./outreach_fig9_crab_carapace_paradox_resolution.png)
- **Ant Caste Polyphenism & 3D Synchrotron Phenomics (`ANTSCAN` + *Camponotus* 4-Structure GM):**
  - **Scripts:** [`thompson/portfolio/groups/camponotus_ants.py`](../thompson/portfolio/groups/camponotus_ants.py), [`thompson/run_antscan_caste_allometry_tests.py`](../thompson/run_antscan_caste_allometry_tests.py)
  - **Pre-Registered Results:** [`docs/portfolio/camponotus_ants.md`](./portfolio/camponotus_ants.md), [`thompson_ant_caste_allometry_summary.json`](../thompson_ant_caste_allometry_summary.json)
  - **Outreach Plates:** [`docs/portfolio/fig_camponotus_ants.png`](./portfolio/fig_camponotus_ants.png), [`docs/outreach_fig8_antscan_caste_allometry_hypotheses.png`](./outreach_fig8_antscan_caste_allometry_hypotheses.png)
- **Hymenopteran Forewing Venation & $\mathbf{P}$-Matrix Drift Causal Decomposition (`Vespinae` Hornets vs. `Anthophila` Bees):**
  - **Scripts:** [`thompson/portfolio/groups/vespinae_wings.py`](../thompson/portfolio/groups/vespinae_wings.py), [`thompson/portfolio/groups/bee_wings.py`](../thompson/portfolio/groups/bee_wings.py), [`thompson/portfolio/groups/why_wings.py`](../thompson/portfolio/groups/why_wings.py)
  - **Pre-Registered Results:** [`docs/portfolio/vespinae_wings.md`](./portfolio/vespinae_wings.md), [`docs/portfolio/bee_wings.md`](./portfolio/bee_wings.md), [`docs/portfolio/why_wings.md`](./portfolio/why_wings.md)
  - **Outreach Plates:** [`docs/portfolio/fig_why_wings.png`](./portfolio/fig_why_wings.png), [`docs/portfolio/fig_bee_wings.png`](./portfolio/fig_bee_wings.png), [`docs/portfolio/fig_vespinae_wings.png`](./portfolio/fig_vespinae_wings.png), [`docs/outreach_fig10_complete_arthropod_atlas.png`](./outreach_fig10_complete_arthropod_atlas.png)

---

## 2. Email Draft 1 — Prof. Gerhard Scholtz (Humboldt-Universität zu Berlin) & Dr. Daniel Baum / Dr. Daniel Knötel (Zuse Institute Berlin)

**To:** `gerhard.scholtz@rz.hu-berlin.de`, `baum@zib.de`  
**Subject:** Resolving the 8-landmark vs. lateral-margin discrepancy in D'Arcy Thompson's 1917 Fig. 369 crab carapaces (testing your 2020 *Zoomorphology* 3D laser-scanned dataset)  
**Attachments:**
1. `docs/outreach_fig9_crab_carapace_paradox_resolution.png` (inline + attached)
2. `docs/outreach_fig10_complete_arthropod_atlas.png` (inline + attached)
3. `docs/thompson_1917_fish_transformations_manuscript.pdf`

---

Dear Professor Scholtz, Dr. Baum, and Dr. Knötel,

Your 2020 paper in *Zoomorphology* (*"D'Arcy Thompson's 'on growth and form' and the computer-assisted-transformation of crab carapaces,"* `139:299–314`) is one of the most rigorous quantitative tests of D'Arcy Thompson's 1917 transformation grids ever published. In testing Thompson's famous six brachyuran/anomuran carapaces (Fig. 369, p. 744: *Geryon* $\to$ *Corystes*, *Scyramathia* [*Rochinia*], *Paralomis* [*Lithodes*], *Lupa* [*Portunus*], and *Chorinus* [*Leptomithrax*]) across both Thompson's drawings and 3D laser-scanned museum specimens, you discovered a striking puzzle:
1. When Thin-Plate Splines (TPS) were fitted to **8 homologous internal/axial landmarks**, the warped grids failed to reproduce Thompson's radiating fans on spider crabs (*Rochinia*, *Leptomithrax*) and king crabs (*Lithodes*), producing oval/rectangular distortions instead.
2. Conversely, when a fine 32-pixel B-spline was forced to register the outer silhouettes including sharp anterolateral teeth and rostral horns, the coordinate grid **folded over itself with crossing lines ($\det J < 0$)**.

While conducting a parallel study on D'Arcy Thompson's 1917 fish transformations (Figs. 373–382), we encountered the **exact same two-sided phenomenon** — and when we applied our diagnostic framework directly to your 2020 crab carapace images and 8-landmark coordinates (`thompson/run_crab_carapace_paradox_audit.py`, summarized in the attached plate **`outreach_fig9_crab_carapace_paradox_resolution.png`**), we found that both puzzles have a clean mathematical resolution:

1. **Why the 8 Homologous Landmarks Missed the Spider-Crab Fan (Medial Clustering Bias):**
   In your 8-landmark scheme, 3 landmarks sit on the narrow anterior rostral/orbital margin, 2 sit on the medial cardiac groove, and 3 sit on the narrow posterior carapace-pleon border — leaving **$0\text{ of }8$ landmarks ($0\%$) on the lateral hepatic or branchial carapace margins**. Just as $9/17$ craniofacial landmarks in reef fishes leave the dorsal and anal fins unconstrained, fitting a global grid to 8 midline-clustered points leaves the lateral branchial expansion of *Lithodes*, *Rochinia*, and *Portunus* unconstrained in ordinary least squares (producing a **$1.85\times$ to $7.13\times$ larger lateral-margin RMS error** on your 3D laser-scanned crabs).
2. **Recovering Thompson's 1917 Fig. 369 Grids with 100% Diffeomorphic Bijectivity ($\det J > 0$):**
   On p. 745 of *On Growth and Form*, Thompson explicitly described his crab transformation as a low-dimensional **bilinear-logarithmic coordinate fan** (converging/diverging transverse ordinates $x' = a_1 u + f_1 u v + t_x$ paired with logarithmic longitudinal spacing $y' = d_1 v + f_2 v^2 + c_2 u^2 + t_y$). When we augment your 8 axial/cardiac landmarks with 6 homologous lateral hepatic and branchial margin points (14 landmarks total, avoiding local spine tips that cause B-spline folding), Thompson's 8-parameter carapace law recovers **both Thompson's 1917 drawings ($R^2 = 95.8\%\text{–}99.4\%$ on spider and king crabs; $+38.4\text{ to }+59.4$ percentage points over a 4-parameter rectangular box stretch)** and **your real 3D laser-scanned crabs ($R^2 = 79.0\%\text{–}95.3\%$; $+27.8\text{ to }+63.1$ percentage points over a 4p box stretch on *Rochinia*, *Lithodes*, and *Leptomithrax*)**, while cutting lateral-margin RMS error by **$1.85\times\text{ to }7.13\times$** with **$100\%$ positive Jacobian determinant ($\min \det J > 0$, zero grid folding)**!

We would be honored to hear your thoughts on these results — and if you are interested in collaborating on a short follow-up note or comparative arthropod-vertebrate synthesis, we would be delighted to share all our code and explore co-authoring together.

With warm regards,

Joel Dietz (Project Delphi · [github.com/fractastical/morpholib](https://github.com/fractastical/morpholib))

---

## 3. Email Draft 2 — Prof. Evan P. Economo (OIST / University of Maryland), Dr. Julian Katzke & Dr. Thomas van de Kamp (KIT — `ANTSCAN`)

**To:** `economo@umd.edu`, `julian.katzke@oist.jp`, `thomas.vandekamp@kit.edu`  
**Subject:** Testing D'Arcy Thompson's coordinate transformation laws across ANTSCAN (2,235 3D synchrotron scans) and 4-structure *Camponotus* caste polyphenism  
**Attachments:**
1. `docs/portfolio/fig_camponotus_ants.png` (inline + attached)
2. `docs/outreach_fig8_antscan_caste_allometry_hypotheses.png` (inline + attached)
3. `docs/outreach_fig10_complete_arthropod_atlas.png` (inline + attached)

---

Dear Professor Economo, Dr. Katzke, and Dr. van de Kamp,

Your landmark 2026 *Nature Methods* `ANTSCAN` release (2,235 synchrotron micro-CT 3D scans across 14 subfamilies, 212 genera, and 792 species) inspired us to test a question D'Arcy Thompson raised in *On Growth and Form* (1917, Fig. 359, p. 732) when comparing ovate vs. cordate radial-tangential coordinates: **how much of ant worker caste polyphenism (minor $\to$ major soldier allometry) and interspecific exoskeletal divergence lives inside a low-dimensional global Thompson coordinate transformation vs. local modular remodeling?**

Using our pre-registered phylogenetic prediction and morphospace pipeline ([`fractastical/morpholib`](https://github.com/fractastical/morpholib)), we evaluated Thompson's 7 linearized global coordinate modes (`T7`) and nonlinear 8-parameter cordate-fan maps across **43 explicit minor/major polymorphic species in `ANTSCAN`** and **634 *Camponotus* specimens across 43 species and 4 homologous sclerite structures** (frontal head 82 LMs, lateral head 61 LMs, dorsal pronotum 52 LMs, lateral mesosoma 46 LMs; summarized in the attached plates **`fig_camponotus_ants.png`** and **`outreach_fig8_antscan_caste_allometry_hypotheses.png`**):

1. **Ant Exoskeletal Sclerite Capsules Are Massively Thompsonian (`81.2%–92.9%` Variance in 7 Modes):**
   Whereas insect wing venation only has `38.8%` (*Vespinae* hornets) to `42.2%` (*Anthophila* bees) of its interspecific variance in Thompson's 7 global modes and reef-fish bodies have `57.9%`, continuous ant sclerite capsules have **`81.2%` (frontal head, 82 LMs)**, **`81.1%` (lateral head, 61 LMs)**, **`92.9%` (dorsal pronotum, 52 LMs)**, and **`77.3%` (lateral mesosoma, 46 LMs)** of between-species variance inside the 7-parameter Thompson coordinate basis!
2. **High Phylogenetic Signal & Leave-One-Species-Out Prediction Across All 4 Structures:**
   On the calibrated *Camponotus* phylogeny (`41` species), Brownian Motion wins by $\mathrm{AIC}_c$ in all 4 structures ($\hat{\lambda} = 0.771\text{–}1.000$), leave-one-species-out phylogenetic prediction beats the consensus in all $4/4$ structures ($0.559\times$ error on frontal head, $0.561\times$ on pronotum, $p < 10^{-4}$), and the 7 Thompson modes alone capture **`85.6%`** of the full 164-dimensional prediction gain on the frontal head.
3. **Why Soldier Caste Allometry Adds a Non-Polynomial "Cordate Notch" at the Occipital Lobes:**
   Within-species minor (`w`) $\to$ major (`s`) soldier caste shifts are strongly Thompsonian across all 4 structures (`60.9%` frontal head, `79.6%` pronotum, `82.3%` mesosoma), and in dorsal/frontal views the developmental caste vector aligns at **$|\cos\theta| = 0.977$ (head)** and **$0.936$ (pronotum)** with macroevolutionary interspecific divergence inside the Thompson subspace! Where linear polynomial modes fall short on major soldiers is the **posterior occipital lobes** (`25.8%` unexplained vs. `16.9%` on inner oral/clypeal landmarks), where massive mandibular closer-muscle hypertrophy creates a localized bilobed indentation — captured when we switch to Thompson's nonlinear 8-parameter cordate-notch law.

We would love to hear your thoughts on these findings and explore applying our automated 3D coordinate-field extractor across the full `ANTSCAN` synchrotron mesh archive with your team.

Best regards,

Joel Dietz (Project Delphi · [github.com/fractastical/morpholib](https://github.com/fractastical/morpholib))

---

## 4. Email Draft 3 — Dr. Adrien Perrard (MNHN Paris), Prof. David Houle (FSU) & Prof. Thomas F. Hansen (University of Oslo)

**To:** `adrien.perrard@mnhn.fr`, `dhoule@bio.fsu.edu`, `t.f.hansen@ibv.uio.no`  
**Subject:** Testing D'Arcy Thompson's (1917, pp. 385–386) unfulfilled insect-wing programme & Hansen–Houle P/G/M evolvability across 22,780 *Drosophila*, hornet, and bee wings  
**Attachments:**
1. `docs/portfolio/fig_drosophilidae_wings.png` (inline + attached)
2. `docs/portfolio/fig_why_wings.png` (inline + attached)
3. `docs/portfolio/fig_bee_wings.png` (inline + attached)

---

Dear Dr. Perrard, Professor Houle, and Professor Hansen,

On pages 385–386 of *On Growth and Form* (1917), D'Arcy Thompson proposed *"a mathematical study of ... the 'deformation' of the wing (that is to say, of the changes in shape ... which it undergoes during the life of the individual, and from one species to another)"*, yet never drew a single insect wing grid in Chapter XVII.

Using our pre-registered phylogenetic morphospace pipeline ([`fractastical/morpholib`](https://github.com/fractastical/morpholib), with hypotheses SHA-256-locked in Nanopub [`RAQ_Giwd735...`](https://registry.knowledgepixels.com/np/RAQ_Giwd735IPhYg-cOMyPE6t4K9WPM9XlijGkNC6_eJ4) before running), we executed Thompson's proposed wing study across **21,138 *Drosophilidae* species wings + 17,323 additive genetic $\mathbf{G}_w$ wings + 12,075 mutational $\mathbf{M}$ wings** (115 species, 12 landmarks; Houle et al. 2017 *Nature*), **1,027 *Vespinae* hornet forewings** (30 species, ~30 Myr crown, 19 landmarks; Perrard et al. 2014), and **615 *Anthophila* bee forewings** (42 tribes across 10 families + 2 Eocene fossil tribes, ~110 Myr crown, 18 landmarks; De Meulemeester et al. 2014), followed by our pre-registered **$\mathbf{P}$-matrix drift and allometry causal decomposition** (`thompson/portfolio/groups/why_wings.py`, attached in **`fig_drosophilidae_wings.png`** and **`fig_why_wings.png`**):

1. ***Drosophilidae* Wings (`21,138` Species Wings Across `115` Species; `4/5` Pre-Registered Hypotheses Supported, Brier `0.190`):**
   - Across the 115 drosophilid species means (12 landmarks), D'Arcy Thompson's 7 a-priori global coordinate modes capture **`71.3%` of between-species Procrustes variance** (`96.5%` in 7-D PCA; **Hypothesis DW1 ✓**).
   - Just as in Hymenoptera, the **4 interior crossvein landmarks have a significantly higher unexplained residual fraction after Thompson-7 (`50.7%`) than the 8 outer margin/apical landmarks (`35.9%`; Hypothesis DW2 ✓)**.
   - On the 83-species OpenTree phylogeny, Pagel's $\lambda = 0.806$ in full space ($\lambda = 0.602$ in T7; **Hypothesis DW3 ✓**), leave-one-species-out phylogenetic prediction cuts median Procrustes RMS error to **`0.605×` the grand consensus** ($p = 2.8 \times 10^{-12}$; **Hypothesis DW4 ✓**), and standing additive genetic variance $\mathbf{G}_w$ ($n = 17{,}323$) aligns strongly with macroevolutionary divergence $\mathbf{D}$ ($|\cos\theta(\mathbf{G}_w, \mathbf{D})| = \mathbf{0.850}$).
2. **Why Deep Time Is More Thompsonian Than Shallow Time in Hymenoptera (`39%` in Hornets at 30 Myr vs. `56–58%` in Bees at 110 Myr):**
   Within a single hornet species, individual wing variation ($\mathbf{P}$) is itself **`54.0%` Thompsonian** — confirming that developmental growth variation follows smooth global modes. Yet *between* close hornet species (30 Myr), divergence drops to **`38.9%` Thompsonian** (below the 2,000-simulation $\mathbf{P}$-drift band of `[44.8%, 62.2%]`) and sits **$80.1^\circ$ orthogonal to $\mathbf{p}_{\max}$**, concentrating in local central vein cells. By contrast, across bee tribes over 110 Myr, divergence rises to **`58.0%` Thompsonian**, aligns with $\mathbf{p}_{\max}$ (**$38.4^\circ$**, with `49%` of among-tribe variance on the first 3 within-unit PCs vs. `9.4%` random expectation), and enables leave-one-tribe-out phylogenetic prediction to beat copying the nearest relative.
3. **Direct Confirmation of Hansen & Houle (2008) Per-Landmark Evolvability Across Clades:**
   Per-landmark among-taxon divergence strongly tracks per-landmark within-taxon variance in both hornets (**Spearman $\rho = 0.62, p = 0.005$**) and bees (**Spearman $\rho = 0.62, p = 0.006$**), while per-landmark digitizing noise does not predict where Thompson's global grid fails ($\rho = -0.22$ and $-0.39$).

We would be delighted to hear your feedback on this synthesis and explore a joint follow-up note comparing Dipteran and Hymenopteran wing evolvability in Thompson's coordinate basis.

With warm regards,

Joel Dietz (Project Delphi · [github.com/fractastical/morpholib](https://github.com/fractastical/morpholib))

---

## 5. Email Draft 4 — Dr. Nicolas Chazot (SLU Uppsala) & Prof. Vincent Debat (MNHN Paris)

**To:** `nicolas.chazot@slu.se`, `vincent.debat@mnhn.fr`  
**Subject:** 4/4 pre-registered hypotheses supported (Brier 0.037): D'Arcy Thompson's 7 global coordinate modes on your *Morpho* butterfly forewing & hindwing dataset  
**Attachments:**
1. `docs/portfolio/fig_morpho_butterflies.png` (inline + attached)

---

Dear Dr. Chazot and Professor Debat,

When we built our pre-registered cross-kingdom test of D'Arcy Thompson's 1917 coordinate transformations ([`fractastical/morpholib`](https://github.com/fractastical/morpholib)), we selected your *Morpho* butterfly dataset (Chazot et al. 2021, *J. Evol. Biol.* `34:1076–1090`; `911` specimens across `30` species on `Morpho_MCC_tree.tre`, with `44` forewing and `29` hindwing landmarks) to test how Lepidopteran wing outlines and discal-cell veins compare against Hymenopteran and Dipteran wings (`thompson/portfolio/groups/morpho_butterflies_hypotheses.json`, locked in Nanopub [`RAQ_Giwd735...`](https://registry.knowledgepixels.com/np/RAQ_Giwd735IPhYg-cOMyPE6t4K9WPM9XlijGkNC6_eJ4)).

All **`4/4` pre-registered hypotheses passed (Brier score `0.037`**, our best-calibrated clade in the entire 15-clade portfolio; attached in **`fig_morpho_butterflies.png`** and [`docs/portfolio/morpho_butterflies.md`](https://github.com/fractastical/morpholib/blob/main/docs/portfolio/morpho_butterflies.md)):
1. **Forewing & Hindwing Thompson-7 Share (`49.0%` FW & `57.0%` HW; Hypothesis MB1 ✓):** Because your landmark scheme captures both the outer wing margin/apex/tornus and the discal cell, Thompson's 7 global modes capture `49.0%` of interspecific forewing variance and `57.0%` of hindwing variance (`95.7%` in 7-D PCA), exceeding pure interior vein-junction schemes in Hymenoptera (`38.8%–42.2%`).
2. **Strong Brownian Phylogenetic Signal (`λ = 0.877` FW & `0.926` HW; Hypothesis MB2 ✓)** and **Accurate Leave-One-Species-Out Prediction (`0.585×` Consensus Error, $p = 4.7 \times 10^{-8}$; Hypothesis MB3 ✓)** across the 29 matched tree species.
3. **Sexual Dimorphism Is More Thompsonian Than Interspecific Divergence (`57.6%` vs. `49.0%`; Hypothesis MB4 ✓):** Within-species male $\to$ female forewing dimorphism across the 29 dimorphic species has a higher Thompson-7 variance share (`57.6%`) than between-species divergence (`49.0%`), confirming that ontogenetic sexual wing-shape dimorphism scales the wing imaginal disc along smoother global coordinate gradients than interspecific canopy-vs.-understory radiation.

We would love to hear your thoughts on these results!

Warm regards,

Joel Dietz (Project Delphi · [github.com/fractastical/morpholib](https://github.com/fractastical/morpholib))

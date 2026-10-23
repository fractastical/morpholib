# Methods & Provenance: The Adversarial Epistemic Loop (AEL) / Delphi-MorphoEval Protocol for Evaluating Michael Levin's "Latent Space of Patterns"

**Primary Author:** Joel Dietz ([Project Delphi](https://github.com/fractastical/project-delphi-ontology) · [Infinite Morphospace](https://github.com/fractastical/morpholib))  
**Date:** October 22, 2026  
**Companion Documents:**
- **4-Page Illustrated Magazine Edition Methods PDF (Sorting, CE 2.0, Thompson & Bioelectric Suite):** [`docs/levin_latent_space_methods_and_provenance.pdf`](./levin_latent_space_methods_and_provenance.pdf) ([LaTeX source](./levin_latent_space_methods_and_provenance.tex))
- **Substack Article (Copy-Paste Ready):** [`docs/SUBSTACK_LEVIN_LATENT_SPACE_ARTICLE.md`](./SUBSTACK_LEVIN_LATENT_SPACE_ARTICLE.md)
- **Sorting Morphogenesis & Causal Emergence 2.0 Academic Manuscript PDF (8 pages):** [`docs/zhang_levin_sorting_morphogenesis_manuscript.pdf`](./zhang_levin_sorting_morphogenesis_manuscript.pdf) ([LaTeX](./zhang_levin_sorting_morphogenesis_manuscript.tex))
- **Sorting Morphogenesis Illustrated Guide PDF (13-Year-Old Level):** [`docs/zhang_levin_13yo_illustrated_guide.pdf`](./zhang_levin_13yo_illustrated_guide.pdf) ([Markdown](../sorting-morphogenesis/EXPLAINED_FOR_13_YEAR_OLDS.md))
- **D'Arcy Thompson (1917) & 10,303-Species Methods Guide:** [`docs/THOMPSON_METHODS.md`](./THOMPSON_METHODS.md) ([4-Page Magazine PDF](./thompson_methods_and_provenance.pdf) · [28-Page Manuscript PDF](./thompson_1917_fish_transformations_manuscript.pdf))

---

## Executive Summary

This document records the exact computational, mathematical, formal-verification, knowledge-graph, and multi-agent methods used to test all seven pillars of Michael Levin's **"Latent Space / Platonic Space of Patterns"** thesis (*Ingressing Minds*, 2025/2026; Zhang, Goldstein & Levin 2025; D'Arcy Thompson 1917; Planarian `2H0T` Bioelectric Memory; Picasso Tadpole Craniofacial Homeostasis; and Xenobot Kinematic Self-Replication). Specifically, mirrors our [`docs/THOMPSON_METHODS.md`](./THOMPSON_METHODS.md) structure by answering five concrete methodological questions:

1. **What AI / LLM Agent Models and Human Reviewers Were Used** (and what exact role each played across multi-language porting, blinded cross-provider auditing, ablation design, and remediation)?
2. **What Mathematical, Algorithmic, Information-Theoretic & Biophysical Models Were Fitted** across all 7 claims (24 models in total)?
3. **How Did the Project Delphi RDF Knowledge Graph & Nanopublication Toolchain Drive the Investigation** (including `pk:evidenceAgainst` remediation queries and 200+ signed Nanopublications)?
4. **What Repository Code & Upstream Commits Were Audited or Reused** (with exact line-by-line traceability from `Zhangtaining/cell_research@1fd2bd59` to our Python 3, C++20, Node.js, and Lean 4 ports)?
5. **How Can Any Reader Reproduce Every Experiment from Scratch** (exact SHA-256 dataset pins and a 12-command deterministic execution map)?

---

## 1. AI / LLM Models, Blinded Cross-Provider Referees & Human Investigators

Every claim evaluated in the Substack article was built and audited inside Cursor using the repository's **12-Step MorphoEval Closed-Loop Protocol**, which pairs builder agents with blinded referee models from competing AI providers so no model grades its own un-audited code:

| Agent / Human Role | Model / Affiliation | Exact Contributions & Artifacts Across the 7 Claims |
| :--- | :--- | :--- |
| **1. Multi-Language Builder, Causal Emergence 2.0 Architect & 43-Turn Discovery Agent** | **`gemini-4-argon`** (`gemini-4-argon-medium`, Google DeepMind) | • Ported `Zhangtaining/cell_research@1fd2bd59` across **Python 3**, **C++20** (`sorting-morphogenesis/cpp/zhang_sorting_morphogenesis.cpp`), **Node.js** (`sorting-morphogenesis/js/zhang_sorting_morphogenesis.mjs`), and **Lean 4** (`morphoeval/lean/loops/Loop_zhang_sorting_morphogenesis.lean`).<br>• Built the 3-ablation Platonic Space audit (`sorting-morphogenesis/audit_platonic_space_claims.py`) and the exact $S_3 \dots S_6$ Markov-chain **Causal Emergence 1.0 & 2.0** analyzer (`sorting-morphogenesis/quantify_emergence_ce2.py`).<br>• Executed the blinded second-assessor MorphoEval pass (`2026-10-17-blinded-sample-audit.yaml`, Spearman $\rho = 0.891$ across 10 scripts; `90.8/100` on Thompson) and the 10,303-species Thompson synthesis. |
| **2. Primary MorphoEval Referee, Pass 3 Bioelectric / Xenobot Builder & Blind-Test Architect** | **`claude-fable-5-1`** (`claude-fable-5-1-thinking-high`, Anthropic) & **`claude-sonnet-5-5-high`** | • Conducted primary MorphoEval audits across the historical and 100-loop suites, authoring the `pk:evidenceAgainst` remediation items in `2026-10-17-thompson-audit.yaml` (L330–337) that sparked the 600-point outline breakthrough.<br>• Built the Pass 3 biophysical extensions for **Planarian Electrodiffusion** (`2017-pietak-bioelectric-morphogen-coupling.py`, 405-cell 2D Voronoi Scharfetter–Gummel network), **Picasso Tadpoles** (`2019-bischof-xenopus-craniofacial-remap.py`, 3D Maxwell cartilage rods + MMP-2/9 zymography), and **Xenobots** (`2020-kriegman-xenobot-evolutionary-biobots.py`, 8×8×7 3D voxel lattice + regularized-Stokeslet squirmer flow), plus their v3 referee audits.<br>• Built the 7-mode Phylogenetic Prediction Engine and `FishShapes v1` blind preregistration/scoring scripts. |
| **3. Independent Third-Company Bioelectric Referee** | **`gpt-5.6-sol`** (`gpt-5.6-sol-medium`, OpenAI) | • Executed the independent cross-company Pass 1 (`44.9/100`) and Pass 2 (`55.6/100`) MorphoEval audits of [`models/2021-Cervera–Levin–Mafe.py`](../models/2021-Cervera–Levin–Mafe.py) (`morphoeval/audits/2026-10-17-cervera-audit-v2.yaml`), catching an unused `f3` loop variable, a hard-coded `0.58` pharynx threshold, a bisection clipping bug in `f2_for_shift`, and a wrong PMID (`34418508 → 34411690`), which drove the creation of our 63-series `PlanformDB 2.5.0` evidence extractor ([`models/planform_bioelectric.py`](../models/planform_bioelectric.py)). |
| **4. Principal Investigator & External Domain Critiques** | **Joel Dietz** (PI, Project Delphi), **Prof. Peter C. Wainwright** (UC Davis), & **Nathan Sweet** (*Sweet Rationalism*) | • **Joel Dietz:** Formulated the knowledge-graph synthesis of Levin's literature; challenged 17-dot stick figures (exposing the $9/17$ skull-dot seesaw); demanded the 4p Box-vs.-Warping deconfounding test; and initiated the 3-ablation + Hoel CE 2.0 audit of *Ingressing Minds* vs. CS blog critiques.<br>• **Prof. Peter C. Wainwright:** Prompted the 600-semilandmark Curve-TPS / EFD head-to-head benchmark and convergent trajectory tests.<br>• **Nathan Sweet (*Sweet Rationalism*):** His blog critique of Levin's sorting paper motivated our Ablation 3 test of unique-key (`Fig. 8A`) vs. duplicate-key multiplicity (`C = 2` anti-clustering vs. `C = 10` clustering). |

---

## 2. Complete Inventory of Mathematical, Algorithmic & Biophysical Models Tested (24 Models Across 7 Claims)

To ensure that every claim in the Substack article was tested against explicit, falsifiable mathematical models rather than verbal metaphors, we implemented and verified **24 distinct models** across five domains:

### A. Decentralized Cell-View Sorting & Symmetric Group $S_n$ Models (Claims 1–3)
1. **Autonomous Cell-View Bubble Sort (`BubbleSortCell`):** Local random neighbor inspection $q \in \{p-1, p+1\}$; swaps if out of order and $\mathrm{status}_q \ne \mathtt{FREEZE}$.
2. **Autonomous Cell-View Insertion Sort (`InsertionSortCell`, With vs. Without L74–76 `FREEZE` Reset):** Steps left with $p-1$ iff prefix $0 \dots p-1$ is sorted; **Ablation 1** compares upstream `if status == FREEZE: prev = -1` against strict textbook prefix checking (`4.90 → 21.95` error jump at $n=60, f=3$).
3. **Autonomous Cell-View Selection Sort (`SelectionSortCell`, Reset-on-Swap vs. Persistent Pointer & `.pyc` Passive-Freeze Rule):** Maintains internal `ideal_position` (`target`) pointer scanning $0 \to n-1$, performing non-local swaps $(p, \text{target})$ across barriers.
4. **Three Concurrency Schedulers (`sorting_morphogenesis.py`):** (i) Random-asynchronous active-cell sampling (`SplitMix64`), (ii) synchronous maximal non-overlapping matching rounds, and (iii) literal per-cell OS threads (`threading.Thread` + `threading.Lock`).
5. **Coxeter Pairwise Inversion Lyapunov Potential $I(v)$ vs. Coarse-Grained Adjacent Monotonicity Error $E(v)$ (**Ablation 2**):**
   $$I(v) = \#\{(i < j) : v_i > v_j\} \quad (\Delta I = -1\text{ on every Bubble swap}), \qquad E(v) = \#\{i : v_i > v_{i+1}\}.$$
6. **Tie-Breaking Stability Ablation (`<` vs. `<=` on Equal Keys) & Duplicate-Multiplicity Domain-Wall Law $A_{\mathrm{final}}(C)$ (**Ablation 3**):** Sweeps $C = n/K \in \{1, 2, 4, 8, 10, 20, 40\}$ copies per key ($n=80$), uncovering **anti-clustering ($0.477 < 0.50$, `S B | S B`) at $C = 2$** and homotypic domain blocks (`SSSSS | BBBBB`, $0.659$) at $C = 10$.

### B. Erik Hoel's Causal Emergence 1.0 (PNAS 2013) & 2.0 (*Patterns* 2026, `PyMergence`) Models (Claim 2)
7. **Exact $N \times N$ Permutation Markov Transition Probability Matrices (TPMs) on $S_3, S_4, S_5, S_6$ ($N = 6, 24, 120, 720$):** Built for both conditional active-swap steps and raw proposal steps (`quantify_emergence_ce2.py`).
8. **Causal Emergence 1.0 ($\Delta\mathrm{EI} = \Delta I_{\mathrm{eff}} + \Delta I_{\mathrm{size}}$; Hoel et al. 2013):** Decomposes Effective Information into intensive effectiveness gain ($+0.476\text{ bits}$ on $I(v)$) plus extensive dimensionality penalty ($-4.663\text{ bits}$ from compressing $720 \to 16$ states).
9. **Causal Emergence 2.0 ($\mathrm{CP}(\pi) = \mathrm{det}(\pi) - \mathrm{deg}(\pi)$; Hoel 2026 *Patterns* `7:101472`):** Intensive Causal Primitive and multi-scale path apportioning across $\pi^{(0)} (S_6, 720) \to \pi^{(1)} ((I,\text{head}), 26\text{–}31) \to \pi^{(2)} (I(v), 16)$, proving $\Delta\mathrm{CP}_{I} = +0.1191 > 0$ ($\mathrm{det}_I = 1.0000$) vs. $\Delta\mathrm{CP}_{E} = -0.2525 < 0$ ($\mathrm{det}_E = 0.7209$).

### C. D'Arcy Thompson 2D Coordinate Field & Macroevolutionary Phylogenetic Models (Claim 4)
10. **11 Coordinate & Outline Transformation Laws:** 4p Procrustes similarity, **4p Orthotropic Rectangular Box Stretch** (zero-skew/zero-curve null), 5p Oblique Shear, 6p Uniform Affine, 6p Conformal Möbius, 8p Oblique Coaxial Circles, 8p Bilinear Fan, 8p Exponential Radial Flare, 12p Quadratic Polynomial Trend Surface, 34p/1200p Thin-Plate Splines (Bookstein 1989, 1997), and 8p Elliptic Fourier Descriptors, plus the baseline-anchored bidirectional Chamfer contour solver (`cKDTree`) and $600 \times 600$ Jacobian bijectivity verifier ($\det D\Phi > 0$).
11. **Multivariate Morphospace & Quantitative-Genetic Models (`791 → 6,614 → 10,303` Species):** Rohlf (1993) Non-Affine Relative Warps (LOFO-CV), Sidlauskas (2008) Phylomorphospace, Adams & Collyer (2009) Phenotypic Trajectory Analysis, and **Schluter's (1996) $\mathbf{G}$ vs. $\mathbf{R}$ Alignment** separating the Global 8p Thompson Morphospace ($|\cos\theta| = 0.935\text{–}0.989$) from the orthogonal Local Craniofacial Competition Module ($\theta = 87.4^\circ$).
12. **5 Phylogenetic Comparative Process Models (`run_phylogenetic_prediction_engine.py`):** White Noise, Brownian Motion, **Pagel's $\lambda$** ($\hat{\lambda} = 0.959$, $\Delta\mathrm{AIC}_c > 2,400$), Ornstein–Uhlenbeck, and Early Burst on 7 linearized Thompson tangent modes across 1,499 tree nodes and 5,180 blind-test species.

### D. Bioelectric, Homeostatic & Xenobot Biophysical Models (Claims 5–7)
13. **Planarian Bistable $V_{\mathrm{mem}}$ & Voltage-Gated Gap-Junction PDE + 2D Finite-Volume Body Mesh (`2011-beane-planarian-hk-atpase-anterior.py`, Slot 081):** Bistable membrane potential coupled to Bell-form innexin gap junctions, $\text{Ca}_{\text{v}}$ influx, slit/netrin mediolateral axon guidance, and stochastic neoblast kinetics across 5 pharmacological regimes (`1H1T` WT, `0H2T` SCH-28080, `2H0T` ivermectin, `2H0T` 8-octanol, `0H2T` nifedipine).
14. **Electrodiffusive Nernst–Planck 5-HT Transport & 405-Cell 2D Voronoi Scharfetter–Gummel Network (`2017-pietak-bioelectric-morphogen-coupling.py`, Slot 082):** Coupled 1D Nernst–Planck + 2D BETSE-style Voronoi electrodiffusion of 5-HT ($z=+1$), $\text{IP}_3$ ($z=-3$), and $\text{Ca}^{2+}$ ($z=+2$) with exact Boltzmann zero-flux equilibrium $c_{\mathrm{post}}/c_{\mathrm{ant}} = \exp(40/25.85) = 4.7002$ and stochastic Euler–Maruyama `1H1T`/`2H0T` blastema branching.
15. **Antagonistic Morphogen Reaction–Diffusion + 5-Band Bioelectric Circuit & 63-Series PlanformDB Extractor (`2021-Cervera–Levin–Mafe.py` & `planform_bioelectric.py`):** Pairs anterior-posterior reaction–diffusion wavefront shifts under gap-junction blockade with Oviedo et al. (2010), Nogi & Levin (2005), and Beane et al. (2011) empirical regeneration frequencies from `planformDB_2.5.0.edb`.
16. **Picasso Tadpole Procrustes Attractor, 3D Maxwell Viscoelastic Cartilage Rods & MMP-2/MMP-9 Zymography PDE (`2019-bischof-xenopus-craniofacial-remap.py`, Slot 085):** Closed-loop relational spring-target Lyapunov descent ($\dot{V} \le 0$) vs. open-loop fixed-displacement null across NF stages $45 \to 65$, coupled in Pass 3 to 5 3D Maxwell cartilage rods (Meckel's, ceratohyal, palatoquadrate) and MMP-2/MMP-9 reaction–diffusion gelatinase fields (reproducing GM6001 arrest).
17. **Active-Inference Variational Free Energy Homeostat (`2015-pezzulo-bioelectric-target-morphology-memory.py`, Slot 084):** Coupled perception-action gradient flow $\dot{\mu} = -\kappa_\mu \partial_\mu F$, $\dot{\mathbf{x}} = -\kappa_a \partial_{\mathbf{x}} F$ ($\dot{F} \le 0$) across 5 salamander limb amputation, ectopic graft, bioelectric setpoint-rewriting, and denervation regimes.
18. **Xenobot 8×8×7 3D Voxel Mass-Spring Lattice & 2D Regularized-Stokeslet Squirmer Stem-Cell Capture (`2020-kriegman-xenobot-evolutionary-biobots.py`, Slot 083):** AFPO phase-noise-robust gait evolution on a 456-node / 3,085-spring 3D lattice with Stokes drag, paired with a force- and torque-free Cortez (2001) regularized-Stokeslet ciliary squirmer solver demonstrating maximum multi-generational kinematic self-replication (`3–4` daughter generations) at C-mouth angle $\theta \approx 105^\circ$.

---

## 3. Line-by-Line Upstream Code Traceability (`Zhangtaining/cell_research@1fd2bd59`)

For Claims 1–3, every algorithmic rule in our four-language replication is pinned to exact file and line numbers in the authors' upstream repository ([`Zhangtaining/cell_research`](https://github.com/Zhangtaining/cell_research/tree/1fd2bd5921c1f6b423a71f691d5189106a8a1020) at commit `1fd2bd5921c1f6b423a71f691d5189106a8a1020`, with replication notes filed upstream in [Issue #2](https://github.com/Zhangtaining/cell_research/issues/2)):

| Rule ID | Paper Figure & Phenomenon | Upstream File & Line Numbers | Exact Upstream Implementation Logic | Our Ported Functions (`Py` / `C++20` / `Node.js` / `Lean 4`) |
| :-: | :--- | :--- | :--- | :--- |
| **`[U1]`** | Bubble local neighbor swap & barrier pinning (`Figs. 4A, 4B, 5C`) | `modules/multithread/BubbleSortCell.py` (`should_move` L38–52, `move` L54–74) | Inspects $p-1$ or $p+1$ at random ($p=0.5$); swaps if out of order and neighbor $\ne \mathtt{FREEZE}$. Zero defect-handling code; `0` cells cross an immovable defect. | `cell_should_move()`, `simulate_cell_view()`, Lean `bubble_adjacent_swap_preserves_left_partition` (Thm 1) |
| **`[U2]`** | **Insertion prefix-reset exception handler at frozen cells** (`Figs. 4A, 5B, 5C, 7B` & **Claim 1**) | `modules/multithread/InsertionSortCell.py` (`is_enable_to_move` **L68–82**, specifically **L74–76**) | Scans prefix $0 \dots p-1$; **explicitly executes `if self.cells[i].status == CellStatus.FREEZE: prev = -1`** so frozen cells reset prefix checking. Ablating L74–76 spikes error $4.5\times$ (`4.90 → 21.95`). | `_left_sorted(ignore_frozen_in_prefix=True/False)`, `left_sorted_with_frozen_reset()`, Lean `insertion_freeze_reset_unblocks_and_checks_post_frozen_order` (Thm 2) |
| **`[U3]`** | Selection `ideal_position` pointer & swap reset (`Figs. 4A, 4B, 5C`) | `modules/multithread/SelectionSortCell.py` (L41–88) & `MultiThreadCell.py` (`swap` L104–118) | Scans `target` $0 \to n-1$; swaps non-locally across barriers. `update()` resets `target = 0` on swap (`11.05×` swaps, Fig. 4A); keeping `target` gives `1.18×` steps (Fig. 4B). | `simulate(selection_reset_on_swap=True/False)`, Lean `selection_nonlocal_swap_strictly_decreases_inversions_window3` (Thm 4) |
| **`[U4]`** | Selection passive (`movable`) frozen bytecode rule (`Fig. 5B`) | `modules/multithread/__pycache__/SelectionSortCell.cpython-310.pyc` | Compiled Python 3.10 bytecode increments `target` when pointing at a passive `FREEZE` cell unless `t == occ.target` (preventing pointer livelock). | `_propose(selection_movable_mode="pyc310")` |
| **`[U5]`** | Delayed Gratification run segmentation (`Figs. 6D, 7A–C` & **Claim 2**) | `analysis/delay_gratification_analysis_for_not_move.py` (L24–89) | Deduplicates consecutive equal $E(v_t)$, groups same-sign $\Delta E_t$ into alternating setback ($y>0$) & recovery ($x>0$) runs: $\mathrm{DG} = \langle (x-y)/y \rangle$. | `delayed_gratification()`, `run_delayed_gratification_lyapunov_audit()`, Lean Thms 3 & 6 |
| **`[U6]`** | Chimeric algotype aggregation (`Figs. 8A, 8D–E, 9, 10` & **Claim 3**) | `analysis/cell_type_aggregation_analysis.py` (L518–538) | $A(t) = \frac{1}{n}\sum_{i=0}^{n-2} \mathbf{1}[\mathrm{algo}_i = \mathrm{algo}_{i+1}]$. | `aggregation_value()`, `run_chimeric_clustering_ablation()`, Lean Thm 5 |

---

## 4. Machine-Checked Lean 4 Theorems & Knowledge-Graph Nanopublications

All mathematical claims across the 7 pillars are verified by the **Lean 4 kernel (`0 sorry`, `0` custom axioms)** and serialized into cryptographically signed RDF Nanopublications:

| Claim / System | Lean 4 Module & Verified Theorem Names (`0 sorry`) | MorphoEval Audit & Signed Nanopub Provenance |
| :--- | :--- | :--- |
| **Claims 1–3: Sorting Morphogenesis & Hoel CE 2.0** | [`morphoeval/lean/loops/Loop_zhang_sorting_morphogenesis.lean`](../morphoeval/lean/loops/Loop_zhang_sorting_morphogenesis.lean) & [`Loop_causal_emergence.lean`](../morphoeval/lean/loops/Loop_causal_emergence.lean):<br>1. `bubble_adjacent_swap_preserves_left_partition`<br>2. `insertion_freeze_reset_unblocks_and_checks_post_frozen_order`<br>3. `lyapunov_descent_vs_monotonicity_error_projection_artifact`<br>4. `adjacent_swap_strictly_decreases_inversions_window3`<br>5. `selection_nonlocal_swap_strictly_decreases_inversions_window3`<br>6. `chimeric_segregation_vs_alternating_list_bounds`<br>7. `hoel_ce2_sorting_inversion_emergence_vs_error_loss` | • MorphoEval v2 score: **94.1 / 100**<br>• **10 Production Nanopubs** (`morphoeval/kg/2026-10-22-zhang-sorting-morphogenesis-audit-v2/`): Claim [`RAH6FDLaacsq...`](https://registry.knowledgepixels.com/np/RAH6FDLaacsq28cjx2NVRYRANqOThwHaHwE58ByxqIWI8), Overall [`RA_mMJXveSMN...`](https://registry.knowledgepixels.com/np/RA_mMJXveSMNlxIjYu_oTtIY233QfJnNCf47yRLfInMuE), Validation [`RAdFmYq_oVoH...`](https://registry.knowledgepixels.com/np/RAdFmYq_oVoHbK2eclzYgRWXJVhSNdiEQBfxp4nhvkzJk)<br>• Causal Emergence 2.0 loop (`Loop_causal_emergence.lean`): **5 theorems (`0 sorry`)**. |
| **Claim 4: D'Arcy Thompson 1917 & 10,303-Species Morphospace** | [`morphoeval/lean/MathOfBiology.lean`](../morphoeval/lean/MathOfBiology.lean) (§5, L536–560):<br>1. `tps_kernel_zero_at_origin`<br>2. `thompson_unit_shear_area_preserving` | • Primary (`claude-fable-5-1`): `76.6` · Blinded 2nd (`gemini-4-argon`): **`90.8`** (Panel `83.7/100`)<br>• **80 Production Nanopubs** (`nanopubs/thompson_predictions/published_manifest_prod.txt`), incl. pre-unblinding blind preregistration [`RAilN4nbRHp3...`](https://registry.knowledgepixels.com/np/RAilN4nbRHp3e3ZHqlbIgeL3UGPRV_KtQHtJD3gzVVdqw) and outcome [`RAJancwGOPEk...`](https://registry.knowledgepixels.com/np/RAJancwGOPEkHzBoZXGv_vu_snW0Lff2FktE85Me1P-fI). |
| **Claim 5: Planarian Bioelectric `2H0T` Memory (`Slot 081`, `Slot 082` & `Cervera 2021`)** | [`morphoeval/lean/loops/Loop_081_planarian_hk_atpase_anterior.lean`](../morphoeval/lean/loops/Loop_081_planarian_hk_atpase_anterior.lean) (6 theorems), [`Loop_082_bioelectric_morphogen_coupling.lean`](../morphoeval/lean/loops/Loop_082_bioelectric_morphogen_coupling.lean) (5 theorems), & `MathOfBiology.lean` (`cervera_equipotential_zero_current`, `cervera_bell_boltzmann_symmetric`, `lobo_2013_wildtype_dimensions`) | • `Slot 081` (Beane 2011): Pass 2 `93.5` $\to$ Pass 3 **`87.7/100`** (`2026-10-19-planarian-hk-atpase-anterior-audit-v3`, 10 signed nanopubs)<br>• `Slot 082` (Pietak & Levin 2017): Pass 2 **`93.8/100`** + Pass 3 2D Voronoi Scharfetter–Gummel<br>• `Cervera 2021`: OpenAI `gpt-5.6-sol` audit (`44.9 → 55.6/100`, 18 production nanopubs). |
| **Claim 6: Picasso Tadpoles & Limb Homeostasis (`Slot 084` & `Slot 085`)** | [`morphoeval/lean/loops/Loop_084_bioelectric_target_morphology_memory.lean`](../morphoeval/lean/loops/Loop_084_bioelectric_target_morphology_memory.lean) (3 theorems) & [`Loop_085_xenopus_craniofacial_remap.lean`](../morphoeval/lean/loops/Loop_085_xenopus_craniofacial_remap.lean) (5 theorems: Lyapunov dissipation, open-loop failure, bilateral symmetry restoration, Maxwell cartilage energy decay) | • `Slot 084` (Pezzulo & Levin 2015): Pass 1 `59.6` $\to$ Pass 2 **`91.6/100`** (20 signed nanopubs)<br>• `Slot 085` (Vandenberg, Adams & Levin 2012): Pass 2 `92.0` $\to$ Pass 3 **`89.0/100`** (`2026-10-19-xenopus-craniofacial-remap-audit-v3`, 10 signed nanopubs). |
| **Claim 7: Xenobot Biomechanics & Kinematic Self-Replication (`Slot 083`)** | [`morphoeval/lean/loops/Loop_083_xenobot_evolutionary_biobots.lean`](../morphoeval/lean/loops/Loop_083_xenobot_evolutionary_biobots.lean) (5 theorems: AFPO fitness bounds, C-mouth kinematic compaction law, `stokeslet_force_free_cancellation_and_mobility_superposition`, `lattice_drag_dissipation_nonneg_and_passive_spring_rest`) | • `Slot 083` (Kriegman, Blackiston, Levin & Bongard 2020/2021): Pass 1 `60.2` $\to$ Pass 2 `91.5` $\to$ Pass 3 **`82.7/100`** (`2026-10-19-xenobot-evolutionary-biobots-audit-v3`, 10 signed nanopubs). |

---

## 5. End-to-End Reproducibility Command Map (12 Commands to Verify Every Claim)

Every figure, JSON metric, C++20/Node.js port, and Lean 4 theorem cited in the Substack article can be reproduced deterministically from the repository root:

```bash
# ============================================================================
# PART I: SORTING MORPHOGENESIS, ABLATIONS 1-3 & CAUSAL EMERGENCE 2.0 (CLAIMS 1-3)
# ============================================================================
# 1. Bit-exact 4-language port verifier (Python 3 + C++20 + Node.js + Lean 4, 0 cell mismatches)
python3 sorting-morphogenesis/verify_multilang_ports.py

# 2. Three-ablation audit of Levin's 'Ingressing Minds' (2025/2026) Sec. 6.3 & Fig. 14 claims
python3 sorting-morphogenesis/audit_platonic_space_claims.py

# 3. Erik Hoel's Causal Emergence 1.0 (PNAS 2013) & 2.0 (Patterns 2026) on S_3..S_6 + C=1..40 sweep
python3 sorting-morphogenesis/quantify_emergence_ce2.py

# 4. Space-time kymographs & all 14 quantitative benchmarks of Zhang, Goldstein & Levin (2025)
python3 sorting-morphogenesis/validate_kymograph_and_paper.py --n 100 --reps 15

# 5. Kernel-check the 7 sorting & Causal Emergence 2.0 theorems in Lean 4 (exit 0, 0 sorry)
lean morphoeval/lean/loops/Loop_zhang_sorting_morphogenesis.lean
lean morphoeval/lean/loops/Loop_causal_emergence.lean

# ============================================================================
# PART II: D'ARCY THOMPSON 1917, 10,303 FISH SPECIES & BLIND TEST (CLAIM 4)
# ============================================================================
# 6. 17-landmark vs. 600-point silhouette paradox & 4-parameter Box-Stretch vs. Thompson Warp audit
python3 thompson/1917-thompson-silhouette-area-paradox.py
python3 thompson/run_box_vs_warping_audit.py

# 7. Three-dataset macroevolutionary validation (791 -> 6,614 -> 10,303 species; 440 families)
python3 thompson/run_three_dataset_validation.py

# 8. Pre-registered blind test scoring against 5,180 never-seen species in FishShapes v1
python3 thompson/score_blind_predictions.py

# ============================================================================
# PART III: BIOELECTRIC MEMORY, PICASSO TADPOLES & XENOBOTS (CLAIMS 5-7)
# ============================================================================
# 9. Planarian H,K-ATPase bistable V_mem (Slot 081) & Nernst-Planck 2D Voronoi 5-HT (Slot 082)
python3 models/2011-beane-planarian-hk-atpase-anterior.py --self-test
python3 models/2017-pietak-bioelectric-morphogen-coupling.py --self-test
python3 models/planform_bioelectric.py --self-test

# 10. Picasso tadpole 3D Maxwell cartilage + MMP zymography (Slot 085) & limb VFE memory (Slot 084)
python3 models/2019-bischof-xenopus-craniofacial-remap.py --self-test
python3 models/2015-pezzulo-bioelectric-target-morphology-memory.py --self-test

# 11. Xenobot 8x8x7 3D voxel lattice & regularized-Stokeslet squirmer self-replication (Slot 083)
python3 models/2020-kriegman-xenobot-evolutionary-biobots.py --self-test

# 12. Kernel-check all core & bioelectric Lean 4 theorems (exit 0, 0 sorry)
lean morphoeval/lean/MathOfBiology.lean
lean morphoeval/lean/loops/Loop_081_planarian_hk_atpase_anterior.lean
lean morphoeval/lean/loops/Loop_082_bioelectric_morphogen_coupling.lean
lean morphoeval/lean/loops/Loop_083_xenobot_evolutionary_biobots.lean
lean morphoeval/lean/loops/Loop_084_bioelectric_target_morphology_memory.lean
lean morphoeval/lean/loops/Loop_085_xenopus_craniofacial_remap.lean
```

# Publication Strategy, Target Venues & Upstream Citations: Planarian and Limb Regeneration as a Multiway Double-Pushout (DPO) Hypergraph Rewrite System

## 1. Manuscript & Reproducible Artifact Package

- **Compiled LaTeX Manuscript PDF:** [`docs/wolfram_planform_multiway_manuscript.pdf`](./wolfram_planform_multiway_manuscript.pdf)
- **LaTeX Source:** [`docs/wolfram_planform_multiway_manuscript.tex`](./wolfram_planform_multiway_manuscript.tex)
- **6-Panel Summary Figure:** [`docs/media/wolfram_planform_multiway_comparison.png`](./docs/media/wolfram_planform_multiway_comparison.png)
- **Executable Python + WXF Bridge & Self-Tests:** [`models/2026-wolfram-planform-multiway-rewriting.py`](../models/2026-wolfram-planform-multiway-rewriting.py)
- **Machine-Checked Lean 4 Module (`0 sorry`):** [`morphoeval/lean/loops/Loop_wolfram_planform_multiway.lean`](../morphoeval/lean/loops/Loop_wolfram_planform_multiway.lean)
- **SHA-256-Locked Reference Fixture:** [`datasets/planform_multiway/planform_multiway_reference.json`](../datasets/planform_multiway/planform_multiway_reference.json) (`52a6fda14d7fd2c9e46864b38b63b22edb625d0a16e2cebf5e1ae62e7a61bea2`)

---

## 2. Primary Upstream Software & Database Citations

1. **Multiway Hypergraph Rewriting Engine (`WolframInstitute/HypergraphRewritingEngine`, `v1.0.0-rc2`):**
   - **Citation:** Assar, R. (2026). *Rewriting the Universe: High-Performance Hypergraph Rewriting for the Wolfram Physics Project*. Technical Report, Wolfram Institute (`https://github.com/WolframInstitute/HypergraphRewritingEngine/blob/master/paper/main.pdf`).
   - **Role in this work:** Used via its standalone C++20 `hg_evolve` binary over WXF stdio for exact McKay IR canonicalization (`ir_core.hpp`), quotient multiplicity reconstruction (`quotient_multiplicity_core.hpp`), and discrete hypergraph geometry (`state_geometry_core.hpp`).
2. **PlanformDB 2.5.0 & LimbformDB 2.1.0 (Lobo Lab / Levin Lab):**
   - **Citations:**
     - Lobo, D., Malone, T. J., & Levin, M. (2013). Planform: an application and database of graph-encoded planarian regenerative experiments. *Bioinformatics* 29(8):1098–1100 (`doi:10.1093/bioinformatics/btt088`).
     - Lobo, D., Malone, T. J., & Levin, M. (2013). Towards a bioinformatics of patterning: a computational approach to understanding regulative morphogenesis. *Biology Open* 2(2):156–169 (`doi:10.1242/bio.20123400`).
     - Lobo, D., Feldman, E. B., Shah, M., Malone, T. J., & Levin, M. (2014). Limbform: a functional ontology-based database of limb regeneration experiments. *Bioinformatics* 30(24):3598–3600 (`doi:10.1093/bioinformatics/btu582`).
     - Lobo, D., Feldman, E. B., Shah, M., Malone, T. J., & Levin, M. (2014). A bioinformatics expert system linking functional data to anatomical outcomes in limb regeneration. *Regeneration* 1(2):37–56 (`doi:10.1002/reg2.13`).

---

## 3. Top Target Venues (Ranked by Fit)

| Rank | Journal / Venue | Why It Fits |
| :-: | :--- | :--- |
| **1** | ***Bioinformatics* (Oxford University Press)** | Published both original database papers: **Planform** (`Bioinformatics 29(8):1098–1100`, 2013) and **Limbform** (`Bioinformatics 30(24):3598–3600`, 2014). A direct follow-up introducing exact McKay IR canonicalization, 7 hidden isomers, and a $44,534\times$ C++20 multiway DPO grammar over both databases is a natural *Bioinformatics* Application Note or Original Paper. |
| **2** | ***Complex Systems*** | Flagship venue for discrete rewriting systems; ideal if framed around multiway DPO reducibility, `OEIS A000055` quotient reconstruction, and Wolfram Hausdorff / Ollivier–Ricci geometry on biological graphs. |
| **3** | ***PLOS Computational Biology* / *Patterns* (Cell Press)** | Ideal for a broader mathematical-biology synthesis combining the 3-rule DPO reducibility hierarchy ($\mathcal{B}_3$), bioelectric `RuleWeights` gating against Oviedo et al. (2010) S1/S2, and machine-checked Lean 4 proofs. |
| **4** | ***Journal of The Royal Society Interface*** | Strong venue for bridging formal graph-rewriting engines with empirical developmental and regenerative morphospaces. |

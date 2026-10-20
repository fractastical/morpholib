# Ripping Leaf Edge Waves & Self-Similar Buckling Cascades (`datasets/ripping_leaf_edge_waves/`)

Reference dataset and biological specimen assets for **Slot 067 (`sharon-ripping-leaf-edge-waves`)**:

- **Primary Landmark Paper:** Sharon, E., Roman, B., Marder, M., Shin, G.-S., & Swinney, H. L. (2002). *Mechanics: Buckling cascades in free sheets.* **Nature**, 419(6907), 579–580. DOI: [`10.1038/419579a`](https://doi.org/10.1038/419579a)
- **Self-Similar Boundary Layer Theory:** Audoly, B., & Boudaoud, A. (2003). *Self-Similar Structures near Boundaries in Strained Systems.* **Physical Review Letters**, 91(8), 086105. DOI: [`10.1103/PhysRevLett.91.086105`](https://doi.org/10.1103/PhysRevLett.91.086105)
- **Leaf Saddle-to-Ripple Morphospace:** Liang, H., & Mahadevan, L. (2009). *The shape of a long leaf.* **Proceedings of the National Academy of Sciences**, 106(52), 22049–22054. DOI: [`10.1073/pnas.0911954106`](https://doi.org/10.1073/pnas.0911954106)
- **Biological & Physical Synthesis:** Sharon, E., Marder, M., & Swinney, H. L. (2004). *Leaves, Flowers and Garbage Bags: Making Waves.* **American Scientist**, 92(3), 254–261. DOI: [`10.1511/2004.3.254`](https://doi.org/10.1511/2004.3.254)

## Biological Specimen (`lactuca_sativa_edge_waves_specimen.jpg`)

- **Organism:** *Lactuca sativa* cv. 'Lollo Bionda' (Ruffled garden lettuce exhibiting multi-generation self-similar marginal buckling waves)
- **Source:** Wikimedia Commons (`File:Lactuca sativa 'Lollo Bionda'.jpg`)
- **License:** CC BY-SA 4.0

## Pass 3 addendum (2026-10-20) — developmental CIN/TCP · auxin/PIN1 · CUC2 coupling

Closes the Pass 2 remediation item ("couple the quasi-static non-Euclidean FvK solver to spatio-temporal CIN/TCP and auxin transport PDEs"). New files:

- `tcp_auxin_developmental_reference.json` — SHA-256 `31eb569d60e0c76531bbcf5e29f2ad0838991ef87de8fcf3ec1bffe5efc99920`, pinned in the script (`EXPECTED_DEVELOPMENTAL_SHA256`). Contains the governing equations of the reduced 1D margin model, the wild-type calibration target (`g_0 = 0.64`, the Pass 2 *Lactuca* value — a cross-module self-consistency target, not an independent measurement) and a **genotype phenotype table** (wild type, *cin*/jaw-D TCP loss-of-function, miR319-resistant TCP over-expression, *cuc2*, *pin1*). Phenotype entries are **ordinal / qualitative transcriptions** of the cited papers; serration-count ranges are broad plausibility bounds, not digitised counts.
- `developmental_citations_pass3.bib` — nine additional references (Nath et al. 2003 *Science*; Palatnik et al. 2003 *Nature*; Bilsborough et al. 2011 *PNAS*; Smith et al. 2006 *PNAS*; Jönsson et al. 2006 *PNAS*; Koyama et al. 2007 *Plant Cell*; Nikovics et al. 2006 *Plant Cell*; Efroni et al. 2008 *Plant Cell*; Hay, Barkoulas & Tsiantis 2006 *Nat. Genet.*). DOIs were transcribed offline and **not re-verified online** in this pass. The Pass 2 `paper_citations.bib` is unchanged (it remains SHA-256-pinned).

Model summary (script section 5b, `simulate_tcp_auxin_margin_development`): auxin with PIN1 up-the-gradient transport gated by CUC2, CUC2 repressed by auxin and by the basipetal CIN/TCP maturation front `m(y,t)`, interior and marginal growth strains `Γ_int`, `Γ_mar` (auxin-boosted margin) on a growing Lagrangian margin lattice, excess metric `g_0(y,t) = ((1+Γ_mar)/(1+Γ_int))² − 1` handed to `relax_foppl_von_karman_cascade`, and serration outgrowth `ds/dt = k_s a (1−m)`. Outputs: `docs/media/ripping_leaf_edge_waves_tcp_auxin_development.png` and `.gif`.

# Slot 019 Reference Dataset — Avian Feather Tract Hexagonal Array, EDA Competence Wave & Curved-Surface Tract Boundaries (`datasets/avian_feather_bud_hexagonal/`)

Reference parameters, Keller–Segel / FGF20 chemotaxis + BMP lateral-inhibition benchmarks, the Ectodysplasin (*EDA*) travelling competence wave, the mesenchymal **density-threshold tract boundary** (Ho et al. 2019 Figs 3–5), the curved trunk-and-limb-bud surface model (Pass 3), perturbation phenotypes (*scaleless* `FGF20 sc/sc`, Latrunculin A mechanical blockade / pause-release, Fc-chEDA1, Ecto-D2, methotrexate), and biological specimen photography for **`2018-glimm-avian-feather-bud-hexagonal.py`**.

## Provenance correction (Pass 3, 2026-10-20)

The Pass 1–2 versions of this dataset (and of the script, audits v1–v2 and `scout_manifest.json`) cited **"Glimm, Bhat & Newman (2014) *Modeling happens in rows: Traveling waves in dermal condensation of avian feather primordia*, Dev. Biol. 395(1):139–149"** as the seed paper. **That paper does not exist.** Crossref resolves `10.1016/j.ydbio.2014.08.030` to Garnatz et al. (Dev. Biol. 395:50–61, FOG‑2/NuRD cardiomyocyte proliferation) and the dispatch's seed DOI `10.1016/j.ydbio.2014.09.006` to Kishi et al. (Dev. Biol. 395:299–306, *Oikopleura* cell migration). The chemotaxis model actually used by the script is **Painter, Ho & Headon (2018)**, and the wave / density-threshold / tract biology is **Ho et al. (2019)**. Two further citation errors were corrected (Ho 2019 title; Michon 2008 venue and DOI). Full details are in `metadata.provenance_corrections` of the reference JSON. The script file name and `sim_id` keep the historical "glimm" token only for audit-chain continuity.

`scout_manifest.json` is left byte-identical (its SHA-256 is pinned) and therefore still carries the non-existent seed paper; its `seed_landmark_paper` / `seed_doi` fields are superseded by the block above.

## Primary Literature (all DOIs Crossref-verified 2026-10-20)

1. **Ho, W. K. W., Freem, L., Zhao, D., Painter, K. J., Woolley, T. E., Gaffney, E. A., McGrew, M. J., Tzika, A., Milinkovitch, M. C., Schneider, P., Drusko, A., Matthäus, F., Glover, J. D., Wells, K. L., Johansson, J. A., Davey, M. G., Sang, H. M., Clinton, M., & Headon, D. J. (2019).** *Feather arrays are patterned by interacting signalling and cell density waves.* **PLOS Biology**, 17(2), e3000132. (`doi:10.1371/journal.pbio.3000132`) — specimen source (Fig 3, CC BY 4.0); EDA competence wave; ~75 % / ~50 % six-sided Voronoi cells (control vs Latrunculin-paused, Fig 2D); margin mesenchymal density 49 / 66 / 98 cells per 100 µm (Fc-chEDA1 / control / Ecto-D2, Fig 5C–D); MTX + Ecto-D2 co-treatment (Fig 5E–F).
2. **Painter, K. J., Ho, W., & Headon, D. J. (2018).** *A chemotaxis model of feather primordia pattern formation during avian development.* **Journal of Theoretical Biology**, 437, 225–238. (`doi:10.1016/j.jtbi.2017.10.026`) — Keller–Segel dermal chemotaxis + FGF20 / BMP lateral inhibition PDE system and its linear stability threshold χ_c.
3. **Michon, F., Forest, L., Collomb, E., Demongeot, J., & Dhouailly, D. (2008).** *BMP2 and BMP7 play antagonistic roles in feather induction.* **Development**, 135(16), 2797–2805. (`doi:10.1242/dev.018341`)
4. **Bailleul, R., Curantz, C., Desmarquet-Trin Dinh, C., Hidalgo, M., Touboul, J., & Manceau, M. (2019).** *Symmetry breaking in the embryonic skin triggers directional and sequential plumage patterning.* **PLOS Biology**, 17(10), e3000448. (`doi:10.1371/journal.pbio.3000448`)
5. **Bausch, A. R., Bowick, M. J., Cacciuto, A., Dinsmore, A. D., Hsu, M. F., Nelson, D. R., Nikolaides, M. G., Travesset, A., & Weitz, D. A. (2003).** *Grain boundary scars and spherical crystallography.* **Science**, 299(5613), 1716–1718. (`doi:10.1126/science.1081160`) — curvature-induced 5/7 disclination theory used for the curved-surface tract dislocations (Lean Theorem 4).

## What Pass 3 added to the reference JSON

- `metadata.provenance_corrections` — the correction record above.
- `ho2019_quantitative_anchors` — six-sided fractions (0.75 / 0.50), margin densities (49 / 66 / 98 cells per 100 µm), tract list, and an explicit list of quantities the paper does **not** report numerically (width ratios, in vivo wave speed in µm/h, apterium width).
- `governing_equations.curved_surface_metric`, `laplace_beltrami`, `geodesic_competence_wave`, `density_threshold_competence`, `nucleation_rule`, `model_boundary_note` (nucleation is a wave-gated inhibitory-disk rule, not emergent from the PDE).
- `curved_surface_model` — Monge-patch trunk ridge + four Gaussian limb buds, grid 256 × 184, limb-height sweep (0, 15, 30, 45), curvature threshold for defect localisation, and listed limitations.
- `experimental_phenotypes` rewritten: the "simultaneous" low-regularity regime is now attributed to the Latrunculin pause/release experiment (Fig 2B/D), not to Fc-chEDA1, which widens the tract by lowering the critical density.

## Vendored Files & SHA-256 Provenance Locks

| File | Description | SHA-256 |
| :--- | :--- | :--- |
| `avian_feather_bud_hexagonal_reference.json` | Governing equations (flat + curved-surface), canonical parameters, Ho 2019 quantitative anchors, specimen ROIs, phenotypes, provenance corrections | `61c579212066cfb2d74369d450734464539786cd32a51bd95bf0f07d505f15e5` |
| `chick_feather_bud_hexagonal_specimen.jpg` | Embryonic chick (*Gallus gallus*) dorsal feather tract E6.5+24h/48h *CTNNB1*/*EDAR*/*EDA* travelling wave and 60° hexagonal primordium lattice (Ho et al. 2019 *PLOS Biology* 17(2):e3000132 Fig. 3, CC BY 4.0) — unchanged since Pass 1 | `ce18b1c0e01ecb20e137a700129b328815ba10f236e8620d20ef66bbc8e7a03c` |
| `scout_manifest.json` | Automated literature & specimen scout dossier for Slot 019 (unchanged; seed citation superseded, see above) | `1ff38c1e609c69ea4826a33e48d962da2821dc4d04051d6919c440326a4da2ec` |
| `paper_citations.bib` | BibTeX for Ho et al. (2019), Painter et al. (2018), Michon et al. (2008), Bailleul et al. (2019), Bausch et al. (2003); fabricated Glimm 2014 entry removed | `721ac5198ac62b2de1386a5a99943e06ee9347af01fb75bcfb8864e3bcb69903` |

The script's `EXPECTED_SHA256` table pins the same four hashes and `--self-test` check 1 fails if any file drifts or if the fabricated DOI reappears in `metadata.sources`.

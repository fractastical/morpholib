# Honda (1971) 3D Dichotomous Tree Branching, Honda & Fisher (1978) Canopy Light Optimization & Leonardo Pipe-Model Dataset (`datasets/dichotomous_tree_branching/`)

Reference dataset and empirical *Cornus controversa* (Pagoda dogwood) tiered crown specimen for **MorphoEval Slot 051 (`honda-dichotomous-tree-branching`)**, implemented in `1976-honda-dichotomous-tree-branching.py` and formally verified in `morphoeval/lean/loops/Loop_051_dichotomous_tree_branching.lean`.

## Primary Literature & DOIs

1. **Honda, H. (1971).** Description of the form of trees by the parameters of the tree-like body: Effects of the branching angle and the branch length on the shape of the tree-like body. *Journal of Theoretical Biology*, 31(2), 331–338. DOI: [`10.1016/0022-5193(71)90031-2`](https://doi.org/10.1016/0022-5193(71)90031-2)
2. **Honda, H., & Fisher, J. B. (1978).** Tree branch angle: maximizing effective leaf area. *Science*, 199(4331), 888–890. DOI: [`10.1126/science.199.4331.888`](https://doi.org/10.1126/science.199.4331.888)
3. **Fisher, J. B., & Honda, H. (1979).** Branch geometry and effective leaf area: a study of *Terminalia*-branching pattern. 1. Theoretical trees. *American Journal of Botany*, 66(6), 633–644. DOI: [`10.1002/j.1537-2197.1979.tb06267.x`](https://doi.org/10.1002/j.1537-2197.1979.tb06267.x)
4. **Shinozaki, K., Yoda, K., Hozumi, K., & Kira, T. (1964).** A quantitative analysis of plant form—the pipe model theory: I. Basic analyses. *Japanese Journal of Ecology*, 14(3), 97–105. DOI: [`10.18960/seitai.14.3_97`](https://doi.org/10.18960/seitai.14.3_97)
5. **Eloy, C. (2011).** Leonardo's rule, self-similarity, and wind-induced stresses in trees. *Physical Review Letters*, 107(25), 258101. DOI: [`10.1103/PhysRevLett.107.258101`](https://doi.org/10.1103/PhysRevLett.107.258101)

## Files & Integrity

- `scout_manifest.json`: Literature and Wikimedia Commons scout dossier for Slot 051 (`3959867d4fc3a5ec79061986076a6da3fd49030e6e6cc7a00935b2422681a488`).
- `dichotomous_tree_branching_reference.json`: Reference parameters and quantitative benchmarks for Honda's (1971) 3D vector-rotation branching recursion, Honda & Fisher's (1978) 2D $(\theta_1, \theta_2)$ effective leaf area optimization, Leonardo da Vinci / Shinozaki pipe-model cross-sectional area conservation ($d_0^2 = d_1^2 + d_2^2$), 4 tree crown archetypes, and *Cornus controversa* specimen morphometrics (`9b8cd89b9fa19fbf89acaf9e5dc891c6f7ada36cb5a97d9a9a1e0007925d3c5e`).
- `paper_citations.bib`: BibTeX citations with verified DOIs (`722524d3620bdc0ee05acf112e400cbc2c5bd373f27cbca6aa1d7378d553b2c0`).
- `cornus_controversa_tiered_crown_specimen.jpg`: Botanical specimen photograph of *Cornus controversa* 'Variegata' (`File:Cornus controversa 'variegata' - NBGB - IMG 4246.JPG`, Wikimedia Commons, Public domain, 960×1280 px, SHA-256: `47cbd7225997789e45e248672847ff97ef9d9411b66e335df91d17c943e7566e`).

## Pass 3 additions (solar-elevation ray tracing of tiered crowns, `2026-10-19-dichotomous-tree-branching-audit-v3`)

The Pass 2 audit asked to couple the 3D Honda tiered crowns to solar-elevation radiative transfer across diurnal sun paths and to compare equatorial vs temperate optimal tier spacing. The script now stacks Honda & Fisher 5-whorl tiers on an excurrent leader (`build_honda_tiered_crown_3d`), ray-traces tier-to-tier shadows (horizontal displacement $\Delta z \cot h$, Beer–Lambert per-tier transmittance $\tau_c = e^{-\mathrm{LAI}_c}$) over 12 Cooper mid-month declinations × hourly sun positions (`ray_trace_tiered_crown_interception`, `evaluate_latitude_tier_spacing_tradeoff`), and measures the specimen's own inter-tier spacing / tier half-width ratio (`measure_specimen_tier_spacing`).

- `solar_tier_spacing_reference.json` (SHA-256 `640596fc8c3df302608366676688dbebd8b51fcc2613ffde7aa4c5cec3688886`, pinned in the script as `EXPECTED_SHA256_PASS3`): solar-geometry constants with sources (axial tilt 23.45°, clear-sky beam transmittance 0.75, Cooper mid-month days), per-tier transmittance $\tau_c = 0.135$ (LAI$_c$ = 2), latitude benchmarks (analytic noon-equinox elevation, day length; Botanic Garden Meise 50.93°N as specimen provenance), the specimen tier-spacing regression value $s/R = 0.376 \pm 0.134$ (projected, computed by the script — **not** an independent literature value) and the model's own Pass 3 predictions (regression only).
- `solar_citations_pass3.bib`: Campbell & Norman (1998) *An Introduction to Environmental Biophysics* Ch. 11 (DOI `10.1007/978-1-4612-1626-1`), Cooper (1969) *Solar Energy* 12:333–346 (DOI `10.1016/0038-092X(69)90047-4`), Monsi & Saeki (1953) *Jpn. J. Bot.* 14:22–52 (English translation *Ann. Bot.* 95:549–567, DOI `10.1093/aob/mci052`). DOIs were not re-verified online during the offline Pass 3 run.

Headline Pass 3 result (7 tiers, $\tau_c = 0.135$): the spacing at which annual beam capture reaches 90 % of the no-self-shading limit is $s_{90}/R = 1.83$ at the equator vs $0.97$ at 45° (ratio 1.89; Meise 0.85, 60° 0.71). The specimen's projected spacing $s/R = 0.376$ gives $E = 0.747$ at Meise vs $0.621$ for the same crown at the equator. Optimal tier spacing vs latitude has no literature benchmark; these are falsifiable model predictions. Diffuse sky light, leaf scattering and photographic perspective are not modelled.

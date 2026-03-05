# External Pre-Existing Dataset Candidates for Morphospace Visuals

This short-list prioritizes datasets/repositories that are public, citable, and suitable for compelling morphospace visualizations without relying on ad-hoc internal-only inputs.

## Recommended Starter Set (Best balance of rigor + practicality)

1. **Dryad geometric morphometrics datasets** (published, DOI-backed, downloadable)  
2. **SB Morphometrics standard datasets** (classic benchmark sets, easy to prototype)  
3. **MorphoSource open-download projects** (high visual impact via 3D forms)

Then optionally add:
4. **Paleobiology Database (PBDB) API** for macroevolutionary/time-aware morphospace layers.

---

## Candidate Repositories

## A) Dryad (DOI datasets with landmark/shape files)

**Why this is strong**
- DOI-citable, tied to peer-reviewed studies.
- Often includes raw landmarks, processed matrices, and analysis scripts.
- Good for reproducible morphospace demos and paper-quality comparisons.

**Examples surfaced**
- Crocodylian craniofacial disparity dataset.
- Characiform fish morphometrics.
- Ariidae skull geometric morphometrics.

**Best-fit visuals**
- Trajectory morphospace
- Disparity-through-time/group
- Pareto/archetype front (if objectives are derivable)

**Integration effort:** Low-Medium

---

## B) SB Morphometrics “standard datasets”

**Why this is strong**
- Widely used benchmark-style morphology datasets.
- Fastest path to stable, comparable visual outputs.

**Best-fit visuals**
- PCA/UMAP morphospace
- Occupancy + disparity decomposition
- Teaching/demo panels in README/docs

**Integration effort:** Low

---

## C) MorphoSource (open 3D morphology media)

**Why this is strong**
- Very compelling visuals (3D meshes/CT-derived forms).
- Large catalog across taxa; strong outreach/presentation value.

**Best-fit visuals**
- 3D morphospace + embedded specimen thumbnails/renders
- Shape manifold movies
- Archetype exemplars at vertices of trade-off fronts

**Caveat**
- Access/licensing may vary by project/media item; filter to open-download + compatible license before use.

**Integration effort:** Medium

---

## D) MorphoBank (project-based morphological matrices)

**Why this is strong**
- Established morphology research platform.
- Useful for matrix-driven morphospace and phylogenetic overlays.

**Best-fit visuals**
- Phylomorphospace-style panels
- Character-matrix disparity maps

**Caveat**
- Some projects have access restrictions or non-uniform export workflows.

**Integration effort:** Medium

---

## E) Paleobiology Database (PBDB) API

**Why this is strong**
- Programmatic API and broad temporal/taxonomic coverage.
- Excellent for time-aware occupancy and macro patterns.

**Best-fit visuals**
- Disparity-through-time
- Chrono-stacked morphospace occupancy
- Geographic/time faceting

**Caveat**
- Trait/morphology completeness is uneven; requires careful filtering and QA.

**Integration effort:** Medium-High

---

## F) Allen Cell Explorer morphology data (cell-shape domain)

**Why this is strong**
- High-quality segmentation-driven morphology features.
- Great for modern, high-dimensional morphospace workflows.

**Best-fit visuals**
- Dense manifold visualizations
- Topological morphospace (Mapper/TDA)

**Caveat**
- Domain is cell morphology (not organism-level developmental anatomy), so use only if domain expansion is acceptable.

**Integration effort:** Medium

---

## Not prioritized for this project

- **OpenTopography**: geospatial terrain focus, not biological morphology curated for this use.
- **EMPIAR**: excellent archive, but generally heavier for this specific morphospace objective unless we commit to EM-centric workflows.

---

## Selection Criteria (what we should enforce)

- Public + reusable license
- Stable URL/DOI and citation metadata
- Sufficient sample size for morphospace structure
- Machine-readable tables/landmarks/meshes
- Clear units and metadata (critical for scale-aware visuals)

---

## Proposed Decision

If your requirement is “external pre-existing data only,” the safest first implementation is:

1. **Dryad dataset ingestion** (one fish/amphibian/crocodylian set)  
2. Build **trajectory + disparity morphospace** from that dataset  
3. Add **MorphoSource** exemplar renders for visual impact

This gives both scientific credibility and compelling visuals quickly.


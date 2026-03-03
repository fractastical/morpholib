# Morphospace Visualization Deep Dive (Literature-Informed)

This document proposes compelling morphospace visualization approaches from academic literature and maps them to this repo's data and workflows.

## Goal

Identify morphospace visuals that are:
- scientifically defensible,
- compelling for papers/talks,
- practical with current outputs (`spark_tracks.csv`, `vector_clusters.csv`, detection summaries),
- compatible with zoom-aware and unit-aware comparisons.

---

## High-Impact Visualization Candidates

## 1) Phylomorphospace / Chronophylomorphospace-Style Lineage Maps

**What it is**  
Plot morphology/phenotype points in reduced space and connect related entities over time or ancestry-like structure.

**Why compelling**  
Shows trajectories and branching rather than static clouds; great for "how forms move through space over time."

**Literature anchors**
- `phytools::phylomorphospace` style plots and 3D variants.
- `deeptime::geom_phylomorpho`.
- Claddis stack/time-space variants (`plot_morphospace_stack`).

**How to adapt here**
- Build pseudo-lineages by grouping runs by condition/timepoint (or actual experiment chronology).
- Features: speed stats, AP/DV occupancy, signal volume, latency, head-tail asymmetry.
- Visual: 2D/3D embedding + path segments by condition/year/dataset.

**Effort**: Medium  
**Data readiness**: Medium

---

## 2) Disparity-Through-Time (DTT) + Occupancy Stacks

**What it is**  
Quantify and animate/plot how occupied morphospace volume, density, and position shift across time bins.

**Why compelling**  
Turns qualitative "looks more diverse now" claims into measurable dynamics.

**Literature anchors**
- Disparity metric frameworks and time-stratified morphospace analyses.
- `dispRity` and related occupancy/disparity methods.

**How to adapt here**
- Bin events/tracks by `time_s` relative to poke (e.g., pre, 0-5s, 5-30s, 30-120s).
- Compute volume/density/centroid-shift metrics in latent space.
- Plot stacked panels or ridge-over-time occupancy.

**Effort**: Medium  
**Data readiness**: High

---

## 3) Adaptive Landscape (Performance/Fitness Surface) Over Morphospace

**What it is**  
Interpolate a scalar performance variable over morphospace (kriged/GP-like surface).

**Why compelling**  
Transforms scatter into interpretable "peaks, ridges, valleys" with biological narrative.

**Literature anchors**
- Morphoscape-style adaptive landscape visualization.
- Kriging/interpolated surface approaches in phenotype spaces.

**How to adapt here**
- Morphospace axes from PCA/UMAP of cluster-level features.
- Surface Z value options:
  - propagation success,
  - mean speed (ecm/s or um/s),
  - inter-embryo transfer likelihood,
  - recovery/decay slope.
- Output: contour map + uncertainty overlay.

**Effort**: Medium-High  
**Data readiness**: Medium

---

## 4) Pareto Front / Archetype Morphospace

**What it is**  
Model trade-offs across tasks and identify boundary/frontier phenotypes (archetypes).

**Why compelling**  
Very strong explanatory structure for "no one phenotype optimizes everything."

**Literature anchors**
- Pareto optimality in phenotype space.
- Geometry of phenotype-space fronts (line/triangle/tetrahedron archetypes).
- Trade-off analyses in shells and life-history systems.

**How to adapt here**
- Define competing objectives, e.g.:
  - high propagation speed,
  - low latency,
  - high spatial reach,
  - low signal dissipation.
- Compute non-dominated points and visualize front in 2D/3D embedding.
- Label frontier archetypes and show representative videos/images.

**Effort**: Medium  
**Data readiness**: Medium

---

## 5) Ontogenetic/Response Trajectory Morphospace

**What it is**  
Trajectory lines through morphospace for developmental stages or response phases.

**Why compelling**  
Ideal for injury-response dynamics (initiation -> propagation -> recovery) as paths.

**Literature anchors**
- Ontogenetic trajectory morphometrics.
- Growth-series morphospace trajectory analyses.

**How to adapt here**
- Per embryo/run, create phase windows (baseline, early wave, propagation, late decay).
- Plot trajectory arrows with velocity coloring.
- Compare trajectory length, turning angle, and endpoint distributions by condition.

**Effort**: Medium  
**Data readiness**: High

---

## 6) Hypervolume Occupancy + Density Decomposition

**What it is**  
Estimate occupied multidimensional volume and partition changes into volume/density/position components.

**Why compelling**  
Distinguishes whether groups differ because they spread, shift, or cluster.

**Literature anchors**
- Occupancy metric comparisons (volume, density, position families).
- Trait-space shift frameworks.

**How to adapt here**
- Compute per-condition hypervolumes in feature space.
- Track overlap/Jaccard, centroid distance, and local density peaks.
- Visual outputs: overlap maps + decomposition bar charts.

**Effort**: Medium  
**Data readiness**: High

---

## 7) Topological Morphospace (Mapper / Persistent Homology)

**What it is**  
Topology-based graph/summary of the data manifold rather than Euclidean-only embedding.

**Why compelling**  
Can reveal branching/loops/multi-regime structures hidden in PCA/UMAP scatter.

**Literature anchors**
- Persistent homology in morphometrics (e.g., leaf morphospace).
- Mapper method surveys and biology applications.

**How to adapt here**
- Run Mapper on cluster-level feature matrix.
- Color nodes by speed, latency, embryo side, condition, zoom bin.
- Use as exploratory and hypothesis-generation panel.

**Effort**: High  
**Data readiness**: Medium

---

## 8) Morphospace Movies (Time-Scrubbed Embedding)

**What it is**  
Animated movement of points/centroids through a fixed morphospace as time advances.

**Why compelling**  
Very presentation-friendly; directly conveys dynamics and condition divergence.

**Literature anchors**
- Time-aware phylomorphospace and animation traditions (including phytools/phylowood ecosystems).

**How to adapt here**
- Keep embedding fixed on full dataset.
- Animate point alpha/size by event intensity and time bin.
- Optionally split by zoom-normalized groups.

**Effort**: Low-Medium  
**Data readiness**: High

---

## Best-Fit for This Repo (Ranked)

1. **Ontogenetic/response trajectories** (fast win, high narrative value)
2. **Disparity-through-time + occupancy stacks** (quantitative and publication-ready)
3. **Pareto/archetype fronts** (strong conceptual framing)
4. **Adaptive landscapes** (great visuals once scalar objective is agreed)
5. **Morphospace movies** (communication multiplier)
6. **Topological Mapper** (powerful but exploratory and heavier)

---

## Feature Set to Use (Current Data)

From `spark_tracks.csv` and `vector_clusters.csv`:
- kinematics: `mean_speed_px_per_s`, `peak_speed_px_per_s`, `net_speed_px_per_s`
- geometry: `ap_norm`, `dv_px`, `dist_from_poke_px`, `embryo_id`
- temporal: `time_s`, `duration_s`, early/late phase slopes
- intensity proxy: `area`, `total_area_px2_frames`
- units: `embryo_length_px`, speed in em/ecm/emm; optionally um/s with run scale
- metadata: zoom (`extract_tiff_metadata.py`) and run-level scale assumptions (`literature_benchmarks.py`)

Recommended normalized feature set for morphospace:
- speed (`speed_em_s` or `speed_um_s`)
- latency (phase-specific)
- normalized distance (`dist_from_poke_ecm`)
- AP-position moments (mean/variance)
- anisotropy (directional dispersion)
- integrated signal volume (normalized)

---

## Zoom/Scale Handling Requirements (Non-Negotiable)

To keep literature-quality comparability:
- never combine runs with unknown or mismatched scale without explicit normalization,
- attach `zoom` + `um_per_px` (or derivation fields) to every plotted point/run,
- prefer embryo-relative units for cross-zoom exploratory maps,
- provide a paired figure:
  - panel A: embryo-relative morphospace (ecm/emm),
  - panel B: physical-units morphospace (um/s, um) for runs with known scale.

---

## 4-Week Practical Roadmap

### Week 1: Foundation
- finalize feature schema + unit conversion columns for morphospace.
- build one clean feature table per run (with zoom/scale fields).

### Week 2: Quick wins
- implement trajectory morphospace and disparity-through-time stack plots.
- add morphospace movie export.

### Week 3: Explanatory models
- implement Pareto front/archetype visualization.
- prototype adaptive landscape surface over 2D embedding.

### Week 4: Advanced/exploratory
- add Mapper-based topological morphospace.
- compare topology by condition and zoom-normalized subsets.

---

## Suggested Figure Panel for a Paper/Talk

1. **Trajectory morphospace** (injury response phases)
2. **Disparity-through-time** (pre/post poke)
3. **Pareto/archetype front** (trade-off interpretation)
4. **Adaptive landscape** (performance peaks)
5. **Zoom-aware comparison inset** (scale assumptions)

This set gives both rigor and visual impact.


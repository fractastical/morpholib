# Ca²⁺ Wave Vector Analysis

This directory contains tools for detecting, tracking, and analyzing Ca²⁺ signaling waves in time-lapse microscopy images of embryos.

## Overview

The pipeline processes multi-page TIFF images to:
- Detect bright Ca²⁺ "spark" events (GCaMP fluorescence signals)
- Track individual events across frames
- Segment embryos and compute anatomical coordinates (AP/DV axes)
- Generate per-frame and per-cluster summaries for downstream analysis

This follows the methodology described in:
- Shannon et al. 2017 (Biophys J): wound-induced Ca²⁺ signal dynamics
- Tung et al. 2024 (Nat Commun): inter-embryo injury waves in Xenopus laevis

## Files

### Core Scripts

- **`wave-vector-tiff-parser.py`**: Main processing script that:
  - Reads TIFF files (supports multi-page, 16-bit images)
  - Detects and tracks Ca²⁺ sparks frame-by-frame
  - Automatically segments embryos and estimates head-tail axes
  - Detects poke/injury sites (optional)
  - Outputs `spark_tracks.csv` with per-frame measurements

- **`spark_tracks_to_clusters.py`**: Aggregates per-frame data into per-cluster summaries
  - Converts `spark_tracks.csv` → `vector_clusters.csv`
  - Computes speeds, path lengths, durations, directionality metrics

- **`visualize_spark_tracks.py`**: Visualization and analysis tools
  - Trajectory plots
  - Speed/direction distributions
  - Time series analysis
  - Spatial heatmaps
  - Per-embryo comparisons

- **`generate_all_hypothesis_plots.py`**: Generate comprehensive plots for all testable hypotheses
  - Activity analysis with standard deviation
  - Inter-embryo comparisons
  - AP position analysis
  - Speed vs time analysis
  - All plots mapped to specific experimental hypotheses

- **`generate_detection_summary.py`**: Manual-QA layer that renders per-video
  visual summaries (outlines, head/tail labels, poke locations) for verifying
  detection assumptions.

### Dense pixel vectors → waves

- **`generate_pixel_brightness_vectors.py`**: Frame-by-frame dense motion of
  every bright pixel (each pixel in frame *t* matched to *t−1* via KD-tree),
  on a linear brightness scale. Outputs `pixel_frame_deltas.csv`.
- **`rollup_pixel_vectors_to_waves.py`**: Clusters the dense pixel deltas into
  per-frame "fronts" and links them across frames into coherent WAVE events
  (`wave_events.csv`: origin, duration, propagation speed, front speed).
- **`batch_wave_catalog.py`**: Runs the two steps above on every calcium video
  and aggregates into a single `wave_catalog.csv` (tagged with stimulus /
  contact / orientation).
- **`wave_laterality_analysis.py`**: Splits the two embryos along the principal
  axis of their wave origins, labels the stimulated vs neighbor embryo, and
  scores neighbor-wave presence, neighbor local (front) speed, onset lag, and
  bidirectional spread per video → `wave_laterality.csv`.

### Mask generation

- **`create_embryo_masks.py`**, **`create_normalized_masks.py`**,
  **`create_size_constrained_masks.py`**: Generate / audit embryo segmentation
  masks for the parser.

### Shared helpers

- **`embryo_region_map.py`**: Maps detected coordinates into named anatomical
  regions via a normalized reference map.
- **`embryo_units.py`**: Converts distances/speeds from pixels into
  embryo-relative units.

### Documentation

- **`wave-vector.md`**: Detailed specification of output CSV formats
- **`SPARK_ANALYSIS_README.md`**: Usage guide for analysis scripts
- **`EXPERIMENTAL_HYPOTHESES.md`**: Mapping of experimental hypotheses to specific analyses
- **`PLOTTING_GUIDE.md`**: Complete guide to generating publication-quality plots
- **`TAIL_LABELING.md`**: Methodology for identifying and labeling tail regions

### Analysis Scripts

- **`analyze_experimental_hypotheses.py`**: Comprehensive analysis script for testing experimental claims
  - Calcium activity comparisons
  - Wave directionality analysis
  - Spatial matching
  - Tail response analysis
  - Condition comparisons (contact vs non-contact, etc.)

### Utilities

- **`diagnose_tiffs.py`**: Diagnostic tool to inspect TIFF file properties
- **`convert_tiffs_for_viewing.py`**: Convert 16-bit TIFFs to 8-bit for preview
- **`plot_poke_locations.py`**: Visualize all poke locations across processing runs

### Simulations

- **`simulations/`**: Simulation framework for generating synthetic Ca²⁺ wave data
  - Generate data with configurable embryos, poke locations, and wave parameters
  - Test hypotheses with controlled conditions
  - Compare simulated vs real data patterns
  - See `simulations/README.md` for details

### Output Data

Generated during processing (not included in repo if large):
- `spark_tracks.csv`: Per-frame Ca²⁺ event data
- `vector_clusters.csv`: Per-cluster summaries
- `plots/`: Visualization outputs

## Quick Start

### 1. Install Dependencies

```bash
pip install opencv-python numpy tifffile pandas matplotlib
```

### 2. Process TIFF Images

```bash
python wave-vector-tiff-parser.py /path/to/tiff/folder 100 \
    --fps 10 \
    --csv spark_tracks.csv \
    --out-video output.mp4
```

**Arguments:**
- `folder` (positional): Folder of TIFF frames, or a single multi-page TIFF
- `poke_frame` (positional): 0-based frame index where injury/poke occurs (t=0 reference)
- `--fps`: Frame rate (frames per second)
- `--csv`: Output CSV path (default `spark_tracks.csv`)
- `--poke-x`, `--poke-y`: Optional poke coordinates (if auto-detection fails)
- `--out-video`: Optional output video with overlays

### 3. Generate Cluster Summaries

```bash
python spark_tracks_to_clusters.py spark_tracks.csv
```

### 4. Create Visualizations

```bash
python visualize_spark_tracks.py spark_tracks.csv \
    --clusters-csv vector_clusters.csv \
    --output-dir plots/
```

## Features

- **16-bit image support**: Works directly with scientific imaging data
- **Multi-page TIFF support**: Handles time-lapse stacks efficiently
- **Automatic embryo detection**: Dynamic edge-based segmentation
- **Anatomical coordinates**: AP/DV position mapping relative to head-tail axis
- **Spark tracking**: Links detections across frames with gap-filling
- **Memory efficient**: Processes one frame at a time to handle large datasets

## Output Format

See `wave-vector.md` for complete documentation of CSV formats.

### `spark_tracks.csv`
Per-frame measurements with columns:
- `track_id`, `frame_idx`, `time_s`
- `x`, `y`: Position in pixels
- `vx`, `vy`, `speed`, `angle_deg`: Velocity and direction
- `area`: Spatial extent
- `embryo_id`: Which embryo (A/B)
- `ap_norm`, `dv_px`: Anatomical coordinates
- `dist_from_poke_px`: Distance from injury site

### `vector_clusters.csv`
Per-cluster summaries with columns:
- `cluster_id`, `n_frames`, `duration_s`
- `start_x_px`, `end_x_px`, `net_displacement_px`, `path_length_px`
- `mean_speed_px_per_s`, `peak_speed_px_per_s`
- `mean_angle_deg`, `angle_dispersion_deg`
- `total_area_px2_frames`: Integrated signal "volume"

## Example Workflow

```bash
# 1. Process images (folder, poke frame 50, 15 fps)
python wave-vector-tiff-parser.py ~/data/embryos 50 --fps 15

# 2. Generate summaries
python spark_tracks_to_clusters.py spark_tracks.csv

# 3. Visualize
python visualize_spark_tracks.py spark_tracks.csv \
    --clusters-csv vector_clusters.csv \
    --output-dir analysis_plots/

# 4. Analyze results
# Use pandas, R, or your preferred analysis tool on the CSV files
```

## Calcium Claims Validation

A second pipeline scores the verbatim claims in `Calcium claims.docx` against the
imaging data and assembles a single combined report,
`analysis_results/claims_all_in_one.pdf`.

### Scripts

- **`claims_doc_parser.py`**: Parses the claims table out of `Calcium claims.docx`
  (one record per claim with layer / pattern / orientation / poke location).
- **`parse_xy_coordinates.py`**: Parses the manual `XY coordinates.xlsx`
  ground-truth (all 39 sheets) into a tidy `analysis_results/xy_ground_truth.csv`
  (`prefix, video_stem, side, landmark, landmark_class, frame, x, y`). This is the
  authoritative poke / head-tail geometry and organ-response layer.
- **`score_landmark_responses.py`**: For every manually-annotated organ landmark
  (cement gland / eye / tail / local), measures the robust-peak ΔF/F₀ in a disk at
  the annotated pixel straight from the C-channel TIFF — an automated corroboration
  of the human annotation (`landmark_responses.csv` + `.png`). Also tags each
  landmark stimulated/neighbor using the real poke side.
- **`wave_laterality_analysis.py`**: Scores neighbor-side waves and bidirectional
  spread per video. When `xy_ground_truth.csv` is present it splits the two embryos
  with the **real poke + head/tail geometry** (assigns each wave origin to the
  nearest annotated embryo segment; the poke side is the stimulated embryo) and
  falls back to the PCA estimate only for single-embryo videos.
- **`generate_claim_wave_overlays.py`**: Per claim, overlays the relevant-side
  wave-front vectors on the brightest microscopy frame of the relevant videos
  (montage PNG + animated GIF per claim group), embedded into the inventory PDF.
- **`generate_tested_claims_summary.py`**: Builds the one-page **summary slide**
  (`claims_tested_summary.pdf`) — every tested claim group with its live verdict
  and a thumbnail graph.
- **`generate_claims_inventory_pdf.py`**: Full inventory of all verbatim claims
  grouped by Layer, each tagged TESTED / TESTABLE NOW / PARTIAL / NOT YET with a
  live verdict (`claims_inventory.pdf`), plus laterality evidence pages.
- **`generate_claims_nomask_report.py`**: All-pixel side-region ΔF/F₀ traces and
  per-claim verdicts (`claims_nomask_report.pdf`).
- **`generate_claim_video_mapping_pdf.py`**: Which Box videos map to each claim
  (`claim_video_mapping.pdf`).
- **`merge_claims_all_in_one.py`**: Merges the summary slide + component PDFs into
  `claims_all_in_one.pdf`.

### Run order

```bash
# 1. Build the wave catalog across all videos (dense vectors + roll-up)
python batch_wave_catalog.py --root "/path/to/Calcium videos"

# 1b. Parse the manual XY ground truth + score organ-response landmarks
python parse_xy_coordinates.py --xlsx "/path/to/XY coordinates.xlsx"
python score_landmark_responses.py

# 1c. Package a versioned Box deliverable (reports + tables + MANIFEST + sidecars)
python package_science_deliverables.py --include-gifs --include-detection-qa

# 2. Score laterality / bidirectionality (uses the real geometry if present)
python wave_laterality_analysis.py

# 2b. (optional) Render per-claim wave-vector overlays on the embryo frame
python generate_claim_wave_overlays.py

# 3. Build the summary slide, then the full claims inventory
python generate_tested_claims_summary.py
python generate_claims_inventory_pdf.py

# 3b. Export probabilistic KG (morphopkg/spc/1.0 TriG + JSON for Probknow)
python export_calcium_claims_to_pkg.py

# 4. (Re)generate the other component reports as needed, then merge
python merge_claims_all_in_one.py
```

The merge prefers `claims_tested_summary.pdf` as the first page and skips any
component PDF that is missing.

### Provenance / versioning

Major pipeline scripts write a sidecar ``<output>.provenance.json`` next to each
primary artifact (git commit, Python + package versions, timestamps, inputs).
CSV outputs also get a ``#`` comment header with the same summary. A rolling
``analysis_results/RUN_INDEX.jsonl`` logs every run. The combined PDF cover and
provenance page include the git commit; Box releases bundle all sidecars under
``04_provenance/pipeline_sidecars/``.

## Notes

- The script automatically detects embryos from the first frames where they're visible
- Poke detection can be automatic (pink arrow overlay) or manual (--poke-x/--poke-y)
- Supports recursive subdirectory traversal for organized data
- Memory-efficient processing handles large time-lapse series


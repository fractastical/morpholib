# AGENTS.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

## Repository shape

This repository is a collection of Python research scripts rather than a single packaged application. The top level contains standalone historical morphology demos and data-driven scripts; the two largest subprojects are:

- `wave-vector-analysis/`: the main Ca²⁺ imaging pipeline and its downstream analysis, visualization, mask-generation, and simulation tools
- `nanopubs/`: tools for generating, canonicalizing, signing, and publishing nanopublications

No additional `WARP.md`, `CLAUDE.md`, Cursor ruleset, or Copilot instructions file was present during review.

## Environment and common commands

### Base setup

The repo is Python-first and uses direct script execution.

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` covers the core scientific stack used by the main scripts. Some specialized workflows need extra packages:

- `pip install openpyxl` for `wave-vector-analysis/compare_xy_coordinates.py`
- `pip install nanopub rdflib requests` for `nanopubs/`

### General validation

There is no unified build, lint, or pytest configuration in the repository. For a quick syntax pass, use:

```bash
python3 -m compileall .
```

Most validation here is script-specific smoke testing.

### Top-level scripts

```bash
# PlanformDB timeline + morphology animation
export PLANFORM_DB_PATH=/path/to/planformDB_2.5.0.edb
python3 1900-planformDB_parser.py

# Or pass the database path explicitly
python3 1900-planformDB_parser.py /path/to/planformDB_2.5.0.edb

# Gray-Scott / Turing pattern animation
python3 1952-turing-morpho.py

# Raup shell parameter exploration
python3 1966-raup.py

# Generate the Raup animation and comparison assets
python3 create_raup_animations.py

# Thompson transformation demo
python3 thompson/1917-thompson.py
```

### Wave-vector analysis pipeline

The core workflow is parser → track CSV → cluster CSV → plots / summaries / hypothesis analysis.

```bash
# Parse a TIFF folder into per-frame spark tracks
python3 wave-vector-analysis/wave-vector-tiff-parser.py /path/to/tiff/folder 0 --fps 1.0 --csv spark_tracks.csv

# Override poke detection manually when needed
python3 wave-vector-analysis/wave-vector-tiff-parser.py /path/to/tiff/folder 0 --fps 1.0 --poke-x 123.5 --poke-y 456.7 --csv spark_tracks.csv

# Aggregate tracks into per-cluster summaries
python3 wave-vector-analysis/spark_tracks_to_clusters.py spark_tracks.csv vector_clusters.csv

# Generate overview plots
python3 wave-vector-analysis/visualize_spark_tracks.py spark_tracks.csv --clusters-csv vector_clusters.csv --output-dir wave-vector-analysis/analysis_plots

# Generate the hypothesis plot set used by the analysis docs
python3 wave-vector-analysis/generate_all_hypothesis_plots.py spark_tracks.csv --clusters-csv vector_clusters.csv --output-dir wave-vector-analysis/analysis_results

# Generate the manual QA / detection summary document
python3 wave-vector-analysis/generate_detection_summary.py spark_tracks.csv --output-dir wave-vector-analysis/analysis_results/detection_summary

# Diagnose unreadable or unusual TIFF stacks
python3 wave-vector-analysis/diagnose_tiffs.py /path/to/tiff/folder

# Check pink-arrow poke annotations for a specific folder/frame count
python3 wave-vector-analysis/check_pink_arrows.py /path/to/tiff/folder 10
```

Useful targeted runs:

```bash
# Scripted parser smoke test across many embryo folders
python3 wave-vector-analysis/test_cement_gland_detection.py

# Compare automatic head/tail detection against the manual Excel annotations
python3 wave-vector-analysis/compare_xy_coordinates.py /path/to/XY\ coordinates.xlsx spark_tracks.csv

# Batch rerun helper for the parser + downstream analyses
./wave-vector-analysis/rerun_full_analysis.sh
```

Mask-generation utilities:

```bash
# Generate size-constrained masks for one TIFF
python3 wave-vector-analysis/create_size_constrained_masks.py path/to/image.tif

# Batch-generate masks for a directory of TIFFs
python3 wave-vector-analysis/create_size_constrained_masks.py /path/to/tiffs --batch
```

Simulation workflow:

```bash
# Generate a predefined simulation scenario
python3 wave-vector-analysis/simulations/generate_simulated_data.py --scenario two_embryos_head_head --output sim_data.csv --duration 30.0

# Convert simulated output with the same downstream summarizer used for real data
python3 wave-vector-analysis/spark_tracks_to_clusters.py sim_data.csv sim_clusters.csv

# Visualize or compare simulated data
python3 wave-vector-analysis/simulations/visualize_simulation.py sim_data.csv --output simulation_plot.png

# Run all predefined simulation scenarios
./wave-vector-analysis/simulations/run_all_scenarios.sh
```

### Nanopublication workflow

```bash
# Generate Planform nanopubs from a local SQLite/EDB database without publishing
python3 nanopubs/planform_to_nanopubs.py --db /path/to/planformDB_2.5.0.edb --out ./nanopubs/planform_nanopubs

# Preview batch publication targets
./nanopubs/publish_batch.sh --dry-run

# Publish to the test registry first
./nanopubs/publish_batch.sh --test

# Use the Python entrypoint directly when needed
python3 nanopubs/publish_all_nanopubs.py --publish test --include waves/
```

Before publishing nanopubs, run the interactive profile setup once:

```bash
python3 -m nanopub setup --newkeys
```

## High-level architecture

### 1. Top-level scripts are independent research artifacts

The repository root is not a package; each script is an entrypoint for a specific historical model or dataset:

- `1900-planformDB_parser.py` reads a local PlanformDB SQLite/EDB database, derives yearly experiment/publication/morphology counts, extracts morphology/organ geometry, then renders both a static plot and an animated morphology timeline.
- `1952-turing-morpho.py`, `1966-raup.py`, `2021-Cervera–Levin–Mafe.py`, and `thompson/1917-thompson.py` are standalone demos that generate plots or animations directly.
- `create_raup_animations.py` is the richer Raup pipeline; it uses `datasets/raup/variables.json` and empirical shell metadata to build the repo’s presentation assets.

These scripts generally write output files into the current working tree and are intended to be run directly, not imported as a library.

### 2. `wave-vector-analysis/` is the main multi-stage data pipeline

This is the most operationally important part of the repo.

#### Core flow

1. `wave-vector-tiff-parser.py` reads multi-page TIFF stacks, detects bright spark events frame by frame, tracks them, tries to segment embryos, assigns head/tail and A/B identities, detects or accepts poke locations, and writes `spark_tracks.csv`.
2. `spark_tracks_to_clusters.py` groups per-frame track states into per-cluster summaries in `vector_clusters.csv`.
3. Downstream scripts consume one or both CSVs:
   - `visualize_spark_tracks.py`
   - `generate_all_hypothesis_plots.py`
   - `analyze_experimental_hypotheses.py`
   - `generate_detection_summary.py`
   - `plot_poke_locations.py`
   - `compare_xy_coordinates.py`

The schema of `spark_tracks.csv` is the repository’s central contract; multiple downstream tools and the simulation framework are built around it.

#### Shared helper logic

- `embryo_region_map.py` maps detected embryo coordinates into named anatomical regions via a normalized reference map and geometric transform.
- `embryo_units.py` converts distances and speeds from pixels into embryo-relative units.

Those helpers encode project-specific biological conventions, so preserve their column semantics if you modify the pipeline.

#### Supporting subflows

- Mask generation scripts (`create_embryo_masks.py`, `create_normalized_masks.py`, `create_size_constrained_masks.py`) exist to improve or audit embryo segmentation and to produce reusable mask assets.
- `generate_detection_summary.py` is the main manual-QA layer: it builds visual summaries for verifying outlines, head/tail labels, poke locations, and related assumptions.
- `simulations/` intentionally emits data in the same format as `spark_tracks.csv`, which lets you reuse the real-data clusterer and analysis scripts on simulated data without adapters.

### 3. `nanopubs/` is a separate Planform-oriented publishing toolchain

The nanopub code is its own workflow:

- `planform_to_nanopubs.py` converts PlanformDB `ResultantMorphology` rows into individual TriG nanopublications.
- `np_canonicalize_slash_uris.py`, `nanopub_restructure.py`, and `np_utils.py` handle file normalization and command-line utility tasks.
- `publish_all_nanopubs.py` and `publish_batch.sh` sign and publish batches, tracking publication state in manifest files.

By default, the batch publisher targets `waves/` and planform-generated files; `npi-waves/` is treated as an alternate/subset set and is not included unless explicitly requested.

## Non-obvious repository constraints

- `1900-planformDB_parser.py` expects a real local PlanformDB file, either through `PLANFORM_DB_PATH` or a positional CLI argument. Without it, the script exits.
- The wave parser’s embryo identity and head/tail assignment are nontrivial and have dedicated reasoning docs in `wave-vector-analysis/HEAD_TAIL_ASSIGNMENT_LOGIC.md`; changes here can ripple into AP coordinates, region labels, poke distances, and all downstream analyses.
- Manual XY annotations are not currently reliable ground truth without transformation. `COORDINATE_SYSTEM_WARNING.md` documents that the Excel coordinates can exceed TIFF image bounds, so treat `compare_xy_coordinates.py` results cautiously until the coordinate-space mismatch is resolved.
- Several wave-analysis docs and shell wrappers assume a repo-root execution context and filenames like `spark_tracks.csv` / `vector_clusters.csv`. If you change output locations, update downstream commands accordingly.
- `nanopubs/publish_batch.sh` will try to activate `../venv` automatically if it exists; publishing is expected to happen only after nanopub profile setup and usually should be tested against the test registry before production.

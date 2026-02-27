# Speed calculations and comparison to literature

**Comparisons and zoom handling:** Use the dedicated module and output file so zoom differentials are explicit and not mixed. See [literature_benchmarks.py](literature_benchmarks.py): it stores literature benchmarks, converts our speeds to μm/s using per-run scale (or embryo length in μm), and writes results to a **separate comparison file** (e.g. `analysis_results/literature_comparison.json`). Each run records its own zoom and `um_per_px` so different TIFF zooms are compared fairly.

## What we compute

- **Speed** in the pipeline is **pixels per second (px/s)**: from frame-to-frame displacement of spark centroids, `speed = hypot(vx, vy)` with `vx, vy` in px/s (see `wave-vector-tiff-parser.py`).
- **Distances** (e.g. `dist_from_poke_px`, `dv_px`) are in **pixels**.
- **Reference embryo length** is written as `embryo_length_px` in `spark_tracks.csv` (head–tail in pixels per run).

So we do **not** currently output speeds in μm/s unless we have a scale (μm per pixel).

## Literature benchmarks (from your reference figures)

These are propagation speeds and distances reported in the figures (μm/s and μm):

| Scenario | Speed (μm/s) | Distance (μm) | Notes |
|----------|--------------|---------------|--------|
| Head–head (neighbor) | 97.64 – 130.19 | 1952.824 | Local response neighbor |
| Tail–head | 10.1 – 10.4 | 1656.845 | Latency 0–5 s + 159 s |
| Head–head (other) | 22.5 – 23.7 | 2203.296 | Latency 0–5 s + 93 s |
| Single embryo (short) | 194.4 – 972.2 | 972.16 (length) | Time 0–5 s |
| Single embryo (long) | 364.3 – 1821.4 | 1821.44 (length) | Time 0–5 s |
| Stimulus → tail in neighbor | 634.9 – 3174.5 | — | — |
| Stimulus → anterior in neighbor | 126.6 – 443.0 | — | — |
| Stimulus → tail in neighbor (2) | 576.6 – 2883.0 | — | — |
| Within embryo: stimulus → tail | 331.1 – 1655.5 | — | — |
| Stimulus and response neighbor | 305.0 – 813.1 | — | — |

So reported speeds range from ~10 μm/s (tail–head) up to ~3000 μm/s in some neighbor propagation; single-embryo “stimulus to tail” is on the order of hundreds to ~1800 μm/s depending on length and time window.

## How to compare

### Option 1: Compare in embryo units (no μm scale needed)

- Convert **our** speeds to **em/s** (embryo lengths per second) using `embryo_length_px`:
  - `speed_em_s = speed_px_s / embryo_length_px`
- Convert **literature** speed to the same units using their stated distance/length:
  - e.g. “97.64 μm/s over 1952.824 μm” → 97.64 / 1952.824 ≈ **0.05 em/s** (if we take that distance as 1 em).
- Then compare our `speed_em_s` (or `speed_ecm_s`, `speed_emm_s`) to these em/s values.

Use the helpers in `embryo_units.py`:

```python
from embryo_units import add_embryo_unit_columns, speed_px_s_to_em_s

# Add speed_em_s, speed_ecm_s, speed_emm_s to spark_tracks (uses embryo_length_px)
df = add_embryo_unit_columns(df, length_column="embryo_length_px")

# Or single conversion
speed_em_s = speed_px_s_to_em_s(speed_px_s=50, reference_length_px=1200)  # ~0.042 em/s
```

### Option 2: Compare in μm/s (need μm per pixel)

- Get **μm per pixel** from:
  - TIFF metadata: e.g. ImageDescription `Unit = µm` and a scale or pixel size, or
  - Assumed embryo length in μm: e.g. if embryo length ≈ 1000 μm and `embryo_length_px` = 1200, then **um_per_px = 1000 / 1200 ≈ 0.833 μm/px**.
- Convert our speed: **speed_μm_s = speed_px_s × um_per_px**.
- Compare directly to the ranges in the table above.

Use:

```python
from embryo_units import speed_px_s_to_um_s

# Example: 100 px/s, 0.833 μm/px → 83.3 μm/s
speed_um_s = speed_px_s_to_um_s(100, um_per_px=0.833)
```

If you use a fixed reference embryo length in μm (e.g. 1000 μm) for all runs:

```python
um_per_px = 1000.0 / df["embryo_length_px"]
df["speed_um_s"] = df.apply(lambda r: speed_px_s_to_um_s(r["speed"], um_per_px[r.name]), axis=1)
```

(Or vectorize with a constant `um_per_px` if you use one reference length.)

## Separate comparison file (zoom differentials)

Comparisons live in a **dedicated file** so zoom/scale are never mixed across runs:

- **Module:** [literature_benchmarks.py](literature_benchmarks.py) — defines `LITERATURE_BENCHMARKS`, `compare_speed_to_benchmarks()`, and `run_comparison()`.
- **Output:** One JSON per run (e.g. `analysis_results/literature_comparison.json`) with:
  - `run_id`, `zoom`, `embryo_length_px`, `embryo_length_um`, `um_per_px`
  - `our_mean_speed_px_s`, `our_mean_speed_um_s`, etc.
  - `benchmark_comparisons`: for each literature scenario, whether our mean speed falls in range and distance to range.

**CLI (one run):**

```bash
python literature_benchmarks.py path/to/spark_tracks.csv --output analysis_results/literature_comparison.json --embryo-length-um 1000
python literature_benchmarks.py path/to/spark_tracks.csv -o analysis_results/run1_comparison.json --embryo-length-um 1000 --tiff-metadata path/to/metadata.json --run-id "1/B - Substack"
```

**Multiple runs (different zooms):** Run the script once per `spark_tracks.csv` (or per folder), each with its own `--output` and optional `--tiff-metadata` / `--zoom`. Then use `load_all_comparisons(comparison_dir)` or `write_comparisons_table(comparison_paths, "comparison_table.csv")` to get one table with all runs and their zoom/scale.

## Summary

- **Current state:** We have speed in **px/s** and distances in **px**, and we have **embryo_length_px** and embryo-unit conversions (em, ecm, emm and em/s, ecm/s, emm/s).
- **To match the figures:** Either (1) express everything in **embryo units** and convert the literature values to em/s using their stated distances, or (2) define **μm/px** (from metadata or assumed embryo length in μm) and convert our speeds to **μm/s** for direct comparison to the numbers in the images.
- **Zoom differentials:** Use [literature_benchmarks.py](literature_benchmarks.py) and a separate comparison file per run; each file records that run’s zoom and scale so comparisons are not mixed across different magnifications.

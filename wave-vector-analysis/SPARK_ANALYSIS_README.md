# Spark Tracks Analysis Tools

This directory contains tools for analyzing and visualizing Ca²⁺ wave data from the `wave-vector-tiff-parser.py` pipeline.

## Files

### 1. `spark_tracks_to_clusters.py`
Converts per-frame `spark_tracks.csv` into per-cluster summaries (`vector_clusters.csv`).

**Usage:**
```bash
python spark_tracks_to_clusters.py spark_tracks.csv [vector_clusters.csv]
```

**Output:** `vector_clusters.csv` with one row per track, containing:
- Duration and frame counts
- Start/end positions
- Net displacement and path length
- Speed statistics (mean, peak, net)
- Direction statistics (mean angle, dispersion)
- Area statistics
- Distance from poke site (if available)

### 2. `visualize_spark_tracks.py`
Creates various visualizations of the spark track data.

**Usage:**
```bash
# Show all plots interactively
python visualize_spark_tracks.py spark_tracks.csv

# Save all plots to a directory
python visualize_spark_tracks.py spark_tracks.csv --output-dir plots/

# Generate specific plots
python visualize_spark_tracks.py spark_tracks.csv --plot trajectories
python visualize_spark_tracks.py spark_tracks.csv --plot time
python visualize_spark_tracks.py spark_tracks.csv --plot heatmap
python visualize_spark_tracks.py spark_tracks.csv --plot embryo

# With clusters data for speed analysis
python visualize_spark_tracks.py spark_tracks.csv --clusters-csv vector_clusters.csv --plot speed
```

**Available plots:**
- `trajectories`: Overlay of all spark trajectories (start=green, end=red)
- `speed`: Distribution of propagation speeds (requires clusters CSV)
- `time`: Time series of active tracks and integrated signal
- `heatmap`: Spatial density map of Ca²⁺ events
- `embryo`: Comparison of dynamics between embryos (if embryo_id available)

## Workflow

1. **Generate tracks CSV** (if not already done):
   ```bash
   python wave-vector-tiff-parser.py /path/to/tiffs --poke-frame 100 --fps 10
   ```

2. **Generate clusters CSV**:
   ```bash
   python spark_tracks_to_clusters.py spark_tracks.csv
   ```

3. **Create visualizations**:
   ```bash
   python visualize_spark_tracks.py spark_tracks.csv --clusters-csv vector_clusters.csv --output-dir analysis_plots/
   ```

4. **Dense pixel frame deltas** (each bright pixel in frame *t* matched to frame *t−1*):
   ```bash
   python generate_pixel_brightness_vectors.py /path/to/video.tif 0 --fps 1.0 \
     --output-dir analysis_results/pixel_vectors --arrow-scale 12 --pdf --mp4
   ```
   Accepts a folder of TIFFs **or** a single multi-page TIFF. Outputs
   `pixel_frame_deltas.csv` (`x_last`→`x_curr`, `delta_x_px`, `speed_px_per_s`,
   brightness delta), per-frame PNGs, PDF/MP4, and `pixel_speed_summary.png`
   (speed distribution + mean speed over time). `--arrow-scale` magnifies the
   tiny 1-px vectors so they are visible.

5. **Roll dense vectors up into WAVE events** (cluster fronts per frame, link
   them across frames):
   ```bash
   python rollup_pixel_vectors_to_waves.py \
     analysis_results/pixel_vectors/pixel_frame_deltas.csv \
     --img-height 936 --img-width 2960
   ```
   Outputs a `waves/` subfolder with `wave_events.csv` (one row per wave:
   origin, duration, `net_direction_deg`, `propagation_speed_px_per_s` =
   centroid motion, `mean_front_speed_px_per_s` = per-pixel motion, peak
   brightness), `wave_front_frames.csv` (per-wave per-frame front),
   `wave_tracks.png` (centroid paths), and `wave_summary.png`.

6. **Score laterality + directionality across the catalog** (which embryo a
   wave belongs to, and whether spread is bidirectional):
   ```bash
   python wave_laterality_analysis.py \
     --catalog-dir analysis_results/wave_catalog
   ```
   For every per-video `waves/wave_events.csv` it fits the principal axis of
   the wave **origins**, splits the two embryos along it (1-D 2-means), labels
   the earliest/brightest cluster the *stimulated* embryo and the other the
   *neighbor*, then writes `wave_laterality.csv` + `wave_laterality.png` with:
   - `neighbor_has_wave`, `n_waves_neighbor`, `neighbor_mean_front_speed`
     (does a wave / local response appear in the neighbor?),
   - `onset_lag_s` (neighbor first wave − stimulated first wave),
   - `bidirectional_stim` (do stimulated-side waves spread BOTH ways along the
     embryo axis?).
   These metrics back the TESTED verdicts in `claims_inventory.pdf`
   (Layer 2 bidirectional claims; Layer 4 neighbor-wave / local-response
   claims).

## Dependencies

- `pandas` - Data manipulation
- `numpy` - Numerical operations
- `matplotlib` - Plotting

Install with:
```bash
pip install pandas numpy matplotlib
```

## Data Format

See `wave-vector.md` for detailed documentation of the CSV formats.


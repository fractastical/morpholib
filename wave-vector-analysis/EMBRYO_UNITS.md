# Embryo units (em, ecm, emm)

We standardize measurements by **picking one embryo length and calling that 1**, then expressing all distances relative to that in **embryo meters (em)**, **embryo centimeters (ecm)**, and **embryo millimeters (emm)**.

## Convention

- **1 em** (embryo meter) = one reference embryo length (e.g. head–tail distance).
- **1 ecm** (embryo centimeter) = 0.01 em = 1/100 embryo length.
- **1 emm** (embryo millimeter) = 0.001 em = 1/1000 embryo length.

So:

- 1 em = 100 ecm = 1000 emm  
- 1 ecm = 10 emm  

## Reference length

The reference embryo length is typically the **head–tail distance in pixels** for the video or embryo you are analyzing:

- **Per video:** Use the standard head–tail length from the parser (mean of both embryos in that video). The parser writes this as `embryo_length_px` in `spark_tracks.csv` when available.
- **Per embryo:** Use that embryo’s head–tail distance in pixels as the reference for its events.
- **Global:** Pick one canonical length (e.g. from a reference video) and use it for all datasets.

## Conversion

Given a distance in pixels and the reference length in pixels:

- **em** = `pixels / reference_length_px`
- **ecm** = `em * 100`
- **emm** = `em * 1000`

Example: reference length = 1200 px, distance = 60 px → 60/1200 = 0.05 em = 5 ecm = 50 emm.

## Usage in code

```python
from embryo_units import px_to_emm, px_to_ecm, px_to_em, add_embryo_unit_columns

# Single conversion
reference_length_px = 1200
dist_emm = px_to_emm(60, reference_length_px)   # 50 emm
dist_ecm = px_to_ecm(60, reference_length_px)   # 5 ecm

# Add ecm/emm columns to a DataFrame (uses embryo_length_px column if present)
df = add_embryo_unit_columns(df, reference_length_px=1200)
# or, if df has embryo_length_px per row:
df = add_embryo_unit_columns(df, length_column="embryo_length_px")
```

## Where the reference comes from

- **wave-vector-tiff-parser:** For each run it computes a standard head–tail length (mean over embryos in that video) and writes it as `embryo_length_px` in `spark_tracks.csv`. Use that column as the reference for that file’s rows.
- **generate_detection_summary / TIFF detection:** Head–tail distance in pixels per embryo is available from detection; use it as reference for that embryo’s measurements.

Reporting distances in ecm or emm makes values comparable across embryos and experiments (e.g. “5 ecm from poke” means 5% of embryo length).

# Data Sources and Assumptions

This document summarizes the sources used in the current detection run and the unit/zoom assumptions.

## Data Sources

- Tracks source: `spark_tracks.csv-derived detections`
- TIFF base path: `/Users/jdietz/Documents/Levin/Embryos`
- OLD mask source: `/Users/jdietz/Documents/GitHub/infinitemorphospace/wave-vector-analysis/embryo_masks_final`
- NEW mask source: `/Users/jdietz/Documents/GitHub/infinitemorphospace/datasets/frog_embryo_data_processed`
- Excel coordinates: `loaded`

## Coverage Summary

- Videos in PDF: **74**
- TIFF files found: **74**
- OLD masks found: **59**
- NEW masks found: **50**
- Both old+new masks found: **45**
- TIFFs with usable um/px calibration: **0**
- NEW masks with parseable zoom metadata: **48**

## Assumptions

- Coordinates are rendered in TIFF pixel space with matplotlib origin at bottom-left.
- Legacy PNG masks are treated as OLD masks; NPZ masks from frog dataset are treated as NEW masks.
- If mask dimensions differ from TIFF dimensions, masks are resized with nearest-neighbor interpolation.
- Speed labels prefer um/s when TIFF ImageDescription contains calibration (Resolution/Pixel width/Voxel size); otherwise px/s.
- Zoom values are parsed from NEW mask folder names like '12 - Zoom = 5.40x' and used as metadata labels.
- Excel coordinates are transformed into current TIFF plotting frame (scaling and y-orientation selection) before overlay.

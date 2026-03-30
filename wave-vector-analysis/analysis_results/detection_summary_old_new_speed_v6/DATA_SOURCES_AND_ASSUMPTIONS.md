# Data Sources and Assumptions

This document summarizes the sources used in the current detection run and the unit/zoom assumptions.

## Data Sources

- Tracks source: `spark_tracks.csv-derived detections`
- TIFF base path: `/Users/jdietz/Documents/Levin/Embryos`
- OLD mask source: `/Users/jdietz/Documents/GitHub/infinitemorphospace/wave-vector-analysis/embryo_masks_final`
- NEW mask source: `/Users/jdietz/Documents/GitHub/infinitemorphospace/datasets/frog_embryo_data_processed`
- Excel coordinates source: `/Users/jdietz/Documents/Levin/Embryos/XY coordinates.xlsx`
- Excel loaded: **True**
- Excel parsed folders: **29**
- Excel parsed folder/video mappings: **29**

## Coverage Summary

- Videos in PDF: **74**
- TIFF files found: **74**
- OLD masks found: **59**
- NEW masks found: **50**
- Both old+new masks found: **45**
- TIFFs with usable um/px calibration: **74**
- NEW masks with parseable zoom metadata: **48**

## Metadata Found in TIFFs

- IJMetadata: **74**
- IJMetadata Zoom: **74**
- IJMetadata dCalibration: **74**
- ImageDescription: **74**
- ImageDescription unit=micron: **74**
- ResolutionUnit: **74**
- XResolution: **74**
- YResolution: **74**

## Calibration Source Breakdown

- ijmetadata:dCalibration: **74**

## Assumptions

- Coordinates are rendered in TIFF pixel space with matplotlib origin at bottom-left.
- Legacy PNG masks are treated as OLD masks; NPZ masks from frog dataset are treated as NEW masks.
- If mask dimensions differ from TIFF dimensions, masks are resized with nearest-neighbor interpolation.
- Speed labels prefer um/s when TIFF ImageDescription contains calibration (Resolution/Pixel width/Voxel size); otherwise px/s.
- Zoom values are parsed from NEW mask folder names like '12 - Zoom = 5.40x' and used as metadata labels.
- Excel coordinates are transformed into current TIFF plotting frame (scaling and y-orientation selection) before overlay.

## Excel Spreadsheet Mapping (How data is read)

- Workbook layout: each sheet is treated as one folder (sheet name -> folder id).
- Header detection: parser first inspects row 1 values for 'ID', 'X', 'Y'; if not found, falls back to column names containing id/X/Y.
- Video selection: parser uses first non-empty value in the first column (excluding head/tail/poke labels) as video key; fallback is first column header.
- Row parsing: ID cells containing 'poke' are interpreted as poke coordinates; IDs like 'A_head', 'A_tail', 'B_head', 'B_tail' map embryo landmarks.
- Missing/NaN X/Y values are skipped.
- At visualization time, video matching prefers exact normalized name, then fuzzy match with threshold; low-confidence matches are skipped.

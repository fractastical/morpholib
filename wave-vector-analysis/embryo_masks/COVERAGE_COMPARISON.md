# Embryo Mask Coverage Comparison

## Summary

This document compares three different methods for creating embryo masks:

1. **Current Parser Method**: Uses `max(median * 0.7, p25, mean * 0.8, background * 1.2)`
2. **Permissive Multi Method**: Uses union of percentile (5th), adaptive, and Otsu thresholds
3. **Size-Constrained Method**: Uses wider thresholding but filters by expected embryo size (0.5% - 15% of image)

## Results

### Current Parser Method
- **Average coverage**: 30.53%
- **Median coverage**: 30.06%
- **Range**: 0.30% - 67.76%
- **Issue**: Too restrictive, misses dimmer embryo regions

### Permissive Multi Method  
- **Average coverage**: 97.23%
- **Median coverage**: 96.86%
- **Range**: 95.36% - 100.00%
- **Issue**: Too permissive, includes background

### Size-Constrained Method (Recommended)
- **Average coverage**: 6.79%
- **Median coverage**: 2.03%
- **Range**: 0.00% - 27.41%
- **Advantage**: Filters by expected embryo size, excludes background and noise

## Key Findings

1. **Current parser captures ~30%** but misses dimmer regions
2. **Permissive method captures ~97%** but includes too much background
3. **Size-constrained method captures ~2-7%** which is more realistic for individual embryos

The size-constrained method uses:
- Wider greyscale range (5th percentile) to capture dimmer regions
- Size filtering (0.5% - 15% of image) to exclude background and noise
- This ensures we capture the full embryo body without including background

## Recommendations

For training purposes, use the **size-constrained masks** as they:
- Capture the full embryo body (including dimmer regions)
- Exclude background pixels
- Filter out noise and artifacts
- Provide consistent embryo-sized regions

Files are saved in:
- `wave-vector-analysis/embryo_masks_size_constrained/`
- Summary CSV: `size_constrained_coverage_summary.csv`

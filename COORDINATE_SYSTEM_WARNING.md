# ⚠️ Coordinate System Mismatch Warning

## Problem Identified

**The pixel coordinates in `XY coordinates.xlsx` do NOT match the TIFF video coordinate system.**

## Evidence

### 1. Coordinate Range Mismatch

- **Excel coordinates:** X values range from 0 to **3604.7 pixels**
- **TIFF image dimensions:** Typical width is **2220-2960 pixels**
- **Example (Folder 1):**
  - Excel B_tail: X = **3604.7 pixels**
  - TIFF image width: **2220-2265 pixels**
  - **Excel coordinate exceeds image width by ~1400 pixels!**

### 2. Specific Examples

From Folder 1 analysis:
- Excel coordinates include values like:
  - `B_tail: (3604.7, 513.5)` - **X exceeds image width**
  - `B_tail: (3514.5, 353.9)` - **X exceeds image width**
  - `B_tail: (3587.4, 523.9)` - **X exceeds image width**
- But TIFF images in folder 1 are only **2220-2265 pixels wide**

### 3. Comparison Report Shows Large Differences

From `head_tail_comparison.md`:
- Head distance differences: **564.4 ± 342.0 pixels** (mean)
- Tail distance differences: **1441.8 ± 426.4 pixels** (mean)
- These large differences suggest coordinate system mismatch

## Possible Causes

1. **Different Image Source:**
   - Excel coordinates might be from a different image (composite, stitched, or different zoom level)
   - Could be from a different imaging session or setup

2. **Coordinate System Transformation:**
   - Excel coordinates might be in a different coordinate space (e.g., microscope stage coordinates, different ROI)
   - Could be from a different software that uses different origin or scaling

3. **Image Cropping/ROI:**
   - TIFF videos might be cropped versions of larger images
   - Excel coordinates might be from the full uncropped image

4. **Multiple Time Points:**
   - Excel file contains coordinates from multiple slices/frames
   - Different frames might have different coordinate systems

## Impact

- **Comparison reports are unreliable** - large differences don't necessarily mean detection errors
- **Visual overlays may be incorrect** - Excel coordinates plotted on TIFF images will be off-screen
- **Validation is compromised** - cannot use Excel coordinates to validate detection accuracy

## Recommendations

### 1. Verify Coordinate Source

**Question to answer:** Where did the Excel coordinates come from?

- What software was used to annotate?
- What image file was used for annotation?
- Are the coordinates from the same TIFF files we're analyzing?

### 2. Check for Transformation

If Excel coordinates are from a different source, we need to:

- **Find the transformation** between Excel coordinate system and TIFF coordinate system
- **Apply transformation** before comparison
- **Validate transformation** using known landmarks

### 3. Re-annotate if Needed

If coordinates are from wrong source:

- Re-annotate using the actual TIFF files being analyzed
- Use ImageJ, Fiji, or similar tool that can read TIFF coordinates directly
- Ensure coordinates are in pixel space of the TIFF files

### 4. Temporary Workaround

Until coordinate system is resolved:

- **Do not trust comparison metrics** - large differences may be due to coordinate mismatch, not detection errors
- **Use visual inspection** - manually check if detected positions look correct on the images
- **Focus on relative positions** - check if head/tail relationships are correct, even if absolute positions differ

## How to Verify

### Step 1: Check a Specific Example

1. Open a TIFF file (e.g., `1/B - Substack (1-301).tif`)
2. Note its dimensions (e.g., 2220 x 600 pixels)
3. Check Excel coordinates for that folder/video
4. See if Excel X coordinates exceed image width

### Step 2: Visual Check

1. Load the TIFF image
2. Plot Excel coordinates on the image
3. Check if coordinates fall within image bounds
4. Check if coordinates align with visible embryo features

### Step 3: Compare with Detection Visualizations

1. Look at `detection_visualizations.pdf`
2. Check if detected head/tail positions look reasonable
3. Compare with Excel coordinates (if plotted)
4. Note any systematic offsets

## Next Steps

1. **Identify coordinate source** - Determine where Excel coordinates came from
2. **Find transformation** - If from different source, find mapping to TIFF coordinates
3. **Update comparison script** - Apply transformation before comparing
4. **Re-validate** - Re-run comparison with corrected coordinates

## Files Affected

- `XY coordinates.xlsx` - Contains coordinates in unknown coordinate system
- `compare_xy_coordinates.py` - Comparison script (assumes same coordinate system)
- `head_tail_comparison.md` - Comparison report (may be unreliable)
- `generate_detection_summary.py` - May plot Excel coordinates incorrectly

## Technical Details

### Current Coordinate Systems

1. **TIFF Images:**
   - Origin: Top-left (0, 0)
   - Units: Pixels
   - Range: 0 to image_width (typically 2220-2960), 0 to image_height (typically 600-900)

2. **Spark Tracks (detected):**
   - Origin: Same as TIFF (top-left)
   - Units: Pixels
   - Range: Matches TIFF dimensions

3. **Excel Coordinates (manual):**
   - Origin: Unknown
   - Units: Appears to be pixels, but different scale
   - Range: 0 to 3604+ (exceeds TIFF width)

### Transformation Needed

If Excel coordinates are from a different source, we need:

```
TIFF_x = f(Excel_x, Excel_y)
TIFF_y = g(Excel_x, Excel_y)
```

Where `f` and `g` are transformation functions (could be scaling, translation, rotation, or more complex).


# XY Coordinates.xlsx Documentation

## Overview

The `XY coordinates.xlsx` file contains manually annotated head and tail coordinates for planarian embryos across multiple experimental folders. This file is used to validate and compare against automatically detected head/tail positions from spark track analysis.

**File Location:** `/Users/jdietz/Documents/Levin/Embryos/XY coordinates.xlsx`

## Structure

### Sheet Organization

- **Total Sheets:** 29 sheets
- **Sheet Names:** Numeric identifiers corresponding to folder numbers: `1`, `3`, `9`, `10`, `11`, `12`, `13`, `14`, `15`, `16`, `17`, `19`, `20`, `21`, `22`, `24`, `25`, `26`, `27`, `30`, `31`, `32`, `33`, `34`, `35`, `36`, `37`, `38`, `39`
- **Matching:** Sheet names match folder numbers in the embryo data directory structure

### Column Structure

Each sheet contains the following columns:

| Column | Description | Data Type |
|--------|-------------|-----------|
| **ID** | Identifier for the measurement point | String |
| **Mean** | Mean intensity value at the coordinate | Numeric |
| **X** | X-coordinate (pixel position) | Numeric |
| **Y** | Y-coordinate (pixel position) | Numeric |
| **Slice** | Frame/slice number in the TIFF stack | Integer |
| **Unnamed: 5** | Empty column (reserved) | - |
| **Unnamed: 6** | Notes/annotations column | String |

### Header Row

- Header row is typically at **row index 1** (second row)
- Some sheets may have multiple header rows or repeated headers within the data

## ID Value Types

The `ID` column contains various identifiers:

### Primary Embryo Labels

1. **Two-Embryo Format:**
   - `A_head` - Head of left embryo (embryo A)
   - `A_tail` - Tail of left embryo (embryo A)
   - `B_head` - Head of right embryo (embryo B)
   - `B_tail` - Tail of right embryo (embryo B)

2. **Single-Embryo Format:**
   - `Head` - Head of single embryo
   - `Tail` - Tail of single embryo
   - Note: Some sheets use lowercase `head`/`tail` or `Head`/`Tail`

### Additional Labels

- `C-substack` - Video/substack identifier (metadata, no coordinates)
- `B10-substack` - Alternative video identifier
- `B15-substack` - Alternative video identifier
- `Poke location` - Location where embryo was poked/stimulated
- `Poke_right side` - Right side poke location
- `Poke_left side` - Left side poke location

### Response Labels

Some sheets include response measurements:

- `A_tail response` - Response at tail of embryo A
- `B_tail response` - Response at tail of embryo B
- `A_anterior response` - Anterior response of embryo A
- `A_mid response` - Mid-body response of embryo A
- `A_belly response` - Belly response of embryo A
- `B_anterior response` - Anterior response of embryo B
- `B_belly response` - Belly response of embryo B
- `A_nose response` - Nose response of embryo A
- `A_cement gland response` - Cement gland response of embryo A
- `Cement gland response` - General cement gland response
- `Tail response` - General tail response
- `A_pre-wound location` - Pre-wound location marker

### Notes/Comments

Some sheets contain notes in the `Unnamed: 6` column:
- `A = left embryo` / `B = right embryo` - Embryo position labels
- Questions/notes about coordinate matching (e.g., "* Does the poke location match the XY of A_mid response?")
- Special instructions (e.g., "* this one has two poke locations, need to repeat with just one")

## Data Patterns

### Typical Data Structure

Most sheets follow this pattern:

1. **Initial measurements** (Slice 1):
   - Head and tail coordinates for each embryo
   - May include multiple time points (Slice 1, Slice 301, etc.)

2. **Repeated headers:**
   - Some sheets have repeated header rows (`ID`, `Mean`, `X`, `Y`, `Slice`) within the data
   - This indicates multiple measurement sets or time points

3. **Metadata rows:**
   - Video identifiers (`C-substack`, `B10-substack`, etc.)
   - Notes about embryo positions
   - Poke locations and response measurements

### Coordinate Ranges

Based on the data:
- **X coordinates:** Typically range from 0 to ~5000 pixels
- **Y coordinates:** Typically range from 0 to ~1400 pixels
- **Mean intensity:** Typically ranges from 300-800 (varies by imaging conditions)

### Slice Numbers

- Most measurements are at **Slice 1** (first frame)
- Some sheets include measurements at **Slice 301** (last frame for 301-frame videos)
- Sheet 13 includes measurements at **Slice 800** (longer video)

## Usage in Analysis

### Matching with Detected Coordinates

The `compare_xy_coordinates.py` script uses this file to:

1. **Parse manual coordinates** from each sheet (matched by folder number)
2. **Extract head/tail positions** for embryos A and B
3. **Match with detected coordinates** from `spark_tracks.csv`
4. **Calculate differences** between manual and automatic detections
5. **Generate comparison reports** showing accuracy metrics

### Matching Logic

- **Folder matching:** Sheet name (e.g., "1", "3", "9") matches folder number
- **Video matching:** Video names are normalized and matched flexibly:
  - Case-insensitive
  - Extensions removed
  - Parentheticals removed
  - Whitespace normalized
- **Embryo matching:** Uses `A_head`, `A_tail`, `B_head`, `B_tail` or `Head`, `Tail` labels

## Data Quality Notes

### Inconsistencies

1. **ID Format Variations:**
   - Some sheets use `Head`/`Tail` (single embryo)
   - Others use `A_head`/`A_tail`/`B_head`/`B_tail` (two embryos)
   - Case variations: `head` vs `Head`, `tail` vs `Tail`

2. **Missing Data:**
   - Some sheets have incomplete coordinate sets
   - Blank rows are common (filtered during parsing)

3. **Multiple Measurements:**
   - Some sheets contain measurements at multiple time points (Slice 1, Slice 301, etc.)
   - The parser typically uses the first valid measurement found

4. **Special Cases:**
   - Sheet 14: Note about two poke locations
   - Sheet 20, 21, 22: Questions about coordinate matching
   - Sheet 25, 26, 27: Questions about spatial location matching

## Example Data Entry

```
ID          Mean    X          Y         Slice
A_head      385     1793.675   568.98    1
A_tail      413     156.123    707.756   1
B_head      380     1805.818   544.694   1
B_tail      416     3471.126   374.694   1
```

## Related Files

- **`compare_xy_coordinates.py`** - Script that parses and compares this data
- **`spark_tracks.csv`** - Contains automatically detected coordinates
- **`head_tail_comparison.md`** - Generated comparison report
- **`head_tail_comparison.csv`** - Generated comparison data

## Notes for Future Use

1. **Consistency:** When adding new data, use consistent ID formats:
   - For two embryos: `A_head`, `A_tail`, `B_head`, `B_tail`
   - For single embryo: `Head`, `Tail`

2. **Video Names:** Include video identifiers (like `C-substack`) to help with matching

3. **Multiple Time Points:** If measuring multiple frames, consider using separate rows with different Slice values

4. **Validation:** The comparison script can help identify mismatches between manual and automatic detections


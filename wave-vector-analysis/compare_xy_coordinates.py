#!/usr/bin/env python3
"""
Parse XY coordinates.xlsx file and compare with detected head/tail positions.

This script:
1. Reads the Excel file with manual head/tail coordinates
2. Extracts detected head/tail positions from detection summary or spark_tracks
3. Calculates differences between manual and detected coordinates
4. Generates a comparison report as an appendix
"""

import pandas as pd
import numpy as np
import argparse
from pathlib import Path
import re
import json
from collections import defaultdict

try:
    import openpyxl  # For reading Excel files
    HAS_OPENPYXL = True
except ImportError:
    HAS_OPENPYXL = False
    print("⚠ Warning: openpyxl not installed. Install with: pip install openpyxl")


def parse_excel_coordinates(excel_path):
    """
    Parse the XY coordinates.xlsx file.
    
    Returns:
        dict: {folder: {video: {embryo_id: {'head': (x, y), 'tail': (x, y)}}}}
    """
    if not HAS_OPENPYXL:
        raise ImportError("openpyxl is required to read Excel files. Install with: pip install openpyxl")
    
    print(f"Reading Excel file: {excel_path}")
    
    # Try to read all sheets
    excel_file = pd.ExcelFile(excel_path)
    print(f"  Found {len(excel_file.sheet_names)} sheet(s): {excel_file.sheet_names}")
    
    coordinates = defaultdict(lambda: defaultdict(dict))
    
    # Read each sheet
    for sheet_name in excel_file.sheet_names:
        print(f"\n  Processing sheet: {sheet_name}")
        df = pd.read_excel(excel_path, sheet_name=sheet_name)
        
        # Print column names to understand structure
        print(f"    Columns: {list(df.columns)}")
        print(f"    Shape: {df.shape}")
        print(f"    First few rows:")
        print(df.head())
        
        # Try to identify the structure
        # Common patterns:
        # - Folder/Video columns
        # - Head X, Head Y, Tail X, Tail Y columns
        # - Embryo A/B columns
        
        # Look for folder/video identifiers
        folder_col = None
        video_col = None
        
        for col in df.columns:
            col_lower = str(col).lower()
            if 'folder' in col_lower or 'folder' in col_lower:
                folder_col = col
            if 'video' in col_lower or 'file' in col_lower or 'name' in col_lower:
                video_col = col
        
        # Look for ID column and coordinate columns (X, Y)
        # The header row is row 0, so check the actual column values
        id_col = None
        x_col = None
        y_col = None
        
        # Check first row to find which column has "ID", "X", "Y"
        if len(df) > 0:
            first_row = df.iloc[0]
            for col_idx, col_name in enumerate(df.columns):
                val = str(first_row[col_name]).strip() if pd.notna(first_row[col_name]) else ""
                val_lower = val.lower()
                if val_lower == 'id' or 'id' in val_lower:
                    id_col = col_name
                elif val_lower == 'x':
                    x_col = col_name
                elif val_lower == 'y':
                    y_col = col_name
        
        # Fallback: check column names directly
        if not id_col:
            for col in df.columns:
                col_str = str(col).lower()
                if col_str == 'id' or 'id' in col_str:
                    id_col = col
                    break
        
        if not x_col:
            for col in df.columns:
                if str(col).strip().upper() == 'X' or 'x' in str(col).lower():
                    x_col = col
                    break
        
        if not y_col:
            for col in df.columns:
                if str(col).strip().upper() == 'Y' or 'y' in str(col).lower():
                    y_col = col
                    break
        
        print(f"\n    Detected columns:")
        print(f"      ID: {id_col}")
        print(f"      X: {x_col}, Y: {y_col}")
        
        # Get folder from sheet name
        folder = sheet_name
        
        # Get video from first column (usually contains video name like "B-substack")
        video = None
        if len(df.columns) > 0:
            first_col = df.columns[0]
            # Get video name from first non-header row
            for idx in range(1, min(5, len(df))):
                if pd.notna(df.iloc[idx][first_col]):
                    video_val = str(df.iloc[idx][first_col]).strip()
                    # Skip if it looks like an ID value
                    if not ('_head' in video_val.lower() or '_tail' in video_val.lower() or video_val.lower() in ['head', 'tail']):
                        video = video_val
                        break
        
        # If video not found, use first column name
        if not video and len(df.columns) > 0:
            video = str(df.columns[0]).strip()
        
        # Parse rows - look for A_head, A_tail, B_head, B_tail in ID column
        if id_col and x_col and y_col:
            for idx, row in df.iterrows():
                try:
                    if pd.isna(row[id_col]):
                        continue
                    
                    id_val = str(row[id_col]).strip()
                    
                    # Parse ID like "A_head", "A_tail", "B_head", "B_tail", or just "Head", "Tail"
                    embryo_id = None
                    is_head = None
                    
                    if '_' in id_val:
                        parts = id_val.split('_')
                        if len(parts) >= 2:
                            embryo_part = parts[0].upper()
                            head_tail_part = parts[1].lower()
                            if embryo_part in ['A', 'B']:
                                embryo_id = embryo_part
                            if 'head' in head_tail_part:
                                is_head = True
                            elif 'tail' in head_tail_part:
                                is_head = False
                    else:
                        # Single embryo case - just "Head" or "Tail"
                        id_lower = id_val.lower()
                        if 'head' in id_lower:
                            is_head = True
                            embryo_id = 'A'  # Default to A for single embryo
                        elif 'tail' in id_lower:
                            is_head = False
                            embryo_id = 'A'  # Default to A for single embryo
                    
                    # Check for poke location
                    if 'poke' in id_val.lower():
                        # This is a poke location
                        if pd.notna(row[x_col]) and pd.notna(row[y_col]):
                            x = float(row[x_col])
                            y = float(row[y_col])
                            if 'poke' not in coordinates[folder][video]:
                                coordinates[folder][video]['poke'] = {}
                            coordinates[folder][video]['poke'] = (x, y)
                            print(f"      ✓ {folder}/{video} Poke: ({x:.1f}, {y:.1f})")
                        continue
                    
                    if embryo_id is None or is_head is None:
                        continue
                    
                    # Get coordinates
                    if pd.notna(row[x_col]) and pd.notna(row[y_col]):
                        x = float(row[x_col])
                        y = float(row[y_col])
                        
                        # Initialize if needed
                        if embryo_id not in coordinates[folder][video]:
                            coordinates[folder][video][embryo_id] = {}
                        
                        if is_head:
                            coordinates[folder][video][embryo_id]['head'] = (x, y)
                        else:
                            coordinates[folder][video][embryo_id]['tail'] = (x, y)
                        
                        print(f"      ✓ {folder}/{video} Embryo {embryo_id} {'Head' if is_head else 'Tail'}: ({x:.1f}, {y:.1f})")
                except Exception as e:
                    print(f"      ⚠ Error parsing row {idx}: {e}")
                    continue
    
    return coordinates


def extract_detected_coordinates(spark_tracks_path, detection_summary_dir=None):
    """
    Extract detected head/tail coordinates from spark_tracks.csv or detection summary.
    
    Returns:
        dict: {folder: {video: {embryo_id: {'head': (x, y), 'tail': (x, y)}}}}
    """
    print(f"\nExtracting detected coordinates from: {spark_tracks_path}")
    
    df_tracks = pd.read_csv(spark_tracks_path)
    print(f"  Loaded {len(df_tracks):,} track states")
    
    detected = defaultdict(lambda: defaultdict(dict))
    
    # Group by folder/video
    df_tracks['base_filename'] = df_tracks['filename'].str.replace(r' \(page \d+\)', '', regex=True)
    
    for base_file in df_tracks['base_filename'].unique():
        # Extract folder and video
        parts = base_file.split('/')
        if len(parts) >= 2:
            folder = parts[0]
            video = '/'.join(parts[1:])
        else:
            folder = "unknown"
            video = base_file
        
        df_file = df_tracks[df_tracks['base_filename'] == base_file].copy()
        
        # For each embryo, get head/tail positions
        for embryo_id in ['A', 'B']:
            df_embryo = df_file[df_file['embryo_id'] == embryo_id].copy()
            if len(df_embryo) == 0:
                continue
            
            # Get head/tail from ap_norm coordinates
            # ap_norm = 0 is head, ap_norm = 1 is tail
            head_points = df_embryo[df_embryo['ap_norm'] <= 0.1]  # Head region
            tail_points = df_embryo[df_embryo['ap_norm'] >= 0.9]  # Tail region
            
            if len(head_points) > 0 and len(tail_points) > 0:
                # Use median position for head and tail
                head_x = head_points['x'].median()
                head_y = head_points['y'].median()
                tail_x = tail_points['x'].median()
                tail_y = tail_points['y'].median()
                
                detected[folder][video][embryo_id] = {
                    'head': (float(head_x), float(head_y)),
                    'tail': (float(tail_x), float(tail_y)),
                    'num_head_points': len(head_points),
                    'num_tail_points': len(tail_points)
                }
    
    print(f"  Found coordinates for {sum(len(videos) for videos in detected.values())} folder/video combinations")
    return detected


def normalize_video_name(video_name):
    """
    Normalize video name for matching.
    Examples:
    - "B-substack" -> "b substack"
    - "B - Substack (1-301).tif" -> "b substack"
    - "B5-substack" -> "b5 substack"
    """
    if not video_name:
        return ""
    # Convert to lowercase
    normalized = str(video_name).lower()
    # Remove file extensions
    normalized = re.sub(r'\.(tif|tiff|mp4)$', '', normalized)
    # Remove parenthetical content like "(1-301)"
    normalized = re.sub(r'\s*\([^)]+\)', '', normalized)
    # Normalize whitespace and separators
    normalized = re.sub(r'[_\-\s]+', ' ', normalized)
    normalized = normalized.strip()
    return normalized


def find_matching_video(manual_video, detected_videos):
    """
    Find the best matching video name from detected videos.
    """
    manual_norm = normalize_video_name(manual_video)
    if not manual_norm:
        return None
    
    best_match = None
    best_score = 0
    
    for detected_video in detected_videos:
        detected_norm = normalize_video_name(detected_video)
        if not detected_norm:
            continue
        
        # Exact match
        if manual_norm == detected_norm:
            return detected_video
        
        # Check if one contains the other
        if manual_norm in detected_norm or detected_norm in manual_norm:
            score = min(len(manual_norm), len(detected_norm)) / max(len(manual_norm), len(detected_norm))
            if score > best_score:
                best_score = score
                best_match = detected_video
    
    return best_match


def calculate_differences(manual_coords, detected_coords):
    """
    Calculate differences between manual and detected coordinates.
    
    Returns:
        list of dicts with comparison data
    """
    comparisons = []
    
    # Normalize folder names to strings for consistent matching
    def normalize_folder(folder):
        """Normalize folder name - should be numeric string."""
        return str(folder).strip()
    
    # Build a mapping: folder -> list of (normalized_video, actual_video) pairs
    detected_video_map = {}
    for folder in detected_coords:
        folder_norm = normalize_folder(folder)
        if folder_norm not in detected_video_map:
            detected_video_map[folder_norm] = []
        for video in detected_coords[folder]:
            norm = normalize_video_name(video)
            detected_video_map[folder_norm].append((norm, video, folder))
    
    # Find all folder/video/embryo combinations
    all_keys = set()
    for folder in manual_coords:
        folder_norm = normalize_folder(folder)
        for video in manual_coords[folder]:
            for embryo_id in manual_coords[folder][video]:
                all_keys.add((folder_norm, video, embryo_id))
    
    for folder in detected_coords:
        folder_norm = normalize_folder(folder)
        for video in detected_coords[folder]:
            for embryo_id in detected_coords[folder][video]:
                all_keys.add((folder_norm, video, embryo_id))
    
    for folder_norm, video, embryo_id in all_keys:
        # Find manual coordinates - match by folder number
        manual = None
        manual_folder = None
        manual_video = None
        for folder in manual_coords:
            if normalize_folder(folder) == folder_norm:
                # Try exact video match first
                if video in manual_coords[folder]:
                    manual = manual_coords[folder][video].get(embryo_id)
                    if manual:
                        manual_folder = folder
                        manual_video = video
                        break
                
                # If no exact match, try normalized video name matching
                if not manual:
                    manual_norm = normalize_video_name(video)
                    for vid in manual_coords[folder]:
                        if normalize_video_name(vid) == manual_norm:
                            manual = manual_coords[folder][vid].get(embryo_id)
                            if manual:
                                manual_folder = folder
                                manual_video = vid
                                break
                    if manual:
                        break
        
        # Find detected coordinates - match by folder number
        detected = None
        detected_folder = None
        detected_video = None
        if folder_norm in detected_video_map:
            # Try exact video match first
            for folder in detected_coords:
                if normalize_folder(folder) == folder_norm:
                    if video in detected_coords[folder]:
                        detected = detected_coords[folder][video].get(embryo_id)
                        if detected:
                            detected_folder = folder
                            detected_video = video
                            break
            
            # If no exact match, try normalized video name matching
            if not detected:
                video_norm = normalize_video_name(video)
                for norm, actual_video, actual_folder in detected_video_map[folder_norm]:
                    if norm == video_norm:
                        detected = detected_coords[actual_folder][actual_video].get(embryo_id)
                        if detected:
                            detected_folder = actual_folder
                            detected_video = actual_video
                            break
        
        if not manual and not detected:
            continue
        
        # Use the actual folder/video names found
        final_folder = manual_folder if manual_folder else (detected_folder if detected_folder else folder_norm)
        final_video = manual_video if manual_video else (detected_video if detected_video else video)
        
        comp = {
            'folder': final_folder,
            'video': final_video,
            'embryo_id': embryo_id,
            'manual_head': manual['head'] if manual else None,
            'manual_tail': manual['tail'] if manual else None,
            'detected_head': detected['head'] if detected else None,
            'detected_tail': detected['tail'] if detected else None,
        }
        
        # Calculate differences if both exist
        if manual and detected:
            head_diff = np.sqrt((manual['head'][0] - detected['head'][0])**2 + 
                               (manual['head'][1] - detected['head'][1])**2)
            tail_diff = np.sqrt((manual['tail'][0] - detected['tail'][0])**2 + 
                               (manual['tail'][1] - detected['tail'][1])**2)
            
            comp['head_distance_diff'] = head_diff
            comp['tail_distance_diff'] = tail_diff
            comp['head_x_diff'] = abs(manual['head'][0] - detected['head'][0])
            comp['head_y_diff'] = abs(manual['head'][1] - detected['head'][1])
            comp['tail_x_diff'] = abs(manual['tail'][0] - detected['tail'][0])
            comp['tail_y_diff'] = abs(manual['tail'][1] - detected['tail'][1])
        
        comparisons.append(comp)
    
    return comparisons


def generate_report(comparisons, output_path):
    """
    Generate a comparison report as markdown and CSV.
    """
    df = pd.DataFrame(comparisons)
    
    # Save CSV
    csv_path = output_path.with_suffix('.csv')
    df.to_csv(csv_path, index=False)
    print(f"\n✓ Saved CSV report: {csv_path}")
    
    # Generate markdown report
    md_path = output_path.with_suffix('.md')
    
    with open(md_path, 'w') as f:
        f.write("# Head/Tail Coordinate Comparison Report\n\n")
        f.write("This report compares manually annotated head/tail coordinates from ")
        f.write("`XY coordinates.xlsx` with automatically detected coordinates.\n\n")
        
        # Summary statistics
        if len(df) > 0:
            # Check if distance columns exist
            has_distances = 'head_distance_diff' in df.columns
            if has_distances:
                matched = df[df['head_distance_diff'].notna()]
            else:
                matched = pd.DataFrame()  # Empty if no matches
                
            if len(matched) > 0:
                f.write("## Summary Statistics\n\n")
                f.write(f"- **Total comparisons:** {len(df)}\n")
                f.write(f"- **Matched pairs:** {len(matched)}\n")
                f.write(f"- **Manual only:** {len(df[df['manual_head'].notna() & df['detected_head'].isna()])}\n")
                f.write(f"- **Detected only:** {len(df[df['manual_head'].isna() & df['detected_head'].notna()])}\n\n")
                
                f.write("### Distance Differences (pixels)\n\n")
                f.write(f"- **Head distance (mean ± std):** {matched['head_distance_diff'].mean():.1f} ± {matched['head_distance_diff'].std():.1f}\n")
                f.write(f"- **Head distance (median):** {matched['head_distance_diff'].median():.1f}\n")
                f.write(f"- **Head distance (min/max):** {matched['head_distance_diff'].min():.1f} / {matched['head_distance_diff'].max():.1f}\n\n")
                
                f.write(f"- **Tail distance (mean ± std):** {matched['tail_distance_diff'].mean():.1f} ± {matched['tail_distance_diff'].std():.1f}\n")
                f.write(f"- **Tail distance (median):** {matched['tail_distance_diff'].median():.1f}\n")
                f.write(f"- **Tail distance (min/max):** {matched['tail_distance_diff'].min():.1f} / {matched['tail_distance_diff'].max():.1f}\n\n")
        
        # Detailed table
        f.write("## Detailed Comparisons\n\n")
        f.write("| Folder | Video | Embryo | Manual Head | Manual Tail | Detected Head | Detected Tail | Head Diff (px) | Tail Diff (px) |\n")
        f.write("|--------|-------|--------|-------------|-------------|---------------|---------------|----------------|----------------|\n")
        
        for _, row in df.iterrows():
            manual_head_str = f"({row['manual_head'][0]:.1f}, {row['manual_head'][1]:.1f})" if pd.notna(row['manual_head']) and row['manual_head'] is not None else "—"
            manual_tail_str = f"({row['manual_tail'][0]:.1f}, {row['manual_tail'][1]:.1f})" if pd.notna(row['manual_tail']) and row['manual_tail'] is not None else "—"
            detected_head_str = f"({row['detected_head'][0]:.1f}, {row['detected_head'][1]:.1f})" if pd.notna(row['detected_head']) and row['detected_head'] is not None else "—"
            detected_tail_str = f"({row['detected_tail'][0]:.1f}, {row['detected_tail'][1]:.1f})" if pd.notna(row['detected_tail']) and row['detected_tail'] is not None else "—"
            head_diff_str = f"{row['head_distance_diff']:.1f}" if 'head_distance_diff' in row and pd.notna(row['head_distance_diff']) else "—"
            tail_diff_str = f"{row['tail_distance_diff']:.1f}" if 'tail_distance_diff' in row and pd.notna(row['tail_distance_diff']) else "—"
            
            f.write(f"| {row['folder']} | {row['video']} | {row['embryo_id']} | {manual_head_str} | {manual_tail_str} | {detected_head_str} | {detected_tail_str} | {head_diff_str} | {tail_diff_str} |\n")
    
    print(f"✓ Saved Markdown report: {md_path}")
    
    return md_path, csv_path


def main():
    parser = argparse.ArgumentParser(
        description="Compare manual XY coordinates with detected head/tail positions"
    )
    parser.add_argument('excel_path', help='Path to XY coordinates.xlsx file')
    parser.add_argument('spark_tracks', help='Path to spark_tracks.csv')
    parser.add_argument('--output', '-o', default='head_tail_comparison',
                       help='Output file prefix (default: head_tail_comparison)')
    
    args = parser.parse_args()
    
    # Parse Excel file
    manual_coords = parse_excel_coordinates(args.excel_path)
    
    # Extract detected coordinates
    detected_coords = extract_detected_coordinates(args.spark_tracks)
    
    # Calculate differences
    comparisons = calculate_differences(manual_coords, detected_coords)
    
    # Generate report
    output_path = Path(args.output)
    generate_report(comparisons, output_path)
    
    print(f"\n✓ Complete! Generated comparison report.")


if __name__ == '__main__':
    main()


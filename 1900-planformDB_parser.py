import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Polygon, Circle, Ellipse, FancyBboxPatch
from pathlib import Path
import numpy as np
from io import BytesIO
import xml.etree.ElementTree as ET
import re
import os
from typing import Optional, Dict, List, Tuple
from collections import defaultdict

# Get database path from environment variable (set via: export PLANFORM_DB_PATH=/path/to/planformDB_2.5.0.edb)
DB_PATH = os.getenv('PLANFORM_DB_PATH', 'path/to/planformDB_2.5.0.edb')

def load_time_series(db_path: str) -> pd.DataFrame:
    """Load yearly statistics: experiments, publications, morphologies."""
    conn = sqlite3.connect(db_path)
    try:
        # Experiments per year (Experiment → Publication.Year)
        exp_years = pd.read_sql_query(
            """
            SELECT p.Year AS Year, COUNT(*) AS Experiments
            FROM Experiment e
            JOIN Publication p ON p.Id = e.Publication
            WHERE p.Year IS NOT NULL
            GROUP BY p.Year
            ORDER BY p.Year
            """,
            conn,
        )
        # Publications per year
        pub_years = pd.read_sql_query(
            """
            SELECT Year, COUNT(*) AS Publications
            FROM Publication
            WHERE Year IS NOT NULL
            GROUP BY Year
            ORDER BY Year
            """,
            conn,
        )
        # Distinct morphologies observed per year
        morph_years = pd.read_sql_query(
            """
            SELECT p.Year AS Year, COUNT(DISTINCT rm.Morphology) AS MorphologiesObserved
            FROM ResultantMorphology rm
            JOIN ResultSet rs ON rs.Id = rm.ResultSet
            JOIN Experiment e ON e.Id = rs.Experiment
            JOIN Publication p ON p.Id = e.Publication
            WHERE p.Year IS NOT NULL AND rm.Frequency > 0
            GROUP BY p.Year
            ORDER BY p.Year
            """,
            conn,
        )
    finally:
        conn.close()

    # Merge and fill
    df = (
        exp_years.merge(pub_years, on="Year", how="outer")
                 .merge(morph_years, on="Year", how="outer")
                 .sort_values("Year")
                 .reset_index(drop=True)
    )
    if not df.empty:
        full_years = pd.DataFrame({"Year": range(int(df["Year"].min()), int(df["Year"].max()) + 1)})
        df = full_years.merge(df, on="Year", how="left").fillna(0)
        for c in ["Experiments", "Publications", "MorphologiesObserved"]:
            df[c] = df[c].astype(int)
        # cumulative total of distinct morphologies seen up to each year
        df["CumulativeMorphologies"] = df["MorphologiesObserved"].cumsum()
    return df


def extract_morphology_shapes(db_path: str) -> pd.DataFrame:
    """
    Extract morphology data with organ positions and region connections.
    Returns DataFrame with Year, MorphologyId, MorphologyName, and organ/region data.
    """
    conn = sqlite3.connect(db_path)
    try:
        # Extract morphologies with their organ positions
        query = """
        SELECT DISTINCT
            p.Year AS Year,
            m.Id AS MorphologyId,
            m.Name AS MorphologyName,
            COUNT(DISTINCT rm.Id) AS ObservationCount,
            SUM(rm.Frequency) AS TotalFrequency
        FROM ResultantMorphology rm
        JOIN ResultSet rs ON rs.Id = rm.ResultSet
        JOIN Experiment e ON e.Id = rs.Experiment
        JOIN Publication p ON p.Id = e.Publication
        JOIN Morphology m ON m.Id = rm.Morphology
        WHERE p.Year IS NOT NULL 
          AND rm.Frequency > 0
        GROUP BY p.Year, m.Id, m.Name
        ORDER BY p.Year, m.Id
        """
        
        df = pd.read_sql_query(query, conn)
        
        # Extract organ data for each morphology
        print("  Extracting organ positions and region connections...")
        organ_data = {}
        
        for morph_id in df['MorphologyId'].unique():
            # Get all organs for this morphology with their types
            organ_query = """
            SELECT o.Id, o.VecX, o.VecY, o.Region, o.SpotOrgan, o.LineOrgan,
                   so.Type as SpotType, so.Rot as SpotRot,
                   lo.Id as LineOrganId
            FROM Morphology m
            JOIN MorphologyAction ma ON m.Id = ma.Morphology
            JOIN Organ o ON ma.Id = o.Id
            LEFT JOIN SpotOrgan so ON o.SpotOrgan = so.Id
            LEFT JOIN LineOrgan lo ON o.LineOrgan = lo.Id
            WHERE m.Id = ?
            """
            organs = pd.read_sql_query(organ_query, conn, params=(morph_id,))
            
            # Get regions with their types (Head=1, Trunk=2, Tail=3)
            region_query = """
            SELECT r.Id, r.Type, rt.Name as TypeName
            FROM Region r
            LEFT JOIN RegionType rt ON r.Type = rt.Id
            WHERE r.Morphology = ?
            ORDER BY r.Type, r.Id
            """
            regions_info = pd.read_sql_query(region_query, conn, params=(morph_id,))
            
            # Get region connections (how regions link together)
            region_link_query = """
            SELECT rl.FromRegion, rl.ToRegion, rl.Dist, rl.Ratio, rl.Ang
            FROM Region r
            JOIN RegionsLink rl ON (r.Id = rl.FromRegion OR r.Id = rl.ToRegion)
            WHERE r.Morphology = ?
            """
            region_links = pd.read_sql_query(region_link_query, conn, params=(morph_id,))
            
            # Build region data with organ positions
            region_data = {}
            for _, region_row in regions_info.iterrows():
                region_id = region_row['Id']
                region_type = region_row['Type']
                type_name = region_row.get('TypeName', 'Unknown')
                
                # Get organs in this region
                region_organs = organs[organs['Region'] == region_id] if not organs.empty else pd.DataFrame()
                
                if not region_organs.empty:
                    xs = region_organs['VecX'].dropna().tolist()
                    ys = region_organs['VecY'].dropna().tolist()
                    
                    region_data[region_id] = {
                        'type': region_type,
                        'type_name': type_name,
                        'organs': region_organs[['VecX', 'VecY']].to_dict('records'),
                        'x_min': min(xs) if xs else 0,
                        'x_max': max(xs) if xs else 0,
                        'y_min': min(ys) if ys else 0,
                        'y_max': max(ys) if ys else 0,
                        'x_center': np.mean(xs) if xs else 0,
                        'y_center': np.mean(ys) if ys else 0
                    }
                else:
                    region_data[region_id] = {
                        'type': region_type,
                        'type_name': type_name,
                        'organs': [],
                        'x_min': 0, 'x_max': 0, 'y_min': 0, 'y_max': 0,
                        'x_center': 0, 'y_center': 0
                    }
            
            # Organ type mapping: SpotType 1=Eye, 2=Brain, 3=Pharynx
            # LineOrgan = VNC (ventral nerve cord)
            organ_data[morph_id] = {
                'organs': organs.to_dict('records') if not organs.empty else [],
                'regions': regions_info.to_dict('records'),
                'region_data': region_data,
                'region_links': region_links.to_dict('records') if not region_links.empty else []
            }
        
        # Store as JSON-serializable string (we'll parse it when rendering)
        import json
        df['OrganData'] = df['MorphologyId'].apply(
            lambda mid: json.dumps(organ_data.get(mid, {}))
        )
        
        print(f"  ✓ Extracted organ data for {len(organ_data)} morphologies")
        
    finally:
        conn.close()
    
    return df


def parse_svg_data(svg_data: Optional[str]) -> Optional[List[Tuple]]:
    """
    Parse SVG data and extract path coordinates or shape elements.
    Returns list of coordinate tuples or shape parameters.
    """
    if not svg_data:
        return None
    
    try:
        # Try parsing as XML/SVG
        if isinstance(svg_data, bytes):
            svg_data = svg_data.decode('utf-8', errors='ignore')
        
        root = ET.fromstring(svg_data)
        
        # Extract paths
        paths = []
        for path in root.findall('.//{http://www.w3.org/2000/svg}path'):
            d_attr = path.get('d', '')
            # Simple path parser (M = move, L = line, C = curve, Z = close)
            coords = []
            for match in re.finditer(r'([MLCZ])\s*([\d\.\-\s,]+)?', d_attr):
                cmd = match.group(1)
                if match.group(2):
                    nums = [float(x) for x in re.findall(r'[\d\.\-]+', match.group(2))]
                    coords.append((cmd, nums))
            if coords:
                paths.append(coords)
        
        # Extract circles, ellipses, rectangles
        for circle in root.findall('.//{http://www.w3.org/2000/svg}circle'):
            cx = float(circle.get('cx', 0))
            cy = float(circle.get('cy', 0))
            r = float(circle.get('r', 0))
            paths.append([('CIRCLE', [cx, cy, r])])
        
        for rect in root.findall('.//{http://www.w3.org/2000/svg}rect'):
            x = float(rect.get('x', 0))
            y = float(rect.get('y', 0))
            w = float(rect.get('width', 0))
            h = float(rect.get('height', 0))
            paths.append([('RECT', [x, y, w, h])])
        
        return paths if paths else None
        
    except Exception as e:
        # If not valid SVG, try parsing as coordinate string
        try:
            # Try comma-separated or space-separated coordinates
            coords = []
            for line in svg_data.strip().split('\n'):
                parts = re.findall(r'[\d\.\-]+', line)
                if len(parts) >= 2:
                    coords.append((float(parts[0]), float(parts[1])))
            return coords if coords else None
        except:
            return None


def render_morphology_shape(ax, organ_data_json: Optional[str], morph_name: str, 
                           x_center: float = 0, y_center: float = 0, 
                           scale: float = 1.0, color: str = 'steelblue', 
                           alpha: float = 0.7, highlight: bool = False):
    """
    Render a morphology as a unified head-tail body, matching PlanformDB visualization.
    Regions are connected head-to-tail to form a single unified body shape.
    """
    import json
    
    if organ_data_json:
        try:
            organ_data = json.loads(organ_data_json)
            regions_info = organ_data.get('regions', [])
            region_data = organ_data.get('region_data', {})
            region_links = organ_data.get('region_links', [])
            organs = organ_data.get('organs', [])
            
            if region_data and regions_info:
                # Build unified body from regions (Head -> Trunk -> Tail)
                # Sort regions by type: Head (1), Trunk (2), Tail (3)
                sorted_regions = sorted(regions_info, key=lambda r: (r.get('Type', 999), r.get('Id', 0)))
                
                # Collect all organ positions for normalization
                all_organs = []
                for region_id, rdata in region_data.items():
                    for organ in rdata.get('organs', []):
                        if organ.get('VecX') is not None and organ.get('VecY') is not None:
                            all_organs.append((organ['VecX'], organ['VecY']))
                
                if all_organs:
                    xs = [p[0] for p in all_organs]
                    ys = [p[1] for p in all_organs]
                    x_min, x_max = min(xs), max(xs)
                    y_min, y_max = min(ys), max(ys)
                    x_range = x_max - x_min if x_max != x_min else 1
                    y_range = y_max - y_min if y_max != y_min else 1
                    
                    # Normalize scale
                    norm_scale = min(scale / max(x_range, y_range), scale * 0.7)
                    
                    # Build unified body outline: single continuous shape from head to tail
                    # Use convex hull of all organs to create unified body
                    try:
                        from scipy.spatial import ConvexHull
                        all_points_array = np.array([[x, y] for x, y in all_organs])
                        if len(all_points_array) >= 3:
                            hull = ConvexHull(all_points_array)
                            hull_points = all_points_array[hull.vertices]
                            
                            # Normalize and center hull points
                            normalized_body = []
                            for x, y in hull_points:
                                nx = x_center + (x - (x_min + x_max) / 2) * norm_scale / max(x_range, y_range)
                                ny = y_center + (y - (y_min + y_max) / 2) * norm_scale / max(x_range, y_range)
                                normalized_body.append([nx, ny])
                            
                            # Draw unified body as single continuous shape (like PlanformDB)
                            edge_width = 3.0 if highlight else 2.0
                            edge_color = 'red' if highlight else 'black'
                            body_poly = Polygon(normalized_body, 
                                              color=color, 
                                              alpha=alpha, 
                                              edgecolor=edge_color, 
                                              linewidth=edge_width,
                                              zorder=2 if highlight else 1)
                            ax.add_patch(body_poly)
                            return
                    except Exception as e:
                        # Fallback if scipy fails
                        pass
                    
                    # Fallback: create unified body from region centers
                    if len(sorted_regions) >= 2:
                        # Build smooth body outline connecting regions
                        region_centers = []
                        for region_info in sorted_regions:
                            region_id = region_info.get('Id')
                            if region_id in region_data:
                                rdata = region_data[region_id]
                                x_c = rdata.get('x_center', 0)
                                y_c = rdata.get('y_center', 0)
                                region_centers.append((x_c, y_c))
                        
                        if len(region_centers) >= 2:
                            # Create unified body shape: elongated oval connecting head to tail
                            # Use first region (head) and last region (tail) to define body axis
                            head_x, head_y = region_centers[0]
                            tail_x, tail_y = region_centers[-1]
                            
                            # Calculate body axis
                            body_length = np.sqrt((tail_x - head_x)**2 + (tail_y - head_y)**2)
                            body_angle = np.arctan2(tail_y - head_y, tail_x - head_x)
                            
                            # Create unified oval body shape along the axis
                            body_width = scale * 0.3
                            body_length_scaled = max(body_length * norm_scale / max(x_range, y_range), scale * 0.6)
                            
                            # Generate oval outline
                            n_points = 20
                            oval_points = []
                            for i in range(n_points):
                                angle = 2 * np.pi * i / n_points
                                # Ellipse in local coordinates
                                local_x = body_length_scaled / 2 * np.cos(angle)
                                local_y = body_width / 2 * np.sin(angle)
                                # Rotate and translate
                                rot_x = local_x * np.cos(body_angle) - local_y * np.sin(body_angle)
                                rot_y = local_x * np.sin(body_angle) + local_y * np.cos(body_angle)
                                
                                center_x = x_center + (head_x + tail_x - (x_min + x_max)) / 2 * norm_scale / max(x_range, y_range)
                                center_y = y_center + (head_y + tail_y - (y_min + y_max)) / 2 * norm_scale / max(x_range, y_range)
                                
                                oval_points.append([center_x + rot_x, center_y + rot_y])
                            
                            edge_width = 3.0 if highlight else 2.0
                            edge_color = 'red' if highlight else 'black'
                            body_poly = Polygon(oval_points, 
                                              color=color, 
                                              alpha=alpha, 
                                              edgecolor=edge_color, 
                                              linewidth=edge_width,
                                              zorder=2 if highlight else 1)
                            ax.add_patch(body_poly)
                            return
                    
                    # If no unified outline, draw regions connected
                    if len(sorted_regions) > 1:
                        # Draw regions as connected segments
                        for i, region_info in enumerate(sorted_regions):
                            region_id = region_info.get('Id')
                            if region_id in region_data:
                                rdata = region_data[region_id]
                                x_c = rdata.get('x_center', 0)
                                y_c = rdata.get('y_center', 0)
                                
                                nx = x_center + (x_c - (x_min + x_max) / 2) * norm_scale / max(x_range, y_range)
                                ny = y_center + (y_c - (y_min + y_max) / 2) * norm_scale / max(x_range, y_range)
                                
                                # Draw region as ellipse/rounded shape
                                region_type = rdata.get('type', 2)
                                if region_type == 1:  # Head
                                    region_shape = Ellipse((nx, ny), 0.3*scale, 0.25*scale, 
                                                         color=color, alpha=alpha, zorder=2)
                                elif region_type == 3:  # Tail
                                    region_shape = Ellipse((nx, ny), 0.25*scale, 0.2*scale, 
                                                         color=color, alpha=alpha, zorder=2)
                                else:  # Trunk
                                    region_shape = Ellipse((nx, ny), 0.35*scale, 0.3*scale, 
                                                         color=color, alpha=alpha, zorder=2)
                                ax.add_patch(region_shape)
                                
                                # Connect to next region
                                if i < len(sorted_regions) - 1:
                                    next_region_id = sorted_regions[i+1].get('Id')
                                    if next_region_id in region_data:
                                        next_rdata = region_data[next_region_id]
                                        next_x = x_center + (next_rdata.get('x_center', 0) - (x_min + x_max) / 2) * norm_scale / max(x_range, y_range)
                                        next_y = y_center + (next_rdata.get('y_center', 0) - (y_min + y_max) / 2) * norm_scale / max(x_range, y_range)
                                        ax.plot([nx, next_x], [ny, next_y], 
                                               color='black', linewidth=2, alpha=0.8, zorder=1)
                        return
        except Exception as e:
            # If parsing fails, fall through to name-based rendering
            pass
    
    # Fallback: render based on morphology name patterns (simplified unified body)
    name_lower = morph_name.lower()
    if 'wild' in name_lower or 'normal' in name_lower:
        # Wild type: elongated body shape
        body_points = np.array([
            [x_center - 0.5*scale, y_center - 0.2*scale],
            [x_center + 0.5*scale, y_center - 0.2*scale],
            [x_center + 0.4*scale, y_center + 0.2*scale],
            [x_center - 0.4*scale, y_center + 0.2*scale]
        ])
        edge_width = 3.0 if highlight else 1.5
        edge_color = 'red' if highlight else 'black'
        body = Polygon(body_points, color=color, alpha=alpha, edgecolor=edge_color, linewidth=edge_width)
        ax.add_patch(body)
    elif 'double' in name_lower and 'head' in name_lower:
        # Double head: body with two anterior bulges
        body_points = np.array([
            [x_center - 0.6*scale, y_center - 0.15*scale],  # Left head
            [x_center - 0.3*scale, y_center - 0.25*scale],
            [x_center - 0.2*scale, y_center - 0.1*scale],
            [x_center + 0.2*scale, y_center - 0.1*scale],   # Right head
            [x_center + 0.3*scale, y_center - 0.25*scale],
            [x_center + 0.6*scale, y_center - 0.15*scale],
            [x_center + 0.4*scale, y_center + 0.2*scale],    # Body
            [x_center - 0.4*scale, y_center + 0.2*scale]
        ])
        edge_width = 3.0 if highlight else 1.5
        edge_color = 'red' if highlight else 'black'
        body = Polygon(body_points, color=color, alpha=alpha, edgecolor=edge_color, linewidth=edge_width)
        ax.add_patch(body)
    elif 'headless' in name_lower:
        # Headless: body without anterior
        body_points = np.array([
            [x_center - 0.4*scale, y_center - 0.2*scale],
            [x_center + 0.4*scale, y_center - 0.2*scale],
            [x_center + 0.3*scale, y_center + 0.2*scale],
            [x_center - 0.3*scale, y_center + 0.2*scale]
        ])
        edge_width = 3.0 if highlight else 1.5
        edge_color = 'red' if highlight else 'black'
        body = Polygon(body_points, color=color, alpha=alpha, edgecolor=edge_color, linewidth=edge_width)
        ax.add_patch(body)
    elif 'tail' in name_lower or 'posterior' in name_lower:
        # Multiple tails: body with posterior extensions
        body_points = np.array([
            [x_center - 0.4*scale, y_center - 0.2*scale],
            [x_center + 0.4*scale, y_center - 0.2*scale],
            [x_center + 0.3*scale, y_center + 0.1*scale],
            [x_center + 0.1*scale, y_center + 0.3*scale],   # Right tail
            [x_center - 0.1*scale, y_center + 0.3*scale],   # Left tail
            [x_center - 0.3*scale, y_center + 0.1*scale]
        ])
        edge_width = 3.0 if highlight else 1.5
        edge_color = 'red' if highlight else 'black'
        body = Polygon(body_points, color=color, alpha=alpha, edgecolor=edge_color, linewidth=edge_width)
        ax.add_patch(body)
    else:
        # Default: simple unified body shape
        body_points = np.array([
            [x_center - 0.4*scale, y_center - 0.2*scale],
            [x_center + 0.4*scale, y_center - 0.2*scale],
            [x_center + 0.3*scale, y_center + 0.2*scale],
            [x_center - 0.3*scale, y_center + 0.2*scale]
        ])
        edge_width = 3.0 if highlight else 1.5
        edge_color = 'red' if highlight else 'black'
        body = Polygon(body_points, color=color, alpha=alpha, edgecolor=edge_color, linewidth=edge_width)
        ax.add_patch(body)
    
    # Render parsed SVG/coordinate data
    try:
        for item in shape_data:
            if isinstance(item, list) and len(item) > 0:
                cmd = item[0] if isinstance(item[0], tuple) else None
                
                if cmd and cmd[0] == 'CIRCLE':
                    _, params = cmd
                    if len(params) >= 3:
                        cx, cy, r = params[0], params[1], params[2]
                        circle = Circle((x_center + cx*scale, y_center + cy*scale), 
                                      r*scale, color=color, alpha=0.7)
                        ax.add_patch(circle)
                
                elif cmd and cmd[0] == 'RECT':
                    _, params = cmd
                    if len(params) >= 4:
                        x, y, w, h = params[0], params[1], params[2], params[3]
                        rect = FancyBboxPatch((x_center + x*scale, y_center + y*scale),
                                            w*scale, h*scale,
                                            color=color, alpha=0.7)
                        ax.add_patch(rect)
                
                elif isinstance(item, tuple) and len(item) == 2:
                    # Coordinate pair
                    x, y = item
                    circle = Circle((x_center + x*scale, y_center + y*scale), 
                                  0.05*scale, color=color, alpha=0.7)
                    ax.add_patch(circle)
                
                else:
                    # Path data - draw as polygon if we have enough points
                    coords = []
                    for point in item:
                        if isinstance(point, tuple) and len(point) >= 2:
                            coords.append((x_center + point[0]*scale, 
                                         y_center + point[1]*scale))
                        elif isinstance(point, tuple) and len(point) == 2:
                            cmd_type, nums = point
                            if cmd_type in ['M', 'L'] and len(nums) >= 2:
                                coords.append((x_center + nums[0]*scale, 
                                             y_center + nums[1]*scale))
                    
                    if len(coords) >= 3:
                        poly = Polygon(coords, color=color, alpha=0.7, edgecolor='black', linewidth=0.5)
                        ax.add_patch(poly)
                    elif len(coords) == 2:
                        ax.plot([coords[0][0], coords[1][0]], 
                               [coords[0][1], coords[1][1]], 
                               color=color, linewidth=2, alpha=0.7)
    except Exception as e:
        # Fallback to simple shape if rendering fails
        circle = Circle((x_center, y_center), 0.3*scale, 
                      color=color, alpha=0.7)
        ax.add_patch(circle)


def get_wild_type_shape(db_path: str):
    """Extract and return the wild-type planarian shape as reference base with organs and regions."""
    conn = sqlite3.connect(db_path)
    try:
        # Get wild type (morphology ID 1) organs with types
        organs_query = """
        SELECT o.VecX, o.VecY, o.Region, r.Type as RegionType,
               o.SpotOrgan, o.LineOrgan, so.Type as SpotType, so.Rot as SpotRot
        FROM MorphologyAction ma
        JOIN Organ o ON ma.Id = o.Id
        JOIN Region r ON o.Region = r.Id
        LEFT JOIN SpotOrgan so ON o.SpotOrgan = so.Id
        WHERE ma.Morphology = 1
        """
        organs = pd.read_sql_query(organs_query, conn)
        
        # Get regions with their parameters
        regions_query = """
        SELECT r.Id as RegionId, r.Type, rp.ParamInd, rp.Value
        FROM Region r
        LEFT JOIN RegionParam rp ON r.Id = rp.Region
        WHERE r.Morphology = 1
        ORDER BY r.Type, r.Id, rp.ParamInd
        """
        region_params = pd.read_sql_query(regions_query, conn)
        
        # Get region connections
        links_query = """
        SELECT rl.FromRegion, rl.ToRegion, rl.Dist, rl.Ratio, rl.Ang
        FROM Region r
        JOIN RegionsLink rl ON (r.Id = rl.FromRegion OR r.Id = rl.ToRegion)
        WHERE r.Morphology = 1
        """
        region_links = pd.read_sql_query(links_query, conn)
        
        if not organs.empty:
            # Get all organ positions
            all_organs = [(o['VecX'], o['VecY']) for _, o in organs.iterrows() 
                         if pd.notna(o['VecX']) and pd.notna(o['VecY'])]
            
            # Separate organs by type for rendering
            eyes = organs[organs['SpotType'] == 1][['VecX', 'VecY', 'SpotRot']].to_dict('records')
            brains = organs[organs['SpotType'] == 2][['VecX', 'VecY', 'SpotRot']].to_dict('records')
            pharynxes = organs[organs['SpotType'] == 3][['VecX', 'VecY', 'SpotRot']].to_dict('records')
            vncs = organs[organs['LineOrgan'].notna()][['VecX', 'VecY']].to_dict('records')
            
            # Build region data with centers and parameters
            regions_data = {}
            for region_id in organs['Region'].unique():
                region_organs = organs[organs['Region'] == region_id]
                if not region_organs.empty:
                    xs = region_organs['VecX'].dropna()
                    ys = region_organs['VecY'].dropna()
                    region_type = region_organs.iloc[0]['RegionType']
                    
                    # Get region parameters
                    region_params_subset = region_params[region_params['RegionId'] == region_id]
                    params = region_params_subset['Value'].tolist() if not region_params_subset.empty else []
                    
                    regions_data[region_id] = {
                        'type': region_type,
                        'x_center': xs.mean() if len(xs) > 0 else 0,
                        'y_center': ys.mean() if len(ys) > 0 else 0,
                        'params': params  # Region shape parameters
                    }
            
            if all_organs:
                xs = [p[0] for p in all_organs]
                ys = [p[1] for p in all_organs]
                
                return {
                    'organs': all_organs,
                    'eyes': eyes,
                    'brains': brains,
                    'pharynxes': pharynxes,
                    'vncs': vncs,
                    'regions': regions_data,
                    'region_links': region_links.to_dict('records'),
                    'x_min': min(xs), 'x_max': max(xs),
                    'y_min': min(ys), 'y_max': max(ys),
                    'x_range': max(xs) - min(xs),
                    'y_range': max(ys) - min(ys)
                }
    finally:
        conn.close()
    
    return None


def render_wild_type_base(ax, wild_type_data, x_center=0, y_center=0, scale=1.0):
    """Render the wild-type planarian as the base reference shape with organs."""
    if not wild_type_data:
        return
    
    all_organs = wild_type_data['organs']
    x_min, x_max = wild_type_data['x_min'], wild_type_data['x_max']
    y_min, y_max = wild_type_data['y_min'], wild_type_data['y_max']
    x_range = wild_type_data['x_range']
    y_range = wild_type_data['y_range']
    
    # Normalize scale
    norm_scale = min(scale / max(x_range, y_range), scale * 0.8)
    
    def normalize_point(x, y):
        """Normalize a point to the display coordinate system."""
        nx = x_center + (x - (x_min + x_max) / 2) * norm_scale / max(x_range, y_range)
        ny = y_center + (y - (y_min + y_max) / 2) * norm_scale / max(x_range, y_range)
        return nx, ny
    
    # Build body shape from graph structure: regions as nodes, organs define positions
    # The body outline connects the outermost points of the graph structure
    regions = wild_type_data.get('regions', {})
    region_links = wild_type_data.get('region_links', [])
    
    # Use actual organ positions to define body outline (graph-based approach)
    # The body shape is defined by the convex hull of all organs, representing
    # the outermost boundary of the graph structure
    try:
        from scipy.spatial import ConvexHull
        points_array = np.array([[x, y] for x, y in all_organs])
        if len(points_array) >= 3:
            hull = ConvexHull(points_array)
            hull_points = points_array[hull.vertices]
            
            # Normalize and center - this creates the body outline from graph nodes
            normalized_body = [normalize_point(x, y) for x, y in hull_points]
            
            # Draw unified body shape (light gray base)
            base_poly = Polygon(normalized_body, 
                              color='lightgray', 
                              alpha=0.3, 
                              edgecolor='gray', 
                              linewidth=1.5,
                              zorder=0)
            ax.add_patch(base_poly)
    except:
        # Fallback if scipy not available
        pass
    else:
        # Fallback: use convex hull if region data not available
        try:
            from scipy.spatial import ConvexHull
            points_array = np.array([[x, y] for x, y in all_organs])
            if len(points_array) >= 3:
                hull = ConvexHull(points_array)
                hull_points = points_array[hull.vertices]
                normalized_body = [normalize_point(x, y) for x, y in hull_points]
                base_poly = Polygon(normalized_body, 
                                  color='lightgray', 
                                  alpha=0.3, 
                                  edgecolor='gray', 
                                  linewidth=1.5,
                                  zorder=0)
                ax.add_patch(base_poly)
        except:
            pass
    
    # Render organs at their graph node positions
    # Organ positions are relative to their region nodes in the graph structure
    organ_scale = norm_scale * 0.12  # Scale for organ size relative to body
    
    # Draw eyes (Type 1) - positioned laterally to Head nodes
    # Eyes should be visible as black circles with white pupils
    for eye in wild_type_data.get('eyes', []):
        if pd.notna(eye.get('VecX')) and pd.notna(eye.get('VecY')):
            ex, ey = normalize_point(eye['VecX'], eye['VecY'])
            # Eye circle (black, larger for visibility)
            eye_radius = organ_scale * 10
            eye_circle = Circle((ex, ey), eye_radius, 
                              color='black', alpha=0.9, zorder=2)
            ax.add_patch(eye_circle)
            # Eye pupil (white center)
            pupil_radius = eye_radius * 0.4
            pupil = Circle((ex, ey), pupil_radius, 
                         color='white', alpha=1.0, zorder=3)
            ax.add_patch(pupil)
    
    # Draw brain lobes (Type 2) - positioned near Head nodes, light green ovals
    for brain in wild_type_data.get('brains', []):
        if pd.notna(brain.get('VecX')) and pd.notna(brain.get('VecY')):
            bx, by = normalize_point(brain['VecX'], brain['VecY'])
            rot = brain.get('SpotRot', 0)
            # Brain as oval, positioned laterally to Head
            brain_ellipse = Ellipse((bx, by), organ_scale * 14, organ_scale * 10,
                                  angle=np.degrees(rot), 
                                  color='lightgreen', alpha=0.7, zorder=2)
            ax.add_patch(brain_ellipse)
    
    # Draw pharynx (Type 3) - positioned along central axis in Trunk
    for pharynx in wild_type_data.get('pharynxes', []):
        if pd.notna(pharynx.get('VecX')) and pd.notna(pharynx.get('VecY')):
            px, py = normalize_point(pharynx['VecX'], pharynx['VecY'])
            rot = pharynx.get('SpotRot', 0)
            # Pharynx as U-shaped ellipse along central axis
            pharynx_ellipse = Ellipse((px, py), organ_scale * 22, organ_scale * 18,
                                    angle=np.degrees(rot),
                                    color='gray', alpha=0.6, zorder=2)
            ax.add_patch(pharynx_ellipse)
    
    # Draw VNCs (ventral nerve cords) - light green curves along body axis
    vncs = wild_type_data.get('vncs', [])
    if len(vncs) >= 2:
        # VNCs run along the body, connecting regions
        vnc_points = [(normalize_point(v['VecX'], v['VecY'])) 
                     for v in vncs if pd.notna(v.get('VecX')) and pd.notna(v.get('VecY'))]
        if len(vnc_points) >= 2:
            # Sort VNC points by Y coordinate to draw as continuous curve
            vnc_points_sorted = sorted(vnc_points, key=lambda p: p[1])
            vnc_xs = [p[0] for p in vnc_points_sorted]
            vnc_ys = [p[1] for p in vnc_points_sorted]
            # Draw VNC as smooth curve along body
            ax.plot(vnc_xs, vnc_ys, color='lightgreen', linewidth=2.5, 
                   alpha=0.7, zorder=2)


def create_morphology_animation(morph_df: pd.DataFrame, db_path: str, 
                               output_path: str = 'planform_morphology_animation.gif',
                               fps: int = 2, figsize: Tuple[int, int] = (16, 10)):
    """
    Create an animated GIF showing NEW morphologies each year highlighted,
    with faded previous morphologies, all superimposed on wild-type base.
    """
    if morph_df.empty:
        print("⚠️  No morphology data to animate.")
        return
    
    # Get wild-type base shape
    print("  Extracting wild-type planarian base shape...")
    wild_type_data = get_wild_type_shape(db_path)
    if not wild_type_data:
        print("  ⚠️  Could not extract wild-type shape, using fallback")
    
    # Group morphologies by year and track which are new
    years = sorted(morph_df['Year'].unique())
    if not years:
        print("⚠️  No years found in morphology data.")
        return
    
    print(f"Creating animation with {len(years)} years ({years[0]} - {years[-1]})...")
    
    # Track when each morphology first appears
    morph_first_year = {}
    for _, row in morph_df.iterrows():
        morph_id = row['MorphologyId']
        year = row['Year']
        if morph_id not in morph_first_year or year < morph_first_year[morph_id]:
            morph_first_year[morph_id] = year
    
    # Set up figure
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_aspect('equal')
    ax.axis('off')
    
    # Color palette for different morphologies
    colors = plt.cm.Set3(np.linspace(0, 1, morph_df['MorphologyId'].nunique()))
    morph_colors = {mid: colors[i] for i, mid in enumerate(morph_df['MorphologyId'].unique())}
    
    # Track cumulative morphologies
    cumulative_morphs = set()
    
    def animate(frame_idx):
        ax.clear()
        ax.set_xlim(-5, 5)
        ax.set_ylim(-5, 5)
        ax.set_aspect('equal')
        ax.axis('off')
        
        current_year = years[min(frame_idx, len(years) - 1)]
        
        # Draw wild-type base shape (centered, large)
        if wild_type_data:
            render_wild_type_base(ax, wild_type_data, x_center=0, y_center=0, scale=3.0)
        
        # Get morphologies up to current year
        year_morphs = morph_df[morph_df['Year'] <= current_year]
        
        # Separate into new (this year) and old (previous years)
        new_morphs = year_morphs[year_morphs['Year'] == current_year]['MorphologyId'].unique()
        old_morphs = year_morphs[year_morphs['Year'] < current_year]['MorphologyId'].unique()
        
        # Update cumulative set
        for morph_id in year_morphs['MorphologyId'].unique():
            cumulative_morphs.add(morph_id)
        
        # Render old morphologies (faded, superimposed on base)
        for morph_id in old_morphs:
            morph_data = morph_df[morph_df['MorphologyId'] == morph_id].iloc[0]
            organ_data_json = morph_data.get('OrganData') if 'OrganData' in morph_data else None
            
            color = morph_colors.get(morph_id, 'steelblue')
            # Render faded (low alpha)
            render_morphology_shape(ax, organ_data_json, morph_data['MorphologyName'],
                                   x_center=0, y_center=0, scale=3.0, 
                                   color=color, alpha=0.15)
        
        # Render new morphologies (highlighted, superimposed on base)
        for morph_id in new_morphs:
            morph_data = morph_df[morph_df['MorphologyId'] == morph_id].iloc[0]
            organ_data_json = morph_data.get('OrganData') if 'OrganData' in morph_data else None
            
            color = morph_colors.get(morph_id, 'red')
            # Render highlighted (full alpha, thicker edge)
            render_morphology_shape(ax, organ_data_json, morph_data['MorphologyName'],
                                   x_center=0, y_center=0, scale=3.0, 
                                   color=color, alpha=0.8, highlight=True)
        
        # Title
        n_new = len(new_morphs)
        n_total = len(cumulative_morphs)
        ax.text(0, 4.8, f'PlanformDB: New Morphologies Documented\nYear {current_year} | {n_new} new | {n_total} total',
               ha='center', va='top', fontsize=14, fontweight='bold')
        
        # Year indicator
        ax.text(-4.5, -4.5, f'Year: {current_year}', 
               ha='left', va='bottom', fontsize=12, 
               bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
        
        # Legend
        if n_new > 0:
            ax.text(4.5, 4.5, f'New this year: {n_new}', 
                   ha='right', va='top', fontsize=10,
                   bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
    
    # Create animation
    num_frames = len(years) + 10  # Extra frames at end
    anim = animation.FuncAnimation(fig, animate, frames=num_frames, 
                                  interval=1000/fps, repeat=True, blit=False)
    
    # Save
    print(f"  Saving animation ({num_frames} frames at {fps} fps)...")
    anim.save(output_path, writer='pillow', fps=fps)
    print(f"  ✓ Saved: {output_path}")
    plt.close()


def plot_time_series(df: pd.DataFrame, title: str = "Innovation Timeline: Experiments, Publications, and Total Morphologies"):
    """Plot the time series data."""
    plt.figure(figsize=(12, 6))
    plt.plot(df["Year"], df["Experiments"], label="Experiments per year", color="#f4a261")
    plt.plot(df["Year"], df["Publications"], label="Publications per year", color="#e9c46a")
    plt.plot(df["Year"], df.get("CumulativeMorphologies", df["MorphologiesObserved"].cumsum()),
             label="Cumulative morphologies", color="#e63946")
    plt.title(title)
    plt.xlabel("Year")
    plt.ylabel("Count")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('parser_output.png', dpi=150, bbox_inches='tight')
    print("  ✓ Saved: parser_output.png")
    plt.close()


if __name__ == "__main__":
    import sys
    
    # Allow command-line argument to override environment variable
    if len(sys.argv) > 1:
        DB_PATH = sys.argv[1]
    
    if DB_PATH == "path/to/planformDB_2.5.0.edb" or not os.path.exists(DB_PATH):
        print("⚠️  Please set PLANFORM_DB_PATH environment variable or provide database path as argument")
        print("   Usage:")
        print("     export PLANFORM_DB_PATH=/path/to/planformDB_2.5.0.edb")
        print("     python 1900-planformDB_parser.py")
        print("   OR:")
        print("     python 1900-planformDB_parser.py <path_to_database.edb>")
        sys.exit(1)
    
    print("=" * 80)
    print("PlanformDB Parser: Time Series & Morphology Animation")
    print("=" * 80)
    
    # Load time series
    print("\n1. Loading time series data...")
    df_yearly = load_time_series(DB_PATH)
    print(f"   ✓ Loaded {len(df_yearly)} years of data")
    print(df_yearly.head(10).to_string(index=False))
    print("\n" + df_yearly.tail(10).to_string(index=False))
    
    # Plot time series
    print("\n2. Creating time series plot...")
    plot_time_series(df_yearly)
    
    # Extract morphology shapes
    print("\n3. Extracting morphology shape data...")
    morph_df = extract_morphology_shapes(DB_PATH)
    print(f"   ✓ Found {len(morph_df)} morphology-year observations")
    if not morph_df.empty:
        print(f"   ✓ Unique morphologies: {morph_df['MorphologyId'].nunique()}")
        print(f"   ✓ Year range: {morph_df['Year'].min()} - {morph_df['Year'].max()}")
    
    # Create animation
    if not morph_df.empty:
        print("\n4. Creating morphology animation...")
        create_morphology_animation(morph_df, DB_PATH, output_path='planform_morphology_animation.gif', fps=2)
    else:
        print("\n4. ⚠️  Skipping animation (no morphology data found)")
    
    print("\n" + "=" * 80)
    print("Done!")
    print("=" * 80)

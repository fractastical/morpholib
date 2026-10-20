#!/usr/bin/env python3
"""
Explore PlanformDB schema to find columns that might contain SVG or shape data.
"""

import sqlite3
import sys

def explore_schema(db_path: str):
    """Explore database schema to find all tables and columns."""
    conn = sqlite3.connect(db_path)
    try:
        cursor = conn.cursor()
        
        # Get all table names
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]
        
        print("=" * 80)
        print(f"Database: {db_path}")
        print(f"Found {len(tables)} tables")
        print("=" * 80)
        
        # Explore each table
        for table in sorted(tables):
            print(f"\n📊 Table: {table}")
            print("-" * 80)
            
            # Get column info
            cursor.execute(f"PRAGMA table_info({table})")
            columns = cursor.fetchall()
            
            for col in columns:
                col_id, col_name, col_type, not_null, default_val, pk = col
                pk_str = " [PRIMARY KEY]" if pk else ""
                null_str = " NOT NULL" if not_null else ""
                default_str = f" DEFAULT {default_val}" if default_val else ""
                print(f"  • {col_name}: {col_type}{null_str}{default_str}{pk_str}")
            
            # Check for potential shape/SVG columns
            shape_keywords = ['svg', 'shape', 'image', 'data', 'blob', 'geometry', 'path', 'coordinate']
            potential_shape_cols = []
            for col in columns:
                col_name = col[1].lower()
                if any(keyword in col_name for keyword in shape_keywords):
                    potential_shape_cols.append(col[1])
            
            if potential_shape_cols:
                print(f"\n  ⚠️  Potential shape/SVG columns: {', '.join(potential_shape_cols)}")
                
                # Sample data from these columns
                for col_name in potential_shape_cols[:3]:  # Limit to first 3
                    try:
                        cursor.execute(f"SELECT {col_name} FROM {table} WHERE {col_name} IS NOT NULL LIMIT 1")
                        sample = cursor.fetchone()
                        if sample:
                            sample_val = sample[0]
                            if isinstance(sample_val, bytes):
                                print(f"    {col_name}: <BLOB, {len(sample_val)} bytes>")
                            elif isinstance(sample_val, str):
                                preview = sample_val[:200] if len(sample_val) > 200 else sample_val
                                print(f"    {col_name}: {repr(preview)}")
                            else:
                                print(f"    {col_name}: {type(sample_val).__name__}")
                    except Exception as e:
                        print(f"    {col_name}: Error reading - {e}")
            
            # Count rows
            cursor.execute(f"SELECT COUNT(*) FROM {table}")
            count = cursor.fetchone()[0]
            print(f"\n  📈 Row count: {count:,}")
        
        # Specifically check Morphology table for shape data
        print("\n" + "=" * 80)
        print("🔍 Detailed Morphology Table Analysis")
        print("=" * 80)
        
        if 'Morphology' in tables:
            cursor.execute("SELECT * FROM Morphology LIMIT 5")
            rows = cursor.fetchall()
            cursor.execute("PRAGMA table_info(Morphology)")
            col_info = cursor.fetchall()
            col_names = [col[1] for col in col_info]
            
            print(f"\nColumn names: {', '.join(col_names)}")
            print(f"\nSample rows:")
            for i, row in enumerate(rows, 1):
                print(f"\n  Row {i}:")
                for col_name, val in zip(col_names, row):
                    if isinstance(val, bytes):
                        print(f"    {col_name}: <BLOB, {len(val)} bytes>")
                    elif isinstance(val, str) and len(val) > 100:
                        print(f"    {col_name}: {val[:100]}...")
                    else:
                        print(f"    {col_name}: {val}")
        
    finally:
        conn.close()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python explore_planform_schema.py <path_to_planformDB.edb>")
        sys.exit(1)
    
    db_path = sys.argv[1]
    explore_schema(db_path)

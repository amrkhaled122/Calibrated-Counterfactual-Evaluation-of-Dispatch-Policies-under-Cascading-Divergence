"""
Inspect parquet file columns and sample data.

Usage:
    python data/preprocessing_code/inspect_parquet.py
    python data/preprocessing_code/inspect_parquet.py --path data/features/offers_observations.parquet
    python data/preprocessing_code/inspect_parquet.py --sample 5
"""
import argparse
from pathlib import Path


def inspect_parquet(parquet_path: str, sample_rows: int = 3):
    """
    Inspect a parquet file: show columns, dtypes, and sample data.
    """
    import polars as pl
    
    print(f"Loading: {parquet_path}")
    df = pl.read_parquet(parquet_path)
    
    print(f"\n{'='*80}")
    print(f"PARQUET FILE: {parquet_path}")
    print(f"{'='*80}")
    print(f"Total rows: {df.height:,}")
    print(f"Total columns: {df.width}")
    
    print(f"\n{'='*80}")
    print("COLUMNS AND DTYPES:")
    print(f"{'='*80}")
    for i, (col, dtype) in enumerate(zip(df.columns, df.dtypes)):
        print(f"  {i+1:3d}. {col:<50} {dtype}")
    
    # Look for cycle-related columns
    print(f"\n{'='*80}")
    print("POTENTIAL CYCLE-RELATED COLUMNS:")
    print(f"{'='*80}")
    cycle_keywords = ['cycle', 'wave', 'batch', 'dispatch', 'time', 'start', 'end', 'period']
    for col in df.columns:
        col_lower = col.lower()
        if any(kw in col_lower for kw in cycle_keywords):
            print(f"  - {col}")
    
    # Sample data
    print(f"\n{'='*80}")
    print(f"SAMPLE DATA ({sample_rows} rows):")
    print(f"{'='*80}")
    print(df.head(sample_rows))
    
    # Time-related columns analysis
    print(f"\n{'='*80}")
    print("TIME COLUMNS ANALYSIS:")
    print(f"{'='*80}")
    time_cols = [c for c in df.columns if 'time' in c.lower()]
    for col in time_cols:
        try:
            col_data = df[col].drop_nulls()
            if len(col_data) > 0:
                min_val = col_data.min()
                max_val = col_data.max()
                print(f"\n  {col}:")
                print(f"    Min: {min_val}")
                print(f"    Max: {max_val}")
                # If it looks like a unix timestamp, convert
                if isinstance(min_val, (int, float)) and min_val > 1600000000:
                    from datetime import datetime
                    print(f"    Min (datetime): {datetime.fromtimestamp(min_val)}")
                    print(f"    Max (datetime): {datetime.fromtimestamp(max_val)}")
        except Exception as e:
            print(f"  {col}: Error - {e}")
    
    # Unique values for potential cycle identifiers
    print(f"\n{'='*80}")
    print("UNIQUE VALUE COUNTS (for potential cycle IDs):")
    print(f"{'='*80}")
    for col in df.columns:
        col_lower = col.lower()
        if any(kw in col_lower for kw in ['cycle', 'wave', 'batch', 'id', 'dispatch']):
            try:
                n_unique = df[col].n_unique()
                print(f"  {col}: {n_unique:,} unique values")
            except:
                pass


def main():
    parser = argparse.ArgumentParser(description="Inspect parquet file")
    parser.add_argument('--path', '-p', 
                        default=str(Path(__file__).parent.parent / 'features' / 'offers_observations.parquet'),
                        help='Path to parquet file')
    parser.add_argument('--sample', '-s', type=int, default=3,
                        help='Number of sample rows to show')
    
    args = parser.parse_args()
    inspect_parquet(args.path, args.sample)


if __name__ == '__main__':
    main()

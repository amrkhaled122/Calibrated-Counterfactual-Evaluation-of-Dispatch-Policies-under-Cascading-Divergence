"""
Extract actual ETA values from historical delivery data.

Filters:
- is_prebook = 0 (not pre-booked orders)
- is_courier_grabbed = 1 (courier actually grabbed the order)

For each order:
- eta_seconds_actual = arrive_time - fetch_time

Output:
- order_id, eta_seconds_actual
"""
import pandas as pd
from pathlib import Path
import argparse


def extract_actual_eta(input_path: str, output_path: str) -> pd.DataFrame:
    """
    Extract actual ETA from historical data.
    
    Args:
        input_path: Path to raw waybill CSV
        output_path: Path to save the extracted ETA dataset
        
    Returns:
        DataFrame with order_id and eta_seconds_actual
    """
    print(f"Loading data from {input_path}...")
    df = pd.read_csv(input_path)
    print(f"Total records: {len(df):,}")
    
    # Check available columns
    print(f"\nAvailable columns: {list(df.columns)}")
    
    # Filter: is_prebook = 0
    if 'is_prebook' in df.columns:
        df_filtered = df[df['is_prebook'] == 0].copy()
        print(f"After is_prebook=0 filter: {len(df_filtered):,}")
    else:
        print("Warning: 'is_prebook' column not found, skipping this filter")
        df_filtered = df.copy()
    
    # Filter: is_courier_grabbed = 1
    if 'is_courier_grabbed' in df.columns:
        df_filtered = df_filtered[df_filtered['is_courier_grabbed'] == 1].copy()
        print(f"After is_courier_grabbed=1 filter: {len(df_filtered):,}")
    else:
        print("Warning: 'is_courier_grabbed' column not found, skipping this filter")
    
    # Check required columns for ETA calculation
    required_cols = ['order_id', 'arrive_time', 'fetch_time']
    missing = [c for c in required_cols if c not in df_filtered.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    
    # Calculate actual ETA = arrive_time - fetch_time
    print("\nCalculating actual ETA (arrive_time - fetch_time)...")
    
    # Convert to numeric, coercing errors
    df_filtered['arrive_time'] = pd.to_numeric(df_filtered['arrive_time'], errors='coerce')
    df_filtered['fetch_time'] = pd.to_numeric(df_filtered['fetch_time'], errors='coerce')
    
    # Drop rows with missing times
    before_drop = len(df_filtered)
    df_filtered = df_filtered.dropna(subset=['arrive_time', 'fetch_time'])
    print(f"Dropped {before_drop - len(df_filtered)} rows with missing arrive_time or fetch_time")
    
    # Calculate ETA
    df_filtered['eta_seconds_actual'] = df_filtered['arrive_time'] - df_filtered['fetch_time']
    
    # Filter out negative or unreasonable ETAs
    df_filtered = df_filtered[df_filtered['eta_seconds_actual'] > 0]
    df_filtered = df_filtered[df_filtered['eta_seconds_actual'] < 7200]  # Max 2 hours
    print(f"After filtering unreasonable ETAs (0 < eta < 7200s): {len(df_filtered):,}")
    
    # Aggregate by order_id (in case of duplicates, take mean)
    result = df_filtered.groupby('order_id').agg({
        'eta_seconds_actual': 'mean'
    }).reset_index()
    
    # Round to integer
    result['eta_seconds_actual'] = result['eta_seconds_actual'].round().astype(int)
    
    print(f"\nFinal dataset: {len(result):,} unique orders")
    print(f"\nETA statistics:")
    print(f"  Min: {result['eta_seconds_actual'].min()} seconds ({result['eta_seconds_actual'].min()/60:.1f} min)")
    print(f"  Max: {result['eta_seconds_actual'].max()} seconds ({result['eta_seconds_actual'].max()/60:.1f} min)")
    print(f"  Mean: {result['eta_seconds_actual'].mean():.0f} seconds ({result['eta_seconds_actual'].mean()/60:.1f} min)")
    print(f"  Median: {result['eta_seconds_actual'].median():.0f} seconds ({result['eta_seconds_actual'].median()/60:.1f} min)")
    
    # Save to CSV
    result.to_csv(output_path, index=False)
    print(f"\nSaved to {output_path}")
    
    return result


def main():
    parser = argparse.ArgumentParser(description="Extract actual ETA from historical data")
    parser.add_argument('--input', '-i', 
                        default=str(Path(__file__).parent.parent / 'all_waybill_info_meituan_0322_edited.csv'),
                        help='Path to input CSV with waybill data')
    parser.add_argument('--output', '-o',
                        default=str(Path(__file__).parent.parent / 'actual_eta_by_order.csv'),
                        help='Path to output CSV')
    
    args = parser.parse_args()
    
    extract_actual_eta(args.input, args.output)


if __name__ == '__main__':
    main()

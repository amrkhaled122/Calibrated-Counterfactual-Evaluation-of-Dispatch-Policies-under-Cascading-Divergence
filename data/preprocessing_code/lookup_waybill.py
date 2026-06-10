"""
Lookup a specific waybill or courier from the original dataset.

Usage:
    python data/preprocessing_code/lookup_waybill.py 52188
    python data/preprocessing_code/lookup_waybill.py 52188 52189 52190
    python data/preprocessing_code/lookup_waybill.py --courier 2055 --hours 3
    python data/preprocessing_code/lookup_waybill.py -c 2055 -h 3
"""
import pandas as pd
from pathlib import Path
import argparse
import sys
from datetime import datetime


def format_timestamp(value):
    """Format a Unix timestamp as readable date if applicable."""
    try:
        ts = int(value)
        if ts > 1600000000:  # Looks like a Unix timestamp
            readable = datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
            return f"{value} ({readable})"
    except:
        pass
    return str(value)


def lookup_waybill(waybill_ids: list, data_path: str = None):
    """
    Lookup waybill(s) from the original dataset and display all columns.
    
    Args:
        waybill_ids: List of waybill IDs to lookup
        data_path: Path to the CSV file
    """
    if data_path is None:
        data_path = Path(__file__).parent.parent / 'all_waybill_info_meituan_0322_edited.csv'
    
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    print(f"Total records: {len(df):,}\n")
    
    for waybill_id in waybill_ids:
        waybill_id = int(waybill_id)
        print("=" * 80)
        print(f"WAYBILL ID: {waybill_id}")
        print("=" * 80)
        
        # Try to find by waybill_id column
        if 'waybill_id' in df.columns:
            row = df[df['waybill_id'] == waybill_id]
        else:
            # Fallback: check if it's an index
            row = df[df.index == waybill_id]
        
        if len(row) == 0:
            print(f"  NOT FOUND in dataset\n")
            continue
        
        # Display all columns vertically for readability
        row_data = row.iloc[0]
        for col in df.columns:
            value = row_data[col]
            # Format timestamps as readable dates
            if 'time' in col.lower() and pd.notna(value):
                print(f"  {col}: {format_timestamp(value)}")
            else:
                print(f"  {col}: {value}")
        
        print()


def lookup_courier(courier_id: int, hours: float = None, data_path: str = None):
    """
    Lookup all orders for a courier within specified hours from dataset start.
    
    Args:
        courier_id: Courier ID to lookup
        hours: Number of hours from the start of dataset (None = all data)
        data_path: Path to the CSV file
    """
    if data_path is None:
        data_path = Path(__file__).parent.parent / 'all_waybill_info_meituan_0322_edited.csv'
    
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    print(f"Total records: {len(df):,}")
    
    # Filter by courier_id
    if 'courier_id' not in df.columns:
        print("ERROR: 'courier_id' column not found in dataset")
        return
    
    courier_df = df[df['courier_id'] == courier_id].copy()
    print(f"Records for courier {courier_id}: {len(courier_df):,}")
    
    if len(courier_df) == 0:
        print(f"  Courier {courier_id} NOT FOUND in dataset")
        return
    
    # Determine time column for filtering
    time_col = None
    for col in ['grab_time', 'actual_dispatch_time', 'fetch_time']:
        if col in courier_df.columns:
            time_col = col
            break
    
    if time_col is None:
        print("WARNING: No time column found for filtering")
    else:
        # Get dataset start time
        min_time = df[time_col].min()
        print(f"Dataset start time ({time_col}): {format_timestamp(min_time)}")
        
        if hours is not None:
            max_time = min_time + (hours * 3600)
            print(f"Filtering to first {hours} hours (until {format_timestamp(max_time)})")
            courier_df = courier_df[courier_df[time_col] <= max_time]
            print(f"Records in time window: {len(courier_df):,}")
    
    if len(courier_df) == 0:
        print(f"  No records for courier {courier_id} in specified time window")
        return
    
    # Sort by time
    if time_col:
        courier_df = courier_df.sort_values(time_col)
    
    # Display summary
    print("\n" + "=" * 100)
    print(f"COURIER {courier_id} - ORDER TIMELINE")
    print("=" * 100)
    
    # Key columns to display in summary
    key_cols = ['waybill_id', 'order_id', 'grab_time', 'fetch_time', 'arrive_time', 
                'is_courier_grabbed', 'is_accept', 'is_completed', 'is_late',
                'sender_lat', 'sender_lng', 'recipient_lat', 'recipient_lng',
                'order_income_value']
    
    available_cols = [c for c in key_cols if c in courier_df.columns]
    
    for idx, (_, row) in enumerate(courier_df.iterrows(), 1):
        print(f"\n--- Order {idx} ---")
        for col in available_cols:
            value = row[col]
            if 'time' in col.lower() and pd.notna(value):
                print(f"  {col}: {format_timestamp(value)}")
            else:
                print(f"  {col}: {value}")
        
        # Calculate durations if possible
        if 'grab_time' in row and 'fetch_time' in row and pd.notna(row['grab_time']) and pd.notna(row['fetch_time']):
            grab_to_fetch = int(row['fetch_time']) - int(row['grab_time'])
            print(f"  [grab -> fetch]: {grab_to_fetch}s ({grab_to_fetch/60:.1f} min)")
        
        if 'fetch_time' in row and 'arrive_time' in row and pd.notna(row['fetch_time']) and pd.notna(row['arrive_time']):
            fetch_to_arrive = int(row['arrive_time']) - int(row['fetch_time'])
            print(f"  [fetch -> arrive]: {fetch_to_arrive}s ({fetch_to_arrive/60:.1f} min)")
        
        if 'grab_time' in row and 'arrive_time' in row and pd.notna(row['grab_time']) and pd.notna(row['arrive_time']):
            total_time = int(row['arrive_time']) - int(row['grab_time'])
            print(f"  [TOTAL grab -> arrive]: {total_time}s ({total_time/60:.1f} min)")
    
    # Print full details option
    print("\n" + "-" * 100)
    print(f"Total orders shown: {len(courier_df)}")
    
    # Show all columns for first order as reference
    if len(courier_df) > 0:
        print("\n[All columns for first order:]")
        first_row = courier_df.iloc[0]
        for col in df.columns:
            value = first_row[col]
            if 'time' in col.lower() and pd.notna(value):
                print(f"  {col}: {format_timestamp(value)}")
            else:
                print(f"  {col}: {value}")


def main():
    parser = argparse.ArgumentParser(description="Lookup waybill(s) or courier from original dataset")
    parser.add_argument('waybill_ids', nargs='*', help='Waybill ID(s) to lookup')
    parser.add_argument('--courier', '-c', type=int, 
                        help='Courier ID to lookup (shows all orders for this courier)')
    parser.add_argument('--hours', '-hr', type=float, 
                        help='Filter to first N hours from dataset start (use with --courier)')
    parser.add_argument('--data', '-d', 
                        default=None,
                        help='Path to CSV file (default: data/all_waybill_info_meituan_0322_edited.csv)')
    
    args = parser.parse_args()
    
    if args.courier is not None:
        # Courier lookup mode
        lookup_courier(args.courier, args.hours, args.data)
    elif args.waybill_ids:
        # Waybill lookup mode
        lookup_waybill(args.waybill_ids, args.data)
    else:
        parser.print_help()
        print("\nExamples:")
        print("  python data/preprocessing_code/lookup_waybill.py 52188")
        print("  python data/preprocessing_code/lookup_waybill.py --courier 2055 --hours 3")
        print("  python data/preprocessing_code/lookup_waybill.py -c 2055 -hr 3")


if __name__ == '__main__':
    main()

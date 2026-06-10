"""
Fix overlapping courier working blocks.

Problem: Some courier blocks have overlapping time ranges, causing utilization
calculation issues (e.g., Block 1 ends at 20:00:28 but Block 2 starts at 19:58:22).

Solution: For each courier, sort blocks by start time, then ensure no block
starts before the previous block ends. If overlap detected, adjust the start
of the later block to be 1 second after the end of the previous block.
"""
import pandas as pd
from pathlib import Path


def fix_overlapping_blocks(input_path: str, output_path: str = None):
    """
    Fix overlapping blocks by adjusting start times.
    
    Args:
        input_path: Path to courier_working_blocks.csv
        output_path: Path to save fixed CSV (if None, overwrites input)
    """
    if output_path is None:
        output_path = input_path
    
    print(f"Loading blocks from {input_path}...")
    df = pd.read_csv(input_path)
    
    print(f"Total blocks: {len(df)}")
    print(f"Unique couriers: {df['courier_id'].nunique()}")
    
    # Track statistics
    total_overlaps_fixed = 0
    total_time_adjusted = 0
    couriers_with_overlaps = set()
    
    # Process each courier separately
    fixed_rows = []
    
    for courier_id, group in df.groupby('courier_id'):
        # Sort by start time
        group = group.sort_values('block_start_time').reset_index(drop=True)
        
        prev_end = None
        for idx, row in group.iterrows():
            start = row['block_start_time']
            end = row['block_end_time']
            day = row['day']
            
            # Check for overlap with previous block
            if prev_end is not None and start < prev_end:
                # Overlap detected!
                old_start = start
                new_start = prev_end + 1  # 1 second after previous block ends
                
                # Also check if new_start would exceed end (invalid block)
                if new_start >= end:
                    # Block is entirely within previous block, skip it
                    # This means it was a sub-block, not a separate shift
                    print(f"  Courier {courier_id}: Skipping block [{old_start}-{end}] "
                          f"(entirely within previous block ending at {prev_end})")
                    continue
                
                total_overlaps_fixed += 1
                time_diff = new_start - old_start
                total_time_adjusted += time_diff
                couriers_with_overlaps.add(courier_id)
                
                start = new_start
            
            fixed_rows.append({
                'courier_id': courier_id,
                'block_start_time': start,
                'block_end_time': end,
                'day': day
            })
            
            prev_end = end
    
    # Create new dataframe
    fixed_df = pd.DataFrame(fixed_rows)
    
    print(f"\n=== Summary ===")
    print(f"Total overlaps fixed: {total_overlaps_fixed}")
    print(f"Couriers with overlaps: {len(couriers_with_overlaps)}")
    print(f"Total time adjusted: {total_time_adjusted} seconds ({total_time_adjusted/3600:.2f} hours)")
    print(f"Blocks before: {len(df)}")
    print(f"Blocks after: {len(fixed_df)}")
    print(f"Blocks removed (sub-blocks): {len(df) - len(fixed_df)}")
    
    # Save
    print(f"\nSaving fixed blocks to {output_path}...")
    fixed_df.to_csv(output_path, index=False)
    print("Done!")
    
    return fixed_df


def verify_no_overlaps(csv_path: str):
    """Verify that no overlapping blocks exist."""
    df = pd.read_csv(csv_path)
    
    overlaps_found = 0
    for courier_id, group in df.groupby('courier_id'):
        group = group.sort_values('block_start_time')
        prev_end = None
        
        for _, row in group.iterrows():
            if prev_end is not None and row['block_start_time'] < prev_end:
                overlaps_found += 1
                print(f"Overlap: Courier {courier_id}: "
                      f"prev_end={prev_end}, curr_start={row['block_start_time']}")
            prev_end = row['block_end_time']
    
    if overlaps_found == 0:
        print("✓ No overlapping blocks found!")
    else:
        print(f"✗ Found {overlaps_found} overlapping blocks")
    
    return overlaps_found == 0


if __name__ == "__main__":
    import sys
    
    # Default path
    blocks_path = "data/courier_working_blocks.csv"
    
    if len(sys.argv) > 1:
        blocks_path = sys.argv[1]
    
    # Fix overlaps
    fix_overlapping_blocks(blocks_path)
    
    # Verify
    print("\n=== Verification ===")
    verify_no_overlaps(blocks_path)

# Data Preprocessing Code

This folder contains the preprocessing pipeline and utility scripts used to generate or inspect derived data under `data/`.

## Files

### Data Files

#### actual_eta_by_order.csv

**Description:** Extracted actual delivery times (ETA) for orders where the courier completed the delivery.

| Property | Value |
|----------|-------|
| Rows | 546,350 (+ header) |
| Columns | 2 |

**Schema:**

| Column | Type | Description |
|--------|------|-------------|
| `order_id` | int | Unique order identifier |
| `eta_seconds_actual` | int | Actual delivery time (arrive_time - fetch_time) in seconds |

**Generation:** Created by `extract_actual_eta.py` from the main waybill data.

**Filters applied:**
- `is_prebook = 0` (excludes pre-booked orders)
- `is_courier_grabbed = 1` (only orders actually accepted by courier)

---

### Utility Scripts

#### extract_actual_eta.py

**Purpose:** Extract actual ETA values from historical delivery data for use in simulation and model validation.

**Usage:**
```powershell
python data/preprocessing_code/extract_actual_eta.py `
  --input data/all_waybill_info_meituan_0322_edited.csv `
  --output data/actual_eta_by_order.csv
```

**Output:** CSV with `order_id` and `eta_seconds_actual` for delivered orders.

---

#### fix_overlapping_blocks.py

**Purpose:** Resolve overlapping courier working blocks in `data/courier_working_blocks.csv`.

**Problem:** Some courier blocks have overlapping time ranges due to data inconsistencies:
- Block 1 ends at 20:00:28
- Block 2 starts at 19:58:22 (overlap!)

**Solution:** For each courier:
1. Sort blocks by start time
2. Detect overlaps
3. Adjust later block's start to 1 second after previous block's end

**Usage:**
```powershell
python data/preprocessing_code/fix_overlapping_blocks.py `
  data/courier_working_blocks.csv
```

---

#### inspect_parquet.py

**Purpose:** Diagnostic tool to inspect Parquet files - view columns, data types, and sample data.

**Usage:**
```powershell
# Default path (offers_observations.parquet)
python data/preprocessing_code/inspect_parquet.py

# Custom path
python data/preprocessing_code/inspect_parquet.py --path data/features/offers_observations.parquet

# More sample rows
python data/preprocessing_code/inspect_parquet.py --sample 10
```

**Output includes:**
- Total rows and columns
- Column names and data types
- Cycle-related columns identification
- Time column analysis (min/max values)
- Sample data preview

---

#### lookup_waybill.py

**Purpose:** Quick lookup tool to find specific waybills or courier records in the raw data.

**Usage:**
```powershell
# Lookup single waybill
python data/preprocessing_code/lookup_waybill.py 52188

# Lookup multiple waybills
python data/preprocessing_code/lookup_waybill.py 52188 52189 52190

# Lookup courier's orders (first 3 hours)
python data/preprocessing_code/lookup_waybill.py --courier 2055 --hours 3

# Short form
python data/preprocessing_code/lookup_waybill.py -c 2055 -hr 3
```

**Output:** Formatted display of all columns for matching records, with timestamps converted to readable format.

---

## Relationship to Main Pipeline

```
data/                           (Raw input)
    │
    └── all_waybill_info_meituan_0322_edited.csv
            │
            ├── data/preprocessing_code/features/ (Feature engineering)
            │
            └── data/preprocessing_code/          (Supplementary tools)
                    │
                    ├── extract_actual_eta.py → actual_eta_by_order.csv
                    │
                    ├── inspect_parquet.py   → Diagnostic inspection
                    │
                    ├── lookup_waybill.py    → Record lookup
                    │
                    └── fix_overlapping_blocks.py → Simulation data fixes
```

## When to Use These Tools

| Tool | Use Case |
|------|----------|
| `extract_actual_eta.py` | Generate ground truth delivery times for simulation validation |
| `fix_overlapping_blocks.py` | Clean up courier utilization data after simulation |
| `inspect_parquet.py` | Debug preprocessed data, check feature availability |
| `lookup_waybill.py` | Investigate specific orders or courier behavior |

---

*Last updated: January 2026*

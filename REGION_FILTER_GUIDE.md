# 🌍 Region Filter Guide - Per-Category Configuration

## Overview

The Per-Category Filter Configuration now includes **region-based filtering**, allowing you to create custom regional indices like:
- **LV FLATS RIGA** (only Riga data)
- **LV FLATS KURZEME + VIDZEME + LATGALE + ZEMGALE** (excluding Riga and Pierīga)
- Any custom combination of regions you need

## How It Works

### Two-Level Region Filtering

The system uses a **two-level region filtering approach**:

1. **Index-Level Regions** (Predefined in code)
   - Each index like "LV FLATS RIGA" has predefined regions
   - These are the default regions for that specific index

2. **Per-Category Region Filter** (NEW - Configurable in UI)
   - Applied to ALL indexes in a category
   - Acts as a **pre-filter** before index-level filtering
   - Creates an **intersection** with index-level regions

### Example Scenarios

#### Scenario 1: No Per-Category Filter (Default Behavior)
```
Category: LV FLATS
├── LV FLATS RIGA → Uses [Rīga]
├── LV FLATS PIE-RIGA → Uses [Pierīga]
└── LV FLATS KURZEME + VIDZEME + LATGALE + ZEMGALE → Uses [Kurzeme, Vidzeme, Latgale, Zemgale]
```

#### Scenario 2: Per-Category Filter Set to [Rīga, Pierīga]
```
Category: LV FLATS (filtered to Rīga + Pierīga)
├── LV FLATS RIGA → Uses [Rīga] ∩ [Rīga, Pierīga] = [Rīga] ✅
├── LV FLATS PIE-RIGA → Uses [Pierīga] ∩ [Rīga, Pierīga] = [Pierīga] ✅
└── LV FLATS KURZEME + VIDZEME + LATGALE + ZEMGALE → Uses [Kurzeme, Vidzeme, Latgale, Zemgale] ∩ [Rīga, Pierīga] = [] ❌ (No data!)
```

#### Scenario 3: Per-Category Filter Set to [Rīga]
```
Category: LV FLATS (filtered to Rīga only)
├── LV FLATS RIGA → Uses [Rīga] ∩ [Rīga] = [Rīga] ✅
├── LV FLATS PIE-RIGA → Uses [Pierīga] ∩ [Rīga] = [] ❌ (No data!)
└── LV FLATS KURZEME + VIDZEME + LATGALE + ZEMGALE → Uses [Kurzeme, Vidzeme, Latgale, Zemgale] ∩ [Rīga] = [] ❌ (No data!)
```

## How to Use

### Step 1: Enable Per-Category Filters

In the sidebar under "🎯 Per-Category Filter Configuration":
1. Check ✅ **"Enable per-category filters"**

### Step 2: Configure Region Filter for a Category

1. Expand the category you want to configure (e.g., "⚙️ Filters for LV FLATS")
2. Under **🌍 Region Selection**, select the regions you want to include:
   - **Rīga** - Riga city
   - **Pierīga** - Riga region
   - **Kurzeme** - Kurzeme region
   - **Vidzeme** - Vidzeme region
   - **Zemgale** - Zemgale region
   - **Latgale** - Latgale region
   - **Unknown** - Unknown/unclassified regions
3. **Leave empty** to use all regions (default behavior)

### Step 3: View Results

- The dashboard will show which regions are active for each category
- Look for: **"🌍 Regions: Rīga + Pierīga"** under each category
- Indexes with no matching data will show warnings

## Use Cases

### Use Case 1: Focus Analysis on Specific Regions
**Goal**: Only analyze Riga and Pieriga markets

**Solution**: 
- Set per-category filter for "LV FLATS" to [Rīga, Pierīga]
- Set per-category filter for "LV HOUSES" to [Rīga, Pierīga]
- Other categories remain unfiltered

### Use Case 2: Exclude Problematic Data
**Goal**: Exclude "Unknown" region data which has quality issues

**Solution**:
- For each category, select only the known regions
- Don't include "Unknown" in the selection

### Use Case 3: Regional Market Comparison
**Goal**: Compare only rural markets (excluding Riga and Pieriga)

**Solution**:
- Set filter to [Kurzeme, Vidzeme, Latgale, Zemgale]
- This will automatically filter out Riga-specific indices

## Important Notes

⚠️ **Warning**: Setting a per-category region filter can result in some indices having **NO DATA** if there's no intersection between:
- The per-category regions you selected
- The index's predefined regions

✅ **Best Practice**: Leave the per-category region filter **EMPTY** unless you need to restrict the analysis to specific regions.

💡 **Tip**: The per-category region filter is most useful for:
- Temporary analysis of specific regions
- Excluding problematic regions
- Creating custom regional groupings

## Available Regions

| Region Code | Description |
|------------|-------------|
| Rīga | Riga city (capital) |
| Pierīga | Riga metropolitan region |
| Kurzeme | Western Latvia |
| Vidzeme | Northern Latvia |
| Zemgale | Southern Latvia |
| Latgale | Eastern Latvia |
| Unknown | Unclassified/unknown regions |

## Display in Dashboard

When per-category region filters are active, you'll see them displayed at the top of each category section:

```
📂 LV FLATS
   🌍 Regions: Rīga + Pierīga
   Price/m²: 500-5000 EUR
   Date: 2020-01-01 to 2025-11-19
```

This helps you quickly identify which filters are active for each category.


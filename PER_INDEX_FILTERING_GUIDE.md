# 🎯 Per-Index Filtering Guide

## ✨ NEW FEATURE: Individual Index Filtering

You can now apply **different filters to each index** within a category, giving you complete control over your analysis!

## 🔍 The Problem This Solves

**Before:** All indexes in "LV FLATS" got the same filters:
- LV FLATS RIGA → Same filters ❌
- LV FLATS PIE-RIGA → Same filters ❌  
- LV FLATS KURZEME + VIDZEME + LATGALE + ZEMGALE → Same filters ❌

**After:** Each index can have its own custom filters:
- LV FLATS RIGA → Filters A ✅
- LV FLATS PIE-RIGA → Filters B ✅
- LV FLATS KURZEME + VIDZEME + LATGALE + ZEMGALE → Filters C ✅

## 📋 How to Use

### Step 1: Enable Per-Category Filters

In the sidebar under "🎯 Per-Category Filter Configuration":
1. Check ✅ **"Enable per-category filters"**

### Step 2: Choose Filter Level

For each category, expand it (e.g., "⚙️ Filters for LV FLATS") and choose:

**Option A: "Entire category"** (Default)
- Apply the same filters to ALL indexes in this category
- Simpler, fewer settings
- Good for consistent filtering across related indexes

**Option B: "Individual indexes"** (NEW!)
- Configure filters separately for each index
- Maximum flexibility
- Perfect when indexes need different treatments

### Step 3A: Category-Level Filtering

If you chose "Entire category":

1. **Region Selection** - Apply to all indexes in category
2. **Price/m² Range** - Same range for all indexes
3. **Price Range** - Same range for all indexes
4. **Area Range** - Same range for all indexes
5. **Date Range** - Same range for all indexes

Example: Set price/m² 500-5000 EUR for ALL LV FLATS indexes

### Step 3B: Individual Index Filtering (NEW!)

If you chose "Individual indexes":

You'll see separate expandable sections for EACH index:

```
📊 Configure Each Index Separately

🎯 LV FLATS RIGA
  └─ Regions: [Select]
  └─ Price/m²: [Enable] Min/Max
  └─ Price: [Enable] Min/Max
  └─ Area: [Enable] Min/Max
  └─ Date: [Enable] From/To

🎯 LV FLATS PIE-RIGA
  └─ Regions: [Select]
  └─ Price/m²: [Enable] Min/Max
  └─ Price: [Enable] Min/Max
  └─ Area: [Enable] Min/Max
  └─ Date: [Enable] From/To

🎯 LV FLATS KURZEME + VIDZEME + LATGALE + ZEMGALE
  └─ Regions: [Select]
  └─ Price/m²: [Enable] Min/Max
  └─ Price: [Enable] Min/Max
  └─ Area: [Enable] Min/Max
  └─ Date: [Enable] From/To
```

### Step 4: Configure Each Index

For each index, you can:
1. **Select regions** to include/exclude
2. **Enable price/m² filter** and set min/max
3. **Enable price filter** and set min/max EUR
4. **Enable area filter** and set min/max m²
5. **Enable date filter** and set from/to dates
6. **Leave unchecked** to use global filters

## 💡 Use Cases

### Use Case 1: Different Price Ranges per Region

**Scenario:** Riga has higher prices than other regions

**Solution:** Use "Individual indexes"
- **LV FLATS RIGA**: Price/m² 800-5000 EUR
- **LV FLATS PIE-RIGA**: Price/m² 600-3000 EUR
- **LV FLATS KURZEME+...**: Price/m² 300-2000 EUR

### Use Case 2: Different Date Ranges

**Scenario:** You want different historical depth per region

**Solution:** Use "Individual indexes"
- **LV FLATS RIGA**: 2014-2025 (full history)
- **LV FLATS PIE-RIGA**: 2018-2025 (recent only)
- **LV FLATS KURZEME+...**: 2020-2025 (very recent)

### Use Case 3: Exclude Regions from Specific Index

**Scenario:** Remove problematic data from one index only

**Solution:** Use "Individual indexes"
- **LV FLATS RIGA**: All regions
- **LV FLATS PIE-RIGA**: Exclude "Unknown" region
- **LV FLATS KURZEME+...**: All regions

### Use Case 4: Focus on Specific Property Sizes

**Scenario:** Different markets have different typical sizes

**Solution:** Use "Individual indexes"
- **LV FLATS RIGA**: 20-150 m² (typical apartments)
- **LV FLATS PIE-RIGA**: 30-200 m² (larger suburban)
- **LV FLATS KURZEME+...**: 25-120 m² (rural typical)

## 🎯 Filter Priority System

When an index is calculated, filters are applied in this order:

1. **Global Filters** (if enabled in main settings)
   - Date, Price, Area, etc.

2. **Per-Category Filters** (if "Entire category" selected)
   - Overrides global filters
   - Applies to ALL indexes in category

3. **Per-Index Filters** (if "Individual indexes" selected)
   - Overrides both global AND category filters
   - Applies ONLY to that specific index

**Priority**: Per-Index > Per-Category > Global > No Filter

## ⚠️ Important Notes

### Empty = Use Default
If you don't enable a filter for an index, it will:
1. Use the global filter (if enabled)
2. Use no filter (if global also disabled)

### Region Filter Behavior
Region filters create an **intersection**:
- Index predefined regions: [Rīga]
- Per-index region filter: [Rīga, Pierīga]
- Final result: [Rīga] (intersection = Rīga only)

### Visual Indicators

**Category-Level:**
```
🎯 Custom filters for LV FLATS: 🌍 Regions: Rīga + Pierīga | Price/m²: 500-5000 EUR
```

**Index-Level:**
```
📊 Index-Level Filtering Active - Different filters applied to each index in this category
```

## 📊 Display Differences

### Category-Level Filtering
Shows a summary bar with all filters that apply to the category

### Index-Level Filtering
Shows a note that index-level filtering is active. Individual filter details are shown in:
- Export → Per-Category Filters sheet
- Data Quality metrics (filters_removed count)

## 🚀 Quick Start Example

**Goal:** Different price ranges for each LV FLATS index

1. Enable per-category filters ✅
2. Expand "⚙️ Filters for LV FLATS"
3. Select **"Individual indexes"** radio button
4. Expand **"🎯 LV FLATS RIGA"**
   - Enable "Price/m² filter"
   - Set: Min 800, Max 5000
5. Expand **"🎯 LV FLATS PIE-RIGA"**
   - Enable "Price/m² filter"
   - Set: Min 600, Max 3000
6. Expand **"🎯 LV FLATS KURZEME + VIDZEME + LATGALE + ZEMGALE"**
   - Enable "Price/m² filter"
   - Set: Min 300, Max 2000
7. Click "🚀 Generate Final Indexes"

**Result:** Each index uses its own price range! ✨

## 🎉 Benefits

✅ **Maximum Flexibility** - Tailor filters to each market
✅ **Better Data Quality** - Remove outliers specific to each region
✅ **Accurate Comparisons** - Each index uses appropriate thresholds
✅ **Easy to Configure** - Clear UI for each index
✅ **Documented** - All settings exported to Excel

---

**Now you have complete control over filtering for each individual index!** 🎯


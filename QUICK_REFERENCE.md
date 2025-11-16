# 🚀 Quick Reference Card
**Latvian Real Estate Price Index Analyzer**

---

## Starting the App
```
Windows: Double-click start_webapp.bat
Manual: python -m streamlit run app.py
```

---

## 3-Step Analysis
1. **Select**: Houses or Apartments (top of page)
2. **Click**: "🚀 Generate Tables" button
3. **Explore**: Click through tabs to view results

---

## Understanding Tabs

| Tab | What It Shows | When to Use |
|-----|---------------|-------------|
| **Summary** | Overview stats, data quality | Start here! |
| **Prices** | €/m² by region & quarter | See market prices |
| **Counts** | # of transactions | Check data reliability |
| **Index** | % change from 2020-Q1 | Compare growth rates |
| **Distribution** | Outliers & statistics | Clean data |

---

## Moving Averages

| Type | Smoothing | Best For |
|------|-----------|----------|
| **Original** | None | Exact quarterly data |
| **MA2** | Light | See recent trends |
| **MA3** | Moderate | Balanced view |
| **MA4** | Heavy | Long-term patterns |

---

## Index Values Explained

| Value | Meaning |
|-------|---------|
| **1.0** | Same as 2020-Q1 |
| **1.25** | +25% since 2020-Q1 |
| **0.85** | -15% since 2020-Q1 |

---

## Sidebar Filters (Top to Bottom)

### ⚙️ Calculation Method
- **Calculated** = Price ÷ Sold Area (precise)
- **Total_EUR_m2** = Pre-calculated (complete)

### 📅 Date & Time
- **Year Range**: Drag slider
- **Quarters**: Check/uncheck boxes

### 🗺️ Location
- **Regions**: "All" or select specific
- **Municipality**: Optional detailed filter

### 🏗️ Building Type
- Apartments: By construction material
- Houses: By property type

### 💰 Price & Area (Advanced)
- Total Price (EUR)
- Price per m² (EUR/m²)
- Sold Area (m²)
- Total Area (m²)
- Land Area (m²) - Houses only

### 📋 Property Details (Advanced)
- Record Type
- Property Parts (Dom_Parts)
- Finishing
- Category

### 🔍 Duplicate Detection
- **Keep all**: No removal
- **Exact duplicates**: 100% matches
- **Address+Date+Price**: Smart removal

### 🎯 Outlier Detection
1. ☑️ Enable checkbox
2. Choose method:
   - **IQR 1.5x**: Standard (5-10% removed)
   - **IQR 3.0x**: Lenient (1-3% removed)
   - **Percentile**: Custom %
3. Grouping:
   - ☑️ Per region (recommended)
   - ☑️ Per quarter (optional)

---

## Interactive Charts

### Controls Below Each Chart:
- **Region selector**: Choose which to display
- **Date range slider**: Zoom into period
- **Hover**: See exact values

### Chart Tips:
- Too crowded? Deselect regions
- Compare trends? Use Index tab
- Long-term view? Use MA4

---

## Advanced Features

### 🔗 Region Merging
**Location**: Bottom of page (after generating tables)

**Steps**:
1. Select 2+ regions
2. Name merged region
3. ☑️ Show comparison (optional)
4. Click "Generate Merged Analysis"

### 🔬 Index Comparison
**Location**: Within each Index tab, scroll down

**Purpose**: Test outlier filters without regenerating

**Steps**:
1. ☑️ Enable comparison
2. Choose outlier method
3. View overlay chart (solid vs dashed)

---

## Exporting Results

### Excel Report
**Location**: Bottom of page

**Contains**:
- Summary statistics
- All 4 Prices tables
- Counts table
- All 4 Index tables

**Steps**:
1. Click "📊 Generate Excel Report"
2. Wait for processing
3. Click "⬇️ Download Excel Report"

**Filename**: `latvian_[type]_index_report_[timestamp].xlsx`

---

## Keyboard Shortcuts

| Key | Action |
|-----|--------|
| **F5** | Refresh page |
| **Ctrl + F5** | Hard refresh (clear cache) |
| **Ctrl + Click** | Multi-select in dropdowns |
| **Esc** | Close popup/modal |

---

## Common Workflows

### Quick Market Check (2 min)
```
1. Select property type
2. Generate Tables
3. View Index-Original tab
4. Check 2-3 regions in chart
```

### Regional Analysis (5 min)
```
1. Select property type
2. Filter: Choose 2-3 regions
3. Filter: Last 2-3 years
4. Generate Tables
5. Compare Prices-Original vs MA4
6. Check Counts for reliability
```

### Clean Data Analysis (15 min)
```
1. Select property type
2. Generate Tables (check Distribution)
3. Enable: Outlier Detection (IQR 1.5x + per region)
4. Enable: Duplicate removal
5. Regenerate Tables
6. Use Index comparison to verify
7. Export to Excel
```

---

## Warning Indicators

### Summary Tab Alerts

| Alert | Meaning | Action |
|-------|---------|--------|
| **⚠️ Data Quality** | Missing values | Try Total_EUR_m2 method |
| **🎯 Outliers Removed** | Extreme values filtered | Check % removed |
| **🔍 Duplicates** | Repeat entries | Consider removal |

### Color Codes in Distribution Tab

| Color | Outlier % | Meaning |
|-------|-----------|---------|
| 🔴 **Red** | >10% | High variability |
| 🟡 **Yellow** | 5-10% | Moderate issues |
| ⚪ **White** | <5% | Stable data |

---

## Troubleshooting Quick Fixes

| Problem | Quick Fix |
|---------|-----------|
| No data showing | Click "Generate Tables" |
| Filters not working | Apply → then Generate |
| Too many outliers | Enable outlier filter |
| Chart too busy | Deselect regions |
| App is slow | Reset filters, fewer regions |
| Can't export | Generate tables first |

---

## Data Quality Checklist

Before finalizing analysis:
- [ ] Check Summary tab for warnings
- [ ] Review Distribution Analysis outliers
- [ ] Verify Counts (>10 per region/quarter)
- [ ] Compare Original vs MA4 (consistency?)
- [ ] Test outlier filter impact (comparison)
- [ ] Check duplicate statistics
- [ ] Document filters used

---

## Best Practices

### ✅ DO:
- Start with Summary tab
- Check transaction counts
- Use MA3/MA4 for trends
- Compare Index (not Prices) across regions
- Document your filters
- Export results regularly

### ❌ DON'T:
- Trust single-transaction quarters
- Compare prices without checking counts
- Ignore outliers in Distribution tab
- Use Original when MA4 shows clearer trend
- Forget to regenerate after filter changes

---

## Statistical Terms (Quick)

| Term | Simple Meaning |
|------|----------------|
| **Mean** | Average |
| **Median** | Middle value |
| **Std Dev** | How spread out |
| **Q1** | 25th percentile |
| **Q3** | 75th percentile |
| **IQR** | Q3 - Q1 (middle 50%) |
| **Outlier** | Extreme value |

---

## Getting Help

### Resources (in order):
1. **This card** - Quick answers
2. **In-app tooltips** - Hover ℹ️ icons
3. **USER_GUIDE.md** - Detailed guide
4. **Summary tab** - Data quality info

### For More Detail:
📖 Open **USER_GUIDE.md** for:
- Step-by-step tutorials
- Detailed feature explanations
- Advanced workflows
- Complete troubleshooting
- FAQ section

---

## Key Takeaways

### 🎯 For Reliable Analysis:
1. **Always check Counts** - Low counts = unreliable
2. **Use Index for comparisons** - Normalizes baselines
3. **Check data quality first** - Summary + Distribution
4. **Clean your data** - Remove outliers/duplicates
5. **Use Moving Averages** - Smooth seasonal noise

### ⚡ For Fast Results:
1. **Use default filters first** - See big picture
2. **Generate once, explore many** - All tabs ready
3. **Export to Excel** - Analyze offline
4. **Merge regions** - Macro view without reprocessing

---

**💡 Remember**: When in doubt, check the Summary tab and Counts tab first!

---

*Print this page and keep it handy while using the app!*

**Full Documentation**: USER_GUIDE.md  
**Version**: 1.0  
**Updated**: November 6, 2024


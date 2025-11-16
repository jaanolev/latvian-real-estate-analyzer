# 📚 User Guide: Latvian Real Estate Price Index Analyzer

**Welcome!** This guide will help you analyze Latvian house and apartment prices with ease. Whether you're a first-time user or looking for advanced features, we've got you covered.

---

## 🚀 Quick Start (5 Minutes)

### Starting the Application

**Windows Users:**
1. Double-click `start_webapp.bat` in the project folder
2. A browser window will open automatically at `http://localhost:8501`

**Alternative Method:**
```powershell
cd C:\Users\annik\Documents\LV_index_interface
python -m streamlit run app.py
```

### Your First Analysis (3 Steps)

1. **Choose Property Type** 
   - At the top of the page, select either **Houses** or **Apartments**
   - The data loads automatically (you'll see a green checkmark ✅)

2. **Click "Generate Tables"**
   - The big blue button in the center
   - Wait a few seconds while tables are calculated

3. **Explore Results**
   - Click through the tabs: Summary, Prices, Counts, Index
   - Hover over charts to see details
   - That's it! You're analyzing real estate data 🎉

---

## 📊 Understanding Your Results

### The Tabs Explained

#### **Summary Tab** 📈
**What it shows:** Overview of all your filtered data

Key Information:
- **Transaction counts** by region
- **Average prices** per square meter
- **Total sales volume** in each region
- **Duplicate analysis** (helps identify data quality issues)
- **Data quality warnings** (missing values, excluded records)

**💡 Tip:** Start here to get the big picture before diving into details!

---

#### **Prices Tabs** (4 versions) 💰

**What they show:** Average price per m² for each region and quarter

**The Four Versions:**
1. **Original** - Raw quarterly data
2. **MA2** - 2-quarter moving average (smooths short-term fluctuations)
3. **MA3** - 3-quarter moving average (smoother trends)
4. **MA4** - 4-quarter moving average (smoothest, shows long-term trends)

**How to read the table:**
- **Rows** = Regions (e.g., Riga, Vidzeme, Kurzeme)
- **Columns** = Time periods (e.g., 2020-Q1, 2020-Q2)
- **Values** = Average price per m² in EUR

**Interactive Chart Below:**
- Select specific regions to compare
- Use date range slider to zoom into time periods
- Hover over lines to see exact values

**When to use which:**
- **Original**: See exact quarterly prices
- **MA2/MA3**: Balance between detail and trend clarity
- **MA4**: Identify long-term market direction

---

#### **Counts Tab** 📊

**What it shows:** Number of transactions per region and quarter

**Why it matters:**
- **High counts** = Reliable price averages (more data points)
- **Low counts** = Take prices with caution (small sample size)
- **Trends** = Market activity levels (busy vs quiet periods)

**💡 Tip:** Always check counts when interpreting price changes. A big price jump with only 2 transactions is less reliable than one with 50!

---

#### **Index Tabs** (4 versions) 📈

**What they show:** Price changes relative to **2020-Q1 = 1.0**

**How to interpret:**
- **1.0** = Same price as 2020-Q1
- **1.25** = 25% more expensive than 2020-Q1
- **0.85** = 15% cheaper than 2020-Q1

**Example:**
If Riga shows **1.45** in 2024-Q2, it means:
- Prices have increased 45% since 2020-Q1
- What cost €1000/m² in 2020-Q1 now costs €1450/m²

**Why use indices:**
- Compare regions that have different baseline prices
- Track relative growth (which region grew fastest?)
- Normalize different markets for comparison

---

#### **Distribution Analysis Tab** 📊

**What it shows:** Statistical outliers and price distributions

**Features:**
1. **Box Plots** - Visualize price spread (median, quartiles, outliers)
2. **Statistical Summary** - See outlier percentages by region/quarter
3. **Histograms** - View frequency distribution of prices
4. **Correlation Analysis** - Area vs Price scatter plots

**When to use:**
- **Data quality checking** - Find unusual transactions
- **Outlier identification** - Before applying filters
- **Understanding market diversity** - How varied are prices?

**Color Coding:**
- 🔴 **Red rows**: >10% outliers (high variability)
- 🟡 **Yellow rows**: 5-10% outliers (moderate variability)
- ⚪ **White rows**: <5% outliers (stable data)

---

## 🎛️ Sidebar Filters (Your Control Panel)

### Basic Filters

#### **📅 Date & Time**
- **Year Range Slider**: Drag to select years (e.g., 2015-2024)
- **Quarters**: Uncheck boxes to exclude specific quarters
  - *Example*: Uncheck Q1 to focus on spring/summer/fall

#### **🗺️ Location**
- **Regions**: 
  - Check "All" to include everything
  - Uncheck "All" to manually select specific regions
  - *Tip*: Compare Riga vs other regions for urban/rural differences

- **Municipality** (optional):
  - Enable for more granular location filtering
  - Select multiple municipalities with Ctrl+Click

#### **🏗️ Building/Property Type**
- **For Apartments**: Filters by building materials (brick, panel, wood)
- **For Houses**: Filters by property types (detached, semi-detached)
- *Tip*: Compare prices between modern vs older construction types

---

### Advanced Filters

#### **💰 Price & Area**
Refine your data by specific ranges:

1. **Price (EUR)** - Total property price
   - *Use case*: Focus on budget segment (e.g., €50k-€150k)

2. **Price per m² (EUR/m²)** - Unit price
   - *Use case*: Remove extremely cheap/expensive outliers

3. **Sold Area (m²)** - Transaction area
   - *Use case*: Focus on standard-sized properties

4. **Total Area (m²)** - Complete property size
   - *Use case*: Exclude micro or mansion properties

5. **Land Area (m²)** - For houses
   - *Use case*: Compare properties with similar lot sizes

**💡 Pro Tip:** Use multiple filters together! Example: Properties €80k-€200k with 50-120m² area

---

#### **🔍 Duplicate Detection**

**Why it matters:** Raw data might contain duplicate listings or transactions

**Options:**
1. **Keep all** - No filtering (default)
2. **Remove exact duplicates** - Removes 100% identical rows
3. **Remove by Address+Date+Price** - Removes likely duplicates

**Recommendation:** 
- Start with "Keep all" to see the data
- Check the Summary tab's "Duplicate Analysis" section
- If duplicates >5%, consider using option 2 or 3

---

#### **🎯 Outlier Detection & Removal**

**What are outliers?** Extreme values that might be:
- Data entry errors (€10/m² or €10,000/m²)
- Unique properties (castles, special deals)
- Genuinely unusual transactions

**How to use:**
1. ☑️ **Enable outlier filtering** checkbox
2. **Select method:**
   - **IQR 1.5x (standard)** - Removes ~5-10% of data (recommended start)
   - **IQR 3.0x (lenient)** - Removes only extreme outliers (~1-3%)
   - **Percentile Method** - Custom: set exact % to remove

3. **Grouping options:**
   - ☑️ **Per region** - Calculate outliers separately for each region
   - ☑️ **Per quarter** - Calculate outliers for each time period
   - Neither checked = Global outliers across all data

**Example Workflow:**
```
1. First run: Keep outliers, check Distribution Analysis tab
2. See many outliers? Enable "IQR 1.5x" 
3. Still too noisy? Check "per region" + "per quarter"
4. Regenerate tables to see cleaned data
```

**⚠️ Warning:** Outliers are removed from ALL tabs (Prices, Counts, Index)

---

### **⚙️ Calculation Method**

**Two ways to calculate price per m²:**

1. **Calculated (Price ÷ Sold Area)** [Default]
   - Formula: `Price_EUR / Sold_Area_m2`
   - ✅ Most accurate for actual transaction price
   - ⚠️ May have missing values if area not recorded

2. **Use Total_EUR_m2 column**
   - Uses pre-calculated field from data source
   - ✅ No missing values (complete dataset)
   - ⚠️ May include fees/adjustments

**Which to choose?**
- If Summary tab shows >20% excluded data → Try method 2
- For standard analysis → Use method 1 (more precise)
- For complete coverage → Use method 2 (no gaps)

---

## 🔗 Advanced Features

### Region Merging

**Location:** Bottom of the page after generating tables

**What it does:** Combines multiple regions into one for aggregate analysis

**Use cases:**
- **Urban vs Rural**: Merge all cities vs all countryside
- **Economic zones**: Combine economically similar regions
- **Custom markets**: Create your own market segments

**How to use:**
1. Scroll to "🔗 Merge & Compare Regions" section
2. Select 2+ regions from dropdown
3. Enter a name for merged region (e.g., "Baltic Coast")
4. ☑️ "Show comparison" to overlay with original regions
5. Click "🚀 Generate Merged Analysis"

**Result:** New tabs showing merged region data across all metrics

---

### Index Comparison with Outlier Filtering

**Location:** Within each Index tab, scroll down to "🔬 Index Comparison"

**What it does:** Apply ADDITIONAL outlier filtering for comparison without changing your main tables

**Why useful:**
- Test outlier impact without regenerating everything
- Compare filtered vs unfiltered trends side-by-side
- Fine-tune outlier settings before committing

**How to use:**
1. Generate main tables first
2. Go to any Index tab
3. Scroll down to comparison section
4. Enable outlier filtering with different settings
5. See overlay chart: solid lines (original) vs dashed lines (filtered)

**💡 Pro Tip:** Try multiple outlier methods here to find the best one, then apply globally!

---

## 📥 Exporting Your Results

### Excel Report

**What's included:**
- Summary statistics table
- All 4 Prices tables (Original + MA2/MA3/MA4)
- Counts table
- All 4 Index tables (Original + MA2/MA3/MA4)

**Steps:**
1. Scroll to bottom: "📥 Export Results"
2. Click "📊 Generate Excel Report"
3. Wait for processing (few seconds)
4. Click "⬇️ Download Excel Report"
5. File saves with timestamp (e.g., `latvian_houses_index_report_20241106_154531.xlsx`)

**Uses:**
- Share with colleagues
- Create presentations
- Further analysis in Excel
- Archive snapshots of your analysis

---

## 🎯 Common Workflows

### Workflow 1: Market Overview (Beginners)

```
1. Select property type (Houses or Apartments)
2. Keep all default filters
3. Click "Generate Tables"
4. Check Summary tab for overall stats
5. View Index-Original tab to see growth trends
6. Select 2-3 regions to compare in the chart
```

**Time:** 2 minutes  
**Best for:** Quick market snapshot

---

### Workflow 2: Regional Deep Dive

```
1. Select property type
2. In sidebar → Location → Uncheck "All"
3. Check only regions you want (e.g., Riga, Pieriga)
4. Adjust date range if needed (e.g., last 3 years)
5. Generate Tables
6. Compare Prices-Original vs Prices-MA4 (spot trends)
7. Check Counts tab (verify data reliability)
8. View Distribution Analysis (understand variability)
```

**Time:** 5-10 minutes  
**Best for:** Detailed regional analysis

---

### Workflow 3: Clean Data Analysis (Advanced)

```
1. Select property type
2. Generate Tables (with defaults)
3. Go to Distribution Analysis tab
4. Check outlier percentages
5. If high outliers (>10%), enable outlier detection:
   - Sidebar → Outlier Detection → Enable
   - Method: IQR 1.5x
   - Check "per region" + "per quarter"
6. Enable duplicate removal: "Remove by Address+Date+Price"
7. Regenerate Tables
8. Compare before/after using Index comparison feature
9. Export clean data to Excel
```

**Time:** 15-20 minutes  
**Best for:** Research, official reports, publications

---

### Workflow 4: Market Segmentation

```
1. Create market segments using filters:
   Segment A: Budget (€50k-€100k, 30-60m²)
   Segment B: Mid-range (€100k-€200k, 60-100m²)
   Segment C: Premium (€200k+, 100m²+)
   
2. For each segment:
   - Apply filters
   - Generate tables
   - Export to Excel
   - Reset filters

3. Compare Excel files to see segment differences
```

**Time:** 30-45 minutes  
**Best for:** Market research, investor analysis

---

## 🐛 Troubleshooting

### "No data available"
**Possible causes:**
- Filters too restrictive (no transactions match criteria)
- Wrong property type selected
- Date range outside available data

**Solutions:**
1. Click "🔄 Reset All Filters" in sidebar
2. Check Summary tab for data availability
3. Gradually add filters one by one

---

### "Tables not showing"
**Possible causes:**
- Forgot to click "Generate Tables"
- Still calculating (wait for success message)

**Solutions:**
1. Look for green "✅ Tables generated!" message
2. If processing >30 seconds, refresh page (F5)
3. Try with fewer filters first

---

### "Chart is too crowded"
**Possible causes:**
- Too many regions selected
- Too many time periods

**Solutions:**
1. Deselect regions in chart controls (below each chart)
2. Use date range slider to zoom into specific period
3. Use Region Merging to combine similar regions

---

### "Prices seem wrong"
**Possible causes:**
- Outliers included
- Wrong calculation method
- Duplicates not removed

**Solutions:**
1. Check Distribution Analysis for outliers
2. Enable outlier filtering
3. Try switching calculation method
4. Check Counts tab (low counts = unreliable averages)

---

### "Export not working"
**Possible causes:**
- Tables not generated yet
- Browser blocking download

**Solutions:**
1. Ensure you've generated tables first
2. Check browser download settings/permissions
3. Try different browser (Chrome, Edge, Firefox)

---

## 💡 Pro Tips & Best Practices

### 📊 Analyzing Trends
- **Use MA3 or MA4** for smooth trend lines (removes seasonal noise)
- **Compare Index tabs** to see relative growth (not absolute prices)
- **Check Counts tab** - rising counts = growing market interest

### 🎯 Data Quality
- **Start with Distribution Analysis** before applying filters
- **Watch outlier %** - above 10% usually means noisy data
- **Check duplicate stats** in Summary - real estate data often has duplicates

### ⚡ Performance
- **Apply filters BEFORE generating tables** (faster processing)
- **Use fewer regions** for quicker interactive charts
- **Reset filters** if app becomes slow

### 📈 Comparisons
- **Use Index tabs** to compare regions with different price levels
- **Merge regions** for macro trend analysis
- **Filter by property type** to compare apples-to-apples

### 💾 Exporting
- **Export after filtering** to save cleaned data
- **Timestamp files** help track analysis versions
- **Export before and after** outlier removal for comparison

---

## 📖 Key Concepts Explained

### Moving Averages (MA)
**Simple explanation:** Average of multiple quarters

**Example:**
- **Q1 price:** €1000/m²
- **Q2 price:** €1100/m²
- **2-quarter MA for Q2:** (€1000 + €1100) / 2 = €1050/m²

**Why use them?**
- Smooth out random fluctuations
- See clearer trends
- Reduce seasonal effects

---

### Price Index
**Simple explanation:** Shows % change from starting point

**Example:**
- **2020-Q1 (base):** €1000/m² → Index = 1.0
- **2022-Q1:** €1300/m² → Index = 1.3 (30% increase)
- **2024-Q1:** €1450/m² → Index = 1.45 (45% increase)

**Why use indices?**
- Compare regions: Riga vs Liepaja both start at 1.0
- Track growth: Which region grew fastest?
- Remove baseline price differences

---

### Outliers
**Simple explanation:** Values far from average

**Examples of outliers:**
- €5/m² (probably data error)
- €5000/m² (luxury penthouse or error)
- Transaction of 500m² apartment (unusual)

**IQR Method:**
- Q1 = 25th percentile, Q3 = 75th percentile
- IQR = Q3 - Q1
- Outliers = outside [Q1 - 1.5×IQR, Q3 + 1.5×IQR]

---

### Statistical Terms

| Term | Meaning | Example |
|------|---------|---------|
| **Mean** | Average | All prices added up ÷ count |
| **Median** | Middle value | 50% above, 50% below |
| **Std Dev** | Spread of data | High = prices vary a lot |
| **Q1** | 25th percentile | 25% of prices below this |
| **Q3** | 75th percentile | 75% of prices below this |
| **IQR** | Q3 - Q1 | Range of middle 50% |

---

## 🎓 Learning Path

### Level 1: Beginner (First Session)
- [ ] Start the application
- [ ] Switch between Houses and Apartments
- [ ] Generate tables with default settings
- [ ] Explore Summary and Index tabs
- [ ] Select regions in chart
- [ ] Export to Excel

**Goal:** Get comfortable with basic interface

---

### Level 2: Intermediate (After 2-3 sessions)
- [ ] Apply date range filters
- [ ] Select specific regions
- [ ] Compare different moving averages
- [ ] Understand Index vs Prices tabs
- [ ] Check transaction counts
- [ ] Use date range slider on charts

**Goal:** Perform targeted analysis with filters

---

### Level 3: Advanced (Ongoing)
- [ ] Use outlier detection
- [ ] Remove duplicates
- [ ] Analyze distribution statistics
- [ ] Merge regions
- [ ] Compare filtered vs unfiltered indices
- [ ] Apply multiple complex filters
- [ ] Interpret all statistical measures

**Goal:** Clean data and perform research-grade analysis

---

## ❓ Frequently Asked Questions

### General Questions

**Q: Can I use this for investment decisions?**  
A: This tool provides data analysis only. Always consult real estate professionals and conduct thorough due diligence before investing.

**Q: How often is the data updated?**  
A: The data comes from the CSV files in the project. Update those files to get new data (ask your administrator).

**Q: Can I analyze multiple property types at once?**  
A: Not directly, but you can:
1. Analyze Houses → Export to Excel
2. Switch to Apartments → Export to Excel
3. Compare the Excel files

**Q: What's the difference between Sold Area and Total Area?**  
A: 
- **Sold Area**: The specific area being sold/bought
- **Total Area**: Complete property size (might include shared spaces)

---

### Technical Questions

**Q: Why are some quarters missing in my table?**  
A: No transactions matched your filters for that quarter. Check Counts tab or loosen filters.

**Q: What if I want to use a different base year than 2020-Q1?**  
A: Currently fixed at 2020-Q1. Contact developer for customization.

**Q: Can I save my filter settings?**  
A: Not currently. Write down your filter settings or export results to preserve your analysis.

**Q: The app is slow with all regions selected. Why?**  
A: More data = more processing. Try:
- Narrow date range
- Fewer regions
- Reduce filters
- Use Region Merging for aggregate view

---

### Data Questions

**Q: What does "excluded from price calculations" mean?**  
A: Some transactions lack required data (price or area), so they're counted but not used for price averages.

**Q: Should I always remove outliers?**  
A: Not always:
- **Keep outliers** if analyzing full market (including extremes)
- **Remove outliers** for typical market trends and indices

**Q: How do I know if my outlier settings are good?**  
A: Check Distribution Analysis tab:
- **Before:** Note outlier percentage
- **After:** Should be <5% for stable regions
- **Use Index comparison** feature to see impact

---

## 🆘 Getting Help

### Resources
1. **This guide** - Comprehensive reference
2. **App interface** - Hover tooltips (ℹ️ icons)
3. **Workflow Guide** (`WORKFLOW_GUIDE.md`) - Git deployment
4. **README** - Technical setup

### Support Workflow
1. Check relevant section in this guide
2. Try Troubleshooting section
3. Review FAQ
4. Note your filters and error messages
5. Contact administrator with details

---

## 📝 Glossary

| Term | Definition |
|------|------------|
| **MA** | Moving Average - smoothed average over multiple periods |
| **Q1, Q2, Q3, Q4** | Quarters of the year (Jan-Mar, Apr-Jun, Jul-Sep, Oct-Dec) |
| **EUR/m²** | Euros per square meter - standard property price metric |
| **IQR** | Interquartile Range - statistical measure of spread |
| **Index** | Normalized value showing change from base period (2020-Q1 = 1.0) |
| **Outlier** | Data point far from the normal range |
| **Duplicate** | Transaction recorded multiple times in dataset |
| **Aggregate** | Combined/summed data from multiple sources |
| **Percentile** | Value below which a percentage of data falls |
| **Region** | Geographic area (e.g., Riga, Kurzeme, Vidzeme) |
| **Municipality** | Smaller administrative unit within a region |
| **Dom_Parts** | Property parts/shares (e.g., 1/1 = whole property) |

---

## 🎉 Next Steps

Now that you've read the guide:

### For First-Time Users:
1. ✅ Start the app (`start_webapp.bat`)
2. ✅ Follow "Quick Start" (top of this guide)
3. ✅ Try "Workflow 1: Market Overview"
4. ✅ Explore one feature at a time

### For Regular Users:
1. ✅ Try an advanced workflow
2. ✅ Experiment with outlier detection
3. ✅ Create merged regions for your market segments
4. ✅ Export and share your analysis

### For Power Users:
1. ✅ Develop your own custom workflows
2. ✅ Compare multiple filter scenarios
3. ✅ Automate recurring analyses
4. ✅ Provide feedback for new features

---

**Happy analyzing! 📊🏠**

*Last updated: November 6, 2024*  
*Version: 1.0*

---

## 📮 Appendix: Sample Analysis Report Structure

Use this template when sharing your findings:

```
# Real Estate Analysis Report
Date: [Date]
Analyst: [Your Name]

## Executive Summary
- Property Type: [Houses/Apartments]
- Time Period: [YYYY-QX to YYYY-QX]
- Regions Analyzed: [List]
- Key Finding: [One sentence]

## Methodology
- Data Source: Latvian Real Estate Transaction Data
- Filters Applied: [List all filters]
- Outlier Treatment: [Yes/No, Method]
- Duplicate Handling: [Method used]

## Key Findings
1. [Finding 1 + supporting data]
2. [Finding 2 + supporting data]
3. [Finding 3 + supporting data]

## Regional Comparison
[Table or chart from Index tab]

## Price Trends
[Chart from Prices MA4 tab]

## Data Quality Notes
- Total Transactions: [Number]
- Outliers Removed: [Number (%)]
- Data Coverage: [% with complete data]

## Recommendations
[Based on your analysis]

## Appendix
- Attached: Excel export with full data
```

---

*Need more help? Check the tooltips in the app (ℹ️ icons) or refer to specific sections of this guide!*


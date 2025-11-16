# Lithuania Analyzer - Quick Start Guide

## 🚀 Getting Started (3 Steps)

### 1. Launch the App
```bash
streamlit run app.py
```

### 2. Select Lithuania Dataset
- At the top of the page, click **"Lithuania - Bigbank Statistics"**

### 3. Choose Property Type
- Select from: **Apartments**, **Houses**, **Office**, **Retail**, or **Hotel/Recreation**

## 📊 Quick Analysis (60 seconds)

1. **Select filters** (sidebar):
   - Choose price metric: **Weighted Average** (recommended)
   - Select regions: **Keep "All" checked** for overview
   - Set base period: **2021-Q1** (default)

2. **Generate analysis**:
   - Click **"🚀 Generate Analysis"** button
   - Wait 2-3 seconds

3. **Explore tabs**:
   - **Summary**: Overview statistics
   - **Prices**: Price trends by region
   - **Index**: Normalized price changes
   - **Regional Comparison**: Compare all regions

## 🎯 Common Use Cases

### Compare Cities
```
1. Go to "Regional Comparison" tab
2. View bar chart of price changes
3. Focus on regions 1 (Vilnius), 3 (Kaunas), 5 (Klaipėda)
```

### Track Market Trends
```
1. Go to "Prices" tab
2. Select 2-3 regions in multiselect
3. Adjust date range slider
4. Observe price trajectory
```

### Calculate Growth Rate
```
1. Go to "Price Index" tab
2. Note latest value for each region
3. Formula: (Index - 1.0) × 100 = % change
   Example: Index = 1.25 → 25% increase since base period
```

## 📥 Export Report

1. Scroll to bottom after generating analysis
2. Click **"📊 Generate Excel Report"**
3. Click **"⬇️ Download Excel Report"**
4. File includes: Prices, Counts, Index, Metadata

## 🔍 Key Regions

| Region | What It Is | Typical Use |
|--------|-----------|-------------|
| 1 | Vilnius City | Capital, highest volume |
| 3 | Kaunas City | 2nd largest city |
| 5 | Klaipėda | Port city |
| 7 | Neringa | Resort (high prices, low volume) |
| 12 | Other municipalities | Rural/small towns |

## 💡 Pro Tips

1. **Use Weighted Average** - Most accurate representation of market
2. **Check Transaction Counts** - More transactions = more reliable data
3. **Resort Areas Are Different** - Regions 7 & 10 have unique dynamics
4. **Compare Same Quarters** - Q2 2023 vs Q2 2024 for seasonality
5. **Office & Retail Have Less Data** - Only 72 quarters vs 216 for apartments

## ⚡ Keyboard Shortcuts

- **Ctrl+R** or **F5**: Refresh app
- **Esc**: Close dropdowns
- Click region names in legend to hide/show lines in charts

## 🆘 Troubleshooting

**No data appears**: Ensure Excel file is in same directory as app.py

**Charts are empty**: Reset filters or choose more regions

**Slow performance**: Reduce number of selected regions

## 📖 More Info

See `LITHUANIA_ANALYZER_GUIDE.md` for comprehensive documentation

---

**Data Source**: Bigbank Lithuania Q2 2025 Transaction Statistics
**Property Types**: Apartments, Houses, Office, Retail, Hotel/Recreation
**Regions**: 12 municipality groups
**Time Coverage**: 2021-Q1 to 2025-Q4 (varies by property type)


# Lithuania Real Estate Price Index Analyzer - User Guide

## Overview

The Lithuania Real Estate Price Index Analyzer provides comprehensive analysis of Lithuanian real estate market data from Bigbank's quarterly transaction statistics. This analyzer has been integrated into the Baltic Real Estate Price Index Analyzer webapp.

## Data Source

**File:** `Bigbank_purchase transaction statistics_202506 Lithuania EN.xlsx`

**Coverage:** Q2 2025 Lithuania real estate transaction statistics

**Publisher:** Bigbank

## Features

### 1. Property Types Supported

- **Apartments** - Residential premises (10-200 sq.m, 80%+ completion)
- **Houses** - Single-family residential buildings (50-500 sq.m)
- **Office Premises** - Administrative purpose properties
- **Retail Premises** - Commercial retail spaces (30-500 sq.m)
- **Hotel/Recreation** - Hotels and public/private recreation (20-200 sq.m)

### 2. Regional Coverage

The analyzer tracks 12 municipality regions across Lithuania:

| Region | Description |
|--------|-------------|
| 1 region | Vilnius City (highest prices, most transactions) |
| 2 region | Vilnius District |
| 3 region | Kaunas City |
| 4 region | Kaunas District |
| 5 region | Klaipėda City & District |
| 6 region | Palanga City |
| 7 region | Neringa (resort area, very high prices) |
| 8 region | Alytus, Šiauliai, Panevėžys cities |
| 9 region | Trakai District |
| 10 region | Druskininkai, Birštonas (resort municipalities) |
| 11 region | 7 mid-sized municipalities combined |
| 12 region | All other municipalities |

### 3. Price Metrics Available

- **Weighted Average Price per m²** (Recommended) - Accounts for transaction sizes
- **Arithmetic Average Price per m²** - Simple average of all transactions
- **Median Price per m²** - Middle value, less affected by outliers

### 4. Data Columns

Each property type includes the following quarterly data:

- **Year** - Transaction year
- **Quarter** - Transaction quarter (Q1-Q4)
- **Count** - Number of transactions
- **Acquired Area (sq.m)** - Total area transacted
- **Price per sqm_avg** - Arithmetic average price per square meter
- **Price per sqm_weighted** - Weighted average price per square meter
- **Price per sqm_median** - Median price per square meter

## How to Use

### Access the Analyzer

1. Run the webapp: `streamlit run app.py`
2. At the top of the page, select **"Lithuania - Bigbank Statistics"**
3. Choose your property type from the horizontal menu

### Filtering Options

#### Sidebar Filters

**Price Metric**
- Select your preferred price calculation method
- Recommended: Weighted Average for more accurate market representation

**Date & Time**
- Year range slider to focus on specific years
- Quarter selector to include/exclude specific quarters

**Regions**
- Select all regions or choose specific ones
- Regions are listed with descriptive names

**Index Base Period**
- Set the base year and quarter for index calculations
- Default: 2021-Q1 (first available quarter)
- Index shows price changes relative to base period (base = 1.0)

**Transaction Filter**
- Set minimum transaction threshold per quarter
- Useful to exclude periods with insufficient data

### Analysis Tabs

#### 1. Summary Tab
- Overview statistics for each region
- Total transactions and acquired area
- Latest vs. all-time average prices
- Price change percentages
- Key insights section with aggregate metrics

#### 2. Prices Tab
- Interactive price per m² table by region and quarter
- Line chart showing price trends over time
- Region selector for chart
- Date range slider for focused analysis

#### 3. Transaction Counts Tab
- Number of transactions by region and quarter
- Helps identify market activity levels
- Interactive visualization

#### 4. Total Area Tab
- Total acquired area (m²) by region and quarter
- Shows market volume trends
- Useful for understanding transaction sizes

#### 5. Price Index Tab
- Normalized price index relative to base period
- Base period = 1.0
- Values > 1.0 indicate price increase
- Values < 1.0 indicate price decrease
- Interactive chart with region and date filtering

#### 6. Regional Comparison Tab
- Side-by-side comparison of all regions
- Initial vs. latest prices
- Absolute and percentage changes
- Bar chart visualization of price changes
- Sorted by performance

### Export Functionality

**Excel Report Generation**
- Click "Generate Excel Report" button
- Downloads comprehensive Excel file with:
  - Prices table
  - Transaction counts table
  - Price index table
  - Metadata sheet with report details
- File naming: `lithuania_{property_type}_index_report_{timestamp}.xlsx`

## Data Quality Notes

### Strengths
- **Official Source**: Data from Bigbank, a reputable financial institution
- **Comprehensive**: Covers all major property types and regions
- **Multiple Metrics**: Three different price calculation methods
- **Quarterly Updates**: Regular reporting cadence
- **Pre-aggregated**: Data already cleaned and aggregated

### Limitations
- **Aggregated Data**: Individual transactions not available
- **Limited History**: Data coverage varies by property type
  - Apartments: 216 quarters of data
  - Houses: 203 quarters
  - Office: 72 quarters (sparser coverage)
  - Retail: 72 quarters
  - Hotel/Recreation: 84 quarters
- **Regional Grouping**: Some regions are combined (e.g., region 11 = 7 municipalities)
- **No Property Details**: No information on specific property characteristics

## Best Practices

### For Market Analysis
1. Use **weighted average** price metric for most accurate market representation
2. Set **minimum transaction threshold** (e.g., 10+) to exclude low-volume quarters
3. Focus on **major cities** (regions 1, 3, 5) for most reliable trends
4. Compare **index values** rather than absolute prices when analyzing trends

### For Regional Comparison
1. Consider **transaction volumes** when comparing regions
2. Resort areas (regions 7, 10) may show higher prices but lower volumes
3. Use **percentage changes** rather than absolute values for fair comparisons

### For Temporal Analysis
1. Compare **same quarters** year-over-year to account for seasonality
2. Use **base period** from stable market conditions
3. Watch for **outlier quarters** with very low transaction counts

## Technical Details

### File Structure
- **Module**: `lithuania_analyzer.py`
- **Integration**: Imported into `app.py` via dataset selector
- **Caching**: Uses Streamlit's `@st.cache_data` for performance
- **Dependencies**: pandas, plotly, streamlit, openpyxl

### Session State Variables
When analysis is generated, the following are stored:
- `lt_prices_df` - Prices pivot table
- `lt_counts_df` - Transaction counts pivot table
- `lt_area_df` - Total area pivot table
- `lt_index_df` - Price index table
- `lt_df_filtered` - Filtered raw data
- `lt_region_map` - Region name mapping
- `lt_property_type` - Selected property type
- `lt_price_metric` - Selected price metric
- `lt_base_year`, `lt_base_quarter` - Index base period

## Comparison with Latvian Analyzer

| Feature | Latvia Analyzer | Lithuania Analyzer |
|---------|----------------|-------------------|
| Data Granularity | Transaction-level | Pre-aggregated quarterly |
| Property Types | 5 types | 5 types |
| Regions | Variable | 12 fixed regions |
| Filters | 15+ detailed filters | 5 essential filters |
| Outlier Detection | Yes | No (pre-cleaned) |
| Duplicate Removal | Yes | No (pre-aggregated) |
| Moving Averages | Yes (2-4 quarters) | No |
| Price Metrics | Calculated + column | 3 pre-calculated |
| Export Options | Multi-sheet Excel | Single Excel |

## Troubleshooting

### "Failed to load data" Error
- Ensure `Bigbank_purchase transaction statistics_202506 Lithuania EN.xlsx` is in the project directory
- Check Excel file has not been renamed or moved
- Verify file contains expected sheet names

### Empty Charts
- Check if selected regions have data for chosen time period
- Verify transaction count threshold isn't too high
- Try resetting filters

### Slow Performance
- Reduce number of selected regions
- Narrow date range
- Clear session state and regenerate analysis

## Future Enhancements

Potential improvements for future versions:
1. Add quarter-over-quarter change calculations
2. Include year-over-year comparison tables
3. Add seasonality analysis
4. Implement forecasting models
5. Add more export formats (CSV, JSON)
6. Create combined Baltic region analysis
7. Add data validation checks

## Support & Feedback

For issues, questions, or feature requests related to the Lithuania analyzer:
- Check this guide first
- Review the main `DOCUMENTATION_INDEX.md`
- Consult the `INTERFACE_GUIDE.md` for general webapp usage
- Check `WORKFLOW_GUIDE.md` for data processing workflows

## Version History

**v1.0 (2024-11-16)**
- Initial release
- Support for all 5 property types
- 12 regions coverage
- 3 price metrics
- 6 analysis tabs
- Excel export functionality
- Full integration with main webapp


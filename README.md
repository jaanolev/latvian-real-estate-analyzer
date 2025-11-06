# 🏠 Latvian Real Estate Price Index Analyzer

A beautiful and intuitive web application for analyzing Latvian house and apartment price indices with interactive visualizations, filtering, and export capabilities.

## 📋 Features

- **Property Type Toggle**: Seamlessly switch between Houses and Apartments analysis
- **Calculation Methods**: Choose between calculated price/area or use existing Total_EUR_m2 column
- **Advanced Filtering**: Filter data by date range, year, region, type, municipality, and category
- **Multiple Analysis Views**:
  - Summary statistics
  - 4 Price tables (Original, 2-quarter MA, 3-quarter MA, 4-quarter MA)
  - Transaction counts by region and quarter
  - 4 Index tables (base: 2020-Q1 = 1.0)
- **Interactive Plots**: Line charts with region selection and date range sliders
- **Region Merging**: Combine multiple regions to analyze aggregated trends
- **Excel Export**: Download comprehensive reports with all tables

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. Open PowerShell and navigate to the project directory:
```powershell
cd C:\Users\annik\Documents\LV_index_interface
```

2. Install required packages:
```powershell
pip install -r requirements.txt
```

### Running the Application

Launch the Streamlit app:
```powershell
streamlit run app.py
```

The app will automatically open in your default web browser at `http://localhost:8501`

## 📊 How to Use

1. **Select Property Type**: Choose between Houses or Apartments at the top of the page
2. **Choose Calculation Method**: Select between calculated or Total_EUR_m2 in the sidebar
3. **Apply Filters**: Use the sidebar to filter your data by various criteria
4. **Generate Tables**: Click the "🚀 Generate Tables" button to calculate all metrics
5. **Explore Tabs**: Navigate through different tabs to view:
   - Summary statistics
   - Price trends with different moving averages
   - Transaction counts
   - Price indices
6. **Interactive Plots**: 
   - Select regions to display
   - Adjust date ranges with the slider
7. **Merge Regions**: Use the standalone merge section to combine multiple regions
8. **Export Results**: Generate and download an Excel report with all tables

## 📈 Data Processing

- **Prices**: Calculated as average Price_EUR per Sold_Area_m2 by region and quarter
- **Moving Averages**: Applied across quarters (2, 3, or 4 quarter windows)
- **Indices**: Normalized to 2020-Q1 = 1.0 for all regions
- **Counts**: Total number of transactions by region and quarter

## 🎨 Built With

- **Streamlit**: Web framework
- **Pandas**: Data manipulation
- **Plotly**: Interactive visualizations
- **OpenPyXL**: Excel export

## 💡 Tips

- Start with a broad filter to see overall trends
- Use moving averages to smooth out seasonal variations
- Compare regions using the interactive plots
- Merge regions to analyze larger market segments
- Export results for further analysis or reporting

---

Made with ❤️ for Latvian real estate analysis


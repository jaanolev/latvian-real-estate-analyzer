import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import io

# Page config
st.set_page_config(
    page_title="Baltic Real Estate Price Index Analyzer",
    page_icon="🏠",
    layout="wide"
)

# Helper functions
def clean_numeric_column(series):
    """Clean numeric columns that have spaces as thousand separators"""
    if series.dtype == 'object':
        return pd.to_numeric(series.str.replace(' ', '').str.replace(',', '.'), errors='coerce')
    return series

@st.cache_data
def load_data(property_type='Houses'):
    """Load and prepare the dataset"""
    # Select the appropriate CSV file
    if property_type == 'Apartments':
        filename = 'LV_apartments_merged_mapped_unfiltered.csv'
        df = pd.read_csv(filename, index_col=0)
    elif property_type == 'Agricultural land':
        filename = 'LV_agriland_merged_mapped_unfiltered.csv'
        df = pd.read_csv(filename, index_col=0)
    elif property_type == 'Forest land':
        filename = 'LV_forestland_merged_mapped_unfiltered.csv'
        df = pd.read_csv(filename, index_col=0)
    elif property_type == 'Other land':
        filename = 'OTHER_LAND_NEW_data_merged_processed_20251119_122634.csv'
        df = pd.read_csv(filename)  # No index_col
    elif property_type == 'Land commercial':
        filename = 'Land_commercial_merged_processed_20251117.csv'
        df = pd.read_csv(filename)  # No index_col
    elif property_type == 'Land residential':
        filename = 'Land_residental_data_merged_processed_20251117_030224.csv'
        df = pd.read_csv(filename)  # No index_col
    elif property_type == 'Premises':
        filename = 'Premises_all_data_merged_processed_20251117_004724.csv'
        df = pd.read_csv(filename)  # No index_col for Premises
    else:
        filename = 'LV_houses_merged_mapped_unfiltered.csv'
        df = pd.read_csv(filename, index_col=0)
    
    # Clean numeric columns
    numeric_cols = ['Sold_Area_m2', 'Total_Area_m2', 'Price_EUR', 'Total_EUR_m2', 'Land_EUR_m2', 'Interior_Area_m2']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = clean_numeric_column(df[col])
    
    # Convert Date column - automatically detect format
    # Try YYYY-MM-DD first, then fall back to auto-detection
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce', format='%Y-%m-%d')
    # For any that failed, try auto-detection
    mask = df['Date'].isna()
    if mask.sum() > 0:
        df.loc[mask, 'Date'] = pd.to_datetime(df.loc[mask, 'Date'], errors='coerce')
    
    # Update Quarter and Year from the parsed Date
    df['Quarter'] = df['Date'].dt.quarter
    df['Year'] = df['Date'].dt.year
    
    # Remove rows with invalid dates, years, or quarters
    df = df.dropna(subset=['Date'])
    df = df[(df['Year'] > 0) & (df['Quarter'] >= 1) & (df['Quarter'] <= 4)]
    
    return df

def calculate_price_per_m2(df, use_total_eur_m2=False, property_type='Houses'):
    """Calculate price per square meter"""
    df = df.copy()
    if use_total_eur_m2:
        # For Agricultural land, Forest land, Land commercial, Land residential, and Other land, use Land_EUR_m2, otherwise use Total_EUR_m2
        if property_type in ['Agricultural land', 'Forest land', 'Land commercial', 'Land residential', 'Other land']:
            if 'Land_EUR_m2' in df.columns:
                df['Price_per_m2'] = df['Land_EUR_m2']
            else:
                # Fallback to calculated method if column doesn't exist
                df['Price_per_m2'] = df['Price_EUR'] / df['Sold_Area_m2']
        else:
            if 'Total_EUR_m2' in df.columns:
                df['Price_per_m2'] = df['Total_EUR_m2']
            else:
                # Fallback to calculated method if column doesn't exist
                df['Price_per_m2'] = df['Price_EUR'] / df['Sold_Area_m2']
    else:
        # Calculate from Price_EUR / Sold_Area_m2
        df['Price_per_m2'] = df['Price_EUR'] / df['Sold_Area_m2']
    return df

def aggregate_by_region_quarter(df, use_total_eur_m2=False, property_type='Houses'):
    """Aggregate data by region and quarter"""
    df = calculate_price_per_m2(df, use_total_eur_m2, property_type)
    
    # Filter out invalid Year and Quarter values
    df = df[(df['Year'] > 0) & (df['Year'].notna()) & 
            (df['Quarter'] >= 1) & (df['Quarter'] <= 4) & (df['Quarter'].notna())].copy()
    
    # Filter out rows with missing data based on calculation method
    if use_total_eur_m2:
        # When using Total_EUR_m2 or Land_EUR_m2, only require the appropriate column to be valid
        if property_type in ['Agricultural land', 'Forest land', 'Land commercial', 'Land residential', 'Other land']:
            if 'Land_EUR_m2' in df.columns:
                df = df[df['Land_EUR_m2'].notna() & (df['Land_EUR_m2'] > 0)].copy()
            else:
                # Fallback to calculated method requirements
                df = df[df['Price_EUR'].notna() & df['Sold_Area_m2'].notna() & (df['Sold_Area_m2'] > 0)].copy()
        else:
            if 'Total_EUR_m2' in df.columns:
                df = df[df['Total_EUR_m2'].notna() & (df['Total_EUR_m2'] > 0)].copy()
            else:
                # Fallback to calculated method requirements
                df = df[df['Price_EUR'].notna() & df['Sold_Area_m2'].notna() & (df['Sold_Area_m2'] > 0)].copy()
    else:
        # When calculating, require both Price_EUR and Sold_Area_m2 to be valid
        df = df[df['Price_EUR'].notna() & df['Sold_Area_m2'].notna() & (df['Sold_Area_m2'] > 0)].copy()
    
    # Convert to int to avoid float issues
    df['Year'] = df['Year'].astype(int)
    df['Quarter'] = df['Quarter'].astype(int)
    
    # Create year-quarter identifier
    df['YearQuarter'] = df['Year'].astype(str) + '-Q' + df['Quarter'].astype(str)
    df['YearQuarterDate'] = pd.to_datetime(df['Year'].astype(str) + '-' + (df['Quarter']*3).astype(str) + '-01')
    
    if use_total_eur_m2:
        # When using Total_EUR_m2, aggregate differently
        agg_df = df.groupby(['region_riga_separate', 'Year', 'Quarter', 'YearQuarter', 'YearQuarterDate']).agg({
            'Price_EUR': 'sum',
            'Sold_Area_m2': 'sum',
            'Price_per_m2': 'mean'  # Average the Total_EUR_m2 values
        }).reset_index()
        agg_df['Avg_Price_per_m2'] = agg_df['Price_per_m2']
    else:
        # Aggregate using calculated method
        agg_df = df.groupby(['region_riga_separate', 'Year', 'Quarter', 'YearQuarter', 'YearQuarterDate']).agg({
            'Price_EUR': 'sum',
            'Sold_Area_m2': 'sum',
            'Price_per_m2': 'mean'
        }).reset_index()
        agg_df['Avg_Price_per_m2'] = agg_df['Price_EUR'] / agg_df['Sold_Area_m2']
    
    return agg_df

def create_prices_table(agg_df, ma_quarters=1):
    """Create prices pivot table with optional moving average"""
    pivot = agg_df.pivot_table(
        index='region_riga_separate',
        columns='YearQuarter',
        values='Avg_Price_per_m2',
        aggfunc='first'
    )
    
    if ma_quarters > 1:
        # Apply moving average across columns
        pivot = pivot.rolling(window=ma_quarters, axis=1, min_periods=1).mean()
    
    return pivot

def create_counts_table(df):
    """Create counts pivot table"""
    counts = df.groupby(['region_riga_separate', 'Year', 'Quarter']).size().reset_index(name='Count')
    # Convert to int first to avoid float notation (e.g., "2024.0-Q1.0")
    counts['Year'] = counts['Year'].astype(int)
    counts['Quarter'] = counts['Quarter'].astype(int)
    counts['YearQuarter'] = counts['Year'].astype(str) + '-Q' + counts['Quarter'].astype(str)
    
    pivot = counts.pivot_table(
        index='region_riga_separate',
        columns='YearQuarter',
        values='Count',
        aggfunc='sum',
        fill_value=0
    )
    
    return pivot

def create_index_table(prices_pivot, property_type='Houses', base_year=None, base_quarter=None):
    """Create index table with base period (2020-Q1 for Houses/Apartments, 2021-Q1 for Agricultural/Land commercial/Other land, 2022-Q2 for Premises, 2023-Q2 for Forest/Land residential)"""
    # Set default base period based on property type
    if base_year is None or base_quarter is None:
        if property_type in ['Forest land', 'Land residential']:
            base_year = 2023
            base_quarter = 2
        elif property_type in ['Agricultural land', 'Land commercial', 'Other land']:
            base_year = 2021
            base_quarter = 1
        elif property_type == 'Premises':
            base_year = 2022
            base_quarter = 2
        else:
            base_year = 2020
            base_quarter = 1
    
    base_col = f'{base_year}-Q{base_quarter}'
    
    if base_col not in prices_pivot.columns:
        st.warning(f"Base period {base_col} not found in data")
        return prices_pivot
    
    # Divide all values by the base column
    index_df = prices_pivot.div(prices_pivot[base_col], axis=0)
    
    return index_df

def detect_outliers(df, use_total_eur_m2=False, method="IQR Method (1.5x - standard)", 
                   lower_percentile=None, upper_percentile=None,
                   per_region=False, per_quarter=False, property_type='Houses'):
    """
    Detect outliers based on price per m² distribution
    Returns a boolean mask where True = keep, False = outlier (remove)
    """
    # Calculate price per m²
    df_temp = calculate_price_per_m2(df.copy(), use_total_eur_m2, property_type)
    
    # Determine which price column to use
    # Always use Price_per_m2 which calculate_price_per_m2 creates (with fallback logic)
    price_col = 'Price_per_m2'
    
    # Initialize mask - True = keep, False = remove
    # Start by keeping everything
    keep_mask = pd.Series(True, index=df.index)
    
    # Only apply outlier detection to rows with valid prices
    valid_price_mask = df_temp[price_col].notna() & (df_temp[price_col] > 0)
    
    # If no valid prices, return keep_mask as is
    if not valid_price_mask.any():
        return keep_mask
    
    # Determine grouping
    if per_region and per_quarter:
        groups = df_temp[valid_price_mask].groupby(['region_riga_separate', 'Year', 'Quarter'])
    elif per_region:
        groups = df_temp[valid_price_mask].groupby(['region_riga_separate'])
    elif per_quarter:
        groups = df_temp[valid_price_mask].groupby(['Year', 'Quarter'])
    else:
        # Global outlier detection - only on valid prices
        groups = [('global', df_temp[valid_price_mask])]
    
    # Apply outlier detection to each group
    for group_key, group_df in groups:
        group_prices = group_df[price_col]
        
        if len(group_prices) < 10:  # Skip if too few data points
            continue
        
        if "Percentile" in method:
            # Percentile method
            lower_bound = group_prices.quantile(lower_percentile / 100.0)
            upper_bound = group_prices.quantile(upper_percentile / 100.0)
        else:
            # IQR method
            Q1 = group_prices.quantile(0.25)
            Q3 = group_prices.quantile(0.75)
            IQR = Q3 - Q1
            
            if "3.0x" in method:
                multiplier = 3.0
            else:
                multiplier = 1.5
            
            lower_bound = Q1 - multiplier * IQR
            upper_bound = Q3 + multiplier * IQR
        
        # Identify outliers in this group (values outside bounds)
        is_outlier = (group_prices < lower_bound) | (group_prices > upper_bound)
        
        # Mark these as False (remove) in keep_mask
        keep_mask.loc[is_outlier[is_outlier].index] = False
    
    # Rows with invalid/missing prices are kept (True) - they'll be filtered later by aggregate function
    # This ensures we don't accidentally remove valid data
    
    return keep_mask

def plot_regions(df_pivot, title, yaxis_title, selected_regions=None, date_range=None):
    """Create interactive line plot"""
    fig = go.Figure()
    
    # Filter by selected regions
    if selected_regions:
        df_plot = df_pivot.loc[selected_regions]
    else:
        df_plot = df_pivot
    
    # Filter by date range if provided
    if date_range:
        cols_to_show = [col for col in df_plot.columns if date_range[0] <= col <= date_range[1]]
        df_plot = df_plot[cols_to_show]
    
    for region in df_plot.index:
        fig.add_trace(go.Scatter(
            x=df_plot.columns,
            y=df_plot.loc[region],
            mode='lines+markers',
            name=region,
            line=dict(width=2),
            marker=dict(size=6)
        ))
    
    fig.update_layout(
        title=title,
        xaxis_title='Quarter',
        yaxis_title=yaxis_title,
        hovermode='x unified',
        height=500,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    fig.update_xaxes(tickangle=45)
    
    return fig

def export_to_excel(summary_stats, prices_tabs, counts_tab, index_tabs):
    """Export all tables to Excel"""
    output = io.BytesIO()
    
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        # Write summary
        summary_stats.to_excel(writer, sheet_name='Summary')
        
        # Write prices tabs
        for i, df in enumerate(prices_tabs):
            sheet_name = f'Prices_Q{i+1}' if i > 0 else 'Prices_Original'
            df.to_excel(writer, sheet_name=sheet_name)
        
        # Write counts
        counts_tab.to_excel(writer, sheet_name='Counts')
        
        # Write index tabs
        for i, df in enumerate(index_tabs):
            sheet_name = f'Index_Q{i+1}' if i > 0 else 'Index_Original'
            df.to_excel(writer, sheet_name=sheet_name)
    
    return output.getvalue()

def show_final_indexes_master_view():
    """Display a master view of all final indexes with predefined regional aggregations"""
    st.subheader("📊 Final Indexes Master View - Predefined Regional Aggregations")
    st.caption("Official index categories combining specific statistical regions")
    
    # Property types to analyze
    property_types = {
        "Apartments": {"file": "LV_apartments_merged_mapped_unfiltered.csv", "base": "2020-Q1", "index_col": 0},
        "Houses": {"file": "LV_houses_merged_mapped_unfiltered.csv", "base": "2020-Q1", "index_col": 0},
        "Land residential": {"file": "Land_residental_data_merged_processed_20251117_030224.csv", "base": "2023-Q2", "index_col": None},
        "Premises": {"file": "Premises_all_data_merged_processed_20251117_004724.csv", "base": "2022-Q2", "index_col": None},
        "Land commercial": {"file": "Land_commercial_merged_processed_20251117.csv", "base": "2021-Q1", "index_col": None},
        "Agricultural land": {"file": "LV_agriland_merged_mapped_unfiltered.csv", "base": "2021-Q1", "index_col": 0},
        "Forest land": {"file": "LV_forestland_merged_mapped_unfiltered.csv", "base": "2023-Q2", "index_col": 0},
        "Other land": {"file": "OTHER_LAND_NEW_data_merged_processed_20251119_122634.csv", "base": "2021-Q1", "index_col": None}
    }
    
    # Define final index aggregations (matching the screenshot)
    final_indexes_config = {
        "LV FLATS": {
            "property_type": "Apartments",
            "indexes": [
                {"name": "LV FLATS RIGA", "regions": ["Rīga"]},
                {"name": "LV FLATS PIE-RIGA", "regions": ["Pierīga"]},
                {"name": "LV FLATS KURZEME + VIDZEME + LATGALE + ZEMGALE", "regions": ["Kurzeme", "Vidzeme", "Latgale", "Zemgale"]}
            ]
        },
        "LV HOUSES": {
            "property_type": "Houses",
            "indexes": [
                {"name": "LV HOUSES PIE-RIGA", "regions": ["Pierīga"]},
                {"name": "LV HOUSES KURZEME", "regions": ["Kurzeme"]},
                {"name": "LV HOUSES LATGALE + ZEMGALE + VIDZEME", "regions": ["Latgale", "Zemgale", "Vidzeme"]}
            ]
        },
        "LV RESIDENTIAL LAND": {
            "property_type": "Land residential",
            "indexes": [
                {"name": "LV RESIDENTIAL LAND RIGA", "regions": ["Rīga"]},
                {"name": "LV RESIDENTIAL LAND NON-RIGA", "regions": ["Pierīga", "Kurzeme", "Vidzeme", "Latgale", "Zemgale"]}
            ]
        },
        "LV COMMERCIAL PROPERTY": {
            "property_type": "Premises",
            "indexes": [
                {"name": "LV COMMERCIAL PROPERTY RIGA", "regions": ["Rīga"]},
                {"name": "LV COMMERCIAL PROPERTY NON-RIGA", "regions": ["Pierīga", "Kurzeme", "Vidzeme", "Latgale", "Zemgale"]}
            ]
        },
        "LV COMMERCIAL LAND": {
            "property_type": "Land commercial",
            "indexes": [
                {"name": "LV COMMERCIAL LAND", "regions": ["Rīga", "Pierīga", "Kurzeme", "Vidzeme", "Latgale", "Zemgale"]}
            ]
        },
        "LV AGRILAND": {
            "property_type": None,  # Special case - combines multiple property types
            "indexes": [
                {"name": "LV AGRILAND", "regions": ["Rīga", "Pierīga", "Kurzeme", "Vidzeme", "Latgale", "Zemgale"], "combine_property_types": ["Agricultural land", "Forest land", "Other land"]}
            ]
        }
    }
    
    # Settings
    st.markdown("### ⚙️ Analysis Settings")
    
    # Create tabs for settings organization
    settings_tabs = st.tabs(["🎯 Basic Settings", "🔍 Outlier Detection", "💰 Price Method", "📊 Display Options"])
    
    with settings_tabs[0]:
        col1, col2 = st.columns(2)
        
        with col1:
            selected_categories = st.multiselect(
                "Index Categories to Display",
                list(final_indexes_config.keys()),
                default=list(final_indexes_config.keys()),
                help="Select which index categories to include"
            )
        
        with col2:
            use_ma = st.checkbox("Use Moving Average", value=False, help="Apply moving average smoothing to indexes")
            if use_ma:
                ma_quarters = st.slider("Moving Average Window (Quarters)", 2, 4, 2)
            else:
                ma_quarters = 1
    
    with settings_tabs[1]:
        st.markdown("##### Outlier Detection Configuration")
        st.caption("Remove extreme values that might skew the index calculations")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            outlier_method = st.selectbox(
                "Detection Method",
                ["None", "IQR Method (1.5x - standard)", "IQR Method (3x - conservative)", "Percentile Method"],
                help="Choose how to detect and remove outliers"
            )
        
        with col2:
            if outlier_method == "Percentile Method":
                lower_percentile = st.slider("Lower Percentile", 0.0, 10.0, 1.0, 0.5)
                upper_percentile = st.slider("Upper Percentile", 90.0, 100.0, 99.0, 0.5)
            else:
                lower_percentile = None
                upper_percentile = None
        
        with col3:
            apply_per_region = st.checkbox("Apply per region", value=False, 
                                          help="Detect outliers within each region separately")
            apply_per_quarter = st.checkbox("Apply per quarter", value=False,
                                           help="Detect outliers within each quarter separately")
    
    with settings_tabs[2]:
        st.markdown("##### Price Calculation Method")
        st.caption("Choose how to calculate price per square meter")
        
        col1, col2 = st.columns(2)
        
        with col1:
            use_total_eur_m2_master = st.radio(
                "Calculation Method",
                ["Use Existing Column", "Calculated (Price ÷ Area)"],
                index=0,
                help="'Use Existing' uses Total_EUR_m2 or Land_EUR_m2 column (recommended). 'Calculated' divides Price by Sold Area."
            )
            use_calculated = (use_total_eur_m2_master == "Calculated (Price ÷ Area)")
        
        with col2:
            st.info(
                "**Use Existing Column** (Recommended): Uses Total_EUR_m2 (Flats/Houses/Premises) or Land_EUR_m2 (Land types)\n\n"
                "**Calculated**: Price_EUR ÷ Sold_Area_m2"
            )
    
    with settings_tabs[3]:
        st.markdown("##### Display Options")
        
        col1, col2 = st.columns(2)
        
        with col1:
            show_transaction_counts = st.checkbox("Show transaction counts", value=True,
                                                 help="Display number of transactions used in calculations")
        
        with col2:
            show_data_quality = st.checkbox("Show data quality metrics", value=False,
                                           help="Display outlier removal statistics and data coverage")
    
    if not selected_categories:
        st.warning("⚠️ Please select at least one index category")
        return
    
    # Generate button
    if st.button("🚀 Generate Final Indexes", type="primary", use_container_width=True):
        with st.spinner("Loading data and calculating final indexes..."):
            final_indexes = {}
            loaded_data_cache = {}  # Cache loaded dataframes
            transaction_counts = {}  # Store transaction counts per index
            data_quality_metrics = {}  # Store quality metrics per index
            
            # Calculate each final index
            for category in selected_categories:
                category_config = final_indexes_config[category]
                
                for index_config in category_config["indexes"]:
                    index_name = index_config["name"]
                    regions_to_combine = index_config["regions"]
                    
                    # Check if this combines multiple property types
                    combine_types = index_config.get("combine_property_types", None)
                    
                    try:
                        # Special handling for combined property types
                        if combine_types:
                            # Load and combine data from multiple property types
                            combined_df = []
                            base_period = None
                            
                            for prop_type in combine_types:
                                if prop_type not in loaded_data_cache:
                                    config = property_types[prop_type]
                                    
                                    # Load data
                                    if config["index_col"] is not None:
                                        df = pd.read_csv(config["file"], index_col=config["index_col"])
                                    else:
                                        df = pd.read_csv(config["file"])
                                    
                                    # Clean and prepare data
                                    numeric_cols = ['Sold_Area_m2', 'Total_Area_m2', 'Price_EUR', 'Total_EUR_m2', 'Land_EUR_m2', 'Interior_Area_m2']
                                    for col in numeric_cols:
                                        if col in df.columns:
                                            df[col] = clean_numeric_column(df[col])
                                    
                                    df['Date'] = pd.to_datetime(df['Date'], errors='coerce', format='%Y-%m-%d')
                                    df['Quarter'] = df['Date'].dt.quarter
                                    df['Year'] = df['Date'].dt.year
                                    df = df.dropna(subset=['Date'])
                                    df = df[(df['Year'] > 0) & (df['Quarter'] >= 1) & (df['Quarter'] <= 4)]
                                    
                                    loaded_data_cache[prop_type] = {
                                        'data': df,
                                        'base': config["base"]
                                    }
                                
                                # Get cached data and filter to regions
                                df_temp = loaded_data_cache[prop_type]['data'].copy()
                                df_temp = df_temp[df_temp['region_riga_separate'].isin(regions_to_combine)]
                                combined_df.append(df_temp)
                                
                                # Use the first property type's base period
                                if base_period is None:
                                    base_period = loaded_data_cache[prop_type]['base']
                            
                            # Concatenate all property types
                            df_filtered = pd.concat(combined_df, ignore_index=True)
                            
                            if len(df_filtered) == 0:
                                st.warning(f"⚠️ No data found for {index_name}")
                                continue
                            
                            # Relabel all as the index name
                            df_filtered['region_riga_separate'] = index_name
                            
                            # Use Agricultural land for property type (for calculation method)
                            prop_type = combine_types[0]
                            
                        else:
                            # Single property type handling
                            prop_type = index_config.get("property_type_override", category_config["property_type"])
                            
                            # Load data if not already cached
                            if prop_type not in loaded_data_cache:
                                config = property_types[prop_type]
                                
                                # Load data
                                if config["index_col"] is not None:
                                    df = pd.read_csv(config["file"], index_col=config["index_col"])
                                else:
                                    df = pd.read_csv(config["file"])
                                
                                # Clean and prepare data
                                numeric_cols = ['Sold_Area_m2', 'Total_Area_m2', 'Price_EUR', 'Total_EUR_m2', 'Land_EUR_m2', 'Interior_Area_m2']
                                for col in numeric_cols:
                                    if col in df.columns:
                                        df[col] = clean_numeric_column(df[col])
                                
                                df['Date'] = pd.to_datetime(df['Date'], errors='coerce', format='%Y-%m-%d')
                                df['Quarter'] = df['Date'].dt.quarter
                                df['Year'] = df['Date'].dt.year
                                df = df.dropna(subset=['Date'])
                                df = df[(df['Year'] > 0) & (df['Quarter'] >= 1) & (df['Quarter'] <= 4)]
                                
                                loaded_data_cache[prop_type] = {
                                    'data': df,
                                    'base': config["base"]
                                }
                            
                            # Get cached data
                            df = loaded_data_cache[prop_type]['data'].copy()
                            base_period = loaded_data_cache[prop_type]['base']
                            
                            # Filter to selected regions
                            df_filtered = df[df['region_riga_separate'].isin(regions_to_combine)].copy()
                            
                            if len(df_filtered) == 0:
                                st.warning(f"⚠️ No data found for {index_name}")
                                continue
                            
                            # Relabel all regions as the index name for aggregation
                            df_filtered['region_riga_separate'] = index_name
                        
                        # Store original count
                        original_count = len(df_filtered)
                        
                        # Apply outlier detection if enabled
                        if outlier_method != "None":
                            df_before_outliers = df_filtered.copy()
                            try:
                                df_filtered = detect_outliers(
                                    df_filtered,
                                    use_total_eur_m2=not use_calculated,
                                    method=outlier_method,
                                    lower_percentile=lower_percentile,
                                    upper_percentile=upper_percentile,
                                    per_region=apply_per_region,
                                    per_quarter=apply_per_quarter,
                                    property_type=prop_type
                                )
                                outliers_removed = original_count - len(df_filtered)
                            except Exception as e:
                                st.warning(f"⚠️ Outlier detection failed for {index_name}: {str(e)}. Proceeding without outlier removal.")
                                df_filtered = df_before_outliers
                                outliers_removed = 0
                        else:
                            outliers_removed = 0
                        
                        # Store transaction counts and quality metrics
                        transaction_counts[index_name] = {
                            'total': len(df_filtered),
                            'original': original_count,
                            'outliers_removed': outliers_removed,
                            'outlier_percentage': (outliers_removed / original_count * 100) if original_count > 0 else 0
                        }
                        
                        # Determine calculation method based on user selection
                        use_total_eur_m2 = not use_calculated
                        
                        # Aggregate and calculate index
                        agg_df = aggregate_by_region_quarter(df_filtered, use_total_eur_m2, prop_type)
                        prices_table = create_prices_table(agg_df, ma_quarters=ma_quarters)
                        
                        # Extract base period info
                        base_year = int(base_period.split('-')[0])
                        base_quarter = int(base_period.split('-Q')[1])
                        
                        index_table = create_index_table(prices_table, prop_type, base_year, base_quarter)
                        
                        # Store the index (it should have one row with the index_name)
                        if len(index_table) > 0:
                            final_indexes[index_name] = {
                                'index': index_table.loc[index_name] if index_name in index_table.index else index_table.iloc[0],
                                'category': category,
                                'base': base_period,
                                'regions': regions_to_combine
                            }
                        
                    except Exception as e:
                        import traceback
                        error_details = traceback.format_exc()
                        st.error(f"⚠️ Could not calculate {index_name}: {str(e)}")
                        with st.expander("Show error details"):
                            st.code(error_details)
                        continue
            
            if not final_indexes:
                st.error("❌ No indexes could be calculated. Please check your data files.")
                return
            
            # Store in session state
            st.session_state['final_indexes'] = final_indexes
            st.session_state['master_categories'] = selected_categories
            st.session_state['master_ma_quarters'] = ma_quarters
            st.session_state['transaction_counts'] = transaction_counts
            st.session_state['data_quality_metrics'] = data_quality_metrics
            st.session_state['show_transaction_counts'] = show_transaction_counts
            st.session_state['show_data_quality'] = show_data_quality
            st.session_state['outlier_method'] = outlier_method
            st.session_state['price_calculation_method'] = use_total_eur_m2_master
        
        # Success message with summary
        total_transactions = sum(tc['total'] for tc in transaction_counts.values())
        total_outliers = sum(tc['outliers_removed'] for tc in transaction_counts.values())
        st.success(f"✅ Calculated {len(final_indexes)} final indexes using {total_transactions:,} transactions ({total_outliers:,} outliers removed)")
    
    # Display results if available
    if 'final_indexes' in st.session_state:
        st.markdown("---")
        st.header("📈 Final Index Results")
        
        final_indexes = st.session_state['final_indexes']
        ma_quarters = st.session_state.get('master_ma_quarters', 1)
        transaction_counts = st.session_state.get('transaction_counts', {})
        show_transaction_counts = st.session_state.get('show_transaction_counts', True)
        show_data_quality = st.session_state.get('show_data_quality', False)
        outlier_method = st.session_state.get('outlier_method', 'None')
        price_method = st.session_state.get('price_calculation_method', 'Use Existing Column')
        
        # Create tabs
        tabs = st.tabs([
            "📊 All Final Indexes",
            "📈 By Category",
            "📉 Time Series",
            "📥 Export"
        ])
        
        # Tab 1: All Final Indexes
        with tabs[0]:
            st.subheader("All Final Indexes - Complete Overview")
            
            # Show analysis settings summary
            col_info1, col_info2, col_info3 = st.columns(3)
            with col_info1:
                st.caption(f"**Outlier Method:** {outlier_method}")
            with col_info2:
                st.caption(f"**Price Method:** {price_method}")
            with col_info3:
                st.caption(f"**Moving Average:** {ma_quarters}Q" if ma_quarters > 1 else "**Moving Average:** None")
            
            st.markdown("---")
            
            # Create comprehensive table
            index_data = []
            for index_name, index_info in final_indexes.items():
                index_series = index_info['index']
                latest_value = index_series.iloc[-1]
                latest_quarter = index_series.index[-1]
                
                row_data = {
                    'Index Name': index_name,
                    'Category': index_info['category'],
                    'Latest Quarter': latest_quarter,
                    'Latest Index': f"{latest_value:.4f}",
                    'Change vs Base': f"{(latest_value - 1.0) * 100:.1f}%",
                    'Base Period': index_info['base'],
                }
                
                # Add transaction counts if enabled
                if show_transaction_counts and index_name in transaction_counts:
                    row_data['Transactions'] = f"{transaction_counts[index_name]['total']:,}"
                
                # Add data quality metrics if enabled
                if show_data_quality and index_name in transaction_counts:
                    row_data['Outliers Removed'] = f"{transaction_counts[index_name]['outliers_removed']:,} ({transaction_counts[index_name]['outlier_percentage']:.1f}%)"
                
                row_data['Regions'] = ', '.join(index_info['regions'])
                index_data.append(row_data)
            
            display_df = pd.DataFrame(index_data)
            st.dataframe(display_df, use_container_width=True, hide_index=True)
            
            # Summary statistics
            st.markdown("#### 📊 Summary Statistics")
            col1, col2, col3, col4 = st.columns(4)
            
            # Convert 'Latest Index' to float for calculations (it's now a string)
            latest_index_values = [float(val) for val in display_df['Latest Index']]
            
            with col1:
                max_idx = max(latest_index_values)
                st.metric("Highest Index", f"{max_idx:.3f}", 
                         f"+{(max_idx - 1.0) * 100:.1f}%")
            
            with col2:
                min_idx = min(latest_index_values)
                st.metric("Lowest Index", f"{min_idx:.3f}",
                         f"{(min_idx - 1.0) * 100:.1f}%")
            
            with col3:
                avg_idx = sum(latest_index_values) / len(latest_index_values)
                st.metric("Average Index", f"{avg_idx:.3f}",
                         f"{(avg_idx - 1.0) * 100:.1f}%")
            
            with col4:
                st.metric("Total Indexes", len(final_indexes))
            
            # Show transaction statistics if enabled
            if show_transaction_counts and transaction_counts:
                st.markdown("#### 📈 Transaction Statistics")
                col1, col2, col3, col4 = st.columns(4)
                
                total_trans = sum(tc['total'] for tc in transaction_counts.values())
                total_original = sum(tc['original'] for tc in transaction_counts.values())
                total_outliers = sum(tc['outliers_removed'] for tc in transaction_counts.values())
                avg_outlier_pct = (total_outliers / total_original * 100) if total_original > 0 else 0
                
                with col1:
                    st.metric("Total Transactions", f"{total_trans:,}")
                
                with col2:
                    st.metric("Original Count", f"{total_original:,}")
                
                with col3:
                    st.metric("Outliers Removed", f"{total_outliers:,}")
                
                with col4:
                    st.metric("Outlier Rate", f"{avg_outlier_pct:.1f}%")
        
        # Tab 2: By Category
        with tabs[1]:
            st.subheader("Final Indexes by Category")
            st.caption("View all indexes grouped by index category")
            
            # Group indexes by category
            categories = {}
            for index_name, index_info in final_indexes.items():
                category = index_info['category']
                if category not in categories:
                    categories[category] = []
                categories[category].append((index_name, index_info))
            
            # Display each category
            for category, indexes in categories.items():
                with st.expander(f"📂 {category}", expanded=True):
                    # Create table for this category
                    cat_data = []
                    for index_name, index_info in indexes:
                        index_series = index_info['index']
                        latest_value = index_series.iloc[-1]
                        latest_quarter = index_series.index[-1]
                        
                        row_data = {
                            'Index Name': index_name,
                            'Latest Quarter': latest_quarter,
                            'Latest Index': f"{latest_value:.4f}",
                            'Change vs Base': f"{(latest_value - 1.0) * 100:.1f}%",
                        }
                        
                        # Add transaction counts if enabled
                        if show_transaction_counts and index_name in transaction_counts:
                            row_data['Transactions'] = f"{transaction_counts[index_name]['total']:,}"
                        
                        row_data['Regions'] = ', '.join(index_info['regions'])
                        cat_data.append(row_data)
                    
                    cat_df = pd.DataFrame(cat_data)
                    st.dataframe(cat_df, use_container_width=True, hide_index=True)
                    
                    # Plot for this category
                    fig_cat = go.Figure()
                    for index_name, index_info in indexes:
                        index_series = index_info['index']
                        fig_cat.add_trace(go.Scatter(
                            x=index_series.index,
                            y=index_series.values,
                            mode='lines+markers',
                            name=index_name,
                            line=dict(width=2),
                            marker=dict(size=6)
                        ))
                    
                    fig_cat.update_layout(
                        title=f"{category} - Index Evolution",
                        xaxis_title="Quarter",
                        yaxis_title="Index (Base = 1.0)",
                        hovermode='x unified',
                        height=400,
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                    )
                    fig_cat.update_xaxes(tickangle=45)
                    st.plotly_chart(fig_cat, use_container_width=True)
        
        # Tab 3: Time Series
        with tabs[2]:
            st.subheader("Time Series Comparison")
            st.caption("Compare multiple final indexes over time")
            
            # Select indexes to compare
            index_names = list(final_indexes.keys())
            selected_indexes = st.multiselect(
                "Select indexes to compare",
                index_names,
                default=index_names[:5] if len(index_names) > 5 else index_names,
                key="timeseries_indexes"
            )
            
            if selected_indexes:
                # Create comparison plot
                fig_ts = go.Figure()
                
                # Function to convert quarter string to datetime for proper sorting
                def quarter_to_date(q_str):
                    """Convert '2020-Q1' to datetime"""
                    try:
                        parts = q_str.split('-Q')
                        if len(parts) == 2:
                            year = int(parts[0])
                            quarter = int(parts[1])
                            month = (quarter - 1) * 3 + 1
                            return pd.Timestamp(year=year, month=month, day=1)
                    except:
                        pass
                    return None
                
                # Collect all unique quarters and convert to datetime
                all_quarters = []
                quarter_mapping = {}  # datetime -> string mapping
                
                for index_name in selected_indexes:
                    index_info = final_indexes[index_name]
                    index_series = index_info['index']
                    
                    for q_str in index_series.index:
                        q_date = quarter_to_date(q_str)
                        if q_date is not None:
                            all_quarters.append(q_date)
                            quarter_mapping[q_date] = q_str
                
                # Get unique quarters and sort chronologically
                unique_quarters_dt = sorted(list(set(all_quarters)))
                
                # Select which ticks to show (every 4th quarter = once per year)
                tick_indices = list(range(0, len(unique_quarters_dt), 4))
                tickvals_dt = [unique_quarters_dt[i] for i in tick_indices if i < len(unique_quarters_dt)]
                ticktext = [quarter_mapping[dt] for dt in tickvals_dt]
                
                # Plot each series with datetime x-axis
                for index_name in selected_indexes:
                    index_info = final_indexes[index_name]
                    index_series = index_info['index']
                    
                    # Convert quarters to datetime for this series
                    x_dates = [quarter_to_date(q) for q in index_series.index]
                    
                    fig_ts.add_trace(go.Scatter(
                        x=x_dates,
                        y=index_series.values,
                        mode='lines+markers',
                        name=index_name,
                        line=dict(width=2),
                        marker=dict(size=5)
                    ))
                
                fig_ts.update_layout(
                    title="Final Indexes Comparison",
                    xaxis_title="Quarter",
                    yaxis_title="Index (Base = 1.0)",
                    hovermode='x unified',
                    height=600,
                    legend=dict(
                        orientation="v", 
                        yanchor="top", 
                        y=0.99, 
                        xanchor="left", 
                        x=1.01,
                        bgcolor="rgba(255, 255, 255, 0.9)",
                        bordercolor="rgba(0, 0, 0, 0.3)",
                        borderwidth=1,
                        font=dict(size=10)
                    ),
                    margin=dict(r=250, b=120, t=80, l=80)
                )
                fig_ts.update_xaxes(
                    tickangle=45,
                    tickmode='array',
                    tickvals=tickvals_dt,
                    ticktext=ticktext
                )
                fig_ts.update_yaxes(
                    gridcolor='lightgray',
                    gridwidth=0.5,
                    zeroline=True,
                    zerolinecolor='black',
                    zerolinewidth=1
                )
                st.plotly_chart(fig_ts, use_container_width=True)
                
                # Data table
                st.markdown("### 📊 Index Values Table")
                
                # Create comparison table
                comparison_data = {}
                for index_name in selected_indexes:
                    index_series = final_indexes[index_name]['index']
                    comparison_data[index_name] = index_series
                
                comparison_df = pd.DataFrame(comparison_data).T
                st.dataframe(comparison_df.round(4), use_container_width=True)
            else:
                st.info("👆 Select at least one index to view the time series")
        
        # Tab 4: Export
        with tabs[3]:
            st.subheader("Export Final Indexes")
            st.caption("Download final indexes in various formats")
            
            # Create comprehensive dataframe
            all_data = {}
            for index_name, index_info in final_indexes.items():
                all_data[index_name] = index_info['index']
            
            export_df = pd.DataFrame(all_data).T
            
            # Display preview
            st.markdown("### 📋 Preview")
            st.dataframe(export_df.round(4), use_container_width=True, height=300)
            
            # Download buttons
            col1, col2 = st.columns(2)
            
            with col1:
                # CSV download
                csv_data = export_df.to_csv()
                st.download_button(
                    label="⬇️ Download CSV",
                    data=csv_data,
                    file_name=f"latvia_final_indexes_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            
            with col2:
                # Excel download
                output = io.BytesIO()
                with pd.ExcelWriter(output, engine='openpyxl') as writer:
                    export_df.to_excel(writer, sheet_name='Final Indexes')
                    
                    # Add metadata sheet
                    metadata = []
                    for index_name, index_info in final_indexes.items():
                        metadata.append({
                            'Index Name': index_name,
                            'Category': index_info['category'],
                            'Base Period': index_info['base'],
                            'Regions': ', '.join(index_info['regions'])
                        })
                    metadata_df = pd.DataFrame(metadata)
                    metadata_df.to_excel(writer, sheet_name='Metadata', index=False)
                
                excel_data = output.getvalue()
                st.download_button(
                    label="⬇️ Download Excel",
                    data=excel_data,
                    file_name=f"latvia_final_indexes_{datetime.now().strftime('%Y%m%d')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True
                )

# Main app
def main():
    st.title("🏠 Baltic Real Estate Price Index Analyzer")
    
    # Dataset/Country selector at the very top
    st.markdown("### 🌍 Select Dataset")
    dataset = st.radio(
        "Choose which dataset to analyze:",
        options=["Latvia - Detailed Transactions", "Lithuania - Bigbank Statistics"],
        horizontal=True,
        help="Switch between Latvian detailed transaction data and Lithuanian aggregated statistics from Bigbank",
        key="dataset_selector"
    )
    
    st.markdown("---")
    
    # Route to appropriate analyzer
    if dataset == "Lithuania - Bigbank Statistics":
        # Import and run Lithuania analyzer
        try:
            from lithuania_analyzer import lithuania_analyzer
            lithuania_analyzer()
            return
        except Exception as e:
            st.error(f"Error loading Lithuania analyzer: {e}")
            st.info("Make sure 'lithuania_analyzer.py' and 'Bigbank_purchase transaction statistics_202506 Lithuania EN.xlsx' are in the same directory.")
            return
    
    # Continue with Latvian analyzer
    st.header("🇱🇻 Latvian Real Estate Price Index Analyzer")
    
    # Analysis mode selector
    analysis_mode = st.radio(
        "**Select Analysis Mode:**",
        options=["📊 Final Indexes Master View", "🔍 Detailed Property Analysis"],
        horizontal=True,
        help="Master View: Overview of all property indexes | Detailed Analysis: Deep dive into specific property type",
        key="analysis_mode_selector"
    )
    
    st.markdown("---")
    
    # If Master View is selected, show the final indexes dashboard
    if analysis_mode == "📊 Final Indexes Master View":
        show_final_indexes_master_view()
        return
    
    # Continue with detailed property analysis
    # Property type selector at the very top
    property_type = st.radio(
        "**Select Property Type:**",
        options=["Houses", "Apartments", "Premises", "Agricultural land", "Forest land", "Land commercial", "Land residential", "Other land"],
        horizontal=True,
        help="Switch between houses, apartments, premises, agricultural land, forest land, commercial land, residential land, and other land analysis",
        key="property_type_selector"
    )
    
    # Clear cached tables when switching property types
    if 'last_property_type' not in st.session_state:
        st.session_state['last_property_type'] = property_type
    elif st.session_state['last_property_type'] != property_type:
        # Property type changed - clear all results
        for key in ['prices_tables', 'counts_table', 'index_tables', 'agg_df', 'df_filtered']:
            if key in st.session_state:
                del st.session_state[key]
        st.session_state['last_property_type'] = property_type
    
    st.markdown("---")
    
    # Load data based on property type
    with st.spinner(f"Loading {property_type.lower()} data..."):
        df = load_data(property_type)
    
    st.success(f"✅ Loaded {len(df):,} {property_type.lower()} records")
    
    # Sidebar filters
    st.sidebar.header("📊 Filters")
    st.sidebar.caption(f"Analyzing: **{property_type}**")
    
    # Price calculation method
    st.sidebar.markdown("---")
    st.sidebar.subheader("⚙️ Calculation Method")
    
    # Adjust options based on property type
    if property_type in ['Agricultural land', 'Forest land', 'Land commercial', 'Land residential', 'Other land']:
        price_method = st.sidebar.radio(
            "Price per m² calculation:",
            options=["Calculated (Price ÷ Sold Area)", "Use Land_EUR_m2 column"],
            index=0,
            help="Choose between calculating price per m² or using the existing Land_EUR_m2 column from the data"
        )
        use_total_eur_m2 = (price_method == "Use Land_EUR_m2 column")
    else:
        price_method = st.sidebar.radio(
            "Price per m² calculation:",
            options=["Calculated (Price ÷ Sold Area)", "Use Total_EUR_m2 column"],
            index=0,
            help="Choose between calculating price per m² or using the existing Total_EUR_m2 column from the data"
        )
        use_total_eur_m2 = (price_method == "Use Total_EUR_m2 column")
    
    # Initialize session state for filters
    if 'apply_filters' not in st.session_state:
        st.session_state['apply_filters'] = False
    
    with st.sidebar.expander("📅 Date & Time", expanded=True):
        # Year range slider
        min_year = int(df['Year'].min())
        max_year = int(df['Year'].max())
        year_range = st.slider(
            "Year Range",
            min_value=min_year,
            max_value=max_year,
            value=(min_year, max_year)
        )
        
        # Quarter filter
        quarters = st.multiselect(
            "Quarter",
            options=[1, 2, 3, 4],
            default=[1, 2, 3, 4],
            format_func=lambda x: f"Q{x}"
        )
    
    with st.sidebar.expander("🗺️ Location", expanded=True):
        # Region filter with "Select All" option
        regions = sorted(df['region_riga_separate'].dropna().unique())
        
        col1, col2 = st.columns([3, 1])
        with col1:
            st.write("**Regions**")
        with col2:
            select_all_regions = st.checkbox("All", value=True, key="all_regions")
        
        if select_all_regions:
            selected_regions = regions
            st.info(f"✓ All {len(regions)} regions selected")
        else:
            selected_regions = []
            for region in regions:
                if st.checkbox(region, value=False, key=f"region_{region}"):
                    selected_regions.append(region)
        
        st.markdown("---")
        
        # Municipality filter - compact
        st.write("**Municipality** (optional)")
        municipalities = sorted(df['municipality'].dropna().unique())
        filter_municipalities = st.checkbox("Filter by municipality", value=False)
        
        if filter_municipalities:
            selected_municipalities = st.multiselect(
                "Select municipalities",
                municipalities,
                default=[],
                label_visibility="collapsed"
            )
        else:
            selected_municipalities = municipalities
    
    with st.sidebar.expander("🏗️ Building/Property Type", expanded=False):
        types = sorted(df['Type'].dropna().unique())
        
        col1, col2 = st.columns([3, 1])
        with col1:
            type_label = "Building Materials" if property_type == "Apartments" else "Property Types"
            st.write(f"**{type_label}**")
        with col2:
            select_all_types = st.checkbox("All", value=True, key="all_types")
        
        if select_all_types:
            selected_types = types
            st.info(f"✓ All {len(types)} types selected")
        else:
            selected_types = []
            for ptype in types:
                if st.checkbox(ptype, value=False, key=f"type_{ptype}"):
                    selected_types.append(ptype)
    
    with st.sidebar.expander("💰 Price & Area", expanded=False):
        # Price filter
        st.write("**Price (EUR)**")
        filter_price = st.checkbox("Filter by price", value=False, key="filter_price")
        if filter_price:
            col1, col2 = st.columns(2)
            with col1:
                price_min = st.slider(
                    "Minimum price",
                    min_value=0,
                    max_value=500000,
                    value=0,
                    step=1000,
                    format="€%d"
                )
            with col2:
                price_max = st.slider(
                    "Maximum price",
                    min_value=0,
                    max_value=5000000,
                    value=5000000,
                    step=10000,
                    format="€%d"
                )
            price_range = (price_min, price_max)
        else:
            price_range = None
        
        st.markdown("---")
        
        # Price per m² filter (Total_EUR_m2 or Land_EUR_m2)
        price_m2_col = 'Land_EUR_m2' if property_type in ['Agricultural land', 'Forest land', 'Land commercial', 'Other land'] else 'Total_EUR_m2'
        st.write("**Price per m² (EUR/m²)**")
        filter_price_m2 = st.checkbox("Filter by price per m²", value=False, key="filter_price_m2")
        if filter_price_m2:
            if price_m2_col in df.columns:
                df_price_m2 = df[price_m2_col].dropna()
                if len(df_price_m2) > 0:
                    min_price_m2 = float(df_price_m2.min())
                    max_price_m2 = float(df_price_m2.max())
                    price_m2_range = st.slider(
                        "Price per m² range",
                        min_value=min_price_m2,
                        max_value=max_price_m2,
                        value=(min_price_m2, max_price_m2),
                        format="€%.0f/m²",
                        label_visibility="collapsed"
                    )
                else:
                    st.warning("No price per m² data available")
                    price_m2_range = None
            else:
                st.warning(f"{price_m2_col} column not found")
                price_m2_range = None
        else:
            price_m2_range = None
        
        st.markdown("---")
        
        # Sold Area filter
        st.write("**Sold Area (m²)**")
        filter_sold_area = st.checkbox("Filter by sold area", value=False, key="filter_sold_area")
        if filter_sold_area:
            df_sold = df['Sold_Area_m2'].dropna()
            if len(df_sold) > 0:
                min_sold = float(df_sold.min())
                max_sold = float(df_sold.max())
                sold_area_range = st.slider(
                    "Sold area range",
                    min_value=min_sold,
                    max_value=max_sold,
                    value=(min_sold, max_sold),
                    format="%.0f m²",
                    label_visibility="collapsed"
                )
            else:
                st.warning("No sold area data available")
                sold_area_range = None
        else:
            sold_area_range = None
        
        st.markdown("---")
        
        # Total Area filter
        st.write("**Total Area (m²)**")
        filter_total_area = st.checkbox("Filter by total area", value=False, key="filter_total_area")
        if filter_total_area:
            df_total = df['Total_Area_m2'].dropna()
            if len(df_total) > 0:
                min_total = float(df_total.min())
                max_total = float(df_total.max())
                total_area_range = st.slider(
                    "Total area range",
                    min_value=min_total,
                    max_value=max_total,
                    value=(min_total, max_total),
                    format="%.0f m²",
                    label_visibility="collapsed"
                )
            else:
                st.warning("No total area data available")
                total_area_range = None
        else:
            total_area_range = None
        
        st.markdown("---")
        
        # Land Area filter
        st.write("**Land Area (m²)**")
        filter_land = st.checkbox("Filter by land area", value=False, key="filter_land")
        if filter_land:
            df_land = df['Land_m2'].dropna()
            if len(df_land) > 0:
                min_land = float(df_land.min())
                max_land = float(df_land.max())
                land_area_range = st.slider(
                    "Land area range",
                    min_value=min_land,
                    max_value=max_land,
                    value=(min_land, max_land),
                    format="%.0f m²",
                    label_visibility="collapsed"
                )
            else:
                st.warning("No land area data available")
                land_area_range = None
        else:
            land_area_range = None
    
    with st.sidebar.expander("📋 Property Details", expanded=False):
        # Record filter (only if column exists)
        if 'Record' in df.columns:
            records = sorted(df['Record'].dropna().unique())
            
            st.write("**Record Type**")
            filter_record = st.checkbox("Filter by record type", value=False, key="filter_record")
            if filter_record:
                selected_records = st.multiselect(
                    "Select records",
                    records,
                    default=records,
                    label_visibility="collapsed"
                )
            else:
                selected_records = records
        else:
            selected_records = None
        
        st.markdown("---")
        
        # Dom_Parts filter (only if column exists)
        if 'Dom_Parts' in df.columns:
            dom_parts = sorted(df['Dom_Parts'].dropna().unique())
        else:
            dom_parts = []
        
        if len(dom_parts) > 0:
            st.write("**Property Parts (Dom_Parts)**")
            filter_dom_parts = st.checkbox("Filter by property parts", value=False, key="filter_dom_parts")
            if filter_dom_parts:
                # Show only common ones by default, with option to show all
                common_parts = ['1/1', '1/2', '1/3', '1/4', '1/5', '1/6']
                available_common = [p for p in common_parts if p in dom_parts]
                
                selected_dom_parts = st.multiselect(
                    "Select property parts",
                    dom_parts,
                    default=available_common if available_common else dom_parts[:5],
                    label_visibility="collapsed",
                    help=f"Total {len(dom_parts)} unique values available"
                )
            else:
                selected_dom_parts = dom_parts
        else:
            selected_dom_parts = None
        
        st.markdown("---")
        
        # Finishing filter
        st.write("**Finishing**")
        filter_finishing = st.checkbox("Filter by finishing", value=False, key="filter_finishing")
        if filter_finishing:
            df_finishing = df['Finishing'].dropna()
            if len(df_finishing) > 0:
                finishing_values = sorted(df_finishing.unique())
                selected_finishing = st.multiselect(
                    "Finishing type",
                    finishing_values,
                    default=finishing_values,
                    label_visibility="collapsed"
                )
            else:
                st.warning("No finishing data available")
                selected_finishing = None
        else:
            selected_finishing = None
        
        st.markdown("---")
        
        # Category filter
        categories = sorted(df['Category'].dropna().unique())
        
        st.write("**Category**")
        filter_category = st.checkbox("Filter by category", value=False, key="filter_category")
        if filter_category:
            selected_categories = st.multiselect(
                "Select categories",
                categories,
                default=categories,
                label_visibility="collapsed"
            )
        else:
            selected_categories = categories
    
    # Duplicate handling
    with st.sidebar.expander("🔍 Duplicate Detection", expanded=False):
        st.write("**Handle Duplicate Transactions**")
        
        duplicate_method = st.radio(
            "Duplicate detection method:",
            options=[
                "Keep all (no filtering)",
                "Remove exact duplicates",
                "Remove by Address+Date+Price"
            ],
            index=0,
            help="Choose how to handle potential duplicate transactions in the dataset"
        )
        
        if duplicate_method != "Keep all (no filtering)":
            st.caption("ℹ️ Duplicates will be removed before analysis (keeping first occurrence)")
    
    # Outlier Detection
    with st.sidebar.expander("🎯 Outlier Detection & Removal", expanded=False):
        st.write("**Remove Statistical Outliers**")
        st.caption("Filter extreme values based on price per m² distribution")
        
        enable_outlier_filter = st.checkbox(
            "Enable outlier filtering",
            value=False,
            key="enable_outlier_filter",
            help="Remove transactions with extreme price per m² values"
        )
        
        if enable_outlier_filter:
            outlier_method = st.radio(
                "Detection method:",
                options=[
                    "IQR Method (1.5x - standard)",
                    "IQR Method (3.0x - lenient)",
                    "Percentile Method"
                ],
                index=0,
                help="IQR = Interquartile Range. 1.5x is standard (removes more outliers), 3.0x is lenient (removes only extreme outliers)"
            )
            
            if "Percentile" in outlier_method:
                col1, col2 = st.columns(2)
                with col1:
                    lower_percentile = st.slider(
                        "Lower %",
                        min_value=0.0,
                        max_value=10.0,
                        value=1.0,
                        step=0.5,
                        format="%.1f%%"
                    )
                with col2:
                    upper_percentile = st.slider(
                        "Upper %",
                        min_value=90.0,
                        max_value=100.0,
                        value=99.0,
                        step=0.5,
                        format="%.1f%%"
                    )
            else:
                lower_percentile = None
                upper_percentile = None
            
            apply_per_region = st.checkbox(
                "Apply per region separately",
                value=False,
                help="Calculate outliers for each region independently (more granular)"
            )
            
            apply_per_quarter = st.checkbox(
                "Apply per quarter separately",
                value=False,
                help="Calculate outliers for each quarter independently (identify temporal anomalies)"
            )
            
            st.caption("⚠️ Outliers will be removed from ALL tabs (Prices, Counts, Index)")
    
    # Reset filters button
    st.sidebar.markdown("---")
    if st.sidebar.button("🔄 Reset All Filters", use_container_width=True):
        st.rerun()
    
    # Apply filters
    df_filtered = df.copy()
    
    # Year range filter
    df_filtered = df_filtered[(df_filtered['Year'] >= year_range[0]) & 
                              (df_filtered['Year'] <= year_range[1])]
    
    # Quarter filter
    if quarters:
        df_filtered = df_filtered[df_filtered['Quarter'].isin(quarters)]
    
    # Region filter
    if selected_regions:
        df_filtered = df_filtered[df_filtered['region_riga_separate'].isin(selected_regions)]
    
    # Type filter
    if selected_types:
        df_filtered = df_filtered[df_filtered['Type'].isin(selected_types)]
    
    # Municipality filter
    if selected_municipalities:
        df_filtered = df_filtered[df_filtered['municipality'].isin(selected_municipalities)]
    
    # Price filter
    if price_range:
        df_filtered = df_filtered[(df_filtered['Price_EUR'] >= price_range[0]) & 
                                  (df_filtered['Price_EUR'] <= price_range[1])]
    
    # Price per m² filter
    if price_m2_range:
        price_m2_col = 'Land_EUR_m2' if property_type in ['Agricultural land', 'Forest land', 'Land commercial', 'Other land'] else 'Total_EUR_m2'
        if price_m2_col in df_filtered.columns:
            df_filtered = df_filtered[(df_filtered[price_m2_col] >= price_m2_range[0]) & 
                                      (df_filtered[price_m2_col] <= price_m2_range[1])]
    
    # Sold Area filter
    if sold_area_range:
        df_filtered = df_filtered[(df_filtered['Sold_Area_m2'] >= sold_area_range[0]) & 
                                  (df_filtered['Sold_Area_m2'] <= sold_area_range[1])]
    
    # Total Area filter
    if total_area_range:
        df_filtered = df_filtered[(df_filtered['Total_Area_m2'] >= total_area_range[0]) & 
                                  (df_filtered['Total_Area_m2'] <= total_area_range[1])]
    
    # Land Area filter
    if land_area_range:
        df_filtered = df_filtered[
            (df_filtered['Land_m2'].notna()) &
            (df_filtered['Land_m2'] >= land_area_range[0]) & 
            (df_filtered['Land_m2'] <= land_area_range[1])
        ]
    
    # Record filter
    if selected_records is not None and 'Record' in df_filtered.columns:
        df_filtered = df_filtered[df_filtered['Record'].isin(selected_records)]
    
    # Dom_Parts filter
    if selected_dom_parts is not None and 'Dom_Parts' in df_filtered.columns:
        df_filtered = df_filtered[df_filtered['Dom_Parts'].isin(selected_dom_parts)]
    
    # Finishing filter
    if selected_finishing:
        df_filtered = df_filtered[
            (df_filtered['Finishing'].notna()) &
            (df_filtered['Finishing'].isin(selected_finishing))
        ]
    
    # Category filter
    if selected_categories:
        df_filtered = df_filtered[df_filtered['Category'].isin(selected_categories)]
    
    # Outlier removal
    records_before_outlier = len(df_filtered)
    outliers_removed = 0
    
    if enable_outlier_filter:
        outlier_mask = detect_outliers(
            df_filtered,
            use_total_eur_m2=use_total_eur_m2,
            method=outlier_method,
            lower_percentile=lower_percentile,
            upper_percentile=upper_percentile,
            per_region=apply_per_region,
            per_quarter=apply_per_quarter,
            property_type=property_type
        )
        df_filtered = df_filtered[outlier_mask]
        outliers_removed = records_before_outlier - len(df_filtered)
    
    # Duplicate removal
    records_before_dedup = len(df_filtered)
    if duplicate_method == "Remove exact duplicates":
        df_filtered = df_filtered.drop_duplicates(keep='first')
    elif duplicate_method == "Remove by Address+Date+Price":
        df_filtered = df_filtered.drop_duplicates(subset=['Address', 'Date', 'Price_EUR'], keep='first')
    records_removed = records_before_dedup - len(df_filtered)
    
    # Display filter summary
    filter_summary = []
    if year_range != (min_year, max_year):
        filter_summary.append(f"Years: {year_range[0]}-{year_range[1]}")
    if len(quarters) < 4:
        filter_summary.append(f"Quarters: {', '.join([f'Q{q}' for q in quarters])}")
    if not select_all_regions:
        filter_summary.append(f"Regions: {len(selected_regions)} selected")
    if filter_municipalities:
        filter_summary.append(f"Municipalities: {len(selected_municipalities)} selected")
    if not select_all_types:
        filter_summary.append(f"Property Types: {len(selected_types)} selected")
    if price_range:
        filter_summary.append(f"Price: €{price_range[0]:,.0f} - €{price_range[1]:,.0f}")
    if price_m2_range:
        filter_summary.append(f"€/m²: {price_m2_range[0]:.0f} - {price_m2_range[1]:.0f}")
    if sold_area_range:
        filter_summary.append(f"Sold Area: {sold_area_range[0]:.0f} - {sold_area_range[1]:.0f} m²")
    if total_area_range:
        filter_summary.append(f"Total Area: {total_area_range[0]:.0f} - {total_area_range[1]:.0f} m²")
    if land_area_range:
        filter_summary.append(f"Land Area: {land_area_range[0]:.0f} - {land_area_range[1]:.0f} m²")
    if selected_records is not None and 'Record' in df.columns:
        records_list = sorted(df['Record'].dropna().unique())
        if len(selected_records) < len(records_list):
            filter_summary.append(f"Records: {len(selected_records)}/{len(records_list)}")
    if selected_dom_parts is not None and 'Dom_Parts' in df.columns:
        dom_parts_list = sorted(df['Dom_Parts'].dropna().unique())
        if len(selected_dom_parts) < len(dom_parts_list):
            filter_summary.append(f"Dom Parts: {len(selected_dom_parts)}/{len(dom_parts_list)}")
    if selected_finishing:
        filter_summary.append(f"Finishing: {len(selected_finishing)} types")
    if filter_category and len(selected_categories) < len(categories):
        filter_summary.append(f"Categories: {len(selected_categories)}/{len(categories)}")
    
    if filter_summary:
        st.info(f"📈 Filtered to {len(df_filtered):,} records | Active filters: {' • '.join(filter_summary)}")
    else:
        st.info(f"📈 Showing all {len(df_filtered):,} records (no filters applied)")
    
    # Show outlier removal summary
    if enable_outlier_filter:
        if outliers_removed > 0:
            outlier_pct = (outliers_removed / records_before_outlier * 100)
            grouping_desc = []
            if apply_per_region:
                grouping_desc.append("per region")
            if apply_per_quarter:
                grouping_desc.append("per quarter")
            grouping_text = " + ".join(grouping_desc) if grouping_desc else "globally"
            
            st.warning(f"🎯 **Outliers Removed:** {outliers_removed:,} transactions removed ({outlier_pct:.1f}% of data before outlier filter) using {outlier_method} ({grouping_text})")
            st.caption(f"📊 Before outlier filter: {records_before_outlier:,} | After: {len(df_filtered):,}")
        else:
            st.info(f"🎯 **Outlier Filter Active:** No outliers detected using {outlier_method}")
    
    # Show duplicate removal summary
    if records_removed > 0:
        dedup_method = "exact duplicates" if duplicate_method == "Remove exact duplicates" else "duplicates (Address+Date+Price)"
        st.warning(f"🔍 **Duplicates Removed:** {records_removed:,} {dedup_method} were removed from the analysis ({records_removed/records_before_dedup*100:.1f}% of filtered data)")
    
    # Generate tables button
    if st.button("🚀 Generate Tables", type="primary", use_container_width=True):
        with st.spinner("Generating tables..."):
            # Aggregate data
            agg_df = aggregate_by_region_quarter(df_filtered, use_total_eur_m2, property_type)
            
            # Calculate all tables
            prices_original = create_prices_table(agg_df, ma_quarters=1)
            prices_ma2 = create_prices_table(agg_df, ma_quarters=2)
            prices_ma3 = create_prices_table(agg_df, ma_quarters=3)
            prices_ma4 = create_prices_table(agg_df, ma_quarters=4)
            
            counts = create_counts_table(df_filtered)
            
            index_original = create_index_table(prices_original, property_type)
            index_ma2 = create_index_table(prices_ma2, property_type)
            index_ma3 = create_index_table(prices_ma3, property_type)
            index_ma4 = create_index_table(prices_ma4, property_type)
            
            # Store in session state
            st.session_state['agg_df'] = agg_df
            st.session_state['prices_tables'] = [prices_original, prices_ma2, prices_ma3, prices_ma4]
            st.session_state['counts_table'] = counts
            st.session_state['index_tables'] = [index_original, index_ma2, index_ma3, index_ma4]
            st.session_state['df_filtered'] = df_filtered
            st.session_state['use_total_eur_m2'] = use_total_eur_m2
            st.session_state['property_type'] = property_type
        
        st.success("✅ Tables generated!")
    
    # Display results if tables are generated
    if 'prices_tables' in st.session_state:
        st.markdown("---")
        st.header(f"📊 Results - {property_type}")
        
        # Create tabs
        tab_names = [
            "Summary",
            "Prices - Original",
            "Prices - MA2",
            "Prices - MA3",
            "Prices - MA4",
            "Counts",
            "Index - Original",
            "Index - MA2",
            "Index - MA3",
            "Index - MA4",
            "📊 Distribution Analysis"
        ]
        
        tabs = st.tabs(tab_names)
        
        # Summary tab
        with tabs[0]:
            st.subheader("📈 Summary Statistics")
            
            df_filt = st.session_state['df_filtered']
            use_total = st.session_state.get('use_total_eur_m2', False)
            prop_type = st.session_state.get('property_type', 'Houses')
            df_filt = calculate_price_per_m2(df_filt, use_total, prop_type)
            
            summary = df_filt.groupby('region_riga_separate').agg({
                'Price_EUR': ['count', 'sum', 'mean', 'median'],
                'Sold_Area_m2': ['sum', 'mean', 'median'],
                'Price_per_m2': ['mean', 'median', 'std']
            }).round(2)
            
            if prop_type in ['Agricultural land', 'Forest land', 'Land commercial', 'Land residential', 'Other land']:
                method_label = "Land_EUR_m2" if use_total else "Calculated (Price ÷ Sold Area)"
            else:
                method_label = "Total_EUR_m2" if use_total else "Calculated (Price ÷ Sold Area)"
            st.caption(f"📊 Using: **{method_label}**")
            
            # Show data quality info based on calculation method
            total_records = len(st.session_state['df_filtered'])
            
            if use_total:
                # Check Total_EUR_m2 or Land_EUR_m2 validity
                if prop_type in ['Agricultural land', 'Forest land', 'Land commercial', 'Land residential', 'Other land']:
                    valid_for_prices = df_filt[df_filt['Land_EUR_m2'].notna() & (df_filt['Land_EUR_m2'] > 0)]
                    method_desc = "missing Land_EUR_m2"
                else:
                    valid_for_prices = df_filt[df_filt['Total_EUR_m2'].notna() & (df_filt['Total_EUR_m2'] > 0)]
                    method_desc = "missing Total_EUR_m2"
            else:
                # Check Price_EUR and Sold_Area_m2 validity
                valid_for_prices = df_filt[df_filt['Price_EUR'].notna() & df_filt['Sold_Area_m2'].notna() & (df_filt['Sold_Area_m2'] > 0)]
                method_desc = "missing Price_EUR or Sold_Area_m2"
            
            valid_count = len(valid_for_prices)
            excluded_count = total_records - valid_count
            
            if excluded_count > 0:
                st.warning(f"⚠️ **Data Quality Note:** {excluded_count:,} of {total_records:,} transactions ({excluded_count/total_records*100:.1f}%) were excluded from price calculations due to {method_desc}. These are still counted in the 'Counts' tab.")
                if not use_total and excluded_count > total_records * 0.2:  # If more than 20% excluded with calculated method
                    if prop_type in ['Agricultural land', 'Forest land', 'Land commercial', 'Land residential', 'Other land']:
                        st.info(f"💡 **Tip:** Try using 'Land_EUR_m2 column' method (in sidebar) for better data coverage!")
                    else:
                        st.info(f"💡 **Tip:** Try using 'Total_EUR_m2 column' method (in sidebar) for better data coverage. It has no missing values and will include all {total_records:,} transactions!")
            else:
                st.success(f"✅ All {total_records:,} transactions have complete data for the selected calculation method!")
            
            # Show duplicate statistics
            st.markdown("---")
            st.markdown("### 🔍 Duplicate Analysis")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                exact_dupes = df_filt.duplicated(keep=False).sum()
                exact_pct = (exact_dupes / len(df_filt) * 100) if len(df_filt) > 0 else 0
                st.metric("Exact Duplicates", f"{exact_dupes:,}", f"{exact_pct:.1f}%")
            
            with col2:
                addr_date_price_dupes = df_filt.duplicated(subset=['Address', 'Date', 'Price_EUR'], keep=False).sum()
                addr_pct = (addr_date_price_dupes / len(df_filt) * 100) if len(df_filt) > 0 else 0
                st.metric("Address+Date+Price Dupes", f"{addr_date_price_dupes:,}", f"{addr_pct:.1f}%")
            
            with col3:
                key_fields_dupes = df_filt.duplicated(subset=['Address', 'Date', 'Price_EUR', 'Sold_Area_m2'], keep=False).sum()
                key_pct = (key_fields_dupes / len(df_filt) * 100) if len(df_filt) > 0 else 0
                st.metric("Key Fields Dupes", f"{key_fields_dupes:,}", f"{key_pct:.1f}%")
            
            st.caption("💡 Use the 'Duplicate Detection' filter in the sidebar to remove duplicates from the analysis")
            
            st.markdown("---")
            st.markdown("### 📊 Regional Summary")
            st.dataframe(summary, use_container_width=True, height=400)
        
        # Prices tabs  
        for i in range(4):
            with tabs[i+1]:
                ma_label = f"Moving Average ({i+1} Quarter{'s' if i > 0 else ''})"
                st.subheader(f"💰 Average Price per m² - {ma_label}")
                
                use_total = st.session_state.get('use_total_eur_m2', False)
                prop_type = st.session_state.get('property_type', 'Houses')
                if prop_type in ['Agricultural land', 'Forest land', 'Land commercial', 'Land residential', 'Other land']:
                    method_label = "Land_EUR_m2" if use_total else "Calculated (Price ÷ Sold Area)"
                else:
                    method_label = "Total_EUR_m2" if use_total else "Calculated (Price ÷ Sold Area)"
                st.caption(f"📊 Using: **{method_label}**")
                
                prices_df = st.session_state['prices_tables'][i]
                st.dataframe(prices_df.round(2), use_container_width=True, height=400)
                
                # Plot
                st.markdown("#### 📉 Interactive Plot")
                
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    available_regions = list(prices_df.index)
                    plot_regions_selected = st.multiselect(
                        "Select regions to display",
                        available_regions,
                        default=available_regions,
                        key=f"regions_prices_{i}"
                    )
                
                with col2:
                    available_quarters = list(prices_df.columns)
                    if len(available_quarters) > 1:
                        date_range_plot = st.select_slider(
                            "Date range",
                            options=available_quarters,
                            value=(available_quarters[0], available_quarters[-1]),
                            key=f"date_range_prices_{i}"
                        )
                    else:
                        date_range_plot = None
                
                fig = plot_regions(
                    prices_df,
                    f"Price per m² Over Time - {ma_label}",
                    "Price per m² (EUR)",
                    selected_regions=plot_regions_selected,
                    date_range=date_range_plot if len(available_quarters) > 1 else None
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # Counts tab
        with tabs[5]:
            st.subheader("📊 Transaction Counts by Region and Quarter")
            
            counts_df = st.session_state['counts_table']
            st.dataframe(counts_df, use_container_width=True, height=400)
            
            # Plot
            st.markdown("#### 📉 Interactive Plot")
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                available_regions = list(counts_df.index)
                plot_regions_selected = st.multiselect(
                    "Select regions to display",
                    available_regions,
                    default=available_regions,
                    key="regions_counts"
                )
            
            with col2:
                available_quarters = list(counts_df.columns)
                if len(available_quarters) > 1:
                    date_range_plot = st.select_slider(
                        "Date range",
                        options=available_quarters,
                        value=(available_quarters[0], available_quarters[-1]),
                        key="date_range_counts"
                    )
                else:
                    date_range_plot = None
            
            fig = plot_regions(
                counts_df,
                "Transaction Counts Over Time",
                "Number of Transactions",
                selected_regions=plot_regions_selected,
                date_range=date_range_plot if len(available_quarters) > 1 else None
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Index tabs
        for i in range(4):
            with tabs[i+6]:
                ma_label = f"Moving Average ({i+1} Quarter{'s' if i > 0 else ''})"
                
                use_total = st.session_state.get('use_total_eur_m2', False)
                prop_type = st.session_state.get('property_type', 'Houses')
                
                # Set base period based on property type
                if prop_type in ['Forest land', 'Land residential']:
                    base_period = "2023-Q2"
                elif prop_type in ['Agricultural land', 'Land commercial', 'Other land']:
                    base_period = "2021-Q1"
                elif prop_type == 'Premises':
                    base_period = "2022-Q2"
                else:
                    base_period = "2020-Q1"
                st.subheader(f"📈 Price Index (Base: {base_period} = 1.0) - {ma_label}")
                
                if prop_type in ['Agricultural land', 'Forest land', 'Land commercial', 'Land residential', 'Other land']:
                    method_label = "Land_EUR_m2" if use_total else "Calculated (Price ÷ Sold Area)"
                else:
                    method_label = "Total_EUR_m2" if use_total else "Calculated (Price ÷ Sold Area)"
                st.caption(f"📊 Using: **{method_label}**")
                
                index_df = st.session_state['index_tables'][i]
                st.dataframe(index_df.round(4), use_container_width=True, height=400)
                
                # Plot
                st.markdown("#### 📉 Interactive Plot")
                
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    available_regions = list(index_df.index)
                    plot_regions_selected = st.multiselect(
                        "Select regions to display",
                        available_regions,
                        default=available_regions,
                        key=f"regions_index_{i}"
                    )
                
                with col2:
                    available_quarters = list(index_df.columns)
                    if len(available_quarters) > 1:
                        date_range_plot = st.select_slider(
                            "Date range",
                            options=available_quarters,
                            value=(available_quarters[0], available_quarters[-1]),
                            key=f"date_range_index_{i}"
                        )
                    else:
                        date_range_plot = None
                
                fig = plot_regions(
                    index_df,
                    f"Price Index Over Time - {ma_label}",
                    f"Index ({base_period} = 1.0)",
                    selected_regions=plot_regions_selected,
                    date_range=date_range_plot if len(available_quarters) > 1 else None
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Outlier Comparison Section
                st.markdown("---")
                st.markdown("### 🔬 Index Comparison with Outlier Filtering")
                st.caption("Apply additional outlier filtering to see how it affects the index - without changing your main tables")
                
                # Outlier controls
                col1, col2, col3 = st.columns([1, 1, 1])
                
                with col1:
                    enable_comparison_outlier = st.checkbox(
                        "Enable outlier filtering for comparison",
                        value=False,
                        key=f"enable_comparison_outlier_{i}"
                    )
                
                if enable_comparison_outlier:
                    with col2:
                        comparison_outlier_method = st.selectbox(
                            "Detection method:",
                            options=[
                                "IQR Method (1.5x - standard)",
                                "IQR Method (3.0x - lenient)",
                                "Percentile Method"
                            ],
                            index=0,
                            key=f"comparison_outlier_method_{i}"
                        )
                    
                    with col3:
                        if "Percentile" in comparison_outlier_method:
                            comparison_lower_pct = st.number_input(
                                "Lower %", 
                                min_value=0.0, 
                                max_value=10.0, 
                                value=1.0, 
                                step=0.5,
                                key=f"comparison_lower_pct_{i}"
                            )
                            comparison_upper_pct = st.number_input(
                                "Upper %", 
                                min_value=90.0, 
                                max_value=100.0, 
                                value=99.0, 
                                step=0.5,
                                key=f"comparison_upper_pct_{i}"
                            )
                        else:
                            comparison_lower_pct = None
                            comparison_upper_pct = None
                    
                    col_a, col_b = st.columns(2)
                    with col_a:
                        comparison_per_region = st.checkbox(
                            "Apply per region",
                            value=False,
                            key=f"comparison_per_region_{i}"
                        )
                    with col_b:
                        comparison_per_quarter = st.checkbox(
                            "Apply per quarter",
                            value=False,
                            key=f"comparison_per_quarter_{i}"
                        )
                    
                    # Apply outlier filtering
                    with st.spinner("Calculating outlier-filtered index..."):
                        df_for_comparison = st.session_state['df_filtered'].copy()
                        prop_type = st.session_state.get('property_type', 'Houses')
                        
                        # Apply outlier filter
                        outlier_mask = detect_outliers(
                            df_for_comparison,
                            use_total_eur_m2=use_total,
                            method=comparison_outlier_method,
                            lower_percentile=comparison_lower_pct,
                            upper_percentile=comparison_upper_pct,
                            per_region=comparison_per_region,
                            per_quarter=comparison_per_quarter,
                            property_type=prop_type
                        )
                        df_outlier_filtered = df_for_comparison[outlier_mask]
                        
                        # Calculate how many outliers removed
                        outliers_count = len(df_for_comparison) - len(df_outlier_filtered)
                        outlier_pct = (outliers_count / len(df_for_comparison) * 100) if len(df_for_comparison) > 0 else 0
                        
                        st.info(f"🎯 **Comparison Filter:** Removed {outliers_count:,} outliers ({outlier_pct:.1f}%) for this comparison")
                        
                        # Recalculate prices and index with outlier-filtered data
                        agg_df_outlier = aggregate_by_region_quarter(df_outlier_filtered, use_total, prop_type)
                        prices_outlier = create_prices_table(agg_df_outlier, ma_quarters=i+1)
                        index_outlier = create_index_table(prices_outlier, prop_type)
                    
                    # Region selector for comparison
                    st.markdown("#### 📊 Comparison Plot")
                    comparison_regions_selected = st.multiselect(
                        "Select regions to compare",
                        available_regions,
                        default=available_regions[:3] if len(available_regions) > 3 else available_regions,
                        key=f"comparison_regions_{i}"
                    )
                    
                    if comparison_regions_selected:
                        # Create comparison plot
                        fig_comparison = go.Figure()
                        
                        # Filter to selected date range
                        if date_range_plot and len(available_quarters) > 1:
                            cols_to_show = [col for col in index_df.columns if date_range_plot[0] <= col <= date_range_plot[1]]
                        else:
                            cols_to_show = index_df.columns
                        
                        for region in comparison_regions_selected:
                            # Original index (solid line)
                            if region in index_df.index:
                                fig_comparison.add_trace(go.Scatter(
                                    x=cols_to_show,
                                    y=index_df.loc[region, cols_to_show],
                                    mode='lines+markers',
                                    name=f"{region} (Original)",
                                    line=dict(width=2, dash='solid'),
                                    marker=dict(size=6),
                                    legendgroup=region
                                ))
                            
                            # Outlier-filtered index (dashed line)
                            if region in index_outlier.index:
                                fig_comparison.add_trace(go.Scatter(
                                    x=cols_to_show,
                                    y=index_outlier.loc[region, cols_to_show],
                                    mode='lines+markers',
                                    name=f"{region} (Outliers Removed)",
                                    line=dict(width=2, dash='dash'),
                                    marker=dict(size=4, symbol='x'),
                                    legendgroup=region
                                ))
                        
                        fig_comparison.update_layout(
                            title=f"Index Comparison: Original vs Outlier-Filtered - {ma_label}",
                            xaxis_title='Quarter',
                            yaxis_title=f'Index ({base_period} = 1.0)',
                            hovermode='x unified',
                            height=600,
                            legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.02)
                        )
                        
                        fig_comparison.update_xaxes(tickangle=45)
                        
                        st.plotly_chart(fig_comparison, use_container_width=True)
                        
                        st.caption("📖 **Legend:** Solid lines = Original data | Dashed lines = After outlier removal")
                    else:
                        st.info("👆 Select at least one region to see the comparison")
                else:
                    st.info("☝️ Enable outlier filtering above to see the comparison plot")
        
        # Distribution Analysis tab
        with tabs[10]:
            st.subheader("📊 Price Distribution Analysis by Quarter")
            st.caption("Identify abnormal price distributions and outliers in your data")
            
            df_filt = st.session_state['df_filtered']
            use_total = st.session_state.get('use_total_eur_m2', False)
            prop_type = st.session_state.get('property_type', 'Houses')
            
            # Calculate BOTH price per m² columns so users can compare
            df_filt = calculate_price_per_m2(df_filt.copy(), use_total_eur_m2=False, property_type=prop_type)  # Calculate Price_per_m2
            
            # Filter for valid data (need at least one valid price method)
            if prop_type in ['Agricultural land', 'Forest land', 'Land commercial', 'Land residential', 'Other land']:
                df_dist = df_filt[
                    (df_filt['Price_EUR'].notna() & df_filt['Sold_Area_m2'].notna() & (df_filt['Sold_Area_m2'] > 0)) |
                    (df_filt['Land_EUR_m2'].notna() & (df_filt['Land_EUR_m2'] > 0))
                ].copy()
                # Default price_col based on user's main calculation method
                price_col = 'Land_EUR_m2' if use_total else 'Price_per_m2'
            else:
                df_dist = df_filt[
                    (df_filt['Price_EUR'].notna() & df_filt['Sold_Area_m2'].notna() & (df_filt['Sold_Area_m2'] > 0)) |
                    (df_filt['Total_EUR_m2'].notna() & (df_filt['Total_EUR_m2'] > 0))
                ].copy()
                # Default price_col based on user's main calculation method
                price_col = 'Total_EUR_m2' if use_total else 'Price_per_m2'
            
            if len(df_dist) == 0:
                st.warning("⚠️ No valid price data available for distribution analysis")
            else:
                # Create YearQuarter column
                df_dist['YearQuarter'] = df_dist['Year'].astype(int).astype(str) + '-Q' + df_dist['Quarter'].astype(int).astype(str)
                
                # Region selector
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    available_regions = sorted(df_dist['region_riga_separate'].unique())
                    selected_regions_dist = st.multiselect(
                        "Select regions to analyze",
                        options=available_regions,
                        default=available_regions[:3] if len(available_regions) > 3 else available_regions,
                        key="dist_regions"
                    )
                
                with col2:
                    if prop_type in ['Agricultural land', 'Forest land', 'Land commercial', 'Land residential', 'Other land']:
                        price_metric = st.radio(
                            "Price metric:",
                            options=[
                                "Price per m² (Calculated)",
                                "Price per m² (Land_EUR_m2)",
                                "Total Price"
                            ],
                            index=0,
                            key="price_metric"
                        )
                        
                        # Show data availability for each metric
                        calc_valid = df_dist['Price_per_m2'].notna().sum()
                        land_eur_valid = df_dist['Land_EUR_m2'].notna().sum() if 'Land_EUR_m2' in df_dist.columns else 0
                        price_valid = df_dist['Price_EUR'].notna().sum()
                        
                        st.caption(f"📊 **Data availability:**")
                        st.caption(f"Calculated: {calc_valid:,} records")
                        st.caption(f"Land_EUR_m2: {land_eur_valid:,} records")
                        st.caption(f"Total Price: {price_valid:,} records")
                        
                        if price_metric == "Total Price":
                            plot_col = 'Price_EUR'
                            y_label = "Price (EUR)"
                        elif price_metric == "Price per m² (Land_EUR_m2)":
                            plot_col = 'Land_EUR_m2'
                            y_label = "Price per m² (EUR) - Land_EUR_m2"
                        else:  # "Price per m² (Calculated)"
                            plot_col = 'Price_per_m2'
                            y_label = "Price per m² (EUR) - Calculated"
                    else:
                        price_metric = st.radio(
                            "Price metric:",
                            options=[
                                "Price per m² (Calculated)",
                                "Price per m² (Total_EUR_m2)",
                                "Total Price"
                            ],
                            index=0,
                            key="price_metric"
                        )
                        
                        # Show data availability for each metric
                        calc_valid = df_dist['Price_per_m2'].notna().sum()
                        total_eur_valid = df_dist['Total_EUR_m2'].notna().sum()
                        price_valid = df_dist['Price_EUR'].notna().sum()
                        
                        st.caption(f"📊 **Data availability:**")
                        st.caption(f"Calculated: {calc_valid:,} records")
                        st.caption(f"Total_EUR_m2: {total_eur_valid:,} records")
                        st.caption(f"Total Price: {price_valid:,} records")
                        
                        if price_metric == "Total Price":
                            plot_col = 'Price_EUR'
                            y_label = "Price (EUR)"
                        elif price_metric == "Price per m² (Total_EUR_m2)":
                            plot_col = 'Total_EUR_m2'
                            y_label = "Price per m² (EUR) - Total_EUR_m2"
                        else:  # "Price per m² (Calculated)"
                            plot_col = 'Price_per_m2'
                            y_label = "Price per m² (EUR) - Calculated"
                
                if selected_regions_dist:
                    df_plot = df_dist[df_dist['region_riga_separate'].isin(selected_regions_dist)].copy()
                    
                    # Box plot by quarter
                    st.markdown("### 📦 Box Plot - Price Distribution by Quarter")
                    
                    fig_box = go.Figure()
                    
                    for region in selected_regions_dist:
                        region_data = df_plot[df_plot['region_riga_separate'] == region]
                        
                        for yq in sorted(region_data['YearQuarter'].unique()):
                            yq_data = region_data[region_data['YearQuarter'] == yq][plot_col].dropna()
                            
                            if len(yq_data) > 0:
                                fig_box.add_trace(go.Box(
                                    y=yq_data,
                                    name=f"{region} - {yq}",
                                    boxmean='sd',  # Show mean and standard deviation
                                    marker_color=px.colors.qualitative.Plotly[available_regions.index(region) % len(px.colors.qualitative.Plotly)]
                                ))
                    
                    fig_box.update_layout(
                        title="Price Distribution by Region and Quarter",
                        yaxis_title=y_label,
                        xaxis_title="Region - Quarter",
                        height=600,
                        showlegend=True,
                        boxmode='group'
                    )
                    
                    st.plotly_chart(fig_box, use_container_width=True)
                    
                    # Statistical Summary
                    st.markdown("### 📈 Statistical Summary by Quarter")
                    
                    stats_data = []
                    for region in selected_regions_dist:
                        region_data = df_plot[df_plot['region_riga_separate'] == region]
                        
                        for yq in sorted(region_data['YearQuarter'].unique()):
                            yq_data = region_data[region_data['YearQuarter'] == yq][plot_col].dropna()
                            
                            if len(yq_data) > 0:
                                # Calculate statistics
                                q1 = yq_data.quantile(0.25)
                                q3 = yq_data.quantile(0.75)
                                iqr = q3 - q1
                                lower_fence = q1 - 1.5 * iqr
                                upper_fence = q3 + 1.5 * iqr
                                outliers = yq_data[(yq_data < lower_fence) | (yq_data > upper_fence)]
                                
                                stats_data.append({
                                    'Region': region,
                                    'Quarter': yq,
                                    'Count': len(yq_data),
                                    'Mean': yq_data.mean(),
                                    'Median': yq_data.median(),
                                    'Std Dev': yq_data.std(),
                                    'Min': yq_data.min(),
                                    'Max': yq_data.max(),
                                    'Q1': q1,
                                    'Q3': q3,
                                    'IQR': iqr,
                                    'Outliers': len(outliers),
                                    'Outlier %': (len(outliers) / len(yq_data) * 100)
                                })
                    
                    stats_df = pd.DataFrame(stats_data)
                    
                    # Highlight quarters with high outlier percentages
                    def highlight_outliers(row):
                        if row['Outlier %'] > 10:
                            return ['background-color: #ffcccc'] * len(row)
                        elif row['Outlier %'] > 5:
                            return ['background-color: #ffffcc'] * len(row)
                        else:
                            return [''] * len(row)
                    
                    st.dataframe(
                        stats_df.style.apply(highlight_outliers, axis=1).format({
                            'Mean': '{:.2f}',
                            'Median': '{:.2f}',
                            'Std Dev': '{:.2f}',
                            'Min': '{:.2f}',
                            'Max': '{:.2f}',
                            'Q1': '{:.2f}',
                            'Q3': '{:.2f}',
                            'IQR': '{:.2f}',
                            'Outlier %': '{:.1f}%'
                        }),
                        use_container_width=True,
                        height=400
                    )
                    
                    st.caption("🔴 Red: >10% outliers | 🟡 Yellow: >5% outliers | ⚪ White: <5% outliers")
                    st.caption("💡 Outliers are defined as values beyond 1.5 × IQR from Q1/Q3")
                    
                    # Histogram view
                    st.markdown("### 📊 Histogram - Price Distribution")
                    
                    selected_quarter = st.selectbox(
                        "Select quarter to analyze in detail:",
                        options=sorted(df_plot['YearQuarter'].unique()),
                        key="hist_quarter"
                    )
                    
                    if selected_quarter:
                        quarter_data = df_plot[df_plot['YearQuarter'] == selected_quarter]
                        
                        fig_hist = go.Figure()
                        
                        for region in selected_regions_dist:
                            region_quarter = quarter_data[quarter_data['region_riga_separate'] == region][plot_col].dropna()
                            
                            if len(region_quarter) > 0:
                                fig_hist.add_trace(go.Histogram(
                                    x=region_quarter,
                                    name=region,
                                    opacity=0.7,
                                    nbinsx=30
                                ))
                        
                        fig_hist.update_layout(
                            title=f"Price Distribution for {selected_quarter}",
                            xaxis_title=y_label,
                            yaxis_title="Frequency",
                            height=500,
                            barmode='overlay',
                            showlegend=True
                        )
                        
                        st.plotly_chart(fig_hist, use_container_width=True)
                
                else:
                    st.info("👆 Select at least one region to see the distribution analysis")
                
                # Area Distribution Analysis
                st.markdown("---")
                st.markdown("## 📏 Area Distribution Analysis")
                st.caption("Analyze property size distributions to identify unusual listings")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    area_type = st.radio(
                        "Area type to analyze:",
                        options=["Sold Area", "Total Area"],
                        index=0,
                        key="area_type"
                    )
                
                with col2:
                    selected_regions_area = st.multiselect(
                        "Select regions for area analysis",
                        options=available_regions,
                        default=available_regions[:3] if len(available_regions) > 3 else available_regions,
                        key="area_regions"
                    )
                
                if selected_regions_area:
                    area_col = 'Sold_Area_m2' if area_type == "Sold Area" else 'Total_Area_m2'
                    area_label = "Sold Area (m²)" if area_type == "Sold Area" else "Total Area (m²)"
                    
                    # Filter for valid area data
                    df_area = df_dist[df_dist[area_col].notna() & (df_dist[area_col] > 0)].copy()
                    df_area_plot = df_area[df_area['region_riga_separate'].isin(selected_regions_area)].copy()
                    
                    if len(df_area_plot) > 0:
                        # Box plot for area distribution
                        st.markdown(f"### 📦 Box Plot - {area_label} by Quarter")
                        
                        fig_area_box = go.Figure()
                        
                        for region in selected_regions_area:
                            region_data = df_area_plot[df_area_plot['region_riga_separate'] == region]
                            
                            for yq in sorted(region_data['YearQuarter'].unique()):
                                yq_data = region_data[region_data['YearQuarter'] == yq][area_col].dropna()
                                
                                if len(yq_data) > 0:
                                    fig_area_box.add_trace(go.Box(
                                        y=yq_data,
                                        name=f"{region} - {yq}",
                                        boxmean='sd',
                                        marker_color=px.colors.qualitative.Set2[available_regions.index(region) % len(px.colors.qualitative.Set2)]
                                    ))
                        
                        fig_area_box.update_layout(
                            title=f"{area_label} Distribution by Region and Quarter",
                            yaxis_title=area_label,
                            xaxis_title="Region - Quarter",
                            height=600,
                            showlegend=True,
                            boxmode='group'
                        )
                        
                        st.plotly_chart(fig_area_box, use_container_width=True)
                        
                        # Statistical Summary for Area
                        st.markdown(f"### 📈 Statistical Summary - {area_label}")
                        
                        area_stats_data = []
                        for region in selected_regions_area:
                            region_data = df_area_plot[df_area_plot['region_riga_separate'] == region]
                            
                            for yq in sorted(region_data['YearQuarter'].unique()):
                                yq_data = region_data[region_data['YearQuarter'] == yq][area_col].dropna()
                                
                                if len(yq_data) > 0:
                                    # Calculate statistics
                                    q1 = yq_data.quantile(0.25)
                                    q3 = yq_data.quantile(0.75)
                                    iqr = q3 - q1
                                    lower_fence = q1 - 1.5 * iqr
                                    upper_fence = q3 + 1.5 * iqr
                                    outliers = yq_data[(yq_data < lower_fence) | (yq_data > upper_fence)]
                                    
                                    area_stats_data.append({
                                        'Region': region,
                                        'Quarter': yq,
                                        'Count': len(yq_data),
                                        'Mean': yq_data.mean(),
                                        'Median': yq_data.median(),
                                        'Std Dev': yq_data.std(),
                                        'Min': yq_data.min(),
                                        'Max': yq_data.max(),
                                        'Q1': q1,
                                        'Q3': q3,
                                        'IQR': iqr,
                                        'Outliers': len(outliers),
                                        'Outlier %': (len(outliers) / len(yq_data) * 100)
                                    })
                        
                        area_stats_df = pd.DataFrame(area_stats_data)
                        
                        # Highlight quarters with high outlier percentages
                        st.dataframe(
                            area_stats_df.style.apply(highlight_outliers, axis=1).format({
                                'Mean': '{:.1f}',
                                'Median': '{:.1f}',
                                'Std Dev': '{:.1f}',
                                'Min': '{:.1f}',
                                'Max': '{:.1f}',
                                'Q1': '{:.1f}',
                                'Q3': '{:.1f}',
                                'IQR': '{:.1f}',
                                'Outlier %': '{:.1f}%'
                            }),
                            use_container_width=True,
                            height=400
                        )
                        
                        st.caption("🔴 Red: >10% outliers | 🟡 Yellow: >5% outliers | ⚪ White: <5% outliers")
                        
                        # Histogram for area
                        st.markdown(f"### 📊 Histogram - {area_label} Distribution")
                        
                        selected_quarter_area = st.selectbox(
                            "Select quarter to analyze in detail:",
                            options=sorted(df_area_plot['YearQuarter'].unique()),
                            key="hist_quarter_area"
                        )
                        
                        if selected_quarter_area:
                            quarter_area_data = df_area_plot[df_area_plot['YearQuarter'] == selected_quarter_area]
                            
                            fig_area_hist = go.Figure()
                            
                            for region in selected_regions_area:
                                region_quarter = quarter_area_data[quarter_area_data['region_riga_separate'] == region][area_col].dropna()
                                
                                if len(region_quarter) > 0:
                                    fig_area_hist.add_trace(go.Histogram(
                                        x=region_quarter,
                                        name=region,
                                        opacity=0.7,
                                        nbinsx=30
                                    ))
                            
                            fig_area_hist.update_layout(
                                title=f"{area_label} Distribution for {selected_quarter_area}",
                                xaxis_title=area_label,
                                yaxis_title="Frequency",
                                height=500,
                                barmode='overlay',
                                showlegend=True
                            )
                            
                            st.plotly_chart(fig_area_hist, use_container_width=True)
                        
                        # Comparison view - Area vs Price
                        st.markdown("---")
                        st.markdown("### 🔀 Correlation Analysis - Area vs Price")
                        st.caption("Identify unusual price-to-area relationships")
                        
                        selected_quarter_scatter = st.selectbox(
                            "Select quarter for scatter plot:",
                            options=sorted(df_area_plot['YearQuarter'].unique()),
                            key="scatter_quarter"
                        )
                        
                        if selected_quarter_scatter:
                            scatter_data = df_area_plot[df_area_plot['YearQuarter'] == selected_quarter_scatter].copy()
                            
                            # Determine price column to use
                            if use_total:
                                scatter_price_col = 'Total_EUR_m2'
                                scatter_y_label = "Price per m² (EUR)"
                            else:
                                scatter_price_col = 'Price_per_m2'
                                scatter_y_label = "Price per m² (EUR)"
                            
                            fig_scatter = go.Figure()
                            
                            for region in selected_regions_area:
                                region_scatter = scatter_data[scatter_data['region_riga_separate'] == region]
                                
                                if len(region_scatter) > 0:
                                    fig_scatter.add_trace(go.Scatter(
                                        x=region_scatter[area_col],
                                        y=region_scatter[scatter_price_col],
                                        mode='markers',
                                        name=region,
                                        marker=dict(
                                            size=8,
                                            opacity=0.6
                                        ),
                                        text=region_scatter.apply(lambda row: f"Price/m²: {row[scatter_price_col]:.0f}<br>Area: {row[area_col]:.0f}<br>Total: €{row['Price_EUR']:,.0f}", axis=1),
                                        hovertemplate='%{text}<extra></extra>'
                                    ))
                            
                            fig_scatter.update_layout(
                                title=f"{area_label} vs Price per m² - {selected_quarter_scatter}",
                                xaxis_title=area_label,
                                yaxis_title=scatter_y_label,
                                height=600,
                                showlegend=True
                            )
                            
                            st.plotly_chart(fig_scatter, use_container_width=True)
                            st.caption("💡 Outliers far from the general trend may indicate data quality issues or unique properties")
                    
                    else:
                        st.warning(f"⚠️ No valid {area_label} data available for selected regions")
                else:
                    st.info("👆 Select at least one region to see the area distribution analysis")
        
        # Export to Excel
        st.markdown("---")
        st.header("📥 Export Results")
        
        if st.button("📊 Generate Excel Report", type="secondary"):
            with st.spinner("Generating Excel report..."):
                df_filt = st.session_state['df_filtered']
                use_total = st.session_state.get('use_total_eur_m2', False)
                prop_type = st.session_state.get('property_type', 'Houses')
                df_filt = calculate_price_per_m2(df_filt, use_total, prop_type)
                
                summary = df_filt.groupby('region_riga_separate').agg({
                    'Price_EUR': ['count', 'sum', 'mean', 'median'],
                    'Sold_Area_m2': ['sum', 'mean', 'median'],
                    'Price_per_m2': ['mean', 'median', 'std']
                }).round(2)
                
                excel_data = export_to_excel(
                    summary,
                    st.session_state['prices_tables'],
                    st.session_state['counts_table'],
                    st.session_state['index_tables']
                )
                
                property_label = property_type.lower()
                st.download_button(
                    label="⬇️ Download Excel Report",
                    data=excel_data,
                    file_name=f"latvian_{property_label}_index_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
                
                st.success("✅ Excel report ready for download!")
        
        # Region Merging Section (standalone)
        st.markdown("---")
        st.header(f"🔗 Merge & Compare Regions - {property_type}")
        st.markdown("**Combine multiple regions to analyze aggregate trends across all metrics**")
        
        # Get all available regions
        all_regions = list(st.session_state['prices_tables'][0].index)
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            merge_regions_selected = st.multiselect(
                "Select 2 or more regions to merge",
                all_regions,
                key="standalone_merge_regions"
            )
        
        with col2:
            merge_name = st.text_input(
                "Merged region name",
                value="Combined Region",
                key="standalone_merge_name"
            )
        
        show_comparison = st.checkbox(
            "Show comparison with individual regions",
            value=True,
            key="standalone_show_comparison"
        )
        
        if st.button("🚀 Generate Merged Analysis", type="primary", key="standalone_merge_btn"):
            if len(merge_regions_selected) < 2:
                st.error("⚠️ Please select at least 2 regions to merge")
            else:
                with st.spinner("Calculating merged region data..."):
                    # Get calculation method from session state
                    use_total = st.session_state.get('use_total_eur_m2', False)
                    prop_type = st.session_state.get('property_type', 'Houses')
                    
                    # Filter to selected regions and relabel as merged
                    df_merged_regions = st.session_state['df_filtered'][
                        st.session_state['df_filtered']['region_riga_separate'].isin(merge_regions_selected)
                    ].copy()
                    df_merged_regions['region_riga_separate'] = merge_name
                    
                    # Aggregate the merged data
                    merged_agg = aggregate_by_region_quarter(df_merged_regions, use_total, prop_type)
                    
                    # Calculate all merged tables
                    merged_prices_orig = create_prices_table(
                        merged_agg[merged_agg['region_riga_separate'] == merge_name],
                        ma_quarters=1
                    )
                    merged_prices_ma2 = create_prices_table(
                        merged_agg[merged_agg['region_riga_separate'] == merge_name],
                        ma_quarters=2
                    )
                    merged_prices_ma3 = create_prices_table(
                        merged_agg[merged_agg['region_riga_separate'] == merge_name],
                        ma_quarters=3
                    )
                    merged_prices_ma4 = create_prices_table(
                        merged_agg[merged_agg['region_riga_separate'] == merge_name],
                        ma_quarters=4
                    )
                    
                    merged_index_orig = create_index_table(merged_prices_orig, prop_type)
                    merged_index_ma2 = create_index_table(merged_prices_ma2, prop_type)
                    merged_index_ma3 = create_index_table(merged_prices_ma3, prop_type)
                    merged_index_ma4 = create_index_table(merged_prices_ma4, prop_type)
                
                st.success(f"✅ Merged analysis generated for: **{merge_name}** ({', '.join(merge_regions_selected)})")
                
                # Set base period based on property type
                if prop_type in ['Forest land', 'Land residential']:
                    base_period = "2023-Q2"
                elif prop_type in ['Agricultural land', 'Land commercial', 'Other land']:
                    base_period = "2021-Q1"
                elif prop_type == 'Premises':
                    base_period = "2022-Q2"
                else:
                    base_period = "2020-Q1"
                
                if prop_type in ['Agricultural land', 'Forest land', 'Land commercial', 'Land residential', 'Other land']:
                    method_label = "Land_EUR_m2" if use_total else "Calculated (Price ÷ Sold Area)"
                else:
                    method_label = "Total_EUR_m2" if use_total else "Calculated (Price ÷ Sold Area)"
                st.info(f"📊 Using calculation method: **{method_label}**")
                
                # Create tabs for merged results
                merge_tabs = st.tabs([
                    "Prices - Original",
                    "Prices - MA2",
                    "Prices - MA3",
                    "Prices - MA4",
                    "Index - Original",
                    "Index - MA2",
                    "Index - MA3",
                    "Index - MA4"
                ])
                
                # Prices tabs
                merged_prices_list = [merged_prices_orig, merged_prices_ma2, merged_prices_ma3, merged_prices_ma4]
                for idx, (tab, merged_prices) in enumerate(zip(merge_tabs[:4], merged_prices_list)):
                    with tab:
                        ma_label = f"MA{idx+1}" if idx > 0 else "Original"
                        st.subheader(f"💰 {merge_name} - Average Price per m² ({ma_label})")
                        
                        st.dataframe(merged_prices.round(2), use_container_width=True)
                        
                        if show_comparison:
                            # Show comparison with individual regions
                            original_regions_data = st.session_state['prices_tables'][idx].loc[merge_regions_selected]
                            comparison_df = pd.concat([original_regions_data, merged_prices])
                            
                            fig = plot_regions(
                                comparison_df,
                                f"Comparison: {merge_name} vs Individual Regions ({ma_label})",
                                "Price per m² (EUR)"
                            )
                        else:
                            fig = plot_regions(
                                merged_prices,
                                f"{merge_name} - Price per m² ({ma_label})",
                                "Price per m² (EUR)"
                            )
                        
                        st.plotly_chart(fig, use_container_width=True)
                
                # Index tabs
                merged_index_list = [merged_index_orig, merged_index_ma2, merged_index_ma3, merged_index_ma4]
                for idx, (tab, merged_index) in enumerate(zip(merge_tabs[4:], merged_index_list)):
                    with tab:
                        ma_label = f"MA{idx+1}" if idx > 0 else "Original"
                        st.subheader(f"📈 {merge_name} - Price Index ({ma_label})")
                        
                        st.dataframe(merged_index.round(4), use_container_width=True)
                        
                        if show_comparison:
                            # Show comparison with individual regions
                            original_regions_data = st.session_state['index_tables'][idx].loc[merge_regions_selected]
                            comparison_df = pd.concat([original_regions_data, merged_index])
                            
                            fig = plot_regions(
                                comparison_df,
                                f"Comparison: {merge_name} vs Individual Regions ({ma_label})",
                                f"Index ({base_period} = 1.0)"
                            )
                        else:
                            fig = plot_regions(
                                merged_index,
                                f"{merge_name} - Price Index ({ma_label})",
                                f"Index ({base_period} = 1.0)"
                            )
                        
                        st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()


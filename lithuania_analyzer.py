import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import io

# Helper functions for Lithuania data
@st.cache_data
def load_lithuania_data(property_type='Apartments'):
    """Load and prepare Lithuanian data from Bigbank Excel file"""
    filename = 'Bigbank_purchase transaction statistics_202506 Lithuania EN.xlsx'
    
    # Map property types to sheet names
    sheet_map = {
        'Apartments': 'apartments_aggregated',
        'Houses': 'Houses agregated data',
        'Office Premises': 'Office pr. agregated data',
        'Retail Premises': 'retail pr._agregated data',
        'Hotel/Recreation': 'hotel_recr._agregated data'
    }
    
    sheet_name = sheet_map.get(property_type)
    if not sheet_name:
        st.error(f"Unknown property type: {property_type}")
        return pd.DataFrame()
    
    df = pd.read_excel(filename, sheet_name=sheet_name)
    
    # Standardize column names
    df = df.rename(columns={
        'Municipality group': 'Region',
        'Count (no of transactions)': 'Count',
        'Acquired area (sq. m)': 'Area_sqm',
        'Price per sqm_avg': 'Price_avg',
        'Price per sqm_weighted': 'Price_weighted',
        'Price per sqm_median': 'Price_median'
    })
    
    # Create YearQuarter identifier
    df['YearQuarter'] = df['Year'].astype(str) + '-Q' + df['Quarter'].astype(str)
    
    # Create date for sorting/filtering
    df['Date'] = pd.to_datetime(df['Year'].astype(str) + '-' + (df['Quarter']*3).astype(str) + '-01')
    
    return df

def create_region_mapping():
    """Create mapping of region numbers to descriptive names"""
    return {
        '1 region': '1 - Vilnius City',
        '2 region': '2 - Vilnius District',
        '3 region': '3 - Kaunas City',
        '4 region': '4 - Kaunas District',
        '5 region': '5 - Klaipėda City & District',
        '6 region': '6 - Palanga City',
        '7 region': '7 - Neringa (Resort)',
        '8 region': '8 - Alytus, Šiauliai, Panevėžys',
        '9 region': '9 - Trakai District',
        '10 region': '10 - Druskininkai, Birštonas',
        '11 region': '11 - Mid-sized municipalities',
        '12 region': '12 - Other municipalities'
    }

def create_prices_table_lt(df, price_metric='Price_weighted', ma_quarters=1):
    """Create prices pivot table with optional moving average"""
    pivot = df.pivot_table(
        index='Region',
        columns='YearQuarter',
        values=price_metric,
        aggfunc='first'
    )
    
    if ma_quarters > 1:
        # Apply moving average across columns
        pivot = pivot.rolling(window=ma_quarters, axis=1, min_periods=1).mean()
    
    return pivot

def create_index_table_lt(df, base_year=2021, base_quarter=1, price_metric='Price_weighted', ma_quarters=1):
    """Create index table with base period and optional moving average"""
    # First create prices table with MA if needed
    prices_pivot = create_prices_table_lt(df, price_metric, ma_quarters)
    
    base_col = f'{base_year}-Q{base_quarter}'
    
    if base_col not in prices_pivot.columns:
        st.warning(f"Base period {base_col} not found in data")
        return prices_pivot
    
    # Divide all values by the base column
    index_df = prices_pivot.div(prices_pivot[base_col], axis=0)
    
    return index_df

def calculate_changes_lt(df_pivot):
    """Calculate quarter-over-quarter and year-over-year changes"""
    # QoQ change
    qoq_change = df_pivot.pct_change(axis=1) * 100
    
    # YoY change (4 quarters back)
    yoy_change = df_pivot.pct_change(periods=4, axis=1) * 100
    
    return qoq_change, yoy_change

def detect_outliers_lt(df, price_metric='Price_weighted', method="IQR Method (1.5x)", 
                       lower_percentile=None, upper_percentile=None, per_region=True):
    """
    Detect outlier quarters based on price distribution
    Returns a boolean mask where True = keep, False = outlier (remove)
    
    Note: For aggregated data, this identifies unusual quarters, not transactions
    """
    # Create a mask - True = keep, False = remove
    keep_mask = pd.Series(True, index=df.index)
    
    # Get prices
    prices = df[price_metric].copy()
    
    # Filter valid prices
    valid_mask = prices.notna() & (prices > 0)
    
    if not valid_mask.any():
        return keep_mask
    
    # Determine grouping
    if per_region:
        groups = df[valid_mask].groupby('Region')
    else:
        # Global - treat all data as one group
        groups = [('global', df[valid_mask])]
    
    # Apply outlier detection to each group
    for group_key, group_df in groups:
        group_prices = group_df[price_metric]
        
        if len(group_prices) < 4:  # Need at least 4 quarters
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
        
        # Identify outliers in this group
        is_outlier = (group_prices < lower_bound) | (group_prices > upper_bound)
        
        # Mark as False (remove) in keep_mask
        keep_mask.loc[is_outlier[is_outlier].index] = False
    
    return keep_mask

def plot_regions_lt(df_pivot, title, yaxis_title, selected_regions=None, date_range=None):
    """Create interactive line plot for Lithuania data"""
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

def export_to_excel_lt(prices_tables, counts_df, area_df, index_tables, qoq_change, yoy_change, property_type):
    """Export Lithuania tables to Excel with multiple moving averages"""
    output = io.BytesIO()
    
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        # Write prices with different MA levels
        for i, prices_df in enumerate(prices_tables):
            sheet_name = f'Prices_MA{i+1}' if i > 0 else 'Prices_Original'
            prices_df.to_excel(writer, sheet_name=sheet_name)
        
        # Write counts
        counts_df.to_excel(writer, sheet_name='Counts')
        
        # Write area
        area_df.to_excel(writer, sheet_name='Area')
        
        # Write indices with different MA levels
        for i, index_df in enumerate(index_tables):
            sheet_name = f'Index_MA{i+1}' if i > 0 else 'Index_Original'
            index_df.to_excel(writer, sheet_name=sheet_name)
        
        # Write changes
        qoq_change.to_excel(writer, sheet_name='QoQ_Changes')
        yoy_change.to_excel(writer, sheet_name='YoY_Changes')
        
        # Add metadata sheet
        metadata = pd.DataFrame({
            'Property Type': [property_type],
            'Report Date': [datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
            'Data Source': ['Bigbank Lithuania Q2 2025'],
            'Sheets': ['Prices (4), Counts, Area, Index (4), Changes (2), Metadata'],
            'Total Sheets': [13]
        })
        metadata.to_excel(writer, sheet_name='Metadata', index=False)
    
    return output.getvalue()

# Main Lithuania analyzer function
def lithuania_analyzer():
    st.title("🏛️ Lithuania Real Estate Price Index Analyzer")
    st.caption("📊 Data source: Bigbank Purchase Transaction Statistics Q2 2025")
    
    # Property type selector
    property_type = st.radio(
        "**Select Property Type:**",
        options=["Apartments", "Houses", "Office Premises", "Retail Premises", "Hotel/Recreation"],
        horizontal=True,
        help="Switch between different property types in Lithuania",
        key="lt_property_type_radio"
    )
    
    st.markdown("---")
    
    # Load data
    with st.spinner(f"Loading {property_type.lower()} data..."):
        df = load_lithuania_data(property_type)
    
    if df.empty:
        st.error("Failed to load data. Please check the Excel file.")
        return
    
    st.success(f"✅ Loaded {len(df):,} records covering {len(df['YearQuarter'].unique())} quarters across {len(df['Region'].unique())} regions")
    
    # Sidebar filters
    st.sidebar.header("📊 Filters - Lithuania")
    st.sidebar.caption(f"Analyzing: **{property_type}**")
    
    # Region mapping
    region_map = create_region_mapping()
    
    # Price metric selector
    st.sidebar.markdown("---")
    st.sidebar.subheader("⚙️ Price Metric")
    price_metric = st.sidebar.radio(
        "Select price metric:",
        options=["Price_weighted", "Price_avg", "Price_median"],
        format_func=lambda x: {
            "Price_weighted": "Weighted Average (recommended)",
            "Price_avg": "Arithmetic Average",
            "Price_median": "Median"
        }[x],
        help="Weighted average accounts for transaction sizes",
        key="lt_price_metric_radio"
    )
    
    # Date filters
    with st.sidebar.expander("📅 Date & Time", expanded=True):
        min_year = int(df['Year'].min())
        max_year = int(df['Year'].max())
        year_range = st.slider(
            "Year Range",
            min_value=min_year,
            max_value=max_year,
            value=(min_year, max_year)
        )
        
        quarters = st.multiselect(
            "Quarter",
            options=[1, 2, 3, 4],
            default=[1, 2, 3, 4],
            format_func=lambda x: f"Q{x}"
        )
    
    # Region filter
    with st.sidebar.expander("🗺️ Regions", expanded=True):
        regions = sorted(df['Region'].unique())
        
        col1, col2 = st.columns([3, 1])
        with col1:
            st.write("**Select Regions**")
        with col2:
            select_all_regions = st.checkbox("All", value=True, key="lt_all_regions")
        
        if select_all_regions:
            selected_regions = regions
            st.info(f"✓ All {len(regions)} regions selected")
        else:
            selected_regions = []
            for region in regions:
                display_name = region_map.get(region, region)
                if st.checkbox(display_name, value=False, key=f"lt_region_{region}"):
                    selected_regions.append(region)
    
    # Base period selector
    with st.sidebar.expander("📈 Index Base Period", expanded=True):
        st.write("**Set base period for index calculation**")
        
        available_years = sorted(df['Year'].unique())
        base_year = st.selectbox(
            "Base Year",
            options=available_years,
            index=0,
            key="lt_base_year_select"
        )
        
        base_quarter = st.selectbox(
            "Base Quarter",
            options=[1, 2, 3, 4],
            format_func=lambda x: f"Q{x}",
            key="lt_base_quarter_select"
        )
        
        st.caption(f"Index base: **{base_year}-Q{base_quarter} = 1.0**")
    
    # Transaction count filter
    with st.sidebar.expander("📊 Transaction Filter", expanded=False):
        min_count = st.slider(
            "Minimum transactions per quarter",
            min_value=0,
            max_value=100,
            value=0,
            help="Filter out quarters with too few transactions"
        )
    
    # Outlier Detection
    with st.sidebar.expander("🎯 Outlier Detection & Removal", expanded=False):
        st.write("**Remove Statistical Outliers**")
        st.caption("Filter unusual quarters based on price distribution")
        
        enable_outlier_filter = st.checkbox(
            "Enable outlier filtering",
            value=False,
            key="lt_enable_outlier_filter",
            help="Remove quarters with extreme price per m² values"
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
                help="IQR = Interquartile Range. 1.5x removes more outliers, 3.0x only extreme ones"
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
                value=True,
                help="Calculate outliers for each region independently (recommended)"
            )
            
            st.caption("⚠️ Outlier quarters will be removed from analysis")
    
    # Reset button
    st.sidebar.markdown("---")
    if st.sidebar.button("🔄 Reset All Filters", use_container_width=True, key="lt_reset"):
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
        df_filtered = df_filtered[df_filtered['Region'].isin(selected_regions)]
    
    # Transaction count filter
    if min_count > 0:
        df_filtered = df_filtered[df_filtered['Count'] >= min_count]
    
    # Outlier removal
    records_before_outlier = len(df_filtered)
    outliers_removed = 0
    
    if enable_outlier_filter:
        outlier_mask = detect_outliers_lt(
            df_filtered,
            price_metric=price_metric,
            method=outlier_method,
            lower_percentile=lower_percentile,
            upper_percentile=upper_percentile,
            per_region=apply_per_region
        )
        df_filtered = df_filtered[outlier_mask]
        outliers_removed = records_before_outlier - len(df_filtered)
    
    # Display filter summary
    filter_summary = []
    if year_range != (min_year, max_year):
        filter_summary.append(f"Years: {year_range[0]}-{year_range[1]}")
    if len(quarters) < 4:
        filter_summary.append(f"Quarters: {', '.join([f'Q{q}' for q in quarters])}")
    if not select_all_regions:
        filter_summary.append(f"Regions: {len(selected_regions)} selected")
    if min_count > 0:
        filter_summary.append(f"Min transactions: {min_count}")
    if enable_outlier_filter:
        filter_summary.append(f"Outlier filtering: {outlier_method}")
    
    if filter_summary:
        st.info(f"📈 Filtered to {len(df_filtered):,} records | Active filters: {' • '.join(filter_summary)}")
    else:
        st.info(f"📈 Showing all {len(df_filtered):,} records (no filters applied)")
    
    # Show outlier removal summary
    if enable_outlier_filter:
        if outliers_removed > 0:
            outlier_pct = (outliers_removed / records_before_outlier * 100)
            grouping_text = "per region" if apply_per_region else "globally"
            
            st.warning(f"🎯 **Outlier Quarters Removed:** {outliers_removed} quarters removed ({outlier_pct:.1f}% of data) using {outlier_method} ({grouping_text})")
            st.caption(f"📊 Before outlier filter: {records_before_outlier} quarters | After: {len(df_filtered)} quarters")
        else:
            st.info(f"🎯 **Outlier Filter Active:** No outlier quarters detected using {outlier_method}")
    
    # Generate analysis button
    if st.button("🚀 Generate Analysis", type="primary", use_container_width=True, key="lt_generate"):
        with st.spinner("Generating analysis..."):
            # Create pivot tables for prices with different moving averages
            prices_original = create_prices_table_lt(df_filtered, price_metric, ma_quarters=1)
            prices_ma2 = create_prices_table_lt(df_filtered, price_metric, ma_quarters=2)
            prices_ma3 = create_prices_table_lt(df_filtered, price_metric, ma_quarters=3)
            prices_ma4 = create_prices_table_lt(df_filtered, price_metric, ma_quarters=4)
            
            # Create pivot table for counts
            counts_df = df_filtered.pivot_table(
                index='Region',
                columns='YearQuarter',
                values='Count',
                aggfunc='first',
                fill_value=0
            )
            
            # Create pivot table for area
            area_df = df_filtered.pivot_table(
                index='Region',
                columns='YearQuarter',
                values='Area_sqm',
                aggfunc='first',
                fill_value=0
            )
            
            # Calculate indices with different moving averages
            index_original = create_index_table_lt(df_filtered, base_year, base_quarter, price_metric, ma_quarters=1)
            index_ma2 = create_index_table_lt(df_filtered, base_year, base_quarter, price_metric, ma_quarters=2)
            index_ma3 = create_index_table_lt(df_filtered, base_year, base_quarter, price_metric, ma_quarters=3)
            index_ma4 = create_index_table_lt(df_filtered, base_year, base_quarter, price_metric, ma_quarters=4)
            
            # Calculate changes (QoQ and YoY)
            qoq_change, yoy_change = calculate_changes_lt(prices_original)
            
            # Store in session state
            st.session_state['lt_prices_tables'] = [prices_original, prices_ma2, prices_ma3, prices_ma4]
            st.session_state['lt_counts_df'] = counts_df
            st.session_state['lt_area_df'] = area_df
            st.session_state['lt_index_tables'] = [index_original, index_ma2, index_ma3, index_ma4]
            st.session_state['lt_qoq_change'] = qoq_change
            st.session_state['lt_yoy_change'] = yoy_change
            st.session_state['lt_df_filtered'] = df_filtered
            st.session_state['lt_region_map'] = region_map
            st.session_state['lt_property_type'] = property_type
            st.session_state['lt_price_metric'] = price_metric
            st.session_state['lt_base_year'] = base_year
            st.session_state['lt_base_quarter'] = base_quarter
        
        st.success("✅ Analysis generated!")
    
    # Display results if analysis is generated
    if 'lt_prices_tables' in st.session_state:
        st.markdown("---")
        st.header(f"📊 Results - {st.session_state['lt_property_type']}")
        
        prices_tables = st.session_state['lt_prices_tables']
        counts_df = st.session_state['lt_counts_df']
        area_df = st.session_state['lt_area_df']
        index_tables = st.session_state['lt_index_tables']
        qoq_change = st.session_state['lt_qoq_change']
        yoy_change = st.session_state['lt_yoy_change']
        region_map = st.session_state['lt_region_map']
        price_metric_name = st.session_state['lt_price_metric']
        base_yr = st.session_state['lt_base_year']
        base_qt = st.session_state['lt_base_quarter']
        
        # Replace region codes with descriptive names for all tables
        for i in range(len(prices_tables)):
            prices_tables[i].index = prices_tables[i].index.map(lambda x: region_map.get(x, x))
        for i in range(len(index_tables)):
            index_tables[i].index = index_tables[i].index.map(lambda x: region_map.get(x, x))
        counts_df.index = counts_df.index.map(lambda x: region_map.get(x, x))
        area_df.index = area_df.index.map(lambda x: region_map.get(x, x))
        qoq_change.index = qoq_change.index.map(lambda x: region_map.get(x, x))
        yoy_change.index = yoy_change.index.map(lambda x: region_map.get(x, x))
        
        # Create tabs
        tab_names = [
            "Summary",
            "Prices - Original",
            "Prices - MA2",
            "Prices - MA3",
            "Prices - MA4",
            "Transaction Counts",
            "Total Area",
            "Index - Original",
            "Index - MA2",
            "Index - MA3",
            "Index - MA4",
            "Changes (QoQ & YoY)",
            "Regional Comparison"
        ]
        
        tabs = st.tabs(tab_names)
        
        # Summary tab
        with tabs[0]:
            st.subheader("📈 Summary Statistics")
            
            metric_label = {
                "Price_weighted": "Weighted Average Price per m²",
                "Price_avg": "Arithmetic Average Price per m²",
                "Price_median": "Median Price per m²"
            }[price_metric_name]
            
            st.caption(f"📊 Using: **{metric_label}**")
            
            # Calculate summary statistics using original prices (first table)
            prices_orig = prices_tables[0]
            summary_data = []
            for region in prices_orig.index:
                region_prices = prices_orig.loc[region].dropna()
                region_counts = counts_df.loc[region].sum()
                region_area = area_df.loc[region].sum()
                
                if len(region_prices) > 0:
                    summary_data.append({
                        'Region': region,
                        'Total Transactions': int(region_counts),
                        'Total Area (m²)': f"{region_area:,.0f}",
                        'Avg Price/m² (Latest)': f"€{region_prices.iloc[-1]:.2f}",
                        'Avg Price/m² (All Time)': f"€{region_prices.mean():.2f}",
                        'Price Change': f"{((region_prices.iloc[-1] / region_prices.iloc[0] - 1) * 100):.1f}%" if len(region_prices) > 1 else "N/A",
                        'Quarters Covered': len(region_prices)
                    })
            
            summary_df = pd.DataFrame(summary_data)
            st.dataframe(summary_df, use_container_width=True, height=400)
            
            # Key insights
            st.markdown("### 🔑 Key Insights")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                total_transactions = int(counts_df.sum().sum())
                st.metric("Total Transactions", f"{total_transactions:,}")
            
            with col2:
                total_area = area_df.sum().sum()
                st.metric("Total Area Acquired", f"{total_area:,.0f} m²")
            
            with col3:
                avg_price_latest = prices_orig.iloc[:, -1].mean()
                st.metric("Avg Price/m² (Latest)", f"€{avg_price_latest:.2f}")
            
            with col4:
                # Calculate average YoY change for latest quarter
                latest_yoy = yoy_change.iloc[:, -1].mean()
                st.metric("Avg YoY Change", f"{latest_yoy:.1f}%")
        
        # Prices tabs (4 levels: Original, MA2, MA3, MA4)
        for i in range(4):
            with tabs[i+1]:
                ma_label = f"Moving Average ({i+1} Quarter{'s' if i > 0 else ''})" if i > 0 else "Original"
                st.subheader(f"💰 Price per m² - {ma_label}")
                st.caption(f"📊 Using: **{metric_label}**")
                
                prices_df_ma = prices_tables[i]
                st.dataframe(prices_df_ma.round(2), use_container_width=True, height=400)
                
                # Plot
                st.markdown("#### 📉 Interactive Plot")
                
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    available_regions = list(prices_df_ma.index)
                    plot_regions_selected = st.multiselect(
                        "Select regions to display",
                        available_regions,
                        default=available_regions,
                        key=f"lt_prices_plot_regions_ma{i}"
                    )
                
                with col2:
                    available_quarters = list(prices_df_ma.columns)
                    if len(available_quarters) > 1:
                        date_range_plot = st.select_slider(
                            "Date range",
                            options=available_quarters,
                            value=(available_quarters[0], available_quarters[-1]),
                            key=f"lt_prices_date_range_ma{i}"
                        )
                    else:
                        date_range_plot = None
                
                fig = plot_regions_lt(
                    prices_df_ma,
                    f"Price per m² Over Time - {ma_label}",
                    "Price per m² (EUR)",
                    selected_regions=plot_regions_selected,
                    date_range=date_range_plot if len(available_quarters) > 1 else None
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # Counts tab
        with tabs[5]:
            st.subheader("📊 Transaction Counts by Region and Quarter")
            
            st.dataframe(counts_df.astype(int), use_container_width=True, height=400)
            
            # Plot
            st.markdown("#### 📉 Interactive Plot")
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                available_regions = list(counts_df.index)
                plot_regions_selected = st.multiselect(
                    "Select regions to display",
                    available_regions,
                    default=available_regions,
                    key="lt_counts_plot_regions"
                )
            
            with col2:
                available_quarters = list(counts_df.columns)
                if len(available_quarters) > 1:
                    date_range_plot = st.select_slider(
                        "Date range",
                        options=available_quarters,
                        value=(available_quarters[0], available_quarters[-1]),
                        key="lt_counts_date_range"
                    )
                else:
                    date_range_plot = None
            
            fig = plot_regions_lt(
                counts_df,
                f"Transaction Counts Over Time - {st.session_state['lt_property_type']}",
                "Number of Transactions",
                selected_regions=plot_regions_selected,
                date_range=date_range_plot if len(available_quarters) > 1 else None
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Area tab
        with tabs[6]:
            st.subheader("📏 Total Acquired Area by Region and Quarter")
            
            st.dataframe(area_df.round(2), use_container_width=True, height=400)
            
            # Plot
            st.markdown("#### 📉 Interactive Plot")
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                available_regions = list(area_df.index)
                plot_regions_selected = st.multiselect(
                    "Select regions to display",
                    available_regions,
                    default=available_regions,
                    key="lt_area_plot_regions"
                )
            
            with col2:
                available_quarters = list(area_df.columns)
                if len(available_quarters) > 1:
                    date_range_plot = st.select_slider(
                        "Date range",
                        options=available_quarters,
                        value=(available_quarters[0], available_quarters[-1]),
                        key="lt_area_date_range"
                    )
                else:
                    date_range_plot = None
            
            fig = plot_regions_lt(
                area_df,
                f"Total Acquired Area Over Time - {st.session_state['lt_property_type']}",
                "Total Area (m²)",
                selected_regions=plot_regions_selected,
                date_range=date_range_plot if len(available_quarters) > 1 else None
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Index tabs (4 levels: Original, MA2, MA3, MA4)
        for i in range(4):
            with tabs[i+7]:
                ma_label = f"Moving Average ({i+1} Quarter{'s' if i > 0 else ''})" if i > 0 else "Original"
                st.subheader(f"📈 Price Index (Base: {base_yr}-Q{base_qt} = 1.0) - {ma_label}")
                st.caption(f"📊 Using: **{metric_label}**")
                
                index_df_ma = index_tables[i]
                st.dataframe(index_df_ma.round(4), use_container_width=True, height=400)
                
                # Plot
                st.markdown("#### 📉 Interactive Plot")
                
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    available_regions = list(index_df_ma.index)
                    plot_regions_selected = st.multiselect(
                        "Select regions to display",
                        available_regions,
                        default=available_regions,
                        key=f"lt_index_plot_regions_ma{i}"
                    )
                
                with col2:
                    available_quarters = list(index_df_ma.columns)
                    if len(available_quarters) > 1:
                        date_range_plot = st.select_slider(
                            "Date range",
                            options=available_quarters,
                            value=(available_quarters[0], available_quarters[-1]),
                            key=f"lt_index_date_range_ma{i}"
                        )
                    else:
                        date_range_plot = None
                
                fig = plot_regions_lt(
                    index_df_ma,
                    f"Price Index Over Time - {ma_label}",
                    f"Index ({base_yr}-Q{base_qt} = 1.0)",
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
                        key=f"lt_enable_comparison_outlier_{i}"
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
                            key=f"lt_comparison_outlier_method_{i}"
                        )
                    
                    with col3:
                        if "Percentile" in comparison_outlier_method:
                            comparison_lower_pct = st.number_input(
                                "Lower %", 
                                min_value=0.0, 
                                max_value=10.0, 
                                value=1.0, 
                                step=0.5,
                                key=f"lt_comparison_lower_pct_{i}"
                            )
                            comparison_upper_pct = st.number_input(
                                "Upper %", 
                                min_value=90.0, 
                                max_value=100.0, 
                                value=99.0, 
                                step=0.5,
                                key=f"lt_comparison_upper_pct_{i}"
                            )
                        else:
                            comparison_lower_pct = None
                            comparison_upper_pct = None
                    
                    col_a, col_b = st.columns(2)
                    with col_a:
                        comparison_per_region = st.checkbox(
                            "Apply per region",
                            value=True,
                            key=f"lt_comparison_per_region_{i}"
                        )
                    
                    # Apply outlier filtering for comparison
                    with st.spinner("Calculating outlier-filtered index..."):
                        df_for_comparison = st.session_state['lt_df_filtered'].copy()
                        
                        # Apply outlier filter
                        outlier_mask = detect_outliers_lt(
                            df_for_comparison,
                            price_metric=price_metric_name,
                            method=comparison_outlier_method,
                            lower_percentile=comparison_lower_pct,
                            upper_percentile=comparison_upper_pct,
                            per_region=comparison_per_region
                        )
                        df_outlier_filtered = df_for_comparison[outlier_mask]
                        
                        # Calculate how many outliers removed
                        outliers_count = len(df_for_comparison) - len(df_outlier_filtered)
                        outlier_pct = (outliers_count / len(df_for_comparison) * 100) if len(df_for_comparison) > 0 else 0
                        
                        st.info(f"🎯 **Comparison Filter:** Removed {outliers_count} quarters ({outlier_pct:.1f}%) for this comparison")
                        
                        # Recalculate prices and index with outlier-filtered data
                        prices_outlier = create_prices_table_lt(df_outlier_filtered, price_metric_name, ma_quarters=i+1)
                        index_outlier = create_index_table_lt(df_outlier_filtered, base_yr, base_qt, price_metric_name, ma_quarters=i+1)
                        
                        # Replace region codes with descriptive names
                        index_outlier.index = index_outlier.index.map(lambda x: region_map.get(x, x))
                    
                    # Region selector for comparison
                    st.markdown("#### 📊 Comparison Plot")
                    comparison_regions_selected = st.multiselect(
                        "Select regions to compare",
                        available_regions,
                        default=available_regions[:3] if len(available_regions) > 3 else available_regions,
                        key=f"lt_comparison_regions_{i}"
                    )
                    
                    if comparison_regions_selected:
                        # Create comparison plot
                        fig_comparison = go.Figure()
                        
                        # Filter to selected date range if applicable
                        if date_range_plot and len(available_quarters) > 1:
                            cols_to_show = [col for col in index_df_ma.columns if date_range_plot[0] <= col <= date_range_plot[1]]
                        else:
                            cols_to_show = index_df_ma.columns
                        
                        for region in comparison_regions_selected:
                            # Original index (solid line)
                            if region in index_df_ma.index:
                                fig_comparison.add_trace(go.Scatter(
                                    x=cols_to_show,
                                    y=index_df_ma.loc[region, cols_to_show],
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
                            yaxis_title=f'Index ({base_yr}-Q{base_qt} = 1.0)',
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
        
        # Changes (QoQ & YoY) tab - NEW!
        with tabs[11]:
            st.subheader("📊 Quarter-over-Quarter & Year-over-Year Changes")
            st.caption("Percentage changes in prices over time")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 📈 Quarter-over-Quarter (QoQ)")
                st.caption("Change vs. previous quarter")
                # Format for display
                qoq_display = qoq_change.round(2).copy()
                for col in qoq_display.columns:
                    qoq_display[col] = qoq_display[col].apply(lambda x: f"{x:.2f}%" if pd.notna(x) else "")
                st.dataframe(qoq_display, use_container_width=True, height=300)
            
            with col2:
                st.markdown("#### 📅 Year-over-Year (YoY)")
                st.caption("Change vs. same quarter last year")
                # Format for display
                yoy_display = yoy_change.round(2).copy()
                for col in yoy_display.columns:
                    yoy_display[col] = yoy_display[col].apply(lambda x: f"{x:.2f}%" if pd.notna(x) else "")
                st.dataframe(yoy_display, 
                           use_container_width=True, height=300)
            
            # Latest changes summary
            st.markdown("#### 🔑 Latest Changes Summary")
            latest_qoq = qoq_change.iloc[:, -1].dropna()
            latest_yoy = yoy_change.iloc[:, -1].dropna()
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Avg QoQ", f"{latest_qoq.mean():.2f}%")
            with col2:
                st.metric("Avg YoY", f"{latest_yoy.mean():.2f}%")
            with col3:
                st.metric("Max YoY Gain", f"{latest_yoy.max():.2f}%")
            with col4:
                best_region = latest_yoy.idxmax()
                st.metric("Best Performer", best_region.split(' - ')[0])
        
        # Regional Comparison tab
        with tabs[12]:
            st.subheader("🔀 Regional Comparison")
            
            st.markdown("#### Compare Latest vs Initial Prices")
            
            # Calculate price changes using original prices
            prices_orig = prices_tables[0]
            comparison_data = []
            for region in prices_orig.index:
                region_prices = prices_orig.loc[region].dropna()
                if len(region_prices) > 1:
                    initial = region_prices.iloc[0]
                    latest = region_prices.iloc[-1]
                    change = ((latest / initial - 1) * 100)
                    
                    comparison_data.append({
                        'Region': region,
                        'Initial Price (€/m²)': initial,
                        'Latest Price (€/m²)': latest,
                        'Absolute Change (€/m²)': latest - initial,
                        'Percentage Change': change,
                        'Period': f"{region_prices.index[0]} to {region_prices.index[-1]}"
                    })
            
            comparison_df = pd.DataFrame(comparison_data)
            comparison_df = comparison_df.sort_values('Percentage Change', ascending=False)
            
            st.dataframe(
                comparison_df.style.format({
                    'Initial Price (€/m²)': '€{:.2f}',
                    'Latest Price (€/m²)': '€{:.2f}',
                    'Absolute Change (€/m²)': '€{:.2f}',
                    'Percentage Change': '{:.1f}%'
                }),
                use_container_width=True,
                height=400
            )
            
            # Bar chart of percentage changes
            st.markdown("#### 📊 Price Changes by Region")
            
            fig_comparison = go.Figure()
            
            colors = ['green' if x > 0 else 'red' for x in comparison_df['Percentage Change']]
            
            fig_comparison.add_trace(go.Bar(
                x=comparison_df['Region'],
                y=comparison_df['Percentage Change'],
                marker_color=colors,
                text=comparison_df['Percentage Change'].apply(lambda x: f"{x:.1f}%"),
                textposition='outside'
            ))
            
            fig_comparison.update_layout(
                title=f"Price Change by Region - {st.session_state['lt_property_type']}",
                xaxis_title='Region',
                yaxis_title='Percentage Change (%)',
                height=500,
                showlegend=False
            )
            
            fig_comparison.update_xaxes(tickangle=45)
            
            st.plotly_chart(fig_comparison, use_container_width=True)
        
        # Export section
        st.markdown("---")
        st.header("📥 Export Results")
        
        if st.button("📊 Generate Excel Report", type="secondary", key="lt_export"):
            with st.spinner("Generating Excel report..."):
                excel_data = export_to_excel_lt(
                    st.session_state['lt_prices_tables'],
                    st.session_state['lt_counts_df'],
                    st.session_state['lt_area_df'],
                    st.session_state['lt_index_tables'],
                    st.session_state['lt_qoq_change'],
                    st.session_state['lt_yoy_change'],
                    st.session_state['lt_property_type']
                )
                
                property_label = st.session_state['lt_property_type'].lower().replace(' ', '_').replace('/', '_')
                st.download_button(
                    label="⬇️ Download Excel Report",
                    data=excel_data,
                    file_name=f"lithuania_{property_label}_index_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
                
                st.success("✅ Excel report ready for download!")

if __name__ == "__main__":
    lithuania_analyzer()


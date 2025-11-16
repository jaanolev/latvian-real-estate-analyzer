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

def create_index_table_lt(df, base_year=2021, base_quarter=1, price_metric='Price_weighted'):
    """Create index table with base period"""
    base_col = f'{base_year}-Q{base_quarter}'
    
    # Pivot the data
    pivot = df.pivot_table(
        index='Region',
        columns='YearQuarter',
        values=price_metric,
        aggfunc='first'
    )
    
    if base_col not in pivot.columns:
        st.warning(f"Base period {base_col} not found in data")
        return pivot
    
    # Divide all values by the base column
    index_df = pivot.div(pivot[base_col], axis=0)
    
    return index_df

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

def export_to_excel_lt(prices_df, counts_df, index_df, property_type):
    """Export Lithuania tables to Excel"""
    output = io.BytesIO()
    
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        # Write prices
        prices_df.to_excel(writer, sheet_name='Prices')
        
        # Write counts
        counts_df.to_excel(writer, sheet_name='Counts')
        
        # Write index
        index_df.to_excel(writer, sheet_name='Index')
        
        # Add metadata sheet
        metadata = pd.DataFrame({
            'Property Type': [property_type],
            'Report Date': [datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
            'Data Source': ['Bigbank Lithuania Q2 2025']
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
        key="lt_property_type"
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
        help="Weighted average accounts for transaction sizes"
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
            key="lt_base_year"
        )
        
        base_quarter = st.selectbox(
            "Base Quarter",
            options=[1, 2, 3, 4],
            format_func=lambda x: f"Q{x}",
            key="lt_base_quarter"
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
    
    if filter_summary:
        st.info(f"📈 Filtered to {len(df_filtered):,} records | Active filters: {' • '.join(filter_summary)}")
    else:
        st.info(f"📈 Showing all {len(df_filtered):,} records (no filters applied)")
    
    # Generate analysis button
    if st.button("🚀 Generate Analysis", type="primary", use_container_width=True, key="lt_generate"):
        with st.spinner("Generating analysis..."):
            # Create pivot tables for prices
            prices_df = df_filtered.pivot_table(
                index='Region',
                columns='YearQuarter',
                values=price_metric,
                aggfunc='first'
            )
            
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
            
            # Calculate index
            index_df = create_index_table_lt(df_filtered, base_year, base_quarter, price_metric)
            
            # Store in session state
            st.session_state['lt_prices_df'] = prices_df
            st.session_state['lt_counts_df'] = counts_df
            st.session_state['lt_area_df'] = area_df
            st.session_state['lt_index_df'] = index_df
            st.session_state['lt_df_filtered'] = df_filtered
            st.session_state['lt_region_map'] = region_map
            st.session_state['lt_property_type'] = property_type
            st.session_state['lt_price_metric'] = price_metric
            st.session_state['lt_base_year'] = base_year
            st.session_state['lt_base_quarter'] = base_quarter
        
        st.success("✅ Analysis generated!")
    
    # Display results if analysis is generated
    if 'lt_prices_df' in st.session_state:
        st.markdown("---")
        st.header(f"📊 Results - {st.session_state['lt_property_type']}")
        
        prices_df = st.session_state['lt_prices_df']
        counts_df = st.session_state['lt_counts_df']
        area_df = st.session_state['lt_area_df']
        index_df = st.session_state['lt_index_df']
        region_map = st.session_state['lt_region_map']
        price_metric_name = st.session_state['lt_price_metric']
        base_yr = st.session_state['lt_base_year']
        base_qt = st.session_state['lt_base_quarter']
        
        # Replace region codes with descriptive names
        prices_df.index = prices_df.index.map(lambda x: region_map.get(x, x))
        counts_df.index = counts_df.index.map(lambda x: region_map.get(x, x))
        area_df.index = area_df.index.map(lambda x: region_map.get(x, x))
        index_df.index = index_df.index.map(lambda x: region_map.get(x, x))
        
        # Create tabs
        tab_names = [
            "Summary",
            "Prices",
            "Transaction Counts",
            "Total Area",
            "Price Index",
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
            
            # Calculate summary statistics
            summary_data = []
            for region in prices_df.index:
                region_prices = prices_df.loc[region].dropna()
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
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                total_transactions = int(counts_df.sum().sum())
                st.metric("Total Transactions", f"{total_transactions:,}")
            
            with col2:
                total_area = area_df.sum().sum()
                st.metric("Total Area Acquired", f"{total_area:,.0f} m²")
            
            with col3:
                avg_price_latest = prices_df.iloc[:, -1].mean()
                st.metric("Average Price/m² (Latest)", f"€{avg_price_latest:.2f}")
        
        # Prices tab
        with tabs[1]:
            st.subheader(f"💰 Price per m² - {metric_label}")
            
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
                    key="lt_prices_plot_regions"
                )
            
            with col2:
                available_quarters = list(prices_df.columns)
                if len(available_quarters) > 1:
                    date_range_plot = st.select_slider(
                        "Date range",
                        options=available_quarters,
                        value=(available_quarters[0], available_quarters[-1]),
                        key="lt_prices_date_range"
                    )
                else:
                    date_range_plot = None
            
            fig = plot_regions_lt(
                prices_df,
                f"Price per m² Over Time - {st.session_state['lt_property_type']}",
                "Price per m² (EUR)",
                selected_regions=plot_regions_selected,
                date_range=date_range_plot if len(available_quarters) > 1 else None
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Counts tab
        with tabs[2]:
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
        with tabs[3]:
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
        
        # Index tab
        with tabs[4]:
            st.subheader(f"📈 Price Index (Base: {base_yr}-Q{base_qt} = 1.0)")
            
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
                    key="lt_index_plot_regions"
                )
            
            with col2:
                available_quarters = list(index_df.columns)
                if len(available_quarters) > 1:
                    date_range_plot = st.select_slider(
                        "Date range",
                        options=available_quarters,
                        value=(available_quarters[0], available_quarters[-1]),
                        key="lt_index_date_range"
                    )
                else:
                    date_range_plot = None
            
            fig = plot_regions_lt(
                index_df,
                f"Price Index Over Time - {st.session_state['lt_property_type']}",
                f"Index ({base_yr}-Q{base_qt} = 1.0)",
                selected_regions=plot_regions_selected,
                date_range=date_range_plot if len(available_quarters) > 1 else None
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Regional Comparison tab
        with tabs[5]:
            st.subheader("🔀 Regional Comparison")
            
            st.markdown("#### Compare Latest vs Initial Prices")
            
            # Calculate price changes
            comparison_data = []
            for region in prices_df.index:
                region_prices = prices_df.loc[region].dropna()
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
                    prices_df,
                    counts_df,
                    index_df,
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


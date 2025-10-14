# This is the main entry point for Streamlit Community Cloud
# Import all the necessary components from our main app

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import os
from datetime import datetime
import io

# Optional Supabase imports
try:
    from supabase import create_client, Client
    SUPABASE_AVAILABLE = True
except ImportError:
    SUPABASE_AVAILABLE = False

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Page config
st.set_page_config(
    page_title="Dynamic Commercial Real Estate Dashboard", 
    layout="wide", 
    page_icon=":office:",
    initial_sidebar_state="expanded"
)

# Import all functions from the main app
from app import (
    load_data, init_supabase, clean_and_validate_data, validate_data_ranges,
    map_to_supabase_schema, upload_to_supabase, clear_supabase_table
)

# Load default data
df = load_data()

# Title and overview
st.title("🏢 Dynamic Commercial Real Estate Dashboard")
st.markdown("Upload a CSV file to generate dynamic charts and analyze real estate data")

# Show different messages based on whether we have data
if len(df) == 0 or df.empty:
    st.info("📁 **Upload a CSV file** using the sidebar to get started with your real estate data analysis!")
    st.markdown("**No data loaded yet.** Upload a CSV file to see charts and analysis.")
    
    # Show example of what the app can do
    st.markdown("### 🎯 What this app can do:")
    st.markdown("""
    - **📊 Dynamic Charts**: Automatically generate visualizations based on your data
    - **🗺️ Interactive Maps**: Plot property locations on a map
    - **🔍 Smart Filtering**: Filter by property type, city, status, and more
    - **☁️ Supabase Integration**: Automatically upload filtered data to your database
    - **💾 Export Options**: Download filtered or raw data as CSV
    """)
else:
    st.success("🚀 **Live Demo**: This app updates automatically when code changes!")

# Sidebar for file upload
st.sidebar.header("📁 Data Upload")
uploaded_file = st.sidebar.file_uploader(
    "Choose a CSV file",
    type="csv",
    help="Upload a CSV file with commercial real estate data"
)

# Use uploaded file if available, otherwise use default
if uploaded_file is not None:
    try:
        # Read uploaded CSV file
        df = pd.read_csv(uploaded_file, skiprows=2, encoding='latin-1')
        df = clean_and_validate_data(df)
        st.success(f"✅ Successfully loaded {len(df)} properties from uploaded CSV")
        
        # Data validation
        validation_results = validate_data_ranges(df)
        if validation_results:
            st.warning("Data Validation Results:")
            for result in validation_results:
                st.write(result)
        else:
            st.success("✅ Data validation passed - no issues found")
        
        # Auto-upload to Supabase if configured
        supabase_client = init_supabase()
        if supabase_client:
            with st.spinner("🔄 Automatically uploading to Supabase..."):
                supabase_data = map_to_supabase_schema(df)
                success, message = upload_to_supabase(supabase_client, supabase_data, auto_upload=True, clear_existing=False)
                if success:
                    st.success(f"🚀 {message}")
                else:
                    st.error(f"❌ {message}")
        else:
            st.info("ℹ️ Supabase not configured - data will be available for manual upload below")
            
    except Exception as e:
        st.error(f"❌ Error processing uploaded file: {str(e)}")
        st.info("Using default dataset instead")
        df = load_data()

st.markdown(f"**{len(df)} properties** currently loaded")

# Sidebar filters (only show if we have data)
if len(df) > 0 and not df.empty:
    st.sidebar.header("Filters")

    property_types = st.sidebar.multiselect(
        "Property Type",
        options=sorted(df['Type'].dropna().unique()) if 'Type' in df.columns else [],
        default=[]
    )

    cities = st.sidebar.multiselect(
        "City",
        options=sorted(df['City'].dropna().unique()) if 'City' in df.columns else [],
        default=[]
    )

    status = st.sidebar.multiselect(
        "Property Status",
        options=sorted(df['Property Status'].dropna().unique()) if 'Property Status' in df.columns else [],
        default=[]
    )
else:
    # Set empty filters when no data
    property_types = []
    cities = []
    status = []

# Apply filters
filtered_df = df.copy()
if property_types and 'Type' in df.columns:
    filtered_df = filtered_df[filtered_df['Type'].isin(property_types)]
if cities and 'City' in df.columns:
    filtered_df = filtered_df[filtered_df['City'].isin(cities)]
if status and 'Property Status' in df.columns:
    filtered_df = filtered_df[filtered_df['Property Status'].isin(status)]

# Only show dashboard if we have data
if len(filtered_df) > 0 and not filtered_df.empty:
    # KPIs
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric("Total Properties", len(filtered_df))

    with col2:
        avg_price = filtered_df['Asking Price'].median() if 'Asking Price' in filtered_df.columns else None
        st.metric("Median Asking Price", f"${avg_price/1e6:.2f}M" if pd.notna(avg_price) else "N/A")

    with col3:
        avg_sqft = filtered_df['SqFt'].median() if 'SqFt' in filtered_df.columns else None
        st.metric("Median SqFt", f"{avg_sqft:,.0f}" if pd.notna(avg_sqft) else "N/A")

    with col4:
        avg_cap = filtered_df['Cap Rate'].median() if 'Cap Rate' in filtered_df.columns else None
        st.metric("Median Cap Rate", f"{avg_cap:.2f}%" if pd.notna(avg_cap) else "N/A")

    with col5:
        avg_dom = filtered_df['Days on Market'].median() if 'Days on Market' in filtered_df.columns else None
        st.metric("Median Days on Market", f"{avg_dom:.0f}" if pd.notna(avg_dom) else "N/A")

    st.divider()

    # Property Type Distribution
    st.subheader("Property Type Distribution")
    type_counts = filtered_df['Type'].value_counts().head(10)
    fig = px.bar(
        x=type_counts.values,
        y=type_counts.index,
        orientation='h',
        labels={'x': 'Count', 'y': 'Property Type'},
        color=type_counts.values,
        color_continuous_scale='Blues'
    )
    fig.update_layout(showlegend=False, height=400)
    st.plotly_chart(fig, use_container_width=True)

    # Asking Price Distribution
    st.subheader("Asking Price Distribution")
    price_data = filtered_df[filtered_df['Asking Price'].notna() & (filtered_df['Asking Price'] > 0)]
    fig = px.histogram(
        price_data,
        x='Asking Price',
        nbins=50,
        labels={'Asking Price': 'Asking Price ($)'},
        color_discrete_sequence=['#1f77b4']
    )
    fig.update_layout(showlegend=False, height=400)
    fig.update_xaxes(tickformat='$,.0f')
    st.plotly_chart(fig, use_container_width=True)

    # Geographic Map
    st.subheader("Property Locations Map")
    map_data = filtered_df[filtered_df['Latitude'].notna() & filtered_df['Longitude'].notna()].copy()

    if len(map_data) > 0:
        map_data['size'] = map_data['Asking Price'].fillna(map_data['Asking Price'].median())
        map_data['size'] = np.log1p(map_data['size']) * 2  # Scale for better visualization

        fig = px.scatter_mapbox(
            map_data,
            lat='Latitude',
            lon='Longitude',
            hover_name='Property Name',
            hover_data={
                'Type': True,
                'City': True,
                'Asking Price': ':$,.0f',
                'SqFt': ':,.0f',
                'Latitude': False,
                'Longitude': False,
                'size': False
            },
            color='Type',
            size='size',
            zoom=7.5,
            height=500
        )
        fig.update_layout(mapbox_style="open-street-map")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No geographic data available for selected filters")

    # Data table
    st.divider()
    st.subheader("Property Data Table")

    display_cols = [
        'Property Name', 'Type', 'City', 'Property Status',
        'Asking Price', 'SqFt', 'Price/SqFt', 'Cap Rate',
        'Days on Market', 'Opportunity Zone'
    ]

    display_df = filtered_df[display_cols].copy()
    st.dataframe(
        display_df.style.format({
            'Asking Price': '${:,.0f}',
            'SqFt': '{:,.0f}',
            'Price/SqFt': '${:.2f}',
            'Cap Rate': '{:.2f}%'
        }, na_rep='N/A'),
        use_container_width=True,
        height=400
    )
else:
    # Show message when no data is available
    st.info("📊 **No data available yet.** Upload a CSV file using the sidebar to see charts and analysis!")

# Supabase Integration (always show)
st.divider()
st.header("☁️ Supabase Integration")

# Initialize Supabase client
supabase_client = init_supabase()

if not SUPABASE_AVAILABLE:
    st.info("ℹ️ Supabase integration is not available due to import issues. All other features work normally.")
elif supabase_client:
    if len(filtered_df) > 0 and not filtered_df.empty:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.write("**Current Filtered Data Preview for Supabase:**")
            supabase_data = map_to_supabase_schema(filtered_df)
            preview_df = pd.DataFrame(supabase_data)
            st.dataframe(preview_df.head(), use_container_width=True)
        
        with col2:
            # Option to clear existing data
            clear_existing = st.checkbox("🗑️ Clear existing data first", 
                                       help="This will delete all existing records from the Supabase deals table before uploading new data")
            
            if st.button("📤 Upload to Supabase", type="primary"):
                with st.spinner("Uploading to Supabase..."):
                    success, message = upload_to_supabase(supabase_client, supabase_data, auto_upload=False, clear_existing=clear_existing)
                    if success:
                        st.success(f"✅ {message}")
                    else:
                        st.error(f"❌ {message}")
            
            # Download Supabase-ready CSV
            csv_data = preview_df.to_csv(index=False)
            st.download_button(
                label="💾 Download Supabase CSV",
                data=csv_data,
                file_name=f'supabase_deals_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv',
                mime='text/csv'
            )
    else:
        st.info("📁 Upload a CSV file first to see Supabase integration options.")
else:
    st.warning("⚠️ Supabase credentials not configured. Please:")
    st.markdown("""
    1. Copy `config_example.py` to `config.py`
    2. Fill in your Supabase URL and anon key
    3. Or set environment variables SUPABASE_URL and SUPABASE_ANON_KEY
    """)

# Download buttons
st.divider()
st.header("💾 Download Data")

if len(filtered_df) > 0 and not filtered_df.empty:
    col1, col2 = st.columns(2)

    with col1:
        st.download_button(
            label="Download Filtered Data as CSV",
            data=filtered_df.to_csv(index=False).encode('utf-8'),
            file_name=f'filtered_properties_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv',
            mime='text/csv'
        )

    with col2:
        st.download_button(
            label="Download All Data as CSV",
            data=df.to_csv(index=False).encode('utf-8'),
            file_name=f'all_properties_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv',
            mime='text/csv'
        )
else:
    st.info("📁 Upload a CSV file first to enable download options.")

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

# Load data
@st.cache_data
def load_data():
    """Load default data if available, otherwise return empty DataFrame"""
    try:
        # Check if file exists first
        if not os.path.exists('Export Results.csv'):
            return pd.DataFrame(columns=[
                'Property Name', 'Type', 'City', 'Address', 'Property Status',
                'Asking Price', 'SqFt', 'Price/SqFt', 'Cap Rate', 'Units',
                'Days on Market', 'Latitude', 'Longitude', 'Opportunity Zone'
            ])
        
        df = pd.read_csv('Export Results.csv', skiprows=2, encoding='latin-1')
        
        # Clean price columns
        price_cols = ['Asking Price', 'Price/SqFt', 'Price/Acre', 'NOI', 'Price/Unit']
        for col in price_cols:
            if col in df.columns:
                df[col] = df[col].astype(str).str.replace('$', '').str.replace(',', '').str.replace(' ', '')
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # Clean percentage columns
        if 'Cap Rate' in df.columns:
            df['Cap Rate'] = df['Cap Rate'].astype(str).str.replace('%', '')
            df['Cap Rate'] = pd.to_numeric(df['Cap Rate'], errors='coerce')

        # Clean numeric columns
        numeric_cols = ['SqFt', 'Lot Size', 'Units', 'Days on Market', 'Remaining Term']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        return df
    except Exception as e:
        # Return empty DataFrame with expected columns if any error occurs
        return pd.DataFrame(columns=[
            'Property Name', 'Type', 'City', 'Address', 'Property Status',
            'Asking Price', 'SqFt', 'Price/SqFt', 'Cap Rate', 'Units',
            'Days on Market', 'Latitude', 'Longitude', 'Opportunity Zone'
        ])

# Initialize Supabase client
@st.cache_resource
def init_supabase():
    """Initialize Supabase client"""
    if not SUPABASE_AVAILABLE:
        return None
        
    # Try environment variables first
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_ANON_KEY")
    
    # If not found, try config file
    if not url or not key:
        try:
            import config
            url = config.SUPABASE_URL
            key = config.SUPABASE_ANON_KEY
        except ImportError:
            pass
    
    if not url or not key or url == "your_supabase_project_url_here":
        return None
    
    try:
        supabase: Client = create_client(url, key)
        return supabase
    except Exception as e:
        st.error(f"❌ Failed to connect to Supabase: {str(e)}")
        return None

def clean_and_validate_data(df):
    """Clean and validate the uploaded CSV data"""
    # Remove empty rows and rows with all NaN values
    df = df.dropna(how='all')
    
    # Clean price columns
    price_cols = ['Asking Price', 'Price/SqFt', 'Price/Acre', 'NOI', 'Price/Unit']
    for col in price_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace('$', '').str.replace(',', '').str.replace(' ', '')
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Clean percentage columns
    if 'Cap Rate' in df.columns:
        df['Cap Rate'] = df['Cap Rate'].astype(str).str.replace('%', '')
        df['Cap Rate'] = pd.to_numeric(df['Cap Rate'], errors='coerce')
    
    # Clean numeric columns
    numeric_cols = ['SqFt', 'Lot Size', 'Units', 'Days on Market', 'Remaining Term']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Clean coordinate columns
    if 'Latitude' in df.columns:
        df['Latitude'] = pd.to_numeric(df['Latitude'], errors='coerce')
    if 'Longitude' in df.columns:
        df['Longitude'] = pd.to_numeric(df['Longitude'], errors='coerce')
    
    # Clean occupancy columns for Supabase mapping
    occupancy_cols = [col for col in df.columns if 'occupancy' in col.lower()]
    for col in occupancy_cols:
        df[col] = df[col].astype(str).str.replace('%', '')
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    return df

def validate_data_ranges(df):
    """Validate data is within appropriate ranges"""
    validation_results = []
    
    # Price validation
    if 'Asking Price' in df.columns:
        price_data = df['Asking Price'].dropna()
        if len(price_data) > 0:
            min_price, max_price = price_data.min(), price_data.max()
            if min_price < 0:
                validation_results.append("⚠️ Negative asking prices found")
            if max_price > 1e12:  # $1 trillion
                validation_results.append("⚠️ Extremely high asking prices found (>$1T)")
    
    # Square footage validation
    if 'SqFt' in df.columns:
        sqft_data = df['SqFt'].dropna()
        if len(sqft_data) > 0:
            min_sqft, max_sqft = sqft_data.min(), sqft_data.max()
            if min_sqft < 0:
                validation_results.append("⚠️ Negative square footage found")
            if max_sqft > 10e6:  # 10M sqft
                validation_results.append("⚠️ Extremely large properties found (>10M sqft)")
    
    # Cap rate validation
    if 'Cap Rate' in df.columns:
        cap_data = df['Cap Rate'].dropna()
        if len(cap_data) > 0:
            min_cap, max_cap = cap_data.min(), cap_data.max()
            if min_cap < 0:
                validation_results.append("⚠️ Negative cap rates found")
            if max_cap > 50:  # 50%
                validation_results.append("⚠️ Extremely high cap rates found (>50%)")
    
    # Coordinate validation
    if 'Latitude' in df.columns and 'Longitude' in df.columns:
        lat_data = df['Latitude'].dropna()
        lon_data = df['Longitude'].dropna()
        if len(lat_data) > 0 and len(lon_data) > 0:
            if lat_data.min() < -90 or lat_data.max() > 90:
                validation_results.append("⚠️ Invalid latitude values found")
            if lon_data.min() < -180 or lon_data.max() > 180:
                validation_results.append("⚠️ Invalid longitude values found")
    
    return validation_results

def map_to_supabase_schema(df):
    """Map CSV data to Supabase deals table schema
    
    This function filters and maps only the fields required by the Supabase deals table:
    - asset_name (from Property Name)
    - full_address (from Address) 
    - total_units (from Units)
    - net_rentable_area_sqft (from SqFt)
    - current_occupancy_pct (from any occupancy column)
    - source_document (set to 'CSV Upload')
    - created_at/updated_at (timestamps)
    """
    supabase_data = []
    
    # Show what fields we're mapping
    st.info("🔄 **Filtering CSV data for Supabase deals table schema:**")
    
    for _, row in df.iterrows():
        # Map to exact Supabase schema fields only
        record = {
            'asset_name': str(row.get('Property Name', '')),
            'full_address': str(row.get('Address', '')),
            'total_units': int(row.get('Units', 0)) if pd.notna(row.get('Units')) and row.get('Units') > 0 else None,
            'net_rentable_area_sqft': int(row.get('SqFt', 0)) if pd.notna(row.get('SqFt')) and row.get('SqFt') > 0 else None,
            'source_document': 'CSV Upload',
            'created_at': datetime.now().isoformat(),
            'updated_at': datetime.now().isoformat()
        }
        
        # Handle occupancy field (look for any column with 'occupancy' in the name)
        # Note: Supabase field is 'current_occupancy_percent', not 'current_occupancy_pct'
        occupancy_cols = [col for col in df.columns if 'occupancy' in col.lower()]
        if occupancy_cols:
            occupancy_value = row.get(occupancy_cols[0])
            if pd.notna(occupancy_value):
                # Convert to decimal format (e.g., 85.5 for 85.5%)
                record['current_occupancy_percent'] = float(occupancy_value)
        
        supabase_data.append(record)
    
    # Show mapping summary
    original_cols = list(df.columns)
    supabase_cols = list(record.keys()) if supabase_data else []
    
    st.markdown(f"""
    **📊 Data Mapping Summary:**
    - **Original CSV columns:** {len(original_cols)} fields
    - **Supabase schema fields:** {len(supabase_cols)} fields  
    - **Records processed:** {len(supabase_data)}
    
    **🎯 Mapped Fields:**
    - `Property Name` → `asset_name`
    - `Address` → `full_address` 
    - `Units` → `total_units`
    - `SqFt` → `net_rentable_area_sqft`
    - `[occupancy column]` → `current_occupancy_percent`
    - `source_document` = 'CSV Upload'
    - `created_at`/`updated_at` = current timestamp
    """)
    
    return supabase_data

def clear_supabase_table(supabase_client):
    """Clear all existing data from the Supabase deals table"""
    if not supabase_client:
        return False, "No Supabase client available"
    
    try:
        # Delete all records from the deals table
        result = supabase_client.table('deals').delete().neq('id', 0).execute()
        return True, f"Successfully cleared all existing records from Supabase deals table"
    except Exception as e:
        return False, f"Failed to clear Supabase table: {str(e)}"

def upload_to_supabase(supabase_client, data, auto_upload=True, clear_existing=False):
    """Upload data to Supabase with progress tracking"""
    if not supabase_client or not data:
        return False, "No data or Supabase client available"
    
    try:
        # Clear existing data if requested
        if clear_existing:
            clear_success, clear_message = clear_supabase_table(supabase_client)
            if not clear_success:
                return False, f"Failed to clear existing data: {clear_message}"
            st.success(f"✅ {clear_message}")
        
        # Insert data in batches for better performance
        batch_size = 100
        total_uploaded = 0
        
        for i in range(0, len(data), batch_size):
            batch = data[i:i + batch_size]
            result = supabase_client.table('deals').insert(batch).execute()
            total_uploaded += len(batch)
            
            if not auto_upload:
                st.progress(min(i + batch_size, len(data)) / len(data))
        
        return True, f"Successfully uploaded {total_uploaded} records to Supabase"
    except Exception as e:
        return False, f"Failed to upload to Supabase: {str(e)}"

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
                success, message = upload_to_supabase(supabase_client, supabase_data, auto_upload=True)
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

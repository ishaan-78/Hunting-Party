import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from supabase import create_client, Client
import os
from datetime import datetime
from dotenv import load_dotenv
import io

# Load environment variables
load_dotenv()

# Page config
st.set_page_config(
    page_title="Dynamic Commercial Real Estate Dashboard", 
    layout="wide", 
    page_icon=":office:",
    initial_sidebar_state="expanded"
)

# Initialize Supabase client
@st.cache_resource
def init_supabase():
    """Initialize Supabase client"""
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
        st.warning("⚠️ Supabase credentials not configured. Please:")
        st.markdown("""
        1. Copy `config_example.py` to `config.py`
        2. Fill in your Supabase URL and anon key
        3. Or set environment variables SUPABASE_URL and SUPABASE_ANON_KEY
        """)
        return None
    
    try:
        supabase: Client = create_client(url, key)
        return supabase
    except Exception as e:
        st.error(f"❌ Failed to connect to Supabase: {str(e)}")
        return None

# Data mapping configuration
CSV_TO_SUPABASE_MAPPING = {
    'Property Name': 'asset_name',
    'Address': 'full_address', 
    'Units': 'total_units',
    'SqFt': 'net_rentable_area_sqft',
    # Note: We'll need to map other fields as they become available in the schema
}

def clean_and_validate_data(df):
    """Clean and validate the uploaded CSV data"""
    st.info("🧹 Cleaning and validating data...")
    
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

def upload_to_supabase(df, supabase_client):
    """Upload cleaned data to Supabase"""
    if not supabase_client:
        return False, "Supabase client not available"
    
    try:
        # Map CSV columns to Supabase table columns
        supabase_data = []
        for _, row in df.iterrows():
            record = {
                'asset_name': row.get('Property Name', ''),
                'full_address': row.get('Address', ''),
                'total_units': row.get('Units') if pd.notna(row.get('Units')) else None,
                'net_rentable_area_sqft': row.get('SqFt') if pd.notna(row.get('SqFt')) else None,
                'created_at': datetime.now().isoformat(),
                'updated_at': datetime.now().isoformat()
            }
            supabase_data.append(record)
        
        # Insert data into Supabase
        result = supabase_client.table('deals').insert(supabase_data).execute()
        
        return True, f"Successfully uploaded {len(supabase_data)} records to Supabase"
    
    except Exception as e:
        return False, f"Failed to upload to Supabase: {str(e)}"

def generate_dynamic_charts(df):
    """Generate charts dynamically based on available data"""
    charts = []
    
    # 1. Property Type Distribution
    if 'Type' in df.columns:
        type_counts = df['Type'].value_counts().head(10)
        if len(type_counts) > 0:
            fig = px.bar(
                x=type_counts.values,
                y=type_counts.index,
                orientation='h',
                labels={'x': 'Count', 'y': 'Property Type'},
                color=type_counts.values,
                color_continuous_scale='Blues',
                title="Property Type Distribution"
            )
            fig.update_layout(showlegend=False, height=400)
            charts.append(("Property Type Distribution", fig))
    
    # 2. Price Distribution
    if 'Asking Price' in df.columns:
        price_data = df[df['Asking Price'].notna() & (df['Asking Price'] > 0)]
        if len(price_data) > 0:
            fig = px.histogram(
                price_data,
                x='Asking Price',
                nbins=50,
                labels={'Asking Price': 'Asking Price ($)'},
                color_discrete_sequence=['#1f77b4'],
                title="Asking Price Distribution"
            )
            fig.update_layout(showlegend=False, height=400)
            fig.update_xaxes(tickformat='$,.0f')
            charts.append(("Asking Price Distribution", fig))
    
    # 3. Geographic Map
    if 'Latitude' in df.columns and 'Longitude' in df.columns:
        map_data = df[df['Latitude'].notna() & df['Longitude'].notna()].copy()
        if len(map_data) > 0:
            map_data['size'] = map_data['Asking Price'].fillna(map_data['Asking Price'].median()) if 'Asking Price' in map_data.columns else 1
            map_data['size'] = np.log1p(map_data['size']) * 2

            fig = px.scatter_mapbox(
                map_data,
                lat='Latitude',
                lon='Longitude',
                hover_name='Property Name' if 'Property Name' in map_data.columns else 'Address',
                hover_data={
                    'Type': True if 'Type' in map_data.columns else False,
                    'City': True if 'City' in map_data.columns else False,
                    'Asking Price': ':$,.0f' if 'Asking Price' in map_data.columns else False,
                    'SqFt': ':,.0f' if 'SqFt' in map_data.columns else False,
                    'Latitude': False,
                    'Longitude': False,
                    'size': False
                },
                color='Type' if 'Type' in map_data.columns else None,
                size='size',
                zoom=7.5,
                height=500,
                title="Property Locations Map"
            )
            fig.update_layout(mapbox_style="open-street-map")
            charts.append(("Property Locations Map", fig))
    
    # 4. Price per SqFt by City
    if 'City' in df.columns and 'Price/SqFt' in df.columns:
        city_price = df.groupby('City')['Price/SqFt'].median().sort_values(ascending=False).head(15)
        if len(city_price) > 0:
            fig = px.bar(
                x=city_price.values,
                y=city_price.index,
                orientation='h',
                labels={'x': 'Price per SqFt ($)', 'y': 'City'},
                color=city_price.values,
                color_continuous_scale='Viridis',
                title="Median Price/SqFt by City (Top 15)"
            )
            fig.update_layout(showlegend=False, height=500)
            charts.append(("Price/SqFt by City", fig))
    
    # 5. Cap Rate Distribution
    if 'Cap Rate' in df.columns:
        cap_data = df[df['Cap Rate'].notna() & (df['Cap Rate'] > 0)]
        if len(cap_data) > 0:
            fig = px.histogram(
                cap_data,
                x='Cap Rate',
                nbins=30,
                labels={'Cap Rate': 'Cap Rate (%)'},
                color_discrete_sequence=['#2ca02c'],
                title="Cap Rate Distribution"
            )
            fig.update_layout(showlegend=False, height=400)
            charts.append(("Cap Rate Distribution", fig))
    
    return charts

def main():
    st.title("🏢 Dynamic Commercial Real Estate Dashboard")
    st.markdown("Upload a CSV file to generate dynamic charts and store data in Supabase")
    
    # Initialize Supabase
    supabase_client = init_supabase()
    
    # Sidebar for file upload
    st.sidebar.header("📁 Data Upload")
    
    uploaded_file = st.sidebar.file_uploader(
        "Choose a CSV file",
        type="csv",
        help="Upload a CSV file with commercial real estate data"
    )
    
    if uploaded_file is not None:
        try:
            # Read CSV file
            df = pd.read_csv(uploaded_file, skiprows=2, encoding='latin-1')
            
            # Clean and validate data
            df = clean_and_validate_data(df)
            
            st.success(f"✅ Successfully loaded {len(df)} properties from CSV")
            
            # Display basic info
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Properties", len(df))
            with col2:
                st.metric("Columns Available", len(df.columns))
            with col3:
                st.metric("Memory Usage", f"{df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
            
            # Data validation
            validation_results = validate_data_ranges(df)
            if validation_results:
                st.warning("Data Validation Results:")
                for result in validation_results:
                    st.write(result)
            else:
                st.success("✅ Data validation passed - no issues found")
            
            # Display column information
            with st.expander("📊 Available Data Columns"):
                col_info = []
                for col in df.columns:
                    non_null_count = df[col].count()
                    null_count = len(df) - non_null_count
                    data_type = str(df[col].dtype)
                    col_info.append({
                        'Column': col,
                        'Data Type': data_type,
                        'Non-Null Values': non_null_count,
                        'Null Values': null_count,
                        'Sample Value': str(df[col].dropna().iloc[0]) if non_null_count > 0 else 'N/A'
                    })
                
                col_df = pd.DataFrame(col_info)
                st.dataframe(col_df, use_container_width=True)
            
            # Generate and display charts
            st.header("📈 Dynamic Charts")
            charts = generate_dynamic_charts(df)
            
            if charts:
                for title, fig in charts:
                    st.subheader(title)
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No charts could be generated with the available data")
            
            # Data preview
            with st.expander("👀 Data Preview"):
                st.dataframe(df.head(10), use_container_width=True)
            
            # Supabase upload section
            st.header("☁️ Supabase Integration")
            
            if supabase_client:
                if st.button("Upload to Supabase", type="primary"):
                    with st.spinner("Uploading to Supabase..."):
                        success, message = upload_to_supabase(df, supabase_client)
                        if success:
                            st.success(f"✅ {message}")
                        else:
                            st.error(f"❌ {message}")
            else:
                st.warning("⚠️ Supabase connection not available. Please check your credentials.")
            
            # Download processed data
            st.header("💾 Download Processed Data")
            csv_buffer = io.StringIO()
            df.to_csv(csv_buffer, index=False)
            
            st.download_button(
                label="Download Processed CSV",
                data=csv_buffer.getvalue().encode('utf-8'),
                file_name=f"processed_real_estate_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime='text/csv'
            )
            
        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")
    
    else:
        st.info("👆 Please upload a CSV file to get started")
        
        # Show sample of existing data if available
        if os.path.exists('Export Results.csv'):
            st.subheader("📋 Sample Data Available")
            try:
                sample_df = pd.read_csv('Export Results.csv', skiprows=2, encoding='latin-1', nrows=5)
                st.dataframe(sample_df, use_container_width=True)
                st.caption("This is a sample of the existing Export Results.csv file")
            except Exception as e:
                st.error(f"Error loading sample data: {str(e)}")

if __name__ == "__main__":
    main()

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

# Page config
st.set_page_config(page_title="NJ Commercial Real Estate Dashboard", layout="wide", page_icon=":office:")

# Load data
@st.cache_data
def load_data():
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

df = load_data()

# Title and overview
st.title("New Jersey Commercial Real Estate Dashboard")
st.markdown(f"**{len(df)} properties** analyzed from Crexi")

# Sidebar filters
st.sidebar.header("Filters")

property_types = st.sidebar.multiselect(
    "Property Type",
    options=sorted(df['Type'].dropna().unique()),
    default=[]
)

cities = st.sidebar.multiselect(
    "City",
    options=sorted(df['City'].dropna().unique()),
    default=[]
)

status = st.sidebar.multiselect(
    "Property Status",
    options=sorted(df['Property Status'].dropna().unique()),
    default=[]
)

# Apply filters
filtered_df = df.copy()
if property_types:
    filtered_df = filtered_df[filtered_df['Type'].isin(property_types)]
if cities:
    filtered_df = filtered_df[filtered_df['City'].isin(cities)]
if status:
    filtered_df = filtered_df[filtered_df['Property Status'].isin(status)]

# KPIs
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric("Total Properties", len(filtered_df))

with col2:
    avg_price = filtered_df['Asking Price'].median()
    st.metric("Median Asking Price", f"${avg_price/1e6:.2f}M" if pd.notna(avg_price) else "N/A")

with col3:
    avg_sqft = filtered_df['SqFt'].median()
    st.metric("Median SqFt", f"{avg_sqft:,.0f}" if pd.notna(avg_sqft) else "N/A")

with col4:
    avg_cap = filtered_df['Cap Rate'].median()
    st.metric("Median Cap Rate", f"{avg_cap:.2f}%" if pd.notna(avg_cap) else "N/A")

with col5:
    avg_dom = filtered_df['Days on Market'].median()
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

# Median Price/SqFt by City
st.subheader("Median Price/SqFt by City (Top 15)")
city_price = filtered_df.groupby('City')['Price/SqFt'].median().sort_values(ascending=False).head(15)
fig = px.bar(
    x=city_price.values,
    y=city_price.index,
    orientation='h',
    labels={'x': 'Price per SqFt ($)', 'y': 'City'},
    color=city_price.values,
    color_continuous_scale='Viridis'
)
fig.update_layout(showlegend=False, height=500)
st.plotly_chart(fig, use_container_width=True)

# Cap Rate Distribution by Property Type
st.subheader("Cap Rate Distribution by Property Type")
cap_data = filtered_df[filtered_df['Cap Rate'].notna() & (filtered_df['Cap Rate'] > 0)]
top_types = cap_data['Type'].value_counts().head(8).index
cap_data_filtered = cap_data[cap_data['Type'].isin(top_types)]

fig = px.box(
    cap_data_filtered,
    x='Type',
    y='Cap Rate',
    color='Type',
    labels={'Cap Rate': 'Cap Rate (%)'}
)
fig.update_layout(showlegend=False, height=500, xaxis_tickangle=-45)
st.plotly_chart(fig, use_container_width=True)

# Days on Market by Property Status
st.subheader("Days on Market by Property Status")
dom_data = filtered_df[filtered_df['Days on Market'].notna()]
fig = px.violin(
    dom_data,
    x='Property Status',
    y='Days on Market',
    color='Property Status',
    box=True
)
fig.update_layout(showlegend=False, height=400)
st.plotly_chart(fig, use_container_width=True)

# Property Size Distribution
st.subheader("Property Size Distribution")
sqft_data = filtered_df[filtered_df['SqFt'].notna() & (filtered_df['SqFt'] > 0)]
fig = px.histogram(
    sqft_data,
    x='SqFt',
    nbins=50,
    labels={'SqFt': 'Square Feet'},
    color_discrete_sequence=['#2ca02c']
)
fig.update_layout(showlegend=False, height=400)
fig.update_xaxes(tickformat=',.0f')
st.plotly_chart(fig, use_container_width=True)

# Asking Price vs Square Footage
st.subheader("Asking Price vs Square Footage")
scatter_data = filtered_df[
    filtered_df['Asking Price'].notna() &
    filtered_df['SqFt'].notna() &
    (filtered_df['Asking Price'] > 0) &
    (filtered_df['SqFt'] > 0)
].copy()

if len(scatter_data) > 0:
    fig = px.scatter(
        scatter_data,
        x='SqFt',
        y='Asking Price',
        color='Type',
        hover_data=['Property Name', 'City'],
        trendline='ols',
        labels={'SqFt': 'Square Feet', 'Asking Price': 'Asking Price ($)'}
    )
    fig.update_yaxes(tickformat='$,.0f')
    fig.update_xaxes(tickformat=',.0f')
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("No data available for this analysis")

# Opportunity Zone Properties
st.subheader("Opportunity Zone Properties")
oz_counts = filtered_df['Opportunity Zone'].value_counts()
fig = px.pie(
    values=oz_counts.values,
    names=oz_counts.index,
    color_discrete_sequence=['#ff7f0e', '#1f77b4']
)
fig.update_traces(textposition='inside', textinfo='percent+label')
st.plotly_chart(fig, use_container_width=True)

# Top 15 Cities by Property Count
st.subheader("Top 15 Cities by Property Count")
city_counts = filtered_df['City'].value_counts().head(15)
fig = px.bar(
    x=city_counts.values,
    y=city_counts.index,
    orientation='h',
    labels={'x': 'Number of Properties', 'y': 'City'},
    color=city_counts.values,
    color_continuous_scale='Reds'
)
fig.update_layout(showlegend=False, height=500)
st.plotly_chart(fig, use_container_width=True)

# Property Status Overview
st.subheader("Property Status Overview")
status_counts = filtered_df['Property Status'].value_counts()
fig = px.bar(
    x=status_counts.index,
    y=status_counts.values,
    labels={'x': 'Status', 'y': 'Count'},
    color=status_counts.values,
    color_continuous_scale='Greens'
)
fig.update_layout(showlegend=False, height=500)
st.plotly_chart(fig, use_container_width=True)

# NOI Analysis
st.divider()
st.subheader("Net Operating Income (NOI) Analysis")

noi_data = filtered_df[filtered_df['NOI'].notna() & (filtered_df['NOI'] > 0)]

if len(noi_data) > 0:
    # NOI Distribution
    st.subheader("NOI Distribution")
    fig = px.histogram(
        noi_data,
        x='NOI',
        nbins=30,
        labels={'NOI': 'Net Operating Income ($)'},
        color_discrete_sequence=['#9467bd']
    )
    fig.update_layout(showlegend=False, height=400)
    fig.update_xaxes(tickformat='$,.0f')
    st.plotly_chart(fig, use_container_width=True)

    # NOI by Property Type
    if len(noi_data['Type'].unique()) > 1:
        st.subheader("NOI by Property Type")
        fig = px.box(
            noi_data,
            x='Type',
            y='NOI',
            color='Type',
            labels={'NOI': 'Net Operating Income ($)'}
        )
        fig.update_layout(showlegend=False, height=500, xaxis_tickangle=-45)
        fig.update_yaxes(tickformat='$,.0f')
        st.plotly_chart(fig, use_container_width=True)

    # NOI vs Asking Price
    price_noi_data = noi_data[noi_data['Asking Price'].notna() & (noi_data['Asking Price'] > 0)]
    if len(price_noi_data) > 0:
        st.subheader("NOI vs Asking Price")
        fig = px.scatter(
            price_noi_data,
            x='NOI',
            y='Asking Price',
            color='Type',
            hover_data=['Property Name', 'City'],
            trendline='ols',
            labels={'NOI': 'Net Operating Income ($)', 'Asking Price': 'Asking Price ($)'}
        )
        fig.update_xaxes(tickformat='$,.0f')
        fig.update_yaxes(tickformat='$,.0f')
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)

    # NOI by City (Top 10)
    city_noi = noi_data.groupby('City')['NOI'].median().sort_values(ascending=False).head(10)
    if len(city_noi) > 1:
        st.subheader("Median NOI by City (Top 10)")
        fig = px.bar(
            x=city_noi.values,
            y=city_noi.index,
            orientation='h',
            labels={'x': 'Median NOI ($)', 'y': 'City'},
            color=city_noi.values,
            color_continuous_scale='Purples'
        )
        fig.update_layout(showlegend=False, height=400)
        fig.update_xaxes(tickformat='$,.0f')
        st.plotly_chart(fig, use_container_width=True)
else:
    st.info("No NOI data available for selected filters")

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

# Download button
st.download_button(
    label="Download Filtered Data as CSV",
    data=filtered_df.to_csv(index=False).encode('utf-8'),
    file_name='filtered_properties.csv',
    mime='text/csv'
)

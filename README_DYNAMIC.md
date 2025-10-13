# Dynamic Commercial Real Estate Dashboard

This is the new dynamic dashboard system that can accept any CSV file and generate charts automatically, while also storing data in Supabase.

## Features

✅ **Dynamic CSV Upload**: Upload any CSV file with real estate data  
✅ **Automatic Chart Generation**: Charts are generated based on available columns  
✅ **Data Validation**: Validates data ranges and quality  
✅ **Supabase Integration**: Stores cleaned data in your Supabase database  
✅ **Flexible Mapping**: Maps CSV columns to your database schema  

## Setup Instructions

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure Supabase Connection

**Option A: Using Config File (Recommended)**
1. Copy `config_example.py` to `config.py`
2. Fill in your Supabase credentials:
```python
SUPABASE_URL = "https://your-project-id.supabase.co"
SUPABASE_ANON_KEY = "your-anon-key-here"
```

**Option B: Using Environment Variables**
```bash
export SUPABASE_URL="https://your-project-id.supabase.co"
export SUPABASE_ANON_KEY="your-anon-key-here"
```

### 3. Run the Dynamic Dashboard
```bash
python -m streamlit run dynamic_dashboard.py --server.port 8502
```

## How to Use

1. **Upload CSV**: Use the file uploader in the sidebar to upload your CSV file
2. **Data Validation**: The system automatically validates your data and shows any issues
3. **View Charts**: Dynamic charts are generated based on your data columns
4. **Upload to Supabase**: Click the button to store cleaned data in your database
5. **Download Processed Data**: Get a cleaned version of your CSV

## Supported Data Columns

The system automatically detects and works with these common real estate columns:

- **Property Information**: Property Name, Address, City, State, Zip
- **Property Details**: Type, Property Status, SqFt, Lot Size, Units
- **Financial Data**: Asking Price, Price/SqFt, Price/Acre, NOI, Cap Rate
- **Market Data**: Days on Market, Opportunity Zone
- **Location Data**: Latitude, Longitude

## Data Mapping to Supabase

Current mapping to the `deals` table:
- `Property Name` → `asset_name`
- `Address` → `full_address`
- `Units` → `total_units`
- `SqFt` → `net_rentable_area_sqft`

*Note: Additional mappings can be added as your Supabase schema expands*

## Data Validation

The system validates:
- ✅ Price ranges (no negative prices, reasonable maximums)
- ✅ Square footage ranges (no negative values, reasonable maximums)
- ✅ Cap rate ranges (0-50% typically)
- ✅ Coordinate validation (valid latitude/longitude)

## Chart Types Generated

Based on available data, the system generates:
- 📊 Property Type Distribution
- 💰 Asking Price Distribution
- 🗺️ Geographic Property Map
- 📈 Price per SqFt by City
- 📉 Cap Rate Distribution

## File Structure

```
crexi-dashboard/
├── app.py                    # Original static dashboard
├── dynamic_dashboard.py      # New dynamic dashboard
├── config_example.py         # Supabase configuration template
├── requirements.txt          # Updated dependencies
├── Export Results.csv        # Sample data
└── README_DYNAMIC.md         # This file
```

## Next Steps

This system provides the foundation for:
- 📄 PDF data extraction
- 📊 Excel model population
- 🔄 Automated data pipelines
- 📱 Mobile-friendly interface

## Troubleshooting

**Supabase Connection Issues:**
- Verify your URL and key are correct
- Check that your Supabase project is active
- Ensure the `deals` table exists in your schema

**CSV Upload Issues:**
- Ensure your CSV has headers
- Check for special characters in data
- Try different encoding (UTF-8, Latin-1)

**Chart Generation Issues:**
- Verify you have numeric data for price/area columns
- Check that location data has valid coordinates
- Ensure property type data is consistent

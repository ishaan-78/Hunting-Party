# 🚀 Deploy to Streamlit Community Cloud

## Quick Setup (2 minutes)

### 1. Configure Supabase (Optional but Recommended)
1. Copy `config_example.py` to `config.py`
2. Fill in your Supabase URL and anon key from your Supabase dashboard
3. The app will automatically upload CSV data to Supabase when configured

### 2. Push to GitHub
```bash
git add .
git commit -m "Ready for deployment"
git push origin main
```

### 3. Deploy on Streamlit Cloud
1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Sign in with GitHub
3. Click "New app"
4. Select repository: `Hunting-Party`
5. Select branch: `main`
6. Main file: `streamlit_app.py`
7. Click "Deploy!"

### 4. Done! 🎉
Your app will be live at: `https://your-app-name.streamlit.app`

## 🔄 Auto-Updates
Every time you push changes to GitHub, your app automatically updates!

## 📁 Files Ready for Deployment
✅ `streamlit_app.py` - Main entry point  
✅ `app.py` - Dashboard code  
✅ `requirements.txt` - Dependencies  
✅ `.streamlit/config.toml` - Configuration  
✅ `config_example.py` - Supabase configuration template

## 🎯 Key Features
✅ **Dynamic CSV Upload** - Upload any CSV file for instant analysis  
✅ **Interactive Charts** - Multiple chart types with filtering  
✅ **Supabase Integration** - Automatic data upload to your database  
✅ **Data Validation** - Built-in validation and error checking  
✅ **Export Functionality** - Download filtered data as CSV

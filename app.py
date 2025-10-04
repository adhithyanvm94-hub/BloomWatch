import streamlit as st
import folium
import geopandas as gpd
from streamlit_folium import st_folium
import ee
import pandas as pd
import numpy as np
import datetime
import requests

# ---------------- Initialize Earth Engine ----------------
ee.Initialize(project='bloomwatch-474012')

# ---------------- Streamlit Config ----------------
st.set_page_config(layout="wide", page_title="BloomWatch India")
st.title("🌱 BloomWatch India — Real-Time Plant Growth & Bloom Monitoring")

# ---------------- Load India States GeoJSON ----------------
states_gdf = gpd.read_file("india_states.geojson")

# ---------------- Sidebar ----------------
with st.sidebar:
    st.header("Controls")
    state_list = states_gdf["name"].unique().tolist()
    state = st.selectbox("Select State", sorted(state_list))
    climate_zone = st.selectbox("Select Climate Zone", ["Arid", "Semi-Arid", "Tropical", "Temperate", "Humid", "Mountain"])
    
    category = st.selectbox("Select Category", ["Fruits", "Flowers", "Crops"])
    species_dict = {
        "Fruits": ["Mango", "Litchi", "Guava", "Banana", "Coconut"],
        "Flowers": ["Hibiscus", "Rose", "Jasmine"],
        "Crops": ["Wheat", "Rice", "Maize", "Sugarcane"]
    }
    species = st.selectbox("Select Species", species_dict[category])
    
    start_date = st.date_input("Start Date", value=datetime.date(2023,1,1))
    end_date = st.date_input("End Date", value=datetime.date.today())
    compute_button = st.button("Run Analysis")

# ---------------- Map ----------------
sel_state = states_gdf[states_gdf["name"] == state].to_crs(epsg=4326)
m = folium.Map(location=[22.0, 79.0], zoom_start=4)

# Add all states
folium.GeoJson(
    states_gdf.to_json(),
    name="States",
    style_function=lambda x: {"fillColor": "lightgray", "color": "black", "weight":1, "fillOpacity":0.1},
    tooltip=folium.GeoJsonTooltip(fields=["name"], aliases=["State:"])
).add_to(m)

# Highlight selected state
folium.GeoJson(
    sel_state.to_json(),
    name="Selected State",
    style_function=lambda x: {"fillColor": "yellow", "color": "red", "weight":2, "fillOpacity":0.3},
    tooltip=folium.GeoJsonTooltip(fields=["name"], aliases=["Selected:"])
).add_to(m)

st_folium(m, width=900, height=500)

# ---------------- MODIS NDVI Analysis ----------------
def get_modis_ndvi(region, start, end):
    start_str = start.strftime("%Y-%m-%d")
    end_str = end.strftime("%Y-%m-%d")
    
    col = ee.ImageCollection("MODIS/006/MOD13Q1").select("NDVI") \
            .filterBounds(region).filterDate(start_str, end_str)
    
    def extract(img):
        date = ee.Date(img.get("system:time_start")).format("YYYY-MM-dd")
        mean = img.reduceRegion(ee.Reducer.mean(), region, 5000)
        return ee.Feature(None, {"date": date, "NDVI": mean.get("NDVI")})
    
    feats = col.map(extract).filter(ee.Filter.notNull(["NDVI"])).getInfo()
    
    records = [(f["properties"]["date"], float(f["properties"]["NDVI"])*0.0001)
               for f in feats["features"] if f["properties"]["NDVI"] is not None]
    
    df = pd.DataFrame(records, columns=["date","NDVI"])
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date")

# ---------------- NASA POWER Climate Data ----------------
def get_climate(state_name, start, end):
    geom = sel_state.iloc[0].geometry.centroid
    lat, lon = geom.y, geom.x
    url = (
        f"https://power.larc.nasa.gov/api/temporal/daily/point"
        f"?parameters=T2M,PRECTOT&community=AG&longitude={lon}&latitude={lat}"
        f"&start={int(start.strftime('%Y%m%d'))}&end={int(end.strftime('%Y%m%d'))}&format=JSON"
    )
    r = requests.get(url).json()
    
    records = []
    params = r.get('properties', {}).get('parameter', {})
    t2m_dict = params.get('T2M', {})
    pre_dict = params.get('PRECTOT', {})
    
    for date in t2m_dict.keys():
        t2m = t2m_dict.get(date, np.nan)
        pre = pre_dict.get(date, np.nan)
        records.append((pd.to_datetime(date), t2m, pre))
        
    df = pd.DataFrame(records, columns=["date","Temperature","Precipitation"])
    return df

# ---------------- Bloom Thresholds ----------------
species_thresholds = {
    "Mango": 0.35, "Litchi": 0.4, "Guava": 0.3, "Banana": 0.25, "Coconut": 0.28,
    "Hibiscus": 0.3, "Rose": 0.32, "Jasmine": 0.3,
    "Wheat": 0.25, "Rice": 0.3, "Maize": 0.28, "Sugarcane": 0.27
}

# ---------------- Climate Zone Adjustments ----------------
climate_adjustments = {
    "Arid": 0.05,
    "Semi-Arid": 0.1,
    "Tropical": 0.25,
    "Temperate": 0.2,
    "Humid": 0.3,
    "Mountain": 0.1
}

# ---------------- Bloom Prediction ----------------
def predict_bloom(ndvi_df, climate_df, species, climate_zone):
    if ndvi_df.empty or climate_df.empty:
        return 0.0, pd.DataFrame()
    
    threshold = species_thresholds.get(species, 0.3)
    adjustment = climate_adjustments.get(climate_zone, 0)
    daily_probs = []
    combined_df = ndvi_df.merge(climate_df, on="date", how="left")
    
    for idx, row in combined_df.iterrows():
        ndvi_val = row["NDVI"]
        temp = row["Temperature"]
        pre = row["Precipitation"]
        
        # Base probability from NDVI
        prob = max(0, (ndvi_val - threshold)/(1-threshold))
        
        # Adjust for climate conditions
        if 20 <= temp <= 35 and pre >= 5:
            prob += 0.3
        
        # Adjust for selected climate zone
        prob += adjustment
        prob = min(1.0, prob)
        
        daily_probs.append(prob)
    
    combined_df["Bloom_Probability"] = daily_probs
    final_prob = np.mean(daily_probs)
    return final_prob, combined_df

# ---------------- Run Analysis ----------------
if compute_button:
    st.subheader(f"🌿 Analysis for {state} — {species} ({climate_zone})")
    coords = sel_state.iloc[0].geometry.__geo_interface__
    region = ee.Geometry(coords)
    
    ndvi_df = get_modis_ndvi(region, start_date, end_date)
    climate_df = get_climate(state, start_date, end_date)
    
    prob, combined_df = predict_bloom(ndvi_df, climate_df, species, climate_zone)
    st.metric("🌱 Bloom Probability", f"{prob*100:.1f}%")
    
    if not combined_df.empty:
        st.subheader("NDVI Time Series")
        st.line_chart(combined_df.set_index("date")[["NDVI"]])
        
        st.subheader("Climate Data (Temperature & Precipitation)")
        st.line_chart(combined_df.set_index("date")[["Temperature","Precipitation"]])
        
        st.subheader("Daily Bloom Probability")
        st.line_chart(combined_df.set_index("date")[["Bloom_Probability"]])

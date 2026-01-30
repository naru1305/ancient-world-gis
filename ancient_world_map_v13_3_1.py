import streamlit as st
import pandas as pd
import sqlite3
import folium
from streamlit_folium import st_folium
import uuid
import numpy as np
import os
import gc  # <--- NEW: Garbage Collector for memory management

# --- 0. AUTOMATIC DATABASE STITCHING ---
DB_FILE = 'AncientWorld_Locations.db'

if not os.path.exists(DB_FILE):
    parts = sorted([f for f in os.listdir('.') if f.startswith('db_part_')])
    if parts:
        with st.spinner(f"Reassembling database from {len(parts)} parts..."):
            with open(DB_FILE, 'wb') as outfile:
                for part in parts:
                    with open(part, 'rb') as infile:
                        outfile.write(infile.read())
    else:
        st.error("❌ Database not found! Please ensure 'AncientWorld_Locations.db' or 'db_part_xxx' files are in the repository.")
        st.stop()

# --- CONFIGURATION ---
PAGE_TITLE = "Ancient World Map (Lite)"
MIN_YEAR_LIMIT = -5000
MAX_YEAR_LIMIT = 640
LABEL_ZOOM_THRESHOLD = 7
MAX_LABELS = 500
# NEW: Hard limit on rendered points to prevent RAM crash
MAX_RENDER_POINTS = 22000 

# --- MAP STYLES ---
STYLE_OPTIONS = {
    "Clean Light": "CartoDB positron",
    "Clean Dark": "CartoDB dark_matter",
    "ESRI World Imagery": "esri_world_imagery",
    "ESRI Ocean Base": "esri_ocean"
}

# --- LANGUAGE MAPPING ---
LANGUAGE_MAP = {
    'grc': 'Ancient Greek', 'grc-Latn': 'Ancient Greek (Latinized)', 'la': 'Latin', 'lat': 'Latin',
    'akk': 'Akkadian', 'sux': 'Sumerian', 'hit': 'Hittite', 'egy': 'Ancient Egyptian', 'cop': 'Coptic',
    'arc': 'Aramaic', 'hbo': 'Biblical Hebrew', 'he': 'Hebrew', 'phn': 'Phoenician', 'peo': 'Old Persian',
    'ett': 'Etruscan', 'en': 'English', 'de': 'German', 'fr': 'French', 'it': 'Italian', 'es': 'Spanish',
    'tr': 'Turkish', 'el': 'Modern Greek', 'ar': 'Arabic', 'fa': 'Persian (Farsi)', 'ru': 'Russian',
    'x-unknown': 'Unknown', 'und': 'Undefined', None: 'Unknown'
}

st.set_page_config(layout="wide", page_title=PAGE_TITLE)

# --- CUSTOM CSS ---
st.markdown("""
<style>
    .ancient-label {
        background-color: rgba(255, 255, 255, 0.7) !important;
        border: 1px solid #ccc !important;
        box-shadow: 1px 1px 2px rgba(0,0,0,0.2) !important;
        font-weight: bold;
        font-size: 11px;
        color: #000;
        padding: 2px 5px !important;
        border-radius: 4px;
    }
    .ancient-label-dark {
        background-color: rgba(0, 0, 0, 0.6) !important;
        border: 1px solid #555 !important;
        box-shadow: 1px 1px 2px rgba(0,0,0,0.5) !important;
        font-weight: bold;
        font-size: 11px;
        color: #fff;
        padding: 2px 5px !important;
        border-radius: 4px;
    }
</style>
""", unsafe_allow_html=True)

# --- STATE MANAGEMENT ---
if 'map_center' not in st.session_state:
    st.session_state['map_center'] = [41.9, 12.5]
if 'map_zoom' not in st.session_state:
    st.session_state['map_zoom'] = 4
if 'render_center' not in st.session_state:
    st.session_state['render_center'] = [41.9, 12.5]
if 'render_zoom' not in st.session_state:
    st.session_state['render_zoom'] = 4
if 'selected_id' not in st.session_state:
    st.session_state['selected_id'] = None
if 'map_key' not in st.session_state:
    st.session_state['map_key'] = str(uuid.uuid4())

# --- DATABASE ---
@st.cache_resource
def get_db_connection():
    return sqlite3.connect(DB_FILE, check_same_thread=False)

@st.cache_data
def load_base_data():
    conn = get_db_connection()
    df_geo = pd.read_sql_query("SELECT place_id, region FROM places_regions", conn)
    places_with_regions = set(df_geo['place_id'])
    
    # NEW: Select only needed columns to save RAM
    df_places = pd.read_sql_query(f"""
        SELECT p.id, p.title, p.representative_latitude as lat, p.representative_longitude as lon,
               pte.final_start, pte.final_end
        FROM places p
        JOIN places_temporal_extent pte ON p.id = pte.place_id
        WHERE p.representative_latitude IS NOT NULL
        AND pte.final_start <= {MAX_YEAR_LIMIT} AND pte.final_end >= {MIN_YEAR_LIMIT}
    """, conn)
    
    # NEW: Memory Optimization (Downcasting)
    df_places['id'] = pd.to_numeric(df_places['id'], errors='coerce').fillna(0).astype('int32')
    df_places['lat'] = pd.to_numeric(df_places['lat'], errors='coerce').astype('float32')
    df_places['lon'] = pd.to_numeric(df_places['lon'], errors='coerce').astype('float32')
    df_places['final_start'] = pd.to_numeric(df_places['final_start'], errors='coerce').fillna(0).astype('int16')
    df_places['final_end'] = pd.to_numeric(df_places['final_end'], errors='coerce').fillna(0).astype('int16')

    df_places = df_places.dropna(subset=['lat', 'lon'])
    df_places = df_places[df_places['id'].isin(places_with_regions)].reset_index(drop=True)
    df_places['search_label'] = df_places['title'] + " (" + df_places['id'].astype(str) + ")"
    
    df_epochs = pd.read_sql_query("SELECT pem.place_id, ed.system_era, ed.system_period, ed.system_subperiod FROM places_epoch_mapping pem JOIN epoch_definitions ed ON pem.epoch_id = ed.id", conn)
    df_original = pd.read_sql_query('SELECT DISTINCT pa.place_id, tp.term FROM pleiades_attestations pa JOIN time_periods tp ON pa.time_period_key = tp."key"', conn)
    df_types = pd.read_sql_query("SELECT place_id, place_type FROM places_place_types", conn)
    
    return df_places, df_geo, df_epochs, df_original, df_types

@st.cache_data
def get_geojson_data(df):
    features = []
    # Use standard tuple iteration for speed/memory
    for row in df.itertuples():
        features.append({
            "type": "Feature",
            "geometry": {
                "type": "Point", "coordinates": [row.lon, row.lat],
            },
            "properties": {
                "id": row.id,
                "title": row.title
            }
        })
    return {"type": "FeatureCollection", "features": features}

def get_place_details(place_id):
    conn = get_db_connection()
    pid = int(place_id)
    place = pd.read_sql_query("SELECT p.*, pte.final_start, pte.final_end FROM places p LEFT JOIN places_temporal_extent pte ON p.id = pte.place_id WHERE p.id = ?", conn, params=(pid,))
    if place.empty: return None, None, None
    place = place.iloc[0]
    names = pd.read_sql_query("SELECT attested_form, language_tag FROM names WHERE place_id = ?", conn, params=(pid,))
    if not names.empty:
        names['language_tag'] = names['language_tag'].map(LANGUAGE_MAP).fillna(names['language_tag'])
        names = names.rename(columns={'attested_form': 'Attested Forms', 'language_tag': 'Languages'})
    types = pd.read_sql_query("SELECT place_type FROM places_place_types WHERE place_id = ?", conn, params=(pid,))
    return place, names, types

def get_labels_near_center(df, center_lat, center_lon, zoom, max_labels=500):
    if df.empty: return df.head(0)
    visible_radius = 360 / (2 ** (zoom - 1))
    df = df.copy()
    df['_dist'] = np.sqrt((df['lat'] - center_lat)**2 + (df['lon'] - center_lon)**2)
    return df[df['_dist'] <= visible_radius].sort_values('_dist').head(max_labels)

def sync_view_state():
    """Syncs the render anchor to the tracker."""
    st.session_state['render_center'] = st.session_state['map_center']
    st.session_state['render_zoom'] = st.session_state['map_zoom']
    # Explicitly clear garbage to free memory
    gc.collect()

# --- LOAD DATA ---
df_places, df_geo, df_epochs, df_original, df_types = load_base_data()

# --- SIDEBAR ---
st.sidebar.title("🌍 Ancient World")
st.sidebar.caption(f"📍 {len(df_places)} places available")
tab_filters, tab_layers = st.sidebar.tabs(["🔍 Filters", "🗺️ Layers"])

with tab_filters:
    def handle_search():
        val = st.session_state.get("search_box")
        if val:
            match = df_places[df_places['search_label'] == val]
            if not match.empty:
                row = match.iloc[0]
                st.session_state['selected_id'] = int(row['id'])
                st.session_state['render_center'] = [float(row['lat']), float(row['lon'])]
                st.session_state['render_zoom'] = 10
                st.session_state['map_center'] = [float(row['lat']), float(row['lon'])]
                st.session_state['map_zoom'] = 10
                st.session_state['map_key'] = str(uuid.uuid4())

    st.selectbox("Find a place:", options=df_places['search_label'].tolist(), index=None, key="search_box", on_change=handle_search, placeholder="Type to search...")
    st.markdown("---")
    
    if 'last_time_range' not in st.session_state: st.session_state['last_time_range'] = (-5000, -4000)
    time_range = st.slider("Time Period", MIN_YEAR_LIMIT, MAX_YEAR_LIMIT, st.session_state['last_time_range'], 50, on_change=sync_view_state)
    st.session_state['last_time_range'] = time_range

    st.markdown("---")
    def get_opt(df, col): return sorted([x for x in df[col].unique() if x and isinstance(x, str)])
    
    s_type = st.multiselect("Place Type", get_opt(df_types, 'place_type'), on_change=sync_view_state)
    s_reg = st.multiselect("Region", get_opt(df_geo, 'region'), on_change=sync_view_state)
    s_era = st.multiselect("System Era", get_opt(df_epochs, 'system_era'), on_change=sync_view_state)
    s_per = st.multiselect("System Period", get_opt(df_epochs, 'system_period'), on_change=sync_view_state)
    s_sub = st.multiselect("System Subperiod", get_opt(df_epochs, 'system_subperiod'), on_change=sync_view_state)
    s_orig = st.multiselect("Original Pleiades Terms", get_opt(df_original, 'term'), on_change=sync_view_state)

    # Apply Filters
    mask_time = (df_places['final_start'] <= time_range[1]) & (df_places['final_end'] >= time_range[0])
    valid_ids = set(df_places.loc[mask_time, 'id'])
    def intersect(c_ids, df, col, vals): return c_ids.intersection(set(df[df[col].isin(vals)]['place_id'])) if vals else c_ids
    valid_ids = intersect(valid_ids, df_types, 'place_type', s_type)
    valid_ids = intersect(valid_ids, df_geo, 'region', s_reg)
    valid_ids = intersect(valid_ids, df_epochs, 'system_era', s_era)
    valid_ids = intersect(valid_ids, df_epochs, 'system_period', s_per)
    valid_ids = intersect(valid_ids, df_epochs, 'system_subperiod', s_sub)
    if s_orig: valid_ids = valid_ids.intersection(set(df_original[df_original['term'].isin(s_orig)]['place_id']))

    df_view = df_places[df_places['id'].isin(valid_ids)].reset_index(drop=True)
    
    # --- MEMORY SAFETY VALVE ---
    # If too many points, truncate them to avoid Free Tier crash
    total_results = len(df_view)
    if total_results > MAX_RENDER_POINTS:
        st.warning(f"⚠️ {total_results} places found. Showing top {MAX_RENDER_POINTS} to save memory. Filter to see specific results.")
        df_view = df_view.head(MAX_RENDER_POINTS)
    else:
        st.caption(f"📍 {total_results} Places shown")
    
    st.markdown("### 📥 Export")
    st.download_button("Download CSV", df_view.to_csv(index=False).encode('utf-8'), "ancient_places.csv", "text/csv")
    
    with st.expander("📋 SQL Query"):
        def to_sql_list(items): return ", ".join([f"'{x}'" for x in items])
        where_clauses = [f"id IN (SELECT place_id FROM places_temporal_extent WHERE final_start <= {time_range[1]} AND final_end >= {time_range[0]})"]
        where_clauses.append("id IN (SELECT DISTINCT place_id FROM places_regions)")
        if s_type: where_clauses.append(f"id IN (SELECT place_id FROM places_place_types WHERE place_type IN ({to_sql_list(s_type)}))")
        if s_reg: where_clauses.append(f"id IN (SELECT place_id FROM places_regions WHERE region IN ({to_sql_list(s_reg)}))")
        if s_era: where_clauses.append(f"id IN (SELECT pem.place_id FROM places_epoch_mapping pem JOIN epoch_definitions ed ON pem.epoch_id = ed.id WHERE ed.system_era IN ({to_sql_list(s_era)}))")
        if s_per: where_clauses.append(f"id IN (SELECT pem.place_id FROM places_epoch_mapping pem JOIN epoch_definitions ed ON pem.epoch_id = ed.id WHERE ed.system_period IN ({to_sql_list(s_per)}))")
        if s_sub: where_clauses.append(f"id IN (SELECT pem.place_id FROM places_epoch_mapping pem JOIN epoch_definitions ed ON pem.epoch_id = ed.id WHERE ed.system_subperiod IN ({to_sql_list(s_sub)}))")
        if s_orig: where_clauses.append(f"id IN (SELECT pa.place_id FROM pleiades_attestations pa JOIN time_periods tp ON pa.time_period_key = tp.\"key\" WHERE tp.term IN ({to_sql_list(s_orig)}))")
        st.code("SELECT * FROM places WHERE \n" + " AND \n".join(where_clauses), language="sql")

with tab_layers:
    st.subheader("Base Map")
    selected_style_name = st.selectbox("Map Style", list(STYLE_OPTIONS.keys()), index=0, on_change=sync_view_state)
    
    st.subheader("Features")
    show_labels = st.toggle("🏷️ Show Labels", value=False, on_change=sync_view_state)
    if show_labels:
        if st.session_state['render_zoom'] < LABEL_ZOOM_THRESHOLD:
            st.caption(f"ℹ️ Zoom in more to see labels (Level {LABEL_ZOOM_THRESHOLD}+)")
        else:
            st.caption(f"✓ Labels active (Max {MAX_LABELS})")
            if st.button("🔄 Refresh Labels (Current View)"):
                sync_view_state()
                st.rerun()
    
    st.toggle("🌊 Major Rivers", value=False, disabled=True)
    st.toggle("🏰 Ancient Borders", value=False, disabled=True)

# === MAIN CONTENT ===
st.title(PAGE_TITLE)

selected_tiles = STYLE_OPTIONS[selected_style_name]
attr, tiles_url = ("Esri", None), None
if "esri" in selected_tiles:
    attr = "Esri"
    if selected_tiles == "esri_world_imagery":
        tiles_url = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
    elif selected_tiles == "esri_ocean":
        tiles_url = "https://server.arcgisonline.com/ArcGIS/rest/services/Ocean/World_Ocean_Base/MapServer/tile/{z}/{y}/{x}"
    selected_tiles = None

is_dark_map = selected_style_name in ["Clean Dark", "ESRI World Imagery", "ESRI Ocean Base"]
label_color = "white" if is_dark_map else "#333333"
label_shadow = "black" if is_dark_map else "white"

# 1. Initialize Map
m = folium.Map(
    location=st.session_state['render_center'],
    zoom_start=st.session_state['render_zoom'],
    tiles=selected_tiles if not tiles_url else None,
    prefer_canvas=True
)
if tiles_url: folium.TileLayer(tiles=tiles_url, attr=attr).add_to(m)

selected_id = st.session_state['selected_id']

# 2. Optimized GeoJSON Layer (Background Points)
df_background = df_view[df_view['id'] != selected_id]

if not df_background.empty:
    geojson_data = get_geojson_data(df_background)
    popup = folium.GeoJsonPopup(fields=["title", "id"], aliases=["Name:", "ID:"], labels=False)
    
    folium.GeoJson(
        geojson_data,
        name="Ancient Sites",
        marker=folium.CircleMarker(radius=6, weight=0, fill_color="#c81e00", fill_opacity=0.7),
        tooltip=folium.GeoJsonTooltip(fields=["title", "id"], aliases=["Name:", "ID:"]),
        popup=popup, 
        zoom_on_click=False
    ).add_to(m)

# 3. Draw Selected Point
if selected_id:
    sel_row = df_places[df_places['id'] == selected_id]
    if not sel_row.empty:
        row = sel_row.iloc[0]
        folium.CircleMarker(
            location=[float(row.lat), float(row.lon)],
            radius=12, color='#000000', weight=3, fill=True, fill_color='#ffdd00', fill_opacity=1.0,
            tooltip=f"<b>{row.title}</b> (Selected)"
        ).add_to(m)

# 4. MARKER LABEL LAYER (CLEAN STYLE)
if show_labels and st.session_state['render_zoom'] >= LABEL_ZOOM_THRESHOLD:
    df_labels = get_labels_near_center(
        df_view, 
        st.session_state['render_center'][0], 
        st.session_state['render_center'][1], 
        st.session_state['render_zoom'], 
        MAX_LABELS
    )
    
    for row in df_labels.itertuples():
        folium.Marker(
            location=[float(row.lat), float(row.lon)],
            icon=folium.DivIcon(
                html=f'''<div style="
                    font-size: 10px; font-weight: 500; color: {label_color};
                    text-shadow: -1px -1px 0 {label_shadow}, 1px -1px 0 {label_shadow}, 
                                 -1px 1px 0 {label_shadow}, 1px 1px 0 {label_shadow};
                    white-space: nowrap; pointer-events: none;
                    position: relative; left: 8px; top: -5px;">{row.title}</div>''',
                icon_size=(0, 0), icon_anchor=(0, 0)
            )
        ).add_to(m)

# === RENDER & INTERACTION ===
output = st_folium(
    m, 
    width=1400, 
    height=650,
    key=st.session_state['map_key'],
    returned_objects=["last_active_drawing", "last_object_clicked", "center", "zoom"] 
)

# === HANDLE MAP INTERACTION ===
if output:
    # 1. Update Tracker
    if output.get("center"):
        st.session_state['map_center'] = [output["center"]["lat"], output["center"]["lng"]]
    if output.get("zoom"):
        st.session_state['map_zoom'] = output["zoom"]

    # 2. Handle Selection
    clicked_data = output.get("last_active_drawing") or output.get("last_object_clicked")

    if clicked_data:
        props = clicked_data.get("properties")
        if props and "id" in props:
            clicked_id = int(props["id"])
            if clicked_id != st.session_state['selected_id']:
                st.session_state['selected_id'] = clicked_id
                sync_view_state()
                st.rerun()

# === CLEAR SELECTION ===
def clear_selection():
    st.session_state['selected_id'] = None

if selected_id is not None:
    st.button("❌ Clear Selection", on_click=clear_selection)

# === DETAILS PANE ===
if selected_id is not None:
    place, names, types = get_place_details(selected_id)
    if place is not None:
        st.markdown("---")
        col1, col2 = st.columns([1, 2])
        with col1:
            st.header(place['title'])
            st.caption(f"ID: {place['id']}")
            if not types.empty: st.write(f"**Type:** {', '.join(types['place_type'].tolist())}")
            st.write(f"**Period:** {int(place['final_start'])} to {int(place['final_end'])}")
            if place.get('description'): st.info(place['description'])
        with col2:
            st.subheader("Attested Names")
            if not names.empty: st.dataframe(names, hide_index=True)
            else: st.write("No recorded names.")

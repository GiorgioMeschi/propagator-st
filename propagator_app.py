

#%%

# propagator_app.py
import streamlit as st
import numpy as np
import tempfile
import rasterio
from rasterio.transform import array_bounds, xy
from io import BytesIO
import matplotlib.pyplot as plt
import folium
from streamlit_folium import st_folium
from matplotlib import cm
from propagator.core import BoundaryConditions, FUEL_SYSTEM_LEGACY, Propagator

st.set_page_config(layout="wide", page_title="Propagator")
st.title("Propagator")

if 'map_center' not in st.session_state:
    st.session_state['map_center'] = None

# persistent session keys
st.session_state.setdefault("overlay_arr", None)
st.session_state.setdefault("overlay_bounds", None)
st.session_state.setdefault("ign_from_map", None)
st.session_state.setdefault("map_zoom", 10)
st.session_state.setdefault("clicked_lat", None)
st.session_state.setdefault("clicked_lng", None)

# helper
def save_open(uploaded):
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".tif")
    tmp.write(uploaded.read()); tmp.flush(); tmp.close()
    return rasterio.open(tmp.name)

# Sidebar: all inputs
with st.sidebar:
    st.header("Inputs & model parameters")
    dem_file = st.file_uploader("DEM", type=["tif", "tiff"])
    veg_file = st.file_uploader("Vegetation", type=["tif", "tiff"])
    st.markdown("---")
    st.write("Clicked coords on map:", st.session_state.clicked_lat, st.session_state.clicked_lng)
    if st.button("Confirm coords as ignition"):
        if st.session_state.clicked_lat is not None:
            st.session_state["ign_from_map"] = (float(st.session_state.clicked_lat), float(st.session_state.clicked_lng))
            st.success(f"Ignition set to: {st.session_state.clicked_lat:.6f}, {st.session_state.clicked_lng:.6f}")
        else:
            st.warning("No clicked point available. Click the map first.")
    wind_speed = st.slider("Wind speed (km/h)", 0.0, 80.0, 10.0, 1.0)
    wind_dir = st.slider("Wind direction (deg clockwise from North)", 0.0, 360.0, 90.0, 1.0)
    rel_humidity = st.slider("Fuel moisture", 0, 100, 0, 1)
    realizations = st.number_input("Realizations", 1, 100, 10, 1)
    max_time_hour = st.number_input("Max sim time (h)", 1, 24, 1, 1) 
    max_time = max_time_hour * 3600 # convert to seconds
    st.markdown("---")
    st.markdown("**Ignition**")
    run = st.button("Run simulation")

# open uploaded rasters (if provided)
src_dem = save_open(dem_file) if dem_file else None
src_veg = save_open(veg_file) if veg_file else None

# center map on dem
if src_dem and st.session_state['map_center'] is None:
    latdem, londem = src_dem.transform * ((src_dem.width // 2), (src_dem.height // 2))
    st.session_state['map_center'] = (londem, latdem)


# Build folium map (do NOT recenter on overlay; reuse last view)
map_center = st.session_state["map_center"]
map_zoom = st.session_state["map_zoom"]
m = folium.Map(location=map_center, zoom_start=map_zoom, tiles="OpenStreetMap") # location=map_center,
folium.TileLayer("OpenStreetMap").add_to(m)

# add text label in the clicked lat lon point
if st.session_state.get("ign_from_map") is not None:
    ign_lat, ign_lon = st.session_state["ign_from_map"]
    folium.Marker(
        location=[ign_lat, ign_lon],
        popup="Ignition point",
        icon=folium.Icon(color="red", icon="fire", prefix="fa"),
    ).add_to(m)

# Add cached overlay if available
if st.session_state["overlay_arr"] is not None and st.session_state["overlay_bounds"] is not None:
    folium.raster_layers.ImageOverlay(
        image=st.session_state["overlay_arr"],
        bounds=st.session_state["overlay_bounds"],
        opacity=1.0,
        name="Fire probability",
        interactive=True,
        cross_origin=False,
        zindex=10,
    ).add_to(m)
    folium.LayerControl().add_to(m)

map_result = st_folium(m, width=1200, height=800) # enlarge map

# capture clicks and map view if available
if isinstance(map_result, dict):
    clicked = map_result.get("last_clicked")
    if clicked:
        st.session_state.clicked_lat = clicked.get("lat")
        st.session_state.clicked_lng = clicked.get("lng")

# Run simulation: compute, clip and cache overlay
if run:
    if src_dem is None or src_veg is None:
        st.error("Both DEM and vegetation rasters are required before running.")
        st.stop()

    dem = src_dem.read(1).astype(np.float32)
    dem_transform = src_dem.transform
    rows, cols = dem.shape
    veg = src_veg.read(1).astype(np.int32)

    # ignition priority
    if st.session_state.get("ign_from_map"):
        ign_lat, ign_lon = st.session_state["ign_from_map"]


    if ign_lat is None or ign_lon is None:
        st.warning("Ignition coordinates not set.")
        st.stop()

    inv = ~dem_transform
    colf, rowf = inv * (ign_lon, ign_lat)
    ign_col, ign_row = int(colf), int(rowf)
    if not (0 <= ign_row < rows and 0 <= ign_col < cols):
        st.error("Ignition outside DEM extent.")
        st.stop()

    st.info(f"Ignition pixel row,col = {ign_row},{ign_col}")

    wind_speed_arr = np.full_like(dem, float(wind_speed), dtype=np.float32)
    wind_dir_arr = np.full_like(dem, float(wind_dir), dtype=np.float32)
    moisture = np.full_like(dem, float(rel_humidity), dtype=np.float32)

    sim = Propagator(dem=dem, 
                     veg=veg, 
                     realizations=int(realizations), 
                     fuels=FUEL_SYSTEM_LEGACY,
                     do_spotting=False, 
                     out_of_bounds_mode="raise")
    
    sim.set_boundary_conditions(BoundaryConditions(time=0, 
                                                   ignitions=[(ign_row, ign_col)],
                                                   wind_speed=wind_speed_arr, 
                                                   wind_dir=wind_dir_arr, 
                                                   moisture=moisture))

    with st.spinner("Running simulation..."):
        while (sim.next_time() is not None) and sim.time <= float(max_time):
            sim.step()

    fire_prob = sim.compute_fire_probability().astype(np.float32)

    # write GeoTIFF (optional)
    out_tif = tempfile.NamedTemporaryFile(delete=False, suffix=".tif").name
    # put out tif in sessions state
    st.session_state["out_tif"] = out_tif
    meta = {"driver": "GTiff", "height": fire_prob.shape[0], "width": fire_prob.shape[1],
            "count": 1, "dtype": "float32", "transform": dem_transform}
    if src_dem.crs:
        meta["crs"] = src_dem.crs
    with rasterio.open(out_tif, "w", **meta) as dst:
        dst.write(fire_prob, 1)

    # discrete palette (0–25% green, 25–50% yellow, 50–75% red, 75–100% magenta), full opacity
    arr = fire_prob
    nonzero = arr[arr > 0]
    vmax = float(np.percentile(nonzero, 99)) if nonzero.size else float(arr.max() or 1.0)
    norm = np.clip(arr / float(vmax), 0.0, 1.0)

    h, w = arr.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    c1 = np.array([0, 255, 0], dtype=np.uint8)
    c2 = np.array([255, 255, 0], dtype=np.uint8)
    c3 = np.array([255, 0, 0], dtype=np.uint8)
    c4 = np.array([180, 0, 180], dtype=np.uint8)

    mask_nonzero = arr > 0
    bins = np.searchsorted([0.25, 0.5, 0.75], norm, side="right")
    rgba[..., 3] = 0
    rgba[mask_nonzero, 3] = 255
    rgba[mask_nonzero & (bins == 0), :3] = c1
    rgba[mask_nonzero & (bins == 1), :3] = c2
    rgba[mask_nonzero & (bins == 2), :3] = c3
    rgba[mask_nonzero & (bins == 3), :3] = c4

    # plotting window ±50 px around ignition
    dx = 200
    x0, x1 = max(0, ign_col - dx), min(cols, ign_col + dx)
    y0, y1 = max(0, ign_row - dx), min(rows, ign_row + dx)
    if x1 <= x0 or y1 <= y0:
        x0, x1, y0, y1 = 0, cols, 0, rows

    rgba_window = rgba[y0:y1, x0:x1, :]
    rgba_flipped = rgba_window

    # compute bounds for the window (UL and LR)
    ulx, uly = xy(dem_transform, y0, x0, offset="ul")
    lrx, lry = xy(dem_transform, y1 - 1, x1 - 1, offset="lr")
    bounds = [[lry, ulx], [uly, lrx]]  # [[south, west],[north, east]]

    st.session_state["overlay_arr"] = rgba_flipped
    st.session_state["overlay_bounds"] = bounds    
    
    
try:
    with open(st.session_state.get('out_tif'), "rb") as f:
        st.sidebar.download_button("Download Fire Probability", f.read(), file_name="fire_probability.tif", mime="image/tiff")
    st.sidebar.success("Simulation finished. Zoom in or out to refresh the map.")

except Exception:
    pass

# put now map center as current lat lon clicked
if st.sidebar.button("Go to fire on map"):
    if st.session_state.get("ign_from_map") is not None:
        lat, lon = st.session_state["ign_from_map"]
        st.session_state["map_center"] = (lat, lon)
        st.session_state["map_zoom"] = 14
    else:
        st.sidebar.warning("No ignition point set.")





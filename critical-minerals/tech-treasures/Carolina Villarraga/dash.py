import streamlit as st
import rasterio
import numpy as np
import pandas as pd
import geopandas as gpd
import plotly.express as px
from matplotlib import pyplot as plt
import folium
from streamlit_folium import st_folium
import matplotlib.cm as cm
import joblib
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
from matplotlib import colors
import plotly.graph_objects as go
gpd.options.use_pygeos = False

# Setting up the page configuration
st.set_page_config(page_title="Mineral Exploration Dashboard", page_icon=":pick:", layout="wide")
#Load Model
rf_model=joblib.load(r'models\random_forest_model.pkl')
# Load the layers
@st.cache_data
def load_geodata(file_path):
    return gpd.read_file(file_path)

geology = load_geodata(r'datasets/geojson/bc_rocks.geojson')
faults = load_geodata(r'datasets\geojson\BC_faults.geojson')
critical_minerals = load_geodata(r'datasets\geojson\critical_minerals.geojson')
bc_cm=load_geodata(r'datasets\geojson\cmBC.geojson')

BC=critical_minerals[(critical_minerals['ProvincesE']=='British Columbia')]

# Title and subtitle
st.title(":pick: Mineral Exploration Data Analysis")
st.markdown("This dashboard provides an initial visualization of critical mineral prospectivity across British Columbia, based on available geospatial data, including gravity and magnetic features. It represents a preliminary mockup, created with limited resources and is not intended for final decision-making. Further refinement and validation are needed to enhance its accuracy for exploration planning.")

st.markdown('<style>div.block-container{padding-top:1rem;}</style>', unsafe_allow_html=True)

# Section: Filter Options in Sidebar
st.sidebar.header("Filter Options")

mineral_type = st.sidebar.multiselect("Select Mineral Type", BC['Commoditie'].unique())
if mineral_type:
    df_filtered = BC[BC['Commoditie'].isin(mineral_type)]
else:
    df_filtered = critical_minerals

# Section 1: Column Layout for Data Overview
col1, col2 = st.columns(2)

with col1:
    st.subheader("Mineral Deposit Summary")
    #summary = df_filtered.groupby('Developmen')[''].sum()
    #st.write(summary)

with col2:
    st.subheader("Location Overview")
    #fig = px.scatter(df_filtered, x='Longitude', y='Latitude', color='Mineral Type', 
                     #title="Deposits by Location")
    #st.plotly_chart(fig, use_container_width=True)
    m = geology.explore('rock_class')
    st_folium(m)

## Section 2 Prospectivity map
st.subheader("Predicted Prospectivity Map")
with rasterio.open('predicted_prospectivity_map_discrete.tif') as src:
    predicted_map = src.read(1)

colors = ['white', 'gray', 'skyblue']  # Define colors for the classes
cmap = ListedColormap(colors)

# Plot the raster using the discrete colormap
plt.figure(figsize=(14, 12))
plt.imshow(predicted_map, cmap=cmap, vmin=0, vmax=2)

cbar = plt.colorbar(ticks=[0, 1, 2])
cbar.ax.set_yticklabels(['Non-Mineralized', 'Low Confidence', 'High Confidence'])
cbar.set_label('Predicted Classes')

#plt.colorbar(ticks=[0, 1, 2], label='Predicted Classes')  # Add colorbar with class labels
plt.title('Predicted Prospectivity Map')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
st.pyplot(plt)
# Section 3: Additional Data and Visualizations
col3, col4 = st.columns(2)

with col3:
    st.subheader("Model Insights")
    importances = rf_model.feature_importances_
    features = ['GRAV_1stVertical', 'GRAV_Horizontal', 'GRAV_Isostatic', 'MAG', 'MAG1DV']

    df = pd.DataFrame({'Feature': features, 'Importance': importances})
    fig = go.Figure([go.Bar(x=df['Feature'], y=df['Importance'], marker_color='skyblue')])
    st.plotly_chart(fig)
    st.markdown("The features above were used in the model, and their importance is based on their contribution to classifying mineralized zones.")

with col4:
  
# KPI Calculations
    
    mineralized_high_conf = np.sum(predicted_map == 2)
    mineralized_low_conf = np.sum(predicted_map == 1)
    non_mineralized = np.sum(predicted_map == 0)

    total_pixels = predicted_map.size
    high_conf_percentage = (mineralized_high_conf / total_pixels) * 100
    low_conf_percentage = (mineralized_low_conf / total_pixels) * 100

    st.metric(label="High Confidence Mineralized Area", value=f"{mineralized_high_conf} km²", delta=f"{high_conf_percentage:.2f}% of total")
    st.metric(label="Low Confidence Mineralized Area", value=f"{mineralized_low_conf} km²", delta=f"{low_conf_percentage:.2f}% of total")
    st.metric(label="Non-Mineralized Area", value=f"{non_mineralized} km²")

    # Example histogram (replace with actual columns)
    #fig4 = px.histogram(df_filtered, x='Deposit Size', title="Deposit Size Distribution")
    #st.plotly_chart(fig4, use_container_width=True)

st.subheader("Future Improvements")

st.markdown("""
- **Model Tuning**: Further hyperparameter tuning to improve classification accuracy.
- **Incorporate More Data**: Adding geochemical data and other geological layers could refine predictions.
- **Synthetic Data**: Potential to generate synthetic non-mineralized samples to improve model balance.
""")


